import Mathlib

namespace NUMINAMATH_CALUDE_operation_terminates_l3516_351657

/-- A sequence of positive integers -/
def Sequence := List Nat

/-- Represents the operation of replacing adjacent numbers -/
inductive Operation
  | replaceLeft (x y : Nat) : Operation  -- Replaces (x, y) with (y+1, x)
  | replaceRight (x y : Nat) : Operation -- Replaces (x, y) with (x-1, x)

/-- Applies an operation to a sequence -/
def applyOperation (s : Sequence) (op : Operation) : Sequence :=
  match s, op with
  | x::y::rest, Operation.replaceLeft x' y' => if x > y then (y+1)::x::rest else s
  | x::y::rest, Operation.replaceRight x' y' => if x > y then (x-1)::x::rest else s
  | _, _ => s

/-- Theorem: The process of applying operations terminates after finite iterations -/
theorem operation_terminates (s : Sequence) : 
  ∃ (n : Nat), ∀ (ops : List Operation), ops.length > n → 
    (ops.foldl applyOperation s = s) := by
  sorry


end NUMINAMATH_CALUDE_operation_terminates_l3516_351657


namespace NUMINAMATH_CALUDE_sin_4phi_value_l3516_351644

theorem sin_4phi_value (φ : ℝ) : 
  Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5 →
  Real.sin (4 * φ) = 12 * Real.sqrt 8 / 625 := by
sorry

end NUMINAMATH_CALUDE_sin_4phi_value_l3516_351644


namespace NUMINAMATH_CALUDE_x_squared_mod_26_l3516_351627

theorem x_squared_mod_26 (x : ℤ) (h1 : 5 * x ≡ 9 [ZMOD 26]) (h2 : 4 * x ≡ 15 [ZMOD 26]) :
  x^2 ≡ 10 [ZMOD 26] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_26_l3516_351627


namespace NUMINAMATH_CALUDE_total_population_theorem_l3516_351600

/-- Represents the population of a school -/
structure SchoolPopulation where
  b : ℕ  -- number of boys
  g : ℕ  -- number of girls
  t : ℕ  -- number of teachers

/-- Checks if the school population satisfies the given conditions -/
def isValidPopulation (p : SchoolPopulation) : Prop :=
  p.b = 4 * p.g ∧ p.g = 2 * p.t

/-- Calculates the total population of the school -/
def totalPopulation (p : SchoolPopulation) : ℕ :=
  p.b + p.g + p.t

/-- Theorem stating that for a valid school population, 
    the total population is equal to 11b/8 -/
theorem total_population_theorem (p : SchoolPopulation) 
  (h : isValidPopulation p) : 
  (totalPopulation p : ℚ) = 11 * (p.b : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_total_population_theorem_l3516_351600


namespace NUMINAMATH_CALUDE_binomial_20_9_l3516_351674

theorem binomial_20_9 (h1 : Nat.choose 18 7 = 31824)
                      (h2 : Nat.choose 18 8 = 43758)
                      (h3 : Nat.choose 18 9 = 43758) :
  Nat.choose 20 9 = 163098 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_9_l3516_351674


namespace NUMINAMATH_CALUDE_orthocenter_coordinates_l3516_351615

/-- The orthocenter of a triangle --/
structure Orthocenter (A B C : ℝ × ℝ) where
  point : ℝ × ℝ
  is_orthocenter : Bool

/-- Definition of triangle ABC --/
def A : ℝ × ℝ := (5, -1)
def B : ℝ × ℝ := (4, -8)
def C : ℝ × ℝ := (-4, -4)

/-- The orthocenter of triangle ABC --/
def triangle_orthocenter : Orthocenter A B C := {
  point := (3, -5),
  is_orthocenter := sorry
}

/-- Theorem: The orthocenter of triangle ABC is (3, -5) --/
theorem orthocenter_coordinates :
  triangle_orthocenter.point = (3, -5) := by sorry

end NUMINAMATH_CALUDE_orthocenter_coordinates_l3516_351615


namespace NUMINAMATH_CALUDE_largest_intersection_point_l3516_351603

/-- Polynomial P(x) -/
def P (a : ℝ) (x : ℝ) : ℝ := x^6 - 13*x^5 + 42*x^4 - 30*x^3 + a*x^2

/-- Line L(x) -/
def L (c : ℝ) (x : ℝ) : ℝ := 3*x + c

/-- The set of intersection points between P and L -/
def intersectionPoints (a c : ℝ) : Set ℝ := {x : ℝ | P a x = L c x}

theorem largest_intersection_point (a c : ℝ) :
  (∃ p q r : ℝ, intersectionPoints a c = {p, q, r} ∧ p < q ∧ q < r) →
  (∀ x : ℝ, x ∉ intersectionPoints a c → P a x < L c x) →
  (∃ x ∈ intersectionPoints a c, ∀ y ∈ intersectionPoints a c, y ≤ x) →
  (∃ x ∈ intersectionPoints a c, x = 4) :=
sorry

end NUMINAMATH_CALUDE_largest_intersection_point_l3516_351603


namespace NUMINAMATH_CALUDE_average_increase_fraction_l3516_351646

-- Define the number of students in the class
def num_students : ℕ := 80

-- Define the correct mark and the wrongly entered mark
def correct_mark : ℕ := 62
def wrong_mark : ℕ := 82

-- Define the increase in total marks due to the error
def mark_difference : ℕ := wrong_mark - correct_mark

-- State the theorem
theorem average_increase_fraction :
  (mark_difference : ℚ) / num_students = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_fraction_l3516_351646


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3516_351609

theorem contrapositive_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3516_351609


namespace NUMINAMATH_CALUDE_division_problem_l3516_351693

theorem division_problem (x y z : ℕ) : 
  x > 0 → 
  x = 7 * y + 3 → 
  2 * x = 3 * y * z + 2 → 
  11 * y - x = 1 → 
  z = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3516_351693


namespace NUMINAMATH_CALUDE_solution_system_equations_l3516_351694

theorem solution_system_equations (x y : ℝ) :
  x ≠ 0 ∧
  |y - x| - |x| / x + 1 = 0 ∧
  |2 * x - y| + |x + y - 1| + |x - y| + y - 1 = 0 →
  y = x ∧ 0 < x ∧ x ≤ 0.5 :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l3516_351694


namespace NUMINAMATH_CALUDE_system_solution_l3516_351648

theorem system_solution (x y z : ℝ) : 
  x + y + z = 2 ∧ 
  x^2 + y^2 + z^2 = 6 ∧ 
  x^3 + y^3 + z^3 = 8 ↔ 
  ((x = 1 ∧ y = 2 ∧ z = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = 2) ∨
   (x = 2 ∧ y = 1 ∧ z = -1) ∨
   (x = 2 ∧ y = -1 ∧ z = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 2) ∨
   (x = -1 ∧ y = 2 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3516_351648


namespace NUMINAMATH_CALUDE_complement_A_union_B_l3516_351696

def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}
def U : Set Int := Set.univ

theorem complement_A_union_B :
  (U \ (A ∪ B)) = {x : Int | ∃ k : Int, x = 3 * k} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l3516_351696


namespace NUMINAMATH_CALUDE_sin_cos_difference_l3516_351623

open Real

theorem sin_cos_difference (α : ℝ) 
  (h : 2 * sin α * cos α = (sin α + cos α)^2 - 1)
  (h1 : (sin α + cos α)^2 - 1 = -24/25) : 
  |sin α - cos α| = 7/5 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l3516_351623


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3516_351604

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3516_351604


namespace NUMINAMATH_CALUDE_sports_club_overlap_l3516_351690

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) : 
  total = 28 → badminton = 17 → tennis = 19 → neither = 2 →
  badminton + tennis - total + neither = 10 := by
sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l3516_351690


namespace NUMINAMATH_CALUDE_f_properties_l3516_351652

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - x + 1

theorem f_properties :
  (∀ a : ℝ, (∀ x > 0, x * (Real.log x + 1/x) ≤ x^2 + a*x + 1) ↔ a ≥ -1) ∧
  (∀ x > 0, (x - 1) * f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3516_351652


namespace NUMINAMATH_CALUDE_brians_math_quiz_l3516_351699

theorem brians_math_quiz (x : ℝ) : (x - 11) / 5 = 31 → (x - 5) / 11 = 15 := by
  sorry

end NUMINAMATH_CALUDE_brians_math_quiz_l3516_351699


namespace NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l3516_351687

theorem odd_divides_power_factorial_minus_one (n : ℕ) (h : n > 0) (hodd : Odd n) :
  n ∣ 2^(n!) - 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l3516_351687


namespace NUMINAMATH_CALUDE_tenth_term_is_one_over_120_l3516_351673

def a (n : ℕ) : ℚ := 1 / (n * (n + 2))

theorem tenth_term_is_one_over_120 : a 10 = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_one_over_120_l3516_351673


namespace NUMINAMATH_CALUDE_rugs_bought_is_twenty_l3516_351658

/-- Calculates the number of rugs bought given buying price, selling price, and total profit -/
def rugs_bought (buying_price selling_price total_profit : ℚ) : ℚ :=
  total_profit / (selling_price - buying_price)

/-- Theorem stating that the number of rugs bought is 20 -/
theorem rugs_bought_is_twenty :
  rugs_bought 40 60 400 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rugs_bought_is_twenty_l3516_351658


namespace NUMINAMATH_CALUDE_f_max_value_l3516_351633

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.sin x) + Real.sin (x - Real.sin x) + (Real.pi / 2 - 2) * Real.sin (Real.sin x)

theorem f_max_value :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (M = (Real.pi - 2) / Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l3516_351633


namespace NUMINAMATH_CALUDE_bill_muffin_batches_l3516_351640

/-- The cost of blueberries in dollars per 6 ounce carton -/
def blueberry_cost : ℚ := 5

/-- The cost of raspberries in dollars per 12 ounce carton -/
def raspberry_cost : ℚ := 3

/-- The amount of fruit in ounces required for each batch of muffins -/
def fruit_per_batch : ℚ := 12

/-- The total savings in dollars by using raspberries instead of blueberries -/
def total_savings : ℚ := 22

/-- The number of batches Bill plans to make -/
def num_batches : ℕ := 3

/-- Theorem stating that given the costs, fruit requirement, and total savings,
    Bill plans to make 3 batches of muffins -/
theorem bill_muffin_batches :
  (blueberry_cost * 2 - raspberry_cost) * (num_batches : ℚ) ≤ total_savings ∧
  (blueberry_cost * 2 - raspberry_cost) * ((num_batches + 1) : ℚ) > total_savings :=
by sorry

end NUMINAMATH_CALUDE_bill_muffin_batches_l3516_351640


namespace NUMINAMATH_CALUDE_music_books_cost_l3516_351668

def total_budget : ℕ := 500
def maths_books : ℕ := 4
def maths_book_price : ℕ := 20
def science_book_price : ℕ := 10
def art_book_price : ℕ := 20

def science_books : ℕ := maths_books + 6
def art_books : ℕ := 2 * maths_books

def maths_cost : ℕ := maths_books * maths_book_price
def science_cost : ℕ := science_books * science_book_price
def art_cost : ℕ := art_books * art_book_price

def total_cost_except_music : ℕ := maths_cost + science_cost + art_cost

theorem music_books_cost (music_cost : ℕ) :
  music_cost = total_budget - total_cost_except_music →
  music_cost = 160 := by
  sorry

end NUMINAMATH_CALUDE_music_books_cost_l3516_351668


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3516_351672

theorem solve_exponential_equation :
  ∃ y : ℝ, (1000 : ℝ)^4 = 10^y ↔ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3516_351672


namespace NUMINAMATH_CALUDE_equation_solution_l3516_351671

theorem equation_solution (x : ℝ) : 
  (x / 3) / 5 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3516_351671


namespace NUMINAMATH_CALUDE_alcohol_mixture_theorem_l3516_351688

/-- Proves that mixing 300 mL of 10% alcohol solution with 100 mL of 30% alcohol solution results in a 15% alcohol solution -/
theorem alcohol_mixture_theorem :
  let x_volume : ℝ := 300
  let x_concentration : ℝ := 0.10
  let y_volume : ℝ := 100
  let y_concentration : ℝ := 0.30
  let total_volume : ℝ := x_volume + y_volume
  let total_alcohol : ℝ := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = 0.15 := by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_theorem_l3516_351688


namespace NUMINAMATH_CALUDE_point_four_units_from_negative_two_l3516_351642

theorem point_four_units_from_negative_two (x : ℝ) : 
  (|x - (-2)| = 4) ↔ (x = 2 ∨ x = -6) := by sorry

end NUMINAMATH_CALUDE_point_four_units_from_negative_two_l3516_351642


namespace NUMINAMATH_CALUDE_bills_trips_l3516_351676

theorem bills_trips (total_trips : ℕ) (jeans_trips : ℕ) (h1 : total_trips = 40) (h2 : jeans_trips = 23) :
  total_trips - jeans_trips = 17 :=
by sorry

end NUMINAMATH_CALUDE_bills_trips_l3516_351676


namespace NUMINAMATH_CALUDE_trick_decks_total_spend_l3516_351656

/-- The total amount spent by Victor and his friend on trick decks -/
def totalSpent (deckCost : ℕ) (victorDecks : ℕ) (friendDecks : ℕ) : ℕ :=
  deckCost * (victorDecks + friendDecks)

/-- Theorem stating the total amount spent by Victor and his friend -/
theorem trick_decks_total_spend :
  totalSpent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_total_spend_l3516_351656


namespace NUMINAMATH_CALUDE_complex_power_result_l3516_351653

theorem complex_power_result : (3 * (Complex.cos (30 * Real.pi / 180)) + 3 * Complex.I * (Complex.sin (30 * Real.pi / 180)))^4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_result_l3516_351653


namespace NUMINAMATH_CALUDE_pe_class_size_l3516_351634

theorem pe_class_size (fourth_grade_classes : ℕ) (students_per_class : ℕ) (total_cupcakes : ℕ) :
  fourth_grade_classes = 3 →
  students_per_class = 30 →
  total_cupcakes = 140 →
  total_cupcakes - (fourth_grade_classes * students_per_class) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_pe_class_size_l3516_351634


namespace NUMINAMATH_CALUDE_least_n_satisfying_conditions_l3516_351651

theorem least_n_satisfying_conditions : ∃ n : ℕ,
  n > 1 ∧
  2*n % 3 = 2 ∧
  3*n % 4 = 3 ∧
  4*n % 5 = 4 ∧
  5*n % 6 = 5 ∧
  (∀ m : ℕ, m > 1 ∧ 
    2*m % 3 = 2 ∧
    3*m % 4 = 3 ∧
    4*m % 5 = 4 ∧
    5*m % 6 = 5 → m ≥ n) ∧
  n = 61 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_conditions_l3516_351651


namespace NUMINAMATH_CALUDE_x_squared_divides_x_plus_y_l3516_351684

theorem x_squared_divides_x_plus_y (x y : ℕ) :
  x^2 ∣ (x^2 + x*y + x + y) → x^2 ∣ (x + y) := by
sorry

end NUMINAMATH_CALUDE_x_squared_divides_x_plus_y_l3516_351684


namespace NUMINAMATH_CALUDE_lenny_money_left_l3516_351641

/-- Calculates the amount of money Lenny has left after his expenses -/
def money_left (initial : ℕ) (expense1 : ℕ) (expense2 : ℕ) : ℕ :=
  initial - (expense1 + expense2)

/-- Proves that Lenny has $39 left after his expenses -/
theorem lenny_money_left :
  money_left 84 24 21 = 39 := by
  sorry

end NUMINAMATH_CALUDE_lenny_money_left_l3516_351641


namespace NUMINAMATH_CALUDE_solve_equation_l3516_351632

theorem solve_equation (p q r s : ℕ+) 
  (h1 : p^3 = q^2) 
  (h2 : r^5 = s^4) 
  (h3 : r - p = 31) : 
  (s : ℤ) - (q : ℤ) = -2351 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3516_351632


namespace NUMINAMATH_CALUDE_intersection_property_l3516_351675

-- Define the parabola and line
def parabola (x : ℝ) : ℝ := 2 * x^2
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the intersection points A and B
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

-- Define the midpoint M
def M (k : ℝ) : ℝ × ℝ := sorry

-- Define point N on x-axis
def N (k : ℝ) : ℝ × ℝ := sorry

-- Define vectors NA and NB
def NA (k : ℝ) : ℝ × ℝ := sorry
def NB (k : ℝ) : ℝ × ℝ := sorry

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

theorem intersection_property (k : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ parabola x₁ = line k x₁ ∧ parabola x₂ = line k x₂) →
  (dot_product (NA k) (NB k) = 0) →
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_property_l3516_351675


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3516_351643

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3*b = 1) :
  1/a + 3/b ≥ 16 :=
sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3*b = 1 ∧ 1/a + 3/b = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3516_351643


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l3516_351631

theorem midpoint_distance_theorem (t : ℝ) : 
  let A : ℝ × ℝ := (2*t - 4, -3)
  let B : ℝ × ℝ := (-6, 2*t + 5)
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (M.1 - B.1)^2 + (M.2 - B.2)^2 = 4*t^2 + 3*t →
  t = (7 + Real.sqrt 185) / 4 ∨ t = (7 - Real.sqrt 185) / 4 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l3516_351631


namespace NUMINAMATH_CALUDE_total_visible_area_formula_l3516_351622

/-- The total area of the visible large rectangle and the additional rectangle, excluding the hole -/
def total_visible_area (x : ℝ) : ℝ :=
  (x + 8) * (x + 6) - (2*x - 4) * (x - 3) + (x + 2) * x

/-- Theorem stating that the total visible area is equal to 26x + 36 -/
theorem total_visible_area_formula (x : ℝ) :
  total_visible_area x = 26 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_total_visible_area_formula_l3516_351622


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3516_351618

/-- Represents a hyperbola in the form x²/a² - y²/b² = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a parabola in the form y² = 2px --/
structure Parabola where
  p : ℝ
  h_pos_p : p > 0

/-- The focal length of a hyperbola is the distance from the center to a focus --/
def focal_length (h : Hyperbola) : ℝ := sorry

/-- The left vertex of a hyperbola --/
def left_vertex (h : Hyperbola) : ℝ × ℝ := sorry

/-- The focus of a parabola --/
def parabola_focus (p : Parabola) : ℝ × ℝ := sorry

/-- The directrix of a parabola --/
def parabola_directrix (p : Parabola) : ℝ × ℝ → Prop := sorry

/-- An asymptote of a hyperbola --/
def hyperbola_asymptote (h : Hyperbola) : ℝ × ℝ → Prop := sorry

/-- The distance between two points --/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focal_length 
  (h : Hyperbola) 
  (p : Parabola) 
  (h_distance : distance (left_vertex h) (parabola_focus p) = 4)
  (h_intersection : ∃ (pt : ℝ × ℝ), hyperbola_asymptote h pt ∧ parabola_directrix p pt ∧ pt = (-2, -1)) :
  focal_length h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3516_351618


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3516_351678

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 5, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![2, 6; -1, 3]

theorem matrix_multiplication_result :
  A * B = !![7, 15; 8, 36] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3516_351678


namespace NUMINAMATH_CALUDE_constant_term_is_negative_one_l3516_351601

-- Define the quadratic equation
def quadratic_equation (x : ℝ) (b : ℝ) : Prop :=
  2 * x^2 - b * x = 1

-- Theorem: The constant term of the quadratic equation 2x^2 - bx = 1 is -1
theorem constant_term_is_negative_one (b : ℝ) :
  ∃ (a c : ℝ), ∀ x, quadratic_equation x b ↔ a * x^2 + b * x + c = 0 ∧ c = -1 :=
sorry

end NUMINAMATH_CALUDE_constant_term_is_negative_one_l3516_351601


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3516_351667

theorem sufficient_not_necessary (x₁ x₂ : ℝ) :
  (∀ x₁ x₂ : ℝ, (x₁ > 1 ∧ x₂ > 1) → (x₁ + x₂ > 2 ∧ x₁ * x₂ > 1)) ∧
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ > 2 ∧ x₁ * x₂ > 1) ∧ ¬(x₁ > 1 ∧ x₂ > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3516_351667


namespace NUMINAMATH_CALUDE_product_first_10000_trailing_zeros_l3516_351670

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The product of the first 10000 natural numbers has 2499 trailing zeros -/
theorem product_first_10000_trailing_zeros :
  trailingZeros 10000 = 2499 := by
  sorry

end NUMINAMATH_CALUDE_product_first_10000_trailing_zeros_l3516_351670


namespace NUMINAMATH_CALUDE_more_heads_than_tails_probability_l3516_351682

/-- The probability of getting more heads than tails when tossing a fair coin 4 times -/
def probability_more_heads_than_tails : ℚ := 5/16

/-- A fair coin is tossed 4 times -/
def num_tosses : ℕ := 4

/-- The probability of getting heads on a single toss of a fair coin -/
def probability_heads : ℚ := 1/2

/-- The probability of getting tails on a single toss of a fair coin -/
def probability_tails : ℚ := 1/2

theorem more_heads_than_tails_probability :
  probability_more_heads_than_tails = 
    (Nat.choose num_tosses 3 : ℚ) * probability_heads^3 * probability_tails +
    (Nat.choose num_tosses 4 : ℚ) * probability_heads^4 :=
by sorry

end NUMINAMATH_CALUDE_more_heads_than_tails_probability_l3516_351682


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_l3516_351639

def condition_P (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 0

def condition_Q (x y : ℝ) : Prop := (x - 1) * (y - 1) = 0

theorem P_sufficient_not_necessary :
  (∀ x y : ℝ, condition_P x y → condition_Q x y) ∧
  ¬(∀ x y : ℝ, condition_Q x y → condition_P x y) := by
  sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_l3516_351639


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3516_351677

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_negative_2023 :
  reciprocal (-2023) = -1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3516_351677


namespace NUMINAMATH_CALUDE_power_mod_28_l3516_351607

theorem power_mod_28 : 17^1801 ≡ 17 [ZMOD 28] := by sorry

end NUMINAMATH_CALUDE_power_mod_28_l3516_351607


namespace NUMINAMATH_CALUDE_group_photo_arrangements_l3516_351619

theorem group_photo_arrangements :
  let total_volunteers : ℕ := 6
  let male_volunteers : ℕ := 4
  let female_volunteers : ℕ := 2
  let elderly_people : ℕ := 2
  let elderly_arrangements : ℕ := 2  -- Number of ways to arrange elderly people
  let female_arrangements : ℕ := 2   -- Number of ways to arrange female volunteers
  let male_arrangements : ℕ := 24    -- Number of ways to arrange male volunteers (4!)
  
  total_volunteers = male_volunteers + female_volunteers + elderly_people →
  (elderly_arrangements * female_arrangements * male_arrangements) = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_group_photo_arrangements_l3516_351619


namespace NUMINAMATH_CALUDE_regression_line_estimate_estimated_y_value_l3516_351614

/-- Given a regression line equation and an x-value, calculate the estimated y-value -/
theorem regression_line_estimate (slope intercept x : ℝ) :
  let regression_line := fun (x : ℝ) => slope * x + intercept
  regression_line x = slope * x + intercept := by sorry

/-- The estimated y-value for the given regression line when x = 28 is 390 -/
theorem estimated_y_value :
  let slope : ℝ := 4.75
  let intercept : ℝ := 257
  let x : ℝ := 28
  let regression_line := fun (x : ℝ) => slope * x + intercept
  regression_line x = 390 := by sorry

end NUMINAMATH_CALUDE_regression_line_estimate_estimated_y_value_l3516_351614


namespace NUMINAMATH_CALUDE_base9_85_to_decimal_l3516_351660

/-- Converts a two-digit number in base 9 to its decimal representation -/
def base9_to_decimal (tens : Nat) (ones : Nat) : Nat :=
  tens * 9 + ones

/-- States that 85 in base 9 is equal to 77 in decimal -/
theorem base9_85_to_decimal : base9_to_decimal 8 5 = 77 := by
  sorry

end NUMINAMATH_CALUDE_base9_85_to_decimal_l3516_351660


namespace NUMINAMATH_CALUDE_girls_in_class_l3516_351692

/-- Represents the number of girls in a class given the total number of students and the ratio of girls to boys. -/
def number_of_girls (total : ℕ) (girl_ratio : ℕ) (boy_ratio : ℕ) : ℕ :=
  (total * girl_ratio) / (girl_ratio + boy_ratio)

/-- Theorem stating that in a class of 63 students with a girl-to-boy ratio of 4:3, there are 36 girls. -/
theorem girls_in_class : number_of_girls 63 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l3516_351692


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l3516_351624

theorem trig_expression_equals_negative_four :
  1 / Real.sin (70 * π / 180) - Real.sqrt 3 / Real.cos (70 * π / 180) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l3516_351624


namespace NUMINAMATH_CALUDE_max_equilateral_triangles_l3516_351662

-- Define the number of line segments
def num_segments : ℕ := 6

-- Define the length of each segment
def segment_length : ℝ := 2

-- Define the side length of the equilateral triangles
def triangle_side_length : ℝ := 2

-- State the theorem
theorem max_equilateral_triangles :
  ∃ (n : ℕ), n ≤ 4 ∧
  (∀ (m : ℕ), (∃ (arrangement : List (List ℕ)),
    (∀ triangle ∈ arrangement, triangle.length = 3 ∧
     (∀ side ∈ triangle, side ≤ num_segments) ∧
     arrangement.length = m) →
    m ≤ n)) ∧
  (∃ (arrangement : List (List ℕ)),
    (∀ triangle ∈ arrangement, triangle.length = 3 ∧
     (∀ side ∈ triangle, side ≤ num_segments) ∧
     arrangement.length = 4)) :=
by sorry

end NUMINAMATH_CALUDE_max_equilateral_triangles_l3516_351662


namespace NUMINAMATH_CALUDE_divisibility_problem_l3516_351697

theorem divisibility_problem (a : ℕ) :
  (∃! n : Fin 4, ¬ (
    (n = 0 → a % 2 = 0) ∧
    (n = 1 → a % 4 = 0) ∧
    (n = 2 → a % 12 = 0) ∧
    (n = 3 → a % 24 = 0)
  )) →
  ¬(a % 24 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3516_351697


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3516_351638

theorem trigonometric_identity :
  3 * Real.tan (10 * π / 180) + 4 * Real.sqrt 3 * Real.sin (10 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3516_351638


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_equivalence_l3516_351683

/-- A line in 2D space defined by parametric equations. -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric line. -/
def givenLine : ParametricLine where
  x := λ t => 5 + 3 * t
  y := λ t => 10 - 4 * t

/-- The Cartesian form of a line: ax + by + c = 0 -/
structure CartesianLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The Cartesian line we want to prove is equivalent. -/
def targetLine : CartesianLine where
  a := 4
  b := 3
  c := -50

/-- 
Theorem: The given parametric line is equivalent to the target Cartesian line.
-/
theorem parametric_to_cartesian_equivalence :
  ∀ t : ℝ, 
  4 * (givenLine.x t) + 3 * (givenLine.y t) - 50 = 0 :=
by
  sorry

#check parametric_to_cartesian_equivalence

end NUMINAMATH_CALUDE_parametric_to_cartesian_equivalence_l3516_351683


namespace NUMINAMATH_CALUDE_perpendicular_bisector_implies_m_equals_three_l3516_351679

/-- Given two points A and B, if the equation of the perpendicular bisector 
    of segment AB is x + 2y - 2 = 0, then the x-coordinate of B is 3. -/
theorem perpendicular_bisector_implies_m_equals_three 
  (A B : ℝ × ℝ) 
  (h1 : A = (1, -2))
  (h2 : B.2 = 2)
  (h3 : ∀ x y : ℝ, (x + 2*y - 2 = 0) ↔ 
    (x = (A.1 + B.1)/2 ∧ y = (A.2 + B.2)/2)) : 
  B.1 = 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_implies_m_equals_three_l3516_351679


namespace NUMINAMATH_CALUDE_expression_simplification_l3516_351695

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 - 3 / (x + 2)) / ((x^2 - 2*x + 1) / (3*x + 6)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3516_351695


namespace NUMINAMATH_CALUDE_project_completion_time_l3516_351610

/-- Given a project requiring 1500 hours and a daily work schedule of 15 hours,
    prove that the number of days needed to complete the project is 100. -/
theorem project_completion_time (project_hours : ℕ) (daily_hours : ℕ) :
  project_hours = 1500 →
  daily_hours = 15 →
  project_hours / daily_hours = 100 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l3516_351610


namespace NUMINAMATH_CALUDE_complex_division_result_l3516_351647

theorem complex_division_result : (4 + 3*I : ℂ) / (2 - I) = 1 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l3516_351647


namespace NUMINAMATH_CALUDE_table_tennis_tournament_impossibility_l3516_351611

theorem table_tennis_tournament_impossibility (k : ℕ) (h : k > 0) :
  let participants := 2 * k
  let total_matches := k * (2 * k - 1)
  let total_judgements := 2 * total_matches
  ¬ ∃ (judgements_per_participant : ℕ),
    (judgements_per_participant * participants = total_judgements ∧
     judgements_per_participant * 2 = 2 * k - 1) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_impossibility_l3516_351611


namespace NUMINAMATH_CALUDE_total_bags_l3516_351659

theorem total_bags (points_per_bag : ℕ) (total_points : ℕ) (unrecycled_bags : ℕ) : 
  points_per_bag = 5 → total_points = 45 → unrecycled_bags = 8 →
  (total_points / points_per_bag + unrecycled_bags : ℕ) = 17 := by
sorry

end NUMINAMATH_CALUDE_total_bags_l3516_351659


namespace NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l3516_351664

theorem equation_represents_two_intersecting_lines :
  ∃ (m₁ m₂ b₁ b₂ : ℝ), m₁ ≠ m₂ ∧
  (∀ x y : ℝ, x^3 * (2*x + 2*y + 3) = y^3 * (2*x + 2*y + 3) ↔ 
    (y = m₁ * x + b₁) ∨ (y = m₂ * x + b₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l3516_351664


namespace NUMINAMATH_CALUDE_sunset_time_theorem_l3516_351630

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  m_valid : minutes < 60

def add_time_and_duration (t : Time) (d : Duration) : Time :=
  sorry

def time_to_12hour_format (t : Time) : Time :=
  sorry

theorem sunset_time_theorem (sunrise : Time) (daylight : Duration) :
  sunrise.hours = 6 ∧ sunrise.minutes = 43 ∧
  daylight.hours = 11 ∧ daylight.minutes = 12 →
  let sunset := add_time_and_duration sunrise daylight
  let sunset_12h := time_to_12hour_format sunset
  sunset_12h.hours = 5 ∧ sunset_12h.minutes = 55 :=
sorry

end NUMINAMATH_CALUDE_sunset_time_theorem_l3516_351630


namespace NUMINAMATH_CALUDE_simplify_expression_l3516_351698

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3516_351698


namespace NUMINAMATH_CALUDE_problem_solution_l3516_351665

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x + 1| - |x - a|

def g (a : ℝ) (x : ℝ) : ℝ := f a x + 3 * |x - a|

theorem problem_solution :
  (∀ x : ℝ, f 2 x ≥ 0 ↔ x ≤ -4 ∨ x ≥ 0) ∧
  (∃ t : ℝ, (∀ x : ℝ, g 1 x ≥ t) ∧
   (∀ m n : ℝ, m > 0 → n > 0 → 2/m + 1/(2*n) = t →
    m + n ≥ 9/8) ∧
   (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2/m + 1/(2*n) = t ∧ m + n = 9/8)) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l3516_351665


namespace NUMINAMATH_CALUDE_student_line_length_l3516_351626

/-- The length of a line of students, given the number of students and the distance between them. -/
def line_length (num_students : ℕ) (distance : ℝ) : ℝ :=
  (num_students - 1 : ℝ) * distance

/-- Theorem stating that the length of a line formed by 51 students with 3 meters between each adjacent pair is 150 meters. -/
theorem student_line_length : line_length 51 3 = 150 := by
  sorry

#eval line_length 51 3

end NUMINAMATH_CALUDE_student_line_length_l3516_351626


namespace NUMINAMATH_CALUDE_initial_worksheets_count_l3516_351680

/-- Given that a teacher would have 20 worksheets to grade after grading 4 and receiving 18 more,
    prove that she initially had 6 worksheets to grade. -/
theorem initial_worksheets_count : ∀ x : ℕ, x - 4 + 18 = 20 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_worksheets_count_l3516_351680


namespace NUMINAMATH_CALUDE_arithmetic_sequence_value_l3516_351625

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- State the theorem
theorem arithmetic_sequence_value (a : ℚ) :
  (∃ (seq : ℕ → ℚ), is_arithmetic_sequence seq ∧ 
    seq 0 = a - 1 ∧ seq 1 = 2*a + 1 ∧ seq 2 = a + 4) →
  a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_value_l3516_351625


namespace NUMINAMATH_CALUDE_impossibility_of_group_division_l3516_351620

theorem impossibility_of_group_division : ¬ ∃ (S : ℕ),
  S + (S + 10) + (S + 20) + (S + 30) = (1980 * 1981) / 2 := by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_group_division_l3516_351620


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3516_351654

theorem cost_price_calculation (selling_price_profit selling_price_loss : ℕ) 
  (h : selling_price_profit - selling_price_loss = 2 * (selling_price_profit - 50)) :
  50 = (selling_price_profit + selling_price_loss) / 2 := by
  sorry

#check cost_price_calculation 57 43

end NUMINAMATH_CALUDE_cost_price_calculation_l3516_351654


namespace NUMINAMATH_CALUDE_minimum_value_of_S_l3516_351606

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℚ := (3 * n^2 - 95 * n) / 2

/-- The minimum value of S(n) for positive integers n -/
def min_S : ℚ := -392

theorem minimum_value_of_S :
  ∀ n : ℕ, n > 0 → S n ≥ min_S :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_S_l3516_351606


namespace NUMINAMATH_CALUDE_gina_rose_cups_per_hour_l3516_351691

theorem gina_rose_cups_per_hour :
  let lily_cups_per_hour : ℕ := 7
  let order_rose_cups : ℕ := 6
  let order_lily_cups : ℕ := 14
  let total_payment : ℕ := 90
  let hourly_rate : ℕ := 30
  let rose_cups_per_hour : ℕ := order_rose_cups / (total_payment / hourly_rate - order_lily_cups / lily_cups_per_hour)
  rose_cups_per_hour = 6 := by
sorry

end NUMINAMATH_CALUDE_gina_rose_cups_per_hour_l3516_351691


namespace NUMINAMATH_CALUDE_golden_ratio_problem_l3516_351689

theorem golden_ratio_problem (m n : ℝ) : 
  m = 2 * Real.sin (18 * π / 180) →
  m^2 + n = 4 →
  (m * Real.sqrt n) / (2 * Real.cos (27 * π / 180)^2 - 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_golden_ratio_problem_l3516_351689


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_2023x_l3516_351612

theorem factorization_x_squared_minus_2023x (x : ℝ) : x^2 - 2023*x = x*(x - 2023) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_2023x_l3516_351612


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l3516_351663

/-- The area of a regular hexagon inscribed in a circle with area 256π -/
theorem inscribed_hexagon_area : 
  ∀ (circle_area : ℝ) (hexagon_area : ℝ),
  circle_area = 256 * Real.pi →
  hexagon_area = 384 * Real.sqrt 3 →
  (∃ (r : ℝ), 
    r > 0 ∧
    circle_area = Real.pi * r^2 ∧
    hexagon_area = 6 * ((r^2 * Real.sqrt 3) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l3516_351663


namespace NUMINAMATH_CALUDE_abc_sum_l3516_351636

theorem abc_sum (a b c : ℝ) (ha : |a| = 1) (hb : |b| = 2) (hc : |c| = 4) (horder : a > b ∧ b > c) :
  a - b + c = -1 ∨ a - b + c = -3 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l3516_351636


namespace NUMINAMATH_CALUDE_solution_set_when_m_eq_3_range_of_m_for_all_x_geq_8_l3516_351613

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

-- Theorem for part 1
theorem solution_set_when_m_eq_3 :
  {x : ℝ | f x 3 ≥ 6} = {x : ℝ | x ≤ -2 ∨ x ≥ 4} :=
sorry

-- Theorem for part 2
theorem range_of_m_for_all_x_geq_8 :
  (∀ x : ℝ, f x m ≥ 8) ↔ (m ≥ 7 ∨ m ≤ -9) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_m_eq_3_range_of_m_for_all_x_geq_8_l3516_351613


namespace NUMINAMATH_CALUDE_angle_610_equivalent_l3516_351666

def same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₁ = θ₂ + k * 360

theorem angle_610_equivalent :
  ∀ k : ℤ, same_terminal_side 610 (k * 360 + 250) := by sorry

end NUMINAMATH_CALUDE_angle_610_equivalent_l3516_351666


namespace NUMINAMATH_CALUDE_division_equality_l3516_351608

theorem division_equality (h : (204 : ℝ) / 12.75 = 16) : (2.04 : ℝ) / 1.275 = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l3516_351608


namespace NUMINAMATH_CALUDE_jose_profit_share_l3516_351628

/-- Calculates the share of profit for an investor based on their investment amount, 
    investment duration, and the total profit. -/
def calculate_profit_share (investment : ℕ) (duration : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * duration * total_profit) / total_investment_months

theorem jose_profit_share :
  let tom_investment := 30000
  let tom_duration := 12
  let jose_investment := 45000
  let jose_duration := 10
  let total_profit := 72000
  let total_investment_months := tom_investment * tom_duration + jose_investment * jose_duration
  calculate_profit_share jose_investment jose_duration total_investment_months total_profit = 40000 := by
sorry

#eval calculate_profit_share 45000 10 810000 72000

end NUMINAMATH_CALUDE_jose_profit_share_l3516_351628


namespace NUMINAMATH_CALUDE_onions_sold_l3516_351617

theorem onions_sold (initial : ℕ) (left : ℕ) (sold : ℕ) : 
  initial = 98 → left = 33 → sold = initial - left → sold = 65 := by
sorry

end NUMINAMATH_CALUDE_onions_sold_l3516_351617


namespace NUMINAMATH_CALUDE_expand_expression_l3516_351605

theorem expand_expression (x y : ℝ) : (3 * x + 15) * (4 * y + 12) = 12 * x * y + 36 * x + 60 * y + 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3516_351605


namespace NUMINAMATH_CALUDE_seokjin_rank_l3516_351602

def jimin_rank : ℕ := 24
def rank_difference : ℕ := 19

theorem seokjin_rank : 
  jimin_rank - rank_difference = 5 := by sorry

end NUMINAMATH_CALUDE_seokjin_rank_l3516_351602


namespace NUMINAMATH_CALUDE_cat_distribution_theorem_l3516_351650

/-- Represents the number of segments white cats are divided into by black cats -/
inductive X
| one
| two
| three
| four

/-- The probability distribution of X -/
def P (x : X) : ℚ :=
  match x with
  | X.one => 1 / 30
  | X.two => 9 / 30
  | X.three => 15 / 30
  | X.four => 5 / 30

theorem cat_distribution_theorem :
  (∀ x : X, 0 ≤ P x ∧ P x ≤ 1) ∧
  (P X.one + P X.two + P X.three + P X.four = 1) :=
sorry

end NUMINAMATH_CALUDE_cat_distribution_theorem_l3516_351650


namespace NUMINAMATH_CALUDE_inverse_computation_l3516_351686

-- Define the function g
def g : ℕ → ℕ
| 1 => 4
| 2 => 9
| 3 => 11
| 5 => 3
| 7 => 6
| 12 => 2
| _ => 0  -- for other inputs, we'll return 0

-- Assume g has an inverse
axiom g_has_inverse : Function.Bijective g

-- Define g_inv as the inverse of g
noncomputable def g_inv : ℕ → ℕ := Function.invFun g

-- State the theorem
theorem inverse_computation :
  g_inv ((g_inv 2 + g_inv 11) / g_inv 3) = 5 := by sorry

end NUMINAMATH_CALUDE_inverse_computation_l3516_351686


namespace NUMINAMATH_CALUDE_garrison_size_l3516_351616

theorem garrison_size (initial_days : ℕ) (reinforcement_size : ℕ) (days_before_reinforcement : ℕ) (remaining_days : ℕ) :
  initial_days = 62 →
  reinforcement_size = 2700 →
  days_before_reinforcement = 15 →
  remaining_days = 20 →
  ∃ (initial_men : ℕ),
    initial_men * (initial_days - days_before_reinforcement) = 
    (initial_men + reinforcement_size) * remaining_days ∧
    initial_men = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_garrison_size_l3516_351616


namespace NUMINAMATH_CALUDE_xiaoli_estimation_l3516_351649

theorem xiaoli_estimation (x y : ℝ) (h : x > y ∧ y > 0) :
  (1.1 * x - (y - 2) = x - y + 0.1 * x + 2) ∧
  (1.1 * x * (y - 2) = 1.1 * x * y - 2.2 * x) := by
  sorry

end NUMINAMATH_CALUDE_xiaoli_estimation_l3516_351649


namespace NUMINAMATH_CALUDE_prize_distribution_correct_l3516_351661

/-- Represents the prize distribution and cost calculation for a school event. -/
def prize_distribution (x : ℕ) : Prop :=
  let first_prize := x
  let second_prize := 4 * x - 10
  let third_prize := 90 - 5 * x
  let total_prizes := first_prize + second_prize + third_prize
  let total_cost := 18 * first_prize + 12 * second_prize + 6 * third_prize
  (total_prizes = 80) ∧ 
  (total_cost = 420 + 36 * x) ∧
  (x = 12 → total_cost = 852)

/-- Theorem stating the correctness of the prize distribution and cost calculation. -/
theorem prize_distribution_correct : 
  ∀ x : ℕ, prize_distribution x := by sorry

end NUMINAMATH_CALUDE_prize_distribution_correct_l3516_351661


namespace NUMINAMATH_CALUDE_remainder_problem_l3516_351685

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 7 = 1) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3516_351685


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l3516_351681

theorem largest_angle_in_triangle (X Y Z : Real) (h_scalene : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) 
  (h_angleY : Y = 25) (h_angleZ : Z = 100) : 
  max X (max Y Z) = 100 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l3516_351681


namespace NUMINAMATH_CALUDE_original_class_strength_l3516_351629

/-- Given an adult class, prove that the original strength was 18 students -/
theorem original_class_strength
  (original_avg : ℝ)
  (new_students : ℕ)
  (new_avg : ℝ)
  (avg_decrease : ℝ)
  (h1 : original_avg = 40)
  (h2 : new_students = 18)
  (h3 : new_avg = 32)
  (h4 : avg_decrease = 4)
  : ∃ (x : ℕ), x * original_avg + new_students * new_avg = (x + new_students) * (original_avg - avg_decrease) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_original_class_strength_l3516_351629


namespace NUMINAMATH_CALUDE_sequence_existence_and_extension_l3516_351637

theorem sequence_existence_and_extension (m : ℕ) (h : m ≥ 2) :
  (∃ x : ℕ → ℤ, ∀ i ∈ Finset.range m, x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) ∧
  (∀ x : ℕ → ℤ, (∀ i ∈ Finset.range m, x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
    ∃ y : ℤ → ℤ, (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
               (∀ i ∈ Finset.range (2 * m), y i = x i)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_and_extension_l3516_351637


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_cos_2beta_l3516_351655

theorem cos_2alpha_plus_cos_2beta (α β : Real) 
  (h1 : Real.sin α + Real.sin β = 1)
  (h2 : Real.cos α + Real.cos β = 0) :
  Real.cos (2 * α) + Real.cos (2 * β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_cos_2beta_l3516_351655


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3516_351621

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => -4/3 * x^3 + 6 * x^2 - 50/3 * x - 14/3
  (q 1 = -8) ∧ (q 2 = -12) ∧ (q 3 = -20) ∧ (q 4 = -40) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3516_351621


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l3516_351645

def x : ℕ := 5 * 16 * 27

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube :
  ∃! y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 → is_perfect_cube (x * z) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l3516_351645


namespace NUMINAMATH_CALUDE_probability_of_prime_ball_l3516_351669

def ball_numbers : List Nat := [3, 4, 5, 6, 7, 8, 11, 13]

def is_prime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d ≤ 1 || n % (d + 2) ≠ 0)

def count_primes (numbers : List Nat) : Nat :=
  (numbers.filter is_prime).length

theorem probability_of_prime_ball :
  (count_primes ball_numbers : Rat) / (ball_numbers.length : Rat) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_prime_ball_l3516_351669


namespace NUMINAMATH_CALUDE_train_speed_l3516_351635

/-- The speed of a train given its length and time to cross an electric pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 150) (h2 : time = 3) :
  length / time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3516_351635
