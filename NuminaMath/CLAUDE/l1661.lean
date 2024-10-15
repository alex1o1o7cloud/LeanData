import Mathlib

namespace NUMINAMATH_CALUDE_division_remainder_l1661_166189

theorem division_remainder : 
  let dividend : ℕ := 220020
  let divisor : ℕ := 555 + 445
  let quotient : ℕ := 2 * (555 - 445)
  dividend % divisor = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1661_166189


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1661_166101

theorem unique_integer_solution (w x y z : ℤ) :
  w^2 + 11*x^2 - 8*y^2 - 12*y*z - 10*z^2 = 0 →
  w = 0 ∧ x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1661_166101


namespace NUMINAMATH_CALUDE_max_cards_from_poster_board_l1661_166133

/-- Represents the dimensions of a rectangular object in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of small rectangles that can fit into a larger square -/
def maxRectangles (square_side : ℕ) (card : Dimensions) : ℕ :=
  (square_side / card.length) * (square_side / card.width)

theorem max_cards_from_poster_board :
  let poster_board_side : ℕ := 12  -- 1 foot = 12 inches
  let card : Dimensions := { length := 2, width := 3 }
  maxRectangles poster_board_side card = 24 := by
sorry

end NUMINAMATH_CALUDE_max_cards_from_poster_board_l1661_166133


namespace NUMINAMATH_CALUDE_chi_square_relationship_certainty_l1661_166118

-- Define the Chi-square test result
def chi_square_result : ℝ := 6.825

-- Define the degrees of freedom for a 2x2 contingency table
def degrees_of_freedom : ℕ := 1

-- Define the critical value for 99% confidence level
def critical_value : ℝ := 6.635

-- Define the certainty level
def certainty_level : ℝ := 0.99

-- Theorem statement
theorem chi_square_relationship_certainty :
  chi_square_result > critical_value →
  certainty_level = 0.99 :=
sorry

end NUMINAMATH_CALUDE_chi_square_relationship_certainty_l1661_166118


namespace NUMINAMATH_CALUDE_max_distance_F₂_to_l_max_value_PF₂_QF₂_range_F₁P_dot_F₁Q_l1661_166184

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define a line passing through F₁
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Define the intersection points P and Q
def Intersection (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ Ellipse x y ∧ Line k x y}

-- State the theorems
theorem max_distance_F₂_to_l :
  ∃ (k : ℝ), ∀ (l : ℝ → ℝ → Prop),
    (∀ (x y : ℝ), l x y ↔ Line k x y) →
    (∃ (d : ℝ), d = 4 ∧ ∀ (p : ℝ × ℝ), l p.1 p.2 → dist F₂ p ≤ d) :=
sorry

theorem max_value_PF₂_QF₂ :
  ∃ (P Q : ℝ × ℝ), P ∈ Intersection k ∧ Q ∈ Intersection k ∧
    ∀ (P' Q' : ℝ × ℝ), P' ∈ Intersection k → Q' ∈ Intersection k →
      dist P' F₂ + dist Q' F₂ ≤ 26/3 :=
sorry

theorem range_F₁P_dot_F₁Q :
  ∀ (k : ℝ), ∀ (P Q : ℝ × ℝ),
    P ∈ Intersection k → Q ∈ Intersection k →
    -5 ≤ (P.1 - F₁.1, P.2 - F₁.2) • (Q.1 - F₁.1, Q.2 - F₁.2) ∧
    (P.1 - F₁.1, P.2 - F₁.2) • (Q.1 - F₁.1, Q.2 - F₁.2) ≤ -25/9 :=
sorry

end NUMINAMATH_CALUDE_max_distance_F₂_to_l_max_value_PF₂_QF₂_range_F₁P_dot_F₁Q_l1661_166184


namespace NUMINAMATH_CALUDE_quadratic_range_l1661_166153

theorem quadratic_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l1661_166153


namespace NUMINAMATH_CALUDE_typing_orders_count_l1661_166106

/-- The number of letters to be typed during the day -/
def total_letters : ℕ := 12

/-- The set of letter numbers that have already been typed -/
def typed_letters : Finset ℕ := {10, 12}

/-- The set of letter numbers that could potentially be in the in-box -/
def potential_inbox : Finset ℕ := Finset.range 10 ∪ {11}

/-- Calculates the number of possible typing orders for the remaining letters -/
def possible_typing_orders : ℕ :=
  (Finset.powerset potential_inbox).sum (fun s => s.card + 1)

/-- The main theorem stating the number of possible typing orders -/
theorem typing_orders_count : possible_typing_orders = 6144 := by
  sorry

end NUMINAMATH_CALUDE_typing_orders_count_l1661_166106


namespace NUMINAMATH_CALUDE_transform_G_to_cup_l1661_166162

-- Define the set of shapes (including letters and symbols)
def Shape : Type := String

-- Define the transformations
def T₁ (s : Shape) : Shape := sorry
def T₂ (s : Shape) : Shape := sorry

-- Define the composition of transformations
def T (s : Shape) : Shape := T₂ (T₁ s)

-- State the theorem
theorem transform_G_to_cup (h1 : T₁ "R" = "y") (h2 : T₂ "y" = "B")
                           (h3 : T₁ "L" = "⌝") (h4 : T₂ "⌝" = "Γ") :
  T "G" = "∪" := by sorry

end NUMINAMATH_CALUDE_transform_G_to_cup_l1661_166162


namespace NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_min_product_exists_l1661_166195

theorem min_product_of_reciprocal_sum (x y : ℕ+) : 
  (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 7 → x * y ≥ 98 := by
  sorry

theorem min_product_exists : 
  ∃ (x y : ℕ+), (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 7 ∧ x * y = 98 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_min_product_exists_l1661_166195


namespace NUMINAMATH_CALUDE_kevin_initial_phones_l1661_166134

/-- The number of phones Kevin had at the beginning of the day -/
def initial_phones : ℕ := 33

/-- The number of phones Kevin repaired by afternoon -/
def repaired_phones : ℕ := 3

/-- The number of phones dropped off by a client -/
def new_phones : ℕ := 6

/-- The number of phones each person (Kevin and coworker) will repair -/
def phones_per_person : ℕ := 9

theorem kevin_initial_phones :
  initial_phones = 33 ∧
  repaired_phones = 3 ∧
  new_phones = 6 ∧
  phones_per_person = 9 →
  initial_phones + new_phones - repaired_phones = 2 * phones_per_person :=
by sorry

end NUMINAMATH_CALUDE_kevin_initial_phones_l1661_166134


namespace NUMINAMATH_CALUDE_equation_solutions_l1661_166142

theorem equation_solutions :
  (∃ (x : ℝ), (1/2) * (2*x - 5)^2 - 2 = 0 ↔ x = 7/2 ∨ x = 3/2) ∧
  (∃ (x : ℝ), x^2 - 4*x - 4 = 0 ↔ x = 2 + 2*Real.sqrt 2 ∨ x = 2 - 2*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1661_166142


namespace NUMINAMATH_CALUDE_handshakes_five_people_l1661_166164

/-- The number of handshakes between n people, where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

/-- There are 5 people in the room. -/
def num_people : ℕ := 5

theorem handshakes_five_people : handshakes num_people = 10 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_five_people_l1661_166164


namespace NUMINAMATH_CALUDE_inequality_proof_l1661_166145

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1661_166145


namespace NUMINAMATH_CALUDE_set_equality_implies_difference_l1661_166127

theorem set_equality_implies_difference (a b : ℝ) :
  ({0, b/a, b} : Set ℝ) = {1, a+b, a} → b - a = 2 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_difference_l1661_166127


namespace NUMINAMATH_CALUDE_min_value_of_M_l1661_166186

theorem min_value_of_M (a b : ℕ+) : 
  ∃ (m : ℕ), m = 3 * a.val ^ 2 - a.val * b.val ^ 2 - 2 * b.val - 4 ∧ 
  m ≥ 2 ∧ 
  ∀ (k : ℕ), k = 3 * a.val ^ 2 - a.val * b.val ^ 2 - 2 * b.val - 4 → k ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_M_l1661_166186


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1661_166104

theorem arccos_one_half_equals_pi_third : 
  Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1661_166104


namespace NUMINAMATH_CALUDE_cost_of_goods_l1661_166174

/-- The cost of goods problem -/
theorem cost_of_goods
  (mango_rice_ratio : ℝ)
  (flour_rice_ratio : ℝ)
  (flour_cost : ℝ)
  (h1 : 10 * mango_rice_ratio = 24)
  (h2 : 6 * flour_rice_ratio = 2)
  (h3 : flour_cost = 21) :
  4 * (24 / 10 * (2 / 6 * flour_cost)) + 3 * (2 / 6 * flour_cost) + 5 * flour_cost = 898.80 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_goods_l1661_166174


namespace NUMINAMATH_CALUDE_ice_cream_stacking_l1661_166110

theorem ice_cream_stacking (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 2) :
  (n! / k!) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_stacking_l1661_166110


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l1661_166120

/-- Represents the number of students in each grade --/
structure Students where
  eighth : ℕ
  seventh : ℕ
  sixth : ℕ

/-- The ratio of 8th-graders to 6th-graders is 5:3 --/
def ratio_8th_to_6th (s : Students) : Prop :=
  5 * s.sixth = 3 * s.eighth

/-- The ratio of 8th-graders to 7th-graders is 8:5 --/
def ratio_8th_to_7th (s : Students) : Prop :=
  8 * s.seventh = 5 * s.eighth

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.eighth + s.seventh + s.sixth

/-- The main theorem: The smallest possible number of students is 89 --/
theorem smallest_number_of_students :
  ∃ (s : Students), ratio_8th_to_6th s ∧ ratio_8th_to_7th s ∧
  (∀ (t : Students), ratio_8th_to_6th t ∧ ratio_8th_to_7th t →
    total_students s ≤ total_students t) ∧
  total_students s = 89 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l1661_166120


namespace NUMINAMATH_CALUDE_pancake_stacks_sold_l1661_166125

/-- The number of stacks of pancakes sold at a fundraiser -/
def pancake_stacks : ℕ := sorry

/-- The cost of one stack of pancakes in dollars -/
def pancake_cost : ℚ := 4

/-- The cost of one slice of bacon in dollars -/
def bacon_cost : ℚ := 2

/-- The number of bacon slices sold -/
def bacon_slices : ℕ := 90

/-- The total revenue from the fundraiser in dollars -/
def total_revenue : ℚ := 420

/-- Theorem stating that the number of pancake stacks sold is 60 -/
theorem pancake_stacks_sold : pancake_stacks = 60 := by sorry

end NUMINAMATH_CALUDE_pancake_stacks_sold_l1661_166125


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1661_166188

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x - 2*x + 1

theorem tangent_line_at_one (x y : ℝ) :
  let f' : ℝ → ℝ := λ t => 2*t * Real.log t + t - 2
  (x + y = 0) ↔ (y - f 1 = f' 1 * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1661_166188


namespace NUMINAMATH_CALUDE_ending_number_of_range_problem_solution_l1661_166192

theorem ending_number_of_range (start : Nat) (count : Nat) (divisor : Nat) : Nat :=
  let first_multiple := ((start + divisor - 1) / divisor) * divisor
  first_multiple + (count - 1) * divisor

theorem problem_solution : 
  ending_number_of_range 49 3 11 = 77 := by
  sorry

end NUMINAMATH_CALUDE_ending_number_of_range_problem_solution_l1661_166192


namespace NUMINAMATH_CALUDE_sams_juice_consumption_l1661_166107

theorem sams_juice_consumption (total_juice : ℚ) (sams_portion : ℚ) : 
  total_juice = 3/7 → sams_portion = 4/5 → sams_portion * total_juice = 12/35 := by
  sorry

end NUMINAMATH_CALUDE_sams_juice_consumption_l1661_166107


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l1661_166137

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles (carol jordan : Rectangle) 
  (h1 : carol.length = 15)
  (h2 : carol.width = 20)
  (h3 : jordan.length = 6)
  (h4 : area carol = area jordan) :
  jordan.width = 50 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l1661_166137


namespace NUMINAMATH_CALUDE_handshake_remainder_l1661_166103

/-- The number of ways 8 people can shake hands, where each person shakes hands with exactly 2 others -/
def M : ℕ := sorry

/-- The group size -/
def group_size : ℕ := 8

/-- The number of handshakes per person -/
def handshakes_per_person : ℕ := 2

theorem handshake_remainder : M ≡ 355 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshake_remainder_l1661_166103


namespace NUMINAMATH_CALUDE_divisibility_property_implies_factor_of_99_l1661_166179

def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisibility_property_implies_factor_of_99 (k : ℕ) :
  (∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) →
  99 ∣ k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_implies_factor_of_99_l1661_166179


namespace NUMINAMATH_CALUDE_expression_evaluation_l1661_166169

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^y + 2 * y^x = 533 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1661_166169


namespace NUMINAMATH_CALUDE_subtraction_division_result_l1661_166112

theorem subtraction_division_result : 1.85 - 1.85 / 1.85 = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_division_result_l1661_166112


namespace NUMINAMATH_CALUDE_f_condition_iff_sum_less_two_l1661_166135

open Real

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The first condition: f(x) = f(2-x) -/
axiom f_symmetry (x : ℝ) : f x = f (2 - x)

/-- The second condition: f'(x)(x-1) > 0 -/
axiom f_derivative_condition (x : ℝ) : deriv f x * (x - 1) > 0

/-- The main theorem -/
theorem f_condition_iff_sum_less_two (x₁ x₂ : ℝ) (h : x₁ < x₂) :
  f x₁ > f x₂ ↔ x₁ + x₂ < 2 :=
sorry

end NUMINAMATH_CALUDE_f_condition_iff_sum_less_two_l1661_166135


namespace NUMINAMATH_CALUDE_always_positive_l1661_166122

theorem always_positive (x : ℝ) : x^2 + |x| + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l1661_166122


namespace NUMINAMATH_CALUDE_min_a_for_f_nonpositive_l1661_166130

noncomputable def f (a x : ℝ) : ℝ := Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x

theorem min_a_for_f_nonpositive :
  (∃ (a : ℝ), ∀ (x : ℝ), x ≥ -2 → f a x ≤ 0) ∧
  (∀ (b : ℝ), b < 1 - 1/Real.exp 1 → ∃ (x : ℝ), x ≥ -2 ∧ f b x > 0) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_f_nonpositive_l1661_166130


namespace NUMINAMATH_CALUDE_gcd_of_polynomials_l1661_166167

def is_even_multiple_of_5959 (b : ℤ) : Prop :=
  ∃ k : ℤ, b = 2 * 5959 * k

theorem gcd_of_polynomials (b : ℤ) (h : is_even_multiple_of_5959 b) :
  Int.gcd (4 * b^2 + 73 * b + 156) (4 * b + 15) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomials_l1661_166167


namespace NUMINAMATH_CALUDE_expression_percentage_of_y_l1661_166155

theorem expression_percentage_of_y (y z : ℝ) (hy : y > 0) :
  ((2 * y + z) / 10 + (3 * y - z) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_percentage_of_y_l1661_166155


namespace NUMINAMATH_CALUDE_figure_50_squares_l1661_166152

def square_count (n : ℕ) : ℕ := 2 * n^2 + 3 * n + 1

theorem figure_50_squares :
  square_count 0 = 1 ∧
  square_count 1 = 6 ∧
  square_count 2 = 15 ∧
  square_count 3 = 28 →
  square_count 50 = 5151 := by
  sorry

end NUMINAMATH_CALUDE_figure_50_squares_l1661_166152


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1661_166148

/-- A circular segment with a 120° arc and height h -/
structure CircularSegment :=
  (h : ℝ)
  (arc_angle : ℝ)
  (arc_angle_eq : arc_angle = 120)

/-- A rectangle inscribed in a circular segment -/
structure InscribedRectangle (seg : CircularSegment) :=
  (AB : ℝ)
  (BC : ℝ)
  (ratio : AB / BC = 1 / 4)
  (BC_on_chord : True)  -- Represents that BC lies on the chord

/-- The area of an inscribed rectangle -/
def area (seg : CircularSegment) (rect : InscribedRectangle seg) : ℝ :=
  rect.AB * rect.BC

/-- Theorem: The area of the inscribed rectangle is 36h²/25 -/
theorem inscribed_rectangle_area (seg : CircularSegment) (rect : InscribedRectangle seg) :
  area seg rect = 36 * seg.h^2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1661_166148


namespace NUMINAMATH_CALUDE_pqr_value_l1661_166131

theorem pqr_value (p q r : ℂ) 
  (eq1 : p * q + 5 * q = -20)
  (eq2 : q * r + 5 * r = -20)
  (eq3 : r * p + 5 * p = -20) : 
  p * q * r = 80 := by
sorry

end NUMINAMATH_CALUDE_pqr_value_l1661_166131


namespace NUMINAMATH_CALUDE_complex_product_real_l1661_166185

theorem complex_product_real (t : ℝ) : 
  let i : ℂ := Complex.I
  let z₁ : ℂ := 3 + 4 * i
  let z₂ : ℂ := t + i
  (z₁ * z₂).im = 0 → t = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l1661_166185


namespace NUMINAMATH_CALUDE_puzzle_solution_l1661_166149

theorem puzzle_solution (triangle square : ℤ) 
  (h1 : 3 + triangle = 5) 
  (h2 : triangle + square = 7) : 
  triangle + triangle + triangle + square + square = 16 := by
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1661_166149


namespace NUMINAMATH_CALUDE_problem_solution_l1661_166111

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0

-- Theorem to prove
theorem problem_solution : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1661_166111


namespace NUMINAMATH_CALUDE_systematic_sample_third_element_l1661_166108

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_sample : ℕ
  interval : ℕ

/-- Checks if a seat number is in the systematic sample -/
def in_sample (s : SystematicSample) (seat : ℕ) : Prop :=
  ∃ k : ℕ, seat = s.first_sample + k * s.interval ∧ seat ≤ s.population_size

theorem systematic_sample_third_element 
  (s : SystematicSample)
  (h_pop : s.population_size = 45)
  (h_sample : s.sample_size = 3)
  (h_interval : s.interval = s.population_size / s.sample_size)
  (h_11 : in_sample s 11)
  (h_41 : in_sample s 41) :
  in_sample s 26 := by
  sorry

#check systematic_sample_third_element

end NUMINAMATH_CALUDE_systematic_sample_third_element_l1661_166108


namespace NUMINAMATH_CALUDE_vacation_miles_per_day_l1661_166151

theorem vacation_miles_per_day 
  (vacation_days : ℝ) 
  (total_miles : ℝ) 
  (h1 : vacation_days = 5.0) 
  (h2 : total_miles = 1250) : 
  total_miles / vacation_days = 250 := by
sorry

end NUMINAMATH_CALUDE_vacation_miles_per_day_l1661_166151


namespace NUMINAMATH_CALUDE_star_power_equality_l1661_166163

/-- The k-th smallest positive integer not in X -/
def f (X : Finset ℕ) (k : ℕ) : ℕ := sorry

/-- The operation * for finite sets of positive integers -/
def starOp (X Y : Finset ℕ) : Finset ℕ :=
  X ∪ (Y.image (f X))

/-- Repeated application of starOp -/
def starPower (X : Finset ℕ) : ℕ → Finset ℕ
  | 0 => X
  | n + 1 => starOp X (starPower X n)

theorem star_power_equality {A B : Finset ℕ} (ha : A.card > 0) (hb : B.card > 0)
    (h : starOp A B = starOp B A) :
    starPower A B.card = starPower B A.card := by
  sorry

end NUMINAMATH_CALUDE_star_power_equality_l1661_166163


namespace NUMINAMATH_CALUDE_carpooling_distance_ratio_l1661_166176

/-- Proves that the ratio of the distance driven between the second friend's house and work
    to the total distance driven to the first and second friend's houses is 3:1 -/
theorem carpooling_distance_ratio :
  let distance_to_first : ℝ := 8
  let distance_to_second : ℝ := distance_to_first / 2
  let distance_to_work : ℝ := 36
  let total_distance_to_friends : ℝ := distance_to_first + distance_to_second
  (distance_to_work / total_distance_to_friends) = 3
  := by sorry

end NUMINAMATH_CALUDE_carpooling_distance_ratio_l1661_166176


namespace NUMINAMATH_CALUDE_angle_measure_of_special_triangle_l1661_166157

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (C : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (angle_range : 0 < C ∧ C < π)
  (side_relation : a^2 + b^2 + a*b = c^2)

-- Theorem statement
theorem angle_measure_of_special_triangle (t : Triangle) : t.C = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_of_special_triangle_l1661_166157


namespace NUMINAMATH_CALUDE_math_fun_books_count_l1661_166129

theorem math_fun_books_count : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 18 * x + 8 * y = 92 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_math_fun_books_count_l1661_166129


namespace NUMINAMATH_CALUDE_cubic_function_extreme_points_l1661_166139

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- Predicate stating that f has exactly two extreme points -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0 ∧
  ∀ z : ℝ, f' a z = 0 → z = x ∨ z = y

theorem cubic_function_extreme_points (a : ℝ) :
  has_two_extreme_points a → a < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extreme_points_l1661_166139


namespace NUMINAMATH_CALUDE_fold_five_cut_once_l1661_166117

/-- The number of segments created by folding a rope n times and then cutting it once -/
def rope_segments (n : ℕ) : ℕ :=
  2^n + 1

/-- Theorem: Folding a rope 5 times and cutting it once results in 33 segments -/
theorem fold_five_cut_once : rope_segments 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fold_five_cut_once_l1661_166117


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_l1661_166114

theorem arithmetic_sequence_range (a : ℝ) :
  (∀ n : ℕ+, (1 + (a + n - 1)) / (a + n - 1) ≤ (1 + (a + 5 - 1)) / (a + 5 - 1)) →
  -4 < a ∧ a < -3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_l1661_166114


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l1661_166115

theorem quadratic_completion_of_square (b : ℝ) (n : ℝ) : 
  (∀ x, x^2 + b*x + 19 = (x + n)^2 - 6) → b > 0 → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l1661_166115


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1661_166172

/-- Given two perpendicular vectors a = (3, -1) and b = (x, -2), prove that x = -2/3 -/
theorem perpendicular_vectors_x_value :
  let a : Fin 2 → ℝ := ![3, -1]
  let b : Fin 2 → ℝ := ![x, -2]
  (∀ i, i < 2 → a i * b i = 0) →
  x = -2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1661_166172


namespace NUMINAMATH_CALUDE_cylinder_cone_volume_relation_l1661_166159

/-- The volume of a cylinder with the same base and height as a cone is 3 times the volume of the cone -/
theorem cylinder_cone_volume_relation (Vcone : ℝ) (Vcylinder : ℝ) :
  Vcone > 0 → Vcylinder = 3 * Vcone := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_volume_relation_l1661_166159


namespace NUMINAMATH_CALUDE_farmers_field_planted_fraction_l1661_166198

theorem farmers_field_planted_fraction 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (leg_lengths : a = 5 ∧ b = 12) 
  (square_side : ℝ) 
  (square_distance_to_hypotenuse : ℝ) 
  (square_distance_condition : square_distance_to_hypotenuse = 3) 
  (square_tangent : square_side ≤ a ∧ square_side ≤ b) 
  (area_equation : (1/2) * c * square_distance_to_hypotenuse = (1/2) * a * b - square_side^2) :
  (((1/2) * a * b - square_side^2) / ((1/2) * a * b)) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_farmers_field_planted_fraction_l1661_166198


namespace NUMINAMATH_CALUDE_special_function_properties_l1661_166160

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

/-- The theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  f 2 = 0 ∧ ∃! v : ℝ, f 2 = v :=
sorry

end NUMINAMATH_CALUDE_special_function_properties_l1661_166160


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l1661_166126

/-- Given a piece of wood and a rope with unknown lengths, prove that the system of equations
    describing their relationship is correct based on the given measurements. -/
theorem sunzi_wood_measurement (x y : ℝ) : 
  (y - x = 4.5 ∧ y > x) →  -- Full rope measurement
  (x - y / 2 = 1 ∧ x > y / 2) →  -- Half rope measurement
  y - x = 4.5 ∧ x - y / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l1661_166126


namespace NUMINAMATH_CALUDE_ann_age_l1661_166109

/-- Ann's age in years -/
def A : ℕ := sorry

/-- Susan's age in years -/
def S : ℕ := sorry

/-- Ann is 5 years older than Susan -/
axiom age_difference : A = S + 5

/-- The sum of their ages is 27 -/
axiom age_sum : A + S = 27

/-- Prove that Ann is 16 years old -/
theorem ann_age : A = 16 := by sorry

end NUMINAMATH_CALUDE_ann_age_l1661_166109


namespace NUMINAMATH_CALUDE_x_power_twelve_l1661_166143

theorem x_power_twelve (x : ℝ) (h : x + 1/x = 3) : x^12 + 1/x^12 = 103682 := by
  sorry

end NUMINAMATH_CALUDE_x_power_twelve_l1661_166143


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1661_166173

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (((3 / (x - 1)) - x - 1) / ((x^2 - 4*x + 4) / (x - 1))) = (2 + x) / (2 - x) ∧
  (((3 / (0 - 1)) - 0 - 1) / ((0^2 - 4*0 + 4) / (0 - 1))) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1661_166173


namespace NUMINAMATH_CALUDE_larger_number_proof_l1661_166147

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 3300) (h3 : a > b) : a = 300 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1661_166147


namespace NUMINAMATH_CALUDE_number_division_remainder_l1661_166165

theorem number_division_remainder (N : ℕ) : 
  (N / 5 = 5 ∧ N % 5 = 0) → N % 11 = 3 := by
sorry

end NUMINAMATH_CALUDE_number_division_remainder_l1661_166165


namespace NUMINAMATH_CALUDE_expression_simplification_l1661_166100

theorem expression_simplification :
  80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1661_166100


namespace NUMINAMATH_CALUDE_servant_months_worked_l1661_166175

/-- Calculates the number of months served given the annual salary and the amount paid -/
def months_served (annual_salary : ℚ) (amount_paid : ℚ) : ℚ :=
  (amount_paid * 12) / annual_salary

theorem servant_months_worked (annual_salary : ℚ) (amount_paid : ℚ) 
  (h1 : annual_salary = 90)
  (h2 : amount_paid = 75) :
  months_served annual_salary amount_paid = 10 := by
  sorry

end NUMINAMATH_CALUDE_servant_months_worked_l1661_166175


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1661_166177

/-- For a rectangle with length to width ratio of 5:2 and diagonal d, 
    the area A can be expressed as A = (10/29)d^2 -/
theorem rectangle_area_diagonal (l w d : ℝ) (h_ratio : l / w = 5 / 2) 
    (h_diagonal : l^2 + w^2 = d^2) : l * w = (10/29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1661_166177


namespace NUMINAMATH_CALUDE_scallop_cost_theorem_l1661_166156

def scallop_cost (people : ℕ) (scallops_per_person : ℕ) (scallops_per_pound : ℕ) 
  (price_per_pound : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_scallops := people * scallops_per_person
  let pounds_needed := total_scallops / scallops_per_pound
  let initial_cost := pounds_needed * price_per_pound
  let discounted_cost := initial_cost * (1 - discount_rate)
  let final_cost := discounted_cost * (1 + tax_rate)
  final_cost

theorem scallop_cost_theorem :
  let result := scallop_cost 8 2 8 24 (1/10) (7/100)
  ⌊result * 100⌋ / 100 = 4622 / 100 := by sorry

end NUMINAMATH_CALUDE_scallop_cost_theorem_l1661_166156


namespace NUMINAMATH_CALUDE_jake_arrival_time_l1661_166121

-- Define the problem parameters
def floors : ℕ := 9
def steps_per_floor : ℕ := 30
def jake_steps_per_second : ℕ := 3
def elevator_time : ℕ := 60  -- in seconds

-- Calculate the total number of steps
def total_steps : ℕ := floors * steps_per_floor

-- Calculate Jake's descent time
def jake_time : ℕ := total_steps / jake_steps_per_second

-- Define the theorem
theorem jake_arrival_time :
  jake_time - elevator_time = 30 := by sorry

end NUMINAMATH_CALUDE_jake_arrival_time_l1661_166121


namespace NUMINAMATH_CALUDE_camp_children_count_l1661_166194

/-- Represents the number of children currently in the camp -/
def current_children : ℕ := sorry

/-- Represents the fraction of boys in the camp -/
def boy_fraction : ℚ := 9/10

/-- Represents the fraction of girls in the camp -/
def girl_fraction : ℚ := 1 - boy_fraction

/-- Represents the desired fraction of girls after adding more boys -/
def desired_girl_fraction : ℚ := 1/20

/-- Represents the number of additional boys to be added -/
def additional_boys : ℕ := 60

/-- Theorem stating that the current number of children in the camp is 60 -/
theorem camp_children_count : current_children = 60 := by
  sorry

end NUMINAMATH_CALUDE_camp_children_count_l1661_166194


namespace NUMINAMATH_CALUDE_exactly_one_correct_probability_l1661_166105

theorem exactly_one_correct_probability 
  (prob_a : ℝ) 
  (prob_b : ℝ) 
  (h_prob_a : prob_a = 0.7) 
  (h_prob_b : prob_b = 0.8) : 
  prob_a * (1 - prob_b) + (1 - prob_a) * prob_b = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_correct_probability_l1661_166105


namespace NUMINAMATH_CALUDE_negative_integer_equation_solution_l1661_166123

theorem negative_integer_equation_solution :
  ∀ N : ℤ, (N < 0) → (2 * N^2 + N = 15) → (N = -3) := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_equation_solution_l1661_166123


namespace NUMINAMATH_CALUDE_remainder_of_first_six_primes_sum_divided_by_seventh_prime_l1661_166144

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  ∃ (q : ℕ), 41 = 17 * q + 7 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_of_first_six_primes_sum_divided_by_seventh_prime_l1661_166144


namespace NUMINAMATH_CALUDE_root_equation_value_l1661_166128

theorem root_equation_value (a : ℝ) : 
  a^2 - 3*a - 1011 = 0 → 2*a^2 - 6*a + 1 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l1661_166128


namespace NUMINAMATH_CALUDE_multiply_by_eleven_l1661_166197

theorem multiply_by_eleven (A B : Nat) (h1 : A < 10) (h2 : B < 10) (h3 : A + B < 10) :
  (10 * A + B) * 11 = 100 * A + 10 * (A + B) + B := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_eleven_l1661_166197


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l1661_166182

theorem fraction_sum_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  b / a + a / b > 2 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l1661_166182


namespace NUMINAMATH_CALUDE_dereks_car_dog_ratio_l1661_166119

/-- Represents Derek's possessions at different ages --/
structure DereksPossessions where
  dogs_at_6 : ℕ
  cars_at_6 : ℕ
  dogs_at_16 : ℕ
  cars_at_16 : ℕ

/-- Theorem stating the ratio of cars to dogs when Derek is 16 --/
theorem dereks_car_dog_ratio (d : DereksPossessions) 
  (h1 : d.dogs_at_6 = 90)
  (h2 : d.dogs_at_6 = 3 * d.cars_at_6)
  (h3 : d.dogs_at_16 = 120)
  (h4 : d.cars_at_16 = d.cars_at_6 + 210)
  : d.cars_at_16 / d.dogs_at_16 = 2 := by
  sorry

#check dereks_car_dog_ratio

end NUMINAMATH_CALUDE_dereks_car_dog_ratio_l1661_166119


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1661_166171

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ + a₆ + a₁₁ = 3,
    prove that a₃ + a₉ = 2 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1661_166171


namespace NUMINAMATH_CALUDE_degree_relation_degree_bound_a2_zero_l1661_166136

/-- A real polynomial -/
def RealPolynomial := ℝ → ℝ

/-- The degree of a polynomial -/
noncomputable def degree (p : RealPolynomial) : ℕ := sorry

/-- Theorem for part (a) -/
theorem degree_relation (p : RealPolynomial) (h : degree p > 2) :
  degree p = 2 + degree (fun x => p (x + 1) + p (x - 1) - 2 * p x) := by sorry

/-- Theorem for part (b) -/
theorem degree_bound (p : RealPolynomial) (r s : ℝ)
  (h : ∀ x : ℝ, p (x + 1) + p (x - 1) - r * p x - s = 0) :
  degree p ≤ 2 := by sorry

/-- Theorem for part (c) -/
theorem a2_zero (p : RealPolynomial) (r : ℝ)
  (h : ∀ x : ℝ, p (x + 1) + p (x - 1) - r * p x = 0) :
  ∃ a₀ a₁, p = fun x => a₁ * x + a₀ := by sorry

end NUMINAMATH_CALUDE_degree_relation_degree_bound_a2_zero_l1661_166136


namespace NUMINAMATH_CALUDE_divisible_by_132_iff_in_list_l1661_166196

def is_valid_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), 
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = 1000 * x + 100 * y + 90 + z

theorem divisible_by_132_iff_in_list (n : ℕ) :
  is_valid_number n ∧ n % 132 = 0 ↔ n ∈ [3696, 4092, 6996, 7392] := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_132_iff_in_list_l1661_166196


namespace NUMINAMATH_CALUDE_inverse_proportion_increasing_l1661_166132

/-- For an inverse proportion function y = (m-5)/x, if y increases as x increases on each branch of its graph, then m < 5 -/
theorem inverse_proportion_increasing (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≠ 0 → x₂ ≠ 0 → x₁ < x₂ → (m - 5) / x₁ < (m - 5) / x₂) → 
  m < 5 :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_increasing_l1661_166132


namespace NUMINAMATH_CALUDE_arrangement_counts_l1661_166154

/-- The number of singing programs -/
def num_singing : ℕ := 5

/-- The number of dance programs -/
def num_dance : ℕ := 4

/-- The total number of programs -/
def total_programs : ℕ := num_singing + num_dance

/-- Calculates the number of permutations of n items taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of arrangements where no two dance programs are adjacent -/
def non_adjacent_arrangements : ℕ :=
  permutations num_singing num_singing * permutations (num_singing + 1) num_dance

/-- The number of arrangements with alternating singing and dance programs -/
def alternating_arrangements : ℕ :=
  permutations num_singing num_singing * permutations num_dance num_dance

theorem arrangement_counts :
  non_adjacent_arrangements = 43200 ∧ alternating_arrangements = 2880 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l1661_166154


namespace NUMINAMATH_CALUDE_vector_operation_l1661_166181

/-- Given two vectors a and b in R², prove that 2a - b equals (0,5) -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (2, -1)) :
  (2 : ℝ) • a - b = (0, 5) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l1661_166181


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l1661_166124

/-- Proves that the desired cost per pound of a candy mixture is $6.00 -/
theorem candy_mixture_cost
  (weight_expensive : ℝ)
  (price_expensive : ℝ)
  (weight_cheap : ℝ)
  (price_cheap : ℝ)
  (h1 : weight_expensive = 25)
  (h2 : price_expensive = 8)
  (h3 : weight_cheap = 50)
  (h4 : price_cheap = 5) :
  (weight_expensive * price_expensive + weight_cheap * price_cheap) /
  (weight_expensive + weight_cheap) = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l1661_166124


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1661_166180

-- Define the complex number
def z : ℂ := Complex.I * (1 - Complex.I)

-- Theorem statement
theorem z_in_first_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1661_166180


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1661_166170

/-- A function that checks if a natural number consists only of 2's and 7's in its decimal representation -/
def only_2_and_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 7

/-- A function that checks if a natural number has at least one 2 and one 7 in its decimal representation -/
def has_2_and_7 (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 7 ∈ n.digits 10

/-- The theorem stating the properties of the smallest number satisfying the given conditions -/
theorem smallest_number_with_conditions (m : ℕ) : 
  (∀ n : ℕ, n < m → ¬(n % 5 = 0 ∧ n % 8 = 0 ∧ only_2_and_7 n ∧ has_2_and_7 n)) →
  m % 5 = 0 ∧ m % 8 = 0 ∧ only_2_and_7 m ∧ has_2_and_7 m →
  m % 10000 = 7272 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1661_166170


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_inequality_solution_3_l1661_166190

-- Define the functions for each inequality
def f₁ (x : ℝ) := (x - 2)^11 * (x + 1)^22 * (x + 3)^33
def f₂ (x : ℝ) := (4*x + 3)^5 * (3*x + 2)^3 * (2*x + 1)
def f₃ (x : ℝ) := (x + 3) * (x + 1)^2 * (x - 2)^3 * (x - 4)

-- Define the solution sets
def S₁ : Set ℝ := {x | x ∈ (Set.Ioo (-3) (-1)) ∪ (Set.Ioo (-1) 2)}
def S₂ : Set ℝ := {x | x ∈ (Set.Iic (-3/4)) ∪ (Set.Icc (-2/3) (-1/2))}
def S₃ : Set ℝ := {x | x ∈ (Set.Iic (-3)) ∪ {-1} ∪ (Set.Icc 2 4)}

-- State the theorems
theorem inequality_solution_1 : {x : ℝ | f₁ x < 0} = S₁ := by sorry

theorem inequality_solution_2 : {x : ℝ | f₂ x ≤ 0} = S₂ := by sorry

theorem inequality_solution_3 : {x : ℝ | f₃ x ≤ 0} = S₃ := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_inequality_solution_3_l1661_166190


namespace NUMINAMATH_CALUDE_base_b_perfect_square_implies_b_greater_than_two_l1661_166113

/-- Represents a number in base b --/
def base_representation (b : ℕ) : ℕ := b^2 + 2*b + 1

/-- Checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem base_b_perfect_square_implies_b_greater_than_two :
  ∀ b : ℕ, is_perfect_square (base_representation b) → b > 2 :=
by sorry

end NUMINAMATH_CALUDE_base_b_perfect_square_implies_b_greater_than_two_l1661_166113


namespace NUMINAMATH_CALUDE_border_collie_catch_up_time_l1661_166140

/-- The time it takes for a border collie to catch up to a thrown ball -/
theorem border_collie_catch_up_time
  (ball_speed : ℝ)
  (ball_flight_time : ℝ)
  (dog_speed : ℝ)
  (h1 : ball_speed = 20)
  (h2 : ball_flight_time = 8)
  (h3 : dog_speed = 5) :
  (ball_speed * ball_flight_time) / dog_speed = 32 := by
  sorry

end NUMINAMATH_CALUDE_border_collie_catch_up_time_l1661_166140


namespace NUMINAMATH_CALUDE_even_product_probability_l1661_166193

-- Define the spinners
def spinner1 : List ℕ := [2, 4, 6, 8]
def spinner2 : List ℕ := [1, 3, 5, 7, 9]

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := n % 2 = 0

-- Define the probability function
def probabilityEvenProduct (s1 s2 : List ℕ) : ℚ :=
  let totalOutcomes := (s1.length * s2.length : ℚ)
  let evenOutcomes := (s1.filter isEven).length * s2.length
  evenOutcomes / totalOutcomes

-- Theorem statement
theorem even_product_probability :
  probabilityEvenProduct spinner1 spinner2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_product_probability_l1661_166193


namespace NUMINAMATH_CALUDE_line_passes_through_P_x_coordinate_range_l1661_166146

-- Define the line l
def line_l (θ : ℝ) (x y : ℝ) : Prop :=
  (Real.cos θ)^2 * x + Real.cos (2*θ) * y - 1 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Theorem 1: Line l always passes through point P
theorem line_passes_through_P :
  ∀ θ : ℝ, line_l θ (point_P.1) (point_P.2) :=
sorry

-- Define the range for x-coordinate of M
def x_range (x : ℝ) : Prop :=
  (2 - Real.sqrt 5) / 2 ≤ x ∧ x ≤ 4/5

-- Theorem 2: The x-coordinate of M is in the specified range
theorem x_coordinate_range :
  ∀ θ x y xm : ℝ,
  line_l θ x y →
  circle_C x y →
  -- Additional conditions for point M would be defined here
  x_range xm :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_x_coordinate_range_l1661_166146


namespace NUMINAMATH_CALUDE_initial_deadlift_weight_l1661_166141

def initial_squat : ℝ := 700
def initial_bench : ℝ := 400
def squat_loss_percentage : ℝ := 30
def deadlift_loss : ℝ := 200
def new_total : ℝ := 1490

theorem initial_deadlift_weight :
  ∃ (initial_deadlift : ℝ),
    initial_deadlift - deadlift_loss +
    initial_bench +
    initial_squat * (1 - squat_loss_percentage / 100) = new_total ∧
    initial_deadlift = 800 := by
  sorry

end NUMINAMATH_CALUDE_initial_deadlift_weight_l1661_166141


namespace NUMINAMATH_CALUDE_hyperbola_focus_range_l1661_166166

theorem hyperbola_focus_range (a b : ℝ) : 
  a > 0 → 
  b > 0 → 
  a^2 + b^2 = 16 → 
  b^2 ≥ 3 * a^2 → 
  0 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_range_l1661_166166


namespace NUMINAMATH_CALUDE_two_equal_intercept_lines_l1661_166168

/-- A line passing through (2, 3) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The intercept of the line on both axes -/
  intercept : ℝ
  /-- The line passes through (2, 3) -/
  passes_through : intercept - 2 = 3 * (intercept - intercept) / intercept

/-- There are exactly two lines passing through (2, 3) with equal intercepts on both axes -/
theorem two_equal_intercept_lines : 
  ∃! (s : Finset EqualInterceptLine), s.card = 2 ∧ 
  (∀ l : EqualInterceptLine, l ∈ s) ∧
  (∀ l : EqualInterceptLine, l ∈ s → l.intercept ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_two_equal_intercept_lines_l1661_166168


namespace NUMINAMATH_CALUDE_xiaolongs_dad_age_l1661_166161

theorem xiaolongs_dad_age (xiaolong_age : ℕ) : 
  xiaolong_age > 0 →
  (9 * xiaolong_age = 9 * xiaolong_age) →  -- Mom's age this year
  (9 * xiaolong_age + 3 = 9 * xiaolong_age + 3) →  -- Dad's age this year
  (9 * xiaolong_age + 4 = 8 * (xiaolong_age + 1)) →  -- Dad's age next year = 8 * Xiaolong's age next year
  9 * xiaolong_age + 3 = 39 := by
sorry

end NUMINAMATH_CALUDE_xiaolongs_dad_age_l1661_166161


namespace NUMINAMATH_CALUDE_solve_equation_l1661_166158

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 4
def g (x : ℝ) : ℝ := x^2 - 2

-- State the theorem
theorem solve_equation (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 18) : a = Real.sqrt (Real.sqrt 14 + 2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1661_166158


namespace NUMINAMATH_CALUDE_age_sum_proof_l1661_166138

-- Define the son's current age
def son_age : ℕ := 36

-- Define the father's current age
def father_age : ℕ := 72

-- Theorem stating the conditions and the result to prove
theorem age_sum_proof :
  -- 18 years ago, father was 3 times as old as son
  (father_age - 18 = 3 * (son_age - 18)) ∧
  -- Now, father is twice as old as son
  (father_age = 2 * son_age) →
  -- The sum of their present ages is 108
  son_age + father_age = 108 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l1661_166138


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1661_166116

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 4/y₀ = 1 ∧ x₀ + y₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1661_166116


namespace NUMINAMATH_CALUDE_simplify_expression_l1661_166183

theorem simplify_expression (x : ℝ) :
  3*x + 4*x^2 + 2 - (5 - 3*x - 5*x^2 + x^3) = -x^3 + 9*x^2 + 6*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1661_166183


namespace NUMINAMATH_CALUDE_consecutive_product_3024_l1661_166102

theorem consecutive_product_3024 :
  ∀ n : ℕ, n > 0 →
  (n * (n + 1) * (n + 2) * (n + 3) = 3024) ↔ n = 6 := by
sorry

end NUMINAMATH_CALUDE_consecutive_product_3024_l1661_166102


namespace NUMINAMATH_CALUDE_bisecting_line_sum_of_squares_l1661_166178

/-- A line with slope 4 that bisects a 3x3 unit square into two equal areas -/
def bisecting_line (a b c : ℝ) : Prop :=
  -- The line has slope 4
  a / b = 4 ∧
  -- The line equation is of the form ax = by + c
  ∀ x y, a * x = b * y + c ↔ y = 4 * x ∧
  -- The line bisects the square into two equal areas
  ∃ x₁ y₁ x₂ y₂, 
    0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 ∧
    0 ≤ y₁ ∧ y₁ < y₂ ∧ y₂ ≤ 3 ∧
    a * x₁ = b * y₁ + c ∧
    a * x₂ = b * y₂ + c ∧
    (3 * y₁ + (3 - y₂) * 3) / 2 = 9 / 2

theorem bisecting_line_sum_of_squares (a b c : ℝ) :
  bisecting_line a b c → a^2 + b^2 + c^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_of_squares_l1661_166178


namespace NUMINAMATH_CALUDE_distribute_6_3_l1661_166199

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 3 distinguishable boxes is 729 -/
theorem distribute_6_3 : distribute 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_3_l1661_166199


namespace NUMINAMATH_CALUDE_circle_radius_l1661_166187

theorem circle_radius (x y : ℝ) : 
  (∀ x y, x^2 - 6*x + y^2 + 2*y + 6 = 0) → 
  ∃ r : ℝ, r = 2 ∧ ∀ x y, (x - 3)^2 + (y + 1)^2 = r^2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l1661_166187


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_two_range_of_m_given_inequality_l1661_166191

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |m * x + 1| + |2 * x - 3|

-- Theorem 1
theorem range_of_x_when_m_is_two :
  ∀ x : ℝ, f 2 x = 4 ↔ -1/2 ≤ x ∧ x ≤ 3/2 := by sorry

-- Theorem 2
theorem range_of_m_given_inequality :
  (∀ a : ℝ, a > 0 → f m 1 ≤ (2 * a^2 + 8) / a) → -8 ≤ m ∧ m ≤ 6 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_two_range_of_m_given_inequality_l1661_166191


namespace NUMINAMATH_CALUDE_negative_four_cubed_equality_l1661_166150

theorem negative_four_cubed_equality : (-4)^3 = -4^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_cubed_equality_l1661_166150
