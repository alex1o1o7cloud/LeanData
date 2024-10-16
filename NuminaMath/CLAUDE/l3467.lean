import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l3467_346799

theorem equation_solution : 
  ∃ x : ℝ, |Real.sqrt (x^2 + 8*x + 20) + Real.sqrt (x^2 - 2*x + 2)| = Real.sqrt 26 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3467_346799


namespace NUMINAMATH_CALUDE_area_of_annular_region_area_of_specific_annular_region_l3467_346728

/-- The area of an annular region between two concentric circles -/
theorem area_of_annular_region (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > r₁) : 
  π * r₂^2 - π * r₁^2 = π * (r₂^2 - r₁^2) :=
by sorry

/-- The area of the annular region between two concentric circles with radii 4 and 7 is 33π -/
theorem area_of_specific_annular_region : 
  π * 7^2 - π * 4^2 = 33 * π :=
by sorry

end NUMINAMATH_CALUDE_area_of_annular_region_area_of_specific_annular_region_l3467_346728


namespace NUMINAMATH_CALUDE_complex_polygon_area_l3467_346772

/-- A complex polygon with specific properties -/
structure ComplexPolygon where
  sides : Nat
  side_length : ℝ
  perimeter : ℝ
  is_perpendicular : Bool
  is_congruent : Bool

/-- The area of the complex polygon -/
noncomputable def polygon_area (p : ComplexPolygon) : ℝ :=
  96

/-- Theorem stating the area of the specific complex polygon -/
theorem complex_polygon_area 
  (p : ComplexPolygon) 
  (h1 : p.sides = 32) 
  (h2 : p.perimeter = 64) 
  (h3 : p.is_perpendicular = true) 
  (h4 : p.is_congruent = true) : 
  polygon_area p = 96 := by
  sorry


end NUMINAMATH_CALUDE_complex_polygon_area_l3467_346772


namespace NUMINAMATH_CALUDE_binomial_expansion_result_l3467_346710

theorem binomial_expansion_result (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_result_l3467_346710


namespace NUMINAMATH_CALUDE_games_that_didnt_work_l3467_346794

/-- The number of games that didn't work, given Edward's game purchases and good games. -/
theorem games_that_didnt_work (friend_games garage_games good_games : ℕ) : 
  friend_games = 41 → garage_games = 14 → good_games = 24 → 
  friend_games + garage_games - good_games = 31 := by
  sorry

end NUMINAMATH_CALUDE_games_that_didnt_work_l3467_346794


namespace NUMINAMATH_CALUDE_circle_chord_problem_l3467_346707

-- Define the circle C
def circle_C (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*a*y + a^2 - 24 = 0

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop :=
  2*x - y = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop :=
  (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Theorem statement
theorem circle_chord_problem :
  ∃ (a : ℝ),
    (∀ x y, circle_C x y a → ∃ x₀ y₀, center_line x₀ y₀ ∧ (x - x₀)^2 + (y - y₀)^2 = 25) ∧
    a = 2 ∧
    (∀ m, ∃ chord_length,
      chord_length = Real.sqrt (4 * (25 - 5)) ∧
      (∀ x y, circle_C x y a ∧ line_l x y m →
        ∃ l, l ≤ chord_length ∧ l^2 = (x - 3)^2 + (y - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_problem_l3467_346707


namespace NUMINAMATH_CALUDE_food_percentage_is_twenty_percent_l3467_346770

/-- Represents the percentage of total amount spent on each category and their respective tax rates -/
structure ShoppingExpenses where
  clothing_percent : Real
  other_percent : Real
  clothing_tax : Real
  other_tax : Real
  total_tax : Real

/-- Calculates the percentage spent on food given the shopping expenses -/
def food_percent (e : ShoppingExpenses) : Real :=
  1 - e.clothing_percent - e.other_percent

/-- Calculates the total tax rate based on the expenses and tax rates -/
def total_tax_rate (e : ShoppingExpenses) : Real :=
  e.clothing_percent * e.clothing_tax + e.other_percent * e.other_tax

/-- Theorem stating that given the shopping conditions, the percentage spent on food is 20% -/
theorem food_percentage_is_twenty_percent (e : ShoppingExpenses) 
  (h1 : e.clothing_percent = 0.5)
  (h2 : e.other_percent = 0.3)
  (h3 : e.clothing_tax = 0.04)
  (h4 : e.other_tax = 0.1)
  (h5 : e.total_tax = 0.05)
  (h6 : total_tax_rate e = e.total_tax) :
  food_percent e = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_food_percentage_is_twenty_percent_l3467_346770


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l3467_346769

theorem no_real_roots_for_nonzero_k :
  ∀ k : ℝ, k ≠ 0 → ¬∃ x : ℝ, x^2 + k*x + 3*k^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l3467_346769


namespace NUMINAMATH_CALUDE_bus_passengers_l3467_346781

theorem bus_passengers (men women : ℕ) : 
  women = men / 3 →
  men - 24 = women + 12 →
  men + women = 72 :=
by sorry

end NUMINAMATH_CALUDE_bus_passengers_l3467_346781


namespace NUMINAMATH_CALUDE_modulo_six_equality_l3467_346785

theorem modulo_six_equality : 47^1860 - 25^1860 ≡ 0 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_modulo_six_equality_l3467_346785


namespace NUMINAMATH_CALUDE_jump_rope_record_rate_l3467_346766

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The record number of consecutive ropes jumped -/
def record_jumps : ℕ := 54000

/-- The time limit in hours -/
def time_limit : ℕ := 5

/-- The required rate of jumps per second -/
def required_rate : ℚ := 3

theorem jump_rope_record_rate :
  (record_jumps : ℚ) / ((time_limit * seconds_per_hour) : ℚ) = required_rate :=
sorry

end NUMINAMATH_CALUDE_jump_rope_record_rate_l3467_346766


namespace NUMINAMATH_CALUDE_total_spent_on_pens_l3467_346759

def brand_x_price : ℝ := 4.00
def brand_y_price : ℝ := 2.20
def total_pens : ℕ := 12
def brand_x_count : ℕ := 6

theorem total_spent_on_pens : 
  brand_x_count * brand_x_price + (total_pens - brand_x_count) * brand_y_price = 37.20 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_on_pens_l3467_346759


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3467_346771

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (x - 2) ≤ 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3467_346771


namespace NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l3467_346798

/-- Proves the percentage decrease in b when a increases by q% for inversely proportional variables -/
theorem inverse_proportion_percentage_change 
  (a b : ℝ) (q : ℝ) (c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : q > 0) :
  (a * b = c) →  -- inverse proportionality condition
  let a' := a * (1 + q / 100)  -- a increased by q%
  let b' := c / a'  -- new b value
  (b - b') / b * 100 = 100 * q / (100 + q) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l3467_346798


namespace NUMINAMATH_CALUDE_root_equation_q_value_l3467_346745

theorem root_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 2/b)^2 - p*(a + 2/b) + q = 0) →
  ((b + 2/a)^2 - p*(b + 2/a) + q = 0) →
  (q = 25/3) := by
sorry

end NUMINAMATH_CALUDE_root_equation_q_value_l3467_346745


namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l3467_346705

theorem students_in_both_band_and_chorus 
  (total_students : ℕ) 
  (band_students : ℕ) 
  (chorus_students : ℕ) 
  (band_or_chorus_students : ℕ) : 
  total_students = 200 →
  band_students = 70 →
  chorus_students = 95 →
  band_or_chorus_students = 150 →
  band_students + chorus_students - band_or_chorus_students = 15 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l3467_346705


namespace NUMINAMATH_CALUDE_discount_difference_l3467_346793

theorem discount_difference (bill : ℝ) (single_discount : ℝ) 
  (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) : 
  bill = 12000 ∧ 
  single_discount = 0.35 ∧ 
  discount1 = 0.25 ∧ 
  discount2 = 0.08 ∧ 
  discount3 = 0.02 → 
  bill * (1 - (1 - discount1) * (1 - discount2) * (1 - discount3)) - 
  bill * single_discount = 314.40 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l3467_346793


namespace NUMINAMATH_CALUDE_multiply_72519_by_9999_l3467_346758

theorem multiply_72519_by_9999 : 72519 * 9999 = 725117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_by_9999_l3467_346758


namespace NUMINAMATH_CALUDE_fruits_given_to_jane_l3467_346711

def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def initial_apples : ℕ := 21
def fruits_left : ℕ := 15

def total_initial_fruits : ℕ := initial_plums + initial_guavas + initial_apples

theorem fruits_given_to_jane : 
  total_initial_fruits - fruits_left = 40 := by sorry

end NUMINAMATH_CALUDE_fruits_given_to_jane_l3467_346711


namespace NUMINAMATH_CALUDE_meeting_speed_l3467_346783

theorem meeting_speed (total_distance : ℝ) (travel_time : ℝ) (speed_difference : ℝ) 
  (h1 : total_distance = 24)
  (h2 : travel_time = 3)
  (h3 : speed_difference = 2)
  (h4 : travel_time * (x + (x + speed_difference)) = total_distance) :
  x = 3 :=
by
  sorry

#check meeting_speed

end NUMINAMATH_CALUDE_meeting_speed_l3467_346783


namespace NUMINAMATH_CALUDE_pizza_pepperoni_ratio_l3467_346718

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_pepperoni : ℕ)

/-- Represents a slice of pizza -/
structure PizzaSlice :=
  (pepperoni : ℕ)

def cut_pizza (p : Pizza) (slice1_pepperoni : ℕ) : PizzaSlice × PizzaSlice :=
  let slice1 := PizzaSlice.mk slice1_pepperoni
  let slice2 := PizzaSlice.mk (p.total_pepperoni - slice1_pepperoni)
  (slice1, slice2)

def pepperoni_ratio (slice1 : PizzaSlice) (slice2 : PizzaSlice) : ℚ :=
  slice1.pepperoni / slice2.pepperoni

theorem pizza_pepperoni_ratio :
  let original_pizza := Pizza.mk 40
  let (jellys_slice, other_slice) := cut_pizza original_pizza 10
  let jellys_slice_after_loss := PizzaSlice.mk (jellys_slice.pepperoni - 1)
  pepperoni_ratio jellys_slice_after_loss other_slice = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pepperoni_ratio_l3467_346718


namespace NUMINAMATH_CALUDE_chess_players_per_game_l3467_346726

theorem chess_players_per_game (total_players : Nat) (total_games : Nat) (players_per_game : Nat) : 
  total_players = 8 → 
  total_games = 28 → 
  (total_players.choose players_per_game) = total_games → 
  players_per_game = 2 := by
sorry

end NUMINAMATH_CALUDE_chess_players_per_game_l3467_346726


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3467_346779

theorem simplify_and_evaluate : 
  let x : ℝ := -2
  (2 * x + 1) * (x - 2) - (2 - x)^2 = -4 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3467_346779


namespace NUMINAMATH_CALUDE_pie_sugar_percentage_l3467_346717

/-- Given a pie weighing 200 grams with 50 grams of sugar, 
    prove that 75% of the pie is not sugar. -/
theorem pie_sugar_percentage 
  (total_weight : ℝ) 
  (sugar_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : sugar_weight = 50) : 
  (total_weight - sugar_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pie_sugar_percentage_l3467_346717


namespace NUMINAMATH_CALUDE_system_solution_and_equality_l3467_346735

theorem system_solution_and_equality (a b c : ℝ) (h : a * b * c ≠ 0) :
  ∃! (x y z : ℝ),
    (b * z + c * y = a ∧ c * x + a * z = b ∧ a * y + b * x = c) ∧
    (x = (b^2 + c^2 - a^2) / (2 * b * c) ∧
     y = (c^2 + a^2 - b^2) / (2 * a * c) ∧
     z = (a^2 + b^2 - c^2) / (2 * a * b)) ∧
    ((1 - x^2) / a^2 = (1 - y^2) / b^2 ∧ (1 - y^2) / b^2 = (1 - z^2) / c^2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_and_equality_l3467_346735


namespace NUMINAMATH_CALUDE_age_ratio_constant_l3467_346774

/-- Given two people p and q, where the ratio of their present ages is 3:4 and their total age is 28,
    prove that p's age was always 3/4 of q's age at any point in the past. -/
theorem age_ratio_constant
  (p q : ℕ) -- present ages of p and q
  (h1 : p * 4 = q * 3) -- ratio of present ages is 3:4
  (h2 : p + q = 28) -- total present age is 28
  (t : ℕ) -- time in the past
  (h3 : t ≤ min p q) -- ensure t is not greater than either age
  : (p - t) * 4 = (q - t) * 3 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_constant_l3467_346774


namespace NUMINAMATH_CALUDE_choir_average_age_l3467_346765

theorem choir_average_age (num_females : ℕ) (num_males : ℕ) 
  (avg_age_females : ℚ) (avg_age_males : ℚ) :
  num_females = 12 →
  num_males = 18 →
  avg_age_females = 28 →
  avg_age_males = 38 →
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 34 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l3467_346765


namespace NUMINAMATH_CALUDE_apple_pie_count_l3467_346787

/-- Represents the number of pies of each type --/
structure PieOrder where
  peach : ℕ
  apple : ℕ
  blueberry : ℕ

/-- Represents the cost of fruit per pound for each type of pie --/
structure FruitCosts where
  peach : ℚ
  apple : ℚ
  blueberry : ℚ

/-- Calculates the total cost of fruit for a given pie order --/
def totalCost (order : PieOrder) (costs : FruitCosts) (poundsPerPie : ℕ) : ℚ :=
  (order.peach * costs.peach + order.apple * costs.apple + order.blueberry * costs.blueberry) * poundsPerPie

theorem apple_pie_count (order : PieOrder) (costs : FruitCosts) (poundsPerPie totalSpent : ℕ) :
  order.peach = 5 →
  order.blueberry = 3 →
  poundsPerPie = 3 →
  costs.peach = 2 →
  costs.apple = 1 →
  costs.blueberry = 1 →
  totalCost order costs poundsPerPie = totalSpent →
  order.apple = 4 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_count_l3467_346787


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3467_346713

/-- Given vectors a and b, find the unique value of t such that a is perpendicular to (t * a + b) -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (6, -4)) :
  ∃! t : ℝ, (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) ∧ t = -5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3467_346713


namespace NUMINAMATH_CALUDE_mass_of_Al2O3_solution_l3467_346729

-- Define the atomic masses
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_O : ℝ := 16.00

-- Define the volume and concentration of the solution
def volume : ℝ := 2.5
def concentration : ℝ := 4

-- Define the molecular weight of Al2O3
def molecular_weight_Al2O3 : ℝ := 2 * atomic_mass_Al + 3 * atomic_mass_O

-- State the theorem
theorem mass_of_Al2O3_solution :
  let moles : ℝ := volume * concentration
  let mass : ℝ := moles * molecular_weight_Al2O3
  mass = 1019.6 := by sorry

end NUMINAMATH_CALUDE_mass_of_Al2O3_solution_l3467_346729


namespace NUMINAMATH_CALUDE_equal_perimeters_l3467_346722

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

-- Define the quadrilateral ABCD
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry

-- Define the inscribed circle and its center I
def inscribedCircle : Circle := sorry
def I : Point := inscribedCircle.center

-- Define the circumcircle ω of triangle ACI
def ω : Circle := sorry

-- Define points X, Y, Z, T on ω
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry
def T : Point := sorry

-- Define a function to calculate the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a function to calculate the perimeter of a quadrilateral
def perimeter (p q r s : Point) : ℝ :=
  distance p q + distance q r + distance r s + distance s p

-- State the theorem
theorem equal_perimeters :
  perimeter A D T X = perimeter C D Y Z := by sorry

end NUMINAMATH_CALUDE_equal_perimeters_l3467_346722


namespace NUMINAMATH_CALUDE_function_decomposition_l3467_346782

-- Define a type for the domain that is symmetric with respect to the origin
structure SymmetricDomain where
  X : Type
  symm : X → X
  symm_involutive : ∀ x, symm (symm x) = x

-- Define a function on the symmetric domain
def Function (D : SymmetricDomain) := D.X → ℝ

-- Define an even function
def IsEven (D : SymmetricDomain) (f : Function D) : Prop :=
  ∀ x, f (D.symm x) = f x

-- Define an odd function
def IsOdd (D : SymmetricDomain) (f : Function D) : Prop :=
  ∀ x, f (D.symm x) = -f x

-- State the theorem
theorem function_decomposition (D : SymmetricDomain) (f : Function D) :
  ∃! (e o : Function D), (∀ x, f x = e x + o x) ∧ IsEven D e ∧ IsOdd D o := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l3467_346782


namespace NUMINAMATH_CALUDE_gunther_tractor_finance_l3467_346780

/-- Calculates the total amount financed for a loan with no interest -/
def total_financed (monthly_payment : ℕ) (payment_duration_years : ℕ) : ℕ :=
  monthly_payment * (payment_duration_years * 12)

/-- Theorem stating that for Gunther's tractor loan, the total financed amount is $9000 -/
theorem gunther_tractor_finance :
  total_financed 150 5 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_gunther_tractor_finance_l3467_346780


namespace NUMINAMATH_CALUDE_booklet_word_count_l3467_346721

theorem booklet_word_count (total_pages : Nat) (max_words_per_page : Nat) (remainder : Nat) (modulus : Nat) : 
  total_pages = 154 →
  max_words_per_page = 120 →
  remainder = 207 →
  modulus = 221 →
  ∃ (words_per_page : Nat),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % modulus = remainder ∧
    words_per_page = 100 := by
  sorry

end NUMINAMATH_CALUDE_booklet_word_count_l3467_346721


namespace NUMINAMATH_CALUDE_cat_groupings_count_l3467_346784

/-- The number of ways to divide 12 cats into groups of 4, 6, and 2,
    with Whiskers in the 4-cat group and Paws in the 6-cat group. -/
def cat_groupings : ℕ :=
  Nat.choose 10 3 * Nat.choose 7 5

theorem cat_groupings_count : cat_groupings = 2520 := by
  sorry

end NUMINAMATH_CALUDE_cat_groupings_count_l3467_346784


namespace NUMINAMATH_CALUDE_monic_polynomial_problem_l3467_346748

theorem monic_polynomial_problem (g : ℝ → ℝ) :
  (∃ b c : ℝ, ∀ x, g x = x^2 + b*x + c) →  -- g is a monic polynomial of degree 2
  g 0 = 6 →                               -- g(0) = 6
  g 1 = 12 →                              -- g(1) = 12
  ∀ x, g x = x^2 + 5*x + 6 :=              -- Conclusion: g(x) = x^2 + 5x + 6
by
  sorry

end NUMINAMATH_CALUDE_monic_polynomial_problem_l3467_346748


namespace NUMINAMATH_CALUDE_correct_operation_l3467_346752

theorem correct_operation (a b : ℝ) : -a^2*b + 2*a^2*b = a^2*b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3467_346752


namespace NUMINAMATH_CALUDE_min_cost_57_and_227_l3467_346786

/-- Calculates the minimum cost for notebooks given the pricing structure and number of notebooks -/
def min_cost (n : ℕ) : ℚ :=
  let single_price := 0.3
  let dozen_price := 3.0
  let bulk_dozen_price := 2.7
  let dozens := n / 12
  let singles := n % 12
  if dozens > 10 then
    bulk_dozen_price * dozens + single_price * singles
  else if singles = 0 then
    dozen_price * dozens
  else
    min (dozen_price * (dozens + 1)) (dozen_price * dozens + single_price * singles)

theorem min_cost_57_and_227 :
  min_cost 57 = 14.7 ∧ min_cost 227 = 51.3 := by sorry

end NUMINAMATH_CALUDE_min_cost_57_and_227_l3467_346786


namespace NUMINAMATH_CALUDE_max_rectangle_area_l3467_346737

theorem max_rectangle_area (perimeter : ℕ) (min_diff : ℕ) : perimeter = 160 → min_diff = 10 → ∃ (length width : ℕ), 
  length + width = perimeter / 2 ∧ 
  length ≥ width + min_diff ∧
  ∀ (l w : ℕ), l + w = perimeter / 2 → l ≥ w + min_diff → l * w ≤ length * width ∧
  length * width = 1575 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l3467_346737


namespace NUMINAMATH_CALUDE_perpendicular_iff_a_eq_one_l3467_346738

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (a x y : ℝ) : Prop := x - a * y = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, line1 x1 y1 ∧ line2 a x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    (x2 - x1) * (y2 - y1) = 0

-- State the theorem
theorem perpendicular_iff_a_eq_one :
  ∀ a : ℝ, perpendicular a ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_iff_a_eq_one_l3467_346738


namespace NUMINAMATH_CALUDE_triangle_height_sum_bound_l3467_346791

/-- For a triangle with side lengths a ≤ b ≤ c, heights h_a, h_b, h_c,
    semiperimeter p, and circumradius R, the sum of heights is bounded. -/
theorem triangle_height_sum_bound (a b c h_a h_b h_c p R : ℝ) :
  a ≤ b → b ≤ c → a > 0 → b > 0 → c > 0 →
  p = (a + b + c) / 2 →
  R > 0 →
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a*c + c^2)) / (4 * p * R) := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_sum_bound_l3467_346791


namespace NUMINAMATH_CALUDE_tournament_max_matches_l3467_346755

/-- Represents a round-robin tennis tournament -/
structure TennisTournament where
  players : ℕ
  original_days : ℕ
  rest_days : ℕ

/-- Calculates the maximum number of matches that can be completed in a tournament -/
def max_matches (t : TennisTournament) : ℕ :=
  min
    ((t.players * (t.players - 1)) / 2)
    ((t.players / 2) * (t.original_days - t.rest_days))

/-- Theorem: In a tournament with 10 players, 9 original days, and 1 rest day, 
    the maximum number of matches is 40 -/
theorem tournament_max_matches :
  let t : TennisTournament := ⟨10, 9, 1⟩
  max_matches t = 40 := by
  sorry


end NUMINAMATH_CALUDE_tournament_max_matches_l3467_346755


namespace NUMINAMATH_CALUDE_sum_of_possible_values_l3467_346788

theorem sum_of_possible_values (x y : ℝ) 
  (h : 2 * x * y - 2 * x / (y^2) - 2 * y / (x^2) = 4) : 
  ∃ (v₁ v₂ : ℝ), (x - 2) * (y - 2) = v₁ ∨ (x - 2) * (y - 2) = v₂ ∧ v₁ + v₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_possible_values_l3467_346788


namespace NUMINAMATH_CALUDE_smallest_with_sum_2011_has_224_digits_l3467_346789

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def numberOfDigits (n : ℕ) : ℕ := sorry

/-- The smallest natural number with a given sum of digits -/
def smallestWithSumOfDigits (s : ℕ) : ℕ := sorry

theorem smallest_with_sum_2011_has_224_digits :
  numberOfDigits (smallestWithSumOfDigits 2011) = 224 := by sorry

end NUMINAMATH_CALUDE_smallest_with_sum_2011_has_224_digits_l3467_346789


namespace NUMINAMATH_CALUDE_square_root_difference_equals_two_sqrt_three_l3467_346731

theorem square_root_difference_equals_two_sqrt_three :
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_equals_two_sqrt_three_l3467_346731


namespace NUMINAMATH_CALUDE_f_of_three_equals_nine_sevenths_l3467_346750

/-- Given f(x) = (2x + 3) / (4x - 5), prove that f(3) = 9/7 -/
theorem f_of_three_equals_nine_sevenths :
  let f : ℝ → ℝ := λ x ↦ (2*x + 3) / (4*x - 5)
  f 3 = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_f_of_three_equals_nine_sevenths_l3467_346750


namespace NUMINAMATH_CALUDE_count_non_adjacent_placements_correct_l3467_346732

/-- Represents an n × n grid board. -/
structure GridBoard where
  n : ℕ

/-- Counts the number of ways to place X and O on the grid such that they are not adjacent. -/
def countNonAdjacentPlacements (board : GridBoard) : ℕ :=
  board.n^4 - 3 * board.n^2 + 2 * board.n

/-- Theorem stating that countNonAdjacentPlacements gives the correct count. -/
theorem count_non_adjacent_placements_correct (board : GridBoard) :
  countNonAdjacentPlacements board =
    board.n^4 - 3 * board.n^2 + 2 * board.n :=
by sorry

end NUMINAMATH_CALUDE_count_non_adjacent_placements_correct_l3467_346732


namespace NUMINAMATH_CALUDE_root_equations_l3467_346701

/-- Given two constants c and d, prove that they satisfy the given conditions -/
theorem root_equations (c d : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (x + c) * (x + d) * (x + 8) = 0 ∧
    (y + c) * (y + d) * (y + 8) = 0 ∧
    (z + c) * (z + d) * (z + 8) = 0 ∧
    (x + 2) ≠ 0 ∧ (y + 2) ≠ 0 ∧ (z + 2) ≠ 0) ∧
  (∃! w : ℝ, (w + 3*c) * (w + 2) * (w + 4) = 0 ∧
    (w + d) ≠ 0 ∧ (w + 8) ≠ 0) →
  c = 2/3 ∧ d = 4 := by
sorry

end NUMINAMATH_CALUDE_root_equations_l3467_346701


namespace NUMINAMATH_CALUDE_new_student_weight_l3467_346724

/-- The weight of the new student given the conditions of the problem -/
theorem new_student_weight (n : ℕ) (initial_weight replaced_weight new_weight : ℝ) 
  (h1 : n = 4)
  (h2 : replaced_weight = 96)
  (h3 : (initial_weight - replaced_weight + new_weight) / n = initial_weight / n - 8) :
  new_weight = 64 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l3467_346724


namespace NUMINAMATH_CALUDE_min_omega_value_l3467_346703

theorem min_omega_value (ω : ℝ) (n : ℤ) : 
  ω > 0 ∧ (4 * π / 3 = n * (2 * π / ω)) → ω ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l3467_346703


namespace NUMINAMATH_CALUDE_kermit_final_positions_l3467_346763

/-- The number of integer coordinate pairs (x, y) satisfying |x| + |y| = n -/
def count_coordinate_pairs (n : ℕ) : ℕ :=
  2 * (n + 1) * (n + 1) - 2 * n * (n + 1) + 1

/-- Kermit's energy in Joules -/
def kermit_energy : ℕ := 100

theorem kermit_final_positions : 
  count_coordinate_pairs kermit_energy = 10201 :=
sorry

end NUMINAMATH_CALUDE_kermit_final_positions_l3467_346763


namespace NUMINAMATH_CALUDE_function_arithmetic_l3467_346796

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x < y → f x < f y

theorem function_arithmetic (f : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing f)
  (h_two : f 2 = 2)
  (h_coprime : ∀ m n : ℕ, Nat.Coprime m n → f (m * n) = f m * f n) :
  ∀ n : ℕ, f n = n :=
sorry

end NUMINAMATH_CALUDE_function_arithmetic_l3467_346796


namespace NUMINAMATH_CALUDE_a_less_than_one_necessary_not_sufficient_for_a_squared_less_than_one_l3467_346775

theorem a_less_than_one_necessary_not_sufficient_for_a_squared_less_than_one :
  ∃ (a : ℝ), (a^2 < 1 → a < 1) ∧ ¬(a < 1 → a^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_a_less_than_one_necessary_not_sufficient_for_a_squared_less_than_one_l3467_346775


namespace NUMINAMATH_CALUDE_range_of_m_l3467_346741

theorem range_of_m (m : ℝ) : 
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) → 
  (m ≥ 4 ∨ m ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3467_346741


namespace NUMINAMATH_CALUDE_dedekind_cut_B_dedekind_cut_D_l3467_346706

-- Define a Dedekind cut
def DedekindCut (M N : Set ℚ) : Prop :=
  (M ∪ N = Set.univ) ∧ 
  (M ∩ N = ∅) ∧ 
  (∀ x ∈ M, ∀ y ∈ N, x < y) ∧
  M.Nonempty ∧ 
  N.Nonempty

-- Statement B
theorem dedekind_cut_B : 
  ∃ M N : Set ℚ, DedekindCut M N ∧ 
  (¬∃ x, x = Sup M) ∧ 
  (∃ y, y = Inf N) :=
sorry

-- Statement D
theorem dedekind_cut_D : 
  ∃ M N : Set ℚ, DedekindCut M N ∧ 
  (¬∃ x, x = Sup M) ∧ 
  (¬∃ y, y = Inf N) :=
sorry

end NUMINAMATH_CALUDE_dedekind_cut_B_dedekind_cut_D_l3467_346706


namespace NUMINAMATH_CALUDE_supermarket_purchase_cost_l3467_346720

/-- Calculates the total cost of items with given quantities, prices, and discounts -/
def totalCost (quantities : List ℕ) (prices : List ℚ) (discounts : List ℚ) : ℚ :=
  List.sum (List.zipWith3 (fun q p d => q * p * (1 - d)) quantities prices discounts)

/-- The problem statement -/
theorem supermarket_purchase_cost : 
  let quantities : List ℕ := [24, 6, 5, 3]
  let prices : List ℚ := [9/5, 17/10, 17/5, 56/5]
  let discounts : List ℚ := [1/5, 1/5, 0, 1/10]
  totalCost quantities prices discounts = 4498/50
  := by sorry

end NUMINAMATH_CALUDE_supermarket_purchase_cost_l3467_346720


namespace NUMINAMATH_CALUDE_firefighter_remaining_money_is_2340_l3467_346760

/-- Calculates the remaining money for a firefighter after monthly expenses --/
def firefighter_remaining_money (hourly_rate : ℚ) (weekly_hours : ℚ) (food_expense : ℚ) (tax_expense : ℚ) : ℚ :=
  let weekly_earnings := hourly_rate * weekly_hours
  let monthly_earnings := weekly_earnings * 4
  let rent_expense := monthly_earnings / 3
  let total_expenses := rent_expense + food_expense + tax_expense
  monthly_earnings - total_expenses

/-- Theorem stating that the firefighter's remaining money is $2340 --/
theorem firefighter_remaining_money_is_2340 :
  firefighter_remaining_money 30 48 500 1000 = 2340 := by
  sorry

#eval firefighter_remaining_money 30 48 500 1000

end NUMINAMATH_CALUDE_firefighter_remaining_money_is_2340_l3467_346760


namespace NUMINAMATH_CALUDE_susie_bob_ratio_l3467_346795

-- Define the number of slices for each pizza size
def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8

-- Define the number of pizzas George purchased
def small_pizzas_bought : ℕ := 3
def large_pizzas_bought : ℕ := 2

-- Define the number of pieces eaten by each person
def george_pieces : ℕ := 3
def bob_pieces : ℕ := george_pieces + 1
def bill_fred_mark_pieces : ℕ := 3 * 3

-- Define the number of slices left over
def leftover_slices : ℕ := 10

-- Calculate the total number of slices
def total_slices : ℕ := small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought

-- Define Susie's pieces as a function of the other variables
def susie_pieces : ℕ := total_slices - leftover_slices - (george_pieces + bob_pieces + bill_fred_mark_pieces)

-- Theorem to prove
theorem susie_bob_ratio :
  susie_pieces * 2 = bob_pieces := by sorry

end NUMINAMATH_CALUDE_susie_bob_ratio_l3467_346795


namespace NUMINAMATH_CALUDE_phone_selling_price_l3467_346712

theorem phone_selling_price 
  (total_phones : ℕ) 
  (initial_investment : ℚ) 
  (profit_ratio : ℚ) :
  total_phones = 200 →
  initial_investment = 3000 →
  profit_ratio = 1/3 →
  (initial_investment + profit_ratio * initial_investment) / total_phones = 20 := by
sorry

end NUMINAMATH_CALUDE_phone_selling_price_l3467_346712


namespace NUMINAMATH_CALUDE_ngo_wage_problem_l3467_346767

/-- The NGO wage problem -/
theorem ngo_wage_problem (illiterate_count : ℕ) (literate_count : ℕ) 
  (initial_illiterate_wage : ℚ) (average_decrease : ℚ) :
  illiterate_count = 20 →
  literate_count = 10 →
  initial_illiterate_wage = 25 →
  average_decrease = 10 →
  ∃ (new_illiterate_wage : ℚ),
    new_illiterate_wage = 10 ∧
    illiterate_count * (initial_illiterate_wage - new_illiterate_wage) = 
      (illiterate_count + literate_count) * average_decrease :=
by sorry

end NUMINAMATH_CALUDE_ngo_wage_problem_l3467_346767


namespace NUMINAMATH_CALUDE_sum_of_p_and_q_l3467_346725

theorem sum_of_p_and_q (p q : ℝ) (h_distinct : p ≠ q) (h_greater : p > q) :
  let M := !![2, -5, 8; 1, p, q; 1, q, p]
  Matrix.det M = 0 → p + q = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_p_and_q_l3467_346725


namespace NUMINAMATH_CALUDE_f_zero_points_iff_k_range_l3467_346742

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1/2) ^ x

def has_three_zero_points (k : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ x, f k (f k x) - 3/2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem f_zero_points_iff_k_range :
  ∀ k, has_three_zero_points k ↔ -1/2 < k ∧ k ≤ -1/4 :=
sorry

end NUMINAMATH_CALUDE_f_zero_points_iff_k_range_l3467_346742


namespace NUMINAMATH_CALUDE_profit_percentage_proof_l3467_346768

/-- Given that the cost price of 20 articles equals the selling price of 16 articles,
    prove that the profit percentage is 25%. -/
theorem profit_percentage_proof (C S : ℝ) (h : 20 * C = 16 * S) :
  (S - C) / C * 100 = 25 :=
sorry

end NUMINAMATH_CALUDE_profit_percentage_proof_l3467_346768


namespace NUMINAMATH_CALUDE_landscape_breadth_l3467_346739

/-- Proves that the breadth of a rectangular landscape is 480 meters given the specified conditions -/
theorem landscape_breadth :
  ∀ (length breadth : ℝ),
  breadth = 8 * length →
  3200 = (1 / 9) * (length * breadth) →
  breadth = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_landscape_breadth_l3467_346739


namespace NUMINAMATH_CALUDE_contrapositive_statement_1_contrapositive_statement_2_negation_statement_3_sufficient_condition_statement_4_l3467_346727

-- Statement 1
theorem contrapositive_statement_1 :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
sorry

-- Statement 2
theorem contrapositive_statement_2 :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

-- Statement 3
theorem negation_statement_3 :
  ¬(∃ x₀ : ℝ, x₀ > 1 ∧ x₀^2 - 2*x₀ - 3 = 0) ↔
  (∀ x : ℝ, x > 1 → x^2 - 2*x - 3 ≠ 0) :=
sorry

-- Statement 4
theorem sufficient_condition_statement_4 (a : ℝ) :
  (∀ x : ℝ, -2 < x ∧ x < -1 → (x + a)*(x + 1) < 0) →
  a > 2 :=
sorry

end NUMINAMATH_CALUDE_contrapositive_statement_1_contrapositive_statement_2_negation_statement_3_sufficient_condition_statement_4_l3467_346727


namespace NUMINAMATH_CALUDE_calculation_proof_l3467_346733

theorem calculation_proof : (2468 * 629) / (1234 * 37) = 34 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_l3467_346733


namespace NUMINAMATH_CALUDE_subtraction_result_l3467_346709

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  valid : hundreds < 10 ∧ tens < 10 ∧ units < 10

/-- The result of subtracting two three-digit numbers -/
def subtract (a b : ThreeDigitNumber) : ThreeDigitNumber :=
  sorry

theorem subtraction_result 
  (a b : ThreeDigitNumber)
  (h_units : a.units = b.units + 6)
  (h_result_units : (subtract a b).units = 5)
  (h_result_tens : (subtract a b).tens = 9)
  (h_no_borrow : a.tens ≥ b.tens) :
  (subtract a b).hundreds = 4 :=
sorry

end NUMINAMATH_CALUDE_subtraction_result_l3467_346709


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l3467_346778

/-- The speed of a man rowing a boat in still water, given the speed of the stream
    and the time taken to row a certain distance downstream. -/
theorem mans_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 8)
  (h2 : downstream_distance = 90)
  (h3 : downstream_time = 5)
  : ∃ (mans_speed : ℝ), mans_speed = 10 ∧ 
    (mans_speed + stream_speed) * downstream_time = downstream_distance :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l3467_346778


namespace NUMINAMATH_CALUDE_orange_crayon_boxes_l3467_346743

theorem orange_crayon_boxes (total_crayons : ℕ) 
  (orange_per_box blue_boxes blue_per_box red_boxes red_per_box : ℕ) : 
  total_crayons = 94 →
  orange_per_box = 8 →
  blue_boxes = 7 →
  blue_per_box = 5 →
  red_boxes = 1 →
  red_per_box = 11 →
  (total_crayons - (blue_boxes * blue_per_box + red_boxes * red_per_box)) / orange_per_box = 6 :=
by
  sorry

#check orange_crayon_boxes

end NUMINAMATH_CALUDE_orange_crayon_boxes_l3467_346743


namespace NUMINAMATH_CALUDE_omega_function_iff_strictly_increasing_l3467_346723

def OmegaFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

theorem omega_function_iff_strictly_increasing (f : ℝ → ℝ) :
  OmegaFunction f ↔ StrictMono f := by sorry

end NUMINAMATH_CALUDE_omega_function_iff_strictly_increasing_l3467_346723


namespace NUMINAMATH_CALUDE_profit_achievement_l3467_346761

/-- The number of pens in a pack -/
def pens_per_pack : ℕ := 4

/-- The cost of a pack of pens in dollars -/
def pack_cost : ℚ := 7

/-- The number of pens sold at the given rate -/
def pens_sold_rate : ℕ := 5

/-- The price for the number of pens sold at the given rate in dollars -/
def price_sold_rate : ℚ := 12

/-- The target profit in dollars -/
def target_profit : ℚ := 50

/-- The minimum number of pens needed to be sold to achieve the target profit -/
def min_pens_to_sell : ℕ := 77

theorem profit_achievement :
  ∃ (n : ℕ), n ≥ min_pens_to_sell ∧
  (n : ℚ) * (price_sold_rate / pens_sold_rate) - 
  (n : ℚ) * (pack_cost / pens_per_pack) ≥ target_profit ∧
  ∀ (m : ℕ), m < min_pens_to_sell →
  (m : ℚ) * (price_sold_rate / pens_sold_rate) - 
  (m : ℚ) * (pack_cost / pens_per_pack) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_achievement_l3467_346761


namespace NUMINAMATH_CALUDE_range_of_a_l3467_346704

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3467_346704


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3467_346751

/-- Given that:
    1. X's current age is 45
    2. Three years ago, X's age was some multiple of Y's age
    3. Seven years from now, the sum of their ages will be 83 years
    Prove that the ratio of X's age to Y's age three years ago is 2:1 -/
theorem age_ratio_problem (x_current y_current : ℕ) : 
  x_current = 45 →
  ∃ k : ℕ, k > 0 ∧ (x_current - 3) = k * (y_current - 3) →
  x_current + y_current + 14 = 83 →
  (x_current - 3) / (y_current - 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3467_346751


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_problem_solution_l3467_346744

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)
def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := n * a₁ + n * (n - 1) * d / 2

theorem arithmetic_sequence_sum (a₁ : ℤ) :
  ∃ d : ℤ, 
    (sum_arithmetic_sequence a₁ d 6 - 2 * sum_arithmetic_sequence a₁ d 3 = 18) → 
    (sum_arithmetic_sequence a₁ d 2017 = 2017) := by
  sorry

-- Main theorem
theorem problem_solution : 
  ∃ d : ℤ, 
    (sum_arithmetic_sequence (-2015) d 6 - 2 * sum_arithmetic_sequence (-2015) d 3 = 18) → 
    (sum_arithmetic_sequence (-2015) d 2017 = 2017) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_problem_solution_l3467_346744


namespace NUMINAMATH_CALUDE_power_two_ge_product_l3467_346736

theorem power_two_ge_product (m n : ℕ) : 2^(m+n-2) ≥ m*n := by
  sorry

end NUMINAMATH_CALUDE_power_two_ge_product_l3467_346736


namespace NUMINAMATH_CALUDE_coefficient_expansion_l3467_346753

theorem coefficient_expansion (a : ℝ) : 
  (∃ c : ℝ, c = 9 ∧ c = 1 + 4 * a) → a = 2 := by sorry

end NUMINAMATH_CALUDE_coefficient_expansion_l3467_346753


namespace NUMINAMATH_CALUDE_sheersCost_is_40_l3467_346746

/-- The cost of window treatments for a house with 3 windows, where each window
    requires a pair of sheers and a pair of drapes. -/
def WindowTreatmentsCost (sheersCost : ℚ) : ℚ :=
  3 * (sheersCost + 60)

/-- Theorem stating that the cost of a pair of sheers is $40, given the conditions. -/
theorem sheersCost_is_40 :
  ∃ (sheersCost : ℚ), WindowTreatmentsCost sheersCost = 300 ∧ sheersCost = 40 :=
sorry

end NUMINAMATH_CALUDE_sheersCost_is_40_l3467_346746


namespace NUMINAMATH_CALUDE_largest_common_divisor_462_330_l3467_346747

theorem largest_common_divisor_462_330 : Nat.gcd 462 330 = 66 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_462_330_l3467_346747


namespace NUMINAMATH_CALUDE_maisy_earnings_difference_l3467_346792

/-- Represents Maisy's job details -/
structure Job where
  hours : ℕ
  wage : ℕ
  bonus : ℕ

/-- Calculates the weekly earnings for a job -/
def weekly_earnings (job : Job) : ℕ :=
  job.hours * job.wage + job.bonus

/-- Theorem: Maisy earns $15 more per week at her new job -/
theorem maisy_earnings_difference :
  let current_job : Job := ⟨8, 10, 0⟩
  let new_job : Job := ⟨4, 15, 35⟩
  weekly_earnings new_job - weekly_earnings current_job = 15 :=
by sorry

end NUMINAMATH_CALUDE_maisy_earnings_difference_l3467_346792


namespace NUMINAMATH_CALUDE_lines_equivalence_l3467_346764

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A cylinder in 3D space -/
structure Cylinder3D where
  axis : Line3D
  radius : ℝ

/-- The set of lines passing through a point and at a given distance from another line -/
def linesAtDistanceFromLine (M : Point3D) (d : ℝ) (AB : Line3D) : Set Line3D :=
  sorry

/-- The set of lines lying in two planes tangent to a cylinder passing through a point -/
def linesInTangentPlanes (M : Point3D) (cylinder : Cylinder3D) : Set Line3D :=
  sorry

/-- Theorem stating the equivalence of the two sets of lines -/
theorem lines_equivalence (M : Point3D) (d : ℝ) (AB : Line3D) :
  let cylinder := Cylinder3D.mk AB d
  linesAtDistanceFromLine M d AB = linesInTangentPlanes M cylinder :=
sorry

end NUMINAMATH_CALUDE_lines_equivalence_l3467_346764


namespace NUMINAMATH_CALUDE_sine_equation_solution_l3467_346730

theorem sine_equation_solution (x : ℝ) (h1 : 3 * Real.sin (2 * x) = 2 * Real.sin x) 
  (h2 : 0 < x ∧ x < π) : x = Real.arccos (1/3) := by
  sorry

end NUMINAMATH_CALUDE_sine_equation_solution_l3467_346730


namespace NUMINAMATH_CALUDE_average_visitors_is_288_l3467_346740

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def average_visitors_per_day (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℚ :=
  let num_sundays : ℕ := 30 / 7
  let num_other_days : ℕ := 30 - num_sundays
  let total_visitors : ℕ := sunday_visitors * num_sundays + other_day_visitors * num_other_days
  (total_visitors : ℚ) / 30

/-- Theorem stating that the average number of visitors per day is 288 -/
theorem average_visitors_is_288 :
  average_visitors_per_day 600 240 = 288 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_is_288_l3467_346740


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3467_346714

theorem cubic_root_sum (a b c : ℝ) : 
  (a^3 = a + 1) → (b^3 = b + 1) → (c^3 = c + 1) →
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3467_346714


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3467_346790

theorem fahrenheit_to_celsius (C F : ℝ) : C = (4 / 7) * (F - 40) → C = 35 → F = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3467_346790


namespace NUMINAMATH_CALUDE_band_to_orchestra_ratio_l3467_346702

/-- The number of male musicians in the orchestra -/
def orchestra_males : ℕ := 11

/-- The number of female musicians in the orchestra -/
def orchestra_females : ℕ := 12

/-- The number of male musicians in the choir -/
def choir_males : ℕ := 12

/-- The number of female musicians in the choir -/
def choir_females : ℕ := 17

/-- The total number of musicians in all groups -/
def total_musicians : ℕ := 98

/-- The number of musicians in the orchestra -/
def orchestra_total : ℕ := orchestra_males + orchestra_females

/-- The number of musicians in the choir -/
def choir_total : ℕ := choir_males + choir_females

theorem band_to_orchestra_ratio :
  ∃ (band_musicians : ℕ),
    band_musicians = 2 * orchestra_total ∧
    orchestra_total + band_musicians + choir_total = total_musicians :=
by sorry

end NUMINAMATH_CALUDE_band_to_orchestra_ratio_l3467_346702


namespace NUMINAMATH_CALUDE_base_2_representation_of_101_l3467_346700

theorem base_2_representation_of_101 : 
  ∃ (a b c d e f g : Nat), 
    (a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 0 ∧ e = 1 ∧ f = 0 ∧ g = 1) ∧
    101 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_101_l3467_346700


namespace NUMINAMATH_CALUDE_min_value_expression_l3467_346749

theorem min_value_expression (a b c : ℕ+) :
  ∃ (x y z : ℕ+), 
    (⌊(8 * (x + y) : ℚ) / z⌋ + ⌊(8 * (x + z) : ℚ) / y⌋ + ⌊(8 * (y + z) : ℚ) / x⌋ = 46) ∧
    ∀ (a b c : ℕ+), 
      ⌊(8 * (a + b) : ℚ) / c⌋ + ⌊(8 * (a + c) : ℚ) / b⌋ + ⌊(8 * (b + c) : ℚ) / a⌋ ≥ 46 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3467_346749


namespace NUMINAMATH_CALUDE_dinner_bill_contribution_l3467_346734

theorem dinner_bill_contribution (num_friends : ℕ) 
  (num_18_meals num_24_meals num_30_meals : ℕ)
  (cost_18_meal cost_24_meal cost_30_meal : ℚ)
  (num_appetizers : ℕ) (cost_appetizer : ℚ)
  (tip_percentage : ℚ)
  (h1 : num_friends = 8)
  (h2 : num_18_meals = 4)
  (h3 : num_24_meals = 2)
  (h4 : num_30_meals = 2)
  (h5 : cost_18_meal = 18)
  (h6 : cost_24_meal = 24)
  (h7 : cost_30_meal = 30)
  (h8 : num_appetizers = 3)
  (h9 : cost_appetizer = 12)
  (h10 : tip_percentage = 12 / 100) :
  let total_cost := num_18_meals * cost_18_meal + 
                    num_24_meals * cost_24_meal + 
                    num_30_meals * cost_30_meal + 
                    num_appetizers * cost_appetizer
  let total_with_tip := total_cost + total_cost * tip_percentage
  let contribution_per_person := total_with_tip / num_friends
  contribution_per_person = 30.24 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_contribution_l3467_346734


namespace NUMINAMATH_CALUDE_bowling_ball_volume_l3467_346715

/-- The remaining volume of a bowling ball after drilling holes -/
theorem bowling_ball_volume (π : ℝ) (h : π > 0) : 
  let sphere_volume := (4/3) * π * (12^3)
  let small_hole_volume := π * (3/2)^2 * 10
  let large_hole_volume := π * 2^2 * 10
  sphere_volume - (2 * small_hole_volume + large_hole_volume) = 2219 * π := by
  sorry

#check bowling_ball_volume

end NUMINAMATH_CALUDE_bowling_ball_volume_l3467_346715


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l3467_346777

theorem absolute_value_sum_zero (a b : ℝ) :
  |3 + a| + |b - 2| = 0 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l3467_346777


namespace NUMINAMATH_CALUDE_min_type_c_cards_l3467_346797

/-- Represents the number of cards sold of each type -/
structure CardSales where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total number of cards sold -/
def total_cards (sales : CardSales) : ℕ :=
  sales.a + sales.b + sales.c

/-- Calculates the total income from card sales -/
def total_income (sales : CardSales) : ℚ :=
  0.5 * sales.a + 1 * sales.b + 2.5 * sales.c

/-- Theorem stating the minimum number of type C cards sold -/
theorem min_type_c_cards (sales : CardSales) 
  (h1 : total_cards sales = 150)
  (h2 : total_income sales = 180) :
  sales.c ≥ 20 := by
  sorry

#check min_type_c_cards

end NUMINAMATH_CALUDE_min_type_c_cards_l3467_346797


namespace NUMINAMATH_CALUDE_common_divisors_9240_10080_l3467_346776

theorem common_divisors_9240_10080 : Nat.card {d : ℕ | d ∣ 9240 ∧ d ∣ 10080} = 48 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_10080_l3467_346776


namespace NUMINAMATH_CALUDE_eggs_bought_l3467_346754

def initial_eggs : ℕ := 98
def final_eggs : ℕ := 106

theorem eggs_bought : final_eggs - initial_eggs = 8 := by
  sorry

end NUMINAMATH_CALUDE_eggs_bought_l3467_346754


namespace NUMINAMATH_CALUDE_probability_not_adjacent_l3467_346762

-- Define the total number of chairs
def total_chairs : ℕ := 10

-- Define the number of available chairs (excluding first and last)
def available_chairs : ℕ := total_chairs - 2

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the number of adjacent pairs in the available chairs
def adjacent_pairs : ℕ := available_chairs - 1

-- Theorem statement
theorem probability_not_adjacent :
  (1 : ℚ) - (adjacent_pairs : ℚ) / (choose available_chairs 2) = 3/4 :=
sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l3467_346762


namespace NUMINAMATH_CALUDE_jerrys_shelf_books_l3467_346757

/-- The number of books on Jerry's shelf -/
def books : ℕ := 9

/-- The initial number of action figures -/
def initial_figures : ℕ := 5

/-- The number of action figures added -/
def added_figures : ℕ := 7

/-- The difference between action figures and books -/
def figure_book_difference : ℕ := 3

theorem jerrys_shelf_books :
  books = initial_figures + added_figures - figure_book_difference := by
  sorry

end NUMINAMATH_CALUDE_jerrys_shelf_books_l3467_346757


namespace NUMINAMATH_CALUDE_average_b_c_is_70_l3467_346756

/-- Given two numbers a and b with an average of 50, and a third number c such that c - a = 40,
    prove that the average of b and c is 70. -/
theorem average_b_c_is_70 (a b c : ℝ) 
    (h1 : (a + b) / 2 = 50)
    (h2 : c - a = 40) : 
  (b + c) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_b_c_is_70_l3467_346756


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l3467_346773

/-- Represents a parabola in the form y = (x - h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { h := 0, k := 0 }

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (units : ℝ) : Parabola :=
  { h := p.h - units, k := p.k }

/-- The equation of a parabola in terms of x -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

theorem shifted_parabola_equation :
  let shifted := shift_parabola original_parabola 2
  ∀ x, parabola_equation shifted x = (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l3467_346773


namespace NUMINAMATH_CALUDE_fraction_simplification_l3467_346719

theorem fraction_simplification (a b x : ℝ) 
  (h1 : x = b / a) 
  (h2 : a ≠ b) 
  (h3 : a ≠ 0) : 
  (2 * a + b) / (2 * a - b) = (2 + x) / (2 - x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3467_346719


namespace NUMINAMATH_CALUDE_system_solution_unique_l3467_346716

theorem system_solution_unique :
  ∃! (x y : ℝ), 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3467_346716


namespace NUMINAMATH_CALUDE_shopping_cart_deletion_l3467_346708

theorem shopping_cart_deletion (initial_items final_items : ℕ) 
  (h1 : initial_items = 18) 
  (h2 : final_items = 8) : 
  initial_items - final_items = 10 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cart_deletion_l3467_346708
