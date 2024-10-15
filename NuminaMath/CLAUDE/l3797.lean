import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_nonnegative_iff_equal_roots_l3797_379738

theorem polynomial_nonnegative_iff_equal_roots (a b c : ℝ) :
  (∀ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) ≥ 0) ↔ 
  (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_iff_equal_roots_l3797_379738


namespace NUMINAMATH_CALUDE_server_performance_l3797_379770

/-- Represents the number of multiplications a server can perform per second -/
def multiplications_per_second : ℕ := 5000

/-- Represents the number of seconds in half an hour -/
def seconds_in_half_hour : ℕ := 1800

/-- Represents the total number of multiplications in half an hour -/
def total_multiplications : ℕ := multiplications_per_second * seconds_in_half_hour

/-- Theorem stating that the server performs 9 million multiplications in half an hour -/
theorem server_performance : total_multiplications = 9000000 := by
  sorry

end NUMINAMATH_CALUDE_server_performance_l3797_379770


namespace NUMINAMATH_CALUDE_total_games_calculation_l3797_379767

/-- The number of football games in one month -/
def games_per_month : ℝ := 323.0

/-- The number of months in a season -/
def season_duration : ℝ := 17.0

/-- The total number of football games in a season -/
def total_games : ℝ := games_per_month * season_duration

theorem total_games_calculation :
  total_games = 5491.0 := by sorry

end NUMINAMATH_CALUDE_total_games_calculation_l3797_379767


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3797_379720

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2 ≥ 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x | x ≤ -Real.sqrt 2 ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3797_379720


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3797_379775

/-- The perimeter of a rectangle with length 15 inches and width 8 inches is 46 inches. -/
theorem rectangle_perimeter : 
  ∀ (length width perimeter : ℕ), 
  length = 15 → 
  width = 8 → 
  perimeter = 2 * (length + width) → 
  perimeter = 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3797_379775


namespace NUMINAMATH_CALUDE_bottle_production_l3797_379757

/-- Given that 6 identical machines produce 240 bottles per minute at a constant rate,
    prove that 10 such machines will produce 1600 bottles in 4 minutes. -/
theorem bottle_production
  (machines : ℕ)
  (bottles_per_minute : ℕ)
  (h1 : machines = 6)
  (h2 : bottles_per_minute = 240)
  (constant_rate : ℕ → ℕ → ℕ) -- Function to calculate production based on number of machines and time
  (h3 : constant_rate machines 1 = bottles_per_minute) -- Production rate for given machines in 1 minute
  : constant_rate 10 4 = 1600 := by
  sorry


end NUMINAMATH_CALUDE_bottle_production_l3797_379757


namespace NUMINAMATH_CALUDE_tangent_line_constraint_l3797_379703

theorem tangent_line_constraint (a : ℝ) : 
  (∀ b : ℝ, ¬∃ x : ℝ, (x^3 - 3*a*x + x = b ∧ 3*x^2 - 3*a = -1)) → 
  a < 1/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_constraint_l3797_379703


namespace NUMINAMATH_CALUDE_translation_increases_y_l3797_379702

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a horizontal translation of a function -/
structure HorizontalTranslation where
  units : ℝ

/-- The original quadratic function y = -x^2 + 1 -/
def original_function : QuadraticFunction :=
  { a := -1, b := 0, c := 1 }

/-- The required translation -/
def translation : HorizontalTranslation :=
  { units := 2 }

/-- Theorem stating that the given translation makes y increase as x increases when x < 2 -/
theorem translation_increases_y (f : QuadraticFunction) (t : HorizontalTranslation) :
  f = original_function →
  t = translation →
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 2 →
  f.a * (x₁ - t.units)^2 + f.b * (x₁ - t.units) + f.c <
  f.a * (x₂ - t.units)^2 + f.b * (x₂ - t.units) + f.c :=
by sorry

end NUMINAMATH_CALUDE_translation_increases_y_l3797_379702


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3797_379742

-- Problem 1
theorem problem_1 (x y : ℝ) : (x + y)^2 + x * (x - 2*y) = 2*x^2 + y^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x ≠ 2 ∧ x ≠ 0) : 
  (x^2 - 6*x + 9) / (x - 2) / (x + 2 - (3*x - 4) / (x - 2)) = (x - 3) / x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3797_379742


namespace NUMINAMATH_CALUDE_part_one_part_two_l3797_379771

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0

def q (m x : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Part I
theorem part_one (m : ℝ) : 
  (m > 0 ∧ (∀ x, ¬(q m x) → ¬(p x))) ↔ (0 < m ∧ m ≤ 2) :=
sorry

-- Part II
theorem part_two (x : ℝ) :
  ((p x ∨ q 7 x) ∧ ¬(p x ∧ q 7 x)) ↔ ((-6 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x < 8)) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3797_379771


namespace NUMINAMATH_CALUDE_inequality_solution_set_min_perimeter_rectangle_min_perimeter_achieved_l3797_379793

-- Problem 1: Inequality solution set
theorem inequality_solution_set (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 3 ↔ x * (2 * x - 3) - 6 ≤ x := by sorry

-- Problem 2: Minimum perimeter of rectangle
theorem min_perimeter_rectangle (l w : ℝ) (h_area : l * w = 16) (h_positive : l > 0 ∧ w > 0) :
  2 * (l + w) ≥ 16 := by sorry

theorem min_perimeter_achieved (l w : ℝ) (h_area : l * w = 16) (h_positive : l > 0 ∧ w > 0) :
  2 * (l + w) = 16 ↔ l = 4 ∧ w = 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_min_perimeter_rectangle_min_perimeter_achieved_l3797_379793


namespace NUMINAMATH_CALUDE_tan_alpha_problem_l3797_379792

theorem tan_alpha_problem (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin (2*α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_problem_l3797_379792


namespace NUMINAMATH_CALUDE_line_segment_point_sum_l3797_379743

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := y = -5/6 * x + 10

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (12, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 10)

/-- Point T is on the line segment PQ -/
def T : ℝ × ℝ → Prop
  | (r, s) => ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- Area of triangle POQ -/
def area_POQ : ℝ := 60

/-- Area of triangle TOP -/
def area_TOP : ℝ := 15

/-- Theorem: If the given conditions are met, then r + s = 11.5 -/
theorem line_segment_point_sum (r s : ℝ) : 
  line_eq r s → T (r, s) → area_POQ = 4 * area_TOP → r + s = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_point_sum_l3797_379743


namespace NUMINAMATH_CALUDE_circle_center_l3797_379739

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 4*y = 16

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 36

theorem circle_center :
  CircleCenter (-4) 2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l3797_379739


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3797_379773

theorem rectangle_area_change (initial_area : ℝ) (length_increase : ℝ) (breadth_decrease : ℝ) :
  initial_area = 150 →
  length_increase = 37.5 →
  breadth_decrease = 18.2 →
  let new_length_factor := 1 + length_increase / 100
  let new_breadth_factor := 1 - breadth_decrease / 100
  let new_area := initial_area * new_length_factor * new_breadth_factor
  ∃ ε > 0, |new_area - 168.825| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3797_379773


namespace NUMINAMATH_CALUDE_unsuitable_temp_l3797_379747

def storage_temp := -18
def temp_range := 2

def is_suitable_temp (temp : Int) : Prop :=
  (storage_temp - temp_range) ≤ temp ∧ temp ≤ (storage_temp + temp_range)

theorem unsuitable_temp :
  ¬(is_suitable_temp (-21)) :=
by
  sorry

end NUMINAMATH_CALUDE_unsuitable_temp_l3797_379747


namespace NUMINAMATH_CALUDE_base_number_proof_l3797_379784

/-- 
Given a real number x, if (x^4 * 3.456789)^12 has 24 digits to the right of the decimal place 
when written as a single term, then x = 10^12.
-/
theorem base_number_proof (x : ℝ) : 
  (∃ n : ℕ, (x^4 * 3.456789)^12 * 10^24 = n) → x = 10^12 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l3797_379784


namespace NUMINAMATH_CALUDE_sqrt_equation_condition_l3797_379797

theorem sqrt_equation_condition (x y : ℝ) : 
  Real.sqrt (3 * x^2 + y^2) = 2 * x + y ↔ x * (x + 4 * y) = 0 ∧ 2 * x + y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_condition_l3797_379797


namespace NUMINAMATH_CALUDE_min_value_expression_l3797_379760

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y > 2 * x) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (z : ℝ), z = (y^2 - 2*x*y + x^2) / (x*y - 2*x^2) → z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3797_379760


namespace NUMINAMATH_CALUDE_numeric_hex_count_l3797_379763

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Converts a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : List HexDigit :=
  sorry

/-- Checks if a hexadecimal representation contains only numeric digits --/
def containsOnlyNumeric (hex : List HexDigit) : Bool :=
  sorry

/-- Counts the number of integers up to n with only numeric hexadecimal digits --/
def countNumericHex (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem numeric_hex_count :
  countNumericHex 500 = 199 :=
sorry

end NUMINAMATH_CALUDE_numeric_hex_count_l3797_379763


namespace NUMINAMATH_CALUDE_glazed_doughnut_cost_l3797_379778

/-- Proves that the cost of each glazed doughnut is $1 given the conditions of the problem -/
theorem glazed_doughnut_cost :
  let total_students : ℕ := 25
  let chocolate_lovers : ℕ := 10
  let glazed_lovers : ℕ := 15
  let chocolate_cost : ℚ := 2
  let total_cost : ℚ := 35
  chocolate_lovers + glazed_lovers = total_students →
  chocolate_lovers * chocolate_cost + glazed_lovers * (total_cost - chocolate_lovers * chocolate_cost) / glazed_lovers = total_cost →
  (total_cost - chocolate_lovers * chocolate_cost) / glazed_lovers = 1 := by
sorry

end NUMINAMATH_CALUDE_glazed_doughnut_cost_l3797_379778


namespace NUMINAMATH_CALUDE_parcel_cost_formula_l3797_379774

def parcel_cost (W : ℕ) : ℕ :=
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10

theorem parcel_cost_formula (W : ℕ) :
  (W ≤ 10 → parcel_cost W = 5 * W + 10) ∧
  (W > 10 → parcel_cost W = 7 * W - 10) := by
  sorry

end NUMINAMATH_CALUDE_parcel_cost_formula_l3797_379774


namespace NUMINAMATH_CALUDE_amount_after_two_years_l3797_379762

theorem amount_after_two_years (initial_amount : ℝ) : 
  initial_amount = 6400 →
  (initial_amount * (81 / 64) : ℝ) = 8100 := by
sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l3797_379762


namespace NUMINAMATH_CALUDE_original_price_from_loss_and_selling_price_l3797_379709

/-- Proves that if an item is sold at a 20% loss for 600 currency units, 
    then its original price was 750 currency units. -/
theorem original_price_from_loss_and_selling_price 
  (selling_price : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : selling_price = 600) 
  (h2 : loss_percentage = 20) : 
  ∃ original_price : ℝ, 
    original_price = 750 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_loss_and_selling_price_l3797_379709


namespace NUMINAMATH_CALUDE_expected_cost_is_3500_l3797_379787

/-- The number of machines -/
def total_machines : ℕ := 5

/-- The number of faulty machines -/
def faulty_machines : ℕ := 2

/-- The cost of testing one machine in yuan -/
def cost_per_test : ℕ := 1000

/-- The possible outcomes of the number of tests needed -/
def possible_tests : List ℕ := [2, 3, 4]

/-- The probabilities corresponding to each outcome -/
def probabilities : List ℚ := [1/10, 3/10, 3/5]

/-- The expected cost of testing in yuan -/
def expected_cost : ℚ := 3500

/-- Theorem stating that the expected cost of testing is 3500 yuan -/
theorem expected_cost_is_3500 :
  (List.sum (List.zipWith (· * ·) (List.map (λ n => n * cost_per_test) possible_tests) probabilities) : ℚ) = expected_cost :=
sorry

end NUMINAMATH_CALUDE_expected_cost_is_3500_l3797_379787


namespace NUMINAMATH_CALUDE_range_of_positive_f_l3797_379706

/-- A function is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_positive_f 
  (f : ℝ → ℝ) 
  (f' : ℝ → ℝ) 
  (hf_odd : OddFunction f)
  (hf_deriv : ∀ x, HasDerivAt f (f' x) x)
  (hf_neg_one : f (-1) = 0)
  (hf_pos : ∀ x > 0, x * f' x - f x > 0) :
  {x | f x > 0} = Set.Ioo (-1) 0 ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_positive_f_l3797_379706


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3797_379752

theorem smaller_number_problem (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : min a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3797_379752


namespace NUMINAMATH_CALUDE_kendra_remaining_words_l3797_379783

/-- Theorem: Given Kendra's goal of learning 60 new words and having already learned 36 words,
    she needs to learn 24 more words to reach her goal. -/
theorem kendra_remaining_words (total_goal : ℕ) (learned : ℕ) (remaining : ℕ) :
  total_goal = 60 →
  learned = 36 →
  remaining = total_goal - learned →
  remaining = 24 :=
by sorry

end NUMINAMATH_CALUDE_kendra_remaining_words_l3797_379783


namespace NUMINAMATH_CALUDE_meaningful_fraction_condition_l3797_379721

theorem meaningful_fraction_condition (x : ℝ) :
  (∃ y, y = (x - 1) / (x + 1)) ↔ x ≠ -1 :=
sorry

end NUMINAMATH_CALUDE_meaningful_fraction_condition_l3797_379721


namespace NUMINAMATH_CALUDE_differential_equation_solution_l3797_379761

/-- The general solution to the differential equation dr - r dφ = 0 -/
theorem differential_equation_solution (r φ : ℝ → ℝ) (C : ℝ) :
  (∀ t, (deriv r t) - r t * (deriv φ t) = 0) ↔
  ∃ C, C > 0 ∧ ∀ t, r t = C * Real.exp (φ t) :=
sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l3797_379761


namespace NUMINAMATH_CALUDE_binomial_fraction_value_l3797_379776

theorem binomial_fraction_value : 
  (Nat.choose 1 2023 * 3^2023) / Nat.choose 4046 2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_binomial_fraction_value_l3797_379776


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_3994001_l3797_379716

theorem sqrt_product_plus_one_equals_3994001 :
  Real.sqrt (1997 * 1998 * 1999 * 2000 + 1) = 3994001 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_3994001_l3797_379716


namespace NUMINAMATH_CALUDE_intersection_locus_l3797_379746

theorem intersection_locus (m : ℝ) (x y : ℝ) : 
  (m * x - y + 1 = 0 ∧ x - m * y - 1 = 0) → 
  (x - y = 0 ∨ x - y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_locus_l3797_379746


namespace NUMINAMATH_CALUDE_each_person_share_l3797_379734

/-- The cost to send a person to Mars in billions of dollars -/
def mars_cost : ℚ := 30

/-- The cost to establish a base on the Moon in billions of dollars -/
def moon_base_cost : ℚ := 10

/-- The number of people sharing the cost in millions -/
def number_of_people : ℚ := 200

/-- The total cost in billions of dollars -/
def total_cost : ℚ := mars_cost + moon_base_cost

/-- Theorem: Each person's share of the total cost is $200 -/
theorem each_person_share :
  (total_cost * 1000) / number_of_people = 200 := by sorry

end NUMINAMATH_CALUDE_each_person_share_l3797_379734


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3797_379764

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- First circle: x^2 + y^2 = 4 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Second circle: (x-3)^2 + (y-4)^2 = 9 -/
def circle2 (x y : ℝ) : Prop := (x-3)^2 + (y-4)^2 = 9

theorem circles_externally_tangent :
  externally_tangent (0, 0) (3, 4) 2 3 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3797_379764


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l3797_379737

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 5

theorem quadratic_function_m_range (a : ℝ) (m : ℝ) :
  (∀ t, f a t = f a (-4 - t)) →
  (∀ x ∈ Set.Icc m 0, f a x ≤ 5) →
  (∃ x ∈ Set.Icc m 0, f a x = 5) →
  (∀ x ∈ Set.Icc m 0, f a x ≥ 1) →
  (∃ x ∈ Set.Icc m 0, f a x = 1) →
  -4 ≤ m ∧ m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l3797_379737


namespace NUMINAMATH_CALUDE_dance_group_size_l3797_379712

theorem dance_group_size (calligraphy_group : ℕ) (dance_group : ℕ) : 
  calligraphy_group = 28 → 
  calligraphy_group = 2 * dance_group + 6 → 
  dance_group = 11 := by
sorry

end NUMINAMATH_CALUDE_dance_group_size_l3797_379712


namespace NUMINAMATH_CALUDE_brunchCombinationsCount_l3797_379782

/-- The number of ways to choose one item from a set of 3, two different items from a set of 4, 
    and one item from another set of 3, where the order of selection doesn't matter. -/
def brunchCombinations : ℕ :=
  3 * (Nat.choose 4 2) * 3

/-- Theorem stating that the number of brunch combinations is 54. -/
theorem brunchCombinationsCount : brunchCombinations = 54 := by
  sorry

end NUMINAMATH_CALUDE_brunchCombinationsCount_l3797_379782


namespace NUMINAMATH_CALUDE_milk_needed_for_recipe_l3797_379707

-- Define the ratio of milk to flour
def milk_to_flour_ratio : ℚ := 75 / 250

-- Define the amount of flour Luca wants to use
def flour_amount : ℚ := 1250

-- Theorem: The amount of milk needed for 1250 mL of flour is 375 mL
theorem milk_needed_for_recipe : 
  milk_to_flour_ratio * flour_amount = 375 := by
  sorry


end NUMINAMATH_CALUDE_milk_needed_for_recipe_l3797_379707


namespace NUMINAMATH_CALUDE_number_divided_by_constant_l3797_379748

theorem number_divided_by_constant (x : ℝ) : x / 0.06 = 16.666666666666668 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_constant_l3797_379748


namespace NUMINAMATH_CALUDE_call_center_team_b_fraction_l3797_379758

/-- Represents the fraction of calls processed by Team B given the relative
    call processing rates and team sizes of two teams in a call center. -/
theorem call_center_team_b_fraction :
  -- Each member of Team A processes 6/5 calls compared to Team B
  ∀ (call_rate_a call_rate_b : ℚ),
  call_rate_a = 6 / 5 * call_rate_b →
  -- Team A has 5/8 as many agents as Team B
  ∀ (team_size_a team_size_b : ℚ),
  team_size_a = 5 / 8 * team_size_b →
  -- The fraction of calls processed by Team B
  (team_size_b * call_rate_b) /
    (team_size_a * call_rate_a + team_size_b * call_rate_b) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_call_center_team_b_fraction_l3797_379758


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3797_379728

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  first_term : a 1 = 1
  third_term : a 3 = -3
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Theorem about the general formula and sum of the sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 3 - 2 * n) ∧
  (∃ k : ℕ, k * (seq.a 1 + seq.a k) / 2 = -35 ∧ k = 7) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3797_379728


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3797_379745

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) → 
  l * w = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3797_379745


namespace NUMINAMATH_CALUDE_eighteen_wheeler_toll_l3797_379759

/-- Calculate the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  1.50 + 0.50 * (axles - 2)

/-- Calculate the number of axles for a truck given the total number of wheels -/
def axles_count (wheels : ℕ) : ℕ :=
  wheels / 2

theorem eighteen_wheeler_toll :
  let wheels : ℕ := 18
  let axles : ℕ := axles_count wheels
  toll axles = 5 := by sorry

end NUMINAMATH_CALUDE_eighteen_wheeler_toll_l3797_379759


namespace NUMINAMATH_CALUDE_x_twelfth_power_l3797_379736

theorem x_twelfth_power (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l3797_379736


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3797_379791

/-- The quadratic function f(x) = -x^2 + 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

/-- y₁ is the value of f at x = -2 -/
def y₁ : ℝ := f (-2)

/-- y₂ is the value of f at x = 2 -/
def y₂ : ℝ := f 2

/-- y₃ is the value of f at x = -4 -/
def y₃ : ℝ := f (-4)

/-- Theorem: For the quadratic function f(x) = -x^2 + 2x + 3,
    if f(-2) = y₁, f(2) = y₂, and f(-4) = y₃, then y₂ > y₁ > y₃ -/
theorem quadratic_inequality : y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3797_379791


namespace NUMINAMATH_CALUDE_equation_solution_l3797_379769

theorem equation_solution (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3797_379769


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l3797_379798

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem composition_of_even_is_even (f : ℝ → ℝ) (hf : IsEven f) : IsEven (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l3797_379798


namespace NUMINAMATH_CALUDE_football_result_unique_solution_l3797_379766

/-- Represents the result of a football team's performance -/
structure FootballResult where
  total_matches : ℕ
  lost_matches : ℕ
  total_points : ℕ
  wins : ℕ
  draws : ℕ

/-- Checks if a FootballResult is valid according to the given rules -/
def is_valid_result (r : FootballResult) : Prop :=
  r.total_matches = r.wins + r.draws + r.lost_matches ∧
  r.total_points = 3 * r.wins + r.draws

/-- Theorem stating the unique solution for the given problem -/
theorem football_result_unique_solution :
  ∃! (r : FootballResult),
    r.total_matches = 15 ∧
    r.lost_matches = 4 ∧
    r.total_points = 29 ∧
    is_valid_result r ∧
    r.wins = 9 ∧
    r.draws = 2 := by
  sorry

end NUMINAMATH_CALUDE_football_result_unique_solution_l3797_379766


namespace NUMINAMATH_CALUDE_abs_equation_unique_solution_l3797_379744

theorem abs_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x - 3| :=
by
  sorry

end NUMINAMATH_CALUDE_abs_equation_unique_solution_l3797_379744


namespace NUMINAMATH_CALUDE_base_8_6_equivalence_l3797_379714

theorem base_8_6_equivalence :
  ∀ (n : ℕ), n > 0 →
  (∃ (C D : ℕ),
    C < 8 ∧ D < 8 ∧
    D < 6 ∧
    n = 8 * C + D ∧
    n = 6 * D + C) →
  n = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_8_6_equivalence_l3797_379714


namespace NUMINAMATH_CALUDE_base9_multiplication_l3797_379756

/-- Represents a number in base 9 --/
def Base9 : Type := ℕ

/-- Converts a base 9 number to a natural number --/
def to_nat (x : Base9) : ℕ := sorry

/-- Converts a natural number to a base 9 number --/
def from_nat (n : ℕ) : Base9 := sorry

/-- Multiplication operation for Base9 numbers --/
def mul_base9 (x y : Base9) : Base9 := sorry

theorem base9_multiplication :
  mul_base9 (from_nat 362) (from_nat 7) = from_nat 2875 :=
sorry

end NUMINAMATH_CALUDE_base9_multiplication_l3797_379756


namespace NUMINAMATH_CALUDE_power_of_three_equals_square_minus_sixteen_l3797_379732

theorem power_of_three_equals_square_minus_sixteen (a n : ℕ+) :
  (3 : ℕ) ^ (n : ℕ) = (a : ℕ) ^ 2 - 16 ↔ a = 5 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equals_square_minus_sixteen_l3797_379732


namespace NUMINAMATH_CALUDE_number_problem_l3797_379755

theorem number_problem : ∃ x : ℝ, 3 * (2 * x + 8) = 84 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3797_379755


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l3797_379799

def initial_amount : ℚ := 180

def sandwich_fraction : ℚ := 1 / 5
def museum_fraction : ℚ := 1 / 6
def book_fraction : ℚ := 1 / 2

def remaining_amount : ℚ := initial_amount * (1 - sandwich_fraction - museum_fraction - book_fraction)

theorem jennifer_remaining_money : remaining_amount = 24 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l3797_379799


namespace NUMINAMATH_CALUDE_anna_money_left_l3797_379725

def original_amount : ℚ := 32
def spent_fraction : ℚ := 1/4

theorem anna_money_left : 
  (1 - spent_fraction) * original_amount = 24 := by
  sorry

end NUMINAMATH_CALUDE_anna_money_left_l3797_379725


namespace NUMINAMATH_CALUDE_expanded_polynomial_terms_count_l3797_379705

theorem expanded_polynomial_terms_count : 
  let factor1 := 4  -- number of terms in (a₁ + a₂ + a₃ + a₄)
  let factor2 := 2  -- number of terms in (b₁ + b₂)
  let factor3 := 3  -- number of terms in (c₁ + c₂ + c₃)
  factor1 * factor2 * factor3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_expanded_polynomial_terms_count_l3797_379705


namespace NUMINAMATH_CALUDE_initial_blue_balls_l3797_379717

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 15 → 
  removed = 3 → 
  prob = 1/3 → 
  (total - removed : ℚ) * prob = (total - removed - (total - removed - prob * (total - removed))) → 
  total - removed - (total - removed - prob * (total - removed)) + removed = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_blue_balls_l3797_379717


namespace NUMINAMATH_CALUDE_ramu_car_profit_percent_l3797_379789

/-- Calculates the profit percent from a car sale -/
def profit_percent (purchase_price repair_cost selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percent for Ramu's car sale is approximately 41.30% -/
theorem ramu_car_profit_percent :
  let purchase_price : ℚ := 34000
  let repair_cost : ℚ := 12000
  let selling_price : ℚ := 65000
  abs (profit_percent purchase_price repair_cost selling_price - 41.30) < 0.01 := by
  sorry

#eval profit_percent 34000 12000 65000

end NUMINAMATH_CALUDE_ramu_car_profit_percent_l3797_379789


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_5_pow_1000_without_zero_l3797_379718

theorem exists_number_divisible_by_5_pow_1000_without_zero : ∃ n : ℕ, 
  (5^1000 ∣ n) ∧ 
  (∀ d : ℕ, d < 10 → (n.digits 10).all (λ digit => digit ≠ d) → d ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_5_pow_1000_without_zero_l3797_379718


namespace NUMINAMATH_CALUDE_percentage_problem_l3797_379740

theorem percentage_problem (P : ℝ) : P = 25 ↔ 0.15 * 40 = (P / 100) * 16 + 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3797_379740


namespace NUMINAMATH_CALUDE_periodic_binomial_remainder_l3797_379704

theorem periodic_binomial_remainder (K : ℕ+) : 
  (∃ (p : ℕ+), ∀ (n : ℕ), n ≥ p → 
    (∃ (T : ℕ+), ∀ (m : ℕ), m ≥ p → 
      (Nat.choose (2*(n+m)) (n+m)) % K = (Nat.choose (2*n) n) % K)) ↔ 
  (K = 1 ∨ K = 2) :=
sorry

end NUMINAMATH_CALUDE_periodic_binomial_remainder_l3797_379704


namespace NUMINAMATH_CALUDE_equation_solution_l3797_379735

theorem equation_solution : ∃! y : ℚ, (1 / 6 : ℚ) + 6 / y = 14 / y + (1 / 14 : ℚ) ∧ y = 84 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3797_379735


namespace NUMINAMATH_CALUDE_expression_values_l3797_379777

theorem expression_values (x y : ℝ) (h1 : x + y = 2) (h2 : y > 0) (h3 : x ≠ 0) :
  (1 / |x| + |x| / (y + 2) = 3/4) ∨ (1 / |x| + |x| / (y + 2) = 5/4) :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l3797_379777


namespace NUMINAMATH_CALUDE_ferns_total_cost_l3797_379796

/-- Calculates the total cost of Fern's purchase --/
def calculate_total_cost (high_heels_price : ℝ) (ballet_slippers_ratio : ℝ) 
  (ballet_slippers_count : ℕ) (purse_price : ℝ) (scarf_price : ℝ) 
  (high_heels_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let ballet_slippers_price := high_heels_price * ballet_slippers_ratio
  let total_ballet_slippers := ballet_slippers_price * ballet_slippers_count
  let discounted_high_heels := high_heels_price * (1 - high_heels_discount)
  let subtotal := discounted_high_heels + total_ballet_slippers + purse_price + scarf_price
  subtotal * (1 + sales_tax)

/-- Theorem stating that Fern's total cost is $348.30 --/
theorem ferns_total_cost : 
  calculate_total_cost 60 (2/3) 5 45 25 0.1 0.075 = 348.30 := by
  sorry

end NUMINAMATH_CALUDE_ferns_total_cost_l3797_379796


namespace NUMINAMATH_CALUDE_burgers_spending_l3797_379795

def total_allowance : ℚ := 50

def movie_fraction : ℚ := 2 / 5
def video_game_fraction : ℚ := 1 / 10
def book_fraction : ℚ := 1 / 4

def spent_on_movies : ℚ := movie_fraction * total_allowance
def spent_on_video_games : ℚ := video_game_fraction * total_allowance
def spent_on_books : ℚ := book_fraction * total_allowance

def total_spent : ℚ := spent_on_movies + spent_on_video_games + spent_on_books

def remaining_for_burgers : ℚ := total_allowance - total_spent

theorem burgers_spending :
  remaining_for_burgers = 12.5 := by sorry

end NUMINAMATH_CALUDE_burgers_spending_l3797_379795


namespace NUMINAMATH_CALUDE_inscribed_pentagon_external_angles_sum_inscribed_pentagon_external_angles_sum_is_720_l3797_379749

/-- Represents a pentagon inscribed in a circle -/
structure InscribedPentagon where
  -- We don't need to define the specific properties of the pentagon,
  -- as the problem doesn't require detailed information about its structure

/-- 
Theorem: For a pentagon inscribed in a circle, the sum of the angles
inscribed in the five segments outside the pentagon but inside the circle
is equal to 720°.
-/
theorem inscribed_pentagon_external_angles_sum
  (p : InscribedPentagon) : Real :=
  720

/-- 
Main theorem: The sum of the angles inscribed in the five segments
outside an inscribed pentagon but inside the circle is 720°.
-/
theorem inscribed_pentagon_external_angles_sum_is_720
  (p : InscribedPentagon) :
  inscribed_pentagon_external_angles_sum p = 720 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_pentagon_external_angles_sum_inscribed_pentagon_external_angles_sum_is_720_l3797_379749


namespace NUMINAMATH_CALUDE_min_sum_of_product_1176_l3797_379726

theorem min_sum_of_product_1176 (a b c : ℕ+) (h : a * b * c = 1176) :
  (∀ x y z : ℕ+, x * y * z = 1176 → a + b + c ≤ x + y + z) →
  a + b + c = 59 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1176_l3797_379726


namespace NUMINAMATH_CALUDE_square_sum_eighteen_l3797_379708

theorem square_sum_eighteen (x y : ℝ) 
  (h1 : y + 9 = (x - 3)^3)
  (h2 : x + 9 = (y - 3)^3)
  (h3 : x ≠ y) : 
  x^2 + y^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eighteen_l3797_379708


namespace NUMINAMATH_CALUDE_house_length_calculation_l3797_379754

/-- Given a house with width 10 feet and a porch measuring 6 feet by 4.5 feet,
    if 232 square feet of shingles are needed to roof both the house and the porch,
    then the length of the house is 20.5 feet. -/
theorem house_length_calculation (house_width porch_length porch_width total_shingle_area : ℝ) :
  house_width = 10 →
  porch_length = 6 →
  porch_width = 4.5 →
  total_shingle_area = 232 →
  ∃ house_length : ℝ,
    house_length * house_width + porch_length * porch_width = total_shingle_area ∧
    house_length = 20.5 :=
by sorry

end NUMINAMATH_CALUDE_house_length_calculation_l3797_379754


namespace NUMINAMATH_CALUDE_total_passengers_taking_l3797_379701

/-- Represents a train type with its characteristics -/
structure TrainType where
  interval : ℕ  -- Arrival interval in minutes
  leaving : ℕ   -- Number of passengers leaving
  taking : ℕ    -- Number of passengers taking

/-- Calculates the number of trains per hour given the arrival interval -/
def trainsPerHour (interval : ℕ) : ℕ := 60 / interval

/-- Calculates the total passengers for a given operation (leaving or taking) per hour -/
def totalPassengers (t : TrainType) (op : TrainType → ℕ) : ℕ :=
  (trainsPerHour t.interval) * (op t)

/-- Theorem: The total number of unique passengers taking trains at each station during an hour is 4360 -/
theorem total_passengers_taking (stationCount : ℕ) (type1 type2 type3 : TrainType) :
  stationCount = 4 →
  type1 = { interval := 10, leaving := 200, taking := 320 } →
  type2 = { interval := 15, leaving := 300, taking := 400 } →
  type3 = { interval := 20, leaving := 150, taking := 280 } →
  (totalPassengers type1 TrainType.taking +
   totalPassengers type2 TrainType.taking +
   totalPassengers type3 TrainType.taking) = 4360 :=
by sorry

end NUMINAMATH_CALUDE_total_passengers_taking_l3797_379701


namespace NUMINAMATH_CALUDE_units_digit_of_50_factorial_l3797_379786

theorem units_digit_of_50_factorial (n : ℕ) : n = 50 → (n.factorial % 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_50_factorial_l3797_379786


namespace NUMINAMATH_CALUDE_largest_difference_l3797_379768

def U : ℕ := 2 * 2010^2011
def V : ℕ := 2010^2011
def W : ℕ := 2009 * 2010^2010
def X : ℕ := 2 * 2010^2010
def Y : ℕ := 2010^2010
def Z : ℕ := 2010^2009

theorem largest_difference : 
  (U - V > V - W) ∧ 
  (U - V > W - X + 100) ∧ 
  (U - V > X - Y) ∧ 
  (U - V > Y - Z) := by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l3797_379768


namespace NUMINAMATH_CALUDE_subset_0_2_is_5th_subset_211_is_01467_l3797_379788

/-- The set E with 10 elements -/
def E : Finset ℕ := Finset.range 10

/-- Function to calculate the k value for a given subset -/
def kValue (subset : Finset ℕ) : ℕ :=
  subset.sum (fun i => 2^i)

/-- The first theorem: {0, 2} (representing {a₁, a₃}) corresponds to k = 5 -/
theorem subset_0_2_is_5th : kValue {0, 2} = 5 := by sorry

/-- The second theorem: k = 211 corresponds to the subset {0, 1, 4, 6, 7} 
    (representing {a₁, a₂, a₅, a₇, a₈}) -/
theorem subset_211_is_01467 : 
  (Finset.filter (fun i => (211 / 2^i) % 2 = 1) E) = {0, 1, 4, 6, 7} := by sorry

end NUMINAMATH_CALUDE_subset_0_2_is_5th_subset_211_is_01467_l3797_379788


namespace NUMINAMATH_CALUDE_max_value_theorem_equality_conditions_l3797_379741

theorem max_value_theorem (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) ≤ 1 :=
sorry

theorem equality_conditions (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) = 1 ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1)) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_equality_conditions_l3797_379741


namespace NUMINAMATH_CALUDE_fraction_inequality_l3797_379722

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  c / a > c / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3797_379722


namespace NUMINAMATH_CALUDE_function_supremum_m_range_l3797_379781

/-- The supremum of the given function for positive real x and y is 25/4 -/
theorem function_supremum : 
  (∀ x y : ℝ, x > 0 → y > 0 → 
    (4*x^4 + 17*x^2*y + 4*y^2) / (x^4 + 2*x^2*y + y^2) ≤ 25/4) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    (4*x^4 + 17*x^2*y + 4*y^2) / (x^4 + 2*x^2*y + y^2) = 25/4) :=
by sorry

/-- The range of m for which the inequality always holds is (25, +∞) -/
theorem m_range (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 
    (4*x^4 + 17*x^2*y + 4*y^2) / (x^4 + 2*x^2*y + y^2) < m/4) ↔ 
  m > 25 :=
by sorry

end NUMINAMATH_CALUDE_function_supremum_m_range_l3797_379781


namespace NUMINAMATH_CALUDE_replacement_process_terminates_l3797_379790

/-- Represents a finite sequence of binary digits (0 or 1) -/
def BinarySequence := List Nat

/-- The operation that replaces "01" with "1000" in a binary sequence -/
def replace_operation (seq : BinarySequence) : BinarySequence :=
  sorry

/-- Predicate to check if a sequence contains the subsequence "01" -/
def has_replaceable_subsequence (seq : BinarySequence) : Prop :=
  sorry

/-- The number of ones in a binary sequence -/
def count_ones (seq : BinarySequence) : Nat :=
  sorry

theorem replacement_process_terminates (initial_seq : BinarySequence) :
  ∃ (n : Nat), ∀ (m : Nat), m ≥ n →
    ¬(has_replaceable_subsequence ((replace_operation^[m]) initial_seq)) :=
  sorry

end NUMINAMATH_CALUDE_replacement_process_terminates_l3797_379790


namespace NUMINAMATH_CALUDE_leftover_apples_for_ivan_l3797_379750

/-- Given a number of initial apples and mini pies, calculate the number of leftover apples -/
def leftover_apples (initial_apples : ℕ) (mini_pies : ℕ) : ℕ :=
  initial_apples - (mini_pies / 2)

/-- Theorem: Given 48 initial apples and 24 mini pies, each requiring 1/2 an apple, 
    the number of leftover apples is 36 -/
theorem leftover_apples_for_ivan : leftover_apples 48 24 = 36 := by
  sorry

end NUMINAMATH_CALUDE_leftover_apples_for_ivan_l3797_379750


namespace NUMINAMATH_CALUDE_last_digit_of_large_power_l3797_379724

theorem last_digit_of_large_power : ∃ (n1 n2 n3 : ℕ), 
  n1 = 99^9 ∧ 
  n2 = 999^n1 ∧ 
  n3 = 9999^n2 ∧ 
  99999^n3 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_power_l3797_379724


namespace NUMINAMATH_CALUDE_franks_breakfast_shopping_l3797_379719

/-- The cost of a bottle of milk in Frank's breakfast shopping -/
def milk_cost : ℝ := 2.5

/-- The cost of 10 buns -/
def buns_cost : ℝ := 1

/-- The number of bottles of milk Frank bought -/
def milk_bottles : ℕ := 1

/-- The cost of the carton of eggs -/
def eggs_cost : ℝ := 3 * milk_cost

/-- The total cost of Frank's breakfast shopping -/
def total_cost : ℝ := 11

theorem franks_breakfast_shopping :
  buns_cost + milk_bottles * milk_cost + eggs_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_franks_breakfast_shopping_l3797_379719


namespace NUMINAMATH_CALUDE_path_cost_calculation_l3797_379700

/-- Calculates the total cost of building paths around a rectangular plot -/
def calculate_path_cost (plot_length : Real) (plot_width : Real) 
                        (gravel_path_width : Real) (concrete_path_width : Real)
                        (gravel_cost_per_sqm : Real) (concrete_cost_per_sqm : Real) : Real :=
  let gravel_path_area := 2 * plot_length * gravel_path_width
  let concrete_path_area := 2 * plot_width * concrete_path_width
  let gravel_cost := gravel_path_area * gravel_cost_per_sqm
  let concrete_cost := concrete_path_area * concrete_cost_per_sqm
  gravel_cost + concrete_cost

/-- Theorem stating that the total cost of building the paths is approximately Rs. 9.78 -/
theorem path_cost_calculation :
  let plot_length := 120
  let plot_width := 0.85
  let gravel_path_width := 0.05
  let concrete_path_width := 0.07
  let gravel_cost_per_sqm := 0.80
  let concrete_cost_per_sqm := 1.50
  abs (calculate_path_cost plot_length plot_width gravel_path_width concrete_path_width
                           gravel_cost_per_sqm concrete_cost_per_sqm - 9.78) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_path_cost_calculation_l3797_379700


namespace NUMINAMATH_CALUDE_range_of_m_l3797_379723

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = x * y)
  (h_inequality : ∃ x y, x > 0 ∧ y > 0 ∧ x + y = x * y ∧ x + 4 * y < m^2 + 8 * m) :
  m < -9 ∨ m > 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3797_379723


namespace NUMINAMATH_CALUDE_five_distinct_roots_l3797_379765

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

-- Define the composition f(f(x))
def f_comp_f (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

-- State the theorem
theorem five_distinct_roots (c : ℝ) : 
  (∃! (roots : Finset ℝ), roots.card = 5 ∧ ∀ x ∈ roots, f_comp_f c x = 0) ↔ (c = 0 ∨ c = 3) :=
sorry

end NUMINAMATH_CALUDE_five_distinct_roots_l3797_379765


namespace NUMINAMATH_CALUDE_sum_9_is_27_l3797_379711

/-- An arithmetic sequence on a line through (5,3) -/
structure ArithmeticSequenceOnLine where
  a : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  on_line : ∀ n : ℕ+, ∃ k m : ℚ, a n = k * n + m ∧ 3 = k * 5 + m

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequenceOnLine) (n : ℕ+) : ℚ :=
  (n : ℚ) * seq.a n

/-- The sum of the first 9 terms of an arithmetic sequence on a line through (5,3) is 27 -/
theorem sum_9_is_27 (seq : ArithmeticSequenceOnLine) : sum_n seq 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_9_is_27_l3797_379711


namespace NUMINAMATH_CALUDE_correct_ticket_count_l3797_379772

/-- Represents the number of first-class tickets bought -/
def first_class_tickets : ℕ := 20

/-- Represents the number of second-class tickets bought -/
def second_class_tickets : ℕ := 45 - first_class_tickets

/-- The total cost of all tickets -/
def total_cost : ℕ := 400

theorem correct_ticket_count :
  first_class_tickets * 10 + second_class_tickets * 8 = total_cost ∧
  first_class_tickets + second_class_tickets = 45 :=
sorry

end NUMINAMATH_CALUDE_correct_ticket_count_l3797_379772


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3797_379710

theorem chess_tournament_participants (total_games : ℕ) 
  (h1 : total_games = 105) : ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = total_games := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l3797_379710


namespace NUMINAMATH_CALUDE_min_value_expression_l3797_379729

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 2) :
  ∃ (m : ℝ), m = 25/2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 2 → (1 + 4*x + 3*y) / (x*y) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3797_379729


namespace NUMINAMATH_CALUDE_hotel_room_charges_l3797_379794

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R - 0.55 * R) 
  (h2 : P = G - 0.10 * G) : 
  R = 2 * G := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l3797_379794


namespace NUMINAMATH_CALUDE_rectangle_area_l3797_379727

/-- Given a rectangle composed of 24 congruent squares arranged in a 6x4 format
    with a diagonal of 10 cm, the total area of the rectangle is 2400/13 square cm. -/
theorem rectangle_area (squares : ℕ) (rows cols : ℕ) (diagonal : ℝ) :
  squares = 24 →
  rows = 6 →
  cols = 4 →
  diagonal = 10 →
  (rows * cols : ℝ) * (diagonal^2 / (rows^2 + cols^2)) = 2400 / 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3797_379727


namespace NUMINAMATH_CALUDE_jellybean_probability_l3797_379779

/-- The probability of selecting exactly one red and two blue jellybeans from a bowl -/
theorem jellybean_probability :
  let total_jellybeans : ℕ := 15
  let red_jellybeans : ℕ := 5
  let blue_jellybeans : ℕ := 3
  let white_jellybeans : ℕ := 7
  let picked_jellybeans : ℕ := 3

  -- Ensure the total number of jellybeans is correct
  total_jellybeans = red_jellybeans + blue_jellybeans + white_jellybeans →

  -- Calculate the probability
  (Nat.choose red_jellybeans 1 * Nat.choose blue_jellybeans 2 : ℚ) /
  Nat.choose total_jellybeans picked_jellybeans = 3 / 91 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l3797_379779


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3797_379753

/-- The number of integer solutions to the equation 2x^2 + 5xy + 3y^2 = 30 -/
def num_solutions : ℕ := 16

/-- The quadratic equation -/
def quadratic_equation (x y : ℤ) : Prop :=
  2 * x^2 + 5 * x * y + 3 * y^2 = 30

/-- Known solution to the equation -/
def known_solution : ℤ × ℤ := (9, -4)

theorem quadratic_equation_solutions :
  (quadratic_equation known_solution.1 known_solution.2) ∧
  (∃ (solutions : Finset (ℤ × ℤ)), 
    solutions.card = num_solutions ∧
    ∀ (sol : ℤ × ℤ), sol ∈ solutions ↔ quadratic_equation sol.1 sol.2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3797_379753


namespace NUMINAMATH_CALUDE_remainder_theorem_l3797_379713

-- Define the polynomial q(x)
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 5

-- State the theorem
theorem remainder_theorem (D E F : ℝ) :
  (∃ k : ℝ, q D E F x = (x - 2) * k + 15) →
  (∃ m : ℝ, q D E F x = (x + 2) * m + 15) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3797_379713


namespace NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l3797_379733

-- Define the binomial expansion function
def binomialCoefficient (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x
def coefficientOfX (a b : ℤ) (n : ℕ) : ℤ :=
  binomialCoefficient n 2 * (b ^ 2)

-- Theorem statement
theorem coefficient_of_x_in_expansion :
  coefficientOfX 1 (-2) 5 = 40 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l3797_379733


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3797_379730

theorem complex_fraction_equality : (1 : ℂ) / (3 * I + 1) = (1 : ℂ) / 10 + (3 : ℂ) * I / 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3797_379730


namespace NUMINAMATH_CALUDE_square_equals_25_l3797_379785

theorem square_equals_25 : {x : ℝ | x^2 = 25} = {-5, 5} := by sorry

end NUMINAMATH_CALUDE_square_equals_25_l3797_379785


namespace NUMINAMATH_CALUDE_work_time_proof_l3797_379731

theorem work_time_proof (a b c h : ℝ) : 
  (1 / a + 1 / b + 1 / c = 1 / (a - 6)) →
  (1 / a + 1 / b + 1 / c = 1 / (b - 1)) →
  (1 / a + 1 / b + 1 / c = 2 / c) →
  (1 / a + 1 / b = 1 / h) →
  (a > 0) → (b > 0) → (c > 0) → (h > 0) →
  h = 4/3 := by
sorry

end NUMINAMATH_CALUDE_work_time_proof_l3797_379731


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l3797_379751

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 6 * a 6 - 8 * a 6 + 4 = 0) →
  (a 10 * a 10 - 8 * a 10 + 4 = 0) →
  (a 8 = 2 ∨ a 8 = -2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l3797_379751


namespace NUMINAMATH_CALUDE_infinitely_many_multiples_of_100_l3797_379780

theorem infinitely_many_multiples_of_100 :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 100 ∣ (2^n + n^2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_multiples_of_100_l3797_379780


namespace NUMINAMATH_CALUDE_salt_mixture_theorem_l3797_379715

/-- Represents the salt mixture problem -/
def SaltMixture (cheap_price cheap_weight expensive_price expensive_weight profit_percentage : ℚ) : Prop :=
  let total_cost : ℚ := cheap_price * cheap_weight + expensive_price * expensive_weight
  let total_weight : ℚ := cheap_weight + expensive_weight
  let profit : ℚ := total_cost * (profit_percentage / 100)
  let selling_price : ℚ := total_cost + profit
  let selling_price_per_pound : ℚ := selling_price / total_weight
  selling_price_per_pound = 48 / 100

/-- The salt mixture theorem -/
theorem salt_mixture_theorem : SaltMixture (38/100) 40 (50/100) 8 20 := by
  sorry

end NUMINAMATH_CALUDE_salt_mixture_theorem_l3797_379715
