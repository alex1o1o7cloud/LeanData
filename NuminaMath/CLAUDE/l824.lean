import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l824_82443

theorem arithmetic_mean_of_fractions : 
  let a := (7 : ℚ) / 10
  let b := (4 : ℚ) / 5
  let c := (3 : ℚ) / 4
  c = (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l824_82443


namespace NUMINAMATH_CALUDE_quad_pair_f_one_l824_82464

/-- Two quadratic polynomials satisfying specific conditions -/
structure QuadraticPair :=
  (f g : ℝ → ℝ)
  (quad_f : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (quad_g : ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c)
  (h1 : f 2 = 2 ∧ f 3 = 2)
  (h2 : g 2 = 2 ∧ g 3 = 2)
  (h3 : g 1 = 3)
  (h4 : f 4 = 7)
  (h5 : g 4 = 4)

/-- The main theorem stating that f(1) = 7 for the given conditions -/
theorem quad_pair_f_one (qp : QuadraticPair) : qp.f 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quad_pair_f_one_l824_82464


namespace NUMINAMATH_CALUDE_consecutive_products_not_end_2019_l824_82412

theorem consecutive_products_not_end_2019 (n : ℤ) : 
  ∃ k : ℕ, ((n - 1) * (n + 1) + n * (n - 1) + n * (n + 1)) % 10000 ≠ 2019 + 10000 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_products_not_end_2019_l824_82412


namespace NUMINAMATH_CALUDE_unit_vector_same_direction_l824_82482

def b : Fin 2 → ℝ := ![(-3), 4]

theorem unit_vector_same_direction (a : Fin 2 → ℝ) : 
  (∀ i, a i * a i = 1) →  -- a is a unit vector
  (∃ c : ℝ, c ≠ 0 ∧ ∀ i, a i = c * b i) →  -- a is in the same direction as b
  a = ![(-3/5), 4/5] := by
sorry

end NUMINAMATH_CALUDE_unit_vector_same_direction_l824_82482


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l824_82454

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 714 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (eq1 : a + b = 18)
  (eq2 : a * b + c + d = 95)
  (eq3 : a * d + b * c = 195)
  (eq4 : c * d = 120) :
  a^2 + b^2 + c^2 + d^2 ≤ 714 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 
    a₀ + b₀ = 18 ∧
    a₀ * b₀ + c₀ + d₀ = 95 ∧
    a₀ * d₀ + b₀ * c₀ = 195 ∧
    c₀ * d₀ = 120 ∧
    a₀^2 + b₀^2 + c₀^2 + d₀^2 = 714 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l824_82454


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l824_82497

theorem complex_magnitude_equation : 
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (8 + 3 * t * Complex.I) = 13 ↔ t = Real.sqrt (105 / 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l824_82497


namespace NUMINAMATH_CALUDE_min_draws_for_eighteen_balls_l824_82478

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to draw to guarantee at least n balls of a single color -/
def minDrawsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

theorem min_draws_for_eighteen_balls (counts : BallCounts) 
  (h_red : counts.red = 30)
  (h_green : counts.green = 23)
  (h_yellow : counts.yellow = 21)
  (h_blue : counts.blue = 17)
  (h_white : counts.white = 14)
  (h_black : counts.black = 12) :
  minDrawsForColor counts 18 = 95 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_eighteen_balls_l824_82478


namespace NUMINAMATH_CALUDE_max_gift_sets_l824_82475

theorem max_gift_sets (total_chocolates total_candies left_chocolates left_candies : ℕ)
  (h1 : total_chocolates = 69)
  (h2 : total_candies = 86)
  (h3 : left_chocolates = 5)
  (h4 : left_candies = 6) :
  Nat.gcd (total_chocolates - left_chocolates) (total_candies - left_candies) = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_gift_sets_l824_82475


namespace NUMINAMATH_CALUDE_coefficient_x4_in_binomial_expansion_l824_82496

theorem coefficient_x4_in_binomial_expansion :
  (Finset.range 11).sum (fun k => (Nat.choose 10 k) * (1^(10 - k)) * (1^k)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_binomial_expansion_l824_82496


namespace NUMINAMATH_CALUDE_even_sum_problem_l824_82437

theorem even_sum_problem (n : ℕ) (h1 : Odd n) 
  (h2 : (n^2 - 1) / 4 = 95 * 96) : n = 191 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_problem_l824_82437


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_two_l824_82449

theorem sqrt_fraction_equals_two (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 7) :
  Real.sqrt ((14 * a^2) / b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_two_l824_82449


namespace NUMINAMATH_CALUDE_pizza_slice_count_l824_82474

/-- Given a number of pizzas and slices per pizza, calculates the total number of slices -/
def total_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  num_pizzas * slices_per_pizza

/-- Proves that 21 pizzas with 8 slices each results in 168 total slices -/
theorem pizza_slice_count : total_slices 21 8 = 168 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_count_l824_82474


namespace NUMINAMATH_CALUDE_leadership_structure_count_is_correct_l824_82462

def tribe_size : ℕ := 15
def num_kings : ℕ := 1
def num_knights : ℕ := 2
def squires_per_knight : ℕ := 3

def leadership_structure_count : ℕ :=
  tribe_size * (tribe_size - 1).choose num_knights *
  (tribe_size - num_kings - num_knights).choose squires_per_knight *
  (tribe_size - num_kings - num_knights - squires_per_knight).choose squires_per_knight

theorem leadership_structure_count_is_correct :
  leadership_structure_count = 27392400 := by sorry

end NUMINAMATH_CALUDE_leadership_structure_count_is_correct_l824_82462


namespace NUMINAMATH_CALUDE_jack_bike_percentage_l824_82415

def original_paycheck : ℝ := 125
def tax_rate : ℝ := 0.20
def savings_amount : ℝ := 20

theorem jack_bike_percentage :
  let after_tax := original_paycheck * (1 - tax_rate)
  let remaining := after_tax - savings_amount
  let bike_percentage := (remaining / after_tax) * 100
  bike_percentage = 80 := by sorry

end NUMINAMATH_CALUDE_jack_bike_percentage_l824_82415


namespace NUMINAMATH_CALUDE_equation_solution_difference_l824_82425

theorem equation_solution_difference : ∃ (a b : ℝ),
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -6 → ((5 * x - 20) / (x^2 + 3*x - 18) = x + 3 ↔ (x = a ∨ x = b))) ∧
  a > b ∧
  a - b = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l824_82425


namespace NUMINAMATH_CALUDE_rectangle_area_l824_82446

/-- Theorem: Area of a rectangle with one side 15 and diagonal 17 is 120 -/
theorem rectangle_area (side : ℝ) (diagonal : ℝ) (area : ℝ) : 
  side = 15 → diagonal = 17 → area = side * (Real.sqrt (diagonal^2 - side^2)) → area = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l824_82446


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l824_82472

/-- Given two points M₁(x₁, y₁, z₁) and M₂(x₂, y₂, z₂), this theorem proves that the x-coordinate
    of the point P(x, 0, 0) on the Ox axis that is equidistant from M₁ and M₂ is given by
    x = (x₂² - x₁² + y₂² - y₁² + z₂² - z₁²) / (2(x₂ - x₁)) -/
theorem equidistant_point_on_x_axis 
  (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) 
  (h : x₁ ≠ x₂) : 
  ∃ x : ℝ, x = (x₂^2 - x₁^2 + y₂^2 - y₁^2 + z₂^2 - z₁^2) / (2 * (x₂ - x₁)) ∧ 
  (x - x₁)^2 + y₁^2 + z₁^2 = (x - x₂)^2 + y₂^2 + z₂^2 :=
by sorry

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l824_82472


namespace NUMINAMATH_CALUDE_greatest_integer_third_side_l824_82400

theorem greatest_integer_third_side (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), c = 17 ∧ 
  (∀ (x : ℕ), x > c → ¬(x < a + b ∧ x > |a - b| ∧ a < b + x ∧ b < a + x)) ∧
  (c < a + b ∧ c > |a - b| ∧ a < b + c ∧ b < a + c) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_third_side_l824_82400


namespace NUMINAMATH_CALUDE_cookies_left_after_week_l824_82426

/-- The number of cookies left after a week -/
def cookiesLeftAfterWeek (initialCookies : ℕ) (cookiesTakenInFourDays : ℕ) : ℕ :=
  initialCookies - 7 * (cookiesTakenInFourDays / 4)

/-- Theorem: The number of cookies left after a week is 28 -/
theorem cookies_left_after_week :
  cookiesLeftAfterWeek 70 24 = 28 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_after_week_l824_82426


namespace NUMINAMATH_CALUDE_circle_radius_l824_82499

theorem circle_radius (x y : ℝ) :
  x^2 + y^2 - 2*x + 6*y + 1 = 0 → ∃ (h k r : ℝ), r = 3 ∧ (x - h)^2 + (y - k)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l824_82499


namespace NUMINAMATH_CALUDE_range_of_a_l824_82438

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0
def q (x a : ℝ) : Prop := x^2 - (2*a - 1)*x + a*(a - 1) ≥ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (0 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l824_82438


namespace NUMINAMATH_CALUDE_angle_measure_proof_l824_82419

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x - 2) = 90) → x = 23 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l824_82419


namespace NUMINAMATH_CALUDE_davids_purchase_cost_l824_82493

/-- The minimum cost to buy a given number of bottles, given the price of individual bottles and packs --/
def min_cost (single_price : ℚ) (pack_price : ℚ) (pack_size : ℕ) (total_bottles : ℕ) : ℚ :=
  let num_packs := total_bottles / pack_size
  let remaining_bottles := total_bottles % pack_size
  num_packs * pack_price + remaining_bottles * single_price

/-- Theorem stating the minimum cost for David's purchase --/
theorem davids_purchase_cost :
  let single_price : ℚ := 280 / 100  -- $2.80
  let pack_price : ℚ := 1500 / 100   -- $15.00
  let pack_size : ℕ := 6
  let total_bottles : ℕ := 22
  min_cost single_price pack_price pack_size total_bottles = 5620 / 100 := by
  sorry


end NUMINAMATH_CALUDE_davids_purchase_cost_l824_82493


namespace NUMINAMATH_CALUDE_probability_of_color_change_is_three_seventeenths_l824_82441

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light -/
def probabilityOfColorChange (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℚ :=
  let totalCycleDuration := cycle.green + cycle.yellow + cycle.red
  let changeWindows := 3 * observationDuration
  ↑changeWindows / ↑totalCycleDuration

/-- The main theorem stating the probability of observing a color change -/
theorem probability_of_color_change_is_three_seventeenths :
  let cycle := TrafficLightCycle.mk 45 5 35
  let observationDuration := 5
  probabilityOfColorChange cycle observationDuration = 3 / 17 := by
  sorry

#eval probabilityOfColorChange (TrafficLightCycle.mk 45 5 35) 5

end NUMINAMATH_CALUDE_probability_of_color_change_is_three_seventeenths_l824_82441


namespace NUMINAMATH_CALUDE_range_of_a_l824_82416

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_a (a : ℝ) : Prop := a ≤ -2 ∨ a = 1

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, (¬(¬(p a) ∨ ¬(q a))) → range_a a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l824_82416


namespace NUMINAMATH_CALUDE_triangle_area_l824_82476

theorem triangle_area (a b c A B C : Real) : 
  a = 7 →
  2 * Real.sin A = Real.sqrt 3 →
  Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14 →
  (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l824_82476


namespace NUMINAMATH_CALUDE_segments_in_proportion_l824_82489

-- Define the set of line segments
def segments : List ℝ := [2, 3, 4, 6]

-- Define what it means for a list of four numbers to be in proportion
def isInProportion (l : List ℝ) : Prop :=
  l.length = 4 ∧ l[0]! * l[3]! = l[1]! * l[2]!

-- Theorem statement
theorem segments_in_proportion : isInProportion segments := by
  sorry

end NUMINAMATH_CALUDE_segments_in_proportion_l824_82489


namespace NUMINAMATH_CALUDE_percentage_increase_relation_l824_82401

theorem percentage_increase_relation (A B k x : ℝ) : 
  A > 0 → B > 0 → k > 1 → A = k * B → A = B * (1 + x / 100) → k = 1 + x / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_relation_l824_82401


namespace NUMINAMATH_CALUDE_min_tries_for_given_counts_l824_82483

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  purple : Nat
  blue : Nat
  yellow : Nat
  green : Nat

/-- The minimum number of tries required to get at least two blue, two yellow, and one green ball -/
def minTriesRequired (counts : BallCounts) : Nat :=
  counts.purple + (counts.yellow - 1) + (counts.green - 1) + 2 + 2

/-- Theorem stating the minimum number of tries required for the given ball counts -/
theorem min_tries_for_given_counts :
  let counts : BallCounts := ⟨9, 7, 13, 6⟩
  minTriesRequired counts = 30 := by sorry

end NUMINAMATH_CALUDE_min_tries_for_given_counts_l824_82483


namespace NUMINAMATH_CALUDE_mismatched_boots_count_l824_82436

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange n distinct items --/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of pairs of boots --/
def num_pairs : ℕ := 6

/-- The number of ways two people can wear mismatched boots --/
def mismatched_boots_ways : ℕ :=
  -- Case 1: Using boots from two pairs
  choose num_pairs 2 * 4 +
  -- Case 2: Using boots from three pairs
  choose num_pairs 3 * 4 * 4 +
  -- Case 3: Using boots from four pairs
  choose num_pairs 4 * factorial 4

theorem mismatched_boots_count :
  mismatched_boots_ways = 740 := by sorry

end NUMINAMATH_CALUDE_mismatched_boots_count_l824_82436


namespace NUMINAMATH_CALUDE_range_of_x_l824_82420

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) :
  x ≤ -2 ∨ x ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l824_82420


namespace NUMINAMATH_CALUDE_expression_value_l824_82470

theorem expression_value : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 10) / (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l824_82470


namespace NUMINAMATH_CALUDE_percentage_decrease_in_hours_l824_82469

/-- Represents Jane's toy bear production --/
structure BearProduction where
  bears_without_assistant : ℝ
  hours_without_assistant : ℝ
  bears_with_assistant : ℝ
  hours_with_assistant : ℝ

/-- The conditions of Jane's toy bear production --/
def production_conditions (p : BearProduction) : Prop :=
  p.bears_with_assistant = 1.8 * p.bears_without_assistant ∧
  (p.bears_with_assistant / p.hours_with_assistant) = 2 * (p.bears_without_assistant / p.hours_without_assistant)

/-- The theorem stating the percentage decrease in hours worked --/
theorem percentage_decrease_in_hours (p : BearProduction) 
  (h : production_conditions p) : 
  (p.hours_without_assistant - p.hours_with_assistant) / p.hours_without_assistant * 100 = 10 := by
  sorry


end NUMINAMATH_CALUDE_percentage_decrease_in_hours_l824_82469


namespace NUMINAMATH_CALUDE_rectangle_width_l824_82455

/-- Given a rectangle with length 18 cm and a largest inscribed circle with area 153.93804002589985 square cm, the width of the rectangle is 14 cm. -/
theorem rectangle_width (length : ℝ) (circle_area : ℝ) (width : ℝ) : 
  length = 18 → 
  circle_area = 153.93804002589985 → 
  circle_area = Real.pi * (width / 2)^2 → 
  width = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l824_82455


namespace NUMINAMATH_CALUDE_profit_difference_l824_82405

def original_profit_percentage : ℝ := 0.1
def new_purchase_discount : ℝ := 0.1
def new_profit_percentage : ℝ := 0.3
def original_selling_price : ℝ := 1099.999999999999

theorem profit_difference :
  let original_purchase_price := original_selling_price / (1 + original_profit_percentage)
  let new_purchase_price := original_purchase_price * (1 - new_purchase_discount)
  let new_selling_price := new_purchase_price * (1 + new_profit_percentage)
  new_selling_price - original_selling_price = 70 := by sorry

end NUMINAMATH_CALUDE_profit_difference_l824_82405


namespace NUMINAMATH_CALUDE_min_value_polynomial_l824_82484

theorem min_value_polynomial (x : ℝ) : 
  (13 - x) * (11 - x) * (13 + x) * (11 + x) + 1000 ≥ 424 :=
by sorry

end NUMINAMATH_CALUDE_min_value_polynomial_l824_82484


namespace NUMINAMATH_CALUDE_max_perimeter_after_cut_l824_82403

theorem max_perimeter_after_cut (original_length original_width cut_length cut_width : ℝ) 
  (h1 : original_length = 20)
  (h2 : original_width = 16)
  (h3 : cut_length = 8)
  (h4 : cut_width = 4)
  (h5 : cut_length ≤ original_length ∧ cut_width ≤ original_width) :
  ∃ (remaining_perimeter : ℝ), 
    remaining_perimeter ≤ 2 * (original_length + original_width) + 2 * min cut_length cut_width ∧
    remaining_perimeter = 88 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_after_cut_l824_82403


namespace NUMINAMATH_CALUDE_max_cookies_eaten_24_l824_82458

/-- Given two siblings sharing cookies, where one eats a positive multiple
    of the other's cookies, this function calculates the maximum number
    of cookies the first sibling could have eaten. -/
def max_cookies_eaten (total_cookies : ℕ) : ℕ :=
  total_cookies / 2

/-- Theorem stating that given 24 cookies shared between two siblings,
    where one sibling eats a positive multiple of the other's cookies,
    the maximum number of cookies the first sibling could have eaten is 12. -/
theorem max_cookies_eaten_24 :
  max_cookies_eaten 24 = 12 := by
  sorry

#eval max_cookies_eaten 24

end NUMINAMATH_CALUDE_max_cookies_eaten_24_l824_82458


namespace NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l824_82413

theorem consecutive_integer_product_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 11 * m) →
  (∃ m : ℤ, n = 10 * m) ∧
  (∃ m : ℤ, n = 22 * m) ∧
  (∃ m : ℤ, n = 33 * m) ∧
  (∃ m : ℤ, n = 66 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 44 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l824_82413


namespace NUMINAMATH_CALUDE_staircase_cutting_count_l824_82439

/-- Represents a staircase with a given number of steps -/
structure Staircase :=
  (steps : ℕ)

/-- Represents a cutting of the staircase into rectangles and a square -/
structure StaircaseCutting :=
  (staircase : Staircase)
  (rectangles : ℕ)
  (squares : ℕ)

/-- Counts the number of ways to cut a staircase -/
def countCuttings (s : Staircase) (r : ℕ) (sq : ℕ) : ℕ :=
  sorry

/-- The main theorem: there are 32 ways to cut a 6-step staircase into 5 rectangles and one square -/
theorem staircase_cutting_count :
  countCuttings (Staircase.mk 6) 5 1 = 32 := by
  sorry

end NUMINAMATH_CALUDE_staircase_cutting_count_l824_82439


namespace NUMINAMATH_CALUDE_exists_quadratic_function_with_conditions_l824_82487

/-- A quadratic function with coefficient a, b, and c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The vertex of a quadratic function is on the negative half of the y-axis -/
def VertexOnNegativeYAxis (a b c : ℝ) : Prop :=
  b = 0 ∧ c < 0

/-- The part of the quadratic function to the left of its axis of symmetry is rising -/
def LeftPartRising (a b c : ℝ) : Prop :=
  a < 0

/-- Theorem stating the existence of a quadratic function satisfying the given conditions -/
theorem exists_quadratic_function_with_conditions : ∃ a b c : ℝ,
  VertexOnNegativeYAxis a b c ∧
  LeftPartRising a b c ∧
  QuadraticFunction a b c = QuadraticFunction (-1) 0 (-1) :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_function_with_conditions_l824_82487


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l824_82490

theorem fraction_sum_equals_one : 3/5 - 1/10 + 1/2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l824_82490


namespace NUMINAMATH_CALUDE_function_composition_equality_l824_82453

/-- Given two functions f and g, where f(x) = Ax³ - B and g(x) = Bx², 
    with B ≠ 0 and f(g(2)) = 0, prove that A = 1 / (64B²) -/
theorem function_composition_equality (A B : ℝ) 
  (hB : B ≠ 0)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x^3 - B)
  (hg : ∀ x, g x = B * x^2)
  (h_comp : f (g 2) = 0) :
  A = 1 / (64 * B^2) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l824_82453


namespace NUMINAMATH_CALUDE_probability_of_event_D_is_one_l824_82451

theorem probability_of_event_D_is_one :
  ∀ x : ℝ,
  (∃ (P_N P_D_given_N P_D : ℝ),
    P_N = 3/8 ∧
    P_D_given_N = x^2 ∧
    P_D = 5/8 + (3/8) * x^2 ∧
    0 ≤ P_N ∧ P_N ≤ 1 ∧
    0 ≤ P_D_given_N ∧ P_D_given_N ≤ 1 ∧
    0 ≤ P_D ∧ P_D ≤ 1) →
  P_D = 1 :=
sorry

end NUMINAMATH_CALUDE_probability_of_event_D_is_one_l824_82451


namespace NUMINAMATH_CALUDE_original_fraction_l824_82411

theorem original_fraction (x y : ℚ) 
  (h1 : x / (y + 1) = 1 / 2) 
  (h2 : (x + 1) / y = 1) : 
  x / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l824_82411


namespace NUMINAMATH_CALUDE_first_day_exceeding_500_l824_82486

def bacterial_population (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_exceeding_500 :
  ∃ n : ℕ, bacterial_population n > 500 ∧ ∀ m : ℕ, m < n → bacterial_population m ≤ 500 :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_first_day_exceeding_500_l824_82486


namespace NUMINAMATH_CALUDE_rotten_eggs_probability_l824_82408

/-- The probability of selecting 2 rotten eggs from a pack of 36 eggs containing 3 rotten eggs -/
theorem rotten_eggs_probability (total_eggs : ℕ) (rotten_eggs : ℕ) (selected_eggs : ℕ) : 
  total_eggs = 36 → rotten_eggs = 3 → selected_eggs = 2 →
  (Nat.choose rotten_eggs selected_eggs : ℚ) / (Nat.choose total_eggs selected_eggs) = 1 / 420 :=
by sorry

end NUMINAMATH_CALUDE_rotten_eggs_probability_l824_82408


namespace NUMINAMATH_CALUDE_hyperbola_equation_l824_82491

-- Define the hyperbola
def Hyperbola (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- State the theorem
theorem hyperbola_equation :
  ∀ (a b c : ℝ),
    -- Conditions
    (2 * a = 8) →  -- Distance between vertices
    (c / a = 5 / 4) →  -- Eccentricity
    (c^2 = a^2 + b^2) →  -- Relation between a, b, and c
    -- Conclusion
    Hyperbola a b c = Hyperbola 4 3 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l824_82491


namespace NUMINAMATH_CALUDE_jenny_recycling_l824_82460

theorem jenny_recycling (total_weight : ℕ) (can_weight : ℕ) (num_cans : ℕ)
  (bottle_price : ℕ) (can_price : ℕ) (total_earnings : ℕ) :
  total_weight = 100 →
  can_weight = 2 →
  num_cans = 20 →
  bottle_price = 10 →
  can_price = 3 →
  total_earnings = 160 →
  ∃ (bottle_weight : ℕ), 
    bottle_weight = 6 ∧
    bottle_weight * ((total_weight - (can_weight * num_cans)) / bottle_weight) = 
      total_weight - (can_weight * num_cans) ∧
    bottle_price * ((total_weight - (can_weight * num_cans)) / bottle_weight) + 
      can_price * num_cans = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_jenny_recycling_l824_82460


namespace NUMINAMATH_CALUDE_actual_daily_production_l824_82432

/-- The actual daily production of TVs given the planned production and early completion. -/
theorem actual_daily_production
  (planned_production : ℕ)
  (planned_days : ℕ)
  (days_ahead : ℕ)
  (h1 : planned_production = 560)
  (h2 : planned_days = 16)
  (h3 : days_ahead = 2)
  : (planned_production : ℚ) / (planned_days - days_ahead) = 40 := by
  sorry

end NUMINAMATH_CALUDE_actual_daily_production_l824_82432


namespace NUMINAMATH_CALUDE_cubic_km_to_m_strip_l824_82406

/-- The length of a strip formed by cutting a cubic kilometer into cubic meters and laying them out in a single line -/
def strip_length : ℝ := 1000000

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

theorem cubic_km_to_m_strip : 
  strip_length = (km_to_m ^ 3) / km_to_m := by sorry

end NUMINAMATH_CALUDE_cubic_km_to_m_strip_l824_82406


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_1800_l824_82456

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

theorem largest_perfect_square_factor_1800 :
  largest_perfect_square_factor 1800 = 900 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_1800_l824_82456


namespace NUMINAMATH_CALUDE_function_local_max_condition_l824_82417

/-- Given a real constant a, prove that for a function f(x) = (x-a)²(x+b)e^x 
    where b is real and x=a is a local maximum point of f(x), 
    then b must be less than -a. -/
theorem function_local_max_condition (a : ℝ) :
  ∀ b : ℝ, (∃ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = (x - a)^2 * (x + b) * Real.exp x) ∧
    (IsLocalMax f a)) →
  b < -a :=
by sorry

end NUMINAMATH_CALUDE_function_local_max_condition_l824_82417


namespace NUMINAMATH_CALUDE_projectiles_meeting_time_l824_82461

/-- Theorem: Time for two projectiles to meet --/
theorem projectiles_meeting_time
  (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : distance = 2520)
  (h2 : speed1 = 432)
  (h3 : speed2 = 576) :
  (distance / (speed1 + speed2)) * 60 = 150 :=
by sorry

end NUMINAMATH_CALUDE_projectiles_meeting_time_l824_82461


namespace NUMINAMATH_CALUDE_tony_winnings_l824_82488

/-- Calculates the total winnings for lottery tickets with identical numbers -/
def totalWinnings (numTickets : ℕ) (winningNumbersPerTicket : ℕ) (valuePerWinningNumber : ℕ) : ℕ :=
  numTickets * winningNumbersPerTicket * valuePerWinningNumber

/-- Theorem: Tony's total winnings are $300 -/
theorem tony_winnings :
  totalWinnings 3 5 20 = 300 := by
  sorry

end NUMINAMATH_CALUDE_tony_winnings_l824_82488


namespace NUMINAMATH_CALUDE_student_committee_size_l824_82414

theorem student_committee_size (ways : ℕ) (h : ways = 30) : 
  (∃ n : ℕ, n * (n - 1) = ways) → 
  (∃! n : ℕ, n > 0 ∧ n * (n - 1) = ways) ∧ 
  (∃ n : ℕ, n > 0 ∧ n * (n - 1) = ways ∧ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_student_committee_size_l824_82414


namespace NUMINAMATH_CALUDE_rectangle_height_decrease_l824_82459

theorem rectangle_height_decrease (b h : ℝ) (h_pos : 0 < b) (h_pos' : 0 < h) :
  let new_base := 1.1 * b
  let new_height := h * (1 - 9 / 11 / 100)
  b * h = new_base * new_height := by
  sorry

end NUMINAMATH_CALUDE_rectangle_height_decrease_l824_82459


namespace NUMINAMATH_CALUDE_unique_integer_property_l824_82447

theorem unique_integer_property (a : ℕ+) : 
  let b := 2 * a ^ 2
  let c := 2 * b ^ 2
  let d := 2 * c ^ 2
  (∃ n k : ℕ, a * 10^(n+k) + b * 10^k + c = d) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_property_l824_82447


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l824_82407

theorem trigonometric_equation_solution (x : ℝ) 
  (h_eq : 8.459 * (Real.cos (x^2))^2 * (Real.tan (x^2) + 2 * Real.tan x) + 
          (Real.tan x)^3 * (1 - (Real.sin (x^2))^2) * (2 - Real.tan x * Real.tan (x^2)) = 0)
  (h_cos : Real.cos x ≠ 0)
  (h_x_sq : ∀ n : ℤ, x^2 ≠ Real.pi/2 + Real.pi * n)
  (h_x_1 : ∀ m : ℤ, x ≠ Real.pi/4 + Real.pi * m/2)
  (h_x_2 : ∀ l : ℤ, x ≠ Real.pi/2 + Real.pi * l) :
  ∃ k : ℕ, x = -1 + Real.sqrt (Real.pi * k + 1) ∨ x = -1 - Real.sqrt (Real.pi * k + 1) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l824_82407


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l824_82450

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧
  ∀ (y : ℕ), y > 0 ∧ (5 * y) % 31 = 17 % 31 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l824_82450


namespace NUMINAMATH_CALUDE_large_bucket_capacity_l824_82424

theorem large_bucket_capacity (small : ℝ) (large : ℝ) 
  (h1 : large = 2 * small + 3)
  (h2 : 2 * small + 5 * large = 63) :
  large = 11 := by
sorry

end NUMINAMATH_CALUDE_large_bucket_capacity_l824_82424


namespace NUMINAMATH_CALUDE_cheryl_unused_material_l824_82473

-- Define the amount of material Cheryl bought of each type
def material1 : ℚ := 3 / 8
def material2 : ℚ := 1 / 3

-- Define the total amount of material Cheryl bought
def total_bought : ℚ := material1 + material2

-- Define the amount of material Cheryl used
def material_used : ℚ := 0.33333333333333326

-- Define the amount of material left unused
def material_left : ℚ := total_bought - material_used

-- Theorem statement
theorem cheryl_unused_material : material_left = 0.375 := by sorry

end NUMINAMATH_CALUDE_cheryl_unused_material_l824_82473


namespace NUMINAMATH_CALUDE_sequence_difference_proof_l824_82466

def arithmetic_sequence_sum (a1 n d : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem sequence_difference_proof : 
  let n1 := (2298 - 2204) / 2 + 1
  let n2 := (400 - 306) / 2 + 1
  arithmetic_sequence_sum 2204 n1 2 - arithmetic_sequence_sum 306 n2 2 = 91056 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_proof_l824_82466


namespace NUMINAMATH_CALUDE_starting_number_sequence_l824_82452

theorem starting_number_sequence (n : ℕ) : 
  (n ≤ 79 ∧ 
   (∃ (a b c d : ℕ), n < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 79 ∧
    n % 11 = 0 ∧ a % 11 = 0 ∧ b % 11 = 0 ∧ c % 11 = 0 ∧ d % 11 = 0) ∧
   (∀ m : ℕ, m < n → ¬(∃ (a b c d : ℕ), m < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 79 ∧
    m % 11 = 0 ∧ a % 11 = 0 ∧ b % 11 = 0 ∧ c % 11 = 0 ∧ d % 11 = 0))) →
  n = 33 := by
sorry

end NUMINAMATH_CALUDE_starting_number_sequence_l824_82452


namespace NUMINAMATH_CALUDE_shopping_remainder_l824_82427

theorem shopping_remainder (initial_amount : ℝ) (grocery_fraction : ℝ) (household_fraction : ℝ) (personal_care_fraction : ℝ) 
  (h1 : initial_amount = 450)
  (h2 : grocery_fraction = 3/5)
  (h3 : household_fraction = 1/6)
  (h4 : personal_care_fraction = 1/10) : 
  initial_amount - (grocery_fraction * initial_amount + household_fraction * initial_amount + personal_care_fraction * initial_amount) = 60 := by
  sorry

end NUMINAMATH_CALUDE_shopping_remainder_l824_82427


namespace NUMINAMATH_CALUDE_youngest_age_l824_82480

/-- Proves the age of the youngest person given the conditions of the problem -/
theorem youngest_age (n : ℕ) (current_avg : ℚ) (birth_avg : ℚ) 
  (h1 : n = 7)
  (h2 : current_avg = 30)
  (h3 : birth_avg = 22) :
  (n * current_avg - (n - 1) * birth_avg) / n = 78 / 7 := by
  sorry

end NUMINAMATH_CALUDE_youngest_age_l824_82480


namespace NUMINAMATH_CALUDE_max_quotient_value_l824_82402

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) :
  (∀ x y, 100 ≤ x ∧ x ≤ 300 → 500 ≤ y ∧ y ≤ 1500 → y / x ≤ b / a) → b / a = 15 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l824_82402


namespace NUMINAMATH_CALUDE_video_game_players_l824_82428

/-- The number of friends who quit the game -/
def quit_players : ℕ := 5

/-- The number of lives each remaining player had -/
def lives_per_player : ℕ := 5

/-- The total number of lives after some players quit -/
def total_lives : ℕ := 15

/-- The initial number of friends playing the video game online -/
def initial_players : ℕ := 8

theorem video_game_players :
  initial_players = quit_players + total_lives / lives_per_player := by
  sorry

end NUMINAMATH_CALUDE_video_game_players_l824_82428


namespace NUMINAMATH_CALUDE_adams_age_l824_82418

theorem adams_age (adam_age eve_age : ℕ) : 
  adam_age = eve_age - 5 →
  eve_age + 1 = 3 * (adam_age - 4) →
  adam_age = 9 := by
sorry

end NUMINAMATH_CALUDE_adams_age_l824_82418


namespace NUMINAMATH_CALUDE_min_radios_problem_l824_82465

/-- Represents the problem of finding the minimum number of radios. -/
theorem min_radios_problem (n d : ℕ) : 
  n > 0 → -- n is positive
  d > 0 → -- d is positive
  (45 : ℚ) - (d + 90 : ℚ) / n = -105 → -- profit equation
  n ≥ 2 := by
  sorry

#check min_radios_problem

end NUMINAMATH_CALUDE_min_radios_problem_l824_82465


namespace NUMINAMATH_CALUDE_buying_more_can_cost_less_buying_101_is_cheaper_l824_82422

/-- The cost function for notebooks -/
def notebook_cost (n : ℕ) : ℝ :=
  if n ≤ 100 then 2.3 * n else 2.2 * n

theorem buying_more_can_cost_less :
  ∃ (n₁ n₂ : ℕ), n₁ < n₂ ∧ notebook_cost n₁ > notebook_cost n₂ :=
sorry

theorem buying_101_is_cheaper :
  notebook_cost 101 < notebook_cost 100 :=
sorry

end NUMINAMATH_CALUDE_buying_more_can_cost_less_buying_101_is_cheaper_l824_82422


namespace NUMINAMATH_CALUDE_annual_increase_rate_l824_82431

theorem annual_increase_rate (initial_value final_value : ℝ) 
  (h1 : initial_value = 6400)
  (h2 : final_value = 8100) :
  ∃ r : ℝ, initial_value * (1 + r)^2 = final_value ∧ r = 0.125 := by
sorry

end NUMINAMATH_CALUDE_annual_increase_rate_l824_82431


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l824_82498

theorem absolute_value_inequality_solution :
  {x : ℤ | |7 * x - 5| ≤ 9} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l824_82498


namespace NUMINAMATH_CALUDE_circle_projection_bodies_l824_82440

/-- A type representing geometric bodies -/
inductive GeometricBody
  | Cone
  | Cylinder
  | Sphere
  | Other

/-- A predicate that determines if a geometric body appears as a circle from a certain perspective -/
def appearsAsCircle (body : GeometricBody) : Prop :=
  sorry

/-- The theorem stating that cones, cylinders, and spheres appear as circles from certain perspectives -/
theorem circle_projection_bodies :
  ∃ (cone cylinder sphere : GeometricBody),
    cone = GeometricBody.Cone ∧
    cylinder = GeometricBody.Cylinder ∧
    sphere = GeometricBody.Sphere ∧
    appearsAsCircle cone ∧
    appearsAsCircle cylinder ∧
    appearsAsCircle sphere :=
  sorry

end NUMINAMATH_CALUDE_circle_projection_bodies_l824_82440


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l824_82410

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l824_82410


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_of_70_factorial_l824_82477

-- Define 70!
def factorial_70 : ℕ := Nat.factorial 70

-- Define the function to get the last two nonzero digits
def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

-- Theorem statement
theorem last_two_nonzero_digits_of_70_factorial :
  last_two_nonzero_digits factorial_70 = 48 := by
  sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_of_70_factorial_l824_82477


namespace NUMINAMATH_CALUDE_apple_distribution_l824_82468

theorem apple_distribution (x y : ℕ) : 
  y = 5 * x + 12 ∧ 0 < 8 * x - y ∧ 8 * x - y < 8 → 
  (x = 5 ∧ y = 37) ∨ (x = 6 ∧ y = 42) := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l824_82468


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l824_82433

theorem exponent_equation_solution :
  ∃ y : ℝ, (3 : ℝ)^(y - 2) = 9^(y - 1) ↔ y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l824_82433


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l824_82442

theorem cube_surface_area_from_volume : 
  ∀ (v : ℝ) (s : ℝ), 
  v = 729 →  -- Given volume
  v = s^3 →  -- Volume formula
  6 * s^2 = 486 -- Surface area formula and result
  := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l824_82442


namespace NUMINAMATH_CALUDE_total_amount_proof_l824_82448

/-- Proves that the total amount is $93,750 given the spending conditions -/
theorem total_amount_proof (raw_materials : ℝ) (machinery : ℝ) (cash_percentage : ℝ) 
  (h1 : raw_materials = 35000)
  (h2 : machinery = 40000)
  (h3 : cash_percentage = 0.20)
  (h4 : ∃ total : ℝ, total = raw_materials + machinery + cash_percentage * total) :
  ∃ total : ℝ, total = 93750 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l824_82448


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_twice_exterior_l824_82404

theorem polygon_sides_when_interior_twice_exterior :
  ∀ n : ℕ,
  (n ≥ 3) →
  ((n - 2) * 180 = 2 * 360) →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_twice_exterior_l824_82404


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_case_l824_82457

/-- The midpoint of a line segment in polar coordinates --/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of a line segment with endpoints (10, π/3) and (10, 2π/3) in polar coordinates is (5√3, π/2) --/
theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 10 (π/3) 10 (2*π/3)
  r = 5 * Real.sqrt 3 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π :=
by sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_case_l824_82457


namespace NUMINAMATH_CALUDE_eraser_cost_l824_82421

def total_money : ℕ := 100
def heaven_spent : ℕ := 30
def brother_highlighters : ℕ := 30
def num_erasers : ℕ := 10

theorem eraser_cost :
  (total_money - heaven_spent - brother_highlighters) / num_erasers = 4 := by
  sorry

end NUMINAMATH_CALUDE_eraser_cost_l824_82421


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l824_82430

theorem point_in_first_quadrant (a : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (-a, a^2)
  P.1 > 0 ∧ P.2 > 0 :=
sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l824_82430


namespace NUMINAMATH_CALUDE_simultaneous_integers_l824_82463

theorem simultaneous_integers (x : ℤ) :
  (∃ y z u : ℤ, (x - 3) = 7 * y ∧ (x - 2) = 5 * z ∧ (x - 4) = 3 * u) ↔
  (∃ t : ℤ, x = 105 * t + 52) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_integers_l824_82463


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_ten_l824_82409

theorem arithmetic_square_root_of_ten : Real.sqrt 10 = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_ten_l824_82409


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l824_82481

theorem sin_cos_sum_equals_sqrt2_over_2 :
  Real.sin (187 * π / 180) * Real.cos (52 * π / 180) +
  Real.cos (7 * π / 180) * Real.sin (52 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l824_82481


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l824_82494

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let focal_distance := 2 * c
  let asymptote_slope := b / a
  let focus_to_asymptote_distance := b * c / Real.sqrt (a^2 + b^2)
  focus_to_asymptote_distance = (1/4) * focal_distance →
  c / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l824_82494


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l824_82429

theorem sum_of_squares_of_roots (p q r : ℂ) : 
  (3 * p^3 - 4 * p^2 + 3 * p + 7 = 0) →
  (3 * q^3 - 4 * q^2 + 3 * q + 7 = 0) →
  (3 * r^3 - 4 * r^2 + 3 * r + 7 = 0) →
  p^2 + q^2 + r^2 = -2/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l824_82429


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l824_82467

/-- The scientific notation of 15.6 billion -/
def scientific_notation_15_6_billion : ℝ := 1.56 * (10 ^ 9)

/-- 15.6 billion as a real number -/
def fifteen_point_six_billion : ℝ := 15600000000

/-- Theorem stating that the scientific notation of 15.6 billion is correct -/
theorem scientific_notation_correct : 
  scientific_notation_15_6_billion = fifteen_point_six_billion := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l824_82467


namespace NUMINAMATH_CALUDE_system_range_of_a_l824_82479

/-- Given a system of linear equations in x and y, prove the range of a -/
theorem system_range_of_a (x y a : ℝ) 
  (eq1 : x + 3*y = 2 + a) 
  (eq2 : 3*x + y = -4*a) 
  (h : x + y > 2) : 
  a < -2 := by
sorry

end NUMINAMATH_CALUDE_system_range_of_a_l824_82479


namespace NUMINAMATH_CALUDE_line_through_coefficient_points_l824_82485

/-- Given two lines passing through a common point, prove that the line
    passing through the points defined by their coefficients has a specific equation. -/
theorem line_through_coefficient_points
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h₂ : 2 * a₂ + 3 * b₂ + 1 = 0) :
  (fun x y : ℝ => 2 * x + 3 * y + 1 = 0) a₁ b₁ ∧ 
  (fun x y : ℝ => 2 * x + 3 * y + 1 = 0) a₂ b₂ := by
  sorry

#check line_through_coefficient_points

end NUMINAMATH_CALUDE_line_through_coefficient_points_l824_82485


namespace NUMINAMATH_CALUDE_problem_statement_l824_82435

theorem problem_statement : (-12 : ℚ) * ((2 : ℚ) / 3 - (1 : ℚ) / 4 + (1 : ℚ) / 6) = -7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l824_82435


namespace NUMINAMATH_CALUDE_excess_purchase_l824_82423

/-- Calculates the excess amount of Chinese herbal medicine purchased given the planned amount and completion percentages -/
theorem excess_purchase (planned_amount : ℝ) (first_half_percent : ℝ) (second_half_percent : ℝ) 
  (h1 : planned_amount = 1500)
  (h2 : first_half_percent = 55)
  (h3 : second_half_percent = 65) :
  (first_half_percent + second_half_percent - 100) / 100 * planned_amount = 300 := by
  sorry

end NUMINAMATH_CALUDE_excess_purchase_l824_82423


namespace NUMINAMATH_CALUDE_gcd_A_B_eq_one_l824_82492

def A : ℤ := 1989^1990 - 1988^1990
def B : ℤ := 1989^1989 - 1988^1989

theorem gcd_A_B_eq_one : Int.gcd A B = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_A_B_eq_one_l824_82492


namespace NUMINAMATH_CALUDE_infinite_nested_sqrt_l824_82471

/-- Given that y is a non-negative real number satisfying y = √(2 - y), prove that y = 1 -/
theorem infinite_nested_sqrt (y : ℝ) (hy : y ≥ 0) (h : y = Real.sqrt (2 - y)) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_nested_sqrt_l824_82471


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l824_82434

theorem circular_seating_arrangement (n : ℕ) (h1 : n ≤ 6) (h2 : Nat.factorial (n - 1) = 144) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l824_82434


namespace NUMINAMATH_CALUDE_frequency_count_theorem_l824_82495

theorem frequency_count_theorem (sample_size : ℕ) (relative_frequency : ℝ) 
  (h1 : sample_size = 100) 
  (h2 : relative_frequency = 0.2) :
  (sample_size : ℝ) * relative_frequency = 20 := by
  sorry

end NUMINAMATH_CALUDE_frequency_count_theorem_l824_82495


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l824_82444

/-- Given that (2x-1)^5 + (x+2)^4 = a + a₁x + a₂x² + a₃x³ + a₄x⁴ + a₅x⁵,
    prove that |a| + |a₂| + |a₄| = 30 -/
theorem sum_of_absolute_coefficients (x a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (2*x - 1)^5 + (x + 2)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 →
  |a| + |a₂| + |a₄| = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l824_82444


namespace NUMINAMATH_CALUDE_karls_total_income_l824_82445

/-- Represents the prices of items in Karl's store -/
structure Prices where
  tshirt : ℚ
  pants : ℚ
  skirt : ℚ
  refurbished_tshirt : ℚ

/-- Represents the quantities of items sold -/
structure QuantitiesSold where
  tshirt : ℕ
  pants : ℕ
  skirt : ℕ
  refurbished_tshirt : ℕ

/-- Calculates the total income given prices and quantities sold -/
def totalIncome (prices : Prices) (quantities : QuantitiesSold) : ℚ :=
  prices.tshirt * quantities.tshirt +
  prices.pants * quantities.pants +
  prices.skirt * quantities.skirt +
  prices.refurbished_tshirt * quantities.refurbished_tshirt

/-- Theorem stating that Karl's total income is $53 -/
theorem karls_total_income :
  let prices : Prices := {
    tshirt := 5,
    pants := 4,
    skirt := 6,
    refurbished_tshirt := 5/2
  }
  let quantities : QuantitiesSold := {
    tshirt := 2,
    pants := 1,
    skirt := 4,
    refurbished_tshirt := 6
  }
  totalIncome prices quantities = 53 := by
  sorry


end NUMINAMATH_CALUDE_karls_total_income_l824_82445
