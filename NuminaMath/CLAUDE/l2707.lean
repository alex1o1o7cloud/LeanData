import Mathlib

namespace NUMINAMATH_CALUDE_sum_x_z_equals_4036_l2707_270710

theorem sum_x_z_equals_4036 (x y z : ℝ) 
  (eq1 : x + y + z = 0)
  (eq2 : 2016 * x + 2017 * y + 2018 * z = 0)
  (eq3 : 2016^2 * x + 2017^2 * y + 2018^2 * z = 2018) :
  x + z = 4036 := by sorry

end NUMINAMATH_CALUDE_sum_x_z_equals_4036_l2707_270710


namespace NUMINAMATH_CALUDE_parabola_directrix_l2707_270741

/-- A parabola with equation x = -1/4 * y^2 has a directrix with equation x = 1 -/
theorem parabola_directrix (y : ℝ) : 
  (∃ x : ℝ, x = -(1/4) * y^2) → 
  (∃ d : ℝ, d = 1 ∧ ∀ p : ℝ × ℝ, p.1 = d ↔ p ∈ {q : ℝ × ℝ | q.1 = 1}) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2707_270741


namespace NUMINAMATH_CALUDE_total_apples_buyable_l2707_270719

def apple_cost : ℕ := 2
def emmy_money : ℕ := 200
def gerry_money : ℕ := 100

theorem total_apples_buyable : 
  (emmy_money + gerry_money) / apple_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_total_apples_buyable_l2707_270719


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2707_270700

theorem fractional_equation_solution : 
  ∃ x : ℝ, (3 - x) / (x - 4) + 1 / (4 - x) = 1 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2707_270700


namespace NUMINAMATH_CALUDE_find_x_l2707_270713

theorem find_x (y : ℝ) (x : ℝ) (h1 : (12 : ℝ)^3 * 6^3 / x = y) (h2 : y = 864) : x = 432 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2707_270713


namespace NUMINAMATH_CALUDE_f_3_range_l2707_270707

-- Define the function f(x) = a x^2 - c
def f (a c x : ℝ) : ℝ := a * x^2 - c

-- State the theorem
theorem f_3_range (a c : ℝ) :
  (∀ x : ℝ, f a c x = a * x^2 - c) →
  (-4 ≤ f a c 1 ∧ f a c 1 ≤ -1) →
  (-1 ≤ f a c 2 ∧ f a c 2 ≤ 5) →
  (-1 ≤ f a c 3 ∧ f a c 3 ≤ 20) :=
by sorry

end NUMINAMATH_CALUDE_f_3_range_l2707_270707


namespace NUMINAMATH_CALUDE_square_of_negative_product_l2707_270729

theorem square_of_negative_product (a b : ℝ) : (-2 * a * b)^2 = 4 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l2707_270729


namespace NUMINAMATH_CALUDE_range_of_x_l2707_270766

theorem range_of_x (x : ℝ) : 
  (1 / x < 3) → (1 / x > -2) → (2 * x - 5 > 0) → (x > 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2707_270766


namespace NUMINAMATH_CALUDE_expression_bounds_l2707_270755

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l2707_270755


namespace NUMINAMATH_CALUDE_exchange_count_l2707_270711

def number_of_people : ℕ := 10

def business_card_exchanges (n : ℕ) : ℕ := n.choose 2

theorem exchange_count : business_card_exchanges number_of_people = 45 := by
  sorry

end NUMINAMATH_CALUDE_exchange_count_l2707_270711


namespace NUMINAMATH_CALUDE_profit_to_cost_ratio_l2707_270791

/-- Given an article with a sale price and cost price, this theorem proves
    that if the ratio of sale price to cost price is 6:2,
    then the ratio of profit to cost price is 2:1. -/
theorem profit_to_cost_ratio
  (sale_price cost_price : ℚ)
  (h : sale_price / cost_price = 6 / 2) :
  (sale_price - cost_price) / cost_price = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_profit_to_cost_ratio_l2707_270791


namespace NUMINAMATH_CALUDE_x_is_perfect_square_l2707_270753

/-- The sequence x_n as defined in the problem -/
def x : ℕ → ℚ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 1
  | (n + 4) => ((n^2 + n + 1) * (n + 1) / n) * x (n + 3) + 
               (n^2 + n + 1) * x (n + 2) - 
               ((n + 1) / n) * x (n + 1)

/-- The theorem stating that all members of x_n are perfect squares -/
theorem x_is_perfect_square : ∀ n : ℕ, ∃ y : ℤ, x n = (y : ℚ)^2 := by
  sorry

end NUMINAMATH_CALUDE_x_is_perfect_square_l2707_270753


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2707_270734

theorem circle_area_from_circumference (c : ℝ) (h : c = 24) :
  let r := c / (2 * Real.pi)
  (Real.pi * r * r) = 144 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2707_270734


namespace NUMINAMATH_CALUDE_pet_store_cages_l2707_270759

theorem pet_store_cages (birds_per_cage : ℕ) (total_birds : ℕ) (num_cages : ℕ) : 
  birds_per_cage = 8 → 
  total_birds = 48 → 
  num_cages * birds_per_cage = total_birds → 
  num_cages = 6 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2707_270759


namespace NUMINAMATH_CALUDE_arbor_day_tree_planting_l2707_270708

theorem arbor_day_tree_planting 
  (original_trees_per_row : ℕ) 
  (original_rows : ℕ) 
  (new_rows : ℕ) 
  (h1 : original_trees_per_row = 20) 
  (h2 : original_rows = 18) 
  (h3 : new_rows = 10) : 
  (original_trees_per_row * original_rows) / new_rows = 36 := by
sorry

end NUMINAMATH_CALUDE_arbor_day_tree_planting_l2707_270708


namespace NUMINAMATH_CALUDE_log_difference_cube_l2707_270726

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the theorem
theorem log_difference_cube (x y a : ℝ) (h : lg x - lg y = a) :
  lg ((x/2)^3) - lg ((y/2)^3) = 3*a := by
  sorry

end NUMINAMATH_CALUDE_log_difference_cube_l2707_270726


namespace NUMINAMATH_CALUDE_basketball_substitutions_l2707_270705

/-- The number of possible substitution methods in a basketball game with specific rules -/
def substitution_methods (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitute_players := total_players - starting_players
  1 + -- No substitutions
  (starting_players * substitute_players) + -- One substitution
  (starting_players * (starting_players - 1) * substitute_players * (substitute_players - 1) / 2) + -- Two substitutions
  (starting_players * (starting_players - 1) * (starting_players - 2) * substitute_players * (substitute_players - 1) * (substitute_players - 2) / 6) -- Three substitutions

/-- The main theorem stating the number of substitution methods and its remainder when divided by 1000 -/
theorem basketball_substitutions :
  let m := substitution_methods 18 9 3
  m = 45010 ∧ m % 1000 = 10 := by
  sorry


end NUMINAMATH_CALUDE_basketball_substitutions_l2707_270705


namespace NUMINAMATH_CALUDE_fifth_friend_payment_l2707_270781

def boat_purchase (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 120 ∧
  a = (1/3) * (b + c + d + e) ∧
  b = (1/4) * (a + c + d + e) ∧
  c = (1/5) * (a + b + d + e)

theorem fifth_friend_payment :
  ∃ a b c d : ℝ, boat_purchase a b c d 13 :=
sorry

end NUMINAMATH_CALUDE_fifth_friend_payment_l2707_270781


namespace NUMINAMATH_CALUDE_arithmetic_progression_square_sum_l2707_270769

def is_four_identical_digits (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 9 ∧ n = k * 1111

theorem arithmetic_progression_square_sum (n : ℕ) : 
  is_four_identical_digits ((n - 2)^2 + n^2 + (n + 2)^2) ↔ n = 43 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_square_sum_l2707_270769


namespace NUMINAMATH_CALUDE_compute_fraction_expression_l2707_270754

theorem compute_fraction_expression : 8 * (1/3)^2 * (2/7) = 16/63 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_expression_l2707_270754


namespace NUMINAMATH_CALUDE_smaller_solution_cube_root_equation_l2707_270784

theorem smaller_solution_cube_root_equation (x : ℝ) :
  (Real.rpow x (1/3 : ℝ) + Real.rpow (16 - x) (1/3 : ℝ) = 2) →
  (x = (1 - Real.sqrt 21 / 3)^3 ∨ x = (1 + Real.sqrt 21 / 3)^3) ∧
  x ≤ (1 + Real.sqrt 21 / 3)^3 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_cube_root_equation_l2707_270784


namespace NUMINAMATH_CALUDE_square_area_larger_than_circle_l2707_270730

theorem square_area_larger_than_circle (R : ℝ) (h : R > 0) : 
  let AB := 2 * R * Real.sin (3 * Real.pi / 8)
  (AB ^ 2) > Real.pi * R ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_larger_than_circle_l2707_270730


namespace NUMINAMATH_CALUDE_item_list_price_equality_l2707_270720

theorem item_list_price_equality (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) → list_price = 40 := by
  sorry

#check item_list_price_equality

end NUMINAMATH_CALUDE_item_list_price_equality_l2707_270720


namespace NUMINAMATH_CALUDE_racing_track_circumference_difference_l2707_270739

theorem racing_track_circumference_difference
  (r : ℝ)
  (inner_radius : ℝ)
  (outer_radius : ℝ)
  (track_width : ℝ)
  (h1 : inner_radius = 2 * r)
  (h2 : outer_radius = inner_radius + track_width)
  (h3 : track_width = 15)
  : 2 * Real.pi * outer_radius - 2 * Real.pi * inner_radius = 30 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_racing_track_circumference_difference_l2707_270739


namespace NUMINAMATH_CALUDE_moving_circle_theorem_l2707_270773

-- Define the moving circle
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_A : center.1 ^ 2 + center.2 ^ 2 = (center.1 - 2) ^ 2 + center.2 ^ 2
  cuts_y_axis : ∃ (y : ℝ), center.1 ^ 2 + (y - center.2) ^ 2 = center.1 ^ 2 + center.2 ^ 2 ∧ y ^ 2 = 4

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define the fixed point N
structure FixedPointN where
  x₀ : ℝ

-- Define the chord BD
structure ChordBD (n : FixedPointN) where
  m : ℝ
  passes_through_N : ∀ (y : ℝ), trajectory (n.x₀ + m * y) y

-- Define the angle BAD
def angle_BAD_obtuse (n : FixedPointN) (bd : ChordBD n) : Prop :=
  ∀ (y₁ y₂ : ℝ), 
    trajectory (n.x₀ + bd.m * y₁) y₁ → 
    trajectory (n.x₀ + bd.m * y₂) y₂ → 
    (n.x₀ + bd.m * y₁ - 2) * (n.x₀ + bd.m * y₂ - 2) + y₁ * y₂ < 0

-- The main theorem
theorem moving_circle_theorem :
  (∀ (mc : MovingCircle), trajectory mc.center.1 mc.center.2) ∧
  (∀ (n : FixedPointN), 
    (∀ (bd : ChordBD n), angle_BAD_obtuse n bd) → 
    (4 - 2 * Real.sqrt 3 < n.x₀ ∧ n.x₀ < 4 + 2 * Real.sqrt 3 ∧ n.x₀ ≠ 2)) :=
sorry

end NUMINAMATH_CALUDE_moving_circle_theorem_l2707_270773


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l2707_270747

theorem chocolate_box_problem (C : ℝ) : 
  C > 0 →  -- Ensure the number of chocolates is positive
  (C / 2 - 0.8 * (C / 2)) + (C / 2 - 0.5 * (C / 2)) = 28 →
  C = 80 := by
sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l2707_270747


namespace NUMINAMATH_CALUDE_square_vertices_distance_sum_l2707_270772

/-- Given a square with side length s, prove that a point P(x,y) satisfies the condition that the sum
    of squares of distances from P to each vertex is 4s² if and only if P lies on a circle centered
    at (s/2, s/2) with radius s/√2 -/
theorem square_vertices_distance_sum (s : ℝ) (x y : ℝ) :
  (x^2 + y^2) + (x^2 + (y - s)^2) + ((x - s)^2 + y^2) + ((x - s)^2 + (y - s)^2) = 4 * s^2 ↔
  (x - s/2)^2 + (y - s/2)^2 = (s/Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_square_vertices_distance_sum_l2707_270772


namespace NUMINAMATH_CALUDE_installation_cost_is_6255_l2707_270721

/-- Calculates the installation cost for a refrigerator purchase --/
def calculate_installation_cost (purchase_price : ℚ) (discount_rate : ℚ) 
  (transport_cost : ℚ) (profit_rate : ℚ) (selling_price : ℚ) : ℚ :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let total_cost := selling_price / (1 + profit_rate)
  total_cost - purchase_price - transport_cost

/-- Proves that the installation cost is 6255, given the problem conditions --/
theorem installation_cost_is_6255 :
  calculate_installation_cost 12500 0.20 125 0.18 18880 = 6255 := by
  sorry

end NUMINAMATH_CALUDE_installation_cost_is_6255_l2707_270721


namespace NUMINAMATH_CALUDE_correct_answers_for_given_score_l2707_270714

/-- Represents a test result -/
structure TestResult where
  totalQuestions : ℕ
  correctAnswers : ℕ
  score : ℤ

/-- Calculates the score based on correct and incorrect answers -/
def calculateScore (correct incorrect : ℕ) : ℤ :=
  (correct : ℤ) - 2 * (incorrect : ℤ)

theorem correct_answers_for_given_score 
  (result : TestResult) 
  (h1 : result.totalQuestions = 100)
  (h2 : result.score = calculateScore result.correctAnswers (result.totalQuestions - result.correctAnswers))
  (h3 : result.score = 76) :
  result.correctAnswers = 92 := by
  sorry


end NUMINAMATH_CALUDE_correct_answers_for_given_score_l2707_270714


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l2707_270737

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_example : 
  parallelogram_area 10 20 = 200 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l2707_270737


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_is_one_l2707_270732

-- Define the polynomial expression
def poly (x : ℝ) : ℝ := 2 * (x^3 - 2*x^2) + 3 * (x^2 - x^3 + x^4) - (5*x^4 - 2*x^3)

-- Theorem stating that the coefficient of x^3 in the expanded form of poly is 1
theorem x_cubed_coefficient_is_one :
  ∃ a b c d, ∀ x, poly x = a*x^4 + b*x^3 + c*x^2 + d*x ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_is_one_l2707_270732


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2707_270796

theorem quadratic_minimum_value : 
  (∀ x : ℝ, x^2 + 4*x + 5 ≥ 1) ∧ (∃ x : ℝ, x^2 + 4*x + 5 = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2707_270796


namespace NUMINAMATH_CALUDE_smallest_integer_a_l2707_270762

theorem smallest_integer_a (a b : ℤ) : 
  (∃ k : ℤ, a > k ∧ a < 21) →
  (b > 19 ∧ b < 31) →
  (a / b : ℚ) ≤ 2/3 →
  (∀ m : ℤ, m < a → m ≤ 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_a_l2707_270762


namespace NUMINAMATH_CALUDE_time_to_find_two_artifacts_l2707_270715

/-- The time it takes to find two artifacts given research and expedition times for the first, 
    and a multiplier for the second. -/
def time_to_find_artifacts (research_time : ℝ) (expedition_time : ℝ) (multiplier : ℝ) : ℝ :=
  let first_artifact_time := research_time + expedition_time
  let second_artifact_time := multiplier * first_artifact_time
  first_artifact_time + second_artifact_time

/-- Theorem stating that under the given conditions, it takes 10 years to find both artifacts. -/
theorem time_to_find_two_artifacts : 
  time_to_find_artifacts 0.5 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_time_to_find_two_artifacts_l2707_270715


namespace NUMINAMATH_CALUDE_product_of_differences_divisible_by_12_l2707_270771

theorem product_of_differences_divisible_by_12 (a b c d : ℤ) :
  ∃ k : ℤ, (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_divisible_by_12_l2707_270771


namespace NUMINAMATH_CALUDE_gold_balance_fraction_is_one_third_l2707_270736

/-- Represents a credit card with a spending limit and balance. -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards and their properties. -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard
  gold_balance_fraction : ℝ
  platinum_balance_fraction : ℝ
  remaining_platinum_fraction : ℝ

/-- Theorem stating the fraction of the gold card's limit that represents the current balance. -/
theorem gold_balance_fraction_is_one_third
  (cards : SallysCards)
  (h1 : cards.platinum.limit = 2 * cards.gold.limit)
  (h2 : cards.gold.balance = cards.gold_balance_fraction * cards.gold.limit)
  (h3 : cards.platinum.balance = (1/4) * cards.platinum.limit)
  (h4 : cards.remaining_platinum_fraction = 0.5833333333333334)
  (h5 : cards.platinum.limit - (cards.platinum.balance + cards.gold.balance) =
        cards.remaining_platinum_fraction * cards.platinum.limit) :
  cards.gold_balance_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_gold_balance_fraction_is_one_third_l2707_270736


namespace NUMINAMATH_CALUDE_sum_45_25_in_base5_l2707_270777

/-- Converts a decimal number to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a base 5 number to decimal -/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two base 5 numbers -/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_45_25_in_base5 :
  let a := 45
  let b := 25
  let a_base5 := toBase5 a
  let b_base5 := toBase5 b
  let sum_base5 := addBase5 a_base5 b_base5
  sum_base5 = [2, 3, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_45_25_in_base5_l2707_270777


namespace NUMINAMATH_CALUDE_right_triangle_altitude_ratio_l2707_270761

theorem right_triangle_altitude_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_ratio : b = (3/2) * a) (d : ℝ) (h_altitude : d^2 = (a*b)/c) :
  (c-d)/d = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_ratio_l2707_270761


namespace NUMINAMATH_CALUDE_expression_value_l2707_270779

theorem expression_value (x y : ℝ) (h : x - 2*y + 3 = 0) : 1 - 2*x + 4*y = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2707_270779


namespace NUMINAMATH_CALUDE_cookies_left_l2707_270712

theorem cookies_left (total_cookies : ℕ) (num_neighbors : ℕ) (intended_per_neighbor : ℕ) 
  (sarah_cookies : ℕ) (h1 : total_cookies = 150) (h2 : num_neighbors = 15) 
  (h3 : intended_per_neighbor = 10) (h4 : sarah_cookies = 12) : 
  total_cookies - (intended_per_neighbor * (num_neighbors - 1)) - sarah_cookies = 8 := by
  sorry

#check cookies_left

end NUMINAMATH_CALUDE_cookies_left_l2707_270712


namespace NUMINAMATH_CALUDE_alloy_ratio_proof_l2707_270789

/-- Proves that the ratio of lead to tin in alloy A is 2:3 given the specified conditions -/
theorem alloy_ratio_proof (alloy_A_weight : ℝ) (alloy_B_weight : ℝ) 
  (tin_copper_ratio_B : ℚ) (total_tin_new_alloy : ℝ) 
  (h1 : alloy_A_weight = 120)
  (h2 : alloy_B_weight = 180)
  (h3 : tin_copper_ratio_B = 3/5)
  (h4 : total_tin_new_alloy = 139.5) :
  ∃ (lead_A tin_A : ℝ), 
    lead_A + tin_A = alloy_A_weight ∧ 
    lead_A / tin_A = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_alloy_ratio_proof_l2707_270789


namespace NUMINAMATH_CALUDE_sum_of_triangle_perimeters_l2707_270782

/-- Given an equilateral triangle with side length 45 cm, if we repeatedly form new equilateral
    triangles by joining the midpoints of the previous triangle's sides, the sum of the perimeters
    of all these triangles is 270 cm. -/
theorem sum_of_triangle_perimeters (s : ℝ) (h : s = 45) :
  let perimeter_sum := (3 * s) / (1 - (1/2 : ℝ))
  perimeter_sum = 270 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangle_perimeters_l2707_270782


namespace NUMINAMATH_CALUDE_red_jellybeans_count_l2707_270742

def total_jellybeans : ℕ := 200
def blue_jellybeans : ℕ := 14
def purple_jellybeans : ℕ := 26
def orange_jellybeans : ℕ := 40

theorem red_jellybeans_count :
  total_jellybeans - (blue_jellybeans + purple_jellybeans + orange_jellybeans) = 120 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybeans_count_l2707_270742


namespace NUMINAMATH_CALUDE_division_reciprocal_l2707_270716

theorem division_reciprocal (a b c d e : ℝ) (ha : a ≠ 0) (hbcde : b - c + d - e ≠ 0) :
  a / (b - c + d - e) = 1 / ((b - c + d - e) / a) := by
  sorry

end NUMINAMATH_CALUDE_division_reciprocal_l2707_270716


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l2707_270733

theorem angle_sum_around_point (x : ℝ) : 
  (3 * x + 7 * x + 4 * x + x = 360) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l2707_270733


namespace NUMINAMATH_CALUDE_nut_distribution_theorem_l2707_270778

/-- Represents a distribution of nuts among three piles -/
structure NutDistribution :=
  (pile1 pile2 pile3 : ℕ)

/-- Represents an operation of moving nuts between piles -/
inductive MoveOperation
  | move12 : MoveOperation  -- Move from pile 1 to pile 2
  | move13 : MoveOperation  -- Move from pile 1 to pile 3
  | move21 : MoveOperation  -- Move from pile 2 to pile 1
  | move23 : MoveOperation  -- Move from pile 2 to pile 3
  | move31 : MoveOperation  -- Move from pile 3 to pile 1
  | move32 : MoveOperation  -- Move from pile 3 to pile 2

/-- Applies a single move operation to a distribution -/
def applyMove (d : NutDistribution) (m : MoveOperation) : NutDistribution :=
  sorry

/-- Checks if a pile has an even number of nuts -/
def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Checks if a distribution has the desired property (one pile with half the nuts) -/
def hasHalfInOnePile (d : NutDistribution) : Prop :=
  let total := d.pile1 + d.pile2 + d.pile3
  d.pile1 = total / 2 ∨ d.pile2 = total / 2 ∨ d.pile3 = total / 2

/-- The main theorem statement -/
theorem nut_distribution_theorem (initial : NutDistribution) :
  isEven (initial.pile1 + initial.pile2 + initial.pile3) →
  ∃ (moves : List MoveOperation), 
    hasHalfInOnePile (moves.foldl applyMove initial) :=
by sorry

end NUMINAMATH_CALUDE_nut_distribution_theorem_l2707_270778


namespace NUMINAMATH_CALUDE_maynards_dog_holes_l2707_270750

theorem maynards_dog_holes : 
  ∀ (total : ℕ) (filled : ℕ) (unfilled : ℕ),
    filled = (75 * total) / 100 →
    unfilled = 2 →
    total = filled + unfilled →
    total = 8 := by
  sorry

end NUMINAMATH_CALUDE_maynards_dog_holes_l2707_270750


namespace NUMINAMATH_CALUDE_other_roots_form_new_equation_l2707_270703

theorem other_roots_form_new_equation (a₁ a₂ a₃ : ℝ) :
  let eq1 := fun x => x^2 + a₁*x + a₂*a₃
  let eq2 := fun x => x^2 + a₂*x + a₁*a₃
  let eq3 := fun x => x^2 + a₃*x + a₁*a₂
  (∃! α, eq1 α = 0 ∧ eq2 α = 0) →
  ∃ β γ, eq1 β = 0 ∧ eq2 γ = 0 ∧ β ≠ γ ∧ eq3 β = 0 ∧ eq3 γ = 0 :=
by sorry


end NUMINAMATH_CALUDE_other_roots_form_new_equation_l2707_270703


namespace NUMINAMATH_CALUDE_smallest_m_is_251_l2707_270764

/-- Represents a circular arrangement of grids with real numbers -/
def CircularGrids (n : ℕ) := Fin n → ℝ

/-- Checks if the difference condition is satisfied for a given grid and step -/
def satisfiesDifferenceCondition (grids : CircularGrids 999) (a k : Fin 999) : Prop :=
  (grids a - grids ((a + k) % 999) = k) ∨ (grids a - grids ((999 + a - k) % 999) = k)

/-- Checks if the consecutive condition is satisfied for a given starting grid -/
def satisfiesConsecutiveCondition (grids : CircularGrids 999) (s : Fin 999) : Prop :=
  (∀ k : Fin 998, grids ((s + k) % 999) = grids s + k) ∨
  (∀ k : Fin 998, grids ((999 + s - k) % 999) = grids s + k)

/-- The main theorem stating that 251 is the smallest positive integer satisfying the conditions -/
theorem smallest_m_is_251 : 
  ∀ m : ℕ+, 
    (m = 251 ↔ 
      (∀ grids : CircularGrids 999, 
        (∀ a : Fin 999, ∀ k : Fin m, satisfiesDifferenceCondition grids a k) →
        (∃ s : Fin 999, satisfiesConsecutiveCondition grids s)) ∧
      (∀ m' : ℕ+, m' < m →
        ∃ grids : CircularGrids 999, 
          (∀ a : Fin 999, ∀ k : Fin m', satisfiesDifferenceCondition grids a k) ∧
          (∀ s : Fin 999, ¬satisfiesConsecutiveCondition grids s))) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_251_l2707_270764


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2707_270722

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2707_270722


namespace NUMINAMATH_CALUDE_equation_has_four_real_solutions_l2707_270717

theorem equation_has_four_real_solutions :
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, (5 * x) / (x^2 + 2*x + 4) + (7 * x) / (x^2 - 7*x + 4) = -2) ∧ 
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_four_real_solutions_l2707_270717


namespace NUMINAMATH_CALUDE_prime_satisfying_condition_l2707_270751

def satisfies_condition (p : Nat) : Prop :=
  Nat.Prime p ∧
  ∀ q : Nat, Nat.Prime q → q < p →
    ∀ k r : Nat, p = k * q + r → 0 ≤ r → r < q →
      ∀ a : Nat, a > 1 → ¬(a^2 ∣ r)

theorem prime_satisfying_condition :
  {p : Nat | satisfies_condition p} = {2, 3, 5, 7, 13} := by sorry

end NUMINAMATH_CALUDE_prime_satisfying_condition_l2707_270751


namespace NUMINAMATH_CALUDE_large_loans_required_l2707_270785

/-- Represents the number of loans of each type required to buy an apartment -/
structure LoanCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Two equivalent ways to buy the apartment -/
def way1 : LoanCombination := { small := 9, medium := 6, large := 1 }
def way2 : LoanCombination := { small := 3, medium := 2, large := 3 }

/-- The theorem states that 4 large loans are required to buy the apartment -/
theorem large_loans_required : ∃ (n : ℕ), n = 4 ∧ 
  way1.small * n + way1.medium * n + way1.large * n = 
  way2.small * n + way2.medium * n + way2.large * n :=
sorry

end NUMINAMATH_CALUDE_large_loans_required_l2707_270785


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2707_270793

theorem triangle_angle_sum (A B C : Real) (h1 : A = 30) (h2 : B = 50) :
  C = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2707_270793


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2707_270748

/-- Given a quadratic equation x^2 - (k+3)x + 2k + 2 = 0, prove:
    1. The equation always has two real roots
    2. When one root is positive and less than 1, -1 < k < 0 -/
theorem quadratic_equation_properties (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (k+3)*x + 2*k + 2
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 → -1 < k ∧ k < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2707_270748


namespace NUMINAMATH_CALUDE_system_solution_l2707_270757

theorem system_solution :
  let x : ℝ := -1
  let y : ℝ := (Real.sqrt 3 + 1) / 2
  (Real.sqrt 3 * x + 2 * y = 1) ∧ (x + 2 * y = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2707_270757


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2707_270780

theorem floor_ceil_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2707_270780


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2707_270731

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^3 + 3*x - 5) - 7*(2*x^2 + x - 8) = 8*x^4 - 8*x^2 - 17*x + 56 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2707_270731


namespace NUMINAMATH_CALUDE_statement_I_statement_II_statement_III_statement_IV_l2707_270735

-- Define the complex square root function
noncomputable def complexSqrt : ℂ → ℂ := sorry

-- Statement (I)
theorem statement_I (a b : ℂ) : complexSqrt (a^2 + b^2) = 0 ↔ a = 0 ∧ b = 0 := by sorry

-- Statement (II)
theorem statement_II : ¬∃ (a b : ℂ), (a ≠ 0 ∨ b ≠ 0) ∧ complexSqrt (a^2 + b^2) = a * b := by sorry

-- Statement (III)
theorem statement_III : ¬∃ (a b : ℂ), (a ≠ 0 ∨ b ≠ 0) ∧ complexSqrt (a^2 + b^2) = a + b := by sorry

-- Statement (IV)
theorem statement_IV : ¬∃ (a b : ℂ), (a ≠ 0 ∨ b ≠ 0) ∧ complexSqrt (a^2 + b^2) = a * b := by sorry

end NUMINAMATH_CALUDE_statement_I_statement_II_statement_III_statement_IV_l2707_270735


namespace NUMINAMATH_CALUDE_worker_ant_ratio_l2707_270727

theorem worker_ant_ratio (total_ants : ℕ) (female_worker_ants : ℕ) 
  (h1 : total_ants = 110)
  (h2 : female_worker_ants = 44)
  (h3 : (female_worker_ants : ℚ) / (female_worker_ants / 0.8 : ℚ) = 0.8) :
  (female_worker_ants / 0.8 : ℚ) / (total_ants : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_worker_ant_ratio_l2707_270727


namespace NUMINAMATH_CALUDE_temperature_difference_l2707_270725

/-- The temperature difference problem -/
theorem temperature_difference
  (morning_temp : ℝ)
  (noon_rise : ℝ)
  (night_drop : ℝ)
  (h_morning : morning_temp = 7)
  (h_noon_rise : noon_rise = 9)
  (h_night_drop : night_drop = 13)
  (h_highest : morning_temp + noon_rise = max morning_temp (morning_temp + noon_rise))
  (h_lowest : morning_temp + noon_rise - night_drop = min (morning_temp + noon_rise) (morning_temp + noon_rise - night_drop)) :
  (morning_temp + noon_rise) - (morning_temp + noon_rise - night_drop) = 13 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l2707_270725


namespace NUMINAMATH_CALUDE_trevor_coin_conversion_l2707_270702

/-- Represents the types of coins in the problem -/
inductive Coin
  | Quarter
  | Dime
  | Nickel
  | Penny

/-- Calculates the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- Represents the coin count in Trevor's bank -/
structure CoinCount where
  total : ℕ
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (cc : CoinCount) : ℕ :=
  cc.quarters * coinValue Coin.Quarter +
  cc.dimes * coinValue Coin.Dime +
  cc.nickels * coinValue Coin.Nickel +
  cc.pennies * coinValue Coin.Penny

/-- Converts total value to $5 bills and $1 coins -/
def convertToBillsAndCoins (value : ℕ) : (ℕ × ℕ) :=
  (value / 500, (value % 500) / 100)

theorem trevor_coin_conversion :
  let cc : CoinCount := {
    total := 153,
    quarters := 45,
    dimes := 34,
    nickels := 19,
    pennies := 153 - 45 - 34 - 19
  }
  let (fiveBills, oneDollars) := convertToBillsAndCoins (totalValue cc)
  fiveBills - oneDollars = 2 := by sorry

end NUMINAMATH_CALUDE_trevor_coin_conversion_l2707_270702


namespace NUMINAMATH_CALUDE_domino_set_0_to_12_l2707_270744

/-- The number of tiles in a domino set with values from 0 to n -/
def dominoCount (n : ℕ) : ℕ := Nat.choose (n + 1) 2

/-- The number of tiles in a standard domino set (0 to 6) -/
def standardDominoCount : ℕ := 28

theorem domino_set_0_to_12 : dominoCount 12 = 91 := by sorry

end NUMINAMATH_CALUDE_domino_set_0_to_12_l2707_270744


namespace NUMINAMATH_CALUDE_smallest_in_A_l2707_270790

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the set A
def A : Set ℕ := {n | 11 ∣ sumOfDigits n ∧ 11 ∣ sumOfDigits (n + 1)}

-- State the theorem
theorem smallest_in_A : 
  2899999 ∈ A ∧ ∀ m ∈ A, m < 2899999 → m = 2899999 := by sorry

end NUMINAMATH_CALUDE_smallest_in_A_l2707_270790


namespace NUMINAMATH_CALUDE_quadratic_equation_no_real_roots_l2707_270794

theorem quadratic_equation_no_real_roots 
  (p q a b c : ℝ) 
  (hp : p > 0) (hq : q > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hpq : p ≠ q)
  (hgeom : a^2 = p*q)  -- Geometric sequence condition
  (harith1 : b - p = c - b) (harith2 : c - b = q - c)  -- Arithmetic sequence conditions
  : (2*a)^2 - 4*b*c < 0 := by
  sorry

#check quadratic_equation_no_real_roots

end NUMINAMATH_CALUDE_quadratic_equation_no_real_roots_l2707_270794


namespace NUMINAMATH_CALUDE_problem_solution_l2707_270724

theorem problem_solution (a b : ℝ) (h : Real.sqrt (a - 3) + abs (4 - b) = 0) :
  (a - b) ^ 2023 = -1 ∧ ∀ x n : ℝ, x > 0 → Real.sqrt x = a + n → Real.sqrt x = b - 2*n → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2707_270724


namespace NUMINAMATH_CALUDE_zachary_did_more_pushups_l2707_270743

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 51

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The difference in push-ups between Zachary and David -/
def pushup_difference : ℕ := zachary_pushups - david_pushups

theorem zachary_did_more_pushups : pushup_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_zachary_did_more_pushups_l2707_270743


namespace NUMINAMATH_CALUDE_max_value_of_g_l2707_270765

/-- Given f(x) = sin x + a cos x with a symmetry axis at x = 5π/3,
    prove that the maximum value of g(x) = a sin x + cos x is 2√3/3 -/
theorem max_value_of_g (a : ℝ) (f g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x + a * Real.cos x)
    (h₂ : ∀ x, f x = f (10 * Real.pi / 3 - x))
    (h₃ : ∀ x, g x = a * Real.sin x + Real.cos x) :
    (∀ x, g x ≤ 2 * Real.sqrt 3 / 3) ∧ ∃ x, g x = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2707_270765


namespace NUMINAMATH_CALUDE_external_angle_bisectors_collinear_l2707_270788

-- Define the basic structures
structure Point := (x : ℝ) (y : ℝ)
structure Line := (a : ℝ) (b : ℝ) (c : ℝ) -- ax + by + c = 0

-- Define the quadrilateral
structure Quadrilateral :=
  (A B C D : Point)
  (is_convex : Bool)

-- Define the intersection points of side extensions
def extension_intersections (q : Quadrilateral) : Point × Point := sorry

-- Define the external angle bisector
def external_angle_bisector (p1 p2 p3 : Point) : Line := sorry

-- Define collinearity
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Main theorem
theorem external_angle_bisectors_collinear (q : Quadrilateral) :
  let (P, Q) := extension_intersections q
  let AC_bisector := external_angle_bisector q.A q.C P
  let BD_bisector := external_angle_bisector q.B q.D Q
  let PQ_bisector := external_angle_bisector P Q q.A
  let I1 := sorry -- Intersection of AC_bisector and BD_bisector
  let I2 := sorry -- Intersection of BD_bisector and PQ_bisector
  let I3 := sorry -- Intersection of PQ_bisector and AC_bisector
  collinear I1 I2 I3 := by sorry

end NUMINAMATH_CALUDE_external_angle_bisectors_collinear_l2707_270788


namespace NUMINAMATH_CALUDE_farm_animal_ratio_l2707_270760

/-- Proves that the initial ratio of horses to cows is 4:1 given the problem conditions --/
theorem farm_animal_ratio :
  ∀ (h c : ℕ),  -- Initial number of horses and cows
  (h - 15 : ℚ) / (c + 15 : ℚ) = 13 / 7 →  -- Ratio after transaction
  h - 15 = c + 15 + 30 →  -- Difference after transaction
  h / c = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_farm_animal_ratio_l2707_270760


namespace NUMINAMATH_CALUDE_bottles_maria_drank_l2707_270786

theorem bottles_maria_drank (initial bottles_bought bottles_remaining : ℕ) : 
  initial = 14 → bottles_bought = 45 → bottles_remaining = 51 → 
  initial + bottles_bought - bottles_remaining = 8 := by
sorry

end NUMINAMATH_CALUDE_bottles_maria_drank_l2707_270786


namespace NUMINAMATH_CALUDE_add_one_five_times_l2707_270768

theorem add_one_five_times (m : ℕ) : 
  let n := m + 5
  n = m + 5 ∧ n - (m + 1) = 4 := by
sorry

end NUMINAMATH_CALUDE_add_one_five_times_l2707_270768


namespace NUMINAMATH_CALUDE_max_positive_numbers_with_zero_average_l2707_270770

theorem max_positive_numbers_with_zero_average (numbers : List ℝ) : 
  numbers.length = 20 → numbers.sum / numbers.length = 0 → 
  (numbers.filter (λ x => x > 0)).length ≤ 19 := by
sorry

end NUMINAMATH_CALUDE_max_positive_numbers_with_zero_average_l2707_270770


namespace NUMINAMATH_CALUDE_caleb_hamburger_cost_l2707_270787

/-- Represents the total cost of Caleb's hamburger purchase --/
def total_cost (single_price : ℚ) (double_price : ℚ) (total_burgers : ℕ) (double_burgers : ℕ) : ℚ :=
  single_price * (total_burgers - double_burgers) + double_price * double_burgers

/-- Theorem stating that Caleb's total spending on hamburgers is $74.50 --/
theorem caleb_hamburger_cost : 
  total_cost 1 (3/2) 50 49 = 149/2 := by
  sorry

end NUMINAMATH_CALUDE_caleb_hamburger_cost_l2707_270787


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2707_270704

/-- Represents the repeating decimal 4.565656... -/
def repeating_decimal : ℚ := 4 + 56 / 99

/-- The fraction representation of 4.565656... -/
def fraction : ℚ := 452 / 99

theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2707_270704


namespace NUMINAMATH_CALUDE_stating_botanical_garden_visitors_l2707_270792

/-- Represents the growth rate of visitors in a botanical garden -/
def growth_rate_equation (x : ℝ) : Prop :=
  (1 + x)^2 = 3

/-- 
Theorem stating that the growth rate equation holds given the conditions:
- The number of visitors in March is three times that of January
- x is the average growth rate of visitors in February and March
-/
theorem botanical_garden_visitors (x : ℝ) 
  (h_march : ∃ (a : ℝ), a > 0 ∧ a * (1 + x)^2 = 3 * a) : 
  growth_rate_equation x := by
  sorry

end NUMINAMATH_CALUDE_stating_botanical_garden_visitors_l2707_270792


namespace NUMINAMATH_CALUDE_sweet_number_existence_l2707_270797

def is_sweet (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [0, 1, 2, 4, 8]

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sweet_number_existence : ∃ n : ℕ,
  n > 0 ∧
  is_sweet n ∧
  is_sweet (n^2) ∧
  is_sweet (n^3) ∧
  digit_sum n = 2014 := by
  sorry

end NUMINAMATH_CALUDE_sweet_number_existence_l2707_270797


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2707_270752

theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : marked_price = 200)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.2) :
  ∃ (cost_price : ℝ),
    cost_price = 150 ∧ 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2707_270752


namespace NUMINAMATH_CALUDE_simplify_expression_l2707_270783

theorem simplify_expression (y : ℝ) :
  3 * y + 9 * y^2 - 15 - (5 - 3 * y - 9 * y^2) = 18 * y^2 + 6 * y - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2707_270783


namespace NUMINAMATH_CALUDE_total_cost_is_56_15_l2707_270756

-- Define the prices and quantities
def spam_price : ℚ := 3
def peanut_butter_price : ℚ := 5
def bread_price : ℚ := 2
def spam_quantity : ℕ := 12
def peanut_butter_quantity : ℕ := 3
def bread_quantity : ℕ := 4

-- Define the discount and tax rates
def spam_discount : ℚ := 0.1
def peanut_butter_tax : ℚ := 0.05

-- Define the total cost function
def total_cost : ℚ :=
  (spam_price * spam_quantity * (1 - spam_discount)) +
  (peanut_butter_price * peanut_butter_quantity * (1 + peanut_butter_tax)) +
  (bread_price * bread_quantity)

-- Theorem statement
theorem total_cost_is_56_15 : total_cost = 56.15 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_56_15_l2707_270756


namespace NUMINAMATH_CALUDE_chef_apples_l2707_270775

/-- Represents the number of apples used to make the pie -/
def apples_used : ℕ := 15

/-- Represents the number of apples left after making the pie -/
def apples_left : ℕ := 4

/-- Represents the total number of apples before making the pie -/
def total_apples : ℕ := apples_used + apples_left

/-- Theorem stating that the total number of apples before making the pie
    is equal to the sum of apples used and apples left -/
theorem chef_apples : total_apples = apples_used + apples_left := by
  sorry

end NUMINAMATH_CALUDE_chef_apples_l2707_270775


namespace NUMINAMATH_CALUDE_multiplication_problem_l2707_270738

theorem multiplication_problem : 10 * (3/27) * 36 = 40 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l2707_270738


namespace NUMINAMATH_CALUDE_three_circles_equal_angle_points_l2707_270723

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Predicate to check if two circles do not intersect and neither is contained within the other -/
def are_separate (c1 c2 : Circle) : Prop := sorry

/-- The locus of points from which two circles are seen at the same angle -/
def equal_angle_locus (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- The angle at which a circle is seen from a point -/
def viewing_angle (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

theorem three_circles_equal_angle_points 
  (k1 k2 k3 : Circle)
  (h12 : are_separate k1 k2)
  (h23 : are_separate k2 k3)
  (h13 : are_separate k1 k3) :
  ∃ p : ℝ × ℝ, 
    viewing_angle k1 p = viewing_angle k2 p ∧ 
    viewing_angle k2 p = viewing_angle k3 p ∧
    p ∈ (equal_angle_locus k1 k2) ∩ (equal_angle_locus k2 k3) := by
  sorry

end NUMINAMATH_CALUDE_three_circles_equal_angle_points_l2707_270723


namespace NUMINAMATH_CALUDE_max_product_of_arithmetic_sequence_l2707_270763

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem max_product_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + 2 * a 6 = 6) :
  (∀ x : ℝ, a 4 * a 6 ≤ x → x ≤ 4) ∧ a 4 * a 6 ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_arithmetic_sequence_l2707_270763


namespace NUMINAMATH_CALUDE_length_of_24_l2707_270749

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer. -/
def length (n : ℕ) : ℕ := sorry

/-- 24 can be expressed as a product of 4 prime factors. -/
theorem length_of_24 : length 24 = 4 := by sorry

end NUMINAMATH_CALUDE_length_of_24_l2707_270749


namespace NUMINAMATH_CALUDE_initial_men_count_l2707_270776

/-- The number of days it takes the initial group to complete the job -/
def initial_days : ℕ := 15

/-- The number of men in the second group -/
def second_group_men : ℕ := 18

/-- The number of days it takes the second group to complete the job -/
def second_group_days : ℕ := 20

/-- The total amount of work in man-days -/
def total_work : ℕ := second_group_men * second_group_days

/-- The number of men initially working on the job -/
def initial_men : ℕ := total_work / initial_days

theorem initial_men_count : initial_men = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l2707_270776


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2707_270758

theorem quadratic_rewrite (b n : ℝ) : 
  (∀ x, x^2 + b*x + 72 = (x + n)^2 + 20) → 
  n > 0 → 
  b = 4 * Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2707_270758


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_6_12_plus_5_13_l2707_270795

theorem smallest_prime_divisor_of_6_12_plus_5_13 :
  (Nat.minFac (6^12 + 5^13) = 5) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_6_12_plus_5_13_l2707_270795


namespace NUMINAMATH_CALUDE_rings_arrangement_count_l2707_270740

def rings : ℕ := 10
def fingers : ℕ := 5
def rings_to_arrange : ℕ := 6

def arrange_rings (total_rings : ℕ) (fingers : ℕ) (rings_to_arrange : ℕ) : ℕ :=
  (total_rings.choose rings_to_arrange) * fingers * (rings_to_arrange.factorial)

theorem rings_arrangement_count :
  arrange_rings rings fingers rings_to_arrange = 756000 := by
  sorry

end NUMINAMATH_CALUDE_rings_arrangement_count_l2707_270740


namespace NUMINAMATH_CALUDE_fish_pond_estimate_l2707_270718

theorem fish_pond_estimate (initial_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initial_marked = 40 →
  second_catch = 100 →
  marked_in_second = 5 →
  (second_catch : ℚ) / marked_in_second = (800 : ℚ) / initial_marked :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_estimate_l2707_270718


namespace NUMINAMATH_CALUDE_power_function_through_sqrt2_l2707_270701

/-- A power function that passes through the point (2, √2) is equal to the square root function. -/
theorem power_function_through_sqrt2 (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x^α) →   -- f is a power function
  f 2 = Real.sqrt 2 →      -- f passes through (2, √2)
  ∀ x > 0, f x = Real.sqrt x := by
sorry

end NUMINAMATH_CALUDE_power_function_through_sqrt2_l2707_270701


namespace NUMINAMATH_CALUDE_MNP_collinear_tangent_equals_PA_l2707_270799

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def S : Circle := sorry
def S₁ : Circle := sorry
def A : Point := sorry
def B : Point := sorry
def M : Point := sorry
def N : Point := sorry
def P : Point := sorry

-- Define the conditions
axiom chord_divides_circle : sorry
axiom S₁_touches_AB_at_M : sorry
axiom S₁_touches_arc_at_N : sorry
axiom P_is_midpoint_of_other_arc : sorry

-- Define helper functions
def collinear (p q r : Point) : Prop := sorry
def tangent_length (p : Point) (c : Circle) : ℝ := sorry
def distance (p q : Point) : ℝ := sorry

-- State the theorems to be proved
theorem MNP_collinear : collinear M N P := sorry

theorem tangent_equals_PA : tangent_length P S₁ = distance P A := sorry

end NUMINAMATH_CALUDE_MNP_collinear_tangent_equals_PA_l2707_270799


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_defined_l2707_270706

theorem sqrt_x_minus_2_defined (x : ℝ) : 
  ∃ y : ℝ, y ^ 2 = x - 2 ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_defined_l2707_270706


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_l2707_270728

/-- Calculates the total cost of decorations for a wedding reception --/
def total_decoration_cost (num_tables : ℕ) 
                          (tablecloth_cost : ℕ) 
                          (place_setting_cost : ℕ) 
                          (place_settings_per_table : ℕ) 
                          (roses_per_centerpiece : ℕ) 
                          (rose_cost : ℕ) 
                          (lilies_per_centerpiece : ℕ) 
                          (lily_cost : ℕ) : ℕ :=
  num_tables * (tablecloth_cost + 
                place_settings_per_table * place_setting_cost + 
                roses_per_centerpiece * rose_cost + 
                lilies_per_centerpiece * lily_cost)

/-- Theorem stating that the total decoration cost for the given conditions is $3500 --/
theorem wedding_decoration_cost : 
  total_decoration_cost 20 25 10 4 10 5 15 4 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_wedding_decoration_cost_l2707_270728


namespace NUMINAMATH_CALUDE_pet_shop_total_l2707_270774

theorem pet_shop_total (dogs cats bunnies : ℕ) : 
  dogs = 154 → 
  dogs * 8 = bunnies * 7 → 
  dogs + bunnies = 330 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_shop_total_l2707_270774


namespace NUMINAMATH_CALUDE_parabola_c_value_l2707_270767

/-- A parabola passing through two points -/
structure Parabola where
  b : ℝ
  c : ℝ
  pass_point_1 : 2 * 1^2 + b * 1 + c = 4
  pass_point_2 : 2 * 3^2 + b * 3 + c = 16

/-- The value of c for the parabola -/
def c_value (p : Parabola) : ℝ := p.c

theorem parabola_c_value (p : Parabola) : c_value p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2707_270767


namespace NUMINAMATH_CALUDE_min_value_of_f_l2707_270746

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 3) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y ∧ f a x = -29 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2707_270746


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2707_270709

theorem simplify_complex_fraction (a b : ℝ) : 
  ((a - b)^2 + a*b) / ((a + b)^2 - a*b) / 
  ((a^5 + b^5 + a^2*b^3 + a^3*b^2) / 
   ((a^3 + b^3 + a^2*b + a*b^2) * (a^3 - b^3))) = a - b :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2707_270709


namespace NUMINAMATH_CALUDE_ellipse_properties_l2707_270745

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define perpendicular rays from origin
def perpendicular_rays (m1 m2 n1 n2 : ℝ) : Prop :=
  m1 * n1 + m2 * n2 = 0

-- Define points M and N on the ellipse
def points_on_ellipse (m1 m2 n1 n2 : ℝ) : Prop :=
  ellipse m1 m2 ∧ ellipse n1 n2

-- Theorem statement
theorem ellipse_properties :
  ∀ (m1 m2 n1 n2 : ℝ),
  perpendicular_rays m1 m2 n1 n2 →
  points_on_ellipse m1 m2 n1 n2 →
  (∃ (e : ℝ), e = 1/2 ∧ e = Real.sqrt (1 - 3/4)) ∧
  (∃ (d : ℝ), d = 2 * Real.sqrt 21 / 7 ∧
    ∀ (k b : ℝ), (m2 = k * m1 + b ∧ n2 = k * n1 + b) →
      d = |b| / Real.sqrt (k^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2707_270745


namespace NUMINAMATH_CALUDE_dino_money_theorem_l2707_270798

/-- Calculates the money Dino has left at the end of the month based on his work hours, rates, and expenses. -/
def dino_money_left (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (expenses : ℕ) : ℕ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - expenses

/-- Theorem stating that Dino has $500 left at the end of the month. -/
theorem dino_money_theorem : dino_money_left 20 30 5 10 20 40 500 = 500 := by
  sorry

end NUMINAMATH_CALUDE_dino_money_theorem_l2707_270798
