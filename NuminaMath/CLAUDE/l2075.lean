import Mathlib

namespace NUMINAMATH_CALUDE_sum_real_coefficients_binomial_expansion_l2075_207501

theorem sum_real_coefficients_binomial_expansion (i : ℂ) :
  let x : ℂ := Complex.I
  let n : ℕ := 1010
  let T : ℝ := (Finset.range (n + 1)).sum (λ k => if k % 2 = 0 then (n.choose k : ℝ) else 0)
  T = 2^(n - 1) :=
sorry

end NUMINAMATH_CALUDE_sum_real_coefficients_binomial_expansion_l2075_207501


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l2075_207508

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := by sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l2075_207508


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2075_207552

/-- Given a triangle with sides 13, 14, and 15, the shortest altitude has length 168/15 -/
theorem shortest_altitude_of_triangle (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h1 := 2 * area / a
  let h2 := 2 * area / b
  let h3 := 2 * area / c
  min h1 (min h2 h3) = 168 / 15 := by
sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2075_207552


namespace NUMINAMATH_CALUDE_ball_cost_l2075_207530

/-- Proves that if Kyoko buys 3 balls for a total cost of $4.62, then each ball costs $1.54. -/
theorem ball_cost (total_cost : ℝ) (num_balls : ℕ) (cost_per_ball : ℝ) 
  (h1 : total_cost = 4.62)
  (h2 : num_balls = 3)
  (h3 : cost_per_ball = total_cost / num_balls) : 
  cost_per_ball = 1.54 := by
  sorry

end NUMINAMATH_CALUDE_ball_cost_l2075_207530


namespace NUMINAMATH_CALUDE_problem_proof_l2075_207583

theorem problem_proof (a b : ℝ) (ha : a > 0) (h : Real.exp a + Real.log b = 1) :
  a * b < 1 ∧ a + b > 1 ∧ Real.exp a + b > 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2075_207583


namespace NUMINAMATH_CALUDE_ashtons_remaining_items_l2075_207591

def pencil_boxes : ℕ := 3
def pencils_per_box : ℕ := 14
def pen_boxes : ℕ := 2
def pens_per_box : ℕ := 10
def pencils_to_brother : ℕ := 6
def pencils_to_friends : ℕ := 12
def pens_to_friends : ℕ := 8

theorem ashtons_remaining_items :
  let initial_pencils := pencil_boxes * pencils_per_box
  let initial_pens := pen_boxes * pens_per_box
  let remaining_pencils := initial_pencils - pencils_to_brother - pencils_to_friends
  let remaining_pens := initial_pens - pens_to_friends
  remaining_pencils + remaining_pens = 36 := by
  sorry

end NUMINAMATH_CALUDE_ashtons_remaining_items_l2075_207591


namespace NUMINAMATH_CALUDE_transportation_theorem_l2075_207522

/-- Represents the capacity and cost of a truck type -/
structure TruckType where
  capacity : ℕ
  cost : ℕ

/-- Represents a transportation plan -/
structure TransportPlan where
  typeA : ℕ
  typeB : ℕ

/-- Solves the transportation problem -/
def solve_transportation_problem (typeA typeB : TruckType) (total_goods : ℕ) : 
  (TruckType × TruckType × TransportPlan) := sorry

theorem transportation_theorem 
  (typeA typeB : TruckType) (total_goods : ℕ) 
  (h1 : 3 * typeA.capacity + 2 * typeB.capacity = 90)
  (h2 : 5 * typeA.capacity + 4 * typeB.capacity = 160)
  (h3 : typeA.cost = 500)
  (h4 : typeB.cost = 400)
  (h5 : total_goods = 190) :
  let (solvedA, solvedB, optimal_plan) := solve_transportation_problem typeA typeB total_goods
  solvedA.capacity = 20 ∧ 
  solvedB.capacity = 15 ∧ 
  optimal_plan.typeA = 8 ∧ 
  optimal_plan.typeB = 2 := by sorry

end NUMINAMATH_CALUDE_transportation_theorem_l2075_207522


namespace NUMINAMATH_CALUDE_pencils_lost_l2075_207582

/-- Given an initial number of pencils and a final number of pencils,
    prove that the number of lost pencils is the difference between them. -/
theorem pencils_lost (initial final : ℕ) (h : initial ≥ final) :
  initial - final = initial - final :=
by sorry

end NUMINAMATH_CALUDE_pencils_lost_l2075_207582


namespace NUMINAMATH_CALUDE_fraction_inequality_l2075_207503

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  a / (a + c) > b / (b + c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2075_207503


namespace NUMINAMATH_CALUDE_quadratic_roots_conditions_l2075_207561

/-- The quadratic equation x^2 + 2x + 2m = 0 has two distinct real roots -/
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*m = 0 ∧ x₂^2 + 2*x₂ + 2*m = 0

/-- The sum of squares of the roots of x^2 + 2x + 2m = 0 is 8 -/
def sum_of_squares_is_8 (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 + 2*x₁ + 2*m = 0 ∧ x₂^2 + 2*x₂ + 2*m = 0 ∧ x₁^2 + x₂^2 = 8

theorem quadratic_roots_conditions (m : ℝ) :
  (has_two_distinct_real_roots m ↔ m < 1/2) ∧
  (sum_of_squares_is_8 m → m = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_conditions_l2075_207561


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l2075_207553

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (5*x - 4)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l2075_207553


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_in_equilateral_pyramid_l2075_207502

/-- A pyramid with an equilateral triangular base and equilateral triangular lateral faces -/
structure EquilateralPyramid where
  base_side_length : ℝ
  lateral_face_is_equilateral : Bool

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  side_length : ℝ
  base_on_pyramid_base : Bool
  top_edges_on_lateral_faces : Bool

/-- The volume of the inscribed cube in the given pyramid -/
def inscribed_cube_volume (p : EquilateralPyramid) (c : InscribedCube) : ℝ :=
  c.side_length ^ 3

theorem inscribed_cube_volume_in_equilateral_pyramid 
  (p : EquilateralPyramid) 
  (c : InscribedCube) 
  (h1 : p.base_side_length = 2)
  (h2 : p.lateral_face_is_equilateral = true)
  (h3 : c.base_on_pyramid_base = true)
  (h4 : c.top_edges_on_lateral_faces = true) :
  inscribed_cube_volume p c = 3 * Real.sqrt 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_in_equilateral_pyramid_l2075_207502


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l2075_207572

/-- Calculates the total grocery bill for Ray's purchase with a store rewards discount --/
theorem rays_grocery_bill :
  let meat_price : ℚ := 5
  let crackers_price : ℚ := 3.5
  let vegetable_price : ℚ := 2
  let vegetable_quantity : ℕ := 4
  let cheese_price : ℚ := 3.5
  let discount_rate : ℚ := 0.1

  let total_before_discount : ℚ := 
    meat_price + crackers_price + (vegetable_price * vegetable_quantity) + cheese_price
  
  let discount_amount : ℚ := total_before_discount * discount_rate
  
  let final_bill : ℚ := total_before_discount - discount_amount

  final_bill = 18 := by sorry

end NUMINAMATH_CALUDE_rays_grocery_bill_l2075_207572


namespace NUMINAMATH_CALUDE_prob_three_non_defective_pencils_l2075_207533

/-- The probability of selecting 3 non-defective pencils from a box of 8 pencils, where 2 are defective. -/
theorem prob_three_non_defective_pencils :
  let total_pencils : ℕ := 8
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils := total_pencils - defective_pencils
  Nat.choose non_defective_pencils selected_pencils / Nat.choose total_pencils selected_pencils = 5 / 14 :=
by sorry

end NUMINAMATH_CALUDE_prob_three_non_defective_pencils_l2075_207533


namespace NUMINAMATH_CALUDE_probability_red_or_blue_specific_l2075_207579

/-- The probability of drawing either a red or blue marble from a bag -/
def probability_red_or_blue (red blue green yellow : ℕ) : ℚ :=
  (red + blue : ℚ) / (red + blue + green + yellow : ℚ)

/-- Theorem: The probability of drawing either a red or blue marble from a bag
    containing 5 red, 3 blue, 4 green, and 6 yellow marbles is 4/9 -/
theorem probability_red_or_blue_specific : probability_red_or_blue 5 3 4 6 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_blue_specific_l2075_207579


namespace NUMINAMATH_CALUDE_binomial_square_proof_l2075_207544

theorem binomial_square_proof :
  ∃ (r s : ℚ), (r * x + s)^2 = (100 / 9 : ℚ) * x^2 + 20 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_proof_l2075_207544


namespace NUMINAMATH_CALUDE_consecutive_odd_product_ends_09_l2075_207594

theorem consecutive_odd_product_ends_09 (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, (10*n - 3) * (10*n - 1) * (10*n + 1) * (10*n + 3) = 100 * k + 9 :=
sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_ends_09_l2075_207594


namespace NUMINAMATH_CALUDE_equal_products_l2075_207510

theorem equal_products : 2 * 20212021 * 1011 * 202320232023 = 43 * 47 * 20232023 * 202220222022 := by
  sorry

end NUMINAMATH_CALUDE_equal_products_l2075_207510


namespace NUMINAMATH_CALUDE_sample_size_is_200_l2075_207581

/-- The expected sample size for a school with given student counts and selection probability -/
def expected_sample_size (freshmen sophomores juniors : ℕ) (prob : ℝ) : ℝ :=
  (freshmen + sophomores + juniors : ℝ) * prob

/-- Theorem stating that the expected sample size is 200 for the given school population and selection probability -/
theorem sample_size_is_200 :
  expected_sample_size 280 320 400 0.2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_200_l2075_207581


namespace NUMINAMATH_CALUDE_part_one_simplification_part_two_simplification_l2075_207505

-- Part 1
theorem part_one_simplification :
  (1 / 2)⁻¹ - (Real.sqrt 2019 - 1)^0 = 1 := by sorry

-- Part 2
theorem part_two_simplification (x y : ℝ) :
  (x - y)^2 - (x + 2*y) * (x - 2*y) = -2*x*y + 5*y^2 := by sorry

end NUMINAMATH_CALUDE_part_one_simplification_part_two_simplification_l2075_207505


namespace NUMINAMATH_CALUDE_square_sum_ge_double_product_l2075_207555

theorem square_sum_ge_double_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_double_product_l2075_207555


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2075_207560

/-- An arithmetic sequence {a_n} with a_1 = 2 and a_3 + a_5 = 10 has a common difference of 1. -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term condition
  a 3 + a 5 = 10 →                     -- sum of 3rd and 5th terms condition
  a 2 - a 1 = 1 :=                     -- common difference is 1
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2075_207560


namespace NUMINAMATH_CALUDE_soap_brand_ratio_l2075_207578

/-- Given a survey of households and their soap brand preferences, 
    prove the ratio of households using only brand B to those using both brands. -/
theorem soap_brand_ratio 
  (total : ℕ) 
  (neither : ℕ) 
  (only_w : ℕ) 
  (both : ℕ) 
  (h1 : total = 200)
  (h2 : neither = 80)
  (h3 : only_w = 60)
  (h4 : both = 40)
  : (total - neither - only_w - both) / both = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_ratio_l2075_207578


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l2075_207520

/-- Represents the number of coins in the final distribution step -/
def x : ℕ := 13

/-- The sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Pete's coins after the distribution -/
def pete_coins : ℕ := 5 * x^2

/-- Paul's coins after the distribution -/
def paul_coins : ℕ := x^2

/-- The total number of coins -/
def total_coins : ℕ := pete_coins + paul_coins

theorem pirate_treasure_distribution :
  (sum_of_squares x = pete_coins) ∧
  (total_coins = 1014) := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l2075_207520


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2075_207538

theorem complex_fraction_simplification (N : ℕ) (h : N = 2^16) :
  (65533^3 + 65534^3 + 65535^3 + 65536^3 + 65537^3 + 65538^3 + 65539^3) / 
  (32765 * 32766 + 32767 * 32768 + 32768 * 32769 + 32770 * 32771 : ℕ) = 7 * N :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2075_207538


namespace NUMINAMATH_CALUDE_parabola_points_relation_l2075_207531

/-- Prove that for points A(1, y₁) and B(2, y₂) lying on the parabola y = a(x+1)² + 2 where a < 0, 
    the relationship 2 > y₁ > y₂ holds. -/
theorem parabola_points_relation (a y₁ y₂ : ℝ) : 
  a < 0 → 
  y₁ = a * (1 + 1)^2 + 2 → 
  y₂ = a * (2 + 1)^2 + 2 → 
  2 > y₁ ∧ y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_relation_l2075_207531


namespace NUMINAMATH_CALUDE_geometric_sum_four_l2075_207592

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sum_four (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 3 = 4 →
  a 2 + a 4 = -10 →
  |q| > 1 →
  a 1 + a 2 + a 3 + a 4 = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_four_l2075_207592


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2075_207584

/-- Given an arithmetic sequence {a_n} where a_5 + a_6 + a_7 = 15,
    prove that the sum (a_3 + a_4 + ... + a_9) is equal to 35. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 + a 7 = 15 →                            -- given condition
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=    -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2075_207584


namespace NUMINAMATH_CALUDE_factory_solution_l2075_207542

def factory_problem (total_employees : ℕ) : Prop :=
  ∃ (employees_17 : ℕ),
    -- 200 employees earn $12/hour
    -- 40 employees earn $14/hour
    -- The rest earn $17/hour
    total_employees = 200 + 40 + employees_17 ∧
    -- The cost for one 8-hour shift is $31840
    31840 = (200 * 12 + 40 * 14 + employees_17 * 17) * 8

theorem factory_solution : ∃ (total_employees : ℕ), factory_problem total_employees ∧ total_employees = 300 := by
  sorry

end NUMINAMATH_CALUDE_factory_solution_l2075_207542


namespace NUMINAMATH_CALUDE_partnership_profit_distribution_l2075_207513

/-- Partnership profit distribution problem -/
theorem partnership_profit_distribution (total_profit : ℚ) 
  (hA : ℚ) (hB : ℚ) (hC : ℚ) (hD : ℚ) :
  hA = 1/3 →
  hB = 1/4 →
  hC = 1/5 →
  hD = 1 - (hA + hB + hC) →
  total_profit = 2415 →
  hA * total_profit = 805 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_distribution_l2075_207513


namespace NUMINAMATH_CALUDE_unknown_number_is_three_l2075_207569

theorem unknown_number_is_three (x n : ℝ) (h1 : (3/2) * x - n = 15) (h2 : x = 12) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_is_three_l2075_207569


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2075_207595

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  (a 1 < 0 ∧ 0 < q ∧ q < 1) →
  (∀ n : ℕ, n > 0 → a (n + 1) > a n) ∧
  ¬(∀ n : ℕ, n > 0 → a (n + 1) > a n → (a 1 < 0 ∧ 0 < q ∧ q < 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2075_207595


namespace NUMINAMATH_CALUDE_runner_problem_l2075_207515

theorem runner_problem (v : ℝ) (h : v > 0) : 
  (40 / v = 8) → (8 - 20 / v = 4) := by sorry

end NUMINAMATH_CALUDE_runner_problem_l2075_207515


namespace NUMINAMATH_CALUDE_factorization_xy_squared_l2075_207534

theorem factorization_xy_squared (x y : ℝ) : x^2*y + x*y^2 = x*y*(x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_l2075_207534


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_is_five_l2075_207570

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_kayak_ratio : ℚ
  total_revenue : ℕ

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (info : RentalInfo) : ℕ :=
  let canoes := (info.total_revenue / (3 * info.canoe_cost + 2 * info.kayak_cost)) * 3
  let kayaks := (info.total_revenue / (3 * info.canoe_cost + 2 * info.kayak_cost)) * 2
  canoes - kayaks

/-- Theorem stating the difference between canoes and kayaks rented --/
theorem canoe_kayak_difference_is_five (info : RentalInfo)
  (h1 : info.canoe_cost = 15)
  (h2 : info.kayak_cost = 18)
  (h3 : info.canoe_kayak_ratio = 3/2)
  (h4 : info.total_revenue = 405) :
  canoe_kayak_difference info = 5 := by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_is_five_l2075_207570


namespace NUMINAMATH_CALUDE_largest_value_l2075_207551

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  a^2 + b^2 = max a (max (1/2) (max (2*a*b) (a^2 + b^2))) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l2075_207551


namespace NUMINAMATH_CALUDE_barry_sitting_time_l2075_207549

/-- Calculates the sitting time between turns for Barry's head-standing routine -/
def calculate_sitting_time (total_time minutes_per_turn number_of_turns : ℕ) : ℕ :=
  let total_standing_time := minutes_per_turn * number_of_turns
  let total_sitting_time := total_time - total_standing_time
  let number_of_breaks := number_of_turns - 1
  (total_sitting_time + number_of_breaks - 1) / number_of_breaks

theorem barry_sitting_time :
  let total_time : ℕ := 120  -- 2 hours in minutes
  let minutes_per_turn : ℕ := 10
  let number_of_turns : ℕ := 8
  calculate_sitting_time total_time minutes_per_turn number_of_turns = 6 := by
  sorry

end NUMINAMATH_CALUDE_barry_sitting_time_l2075_207549


namespace NUMINAMATH_CALUDE_total_cost_price_l2075_207543

/-- Represents the cost and selling information for a fruit --/
structure Fruit where
  sellingPrice : ℚ
  lossRatio : ℚ

/-- Calculates the cost price of a fruit given its selling price and loss ratio --/
def costPrice (fruit : Fruit) : ℚ :=
  fruit.sellingPrice / (1 - fruit.lossRatio)

/-- The apple sold in the shop --/
def apple : Fruit := { sellingPrice := 30, lossRatio := 1/5 }

/-- The orange sold in the shop --/
def orange : Fruit := { sellingPrice := 45, lossRatio := 1/4 }

/-- The banana sold in the shop --/
def banana : Fruit := { sellingPrice := 15, lossRatio := 1/6 }

/-- Theorem stating the total cost price of all three fruits --/
theorem total_cost_price :
  costPrice apple + costPrice orange + costPrice banana = 115.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_price_l2075_207543


namespace NUMINAMATH_CALUDE_relay_race_total_time_l2075_207547

/-- The time taken by the relay team to finish the race -/
def relay_race_time (mary_time susan_time jen_time tiffany_time : ℕ) : ℕ :=
  mary_time + susan_time + jen_time + tiffany_time

/-- Theorem stating the total time for the relay race -/
theorem relay_race_total_time : ∃ (mary_time susan_time jen_time tiffany_time : ℕ),
  mary_time = 2 * susan_time ∧
  susan_time = jen_time + 10 ∧
  jen_time = 30 ∧
  tiffany_time = mary_time - 7 ∧
  relay_race_time mary_time susan_time jen_time tiffany_time = 223 := by
  sorry


end NUMINAMATH_CALUDE_relay_race_total_time_l2075_207547


namespace NUMINAMATH_CALUDE_number_of_females_l2075_207558

theorem number_of_females (total : ℕ) (avg_all : ℚ) (avg_male : ℚ) (avg_female : ℚ) 
  (h1 : total = 140)
  (h2 : avg_all = 24)
  (h3 : avg_male = 21)
  (h4 : avg_female = 28) :
  ∃ (females : ℕ), females = 60 ∧ 
    avg_all * total = avg_female * females + avg_male * (total - females) :=
by sorry

end NUMINAMATH_CALUDE_number_of_females_l2075_207558


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2075_207504

/-- Given a line segment in the Cartesian plane with midpoint (2020, 11), 
    one endpoint at (a, 0), and the other endpoint on the line y = x, 
    prove that a = 4018 -/
theorem line_segment_endpoint (a : ℝ) : 
  (∃ t : ℝ, (a + t) / 2 = 2020 ∧ t / 2 = 11 ∧ t = t) → a = 4018 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2075_207504


namespace NUMINAMATH_CALUDE_middle_card_is_six_l2075_207536

theorem middle_card_is_six (a b c : ℕ) : 
  0 < a → 0 < b → 0 < c →
  a < b → b < c →
  a + b + c = 15 →
  (∀ x y z, x < y ∧ y < z ∧ x + y + z = 15 → x ≠ 3 ∨ (y ≠ 4 ∧ y ≠ 5)) →
  (∀ x y z, x < y ∧ y < z ∧ x + y + z = 15 → z ≠ 12 ∧ z ≠ 11 ∧ z ≠ 7) →
  (∃ p q, p < b ∧ b < q ∧ p + b + q = 15 ∧ (p ≠ a ∨ q ≠ c)) →
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_middle_card_is_six_l2075_207536


namespace NUMINAMATH_CALUDE_inequality_proof_l2075_207585

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : d > 0) :
  d / c < (d + 4) / (c + 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2075_207585


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l2075_207556

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by_9 (529000 + d * 100 + 46) ∧
    ∀ (d' : ℕ), d' < d → ¬is_divisible_by_9 (529000 + d' * 100 + 46) :=
by
  use 1
  sorry

#check smallest_digit_for_divisibility

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l2075_207556


namespace NUMINAMATH_CALUDE_floor_sum_count_l2075_207557

def count_integers (max : ℕ) : ℕ :=
  let count_for_form (k : ℕ) := (max - k) / 7 + 1
  (count_for_form 0) + (count_for_form 1) + (count_for_form 3) + (count_for_form 4)

theorem floor_sum_count :
  count_integers 1000 = 568 := by sorry

end NUMINAMATH_CALUDE_floor_sum_count_l2075_207557


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2075_207516

/-- Theorem: Weight of the new person in a group replacement scenario -/
theorem weight_of_new_person
  (n : ℕ) -- Number of people in the group
  (w : ℝ) -- Total weight of the original group
  (r : ℝ) -- Weight of the person being replaced
  (i : ℝ) -- Increase in average weight
  (h1 : n = 15) -- There are 15 people initially
  (h2 : r = 75) -- The replaced person weighs 75 kg
  (h3 : i = 3.2) -- The average weight increases by 3.2 kg
  (h4 : (w - r + (w / n + n * i)) / n = w / n + i) -- Equation for the new average weight
  : w / n + n * i - r = 123 := by
  sorry

#check weight_of_new_person

end NUMINAMATH_CALUDE_weight_of_new_person_l2075_207516


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2075_207593

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence
    with first term -5 and common difference 6 is 220 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum (-5) 6 10 = 220 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2075_207593


namespace NUMINAMATH_CALUDE_kylie_coins_left_l2075_207564

/-- Calculates the number of coins Kylie is left with after various transactions --/
def coins_left (piggy_bank : ℕ) (from_brother : ℕ) (from_father : ℕ) (given_to_friend : ℕ) : ℕ :=
  piggy_bank + from_brother + from_father - given_to_friend

/-- Theorem stating that Kylie is left with 15 coins --/
theorem kylie_coins_left : 
  coins_left 15 13 8 21 = 15 := by
  sorry

end NUMINAMATH_CALUDE_kylie_coins_left_l2075_207564


namespace NUMINAMATH_CALUDE_juggler_path_radius_l2075_207577

/-- The equation of the path described by the juggler's balls -/
def path_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 5 = 2*x + 4*y

/-- The radius of the path described by the juggler's balls -/
def path_radius : ℝ := 0

/-- Theorem stating that the radius of the path is 0 -/
theorem juggler_path_radius :
  ∀ x y : ℝ, path_equation x y → (x - 1)^2 + (y - 2)^2 = path_radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_juggler_path_radius_l2075_207577


namespace NUMINAMATH_CALUDE_exercise_book_price_l2075_207521

/-- The price of an exercise book in yuan -/
def price_per_book : ℚ := 0.55

/-- The number of books Xiaoming took -/
def xiaoming_books : ℕ := 8

/-- The number of books Xiaohong took -/
def xiaohong_books : ℕ := 12

/-- The amount Xiaohong gave to Xiaoming in yuan -/
def amount_given : ℚ := 1.1

theorem exercise_book_price :
  (xiaoming_books + xiaohong_books : ℚ) * price_per_book / 2 =
    (xiaoming_books : ℚ) * price_per_book + amount_given / 2 ∧
  (xiaoming_books + xiaohong_books : ℚ) * price_per_book / 2 =
    (xiaohong_books : ℚ) * price_per_book - amount_given / 2 :=
sorry

end NUMINAMATH_CALUDE_exercise_book_price_l2075_207521


namespace NUMINAMATH_CALUDE_dot_product_is_2020_l2075_207571

/-- A trapezoid with perpendicular diagonals -/
structure PerpendicularDiagonalTrapezoid where
  -- Points of the trapezoid
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- AB is a base of length 101
  AB_length : dist A B = 101
  -- CD is a base of length 20
  CD_length : dist C D = 20
  -- ABCD is a trapezoid (parallel sides)
  is_trapezoid : (B.1 - A.1) * (D.2 - C.2) = (D.1 - C.1) * (B.2 - A.2)
  -- Diagonals are perpendicular
  diagonals_perpendicular : (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0

/-- The dot product of vectors AD and BC in a trapezoid with perpendicular diagonals -/
def dot_product_AD_BC (t : PerpendicularDiagonalTrapezoid) : ℝ :=
  let AD := (t.D.1 - t.A.1, t.D.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  AD.1 * BC.1 + AD.2 * BC.2

/-- Theorem: The dot product of AD and BC is 2020 -/
theorem dot_product_is_2020 (t : PerpendicularDiagonalTrapezoid) :
  dot_product_AD_BC t = 2020 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_is_2020_l2075_207571


namespace NUMINAMATH_CALUDE_beavers_working_on_home_l2075_207586

/-- The number of beavers initially working on their home -/
def initial_beavers : ℕ := 2

/-- The number of beavers that went for a swim -/
def swimming_beavers : ℕ := 1

/-- The number of beavers still working on their home -/
def remaining_beavers : ℕ := initial_beavers - swimming_beavers

theorem beavers_working_on_home : remaining_beavers = 1 := by
  sorry

end NUMINAMATH_CALUDE_beavers_working_on_home_l2075_207586


namespace NUMINAMATH_CALUDE_not_in_third_quadrant_l2075_207573

def linear_function (x : ℝ) : ℝ := -2 * x + 5

theorem not_in_third_quadrant :
  ∀ x y : ℝ, y = linear_function x → ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_not_in_third_quadrant_l2075_207573


namespace NUMINAMATH_CALUDE_sin_negative_1665_degrees_l2075_207565

theorem sin_negative_1665_degrees :
  Real.sin ((-1665 : ℝ) * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1665_degrees_l2075_207565


namespace NUMINAMATH_CALUDE_paint_on_third_day_l2075_207568

/-- The amount of paint available on the third day of a room refresh project -/
theorem paint_on_third_day (initial_paint : ℝ) (added_paint : ℝ) : 
  initial_paint = 80 → 
  added_paint = 20 → 
  (initial_paint / 2 + added_paint) / 2 = 30 := by
sorry

end NUMINAMATH_CALUDE_paint_on_third_day_l2075_207568


namespace NUMINAMATH_CALUDE_sum_of_squares_of_factors_72_l2075_207554

def sum_of_squares_of_factors (n : ℕ) : ℕ := sorry

theorem sum_of_squares_of_factors_72 : sum_of_squares_of_factors 72 = 7735 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_factors_72_l2075_207554


namespace NUMINAMATH_CALUDE_right_triangle_coordinate_l2075_207539

/-- Given a right triangle ABC with vertices A(0, 0), B(0, 4a - 2), and C(x, 4a - 2),
    if the area of the triangle is 63, then the x-coordinate of point C is 126 / (4a - 2). -/
theorem right_triangle_coordinate (a : ℝ) (x : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 4 * a - 2)
  let C : ℝ × ℝ := (x, 4 * a - 2)
  (4 * a - 2 ≠ 0) →
  (1 / 2 : ℝ) * x * (4 * a - 2) = 63 →
  x = 126 / (4 * a - 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_coordinate_l2075_207539


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_l2075_207528

/-- The number of dogwood trees planted in a park --/
theorem dogwood_trees_planted (initial_trees final_trees : ℕ) 
  (h1 : initial_trees = 34)
  (h2 : final_trees = 83) :
  final_trees - initial_trees = 49 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_l2075_207528


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l2075_207559

/-- Two lines in the form ax + by + c = 0 are parallel if and only if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- The first line: 2x + (m+1)y + 4 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 1) * y + 4 = 0

/-- The second line: mx + 3y - 2 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := m * x + 3 * y - 2 = 0

theorem parallel_lines_m_values :
  ∀ m : ℝ, are_parallel 2 (m + 1) m 3 ↔ (m = -3 ∨ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l2075_207559


namespace NUMINAMATH_CALUDE_hcf_problem_l2075_207524

theorem hcf_problem (a b hcf : ℕ) (h1 : a = 391) (h2 : a ≥ b) 
  (h3 : ∃ (lcm : ℕ), lcm = hcf * 16 * 17 ∧ lcm = a * b / hcf) : hcf = 23 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l2075_207524


namespace NUMINAMATH_CALUDE_original_number_proof_l2075_207532

theorem original_number_proof (N : ℕ) : N = 28 ↔ 
  (∃ k : ℕ, N - 11 = 17 * k) ∧ 
  (∀ x : ℕ, x < 11 → ¬(∃ m : ℕ, N - x = 17 * m)) ∧
  (∀ M : ℕ, M < N → ¬(∃ k : ℕ, M - 11 = 17 * k) ∨ 
    (∃ x : ℕ, x < 11 ∧ ∃ m : ℕ, M - x = 17 * m)) :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l2075_207532


namespace NUMINAMATH_CALUDE_log_equation_solution_l2075_207518

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 4 + 2 * (Real.log x / Real.log 8) = 7 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2075_207518


namespace NUMINAMATH_CALUDE_no_functions_exist_l2075_207511

theorem no_functions_exist : ¬ ∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_functions_exist_l2075_207511


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2075_207535

/-- An isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  /-- The length of the equal sides of the isosceles triangle -/
  a : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The ratio of AN to AB, where N is the point where a line parallel to BC 
      and tangent to the inscribed circle intersects AB -/
  an_ratio : ℝ
  /-- Condition that the triangle is isosceles -/
  isosceles : a > 0
  /-- Condition that AN = 3/8 * AB -/
  an_condition : an_ratio = 3/8
  /-- Condition that the area of the triangle is 12 -/
  area_condition : area = 12

/-- Theorem: If the conditions are met, the radius of the inscribed circle is 3/2 -/
theorem inscribed_circle_radius 
  (t : IsoscelesTriangleWithInscribedCircle) : t.r = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2075_207535


namespace NUMINAMATH_CALUDE_gcd_20244_46656_l2075_207566

theorem gcd_20244_46656 : Nat.gcd 20244 46656 = 54 := by
  sorry

end NUMINAMATH_CALUDE_gcd_20244_46656_l2075_207566


namespace NUMINAMATH_CALUDE_houses_around_square_l2075_207563

/-- The number of houses around the square. -/
def n : ℕ := 32

/-- Maria's count for a given house. -/
def M (k : ℕ) : ℕ := k % n

/-- João's count for a given house. -/
def J (k : ℕ) : ℕ := k % n

/-- Theorem stating the number of houses around the square. -/
theorem houses_around_square :
  (M 5 = J 12) ∧ (J 5 = M 30) → n = 32 := by
  sorry

end NUMINAMATH_CALUDE_houses_around_square_l2075_207563


namespace NUMINAMATH_CALUDE_code_cracking_probabilities_l2075_207527

/-- The probability of person A succeeding -/
def prob_A : ℚ := 1/2

/-- The probability of person B succeeding -/
def prob_B : ℚ := 3/5

/-- The probability of person C succeeding -/
def prob_C : ℚ := 3/4

/-- The probability of exactly one person succeeding -/
def prob_exactly_one : ℚ := 
  prob_A * (1 - prob_B) * (1 - prob_C) + 
  (1 - prob_A) * prob_B * (1 - prob_C) + 
  (1 - prob_A) * (1 - prob_B) * prob_C

/-- The probability of the code being successfully cracked -/
def prob_success : ℚ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- The minimum number of people like C needed for at least 95% success rate -/
def min_people_C : ℕ := 3

theorem code_cracking_probabilities :
  prob_exactly_one = 11/40 ∧ 
  prob_success = 19/20 ∧
  (∀ n : ℕ, n ≥ min_people_C → 1 - (1 - prob_C)^n ≥ 95/100) ∧
  (∀ n : ℕ, n < min_people_C → 1 - (1 - prob_C)^n < 95/100) :=
sorry

end NUMINAMATH_CALUDE_code_cracking_probabilities_l2075_207527


namespace NUMINAMATH_CALUDE_high_school_student_distribution_l2075_207588

theorem high_school_student_distribution :
  ∀ (total juniors sophomores freshmen seniors : ℕ),
    total = 800 →
    juniors = (27 * total) / 100 →
    sophomores = total - (75 * total) / 100 →
    seniors = 160 →
    freshmen = total - (juniors + sophomores + seniors) →
    freshmen - sophomores = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_high_school_student_distribution_l2075_207588


namespace NUMINAMATH_CALUDE_min_value_product_min_value_achieved_l2075_207514

theorem min_value_product (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 6) : 
  x^3 * y^2 * z ≥ 1/108 := by
sorry

theorem min_value_achieved (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 6) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 1/x₀ + 1/y₀ + 1/z₀ = 6 ∧ x₀^3 * y₀^2 * z₀ = 1/108 := by
sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_achieved_l2075_207514


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2075_207517

theorem arithmetic_calculation : 1375 + 150 / 50 * 3 - 275 = 1109 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2075_207517


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l2075_207507

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1 -/
theorem man_son_age_ratio :
  ∀ (man_age son_age : ℕ),
    man_age = son_age + 32 →
    son_age = 30 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l2075_207507


namespace NUMINAMATH_CALUDE_square_area_tripled_side_l2075_207509

theorem square_area_tripled_side (s : ℝ) (h : s > 0) :
  (3 * s)^2 = 9 * s^2 :=
by sorry

end NUMINAMATH_CALUDE_square_area_tripled_side_l2075_207509


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l2075_207599

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 24) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l2075_207599


namespace NUMINAMATH_CALUDE_committee_count_l2075_207597

/-- Represents a department in the division of sciences -/
inductive Department
| Mathematics
| Statistics
| ComputerScience
| Physics

/-- Represents the gender of a professor -/
inductive Gender
| Male
| Female

/-- Represents the number of professors in each department by gender -/
def professors_count (d : Department) (g : Gender) : Nat :=
  match d, g with
  | Department.Physics, _ => 1
  | _, _ => 3

/-- Represents the total number of professors to be selected from each department -/
def selection_count (d : Department) : Nat :=
  match d with
  | Department.Physics => 1
  | _ => 2

/-- Calculates the number of ways to select professors from a department -/
def department_selection_ways (d : Department) : Nat :=
  (professors_count d Gender.Male).choose (selection_count d) *
  (professors_count d Gender.Female).choose (selection_count d)

/-- Theorem: The number of possible committees is 729 -/
theorem committee_count : 
  (department_selection_ways Department.Mathematics) *
  (department_selection_ways Department.Statistics) *
  (department_selection_ways Department.ComputerScience) *
  (department_selection_ways Department.Physics) = 729 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l2075_207597


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_minus_n_equals_8_l2075_207548

def vector_a : Fin 3 → ℝ := ![1, 3, -2]
def vector_b (m n : ℝ) : Fin 3 → ℝ := ![2, m + 1, n - 1]

def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 3, v i = k * u i

theorem parallel_vectors_imply_m_minus_n_equals_8 (m n : ℝ) :
  parallel vector_a (vector_b m n) → m - n = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_minus_n_equals_8_l2075_207548


namespace NUMINAMATH_CALUDE_max_value_is_16_l2075_207562

/-- A function f(x) that is symmetric about x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- Symmetry condition: f(x) = f(-4-x) for all x -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x, f a b x = f a b (-4-x)

/-- The maximum value of f(x) is 16 -/
theorem max_value_is_16 (a b : ℝ) (h : is_symmetric a b) :
  ∃ M, M = 16 ∧ ∀ x, f a b x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_is_16_l2075_207562


namespace NUMINAMATH_CALUDE_special_function_property_l2075_207541

/-- A function satisfying the given property for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x^2 - f x * f y + f y^2)

/-- Theorem stating that if f is a special function, then f(1996x) = 1996f(x) for all real x -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x := by
  sorry


end NUMINAMATH_CALUDE_special_function_property_l2075_207541


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2075_207550

/-- The area of a right triangle with legs of length 36 and 48 is 864 -/
theorem right_triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun leg1 leg2 area =>
    leg1 = 36 ∧ leg2 = 48 → area = (1 / 2) * leg1 * leg2 → area = 864

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 36 48 864 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2075_207550


namespace NUMINAMATH_CALUDE_apple_cost_per_kg_main_apple_cost_theorem_l2075_207590

/-- Represents the cost structure of apples -/
structure AppleCost where
  p : ℝ  -- Cost per kg for first 30 kgs
  q : ℝ  -- Cost per kg for additional kgs

/-- Theorem stating the cost per kg for first 30 kgs of apples -/
theorem apple_cost_per_kg (cost : AppleCost) : cost.p = 10 :=
  by
  have h1 : 30 * cost.p + 3 * cost.q = 360 := by sorry
  have h2 : 30 * cost.p + 6 * cost.q = 420 := by sorry
  have h3 : 25 * cost.p = 250 := by sorry
  sorry

/-- Main theorem proving the cost per kg for first 30 kgs of apples -/
theorem main_apple_cost_theorem : ∃ (cost : AppleCost), cost.p = 10 :=
  by
  sorry

end NUMINAMATH_CALUDE_apple_cost_per_kg_main_apple_cost_theorem_l2075_207590


namespace NUMINAMATH_CALUDE_max_value_of_f_l2075_207580

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 8

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2075_207580


namespace NUMINAMATH_CALUDE_intersection_characterization_l2075_207525

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x + 1 ≥ 0}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- Define the intersection of A and B
def A_inter_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_characterization : 
  A_inter_B = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_characterization_l2075_207525


namespace NUMINAMATH_CALUDE_charge_200_400_undetermined_l2075_207545

/-- Represents the monthly phone bill for a customer -/
structure PhoneBill where
  fixed_rental : ℝ
  free_calls : ℕ
  charge_200_400 : ℝ
  charge_400_plus : ℝ
  february_calls : ℕ
  march_calls : ℕ
  march_discount : ℝ

/-- The phone bill satisfies the given conditions -/
def satisfies_conditions (bill : PhoneBill) : Prop :=
  bill.fixed_rental = 350 ∧
  bill.free_calls = 200 ∧
  bill.charge_400_plus = 1.6 ∧
  bill.february_calls = 150 ∧
  bill.march_calls = 250 ∧
  bill.march_discount = 0.28

/-- Theorem stating that the charge per call when exceeding 200 calls cannot be determined -/
theorem charge_200_400_undetermined (bill : PhoneBill) 
  (h : satisfies_conditions bill) : 
  ¬ ∃ (x : ℝ), ∀ (b : PhoneBill), satisfies_conditions b → b.charge_200_400 = x :=
sorry

end NUMINAMATH_CALUDE_charge_200_400_undetermined_l2075_207545


namespace NUMINAMATH_CALUDE_benjamin_steps_to_times_square_l2075_207546

/-- The number of steps Benjamin took to reach Rockefeller Center -/
def steps_to_rockefeller : ℕ := 354

/-- The number of steps Benjamin took from Rockefeller Center to Times Square -/
def steps_rockefeller_to_times_square : ℕ := 228

/-- The total number of steps Benjamin took before reaching Times Square -/
def total_steps : ℕ := steps_to_rockefeller + steps_rockefeller_to_times_square

theorem benjamin_steps_to_times_square : total_steps = 582 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_steps_to_times_square_l2075_207546


namespace NUMINAMATH_CALUDE_black_area_after_seven_changes_l2075_207526

/-- Represents the fraction of black area remaining after a number of changes -/
def blackFraction (changes : ℕ) : ℚ :=
  (8/9) ^ changes

/-- The number of changes applied to the triangle -/
def numChanges : ℕ := 7

/-- Theorem stating the fraction of black area after seven changes -/
theorem black_area_after_seven_changes :
  blackFraction numChanges = 2097152/4782969 := by
  sorry

#eval blackFraction numChanges

end NUMINAMATH_CALUDE_black_area_after_seven_changes_l2075_207526


namespace NUMINAMATH_CALUDE_french_toast_weekends_l2075_207506

/-- Represents the number of slices used per weekend -/
def slices_per_weekend : ℚ := 3

/-- Represents the number of slices in a loaf of bread -/
def slices_per_loaf : ℕ := 12

/-- Represents the number of loaves of bread used -/
def loaves_used : ℕ := 26

/-- Theorem stating that 26 loaves of bread cover 104 weekends of french toast making -/
theorem french_toast_weekends : 
  (loaves_used : ℚ) * (slices_per_loaf : ℚ) / slices_per_weekend = 104 := by
  sorry

end NUMINAMATH_CALUDE_french_toast_weekends_l2075_207506


namespace NUMINAMATH_CALUDE_classroom_students_classroom_students_proof_l2075_207589

theorem classroom_students : ℕ → Prop :=
  fun S : ℕ =>
    let boys := S / 3
    let girls := S - boys
    let girls_with_dogs := (40 * girls) / 100
    let girls_with_cats := (20 * girls) / 100
    let girls_without_pets := girls - girls_with_dogs - girls_with_cats
    girls_without_pets = 8 → S = 30

-- The proof goes here
theorem classroom_students_proof : classroom_students 30 := by
  sorry

end NUMINAMATH_CALUDE_classroom_students_classroom_students_proof_l2075_207589


namespace NUMINAMATH_CALUDE_sally_coins_theorem_l2075_207500

def initial_pennies : ℕ := 8
def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def mom_nickels : ℕ := 2
def penny_value : ℕ := 1
def nickel_value : ℕ := 5

theorem sally_coins_theorem :
  let total_nickels := initial_nickels + dad_nickels + mom_nickels
  let total_value := initial_pennies * penny_value + total_nickels * nickel_value
  total_nickels = 18 ∧ total_value = 98 := by sorry

end NUMINAMATH_CALUDE_sally_coins_theorem_l2075_207500


namespace NUMINAMATH_CALUDE_fence_cost_l2075_207540

/-- The cost of building a fence around a circular plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 289 →
  price_per_foot = 58 →
  cost = 2 * (Real.sqrt (289 * Real.pi)) * price_per_foot →
  cost = 1972 :=
by
  sorry

#check fence_cost

end NUMINAMATH_CALUDE_fence_cost_l2075_207540


namespace NUMINAMATH_CALUDE_room_area_difference_l2075_207576

-- Define the dimensions of the rooms
def largest_room_width : ℕ := 45
def largest_room_length : ℕ := 30
def smallest_room_width : ℕ := 15
def smallest_room_length : ℕ := 8

-- Define the function to calculate the area of a rectangular room
def room_area (width : ℕ) (length : ℕ) : ℕ := width * length

-- Theorem statement
theorem room_area_difference :
  room_area largest_room_width largest_room_length - 
  room_area smallest_room_width smallest_room_length = 1230 := by
  sorry

end NUMINAMATH_CALUDE_room_area_difference_l2075_207576


namespace NUMINAMATH_CALUDE_carmichael_561_l2075_207523

theorem carmichael_561 (a : ℤ) : a ^ 561 ≡ a [ZMOD 561] := by
  sorry

end NUMINAMATH_CALUDE_carmichael_561_l2075_207523


namespace NUMINAMATH_CALUDE_original_denominator_proof_l2075_207575

theorem original_denominator_proof (d : ℚ) : 
  (2 : ℚ) / d ≠ 0 →
  (2 + 7 : ℚ) / (d + 7) = (1 : ℚ) / 3 →
  d = 20 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l2075_207575


namespace NUMINAMATH_CALUDE_mn_value_l2075_207519

theorem mn_value (m n : ℕ+) (h : m.val^2 + n.val^2 + 4*m.val - 46 = 0) :
  m.val * n.val = 5 ∨ m.val * n.val = 15 := by
sorry

end NUMINAMATH_CALUDE_mn_value_l2075_207519


namespace NUMINAMATH_CALUDE_symmetric_point_example_l2075_207596

/-- Given a point A and a point of symmetry, find the symmetric point -/
def symmetric_point (A : ℝ × ℝ × ℝ) (sym : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (2 * sym.1 - A.1, 2 * sym.2.1 - A.2.1, 2 * sym.2.2 - A.2.2)

/-- The theorem states that the symmetric point of A(3, -2, 4) with respect to (0, 1, -3) is (-3, 4, -10) -/
theorem symmetric_point_example : 
  symmetric_point (3, -2, 4) (0, 1, -3) = (-3, 4, -10) := by
  sorry

#eval symmetric_point (3, -2, 4) (0, 1, -3)

end NUMINAMATH_CALUDE_symmetric_point_example_l2075_207596


namespace NUMINAMATH_CALUDE_book_reading_percentage_l2075_207529

theorem book_reading_percentage (total_pages : ℕ) (second_night_percent : ℝ) 
  (third_night_percent : ℝ) (pages_left : ℕ) : ℝ :=
by
  have h1 : total_pages = 500 := by sorry
  have h2 : second_night_percent = 20 := by sorry
  have h3 : third_night_percent = 30 := by sorry
  have h4 : pages_left = 150 := by sorry
  
  -- Define the first night percentage
  let first_night_percent : ℝ := 20

  -- Prove that the first night percentage is correct
  have h5 : first_night_percent / 100 * total_pages + 
            second_night_percent / 100 * total_pages + 
            third_night_percent / 100 * total_pages = 
            total_pages - pages_left := by sorry

  exact first_night_percent

end NUMINAMATH_CALUDE_book_reading_percentage_l2075_207529


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2075_207512

theorem polynomial_division_remainder (x : ℂ) : 
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2075_207512


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2075_207537

theorem binomial_divisibility (k : ℤ) : k ≠ 1 ↔ ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ i : ℕ, (f i + k : ℤ) ∣ Nat.choose (2 * f i) (f i) → False :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2075_207537


namespace NUMINAMATH_CALUDE_sqrt_greater_than_sum_l2075_207567

theorem sqrt_greater_than_sum (a b x y : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) 
  (hab : a^2 + b^2 < 1) : 
  Real.sqrt (x^2 + y^2) > a*x + b*y := by
sorry

end NUMINAMATH_CALUDE_sqrt_greater_than_sum_l2075_207567


namespace NUMINAMATH_CALUDE_hundred_days_from_friday_is_sunday_l2075_207587

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem hundred_days_from_friday_is_sunday :
  advanceDay DayOfWeek.Friday 100 = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_hundred_days_from_friday_is_sunday_l2075_207587


namespace NUMINAMATH_CALUDE_solution_y_original_amount_l2075_207598

/-- Represents the composition of a solution --/
structure Solution where
  total : ℝ
  liquid_x_percent : ℝ
  water_percent : ℝ

/-- The problem statement --/
theorem solution_y_original_amount
  (y : Solution)
  (h1 : y.liquid_x_percent = 0.3)
  (h2 : y.water_percent = 0.7)
  (h3 : y.liquid_x_percent + y.water_percent = 1)
  (evaporated_water : ℝ)
  (h4 : evaporated_water = 4)
  (added_solution : Solution)
  (h5 : added_solution.total = 4)
  (h6 : added_solution.liquid_x_percent = 0.3)
  (h7 : added_solution.water_percent = 0.7)
  (new_solution : Solution)
  (h8 : new_solution.total = y.total)
  (h9 : new_solution.liquid_x_percent = 0.45)
  (h10 : y.total * y.liquid_x_percent + added_solution.total * added_solution.liquid_x_percent
       = new_solution.total * new_solution.liquid_x_percent) :
  y.total = 8 := by
  sorry


end NUMINAMATH_CALUDE_solution_y_original_amount_l2075_207598


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l2075_207574

theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo (π/6) (π/3), Monotone (fun x => (a - Real.sin x) / Real.cos x)) →
  a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l2075_207574
