import Mathlib

namespace NUMINAMATH_CALUDE_lilias_peaches_l2597_259755

/-- Represents the problem of calculating how many peaches Lilia sold to her friends. -/
theorem lilias_peaches (total_peaches : ℕ) (friends_price : ℚ) (relatives_peaches : ℕ) (relatives_price : ℚ) (kept_peaches : ℕ) (total_earned : ℚ) (total_sold : ℕ) :
  total_peaches = 15 →
  friends_price = 2 →
  relatives_peaches = 4 →
  relatives_price = 5/4 →
  kept_peaches = 1 →
  total_earned = 25 →
  total_sold = 14 →
  ∃ (friends_peaches : ℕ), 
    friends_peaches + relatives_peaches + kept_peaches = total_peaches ∧
    friends_peaches * friends_price + relatives_peaches * relatives_price = total_earned ∧
    friends_peaches = 10 :=
by sorry

end NUMINAMATH_CALUDE_lilias_peaches_l2597_259755


namespace NUMINAMATH_CALUDE_cyclists_speed_l2597_259702

/-- Cyclist's trip problem -/
theorem cyclists_speed (v : ℝ) : 
  v > 0 → -- The speed is positive
  (9 / v + 12 / 9 : ℝ) = 21 / 10.08 → -- Total time equation
  v = 12 := by
sorry

end NUMINAMATH_CALUDE_cyclists_speed_l2597_259702


namespace NUMINAMATH_CALUDE_binomial_15_4_l2597_259797

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l2597_259797


namespace NUMINAMATH_CALUDE_bells_lcm_l2597_259749

/-- The time interval (in minutes) between consecutive rings of the library bell -/
def library_interval : ℕ := 18

/-- The time interval (in minutes) between consecutive rings of the community center bell -/
def community_interval : ℕ := 24

/-- The time interval (in minutes) between consecutive rings of the restaurant bell -/
def restaurant_interval : ℕ := 30

/-- The theorem states that the least common multiple of the three bell intervals is 360 minutes -/
theorem bells_lcm :
  lcm (lcm library_interval community_interval) restaurant_interval = 360 :=
by sorry

end NUMINAMATH_CALUDE_bells_lcm_l2597_259749


namespace NUMINAMATH_CALUDE_rectangle_probability_l2597_259716

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The probability that a randomly selected point from a rectangle is closer to one point than another --/
def closerProbability (r : Rectangle) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem to be proved --/
theorem rectangle_probability : 
  let r := Rectangle.mk 0 0 3 2
  closerProbability r (0, 0) (4, 0) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_probability_l2597_259716


namespace NUMINAMATH_CALUDE_original_painting_width_l2597_259727

/-- Given a painting and its enlarged print, calculate the width of the original painting. -/
theorem original_painting_width
  (original_height : ℝ)
  (print_height : ℝ)
  (print_width : ℝ)
  (h1 : original_height = 10)
  (h2 : print_height = 25)
  (h3 : print_width = 37.5) :
  print_width / (print_height / original_height) = 15 :=
by sorry

#check original_painting_width

end NUMINAMATH_CALUDE_original_painting_width_l2597_259727


namespace NUMINAMATH_CALUDE_min_value_expression_l2597_259769

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b > 0) (hsum : 2 * a + b = 1) :
  1 / a + 2 / b ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ ≠ 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2597_259769


namespace NUMINAMATH_CALUDE_hexagon_side_length_l2597_259709

/-- The side length of a regular hexagon given the distance between opposite sides -/
theorem hexagon_side_length (opposite_distance : ℝ) : 
  opposite_distance = 18 → 
  ∃ (side_length : ℝ), side_length = 12 * Real.sqrt 3 ∧ 
    opposite_distance = (Real.sqrt 3 / 2) * side_length :=
by sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l2597_259709


namespace NUMINAMATH_CALUDE_range_of_a_for_R_solution_set_l2597_259745

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (a - 2) * x^2 + 4 * (a - 2) * x - 4

-- Define the property that the solution set is ℝ
def solution_set_is_R (a : ℝ) : Prop := ∀ x, f a x < 0

-- Theorem statement
theorem range_of_a_for_R_solution_set :
  {a : ℝ | solution_set_is_R a} = Set.Ioo 1 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_R_solution_set_l2597_259745


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2597_259768

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (9 ∣ (n + 7)) ∧ 
  (6 ∣ (n - 9)) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (9 ∣ (m + 7)) ∧ (6 ∣ (m - 9))) → False) ∧
  n = 101 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2597_259768


namespace NUMINAMATH_CALUDE_truthful_dwarfs_l2597_259707

theorem truthful_dwarfs (n : ℕ) (h_n : n = 10) 
  (raised_vanilla raised_chocolate raised_fruit : ℕ)
  (h_vanilla : raised_vanilla = n)
  (h_chocolate : raised_chocolate = n / 2)
  (h_fruit : raised_fruit = 1) :
  ∃ (truthful liars : ℕ),
    truthful + liars = n ∧
    truthful + 2 * liars = raised_vanilla + raised_chocolate + raised_fruit ∧
    truthful = 4 ∧
    liars = 6 := by
  sorry

end NUMINAMATH_CALUDE_truthful_dwarfs_l2597_259707


namespace NUMINAMATH_CALUDE_smallest_a_value_l2597_259712

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ x : ℤ, Real.sin (a * x + b + π / 4) = Real.sin (15 * x + π / 4)) :
  a ≥ 15 ∧ ∃ a₀ : ℝ, a₀ = 15 ∧ a₀ ≥ 0 ∧
    ∀ x : ℤ, Real.sin (a₀ * x + π / 4) = Real.sin (15 * x + π / 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2597_259712


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l2597_259750

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem collinear_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  collinear a b → m = -4 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l2597_259750


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_p_false_and_q_true_l2597_259788

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - a*x + a > 0

def q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / (a^2 + 12) - y^2 / (4 - a^2) = 1

-- Theorem for the first part
theorem range_when_p_true :
  {a : ℝ | p a} = {a : ℝ | 0 < a ∧ a < 4} :=
sorry

-- Theorem for the second part
theorem range_when_p_false_and_q_true :
  {a : ℝ | ¬(p a) ∧ q a} = {a : ℝ | -2 < a ∧ a ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_p_false_and_q_true_l2597_259788


namespace NUMINAMATH_CALUDE_equation_solutions_l2597_259719

theorem equation_solutions :
  (∃ y₁ y₂ : ℝ, (2 * y₁^2 + 3 * y₁ - 1 = 0 ∧ 
                 2 * y₂^2 + 3 * y₂ - 1 = 0 ∧
                 y₁ = (-3 + Real.sqrt 17) / 4 ∧
                 y₂ = (-3 - Real.sqrt 17) / 4)) ∧
  (∃ x : ℝ, x * (x - 4) = -4 ∧ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2597_259719


namespace NUMINAMATH_CALUDE_sustainable_tree_planting_l2597_259773

theorem sustainable_tree_planting (trees_first_half trees_second_half trees_to_plant : ℕ) :
  trees_first_half = 200 →
  trees_second_half = 300 →
  trees_to_plant = 1500 →
  (trees_to_plant : ℚ) / (trees_first_half + trees_second_half : ℚ) = 3 := by
sorry

end NUMINAMATH_CALUDE_sustainable_tree_planting_l2597_259773


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2597_259752

-- Define the polynomial
def P (k x : ℝ) : ℝ := k * (x^2 - x) + x + 7

-- Define the condition for k1 and k2
def K_condition (k : ℝ) : Prop :=
  ∃ a b : ℝ, P k a = 0 ∧ P k b = 0 ∧ a/b + b/a = 3/7

-- State the theorem
theorem root_sum_theorem (k1 k2 : ℝ) :
  K_condition k1 ∧ K_condition k2 →
  k1/k2 + k2/k1 = 322 :=
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2597_259752


namespace NUMINAMATH_CALUDE_handshake_count_total_handshakes_l2597_259785

theorem handshake_count : ℕ :=
  let twin_sets : ℕ := 12
  let triplet_sets : ℕ := 8
  let twins : ℕ := twin_sets * 2
  let triplets : ℕ := triplet_sets * 3
  let twin_handshakes : ℕ := (twins * (twins - 2)) / 2
  let triplet_handshakes : ℕ := (triplets * (triplets - 3)) / 2
  let cross_handshakes : ℕ := twins * (2 * triplets / 3)
  twin_handshakes + triplet_handshakes + cross_handshakes

theorem total_handshakes : handshake_count = 900 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_total_handshakes_l2597_259785


namespace NUMINAMATH_CALUDE_tan_four_greater_than_tan_three_l2597_259748

theorem tan_four_greater_than_tan_three :
  π / 2 < 3 ∧ 3 < π ∧ π < 4 ∧ 4 < 3 * π / 2 →
  Real.tan 4 > Real.tan 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_four_greater_than_tan_three_l2597_259748


namespace NUMINAMATH_CALUDE_log_stack_count_15_5_l2597_259722

/-- The number of logs in a stack with a given bottom and top row count -/
def logStackCount (bottom top : ℕ) : ℕ :=
  let n := bottom - top + 1
  n * (bottom + top) / 2

/-- Theorem: A stack of logs with 15 on the bottom row and 5 on the top row has 110 logs -/
theorem log_stack_count_15_5 : logStackCount 15 5 = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_count_15_5_l2597_259722


namespace NUMINAMATH_CALUDE_ratio_problem_l2597_259758

theorem ratio_problem (a b c : ℝ) (h1 : a / b = 6 / 5) (h2 : b / c = 8 / 7) :
  (7 * a + 6 * b + 5 * c) / (7 * a - 6 * b + 5 * c) = 751 / 271 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2597_259758


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l2597_259729

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l2597_259729


namespace NUMINAMATH_CALUDE_hurricane_damage_conversion_l2597_259706

theorem hurricane_damage_conversion (damage_aud : ℝ) (exchange_rate : ℝ) : 
  damage_aud = 45000000 → 
  exchange_rate = 2 → 
  damage_aud / exchange_rate = 22500000 :=
by sorry

end NUMINAMATH_CALUDE_hurricane_damage_conversion_l2597_259706


namespace NUMINAMATH_CALUDE_min_value_of_rounded_sum_l2597_259780

-- Define the rounding functions
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  (roundToNearest (x * 10)) / 10

-- Define the main theorem
theorem min_value_of_rounded_sum (a b : ℝ) 
  (h1 : roundToNearestTenth a + roundToNearest b = 98.6)
  (h2 : roundToNearest a + roundToNearestTenth b = 99.3) :
  roundToNearest (10 * (a + b)) ≥ 988 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_rounded_sum_l2597_259780


namespace NUMINAMATH_CALUDE_min_pizzas_correct_l2597_259793

/-- The cost of the car John bought -/
def car_cost : ℕ := 5000

/-- The amount John earns per pizza delivered -/
def earnings_per_pizza : ℕ := 10

/-- The amount John spends on gas per pizza delivered -/
def gas_cost_per_pizza : ℕ := 3

/-- The net profit John makes per pizza delivered -/
def net_profit_per_pizza : ℕ := earnings_per_pizza - gas_cost_per_pizza

/-- The minimum number of pizzas John must deliver to earn back the car cost -/
def min_pizzas : ℕ := (car_cost + net_profit_per_pizza - 1) / net_profit_per_pizza

theorem min_pizzas_correct :
  min_pizzas * net_profit_per_pizza ≥ car_cost ∧
  ∀ n : ℕ, n < min_pizzas → n * net_profit_per_pizza < car_cost :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_correct_l2597_259793


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2597_259714

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q^(n - 1)

/-- The sequence is increasing -/
def is_increasing (a : ℕ → ℝ) : Prop := ∀ n, a n < a (n + 1)

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∃ a₁ q, a = geometric_sequence a₁ q) 
  (h_a₁ : a 1 = -2)
  (h_inc : is_increasing a)
  (h_eq : ∀ n, 3 * (a n + a (n + 2)) = 10 * a (n + 1)) :
  ∃ q, a = geometric_sequence (-2) q ∧ q = 1/3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2597_259714


namespace NUMINAMATH_CALUDE_horner_method_proof_polynomial_value_at_2_l2597_259741

def horner_polynomial (x : ℝ) : ℝ :=
  ((((2 * x + 4) * x - 2) * x - 3) * x + 1) * x

theorem horner_method_proof :
  horner_polynomial 2 = 2 * 2^5 - 3 * 2^2 + 4 * 2^4 - 2 * 2^3 + 2 :=
by sorry

theorem polynomial_value_at_2 :
  horner_polynomial 2 = 102 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_proof_polynomial_value_at_2_l2597_259741


namespace NUMINAMATH_CALUDE_polynomial_square_condition_l2597_259700

theorem polynomial_square_condition (P : Polynomial ℤ) : 
  (∃ R : Polynomial ℤ, (X^2 + 6*X + 10) * P^2 - 1 = R^2) → P = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_square_condition_l2597_259700


namespace NUMINAMATH_CALUDE_calculate_expression_l2597_259723

theorem calculate_expression : 150 * (150 - 5) + (150 * 150 + 5) = 44255 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2597_259723


namespace NUMINAMATH_CALUDE_intersection_M_N_l2597_259726

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2597_259726


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l2597_259725

/-- Calculates the number of female students in a stratified sample -/
def femaleInSample (totalPopulation malePopulation sampleSize : ℕ) : ℕ :=
  let femalePopulation := totalPopulation - malePopulation
  (femalePopulation * sampleSize) / totalPopulation

theorem stratified_sample_female_count :
  femaleInSample 900 500 45 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_female_count_l2597_259725


namespace NUMINAMATH_CALUDE_min_throws_for_repeat_sum_l2597_259795

/-- Represents a fair six-sided die -/
def Die : Type := Fin 6

/-- The sum of four dice rolls -/
def DiceSum : Type := Nat

/-- The minimum possible sum when rolling four dice -/
def minSum : Nat := 4

/-- The maximum possible sum when rolling four dice -/
def maxSum : Nat := 24

/-- The number of possible unique sums when rolling four dice -/
def uniqueSums : Nat := maxSum - minSum + 1

/-- 
  Theorem: The minimum number of throws needed to ensure the same sum 
  is rolled twice with four fair six-sided dice is 22.
-/
theorem min_throws_for_repeat_sum : 
  (uniqueSums + 1 : Nat) = 22 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeat_sum_l2597_259795


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2597_259704

/-- Given a line with equation 4x - 5y = 20, the slope of the perpendicular line is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 20) → 
  ∃ (m : ℝ), m = -5/4 ∧ m * (4/5) = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2597_259704


namespace NUMINAMATH_CALUDE_number_equals_two_thirds_a_l2597_259763

/-- Given a and n are real numbers satisfying certain conditions, 
    prove that n equals 2a/3 -/
theorem number_equals_two_thirds_a (a n : ℝ) 
  (h1 : 2 * a = 3 * n) 
  (h2 : a * n ≠ 0) 
  (h3 : (a / 3) / (n / 2) = 1) : 
  n = 2 * a / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_two_thirds_a_l2597_259763


namespace NUMINAMATH_CALUDE_cos_B_value_angle_A_value_projection_BC_BA_l2597_259711

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi
axiom side_angle_correspondence : a = 2 * Real.sin (A / 2) ∧ 
                                  b = 2 * Real.sin (B / 2) ∧ 
                                  c = 2 * Real.sin (C / 2)
axiom line_condition : 2 * a * Real.cos B - b * Real.cos C = c * Real.cos B

-- Define the specific values for a and b
axiom a_value : a = 2 * Real.sqrt 3 / 3
axiom b_value : b = 2

-- Theorem statements
theorem cos_B_value : Real.cos B = 1 / 2 := by sorry

theorem angle_A_value : A = Real.arccos (Real.sqrt 3 / 3) := by sorry

theorem projection_BC_BA : a * Real.cos B = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_cos_B_value_angle_A_value_projection_BC_BA_l2597_259711


namespace NUMINAMATH_CALUDE_frog_probability_l2597_259784

-- Define the number of lily pads
def num_pads : ℕ := 9

-- Define the set of predator positions
def predator_positions : Set ℕ := {2, 5, 6}

-- Define the target position
def target_position : ℕ := 7

-- Define the probability of moving 1 or 2 positions
def move_probability : ℚ := 1/2

-- Define the function to calculate the probability of reaching the target
def reach_probability (start : ℕ) (target : ℕ) (predators : Set ℕ) (p : ℚ) : ℚ :=
  sorry

-- Theorem statement
theorem frog_probability :
  reach_probability 0 target_position predator_positions move_probability = 1/16 :=
sorry

end NUMINAMATH_CALUDE_frog_probability_l2597_259784


namespace NUMINAMATH_CALUDE_equation_equality_l2597_259760

theorem equation_equality : 2 * 18 * 14 = 6 * 12 * 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l2597_259760


namespace NUMINAMATH_CALUDE_fraction_addition_l2597_259720

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2597_259720


namespace NUMINAMATH_CALUDE_arbor_day_planting_l2597_259739

theorem arbor_day_planting (class_average : ℝ) (girls_trees : ℝ) (boys_trees : ℝ) :
  class_average = 6 →
  girls_trees = 15 →
  (1 / boys_trees + 1 / girls_trees = 1 / class_average) →
  boys_trees = 10 := by
  sorry

end NUMINAMATH_CALUDE_arbor_day_planting_l2597_259739


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2597_259774

theorem geometric_sequence_sum (a r : ℚ) (n : ℕ) (h1 : a = 1/2) (h2 : r = 1/3) (h3 : n = 8) :
  (a * (1 - r^n)) / (1 - r) = 9840/6561 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2597_259774


namespace NUMINAMATH_CALUDE_public_transportation_users_l2597_259754

/-- Calculates the number of employees using public transportation -/
theorem public_transportation_users
  (total_employees : ℕ)
  (drive_percentage : ℚ)
  (public_transport_fraction : ℚ)
  (h1 : total_employees = 100)
  (h2 : drive_percentage = 60 / 100)
  (h3 : public_transport_fraction = 1 / 2) :
  ⌊(total_employees : ℚ) * (1 - drive_percentage) * public_transport_fraction⌋ = 20 := by
  sorry

end NUMINAMATH_CALUDE_public_transportation_users_l2597_259754


namespace NUMINAMATH_CALUDE_rectangleB_is_top_leftmost_l2597_259732

-- Define a structure for rectangles
structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the six rectangles
def rectangleA : Rectangle := ⟨2, 7, 4, 7⟩
def rectangleB : Rectangle := ⟨0, 6, 8, 5⟩
def rectangleC : Rectangle := ⟨6, 3, 1, 1⟩
def rectangleD : Rectangle := ⟨8, 4, 0, 2⟩
def rectangleE : Rectangle := ⟨5, 9, 3, 6⟩
def rectangleF : Rectangle := ⟨7, 5, 9, 0⟩

-- Define a function to check if a rectangle is leftmost
def isLeftmost (r : Rectangle) : Prop :=
  ∀ other : Rectangle, r.w ≤ other.w

-- Define a function to check if a rectangle is topmost among leftmost rectangles
def isTopmostLeftmost (r : Rectangle) : Prop :=
  isLeftmost r ∧ ∀ other : Rectangle, isLeftmost other → r.y ≥ other.y

-- Theorem stating that Rectangle B is the top leftmost rectangle
theorem rectangleB_is_top_leftmost :
  isTopmostLeftmost rectangleB :=
sorry


end NUMINAMATH_CALUDE_rectangleB_is_top_leftmost_l2597_259732


namespace NUMINAMATH_CALUDE_min_radius_for_area_l2597_259721

/-- The minimum radius of a circle with an area of at least 314 square feet is 10 feet. -/
theorem min_radius_for_area (π : ℝ) (h : π > 0) : 
  (∀ r : ℝ, π * r^2 ≥ 314 → r ≥ 10) ∧ (∃ r : ℝ, π * r^2 = 314 ∧ r = 10) := by
  sorry

end NUMINAMATH_CALUDE_min_radius_for_area_l2597_259721


namespace NUMINAMATH_CALUDE_sequence_problem_l2597_259790

/-- An arithmetic sequence where no term is 0 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m ∧ ∀ k, a k ≠ 0

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, b (n + 1) / b n = b (m + 1) / b m

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_eq : a 5 - a 7 ^ 2 + a 9 = 0)
    (h_b7 : b 7 = a 7) :
  b 2 * b 8 * b 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l2597_259790


namespace NUMINAMATH_CALUDE_multiply_57_47_l2597_259789

theorem multiply_57_47 : 57 * 47 = 2820 := by
  sorry

end NUMINAMATH_CALUDE_multiply_57_47_l2597_259789


namespace NUMINAMATH_CALUDE_min_value_fraction_l2597_259761

theorem min_value_fraction (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_eq_one : x + y + z + w = 1) :
  ∀ a b c d : ℝ, 0 < a → 0 < b → 0 < c → 0 < d → a + b + c + d = 1 →
  (x + y + z) / (x * y * z * w) ≤ (a + b + c) / (a * b * c * d) ∧
  (x + y + z) / (x * y * z * w) = 144 := by
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2597_259761


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l2597_259718

theorem polynomial_sum_theorem (d : ℝ) :
  let f : ℝ → ℝ := λ x => 15 * x^3 + 17 * x + 18 + 19 * x^2
  let g : ℝ → ℝ := λ x => 3 * x^3 + 4 * x + 2
  ∃ (p q r s : ℤ),
    (∀ x, f x + g x = p * x^3 + q * x^2 + r * x + s) ∧
    p + q + r + s = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l2597_259718


namespace NUMINAMATH_CALUDE_q_subset_p_intersect_q_iff_a_in_range_l2597_259764

-- Define sets P and Q
def P : Set ℝ := {x | 3 < x ∧ x ≤ 22}
def Q (a : ℝ) : Set ℝ := {x | 2 * a + 1 ≤ x ∧ x < 3 * a - 5}

-- State the theorem
theorem q_subset_p_intersect_q_iff_a_in_range :
  ∀ a : ℝ, (Q a).Nonempty → (Q a ⊆ (P ∩ Q a) ↔ 6 < a ∧ a ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_q_subset_p_intersect_q_iff_a_in_range_l2597_259764


namespace NUMINAMATH_CALUDE_connie_red_markers_l2597_259791

theorem connie_red_markers (total_markers blue_markers : ℕ) 
  (h1 : total_markers = 3343)
  (h2 : blue_markers = 1028) :
  total_markers - blue_markers = 2315 :=
by
  sorry

end NUMINAMATH_CALUDE_connie_red_markers_l2597_259791


namespace NUMINAMATH_CALUDE_reinforcement_arrival_time_l2597_259783

/-- Calculates the number of days that passed before reinforcement arrived --/
def days_before_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) 
  (reinforcement : ℕ) (remaining_provisions : ℕ) : ℕ :=
  (initial_garrison * initial_provisions - (initial_garrison + reinforcement) * remaining_provisions) / initial_garrison

/-- Theorem stating that 21 days passed before reinforcement arrived --/
theorem reinforcement_arrival_time : 
  days_before_reinforcement 2000 54 1300 20 = 21 := by
  sorry


end NUMINAMATH_CALUDE_reinforcement_arrival_time_l2597_259783


namespace NUMINAMATH_CALUDE_base_eight_1263_equals_691_l2597_259777

def base_eight_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_eight_1263_equals_691 :
  base_eight_to_ten [3, 6, 2, 1] = 691 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_1263_equals_691_l2597_259777


namespace NUMINAMATH_CALUDE_hyperbola_rational_parameterization_l2597_259753

theorem hyperbola_rational_parameterization
  (x p q : ℚ) 
  (h : p^2 - x*q^2 = 1) :
  ∃ (a b : ℤ), 
    p = (a^2 + x*b^2) / (a^2 - x*b^2) ∧
    q = 2*a*b / (a^2 - x*b^2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_rational_parameterization_l2597_259753


namespace NUMINAMATH_CALUDE_delegate_seating_probability_l2597_259736

/-- Represents the number of delegates -/
def num_delegates : ℕ := 8

/-- Represents the number of countries -/
def num_countries : ℕ := 4

/-- Represents the number of delegates per country -/
def delegates_per_country : ℕ := 2

/-- Represents the number of seats at the round table -/
def num_seats : ℕ := 8

/-- Calculates the total number of possible seating arrangements -/
def total_arrangements : ℕ := num_delegates.factorial / (delegates_per_country.factorial ^ num_countries)

/-- Calculates the number of favorable seating arrangements -/
def favorable_arrangements : ℕ := total_arrangements - 324

/-- The probability that each delegate sits next to at least one delegate from another country -/
def probability : ℚ := favorable_arrangements / total_arrangements

theorem delegate_seating_probability :
  probability = 131 / 140 := by sorry

end NUMINAMATH_CALUDE_delegate_seating_probability_l2597_259736


namespace NUMINAMATH_CALUDE_ellipse_dot_product_constant_l2597_259746

/-- The ellipse with semi-major axis 2 and semi-minor axis √2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 2) = 1}

/-- The line x = 2 -/
def Line : Set (ℝ × ℝ) :=
  {p | p.1 = 2}

/-- Left vertex of the ellipse -/
def A₁ : ℝ × ℝ := (-2, 0)

/-- Theorem: For any point C on the ellipse and D on the line x = 2,
    if A₁C = 2CD, then OC · OD = 4 -/
theorem ellipse_dot_product_constant
    (C : ℝ × ℝ) (hC : C ∈ Ellipse)
    (D : ℝ × ℝ) (hD : D ∈ Line)
    (h : dist A₁ C = 2 * dist C D) :
  C.1 * D.1 + C.2 * D.2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_dot_product_constant_l2597_259746


namespace NUMINAMATH_CALUDE_square_of_rational_l2597_259734

theorem square_of_rational (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by
sorry

end NUMINAMATH_CALUDE_square_of_rational_l2597_259734


namespace NUMINAMATH_CALUDE_friend_age_problem_l2597_259767

theorem friend_age_problem (A B C : ℕ) 
  (h1 : A - B = 2)
  (h2 : A - C = 5)
  (h3 : A + B + C = 110) :
  A = 39 := by
sorry

end NUMINAMATH_CALUDE_friend_age_problem_l2597_259767


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2597_259756

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) : 
  1/a + 4/b + 9/c + 16/d + 25/e + 36/f ≥ 441/10 ∧
  ∃ a' b' c' d' e' f' : ℝ, 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧
    a' + b' + c' + d' + e' + f' = 10 ∧
    1/a' + 4/b' + 9/c' + 16/d' + 25/e' + 36/f' = 441/10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2597_259756


namespace NUMINAMATH_CALUDE_sarah_score_l2597_259759

theorem sarah_score (s g : ℕ) (h1 : s = g + 30) (h2 : (s + g) / 2 = 108) : s = 123 := by
  sorry

end NUMINAMATH_CALUDE_sarah_score_l2597_259759


namespace NUMINAMATH_CALUDE_first_question_percentage_l2597_259787

theorem first_question_percentage
  (second_correct : Real)
  (neither_correct : Real)
  (both_correct : Real)
  (h1 : second_correct = 0.3)
  (h2 : neither_correct = 0.2)
  (h3 : both_correct = 0.25) :
  ∃ (first_correct : Real),
    first_correct = 0.75 ∧
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by sorry

end NUMINAMATH_CALUDE_first_question_percentage_l2597_259787


namespace NUMINAMATH_CALUDE_factorial_a_ratio_l2597_259779

/-- Definition of n_a! for positive n and a -/
def factorial_a (n a : ℕ) : ℕ :=
  (List.range ((n / a) + 1)).foldl (fun acc k => acc * (n - k * a)) n

/-- Theorem stating that 96_4! / 48_3! = 2^8 -/
theorem factorial_a_ratio : (factorial_a 96 4) / (factorial_a 48 3) = 2^8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_a_ratio_l2597_259779


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2597_259778

def f (a b x : ℚ) : ℚ := a * x^3 - 7 * x^2 + b * x - 6

theorem polynomial_remainder (a b : ℚ) :
  (f a b 2 = -8) ∧ (f a b (-1) = -18) → a = 2/3 ∧ b = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2597_259778


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_sum_l2597_259733

/-- Two different two-digit positive integers -/
def is_valid_pair (x y : ℕ) : Prop :=
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x ≠ y

/-- All four digits in the two numbers are unique -/
def has_unique_digits (x y : ℕ) : Prop :=
  let digits := [x / 10, x % 10, y / 10, y % 10]
  List.Nodup digits

/-- The sum is a two-digit number -/
def is_two_digit_sum (x y : ℕ) : Prop :=
  10 ≤ x + y ∧ x + y < 100

/-- The sum of digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

/-- The main theorem -/
theorem smallest_digit_sum_of_sum :
  ∃ (x y : ℕ), 
    is_valid_pair x y ∧ 
    has_unique_digits x y ∧ 
    is_two_digit_sum x y ∧
    ∀ (a b : ℕ), 
      is_valid_pair a b → 
      has_unique_digits a b → 
      is_two_digit_sum a b → 
      digit_sum (x + y) ≤ digit_sum (a + b) ∧
      digit_sum (x + y) = 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_sum_l2597_259733


namespace NUMINAMATH_CALUDE_cosine_sum_constant_l2597_259771

theorem cosine_sum_constant (A B C : ℝ) 
  (h1 : Real.cos A + Real.cos B + Real.cos C = 0)
  (h2 : Real.sin A + Real.sin B + Real.sin C = 0) : 
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_constant_l2597_259771


namespace NUMINAMATH_CALUDE_x_value_and_upper_bound_l2597_259708

theorem x_value_and_upper_bound :
  ∀ (x : ℤ) (u : ℚ),
    0 < x ∧ x < 7 ∧
    0 < x ∧ x < 15 ∧
    -1 < x ∧ x < 5 ∧
    0 < x ∧ x < u ∧
    x + 2 < 4 →
    x = 1 ∧ 1 < u ∧ u < 2 :=
by sorry

end NUMINAMATH_CALUDE_x_value_and_upper_bound_l2597_259708


namespace NUMINAMATH_CALUDE_problem_solution_l2597_259757

theorem problem_solution : 
  ((-0.125 : ℝ)^2023 * 8^2024 = -8) ∧ 
  (((-27 : ℝ)^(1/3 : ℝ) + (5^2 : ℝ)^(1/2 : ℝ) - 2/3 * ((9/4 : ℝ)^(1/2 : ℝ))) = 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2597_259757


namespace NUMINAMATH_CALUDE_original_number_proof_l2597_259770

theorem original_number_proof (numbers : Finset ℕ) (original_sum : ℕ) (changed_sum : ℕ) : 
  numbers.card = 7 →
  original_sum / numbers.card = 7 →
  changed_sum / numbers.card = 8 →
  changed_sum = original_sum - (original_sum / numbers.card) + 9 →
  original_sum / numbers.card = 2 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2597_259770


namespace NUMINAMATH_CALUDE_wine_pouring_equivalence_l2597_259751

/-- Represents the state of the four glasses --/
structure GlassState :=
  (glass1 : ℕ)
  (glass2 : ℕ)
  (glass3 : ℕ)
  (glass4 : ℕ)

/-- Represents a single pouring operation --/
inductive PourOperation
  | pour1to2
  | pour1to3
  | pour1to4
  | pour2to1
  | pour2to3
  | pour2to4
  | pour3to1
  | pour3to2
  | pour3to4
  | pour4to1
  | pour4to2
  | pour4to3

/-- Applies a single pouring operation to a glass state --/
def applyOperation (state : GlassState) (op : PourOperation) (m n k : ℕ) : GlassState :=
  sorry

/-- Checks if a specific amount can be achieved in any glass --/
def canAchieveAmount (m n k s : ℕ) : Prop :=
  ∃ (operations : List PourOperation),
    let finalState := operations.foldl (λ state op => applyOperation state op m n k)
                        (GlassState.mk 0 0 0 (m + n + k))
    finalState.glass1 = s ∨ finalState.glass2 = s ∨ finalState.glass3 = s ∨ finalState.glass4 = s

/-- The main theorem stating the equivalence --/
theorem wine_pouring_equivalence (m n k : ℕ) :
  (∀ s : ℕ, s < m + n + k → canAchieveAmount m n k s) ↔ Nat.gcd m (Nat.gcd n k) = 1 :=
sorry

end NUMINAMATH_CALUDE_wine_pouring_equivalence_l2597_259751


namespace NUMINAMATH_CALUDE_exam_candidates_girls_l2597_259744

theorem exam_candidates_girls (total : ℕ) (boys_pass_rate girls_pass_rate fail_rate : ℚ) :
  total = 2000 ∧
  boys_pass_rate = 28/100 ∧
  girls_pass_rate = 32/100 ∧
  fail_rate = 702/1000 →
  ∃ (girls : ℕ), 
    girls + (total - girls) = total ∧
    (girls_pass_rate * girls + boys_pass_rate * (total - girls)) / total = 1 - fail_rate ∧
    girls = 900 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_girls_l2597_259744


namespace NUMINAMATH_CALUDE_cube_has_six_faces_l2597_259737

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- The number of faces of a cube -/
def num_faces (c : Cube) : ℕ := 6

/-- Theorem: A cube has 6 faces -/
theorem cube_has_six_faces (c : Cube) : num_faces c = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_has_six_faces_l2597_259737


namespace NUMINAMATH_CALUDE_equation_solution_l2597_259738

theorem equation_solution (x : ℝ) (h : x + 2 ≠ 0) :
  (x / (x + 2) + 1 = 1 / (x + 2)) ↔ (x = -1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2597_259738


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_twenty_fourth_l2597_259796

theorem reciprocal_of_negative_one_twenty_fourth :
  ((-1 / 24)⁻¹ : ℚ) = -24 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_twenty_fourth_l2597_259796


namespace NUMINAMATH_CALUDE_intersection_A_B_l2597_259717

def A : Set ℝ := {x : ℝ | |x - 2| < 1}
def B : Set ℝ := Set.range (Int.cast : ℤ → ℝ)

theorem intersection_A_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2597_259717


namespace NUMINAMATH_CALUDE_shelter_adoption_rate_l2597_259730

def puppies_adopted_per_day (initial_puppies : ℕ) (additional_puppies : ℕ) (total_days : ℕ) : ℕ :=
  (initial_puppies + additional_puppies) / total_days

theorem shelter_adoption_rate :
  puppies_adopted_per_day 9 12 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_shelter_adoption_rate_l2597_259730


namespace NUMINAMATH_CALUDE_vector_sum_zero_l2597_259710

variable {V : Type*} [AddCommGroup V]

def closed_polygon (a b c f : V) : Prop :=
  a + (c - b) + (f - c) + (b - f) = 0

theorem vector_sum_zero (a b c f : V) (h : closed_polygon a b c f) :
  (b - a) + (f - c) + (c - b) + (a - f) = 0 := by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l2597_259710


namespace NUMINAMATH_CALUDE_original_class_size_l2597_259775

theorem original_class_size (N : ℕ) : 
  (N > 0) →
  (40 * N + 8 * 32 = 36 * (N + 8)) →
  N = 8 := by
sorry

end NUMINAMATH_CALUDE_original_class_size_l2597_259775


namespace NUMINAMATH_CALUDE_square_side_length_l2597_259743

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 6)
  (h2 : rectangle_length = 24)
  (h3 : rectangle_width > 0)
  (h4 : rectangle_length > 0) :
  ∃ (square_side : ℝ), 
    square_side ^ 2 = rectangle_width * rectangle_length ∧ 
    square_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2597_259743


namespace NUMINAMATH_CALUDE_function_symmetry_l2597_259792

/-- Given a function f(x) = ax^4 + b*cos(x) - x, if f(-3) = 7, then f(3) = 1 -/
theorem function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^4 + b * Real.cos x - x
  f (-3) = 7 → f 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2597_259792


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_odds_l2597_259742

def is_sum_of_five_consecutive_odds (n : ℤ) : Prop :=
  ∃ k : ℤ, n = (2*k-3) + (2*k-1) + (2*k+1) + (2*k+3) + (2*k+5)

theorem sum_of_five_consecutive_odds :
  is_sum_of_five_consecutive_odds 25 ∧
  is_sum_of_five_consecutive_odds 55 ∧
  is_sum_of_five_consecutive_odds 85 ∧
  is_sum_of_five_consecutive_odds 105 ∧
  ¬ is_sum_of_five_consecutive_odds 150 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_odds_l2597_259742


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2597_259735

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: Eccentricity of a special hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (hexagon_condition : ∃ (c : ℝ), c > 0 ∧ 
    (∃ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ 
      x^2 + y^2 = c^2 ∧ 
      -- The following condition represents that the intersections form a regular hexagon
      2 * h.a = (Real.sqrt 3 - 1) * c)) :
  eccentricity h = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2597_259735


namespace NUMINAMATH_CALUDE_bear_buns_l2597_259703

theorem bear_buns (x : ℚ) : 
  (x / 8 - 7 / 8 = 0) → x = 7 := by sorry

end NUMINAMATH_CALUDE_bear_buns_l2597_259703


namespace NUMINAMATH_CALUDE_garden_ratio_l2597_259765

/-- A rectangular garden with given perimeter and length has a specific length-to-width ratio -/
theorem garden_ratio (perimeter : ℝ) (length : ℝ) (width : ℝ) 
  (h_perimeter : perimeter = 900)
  (h_length : length = 300)
  (h_rectangle : perimeter = 2 * (length + width)) : 
  length / width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l2597_259765


namespace NUMINAMATH_CALUDE_selection_schemes_count_l2597_259713

/-- The number of people in the group -/
def totalPeople : ℕ := 5

/-- The number of cities to be visited -/
def totalCities : ℕ := 4

/-- The number of people who can visit Paris (excluding A) -/
def parisVisitors : ℕ := totalPeople - 1

/-- Calculate the number of selection schemes -/
def selectionSchemes : ℕ :=
  parisVisitors * (totalPeople - 1) * (totalPeople - 2) * (totalPeople - 3)

/-- Theorem stating the number of selection schemes is 96 -/
theorem selection_schemes_count :
  selectionSchemes = 96 := by sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l2597_259713


namespace NUMINAMATH_CALUDE_base9_3671_equals_base10_2737_l2597_259731

def base9_to_base10 (n : Nat) : Nat :=
  (n / 1000) * (9^3) + ((n / 100) % 10) * (9^2) + ((n / 10) % 10) * 9 + (n % 10)

theorem base9_3671_equals_base10_2737 :
  base9_to_base10 3671 = 2737 := by
  sorry

end NUMINAMATH_CALUDE_base9_3671_equals_base10_2737_l2597_259731


namespace NUMINAMATH_CALUDE_carolyn_initial_marbles_l2597_259799

/-- The number of marbles Carolyn shared with Diana -/
def marbles_shared : ℕ := 42

/-- The number of marbles Carolyn had left after sharing -/
def marbles_left : ℕ := 5

/-- The number of oranges Carolyn started with (not used in the proof, but mentioned in the problem) -/
def initial_oranges : ℕ := 6

/-- Carolyn's initial number of marbles -/
def initial_marbles : ℕ := marbles_shared + marbles_left

theorem carolyn_initial_marbles :
  initial_marbles = 47 :=
by sorry

end NUMINAMATH_CALUDE_carolyn_initial_marbles_l2597_259799


namespace NUMINAMATH_CALUDE_equation_solution_l2597_259794

theorem equation_solution : 
  ∃! x : ℚ, (3 * x - 1) / (4 * x - 4) = 2 / 3 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2597_259794


namespace NUMINAMATH_CALUDE_s_range_for_composites_l2597_259762

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def s (n : ℕ) : ℕ := sorry

theorem s_range_for_composites :
  (∀ n : ℕ, is_composite n → s n ≥ 12) ∧
  (∀ m : ℕ, m ≥ 12 → ∃ n : ℕ, is_composite n ∧ s n = m) :=
sorry

end NUMINAMATH_CALUDE_s_range_for_composites_l2597_259762


namespace NUMINAMATH_CALUDE_product_of_1101_base2_and_102_base3_l2597_259766

def base2_to_dec (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, b) acc => acc + b * 2^i) 0

def base3_to_dec (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, b) acc => acc + b * 3^i) 0

theorem product_of_1101_base2_and_102_base3 :
  let n1 := base2_to_dec [1, 0, 1, 1]
  let n2 := base3_to_dec [2, 0, 1]
  n1 * n2 = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_of_1101_base2_and_102_base3_l2597_259766


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2597_259782

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2597_259782


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2597_259715

theorem largest_prime_divisor_to_test (n : ℕ) : 
  1000 ≤ n ∧ n ≤ 1050 → 
  (∀ p : ℕ, Prime p ∧ p ≤ 31 → ¬(p ∣ n)) → 
  Prime n ∨ n = 1 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2597_259715


namespace NUMINAMATH_CALUDE_complex_sum_powers_l2597_259798

theorem complex_sum_powers (w : ℂ) (hw : w^2 - w + 1 = 0) : 
  w^101 + w^102 + w^103 + w^104 + w^105 = 4*w - 1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l2597_259798


namespace NUMINAMATH_CALUDE_function_inequality_l2597_259781

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : is_periodic f 2)
  (h_monotone : is_monotone_decreasing f (-1) 0)
  (a b c : ℝ)
  (h_a : a = f (-2.8))
  (h_b : b = f (-1.6))
  (h_c : c = f 0.5) :
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2597_259781


namespace NUMINAMATH_CALUDE_two_numbers_satisfy_conditions_l2597_259701

/-- A function that checks if a number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- A function that checks if a two-digit number is a square --/
def isTwoDigitSquare (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ isPerfectSquare n

/-- A function that checks if a number is a single-digit square (1, 4, or 9) --/
def isSingleDigitSquare (n : ℕ) : Prop :=
  n = 1 ∨ n = 4 ∨ n = 9

/-- A function that returns the first two digits of a five-digit number --/
def firstTwoDigits (n : ℕ) : ℕ :=
  n / 1000

/-- A function that returns the sum of the third and fourth digits of a five-digit number --/
def sumMiddleTwoDigits (n : ℕ) : ℕ :=
  (n / 100 % 10) + (n / 10 % 10)

/-- A function that checks if a number satisfies all the given conditions --/
def satisfiesAllConditions (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧  -- five-digit number
  (∀ d, d ∈ [n / 10000, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] → d ≠ 0) ∧  -- no digit is zero
  isPerfectSquare n ∧  -- perfect square
  isTwoDigitSquare (firstTwoDigits n) ∧  -- first two digits form a square
  isSingleDigitSquare (sumMiddleTwoDigits n) ∧  -- sum of middle two digits is a single-digit square
  n % 7 = 0  -- divisible by 7

/-- The main theorem stating that exactly two numbers satisfy all conditions --/
theorem two_numbers_satisfy_conditions : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfiesAllConditions n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_two_numbers_satisfy_conditions_l2597_259701


namespace NUMINAMATH_CALUDE_g_zero_at_three_l2597_259740

def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + x^3 - 4 * x^2 + 5 * x + s

theorem g_zero_at_three (s : ℝ) : g 3 s = 0 ↔ s = -573 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_three_l2597_259740


namespace NUMINAMATH_CALUDE_fraction_equality_l2597_259705

theorem fraction_equality : (1^4 + 2009^4 + 2010^4) / (1^2 + 2009^2 + 2010^2) = 4038091 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2597_259705


namespace NUMINAMATH_CALUDE_expression_evaluation_l2597_259747

theorem expression_evaluation : (3 * 10^9) / (6 * 10^5) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2597_259747


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l2597_259772

theorem complex_magnitude_one (z : ℂ) (h : 3 * z^6 + 2 * Complex.I * z^5 - 2 * z - 3 * Complex.I = 0) : 
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l2597_259772


namespace NUMINAMATH_CALUDE_unique_square_sum_pair_l2597_259776

theorem unique_square_sum_pair : 
  ∃! (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    (∃ (m n : ℕ), 100 * a + b = m^2 ∧ 201 * a + b = n^2) ∧
    a = 17 ∧ b = 64 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_sum_pair_l2597_259776


namespace NUMINAMATH_CALUDE_solution_set_a_gt_1_solution_set_a_eq_1_solution_set_a_lt_1_a_range_subset_l2597_259728

-- Define the inequality function
def f (a x : ℝ) : ℝ := (a * x - (a - 2)) * (x + 1)

-- Define the solution set P
def P (a : ℝ) : Set ℝ := {x | f a x > 0}

-- Theorem for the solution set when a > 1
theorem solution_set_a_gt_1 (a : ℝ) (h : a > 1) :
  P a = {x | x < -1 ∨ x > (a - 2) / a} := by sorry

-- Theorem for the solution set when a = 1
theorem solution_set_a_eq_1 :
  P 1 = {x | x ≠ -1} := by sorry

-- Theorem for the solution set when 0 < a < 1
theorem solution_set_a_lt_1 (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  P a = {x | x < (a - 2) / a ∨ x > -1} := by sorry

-- Theorem for the range of a when {x | -3 < x < -1} ⊆ P
theorem a_range_subset (a : ℝ) (h : {x : ℝ | -3 < x ∧ x < -1} ⊆ P a) :
  a ∈ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_gt_1_solution_set_a_eq_1_solution_set_a_lt_1_a_range_subset_l2597_259728


namespace NUMINAMATH_CALUDE_eight_to_one_l2597_259786

theorem eight_to_one : (8/8)^(8/8) * 8/8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eight_to_one_l2597_259786


namespace NUMINAMATH_CALUDE_william_land_percentage_l2597_259724

-- Define the tax amounts
def total_village_tax : ℝ := 3840
def william_tax : ℝ := 480

-- Define the theorem
theorem william_land_percentage :
  william_tax / total_village_tax * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_william_land_percentage_l2597_259724
