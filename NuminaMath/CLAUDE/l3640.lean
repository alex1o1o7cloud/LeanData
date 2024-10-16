import Mathlib

namespace NUMINAMATH_CALUDE_xy_value_l3640_364082

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x+y) = 16)
  (h2 : (27 : ℝ)^(x+y) / (9 : ℝ)^(4*y) = 729) : 
  x * y = 48 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l3640_364082


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l3640_364065

/-- Represents the cost and quantity of prizes A and B --/
structure PrizePurchase where
  costA : ℕ  -- Cost of prize A
  costB : ℕ  -- Cost of prize B
  quantityA : ℕ  -- Quantity of prize A
  quantityB : ℕ  -- Quantity of prize B

/-- Conditions for the prize purchase problem --/
def PrizePurchaseConditions (p : PrizePurchase) : Prop :=
  3 * p.costA + 2 * p.costB = 390 ∧  -- Condition 1
  4 * p.costA = 5 * p.costB + 60 ∧  -- Condition 2
  p.quantityA + p.quantityB = 30 ∧  -- Condition 3
  p.quantityA ≥ p.quantityB / 2 ∧  -- Condition 4
  p.costA * p.quantityA + p.costB * p.quantityB ≤ 2170  -- Condition 5

/-- The theorem to be proved --/
theorem minimum_cost_theorem (p : PrizePurchase) 
  (h : PrizePurchaseConditions p) : 
  p.costA = 90 ∧ p.costB = 60 ∧ 
  p.quantityA * p.costA + p.quantityB * p.costB ≥ 2100 :=
sorry

end NUMINAMATH_CALUDE_minimum_cost_theorem_l3640_364065


namespace NUMINAMATH_CALUDE_equation_solution_set_l3640_364057

theorem equation_solution_set : 
  {x : ℝ | x^6 + x^2 = (2*x + 3)^3 + 2*x + 3} = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3640_364057


namespace NUMINAMATH_CALUDE_part_one_part_two_l3640_364056

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a|

-- Part 1
theorem part_one (a : ℝ) :
  (∀ x, f a x ≥ |2*x + 3| ↔ x ∈ Set.Icc (-3) (-1)) →
  a = 0 :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x, f a x + |x - a| ≥ a^2 - 2*a) →
  0 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3640_364056


namespace NUMINAMATH_CALUDE_stamp_trade_l3640_364089

theorem stamp_trade (anna_initial alison_initial jeff_initial anna_final : ℕ) 
  (h1 : anna_initial = 37)
  (h2 : alison_initial = 28)
  (h3 : jeff_initial = 31)
  (h4 : anna_final = 50) : 
  (anna_initial + alison_initial / 2) - anna_final = 1 := by
  sorry

end NUMINAMATH_CALUDE_stamp_trade_l3640_364089


namespace NUMINAMATH_CALUDE_greatest_a_no_integral_solution_l3640_364049

theorem greatest_a_no_integral_solution :
  (∀ a : ℤ, (∀ x : ℤ, ¬(|x + 1| < a - (3/2))) → a ≤ 1) ∧
  (∃ x : ℤ, |x + 1| < 2 - (3/2)) :=
sorry

end NUMINAMATH_CALUDE_greatest_a_no_integral_solution_l3640_364049


namespace NUMINAMATH_CALUDE_m_greater_than_n_l3640_364077

theorem m_greater_than_n (a : ℝ) : 2 * a * (a - 2) + 7 > (a - 2) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l3640_364077


namespace NUMINAMATH_CALUDE_probability_one_letter_each_name_l3640_364098

theorem probability_one_letter_each_name :
  let total_cards : ℕ := 12
  let alex_cards : ℕ := 6
  let jamie_cards : ℕ := 6
  let prob_one_each : ℚ := (alex_cards * jamie_cards) / (total_cards * (total_cards - 1))
  prob_one_each = 6 / 11 := by
sorry

end NUMINAMATH_CALUDE_probability_one_letter_each_name_l3640_364098


namespace NUMINAMATH_CALUDE_charlie_feather_collection_l3640_364062

/-- The number of sets of wings Charlie needs to make -/
def num_sets : ℕ := 2

/-- The number of feathers required for each set of wings -/
def feathers_per_set : ℕ := 900

/-- The number of feathers Charlie already has -/
def feathers_collected : ℕ := 387

/-- The total number of additional feathers Charlie needs to collect -/
def additional_feathers_needed : ℕ := num_sets * feathers_per_set - feathers_collected

theorem charlie_feather_collection :
  additional_feathers_needed = 1413 := by sorry

end NUMINAMATH_CALUDE_charlie_feather_collection_l3640_364062


namespace NUMINAMATH_CALUDE_total_birds_count_l3640_364044

/-- Proves that given the specified conditions, the total number of birds is 185 -/
theorem total_birds_count (chickens ducks : ℕ) 
  (h1 : ducks = 4 * chickens + 10) 
  (h2 : ducks = 150) : 
  chickens + ducks = 185 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_count_l3640_364044


namespace NUMINAMATH_CALUDE_power_of_ten_equation_l3640_364017

theorem power_of_ten_equation (x : ℕ) : (10^x) / (10^650) = 100000 ↔ x = 655 := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_equation_l3640_364017


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l3640_364075

/-- Given a set of values, proves that the initial mean before correcting an error
    is equal to the expected value. -/
theorem initial_mean_calculation (n : ℕ) (correct_value incorrect_value : ℝ) 
  (correct_mean : ℝ) (expected_initial_mean : ℝ) :
  n = 30 →
  correct_value = 165 →
  incorrect_value = 135 →
  correct_mean = 251 →
  expected_initial_mean = 250 →
  (n * correct_mean - (correct_value - incorrect_value)) / n = expected_initial_mean :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l3640_364075


namespace NUMINAMATH_CALUDE_store_transaction_loss_l3640_364084

def selling_price : ℝ := 60

theorem store_transaction_loss (cost_price_1 cost_price_2 : ℝ) 
  (h1 : (selling_price - cost_price_1) / cost_price_1 = 1/2)
  (h2 : (cost_price_2 - selling_price) / cost_price_2 = 1/2) :
  2 * selling_price - (cost_price_1 + cost_price_2) = -selling_price / 3 := by
  sorry

end NUMINAMATH_CALUDE_store_transaction_loss_l3640_364084


namespace NUMINAMATH_CALUDE_zero_point_existence_not_necessary_l3640_364036

def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

theorem zero_point_existence (a : ℝ) (h : a > 2) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 0, f a x = 0 :=
sorry

theorem not_necessary (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 0, f a x = 0) → a > 2 → False :=
sorry

end NUMINAMATH_CALUDE_zero_point_existence_not_necessary_l3640_364036


namespace NUMINAMATH_CALUDE_circle_properties_l3640_364030

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line equation for the center
def center_line (x y : ℝ) : Prop := y = -2 * x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

theorem circle_properties :
  -- The circle passes through point A(2, -1)
  circle_equation 2 (-1) ∧
  -- The center of the circle is on the line y = -2x
  ∃ (cx cy : ℝ), center_line cx cy ∧ 
    ∀ (x y : ℝ), circle_equation x y ↔ (x - cx)^2 + (y - cy)^2 = 2 ∧
  -- The circle is tangent to the line x + y - 1 = 0
  ∃ (tx ty : ℝ), tangent_line tx ty ∧
    circle_equation tx ty ∧
    ∀ (x y : ℝ), tangent_line x y → 
      ((x - tx)^2 + (y - ty)^2 < 2 ∨ (x = tx ∧ y = ty))
  := by sorry

end NUMINAMATH_CALUDE_circle_properties_l3640_364030


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3640_364088

theorem algebraic_expression_value (x y : ℝ) (h : 2 * x - y = 2) : 
  6 * x - 3 * y + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3640_364088


namespace NUMINAMATH_CALUDE_quadratic_sum_l3640_364018

/-- Given a quadratic equation x^2 - 10x + 15 = 0 that can be rewritten as (x + b)^2 = c,
    prove that b + c = 5 -/
theorem quadratic_sum (b c : ℝ) : 
  (∀ x, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3640_364018


namespace NUMINAMATH_CALUDE_parabola_point_coordinate_l3640_364038

/-- Proves that for a point on a parabola with specific distance to focus, its x-coordinate is 1/2 -/
theorem parabola_point_coordinate (x₀ y₀ : ℝ) : 
  y₀^2 = 2*x₀ →                                    -- Point (x₀, y₀) is on the parabola y² = 2x
  ((x₀ - 1/2)^2 + y₀^2) = (2*x₀)^2 →               -- Distance from (x₀, y₀) to focus (1/2, 0) is 2x₀
  x₀ = 1/2 := by
sorry


end NUMINAMATH_CALUDE_parabola_point_coordinate_l3640_364038


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_minus_i_l3640_364037

theorem imaginary_part_of_one_over_one_minus_i :
  let i : ℂ := Complex.I
  let z : ℂ := 1 / (1 - i)
  Complex.im z = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_minus_i_l3640_364037


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l3640_364066

/-- The number of dogwood trees currently in the park -/
def current_trees : ℕ := 39

/-- The number of trees to be planted today -/
def planted_today : ℕ := 41

/-- The number of trees to be planted tomorrow -/
def planted_tomorrow : ℕ := 20

/-- The total number of trees after planting -/
def total_trees : ℕ := 100

/-- Theorem stating that the current number of trees plus the trees to be planted
    equals the total number of trees after planting -/
theorem dogwood_tree_count : 
  current_trees + planted_today + planted_tomorrow = total_trees := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l3640_364066


namespace NUMINAMATH_CALUDE_linear_relationship_scaling_l3640_364090

/-- Given a linear relationship between x and y, this theorem proves that
    if an increase of 5 units in x results in an increase of 11 units in y,
    then an increase of 20 units in x will result in an increase of 44 units in y. -/
theorem linear_relationship_scaling (f : ℝ → ℝ) (h : ∀ x, f (x + 5) = f x + 11) :
  ∀ x, f (x + 20) = f x + 44 := by
  sorry

end NUMINAMATH_CALUDE_linear_relationship_scaling_l3640_364090


namespace NUMINAMATH_CALUDE_problem_solution_l3640_364031

theorem problem_solution :
  ∀ M : ℝ, (5 + 6 + 7) / 3 = (1988 + 1989 + 1990) / M → M = 994.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3640_364031


namespace NUMINAMATH_CALUDE_min_t_for_equations_l3640_364071

theorem min_t_for_equations (a b c d e : ℝ) 
  (h_non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0) 
  (h_sum_pos : a + b + c + d + e > 0) :
  (∃ t : ℝ, t = Real.sqrt 2 ∧ 
    a + c = t * b ∧ 
    b + d = t * c ∧ 
    c + e = t * d) ∧
  (∀ s : ℝ, (a + c = s * b ∧ b + d = s * c ∧ c + e = s * d) → s ≥ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_t_for_equations_l3640_364071


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3640_364048

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 ≥ 1) ↔ ∃ x₀ : ℝ, x₀^2 < 1 := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3640_364048


namespace NUMINAMATH_CALUDE_solution_set_rational_inequality_l3640_364072

theorem solution_set_rational_inequality :
  {x : ℝ | (x - 2) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_rational_inequality_l3640_364072


namespace NUMINAMATH_CALUDE_square_root_range_l3640_364080

theorem square_root_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_square_root_range_l3640_364080


namespace NUMINAMATH_CALUDE_train_cars_count_l3640_364000

/-- Represents a train with a consistent speed --/
structure Train where
  cars_per_12_seconds : ℕ
  total_passing_time : ℕ

/-- Calculates the total number of cars in the train --/
def total_cars (t : Train) : ℕ :=
  (t.cars_per_12_seconds * t.total_passing_time) / 12

/-- Theorem stating that a train with 8 cars passing in 12 seconds 
    and taking 210 seconds to pass has 140 cars --/
theorem train_cars_count :
  ∀ (t : Train), t.cars_per_12_seconds = 8 ∧ t.total_passing_time = 210 → 
  total_cars t = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l3640_364000


namespace NUMINAMATH_CALUDE_speed_of_A_is_correct_l3640_364042

/-- Represents the speed of boy A in mph -/
def speed_A : ℝ := 7.5

/-- Represents the speed of boy B in mph -/
def speed_B : ℝ := speed_A + 5

/-- Represents the speed of boy C in mph -/
def speed_C : ℝ := speed_A + 3

/-- Represents the total distance between Port Jervis and Poughkeepsie in miles -/
def total_distance : ℝ := 80

/-- Represents the distance from Poughkeepsie where A and B meet in miles -/
def meeting_distance : ℝ := 20

theorem speed_of_A_is_correct :
  speed_A * (total_distance / speed_B + meeting_distance / speed_B) =
  total_distance - meeting_distance := by sorry

end NUMINAMATH_CALUDE_speed_of_A_is_correct_l3640_364042


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l3640_364007

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 252) 
  (h2 : a + b + c = 22) : 
  a*b + b*c + c*a = 116 := by
sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l3640_364007


namespace NUMINAMATH_CALUDE_max_students_distribution_l3640_364083

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1340) (h2 : pencils = 1280) :
  (∃ (students : ℕ), students > 0 ∧ pens % students = 0 ∧ pencils % students = 0 ∧
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) ↔
  (∃ (max_students : ℕ), max_students = Nat.gcd pens pencils) :=
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3640_364083


namespace NUMINAMATH_CALUDE_existence_of_reduction_sequence_l3640_364052

/-- The game operation: either multiply by 2 or remove the unit digit -/
inductive GameOperation
| multiply_by_two
| remove_unit_digit

/-- Apply a single game operation to a natural number -/
def apply_operation (n : ℕ) (op : GameOperation) : ℕ :=
  match op with
  | GameOperation.multiply_by_two => 2 * n
  | GameOperation.remove_unit_digit => n / 10

/-- Predicate to check if a sequence of operations reduces a number to 1 -/
def reduces_to_one (start : ℕ) (ops : List GameOperation) : Prop :=
  start ≠ 0 ∧ List.foldl apply_operation start ops = 1

/-- Theorem: For any non-zero natural number, there exists a sequence of operations that reduces it to 1 -/
theorem existence_of_reduction_sequence (n : ℕ) : 
  n ≠ 0 → ∃ (ops : List GameOperation), reduces_to_one n ops := by
  sorry


end NUMINAMATH_CALUDE_existence_of_reduction_sequence_l3640_364052


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3640_364002

theorem absolute_value_inequality (x : ℝ) : |2*x - 1| < 3 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3640_364002


namespace NUMINAMATH_CALUDE_tournament_outcomes_l3640_364070

/-- The number of teams in the tournament -/
def num_teams : ℕ := 5

/-- The number of games each team plays -/
def games_per_team : ℕ := num_teams - 1

/-- The total number of games in the tournament -/
def total_games : ℕ := (num_teams * games_per_team) / 2

/-- The number of ways the games can occur with the given conditions -/
def valid_outcomes : ℕ := 2^total_games - 2 * num_teams * 2^(games_per_team * (num_teams - 2) / 2) + num_teams * (num_teams - 1) * 2^((num_teams - 2) * (num_teams - 3) / 2)

theorem tournament_outcomes :
  valid_outcomes = 544 :=
sorry

end NUMINAMATH_CALUDE_tournament_outcomes_l3640_364070


namespace NUMINAMATH_CALUDE_fraction_equality_implies_relationship_l3640_364009

theorem fraction_equality_implies_relationship (a b c d : ℝ) :
  (a + b + 1) / (b + c + 2) = (c + d + 1) / (d + a + 2) →
  (a - c) * (a + b + c + d + 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_relationship_l3640_364009


namespace NUMINAMATH_CALUDE_tricycle_count_l3640_364096

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) :
  total_children = 10 →
  total_wheels = 26 →
  ∃ (walking bicycles tricycles : ℕ),
    walking + bicycles + tricycles = total_children ∧
    2 * bicycles + 3 * tricycles = total_wheels ∧
    tricycles = 6 :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l3640_364096


namespace NUMINAMATH_CALUDE_total_accessories_is_712_l3640_364024

/-- Calculates the total number of accessories used by Jane and Emily for their dresses -/
def total_accessories : ℕ :=
  let jane_dresses := 4 * 10
  let emily_dresses := 3 * 8
  let jane_accessories_per_dress := 3 + 2 + 1 + 4
  let emily_accessories_per_dress := 2 + 3 + 2 + 5 + 1
  jane_dresses * jane_accessories_per_dress + emily_dresses * emily_accessories_per_dress

/-- Theorem stating that the total number of accessories is 712 -/
theorem total_accessories_is_712 : total_accessories = 712 := by
  sorry

end NUMINAMATH_CALUDE_total_accessories_is_712_l3640_364024


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3640_364019

theorem smaller_number_proof (L S : ℕ) (h1 : L - S = 2395) (h2 : L = 6 * S + 15) : S = 476 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3640_364019


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l3640_364064

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

-- Define the theorem
theorem geometric_sequence_a8 (a₁ : ℝ) (q : ℝ) :
  (a₁ * (a₁ * q^2) = 4) →
  (a₁ * q^8 = 256) →
  (geometric_sequence a₁ q 8 = 128 ∨ geometric_sequence a₁ q 8 = -128) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_a8_l3640_364064


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3640_364081

theorem regular_polygon_exterior_angle (n : ℕ) : 
  (n > 2) → (360 / n = 72) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3640_364081


namespace NUMINAMATH_CALUDE_four_m_squared_minus_n_squared_l3640_364008

theorem four_m_squared_minus_n_squared (m n : ℝ) 
  (h1 : 2*m + n = 3) (h2 : 2*m - n = 1) : 4*m^2 - n^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_m_squared_minus_n_squared_l3640_364008


namespace NUMINAMATH_CALUDE_rational_fraction_equality_l3640_364015

theorem rational_fraction_equality (a b : ℚ) 
  (h1 : (a + 2*b) / (2*a - b) = 2)
  (h2 : 3*a - 2*b ≠ 0) :
  (3*a + 2*b) / (3*a - 2*b) = 3 := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_equality_l3640_364015


namespace NUMINAMATH_CALUDE_problem_statement_l3640_364063

theorem problem_statement : (1 / ((-2^4)^2)) * ((-2)^7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3640_364063


namespace NUMINAMATH_CALUDE_dice_probability_l3640_364032

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The number of dice that should show numbers less than 5 -/
def num_less_than_five : ℕ := 4

/-- The probability of rolling a number less than 5 on a single die -/
def prob_less_than_five : ℚ := 1 / 2

/-- The probability of rolling a number 5 or greater on a single die -/
def prob_five_or_greater : ℚ := 1 - prob_less_than_five

theorem dice_probability :
  (Nat.choose num_dice num_less_than_five : ℚ) *
  (prob_less_than_five ^ num_less_than_five) *
  (prob_five_or_greater ^ (num_dice - num_less_than_five)) =
  35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l3640_364032


namespace NUMINAMATH_CALUDE_valid_sets_l3640_364079

theorem valid_sets (A : Set ℕ) : 
  (∀ m n : ℕ, m + n ∈ A → m * n ∈ A) ↔ 
  A = ∅ ∨ A = {0} ∨ A = {0, 1} ∨ A = {0, 1, 2} ∨ 
  A = {0, 1, 2, 3} ∨ A = {0, 1, 2, 3, 4} ∨ A = Set.univ :=
sorry

end NUMINAMATH_CALUDE_valid_sets_l3640_364079


namespace NUMINAMATH_CALUDE_larger_number_l3640_364086

theorem larger_number (P Q : ℝ) (h1 : P = Real.sqrt 2) (h2 : Q = Real.sqrt 6 - Real.sqrt 2) : P > Q := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l3640_364086


namespace NUMINAMATH_CALUDE_shoes_to_sell_l3640_364050

def monthly_goal : ℕ := 80
def sold_last_week : ℕ := 27
def sold_this_week : ℕ := 12

theorem shoes_to_sell : monthly_goal - (sold_last_week + sold_this_week) = 41 := by
  sorry

end NUMINAMATH_CALUDE_shoes_to_sell_l3640_364050


namespace NUMINAMATH_CALUDE_connor_test_scores_l3640_364053

theorem connor_test_scores (test1 test2 test3 test4 : ℕ) : 
  test1 = 82 →
  test2 = 75 →
  test1 ≤ 100 ∧ test2 ≤ 100 ∧ test3 ≤ 100 ∧ test4 ≤ 100 →
  (test1 + test2 + test3 + test4) / 4 = 85 →
  (test3 = 83 ∧ test4 = 100) ∨ (test3 = 100 ∧ test4 = 83) :=
by sorry

end NUMINAMATH_CALUDE_connor_test_scores_l3640_364053


namespace NUMINAMATH_CALUDE_solve_baseball_cards_problem_l3640_364060

/-- The number of cards Brandon, Malcom, and Ella have, and the combined remaining cards after transactions -/
def baseball_cards_problem (brandon_cards : ℕ) (malcom_extra : ℕ) (ella_less : ℕ) 
  (malcom_fraction : ℚ) (ella_fraction : ℚ) : Prop :=
  let malcom_cards := brandon_cards + malcom_extra
  let ella_cards := malcom_cards - ella_less
  let malcom_remaining := malcom_cards - Int.floor (malcom_fraction * malcom_cards)
  let ella_remaining := ella_cards - Int.floor (ella_fraction * ella_cards)
  malcom_remaining + ella_remaining = 32

/-- Theorem statement for the baseball cards problem -/
theorem solve_baseball_cards_problem : 
  baseball_cards_problem 20 12 5 (2/3) (1/4) := by sorry

end NUMINAMATH_CALUDE_solve_baseball_cards_problem_l3640_364060


namespace NUMINAMATH_CALUDE_equation_solutions_inequality_solutions_l3640_364068

/-- Part a: Solutions for 1/x + 1/y + 1/z = 1 where x, y, z are natural numbers -/
def solutions_a : Set (ℕ × ℕ × ℕ) :=
  {(3, 3, 3), (6, 3, 2), (4, 4, 2)}

/-- Part b: Solutions for 1/x + 1/y + 1/z > 1 where x, y, z are natural numbers greater than 1 -/
def solutions_b : Set (ℕ × ℕ × ℕ) :=
  {(x, 2, 2) | x > 1} ∪ {(3, 3, 2), (4, 3, 2), (5, 3, 2)}

theorem equation_solutions (x y z : ℕ) :
  (1 / x + 1 / y + 1 / z = 1) ↔ (x, y, z) ∈ solutions_a := by
  sorry

theorem inequality_solutions (x y z : ℕ) :
  (x > 1 ∧ y > 1 ∧ z > 1 ∧ 1 / x + 1 / y + 1 / z > 1) ↔ (x, y, z) ∈ solutions_b := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_inequality_solutions_l3640_364068


namespace NUMINAMATH_CALUDE_three_inequalities_l3640_364095

theorem three_inequalities (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x + y) * (y + z) * (z + x) ≥ 8 * x * y * z) ∧
  (x^2 + y^2 + z^2 ≥ x*y + y*z + z*x) ∧
  (x^x * y^y * z^z ≥ (x*y*z)^((x+y+z)/3)) := by
  sorry

end NUMINAMATH_CALUDE_three_inequalities_l3640_364095


namespace NUMINAMATH_CALUDE_sum_m_n_range_l3640_364039

/-- A quadratic function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- The theorem stating that given the conditions, m + n is in [-4, 0] -/
theorem sum_m_n_range (m n : ℝ) (h1 : m ≤ n) (h2 : ∀ x ∈ Set.Icc m n, -1 ≤ f x ∧ f x ≤ 3) :
  -4 ≤ m + n ∧ m + n ≤ 0 := by sorry

end NUMINAMATH_CALUDE_sum_m_n_range_l3640_364039


namespace NUMINAMATH_CALUDE_farm_ratio_change_l3640_364076

theorem farm_ratio_change (H C : ℕ) : 
  H = 6 * C →  -- Initial ratio of horses to cows is 6:1
  H - 15 = (C + 15) + 70 →  -- After transaction, 70 more horses than cows
  (H - 15) / (C + 15) = 3  -- New ratio of horses to cows is 3:1
  := by sorry

end NUMINAMATH_CALUDE_farm_ratio_change_l3640_364076


namespace NUMINAMATH_CALUDE_worker_daily_rate_l3640_364013

/-- Proves that a worker's daily rate is $150 given the specified conditions -/
theorem worker_daily_rate (daily_rate : ℝ) (overtime_rate : ℝ) (total_days : ℝ) 
  (overtime_hours : ℝ) (total_pay : ℝ) : 
  overtime_rate = 5 →
  total_days = 5 →
  overtime_hours = 4 →
  total_pay = 770 →
  total_pay = daily_rate * total_days + overtime_rate * overtime_hours →
  daily_rate = 150 := by
  sorry

end NUMINAMATH_CALUDE_worker_daily_rate_l3640_364013


namespace NUMINAMATH_CALUDE_smallest_divisible_by_3_and_4_l3640_364027

theorem smallest_divisible_by_3_and_4 : 
  ∀ n : ℕ, n > 0 ∧ 3 ∣ n ∧ 4 ∣ n → n ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_3_and_4_l3640_364027


namespace NUMINAMATH_CALUDE_sqrt_3_minus_2_squared_l3640_364006

theorem sqrt_3_minus_2_squared : (Real.sqrt 3 - 2) * (Real.sqrt 3 - 2) = 7 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_2_squared_l3640_364006


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l3640_364029

theorem peanut_butter_servings 
  (jar_content : ℚ) 
  (serving_size : ℚ) 
  (h1 : jar_content = 35 + 4/5)
  (h2 : serving_size = 5/2) : 
  jar_content / serving_size = 14 + 8/25 := by
sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l3640_364029


namespace NUMINAMATH_CALUDE_eunji_lives_higher_l3640_364074

def yoojung_floor : ℕ := 17
def eunji_floor : ℕ := 25

theorem eunji_lives_higher : eunji_floor > yoojung_floor := by
  sorry

end NUMINAMATH_CALUDE_eunji_lives_higher_l3640_364074


namespace NUMINAMATH_CALUDE_intersection_M_N_l3640_364087

-- Define set M
def M : Set ℝ := {x | x^2 - x ≤ 0}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ico 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3640_364087


namespace NUMINAMATH_CALUDE_coin_value_equality_l3640_364004

/-- The value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- The theorem stating the equality of coin values -/
theorem coin_value_equality (n : ℕ) : 
  15 * coin_value "quarter" + 20 * coin_value "dime" = 
  10 * coin_value "quarter" + n * coin_value "dime" + 5 * coin_value "nickel" → 
  n = 30 := by
  sorry

#check coin_value_equality

end NUMINAMATH_CALUDE_coin_value_equality_l3640_364004


namespace NUMINAMATH_CALUDE_cone_height_from_sphere_waste_l3640_364093

/-- Given a sphere and a cone carved from it, prove the height of the cone when 75% of wood is wasted -/
theorem cone_height_from_sphere_waste (r : ℝ) (h : ℝ) : 
  r = 9 →  -- sphere radius
  (4/3) * Real.pi * r^3 * (1 - 0.75) = (1/3) * Real.pi * r^2 * h → -- 75% wood wasted
  h = 27 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_sphere_waste_l3640_364093


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3640_364001

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

/-- The 150th term of the specific arithmetic sequence -/
def term_150 : ℝ :=
  arithmetic_sequence 3 5 150

theorem arithmetic_sequence_150th_term :
  term_150 = 748 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3640_364001


namespace NUMINAMATH_CALUDE_two_solutions_l3640_364047

/-- The number of positive integers satisfying the equation -/
def solution_count : ℕ := 2

/-- Predicate for integers satisfying the equation -/
def satisfies_equation (n : ℕ) : Prop :=
  (n + 800) / 80 = ⌊Real.sqrt n⌋

/-- Theorem stating that exactly two positive integers satisfy the equation -/
theorem two_solutions :
  (∃ (a b : ℕ), a ≠ b ∧ satisfies_equation a ∧ satisfies_equation b) ∧
  (∀ (n : ℕ), satisfies_equation n → n = a ∨ n = b) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l3640_364047


namespace NUMINAMATH_CALUDE_john_total_weight_l3640_364011

/-- The total weight moved by John during his workout -/
def total_weight_moved (weight_per_rep : ℕ) (reps_per_set : ℕ) (num_sets : ℕ) : ℕ :=
  weight_per_rep * reps_per_set * num_sets

/-- Theorem stating that John moves 450 pounds in total -/
theorem john_total_weight :
  total_weight_moved 15 10 3 = 450 := by
  sorry

end NUMINAMATH_CALUDE_john_total_weight_l3640_364011


namespace NUMINAMATH_CALUDE_absolute_value_plus_pi_minus_two_to_zero_l3640_364067

theorem absolute_value_plus_pi_minus_two_to_zero :
  |(-3 : ℝ)| + (π - 2)^(0 : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_plus_pi_minus_two_to_zero_l3640_364067


namespace NUMINAMATH_CALUDE_abc_sum_range_l3640_364020

theorem abc_sum_range (a b c : ℝ) (h : a + b + 2*c = 0) :
  ∃ y : ℝ, y ≤ 0 ∧ a*b + a*c + b*c = y ∧
  ∀ z : ℝ, z ≤ 0 → ∃ a' b' c' : ℝ, a' + b' + 2*c' = 0 ∧ a'*b' + a'*c' + b'*c' = z :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_range_l3640_364020


namespace NUMINAMATH_CALUDE_circle_area_decrease_l3640_364059

/-- Given three circles with radii r1, r2, and r3, prove that the decrease in their combined area
    when each radius is reduced by 50% is equal to 75% of their original combined area. -/
theorem circle_area_decrease (r1 r2 r3 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hr3 : r3 > 0) :
  let original_area := π * (r1^2 + r2^2 + r3^2)
  let new_area := π * ((r1/2)^2 + (r2/2)^2 + (r3/2)^2)
  original_area - new_area = (3/4) * original_area :=
by sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l3640_364059


namespace NUMINAMATH_CALUDE_population_increase_rate_l3640_364021

theorem population_increase_rate 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (increase_rate : ℚ) :
  initial_population = 2000 →
  final_population = 2400 →
  increase_rate = (final_population - initial_population) / initial_population * 100 →
  increase_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_l3640_364021


namespace NUMINAMATH_CALUDE_church_attendance_l3640_364085

/-- The total number of people in the church is the sum of children, male adults, and female adults. -/
theorem church_attendance (children : ℕ) (male_adults : ℕ) (female_adults : ℕ) 
  (h1 : children = 80) 
  (h2 : male_adults = 60) 
  (h3 : female_adults = 60) : 
  children + male_adults + female_adults = 200 := by
  sorry

#check church_attendance

end NUMINAMATH_CALUDE_church_attendance_l3640_364085


namespace NUMINAMATH_CALUDE_range_of_a_l3640_364078

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + a| > 2) ↔ (a < -1 ∨ a > 3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3640_364078


namespace NUMINAMATH_CALUDE_elizabeth_pen_purchase_l3640_364046

/-- Calculates the number of pens Elizabeth can buy given her budget and pencil purchase. -/
theorem elizabeth_pen_purchase 
  (total_budget : ℚ)
  (pencil_cost : ℚ)
  (pen_cost : ℚ)
  (pencil_count : ℕ)
  (h1 : total_budget = 20)
  (h2 : pencil_cost = 8/5)  -- $1.60 expressed as a rational number
  (h3 : pen_cost = 2)
  (h4 : pencil_count = 5) :
  (total_budget - pencil_cost * ↑pencil_count) / pen_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_pen_purchase_l3640_364046


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l3640_364034

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^13 + i^18 + i^23 + i^28 + i^33 = i := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l3640_364034


namespace NUMINAMATH_CALUDE_sine_sum_acute_triangle_l3640_364054

theorem sine_sum_acute_triangle (α β γ : Real) 
  (acute_triangle : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π)
  (acute_angles : α < π/2 ∧ β < π/2 ∧ γ < π/2) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end NUMINAMATH_CALUDE_sine_sum_acute_triangle_l3640_364054


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3640_364014

theorem absolute_value_inequality (x : ℝ) :
  |x + 3| > 1 ↔ x < -4 ∨ x > -2 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3640_364014


namespace NUMINAMATH_CALUDE_pictures_per_album_l3640_364045

theorem pictures_per_album 
  (total_pictures : ℕ) 
  (phone_pictures camera_pictures : ℕ) 
  (num_albums : ℕ) 
  (h1 : total_pictures = phone_pictures + camera_pictures)
  (h2 : phone_pictures = 5)
  (h3 : camera_pictures = 35)
  (h4 : num_albums = 8)
  (h5 : total_pictures % num_albums = 0) :
  total_pictures / num_albums = 5 := by
sorry

end NUMINAMATH_CALUDE_pictures_per_album_l3640_364045


namespace NUMINAMATH_CALUDE_cashier_money_value_l3640_364026

def total_bills : ℕ := 30
def ten_dollar_bills : ℕ := 27
def twenty_dollar_bills : ℕ := 3
def ten_dollar_value : ℕ := 10
def twenty_dollar_value : ℕ := 20

theorem cashier_money_value :
  ten_dollar_bills + twenty_dollar_bills = total_bills →
  ten_dollar_bills * ten_dollar_value + twenty_dollar_bills * twenty_dollar_value = 330 :=
by
  sorry

end NUMINAMATH_CALUDE_cashier_money_value_l3640_364026


namespace NUMINAMATH_CALUDE_abc_is_zero_l3640_364058

theorem abc_is_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) :
  a * b * c = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_is_zero_l3640_364058


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3640_364023

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3/5) 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3640_364023


namespace NUMINAMATH_CALUDE_inequality_solution_l3640_364033

theorem inequality_solution (x : ℝ) (h : x ≠ 1) :
  1 / (x - 1) ≤ 1 ↔ x < 1 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3640_364033


namespace NUMINAMATH_CALUDE_total_enjoyable_gameplay_l3640_364028

/-- Calculates the total enjoyable gameplay time given the conditions of the game, expansion, and mod. -/
theorem total_enjoyable_gameplay 
  (original_game_hours : ℝ)
  (original_game_boring_percent : ℝ)
  (expansion_hours : ℝ)
  (expansion_load_screen_percent : ℝ)
  (expansion_inventory_percent : ℝ)
  (mod_skip_percent : ℝ)
  (h1 : original_game_hours = 150)
  (h2 : original_game_boring_percent = 0.7)
  (h3 : expansion_hours = 50)
  (h4 : expansion_load_screen_percent = 0.25)
  (h5 : expansion_inventory_percent = 0.25)
  (h6 : mod_skip_percent = 0.15) :
  let original_enjoyable := original_game_hours * (1 - original_game_boring_percent)
  let expansion_enjoyable := expansion_hours * (1 - expansion_load_screen_percent) * (1 - expansion_inventory_percent)
  let total_tedious := original_game_hours * original_game_boring_percent + 
                       expansion_hours * (expansion_load_screen_percent + (1 - expansion_load_screen_percent) * expansion_inventory_percent)
  let mod_skipped := total_tedious * mod_skip_percent
  original_enjoyable + expansion_enjoyable + mod_skipped = 92.15625 := by
  sorry


end NUMINAMATH_CALUDE_total_enjoyable_gameplay_l3640_364028


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3640_364092

theorem modulus_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.abs ((1 + i) / i) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3640_364092


namespace NUMINAMATH_CALUDE_quadratic_solutions_parabola_vertex_l3640_364073

-- Part 1: Quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + 4*x - 2 = 0

theorem quadratic_solutions :
  ∃ x1 x2 : ℝ, x1 = -2 + Real.sqrt 6 ∧ x2 = -2 - Real.sqrt 6 ∧
  quadratic_equation x1 ∧ quadratic_equation x2 :=
sorry

-- Part 2: Parabola vertex
def parabola (x y : ℝ) : Prop :=
  y = 2*x^2 - 4*x + 6

theorem parabola_vertex :
  ∃ x y : ℝ, x = 1 ∧ y = 4 ∧ parabola x y ∧
  ∀ x' y' : ℝ, parabola x' y' → y ≤ y' :=
sorry

end NUMINAMATH_CALUDE_quadratic_solutions_parabola_vertex_l3640_364073


namespace NUMINAMATH_CALUDE_coin_denomination_problem_l3640_364022

theorem coin_denomination_problem (total_coins : ℕ) (unknown_coins : ℕ) (known_coins : ℕ) 
  (known_coin_value : ℕ) (total_value : ℕ) (x : ℕ) :
  total_coins = 324 →
  unknown_coins = 220 →
  known_coins = total_coins - unknown_coins →
  known_coin_value = 25 →
  total_value = 7000 →
  unknown_coins * x + known_coins * known_coin_value = total_value →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_coin_denomination_problem_l3640_364022


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l3640_364061

theorem lcm_factor_problem (A B : ℕ+) (Y : ℕ+) : 
  Nat.gcd A B = 23 →
  A = 391 →
  Nat.lcm A B = 23 * 13 * Y →
  Y = 17 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l3640_364061


namespace NUMINAMATH_CALUDE_A_minus_2B_general_A_minus_2B_specific_l3640_364051

-- Define the algebraic expressions A and B
def A (x y : ℝ) : ℝ := 3 * x^2 - 5 * x * y - 2 * y^2
def B (x y : ℝ) : ℝ := x^2 - 3 * y

-- Theorem for part 1
theorem A_minus_2B_general (x y : ℝ) : 
  A x y - 2 * B x y = x^2 - 5 * x * y - 2 * y^2 + 6 * y := by sorry

-- Theorem for part 2
theorem A_minus_2B_specific : 
  A 2 (-1) - 2 * B 2 (-1) = 6 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_general_A_minus_2B_specific_l3640_364051


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l3640_364025

def sequence_term (n : ℕ+) : ℚ := 1 / (n * (n + 1))

def sum_of_terms (n : ℕ+) : ℚ := n / (n + 1)

theorem sequence_sum_theorem (n : ℕ+) :
  (∀ k : ℕ+, k ≤ n → sequence_term k = 1 / (k * (k + 1))) →
  sum_of_terms n = 10 / 11 →
  n = 10 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l3640_364025


namespace NUMINAMATH_CALUDE_solution_transformation_l3640_364016

theorem solution_transformation (k x y : ℤ) 
  (h1 : ∃ n : ℤ, 15 * n = k)
  (h2 : x^2 - 2*y^2 = k) :
  ∃ t u : ℤ, t^2 - 2*u^2 = -k ∧ 
  ((t = x + 2*y ∧ u = x + y) ∨ (t = x - 2*y ∧ u = x - y)) := by
  sorry

end NUMINAMATH_CALUDE_solution_transformation_l3640_364016


namespace NUMINAMATH_CALUDE_problem_1_l3640_364035

theorem problem_1 : 40 + (1/6 - 2/3 + 3/4) * 12 = 43 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3640_364035


namespace NUMINAMATH_CALUDE_function_equation_solution_l3640_364069

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x * y * f y) : 
  ∀ x : ℝ, f x = 0 ∨ f x = x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3640_364069


namespace NUMINAMATH_CALUDE_partnership_profit_l3640_364094

theorem partnership_profit (J M : ℕ) (P : ℚ) : 
  J = 700 →
  M = 300 →
  (P / 6 + (J * 2 * P) / (3 * (J + M))) - (P / 6 + (M * 2 * P) / (3 * (J + M))) = 800 →
  P = 3000 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_l3640_364094


namespace NUMINAMATH_CALUDE_reading_pattern_l3640_364091

theorem reading_pattern (x y : ℝ) : 
  (∀ (days_xiaoming days_xiaoying : ℕ), 
    days_xiaoming = 3 ∧ days_xiaoying = 5 → 
    days_xiaoming * x + 6 = days_xiaoying * y) ∧
  (y = x - 10) →
  3 * x = 5 * y - 6 ∧ y = 2 * x - 10 := by
sorry

end NUMINAMATH_CALUDE_reading_pattern_l3640_364091


namespace NUMINAMATH_CALUDE_socks_expense_is_eleven_l3640_364055

/-- The amount spent on socks given a budget and other expenses --/
def socks_expense (budget : ℕ) (shirt_cost pants_cost coat_cost belt_cost shoes_cost amount_left : ℕ) : ℕ :=
  budget - (shirt_cost + pants_cost + coat_cost + belt_cost + shoes_cost + amount_left)

/-- Theorem: Given the specific budget and expenses, the amount spent on socks is $11 --/
theorem socks_expense_is_eleven :
  socks_expense 200 30 46 38 18 41 16 = 11 := by
  sorry

end NUMINAMATH_CALUDE_socks_expense_is_eleven_l3640_364055


namespace NUMINAMATH_CALUDE_computer_knowledge_competition_compositions_l3640_364003

theorem computer_knowledge_competition_compositions :
  let n : ℕ := 8  -- number of people in each group
  let k : ℕ := 4  -- number of people to be selected from each group
  Nat.choose n k * Nat.choose n k = 4900 := by
  sorry

end NUMINAMATH_CALUDE_computer_knowledge_competition_compositions_l3640_364003


namespace NUMINAMATH_CALUDE_system_solution_l3640_364097

theorem system_solution :
  ∃ (m n : ℚ), m / 3 + n / 2 = 1 ∧ m - 2 * n = 2 ∧ m = 18 / 7 ∧ n = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3640_364097


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3640_364010

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (2, y)
  parallel a b → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3640_364010


namespace NUMINAMATH_CALUDE_problem_cube_surface_area_l3640_364099

/-- Represents a cube structure -/
structure CubeStructure where
  size : ℕ
  smallCubeSize : ℕ
  removedCubes : ℕ

/-- Calculate the surface area of the cube structure -/
def surfaceArea (c : CubeStructure) : ℕ :=
  sorry

/-- The specific cube structure in the problem -/
def problemCube : CubeStructure :=
  { size := 8
  , smallCubeSize := 2
  , removedCubes := 4 }

/-- Theorem stating that the surface area of the problem cube is 1632 -/
theorem problem_cube_surface_area :
  surfaceArea problemCube = 1632 :=
sorry

end NUMINAMATH_CALUDE_problem_cube_surface_area_l3640_364099


namespace NUMINAMATH_CALUDE_equation_solution_set_l3640_364005

-- Define the equation
def equation (x : ℝ) : Prop := Real.log (Real.sqrt 3 * Real.sin x) = Real.log (-Real.cos x)

-- Define the solution set
def solution_set : Set ℝ := {x | ∃ k : ℤ, x = 2 * k * Real.pi + 5 * Real.pi / 6}

-- Theorem statement
theorem equation_solution_set : {x : ℝ | equation x} = solution_set := by sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3640_364005


namespace NUMINAMATH_CALUDE_slope_angle_of_negative_sqrt3_over_3_l3640_364041

/-- The slope angle of a line with slope -√3/3 is 5π/6 -/
theorem slope_angle_of_negative_sqrt3_over_3 :
  let slope : ℝ := -Real.sqrt 3 / 3
  let slope_angle : ℝ := Real.arctan slope
  slope_angle = 5 * Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_negative_sqrt3_over_3_l3640_364041


namespace NUMINAMATH_CALUDE_intersection_point_integer_coordinates_l3640_364043

theorem intersection_point_integer_coordinates (m : ℕ+) : 
  (∃ x y : ℤ, 17 * x + 7 * y = 1000 ∧ y = m * x + 2) ↔ m = 68 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_integer_coordinates_l3640_364043


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3640_364040

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x > x) → a ∈ Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3640_364040


namespace NUMINAMATH_CALUDE_equation_equivalence_l3640_364012

theorem equation_equivalence (x : ℝ) : 
  (1 / 2 - (x - 1) / 3 = 1) ↔ (3 - 2 * (x - 1) = 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3640_364012
