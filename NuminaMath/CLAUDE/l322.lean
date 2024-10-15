import Mathlib

namespace NUMINAMATH_CALUDE_parabola_with_directrix_neg_one_l322_32230

/-- A parabola is defined by its directrix and focus. This structure represents a parabola with a vertical directrix. -/
structure Parabola where
  /-- The x-coordinate of the directrix -/
  directrix : ℝ

/-- The standard equation of a parabola with a vertical directrix -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (y^2 = 4*(x - p.directrix/2))

/-- Theorem: For a parabola with directrix x = -1, its standard equation is y^2 = 4x -/
theorem parabola_with_directrix_neg_one (p : Parabola) (h : p.directrix = -1) :
  standardEquation p ↔ ∀ x y : ℝ, (y^2 = 4*x) :=
sorry

end NUMINAMATH_CALUDE_parabola_with_directrix_neg_one_l322_32230


namespace NUMINAMATH_CALUDE_oak_trees_planted_l322_32245

/-- The number of oak trees planted by workers in a park. -/
def trees_planted (initial_trees final_trees : ℕ) : ℕ :=
  final_trees - initial_trees

/-- Theorem: Given 5 initial oak trees and 9 final oak trees, the number of trees planted is 4. -/
theorem oak_trees_planted :
  let initial_trees : ℕ := 5
  let final_trees : ℕ := 9
  trees_planted initial_trees final_trees = 4 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_planted_l322_32245


namespace NUMINAMATH_CALUDE_second_platform_length_l322_32292

/-- Given a train and two platforms, calculate the length of the second platform. -/
theorem second_platform_length
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (first_crossing_time : ℝ)
  (second_crossing_time : ℝ)
  (h1 : train_length = 30)
  (h2 : first_platform_length = 90)
  (h3 : first_crossing_time = 12)
  (h4 : second_crossing_time = 15)
  (h5 : train_length > 0)
  (h6 : first_platform_length > 0)
  (h7 : first_crossing_time > 0)
  (h8 : second_crossing_time > 0) :
  let speed := (train_length + first_platform_length) / first_crossing_time
  let second_platform_length := speed * second_crossing_time - train_length
  second_platform_length = 120 := by
sorry


end NUMINAMATH_CALUDE_second_platform_length_l322_32292


namespace NUMINAMATH_CALUDE_exists_divisor_friendly_bijection_l322_32237

/-- The number of positive divisors of a positive integer n -/
def d (n : ℕ+) : ℕ := sorry

/-- A bijection is divisor-friendly if it satisfies the given property -/
def divisor_friendly (F : ℕ+ → ℕ+) : Prop :=
  Function.Bijective F ∧ ∀ m n : ℕ+, d (F (m * n)) = d (F m) * d (F n)

/-- There exists a divisor-friendly bijection -/
theorem exists_divisor_friendly_bijection : ∃ F : ℕ+ → ℕ+, divisor_friendly F := by
  sorry

end NUMINAMATH_CALUDE_exists_divisor_friendly_bijection_l322_32237


namespace NUMINAMATH_CALUDE_average_equals_median_l322_32295

theorem average_equals_median (n : ℕ) (k : ℕ) (x : ℝ) : 
  n > 0 → 
  k > 0 → 
  x > 0 → 
  n = 14 → 
  (x * (k + 1) / 2)^2 = (2 * n)^2 → 
  x = n := by
sorry

end NUMINAMATH_CALUDE_average_equals_median_l322_32295


namespace NUMINAMATH_CALUDE_sally_boxes_proof_l322_32200

/-- The number of boxes Sally sold on Saturday -/
def saturday_boxes : ℕ := 60

/-- The number of boxes Sally sold on Sunday -/
def sunday_boxes : ℕ := (3 * saturday_boxes) / 2

/-- The total number of boxes Sally sold over two days -/
def total_boxes : ℕ := 150

theorem sally_boxes_proof :
  saturday_boxes + sunday_boxes = total_boxes ∧
  sunday_boxes = (3 * saturday_boxes) / 2 :=
sorry

end NUMINAMATH_CALUDE_sally_boxes_proof_l322_32200


namespace NUMINAMATH_CALUDE_system_solution_l322_32209

/-- Prove that the solution to the system of equations:
    4x - 3y = -10
    6x + 5y = -13
    is (-89/38, 0.21053) -/
theorem system_solution : 
  ∃ (x y : ℝ), 
    (4 * x - 3 * y = -10) ∧ 
    (6 * x + 5 * y = -13) ∧ 
    (x = -89 / 38) ∧ 
    (y = 0.21053) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l322_32209


namespace NUMINAMATH_CALUDE_min_probability_cards_l322_32257

/-- Represents the probability of a card being red-side up after two flips -/
def probability_red (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2 : ℚ) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2 : ℚ) / 676

/-- The total number of cards -/
def total_cards : ℕ := 50

/-- The number of cards flipped in each operation -/
def flip_size : ℕ := 25

/-- Theorem stating that cards 13 and 38 have the lowest probability of being red-side up -/
theorem min_probability_cards :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ total_cards →
    (probability_red 13 ≤ probability_red k ∧
     probability_red 38 ≤ probability_red k) :=
by sorry

end NUMINAMATH_CALUDE_min_probability_cards_l322_32257


namespace NUMINAMATH_CALUDE_absolute_value_equality_l322_32281

theorem absolute_value_equality (x : ℝ) (h : x > 3) : 
  |x - Real.sqrt ((x - 3)^2)| = 3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l322_32281


namespace NUMINAMATH_CALUDE_tile_covers_25_squares_l322_32212

/-- Represents a square tile -/
structure Tile :=
  (sideLength : ℝ)

/-- Represents a checkerboard -/
structure Checkerboard :=
  (size : ℕ)
  (squareWidth : ℝ)

/-- Counts the number of squares completely covered by a tile on a checkerboard -/
def countCoveredSquares (t : Tile) (c : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating that a square tile with side length D placed on a 10x10 checkerboard
    with square width D, such that their centers coincide, covers exactly 25 squares -/
theorem tile_covers_25_squares (D : ℝ) (D_pos : D > 0) :
  let t : Tile := { sideLength := D }
  let c : Checkerboard := { size := 10, squareWidth := D }
  countCoveredSquares t c = 25 :=
sorry

end NUMINAMATH_CALUDE_tile_covers_25_squares_l322_32212


namespace NUMINAMATH_CALUDE_log_base_a1_13_l322_32255

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem log_base_a1_13 (a : ℕ → ℝ) :
  geometric_sequence a → a 9 = 13 → a 13 = 1 → Real.log 13 / Real.log (a 1) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_log_base_a1_13_l322_32255


namespace NUMINAMATH_CALUDE_defective_film_probability_l322_32282

/-- The probability of selecting a defective X-ray film from a warehouse with
    specified conditions. -/
theorem defective_film_probability :
  let total_boxes : ℕ := 10
  let boxes_a : ℕ := 5
  let boxes_b : ℕ := 3
  let boxes_c : ℕ := 2
  let defective_rate_a : ℚ := 1 / 10
  let defective_rate_b : ℚ := 1 / 15
  let defective_rate_c : ℚ := 1 / 20
  let prob_a : ℚ := boxes_a / total_boxes
  let prob_b : ℚ := boxes_b / total_boxes
  let prob_c : ℚ := boxes_c / total_boxes
  let total_prob : ℚ := prob_a * defective_rate_a + prob_b * defective_rate_b + prob_c * defective_rate_c
  total_prob = 8 / 100 :=
by sorry

end NUMINAMATH_CALUDE_defective_film_probability_l322_32282


namespace NUMINAMATH_CALUDE_extreme_values_sum_reciprocals_l322_32271

theorem extreme_values_sum_reciprocals (x y : ℝ) :
  (4 * x^2 - 5 * x * y + 4 * y^2 = 5) →
  let S := x^2 + y^2
  (∃ S_max : ℝ, ∀ x y : ℝ, (4 * x^2 - 5 * x * y + 4 * y^2 = 5) → x^2 + y^2 ≤ S_max) ∧
  (∃ S_min : ℝ, ∀ x y : ℝ, (4 * x^2 - 5 * x * y + 4 * y^2 = 5) → S_min ≤ x^2 + y^2) ∧
  (1 / (10/3) + 1 / (10/13) = 8/5) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_sum_reciprocals_l322_32271


namespace NUMINAMATH_CALUDE_fraction_simplification_l322_32268

theorem fraction_simplification (x : ℝ) (h : x ≠ 3) :
  (3 * x) / (x - 3) + (x + 6) / (3 - x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l322_32268


namespace NUMINAMATH_CALUDE_pizzas_successfully_served_l322_32260

theorem pizzas_successfully_served 
  (total_served : ℕ) 
  (returned : ℕ) 
  (h1 : total_served = 9) 
  (h2 : returned = 6) : 
  total_served - returned = 3 :=
by sorry

end NUMINAMATH_CALUDE_pizzas_successfully_served_l322_32260


namespace NUMINAMATH_CALUDE_probability_product_less_than_30_l322_32224

def paco_spinner : Finset ℕ := Finset.range 5
def manu_spinner : Finset ℕ := Finset.range 12

def product_less_than_30 (x : ℕ) (y : ℕ) : Bool :=
  x * y < 30

theorem probability_product_less_than_30 :
  (Finset.filter (λ (pair : ℕ × ℕ) => product_less_than_30 (pair.1 + 1) (pair.2 + 1))
    (paco_spinner.product manu_spinner)).card / (paco_spinner.card * manu_spinner.card : ℚ) = 51 / 60 :=
sorry

end NUMINAMATH_CALUDE_probability_product_less_than_30_l322_32224


namespace NUMINAMATH_CALUDE_third_batch_average_l322_32267

theorem third_batch_average (n₁ n₂ n₃ : ℕ) (a₁ a₂ a_total : ℚ) :
  n₁ = 40 →
  n₂ = 50 →
  n₃ = 60 →
  a₁ = 45 →
  a₂ = 55 →
  a_total = 56333333333333336 / 1000000000000000 →
  (n₁ * a₁ + n₂ * a₂ + n₃ * (3900 / 60)) / (n₁ + n₂ + n₃) = a_total :=
by sorry

end NUMINAMATH_CALUDE_third_batch_average_l322_32267


namespace NUMINAMATH_CALUDE_solve_linear_equation_l322_32220

theorem solve_linear_equation (x : ℝ) :
  3 * x - 7 = 11 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l322_32220


namespace NUMINAMATH_CALUDE_dollar_op_neg_two_three_l322_32208

def dollar_op (a b : ℤ) : ℤ := a * (b + 1) + a * b

theorem dollar_op_neg_two_three : dollar_op (-2) 3 = -14 := by sorry

end NUMINAMATH_CALUDE_dollar_op_neg_two_three_l322_32208


namespace NUMINAMATH_CALUDE_sara_received_six_kittens_l322_32256

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara (initial : ℕ) (to_jessica : ℕ) (left : ℕ) : ℕ :=
  initial - to_jessica - left

/-- Proof that Tim gave 6 kittens to Sara -/
theorem sara_received_six_kittens :
  kittens_to_sara 18 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sara_received_six_kittens_l322_32256


namespace NUMINAMATH_CALUDE_cost_of_bacon_bacon_cost_is_ten_l322_32211

/-- The cost of bacon given Joan's shopping scenario -/
theorem cost_of_bacon (total_budget : ℕ) (hummus_cost : ℕ) (hummus_quantity : ℕ)
  (chicken_cost : ℕ) (vegetable_cost : ℕ) (apple_cost : ℕ) (apple_quantity : ℕ) : ℕ :=
  by
  -- Define the conditions
  have h1 : total_budget = 60 := by sorry
  have h2 : hummus_cost = 5 := by sorry
  have h3 : hummus_quantity = 2 := by sorry
  have h4 : chicken_cost = 20 := by sorry
  have h5 : vegetable_cost = 10 := by sorry
  have h6 : apple_cost = 2 := by sorry
  have h7 : apple_quantity = 5 := by sorry

  -- Prove that the cost of bacon is 10
  sorry

/-- The main theorem stating that the cost of bacon is 10 -/
theorem bacon_cost_is_ten : cost_of_bacon 60 5 2 20 10 2 5 = 10 := by sorry

end NUMINAMATH_CALUDE_cost_of_bacon_bacon_cost_is_ten_l322_32211


namespace NUMINAMATH_CALUDE_average_of_first_group_l322_32238

theorem average_of_first_group (total_average : ℝ) (second_group_average : ℝ) (third_group_average : ℝ)
  (h1 : total_average = 2.80)
  (h2 : second_group_average = 2.3)
  (h3 : third_group_average = 3.7) :
  let total_sum := 6 * total_average
  let second_group_sum := 2 * second_group_average
  let third_group_sum := 2 * third_group_average
  let first_group_sum := total_sum - second_group_sum - third_group_sum
  first_group_sum / 2 = 2.4 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_group_l322_32238


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l322_32236

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 →
    (x^2 - 7) / ((x - 2) * (x - 3) * (x - 5)) = A / (x - 2) + B / (x - 3) + C / (x - 5)) ↔
  A = -1 ∧ B = -1 ∧ C = 3 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l322_32236


namespace NUMINAMATH_CALUDE_combined_age_l322_32210

theorem combined_age (tony_age belinda_age : ℕ) : 
  tony_age = 16 →
  belinda_age = 40 →
  belinda_age = 2 * tony_age + 8 →
  tony_age + belinda_age = 56 :=
by sorry

end NUMINAMATH_CALUDE_combined_age_l322_32210


namespace NUMINAMATH_CALUDE_problem_1_l322_32280

theorem problem_1 : (-2)^3 + (1/9)⁻¹ - (3.14 - Real.pi)^0 = 0 := by sorry

end NUMINAMATH_CALUDE_problem_1_l322_32280


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l322_32254

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 32 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l322_32254


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l322_32213

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = y
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 = x

-- Define the line
def intersection_line (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ (x y : ℝ), circle_O1 x y ∧ circle_O2 x y → intersection_line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l322_32213


namespace NUMINAMATH_CALUDE_sqrt_4_times_9_sqrt_49_over_36_cube_root_a_to_6_sqrt_9a_squared_l322_32241

-- Part a
theorem sqrt_4_times_9 : Real.sqrt (4 * 9) = 6 := by sorry

-- Part b
theorem sqrt_49_over_36 : Real.sqrt (49 / 36) = 7 / 6 := by sorry

-- Part c
theorem cube_root_a_to_6 (a : ℝ) : (a^6)^(1/3 : ℝ) = a^2 := by sorry

-- Part d
theorem sqrt_9a_squared (a : ℝ) : Real.sqrt (9 * a^2) = 3 * a := by sorry

end NUMINAMATH_CALUDE_sqrt_4_times_9_sqrt_49_over_36_cube_root_a_to_6_sqrt_9a_squared_l322_32241


namespace NUMINAMATH_CALUDE_sigma_phi_bounds_l322_32266

open Nat Real

/-- The sum of divisors function -/
noncomputable def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
noncomputable def phi (n : ℕ) : ℕ := sorry

theorem sigma_phi_bounds (n : ℕ) (h : n > 0) : 
  (sigma n * phi n : ℝ) < n^2 ∧ 
  ∃ c : ℝ, c > 0 ∧ ∀ m : ℕ, m > 0 → (sigma m * phi m : ℝ) ≥ c * m^2 := by
  sorry

end NUMINAMATH_CALUDE_sigma_phi_bounds_l322_32266


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l322_32294

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x : ℝ, x^4 - x^3 + x^2 + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l322_32294


namespace NUMINAMATH_CALUDE_square_equality_solution_l322_32287

theorem square_equality_solution : ∃ x : ℝ, (9 - x)^2 = x^2 ∧ x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_solution_l322_32287


namespace NUMINAMATH_CALUDE_quadratic_intersection_l322_32223

/-- Given two quadratic functions f(x) = ax^2 + bx + c and g(x) = 4ax^2 + 2bx + c,
    where b ≠ 0 and c ≠ 0, their intersection points are x = 0 and x = -b/(3a) -/
theorem quadratic_intersection
  (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f := fun x : ℝ => a * x^2 + b * x + c
  let g := fun x : ℝ => 4 * a * x^2 + 2 * b * x + c
  (∃ y, f 0 = y ∧ g 0 = y) ∧
  (∃ y, f (-b / (3 * a)) = y ∧ g (-b / (3 * a)) = y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l322_32223


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l322_32286

/-- Proves that an investment of $31,200 with a simple annual interest rate of 9% yields a monthly interest payment of $234 -/
theorem investment_interest_calculation (principal : ℝ) (annual_rate : ℝ) (monthly_interest : ℝ) : 
  principal = 31200 ∧ annual_rate = 0.09 → monthly_interest = 234 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l322_32286


namespace NUMINAMATH_CALUDE_fixed_point_quadratic_function_l322_32214

theorem fixed_point_quadratic_function (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - (2-m)*x + m
  f (-1) = 3 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_quadratic_function_l322_32214


namespace NUMINAMATH_CALUDE_tan_neg_x_domain_l322_32299

theorem tan_neg_x_domain :
  {x : ℝ | ∀ n : ℤ, x ≠ -π/2 + n*π} = {x : ℝ | ∃ y : ℝ, y = Real.tan (-x)} :=
by sorry

end NUMINAMATH_CALUDE_tan_neg_x_domain_l322_32299


namespace NUMINAMATH_CALUDE_sphere_radius_at_specific_time_l322_32226

/-- The radius of a sphere with a variable density distribution and time-dependent curved surface area. -/
theorem sphere_radius_at_specific_time
  (k ω β c : ℝ)
  (ρ : ℝ → ℝ)
  (A : ℝ → ℝ)
  (h1 : ∀ r, ρ r = k * r^2)
  (h2 : ∀ t, A t = ω * Real.sin (β * t) + c)
  (h3 : A (Real.pi / (2 * β)) = 64 * Real.pi) :
  ∃ r : ℝ, r = 4 ∧ A (Real.pi / (2 * β)) = 4 * Real.pi * r^2 :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_at_specific_time_l322_32226


namespace NUMINAMATH_CALUDE_beverage_production_l322_32203

/-- Represents the number of bottles of beverage A -/
def bottles_A : ℕ := sorry

/-- Represents the number of bottles of beverage B -/
def bottles_B : ℕ := sorry

/-- The amount of additive (in grams) required for one bottle of beverage A -/
def additive_A : ℚ := 1/5

/-- The amount of additive (in grams) required for one bottle of beverage B -/
def additive_B : ℚ := 3/10

/-- The total number of bottles produced -/
def total_bottles : ℕ := 200

/-- The total amount of additive used (in grams) -/
def total_additive : ℚ := 54

theorem beverage_production :
  bottles_A + bottles_B = total_bottles ∧
  additive_A * bottles_A + additive_B * bottles_B = total_additive ∧
  bottles_A = 60 ∧
  bottles_B = 140 := by sorry

end NUMINAMATH_CALUDE_beverage_production_l322_32203


namespace NUMINAMATH_CALUDE_bellas_score_l322_32291

theorem bellas_score (total_students : ℕ) (avg_without_bella : ℚ) (avg_with_bella : ℚ) :
  total_students = 20 →
  avg_without_bella = 82 →
  avg_with_bella = 85 →
  (total_students * avg_with_bella - (total_students - 1) * avg_without_bella : ℚ) = 142 :=
by sorry

end NUMINAMATH_CALUDE_bellas_score_l322_32291


namespace NUMINAMATH_CALUDE_iris_blueberries_l322_32239

/-- The number of blueberries Iris picked -/
def blueberries : ℕ := 30

/-- The number of cranberries Iris' sister picked -/
def cranberries : ℕ := 20

/-- The number of raspberries Iris' brother picked -/
def raspberries : ℕ := 10

/-- The fraction of total berries that are rotten -/
def rotten_fraction : ℚ := 1/3

/-- The fraction of fresh berries that need to be kept -/
def kept_fraction : ℚ := 1/2

/-- The number of berries they can sell -/
def sellable_berries : ℕ := 20

theorem iris_blueberries :
  blueberries = 30 ∧
  (1 - rotten_fraction) * (1 - kept_fraction) * (blueberries + cranberries + raspberries : ℚ) = sellable_berries := by
  sorry

end NUMINAMATH_CALUDE_iris_blueberries_l322_32239


namespace NUMINAMATH_CALUDE_basketball_conference_games_l322_32265

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 5

/-- The total number of games in a season for the basketball conference -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem basketball_conference_games :
  total_games = 185 := by sorry

end NUMINAMATH_CALUDE_basketball_conference_games_l322_32265


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l322_32298

/-- Given a line segment with midpoint (2, 3) and one endpoint (-1, 7),
    prove that the other endpoint is (5, -1). -/
theorem other_endpoint_of_line_segment (A B M : ℝ × ℝ) : 
  M = (2, 3) → A = (-1, 7) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → B = (5, -1) := by
  sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l322_32298


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l322_32219

theorem simplify_trig_expression (α : Real) (h : α ∈ Set.Ioo (π/2) (3*π/4)) :
  Real.sqrt (2 - 2 * Real.sin (2 * α)) - Real.sqrt (1 + Real.cos (2 * α)) = Real.sqrt 2 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l322_32219


namespace NUMINAMATH_CALUDE_scientific_notation_864000_l322_32258

theorem scientific_notation_864000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 864000 = a * (10 : ℝ) ^ n :=
by
  use 8.64, 5
  sorry

end NUMINAMATH_CALUDE_scientific_notation_864000_l322_32258


namespace NUMINAMATH_CALUDE_factory_production_correct_factory_produces_90_refrigerators_per_hour_l322_32247

/-- Represents the production of a factory making refrigerators and coolers -/
structure FactoryProduction where
  refrigerators_per_hour : ℕ
  coolers_per_hour : ℕ
  total_products : ℕ
  days : ℕ
  hours_per_day : ℕ

/-- The conditions of the factory production problem -/
def factory_conditions : FactoryProduction where
  refrigerators_per_hour := 90  -- This is what we want to prove
  coolers_per_hour := 90 + 70
  total_products := 11250
  days := 5
  hours_per_day := 9

/-- Theorem stating that the given conditions satisfy the problem requirements -/
theorem factory_production_correct (fp : FactoryProduction) : 
  fp.coolers_per_hour = fp.refrigerators_per_hour + 70 →
  fp.total_products = (fp.refrigerators_per_hour + fp.coolers_per_hour) * fp.days * fp.hours_per_day →
  fp.refrigerators_per_hour = 90 :=
by
  sorry

/-- The main theorem proving that the factory produces 90 refrigerators per hour -/
theorem factory_produces_90_refrigerators_per_hour : 
  factory_conditions.refrigerators_per_hour = 90 :=
by
  apply factory_production_correct factory_conditions
  · -- Prove that coolers_per_hour = refrigerators_per_hour + 70
    sorry
  · -- Prove that total_products = (refrigerators_per_hour + coolers_per_hour) * days * hours_per_day
    sorry

end NUMINAMATH_CALUDE_factory_production_correct_factory_produces_90_refrigerators_per_hour_l322_32247


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_sides_l322_32263

/-- A right-angled triangle with sides in arithmetic progression and area 486 dm² has sides 27 dm, 36 dm, and 45 dm. -/
theorem right_triangle_arithmetic_sides (a b c : ℝ) : 
  (a * a + b * b = c * c) →  -- Pythagorean theorem
  (b - a = c - b) →  -- Sides in arithmetic progression
  (a * b / 2 = 486) →  -- Area of the triangle
  (a = 27 ∧ b = 36 ∧ c = 45) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_sides_l322_32263


namespace NUMINAMATH_CALUDE_post_height_l322_32234

/-- The height of a cylindrical post given a squirrel's spiral path. -/
theorem post_height (circuit_rise : ℝ) (post_circumference : ℝ) (total_distance : ℝ) : 
  circuit_rise = 4 →
  post_circumference = 3 →
  total_distance = 9 →
  (total_distance / post_circumference) * circuit_rise = 12 :=
by sorry

end NUMINAMATH_CALUDE_post_height_l322_32234


namespace NUMINAMATH_CALUDE_girls_on_playground_l322_32284

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 117) 
  (h2 : boys = 40) : 
  total_children - boys = 77 := by
sorry

end NUMINAMATH_CALUDE_girls_on_playground_l322_32284


namespace NUMINAMATH_CALUDE_methane_moles_in_reaction_l322_32249

/-- 
Proves that the number of moles of Methane combined is equal to 1, given the conditions of the chemical reaction.
-/
theorem methane_moles_in_reaction (x : ℝ) : 
  (x > 0) →  -- Assuming positive number of moles
  (∃ y : ℝ, y > 0 ∧ x + 4 = y + 4) →  -- Mass balance equation
  (1 : ℝ) / x = (1 : ℝ) / 1 →  -- Stoichiometric ratio
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_methane_moles_in_reaction_l322_32249


namespace NUMINAMATH_CALUDE_intersection_point_is_e_e_l322_32228

theorem intersection_point_is_e_e (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x = Real.exp 1 ∧ y = Real.exp 1) →
  (x^y = y^x ∧ y = x) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_e_e_l322_32228


namespace NUMINAMATH_CALUDE_correct_vs_incorrect_calculation_l322_32273

theorem correct_vs_incorrect_calculation : 
  (12 - (3 * 4)) - ((12 - 3) * 4) = -36 := by sorry

end NUMINAMATH_CALUDE_correct_vs_incorrect_calculation_l322_32273


namespace NUMINAMATH_CALUDE_max_distance_complex_circle_l322_32275

theorem max_distance_complex_circle : 
  ∃ (M : ℝ), M = 7 ∧ 
  ∀ (z : ℂ), Complex.abs (z - (4 - 4*I)) ≤ 2 → Complex.abs (z - 1) ≤ M ∧ 
  ∃ (w : ℂ), Complex.abs (w - (4 - 4*I)) ≤ 2 ∧ Complex.abs (w - 1) = M :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_circle_l322_32275


namespace NUMINAMATH_CALUDE_min_disks_needed_prove_min_disks_l322_32221

/-- Represents the storage capacity of a disk in MB -/
def disk_capacity : ℚ := 2

/-- Represents the total number of files -/
def total_files : ℕ := 36

/-- Represents the number of 1.2 MB files -/
def large_files : ℕ := 5

/-- Represents the number of 0.6 MB files -/
def medium_files : ℕ := 16

/-- Represents the size of large files in MB -/
def large_file_size : ℚ := 1.2

/-- Represents the size of medium files in MB -/
def medium_file_size : ℚ := 0.6

/-- Represents the size of small files in MB -/
def small_file_size : ℚ := 0.2

/-- Calculates the number of small files -/
def small_files : ℕ := total_files - large_files - medium_files

/-- Theorem stating the minimum number of disks needed -/
theorem min_disks_needed : ℕ := 14

/-- Proof of the minimum number of disks needed -/
theorem prove_min_disks : min_disks_needed = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_needed_prove_min_disks_l322_32221


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l322_32233

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A quadrilateral in a 2D Cartesian coordinate system -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Predicate to check if two line segments are parallel -/
def parallel (p1 p2 p3 p4 : Point2D) : Prop :=
  (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x)

theorem parallelogram_fourth_vertex 
  (q : Quadrilateral)
  (h1 : parallel q.A q.B q.D q.C)
  (h2 : parallel q.A q.D q.B q.C)
  (h3 : q.A = Point2D.mk (-2) 0)
  (h4 : q.B = Point2D.mk 6 8)
  (h5 : q.C = Point2D.mk 8 6) :
  q.D = Point2D.mk 0 (-2) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l322_32233


namespace NUMINAMATH_CALUDE_function_roots_imply_a_range_l322_32207

theorem function_roots_imply_a_range (a b : ℝ) :
  (∀ x : ℝ, (a * x^2 + b * (x + 1) - 2 = x) → (∃ y z : ℝ, y ≠ z ∧ a * y^2 + b * (y + 1) - 2 = y ∧ a * z^2 + b * (z + 1) - 2 = z)) →
  (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_function_roots_imply_a_range_l322_32207


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l322_32297

theorem complex_number_magnitude (z : ℂ) :
  (1 - z) / (1 + z) = Complex.I ^ 2018 + Complex.I ^ 2019 →
  Complex.abs (2 + z) = 5 * Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l322_32297


namespace NUMINAMATH_CALUDE_other_man_age_is_ten_l322_32269

/-- The age of the other replaced man given the conditions of the problem -/
def other_man_age (initial_men : ℕ) (replaced_men : ℕ) (age_increase : ℕ) 
  (known_man_age : ℕ) (women_avg_age : ℕ) : ℕ :=
  26 - age_increase

/-- Theorem stating the age of the other replaced man -/
theorem other_man_age_is_ten 
  (initial_men : ℕ) 
  (replaced_men : ℕ) 
  (age_increase : ℕ) 
  (known_man_age : ℕ) 
  (women_avg_age : ℕ) 
  (h1 : initial_men = 8)
  (h2 : replaced_men = 2)
  (h3 : age_increase = 2)
  (h4 : known_man_age = 20)
  (h5 : women_avg_age = 23) :
  other_man_age initial_men replaced_men age_increase known_man_age women_avg_age = 10 := by
  sorry


end NUMINAMATH_CALUDE_other_man_age_is_ten_l322_32269


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_l322_32242

theorem arithmetic_sequence_sum_mod (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 3 →
  d = 5 →
  aₙ = 103 →
  0 ≤ n →
  n < 17 →
  (n : ℤ) ≡ (n * (a₁ + aₙ) / 2) [ZMOD 17] →
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_l322_32242


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_3a_l322_32296

theorem factorization_a_squared_minus_3a (a : ℝ) : a^2 - 3*a = a*(a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_3a_l322_32296


namespace NUMINAMATH_CALUDE_cube_properties_l322_32274

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D
  edgeLength : ℝ

/-- Returns true if two lines are skew -/
def areSkewLines (l1 l2 : Line3D) : Prop := sorry

/-- Returns true if a line is perpendicular to two other lines -/
def isPerpendicularToLines (l : Line3D) (l1 l2 : Line3D) : Prop := sorry

/-- Calculates the distance between two skew lines -/
def distanceBetweenSkewLines (l1 l2 : Line3D) : ℝ := sorry

theorem cube_properties (cube : Cube) :
  let AA₁ : Line3D := { point := cube.A, direction := { x := 0, y := 0, z := 1 } }
  let BC : Line3D := { point := cube.B, direction := { x := 1, y := 0, z := 0 } }
  let AB : Line3D := { point := cube.A, direction := { x := 1, y := 0, z := 0 } }
  areSkewLines AA₁ BC ∧
  isPerpendicularToLines AB AA₁ BC ∧
  distanceBetweenSkewLines AA₁ BC = cube.edgeLength := by
  sorry

end NUMINAMATH_CALUDE_cube_properties_l322_32274


namespace NUMINAMATH_CALUDE_deposit_calculation_l322_32251

theorem deposit_calculation (initial_deposit : ℚ) : 
  (initial_deposit - initial_deposit / 4 - (initial_deposit - initial_deposit / 4) * 4 / 9 - 640) = 3 / 20 * initial_deposit →
  initial_deposit = 2400 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l322_32251


namespace NUMINAMATH_CALUDE_moe_has_least_money_l322_32261

-- Define the set of people
inductive Person : Type
| Bo : Person
| Coe : Person
| Flo : Person
| Jo : Person
| Moe : Person

-- Define the money function
variable (money : Person → ℝ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom flo_more_than_jo_bo : money Person.Flo > money Person.Jo ∧ money Person.Flo > money Person.Bo
axiom bo_coe_more_than_moe : money Person.Bo > money Person.Moe ∧ money Person.Coe > money Person.Moe
axiom jo_between_bo_moe : money Person.Jo > money Person.Moe ∧ money Person.Jo < money Person.Bo

-- Define the theorem
theorem moe_has_least_money :
  ∀ (p : Person), p ≠ Person.Moe → money Person.Moe < money p :=
sorry

end NUMINAMATH_CALUDE_moe_has_least_money_l322_32261


namespace NUMINAMATH_CALUDE_stock_price_problem_l322_32240

theorem stock_price_problem (price_less_expensive : ℝ) (price_more_expensive : ℝ) : 
  price_more_expensive = 2 * price_less_expensive →
  14 * price_more_expensive + 26 * price_less_expensive = 2106 →
  price_more_expensive = 78 := by
sorry

end NUMINAMATH_CALUDE_stock_price_problem_l322_32240


namespace NUMINAMATH_CALUDE_total_project_hours_l322_32283

def project_hours (kate_hours : ℝ) : ℝ × ℝ × ℝ :=
  let pat_hours := 2 * kate_hours
  let mark_hours := kate_hours + 75
  (pat_hours, kate_hours, mark_hours)

theorem total_project_hours :
  ∃ (kate_hours : ℝ),
    let (pat_hours, _, mark_hours) := project_hours kate_hours
    pat_hours = (1/3) * mark_hours ∧
    (pat_hours + kate_hours + mark_hours) = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_project_hours_l322_32283


namespace NUMINAMATH_CALUDE_base_conversion_l322_32290

theorem base_conversion (b : ℝ) (h : b > 0) : 53 = 1 * b^2 + 0 * b + 3 → b = Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l322_32290


namespace NUMINAMATH_CALUDE_fifth_quiz_score_l322_32206

def quiz_scores : List ℕ := [90, 98, 92, 94]
def desired_average : ℕ := 94
def total_quizzes : ℕ := 5

theorem fifth_quiz_score (scores : List ℕ) (avg : ℕ) (total : ℕ) :
  scores = quiz_scores ∧ avg = desired_average ∧ total = total_quizzes →
  (scores.sum + (avg * total - scores.sum)) / total = avg ∧
  avg * total - scores.sum = 96 := by
  sorry

end NUMINAMATH_CALUDE_fifth_quiz_score_l322_32206


namespace NUMINAMATH_CALUDE_fraction_equality_l322_32222

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 25)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 8) :
  a / d = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l322_32222


namespace NUMINAMATH_CALUDE_k_value_proof_l322_32288

theorem k_value_proof (k : ℝ) (h1 : k ≠ 0) 
  (h2 : ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 12)) : 
  k = 12 := by
  sorry

end NUMINAMATH_CALUDE_k_value_proof_l322_32288


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l322_32264

-- Define the polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum_simplification :
  ∀ x : ℝ, p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l322_32264


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_l322_32289

/-- A configuration of four mutually tangent spheres on a plane -/
structure SphericalConfiguration where
  radius : ℝ
  mutually_tangent : Bool
  on_plane : Bool

/-- A tetrahedron circumscribed around four spheres -/
structure CircumscribedTetrahedron where
  spheres : SphericalConfiguration
  edge_length : ℝ

/-- The theorem stating that the edge length of a tetrahedron circumscribed around
    four mutually tangent spheres of radius 2 is equal to 4 -/
theorem tetrahedron_edge_length 
  (config : SphericalConfiguration) 
  (tetra : CircumscribedTetrahedron) :
  config.radius = 2 ∧ 
  config.mutually_tangent = true ∧ 
  config.on_plane = true ∧
  tetra.spheres = config →
  tetra.edge_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_l322_32289


namespace NUMINAMATH_CALUDE_middle_circle_radius_l322_32248

/-- Represents the radii of five circles in an arithmetic sequence -/
def CircleRadii := Fin 5 → ℝ

/-- The property that the radii form an arithmetic sequence -/
def is_arithmetic_sequence (r : CircleRadii) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 4, r (i + 1) = r i + d

/-- The theorem statement -/
theorem middle_circle_radius 
  (r : CircleRadii) 
  (h_arithmetic : is_arithmetic_sequence r)
  (h_smallest : r 0 = 6)
  (h_largest : r 4 = 30) :
  r 2 = 18 := by
sorry

end NUMINAMATH_CALUDE_middle_circle_radius_l322_32248


namespace NUMINAMATH_CALUDE_investment_time_solution_l322_32232

/-- Represents a partner in the investment problem -/
structure Partner where
  investment : ℝ
  time : ℝ
  profit : ℝ

/-- The investment problem -/
def InvestmentProblem (p q : Partner) : Prop :=
  p.investment / q.investment = 7 / 5 ∧
  p.profit / q.profit = 7 / 10 ∧
  p.time = 7 ∧
  p.investment * p.time / (q.investment * q.time) = p.profit / q.profit

theorem investment_time_solution (p q : Partner) :
  InvestmentProblem p q → q.time = 14 := by
  sorry

end NUMINAMATH_CALUDE_investment_time_solution_l322_32232


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l322_32262

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, mx - y + 1 = 0 ∧ x + m^2*y - 2 = 0 → 
   (m * 1) * 1 + 1 * m^2 = 0) → 
  m = 0 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l322_32262


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l322_32244

/-- The number of blocks in the neighborhood -/
def num_blocks : ℕ := 16

/-- The number of junk mail pieces given to each house -/
def mail_per_house : ℕ := 4

/-- The total number of junk mail pieces given out -/
def total_mail : ℕ := 1088

/-- The number of houses in each block -/
def houses_per_block : ℕ := 17

theorem junk_mail_distribution :
  houses_per_block * num_blocks * mail_per_house = total_mail :=
by sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l322_32244


namespace NUMINAMATH_CALUDE_senior_discount_percentage_l322_32225

def shorts_price : ℝ := 15
def shirts_price : ℝ := 17
def num_shorts : ℕ := 3
def num_shirts : ℕ := 5
def total_paid : ℝ := 117

theorem senior_discount_percentage :
  let total_cost := shorts_price * num_shorts + shirts_price * num_shirts
  let discount := total_cost - total_paid
  let discount_percentage := (discount / total_cost) * 100
  discount_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_senior_discount_percentage_l322_32225


namespace NUMINAMATH_CALUDE_librarian_crates_l322_32235

theorem librarian_crates (novels comics documentaries albums : ℕ) 
  (items_per_crate : ℕ) (h1 : novels = 145) (h2 : comics = 271) 
  (h3 : documentaries = 419) (h4 : albums = 209) (h5 : items_per_crate = 9) : 
  (novels + comics + documentaries + albums + items_per_crate - 1) / items_per_crate = 117 := by
  sorry

end NUMINAMATH_CALUDE_librarian_crates_l322_32235


namespace NUMINAMATH_CALUDE_mikes_books_l322_32259

theorem mikes_books (initial_books new_books : ℕ) : 
  initial_books = 35 → new_books = 56 → initial_books + new_books = 91 := by
  sorry

end NUMINAMATH_CALUDE_mikes_books_l322_32259


namespace NUMINAMATH_CALUDE_john_pushups_l322_32253

theorem john_pushups (zachary_pushups : ℕ) (david_more_than_zachary : ℕ) (john_less_than_david : ℕ)
  (h1 : zachary_pushups = 51)
  (h2 : david_more_than_zachary = 22)
  (h3 : john_less_than_david = 4) :
  zachary_pushups + david_more_than_zachary - john_less_than_david = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_john_pushups_l322_32253


namespace NUMINAMATH_CALUDE_triangle_area_example_l322_32276

/-- The area of a triangle given its vertices -/
def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  let v := (a.1 - c.1, a.2 - c.2)
  let w := (b.1 - c.1, b.2 - c.2)
  0.5 * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (3, -5), (-2, 0), and (5, -8) is 2.5 -/
theorem triangle_area_example : triangleArea (3, -5) (-2, 0) (5, -8) = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_example_l322_32276


namespace NUMINAMATH_CALUDE_athena_spent_14_l322_32205

/-- The total amount Athena spent on snacks for her friends -/
def total_spent (sandwich_price : ℚ) (sandwich_quantity : ℕ) (drink_price : ℚ) (drink_quantity : ℕ) : ℚ :=
  sandwich_price * sandwich_quantity + drink_price * drink_quantity

/-- Theorem: Athena spent $14 in total -/
theorem athena_spent_14 :
  total_spent 3 3 (5/2) 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_athena_spent_14_l322_32205


namespace NUMINAMATH_CALUDE_tournament_outcomes_l322_32217

/-- Represents the number of players in the tournament -/
def num_players : Nat := 5

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : Nat := 2

/-- Represents the number of elimination rounds -/
def num_rounds : Nat := 4

/-- Calculates the total number of possible outcomes for the tournament -/
def total_outcomes : Nat := outcomes_per_match ^ num_rounds

/-- Theorem stating that the total number of possible outcomes is 16 -/
theorem tournament_outcomes :
  total_outcomes = 16 := by sorry

end NUMINAMATH_CALUDE_tournament_outcomes_l322_32217


namespace NUMINAMATH_CALUDE_quadratic_properties_l322_32215

variable (a b c p q : ℝ)
variable (f : ℝ → ℝ)

-- Define the quadratic function
def is_quadratic (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_properties (h : is_quadratic f a b c) (hpq : p ≠ q) :
  (f p = f q → f (p + q) = c) ∧
  (f (p + q) = c → p + q = 0 ∨ f p = f q) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l322_32215


namespace NUMINAMATH_CALUDE_base_ten_solution_l322_32278

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if the equation 253_b + 146_b = 410_b holds for a given base b --/
def equation_holds (b : Nat) : Prop :=
  to_decimal [2, 5, 3] b + to_decimal [1, 4, 6] b = to_decimal [4, 1, 0] b

theorem base_ten_solution :
  ∃ (b : Nat), b > 9 ∧ equation_holds b ∧ ∀ (x : Nat), x ≠ b → ¬(equation_holds x) :=
sorry

end NUMINAMATH_CALUDE_base_ten_solution_l322_32278


namespace NUMINAMATH_CALUDE_parabola_translation_specific_parabola_translation_l322_32285

/-- Represents a parabola in the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a,
    b := p.a * h^2 + p.b + v }

theorem parabola_translation (p : Parabola) (h v : ℝ) :
  (translate p h v).a * (X - h)^2 + (translate p h v).b = p.a * X^2 + p.b + v :=
by sorry

theorem specific_parabola_translation :
  let p : Parabola := { a := 2, b := 3 }
  let translated := translate p 3 2
  translated.a * (X - 3)^2 + translated.b = 2 * (X - 3)^2 + 5 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_specific_parabola_translation_l322_32285


namespace NUMINAMATH_CALUDE_annie_travel_distance_l322_32270

/-- The number of blocks Annie walked from her house to the bus stop -/
def blocks_to_bus_stop : ℕ := 5

/-- The number of blocks Annie rode the bus to the coffee shop -/
def blocks_on_bus : ℕ := 7

/-- The total number of blocks Annie traveled in her round trip -/
def total_blocks : ℕ := 2 * (blocks_to_bus_stop + blocks_on_bus)

theorem annie_travel_distance : total_blocks = 24 := by sorry

end NUMINAMATH_CALUDE_annie_travel_distance_l322_32270


namespace NUMINAMATH_CALUDE_exists_right_triangle_with_different_colors_l322_32252

-- Define the color type
inductive Color
  | Blue
  | Green
  | Red

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- State the existence of at least one point of each color
axiom exists_blue : ∃ p : Point, coloring p = Color.Blue
axiom exists_green : ∃ p : Point, coloring p = Color.Green
axiom exists_red : ∃ p : Point, coloring p = Color.Red

-- Define a right triangle
def is_right_triangle (p q r : Point) : Prop := sorry

-- State the theorem
theorem exists_right_triangle_with_different_colors :
  ∃ p q r : Point, is_right_triangle p q r ∧
    coloring p ≠ coloring q ∧
    coloring q ≠ coloring r ∧
    coloring r ≠ coloring p :=
sorry

end NUMINAMATH_CALUDE_exists_right_triangle_with_different_colors_l322_32252


namespace NUMINAMATH_CALUDE_work_rates_solution_l322_32201

/-- Work rates of workers -/
structure WorkRates where
  casey : ℚ
  bill : ℚ
  alec : ℚ

/-- Given conditions about job completion times -/
def job_conditions (w : WorkRates) : Prop :=
  10 * (w.casey + w.bill) = 1 ∧
  9 * (w.casey + w.alec) = 1 ∧
  8 * (w.alec + w.bill) = 1

/-- Theorem stating the work rates of Casey, Bill, and Alec -/
theorem work_rates_solution :
  ∃ w : WorkRates,
    job_conditions w ∧
    w.casey = (12.8 - 41) / 720 ∧
    w.bill = 41 / 720 ∧
    w.alec = 49 / 720 := by
  sorry

end NUMINAMATH_CALUDE_work_rates_solution_l322_32201


namespace NUMINAMATH_CALUDE_translation_right_4_units_l322_32202

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_4_units :
  let P : Point := { x := -5, y := 4 }
  let P' : Point := translateRight P 4
  P'.x = -1 ∧ P'.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_4_units_l322_32202


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l322_32279

def systematicSample (totalItems : Nat) (sampleSize : Nat) : List Nat :=
  sorry

theorem correct_systematic_sample :
  let totalItems : Nat := 50
  let sampleSize : Nat := 5
  let samplingInterval : Nat := totalItems / sampleSize
  let sample := systematicSample totalItems sampleSize
  samplingInterval = 10 ∧ sample = [7, 17, 27, 37, 47] := by sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l322_32279


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l322_32246

theorem negation_of_cube_odd_is_odd :
  (¬ ∀ n : ℤ, Odd n → Odd (n^3)) ↔ (∃ n : ℤ, Odd n ∧ ¬Odd (n^3)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l322_32246


namespace NUMINAMATH_CALUDE_prop_C_and_D_l322_32277

theorem prop_C_and_D : 
  (∀ a b : ℝ, a > b → a^3 > b^3) ∧ 
  (∀ a b c d : ℝ, (a > b ∧ c > d) → a - d > b - c) := by
  sorry

end NUMINAMATH_CALUDE_prop_C_and_D_l322_32277


namespace NUMINAMATH_CALUDE_distribute_seven_into_four_l322_32250

/-- Number of ways to distribute indistinguishable objects into distinct containers -/
def distribute_objects (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 indistinguishable objects into 4 distinct containers -/
theorem distribute_seven_into_four :
  distribute_objects 7 4 = 132 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_into_four_l322_32250


namespace NUMINAMATH_CALUDE_exists_21_game_period_l322_32243

-- Define the type for the sequence of cumulative games
def CumulativeGames := Nat → Nat

-- Define the properties of the sequence
def ValidSequence (a : CumulativeGames) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, a (n + 7) - a n ≤ 10) ∧
  (a 0 ≥ 1) ∧ (a 42 ≤ 60)

-- Theorem statement
theorem exists_21_game_period (a : CumulativeGames) 
  (h : ValidSequence a) : 
  ∃ k n : Nat, k + n ≤ 42 ∧ a (k + n) - a k = 21 := by
  sorry

end NUMINAMATH_CALUDE_exists_21_game_period_l322_32243


namespace NUMINAMATH_CALUDE_reemas_correct_marks_l322_32231

/-- Proves that given a class of 35 students with an initial average of 72,
    if one student's marks are changed from 46 to x, resulting in a new average of 71.71,
    then x = 36.85 -/
theorem reemas_correct_marks 
  (num_students : Nat)
  (initial_average : ℚ)
  (incorrect_marks : ℚ)
  (new_average : ℚ)
  (h1 : num_students = 35)
  (h2 : initial_average = 72)
  (h3 : incorrect_marks = 46)
  (h4 : new_average = 71.71)
  : ∃ x : ℚ, x = 36.85 ∧ 
    (num_students : ℚ) * initial_average - incorrect_marks + x = 
    (num_students : ℚ) * new_average :=
by
  sorry


end NUMINAMATH_CALUDE_reemas_correct_marks_l322_32231


namespace NUMINAMATH_CALUDE_at_least_four_same_acquaintances_l322_32272

theorem at_least_four_same_acquaintances :
  ∀ (contestants : Finset Nat) (acquaintances : Nat → Finset Nat),
    contestants.card = 90 →
    (∀ x ∈ contestants, (acquaintances x).card ≥ 60) →
    (∀ x ∈ contestants, (acquaintances x) ⊆ contestants) →
    (∀ x ∈ contestants, x ∉ acquaintances x) →
    ∃ n : Nat, ∃ s : Finset Nat, s ⊆ contestants ∧ s.card ≥ 4 ∧
      ∀ x ∈ s, (acquaintances x).card = n :=
by
  sorry


end NUMINAMATH_CALUDE_at_least_four_same_acquaintances_l322_32272


namespace NUMINAMATH_CALUDE_sum_less_than_addends_l322_32218

theorem sum_less_than_addends : ∃ a b : ℝ, a + b < a ∧ a + b < b := by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_addends_l322_32218


namespace NUMINAMATH_CALUDE_multipleOfThree_is_closed_l322_32227

def ClosedSet (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def MultipleOfThree : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem multipleOfThree_is_closed : ClosedSet MultipleOfThree := by
  sorry

end NUMINAMATH_CALUDE_multipleOfThree_is_closed_l322_32227


namespace NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l322_32293

/-- The number of pumpkins at Moonglow Orchard -/
def x : ℕ := 14

/-- The number of pumpkins at Sunshine Orchard -/
def y : ℕ := 3 * x^2 + 12

theorem sunshine_orchard_pumpkins : y = 600 := by
  sorry

end NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l322_32293


namespace NUMINAMATH_CALUDE_factorization_equality_l322_32229

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l322_32229


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l322_32216

theorem fixed_point_on_line (m : ℝ) : 
  (m + 2) * (-4/5) + (m - 3) * (4/5) + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l322_32216


namespace NUMINAMATH_CALUDE_all_cards_same_number_l322_32204

theorem all_cards_same_number (n : ℕ) (c : Fin n → ℕ) : 
  (∀ i : Fin n, c i ∈ Finset.range n) →
  (∀ s : Finset (Fin n), (s.sum c) % (n + 1) ≠ 0) →
  (∀ i j : Fin n, c i = c j) :=
by sorry

end NUMINAMATH_CALUDE_all_cards_same_number_l322_32204
