import Mathlib

namespace ends_in_zero_l3674_367437

theorem ends_in_zero (a : ℤ) (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℤ, a^(2^n + 1) - a = 10 * k := by
  sorry

end ends_in_zero_l3674_367437


namespace minimum_of_x_squared_l3674_367445

theorem minimum_of_x_squared :
  ∃ (m : ℝ), m = 0 ∧ ∀ x : ℝ, x^2 ≥ m := by sorry

end minimum_of_x_squared_l3674_367445


namespace catch_up_distance_l3674_367485

/-- The problem of two people traveling at different speeds --/
theorem catch_up_distance
  (speed_a : ℝ)
  (speed_b : ℝ)
  (delay : ℝ)
  (h1 : speed_a = 10)
  (h2 : speed_b = 20)
  (h3 : delay = 6)
  : speed_b * (speed_a * delay / (speed_b - speed_a)) = 120 :=
by sorry

end catch_up_distance_l3674_367485


namespace probability_abs_diff_gt_half_l3674_367477

/-- A coin flip result -/
inductive CoinFlip
| Heads
| Tails

/-- The result of the number selection process -/
inductive NumberSelection
| Uniform : ℝ → NumberSelection
| Zero
| One

/-- The process of selecting a number based on coin flips -/
def selectNumber (flip1 : CoinFlip) (flip2 : CoinFlip) (u : ℝ) : NumberSelection :=
  match flip1 with
  | CoinFlip.Heads => match flip2 with
    | CoinFlip.Heads => NumberSelection.Zero
    | CoinFlip.Tails => NumberSelection.One
  | CoinFlip.Tails => NumberSelection.Uniform u

/-- The probability measure for the problem -/
noncomputable def P : Set (NumberSelection × NumberSelection) → ℝ := sorry

/-- The event that |x-y| > 1/2 -/
def event : Set (NumberSelection × NumberSelection) :=
  {pair | let (x, y) := pair
          match x, y with
          | NumberSelection.Uniform x', NumberSelection.Uniform y' => |x' - y'| > 1/2
          | NumberSelection.Zero, NumberSelection.Uniform y' => y' < 1/2
          | NumberSelection.One, NumberSelection.Uniform y' => y' < 1/2
          | NumberSelection.Uniform x', NumberSelection.Zero => x' > 1/2
          | NumberSelection.Uniform x', NumberSelection.One => x' < 1/2
          | NumberSelection.Zero, NumberSelection.One => true
          | NumberSelection.One, NumberSelection.Zero => true
          | _, _ => false}

theorem probability_abs_diff_gt_half :
  P event = 7/16 := by sorry

end probability_abs_diff_gt_half_l3674_367477


namespace inequality_implication_l3674_367426

theorem inequality_implication (a b c d e : ℝ) :
  a * b^2 * c^3 * d^4 * e^5 < 0 → a * b^2 * c * d^4 * e < 0 := by
  sorry

end inequality_implication_l3674_367426


namespace joyce_land_theorem_l3674_367496

/-- Calculates the suitable land for growing vegetables given the previous property size,
    the factor by which the new property is larger, and the size of the pond. -/
def suitable_land (previous_property : ℝ) (size_factor : ℝ) (pond_size : ℝ) : ℝ :=
  previous_property * size_factor - pond_size

/-- Theorem stating that given a previous property of 2 acres, a new property 8 times larger,
    and a 3-acre pond, the land suitable for growing vegetables is 13 acres. -/
theorem joyce_land_theorem :
  suitable_land 2 8 3 = 13 := by
  sorry

end joyce_land_theorem_l3674_367496


namespace simple_interest_rate_correct_l3674_367412

/-- The simple interest rate that makes a sum of money increase to 7/6 of itself in 6 years -/
def simple_interest_rate : ℚ :=
  100 / 36

/-- The time period in years -/
def time_period : ℕ := 6

/-- The ratio of final amount to initial amount -/
def final_to_initial_ratio : ℚ := 7 / 6

theorem simple_interest_rate_correct : 
  final_to_initial_ratio = 1 + (simple_interest_rate * time_period) / 100 :=
by sorry

end simple_interest_rate_correct_l3674_367412


namespace floor_paving_cost_l3674_367401

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (rate : ℝ) 
  (h1 : length = 5) 
  (h2 : width = 4.75) 
  (h3 : rate = 900) : 
  length * width * rate = 21375 := by
  sorry

end floor_paving_cost_l3674_367401


namespace perpendicular_tangents_condition_l3674_367441

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + x^2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * a * Real.sin (x/2) * Real.cos (x/2) - x

theorem perpendicular_tangents_condition (a : ℝ) : 
  (∀ x₁ : ℝ, x₁ > -1 → ∃ x₂ : ℝ, 
    (1 / (x₁ + 1) + 2 * x₁) * (Real.sqrt 2 / 2 * Real.cos x₂ - 1) = -1) → 
  |a| ≥ Real.sqrt 2 :=
by sorry

end perpendicular_tangents_condition_l3674_367441


namespace percentage_of_men_in_company_l3674_367490

theorem percentage_of_men_in_company 
  (total_employees : ℝ) 
  (men : ℝ) 
  (women : ℝ) 
  (h1 : men + women = total_employees)
  (h2 : men * 0.5 + women * 0.1666666666666669 = total_employees * 0.4)
  (h3 : men > 0)
  (h4 : women > 0)
  (h5 : total_employees > 0) : 
  men / total_employees = 0.7 := by
sorry

end percentage_of_men_in_company_l3674_367490


namespace complex_cube_roots_sum_of_powers_l3674_367435

theorem complex_cube_roots_sum_of_powers (ω ω' : ℂ) :
  ω^3 = 1 → ω'^3 = 1 → ω = (-1 + Complex.I * Real.sqrt 3) / 2 → ω' = (-1 - Complex.I * Real.sqrt 3) / 2 →
  ω^12 + ω'^12 = 2 := by
  sorry

end complex_cube_roots_sum_of_powers_l3674_367435


namespace base_number_proof_l3674_367414

theorem base_number_proof (x : ℝ) : 16^8 = x^16 → x = 4 := by
  sorry

end base_number_proof_l3674_367414


namespace strawberry_harvest_l3674_367484

/-- Calculates the total number of strawberries harvested from a rectangular garden -/
theorem strawberry_harvest (length width : ℕ) (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) :
  length = 10 →
  width = 7 →
  plants_per_sqft = 3 →
  strawberries_per_plant = 12 →
  length * width * plants_per_sqft * strawberries_per_plant = 2520 := by
  sorry

end strawberry_harvest_l3674_367484


namespace geometric_arithmetic_sequence_l3674_367463

theorem geometric_arithmetic_sequence (a₁ : ℝ) (h : a₁ ≠ 0) :
  ∃! (s : Finset ℝ), s.card = 2 ∧
    ∀ q ∈ s, 2 * (a₁ * q^4) = 4 * a₁ + (-2 * (a₁ * q^2)) :=
by sorry

end geometric_arithmetic_sequence_l3674_367463


namespace hexagon_angle_measure_l3674_367406

theorem hexagon_angle_measure :
  ∀ (a b c d e f : ℝ),
    a = 135 ∧ b = 120 ∧ c = 105 ∧ d = 150 ∧ e = 110 →
    (a + b + c + d + e + f = 720) →
    f = 100 := by
  sorry

end hexagon_angle_measure_l3674_367406


namespace vegetable_difference_is_30_l3674_367443

/-- Calculates the difference between initial and remaining vegetables after exchanges --/
def vegetable_difference (
  initial_tomatoes : ℕ)
  (initial_carrots : ℕ)
  (initial_cucumbers : ℕ)
  (initial_bell_peppers : ℕ)
  (picked_tomatoes : ℕ)
  (picked_carrots : ℕ)
  (picked_cucumbers : ℕ)
  (picked_bell_peppers : ℕ)
  (neighbor1_tomatoes : ℕ)
  (neighbor1_carrots : ℕ)
  (neighbor2_tomatoes : ℕ)
  (neighbor2_cucumbers : ℕ)
  (neighbor2_radishes : ℕ)
  (neighbor3_bell_peppers : ℕ) : ℕ :=
  let initial_total := initial_tomatoes + initial_carrots + initial_cucumbers + initial_bell_peppers
  let remaining_tomatoes := initial_tomatoes - picked_tomatoes - neighbor1_tomatoes - neighbor2_tomatoes
  let remaining_carrots := initial_carrots - picked_carrots - neighbor1_carrots
  let remaining_cucumbers := initial_cucumbers - picked_cucumbers - neighbor2_cucumbers
  let remaining_bell_peppers := initial_bell_peppers - picked_bell_peppers - neighbor3_bell_peppers
  let remaining_total := remaining_tomatoes + remaining_carrots + remaining_cucumbers + remaining_bell_peppers + neighbor2_radishes
  initial_total - remaining_total

/-- The difference between initial and remaining vegetables is 30 --/
theorem vegetable_difference_is_30 : 
  vegetable_difference 17 13 8 15 5 6 3 8 3 2 2 3 5 3 = 30 := by
  sorry

end vegetable_difference_is_30_l3674_367443


namespace aiyanna_cookies_l3674_367469

def alyssa_cookies : ℕ := 129
def cookie_difference : ℕ := 11

theorem aiyanna_cookies : ℕ := alyssa_cookies + cookie_difference

#check aiyanna_cookies -- This should return 140

end aiyanna_cookies_l3674_367469


namespace subset_condition_l3674_367480

theorem subset_condition (a : ℝ) : 
  let A := {x : ℝ | |x - (a+1)^2/2| ≤ (a-1)^2/2}
  let B := {x : ℝ | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0}
  (A ⊆ B) ↔ (1 ≤ a ∧ a ≤ 3) ∨ a = -1 := by sorry

end subset_condition_l3674_367480


namespace complex_number_equality_l3674_367492

theorem complex_number_equality (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (2 - Complex.I)
  Complex.re z = Complex.im z → a = 1/3 := by
sorry

end complex_number_equality_l3674_367492


namespace polynomial_division_quotient_l3674_367468

theorem polynomial_division_quotient : 
  ∀ (x : ℝ), (10 * x^3 + 20 * x^2 - 9 * x + 6) = (2 * x + 6) * (5 * x^2 - 5 * x + 3) + (-57) := by
  sorry

end polynomial_division_quotient_l3674_367468


namespace seashell_ratio_correct_l3674_367446

/-- Represents the number of seashells found by each person -/
structure SeashellCount where
  mary : ℕ
  jessica : ℕ
  linda : ℕ

/-- Represents a ratio as a triple of natural numbers -/
structure Ratio where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The actual seashell counts -/
def actualCounts : SeashellCount :=
  { mary := 18, jessica := 41, linda := 27 }

/-- The expected ratio -/
def expectedRatio : Ratio :=
  { first := 18, second := 41, third := 27 }

/-- Theorem stating that the ratio of seashells found is as expected -/
theorem seashell_ratio_correct :
  let counts := actualCounts
  (counts.mary : ℚ) / (counts.jessica : ℚ) = (expectedRatio.first : ℚ) / (expectedRatio.second : ℚ) ∧
  (counts.jessica : ℚ) / (counts.linda : ℚ) = (expectedRatio.second : ℚ) / (expectedRatio.third : ℚ) :=
sorry

end seashell_ratio_correct_l3674_367446


namespace circumscribed_circle_area_l3674_367418

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  let R := s / Real.sqrt 3
  π * R^2 = 48 * π := by sorry

end circumscribed_circle_area_l3674_367418


namespace coin_difference_l3674_367471

def coin_values : List ℕ := [5, 10, 25, 50]

def target_amount : ℕ := 75

def min_coins (values : List ℕ) (target : ℕ) : ℕ := sorry

def max_coins (values : List ℕ) (target : ℕ) : ℕ := sorry

theorem coin_difference :
  max_coins coin_values target_amount - min_coins coin_values target_amount = 13 := by
  sorry

end coin_difference_l3674_367471


namespace apple_relationship_l3674_367452

/-- Proves the relationship between bruised and wormy apples --/
theorem apple_relationship (total_apples wormy_ratio raw_apples : ℕ) 
  (h1 : total_apples = 85)
  (h2 : wormy_ratio = 5)
  (h3 : raw_apples = 42) : 
  ∃ (bruised wormy : ℕ), 
    wormy = total_apples / wormy_ratio ∧ 
    bruised = total_apples - raw_apples - wormy ∧ 
    bruised = wormy + 9 :=
by sorry

end apple_relationship_l3674_367452


namespace box_volume_increase_l3674_367451

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4000)
  (surface_area : 2 * l * w + 2 * w * h + 2 * h * l = 1680)
  (edge_sum : 4 * l + 4 * w + 4 * h = 200) :
  (l + 2) * (w + 3) * (h + 1) = 5736 := by
  sorry

end box_volume_increase_l3674_367451


namespace solution_set_part_I_range_of_a_part_II_l3674_367424

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part I
theorem solution_set_part_I :
  ∀ x : ℝ, f (-2) x + f (-2) (2*x) > 2 ↔ x < -2 ∨ x > -2/3 :=
sorry

-- Theorem for part II
theorem range_of_a_part_II :
  ∀ a : ℝ, a < 0 → (∃ x : ℝ, f a x + f a (2*x) < 1/2) → -1 < a ∧ a < 0 :=
sorry

end solution_set_part_I_range_of_a_part_II_l3674_367424


namespace system_solution_expression_simplification_l3674_367440

-- Problem 1
theorem system_solution (s t : ℚ) : 
  2 * s + 3 * t = 2 ∧ 2 * s - 6 * t = -1 → s = 1/2 ∧ t = 1/3 := by sorry

-- Problem 2
theorem expression_simplification (x y : ℚ) (h : x ≠ 0) : 
  ((x - y)^2 + (x + y) * (x - y)) / (2 * x) = x - y := by sorry

end system_solution_expression_simplification_l3674_367440


namespace garden_snake_length_l3674_367459

-- Define the lengths of the snakes
def boa_length : Float := 1.428571429
def garden_snake_ratio : Float := 7.0

-- Theorem statement
theorem garden_snake_length : 
  boa_length * garden_snake_ratio = 10.0 := by sorry

end garden_snake_length_l3674_367459


namespace victor_percentage_proof_l3674_367402

def max_marks : ℝ := 450
def victor_marks : ℝ := 405

theorem victor_percentage_proof :
  (victor_marks / max_marks) * 100 = 90 := by
  sorry

end victor_percentage_proof_l3674_367402


namespace sum_of_factorization_coefficients_l3674_367488

theorem sum_of_factorization_coefficients :
  ∀ (a b c : ℤ),
  (∀ x : ℝ, x^2 + 17*x + 70 = (x + a) * (x + b)) →
  (∀ x : ℝ, x^2 - 19*x + 84 = (x - b) * (x - c)) →
  a + b + c = 29 := by
sorry

end sum_of_factorization_coefficients_l3674_367488


namespace sequence_monotonicity_l3674_367474

theorem sequence_monotonicity (k : ℝ) : 
  (∀ n : ℕ, (n + 1)^2 + k*(n + 1) + 2 > n^2 + k*n + 2) → k > -3 := by
  sorry

end sequence_monotonicity_l3674_367474


namespace billion_to_scientific_notation_l3674_367457

theorem billion_to_scientific_notation :
  (6.1 : ℝ) * 1000000000 = (6.1 : ℝ) * (10 ^ 8) :=
by sorry

end billion_to_scientific_notation_l3674_367457


namespace solution_set_of_decreasing_function_l3674_367436

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem solution_set_of_decreasing_function (f : ℝ → ℝ) 
  (h : DecreasingFunction f) : 
  {x : ℝ | f x > f 1} = {x : ℝ | x < 1} := by
  sorry

end solution_set_of_decreasing_function_l3674_367436


namespace total_milk_production_l3674_367427

/-- 
Given two groups of cows with their respective milk production rates,
this theorem proves the total milk production for both groups over a specified period.
-/
theorem total_milk_production 
  (a b c x y z w : ℝ) 
  (ha : a > 0) 
  (hb : b ≥ 0) 
  (hc : c > 0) 
  (hx : x > 0) 
  (hy : y ≥ 0) 
  (hz : z > 0) 
  (hw : w ≥ 0) :
  let group_a_rate := b / c
  let group_b_rate := y / z
  (group_a_rate + group_b_rate) * w = b * w / c + y * w / z := by
  sorry

#check total_milk_production

end total_milk_production_l3674_367427


namespace system_of_equations_solution_l3674_367472

theorem system_of_equations_solution :
  ∃! (x y z : ℚ),
    x + 2 * y - z = 20 ∧
    y = 5 ∧
    3 * x + 4 * z = 40 ∧
    x = 80 / 7 ∧
    z = 10 / 7 := by
  sorry

end system_of_equations_solution_l3674_367472


namespace unique_divisor_l3674_367495

def is_valid_divisor (d : ℕ) : Prop :=
  ∃ (sequence : Finset ℕ),
    (sequence.card = 8) ∧
    (∀ n ∈ sequence, 29 ≤ n ∧ n ≤ 119) ∧
    (∀ n ∈ sequence, n % d = 0)

theorem unique_divisor :
  ∃! d : ℕ, is_valid_divisor d ∧ d = 13 := by sorry

end unique_divisor_l3674_367495


namespace new_average_age_l3674_367462

/-- Calculates the new average age of a group after new members join -/
theorem new_average_age
  (initial_count : ℕ)
  (initial_avg_age : ℚ)
  (new_count : ℕ)
  (new_avg_age : ℚ)
  (h1 : initial_count = 20)
  (h2 : initial_avg_age = 16)
  (h3 : new_count = 20)
  (h4 : new_avg_age = 15) :
  let total_initial_age := initial_count * initial_avg_age
  let total_new_age := new_count * new_avg_age
  let total_count := initial_count + new_count
  let new_avg := (total_initial_age + total_new_age) / total_count
  new_avg = 15.5 := by
  sorry

end new_average_age_l3674_367462


namespace max_cells_visitable_l3674_367466

/-- Represents a rectangular board -/
structure Board where
  rows : Nat
  cols : Nat

/-- Represents a cube with one painted face -/
structure Cube where
  side : Nat
  painted_face : Nat

/-- Defines the maximum number of cells a cube can visit on a board without the painted face touching -/
def max_visitable_cells (b : Board) (c : Cube) : Nat :=
  b.rows * b.cols

/-- Theorem stating that the maximum number of visitable cells equals the total number of cells on the board -/
theorem max_cells_visitable (b : Board) (c : Cube) 
  (h1 : b.rows = 7) 
  (h2 : b.cols = 12) 
  (h3 : c.side = 1) 
  (h4 : c.painted_face ≤ 6) :
  max_visitable_cells b c = b.rows * b.cols := by
  sorry

end max_cells_visitable_l3674_367466


namespace crate_height_difference_l3674_367413

/-- The number of cans in each crate -/
def num_cans : ℕ := 300

/-- The diameter of each can in cm -/
def can_diameter : ℕ := 12

/-- The number of rows in triangular stacking -/
def triangular_rows : ℕ := 24

/-- The number of rows in square stacking -/
def square_rows : ℕ := 18

/-- The height of the triangular stacking in cm -/
def triangular_height : ℕ := triangular_rows * can_diameter

/-- The height of the square stacking in cm -/
def square_height : ℕ := square_rows * can_diameter

theorem crate_height_difference :
  triangular_height - square_height = 72 :=
sorry

end crate_height_difference_l3674_367413


namespace three_sum_exists_l3674_367465

theorem three_sum_exists (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h_strict_increasing : ∀ i j : Fin (n + 1), i < j → a i < a j)
  (h_upper_bound : ∀ i : Fin (n + 1), a i < 2 * n) :
  ∃ i j k : Fin (n + 1), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i + a j = a k :=
by sorry

end three_sum_exists_l3674_367465


namespace olive_oil_price_increase_l3674_367432

def highest_price : ℝ := 24
def lowest_price : ℝ := 16

theorem olive_oil_price_increase :
  (highest_price - lowest_price) / lowest_price * 100 = 50 := by
  sorry

end olive_oil_price_increase_l3674_367432


namespace or_not_implies_q_l3674_367487

theorem or_not_implies_q (p q : Prop) : (p ∨ q) → ¬p → q := by
  sorry

end or_not_implies_q_l3674_367487


namespace triangle_angle_and_vector_properties_l3674_367486

theorem triangle_angle_and_vector_properties 
  (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (m : ℝ × ℝ)
  (h_m : m = (Real.tan A + Real.tan B, Real.sqrt 3))
  (n : ℝ × ℝ)
  (h_n : n = (1, 1 - Real.tan A * Real.tan B))
  (h_perp : m.1 * n.1 + m.2 * n.2 = 0)
  (a : ℝ × ℝ)
  (h_a : a = (Real.sqrt 2 * Real.cos ((A + B) / 2), Real.sin ((A - B) / 2)))
  (h_norm_a : a.1^2 + a.2^2 = 3/2) : 
  C = Real.pi / 3 ∧ Real.tan A * Real.tan B = 1/3 := by
  sorry

end triangle_angle_and_vector_properties_l3674_367486


namespace sin_inequality_solution_set_l3674_367455

theorem sin_inequality_solution_set (a : ℝ) (θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (h3 : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2*n - 1)*π - θ < x ∧ x < 2*n*π + θ} = {x : ℝ | Real.sin x < a} :=
by sorry

end sin_inequality_solution_set_l3674_367455


namespace cars_return_to_start_l3674_367478

/-- Represents the state of cars on a circular track -/
def TrackState (n : ℕ) := Fin n → Fin n

/-- The permutation of car positions after one hour -/
def hourlyPermutation (n : ℕ) : TrackState n → TrackState n := sorry

/-- Theorem: There exists a time when all cars return to their original positions -/
theorem cars_return_to_start (n : ℕ) : 
  ∃ d : ℕ+, ∀ initial : TrackState n, (hourlyPermutation n)^[d] initial = initial := by
  sorry


end cars_return_to_start_l3674_367478


namespace total_trees_planted_l3674_367428

def trees_planted (fourth_grade fifth_grade sixth_grade : ℕ) : Prop :=
  fourth_grade = 30 ∧
  fifth_grade = 2 * fourth_grade ∧
  sixth_grade = 3 * fifth_grade - 30

theorem total_trees_planted :
  ∀ fourth_grade fifth_grade sixth_grade : ℕ,
  trees_planted fourth_grade fifth_grade sixth_grade →
  fourth_grade + fifth_grade + sixth_grade = 240 :=
by sorry

end total_trees_planted_l3674_367428


namespace geometric_sequence_formula_l3674_367421

/-- Given a positive geometric sequence {a_n} with a_2 = 2 and 2a_3 + a_4 = 16,
    prove that the general term formula is a_n = 2^(n-1) -/
theorem geometric_sequence_formula (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_a2 : a 2 = 2)
  (h_sum : 2 * a 3 + a 4 = 16) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end geometric_sequence_formula_l3674_367421


namespace water_cooler_problem_l3674_367429

/-- Represents the problem of calculating remaining water in coolers after filling cups for a meeting --/
theorem water_cooler_problem (gallons_per_ounce : ℚ) 
  (first_cooler_gallons second_cooler_gallons : ℚ)
  (small_cup_ounces large_cup_ounces : ℚ)
  (rows chairs_per_row : ℕ) :
  first_cooler_gallons = 4.5 →
  second_cooler_gallons = 3.25 →
  small_cup_ounces = 4 →
  large_cup_ounces = 8 →
  rows = 7 →
  chairs_per_row = 12 →
  gallons_per_ounce = 1 / 128 →
  (first_cooler_gallons / gallons_per_ounce) - 
    (↑(rows * chairs_per_row) * small_cup_ounces) = 240 :=
by sorry

end water_cooler_problem_l3674_367429


namespace cost_per_page_is_five_l3674_367403

/-- Calculates the cost per page in cents -/
def cost_per_page (notebooks : ℕ) (pages_per_notebook : ℕ) (total_cost_dollars : ℕ) : ℚ :=
  (total_cost_dollars * 100) / (notebooks * pages_per_notebook)

/-- Proves that the cost per page is 5 cents given the problem conditions -/
theorem cost_per_page_is_five :
  cost_per_page 2 50 5 = 5 := by
  sorry

end cost_per_page_is_five_l3674_367403


namespace base_8_243_equals_163_l3674_367475

def base_8_to_10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

theorem base_8_243_equals_163 :
  base_8_to_10 2 4 3 = 163 := by
  sorry

end base_8_243_equals_163_l3674_367475


namespace trigonometric_calculation_and_algebraic_simplification_l3674_367411

theorem trigonometric_calculation_and_algebraic_simplification :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (30 * π / 180) + |(-1/2)| = 1) ∧
  (let a := 2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180)
   let b := 3
   1 - (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) = -Real.sqrt 3) := by
  sorry

end trigonometric_calculation_and_algebraic_simplification_l3674_367411


namespace shifted_roots_polynomial_l3674_367481

theorem shifted_roots_polynomial (r₁ r₂ : ℝ) (h_sum : r₁ + r₂ = 15) (h_prod : r₁ * r₂ = 36) :
  (X - (r₁ + 3)) * (X - (r₂ + 3)) = X^2 - 21*X + 90 :=
by sorry

end shifted_roots_polynomial_l3674_367481


namespace fraction_problem_l3674_367425

theorem fraction_problem (x y : ℚ) : 
  y / (x - 1) = 1 / 3 → (y + 4) / x = 1 / 2 → y / x = 7 / 22 := by
  sorry

end fraction_problem_l3674_367425


namespace motorist_gas_plan_l3674_367497

/-- The number of gallons a motorist initially planned to buy given certain conditions -/
theorem motorist_gas_plan (actual_price expected_price_difference affordable_gallons : ℚ) :
  actual_price = 150 ∧ 
  expected_price_difference = 30 ∧ 
  affordable_gallons = 10 →
  (actual_price * affordable_gallons) / (actual_price - expected_price_difference) = 25/2 :=
by sorry

end motorist_gas_plan_l3674_367497


namespace solution_exists_l3674_367476

theorem solution_exists : ∃ M : ℤ, (14 : ℤ)^2 * (35 : ℤ)^2 = (10 : ℤ)^2 * (M - 10)^2 := by
  use 59
  sorry

#check solution_exists

end solution_exists_l3674_367476


namespace division_equality_not_always_true_l3674_367458

theorem division_equality_not_always_true (x y m : ℝ) :
  ¬(∀ x y m : ℝ, x = y → x / m = y / m) :=
sorry

end division_equality_not_always_true_l3674_367458


namespace triangle_angle_measure_l3674_367422

/-- Given a triangle ABD where angle ABC is a straight angle (180°),
    angle CBD is 133°, and one angle in triangle ABD is 31°,
    prove that the measure of the remaining angle y in triangle ABD is 102°. -/
theorem triangle_angle_measure (angle_CBD : ℝ) (angle_in_ABD : ℝ) :
  angle_CBD = 133 →
  angle_in_ABD = 31 →
  let angle_ABD : ℝ := 180 - angle_CBD
  180 - (angle_ABD + angle_in_ABD) = 102 := by
  sorry

end triangle_angle_measure_l3674_367422


namespace line_tangent_to_circle_l3674_367467

/-- A line mx-y+2=0 is tangent to the circle x^2+y^2=1 if and only if m = ± √3 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∃ (x y : ℝ), m*x - y + 2 = 0 ∧ x^2 + y^2 = 1 ∧ 
   ∀ (x' y' : ℝ), m*x' - y' + 2 = 0 → x'^2 + y'^2 ≥ 1) ↔ 
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 :=
sorry

end line_tangent_to_circle_l3674_367467


namespace min_correct_answers_to_win_l3674_367447

/-- Represents the scoring system and conditions of the quiz -/
structure QuizRules where
  total_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℕ
  unanswered : ℕ
  min_score_to_win : ℕ

/-- Calculates the score based on the number of correct answers -/
def calculate_score (rules : QuizRules) (correct_answers : ℕ) : ℤ :=
  (correct_answers : ℤ) * rules.correct_points -
  (rules.total_questions - rules.unanswered - correct_answers : ℤ) * rules.incorrect_points

/-- Theorem stating the minimum number of correct answers needed to win -/
theorem min_correct_answers_to_win (rules : QuizRules)
  (h1 : rules.total_questions = 25)
  (h2 : rules.correct_points = 4)
  (h3 : rules.incorrect_points = 2)
  (h4 : rules.unanswered = 2)
  (h5 : rules.min_score_to_win = 80) :
  ∀ x : ℕ, x ≥ 22 ↔ calculate_score rules x > rules.min_score_to_win :=
sorry

end min_correct_answers_to_win_l3674_367447


namespace derivative_h_at_one_l3674_367439

-- Define a function f
variable (f : ℝ → ℝ)

-- Define g(x) = f(x) - f(2x)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - f (2 * x)

-- Define h(x) = f(x) - f(4x)
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - f (4 * x)

-- State the theorem
theorem derivative_h_at_one (f : ℝ → ℝ) 
  (hg1 : deriv (g f) 1 = 5)
  (hg2 : deriv (g f) 2 = 7) :
  deriv (h f) 1 = 19 := by
  sorry

end derivative_h_at_one_l3674_367439


namespace complement_M_intersect_N_l3674_367453

open Set

-- Define the universal set I as the set of real numbers
def I : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | x < 1}

-- Theorem statement
theorem complement_M_intersect_N :
  (I \ M) ∩ N = {x : ℝ | x < -2} := by
  sorry

end complement_M_intersect_N_l3674_367453


namespace probability_shaded_is_half_l3674_367434

/-- Represents a triangle in the diagram -/
structure Triangle where
  is_shaded : Bool

/-- The diagram containing the triangles -/
structure Diagram where
  triangles : Finset Triangle

/-- Calculates the probability of selecting a shaded triangle -/
def probability_shaded (d : Diagram) : ℚ :=
  (d.triangles.filter (·.is_shaded)).card / d.triangles.card

/-- The theorem statement -/
theorem probability_shaded_is_half (d : Diagram) :
    d.triangles.card = 4 ∧ 
    (d.triangles.filter (·.is_shaded)).card > 0 →
    probability_shaded d = 1/2 := by
  sorry

end probability_shaded_is_half_l3674_367434


namespace max_five_sunday_months_correct_five_is_max_l3674_367479

/-- Represents a year, which can be either common (365 days) or leap (366 days) -/
inductive Year
| Common
| Leap

/-- Represents a month in a year -/
structure Month where
  days : Nat
  h1 : days ≥ 28
  h2 : days ≤ 31

/-- The number of Sundays in a month -/
def sundays (m : Month) : Nat :=
  if m.days ≥ 35 then 5 else 4

/-- The maximum number of months with 5 Sundays in a year -/
def max_five_sunday_months (y : Year) : Nat :=
  match y with
  | Year.Common => 4
  | Year.Leap => 5

theorem max_five_sunday_months_correct (y : Year) :
  max_five_sunday_months y = 
    match y with
    | Year.Common => 4
    | Year.Leap => 5 :=
by
  sorry

theorem five_is_max (y : Year) :
  max_five_sunday_months y ≤ 5 :=
by
  sorry

end max_five_sunday_months_correct_five_is_max_l3674_367479


namespace shannon_bracelets_l3674_367405

/-- Given Shannon has 48 heart-shaped stones and each bracelet requires 8 stones,
    prove that she can make 6 bracelets. -/
theorem shannon_bracelets :
  let total_stones : ℕ := 48
  let stones_per_bracelet : ℕ := 8
  let num_bracelets : ℕ := total_stones / stones_per_bracelet
  num_bracelets = 6 := by
sorry

end shannon_bracelets_l3674_367405


namespace line_plane_relationships_l3674_367408

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the relationships between planes
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the relationship between lines
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the given conditions
variable (l m : Line) (α β : Plane)
variable (h1 : perpendicular l α)
variable (h2 : parallel m β)

-- State the theorem
theorem line_plane_relationships :
  (plane_parallel α β → line_perpendicular l m) ∧
  (line_parallel l m → plane_perpendicular α β) :=
sorry

end line_plane_relationships_l3674_367408


namespace solve_for_x_l3674_367491

theorem solve_for_x (x y : ℚ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 := by
  sorry

end solve_for_x_l3674_367491


namespace dolphin_training_hours_l3674_367498

theorem dolphin_training_hours 
  (num_dolphins : ℕ) 
  (hours_per_dolphin : ℕ) 
  (num_trainers : ℕ) 
  (h1 : num_dolphins = 4)
  (h2 : hours_per_dolphin = 3)
  (h3 : num_trainers = 2)
  : (num_dolphins * hours_per_dolphin) / num_trainers = 6 := by
  sorry

end dolphin_training_hours_l3674_367498


namespace meeting_handshakes_l3674_367415

/-- The number of people in the meeting -/
def total_people : ℕ := 40

/-- The number of people who know each other -/
def group_a : ℕ := 25

/-- The number of people who don't know anyone -/
def group_b : ℕ := 15

/-- Calculate the number of handshakes between two groups -/
def inter_group_handshakes (g1 g2 : ℕ) : ℕ := g1 * g2

/-- Calculate the number of handshakes within a group where no one knows each other -/
def intra_group_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of handshakes in the meeting -/
def total_handshakes : ℕ := 
  inter_group_handshakes group_a group_b + intra_group_handshakes group_b

theorem meeting_handshakes : 
  total_people = group_a + group_b → total_handshakes = 480 := by
  sorry

end meeting_handshakes_l3674_367415


namespace evaluate_nested_expression_l3674_367473

def f (x : ℕ) : ℕ := 3 * (3 * (3 * (3 * (3 * x + 2) + 2) + 2) + 2) + 2

theorem evaluate_nested_expression :
  f 5 = 1457 := by
  sorry

end evaluate_nested_expression_l3674_367473


namespace odd_function_with_period_4_l3674_367483

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_with_period_4 (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 4) 
  (h_min_period : ∀ p, 0 < p → p < 4 → ¬ has_period f p) : 
  f 2 = 0 := by
sorry

end odd_function_with_period_4_l3674_367483


namespace binomial_coefficient_two_l3674_367442

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n.val 2 = n.val * (n.val - 1) / 2 := by
  sorry

end binomial_coefficient_two_l3674_367442


namespace min_sum_squares_min_sum_squares_equality_condition_l3674_367470

theorem min_sum_squares (a b c d : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) 
  (sum_eq : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 ≥ 1990 := by
sorry

theorem min_sum_squares_equality_condition (a b c d : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) 
  (sum_eq : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 = 1990 ↔ a = b ∧ b = c ∧ c = d := by
sorry

end min_sum_squares_min_sum_squares_equality_condition_l3674_367470


namespace smallest_integers_difference_smallest_integers_difference_is_27720_l3674_367489

theorem smallest_integers_difference : ℕ → Prop :=
  fun d =>
    ∃ n₁ n₂ : ℕ,
      n₁ > 1 ∧ n₂ > 1 ∧
      n₁ < n₂ ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₁ % k = 1) ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₂ % k = 1) ∧
      (∀ m : ℕ, m > 1 → m < n₂ → m ≠ n₁ → ∃ k : ℕ, 2 ≤ k ∧ k ≤ 11 ∧ m % k ≠ 1) ∧
      d = n₂ - n₁

theorem smallest_integers_difference_is_27720 : smallest_integers_difference 27720 := by
  sorry

end smallest_integers_difference_smallest_integers_difference_is_27720_l3674_367489


namespace complex_modulus_problem_l3674_367404

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l3674_367404


namespace block_count_is_eight_l3674_367431

/-- Represents the orthographic views of a geometric body -/
structure OrthographicViews where
  front : Nat
  top : Nat
  side : Nat

/-- Calculates the number of blocks in a geometric body based on its orthographic views -/
def countBlocks (views : OrthographicViews) : Nat :=
  sorry

/-- The specific orthographic views for the given problem -/
def problemViews : OrthographicViews :=
  { front := 6, top := 6, side := 4 }

/-- Theorem stating that the number of blocks for the given views is 8 -/
theorem block_count_is_eight :
  countBlocks problemViews = 8 := by
  sorry

end block_count_is_eight_l3674_367431


namespace no_multiples_of_2310_in_power_difference_form_l3674_367494

theorem no_multiples_of_2310_in_power_difference_form :
  ¬ ∃ (k i j : ℕ), 
    0 ≤ i ∧ i < j ∧ j ≤ 50 ∧ 
    k * 2310 = 2^j - 2^i ∧ 
    k > 0 :=
by sorry

end no_multiples_of_2310_in_power_difference_form_l3674_367494


namespace pirate_treasure_probability_l3674_367416

def num_islands : ℕ := 7
def num_treasure_islands : ℕ := 4

def prob_treasure : ℚ := 1/5
def prob_trap : ℚ := 1/10
def prob_neither : ℚ := 7/10

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  12005/625000 := by sorry

end pirate_treasure_probability_l3674_367416


namespace correct_arrangement_l3674_367448

-- Define the set of friends
inductive Friend : Type
  | Amy : Friend
  | Bob : Friend
  | Celine : Friend
  | David : Friend

-- Define a height comparison relation
def taller_than : Friend → Friend → Prop := sorry

-- Define the statements
def statement_I : Prop := ¬(taller_than Friend.Celine Friend.Amy ∧ taller_than Friend.Celine Friend.Bob ∧ taller_than Friend.Celine Friend.David)
def statement_II : Prop := ∀ f : Friend, f ≠ Friend.Bob → taller_than f Friend.Bob
def statement_III : Prop := ∃ f₁ f₂ : Friend, taller_than f₁ Friend.Amy ∧ taller_than Friend.Amy f₂
def statement_IV : Prop := taller_than Friend.David Friend.Bob ∧ taller_than Friend.Amy Friend.David

-- Define the condition that exactly one statement is true
def exactly_one_true : Prop :=
  (statement_I ∧ ¬statement_II ∧ ¬statement_III ∧ ¬statement_IV) ∨
  (¬statement_I ∧ statement_II ∧ ¬statement_III ∧ ¬statement_IV) ∨
  (¬statement_I ∧ ¬statement_II ∧ statement_III ∧ ¬statement_IV) ∨
  (¬statement_I ∧ ¬statement_II ∧ ¬statement_III ∧ statement_IV)

-- Theorem to prove
theorem correct_arrangement (h : exactly_one_true) :
  taller_than Friend.Celine Friend.Amy ∧
  taller_than Friend.Amy Friend.David ∧
  taller_than Friend.David Friend.Bob :=
sorry

end correct_arrangement_l3674_367448


namespace sin_75_165_minus_sin_15_105_eq_zero_l3674_367454

theorem sin_75_165_minus_sin_15_105_eq_zero :
  Real.sin (75 * π / 180) * Real.sin (165 * π / 180) -
  Real.sin (15 * π / 180) * Real.sin (105 * π / 180) = 0 := by
  sorry

end sin_75_165_minus_sin_15_105_eq_zero_l3674_367454


namespace median_length_inequality_l3674_367493

theorem median_length_inequality (a b c s_a : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Triangle sides are positive
  a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
  s_a > 0 ∧  -- Median length is positive
  s_a^2 = (b^2 + c^2) / 4 - a^2 / 16  -- Median length formula
  →
  s_a < (b + c) / 2 := by
sorry

end median_length_inequality_l3674_367493


namespace event_A_not_random_l3674_367430

-- Define the type for events
inductive Event
| A : Event  -- The sun rises in the east and it rains in the west
| B : Event  -- It's not cold when it snows but cold when the snow melts
| C : Event  -- It often rains during the Qingming festival
| D : Event  -- It's sunny every day when the plums turn yellow

-- Define what it means for an event to be random
def isRandomEvent (e : Event) : Prop := sorry

-- Define what it means for an event to be based on natural laws
def isBasedOnNaturalLaws (e : Event) : Prop := sorry

-- Axiom: Events based on natural laws are not random
axiom natural_law_not_random : ∀ (e : Event), isBasedOnNaturalLaws e → ¬isRandomEvent e

-- Theorem: Event A is not a random event
theorem event_A_not_random : ¬isRandomEvent Event.A := by
  sorry

end event_A_not_random_l3674_367430


namespace percentage_of_boys_studying_science_l3674_367423

theorem percentage_of_boys_studying_science 
  (total_boys : ℕ) 
  (school_A_percentage : ℚ) 
  (non_science_boys : ℕ) 
  (h1 : total_boys = 300)
  (h2 : school_A_percentage = 1/5)
  (h3 : non_science_boys = 42) :
  (↑((school_A_percentage * ↑total_boys - ↑non_science_boys) / (school_A_percentage * ↑total_boys)) : ℚ) = 3/10 :=
sorry

end percentage_of_boys_studying_science_l3674_367423


namespace smallest_multiple_l3674_367410

theorem smallest_multiple (n : ℕ) : n = 3441 ↔ 
  n > 0 ∧ 
  37 ∣ n ∧ 
  n % 103 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 37 ∣ m → m % 103 = 7 → n ≤ m :=
by sorry

end smallest_multiple_l3674_367410


namespace simplify_expression_l3674_367417

theorem simplify_expression (x w : ℝ) : 3*x + 4*w - 2*x + 6 - 5*w - 5 = x - w + 1 := by
  sorry

end simplify_expression_l3674_367417


namespace hyperbola_focus_coordinates_l3674_367420

/-- Given a hyperbola with equation (x-5)^2/7^2 - (y-10)^2/15^2 = 1, 
    the focus with the larger x-coordinate has coordinates (5 + √274, 10) -/
theorem hyperbola_focus_coordinates (x y : ℝ) : 
  ((x - 5)^2 / 7^2) - ((y - 10)^2 / 15^2) = 1 →
  ∃ (f_x f_y : ℝ), f_x > 5 ∧ f_y = 10 ∧ 
  f_x = 5 + Real.sqrt 274 ∧
  ((f_x - 5)^2 / 7^2) - ((f_y - 10)^2 / 15^2) = 1 ∧
  ∀ (x' y' : ℝ), x' > 5 ∧ 
    ((x' - 5)^2 / 7^2) - ((y' - 10)^2 / 15^2) = 1 →
    x' ≤ f_x :=
by sorry

end hyperbola_focus_coordinates_l3674_367420


namespace james_bills_denomination_l3674_367438

/-- Proves that the denomination of each bill James found is $20 -/
theorem james_bills_denomination (initial_amount : ℕ) (final_amount : ℕ) (num_bills : ℕ) :
  initial_amount = 75 →
  final_amount = 135 →
  num_bills = 3 →
  (final_amount - initial_amount) / num_bills = 20 :=
by sorry

end james_bills_denomination_l3674_367438


namespace prob_connected_formula_l3674_367407

/-- The number of vertices in the graph -/
def n : ℕ := 20

/-- The number of edges removed -/
def k : ℕ := 35

/-- The total number of edges in a complete graph with n vertices -/
def total_edges : ℕ := n * (n - 1) / 2

/-- The probability that the graph remains connected after removing k edges -/
def prob_connected : ℚ :=
  1 - (n : ℚ) * (Nat.choose (total_edges - n + 1) (k - n + 1) : ℚ) / (Nat.choose total_edges k : ℚ)

theorem prob_connected_formula :
  prob_connected = 1 - (20 : ℚ) * (Nat.choose 171 16 : ℚ) / (Nat.choose 190 35 : ℚ) :=
sorry

end prob_connected_formula_l3674_367407


namespace inequality1_sufficient_not_necessary_l3674_367456

-- Define the two inequalities
def inequality1 (x : ℝ) : Prop := 1 + 3 / (x - 1) ≥ 0
def inequality2 (x : ℝ) : Prop := (x + 2) * (x - 1) ≥ 0

-- Theorem statement
theorem inequality1_sufficient_not_necessary :
  (∀ x, inequality1 x → inequality2 x) ∧
  (∃ x, inequality2 x ∧ ¬inequality1 x) := by sorry

end inequality1_sufficient_not_necessary_l3674_367456


namespace circle_equation_from_diameter_find_circle_parameter_l3674_367464

-- Part 1
theorem circle_equation_from_diameter (P₁ P₂ : ℝ × ℝ) (h : P₁ = (4, 9) ∧ P₂ = (6, 3)) :
  ∃ C : ℝ × ℝ, ∃ r : ℝ, ∀ x y : ℝ,
    (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ (x - 5)^2 + (y - 6)^2 = 10 :=
sorry

-- Part 2
theorem find_circle_parameter (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x - y + 3 = 0 ∧ (x - a)^2 + (y - 2)^2 = 4) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ - y₁ + 3 = 0 ∧ x₂ - y₂ + 3 = 0 ∧
    (x₁ - a)^2 + (y₁ - 2)^2 = 4 ∧ (x₂ - a)^2 + (y₂ - 2)^2 = 4 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  a = 1 :=
sorry

end circle_equation_from_diameter_find_circle_parameter_l3674_367464


namespace pauline_bell_peppers_l3674_367419

/-- The number of bell peppers Pauline bought -/
def num_bell_peppers : ℕ := 4

/-- The cost of taco shells in dollars -/
def taco_shells_cost : ℚ := 5

/-- The cost of each bell pepper in dollars -/
def bell_pepper_cost : ℚ := 3/2

/-- The cost of meat per pound in dollars -/
def meat_cost_per_pound : ℚ := 3

/-- The amount of meat Pauline bought in pounds -/
def meat_amount : ℚ := 2

/-- The total amount Pauline spent in dollars -/
def total_spent : ℚ := 17

theorem pauline_bell_peppers :
  num_bell_peppers = (total_spent - (taco_shells_cost + meat_cost_per_pound * meat_amount)) / bell_pepper_cost := by
  sorry

end pauline_bell_peppers_l3674_367419


namespace distinct_values_count_l3674_367499

def odd_integers_less_than_15 : Finset ℕ :=
  {1, 3, 5, 7, 9, 11, 13}

def expression (p q : ℕ) : ℤ :=
  p * q - (p + q)

theorem distinct_values_count :
  Finset.card (Finset.image₂ expression odd_integers_less_than_15 odd_integers_less_than_15) = 28 :=
by sorry

end distinct_values_count_l3674_367499


namespace original_equals_scientific_l3674_367460

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 4370000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { significand := 4.37
    exponent := 6
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.significand * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end original_equals_scientific_l3674_367460


namespace truck_capacity_problem_l3674_367450

/-- The capacity of a large truck in tons -/
def large_truck_capacity : ℝ := sorry

/-- The capacity of a small truck in tons -/
def small_truck_capacity : ℝ := sorry

/-- The total capacity of a given number of large and small trucks -/
def total_capacity (large_trucks small_trucks : ℕ) : ℝ :=
  (large_trucks : ℝ) * large_truck_capacity + (small_trucks : ℝ) * small_truck_capacity

theorem truck_capacity_problem :
  total_capacity 3 4 = 22 ∧ total_capacity 5 2 = 25 →
  total_capacity 4 3 = 23.5 := by
  sorry

end truck_capacity_problem_l3674_367450


namespace train_length_calculation_l3674_367444

/-- Calculates the length of a train given its speed and time to cross a pole. -/
theorem train_length_calculation (speed_km_hr : ℝ) (time_seconds : ℝ) : 
  speed_km_hr = 30 → time_seconds = 12 → 
  ∃ (length_meters : ℝ), 
    (abs (length_meters - 100) < 1) ∧ 
    (length_meters = speed_km_hr * (1000 / 3600) * time_seconds) := by
  sorry

end train_length_calculation_l3674_367444


namespace fraction_simplification_l3674_367449

theorem fraction_simplification (x : ℝ) : (3*x - 2)/4 + (5 - 2*x)/3 = (x + 14)/12 := by
  sorry

end fraction_simplification_l3674_367449


namespace distance_to_complex_point_l3674_367482

theorem distance_to_complex_point : ∃ (z : ℂ), z = 3 / (2 - Complex.I)^2 ∧ Complex.abs z = 3 / 5 := by
  sorry

end distance_to_complex_point_l3674_367482


namespace triangle_inequality_l3674_367400

theorem triangle_inequality (a b c S : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : S > 0) (h₅ : a + b > c) (h₆ : b + c > a) (h₇ : c + a > b)
  (h₈ : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  c^2 - a^2 - b^2 + 4*a*b ≥ 4 * Real.sqrt 3 * S := by
  sorry

end triangle_inequality_l3674_367400


namespace tangent_line_slope_l3674_367433

/-- If the line y = kx is tangent to the curve y = x + exp(-x), then k = 1 - exp(1) -/
theorem tangent_line_slope (k : ℝ) : 
  (∃ x₀ : ℝ, k * x₀ = x₀ + Real.exp (-x₀) ∧ 
             k = 1 - Real.exp (-x₀)) → 
  k = 1 - Real.exp 1 := by
sorry

end tangent_line_slope_l3674_367433


namespace set_operations_and_subset_l3674_367461

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 6}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x : ℝ | 2 < x ∧ x < 10}) ∧
  (∀ a : ℝ, A ⊆ C a ↔ 2 ≤ a ∧ a < 3) :=
by sorry

end set_operations_and_subset_l3674_367461


namespace license_plate_count_l3674_367409

/-- The number of consonants in the alphabet, including Y -/
def num_consonants : ℕ := 21

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants^2 * num_vowels^2 * num_digits^2

theorem license_plate_count : total_license_plates = 1102500 := by
  sorry

end license_plate_count_l3674_367409
