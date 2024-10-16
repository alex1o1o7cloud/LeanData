import Mathlib

namespace NUMINAMATH_CALUDE_construction_delay_l747_74708

/-- Represents the efficiency of a group of workers -/
structure WorkerGroup where
  count : ℕ
  efficiency : ℕ
  startDay : ℕ

/-- Calculates the total work units completed by a worker group -/
def totalWorkUnits (wg : WorkerGroup) (totalDays : ℕ) : ℕ :=
  wg.count * wg.efficiency * (totalDays - wg.startDay)

/-- Theorem: The construction would be 244 days behind schedule without additional workers -/
theorem construction_delay :
  let totalDays : ℕ := 150
  let initialWorkers : WorkerGroup := ⟨100, 1, 0⟩
  let additionalWorkers : List WorkerGroup := [
    ⟨50, 2, 30⟩,
    ⟨25, 3, 45⟩,
    ⟨15, 4, 75⟩
  ]
  let additionalWorkUnits : ℕ := (additionalWorkers.map (totalWorkUnits · totalDays)).sum
  let daysWithoutAdditional : ℕ := totalDays + (additionalWorkUnits / initialWorkers.count / initialWorkers.efficiency)
  daysWithoutAdditional - totalDays = 244 := by
    sorry

end NUMINAMATH_CALUDE_construction_delay_l747_74708


namespace NUMINAMATH_CALUDE_no_cube_sum_4099_l747_74763

theorem no_cube_sum_4099 : 
  ∀ a b : ℤ, a^3 + b^3 ≠ 4099 :=
by
  sorry

#check no_cube_sum_4099

end NUMINAMATH_CALUDE_no_cube_sum_4099_l747_74763


namespace NUMINAMATH_CALUDE_rectangle_area_l747_74732

theorem rectangle_area (a b : ℝ) (h1 : a = 10) (h2 : 2*a + 2*b = 40) : a * b = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l747_74732


namespace NUMINAMATH_CALUDE_matrix_operation_proof_l747_74768

theorem matrix_operation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -3; 0, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-7, 9; 6, -10]
  2 • A + B = !![1, 3; 6, 0] := by
  sorry

end NUMINAMATH_CALUDE_matrix_operation_proof_l747_74768


namespace NUMINAMATH_CALUDE_min_sum_squares_l747_74799

theorem min_sum_squares (a b t : ℝ) (h : a + b = t) :
  ∃ (min : ℝ), min = t^2 / 2 ∧ ∀ (x y : ℝ), x + y = t → x^2 + y^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l747_74799


namespace NUMINAMATH_CALUDE_solve_systems_of_equations_l747_74719

theorem solve_systems_of_equations :
  -- First system
  (∃ x y : ℝ, 3 * x - y = 8 ∧ 3 * x - 5 * y = -20 → x = 5 ∧ y = 7) ∧
  -- Second system
  (∃ x y : ℝ, x / 3 - y / 2 = -1 ∧ 3 * x - 2 * y = 1 → x = 3 ∧ y = 4) :=
by sorry


end NUMINAMATH_CALUDE_solve_systems_of_equations_l747_74719


namespace NUMINAMATH_CALUDE_sum_PV_squared_l747_74786

-- Define the triangle PQR
def PQR : Set (ℝ × ℝ) := sorry

-- Define the property of PQR being equilateral with side length 10
def is_equilateral_10 (triangle : Set (ℝ × ℝ)) : Prop := sorry

-- Define the four triangles PU₁V₁, PU₁V₂, PU₂V₃, and PU₂V₄
def PU1V1 : Set (ℝ × ℝ) := sorry
def PU1V2 : Set (ℝ × ℝ) := sorry
def PU2V3 : Set (ℝ × ℝ) := sorry
def PU2V4 : Set (ℝ × ℝ) := sorry

-- Define the property of a triangle being congruent to PQR
def is_congruent_to_PQR (triangle : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property QU₁ = QU₂ = 3
def QU1_QU2_eq_3 : Prop := sorry

-- Define the function to calculate PVₖ
def PV (k : ℕ) : ℝ := sorry

-- Theorem statement
theorem sum_PV_squared :
  is_equilateral_10 PQR ∧
  is_congruent_to_PQR PU1V1 ∧
  is_congruent_to_PQR PU1V2 ∧
  is_congruent_to_PQR PU2V3 ∧
  is_congruent_to_PQR PU2V4 ∧
  QU1_QU2_eq_3 →
  (PV 1)^2 + (PV 2)^2 + (PV 3)^2 + (PV 4)^2 = 800 := by sorry

end NUMINAMATH_CALUDE_sum_PV_squared_l747_74786


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l747_74728

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families * total_average) / (total_families - childless_families) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l747_74728


namespace NUMINAMATH_CALUDE_jeans_bought_l747_74769

/-- Given a clothing sale with specific prices and quantities, prove the number of jeans bought. -/
theorem jeans_bought (shirt_price hat_price jeans_price total_cost : ℕ) 
  (shirts_bought hats_bought : ℕ) : 
  shirt_price = 5 →
  hat_price = 4 →
  jeans_price = 10 →
  total_cost = 51 →
  shirts_bought = 3 →
  hats_bought = 4 →
  ∃ (jeans_bought : ℕ), 
    jeans_bought = 2 ∧ 
    total_cost = shirt_price * shirts_bought + hat_price * hats_bought + jeans_price * jeans_bought :=
by sorry

end NUMINAMATH_CALUDE_jeans_bought_l747_74769


namespace NUMINAMATH_CALUDE_speed_ratio_problem_l747_74718

/-- The ratio of speeds between two people walking in opposite directions -/
theorem speed_ratio_problem (v₁ v₂ : ℝ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) : 
  (v₁ / v₂ * 60 = v₂ / v₁ * 60 + 35) → v₁ / v₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_problem_l747_74718


namespace NUMINAMATH_CALUDE_opposite_of_pi_l747_74745

theorem opposite_of_pi : -π = -π := by sorry

end NUMINAMATH_CALUDE_opposite_of_pi_l747_74745


namespace NUMINAMATH_CALUDE_only_c_is_perfect_square_l747_74730

def option_a : ℕ := 3^3 * 4^4 * 7^7
def option_b : ℕ := 3^4 * 4^5 * 7^6
def option_c : ℕ := 3^6 * 4^6 * 7^4
def option_d : ℕ := 3^5 * 4^4 * 7^6
def option_e : ℕ := 3^6 * 4^7 * 7^5

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem only_c_is_perfect_square : 
  is_perfect_square option_c ∧ 
  ¬is_perfect_square option_a ∧ 
  ¬is_perfect_square option_b ∧ 
  ¬is_perfect_square option_d ∧ 
  ¬is_perfect_square option_e :=
sorry

end NUMINAMATH_CALUDE_only_c_is_perfect_square_l747_74730


namespace NUMINAMATH_CALUDE_basketball_score_proof_l747_74712

theorem basketball_score_proof (jon_score jack_score tom_score : ℕ) : 
  jon_score = 3 →
  jack_score = jon_score + 5 →
  tom_score = jon_score + jack_score - 4 →
  jon_score + jack_score + tom_score = 18 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l747_74712


namespace NUMINAMATH_CALUDE_factorization_cubic_factorization_fifth_power_l747_74797

-- We don't need to prove the first part as no specific factorization was provided

-- Prove the factorization of x^3 + 2x^2 + 4x + 3
theorem factorization_cubic (x : ℝ) : 
  x^3 + 2*x^2 + 4*x + 3 = (x + 1) * (x^2 + x + 3) := by
sorry

-- Prove the factorization of x^5 - 1
theorem factorization_fifth_power (x : ℝ) : 
  x^5 - 1 = (x - 1) * (x^4 + x^3 + x^2 + x + 1) := by
sorry

end NUMINAMATH_CALUDE_factorization_cubic_factorization_fifth_power_l747_74797


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l747_74762

theorem larger_number_of_pair (x y : ℝ) : 
  x - y = 7 → x + y = 47 → max x y = 27 := by sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l747_74762


namespace NUMINAMATH_CALUDE_min_draws_for_fifteen_l747_74710

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem min_draws_for_fifteen (counts : BallCounts) :
  counts.red = 30 ∧ counts.green = 25 ∧ counts.yellow = 23 ∧
  counts.blue = 14 ∧ counts.white = 13 ∧ counts.black = 10 →
  minDraws counts 15 = 80 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_fifteen_l747_74710


namespace NUMINAMATH_CALUDE_computer_program_output_l747_74747

theorem computer_program_output (x : ℝ) (y : ℝ) : 
  x = Real.sqrt 3 - 2 → y = Real.sqrt ((x^2).sqrt - 2) → y = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_computer_program_output_l747_74747


namespace NUMINAMATH_CALUDE_selling_price_formula_l747_74737

/-- Calculates the selling price of a refrigerator to achieve a specific profit margin -/
def calculate_selling_price (L : ℝ) : ℝ :=
  let first_discount := 0.2
  let second_discount := 0.1
  let additional_costs := 475
  let profit_margin := 0.18
  let discounted_price := L * (1 - first_discount) * (1 - second_discount)
  let total_cost := discounted_price + additional_costs
  total_cost + L * profit_margin

/-- Theorem stating the correct selling price formula -/
theorem selling_price_formula (L : ℝ) :
  calculate_selling_price L = 0.9 * L + 475 := by
  sorry

#eval calculate_selling_price 1000  -- Example calculation

end NUMINAMATH_CALUDE_selling_price_formula_l747_74737


namespace NUMINAMATH_CALUDE_range_of_a_l747_74765

open Set

/-- The equation that must have 3 distinct real solutions -/
def equation (a x : ℝ) : ℝ := 2 * x * |x| - (a - 2) * x + |x| - a + 1

/-- The condition that the equation has 3 distinct real solutions -/
def has_three_distinct_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    equation a x₁ = 0 ∧ equation a x₂ = 0 ∧ equation a x₃ = 0

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  has_three_distinct_solutions a → a ∈ Ioi 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l747_74765


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l747_74738

theorem cubic_roots_sum_of_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l747_74738


namespace NUMINAMATH_CALUDE_min_a_value_l747_74773

theorem min_a_value (x y : ℝ) (hx : x ∈ Set.Icc 1 2) (hy : y ∈ Set.Icc 4 5) :
  ∃ (a : ℝ), (∀ (x' y' : ℝ), x' ∈ Set.Icc 1 2 → y' ∈ Set.Icc 4 5 → x' * y' ≤ a * x' ^ 2 + 2 * y' ^ 2) ∧ 
  (∀ (b : ℝ), (∀ (x' y' : ℝ), x' ∈ Set.Icc 1 2 → y' ∈ Set.Icc 4 5 → x' * y' ≤ b * x' ^ 2 + 2 * y' ^ 2) → b ≥ -6) :=
by
  sorry

#check min_a_value

end NUMINAMATH_CALUDE_min_a_value_l747_74773


namespace NUMINAMATH_CALUDE_largest_value_problem_l747_74726

theorem largest_value_problem : 
  let a := 12345 + 1/5678
  let b := 12345 - 1/5678
  let c := 12345 * 1/5678
  let d := 12345 / (1/5678)
  let e := 12345.5678
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_value_problem_l747_74726


namespace NUMINAMATH_CALUDE_early_arrival_time_l747_74736

/-- 
Given a boy's usual time to reach school and his faster rate relative to his usual rate,
calculate how many minutes earlier he arrives when walking at the faster rate.
-/
theorem early_arrival_time (usual_time : ℝ) (faster_rate_ratio : ℝ) 
  (h1 : usual_time = 28)
  (h2 : faster_rate_ratio = 7/6) : 
  usual_time - (usual_time / faster_rate_ratio) = 4 := by
  sorry

end NUMINAMATH_CALUDE_early_arrival_time_l747_74736


namespace NUMINAMATH_CALUDE_power_sum_difference_l747_74725

theorem power_sum_difference : 2^1 + 1^0 - 3^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l747_74725


namespace NUMINAMATH_CALUDE_earth_capacity_theorem_l747_74790

/-- Represents the Earth's resource capacity --/
structure EarthCapacity where
  peopleA : ℕ  -- Number of people in scenario A
  yearsA : ℕ   -- Number of years in scenario A
  peopleB : ℕ  -- Number of people in scenario B
  yearsB : ℕ   -- Number of years in scenario B

/-- Calculates the maximum sustainable population given Earth's resource capacity --/
def maxSustainablePopulation (capacity : EarthCapacity) : ℕ :=
  ((capacity.peopleB * capacity.yearsB - capacity.peopleA * capacity.yearsA) / (capacity.yearsB - capacity.yearsA))

/-- Theorem stating the maximum sustainable population for given conditions --/
theorem earth_capacity_theorem (capacity : EarthCapacity) 
  (h1 : capacity.peopleA = 11)
  (h2 : capacity.yearsA = 90)
  (h3 : capacity.peopleB = 9)
  (h4 : capacity.yearsB = 210) :
  maxSustainablePopulation capacity = 75 := by
  sorry

end NUMINAMATH_CALUDE_earth_capacity_theorem_l747_74790


namespace NUMINAMATH_CALUDE_output_is_17_when_input_is_8_l747_74759

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 22 then step1 - 7 else step1 + 10

theorem output_is_17_when_input_is_8 : function_machine 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_output_is_17_when_input_is_8_l747_74759


namespace NUMINAMATH_CALUDE_students_neither_music_nor_art_l747_74715

theorem students_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 50) 
  (h3 : art = 20) 
  (h4 : both = 10) : 
  total - (music + art - both) = 440 := by
sorry

end NUMINAMATH_CALUDE_students_neither_music_nor_art_l747_74715


namespace NUMINAMATH_CALUDE_shadow_length_problem_l747_74735

theorem shadow_length_problem (lamp_height person_height initial_distance initial_shadow new_distance : ℝ) :
  lamp_height = 8 →
  initial_distance = 12 →
  initial_shadow = 4 →
  new_distance = 8 →
  person_height / initial_shadow = lamp_height / (initial_distance + initial_shadow) →
  person_height / (lamp_height / new_distance * new_distance - new_distance) = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_shadow_length_problem_l747_74735


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_empty_intersection_iff_a_in_range_l747_74758

def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 1| ≤ a ∧ a > 0}

def B : Set ℝ := {x : ℝ | x > 2 ∨ x < -2}

theorem intersection_when_a_is_two :
  A 2 ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

theorem empty_intersection_iff_a_in_range :
  ∀ a : ℝ, A a ∩ B = ∅ ↔ 0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_empty_intersection_iff_a_in_range_l747_74758


namespace NUMINAMATH_CALUDE_prob_fewer_heads_12_coins_l747_74716

/-- The number of coin flips -/
def n : ℕ := 12

/-- The probability of getting heads on a single fair coin flip -/
def p : ℚ := 1/2

/-- The probability of getting fewer heads than tails in n fair coin flips -/
def prob_fewer_heads (n : ℕ) (p : ℚ) : ℚ :=
  1/2 * (1 - (n.choose (n/2)) / (2^n : ℚ))

theorem prob_fewer_heads_12_coins : 
  prob_fewer_heads n p = 793/2048 := by sorry

end NUMINAMATH_CALUDE_prob_fewer_heads_12_coins_l747_74716


namespace NUMINAMATH_CALUDE_power_of_two_properties_l747_74703

theorem power_of_two_properties (n : ℕ) :
  (∃ k : ℕ, n = 3 * k ↔ 7 ∣ (2^n - 1)) ∧
  ¬(7 ∣ (2^n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_properties_l747_74703


namespace NUMINAMATH_CALUDE_circle_diameter_theorem_l747_74767

theorem circle_diameter_theorem (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 16 * Real.pi → A = Real.pi * r^2 → d = 2 * r → 3 * d = 24 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_theorem_l747_74767


namespace NUMINAMATH_CALUDE_smallest_w_l747_74706

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w : 
  ∀ w : ℕ, 
    w > 0 →
    is_factor (2^6) (1152 * w) →
    is_factor (3^4) (1152 * w) →
    is_factor (5^3) (1152 * w) →
    is_factor (7^2) (1152 * w) →
    is_factor 11 (1152 * w) →
    w ≥ 16275 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l747_74706


namespace NUMINAMATH_CALUDE_average_problem_l747_74704

theorem average_problem (t b c : ℝ) :
  (t + b + c + 14 + 15) / 5 = 12 →
  (t + b + c + 29) / 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l747_74704


namespace NUMINAMATH_CALUDE_min_cans_for_drinks_l747_74780

/-- Represents the available can sizes in liters -/
inductive CanSize
  | half
  | one
  | two

/-- Calculates the number of cans needed for a given volume and can size -/
def cansNeeded (volume : ℕ) (size : CanSize) : ℕ :=
  match size with
  | CanSize.half => volume * 2
  | CanSize.one => volume
  | CanSize.two => volume / 2

/-- Finds the minimum number of cans needed for a given volume -/
def minCansForVolume (volume : ℕ) : ℕ :=
  min (cansNeeded volume CanSize.half)
    (min (cansNeeded volume CanSize.one)
      (cansNeeded volume CanSize.two))

/-- The main theorem stating the minimum number of cans required -/
theorem min_cans_for_drinks :
  minCansForVolume 60 +
  minCansForVolume 220 +
  minCansForVolume 500 +
  minCansForVolume 315 +
  minCansForVolume 125 = 830 := by
  sorry


end NUMINAMATH_CALUDE_min_cans_for_drinks_l747_74780


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l747_74796

theorem number_satisfying_condition : ∃ x : ℝ, 0.65 * x = 0.8 * x - 21 ∧ x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l747_74796


namespace NUMINAMATH_CALUDE_jerry_tom_distance_difference_l747_74789

/-- The difference in distance run by Jerry and Tom around a square block -/
def distance_difference (block_side : ℝ) (street_width : ℝ) : ℝ :=
  4 * (block_side + 2 * street_width) - 4 * block_side

/-- Theorem stating the difference in distance run by Jerry and Tom -/
theorem jerry_tom_distance_difference :
  distance_difference 500 30 = 240 := by
  sorry

end NUMINAMATH_CALUDE_jerry_tom_distance_difference_l747_74789


namespace NUMINAMATH_CALUDE_triangle_altitude_bound_l747_74701

/-- For any triangle with perimeter 2, there exists at least one altitude that is less than or equal to 1/√3. -/
theorem triangle_altitude_bound (a b c : ℝ) (h_perimeter : a + b + c = 2) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) :
  ∃ h : ℝ, h ≤ 1 / Real.sqrt 3 ∧ (h = 2 * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) / a ∨
                                  h = 2 * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) / b ∨
                                  h = 2 * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) / c) :=
by sorry


end NUMINAMATH_CALUDE_triangle_altitude_bound_l747_74701


namespace NUMINAMATH_CALUDE_tom_missed_no_games_l747_74700

/-- The number of hockey games Tom missed this year -/
def games_missed_this_year (games_this_year games_last_year total_games : ℕ) : ℕ :=
  total_games - (games_this_year + games_last_year)

/-- Theorem: Tom missed 0 hockey games this year -/
theorem tom_missed_no_games :
  games_missed_this_year 4 9 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tom_missed_no_games_l747_74700


namespace NUMINAMATH_CALUDE_greatest_common_divisor_540_126_under_60_l747_74702

theorem greatest_common_divisor_540_126_under_60 : 
  Nat.gcd 540 126 < 60 ∧ 
  ∀ d : Nat, d ∣ 540 ∧ d ∣ 126 ∧ d < 60 → d ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_540_126_under_60_l747_74702


namespace NUMINAMATH_CALUDE_expression_expansion_l747_74793

theorem expression_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (7 / x^2 - 5 * x^3 + 2) = 3 / x^2 - 15 * x^3 / 7 + 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_expansion_l747_74793


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l747_74787

def N (p q r : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![p, q, r],
    ![q, r, p],
    ![r, p, q]]

theorem cube_root_unity_sum (p q r : ℂ) :
  N p q r ^ 3 = 1 →
  p * q * r = -1 →
  p^3 + q^3 + r^3 = -2 ∨ p^3 + q^3 + r^3 = -4 := by
sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l747_74787


namespace NUMINAMATH_CALUDE_three_multiples_of_three_l747_74707

theorem three_multiples_of_three (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →
  x % 3 = 0 ∧ y % 3 = 0 ∧ z % 3 = 0 →
  x + y + z = 36 →
  (∃ m : ℕ, x = 3 * m ∧ y = 3 * (m + 1) ∧ z = 3 * (m + 2)) ∧
  (∃ n : ℕ, x = 6 * n ∧ y = 6 * (n + 1) ∧ z = 6 * (n + 2)) :=
by sorry

#check three_multiples_of_three

end NUMINAMATH_CALUDE_three_multiples_of_three_l747_74707


namespace NUMINAMATH_CALUDE_binomial_12_9_l747_74788

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l747_74788


namespace NUMINAMATH_CALUDE_total_donation_equals_854_l747_74733

/-- Represents a fundraising event with earnings and donation percentage -/
structure FundraisingEvent where
  earnings : ℝ
  donationPercentage : ℝ

/-- Calculates the donation amount for a fundraising event -/
def donationAmount (event : FundraisingEvent) : ℝ :=
  event.earnings * event.donationPercentage

/-- Theorem: The total donation from five fundraising events equals $854 -/
theorem total_donation_equals_854 
  (carWash : FundraisingEvent)
  (bakeSale : FundraisingEvent)
  (mowingLawns : FundraisingEvent)
  (handmadeCrafts : FundraisingEvent)
  (charityConcert : FundraisingEvent)
  (h1 : carWash.earnings = 200 ∧ carWash.donationPercentage = 0.9)
  (h2 : bakeSale.earnings = 160 ∧ bakeSale.donationPercentage = 0.8)
  (h3 : mowingLawns.earnings = 120 ∧ mowingLawns.donationPercentage = 1)
  (h4 : handmadeCrafts.earnings = 180 ∧ handmadeCrafts.donationPercentage = 0.7)
  (h5 : charityConcert.earnings = 500 ∧ charityConcert.donationPercentage = 0.6)
  : donationAmount carWash + donationAmount bakeSale + donationAmount mowingLawns + 
    donationAmount handmadeCrafts + donationAmount charityConcert = 854 := by
  sorry


end NUMINAMATH_CALUDE_total_donation_equals_854_l747_74733


namespace NUMINAMATH_CALUDE_f_lower_bound_l747_74794

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x - log x

theorem f_lower_bound (a : ℝ) (h : a ≤ -1 / Real.exp 2) :
  ∀ x > 0, f a x ≥ 2 * a * x - x * Real.exp (a * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_lower_bound_l747_74794


namespace NUMINAMATH_CALUDE_steps_between_correct_l747_74705

/-- The number of steps taken from the Empire State Building to Madison Square Garden -/
def steps_between (total_steps down_steps : ℕ) : ℕ :=
  total_steps - down_steps

/-- Theorem stating that the steps between buildings is the difference of total steps and steps down -/
theorem steps_between_correct (total_steps down_steps : ℕ) 
  (h1 : total_steps ≥ down_steps)
  (h2 : total_steps = 991)
  (h3 : down_steps = 676) : 
  steps_between total_steps down_steps = 315 := by
  sorry

end NUMINAMATH_CALUDE_steps_between_correct_l747_74705


namespace NUMINAMATH_CALUDE_strawberry_growth_rate_l747_74722

/-- Represents the growth of strawberry plants over time -/
def strawberry_growth (initial_plants : ℕ) (months : ℕ) (plants_given_away : ℕ) (final_plants : ℕ) (growth_rate : ℕ) : Prop :=
  initial_plants + growth_rate * months - plants_given_away = final_plants

/-- Theorem stating that under the given conditions, the growth rate is 7 plants per month -/
theorem strawberry_growth_rate :
  strawberry_growth 3 3 4 20 7 := by
  sorry


end NUMINAMATH_CALUDE_strawberry_growth_rate_l747_74722


namespace NUMINAMATH_CALUDE_yellow_cards_per_player_l747_74766

theorem yellow_cards_per_player (total_players : ℕ) (uncautioned_players : ℕ) (red_cards : ℕ) 
  (h1 : total_players = 11)
  (h2 : uncautioned_players = 5)
  (h3 : red_cards = 3) :
  (total_players - uncautioned_players) * ((red_cards * 2) / (total_players - uncautioned_players)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_yellow_cards_per_player_l747_74766


namespace NUMINAMATH_CALUDE_gym_class_size_l747_74778

/-- The number of students in the third group -/
def third_group_size : ℕ := 37

/-- The percentage of students in the third group -/
def third_group_percentage : ℚ := 1/2

/-- The total number of students in the gym class -/
def total_students : ℕ := 74

theorem gym_class_size :
  (third_group_size : ℚ) / third_group_percentage = total_students := by
  sorry

end NUMINAMATH_CALUDE_gym_class_size_l747_74778


namespace NUMINAMATH_CALUDE_simplify_expression_l747_74713

theorem simplify_expression (a : ℝ) : (3 * a)^2 * a^5 = 9 * a^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l747_74713


namespace NUMINAMATH_CALUDE_walking_jogging_time_difference_l747_74754

/-- 
Given:
- Linda walks at 4 miles per hour
- Tom jogs at 9 miles per hour
- Linda starts walking 1 hour before Tom
- They walk in opposite directions

Prove that the difference in time (in minutes) for Tom to cover half and twice Linda's distance is 40 minutes.
-/
theorem walking_jogging_time_difference 
  (linda_speed : ℝ) 
  (tom_speed : ℝ) 
  (head_start : ℝ) :
  linda_speed = 4 →
  tom_speed = 9 →
  head_start = 1 →
  let linda_distance := linda_speed * head_start
  let half_distance := linda_distance / 2
  let double_distance := linda_distance * 2
  let time_half := (half_distance / tom_speed) * 60
  let time_double := (double_distance / tom_speed) * 60
  time_double - time_half = 40 := by
  sorry

end NUMINAMATH_CALUDE_walking_jogging_time_difference_l747_74754


namespace NUMINAMATH_CALUDE_chessboard_cover_l747_74723

def coverWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | k + 3 => coverWays (k + 2) + coverWays (k + 1)

theorem chessboard_cover : coverWays 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_cover_l747_74723


namespace NUMINAMATH_CALUDE_square_figure_perimeter_l747_74743

/-- A figure composed of two rows of three consecutive unit squares, with the top row directly above the bottom row -/
structure SquareFigure where
  /-- The side length of each square -/
  side_length : ℝ
  /-- The number of squares in each row -/
  squares_per_row : ℕ
  /-- The number of rows -/
  rows : ℕ
  /-- The side length is 1 -/
  unit_side : side_length = 1
  /-- There are three squares in each row -/
  three_squares : squares_per_row = 3
  /-- There are two rows -/
  two_rows : rows = 2

/-- The perimeter of the SquareFigure -/
def perimeter (fig : SquareFigure) : ℝ :=
  2 * fig.side_length * fig.squares_per_row + 2 * fig.side_length * fig.rows

/-- Theorem stating that the perimeter of the SquareFigure is 16 -/
theorem square_figure_perimeter (fig : SquareFigure) : perimeter fig = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_figure_perimeter_l747_74743


namespace NUMINAMATH_CALUDE_base_conversion_sum_l747_74741

-- Define the base conversions
def base_8_to_10 (n : ℕ) : ℕ := 
  2 * (8^2) + 5 * (8^1) + 4 * (8^0)

def base_3_to_10 (n : ℕ) : ℕ := 
  1 * (3^1) + 3 * (3^0)

def base_7_to_10 (n : ℕ) : ℕ := 
  2 * (7^2) + 3 * (7^1) + 2 * (7^0)

def base_5_to_10 (n : ℕ) : ℕ := 
  3 * (5^1) + 2 * (5^0)

-- Theorem statement
theorem base_conversion_sum :
  (base_8_to_10 254 / base_3_to_10 13) + (base_7_to_10 232 / base_5_to_10 32) = 35 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l747_74741


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l747_74724

theorem cubic_equation_solutions :
  ∀ x : ℝ, (x^3 - 3*x^2*(Real.sqrt 3) + 9*x - 3*(Real.sqrt 3)) + (x - Real.sqrt 3)^2 = 0 ↔ 
  x = Real.sqrt 3 ∨ x = -1 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l747_74724


namespace NUMINAMATH_CALUDE_orange_roses_count_l747_74774

theorem orange_roses_count (red_roses : ℕ) (pink_roses : ℕ) (yellow_roses : ℕ) 
  (total_picked : ℕ) (h1 : red_roses = 12) (h2 : pink_roses = 18) 
  (h3 : yellow_roses = 20) (h4 : total_picked = 22) :
  ∃ (orange_roses : ℕ), 
    orange_roses = 8 ∧ 
    total_picked = red_roses / 2 + pink_roses / 2 + yellow_roses / 4 + orange_roses / 4 :=
by sorry

end NUMINAMATH_CALUDE_orange_roses_count_l747_74774


namespace NUMINAMATH_CALUDE_platform_length_calculation_platform_length_proof_l747_74760

/-- Calculates the length of a platform given train parameters --/
theorem platform_length_calculation 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time
  let platform_length := total_distance - train_length
  platform_length

/-- Proves that the platform length is approximately 190.08 m given the specified conditions --/
theorem platform_length_proof 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 90) 
  (h2 : train_speed_kmph = 56) 
  (h3 : crossing_time = 18) : 
  ∃ (ε : ℝ), ε > 0 ∧ abs (platform_length_calculation train_length train_speed_kmph crossing_time - 190.08) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_platform_length_proof_l747_74760


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l747_74752

/-- Represents a square quilt made of smaller squares -/
structure Quilt :=
  (size : Nat)
  (shaded_row : Nat)
  (shaded_column : Nat)

/-- Calculates the fraction of shaded area in a quilt -/
def shaded_fraction (q : Quilt) : Rat :=
  let total_squares := q.size * q.size
  let shaded_squares := q.size + q.size - 1
  shaded_squares / total_squares

/-- Theorem stating that for a 4x4 quilt with one shaded row and column, 
    the shaded fraction is 7/16 -/
theorem quilt_shaded_fraction :
  ∀ (q : Quilt), q.size = 4 → shaded_fraction q = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l747_74752


namespace NUMINAMATH_CALUDE_cos_pi_plus_alpha_implies_sin_5pi_over_2_minus_alpha_l747_74746

theorem cos_pi_plus_alpha_implies_sin_5pi_over_2_minus_alpha 
  (α : Real) 
  (h : Real.cos (Real.pi + α) = -1/3) : 
  Real.sin ((5/2) * Real.pi - α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_alpha_implies_sin_5pi_over_2_minus_alpha_l747_74746


namespace NUMINAMATH_CALUDE_union_determines_m_l747_74792

theorem union_determines_m (A B : Set ℕ) (m : ℕ) : 
  A = {1, 3, m} → 
  B = {3, 4} → 
  A ∪ B = {1, 2, 3, 4} → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_union_determines_m_l747_74792


namespace NUMINAMATH_CALUDE_josh_gummy_bears_l747_74749

theorem josh_gummy_bears (initial_candies : ℕ) : 
  (∃ (remaining_after_siblings : ℕ) (remaining_after_friend : ℕ),
    initial_candies = 3 * 10 + remaining_after_siblings ∧
    remaining_after_siblings = 2 * remaining_after_friend ∧
    remaining_after_friend = 16 + 19) →
  initial_candies = 100 := by
sorry

end NUMINAMATH_CALUDE_josh_gummy_bears_l747_74749


namespace NUMINAMATH_CALUDE_f_symmetry_and_increase_l747_74785

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 - 1/2

def is_center_of_symmetry (c : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (c.1 + x) = f (c.1 - x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_symmetry_and_increase :
  (∀ k : ℤ, is_center_of_symmetry (k * Real.pi / 2 + Real.pi / 12, -1) f) ∧
  is_increasing_on f 0 (Real.pi / 3) ∧
  is_increasing_on f (5 * Real.pi / 6) Real.pi :=
sorry

end NUMINAMATH_CALUDE_f_symmetry_and_increase_l747_74785


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l747_74721

theorem jelly_bean_probability (p_red p_green p_orange_yellow : ℝ) :
  p_red = 0.25 →
  p_green = 0.35 →
  p_red + p_green + p_orange_yellow = 1 →
  p_orange_yellow = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l747_74721


namespace NUMINAMATH_CALUDE_min_value_inequality_l747_74753

theorem min_value_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l747_74753


namespace NUMINAMATH_CALUDE_notebook_price_l747_74750

theorem notebook_price (notebook_count : ℕ) (pencil_price pen_price total_spent : ℚ) : 
  notebook_count = 3 →
  pencil_price = 1.5 →
  pen_price = 1.7 →
  total_spent = 6.8 →
  ∃ (notebook_price : ℚ), 
    notebook_count * notebook_price + pencil_price + pen_price = total_spent ∧
    notebook_price = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_notebook_price_l747_74750


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l747_74783

def a (n : ℕ) := 2 * (n + 1) + 3

theorem arithmetic_sequence_proof :
  ∀ n : ℕ, a (n + 1) - a n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l747_74783


namespace NUMINAMATH_CALUDE_factor_problem_l747_74755

theorem factor_problem (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 6 → 
  (2 * initial_number + 9) * factor = 63 → 
  factor = 3 := by
sorry

end NUMINAMATH_CALUDE_factor_problem_l747_74755


namespace NUMINAMATH_CALUDE_increasing_function_bounds_l747_74764

theorem increasing_function_bounds (k : ℕ) (f : ℕ → ℕ) 
  (h_increasing : ∀ m n, m < n → f m < f n) 
  (h_functional : ∀ n, f (f n) = k * n) :
  ∀ n, (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ ((k + 1) * n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_bounds_l747_74764


namespace NUMINAMATH_CALUDE_at_most_one_integer_solution_l747_74777

theorem at_most_one_integer_solution (a b : ℤ) :
  ∃! (n : ℕ), ∃ (x : ℤ), (x - a) * (x - b) * (x - 3) + 1 = 0 ∧ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_integer_solution_l747_74777


namespace NUMINAMATH_CALUDE_certain_time_in_seconds_l747_74714

/-- Given that 8 is to a certain time as 8 is to 4 minutes, and there are 60 seconds in a minute,
    prove that the certain time is 240 seconds. -/
theorem certain_time_in_seconds : ∃ (t : ℕ), t = 240 ∧ (8 : ℚ) / t = 8 / (4 * 60) := by
  sorry

end NUMINAMATH_CALUDE_certain_time_in_seconds_l747_74714


namespace NUMINAMATH_CALUDE_cory_fruit_sequences_l747_74784

/-- The number of distinct sequences for eating fruits -/
def fruitSequences (apples oranges bananas pears : ℕ) : ℕ :=
  let total := apples + oranges + bananas + pears
  Nat.factorial total / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas * Nat.factorial pears)

/-- Theorem: The number of distinct sequences for eating 4 apples, 2 oranges, 1 banana, and 2 pears over 8 days is 420 -/
theorem cory_fruit_sequences :
  fruitSequences 4 2 1 2 = 420 := by
  sorry

#eval fruitSequences 4 2 1 2

end NUMINAMATH_CALUDE_cory_fruit_sequences_l747_74784


namespace NUMINAMATH_CALUDE_series_sum_l747_74782

/-- The sum of the infinite series 2 + ∑(k=1 to ∞) ((k+2)*(1/1000)^(k-1)) is equal to 3000000/998001 -/
theorem series_sum : 
  let S := 2 + ∑' k, (k + 2) * (1 / 1000) ^ (k - 1)
  S = 3000000 / 998001 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l747_74782


namespace NUMINAMATH_CALUDE_triangle_ratio_l747_74798

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  A = Real.pi / 3 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
sorry


end NUMINAMATH_CALUDE_triangle_ratio_l747_74798


namespace NUMINAMATH_CALUDE_sqrt_difference_power_l747_74720

theorem sqrt_difference_power (m n : ℕ) :
  ∃ k : ℕ, (Real.sqrt m - Real.sqrt (m - 1)) ^ n = Real.sqrt k - Real.sqrt (k - 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_difference_power_l747_74720


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l747_74751

theorem sphere_volume_ratio (r₁ r₂ r₃ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ = 2 * r₁) (h₃ : r₃ = 3 * r₁) :
  (4 / 3) * π * r₃^3 = 3 * ((4 / 3) * π * r₁^3 + (4 / 3) * π * r₂^3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l747_74751


namespace NUMINAMATH_CALUDE_inverse_of_A_l747_74744

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem inverse_of_A :
  let inv_A : Matrix (Fin 2) (Fin 2) ℝ := !![-1, -3; -2, -5]
  Matrix.det A ≠ 0 → A * inv_A = 1 ∧ inv_A * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_l747_74744


namespace NUMINAMATH_CALUDE_max_triangles_is_eleven_l747_74709

/-- Represents an equilateral triangle with a line segment connecting midpoints of two sides -/
structure TriangleWithMidLine :=
  (side_length : ℝ)
  (midline_angle : ℝ)

/-- Represents the configuration of two overlapping triangles -/
structure OverlappingTriangles :=
  (triangle_a : TriangleWithMidLine)
  (triangle_b : TriangleWithMidLine)
  (overlap_distance : ℝ)

/-- Counts the number of smaller triangles formed in a given configuration -/
def count_triangles (config : OverlappingTriangles) : ℕ :=
  sorry

/-- The maximum number of smaller triangles that can be formed -/
def max_triangle_count : ℕ := 11

/-- Theorem stating that the maximum number of smaller triangles is 11 -/
theorem max_triangles_is_eleven :
  ∀ config : OverlappingTriangles, count_triangles config ≤ max_triangle_count :=
sorry

end NUMINAMATH_CALUDE_max_triangles_is_eleven_l747_74709


namespace NUMINAMATH_CALUDE_triangle_inequality_l747_74775

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) : 
  3 * (a * b + a * c + b * c) ≤ (a + b + c)^2 ∧ (a + b + c)^2 < 4 * (a * b + a * c + b * c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l747_74775


namespace NUMINAMATH_CALUDE_hyperbola_focal_coordinates_l747_74739

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The focal coordinates of the hyperbola -/
def focal_coordinates : Set (ℝ × ℝ) := {(-5, 0), (5, 0)}

/-- Theorem: The focal coordinates of the hyperbola x^2/16 - y^2/9 = 1 are (-5, 0) and (5, 0) -/
theorem hyperbola_focal_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ focal_coordinates :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_coordinates_l747_74739


namespace NUMINAMATH_CALUDE_hike_water_theorem_l747_74734

/-- Represents the water consumption during Harry's hike --/
structure HikeWater where
  duration : ℝ  -- Duration of the hike in hours
  distance : ℝ  -- Total distance of the hike in miles
  leak_rate : ℝ  -- Leak rate of the canteen in cups per hour
  last_mile_consumption : ℝ  -- Water consumed in the last mile in cups
  first_miles_consumption : ℝ  -- Water consumed per mile for the first 3 miles
  remaining_water : ℝ  -- Water remaining at the end of the hike in cups

/-- Calculates the initial amount of water in the canteen --/
def initial_water (h : HikeWater) : ℝ :=
  h.leak_rate * h.duration +
  h.last_mile_consumption +
  h.first_miles_consumption * (h.distance - 1) +
  h.remaining_water

/-- Theorem stating that the initial amount of water in the canteen was 10 cups --/
theorem hike_water_theorem (h : HikeWater)
  (h_duration : h.duration = 2)
  (h_distance : h.distance = 4)
  (h_leak_rate : h.leak_rate = 1)
  (h_last_mile : h.last_mile_consumption = 3)
  (h_first_miles : h.first_miles_consumption = 1)
  (h_remaining : h.remaining_water = 2) :
  initial_water h = 10 := by
  sorry


end NUMINAMATH_CALUDE_hike_water_theorem_l747_74734


namespace NUMINAMATH_CALUDE_share_in_ratio_l747_74740

theorem share_in_ratio (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) (h1 : total = 4320) (h2 : ratio1 = 2) (h3 : ratio2 = 4) (h4 : ratio3 = 6) :
  let sum_ratio := ratio1 + ratio2 + ratio3
  let share1 := total * ratio1 / sum_ratio
  share1 = 720 := by
sorry

end NUMINAMATH_CALUDE_share_in_ratio_l747_74740


namespace NUMINAMATH_CALUDE_cone_height_ratio_l747_74717

/-- Theorem: Ratio of shortened height to original height of a cone -/
theorem cone_height_ratio (r : ℝ) (h₀ : ℝ) (h : ℝ) :
  r > 0 ∧ h₀ > 0 ∧ h > 0 →
  2 * Real.pi * r = 20 * Real.pi →
  h₀ = 50 →
  (1/3) * Real.pi * r^2 * h = 500 * Real.pi →
  h / h₀ = 3/10 := by sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l747_74717


namespace NUMINAMATH_CALUDE_twenty_team_tournament_matches_l747_74779

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  num_matches : ℕ

/-- Calculates the number of matches needed in a single-elimination tournament. -/
def matches_needed (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 20 teams requires 19 matches. -/
theorem twenty_team_tournament_matches :
  ∀ t : Tournament, t.num_teams = 20 → t.num_matches = matches_needed t.num_teams := by
  sorry

end NUMINAMATH_CALUDE_twenty_team_tournament_matches_l747_74779


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_equality_l747_74729

theorem sqrt_abs_sum_equality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a^2 + b^2) + |a - b| = a + b ↔ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_equality_l747_74729


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l747_74727

/-- Given a line segment with midpoint (2, 3) and one endpoint (5, -1), prove that the other endpoint is (-1, 7) -/
theorem other_endpoint_of_line_segment (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (2, 3) → endpoint1 = (5, -1) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-1, 7) := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l747_74727


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l747_74757

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (2, x + 2)
  parallel a b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l747_74757


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l747_74781

theorem triangle_angle_measure (b c S_ABC : ℝ) (h1 : b = 8) (h2 : c = 8 * Real.sqrt 3) 
  (h3 : S_ABC = 16 * Real.sqrt 3) :
  ∃ A : ℝ, (A = π / 6 ∨ A = 5 * π / 6) ∧ 
    S_ABC = (1/2) * b * c * Real.sin A ∧ 0 < A ∧ A < π :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l747_74781


namespace NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_5_l747_74776

theorem remainder_11_pow_2023_mod_5 : 11^2023 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_5_l747_74776


namespace NUMINAMATH_CALUDE_min_communication_size_l747_74742

/-- Represents a set of cards with positive numbers -/
def CardSet := Finset ℕ+

/-- The number of cards -/
def n : ℕ := 100

/-- A function that takes a set of cards and returns a set of communicated values -/
def communicate (cards : CardSet) : Finset ℕ := sorry

/-- Predicate to check if a set of communicated values uniquely determines the original card set -/
def uniquely_determines (comms : Finset ℕ) (cards : CardSet) : Prop := sorry

theorem min_communication_size :
  ∀ (cards : CardSet),
  (cards.card = n) →
  ∃ (comms : Finset ℕ),
    (communicate cards = comms) ∧
    (uniquely_determines comms cards) ∧
    (comms.card = n + 1) ∧
    (∀ (comms' : Finset ℕ),
      (communicate cards = comms') ∧
      (uniquely_determines comms' cards) →
      (comms'.card ≥ n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_min_communication_size_l747_74742


namespace NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_l747_74771

open Real

theorem smallest_angle_for_complete_circle : 
  ∃ t : ℝ, t > 0 ∧ 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = sin θ) ∧
  (∀ t' : ℝ, 0 < t' ∧ t' < t → 
    ¬(∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t' → ∃ r : ℝ, r = sin θ ∧ 
      ∃ x y : ℝ, x = r * cos θ ∧ y = r * sin θ ∧ 
      (∀ x' y' : ℝ, ∃ θ' : ℝ, 0 ≤ θ' ∧ θ' ≤ t' ∧ 
        x' = (sin θ') * cos θ' ∧ y' = (sin θ') * sin θ'))) ∧
  t = 2 * π :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_l747_74771


namespace NUMINAMATH_CALUDE_f_f_zero_equals_three_pi_squared_minus_four_l747_74795

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

theorem f_f_zero_equals_three_pi_squared_minus_four :
  f (f 0) = 3 * Real.pi^2 - 4 := by sorry

end NUMINAMATH_CALUDE_f_f_zero_equals_three_pi_squared_minus_four_l747_74795


namespace NUMINAMATH_CALUDE_tons_to_kilograms_l747_74711

-- Define the mass units
def ton : ℝ := 1000
def kilogram : ℝ := 1

-- State the theorem
theorem tons_to_kilograms : 24 * ton = 24000 * kilogram := by sorry

end NUMINAMATH_CALUDE_tons_to_kilograms_l747_74711


namespace NUMINAMATH_CALUDE_investment_average_rate_l747_74791

/-- Proves that given a total investment split between two rates with equal returns, the average rate is as expected -/
theorem investment_average_rate 
  (total_investment : ℝ) 
  (rate1 rate2 : ℝ) 
  (h_total : total_investment = 5000)
  (h_rates : rate1 = 0.05 ∧ rate2 = 0.03)
  (h_equal_returns : ∃ (x : ℝ), x * rate1 = (total_investment - x) * rate2)
  : (((rate1 * (total_investment * rate1 / (rate1 + rate2))) + 
     (rate2 * (total_investment * rate2 / (rate1 + rate2)))) / total_investment) = 0.0375 :=
sorry

end NUMINAMATH_CALUDE_investment_average_rate_l747_74791


namespace NUMINAMATH_CALUDE_kris_herbert_age_difference_l747_74748

/-- The age difference between two people --/
def age_difference (age1 : ℕ) (age2 : ℕ) : ℕ := 
  if age1 ≥ age2 then age1 - age2 else age2 - age1

/-- Theorem: The age difference between Kris and Herbert is 10 years --/
theorem kris_herbert_age_difference : 
  let kris_age : ℕ := 24
  let herbert_age_next_year : ℕ := 15
  let herbert_age : ℕ := herbert_age_next_year - 1
  age_difference kris_age herbert_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_kris_herbert_age_difference_l747_74748


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_approximate_value_of_b_l747_74731

theorem geometric_sequence_second_term (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 30 * r = b ∧ b * r = 9/4) → 
  b^2 = 270/4 :=
by sorry

theorem approximate_value_of_b : 
  ∃ b : ℝ, b > 0 ∧ 
  (∃ r : ℝ, 30 * r = b ∧ b * r = 9/4) ∧ 
  (b^2 = 270/4) ∧ 
  (abs (b - 8.215838362) < 0.000000001) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_approximate_value_of_b_l747_74731


namespace NUMINAMATH_CALUDE_angle_C_value_l747_74770

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x + (Real.sqrt 3/2) * Real.cos x

theorem angle_C_value (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  f A = Real.sqrt 3 / 2 ∧
  a = (Real.sqrt 3 / 2) * b ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C →
  C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_value_l747_74770


namespace NUMINAMATH_CALUDE_gear_alignment_l747_74756

theorem gear_alignment (n : ℕ) (h1 : n = 6) :
  ∃ (rotation : Fin 32), ∀ (i : Fin n),
    (i.val + rotation : Fin 32) ∉ {j : Fin 32 | j.val < n} :=
sorry

end NUMINAMATH_CALUDE_gear_alignment_l747_74756


namespace NUMINAMATH_CALUDE_constant_function_solution_l747_74761

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y = f (x - y)

theorem constant_function_solution
  (f : ℝ → ℝ)
  (hf : FunctionalEquation f)
  (hnz : ∃ x, f x ≠ 0) :
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k := by
  sorry

end NUMINAMATH_CALUDE_constant_function_solution_l747_74761


namespace NUMINAMATH_CALUDE_total_elephants_count_l747_74772

def elephants_we_preserve : ℕ := 70

def elephants_gestures_for_good : ℕ := 3 * elephants_we_preserve

def total_elephants : ℕ := elephants_we_preserve + elephants_gestures_for_good

theorem total_elephants_count : total_elephants = 280 := by
  sorry

end NUMINAMATH_CALUDE_total_elephants_count_l747_74772
