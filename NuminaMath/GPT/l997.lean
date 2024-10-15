import Mathlib

namespace NUMINAMATH_GPT_calc_expression_result_l997_99764

theorem calc_expression_result :
  (16^12 * 8^8 / 2^60 = 4096) :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_result_l997_99764


namespace NUMINAMATH_GPT_min_max_value_z_l997_99718

theorem min_max_value_z (x y z : ℝ) (h1 : x^2 ≤ y + z) (h2 : y^2 ≤ z + x) (h3 : z^2 ≤ x + y) :
  -1/4 ≤ z ∧ z ≤ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_max_value_z_l997_99718


namespace NUMINAMATH_GPT_root_interval_exists_l997_99707

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - x + 1

theorem root_interval_exists :
  (f 2 > 0) →
  (f 3 < 0) →
  ∃ ξ, 2 < ξ ∧ ξ < 3 ∧ f ξ = 0 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_root_interval_exists_l997_99707


namespace NUMINAMATH_GPT_intersection_point_exists_l997_99794

noncomputable def line1 (t : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + 6 * t)
noncomputable def line2 (u : ℝ) : ℝ × ℝ := (3 + u, 8 + 3 * u)

theorem intersection_point_exists :
  ∃ t u : ℝ, line1 t = (1, 2) ∧ line2 u = (1, 2) := 
by
  sorry

end NUMINAMATH_GPT_intersection_point_exists_l997_99794


namespace NUMINAMATH_GPT_gold_cube_profit_multiple_l997_99726

theorem gold_cube_profit_multiple :
  let side_length : ℝ := 6
  let density : ℝ := 19
  let cost_per_gram : ℝ := 60
  let profit : ℝ := 123120
  let volume := side_length ^ 3
  let mass := density * volume
  let cost := mass * cost_per_gram
  let selling_price := cost + profit
  let multiple := selling_price / cost
  multiple = 1.5 := by
  sorry

end NUMINAMATH_GPT_gold_cube_profit_multiple_l997_99726


namespace NUMINAMATH_GPT_ratio_future_age_l997_99783

variables (S : ℕ) (M : ℕ) (S_future : ℕ) (M_future : ℕ)

def son_age := 44
def man_age := son_age + 46
def son_age_future := son_age + 2
def man_age_future := man_age + 2

theorem ratio_future_age : man_age_future / son_age_future = 2 := by
  -- You can add the proof here if you want
  sorry

end NUMINAMATH_GPT_ratio_future_age_l997_99783


namespace NUMINAMATH_GPT_negation_proposition_equivalence_l997_99759

theorem negation_proposition_equivalence : 
    (¬ ∃ x_0 : ℝ, (x_0^2 + 1 > 0) ∨ (x_0 > Real.sin x_0)) ↔ 
    (∀ x : ℝ, (x^2 + 1 ≤ 0) ∧ (x ≤ Real.sin x)) :=
by 
    sorry

end NUMINAMATH_GPT_negation_proposition_equivalence_l997_99759


namespace NUMINAMATH_GPT_range_of_m_l997_99711

def p (m : ℝ) : Prop := m^2 - 4 > 0 ∧ m > 0
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0
def condition1 (m : ℝ) : Prop := p m ∨ q m
def condition2 (m : ℝ) : Prop := ¬ (p m ∧ q m)

theorem range_of_m (m : ℝ) : condition1 m ∧ condition2 m → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l997_99711


namespace NUMINAMATH_GPT_pair_solution_l997_99768

theorem pair_solution (a b : ℕ) (h_b_ne_1 : b ≠ 1) :
  (a + 1 ∣ a^3 * b - 1) → (b - 1 ∣ b^3 * a + 1) →
  (a, b) = (0, 0) ∨ (a, b) = (0, 2) ∨ (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3) :=
by
  sorry

end NUMINAMATH_GPT_pair_solution_l997_99768


namespace NUMINAMATH_GPT_california_vs_texas_license_plates_l997_99742

theorem california_vs_texas_license_plates :
  (26^4 * 10^4) - (26^3 * 10^3) = 4553200000 :=
by
  sorry

end NUMINAMATH_GPT_california_vs_texas_license_plates_l997_99742


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l997_99771

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (a6 : a 6 = -3) 
  (S6 : S 6 = 12)
  (h_sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 1 - a 0)) / 2)
  : a 5 = -1 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l997_99771


namespace NUMINAMATH_GPT_mandy_chocolate_pieces_l997_99782

def chocolate_pieces_total : ℕ := 60
def half (n : ℕ) : ℕ := n / 2

def michael_taken : ℕ := half chocolate_pieces_total
def paige_taken : ℕ := half (chocolate_pieces_total - michael_taken)
def ben_taken : ℕ := half (chocolate_pieces_total - michael_taken - paige_taken)
def mandy_left : ℕ := chocolate_pieces_total - michael_taken - paige_taken - ben_taken

theorem mandy_chocolate_pieces : mandy_left = 8 :=
  by
  -- proof to be provided here
  sorry

end NUMINAMATH_GPT_mandy_chocolate_pieces_l997_99782


namespace NUMINAMATH_GPT_nell_has_cards_left_l997_99796

def initial_cards : ℕ := 242
def cards_given_away : ℕ := 136

theorem nell_has_cards_left :
  initial_cards - cards_given_away = 106 :=
by
  sorry

end NUMINAMATH_GPT_nell_has_cards_left_l997_99796


namespace NUMINAMATH_GPT_second_metal_gold_percentage_l997_99728

theorem second_metal_gold_percentage (w_final : ℝ) (p_final : ℝ) (w_part : ℝ) (p_part1 : ℝ) (w_part1 : ℝ) (w_part2 : ℝ)
  (h_w_final : w_final = 12.4) (h_p_final : p_final = 0.5) (h_w_part : w_part = 6.2) (h_p_part1 : p_part1 = 0.6)
  (h_w_part1 : w_part1 = 6.2) (h_w_part2 : w_part2 = 6.2) :
  ∃ p_part2 : ℝ, p_part2 = 0.4 :=
by sorry

end NUMINAMATH_GPT_second_metal_gold_percentage_l997_99728


namespace NUMINAMATH_GPT_solve_tangent_problem_l997_99744

noncomputable def problem_statement : Prop :=
  ∃ (n : ℤ), (-90 < n ∧ n < 90) ∧ (Real.tan (n * Real.pi / 180) = Real.tan (255 * Real.pi / 180)) ∧ (n = 75)

-- This is the statement of the problem we are proving.
theorem solve_tangent_problem : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_solve_tangent_problem_l997_99744


namespace NUMINAMATH_GPT_polynomial_inequality_solution_l997_99763

theorem polynomial_inequality_solution (x : ℝ) :
  (x < 5 - Real.sqrt 29 ∨ x > 5 + Real.sqrt 29) →
  x^3 - 12 * x^2 + 36 * x + 8 > 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_inequality_solution_l997_99763


namespace NUMINAMATH_GPT_pet_store_cats_left_l997_99743

theorem pet_store_cats_left (siamese house sold : ℕ) (h_siamese : siamese = 38) (h_house : house = 25) (h_sold : sold = 45) :
  siamese + house - sold = 18 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_cats_left_l997_99743


namespace NUMINAMATH_GPT_orange_bin_count_l997_99716

theorem orange_bin_count (initial_count throw_away add_new : ℕ) 
  (h1 : initial_count = 40) 
  (h2 : throw_away = 37) 
  (h3 : add_new = 7) : 
  initial_count - throw_away + add_new = 10 := 
by 
  sorry

end NUMINAMATH_GPT_orange_bin_count_l997_99716


namespace NUMINAMATH_GPT_fraction_expression_of_repeating_decimal_l997_99795

theorem fraction_expression_of_repeating_decimal :
  ∃ (x : ℕ), x = 79061333 ∧ (∀ y : ℚ, y = 0.71 + 264 * (1/999900) → x / 999900 = y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_expression_of_repeating_decimal_l997_99795


namespace NUMINAMATH_GPT_consecutive_negatives_product_sum_l997_99705

theorem consecutive_negatives_product_sum:
  ∃ (n: ℤ), n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 3080 ∧ n + (n + 1) = -111 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_negatives_product_sum_l997_99705


namespace NUMINAMATH_GPT_shortest_player_height_l997_99780

theorem shortest_player_height :
  ∀ (tallest_height difference : ℝ), 
    tallest_height = 77.75 ∧ difference = 9.5 → 
    tallest_height - difference = 68.25 :=
by
  intros tallest_height difference h
  cases h
  sorry

end NUMINAMATH_GPT_shortest_player_height_l997_99780


namespace NUMINAMATH_GPT_negation_of_universal_l997_99713

theorem negation_of_universal:
  ¬(∀ x : ℝ, (0 < x ∧ x < (π / 2)) → x > Real.sin x) ↔
  ∃ x : ℝ, (0 < x ∧ x < (π / 2)) ∧ x ≤ Real.sin x := by
  sorry

end NUMINAMATH_GPT_negation_of_universal_l997_99713


namespace NUMINAMATH_GPT_inequality_ab_leq_a_b_l997_99729

theorem inequality_ab_leq_a_b (a b : ℝ) (x : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  a * b ≤ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2)
  ∧ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2) ≤ (a + b)^2 / 4 := 
sorry

end NUMINAMATH_GPT_inequality_ab_leq_a_b_l997_99729


namespace NUMINAMATH_GPT_find_sum_of_bounds_l997_99767

variable (x y z : ℝ)

theorem find_sum_of_bounds (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) : 
  let m := min x (min y z)
  let M := max x (max y z)
  m + M = 8 / 3 :=
sorry

end NUMINAMATH_GPT_find_sum_of_bounds_l997_99767


namespace NUMINAMATH_GPT_divisibility_of_expression_l997_99725

open Int

theorem divisibility_of_expression (a b : ℤ) (ha : Prime a) (hb : Prime b) (ha_gt7 : a > 7) (hb_gt7 : b > 7) :
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) :=
sorry

end NUMINAMATH_GPT_divisibility_of_expression_l997_99725


namespace NUMINAMATH_GPT_ratio_identity_l997_99762

-- Given system of equations
def system_of_equations (k : ℚ) (x y z : ℚ) :=
  x + k * y + 2 * z = 0 ∧
  2 * x + k * y + 3 * z = 0 ∧
  3 * x + 5 * y + 4 * z = 0

-- Prove that for k = -7/5, the system has a nontrivial solution and 
-- that the ratio xz / y^2 equals -25
theorem ratio_identity (x y z : ℚ) (k : ℚ) (h : system_of_equations k x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  k = -7 / 5 → x * z / y^2 = -25 :=
by
  sorry

end NUMINAMATH_GPT_ratio_identity_l997_99762


namespace NUMINAMATH_GPT_volume_of_prism_l997_99760

theorem volume_of_prism :
  ∃ (a b c : ℝ), ab * bc * ac = 762 ∧ (ab = 56) ∧ (bc = 63) ∧ (ac = 72) ∧ (b = 2 * a) :=
sorry

end NUMINAMATH_GPT_volume_of_prism_l997_99760


namespace NUMINAMATH_GPT_power_of_fraction_l997_99798

theorem power_of_fraction :
  ( (2 / 5: ℝ) ^ 7 = 128 / 78125) :=
by
  sorry

end NUMINAMATH_GPT_power_of_fraction_l997_99798


namespace NUMINAMATH_GPT_isosceles_triangle_k_l997_99761

theorem isosceles_triangle_k (m n k : ℝ) (h_iso : (m = 4 ∨ n = 4 ∨ m = n) ∧ (m ≠ n ∨ (m = n ∧ m + m > 4))) 
  (h_roots : ∀ x, x^2 - 6*x + (k + 2) = 0 → (x = m ∨ x = n)) : k = 6 ∨ k = 7 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_k_l997_99761


namespace NUMINAMATH_GPT_coffee_shrinkage_l997_99727

theorem coffee_shrinkage :
  let initial_volume_per_cup := 8
  let shrink_factor := 0.5
  let number_of_cups := 5
  let final_volume_per_cup := initial_volume_per_cup * shrink_factor
  let total_remaining_coffee := final_volume_per_cup * number_of_cups
  total_remaining_coffee = 20 :=
by
  -- This is where the steps of the solution would go.
  -- We'll put a sorry here to indicate omission of proof.
  sorry

end NUMINAMATH_GPT_coffee_shrinkage_l997_99727


namespace NUMINAMATH_GPT_point_in_third_quadrant_l997_99745

theorem point_in_third_quadrant (x y : ℝ) (h1 : x = -3) (h2 : y = -2) : 
  x < 0 ∧ y < 0 :=
by
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l997_99745


namespace NUMINAMATH_GPT_suitable_comprehensive_survey_l997_99787

theorem suitable_comprehensive_survey :
  ¬(A = "comprehensive") ∧ ¬(B = "comprehensive") ∧ (C = "comprehensive") ∧ ¬(D = "comprehensive") → 
  suitable_survey = "C" :=
by
  sorry

end NUMINAMATH_GPT_suitable_comprehensive_survey_l997_99787


namespace NUMINAMATH_GPT_rental_difference_l997_99720

variable (C K : ℕ)

theorem rental_difference
  (hc : 15 * C + 18 * K = 405)
  (hr : 3 * K = 2 * C) :
  C - K = 5 :=
sorry

end NUMINAMATH_GPT_rental_difference_l997_99720


namespace NUMINAMATH_GPT_inv_sum_eq_six_l997_99781

theorem inv_sum_eq_six (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a + b = 6 * (a * b)) : 1 / a + 1 / b = 6 := 
by 
  sorry

end NUMINAMATH_GPT_inv_sum_eq_six_l997_99781


namespace NUMINAMATH_GPT_julia_birth_year_is_1979_l997_99702

-- Definitions based on conditions
def wayne_age_in_2021 : ℕ := 37
def wayne_birth_year : ℕ := 2021 - wayne_age_in_2021
def peter_birth_year : ℕ := wayne_birth_year - 3
def julia_birth_year : ℕ := peter_birth_year - 2

-- Theorem to prove
theorem julia_birth_year_is_1979 : julia_birth_year = 1979 := by
  sorry

end NUMINAMATH_GPT_julia_birth_year_is_1979_l997_99702


namespace NUMINAMATH_GPT_probability_of_rain_on_at_least_one_day_is_correct_l997_99715

def rain_on_friday_probability : ℝ := 0.30
def rain_on_saturday_probability : ℝ := 0.45
def rain_on_sunday_probability : ℝ := 0.50

def rain_on_at_least_one_day_probability : ℝ := 1 - (1 - rain_on_friday_probability) * (1 - rain_on_saturday_probability) * (1 - rain_on_sunday_probability)

theorem probability_of_rain_on_at_least_one_day_is_correct :
  rain_on_at_least_one_day_probability = 0.8075 := by
sorry

end NUMINAMATH_GPT_probability_of_rain_on_at_least_one_day_is_correct_l997_99715


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l997_99712

theorem common_ratio_geometric_sequence (q : ℝ) (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = q)
  (h₂ : a 3 = q^2)
  (h₃ : (4 * a 1 + a 3 = 2 * 2 * a 2)) :
  q = 2 :=
by sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l997_99712


namespace NUMINAMATH_GPT_leak_empties_tank_in_18_hours_l997_99776

theorem leak_empties_tank_in_18_hours :
  let A : ℚ := 1 / 6
  let L : ℚ := 1 / 6 - 1 / 9
  (1 / L) = 18 := by
    sorry

end NUMINAMATH_GPT_leak_empties_tank_in_18_hours_l997_99776


namespace NUMINAMATH_GPT_find_line_equation_l997_99709

theorem find_line_equation 
  (A : ℝ × ℝ) (hA : A = (-2, -3)) 
  (h_perpendicular : ∃ k b : ℝ, ∀ x y, 3 * x + 4 * y - 3 = 0 → k * x + y = b) :
  ∃ k' b' : ℝ, (∀ x y, k' * x + y = b' → y = (4 / 3) * x + 1 / 3) ∧ (k' = 4 ∧ b' = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_line_equation_l997_99709


namespace NUMINAMATH_GPT_find_E_l997_99706

variable (x E x1 x2 : ℝ)

/-- Given conditions as assumptions: -/
axiom h1 : (x + 3)^2 / E = 2
axiom h2 : x1 - x2 = 14

/-- Prove the required expression for E in terms of x: -/
theorem find_E : E = (x + 3)^2 / 2 := sorry

end NUMINAMATH_GPT_find_E_l997_99706


namespace NUMINAMATH_GPT_weaving_problem_solution_l997_99734

noncomputable def daily_increase :=
  let a1 := 5
  let n := 30
  let sum_total := 390
  let d := (sum_total - a1 * n) * 2 / (n * (n - 1))
  d

theorem weaving_problem_solution :
  daily_increase = 16 / 29 :=
by
  sorry

end NUMINAMATH_GPT_weaving_problem_solution_l997_99734


namespace NUMINAMATH_GPT_line_circle_intersection_common_points_l997_99774

noncomputable def radius (d : ℝ) := d / 2

theorem line_circle_intersection_common_points 
  (diameter : ℝ) (distance_from_center_to_line : ℝ) 
  (h_dlt_r : distance_from_center_to_line < radius diameter) :
  ∃ common_points : ℕ, common_points = 2 :=
by
  sorry

end NUMINAMATH_GPT_line_circle_intersection_common_points_l997_99774


namespace NUMINAMATH_GPT_coeff_a_zero_l997_99775

-- Define the problem in Lean 4

theorem coeff_a_zero (a b c : ℝ) (h : ∀ p : ℝ, 0 < p → ∀ x, a * x^2 + b * x + c + p = 0 → 0 < x) :
  a = 0 :=
sorry

end NUMINAMATH_GPT_coeff_a_zero_l997_99775


namespace NUMINAMATH_GPT_correct_calculation_l997_99797

theorem correct_calculation :
  (-2 * a * b^2)^3 = -8 * a^3 * b^6 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l997_99797


namespace NUMINAMATH_GPT_snake_price_correct_l997_99750

-- Define the conditions
def num_snakes : ℕ := 3
def eggs_per_snake : ℕ := 2
def total_eggs : ℕ := num_snakes * eggs_per_snake
def super_rare_multiple : ℕ := 4
def total_revenue : ℕ := 2250

-- The question: How much does each regular baby snake sell for?
def price_of_regular_baby_snake := 250

-- The proof statement
theorem snake_price_correct
  (x : ℕ)
  (h1 : total_eggs = 6)
  (h2 : 5 * x + super_rare_multiple * x = total_revenue)
  :
  x = price_of_regular_baby_snake := 
sorry

end NUMINAMATH_GPT_snake_price_correct_l997_99750


namespace NUMINAMATH_GPT_tyson_age_l997_99784

noncomputable def age_proof : Prop :=
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  t = 20           -- Statement that needs to be proved

theorem tyson_age : age_proof :=
by
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  show t = 20
  sorry

end NUMINAMATH_GPT_tyson_age_l997_99784


namespace NUMINAMATH_GPT_tank_insulation_cost_l997_99710

theorem tank_insulation_cost (l w h : ℝ) (cost_per_sqft : ℝ) (SA : ℝ) (C : ℝ) 
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost_per_sqft : cost_per_sqft = 20) 
  (h_SA : SA = 2 * l * w + 2 * l * h + 2 * w * h)
  (h_C : C = SA * cost_per_sqft) :
  C = 1440 := 
by
  -- proof will be filled in here
  sorry

end NUMINAMATH_GPT_tank_insulation_cost_l997_99710


namespace NUMINAMATH_GPT_board_train_immediately_probability_l997_99790

-- Define conditions
def total_time : ℝ := 10
def favorable_time : ℝ := 1

-- Define the probability P(A) as favorable_time / total_time
noncomputable def probability_A : ℝ := favorable_time / total_time

-- State the proposition to prove that the probability is 1/10
theorem board_train_immediately_probability : probability_A = 1 / 10 :=
by sorry

end NUMINAMATH_GPT_board_train_immediately_probability_l997_99790


namespace NUMINAMATH_GPT_alice_bob_meet_l997_99779

theorem alice_bob_meet (n : ℕ) (h_n : n = 18) (alice_move : ℕ) (bob_move : ℕ)
  (h_alice : alice_move = 7) (h_bob : bob_move = 13) :
  ∃ k : ℕ, alice_move * k % n = (n - bob_move) * k % n :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_meet_l997_99779


namespace NUMINAMATH_GPT_equilateral_triangle_square_ratio_l997_99758

theorem equilateral_triangle_square_ratio (t s : ℕ) (h_t : 3 * t = 12) (h_s : 4 * s = 12) :
  t / s = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_square_ratio_l997_99758


namespace NUMINAMATH_GPT_maximum_profit_l997_99792

/-- 
Given:
- The fixed cost is 3000 (in thousand yuan).
- The revenue per hundred vehicles is 500 (in thousand yuan).
- The additional cost y is defined as follows:
  - y = 10*x^2 + 100*x for 0 < x < 40
  - y = 501*x + 10000/x - 4500 for x ≥ 40
  
Prove:
1. The profit S(x) (in thousand yuan) in 2020 is:
   - S(x) = -10*x^2 + 400*x - 3000 for 0 < x < 40
   - S(x) = 1500 - x - 10000/x for x ≥ 40
2. The production volume x (in hundreds of vehicles) to achieve the maximum profit is 100,
   and the maximum profit is 1300 (in thousand yuan).
-/
noncomputable def profit_function (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 40) then
    -10 * x^2 + 400 * x - 3000
  else if (x ≥ 40) then
    1500 - x - 10000 / x
  else
    0 -- Undefined for other values, though our x will always be positive in our case

theorem maximum_profit : ∃ x : ℝ, 0 < x ∧ 
  (profit_function x = 1300 ∧ x = 100) ∧
  ∀ y, 0 < y → profit_function y ≤ 1300 :=
sorry

end NUMINAMATH_GPT_maximum_profit_l997_99792


namespace NUMINAMATH_GPT_total_marks_is_275_l997_99724

-- Definitions of scores in each subject
def science_score : ℕ := 70
def music_score : ℕ := 80
def social_studies_score : ℕ := 85
def physics_score : ℕ := music_score / 2

-- Definition of total marks
def total_marks : ℕ := science_score + music_score + social_studies_score + physics_score

-- Theorem to prove that total marks is 275
theorem total_marks_is_275 : total_marks = 275 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_total_marks_is_275_l997_99724


namespace NUMINAMATH_GPT_tom_buys_papayas_l997_99723

-- Defining constants for the costs of each fruit
def lemon_cost : ℕ := 2
def papaya_cost : ℕ := 1
def mango_cost : ℕ := 4

-- Defining the number of each fruit Tom buys
def lemons_bought : ℕ := 6
def mangos_bought : ℕ := 2
def total_paid : ℕ := 21

-- Defining the function to calculate the total cost 
def total_cost (P : ℕ) : ℕ := (lemons_bought * lemon_cost) + (mangos_bought * mango_cost) + (P * papaya_cost)

-- Defining the function to calculate the discount based on the total number of fruits
def discount (P : ℕ) : ℕ := (lemons_bought + mangos_bought + P) / 4

-- Main theorem to prove
theorem tom_buys_papayas (P : ℕ) : total_cost P - discount P = total_paid → P = 4 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_tom_buys_papayas_l997_99723


namespace NUMINAMATH_GPT_original_profit_margin_l997_99765

theorem original_profit_margin (x : ℝ) (h1 : x - 0.9 / 0.9 = 12 / 100) : (x - 1) / 1 * 100 = 8 :=
by
  sorry

end NUMINAMATH_GPT_original_profit_margin_l997_99765


namespace NUMINAMATH_GPT_integer_satisfaction_l997_99751

theorem integer_satisfaction (x : ℤ) : 
  (x + 15 ≥ 16 ∧ -3 * x ≥ -15) ↔ (1 ≤ x ∧ x ≤ 5) :=
by 
  sorry

end NUMINAMATH_GPT_integer_satisfaction_l997_99751


namespace NUMINAMATH_GPT_small_seats_capacity_l997_99769

-- Definitions
def num_small_seats : ℕ := 2
def people_per_small_seat : ℕ := 14

-- Statement to prove
theorem small_seats_capacity :
  num_small_seats * people_per_small_seat = 28 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_small_seats_capacity_l997_99769


namespace NUMINAMATH_GPT_residential_ratio_l997_99740

theorem residential_ratio (B R O E : ℕ) (h1 : B = 300) (h2 : E = 75) (h3 : E = O ∧ R + 2 * E = B) : R / B = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_residential_ratio_l997_99740


namespace NUMINAMATH_GPT_perpendicular_pair_is_14_l997_99789

variable (x y : ℝ)

def equation1 := 4 * y - 3 * x = 16
def equation2 := -3 * x - 4 * y = 15
def equation3 := 4 * y + 3 * x = 16
def equation4 := 3 * y + 4 * x = 15

theorem perpendicular_pair_is_14 : (∃ y1 y2 x1 x2 : ℝ,
  4 * y1 - 3 * x1 = 16 ∧ 3 * y2 + 4 * x2 = 15 ∧ (3 / 4) * (-4 / 3) = -1) :=
sorry

end NUMINAMATH_GPT_perpendicular_pair_is_14_l997_99789


namespace NUMINAMATH_GPT_candy_division_l997_99708

theorem candy_division 
  (total_candy : ℕ)
  (total_bags : ℕ)
  (candies_per_bag : ℕ)
  (chocolate_heart_bags : ℕ)
  (fruit_jelly_bags : ℕ)
  (caramel_chew_bags : ℕ) 
  (H1 : total_candy = 260)
  (H2 : total_bags = 13)
  (H3 : candies_per_bag = total_candy / total_bags)
  (H4 : chocolate_heart_bags = 4)
  (H5 : fruit_jelly_bags = 3)
  (H6 : caramel_chew_bags = total_bags - chocolate_heart_bags - fruit_jelly_bags)
  (H7 : candies_per_bag = 20) :
  (chocolate_heart_bags * candies_per_bag) + 
  (fruit_jelly_bags * candies_per_bag) + 
  (caramel_chew_bags * candies_per_bag) = 260 :=
sorry

end NUMINAMATH_GPT_candy_division_l997_99708


namespace NUMINAMATH_GPT_problem_statement_l997_99747

noncomputable def inequality_not_necessarily_true (a b c : ℝ) :=
  c < b ∧ b < a ∧ a * c < 0

theorem problem_statement (a b c : ℝ) (h : inequality_not_necessarily_true a b c) : ¬ (∃ a b c : ℝ, c < b ∧ b < a ∧ a * c < 0 ∧ ¬ (b^2/c > a^2/c)) :=
by sorry

end NUMINAMATH_GPT_problem_statement_l997_99747


namespace NUMINAMATH_GPT_five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l997_99736

-- Prove that the number of five-digit numbers is 27216
theorem five_digit_numbers_count : ∃ n, n = 9 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the number of five-digit numbers greater than or equal to 30000 is 21168
theorem five_digit_numbers_ge_30000 : 
  ∃ n, n = 7 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the rank of 50124 among five-digit numbers with distinct digits in descending order is 15119
theorem rank_of_50124 : 
  ∃ n, n = (Nat.factorial 5) - 1 := by
  sorry

end NUMINAMATH_GPT_five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l997_99736


namespace NUMINAMATH_GPT_sarah_proof_l997_99730

-- Defining cards and conditions
inductive Card
| P : Card
| A : Card
| C5 : Card
| C4 : Card
| C7 : Card

-- Definition of vowel
def is_vowel : Card → Prop
| Card.P => false
| Card.A => true
| _ => false

-- Definition of prime numbers for the sides
def is_prime : Card → Prop
| Card.C5 => true
| Card.C4 => false
| Card.C7 => true
| _ => false

-- Tom's statement
def toms_statement (c : Card) : Prop :=
is_vowel c → is_prime c

-- Sarah shows Tom was wrong by turning over one card
theorem sarah_proof : ∃ c, toms_statement c = false ∧ c = Card.A :=
sorry

end NUMINAMATH_GPT_sarah_proof_l997_99730


namespace NUMINAMATH_GPT_domain_of_ratio_function_l997_99748

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f (2 ^ x)

theorem domain_of_ratio_function (D : Set ℝ) (hD : D = Set.Icc 1 2):
  ∀ f : ℝ → ℝ, (∀ x, g x = f (2 ^ x)) →
  ∃ D' : Set ℝ, D' = {x | 2 ≤ x ∧ x ≤ 4} →
  ∀ y : ℝ, (2 ≤ y ∧ y ≤ 4) → ∃ x : ℝ, y = x + 1 ∧ x ≠ 1 → (1 < x ∧ x ≤ 3) :=
sorry

end NUMINAMATH_GPT_domain_of_ratio_function_l997_99748


namespace NUMINAMATH_GPT_correct_assignment_statement_l997_99731

theorem correct_assignment_statement (n m : ℕ) : 
  ¬ (4 = n) ∧ ¬ (n + 1 = m) ∧ ¬ (m + n = 0) :=
by
  sorry

end NUMINAMATH_GPT_correct_assignment_statement_l997_99731


namespace NUMINAMATH_GPT_no_obtuse_equilateral_triangle_exists_l997_99719

theorem no_obtuse_equilateral_triangle_exists :
  ¬(∃ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = π ∧ a > π/2 ∧ b > π/2 ∧ c > π/2) :=
sorry

end NUMINAMATH_GPT_no_obtuse_equilateral_triangle_exists_l997_99719


namespace NUMINAMATH_GPT_find_integer_solutions_l997_99738

theorem find_integer_solutions :
  (a b : ℤ) →
  3 * a^2 * b^2 + b^2 = 517 + 30 * a^2 →
  (a = 2 ∧ b = 7) ∨ (a = -2 ∧ b = 7) ∨ (a = 2 ∧ b = -7) ∨ (a = -2 ∧ b = -7) :=
sorry

end NUMINAMATH_GPT_find_integer_solutions_l997_99738


namespace NUMINAMATH_GPT_average_last_two_numbers_l997_99704

theorem average_last_two_numbers (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d + e + f + g) / 7 = 63) 
  (h2 : (a + b + c) / 3 = 58) 
  (h3 : (d + e) / 2 = 70) :
  ((f + g) / 2) = 63.5 := 
sorry

end NUMINAMATH_GPT_average_last_two_numbers_l997_99704


namespace NUMINAMATH_GPT_find_cost_price_l997_99793

theorem find_cost_price (C : ℝ) (h1 : 0.88 * C + 1500 = 1.12 * C) : C = 6250 := 
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l997_99793


namespace NUMINAMATH_GPT_expected_winnings_correct_l997_99786

def winnings (roll : ℕ) : ℚ :=
  if roll % 2 = 1 then 0
  else if roll % 4 = 0 then 2 * roll
  else roll

def expected_winnings : ℚ :=
  (winnings 1) / 8 + (winnings 2) / 8 +
  (winnings 3) / 8 + (winnings 4) / 8 +
  (winnings 5) / 8 + (winnings 6) / 8 +
  (winnings 7) / 8 + (winnings 8) / 8

theorem expected_winnings_correct : expected_winnings = 3.75 := by 
  sorry

end NUMINAMATH_GPT_expected_winnings_correct_l997_99786


namespace NUMINAMATH_GPT_least_number_of_apples_l997_99770

theorem least_number_of_apples (b : ℕ) : (b % 3 = 2) → (b % 4 = 3) → (b % 5 = 1) → b = 11 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_least_number_of_apples_l997_99770


namespace NUMINAMATH_GPT_total_screens_sold_is_45000_l997_99737

-- Define the number of screens sold in each month based on X
variables (X : ℕ)

-- Conditions given in the problem
def screens_in_January := X
def screens_in_February := 2 * X
def screens_in_March := (screens_in_January X + screens_in_February X) / 2
def screens_in_April := min (2 * screens_in_March X) 20000

-- Given that April sales were 18000
axiom apr_sales_18000 : screens_in_April X = 18000

-- Total sales is the sum of sales from January to April
def total_sales := screens_in_January X + screens_in_February X + screens_in_March X + 18000

-- Prove that total sales is 45000
theorem total_screens_sold_is_45000 : total_sales X = 45000 :=
by sorry

end NUMINAMATH_GPT_total_screens_sold_is_45000_l997_99737


namespace NUMINAMATH_GPT_x_div_11p_is_integer_l997_99777

theorem x_div_11p_is_integer (x p : ℕ) (h1 : x > 0) (h2 : Prime p) (h3 : x = 66) : ∃ k : ℤ, x / (11 * p) = k := by
  sorry

end NUMINAMATH_GPT_x_div_11p_is_integer_l997_99777


namespace NUMINAMATH_GPT_vivians_mail_in_august_l997_99791

-- Definitions based on the conditions provided
def mail_july : ℕ := 40
def business_days_august : ℕ := 22
def weekend_days_august : ℕ := 9

-- Lean 4 statement to prove the equivalent proof problem
theorem vivians_mail_in_august :
  let mail_business_days := 2 * mail_july
  let total_mail_business_days := business_days_august * mail_business_days
  let mail_weekend_days := mail_july / 2
  let total_mail_weekend_days := weekend_days_august * mail_weekend_days
  total_mail_business_days + total_mail_weekend_days = 1940 := by
  sorry

end NUMINAMATH_GPT_vivians_mail_in_august_l997_99791


namespace NUMINAMATH_GPT_length_increase_percentage_l997_99754

theorem length_increase_percentage (L B : ℝ) (x : ℝ) (h1 : (L + (x / 100) * L) * (B - (5 / 100) * B) = 1.14 * L * B) : x = 20 := by 
  sorry

end NUMINAMATH_GPT_length_increase_percentage_l997_99754


namespace NUMINAMATH_GPT_no_natural_numbers_satisfy_equation_l997_99788

theorem no_natural_numbers_satisfy_equation :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y + x + y = 2019 :=
by
  sorry

end NUMINAMATH_GPT_no_natural_numbers_satisfy_equation_l997_99788


namespace NUMINAMATH_GPT_ryan_learning_hours_l997_99757

theorem ryan_learning_hours (total_hours : ℕ) (chinese_hours : ℕ) (english_hours : ℕ) 
  (h1 : total_hours = 3) (h2 : chinese_hours = 1) : 
  english_hours = 2 :=
by 
  sorry

end NUMINAMATH_GPT_ryan_learning_hours_l997_99757


namespace NUMINAMATH_GPT_maximize_profit_correct_l997_99753

noncomputable def maximize_profit : ℝ × ℝ :=
  let initial_selling_price : ℝ := 50
  let purchase_price : ℝ := 40
  let initial_sales_volume : ℝ := 500
  let sales_volume_decrease_rate : ℝ := 10
  let x := 20
  let optimal_selling_price := initial_selling_price + x
  let maximum_profit := -10 * x^2 + 400 * x + 5000
  (optimal_selling_price, maximum_profit)

theorem maximize_profit_correct :
  maximize_profit = (70, 9000) :=
  sorry

end NUMINAMATH_GPT_maximize_profit_correct_l997_99753


namespace NUMINAMATH_GPT_length_of_second_train_is_correct_l997_99722

-- Define the known values and conditions
def speed_train1_kmph := 120
def speed_train2_kmph := 80
def length_train1_m := 280
def crossing_time_s := 9

-- Convert speeds from km/h to m/s
def kmph_to_mps (kmph : ℕ) : ℚ := kmph * 1000 / 3600

def speed_train1_mps := kmph_to_mps speed_train1_kmph
def speed_train2_mps := kmph_to_mps speed_train2_kmph

-- Calculate relative speed
def relative_speed_mps := speed_train1_mps + speed_train2_mps

-- Calculate total distance covered when crossing
def total_distance_m := relative_speed_mps * crossing_time_s

-- The length of the second train
def length_train2_m := total_distance_m - length_train1_m

-- Prove the length of the second train
theorem length_of_second_train_is_correct : length_train2_m = 219.95 := by {
  sorry
}

end NUMINAMATH_GPT_length_of_second_train_is_correct_l997_99722


namespace NUMINAMATH_GPT_proof_problem_l997_99700

def star (a b : ℕ) : ℕ := a - a / b

theorem proof_problem : star 18 6 + 2 * 6 = 27 := 
by
  admit  -- proof goes here

end NUMINAMATH_GPT_proof_problem_l997_99700


namespace NUMINAMATH_GPT_exists_a_div_by_3_l997_99778

theorem exists_a_div_by_3 (a : ℝ) (h : ∀ n : ℕ, ∃ k : ℤ, a * n * (n + 2) * (n + 4) = k) :
  ∃ k : ℤ, a = k / 3 :=
by
  sorry

end NUMINAMATH_GPT_exists_a_div_by_3_l997_99778


namespace NUMINAMATH_GPT_altitude_length_of_right_triangle_l997_99732

theorem altitude_length_of_right_triangle 
    (a b c : ℝ) 
    (h1 : a = 8) 
    (h2 : b = 15) 
    (h3 : c = 17) 
    (h4 : a^2 + b^2 = c^2) 
    : (2 * (1/2 * a * b))/c = 120/17 := 
by {
  sorry
}

end NUMINAMATH_GPT_altitude_length_of_right_triangle_l997_99732


namespace NUMINAMATH_GPT_find_n_from_digits_sum_l997_99721

theorem find_n_from_digits_sum (n : ℕ) (h1 : 777 = (9 * 1) + ((99 - 10 + 1) * 2) + (n - 99) * 3) : n = 295 :=
sorry

end NUMINAMATH_GPT_find_n_from_digits_sum_l997_99721


namespace NUMINAMATH_GPT_car_time_passed_l997_99714

variable (speed : ℝ) (distance : ℝ) (time_passed : ℝ)

theorem car_time_passed (h_speed : speed = 2) (h_distance : distance = 2) :
  time_passed = distance / speed := by
  rw [h_speed, h_distance]
  norm_num
  sorry

end NUMINAMATH_GPT_car_time_passed_l997_99714


namespace NUMINAMATH_GPT_nails_needed_l997_99799

-- Define the number of nails needed for each plank
def nails_per_plank : ℕ := 2

-- Define the number of planks used by John
def planks_used : ℕ := 16

-- The total number of nails needed.
theorem nails_needed : (nails_per_plank * planks_used) = 32 :=
by
  -- Our goal is to prove that nails_per_plank * planks_used = 32
  sorry

end NUMINAMATH_GPT_nails_needed_l997_99799


namespace NUMINAMATH_GPT_average_salary_difference_l997_99755

theorem average_salary_difference :
  let total_payroll_factory := 30000
  let num_factory_workers := 15
  let total_payroll_office := 75000
  let num_office_workers := 30
  (total_payroll_office / num_office_workers) - (total_payroll_factory / num_factory_workers) = 500 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_difference_l997_99755


namespace NUMINAMATH_GPT_evaluate_expression_l997_99766

theorem evaluate_expression (a x : ℤ) (h : x = a + 5) : 2 * x - a + 4 = a + 14 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l997_99766


namespace NUMINAMATH_GPT_smallest_y_l997_99701

theorem smallest_y (y : ℤ) :
  (∃ k : ℤ, y^2 + 3*y + 7 = k*(y-2)) ↔ y = -15 :=
sorry

end NUMINAMATH_GPT_smallest_y_l997_99701


namespace NUMINAMATH_GPT_simplify_abs_expression_l997_99739

theorem simplify_abs_expression (x : ℝ) : 
  |2*x + 1| - |x - 3| + |x - 6| = 
  if x < -1/2 then -2*x + 2 
  else if x < 3 then 2*x + 4 
  else if x < 6 then 10 
  else 2*x - 2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_abs_expression_l997_99739


namespace NUMINAMATH_GPT_f_is_n_l997_99703

noncomputable def f : ℕ+ → ℤ :=
  sorry

def f_defined_for_all_positive_integers (n : ℕ+) : Prop :=
  ∃ k, f n = k

def f_is_integer (n : ℕ+) : Prop :=
  ∃ k : ℤ, f n = k

def f_two_is_two : Prop :=
  f 2 = 2

def f_multiply_rule (m n : ℕ+) : Prop :=
  f (m * n) = f m * f n

def f_ordered (m n : ℕ+) (h : m > n) : Prop :=
  f m > f n

theorem f_is_n (n : ℕ+) :
  (f_defined_for_all_positive_integers n) →
  (f_is_integer n) →
  (f_two_is_two) →
  (∀ m n, f_multiply_rule m n) →
  (∀ m n (h : m > n), f_ordered m n h) →
  f n = n :=
sorry

end NUMINAMATH_GPT_f_is_n_l997_99703


namespace NUMINAMATH_GPT_number_of_people_l997_99785

def avg_weight_increase : ℝ := 2.5
def old_person_weight : ℝ := 45
def new_person_weight : ℝ := 65

theorem number_of_people (n : ℕ) 
  (h1 : avg_weight_increase = 2.5) 
  (h2 : old_person_weight = 45) 
  (h3 : new_person_weight = 65) :
  n = 8 :=
  sorry

end NUMINAMATH_GPT_number_of_people_l997_99785


namespace NUMINAMATH_GPT_max_subset_size_l997_99749

theorem max_subset_size :
  ∃ S : Finset ℕ, (∀ (x y : ℕ), x ∈ S → y ∈ S → y ≠ 2 * x) →
  S.card = 1335 :=
sorry

end NUMINAMATH_GPT_max_subset_size_l997_99749


namespace NUMINAMATH_GPT_mod_add_l997_99756

theorem mod_add (n : ℕ) (h : n % 5 = 3) : (n + 2025) % 5 = 3 := by
  sorry

end NUMINAMATH_GPT_mod_add_l997_99756


namespace NUMINAMATH_GPT_MaxCandy_l997_99741

theorem MaxCandy (frankieCandy : ℕ) (extraCandy : ℕ) (maxCandy : ℕ) 
  (h1 : frankieCandy = 74) (h2 : extraCandy = 18) (h3 : maxCandy = frankieCandy + extraCandy) :
  maxCandy = 92 := 
by
  sorry

end NUMINAMATH_GPT_MaxCandy_l997_99741


namespace NUMINAMATH_GPT_ROI_diff_after_2_years_is_10_l997_99773

variables (investment_Emma : ℝ) (investment_Briana : ℝ)
variables (yield_Emma : ℝ) (yield_Briana : ℝ)
variables (years : ℝ)

def annual_ROI_Emma (investment_Emma yield_Emma : ℝ) : ℝ :=
  yield_Emma * investment_Emma

def annual_ROI_Briana (investment_Briana yield_Briana : ℝ) : ℝ :=
  yield_Briana * investment_Briana

def total_ROI_Emma (investment_Emma yield_Emma years : ℝ) : ℝ :=
  annual_ROI_Emma investment_Emma yield_Emma * years

def total_ROI_Briana (investment_Briana yield_Briana years : ℝ) : ℝ :=
  annual_ROI_Briana investment_Briana yield_Briana * years

def ROI_difference (investment_Emma investment_Briana yield_Emma yield_Briana years : ℝ) : ℝ :=
  total_ROI_Briana investment_Briana yield_Briana years - total_ROI_Emma investment_Emma yield_Emma years

theorem ROI_diff_after_2_years_is_10 :
  ROI_difference 300 500 0.15 0.10 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_ROI_diff_after_2_years_is_10_l997_99773


namespace NUMINAMATH_GPT_min_value_problem_l997_99752

theorem min_value_problem 
  (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ (min_val : ℝ), min_val = 2 * x + 3 * y^2 ∧ min_val = 8 / 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_problem_l997_99752


namespace NUMINAMATH_GPT_sequence_formula_l997_99717

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 2 → a n = 3 * a (n - 1) + 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^n - 2 :=
by 
sorry

end NUMINAMATH_GPT_sequence_formula_l997_99717


namespace NUMINAMATH_GPT_central_cell_value_l997_99746

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end NUMINAMATH_GPT_central_cell_value_l997_99746


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l997_99772

noncomputable def M : Set ℝ := {x | x - 2 > 0}
noncomputable def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

theorem intersection_of_M_and_N :
  M ∩ N = {x | x > 2} :=
sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l997_99772


namespace NUMINAMATH_GPT_grasshopper_frog_jump_difference_l997_99733

theorem grasshopper_frog_jump_difference :
  let grasshopper_jump := 19
  let frog_jump := 15
  grasshopper_jump - frog_jump = 4 :=
by
  let grasshopper_jump := 19
  let frog_jump := 15
  sorry

end NUMINAMATH_GPT_grasshopper_frog_jump_difference_l997_99733


namespace NUMINAMATH_GPT_problem_sol_max_distance_from_circle_to_line_l997_99735

noncomputable def max_distance_circle_line : ℝ :=
  let ρ (θ : ℝ) : ℝ := 8 * Real.sin θ
  let line (θ : ℝ) : Prop := θ = Real.pi / 3
  let circle_center := (0, 4)
  let line_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x
  let shortest_distance := 2  -- Already calculated in solution
  let radius := 4
  shortest_distance + radius

theorem problem_sol_max_distance_from_circle_to_line :
  max_distance_circle_line = 6 :=
by
  unfold max_distance_circle_line
  sorry

end NUMINAMATH_GPT_problem_sol_max_distance_from_circle_to_line_l997_99735
