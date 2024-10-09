import Mathlib

namespace digits_arithmetic_l499_49974

theorem digits_arithmetic :
  (12 / 3 / 4) * (56 / 7 / 8) = 1 :=
by
  sorry

end digits_arithmetic_l499_49974


namespace inscribed_squares_equilateral_triangle_l499_49975

theorem inscribed_squares_equilateral_triangle (a b c h_a h_b h_c : ℝ) 
  (h1 : a * h_a / (a + h_a) = b * h_b / (b + h_b))
  (h2 : b * h_b / (b + h_b) = c * h_c / (c + h_c)) :
  a = b ∧ b = c ∧ h_a = h_b ∧ h_b = h_c :=
sorry

end inscribed_squares_equilateral_triangle_l499_49975


namespace third_candidate_votes_l499_49983

-- Definition of the problem's conditions
variables (total_votes winning_votes candidate2_votes : ℕ)
variables (winning_percentage : ℚ)

-- Conditions given in the problem
def conditions : Prop :=
  winning_votes = 11628 ∧
  winning_percentage = 0.4969230769230769 ∧
  (total_votes : ℚ) = winning_votes / winning_percentage ∧
  candidate2_votes = 7636

-- The theorem we need to prove
theorem third_candidate_votes (total_votes winning_votes candidate2_votes : ℕ)
    (winning_percentage : ℚ)
    (h : conditions total_votes winning_votes candidate2_votes winning_percentage) :
    total_votes - (winning_votes + candidate2_votes) = 4136 := 
  sorry

end third_candidate_votes_l499_49983


namespace a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l499_49990

noncomputable def a_0 (n : ℕ) : ℕ := 2^n
noncomputable def S_n (n : ℕ) : ℕ := 3^n - 2^n
noncomputable def T_n (n : ℕ) : ℕ := (n - 2) * 2^n + 2 * n^2

theorem a_0_eq_2_pow_n (n : ℕ) (h : n > 0) : a_0 n = 2^n := sorry

theorem S_n_eq_3_pow_n_minus_2_pow_n (n : ℕ) (h : n > 0) : S_n n = 3^n - 2^n := sorry

theorem S_n_magnitude_comparison : 
  ∀ (n : ℕ), 
    (n = 1 → S_n n > T_n n) ∧
    (n = 2 ∨ n = 3 → S_n n < T_n n) ∧
    (n ≥ 4 → S_n n > T_n n) := sorry

end a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l499_49990


namespace mike_total_spending_l499_49905

def mike_spent_on_speakers : ℝ := 235.87
def mike_spent_on_tires : ℝ := 281.45
def mike_spent_on_steering_wheel_cover : ℝ := 179.99
def mike_spent_on_seat_covers : ℝ := 122.31
def mike_spent_on_headlights : ℝ := 98.63

theorem mike_total_spending :
  mike_spent_on_speakers + mike_spent_on_tires + mike_spent_on_steering_wheel_cover + mike_spent_on_seat_covers + mike_spent_on_headlights = 918.25 :=
  sorry

end mike_total_spending_l499_49905


namespace count_powers_of_2_not_4_under_2000000_l499_49953

theorem count_powers_of_2_not_4_under_2000000 :
  ∃ n, ∀ x, x < 2000000 → (∃ k, x = 2 ^ k ∧ (∀ m, x ≠ 4 ^ m)) ↔ x > 0 ∧ x < 2 ^ (n + 1) := by
  sorry

end count_powers_of_2_not_4_under_2000000_l499_49953


namespace non_zero_real_solution_l499_49904

theorem non_zero_real_solution (x : ℝ) (hx : x ≠ 0) (h : (3 * x)^5 = (9 * x)^4) : x = 27 :=
sorry

end non_zero_real_solution_l499_49904


namespace seconds_in_3_hours_45_minutes_l499_49968

theorem seconds_in_3_hours_45_minutes :
  let hours := 3
  let minutes := 45
  let minutes_in_hour := 60
  let seconds_in_minute := 60
  (hours * minutes_in_hour + minutes) * seconds_in_minute = 13500 := by
  sorry

end seconds_in_3_hours_45_minutes_l499_49968


namespace solution_proof_l499_49961

noncomputable def problem_statement : Prop :=
  ((16^(1/4) * 32^(1/5)) + 64^(1/6)) = 6

theorem solution_proof : problem_statement :=
by
  sorry

end solution_proof_l499_49961


namespace tiles_on_square_area_l499_49917

theorem tiles_on_square_area (n : ℕ) (h1 : 2 * n - 1 = 25) : n ^ 2 = 169 :=
by
  sorry

end tiles_on_square_area_l499_49917


namespace find_a_for_parabola_l499_49992

theorem find_a_for_parabola (a : ℝ) :
  (∃ y : ℝ, y = a * (-1 / 2)^2) → a = 1 / 2 :=
by
  sorry

end find_a_for_parabola_l499_49992


namespace A_wins_if_perfect_square_or_prime_l499_49944

theorem A_wins_if_perfect_square_or_prime (n : ℕ) (h_pos : 0 < n) : 
  (∃ A_wins : Bool, A_wins = true ↔ (∃ k : ℕ, n = k^2) ∨ (∃ p : ℕ, Nat.Prime p ∧ n = p)) :=
by
  sorry

end A_wins_if_perfect_square_or_prime_l499_49944


namespace solve_diophantine_eq_l499_49936

theorem solve_diophantine_eq (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^2 = b * (b + 7) ↔ (a = 12 ∧ b = 9) ∨ (a = 0 ∧ b = 0) := 
by 
  sorry

end solve_diophantine_eq_l499_49936


namespace vertices_after_removal_l499_49935

theorem vertices_after_removal (a b : ℕ) (h₁ : a = 5) (h₂ : b = 2) : 
  let initial_vertices := 8
  let removed_vertices := initial_vertices
  let new_vertices := 8 * 9
  let final_vertices := new_vertices - removed_vertices
  final_vertices = 64 :=
by
  sorry

end vertices_after_removal_l499_49935


namespace garden_perimeter_is_64_l499_49923

theorem garden_perimeter_is_64 :
    ∀ (width_garden length_garden width_playground length_playground : ℕ),
    width_garden = 24 →
    width_playground = 12 →
    length_playground = 16 →
    width_playground * length_playground = width_garden * length_garden →
    2 * length_garden + 2 * width_garden = 64 :=
by
  intros width_garden length_garden width_playground length_playground
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end garden_perimeter_is_64_l499_49923


namespace n_minus_m_eq_200_l499_49928

-- Define the parameters
variable (m n x : ℝ)

-- State the conditions
def condition1 : Prop := m ≤ 8 * x - 1 ∧ 8 * x - 1 ≤ n 
def condition2 : Prop := (n + 1)/8 - (m + 1)/8 = 25

-- State the theorem to prove
theorem n_minus_m_eq_200 (h1 : condition1 m n x) (h2 : condition2 m n) : n - m = 200 := 
by 
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end n_minus_m_eq_200_l499_49928


namespace symmetric_point_origin_l499_49929

theorem symmetric_point_origin (A : ℝ × ℝ) (A_sym : ℝ × ℝ) (h : A = (3, -2)) (h_sym : A_sym = (-A.1, -A.2)) : A_sym = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l499_49929


namespace total_prizes_l499_49922

-- Definitions of the conditions
def stuffedAnimals : ℕ := 14
def frisbees : ℕ := 18
def yoYos : ℕ := 18

-- The statement to be proved
theorem total_prizes : stuffedAnimals + frisbees + yoYos = 50 := by
  sorry

end total_prizes_l499_49922


namespace series_sum_eq_l499_49954

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * ↑n + 3) / ((4 * ↑n + 1)^2 * (4 * ↑n + 5)^2)

theorem series_sum_eq :
  (∑' n, series_term n) = 1 / 800 :=
by
  sorry

end series_sum_eq_l499_49954


namespace sufficient_but_not_necessary_condition_l499_49939

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1) → (|x + 1| + |x - 1| = 2 * |x|) ∧ ¬((x ≥ 1) ↔ (|x + 1| + |x - 1| = 2 * |x|)) := by
  sorry

end sufficient_but_not_necessary_condition_l499_49939


namespace negation_of_squared_inequality_l499_49914

theorem negation_of_squared_inequality (p : ∀ n : ℕ, n^2 ≤ 2*n + 5) : 
  ∃ n : ℕ, n^2 > 2*n + 5 :=
sorry

end negation_of_squared_inequality_l499_49914


namespace multiply_and_divide_equiv_l499_49915

/-- Defines the operation of first multiplying by 4/5 and then dividing by 4/7 -/
def multiply_and_divide (x : ℚ) : ℚ :=
  (x * (4 / 5)) / (4 / 7)

/-- Statement to prove the operation is equivalent to multiplying by 7/5 -/
theorem multiply_and_divide_equiv (x : ℚ) : 
  multiply_and_divide x = x * (7 / 5) :=
by 
  -- This requires a proof, which we can assume here
  sorry

end multiply_and_divide_equiv_l499_49915


namespace smallest_perfect_square_divisible_by_2_3_5_l499_49913

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l499_49913


namespace average_speed_second_half_l499_49908

theorem average_speed_second_half
  (d : ℕ) (s1 : ℕ) (t : ℕ)
  (h1 : d = 3600)
  (h2 : s1 = 90)
  (h3 : t = 30) :
  (d / 2) / (t - (d / 2 / s1)) = 180 := by
  sorry

end average_speed_second_half_l499_49908


namespace pizzas_returned_l499_49911

theorem pizzas_returned (total_pizzas served_pizzas : ℕ) (h_total : total_pizzas = 9) (h_served : served_pizzas = 3) : (total_pizzas - served_pizzas) = 6 :=
by
  sorry

end pizzas_returned_l499_49911


namespace problem_1_problem_2_l499_49969

def setP (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def setS (x : ℝ) (m : ℝ) : Prop := |x - 1| ≤ m

theorem problem_1 (m : ℝ) : (m ∈ Set.Iic (3)) → ∀ x, (setP x ∨ setS x m) → setP x := sorry

theorem problem_2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (setP x ↔ setS x m) := sorry

end problem_1_problem_2_l499_49969


namespace prime_solution_unique_l499_49918

theorem prime_solution_unique {x y : ℕ} 
  (hx : Nat.Prime x)
  (hy : Nat.Prime y)
  (h : x ^ y - y ^ x = x * y ^ 2 - 19) :
  (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) :=
sorry

end prime_solution_unique_l499_49918


namespace point_in_second_quadrant_l499_49937

theorem point_in_second_quadrant (x : ℝ) (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
by
  sorry

end point_in_second_quadrant_l499_49937


namespace fraction_calculation_l499_49950

theorem fraction_calculation :
  let a := (1 / 2) + (1 / 3)
  let b := (2 / 7) + (1 / 4)
  ((a / b) * (3 / 5)) = (14 / 15) :=
by
  sorry

end fraction_calculation_l499_49950


namespace find_b_l499_49920

-- Definitions based on the conditions in the problem
def eq1 (a : ℝ) := 3 * a + 3 = 0
def eq2 (a b : ℝ) := 2 * b - a = 4

-- Statement of the proof problem
theorem find_b (a b : ℝ) (h1 : eq1 a) (h2 : eq2 a b) : b = 3 / 2 :=
by
  sorry

end find_b_l499_49920


namespace best_fitting_model_is_model_2_l499_49942

-- Variables representing the correlation coefficients of the four models
def R2_model_1 : ℝ := 0.86
def R2_model_2 : ℝ := 0.96
def R2_model_3 : ℝ := 0.73
def R2_model_4 : ℝ := 0.66

-- Statement asserting that Model 2 has the best fitting effect
theorem best_fitting_model_is_model_2 :
  R2_model_2 = 0.96 ∧ R2_model_2 > R2_model_1 ∧ R2_model_2 > R2_model_3 ∧ R2_model_2 > R2_model_4 :=
by {
  sorry
}

end best_fitting_model_is_model_2_l499_49942


namespace find_valid_N_l499_49965

def is_divisible_by_10_consec (N : ℕ) : Prop :=
  ∀ m : ℕ, (N % (List.prod (List.range' m 10)) = 0)

def is_not_divisible_by_11_consec (N : ℕ) : Prop :=
  ∀ m : ℕ, ¬ (N % (List.prod (List.range' m 11)) = 0)

theorem find_valid_N (N : ℕ) :
  (is_divisible_by_10_consec N ∧ is_not_divisible_by_11_consec N) ↔
  (∃ k : ℕ, (k > 0) ∧ ¬ (k % 11 = 0) ∧ N = k * Nat.factorial 10) :=
sorry

end find_valid_N_l499_49965


namespace chord_length_perpendicular_l499_49906

theorem chord_length_perpendicular 
  (R a b : ℝ)  
  (h1 : a + b = R)
  (h2 : (1 / 2) * Real.pi * R^2 - (1 / 2) * Real.pi * (a^2 + b^2) = 10 * Real.pi) :
  2 * Real.sqrt 10 = 6.32 :=
by 
  sorry

end chord_length_perpendicular_l499_49906


namespace max_tickets_l499_49955


theorem max_tickets (cost_regular : ℕ) (cost_discounted : ℕ) (threshold : ℕ) (total_money : ℕ) 
  (h1 : cost_regular = 15) 
  (h2 : cost_discounted = 12) 
  (h3 : threshold = 5)
  (h4 : total_money = 150) 
  : (total_money / cost_regular ≤ 10) ∧ 
    ((total_money - threshold * cost_regular) / cost_discounted + threshold = 11) :=
by
  sorry

end max_tickets_l499_49955


namespace value_of_a_l499_49981

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (5-x)/(x-2) ≥ 0 ↔ -3 < x ∧ x < a) → a > 5 :=
by
  intro h
  sorry

end value_of_a_l499_49981


namespace f_neg_one_value_l499_49946

theorem f_neg_one_value (f : ℝ → ℝ) (b : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x → f x = 2^x + 2 * x + b) :
  f (-1) = -3 := by
sorry

end f_neg_one_value_l499_49946


namespace Sasha_can_write_2011_l499_49970

theorem Sasha_can_write_2011 (N : ℕ) (hN : N > 1) : 
    ∃ (s : ℕ → ℕ), (s 0 = N) ∧ (∃ n, s n = 2011) ∧ 
    (∀ k, ∃ d, d > 1 ∧ (s (k + 1) = s k + d ∨ s (k + 1) = s k - d)) :=
sorry

end Sasha_can_write_2011_l499_49970


namespace ellipse_foci_distance_l499_49977

theorem ellipse_foci_distance 
  (h : ∀ x y : ℝ, 9 * x^2 + y^2 = 144) : 
  ∃ c : ℝ, c = 16 * Real.sqrt 2 :=
  sorry

end ellipse_foci_distance_l499_49977


namespace min_m_n_sum_l499_49958

theorem min_m_n_sum (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_eq : 45 * m = n^3) : m + n = 90 :=
sorry

end min_m_n_sum_l499_49958


namespace final_result_l499_49976

/-- A student chose a number, multiplied it by 5, then subtracted 138 
from the result. The number he chose was 48. What was the final result 
after subtracting 138? -/
theorem final_result (x : ℕ) (h1 : x = 48) : (x * 5) - 138 = 102 := by
  sorry

end final_result_l499_49976


namespace simplify_expr1_simplify_expr2_l499_49997

-- First expression
theorem simplify_expr1 (a b : ℝ) : a * (a - b) - (a + b) * (a - 2 * b) = 2 * b ^ 2 :=
by
  sorry

-- Second expression
theorem simplify_expr2 (x : ℝ) : 
  ( ( (4 * x - 9) / (3 - x) - x + 3 ) / ( (x ^ 2 - 4) / (x - 3) ) ) = - (x / (x + 2)) :=
by
  sorry

end simplify_expr1_simplify_expr2_l499_49997


namespace isosceles_perimeter_l499_49957

theorem isosceles_perimeter (peri_eqt : ℕ) (side_eqt : ℕ) (base_iso : ℕ) (side_iso : ℕ)
    (h1 : peri_eqt = 60)
    (h2 : side_eqt = peri_eqt / 3)
    (h3 : side_iso = side_eqt)
    (h4 : base_iso = 25) :
  2 * side_iso + base_iso = 65 :=
by
  sorry

end isosceles_perimeter_l499_49957


namespace more_sightings_than_triple_cape_may_l499_49960

def daytona_shark_sightings := 26
def cape_may_shark_sightings := 7

theorem more_sightings_than_triple_cape_may :
  daytona_shark_sightings - 3 * cape_may_shark_sightings = 5 :=
by
  sorry

end more_sightings_than_triple_cape_may_l499_49960


namespace total_distance_traveled_l499_49921

theorem total_distance_traveled (d d1 d2 d3 d4 d5 : ℕ) 
  (h1 : d1 = d)
  (h2 : d2 = 2 * d)
  (h3 : d3 = 40)
  (h4 : d = 2 * d3)
  (h5 : d4 = 2 * (d1 + d2 + d3))
  (h6 : d5 = 3 * d4 / 2) 
  : d1 + d2 + d3 + d4 + d5 = 1680 :=
by
  have hd : d = 80 := sorry
  have hd1 : d1 = 80 := sorry
  have hd2 : d2 = 160 := sorry
  have hd4 : d4 = 560 := sorry
  have hd5 : d5 = 840 := sorry
  sorry

end total_distance_traveled_l499_49921


namespace find_p_q_of_divisible_polynomial_l499_49964

theorem find_p_q_of_divisible_polynomial :
  ∃ p q : ℤ, (p, q) = (-7, -12) ∧
    (∀ x : ℤ, (x^5 - x^4 + x^3 - p*x^2 + q*x + 4 = 0) → (x = -2 ∨ x = 1)) :=
by
  sorry

end find_p_q_of_divisible_polynomial_l499_49964


namespace animal_group_divisor_l499_49925

theorem animal_group_divisor (cows sheep goats total groups : ℕ)
    (hc : cows = 24) 
    (hs : sheep = 7) 
    (hg : goats = 113) 
    (ht : total = cows + sheep + goats) 
    (htotal : total = 144) 
    (hdiv : groups ∣ total) 
    (hexclude1 : groups ≠ 1) 
    (hexclude144 : groups ≠ 144) : 
    ∃ g, g = groups ∧ g ∈ [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72] :=
  by 
  sorry

end animal_group_divisor_l499_49925


namespace arithmetic_sequence_a10_l499_49932

theorem arithmetic_sequence_a10 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 7 = 9) (h2 : a 13 = -3) 
  (ha : ∀ n, a n = a1 + (n - 1) * d) :
  a 10 = 3 :=
by sorry

end arithmetic_sequence_a10_l499_49932


namespace worker_bees_in_hive_l499_49952

variable (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ)

def finalWorkerBees (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ) : ℕ :=
  initialWorkerBees - leavingWorkerBees + returningWorkerBees

theorem worker_bees_in_hive
  (initialWorkerBees : ℕ := 400)
  (leavingWorkerBees : ℕ := 28)
  (returningWorkerBees : ℕ := 15) :
  finalWorkerBees initialWorkerBees leavingWorkerBees returningWorkerBees = 387 := by
  sorry

end worker_bees_in_hive_l499_49952


namespace find_x_squared_add_y_squared_l499_49967

noncomputable def x_squared_add_y_squared (x y : ℝ) : ℝ :=
  x^2 + y^2

theorem find_x_squared_add_y_squared (x y : ℝ) 
  (h1 : x + y = 48)
  (h2 : x * y = 168) :
  x_squared_add_y_squared x y = 1968 :=
by
  sorry

end find_x_squared_add_y_squared_l499_49967


namespace triangle_square_side_length_ratio_l499_49987

theorem triangle_square_side_length_ratio (t s : ℝ) (ht : 3 * t = 12) (hs : 4 * s = 12) : 
  t / s = 4 / 3 :=
by
  sorry

end triangle_square_side_length_ratio_l499_49987


namespace total_feed_amount_l499_49947

theorem total_feed_amount (x : ℝ) : 
  (17 * 0.18) + (x * 0.53) = (17 + x) * 0.36 → 17 + x = 35 :=
by
  intros h
  sorry

end total_feed_amount_l499_49947


namespace fraction_subtraction_l499_49956

theorem fraction_subtraction :
  (9 / 19) - (5 / 57) - (2 / 38) = 1 / 3 := by
sorry

end fraction_subtraction_l499_49956


namespace smallest_common_students_l499_49933

theorem smallest_common_students 
    (z : ℕ) (k : ℕ) (j : ℕ) 
    (hz : z = k ∧ k = j) 
    (hz_ratio : ∃ x : ℕ, z = 3 * x ∧ k = 2 * x ∧ j = 5 * x)
    (hz_group : ∃ y : ℕ, z = 14 * y) 
    (hk_group : ∃ w : ℕ, k = 10 * w) 
    (hj_group : ∃ v : ℕ, j = 15 * v) : 
    z = 630 ∧ k = 420 ∧ j = 1050 :=
    sorry

end smallest_common_students_l499_49933


namespace proof_of_A_inter_complement_B_l499_49999

variable (U : Set Nat) 
variable (A B : Set Nat)

theorem proof_of_A_inter_complement_B :
    (U = {1, 2, 3, 4}) →
    (B = {1, 2}) →
    (compl (A ∪ B) = {4}) →
    (A ∩ compl B = {3}) :=
by
  intros hU hB hCompl
  sorry

end proof_of_A_inter_complement_B_l499_49999


namespace sufficient_but_not_necessary_condition_l499_49980

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 0 → x^2 > 0) ∧ ¬(x^2 > 0 → x > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l499_49980


namespace number_of_pecan_pies_is_4_l499_49910

theorem number_of_pecan_pies_is_4 (apple_pies pumpkin_pies total_pies pecan_pies : ℕ) 
  (h1 : apple_pies = 2) 
  (h2 : pumpkin_pies = 7) 
  (h3 : total_pies = 13) 
  (h4 : pecan_pies = total_pies - (apple_pies + pumpkin_pies)) 
  : pecan_pies = 4 := 
by 
  sorry

end number_of_pecan_pies_is_4_l499_49910


namespace most_likely_outcome_is_draw_l499_49903

variable (P_A_wins : ℝ) (P_A_not_loses : ℝ)

def P_draw (P_A_wins P_A_not_loses : ℝ) : ℝ := 
  P_A_not_loses - P_A_wins

def P_B_wins (P_A_not_loses P_A_wins : ℝ) : ℝ :=
  1 - P_A_not_loses

theorem most_likely_outcome_is_draw 
  (h₁: P_A_wins = 0.3) 
  (h₂: P_A_not_loses = 0.7)
  (h₃: 0 ≤ P_A_wins) 
  (h₄: P_A_wins ≤ 1) 
  (h₅: 0 ≤ P_A_not_loses) 
  (h₆: P_A_not_loses ≤ 1) : 
  max (P_A_wins) (max (P_B_wins P_A_not_loses P_A_wins) (P_draw P_A_wins P_A_not_loses)) = P_draw P_A_wins P_A_not_loses :=
by
  sorry

end most_likely_outcome_is_draw_l499_49903


namespace compound_interest_calculation_l499_49949

theorem compound_interest_calculation :
  let P_SI := 1750.0000000000018
  let r_SI := 0.08
  let t_SI := 3
  let r_CI := 0.10
  let t_CI := 2
  let SI := P_SI * r_SI * t_SI
  let CI (P_CI : ℝ) := P_CI * ((1 + r_CI) ^ t_CI - 1)
  (SI = 420.0000000000004) →
  (SI = (1 / 2) * CI P_CI) →
  P_CI = 4000.000000000004 :=
by
  intros P_SI r_SI t_SI r_CI t_CI SI CI h1 h2
  sorry

end compound_interest_calculation_l499_49949


namespace sum_of_other_endpoint_coordinates_l499_49902

theorem sum_of_other_endpoint_coordinates (x y : ℤ)
  (h1 : (6 + x) / 2 = 3)
  (h2 : (-1 + y) / 2 = 6) :
  x + y = 13 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l499_49902


namespace original_days_l499_49985

-- Definitions based on the given problem conditions
def totalLaborers : ℝ := 17.5
def absentLaborers : ℝ := 7
def workingLaborers : ℝ := totalLaborers - absentLaborers
def workDaysByWorkingLaborers : ℝ := 10
def totalLaborDays : ℝ := workingLaborers * workDaysByWorkingLaborers

theorem original_days (D : ℝ) (h : totalLaborers * D = totalLaborDays) : D = 6 := sorry

end original_days_l499_49985


namespace relationship_among_abc_l499_49934

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 0.4
noncomputable def c : ℝ := 0.4 ^ 3

theorem relationship_among_abc : a > c ∧ c > b := by
  sorry

end relationship_among_abc_l499_49934


namespace find_other_endpoint_of_diameter_l499_49988

noncomputable def circle_center : (ℝ × ℝ) := (4, -2)
noncomputable def one_endpoint_of_diameter : (ℝ × ℝ) := (7, 5)
noncomputable def other_endpoint_of_diameter : (ℝ × ℝ) := (1, -9)

theorem find_other_endpoint_of_diameter :
  let (cx, cy) := circle_center
  let (x1, y1) := one_endpoint_of_diameter
  let (x2, y2) := other_endpoint_of_diameter
  (x2, y2) = (2 * cx - x1, 2 * cy - y1) :=
by
  sorry

end find_other_endpoint_of_diameter_l499_49988


namespace solution_set_ineq_l499_49973

theorem solution_set_ineq (m : ℝ) (hm : m > 1) :
  {x : ℝ | x^2 + (m-1) * x - m >= 0} = {x : ℝ | x <= -m ∨ x >= 1} :=
sorry

end solution_set_ineq_l499_49973


namespace garden_width_l499_49998

theorem garden_width (w : ℕ) (h1 : ∀ l : ℕ, l = w + 12 → l * w ≥ 120) : w = 6 := 
by
  sorry

end garden_width_l499_49998


namespace carina_total_coffee_l499_49996

def number_of_ten_ounce_packages : ℕ := 4
def number_of_five_ounce_packages : ℕ := number_of_ten_ounce_packages + 2
def ounces_in_each_ten_ounce_package : ℕ := 10
def ounces_in_each_five_ounce_package : ℕ := 5

def total_coffee_ounces : ℕ := 
  (number_of_ten_ounce_packages * ounces_in_each_ten_ounce_package) +
  (number_of_five_ounce_packages * ounces_in_each_five_ounce_package)

theorem carina_total_coffee : total_coffee_ounces = 70 := by
  -- proof to be provided
  sorry

end carina_total_coffee_l499_49996


namespace proof_problem_l499_49972

variable {x y : ℝ}

def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y = 1

theorem proof_problem (h : conditions x y) :
  x + y - 4 * x * y ≥ 0 ∧ (1 / x) + 4 / (1 + y) ≥ 9 / 2 :=
by
  sorry

end proof_problem_l499_49972


namespace small_cubes_with_two_faces_painted_l499_49993

theorem small_cubes_with_two_faces_painted :
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  12 * (n - 2) = 36 :=
by
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  exact sorry

end small_cubes_with_two_faces_painted_l499_49993


namespace one_fourth_in_one_eighth_l499_49948

theorem one_fourth_in_one_eighth : (1/8 : ℚ) / (1/4) = (1/2) := 
by
  sorry

end one_fourth_in_one_eighth_l499_49948


namespace f_inequality_l499_49912

def f (x : ℝ) : ℝ := sorry

axiom f_defined : ∀ x : ℝ, 0 < x → ∃ y : ℝ, f x = y

axiom f_additive : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y

axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

axiom f_two : f 2 = 1

theorem f_inequality (x : ℝ) : 3 < x → x ≤ 4 → f x + f (x - 3) ≤ 2 :=
sorry

end f_inequality_l499_49912


namespace total_pages_read_l499_49979

def pages_read_yesterday : ℕ := 21
def pages_read_today : ℕ := 17

theorem total_pages_read : pages_read_yesterday + pages_read_today = 38 :=
by
  sorry

end total_pages_read_l499_49979


namespace jimmy_sells_less_l499_49916

-- Definitions based on conditions
def num_figures : ℕ := 5
def value_figure_1_to_4 : ℕ := 15
def value_figure_5 : ℕ := 20
def total_earned : ℕ := 55

-- Formulation of the problem statement in Lean
theorem jimmy_sells_less (total_value : ℕ := (4 * value_figure_1_to_4) + value_figure_5) (difference : ℕ := total_value - total_earned) (amount_less_per_figure : ℕ := difference / num_figures) : amount_less_per_figure = 5 := by
  sorry

end jimmy_sells_less_l499_49916


namespace arithmetic_sequence_ratio_l499_49991

-- Definitions and conditions from the problem
variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
variable (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
variable (h3 : ∀ n, S n / T n = (3 * n - 1) / (n + 3))

-- The theorem that will give us the required answer
theorem arithmetic_sequence_ratio : 
  (a 8) / (b 5 + b 11) = 11 / 9 := by 
  have h4 := h3 15
  sorry

end arithmetic_sequence_ratio_l499_49991


namespace weekly_allowance_l499_49926

theorem weekly_allowance
  (video_game_cost : ℝ)
  (sales_tax_percentage : ℝ)
  (weeks_to_save : ℕ)
  (total_with_tax : ℝ)
  (total_savings : ℝ) :
  video_game_cost = 50 →
  sales_tax_percentage = 0.10 →
  weeks_to_save = 11 →
  total_with_tax = video_game_cost * (1 + sales_tax_percentage) →
  total_savings = weeks_to_save * (0.5 * total_savings) →
  total_savings = total_with_tax →
  total_savings = 55 :=
by
  intros
  sorry

end weekly_allowance_l499_49926


namespace people_landed_in_virginia_l499_49982

def initial_passengers : ℕ := 124
def texas_out : ℕ := 58
def texas_in : ℕ := 24
def north_carolina_out : ℕ := 47
def north_carolina_in : ℕ := 14
def crew_members : ℕ := 10

def final_passengers := initial_passengers - texas_out + texas_in - north_carolina_out + north_carolina_in
def total_people_landed := final_passengers + crew_members

theorem people_landed_in_virginia : total_people_landed = 67 :=
by
  sorry

end people_landed_in_virginia_l499_49982


namespace sector_area_max_radius_l499_49901

noncomputable def arc_length (R : ℝ) : ℝ := 20 - 2 * R

noncomputable def sector_area (R : ℝ) : ℝ :=
  let l := arc_length R
  0.5 * l * R

theorem sector_area_max_radius :
  ∃ (R : ℝ), sector_area R = -R^2 + 10 * R ∧
             R = 5 :=
sorry

end sector_area_max_radius_l499_49901


namespace gcd_459_357_eq_51_l499_49900

theorem gcd_459_357_eq_51 :
  gcd 459 357 = 51 := 
by
  sorry

end gcd_459_357_eq_51_l499_49900


namespace girls_exceed_boys_by_402_l499_49963

theorem girls_exceed_boys_by_402 : 
  let girls := 739
  let boys := 337
  girls - boys = 402 :=
by
  sorry

end girls_exceed_boys_by_402_l499_49963


namespace loss_per_metre_l499_49951

theorem loss_per_metre
  (total_metres : ℕ)
  (selling_price : ℕ)
  (cost_price_per_m: ℕ)
  (selling_price_total : selling_price = 18000)
  (cost_price_per_m_def : cost_price_per_m = 95)
  (total_metres_def : total_metres = 200) :
  ((cost_price_per_m * total_metres - selling_price) / total_metres) = 5 :=
by
  sorry

end loss_per_metre_l499_49951


namespace base7_to_base5_l499_49907

theorem base7_to_base5 (n : ℕ) (h : n = 305) : 
    3 * 7 ^ 2 + 0 * 7 ^ 1 + 5 = 152 → 152 = 1 * 5 ^ 3 + 1 * 5 ^ 2 + 0 * 5 ^ 1 + 2 * 5 ^ 0 → 305 = 1102 :=
by
  intros h1 h2
  sorry

end base7_to_base5_l499_49907


namespace unique_sequence_l499_49945

theorem unique_sequence (a : ℕ → ℝ) 
  (h1 : a 0 = 1) 
  (h2 : ∀ n : ℕ, a n > 0) 
  (h3 : ∀ n : ℕ, a n - a (n + 1) = a (n + 2)) : 
  ∀ n : ℕ, a n = ( (-1 + Real.sqrt 5) / 2)^n := 
sorry

end unique_sequence_l499_49945


namespace lower_limit_of_b_l499_49930

theorem lower_limit_of_b (a : ℤ) (b : ℤ) (h₁ : 8 < a ∧ a < 15) (h₂ : ∃ x, x < b ∧ b < 21) (h₃ : (14 : ℚ) / b - (9 : ℚ) / b = 1.55) : b = 4 :=
by
  sorry

end lower_limit_of_b_l499_49930


namespace count_even_integers_between_l499_49943

theorem count_even_integers_between : 
    let lower := 18 / 5
    let upper := 45 / 2
    ∃ (count : ℕ), (∀ n : ℕ, lower < n ∧ n < upper → n % 2 = 0 → n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 10 ∨ n = 12 ∨ n = 14 ∨ n = 16 ∨ n = 18 ∨ n = 20 ∨ n = 22) ∧ count = 10 :=
by
  sorry

end count_even_integers_between_l499_49943


namespace number_of_possible_monograms_l499_49986

-- Define the set of letters before 'M'
def letters_before_M : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'}

-- Define the set of letters after 'M'
def letters_after_M : Finset Char := {'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

-- State the theorem 
theorem number_of_possible_monograms : 
  (letters_before_M.card * letters_after_M.card) = 156 :=
by
  sorry

end number_of_possible_monograms_l499_49986


namespace fraction_calculation_l499_49994

theorem fraction_calculation : 
  (1/2 - 1/3) / (3/7 * 2/8) = 14/9 :=
by
  sorry

end fraction_calculation_l499_49994


namespace monotonic_intervals_l499_49978

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_intervals :
  (∀ x (h : 0 < x ∧ x < Real.exp 1), 0 < f x) ∧
  (∀ x (h : Real.exp 1 < x), f x < 0) :=
by
  sorry

end monotonic_intervals_l499_49978


namespace sum_M_N_K_l499_49919

theorem sum_M_N_K (d K M N : ℤ) 
(h : ∀ x : ℤ, (x^2 + 3*x + 1) ∣ (x^4 - d*x^3 + M*x^2 + N*x + K)) :
  M + N + K = 5*K - 4*d - 11 := 
sorry

end sum_M_N_K_l499_49919


namespace total_time_taken_l499_49909

theorem total_time_taken
  (speed_boat : ℝ)
  (speed_stream : ℝ)
  (distance : ℝ)
  (h_boat : speed_boat = 12)
  (h_stream : speed_stream = 5)
  (h_distance : distance = 325) :
  (distance / (speed_boat - speed_stream) + distance / (speed_boat + speed_stream)) = 65.55 :=
by
  sorry

end total_time_taken_l499_49909


namespace tan_sum_to_expression_l499_49962

theorem tan_sum_to_expression (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 :=
by 
  sorry

end tan_sum_to_expression_l499_49962


namespace sum_of_reciprocals_factors_12_l499_49941

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l499_49941


namespace probability_of_selecting_boy_given_girl_A_selected_l499_49984

-- Define the total number of girls and boys
def total_girls : ℕ := 5
def total_boys : ℕ := 2

-- Define the group size to be selected
def group_size : ℕ := 3

-- Define the probability of selecting at least one boy given girl A is selected
def probability_at_least_one_boy_given_girl_A : ℚ := 3 / 5

-- Math problem reformulated as a Lean theorem
theorem probability_of_selecting_boy_given_girl_A_selected : 
  (total_girls = 5) → (total_boys = 2) → (group_size = 3) → 
  (probability_at_least_one_boy_given_girl_A = 3 / 5) :=
by sorry

end probability_of_selecting_boy_given_girl_A_selected_l499_49984


namespace kanul_total_amount_l499_49931

def kanul_spent : ℝ := 3000 + 1000
def kanul_spent_percentage (T : ℝ) : ℝ := 0.30 * T

theorem kanul_total_amount (T : ℝ) (h : T = kanul_spent + kanul_spent_percentage T) :
  T = 5714.29 := sorry

end kanul_total_amount_l499_49931


namespace solve_system_of_equations_l499_49966

theorem solve_system_of_equations (a b c x y z : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  (a * y + b * x = c) ∧ (c * x + a * z = b) ∧ (b * z + c * y = a) →
  (x = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
  (y = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
  (z = (a^2 + b^2 - c^2) / (2 * a * b)) :=
by
  sorry

end solve_system_of_equations_l499_49966


namespace volume_of_given_sphere_l499_49989

noncomputable def volume_of_sphere (A d : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (Real.sqrt (d^2 + A / Real.pi))^3

theorem volume_of_given_sphere
  (hA : 2 * Real.pi = 2 * Real.pi)
  (hd : 1 = 1):
  volume_of_sphere (2 * Real.pi) 1 = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end volume_of_given_sphere_l499_49989


namespace smallest_b_l499_49924

theorem smallest_b {a b c d : ℕ} (r : ℕ) 
  (h1 : a = b - r) (h2 : c = b + r) (h3 : d = b + 2 * r) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h5 : a * b * c * d = 256) : b = 4 :=
by
  sorry

end smallest_b_l499_49924


namespace points_do_not_exist_l499_49959

/-- 
  If \( A, B, C, D \) are four points in space and 
  \( AB = 8 \) cm, 
  \( CD = 8 \) cm, 
  \( AC = 10 \) cm, 
  \( BD = 10 \) cm, 
  \( AD = 13 \) cm, 
  \( BC = 13 \) cm, 
  then such points \( A, B, C, D \) cannot exist.
-/
theorem points_do_not_exist 
  (A B C D : Type)
  (AB CD AC BD AD BC : ℝ) 
  (h1 : AB = 8) 
  (h2 : CD = 8) 
  (h3 : AC = 10)
  (h4 : BD = 10)
  (h5 : AD = 13)
  (h6 : BC = 13) : 
  false :=
sorry

end points_do_not_exist_l499_49959


namespace range_x_plus_y_l499_49938

theorem range_x_plus_y (x y : ℝ) (h : x^3 + y^3 = 2) : 0 < x + y ∧ x + y ≤ 2 :=
by {
  sorry
}

end range_x_plus_y_l499_49938


namespace clea_total_time_l499_49927

-- Definitions based on conditions given
def walking_time_on_stationary (x y : ℝ) (h1 : 80 * x = y) : ℝ :=
  80

def walking_time_on_moving (x y : ℝ) (k : ℝ) (h2 : 32 * (x + k) = y) : ℝ :=
  32

def escalator_speed (x k : ℝ) (h3 : k = 1.5 * x) : ℝ :=
  1.5 * x

-- The actual theorem based on the question
theorem clea_total_time 
  (x y k : ℝ)
  (h1 : 80 * x = y)
  (h2 : 32 * (x + k) = y)
  (h3 : k = 1.5 * x) :
  let t1 := y / (2 * x)
  let t2 := y / (3 * x)
  t1 + t2 = 200 / 3 :=
by
  sorry

end clea_total_time_l499_49927


namespace min_value_abs_x1_x2_l499_49940

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem min_value_abs_x1_x2 
  (a : ℝ) (x1 x2 : ℝ)
  (h_symm : ∃ k : ℤ, -π / 6 - (Real.arctan (Real.sqrt 3 / a)) = (k * π + π / 2))
  (h_diff : f a x1 - f a x2 = -4) :
  |x1 + x2| = (2 * π) / 3 := 
sorry

end min_value_abs_x1_x2_l499_49940


namespace problem_f_symmetric_l499_49995

theorem problem_f_symmetric (f : ℝ → ℝ) (k : ℝ) (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + k * f b) (h_not_zero : ∃ x : ℝ, f x ≠ 0) :
  ∀ x : ℝ, f (-x) = f x :=
sorry

end problem_f_symmetric_l499_49995


namespace range_of_x_for_y1_gt_y2_l499_49971

noncomputable def y1 (x : ℝ) : ℝ := x - 3
noncomputable def y2 (x : ℝ) : ℝ := 4 / x

theorem range_of_x_for_y1_gt_y2 :
  ∀ x : ℝ, (y1 x > y2 x) ↔ ((-1 < x ∧ x < 0) ∨ (x > 4)) := by
  sorry

end range_of_x_for_y1_gt_y2_l499_49971
