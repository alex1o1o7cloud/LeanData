import Mathlib

namespace num_pos_pairs_l1659_165972

theorem num_pos_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 + 3 * n < 40) :
  ∃ k : ℕ, k = 45 :=
by {
  -- Additional setup and configuration if needed
  -- ...
  sorry
}

end num_pos_pairs_l1659_165972


namespace triangle_classification_l1659_165954

theorem triangle_classification 
  (a b c : ℝ) 
  (h : (b^2 + a^2) * (b - a) = b * c^2 - a * c^2) : 
  (a = b ∨ a^2 + b^2 = c^2) :=
by sorry

end triangle_classification_l1659_165954


namespace denomination_is_100_l1659_165980

-- Define the initial conditions
def num_bills : ℕ := 8
def total_savings : ℕ := 800

-- Define the denomination of the bills
def denomination_bills (num_bills : ℕ) (total_savings : ℕ) : ℕ := 
  total_savings / num_bills

-- The theorem stating the denomination is $100
theorem denomination_is_100 :
  denomination_bills num_bills total_savings = 100 := by
  sorry

end denomination_is_100_l1659_165980


namespace units_digit_of_n_l1659_165946

-- Definitions
def units_digit (x : ℕ) : ℕ := x % 10

-- Conditions
variables (m n : ℕ)
axiom condition1 : m * n = 23^5
axiom condition2 : units_digit m = 4

-- Theorem statement
theorem units_digit_of_n : units_digit n = 8 :=
sorry

end units_digit_of_n_l1659_165946


namespace find_p_l1659_165993

theorem find_p (p : ℤ)
  (h1 : ∀ (u v : ℤ), u > 0 → v > 0 → 5 * u ^ 2 - 5 * p * u + (66 * p - 1) = 0 ∧
    5 * v ^ 2 - 5 * p * v + (66 * p - 1) = 0) :
  p = 76 :=
sorry

end find_p_l1659_165993


namespace num_handshakes_ten_women_l1659_165974

def num_handshakes (n : ℕ) : ℕ :=
(n * (n - 1)) / 2

theorem num_handshakes_ten_women :
  num_handshakes 10 = 45 :=
by
  sorry

end num_handshakes_ten_women_l1659_165974


namespace hyperbola_asymptotes_l1659_165909

theorem hyperbola_asymptotes :
  ∀ x y : ℝ,
  (x ^ 2 / 4 - y ^ 2 / 16 = 1) → (y = 2 * x) ∨ (y = -2 * x) :=
sorry

end hyperbola_asymptotes_l1659_165909


namespace third_side_length_l1659_165900

theorem third_side_length (a b : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  ∃ x : ℝ, (a = 3 ∧ b = 4) ∧ (x = 5 ∨ x = Real.sqrt 7) :=
by
  sorry

end third_side_length_l1659_165900


namespace good_jars_l1659_165945

def original_cartons : Nat := 50
def jars_per_carton : Nat := 20
def less_cartons_received : Nat := 20
def damaged_jars_per_5_cartons : Nat := 3
def total_damaged_cartons : Nat := 1
def total_good_jars : Nat := 565

theorem good_jars (original_cartons jars_per_carton less_cartons_received damaged_jars_per_5_cartons total_damaged_cartons : Nat) :
  (original_cartons - less_cartons_received) * jars_per_carton 
  - (5 * damaged_jars_per_5_cartons + total_damaged_cartons * jars_per_carton) = total_good_jars := 
by 
  sorry

end good_jars_l1659_165945


namespace sum_of_roots_of_quadratic_l1659_165902

noncomputable def x1_x2_roots_properties : Prop :=
  ∃ x₁ x₂ : ℝ, (x₁ + x₂ = 3) ∧ (x₁ * x₂ = -4)

theorem sum_of_roots_of_quadratic :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) → (x₁ + x₂ = 3) :=
by
  sorry

end sum_of_roots_of_quadratic_l1659_165902


namespace broker_investment_increase_l1659_165947

noncomputable def final_value_stock_A := 
  let initial := 100.0
  let year1 := initial * (1 + 0.80)
  let year2 := year1 * (1 - 0.30)
  year2 * (1 + 0.10)

noncomputable def final_value_stock_B := 
  let initial := 100.0
  let year1 := initial * (1 + 0.50)
  let year2 := year1 * (1 - 0.10)
  year2 * (1 - 0.25)

noncomputable def final_value_stock_C := 
  let initial := 100.0
  let year1 := initial * (1 - 0.30)
  let year2 := year1 * (1 - 0.40)
  year2 * (1 + 0.80)

noncomputable def final_value_stock_D := 
  let initial := 100.0
  let year1 := initial * (1 + 0.40)
  let year2 := year1 * (1 + 0.20)
  year2 * (1 - 0.15)

noncomputable def total_final_value := 
  final_value_stock_A + final_value_stock_B + final_value_stock_C + final_value_stock_D

noncomputable def initial_total_value := 4 * 100.0

noncomputable def net_increase := total_final_value - initial_total_value

noncomputable def net_increase_percentage := (net_increase / initial_total_value) * 100

theorem broker_investment_increase : net_increase_percentage = 14.5625 := 
by
  sorry

end broker_investment_increase_l1659_165947


namespace cost_price_as_percentage_l1659_165933

theorem cost_price_as_percentage (SP CP : ℝ) 
  (profit_percentage : ℝ := 4.166666666666666) 
  (P : ℝ := SP - CP)
  (profit_eq : P = (profit_percentage / 100) * SP) :
  CP = (95.83333333333334 / 100) * SP := 
by
  sorry

end cost_price_as_percentage_l1659_165933


namespace zeros_of_f_is_pm3_l1659_165903

def f (x : ℝ) : ℝ := x^2 - 9

theorem zeros_of_f_is_pm3 :
  ∃ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -3 :=
by sorry

end zeros_of_f_is_pm3_l1659_165903


namespace rectangular_to_polar_l1659_165981

theorem rectangular_to_polar : 
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (3 * Real.sqrt 2, 7 * Real.pi / 4) := 
by
  sorry

end rectangular_to_polar_l1659_165981


namespace chosen_number_is_reconstructed_l1659_165914

theorem chosen_number_is_reconstructed (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 26) :
  ∃ (a0 a1 a2 : ℤ), (a0 = 0 ∨ a0 = 1 ∨ a0 = 2) ∧ 
                     (a1 = 0 ∨ a1 = 1 ∨ a1 = 2) ∧ 
                     (a2 = 0 ∨ a2 = 1 ∨ a2 = 2) ∧ 
                     n = a0 * 3^0 + a1 * 3^1 + a2 * 3^2 ∧ 
                     n = (if a0 = 1 then 1 else 0) + (if a0 = 2 then 2 else 0) +
                         (if a1 = 1 then 3 else 0) + (if a1 = 2 then 6 else 0) +
                         (if a2 = 1 then 9 else 0) + (if a2 = 2 then 18 else 0) := 
sorry

end chosen_number_is_reconstructed_l1659_165914


namespace necessary_but_not_sufficient_l1659_165991

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 3 * x - 4 = 0) -> (x = 4 ∨ x = -1) ∧ ¬(x = 4 ∨ x = -1 -> x = 4) :=
by sorry

end necessary_but_not_sufficient_l1659_165991


namespace product_divisible_by_60_l1659_165939

theorem product_divisible_by_60 {a : ℤ} : 
  60 ∣ ((a^2 - 1) * a^2 * (a^2 + 1)) := 
by sorry

end product_divisible_by_60_l1659_165939


namespace range_of_m_l1659_165924

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

def tangent_points (m : ℝ) (x₀ : ℝ) : Prop := 
  2 * x₀ ^ 3 - 3 * x₀ ^ 2 + m + 3 = 0

theorem range_of_m (m : ℝ) :
  (∀ x₀, tangent_points m x₀) ∧ m ≠ -2 → (-3 < m ∧ m < -2) :=
sorry

end range_of_m_l1659_165924


namespace compare_a_b_c_l1659_165986

noncomputable def a : ℝ := Real.log (Real.sqrt 2)
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := 1 / Real.exp 1

theorem compare_a_b_c : a < b ∧ b < c := by
  -- Proof will be done here
  sorry

end compare_a_b_c_l1659_165986


namespace percentage_reduction_l1659_165983

theorem percentage_reduction :
  let P := 60
  let R := 45
  (900 / R) - (900 / P) = 5 →
  (P - R) / P * 100 = 25 :=
by 
  intros P R h
  have h1 : R = 45 := rfl
  have h2 : P = 60 := sorry
  rw [h1] at h
  rw [h2]
  sorry -- detailed steps to be filled in the proof

end percentage_reduction_l1659_165983


namespace total_time_spent_l1659_165920

-- Define time spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_third_step : ℕ := time_first_step + time_second_step

-- Prove the total time spent
theorem total_time_spent : 
  time_first_step + time_second_step + time_third_step = 90 := by
  sorry

end total_time_spent_l1659_165920


namespace range_of_a_l1659_165922

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a - x > 1 → x < 2 * a - 1)) ∧
  (∀ x : ℝ, (2 * x + 5 > 3 * a → x > (3 * a - 5) / 2)) ∧
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 6 →
    (x < 2 * a - 1 ∧ x > (3 * a - 5) / 2))) →
  7 / 3 ≤ a ∧ a ≤ 7 / 2 :=
by
  sorry

end range_of_a_l1659_165922


namespace bob_distance_when_meet_l1659_165906

theorem bob_distance_when_meet (total_distance : ℕ) (yolanda_speed : ℕ) (bob_speed : ℕ) 
    (yolanda_additional_distance : ℕ) (t : ℕ) :
    total_distance = 31 ∧ yolanda_speed = 3 ∧ bob_speed = 4 ∧ yolanda_additional_distance = 3 
    ∧ 7 * t = 28 → 4 * t = 16 := by
    sorry

end bob_distance_when_meet_l1659_165906


namespace speed_of_first_train_is_correct_l1659_165959

-- Define the lengths of the trains
def length_train1 : ℕ := 110
def length_train2 : ℕ := 200

-- Define the speed of the second train in kmph
def speed_train2 : ℕ := 65

-- Define the time they take to clear each other in seconds
def time_clear_seconds : ℚ := 7.695936049253991

-- Define the speed of the first train
def speed_train1 : ℚ :=
  let time_clear_hours : ℚ := time_clear_seconds / 3600
  let total_distance_km : ℚ := (length_train1 + length_train2) / 1000
  let relative_speed_kmph : ℚ := total_distance_km / time_clear_hours 
  relative_speed_kmph - speed_train2

-- The proof problem is to show that the speed of the first train is 80.069 kmph
theorem speed_of_first_train_is_correct : speed_train1 = 80.069 := by
  sorry

end speed_of_first_train_is_correct_l1659_165959


namespace log_order_preservation_l1659_165904

theorem log_order_preservation {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (Real.log a > Real.log b) → (a > b) :=
by
  sorry

end log_order_preservation_l1659_165904


namespace condition_necessary_but_not_sufficient_l1659_165932

variable (a : ℝ)

theorem condition_necessary_but_not_sufficient (h : a^2 < 1) : (a < 1) ∧ (¬(a < 1 → a^2 < 1)) := sorry

end condition_necessary_but_not_sufficient_l1659_165932


namespace lions_min_games_for_90_percent_wins_l1659_165966

theorem lions_min_games_for_90_percent_wins : 
  ∀ N : ℕ, (N ≥ 26) ↔ 1 + N ≥ (9 * (4 + N)) / 10 := 
by 
  sorry

end lions_min_games_for_90_percent_wins_l1659_165966


namespace solve_for_x_values_for_matrix_l1659_165990

def matrix_equals_neg_two (x : ℝ) : Prop :=
  let a := 3 * x
  let b := x
  let c := 4
  let d := 2 * x
  (a * b - c * d = -2)

theorem solve_for_x_values_for_matrix : 
  ∃ (x : ℝ), matrix_equals_neg_two x ↔ (x = (4 + Real.sqrt 10) / 3 ∨ x = (4 - Real.sqrt 10) / 3) :=
sorry

end solve_for_x_values_for_matrix_l1659_165990


namespace range_of_x_l1659_165961

theorem range_of_x (m : ℝ) (x : ℝ) (h : 0 < m ∧ m ≤ 5) : 
  (x^2 + (2 * m - 1) * x > 4 * x + 2 * m - 4) ↔ (x < -6 ∨ x > 4) := 
sorry

end range_of_x_l1659_165961


namespace find_m_l1659_165949

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = -3) (h2 : x1 * x2 = m) (h3 : 1 / x1 + 1 / x2 = 1) : m = -3 :=
by
  sorry

end find_m_l1659_165949


namespace mean_score_is_93_l1659_165952

-- Define Jane's scores as a list
def scores : List ℕ := [98, 97, 92, 85, 93]

-- Define the mean of the scores
noncomputable def mean (lst : List ℕ) : ℚ := 
  (lst.foldl (· + ·) 0 : ℚ) / lst.length

-- The theorem to prove
theorem mean_score_is_93 : mean scores = 93 := by
  sorry

end mean_score_is_93_l1659_165952


namespace a1_greater_than_floor_2n_over_3_l1659_165979

theorem a1_greater_than_floor_2n_over_3
  (n : ℕ)
  (a : ℕ → ℕ)
  (h1 : ∀ i j : ℕ, i < j → i ≤ n ∧ j ≤ n → a i < a j)
  (h2 : ∀ i j : ℕ, i ≠ j → i ≤ n ∧ j ≤ n → lcm (a i) (a j) > 2 * n)
  (h_max : ∀ i : ℕ, i ≤ n → a i ≤ 2 * n) :
  a 1 > (2 * n) / 3 :=
by
  sorry

end a1_greater_than_floor_2n_over_3_l1659_165979


namespace right_triangle_hypotenuse_l1659_165992

theorem right_triangle_hypotenuse (x : ℝ) (h : x^2 = 3^2 + 5^2) : x = Real.sqrt 34 :=
by sorry

end right_triangle_hypotenuse_l1659_165992


namespace find_wall_width_l1659_165910

noncomputable def wall_width (painting_width : ℝ) (painting_height : ℝ) (wall_height : ℝ) (painting_coverage : ℝ) : ℝ :=
  (painting_width * painting_height) / (painting_coverage * wall_height)

-- Given constants
def painting_width : ℝ := 2
def painting_height : ℝ := 4
def wall_height : ℝ := 5
def painting_coverage : ℝ := 0.16
def expected_width : ℝ := 10

theorem find_wall_width : wall_width painting_width painting_height wall_height painting_coverage = expected_width := 
by
  sorry

end find_wall_width_l1659_165910


namespace remainder_of_polynomial_division_l1659_165919

-- Define the polynomial f(r)
def f (r : ℝ) : ℝ := r ^ 15 + 1

-- Define the polynomial divisor g(r)
def g (r : ℝ) : ℝ := r + 1

-- State the theorem about the remainder when f(r) is divided by g(r)
theorem remainder_of_polynomial_division : 
  (f (-1)) = 0 := by
  -- Skipping the proof for now
  sorry

end remainder_of_polynomial_division_l1659_165919


namespace necessary_but_not_sufficient_l1659_165978

   theorem necessary_but_not_sufficient (a : ℝ) : a^2 > a → (a > 1) :=
   by {
     sorry
   }
   
end necessary_but_not_sufficient_l1659_165978


namespace number_of_boys_selected_l1659_165962

theorem number_of_boys_selected {boys girls selections : ℕ} 
  (h_boys : boys = 11) (h_girls : girls = 10) (h_selections : selections = 6600) : 
  ∃ (k : ℕ), k = 2 :=
sorry

end number_of_boys_selected_l1659_165962


namespace ratio_of_time_l1659_165964

theorem ratio_of_time (T_A T_B : ℝ) (h1 : T_A = 8) (h2 : 1 / T_A + 1 / T_B = 0.375) :
  T_B / T_A = 1 / 2 :=
by 
  sorry

end ratio_of_time_l1659_165964


namespace geometric_product_l1659_165942

theorem geometric_product (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 10) 
  (h2 : 1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 + 1 / a 6 = 5) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 8 :=
sorry

end geometric_product_l1659_165942


namespace find_base_number_l1659_165971

-- Define the base number
def base_number (x : ℕ) (k : ℕ) : Prop := x ^ k > 4 ^ 22

-- State the theorem based on the problem conditions
theorem find_base_number : ∃ x : ℕ, ∀ k : ℕ, (k = 8) → (base_number x k) → (x = 64) :=
by sorry

end find_base_number_l1659_165971


namespace rectangle_area_divisible_by_12_l1659_165931

theorem rectangle_area_divisible_by_12
  (x y z : ℤ)
  (h : x^2 + y^2 = z^2) :
  12 ∣ (x * y) :=
sorry

end rectangle_area_divisible_by_12_l1659_165931


namespace football_goal_average_increase_l1659_165913

theorem football_goal_average_increase :
  ∀ (A : ℝ), 4 * A + 2 = 8 → (8 / 5) - A = 0.1 :=
by
  intro A
  intro h
  sorry -- Proof to be filled in

end football_goal_average_increase_l1659_165913


namespace find_values_of_A_l1659_165989

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_values_of_A (A B C : ℕ) :
  sum_of_digits A = B ∧
  sum_of_digits B = C ∧
  A + B + C = 60 →
  (A = 44 ∨ A = 50 ∨ A = 47) :=
by
  sorry

end find_values_of_A_l1659_165989


namespace quadratic_has_two_distinct_real_roots_l1659_165921

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 + 2 * k * r1 + (k - 1) = 0 ∧ r2^2 + 2 * k * r2 + (k - 1) = 0 := 
by 
  sorry

end quadratic_has_two_distinct_real_roots_l1659_165921


namespace possible_values_of_x_l1659_165927

theorem possible_values_of_x (x : ℝ) (h : (x^2 - 1) / x = 0) (hx : x ≠ 0) : x = 1 ∨ x = -1 :=
  sorry

end possible_values_of_x_l1659_165927


namespace download_time_correct_l1659_165997

-- Define the given conditions
def total_size : ℕ := 880
def downloaded : ℕ := 310
def speed : ℕ := 3

-- Calculate the remaining time to download
def time_remaining : ℕ := (total_size - downloaded) / speed

-- Theorem statement that needs to be proven
theorem download_time_correct : time_remaining = 190 := by
  -- Proof goes here
  sorry

end download_time_correct_l1659_165997


namespace earnings_difference_l1659_165970

def total_earnings : ℕ := 3875
def first_job_earnings : ℕ := 2125
def second_job_earnings := total_earnings - first_job_earnings

theorem earnings_difference : (first_job_earnings - second_job_earnings) = 375 := by
  sorry

end earnings_difference_l1659_165970


namespace car_b_speed_l1659_165934

theorem car_b_speed :
  ∀ (v : ℕ),
    (232 - 4 * v = 32) →
    v = 50 :=
  by
  sorry

end car_b_speed_l1659_165934


namespace diving_competition_score_l1659_165950

theorem diving_competition_score 
  (scores : List ℝ)
  (h : scores = [7.5, 8.0, 9.0, 6.0, 8.8])
  (degree_of_difficulty : ℝ)
  (hd : degree_of_difficulty = 3.2) :
  let sorted_scores := scores.erase 9.0 |>.erase 6.0
  let remaining_sum := sorted_scores.sum
  remaining_sum * degree_of_difficulty = 77.76 :=
by
  sorry

end diving_competition_score_l1659_165950


namespace rate_of_current_l1659_165901

theorem rate_of_current (speed_boat_still_water : ℕ) (time_hours : ℚ) (distance_downstream : ℚ)
    (h_speed_boat_still_water : speed_boat_still_water = 20)
    (h_time_hours : time_hours = 15 / 60)
    (h_distance_downstream : distance_downstream = 6.25) :
    ∃ c : ℚ, distance_downstream = (speed_boat_still_water + c) * time_hours ∧ c = 5 :=
by
    sorry

end rate_of_current_l1659_165901


namespace sum_of_areas_of_circles_l1659_165976

noncomputable def radius (n : ℕ) : ℝ :=
  3 / 3^n

noncomputable def area (n : ℕ) : ℝ :=
  Real.pi * (radius n)^2

noncomputable def total_area : ℝ :=
  ∑' n, area n

theorem sum_of_areas_of_circles:
  total_area = (9 * Real.pi) / 8 :=
by
  sorry

end sum_of_areas_of_circles_l1659_165976


namespace percentage_increase_equiv_l1659_165936

theorem percentage_increase_equiv {P : ℝ} : 
  (P * (1 + 0.08) * (1 + 0.08)) = (P * 1.1664) :=
by
  sorry

end percentage_increase_equiv_l1659_165936


namespace rationalize_denominator_l1659_165929

theorem rationalize_denominator : 
  let A := -13 
  let B := -9
  let C := 3
  let D := 2
  let E := 165
  let F := 51
  A + B + C + D + E + F = 199 := by
sorry

end rationalize_denominator_l1659_165929


namespace max_value_x2_plus_2xy_l1659_165912

open Real

theorem max_value_x2_plus_2xy (x y : ℝ) (h : x + y = 5) : 
  ∃ (M : ℝ), (M = x^2 + 2 * x * y) ∧ (∀ z w : ℝ, z + w = 5 → z^2 + 2 * z * w ≤ M) :=
by
  sorry

end max_value_x2_plus_2xy_l1659_165912


namespace sin_alpha_beta_l1659_165907

theorem sin_alpha_beta (a b c α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
    (h3 : a * Real.cos α + b * Real.sin α + c = 0) (h4 : a * Real.cos β + b * Real.sin β + c = 0) 
    (h5 : α ≠ β) : Real.sin (α + β) = (2 * a * b) / (a ^ 2 + b ^ 2) := 
sorry

end sin_alpha_beta_l1659_165907


namespace square_tiles_count_l1659_165953

theorem square_tiles_count (a b : ℕ) (h1 : a + b = 25) (h2 : 3 * a + 4 * b = 84) : b = 9 := by
  sorry

end square_tiles_count_l1659_165953


namespace gdp_scientific_notation_l1659_165975

theorem gdp_scientific_notation (trillion : ℕ) (five_year_growth : ℝ) (gdp : ℝ) :
  trillion = 10^12 ∧ 1 ≤ gdp / 10^14 ∧ gdp / 10^14 < 10 ∧ gdp = 121 * 10^12 → gdp = 1.21 * 10^14
:= by
  sorry

end gdp_scientific_notation_l1659_165975


namespace continuous_stripe_probability_l1659_165908

open ProbabilityTheory

noncomputable def total_stripe_combinations : ℕ := 4 ^ 6

noncomputable def favorable_stripe_outcomes : ℕ := 3 * 4

theorem continuous_stripe_probability :
  (favorable_stripe_outcomes : ℚ) / (total_stripe_combinations : ℚ) = 3 / 1024 := by
  sorry

end continuous_stripe_probability_l1659_165908


namespace rhombus_area_8_cm2_l1659_165940

open Real

noncomputable def rhombus_area (side : ℝ) (angle : ℝ) : ℝ :=
  (side * side * sin angle) / 2 * 2

theorem rhombus_area_8_cm2 (side : ℝ) (angle : ℝ) (h1 : side = 4) (h2 : angle = π / 4) : rhombus_area side angle = 8 :=
by
  -- Definitions and calculations are omitted and replaced with 'sorry'
  sorry

end rhombus_area_8_cm2_l1659_165940


namespace number_of_boys_l1659_165963

def initial_girls : ℕ := 706
def new_girls : ℕ := 418
def total_pupils : ℕ := 1346
def total_girls := initial_girls + new_girls

theorem number_of_boys : 
  total_pupils = total_girls + 222 := 
by
  sorry

end number_of_boys_l1659_165963


namespace shifted_parabola_eq_l1659_165982

theorem shifted_parabola_eq :
  ∀ x, (∃ y, y = 2 * (x - 3)^2 + 2) →
       (∃ y, y = 2 * (x + 0)^2 + 4) :=
by sorry

end shifted_parabola_eq_l1659_165982


namespace min_m_value_arithmetic_seq_l1659_165967

theorem min_m_value_arithmetic_seq :
  ∀ (a S : ℕ → ℚ) (m : ℕ),
  (∀ n : ℕ, a (n+2) = 5 ∧ a (n+6) = 21) →
  (∀ n : ℕ, S (n+1) = S n + 1 / a (n+1)) →
  (∀ n : ℕ, S (2 * n + 1) - S n ≤ m / 15) →
  ∀ n : ℕ, m = 5 :=
sorry

end min_m_value_arithmetic_seq_l1659_165967


namespace isosceles_right_triangle_area_l1659_165958

theorem isosceles_right_triangle_area (a : ℝ) (h : ℝ) (p : ℝ) 
  (h_triangle : h = a * Real.sqrt 2) 
  (hypotenuse_is_16 : h = 16) :
  (1 / 2) * a * a = 64 := 
by
  -- Skip the proof as per guidelines
  sorry

end isosceles_right_triangle_area_l1659_165958


namespace factorize_quadratic_l1659_165925

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by
  sorry

end factorize_quadratic_l1659_165925


namespace smallest_k_l1659_165968

theorem smallest_k (k : ℕ) (h : 201 ≡ 9 [MOD 24]) : k = 1 := by
  sorry

end smallest_k_l1659_165968


namespace remainder_polynomial_division_l1659_165930

theorem remainder_polynomial_division :
  ∀ (x : ℝ), (2 * x^2 - 21 * x + 55) % (x + 3) = 136 := 
sorry

end remainder_polynomial_division_l1659_165930


namespace find_number_l1659_165995

theorem find_number (n : ℤ) (h : 7 * n - 15 = 2 * n + 10) : n = 5 :=
sorry

end find_number_l1659_165995


namespace inequality_result_l1659_165943

theorem inequality_result
  (a b : ℝ) 
  (x y : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) :
  x + y ≤ 0 :=
sorry

end inequality_result_l1659_165943


namespace probability_ephraim_fiona_same_heads_as_keiko_l1659_165969

/-- Define a function to calculate the probability that Keiko, Ephraim, and Fiona get the same number of heads. -/
def probability_same_heads : ℚ :=
  let total_outcomes := (2^2) * (2^3) * (2^3)
  let successful_outcomes := 13
  successful_outcomes / total_outcomes

/-- Theorem stating the problem condition and expected probability. -/
theorem probability_ephraim_fiona_same_heads_as_keiko
  (h_keiko : ℕ := 2) -- Keiko tosses two coins
  (h_ephraim : ℕ := 3) -- Ephraim tosses three coins
  (h_fiona : ℕ := 3) -- Fiona tosses three coins
  -- Expected probability that both Ephraim and Fiona get the same number of heads as Keiko
  : probability_same_heads = 13 / 256 :=
sorry

end probability_ephraim_fiona_same_heads_as_keiko_l1659_165969


namespace initial_games_l1659_165965

theorem initial_games (X : ℕ) (h1 : X + 31 - 105 = 6) : X = 80 :=
by
  sorry

end initial_games_l1659_165965


namespace sum_of_coefficients_l1659_165915

theorem sum_of_coefficients :
  (Nat.choose 50 3 + Nat.choose 50 5) = 2138360 := 
by 
  sorry

end sum_of_coefficients_l1659_165915


namespace m_zero_sufficient_but_not_necessary_l1659_165928

-- Define the sequence a_n
variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the condition for equal difference of squares sequence
def equal_diff_of_squares_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, (a (n+1))^2 - (a n)^2 = d

-- Define the sequence b_n as an arithmetic sequence with common difference m
variable (b : ℕ → ℝ)
variable (m : ℝ)

def arithmetic_sequence (b : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n, b (n+1) - b n = m

-- Prove "m = 0" is a sufficient but not necessary condition for {b_n} to be an equal difference of squares sequence
theorem m_zero_sufficient_but_not_necessary (a b : ℕ → ℝ) (d m : ℝ) :
  equal_diff_of_squares_sequence a d → arithmetic_sequence b m → (m = 0 → equal_diff_of_squares_sequence b d) ∧ (¬(m ≠ 0) → equal_diff_of_squares_sequence b d) :=
sorry


end m_zero_sufficient_but_not_necessary_l1659_165928


namespace largest_prime_factor_of_4620_l1659_165984

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / m → ¬ (m ∣ n)

def prime_factors (n : ℕ) : List ℕ :=
  -- assumes a well-defined function that generates the prime factor list
  -- this is a placeholder function for demonstrating purposes
  sorry

def largest_prime_factor (l : List ℕ) : ℕ :=
  l.foldr max 0

theorem largest_prime_factor_of_4620 : largest_prime_factor (prime_factors 4620) = 11 :=
by
  sorry

end largest_prime_factor_of_4620_l1659_165984


namespace calculate_amount_after_two_years_l1659_165937

noncomputable def amount_after_years (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + rate) ^ years

theorem calculate_amount_after_two_years :
  amount_after_years 51200 0.125 2 = 64800 :=
by
  sorry

end calculate_amount_after_two_years_l1659_165937


namespace romeo_total_profit_is_55_l1659_165926

-- Defining the conditions
def number_of_bars : ℕ := 5
def cost_per_bar : ℕ := 5
def packaging_cost_per_bar : ℕ := 2
def total_selling_price : ℕ := 90

-- Defining the profit calculation
def total_cost_per_bar := cost_per_bar + packaging_cost_per_bar
def selling_price_per_bar := total_selling_price / number_of_bars
def profit_per_bar := selling_price_per_bar - total_cost_per_bar
def total_profit := profit_per_bar * number_of_bars

-- Proving the total profit
theorem romeo_total_profit_is_55 : total_profit = 55 :=
by
  sorry

end romeo_total_profit_is_55_l1659_165926


namespace barbara_candies_l1659_165916

theorem barbara_candies :
  ∀ (initial left used : ℝ), initial = 18 ∧ left = 9 → initial - left = used → used = 9 :=
by
  intros initial left used h1 h2
  sorry

end barbara_candies_l1659_165916


namespace ryan_correct_percentage_l1659_165917

theorem ryan_correct_percentage :
  let problems1 := 25
  let correct1 := 0.8 * problems1
  let problems2 := 40
  let correct2 := 0.9 * problems2
  let problems3 := 10
  let correct3 := 0.7 * problems3
  let total_problems := problems1 + problems2 + problems3
  let total_correct := correct1 + correct2 + correct3
  (total_correct / total_problems) = 0.84 :=
by 
  sorry

end ryan_correct_percentage_l1659_165917


namespace volume_of_one_wedge_l1659_165987

theorem volume_of_one_wedge 
  (circumference : ℝ)
  (h : circumference = 15 * Real.pi) 
  (radius : ℝ) 
  (volume : ℝ) 
  (wedge_volume : ℝ) 
  (h_radius : radius = 7.5)
  (h_volume : volume = (4 / 3) * Real.pi * radius^3)
  (h_wedge_volume : wedge_volume = volume / 5)
  : wedge_volume = 112.5 * Real.pi :=
by
  sorry

end volume_of_one_wedge_l1659_165987


namespace alcohol_quantity_l1659_165999

theorem alcohol_quantity (A W : ℝ) (h1 : A / W = 4 / 3) (h2 : A / (W + 8) = 4 / 5) : A = 16 := 
by
  sorry

end alcohol_quantity_l1659_165999


namespace right_triangle_segment_ratio_l1659_165985

-- Definitions of the triangle sides and hypotenuse
def right_triangle (AB BC : ℝ) : Prop :=
  AB/BC = 4/3

def hypotenuse (AB BC AC : ℝ) : Prop :=
  AC^2 = AB^2 + BC^2

def perpendicular_segment_ratio (AD CD : ℝ) : Prop :=
  AD / CD = 9/16

-- Final statement of the problem
theorem right_triangle_segment_ratio
  (AB BC AC AD CD : ℝ)
  (h1 : right_triangle AB BC)
  (h2 : hypotenuse AB BC AC)
  (h3 : perpendicular_segment_ratio AD CD) :
  CD / AD = 16/9 := sorry

end right_triangle_segment_ratio_l1659_165985


namespace joan_football_games_l1659_165996

theorem joan_football_games (games_this_year games_last_year total_games: ℕ)
  (h1 : games_this_year = 4)
  (h2 : games_last_year = 9)
  (h3 : total_games = games_this_year + games_last_year) :
  total_games = 13 := 
by
  sorry

end joan_football_games_l1659_165996


namespace tax_diminished_by_20_percent_l1659_165948

theorem tax_diminished_by_20_percent
(T C : ℝ) 
(hT : T > 0) 
(hC : C > 0) 
(X : ℝ) 
(h_increased_consumption : ∀ (T C : ℝ), (C * 1.15) = C + 0.15 * C)
(h_decrease_revenue : T * (1 - X / 100) * C * 1.15 = T * C * 0.92) :
X = 20 := 
sorry

end tax_diminished_by_20_percent_l1659_165948


namespace emerson_row_distance_l1659_165935

theorem emerson_row_distance (d1 d2 total : ℕ) (h1 : d1 = 6) (h2 : d2 = 18) (h3 : total = 39) :
  15 = total - (d1 + d2) :=
by sorry

end emerson_row_distance_l1659_165935


namespace n_squared_divisible_by_144_l1659_165960

theorem n_squared_divisible_by_144 (n : ℕ) (h1 : 0 < n) (h2 : ∃ t : ℕ, t = 12 ∧ ∀ d : ℕ, d ∣ n → d ≤ t) : 144 ∣ n^2 :=
sorry

end n_squared_divisible_by_144_l1659_165960


namespace cos_double_angle_l1659_165994

theorem cos_double_angle (α : ℝ) (h : ‖(Real.cos α, Real.sqrt 2 / 2)‖ = Real.sqrt 3 / 2) : Real.cos (2 * α) = -1 / 2 :=
sorry

end cos_double_angle_l1659_165994


namespace first_three_digits_of_quotient_are_239_l1659_165944

noncomputable def a : ℝ := 0.12345678910114748495051
noncomputable def b_lower_bound : ℝ := 0.515
noncomputable def b_upper_bound : ℝ := 0.516

theorem first_three_digits_of_quotient_are_239 (b : ℝ) (hb : b_lower_bound < b ∧ b < b_upper_bound) :
    0.239 * b < a ∧ a < 0.24 * b := 
sorry

end first_three_digits_of_quotient_are_239_l1659_165944


namespace find_x_plus_y_l1659_165938

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 7 * y = 17) (h2 : 3 * x + 5 * y = 11) : x + y = 83 / 23 :=
sorry

end find_x_plus_y_l1659_165938


namespace general_term_sum_bn_l1659_165956

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + 2 * n
noncomputable def a (n : ℕ) : ℕ := 4 * n
noncomputable def b (n : ℕ) : ℕ := 2 ^ (4 * n)
noncomputable def T (n : ℕ) : ℝ := (16 / 15) * (16^n - 1)

theorem general_term (n : ℕ) (h1 : S n = 2 * n^2 + 2 * n) 
    (h2 : S (n-1) = 2 * (n-1)^2 + 2 * (n-1))
    (h3 : n ≥ 1) : a n = 4 * n :=
by sorry

theorem sum_bn (n : ℕ) (h : ∀ n, (b n, a n) = ((2 ^ (4 * n)), 4 * n)) : 
    T n = (16 / 15) * (16^n - 1) :=
by sorry

end general_term_sum_bn_l1659_165956


namespace average_weight_increase_l1659_165998

theorem average_weight_increase 
  (n : ℕ) (old_weight new_weight : ℝ) (group_size := 8) 
  (old_weight := 70) (new_weight := 90) : 
  ((new_weight - old_weight) / group_size) = 2.5 := 
by sorry

end average_weight_increase_l1659_165998


namespace express_y_in_terms_of_y_l1659_165973

variable (x : ℝ)

theorem express_y_in_terms_of_y (y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
sorry

end express_y_in_terms_of_y_l1659_165973


namespace veronica_loss_more_than_seth_l1659_165911

noncomputable def seth_loss : ℝ := 17.5
noncomputable def jerome_loss : ℝ := 3 * seth_loss
noncomputable def total_loss : ℝ := 89
noncomputable def veronica_loss : ℝ := total_loss - (seth_loss + jerome_loss)

theorem veronica_loss_more_than_seth :
  veronica_loss - seth_loss = 1.5 :=
by
  have h_seth_loss : seth_loss = 17.5 := rfl
  have h_jerome_loss : jerome_loss = 3 * seth_loss := rfl
  have h_total_loss : total_loss = 89 := rfl
  have h_veronica_loss : veronica_loss = total_loss - (seth_loss + jerome_loss) := rfl
  sorry

end veronica_loss_more_than_seth_l1659_165911


namespace x_lt_1_iff_x_abs_x_lt_1_l1659_165977

theorem x_lt_1_iff_x_abs_x_lt_1 (x : ℝ) : x < 1 ↔ x * |x| < 1 :=
sorry

end x_lt_1_iff_x_abs_x_lt_1_l1659_165977


namespace roberts_total_sales_l1659_165955

theorem roberts_total_sales 
  (basic_salary : ℝ := 1250) 
  (commission_rate : ℝ := 0.10) 
  (savings_rate : ℝ := 0.20) 
  (monthly_expenses : ℝ := 2888) 
  (S : ℝ) : S = 23600 :=
by
  have total_earnings := basic_salary + commission_rate * S
  have used_for_expenses := (1 - savings_rate) * total_earnings
  have expenses_eq : used_for_expenses = monthly_expenses := sorry
  have expense_calc : (1 - savings_rate) * (basic_salary + commission_rate * S) = monthly_expenses := sorry
  have simplify_eq : 0.80 * (1250 + 0.10 * S) = 2888 := sorry
  have open_eq : 1000 + 0.08 * S = 2888 := sorry
  have isolate_S : 0.08 * S = 1888 := sorry
  have solve_S : S = 1888 / 0.08 := sorry
  have final_S : S = 23600 := sorry
  exact final_S

end roberts_total_sales_l1659_165955


namespace distance_from_y_axis_l1659_165951

theorem distance_from_y_axis (P : ℝ × ℝ) (x : ℝ) (hx : P = (x, -9)) 
  (h : (abs (P.2) = 1/2 * abs (P.1))) :
  abs x = 18 :=
by
  sorry

end distance_from_y_axis_l1659_165951


namespace functional_eq_zero_l1659_165923

theorem functional_eq_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x * f x + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_eq_zero_l1659_165923


namespace maximum_teams_tied_for_most_wins_l1659_165941

/-- In a round-robin tournament with 8 teams, each team plays one game
    against each other team, and each game results in one team winning
    and one team losing. -/
theorem maximum_teams_tied_for_most_wins :
  ∀ (teams games wins : ℕ), 
    teams = 8 → 
    games = (teams * (teams - 1)) / 2 →
    wins = 28 →
    ∃ (max_tied_teams : ℕ), max_tied_teams = 5 :=
by
  sorry

end maximum_teams_tied_for_most_wins_l1659_165941


namespace remaining_amount_l1659_165918

def initial_amount : ℕ := 18
def spent_amount : ℕ := 16

theorem remaining_amount : initial_amount - spent_amount = 2 := 
by sorry

end remaining_amount_l1659_165918


namespace education_budget_l1659_165988

-- Definitions of the conditions
def total_budget : ℕ := 32 * 10^6  -- 32 million
def policing_budget : ℕ := total_budget / 2
def public_spaces_budget : ℕ := 4 * 10^6  -- 4 million

-- The theorem statement
theorem education_budget :
  total_budget - (policing_budget + public_spaces_budget) = 12 * 10^6 :=
by
  sorry

end education_budget_l1659_165988


namespace brenda_initial_points_l1659_165905

theorem brenda_initial_points
  (b : ℕ)  -- points scored by Brenda in her play
  (initial_advantage :ℕ := 22)  -- Brenda is initially 22 points ahead
  (david_score : ℕ := 32)  -- David scores 32 points
  (final_advantage : ℕ := 5)  -- Brenda is 5 points ahead after both plays
  (h : initial_advantage + b - david_score = final_advantage) :
  b = 15 :=
by
  sorry

end brenda_initial_points_l1659_165905


namespace even_function_a_value_l1659_165957

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x - 1) = ((-x)^2 + a * (-x) - 1)) ↔ a = 0 :=
by
  sorry

end even_function_a_value_l1659_165957
