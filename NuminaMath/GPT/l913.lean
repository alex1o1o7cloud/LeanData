import Mathlib

namespace find_b_l913_91390

variables (a b : ℕ)

theorem find_b
  (h1 : a = 105)
  (h2 : a ^ 3 = 21 * 25 * 315 * b) :
  b = 7 :=
sorry

end find_b_l913_91390


namespace not_factorial_tail_numbers_lt_1992_l913_91373

noncomputable def factorial_tail_number_count (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5) + factorial_tail_number_count (n / 5)

theorem not_factorial_tail_numbers_lt_1992 :
  ∃ n, n < 1992 ∧ n = 1992 - (1992 / 5 + (1992 / 25 + (1992 / 125 + (1992 / 625 + 0)))) :=
sorry

end not_factorial_tail_numbers_lt_1992_l913_91373


namespace sum_of_cube_faces_l913_91375

theorem sum_of_cube_faces (a d b e c f : ℕ) (h1: a > 0) (h2: d > 0) (h3: b > 0) (h4: e > 0) (h5: c > 0) (h6: f > 0)
(h7 : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1491) :
  a + d + b + e + c + f = 41 := 
sorry

end sum_of_cube_faces_l913_91375


namespace smallest_d_value_l913_91336

theorem smallest_d_value : 
  ∃ d : ℝ, (d ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d + 5) = 4 * d) ∧ ∀ d' : ℝ, (d' ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d' + 5) = 4 * d') → (3 ≤ d') → d = 3 := 
by
  sorry

end smallest_d_value_l913_91336


namespace arithmetic_sequence_sum_l913_91367

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_9 : ℝ)
  (h1 : a 1 + a 4 + a 7 = 15)
  (h2 : a 3 + a 6 + a 9 = 3)
  (h_arith : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) :
  S_9 = 27 :=
by
  sorry

end arithmetic_sequence_sum_l913_91367


namespace find_number_l913_91392

theorem find_number (x : ℤ) : 17 * (x + 99) = 3111 → x = 84 :=
by
  sorry

end find_number_l913_91392


namespace trip_duration_17_hours_l913_91383

theorem trip_duration_17_hours :
  ∃ T : ℝ, 
    (∀ d₁ d₂ : ℝ,
      (d₁ / 30 + 1 + (150 - d₁) / 4 = T) ∧ 
      (d₁ / 30 + d₂ / 30 + (150 - (d₁ - d₂)) / 30 = T) ∧ 
      ((d₁ - d₂) / 4 + (150 - (d₁ - d₂)) / 30 = T))
  → T = 17 :=
by
  sorry

end trip_duration_17_hours_l913_91383


namespace Chris_has_6_Teslas_l913_91378

theorem Chris_has_6_Teslas (x y z : ℕ) (h1 : z = 13) (h2 : z = x + 10) (h3 : x = y / 2):
  y = 6 :=
by
  sorry

end Chris_has_6_Teslas_l913_91378


namespace sequence_periodic_l913_91388

theorem sequence_periodic (a : ℕ → ℕ) (h : ∀ n > 2, a (n + 1) = (a n ^ n + a (n - 1)) % 10) :
  ∃ n₀, ∀ k, a (n₀ + k) = a (n₀ + k + 4) :=
by {
  sorry
}

end sequence_periodic_l913_91388


namespace complex_fifth_roots_wrong_statement_l913_91358

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)
noncomputable def y : ℂ := Complex.exp (-2 * Real.pi * Complex.I / 5)

theorem complex_fifth_roots_wrong_statement :
  ¬(x^5 + y^5 = 1) :=
sorry

end complex_fifth_roots_wrong_statement_l913_91358


namespace complaints_over_3_days_l913_91397

def normal_complaints_per_day : ℕ := 120

def short_staffed_complaints_per_day : ℕ := normal_complaints_per_day * 4 / 3

def short_staffed_and_broken_self_checkout_complaints_per_day : ℕ := short_staffed_complaints_per_day * 12 / 10

def days_short_staffed_and_broken_self_checkout : ℕ := 3

def total_complaints (days : ℕ) (complaints_per_day : ℕ) : ℕ :=
  days * complaints_per_day

theorem complaints_over_3_days
  (n : ℕ := normal_complaints_per_day)
  (a : ℕ := short_staffed_complaints_per_day)
  (b : ℕ := short_staffed_and_broken_self_checkout_complaints_per_day)
  (d : ℕ := days_short_staffed_and_broken_self_checkout)
  : total_complaints d b = 576 :=
by {
  -- This is where the proof would go, e.g., using sorry to skip the proof for now.
  sorry
}

end complaints_over_3_days_l913_91397


namespace circle_packing_line_equation_l913_91323

theorem circle_packing_line_equation
  (d : ℝ) (n1 n2 n3 : ℕ) (slope : ℝ)
  (l_intersects_tangencies : ℝ → ℝ → Prop)
  (l_divides_R : Prop)
  (gcd_condition : ℕ → ℕ → ℕ → ℕ)
  (a b c : ℕ)
  (a_pos : 0 < a) (b_neg : b < 0) (c_pos : 0 < c)
  (gcd_abc : gcd_condition a b c = 1)
  (correct_equation_format : Prop) :
  n1 = 4 ∧ n2 = 4 ∧ n3 = 2 →
  d = 2 →
  slope = 5 →
  l_divides_R →
  l_intersects_tangencies 1 1 →
  l_intersects_tangencies 4 6 → 
  correct_equation_format → 
  a^2 + b^2 + c^2 = 42 :=
by sorry

end circle_packing_line_equation_l913_91323


namespace triangle_constructibility_l913_91315

variables (a b c γ : ℝ)

-- definition of the problem conditions
def valid_triangle_constructibility_conditions (a b_c_diff γ : ℝ) : Prop :=
  γ < 90 ∧ b_c_diff < a * Real.cos γ

-- constructibility condition
def is_constructible (a b c γ : ℝ) : Prop :=
  b - c < a * Real.cos γ

-- final theorem statement
theorem triangle_constructibility (a b c γ : ℝ) (h1 : γ < 90) (h2 : b > c) :
  (b - c < a * Real.cos γ) ↔ valid_triangle_constructibility_conditions a (b - c) γ :=
by sorry

end triangle_constructibility_l913_91315


namespace tangent_slope_at_zero_l913_91345

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 1)

theorem tangent_slope_at_zero :
  (deriv f 0) = 1 := by 
  sorry

end tangent_slope_at_zero_l913_91345


namespace power_function_monotonic_decreasing_l913_91368

theorem power_function_monotonic_decreasing (α : ℝ) (h : ∀ x y : ℝ, 0 < x → x < y → x^α > y^α) : α < 0 :=
sorry

end power_function_monotonic_decreasing_l913_91368


namespace problem_l913_91384

theorem problem : 3 + 15 / 3 - 2^3 = 0 := by
  sorry

end problem_l913_91384


namespace books_in_bin_after_actions_l913_91386

theorem books_in_bin_after_actions (x y : ℕ) (z : ℕ) (hx : x = 4) (hy : y = 3) (hz : z = 250) : x - y + (z / 100) * x = 11 :=
by
  rw [hx, hy, hz]
  -- x - y + (z / 100) * x = 4 - 3 + (250 / 100) * 4
  norm_num
  sorry

end books_in_bin_after_actions_l913_91386


namespace complex_number_solution_l913_91353

theorem complex_number_solution (z : ℂ) (i : ℂ) (H1 : i * i = -1) (H2 : z * i = 2 - 2 * i) : z = -2 - 2 * i :=
by
  sorry

end complex_number_solution_l913_91353


namespace work_days_l913_91369

theorem work_days (Dx Dy : ℝ) (H1 : Dy = 45) (H2 : 8 / Dx + 36 / Dy = 1) : Dx = 40 :=
by
  sorry

end work_days_l913_91369


namespace decimal_to_binary_correct_l913_91385

-- Define the decimal number
def decimal_number : ℕ := 25

-- Define the binary equivalent of 25
def binary_representation : ℕ := 0b11001

-- The condition indicating how the conversion is done
def is_binary_representation (decimal : ℕ) (binary : ℕ) : Prop :=
  -- Check if the binary representation matches the manual decomposition
  decimal = (binary / 2^4) * 2^4 + 
            ((binary % 2^4) / 2^3) * 2^3 + 
            (((binary % 2^4) % 2^3) / 2^2) * 2^2 + 
            ((((binary % 2^4) % 2^3) % 2^2) / 2^1) * 2^1 + 
            (((((binary % 2^4) % 2^3) % 2^2) % 2^1) / 2^0) * 2^0

-- Proof statement
theorem decimal_to_binary_correct : is_binary_representation decimal_number binary_representation :=
  by sorry

end decimal_to_binary_correct_l913_91385


namespace waiter_customers_before_lunch_l913_91355

theorem waiter_customers_before_lunch (X : ℕ) (A : X + 20 = 49) : X = 29 := by
  -- The proof is omitted based on the instructions
  sorry

end waiter_customers_before_lunch_l913_91355


namespace least_number_conditioned_l913_91317

theorem least_number_conditioned (n : ℕ) :
  n % 56 = 3 ∧ n % 78 = 3 ∧ n % 9 = 0 ↔ n = 2187 := 
sorry

end least_number_conditioned_l913_91317


namespace original_proposition_converse_negation_contrapositive_l913_91394

variable {a b : ℝ}

-- Original Proposition: If \( x^2 + ax + b \leq 0 \) has a non-empty solution set, then \( a^2 - 4b \geq 0 \)
theorem original_proposition (h : ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b ≥ 0 := sorry

-- Converse: If \( a^2 - 4b \geq 0 \), then \( x^2 + ax + b \leq 0 \) has a non-empty solution set
theorem converse (h : a^2 - 4 * b ≥ 0) : ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

-- Negation: If \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set, then \( a^2 - 4b < 0 \)
theorem negation (h : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b < 0 := sorry

-- Contrapositive: If \( a^2 - 4b < 0 \), then \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set
theorem contrapositive (h : a^2 - 4 * b < 0) : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

end original_proposition_converse_negation_contrapositive_l913_91394


namespace calculate_income_l913_91341

theorem calculate_income (I : ℝ) (T : ℝ) (a b c d : ℝ) (h1 : a = 0.15) (h2 : b = 40000) (h3 : c = 0.20) (h4 : T = 8000) (h5 : T = a * b + c * (I - b)) : I = 50000 :=
by
  sorry

end calculate_income_l913_91341


namespace midpoint_chord_hyperbola_l913_91361

theorem midpoint_chord_hyperbola (a b : ℝ) : 
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (∃ (mx my : ℝ), (mx / a^2 + my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2))) →
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) →
  ∃ (mx my : ℝ), (mx / a^2 - my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2) := 
sorry

end midpoint_chord_hyperbola_l913_91361


namespace new_students_joined_l913_91301

theorem new_students_joined (orig_avg_age new_avg_age : ℕ) (decrease_in_avg_age : ℕ) (orig_strength : ℕ) (new_students_avg_age : ℕ) :
  orig_avg_age = 40 ∧ new_avg_age = 36 ∧ decrease_in_avg_age = 4 ∧ orig_strength = 18 ∧ new_students_avg_age = 32 →
  ∃ x : ℕ, ((orig_strength * orig_avg_age) + (x * new_students_avg_age) = new_avg_age * (orig_strength + x)) ∧ x = 18 :=
by
  sorry

end new_students_joined_l913_91301


namespace problem_statement_l913_91360

-- Given conditions
noncomputable def S : ℕ → ℝ := sorry
axiom S_3_eq_2 : S 3 = 2
axiom S_6_eq_6 : S 6 = 6

-- Prove that a_{13} + a_{14} + a_{15} = 32
theorem problem_statement : (S 15 - S 12) = 32 :=
by sorry

end problem_statement_l913_91360


namespace find_m_of_parabola_and_line_l913_91366

theorem find_m_of_parabola_and_line (k m x1 x2 : ℝ) 
  (h_parabola_line : ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2} → 
                                   y = k * x + m → true)
  (h_intersection : x1 * x2 = -4) : m = 1 := 
sorry

end find_m_of_parabola_and_line_l913_91366


namespace shaded_l_shaped_area_l913_91313

def square (side : ℕ) : ℕ := side * side
def rectangle (length width : ℕ) : ℕ := length * width

theorem shaded_l_shaped_area :
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  areaABCD - total_area_small_shapes = 20 :=
by
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  have h : areaABCD - total_area_small_shapes = 20 := sorry
  exact h

end shaded_l_shaped_area_l913_91313


namespace difference_of_roots_l913_91304

theorem difference_of_roots 
  (a b c : ℝ)
  (h : ∀ x, x^2 - 2 * (a^2 + b^2 + c^2 - 2 * a * c) * x + (b^2 - a^2 - c^2 + 2 * a * c)^2 = 0) :
  ∃ (x1 x2 : ℝ), (x1 - x2 = 4 * b * (a - c)) ∨ (x1 - x2 = -4 * b * (a - c)) :=
sorry

end difference_of_roots_l913_91304


namespace max_value_of_f_l913_91376

def f (x : ℝ) : ℝ := 10 * x - 2 * x ^ 2

theorem max_value_of_f : ∃ M : ℝ, (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) :=
  ⟨12.5, sorry⟩

end max_value_of_f_l913_91376


namespace boy_speed_l913_91338

theorem boy_speed (d : ℝ) (v₁ v₂ : ℝ) (t₁ t₂ l e : ℝ) :
  d = 2 ∧ v₂ = 8 ∧ l = 7 / 60 ∧ e = 8 / 60 ∧ t₁ = d / v₁ ∧ t₂ = d / v₂ ∧ t₁ - t₂ = l + e → v₁ = 4 :=
by
  sorry

end boy_speed_l913_91338


namespace selling_price_of_radio_l913_91393

theorem selling_price_of_radio
  (cost_price : ℝ)
  (loss_percentage : ℝ) :
  loss_percentage = 13 → cost_price = 1500 → 
  (cost_price - (loss_percentage / 100) * cost_price) = 1305 :=
by
  intros h1 h2
  sorry

end selling_price_of_radio_l913_91393


namespace digital_root_8_pow_n_l913_91399

-- Define the conditions
def n : ℕ := 1989

-- Define the simplified problem
def digital_root (x : ℕ) : ℕ := if x % 9 = 0 then 9 else x % 9

-- Statement of the problem
theorem digital_root_8_pow_n : digital_root (8 ^ n) = 8 := by
  have mod_nine_eq : 8^n % 9 = 8 := by
    sorry
  simp [digital_root, mod_nine_eq]

end digital_root_8_pow_n_l913_91399


namespace weight_of_one_bowling_ball_l913_91371

-- Definitions from the problem conditions
def weight_canoe := 36
def num_canoes := 4
def num_bowling_balls := 9

-- Calculate the total weight of the canoes
def total_weight_canoes := num_canoes * weight_canoe

-- Prove the weight of one bowling ball
theorem weight_of_one_bowling_ball : (total_weight_canoes / num_bowling_balls) = 16 := by
  sorry

end weight_of_one_bowling_ball_l913_91371


namespace remainder_of_n_plus_2024_l913_91377

-- Define the assumptions
def n : ℤ := sorry  -- n will be some integer
def k : ℤ := sorry  -- k will be some integer

-- Main statement to be proved
theorem remainder_of_n_plus_2024 (h : n % 8 = 3) : (n + 2024) % 8 = 3 := sorry

end remainder_of_n_plus_2024_l913_91377


namespace value_of_a_l913_91351

theorem value_of_a (m n a : ℚ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + a) + 5) : 
  a = 2 / 5 :=
by
  sorry

end value_of_a_l913_91351


namespace prob_4_consecutive_baskets_prob_exactly_4_baskets_l913_91362

theorem prob_4_consecutive_baskets 
  (p : ℝ) (h : p = 1/2) : 
  (p^4 * (1 - p) + (1 - p) * p^4) = 1/16 :=
by sorry

theorem prob_exactly_4_baskets 
  (p : ℝ) (h : p = 1/2) : 
  5 * p^4 * (1 - p) = 5/32 :=
by sorry

end prob_4_consecutive_baskets_prob_exactly_4_baskets_l913_91362


namespace solve_inequality_l913_91326

theorem solve_inequality (x : ℝ) : (x + 1) / (x - 2) + (x - 3) / (3 * x) ≥ 4 ↔ x ∈ Set.Ico (-1/4) 0 ∪ Set.Ioc 2 3 := 
sorry

end solve_inequality_l913_91326


namespace correct_computation_gives_l913_91357

variable (x : ℝ)

theorem correct_computation_gives :
  ((3 * x - 12) / 6 = 60) → ((x / 3) + 12 = 160 / 3) :=
by
  sorry

end correct_computation_gives_l913_91357


namespace sum_of_digits_of_63_l913_91349

theorem sum_of_digits_of_63 (x y : ℕ) (h : 10 * x + y = 63) (h1 : x + y = 9) (h2 : x - y = 3) : x + y = 9 :=
by
  sorry

end sum_of_digits_of_63_l913_91349


namespace fraction_of_bones_in_foot_is_approx_one_eighth_l913_91306

def number_bones_human_body : ℕ := 206
def number_bones_one_foot : ℕ := 26
def fraction_bones_one_foot (total_bones foot_bones : ℕ) : ℚ := foot_bones / total_bones

theorem fraction_of_bones_in_foot_is_approx_one_eighth :
  fraction_bones_one_foot number_bones_human_body number_bones_one_foot = 13 / 103 ∧ 
  (abs ((13 / 103 : ℚ) - (1 / 8)) < 1 / 103) := 
sorry

end fraction_of_bones_in_foot_is_approx_one_eighth_l913_91306


namespace total_volume_of_water_in_container_l913_91319

def volume_each_hemisphere : ℝ := 4
def number_of_hemispheres : ℝ := 2735

theorem total_volume_of_water_in_container :
  (volume_each_hemisphere * number_of_hemispheres) = 10940 :=
by
  sorry

end total_volume_of_water_in_container_l913_91319


namespace certain_number_div_5000_l913_91344

theorem certain_number_div_5000 (num : ℝ) (h : num / 5000 = 0.0114) : num = 57 :=
sorry

end certain_number_div_5000_l913_91344


namespace largest_rhombus_diagonal_in_circle_l913_91382

theorem largest_rhombus_diagonal_in_circle (r : ℝ) (h : r = 10) : (2 * r = 20) :=
by
  sorry

end largest_rhombus_diagonal_in_circle_l913_91382


namespace pirate_treasure_division_l913_91303

theorem pirate_treasure_division (initial_treasure : ℕ) (p1_share p2_share p3_share p4_share p5_share remaining : ℕ)
  (h_initial : initial_treasure = 3000)
  (h_p1_share : p1_share = initial_treasure / 10)
  (h_p1_rem : remaining = initial_treasure - p1_share)
  (h_p2_share : p2_share = 2 * remaining / 10)
  (h_p2_rem : remaining = remaining - p2_share)
  (h_p3_share : p3_share = 3 * remaining / 10)
  (h_p3_rem : remaining = remaining - p3_share)
  (h_p4_share : p4_share = 4 * remaining / 10)
  (h_p4_rem : remaining = remaining - p4_share)
  (h_p5_share : p5_share = 5 * remaining / 10)
  (h_p5_rem : remaining = remaining - p5_share)
  (p6_p9_total : ℕ)
  (h_p6_p9_total : p6_p9_total = 20 * 4)
  (final_remaining : ℕ)
  (h_final_remaining : final_remaining = remaining - p6_p9_total) :
  final_remaining = 376 :=
by sorry

end pirate_treasure_division_l913_91303


namespace solve_for_x_and_n_l913_91396

theorem solve_for_x_and_n (x n : ℕ) : 2^n = x^2 + 1 ↔ (x = 0 ∧ n = 0) ∨ (x = 1 ∧ n = 1) := 
sorry

end solve_for_x_and_n_l913_91396


namespace cirrus_clouds_count_l913_91350

theorem cirrus_clouds_count 
  (cirrus_clouds cumulus_clouds cumulonimbus_clouds : ℕ)
  (h1 : cirrus_clouds = 4 * cumulus_clouds)
  (h2 : cumulus_clouds = 12 * cumulonimbus_clouds)
  (h3 : cumulonimbus_clouds = 3) : 
  cirrus_clouds = 144 :=
by sorry

end cirrus_clouds_count_l913_91350


namespace tower_height_l913_91379

theorem tower_height (h : ℝ) (hd : ¬ (h ≥ 200)) (he : ¬ (h ≤ 150)) (hf : ¬ (h ≤ 180)) : 180 < h ∧ h < 200 := 
by 
  sorry

end tower_height_l913_91379


namespace evaluate_expression_l913_91325

theorem evaluate_expression : 
  (Real.sqrt 3 + 3 + (1 / (Real.sqrt 3 + 3))^2 + 1 / (3 - Real.sqrt 3)) = Real.sqrt 3 + 3 + 5 / 6 := by
  sorry

end evaluate_expression_l913_91325


namespace mike_earnings_l913_91334

def prices : List ℕ := [5, 7, 12, 9, 6, 15, 11, 10]

theorem mike_earnings :
  List.sum prices = 75 :=
by
  sorry

end mike_earnings_l913_91334


namespace uma_income_l913_91309

theorem uma_income
  (x y : ℝ)
  (h1 : 8 * x - 7 * y = 2000)
  (h2 : 7 * x - 6 * y = 2000) :
  8 * x = 16000 := by
  sorry

end uma_income_l913_91309


namespace intersection_coordinates_l913_91308

theorem intersection_coordinates (x y : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : y = x + 1) : 
  x = 2 ∧ y = 3 := 
by 
  sorry

end intersection_coordinates_l913_91308


namespace intersection_eq_union_eq_l913_91333

noncomputable def A := {x : ℝ | -2 < x ∧ x <= 3}
noncomputable def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_eq : A ∩ B = {x : ℝ | -2 < x ∧ x < -1} := by
  sorry

theorem union_eq : A ∪ B = {x : ℝ | x <= 3 ∨ x > 4} := by
  sorry

end intersection_eq_union_eq_l913_91333


namespace prob_high_quality_correct_l913_91372

noncomputable def prob_high_quality_seeds :=
  let p_first := 0.955
  let p_second := 0.02
  let p_third := 0.015
  let p_fourth := 0.01
  let p_hq_first := 0.5
  let p_hq_second := 0.15
  let p_hq_third := 0.1
  let p_hq_fourth := 0.05
  let p_hq := p_first * p_hq_first + p_second * p_hq_second + p_third * p_hq_third + p_fourth * p_hq_fourth
  p_hq

theorem prob_high_quality_correct : prob_high_quality_seeds = 0.4825 :=
  by sorry

end prob_high_quality_correct_l913_91372


namespace max_area_of_triangle_l913_91346

noncomputable def maxAreaTriangle (m_a m_b m_c : ℝ) : ℝ :=
  1/3 * Real.sqrt (2 * (m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4))

theorem max_area_of_triangle (m_a m_b m_c : ℝ) (h1 : m_a ≤ 2) (h2 : m_b ≤ 3) (h3 : m_c ≤ 4) :
  maxAreaTriangle m_a m_b m_c ≤ 4 :=
sorry

end max_area_of_triangle_l913_91346


namespace opposite_direction_of_vectors_l913_91332

theorem opposite_direction_of_vectors
  (x : ℝ)
  (a : ℝ × ℝ := (x, 1))
  (b : ℝ × ℝ := (4, x)) :
  (∃ k : ℝ, k ≠ 0 ∧ a = -k • b) → x = -2 := 
sorry

end opposite_direction_of_vectors_l913_91332


namespace ball_hits_ground_at_l913_91354

variable (t : ℚ) 

def height_eqn (t : ℚ) : ℚ :=
  -16 * t^2 + 30 * t + 50

theorem ball_hits_ground_at :
  (height_eqn t = 0) -> t = 47 / 16 :=
by
  sorry

end ball_hits_ground_at_l913_91354


namespace DE_minimal_length_in_triangle_l913_91328

noncomputable def min_length_DE (BC AC : ℝ) (angle_B : ℝ) : ℝ :=
  if BC = 5 ∧ AC = 12 ∧ angle_B = 13 then 2 * Real.sqrt 3 else sorry

theorem DE_minimal_length_in_triangle :
  min_length_DE 5 12 13 = 2 * Real.sqrt 3 :=
sorry

end DE_minimal_length_in_triangle_l913_91328


namespace find_special_four_digit_square_l913_91380

theorem find_special_four_digit_square :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ ∃ (a b c d : ℕ), 
    n = 1000 * a + 100 * b + 10 * c + d ∧
    n = 8281 ∧
    a = c ∧
    b + 1 = d ∧
    n = (91 : ℕ) ^ 2 :=
by
  sorry

end find_special_four_digit_square_l913_91380


namespace diagonal_of_square_l913_91389

-- Definitions based on conditions
def square_area := 8 -- Area of the square is 8 square centimeters

def diagonal_length (x : ℝ) : Prop :=
  (1/2) * x ^ 2 = square_area

-- Proof problem statement
theorem diagonal_of_square : ∃ x : ℝ, diagonal_length x ∧ x = 4 := 
sorry  -- statement only, proof skipped

end diagonal_of_square_l913_91389


namespace probability_of_diamond_or_ace_at_least_one_l913_91312

noncomputable def prob_at_least_one_diamond_or_ace : ℚ := 
  1 - (9 / 13) ^ 2

theorem probability_of_diamond_or_ace_at_least_one :
  prob_at_least_one_diamond_or_ace = 88 / 169 := 
by
  sorry

end probability_of_diamond_or_ace_at_least_one_l913_91312


namespace radius_comparison_l913_91364

theorem radius_comparison 
  (a b c : ℝ)
  (da db dc r ρ : ℝ)
  (h₁ : da ≤ r)
  (h₂ : db ≤ r)
  (h₃ : dc ≤ r)
  (h₄ : 1 / 2 * (a * da + b * db + c * dc) = ρ * ((a + b + c) / 2)) :
  r ≥ ρ := 
sorry

end radius_comparison_l913_91364


namespace arrange_books_l913_91331

-- Definition of the problem
def total_books : ℕ := 5 + 3

-- Definition of the combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Prove that arranging 5 copies of Introduction to Geometry and 
-- 3 copies of Introduction to Number Theory into total_books positions can be done in 56 ways.
theorem arrange_books : combination total_books 5 = 56 := by
  sorry

end arrange_books_l913_91331


namespace simplify_expression_l913_91352

variable {x : ℝ}

theorem simplify_expression : 8 * x - 3 + 2 * x - 7 + 4 * x + 15 = 14 * x + 5 :=
by
  sorry

end simplify_expression_l913_91352


namespace triangle_area_x_value_l913_91302

theorem triangle_area_x_value (x : ℝ) (h1 : x > 0) (h2 : 1 / 2 * x * (2 * x) = 64) : x = 8 :=
by
  sorry

end triangle_area_x_value_l913_91302


namespace translation_result_l913_91370

-- Define the original point A
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := 3, y := -2 }

-- Define the translation function
def translate_right (p : Point) (dx : ℤ) : Point :=
  { x := p.x + dx, y := p.y }

-- Prove that translating point A 2 units to the right gives point A'
theorem translation_result :
  translate_right A 2 = { x := 5, y := -2 } :=
by sorry

end translation_result_l913_91370


namespace new_ratio_l913_91337

def milk_to_water_initial_ratio (M W : ℕ) : Prop := 4 * W = M

def total_volume (V M W : ℕ) : Prop := V = M + W

def new_water_volume (W_new W A : ℕ) : Prop := W_new = W + A

theorem new_ratio (V M W W_new A : ℕ) 
  (h1: milk_to_water_initial_ratio M W) 
  (h2: total_volume V M W) 
  (h3: A = 23) 
  (h4: new_water_volume W_new W A) 
  (h5: V = 45) 
  : 9 * W_new = 8 * M :=
by 
  sorry

end new_ratio_l913_91337


namespace triangle_area_zero_vertex_l913_91324

theorem triangle_area_zero_vertex (x1 y1 x2 y2 : ℝ) :
  (1 / 2) * |x1 * y2 - x2 * y1| = 
    abs (1 / 2 * (x1 * y2 - x2 * y1)) := 
sorry

end triangle_area_zero_vertex_l913_91324


namespace probability_circle_containment_l913_91307

theorem probability_circle_containment :
  let a_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
  let circle_C_contained (a : ℕ) : Prop := a > 3
  let m : ℕ := (a_set.filter circle_C_contained).card
  let n : ℕ := a_set.card
  let p : ℚ := m / n
  p = 4 / 7 := 
by
  sorry

end probability_circle_containment_l913_91307


namespace lisa_interest_l913_91322

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem lisa_interest (hP : ℝ := 1500) (hr : ℝ := 0.02) (hn : ℕ := 10) :
  (compound_interest hP hr hn - hP) = 328.49 :=
by
  sorry

end lisa_interest_l913_91322


namespace S_11_eq_22_l913_91398

variable {S : ℕ → ℕ}

-- Condition: given that S_8 - S_3 = 10
axiom h : S 8 - S 3 = 10

-- Proof goal: we want to show that S_11 = 22
theorem S_11_eq_22 : S 11 = 22 :=
by
  sorry

end S_11_eq_22_l913_91398


namespace least_positive_integer_solution_l913_91310

theorem least_positive_integer_solution :
  ∃ x : ℤ, x > 0 ∧ ∃ n : ℤ, (3 * x + 29)^2 = 43 * n ∧ x = 19 :=
by
  sorry

end least_positive_integer_solution_l913_91310


namespace problem_conditions_l913_91387

theorem problem_conditions (x y : ℝ) (hx : x * (Real.exp x + Real.log x + x) = 1) (hy : y * (2 * Real.log y + Real.log (Real.log y)) = 1) :
  (0 < x ∧ x < 1) ∧ (y - x > 1) ∧ (y - x < 3 / 2) :=
by
  sorry

end problem_conditions_l913_91387


namespace total_questions_in_two_hours_l913_91359

theorem total_questions_in_two_hours (r : ℝ) : 
  let Fiona_questions := 36 
  let Shirley_questions := Fiona_questions * r
  let Kiana_questions := (Fiona_questions + Shirley_questions) / 2
  let one_hour_total := Fiona_questions + Shirley_questions + Kiana_questions
  let two_hour_total := 2 * one_hour_total
  two_hour_total = 108 + 108 * r :=
by
  sorry

end total_questions_in_two_hours_l913_91359


namespace composite_p_squared_plus_36_l913_91381

theorem composite_p_squared_plus_36 (p : ℕ) (h_prime : Prime p) : 
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ (k * m = p^2 + 36) :=
by {
  sorry
}

end composite_p_squared_plus_36_l913_91381


namespace num_races_necessary_l913_91321

/-- There are 300 sprinters registered for a 200-meter dash at a local track meet,
where the track has only 8 lanes. In each race, 3 of the competitors advance to the
next round, while the rest are eliminated immediately. Determine how many races are
needed to identify the champion sprinter. -/
def num_races_to_champion (total_sprinters : ℕ) (lanes : ℕ) (advance_per_race : ℕ) : ℕ :=
  if h : advance_per_race < lanes ∧ lanes > 0 then
    let eliminations_per_race := lanes - advance_per_race
    let total_eliminations := total_sprinters - 1
    Nat.ceil (total_eliminations / eliminations_per_race)
  else
    0

theorem num_races_necessary
  (total_sprinters : ℕ)
  (lanes : ℕ)
  (advance_per_race : ℕ)
  (h_total_sprinters : total_sprinters = 300)
  (h_lanes : lanes = 8)
  (h_advance_per_race : advance_per_race = 3) :
  num_races_to_champion total_sprinters lanes advance_per_race = 60 := by
  sorry

end num_races_necessary_l913_91321


namespace range_of_m_l913_91320

-- Definitions for the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Prove that m ≥ 2 given the condition A ∪ B = A 
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≥ 2 :=
by
  sorry

end range_of_m_l913_91320


namespace find_a1_l913_91340

-- Definitions stemming from the conditions in the problem
def arithmetic_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

def is_geometric (a₁ a₃ a₆ : ℕ) : Prop :=
  ∃ r : ℕ, a₃ = r * a₁ ∧ a₆ = r^2 * a₁

theorem find_a1 :
  ∀ a₁ : ℕ,
    (arithmetic_seq a₁ 3 1 = a₁) ∧
    (arithmetic_seq a₁ 3 3 = a₁ + 6) ∧
    (arithmetic_seq a₁ 3 6 = a₁ + 15) ∧
    is_geometric a₁ (a₁ + 6) (a₁ + 15) →
    a₁ = 12 :=
by
  intros
  sorry

end find_a1_l913_91340


namespace negation_is_correct_l913_91300

-- Define the original proposition as a predicate on real numbers.
def original_prop : Prop := ∀ x : ℝ, 4*x^2 - 3*x + 2 < 0

-- State the negation of the original proposition
def negation_of_original_prop : Prop := ∃ x : ℝ, 4*x^2 - 3*x + 2 ≥ 0

-- The theorem to prove the correctness of the negation of the original proposition
theorem negation_is_correct : ¬original_prop ↔ negation_of_original_prop := by
  sorry

end negation_is_correct_l913_91300


namespace massive_crate_chocolate_bars_l913_91347

theorem massive_crate_chocolate_bars :
  (54 * 24 * 37 = 47952) :=
by
  sorry

end massive_crate_chocolate_bars_l913_91347


namespace simple_interest_rate_l913_91329

-- Define the entities and conditions
variables (P A T : ℝ) (R : ℝ)

-- Conditions given in the problem
def principal := P = 12500
def amount := A = 16750
def time := T = 8

-- Result that needs to be proved
def correct_rate := R = 4.25

-- Main statement to be proven: Given the conditions, the rate is 4.25%
theorem simple_interest_rate :
  principal P → amount A → time T → (A - P = (P * R * T) / 100) → correct_rate R :=
by
  intros hP hA hT hSI
  sorry

end simple_interest_rate_l913_91329


namespace eccentricity_of_ellipse_l913_91311

theorem eccentricity_of_ellipse :
  ∀ (A B : ℝ × ℝ) (has_axes_intersection : A.2 = 0 ∧ B.2 = 0) 
    (product_of_slopes : ∀ (P : ℝ × ℝ), P ≠ A ∧ P ≠ B → (P.2 / (P.1 - A.1)) * (P.2 / (P.1 + B.1)) = -1/2),
  ∃ (e : ℝ), e = 1 / Real.sqrt 2 :=
by
  sorry

end eccentricity_of_ellipse_l913_91311


namespace fraction_inequality_l913_91363

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < ((b + m) / (a + m)) :=
sorry

end fraction_inequality_l913_91363


namespace profit_percent_l913_91335

variable (C S : ℝ)
variable (h : (1 / 3) * S = 0.8 * C)

theorem profit_percent (h : (1 / 3) * S = 0.8 * C) : 
  ((S - C) / C) * 100 = 140 := 
by
  sorry

end profit_percent_l913_91335


namespace math_students_count_l913_91395

noncomputable def students_in_math (total_students history_students english_students all_three_classes two_classes : ℕ) : ℕ :=
total_students - history_students - english_students + (two_classes - all_three_classes)

theorem math_students_count :
  students_in_math 68 21 34 3 7 = 14 :=
by
  sorry

end math_students_count_l913_91395


namespace renata_final_money_l913_91330

-- Defining the initial condition and the sequence of financial transactions.
def initial_money := 10
def donation := 4
def prize := 90
def slot_loss1 := 50
def slot_loss2 := 10
def slot_loss3 := 5
def water_cost := 1
def lottery_ticket_cost := 1
def lottery_prize := 65

-- Prove that given all these transactions, the final amount of money is $94.
theorem renata_final_money :
  initial_money 
  - donation 
  + prize 
  - slot_loss1 
  - slot_loss2 
  - slot_loss3 
  - water_cost 
  - lottery_ticket_cost 
  + lottery_prize 
  = 94 := 
by
  sorry

end renata_final_money_l913_91330


namespace sequence_x_y_sum_l913_91327

theorem sequence_x_y_sum :
  ∃ (r x y : ℝ), 
    (r * 3125 = 625) ∧ 
    (r * 625 = 125) ∧ 
    (r * 125 = x) ∧ 
    (r * x = y) ∧ 
    (r * y = 1) ∧
    (r * 1 = 1/5) ∧ 
    (r * (1/5) = 1/25) ∧ 
    x + y = 30 := 
by
  -- A placeholder for the actual proof
  sorry

end sequence_x_y_sum_l913_91327


namespace correct_option_C_l913_91343

theorem correct_option_C (m n : ℤ) : 
  (4 * m + 1) * 2 * m = 8 * m^2 + 2 * m :=
by
  sorry

end correct_option_C_l913_91343


namespace race_time_l913_91391

theorem race_time (v_A v_B : ℝ) (t tB : ℝ) (h1 : 200 / v_A = t) (h2 : 144 / v_B = t) (h3 : 200 / v_B = t + 7) : t = 18 :=
by
  sorry

end race_time_l913_91391


namespace average_rst_l913_91374

theorem average_rst (r s t : ℝ) (h : (5 / 2) * (r + s + t) = 25) :
  (r + s + t) / 3 = 10 / 3 :=
sorry

end average_rst_l913_91374


namespace smallest_special_number_l913_91365

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l913_91365


namespace inequality_5positives_l913_91314

variable {x1 x2 x3 x4 x5 : ℝ}

theorem inequality_5positives (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end inequality_5positives_l913_91314


namespace greatest_s_property_l913_91339

noncomputable def find_greatest_s (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] : ℕ :=
if h : m > 0 ∧ n > 0 then m else 0

theorem greatest_s_property (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] (H : 0 < m) (H1 : 0 < n)  :
  ∃ s, (s = find_greatest_s m n p) ∧ s * n * p ≤ m * n * p :=
by 
  sorry

end greatest_s_property_l913_91339


namespace direct_proportion_k_l913_91305

theorem direct_proportion_k (k x : ℝ) : ((k-1) * x + k^2 - 1 = 0) ∧ (k ≠ 1) ↔ k = -1 := 
sorry

end direct_proportion_k_l913_91305


namespace find_f_1988_l913_91318

def f : ℕ+ → ℕ+ := sorry

axiom functional_equation (m n : ℕ+) : f (f m + f n) = m + n

theorem find_f_1988 : f 1988 = 1988 :=
by sorry

end find_f_1988_l913_91318


namespace height_of_sarah_building_l913_91316

-- Define the conditions
def shadow_length_building : ℝ := 75
def height_pole : ℝ := 15
def shadow_length_pole : ℝ := 30

-- Define the height of the building
def height_building : ℝ := 38

-- Height of Sarah's building given the conditions
theorem height_of_sarah_building (h : ℝ) (H1 : shadow_length_building = 75)
    (H2 : height_pole = 15) (H3 : shadow_length_pole = 30) :
    h = height_building :=
by
  -- State the ratio of the height of the pole to its shadow
  have ratio_pole : ℝ := height_pole / shadow_length_pole

  -- Set up the ratio for Sarah's building and solve for h
  have h_eq : ℝ := ratio_pole * shadow_length_building

  -- Provide the proof (skipped here)
  sorry

end height_of_sarah_building_l913_91316


namespace power_function_odd_l913_91348

-- Define the conditions
def f : ℝ → ℝ := sorry
def condition1 (f : ℝ → ℝ) : Prop := f 1 = 3

-- Define the statement of the problem as a Lean theorem
theorem power_function_odd (f : ℝ → ℝ) (h : condition1 f) : ∀ x, f (-x) = -f x := sorry

end power_function_odd_l913_91348


namespace rectangle_width_length_ratio_l913_91356

theorem rectangle_width_length_ratio (w : ℕ) (h : ℕ) (P : ℕ) (H1 : h = 10) (H2 : P = 30) (H3 : 2 * w + 2 * h = P) :
  w / h = 1 / 2 :=
by
  sorry

end rectangle_width_length_ratio_l913_91356


namespace indoor_tables_count_l913_91342

theorem indoor_tables_count
  (I : ℕ)  -- the number of indoor tables
  (O : ℕ)  -- the number of outdoor tables
  (H1 : O = 12)  -- Condition 1: O = 12
  (H2 : 3 * I + 3 * O = 60)  -- Condition 2: Total number of chairs
  : I = 8 :=
by
  -- Insert the actual proof here
  sorry

end indoor_tables_count_l913_91342
