import Mathlib

namespace NUMINAMATH_GPT_zero_squared_sum_l408_40849

theorem zero_squared_sum (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := 
by 
  sorry

end NUMINAMATH_GPT_zero_squared_sum_l408_40849


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l408_40895

theorem cylinder_volume_ratio (a b : ℕ) (h_dim : (a, b) = (9, 12)) :
  let r₁ := (a : ℝ) / (2 * Real.pi)
  let h₁ := (↑b : ℝ)
  let V₁ := (Real.pi * r₁^2 * h₁)
  let r₂ := (b : ℝ) / (2 * Real.pi)
  let h₂ := (↑a : ℝ)
  let V₂ := (Real.pi * r₂^2 * h₂)
  (if V₂ > V₁ then V₂ / V₁ else V₁ / V₂) = (16 / 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_cylinder_volume_ratio_l408_40895


namespace NUMINAMATH_GPT_find_x_l408_40808

theorem find_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) : x = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_x_l408_40808


namespace NUMINAMATH_GPT_find_k_l408_40830

theorem find_k (k : ℤ) (x : ℚ) (h1 : 5 * x + 3 * k = 24) (h2 : 5 * x + 3 = 0) : k = 9 := 
by
  sorry

end NUMINAMATH_GPT_find_k_l408_40830


namespace NUMINAMATH_GPT_base_seven_sum_l408_40814

def base_seven_to_ten (n : ℕ) : ℕ := 3 * 7^1 + 5 * 7^0   -- Converts 35_7 to base 10
def base_seven_to_ten' (m : ℕ) : ℕ := 1 * 7^1 + 2 * 7^0  -- Converts 12_7 to base 10

noncomputable def base_ten_product (a b : ℕ) : ℕ := (a * b) -- Computes product in base 10

noncomputable def base_ten_to_seven (p : ℕ) : ℕ :=        -- Converts base 10 to base 7
  let p1 := (p / 7 / 7) % 7
  let p2 := (p / 7) % 7
  let p3 := p % 7
  p1 * 100 + p2 * 10 + p3

noncomputable def sum_of_digits (a : ℕ) : ℕ :=             -- Sums digits in base 7
  let d1 := (a / 100) % 10
  let d2 := (a / 10) % 10
  let d3 := a % 10
  d1 + d2 + d3

noncomputable def base_ten_to_seven' (s : ℕ) : ℕ :=        -- Converts sum back to base 7
  let s1 := s / 7
  let s2 := s % 7
  s1 * 10 + s2

theorem base_seven_sum (n m : ℕ) : base_ten_to_seven' (sum_of_digits (base_ten_to_seven (base_ten_product (base_seven_to_ten n) (base_seven_to_ten' m)))) = 15 :=
by
  sorry

end NUMINAMATH_GPT_base_seven_sum_l408_40814


namespace NUMINAMATH_GPT_original_number_is_10_l408_40803

theorem original_number_is_10 (x : ℝ) (h : 2 * x + 5 = x / 2 + 20) : x = 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_original_number_is_10_l408_40803


namespace NUMINAMATH_GPT_rectangle_area_expression_l408_40863

theorem rectangle_area_expression {d x : ℝ} (h : d^2 = 29 * x^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = (10 / 29) :=
by {
 sorry
}

end NUMINAMATH_GPT_rectangle_area_expression_l408_40863


namespace NUMINAMATH_GPT_product_of_2020_numbers_even_l408_40884

theorem product_of_2020_numbers_even (a : ℕ → ℕ) 
  (h : (Finset.sum (Finset.range 2020) a) % 2 = 1) : 
  (Finset.prod (Finset.range 2020) a) % 2 = 0 :=
sorry

end NUMINAMATH_GPT_product_of_2020_numbers_even_l408_40884


namespace NUMINAMATH_GPT_range_of_b_l408_40843

theorem range_of_b (a b : ℝ) (h₁ : a ≤ -1) (h₂ : a * 2 * b - b - 3 * a ≥ 0) : b ≤ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_b_l408_40843


namespace NUMINAMATH_GPT_right_triangle_relation_l408_40893

theorem right_triangle_relation (a b c x : ℝ)
  (h : c^2 = a^2 + b^2)
  (altitude : a * b = c * x) :
  (1 / x^2) = (1 / a^2) + (1 / b^2) :=
sorry

end NUMINAMATH_GPT_right_triangle_relation_l408_40893


namespace NUMINAMATH_GPT_cole_drive_time_l408_40806

theorem cole_drive_time (D T1 T2 : ℝ) (h1 : T1 = D / 75) 
  (h2 : T2 = D / 105) (h3 : T1 + T2 = 6) : 
  (T1 * 60 = 210) :=
by sorry

end NUMINAMATH_GPT_cole_drive_time_l408_40806


namespace NUMINAMATH_GPT_equations_create_24_l408_40805

theorem equations_create_24 :
  ∃ (eq1 eq2 : ℤ),
  ((eq1 = 3 * (-6 + 4 + 10) ∧ eq1 = 24) ∧ 
   (eq2 = 4 - (-6 / 3) * 10 ∧ eq2 = 24)) ∧ 
   eq1 ≠ eq2 := 
by
  sorry

end NUMINAMATH_GPT_equations_create_24_l408_40805


namespace NUMINAMATH_GPT_graph_inverse_prop_function_quadrants_l408_40832

theorem graph_inverse_prop_function_quadrants :
  ∀ x : ℝ, x ≠ 0 → (x > 0 ∧ y = 4 / x → y > 0) ∨ (x < 0 ∧ y = 4 / x → y < 0) := 
sorry

end NUMINAMATH_GPT_graph_inverse_prop_function_quadrants_l408_40832


namespace NUMINAMATH_GPT_problem_proof_l408_40847

-- Formalizing the conditions of the problem
variable {a : ℕ → ℝ}  -- Define the arithmetic sequence
variable (d : ℝ)      -- Common difference of the arithmetic sequence
variable (a₅ a₆ a₇ : ℝ)  -- Specific terms in the sequence

-- The condition given in the problem
axiom cond1 : a 5 + a 6 + a 7 = 15

-- A definition for an arithmetic sequence
noncomputable def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Using the axiom to deduce that a₆ = 5
axiom prop_arithmetic : is_arithmetic_seq a d

-- We want to prove that sum of terms from a₃ to a₉ = 35
theorem problem_proof : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by sorry

end NUMINAMATH_GPT_problem_proof_l408_40847


namespace NUMINAMATH_GPT_integral_sqrt_a_squared_minus_x_squared_l408_40848

open Real

theorem integral_sqrt_a_squared_minus_x_squared (a : ℝ) :
  (∫ x in -a..a, sqrt (a^2 - x^2)) = 1/2 * π * a^2 :=
by
  sorry

end NUMINAMATH_GPT_integral_sqrt_a_squared_minus_x_squared_l408_40848


namespace NUMINAMATH_GPT_water_added_l408_40882

theorem water_added (x : ℝ) (salt_initial_percentage : ℝ) (salt_final_percentage : ℝ) 
   (evap_fraction : ℝ) (salt_added : ℝ) (W : ℝ) 
   (hx : x = 150) (h_initial_salt : salt_initial_percentage = 0.2) 
   (h_final_salt : salt_final_percentage = 1 / 3) 
   (h_evap_fraction : evap_fraction = 1 / 4) 
   (h_salt_added : salt_added = 20) : 
  W = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_water_added_l408_40882


namespace NUMINAMATH_GPT_hurricane_damage_in_euros_l408_40857

-- Define the conditions
def usd_damage : ℝ := 45000000  -- Damage in US dollars
def exchange_rate : ℝ := 0.9    -- Exchange rate from US dollars to Euros

-- Define the target value in Euros
def eur_damage : ℝ := 40500000  -- Expected damage in Euros

-- The theorem to prove
theorem hurricane_damage_in_euros :
  usd_damage * exchange_rate = eur_damage :=
by
  sorry

end NUMINAMATH_GPT_hurricane_damage_in_euros_l408_40857


namespace NUMINAMATH_GPT_no_integer_solutions_l408_40811

theorem no_integer_solutions (P Q : Polynomial ℤ) (a : ℤ) (hP1 : P.eval a = 0) 
  (hP2 : P.eval (a + 1997) = 0) (hQ : Q.eval 1998 = 2000) : 
  ¬ ∃ x : ℤ, Q.eval (P.eval x) = 1 := 
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l408_40811


namespace NUMINAMATH_GPT_minimal_total_cost_l408_40822

def waterway_length : ℝ := 100
def max_speed : ℝ := 50
def other_costs_per_hour : ℝ := 3240
def speed_at_ten_cost : ℝ := 10
def fuel_cost_at_ten : ℝ := 60
def proportionality_constant : ℝ := 0.06

noncomputable def total_cost (v : ℝ) : ℝ :=
  6 * v^2 + 324000 / v

theorem minimal_total_cost :
  (∃ v : ℝ, 0 < v ∧ v ≤ max_speed ∧ total_cost v = 16200) ∧ 
  (∀ v : ℝ, 0 < v ∧ v ≤ max_speed → total_cost v ≥ 16200) :=
sorry

end NUMINAMATH_GPT_minimal_total_cost_l408_40822


namespace NUMINAMATH_GPT_notebook_pre_tax_cost_eq_l408_40854

theorem notebook_pre_tax_cost_eq :
  (∃ (n c X : ℝ), n + c = 3 ∧ n = 2 + c ∧ 1.1 * X = 3.3 ∧ X = n + c → n = 2.5) :=
by
  sorry

end NUMINAMATH_GPT_notebook_pre_tax_cost_eq_l408_40854


namespace NUMINAMATH_GPT_inequality_proof_l408_40807

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 1) :
    (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l408_40807


namespace NUMINAMATH_GPT_distance_to_Rock_Mist_Mountains_l408_40894

theorem distance_to_Rock_Mist_Mountains (d_Sky_Falls : ℕ) (multiplier : ℕ) (d_Rock_Mist : ℕ) :
  d_Sky_Falls = 8 → multiplier = 50 → d_Rock_Mist = d_Sky_Falls * multiplier → d_Rock_Mist = 400 :=
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_distance_to_Rock_Mist_Mountains_l408_40894


namespace NUMINAMATH_GPT_Brian_Frodo_ratio_l408_40873

-- Definitions from the conditions
def Lily_tennis_balls : Int := 3
def Frodo_tennis_balls : Int := Lily_tennis_balls + 8
def Brian_tennis_balls : Int := 22

-- The proof statement
theorem Brian_Frodo_ratio :
  Brian_tennis_balls / Frodo_tennis_balls = 2 := by
  sorry

end NUMINAMATH_GPT_Brian_Frodo_ratio_l408_40873


namespace NUMINAMATH_GPT_florist_bouquets_is_36_l408_40864

noncomputable def florist_bouquets : Prop :=
  let r := 125
  let y := 125
  let o := 125
  let p := 125
  let rk := 45
  let yk := 61
  let ok := 30
  let pk := 40
  let initial_flowers := r + y + o + p
  let total_killed := rk + yk + ok + pk
  let remaining_flowers := initial_flowers - total_killed
  let flowers_per_bouquet := 9
  let bouquets := remaining_flowers / flowers_per_bouquet
  bouquets = 36

theorem florist_bouquets_is_36 : florist_bouquets :=
  by
    sorry

end NUMINAMATH_GPT_florist_bouquets_is_36_l408_40864


namespace NUMINAMATH_GPT_find_a_l408_40888

theorem find_a (a : ℕ) (h : a * 2 * 2^3 = 2^6) : a = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l408_40888


namespace NUMINAMATH_GPT_read_time_proof_l408_40899

noncomputable def read_time_problem : Prop :=
  ∃ (x y : ℕ), 
    x > 0 ∧
    y = 480 / x ∧
    (y - 5) = 480 / (x + 16) ∧
    y = 15

theorem read_time_proof : read_time_problem := 
sorry

end NUMINAMATH_GPT_read_time_proof_l408_40899


namespace NUMINAMATH_GPT_laptop_price_l408_40876

theorem laptop_price (cost upfront : ℝ) (upfront_percentage : ℝ) (upfront_eq : upfront = 240) (upfront_percentage_eq : upfront_percentage = 20) : 
  cost = 1200 :=
by
  sorry

end NUMINAMATH_GPT_laptop_price_l408_40876


namespace NUMINAMATH_GPT_fresh_grapes_weight_l408_40837

theorem fresh_grapes_weight :
  ∀ (F : ℝ), (∀ (water_content_fresh : ℝ) (water_content_dried : ℝ) (weight_dried : ℝ),
    water_content_fresh = 0.90 → water_content_dried = 0.20 → weight_dried = 3.125 →
    (F * 0.10 = 0.80 * weight_dried) → F = 78.125) := 
by
  intros F
  intros water_content_fresh water_content_dried weight_dried
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fresh_grapes_weight_l408_40837


namespace NUMINAMATH_GPT_domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l408_40842

-- For the function y = sqrt(3 + 2x)
theorem domain_sqrt_3_plus_2x (x : ℝ) : 3 + 2 * x ≥ 0 -> x ∈ Set.Ici (-3 / 2) :=
by
  sorry

-- For the function f(x) = 1 + sqrt(9 - x^2)
theorem domain_1_plus_sqrt_9_minus_x2 (x : ℝ) : 9 - x^2 ≥ 0 -> x ∈ Set.Icc (-3) 3 :=
by
  sorry

-- For the function φ(x) = sqrt(log((5x - x^2) / 4))
theorem domain_sqrt_log_5x_minus_x2_over_4 (x : ℝ) : (5 * x - x^2) / 4 > 0 ∧ (5 * x - x^2) / 4 ≥ 1 -> x ∈ Set.Icc 1 4 :=
by
  sorry

-- For the function y = sqrt(3 - x) + arccos((x - 2) / 3)
theorem domain_sqrt_3_minus_x_plus_arccos (x : ℝ) : 3 - x ≥ 0 ∧ -1 ≤ (x - 2) / 3 ∧ (x - 2) / 3 ≤ 1 -> x ∈ Set.Icc (-1) 3 :=
by
  sorry

end NUMINAMATH_GPT_domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l408_40842


namespace NUMINAMATH_GPT_abs_neg_three_eq_three_l408_40878

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end NUMINAMATH_GPT_abs_neg_three_eq_three_l408_40878


namespace NUMINAMATH_GPT_dark_squares_exceed_light_squares_by_one_l408_40853

theorem dark_squares_exceed_light_squares_by_one :
  let dark_squares := 25
  let light_squares := 24
  dark_squares - light_squares = 1 :=
by
  sorry

end NUMINAMATH_GPT_dark_squares_exceed_light_squares_by_one_l408_40853


namespace NUMINAMATH_GPT_simplify_expr_l408_40839

variable (a b : ℤ)

theorem simplify_expr :
  (22 * a + 60 * b) + (10 * a + 29 * b) - (9 * a + 50 * b) = 23 * a + 39 * b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l408_40839


namespace NUMINAMATH_GPT_polygon_sides_l408_40883

theorem polygon_sides (h : ∀ (n : ℕ), 360 / n = 36) : 10 = 10 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_l408_40883


namespace NUMINAMATH_GPT_doug_initial_marbles_l408_40823

theorem doug_initial_marbles (ed_marbles : ℕ) (diff_ed_doug : ℕ) (final_ed_marbles : ed_marbles = 27) (diff : diff_ed_doug = 5) :
  ∃ doug_initial_marbles : ℕ, doug_initial_marbles = 22 :=
by
  sorry

end NUMINAMATH_GPT_doug_initial_marbles_l408_40823


namespace NUMINAMATH_GPT_remainder_52_l408_40872

theorem remainder_52 (x y : ℕ) (k m : ℤ)
  (h₁ : x = 246 * k + 37)
  (h₂ : y = 357 * m + 53) :
  (x + y + 97) % 123 = 52 := by
  sorry

end NUMINAMATH_GPT_remainder_52_l408_40872


namespace NUMINAMATH_GPT_number_of_elements_in_set_S_l408_40835

-- Define the set S and its conditions
variable (S : Set ℝ) (n : ℝ) (sumS : ℝ)

-- Conditions given in the problem
axiom avg_S : sumS / n = 6.2
axiom new_avg_S : (sumS + 8) / n = 7

-- The statement to be proved
theorem number_of_elements_in_set_S : n = 10 := by 
  sorry

end NUMINAMATH_GPT_number_of_elements_in_set_S_l408_40835


namespace NUMINAMATH_GPT_eval_power_81_11_over_4_l408_40850

theorem eval_power_81_11_over_4 : 81^(11/4) = 177147 := by
  sorry

end NUMINAMATH_GPT_eval_power_81_11_over_4_l408_40850


namespace NUMINAMATH_GPT_interesting_seven_digit_numbers_l408_40800

theorem interesting_seven_digit_numbers :
  ∃ n : Fin 2 → ℕ, (∀ i : Fin 2, n i = 128) :=
by sorry

end NUMINAMATH_GPT_interesting_seven_digit_numbers_l408_40800


namespace NUMINAMATH_GPT_painter_total_fence_painted_l408_40859

theorem painter_total_fence_painted : 
  ∀ (L T W Th F : ℕ), 
  (T = W) → (W = Th) → 
  (L = T / 2) → 
  (F = 2 * T * (6 / 8)) → 
  (F = L + 300) → 
  (L + T + W + Th + F = 1500) :=
by
  sorry

end NUMINAMATH_GPT_painter_total_fence_painted_l408_40859


namespace NUMINAMATH_GPT_sin_half_pi_plus_A_l408_40877

theorem sin_half_pi_plus_A (A : Real) (h : Real.cos (Real.pi + A) = -1 / 2) :
  Real.sin (Real.pi / 2 + A) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_half_pi_plus_A_l408_40877


namespace NUMINAMATH_GPT_max_modulus_z_i_l408_40858

open Complex

theorem max_modulus_z_i (z : ℂ) (hz : abs z = 2) : ∃ z₂ : ℂ, abs z₂ = 2 ∧ abs (z₂ - I) = 3 :=
sorry

end NUMINAMATH_GPT_max_modulus_z_i_l408_40858


namespace NUMINAMATH_GPT_no_such_integers_and_function_l408_40869

theorem no_such_integers_and_function (f : ℝ → ℝ) (m n : ℤ) (h1 : ∀ x, f (f x) = 2 * f x - x - 2) (h2 : (m : ℝ) ≤ (n : ℝ) ∧ f m = n) : False :=
sorry

end NUMINAMATH_GPT_no_such_integers_and_function_l408_40869


namespace NUMINAMATH_GPT_number_of_bowls_l408_40880

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end NUMINAMATH_GPT_number_of_bowls_l408_40880


namespace NUMINAMATH_GPT_fourth_is_20_fewer_than_third_l408_40856

-- Definitions of the number of road signs at each intersection
def first_intersection := 40
def second_intersection := first_intersection + first_intersection / 4
def third_intersection := 2 * second_intersection
def total_signs := 270
def fourth_intersection := total_signs - (first_intersection + second_intersection + third_intersection)

-- Proving the fourth intersection has 20 fewer signs than the third intersection
theorem fourth_is_20_fewer_than_third : third_intersection - fourth_intersection = 20 :=
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_fourth_is_20_fewer_than_third_l408_40856


namespace NUMINAMATH_GPT_dave_initial_video_games_l408_40816

theorem dave_initial_video_games (non_working_games working_game_price total_earnings : ℕ) 
  (h1 : non_working_games = 2) 
  (h2 : working_game_price = 4) 
  (h3 : total_earnings = 32) : 
  non_working_games + total_earnings / working_game_price = 10 := 
by 
  sorry

end NUMINAMATH_GPT_dave_initial_video_games_l408_40816


namespace NUMINAMATH_GPT_multiplication_simplify_l408_40834

theorem multiplication_simplify :
  12 * (1 / 8) * 32 = 48 := 
sorry

end NUMINAMATH_GPT_multiplication_simplify_l408_40834


namespace NUMINAMATH_GPT_elroy_more_miles_l408_40844

theorem elroy_more_miles (m_last_year : ℝ) (m_this_year : ℝ) (collect_last_year : ℝ) :
  m_last_year = 4 → m_this_year = 2.75 → collect_last_year = 44 → 
  (collect_last_year / m_this_year - collect_last_year / m_last_year = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_elroy_more_miles_l408_40844


namespace NUMINAMATH_GPT_y_value_l408_40840

theorem y_value (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + 1 / y = 8) (h4 : y + 1 / x = 7 / 12) (h5 : x + y = 7) : y = 49 / 103 :=
by
  sorry

end NUMINAMATH_GPT_y_value_l408_40840


namespace NUMINAMATH_GPT_larger_number_hcf_lcm_l408_40867

theorem larger_number_hcf_lcm (a b : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 20) 
  (h_factor1 : factor1 = 13) 
  (h_factor2 : factor2 = 14) 
  (h_ab_hcf : Nat.gcd a b = hcf)
  (h_ab_lcm : Nat.lcm a b = hcf * factor1 * factor2) :
  max a b = 280 :=
by 
  sorry

end NUMINAMATH_GPT_larger_number_hcf_lcm_l408_40867


namespace NUMINAMATH_GPT_find_s_squared_l408_40829

-- Define the conditions and entities in Lean
variable (s : ℝ)
def passesThrough (x y : ℝ) (a b : ℝ) : Prop :=
  (y^2 / 9) - (x^2 / a^2) = 1

-- State the given conditions as hypotheses
axiom h₀ : passesThrough 0 3 3 1
axiom h₁ : passesThrough 5 (-3) 25 1
axiom h₂ : passesThrough s (-4) 25 1

-- State the theorem we want to prove
theorem find_s_squared : s^2 = 175 / 9 := by
  sorry

end NUMINAMATH_GPT_find_s_squared_l408_40829


namespace NUMINAMATH_GPT_triangle_inequality_violation_l408_40815

theorem triangle_inequality_violation (a b c : ℝ) (ha : a = 1) (hb : b = 2) (hc : c = 7) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  rw [ha, hb, hc]
  simp
  sorry

end NUMINAMATH_GPT_triangle_inequality_violation_l408_40815


namespace NUMINAMATH_GPT_number_of_people_l408_40887

-- Define the total number of candy bars
def total_candy_bars : ℝ := 5.0

-- Define the amount of candy each person gets
def candy_per_person : ℝ := 1.66666666699999

-- Define a theorem to state that dividing the total candy bars by candy per person gives 3 people
theorem number_of_people : total_candy_bars / candy_per_person = 3 :=
  by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_number_of_people_l408_40887


namespace NUMINAMATH_GPT_rectangle_area_l408_40861

open Classical

noncomputable def point := {x : ℝ × ℝ // x.1 >= 0 ∧ x.2 >= 0}

structure Triangle :=
  (X Y Z : point)

structure Rectangle :=
  (P Q R S : point)

def height_from (t : Triangle) : ℝ :=
  8

def xz_length (t : Triangle) : ℝ :=
  15

def ps_on_xz (r : Rectangle) (t : Triangle) : Prop :=
  r.S.val.1 = r.P.val.1 ∧ r.S.val.1 = t.X.val.1 ∧ r.S.val.2 = 0 ∧ r.P.val.2 = 0

def pq_is_one_third_ps (r : Rectangle) : Prop :=
  dist r.P.1 r.Q.1 = (1/3) * dist r.P.1 r.S.1

theorem rectangle_area : ∀ (R : Rectangle) (T : Triangle),
  height_from T = 8 → xz_length T = 15 → ps_on_xz R T → pq_is_one_third_ps R →
  (dist R.P.1 R.Q.1) * (dist R.P.1 R.S.1) = 4800/169 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rectangle_area_l408_40861


namespace NUMINAMATH_GPT_cos_angle_l408_40852

noncomputable def angle := -19 * Real.pi / 6

theorem cos_angle : Real.cos angle = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_GPT_cos_angle_l408_40852


namespace NUMINAMATH_GPT_max_trig_expression_l408_40879

open Real

theorem max_trig_expression (x y z : ℝ) :
  (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 := sorry

end NUMINAMATH_GPT_max_trig_expression_l408_40879


namespace NUMINAMATH_GPT_determine_base_l408_40871

theorem determine_base (x : ℕ) (h : 2 * x^3 + x + 6 = x^3 + 2 * x + 342) : x = 7 := 
sorry

end NUMINAMATH_GPT_determine_base_l408_40871


namespace NUMINAMATH_GPT_calculate_expression_l408_40892

theorem calculate_expression : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l408_40892


namespace NUMINAMATH_GPT_boys_girls_dance_l408_40819

theorem boys_girls_dance (b g : ℕ) 
  (h : ∀ n, (n <= b) → (n + 7) ≤ g) 
  (hb_lasts : b + 7 = g) :
  b = g - 7 := by
  sorry

end NUMINAMATH_GPT_boys_girls_dance_l408_40819


namespace NUMINAMATH_GPT_solution_set_l408_40862

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem solution_set (x : ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_def : ∀ x : ℝ, x >= 0 → f x = x^2 - 4 * x) :
    f (x + 2) < 5 ↔ -7 < x ∧ x < 3 :=
sorry

end NUMINAMATH_GPT_solution_set_l408_40862


namespace NUMINAMATH_GPT_transform_cos_to_base_form_l408_40845

theorem transform_cos_to_base_form :
  let f (x : ℝ) := Real.cos (2 * x + (Real.pi / 3))
  let g (x : ℝ) := Real.cos (2 * x)
  ∃ (shift : ℝ), shift = Real.pi / 6 ∧
    (∀ x : ℝ, f (x - shift) = g x) :=
by
  let f := λ x : ℝ => Real.cos (2 * x + (Real.pi / 3))
  let g := λ x : ℝ => Real.cos (2 * x)
  use Real.pi / 6
  sorry

end NUMINAMATH_GPT_transform_cos_to_base_form_l408_40845


namespace NUMINAMATH_GPT_number_of_height_groups_l408_40833

theorem number_of_height_groups
  (max_height : ℕ) (min_height : ℕ) (class_width : ℕ)
  (h_max : max_height = 186)
  (h_min : min_height = 167)
  (h_class_width : class_width = 3) :
  (max_height - min_height + class_width - 1) / class_width = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_height_groups_l408_40833


namespace NUMINAMATH_GPT_diamonds_balance_emerald_l408_40898

theorem diamonds_balance_emerald (D E : ℝ) (h1 : 9 * D = 4 * E) (h2 : 9 * D + E = 4 * E) : 3 * D = E := by
  sorry

end NUMINAMATH_GPT_diamonds_balance_emerald_l408_40898


namespace NUMINAMATH_GPT_shape_volume_to_surface_area_ratio_l408_40855

/-- 
Define the volume and surface area of our specific shape with given conditions:
1. Five unit cubes in a straight line.
2. An additional cube on top of the second cube.
3. Another cube beneath the fourth cube.

Prove that the ratio of the volume to the surface area is \( \frac{1}{4} \).
-/
theorem shape_volume_to_surface_area_ratio :
  let volume := 7
  let surface_area := 28
  volume / surface_area = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_shape_volume_to_surface_area_ratio_l408_40855


namespace NUMINAMATH_GPT_find_root_and_m_l408_40809

theorem find_root_and_m (m x₂ : ℝ) (h₁ : (1 : ℝ) * x₂ = 3) (h₂ : (1 : ℝ) + x₂ = -m) : 
  x₂ = 3 ∧ m = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_root_and_m_l408_40809


namespace NUMINAMATH_GPT_green_hats_count_l408_40813

theorem green_hats_count 
  (B G : ℕ)
  (h1 : B + G = 85)
  (h2 : 6 * B + 7 * G = 530) : 
  G = 20 :=
by
  sorry

end NUMINAMATH_GPT_green_hats_count_l408_40813


namespace NUMINAMATH_GPT_age_ratio_correct_l408_40889

noncomputable def RahulDeepakAgeRatio : Prop :=
  let R := 20
  let D := 8
  R / D = 5 / 2

theorem age_ratio_correct (R D : ℕ) (h1 : R + 6 = 26) (h2 : D = 8) : RahulDeepakAgeRatio :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_age_ratio_correct_l408_40889


namespace NUMINAMATH_GPT_trig_identity_l408_40886

variable (α : ℝ)
variable (h : Real.sin α = 3 / 5)

theorem trig_identity : Real.sin (Real.pi / 2 + 2 * α) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l408_40886


namespace NUMINAMATH_GPT_tom_total_expenditure_l408_40821

noncomputable def tom_spent_total : ℝ :=
  let skateboard_price := 9.46
  let skateboard_discount := 0.10 * skateboard_price
  let discounted_skateboard := skateboard_price - skateboard_discount

  let marbles_price := 9.56
  let marbles_discount := 0.10 * marbles_price
  let discounted_marbles := marbles_price - marbles_discount

  let shorts_price := 14.50

  let figures_price := 12.60
  let figures_discount := 0.20 * figures_price
  let discounted_figures := figures_price - figures_discount

  let puzzle_price := 6.35
  let puzzle_discount := 0.15 * puzzle_price
  let discounted_puzzle := puzzle_price - puzzle_discount

  let game_price_eur := 20.50
  let game_discount_eur := 0.05 * game_price_eur
  let discounted_game_eur := game_price_eur - game_discount_eur
  let exchange_rate := 1.12
  let discounted_game_usd := discounted_game_eur * exchange_rate

  discounted_skateboard + discounted_marbles + shorts_price + discounted_figures + discounted_puzzle + discounted_game_usd

theorem tom_total_expenditure : abs (tom_spent_total - 68.91) < 0.01 :=
by norm_num1; sorry

end NUMINAMATH_GPT_tom_total_expenditure_l408_40821


namespace NUMINAMATH_GPT_greatest_x_for_A_is_perfect_square_l408_40810

theorem greatest_x_for_A_is_perfect_square :
  ∃ x : ℕ, x = 2008 ∧ ∀ y : ℕ, (∃ k : ℕ, 2^182 + 4^y + 8^700 = k^2) → y ≤ 2008 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_x_for_A_is_perfect_square_l408_40810


namespace NUMINAMATH_GPT_inequality_sol_range_t_l408_40875

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem inequality_sol : {x : ℝ | f x > 2} = {x : ℝ | x < -5} ∪ {x : ℝ | 1 < x} :=
sorry

theorem range_t (t : ℝ) : (∀ x : ℝ, f x ≥ t^2 - 11/2 * t) ↔ (1/2 ≤ t ∧ t ≤ 5) :=
sorry

end NUMINAMATH_GPT_inequality_sol_range_t_l408_40875


namespace NUMINAMATH_GPT_somu_age_ratio_l408_40831

theorem somu_age_ratio (S F : ℕ) (h1 : S = 20) (h2 : S - 10 = (F - 10) / 5) : S / F = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_somu_age_ratio_l408_40831


namespace NUMINAMATH_GPT_max_cables_to_ensure_communication_l408_40818

theorem max_cables_to_ensure_communication
    (A B : ℕ) (n : ℕ) 
    (hA : A = 16) (hB : B = 12) (hn : n = 28) :
    (A * B ≤ 192) ∧ (A * B = 192) :=
by
  sorry

end NUMINAMATH_GPT_max_cables_to_ensure_communication_l408_40818


namespace NUMINAMATH_GPT_correct_system_of_equations_l408_40841

variable (x y : Real)

-- Conditions
def condition1 : Prop := y = x + 4.5
def condition2 : Prop := 0.5 * y = x - 1

-- Main statement representing the correct system of equations
theorem correct_system_of_equations : condition1 x y ∧ condition2 x y :=
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l408_40841


namespace NUMINAMATH_GPT_methane_tetrahedron_dot_product_l408_40826

noncomputable def tetrahedron_vectors_dot_product_sum : ℝ :=
  let edge_length := 1
  let dot_product := -1 / 3 * edge_length^2
  let pair_count := 6 -- number of pairs in sum of dot products
  pair_count * dot_product

theorem methane_tetrahedron_dot_product :
  tetrahedron_vectors_dot_product_sum = - (3 / 4) := by
  sorry

end NUMINAMATH_GPT_methane_tetrahedron_dot_product_l408_40826


namespace NUMINAMATH_GPT_simplify_division_l408_40838

theorem simplify_division :
  (27 * 10^9) / (9 * 10^5) = 30000 :=
  sorry

end NUMINAMATH_GPT_simplify_division_l408_40838


namespace NUMINAMATH_GPT_sin_cos_value_sin_plus_cos_value_l408_40860

noncomputable def given_condition (θ : ℝ) : Prop := 
  (Real.tan θ + 1 / Real.tan θ = 2)

theorem sin_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ * Real.cos θ = 1 / 2 :=
sorry

theorem sin_plus_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ + Real.cos θ = Real.sqrt 2 ∨ Real.sin θ + Real.cos θ = -Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_sin_cos_value_sin_plus_cos_value_l408_40860


namespace NUMINAMATH_GPT_number_of_girls_l408_40851

theorem number_of_girls (B G : ℕ) (h₁ : B = 6 * G / 5) (h₂ : B + G = 440) : G = 200 :=
by {
  sorry -- Proof steps here
}

end NUMINAMATH_GPT_number_of_girls_l408_40851


namespace NUMINAMATH_GPT_range_of_k_l408_40896

theorem range_of_k (k : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) ∧ 
   (x₁^3 - 3*x₁ = k ∧ x₂^3 - 3*x₂ = k ∧ x₃^3 - 3*x₃ = k)) ↔ (-2 < k ∧ k < 2) :=
sorry

end NUMINAMATH_GPT_range_of_k_l408_40896


namespace NUMINAMATH_GPT_multiplicative_inverse_correct_l408_40824

def A : ℕ := 123456
def B : ℕ := 654321
def m : ℕ := 1234567
def AB_mod : ℕ := (A * B) % m

def N : ℕ := 513629

theorem multiplicative_inverse_correct (h : AB_mod = 470160) : (470160 * N) % m = 1 := 
by 
  have hN : N = 513629 := rfl
  have hAB : AB_mod = 470160 := h
  sorry

end NUMINAMATH_GPT_multiplicative_inverse_correct_l408_40824


namespace NUMINAMATH_GPT_sequence_a2018_l408_40827

theorem sequence_a2018 (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) - 2 * a (n + 1) + a n = 1) 
  (h2 : a 18 = 0) 
  (h3 : a 2017 = 0) :
  a 2018 = 1000 :=
sorry

end NUMINAMATH_GPT_sequence_a2018_l408_40827


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l408_40868

theorem geometric_series_common_ratio (a S r : ℝ) 
  (hS : S = a / (1 - r)) 
  (h_modified : (a * r^2) / (1 - r) = S / 16) : 
  r = 1/4 ∨ r = -1/4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l408_40868


namespace NUMINAMATH_GPT_differential_savings_l408_40897

-- Defining conditions given in the problem
def initial_tax_rate : ℝ := 0.45
def new_tax_rate : ℝ := 0.30
def annual_income : ℝ := 48000

-- Statement of the theorem to prove the differential savings
theorem differential_savings : (annual_income * initial_tax_rate) - (annual_income * new_tax_rate) = 7200 := by
  sorry  -- providing the proof is not required

end NUMINAMATH_GPT_differential_savings_l408_40897


namespace NUMINAMATH_GPT_equation_of_symmetric_line_l408_40870

theorem equation_of_symmetric_line
  (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  (∀ x : ℝ, ∃ y : ℝ, (x = a * y + b)) → (∀ x : ℝ, ∃ y : ℝ, (y = (1/a) * x - (b/a))) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_symmetric_line_l408_40870


namespace NUMINAMATH_GPT_walkway_time_stopped_l408_40828

noncomputable def effective_speed_with_walkway (v_p v_w : ℝ) : ℝ := v_p + v_w
noncomputable def effective_speed_against_walkway (v_p v_w : ℝ) : ℝ := v_p - v_w

theorem walkway_time_stopped (v_p v_w : ℝ) (h1 : effective_speed_with_walkway v_p v_w = 2)
                            (h2 : effective_speed_against_walkway v_p v_w = 2 / 3) :
    (200 / v_p) = 150 :=
by sorry

end NUMINAMATH_GPT_walkway_time_stopped_l408_40828


namespace NUMINAMATH_GPT_mutually_exclusive_not_opposite_l408_40866

universe u

-- Define the colors and people involved
inductive Color
| black
| red
| white

inductive Person 
| A
| B
| C

-- Define a function that distributes the cards amongst the people
def distributes (cards : List Color) (people : List Person) : People -> Color :=
  sorry

-- Define events as propositions
def A_gets_red (d : Person -> Color) : Prop :=
  d Person.A = Color.red

def B_gets_red (d : Person -> Color) : Prop :=
  d Person.B = Color.red

-- The main theorem stating the problem
theorem mutually_exclusive_not_opposite 
  (d : Person -> Color)
  (h : A_gets_red d → ¬ B_gets_red d) : 
  ¬ ( ∀ (p : Prop), A_gets_red d ↔ p ) → B_gets_red d :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_not_opposite_l408_40866


namespace NUMINAMATH_GPT_find_A_for_diamond_l408_40846

def diamond (A B : ℕ) : ℕ := 4 * A + 3 * B + 7

theorem find_A_for_diamond (A : ℕ) (h : diamond A 7 = 76) : A = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_A_for_diamond_l408_40846


namespace NUMINAMATH_GPT_ice_cream_flavors_l408_40801

theorem ice_cream_flavors : (Nat.choose (4 + 4 - 1) (4 - 1) = 35) :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_flavors_l408_40801


namespace NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_same_plane_l408_40874

variable {Plane Line : Type}
variable {α β γ : Plane}
variable {m n : Line}

-- Define perpendicularity and parallelism as axioms for simplicity
axiom perp (L : Line) (P : Plane) : Prop
axiom parallel (L1 L2 : Line) : Prop

-- Assume conditions for the theorem
variables (h1 : perp m α) (h2 : perp n α)

-- The theorem proving the required relationship
theorem lines_parallel_if_perpendicular_to_same_plane : parallel m n := 
by
  sorry

end NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_same_plane_l408_40874


namespace NUMINAMATH_GPT_oscar_leap_difference_in_feet_l408_40820

theorem oscar_leap_difference_in_feet 
  (strides_per_gap : ℕ) 
  (leaps_per_gap : ℕ) 
  (total_distance : ℕ) 
  (num_poles : ℕ)
  (h1 : strides_per_gap = 54) 
  (h2 : leaps_per_gap = 15) 
  (h3 : total_distance = 5280) 
  (h4 : num_poles = 51) 
  : (total_distance / (strides_per_gap * (num_poles - 1)) -
       total_distance / (leaps_per_gap * (num_poles - 1)) = 5) :=
by
  sorry

end NUMINAMATH_GPT_oscar_leap_difference_in_feet_l408_40820


namespace NUMINAMATH_GPT_log_diff_eq_35_l408_40825

theorem log_diff_eq_35 {a b : ℝ} (h₁ : a > b) (h₂ : b > 1)
  (h₃ : (1 / Real.log a / Real.log b) + (1 / (Real.log b / Real.log a)) = Real.sqrt 1229) :
  (1 / (Real.log b / Real.log (a * b))) - (1 / (Real.log a / Real.log (a * b))) = 35 :=
sorry

end NUMINAMATH_GPT_log_diff_eq_35_l408_40825


namespace NUMINAMATH_GPT_minimum_value_fraction_l408_40804

theorem minimum_value_fraction (x : ℝ) (h : x > 6) : (∃ c : ℝ, c = 12 ∧ ((x = c) → (x^2 / (x - 6) = 18)))
  ∧ (∀ y : ℝ, y > 6 → y^2 / (y - 6) ≥ 18) :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_fraction_l408_40804


namespace NUMINAMATH_GPT_marie_profit_l408_40865

-- Define constants and conditions
def loaves_baked : ℕ := 60
def morning_price : ℝ := 3.00
def discount : ℝ := 0.25
def afternoon_price : ℝ := morning_price * (1 - discount)
def cost_per_loaf : ℝ := 1.00
def donated_loaves : ℕ := 5

-- Define the number of loaves sold and revenue
def morning_loaves : ℕ := loaves_baked / 3
def morning_revenue : ℝ := morning_loaves * morning_price

def remaining_after_morning : ℕ := loaves_baked - morning_loaves
def afternoon_loaves : ℕ := remaining_after_morning / 2
def afternoon_revenue : ℝ := afternoon_loaves * afternoon_price

def remaining_after_afternoon : ℕ := remaining_after_morning - afternoon_loaves
def unsold_loaves : ℕ := remaining_after_afternoon - donated_loaves

-- Define the total revenue and cost
def total_revenue : ℝ := morning_revenue + afternoon_revenue
def total_cost : ℝ := loaves_baked * cost_per_loaf

-- Define the profit
def profit : ℝ := total_revenue - total_cost

-- State the proof problem
theorem marie_profit : profit = 45 := by
  sorry

end NUMINAMATH_GPT_marie_profit_l408_40865


namespace NUMINAMATH_GPT_original_average_of_numbers_l408_40836

theorem original_average_of_numbers 
  (A : ℝ) 
  (h : (A * 15) + (11 * 15) = 51 * 15) : 
  A = 40 :=
sorry

end NUMINAMATH_GPT_original_average_of_numbers_l408_40836


namespace NUMINAMATH_GPT_problem_statement_l408_40802

variables {p q r s : ℝ}

theorem problem_statement 
  (h : (p - q) * (r - s) / (q - r) * (s - p) = 3 / 7) : 
  (p - r) * (q - s) / (p - q) * (r - s) = -4 / 3 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l408_40802


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l408_40885

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Definition of arithmetic sequence sum
def arithmeticSum (n : ℕ) : ℚ :=
  (n / 2) * (a 1 + a n)

-- Given condition
axiom condition : (a 6) / (a 5) = 9 / 11

theorem arithmetic_sequence_sum_ratio :
  (S 11) / (S 9) = 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l408_40885


namespace NUMINAMATH_GPT_range_of_S_l408_40881

theorem range_of_S (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 1) (S : ℝ) (hS : S = 3 * x^2 - 2 * y^2) :
  -2 / 3 < S ∧ S ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_S_l408_40881


namespace NUMINAMATH_GPT_polynomial_coefficients_sum_l408_40891

theorem polynomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ), 
  (∀ x : ℚ, (3 * x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 +
                            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
                            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  (a_0 = -512) →
  ((a_0 + a_1 * (1/3) + a_2 * (1/3)^2 + a_3 * (1/3)^3 + 
    a_4 * (1/3)^4 + a_5 * (1/3)^5 + a_6 * (1/3)^6 + 
    a_7 * (1/3)^7 + a_8 * (1/3)^8 + a_9 * (1/3)^9) = -1) →
  (a_1 / 3 + a_2 / 3^2 + a_3 / 3^3 + a_4 / 3^4 + a_5 / 3^5 + 
   a_6 / 3^6 + a_7 / 3^7 + a_8 / 3^8 + a_9 / 3^9 = 511) :=
by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_sum_l408_40891


namespace NUMINAMATH_GPT_M_intersect_N_eq_l408_40817

def M : Set ℝ := { y | ∃ x, y = x ^ 2 }
def N : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x^2 / 2 + y^2 ≤ 1) }

theorem M_intersect_N_eq : M ∩ { y | (y ∈ Set.univ) } = { y | 0 ≤ y ∧ y ≤ Real.sqrt 2 } :=
by
  sorry

end NUMINAMATH_GPT_M_intersect_N_eq_l408_40817


namespace NUMINAMATH_GPT_clock_hands_overlap_24_hours_l408_40812

theorem clock_hands_overlap_24_hours : 
  (∀ t : ℕ, t < 12 →  ∃ n : ℕ, (n = 11 ∧ (∃ h m : ℕ, h * 60 + m = t * 60 + m))) →
  (∃ k : ℕ, k = 22) :=
by
  sorry

end NUMINAMATH_GPT_clock_hands_overlap_24_hours_l408_40812


namespace NUMINAMATH_GPT_find_m_l408_40890

theorem find_m (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 < 2) ∧ (m^2 + m = 2) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l408_40890
