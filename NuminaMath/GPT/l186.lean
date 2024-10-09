import Mathlib

namespace lisa_cleaning_time_l186_18650

theorem lisa_cleaning_time (L : ℝ) (h1 : (1 / L) + (1 / 12) = 1 / 4.8) : L = 8 :=
sorry

end lisa_cleaning_time_l186_18650


namespace service_center_location_l186_18680

def serviceCenterMilepost (x3 x10 : ℕ) (r : ℚ) : ℚ :=
  x3 + r * (x10 - x3)

theorem service_center_location :
  (serviceCenterMilepost 50 170 (2/3) : ℚ) = 130 :=
by
  -- placeholder for the actual proof
  sorry

end service_center_location_l186_18680


namespace find_second_offset_l186_18686

theorem find_second_offset 
  (diagonal : ℝ) (offset1 : ℝ) (area_quad : ℝ) (offset2 : ℝ)
  (h1 : diagonal = 20) (h2 : offset1 = 9) (h3 : area_quad = 150) :
  offset2 = 6 :=
by
  sorry

end find_second_offset_l186_18686


namespace expression_of_fn_l186_18669

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
if n = 0 then x else f (n - 1) x / (1 + n * x)

theorem expression_of_fn (n : ℕ) (x : ℝ) (hn : 1 ≤ n) : f n x = x / (1 + n * x) :=
sorry

end expression_of_fn_l186_18669


namespace minimum_value_frac_l186_18677

theorem minimum_value_frac (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 2) :
  (p + q) / (p * q * r) ≥ 9 :=
sorry

end minimum_value_frac_l186_18677


namespace polynomial_geometric_roots_k_value_l186_18678

theorem polynomial_geometric_roots_k_value 
    (j k : ℝ)
    (h : ∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 0 ∧ 
      (∀ u v : ℝ, (u = a ∨ u = a * r ∨ u = a * r^2 ∨ u = a * r^3) →
        (v = a ∨ v = a * r ∨ v = a * r^2 ∨ v = a * r^3) →
        u ≠ v) ∧ 
      (a + a * r + a * r^2 + a * r^3 = 0) ∧
      (a^4 * r^6 = 900)) :
  k = -900 :=
sorry

end polynomial_geometric_roots_k_value_l186_18678


namespace gravel_weight_is_correct_l186_18640

def weight_of_gravel (total_weight : ℝ) (fraction_sand : ℝ) (fraction_water : ℝ) : ℝ :=
  total_weight - (fraction_sand * total_weight + fraction_water * total_weight)

theorem gravel_weight_is_correct :
  weight_of_gravel 23.999999999999996 (1 / 3) (1 / 4) = 10 :=
by
  sorry

end gravel_weight_is_correct_l186_18640


namespace larger_of_two_numbers_l186_18647

theorem larger_of_two_numbers (A B : ℕ) (hcf lcm : ℕ) (h1 : hcf = 23)
                              (h2 : lcm = hcf * 14 * 15) 
                              (h3 : lcm = A * B) (h4 : A = 23 * 14) 
                              (h5 : B = 23 * 15) : max A B = 345 :=
    sorry

end larger_of_two_numbers_l186_18647


namespace investment_final_value_l186_18671

theorem investment_final_value 
  (original_investment : ℝ) 
  (increase_percentage : ℝ) 
  (original_investment_eq : original_investment = 12500)
  (increase_percentage_eq : increase_percentage = 2.15) : 
  original_investment * (1 + increase_percentage) = 39375 := 
by
  sorry

end investment_final_value_l186_18671


namespace sum_of_children_ages_l186_18628

theorem sum_of_children_ages :
  ∃ E: ℕ, E = 12 ∧ 
  (∃ a b c d e : ℕ, a = E ∧ b = E - 2 ∧ c = E - 4 ∧ d = E - 6 ∧ e = E - 8 ∧ 
   a + b + c + d + e = 40) :=
sorry

end sum_of_children_ages_l186_18628


namespace radius_increase_is_0_31_l186_18643

noncomputable def increase_in_radius (initial_radius : ℝ) (odometer_summer : ℝ) (odometer_winter : ℝ) (miles_to_inches : ℝ) : ℝ :=
  let circumference_summer := 2 * Real.pi * initial_radius
  let distance_per_rotation_summer := circumference_summer / miles_to_inches
  let rotations_summer := odometer_summer / distance_per_rotation_summer
  let rotations_winter := odometer_winter / distance_per_rotation_summer
  let distance_winter := rotations_winter * distance_per_rotation_summer
  let new_radius := (distance_winter * miles_to_inches) / (2 * rotations_winter * Real.pi)
  new_radius - initial_radius

theorem radius_increase_is_0_31 : 
    increase_in_radius 16 530 520 63360 = 0.31 := 
by
    sorry

end radius_increase_is_0_31_l186_18643


namespace g_at_6_l186_18629

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_at_3 : g 3 = 4

theorem g_at_6 : g 6 = 8 :=
by 
  sorry

end g_at_6_l186_18629


namespace tan_4530_l186_18699

noncomputable def tan_of_angle (deg : ℝ) : ℝ := Real.tan (deg * Real.pi / 180)

theorem tan_4530 : tan_of_angle 4530 = -1 / Real.sqrt 3 := sorry

end tan_4530_l186_18699


namespace value_of_c_l186_18665

theorem value_of_c (a b c : ℕ) (hab : b = 1) (hd : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_pow : (10 * a + b)^2 = 100 * c + 10 * c + b) (h_gt : 100 * c + 10 * c + b > 300) : 
  c = 4 :=
sorry

end value_of_c_l186_18665


namespace range_of_m_l186_18641

theorem range_of_m (m : ℝ) (h1 : m + 3 > 0) (h2 : m - 1 < 0) : -3 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l186_18641


namespace abc_equality_l186_18681

theorem abc_equality (a b c : ℕ) (h1 : b = a^2 - a) (h2 : c = b^2 - b) (h3 : a = c^2 - c) : 
  a = 2 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end abc_equality_l186_18681


namespace number_of_players_l186_18655

variable (total_socks : ℕ) (socks_per_player : ℕ)

theorem number_of_players (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 := by
  -- proof steps will go here
  sorry

end number_of_players_l186_18655


namespace keith_gave_away_p_l186_18659

theorem keith_gave_away_p (k_init : Nat) (m_init : Nat) (final_pears : Nat) (k_gave_away : Nat) (total_init: Nat := k_init + m_init) :
  k_init = 47 →
  m_init = 12 →
  final_pears = 13 →
  k_gave_away = total_init - final_pears →
  k_gave_away = 46 :=
by
  -- Insert proof here (skip using sorry)
  sorry

end keith_gave_away_p_l186_18659


namespace winnie_proof_l186_18667

def winnie_problem : Prop :=
  let initial_count := 2017
  let multiples_of_3 := initial_count / 3
  let multiples_of_6 := initial_count / 6
  let multiples_of_27 := initial_count / 27
  let multiples_to_erase_3 := multiples_of_3
  let multiples_to_reinstate_6 := multiples_of_6
  let multiples_to_erase_27 := multiples_of_27
  let final_count := initial_count - multiples_to_erase_3 + multiples_to_reinstate_6 - multiples_to_erase_27
  initial_count - final_count = 373

theorem winnie_proof : winnie_problem := by
  sorry

end winnie_proof_l186_18667


namespace max_not_divisible_by_3_l186_18692

theorem max_not_divisible_by_3 (a b c d e f : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) (h7 : 3 ∣ (a * b * c * d * e * f)) : 
  ∃ x y z u v, ((x = a ∧ y = b ∧ z = c ∧ u = d ∧ v = e) ∨ (x = a ∧ y = b ∧ z = c ∧ u = d ∧ v = f) ∨ (x = a ∧ y = b ∧ z = c ∧ u = e ∧ v = f) ∨ (x = a ∧ y = b ∧ z = d ∧ u = e ∧ v = f) ∨ (x = a ∧ y = c ∧ z = d ∧ u = e ∧ v = f) ∨ (x = b ∧ y = c ∧ z = d ∧ u = e ∧ v = f)) ∧ (¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z) ∧ ¬ (3 ∣ u) ∧ ¬ (3 ∣ v)) :=
sorry

end max_not_divisible_by_3_l186_18692


namespace ten_crates_probability_l186_18604

theorem ten_crates_probability (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  let num_crates := 10
  let crate_dimensions := [3, 4, 6]
  let target_height := 41

  -- Definition of the generating function coefficients and constraints will be complex,
  -- so stating the specific problem directly.
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ m = 190 ∧ n = 2187 →
  let probability := (m : ℚ) / (n : ℚ)
  probability = (190 : ℚ) / 2187 := 
by
  sorry

end ten_crates_probability_l186_18604


namespace cone_base_radius_l186_18679

theorem cone_base_radius (R : ℝ) (theta : ℝ) (radius : ℝ) (hR : R = 30) (hTheta : theta = 120) :
    2 * Real.pi * radius = (theta / 360) * 2 * Real.pi * R → radius = 10 :=
by
  intros h
  sorry

end cone_base_radius_l186_18679


namespace abs_neg_eq_five_l186_18612

theorem abs_neg_eq_five (a : ℝ) : abs (-a) = 5 ↔ (a = 5 ∨ a = -5) :=
by
  sorry

end abs_neg_eq_five_l186_18612


namespace triangle_obtuse_of_eccentricities_l186_18696

noncomputable def is_obtuse_triangle (a b m : ℝ) : Prop :=
  a^2 + b^2 - m^2 < 0

theorem triangle_obtuse_of_eccentricities (a b m : ℝ) (ha : a > 0) (hm : m > b) (hb : b > 0)
  (ecc_cond : (Real.sqrt (a^2 + b^2) / a) * (Real.sqrt (m^2 - b^2) / m) > 1) :
  is_obtuse_triangle a b m := 
sorry

end triangle_obtuse_of_eccentricities_l186_18696


namespace distance_proof_l186_18616

theorem distance_proof (d : ℝ) (h1 : d < 6) (h2 : d > 5) (h3 : d > 4) : d ∈ Set.Ioo 5 6 :=
by
  sorry

end distance_proof_l186_18616


namespace slope_of_line_l186_18615

theorem slope_of_line (x y : ℝ) (h : 6 * x + 7 * y - 3 = 0) : - (6 / 7) = -6 / 7 := 
by
  sorry

end slope_of_line_l186_18615


namespace sin_max_value_l186_18625

open Real

theorem sin_max_value (a b : ℝ) (h : sin (a + b) = sin a + sin b) :
  sin a ≤ 1 :=
by
  sorry

end sin_max_value_l186_18625


namespace original_painting_width_l186_18668

theorem original_painting_width {W : ℝ} 
  (orig_height : ℝ) (print_height : ℝ) (print_width : ℝ)
  (h1 : orig_height = 10) 
  (h2 : print_height = 25)
  (h3 : print_width = 37.5) :
  W = 15 :=
  sorry

end original_painting_width_l186_18668


namespace determine_positive_intervals_l186_18648

noncomputable def positive_intervals (x : ℝ) : Prop :=
  (x+1) * (x-1) * (x-3) > 0

theorem determine_positive_intervals :
  ∀ x : ℝ, (positive_intervals x ↔ (x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) ∨ x ∈ Set.Ioi (3 : ℝ))) :=
by
  sorry

end determine_positive_intervals_l186_18648


namespace tips_collected_l186_18670

-- Definitions based on conditions
def total_collected : ℕ := 240
def hourly_wage : ℕ := 10
def hours_worked : ℕ := 19

-- Correct answer translated into a proof problem
theorem tips_collected : total_collected - (hours_worked * hourly_wage) = 50 := by
  sorry

end tips_collected_l186_18670


namespace expected_number_of_defective_products_l186_18614

theorem expected_number_of_defective_products 
  (N : ℕ) (D : ℕ) (n : ℕ) (hN : N = 15000) (hD : D = 1000) (hn : n = 150) :
  n * (D / N : ℚ) = 10 := 
by {
  sorry
}

end expected_number_of_defective_products_l186_18614


namespace point_not_in_second_quadrant_l186_18660

theorem point_not_in_second_quadrant (m : ℝ) : ¬ (m^2 + m ≤ 0 ∧ m - 1 ≥ 0) :=
by
  sorry

end point_not_in_second_quadrant_l186_18660


namespace neg_five_power_zero_simplify_expression_l186_18635

-- Proof statement for the first question.
theorem neg_five_power_zero : (-5 : ℝ)^0 = 1 := 
by sorry

-- Proof statement for the second question.
theorem simplify_expression (a b : ℝ) : ((-2 * a^2)^2) * (3 * a * b^2) = 12 * a^5 * b^2 := 
by sorry

end neg_five_power_zero_simplify_expression_l186_18635


namespace segment_AC_length_l186_18600

-- Define segments AB and BC
def AB : ℝ := 4
def BC : ℝ := 3

-- Define segment AC in terms of the conditions given
def AC_case1 : ℝ := AB - BC
def AC_case2 : ℝ := AB + BC

-- The proof problem statement
theorem segment_AC_length : AC_case1 = 1 ∨ AC_case2 = 7 := by
  sorry

end segment_AC_length_l186_18600


namespace fractionD_is_unchanged_l186_18606

-- Define variables x and y
variable (x y : ℚ)

-- Define the fractions
def fractionA := x / (y + 1)
def fractionB := (x + y) / (x + 1)
def fractionC := (x * y) / (x + y)
def fractionD := (2 * x) / (3 * x - y)

-- Define the transformation
def transform (a b : ℚ) : ℚ × ℚ := (3 * a, 3 * b)

-- Define the new fractions after transformation
def newFractionA := (3 * x) / (3 * y + 1)
def newFractionB := (3 * x + 3 * y) / (3 * x + 1)
def newFractionC := (9 * x * y) / (3 * x + 3 * y)
def newFractionD := (6 * x) / (9 * x - 3 * y)

-- The proof problem statement
theorem fractionD_is_unchanged :
  fractionD x y = newFractionD x y ∧
  fractionA x y ≠ newFractionA x y ∧
  fractionB x y ≠ newFractionB x y ∧
  fractionC x y ≠ newFractionC x y := sorry

end fractionD_is_unchanged_l186_18606


namespace probability_team_A_champions_l186_18626

theorem probability_team_A_champions : 
  let p : ℚ := 1 / 2 
  let prob_team_A_win_next := p
  let prob_team_B_win_next_A_win_after := p * p
  prob_team_A_win_next + prob_team_B_win_next_A_win_after = 3 / 4 :=
by
  sorry

end probability_team_A_champions_l186_18626


namespace spellbook_cost_in_gold_l186_18652

-- Define the constants
def num_spellbooks : ℕ := 5
def cost_potion_kit_in_silver : ℕ := 20
def num_potion_kits : ℕ := 3
def cost_owl_in_gold : ℕ := 28
def conversion_rate : ℕ := 9
def total_payment_in_silver : ℕ := 537

-- Define the problem to prove the cost of each spellbook in gold given the conditions
theorem spellbook_cost_in_gold : (total_payment_in_silver 
  - (cost_potion_kit_in_silver * num_potion_kits + cost_owl_in_gold * conversion_rate)) / num_spellbooks / conversion_rate = 5 := 
  by
  sorry

end spellbook_cost_in_gold_l186_18652


namespace value_of_X_when_S_reaches_15000_l186_18639

def X : Nat → Nat
| 0       => 5
| (n + 1) => X n + 3

def S : Nat → Nat
| 0       => 0
| (n + 1) => S n + X (n + 1)

theorem value_of_X_when_S_reaches_15000 :
  ∃ n, S n ≥ 15000 ∧ X n = 299 := by
  sorry

end value_of_X_when_S_reaches_15000_l186_18639


namespace sum_of_coefficients_l186_18638

theorem sum_of_coefficients :
  ∃ (A B C D E F G H J K : ℤ),
  (∀ x y : ℤ, 125 * x ^ 8 - 2401 * y ^ 8 = (A * x + B * y) * (C * x ^ 4 + D * x * y + E * y ^ 4) * (F * x + G * y) * (H * x ^ 4 + J * x * y + K * y ^ 4))
  ∧ A + B + C + D + E + F + G + H + J + K = 102 := 
sorry

end sum_of_coefficients_l186_18638


namespace b_investment_l186_18691

noncomputable def B_share := 880
noncomputable def A_share := 560
noncomputable def A_investment := 7000
noncomputable def C_investment := 18000
noncomputable def total_investment (B: ℝ) := A_investment + B + C_investment

theorem b_investment (B : ℝ) (P : ℝ)
    (h1 : 7000 / total_investment B * P = A_share)
    (h2 : B / total_investment B * P = B_share) : B = 8000 :=
by
  sorry

end b_investment_l186_18691


namespace product_is_zero_l186_18624

theorem product_is_zero (b : ℤ) (h : b = 3) :
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b * (b + 1) * (b + 2) = 0 :=
by {
  -- Substituting b = 3
  -- (3-5) * (3-4) * (3-3) * (3-2) * (3-1) * 3 * (3+1) * (3+2)
  -- = (-2) * (-1) * 0 * 1 * 2 * 3 * 4 * 5
  -- = 0
  sorry
}

end product_is_zero_l186_18624


namespace evaluate_fg_of_8_l186_18682

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem evaluate_fg_of_8 : f (g 8) = 211 :=
by
  sorry

end evaluate_fg_of_8_l186_18682


namespace velocity_is_zero_at_t_equals_2_l186_18609

def displacement (t : ℝ) : ℝ := -2 * t^2 + 8 * t

theorem velocity_is_zero_at_t_equals_2 : (deriv displacement 2 = 0) :=
by
  -- The definition step from (a). 
  let v := deriv displacement
  -- This would skip the proof itself, as instructed.
  sorry

end velocity_is_zero_at_t_equals_2_l186_18609


namespace unique_quotient_is_9742_l186_18605

theorem unique_quotient_is_9742 :
  ∃ (d4 d3 d2 d1 : ℕ),
    (d2 = d1 + 2) ∧
    (d4 = d3 + 2) ∧
    (0 ≤ d1 ∧ d1 ≤ 9) ∧
    (0 ≤ d2 ∧ d2 ≤ 9) ∧
    (0 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d4 * 1000 + d3 * 100 + d2 * 10 + d1 = 9742) :=
by sorry

end unique_quotient_is_9742_l186_18605


namespace arithmetic_sequence_a20_l186_18684

theorem arithmetic_sequence_a20 :
  (∀ n : ℕ, n > 0 → ∃ a : ℕ → ℕ, a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n + 2)) → 
  (∃ a : ℕ → ℕ, a 20 = 39) :=
by
  sorry

end arithmetic_sequence_a20_l186_18684


namespace sum_of_areas_of_tangent_circles_l186_18649

theorem sum_of_areas_of_tangent_circles :
  ∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧
    (r + s = 3) ∧
    (r + t = 4) ∧
    (s + t = 5) ∧
    π * (r^2 + s^2 + t^2) = 14 * π :=
by
  sorry

end sum_of_areas_of_tangent_circles_l186_18649


namespace circle_tangent_to_y_axis_l186_18697

/-- The relationship between the circle with the focal radius |PF| of the parabola y^2 = 2px (where p > 0)
as its diameter and the y-axis -/
theorem circle_tangent_to_y_axis
  (p : ℝ) (hp : p > 0)
  (x1 y1 : ℝ)
  (focus : ℝ × ℝ := (p / 2, 0))
  (P : ℝ × ℝ := (x1, y1))
  (center : ℝ × ℝ := ((2 * x1 + p) / 4, y1 / 2))
  (radius : ℝ := (2 * x1 + p) / 4) :
  -- proof that the circle with PF as its diameter is tangent to the y-axis
  ∃ k : ℝ, k = radius ∧ (center.1 = k) :=
sorry

end circle_tangent_to_y_axis_l186_18697


namespace probability_of_AB_not_selected_l186_18694

-- The definition for the probability of not selecting both A and B 
def probability_not_selected : ℚ :=
  let total_ways := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial (4 - 2))
  let favorable_ways := 1 -- Only the selection of C and D
  favorable_ways / total_ways

-- The theorem stating the desired probability
theorem probability_of_AB_not_selected : probability_not_selected = 1 / 6 :=
by
  sorry

end probability_of_AB_not_selected_l186_18694


namespace correct_result_l186_18619

theorem correct_result (x : ℝ) (h : x / 6 = 52) : x + 40 = 352 := by
  sorry

end correct_result_l186_18619


namespace penelope_saving_days_l186_18631

theorem penelope_saving_days :
  ∀ (daily_savings total_saved : ℕ),
  daily_savings = 24 ∧ total_saved = 8760 →
    total_saved / daily_savings = 365 :=
by
  rintro _ _ ⟨rfl, rfl⟩
  sorry

end penelope_saving_days_l186_18631


namespace angle_measure_l186_18646

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l186_18646


namespace probability_first_green_then_blue_l186_18632

variable {α : Type} [Fintype α]

noncomputable def prob_first_green_second_blue : ℚ := 
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := (green_marbles : ℚ) / total_marbles
  let prob_second_blue := (blue_marbles : ℚ) / (total_marbles - 1)
  (prob_first_green * prob_second_blue)

theorem probability_first_green_then_blue :
  prob_first_green_second_blue = 4 / 15 := by
  sorry

end probability_first_green_then_blue_l186_18632


namespace initial_temperature_is_20_l186_18645

-- Define the initial temperature, final temperature, rate of increase and time
def T_initial (T_final : ℕ) (rate_of_increase : ℕ) (time : ℕ) : ℕ :=
  T_final - rate_of_increase * time

-- Statement: The initial temperature is 20 degrees given the specified conditions.
theorem initial_temperature_is_20 :
  T_initial 100 5 16 = 20 :=
by
  sorry

end initial_temperature_is_20_l186_18645


namespace problem_a4_inv_a4_eq_seven_l186_18601

theorem problem_a4_inv_a4_eq_seven (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + (1/a)^4 = 7 :=
sorry

end problem_a4_inv_a4_eq_seven_l186_18601


namespace clock_angle_34030_l186_18608

noncomputable def calculate_angle (h m s : ℕ) : ℚ :=
  abs ((60 * h - 11 * (m + s / 60)) / 2)

theorem clock_angle_34030 : calculate_angle 3 40 30 = 130 :=
by
  sorry

end clock_angle_34030_l186_18608


namespace find_other_number_l186_18623

theorem find_other_number (x : ℕ) (h1 : 10 + x = 30) : x = 20 := by
  sorry

end find_other_number_l186_18623


namespace volume_of_pyramid_l186_18693

noncomputable def pyramid_volume : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 0)
  let C : ℝ × ℝ := (12, 20)
  let D : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) -- Midpoint of BC
  let E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) -- Midpoint of AC
  let F : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) -- Midpoint of AB
  let height : ℝ := 8.42 -- Vertically above the orthocenter
  let base_area : ℝ := 110 -- Area of the midpoint triangle
  (1 / 3) * base_area * height

theorem volume_of_pyramid : pyramid_volume = 309.07 :=
  by
    sorry

end volume_of_pyramid_l186_18693


namespace track_length_l186_18630

theorem track_length (h₁ : ∀ (x : ℕ), (exists y₁ y₂ : ℕ, y₁ = 120 ∧ y₂ = 180 ∧ y₁ + y₂ = x ∧ (y₂ - y₁ = 60) ∧ (y₂ = x - 120))) : 
  ∃ x : ℕ, x = 600 := by
  sorry

end track_length_l186_18630


namespace range_of_n_l186_18621

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.exp 1) * Real.exp x + (1 / 2) * x^2 - x

theorem range_of_n :
  (∃ m : ℝ, f m ≤ 2 * n^2 - n) ↔ (n ≤ -1/2 ∨ 1 ≤ n) :=
sorry

end range_of_n_l186_18621


namespace sum_of_c_n_l186_18610

variable {a_n : ℕ → ℕ}    -- Sequence {a_n}
variable {b_n : ℕ → ℕ}    -- Sequence {b_n}
variable {c_n : ℕ → ℕ}    -- Sequence {c_n}
variable {S_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {a_n}
variable {T_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {c_n}

axiom a3 : a_n 3 = 7
axiom S6 : S_n 6 = 48
axiom b_recur : ∀ n : ℕ, 2 * b_n (n + 1) = b_n n + 2
axiom b1 : b_n 1 = 3
axiom c_def : ∀ n : ℕ, c_n n = a_n n * (b_n n - 2)

theorem sum_of_c_n : ∀ n : ℕ, T_n n = 10 - (2*n + 5) * (1 / (2^(n-1))) :=
by
  -- Proof omitted
  sorry

end sum_of_c_n_l186_18610


namespace unfenced_side_length_l186_18654

-- Define the conditions
variables (L W : ℝ)
axiom area_condition : L * W = 480
axiom fence_condition : 2 * W + L = 64

-- Prove the unfenced side of the yard (L) is 40 feet
theorem unfenced_side_length : L = 40 :=
by
  -- Conditions, definitions, and properties go here.
  -- But we leave the proof as a placeholder since the statement is sufficient.
  sorry

end unfenced_side_length_l186_18654


namespace term_transition_addition_l186_18617

theorem term_transition_addition (k : Nat) :
  (2:ℚ) / ((k + 1) * (k + 2)) = ((2:ℚ) / ((k * (k + 1))) - ((2:ℚ) / ((k + 1) * (k + 2)))) := 
sorry

end term_transition_addition_l186_18617


namespace drum_filled_capacity_l186_18653

theorem drum_filled_capacity (C : ℝ) (h1 : 0 < C) :
    (4 / 5) * C + (1 / 2) * C = (13 / 10) * C :=
by
  sorry

end drum_filled_capacity_l186_18653


namespace bridget_poster_board_side_length_l186_18637

theorem bridget_poster_board_side_length
  (num_cards : ℕ)
  (card_length : ℕ)
  (card_width : ℕ)
  (posterboard_area : ℕ)
  (posterboard_side_length_feet : ℕ)
  (posterboard_side_length_inches : ℕ)
  (cards_area : ℕ) :
  num_cards = 24 ∧
  card_length = 2 ∧
  card_width = 3 ∧
  posterboard_area = posterboard_side_length_inches ^ 2 ∧
  cards_area = num_cards * (card_length * card_width) ∧
  cards_area = posterboard_area ∧
  posterboard_side_length_inches = 12 ∧
  posterboard_side_length_feet = posterboard_side_length_inches / 12 →
  posterboard_side_length_feet = 1 :=
sorry

end bridget_poster_board_side_length_l186_18637


namespace sam_paid_amount_l186_18611

theorem sam_paid_amount (F : ℝ) (Joe Peter Sam : ℝ) 
  (h1 : Joe = (1/4)*F + 7) 
  (h2 : Peter = (1/3)*F - 7) 
  (h3 : Sam = (1/2)*F - 12)
  (h4 : Joe + Peter + Sam = F) : 
  Sam = 60 := 
by 
  sorry

end sam_paid_amount_l186_18611


namespace problem1_correct_solution_problem2_correct_solution_l186_18618

noncomputable def g (x a : ℝ) : ℝ := |x| + 2 * |x + 2 - a|

/-- 
    Prove that the set {x | -2/3 ≤ x ≤ 2} satisfies g(x) ≤ 4 when a = 3 
--/
theorem problem1_correct_solution (x : ℝ) : g x 3 ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 :=
by
  sorry

noncomputable def f (x a : ℝ) : ℝ := g (x - 2) a

/-- 
    Prove that the range of a such that f(x) ≥ 1 for all x ∈ ℝ 
    is a ≤ 1 or a ≥ 3
--/
theorem problem2_correct_solution (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 :=
by
  sorry

end problem1_correct_solution_problem2_correct_solution_l186_18618


namespace find_number_l186_18620

theorem find_number (x n : ℤ) (h1 : 5 * x + n = 10 * x - 17) (h2 : x = 4) : n = 3 := by
  sorry

end find_number_l186_18620


namespace largest_sum_of_watch_digits_l186_18698

theorem largest_sum_of_watch_digits : ∃ s : ℕ, s = 23 ∧ 
  (∀ h m : ℕ, h < 24 → m < 60 → s ≤ (h / 10 + h % 10 + m / 10 + m % 10)) :=
by
  sorry

end largest_sum_of_watch_digits_l186_18698


namespace rectangle_side_lengths_l186_18663

theorem rectangle_side_lengths (x y : ℝ) (h1 : 2 * x + 4 = 10) (h2 : 8 * y - 2 = 10) : x + y = 4.5 := by
  sorry

end rectangle_side_lengths_l186_18663


namespace h_is_decreasing_intervals_l186_18633

noncomputable def f (x : ℝ) := if x >= 1 then x - 2 else 0
noncomputable def g (x : ℝ) := if x <= 2 then -2 * x + 3 else 0

noncomputable def h (x : ℝ) :=
  if x >= 1 ∧ x <= 2 then f x * g x
  else if x >= 1 then f x
  else if x <= 2 then g x
  else 0

theorem h_is_decreasing_intervals :
  (∀ x1 x2 : ℝ, x1 < x2 → x1 < 1 → h x1 > h x2) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → x1 ≥ 7 / 4 → x2 ≤ 2 → h x1 ≥ h x2) :=
by
  sorry

end h_is_decreasing_intervals_l186_18633


namespace greatest_possible_value_of_x_l186_18683

theorem greatest_possible_value_of_x (x : ℕ) (H : Nat.lcm (Nat.lcm x 12) 18 = 108) : x ≤ 108 := sorry

end greatest_possible_value_of_x_l186_18683


namespace find_prime_pairs_l186_18636

theorem find_prime_pairs :
  ∃ p q : ℕ, Prime p ∧ Prime q ∧
    ∃ a b : ℕ, a^2 = p - q ∧ b^2 = pq - q ∧ (p = 3 ∧ q = 2) :=
by
  sorry

end find_prime_pairs_l186_18636


namespace probability_final_marble_red_l186_18689

theorem probability_final_marble_red :
  let P_wA := 5/8
  let P_bA := 3/8
  let P_gB_p1 := 6/10
  let P_rB_p2 := 4/10
  let P_gC_p1 := 4/7
  let P_rC_p2 := 3/7
  let P_wr_b_g := P_wA * P_gB_p1 * P_rB_p2
  let P_blk_g_red := P_bA * P_gC_p1 * P_rC_p2
  (P_wr_b_g + P_blk_g_red) = 79/980 :=
by {
  let P_wA := 5/8
  let P_bA := 3/8
  let P_gB_p1 := 6/10
  let P_rB_p2 := 4/10
  let P_gC_p1 := 4/7
  let P_rC_p2 := 3/7
  let P_wr_b_g := P_wA * P_gB_p1 * P_rB_p2
  let P_blk_g_red := P_bA * P_gC_p1 * P_rC_p2
  show (P_wr_b_g + P_blk_g_red) = 79/980
  sorry
}

end probability_final_marble_red_l186_18689


namespace common_difference_unique_l186_18651

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1 : ℝ, ∀ n : ℕ, a n = a1 + n * d

theorem common_difference_unique {a : ℕ → ℝ}
  (h1 : a 2 = 5)
  (h2 : a 3 + a 5 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) ∧ d = -2 :=
sorry

end common_difference_unique_l186_18651


namespace acute_triangle_angle_measure_acute_triangle_side_range_l186_18627

theorem acute_triangle_angle_measure (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) : B = π / 3 :=
by
  sorry

theorem acute_triangle_side_range (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) (hB : B = π / 3) (hb : b = 3) :
  3 * Real.sqrt 3 < a + c ∧ a + c ≤ 6 :=
by
  sorry

end acute_triangle_angle_measure_acute_triangle_side_range_l186_18627


namespace faster_train_length_l186_18675

theorem faster_train_length
  (speed_faster : ℝ)
  (speed_slower : ℝ)
  (time_to_cross : ℝ)
  (relative_speed_limit: ℝ)
  (h1 : speed_faster = 108 * 1000 / 3600)
  (h2: speed_slower = 36 * 1000 / 3600)
  (h3: time_to_cross = 17)
  (h4: relative_speed_limit = 2) :
  (speed_faster - speed_slower) * time_to_cross = 340 := 
sorry

end faster_train_length_l186_18675


namespace div_add_fraction_l186_18690

theorem div_add_fraction : (3 / 7) / 4 + 2 = 59 / 28 :=
by
  sorry

end div_add_fraction_l186_18690


namespace magician_ball_count_l186_18644

theorem magician_ball_count (k : ℕ) : ∃ k : ℕ, 6 * k + 7 = 1993 :=
by sorry

end magician_ball_count_l186_18644


namespace length_of_CD_l186_18688

theorem length_of_CD {L : ℝ} (h₁ : 16 * Real.pi * L + (256 / 3) * Real.pi = 432 * Real.pi) :
  L = (50 / 3) :=
by
  sorry

end length_of_CD_l186_18688


namespace length_sixth_episode_l186_18687

def length_first_episode : ℕ := 58
def length_second_episode : ℕ := 62
def length_third_episode : ℕ := 65
def length_fourth_episode : ℕ := 71
def length_fifth_episode : ℕ := 79
def total_viewing_time : ℕ := 450

theorem length_sixth_episode :
  length_first_episode + length_second_episode + length_third_episode + length_fourth_episode + length_fifth_episode + 115 = total_viewing_time := by
  sorry

end length_sixth_episode_l186_18687


namespace polygon_area_correct_l186_18673

-- Define the coordinates of the vertices
def vertex1 := (2, 1)
def vertex2 := (4, 3)
def vertex3 := (6, 1)
def vertex4 := (4, 6)

-- Define a function to calculate the area using the Shoelace Theorem
noncomputable def shoelace_area (vertices : List (ℕ × ℕ)) : ℚ :=
  let xys := vertices ++ [vertices.head!]
  let sum1 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => x1 * y2)
  let sum2 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => y1 * x2)
  (sum1.sum - sum2.sum : ℚ) / 2

-- Instantiate the specific vertices
def polygon := [vertex1, vertex2, vertex3, vertex4]

-- The theorem statement
theorem polygon_area_correct : shoelace_area polygon = 6 := by
  sorry

end polygon_area_correct_l186_18673


namespace city_of_archimedes_schools_l186_18607

noncomputable def numberOfSchools : ℕ := 32

theorem city_of_archimedes_schools :
  ∃ n : ℕ, (∀ s : Set ℕ, s = {45, 68, 113} →
  (∀ x ∈ s, x > 1 → 4 * n = x + 1 → (2 * n ≤ x ∧ 2 * n + 1 ≥ x) ))
  ∧ n = numberOfSchools :=
sorry

end city_of_archimedes_schools_l186_18607


namespace part_1_part_2_l186_18613

theorem part_1 (a : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) (n : ℕ) (hn_pos : 0 < n) : 
  a (n + 1) - 2 * a n = 0 :=
sorry

theorem part_2 (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) :
  (∀ n, b n = 1 / (a n * a (n + 1))) → ∀ n, S n = (1/6) * (1 - (1/4)^n) :=
sorry

end part_1_part_2_l186_18613


namespace rabbits_in_cage_l186_18676

theorem rabbits_in_cage (heads legs : ℝ) (total_heads : heads = 40) 
  (condition : legs = 8 + 10 * (2 * (heads - rabbits))) :
  ∃ rabbits : ℝ, rabbits = 33 :=
by
  sorry

end rabbits_in_cage_l186_18676


namespace power_addition_l186_18658

variable {R : Type*} [CommRing R]

theorem power_addition (x : R) (m n : ℕ) (h1 : x^m = 6) (h2 : x^n = 2) : x^(m + n) = 12 :=
by
  sorry

end power_addition_l186_18658


namespace nelly_refrigerator_payment_l186_18661

theorem nelly_refrigerator_payment (T : ℝ) (p1 p2 p3 : ℝ) (p1_percent p2_percent p3_percent : ℝ)
  (h1 : p1 = 875) (h2 : p2 = 650) (h3 : p3 = 1200)
  (h4 : p1_percent = 0.25) (h5 : p2_percent = 0.15) (h6 : p3_percent = 0.35)
  (total_paid := p1 + p2 + p3)
  (percent_paid := p1_percent + p2_percent + p3_percent)
  (total_cost := total_paid / percent_paid)
  (remaining := total_cost - total_paid) :
  remaining = 908.33 := by
  sorry

end nelly_refrigerator_payment_l186_18661


namespace function_range_l186_18603

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x^2 - 1) * (x^2 + a * x + b)

theorem function_range (a b : ℝ) (h_symm : ∀ x : ℝ, f (6 - x) a b = f x a b) :
  a = -12 ∧ b = 35 ∧ (∀ y, ∃ x : ℝ, f x (-12) 35 = y ↔ -36 ≤ y) :=
by
  sorry

end function_range_l186_18603


namespace find_number_l186_18685

theorem find_number (x : ℝ) (h : 3034 - x / 20.04 = 2984) : x = 1002 :=
sorry

end find_number_l186_18685


namespace max_mogs_l186_18622

theorem max_mogs : ∃ x y z : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 3 * x + 4 * y + 8 * z = 100 ∧ z = 10 :=
by
  sorry

end max_mogs_l186_18622


namespace geometric_sequence_third_term_l186_18656

theorem geometric_sequence_third_term (r : ℕ) (a : ℕ) (h1 : a = 6) (h2 : a * r^3 = 384) : a * r^2 = 96 :=
by
  sorry

end geometric_sequence_third_term_l186_18656


namespace product_586645_9999_l186_18674

theorem product_586645_9999 :
  586645 * 9999 = 5865885355 :=
by
  sorry

end product_586645_9999_l186_18674


namespace trigonometric_identity_l186_18602

open Real

variable (α : ℝ)

theorem trigonometric_identity (h : tan (π - α) = 2) :
  (sin (π / 2 + α) + sin (π - α)) / (cos (3 * π / 2 + α) + 2 * cos (π + α)) = 1 / 4 :=
  sorry

end trigonometric_identity_l186_18602


namespace number_of_multiples_143_l186_18672

theorem number_of_multiples_143
  (h1 : 143 = 11 * 13)
  (h2 : ∀ i j : ℕ, 10^j - 10^i = 10^i * (10^(j-i) - 1))
  (h3 : ∀ i : ℕ, gcd (10^i) 143 = 1)
  (h4 : ∀ k : ℕ, 143 ∣ 10^k - 1 ↔ k % 6 = 0)
  (h5 : ∀ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ 99)
  : ∃ n : ℕ, n = 784 :=
by
  sorry

end number_of_multiples_143_l186_18672


namespace line_through_points_has_sum_m_b_3_l186_18642

-- Define the structure that two points are given
structure LineThroughPoints (P1 P2 : ℝ × ℝ) : Prop :=
  (slope_intercept_form : ∃ m b, (P1.snd = m * P1.fst + b) ∧ (P2.snd = b)) 

-- Define the particular points
def point1 : ℝ × ℝ := (-2, 0)
def point2 : ℝ × ℝ := (0, 2)

-- The theorem statement
theorem line_through_points_has_sum_m_b_3 
  (h : LineThroughPoints point1 point2) : 
  ∃ m b, (point1.snd = m * point1.fst + b) ∧ (point2.snd = b) ∧ (m + b = 3) :=
by
  sorry

end line_through_points_has_sum_m_b_3_l186_18642


namespace octagon_side_length_l186_18662

theorem octagon_side_length 
  (num_sides : ℕ) 
  (perimeter : ℝ) 
  (h_sides : num_sides = 8) 
  (h_perimeter : perimeter = 23.6) :
  (perimeter / num_sides) = 2.95 :=
by
  have h_valid_sides : num_sides = 8 := h_sides
  have h_valid_perimeter : perimeter = 23.6 := h_perimeter
  sorry

end octagon_side_length_l186_18662


namespace regular_hexagon_area_inscribed_in_circle_l186_18634

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end regular_hexagon_area_inscribed_in_circle_l186_18634


namespace algebraic_expression_value_l186_18657

theorem algebraic_expression_value {x : ℝ} (h : x * (x + 2) = 2023) : 2 * (x + 3) * (x - 1) - 2018 = 2022 := 
by 
  sorry

end algebraic_expression_value_l186_18657


namespace stripe_area_is_640pi_l186_18666

noncomputable def cylinder_stripe_area (diameter height stripe_width : ℝ) (revolutions : ℕ) : ℝ :=
  let circumference := Real.pi * diameter
  let length := circumference * (revolutions : ℝ)
  stripe_width * length

theorem stripe_area_is_640pi :
  cylinder_stripe_area 20 100 4 4 = 640 * Real.pi :=
by 
  sorry

end stripe_area_is_640pi_l186_18666


namespace volume_of_solid_of_revolution_l186_18695

theorem volume_of_solid_of_revolution (a : ℝ) : 
  let h := a / 2
  let r := (Real.sqrt 3 / 2) * a
  2 * (1 / 3) * π * r^2 * h = (π * a^3) / 4 :=
by
  sorry

end volume_of_solid_of_revolution_l186_18695


namespace three_x_y_z_l186_18664

variable (x y z : ℝ)

def equation1 : Prop := y + z = 17 - 2 * x
def equation2 : Prop := x + z = -11 - 2 * y
def equation3 : Prop := x + y = 9 - 2 * z

theorem three_x_y_z : equation1 x y z ∧ equation2 x y z ∧ equation3 x y z → 3 * x + 3 * y + 3 * z = 45 / 4 :=
by
  intros h
  sorry

end three_x_y_z_l186_18664
