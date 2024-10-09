import Mathlib

namespace range_of_k_l686_68620

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 2 then 2 / x else (x - 1)^3

theorem range_of_k (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 - k = 0 ∧ f x2 - k = 0) ↔ (0 < k ∧ k < 1) := sorry

end range_of_k_l686_68620


namespace inequality_solution_l686_68636

theorem inequality_solution (a : ℝ) (h : a^2 > 2 * a - 1) : a ≠ 1 := 
sorry

end inequality_solution_l686_68636


namespace jogging_time_after_two_weeks_l686_68676

noncomputable def daily_jogging_hours : ℝ := 1.5
noncomputable def days_in_two_weeks : ℕ := 14

theorem jogging_time_after_two_weeks : daily_jogging_hours * days_in_two_weeks = 21 := by
  sorry

end jogging_time_after_two_weeks_l686_68676


namespace sin_240_eq_neg_sqrt3_div_2_l686_68606

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l686_68606


namespace annual_income_of_A_l686_68654

variable (Cm : ℝ)
variable (Bm : ℝ)
variable (Am : ℝ)
variable (Aa : ℝ)

-- Given conditions
axiom h1 : Cm = 12000
axiom h2 : Bm = Cm + 0.12 * Cm
axiom h3 : (Am / Bm) = 5 / 2

-- Statement to prove
theorem annual_income_of_A : Aa = 403200 := by
  sorry

end annual_income_of_A_l686_68654


namespace triangle_height_l686_68609

theorem triangle_height (s h : ℝ) 
  (area_square : s^2 = s * s) 
  (area_triangle : 1/2 * s * h = s^2) 
  (areas_equal : s^2 = s^2) : 
  h = 2 * s := 
sorry

end triangle_height_l686_68609


namespace intersection_eq_l686_68663

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x ≤ 2}

theorem intersection_eq : P ∩ Q = {1, 2} :=
by
  sorry

end intersection_eq_l686_68663


namespace geometric_sequence_formula_l686_68688

noncomputable def a_n (q : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else 2^(n - 1)

theorem geometric_sequence_formula (q : ℝ) (S : ℕ → ℝ) (n : ℕ) (hn : n > 0) :
  a_n q n = 2^(n - 1) :=
sorry

end geometric_sequence_formula_l686_68688


namespace find_n_tan_eq_348_l686_68660

theorem find_n_tan_eq_348 (n : ℤ) (h1 : -90 < n) (h2 : n < 90) : 
  (Real.tan (n * Real.pi / 180) = Real.tan (348 * Real.pi / 180)) ↔ (n = -12) := by
  sorry

end find_n_tan_eq_348_l686_68660


namespace find_c_value_l686_68686

theorem find_c_value (c : ℝ)
  (h : 4 * (3.6 * 0.48 * c / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) :
  c = 2.5 :=
by sorry

end find_c_value_l686_68686


namespace orthogonal_projection_l686_68664

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_squared := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_squared * u.1, dot_uv / norm_u_squared * u.2)

theorem orthogonal_projection
  (a b : ℝ × ℝ)
  (h_orth : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj a (4, -4) = (-4/5, -8/5)) :
  proj b (4, -4) = (24/5, -12/5) :=
sorry

end orthogonal_projection_l686_68664


namespace third_side_range_l686_68644

theorem third_side_range (a : ℝ) (h₃ : 0 < a ∧ a ≠ 0) (h₅ : 0 < a ∧ a ≠ 0): 
  (2 < a ∧ a < 8) ↔ (3 - 5 < a ∧ a < 3 + 5) :=
by
  sorry

end third_side_range_l686_68644


namespace no_valid_positive_x_l686_68615

theorem no_valid_positive_x
  (π : Real)
  (R H x : Real)
  (hR : R = 5)
  (hH : H = 10)
  (hx_pos : x > 0) :
  ¬π * (R + x) ^ 2 * H = π * R ^ 2 * (H + x) :=
by
  sorry

end no_valid_positive_x_l686_68615


namespace quadrilateral_area_l686_68626

theorem quadrilateral_area (EF FG EH HG : ℕ) (hEFH : EF * EF + FG * FG = 25)
(hEHG : EH * EH + HG * HG = 25) (h_distinct : EF ≠ EH ∧ FG ≠ HG) 
(h_greater_one : EF > 1 ∧ FG > 1 ∧ EH > 1 ∧ HG > 1) :
  (EF * FG) / 2 + (EH * HG) / 2 = 12 := 
sorry

end quadrilateral_area_l686_68626


namespace sqrt_meaningful_range_l686_68621

theorem sqrt_meaningful_range {x : ℝ} (h : x - 1 ≥ 0) : x ≥ 1 :=
sorry

end sqrt_meaningful_range_l686_68621


namespace taxes_are_135_l686_68629

def gross_pay : ℕ := 450
def net_pay : ℕ := 315
def taxes_paid (G N: ℕ) : ℕ := G - N

theorem taxes_are_135 : taxes_paid gross_pay net_pay = 135 := by
  sorry

end taxes_are_135_l686_68629


namespace determine_chris_age_l686_68648

theorem determine_chris_age (a b c : ℚ)
  (h1 : (a + b + c) / 3 = 10)
  (h2 : c - 5 = 2 * a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 283 / 15 :=
by
  sorry

end determine_chris_age_l686_68648


namespace girl_boy_lineup_probability_l686_68601

theorem girl_boy_lineup_probability :
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  (valid_configurations : ℚ) / total_configurations = 0.058 :=
by
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  have h : (valid_configurations : ℚ) / total_configurations = 0.058 := sorry
  exact h

end girl_boy_lineup_probability_l686_68601


namespace minimum_a_l686_68643

theorem minimum_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → 
  a ≥ -5/2 :=
sorry

end minimum_a_l686_68643


namespace find_a3_in_arith_geo_seq_l686_68697

theorem find_a3_in_arith_geo_seq
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : S 6 / S 3 = -19 / 8)
  (h2 : a 4 - a 2 = -15 / 8) :
  a 3 = 9 / 4 :=
sorry

end find_a3_in_arith_geo_seq_l686_68697


namespace no_solution_fraction_equation_l686_68655

theorem no_solution_fraction_equation (x : ℝ) (h : x ≠ 2) : 
  (1 - x) / (x - 2) + 2 = 1 / (2 - x) → false :=
by 
  intro h_eq
  sorry

end no_solution_fraction_equation_l686_68655


namespace root_range_of_f_eq_zero_solution_set_of_f_le_zero_l686_68625

variable (m : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := m * x^2 + (2 * m + 1) * x + 2

theorem root_range_of_f_eq_zero (h : ∃ r1 r2 : ℝ, r1 > 1 ∧ r2 < 1 ∧ f r1 = 0 ∧ f r2 = 0) : -1 < m ∧ m < 0 :=
sorry

theorem solution_set_of_f_le_zero : 
  (m = 0 -> ∀ x, f x ≤ 0 ↔ x ≤ - 2) ∧
  (m < 0 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) ∧
  (0 < m ∧ m < 1/2 -> ∀ x, f x ≤ 0 ↔ - (1/m) ≤ x ∧ x ≤ - 2) ∧
  (m = 1/2 -> ∀ x, f x ≤ 0 ↔ x = - 2) ∧
  (m > 1/2 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) :=
sorry

end root_range_of_f_eq_zero_solution_set_of_f_le_zero_l686_68625


namespace product_of_even_and_odd_is_odd_l686_68672

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x
def odd_product (f g : ℝ → ℝ) : Prop := ∀ x, (f x) * (g x) = - (f x) * (g x)
 
theorem product_of_even_and_odd_is_odd 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : odd_function g) : odd_product f g :=
by
  sorry

end product_of_even_and_odd_is_odd_l686_68672


namespace number_of_m_values_l686_68604

theorem number_of_m_values (m : ℕ) (h1 : 4 * m > 11) (h2 : m < 12) : 
  11 - 3 + 1 = 9 := 
sorry

end number_of_m_values_l686_68604


namespace distance_from_pole_l686_68617

-- Define the structure for polar coordinates.
structure PolarCoordinates where
  r : ℝ
  θ : ℝ

-- Define point A with its polar coordinates.
def A : PolarCoordinates := { r := 3, θ := -4 }

-- State the problem to prove that the distance |OA| is 3.
theorem distance_from_pole (A : PolarCoordinates) : A.r = 3 :=
by {
  sorry
}

end distance_from_pole_l686_68617


namespace average_age_of_team_l686_68670

theorem average_age_of_team (A : ℝ) : 
    (11 * A =
         9 * (A - 1) + 53) → 
    A = 31 := 
by 
  sorry

end average_age_of_team_l686_68670


namespace inequality_proof_l686_68666

theorem inequality_proof (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = x^2 + 3 * x + 2) →
  a > 0 →
  b > 0 →
  b ≤ a / 7 →
  (∀ x : ℝ, |x + 2| < b → |f x + 4| < a) :=
by
  sorry

end inequality_proof_l686_68666


namespace breadth_is_13_l686_68680

variable (b l : ℕ) (breadth : ℕ)

/-
We have the following conditions:
1. The area of the rectangular plot is 23 times its breadth.
2. The difference between the length and the breadth is 10 metres.
We need to prove that the breadth of the plot is 13 metres.
-/

theorem breadth_is_13
  (h1 : l * b = 23 * b)
  (h2 : l - b = 10) :
  b = 13 := 
sorry

end breadth_is_13_l686_68680


namespace stockholm_uppsala_distance_l686_68614

variable (map_distance : ℝ) (scale_factor : ℝ)

def actual_distance (d : ℝ) (s : ℝ) : ℝ := d * s

theorem stockholm_uppsala_distance :
  actual_distance 65 20 = 1300 := by
  sorry

end stockholm_uppsala_distance_l686_68614


namespace young_or_old_woman_lawyer_probability_l686_68669

/-- 
40 percent of the members of a study group are women.
Among these women, 30 percent are young lawyers.
10 percent are old lawyers.
Prove the probability that a member randomly selected is a young or old woman lawyer is 0.16.
-/
theorem young_or_old_woman_lawyer_probability :
  let total_members := 100
  let women_percentage := 40
  let young_lawyers_percentage := 30
  let old_lawyers_percentage := 10
  let total_women := (women_percentage * total_members) / 100
  let young_women_lawyers := (young_lawyers_percentage * total_women) / 100
  let old_women_lawyers := (old_lawyers_percentage * total_women) / 100
  let women_lawyers := young_women_lawyers + old_women_lawyers
  let probability := women_lawyers / total_members
  probability = 0.16 := 
by {
  sorry
}

end young_or_old_woman_lawyer_probability_l686_68669


namespace div_trans_l686_68642

variable {a b c : ℝ}

theorem div_trans :
  a / b = 3 → b / c = 5 / 2 → c / a = 2 / 15 :=
  by
  intro h1 h2
  sorry

end div_trans_l686_68642


namespace total_people_waiting_l686_68633

theorem total_people_waiting 
  (initial_first_line : ℕ := 7)
  (left_first_line : ℕ := 4)
  (joined_first_line : ℕ := 8)
  (initial_second_line : ℕ := 12)
  (left_second_line : ℕ := 3)
  (joined_second_line : ℕ := 10)
  (initial_third_line : ℕ := 15)
  (left_third_line : ℕ := 5)
  (joined_third_line : ℕ := 7) :
  (initial_first_line - left_first_line + joined_first_line) +
  (initial_second_line - left_second_line + joined_second_line) +
  (initial_third_line - left_third_line + joined_third_line) = 47 :=
by
  sorry

end total_people_waiting_l686_68633


namespace quotient_correct_l686_68673

def dividend : ℤ := 474232
def divisor : ℤ := 800
def remainder : ℤ := -968

theorem quotient_correct : (dividend + abs remainder) / divisor = 594 := by
  sorry

end quotient_correct_l686_68673


namespace complex_division_l686_68605

noncomputable def imagine_unit : ℂ := Complex.I

theorem complex_division :
  (Complex.mk (-3) 1) / (Complex.mk 1 (-1)) = (Complex.mk (-2) 1) :=
by
sorry

end complex_division_l686_68605


namespace regular_polygon_sides_l686_68682

theorem regular_polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : 
  n = 6 :=
sorry

end regular_polygon_sides_l686_68682


namespace sequence_value_x_l686_68681

theorem sequence_value_x (x : ℕ) (h1 : 1 + 3 = 4) (h2 : 4 + 3 = 7) (h3 : 7 + 3 = 10) (h4 : 10 + 3 = x) (h5 : x + 3 = 16) : x = 13 := by
  sorry

end sequence_value_x_l686_68681


namespace max_choir_members_l686_68622

theorem max_choir_members : 
  ∃ (m : ℕ), 
    (∃ k : ℕ, m = k^2 + 11) ∧ 
    (∃ n : ℕ, m = n * (n + 5)) ∧ 
    (∀ m' : ℕ, 
      ((∃ k' : ℕ, m' = k' * k' + 11) ∧ 
       (∃ n' : ℕ, m' = n' * (n' + 5))) → 
      m' ≤ 266) ∧ 
    m = 266 :=
by sorry

end max_choir_members_l686_68622


namespace decrease_hours_by_13_percent_l686_68656

theorem decrease_hours_by_13_percent (W H : ℝ) (hW_pos : W > 0) (hH_pos : H > 0) :
  let W_new := 1.15 * W
  let H_new := H / 1.15
  let income_decrease_percentage := (1 - H_new / H) * 100
  abs (income_decrease_percentage - 13.04) < 0.01 := 
by
  sorry

end decrease_hours_by_13_percent_l686_68656


namespace arccos_range_l686_68657

theorem arccos_range (a : ℝ) (x : ℝ) (h₀ : x = Real.sin a) 
  (h₁ : -Real.pi / 4 ≤ a ∧ a ≤ 3 * Real.pi / 4) :
  ∀ y, y = Real.arccos x → 0 ≤ y ∧ y ≤ 3 * Real.pi / 4 := 
sorry

end arccos_range_l686_68657


namespace substract_repeating_decimal_l686_68645

noncomputable def repeating_decimal : ℝ := 1 / 3

theorem substract_repeating_decimal (x : ℝ) (h : x = repeating_decimal) : 
  1 - x = 2 / 3 :=
by
  sorry

end substract_repeating_decimal_l686_68645


namespace min_distance_racetracks_l686_68661

theorem min_distance_racetracks : 
  ∀ A B : ℝ × ℝ, (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (((B.1 - 1) ^ 2) / 16 + (B.2 ^ 2) / 4 = 1) → 
  dist A B ≥ (Real.sqrt 33 - 3) / 3 := by
  sorry

end min_distance_racetracks_l686_68661


namespace remainder_of_sum_modulo_9_l686_68684

theorem remainder_of_sum_modulo_9 : 
  (8230 + 8231 + 8232 + 8233 + 8234 + 8235) % 9 = 0 := by
  sorry

end remainder_of_sum_modulo_9_l686_68684


namespace hex_product_l686_68607

def hex_to_dec (h : Char) : Nat :=
  match h with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | c   => c.toNat - '0'.toNat

noncomputable def dec_to_hex (n : Nat) : String :=
  let q := n / 16
  let r := n % 16
  let r_hex := if r < 10 then Char.ofNat (r + '0'.toNat) else Char.ofNat (r - 10 + 'A'.toNat)
  (if q > 0 then toString q else "") ++ Char.toString r_hex

theorem hex_product :
  dec_to_hex (hex_to_dec 'A' * hex_to_dec 'B') = "6E" :=
by
  sorry

end hex_product_l686_68607


namespace trivia_team_members_l686_68628

theorem trivia_team_members (x : ℕ) (h : 3 * (x - 6) = 27) : x = 15 := 
by
  sorry

end trivia_team_members_l686_68628


namespace sum_seven_consecutive_integers_l686_68668

theorem sum_seven_consecutive_integers (n : ℕ) : 
  ∃ k : ℕ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) = 7 * k := 
by 
  -- Use sum of integers and factor to demonstrate that the sum is multiple of 7
  sorry

end sum_seven_consecutive_integers_l686_68668


namespace find_n_for_arithmetic_sequence_l686_68616

variable {a : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (a₁ : ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + n * d

theorem find_n_for_arithmetic_sequence (h_arith : is_arithmetic_sequence a (-1) 2)
  (h_nth_term : ∃ n : ℕ, a n = 15) : ∃ n : ℕ, n = 9 :=
by
  sorry

end find_n_for_arithmetic_sequence_l686_68616


namespace k_at_1_value_l686_68658

def h (x p : ℝ) := x^3 + p * x^2 + 2 * x + 20
def k (x p q r : ℝ) := x^4 + 2 * x^3 + q * x^2 + 50 * x + r

theorem k_at_1_value (p q r : ℝ) (h_distinct_roots : ∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ → h x₁ p = 0 → h x₂ p = 0 → h x₃ p = 0 → k x₁ p q r = 0 ∧ k x₂ p q r = 0 ∧ k x₃ p q r = 0):
  k 1 (-28) (2 - -28 * -30) (-20 * -30) = -155 :=
by
  sorry

end k_at_1_value_l686_68658


namespace reciprocals_expression_eq_zero_l686_68677

theorem reciprocals_expression_eq_zero {m n : ℝ} (h : m * n = 1) : (2 * m - 2 / n) * (1 / m + n) = 0 :=
by
  sorry

end reciprocals_expression_eq_zero_l686_68677


namespace find_divisor_l686_68679

theorem find_divisor (d : ℕ) (h1 : 109 % d = 1) (h2 : 109 / d = 9) : d = 12 := by
  sorry

end find_divisor_l686_68679


namespace fifth_inequality_l686_68696

theorem fifth_inequality :
  1 + (1 / (2^2 : ℝ)) + (1 / (3^2 : ℝ)) + (1 / (4^2 : ℝ)) + (1 / (5^2 : ℝ)) + (1 / (6^2 : ℝ)) < (11 / 6 : ℝ) :=
by
  sorry

end fifth_inequality_l686_68696


namespace carol_points_loss_l686_68690

theorem carol_points_loss 
  (first_round_points : ℕ) (second_round_points : ℕ) (end_game_points : ℕ) 
  (h1 : first_round_points = 17) 
  (h2 : second_round_points = 6) 
  (h3 : end_game_points = 7) : 
  (first_round_points + second_round_points - end_game_points = 16) :=
by 
  sorry

end carol_points_loss_l686_68690


namespace sequence_nonzero_l686_68699

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  ∀ n ≥ 3, 
    if (a (n - 1) * a (n - 2)) % 2 = 0 then 
      a n = 5 * (a (n - 1)) - 3 * (a (n - 2)) 
    else 
      a n = (a (n - 1)) - (a (n - 2))

theorem sequence_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, a n ≠ 0 := 
by sorry

end sequence_nonzero_l686_68699


namespace ratio_of_wire_lengths_l686_68610

theorem ratio_of_wire_lengths (b_pieces : ℕ) (b_piece_length : ℕ)
  (c_piece_length : ℕ) (cubes_volume : ℕ) :
  b_pieces = 12 →
  b_piece_length = 8 →
  c_piece_length = 2 →
  cubes_volume = (b_piece_length ^ 3) →
  b_pieces * b_piece_length * cubes_volume
    / (cubes_volume * (12 * c_piece_length)) = (1 / 128) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_wire_lengths_l686_68610


namespace speed_of_car_first_hour_98_l686_68671

def car_speed_in_first_hour_is_98 (x : ℕ) : Prop :=
  (70 + x) / 2 = 84 → x = 98

theorem speed_of_car_first_hour_98 (x : ℕ) (h : car_speed_in_first_hour_is_98 x) : x = 98 :=
  by
  sorry

end speed_of_car_first_hour_98_l686_68671


namespace inequality_negative_solution_l686_68650

theorem inequality_negative_solution (a : ℝ) (h : a ≥ -17/4 ∧ a < 4) : 
  ∃ x : ℝ, x < 0 ∧ x^2 < 4 - |x - a| :=
by
  sorry

end inequality_negative_solution_l686_68650


namespace debby_photos_of_friends_l686_68631

theorem debby_photos_of_friends (F : ℕ) (h1 : 23 + F = 86) : F = 63 := by
  -- Proof steps will go here
  sorry

end debby_photos_of_friends_l686_68631


namespace find_m_l686_68639

variables (m : ℝ)
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, m)
def c : ℝ × ℝ := (-1, 2)

-- Define the property of vector parallelism in ℝ.
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Statement to be proven
theorem find_m :
    parallel (1, m - 1) c →
    m = -1 :=
by
  sorry

end find_m_l686_68639


namespace football_cost_l686_68624

-- Definitions derived from conditions
def marbles_cost : ℝ := 9.05
def baseball_cost : ℝ := 6.52
def total_spent : ℝ := 20.52

-- The statement to prove the cost of the football
theorem football_cost :
  ∃ (football_cost : ℝ), football_cost = total_spent - marbles_cost - baseball_cost :=
sorry

end football_cost_l686_68624


namespace triangle_perimeter_correct_l686_68674

noncomputable def triangle_perimeter (a b c : ℕ) : ℕ :=
    a + b + c

theorem triangle_perimeter_correct (a b c : ℕ) (h1 : a = b - 1) (h2 : b = c - 1) (h3 : c = 2 * a) : triangle_perimeter a b c = 15 :=
    sorry

end triangle_perimeter_correct_l686_68674


namespace remainder_91_pow_91_mod_100_l686_68602

theorem remainder_91_pow_91_mod_100 : Nat.mod (91 ^ 91) 100 = 91 :=
by
  sorry

end remainder_91_pow_91_mod_100_l686_68602


namespace initial_number_of_students_l686_68611

theorem initial_number_of_students (W : ℝ) (n : ℕ) (new_student_weight avg_weight1 avg_weight2 : ℝ)
  (h1 : avg_weight1 = 15)
  (h2 : new_student_weight = 13)
  (h3 : avg_weight2 = 14.9)
  (h4 : W = n * avg_weight1)
  (h5 : W + new_student_weight = (n + 1) * avg_weight2) : n = 19 := 
by
  sorry

end initial_number_of_students_l686_68611


namespace transform_polynomial_eq_correct_factorization_positive_polynomial_gt_zero_l686_68641

-- Define the polynomial transformation
def transform_polynomial (x : ℝ) : ℝ := x^2 + 8 * x - 1

-- Transformation problem
theorem transform_polynomial_eq (x m n : ℝ) :
  (x + 4)^2 - 17 = transform_polynomial x := 
sorry

-- Define the polynomial for correction
def factor_polynomial (x : ℝ) : ℝ := x^2 - 3 * x - 40

-- Factoring correction problem
theorem correct_factorization (x : ℝ) :
  factor_polynomial x = (x + 5) * (x - 8) := 
sorry

-- Define the polynomial for the positivity proof
def positive_polynomial (x y : ℝ) : ℝ := x^2 + y^2 - 2 * x - 4 * y + 16

-- Positive polynomial proof
theorem positive_polynomial_gt_zero (x y : ℝ) :
  positive_polynomial x y > 0 := 
sorry

end transform_polynomial_eq_correct_factorization_positive_polynomial_gt_zero_l686_68641


namespace square_of_complex_l686_68662

def z : Complex := 5 - 2 * Complex.I

theorem square_of_complex : z^2 = 21 - 20 * Complex.I := by
  sorry

end square_of_complex_l686_68662


namespace solution_set_of_inequality_l686_68630

theorem solution_set_of_inequality (a x : ℝ) (h : 1 < a) :
  (x - a) * (x - (1 / a)) > 0 ↔ x < 1 / a ∨ x > a :=
by
  sorry

end solution_set_of_inequality_l686_68630


namespace max_value_ineq_l686_68692

variables {R : Type} [LinearOrderedField R]

theorem max_value_ineq (a b c x y z : R) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : 0 ≤ x) (h5 : 0 ≤ y) (h6 : 0 ≤ z)
  (h7 : a + b + c = 1) (h8 : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≤ 1 / 16 :=
sorry

end max_value_ineq_l686_68692


namespace Sam_distance_l686_68612

theorem Sam_distance (miles_Marguerite: ℝ) (hours_Marguerite: ℕ) (hours_Sam: ℕ) (speed_factor: ℝ) 
  (h1: miles_Marguerite = 150) 
  (h2: hours_Marguerite = 3) 
  (h3: hours_Sam = 4)
  (h4: speed_factor = 1.2) :
  let average_speed_Marguerite := miles_Marguerite / hours_Marguerite
  let average_speed_Sam := speed_factor * average_speed_Marguerite
  let distance_Sam := average_speed_Sam * hours_Sam
  distance_Sam = 240 := 
by 
  sorry

end Sam_distance_l686_68612


namespace reduced_price_l686_68635

open Real

noncomputable def original_price : ℝ := 33.33

variables (P R: ℝ) (Q : ℝ)

theorem reduced_price
  (h1 : R = 0.75 * P)
  (h2 : P * 500 / P = 500)
  (h3 : 0.75 * P * (Q + 5) = 500)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  -- The proof will be provided here
  sorry

end reduced_price_l686_68635


namespace find_c_if_quadratic_lt_zero_l686_68653

theorem find_c_if_quadratic_lt_zero (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c * x - 12 < 0 ↔ (x < 2 ∨ x > 7)) → c = 9 := 
by
  sorry

end find_c_if_quadratic_lt_zero_l686_68653


namespace third_quadrant_angle_bisector_l686_68627

theorem third_quadrant_angle_bisector
  (a b : ℝ)
  (hA : A = (-4,a))
  (hB : B = (-2,b))
  (h_lineA : a = -4)
  (h_lineB : b = -2)
  : a + b + a * b = 2 :=
by
  sorry

end third_quadrant_angle_bisector_l686_68627


namespace largest_area_polygons_l686_68678

-- Define the area of each polygon
def area_P := 4
def area_Q := 6
def area_R := 3 + 3 * (1 / 2)
def area_S := 6 * (1 / 2)
def area_T := 5 + 2 * (1 / 2)

-- Proof of the polygons with the largest area
theorem largest_area_polygons : (area_Q = 6 ∧ area_T = 6) ∧ area_Q ≥ area_P ∧ area_Q ≥ area_R ∧ area_Q ≥ area_S :=
by
  sorry

end largest_area_polygons_l686_68678


namespace positive_difference_16_l686_68649

def avg_is_37 (y : ℤ) : Prop := (45 + y) / 2 = 37

def positive_difference (a b : ℤ) : ℤ := if a > b then a - b else b - a

theorem positive_difference_16 (y : ℤ) (h : avg_is_37 y) : positive_difference 45 y = 16 :=
by
  sorry

end positive_difference_16_l686_68649


namespace value_of_each_other_toy_l686_68667

-- Definitions for the conditions
def total_toys : ℕ := 9
def total_worth : ℕ := 52
def single_toy_value : ℕ := 12

-- Definition to represent the value of each of the other toys
def other_toys_value (same_value : ℕ) : Prop :=
  (total_worth - single_toy_value) / (total_toys - 1) = same_value

-- The theorem to be proven
theorem value_of_each_other_toy : other_toys_value 5 :=
  sorry

end value_of_each_other_toy_l686_68667


namespace general_term_a_n_sum_of_b_n_l686_68619

-- Proof Problem 1: General term of sequence {a_n}
theorem general_term_a_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4) 
    (h3 : ∀ n ≥ 2, a (n+1) - a n = 2) : 
    ∀ n, a n = 2 * n :=
by
  sorry

-- Proof Problem 2: Sum of the first n terms of sequence {b_n}
theorem sum_of_b_n (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
    (h : ∀ n, (1 / (a n ^ 2 - 1) : ℝ) + b n = 2^n) :
    T n = 2^(n+1) - n / (2*n + 1) :=
by
  sorry

end general_term_a_n_sum_of_b_n_l686_68619


namespace second_number_value_l686_68652

theorem second_number_value 
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a / b = 3 / 4)
  (h3 : b / c = 2 / 5) :
  b = 480 / 17 :=
by
  sorry

end second_number_value_l686_68652


namespace roots_ratio_sum_l686_68698

theorem roots_ratio_sum (a b m : ℝ) 
  (m1 m2 : ℝ)
  (h_roots : a ≠ b ∧ b ≠ 0 ∧ m ≠ 0 ∧ a ≠ 0 ∧ 
    ∀ x : ℝ, m * (x^2 - 3 * x) + 2 * x + 7 = 0 → (x = a ∨ x = b)) 
  (h_ratio : (a / b) + (b / a) = 7 / 3)
  (h_m1_m2_eq : ((3 * m - 2) ^ 2) / (7 * m) - 2 = 7 / 3)
  (h_m_vieta : (3 * m - 2) ^ 2 - 27 * m * (91 / 3) = 0) :
  (m1 + m2 = 127 / 27) ∧ (m1 * m2 = 4 / 9) →
  ((m1 / m2) + (m2 / m1) = 47.78) :=
sorry

end roots_ratio_sum_l686_68698


namespace derivative_of_sin_squared_is_sin_2x_l686_68634

open Real

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2

theorem derivative_of_sin_squared_is_sin_2x : 
  ∀ x : ℝ, deriv f x = sin (2 * x) :=
by
  sorry

end derivative_of_sin_squared_is_sin_2x_l686_68634


namespace IncorrectStatement_l686_68640

-- Definitions of the events
def EventA (planeShot : ℕ → Prop) : Prop := planeShot 1 ∧ planeShot 2
def EventB (planeShot : ℕ → Prop) : Prop := ¬planeShot 1 ∧ ¬planeShot 2
def EventC (planeShot : ℕ → Prop) : Prop := (planeShot 1 ∧ ¬planeShot 2) ∨ (¬planeShot 1 ∧ planeShot 2)
def EventD (planeShot : ℕ → Prop) : Prop := planeShot 1 ∨ planeShot 2

-- Theorem statement to be proved (negation of the incorrect statement)
theorem IncorrectStatement (planeShot : ℕ → Prop) :
  ¬((EventA planeShot ∨ EventC planeShot) = (EventB planeShot ∨ EventD planeShot)) :=
by
  sorry

end IncorrectStatement_l686_68640


namespace number_of_integers_in_double_inequality_l686_68695

noncomputable def pi_approx : ℝ := 3.14
noncomputable def sqrt_pi_approx : ℝ := Real.sqrt pi_approx
noncomputable def lower_bound : ℝ := -12 * sqrt_pi_approx
noncomputable def upper_bound : ℝ := 15 * pi_approx

theorem number_of_integers_in_double_inequality : 
  ∃ n : ℕ, n = 13 ∧ ∀ k : ℤ, lower_bound ≤ (k^2 : ℝ) ∧ (k^2 : ℝ) ≤ upper_bound → (-6 ≤ k ∧ k ≤ 6) :=
by
  sorry

end number_of_integers_in_double_inequality_l686_68695


namespace abc_plus_2_gt_a_plus_b_plus_c_l686_68693

theorem abc_plus_2_gt_a_plus_b_plus_c (a b c : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) (h3 : |c| < 1) : abc + 2 > a + b + c :=
by
  sorry

end abc_plus_2_gt_a_plus_b_plus_c_l686_68693


namespace Adam_total_shopping_cost_l686_68637

theorem Adam_total_shopping_cost :
  let sandwiches := 3
  let sandwich_cost := 3
  let water_cost := 2
  (sandwiches * sandwich_cost + water_cost) = 11 := 
by
  sorry

end Adam_total_shopping_cost_l686_68637


namespace tickets_per_box_l686_68646

-- Definitions
def boxes (G: Type) : ℕ := 9
def total_tickets (G: Type) : ℕ := 45

-- Theorem statement
theorem tickets_per_box (G: Type) : total_tickets G / boxes G = 5 :=
by
  sorry

end tickets_per_box_l686_68646


namespace intersection_M_N_l686_68694

def M : Set ℝ := { x | x^2 ≤ 4 }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≥ 1 }

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end intersection_M_N_l686_68694


namespace article_filling_correct_l686_68632

-- definitions based on conditions provided
def Gottlieb_Daimler := "Gottlieb Daimler was a German engineer."
def Invented_Car := "Daimler is normally believed to have invented the car."

-- Statement we want to prove
theorem article_filling_correct : 
  (Gottlieb_Daimler = "Gottlieb Daimler was a German engineer.") ∧ 
  (Invented_Car = "Daimler is normally believed to have invented the car.") →
  ("Gottlieb Daimler, a German engineer, is normally believed to have invented the car." = 
   "Gottlieb Daimler, a German engineer, is normally believed to have invented the car.") :=
by
  sorry

end article_filling_correct_l686_68632


namespace sequence_b_10_eq_110_l686_68613

theorem sequence_b_10_eq_110 :
  (∃ (b : ℕ → ℕ), b 1 = 2 ∧ (∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n) ∧ b 10 = 110) :=
sorry

end sequence_b_10_eq_110_l686_68613


namespace sum_of_eight_numbers_l686_68638

theorem sum_of_eight_numbers (average : ℝ) (h : average = 5) :
  (8 * average) = 40 :=
by
  sorry

end sum_of_eight_numbers_l686_68638


namespace domain_of_g_l686_68675

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

theorem domain_of_g :
  {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)} = -- Expected domain of g(x)
  { x : ℝ |
    (0 ≤ x ∧ x ≤ 6) ∧ -- Domain of f is 0 ≤ x ≤ 6
    2 * x ≤ 6 ∧ -- For g(x) to be in the domain of f(2x)
    0 ≤ 2 * x ∧ -- Ensures 2x fits within the domain 0 < 2x < 6
    x ≠ 2 } -- x cannot be 2
:= sorry

end domain_of_g_l686_68675


namespace remaining_volume_l686_68623

-- Given
variables (a d : ℚ) 
-- Define the volumes of sections as arithmetic sequence terms
def volume (n : ℕ) := a + n*d

-- Define total volume of bottom three sections
def bottomThreeVolume := volume a 0 + volume a d + volume a (2 * d) = 4

-- Define total volume of top four sections
def topFourVolume := volume a (5 * d) + volume a (6 * d) + volume a (7 * d) + volume a (8 * d) = 3

-- Define the volumes of the two middle sections
def middleTwoVolume := volume a (3 * d) + volume a (4 * d) = 2 + 3 / 22

-- Prove that the total volume of the remaining two sections is 2 3/22
theorem remaining_volume : bottomThreeVolume a d ∧ topFourVolume a d → middleTwoVolume a d :=
sorry  -- Placeholder for the actual proof

end remaining_volume_l686_68623


namespace sum_of_coefficients_l686_68689

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) :
  (1 - 2 * x)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                  a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 = -1 :=
sorry

end sum_of_coefficients_l686_68689


namespace find_first_number_l686_68685

theorem find_first_number 
  (first_number second_number hcf lcm : ℕ) 
  (hCF_condition : hcf = 12) 
  (lCM_condition : lcm = 396) 
  (one_number_condition : first_number = 99) 
  (relation_condition : first_number * second_number = hcf * lcm) : 
  second_number = 48 :=
by
  sorry

end find_first_number_l686_68685


namespace cube_difference_l686_68683

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 :=
sorry

end cube_difference_l686_68683


namespace base7_addition_XY_l686_68691

theorem base7_addition_XY (X Y : ℕ) (h1 : (Y + 2) % 7 = X % 7) (h2 : (X + 5) % 7 = 9 % 7) : X + Y = 6 :=
by sorry

end base7_addition_XY_l686_68691


namespace minimum_value_of_f_l686_68665

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2) + 2 * x

theorem minimum_value_of_f (h : ∀ x > 0, f x ≥ 3) : ∃ x, x > 0 ∧ f x = 3 :=
by
  sorry

end minimum_value_of_f_l686_68665


namespace pastries_made_l686_68603

theorem pastries_made (P cakes_sold pastries_sold extra_pastries : ℕ)
  (h1 : cakes_sold = 78)
  (h2 : pastries_sold = 154)
  (h3 : extra_pastries = 76)
  (h4 : pastries_sold = cakes_sold + extra_pastries) :
  P = 154 := sorry

end pastries_made_l686_68603


namespace find_FC_l686_68647

-- Define all given values and relationships
variables (DC CB AD AB ED FC : ℝ)
variables (h1 : DC = 9) (h2 : CB = 6)
variables (h3 : AB = (1/3) * AD)
variables (h4 : ED = (2/3) * AD)

-- Define the goal
theorem find_FC :
  FC = 9 :=
sorry

end find_FC_l686_68647


namespace employee_salary_l686_68608

theorem employee_salary (A B : ℝ) (h1 : A + B = 560) (h2 : A = 1.5 * B) : B = 224 :=
by
  sorry

end employee_salary_l686_68608


namespace find_constants_PQR_l686_68659

theorem find_constants_PQR :
  ∃ P Q R : ℚ, 
    (P = (-8 / 15)) ∧ 
    (Q = (-7 / 6)) ∧ 
    (R = (27 / 10)) ∧
    (∀ x : ℚ, 
      (x - 1) ≠ 0 ∧ (x - 4) ≠ 0 ∧ (x - 6) ≠ 0 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by
  sorry

end find_constants_PQR_l686_68659


namespace find_f_at_4_l686_68600

def f (n : ℕ) : ℕ := sorry -- We define the function f.

theorem find_f_at_4 : (∀ x : ℕ, f (2 * x) = 3 * x^2 + 1) → f 4 = 13 :=
by
  sorry

end find_f_at_4_l686_68600


namespace find_common_ratio_l686_68687

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_common_ratio (a1 a4 q : ℝ) (hq : q ^ 3 = 8) (ha1 : a1 = 8) (ha4 : a4 = 64)
  (a_def : is_geometric_sequence (fun n => a1 * q ^ n) q) :
  q = 2 :=
by
  sorry

end find_common_ratio_l686_68687


namespace min_sum_geometric_sequence_l686_68618

noncomputable def sequence_min_value (a : ℕ → ℝ) : ℝ :=
  a 4 + a 3 - 2 * a 2 - 2 * a 1

theorem min_sum_geometric_sequence (a : ℕ → ℝ)
  (h : sequence_min_value a = 6) :
  a 5 + a 6 = 48 := 
by
  sorry

end min_sum_geometric_sequence_l686_68618


namespace find_k_of_division_property_l686_68651

theorem find_k_of_division_property (k : ℝ) :
  (3 * (1 / 3)^3 - k * (1 / 3)^2 + 4) % (3 * (1 / 3) - 1) = 5 → k = -8 :=
by sorry

end find_k_of_division_property_l686_68651
