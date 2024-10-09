import Mathlib

namespace sum_of_squares_l1605_160550

variable {x y : ℝ}

theorem sum_of_squares (h1 : x + y = 20) (h2 : x * y = 100) : x^2 + y^2 = 200 :=
sorry

end sum_of_squares_l1605_160550


namespace melissa_remaining_bananas_l1605_160581

theorem melissa_remaining_bananas :
  let initial_bananas := 88
  let shared_bananas := 4
  initial_bananas - shared_bananas = 84 :=
by
  sorry

end melissa_remaining_bananas_l1605_160581


namespace decompose_max_product_l1605_160543

theorem decompose_max_product (a : ℝ) (h_pos : a > 0) :
  ∃ x y : ℝ, x + y = a ∧ x * y ≤ (a / 2) * (a / 2) :=
by
  sorry

end decompose_max_product_l1605_160543


namespace banana_popsicles_count_l1605_160531

theorem banana_popsicles_count 
  (grape_popsicles cherry_popsicles total_popsicles : ℕ)
  (h1 : grape_popsicles = 2)
  (h2 : cherry_popsicles = 13)
  (h3 : total_popsicles = 17) :
  total_popsicles - (grape_popsicles + cherry_popsicles) = 2 := by
  sorry

end banana_popsicles_count_l1605_160531


namespace surface_area_of_cone_l1605_160593

-- Definitions based solely on conditions
def central_angle (θ : ℝ) := θ = (2 * Real.pi) / 3
def slant_height (l : ℝ) := l = 2
def radius_cone (r : ℝ) := ∃ (θ l : ℝ), central_angle θ ∧ slant_height l ∧ θ * l = 2 * Real.pi * r
def lateral_surface_area (A₁ : ℝ) (r l : ℝ) := A₁ = Real.pi * r * l
def base_area (A₂ : ℝ) (r : ℝ) := A₂ = Real.pi * r^2
def total_surface_area (A A₁ A₂ : ℝ) := A = A₁ + A₂

-- The theorem proving the total surface area is as specified
theorem surface_area_of_cone :
  ∃ (r l A₁ A₂ A : ℝ), central_angle ((2 * Real.pi) / 3) ∧ slant_height 2 ∧ radius_cone r ∧
  lateral_surface_area A₁ r 2 ∧ base_area A₂ r ∧ total_surface_area A A₁ A₂ ∧ A = (16 * Real.pi) / 9 := sorry

end surface_area_of_cone_l1605_160593


namespace min_value_f_l1605_160585

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1)^2 + (Real.exp (-x) - 1)^2

theorem min_value_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 :=
sorry

end min_value_f_l1605_160585


namespace expression_equality_l1605_160571

theorem expression_equality (a b c : ℝ) : a * (a + b - c) = a^2 + a * b - a * c :=
by
  sorry

end expression_equality_l1605_160571


namespace remainder_of_127_div_25_is_2_l1605_160523

theorem remainder_of_127_div_25_is_2 : ∃ r, 127 = 25 * 5 + r ∧ r = 2 := by
  have h1 : 127 = 25 * 5 + (127 - 25 * 5) := by rw [mul_comm 25 5, mul_comm 5 25]
  have h2 : 127 - 25 * 5 = 2 := by norm_num
  exact ⟨127 - 25 * 5, h1, h2⟩

end remainder_of_127_div_25_is_2_l1605_160523


namespace cone_cylinder_volume_ratio_l1605_160589

theorem cone_cylinder_volume_ratio (h r : ℝ) (hc_pos : h > 0) (r_pos : r > 0) :
  let V_cylinder := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * (3 / 4 * h)
  (V_cone / V_cylinder) = 1 / 4 := 
by 
  sorry

end cone_cylinder_volume_ratio_l1605_160589


namespace striped_jerseys_count_l1605_160556

-- Define the cost of long-sleeved jerseys
def cost_long_sleeved := 15
-- Define the cost of striped jerseys
def cost_striped := 10
-- Define the number of long-sleeved jerseys bought
def num_long_sleeved := 4
-- Define the total amount spent
def total_spent := 80

-- Define a theorem to prove the number of striped jerseys bought
theorem striped_jerseys_count : ∃ x : ℕ, x * cost_striped = total_spent - num_long_sleeved * cost_long_sleeved ∧ x = 2 := 
by 
-- TODO: The proof steps would go here, but for this exercise, we use 'sorry' to skip the proof.
sorry

end striped_jerseys_count_l1605_160556


namespace rainfall_difference_l1605_160575

-- Define the conditions
def day1_rainfall := 26
def day2_rainfall := 34
def average_rainfall := 140
def less_rainfall := 58

-- Calculate the total rainfall this year in the first three days
def total_rainfall_this_year := average_rainfall - less_rainfall

-- Calculate the total rainfall in the first two days
def total_first_two_days := day1_rainfall + day2_rainfall

-- Calculate the rainfall on the third day
def day3_rainfall := total_rainfall_this_year - total_first_two_days

-- The proof problem
theorem rainfall_difference : day2_rainfall - day3_rainfall = 12 := 
by
  sorry

end rainfall_difference_l1605_160575


namespace x_must_be_negative_l1605_160544

theorem x_must_be_negative (x y : ℝ) (h1 : y ≠ 0) (h2 : y > 0) (h3 : x / y < -3) : x < 0 :=
by 
  sorry

end x_must_be_negative_l1605_160544


namespace car_speed_in_second_hour_l1605_160504

theorem car_speed_in_second_hour (x : ℕ) : 84 = (98 + x) / 2 → x = 70 := 
sorry

end car_speed_in_second_hour_l1605_160504


namespace mass_of_empty_glass_l1605_160501

theorem mass_of_empty_glass (mass_full : ℕ) (mass_half : ℕ) (G : ℕ) :
  mass_full = 1000 →
  mass_half = 700 →
  G = mass_full - (mass_full - mass_half) * 2 →
  G = 400 :=
by
  intros h_full h_half h_G_eq
  sorry

end mass_of_empty_glass_l1605_160501


namespace ratio_blue_to_total_l1605_160506

theorem ratio_blue_to_total (total_marbles red_marbles green_marbles yellow_marbles blue_marbles : ℕ)
    (h_total : total_marbles = 164)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27)
    (h_yellow : yellow_marbles = 14)
    (h_blue : blue_marbles = total_marbles - (red_marbles + green_marbles + yellow_marbles)) :
  blue_marbles / total_marbles = 1 / 2 :=
by
  sorry

end ratio_blue_to_total_l1605_160506


namespace g_of_f_neg_5_l1605_160565

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 8

-- Assume g(42) = 17
axiom g_f_5_eq_17 : ∀ (g : ℝ → ℝ), g (f 5) = 17

-- State the theorem to be proven
theorem g_of_f_neg_5 (g : ℝ → ℝ) : g (f (-5)) = 17 :=
by
  sorry

end g_of_f_neg_5_l1605_160565


namespace university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l1605_160517

-- Part 1
def probability_A_exactly_one_subject : ℚ :=
  3 * (1/2) * (1/2)^2

def probability_B_exactly_one_subject (m : ℚ) : ℚ :=
  (1/6) * (2/5)^2 + (5/6) * (3/5) * (2/5) * 2

theorem university_A_pass_one_subject : probability_A_exactly_one_subject = 3/8 :=
sorry

theorem university_B_pass_one_subject_when_m_3_5 : probability_B_exactly_one_subject (3/5) = 32/75 :=
sorry

-- Part 2
def expected_A : ℚ :=
  3 * (1/2)

def expected_B (m : ℚ) : ℚ :=
  ((17 - 7 * m) / 30) + (2 * (3 + 14 * m) / 30) + (3 * m / 10)

theorem preferred_range_of_m : 0 < m ∧ m < 11/15 → expected_A > expected_B m :=
sorry

end university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l1605_160517


namespace candy_count_in_third_set_l1605_160564

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l1605_160564


namespace sequence_converges_to_one_l1605_160534

noncomputable def u (n : ℕ) : ℝ :=
1 + (Real.sin n) / n

theorem sequence_converges_to_one :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n - 1| ≤ ε :=
sorry

end sequence_converges_to_one_l1605_160534


namespace consumer_credit_amount_l1605_160529

theorem consumer_credit_amount
  (C A : ℝ)
  (h1 : A = 0.20 * C)
  (h2 : 57 = 1/3 * A) :
  C = 855 := by
  sorry

end consumer_credit_amount_l1605_160529


namespace train_crosses_platform_in_34_seconds_l1605_160533

theorem train_crosses_platform_in_34_seconds 
    (train_speed_kmph : ℕ) 
    (time_cross_man_sec : ℕ) 
    (platform_length_m : ℕ) 
    (h_speed : train_speed_kmph = 72) 
    (h_time : time_cross_man_sec = 18) 
    (h_platform_length : platform_length_m = 320) 
    : (platform_length_m + (train_speed_kmph * 1000 / 3600) * time_cross_man_sec) / (train_speed_kmph * 1000 / 3600) = 34 :=
by
    sorry

end train_crosses_platform_in_34_seconds_l1605_160533


namespace mixed_oil_rate_per_litre_l1605_160520

variables (volume1 : ℝ) (price1 : ℝ) (volume2 : ℝ) (price2 : ℝ)

def total_cost (v p : ℝ) : ℝ := v * p
def total_volume (v1 v2 : ℝ) : ℝ := v1 + v2

theorem mixed_oil_rate_per_litre (h1 : volume1 = 10) (h2 : price1 = 55) (h3 : volume2 = 5) (h4 : price2 = 66) :
  (total_cost volume1 price1 + total_cost volume2 price2) / total_volume volume1 volume2 = 58.67 := 
by
  sorry

end mixed_oil_rate_per_litre_l1605_160520


namespace inserting_eights_is_composite_l1605_160580

theorem inserting_eights_is_composite (n : ℕ) : ¬ Nat.Prime (2000 * 10^n + 8 * ((10^n - 1) / 9) + 21) := 
by sorry

end inserting_eights_is_composite_l1605_160580


namespace fraction_comparison_and_differences_l1605_160569

theorem fraction_comparison_and_differences :
  (1/3 < 0.5) ∧ (0.5 < 3/5) ∧ 
  (0.5 - 1/3 = 1/6) ∧ 
  (3/5 - 0.5 = 1/10) :=
by
  sorry

end fraction_comparison_and_differences_l1605_160569


namespace serves_probability_l1605_160576

variable (p : ℝ) (hpos : 0 < p) (hneq0 : p ≠ 0)

def ExpectedServes (p : ℝ) : ℝ :=
  p + 2 * p * (1 - p) + 3 * (1 - p) ^ 2

theorem serves_probability (h : ExpectedServes p > 1.75) : 0 < p ∧ p < 1 / 2 :=
  sorry

end serves_probability_l1605_160576


namespace exactly_two_statements_true_l1605_160522

noncomputable def f : ℝ → ℝ := sorry -- Definition of f satisfying the conditions

-- Conditions
axiom functional_eq (x : ℝ) : f (x + 3/2) + f x = 0
axiom odd_function (x : ℝ) : f (- x - 3/4) = - f (x - 3/4)

-- Proof statement
theorem exactly_two_statements_true : 
  (¬(∀ (T : ℝ), T > 0 → (∀ (x : ℝ), f (x + T) = f x) → T = 3/2) ∧
   (∀ (x : ℝ), f (-x - 3/4) = - f (x - 3/4)) ∧
   (¬(∀ (x : ℝ), f x = f (-x)))) :=
sorry

end exactly_two_statements_true_l1605_160522


namespace f_diff_l1605_160515

-- Define the function f(n)
def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n + 1 + 1)).sum (λ i => 1 / (n + i + 1))

-- The theorem stating the main problem
theorem f_diff (k : ℕ) : 
  f (k + 1) - f k = (1 / (3 * k + 2)) + (1 / (3 * k + 3)) + (1 / (3 * k + 4)) - (1 / (k + 1)) :=
by
  sorry

end f_diff_l1605_160515


namespace joe_first_lift_weight_l1605_160539

theorem joe_first_lift_weight (x y : ℕ) (h1 : x + y = 1500) (h2 : 2 * x = y + 300) : x = 600 :=
by
  sorry

end joe_first_lift_weight_l1605_160539


namespace solve_for_x_l1605_160546

theorem solve_for_x (x : ℝ) : 3 * x^2 + 15 = 3 * (2 * x + 20) → (x = 5 ∨ x = -3) :=
sorry

end solve_for_x_l1605_160546


namespace solve_inequality_l1605_160551

-- Define the odd and monotonically decreasing function
noncomputable def f : ℝ → ℝ := sorry

-- Assume the given conditions
axiom odd_f : ∀ x, f (-x) = -f x
axiom decreasing_f : ∀ x y, x < y → y < 0 → f x > f y
axiom f_at_2 : f 2 = 0

-- The proof statement
theorem solve_inequality (x : ℝ) : (x - 1) * f (x + 1) > 0 ↔ -3 < x ∧ x < -1 :=
by
  -- Proof omitted
  sorry

end solve_inequality_l1605_160551


namespace num_books_second_shop_l1605_160574

-- Define the conditions
def num_books_first_shop : ℕ := 32
def cost_first_shop : ℕ := 1500
def cost_second_shop : ℕ := 340
def avg_price_per_book : ℕ := 20

-- Define the proof statement
theorem num_books_second_shop : 
  (num_books_first_shop + (cost_second_shop + cost_first_shop) / avg_price_per_book) - num_books_first_shop = 60 := by
  sorry

end num_books_second_shop_l1605_160574


namespace votes_distribution_l1605_160554

theorem votes_distribution (W : ℕ) 
  (h1 : W + (W - 53) + (W - 79) + (W - 105) = 963) 
  : W = 300 ∧ 247 = W - 53 ∧ 221 = W - 79 ∧ 195 = W - 105 :=
by
  sorry

end votes_distribution_l1605_160554


namespace josh_marbles_l1605_160548

theorem josh_marbles (original_marble : ℝ) (given_marble : ℝ)
  (h1 : original_marble = 22.5) (h2 : given_marble = 20.75) :
  original_marble + given_marble = 43.25 := by
  sorry

end josh_marbles_l1605_160548


namespace nat_pairs_solution_l1605_160572

theorem nat_pairs_solution (a b : ℕ) :
  a * (a + 5) = b * (b + 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) :=
by
  sorry

end nat_pairs_solution_l1605_160572


namespace part_I_equality_condition_part_II_l1605_160592

-- Lean statement for Part (I)
theorem part_I (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) : 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ 5 :=
sorry

theorem equality_condition (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) :
  (2 * Real.sqrt x + Real.sqrt (5 - x) = 5) ↔ (x = 4) :=
sorry

-- Lean statement for Part (II)
theorem part_II (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 5) → 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ |m - 2|) →
  (m ≥ 7 ∨ m ≤ -3) :=
sorry

end part_I_equality_condition_part_II_l1605_160592


namespace eighth_triangular_number_l1605_160552

def triangular_number (n: ℕ) : ℕ := n * (n + 1) / 2

theorem eighth_triangular_number : triangular_number 8 = 36 :=
by
  -- Proof here
  sorry

end eighth_triangular_number_l1605_160552


namespace storybook_pages_l1605_160545

def reading_start_date := 10
def reading_end_date := 20
def pages_per_day := 11
def number_of_days := reading_end_date - reading_start_date + 1
def total_pages := pages_per_day * number_of_days

theorem storybook_pages : total_pages = 121 := by
  sorry

end storybook_pages_l1605_160545


namespace largest_integer_solution_l1605_160588

theorem largest_integer_solution (x : ℤ) : 
  x < (92 / 21 : ℝ) → ∀ y : ℤ, y < (92 / 21 : ℝ) → y ≤ x :=
by
  sorry

end largest_integer_solution_l1605_160588


namespace find_interest_rate_l1605_160547

def interest_rate_borrowed (p_borrowed: ℝ) (p_lent: ℝ) (time: ℝ) (rate_lent: ℝ) (gain: ℝ) (r: ℝ) : Prop :=
  let interest_from_ramu := p_lent * rate_lent * time / 100
  let interest_to_anwar := p_borrowed * r * time / 100
  gain = interest_from_ramu - interest_to_anwar

theorem find_interest_rate :
  interest_rate_borrowed 3900 5655 3 9 824.85 5.95 := sorry

end find_interest_rate_l1605_160547


namespace average_remaining_numbers_l1605_160541

theorem average_remaining_numbers (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50 : ℝ) = 38) 
  (h_discard : 45 ∈ numbers ∧ 55 ∈ numbers) :
  let new_sum := numbers.sum - 45 - 55
  let new_len := 50 - 2
  (new_sum / new_len : ℝ) = 37.5 :=
by
  sorry

end average_remaining_numbers_l1605_160541


namespace min_value_expression_l1605_160599

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) (hxy : x + y = 6) : 
  ( (x - 1)^2 / (y - 2) + ( (y - 1)^2 / (x - 2) ) ) >= 8 :=
by 
  sorry

end min_value_expression_l1605_160599


namespace square_area_relation_l1605_160526

variable {lA lB : ℝ}

theorem square_area_relation (h : lB = 4 * lA) : lB^2 = 16 * lA^2 :=
by sorry

end square_area_relation_l1605_160526


namespace Petya_workout_duration_l1605_160577

theorem Petya_workout_duration :
  ∃ x : ℕ, (x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 135) ∧
            (x + 7 > x) ∧
            (x + 14 > x + 7) ∧
            (x + 21 > x + 14) ∧
            (x + 28 > x + 21) ∧
            x = 13 :=
by sorry

end Petya_workout_duration_l1605_160577


namespace bouquet_carnations_l1605_160566

def proportion_carnations (P : ℚ) (R : ℚ) (PC : ℚ) (RC : ℚ) : ℚ := PC + RC

theorem bouquet_carnations :
  let P := (7 / 10 : ℚ)
  let R := (3 / 10 : ℚ)
  let PC := (1 / 2) * P
  let RC := (2 / 3) * R
  let C := proportion_carnations P R PC RC
  (C * 100) = 55 :=
by
  sorry

end bouquet_carnations_l1605_160566


namespace greatest_k_for_quadratic_roots_diff_l1605_160516

theorem greatest_k_for_quadratic_roots_diff (k : ℝ)
  (H : ∀ x: ℝ, (x^2 + k * x + 8 = 0) → (∃ a b : ℝ, a ≠ b ∧ (a - b)^2 = 84)) :
  k = 2 * Real.sqrt 29 :=
by
  sorry

end greatest_k_for_quadratic_roots_diff_l1605_160516


namespace winning_percentage_is_65_l1605_160530

theorem winning_percentage_is_65 
  (total_games won_games : ℕ) 
  (h1 : total_games = 280) 
  (h2 : won_games = 182) :
  ((won_games : ℚ) / (total_games : ℚ)) * 100 = 65 :=
by
  sorry

end winning_percentage_is_65_l1605_160530


namespace cistern_leak_time_l1605_160595

theorem cistern_leak_time (R : ℝ) (L : ℝ) (eff_R : ℝ) : 
  (R = 1/5) → 
  (eff_R = 1/6) → 
  (eff_R = R - L) → 
  (1 / L = 30) :=
by
  intros hR heffR heffRate
  sorry

end cistern_leak_time_l1605_160595


namespace sum_first_100_sum_51_to_100_l1605_160567

noncomputable def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem sum_first_100 : sum_natural_numbers 100 = 5050 :=
  sorry

theorem sum_51_to_100 : sum_natural_numbers 100 - sum_natural_numbers 50 = 3775 :=
  sorry

end sum_first_100_sum_51_to_100_l1605_160567


namespace driving_time_in_fog_is_correct_l1605_160553

-- Define constants for speeds (in miles per minute)
def speed_sunny : ℚ := 35 / 60
def speed_rain : ℚ := 25 / 60
def speed_fog : ℚ := 15 / 60

-- Total distance and time
def total_distance : ℚ := 19.5
def total_time : ℚ := 45

-- Time variables for rain and fog
variables (t_r t_f : ℚ)

-- Define the driving distance equation
def distance_eq : Prop :=
  speed_sunny * (total_time - t_r - t_f) + speed_rain * t_r + speed_fog * t_f = total_distance

-- Prove the time driven in fog equals 10.25 minutes
theorem driving_time_in_fog_is_correct (h : distance_eq t_r t_f) : t_f = 10.25 :=
sorry

end driving_time_in_fog_is_correct_l1605_160553


namespace descent_time_on_moving_escalator_standing_l1605_160590

theorem descent_time_on_moving_escalator_standing (l v_mont v_ek t : ℝ)
  (H1 : l / v_mont = 42)
  (H2 : l / (v_mont + v_ek) = 24)
  : t = 56 := by
  sorry

end descent_time_on_moving_escalator_standing_l1605_160590


namespace handshake_problem_l1605_160586

theorem handshake_problem (x : ℕ) (hx : (x * (x - 1)) / 2 = 55) : x = 11 := 
sorry

end handshake_problem_l1605_160586


namespace num_valid_m_l1605_160559

theorem num_valid_m (m : ℕ) : (∃ n : ℕ, n * (m^2 - 3) = 1722) → ∃ p : ℕ, p = 3 := 
  by
  sorry

end num_valid_m_l1605_160559


namespace max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l1605_160512

noncomputable def max_perimeter_of_right_angled_quadrilateral (r : ℝ) : ℝ :=
  4 * r * Real.sqrt 2

theorem max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2
  (r : ℝ) :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 4 * r^2 → 2 * (x + y) ≤ max_perimeter_of_right_angled_quadrilateral r)
  ∧ (k = max_perimeter_of_right_angled_quadrilateral r) :=
sorry

end max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l1605_160512


namespace probability_of_graduate_degree_l1605_160549

variables (G C N : ℕ)
axiom h1 : G / N = 1 / 8
axiom h2 : C / N = 2 / 3

noncomputable def total_college_graduates (G C : ℕ) : ℕ := G + C

noncomputable def probability_graduate_degree (G C : ℕ) : ℚ := G / (total_college_graduates G C)

theorem probability_of_graduate_degree :
  probability_graduate_degree 3 16 = 3 / 19 :=
by 
  -- Here, we need to prove that the probability of picking a college graduate with a graduate degree
  -- is 3 / 19 given the conditions.
  sorry

end probability_of_graduate_degree_l1605_160549


namespace window_side_length_is_five_l1605_160542

def pane_width (x : ℝ) : ℝ := x
def pane_height (x : ℝ) : ℝ := 3 * x
def border_width : ℝ := 1
def pane_rows : ℕ := 2
def pane_columns : ℕ := 3

theorem window_side_length_is_five (x : ℝ) (h : pane_height x = 3 * pane_width x) : 
  (3 * x + 4 = 6 * x + 3) -> (3 * x + 4 = 5) :=
by
  intros h1
  sorry

end window_side_length_is_five_l1605_160542


namespace two_point_distribution_p_value_l1605_160579

noncomputable def X : Type := ℕ -- discrete random variable (two-point)
def p (E_X2 : ℝ): ℝ := E_X2 -- p == E(X)

theorem two_point_distribution_p_value (var_X : ℝ) (E_X : ℝ) (E_X2 : ℝ) 
    (h1 : var_X = 2 / 9) 
    (h2 : E_X = p E_X2) 
    (h3 : E_X2 = E_X): 
    E_X = 1 / 3 ∨ E_X = 2 / 3 :=
by
  sorry

end two_point_distribution_p_value_l1605_160579


namespace transformations_map_onto_itself_l1605_160562

noncomputable def recurring_pattern_map_count (s : ℝ) : ℕ := sorry

theorem transformations_map_onto_itself (s : ℝ) :
  recurring_pattern_map_count s = 2 := sorry

end transformations_map_onto_itself_l1605_160562


namespace average_speed_l1605_160511

def total_distance : ℝ := 200
def total_time : ℝ := 40

theorem average_speed (d t : ℝ) (h₁: d = total_distance) (h₂: t = total_time) : d / t = 5 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end average_speed_l1605_160511


namespace round_trip_time_l1605_160513

noncomputable def time_to_complete_trip (speed_without_load speed_with_load distance rest_stops_in_minutes : ℝ) : ℝ :=
  let rest_stops_in_hours := rest_stops_in_minutes / 60
  let half_rest_time := 2 * rest_stops_in_hours
  let total_rest_time := 2 * half_rest_time
  let travel_time_with_load := distance / speed_with_load
  let travel_time_without_load := distance / speed_without_load
  travel_time_with_load + travel_time_without_load + total_rest_time

theorem round_trip_time :
  time_to_complete_trip 13 11 143 30 = 26 :=
sorry

end round_trip_time_l1605_160513


namespace project_completion_time_l1605_160596

theorem project_completion_time :
  let A_work_rate := (1 / 30) * (2 / 3)
  let B_work_rate := (1 / 60) * (3 / 4)
  let C_work_rate := (1 / 40) * (5 / 6)
  let combined_work_rate_per_12_days := 12 * (A_work_rate + B_work_rate + C_work_rate)
  let remaining_work_after_12_days := 1 - (2 / 3)
  let additional_work_rates_over_5_days := 
        5 * A_work_rate + 
        5 * B_work_rate + 
        5 * C_work_rate
  let remaining_work_after_5_days := remaining_work_after_12_days - additional_work_rates_over_5_days
  let B_additional_time := remaining_work_after_5_days / B_work_rate
  12 + 5 + B_additional_time = 17.5 :=
sorry

end project_completion_time_l1605_160596


namespace cubic_sum_l1605_160560

theorem cubic_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end cubic_sum_l1605_160560


namespace largest_of_five_consecutive_composite_integers_under_40_l1605_160528

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def five_consecutive_composite_integers_under_40 : List ℕ :=
[32, 33, 34, 35, 36]

theorem largest_of_five_consecutive_composite_integers_under_40 :
  ∀ n ∈ five_consecutive_composite_integers_under_40,
  n < 40 ∧ ∀ k, (k ∈ five_consecutive_composite_integers_under_40 →
  ¬ is_prime k) →
  List.maximum five_consecutive_composite_integers_under_40 = some 36 :=
by
  sorry

end largest_of_five_consecutive_composite_integers_under_40_l1605_160528


namespace two_digit_multiple_condition_l1605_160540

theorem two_digit_multiple_condition :
  ∃ x : ℕ, 10 ≤ x ∧ x < 100 ∧ ∃ k : ℤ, x = 30 * k + 2 :=
by
  sorry

end two_digit_multiple_condition_l1605_160540


namespace min_abc_value_l1605_160524

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem min_abc_value
  (a b c : ℕ)
  (h1: is_prime a)
  (h2 : is_prime b)
  (h3 : is_prime c)
  (h4 : a^5 ∣ (b^2 - c))
  (h5 : ∃ k : ℕ, (b + c) = k^2) :
  a * b * c = 1958 := sorry

end min_abc_value_l1605_160524


namespace find_m_l1605_160514

theorem find_m {A B : Set ℝ} (m : ℝ) :
  (A = {x : ℝ | x^2 + x - 12 = 0}) →
  (B = {x : ℝ | mx + 1 = 0}) →
  (A ∩ B = {3}) →
  m = -1 / 3 := 
by
  intros hA hB h_inter
  sorry

end find_m_l1605_160514


namespace probability_of_meeting_at_cafe_l1605_160563

open Set

/-- Define the unit square where each side represents 1 hour (from 2:00 to 3:00 PM). -/
def unit_square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

/-- Define the overlap condition for Cara and David meeting at the café. -/
def overlap_region : Set (ℝ × ℝ) :=
  { p | max (p.1 - 0.5) 0 ≤ p.2 ∧ p.2 ≤ min (p.1 + 0.5) 1 }

/-- The area of the overlap region within the unit square. -/
noncomputable def overlap_area : ℝ :=
  ∫ x in Icc 0 1, (min (x + 0.5) 1 - max (x - 0.5) 0)

theorem probability_of_meeting_at_cafe : overlap_area / 1 = 1 / 2 :=
by
  sorry

end probability_of_meeting_at_cafe_l1605_160563


namespace factorial_division_l1605_160536

-- Definitions of factorial used in Lean according to math problem statement.
open Nat

-- Statement of the proof problem in Lean 4.
theorem factorial_division : (12! - 11!) / 10! = 121 := by
  sorry

end factorial_division_l1605_160536


namespace probability_king_even_coords_2008_l1605_160558

noncomputable def king_probability_even_coords (turns : ℕ) : ℝ :=
  let p_stay := 0.4
  let p_edge := 0.1
  let p_diag := 0.05
  if turns = 2008 then
    (5 ^ 2008 + 1) / (2 * 5 ^ 2008)
  else
    0 -- default value for other cases

theorem probability_king_even_coords_2008 :
  king_probability_even_coords 2008 = (5 ^ 2008 + 1) / (2 * 5 ^ 2008) :=
by
  sorry

end probability_king_even_coords_2008_l1605_160558


namespace inclination_angle_range_l1605_160538

theorem inclination_angle_range :
  let Γ := fun x y : ℝ => x * abs x + y * abs y = 1
  let line (m : ℝ) := fun x y : ℝ => y = m * (x - 1)
  ∀ m : ℝ,
  (∃ p1 p2 p3 : ℝ × ℝ, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    line m p1.1 p1.2 ∧ Γ p1.1 p1.2 ∧ 
    line m p2.1 p2.2 ∧ Γ p2.1 p2.2 ∧ 
    line m p3.1 p3.2 ∧ Γ p3.1 p3.2) →
  (∃ θ : ℝ, θ ∈ (Set.Ioo (Real.pi / 2) (3 * Real.pi / 4) ∪ 
                  Set.Ioo (3 * Real.pi / 4) (Real.pi - Real.arctan (Real.sqrt 2 / 2)))) :=
sorry

end inclination_angle_range_l1605_160538


namespace right_triangle_other_side_l1605_160521

theorem right_triangle_other_side (c a : ℝ) (h_c : c = 10) (h_a : a = 6) : ∃ b : ℝ, b^2 = c^2 - a^2 ∧ b = 8 :=
by
  use 8
  rw [h_c, h_a]
  simp
  sorry

end right_triangle_other_side_l1605_160521


namespace Chad_saves_40_percent_of_his_earnings_l1605_160509

theorem Chad_saves_40_percent_of_his_earnings :
  let earnings_mow := 600
  let earnings_birthday := 250
  let earnings_games := 150
  let earnings_oddjobs := 150
  let amount_saved := 460
  (amount_saved / (earnings_mow + earnings_birthday + earnings_games + earnings_oddjobs) * 100) = 40 :=
by
  sorry

end Chad_saves_40_percent_of_his_earnings_l1605_160509


namespace arithmetic_geometric_sequence_ratio_l1605_160561

section
variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables {d : ℝ}

-- Definition of the arithmetic sequence and its sum
def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
  ∀ n, a n = a₁ + (n - 1) * d

def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given conditions
-- 1. S is the sum of the first n terms of the arithmetic sequence a
axiom sn_arith_seq : sum_arithmetic_sequence S a

-- 2. a_1, a_3, and a_4 form a geometric sequence
axiom geom_seq : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)

-- Goal is to prove the given ratio equation
theorem arithmetic_geometric_sequence_ratio (h : ∀ n, a n = -4 * d + (n - 1) * d) :
  (S 3 - S 2) / (S 5 - S 3) = 2 :=
sorry
end

end arithmetic_geometric_sequence_ratio_l1605_160561


namespace proof_option_b_and_c_l1605_160508

variable (a b c : ℝ)

theorem proof_option_b_and_c (h₀ : a > b) (h₁ : b > 0) (h₂ : c ≠ 0) :
  (b / a < (b + c^2) / (a + c^2)) ∧ (a^2 - 1 / a > b^2 - 1 / b) :=
by
  sorry

end proof_option_b_and_c_l1605_160508


namespace polynomial_factors_l1605_160568

theorem polynomial_factors (t q : ℤ) (h1 : 81 - 3 * t + q = 0) (h2 : -3 + t + q = 0) : |3 * t - 2 * q| = 99 :=
sorry

end polynomial_factors_l1605_160568


namespace gcd_a_b_l1605_160518

def a : ℕ := 333333333
def b : ℕ := 555555555

theorem gcd_a_b : Nat.gcd a b = 111111111 := 
by
  sorry

end gcd_a_b_l1605_160518


namespace prob_selecting_green_ball_l1605_160591

-- Definition of the number of red and green balls in each container
def containerI_red := 10
def containerI_green := 5
def containerII_red := 3
def containerII_green := 5
def containerIII_red := 2
def containerIII_green := 6
def containerIV_red := 4
def containerIV_green := 4

-- Total number of balls in each container
def total_balls_I := containerI_red + containerI_green
def total_balls_II := containerII_red + containerII_green
def total_balls_III := containerIII_red + containerIII_green
def total_balls_IV := containerIV_red + containerIV_green

-- Probability of selecting a green ball from each container
def prob_green_I := containerI_green / total_balls_I
def prob_green_II := containerII_green / total_balls_II
def prob_green_III := containerIII_green / total_balls_III
def prob_green_IV := containerIV_green / total_balls_IV

-- Probability of selecting any one container
def prob_select_container := (1:ℚ) / 4

-- Combined probability for a green ball from each container
def combined_prob_I := prob_select_container * prob_green_I 
def combined_prob_II := prob_select_container * prob_green_II 
def combined_prob_III := prob_select_container * prob_green_III 
def combined_prob_IV := prob_select_container * prob_green_IV 

-- Total probability of selecting a green ball
def total_prob_green := combined_prob_I + combined_prob_II + combined_prob_III + combined_prob_IV 

-- Theorem to prove
theorem prob_selecting_green_ball : total_prob_green = 53 / 96 :=
by sorry

end prob_selecting_green_ball_l1605_160591


namespace quadratic_real_roots_leq_l1605_160597

theorem quadratic_real_roots_leq (m : ℝ) :
  ∃ x : ℝ, x^2 - 3 * x + 2 * m = 0 → m ≤ 9 / 8 :=
by
  sorry

end quadratic_real_roots_leq_l1605_160597


namespace total_games_in_season_l1605_160570

-- Definitions based on the conditions
def num_teams := 16
def teams_per_division := 8
def num_divisions := num_teams / teams_per_division

-- Each team plays every other team in its division twice
def games_within_division_per_team := (teams_per_division - 1) * 2

-- Each team plays every team in the other division once
def games_across_divisions_per_team := teams_per_division

-- Total games per team
def games_per_team := games_within_division_per_team + games_across_divisions_per_team

-- Total preliminary games for all teams (each game is counted twice)
def preliminary_total_games := games_per_team * num_teams

-- Since each game is counted twice, the final number of games
def total_games := preliminary_total_games / 2

theorem total_games_in_season : total_games = 176 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end total_games_in_season_l1605_160570


namespace probability_of_correct_match_l1605_160555

theorem probability_of_correct_match :
  let n := 3
  let total_arrangements := Nat.factorial n
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = ((1: ℤ) / 6) :=
by
  sorry

end probability_of_correct_match_l1605_160555


namespace value_of_s_l1605_160583

-- Conditions: (m - 8) is a factor of m^2 - sm - 24

theorem value_of_s (s : ℤ) (m : ℤ) (h : (m - 8) ∣ (m^2 - s*m - 24)) : s = 5 :=
by
  sorry

end value_of_s_l1605_160583


namespace problem1_problem2_l1605_160500

-- Definition of a double root equation with the given condition
def is_double_root_equation (a b c : ℝ) := 
  ∃ x1 x2 : ℝ, a * x1 = 2 * a * x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0

-- Proving that x² - 6x + 8 = 0 is a double root equation
theorem problem1 : is_double_root_equation 1 (-6) 8 :=
  sorry

-- Proving that if (x-8)(x-n) = 0 is a double root equation, n is either 4 or 16
theorem problem2 (n : ℝ) (h : is_double_root_equation 1 (-8 - n) (8 * n)) :
  n = 4 ∨ n = 16 :=
  sorry

end problem1_problem2_l1605_160500


namespace boy_work_completion_days_l1605_160505

theorem boy_work_completion_days (M W B : ℚ) (D : ℚ)
  (h1 : M + W + B = 1 / 4)
  (h2 : M = 1 / 6)
  (h3 : W = 1 / 36)
  (h4 : B = 1 / D) :
  D = 18 := by
  sorry

end boy_work_completion_days_l1605_160505


namespace pencil_cost_l1605_160532

theorem pencil_cost (p e : ℝ) (h1 : p + e = 3.40) (h2 : p = 3 + e) : p = 3.20 :=
by
  sorry

end pencil_cost_l1605_160532


namespace fourth_root_squared_cubed_l1605_160594

theorem fourth_root_squared_cubed (x : ℝ) (h : (x^(1/4))^2^3 = 1296) : x = 256 :=
sorry

end fourth_root_squared_cubed_l1605_160594


namespace speed_against_current_l1605_160519

theorem speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) (man_speed_against_current : ℝ) 
  (h : speed_with_current = 12) (h1 : current_speed = 2) : man_speed_against_current = 8 :=
by
  sorry

end speed_against_current_l1605_160519


namespace max_distance_l1605_160502

noncomputable def starting_cost : ℝ := 10
noncomputable def additional_cost_per_km : ℝ := 1.5
noncomputable def round_up : ℝ := 1
noncomputable def total_fare : ℝ := 19

theorem max_distance (x : ℝ) : (starting_cost + additional_cost_per_km * (x - 4)) = total_fare → x = 10 :=
by sorry

end max_distance_l1605_160502


namespace infinite_series_eval_l1605_160525

open Filter
open Real
open Topology
open BigOperators

-- Define the relevant expression for the infinite sum
noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, (n / (n^4 - 4 * n^2 + 8))

-- The theorem statement
theorem infinite_series_eval : infinite_series_sum = 5 / 24 :=
by sorry

end infinite_series_eval_l1605_160525


namespace high_school_students_total_l1605_160587

theorem high_school_students_total
    (students_taking_music : ℕ)
    (students_taking_art : ℕ)
    (students_taking_both_music_and_art : ℕ)
    (students_taking_neither : ℕ)
    (h1 : students_taking_music = 50)
    (h2 : students_taking_art = 20)
    (h3 : students_taking_both_music_and_art = 10)
    (h4 : students_taking_neither = 440) :
    students_taking_music - students_taking_both_music_and_art + students_taking_art - students_taking_both_music_and_art + students_taking_both_music_and_art + students_taking_neither = 500 :=
by
  sorry

end high_school_students_total_l1605_160587


namespace smallest_a_l1605_160584

theorem smallest_a (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) : a = 17 :=
sorry

end smallest_a_l1605_160584


namespace tan_double_angle_l1605_160582

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (2 * α) = -4 / 3 :=
by
  sorry

end tan_double_angle_l1605_160582


namespace ratio_of_floors_l1605_160503

-- Define the number of floors of each building
def floors_building_A := 4
def floors_building_B := 4 + 9
def floors_building_C := 59

-- Prove the ratio of floors in Building C to Building B
theorem ratio_of_floors :
  floors_building_C / floors_building_B = 59 / 13 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_floors_l1605_160503


namespace greatest_good_t_l1605_160510

noncomputable def S (a t : ℕ) : Set ℕ := {x | ∃ n : ℕ, x = a + 1 + n ∧ n < t}

def is_good (S : Set ℕ) (k : ℕ) : Prop :=
∃ (coloring : ℕ → Fin k), ∀ (x y : ℕ), x ≠ y → x + y ∈ S → coloring x ≠ coloring y

theorem greatest_good_t {k : ℕ} (hk : k > 1) : ∃ t, ∀ a, is_good (S a t) k ∧ 
  ∀ t' > t, ¬ ∀ a, is_good (S a t') k := 
sorry

end greatest_good_t_l1605_160510


namespace storage_methods_l1605_160535

-- Definitions for the vertices and edges of the pyramid
structure Pyramid :=
  (P A B C D : Type)
  
-- Edges of the pyramid represented by pairs of vertices
def edges (P A B C D : Type) := [(P, A), (P, B), (P, C), (P, D), (A, B), (A, C), (A, D), (B, C), (B, D), (C, D)]

-- Safe storage condition: No edges sharing a common vertex in the same warehouse
def safe (edge1 edge2 : (Type × Type)) : Prop :=
  edge1.1 ≠ edge2.1 ∧ edge1.1 ≠ edge2.2 ∧ edge1.2 ≠ edge2.1 ∧ edge1.2 ≠ edge2.2

-- The number of different methods to store the chemical products safely
def number_of_safe_storage_methods : Nat :=
  -- We should replace this part by actual calculation or combinatorial methods relevant to the problem
  48

theorem storage_methods (P A B C D : Type) : number_of_safe_storage_methods = 48 :=
  sorry

end storage_methods_l1605_160535


namespace archer_total_fish_caught_l1605_160537

noncomputable def total_fish_caught (initial : ℕ) (second_extra : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := initial + second_extra
  let total_after_two_rounds := initial + second_round
  let third_round := second_round + (third_round_percentage * second_round) / 100
  total_after_two_rounds + third_round

theorem archer_total_fish_caught :
  total_fish_caught 8 12 60 = 60 :=
by
  -- Theorem statement to prove the total fish caught equals 60 given the conditions.
  sorry

end archer_total_fish_caught_l1605_160537


namespace chosen_number_l1605_160507

theorem chosen_number (x : ℤ) (h : x / 12 - 240 = 8) : x = 2976 :=
by sorry

end chosen_number_l1605_160507


namespace parking_space_unpainted_side_l1605_160573

theorem parking_space_unpainted_side 
  (L W : ℝ) 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 125) : 
  L = 8.90 := 
by 
  sorry

end parking_space_unpainted_side_l1605_160573


namespace problem_ineq_l1605_160598

theorem problem_ineq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq : a ≠ b) :
  (a^2 * b + a + b^2) * (a * b^2 + a^2 + b) > 9 * a^2 * b^2 := 
by 
  sorry

end problem_ineq_l1605_160598


namespace arithmetic_progression_x_value_l1605_160527

theorem arithmetic_progression_x_value (x: ℝ) (h1: 3*x - 1 - (2*x - 3) = 4*x + 1 - (3*x - 1)) : x = 3 :=
by
  sorry

end arithmetic_progression_x_value_l1605_160527


namespace total_cars_made_in_two_days_l1605_160557

-- Define the number of cars made yesterday.
def cars_yesterday : ℕ := 60

-- Define the number of cars made today, which is twice the number of cars made yesterday.
def cars_today : ℕ := 2 * cars_yesterday

-- Define the total number of cars made over the two days.
def total_cars : ℕ := cars_yesterday + cars_today

-- Proof statement: Prove that the total number of cars made in these two days is 180.
theorem total_cars_made_in_two_days : total_cars = 180 := by
  -- Here, we would provide the proof, but we'll use sorry to skip it.
  sorry

end total_cars_made_in_two_days_l1605_160557


namespace pencils_calculation_l1605_160578

variable (C B D : ℕ)

theorem pencils_calculation : 
  (C = B + 5) ∧
  (B = 2 * D - 3) ∧
  (C = 20) →
  D = 9 :=
by sorry

end pencils_calculation_l1605_160578
