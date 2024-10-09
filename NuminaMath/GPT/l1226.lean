import Mathlib

namespace max_correct_answers_l1226_122619

theorem max_correct_answers :
  ∃ (c w b : ℕ), c + w + b = 25 ∧ 4 * c - 3 * w = 57 ∧ c = 18 :=
by {
  sorry
}

end max_correct_answers_l1226_122619


namespace crayon_ratio_l1226_122616

theorem crayon_ratio :
  ∀ (Karen Beatrice Gilbert Judah : ℕ),
    Karen = 128 →
    Beatrice = Karen / 2 →
    Beatrice = Gilbert →
    Gilbert = 4 * Judah →
    Judah = 8 →
    Beatrice / Gilbert = 1 :=
by
  intros Karen Beatrice Gilbert Judah hKaren hBeatrice hEqual hGilbert hJudah
  sorry

end crayon_ratio_l1226_122616


namespace sphere_radius_and_volume_l1226_122652

theorem sphere_radius_and_volume (A : ℝ) (d : ℝ) (π : ℝ) (r : ℝ) (R : ℝ) (V : ℝ) 
  (h_cross_section : A = π) (h_distance : d = 1) (h_radius : r = 1) :
  R = Real.sqrt (r^2 + d^2) ∧ V = (4 / 3) * π * R^3 := 
by
  sorry

end sphere_radius_and_volume_l1226_122652


namespace total_boxes_sold_is_189_l1226_122660

-- Define the conditions
def boxes_sold_friday : ℕ := 40
def boxes_sold_saturday := 2 * boxes_sold_friday - 10
def boxes_sold_sunday := boxes_sold_saturday / 2
def boxes_sold_monday := boxes_sold_sunday + (boxes_sold_sunday / 4)

-- Define the total boxes sold over the four days
def total_boxes_sold := boxes_sold_friday + boxes_sold_saturday + boxes_sold_sunday + boxes_sold_monday

-- Theorem to prove the total number of boxes sold is 189
theorem total_boxes_sold_is_189 : total_boxes_sold = 189 := by
  sorry

end total_boxes_sold_is_189_l1226_122660


namespace sqrt_product_l1226_122669

theorem sqrt_product (a b c : ℝ) (ha : a = 72) (hb : b = 18) (hc : c = 8) :
  (Real.sqrt a) * (Real.sqrt b) * (Real.sqrt c) = 72 * Real.sqrt 2 :=
by
  sorry

end sqrt_product_l1226_122669


namespace train_speed_and_length_l1226_122679

theorem train_speed_and_length (V l : ℝ) 
  (h1 : 7 * V = l) 
  (h2 : 25 * V = 378 + l) : 
  V = 21 ∧ l = 147 :=
by
  sorry

end train_speed_and_length_l1226_122679


namespace sum_of_odd_coefficients_in_binomial_expansion_l1226_122604

theorem sum_of_odd_coefficients_in_binomial_expansion :
  let a_0 := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  (a_1 + a_3 + a_5 + a_7 + a_9) = 512 := by
  sorry

end sum_of_odd_coefficients_in_binomial_expansion_l1226_122604


namespace average_visitors_per_day_l1226_122608

theorem average_visitors_per_day:
  (∃ (Sundays OtherDays: ℕ) (visitors_per_sunday visitors_per_other_day: ℕ),
    Sundays = 4 ∧
    OtherDays = 26 ∧
    visitors_per_sunday = 600 ∧
    visitors_per_other_day = 240 ∧
    (Sundays + OtherDays = 30) ∧
    (Sundays * visitors_per_sunday + OtherDays * visitors_per_other_day) / 30 = 288) :=
sorry

end average_visitors_per_day_l1226_122608


namespace expected_digits_die_l1226_122675

noncomputable def expected_number_of_digits (numbers : List ℕ) : ℚ :=
  let one_digit_numbers := numbers.filter (λ n => n < 10)
  let two_digit_numbers := numbers.filter (λ n => n >= 10)
  let p_one_digit := (one_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  let p_two_digit := (two_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  p_one_digit * 1 + p_two_digit * 2

theorem expected_digits_die :
  expected_number_of_digits [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] = 1.5833 := 
by
  sorry

end expected_digits_die_l1226_122675


namespace percentage_increase_on_bought_price_l1226_122611

-- Define the conditions as Lean definitions
def original_price (P : ℝ) : ℝ := P
def bought_price (P : ℝ) : ℝ := 0.90 * P
def selling_price (P : ℝ) : ℝ := 1.62000000000000014 * P

-- Lean statement to prove the required result
theorem percentage_increase_on_bought_price (P : ℝ) :
  (selling_price P - bought_price P) / bought_price P * 100 = 80.00000000000002 := by
  sorry

end percentage_increase_on_bought_price_l1226_122611


namespace largest_a_for_integer_solution_l1226_122689

noncomputable def largest_integer_a : ℤ := 11

theorem largest_a_for_integer_solution :
  ∃ (x : ℤ), ∃ (a : ℤ), 
  (∃ (a : ℤ), a ≤ largest_integer_a) ∧
  (a = largest_integer_a → (
    (x^2 - (a + 7) * x + 7 * a)^3 = -3^3)) := 
by 
  sorry

end largest_a_for_integer_solution_l1226_122689


namespace find_base_17_digit_l1226_122639

theorem find_base_17_digit (a : ℕ) (h1 : 0 ≤ a ∧ a < 17) 
  (h2 : (25 + a) % 16 = 0) : a = 7 :=
sorry

end find_base_17_digit_l1226_122639


namespace blue_eyes_blonde_hair_logic_l1226_122672

theorem blue_eyes_blonde_hair_logic :
  ∀ (a b c d : ℝ), 
  (a / (a + b) > (a + c) / (a + b + c + d)) →
  (a / (a + c) > (a + b) / (a + b + c + d)) :=
by
  intro a b c d h
  sorry

end blue_eyes_blonde_hair_logic_l1226_122672


namespace solve_for_x_l1226_122601

theorem solve_for_x (x : ℝ) (h : |2000 * x + 2000| = 20 * 2000) : x = 19 ∨ x = -21 := 
by
  sorry

end solve_for_x_l1226_122601


namespace problem_statement_l1226_122684

-- Definitions of the conditions
variables (x y z w : ℕ)

-- The proof problem
theorem problem_statement
  (hx : x^3 = y^2)
  (hz : z^4 = w^3)
  (hzx : z - x = 17)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (hz_pos : z > 0)
  (hw_pos : w > 0) :
  w - y = 229 :=
sorry

end problem_statement_l1226_122684


namespace parabola_vertex_y_axis_opens_upwards_l1226_122631

theorem parabola_vertex_y_axis_opens_upwards :
  ∃ (a b c : ℝ), (a > 0) ∧ (b = 0) ∧ y = a * x^2 + b * x + c := 
sorry

end parabola_vertex_y_axis_opens_upwards_l1226_122631


namespace hyperbola_range_l1226_122680

theorem hyperbola_range (m : ℝ) : (∃ x y : ℝ, (x^2 / (|m| - 1) - y^2 / (m - 2) = 1)) → (-1 < m ∧ m < 1) ∨ (m > 2) := by
  sorry

end hyperbola_range_l1226_122680


namespace number_of_sides_of_polygon_l1226_122654

theorem number_of_sides_of_polygon (n : ℕ) : (n - 2) * 180 = 720 → n = 6 :=
by
  sorry

end number_of_sides_of_polygon_l1226_122654


namespace number_of_persons_in_first_group_eq_39_l1226_122620

theorem number_of_persons_in_first_group_eq_39 :
  ∀ (P : ℕ),
    (P * 12 * 5 = 15 * 26 * 6) →
    P = 39 :=
by
  intros P h
  have h1 : P = (15 * 26 * 6) / (12 * 5) := sorry
  simp at h1
  exact h1

end number_of_persons_in_first_group_eq_39_l1226_122620


namespace pradeep_failure_marks_l1226_122686

theorem pradeep_failure_marks :
  let total_marks := 925
  let pradeep_score := 160
  let passing_percentage := 20
  let passing_marks := (passing_percentage / 100) * total_marks
  let failed_by := passing_marks - pradeep_score
  failed_by = 25 :=
by
  sorry

end pradeep_failure_marks_l1226_122686


namespace short_side_is_7_l1226_122667

variable (L S : ℕ)

-- Given conditions
def perimeter : ℕ := 38
def long_side : ℕ := 12

-- In Lean, prove that the short side is 7 given L and P
theorem short_side_is_7 (h1 : 2 * L + 2 * S = perimeter) (h2 : L = long_side) : S = 7 := by
  sorry

end short_side_is_7_l1226_122667


namespace sufficient_but_not_necessary_l1226_122606

theorem sufficient_but_not_necessary {a b : ℝ} (h : a > b ∧ b > 0) : 
  a^2 > b^2 ∧ (¬ (a^2 > b^2 → a > b ∧ b > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_l1226_122606


namespace train_length_eq_1800_l1226_122653

theorem train_length_eq_1800 (speed_kmh : ℕ) (time_sec : ℕ) (distance : ℕ) (L : ℕ)
  (h_speed : speed_kmh = 216)
  (h_time : time_sec = 60)
  (h_distance : distance = 60 * time_sec)
  (h_total_distance : distance = 2 * L) :
  L = 1800 := by
  sorry

end train_length_eq_1800_l1226_122653


namespace roots_quadratic_l1226_122641

open Real

theorem roots_quadratic (a : ℤ) :
  (∃ (u v : ℤ), u ≠ v ∧ u + v = -a ∧ u * v = 2 * a) ↔ (a = -1 ∨ a = 9) :=
by
  sorry

end roots_quadratic_l1226_122641


namespace band_song_arrangements_l1226_122625

theorem band_song_arrangements (n : ℕ) (t : ℕ) (r : ℕ) 
  (h1 : n = 8) (h2 : t = 3) (h3 : r = 5) : 
  ∃ (ways : ℕ), ways = 14400 := by
  sorry

end band_song_arrangements_l1226_122625


namespace seashells_in_jar_at_end_of_month_l1226_122603

noncomputable def seashells_in_week (initial: ℕ) (increment: ℕ) (week: ℕ) : ℕ :=
  initial + increment * week

theorem seashells_in_jar_at_end_of_month :
  seashells_in_week 50 20 0 +
  seashells_in_week 50 20 1 +
  seashells_in_week 50 20 2 +
  seashells_in_week 50 20 3 = 320 :=
sorry

end seashells_in_jar_at_end_of_month_l1226_122603


namespace largest_eight_digit_number_contains_even_digits_l1226_122649

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l1226_122649


namespace community_members_after_five_years_l1226_122624

theorem community_members_after_five_years:
  ∀ (a : ℕ → ℕ),
  a 0 = 20 →
  (∀ k : ℕ, a (k + 1) = 4 * a k - 15) →
  a 5 = 15365 :=
by
  intros a h₀ h₁
  sorry

end community_members_after_five_years_l1226_122624


namespace cost_of_video_game_console_l1226_122605

-- Define the problem conditions
def earnings_Mar_to_Aug : ℕ := 460
def hours_Mar_to_Aug : ℕ := 23
def earnings_per_hour : ℕ := earnings_Mar_to_Aug / hours_Mar_to_Aug
def hours_Sep_to_Feb : ℕ := 8
def cost_car_fix : ℕ := 340
def additional_hours_needed : ℕ := 16

-- Proof that the cost of the video game console is $600
theorem cost_of_video_game_console :
  let initial_earnings := earnings_Mar_to_Aug
  let earnings_from_Sep_to_Feb := hours_Sep_to_Feb * earnings_per_hour
  let total_earnings_before_expenses := initial_earnings + earnings_from_Sep_to_Feb
  let current_savings := total_earnings_before_expenses - cost_car_fix
  let earnings_after_additional_work := additional_hours_needed * earnings_per_hour
  let total_savings := current_savings + earnings_after_additional_work
  total_savings = 600 :=
by
  sorry

end cost_of_video_game_console_l1226_122605


namespace multiple_solutions_no_solution_2891_l1226_122617

theorem multiple_solutions (n : ℤ) (x y : ℤ) (h1 : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (u v : ℤ), u ≠ x ∧ v ≠ y ∧ u^3 - 3 * u * v^2 + v^3 = n :=
  sorry

theorem no_solution_2891 (x y : ℤ) (h2 : x^3 - 3 * x * y^2 + y^3 = 2891) :
  false :=
  sorry

end multiple_solutions_no_solution_2891_l1226_122617


namespace find_value_of_reciprocal_sin_double_angle_l1226_122688

open Real

noncomputable def point := ℝ × ℝ

def term_side_angle_passes_through (α : ℝ) (P : point) :=
  ∃ (r : ℝ), P = (r * cos α, r * sin α)

theorem find_value_of_reciprocal_sin_double_angle (α : ℝ) (P : point) (h : term_side_angle_passes_through α P) :
  P = (-2, 1) → (1 / sin (2 * α)) = -5 / 4 :=
by
  intro hP
  sorry

end find_value_of_reciprocal_sin_double_angle_l1226_122688


namespace area_of_R3_l1226_122615

theorem area_of_R3 (r1 r2 r3 : ℝ) (h1: r1^2 = 25) 
                   (h2: r2 = (2/3) * r1) (h3: r3 = (2/3) * r2) :
                   r3^2 = 400 / 81 := 
by
  sorry

end area_of_R3_l1226_122615


namespace max_k_value_l1226_122635

theorem max_k_value (m : ℝ) (h : 0 < m ∧ m < 1/2) : 
  ∃ k : ℝ, (∀ m, 0 < m ∧ m < 1/2 → (1 / m + 2 / (1 - 2 * m)) ≥ k) ∧ k = 8 :=
by sorry

end max_k_value_l1226_122635


namespace probability_of_black_yellow_green_probability_of_not_red_or_green_l1226_122685

namespace ProbabilityProof

/- Definitions of events A, B, C, D representing probabilities as real numbers -/
variables (P_A P_B P_C P_D : ℝ)

/- Conditions stated in the problem -/
def conditions (h1 : P_A = 1 / 3)
               (h2 : P_B + P_C = 5 / 12)
               (h3 : P_C + P_D = 5 / 12)
               (h4 : P_A + P_B + P_C + P_D = 1) :=
  true

/- Proof that P(B) = 1/4, P(C) = 1/6, and P(D) = 1/4 given the conditions -/
theorem probability_of_black_yellow_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1) :
  P_B = 1 / 4 ∧ P_C = 1 / 6 ∧ P_D = 1 / 4 :=
by
  sorry

/- Proof that the probability of not drawing a red or green ball is 5/12 -/
theorem probability_of_not_red_or_green
  (P_A P_B P_C P_D : ℝ)
  (h1 : P_A = 1 / 3)
  (h2 : P_B + P_C = 5 / 12)
  (h3 : P_C + P_D = 5 / 12)
  (h4 : P_A + P_B + P_C + P_D = 1)
  (h5 : P_B = 1 / 4)
  (h6 : P_C = 1 / 6)
  (h7 : P_D = 1 / 4) :
  1 - (P_A + P_D) = 5 / 12 :=
by
  sorry

end ProbabilityProof

end probability_of_black_yellow_green_probability_of_not_red_or_green_l1226_122685


namespace fresh_grapes_water_percentage_l1226_122607

/--
Given:
- Fresh grapes contain a certain percentage (P%) of water by weight.
- Dried grapes contain 25% water by weight.
- The weight of dry grapes obtained from 200 kg of fresh grapes is 66.67 kg.

Prove:
- The percentage of water (P) in fresh grapes is 75%.
-/
theorem fresh_grapes_water_percentage
  (P : ℝ) (H1 : ∃ P, P / 100 * 200 = 0.75 * 66.67) :
  P = 75 :=
sorry

end fresh_grapes_water_percentage_l1226_122607


namespace integer_solutions_of_linear_diophantine_eq_l1226_122628

theorem integer_solutions_of_linear_diophantine_eq 
  (a b c : ℤ)
  (h_gcd : Int.gcd a b = 1)
  (x₀ y₀ : ℤ)
  (h_particular_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, (a * x + b * y = c) → ∃ (k : ℤ), (x = x₀ + k * b) ∧ (y = y₀ - k * a) := 
by
  sorry

end integer_solutions_of_linear_diophantine_eq_l1226_122628


namespace unique_solution_l1226_122614

theorem unique_solution (m n : ℕ) (h1 : n^4 ∣ 2 * m^5 - 1) (h2 : m^4 ∣ 2 * n^5 + 1) : m = 1 ∧ n = 1 :=
by
  sorry

end unique_solution_l1226_122614


namespace min_y_value_l1226_122650

noncomputable def y (x : ℝ) : ℝ :=
  (x - 6.5)^2 + (x - 5.9)^2 + (x - 6.0)^2 + (x - 6.7)^2 + (x - 4.5)^2

theorem min_y_value : 
  ∃ x : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x' : ℝ, abs (x' - 5.92) < δ → abs (y x' - y 5.92) < ε) :=
sorry

end min_y_value_l1226_122650


namespace min_value_of_x2_plus_y2_l1226_122656

open Real

theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 - 4 * x + 1 = 0) :
  x^2 + y^2 ≥ 7 - 4 * sqrt 3 := sorry

end min_value_of_x2_plus_y2_l1226_122656


namespace quadratic_unique_solution_k_neg_l1226_122696

theorem quadratic_unique_solution_k_neg (k : ℝ) :
  (∃ x : ℝ, 9 * x^2 + k * x + 36 = 0 ∧ ∀ y : ℝ, 9 * y^2 + k * y + 36 = 0 → y = x) →
  k = -36 :=
by
  sorry

end quadratic_unique_solution_k_neg_l1226_122696


namespace water_usage_l1226_122647

theorem water_usage (payment : ℝ) (usage : ℝ) : 
  payment = 7.2 → (usage ≤ 6 → payment = usage * 0.8) → (usage > 6 → payment = 4.8 + (usage - 6) * 1.2) → usage = 8 :=
by
  sorry

end water_usage_l1226_122647


namespace beetle_number_of_routes_128_l1226_122674

noncomputable def beetle_routes (A B : Type) : Nat :=
  let choices_at_first_step := 4
  let choices_at_second_step := 4
  let choices_at_third_step := 4
  let choices_at_final_step := 2
  choices_at_first_step * choices_at_second_step * choices_at_third_step * choices_at_final_step

theorem beetle_number_of_routes_128 (A B : Type) :
  beetle_routes A B = 128 :=
  by sorry

end beetle_number_of_routes_128_l1226_122674


namespace nancy_indian_food_freq_l1226_122646

-- Definitions based on the problem
def antacids_per_indian_day := 3
def antacids_per_mexican_day := 2
def antacids_per_other_day := 1
def mexican_per_week := 2
def total_antacids_per_month := 60
def weeks_per_month := 4
def days_per_week := 7

-- The proof statement
theorem nancy_indian_food_freq :
  ∃ (I : ℕ), (total_antacids_per_month = 
    weeks_per_month * (antacids_per_indian_day * I + 
    antacids_per_mexican_day * mexican_per_week + 
    antacids_per_other_day * (days_per_week - I - mexican_per_week))) ∧ I = 3 :=
by
  sorry

end nancy_indian_food_freq_l1226_122646


namespace constant_abs_difference_l1226_122644

variable (a : ℕ → ℝ)

-- Define the condition for the recurrence relation
def recurrence_relation : Prop := ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n

-- State the theorem
theorem constant_abs_difference (h : recurrence_relation a) : ∃ C : ℝ, ∀ n ≥ 2, |(a n)^2 - (a (n-1)) * (a (n+1))| = C :=
    sorry

end constant_abs_difference_l1226_122644


namespace sum_of_11378_and_121_is_odd_l1226_122637

theorem sum_of_11378_and_121_is_odd (h1 : Even 11378) (h2 : Odd 121) : Odd (11378 + 121) :=
by
  sorry

end sum_of_11378_and_121_is_odd_l1226_122637


namespace hyperbola_imaginary_axis_twice_real_axis_l1226_122676

theorem hyperbola_imaginary_axis_twice_real_axis (m : ℝ) : 
  (exists (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0), mx^2 + b^2 * y^2 = b^2) ∧
  (b = 2 * a) ∧ (m < 0) → 
  m = -1 / 4 := 
sorry

end hyperbola_imaginary_axis_twice_real_axis_l1226_122676


namespace pond_depth_l1226_122638

theorem pond_depth (L W V D : ℝ) (hL : L = 20) (hW : W = 10) (hV : V = 1000) :
    V = L * W * D ↔ D = 5 := 
by
  rw [hL, hW, hV]
  constructor
  · intro h1
    linarith
  · intro h2
    rw [h2]
    linarith

#check pond_depth

end pond_depth_l1226_122638


namespace circles_coincide_l1226_122665

-- Definitions for circle being inscribed in an angle and touching each other
structure Circle :=
  (radius : ℝ)
  (center: ℝ × ℝ)

def inscribed_in_angle (c : Circle) (θ: ℝ) : Prop :=
  -- Placeholder definition for circle inscribed in an angle
  sorry

def touches (c₁ c₂ : Circle) : Prop :=
  -- Placeholder definition for circles touching each other
  sorry

-- The angles of the triangle ABC are A, B, and C.
-- We are given the following conditions:
variables (A B C : ℝ) -- angles
variables (S1 S2 S3 S4 S5 S6 S7: Circle) -- circles

-- Circle S1 is inscribed in angle A
axiom S1_condition : inscribed_in_angle S1 A

-- Circle S2 is inscribed in angle B and touches S1 externally
axiom S2_condition : inscribed_in_angle S2 B ∧ touches S2 S1

-- Circle S3 is inscribed in angle C and touches S2
axiom S3_condition : inscribed_in_angle S3 C ∧ touches S3 S2

-- Circle S4 is inscribed in angle A and touches S3
axiom S4_condition : inscribed_in_angle S4 A ∧ touches S4 S3

-- We repeat this pattern up to circle S7
axiom S5_condition : inscribed_in_angle S5 B ∧ touches S5 S4
axiom S6_condition : inscribed_in_angle S6 C ∧ touches S6 S5
axiom S7_condition : inscribed_in_angle S7 A ∧ touches S7 S6

-- We need to prove the circle S7 coincides with S1
theorem circles_coincide : S7 = S1 :=
by
  -- Proof is skipped using sorry
  sorry

end circles_coincide_l1226_122665


namespace words_memorized_on_fourth_day_l1226_122694

-- Definitions for the conditions
def first_three_days_words (k : ℕ) : ℕ := 3 * k
def last_four_days_words (k : ℕ) : ℕ := 4 * k
def fourth_day_words (k : ℕ) (a : ℕ) : ℕ := a
def last_three_days_words (k : ℕ) (a : ℕ) : ℕ := last_four_days_words k - a

-- Problem Statement
theorem words_memorized_on_fourth_day {k a : ℕ} (h1 : first_three_days_words k + last_four_days_words k > 100)
    (h2 : first_three_days_words k * 6 = 5 * (4 * k - a))
    (h3 : 21 * (2 * k / 3) = 100) : 
    a = 10 :=
by 
  sorry

end words_memorized_on_fourth_day_l1226_122694


namespace sin_alpha_value_l1226_122627

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_value_l1226_122627


namespace exponent_multiplication_l1226_122626

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l1226_122626


namespace exists_infinitely_many_n_l1226_122642

def digit_sum (m : ℕ) : ℕ := sorry  -- Define the digit sum function

theorem exists_infinitely_many_n (S : ℕ → ℕ)
  (hS : ∀ m : ℕ, S m = digit_sum m) :
  ∃ᶠ n in at_top, S (3^n) ≥ S (3^(n + 1)) := 
sorry

end exists_infinitely_many_n_l1226_122642


namespace choose_integers_l1226_122621

def smallest_prime_divisor (n : ℕ) : ℕ := sorry
def number_of_divisors (n : ℕ) : ℕ := sorry

theorem choose_integers :
  ∃ (a : ℕ → ℕ), (∀ i, i < 2022 → a i < a (i + 1)) ∧
  (∀ k, 1 ≤ k ∧ k ≤ 2022 →
    number_of_divisors (a (k + 1) - a k - 1) > 2023^k ∧
    smallest_prime_divisor (a (k + 1) - a k) > 2023^k
  ) :=
sorry

end choose_integers_l1226_122621


namespace smallest_integer_y_l1226_122668

theorem smallest_integer_y : ∃ y : ℤ, (8:ℚ) / 11 < y / 17 ∧ ∀ z : ℤ, ((8:ℚ) / 11 < z / 17 → y ≤ z) :=
by
  sorry

end smallest_integer_y_l1226_122668


namespace find_unknown_number_l1226_122698

theorem find_unknown_number : 
  ∃ x : ℚ, (x * 7) / (10 * 17) = 10000 ∧ x = 1700000 / 7 :=
by
  sorry

end find_unknown_number_l1226_122698


namespace problem1_solution_problem2_solution_l1226_122613

noncomputable def problem1 : Real :=
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2)

noncomputable def problem2 : Real :=
  (2 * Real.sqrt 3 + Real.sqrt 6) * (2 * Real.sqrt 3 - Real.sqrt 6)

theorem problem1_solution : problem1 = 0 := by
  sorry

theorem problem2_solution : problem2 = 6 := by
  sorry

end problem1_solution_problem2_solution_l1226_122613


namespace smallest_positive_integer_ends_6996_l1226_122673

theorem smallest_positive_integer_ends_6996 :
  ∃ m : ℕ, (m % 4 = 0 ∧ m % 9 = 0 ∧ ∀ d ∈ m.digits 10, d = 6 ∨ d = 9 ∧ m.digits 10 ∩ {6, 9} ≠ ∅ ∧ m % 10000 = 6996) :=
sorry

end smallest_positive_integer_ends_6996_l1226_122673


namespace number_of_two_digit_integers_l1226_122699

def digits : Finset ℕ := {2, 4, 6, 7, 8}

theorem number_of_two_digit_integers : 
  (digits.card * (digits.card - 1)) = 20 := 
by
  sorry

end number_of_two_digit_integers_l1226_122699


namespace twice_midpoint_l1226_122671

open Complex

def z1 : ℂ := -7 + 5 * I
def z2 : ℂ := 9 - 11 * I

theorem twice_midpoint : 2 * ((z1 + z2) / 2) = 2 - 6 * I := 
by
  -- Sorry is used to skip the proof
  sorry

end twice_midpoint_l1226_122671


namespace smallest_rel_prime_greater_than_one_l1226_122690

theorem smallest_rel_prime_greater_than_one (n : ℕ) (h : n > 1) (h0: ∀ (m : ℕ), m > 1 ∧ Nat.gcd m 2100 = 1 → 11 ≤ m):
  Nat.gcd n 2100 = 1 → n = 11 :=
by
  -- Proof skipped
  sorry

end smallest_rel_prime_greater_than_one_l1226_122690


namespace divides_six_ab_l1226_122692

theorem divides_six_ab 
  (a b n : ℕ) 
  (hb : b < 10) 
  (hn : n > 3) 
  (h_eq : 2^n = 10 * a + b) : 
  6 ∣ (a * b) :=
sorry

end divides_six_ab_l1226_122692


namespace cos_double_angle_l1226_122645

variable (α : ℝ)
variable (h : Real.cos α = 2/3)

theorem cos_double_angle : Real.cos (2 * α) = -1/9 :=
  by
  sorry

end cos_double_angle_l1226_122645


namespace oysters_eaten_l1226_122678

-- Define the conditions in Lean
def Squido_oysters : ℕ := 200
def Crabby_oysters (Squido_oysters : ℕ) : ℕ := 2 * Squido_oysters

-- Statement to prove
theorem oysters_eaten (Squido_oysters Crabby_oysters : ℕ) (h1 : Crabby_oysters = 2 * Squido_oysters) : 
  Squido_oysters + Crabby_oysters = 600 :=
by
  sorry

end oysters_eaten_l1226_122678


namespace five_digit_palindromes_count_l1226_122610

def num_five_digit_palindromes : ℕ :=
  let choices_for_A := 9
  let choices_for_B := 10
  let choices_for_C := 10
  choices_for_A * choices_for_B * choices_for_C

theorem five_digit_palindromes_count : num_five_digit_palindromes = 900 :=
by
  unfold num_five_digit_palindromes
  sorry

end five_digit_palindromes_count_l1226_122610


namespace value_of_coins_l1226_122681

theorem value_of_coins (n d : ℕ) (hn : n + d = 30)
    (hv : 10 * n + 5 * d = 5 * n + 10 * d + 90) :
    300 - 5 * n = 180 := by
  sorry

end value_of_coins_l1226_122681


namespace hyperbola_standard_equation_equation_of_line_L_l1226_122695

open Real

noncomputable def hyperbola (x y : ℝ) : Prop :=
  y^2 - x^2 / 3 = 1

noncomputable def focus_on_y_axis := ∃ c : ℝ, c = 2

noncomputable def asymptote (x y : ℝ) : Prop := 
  y = sqrt 3 / 3 * x ∨ y = - sqrt 3 / 3 * x

noncomputable def point_A := (1, 1 / 2)

noncomputable def line_L (x y : ℝ) : Prop :=
  4 * x - 6 * y - 1 = 0

theorem hyperbola_standard_equation :
  ∃ (x y: ℝ), hyperbola x y :=
sorry

theorem equation_of_line_L :
  ∀ (x y : ℝ), point_A = (1, 1 / 2) ∧ line_L x y :=
sorry

end hyperbola_standard_equation_equation_of_line_L_l1226_122695


namespace smallest_positive_integer_in_form_l1226_122633

theorem smallest_positive_integer_in_form :
  ∃ (m n p : ℤ), 1234 * m + 56789 * n + 345 * p = 1 := sorry

end smallest_positive_integer_in_form_l1226_122633


namespace work_completion_days_l1226_122677

theorem work_completion_days (a b : ℕ) (h1 : a + b = 6) (h2 : a + b = 15 / 4) : a = 6 :=
by
  sorry

end work_completion_days_l1226_122677


namespace max_remainder_when_divided_by_8_l1226_122659

-- Define the problem: greatest possible remainder when apples divided by 8.
theorem max_remainder_when_divided_by_8 (n : ℕ) : ∃ r : ℕ, r < 8 ∧ r = 7 ∧ n % 8 = r := 
sorry

end max_remainder_when_divided_by_8_l1226_122659


namespace cost_price_of_apple_is_18_l1226_122632

noncomputable def cp (sp : ℝ) (loss_fraction : ℝ) : ℝ := sp / (1 - loss_fraction)

theorem cost_price_of_apple_is_18 :
  cp 15 (1/6) = 18 :=
by
  sorry

end cost_price_of_apple_is_18_l1226_122632


namespace p_eval_at_neg_one_l1226_122666

noncomputable def p (x : ℝ) : ℝ :=
  x^2 - 2*x + 9

theorem p_eval_at_neg_one : p (-1) = 12 := by
  sorry

end p_eval_at_neg_one_l1226_122666


namespace min_ab_correct_l1226_122682

noncomputable def min_ab (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + bc + ac = 2) : ℝ :=
  (6 - 2 * Real.sqrt 3) / 3

theorem min_ab_correct (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + bc + ac = 2) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) :
  a + b ≥ min_ab a b c h1 h2 :=
sorry

end min_ab_correct_l1226_122682


namespace no_fractional_solution_l1226_122655

theorem no_fractional_solution (x y : ℚ)
  (h₁ : ∃ m : ℤ, 13 * x + 4 * y = m)
  (h₂ : ∃ n : ℤ, 10 * x + 3 * y = n) :
  (∃ a b : ℤ, x ≠ a ∧ y ≠ b) → false :=
by {
  sorry
}

end no_fractional_solution_l1226_122655


namespace curve_properties_l1226_122658

noncomputable def curve (x y : ℝ) : Prop := Real.sqrt x + Real.sqrt y = 1

theorem curve_properties :
  curve 1 0 ∧ curve 0 1 ∧ curve (1/4) (1/4) ∧ 
  (∀ p : ℝ × ℝ, curve p.1 p.2 → curve p.2 p.1) :=
by
  sorry

end curve_properties_l1226_122658


namespace find_a_n_find_b_n_find_T_n_l1226_122670

-- definitions of sequences and common ratios
variable (a_n b_n : ℕ → ℕ)
variable (S_n T_n : ℕ → ℕ)
variable (q : ℝ)
variable (n : ℕ)

-- conditions
axiom a1 : a_n 1 = 1
axiom S3 : S_n 3 = 9
axiom b1 : b_n 1 = 1
axiom b3 : b_n 3 = 20
axiom q_pos : q > 0
axiom geo_seq : (∀ n, b_n n / a_n n = q ^ (n - 1))

-- goals to prove
theorem find_a_n : ∀ n, a_n n = 2 * n - 1 := 
by sorry

theorem find_b_n : ∀ n, b_n n = (2 * n - 1) * 2 ^ (n - 1) := 
by sorry

theorem find_T_n : ∀ n, T_n n = (2 * n - 3) * 2 ^ n + 3 :=
by sorry

end find_a_n_find_b_n_find_T_n_l1226_122670


namespace find_f_of_five_thirds_l1226_122640

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem find_f_of_five_thirds (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_fun : ∀ x : ℝ, f (1 + x) = f (-x))
  (h_val : f (-1 / 3) = 1 / 3) : 
  f (5 / 3) = 1 / 3 :=
  sorry

end find_f_of_five_thirds_l1226_122640


namespace seashells_needed_to_reach_target_l1226_122648

-- Definitions based on the conditions
def current_seashells : ℕ := 19
def target_seashells : ℕ := 25

-- Statement to prove
theorem seashells_needed_to_reach_target : target_seashells - current_seashells = 6 :=
by
  sorry

end seashells_needed_to_reach_target_l1226_122648


namespace banana_orange_equivalence_l1226_122693

/-- Given that 3/4 of 12 bananas are worth 9 oranges,
    prove that 1/3 of 9 bananas are worth 3 oranges. -/
theorem banana_orange_equivalence :
  (3 / 4) * 12 = 9 → (1 / 3) * 9 = 3 :=
by
  intro h
  have h1 : (9 : ℝ) = 9 := by sorry -- This is from the provided condition
  have h2 : 1 * 9 = 1 * 9 := by sorry -- Deducing from h1: 9 = 9
  have h3 : 9 = 9 := by sorry -- concluding 9 bananas = 9 oranges
  have h4 : (1 / 3) * 9 = 3 := by sorry -- 1/3 of 9
  exact h4

end banana_orange_equivalence_l1226_122693


namespace houses_before_boom_l1226_122602

theorem houses_before_boom (current_houses built_during_boom houses_before : ℕ) 
  (h1 : current_houses = 2000)
  (h2 : built_during_boom = 574)
  (h3 : current_houses = houses_before + built_during_boom) : 
  houses_before = 1426 := 
by
  -- Proof omitted
  sorry

end houses_before_boom_l1226_122602


namespace sum_of_legs_is_104_l1226_122636

theorem sum_of_legs_is_104 (x : ℕ) (h₁ : x^2 + (x + 2)^2 = 53^2) : x + (x + 2) = 104 := sorry

end sum_of_legs_is_104_l1226_122636


namespace sequence_G_51_l1226_122662

theorem sequence_G_51 :
  ∀ G : ℕ → ℚ, 
  (∀ n : ℕ, G (n + 1) = (3 * G n + 2) / 2) → 
  G 1 = 3 → 
  G 51 = (3^51 + 1) / 2 := by 
  sorry

end sequence_G_51_l1226_122662


namespace no_solution_bills_l1226_122661

theorem no_solution_bills (x y z : ℕ) (h1 : x + y + z = 10) (h2 : x + 3 * y + 5 * z = 25) : false :=
by
  sorry

end no_solution_bills_l1226_122661


namespace sixth_graders_more_than_seventh_l1226_122634

theorem sixth_graders_more_than_seventh (c_pencil : ℕ) (h_cents : c_pencil > 0)
    (h_cond : ∀ n : ℕ, n * c_pencil = 221 ∨ n * c_pencil = 286)
    (h_sixth_graders : 35 > 0) :
    ∃ n6 n7 : ℕ, n6 > n7 ∧ n6 - n7 = 5 :=
by
  sorry

end sixth_graders_more_than_seventh_l1226_122634


namespace pictures_per_day_calc_l1226_122664

def years : ℕ := 3
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

def number_of_cards : ℕ := total_spent / cost_per_card
def total_images : ℕ := number_of_cards * images_per_card
def days_in_year : ℕ := 365
def total_days : ℕ := years * days_in_year

theorem pictures_per_day_calc : 
  (total_images / total_days) = 10 := 
by
  sorry

end pictures_per_day_calc_l1226_122664


namespace weight_difference_l1226_122618

open Real

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h1 : (W_A + W_B + W_C) / 3 = 50)
  (h2 : W_A = 73)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 53)
  (h4 : (W_B + W_C + W_D + W_E) / 4 = 51) :
  W_E - W_D = 3 := 
sorry

end weight_difference_l1226_122618


namespace value_is_100_l1226_122629

theorem value_is_100 (number : ℕ) (h : number = 20) : 5 * number = 100 :=
by
  sorry

end value_is_100_l1226_122629


namespace min_x_plus_9y_l1226_122687

variable {x y : ℝ}

theorem min_x_plus_9y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / y = 1) : x + 9 * y ≥ 16 :=
  sorry

end min_x_plus_9y_l1226_122687


namespace triangle_inequality_l1226_122609
-- Import necessary libraries

-- Define the problem
theorem triangle_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (α β γ : ℝ) (h_alpha : α = 2 * Real.sqrt (b * c)) (h_beta : β = 2 * Real.sqrt (c * a)) (h_gamma : γ = 2 * Real.sqrt (a * b)) :
  (a / α) + (b / β) + (c / γ) ≥ (3 / 2) :=
by
  sorry

end triangle_inequality_l1226_122609


namespace solution_set_of_inequality_l1226_122643

theorem solution_set_of_inequality (x : ℝ) (h : |x - 1| < 1) : 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l1226_122643


namespace mutually_exclusive_any_two_l1226_122683

variables (A B C : Prop)
axiom all_not_defective : A
axiom all_defective : B
axiom not_all_defective : C

theorem mutually_exclusive_any_two :
  (¬(A ∧ B)) ∧ (¬(A ∧ C)) ∧ (¬(B ∧ C)) :=
sorry

end mutually_exclusive_any_two_l1226_122683


namespace men_in_second_group_l1226_122657

theorem men_in_second_group (M : ℕ) : 
    (18 * 20 = M * 24) → M = 15 :=
by
  intro h
  sorry

end men_in_second_group_l1226_122657


namespace total_number_of_edges_in_hexahedron_is_12_l1226_122622

-- Define a hexahedron
structure Hexahedron where
  face_count : Nat
  edges_per_face : Nat
  edge_sharing : Nat

-- Total edges calculation function
def total_edges (h : Hexahedron) : Nat := (h.face_count * h.edges_per_face) / h.edge_sharing

-- The specific hexahedron (cube) in question
def cube : Hexahedron := {
  face_count := 6,
  edges_per_face := 4,
  edge_sharing := 2
}

-- The theorem to prove the number of edges in a hexahedron
theorem total_number_of_edges_in_hexahedron_is_12 : total_edges cube = 12 := by
  sorry

end total_number_of_edges_in_hexahedron_is_12_l1226_122622


namespace joe_time_to_store_l1226_122600

theorem joe_time_to_store :
  ∀ (r_w : ℝ) (r_r : ℝ) (t_w t_r t_total : ℝ), 
   (r_r = 2 * r_w) → (t_w = 10) → (t_r = t_w / 2) → (t_total = t_w + t_r) → (t_total = 15) := 
by
  intros r_w r_r t_w t_r t_total hrw hrw_eq hr_tw hr_t_total
  sorry

end joe_time_to_store_l1226_122600


namespace repaved_inches_before_today_l1226_122651

theorem repaved_inches_before_today :
  let A := 4000
  let B := 3500
  let C := 2500
  let repaved_A := 0.70 * A
  let repaved_B := 0.60 * B
  let repaved_C := 0.80 * C
  let total_repaved_before := repaved_A + repaved_B + repaved_C
  let repaved_today := 950
  let new_total_repaved := total_repaved_before + repaved_today
  new_total_repaved - repaved_today = 6900 :=
by
  sorry

end repaved_inches_before_today_l1226_122651


namespace least_three_digit_eleven_heavy_l1226_122623

def isElevenHeavy (n : ℕ) : Prop :=
  n % 11 > 6

theorem least_three_digit_eleven_heavy : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ isElevenHeavy n ∧ ∀ m : ℕ, (m >= 100 ∧ m < 1000 ∧ isElevenHeavy m) → n ≤ m :=
sorry

end least_three_digit_eleven_heavy_l1226_122623


namespace regression_line_zero_corr_l1226_122612

-- Definitions based on conditions
variables {X Y : Type}
variables [LinearOrder X] [LinearOrder Y]
variables {f : X → Y}  -- representing the regression line

-- Condition: Regression coefficient b = 0
def regression_coefficient_zero (b : ℝ) : Prop := b = 0

-- Definition of correlation coefficient; here symbolically represented since full derivation requires in-depth statistics definitions
def correlation_coefficient (r : ℝ) : ℝ := r

-- The mathematical goal to prove
theorem regression_line_zero_corr {b r : ℝ} 
  (hb : regression_coefficient_zero b) : correlation_coefficient r = 0 := 
by
  sorry

end regression_line_zero_corr_l1226_122612


namespace infinite_natural_solutions_l1226_122697

theorem infinite_natural_solutions : ∀ n : ℕ, ∃ x y z : ℕ, (x + y + z)^2 + 2 * (x + y + z) = 5 * (x * y + y * z + z * x) :=
by
  sorry

end infinite_natural_solutions_l1226_122697


namespace curve_crosses_itself_and_point_of_crossing_l1226_122691

-- Define the function for x and y
def x (t : ℝ) : ℝ := t^2 + 1
def y (t : ℝ) : ℝ := t^4 - 9 * t^2 + 6

-- Definition of the curve crossing itself and the point of crossing
theorem curve_crosses_itself_and_point_of_crossing :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁ = 10 ∧ y t₁ = 6) :=
by
  sorry

end curve_crosses_itself_and_point_of_crossing_l1226_122691


namespace reciprocal_neg_half_l1226_122663

theorem reciprocal_neg_half : 1 / (-1 / 2 : ℝ) = (-2 : ℝ) :=
by
  sorry

end reciprocal_neg_half_l1226_122663


namespace big_bea_bananas_l1226_122630

theorem big_bea_bananas :
  ∃ (b : ℕ), (b + (b + 8) + (b + 16) + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 196) ∧ (b + 48 = 52) := by
  sorry

end big_bea_bananas_l1226_122630
