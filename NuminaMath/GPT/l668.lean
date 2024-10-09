import Mathlib

namespace sequence_odd_for_all_n_greater_than_1_l668_66838

theorem sequence_odd_for_all_n_greater_than_1 (a : ℕ → ℤ) :
  (a 1 = 2) →
  (a 2 = 7) →
  (∀ n, 2 ≤ n → (-1/2 : ℚ) < (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ∧ (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ≤ (1/2 : ℚ)) →
  ∀ n, 1 < n → Odd (a n) := 
sorry

end sequence_odd_for_all_n_greater_than_1_l668_66838


namespace walk_to_school_l668_66822

theorem walk_to_school (W P : ℕ) (h1 : W + P = 41) (h2 : W = P + 3) : W = 22 :=
by 
  sorry

end walk_to_school_l668_66822


namespace smallest_n_exists_l668_66809

theorem smallest_n_exists (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : 8 / 15 < n / (n + k)) (h4 : n / (n + k) < 7 / 13) : 
  n = 15 :=
  sorry

end smallest_n_exists_l668_66809


namespace combined_length_of_trains_l668_66810

theorem combined_length_of_trains
  (speed_A_kmph : ℕ) (speed_B_kmph : ℕ)
  (platform_length : ℕ) (time_A_sec : ℕ) (time_B_sec : ℕ)
  (h_speed_A : speed_A_kmph = 72) (h_speed_B : speed_B_kmph = 90)
  (h_platform_length : platform_length = 300)
  (h_time_A : time_A_sec = 30) (h_time_B : time_B_sec = 24) :
  let speed_A_ms := speed_A_kmph * 5 / 18
  let speed_B_ms := speed_B_kmph * 5 / 18
  let distance_A := speed_A_ms * time_A_sec
  let distance_B := speed_B_ms * time_B_sec
  let length_A := distance_A - platform_length
  let length_B := distance_B - platform_length
  length_A + length_B = 600 :=
by
  sorry

end combined_length_of_trains_l668_66810


namespace binomial_expansion_conditions_l668_66852

noncomputable def binomial_expansion (a b : ℝ) (x y : ℝ) (n : ℕ) : ℝ :=
(1 + a*x + b*y)^n

theorem binomial_expansion_conditions
  (a b : ℝ) (n : ℕ) 
  (h1 : (1 + b)^n = 243)
  (h2 : (1 + |a|)^n = 32) :
  a = 1 ∧ b = 2 ∧ n = 5 := by
  sorry

end binomial_expansion_conditions_l668_66852


namespace greatest_int_less_than_neg_17_div_3_l668_66883

theorem greatest_int_less_than_neg_17_div_3 : 
  ∀ (x : ℚ), x = -17/3 → ⌊x⌋ = -6 :=
by
  sorry

end greatest_int_less_than_neg_17_div_3_l668_66883


namespace sum_first_110_terms_l668_66813

noncomputable def sum_arithmetic (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem sum_first_110_terms (a1 d : ℚ) (h1 : sum_arithmetic a1 d 10 = 100)
  (h2 : sum_arithmetic a1 d 100 = 10) : sum_arithmetic a1 d 110 = -110 := by
  sorry

end sum_first_110_terms_l668_66813


namespace calculate_f_g_f_l668_66899

def f (x : ℤ) : ℤ := 5 * x + 5
def g (x : ℤ) : ℤ := 6 * x + 5

theorem calculate_f_g_f : f (g (f 3)) = 630 := by
  sorry

end calculate_f_g_f_l668_66899


namespace weight_shaina_receives_l668_66825

namespace ChocolateProblem

-- Definitions based on conditions
def total_chocolate : ℚ := 60 / 7
def piles : ℚ := 5
def weight_per_pile : ℚ := total_chocolate / piles
def shaina_piles : ℚ := 2

-- Proposition to represent the question and correct answer
theorem weight_shaina_receives : 
  (weight_per_pile * shaina_piles) = 24 / 7 := 
by
  sorry

end ChocolateProblem

end weight_shaina_receives_l668_66825


namespace find_a8_l668_66880

variable (a : ℕ → ℝ)
variable (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a8 
  (hq : is_geometric_sequence a q)
  (h1 : a 1 * a 3 = 4)
  (h2 : a 9 = 256) : 
  a 8 = 128 ∨ a 8 = -128 :=
by
  sorry

end find_a8_l668_66880


namespace repeating_decimal_ratio_eq_4_l668_66832

-- Definitions for repeating decimals
def rep_dec_36 := 0.36 -- 0.\overline{36}
def rep_dec_09 := 0.09 -- 0.\overline{09}

-- Lean 4 statement of proof problem
theorem repeating_decimal_ratio_eq_4 :
  (rep_dec_36 / rep_dec_09) = 4 :=
sorry

end repeating_decimal_ratio_eq_4_l668_66832


namespace no_solution_for_n_eq_neg1_l668_66896

theorem no_solution_for_n_eq_neg1 (x y z : ℝ) : ¬ (∃ x y z, (-1) * x^2 + y = 2 ∧ (-1) * y^2 + z = 2 ∧ (-1) * z^2 + x = 2) :=
by
  sorry

end no_solution_for_n_eq_neg1_l668_66896


namespace trains_cross_time_l668_66803

def speed_in_m_per_s (speed_in_km_per_hr : Float) : Float :=
  (speed_in_km_per_hr * 1000) / 3600

def relative_speed (speed1 : Float) (speed2 : Float) : Float :=
  speed1 + speed2

def total_distance (length1 : Float) (length2 : Float) : Float :=
  length1 + length2

def time_to_cross (total_dist : Float) (relative_spd : Float) : Float :=
  total_dist / relative_spd

theorem trains_cross_time 
  (length_train1 : Float := 270)
  (speed_train1 : Float := 120)
  (length_train2 : Float := 230.04)
  (speed_train2 : Float := 80) :
  time_to_cross (total_distance length_train1 length_train2) 
                (relative_speed (speed_in_m_per_s speed_train1) 
                                (speed_in_m_per_s speed_train2)) = 9 := 
by
  sorry

end trains_cross_time_l668_66803


namespace find_m_n_l668_66807

theorem find_m_n (m n : ℕ) (h : (1/5 : ℝ)^m * (1/4 : ℝ)^n = 1 / (10 : ℝ)^4) : m = 4 ∧ n = 2 :=
sorry

end find_m_n_l668_66807


namespace point_on_transformed_graph_l668_66811

theorem point_on_transformed_graph 
  (f : ℝ → ℝ)
  (h1 : f 12 = 5)
  (x y : ℝ)
  (h2 : 1.5 * y = (f (3 * x) + 3) / 3)
  (point_x : x = 4)
  (point_y : y = 16 / 9) 
  : x + y = 52 / 9 :=
by
  sorry

end point_on_transformed_graph_l668_66811


namespace oranges_per_pack_correct_l668_66881

-- Definitions for the conditions.
def num_trees : Nat := 10
def oranges_per_tree_per_day : Nat := 12
def price_per_pack : Nat := 2
def total_earnings : Nat := 840
def weeks : Nat := 3
def days_per_week : Nat := 7

-- Theorem statement:
theorem oranges_per_pack_correct :
  let oranges_per_day := num_trees * oranges_per_tree_per_day
  let total_days := weeks * days_per_week
  let total_oranges := oranges_per_day * total_days
  let num_packs := total_earnings / price_per_pack
  total_oranges / num_packs = 6 :=
by
  sorry

end oranges_per_pack_correct_l668_66881


namespace angle_D_measure_l668_66844

theorem angle_D_measure 
  (A B C D : Type)
  (angleA : ℝ)
  (angleB : ℝ)
  (angleC : ℝ)
  (angleD : ℝ)
  (BD_bisector : ℝ → ℝ) :
  angleA = 85 ∧ angleB = 50 ∧ angleC = 25 ∧ BD_bisector angleB = 25 →
  angleD = 130 :=
by
  intro h
  have hA := h.1
  have hB := h.2.1
  have hC := h.2.2.1
  have hBD := h.2.2.2
  sorry

end angle_D_measure_l668_66844


namespace Dave_needs_31_gallons_l668_66850

noncomputable def numberOfGallons (numberOfTanks : ℕ) (height : ℝ) (diameter : ℝ) (coveragePerGallon : ℝ) : ℕ :=
  let radius := diameter / 2
  let lateral_surface_area := 2 * Real.pi * radius * height
  let total_surface_area := lateral_surface_area * numberOfTanks
  let gallons_needed := total_surface_area / coveragePerGallon
  Nat.ceil gallons_needed

theorem Dave_needs_31_gallons :
  numberOfGallons 20 24 8 400 = 31 :=
by
  sorry

end Dave_needs_31_gallons_l668_66850


namespace factorize_expression_l668_66817

theorem factorize_expression (y a : ℝ) : 
  3 * y * a ^ 2 - 6 * y * a + 3 * y = 3 * y * (a - 1) ^ 2 :=
by
  sorry

end factorize_expression_l668_66817


namespace find_n_l668_66895

variable (x n : ℝ)

-- Definitions
def positive (x : ℝ) : Prop := x > 0
def equation (x n : ℝ) : Prop := x / n + x / 25 = 0.06 * x

-- Theorem statement
theorem find_n (h1 : positive x) (h2 : equation x n) : n = 50 :=
sorry

end find_n_l668_66895


namespace sqrt_expr_evaluation_l668_66826

theorem sqrt_expr_evaluation : 
  (Real.sqrt 24) - 3 * (Real.sqrt (1 / 6)) + (Real.sqrt 6) = (5 * Real.sqrt 6) / 2 :=
by
  sorry

end sqrt_expr_evaluation_l668_66826


namespace value_of_a_b_c_l668_66833

noncomputable def absolute_value (x : ℤ) : ℤ := abs x

theorem value_of_a_b_c (a b c : ℤ)
  (ha : absolute_value a = 1)
  (hb : absolute_value b = 2)
  (hc : absolute_value c = 3)
  (h : a > b ∧ b > c) :
  a + b - c = 2 ∨ a + b - c = 0 :=
by
  sorry

end value_of_a_b_c_l668_66833


namespace quadratic_expression_neg_for_all_x_l668_66834

theorem quadratic_expression_neg_for_all_x (m : ℝ) :
  (∀ x : ℝ, m*x^2 + (m-1)*x + (m-1) < 0) ↔ m < -1/3 :=
sorry

end quadratic_expression_neg_for_all_x_l668_66834


namespace percentage_concentration_acid_l668_66806

-- Definitions based on the given conditions
def volume_acid : ℝ := 1.6
def total_volume : ℝ := 8.0

-- Lean statement to prove the percentage concentration is 20%
theorem percentage_concentration_acid : (volume_acid / total_volume) * 100 = 20 := by
  sorry

end percentage_concentration_acid_l668_66806


namespace region_of_inequality_l668_66872

theorem region_of_inequality (x y : ℝ) : (x + y - 6 < 0) → y < -x + 6 := by
  sorry

end region_of_inequality_l668_66872


namespace distance_center_to_plane_l668_66882

theorem distance_center_to_plane (r : ℝ) (a b : ℝ) (h : a ^ 2 + b ^ 2 = 10 ^ 2) (d : ℝ) : 
  r = 13 → a = 6 → b = 8 → d = 12 := 
by 
  sorry

end distance_center_to_plane_l668_66882


namespace roots_of_equation_l668_66875

theorem roots_of_equation:
  ∀ x : ℝ, (x - 2) * (x - 3) = x - 2 → x = 2 ∨ x = 4 := by
  sorry

end roots_of_equation_l668_66875


namespace gcd_45_75_l668_66853

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l668_66853


namespace solution_set_a_eq_1_find_a_min_value_3_l668_66864

open Real

noncomputable def f (x a : ℝ) := 2 * abs (x + 1) + abs (x - a)

-- The statement for the first question
theorem solution_set_a_eq_1 (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -2 ∨ x ≥ (4 / 3) := 
by sorry

-- The statement for the second question
theorem find_a_min_value_3 (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 3) ∧ (∃ x : ℝ, f x a = 3) ↔ a = 2 ∨ a = -4 := 
by sorry

end solution_set_a_eq_1_find_a_min_value_3_l668_66864


namespace math_problem_l668_66866

variables {x y z a b c : ℝ}

theorem math_problem
  (h₁ : x / a + y / b + z / c = 4)
  (h₂ : a / x + b / y + c / z = 2) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end math_problem_l668_66866


namespace exists_infinitely_many_gcd_condition_l668_66857

theorem exists_infinitely_many_gcd_condition (a : ℕ → ℕ) (h : ∀ n : ℕ, ∃ m : ℕ, a m = n) :
  ∃ᶠ i in at_top, Nat.gcd (a i) (a (i + 1)) ≤ (3 * i) / 4 :=
sorry

end exists_infinitely_many_gcd_condition_l668_66857


namespace triangle_properties_l668_66889

theorem triangle_properties (a b c : ℝ) (h1 : a / b = 5 / 12) (h2 : b / c = 12 / 13) (h3 : a + b + c = 60) :
  (a^2 + b^2 = c^2) ∧ ((1 / 2) * a * b > 100) :=
by
  sorry

end triangle_properties_l668_66889


namespace total_square_miles_of_plains_l668_66863

-- Defining conditions
def region_east_of_b : ℕ := 200
def region_east_of_a : ℕ := region_east_of_b - 50

-- To test this statement in Lean 4
theorem total_square_miles_of_plains : region_east_of_a + region_east_of_b = 350 := by
  sorry

end total_square_miles_of_plains_l668_66863


namespace quarters_spent_l668_66848

variable (q_initial q_left q_spent : ℕ)

theorem quarters_spent (h1 : q_initial = 11) (h2 : q_left = 7) : q_spent = q_initial - q_left ∧ q_spent = 4 :=
by
  sorry

end quarters_spent_l668_66848


namespace split_terms_addition_l668_66815

theorem split_terms_addition : 
  (-2017 - (2/3)) + (2016 + (3/4)) + (-2015 - (5/6)) + (16 + (1/2)) = -2000 - (1/4) :=
by
  sorry

end split_terms_addition_l668_66815


namespace ribbon_initial_amount_l668_66893

theorem ribbon_initial_amount (x : ℕ) (gift_count : ℕ) (ribbon_per_gift : ℕ) (ribbon_left : ℕ)
  (H1 : ribbon_per_gift = 2) (H2 : gift_count = 6) (H3 : ribbon_left = 6)
  (H4 : x = gift_count * ribbon_per_gift + ribbon_left) : x = 18 :=
by
  rw [H1, H2, H3] at H4
  exact H4

end ribbon_initial_amount_l668_66893


namespace values_of_2n_plus_m_l668_66845

theorem values_of_2n_plus_m (n m : ℤ) (h1 : 3 * n - m ≤ 4) (h2 : n + m ≥ 27) (h3 : 3 * m - 2 * n ≤ 45) 
  (h4 : n = 8) (h5 : m = 20) : 2 * n + m = 36 := by
  sorry

end values_of_2n_plus_m_l668_66845


namespace factorize_expression_l668_66851

theorem factorize_expression (a m n : ℝ) : a * m^2 - 2 * a * m * n + a * n^2 = a * (m - n)^2 :=
by
  sorry

end factorize_expression_l668_66851


namespace problem_a_plus_b_equals_10_l668_66824

theorem problem_a_plus_b_equals_10 (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) 
  (h_equation : 3 * a + 4 * b = 10 * a + b) : a + b = 10 :=
by {
  sorry
}

end problem_a_plus_b_equals_10_l668_66824


namespace vector_solution_l668_66886

theorem vector_solution :
  let u := -6 / 41
  let v := -46 / 41
  let vec1 := (⟨3, -2⟩: ℝ × ℝ)
  let vec2 := (⟨5, -7⟩: ℝ × ℝ)
  let vec3 := (⟨0, 3⟩: ℝ × ℝ)
  let vec4 := (⟨-3, 4⟩: ℝ × ℝ)
  (vec1 + u • vec2 = vec3 + v • vec4) := by
  sorry

end vector_solution_l668_66886


namespace min_f_in_interval_l668_66828

open Real

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) - 2 * sqrt 3 * sin (ω * x / 2) ^ 2 + sqrt 3

theorem min_f_in_interval (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 <= x ∧ x <= π / 2 → f 1 x >= f 1 (π / 3)) :=
by sorry

end min_f_in_interval_l668_66828


namespace sum_of_solutions_l668_66812

def equation (x : ℝ) : Prop := (6 * x) / 30 = 8 / x

theorem sum_of_solutions : ∀ x1 x2 : ℝ, equation x1 → equation x2 → x1 + x2 = 0 := by
  sorry

end sum_of_solutions_l668_66812


namespace hannah_monday_run_l668_66888

-- Definitions of the conditions
def ran_on_wednesday : ℕ := 4816
def ran_on_friday : ℕ := 2095
def extra_on_monday : ℕ := 2089

-- Translations to set the total combined distance and the distance ran on Monday
def combined_distance := ran_on_wednesday + ran_on_friday
def ran_on_monday := combined_distance + extra_on_monday

-- A statement to show she ran 9 kilometers on Monday
theorem hannah_monday_run :
  ran_on_monday = 9000 / 1000 * 1000 := sorry

end hannah_monday_run_l668_66888


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l668_66841

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l668_66841


namespace find_a_l668_66831

theorem find_a 
  (a b c : ℚ) 
  (h1 : a + b = c) 
  (h2 : b + c + 2 * b = 11) 
  (h3 : c = 7) :
  a = 17 / 3 :=
by
  sorry

end find_a_l668_66831


namespace average_weight_a_b_l668_66802

theorem average_weight_a_b (A B C : ℝ) 
    (h1 : (A + B + C) / 3 = 45) 
    (h2 : (B + C) / 2 = 44) 
    (h3 : B = 33) : 
    (A + B) / 2 = 40 := 
by 
  sorry

end average_weight_a_b_l668_66802


namespace sasha_salt_factor_l668_66894

theorem sasha_salt_factor (x y : ℝ) : 
  (y = 2 * x) →
  (x + y = 2 * x + y / 2) →
  (3 * x / (2 * x) = 1.5) :=
by
  intros h₁ h₂
  sorry

end sasha_salt_factor_l668_66894


namespace solve_quadratic_eq_solve_cubic_eq_l668_66865

-- Problem 1: 4x^2 - 9 = 0 implies x = ± 3/2
theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2 :=
by sorry

-- Problem 2: 64 * (x + 1)^3 = -125 implies x = -9/4
theorem solve_cubic_eq (x : ℝ) : 64 * (x + 1)^3 = -125 ↔ x = -9/4 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l668_66865


namespace studentC_spending_l668_66846

-- Definitions based on the problem conditions

-- Prices of Type A and Type B notebooks, respectively
variables (x y : ℝ)

-- Number of each type of notebook bought by Student A
def studentA : Prop := x + y = 3

-- Number of Type A notebooks bought by Student B
variables (a : ℕ)

-- Total cost and number of notebooks bought by Student B
def studentB : Prop := (x * a + y * (8 - a) = 11)

-- Constraints on the number of Type A and B notebooks bought by Student C
def studentC_notebooks : Prop := ∃ b : ℕ, b = 8 - a ∧ b = a

-- The total amount spent by Student C
def studentC_cost : ℝ := (8 - a) * x + a * y

-- The statement asserting the cost is 13 yuan
theorem studentC_spending (x y : ℝ) (a : ℕ) (hA : studentA x y) (hB : studentB x y a) (hC : studentC_notebooks a) : studentC_cost x y a = 13 := sorry

end studentC_spending_l668_66846


namespace expression_evaluation_l668_66861

theorem expression_evaluation :
  10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 :=
by
  sorry

end expression_evaluation_l668_66861


namespace equal_angles_not_necessarily_vertical_l668_66871

-- Define what it means for angles to be vertical
def is_vertical_angle (a b : ℝ) : Prop :=
∃ l1 l2 : ℝ, a = 180 - b ∧ (l1 + l2 == 180 ∨ l1 == 0 ∨ l2 == 0)

-- Define what it means for angles to be equal
def are_equal_angles (a b : ℝ) : Prop := a = b

-- Proposition to be proved
theorem equal_angles_not_necessarily_vertical (a b : ℝ) (h : are_equal_angles a b) : ¬ is_vertical_angle a b :=
by
  sorry

end equal_angles_not_necessarily_vertical_l668_66871


namespace hyperbola_asymptote_passing_through_point_l668_66879

theorem hyperbola_asymptote_passing_through_point (a : ℝ) (h_pos : a > 0) :
  (∃ m : ℝ, ∃ b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ (x, y) = (2, 1) ∧ m = 2 / a) → a = 4 :=
by
  sorry

end hyperbola_asymptote_passing_through_point_l668_66879


namespace divisibility_of_n_l668_66854

theorem divisibility_of_n
  (n : ℕ) (n_gt_1 : n > 1)
  (h : n ∣ (6^n - 1)) : 5 ∣ n :=
by
  sorry

end divisibility_of_n_l668_66854


namespace max_sum_of_integer_pairs_l668_66800

theorem max_sum_of_integer_pairs (x y : ℤ) (h : (x-1)^2 + (y+2)^2 = 36) : 
  max (x + y) = 5 :=
sorry

end max_sum_of_integer_pairs_l668_66800


namespace hot_dog_cost_l668_66801

variables (h d : ℝ)

theorem hot_dog_cost :
  (3 * h + 4 * d = 10) →
  (2 * h + 3 * d = 7) →
  d = 1 :=
by
  intros h_eq d_eq
  -- Proof skipped
  sorry

end hot_dog_cost_l668_66801


namespace rectangle_shorter_side_length_l668_66897

theorem rectangle_shorter_side_length (rope_length : ℕ) (long_side : ℕ) : 
  rope_length = 100 → long_side = 28 → 
  ∃ short_side : ℕ, (2 * long_side + 2 * short_side = rope_length) ∧ short_side = 22 :=
by
  sorry

end rectangle_shorter_side_length_l668_66897


namespace problem_solution_l668_66898

def f (x y : ℝ) : ℝ :=
  (x - y) * x * y * (x + y) * (2 * x^2 - 5 * x * y + 2 * y^2)

theorem problem_solution :
  (∀ x y : ℝ, f x y + f y x = 0) ∧
  (∀ x y : ℝ, f x (x + y) + f y (x + y) = 0) :=
by
  sorry

end problem_solution_l668_66898


namespace tan_pi_minus_alpha_l668_66849

theorem tan_pi_minus_alpha (α : ℝ) (h : 3 * Real.sin α = Real.cos α) : Real.tan (π - α) = -1 / 3 :=
by
  sorry

end tan_pi_minus_alpha_l668_66849


namespace determine_x_l668_66862

noncomputable def proof_problem (x : ℝ) (y : ℝ) : Prop :=
  y > 0 → 2 * (x * y^2 + x^2 * y + 2 * y^2 + 2 * x * y) / (x + y) > 3 * x^2 * y

theorem determine_x (x : ℝ) : 
  (∀ (y : ℝ), y > 0 → proof_problem x y) ↔ 0 ≤ x ∧ x < (1 + Real.sqrt 13) / 3 := 
sorry

end determine_x_l668_66862


namespace simplify_expression_l668_66840

noncomputable def y := 
  Real.cos (2 * Real.pi / 15) + 
  Real.cos (4 * Real.pi / 15) + 
  Real.cos (8 * Real.pi / 15) + 
  Real.cos (14 * Real.pi / 15)

theorem simplify_expression : 
  y = (-1 + Real.sqrt 61) / 4 := 
sorry

end simplify_expression_l668_66840


namespace arithmetic_seq_sum_2013_l668_66858

noncomputable def a1 : ℤ := -2013
noncomputable def S (n d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_seq_sum_2013 :
  ∃ d : ℤ, (S 12 d / 12 - S 10 d / 10 = 2) → S 2013 d = -2013 :=
by
  sorry

end arithmetic_seq_sum_2013_l668_66858


namespace greatest_three_digit_number_l668_66808

theorem greatest_three_digit_number : ∃ n : ℕ, n < 1000 ∧ n >= 100 ∧ (n + 1) % 8 = 0 ∧ (n - 4) % 7 = 0 ∧ n = 967 :=
by
  sorry

end greatest_three_digit_number_l668_66808


namespace exists_K_p_l668_66890

noncomputable def constant_K_p (p : ℝ) (hp : p > 1) : ℝ :=
  (p * p) / (p - 1)

theorem exists_K_p (p : ℝ) (hp : p > 1) :
  ∃ K_p > 0, ∀ x y : ℝ, |x|^p + |y|^p = 2 → (x - y)^2 ≤ K_p * (4 - (x + y)^2) :=
by
  use constant_K_p p hp
  sorry

end exists_K_p_l668_66890


namespace sum_of_roots_l668_66819

theorem sum_of_roots (x₁ x₂ : ℝ) (h1 : x₁^2 = 2 * x₁ + 1) (h2 : x₂^2 = 2 * x₂ + 1) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l668_66819


namespace cost_of_adult_ticket_l668_66847

theorem cost_of_adult_ticket (A : ℝ) (H1 : ∀ (cost_child : ℝ), cost_child = 7) 
                             (H2 : ∀ (num_adults : ℝ), num_adults = 2) 
                             (H3 : ∀ (num_children : ℝ), num_children = 2) 
                             (H4 : ∀ (total_cost : ℝ), total_cost = 58) :
    A = 22 :=
by
  -- You can assume variables for children's cost, number of adults, and number of children
  let cost_child := 7
  let num_adults := 2
  let num_children := 2
  let total_cost := 58
  
  -- Formalize the conditions given
  have H_children_cost : num_children * cost_child = 14 := by simp [cost_child, num_children]
  
  -- Establish the total cost equation
  have H_total_equation : num_adults * A + num_children * cost_child = total_cost := 
    by sorry  -- (Total_equation_proof)
  
  -- Solve for A
  sorry  -- Proof step

end cost_of_adult_ticket_l668_66847


namespace paco_initial_sweet_cookies_l668_66869

theorem paco_initial_sweet_cookies
    (x : ℕ)  -- Paco's initial number of sweet cookies
    (eaten_sweet : ℕ)  -- number of sweet cookies Paco ate
    (left_sweet : ℕ)  -- number of sweet cookies Paco had left
    (h1 : eaten_sweet = 15)  -- Paco ate 15 sweet cookies
    (h2 : left_sweet = 19)  -- Paco had 19 sweet cookies left
    (h3 : x - eaten_sweet = left_sweet)  -- After eating, Paco had 19 sweet cookies left
    : x = 34 :=  -- Paco initially had 34 sweet cookies
sorry

end paco_initial_sweet_cookies_l668_66869


namespace minor_axis_length_l668_66856

theorem minor_axis_length (h : ∀ x y : ℝ, x^2 / 4 + y^2 / 36 = 1) : 
  ∃ b : ℝ, b = 2 ∧ 2 * b = 4 :=
by
  sorry

end minor_axis_length_l668_66856


namespace find_b_and_c_l668_66885

variable (U : Set ℝ) -- Define the universal set U
variable (A : Set ℝ) -- Define the set A
variables (b c : ℝ) -- Variables for coefficients

-- Conditions that U = {2, 3, 5} and A = { x | x^2 + bx + c = 0 }
def cond_universal_set := U = {2, 3, 5}
def cond_set_A := A = { x | x^2 + b * x + c = 0 }

-- Condition for the complement of A w.r.t U being {2}
def cond_complement := (U \ A) = {2}

-- The statement to be proved
theorem find_b_and_c : 
  cond_universal_set U →
  cond_set_A A b c →
  cond_complement U A →
  b = -8 ∧ c = 15 :=
by
  intros
  sorry

end find_b_and_c_l668_66885


namespace range_of_x_for_odd_monotonic_function_l668_66818

theorem range_of_x_for_odd_monotonic_function 
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_increasing_on_R : ∀ x y : ℝ, x ≤ y → f x ≤ f y) :
  ∀ x : ℝ, (0 < x) → ( (|f (Real.log x) - f (Real.log (1 / x))| / 2) < f 1 ) → (Real.exp (-1) < x ∧ x < Real.exp 1) := 
by
  sorry

end range_of_x_for_odd_monotonic_function_l668_66818


namespace total_length_of_rubber_pen_pencil_l668_66814

variable (rubber pen pencil : ℕ)

theorem total_length_of_rubber_pen_pencil 
  (h1 : pen = rubber + 3)
  (h2 : pen = pencil - 2)
  (h3 : pencil = 12) : rubber + pen + pencil = 29 := by
  sorry

end total_length_of_rubber_pen_pencil_l668_66814


namespace correct_minutes_added_l668_66821

theorem correct_minutes_added :
  let time_lost_per_day : ℚ := 3 + 1/4
  let start_time := 1 -- in P.M. on March 15
  let end_time := 3 -- in P.M. on March 22
  let total_days := 7 -- days from March 15 to March 22
  let extra_hours := 2 -- hours on March 22 from 1 P.M. to 3 P.M.
  let total_hours := (total_days * 24) + extra_hours
  let time_lost_per_minute := time_lost_per_day / (24 * 60)
  let total_time_lost := total_hours * time_lost_per_minute
  let total_time_lost_minutes := total_time_lost * 60
  n = total_time_lost_minutes 
→ n = 221 / 96 := 
sorry

end correct_minutes_added_l668_66821


namespace sqrt_pow_mul_l668_66870

theorem sqrt_pow_mul (a b : ℝ) : (a = 3) → (b = 5) → (Real.sqrt (a^2 * b^6) = 375) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end sqrt_pow_mul_l668_66870


namespace line_always_passes_through_fixed_point_l668_66837

theorem line_always_passes_through_fixed_point :
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = m * x + 2 * m + 1) ∧ (x = -2) ∧ (y = 1) :=
by
  sorry

end line_always_passes_through_fixed_point_l668_66837


namespace range_of_b_distance_when_b_eq_one_l668_66867

-- Definitions for conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def line (x y b : ℝ) : Prop := y = x + b
def intersect (x y b : ℝ) : Prop := ellipse x y ∧ line x y b

-- Prove the range of b for which there are two distinct intersection points
theorem range_of_b (b : ℝ) : (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ intersect x1 y1 b ∧ intersect x2 y2 b) ↔ (-Real.sqrt 3 < b ∧ b < Real.sqrt 3) :=
by sorry

-- Prove the distance between points A and B when b = 1
theorem distance_when_b_eq_one : 
  ∃ x1 y1 x2 y2, intersect x1 y1 1 ∧ intersect x2 y2 1 ∧ Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * Real.sqrt 2 / 3 :=
by sorry

end range_of_b_distance_when_b_eq_one_l668_66867


namespace part1_part2_l668_66843

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem part1 (x : ℝ) : (f x)^2 - (g x)^2 = -4 :=
by sorry

theorem part2 (x y : ℝ) (h1 : f x * f y = 4) (h2 : g x * g y = 8) : 
  g (x + y) / g (x - y) = 3 :=
by sorry

end part1_part2_l668_66843


namespace volume_at_target_temperature_l668_66855

-- Volume expansion relationship
def volume_change_per_degree_rise (ΔT V_real : ℝ) : Prop :=
  ΔT = 2 ∧ V_real = 3

-- Initial conditions
def initial_conditions (V_initial T_initial : ℝ) : Prop :=
  V_initial = 36 ∧ T_initial = 30

-- Target temperature
def target_temperature (T_target : ℝ) : Prop :=
  T_target = 20

-- Theorem stating the volume at the target temperature
theorem volume_at_target_temperature (ΔT V_real T_initial V_initial T_target V_target : ℝ) 
  (h_rel : volume_change_per_degree_rise ΔT V_real)
  (h_init : initial_conditions V_initial T_initial)
  (h_target : target_temperature T_target) :
  V_target = V_initial + V_real * ((T_target - T_initial) / ΔT) :=
by
  -- Insert proof here
  sorry

end volume_at_target_temperature_l668_66855


namespace roots_of_quadratic_l668_66830

theorem roots_of_quadratic :
  ∃ (b c : ℝ), ( ∀ (x : ℝ), x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -2) :=
sorry

end roots_of_quadratic_l668_66830


namespace payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l668_66804

variable (x : ℕ)
variable (hx : x > 10)

noncomputable def option1_payment (x : ℕ) : ℕ := 200 * x + 8000
noncomputable def option2_payment (x : ℕ) : ℕ := 180 * x + 9000

theorem payment_option1 (x : ℕ) (hx : x > 10) : option1_payment x = 200 * x + 8000 :=
by sorry

theorem payment_option2 (x : ℕ) (hx : x > 10) : option2_payment x = 180 * x + 9000 :=
by sorry

theorem cost_effective_option (x : ℕ) (hx : x > 10) (h30 : x = 30) : option1_payment 30 < option2_payment 30 :=
by sorry

theorem most_cost_effective_plan (h30 : x = 30) : (10000 + 3600 = 13600) :=
by sorry

end payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l668_66804


namespace exists_adj_diff_gt_3_max_min_adj_diff_l668_66874
-- Import needed libraries

-- Definition of the given problem and statement of the parts (a) and (b)

-- Part (a)
theorem exists_adj_diff_gt_3 (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∃ i j : Fin 18, adj i j ∧ |arrangement i - arrangement j| > 3) :=
sorry

-- Part (b)
theorem max_min_adj_diff (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∀ i j : Fin 18, adj i j → |arrangement i - arrangement j| ≥ 6) :=
sorry

end exists_adj_diff_gt_3_max_min_adj_diff_l668_66874


namespace S_10_eq_110_l668_66860

-- Conditions
def a (n : ℕ) : ℕ := sorry  -- Assuming general term definition of arithmetic sequence
def S (n : ℕ) : ℕ := sorry  -- Assuming sum definition of arithmetic sequence

axiom a_3_eq_16 : a 3 = 16
axiom S_20_eq_20 : S 20 = 20

-- Prove
theorem S_10_eq_110 : S 10 = 110 :=
  by
  sorry

end S_10_eq_110_l668_66860


namespace minimum_button_presses_l668_66877

theorem minimum_button_presses :
  ∃ (r y g : ℕ), 
    2 * y - r = 3 ∧ 2 * g - y = 3 ∧ r + y + g = 9 :=
by sorry

end minimum_button_presses_l668_66877


namespace minimum_value_l668_66827

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 / x + 1 / y = 1) : 3 * x + 4 * y ≥ 25 :=
sorry

end minimum_value_l668_66827


namespace smallest_m_l668_66842

-- Definitions of lengths and properties of the pieces
variable {lengths : Fin 21 → ℝ} 
variable (h_all_pos : ∀ i, lengths i > 0)
variable (h_total_length : (Finset.univ : Finset (Fin 21)).sum lengths = 21)
variable (h_max_factor : ∀ i j, max (lengths i) (lengths j) ≤ 3 * min (lengths i) (lengths j))

-- Proof statement
theorem smallest_m (m : ℝ) (hm : ∀ i j, max (lengths i) (lengths j) ≤ m * min (lengths i) (lengths j)) : 
  m ≥ 1 := 
sorry

end smallest_m_l668_66842


namespace calculation_result_l668_66823

theorem calculation_result :
  5 * 7 - 6 * 8 + 9 * 2 + 7 * 3 = 26 :=
by sorry

end calculation_result_l668_66823


namespace functional_equation_solution_l668_66887

open Nat

theorem functional_equation_solution :
  (∀ (f : ℕ → ℕ), 
    (∀ (x y : ℕ), 0 ≤ y + f x - (Nat.iterate f (f y) x) ∧ (y + f x - (Nat.iterate f (f y) x) ≤ 1)) →
    (∀ n, f n = n + 1)) :=
by
  intro f h
  sorry

end functional_equation_solution_l668_66887


namespace mika_initial_stickers_l668_66878

theorem mika_initial_stickers :
  let store_stickers := 26.0
  let birthday_stickers := 20.0 
  let sister_stickers := 6.0 
  let mother_stickers := 58.0 
  let total_stickers := 130.0 
  ∃ x : Real, x + store_stickers + birthday_stickers + sister_stickers + mother_stickers = total_stickers ∧ x = 20.0 := 
by 
  sorry

end mika_initial_stickers_l668_66878


namespace bill_profit_difference_l668_66805

theorem bill_profit_difference (P SP NSP NP : ℝ) 
  (h1 : SP = 1.10 * P)
  (h2 : SP = 659.9999999999994)
  (h3 : NP = 0.90 * P)
  (h4 : NSP = 1.30 * NP) :
  NSP - SP = 42 := 
sorry

end bill_profit_difference_l668_66805


namespace ellipse_equation_l668_66868

open Real

theorem ellipse_equation (x y : ℝ) (h₁ : (- sqrt 15) = x) (h₂ : (5 / 2) = y)
  (h₃ : ∃ (a b : ℝ), (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + 5) 
  ∧ b^2 = 20 ∧ a^2 = 25) :
  (x^2 / 20 + y^2 / 25 = 1) :=
sorry

end ellipse_equation_l668_66868


namespace tan_ratio_alpha_beta_l668_66859

theorem tan_ratio_alpha_beta 
  (α β : ℝ) 
  (h1 : Real.sin (α + β) = 1 / 5) 
  (h2 : Real.sin (α - β) = 3 / 5) : 
  Real.tan α / Real.tan β = -1 :=
sorry

end tan_ratio_alpha_beta_l668_66859


namespace residue_5_pow_1234_mod_13_l668_66836

theorem residue_5_pow_1234_mod_13 : ∃ k : ℤ, 5^1234 = 13 * k + 12 :=
by
  sorry

end residue_5_pow_1234_mod_13_l668_66836


namespace robin_made_more_cupcakes_l668_66891

theorem robin_made_more_cupcakes (initial final sold made: ℕ)
  (h1 : initial = 42)
  (h2 : sold = 22)
  (h3 : final = 59)
  (h4 : initial - sold + made = final) :
  made = 39 :=
  sorry

end robin_made_more_cupcakes_l668_66891


namespace set_representation_l668_66839

theorem set_representation :
  {p : ℕ × ℕ | 2 * p.1 + 3 * p.2 = 16} = {(2, 4), (5, 2), (8, 0)} :=
by
  sorry

end set_representation_l668_66839


namespace range_of_slopes_of_line_AB_l668_66835

variables {x y : ℝ}

/-- (O is the coordinate origin),
    (the parabola y² = 4x),
    (points A and B in the first quadrant),
    (the product of the slopes of lines OA and OB being 1) -/
theorem range_of_slopes_of_line_AB
  (O : ℝ) 
  (A B : ℝ × ℝ)
  (hxA : 0 < A.fst)
  (hyA : 0 < A.snd)
  (hxB : 0 < B.fst)
  (hyB : 0 < B.snd)
  (hA_on_parabola : A.snd^2 = 4 * A.fst)
  (hB_on_parabola : B.snd^2 = 4 * B.fst)
  (h_product_slopes : (A.snd / A.fst) * (B.snd / B.fst) = 1) :
  (0 < (B.snd - A.snd) / (B.fst - A.fst) ∧ (B.snd - A.snd) / (B.fst - A.fst) < 1/2) := 
by
  sorry

end range_of_slopes_of_line_AB_l668_66835


namespace Razorback_tshirt_shop_sales_l668_66884

theorem Razorback_tshirt_shop_sales :
  let tshirt_price := 98
  let hat_price := 45
  let scarf_price := 60
  let tshirts_sold_arkansas := 42
  let hats_sold_arkansas := 32
  let scarves_sold_arkansas := 15
  (tshirts_sold_arkansas * tshirt_price + hats_sold_arkansas * hat_price + scarves_sold_arkansas * scarf_price) = 6456 :=
by
  sorry

end Razorback_tshirt_shop_sales_l668_66884


namespace power_expression_l668_66892

theorem power_expression : (1 / ((-5)^4)^2) * (-5)^9 = -5 := sorry

end power_expression_l668_66892


namespace sum_of_A_and_B_zero_l668_66820

theorem sum_of_A_and_B_zero
  (A B C : ℝ)
  (h1 : A ≠ B)
  (h2 : C ≠ 0)
  (f g : ℝ → ℝ)
  (h3 : ∀ x, f x = A * x + B + C)
  (h4 : ∀ x, g x = B * x + A - C)
  (h5 : ∀ x, f (g x) - g (f x) = 2 * C) : A + B = 0 :=
sorry

end sum_of_A_and_B_zero_l668_66820


namespace convert_3241_quinary_to_septenary_l668_66876

/-- Convert quinary number 3241_(5) to septenary number, yielding 1205_(7). -/
theorem convert_3241_quinary_to_septenary : 
  let quinary := 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0
  let septenary := 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0
  quinary = 446 → septenary = 1205 :=
by
  intros
  -- Quinary to Decimal
  have h₁ : 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0 = 446 := by norm_num
  -- Decimal to Septenary
  have h₂ : 446 = 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0 := by norm_num
  exact sorry

end convert_3241_quinary_to_septenary_l668_66876


namespace expr_value_l668_66829

variable (a : ℝ)
variable (h : a^2 - 3 * a - 1011 = 0)

theorem expr_value : 2 * a^2 - 6 * a + 1 = 2023 :=
by
  -- insert proof here
  sorry

end expr_value_l668_66829


namespace rancher_lasso_probability_l668_66816

theorem rancher_lasso_probability : 
  let p_success := 1 / 2
  let p_failure := 1 - p_success
  (1 - p_failure ^ 3) = (7 / 8) := by
  sorry

end rancher_lasso_probability_l668_66816


namespace problem1_problem2_l668_66873

-- Define the sets of balls and boxes
inductive Ball
| ball1 | ball2 | ball3 | ball4

inductive Box
| boxA | boxB | boxC

-- Define the arrangements for the first problem
def arrangements_condition1 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball3 = Box.boxB) ∧
  (∃ b1 b2 b3 : Box, b1 ≠ b2 ∧ b2 ≠ b3 ∧ b3 ≠ b1 ∧ 
    ∃ (f : Ball → Box), 
      (f Ball.ball1 = b1) ∧ (f Ball.ball2 = b2) ∧ (f Ball.ball3 = Box.boxB) ∧ (f Ball.ball4 = b3))

-- Define the proof statement for the first problem
theorem problem1 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition1 arrangement → n = 7) :=
sorry

-- Define the arrangements for the second problem
def arrangements_condition2 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball1 ≠ Box.boxA) ∧
  (arrangement Ball.ball2 ≠ Box.boxB)

-- Define the proof statement for the second problem
theorem problem2 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition2 arrangement → n = 36) :=
sorry

end problem1_problem2_l668_66873
