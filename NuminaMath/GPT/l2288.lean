import Mathlib

namespace reflect_point_across_x_axis_l2288_228884

theorem reflect_point_across_x_axis {x y : ℝ} (h : (x, y) = (2, 3)) : (x, -y) = (2, -3) :=
by
  sorry

end reflect_point_across_x_axis_l2288_228884


namespace student_marks_l2288_228867

variable (M P C : ℕ)

theorem student_marks (h1 : C = P + 20) (h2 : (M + C) / 2 = 20) : M + P = 20 :=
by
  sorry

end student_marks_l2288_228867


namespace females_in_orchestra_not_in_band_l2288_228862

theorem females_in_orchestra_not_in_band 
  (females_in_band : ℤ) 
  (males_in_band : ℤ) 
  (females_in_orchestra : ℤ) 
  (males_in_orchestra : ℤ) 
  (females_in_both : ℤ) 
  (total_members : ℤ) 
  (h1 : females_in_band = 120) 
  (h2 : males_in_band = 100) 
  (h3 : females_in_orchestra = 100) 
  (h4 : males_in_orchestra = 120) 
  (h5 : females_in_both = 80) 
  (h6 : total_members = 260) : 
  (females_in_orchestra - females_in_both = 20) := 
  sorry

end females_in_orchestra_not_in_band_l2288_228862


namespace triangle_expression_l2288_228836

open Real

variable (D E F : ℝ)
variable (DE DF EF : ℝ)

-- conditions
def triangleDEF : Prop := DE = 7 ∧ DF = 9 ∧ EF = 8

theorem triangle_expression (h : triangleDEF DE DF EF) :
  (cos ((D - E)/2) / sin (F/2) - sin ((D - E)/2) / cos (F/2)) = 81/28 :=
by
  have h1 : DE = 7 := h.1
  have h2 : DF = 9 := h.2.1
  have h3 : EF = 8 := h.2.2
  sorry

end triangle_expression_l2288_228836


namespace bushes_needed_for_60_zucchinis_l2288_228845

-- Each blueberry bush yields 10 containers of blueberries.
def containers_per_bush : ℕ := 10

-- 6 containers of blueberries can be traded for 3 zucchinis.
def containers_to_zucchinis (containers zucchinis : ℕ) : Prop := containers = 6 ∧ zucchinis = 3

theorem bushes_needed_for_60_zucchinis (bushes containers zucchinis : ℕ) :
  containers_per_bush = 10 →
  containers_to_zucchinis 6 3 →
  zucchinis = 60 →
  bushes = 12 :=
by
  intros h1 h2 h3
  sorry

end bushes_needed_for_60_zucchinis_l2288_228845


namespace roots_exist_range_k_l2288_228819

theorem roots_exist_range_k (k : ℝ) : 
  (∃ x1 x2 : ℝ, (2 * k * x1^2 + (8 * k + 1) * x1 + 8 * k = 0) ∧ 
                 (2 * k * x2^2 + (8 * k + 1) * x2 + 8 * k = 0)) ↔ 
  (k ≥ -1/16 ∧ k ≠ 0) :=
sorry

end roots_exist_range_k_l2288_228819


namespace san_francisco_superbowl_probability_l2288_228824

theorem san_francisco_superbowl_probability
  (P_play P_not_play : ℝ)
  (k : ℝ)
  (h1 : P_play = k * P_not_play)
  (h2 : P_play + P_not_play = 1) :
  k > 0 :=
sorry

end san_francisco_superbowl_probability_l2288_228824


namespace probability_factor_less_than_eight_l2288_228880

theorem probability_factor_less_than_eight (n : ℕ) (h72 : n = 72) :
  (∃ k < 8, k ∣ n) →
  (∃ p q, p/q = 5/12) :=
by
  sorry

end probability_factor_less_than_eight_l2288_228880


namespace combined_volume_cone_hemisphere_cylinder_l2288_228807

theorem combined_volume_cone_hemisphere_cylinder (r h : ℝ)
  (vol_cylinder : ℝ) (vol_cone : ℝ) (vol_hemisphere : ℝ)
  (H1 : vol_cylinder = 72 * π)
  (H2 : vol_cylinder = π * r^2 * h)
  (H3 : vol_cone = (1/3) * π * r^2 * h)
  (H4 : vol_hemisphere = (2/3) * π * r^3)
  (H5 : vol_cylinder = vol_cone + vol_hemisphere) :
  vol_cylinder = 72 * π :=
by
  sorry

end combined_volume_cone_hemisphere_cylinder_l2288_228807


namespace angle_diff_l2288_228838

-- Given conditions as definitions
def angle_A : ℝ := 120
def angle_B : ℝ := 50
def angle_D : ℝ := 60
def angle_E : ℝ := 140

-- Prove the difference between angle BCD and angle AFE is 10 degrees
theorem angle_diff (AB_parallel_DE : ∀ (A B D E : ℝ), AB_parallel_DE)
                 (angle_A_def : angle_A = 120)
                 (angle_B_def : angle_B = 50)
                 (angle_D_def : angle_D = 60)
                 (angle_E_def : angle_E = 140) :
    let angle_3 : ℝ := 180 - angle_A
    let angle_4 : ℝ := 180 - angle_E
    let angle_BCD : ℝ := angle_B + angle_D
    let angle_AFE : ℝ := angle_3 + angle_4
    angle_BCD - angle_AFE = 10 :=
by {
  sorry
}

end angle_diff_l2288_228838


namespace fraction_equality_l2288_228851

theorem fraction_equality
  (a b c d : ℝ) 
  (h1 : b ≠ c)
  (h2 : (a * c - b^2) / (a - 2 * b + c) = (b * d - c^2) / (b - 2 * c + d)) : 
  (a * c - b^2) / (a - 2 * b + c) = (a * d - b * c) / (a - b - c + d) ∧
  (b * d - c^2) / (b - 2 * c + d) = (a * d - b * c) / (a - b - c + d) := 
by
  sorry

end fraction_equality_l2288_228851


namespace nuts_per_box_l2288_228863

theorem nuts_per_box (N : ℕ)  
  (h1 : ∀ (boxes bolts_per_box : ℕ), boxes = 7 ∧ bolts_per_box = 11 → boxes * bolts_per_box = 77)
  (h2 : ∀ (boxes: ℕ), boxes = 3 → boxes * N = 3 * N)
  (h3 : ∀ (used_bolts purchased_bolts remaining_bolts : ℕ), purchased_bolts = 77 ∧ remaining_bolts = 3 → used_bolts = purchased_bolts - remaining_bolts)
  (h4 : ∀ (used_nuts purchased_nuts remaining_nuts : ℕ), purchased_nuts = 3 * N ∧ remaining_nuts = 6 → used_nuts = purchased_nuts - remaining_nuts)
  (h5 : ∀ (used_bolts used_nuts total_used : ℕ), used_bolts = 74 ∧ used_nuts = 3 * N - 6 → total_used = used_bolts + used_nuts)
  (h6 : total_used_bolts_and_nuts = 113) :
  N = 15 :=
by
  sorry

end nuts_per_box_l2288_228863


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l2288_228820

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l2288_228820


namespace estimate_fish_in_pond_l2288_228821

theorem estimate_fish_in_pond
  (n m k : ℕ)
  (h_pr: k = 200)
  (h_cr: k = 8)
  (h_m: n = 200):
  n / (m / k) = 5000 := sorry

end estimate_fish_in_pond_l2288_228821


namespace total_present_ages_l2288_228846

theorem total_present_ages (P Q : ℕ) 
    (h1 : P - 12 = (1 / 2) * (Q - 12))
    (h2 : P = (3 / 4) * Q) : P + Q = 42 :=
by
  sorry

end total_present_ages_l2288_228846


namespace no_five_consecutive_divisible_by_2025_l2288_228885

def seq (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2025 : 
  ¬ ∃ (a : ℕ), (∀ (i : ℕ), i < 5 → 2025 ∣ seq (a + i)) := 
sorry

end no_five_consecutive_divisible_by_2025_l2288_228885


namespace sqrt_infinite_nested_problem_l2288_228870

theorem sqrt_infinite_nested_problem :
  ∃ m : ℝ, m = Real.sqrt (6 + m) ∧ m = 3 :=
by
  sorry

end sqrt_infinite_nested_problem_l2288_228870


namespace factor_z4_minus_81_l2288_228804

theorem factor_z4_minus_81 :
  (z^4 - 81) = (z - 3) * (z + 3) * (z^2 + 9) :=
by
  sorry

end factor_z4_minus_81_l2288_228804


namespace solve_expression_l2288_228854

theorem solve_expression : (0.76 ^ 3 - 0.008) / (0.76 ^ 2 + 0.76 * 0.2 + 0.04) = 0.560 := 
by
  sorry

end solve_expression_l2288_228854


namespace probability_of_triangle_with_nonagon_side_l2288_228882

-- Definitions based on the given conditions
def num_vertices : ℕ := 9

def total_triangles : ℕ := Nat.choose num_vertices 3

def favorable_outcomes : ℕ :=
  let one_side_is_side_of_nonagon := num_vertices * 5
  let two_sides_are_sides_of_nonagon := num_vertices
  one_side_is_side_of_nonagon + two_sides_are_sides_of_nonagon

def probability : ℚ := favorable_outcomes / total_triangles

-- Lean 4 statement to prove the equivalence of the probability calculation
theorem probability_of_triangle_with_nonagon_side :
  probability = 9 / 14 :=
by
  sorry

end probability_of_triangle_with_nonagon_side_l2288_228882


namespace find_point_N_l2288_228841

-- Definition of symmetrical reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Given condition
def point_M : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem find_point_N : reflect_x point_M = (1, -3) :=
by
  sorry

end find_point_N_l2288_228841


namespace probability_B_and_C_exactly_two_out_of_A_B_C_l2288_228856

variables (A B C : Prop)
noncomputable def P : Prop → ℚ := sorry

axiom hA : P A = 3 / 4
axiom hAC : P (¬ A ∧ ¬ C) = 1 / 12
axiom hBC : P (B ∧ C) = 1 / 4

theorem probability_B_and_C : P B = 3 / 8 ∧ P C = 2 / 3 :=
sorry

theorem exactly_two_out_of_A_B_C : 
  P (A ∧ B ∧ ¬ C) + P (A ∧ ¬ B ∧ C) + P (¬ A ∧ B ∧ C) = 15 / 32 :=
sorry

end probability_B_and_C_exactly_two_out_of_A_B_C_l2288_228856


namespace sixth_term_of_arithmetic_sequence_l2288_228842

noncomputable def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + (n * (n - 1) / 2) * d

theorem sixth_term_of_arithmetic_sequence
  (a d : ℕ)
  (h₁ : sum_first_n_terms a d 4 = 10)
  (h₂ : a + 4 * d = 5) :
  a + 5 * d = 6 :=
by {
  sorry
}

end sixth_term_of_arithmetic_sequence_l2288_228842


namespace total_practice_hours_l2288_228837

def weekly_practice_hours : ℕ := 4
def weeks_per_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours :
  weekly_practice_hours * weeks_per_month * months = 80 := by
  sorry

end total_practice_hours_l2288_228837


namespace male_worker_ants_percentage_l2288_228832

theorem male_worker_ants_percentage 
  (total_ants : ℕ) 
  (half_ants : ℕ) 
  (female_worker_ants : ℕ) 
  (h1 : total_ants = 110) 
  (h2 : half_ants = total_ants / 2) 
  (h3 : female_worker_ants = 44) :
  (half_ants - female_worker_ants) * 100 / half_ants = 20 := by
  sorry

end male_worker_ants_percentage_l2288_228832


namespace simplification_l2288_228890

theorem simplification (a b c : ℤ) :
  (12 * a + 35 * b + 17 * c) + (13 * a - 15 * b + 8 * c) - (8 * a + 28 * b - 25 * c) = 17 * a - 8 * b + 50 * c :=
by
  sorry

end simplification_l2288_228890


namespace vector_projection_line_l2288_228875

theorem vector_projection_line (v : ℝ × ℝ) 
  (h : ∃ (x y : ℝ), v = (x, y) ∧ 
       (3 * x + 4 * y) / (3 ^ 2 + 4 ^ 2) = 1) :
  ∃ (x y : ℝ), v = (x, y) ∧ y = -3 / 4 * x + 25 / 4 :=
by
  sorry

end vector_projection_line_l2288_228875


namespace cannot_be_zero_l2288_228892

noncomputable def P (x : ℝ) (a b c d e : ℝ) := x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem cannot_be_zero (a b c d e : ℝ) (p q r s : ℝ) :
  e = 0 ∧ c = 0 ∧ (∀ x, P x a b c d e = x * (x - p) * (x - q) * (x - r) * (x - s)) ∧ 
  (p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  d ≠ 0 := 
by {
  sorry
}

end cannot_be_zero_l2288_228892


namespace cos_pi_div_four_minus_alpha_l2288_228857

theorem cos_pi_div_four_minus_alpha (α : ℝ) (h : Real.sin (π / 4 + α) = 2 / 3) : 
    Real.cos (π / 4 - α) = -Real.sqrt 5 / 3 :=
sorry

end cos_pi_div_four_minus_alpha_l2288_228857


namespace find_first_number_l2288_228805

theorem find_first_number (sum_is_33 : ∃ x y : ℕ, x + y = 33) (second_is_twice_first : ∃ x y : ℕ, y = 2 * x) (second_is_22 : ∃ y : ℕ, y = 22) : ∃ x : ℕ, x = 11 :=
by
  sorry

end find_first_number_l2288_228805


namespace find_years_in_future_l2288_228894

theorem find_years_in_future 
  (S F : ℕ)
  (h1 : F = 4 * S + 4)
  (h2 : F = 44) :
  ∃ x : ℕ, F + x = 2 * (S + x) + 20 ∧ x = 4 :=
by 
  sorry

end find_years_in_future_l2288_228894


namespace product_of_roots_in_range_l2288_228859

noncomputable def f (x : ℝ) : ℝ := abs (abs (x - 1) - 1)

theorem product_of_roots_in_range (m : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∃ x1 x2 x3 x4 : ℝ, 
        f x1 = m ∧ 
        f x2 = m ∧ 
        f x3 = m ∧ 
        f x4 = m ∧ 
        x1 ≠ x2 ∧ 
        x1 ≠ x3 ∧ 
        x1 ≠ x4 ∧ 
        x2 ≠ x3 ∧ 
        x2 ≠ x4 ∧ 
        x3 ≠ x4) :
  ∃ p : ℝ, p = (m * (2 - m) * (m + 2) * (-m)) ∧ -3 < p ∧ p < 0 :=
sorry

end product_of_roots_in_range_l2288_228859


namespace interest_for_1_rs_l2288_228879

theorem interest_for_1_rs (I₅₀₀₀ : ℝ) (P : ℝ) (h : I₅₀₀₀ = 200) (hP : P = 5000) : I₅₀₀₀ / P = 0.04 :=
by
  rw [h, hP]
  norm_num

end interest_for_1_rs_l2288_228879


namespace birds_more_than_half_sunflower_seeds_l2288_228810

theorem birds_more_than_half_sunflower_seeds :
  ∃ (n : ℕ), n = 3 ∧ ((4 / 5)^n * (2 / 5) + (2 / 5) > 1 / 2) :=
by
  sorry

end birds_more_than_half_sunflower_seeds_l2288_228810


namespace numberOfRottweilers_l2288_228883

-- Define the grooming times in minutes for each type of dog
def groomingTimeRottweiler := 20
def groomingTimeCollie := 10
def groomingTimeChihuahua := 45

-- Define the number of each type of dog groomed
def numberOfCollies := 9
def numberOfChihuahuas := 1

-- Define the total grooming time in minutes
def totalGroomingTime := 255

-- Compute the time spent on grooming Collies
def timeSpentOnCollies := numberOfCollies * groomingTimeCollie

-- Compute the time spent on grooming Chihuahuas
def timeSpentOnChihuahuas := numberOfChihuahuas * groomingTimeChihuahua

-- Compute the time spent on grooming Rottweilers
def timeSpentOnRottweilers := totalGroomingTime - timeSpentOnCollies - timeSpentOnChihuahuas

-- The main theorem statement
theorem numberOfRottweilers :
  timeSpentOnRottweilers / groomingTimeRottweiler = 6 :=
by
  -- Proof placeholder
  sorry

end numberOfRottweilers_l2288_228883


namespace price_change_on_eggs_and_apples_l2288_228816

theorem price_change_on_eggs_and_apples :
  let initial_egg_price := 1.00
  let initial_apple_price := 1.00
  let egg_drop_percent := 0.10
  let apple_increase_percent := 0.02
  let new_egg_price := initial_egg_price * (1 - egg_drop_percent)
  let new_apple_price := initial_apple_price * (1 + apple_increase_percent)
  let initial_total := initial_egg_price + initial_apple_price
  let new_total := new_egg_price + new_apple_price
  let percent_change := ((new_total - initial_total) / initial_total) * 100
  percent_change = -4 :=
by
  sorry

end price_change_on_eggs_and_apples_l2288_228816


namespace quadratic_roots_distinct_l2288_228861

variable (a b c : ℤ)

theorem quadratic_roots_distinct (h_eq : 3 * a^2 - 3 * a - 4 = 0) : ∃ (x y : ℝ), x ≠ y ∧ (3 * x^2 - 3 * x - 4 = 0) ∧ (3 * y^2 - 3 * y - 4 = 0) := 
  sorry

end quadratic_roots_distinct_l2288_228861


namespace sum_is_odd_prob_l2288_228844

-- A type representing the spinner results, which can be either 1, 2, 3 or 4.
inductive SpinnerResult
| one : SpinnerResult
| two : SpinnerResult
| three : SpinnerResult
| four : SpinnerResult

open SpinnerResult

-- Function to determine if a spinner result is odd.
def isOdd (r : SpinnerResult) : Bool :=
  match r with
  | one => true
  | three => true
  | two => false
  | four => false

-- Defining the spinners P, Q, R, and S.
noncomputable def P : SpinnerResult := SpinnerResult.one -- example, could vary
noncomputable def Q : SpinnerResult := SpinnerResult.two -- example, could vary
noncomputable def R : SpinnerResult := SpinnerResult.three -- example, could vary
noncomputable def S : SpinnerResult := SpinnerResult.four -- example, could vary

-- Probability calculation function
def probabilityOddSum : ℚ :=
  let probOdd := 1 / 2
  let probEven := 1 / 2
  let scenario1 := 4 * probOdd * probEven^3
  let scenario2 := 4 * probOdd^3 * probEven
  scenario1 + scenario2

-- The theorem to be stated
theorem sum_is_odd_prob :
  probabilityOddSum = 1 / 2 := by
  sorry

end sum_is_odd_prob_l2288_228844


namespace triangle_area_ratio_l2288_228874

theorem triangle_area_ratio {A B C : ℝ} {a b c : ℝ} 
  (h : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2 / 3) 
  (S1 : ℝ) (S2 : ℝ) :
  S1 / S2 = 1 / (3 * Real.pi) :=
sorry

end triangle_area_ratio_l2288_228874


namespace prime_iff_sum_four_distinct_products_l2288_228868

variable (n : ℕ) (a b c d : ℕ)

theorem prime_iff_sum_four_distinct_products (h : n ≥ 5) :
  (Prime n ↔ ∀ (a b c d : ℕ), n = a + b + c + d → a > 0 → b > 0 → c > 0 → d > 0 → ab ≠ cd) :=
sorry

end prime_iff_sum_four_distinct_products_l2288_228868


namespace total_repairs_cost_eq_l2288_228893

-- Assume the initial cost of the scooter is represented by a real number C.
variable (C : ℝ)

-- Given conditions
def spent_on_first_repair := 0.05 * C
def spent_on_second_repair := 0.10 * C
def spent_on_third_repair := 0.07 * C

-- Total repairs expenditure
def total_repairs := spent_on_first_repair C + spent_on_second_repair C + spent_on_third_repair C

-- Selling price and profit
def selling_price := 1.25 * C
def profit := 1500
def profit_calc := selling_price C - (C + total_repairs C)

-- Statement to be proved: The total repairs is equal to $11,000.
theorem total_repairs_cost_eq : total_repairs 50000 = 11000 := by
  sorry

end total_repairs_cost_eq_l2288_228893


namespace even_function_k_value_l2288_228864

theorem even_function_k_value (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = k * x^2 + (k - 1) * x + 2)
  (even_f : ∀ x : ℝ, f x = f (-x)) : k = 1 :=
by
  -- Proof would go here
  sorry

end even_function_k_value_l2288_228864


namespace initial_ratio_of_milk_to_water_l2288_228800

theorem initial_ratio_of_milk_to_water (M W : ℕ) (h1 : M + W = 60) (h2 : 2 * M = W + 60) : M / W = 2 :=
by
  sorry

end initial_ratio_of_milk_to_water_l2288_228800


namespace sufficient_but_not_necessary_condition_for_hyperbola_l2288_228855

theorem sufficient_but_not_necessary_condition_for_hyperbola (k : ℝ) :
  (∃ k : ℝ, k > 3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) ∧ 
  (∃ k : ℝ, k < -3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) :=
    sorry

end sufficient_but_not_necessary_condition_for_hyperbola_l2288_228855


namespace weight_of_empty_box_l2288_228808

theorem weight_of_empty_box (w12 w8 w : ℝ) (h1 : w12 = 11.48) (h2 : w8 = 8.12) (h3 : ∀ b : ℕ, b > 0 → w = 0.84) :
  w8 - 8 * w = 1.40 :=
by
  sorry

end weight_of_empty_box_l2288_228808


namespace part1_solution_set_part2_range_of_m_l2288_228827

noncomputable def f (x : ℝ) : ℝ := abs (x + 2) * abs (x - 3)

theorem part1_solution_set :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} :=
sorry

theorem part2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≤ abs (3 * m - 2)) → m ∈ Set.Iic (-1) ∪ Set.Ici (7 / 3) :=
sorry

end part1_solution_set_part2_range_of_m_l2288_228827


namespace num_5_digit_even_div_by_5_l2288_228878

theorem num_5_digit_even_div_by_5 : ∃! (n : ℕ), n = 500 ∧ ∀ (d : ℕ), 
  10000 ≤ d ∧ d ≤ 99999 → 
  (∀ i, i ∈ [0, 1, 2, 3, 4] → ((d / 10^i) % 10) % 2 = 0) ∧
  (d % 10 = 0) → 
  n = 500 := sorry

end num_5_digit_even_div_by_5_l2288_228878


namespace amount_borrowed_from_bank_l2288_228801

-- Definitions of the conditions
def car_price : ℝ := 35000
def total_payment : ℝ := 38000
def interest_rate : ℝ := 0.15

theorem amount_borrowed_from_bank :
  total_payment - car_price = interest_rate * (total_payment - car_price) / interest_rate := sorry

end amount_borrowed_from_bank_l2288_228801


namespace expand_product_l2288_228873

theorem expand_product (x : ℝ) : 4 * (x + 3) * (x + 6) = 4 * x^2 + 36 * x + 72 :=
by
  sorry

end expand_product_l2288_228873


namespace derivative_of_y_l2288_228813

noncomputable def y (x : ℝ) : ℝ :=
  (4 * x + 1) / (16 * x^2 + 8 * x + 3) + (1 / Real.sqrt 2) * Real.arctan ((4 * x + 1) / Real.sqrt 2)

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 16 / (16 * x^2 + 8 * x + 3)^2 :=
by 
  sorry

end derivative_of_y_l2288_228813


namespace associate_professors_bring_one_chart_l2288_228806

theorem associate_professors_bring_one_chart
(A B C : ℕ) (h1 : 2 * A + B = 7) (h2 : A * C + 2 * B = 11) (h3 : A + B = 6) : C = 1 :=
by sorry

end associate_professors_bring_one_chart_l2288_228806


namespace cylinder_volume_ratio_l2288_228848

theorem cylinder_volume_ratio (w h : ℝ) (w_pos : w = 6) (h_pos : h = 9) :
  let r1 := w / (2 * Real.pi)
  let h1 := h
  let V1 := Real.pi * r1^2 * h1
  let r2 := h / (2 * Real.pi)
  let h2 := w
  let V2 := Real.pi * r2^2 * h2
  V2 / V1 = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l2288_228848


namespace inequality_holds_equality_cases_l2288_228895

noncomputable def posReal : Type := { x : ℝ // 0 < x }

variables (a b c d : posReal)

theorem inequality_holds (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) ≥ 0 :=
sorry

theorem equality_cases (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) = 0 ↔
  (a.1 = c.1 ∧ b.1 = d.1) :=
sorry

end inequality_holds_equality_cases_l2288_228895


namespace doughnuts_per_person_l2288_228871

-- Define the number of dozens bought by Samuel
def samuel_dozens : ℕ := 2

-- Define the number of dozens bought by Cathy
def cathy_dozens : ℕ := 3

-- Define the number of doughnuts in one dozen
def dozen : ℕ := 12

-- Define the total number of people
def total_people : ℕ := 10

-- Prove that each person receives 6 doughnuts
theorem doughnuts_per_person :
  ((samuel_dozens * dozen) + (cathy_dozens * dozen)) / total_people = 6 :=
by
  sorry

end doughnuts_per_person_l2288_228871


namespace sum_of_real_solutions_l2288_228817

theorem sum_of_real_solutions :
  ∀ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 11 * x) →
  (∃ r1 r2 : ℝ, r1 + r2 = 46 / 13) :=
by
  sorry

end sum_of_real_solutions_l2288_228817


namespace sunny_ahead_in_second_race_l2288_228814

theorem sunny_ahead_in_second_race
  (s w : ℝ)
  (h1 : s / w = 8 / 7) :
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  450 - distance_windy_in_time_sunny = 12.5 :=
by
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  sorry

end sunny_ahead_in_second_race_l2288_228814


namespace math_proof_problem_l2288_228858

theorem math_proof_problem
  (a b c : ℝ)
  (h : a ≠ b)
  (h1 : b ≠ c)
  (h2 : c ≠ a)
  (h3 : (a / (2 * (b - c))) + (b / (2 * (c - a))) + (c / (2 * (a - b))) = 0) :
  (a / (b - c)^3) + (b / (c - a)^3) + (c / (a - b)^3) = 0 := 
by
  sorry

end math_proof_problem_l2288_228858


namespace share_of_a_120_l2288_228822

theorem share_of_a_120 (A B C : ℝ) 
  (h1 : A = (2 / 3) * (B + C)) 
  (h2 : B = (6 / 9) * (A + C)) 
  (h3 : A + B + C = 300) : 
  A = 120 := 
by 
  sorry

end share_of_a_120_l2288_228822


namespace gwen_books_collection_l2288_228830

theorem gwen_books_collection :
  let mystery_books := 8 * 6
  let picture_books := 5 * 4
  let science_books := 4 * 7
  let non_fiction_books := 3 * 5
  let lent_mystery_books := 2
  let lent_science_books := 3
  let borrowed_picture_books := 5
  mystery_books - lent_mystery_books + picture_books - borrowed_picture_books + borrowed_picture_books + science_books - lent_science_books + non_fiction_books = 106 := by
  sorry

end gwen_books_collection_l2288_228830


namespace sum_a_b_max_power_l2288_228897

theorem sum_a_b_max_power (a b : ℕ) (h_pos : 0 < a) (h_b_gt_1 : 1 < b) (h_lt_600 : a ^ b < 600) : a + b = 26 :=
sorry

end sum_a_b_max_power_l2288_228897


namespace find_divisor_l2288_228887

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) 
  (h1 : dividend = 62976) 
  (h2 : quotient = 123) 
  (h3 : dividend = divisor * quotient) 
  : divisor = 512 := 
by
  sorry

end find_divisor_l2288_228887


namespace inequality_proof_l2288_228829

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (2 * x * y) / (x + y) + Real.sqrt ((x ^ 2 + y ^ 2) / 2) ≥ (x + y) / 2 + Real.sqrt (x * y) :=
by
  sorry

end inequality_proof_l2288_228829


namespace sandwiches_final_count_l2288_228828

def sandwiches_left (initial : ℕ) (eaten_by_ruth : ℕ) (given_to_brother : ℕ) (eaten_by_first_cousin : ℕ) (eaten_by_other_cousins : ℕ) : ℕ :=
  initial - (eaten_by_ruth + given_to_brother + eaten_by_first_cousin + eaten_by_other_cousins)

theorem sandwiches_final_count :
  sandwiches_left 10 1 2 2 2 = 3 := by
  sorry

end sandwiches_final_count_l2288_228828


namespace distance_BC_l2288_228899

variable (AC AB : ℝ) (angleACB : ℝ)
  (hAC : AC = 2)
  (hAB : AB = 3)
  (hAngle : angleACB = 120)

theorem distance_BC (BC : ℝ) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end distance_BC_l2288_228899


namespace ball_hits_ground_l2288_228888

theorem ball_hits_ground :
  ∃ t : ℝ, -16 * t^2 + 20 * t + 100 = 0 ∧ t = (5 + Real.sqrt 425) / 8 :=
by
  sorry

end ball_hits_ground_l2288_228888


namespace custom_op_evaluation_l2288_228811

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : (custom_op 9 6) - (custom_op 6 9) = -12 := by
  sorry

end custom_op_evaluation_l2288_228811


namespace clara_weight_l2288_228826

theorem clara_weight (a c : ℝ) (h1 : a + c = 220) (h2 : c - a = c / 3) : c = 88 :=
by
  sorry

end clara_weight_l2288_228826


namespace min_value_of_f_l2288_228835

noncomputable def f (x : ℝ) : ℝ := 4 * x + 2 / x

theorem min_value_of_f (x : ℝ) (hx : x > 0) : ∃ y : ℝ, (∀ z : ℝ, z > 0 → f z ≥ y) ∧ y = 4 * Real.sqrt 2 :=
sorry

end min_value_of_f_l2288_228835


namespace subset_M_P_N_l2288_228865

def setM : Set (ℝ × ℝ) := {p | |p.1| + |p.2| < 1}

def setN : Set (ℝ × ℝ) := 
  {p | (Real.sqrt ((p.1 - 1 / 2) ^ 2 + (p.2 + 1 / 2) ^ 2) + Real.sqrt ((p.1 + 1 / 2) ^ 2 + (p.2 - 1 / 2) ^ 2)) < 2 * Real.sqrt 2}

def setP : Set (ℝ × ℝ) := 
  {p | |p.1 + p.2| < 1 ∧ |p.1| < 1 ∧ |p.2| < 1}

theorem subset_M_P_N : setM ⊆ setP ∧ setP ⊆ setN := by
  sorry

end subset_M_P_N_l2288_228865


namespace domain_f_1_minus_2x_is_0_to_half_l2288_228881

-- Define the domain of f(x) as a set.
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Define the domain condition for f(1 - 2*x).
def domain_f_1_minus_2x (x : ℝ) : Prop := 0 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 1

-- State the theorem: If x is in the domain of f(1 - 2*x), then x is in [0, 1/2].
theorem domain_f_1_minus_2x_is_0_to_half :
  ∀ x : ℝ, domain_f_1_minus_2x x ↔ (0 ≤ x ∧ x ≤ 1 / 2) := by
  sorry

end domain_f_1_minus_2x_is_0_to_half_l2288_228881


namespace area_larger_sphere_l2288_228860

noncomputable def sphere_area_relation (A1: ℝ) (R1 R2: ℝ) := R2^2 / R1^2 * A1

-- Given Conditions
def radius_smaller_sphere : ℝ := 4.0  -- R1
def radius_larger_sphere : ℝ := 6.0    -- R2
def area_smaller_sphere : ℝ := 17.0    -- A1

-- Target Area Calculation based on Proportional Relationship
theorem area_larger_sphere :
  sphere_area_relation area_smaller_sphere radius_smaller_sphere radius_larger_sphere = 38.25 :=
by
  sorry

end area_larger_sphere_l2288_228860


namespace speed_in_still_water_l2288_228812

theorem speed_in_still_water (upstream_speed : ℝ) (downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 45) (h_downstream : downstream_speed = 55) : 
  (upstream_speed + downstream_speed) / 2 = 50 := 
by
  rw [h_upstream, h_downstream] 
  norm_num  -- simplifies the numeric expression
  done

end speed_in_still_water_l2288_228812


namespace andrew_ruined_planks_l2288_228852

variable (b L k g h leftover plank_total ruin_bedroom ruin_guest : ℕ)

-- Conditions
def bedroom_planks := b
def living_room_planks := L
def kitchen_planks := k
def guest_bedroom_planks := g
def hallway_planks := h
def planks_leftover := leftover

-- Values
axiom bedroom_planks_val : bedroom_planks = 8
axiom living_room_planks_val : living_room_planks = 20
axiom kitchen_planks_val : kitchen_planks = 11
axiom guest_bedroom_planks_val : guest_bedroom_planks = bedroom_planks - 2
axiom hallway_planks_val : hallway_planks = 4
axiom planks_leftover_val : planks_leftover = 6

-- Total planks used and total planks had
def total_planks_used := bedroom_planks + living_room_planks + kitchen_planks + guest_bedroom_planks + (2 * hallway_planks)
def total_planks_had := total_planks_used + planks_leftover

-- Planks ruined
def planks_ruined_in_bedroom := ruin_bedroom
def planks_ruined_in_guest_bedroom := ruin_guest

-- Theorem to be proven
theorem andrew_ruined_planks :
  (planks_ruined_in_bedroom = total_planks_had - total_planks_used) ∧
  (planks_ruined_in_guest_bedroom = planks_ruined_in_bedroom) :=
by
  sorry

end andrew_ruined_planks_l2288_228852


namespace cos_arcsin_l2288_228818

theorem cos_arcsin (h : (7:ℝ) / 25 ≤ 1) : Real.cos (Real.arcsin ((7:ℝ) / 25)) = (24:ℝ) / 25 := by
  -- Proof to be provided
  sorry

end cos_arcsin_l2288_228818


namespace line_passing_through_points_l2288_228834

theorem line_passing_through_points (a_1 b_1 a_2 b_2 : ℝ) 
  (h1 : 2 * a_1 + 3 * b_1 + 1 = 0)
  (h2 : 2 * a_2 + 3 * b_2 + 1 = 0) : 
  ∃ (m n : ℝ), (∀ x y : ℝ, (y - b_1) * (x - a_2) = (y - b_2) * (x - a_1)) → (m = 2 ∧ n = 3) :=
by { sorry }

end line_passing_through_points_l2288_228834


namespace arithmetic_calculation_l2288_228872

theorem arithmetic_calculation : 3 - (-5) + 7 = 15 := by
  sorry

end arithmetic_calculation_l2288_228872


namespace Steve_bakes_more_apple_pies_l2288_228803

def Steve_bakes (days_apple days_cherry pies_per_day : ℕ) : ℕ :=
  (days_apple * pies_per_day) - (days_cherry * pies_per_day)

theorem Steve_bakes_more_apple_pies :
  Steve_bakes 3 2 12 = 12 :=
by
  sorry

end Steve_bakes_more_apple_pies_l2288_228803


namespace exists_integers_x_y_z_l2288_228833

theorem exists_integers_x_y_z (n : ℕ) : 
  ∃ x y z : ℤ, (x^2 + y^2 + z^2 = 3^(2^n)) ∧ (Int.gcd x (Int.gcd y z) = 1) :=
sorry

end exists_integers_x_y_z_l2288_228833


namespace charlie_extra_charge_l2288_228809

-- Define the data plan and cost structure
def data_plan_limit : ℕ := 8  -- GB
def extra_cost_per_gb : ℕ := 10  -- $ per GB

-- Define Charlie's data usage over each week
def usage_week_1 : ℕ := 2  -- GB
def usage_week_2 : ℕ := 3  -- GB
def usage_week_3 : ℕ := 5  -- GB
def usage_week_4 : ℕ := 10  -- GB

-- Calculate the total data usage and the extra data used
def total_usage : ℕ := usage_week_1 + usage_week_2 + usage_week_3 + usage_week_4
def extra_usage : ℕ := if total_usage > data_plan_limit then total_usage - data_plan_limit else 0
def extra_charge : ℕ := extra_usage * extra_cost_per_gb

-- Theorem to prove the extra charge
theorem charlie_extra_charge : extra_charge = 120 := by
  -- Skipping the proof
  sorry

end charlie_extra_charge_l2288_228809


namespace count_white_balls_l2288_228850

variable (W B : ℕ)

theorem count_white_balls
  (h_total : W + B = 30)
  (h_white : ∀ S : Finset ℕ, S.card = 12 → ∃ w ∈ S, w < W)
  (h_black : ∀ S : Finset ℕ, S.card = 20 → ∃ b ∈ S, b < B) :
  W = 19 :=
sorry

end count_white_balls_l2288_228850


namespace range_m_l2288_228849

open Set

noncomputable def A : Set ℝ := { x : ℝ | -5 ≤ x ∧ x ≤ 3 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | m + 1 < x ∧ x < 2 * m + 3 }

theorem range_m (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ↔ m ≤ 0 :=
by
  sorry

end range_m_l2288_228849


namespace g_of_neg2_l2288_228896

def g (x : ℤ) : ℤ := x^3 - x^2 + x

theorem g_of_neg2 : g (-2) = -14 := 
by
  sorry

end g_of_neg2_l2288_228896


namespace only_value_of_k_l2288_228815

def A (k a b : ℕ) : ℚ := (a + b : ℚ) / (a^2 + k^2 * b^2 - k^2 * a * b : ℚ)

theorem only_value_of_k : (∀ a b : ℕ, 0 < a → 0 < b → ¬ (∃ c d : ℕ, 1 < c ∧ A 1 a b = (c : ℚ) / (d : ℚ))) → k = 1 := 
    by sorry  -- proof omitted

-- Note: 'only_value_of_k' states that given the conditions, there is no k > 1 that makes A(k, a, b) a composite number, hence k must be 1.

end only_value_of_k_l2288_228815


namespace f_value_neg_five_half_one_l2288_228877

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom interval_definition : ∀ x, 0 < x ∧ x < 1 → f x = (4:ℝ) ^ x

-- The statement to prove
theorem f_value_neg_five_half_one : f (-5/2) + f 1 = -2 :=
by
  sorry

end f_value_neg_five_half_one_l2288_228877


namespace not_divisible_by_4_l2288_228839

theorem not_divisible_by_4 (n : Int) : ¬ (1 + n + n^2 + n^3 + n^4) % 4 = 0 := by
  sorry

end not_divisible_by_4_l2288_228839


namespace tan_pi_over_4_plus_alpha_eq_two_l2288_228843

theorem tan_pi_over_4_plus_alpha_eq_two
  (α : ℂ) 
  (h : Complex.tan ((π / 4) + α) = 2) : 
  (1 / (2 * Complex.sin α * Complex.cos α + (Complex.cos α)^2)) = (2 / 3) :=
by
  sorry

end tan_pi_over_4_plus_alpha_eq_two_l2288_228843


namespace students_who_like_both_channels_l2288_228866

theorem students_who_like_both_channels (total_students : ℕ) 
    (sports_channel : ℕ) (arts_channel : ℕ) (neither_channel : ℕ)
    (h_total : total_students = 100) (h_sports : sports_channel = 68) 
    (h_arts : arts_channel = 55) (h_neither : neither_channel = 3) :
    ∃ x, (x = 26) :=
by
  have h_at_least_one := total_students - neither_channel
  have h_A_union_B := sports_channel + arts_channel - h_at_least_one
  use h_A_union_B
  sorry

end students_who_like_both_channels_l2288_228866


namespace vector_addition_parallel_l2288_228898

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem vector_addition_parallel:
  ∀ x : ℝ, parallel (2, 1) (x, -2) → a + b x = ((-2 : ℝ), -1) :=
by
  intros x h
  sorry

end vector_addition_parallel_l2288_228898


namespace determine_Q_l2288_228891

def P : Set ℕ := {1, 2}

def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem determine_Q : Q = {2, 3, 4} :=
by
  sorry

end determine_Q_l2288_228891


namespace unique_function_property_l2288_228831

theorem unique_function_property (f : ℕ → ℕ) (h : ∀ m n : ℕ, f m + f n ∣ m + n) :
  ∀ m : ℕ, f m = m :=
by
  sorry

end unique_function_property_l2288_228831


namespace period_of_3sin_minus_4cos_l2288_228847

theorem period_of_3sin_minus_4cos (x : ℝ) : 
  ∃ T : ℝ, T = 2 * Real.pi ∧ (∀ x, 3 * Real.sin x - 4 * Real.cos x = 3 * Real.sin (x + T) - 4 * Real.cos (x + T)) :=
sorry

end period_of_3sin_minus_4cos_l2288_228847


namespace find_y_l2288_228889

theorem find_y (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_rem : x % y = 3) (h_div : (x:ℝ) / y = 96.15) : y = 20 :=
by
  sorry

end find_y_l2288_228889


namespace no_real_solution_of_fraction_eq_l2288_228853

theorem no_real_solution_of_fraction_eq (m : ℝ) :
  (∀ x : ℝ, (x - 1) / (x + 4) ≠ m / (x + 4)) → m = -5 :=
sorry

end no_real_solution_of_fraction_eq_l2288_228853


namespace evaluate_expression_l2288_228886

theorem evaluate_expression : (1023 * 1023) - (1022 * 1024) = 1 := by
  sorry

end evaluate_expression_l2288_228886


namespace parallel_lines_slope_l2288_228825

theorem parallel_lines_slope (m : ℝ) (h : (x + (1 + m) * y + m - 2 = 0) ∧ (m * x + 2 * y + 6 = 0)) :
  m = 1 ∨ m = -2 :=
  sorry

end parallel_lines_slope_l2288_228825


namespace final_silver_tokens_l2288_228869

structure TokenCounts :=
  (red : ℕ)
  (blue : ℕ)

def initial_tokens : TokenCounts := { red := 100, blue := 50 }

def exchange_booth1 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red - 3, blue := tokens.blue + 2 }

def exchange_booth2 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red + 1, blue := tokens.blue - 3 }

noncomputable def max_exchanges (initial : TokenCounts) : ℕ × ℕ :=
  let x := 48
  let y := 47
  (x, y)

noncomputable def silver_tokens (x y : ℕ) : ℕ := x + y

theorem final_silver_tokens (x y : ℕ) (tokens : TokenCounts) 
  (hx : tokens.red = initial_tokens.red - 3 * x + y)
  (hy : tokens.blue = initial_tokens.blue + 2 * x - 3 * y) 
  (hx_le : tokens.red >= 3 → false)
  (hy_le : tokens.blue >= 3 → false) : 
  silver_tokens x y = 95 :=
by {
  sorry
}

end final_silver_tokens_l2288_228869


namespace range_of_k_l2288_228802

open BigOperators

theorem range_of_k
  {f : ℝ → ℝ}
  (k : ℝ)
  (h : ∀ x : ℝ, f x = 32 * x - (k + 1) * 3^x + 2)
  (H : ∀ x : ℝ, f x > 0) :
  k < 1 /2 := 
sorry

end range_of_k_l2288_228802


namespace area_of_intersection_of_two_circles_l2288_228823

open Real

noncomputable def area_intersection (r : ℝ) (c1 c2 : ℝ × ℝ) : ℝ :=
  let quarter_circle_area := (1/4) * π * r^2
  let triangle_area := (1/2) * r^2
  let segment_area := quarter_circle_area - triangle_area
  2 * segment_area

theorem area_of_intersection_of_two_circles :
  area_intersection 3 (3, 0) (0, 3) = (9 * π - 18) / 2 :=
by
  -- This will be proven by the steps of the provided solution.
  sorry

end area_of_intersection_of_two_circles_l2288_228823


namespace sum_divisible_by_10_l2288_228876

theorem sum_divisible_by_10 :
    (111 ^ 111 + 112 ^ 112 + 113 ^ 113) % 10 = 0 :=
by
  sorry

end sum_divisible_by_10_l2288_228876


namespace unique_solution_for_lines_intersection_l2288_228840

theorem unique_solution_for_lines_intersection (n : ℕ) (h : n * (n - 1) / 2 = 2) : n = 2 :=
by
  sorry

end unique_solution_for_lines_intersection_l2288_228840
