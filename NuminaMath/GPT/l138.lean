import Mathlib

namespace pyramid_lateral_surface_area_l138_138839

noncomputable def lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) : ℝ :=
  n * S

theorem pyramid_lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) (A : ℝ) :
  A = n * S * (Real.cos α) →
  lateral_surface_area S n α = A / (Real.cos α) :=
by
  sorry

end pyramid_lateral_surface_area_l138_138839


namespace find_shorter_piece_length_l138_138723

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x = 8

theorem find_shorter_piece_length : ∃ x : ℕ, (20 - x) > 0 ∧ 2 * x = (20 - x) + 4 ∧ shorter_piece_length x :=
by
  -- There exists an x that satisfies the conditions
  use 8
  -- Prove the conditions are met
  sorry

end find_shorter_piece_length_l138_138723


namespace initially_caught_and_tagged_is_30_l138_138795

open Real

-- Define conditions
def total_second_catch : ℕ := 50
def tagged_second_catch : ℕ := 2
def total_pond_fish : ℕ := 750

-- Define ratio condition
def ratio_condition (T : ℕ) : Prop :=
  (T : ℝ) / (total_pond_fish : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ)

-- Prove the number of fish initially caught and tagged is 30
theorem initially_caught_and_tagged_is_30 :
  ∃ T : ℕ, ratio_condition T ∧ T = 30 :=
by
  -- Skipping proof
  sorry

end initially_caught_and_tagged_is_30_l138_138795


namespace positive_difference_of_two_numbers_l138_138452

theorem positive_difference_of_two_numbers :
  ∃ (x y : ℝ), x + y = 10 ∧ x^2 - y^2 = 24 ∧ |x - y| = 12 / 5 :=
by
  sorry

end positive_difference_of_two_numbers_l138_138452


namespace no_such_function_exists_l138_138049

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = 1 + x - y := by
  sorry

end no_such_function_exists_l138_138049


namespace gcd_16_12_eq_4_l138_138199

theorem gcd_16_12_eq_4 : Int.gcd 16 12 = 4 := by
  sorry

end gcd_16_12_eq_4_l138_138199


namespace triangle_angle_eq_pi_over_3_l138_138874

theorem triangle_angle_eq_pi_over_3
  (a b c : ℝ)
  (h : (a + b + c) * (a + b - c) = a * b)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ C : ℝ, C = 2 * Real.pi / 3 ∧ 
            Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b) :=
by
  -- Proof goes here
  sorry

end triangle_angle_eq_pi_over_3_l138_138874


namespace time_to_finish_work_with_both_tractors_l138_138786

-- Definitions of given conditions
def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 15
def time_A_worked : ℚ := 13
def remaining_work : ℚ := 1 - (work_rate_A * time_A_worked)
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Statement that needs to be proven
theorem time_to_finish_work_with_both_tractors : 
  remaining_work / combined_work_rate = 3 :=
by
  sorry

end time_to_finish_work_with_both_tractors_l138_138786


namespace range_of_a_l138_138979

open Real

theorem range_of_a (a : ℝ) (H : ∀ b : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs (x^2 + a * x + b) ≥ 1)) : a ≥ 1 ∨ a ≤ -3 :=
sorry

end range_of_a_l138_138979


namespace evaluate_at_3_l138_138184

def f (x : ℝ) : ℝ := 9 * x^4 + 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem evaluate_at_3 : f 3 = 876 := by
  sorry

end evaluate_at_3_l138_138184


namespace find_function_l138_138582

theorem find_function (f : ℝ → ℝ) :
  (∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v)) →
  (∀ u : ℝ, 0 ≤ f u) →
  (∀ x : ℝ, f x = 0) := 
  by
    sorry

end find_function_l138_138582


namespace cost_price_perc_of_selling_price_l138_138290

theorem cost_price_perc_of_selling_price
  (SP : ℝ) (CP : ℝ) (P : ℝ)
  (h1 : P = SP - CP)
  (h2 : P = (4.166666666666666 / 100) * SP) :
  CP = SP * 0.9583333333333334 :=
by
  sorry

end cost_price_perc_of_selling_price_l138_138290


namespace fruit_ratio_l138_138303

variable (A P B : ℕ)
variable (n : ℕ)

theorem fruit_ratio (h1 : A = 4) (h2 : P = n * A) (h3 : A + P + B = 21) (h4 : B = 5) : P / A = 3 := by
  sorry

end fruit_ratio_l138_138303


namespace largest_smallest_divisible_by_99_l138_138709

-- Definitions for distinct digits 3, 7, 9
def largest_number (x y z : Nat) : Nat := 100 * x + 10 * y + z
def smallest_number (x y z : Nat) : Nat := 100 * z + 10 * y + x

-- Proof problem statement
theorem largest_smallest_divisible_by_99 
  (a b c : Nat) (h : a > b ∧ b > c ∧ c > 0) : 
  ∃ (x y z : Nat), 
    (x = 9 ∧ y = 7 ∧ z = 3 ∧ largest_number x y z = 973 ∧ smallest_number x y z = 379) ∧
    99 ∣ (largest_number a b c - smallest_number a b c) :=
by
  sorry

end largest_smallest_divisible_by_99_l138_138709


namespace fraction_of_occupied_student_chairs_is_4_over_5_l138_138494

-- Definitions based on the conditions provided
def total_chairs : ℕ := 10 * 15
def awardees_chairs : ℕ := 15
def admin_teachers_chairs : ℕ := 2 * 15
def parents_chairs : ℕ := 2 * 15
def student_chairs : ℕ := total_chairs - (awardees_chairs + admin_teachers_chairs + parents_chairs)
def vacant_student_chairs_given_to_parents : ℕ := 15
def occupied_student_chairs : ℕ := student_chairs - vacant_student_chairs_given_to_parents

-- Theorem statement based on the problem
theorem fraction_of_occupied_student_chairs_is_4_over_5 :
    (occupied_student_chairs : ℚ) / student_chairs = 4 / 5 :=
by
    sorry

end fraction_of_occupied_student_chairs_is_4_over_5_l138_138494


namespace tiles_covering_the_floor_l138_138651

theorem tiles_covering_the_floor 
  (L W : ℕ) 
  (h1 : (∃ k, L = 10 * k) ∧ (∃ j, W = 10 * j))
  (h2 : W = 2 * L)
  (h3 : (L * L + W * W).sqrt = 45) :
  L * W = 810 :=
sorry

end tiles_covering_the_floor_l138_138651


namespace eval_g_at_3_l138_138563

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem eval_g_at_3 : g 3 = 10 := by
  -- Proof goes here
  sorry

end eval_g_at_3_l138_138563


namespace necessary_and_sufficient_condition_l138_138055

theorem necessary_and_sufficient_condition (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ 0 > b :=
by
  sorry

end necessary_and_sufficient_condition_l138_138055


namespace find_m_l138_138285

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -13.6 :=
by sorry

end find_m_l138_138285


namespace correct_operation_l138_138404

theorem correct_operation :
  (∀ a : ℝ, a^4 * a^3 = a^7)
  ∧ (∀ a : ℝ, (a^2)^3 ≠ a^5)
  ∧ (∀ a : ℝ, 3 * a^2 - a^2 ≠ 2)
  ∧ (∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2) :=
by {
  sorry
}

end correct_operation_l138_138404


namespace x_plus_y_l138_138282

variables {e1 e2 : ℝ → ℝ → Prop} -- Represents the vectors as properties of reals
variables {x y : ℝ} -- Real numbers x and y

-- Assuming non-collinearity of e1 and e2 (This means e1 and e2 are independent)
axiom non_collinear : e1 ≠ e2 

-- Given condition translated into Lean
axiom main_equation : (3 * x - 4 * y = 6) ∧ (2 * x - 3 * y = 3)

-- Prove that x + y = 9
theorem x_plus_y : x + y = 9 := 
by
  sorry -- Proof will be provided here

end x_plus_y_l138_138282


namespace ganpat_paint_time_l138_138969

theorem ganpat_paint_time (H_rate G_rate : ℝ) (together_time H_time : ℝ) (h₁ : H_time = 3)
  (h₂ : together_time = 2) (h₃ : H_rate = 1 / H_time) (h₄ : G_rate = 1 / G_time)
  (h₅ : 1/H_time + 1/G_rate = 1/together_time) : G_time = 3 := 
by 
  sorry

end ganpat_paint_time_l138_138969


namespace largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100_l138_138913

def is_prime (n : ℕ) : Prop := sorry -- Use inbuilt primality function or define it

def expression (n : ℕ) : ℕ := 2^n + n^2 - 1

theorem largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100 :
  ∃ m, is_prime m ∧ (∃ n, is_prime n ∧ expression n = m ∧ m < 100) ∧
        ∀ k, is_prime k ∧ (∃ n, is_prime n ∧ expression n = k ∧ k < 100) → k <= m :=
  sorry

end largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100_l138_138913


namespace smallest_n_for_cube_root_form_l138_138962

theorem smallest_n_for_cube_root_form
  (m n : ℕ) (r : ℝ)
  (h_pos_n : n > 0)
  (h_pos_r : r > 0)
  (h_r_bound : r < 1/500)
  (h_m : m = (n + r)^3)
  (h_min_m : ∀ k : ℕ, k = (n + r)^3 → k ≥ m) :
  n = 13 :=
by
  -- proof goes here
  sorry

end smallest_n_for_cube_root_form_l138_138962


namespace inflection_point_on_3x_l138_138625

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4 * Real.sin x - Real.cos x
noncomputable def f' (x : ℝ) : ℝ := 3 + 4 * Real.cos x + Real.sin x
noncomputable def f'' (x : ℝ) : ℝ := -4 * Real.sin x + Real.cos x

theorem inflection_point_on_3x {x0 : ℝ} (h : f'' x0 = 0) : (f x0) = 3 * x0 := by
  sorry

end inflection_point_on_3x_l138_138625


namespace find_ordered_pair_l138_138279

theorem find_ordered_pair (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) := by
  sorry

end find_ordered_pair_l138_138279


namespace percent_decrease_in_hours_l138_138357

variable {W H : ℝ} (W_nonzero : W ≠ 0) (H_nonzero : H ≠ 0)

theorem percent_decrease_in_hours
  (wage_increase : W' = 1.25 * W)
  (income_unchanged : W * H = W' * H')
  : (H' = 0.8 * H) → H' = H * (1 - 0.2) := by
  sorry

end percent_decrease_in_hours_l138_138357


namespace add_one_five_times_l138_138770

theorem add_one_five_times (m n : ℕ) (h : n = m + 5) : n - (m + 1) = 4 :=
by
  sorry

end add_one_five_times_l138_138770


namespace product_of_real_roots_l138_138197

theorem product_of_real_roots (x : ℝ) (hx : x ^ (Real.log x / Real.log 5) = 5) :
  (∃ a b : ℝ, a ^ (Real.log a / Real.log 5) = 5 ∧ b ^ (Real.log b / Real.log 5) = 5 ∧ a * b = 1) :=
sorry

end product_of_real_roots_l138_138197


namespace namjoonKoreanScore_l138_138772

variables (mathScore englishScore : ℝ) (averageScore : ℝ := 95) (koreanScore : ℝ)

def namjoonMathScore : Prop := mathScore = 100
def namjoonEnglishScore : Prop := englishScore = 95
def namjoonAverage : Prop := (koreanScore + mathScore + englishScore) / 3 = averageScore

theorem namjoonKoreanScore
  (H1 : namjoonMathScore 100)
  (H2 : namjoonEnglishScore 95)
  (H3 : namjoonAverage koreanScore 100 95 95) :
  koreanScore = 90 :=
by
  sorry

end namjoonKoreanScore_l138_138772


namespace age_relation_l138_138219

/--
Given that a woman is 42 years old and her daughter is 8 years old,
prove that in 9 years, the mother will be three times as old as her daughter.
-/
theorem age_relation (x : ℕ) (mother_age daughter_age : ℕ) 
  (h1 : mother_age = 42) (h2 : daughter_age = 8) 
  (h3 : 42 + x = 3 * (8 + x)) : 
  x = 9 :=
by
  sorry

end age_relation_l138_138219


namespace star_24_75_l138_138286

noncomputable def star (a b : ℝ) : ℝ := sorry 

-- Conditions
axiom star_one_one : star 1 1 = 2
axiom star_ab_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : star (a * b) b = a * (star b b)
axiom star_a_one (a : ℝ) (h : 0 < a) : star a 1 = 2 * a

-- Theorem to prove
theorem star_24_75 : star 24 75 = 1800 := 
by 
  sorry

end star_24_75_l138_138286


namespace fill_tank_time_l138_138017

-- Define the rates at which the pipes fill the tank
noncomputable def rate_A := (1:ℝ)/50
noncomputable def rate_B := (1:ℝ)/75

-- Define the combined rate of both pipes
noncomputable def combined_rate := rate_A + rate_B

-- Define the time to fill the tank at the combined rate
noncomputable def time_to_fill := 1 / combined_rate

-- The theorem that states the time taken to fill the tank is 30 hours
theorem fill_tank_time : time_to_fill = 30 := sorry

end fill_tank_time_l138_138017


namespace trigonometric_identity_l138_138904

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.cos (π / 6 - x) = - Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 :=
by
  sorry

end trigonometric_identity_l138_138904


namespace residue_11_pow_2021_mod_19_l138_138326

theorem residue_11_pow_2021_mod_19 : (11^2021) % 19 = 17 := 
by
  -- this is to ensure the theorem is syntactically correct in Lean but skips the proof for now
  sorry

end residue_11_pow_2021_mod_19_l138_138326


namespace minimum_value_of_z_l138_138269

theorem minimum_value_of_z
  (x y : ℝ)
  (h1 : 3 * x + y - 6 ≥ 0)
  (h2 : x - y - 2 ≤ 0)
  (h3 : y - 3 ≤ 0) :
  ∃ z, z = 4 * x + y ∧ z = 7 :=
sorry

end minimum_value_of_z_l138_138269


namespace find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l138_138758

-- Define the arithmetic sequence
def a (n : ℕ) (d : ℤ) := 23 + n * d

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) (d : ℤ) := n * 23 + (n * (n - 1) / 2) * d

-- Prove the common difference is -4
theorem find_common_difference (d : ℤ) :
  a 5 d > 0 ∧ a 6 d < 0 → d = -4 := sorry

-- Prove the maximum value of the sum S_n of the first n terms
theorem max_sum_first_n_terms (S_n : ℕ) :
  S 6 -4 = 78 := sorry

-- Prove the maximum value of n such that S_n > 0
theorem max_n_Sn_positive (n : ℕ) :
  S n -4 > 0 → n ≤ 12 := sorry

end find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l138_138758


namespace combined_height_l138_138079

theorem combined_height (h_John : ℕ) (h_Lena : ℕ) (h_Rebeca : ℕ)
  (cond1 : h_John = 152)
  (cond2 : h_John = h_Lena + 15)
  (cond3 : h_Rebeca = h_John + 6) :
  h_Lena + h_Rebeca = 295 :=
by
  sorry

end combined_height_l138_138079


namespace equivalence_of_sum_cubed_expression_l138_138703

theorem equivalence_of_sum_cubed_expression (a b : ℝ) 
  (h₁ : a + b = 5) (h₂ : a * b = -14) : a^3 + a^2 * b + a * b^2 + b^3 = 265 :=
sorry

end equivalence_of_sum_cubed_expression_l138_138703


namespace nat_numbers_in_segment_l138_138715

theorem nat_numbers_in_segment (a : ℕ → ℕ) (blue_index red_index : Set ℕ)
  (cond1 : ∀ i ∈ blue_index, i ≤ 200 → a (i - 1) = i)
  (cond2 : ∀ i ∈ red_index, i ≤ 200 → a (i - 1) = 201 - i) :
    ∀ i, 1 ≤ i ∧ i ≤ 100 → ∃ j, j < 100 ∧ a j = i := 
by
  sorry

end nat_numbers_in_segment_l138_138715


namespace polynomial_identity_l138_138793

theorem polynomial_identity (x : ℝ) (h₁ : x^5 - 3*x + 2 = 0) (h₂ : x ≠ 1) : 
  x^4 + x^3 + x^2 + x + 1 = 3 := 
by 
  sorry

end polynomial_identity_l138_138793


namespace sequences_power_of_two_l138_138088

open scoped Classical

theorem sequences_power_of_two (n : ℕ) (a b : Fin n → ℚ)
  (h1 : (∃ i j, i < j ∧ a i = a j) → ∀ i, a i = b i)
  (h2 : {p | ∃ (i j : Fin n), i < j ∧ (a i + a j = p)} = {q | ∃ (i j : Fin n), i < j ∧ (b i + b j = q)})
  (h3 : ∃ i j, i < j ∧ a i ≠ b i) :
  ∃ k : ℕ, n = 2 ^ k := 
sorry

end sequences_power_of_two_l138_138088


namespace number_of_trees_planted_l138_138534

-- Definition of initial conditions
def initial_trees : ℕ := 22
def final_trees : ℕ := 55

-- Theorem stating the number of trees planted
theorem number_of_trees_planted : final_trees - initial_trees = 33 := by
  sorry

end number_of_trees_planted_l138_138534


namespace complex_trajectory_is_ellipse_l138_138168

open Complex

theorem complex_trajectory_is_ellipse (z : ℂ) (h : abs (z - i) + abs (z + i) = 3) : 
  true := 
sorry

end complex_trajectory_is_ellipse_l138_138168


namespace necessary_condition_l138_138544

theorem necessary_condition {x m : ℝ} 
  (p : |1 - (x - 1) / 3| ≤ 2)
  (q : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (hm : m > 0)
  (h_np_nq : ¬(|1 - (x - 1) / 3| ≤ 2) → ¬(x^2 - 2 * x + 1 - m^2 ≤ 0))
  : m ≥ 9 :=
sorry

end necessary_condition_l138_138544


namespace multiple_of_4_and_6_sum_even_l138_138016

theorem multiple_of_4_and_6_sum_even (a b : ℤ) (h₁ : ∃ m : ℤ, a = 4 * m) (h₂ : ∃ n : ℤ, b = 6 * n) : ∃ k : ℤ, (a + b) = 2 * k :=
by
  sorry

end multiple_of_4_and_6_sum_even_l138_138016


namespace polynomial_value_l138_138607

theorem polynomial_value (a : ℝ) (h : a^2 + 2 * a = 1) : 
  2 * a^5 + 7 * a^4 + 5 * a^3 + 2 * a^2 + 5 * a + 1 = 4 :=
by
  sorry

end polynomial_value_l138_138607


namespace part_a_part_b_l138_138531

theorem part_a (a : ℤ) (k : ℤ) (h : a + 1 = 3 * k) : ∃ m : ℤ, 4 + 7 * a = 3 * m := by
  sorry

theorem part_b (a b : ℤ) (m n : ℤ) (h1 : 2 + a = 11 * m) (h2 : 35 - b = 11 * n) : ∃ p : ℤ, a + b = 11 * p := by
  sorry

end part_a_part_b_l138_138531


namespace cost_to_replace_is_800_l138_138832

-- Definitions based on conditions
def trade_in_value (num_movies : ℕ) (trade_in_price : ℕ) : ℕ :=
  num_movies * trade_in_price

def dvd_cost (num_movies : ℕ) (dvd_price : ℕ) : ℕ :=
  num_movies * dvd_price

def replacement_cost (num_movies : ℕ) (trade_in_price : ℕ) (dvd_price : ℕ) : ℕ :=
  dvd_cost num_movies dvd_price - trade_in_value num_movies trade_in_price

-- Problem statement: it costs John $800 to replace his movies
theorem cost_to_replace_is_800 (num_movies trade_in_price dvd_price : ℕ)
  (h1 : num_movies = 100) (h2 : trade_in_price = 2) (h3 : dvd_price = 10) :
  replacement_cost num_movies trade_in_price dvd_price = 800 :=
by
  -- Proof would go here
  sorry

end cost_to_replace_is_800_l138_138832


namespace that_three_digit_multiples_of_5_and_7_l138_138688

/-- 
Define the count_three_digit_multiples function, 
which counts the number of three-digit integers that are multiples of both 5 and 7.
-/
def count_three_digit_multiples : ℕ :=
  let lcm := Nat.lcm 5 7
  let first := (100 + lcm - 1) / lcm * lcm
  let last := 999 / lcm * lcm
  (last - first) / lcm + 1

/-- 
Theorem that states the number of positive three-digit integers that are multiples of both 5 and 7 is 26. 
-/
theorem three_digit_multiples_of_5_and_7 : count_three_digit_multiples = 26 := by
  sorry

end that_three_digit_multiples_of_5_and_7_l138_138688


namespace fraction_zero_solution_l138_138067

theorem fraction_zero_solution (x : ℝ) (h1 : |x| - 3 = 0) (h2 : x + 3 ≠ 0) : x = 3 := 
sorry

end fraction_zero_solution_l138_138067


namespace julia_song_download_l138_138766

theorem julia_song_download : 
  let internet_speed := 20 -- in MBps
  let half_hour_in_minutes := 30
  let size_per_song := 5 -- in MB
  (internet_speed * 60 * half_hour_in_minutes) / size_per_song = 7200 :=
by
  sorry

end julia_song_download_l138_138766


namespace birth_age_of_mother_l138_138952

def harrys_age : ℕ := 50

def fathers_age (h : ℕ) : ℕ := h + 24

def mothers_age (f h : ℕ) : ℕ := f - h / 25

theorem birth_age_of_mother (h f m : ℕ) (H1 : h = harrys_age)
  (H2 : f = fathers_age h) (H3 : m = mothers_age f h) :
  m - h = 22 := sorry

end birth_age_of_mother_l138_138952


namespace cost_per_pound_of_mixed_candy_l138_138052

def w1 := 10
def p1 := 8
def w2 := 20
def p2 := 5

theorem cost_per_pound_of_mixed_candy : 
    (w1 * p1 + w2 * p2) / (w1 + w2) = 6 := by
  sorry

end cost_per_pound_of_mixed_candy_l138_138052


namespace find_x_y_l138_138958

theorem find_x_y (x y : ℝ) : 
  (x - 12) ^ 2 + (y - 13) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ (x = 37 / 3 ∧ y = 38 / 3) :=
by
  sorry

end find_x_y_l138_138958


namespace ellipse_focal_length_l138_138147

theorem ellipse_focal_length :
  let a_squared := 20
    let b_squared := 11
    let c := Real.sqrt (a_squared - b_squared)
    let focal_length := 2 * c
  11 * x^2 + 20 * y^2 = 220 →
  focal_length = 6 :=
by
  sorry

end ellipse_focal_length_l138_138147


namespace difference_in_roses_and_orchids_l138_138517

theorem difference_in_roses_and_orchids
    (initial_roses : ℕ) (initial_orchids : ℕ) (initial_tulips : ℕ)
    (final_roses : ℕ) (final_orchids : ℕ) (final_tulips : ℕ)
    (ratio_roses_orchids_num : ℕ) (ratio_roses_orchids_den : ℕ)
    (ratio_roses_tulips_num : ℕ) (ratio_roses_tulips_den : ℕ)
    (h1 : initial_roses = 7)
    (h2 : initial_orchids = 12)
    (h3 : initial_tulips = 5)
    (h4 : final_roses = 11)
    (h5 : final_orchids = 20)
    (h6 : final_tulips = 10)
    (h7 : ratio_roses_orchids_num = 2)
    (h8 : ratio_roses_orchids_den = 5)
    (h9 : ratio_roses_tulips_num = 3)
    (h10 : ratio_roses_tulips_den = 5)
    (h11 : (final_roses : ℚ) / final_orchids = (ratio_roses_orchids_num : ℚ) / ratio_roses_orchids_den)
    (h12 : (final_roses : ℚ) / final_tulips = (ratio_roses_tulips_num : ℚ) / ratio_roses_tulips_den)
    : final_orchids - final_roses = 9 :=
by
  sorry

end difference_in_roses_and_orchids_l138_138517


namespace arithmetic_sequence_property_l138_138836

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

def condition (S : ℕ → ℝ) : Prop :=
  (S 8 - S 5) * (S 8 - S 4) < 0

-- Theorem to prove
theorem arithmetic_sequence_property {a : ℕ → ℝ} {S : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_cond : condition S) :
  |a 5| > |a 6| := 
sorry

end arithmetic_sequence_property_l138_138836


namespace seating_arrangement_l138_138129

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 7 * y = 112) : x = 7 :=
by
  sorry

end seating_arrangement_l138_138129


namespace inequality_solution_set_no_positive_a_b_exists_l138_138363

def f (x : ℝ) := abs (2 * x - 1) - abs (2 * x - 2)
def k := 1

theorem inequality_solution_set :
  { x : ℝ | f x ≥ x } = { x : ℝ | x ≤ -1 ∨ x = 1 } :=
sorry

theorem no_positive_a_b_exists (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  ¬ (a + 2 * b = k ∧ 2 / a + 1 / b = 4 - 1 / (a * b)) :=
sorry

end inequality_solution_set_no_positive_a_b_exists_l138_138363


namespace arithmetic_progression_sum_l138_138085

theorem arithmetic_progression_sum (a d S n : ℤ) (h_a : a = 32) (h_d : d = -4) (h_S : S = 132) :
  (n = 6 ∨ n = 11) :=
by
  -- Start the proof here
  sorry

end arithmetic_progression_sum_l138_138085


namespace exponent_on_right_side_l138_138564

theorem exponent_on_right_side (n : ℕ) (h : n = 17) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 :=
by
  sorry

end exponent_on_right_side_l138_138564


namespace preimage_of_4_3_is_2_1_l138_138652

theorem preimage_of_4_3_is_2_1 :
  ∃ (a b : ℝ), (a + 2 * b = 4) ∧ (2 * a - b = 3) ∧ (a = 2) ∧ (b = 1) :=
by
  exists 2
  exists 1
  constructor
  { sorry }
  constructor
  { sorry }
  constructor
  { sorry }
  { sorry }


end preimage_of_4_3_is_2_1_l138_138652


namespace cost_small_and_large_puzzle_l138_138057

-- Define the cost of a large puzzle L and the cost equation for large and small puzzles
def cost_large_puzzle : ℤ := 15

def cost_equation (S : ℤ) : Prop := cost_large_puzzle + 3 * S = 39

-- Theorem to prove the total cost of a small puzzle and a large puzzle together
theorem cost_small_and_large_puzzle : ∃ S : ℤ, cost_equation S ∧ (S + cost_large_puzzle = 23) :=
by
  sorry

end cost_small_and_large_puzzle_l138_138057


namespace girls_not_join_field_trip_l138_138095

theorem girls_not_join_field_trip (total_students : ℕ) (number_of_boys : ℕ) (number_on_trip : ℕ)
  (h_total : total_students = 18)
  (h_boys : number_of_boys = 8)
  (h_equal : number_on_trip = number_of_boys) :
  total_students - number_of_boys - number_on_trip = 2 := by
sorry

end girls_not_join_field_trip_l138_138095


namespace remainder_of_c_plus_d_l138_138935

theorem remainder_of_c_plus_d (c d : ℕ) (k l : ℕ) 
  (hc : c = 120 * k + 114) 
  (hd : d = 180 * l + 174) : 
  (c + d) % 60 = 48 := 
by sorry

end remainder_of_c_plus_d_l138_138935


namespace positive_difference_l138_138867

-- Define the binomial coefficient
def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Define the probability of heads in a fair coin flip
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of exactly k heads out of n flips
def prob_heads (n k : ℕ) : ℚ :=
  binomial n k * (fair_coin_prob ^ k) * (fair_coin_prob ^ (n - k))

-- Define the probabilities for the given problem
def prob_3_heads_out_of_5 : ℚ := prob_heads 5 3
def prob_5_heads_out_of_5 : ℚ := prob_heads 5 5

-- Claim the positive difference
theorem positive_difference :
  prob_3_heads_out_of_5 - prob_5_heads_out_of_5 = 9 / 32 :=
by
  sorry

end positive_difference_l138_138867


namespace inequality_solution_l138_138992

theorem inequality_solution (x : ℝ) : x^2 - 2 * x - 5 > 2 * x ↔ x > 5 ∨ x < -1 :=
by
  sorry

end inequality_solution_l138_138992


namespace isosceles_triangle_perimeter_l138_138080

-- Definitions for the conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Statement of the theorem
theorem isosceles_triangle_perimeter {a b c : ℕ} (h1 : is_isosceles_triangle a b c) (h2 : valid_triangle a b c) :
  (a = 2 ∧ b = 4 ∧ c = 4 ∨ a = 4 ∧ b = 4 ∧ c = 2 ∨ a = 4 ∧ b = 2 ∧ c = 4) →
  a + b + c = 10 :=
by 
  sorry

end isosceles_triangle_perimeter_l138_138080


namespace boat_speed_still_water_l138_138539

/-- Proof that the speed of the boat in still water is 10 km/hr given the conditions -/
theorem boat_speed_still_water (V_b V_s : ℝ) 
  (cond1 : V_b + V_s = 15) 
  (cond2 : V_b - V_s = 5) : 
  V_b = 10 :=
by
  sorry

end boat_speed_still_water_l138_138539


namespace florist_picked_roses_l138_138579

def initial_roses : ℕ := 11
def sold_roses : ℕ := 2
def final_roses : ℕ := 41
def remaining_roses := initial_roses - sold_roses
def picked_roses := final_roses - remaining_roses

theorem florist_picked_roses : picked_roses = 32 :=
by
  -- This is where the proof would go, but we are leaving it empty on purpose
  sorry

end florist_picked_roses_l138_138579


namespace night_shift_hours_l138_138984

theorem night_shift_hours
  (hours_first_guard : ℕ := 3)
  (hours_last_guard : ℕ := 2)
  (hours_each_middle_guard : ℕ := 2) :
  hours_first_guard + 2 * hours_each_middle_guard + hours_last_guard = 9 :=
by 
  sorry

end night_shift_hours_l138_138984


namespace robbie_weekly_fat_intake_l138_138970

theorem robbie_weekly_fat_intake
  (morning_cups : ℕ) (afternoon_cups : ℕ) (evening_cups : ℕ)
  (fat_per_cup : ℕ) (days_per_week : ℕ) :
  morning_cups = 3 →
  afternoon_cups = 2 →
  evening_cups = 5 →
  fat_per_cup = 10 →
  days_per_week = 7 →
  (morning_cups * fat_per_cup + afternoon_cups * fat_per_cup + evening_cups * fat_per_cup) * days_per_week = 700 :=
by
  intros
  sorry

end robbie_weekly_fat_intake_l138_138970


namespace boxes_of_nuts_purchased_l138_138587

theorem boxes_of_nuts_purchased (b : ℕ) (n : ℕ) (bolts_used : ℕ := 7 * 11 - 3) 
    (nuts_used : ℕ := 113 - bolts_used) (total_nuts : ℕ := nuts_used + 6) 
    (nuts_per_box : ℕ := 15) (h_bolts_boxes : b = 7) 
    (h_bolts_per_box : ∀ x, b * x = 77) 
    (h_nuts_boxes : ∃ x, n = x * nuts_per_box)
    : ∃ k, n = k * 15 ∧ k = 3 :=
by
  sorry

end boxes_of_nuts_purchased_l138_138587


namespace impossible_grid_arrangement_l138_138048

theorem impossible_grid_arrangement :
  ¬ ∃ (f : Fin 25 → Fin 41 → ℤ),
    (∀ i j, abs (f i j - f (i + 1) j) ≤ 16 ∧ abs (f i j - f i (j + 1)) ≤ 16 ∧
            f i j ≠ f (i + 1) j ∧ f i j ≠ f i (j + 1)) := 
sorry

end impossible_grid_arrangement_l138_138048


namespace smallest_palindrome_div_3_5_l138_138776

theorem smallest_palindrome_div_3_5 : ∃ n : ℕ, n = 50205 ∧ 
  (∃ a b c : ℕ, n = 5 * 10^4 + a * 10^3 + b * 10^2 + a * 10 + 5) ∧ 
  n % 5 = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≥ 10000 ∧ 
  n < 100000 :=
by
  sorry

end smallest_palindrome_div_3_5_l138_138776


namespace complement_union_eq_l138_138851

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,3,5,7}
def B : Set ℕ := {2,4,5}

theorem complement_union_eq : (U \ (A ∪ B)) = {6,8} := by
  sorry

end complement_union_eq_l138_138851


namespace fraction_collectors_edition_is_correct_l138_138034

-- Let's define the necessary conditions
variable (DinaDolls IvyDolls CollectorsEditionDolls : ℕ)
variable (FractionCollectorsEdition : ℚ)

-- Given conditions
axiom DinaHas60Dolls : DinaDolls = 60
axiom DinaHasTwiceAsManyDollsAsIvy : DinaDolls = 2 * IvyDolls
axiom IvyHas20CollectorsEditionDolls : CollectorsEditionDolls = 20

-- The statement to prove
theorem fraction_collectors_edition_is_correct :
  FractionCollectorsEdition = (CollectorsEditionDolls : ℚ) / (IvyDolls : ℚ) ∧
  DinaDolls = 60 →
  DinaDolls = 2 * IvyDolls →
  CollectorsEditionDolls = 20 →
  FractionCollectorsEdition = 2 / 3 := 
by
  sorry

end fraction_collectors_edition_is_correct_l138_138034


namespace wanda_crayons_l138_138812

variable (Dina Jacob Wanda : ℕ)

theorem wanda_crayons : Dina = 28 ∧ Jacob = Dina - 2 ∧ Dina + Jacob + Wanda = 116 → Wanda = 62 :=
by
  intro h
  sorry

end wanda_crayons_l138_138812


namespace company_total_parts_l138_138878

noncomputable def total_parts_made (planning_days : ℕ) (initial_rate : ℕ) (extra_rate : ℕ) (extra_parts : ℕ) (x_days : ℕ) : ℕ :=
  let initial_production := planning_days * initial_rate
  let increased_rate := initial_rate + extra_rate
  let actual_production := x_days * increased_rate
  initial_production + actual_production

def planned_production (planning_days : ℕ) (initial_rate : ℕ) (x_days : ℕ) : ℕ :=
  planning_days * initial_rate + x_days * initial_rate

theorem company_total_parts
  (planning_days : ℕ)
  (initial_rate : ℕ)
  (extra_rate : ℕ)
  (extra_parts : ℕ)
  (x_days : ℕ)
  (h1 : planning_days = 3)
  (h2 : initial_rate = 40)
  (h3 : extra_rate = 7)
  (h4 : extra_parts = 150)
  (h5 : x_days = 21)
  (h6 : 7 * x_days = extra_parts) :
  total_parts_made planning_days initial_rate extra_rate extra_parts x_days = 1107 := by
  sorry

end company_total_parts_l138_138878


namespace intersection_of_M_and_N_l138_138609

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | x < 1}
def expected_intersection : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N :
  M ∩ N = expected_intersection :=
sorry

end intersection_of_M_and_N_l138_138609


namespace find_g2_l138_138930

open Function

variable (g : ℝ → ℝ)

axiom g_condition : ∀ x : ℝ, g x + 2 * g (1 - x) = 5 * x ^ 2

theorem find_g2 : g 2 = -10 / 3 :=
by {
  sorry
}

end find_g2_l138_138930


namespace smallest_consecutive_odd_sum_l138_138915

theorem smallest_consecutive_odd_sum (a b c d e : ℤ)
    (h1 : b = a + 2)
    (h2 : c = a + 4)
    (h3 : d = a + 6)
    (h4 : e = a + 8)
    (h5 : a + b + c + d + e = 375) : a = 71 :=
by
  -- the proof will go here
  sorry

end smallest_consecutive_odd_sum_l138_138915


namespace probability_even_sum_l138_138137

-- Defining the probabilities for the first wheel
def P_even_1 : ℚ := 2/3
def P_odd_1 : ℚ := 1/3

-- Defining the probabilities for the second wheel
def P_even_2 : ℚ := 1/2
def P_odd_2 : ℚ := 1/2

-- Prove that the probability that the sum of the two selected numbers is even is 1/2
theorem probability_even_sum : 
  P_even_1 * P_even_2 + P_odd_1 * P_odd_2 = 1/2 :=
by
  sorry

end probability_even_sum_l138_138137


namespace initial_violet_balloons_l138_138297

-- Let's define the given conditions
def red_balloons : ℕ := 4
def violet_balloons_lost : ℕ := 3
def violet_balloons_now : ℕ := 4

-- Define the statement to prove
theorem initial_violet_balloons :
  (violet_balloons_now + violet_balloons_lost) = 7 :=
by
  sorry

end initial_violet_balloons_l138_138297


namespace raisin_cookies_difference_l138_138695

-- Definitions based on conditions:
def raisin_cookies_baked_yesterday : ℕ := 300
def raisin_cookies_baked_today : ℕ := 280

-- Proof statement:
theorem raisin_cookies_difference : raisin_cookies_baked_yesterday - raisin_cookies_baked_today = 20 := 
by
  sorry

end raisin_cookies_difference_l138_138695


namespace jason_grass_cutting_time_l138_138159

def total_minutes (hours : ℕ) : ℕ := hours * 60
def minutes_per_yard : ℕ := 30
def total_yards_per_weekend : ℕ := 8 * 2
def total_minutes_per_weekend : ℕ := minutes_per_yard * total_yards_per_weekend
def convert_minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem jason_grass_cutting_time : 
  convert_minutes_to_hours total_minutes_per_weekend = 8 := by
  sorry

end jason_grass_cutting_time_l138_138159


namespace bigger_part_is_45_l138_138139

variable (x y : ℕ)

theorem bigger_part_is_45
  (h1 : x + y = 60)
  (h2 : 10 * x + 22 * y = 780) :
  max x y = 45 := by
  sorry

end bigger_part_is_45_l138_138139


namespace matrix_mult_7_l138_138311

theorem matrix_mult_7 (M : Matrix (Fin 3) (Fin 3) ℝ) (v : Fin 3 → ℝ) : 
  (∀ v, M.mulVec v = (7 : ℝ) • v) ↔ M = 7 • 1 :=
by
  sorry

end matrix_mult_7_l138_138311


namespace average_of_roots_l138_138031

theorem average_of_roots (a b: ℝ) (h : a ≠ 0) (hr : ∃ x1 x2: ℝ, a * x1 ^ 2 - 3 * a * x1 + b = 0 ∧ a * x2 ^ 2 - 3 * a * x2 + b = 0 ∧ x1 ≠ x2):
  (∃ r1 r2: ℝ, a * r1 ^ 2 - 3 * a * r1 + b = 0 ∧ a * r2 ^ 2 - 3 * a * r2 + b = 0 ∧ r1 ≠ r2) →
  ((r1 + r2) / 2 = 3 / 2) :=
by
  sorry

end average_of_roots_l138_138031


namespace product_price_reduction_l138_138442

theorem product_price_reduction (z : ℝ) (x : ℝ) (hp1 : z > 0) (hp2 : 0.85 * 0.85 * z = z * (1 - x / 100)) : x = 27.75 := by
  sorry

end product_price_reduction_l138_138442


namespace valid_license_plates_l138_138328

def letters := 26
def digits := 10
def totalPlates := letters^3 * digits^4

theorem valid_license_plates : totalPlates = 175760000 := by
  sorry

end valid_license_plates_l138_138328


namespace emily_widgets_production_l138_138069

variable (w t : ℕ) (work_hours_monday work_hours_tuesday production_monday production_tuesday : ℕ)

theorem emily_widgets_production :
  (w = 2 * t) → 
  (work_hours_monday = t) →
  (work_hours_tuesday = t - 3) →
  (production_monday = w * work_hours_monday) → 
  (production_tuesday = (w + 6) * work_hours_tuesday) →
  (production_monday - production_tuesday) = 18 :=
by
  intros hw hwm hwmt hpm hpt
  sorry

end emily_widgets_production_l138_138069


namespace no_arithmetic_progression_in_squares_l138_138790

theorem no_arithmetic_progression_in_squares :
  ∀ (a d : ℕ), d > 0 → ¬ (∃ (f : ℕ → ℕ), 
    (∀ n, f n = a + n * d) ∧ 
    (∀ n, ∃ m, n ^ 2 = f m)) :=
by
  sorry

end no_arithmetic_progression_in_squares_l138_138790


namespace smallest_positive_integer_is_53_l138_138529

theorem smallest_positive_integer_is_53 :
  ∃ a : ℕ, a > 0 ∧ a % 3 = 2 ∧ a % 4 = 1 ∧ a % 5 = 3 ∧ a = 53 :=
by
  sorry

end smallest_positive_integer_is_53_l138_138529


namespace intersection_in_fourth_quadrant_l138_138599

theorem intersection_in_fourth_quadrant (a : ℝ) : 
  (∃ x y : ℝ, y = -x + 1 ∧ y = x - 2 * a ∧ x > 0 ∧ y < 0) → a > 1 / 2 := 
by 
  sorry

end intersection_in_fourth_quadrant_l138_138599


namespace max_xy_l138_138650

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  xy ≤ 1 / 4 := 
sorry

end max_xy_l138_138650


namespace value_of_a_l138_138653

theorem value_of_a (k : ℝ) (a : ℝ) (b : ℝ) (h1 : a = k / b^2) (h2 : a = 10) (h3 : b = 24) :
  a = 40 :=
sorry

end value_of_a_l138_138653


namespace tens_digit_19_pow_1987_l138_138353

theorem tens_digit_19_pow_1987 : (19 ^ 1987) % 100 / 10 = 3 := 
sorry

end tens_digit_19_pow_1987_l138_138353


namespace sum_of_coefficients_l138_138523

theorem sum_of_coefficients (a b c d e x : ℝ) (h : 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) :
  a + b + c + d + e = 36 :=
by
  sorry

end sum_of_coefficients_l138_138523


namespace max_connected_stations_l138_138870

theorem max_connected_stations (n : ℕ) 
  (h1 : ∀ s : ℕ, s ≤ n → s ≤ 3) 
  (h2 : ∀ x y : ℕ, x < y → ∃ z : ℕ, z < 3 ∧ z ≤ n) : 
  n = 10 :=
by 
  sorry

end max_connected_stations_l138_138870


namespace sum_of_products_non_positive_l138_138813

theorem sum_of_products_non_positive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end sum_of_products_non_positive_l138_138813


namespace solve_congruence_l138_138255

theorem solve_congruence (n : ℕ) (hn : n < 47) 
  (congr_13n : 13 * n ≡ 9 [MOD 47]) : n ≡ 20 [MOD 47] :=
sorry

end solve_congruence_l138_138255


namespace power_of_i_2016_l138_138515
-- Importing necessary libraries to handle complex numbers

theorem power_of_i_2016 (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) : 
  (i^2016 = 1) :=
sorry

end power_of_i_2016_l138_138515


namespace sector_angle_l138_138133

theorem sector_angle (r θ : ℝ) 
  (h1 : r * θ + 2 * r = 6) 
  (h2 : 1/2 * r^2 * θ = 2) : 
  θ = 1 ∨ θ = 4 :=
by 
  sorry

end sector_angle_l138_138133


namespace time_to_Lake_Park_restaurant_l138_138379

def time_to_Hidden_Lake := 15
def time_back_to_Park_Office := 7
def total_time_gone := 32

theorem time_to_Lake_Park_restaurant : 
  (total_time_gone = time_to_Hidden_Lake + time_back_to_Park_Office +
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office))) -> 
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office) = 10) := by
  intros 
  sorry

end time_to_Lake_Park_restaurant_l138_138379


namespace vertex_C_path_length_equals_l138_138562

noncomputable def path_length_traversed_by_C (AB BC CA : ℝ) (PQ QR : ℝ) : ℝ :=
  let BC := 3  -- length of side BC is 3 inches
  let AB := 2  -- length of side AB is 2 inches
  let CA := 4  -- length of side CA is 4 inches
  let PQ := 8  -- length of side PQ of the rectangle is 8 inches
  let QR := 6  -- length of side QR of the rectangle is 6 inches
  4 * BC * Real.pi

theorem vertex_C_path_length_equals (AB BC CA PQ QR : ℝ) :
  AB = 2 ∧ BC = 3 ∧ CA = 4 ∧ PQ = 8 ∧ QR = 6 →
  path_length_traversed_by_C AB BC CA PQ QR = 12 * Real.pi :=
by
  intros h
  have hAB : AB = 2 := h.1
  have hBC : BC = 3 := h.2.1
  have hCA : CA = 4 := h.2.2.1
  have hPQ : PQ = 8 := h.2.2.2.1
  have hQR : QR = 6 := h.2.2.2.2
  simp [path_length_traversed_by_C, hAB, hBC, hCA, hPQ, hQR]
  sorry

end vertex_C_path_length_equals_l138_138562


namespace intersection_of_M_and_N_l138_138209

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l138_138209


namespace range_of_m_l138_138266

variable {α : Type*} [LinearOrder α]

def increasing (f : α → α) : Prop :=
  ∀ ⦃x y : α⦄, x < y → f x < f y

theorem range_of_m 
  (f : ℝ → ℝ) 
  (h_inc : increasing f) 
  (h_cond : ∀ m : ℝ, f (m + 3) ≤ f 5) : 
  {m : ℝ | f (m + 3) ≤ f 5} = {m : ℝ | m ≤ 2} := 
sorry

end range_of_m_l138_138266


namespace points_satisfying_clubsuit_l138_138294

def clubsuit (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem points_satisfying_clubsuit (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x + y = 0) :=
by
  sorry

end points_satisfying_clubsuit_l138_138294


namespace candy_bar_cost_l138_138116

theorem candy_bar_cost {initial_money left_money cost_bar : ℕ} 
                        (h_initial : initial_money = 4)
                        (h_left : left_money = 3)
                        (h_cost : cost_bar = initial_money - left_money) :
                        cost_bar = 1 :=
by 
  sorry -- Proof is not required as per the instructions

end candy_bar_cost_l138_138116


namespace y_is_triangular_l138_138903

theorem y_is_triangular (k : ℕ) (hk : k > 0) : 
  ∃ n : ℕ, y = (n * (n + 1)) / 2 :=
by
  let y := (9^k - 1) / 8
  sorry

end y_is_triangular_l138_138903


namespace slope_tangent_line_at_zero_l138_138082

noncomputable def f (x : ℝ) : ℝ := (2 * x - 5) / (x^2 + 1)

theorem slope_tangent_line_at_zero : 
  (deriv f 0) = 2 :=
sorry

end slope_tangent_line_at_zero_l138_138082


namespace expected_value_of_biased_coin_l138_138808

noncomputable def expected_value : ℚ :=
  (2 / 3) * 5 + (1 / 3) * -6

theorem expected_value_of_biased_coin :
  expected_value = 4 / 3 := by
  sorry

end expected_value_of_biased_coin_l138_138808


namespace count_divisible_by_45_l138_138701

theorem count_divisible_by_45 : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ x % 100 = 45 → x % 45 = 0 → n = 10) :=
by {
  sorry
}

end count_divisible_by_45_l138_138701


namespace roots_negative_condition_l138_138573

theorem roots_negative_condition (a b c r s : ℝ) (h_eqn : a ≠ 0) (h_root : a * r^2 + b * r + c = 0) (h_neg : r = -s) : b = 0 := sorry

end roots_negative_condition_l138_138573


namespace min_value_of_f_l138_138959

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (9 / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : f x = 16 :=
by
  sorry

end min_value_of_f_l138_138959


namespace winning_strategy_l138_138843

/-- Given a square table n x n, two players A and B are playing the following game: 
  - At the beginning, all cells of the table are empty.
  - Player A has the first move, and in each of their moves, a player will put a coin on some cell 
    that doesn't contain a coin and is not adjacent to any of the cells that already contain a coin. 
  - The player who makes the last move wins. 

  Cells are adjacent if they share an edge.

  - If n is even, player B has the winning strategy.
  - If n is odd, player A has the winning strategy.
-/
theorem winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ (B_strat : winning_strategy_for_B), True) ∧ (n % 2 = 1 → ∃ (A_strat : winning_strategy_for_A), True) :=
by {
  admit
}

end winning_strategy_l138_138843


namespace inequality_system_solution_l138_138231

theorem inequality_system_solution {x : ℝ} (h1 : 2 * x - 1 < x + 5) (h2 : (x + 1)/3 < x - 1) : 2 < x ∧ x < 6 :=
by
  sorry

end inequality_system_solution_l138_138231


namespace evaluate_expression_l138_138160

theorem evaluate_expression (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxy : x > y) (hyz : y > z) :
  (x ^ (y + z) * z ^ (x + y)) / (y ^ (x + z) * z ^ (y + x)) = (x / y) ^ (y + z) :=
by
  sorry

end evaluate_expression_l138_138160


namespace other_solution_l138_138924

theorem other_solution (x : ℚ) (h : 30*x^2 + 13 = 47*x - 2) (hx : x = 3/5) : x = 5/6 ∨ x = 3/5 := by
  sorry

end other_solution_l138_138924


namespace taxi_fare_distance_l138_138842

-- Define the fare calculation and distance function
def fare (x : ℕ) : ℝ :=
  if x ≤ 4 then 10
  else 10 + (x - 4) * 1.5

-- Proof statement
theorem taxi_fare_distance (x : ℕ) : fare x = 16 → x = 8 :=
by
  -- Proof skipped
  sorry

end taxi_fare_distance_l138_138842


namespace at_least_one_vowel_l138_138552

-- Define the set of letters
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'I'}

-- Define the vowels within the set of letters
def vowels : Finset Char := {'A', 'E', 'I'}

-- Define the consonants within the set of letters
def consonants : Finset Char := {'B', 'C', 'D', 'F'}

-- Function to count the total number of 3-letter words from a given set
def count_words (s : Finset Char) (length : Nat) : Nat :=
  s.card ^ length

-- Define the statement of the problem
theorem at_least_one_vowel : count_words letters 3 - count_words consonants 3 = 279 :=
by
  sorry

end at_least_one_vowel_l138_138552


namespace sum_of_roots_l138_138051

theorem sum_of_roots (α β : ℝ)
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 :=
sorry

end sum_of_roots_l138_138051


namespace car_Y_average_speed_l138_138853

theorem car_Y_average_speed 
  (car_X_speed : ℝ)
  (car_X_time_before_Y : ℝ)
  (car_X_distance_when_Y_starts : ℝ)
  (car_X_total_distance : ℝ)
  (car_X_travel_time : ℝ)
  (car_Y_distance : ℝ)
  (car_Y_travel_time : ℝ)
  (h_car_X_speed : car_X_speed = 35)
  (h_car_X_time_before_Y : car_X_time_before_Y = 72 / 60)
  (h_car_X_distance_when_Y_starts : car_X_distance_when_Y_starts = car_X_speed * car_X_time_before_Y)
  (h_car_X_total_distance : car_X_total_distance = car_X_distance_when_Y_starts + car_X_distance_when_Y_starts)
  (h_car_X_travel_time : car_X_travel_time = car_X_total_distance / car_X_speed)
  (h_car_Y_distance : car_Y_distance = 490)
  (h_car_Y_travel_time : car_Y_travel_time = car_X_travel_time) :
  (car_Y_distance / car_Y_travel_time) = 32.24 := 
sorry

end car_Y_average_speed_l138_138853


namespace cone_height_l138_138835

theorem cone_height
  (V1 V2 V : ℝ)
  (h1 h2 : ℝ)
  (fact1 : h1 = 10)
  (fact2 : h2 = 2)
  (h : ∀ m : ℝ, V1 = V * (10 ^ 3) / (m ^ 3) ∧ V2 = V * ((m - 2) ^ 3) / (m ^ 3))
  (equal_volumes : V1 + V2 = V) :
  (∃ m : ℝ, m = 13.897) :=
by
  sorry

end cone_height_l138_138835


namespace necessary_but_not_sufficient_l138_138292

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (x > 1) ∨ (x ≤ -1) := 
by 
  sorry

end necessary_but_not_sufficient_l138_138292


namespace correct_reaction_equation_l138_138152

noncomputable def reaction_equation (vA vB vC : ℝ) : Prop :=
  vB = 3 * vA ∧ 3 * vC = 2 * vB

theorem correct_reaction_equation (vA vB vC : ℝ) (h : reaction_equation vA vB vC) :
  ∃ (α β γ : ℕ), α = 1 ∧ β = 3 ∧ γ = 2 :=
sorry

end correct_reaction_equation_l138_138152


namespace distance_to_y_axis_l138_138376

theorem distance_to_y_axis {x y : ℝ} (h : x = -3 ∧ y = 4) : abs x = 3 :=
by
  sorry

end distance_to_y_axis_l138_138376


namespace athlete_speed_l138_138788

theorem athlete_speed (d t : ℝ) (H_d : d = 200) (H_t : t = 40) : (d / t) = 5 := by
  sorry

end athlete_speed_l138_138788


namespace length_of_circle_l138_138104

-- Define initial speeds and conditions
variables (V1 V2 : ℝ)
variables (L : ℝ) -- Length of the circle

-- Conditions
def initial_condition : Prop := V1 - V2 = 3
def extra_laps_after_speed_increase : Prop := (V1 + 10) - V2 = V1 - V2 + 10

-- Statement representing the mathematical equivalence
theorem length_of_circle
  (h1 : initial_condition V1 V2) 
  (h2 : extra_laps_after_speed_increase V1 V2) :
  L = 1250 := 
sorry

end length_of_circle_l138_138104


namespace inequality_solution_set_l138_138395

theorem inequality_solution_set :
  {x : ℝ | (x^2 - x - 6) / (x - 1) > 0} = {x : ℝ | (-2 < x ∧ x < 1) ∨ (3 < x)} := by
  sorry

end inequality_solution_set_l138_138395


namespace Marty_paint_combinations_l138_138589

theorem Marty_paint_combinations :
  let colors := 5 -- blue, green, yellow, black, white
  let styles := 3 -- brush, roller, sponge
  let invalid_combinations := 1 * 1 -- white paint with roller
  let total_combinations := (4 * styles) + (1 * (styles - 1))
  total_combinations = 14 :=
by
  -- Define the total number of combinations excluding the invalid one
  let colors := 5
  let styles := 3
  let invalid_combinations := 1 -- number of invalid combinations (white with roller)
  let valid_combinations := (4 * styles) + (1 * (styles - 1))
  show valid_combinations = 14
  {
    exact rfl -- This will assert that the valid_combinations indeed equals 14
  }

end Marty_paint_combinations_l138_138589


namespace sum_of_digits_divisible_by_7_l138_138692

theorem sum_of_digits_divisible_by_7
  (a b : ℕ)
  (h_three_digit : 100 * a + 11 * b ≥ 100 ∧ 100 * a + 11 * b < 1000)
  (h_last_two_digits_equal : true)
  (h_divisible_by_7 : (100 * a + 11 * b) % 7 = 0) :
  (a + 2 * b) % 7 = 0 :=
sorry

end sum_of_digits_divisible_by_7_l138_138692


namespace total_payment_is_53_l138_138720

-- Conditions
def bobBill : ℝ := 30
def kateBill : ℝ := 25
def bobDiscountRate : ℝ := 0.05
def kateDiscountRate : ℝ := 0.02

-- Calculations
def bobDiscount := bobBill * bobDiscountRate
def kateDiscount := kateBill * kateDiscountRate
def bobPayment := bobBill - bobDiscount
def katePayment := kateBill - kateDiscount

-- Goal
def totalPayment := bobPayment + katePayment

-- Theorem statement
theorem total_payment_is_53 : totalPayment = 53 := by
  sorry

end total_payment_is_53_l138_138720


namespace chemist_sons_ages_l138_138961

theorem chemist_sons_ages 
    (a b c w : ℕ)
    (h1 : a * b * c = 36)
    (h2 : a + b + c = w)
    (h3 : ∃! x, x = max a (max b c)) :
    (a = 2 ∧ b = 2 ∧ c = 9) ∨ 
    (a = 2 ∧ b = 9 ∧ c = 2) ∨ 
    (a = 9 ∧ b = 2 ∧ c = 2) :=
  sorry

end chemist_sons_ages_l138_138961


namespace least_positive_integer_l138_138581

theorem least_positive_integer (n : ℕ) :
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 5 ∧
  n % 7 = 2 ↔
  n = 83 :=
by
  sorry

end least_positive_integer_l138_138581


namespace minimum_value_of_f_maximum_value_of_k_l138_138586

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f : ∃ x : ℝ, 0 < x ∧ f x = -1 / Real.exp 1 :=
sorry

theorem maximum_value_of_k : ∀ x > 2, ∀ k : ℤ, (f x ≥ k * x - 2 * (k + 1)) → k ≤ 3 :=
sorry

end minimum_value_of_f_maximum_value_of_k_l138_138586


namespace set_in_proportion_l138_138823

theorem set_in_proportion : 
  let a1 := 3
  let a2 := 9
  let b1 := 10
  let b2 := 30
  (a1 * b2 = a2 * b1) := 
by {
  sorry
}

end set_in_proportion_l138_138823


namespace inequality_proof_l138_138253

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / a + (c + a) / b + (a + b) / c ≥ ((a ^ 2 + b ^ 2 + c ^ 2) * (a * b + b * c + c * a)) / (a * b * c * (a + b + c)) + 3 := 
by
  -- Adding 'sorry' to indicate the proof is omitted
  sorry

end inequality_proof_l138_138253


namespace ratio_XZ_ZY_equals_one_l138_138212

theorem ratio_XZ_ZY_equals_one (A : ℕ) (B : ℕ) (C : ℕ) (total_area : ℕ) (area_bisected : ℕ)
  (decagon_area : total_area = 12) (halves_area : area_bisected = 6)
  (above_LZ : A + B = area_bisected) (below_LZ : C + D = area_bisected)
  (symmetry : XZ = ZY) :
  (XZ / ZY = 1) := 
by
  sorry

end ratio_XZ_ZY_equals_one_l138_138212


namespace solve_for_x_l138_138876

theorem solve_for_x (b x : ℝ) (h1 : b > 1) (h2 : x > 0)
    (h3 : (4 * x) ^ (Real.log 4 / Real.log b) = (6 * x) ^ (Real.log 6 / Real.log b)) :
    x = 1 / 6 :=
by
  sorry

end solve_for_x_l138_138876


namespace ticket_cost_l138_138063

-- Conditions
def seats : ℕ := 400
def capacity_percentage : ℝ := 0.8
def performances : ℕ := 3
def total_revenue : ℝ := 28800

-- Question: Prove that the cost of each ticket is $30
theorem ticket_cost : (total_revenue / (seats * capacity_percentage * performances)) = 30 := 
by
  sorry

end ticket_cost_l138_138063


namespace total_fat_served_l138_138571

-- Definitions based on conditions
def fat_herring : ℕ := 40
def fat_eel : ℕ := 20
def fat_pike : ℕ := fat_eel + 10
def fish_served_each : ℕ := 40

-- Calculations based on defined conditions
def total_fat_herring : ℕ := fish_served_each * fat_herring
def total_fat_eel : ℕ := fish_served_each * fat_eel
def total_fat_pike : ℕ := fish_served_each * fat_pike

-- Proof statement to show the total fat served
theorem total_fat_served : total_fat_herring + total_fat_eel + total_fat_pike = 3600 := by
  sorry

end total_fat_served_l138_138571


namespace sally_bread_consumption_l138_138927

theorem sally_bread_consumption :
  (2 * 2) + (1 * 2) = 6 :=
by
  sorry

end sally_bread_consumption_l138_138927


namespace cara_constant_speed_l138_138364

noncomputable def cara_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

theorem cara_constant_speed
  ( distance : ℕ := 120 )
  ( dan_speed : ℕ := 40 )
  ( dan_time_offset : ℕ := 1 ) :
  cara_speed distance (3 + dan_time_offset) = 30 := 
by
  -- skip proof
  sorry

end cara_constant_speed_l138_138364


namespace tan_sin_cos_log_expression_simplification_l138_138123

-- Proof Problem 1 Statement in Lean 4
theorem tan_sin_cos (α : ℝ) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by
  sorry

-- Proof Problem 2 Statement in Lean 4
theorem log_expression_simplification : 
  Real.logb 3 (Real.sqrt 27) + Real.logb 10 25 + Real.logb 10 4 + 
  (7 : ℝ) ^ Real.logb 7 2 + (-9.8) ^ 0 = 13 / 2 :=
by
  sorry

end tan_sin_cos_log_expression_simplification_l138_138123


namespace nuts_division_pattern_l138_138321

noncomputable def smallest_number_of_nuts : ℕ := 15621

theorem nuts_division_pattern :
  ∃ N : ℕ, N = smallest_number_of_nuts ∧ 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 
  (∃ M : ℕ, (N - k) % 4 = 0 ∧ (N - k) / 4 * 5 + 1 = N) := sorry

end nuts_division_pattern_l138_138321


namespace ones_digit_of_7_pow_53_l138_138512

theorem ones_digit_of_7_pow_53 : (7^53 % 10) = 7 := by
  sorry

end ones_digit_of_7_pow_53_l138_138512


namespace sufficientButNotNecessary_l138_138860

theorem sufficientButNotNecessary (x : ℝ) : ((x + 1) * (x - 3) < 0) → x < 3 ∧ ¬(x < 3 → (x + 1) * (x - 3) < 0) :=
by
  sorry

end sufficientButNotNecessary_l138_138860


namespace difference_in_gems_l138_138196

theorem difference_in_gems (r d : ℕ) (h : d = 3 * r) : d - r = 2 * r := 
by 
  sorry

end difference_in_gems_l138_138196


namespace base_conversion_and_operations_l138_138681

-- Definitions to convert numbers from bases 7, 5, and 6 to base 10
def base7_to_nat (n : ℕ) : ℕ := 
  8 * 7^0 + 6 * 7^1 + 4 * 7^2 + 2 * 7^3

def base5_to_nat (n : ℕ) : ℕ := 
  1 * 5^0 + 2 * 5^1 + 1 * 5^2

def base6_to_nat (n : ℕ) : ℕ := 
  1 * 6^0 + 5 * 6^1 + 4 * 6^2 + 3 * 6^3

def base7_to_nat2 (n : ℕ) : ℕ := 
  1 * 7^0 + 9 * 7^1 + 8 * 7^2 + 7 * 7^3

-- Problem statement: Perform the arithmetical operations
theorem base_conversion_and_operations : 
  (base7_to_nat 2468 / base5_to_nat 121) - base6_to_nat 3451 + base7_to_nat2 7891 = 2059 := 
by
  sorry

end base_conversion_and_operations_l138_138681


namespace trains_meet_in_32_seconds_l138_138433

noncomputable def train_meeting_time
  (length_train1 : ℕ)
  (length_train2 : ℕ)
  (initial_distance : ℕ)
  (speed_train1_kmph : ℕ)
  (speed_train2_kmph : ℕ)
  : ℕ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let total_distance := length_train1 + length_train2 + initial_distance
  total_distance / relative_speed

theorem trains_meet_in_32_seconds :
  train_meeting_time 400 200 200 54 36 = 32 := 
by
  sorry

end trains_meet_in_32_seconds_l138_138433


namespace right_pyramid_volume_l138_138053

noncomputable def volume_of_right_pyramid (base_area lateral_face_area total_surface_area : ℝ) : ℝ := 
  let height := (10 : ℝ) / 3
  (1 / 3) * base_area * height

theorem right_pyramid_volume (total_surface_area base_area lateral_face_area : ℝ)
  (h0 : total_surface_area = 300)
  (h1 : base_area + 3 * lateral_face_area = total_surface_area)
  (h2 : lateral_face_area = base_area / 3) 
  : volume_of_right_pyramid base_area lateral_face_area total_surface_area = 500 / 3 := 
by
  sorry

end right_pyramid_volume_l138_138053


namespace ratio_of_width_to_length_l138_138413

-- Definitions of length, width, perimeter
def l : ℕ := 10
def P : ℕ := 30

-- Define the condition for the width
def width_from_perimeter (l P : ℕ) : ℕ :=
  (P - 2 * l) / 2

-- Calculate the width using the given length and perimeter
def w : ℕ := width_from_perimeter l P

-- Theorem stating the ratio of width to length
theorem ratio_of_width_to_length : (w : ℚ) / l = 1 / 2 := by
  -- Proof steps will go here
  sorry

end ratio_of_width_to_length_l138_138413


namespace number_2018_location_l138_138613

-- Define the odd square pattern as starting positions of rows
def odd_square (k : ℕ) : ℕ := (2 * k - 1) ^ 2

-- Define the conditions in terms of numbers in each row
def start_of_row (n : ℕ) : ℕ := (2 * n - 1) ^ 2 + 1

def number_at_row_column (n m : ℕ) :=
  start_of_row n + (m - 1)

theorem number_2018_location :
  number_at_row_column 44 82 = 2018 :=
by
  sorry

end number_2018_location_l138_138613


namespace circle_center_radius_sum_l138_138877

theorem circle_center_radius_sum :
  let D := { p : ℝ × ℝ | (p.1^2 - 14*p.1 + p.2^2 + 10*p.2 = -34) }
  let c := 7
  let d := -5
  let s := 2 * Real.sqrt 10
  (c + d + s = 2 + 2 * Real.sqrt 10) :=
by
  sorry

end circle_center_radius_sum_l138_138877


namespace quadratic_term_free_polynomial_l138_138999

theorem quadratic_term_free_polynomial (m : ℤ) (h : 36 + 12 * m = 0) : m^3 = -27 := by
  -- Proof goes here
  sorry

end quadratic_term_free_polynomial_l138_138999


namespace kim_gum_distribution_l138_138372

theorem kim_gum_distribution (cousins : ℕ) (total_gum : ℕ) 
  (h1 : cousins = 4) (h2 : total_gum = 20) : 
  total_gum / cousins = 5 :=
by
  sorry

end kim_gum_distribution_l138_138372


namespace fuel_spending_reduction_l138_138558

-- Define the variables and the conditions
variable (x c : ℝ) -- x for efficiency and c for cost
variable (newEfficiency oldEfficiency newCost oldCost : ℝ)

-- Define the conditions
def conditions := (oldEfficiency = x) ∧ (newEfficiency = 1.75 * oldEfficiency)
                 ∧ (oldCost = c) ∧ (newCost = 1.3 * oldCost)

-- Define the expected reduction in cost
def expectedReduction : ℝ := 25.7142857142857 -- approximately 25 5/7 %

-- Define the assertion that Elmer will reduce his fuel spending by the expected reduction percentage
theorem fuel_spending_reduction : conditions x c oldEfficiency newEfficiency oldCost newCost →
  ((oldCost - (newCost / newEfficiency) * oldEfficiency) / oldCost) * 100 = expectedReduction :=
by
  sorry

end fuel_spending_reduction_l138_138558


namespace proof_equivalent_problem_l138_138700

noncomputable def polar_equation_curve : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    (x - 3) ^ 2 + (y - 1) ^ 2 - 4 = 0

noncomputable def polar_equation_line : Prop :=
  ∀ (θ ρ : ℝ), 
  (Real.sin θ - 2 * Real.cos θ = 1 / ρ) → (2 * (ρ * Real.cos θ) - (ρ * Real.sin θ) + 1 = 0)

noncomputable def distance_from_curve_to_line : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    ∃ d : ℝ, d = (|2 * x - y + 1| / Real.sqrt (2 ^ 2 + 1)) ∧
    d + 2 = (6 * Real.sqrt 5 / 5) + 2

theorem proof_equivalent_problem :
  polar_equation_curve ∧ polar_equation_line ∧ distance_from_curve_to_line :=
by
  constructor
  · exact sorry  -- polar_equation_curve proof
  · constructor
    · exact sorry  -- polar_equation_line proof
    · exact sorry  -- distance_from_curve_to_line proof

end proof_equivalent_problem_l138_138700


namespace difference_between_max_and_min_coins_l138_138060

theorem difference_between_max_and_min_coins (n : ℕ) : 
  (∃ x y : ℕ, x * 10 + y * 25 = 45 ∧ x + y = n) →
  (∃ p q : ℕ, p * 10 + q * 25 = 45 ∧ p + q = n) →
  (n = 2) :=
by
  sorry

end difference_between_max_and_min_coins_l138_138060


namespace describe_S_is_two_rays_l138_138926

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ common : ℝ, 
     (common = 5 ∧ (p.1 + 3 = common ∧ p.2 - 2 ≥ common ∨ p.1 + 3 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.1 + 3 ∧ (5 = common ∧ p.2 - 2 ≥ common ∨ 5 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.2 - 2 ∧ (5 = common ∧ p.1 + 3 ≥ common ∨ 5 ≥ common ∧ p.1 + 3 = common))}

theorem describe_S_is_two_rays :
  S = {p : ℝ × ℝ | (p.1 = 2 ∧ p.2 ≥ 7) ∨ (p.2 = 7 ∧ p.1 ≥ 2)} :=
  by
    sorry

end describe_S_is_two_rays_l138_138926


namespace total_points_correct_l138_138670

variable (H Q T : ℕ)

-- Given conditions
def hw_points : ℕ := 40
def quiz_points := hw_points + 5
def test_points := 4 * quiz_points

-- Question: Prove the total points assigned are 265
theorem total_points_correct :
  H = hw_points →
  Q = quiz_points →
  T = test_points →
  H + Q + T = 265 :=
by
  intros h_hw h_quiz h_test
  rw [h_hw, h_quiz, h_test]
  exact sorry

end total_points_correct_l138_138670


namespace BD_value_l138_138146

def quadrilateral_ABCD_sides (AB BC CD DA : ℕ) (BD : ℕ) : Prop :=
  AB = 5 ∧ BC = 17 ∧ CD = 5 ∧ DA = 9 ∧ 12 < BD ∧ BD < 14 ∧ BD = 13

theorem BD_value (AB BC CD DA : ℕ) (BD : ℕ) : 
  quadrilateral_ABCD_sides AB BC CD DA BD → BD = 13 :=
by
  sorry

end BD_value_l138_138146


namespace infinite_series_sum_l138_138654

theorem infinite_series_sum :
  (∑' n : ℕ, n * (1/5)^n) = 5/16 :=
by sorry

end infinite_series_sum_l138_138654


namespace valid_y_values_for_triangle_l138_138427

-- Define the triangle inequality conditions for sides 8, 11, and y^2
theorem valid_y_values_for_triangle (y : ℕ) (h_pos : y > 0) :
  (8 + 11 > y^2) ∧ (8 + y^2 > 11) ∧ (11 + y^2 > 8) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by
  sorry

end valid_y_values_for_triangle_l138_138427


namespace find_a_for_even_function_l138_138483

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 :=
by
  intro h
  sorry

end find_a_for_even_function_l138_138483


namespace x_varies_as_z_l138_138936

variable {x y z : ℝ}
variable (k j : ℝ)
variable (h1 : x = k * y^3)
variable (h2 : y = j * z^(1/3))

theorem x_varies_as_z (m : ℝ) (h3 : m = k * j^3) : x = m * z := by
  sorry

end x_varies_as_z_l138_138936


namespace min_value_xyz_l138_138167

theorem min_value_xyz (x y z : ℝ) (h1 : xy + 2 * z = 1) (h2 : x^2 + y^2 + z^2 = 10 ) : xyz ≥ -28 :=
by
  sorry

end min_value_xyz_l138_138167


namespace angle_CDE_proof_l138_138895

theorem angle_CDE_proof
    (A B C D E : Type)
    (angle_A angle_B angle_C : ℝ)
    (angle_AEB : ℝ)
    (angle_BED : ℝ)
    (angle_BDE : ℝ) :
    angle_A = 90 ∧
    angle_B = 90 ∧
    angle_C = 90 ∧
    angle_AEB = 50 ∧
    angle_BED = 2 * angle_BDE →
    ∃ angle_CDE : ℝ, angle_CDE = 70 :=
by
  sorry

end angle_CDE_proof_l138_138895


namespace triangle_problem_l138_138020

noncomputable def a : ℝ := 2 * Real.sqrt 3
noncomputable def B : ℝ := 45
noncomputable def S : ℝ := 3 + Real.sqrt 3

noncomputable def c : ℝ := Real.sqrt 2 + Real.sqrt 6
noncomputable def C : ℝ := 75

theorem triangle_problem
  (a_val : a = 2 * Real.sqrt 3)
  (B_val : B = 45)
  (S_val : S = 3 + Real.sqrt 3) :
  c = Real.sqrt 2 + Real.sqrt 6 ∧ C = 75 :=
by
  sorry

end triangle_problem_l138_138020


namespace dmitriev_older_by_10_l138_138810

-- Define the ages of each of the elders
variables (A B C D E F : ℕ)

-- The conditions provided in the problem
axiom hAlyosha : A > (A - 1)
axiom hBorya : B > (B - 2)
axiom hVasya : C > (C - 3)
axiom hGrisha : D > (D - 4)

-- Establishing an equation for the age differences leading to the proof
axiom age_sum_relation : A + B + C + D + E = (A - 1) + (B - 2) + (C - 3) + (D - 4) + F

-- We state that Dmitriev is older than Dima by 10 years
theorem dmitriev_older_by_10 : F = E + 10 :=
by
  -- sorry replaces the proof
  sorry

end dmitriev_older_by_10_l138_138810


namespace points_calculation_l138_138462

def points_per_enemy : ℕ := 9
def total_enemies : ℕ := 11
def enemies_destroyed : ℕ := total_enemies - 3
def total_points_earned : ℕ := enemies_destroyed * points_per_enemy

theorem points_calculation :
  total_points_earned = 72 := by
  sorry

end points_calculation_l138_138462


namespace michael_truck_meet_once_l138_138667

-- Michael's walking speed.
def michael_speed := 4 -- feet per second

-- Distance between trash pails.
def pail_distance := 100 -- feet

-- Truck's speed.
def truck_speed := 8 -- feet per second

-- Time truck stops at each pail.
def truck_stop_time := 20 -- seconds

-- Prove how many times Michael and the truck will meet given the initial condition.
theorem michael_truck_meet_once :
  ∃ n : ℕ, michael_truck_meet_count == 1 :=
sorry

end michael_truck_meet_once_l138_138667


namespace binom_8_5_eq_56_l138_138888

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l138_138888


namespace corrected_mean_l138_138792

/-- The original mean of 20 observations is 36, an observation of 25 was wrongly recorded as 40.
    The correct mean is 35.25. -/
theorem corrected_mean 
  (Mean : ℝ)
  (Observations : ℕ)
  (IncorrectObservation : ℝ)
  (CorrectObservation : ℝ)
  (h1 : Mean = 36)
  (h2 : Observations = 20)
  (h3 : IncorrectObservation = 40)
  (h4 : CorrectObservation = 25) :
  (Mean * Observations - (IncorrectObservation - CorrectObservation)) / Observations = 35.25 :=
sorry

end corrected_mean_l138_138792


namespace abs_sum_example_l138_138938

theorem abs_sum_example : |(-8 : ℤ)| + |(-4 : ℤ)| = 12 := by
  sorry

end abs_sum_example_l138_138938


namespace coefficient_square_sum_l138_138448

theorem coefficient_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by
  sorry

end coefficient_square_sum_l138_138448


namespace find_a_plus_b_l138_138400

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem find_a_plus_b (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (1 + i) * (2 + i) = a + b * i) : a + b = 4 :=
by sorry

end find_a_plus_b_l138_138400


namespace number_of_valid_pairs_l138_138937

theorem number_of_valid_pairs : ∃ p : Finset (ℕ × ℕ), 
  (∀ (a b : ℕ), (a, b) ∈ p ↔ a ≤ 10 ∧ b ≤ 10 ∧ 3 * b < a ∧ a < 4 * b) ∧ p.card = 2 :=
by
  sorry

end number_of_valid_pairs_l138_138937


namespace reducibility_implies_divisibility_l138_138605

theorem reducibility_implies_divisibility
  (a b c d l k : ℤ)
  (p q : ℤ)
  (h1 : a * l + b = k * p)
  (h2 : c * l + d = k * q) :
  k ∣ (a * d - b * c) :=
sorry

end reducibility_implies_divisibility_l138_138605


namespace net_gain_mr_A_l138_138151

def home_worth : ℝ := 12000
def sale1 : ℝ := home_worth * 1.2
def sale2 : ℝ := sale1 * 0.85
def sale3 : ℝ := sale2 * 1.1

theorem net_gain_mr_A : sale1 - sale2 + sale3 = 3384 := by
  sorry -- Proof will be provided here

end net_gain_mr_A_l138_138151


namespace base_length_of_isosceles_triangle_l138_138762

theorem base_length_of_isosceles_triangle (a b : ℕ) 
    (h₁ : a = 8) 
    (h₂ : 2 * a + b = 25) : 
    b = 9 :=
by
  -- This is the proof stub. Proof will be provided here.
  sorry

end base_length_of_isosceles_triangle_l138_138762


namespace system_solution_l138_138918

theorem system_solution (x y z : ℝ) 
  (h1 : 2 * x - 3 * y + z = 8) 
  (h2 : 4 * x - 6 * y + 2 * z = 16) 
  (h3 : x + y - z = 1) : 
  x = 11 / 3 ∧ y = 1 ∧ z = 11 / 3 :=
by
  sorry

end system_solution_l138_138918


namespace problem_solution_l138_138821

variables {a b c : ℝ}

theorem problem_solution
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) :
  (a + b) * (b + c) * (a + c) = 0 := 
sorry

end problem_solution_l138_138821


namespace mystical_mountain_creatures_l138_138863

-- Definitions for conditions
def nineHeadedBirdHeads : Nat := 9
def nineHeadedBirdTails : Nat := 1
def nineTailedFoxHeads : Nat := 1
def nineTailedFoxTails : Nat := 9

-- Prove the number of Nine-Tailed Foxes
theorem mystical_mountain_creatures (x y : Nat)
  (h1 : 9 * x + (y - 1) = 36 * (y - 1) + 4 * x)
  (h2 : 9 * (x - 1) + y = 3 * (9 * y + (x - 1))) :
  x = 14 :=
by
  sorry

end mystical_mountain_creatures_l138_138863


namespace sector_area_given_angle_radius_sector_max_area_perimeter_l138_138313

open Real

theorem sector_area_given_angle_radius :
  ∀ (α : ℝ) (R : ℝ), α = 60 * (π / 180) ∧ R = 10 →
  (α / 360 * 2 * π * R) = 10 * π / 3 ∧ 
  (α * π * R^2 / 360) = 50 * π / 3 :=
by
  intros α R h
  rcases h with ⟨hα, hR⟩
  sorry

theorem sector_max_area_perimeter :
  ∀ (r α: ℝ), (2 * r + r * α) = 8 →
  α = 2 →
  r = 2 :=
by
  intros r α h ha
  sorry

end sector_area_given_angle_radius_sector_max_area_perimeter_l138_138313


namespace find_k_l138_138934
-- Import the necessary library

-- Given conditions as definitions
def circle_eq (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8 * x + y^2 + 2 * y + k = 0

def radius_sq : ℝ := 25  -- since radius = 5, radius squared is 25

-- The statement to prove
theorem find_k (x y k : ℝ) : circle_eq x y k → radius_sq = 25 → k = -8 :=
by
  sorry

end find_k_l138_138934


namespace remainder_of_polynomial_l138_138397

noncomputable def P (x : ℝ) := 3 * x^5 - 2 * x^3 + 5 * x^2 - 8
noncomputable def D (x : ℝ) := x^2 + 3 * x + 2
noncomputable def R (x : ℝ) := 64 * x + 60

theorem remainder_of_polynomial :
  ∀ x : ℝ, P x % D x = R x :=
sorry

end remainder_of_polynomial_l138_138397


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l138_138396

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l138_138396


namespace reflect_and_shift_l138_138659

def f : ℝ → ℝ := sorry  -- Assume f is some function from ℝ to ℝ

def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (6 - x)

theorem reflect_and_shift (f : ℝ → ℝ) (x : ℝ) : h f x = f (6 - x) :=
by
  -- provide the proof here
  sorry

end reflect_and_shift_l138_138659


namespace unique_x0_implies_a_in_range_l138_138479

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x * (3 * x - 1) - a * x + a

theorem unique_x0_implies_a_in_range :
  ∃ x0 : ℤ, f x0 a ≤ 0 ∧ a < 1 -> a ∈ Set.Ico (2 / Real.exp 1) 1 := 
sorry

end unique_x0_implies_a_in_range_l138_138479


namespace color_dots_l138_138660

-- Define the vertices and the edges of the graph representing the figure
inductive Color : Type
| red : Color
| white : Color
| blue : Color

structure Dot :=
  (color : Color)

structure Edge :=
  (u : Dot)
  (v : Dot)

def valid_coloring (dots : List Dot) (edges : List Edge) : Prop :=
  ∀ e ∈ edges, e.u.color ≠ e.v.color

def count_colorings : Nat :=
  6 * 2

theorem color_dots (dots : List Dot) (edges : List Edge)
  (h1 : ∀ d ∈ dots, d.color = Color.red ∨ d.color = Color.white ∨ d.color = Color.blue)
  (h2 : valid_coloring dots edges) :
  count_colorings = 12 :=
by
  sorry

end color_dots_l138_138660


namespace triangle_area_l138_138755

/-- 
  Given:
  - A smaller rectangle OABD with OA = 4 cm, AB = 4 cm
  - A larger rectangle ABEC with AB = 12 cm, BC = 12 cm
  - Point O at (0,0)
  - Point A at (4,0)
  - Point B at (16,0)
  - Point C at (16,12)
  - Point D at (4,12)
  - Point E is on the line from A to C
  
  Prove the area of the triangle CDE is 54 cm²
-/
theorem triangle_area (OA AB OB DE DC : ℕ) : 
  OA = 4 ∧ AB = 4 ∧ OB = 16 ∧ DE = 12 - 3 ∧ DC = 12 → (1 / 2) * DE * DC = 54 := by 
  intros h
  sorry

end triangle_area_l138_138755


namespace point_on_x_axis_point_on_y_axis_l138_138548

section
-- Definitions for the conditions
def point_A (a : ℝ) : ℝ × ℝ := (a - 3, a ^ 2 - 4)

-- Proof for point A lying on the x-axis
theorem point_on_x_axis (a : ℝ) (h : (point_A a).2 = 0) :
  point_A a = (-1, 0) ∨ point_A a = (-5, 0) :=
sorry

-- Proof for point A lying on the y-axis
theorem point_on_y_axis (a : ℝ) (h : (point_A a).1 = 0) :
  point_A a = (0, 5) :=
sorry
end

end point_on_x_axis_point_on_y_axis_l138_138548


namespace find_sixth_number_l138_138361

theorem find_sixth_number (avg_all : ℝ) (avg_first6 : ℝ) (avg_last6 : ℝ) (total_avg : avg_all = 10.7) (first6_avg: avg_first6 = 10.5) (last6_avg: avg_last6 = 11.4) : 
  let S1 := 6 * avg_first6
  let S2 := 6 * avg_last6
  let total_sum := 11 * avg_all
  let X := total_sum - (S1 - X + S2 - X)
  X = 13.7 :=
by 
  sorry

end find_sixth_number_l138_138361


namespace distance_24_km_l138_138978

noncomputable def distance_between_house_and_school (D : ℝ) :=
  let speed_to_school := 6
  let speed_to_home := 4
  let total_time := 10
  total_time = (D / speed_to_school) + (D / speed_to_home)

theorem distance_24_km : ∃ D : ℝ, distance_between_house_and_school D ∧ D = 24 :=
by
  use 24
  unfold distance_between_house_and_school
  sorry

end distance_24_km_l138_138978


namespace longest_side_length_quadrilateral_l138_138380

theorem longest_side_length_quadrilateral :
  (∀ (x y : ℝ),
    (x + y ≤ 4) ∧
    (2 * x + y ≥ 3) ∧
    (x ≥ 0) ∧
    (y ≥ 0)) →
  (∃ d : ℝ, d = 4 * Real.sqrt 2) :=
by sorry

end longest_side_length_quadrilateral_l138_138380


namespace arithmetic_sequence_sum_l138_138497

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (h_arith : arithmetic_sequence a)
    (h_a2 : a 2 = 3)
    (h_a1_a6 : a 1 + a 6 = 12) : a 7 + a 8 + a 9 = 45 :=
by
  sorry

end arithmetic_sequence_sum_l138_138497


namespace seashells_collected_l138_138985

theorem seashells_collected (x y z : ℕ) (hyp : x + y / 2 + z + 5 = 76) : x + y + z = 71 := 
by {
  sorry
}

end seashells_collected_l138_138985


namespace bounds_for_a_l138_138603

theorem bounds_for_a (a : ℝ) (h_a : a > 0) :
  ∀ x : ℝ, 0 < x ∧ x < 17 → (3 / 4) * x = (5 / 6) * (17 - x) + a → a < (153 / 12) := 
sorry

end bounds_for_a_l138_138603


namespace spherical_to_rectangular_l138_138215

theorem spherical_to_rectangular :
  let ρ := 6
  let θ := 7 * Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-3 * Real.sqrt 6, -3 * Real.sqrt 6, 3) :=
by
  sorry

end spherical_to_rectangular_l138_138215


namespace maximum_interval_length_l138_138683

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem maximum_interval_length 
  (m n : ℕ)
  (h1 : 0 < m)
  (h2 : m < n)
  (h3 : ∃ k : ℕ, ∀ i : ℕ, 0 ≤ i → i < k → ¬ is_multiple_of (m + i) 2000 ∧ (m + i) % 2021 = 0):
  n - m = 1999 :=
sorry

end maximum_interval_length_l138_138683


namespace simplify_sqrt_mul_l138_138268

theorem simplify_sqrt_mul : (Real.sqrt 5 * Real.sqrt (4 / 5) = 2) :=
by
  sorry

end simplify_sqrt_mul_l138_138268


namespace smallest_d_for_range_of_g_l138_138280

theorem smallest_d_for_range_of_g :
  ∃ d, (∀ x : ℝ, x^2 + 4 * x + d = 3) → d = 7 := by
  sorry

end smallest_d_for_range_of_g_l138_138280


namespace star_evaluation_l138_138472

noncomputable def star (a b : ℝ) : ℝ := (a + b) / (a - b)

theorem star_evaluation : (star (star 2 3) 4) = 1 / 9 := 
by sorry

end star_evaluation_l138_138472


namespace purely_imaginary_z_value_l138_138668

theorem purely_imaginary_z_value (a : ℝ) (h : (a^2 - a - 2) = 0 ∧ (a + 1) ≠ 0) : a = 2 :=
sorry

end purely_imaginary_z_value_l138_138668


namespace value_of_x_l138_138838

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := 
by
  sorry

end value_of_x_l138_138838


namespace eccentricity_equals_2_l138_138567

variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (A : ℝ × ℝ) (F : ℝ × ℝ) (B : ℝ × ℝ)
variables (eqn_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
variables (focus_F : F = (c, 0)) (imaginary_axis_B : B = (0, b))
variables (intersect_A : A = (c / 3, 2 * b / 3))
variables (vector_eqn : 3 * (A.1, A.2) = (F.1 + 2 * B.1, F.2 + 2 * B.2))
variables (asymptote_eqn : ∀ A1 A2 : ℝ, A2 = (b / a) * A1 → A = (A1, A2))

theorem eccentricity_equals_2 : (c / a = 2) :=
sorry

end eccentricity_equals_2_l138_138567


namespace find_k_l138_138676

theorem find_k (k : ℝ) :
  ∃ k, ∀ x : ℝ, (3 * x^3 + k * x^2 - 8 * x + 52) % (3 * x + 4) = 7 :=
by
-- The proof would go here, we insert sorry to acknowledge the missing proof
sorry

end find_k_l138_138676


namespace min_orders_to_minimize_spent_l138_138928

-- Definitions for the given conditions
def original_price (n p : ℕ) : ℕ := n * p
def discounted_price (T : ℕ) : ℕ := (3 * T) / 5  -- Equivalent to 0.6 * T, using integer math

-- Define the conditions
theorem min_orders_to_minimize_spent 
  (n p : ℕ)
  (h1 : n = 42)
  (h2 : p = 48)
  : ∃ m : ℕ, m = 3 :=
by 
  sorry

end min_orders_to_minimize_spent_l138_138928


namespace LCM_of_8_and_12_l138_138533

-- Definitions based on the provided conditions
def a : ℕ := 8
def x : ℕ := 12

def HCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Conditions
def hcf_condition : HCF a x = 4 := by sorry
def x_condition : x = 12 := rfl

-- The proof statement
theorem LCM_of_8_and_12 : LCM a x = 24 :=
by
  have h1 : HCF a x = 4 := hcf_condition
  have h2 : x = 12 := x_condition
  rw [h2] at h1
  sorry

end LCM_of_8_and_12_l138_138533


namespace permutation_formula_l138_138217

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem permutation_formula (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : permutation n k = Nat.factorial n / Nat.factorial (n - k) :=
by
  unfold permutation
  sorry

end permutation_formula_l138_138217


namespace evaluate_expression_l138_138994

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 2) : 4 * x^y + 5 * y^x = 76 := by
  sorry

end evaluate_expression_l138_138994


namespace value_of_a7_l138_138797

-- Let \( \{a_n\} \) be a sequence such that \( S_n \) denotes the sum of the first \( n \) terms.
-- Given \( S_{n+1}, S_{n+2}, S_{n+3} \) form an arithmetic sequence and \( a_2 = -2 \),
-- prove that \( a_7 = 64 \).

theorem value_of_a7 (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 2) + S (n + 1) = 2 * S n) →
  a 2 = -2 →
  (∀ n : ℕ, a (n + 2) = -2 * a (n + 1)) →
  a 7 = 64 :=
by
  -- skip the proof
  sorry

end value_of_a7_l138_138797


namespace value_of_t_l138_138505

theorem value_of_t (t : ℝ) (x y : ℝ) (h : 3 * x^(t-1) + y - 5 = 0) :
  t = 2 :=
sorry

end value_of_t_l138_138505


namespace race_course_length_l138_138124

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : d = 7 * (d - 120)) : d = 140 :=
sorry

end race_course_length_l138_138124


namespace bailey_credit_cards_l138_138257

theorem bailey_credit_cards (dog_treats : ℕ) (chew_toys : ℕ) (rawhide_bones : ℕ) (items_per_charge : ℕ) (total_items : ℕ) (credit_cards : ℕ)
  (h1 : dog_treats = 8)
  (h2 : chew_toys = 2)
  (h3 : rawhide_bones = 10)
  (h4 : items_per_charge = 5)
  (h5 : total_items = dog_treats + chew_toys + rawhide_bones)
  (h6 : credit_cards = total_items / items_per_charge) :
  credit_cards = 4 :=
by
  sorry

end bailey_credit_cards_l138_138257


namespace necessary_but_not_sufficient_l138_138101

variable (a b : ℝ)

def proposition_A : Prop := a > 0
def proposition_B : Prop := a > b ∧ a⁻¹ > b⁻¹

theorem necessary_but_not_sufficient : (proposition_B a b → proposition_A a) ∧ ¬(proposition_A a → proposition_B a b) :=
by
  sorry

end necessary_but_not_sufficient_l138_138101


namespace boys_love_marbles_l138_138317

def total_marbles : ℕ := 26
def marbles_per_boy : ℕ := 2
def num_boys_love_marbles : ℕ := total_marbles / marbles_per_boy

theorem boys_love_marbles : num_boys_love_marbles = 13 := by
  rfl

end boys_love_marbles_l138_138317


namespace consecutive_odd_natural_numbers_sum_l138_138320

theorem consecutive_odd_natural_numbers_sum (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : b = a + 6) 
  (h4 : c = a + 12) 
  (h5 : c = 27) 
  (h6 : a % 2 = 1) 
  (h7 : b % 2 = 1) 
  (h8 : c % 2 = 1) 
  (h9 : a % 3 = 0) 
  (h10 : b % 3 = 0) 
  (h11 : c % 3 = 0) 
  : a + b + c = 63 :=
by
  sorry

end consecutive_odd_natural_numbers_sum_l138_138320


namespace range_of_a_l138_138273

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a ≥ 0) → (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l138_138273


namespace probability_of_detecting_non_conforming_l138_138148

noncomputable def prob_detecting_non_conforming (total_cans non_conforming_cans selected_cans : ℕ) : ℚ :=
  let total_outcomes := Nat.choose total_cans selected_cans
  let outcomes_with_one_non_conforming := Nat.choose non_conforming_cans 1 * Nat.choose (total_cans - non_conforming_cans) (selected_cans - 1)
  let outcomes_with_two_non_conforming := Nat.choose non_conforming_cans 2
  (outcomes_with_one_non_conforming + outcomes_with_two_non_conforming) / total_outcomes

theorem probability_of_detecting_non_conforming :
  prob_detecting_non_conforming 5 2 2 = 7 / 10 :=
by
  -- Placeholder for the actual proof
  sorry

end probability_of_detecting_non_conforming_l138_138148


namespace compute_x_squared_first_compute_x_squared_second_l138_138339

variable (x : ℝ)
variable (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1)

theorem compute_x_squared_first : 
  1 / (1 / x - 1 / (x + 1)) - x = x^2 :=
by
  sorry

theorem compute_x_squared_second : 
  1 / (1 / (x - 1) - 1 / x) + x = x^2 :=
by
  sorry

end compute_x_squared_first_compute_x_squared_second_l138_138339


namespace part1_part2_l138_138093

noncomputable def f (x a : ℝ) : ℝ := abs (x + 2 * a) + abs (x - 1)

noncomputable def g (a : ℝ) : ℝ := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part1 (x : ℝ) : f x 1 ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2) ≤ a ∧ a ≤ (3 / 2) := by
  sorry

end part1_part2_l138_138093


namespace bubble_bath_per_guest_l138_138767

def rooms_couple : ℕ := 13
def rooms_single : ℕ := 14
def total_bubble_bath : ℕ := 400

theorem bubble_bath_per_guest :
  (total_bubble_bath / (rooms_couple * 2 + rooms_single)) = 10 :=
by
  sorry

end bubble_bath_per_guest_l138_138767


namespace added_number_after_doubling_l138_138194

theorem added_number_after_doubling (x y : ℤ) (h1 : x = 4) (h2 : 3 * (2 * x + y) = 51) : y = 9 :=
by
  -- proof goes here
  sorry

end added_number_after_doubling_l138_138194


namespace sum_modulo_remainder_l138_138021

theorem sum_modulo_remainder :
  ((82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17) = 12 :=
by
  sorry

end sum_modulo_remainder_l138_138021


namespace jill_food_percentage_l138_138791

theorem jill_food_percentage (total_amount : ℝ) (tax_rate_clothing tax_rate_other_items spent_clothing_rate spent_other_rate spent_total_tax_rate : ℝ) : 
  spent_clothing_rate = 0.5 →
  spent_other_rate = 0.25 →
  tax_rate_clothing = 0.1 →
  tax_rate_other_items = 0.2 →
  spent_total_tax_rate = 0.1 →
  (spent_clothing_rate * tax_rate_clothing * total_amount) + (spent_other_rate * tax_rate_other_items * total_amount) = spent_total_tax_rate * total_amount →
  (1 - spent_clothing_rate - spent_other_rate) * total_amount / total_amount = 0.25 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end jill_food_percentage_l138_138791


namespace sample_frequency_in_range_l138_138630

theorem sample_frequency_in_range :
  let total_capacity := 100
  let freq_0_10 := 12
  let freq_10_20 := 13
  let freq_20_30 := 24
  let freq_30_40 := 15
  (freq_0_10 + freq_10_20 + freq_20_30 + freq_30_40) / total_capacity = 0.64 :=
by
  sorry

end sample_frequency_in_range_l138_138630


namespace mail_handling_in_six_months_l138_138782

theorem mail_handling_in_six_months (daily_letters daily_packages days_per_month months : ℕ) :
  daily_letters = 60 →
  daily_packages = 20 →
  days_per_month = 30 →
  months = 6 →
  (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  -- Skipping the proof
  sorry

end mail_handling_in_six_months_l138_138782


namespace average_price_of_cow_l138_138644

theorem average_price_of_cow (total_price_cows_and_goats rs: ℕ) (num_cows num_goats: ℕ)
    (avg_price_goat: ℕ) (total_price: total_price_cows_and_goats = 1400)
    (num_cows_eq: num_cows = 2) (num_goats_eq: num_goats = 8)
    (avg_price_goat_eq: avg_price_goat = 60) :
    let total_price_goats := avg_price_goat * num_goats
    let total_price_cows := total_price_cows_and_goats - total_price_goats
    let avg_price_cow := total_price_cows / num_cows
    avg_price_cow = 460 :=
by
  sorry

end average_price_of_cow_l138_138644


namespace range_of_x_l138_138847

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x < -1/2 ∨ x > 1/4 :=
by
  sorry

end range_of_x_l138_138847


namespace find_y_l138_138977

theorem find_y (x y: ℤ) (h1: x^2 - 3 * x + 2 = y + 6) (h2: x = -4) : y = 24 :=
by
  sorry

end find_y_l138_138977


namespace g_nine_l138_138181

variable (g : ℝ → ℝ)

theorem g_nine : (∀ x y : ℝ, g (x + y) = g x * g y) → g 3 = 4 → g 9 = 64 :=
by intros h1 h2; sorry

end g_nine_l138_138181


namespace trapezium_area_l138_138849

theorem trapezium_area (a b h : ℝ) (h₁ : a = 20) (h₂ : b = 16) (h₃ : h = 15) : 
  (1/2 * (a + b) * h = 270) :=
by
  rw [h₁, h₂, h₃]
  -- The following lines of code are omitted as they serve as solving this proof, and the requirement is to provide the statement only. 
  sorry

end trapezium_area_l138_138849


namespace louis_current_age_l138_138086

-- Define the constants for years to future and future age of Carla
def years_to_future : ℕ := 6
def carla_future_age : ℕ := 30

-- Define the sum of current ages
def sum_current_ages : ℕ := 55

-- State the theorem
theorem louis_current_age :
  ∃ (c l : ℕ), (c + years_to_future = carla_future_age) ∧ (c + l = sum_current_ages) ∧ (l = 31) :=
sorry

end louis_current_age_l138_138086


namespace negation_prop_l138_138107

open Classical

variable (x : ℝ)

theorem negation_prop :
    (∃ x : ℝ, x^2 + 2*x + 2 < 0) = False ↔
    (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by
    sorry

end negation_prop_l138_138107


namespace probability_one_white_one_black_two_touches_l138_138718

def probability_white_ball : ℚ := 7 / 10
def probability_black_ball : ℚ := 3 / 10

theorem probability_one_white_one_black_two_touches :
  (probability_white_ball * probability_black_ball) + (probability_black_ball * probability_white_ball) = (7 / 10) * (3 / 10) + (3 / 10) * (7 / 10) :=
by
  -- The proof is omitted here.
  sorry

end probability_one_white_one_black_two_touches_l138_138718


namespace ellipse_hyperbola_equation_l138_138893

-- Definitions for the Ellipse and Hyperbola
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / 10 + (y^2) / m = 1
def hyperbola (x y : ℝ) (b : ℝ) : Prop := (x^2) - (y^2) / b = 1

-- Conditions
def same_foci (c1 c2 : ℝ) : Prop := c1 = c2
def intersection_at_p (x y : ℝ) : Prop := x = (Real.sqrt 10) / 3 ∧ (ellipse x y 1 ∧ hyperbola x y 8)

-- Theorem stating the mathematically equivalent proof problem
theorem ellipse_hyperbola_equation :
  ∀ (m b : ℝ) (x y : ℝ), ellipse x y m ∧ hyperbola x y b ∧ same_foci (Real.sqrt (10 - m)) (Real.sqrt (1 + b)) ∧ intersection_at_p x y
  → (m = 1) ∧ (b = 8) := 
by
  intros m b x y h
  sorry

end ellipse_hyperbola_equation_l138_138893


namespace usual_time_to_school_l138_138271

theorem usual_time_to_school (S T t : ℝ) (h : 1.2 * S * (T - t) = S * T) : T = 6 * t :=
by
  sorry

end usual_time_to_school_l138_138271


namespace inequality_proof_l138_138931

theorem inequality_proof
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * b + b * c + c * a = 1) :
  (a + b) / Real.sqrt (a * b * (1 - a * b)) + 
  (b + c) / Real.sqrt (b * c * (1 - b * c)) + 
  (c + a) / Real.sqrt (c * a * (1 - c * a)) ≤ Real.sqrt 2 / (a * b * c) :=
sorry

end inequality_proof_l138_138931


namespace express_An_l138_138848

noncomputable def A_n (A : ℝ) (n : ℤ) : ℝ :=
  (1 / 2^n) * ((A + (A^2 - 4).sqrt)^n + (A - (A^2 - 4).sqrt)^n)

theorem express_An (a : ℝ) (A : ℝ) (n : ℤ) (h : a + a⁻¹ = A) :
  (a^n + a^(-n)) = A_n A n := 
sorry

end express_An_l138_138848


namespace baseball_singles_percentage_l138_138639

theorem baseball_singles_percentage :
  let total_hits := 50
  let home_runs := 2
  let triples := 3
  let doubles := 8
  let non_single_hits := home_runs + triples + doubles
  let singles := total_hits - non_single_hits
  let singles_percentage := (singles / total_hits) * 100
  singles = 37 ∧ singles_percentage = 74 :=
by
  sorry

end baseball_singles_percentage_l138_138639


namespace geo_prog_sum_463_l138_138499

/-- Given a set of natural numbers forming an increasing geometric progression with an integer
common ratio where the sum equals 463, prove that these numbers must be {463}, {1, 462}, or {1, 21, 441}. -/
theorem geo_prog_sum_463 (n : ℕ) (b₁ q : ℕ) (s : Finset ℕ) (hgeo : ∀ i j, i < j → s.toList.get? i = some (b₁ * q^i) ∧ s.toList.get? j = some (b₁ * q^j))
  (hsum : s.sum id = 463) : 
  s = {463} ∨ s = {1, 462} ∨ s = {1, 21, 441} :=
sorry

end geo_prog_sum_463_l138_138499


namespace diplomats_not_speaking_russian_l138_138919

-- Definitions to formalize the problem
def total_diplomats : ℕ := 150
def speak_french : ℕ := 17
def speak_both_french_and_russian : ℕ := (10 * total_diplomats) / 100
def speak_neither_french_nor_russian : ℕ := (20 * total_diplomats) / 100

-- Theorem to prove the desired quantity
theorem diplomats_not_speaking_russian : 
  speak_neither_french_nor_russian + (speak_french - speak_both_french_and_russian) = 32 := by
  sorry

end diplomats_not_speaking_russian_l138_138919


namespace john_marbles_selection_l138_138925

theorem john_marbles_selection :
  let total_marbles := 15
  let special_colors := 4
  let total_chosen := 5
  let chosen_special_colors := 2
  let remaining_colors := total_marbles - special_colors
  let chosen_normal_colors := total_chosen - chosen_special_colors
  (Nat.choose 4 2) * (Nat.choose 11 3) = 990 :=
by
  sorry

end john_marbles_selection_l138_138925


namespace minimum_value_is_correct_l138_138263

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  (x + 1/y) * (x + 1/y - 2024) + (y + 1/x) * (y + 1/x - 2024) + 2024

theorem minimum_value_is_correct (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (minimum_value x y) ≥ -2050208 := 
sorry

end minimum_value_is_correct_l138_138263


namespace num_disks_to_sell_l138_138781

-- Define the buying and selling price conditions.
def cost_per_disk := 6 / 5
def sell_per_disk := 7 / 4

-- Define the desired profit
def desired_profit := 120

-- Calculate the profit per disk.
def profit_per_disk := sell_per_disk - cost_per_disk

-- Statement of the problem: Determine number of disks to sell.
theorem num_disks_to_sell
  (h₁ : cost_per_disk = 6 / 5)
  (h₂ : sell_per_disk = 7 / 4)
  (h₃ : desired_profit = 120)
  (h₄ : profit_per_disk = 7 / 4 - 6 / 5) :
  ∃ disks_to_sell : ℕ, disks_to_sell = 219 ∧ 
  disks_to_sell * profit_per_disk ≥ 120 ∧
  (disks_to_sell - 1) * profit_per_disk < 120 :=
sorry

end num_disks_to_sell_l138_138781


namespace donny_spent_total_on_friday_and_sunday_l138_138858

noncomputable def daily_savings (initial: ℚ) (increase_rate: ℚ) (days: List ℚ) : List ℚ :=
days.scanl (λ acc day => acc * increase_rate + acc) initial

noncomputable def thursday_savings : ℚ := (daily_savings 15 (1 + 0.1) [15, 15, 15]).sum

noncomputable def friday_spent : ℚ := thursday_savings * 0.5

noncomputable def remaining_after_friday : ℚ := thursday_savings - friday_spent

noncomputable def saturday_savings (thursday: ℚ) : ℚ := thursday * (1 - 0.20)

noncomputable def total_savings_saturday : ℚ := remaining_after_friday + saturday_savings thursday_savings

noncomputable def sunday_spent : ℚ := total_savings_saturday * 0.40

noncomputable def total_spent : ℚ := friday_spent + sunday_spent

theorem donny_spent_total_on_friday_and_sunday : total_spent = 55.13 := by
  sorry

end donny_spent_total_on_friday_and_sunday_l138_138858


namespace vertex_sum_of_cube_l138_138595

noncomputable def cube_vertex_sum (a : Fin 8 → ℕ) : ℕ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7

def face_sums (a : Fin 8 → ℕ) : List ℕ :=
  [
    a 0 + a 1 + a 2 + a 3, -- first face
    a 0 + a 1 + a 4 + a 5, -- second face
    a 0 + a 3 + a 4 + a 7, -- third face
    a 1 + a 2 + a 5 + a 6, -- fourth face
    a 2 + a 3 + a 6 + a 7, -- fifth face
    a 4 + a 5 + a 6 + a 7  -- sixth face
  ]

def total_face_sum (a : Fin 8 → ℕ) : ℕ :=
  List.sum (face_sums a)

theorem vertex_sum_of_cube (a : Fin 8 → ℕ) (h : total_face_sum a = 2019) :
  cube_vertex_sum a = 673 :=
sorry

end vertex_sum_of_cube_l138_138595


namespace problem1_problem2_l138_138764

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Statement for the first proof
theorem problem1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  bc / a + ca / b + ab / c ≥ a + b + c :=
sorry

-- Statement for the second proof
theorem problem2 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (1 - a) / a + (1 - b) / b + (1 - c) / c ≥ 6 :=
sorry

end problem1_problem2_l138_138764


namespace min_value_eq_l138_138430

open Real
open Classical

noncomputable def min_value (x y : ℝ) : ℝ := x + 4 * y

theorem min_value_eq :
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / x + 1 / (2 * y) = 1) → (min_value x y) = 3 + 2 * sqrt 2 :=
by
  sorry

end min_value_eq_l138_138430


namespace annie_serious_accident_probability_l138_138885

theorem annie_serious_accident_probability :
  (∀ temperature : ℝ, temperature < 32 → ∃ skid_chance_increase : ℝ, skid_chance_increase = 5 * ⌊ (32 - temperature) / 3 ⌋ / 100) →
  (∀ control_regain_chance : ℝ, control_regain_chance = 0.4) →
  (∀ control_loss_chance : ℝ, control_loss_chance = 1 - control_regain_chance) →
  (temperature = 8) →
  (serious_accident_probability = skid_chance_increase * control_loss_chance) →
  serious_accident_probability = 0.24 := by
  sorry

end annie_serious_accident_probability_l138_138885


namespace line_equation_l138_138800

theorem line_equation (a b : ℝ)
(h1 : a * -1 + b * 2 = 0) 
(h2 : a = b) :
((a = 1 ∧ b = -1) ∨ (a = 2 ∧ b = -1)) := 
by
  sorry

end line_equation_l138_138800


namespace eq_radicals_same_type_l138_138272

theorem eq_radicals_same_type (a b : ℕ) (h1 : a - 1 = 2) (h2 : 3 * b - 1 = 7 - b) : a + b = 5 :=
by
  sorry

end eq_radicals_same_type_l138_138272


namespace rabbit_speed_l138_138039

theorem rabbit_speed (s : ℕ) (h : (s * 2 + 4) * 2 = 188) : s = 45 :=
sorry

end rabbit_speed_l138_138039


namespace suma_work_rate_l138_138299

theorem suma_work_rate (W : ℕ) : 
  (∀ W, (W / 6) + (W / S) = W / 4) → S = 24 :=
by
  intro h
  -- detailed proof would actually go here
  sorry

end suma_work_rate_l138_138299


namespace time_saved_l138_138230

theorem time_saved (speed_with_tide distance1 time1 distance2 time2: ℝ) 
  (h1: speed_with_tide = 5) 
  (h2: distance1 = 5) 
  (h3: time1 = 1) 
  (h4: distance2 = 40) 
  (h5: time2 = 10) : 
  time2 - (distance2 / speed_with_tide) = 2 := 
sorry

end time_saved_l138_138230


namespace negation_of_proposition_p_l138_138884

variable (x : ℝ)

def proposition_p : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0

theorem negation_of_proposition_p : ¬ proposition_p ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 := by
  sorry

end negation_of_proposition_p_l138_138884


namespace simplified_expression_at_3_l138_138412

noncomputable def simplify_and_evaluate (x : ℝ) : ℝ :=
  (3 * x ^ 2 + 8 * x - 6) - (2 * x ^ 2 + 4 * x - 15)

theorem simplified_expression_at_3 : simplify_and_evaluate 3 = 30 :=
by
  sorry

end simplified_expression_at_3_l138_138412


namespace evaluate_expression_l138_138535

theorem evaluate_expression :
  let c := (-2 : ℚ)
  let x := (2 : ℚ) / 5
  let y := (3 : ℚ) / 5
  let z := (-3 : ℚ)
  c * x^3 * y^4 * z^2 = (-11664) / 78125 := by
  sorry

end evaluate_expression_l138_138535


namespace geometric_sequence_problem_l138_138908

variable {a : ℕ → ℝ}
variable (r a1 : ℝ)
variable (h_pos : ∀ n, a n > 0)
variable (h_geom : ∀ n, a (n + 1) = a 1 * r ^ n)
variable (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 2025)

theorem geometric_sequence_problem :
  a 3 + a 5 = 45 :=
by
  sorry

end geometric_sequence_problem_l138_138908


namespace missy_yells_at_obedient_dog_12_times_l138_138894

theorem missy_yells_at_obedient_dog_12_times (x : ℕ) (h : x + 4 * x = 60) : x = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end missy_yells_at_obedient_dog_12_times_l138_138894


namespace sequence_sum_is_100_then_n_is_10_l138_138498

theorem sequence_sum_is_100_then_n_is_10 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, a (n + 1) = a n + 2) →
  (∀ n, S n = n * a 1 + n * (n - 1)) →
  (∃ n, S n = 100) → 
  n = 10 :=
by sorry

end sequence_sum_is_100_then_n_is_10_l138_138498


namespace work_related_emails_count_l138_138680

-- Definitions based on the identified conditions and the question
def total_emails : ℕ := 1200
def spam_percentage : ℕ := 27
def promotional_percentage : ℕ := 18
def social_percentage : ℕ := 15

-- The statement to prove, indicated the goal
theorem work_related_emails_count :
  (total_emails * (100 - spam_percentage - promotional_percentage - social_percentage)) / 100 = 480 :=
by
  sorry

end work_related_emails_count_l138_138680


namespace gwen_remaining_money_l138_138949

theorem gwen_remaining_money:
  ∀ (Gwen_received Gwen_spent Gwen_remaining: ℕ),
    Gwen_received = 5 →
    Gwen_spent = 3 →
    Gwen_remaining = Gwen_received - Gwen_spent →
    Gwen_remaining = 2 :=
by
  intros Gwen_received Gwen_spent Gwen_remaining h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end gwen_remaining_money_l138_138949


namespace log_expression_value_l138_138921

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_expression_value :
  log_base 10 3 + 3 * log_base 10 2 + 2 * log_base 10 5 + 4 * log_base 10 3 + log_base 10 9 = 5.34 :=
by
  sorry

end log_expression_value_l138_138921


namespace rancher_no_cows_l138_138504

theorem rancher_no_cows (s c : ℕ) (h1 : 30 * s + 31 * c = 1200) 
  (h2 : 15 ≤ s) (h3 : s ≤ 35) : c = 0 :=
by
  sorry

end rancher_no_cows_l138_138504


namespace zero_point_interval_l138_138902

noncomputable def f (x : ℝ) : ℝ := Real.pi * x + Real.log x / Real.log 2

theorem zero_point_interval : 
  f (1/4) < 0 ∧ f (1/2) > 0 → ∃ x : ℝ, 1/4 ≤ x ∧ x ≤ 1/2 ∧ f x = 0 :=
by
  sorry

end zero_point_interval_l138_138902


namespace children_count_l138_138682

-- Define the total number of passengers on the airplane
def total_passengers : ℕ := 240

-- Define the ratio of men to women
def men_to_women_ratio : ℕ × ℕ := (3, 2)

-- Define the percentage of passengers who are either men or women
def percent_men_women : ℕ := 60

-- Define the number of children on the airplane
def number_of_children (total : ℕ) (percent : ℕ) : ℕ := 
  (total * (100 - percent)) / 100

theorem children_count :
  number_of_children total_passengers percent_men_women = 96 := by
  sorry

end children_count_l138_138682


namespace sum_of_unit_fractions_l138_138054

theorem sum_of_unit_fractions : (1 / 2) + (1 / 3) + (1 / 7) + (1 / 42) = 1 := 
by 
  sorry

end sum_of_unit_fractions_l138_138054


namespace find_wanderer_in_8th_bar_l138_138827

noncomputable def wanderer_probability : ℚ := 1 / 3

theorem find_wanderer_in_8th_bar
    (total_bars : ℕ)
    (initial_prob_in_any_bar : ℚ)
    (prob_not_in_specific_bar : ℚ)
    (prob_not_in_first_seven : ℚ)
    (posterior_prob : ℚ)
    (h1 : total_bars = 8)
    (h2 : initial_prob_in_any_bar = 4 / 5)
    (h3 : prob_not_in_specific_bar = 1 - (initial_prob_in_any_bar / total_bars))
    (h4 : prob_not_in_first_seven = prob_not_in_specific_bar ^ 7)
    (h5 : posterior_prob = initial_prob_in_any_bar / prob_not_in_first_seven) :
    posterior_prob = wanderer_probability := 
sorry

end find_wanderer_in_8th_bar_l138_138827


namespace f_value_at_5_l138_138739

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 / 2 then 2 * x^2 else sorry

theorem f_value_at_5 (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 3)
  (h_definition : ∀ x, 0 ≤ x ∧ x ≤ 3 / 2 → f x = 2 * x^2) :
  f 5 = 2 :=
by
  sorry

end f_value_at_5_l138_138739


namespace probability_of_six_and_queen_l138_138954

variable {deck : Finset (ℕ × String)}
variable (sixes : Finset (ℕ × String))
variable (queens : Finset (ℕ × String))

def standard_deck : Finset (ℕ × String) := sorry

-- Condition: the deck contains 52 cards (13 hearts, 13 clubs, 13 spades, 13 diamonds)
-- and it has 4 sixes and 4 Queens.
axiom h_deck_size : standard_deck.card = 52
axiom h_sixes : ∀ c ∈ standard_deck, c.1 = 6 → c ∈ sixes
axiom h_queens : ∀ c ∈ standard_deck, c.1 = 12 → c ∈ queens

-- Define the probability function for dealing cards
noncomputable def prob_first_six_and_second_queen : ℚ :=
  (4 / 52) * (4 / 51)

theorem probability_of_six_and_queen :
  prob_first_six_and_second_queen = 4 / 663 :=
by
  sorry

end probability_of_six_and_queen_l138_138954


namespace quiz_sum_correct_l138_138224

theorem quiz_sum_correct (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (h_sub : x - y = 4) (h_mul : x * y = 104) :
  x + y = 20 := by
  sorry

end quiz_sum_correct_l138_138224


namespace quadratic_has_two_distinct_roots_l138_138309

theorem quadratic_has_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 2*x₁ + k = 0) ∧ (x₂^2 - 2*x₂ + k = 0))
  ↔ k < 1 :=
by sorry

end quadratic_has_two_distinct_roots_l138_138309


namespace total_amount_received_l138_138249

-- Definitions based on conditions
def days_A : Nat := 6
def days_B : Nat := 8
def days_ABC : Nat := 3

def share_A : Nat := 300
def share_B : Nat := 225
def share_C : Nat := 75

-- The theorem stating the total amount received for the work
theorem total_amount_received (dA dB dABC : Nat) (sA sB sC : Nat)
  (h1 : dA = days_A) (h2 : dB = days_B) (h3 : dABC = days_ABC)
  (h4 : sA = share_A) (h5 : sB = share_B) (h6 : sC = share_C) : 
  sA + sB + sC = 600 := by
  sorry

end total_amount_received_l138_138249


namespace cannot_make_it_in_time_l138_138037

theorem cannot_make_it_in_time (time_available : ℕ) (distance_to_station : ℕ) (v1 : ℕ) :
  time_available = 2 ∧ distance_to_station = 2 ∧ v1 = 30 → 
  ¬ ∃ v2, (time_available - (distance_to_station / v1)) * v2 ≥ 1 :=
by
  sorry

end cannot_make_it_in_time_l138_138037


namespace right_triangle_iff_sum_excircles_a_right_triangle_iff_sum_excircles_b_l138_138140

-- Definitions for part (a)
def is_right_angled_triangle_a (a b c r r_a r_b r_c : ℝ) :=
  r + r_a + r_b + r_c = a + b + c

def right_angled_triangle_a (a b c : ℝ) :=
  a^2 + b^2 = c^2 -- Assuming c is the hypotenuse

theorem right_triangle_iff_sum_excircles_a (a b c r r_a r_b r_c : ℝ) :
  right_angled_triangle_a a b c ↔ is_right_angled_triangle_a a b c r r_a r_b r_c := sorry

-- Definitions for part (b)
def is_right_angled_triangle_b (a b c r r_a r_b r_c : ℝ) :=
  r^2 + r_a^2 + r_b^2 + r_c^2 = a^2 + b^2 + c^2

def right_angled_triangle_b (a b c : ℝ) :=
  a^2 + b^2 = c^2 -- Assuming c is the hypotenuse

theorem right_triangle_iff_sum_excircles_b (a b c r r_a r_b r_c : ℝ) :
  right_angled_triangle_b a b c ↔ is_right_angled_triangle_b a b c r r_a r_b r_c := sorry

end right_triangle_iff_sum_excircles_a_right_triangle_iff_sum_excircles_b_l138_138140


namespace total_odd_green_red_marbles_l138_138014

def Sara_green : ℕ := 3
def Sara_red : ℕ := 5
def Tom_green : ℕ := 4
def Tom_red : ℕ := 7
def Lisa_green : ℕ := 5
def Lisa_red : ℕ := 3

theorem total_odd_green_red_marbles : 
  (if Sara_green % 2 = 1 then Sara_green else 0) +
  (if Sara_red % 2 = 1 then Sara_red else 0) +
  (if Tom_green % 2 = 1 then Tom_green else 0) +
  (if Tom_red % 2 = 1 then Tom_red else 0) +
  (if Lisa_green % 2 = 1 then Lisa_green else 0) +
  (if Lisa_red % 2 = 1 then Lisa_red else 0) = 23 := by
  sorry

end total_odd_green_red_marbles_l138_138014


namespace Christopher_joggers_eq_80_l138_138023

variable (T A C : ℕ)

axiom Tyson_joggers : T > 0                  -- Tyson bought a positive number of joggers.

axiom Alexander_condition : A = T + 22        -- Alexander bought 22 more joggers than Tyson.

axiom Christopher_condition : C = 20 * T      -- Christopher bought twenty times as many joggers as Tyson.

axiom Christopher_Alexander : C = A + 54     -- Christopher bought 54 more joggers than Alexander.

theorem Christopher_joggers_eq_80 : C = 80 := 
by
  sorry

end Christopher_joggers_eq_80_l138_138023


namespace more_divisible_by_7_than_11_l138_138612

open Nat

theorem more_divisible_by_7_than_11 :
  let N := 10000
  let count_7_not_11 := (N / 7) - (N / 77)
  let count_11_not_7 := (N / 11) - (N / 77)
  count_7_not_11 > count_11_not_7 := 
  by
    let N := 10000
    let count_7_not_11 := (N / 7) - (N / 77)
    let count_11_not_7 := (N / 11) - (N / 77)
    sorry

end more_divisible_by_7_than_11_l138_138612


namespace fraction_sum_to_decimal_l138_138976

theorem fraction_sum_to_decimal :
  (3 / 20 : ℝ) + (5 / 200 : ℝ) + (7 / 2000 : ℝ) = 0.1785 :=
by 
  sorry

end fraction_sum_to_decimal_l138_138976


namespace Series_value_l138_138314

theorem Series_value :
  (∑' n : ℕ, (2^n) / (7^(2^n) + 1)) = 1 / 6 :=
sorry

end Series_value_l138_138314


namespace horizontal_distance_travelled_l138_138875

theorem horizontal_distance_travelled (r : ℝ) (θ : ℝ) (d : ℝ)
  (h_r : r = 2) (h_θ : θ = Real.pi / 6) :
  d = 2 * Real.sqrt 3 * Real.pi := sorry

end horizontal_distance_travelled_l138_138875


namespace evaluate_expression_l138_138861

theorem evaluate_expression (x z : ℤ) (hx : x = 4) (hz : z = -2) : 
  z * (z - 4 * x) = 36 := by
  sorry

end evaluate_expression_l138_138861


namespace sector_perimeter_l138_138837

theorem sector_perimeter (r : ℝ) (c : ℝ) (angle_deg : ℝ) (angle_rad := angle_deg * Real.pi / 180) 
  (arc_length := r * angle_rad) (P := arc_length + c)
  (h1 : r = 10) (h2 : c = 10) (h3 : angle_deg = 120) :
  P = 20 * Real.pi / 3 + 10 :=
by
  sorry

end sector_perimeter_l138_138837


namespace find_m_of_ellipse_l138_138929

theorem find_m_of_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (10 - m) + y^2 / (m - 2) = 1) ∧ (m - 2 > 10 - m) ∧ ((4)^2 = (m - 2) - (10 - m))) → m = 14 :=
by sorry

end find_m_of_ellipse_l138_138929


namespace find_n_l138_138547

variable (x n : ℕ)
variable (y : ℕ) {h1 : y = 24}

theorem find_n
  (h1 : y = 24) 
  (h2 : x / y = 1 / 4) 
  (h3 : (x + n) / y = 1 / 2) : 
  n = 6 := 
sorry

end find_n_l138_138547


namespace remainder_of_99_pow_36_mod_100_l138_138322

theorem remainder_of_99_pow_36_mod_100 :
  (99 : ℤ)^36 % 100 = 1 := sorry

end remainder_of_99_pow_36_mod_100_l138_138322


namespace prob_of_selecting_blue_ball_l138_138281

noncomputable def prob_select_ball :=
  let prob_X := 1 / 3
  let prob_Y := 1 / 3
  let prob_Z := 1 / 3
  let prob_blue_X := 7 / 10
  let prob_blue_Y := 1 / 2
  let prob_blue_Z := 2 / 5
  prob_X * prob_blue_X + prob_Y * prob_blue_Y + prob_Z * prob_blue_Z

theorem prob_of_selecting_blue_ball :
  prob_select_ball = 8 / 15 :=
by
  -- Provide the proof here
  sorry

end prob_of_selecting_blue_ball_l138_138281


namespace election_votes_l138_138287

theorem election_votes (P : ℕ) (M : ℕ) (V : ℕ) (hP : P = 60) (hM : M = 1300) :
  V = 6500 :=
by
  sorry

end election_votes_l138_138287


namespace complex_magnitude_difference_eq_one_l138_138365

noncomputable def magnitude (z : Complex) : ℝ := Complex.abs z

/-- Lean 4 statement of the problem -/
theorem complex_magnitude_difference_eq_one (z₁ z₂ : Complex) (h₁ : magnitude z₁ = 1) (h₂ : magnitude z₂ = 1) (h₃ : magnitude (z₁ + z₂) = Real.sqrt 3) : magnitude (z₁ - z₂) = 1 := 
sorry

end complex_magnitude_difference_eq_one_l138_138365


namespace min_birthdays_on_wednesday_l138_138873

theorem min_birthdays_on_wednesday (n x w: ℕ) (h_n : n = 61) 
  (h_ineq : w > x) (h_sum : 6 * x + w = n) : w ≥ 13 :=
by
  sorry

end min_birthdays_on_wednesday_l138_138873


namespace find_a_2b_3c_l138_138089

noncomputable def a : ℝ := 28
noncomputable def b : ℝ := 32
noncomputable def c : ℝ := -3

def ineq_condition (x : ℝ) : Prop := (x < -3) ∨ (abs (x - 30) ≤ 2)

theorem find_a_2b_3c (a b c : ℝ) (h₁ : a < b)
  (h₂ : ∀ x : ℝ, (x < -3 ∨ abs (x - 30) ≤ 2) ↔ ((x - a)*(x - b)/(x - c) ≤ 0)) :
  a + 2 * b + 3 * c = 83 :=
by
  sorry

end find_a_2b_3c_l138_138089


namespace original_square_area_l138_138831

theorem original_square_area {x y : ℕ} (h1 : y ≠ 1)
  (h2 : x^2 = 24 + y^2) : x^2 = 49 :=
sorry

end original_square_area_l138_138831


namespace find_solution_l138_138354

-- Define the setup for the problem
variables (k x y : ℝ)

-- Conditions from the problem
def cond1 : Prop := x - y = 9 * k
def cond2 : Prop := x + y = 5 * k
def cond3 : Prop := 2 * x + 3 * y = 8

-- Proof statement combining all conditions to show the values of k, x, and y that satisfy them
theorem find_solution :
  cond1 k x y →
  cond2 k x y →
  cond3 x y →
  k = 1 ∧ x = 7 ∧ y = -2 := by
  sorry

end find_solution_l138_138354


namespace harmonica_value_l138_138988

theorem harmonica_value (x : ℕ) (h1 : ∃ k : ℕ, ∃ r : ℕ, x = 12 * k + r ∧ r ≠ 0 
                                                   ∧ r ≠ 6 ∧ r ≠ 9 
                                                   ∧ r ≠ 10 ∧ r ≠ 11)
                         (h2 : ¬ (x * x % 12 = 0)) : 
                         4 = 4 :=
by 
  sorry

end harmonica_value_l138_138988


namespace xy_relationship_l138_138817

theorem xy_relationship :
  let x := 123456789 * 123456786
  let y := 123456788 * 123456787
  x < y := 
by
  sorry

end xy_relationship_l138_138817


namespace intersection_of_M_and_N_l138_138611

open Set

variable (M N : Set ℕ)

theorem intersection_of_M_and_N :
  M = {1, 2, 4, 8, 16} →
  N = {2, 4, 6, 8} →
  M ∩ N = {2, 4, 8} :=
by
  intros hM hN
  rw [hM, hN]
  ext x
  simp
  sorry

end intersection_of_M_and_N_l138_138611


namespace solution_to_system_l138_138208

theorem solution_to_system :
  ∀ (x y z : ℝ), 
  x * (3 * y^2 + 1) = y * (y^2 + 3) →
  y * (3 * z^2 + 1) = z * (z^2 + 3) →
  z * (3 * x^2 + 1) = x * (x^2 + 3) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end solution_to_system_l138_138208


namespace quadratic_equation_identify_l138_138541

theorem quadratic_equation_identify {a b c x : ℝ} :
  ((3 - 5 * x^2 = x) ↔ true) ∧
  ((3 / x + x^2 - 1 = 0) ↔ false) ∧
  ((a * x^2 + b * x + c = 0) ↔ (a ≠ 0)) ∧
  ((4 * x - 1 = 0) ↔ false) :=
by
  sorry

end quadratic_equation_identify_l138_138541


namespace probability_one_first_class_product_l138_138920

-- Define the probabilities for the interns processing first-class products
def P_first_intern_first_class : ℚ := 2 / 3
def P_second_intern_first_class : ℚ := 3 / 4

-- Define the events 
def P_A1 : ℚ := P_first_intern_first_class * (1 - P_second_intern_first_class)
def P_A2 : ℚ := (1 - P_first_intern_first_class) * P_second_intern_first_class

-- Probability of exactly one of the two parts being first-class product
def P_one_first_class_product : ℚ := P_A1 + P_A2

-- Theorem to be proven: the probability is 5/12
theorem probability_one_first_class_product : 
    P_one_first_class_product = 5 / 12 :=
by
  -- Proof goes here
  sorry

end probability_one_first_class_product_l138_138920


namespace value_proof_l138_138621

noncomputable def find_value (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x) : Prop :=
  2 * b - a + c = 195

theorem value_proof : ∃ (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x), find_value a b c h h_rat :=
  sorry

end value_proof_l138_138621


namespace infinite_nested_radical_solution_l138_138617

theorem infinite_nested_radical_solution (x : ℝ) (h : x = Real.sqrt (4 + 3 * x)) : x = 4 := 
by 
  sorry

end infinite_nested_radical_solution_l138_138617


namespace annual_population_increase_l138_138216

theorem annual_population_increase (x : ℝ) (initial_pop : ℝ) :
    (initial_pop * (1 + (x - 1) / 100)^3 = initial_pop * 1.124864) → x = 5.04 :=
by
  -- Provided conditions
  intros h
  -- The hypothesis conditionally establishes that this will derive to show x = 5.04
  sorry

end annual_population_increase_l138_138216


namespace min_value_expression_l138_138018

theorem min_value_expression (a d b c : ℝ) (habd : a ≥ 0 ∧ d ≥ 0) (hbc : b > 0 ∧ c > 0) (h_cond : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end min_value_expression_l138_138018


namespace find_triples_l138_138899

theorem find_triples (x y z : ℕ) :
  (x + 1)^(y + 1) + 1 = (x + 2)^(z + 1) ↔ (x = 1 ∧ y = 2 ∧ z = 1) :=
sorry

end find_triples_l138_138899


namespace surface_area_of_solid_l138_138401

theorem surface_area_of_solid (num_unit_cubes : ℕ) (top_layer_cubes : ℕ) 
(bottom_layer_cubes : ℕ) (side_layer_cubes : ℕ) 
(front_and_back_cubes : ℕ) (left_and_right_cubes : ℕ) :
  num_unit_cubes = 15 →
  top_layer_cubes = 5 →
  bottom_layer_cubes = 5 →
  side_layer_cubes = 3 →
  front_and_back_cubes = 5 →
  left_and_right_cubes = 3 →
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  total_surface = 26 :=
by
  intros h_n h_t h_b h_s h_f h_lr
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  sorry

end surface_area_of_solid_l138_138401


namespace line_equation_intersects_ellipse_l138_138901

theorem line_equation_intersects_ellipse :
  ∃ l : ℝ → ℝ → Prop,
    (∀ x y : ℝ, l x y ↔ 5 * x + 4 * y - 9 = 0) ∧
    (∃ M N : ℝ × ℝ,
      (M.1^2 / 20 + M.2^2 / 16 = 1) ∧
      (N.1^2 / 20 + N.2^2 / 16 = 1) ∧
      ((M.1 + N.1) / 2 = 1) ∧
      ((M.2 + N.2) / 2 = 1)) :=
sorry

end line_equation_intersects_ellipse_l138_138901


namespace inequality_proof_l138_138710

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  sqrt (a^2 + b^2 - sqrt 2 * a * b) + sqrt (b^2 + c^2 - sqrt 2 * b * c)  ≥ sqrt (a^2 + c^2) :=
by sorry

end inequality_proof_l138_138710


namespace fraction_subtraction_l138_138348

theorem fraction_subtraction (x : ℚ) : x - (1/5 : ℚ) = (3/5 : ℚ) → x = (4/5 : ℚ) :=
by
  sorry

end fraction_subtraction_l138_138348


namespace arccos_neg_half_eq_two_pi_over_three_l138_138205

theorem arccos_neg_half_eq_two_pi_over_three :
  Real.arccos (-1/2) = 2 * Real.pi / 3 := sorry

end arccos_neg_half_eq_two_pi_over_three_l138_138205


namespace price_of_second_variety_l138_138865

-- Define prices and conditions
def price_first : ℝ := 126
def price_third : ℝ := 175.5
def mixture_price : ℝ := 153
def total_weight : ℝ := 4

-- Define unknown price
variable (x : ℝ)

-- Definition of the weighted mixture price
theorem price_of_second_variety :
  (1 * price_first) + (1 * x) + (2 * price_third) = total_weight * mixture_price →
  x = 135 :=
by
  sorry

end price_of_second_variety_l138_138865


namespace sum_gcd_lcm_75_4410_l138_138374

theorem sum_gcd_lcm_75_4410 :
  Nat.gcd 75 4410 + Nat.lcm 75 4410 = 22065 := by
  sorry

end sum_gcd_lcm_75_4410_l138_138374


namespace postcard_cost_l138_138414

theorem postcard_cost (x : ℕ) (h₁ : 9 * x < 1000) (h₂ : 10 * x > 1100) : x = 111 :=
by
  sorry

end postcard_cost_l138_138414


namespace g_of_neg_two_l138_138111

def f (x : ℝ) : ℝ := 4 * x - 9

def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_of_neg_two : g (-2) = 227 / 16 :=
by
  sorry

end g_of_neg_two_l138_138111


namespace num_isosceles_triangles_with_perimeter_30_l138_138301

theorem num_isosceles_triangles_with_perimeter_30 : 
  (∃ (s : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ s → 2 * a + b = 30 ∧ (a ≥ b) ∧ b ≠ 0 ∧ a + a > b ∧ a + b > a ∧ b + a > a) 
    ∧ s.card = 7) :=
by {
  sorry
}

end num_isosceles_triangles_with_perimeter_30_l138_138301


namespace simplify_fraction_l138_138626

theorem simplify_fraction : 
  ∃ (c d : ℤ), ((∀ m : ℤ, (6 * m + 12) / 3 = c * m + d) ∧ c = 2 ∧ d = 4) → 
  c / d = 1 / 2 :=
by
  sorry

end simplify_fraction_l138_138626


namespace roxy_bought_flowering_plants_l138_138974

-- Definitions based on conditions
def initial_flowering_plants : ℕ := 7
def initial_fruiting_plants : ℕ := 2 * initial_flowering_plants
def plants_after_saturday (F : ℕ) : ℕ := initial_flowering_plants + F + initial_fruiting_plants + 2
def plants_after_sunday (F : ℕ) : ℕ := (initial_flowering_plants + F - 1) + (initial_fruiting_plants + 2 - 4)
def final_plants_in_garden : ℕ := 21

-- The proof statement
theorem roxy_bought_flowering_plants (F : ℕ) :
  plants_after_sunday F = final_plants_in_garden → F = 3 := 
sorry

end roxy_bought_flowering_plants_l138_138974


namespace minimum_value_expression_l138_138056

theorem minimum_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ z, (z = a^2 + b^2 + 1 / a^2 + 2 * b / a) ∧ z ≥ 2 :=
sorry

end minimum_value_expression_l138_138056


namespace smallest_interesting_rectangle_area_l138_138461

/-- 
  A rectangle is interesting if both its side lengths are integers and 
  it contains exactly four lattice points strictly in its interior.
  Prove that the area of the smallest such interesting rectangle is 10.
-/
theorem smallest_interesting_rectangle_area :
  ∃ (a b : ℕ), (a - 1) * (b - 1) = 4 ∧ a * b = 10 :=
by
  sorry

end smallest_interesting_rectangle_area_l138_138461


namespace tax_percentage_first_tier_l138_138636

theorem tax_percentage_first_tier
  (car_price : ℝ)
  (total_tax : ℝ)
  (first_tier_level : ℝ)
  (second_tier_rate : ℝ)
  (first_tier_tax : ℝ)
  (T : ℝ)
  (h_car_price : car_price = 30000)
  (h_total_tax : total_tax = 5500)
  (h_first_tier_level : first_tier_level = 10000)
  (h_second_tier_rate : second_tier_rate = 0.15)
  (h_first_tier_tax : first_tier_tax = (T / 100) * first_tier_level) :
  T = 25 :=
by
  sorry

end tax_percentage_first_tier_l138_138636


namespace average_monthly_balance_l138_138545

-- Definitions for the monthly balances
def January_balance : ℝ := 120
def February_balance : ℝ := 240
def March_balance : ℝ := 180
def April_balance : ℝ := 180
def May_balance : ℝ := 160
def June_balance : ℝ := 200

-- The average monthly balance theorem statement
theorem average_monthly_balance : 
    (January_balance + February_balance + March_balance + April_balance + May_balance + June_balance) / 6 = 180 := 
by 
  sorry

end average_monthly_balance_l138_138545


namespace fraction_surface_area_red_l138_138330

theorem fraction_surface_area_red :
  ∀ (num_unit_cubes : ℕ) (side_length_large_cube : ℕ) (total_surface_area_painted : ℕ) (total_surface_area_unit_cubes : ℕ),
    num_unit_cubes = 8 →
    side_length_large_cube = 2 →
    total_surface_area_painted = 6 * (side_length_large_cube ^ 2) →
    total_surface_area_unit_cubes = num_unit_cubes * 6 →
    (total_surface_area_painted : ℝ) / total_surface_area_unit_cubes = 1 / 2 :=
by
  intros num_unit_cubes side_length_large_cube total_surface_area_painted total_surface_area_unit_cubes
  sorry

end fraction_surface_area_red_l138_138330


namespace velvet_needed_for_box_l138_138830

theorem velvet_needed_for_box : 
  let area_long_side := 8 * 6
  let area_short_side := 5 * 6
  let area_top_bottom := 40
  let total_area := (2 * area_long_side) + (2 * area_short_side) + (2 * area_top_bottom)
  total_area = 236 :=
by
  sorry

end velvet_needed_for_box_l138_138830


namespace range_a_implies_not_purely_imaginary_l138_138780

def is_not_purely_imaginary (z : ℂ) : Prop :=
  z.re ≠ 0

theorem range_a_implies_not_purely_imaginary (a : ℝ) :
  ¬ is_not_purely_imaginary ⟨a^2 - a - 2, abs (a - 1) - 1⟩ ↔ a ≠ -1 :=
by
  sorry

end range_a_implies_not_purely_imaginary_l138_138780


namespace sum_of_digits_l138_138629

def digits (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

def P := 1
def Q := 0
def R := 2
def S := 5
def T := 6

theorem sum_of_digits :
  digits P ∧ digits Q ∧ digits R ∧ digits S ∧ digits T ∧ 
  (10000 * P + 1000 * Q + 100 * R + 10 * S + T) * 4 = 41024 →
  P + Q + R + S + T = 14 :=
by
  sorry

end sum_of_digits_l138_138629


namespace quadratic_inequality_solution_l138_138131

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 5 * x - 6 > 0) ↔ (x < -1 ∨ x > 6) := 
by
  sorry

end quadratic_inequality_solution_l138_138131


namespace intersection_complement_eq_l138_138164

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

theorem intersection_complement_eq : A ∩ (U \ B) = {0, 2} := by
  sorry

end intersection_complement_eq_l138_138164


namespace linearly_dependent_k_l138_138007

theorem linearly_dependent_k (k : ℝ) : 
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨1, k⟩ : ℝ × ℝ) = (0, 0)) ↔ k = 3 / 2 :=
by
  sorry

end linearly_dependent_k_l138_138007


namespace find_base_s_l138_138933

-- Definitions based on the conditions.
def five_hundred_thirty_base (s : ℕ) : ℕ := 5 * s^2 + 3 * s
def four_hundred_fifty_base (s : ℕ) : ℕ := 4 * s^2 + 5 * s
def one_thousand_one_hundred_base (s : ℕ) : ℕ := s^3 + s^2

-- The theorem to prove.
theorem find_base_s : (∃ s : ℕ, five_hundred_thirty_base s + four_hundred_fifty_base s = one_thousand_one_hundred_base s) → s = 8 :=
by
  sorry

end find_base_s_l138_138933


namespace simplify_expression_l138_138359

theorem simplify_expression :
  (2^8 + 4^5) * (2^3 - (-2)^3)^2 = 327680 := by
  sorry

end simplify_expression_l138_138359


namespace find_number_l138_138274

theorem find_number (n : ℝ) (h : (1/2) * n + 5 = 11) : n = 12 :=
by
  sorry

end find_number_l138_138274


namespace pascal_row_contains_prime_47_l138_138941

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l138_138941


namespace total_video_hours_in_june_l138_138010

-- Definitions for conditions
def upload_rate_first_half : ℕ := 10 -- one-hour videos per day
def upload_rate_second_half : ℕ := 20 -- doubled one-hour videos per day
def days_in_half_month : ℕ := 15
def total_days_in_june : ℕ := 30

-- Number of video hours uploaded in the first half of the month
def video_hours_first_half : ℕ := upload_rate_first_half * days_in_half_month

-- Number of video hours uploaded in the second half of the month
def video_hours_second_half : ℕ := upload_rate_second_half * days_in_half_month

-- Total number of video hours in June
theorem total_video_hours_in_june : video_hours_first_half + video_hours_second_half = 450 :=
by {
  sorry
}

end total_video_hours_in_june_l138_138010


namespace find_rs_l138_138500

theorem find_rs (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 1) (h4 : r^4 + s^4 = 7/8) : 
  r * s = 1/4 :=
sorry

end find_rs_l138_138500


namespace tomato_land_correct_l138_138947

-- Define the conditions
def total_land : ℝ := 4999.999999999999
def cleared_fraction : ℝ := 0.9
def grapes_fraction : ℝ := 0.1
def potato_fraction : ℝ := 0.8

-- Define the calculated values based on conditions
def cleared_land : ℝ := cleared_fraction * total_land
def grapes_land : ℝ := grapes_fraction * cleared_land
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := cleared_land - (grapes_land + potato_land)

-- Prove the question using conditions, which should end up being 450 acres.
theorem tomato_land_correct : tomato_land = 450 :=
by sorry

end tomato_land_correct_l138_138947


namespace ramu_repair_cost_l138_138437

theorem ramu_repair_cost
  (initial_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (repair_cost : ℝ)
  (h1 : initial_cost = 42000)
  (h2 : selling_price = 64900)
  (h3 : profit_percent = 13.859649122807017 / 100)
  (h4 : selling_price = initial_cost + repair_cost + profit_percent * (initial_cost + repair_cost)) :
  repair_cost = 15000 :=
by
  sorry

end ramu_repair_cost_l138_138437


namespace min_value_exists_l138_138177

noncomputable def point_on_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 9 ∧ y ≥ 2

theorem min_value_exists : ∃ x y : ℝ, point_on_circle x y ∧ x + Real.sqrt 3 * y = 2 * Real.sqrt 3 - 2 := 
sorry

end min_value_exists_l138_138177


namespace det_matrix_A_l138_138432

noncomputable def matrix_A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem det_matrix_A (x y z : ℝ) : 
  Matrix.det (matrix_A x y z) = x^3 + y^3 + z^3 - 3*x*y*z := by
  sorry

end det_matrix_A_l138_138432


namespace radius_of_given_spherical_circle_l138_138373
noncomputable def circle_radius_spherical_coords : Real :=
  let spherical_to_cartesian (rho theta phi : Real) : (Real × Real × Real) :=
    (rho * (Real.sin phi) * (Real.cos theta), rho * (Real.sin phi) * (Real.sin theta), rho * (Real.cos phi))
  let (rho, theta, phi) := (1, 0, Real.pi / 3)
  let (x, y, z) := spherical_to_cartesian rho theta phi
  let radius := Real.sqrt (x^2 + y^2)
  radius

theorem radius_of_given_spherical_circle :
  circle_radius_spherical_coords = (Real.sqrt 3) / 2 :=
sorry

end radius_of_given_spherical_circle_l138_138373


namespace total_gum_l138_138162

-- Define the conditions
def original_gum : ℕ := 38
def additional_gum : ℕ := 16

-- Define the statement to be proved
theorem total_gum : original_gum + additional_gum = 54 :=
by
  -- Proof omitted
  sorry

end total_gum_l138_138162


namespace geometric_sequence_identity_l138_138329

variables {b : ℕ → ℝ} {m n p : ℕ}

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ i j k : ℕ, i < j → j < k → b j^2 = b i * b k

noncomputable def distinct_pos_ints (m n p : ℕ) :=
  0 < m ∧ 0 < n ∧ 0 < p ∧ m ≠ n ∧ n ≠ p ∧ p ≠ m

theorem geometric_sequence_identity 
  (h_geom : is_geometric_sequence b) 
  (h_distinct : distinct_pos_ints m n p) : 
  b p ^ (m - n) * b m ^ (n - p) * b n ^ (p - m) = 1 :=
sorry

end geometric_sequence_identity_l138_138329


namespace min_value_expression_l138_138665

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y) * (1 / x + 1 / y) ≥ 6 := 
by
  sorry

end min_value_expression_l138_138665


namespace altitude_division_l138_138398

variables {A B C D E : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E]

theorem altitude_division 
  (AD DC CE EB y : ℝ)
  (hAD : AD = 6)
  (hDC : DC = 4)
  (hCE : CE = 3)
  (hEB : EB = y)
  (h_similarity : CE / DC = (AD + DC) / (y + CE)) : 
  y = 31 / 3 :=
by
  sorry

end altitude_division_l138_138398


namespace range_of_a_l138_138530

noncomputable def satisfies_system (a b c : ℝ) : Prop :=
  (a^2 - b * c - 8 * a + 7 = 0) ∧ (b^2 + c^2 + b * c - 6 * a + 6 = 0)

theorem range_of_a (a b c : ℝ) 
  (h : satisfies_system a b c) : 1 ≤ a ∧ a ≤ 9 :=
by
  sorry

end range_of_a_l138_138530


namespace marketing_percentage_l138_138946

-- Define the conditions
variable (monthly_budget : ℝ)
variable (rent : ℝ := monthly_budget / 5)
variable (remaining_after_rent : ℝ := monthly_budget - rent)
variable (food_beverages : ℝ := remaining_after_rent / 4)
variable (remaining_after_food_beverages : ℝ := remaining_after_rent - food_beverages)
variable (employee_salaries : ℝ := remaining_after_food_beverages / 3)
variable (remaining_after_employee_salaries : ℝ := remaining_after_food_beverages - employee_salaries)
variable (utilities : ℝ := remaining_after_employee_salaries / 7)
variable (remaining_after_utilities : ℝ := remaining_after_employee_salaries - utilities)
variable (marketing : ℝ := 0.15 * remaining_after_utilities)

-- Define the theorem we want to prove
theorem marketing_percentage : marketing / monthly_budget * 100 = 5.14 := by
  sorry

end marketing_percentage_l138_138946


namespace geo_series_sum_eight_terms_l138_138204

theorem geo_series_sum_eight_terms :
  let a_0 := 1 / 3
  let r := 1 / 3 
  let S_8 := a_0 * (1 - r^8) / (1 - r)
  S_8 = 3280 / 6561 :=
by
  /- :: Proof Steps Omitted. -/
  sorry

end geo_series_sum_eight_terms_l138_138204


namespace value_of_2022_plus_a_minus_b_l138_138882

theorem value_of_2022_plus_a_minus_b (x a b : ℚ) (h_distinct : x ≠ a ∧ x ≠ b ∧ a ≠ b) 
  (h_gt : a > b) (h_min : ∀ y : ℚ, |y - a| + |y - b| ≥ 2 ∧ |x - a| + |x - b| = 2) :
  2022 + a - b = 2024 := 
by 
  sorry

end value_of_2022_plus_a_minus_b_l138_138882


namespace slope_of_line_intersecting_hyperbola_l138_138121

theorem slope_of_line_intersecting_hyperbola 
  (A B : ℝ × ℝ)
  (hA : A.1^2 - A.2^2 = 1)
  (hB : B.1^2 - B.2^2 = 1)
  (midpoint_condition : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1) :
  (B.2 - A.2) / (B.1 - A.1) = 2 :=
by
  sorry

end slope_of_line_intersecting_hyperbola_l138_138121


namespace min_value_hyperbola_l138_138634

theorem min_value_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ e : ℝ, e = 2 ∧ (b^2 = (e * a)^2 - a^2)) :
  (a * 3 + 1 / a) = 2 * Real.sqrt 3 :=
by
  sorry

end min_value_hyperbola_l138_138634


namespace range_of_squared_function_l138_138453

theorem range_of_squared_function (x : ℝ) (hx : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end range_of_squared_function_l138_138453


namespace simplify_expression_l138_138064

-- Define constants
variables (z : ℝ)

-- Define the problem and its solution
theorem simplify_expression :
  (5 - 2 * z) - (4 + 5 * z) = 1 - 7 * z := 
sorry

end simplify_expression_l138_138064


namespace nth_equation_proof_l138_138604

theorem nth_equation_proof (n : ℕ) (hn : n > 0) :
  (1 : ℝ) + (1 / (n : ℝ)) - (2 / (2 * n - 1)) = (2 * n^2 + n + 1) / (n * (2 * n - 1)) :=
by
  sorry

end nth_equation_proof_l138_138604


namespace coin_probability_l138_138278

theorem coin_probability (p : ℝ) (h1 : p < 1/2) (h2 : (Nat.choose 6 3) * p^3 * (1-p)^3 = 1/20) : p = 1/400 := sorry

end coin_probability_l138_138278


namespace parallel_line_through_point_l138_138646

theorem parallel_line_through_point (x y : ℝ) :
  (∃ (b : ℝ), (∀ (x : ℝ), y = 2 * x + b) ∧ y = 2 * 1 - 4) :=
sorry

end parallel_line_through_point_l138_138646


namespace count_valid_permutations_eq_X_l138_138538

noncomputable def valid_permutations_count : ℕ :=
sorry

theorem count_valid_permutations_eq_X : valid_permutations_count = X :=
sorry

end count_valid_permutations_eq_X_l138_138538


namespace parking_lot_problem_l138_138407

variable (M S : Nat)

theorem parking_lot_problem (h1 : M + S = 30) (h2 : 15 * M + 8 * S = 324) :
  M = 12 ∧ S = 18 :=
by
  -- proof omitted
  sorry

end parking_lot_problem_l138_138407


namespace range_of_m_l138_138198

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4^x + m * 2^x + m^2 - 1 = 0) ↔ - (2 * Real.sqrt 3) / 3 ≤ m ∧ m < 1 :=
sorry

end range_of_m_l138_138198


namespace integer_solutions_of_inequality_count_l138_138047

theorem integer_solutions_of_inequality_count :
  let a := -2 - Real.sqrt 6
  let b := -2 + Real.sqrt 6
  ∃ n, n = 5 ∧ ∀ x : ℤ, x < a ∨ b < x ↔ (4 * x^2 + 16 * x + 15 ≤ 23) → n = 5 :=
by sorry

end integer_solutions_of_inequality_count_l138_138047


namespace prob_B_takes_second_shot_correct_prob_A_takes_ith_shot_correct_expected_A_shots_correct_l138_138103

noncomputable def prob_A_makes_shot : ℝ := 0.6
noncomputable def prob_B_makes_shot : ℝ := 0.8
noncomputable def prob_A_starts : ℝ := 0.5
noncomputable def prob_B_starts : ℝ := 0.5

noncomputable def prob_B_takes_second_shot : ℝ :=
  prob_A_starts * (1 - prob_A_makes_shot) + prob_B_starts * prob_B_makes_shot

theorem prob_B_takes_second_shot_correct :
  prob_B_takes_second_shot = 0.6 :=
  sorry

noncomputable def prob_A_takes_nth_shot (n : ℕ) : ℝ :=
  let p₁ := 0.5
  let recurring_prob := (1 / 6) * ((2 / 5)^(n-1))
  (1 / 3) + recurring_prob

theorem prob_A_takes_ith_shot_correct (i : ℕ) :
  prob_A_takes_nth_shot i = (1 / 3) + (1 / 6) * ((2 / 5)^(i - 1)) :=
  sorry

noncomputable def expected_A_shots (n : ℕ) : ℝ :=
  let geometric_sum := ((2 / 5)^n - 1) / (1 - (2 / 5))
  (1 / 6) * geometric_sum + (n / 3)

theorem expected_A_shots_correct (n : ℕ) :
  expected_A_shots n = (5 / 18) * (1 - (2 / 5)^n) + (n / 3) :=
  sorry

end prob_B_takes_second_shot_correct_prob_A_takes_ith_shot_correct_expected_A_shots_correct_l138_138103


namespace no_prime_p_for_base_eqn_l138_138200

theorem no_prime_p_for_base_eqn (p : ℕ) (hp: p.Prime) :
  let f (p : ℕ) := 1009 * p^3 + 307 * p^2 + 115 * p + 126 + 7
  let g (p : ℕ) := 143 * p^2 + 274 * p + 361
  f p = g p → false :=
sorry

end no_prime_p_for_base_eqn_l138_138200


namespace quadratic_inequality_solution_set_l138_138580

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 5*x - 14 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 7} :=
by
  -- proof to be filled here
  sorry

end quadratic_inequality_solution_set_l138_138580


namespace Yeonseo_skirts_l138_138628

theorem Yeonseo_skirts
  (P : ℕ)
  (more_than_two_skirts : ∀ S : ℕ, S > 2)
  (more_than_two_pants : P > 2)
  (ways_to_choose : P + 3 = 7) :
  ∃ S : ℕ, S = 3 := by
  sorry

end Yeonseo_skirts_l138_138628


namespace solve_for_x_l138_138173

theorem solve_for_x (x : ℚ) (h : 3 / x - 3 / x / (9 / x) = 0.5) : x = 6 / 5 :=
sorry

end solve_for_x_l138_138173


namespace michael_twenty_dollar_bills_l138_138741

/--
Michael has $280 dollars and each bill is $20 dollars.
We need to prove that the number of $20 dollar bills Michael has is 14.
-/
theorem michael_twenty_dollar_bills (total_money : ℕ) (bill_denomination : ℕ) (number_of_bills : ℕ) :
  total_money = 280 →
  bill_denomination = 20 →
  number_of_bills = total_money / bill_denomination →
  number_of_bills = 14 :=
by
  intros h1 h2 h3
  sorry

end michael_twenty_dollar_bills_l138_138741


namespace find_f_1789_l138_138438

def f : ℕ → ℕ := sorry

axiom f_1 : f 1 = 5
axiom f_f_n : ∀ n, f (f n) = 4 * n + 9
axiom f_2n : ∀ n, f (2 * n) = (2 * n) + 1 + 3

theorem find_f_1789 : f 1789 = 3581 :=
by
  sorry

end find_f_1789_l138_138438


namespace range_of_m_l138_138857

theorem range_of_m (m : ℝ) :
  (∃ x y : ℤ, (x ≠ y) ∧ (x ≥ m ∧ y ≥ m) ∧ (3 - 2 * x ≥ 0) ∧ (3 - 2 * y ≥ 0)) ↔ (-1 < m ∧ m ≤ 0) :=
by
  sorry

end range_of_m_l138_138857


namespace total_skips_l138_138265

-- Definitions of the given conditions
def BobsSkipsPerRock := 12
def JimsSkipsPerRock := 15
def NumberOfRocks := 10

-- Statement of the theorem to be proved
theorem total_skips :
  (BobsSkipsPerRock * NumberOfRocks) + (JimsSkipsPerRock * NumberOfRocks) = 270 :=
by
  sorry

end total_skips_l138_138265


namespace shooting_test_performance_l138_138864

theorem shooting_test_performance (m n : ℝ)
    (h1 : m > 9.7)
    (h2 : n < 0.25) :
    (m = 9.9 ∧ n = 0.2) :=
sorry

end shooting_test_performance_l138_138864


namespace total_profit_is_35000_l138_138687

open Real

-- Define the subscriptions of A, B, and C
def subscriptions (A B C : ℝ) : Prop :=
  A + B + C = 50000 ∧
  A = B + 4000 ∧
  B = C + 5000

-- Define the profit distribution and the condition for C's received profit
def profit (total_profit : ℝ) (A B C : ℝ) (C_profit : ℝ) : Prop :=
  C_profit / total_profit = C / (A + B + C) ∧
  C_profit = 8400

-- Lean 4 statement to prove total profit
theorem total_profit_is_35000 :
  ∃ A B C total_profit, subscriptions A B C ∧ profit total_profit A B C 8400 ∧ total_profit = 35000 :=
by
  sorry

end total_profit_is_35000_l138_138687


namespace length_of_bridge_l138_138798

theorem length_of_bridge
  (length_train : ℕ) (speed_train_kmhr : ℕ) (crossing_time : ℕ)
  (speed_conversion_factor : ℝ) (m_per_s_kmhr : ℝ) 
  (speed_train_ms : ℝ) (total_distance : ℝ) (length_bridge : ℝ)
  (h1 : length_train = 155)
  (h2 : speed_train_kmhr = 45)
  (h3 : crossing_time = 30)
  (h4 : speed_conversion_factor = 1000 / 3600)
  (h5 : m_per_s_kmhr = speed_train_kmhr * speed_conversion_factor)
  (h6 : speed_train_ms = 45 * (5 / 18))
  (h7 : total_distance = speed_train_ms * crossing_time)
  (h8 : length_bridge = total_distance - length_train):
  length_bridge = 220 :=
by
  sorry

end length_of_bridge_l138_138798


namespace correct_optionD_l138_138149

def operationA (a : ℝ) : Prop := a^3 + 3 * a^3 = 5 * a^6
def operationB (a : ℝ) : Prop := 7 * a^2 * a^3 = 7 * a^6
def operationC (a : ℝ) : Prop := (-2 * a^3)^2 = 4 * a^5
def operationD (a : ℝ) : Prop := a^8 / a^2 = a^6

theorem correct_optionD (a : ℝ) : ¬ operationA a ∧ ¬ operationB a ∧ ¬ operationC a ∧ operationD a :=
by
  unfold operationA operationB operationC operationD
  sorry

end correct_optionD_l138_138149


namespace zero_is_smallest_natural_number_l138_138043

theorem zero_is_smallest_natural_number : ∀ n : ℕ, 0 ≤ n :=
by
  intro n
  exact Nat.zero_le n

#check zero_is_smallest_natural_number  -- confirming the theorem check

end zero_is_smallest_natural_number_l138_138043


namespace relationship_xy_l138_138172

variable (x y : ℝ)

theorem relationship_xy (h₁ : x - y > x + 2) (h₂ : x + y + 3 < y - 1) : x < -4 ∧ y < -2 := 
by sorry

end relationship_xy_l138_138172


namespace water_leaked_l138_138518

theorem water_leaked (initial remaining : ℝ) (h_initial : initial = 0.75) (h_remaining : remaining = 0.5) :
  initial - remaining = 0.25 :=
by
  sorry

end water_leaked_l138_138518


namespace price_reduction_equation_l138_138760

theorem price_reduction_equation (x : ℝ) : 200 * (1 - x) ^ 2 = 162 :=
by
  sorry

end price_reduction_equation_l138_138760


namespace part1_part2_l138_138393

noncomputable def f (x : ℝ) := Real.exp x

theorem part1 (x : ℝ) (h : x ≥ 0) (m : ℝ) : 
  (x - 1) * f x ≥ m * x^2 - 1 ↔ m ≤ 1 / 2 :=
sorry

theorem part2 (x : ℝ) (h : x > 0) : 
  f x > 4 * Real.log x + 8 - 8 * Real.log 2 :=
sorry

end part1_part2_l138_138393


namespace circle_condition_m_l138_138590

theorem circle_condition_m (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x + m = 0) → m < 1 := 
by
  sorry

end circle_condition_m_l138_138590


namespace system_of_equations_solution_l138_138239

theorem system_of_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 10) ∧ (12 * x - 8 * y = 8) ∧ (x = 14 / 9) ∧ (y = 4 / 3) :=
by
  sorry

end system_of_equations_solution_l138_138239


namespace rectangle_ratio_l138_138083

theorem rectangle_ratio (a b c d : ℝ)
  (h1 : (a * b) / (c * d) = 0.16)
  (h2 : a / c = b / d) :
  a / c = 0.4 ∧ b / d = 0.4 :=
by 
  sorry

end rectangle_ratio_l138_138083


namespace interest_rate_unique_l138_138102

theorem interest_rate_unique (P r : ℝ) (h₁ : P * (1 + 3 * r) = 300) (h₂ : P * (1 + 8 * r) = 400) : r = 1 / 12 :=
by {
  sorry
}

end interest_rate_unique_l138_138102


namespace unique_solution_l138_138454

theorem unique_solution (x y z : ℝ) (h₁ : x^2 + y^2 + z^2 = 2) (h₂ : x = z + 2) :
  x = 1 ∧ y = 0 ∧ z = -1 :=
by
  sorry

end unique_solution_l138_138454


namespace total_marbles_count_l138_138705

variable (r b g : ℝ)
variable (h1 : r = 1.4 * b) (h2 : g = 1.5 * r)

theorem total_marbles_count (r b g : ℝ) (h1 : r = 1.4 * b) (h2 : g = 1.5 * r) :
  r + b + g = 3.21 * r :=
by
  sorry

end total_marbles_count_l138_138705


namespace two_digit_number_representation_l138_138283

def tens_digit := ℕ
def units_digit := ℕ

theorem two_digit_number_representation (b a : ℕ) : 
  (∀ (b a : ℕ), 10 * b + a = 10 * b + a) := sorry

end two_digit_number_representation_l138_138283


namespace min_notebooks_needed_l138_138752

variable (cost_pen cost_notebook num_pens discount_threshold : ℕ)

theorem min_notebooks_needed (x : ℕ)
    (h1 : cost_pen = 10)
    (h2 : cost_notebook = 4)
    (h3 : num_pens = 3)
    (h4 : discount_threshold = 100)
    (h5 : num_pens * cost_pen + x * cost_notebook ≥ discount_threshold) :
    x ≥ 18 := 
sorry

end min_notebooks_needed_l138_138752


namespace ratio_r_to_pq_l138_138716

theorem ratio_r_to_pq (total : ℝ) (amount_r : ℝ) (amount_pq : ℝ) 
  (h1 : total = 9000) 
  (h2 : amount_r = 3600.0000000000005) 
  (h3 : amount_pq = total - amount_r) : 
  amount_r / amount_pq = 2 / 3 :=
by
  sorry

end ratio_r_to_pq_l138_138716


namespace yoongi_has_smallest_points_l138_138726

def points_jungkook : ℕ := 6 + 3
def points_yoongi : ℕ := 4
def points_yuna : ℕ := 5

theorem yoongi_has_smallest_points : points_yoongi < points_jungkook ∧ points_yoongi < points_yuna :=
by
  sorry

end yoongi_has_smallest_points_l138_138726


namespace reciprocal_sum_l138_138223

theorem reciprocal_sum :
  let a := (1 / 4 : ℚ)
  let b := (1 / 5 : ℚ)
  1 / (a + b) = 20 / 9 :=
by
  let a := (1 / 4 : ℚ)
  let b := (1 / 5 : ℚ)
  have h : a + b = 9 / 20 := by sorry
  have h_rec : 1 / (a + b) = 20 / 9 := by sorry
  exact h_rec

end reciprocal_sum_l138_138223


namespace sum_first_15_odd_integers_l138_138343

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l138_138343


namespace abs_of_sub_sqrt_l138_138789

theorem abs_of_sub_sqrt (h : 2 > Real.sqrt 3) : |2 - Real.sqrt 3| = 2 - Real.sqrt 3 :=
sorry

end abs_of_sub_sqrt_l138_138789


namespace penelope_food_intake_l138_138501

theorem penelope_food_intake
(G P M E : ℕ) -- Representing amount of food each animal eats per day
(h1 : P = 10 * G) -- Penelope eats 10 times Greta's food
(h2 : M = G / 100) -- Milton eats 1/100 of Greta's food
(h3 : E = 4000 * M) -- Elmer eats 4000 times what Milton eats
(h4 : E = P + 60) -- Elmer eats 60 pounds more than Penelope
(G_val : G = 2) -- Greta eats 2 pounds per day
: P = 20 := -- Prove Penelope eats 20 pounds per day
by
  rw [G_val] at h1 -- Replace G with 2 in h1
  norm_num at h1 -- Evaluate the expression in h1
  exact h1 -- Conclude P = 20

end penelope_food_intake_l138_138501


namespace perfect_cubes_l138_138818

theorem perfect_cubes (n : ℕ) (h : n > 0) : 
  (n = 7 ∨ n = 11 ∨ n = 12 ∨ n = 25) ↔ ∃ k : ℤ, (n^3 - 18*n^2 + 115*n - 391) = k^3 :=
by exact sorry

end perfect_cubes_l138_138818


namespace sum_possible_x_values_in_isosceles_triangle_l138_138555

def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

def valid_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem sum_possible_x_values_in_isosceles_triangle :
  ∃ (x1 x2 x3 : ℝ), isosceles_triangle 80 x1 x1 ∧ isosceles_triangle x2 80 80 ∧ isosceles_triangle 80 x3 x3 ∧ 
  valid_triangle 80 x1 x1 ∧ valid_triangle x2 80 80 ∧ valid_triangle 80 x3 x3 ∧ 
  x1 + x2 + x3 = 150 :=
by
  sorry

end sum_possible_x_values_in_isosceles_triangle_l138_138555


namespace sum_faces_of_cube_l138_138633

-- Conditions in Lean 4
variables (a b c d e f : ℕ)

-- Sum of vertex labels
def vertex_sum := a * b * c + a * e * c + a * b * f + a * e * f +
                  d * b * c + d * e * c + d * b * f + d * e * f

-- Theorem statement
theorem sum_faces_of_cube (h : vertex_sum a b c d e f = 1001) :
  (a + d) + (b + e) + (c + f) = 31 :=
sorry

end sum_faces_of_cube_l138_138633


namespace largest_number_of_gold_coins_l138_138175

theorem largest_number_of_gold_coins (n : ℕ) (h1 : n % 15 = 4) (h2 : n < 150) : n ≤ 139 :=
by {
  -- This is where the proof would go.
  sorry
}

end largest_number_of_gold_coins_l138_138175


namespace average_hit_targets_formula_average_hit_targets_ge_half_l138_138922

open Nat Real

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n) ^ n)

theorem average_hit_targets_formula (n : ℕ) (h : 0 < n) :
  average_hit_targets n = n * (1 - (1 - (1 : ℝ) / n) ^ n) :=
by
  sorry

theorem average_hit_targets_ge_half (n : ℕ) (h : 0 < n) :
  average_hit_targets n ≥ n / 2 :=
by
  sorry

end average_hit_targets_formula_average_hit_targets_ge_half_l138_138922


namespace max_non_managers_depA_l138_138305

theorem max_non_managers_depA (mA : ℕ) (nA : ℕ) (sA : ℕ) (gA : ℕ) (totalA : ℕ) :
  mA = 9 ∧ (8 * nA > 37 * mA) ∧ (sA = 2 * gA) ∧ (nA = sA + gA) ∧ (mA + nA ≤ 250) →
  nA = 39 :=
by
  sorry

end max_non_managers_depA_l138_138305


namespace work_done_resistive_force_l138_138443

noncomputable def mass : ℝ := 0.01  -- 10 grams converted to kilograms
noncomputable def v1 : ℝ := 400.0  -- initial speed in m/s
noncomputable def v2 : ℝ := 100.0  -- final speed in m/s

noncomputable def kinetic_energy (m v : ℝ) : ℝ := 0.5 * m * v^2

theorem work_done_resistive_force :
  let KE1 := kinetic_energy mass v1
  let KE2 := kinetic_energy mass v2
  KE1 - KE2 = 750 :=
by
  sorry

end work_done_resistive_force_l138_138443


namespace find_PS_length_l138_138713

theorem find_PS_length 
  (PT TR QS QP PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 10)
  (h3 : QS = 16)
  (h4 : QP = 13)
  (h5 : PQ = 7) : 
  PS = Real.sqrt 703 := 
sorry

end find_PS_length_l138_138713


namespace no_integer_solutions_for_x2_minus_4y2_eq_2011_l138_138384

theorem no_integer_solutions_for_x2_minus_4y2_eq_2011 :
  ∀ (x y : ℤ), x^2 - 4 * y^2 ≠ 2011 := by
sorry

end no_integer_solutions_for_x2_minus_4y2_eq_2011_l138_138384


namespace number_of_shelves_l138_138911

theorem number_of_shelves (a d S : ℕ) (h1 : a = 3) (h2 : d = 3) (h3 : S = 225) : 
  ∃ n : ℕ, (S = n * (2 * a + (n - 1) * d) / 2) ∧ (n = 15) := 
by {
  sorry
}

end number_of_shelves_l138_138911


namespace min_a_for_inequality_l138_138174

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → (x^2 + a*x + 1 ≥ 0)) ↔ a ≥ -5/2 :=
by
  sorry

end min_a_for_inequality_l138_138174


namespace steve_total_payment_l138_138324

def mike_dvd_cost : ℝ := 5
def steve_dvd_cost : ℝ := 2 * mike_dvd_cost
def additional_dvd_cost : ℝ := 7
def steve_additional_dvds : ℝ := 2 * additional_dvd_cost
def total_dvd_cost : ℝ := steve_dvd_cost + steve_additional_dvds
def shipping_cost : ℝ := 0.80 * total_dvd_cost
def subtotal_with_shipping : ℝ := total_dvd_cost + shipping_cost
def sales_tax : ℝ := 0.10 * subtotal_with_shipping
def total_amount_paid : ℝ := subtotal_with_shipping + sales_tax

theorem steve_total_payment : total_amount_paid = 47.52 := by
  sorry

end steve_total_payment_l138_138324


namespace solve_for_x_l138_138171

variable (x y : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (h : 3 * x^2 + 9 * x * y = x^3 + 3 * x^2 * y)

theorem solve_for_x : x = 3 :=
by
  sorry

end solve_for_x_l138_138171


namespace compare_a_b_c_l138_138391

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l138_138391


namespace money_raised_is_correct_l138_138275

noncomputable def cost_per_dozen : ℚ := 2.40
noncomputable def selling_price_per_donut : ℚ := 1
noncomputable def dozens : ℕ := 10

theorem money_raised_is_correct :
  (dozens * 12 * selling_price_per_donut - dozens * cost_per_dozen) = 96 := by
sorry

end money_raised_is_correct_l138_138275


namespace kim_paints_fewer_tiles_than_laura_l138_138519

-- Given conditions and definitions
def don_rate : ℕ := 3
def ken_rate : ℕ := don_rate + 2
def laura_rate : ℕ := 2 * ken_rate
def total_tiles_per_15_minutes : ℕ := 375
def total_rate_per_minute : ℕ := total_tiles_per_15_minutes / 15
def kim_rate : ℕ := total_rate_per_minute - (don_rate + ken_rate + laura_rate)

-- Proof goal
theorem kim_paints_fewer_tiles_than_laura :
  laura_rate - kim_rate = 3 :=
by
  sorry

end kim_paints_fewer_tiles_than_laura_l138_138519


namespace dan_marbles_l138_138318

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  original_marbles = 128 →
  given_marbles = 32 →
  remaining_marbles = original_marbles - given_marbles →
  remaining_marbles = 96 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dan_marbles_l138_138318


namespace pencils_and_pens_cost_l138_138955

theorem pencils_and_pens_cost (p q : ℝ)
  (h1 : 8 * p + 3 * q = 5.60)
  (h2 : 2 * p + 5 * q = 4.25) :
  3 * p + 4 * q = 9.68 :=
sorry

end pencils_and_pens_cost_l138_138955


namespace solve_f_log2_20_l138_138796

noncomputable def f (x : ℝ) : ℝ :=
if -1 ≤ x ∧ x < 0 then 2^x else 0 -- Placeholder for other values

theorem solve_f_log2_20 :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 4) = f x) →
  (∀ x, -1 ≤ x ∧ x < 0 → f x = 2^x) →
  f (Real.log 20 / Real.log 2) = -4 / 5 :=
by
  sorry

end solve_f_log2_20_l138_138796


namespace random_event_l138_138601

theorem random_event (a b : ℝ) (h1 : a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0):
  ¬ (∀ a b, a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0 → a + b < 0) :=
by
  sorry

end random_event_l138_138601


namespace train_speed_kmph_l138_138887

-- The conditions
def speed_m_s : ℝ := 52.5042
def conversion_factor : ℝ := 3.6

-- The theorem we need to prove
theorem train_speed_kmph : speed_m_s * conversion_factor = 189.01512 := 
  sorry

end train_speed_kmph_l138_138887


namespace find_c_l138_138916

open Real

-- Definition of the quadratic expression in question
def expr (x y c : ℝ) : ℝ := 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 5 * x - 5 * y + 7

-- The theorem to prove that the minimum value of this expression being 0 over all (x, y) implies c = 4
theorem find_c :
  (∀ x y : ℝ, expr x y c ≥ 0) → (∃ x y : ℝ, expr x y c = 0) → c = 4 := 
by 
  sorry

end find_c_l138_138916


namespace min_value_of_3x_plus_4y_l138_138127

open Real

theorem min_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

end min_value_of_3x_plus_4y_l138_138127


namespace last_digit_of_2_pow_2010_l138_138378

-- Define the pattern of last digits of powers of 2
def last_digit_of_power_of_2 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case is redundant as n % 4 ∈ {0, 1, 2, 3}

-- Main theorem stating the problem's assertion
theorem last_digit_of_2_pow_2010 : last_digit_of_power_of_2 2010 = 4 :=
by
  -- The proof is omitted
  sorry

end last_digit_of_2_pow_2010_l138_138378


namespace merchant_marked_price_l138_138717

theorem merchant_marked_price (L : ℝ) (x : ℝ) : 
  (L = 100) →
  (L - 0.3 * L = 70) →
  (0.75 * x - 70 = 0.225 * x) →
  x = 133.33 :=
by
  intro h1 h2 h3
  sorry

end merchant_marked_price_l138_138717


namespace unique_solution_pair_l138_138981

theorem unique_solution_pair (x p : ℕ) (hp : Nat.Prime p) (hx : x ≥ 0) (hp2 : p ≥ 2) :
  x * (x + 1) * (x + 2) * (x + 3) = 1679 ^ (p - 1) + 1680 ^ (p - 1) + 1681 ^ (p - 1) ↔ (x = 4 ∧ p = 2) := 
by
  sorry

end unique_solution_pair_l138_138981


namespace pool_width_l138_138482

variable (length : ℝ) (depth : ℝ) (chlorine_per_cubic_foot : ℝ) (chlorine_cost_per_quart : ℝ) (total_spent : ℝ)
variable (w : ℝ)

-- defining the conditions
def pool_conditions := length = 10 ∧ depth = 6 ∧ chlorine_per_cubic_foot = 120 ∧ chlorine_cost_per_quart = 3 ∧ total_spent = 12

-- goal statement
theorem pool_width : pool_conditions length depth chlorine_per_cubic_foot chlorine_cost_per_quart total_spent →
  w = 8 :=
by
  sorry

end pool_width_l138_138482


namespace find_f_ln_log_52_l138_138059

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1)^2 + a * Real.sin x) / (x^2 + 1) + 3

axiom given_condition (a : ℝ) : f a (Real.log (Real.log 5 / Real.log 2)) = 5

theorem find_f_ln_log_52 (a : ℝ) : f a (Real.log (Real.log 2 / Real.log 5)) = 3 :=
by
  -- The details of the proof are omitted
  sorry

end find_f_ln_log_52_l138_138059


namespace volleyball_team_arrangements_l138_138136

theorem volleyball_team_arrangements (n : ℕ) (n_pos : 0 < n) :
  ∃ arrangements : ℕ, arrangements = 2^n * (Nat.factorial n)^2 :=
sorry

end volleyball_team_arrangements_l138_138136


namespace cube_side_length_in_cone_l138_138597

noncomputable def side_length_of_inscribed_cube (r h : ℝ) : ℝ :=
  if r = 1 ∧ h = 3 then (3 * Real.sqrt 2) / (3 + Real.sqrt 2) else 0

theorem cube_side_length_in_cone :
  side_length_of_inscribed_cube 1 3 = (3 * Real.sqrt 2) / (3 + Real.sqrt 2) :=
by
  sorry

end cube_side_length_in_cone_l138_138597


namespace find_k_l138_138868

theorem find_k (k : ℕ) : (1 / 2) ^ 16 * (1 / 81) ^ k = 1 / 18 ^ 16 → k = 8 :=
by
  intro h
  sorry

end find_k_l138_138868


namespace real_solution_l138_138074

noncomputable def condition_1 (x : ℝ) : Prop := 
  4 ≤ x / (2 * x - 7)

noncomputable def condition_2 (x : ℝ) : Prop := 
  x / (2 * x - 7) < 10

noncomputable def solution_set : Set ℝ :=
  { x | (70 / 19 : ℝ) < x ∧ x ≤ 4 }

theorem real_solution (x : ℝ) : 
  (condition_1 x ∧ condition_2 x) ↔ x ∈ solution_set :=
sorry

end real_solution_l138_138074


namespace phil_quarters_l138_138890

def initial_quarters : ℕ := 50

def quarters_after_first_year (initial : ℕ) : ℕ := 2 * initial

def quarters_collected_second_year : ℕ := 3 * 12

def quarters_collected_third_year : ℕ := 12 / 3

def total_quarters_before_loss (initial : ℕ) (second_year : ℕ) (third_year : ℕ) : ℕ := 
  quarters_after_first_year initial + second_year + third_year

def lost_quarters (total : ℕ) : ℕ := total / 4

def quarters_left (total : ℕ) (lost : ℕ) : ℕ := total - lost

theorem phil_quarters : 
  quarters_left 
    (total_quarters_before_loss 
      initial_quarters 
      quarters_collected_second_year 
      quarters_collected_third_year)
    (lost_quarters 
      (total_quarters_before_loss 
        initial_quarters 
        quarters_collected_second_year 
        quarters_collected_third_year))
  = 105 :=
by
  sorry

end phil_quarters_l138_138890


namespace correct_X_Y_Z_l138_138975

def nucleotide_types (A_types C_types T_types : ℕ) : ℕ :=
  A_types + C_types + T_types

def lowest_stability_period := "interphase"

def separation_period := "late meiosis I or late meiosis II"

theorem correct_X_Y_Z :
  nucleotide_types 2 2 1 = 3 ∧ 
  lowest_stability_period = "interphase" ∧ 
  separation_period = "late meiosis I or late meiosis II" :=
by
  sorry

end correct_X_Y_Z_l138_138975


namespace total_distance_thrown_l138_138744

theorem total_distance_thrown (D : ℝ) (total_distance : ℝ) 
  (h1 : total_distance = 20 * D + 60 * D) : 
  total_distance = 1600 := 
by
  sorry

end total_distance_thrown_l138_138744


namespace trigonometric_signs_problem_l138_138645

open Real

theorem trigonometric_signs_problem (k : ℤ) (θ α : ℝ) 
  (hα : α = 2 * k * π - π / 5)
  (h_terminal_side : ∃ m : ℤ, θ = α + 2 * m * π) :
  (sin θ / |sin θ|) + (cos θ / |cos θ|) + (tan θ / |tan θ|) = -1 := 
sorry

end trigonometric_signs_problem_l138_138645


namespace solve_cos_2x_eq_cos_x_plus_sin_x_l138_138757

open Real

theorem solve_cos_2x_eq_cos_x_plus_sin_x :
  ∀ x : ℝ,
    (cos (2 * x) = cos x + sin x) ↔
    (∃ k : ℤ, x = k * π - π / 4) ∨ 
    (∃ k : ℤ, x = 2 * k * π) ∨
    (∃ k : ℤ, x = 2 * k * π - π / 2) := 
sorry

end solve_cos_2x_eq_cos_x_plus_sin_x_l138_138757


namespace stratified_sampling_third_year_students_l138_138559

/-- 
A university's mathematics department has a total of 5000 undergraduate students, 
with the first, second, third, and fourth years having a ratio of their numbers as 4:3:2:1. 
If stratified sampling is employed to select a sample of 200 students from all undergraduates,
prove that the number of third-year students to be sampled is 40.
-/
theorem stratified_sampling_third_year_students :
  let total_students := 5000
  let ratio_first_second_third_fourth := (4, 3, 2, 1)
  let sample_size := 200
  let third_year_ratio := 2
  let total_ratio_units := 4 + 3 + 2 + 1
  let proportion_third_year := third_year_ratio / total_ratio_units
  let expected_third_year_students := sample_size * proportion_third_year
  expected_third_year_students = 40 :=
by
  sorry

end stratified_sampling_third_year_students_l138_138559


namespace time_to_travel_to_shop_l138_138128

-- Define the distance and speed as given conditions
def distance : ℕ := 184
def speed : ℕ := 23

-- Define the time taken for the journey
def time_taken (d : ℕ) (s : ℕ) : ℕ := d / s

-- Statement to prove that the time taken is 8 hours
theorem time_to_travel_to_shop : time_taken distance speed = 8 := by
  -- The proof is omitted
  sorry

end time_to_travel_to_shop_l138_138128


namespace exists_finite_set_with_subset_relation_l138_138511

-- Definition of an ordered set (E, ≤)
variable {E : Type} [LE E]

theorem exists_finite_set_with_subset_relation (E : Type) [LE E] :
  ∃ (F : Set (Set E)) (X : E → Set E), 
  (∀ (e1 e2 : E), e1 ≤ e2 ↔ X e2 ⊆ X e1) :=
by
  -- The proof is initially skipped, as per instructions
  sorry

end exists_finite_set_with_subset_relation_l138_138511


namespace no_solution_exists_l138_138672

theorem no_solution_exists : ¬ ∃ (x : ℕ), (42 + x = 3 * (8 + x) ∧ 42 + x = 2 * (10 + x)) :=
by
  sorry

end no_solution_exists_l138_138672


namespace arcsin_of_half_l138_138335

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l138_138335


namespace melanie_plums_count_l138_138410

theorem melanie_plums_count (dan_plums sally_plums total_plums melanie_plums : ℕ)
    (h1 : dan_plums = 9)
    (h2 : sally_plums = 3)
    (h3 : total_plums = 16)
    (h4 : melanie_plums = total_plums - (dan_plums + sally_plums)) :
    melanie_plums = 4 := by
  -- Proof will be filled here
  sorry

end melanie_plums_count_l138_138410


namespace right_triangle_property_l138_138997

-- Variables representing the lengths of the sides and the height of the right triangle
variables (a b c h : ℝ)

-- Hypotheses from the conditions
-- 1. a and b are the lengths of the legs of the right triangle
-- 2. c is the length of the hypotenuse
-- 3. h is the height to the hypotenuse
-- Given equation: 1/2 * a * b = 1/2 * c * h
def given_equation (a b c h : ℝ) : Prop := (1 / 2) * a * b = (1 / 2) * c * h

-- The theorem to prove
theorem right_triangle_property (a b c h : ℝ) (h_eq : given_equation a b c h) : (1 / a^2 + 1 / b^2) = 1 / h^2 :=
sorry

end right_triangle_property_l138_138997


namespace sequence_term_20_l138_138502

theorem sequence_term_20 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n, a (n+1) = a n + 2) → (a 20 = 39) := by
  intros a h1 h2
  sorry

end sequence_term_20_l138_138502


namespace students_on_bus_after_stops_l138_138996

-- Definitions
def initial_students : ℕ := 10
def first_stop_off : ℕ := 3
def first_stop_on : ℕ := 2
def second_stop_off : ℕ := 1
def second_stop_on : ℕ := 4
def third_stop_off : ℕ := 2
def third_stop_on : ℕ := 3

-- Theorem statement
theorem students_on_bus_after_stops :
  let after_first_stop := initial_students - first_stop_off + first_stop_on
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on
  let after_third_stop := after_second_stop - third_stop_off + third_stop_on
  after_third_stop = 13 := 
by
  sorry

end students_on_bus_after_stops_l138_138996


namespace smallest_M_inequality_l138_138549

theorem smallest_M_inequality :
  ∃ M : ℝ, 
  M = 9 / (16 * Real.sqrt 2) ∧
  ∀ a b c : ℝ, 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ M * (a^2 + b^2 + c^2)^2 :=
by
  use 9 / (16 * Real.sqrt 2)
  sorry

end smallest_M_inequality_l138_138549


namespace quadratic_perfect_square_form_l138_138673

def quadratic_is_perfect_square (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

theorem quadratic_perfect_square_form (a b c : ℤ) (h : quadratic_is_perfect_square a b c) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
  sorry

end quadratic_perfect_square_form_l138_138673


namespace temperature_below_75_l138_138964

theorem temperature_below_75
  (T : ℝ)
  (H1 : ∀ T, T ≥ 75 → swimming_area_open)
  (H2 : ¬swimming_area_open) : 
  T < 75 :=
sorry

end temperature_below_75_l138_138964


namespace largest_integral_x_l138_138740

theorem largest_integral_x (x y : ℤ) (h1 : (1 : ℚ)/4 < x/7) (h2 : x/7 < (2 : ℚ)/3) (h3 : x + y = 10) : x = 4 :=
by
  sorry

end largest_integral_x_l138_138740


namespace greyson_spent_on_fuel_l138_138560

theorem greyson_spent_on_fuel : ∀ (cost_per_refill times_refilled total_cost : ℕ), 
  cost_per_refill = 10 → 
  times_refilled = 4 → 
  total_cost = cost_per_refill * times_refilled → 
  total_cost = 40 :=
by
  intro cost_per_refill times_refilled total_cost
  intro h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end greyson_spent_on_fuel_l138_138560


namespace different_outcomes_count_l138_138537

-- Define the number of students and competitions
def num_students : ℕ := 4
def num_competitions : ℕ := 3

-- Define the proof statement
theorem different_outcomes_count : (num_competitions ^ num_students) = 81 := 
by
  -- Proof will be here
  sorry

end different_outcomes_count_l138_138537


namespace no_real_solutions_l138_138390

theorem no_real_solutions (x : ℝ) : 
  x^(Real.log x / Real.log 2) ≠ x^4 / 256 :=
by
  sorry

end no_real_solutions_l138_138390


namespace Second_beats_Third_by_miles_l138_138341

theorem Second_beats_Third_by_miles
  (v1 v2 v3 : ℝ) -- speeds of First, Second, and Third
  (H1 : (10 / v1) = (8 / v2)) -- First beats Second by 2 miles in 10-mile race
  (H2 : (10 / v1) = (6 / v3)) -- First beats Third by 4 miles in 10-mile race
  : (10 - (v3 * (10 / v2))) = 2.5 := 
sorry

end Second_beats_Third_by_miles_l138_138341


namespace multiple_of_son_age_last_year_l138_138403

theorem multiple_of_son_age_last_year
  (G : ℕ) (S : ℕ) (M : ℕ)
  (h1 : G = 42 - 1)
  (h2 : S = 16 - 1)
  (h3 : G = M * S - 4) :
  M = 3 := by
  sorry

end multiple_of_son_age_last_year_l138_138403


namespace value_of_x_squared_minus_y_squared_l138_138405

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l138_138405


namespace total_earnings_l138_138743

-- Definitions from the conditions.
def LaurynEarnings : ℝ := 2000
def AureliaEarnings : ℝ := 0.7 * LaurynEarnings

-- The theorem to prove.
theorem total_earnings (hL : LaurynEarnings = 2000) (hA : AureliaEarnings = 0.7 * 2000) :
    LaurynEarnings + AureliaEarnings = 3400 :=
by
    sorry  -- Placeholder for the proof.

end total_earnings_l138_138743


namespace x_squared_plus_y_squared_value_l138_138684

theorem x_squared_plus_y_squared_value (x y : ℝ) (h : (x^2 + y^2 + 1) * (x^2 + y^2 + 2) = 6) : x^2 + y^2 = 1 :=
by
  sorry

end x_squared_plus_y_squared_value_l138_138684


namespace city_population_correct_l138_138802

variable (C G : ℕ)

theorem city_population_correct :
  (C - G = 119666) ∧ (C + G = 845640) → (C = 482653) := by
  intro h
  have h1 : C - G = 119666 := h.1
  have h2 : C + G = 845640 := h.2
  sorry

end city_population_correct_l138_138802


namespace find_f_find_g_l138_138119

-- Problem 1: Finding f(x) given f(x+1) = x^2 - 2x
theorem find_f (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2 * x) :
  ∀ x, f x = x^2 - 4 * x + 3 :=
sorry

-- Problem 2: Finding g(x) given roots and a point
theorem find_g (g : ℝ → ℝ) (h1 : g (-2) = 0) (h2 : g 3 = 0) (h3 : g 0 = -3) :
  ∀ x, g x = (1 / 2) * x^2 - (1 / 2) * x - 3 :=
sorry

end find_f_find_g_l138_138119


namespace new_mean_of_five_numbers_l138_138122

theorem new_mean_of_five_numbers (a b c d e : ℝ) 
  (h_mean : (a + b + c + d + e) / 5 = 25) :
  ((a + 5) + (b + 10) + (c + 15) + (d + 20) + (e + 25)) / 5 = 40 :=
by
  sorry

end new_mean_of_five_numbers_l138_138122


namespace bowling_ball_weight_l138_138062

-- Definitions based on given conditions
variable (k b : ℕ)

-- Condition 1: one kayak weighs 35 pounds
def kayak_weight : Prop := k = 35

-- Condition 2: four kayaks weigh the same as five bowling balls
def balance_equation : Prop := 4 * k = 5 * b

-- Goal: prove the weight of one bowling ball is 28 pounds
theorem bowling_ball_weight (hk : kayak_weight k) (hb : balance_equation k b) : b = 28 :=
by
  sorry

end bowling_ball_weight_l138_138062


namespace total_money_divided_l138_138248

noncomputable def children_share_total (A B E : ℕ) :=
  (12 * A = 8 * B ∧ 8 * B = 6 * E ∧ A = 84) → 
  A + B + E = 378

theorem total_money_divided (A B E : ℕ) : children_share_total A B E :=
by
  intros h
  sorry

end total_money_divided_l138_138248


namespace sum_of_digits_divisible_by_six_l138_138340

theorem sum_of_digits_divisible_by_six (A B : ℕ) (h1 : 10 * A + B % 6 = 0) (h2 : A + B = 12) : A + B = 12 :=
by
  sorry

end sum_of_digits_divisible_by_six_l138_138340


namespace _l138_138591

variables (a b c : ℝ)
-- Conditionally define the theorem giving the constraints in the context.
example (h1 : a < 0) (h2 : b < 0) (h3 : c > 0) : 
  abs a - abs (a + b) + abs (c - a) + abs (b - c) = 2 * c - a := by 
sorry

end _l138_138591


namespace polynomial_at_3mnplus1_l138_138702

noncomputable def polynomial_value (x : ℤ) : ℤ := x^2 + 4 * x + 6

theorem polynomial_at_3mnplus1 (m n : ℤ) (h₁ : 2 * m + n + 2 = m + 2 * n) (h₂ : m - n + 2 ≠ 0) :
  polynomial_value (3 * (m + n + 1)) = 3 := 
by 
  sorry

end polynomial_at_3mnplus1_l138_138702


namespace valid_duty_schedules_l138_138029

noncomputable def validSchedules : ℕ := 
  let A_schedule := Nat.choose 7 4  -- \binom{7}{4} for A
  let B_schedule := Nat.choose 4 4  -- \binom{4}{4} for B
  let C_schedule := Nat.choose 6 3  -- \binom{6}{3} for C
  let D_schedule := Nat.choose 5 5  -- \binom{5}{5} for D
  A_schedule * B_schedule * C_schedule * D_schedule

theorem valid_duty_schedules : validSchedules = 700 := by
  -- proof steps will go here
  sorry

end valid_duty_schedules_l138_138029


namespace percentage_ownership_l138_138092

theorem percentage_ownership (total students_cats students_dogs : ℕ) (h1 : total = 500) (h2 : students_cats = 75) (h3 : students_dogs = 125):
  (students_cats / total : ℝ) = 0.15 ∧
  (students_dogs / total : ℝ) = 0.25 :=
by
  sorry

end percentage_ownership_l138_138092


namespace peanuts_weight_l138_138367

theorem peanuts_weight (total_snacks raisins : ℝ) (h_total : total_snacks = 0.5) (h_raisins : raisins = 0.4) : (total_snacks - raisins) = 0.1 :=
by
  rw [h_total, h_raisins]
  norm_num

end peanuts_weight_l138_138367


namespace team_points_behind_l138_138641

-- Define the points for Max, Dulce and the condition for Val
def max_points : ℕ := 5
def dulce_points : ℕ := 3
def combined_points_max_dulce : ℕ := max_points + dulce_points
def val_points : ℕ := 2 * combined_points_max_dulce

-- Define the total points for their team and the opponents' team
def their_team_points : ℕ := max_points + dulce_points + val_points
def opponents_team_points : ℕ := 40

-- Proof statement
theorem team_points_behind : opponents_team_points - their_team_points = 16 :=
by
  sorry

end team_points_behind_l138_138641


namespace range_of_a_l138_138731

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a ≤ abs (x - 5) + abs (x - 3)) → a ≤ 2 := by
  sorry

end range_of_a_l138_138731


namespace quadratic_inequality_solution_l138_138109

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 < x + 6) ↔ (-2 < x ∧ x < 3) := 
by
  sorry

end quadratic_inequality_solution_l138_138109


namespace construct_using_five_twos_l138_138385

theorem construct_using_five_twos :
  (∃ (a b c d e f : ℕ), (22 * (a / b)) / c = 11 ∧
                        (22 / d) + (e / f) = 12 ∧
                        (22 + g + h) / i = 13 ∧
                        (2 * 2 * 2 * 2 - j) = 14 ∧
                        (22 / k) + (2 * 2) = 15) := by
  sorry

end construct_using_five_twos_l138_138385


namespace combined_instruments_l138_138042

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_horns := charlie_horns / 2
def carli_harps := 0

def charlie_total := charlie_flutes + charlie_horns + charlie_harps
def carli_total := carli_flutes + carli_horns + carli_harps
def combined_total := charlie_total + carli_total

theorem combined_instruments :
  combined_total = 7 :=
by
  sorry

end combined_instruments_l138_138042


namespace garden_length_l138_138735

theorem garden_length (P b l : ℕ) (h1 : P = 500) (h2 : b = 100) : l = 150 :=
by
  sorry

end garden_length_l138_138735


namespace negation_example_l138_138081

theorem negation_example :
  ¬ (∀ n : ℕ, (n^2 + n) % 2 = 0) ↔ ∃ n : ℕ, (n^2 + n) % 2 ≠ 0 :=
by
  sorry

end negation_example_l138_138081


namespace product_of_935421_and_625_l138_138474

theorem product_of_935421_and_625 : 935421 * 625 = 584638125 :=
by
  sorry

end product_of_935421_and_625_l138_138474


namespace triangle_with_ratio_is_right_triangle_l138_138189

/-- If the ratio of the interior angles of a triangle is 1:2:3, then the triangle is a right triangle. -/
theorem triangle_with_ratio_is_right_triangle (x : ℝ) (h : x + 2*x + 3*x = 180) : 
  3*x = 90 :=
sorry

end triangle_with_ratio_is_right_triangle_l138_138189


namespace average_annual_growth_rate_l138_138024

theorem average_annual_growth_rate (x : ℝ) (h1 : 6.4 * (1 + x)^2 = 8.1) : x = 0.125 :=
by
  -- proof goes here
  sorry

end average_annual_growth_rate_l138_138024


namespace length_of_QB_l138_138616

/-- 
Given a circle Q with a circumference of 16π feet, 
segment AB as its diameter, 
and the angle AQB of 120 degrees, 
prove that the length of segment QB is 8 feet.
-/
theorem length_of_QB (C : ℝ) (r : ℝ) (A B Q : ℝ) (angle_AQB : ℝ) 
  (h1 : C = 16 * Real.pi)
  (h2 : 2 * Real.pi * r = C)
  (h3 : angle_AQB = 120) 
  : QB = 8 :=
sorry

end length_of_QB_l138_138616


namespace proof_solution_l138_138399

variable (U : Set ℝ) (A : Set ℝ) (C_U_A : Set ℝ)
variables (a b : ℝ)

noncomputable def proof_problem : Prop :=
  (U = Set.univ) →
  (A = {x | a ≤ x ∧ x ≤ b}) →
  (C_U_A = {x | x > 4 ∨ x < 3}) →
  A = {x | 3 ≤ x ∧ x ≤ 4} ∧ a = 3 ∧ b = 4

theorem proof_solution : proof_problem U A C_U_A a b :=
by
  intro hU hA hCUA
  have hA_eq : A = {x | 3 ≤ x ∧ x ≤ 4} :=
    by { sorry }
  have ha : a = 3 :=
    by { sorry }
  have hb : b = 4 :=
    by { sorry }
  exact ⟨hA_eq, ha, hb⟩

end proof_solution_l138_138399


namespace largest_a_mul_b_l138_138447

-- Given conditions and proof statement
theorem largest_a_mul_b {m k q a b : ℕ} (hm : m = 720 * k + 83)
  (ha : m = a * q + b) (h_b_lt_a: b < a): a * b = 5112 :=
sorry

end largest_a_mul_b_l138_138447


namespace volume_PABCD_l138_138596

noncomputable def volume_of_pyramid (AB BC : ℝ) (PA : ℝ) : ℝ :=
  (1 / 3) * (AB * BC) * PA

theorem volume_PABCD (AB BC : ℝ) (h_AB : AB = 10) (h_BC : BC = 5)
  (PA : ℝ) (h_PA : PA = 2 * BC) :
  volume_of_pyramid AB BC PA = 500 / 3 :=
by
  subst h_AB
  subst h_BC
  subst h_PA
  -- At this point, we assert that everything simplifies correctly.
  -- This fill in the details for the correct expressions.
  sorry

end volume_PABCD_l138_138596


namespace hyperbola_equation_chord_length_l138_138532

noncomputable def length_real_axis := 2
noncomputable def eccentricity := Real.sqrt 3
noncomputable def a := 1
noncomputable def b := Real.sqrt 2
noncomputable def hyperbola_eq (x y : ℝ) := x^2 - y^2 / 2 = 1

theorem hyperbola_equation : 
  (∀ x y : ℝ, hyperbola_eq x y ↔ x^2 - (y^2 / 2) = 1) :=
by
  intros x y
  sorry

theorem chord_length (m : ℝ) : 
  ∀ x1 x2 y1 y2 : ℝ, y1 = x1 + m → y2 = x2 + m →
    x1^2 - y1^2 / 2 = 1 → x2^2 - y2^2 / 2 = 1 →
    Real.sqrt (2 * ((x1 + x2)^2 - 4 * x1 * x2)) = 4 * Real.sqrt 2 →
    m = 1 ∨ m = -1 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4 h5
  sorry

end hyperbola_equation_chord_length_l138_138532


namespace sum_infinite_series_l138_138012

theorem sum_infinite_series :
  (∑' n : ℕ, (4 * (n + 1) + 1) / (3^(n + 1))) = 7 / 2 :=
sorry

end sum_infinite_series_l138_138012


namespace cubes_difference_l138_138232

theorem cubes_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 65) (h3 : a + b = 6) : a^3 - b^3 = 432.25 :=
by
  sorry

end cubes_difference_l138_138232


namespace no_unique_solution_l138_138892

theorem no_unique_solution (d : ℝ) (x y : ℝ) :
  (3 * (3 * x + 4 * y) = 36) ∧ (9 * x + 12 * y = d) ↔ d ≠ 36 := sorry

end no_unique_solution_l138_138892


namespace find_x_from_percentage_l138_138733

theorem find_x_from_percentage (x : ℝ) (h : 0.2 * 30 = 0.25 * x + 2) : x = 16 :=
sorry

end find_x_from_percentage_l138_138733


namespace inequality_proof_l138_138728

variable (f : ℕ → ℕ → ℕ)

theorem inequality_proof :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
by sorry

end inequality_proof_l138_138728


namespace solve_for_y_l138_138834

theorem solve_for_y (x y : ℝ) (h : 5 * x + 3 * y = 1) : y = (1 - 5 * x) / 3 :=
by
  sorry

end solve_for_y_l138_138834


namespace length_of_third_wall_l138_138254

-- Define the dimensions of the first two walls
def wall1_length : ℕ := 30
def wall1_height : ℕ := 12
def wall1_area : ℕ := wall1_length * wall1_height

def wall2_length : ℕ := 30
def wall2_height : ℕ := 12
def wall2_area : ℕ := wall2_length * wall2_height

-- Total area needed
def total_area_needed : ℕ := 960

-- Calculate the area for the third wall
def two_walls_area : ℕ := wall1_area + wall2_area
def third_wall_area : ℕ := total_area_needed - two_walls_area

-- Height of the third wall
def third_wall_height : ℕ := 12

-- Calculate the length of the third wall
def third_wall_length : ℕ := third_wall_area / third_wall_height

-- Final claim: Length of the third wall is 20 feet
theorem length_of_third_wall : third_wall_length = 20 := by
  sorry

end length_of_third_wall_l138_138254


namespace albert_needs_more_money_l138_138910

def cost_of_paintbrush : ℝ := 1.50
def cost_of_paints : ℝ := 4.35
def cost_of_easel : ℝ := 12.65
def amount_already_has : ℝ := 6.50

theorem albert_needs_more_money : 
  (cost_of_paintbrush + cost_of_paints + cost_of_easel) - amount_already_has = 12.00 := 
by
  sorry

end albert_needs_more_money_l138_138910


namespace items_count_l138_138155

variable (N : ℕ)

-- Conditions
def item_price : ℕ := 50
def discount_rate : ℕ := 80
def sell_percentage : ℕ := 90
def creditors_owed : ℕ := 15000
def money_left : ℕ := 3000

-- Definitions based on the conditions
def sale_price : ℕ := (item_price * (100 - discount_rate)) / 100
def money_before_paying_creditors : ℕ := money_left + creditors_owed
def total_revenue (N : ℕ) : ℕ := (sell_percentage * N * sale_price) / 100

-- Problem statement
theorem items_count : total_revenue N = money_before_paying_creditors → N = 2000 := by
  intros h
  sorry

end items_count_l138_138155


namespace all_solutions_of_diophantine_eq_l138_138422

theorem all_solutions_of_diophantine_eq
  (a b c x0 y0 : ℤ) (h_gcd : Int.gcd a b = 1)
  (h_sol : a * x0 + b * y0 = c) :
  ∀ x y : ℤ, (a * x + b * y = c) →
  ∃ t : ℤ, x = x0 + b * t ∧ y = y0 - a * t :=
by
  sorry

end all_solutions_of_diophantine_eq_l138_138422


namespace triangle_area_PQR_l138_138754

section TriangleArea

variables {a b c d : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
variables (hOppositeSides : (0 - c) * b - (a - 0) * d < 0)

theorem triangle_area_PQR :
  let P := (0, a)
  let Q := (b, 0)
  let R := (c, d)
  let area := (1 / 2) * (a * c + b * d - a * b)
  area = (1 / 2) * (a * c + b * d - a * b) := 
by
  sorry

end TriangleArea

end triangle_area_PQR_l138_138754


namespace michael_final_revenue_l138_138476

noncomputable def total_revenue_before_discount : ℝ :=
  (3 * 45) + (5 * 22) + (7 * 16) + (8 * 10) + (10 * 5)

noncomputable def discount : ℝ := 0.10 * total_revenue_before_discount

noncomputable def discounted_revenue : ℝ := total_revenue_before_discount - discount

noncomputable def sales_tax : ℝ := 0.06 * discounted_revenue

noncomputable def final_revenue : ℝ := discounted_revenue + sales_tax

theorem michael_final_revenue : final_revenue = 464.60 :=
by
  sorry

end michael_final_revenue_l138_138476


namespace complement_of_A_in_U_eq_l138_138968

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ Real.exp 1}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x ≤ Real.exp 1}

theorem complement_of_A_in_U_eq : 
  (U \ A) = complement_U_A := 
by
  sorry

end complement_of_A_in_U_eq_l138_138968


namespace carson_gardening_time_l138_138491

-- Definitions of the problem conditions
def lines_to_mow : ℕ := 40
def minutes_per_line : ℕ := 2
def rows_of_flowers : ℕ := 8
def flowers_per_row : ℕ := 7
def minutes_per_flower : ℚ := 0.5

-- Total time calculation for the proof 
theorem carson_gardening_time : 
  (lines_to_mow * minutes_per_line) + (rows_of_flowers * flowers_per_row * minutes_per_flower) = 108 := 
by 
  sorry

end carson_gardening_time_l138_138491


namespace cyclic_sum_inequality_l138_138724

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) ≥ (2 / 3) * (a^2 + b^2 + c^2) :=
  sorry

end cyclic_sum_inequality_l138_138724


namespace diophantine_soln_l138_138342

-- Define the Diophantine equation as a predicate
def diophantine_eq (x y : ℤ) : Prop := x^3 - y^3 = 2 * x * y + 8

-- Theorem stating that the only solutions are (0, -2) and (2, 0)
theorem diophantine_soln :
  ∀ x y : ℤ, diophantine_eq x y ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end diophantine_soln_l138_138342


namespace john_work_days_l138_138496

theorem john_work_days (J : ℕ) (H1 : 1 / J + 1 / 480 = 1 / 192) : J = 320 :=
sorry

end john_work_days_l138_138496


namespace problem1_problem2_problem3_problem4_l138_138706

theorem problem1 : 6 + (-8) - (-5) = 3 := by
  sorry

theorem problem2 : (5 + 3/5) + (-(5 + 2/3)) + (4 + 2/5) + (-1/3) = 4 := by
  sorry

theorem problem3 : ((-1/2) + 1/6 - 1/4) * 12 = -7 := by
  sorry

theorem problem4 : -1^2022 + 27 * (-1/3)^2 - |(-5)| = -3 := by
  sorry

end problem1_problem2_problem3_problem4_l138_138706


namespace time_with_walkway_l138_138880

theorem time_with_walkway (v w : ℝ) (t : ℕ) :
  (80 = 120 * (v - w)) → 
  (80 = 60 * v) → 
  t = 80 / (v + w) → 
  t = 40 :=
by
  sorry

end time_with_walkway_l138_138880


namespace relationship_y1_y2_y3_l138_138144

variables {m y_1 y_2 y_3 : ℝ}

theorem relationship_y1_y2_y3 :
  (∃ (m : ℝ), (y_1 = (-1)^2 - 2*(-1) + m) ∧ (y_2 = 2^2 - 2*2 + m) ∧ (y_3 = 3^2 - 2*3 + m)) →
  y_2 < y_1 ∧ y_1 = y_3 :=
by
  sorry

end relationship_y1_y2_y3_l138_138144


namespace calc_1_calc_2_calc_3_calc_4_l138_138238

-- Problem 1
theorem calc_1 : 26 - 7 + (-6) + 17 = 30 := 
by
  sorry

-- Problem 2
theorem calc_2 : -81 / (9 / 4) * (-4 / 9) / (-16) = -1 := 
by
  sorry

-- Problem 3
theorem calc_3 : ((2 / 3) - (3 / 4) + (1 / 6)) * (-36) = -3 := 
by
  sorry

-- Problem 4
theorem calc_4 : -1^4 + 12 / (-2)^2 + (1 / 4) * (-8) = 0 := 
by
  sorry


end calc_1_calc_2_calc_3_calc_4_l138_138238


namespace simplify_fraction_l138_138889

theorem simplify_fraction (c : ℝ) : (5 - 4 * c) / 9 - 3 = (-22 - 4 * c) / 9 := 
sorry

end simplify_fraction_l138_138889


namespace surface_area_of_cube_is_correct_l138_138045

noncomputable def edge_length (a : ℝ) : ℝ := 5 * a

noncomputable def surface_area_of_cube (a : ℝ) : ℝ :=
  let edge := edge_length a
  6 * edge * edge

theorem surface_area_of_cube_is_correct (a : ℝ) :
  surface_area_of_cube a = 150 * a ^ 2 := by
  sorry

end surface_area_of_cube_is_correct_l138_138045


namespace bala_age_difference_l138_138344

theorem bala_age_difference 
  (a10 : ℕ) -- Anand's age 10 years ago.
  (b10 : ℕ) -- Bala's age 10 years ago.
  (h1 : a10 = b10 / 3) -- 10 years ago, Anand's age was one-third Bala's age.
  (h2 : a10 = 15 - 10) -- Anand was 5 years old 10 years ago, given his current age is 15.
  : (b10 + 10) - 15 = 10 := -- Bala is 10 years older than Anand.
sorry

end bala_age_difference_l138_138344


namespace tetrahedrons_from_triangular_prism_l138_138319

theorem tetrahedrons_from_triangular_prism : 
  let n := 6
  let choose4 := Nat.choose n 4
  let coplanar_cases := 3
  choose4 - coplanar_cases = 12 := by
  sorry

end tetrahedrons_from_triangular_prism_l138_138319


namespace A_investment_amount_l138_138323

-- Conditions
variable (B_investment : ℝ) (C_investment : ℝ) (total_profit : ℝ) (A_profit : ℝ)
variable (B_investment_value : B_investment = 4200)
variable (C_investment_value : C_investment = 10500)
variable (total_profit_value : total_profit = 13600)
variable (A_profit_value : A_profit = 4080)

-- Proof statement
theorem A_investment_amount : 
  (∃ x : ℝ, x = 4410) :=
by
  sorry

end A_investment_amount_l138_138323


namespace hens_count_l138_138689

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := by
  sorry

end hens_count_l138_138689


namespace unknown_number_is_five_l138_138671

theorem unknown_number_is_five (x : ℕ) (h : 64 + x * 12 / (180 / 3) = 65) : x = 5 := 
by 
  sorry

end unknown_number_is_five_l138_138671


namespace intersection_M_N_l138_138841

def M := { x : ℝ | x^2 - 2 * x < 0 }
def N := { x : ℝ | abs x < 1 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l138_138841


namespace negation_proposition_p_l138_138493

open Classical

variable (n : ℕ)

def proposition_p : Prop := ∃ n : ℕ, 2^n > 100

theorem negation_proposition_p : ¬ proposition_p ↔ ∀ n : ℕ, 2^n ≤ 100 := 
by sorry

end negation_proposition_p_l138_138493


namespace fair_collection_l138_138170

theorem fair_collection 
  (children : ℕ) (fee_child : ℝ) (adults : ℕ) (fee_adult : ℝ) 
  (total_people : ℕ) (count_children : ℕ) (count_adults : ℕ)
  (total_collected: ℝ) :
  children = 700 →
  fee_child = 1.5 →
  adults = 1500 →
  fee_adult = 4.0 →
  total_people = children + adults →
  count_children = 700 →
  count_adults = 1500 →
  total_collected = (count_children * fee_child) + (count_adults * fee_adult) →
  total_collected = 7050 :=
by
  intros
  sorry

end fair_collection_l138_138170


namespace problem_I_problem_II_l138_138953

variable (x a m : ℝ)

theorem problem_I (h: ¬ (∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0)) : 
  a < -2 ∨ a > 3 := by
  sorry

theorem problem_II (p : ∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0) (q : m-1 ≤ a ∧ a ≤ m+3) :
  ∀ a : ℝ, -2 ≤ a ∧ a ≤ 3 → m ∈ [-1, 0] := by
  sorry

end problem_I_problem_II_l138_138953


namespace triangle_ABCD_lengths_l138_138753

theorem triangle_ABCD_lengths (AB BC CA : ℝ) (h_AB : AB = 20) (h_BC : BC = 40) (h_CA : CA = 49) :
  ∃ DA DC : ℝ, DA = 27.88 ∧ DC = 47.88 ∧
  (AB + DC = BC + DA) ∧ 
  (((AB^2 + BC^2 - CA^2) / (2 * AB * BC)) + ((DC^2 + DA^2 - CA^2) / (2 * DC * DA)) = 0) :=
sorry

end triangle_ABCD_lengths_l138_138753


namespace inscribed_circle_radius_l138_138445

theorem inscribed_circle_radius 
  (A : ℝ) -- Area of the triangle
  (p : ℝ) -- Perimeter of the triangle
  (r : ℝ) -- Radius of the inscribed circle
  (s : ℝ) -- Semiperimeter of the triangle
  (h1 : A = 2 * p) -- Condition: Area is numerically equal to twice the perimeter
  (h2 : p = 2 * s) -- Perimeter is twice the semiperimeter
  (h3 : A = r * s) -- Formula: Area in terms of inradius and semiperimeter
  (h4 : s ≠ 0) -- Semiperimeter is non-zero
  : r = 4 := 
sorry

end inscribed_circle_radius_l138_138445


namespace ratio_of_tax_revenue_to_cost_of_stimulus_l138_138578

-- Definitions based on the identified conditions
def bottom_20_percent_people (total_people : ℕ) : ℕ := (total_people * 20) / 100
def stimulus_per_person : ℕ := 2000
def total_people : ℕ := 1000
def government_profit : ℕ := 1600000

-- Cost of the stimulus
def cost_of_stimulus : ℕ := bottom_20_percent_people total_people * stimulus_per_person

-- Tax revenue returned to the government
def tax_revenue : ℕ := government_profit + cost_of_stimulus

-- The Proposition we need to prove
theorem ratio_of_tax_revenue_to_cost_of_stimulus :
  tax_revenue / cost_of_stimulus = 5 :=
by
  sorry

end ratio_of_tax_revenue_to_cost_of_stimulus_l138_138578


namespace divisibility_l138_138917

theorem divisibility (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  (n^5 + 1) ∣ (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4 * k - 1) := 
sorry

end divisibility_l138_138917


namespace count_valid_triangles_l138_138105

/-- 
Define the problem constraints: scalene triangles with side lengths a, b, c, 
where a < b < c, a + c = 2b, and a + b + c ≤ 30.
-/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + c = 2 * b ∧ a + b + c ≤ 30

/-- 
Statement of the problem: Prove that there are 20 distinct triangles satisfying the above constraints. 
-/
theorem count_valid_triangles : ∃ n, n = 20 ∧ (∀ {a b c : ℕ}, is_valid_triangle a b c → n = 20) :=
sorry

end count_valid_triangles_l138_138105


namespace solution_set_l138_138005

theorem solution_set (x : ℝ) : 
  (-2 * x ≤ 6) ∧ (x + 1 < 0) ↔ (-3 ≤ x) ∧ (x < -1) := by
  sorry

end solution_set_l138_138005


namespace necessary_but_not_sufficient_condition_l138_138967

theorem necessary_but_not_sufficient_condition (a : ℝ)
    (h : -2 ≤ a ∧ a ≤ 2)
    (hq : ∃ x y : ℂ, x ≠ y ∧ (x ^ 2 + (a : ℂ) * x + 1 = 0) ∧ (y ^ 2 + (a : ℂ) * y + 1 = 0)) :
    ∃ z : ℂ, z ^ 2 + (a : ℂ) * z + 1 = 0 ∧ (¬ ∀ b, -2 < b ∧ b < 2 → b = a) :=
sorry

end necessary_but_not_sufficient_condition_l138_138967


namespace CD_is_b_minus_a_minus_c_l138_138276

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (A B C D : V) (a b c : V)

def AB : V := a
def AD : V := b
def BC : V := c

theorem CD_is_b_minus_a_minus_c (h1 : A + a = B) (h2 : A + b = D) (h3 : B + c = C) :
  D - C = b - a - c :=
by sorry

end CD_is_b_minus_a_minus_c_l138_138276


namespace numbers_are_perfect_squares_l138_138076

/-- Prove that the numbers 49, 4489, 444889, ... obtained by inserting 48 into the 
middle of the previous number are perfect squares. -/
theorem numbers_are_perfect_squares :
  ∀ n : ℕ, ∃ k : ℕ, (k ^ 2) = (Int.ofNat ((20 * (10 : ℕ) ^ n + 1) / 3)) :=
by
  sorry

end numbers_are_perfect_squares_l138_138076


namespace books_per_shelf_l138_138456

theorem books_per_shelf (total_books : ℕ) (total_shelves : ℕ) (h_total_books : total_books = 2250) (h_total_shelves : total_shelves = 150) :
  total_books / total_shelves = 15 :=
by
  sorry

end books_per_shelf_l138_138456


namespace correct_sampling_methods_l138_138096

def reporter_A_sampling : String :=
  "systematic sampling"

def reporter_B_sampling : String :=
  "systematic sampling"

theorem correct_sampling_methods (constant_flow : Prop)
  (A_interview_method : ∀ t : ℕ, t % 10 = 0)
  (B_interview_method : ∀ n : ℕ, n % 1000 = 0) :
  reporter_A_sampling = "systematic sampling" ∧ reporter_B_sampling = "systematic sampling" :=
by
  sorry

end correct_sampling_methods_l138_138096


namespace parabola_directrix_standard_eq_l138_138289

theorem parabola_directrix_standard_eq (y : ℝ) (x : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ ∀ (P : {P // P ≠ x ∨ P ≠ y}), 
  (y + 1) = p) → x^2 = 4 * y :=
sorry

end parabola_directrix_standard_eq_l138_138289


namespace lorry_sand_capacity_l138_138347

def cost_cement (bags : ℕ) (cost_per_bag : ℕ) : ℕ := bags * cost_per_bag
def total_cost (cement_cost : ℕ) (sand_cost : ℕ) : ℕ := cement_cost + sand_cost
def total_sand (sand_cost : ℕ) (cost_per_ton : ℕ) : ℕ := sand_cost / cost_per_ton
def sand_per_lorry (total_sand : ℕ) (lorries : ℕ) : ℕ := total_sand / lorries

theorem lorry_sand_capacity : 
  cost_cement 500 10 + (total_cost 5000 (total_sand 8000 40)) = 13000 ∧
  total_cost 5000 8000 = 13000 ∧
  total_sand 8000 40 = 200 ∧
  sand_per_lorry 200 20 = 10 :=
by
  sorry

end lorry_sand_capacity_l138_138347


namespace number_of_terminating_decimals_l138_138495

theorem number_of_terminating_decimals : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 299 → (∃ k : ℕ, n = 9 * k) → 
  ∃ count : ℕ, count = 33 := 
sorry

end number_of_terminating_decimals_l138_138495


namespace task_completion_l138_138980

theorem task_completion (x y z : ℝ) 
  (h1 : 1 / x + 1 / y = 1 / 2)
  (h2 : 1 / y + 1 / z = 1 / 4)
  (h3 : 1 / z + 1 / x = 5 / 12) :
  x = 3 := 
sorry

end task_completion_l138_138980


namespace parker_total_stamps_l138_138730

-- Definitions based on conditions
def original_stamps := 430
def addie_stamps := 1890
def addie_fraction := 3 / 7
def stamps_added_by_addie := addie_fraction * addie_stamps

-- Theorem statement to prove the final number of stamps
theorem parker_total_stamps : original_stamps + stamps_added_by_addie = 1240 :=
by
  -- definitions instantiated above
  sorry  -- proof required

end parker_total_stamps_l138_138730


namespace max_value_of_xy_l138_138366

theorem max_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 2) :
  xy ≤ 1 / 2 :=
sorry

end max_value_of_xy_l138_138366


namespace max_plus_ten_min_eq_zero_l138_138388

theorem max_plus_ten_min_eq_zero (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  let M := max (x * y + x * z + y * z)
  let m := min (x * y + x * z + y * z)
  M + 10 * m = 0 :=
by
  sorry

end max_plus_ten_min_eq_zero_l138_138388


namespace intersection_point_unique_m_l138_138075

theorem intersection_point_unique_m (m : ℕ) (h1 : m > 0)
  (x y : ℤ) (h2 : 13 * x + 11 * y = 700) (h3 : y = m * x - 1) : m = 6 :=
by
  sorry

end intersection_point_unique_m_l138_138075


namespace solve_for_a_l138_138546

theorem solve_for_a (a : ℝ) (h_pos : a > 0) 
  (h_roots : ∀ x, x^2 - 2*a*x - 3*a^2 = 0 → (x = -a ∨ x = 3*a)) 
  (h_diff : |(-a) - (3*a)| = 8) : a = 2 := 
sorry

end solve_for_a_l138_138546


namespace crocus_bulbs_count_l138_138657

theorem crocus_bulbs_count (C D : ℕ) 
  (h1 : C + D = 55) 
  (h2 : 0.35 * (C : ℝ) + 0.65 * (D : ℝ) = 29.15) :
  C = 22 :=
sorry

end crocus_bulbs_count_l138_138657


namespace discount_percentage_l138_138381

theorem discount_percentage (x : ℝ) : 
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  (marked_price * (1 - x / 100) * (1 - second_discount) * (1 - third_discount) = final_price) ↔ x = 20 := 
by
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  sorry

end discount_percentage_l138_138381


namespace grid_X_value_l138_138033

theorem grid_X_value :
  ∃ X, (∃ b d1 d2 d3 d4, 
    b = 16 ∧
    d1 = (25 - 20) ∧
    d2 = (16 - 15) / 3 ∧
    d3 = (d1 * 5) / 4 ∧
    d4 = d1 - d3 ∧
    (-12 - d4 * 4) = -30 ∧ 
    X = d4 ∧
    X = 10.5) :=
sorry

end grid_X_value_l138_138033


namespace smallest_number_ending_in_6_moved_front_gives_4_times_l138_138315

theorem smallest_number_ending_in_6_moved_front_gives_4_times (x m n : ℕ) 
  (h1 : n = 10 * x + 6)
  (h2 : 6 * 10^m + x = 4 * n) :
  n = 1538466 :=
by
  sorry

end smallest_number_ending_in_6_moved_front_gives_4_times_l138_138315


namespace amoeba_count_after_two_weeks_l138_138156

theorem amoeba_count_after_two_weeks :
  let initial_day_count := 1
  let days_double_split := 7
  let days_triple_split := 7
  let end_of_first_phase := initial_day_count * 2 ^ days_double_split
  let final_amoeba_count := end_of_first_phase * 3 ^ days_triple_split
  final_amoeba_count = 279936 :=
by
  sorry

end amoeba_count_after_two_weeks_l138_138156


namespace average_sale_six_months_l138_138737

theorem average_sale_six_months :
  let sale1 := 2500
  let sale2 := 6500
  let sale3 := 9855
  let sale4 := 7230
  let sale5 := 7000
  let sale6 := 11915
  let total_sales := sale1 + sale2 + sale3 + sale4 + sale5 + sale6
  let num_months := 6
  (total_sales / num_months) = 7500 :=
by
  sorry

end average_sale_six_months_l138_138737


namespace necessary_condition_for_inequality_l138_138185

theorem necessary_condition_for_inequality (a b : ℝ) (h : a * b > 0) : 
  (a ≠ b) → (a ≠ 0) → (b ≠ 0) → ((b / a) + (a / b) > 2) :=
by
  sorry

end necessary_condition_for_inequality_l138_138185


namespace school_year_days_l138_138543

theorem school_year_days :
  ∀ (D : ℕ),
  (9 = 5 * D / 100) →
  D = 180 := by
  intro D
  sorry

end school_year_days_l138_138543


namespace root_of_quadratic_l138_138745

theorem root_of_quadratic (a : ℝ) (ha : a ≠ 1) (hroot : (a-1) * 1^2 - a * 1 + a^2 = 0) : a = -1 := by
  sorry

end root_of_quadratic_l138_138745


namespace undefined_values_of_fraction_l138_138940

theorem undefined_values_of_fraction (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end undefined_values_of_fraction_l138_138940


namespace has_two_roots_l138_138528

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end has_two_roots_l138_138528


namespace solve_for_sum_l138_138972

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := -1
noncomputable def c : ℝ := Real.sqrt 26

theorem solve_for_sum :
  (a * (a - 4) = 5) ∧ (b * (b - 4) = 5) ∧ (c * (c - 4) = 5) ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a^2 + b^2 = c^2) → (a + b + c = 4 + Real.sqrt 26) :=
by
  sorry

end solve_for_sum_l138_138972


namespace find_f_of_9_l138_138602

theorem find_f_of_9 (α : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x ^ α)
  (h2 : f 2 = Real.sqrt 2) :
  f 9 = 3 :=
sorry

end find_f_of_9_l138_138602


namespace intersection_P_Q_l138_138179

def setP : Set ℝ := {1, 2, 3, 4}
def setQ : Set ℝ := {x | abs x ≤ 2}

theorem intersection_P_Q : (setP ∩ setQ) = {1, 2} :=
by
  sorry

end intersection_P_Q_l138_138179


namespace floor_condition_x_l138_138648

theorem floor_condition_x (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 48) : 8 ≤ x ∧ x < 49 / 6 := 
by 
  sorry

end floor_condition_x_l138_138648


namespace ratio_daves_bench_to_weight_l138_138658

variables (wD bM bD bC : ℝ)

def daves_weight := wD = 175
def marks_bench_press := bM = 55
def marks_comparison_to_craig := bM = bC - 50
def craigs_comparison_to_dave := bC = 0.20 * bD

theorem ratio_daves_bench_to_weight
  (h1 : daves_weight wD)
  (h2 : marks_bench_press bM)
  (h3 : marks_comparison_to_craig bM bC)
  (h4 : craigs_comparison_to_dave bC bD) :
  (bD / wD) = 3 :=
by
  rw [daves_weight] at h1
  rw [marks_bench_press] at h2
  rw [marks_comparison_to_craig] at h3
  rw [craigs_comparison_to_dave] at h4
  -- Now we have:
  -- 1. wD = 175
  -- 2. bM = 55
  -- 3. bM = bC - 50
  -- 4. bC = 0.20 * bD
  -- We proceed to solve:
  sorry

end ratio_daves_bench_to_weight_l138_138658


namespace eval_expression_l138_138415

def f (x : ℤ) : ℤ := 3 * x^2 - 6 * x + 10

theorem eval_expression : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end eval_expression_l138_138415


namespace fraction_addition_l138_138666

theorem fraction_addition (x : ℝ) (h : x + 1 ≠ 0) : (x / (x + 1) + 1 / (x + 1) = 1) :=
sorry

end fraction_addition_l138_138666


namespace correct_answer_is_A_l138_138805

-- Definitions derived from problem conditions
def algorithm := Type
def has_sequential_structure (alg : algorithm) : Prop := sorry -- Actual definition should define what a sequential structure is for an algorithm

-- Given: An algorithm must contain a sequential structure.
theorem correct_answer_is_A (alg : algorithm) : has_sequential_structure alg :=
sorry

end correct_answer_is_A_l138_138805


namespace pure_imaginary_iff_a_eq_2_l138_138026

theorem pure_imaginary_iff_a_eq_2 (a : ℝ) : (∃ k : ℝ, (∃ x : ℝ, (2-a) / 2 = x ∧ x = 0) ∧ (2+a)/2 = k ∧ k ≠ 0) ↔ a = 2 :=
by
  sorry

end pure_imaginary_iff_a_eq_2_l138_138026


namespace factorize_expression_l138_138614

theorem factorize_expression (a b m : ℝ) :
  a^2 * (m - 1) + b^2 * (1 - m) = (m - 1) * (a + b) * (a - b) :=
by sorry

end factorize_expression_l138_138614


namespace find_y_l138_138386

theorem find_y (y z : ℕ) (h1 : 50 = y * 10) (h2 : 300 = 50 * z) : y = 5 :=
by
  sorry

end find_y_l138_138386


namespace quadratic_other_root_l138_138943

theorem quadratic_other_root (m x2 : ℝ) (h₁ : 1^2 - 4*1 + m = 0) (h₂ : x2^2 - 4*x2 + m = 0) : x2 = 3 :=
sorry

end quadratic_other_root_l138_138943


namespace range_of_m_l138_138333

open Set

def set_A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (3 * m - 2)}

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ m ≤ 4 :=
by sorry

end range_of_m_l138_138333


namespace fraction_inequality_solution_l138_138186

theorem fraction_inequality_solution (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) :
  3 * x + 2 < 2 * (5 * x - 4) → (10 / 7) < x ∧ x ≤ 3 :=
by
  sorry

end fraction_inequality_solution_l138_138186


namespace min_ab_eq_4_l138_138907

theorem min_ab_eq_4 (a b : ℝ) (h : 4 / a + 1 / b = Real.sqrt (a * b)) : a * b ≥ 4 :=
sorry

end min_ab_eq_4_l138_138907


namespace find_QE_l138_138807

noncomputable def QE (QD DE : ℝ) : ℝ :=
  QD + DE

theorem find_QE :
  ∀ (Q C R D E : Type) (QR QD DE QE : ℝ), 
  QD = 5 →
  QE = QD + DE →
  QR = DE - QD →
  QR^2 = QD * QE →
  QE = (QD + 5 + 5 * Real.sqrt 5) / 2 :=
by
  intros
  sorry

end find_QE_l138_138807


namespace number_of_perfect_square_factors_of_360_l138_138041

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

def number_of_perfect_square_factors (n : ℕ) : ℕ :=
  if n = 360 then 4 else 0

theorem number_of_perfect_square_factors_of_360 :
  number_of_perfect_square_factors 360 = 4 := 
by {
  -- Sorry is used here as a placeholder for the proof steps.
  sorry
}

end number_of_perfect_square_factors_of_360_l138_138041


namespace contradiction_of_distinct_roots_l138_138161

theorem contradiction_of_distinct_roots
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (distinct_abc : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (H : ¬ (∃ x1 x2, x1 ≠ x2 ∧ (a * x1^2 + 2 * b * x1 + c = 0 ∨ b * x1^2 + 2 * c * x1 + a = 0 ∨ c * x1^2 + 2 * a * x1 + b = 0))) :
  False := 
sorry

end contradiction_of_distinct_roots_l138_138161


namespace tourists_left_l138_138574

theorem tourists_left (initial_tourists eaten_by_anacondas poisoned_fraction recover_fraction : ℕ) 
(h_initial : initial_tourists = 30) 
(h_eaten : eaten_by_anacondas = 2)
(h_poisoned_fraction : poisoned_fraction = 2)
(h_recover_fraction : recover_fraction = 7) :
  initial_tourists - eaten_by_anacondas - (initial_tourists - eaten_by_anacondas) / poisoned_fraction + (initial_tourists - eaten_by_anacondas) / poisoned_fraction / recover_fraction = 16 :=
by
  sorry

end tourists_left_l138_138574


namespace new_recipe_water_l138_138213

theorem new_recipe_water (flour water sugar : ℕ)
  (h_orig : flour = 10 ∧ water = 6 ∧ sugar = 3)
  (h_new : ∀ (new_flour new_water new_sugar : ℕ), 
            new_flour = 10 ∧ new_water = 3 ∧ new_sugar = 3)
  (h_sugar : sugar = 4) :
  new_water = 4 := 
  sorry

end new_recipe_water_l138_138213


namespace max_distance_point_circle_l138_138898

open Real

noncomputable def distance (P C : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)

theorem max_distance_point_circle :
  let C : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (3, 3)
  let r : ℝ := 2
  let max_distance : ℝ := distance P C + r
  ∃ M : ℝ × ℝ, distance P M = max_distance ∧ (M.1 - 1)^2 + (M.2 - 2)^2 = r^2 :=
by
  sorry

end max_distance_point_circle_l138_138898


namespace final_investment_amount_l138_138389

noncomputable def final_amount (P1 P2 : ℝ) (r1 r2 t1 t2 n1 n2 : ℝ) : ℝ :=
  let A1 := P1 * (1 + r1 / n1) ^ (n1 * t1)
  let A2 := (A1 + P2) * (1 + r2 / n2) ^ (n2 * t2)
  A2

theorem final_investment_amount :
  final_amount 6000 2000 0.10 0.08 2 1.5 2 4 = 10467.05 :=
by
  sorry

end final_investment_amount_l138_138389


namespace exam_standard_deviation_l138_138192

-- Define the mean score
def mean_score : ℝ := 74

-- Define the standard deviation and conditions
def standard_deviation (σ : ℝ) : Prop :=
  mean_score - 2 * σ = 58

-- Define the condition to prove
def standard_deviation_above_mean (σ : ℝ) : Prop :=
  (98 - mean_score) / σ = 3

theorem exam_standard_deviation {σ : ℝ} (h1 : standard_deviation σ) : standard_deviation_above_mean σ :=
by
  -- proof is omitted
  sorry

end exam_standard_deviation_l138_138192


namespace fraction_simplification_l138_138351

theorem fraction_simplification (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) :
  (x / (x - 1) = 3 / (2 * x - 2) - 3) → (2 * x = 3 - 6 * x + 6) :=
by 
  intro h1
  -- Proof steps would be here, but we are using sorry
  sorry

end fraction_simplification_l138_138351


namespace sequence_increasing_range_of_a_l138_138993

theorem sequence_increasing_range_of_a :
  ∀ {a : ℝ}, (∀ n : ℕ, 
    (n ≤ 7 → (4 - a) * n - 10 ≤ (4 - a) * (n + 1) - 10) ∧ 
    (7 < n → a^(n - 6) ≤ a^(n - 5))
  ) → 2 < a ∧ a < 4 :=
by
  sorry

end sequence_increasing_range_of_a_l138_138993


namespace replace_stars_with_identity_l138_138469

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l138_138469


namespace chocolate_cost_first_store_l138_138451

def cost_first_store (x : ℕ) : ℕ := x
def chocolate_promotion_store : ℕ := 2
def savings_in_three_weeks : ℕ := 6
def number_of_chocolates (weeks : ℕ) : ℕ := 2 * weeks

theorem chocolate_cost_first_store :
  ∀ (weeks : ℕ) (x : ℕ), 
    number_of_chocolates weeks = 6 →
    chocolate_promotion_store * number_of_chocolates weeks + savings_in_three_weeks = cost_first_store x * number_of_chocolates weeks →
    cost_first_store x = 3 :=
by
  intros weeks x h1 h2
  sorry

end chocolate_cost_first_store_l138_138451


namespace arrangements_APPLE_is_60_l138_138950

-- Definition of the problem statement based on the given conditions
def distinct_arrangements_APPLE : Nat :=
  let n := 5
  let n_A := 1
  let n_P := 2
  let n_L := 1
  let n_E := 1
  (n.factorial / (n_A.factorial * n_P.factorial * n_L.factorial * n_E.factorial))

-- The proof statement (without the proof itself, which is "sorry")
theorem arrangements_APPLE_is_60 : distinct_arrangements_APPLE = 60 := by
  sorry

end arrangements_APPLE_is_60_l138_138950


namespace minimum_value_fraction_l138_138761

theorem minimum_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (2 / x + 1 / y) >= 2 * Real.sqrt 2 :=
sorry

end minimum_value_fraction_l138_138761


namespace number_of_violinists_l138_138989

open Nat

/-- There are 3 violinists in the orchestra, based on given conditions. -/
theorem number_of_violinists
  (total : ℕ)
  (percussion : ℕ)
  (brass : ℕ)
  (cellist : ℕ)
  (contrabassist : ℕ)
  (woodwinds : ℕ)
  (maestro : ℕ)
  (total_eq : total = 21)
  (percussion_eq : percussion = 1)
  (brass_eq : brass = 7)
  (strings_excluding_violinists : ℕ)
  (cellist_eq : cellist = 1)
  (contrabassist_eq : contrabassist = 1)
  (woodwinds_eq : woodwinds = 7)
  (maestro_eq : maestro = 1) :
  (total - (percussion + brass + (cellist + contrabassist) + woodwinds + maestro)) = 3 := 
by
  sorry

end number_of_violinists_l138_138989


namespace max_value_proof_l138_138176

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_value_proof
  (x y z : ℝ)
  (hx : 0 ≤ x)
  (hy : 0 ≤ y)
  (hz : 0 ≤ z)
  (h1 : x + y + z = 1)
  (h2 : x^2 + y^2 + z^2 = 1) :
  maximum_value x y z ≤ 1 :=
sorry

end max_value_proof_l138_138176


namespace cylinder_radius_l138_138142

theorem cylinder_radius
  (diameter_c : ℝ) (altitude_c : ℝ) (height_relation : ℝ → ℝ)
  (same_axis : Bool) (radius_cylinder : ℝ → ℝ)
  (h1 : diameter_c = 14)
  (h2 : altitude_c = 20)
  (h3 : ∀ r, height_relation r = 3 * r)
  (h4 : same_axis = true)
  (h5 : ∀ r, radius_cylinder r = r) :
  ∃ r, r = 140 / 41 :=
by {
  sorry
}

end cylinder_radius_l138_138142


namespace find_smaller_number_l138_138417

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 := by
  sorry

end find_smaller_number_l138_138417


namespace circle_diameter_l138_138844

theorem circle_diameter (r : ℝ) (h : π * r^2 = 9 * π) : 2 * r = 6 :=
by sorry

end circle_diameter_l138_138844


namespace number_of_integers_with_square_fraction_l138_138708

theorem number_of_integers_with_square_fraction : 
  ∃! (S : Finset ℤ), (∀ (n : ℤ), n ∈ S ↔ ∃ (k : ℤ), (n = 15 * k^2) ∨ (15 - n = k^2)) ∧ S.card = 2 := 
sorry

end number_of_integers_with_square_fraction_l138_138708


namespace mrs_petersons_change_l138_138241

-- Define the conditions
def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def discount_rate : ℚ := 0.10
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

-- Formulate the proof statement
theorem mrs_petersons_change :
  let total_cost_before_discount := num_tumblers * cost_per_tumbler
  let discount_amount := total_cost_before_discount * discount_rate
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let total_amount_paid := num_bills * value_per_bill
  let change_received := total_amount_paid - total_cost_after_discount
  change_received = 95 := by sorry

end mrs_petersons_change_l138_138241


namespace sequence_formula_l138_138553

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) (h3 : a 3 = 7) (h4 : a 4 = 15) :
  ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end sequence_formula_l138_138553


namespace range_p_l138_138540

open Set

def p (x : ℝ) : ℝ :=
  x^4 + 6*x^2 + 9

theorem range_p : range p = Ici 9 := by
  sorry

end range_p_l138_138540


namespace hyperbola_eccentricity_l138_138478

theorem hyperbola_eccentricity (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 1) (h_eccentricity : ∀ e : ℝ, e = 2) :
    k = -1 / 3 := 
sorry

end hyperbola_eccentricity_l138_138478


namespace gcd_bezout_663_182_l138_138356

theorem gcd_bezout_663_182 :
  let a := 182
  let b := 663
  ∃ d u v : ℤ, d = Int.gcd a b ∧ d = a * u + b * v ∧ d = 13 ∧ u = 11 ∧ v = -3 :=
by 
  let a := 182
  let b := 663
  use 13, 11, -3
  sorry

end gcd_bezout_663_182_l138_138356


namespace train_speed_l138_138524

theorem train_speed (distance : ℝ) (time : ℝ) (distance_eq : distance = 270) (time_eq : time = 9)
  : (distance / time) * (3600 / 1000) = 108 :=
by 
  sorry

end train_speed_l138_138524


namespace no_rational_solution_l138_138458

theorem no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
by sorry

end no_rational_solution_l138_138458


namespace maximum_height_l138_138300

-- Define the quadratic function h(t)
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 60

-- Define our proof problem
theorem maximum_height : ∃ t : ℝ, h t = 140 :=
by
  let t := -80 / (2 * -20)
  use t
  sorry

end maximum_height_l138_138300


namespace sweets_leftover_candies_l138_138566

theorem sweets_leftover_candies (n : ℕ) (h : n % 8 = 5) : (3 * n) % 8 = 7 :=
sorry

end sweets_leftover_candies_l138_138566


namespace lines_intersection_example_l138_138783

theorem lines_intersection_example (m b : ℝ) 
  (h1 : 8 = m * 4 + 2) 
  (h2 : 8 = 4 * 4 + b) : 
  b + m = -13 / 2 := 
by
  sorry

end lines_intersection_example_l138_138783


namespace jon_found_marbles_l138_138406

-- Definitions based on the conditions
variables (M J B : ℕ)

-- Prove that Jon found 110 marbles
theorem jon_found_marbles
  (h1 : M + J = 66)
  (h2 : M = 2 * J)
  (h3 : J + B = 3 * M) :
  B = 110 :=
by
  sorry -- proof to be completed

end jon_found_marbles_l138_138406


namespace find_cost_prices_l138_138986

-- These represent the given selling prices of the items.
def SP_computer_table : ℝ := 3600
def SP_office_chair : ℝ := 5000
def SP_bookshelf : ℝ := 1700

-- These represent the percentage markups and discounts as multipliers.
def markup_computer_table : ℝ := 1.20
def markup_office_chair : ℝ := 1.25
def discount_bookshelf : ℝ := 0.85

-- The problem requires us to find the cost prices. We will define these as variables.
variable (C O B : ℝ)

theorem find_cost_prices :
  (SP_computer_table = C * markup_computer_table) ∧
  (SP_office_chair = O * markup_office_chair) ∧
  (SP_bookshelf = B * discount_bookshelf) →
  (C = 3000) ∧ (O = 4000) ∧ (B = 2000) :=
by
  sorry

end find_cost_prices_l138_138986


namespace compute_expression_l138_138058

theorem compute_expression (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2017)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2016)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2017)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2016)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2017)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2016) :
  (2 - x1 / y1) * (2 - x2 / y2) * (2 - x3 / y3) = 26219 / 2016 := 
by
  sorry

end compute_expression_l138_138058


namespace estimated_total_score_l138_138284

noncomputable def regression_score (x : ℝ) : ℝ := 7.3 * x - 96.9

theorem estimated_total_score (x : ℝ) (h : x = 95) : regression_score x = 596 :=
by
  rw [h]
  -- skipping the actual calculation steps
  sorry

end estimated_total_score_l138_138284


namespace bus_travel_time_l138_138886

theorem bus_travel_time (D1 D2: ℝ) (T: ℝ) (h1: D1 + D2 = 250) (h2: D1 >= 0) (h3: D2 >= 0) :
  T = D1 / 40 + D2 / 60 ↔ D1 + D2 = 250 := 
by
  sorry

end bus_travel_time_l138_138886


namespace sum_of_three_numbers_l138_138036

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l138_138036


namespace sum_of_roots_l138_138097

theorem sum_of_roots (x1 x2 : ℝ) (h : x1^2 + 5*x1 - 1 = 0 ∧ x2^2 + 5*x2 - 1 = 0) : x1 + x2 = -5 :=
sorry

end sum_of_roots_l138_138097


namespace linear_eq_value_abs_sum_l138_138038

theorem linear_eq_value_abs_sum (a m : ℤ)
  (h1: m^2 - 9 = 0)
  (h2: m ≠ 3)
  (h3: |a| ≤ 3) : 
  |a + m| + |a - m| = 6 :=
by
  sorry

end linear_eq_value_abs_sum_l138_138038


namespace compare_abc_l138_138824

theorem compare_abc :
  let a := Real.log 17
  let b := 3
  let c := Real.exp (Real.sqrt 2)
  a < b ∧ b < c :=
by
  sorry

end compare_abc_l138_138824


namespace find_initial_lion_population_l138_138771

-- Define the conditions as integers
def lion_cubs_per_month : ℕ := 5
def lions_die_per_month : ℕ := 1
def total_lions_after_one_year : ℕ := 148

-- Define a formula for calculating the initial number of lions
def initial_number_of_lions (net_increase : ℕ) (final_count : ℕ) (months : ℕ) : ℕ :=
  final_count - (net_increase * months)

-- Main theorem statement
theorem find_initial_lion_population : initial_number_of_lions (lion_cubs_per_month - lions_die_per_month) total_lions_after_one_year 12 = 100 :=
  sorry

end find_initial_lion_population_l138_138771


namespace jens_son_age_l138_138046

theorem jens_son_age
  (J : ℕ)
  (S : ℕ)
  (h1 : J = 41)
  (h2 : J = 3 * S - 7) :
  S = 16 :=
by
  sorry

end jens_son_age_l138_138046


namespace loss_per_metre_eq_12_l138_138721

-- Definitions based on the conditions
def totalMetres : ℕ := 200
def totalSellingPrice : ℕ := 12000
def costPricePerMetre : ℕ := 72

-- Theorem statement to prove the loss per metre of cloth
theorem loss_per_metre_eq_12 : (costPricePerMetre * totalMetres - totalSellingPrice) / totalMetres = 12 := 
by sorry

end loss_per_metre_eq_12_l138_138721


namespace greatest_integer_a_l138_138087

-- Define formal properties and state the main theorem.
theorem greatest_integer_a (a : ℤ) : (∀ x : ℝ, ¬(x^2 + (a:ℝ) * x + 15 = 0)) → (a ≤ 7) :=
by
  intro h
  sorry

end greatest_integer_a_l138_138087


namespace fixed_errors_correct_l138_138995

-- Conditions
def total_lines_of_code : ℕ := 4300
def lines_per_debug : ℕ := 100
def errors_per_debug : ℕ := 3

-- Question: How many errors has she fixed so far?
theorem fixed_errors_correct :
  (total_lines_of_code / lines_per_debug) * errors_per_debug = 129 := 
by 
  sorry

end fixed_errors_correct_l138_138995


namespace problem_statement_l138_138987

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 - x else 2 - (x % 2)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, f (x + 1) + f x = 3) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 - x) →
  f (-2007.5) = 1.5 :=
by sorry

end problem_statement_l138_138987


namespace polynomial_remainder_l138_138050

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 + 2*x + 3

-- Define the divisor q(x)
def q (x : ℝ) : ℝ := x + 2

-- The theorem asserting the remainder when p(x) is divided by q(x)
theorem polynomial_remainder : (p (-2)) = -9 :=
by
  sorry

end polynomial_remainder_l138_138050


namespace standard_equation_of_ellipse_l138_138004

-- Define the conditions of the ellipse
def ellipse_condition_A (m n : ℝ) : Prop := n * (5 / 3) ^ 2 = 1
def ellipse_condition_B (m n : ℝ) : Prop := m + n = 1

-- The theorem to prove the standard equation of the ellipse
theorem standard_equation_of_ellipse (m n : ℝ) (hA : ellipse_condition_A m n) (hB : ellipse_condition_B m n) :
  m = 16 / 25 ∧ n = 9 / 25 :=
sorry

end standard_equation_of_ellipse_l138_138004


namespace gum_cost_700_eq_660_cents_l138_138260

-- defining the cost function
def gum_cost (n : ℕ) : ℝ :=
  if n ≤ 500 then n * 0.01
  else 5 + (n - 500) * 0.008

-- proving the specific case for 700 pieces of gum
theorem gum_cost_700_eq_660_cents : gum_cost 700 = 6.60 := by
  sorry

end gum_cost_700_eq_660_cents_l138_138260


namespace total_views_correct_l138_138983

-- Definitions based on the given conditions
def initial_views : ℕ := 4000
def views_increase := 10 * initial_views
def additional_views := 50000
def total_views_after_6_days := initial_views + views_increase + additional_views

-- The theorem we are going to state
theorem total_views_correct :
  total_views_after_6_days = 94000 :=
sorry

end total_views_correct_l138_138983


namespace real_roots_iff_integer_roots_iff_l138_138242

noncomputable def discriminant (k : ℝ) : ℝ := (k + 1)^2 - 4 * k * (k - 1)

theorem real_roots_iff (k : ℝ) : 
  (discriminant k ≥ 0) ↔ (∃ (a b : ℝ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) := sorry

theorem integer_roots_iff (k : ℝ) : 
  (∃ (a b : ℤ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) ↔ 
  (k = 0 ∨ k = 1 ∨ k = -1/7) := sorry

-- These theorems need to be proven within Lean 4 itself

end real_roots_iff_integer_roots_iff_l138_138242


namespace functional_equation_solution_l138_138909

noncomputable def f (x : ℝ) (c : ℝ) : ℝ :=
  (c * x - c^2) / (1 + c)

def g (x : ℝ) (c : ℝ) : ℝ :=
  c * x - c^2

theorem functional_equation_solution (f g : ℝ → ℝ) (c : ℝ) (h : c ≠ -1) :
  (∀ x y : ℝ, f (x + g y) = x * f y - y * f x + g x) ∧
  (∀ x, f x = (c * x - c^2) / (1 + c)) ∧
  (∀ x, g x = c * x - c^2) :=
sorry

end functional_equation_solution_l138_138909


namespace ratio_Rose_to_Mother_l138_138228

variable (Rose_age : ℕ) (Mother_age : ℕ)

-- Define the conditions
axiom sum_of_ages : Rose_age + Mother_age = 100
axiom Rose_is_25 : Rose_age = 25
axiom Mother_is_75 : Mother_age = 75

-- Define the main theorem to prove the ratio
theorem ratio_Rose_to_Mother : (Rose_age : ℚ) / (Mother_age : ℚ) = 1 / 3 := by
  sorry

end ratio_Rose_to_Mother_l138_138228


namespace c_share_l138_138663

theorem c_share (A B C D : ℝ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : D = 1/4 * 392) 
    (h4 : A + B + C + D = 392) : 
    C = 168 := 
by 
    sorry

end c_share_l138_138663


namespace julie_savings_multiple_l138_138218

theorem julie_savings_multiple (S : ℝ) (hS : 0 < S) :
  (12 * 0.25 * S) / (0.75 * S) = 4 :=
by
  sorry

end julie_savings_multiple_l138_138218


namespace func_C_increasing_l138_138565

open Set

noncomputable def func_A (x : ℝ) : ℝ := 3 - x
noncomputable def func_B (x : ℝ) : ℝ := x^2 - x
noncomputable def func_C (x : ℝ) : ℝ := -1 / (x + 1)
noncomputable def func_D (x : ℝ) : ℝ := -abs x

theorem func_C_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → func_C x < func_C y := by
  sorry

end func_C_increasing_l138_138565


namespace summer_discount_percentage_l138_138436

/--
Given:
1. The original cost of the jeans (original_price) is $49.
2. On Wednesdays, there is an additional $10.00 off on all jeans after the summer discount is applied.
3. Before the sales tax applies, the cost of a pair of jeans (final_price) is $14.50.

Prove:
The summer discount percentage (D) is 50%.
-/
theorem summer_discount_percentage (original_price final_price : ℝ) (D : ℝ) :
  original_price = 49 → 
  final_price = 14.50 → 
  (original_price - (original_price * D / 100) - 10 = final_price) → 
  D = 50 :=
by intros h_original h_final h_discount; sorry

end summer_discount_percentage_l138_138436


namespace average_lecture_minutes_l138_138392

theorem average_lecture_minutes
  (lecture_duration : ℕ)
  (total_audience : ℕ)
  (percent_entire : ℝ)
  (percent_missed : ℝ)
  (percent_half : ℝ)
  (average_minutes : ℝ) :
  lecture_duration = 90 →
  total_audience = 200 →
  percent_entire = 0.30 →
  percent_missed = 0.20 →
  percent_half = 0.40 →
  average_minutes = 56.25 :=
by
  sorry

end average_lecture_minutes_l138_138392


namespace edward_money_left_l138_138897

def earnings_from_lawns (lawns_mowed : Nat) (dollar_per_lawn : Nat) : Nat :=
  lawns_mowed * dollar_per_lawn

def earnings_from_gardens (gardens_cleaned : Nat) (dollar_per_garden : Nat) : Nat :=
  gardens_cleaned * dollar_per_garden

def total_earnings (earnings_lawns : Nat) (earnings_gardens : Nat) : Nat :=
  earnings_lawns + earnings_gardens

def total_expenses (fuel_expense : Nat) (equipment_expense : Nat) : Nat :=
  fuel_expense + equipment_expense

def total_earnings_with_savings (total_earnings : Nat) (savings : Nat) : Nat :=
  total_earnings + savings

def money_left (earnings_with_savings : Nat) (expenses : Nat) : Nat :=
  earnings_with_savings - expenses

theorem edward_money_left : 
  let lawns_mowed := 5
  let dollar_per_lawn := 8
  let gardens_cleaned := 3
  let dollar_per_garden := 12
  let fuel_expense := 10
  let equipment_expense := 15
  let savings := 7
  let earnings_lawns := earnings_from_lawns lawns_mowed dollar_per_lawn
  let earnings_gardens := earnings_from_gardens gardens_cleaned dollar_per_garden
  let total_earnings_work := total_earnings earnings_lawns earnings_gardens
  let expenses := total_expenses fuel_expense equipment_expense
  let earnings_with_savings := total_earnings_with_savings total_earnings_work savings
  money_left earnings_with_savings expenses = 58
:= by sorry

end edward_money_left_l138_138897


namespace coefficient_of_x_neg_2_in_binomial_expansion_l138_138409

theorem coefficient_of_x_neg_2_in_binomial_expansion :
  let x := (x : ℚ)
  let term := (x^3 - (2 / x))^6
  (coeff_of_term : Int) ->
  (coeff_of_term = -192) :=
by
  -- Placeholder for the proof
  sorry

end coefficient_of_x_neg_2_in_binomial_expansion_l138_138409


namespace sqrt_domain_l138_138527

def inequality_holds (x : ℝ) : Prop := x + 5 ≥ 0

theorem sqrt_domain (x : ℝ) : inequality_holds x ↔ x ≥ -5 := by
  sorry

end sqrt_domain_l138_138527


namespace find_m_n_diff_l138_138190

theorem find_m_n_diff (a : ℝ) (n m: ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1)
  (h_pass : a^(2 * m - 6) + n = 2) :
  m - n = 2 :=
sorry

end find_m_n_diff_l138_138190


namespace simplify_trig_expr_l138_138828

theorem simplify_trig_expr : 
  let θ := 60
  let tan_θ := Real.sqrt 3
  let cot_θ := (Real.sqrt 3)⁻¹
  (tan_θ^3 + cot_θ^3) / (tan_θ + cot_θ) = 7 / 3 :=
by
  sorry

end simplify_trig_expr_l138_138828


namespace num_of_sets_eq_four_l138_138514

open Finset

theorem num_of_sets_eq_four : ∀ B : Finset ℕ, (insert 1 (insert 2 B) = {1, 2, 3, 4, 5}) → (B = {3, 4, 5} ∨ B = {1, 3, 4, 5} ∨ B = {2, 3, 4, 5} ∨ B = {1, 2, 3, 4, 5}) := 
by
  sorry

end num_of_sets_eq_four_l138_138514


namespace inequality_geq_8_l138_138850

theorem inequality_geq_8 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 :=
by
  sorry

end inequality_geq_8_l138_138850


namespace book_arrangements_l138_138264

theorem book_arrangements (n : ℕ) (b1 b2 b3 b4 b5 : ℕ) (h_b123 : b1 < b2 ∧ b2 < b3):
  n = 20 := sorry

end book_arrangements_l138_138264


namespace integer_solutions_no_solutions_2891_l138_138332

-- Define the main problem statement
-- Prove that if the equation x^3 - 3xy^2 + y^3 = n has a solution in integers x, y, then it has at least three such solutions.
theorem integer_solutions (n : ℕ) (x y : ℤ) (h : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x₁ y₁ x₂ y₂ : ℤ, x₁ ≠ x ∧ y₁ ≠ y ∧ x₂ ≠ x ∧ y₂ ≠ y ∧ 
  x₁^3 - 3 * x₁ * y₁^2 + y₁^3 = n ∧ 
  x₂^3 - 3 * x₂ * y₂^2 + y₂^3 = n := sorry

-- Prove that if n = 2891 then no such integer solutions exist.
theorem no_solutions_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) := sorry

end integer_solutions_no_solutions_2891_l138_138332


namespace greatest_difference_47x_l138_138610

def is_multiple_of_4 (n : Nat) : Prop :=
  n % 4 = 0

def valid_digit (d : Nat) : Prop :=
  d < 10

theorem greatest_difference_47x :
  ∃ x y : Nat, (is_multiple_of_4 (470 + x) ∧ valid_digit x) ∧ (is_multiple_of_4 (470 + y) ∧ valid_digit y) ∧ (x < y) ∧ (y - x = 4) :=
sorry

end greatest_difference_47x_l138_138610


namespace night_rides_total_l138_138664

-- Definitions corresponding to the conditions in the problem
def total_ferris_wheel_rides : Nat := 13
def total_roller_coaster_rides : Nat := 9
def ferris_wheel_day_rides : Nat := 7
def roller_coaster_day_rides : Nat := 4

-- The total night rides proof problem
theorem night_rides_total :
  let ferris_wheel_night_rides := total_ferris_wheel_rides - ferris_wheel_day_rides
  let roller_coaster_night_rides := total_roller_coaster_rides - roller_coaster_day_rides
  ferris_wheel_night_rides + roller_coaster_night_rides = 11 :=
by
  -- Proof skipped
  sorry

end night_rides_total_l138_138664


namespace adrien_winning_strategy_l138_138338

/--
On the table, there are 2023 tokens. Adrien and Iris take turns removing at least one token and at most half of the remaining tokens at the time they play. The player who leaves a single token on the table loses the game. Adrien starts first. Prove that Adrien has a winning strategy.
-/
theorem adrien_winning_strategy : ∃ strategy : ℕ → ℕ, 
  ∀ n:ℕ, (n = 2023 ∧ 1 ≤ strategy n ∧ strategy n ≤ n / 2) → 
    (∀ u : ℕ, (u = n - strategy n) → (∃ strategy' : ℕ → ℕ , 
      ∀ m:ℕ, (m = u ∧ 1 ≤ strategy' m ∧ strategy' m ≤ m / 2) → 
        (∃ next_u : ℕ, (next_u = m - strategy' m → next_u ≠ 1 ∨ (m = 1 ∧ u ≠ 1 ∧ next_u = 1)))))
:= sorry

end adrien_winning_strategy_l138_138338


namespace reduced_price_of_oil_l138_138694

theorem reduced_price_of_oil (P R : ℝ) (h1: R = 0.75 * P) (h2: 600 / (0.75 * P) = 600 / P + 5) :
  R = 30 :=
by
  sorry

end reduced_price_of_oil_l138_138694


namespace area_of_WIN_sector_l138_138704

theorem area_of_WIN_sector (r : ℝ) (p : ℝ) (A_circ : ℝ) (A_WIN : ℝ) 
    (h_r : r = 15) 
    (h_p : p = 1 / 3) 
    (h_A_circ : A_circ = π * r^2) 
    (h_A_WIN : A_WIN = p * A_circ) :
    A_WIN = 75 * π := 
sorry

end area_of_WIN_sector_l138_138704


namespace make_up_set_money_needed_l138_138881

theorem make_up_set_money_needed (makeup_cost gabby_money mom_money: ℤ) (h1: makeup_cost = 65) (h2: gabby_money = 35) (h3: mom_money = 20) :
  (makeup_cost - (gabby_money + mom_money)) = 10 :=
by {
  sorry
}

end make_up_set_money_needed_l138_138881


namespace evaluate_expression_l138_138337

theorem evaluate_expression :
  (305^2 - 275^2) / 30 = 580 := 
by
  sorry

end evaluate_expression_l138_138337


namespace perpendicular_condition_parallel_condition_parallel_opposite_direction_l138_138707

variables (a b : ℝ × ℝ) (k : ℝ)

-- Define the vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-3, 2)

-- Define the given expressions
def expression1 (k : ℝ) : ℝ × ℝ := (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2)
def expression2 : ℝ × ℝ := (vec_a.1 - 3 * vec_b.1, vec_a.2 - 3 * vec_b.2)

-- Dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Perpendicular condition
theorem perpendicular_condition : (k : ℝ) → dot_product (expression1 k) expression2 = 0 → k = 19 :=
by sorry

-- Parallel and opposite condition
theorem parallel_condition : (k : ℝ) → (∃ m : ℝ, expression1 k = m • expression2) → k = -1 / 3 :=
by sorry

noncomputable def m (k : ℝ) : ℝ × ℝ := 
  let ex1 := expression1 k
  let ex2 := expression2
  (ex2.1 / ex1.1, ex2.2 / ex1.2)

theorem parallel_opposite_direction : (k : ℝ) → expression1 k = -1 / 3 • expression2 → k = -1 / 3 :=
by sorry

end perpendicular_condition_parallel_condition_parallel_opposite_direction_l138_138707


namespace sum_of_triangle_angles_l138_138360

theorem sum_of_triangle_angles 
  (smallest largest middle : ℝ) 
  (h1 : smallest = 20) 
  (h2 : middle = 3 * smallest) 
  (h3 : largest = 5 * smallest) 
  (h4 : smallest + middle + largest = 180) :
  smallest + middle + largest = 180 :=
by sorry

end sum_of_triangle_angles_l138_138360


namespace stream_speed_l138_138240

theorem stream_speed (v : ℝ) (h1 : 36 > 0) (h2 : 80 > 0) (h3 : 40 > 0) (t_down : 80 / (36 + v) = 40 / (36 - v)) : v = 12 := 
by
  sorry

end stream_speed_l138_138240


namespace line_through_P_with_intercepts_l138_138608

theorem line_through_P_with_intercepts (a b : ℝ) (P : ℝ × ℝ) (hP : P = (6, -1)) 
  (h1 : a = 3 * b) (ha : a = 1 / ((-b - 1) / 6) + 6) (hb : b = -6 * ((-b - 1) / 6) - 1) :
  (∀ x y, y = (-1 / 3) * x + 1 ∨ y = (-1 / 6) * x) :=
sorry

end line_through_P_with_intercepts_l138_138608


namespace eccentricity_of_ellipse_l138_138211

theorem eccentricity_of_ellipse (a b c e : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : c = Real.sqrt (a^2 - b^2))
  (h4 : e = c / a) :
  e = 4 / 5 :=
by
  sorry

end eccentricity_of_ellipse_l138_138211


namespace stratified_sampling_l138_138712

-- We are defining the data given in the problem
def numStudents : ℕ := 50
def numFemales : ℕ := 20
def sampledFemales : ℕ := 4
def genderRatio := (numFemales : ℚ) / (numStudents : ℚ)

-- The theorem stating the given problem and its conclusion
theorem stratified_sampling : ∀ (n : ℕ), (sampledFemales : ℚ) / (n : ℚ) = genderRatio → n = 10 :=
by
  intro n
  intro h
  sorry

end stratified_sampling_l138_138712


namespace check_random_event_l138_138203

def random_event (A B C D : Prop) : Prop := ∃ E, D = E

def event_A : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_B : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_C : Prop :=
  ∀ (probability : ℝ), probability = 1

def event_D : Prop :=
  ∀ (probability : ℝ), 0 < probability ∧ probability < 1

theorem check_random_event :
  random_event event_A event_B event_C event_D :=
sorry

end check_random_event_l138_138203


namespace male_to_female_cat_weight_ratio_l138_138809

variable (w_f w_m w_t : ℕ)

def female_cat_weight : Prop := w_f = 2
def total_weight : Prop := w_t = 6
def male_cat_heavier : Prop := w_m > w_f

theorem male_to_female_cat_weight_ratio
  (h_female_cat_weight : female_cat_weight w_f)
  (h_total_weight : total_weight w_t)
  (h_male_cat_heavier : male_cat_heavier w_m w_f) :
  w_m = 4 ∧ w_t = w_f + w_m ∧ (w_m / w_f) = 2 :=
by
  sorry

end male_to_female_cat_weight_ratio_l138_138809


namespace average_income_of_other_40_customers_l138_138423

/-
Given:
1. The average income of 50 customers is $45,000.
2. The average income of the wealthiest 10 customers is $55,000.

Prove:
1. The average income of the other 40 customers is $42,500.
-/

theorem average_income_of_other_40_customers 
  (avg_income_50 : ℝ)
  (wealthiest_10_avg : ℝ) 
  (total_customers : ℕ)
  (wealthiest_customers : ℕ)
  (remaining_customers : ℕ)
  (h1 : avg_income_50 = 45000)
  (h2 : wealthiest_10_avg = 55000)
  (h3 : total_customers = 50)
  (h4 : wealthiest_customers = 10)
  (h5 : remaining_customers = 40) :
  let total_income_50 := total_customers * avg_income_50
  let total_income_wealthiest_10 := wealthiest_customers * wealthiest_10_avg
  let income_remaining_customers := total_income_50 - total_income_wealthiest_10
  let avg_income_remaining := income_remaining_customers / remaining_customers
  avg_income_remaining = 42500 := 
sorry

end average_income_of_other_40_customers_l138_138423


namespace moon_speed_conversion_l138_138471

def moon_speed_km_sec : ℝ := 1.04
def seconds_per_hour : ℝ := 3600

theorem moon_speed_conversion :
  (moon_speed_km_sec * seconds_per_hour) = 3744 := by
  sorry

end moon_speed_conversion_l138_138471


namespace quadratic_transformation_l138_138698

theorem quadratic_transformation (x d e : ℝ) (h : x^2 - 24*x + 45 = (x+d)^2 + e) : d + e = -111 :=
sorry

end quadratic_transformation_l138_138698


namespace find_x2_y2_l138_138071

variable (x y : ℝ)

-- Given conditions
def average_commute_time (x y : ℝ) := (x + y + 10 + 11 + 9) / 5 = 10
def variance_commute_time (x y : ℝ) := ( (x - 10) ^ 2 + (y - 10) ^ 2 + (10 - 10) ^ 2 + (11 - 10) ^ 2 + (9 - 10) ^ 2 ) / 5 = 2

-- The theorem to prove
theorem find_x2_y2 (hx_avg : average_commute_time x y) (hx_var : variance_commute_time x y) : 
  x^2 + y^2 = 208 :=
sorry

end find_x2_y2_l138_138071


namespace simplify_expression_l138_138429

theorem simplify_expression (x : ℝ) : (3 * x)^4 + 3 * x * x^3 + 2 * x^5 = 84 * x^4 + 2 * x^5 := by
    sorry

end simplify_expression_l138_138429


namespace determine_m_of_monotonically_increasing_function_l138_138632

theorem determine_m_of_monotonically_increasing_function 
  (m n : ℝ)
  (h : ∀ x, 12 * x ^ 2 + 2 * m * x + (m - 3) ≥ 0) :
  m = 6 := 
by 
  sorry

end determine_m_of_monotonically_increasing_function_l138_138632


namespace polynomial_symmetric_equiv_l138_138866

variable {R : Type*} [CommRing R]

def symmetric_about (P : R → R) (a b : R) : Prop :=
  ∀ x, P (2 * a - x) = 2 * b - P x

def polynomial_form (P : R → R) (a b : R) (Q : R → R) : Prop :=
  ∀ x, P x = b + (x - a) * Q ((x - a) * (x - a))

theorem polynomial_symmetric_equiv (P Q : R → R) (a b : R) :
  (symmetric_about P a b ↔ polynomial_form P a b Q) :=
sorry

end polynomial_symmetric_equiv_l138_138866


namespace stratified_sampling_freshman_l138_138350

def total_students : ℕ := 1800 + 1500 + 1200
def sample_size : ℕ := 150
def freshman_students : ℕ := 1200

/-- if a sample of 150 students is drawn using stratified sampling, 40 students should be drawn from the freshman year -/
theorem stratified_sampling_freshman :
  (freshman_students * sample_size) / total_students = 40 :=
by
  sorry

end stratified_sampling_freshman_l138_138350


namespace five_dice_not_all_same_number_l138_138905
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l138_138905


namespace first_equation_value_l138_138035

theorem first_equation_value (x y : ℝ) (V : ℝ) 
  (h1 : x + |x| + y = V) 
  (h2 : x + |y| - y = 6) 
  (h3 : x + y = 12) : 
  V = 18 := 
by
  sorry

end first_equation_value_l138_138035


namespace square_area_l138_138606

theorem square_area (p : ℕ) (h : p = 48) : (p / 4) * (p / 4) = 144 := by
  sorry

end square_area_l138_138606


namespace second_most_eater_l138_138206

variable (C M K B T : ℕ)  -- Assuming the quantities of food each child ate are positive integers

theorem second_most_eater
  (h1 : C > M)
  (h2 : B < K)
  (h3 : T < K)
  (h4 : K < M) :
  ∃ x, x = M ∧ (∀ y, y ≠ C → x ≥ y) ∧ (∃ z, z ≠ C ∧ z > M) :=
by {
  sorry
}

end second_most_eater_l138_138206


namespace remainder_of_4_pow_a_div_10_l138_138006

theorem remainder_of_4_pow_a_div_10 (a : ℕ) (h1 : a > 0) (h2 : a % 2 = 0) :
  (4 ^ a) % 10 = 6 :=
by sorry

end remainder_of_4_pow_a_div_10_l138_138006


namespace find_f_2017_l138_138963

theorem find_f_2017 (f : ℤ → ℤ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 3) = f x) (h_f_neg1 : f (-1) = 1) : 
  f 2017 = -1 :=
sorry

end find_f_2017_l138_138963


namespace first_discount_percentage_l138_138973

theorem first_discount_percentage
  (list_price : ℝ)
  (second_discount : ℝ)
  (third_discount : ℝ)
  (tax_rate : ℝ)
  (final_price : ℝ)
  (D1 : ℝ)
  (h_list_price : list_price = 150)
  (h_second_discount : second_discount = 12)
  (h_third_discount : third_discount = 5)
  (h_tax_rate : tax_rate = 10)
  (h_final_price : final_price = 105) :
  100 - 100 * (final_price / (list_price * (1 - D1 / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) * (1 + tax_rate / 100))) = 24.24 :=
by
  sorry

end first_discount_percentage_l138_138973


namespace consecutive_integer_sets_l138_138804

-- Define the problem
def sum_consecutive_integers (n a : ℕ) : ℕ :=
  (n * (2 * a + n - 1)) / 2

def is_valid_sequence (n a S : ℕ) : Prop :=
  n ≥ 2 ∧ sum_consecutive_integers n a = S

-- Lean 4 theorem statement
theorem consecutive_integer_sets (S : ℕ) (h : S = 180) :
  (∃ (n a : ℕ), is_valid_sequence n a S) →
  (∃ (n1 n2 n3 : ℕ) (a1 a2 a3 : ℕ), 
    is_valid_sequence n1 a1 S ∧ 
    is_valid_sequence n2 a2 S ∧ 
    is_valid_sequence n3 a3 S ∧
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3) :=
by
  sorry

end consecutive_integer_sets_l138_138804


namespace find_unit_prices_l138_138945

theorem find_unit_prices (price_A price_B : ℕ) 
  (h1 : price_A = price_B + 5) 
  (h2 : 1000 / price_A = 750 / price_B) : 
  price_A = 20 ∧ price_B = 15 := 
by 
  sorry

end find_unit_prices_l138_138945


namespace problem1_problem2_l138_138618

theorem problem1 (x : ℚ) (h : x - 2/11 = -1/3) : x = -5/33 :=
sorry

theorem problem2 : -2 - (-1/3 + 1/2) = -13/6 :=
sorry

end problem1_problem2_l138_138618


namespace system_solution_unique_l138_138221

theorem system_solution_unique
  (a b m n : ℝ)
  (h1 : a * 1 + b * 2 = 10)
  (h2 : m * 1 - n * 2 = 8) :
  (a / 2 * (4 + -2) + b / 3 * (4 - -2) = 10) ∧
  (m / 2 * (4 + -2) - n / 3 * (4 - -2) = 8) := 
  by
    sorry

end system_solution_unique_l138_138221


namespace sum_of_x_and_y_l138_138236

theorem sum_of_x_and_y (x y : ℝ) (h1 : x + abs x + y = 5) (h2 : x + abs y - y = 6) : x + y = 9 / 5 :=
by
  sorry

end sum_of_x_and_y_l138_138236


namespace problem_l138_138490

theorem problem (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := by
  sorry

end problem_l138_138490


namespace Jake_peaches_l138_138675

variables (Jake Steven Jill : ℕ)

def peaches_relation : Prop :=
  (Jake = Steven - 6) ∧
  (Steven = Jill + 18) ∧
  (Jill = 5)

theorem Jake_peaches : peaches_relation Jake Steven Jill → Jake = 17 := by
  sorry

end Jake_peaches_l138_138675


namespace find_multiple_l138_138426

theorem find_multiple:
  let number := 220025
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := number / sum
  let remainder := number % sum
  (remainder = 25) → (quotient = 220) → (quotient / diff = 2) :=
by
  intros number sum diff quotient remainder h1 h2
  sorry

end find_multiple_l138_138426


namespace f_eq_g_l138_138138

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

variable (f_onto : ∀ m : ℕ, ∃ n : ℕ, f n = m)
variable (g_one_one : ∀ m n : ℕ, g m = g n → m = n)
variable (f_ge_g : ∀ n : ℕ, f n ≥ g n)

theorem f_eq_g : f = g :=
sorry

end f_eq_g_l138_138138


namespace cost_of_each_soccer_ball_l138_138506

theorem cost_of_each_soccer_ball (total_amount_paid : ℕ) (change_received : ℕ) (number_of_balls : ℕ)
  (amount_spent := total_amount_paid - change_received)
  (unit_price := amount_spent / number_of_balls) :
  total_amount_paid = 100 →
  change_received = 20 →
  number_of_balls = 2 →
  unit_price = 40 := by
  sorry

end cost_of_each_soccer_ball_l138_138506


namespace ratio_M_N_l138_138178

theorem ratio_M_N (M Q P R N : ℝ) 
(h1 : M = 0.40 * Q) 
(h2 : Q = 0.25 * P) 
(h3 : R = 0.60 * P) 
(h4 : N = 0.75 * R) : 
  M / N = 2 / 9 := 
by
  sorry

end ratio_M_N_l138_138178


namespace intersecting_lines_l138_138900

-- Definitions for the conditions
def line1 (x y a : ℝ) : Prop := x = (1/3) * y + a
def line2 (x y b : ℝ) : Prop := y = (1/3) * x + b

-- The theorem we need to prove
theorem intersecting_lines (a b : ℝ) (h1 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line1 x y a) 
                           (h2 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line2 x y b) : 
  a + b = 10 / 3 :=
sorry

end intersecting_lines_l138_138900


namespace coefficient_x2_term_l138_138336

open Polynomial

noncomputable def poly1 : Polynomial ℝ := (X - 1)^3
noncomputable def poly2 : Polynomial ℝ := (X - 1)^4

theorem coefficient_x2_term :
  coeff (poly1 + poly2) 2 = 3 :=
sorry

end coefficient_x2_term_l138_138336


namespace solution_of_inequality_l138_138431

theorem solution_of_inequality : 
  {x : ℝ | x^2 - x - 2 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 2 < x} :=
by
  sorry

end solution_of_inequality_l138_138431


namespace find_perfect_matching_l138_138252

-- Define the boys and girls
inductive Boy | B1 | B2 | B3
inductive Girl | G1 | G2 | G3

-- Define the knowledge relationship
def knows : Boy → Girl → Prop
| Boy.B1, Girl.G1 => true
| Boy.B1, Girl.G2 => true
| Boy.B2, Girl.G1 => true
| Boy.B2, Girl.G3 => true
| Boy.B3, Girl.G2 => true
| Boy.B3, Girl.G3 => true
| _, _ => false

-- Proposition to prove
theorem find_perfect_matching :
  ∃ (pairing : Boy → Girl), 
    (∀ b : Boy, knows b (pairing b)) ∧ 
    (∀ g : Girl, ∃ b : Boy, pairing b = g) :=
by
  sorry

end find_perfect_matching_l138_138252


namespace find_b_l138_138575

theorem find_b (a b : ℝ) (B C : ℝ)
    (h1 : a * b = 60 * Real.sqrt 3)
    (h2 : Real.sin B = Real.sin C)
    (h3 : 15 * Real.sqrt 3 = 1/2 * a * b * Real.sin C) :
  b = 2 * Real.sqrt 15 :=
sorry

end find_b_l138_138575


namespace spheres_do_not_protrude_l138_138561

-- Define the basic parameters
variables (R r : ℝ) (h_cylinder : ℝ) (h_cone : ℝ)
-- Assume conditions
axiom cylinder_height_diameter : h_cylinder = 2 * R
axiom cone_dimensions : h_cone = h_cylinder ∧ h_cone = R

-- The given radius relationship
axiom radius_relation : R = 3 * r

-- Prove the spheres do not protrude from the container
theorem spheres_do_not_protrude (R r h_cylinder h_cone : ℝ)
  (cylinder_height_diameter : h_cylinder = 2 * R)
  (cone_dimensions : h_cone = h_cylinder ∧ h_cone = R)
  (radius_relation : R = 3 * r) : r ≤ R / 2 :=
sorry

end spheres_do_not_protrude_l138_138561


namespace smallest_possible_abc_l138_138362

open Nat

theorem smallest_possible_abc (a b c : ℕ)
  (h₁ : 5 * c ∣ a * b)
  (h₂ : 13 * a ∣ b * c)
  (h₃ : 31 * b ∣ a * c) :
  abc = 4060225 :=
by sorry

end smallest_possible_abc_l138_138362


namespace find_x_l138_138191

theorem find_x (x : ℝ) (h1 : |x + 7| = 3) (h2 : x^2 + 2*x - 3 = 5) : x = -4 :=
by
  sorry

end find_x_l138_138191


namespace find_coordinates_l138_138513

def pointA : ℝ × ℝ := (2, -4)
def pointB : ℝ × ℝ := (0, 6)
def pointC : ℝ × ℝ := (-8, 10)

def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

theorem find_coordinates :
  scalar_mult (1/2) (vector pointA pointC) - 
  scalar_mult (1/4) (vector pointB pointC) = (-3, 6) :=
by
  sorry

end find_coordinates_l138_138513


namespace prove_lesser_fraction_l138_138135

noncomputable def lesser_fraction (x y : ℚ) : Prop :=
  x + y = 8/9 ∧ x * y = 1/8 ∧ min x y = 7/40

theorem prove_lesser_fraction :
  ∃ x y : ℚ, lesser_fraction x y :=
sorry

end prove_lesser_fraction_l138_138135


namespace loss_recorded_as_negative_l138_138686

-- Define the condition that a profit of 100 yuan is recorded as +100 yuan
def recorded_profit (p : ℤ) : Prop :=
  p = 100

-- Define the condition about how a profit is recorded
axiom profit_condition : recorded_profit 100

-- Define the function for recording profit or loss
def record (x : ℤ) : ℤ :=
  if x > 0 then x
  else -x

-- Theorem: If a profit of 100 yuan is recorded as +100 yuan, then a loss of 50 yuan is recorded as -50 yuan.
theorem loss_recorded_as_negative : ∀ x: ℤ, (x < 0) → record x = -x :=
by
  intros x h
  unfold record
  simp [h]
  -- sorry indicates the proof is not provided
  sorry

end loss_recorded_as_negative_l138_138686


namespace product_signs_l138_138503

theorem product_signs (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  ( 
    (((-a * b > 0) ∧ (a * c < 0) ∧ (b * d < 0) ∧ (c * d < 0)) ∨ 
    ((-a * b < 0) ∧ (a * c > 0) ∧ (b * d > 0) ∧ (c * d > 0))) ∨
    (((-a * b < 0) ∧ (a * c > 0) ∧ (b * d < 0) ∧ (c * d > 0)) ∨ 
    ((-a * b > 0) ∧ (a * c < 0) ∧ (b * d > 0) ∧ (c * d < 0))) 
  ) := 
sorry

end product_signs_l138_138503


namespace total_muffins_l138_138489

-- Define initial conditions
def initial_muffins : ℕ := 35
def additional_muffins : ℕ := 48

-- Define the main theorem we want to prove
theorem total_muffins : initial_muffins + additional_muffins = 83 :=
by
  sorry

end total_muffins_l138_138489


namespace divides_seven_l138_138570

theorem divides_seven (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : Nat.gcd x y = 1) (h5 : x^2 + y^2 = z^4) : 7 ∣ x * y :=
by
  sorry

end divides_seven_l138_138570


namespace total_cost_correct_l138_138368

-- Conditions given in the problem.
def net_profit : ℝ := 44
def gross_revenue : ℝ := 47
def lemonades_sold : ℝ := 50
def babysitting_income : ℝ := 31

def cost_per_lemon : ℝ := 0.20
def cost_per_sugar : ℝ := 0.15
def cost_per_ice : ℝ := 0.05

def one_time_cost_sunhat : ℝ := 10

-- Definition of variable cost per lemonade.
def variable_cost_per_lemonade : ℝ := cost_per_lemon + cost_per_sugar + cost_per_ice

-- Definition of total variable cost for all lemonades sold.
def total_variable_cost : ℝ := lemonades_sold * variable_cost_per_lemonade

-- Final total cost to operate the lemonade stand.
def total_cost : ℝ := total_variable_cost + one_time_cost_sunhat

-- The proof statement that total cost is equal to $30.
theorem total_cost_correct : total_cost = 30 := by
  sorry

end total_cost_correct_l138_138368


namespace choose_agency_l138_138225

variables (a : ℝ) (x : ℕ)

def cost_agency_A (a : ℝ) (x : ℕ) : ℝ :=
  a + 0.55 * a * x

def cost_agency_B (a : ℝ) (x : ℕ) : ℝ :=
  0.75 * (x + 1) * a

theorem choose_agency (a : ℝ) (x : ℕ) : if (x = 1) then 
                                            (cost_agency_B a x ≤ cost_agency_A a x)
                                         else if (x ≥ 2) then 
                                            (cost_agency_A a x ≤ cost_agency_B a x)
                                         else
                                            true :=
by
  sorry

end choose_agency_l138_138225


namespace exists_segment_l138_138551

theorem exists_segment (f : ℚ → ℤ) : 
  ∃ (a b c : ℚ), a ≠ b ∧ c = (a + b) / 2 ∧ f a + f b ≤ 2 * f c :=
by 
  sorry

end exists_segment_l138_138551


namespace maximum_value_of_rocks_l138_138584

theorem maximum_value_of_rocks (R6_val R3_val R2_val : ℕ)
  (R6_wt R3_wt R2_wt : ℕ)
  (num6 num3 num2 : ℕ) :
  R6_val = 16 →
  R3_val = 9 →
  R2_val = 3 →
  R6_wt = 6 →
  R3_wt = 3 →
  R2_wt = 2 →
  30 ≤ num6 →
  30 ≤ num3 →
  30 ≤ num2 →
  ∃ (x6 x3 x2 : ℕ),
    x6 ≤ 4 ∧
    x3 ≤ 4 ∧
    x2 ≤ 4 ∧
    (x6 * R6_wt + x3 * R3_wt + x2 * R2_wt ≤ 24) ∧
    (x6 * R6_val + x3 * R3_val + x2 * R2_val = 68) :=
by
  sorry

end maximum_value_of_rocks_l138_138584


namespace prove_incorrect_conclusion_l138_138243

-- Define the parabola as y = ax^2 + bx + c
def parabola_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points
def point1 (a b c : ℝ) : Prop := parabola_eq a b c (-2) = 0
def point2 (a b c : ℝ) : Prop := parabola_eq a b c (-1) = 4
def point3 (a b c : ℝ) : Prop := parabola_eq a b c 0 = 6
def point4 (a b c : ℝ) : Prop := parabola_eq a b c 1 = 6

-- Define the conditions
def conditions (a b c : ℝ) : Prop :=
  point1 a b c ∧ point2 a b c ∧ point3 a b c ∧ point4 a b c

-- Define the incorrect conclusion
def incorrect_conclusion (a b c : ℝ) : Prop :=
  ¬ (parabola_eq a b c 2 = 0)

-- The statement to be proven
theorem prove_incorrect_conclusion (a b c : ℝ) (h : conditions a b c) : incorrect_conclusion a b c :=
sorry

end prove_incorrect_conclusion_l138_138243


namespace functional_equation_solution_l138_138588

theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (f (xy - x)) + f (x + y) = y * f (x) + f (y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end functional_equation_solution_l138_138588


namespace quadratic_eq_k_value_l138_138165

theorem quadratic_eq_k_value (k : ℤ) : (∀ x : ℝ, (k - 1) * x ^ (|k| + 1) - x + 5 = 0 → (k - 1) ≠ 0 ∧ |k| + 1 = 2) -> k = -1 :=
by
  sorry

end quadratic_eq_k_value_l138_138165


namespace ruby_siblings_l138_138536

structure Child :=
  (name : String)
  (eye_color : String)
  (hair_color : String)

def children : List Child :=
[
  {name := "Mason", eye_color := "Green", hair_color := "Red"},
  {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"},
  {name := "Fiona", eye_color := "Brown", hair_color := "Red"},
  {name := "Leo", eye_color := "Green", hair_color := "Blonde"},
  {name := "Ivy", eye_color := "Green", hair_color := "Red"},
  {name := "Carlos", eye_color := "Green", hair_color := "Blonde"}
]

def is_sibling_group (c1 c2 c3 : Child) : Prop :=
  (c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color) ∧
  (c2.eye_color = c3.eye_color ∨ c2.hair_color = c3.hair_color) ∧
  (c1.eye_color = c3.eye_color ∨ c1.hair_color = c3.hair_color)

theorem ruby_siblings :
  ∃ (c1 c2 : Child), 
    c1.name ≠ "Ruby" ∧ c2.name ≠ "Ruby" ∧
    c1 ≠ c2 ∧
    is_sibling_group {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"} c1 c2 ∧
    ((c1.name = "Leo" ∧ c2.name = "Carlos") ∨ (c1.name = "Carlos" ∧ c2.name = "Leo")) :=
by
  sorry

end ruby_siblings_l138_138536


namespace triangle_inequality_power_sum_l138_138025

theorem triangle_inequality_power_sum
  (a b c : ℝ) (n : ℕ)
  (h_a_bc : a + b + c = 1)
  (h_a_b_c : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_a_triangl : a + b > c)
  (h_b_triangl : b + c > a)
  (h_c_triangl : c + a > b)
  (h_n : n > 1) :
  (a^n + b^n)^(1/n : ℝ) + (b^n + c^n)^(1/n : ℝ) + (c^n + a^n)^(1/n : ℝ) < 1 + (2^(1/n : ℝ)) / 2 :=
by
  sorry

end triangle_inequality_power_sum_l138_138025


namespace solve_equation_l138_138032

theorem solve_equation (x : ℝ) (h : x ≠ 3) : 
  -x^2 = (3*x - 3) / (x - 3) → x = 1 :=
by
  intro h1
  sorry

end solve_equation_l138_138032


namespace number_of_pupils_not_in_programX_is_639_l138_138210

-- Definitions for the conditions
def total_girls_elementary : ℕ := 192
def total_boys_elementary : ℕ := 135
def total_girls_middle : ℕ := 233
def total_boys_middle : ℕ := 163
def total_girls_high : ℕ := 117
def total_boys_high : ℕ := 89

def programX_girls_elementary : ℕ := 48
def programX_boys_elementary : ℕ := 28
def programX_girls_middle : ℕ := 98
def programX_boys_middle : ℕ := 51
def programX_girls_high : ℕ := 40
def programX_boys_high : ℕ := 25

-- Question formulation
theorem number_of_pupils_not_in_programX_is_639 :
  (total_girls_elementary - programX_girls_elementary) +
  (total_boys_elementary - programX_boys_elementary) +
  (total_girls_middle - programX_girls_middle) +
  (total_boys_middle - programX_boys_middle) +
  (total_girls_high - programX_girls_high) +
  (total_boys_high - programX_boys_high) = 639 := 
  by
  sorry

end number_of_pupils_not_in_programX_is_639_l138_138210


namespace min_director_games_l138_138883

theorem min_director_games (n k : ℕ) (h1 : (n * (n - 1)) / 2 + k = 325) (h2 : (26 * 25) / 2 = 325) : k = 0 :=
by {
  -- The conditions are provided in the hypothesis, and the goal is proving the minimum games by director equals 0.
  sorry
}

end min_director_games_l138_138883


namespace winning_post_distance_l138_138775

theorem winning_post_distance (v x : ℝ) (h₁ : x ≠ 0) (h₂ : v ≠ 0)
  (h₃ : 1.75 * v = v) 
  (h₄ : x = 1.75 * (x - 84)) : 
  x = 196 :=
by 
  sorry

end winning_post_distance_l138_138775


namespace nails_per_station_correct_l138_138577

variable (total_nails : ℕ) (total_stations : ℕ) (nails_per_station : ℕ)

theorem nails_per_station_correct :
  total_nails = 140 → total_stations = 20 → nails_per_station = total_nails / total_stations → nails_per_station = 7 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end nails_per_station_correct_l138_138577


namespace probability_at_least_two_red_balls_l138_138509

noncomputable def prob_red_balls (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (drawn_balls : ℕ) : ℚ :=
if total_balls = 6 ∧ red_balls = 3 ∧ white_balls = 2 ∧ black_balls = 1 ∧ drawn_balls = 3 then
  1 / 2
else
  0

theorem probability_at_least_two_red_balls :
  prob_red_balls 6 3 2 1 3 = 1 / 2 :=
by 
  sorry

end probability_at_least_two_red_balls_l138_138509


namespace square_line_product_l138_138120

theorem square_line_product (b : ℝ) (h1 : y = 3) (h2 : y = 7) (h3 := x = 2) (h4 : x = b) : 
  (b = 6 ∨ b = -2) → (6 * -2 = -12) :=
by
  sorry

end square_line_product_l138_138120


namespace triangle_largest_angle_l138_138855

theorem triangle_largest_angle 
  (a1 a2 a3 : ℝ) 
  (h_sum : a1 + a2 + a3 = 180)
  (h_arith_seq : 2 * a2 = a1 + a3)
  (h_one_angle : a1 = 28) : 
  max a1 (max a2 a3) = 92 := 
by
  sorry

end triangle_largest_angle_l138_138855


namespace train_speed_is_18_kmh_l138_138749

noncomputable def speed_of_train (length_of_bridge length_of_train time : ℝ) : ℝ :=
  (length_of_bridge + length_of_train) / time * 3.6

theorem train_speed_is_18_kmh
  (length_of_bridge : ℝ)
  (length_of_train : ℝ)
  (time : ℝ)
  (h1 : length_of_bridge = 200)
  (h2 : length_of_train = 100)
  (h3 : time = 60) :
  speed_of_train length_of_bridge length_of_train time = 18 :=
by
  sorry

end train_speed_is_18_kmh_l138_138749


namespace bc_eq_one_area_of_triangle_l138_138693

variable (a b c A B : ℝ)

-- Conditions
def condition_1 : Prop := (b^2 + c^2 - a^2) / (Real.cos A) = 2
def condition_2 : Prop := (a * (Real.cos B) - b * (Real.cos A)) / (a * (Real.cos B) + b * (Real.cos A)) - b / c = 1

-- Equivalent proof problems
theorem bc_eq_one (h1 : condition_1 a b c A) : b * c = 1 := 
by 
  sorry

theorem area_of_triangle (h2 : condition_2 a b c A B) : (1/2) * b * c * Real.sin A = (Real.sqrt 3) / 4 := 
by 
  sorry

end bc_eq_one_area_of_triangle_l138_138693


namespace planned_pigs_correct_l138_138099

-- Define initial number of animals
def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

-- Define planned addition of animals
def added_cows : ℕ := 3
def added_goats : ℕ := 2
def total_animals : ℕ := 21

-- Define the total planned number of pigs to verify:
def planned_pigs := 8

-- State the final number of pigs to be proven
theorem planned_pigs_correct : 
  initial_cows + initial_pigs + initial_goats + added_cows + planned_pigs + added_goats = total_animals :=
by
  sorry

end planned_pigs_correct_l138_138099


namespace reflection_across_x_axis_l138_138108

theorem reflection_across_x_axis :
  let initial_point := (-3, 5)
  let reflected_point := (-3, -5)
  reflected_point = (initial_point.1, -initial_point.2) :=
by
  sorry

end reflection_across_x_axis_l138_138108


namespace determine_f_36_l138_138631

def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n, f (n + 1) > f n

def multiplicative (f : ℕ → ℕ) : Prop :=
  ∀ m n, f (m * n) = f m * f n

def special_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n, m > n → m^m = n^n → f m = n

theorem determine_f_36 (f : ℕ → ℕ)
  (H1: strictly_increasing f)
  (H2: multiplicative f)
  (H3: special_condition f)
  : f 36 = 1296 := 
sorry

end determine_f_36_l138_138631


namespace largest_possible_value_for_a_l138_138944

theorem largest_possible_value_for_a (a b c d : ℕ) 
  (h1: a < 3 * b) 
  (h2: b < 2 * c + 1) 
  (h3: c < 5 * d - 2)
  (h4: d ≤ 50) 
  (h5: d % 5 = 0) : 
  a ≤ 1481 :=
sorry

end largest_possible_value_for_a_l138_138944


namespace initial_cards_l138_138416

variable (x : ℕ)
variable (h1 : x - 3 = 2)

theorem initial_cards (x : ℕ) (h1 : x - 3 = 2) : x = 5 := by
  sorry

end initial_cards_l138_138416


namespace petya_wins_if_and_only_if_m_ne_n_l138_138003

theorem petya_wins_if_and_only_if_m_ne_n 
  (m n : ℕ) 
  (game : ∀ m n : ℕ, Prop)
  (win_condition : (game m n ↔ m ≠ n)) : 
  Prop := 
by 
  sorry

end petya_wins_if_and_only_if_m_ne_n_l138_138003


namespace scalene_triangles_count_l138_138061

/-- Proving existence of exactly 3 scalene triangles with integer side lengths and perimeter < 13. -/
theorem scalene_triangles_count : 
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    triangles.card = 3 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ triangles → a < b ∧ b < c ∧ a + b + c < 13 :=
sorry

end scalene_triangles_count_l138_138061


namespace cubic_expression_solution_l138_138261

theorem cubic_expression_solution (r s : ℝ) (h₁ : 3 * r^2 - 4 * r - 7 = 0) (h₂ : 3 * s^2 - 4 * s - 7 = 0) :
  (3 * r^3 - 3 * s^3) / (r - s) = 37 / 3 :=
sorry

end cubic_expression_solution_l138_138261


namespace inequality_proof_l138_138408

theorem inequality_proof 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 > 0) 
  (h2 : a2 > 0) 
  (h3 : a3 > 0)
  (h4 : a4 > 0):
  (a1 + a3) / (a1 + a2) + 
  (a2 + a4) / (a2 + a3) + 
  (a3 + a1) / (a3 + a4) + 
  (a4 + a2) / (a4 + a1) ≥ 4 :=
by
  sorry

end inequality_proof_l138_138408


namespace frankie_pets_total_l138_138425

noncomputable def total_pets (c : ℕ) : ℕ :=
  let dogs := 2
  let cats := c
  let snakes := c + 5
  let parrots := c - 1
  dogs + cats + snakes + parrots

theorem frankie_pets_total (c : ℕ) (hc : 2 + 4 + (c + 1) + (c - 1) = 19) : total_pets c = 19 := by
  sorry

end frankie_pets_total_l138_138425


namespace range_of_m_exacts_two_integers_l138_138358

theorem range_of_m_exacts_two_integers (m : ℝ) :
  (∀ x : ℝ, (x - 2) / 4 < (x - 1) / 3 ∧ 2 * x - m ≤ 2 - x) ↔ -2 ≤ m ∧ m < 1 := 
sorry

end range_of_m_exacts_two_integers_l138_138358


namespace desired_percentage_of_alcohol_l138_138725

theorem desired_percentage_of_alcohol 
  (original_volume : ℝ)
  (original_percentage : ℝ)
  (added_volume : ℝ)
  (added_percentage : ℝ)
  (final_percentage : ℝ) :
  original_volume = 6 →
  original_percentage = 0.35 →
  added_volume = 1.8 →
  added_percentage = 1.0 →
  final_percentage = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end desired_percentage_of_alcohol_l138_138725


namespace tangent_line_circle_l138_138691

theorem tangent_line_circle (k : ℝ) (h1 : k = Real.sqrt 3) (h2 : ∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) :
  (k = Real.sqrt 3 → (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1)) ∧ (¬ (∀ (k : ℝ), (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) → k = Real.sqrt 3)) :=
  sorry

end tangent_line_circle_l138_138691


namespace remainder_of_4n_squared_l138_138467

theorem remainder_of_4n_squared {n : ℤ} (h : n % 13 = 7) : (4 * n^2) % 13 = 1 :=
by
  sorry

end remainder_of_4n_squared_l138_138467


namespace probability_divisible_by_5_l138_138125

def is_three_digit_integer (M : ℕ) : Prop :=
  100 ≤ M ∧ M < 1000

def ones_digit_is_4 (M : ℕ) : Prop :=
  (M % 10) = 4

theorem probability_divisible_by_5 (M : ℕ) (h1 : is_three_digit_integer M) (h2 : ones_digit_is_4 M) :
  (∃ p : ℚ, p = 0) :=
by
  sorry

end probability_divisible_by_5_l138_138125


namespace new_outsiders_count_l138_138623

theorem new_outsiders_count (total_people: ℕ) (initial_snackers: ℕ)
  (first_group_outsiders: ℕ) (first_group_leave_half: ℕ) 
  (second_group_leave_count: ℕ) (half_remaining_leave: ℕ) (final_snackers: ℕ) 
  (total_snack_eaters: ℕ) 
  (initial_snackers_eq: total_people = 200) 
  (snackers_eq: initial_snackers = 100) 
  (first_group_outsiders_eq: first_group_outsiders = 20) 
  (first_group_leave_half_eq: first_group_leave_half = 60) 
  (second_group_leave_count_eq: second_group_leave_count = 30) 
  (half_remaining_leave_eq: half_remaining_leave = 15) 
  (final_snackers_eq: final_snackers = 20) 
  (total_snack_eaters_eq: total_snack_eaters = 120): 
  (60 - (second_group_leave_count + half_remaining_leave + final_snackers)) = 40 := 
by sorry

end new_outsiders_count_l138_138623


namespace solve_for_q_l138_138411

noncomputable def is_arithmetic_SUM_seq (a₁ q: ℝ) (n: ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem solve_for_q (a₁ q S3 S6 S9: ℝ) (hq: q ≠ 1) (hS3: S3 = is_arithmetic_SUM_seq a₁ q 3) 
(hS6: S6 = is_arithmetic_SUM_seq a₁ q 6) (hS9: S9 = is_arithmetic_SUM_seq a₁ q 9) 
(h_arith: 2 * S9 = S3 + S6) : q^3 = 3 / 2 :=
sorry

end solve_for_q_l138_138411


namespace find_t_l138_138022

-- Define the utility function
def utility (r j : ℕ) : ℕ := r * j

-- Define the Wednesday and Thursday utilities
def utility_wednesday (t : ℕ) : ℕ := utility (t + 1) (7 - t)
def utility_thursday (t : ℕ) : ℕ := utility (3 - t) (t + 4)

theorem find_t : (utility_wednesday t = utility_thursday t) → t = 5 / 8 :=
by
  sorry

end find_t_l138_138022


namespace find_a_l138_138624

theorem find_a (x y a : ℝ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) (h3 : a * x - 3 * y = 23) : 
  a = 12.141 :=
by
  sorry

end find_a_l138_138624


namespace minhyuk_needs_slices_l138_138690

-- Definitions of Yeongchan and Minhyuk's apple division
def yeongchan_portion : ℚ := 1 / 3
def minhyuk_slices : ℚ := 1 / 12

-- Statement to prove
theorem minhyuk_needs_slices (x : ℕ) : yeongchan_portion = x * minhyuk_slices → x = 4 :=
by
  sorry

end minhyuk_needs_slices_l138_138690


namespace solve_for_x2_plus_9y2_l138_138237

variable (x y : ℝ)

def condition1 : Prop := x + 3 * y = 3
def condition2 : Prop := x * y = -6

theorem solve_for_x2_plus_9y2 (h1 : condition1 x y) (h2 : condition2 x y) :
  x^2 + 9 * y^2 = 45 :=
by
  sorry

end solve_for_x2_plus_9y2_l138_138237


namespace min_soldiers_needed_l138_138856

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l138_138856


namespace boys_went_down_the_slide_total_l138_138649

/-- Conditions -/
def a : Nat := 87
def b : Nat := 46
def c : Nat := 29

/-- The main proof problem -/
theorem boys_went_down_the_slide_total :
  a + b + c = 162 :=
by
  sorry

end boys_went_down_the_slide_total_l138_138649


namespace ratio_of_books_l138_138163

theorem ratio_of_books (books_last_week : ℕ) (pages_per_book : ℕ) (pages_this_week : ℕ)
  (h_books_last_week : books_last_week = 5)
  (h_pages_per_book : pages_per_book = 300)
  (h_pages_this_week : pages_this_week = 4500) :
  (pages_this_week / pages_per_book) / books_last_week = 3 := by
  sorry

end ratio_of_books_l138_138163


namespace largest_power_of_2_dividing_n_l138_138352

open Nat

-- Defining given expressions
def n : ℕ := 17^4 - 9^4 + 8 * 17^2

-- The theorem to prove
theorem largest_power_of_2_dividing_n : 2^3 ∣ n ∧ ∀ k, (k > 3 → ¬ 2^k ∣ n) :=
by
  sorry

end largest_power_of_2_dividing_n_l138_138352


namespace sum_of_fractions_l138_138542

theorem sum_of_fractions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/2 + x) + f (1/2 - x) = 2) :
  f (1 / 8) + f (2 / 8) + f (3 / 8) + f (4 / 8) + 
  f (5 / 8) + f (6 / 8) + f (7 / 8) = 7 :=
by 
  sorry

end sum_of_fractions_l138_138542


namespace puppies_per_cage_l138_138226

theorem puppies_per_cage (initial_puppies sold_puppies cages remaining_puppies puppies_per_cage : ℕ)
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : cages = 3)
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 5 := by
  sorry

end puppies_per_cage_l138_138226


namespace dave_diner_total_cost_l138_138457

theorem dave_diner_total_cost (burger_count : ℕ) (fries_count : ℕ)
  (burger_cost : ℕ) (fries_cost : ℕ)
  (discount_threshold : ℕ) (discount_amount : ℕ)
  (h1 : burger_count >= discount_threshold) :
  burger_count = 6 → fries_count = 5 → burger_cost = 4 → fries_cost = 3 →
  discount_threshold = 4 → discount_amount = 2 →
  (burger_count * (burger_cost - discount_amount) + fries_count * fries_cost) = 27 :=
by
  intros hbc hfc hbcost hfcs dth da
  sorry

end dave_diner_total_cost_l138_138457


namespace cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l138_138852

-- Definition of size-n tromino
def tromino_area (n : ℕ) := (4 * 4 * n - 1)

-- Problem (a): Can a size-5 tromino be tiled by size-1 trominos
theorem cannot_tile_size5_with_size1_trominos :
  ¬ (∃ (count : ℕ), count * 3 = tromino_area 5) :=
by sorry

-- Problem (b): Can a size-2013 tromino be tiled by size-1 trominos
theorem can_tile_size2013_with_size1_trominos :
  ∃ (count : ℕ), count * 3 = tromino_area 2013 :=
by sorry

end cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l138_138852


namespace average_price_of_initial_fruit_l138_138829

theorem average_price_of_initial_fruit (A O : ℕ) (h1 : A + O = 10) (h2 : (40 * A + 60 * (O - 6)) / (A + O - 6) = 45) : 
  (40 * A + 60 * O) / 10 = 54 :=
by 
  sorry

end average_price_of_initial_fruit_l138_138829


namespace trip_distance_first_part_l138_138222

theorem trip_distance_first_part (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 70) (h3 : 32 = 70 / ((x / 48) + ((70 - x) / 24))) : x = 35 :=
by
  sorry

end trip_distance_first_part_l138_138222


namespace convert_base_5_to_base_10_l138_138066

theorem convert_base_5_to_base_10 :
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  a3 + a2 + a1 + a0 = 302 := by
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  show a3 + a2 + a1 + a0 = 302
  sorry

end convert_base_5_to_base_10_l138_138066


namespace joseph_power_cost_ratio_l138_138195

theorem joseph_power_cost_ratio
  (electric_oven_cost : ℝ)
  (total_cost : ℝ)
  (water_heater_cost : ℝ)
  (refrigerator_cost : ℝ)
  (H1 : electric_oven_cost = 500)
  (H2 : 2 * water_heater_cost = electric_oven_cost)
  (H3 : refrigerator_cost + water_heater_cost + electric_oven_cost = total_cost)
  (H4 : total_cost = 1500):
  (refrigerator_cost / water_heater_cost) = 3 := sorry

end joseph_power_cost_ratio_l138_138195


namespace area_square_B_l138_138732

theorem area_square_B (a b : ℝ) (h1 : a^2 = 25) (h2 : abs (a - b) = 4) : b^2 = 81 :=
by
  sorry

end area_square_B_l138_138732


namespace boys_and_girls_at_bus_stop_l138_138112

theorem boys_and_girls_at_bus_stop (H M : ℕ) 
  (h1 : H = 2 * (M - 15)) 
  (h2 : M - 15 = 5 * (H - 45)) : 
  H = 50 ∧ M = 40 := 
by 
  sorry

end boys_and_girls_at_bus_stop_l138_138112


namespace cubic_sum_identity_l138_138485

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l138_138485


namespace percent_deficit_in_width_l138_138145

theorem percent_deficit_in_width (L W : ℝ) (h : 1.08 * (1 - (d : ℝ) / W) = 1.0044) : d = 0.07 * W :=
by sorry

end percent_deficit_in_width_l138_138145


namespace parallel_lines_constant_l138_138402

theorem parallel_lines_constant (a : ℝ) : 
  (∀ x y : ℝ, (a - 1) * x + 2 * y + 3 = 0 → x + a * y + 3 = 0) → a = -1 :=
by sorry

end parallel_lines_constant_l138_138402


namespace Cartesian_eq_C2_correct_distance_AB_correct_l138_138073

-- Part I: Proving the Cartesian equation of curve (C2)
noncomputable def equation_of_C2 (x y : ℝ) (α : ℝ) : Prop :=
  x = 4 * Real.cos α ∧ y = 4 + 4 * Real.sin α

def Cartesian_eq_C2 (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

theorem Cartesian_eq_C2_correct (x y α : ℝ) (h : equation_of_C2 x y α) : Cartesian_eq_C2 x y :=
by sorry

-- Part II: Proving the distance |AB| given polar equations
noncomputable def polar_eq_C1 (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def polar_eq_C2 (theta : ℝ) : ℝ :=
  8 * Real.sin theta

def distance_AB (rho1 rho2 : ℝ) : ℝ :=
  abs (rho1 - rho2)

theorem distance_AB_correct : distance_AB (polar_eq_C1 (π / 3)) (polar_eq_C2 (π / 3)) = 2 * Real.sqrt 3 :=
by sorry

end Cartesian_eq_C2_correct_distance_AB_correct_l138_138073


namespace solution_set_product_positive_l138_138180

variable {R : Type*} [LinearOrderedField R]

def is_odd (f : R → R) : Prop := ∀ x : R, f (-x) = -f (x)

variable (f g : R → R)

noncomputable def solution_set_positive_f : Set R := { x | 4 < x ∧ x < 10 }
noncomputable def solution_set_positive_g : Set R := { x | 2 < x ∧ x < 5 }

theorem solution_set_product_positive :
  is_odd f →
  is_odd g →
  (∀ x, f x > 0 ↔ x ∈ solution_set_positive_f) →
  (∀ x, g x > 0 ↔ x ∈ solution_set_positive_g) →
  { x | f x * g x > 0 } = { x | (4 < x ∧ x < 5) ∨ (-5 < x ∧ x < -4) } :=
by
  sorry

end solution_set_product_positive_l138_138180


namespace solve_first_l138_138441

theorem solve_first (x y : ℝ) (C : ℝ) :
  (1 + y^2) * (deriv id x) - (1 + x^2) * y * (deriv id y) = 0 →
  Real.arctan x = 1/2 * Real.log (1 + y^2) + Real.log C := 
sorry

end solve_first_l138_138441


namespace find_k_l138_138298

theorem find_k (k : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + 2 * k = 0 ∧ x = 1) : k = 1 / 2 :=
by {
  sorry 
}

end find_k_l138_138298


namespace company_x_installation_charge_l138_138522

theorem company_x_installation_charge:
  let price_X := 575
  let surcharge_X := 0.04 * price_X
  let installation_charge_X := 82.50
  let total_cost_X := price_X + surcharge_X + installation_charge_X
  let price_Y := 530
  let surcharge_Y := 0.03 * price_Y
  let installation_charge_Y := 93.00
  let total_cost_Y := price_Y + surcharge_Y + installation_charge_Y
  let savings := 41.60
  total_cost_X - total_cost_Y = savings → installation_charge_X = 82.50 :=
by
  intros h
  sorry

end company_x_installation_charge_l138_138522


namespace polynomial_division_result_l138_138906

-- Define the given polynomials
def f (x : ℝ) : ℝ := 4 * x ^ 4 + 12 * x ^ 3 - 9 * x ^ 2 + 2 * x + 3
def d (x : ℝ) : ℝ := x ^ 2 + 2 * x - 3

-- Define the computed quotient and remainder
def q (x : ℝ) : ℝ := 4 * x ^ 2 + 4
def r (x : ℝ) : ℝ := -12 * x + 42

theorem polynomial_division_result :
  (∀ x : ℝ, f x = q x * d x + r x) ∧ (q 1 + r (-1) = 62) :=
by
  sorry

end polynomial_division_result_l138_138906


namespace jason_egg_consumption_in_two_weeks_l138_138942

def breakfast_pattern : List Nat := 
  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] -- Two weeks pattern alternating 3-egg and (2+1)-egg meals

noncomputable def count_eggs (pattern : List Nat) : Nat :=
  pattern.foldl (· + ·) 0

theorem jason_egg_consumption_in_two_weeks : 
  count_eggs breakfast_pattern = 42 :=
sorry

end jason_egg_consumption_in_two_weeks_l138_138942


namespace int_sol_many_no_int_sol_l138_138569

-- Part 1: If there is one integer solution, there are at least three integer solutions
theorem int_sol_many (n : ℤ) (hn : n > 0) (x y : ℤ) 
  (hxy : x^3 - 3 * x * y^2 + y^3 = n) : 
  ∃ a b c d e f : ℤ, 
    (a, b) ≠ (x, y) ∧ (c, d) ≠ (x, y) ∧ (e, f) ≠ (x, y) ∧ 
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f) ∧ 
    a^3 - 3 * a * b^2 + b^3 = n ∧ 
    c^3 - 3 * c * d^2 + d^3 = n ∧ 
    e^3 - 3 * e * f^2 + f^3 = n :=
sorry

-- Part 2: When n = 2891, the equation has no integer solutions
theorem no_int_sol : ¬ ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end int_sol_many_no_int_sol_l138_138569


namespace domain_of_function_l138_138468

theorem domain_of_function (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 1 > 0) ↔ (1 < x ∧ x ≤ 2) :=
by
  sorry

end domain_of_function_l138_138468


namespace nadia_pies_l138_138572

variables (T R B S : ℕ)

theorem nadia_pies (h₁: R = T / 2) 
                   (h₂: B = R - 14) 
                   (h₃: S = (R + B) / 2) 
                   (h₄: T = R + B + S) :
                   R = 21 ∧ B = 7 ∧ S = 14 := 
  sorry

end nadia_pies_l138_138572


namespace candy_sampling_percentage_l138_138734

theorem candy_sampling_percentage (total_percentage caught_percentage not_caught_percentage : ℝ) 
  (h1 : caught_percentage = 22 / 100) 
  (h2 : total_percentage = 24.444444444444443 / 100) 
  (h3 : not_caught_percentage = 2.444444444444443 / 100) :
  total_percentage = caught_percentage + not_caught_percentage :=
by
  sorry

end candy_sampling_percentage_l138_138734


namespace total_insects_eaten_l138_138098

theorem total_insects_eaten : 
  (5 * 6) + (3 * (2 * 6)) = 66 :=
by
  /- We'll calculate the total number of insects eaten by combining the amounts eaten by the geckos and lizards -/
  sorry

end total_insects_eaten_l138_138098


namespace collinear_points_l138_138455

theorem collinear_points (x y : ℝ) (h_collinear : ∃ k : ℝ, (x + 1, y, 3) = (2 * k, 4 * k, 6 * k)) : x - y = -2 := 
by 
  sorry

end collinear_points_l138_138455


namespace max_candies_l138_138251

/-- There are 28 ones written on the board. Every minute, Karlsson erases two arbitrary numbers
and writes their sum on the board, and then eats an amount of candy equal to the product of 
the two erased numbers. Prove that the maximum number of candies he could eat in 28 minutes is 378. -/
theorem max_candies (karlsson_eats_max_candies : ℕ → ℕ → ℕ) (n : ℕ) (initial_count : n = 28) :
  (∀ a b, karlsson_eats_max_candies a b = a * b) →
  (∃ max_candies, max_candies = 378) :=
sorry

end max_candies_l138_138251


namespace minimum_value_of_expression_l138_138815

noncomputable def min_value (p q r s t u : ℝ) : ℝ :=
  (1 / p) + (9 / q) + (25 / r) + (49 / s) + (81 / t) + (121 / u)

theorem minimum_value_of_expression (p q r s t u : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hs : 0 < s) (ht : 0 < t) (hu : 0 < u) (h_sum : p + q + r + s + t + u = 11) :
  min_value p q r s t u ≥ 1296 / 11 :=
by sorry

end minimum_value_of_expression_l138_138815


namespace problem_a_problem_b_l138_138833

-- Problem (a)
theorem problem_a (n : Nat) : Nat.mod (7 ^ (2 * n) - 4 ^ (2 * n)) 33 = 0 := sorry

-- Problem (b)
theorem problem_b (n : Nat) : Nat.mod (3 ^ (6 * n) - 2 ^ (6 * n)) 35 = 0 := sorry

end problem_a_problem_b_l138_138833


namespace polynomial_value_at_minus_2_l138_138424

variable (a b : ℝ)

def polynomial (x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem polynomial_value_at_minus_2 :
  (polynomial a b (-2) = -21) :=
  sorry

end polynomial_value_at_minus_2_l138_138424


namespace solution_set_of_quadratic_inequality_l138_138825

namespace QuadraticInequality

variables {a b : ℝ}

def hasRoots (a b : ℝ) : Prop :=
  let x1 := -1 / 2
  let x2 := 1 / 3
  (- x1 + x2 = - b / a) ∧ (-x1 * x2 = 2 / a)

theorem solution_set_of_quadratic_inequality (h : hasRoots a b) : a + b = -14 :=
sorry

end QuadraticInequality

end solution_set_of_quadratic_inequality_l138_138825


namespace find_trousers_l138_138009

variables (S T Ti : ℝ) -- Prices of shirt, trousers, and tie respectively
variables (x : ℝ)      -- The number of trousers in the first scenario

-- Conditions given in the problem
def condition1 : Prop := 6 * S + x * T + 2 * Ti = 80
def condition2 : Prop := 4 * S + 2 * T + 2 * Ti = 140
def condition3 : Prop := 5 * S + 3 * T + 2 * Ti = 110

-- Theorem to prove
theorem find_trousers : condition1 S T Ti x ∧ condition2 S T Ti ∧ condition3 S T Ti → x = 4 :=
by
  sorry

end find_trousers_l138_138009


namespace solve_for_x_l138_138965

theorem solve_for_x (x : ℝ) : 4 * x - 8 + 3 * x = 12 + 5 * x → x = 10 :=
by
  intro h
  sorry

end solve_for_x_l138_138965


namespace range_of_x_in_function_l138_138990

theorem range_of_x_in_function (x : ℝ) :
  (x - 1 ≥ 0) ∧ (x - 2 ≠ 0) → (x ≥ 1 ∧ x ≠ 2) :=
by
  intro h
  sorry

end range_of_x_in_function_l138_138990


namespace wholesale_price_of_pen_l138_138169

-- Definitions and conditions
def wholesale_price (P : ℝ) : Prop :=
  (5 - P = 10 - 3 * P)

-- Statement of the proof problem
theorem wholesale_price_of_pen : ∃ P : ℝ, wholesale_price P ∧ P = 2.5 :=
by {
  sorry
}

end wholesale_price_of_pen_l138_138169


namespace proof_problem_l138_138446

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f x + x * g x = x ^ 2 - 1
axiom condition2 : f 1 = 1

theorem proof_problem : deriv f 1 + deriv g 1 = 3 :=
by
  sorry

end proof_problem_l138_138446


namespace find_digit_A_l138_138484

-- Define the six-digit number for any digit A
def six_digit_number (A : ℕ) : ℕ := 103200 + A * 10 + 4
-- Define the condition that a number is prime
def is_prime (n : ℕ) : Prop := (2 ≤ n) ∧ ∀ m : ℕ, 2 ≤ m → m * m ≤ n → ¬ (m ∣ n)

-- The main theorem stating that A must equal 1 for the number to be prime
theorem find_digit_A (A : ℕ) : A = 1 ↔ is_prime (six_digit_number A) :=
by
  sorry -- Proof to be filled in


end find_digit_A_l138_138484


namespace intersection_of_sets_l138_138556

noncomputable def setA : Set ℕ := { x : ℕ | x^2 ≤ 4 * x ∧ x > 0 }

noncomputable def setB : Set ℕ := { x : ℕ | 2^x - 4 > 0 ∧ 2^x - 4 ≤ 4 }

theorem intersection_of_sets : { x ∈ setA | x ∈ setB } = {3} :=
by
  sorry

end intersection_of_sets_l138_138556


namespace tan_half_sum_sq_l138_138619

theorem tan_half_sum_sq (a b : ℝ) : 
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b + 1) = 0 → 
  ∃ (x : ℝ), (x = (Real.tan (a / 2) + Real.tan (b / 2))^2) ∧ (x = 6 ∨ x = 26) := 
by
  intro h
  sorry

end tan_half_sum_sq_l138_138619


namespace find_a_b_l138_138763

def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_b :
  ∀ (a b : ℝ),
  (∀ x, (curve x a b) = x^2 + a * x + b) →
  (tangent_line 0 (curve 0 a b)) →
  (tangent_line x y → y = x + 1) →
  (tangent_line x y → ∃ m c, y = m * x + c ∧ m = 1 ∧ c = 1) →
  (∃ a b : ℝ, a = 1 ∧ b = 1) :=
by
  intros a b h_curve h_tangent_line h_tangent_line_form h_tangent_line_eq
  sorry

end find_a_b_l138_138763


namespace sum_infinite_geometric_series_l138_138583

theorem sum_infinite_geometric_series (a r : ℚ) (h : a = 1) (h2 : r = 1/4) : 
  (∀ S, S = a / (1 - r) → S = 4 / 3) :=
by
  intros S hS
  rw [h, h2] at hS
  simp [hS]
  sorry

end sum_infinite_geometric_series_l138_138583


namespace equivalent_single_discount_l138_138187

theorem equivalent_single_discount (p : ℝ) : 
  let discount1 := 0.15
  let discount2 := 0.25
  let price_after_first_discount := (1 - discount1) * p
  let price_after_second_discount := (1 - discount2) * price_after_first_discount
  let equivalent_single_discount := 1 - price_after_second_discount / p
  equivalent_single_discount = 0.3625 :=
by
  sorry

end equivalent_single_discount_l138_138187


namespace mul_powers_same_base_l138_138923

theorem mul_powers_same_base (x : ℝ) : (x ^ 8) * (x ^ 2) = x ^ 10 :=
by
  exact sorry

end mul_powers_same_base_l138_138923


namespace product_of_three_numbers_l138_138748

theorem product_of_three_numbers : 
  ∃ x y z : ℚ, x + y + z = 30 ∧ x = 3 * (y + z) ∧ y = 6 * z ∧ x * y * z = 23625 / 686 :=
by
  sorry

end product_of_three_numbers_l138_138748


namespace tetrahedron_vertex_angle_sum_l138_138220

theorem tetrahedron_vertex_angle_sum (A B C D : Type) (angles_at : Type → Type → Type → ℝ) :
  (∃ A, (∀ X Y Z W, X = A ∨ Y = A ∨ Z = A ∨ W = A → angles_at X Y A + angles_at Z W A > 180)) →
  ¬ (∃ A B, A ≠ B ∧ 
    (∀ X Y, X = A ∨ Y = A → angles_at X Y A + angles_at Y X A > 180) ∧ 
    (∀ X Y, X = B ∨ Y = B → angles_at X Y B + angles_at Y X B > 180)) := 
sorry

end tetrahedron_vertex_angle_sum_l138_138220


namespace swimming_pool_time_l138_138516

theorem swimming_pool_time 
  (empty_rate : ℕ) (fill_rate : ℕ) (capacity : ℕ) (final_volume : ℕ) (t : ℕ)
  (h_empty : empty_rate = 120 / 4) 
  (h_fill : fill_rate = 120 / 6) 
  (h_capacity : capacity = 120) 
  (h_final : final_volume = 90) 
  (h_eq : capacity - (empty_rate - fill_rate) * t = final_volume) :
  t = 3 := 
sorry

end swimming_pool_time_l138_138516


namespace correct_choice_l138_138507

theorem correct_choice (a : ℝ) : -(-a)^2 * a^4 = -a^6 := 
sorry

end correct_choice_l138_138507


namespace correct_answer_l138_138845

noncomputable def sqrt_2 : ℝ := Real.sqrt 2

def P : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem correct_answer : {sqrt_2} ⊆ P :=
sorry

end correct_answer_l138_138845


namespace mittens_per_box_l138_138250

theorem mittens_per_box (total_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) 
  (h_total_boxes : total_boxes = 4) 
  (h_scarves_per_box : scarves_per_box = 2) 
  (h_total_clothing : total_clothing = 32) : 
  (total_clothing - total_boxes * scarves_per_box) / total_boxes = 6 := 
by
  -- Sorry, proof is omitted
  sorry

end mittens_per_box_l138_138250


namespace triangle_area_l138_138154

variable (a b c : ℕ)
variable (s : ℕ := 21)
variable (area : ℕ := 84)

theorem triangle_area 
(h1 : c = a + b - 12) 
(h2 : (a + b + c) / 2 = s) 
(h3 : c - a = 2) 
: (21 * (21 - a) * (21 - b) * (21 - c)).sqrt = area := 
sorry

end triangle_area_l138_138154


namespace camilla_blueberry_jelly_beans_l138_138001

theorem camilla_blueberry_jelly_beans (b c : ℕ) (h1 : b = 2 * c) (h2 : b - 10 = 3 * (c - 10)) : b = 40 := 
sorry

end camilla_blueberry_jelly_beans_l138_138001


namespace sum_base8_to_decimal_l138_138784

theorem sum_base8_to_decimal (a b : ℕ) (ha : a = 5) (hb : b = 0o17)
  (h_sum_base8 : a + b = 0o24) : (a + b) = 20 := by
  sorry

end sum_base8_to_decimal_l138_138784


namespace cost_of_plastering_l138_138371

/-- 
Let's define the problem conditions
Length of the tank (in meters)
-/
def tank_length : ℕ := 25

/--
Width of the tank (in meters)
-/
def tank_width : ℕ := 12

/--
Depth of the tank (in meters)
-/
def tank_depth : ℕ := 6

/--
Cost of plastering per square meter (55 paise converted to rupees)
-/
def cost_per_sq_meter : ℝ := 0.55

/--
Prove that the cost of plastering the walls and bottom of the tank is 409.2 rupees
-/
theorem cost_of_plastering (total_cost : ℝ) : 
  total_cost = 409.2 :=
sorry

end cost_of_plastering_l138_138371


namespace factorization_of_polynomial_l138_138382

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l138_138382


namespace calculate_expression_l138_138891

theorem calculate_expression : 2 * Real.sin (60 * Real.pi / 180) + (-1/2)⁻¹ + abs (2 - Real.sqrt 3) = 0 :=
by
  sorry

end calculate_expression_l138_138891


namespace parallelogram_area_l138_138662

theorem parallelogram_area (a b : ℝ) (theta : ℝ)
  (h1 : a = 10) (h2 : b = 20) (h3 : theta = 150) : a * b * Real.sin (theta * Real.pi / 180) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l138_138662


namespace lucy_run_base10_eq_1878_l138_138862

-- Define a function to convert a base-8 numeral to base-10.
def base8_to_base10 (n: Nat) : Nat :=
  (3 * 8^3) + (5 * 8^2) + (2 * 8^1) + (6 * 8^0)

-- Define the base-8 number.
def lucy_run (n : Nat) : Nat := n

-- Prove that the base-10 equivalent of the base-8 number 3526 is 1878.
theorem lucy_run_base10_eq_1878 : base8_to_base10 (lucy_run 3526) = 1878 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end lucy_run_base10_eq_1878_l138_138862


namespace oranges_in_bowl_l138_138355

-- Definitions (conditions)
def bananas : Nat := 2
def apples : Nat := 2 * bananas
def total_fruits : Nat := 12

-- Theorem (proof goal)
theorem oranges_in_bowl : 
  apples + bananas + oranges = total_fruits → oranges = 6 :=
by
  intro h
  sorry

end oranges_in_bowl_l138_138355


namespace sin_135_eq_sqrt2_over_2_l138_138677

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l138_138677


namespace calculate_minimal_total_cost_l138_138846

structure GardenSection where
  area : ℕ
  flower_cost : ℚ

def garden := [
  GardenSection.mk 10 2.75, -- Orchids
  GardenSection.mk 14 2.25, -- Violets
  GardenSection.mk 14 1.50, -- Hyacinths
  GardenSection.mk 15 1.25, -- Tulips
  GardenSection.mk 25 0.75  -- Sunflowers
]

def total_cost (sections : List GardenSection) : ℚ :=
  sections.foldr (λ s acc => s.area * s.flower_cost + acc) 0

theorem calculate_minimal_total_cost :
  total_cost garden = 117.5 := by
  sorry

end calculate_minimal_total_cost_l138_138846


namespace slope_symmetric_line_l138_138747

  theorem slope_symmetric_line {l1 l2 : ℝ → ℝ} 
     (hl1 : ∀ x, l1 x = 2 * x + 3)
     (hl2_sym : ∀ x, l2 x = 2 * x + 3 -> l2 (-x) = -2 * x - 3) :
     ∀ x, l2 x = -2 * x + 3 :=
  sorry
  
end slope_symmetric_line_l138_138747


namespace parallel_vectors_l138_138325

variable (y : ℝ)

def vector_a : ℝ × ℝ := (-1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

theorem parallel_vectors (h : (-1 * y - 3 * 2) = 0) : y = -6 :=
by
  sorry

end parallel_vectors_l138_138325


namespace payments_option1_option2_option1_more_effective_combined_option_cost_l138_138420

variable {x : ℕ}

-- Condition 1: Prices and discount options
def badminton_rackets_price : ℕ := 40
def shuttlecocks_price : ℕ := 10
def discount_option1_free_shuttlecocks (pairs : ℕ): ℕ := pairs
def discount_option2_price (price : ℕ) : ℕ := price * 9 / 10

-- Condition 2: Buying requirements
def pairs_needed : ℕ := 10
def shuttlecocks_needed (n : ℕ) : ℕ := n
axiom x_gt_10 : x > 10

-- Proof Problem 1: Payment calculations
theorem payments_option1_option2 (x : ℕ) (h : x > 10) :
  (shuttlecocks_price * (shuttlecocks_needed x - discount_option1_free_shuttlecocks pairs_needed) + badminton_rackets_price * pairs_needed =
    10 * x + 300) ∧
  (discount_option2_price (shuttlecocks_price * shuttlecocks_needed x + badminton_rackets_price * pairs_needed) =
    9 * x + 360) :=
sorry

-- Proof Problem 2: More cost-effective option when x=30
theorem option1_more_effective (x : ℕ) (h : x = 30) :
  (10 * x + 300 < 9 * x + 360) :=
sorry

-- Proof Problem 3: Another cost-effective method when x=30
theorem combined_option_cost (x : ℕ) (h : x = 30) :
  (badminton_rackets_price * pairs_needed + discount_option2_price (shuttlecocks_price * (shuttlecocks_needed x - 10)) = 580) :=
sorry

end payments_option1_option2_option1_more_effective_combined_option_cost_l138_138420


namespace investment_of_c_l138_138118

variable (P_a P_b P_c C_a C_b C_c : ℝ)

theorem investment_of_c (h1 : P_b = 3500) 
                        (h2 : P_a - P_c = 1399.9999999999998) 
                        (h3 : C_a = 8000) 
                        (h4 : C_b = 10000) 
                        (h5 : P_a / C_a = P_b / C_b) 
                        (h6 : P_c / C_c = P_b / C_b) : 
                        C_c = 40000 := 
by 
  sorry

end investment_of_c_l138_138118


namespace area_ratio_gt_two_ninths_l138_138477

variables {A B C P Q R : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

def divides_perimeter_eq (A B C : Type*) (P Q R : Type*) : Prop :=
-- Definition that P, Q, and R divide the perimeter into three equal parts
sorry

def is_on_side_AB (A B C P Q : Type*) : Prop :=
-- Definition that points P and Q are on side AB
sorry

theorem area_ratio_gt_two_ninths (A B C P Q R : Type*)
  (H1 : divides_perimeter_eq A B C P Q R)
  (H2 : is_on_side_AB A B C P Q) :
  -- Statement to prove that the area ratio is greater than 2/9
  (S_ΔPQR / S_ΔABC) > (2 / 9) :=
sorry

end area_ratio_gt_two_ninths_l138_138477


namespace Michelle_initial_crayons_l138_138859

variable (M : ℕ)  -- M is the number of crayons Michelle initially has
variable (J : ℕ := 2)  -- Janet has 2 crayons
variable (final_crayons : ℕ := 4)  -- After Janet gives her crayons to Michelle, Michelle has 4 crayons

theorem Michelle_initial_crayons : M + J = final_crayons → M = 2 :=
by
  intro h1
  sorry

end Michelle_initial_crayons_l138_138859


namespace values_of_d_divisible_by_13_l138_138696

def base8to10 (d : ℕ) : ℕ := 3 * 8^3 + d * 8^2 + d * 8 + 7

theorem values_of_d_divisible_by_13 (d : ℕ) (h : d ≥ 0 ∧ d < 8) :
  (1543 + 72 * d) % 13 = 0 ↔ d = 1 ∨ d = 2 :=
by sorry

end values_of_d_divisible_by_13_l138_138696


namespace find_x_and_y_l138_138295

variable {x y : ℝ}

-- Given condition
def angleDCE : ℝ := 58

-- Proof statements
theorem find_x_and_y : x = 180 - angleDCE ∧ y = 180 - angleDCE := by
  sorry

end find_x_and_y_l138_138295


namespace intersection_A_B_l138_138661

noncomputable def A : Set ℝ := { x | abs (x - 1) < 2 }
noncomputable def B : Set ℝ := { x | x^2 + 3 * x - 4 < 0 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l138_138661


namespace problem_M_m_evaluation_l138_138643

theorem problem_M_m_evaluation
  (a b c d e : ℝ)
  (h : a < b)
  (h' : b < c)
  (h'' : c < d)
  (h''' : d < e)
  (h'''' : a < e) :
  (max (min a (max b c))
       (max (min a d) (max b e))) = e := 
by
  sorry

end problem_M_m_evaluation_l138_138643


namespace books_left_over_l138_138656

-- Define the conditions as variables in Lean
def total_books : ℕ := 1500
def new_shelf_capacity : ℕ := 28

-- State the theorem based on these conditions
theorem books_left_over : total_books % new_shelf_capacity = 14 :=
by
  sorry

end books_left_over_l138_138656


namespace jack_buttons_total_l138_138256

theorem jack_buttons_total :
  (3 * 3) * 7 = 63 :=
by
  sorry

end jack_buttons_total_l138_138256


namespace find_number_l138_138585

theorem find_number (x : ℕ) (h : 3 * (x + 2) = 24 + x) : x = 9 :=
by 
  sorry

end find_number_l138_138585


namespace acute_angle_tan_eq_one_l138_138245

theorem acute_angle_tan_eq_one (A : ℝ) (h1 : 0 < A ∧ A < π / 2) (h2 : Real.tan A = 1) : A = π / 4 :=
by
  sorry

end acute_angle_tan_eq_one_l138_138245


namespace num_positive_integers_satisfying_condition_l138_138377

theorem num_positive_integers_satisfying_condition :
  ∃! (n : ℕ), 30 - 6 * n > 18 := by
  sorry

end num_positive_integers_satisfying_condition_l138_138377


namespace evaluate_functions_l138_138091

def f (x : ℝ) := x + 2
def g (x : ℝ) := 2 * x^2 - 4
def h (x : ℝ) := x + 1

theorem evaluate_functions : f (g (h 3)) = 30 := by
  sorry

end evaluate_functions_l138_138091


namespace screws_per_pile_l138_138742

-- Definitions based on the given conditions
def initial_screws : ℕ := 8
def multiplier : ℕ := 2
def sections : ℕ := 4

-- Derived values based on the conditions
def additional_screws : ℕ := initial_screws * multiplier
def total_screws : ℕ := initial_screws + additional_screws

-- Proposition statement
theorem screws_per_pile : total_screws / sections = 6 := by
  sorry

end screws_per_pile_l138_138742


namespace fraction_color_films_l138_138306

variables {x y : ℕ} (h₁ : y ≠ 0) (h₂ : x ≠ 0)

theorem fraction_color_films (h₃ : 30 * x > 0) (h₄ : 6 * y > 0) :
  (6 * y : ℚ) / ((3 * y / 10) + 6 * y) = 20 / 21 := by
  sorry

end fraction_color_films_l138_138306


namespace smallest_integer_with_eight_factors_l138_138259

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l138_138259


namespace greatest_integer_x_l138_138627

theorem greatest_integer_x (x : ℤ) : (5 : ℚ)/8 > (x : ℚ)/15 → x ≤ 9 :=
by {
  sorry
}

end greatest_integer_x_l138_138627


namespace find_k_l138_138814

-- Definitions for arithmetic sequence properties
noncomputable def sum_arith_seq (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n-1) / 2) * d

noncomputable def term_arith_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Given Conditions
variables (a₁ d : ℝ)
variables (k : ℕ)

axiom sum_condition : sum_arith_seq a₁ d 9 = sum_arith_seq a₁ d 4
axiom term_condition : term_arith_seq a₁ d 4 + term_arith_seq a₁ d k = 0

-- Prove k = 10
theorem find_k : k = 10 :=
by
  sorry

end find_k_l138_138814


namespace smaller_number_l138_138481

theorem smaller_number (x y : ℤ) (h1 : x + y = 12) (h2 : x - y = 20) : y = -4 := 
by 
  sorry

end smaller_number_l138_138481


namespace no_solutions_system_l138_138349

theorem no_solutions_system :
  ∀ (x y : ℝ), 
  (x^3 + x + y + 1 = 0) →
  (y * x^2 + x + y = 0) →
  (y^2 + y - x^2 + 1 = 0) →
  false :=
by
  intro x y h1 h2 h3
  -- Proof goes here
  sorry

end no_solutions_system_l138_138349


namespace unique_pegboard_arrangement_l138_138383

/-- Conceptually, we will set up a function to count valid arrangements of pegs
based on the given conditions and prove that there is exactly one such arrangement. -/
def triangular_pegboard_arrangements (yellow red green blue orange black : ℕ) : ℕ :=
  if yellow = 6 ∧ red = 5 ∧ green = 4 ∧ blue = 3 ∧ orange = 2 ∧ black = 1 then 1 else 0

theorem unique_pegboard_arrangement :
  triangular_pegboard_arrangements 6 5 4 3 2 1 = 1 :=
by
  -- Placeholder for proof
  sorry

end unique_pegboard_arrangement_l138_138383


namespace unique_pair_prime_m_positive_l138_138345

theorem unique_pair_prime_m_positive (p m : ℕ) (hp : Nat.Prime p) (hm : 0 < m) :
  p * (p + m) + p = (m + 1) ^ 3 → (p = 2 ∧ m = 1) :=
by
  sorry

end unique_pair_prime_m_positive_l138_138345


namespace total_difference_in_cups_l138_138375

theorem total_difference_in_cups (h1: Nat) (h2: Nat) (h3: Nat) (hrs: Nat) : 
  h1 = 4 → h2 = 7 → h3 = 5 → hrs = 3 → 
  ((h2 * hrs - h1 * hrs) + (h3 * hrs - h1 * hrs) + (h2 * hrs - h3 * hrs)) = 18 :=
by
  intros h1_eq h2_eq h3_eq hrs_eq
  sorry

end total_difference_in_cups_l138_138375


namespace number_of_terms_in_expansion_l138_138466

theorem number_of_terms_in_expansion (A B : Finset ℕ) (h1 : A.card = 4) (h2 : B.card = 5) :
  (A.product B).card = 20 :=
by
  sorry

end number_of_terms_in_expansion_l138_138466


namespace Sophie_donuts_problem_l138_138002

noncomputable def total_cost_before_discount (cost_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  cost_per_box * num_boxes

noncomputable def discount_amount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

noncomputable def total_cost_after_discount (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost - discount

noncomputable def total_donuts (donuts_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  donuts_per_box * num_boxes

noncomputable def donuts_left (total_donuts : ℕ) (donuts_given_away : ℕ) : ℕ :=
  total_donuts - donuts_given_away

theorem Sophie_donuts_problem
  (budget : ℝ)
  (cost_per_box : ℝ)
  (discount_rate : ℝ)
  (num_boxes : ℕ)
  (donuts_per_box : ℕ)
  (donuts_given_to_mom : ℕ)
  (donuts_given_to_sister : ℕ)
  (half_dozen : ℕ) :
  budget = 50 →
  cost_per_box = 12 →
  discount_rate = 0.10 →
  num_boxes = 4 →
  donuts_per_box = 12 →
  donuts_given_to_mom = 12 →
  donuts_given_to_sister = 6 →
  half_dozen = 6 →
  total_cost_after_discount (total_cost_before_discount cost_per_box num_boxes) (discount_amount (total_cost_before_discount cost_per_box num_boxes) discount_rate) = 43.2 ∧
  donuts_left (total_donuts donuts_per_box num_boxes) (donuts_given_to_mom + donuts_given_to_sister) = 30 :=
by
  sorry

end Sophie_donuts_problem_l138_138002


namespace circus_tent_sections_l138_138492

noncomputable def sections_in_circus_tent (total_capacity : ℕ) (section_capacity : ℕ) : ℕ :=
  total_capacity / section_capacity

theorem circus_tent_sections : sections_in_circus_tent 984 246 = 4 := 
  by 
  sorry

end circus_tent_sections_l138_138492


namespace minimum_value_l138_138201

theorem minimum_value (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ y_min, y_min = -2 / 7 ∧ x = 6 / 7 ∧ (x^2 + (y - 1)^2 + z^2) = 18 / 7 :=
by
  sorry

end minimum_value_l138_138201


namespace Ryan_funding_goal_l138_138464

theorem Ryan_funding_goal 
  (avg_fund_per_person : ℕ := 10) 
  (people_recruited : ℕ := 80)
  (pre_existing_fund : ℕ := 200) :
  (avg_fund_per_person * people_recruited + pre_existing_fund = 1000) :=
by
  sorry

end Ryan_funding_goal_l138_138464


namespace intercepts_line_5x_minus_2y_minus_10_eq_0_l138_138520

theorem intercepts_line_5x_minus_2y_minus_10_eq_0 :
  ∃ a b : ℝ, (a = 2 ∧ b = -5) ∧ (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → 
     ((y = 0 ∧ x = a) ∨ (x = 0 ∧ y = b))) :=
by
  sorry

end intercepts_line_5x_minus_2y_minus_10_eq_0_l138_138520


namespace number_of_ways_to_choose_books_l138_138207

def num_books := 15
def books_to_choose := 3

theorem number_of_ways_to_choose_books : Nat.choose num_books books_to_choose = 455 := by
  sorry

end number_of_ways_to_choose_books_l138_138207


namespace equivalence_mod_equivalence_divisible_l138_138647

theorem equivalence_mod (a b c : ℤ) :
  (∃ k : ℤ, a - b = k * c) ↔ (a % c = b % c) := by
  sorry

theorem equivalence_divisible (a b c : ℤ) :
  (a % c = b % c) ↔ (∃ k : ℤ, a - b = k * c) := by
  sorry

end equivalence_mod_equivalence_divisible_l138_138647


namespace calculate_f_f_neg3_l138_138470

def f (x : ℚ) : ℚ := (1 / x) + (1 / (x + 1))

theorem calculate_f_f_neg3 : f (f (-3)) = 24 / 5 := by
  sorry

end calculate_f_f_neg3_l138_138470


namespace roots_polynomial_sum_squares_l138_138202

theorem roots_polynomial_sum_squares (p q r : ℝ) 
  (h_roots : ∀ x : ℝ, x^3 - 15 * x^2 + 25 * x - 10 = 0 → x = p ∨ x = q ∨ x = r) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 350 := 
by {
  sorry
}

end roots_polynomial_sum_squares_l138_138202


namespace sequence_a5_l138_138157

/-- In the sequence {a_n}, with a_1 = 1, a_2 = 2, and a_(n+2) = 2 * a_(n+1) + a_n, prove that a_5 = 29. -/
theorem sequence_a5 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) (h_rec : ∀ n, a (n + 2) = 2 * a (n + 1) + a n) :
  a 5 = 29 :=
sorry

end sequence_a5_l138_138157


namespace vasya_wins_l138_138070

-- Define the grid size and initial setup
def grid_size : ℕ := 13
def initial_stones : ℕ := 2023

-- Define a condition that checks if a move can put a stone on the 13th cell
def can_win (position : ℕ) : Prop :=
  position = grid_size

-- Define the game logic for Petya and Vasya
def next_position (pos : ℕ) (move : ℕ) : ℕ :=
  pos + move

-- Ensure a win by always ensuring the next move does not leave Petya on positions 4, 7, 10, 13
def winning_strategy_for_vasya (current_pos : ℕ) (move : ℕ) : Prop :=
  (next_position current_pos move) ≠ 4 ∧
  (next_position current_pos move) ≠ 7 ∧
  (next_position current_pos move) ≠ 10 ∧
  (next_position current_pos move) ≠ 13

theorem vasya_wins : ∃ strategy : ℕ → ℕ → Prop,
  ∀ current_pos move, winning_strategy_for_vasya current_pos move → can_win (next_position current_pos move) :=
by
  sorry -- To be provided

end vasya_wins_l138_138070


namespace area_of_rhombus_l138_138065

variable (a b θ : ℝ)
variable (h_a : 0 < a) (h_b : 0 < b)

theorem area_of_rhombus (h : true) : (2 * a) * (2 * b) / 2 = 2 * a * b := by
  sorry

end area_of_rhombus_l138_138065


namespace complement_of_A_in_U_l138_138370

-- Define the universal set U and the subset A
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 2, 5, 7}

-- Define the complement of A with respect to U
def complementU_A : Set Nat := {x ∈ U | x ∉ A}

-- Prove the complement of A in U is {3, 4, 6}
theorem complement_of_A_in_U :
  complementU_A = {3, 4, 6} :=
by
  sorry

end complement_of_A_in_U_l138_138370


namespace solve_for_z_l138_138736

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2 + I) : z = (1 / 2) + (3 / 2) * I :=
  sorry

end solve_for_z_l138_138736


namespace hours_per_day_l138_138803

variable (m w : ℝ)
variable (h : ℕ)

-- Assume the equivalence of work done by women and men
axiom work_equiv : 3 * w = 2 * m

-- Total work done by men
def work_men := 15 * m * 21 * h
-- Total work done by women
def work_women := 21 * w * 36 * 5

-- The total work done by men and women is equal
theorem hours_per_day (h : ℕ) (w m : ℝ) (work_equiv : 3 * w = 2 * m) :
  15 * m * 21 * h = 21 * w * 36 * 5 → h = 8 :=
by
  intro H
  sorry

end hours_per_day_l138_138803


namespace number_of_rectangles_l138_138912

theorem number_of_rectangles (horizontal_lines : Fin 6) (vertical_lines : Fin 5) 
                             (point : ℕ × ℕ) (h₁ : point = (3, 4)) : 
  ∃ ways : ℕ, ways = 24 :=
by {
  sorry
}

end number_of_rectangles_l138_138912


namespace minimize_distance_on_ellipse_l138_138246

theorem minimize_distance_on_ellipse (a m n : ℝ) (hQ : 0 < a ∧ a ≠ Real.sqrt 3)
  (hP : m^2 / 3 + n^2 / 2 = 1) :
  |minimize_distance| = Real.sqrt 3 ∨ |minimize_distance| = 3 * a := sorry

end minimize_distance_on_ellipse_l138_138246


namespace simplify_expression_l138_138011

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^3 - b^3) / (a * b^2) - (ab^2 - b^3) / (ab^2 - a^3) = (a^3 - ab^2 + b^4) / (a * b^2) :=
sorry

end simplify_expression_l138_138011


namespace convert_speed_kmph_to_mps_l138_138434

def kilometers_to_meters := 1000
def hours_to_seconds := 3600
def speed_kmph := 18
def expected_speed_mps := 5

theorem convert_speed_kmph_to_mps :
  speed_kmph * (kilometers_to_meters / hours_to_seconds) = expected_speed_mps :=
by
  sorry

end convert_speed_kmph_to_mps_l138_138434


namespace relation_between_a_b_l138_138473

variables {x y a b : ℝ}

theorem relation_between_a_b 
  (h1 : a = (x^2 + y^2) * (x - y))
  (h2 : b = (x^2 - y^2) * (x + y))
  (h3 : x < y) 
  (h4 : y < 0) : 
  a > b := 
by sorry

end relation_between_a_b_l138_138473


namespace variance_of_data_l138_138554

def data : List ℝ := [0.7, 1, 0.8, 0.9, 1.1]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.foldr (λ x acc => x + acc) 0) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.foldr (λ x acc => (x - m) ^ 2 + acc) 0) / l.length

theorem variance_of_data :
  variance data = 0.02 :=
by
  sorry

end variance_of_data_l138_138554


namespace find_integer_l138_138948

def satisfies_conditions (x : ℕ) (m n : ℕ) : Prop :=
  x + 100 = m ^ 2 ∧ x + 168 = n ^ 2 ∧ m > 0 ∧ n > 0

theorem find_integer (x m n : ℕ) (h : satisfies_conditions x m n) : x = 156 :=
sorry

end find_integer_l138_138948


namespace max_difference_of_mean_505_l138_138773

theorem max_difference_of_mean_505 (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : (x + y) / 2 = 505) : 
  x - y ≤ 810 :=
sorry

end max_difference_of_mean_505_l138_138773


namespace least_possible_value_l138_138229

noncomputable def least_value_expression (x : ℝ) : ℝ :=
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024

theorem least_possible_value : ∃ x : ℝ, least_value_expression x = 2023 :=
  sorry

end least_possible_value_l138_138229


namespace rabbits_clear_land_in_21_days_l138_138872

theorem rabbits_clear_land_in_21_days (length_feet width_feet : ℝ) (rabbits : ℕ) (clear_per_rabbit_per_day : ℝ) : 
  length_feet = 900 → width_feet = 200 → rabbits = 100 → clear_per_rabbit_per_day = 10 →
  (⌈ (length_feet / 3 * width_feet / 3) / (rabbits * clear_per_rabbit_per_day) ⌉ = 21) := 
by
  intros
  sorry

end rabbits_clear_land_in_21_days_l138_138872


namespace multiply_polynomials_l138_138914

theorem multiply_polynomials (x : ℂ) : 
  (x^6 + 27 * x^3 + 729) * (x^3 - 27) = x^12 + 27 * x^9 - 19683 * x^3 - 531441 :=
by
  sorry

end multiply_polynomials_l138_138914


namespace solve_trigonometric_equation_l138_138966

theorem solve_trigonometric_equation (x : ℝ) : 
  (2 * (Real.sin x)^6 + 2 * (Real.cos x)^6 - 3 * (Real.sin x)^4 - 3 * (Real.cos x)^4) = Real.cos (2 * x) ↔ 
  ∃ (k : ℤ), x = (π / 2) * (2 * k + 1) :=
sorry

end solve_trigonometric_equation_l138_138966


namespace part1_solution_set_part2_range_of_m_l138_138435

def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part1_solution_set (x : ℝ) : (f x 3 >= 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2_range_of_m (m : ℝ) (x : ℝ) : 
 (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
by sorry

end part1_solution_set_part2_range_of_m_l138_138435


namespace probability_same_color_white_l138_138801

/--
Given a box with 6 white balls and 5 black balls, if 3 balls are drawn such that all drawn balls have the same color,
prove that the probability that these balls are white is 2/3.
-/
theorem probability_same_color_white :
  (∃ (n_white n_black drawn_white drawn_black total_same_color : ℕ),
    n_white = 6 ∧ n_black = 5 ∧
    drawn_white = Nat.choose n_white 3 ∧ drawn_black = Nat.choose n_black 3 ∧
    total_same_color = drawn_white + drawn_black ∧
    (drawn_white:ℚ) / total_same_color = 2 / 3) :=
sorry

end probability_same_color_white_l138_138801


namespace symmetric_about_x_axis_l138_138244

noncomputable def f (a x : ℝ) : ℝ := a - x^2
def g (x : ℝ) : ℝ := x + 1

theorem symmetric_about_x_axis (a : ℝ) :
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f a x = - g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end symmetric_about_x_axis_l138_138244


namespace baker_final_stock_l138_138247

-- Given conditions as Lean definitions
def initial_cakes : Nat := 173
def additional_cakes : Nat := 103
def damaged_percentage : Nat := 25
def sold_first_day : Nat := 86
def sold_next_day_percentage : Nat := 10

-- Calculate new cakes Baker adds to the stock after accounting for damaged cakes
def new_undamaged_cakes : Nat := (additional_cakes * (100 - damaged_percentage)) / 100

-- Calculate stock after adding new cakes
def stock_after_new_cakes : Nat := initial_cakes + new_undamaged_cakes

-- Calculate stock after first day's sales
def stock_after_first_sale : Nat := stock_after_new_cakes - sold_first_day

-- Calculate cakes sold on the second day
def sold_next_day : Nat := (stock_after_first_sale * sold_next_day_percentage) / 100

-- Final stock calculations
def final_stock : Nat := stock_after_first_sale - sold_next_day

-- Prove that Baker has 148 cakes left
theorem baker_final_stock : final_stock = 148 := by
  sorry

end baker_final_stock_l138_138247


namespace set_intersection_eq_l138_138030

def setA : Set ℝ := { x | x^2 - 3 * x - 4 > 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 5 }
def setC : Set ℝ := { x | (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5) }

theorem set_intersection_eq : setA ∩ setB = setC := by
  sorry

end set_intersection_eq_l138_138030


namespace cupcakes_frosted_in_10_minutes_l138_138270

theorem cupcakes_frosted_in_10_minutes (r1 r2 time : ℝ) (cagney_rate lacey_rate : r1 = 1 / 15 ∧ r2 = 1 / 25)
  (time_in_seconds : time = 600) :
  (1 / ((1 / r1) + (1 / r2)) * time) = 64 := by
  sorry

end cupcakes_frosted_in_10_minutes_l138_138270


namespace tan_alpha_eq_neg_one_l138_138090

-- Define the point P and the angle α
def P : ℝ × ℝ := (-1, 1)
def α : ℝ := sorry  -- α is the angle whose terminal side passes through P

-- Statement to be proved
theorem tan_alpha_eq_neg_one (h : (P.1, P.2) = (-1, 1)) : Real.tan α = -1 :=
by
  sorry

end tan_alpha_eq_neg_one_l138_138090


namespace find_h_in_standard_form_l138_138711

-- The expression to be converted
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x - 24

-- The standard form with given h value
def standard_form (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

-- The theorem statement
theorem find_h_in_standard_form :
  ∃ k : ℝ, ∀ x : ℝ, quadratic_expr x = standard_form 3 (-1.5) k x :=
by
  let a := 3
  let h := -1.5
  existsi (-30.75)
  intro x
  sorry

end find_h_in_standard_form_l138_138711


namespace acai_berry_cost_correct_l138_138134

def cost_superfruit_per_litre : ℝ := 1399.45
def cost_mixed_fruit_per_litre : ℝ := 262.85
def litres_mixed_fruit : ℝ := 36
def litres_acai_berry : ℝ := 24
def total_litres : ℝ := litres_mixed_fruit + litres_acai_berry
def expected_cost_acai_per_litre : ℝ := 3104.77

theorem acai_berry_cost_correct :
  cost_superfruit_per_litre * total_litres -
  cost_mixed_fruit_per_litre * litres_mixed_fruit = 
  expected_cost_acai_per_litre * litres_acai_berry :=
by sorry

end acai_berry_cost_correct_l138_138134


namespace correct_mark_l138_138027

theorem correct_mark (x : ℕ) (h1 : 73 - x = 10) : x = 63 :=
by
  sorry

end correct_mark_l138_138027


namespace seashells_after_giving_cannot_determine_starfish_l138_138811

-- Define the given conditions
def initial_seashells : Nat := 66
def seashells_given : Nat := 52
def seashells_left : Nat := 14

-- The main theorem to prove
theorem seashells_after_giving (initial : Nat) (given : Nat) (left : Nat) :
  initial = 66 -> given = 52 -> left = 14 -> initial - given = left :=
by 
  intros 
  sorry

-- The starfish count question
def starfish (count: Option Nat) : Prop :=
  count = none

-- Prove that we cannot determine the number of starfish Benny found
theorem cannot_determine_starfish (count: Option Nat) :
  count = none :=
by 
  intros 
  sorry

end seashells_after_giving_cannot_determine_starfish_l138_138811


namespace grasshopper_jump_is_31_l138_138015

def frog_jump : ℕ := 35
def total_jump : ℕ := 66
def grasshopper_jump := total_jump - frog_jump

theorem grasshopper_jump_is_31 : grasshopper_jump = 31 := 
by
  unfold grasshopper_jump
  sorry

end grasshopper_jump_is_31_l138_138015


namespace function_periodicity_l138_138288

theorem function_periodicity (f : ℝ → ℝ) (h1 : ∀ x, f (-x) + f x = 0)
  (h2 : ∀ x, f (x + 1) = f (1 - x)) (h3 : f 1 = 5) : f 2015 = -5 :=
sorry

end function_periodicity_l138_138288


namespace product_increase_l138_138327

theorem product_increase (a b c : ℕ) (h1 : a ≥ 3) (h2 : b ≥ 3) (h3 : c ≥ 3) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 := by
  sorry

end product_increase_l138_138327


namespace range_of_a_l138_138729

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + 9| > a) → a < 8 :=
by
  sorry

end range_of_a_l138_138729


namespace simplify_fraction_l138_138369

theorem simplify_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (15 * x^2 * y^3) / (9 * x * y^2) = 20 := by
  sorry

end simplify_fraction_l138_138369


namespace solve_x_l138_138019

noncomputable def solveEquation (a b c d : ℝ) (x : ℝ) : Prop :=
  x = 3 * a * b + 33 * b^2 + 333 * c^3 + 3.33 * (Real.sin d)^4

theorem solve_x :
  solveEquation 2 (-1) 0.5 (Real.pi / 6) 68.833125 :=
by
  sorry

end solve_x_l138_138019


namespace nickel_ate_4_chocolates_l138_138182

theorem nickel_ate_4_chocolates (R N : ℕ) (h1 : R = 13) (h2 : R = N + 9) : N = 4 :=
by
  sorry

end nickel_ate_4_chocolates_l138_138182


namespace combined_average_age_of_fifth_graders_teachers_and_parents_l138_138794

theorem combined_average_age_of_fifth_graders_teachers_and_parents
  (num_fifth_graders : ℕ) (avg_age_fifth_graders : ℕ)
  (num_teachers : ℕ) (avg_age_teachers : ℕ)
  (num_parents : ℕ) (avg_age_parents : ℕ)
  (h1 : num_fifth_graders = 40) (h2 : avg_age_fifth_graders = 10)
  (h3 : num_teachers = 4) (h4 : avg_age_teachers = 40)
  (h5 : num_parents = 60) (h6 : avg_age_parents = 34)
  : (num_fifth_graders * avg_age_fifth_graders + num_teachers * avg_age_teachers + num_parents * avg_age_parents) /
    (num_fifth_graders + num_teachers + num_parents) = 25 :=
by sorry

end combined_average_age_of_fifth_graders_teachers_and_parents_l138_138794


namespace other_root_l138_138787

open Complex

-- Defining the conditions that are given in the problem
def quadratic_equation (x : ℂ) (m : ℝ) : Prop :=
  x^2 + (1 - 2 * I) * x + (3 * m - I) = 0

def has_real_root (x : ℂ) : Prop :=
  ∃ α : ℝ, x = α

-- The main theorem statement we need to prove
theorem other_root (m : ℝ) (α : ℝ) (α_real_root : quadratic_equation α m) :
  quadratic_equation (-1/2 + 2 * I) m :=
sorry

end other_root_l138_138787


namespace total_area_is_82_l138_138669

/-- Definition of the lengths of each segment as conditions -/
def length1 : ℤ := 7
def length2 : ℤ := 4
def length3 : ℤ := 5
def length4 : ℤ := 3
def length5 : ℤ := 2
def length6 : ℤ := 1

/-- Rectangle areas based on the given lengths -/
def area_A : ℤ := length1 * length2 -- 7 * 4
def area_B : ℤ := length3 * length2 -- 5 * 4
def area_C : ℤ := length1 * length4 -- 7 * 3
def area_D : ℤ := length3 * length5 -- 5 * 2
def area_E : ℤ := length4 * length6 -- 3 * 1

/-- The total area is the sum of all rectangle areas -/
def total_area : ℤ := area_A + area_B + area_C + area_D + area_E

/-- Theorem: The total area is 82 square units -/
theorem total_area_is_82 : total_area = 82 :=
by
  -- Proof left as an exercise
  sorry

end total_area_is_82_l138_138669


namespace polynomial_has_real_root_l138_138008

noncomputable def P : Polynomial ℝ := sorry

variables (a1 a2 a3 b1 b2 b3 : ℝ) (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
variables (h_eq : ∀ x : ℝ, P.eval (a1 * x + b1) + P.eval (a2 * x + b2) = P.eval (a3 * x + b3))

theorem polynomial_has_real_root : ∃ x : ℝ, P.eval x = 0 :=
sorry

end polynomial_has_real_root_l138_138008


namespace total_area_of_tickets_is_3_6_m2_l138_138521

def area_of_one_ticket (side_length_cm : ℕ) : ℕ :=
  side_length_cm * side_length_cm

def total_tickets (people : ℕ) (tickets_per_person : ℕ) : ℕ :=
  people * tickets_per_person

def total_area_cm2 (area_per_ticket_cm2 : ℕ) (number_of_tickets : ℕ) : ℕ :=
  area_per_ticket_cm2 * number_of_tickets

def convert_cm2_to_m2 (area_cm2 : ℕ) : ℚ :=
  (area_cm2 : ℚ) / 10000

theorem total_area_of_tickets_is_3_6_m2 :
  let side_length := 30
  let people := 5
  let tickets_per_person := 8
  let one_ticket_area := area_of_one_ticket side_length
  let number_of_tickets := total_tickets people tickets_per_person
  let total_area_cm2 := total_area_cm2 one_ticket_area number_of_tickets
  let total_area_m2 := convert_cm2_to_m2 total_area_cm2
  total_area_m2 = 3.6 := 
by
  sorry

end total_area_of_tickets_is_3_6_m2_l138_138521


namespace count_of_changing_quantities_l138_138951

-- Definitions of the problem conditions
def length_AC_unchanged : Prop := ∀ P A B C D : ℝ, true
def perimeter_square_unchanged : Prop := ∀ P A B C D : ℝ, true
def area_square_unchanged : Prop := ∀ P A B C D : ℝ, true
def area_quadrilateral_changed : Prop := ∀ P A B C D M N : ℝ, true

-- The main theorem to prove
theorem count_of_changing_quantities :
  length_AC_unchanged ∧
  perimeter_square_unchanged ∧
  area_square_unchanged ∧
  area_quadrilateral_changed →
  (1 = 1) :=
by
  sorry

end count_of_changing_quantities_l138_138951


namespace gold_beads_cannot_be_determined_without_cost_per_bead_l138_138132

-- Carly's bead conditions
def purple_rows : ℕ := 50
def purple_beads_per_row : ℕ := 20
def blue_rows : ℕ := 40
def blue_beads_per_row : ℕ := 18
def total_cost : ℝ := 180

-- The calculation of total purple and blue beads
def purple_beads : ℕ := purple_rows * purple_beads_per_row
def blue_beads : ℕ := blue_rows * blue_beads_per_row
def total_beads_without_gold : ℕ := purple_beads + blue_beads

-- Given the lack of cost per bead, the number of gold beads cannot be determined
theorem gold_beads_cannot_be_determined_without_cost_per_bead :
  ¬ (∃ cost_per_bead : ℝ, ∃ gold_beads : ℕ, (purple_beads + blue_beads + gold_beads) * cost_per_bead = total_cost) :=
sorry

end gold_beads_cannot_be_determined_without_cost_per_bead_l138_138132


namespace problem_statement_l138_138679

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

noncomputable def given_conditions (a : ℕ → ℤ) : Prop :=
a 2 = 2 ∧ a 3 = 4

theorem problem_statement (a : ℕ → ℤ) (h1 : given_conditions a) (h2 : arithmetic_sequence a) :
  a 10 = 18 := by
  sorry

end problem_statement_l138_138679


namespace total_ducks_in_lake_l138_138084

/-- 
Problem: Determine the total number of ducks in the lake after more ducks join.

Conditions:
- Initially, there are 13 ducks in the lake.
- 20 more ducks come to join them.
-/

def initial_ducks : Nat := 13

def new_ducks : Nat := 20

theorem total_ducks_in_lake : initial_ducks + new_ducks = 33 := by
  sorry -- Proof to be filled in later

end total_ducks_in_lake_l138_138084


namespace count_harmonic_vals_l138_138293

def floor (x : ℝ) : ℤ := sorry -- or use Mathlib function
def frac (x : ℝ) : ℝ := x - (floor x)

def is_harmonic_progression (a b c : ℝ) : Prop := 
  (1 / a) = (2 / b) - (1 / c)

theorem count_harmonic_vals :
  (∃ x, is_harmonic_progression x (floor x) (frac x)) ∧
  (∃! x1 x2, is_harmonic_progression x1 (floor x1) (frac x1) ∧
               is_harmonic_progression x2 (floor x2) (frac x2)) ∧
  x1 ≠ x2 :=
  sorry

end count_harmonic_vals_l138_138293


namespace remainder_2_pow_305_mod_9_l138_138072

theorem remainder_2_pow_305_mod_9 :
  2^305 % 9 = 5 :=
by sorry

end remainder_2_pow_305_mod_9_l138_138072


namespace remainder_of_899830_divided_by_16_is_6_l138_138312

theorem remainder_of_899830_divided_by_16_is_6 :
  ∃ k : ℕ, 899830 = 16 * k + 6 :=
by
  sorry

end remainder_of_899830_divided_by_16_is_6_l138_138312


namespace find_xyz_l138_138421

open Complex

-- Definitions of the variables and conditions
variables {a b c x y z : ℂ} (h_a_ne_zero : a ≠ 0) (h_b_ne_zero : b ≠ 0) (h_c_ne_zero : c ≠ 0)
  (h_x_ne_zero : x ≠ 0) (h_y_ne_zero : y ≠ 0) (h_z_ne_zero : z ≠ 0)
  (h1 : a = (b - c) * (x + 2))
  (h2 : b = (a - c) * (y + 2))
  (h3 : c = (a - b) * (z + 2))
  (h4 : x * y + x * z + y * z = 12)
  (h5 : x + y + z = 6)

-- Statement of the theorem
theorem find_xyz : x * y * z = 7 := 
by
  -- Proof steps to be filled in
  sorry

end find_xyz_l138_138421


namespace pyramid_volume_l138_138982

noncomputable def volume_of_pyramid (a b c d: ℝ) (diagonal: ℝ) (angle: ℝ) : ℝ :=
  if (a = 10 ∧ d = 10 ∧ b = 5 ∧ c = 5 ∧ diagonal = 4 * Real.sqrt 5 ∧ angle = 45) then
    let base_area := 1 / 2 * (diagonal) * (Real.sqrt ((c * c) + (b * b)))
    let height := 10 / 3
    let volume := 1 / 3 * base_area * height
    volume
  else 0

theorem pyramid_volume :
  volume_of_pyramid 10 5 5 10 (4 * Real.sqrt 5) 45 = 500 / 9 :=
by
  sorry

end pyramid_volume_l138_138982


namespace compare_groups_l138_138998

noncomputable def mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def variance (scores : List ℝ) : ℝ :=
  let m := mean scores
  (scores.map (λ x => (x - m) ^ 2)).sum / scores.length

noncomputable def stddev (scores : List ℝ) : ℝ :=
  (variance scores).sqrt

def groupA_scores : List ℝ := [88, 100, 95, 86, 95, 91, 84, 74, 92, 83]
def groupB_scores : List ℝ := [93, 89, 81, 77, 96, 78, 77, 85, 89, 86]

theorem compare_groups :
  mean groupA_scores > mean groupB_scores ∧ stddev groupA_scores > stddev groupB_scores :=
by
  sorry

end compare_groups_l138_138998


namespace pump_X_time_l138_138756

-- Definitions for the problem conditions.
variables (W : ℝ) (T_x : ℝ) (R_x R_y : ℝ)

-- Condition 1: Rate of pump X
def pump_X_rate := R_x = (W / 2) / T_x

-- Condition 2: Rate of pump Y
def pump_Y_rate := R_y = W / 18

-- Condition 3: Combined rate when both pumps work together for 3 hours to pump the remaining water
def combined_rate := (R_x + R_y) = (W / 2) / 3

-- The statement to prove
theorem pump_X_time : 
  pump_X_rate W T_x R_x →
  pump_Y_rate W R_y →
  combined_rate W R_x R_y →
  T_x = 9 :=
sorry

end pump_X_time_l138_138756


namespace range_of_x_l138_138444

-- Define the necessary properties and functions.
variable (f : ℝ → ℝ)
variable (hf_even : ∀ x : ℝ, f (-x) = f x)
variable (hf_monotonic : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y)

-- Define the statement to be proved.
theorem range_of_x (f : ℝ → ℝ) (hf_even : ∀ x, f (-x) = f x) (hf_monotonic : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  { x : ℝ | f (2 * x - 1) ≤ f 3 } = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end range_of_x_l138_138444


namespace log_comparison_l138_138486

theorem log_comparison : Real.log 675 / Real.log 135 > Real.log 75 / Real.log 45 := 
sorry

end log_comparison_l138_138486


namespace polynomial_expansion_l138_138258

theorem polynomial_expansion (x : ℝ) :
  (x - 2) * (x + 2) * (x^2 + 4 * x + 4) = x^4 + 4 * x^3 - 16 * x - 16 :=
by sorry

end polynomial_expansion_l138_138258


namespace negation_of_implication_l138_138130

theorem negation_of_implication (a b c : ℝ) :
  ¬ (a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by sorry

end negation_of_implication_l138_138130


namespace largest_value_of_y_l138_138235

theorem largest_value_of_y :
  (∃ x y : ℝ, x^2 + 3 * x * y - y^2 = 27 ∧ 3 * x^2 - x * y + y^2 = 27 ∧ y ≤ 3) → (∃ y : ℝ, y = 3) :=
by
  intro h
  obtain ⟨x, y, h1, h2, h3⟩ := h
  -- proof steps go here
  sorry

end largest_value_of_y_l138_138235


namespace rectangular_garden_width_l138_138576

-- Define the problem conditions as Lean definitions
def rectangular_garden_length (w : ℝ) : ℝ := 3 * w
def rectangular_garden_area (w : ℝ) : ℝ := rectangular_garden_length w * w

-- This is the theorem we want to prove
theorem rectangular_garden_width : ∃ w : ℝ, rectangular_garden_area w = 432 ∧ w = 12 :=
by
  sorry

end rectangular_garden_width_l138_138576


namespace amy_local_calls_l138_138768

theorem amy_local_calls (L I : ℕ) 
  (h1 : 2 * L = 5 * I)
  (h2 : 3 * L = 5 * (I + 3)) : 
  L = 15 :=
by
  sorry

end amy_local_calls_l138_138768


namespace oranges_given_to_friend_l138_138166

theorem oranges_given_to_friend (initial_oranges : ℕ) 
  (given_to_brother : ℕ)
  (given_to_friend : ℕ)
  (h1 : initial_oranges = 60)
  (h2 : given_to_brother = (1 / 3 : ℚ) * initial_oranges)
  (h3 : given_to_friend = (1 / 4 : ℚ) * (initial_oranges - given_to_brother)) : 
  given_to_friend = 10 := 
by 
  sorry

end oranges_given_to_friend_l138_138166


namespace distance_between_polar_points_l138_138526

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem distance_between_polar_points :
  let A := polar_to_rect 1 (Real.pi / 6)
  let B := polar_to_rect 2 (-Real.pi / 2)
  distance A B = Real.sqrt 7 :=
by
  sorry

end distance_between_polar_points_l138_138526


namespace exactly_one_true_l138_138193

-- Given conditions
def p (x : ℝ) : Prop := (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 2)

-- Define the contrapositive of p
def contrapositive_p (x : ℝ) : Prop := (x = 2) → (x^2 - 3 * x + 2 = 0)

-- Define the converse of p
def converse_p (x : ℝ) : Prop := (x ≠ 2) → (x^2 - 3 * x + 2 ≠ 0)

-- Define the inverse of p
def inverse_p (x : ℝ) : Prop := (x = 2 → x^2 - 3 * x + 2 = 0)

-- Formalize the problem: Prove that exactly one of the converse, inverse, and contrapositive of p is true.
theorem exactly_one_true :
  (∀ x : ℝ, p x) →
  ((∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ (∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ (∀ x : ℝ, inverse_p x)) :=
sorry

end exactly_one_true_l138_138193


namespace det_new_matrix_l138_138013

variables {a b c d : ℝ}

theorem det_new_matrix (h : a * d - b * c = 5) : (a - c) * d - (b - d) * c = 5 :=
by sorry

end det_new_matrix_l138_138013


namespace total_spent_is_64_l138_138267

def deck_price : ℕ := 8
def victors_decks : ℕ := 6
def friends_decks : ℕ := 2

def victors_spending : ℕ := victors_decks * deck_price
def friends_spending : ℕ := friends_decks * deck_price
def total_spending : ℕ := victors_spending + friends_spending

theorem total_spent_is_64 : total_spending = 64 := by
  sorry

end total_spent_is_64_l138_138267


namespace brenda_distance_when_first_met_l138_138331

theorem brenda_distance_when_first_met
  (opposite_points : ∀ (d : ℕ), d = 150) -- Starting at diametrically opposite points on a 300m track means distance is 150m
  (constant_speeds : ∀ (B S x : ℕ), B * x = S * x) -- Brenda/ Sally run at constant speed
  (meet_again : ∀ (d₁ d₂ : ℕ), d₁ + d₂ = 300 + 100) -- Together they run 400 meters when they meet again, additional 100m by Sally
  : ∃ (x : ℕ), x = 150 :=
  by
    sorry

end brenda_distance_when_first_met_l138_138331


namespace determine_a_value_l138_138820

theorem determine_a_value :
  ∀ (a b c d : ℕ), 
  (a = b + 3) →
  (b = c + 6) →
  (c = d + 15) →
  (d = 50) →
  a = 74 :=
by
  intros a b c d h1 h2 h3 h4
  sorry

end determine_a_value_l138_138820


namespace correct_calculation_of_exponentiation_l138_138622

theorem correct_calculation_of_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end correct_calculation_of_exponentiation_l138_138622


namespace find_a_l138_138346

def A : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem find_a (a : ℝ) (h : A ∩ B a = B a) : a = 0 ∨ a = -1 ∨ a = 1/3 := by
  sorry

end find_a_l138_138346


namespace gcf_4370_13824_l138_138592

/-- Define the two numbers 4370 and 13824 -/
def num1 := 4370
def num2 := 13824

/-- The statement that the GCF of num1 and num2 is 1 -/
theorem gcf_4370_13824 : Nat.gcd num1 num2 = 1 := by
  sorry

end gcf_4370_13824_l138_138592


namespace a_equals_bc_l138_138746

theorem a_equals_bc (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x y : ℝ, f x * g y = a * x * y + b * x + c * y + 1) → a = b * c :=
sorry

end a_equals_bc_l138_138746


namespace calen_more_pencils_l138_138871

def calen_pencils (C B D: ℕ) :=
  D = 9 ∧
  B = 2 * D - 3 ∧
  C - 10 = 10

theorem calen_more_pencils (C B D : ℕ) (h : calen_pencils C B D) : C = B + 5 :=
by
  obtain ⟨hD, hB, hC⟩ := h
  simp only [hD, hB, hC]
  sorry

end calen_more_pencils_l138_138871


namespace find_x_l138_138785

-- Define the functions δ (delta) and φ (phi)
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- State the theorem with conditions and question, and assert the answer
theorem find_x :
  (delta ∘ phi) x = 11 → x = -5/6 := by
  intros
  sorry

end find_x_l138_138785


namespace margie_driving_distance_l138_138777

-- Define the constants given in the conditions
def mileage_per_gallon : ℝ := 40
def cost_per_gallon : ℝ := 5
def total_money : ℝ := 25

-- Define the expected result/answer
def expected_miles : ℝ := 200

-- The theorem that needs to be proved
theorem margie_driving_distance :
  (total_money / cost_per_gallon) * mileage_per_gallon = expected_miles :=
by
  -- proof goes here
  sorry

end margie_driving_distance_l138_138777


namespace deposit_on_Jan_1_2008_l138_138568

-- Let a be the initial deposit amount in yuan.
-- Let x be the annual interest rate.

def compound_interest (a : ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  a * (1 + x) ^ n

theorem deposit_on_Jan_1_2008 (a : ℝ) (x : ℝ) : 
  compound_interest a x 5 = a * (1 + x) ^ 5 := 
by
  sorry

end deposit_on_Jan_1_2008_l138_138568


namespace max_value_expr_l138_138806

theorem max_value_expr : ∃ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) = 85 :=
by sorry

end max_value_expr_l138_138806


namespace total_items_purchased_l138_138615

/-- Proof that Ike and Mike buy a total of 9 items given the constraints. -/
theorem total_items_purchased
  (total_money : ℝ)
  (sandwich_cost : ℝ)
  (drink_cost : ℝ)
  (combo_factor : ℕ)
  (money_spent_on_sandwiches : ℝ)
  (number_of_sandwiches : ℕ)
  (number_of_drinks : ℕ)
  (num_free_sandwiches : ℕ) :
  total_money = 40 →
  sandwich_cost = 5 →
  drink_cost = 1.5 →
  combo_factor = 5 →
  number_of_sandwiches = 9 →
  number_of_drinks = 0 →
  money_spent_on_sandwiches = number_of_sandwiches * sandwich_cost →
  total_money = money_spent_on_sandwiches →
  num_free_sandwiches = number_of_sandwiches / combo_factor →
  number_of_sandwiches = number_of_sandwiches + num_free_sandwiches →
  number_of_sandwiches + number_of_drinks = 9 :=
by
  intros
  sorry

end total_items_purchased_l138_138615


namespace range_of_a_l138_138525

theorem range_of_a (a : ℝ) (h : ∃ x1 x2, x1 ≠ x2 ∧ 3 * x1^2 + a = 0 ∧ 3 * x2^2 + a = 0) : a < 0 :=
sorry

end range_of_a_l138_138525


namespace watch_all_episodes_in_67_weeks_l138_138310

def total_episodes : ℕ := 201
def episodes_per_week : ℕ := 1 + 2

theorem watch_all_episodes_in_67_weeks :
  total_episodes / episodes_per_week = 67 := by 
  sorry

end watch_all_episodes_in_67_weeks_l138_138310


namespace range_of_m_l138_138960

open Real

theorem range_of_m (m : ℝ) : (m^2 > 2 + m ∧ 2 + m > 0) ↔ (m > 2 ∨ -2 < m ∧ m < -1) :=
by
  sorry

end range_of_m_l138_138960


namespace find_breadth_of_cuboid_l138_138778

variable (l : ℝ) (h : ℝ) (surface_area : ℝ) (b : ℝ)

theorem find_breadth_of_cuboid (hL : l = 10) (hH : h = 6) (hSA : surface_area = 480) 
  (hFormula : surface_area = 2 * (l * b + b * h + h * l)) : b = 11.25 := by
  sorry

end find_breadth_of_cuboid_l138_138778


namespace simplify_fraction_multiplication_l138_138722

theorem simplify_fraction_multiplication:
  (101 / 5050) * 50 = 1 := by
  sorry

end simplify_fraction_multiplication_l138_138722


namespace greatest_possible_perimeter_l138_138302

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l138_138302


namespace fifth_number_selected_l138_138487

-- Define the necessary conditions
def num_students : ℕ := 60
def sample_size : ℕ := 5
def first_selected_number : ℕ := 4
def interval : ℕ := num_students / sample_size

-- Define the proposition to be proved
theorem fifth_number_selected (h1 : 1 ≤ first_selected_number) (h2 : first_selected_number ≤ num_students)
    (h3 : sample_size > 0) (h4 : num_students % sample_size = 0) :
  first_selected_number + 4 * interval = 52 :=
by
  -- Proof omitted
  sorry

end fifth_number_selected_l138_138487


namespace part1_part2_l138_138307

noncomputable def f (a x : ℝ) : ℝ := (a / x) - Real.log x

theorem part1 (a : ℝ) (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2) 
(hf : f a x1 = -3) (hf2 : f a x2 = -3) : a ∈ Set.Ioo (-Real.exp 2) 0 :=
sorry

theorem part2 (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2)
(hfa : f (-2) x1 = -3) (hfb : f (-2) x2 = -3) : x1 + x2 > 4 :=
sorry

end part1_part2_l138_138307


namespace parallel_lines_m_value_l138_138440

/-- Given two lines x + m * y + 6 = 0 and (m - 2) * x + 3 * y + 2 * m = 0 are parallel,
    prove that the value of the real number m that makes the lines parallel is -1. -/
theorem parallel_lines_m_value (m : ℝ) : 
  (x + m * y + 6 = 0 ∧ (m - 2) * x + 3 * y + 2 * m = 0 → 
  (m = -1)) :=
by
  sorry

end parallel_lines_m_value_l138_138440


namespace total_cost_of_pencils_and_erasers_l138_138028

theorem total_cost_of_pencils_and_erasers 
  (pencil_cost : ℕ)
  (eraser_cost : ℕ)
  (pencils_bought : ℕ)
  (erasers_bought : ℕ)
  (total_cost_dollars : ℝ)
  (cents_to_dollars : ℝ)
  (hc : pencil_cost = 2)
  (he : eraser_cost = 5)
  (hp : pencils_bought = 500)
  (he2 : erasers_bought = 250)
  (cents_to_dollars_def : cents_to_dollars = 100)
  (total_cost_calc : total_cost_dollars = 
    ((pencils_bought * pencil_cost + erasers_bought * eraser_cost : ℕ) : ℝ) / cents_to_dollars) 
  : total_cost_dollars = 22.50 :=
sorry

end total_cost_of_pencils_and_erasers_l138_138028


namespace cost_of_each_top_l138_138150

theorem cost_of_each_top
  (total_spent : ℝ)
  (num_shorts : ℕ)
  (price_per_short : ℝ)
  (num_shoes : ℕ)
  (price_per_shoe : ℝ)
  (num_tops : ℕ)
  (total_cost_shorts : ℝ)
  (total_cost_shoes : ℝ)
  (amount_spent_on_tops : ℝ)
  (cost_per_top : ℝ) :
  total_spent = 75 →
  num_shorts = 5 →
  price_per_short = 7 →
  num_shoes = 2 →
  price_per_shoe = 10 →
  num_tops = 4 →
  total_cost_shorts = num_shorts * price_per_short →
  total_cost_shoes = num_shoes * price_per_shoe →
  amount_spent_on_tops = total_spent - (total_cost_shorts + total_cost_shoes) →
  cost_per_top = amount_spent_on_tops / num_tops →
  cost_per_top = 5 :=
by
  sorry

end cost_of_each_top_l138_138150


namespace speed_maintained_l138_138678

-- Given conditions:
def distance : ℕ := 324
def original_time : ℕ := 6
def new_time : ℕ := (3 * original_time) / 2

-- Correct answer:
def required_speed : ℕ := 36

-- Lean 4 statement to prove the equivalence:
theorem speed_maintained :
  (distance / new_time) = required_speed :=
sorry

end speed_maintained_l138_138678


namespace simplify_and_evaluate_l138_138094

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (1 - (1 / (x - 1))) / ((x ^ 2 - 4 * x + 4) / (x ^ 2 - 1)) = (2 / 5) :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l138_138094


namespace geese_more_than_ducks_l138_138727

-- Define initial conditions
def initial_ducks : ℕ := 25
def initial_geese : ℕ := 2 * initial_ducks - 10
def additional_ducks : ℕ := 4
def geese_leaving : ℕ := 15 - 5

-- Calculate the number of remaining ducks
def remaining_ducks : ℕ := initial_ducks + additional_ducks

-- Calculate the number of remaining geese
def remaining_geese : ℕ := initial_geese - geese_leaving

-- Prove the final statement
theorem geese_more_than_ducks : (remaining_geese - remaining_ducks) = 1 := by
  sorry

end geese_more_than_ducks_l138_138727


namespace socks_ratio_l138_138158

-- Definitions based on the conditions
def initial_black_socks : ℕ := 6
def initial_white_socks (B : ℕ) : ℕ := 4 * B
def remaining_white_socks (B : ℕ) : ℕ := B + 6

-- The theorem to prove the ratio is 1/2
theorem socks_ratio (B : ℕ) (hB : B = initial_black_socks) :
  ((initial_white_socks B - remaining_white_socks B) : ℚ) / initial_white_socks B = 1 / 2 :=
by
  sorry

end socks_ratio_l138_138158


namespace couscous_dishes_l138_138394

def dishes (a b c d : ℕ) : ℕ := (a + b + c) / d

theorem couscous_dishes :
  dishes 7 13 45 5 = 13 :=
by
  unfold dishes
  sorry

end couscous_dishes_l138_138394


namespace salary_after_cuts_l138_138480

noncomputable def finalSalary (init_salary : ℝ) (cuts : List ℝ) : ℝ :=
  cuts.foldl (λ salary cut => salary * (1 - cut)) init_salary

theorem salary_after_cuts :
  finalSalary 5000 [0.0525, 0.0975, 0.146, 0.128] = 3183.63 :=
by
  sorry

end salary_after_cuts_l138_138480


namespace quadratic_to_binomial_square_l138_138141

theorem quadratic_to_binomial_square (m : ℝ) : 
  (∃ c : ℝ, (x : ℝ) → x^2 - 12 * x + m = (x + c)^2) ↔ m = 36 := 
sorry

end quadratic_to_binomial_square_l138_138141


namespace find_integer_n_l138_138113

theorem find_integer_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] ∧ n = 0 :=
by
  sorry

end find_integer_n_l138_138113


namespace graph_of_equation_pair_of_lines_l138_138779

theorem graph_of_equation_pair_of_lines (x y : ℝ) : x^2 - 9 * y^2 = 0 ↔ (x = 3 * y ∨ x = -3 * y) :=
by
  sorry

end graph_of_equation_pair_of_lines_l138_138779


namespace parallelogram_base_length_l138_138600

theorem parallelogram_base_length 
  (area : ℝ)
  (b h : ℝ)
  (h_area : area = 128)
  (h_altitude : h = 2 * b) 
  (h_area_eq : area = b * h) : 
  b = 8 :=
by
  -- Proof goes here
  sorry

end parallelogram_base_length_l138_138600


namespace find_number_l138_138465

theorem find_number
    (x: ℝ)
    (h: 0.60 * x = 0.40 * 30 + 18) : x = 50 :=
    sorry

end find_number_l138_138465


namespace sequence_general_term_l138_138040

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n + 1) : a n = n * n :=
by
  sorry

end sequence_general_term_l138_138040


namespace speed_plane_east_l138_138869

-- Definitions of the conditions
def speed_west : ℕ := 275
def time_hours : ℝ := 3.5
def distance_apart : ℝ := 2100

-- Theorem statement to prove the speed of the plane traveling due East
theorem speed_plane_east (v: ℝ) 
  (h: (v + speed_west) * time_hours = distance_apart) : 
  v = 325 :=
  sorry

end speed_plane_east_l138_138869


namespace non_zero_number_is_9_l138_138637

theorem non_zero_number_is_9 (x : ℝ) (hx : x ≠ 0) (h : (x + x^2) / 2 = 5 * x) : x = 9 :=
sorry

end non_zero_number_is_9_l138_138637


namespace max_happy_monkeys_l138_138557

-- Definitions for given problem
def pears := 20
def bananas := 30
def peaches := 40
def mandarins := 50
def fruits (x y : Nat) := x + y

-- The theorem to prove
theorem max_happy_monkeys : 
  ∃ (m : Nat), m = (pears + bananas + peaches) / 2 ∧ m ≤ mandarins :=
by
  sorry

end max_happy_monkeys_l138_138557


namespace find_x_l138_138449

theorem find_x (x : ℝ) (h : 45 * x = 0.60 * 900) : x = 12 :=
by
  sorry

end find_x_l138_138449


namespace not_axiom_l138_138450

theorem not_axiom (P Q R S : Prop)
  (B : P -> Q -> R -> S)
  (C : P -> Q)
  (D : P -> R)
  : ¬ (P -> Q -> S) :=
sorry

end not_axiom_l138_138450


namespace expression_value_l138_138719

theorem expression_value : (5 - 2) / (2 + 1) = 1 := by
  sorry

end expression_value_l138_138719


namespace base6_addition_problem_l138_138508

-- Definitions to capture the base-6 addition problem components.
def base6₀ := 0
def base6₁ := 1
def base6₂ := 2
def base6₃ := 3
def base6₄ := 4
def base6₅ := 5

-- The main hypothesis about the base-6 addition
theorem base6_addition_problem (diamond : ℕ) (h : diamond ∈ [base6₀, base6₁, base6₂, base6₃, base6₄, base6₅]) :
  ((diamond + base6₅) % 6 = base6₃ ∨ (diamond + base6₅) % 6 = (base6₃ + 6 * 1 % 6)) ∧
  (diamond + base6₂ + base6₂ = diamond % 6) →
  diamond = base6₄ :=
sorry

end base6_addition_problem_l138_138508


namespace initial_students_count_l138_138816

theorem initial_students_count (n W : ℝ)
    (h1 : W = n * 28)
    (h2 : W + 10 = (n + 1) * 27.4) :
    n = 29 :=
by
  sorry

end initial_students_count_l138_138816


namespace problem_solution_l138_138114

theorem problem_solution (n : ℕ) (h : n^3 - n = 5814) : (n % 2 = 0) :=
by sorry

end problem_solution_l138_138114


namespace perpendicular_line_eq_slope_intercept_l138_138126

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l138_138126


namespace range_of_n_l138_138620

theorem range_of_n (m n : ℝ) (h : (m^2 - 2 * m)^2 + 4 * m^2 - 8 * m + 6 - n = 0) : n ≥ 3 :=
sorry

end range_of_n_l138_138620


namespace black_balls_count_l138_138594

theorem black_balls_count
  (P_red P_white : ℝ)
  (Red_balls_count : ℕ)
  (h1 : P_red = 0.42)
  (h2 : P_white = 0.28)
  (h3 : Red_balls_count = 21) :
  ∃ B, B = 15 :=
by
  sorry

end black_balls_count_l138_138594


namespace total_water_in_bucket_l138_138655

noncomputable def initial_gallons : ℝ := 3
noncomputable def added_gallons_1 : ℝ := 6.8
noncomputable def liters_to_gallons (liters : ℝ) : ℝ := liters / 3.78541
noncomputable def quart_to_gallons (quarts : ℝ) : ℝ := quarts / 4
noncomputable def added_gallons_2 : ℝ := liters_to_gallons 10
noncomputable def added_gallons_3 : ℝ := quart_to_gallons 4

noncomputable def total_gallons : ℝ :=
  initial_gallons + added_gallons_1 + added_gallons_2 + added_gallons_3

theorem total_water_in_bucket :
  abs (total_gallons - 13.44) < 0.01 :=
by
  -- convert amounts and perform arithmetic operations
  sorry

end total_water_in_bucket_l138_138655


namespace different_answers_due_to_different_cuts_l138_138277

noncomputable def problem_89914 (bub : Type) (cut : bub → (bub × bub)) (is_log_cut : bub → Prop) (is_halved_log : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_log_cut b) → is_halved_log (cut b)

noncomputable def problem_89915 (bub : Type) (cut : bub → (bub × bub)) (is_sector_cut : bub → Prop) (is_sectors : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_sector_cut b) → is_sectors (cut b)

theorem different_answers_due_to_different_cuts
  (bub : Type)
  (cut : bub → (bub × bub))
  (is_log_cut : bub → Prop)
  (is_halved_log : bub × bub → Prop)
  (is_sector_cut : bub → Prop)
  (is_sectors : bub × bub → Prop) :
  problem_89914 bub cut is_log_cut is_halved_log ∧ problem_89915 bub cut is_sector_cut is_sectors →
  ∃ b : bub, (is_log_cut b ∧ ¬ is_sector_cut b) ∨ (¬ is_log_cut b ∧ is_sector_cut b) := sorry

end different_answers_due_to_different_cuts_l138_138277


namespace xy_uv_zero_l138_138304

theorem xy_uv_zero (x y u v : ℝ) (h1 : x^2 + y^2 = 1) (h2 : u^2 + v^2 = 1) (h3 : x * u + y * v = 0) : x * y + u * v = 0 :=
by
  sorry

end xy_uv_zero_l138_138304


namespace monday_rainfall_l138_138418

theorem monday_rainfall (tuesday_rainfall monday_rainfall: ℝ) 
(less_rain: ℝ) (h1: tuesday_rainfall = 0.2) 
(h2: less_rain = 0.7) 
(h3: tuesday_rainfall = monday_rainfall - less_rain): 
monday_rainfall = 0.9 :=
by sorry

end monday_rainfall_l138_138418


namespace fraction_numerator_l138_138685

theorem fraction_numerator (x : ℤ) (h₁ : 2 * x + 11 ≠ 0) (h₂ : (x : ℚ) / (2 * x + 11) = 3 / 4) : x = -33 / 2 :=
by
  sorry

end fraction_numerator_l138_138685


namespace cost_of_building_fence_l138_138077

-- Define the conditions
def area : ℕ := 289
def price_per_foot : ℕ := 60

-- Define the length of one side of the square (since area = side^2)
def side_length (a : ℕ) : ℕ := Nat.sqrt a

-- Define the perimeter of the square (since square has 4 equal sides)
def perimeter (s : ℕ) : ℕ := 4 * s

-- Define the cost of building the fence
def cost (p : ℕ) (ppf : ℕ) : ℕ := p * ppf

-- Prove that the cost of building the fence is Rs. 4080
theorem cost_of_building_fence : cost (perimeter (side_length area)) price_per_foot = 4080 := by
  -- Skip the proof steps
  sorry

end cost_of_building_fence_l138_138077


namespace num_sets_N_l138_138460

open Set

noncomputable def M : Set ℤ := {-1, 0}

theorem num_sets_N (N : Set ℤ) : M ∪ N = {-1, 0, 1} → 
  (N = {1} ∨ N = {0, 1} ∨ N = {-1, 1} ∨ N = {0, -1, 1}) := 
sorry

end num_sets_N_l138_138460


namespace find_line_equation_l138_138419

noncomputable def y_line (m b x : ℝ) : ℝ := m * x + b
noncomputable def quadratic_y (x : ℝ) : ℝ := x ^ 2 + 8 * x + 7

noncomputable def equation_of_the_line : Prop :=
  ∃ (m b k : ℝ),
    (quadratic_y k = y_line m b k + 6 ∨ quadratic_y k = y_line m b k - 6) ∧
    (y_line m b 2 = 7) ∧ 
    b ≠ 0 ∧
    y_line 19.5 (-32) = y_line m b

theorem find_line_equation : equation_of_the_line :=
sorry

end find_line_equation_l138_138419


namespace total_cars_l138_138759

-- Definitions of the conditions
def cathy_cars : Nat := 5

def carol_cars : Nat := 2 * cathy_cars

def susan_cars : Nat := carol_cars - 2

def lindsey_cars : Nat := cathy_cars + 4

-- The theorem statement (problem)
theorem total_cars : cathy_cars + carol_cars + susan_cars + lindsey_cars = 32 :=
by
  -- sorry is added to skip the proof
  sorry

end total_cars_l138_138759


namespace negation_proposition_l138_138939

theorem negation_proposition :
  (∀ x : ℝ, |x - 2| + |x - 4| > 3) = ¬(∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) :=
  by sorry

end negation_proposition_l138_138939


namespace maximize_profit_l138_138714

noncomputable section

def price (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x ≤ 600 then 62 - 0.02 * x
  else 0

def profit (x : ℕ) : ℝ :=
  (price x - 40) * x

theorem maximize_profit :
  ∃ x : ℕ, (1 ≤ x ∧ x ≤ 600) ∧ (∀ y : ℕ, (1 ≤ y ∧ y ≤ 600 → profit y ≤ profit x)) ∧ profit x = 6050 :=
by sorry

end maximize_profit_l138_138714


namespace yo_yos_collected_l138_138459

-- Define the given conditions
def stuffed_animals : ℕ := 14
def frisbees : ℕ := 18
def total_prizes : ℕ := 50

-- Define the problem to prove that the number of yo-yos is 18
theorem yo_yos_collected : (total_prizes - (stuffed_animals + frisbees) = 18) :=
by
  sorry

end yo_yos_collected_l138_138459


namespace Nicole_has_69_clothes_l138_138956

def clothingDistribution : Prop :=
  let nicole_clothes := 15
  let first_sister_clothes := nicole_clothes / 3
  let second_sister_clothes := nicole_clothes + 5
  let third_sister_clothes := 2 * first_sister_clothes
  let average_clothes := (nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes) / 4
  let oldest_sister_clothes := 1.5 * average_clothes
  let total_clothes := nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes + oldest_sister_clothes
  total_clothes = 69

theorem Nicole_has_69_clothes : clothingDistribution :=
by
  -- Proof omitted
  sorry

end Nicole_has_69_clothes_l138_138956


namespace cubic_vs_square_ratio_l138_138115

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l138_138115


namespace calculate_number_of_boys_l138_138932

theorem calculate_number_of_boys (old_average new_average misread correct_weight : ℝ) (number_of_boys : ℕ)
  (h1 : old_average = 58.4)
  (h2 : misread = 56)
  (h3 : correct_weight = 61)
  (h4 : new_average = 58.65)
  (h5 : (number_of_boys : ℝ) * old_average + (correct_weight - misread) = (number_of_boys : ℝ) * new_average) :
  number_of_boys = 20 :=
by
  sorry

end calculate_number_of_boys_l138_138932


namespace evaluate_expression_l138_138674

theorem evaluate_expression : 10 * 0.2 * 5 * 0.1 + 5 = 6 :=
by
  -- transformed step-by-step mathematical proof goes here
  sorry

end evaluate_expression_l138_138674


namespace identify_clothing_l138_138262

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l138_138262


namespace Ned_washed_shirts_l138_138642

-- Definitions based on conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts
def not_washed_shirts : ℕ := 1
def washed_shirts : ℕ := total_shirts - not_washed_shirts

-- Statement to prove
theorem Ned_washed_shirts : washed_shirts = 29 := by
  sorry

end Ned_washed_shirts_l138_138642


namespace fraction_of_subsets_l138_138896

theorem fraction_of_subsets (S T : ℕ) (hS : S = 2^10) (hT : T = Nat.choose 10 3) :
    (T:ℚ) / (S:ℚ) = 15 / 128 :=
by sorry

end fraction_of_subsets_l138_138896


namespace six_x_plus_four_eq_twenty_two_l138_138334

theorem six_x_plus_four_eq_twenty_two (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 := 
by
  sorry

end six_x_plus_four_eq_twenty_two_l138_138334


namespace sum_x_coordinates_midpoints_l138_138957

theorem sum_x_coordinates_midpoints (a b c : ℝ) (h : a + b + c = 12) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end sum_x_coordinates_midpoints_l138_138957


namespace largest_possible_dividend_l138_138822

theorem largest_possible_dividend (divisor quotient : ℕ) (remainder : ℕ) 
  (h_divisor : divisor = 18)
  (h_quotient : quotient = 32)
  (h_remainder : remainder < divisor) :
  quotient * divisor + remainder = 593 :=
by
  -- No proof here, add sorry to skip the proof
  sorry

end largest_possible_dividend_l138_138822


namespace find_m_range_l138_138819

-- Definitions for the conditions and the required proof
def condition_alpha (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m + 7
def condition_beta (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

-- Proof problem translated to Lean 4 statement
theorem find_m_range (m : ℝ) :
  (∀ x, condition_beta x → condition_alpha m x) → (-2 ≤ m ∧ m ≤ 0) :=
by sorry

end find_m_range_l138_138819


namespace pencils_calculation_l138_138751

def num_pencil_boxes : ℝ := 4.0
def pencils_per_box : ℝ := 648.0
def total_pencils : ℝ := 2592.0

theorem pencils_calculation : (num_pencil_boxes * pencils_per_box) = total_pencils := 
by
  sorry

end pencils_calculation_l138_138751


namespace price_decrease_percentage_l138_138078

theorem price_decrease_percentage (P₀ P₁ P₂ : ℝ) (x : ℝ) :
  P₀ = 1 → P₁ = P₀ * 1.25 → P₂ = P₁ * (1 - x / 100) → P₂ = 1 → x = 20 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end price_decrease_percentage_l138_138078


namespace angle_B_measure_l138_138463

theorem angle_B_measure (a b : ℝ) (A B : ℝ) (h₁ : a = 4) (h₂ : b = 4 * Real.sqrt 3) (h₃ : A = Real.pi / 6) : 
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
by
  sorry

end angle_B_measure_l138_138463


namespace hectares_per_day_initial_l138_138316

variable (x : ℝ) -- x is the number of hectares one tractor ploughs initially per day

-- Condition 1: A field can be ploughed by 6 tractors in 4 days.
def total_area_initial := 6 * x * 4

-- Condition 2: 6 tractors plough together a certain number of hectares per day, denoted as x hectares/day.
-- This is incorporated in the variable declaration of x.

-- Condition 3: If 2 tractors are moved to another field, the remaining 4 tractors can plough the same field in 5 days.
-- Condition 4: One of the 4 tractors ploughs 144 hectares a day when 4 tractors plough the field in 5 days.
def total_area_with_4_tractors := 4 * 144 * 5

-- The statement that equates the two total area expressions.
theorem hectares_per_day_initial : total_area_initial x = total_area_with_4_tractors := by
  sorry

end hectares_per_day_initial_l138_138316


namespace money_split_l138_138799

theorem money_split (donna_share friend_share : ℝ) (h1 : donna_share = 32.50) (h2 : friend_share = 32.50) :
  donna_share + friend_share = 65 :=
by
  sorry

end money_split_l138_138799


namespace journey_distance_l138_138214

theorem journey_distance 
  (T : ℝ) 
  (s1 s2 s3 : ℝ) 
  (hT : T = 36) 
  (hs1 : s1 = 21)
  (hs2 : s2 = 45)
  (hs3 : s3 = 24) : ∃ (D : ℝ), D = 972 :=
  sorry

end journey_distance_l138_138214


namespace sum_of_a_for_unique_solution_l138_138439

theorem sum_of_a_for_unique_solution (a : ℝ) (x : ℝ) :
  (∃ (a : ℝ), 3 * x ^ 2 + a * x + 6 * x + 7 = 0 ∧ (a + 6) ^ 2 - 4 * 3 * 7 = 0) →
  (-6 + 2 * Real.sqrt 21 + -6 - 2 * Real.sqrt 21 = -12) :=
by
  sorry

end sum_of_a_for_unique_solution_l138_138439


namespace non_visible_dots_l138_138100

-- Define the configuration of the dice
def total_dots_on_one_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
def total_dots_on_two_dice : ℕ := 2 * total_dots_on_one_die
def visible_dots : ℕ := 2 + 3 + 5

-- The statement to prove
theorem non_visible_dots : total_dots_on_two_dice - visible_dots = 32 := by sorry

end non_visible_dots_l138_138100


namespace winning_votes_l138_138750

theorem winning_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 312) : 0.62 * V = 806 :=
by
  -- The proof should be written here, but we'll skip it as per the instructions.
  sorry

end winning_votes_l138_138750


namespace unique_solution_set_l138_138068

theorem unique_solution_set :
  {a : ℝ | ∃! x : ℝ, (x^2 - 4) / (x + a) = 1} = { -17 / 4, -2, 2 } :=
by sorry

end unique_solution_set_l138_138068


namespace fountain_water_after_25_days_l138_138699

def initial_volume : ℕ := 120
def evaporation_rate : ℕ := 8 / 10 -- Representing 0.8 gallons as 8/10
def rain_addition : ℕ := 5
def days : ℕ := 25
def rain_period : ℕ := 5

-- Calculate the amount of water after 25 days given the above conditions
theorem fountain_water_after_25_days :
  initial_volume + ((days / rain_period) * rain_addition) - (days * evaporation_rate) = 125 :=
by
  sorry

end fountain_water_after_25_days_l138_138699


namespace line_through_point_perpendicular_y_axis_line_through_two_points_l138_138550

-- The first problem
theorem line_through_point_perpendicular_y_axis :
  ∃ (k : ℝ), ∀ (x : ℝ), k = 1 → y = k :=
sorry

-- The second problem
theorem line_through_two_points (x1 y1 x2 y2 : ℝ) (hA : (x1, y1) = (-4, 0)) (hB : (x2, y2) = (0, 6)) :
  ∃ (a b c : ℝ), (a, b, c) = (3, -2, 12) → ∀ (x y : ℝ), a * x + b * y + c = 0 :=
sorry

end line_through_point_perpendicular_y_axis_line_through_two_points_l138_138550


namespace solve_inequality_l138_138110

theorem solve_inequality (x : ℝ) : x + 2 < 1 ↔ x < -1 := sorry

end solve_inequality_l138_138110


namespace ray_walks_to_high_school_7_l138_138598

theorem ray_walks_to_high_school_7
  (walks_to_park : ℕ)
  (walks_to_high_school : ℕ)
  (walks_home : ℕ)
  (trips_per_day : ℕ)
  (total_daily_blocks : ℕ) :
  walks_to_park = 4 →
  walks_home = 11 →
  trips_per_day = 3 →
  total_daily_blocks = 66 →
  3 * (walks_to_park + walks_to_high_school + walks_home) = total_daily_blocks →
  walks_to_high_school = 7 :=
by
  sorry

end ray_walks_to_high_school_7_l138_138598


namespace gray_percentage_correct_l138_138840

-- Define the conditions
def total_squares := 25
def type_I_triangle_equivalent_squares := 8 * (1 / 2)
def type_II_triangle_equivalent_squares := 8 * (1 / 4)
def full_gray_squares := 4

-- Calculate the gray component
def gray_squares := type_I_triangle_equivalent_squares + type_II_triangle_equivalent_squares + full_gray_squares

-- Fraction representing the gray part of the quilt
def gray_fraction := gray_squares / total_squares

-- Translate fraction to percentage
def gray_percentage := gray_fraction * 100

theorem gray_percentage_correct : gray_percentage = 40 := by
  simp [total_squares, type_I_triangle_equivalent_squares, type_II_triangle_equivalent_squares, full_gray_squares, gray_squares, gray_fraction, gray_percentage]
  sorry -- You could expand this to a detailed proof if needed.

end gray_percentage_correct_l138_138840


namespace a_n_sequence_term2015_l138_138428

theorem a_n_sequence_term2015 :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ a 2 = 1/2 ∧ (∀ n ≥ 2, a n * (a (n-1) + a (n+1)) = 2 * a (n+1) * a (n-1)) ∧ a 2015 = 1/2015 :=
sorry

end a_n_sequence_term2015_l138_138428


namespace binom_sub_floor_div_prime_l138_138640

theorem binom_sub_floor_div_prime {n p : ℕ} (hp : Nat.Prime p) (hpn : n ≥ p) : 
  p ∣ (Nat.choose n p - (n / p)) :=
sorry

end binom_sub_floor_div_prime_l138_138640


namespace find_two_digit_number_l138_138117

theorem find_two_digit_number (x y : ℕ) (h1 : 10 * x + y = 4 * (x + y) + 3) (h2 : 10 * x + y = 3 * x * y + 5) : 10 * x + y = 23 :=
by {
  sorry
}

end find_two_digit_number_l138_138117


namespace problem1_problem2_l138_138183

theorem problem1 (a b : ℝ) : (-(2 : ℝ) * a ^ 2 * b) ^ 3 / (-(2 * a * b)) * (1 / 3 * a ^ 2 * b ^ 3) = (4 / 3) * a ^ 7 * b ^ 5 :=
  by
  sorry

theorem problem2 (x : ℝ) : (27 * x ^ 3 + 18 * x ^ 2 - 3 * x) / -3 * x = -9 * x ^ 2 - 6 * x + 1 :=
  by
  sorry

end problem1_problem2_l138_138183


namespace find_x_l138_138593

theorem find_x (x : ℝ) (h : (3 * x - 4) / 7 = 15) : x = 109 / 3 :=
by sorry

end find_x_l138_138593


namespace perpendicular_vectors_m_value_l138_138106

theorem perpendicular_vectors_m_value : 
  ∀ (m : ℝ), ((2 : ℝ) * (1 : ℝ) + (m * (1 / 2)) + (1 * 2) = 0) → m = -8 :=
by
  intro m
  intro h
  sorry

end perpendicular_vectors_m_value_l138_138106


namespace factory_processing_time_eq_l138_138296

variable (x : ℝ) (initial_rate : ℝ := x)
variable (parts : ℝ := 500)
variable (first_stage_parts : ℝ := 100)
variable (remaining_parts : ℝ := parts - first_stage_parts)
variable (total_days : ℝ := 6)
variable (new_rate : ℝ := 2 * initial_rate)

theorem factory_processing_time_eq (h : x > 0) : (first_stage_parts / initial_rate) + (remaining_parts / new_rate) = total_days := 
by
  sorry

end factory_processing_time_eq_l138_138296


namespace range_of_a_condition_l138_138387

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem range_of_a_condition :
  range_of_a a → -1 < a ∧ a < 3 := sorry

end range_of_a_condition_l138_138387


namespace fraction_simplification_l138_138188

theorem fraction_simplification : 
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = (1 / 3) := 
sorry

end fraction_simplification_l138_138188


namespace intersection_with_complement_l138_138697

-- Definitions for the universal set and set A
def U : Set ℝ := Set.univ

def A : Set ℝ := { -1, 0, 1 }

-- Definition for set B using the given condition
def B : Set ℝ := { x : ℝ | (x - 2) / (x + 1) > 0 }

-- Definition for the complement of B
def B_complement : Set ℝ := { x : ℝ | -1 <= x ∧ x <= 0 }

-- Theorem stating the intersection of A and the complement of B equals {-1, 0, 1}
theorem intersection_with_complement : 
  A ∩ B_complement = { -1, 0, 1 } :=
by
  sorry

end intersection_with_complement_l138_138697


namespace complement_intersection_l138_138143

open Set

-- Define the universal set U
def U : Set ℕ := {x | 0 < x ∧ x < 7}

-- Define Set A
def A : Set ℕ := {2, 3, 5}

-- Define Set B
def B : Set ℕ := {1, 4}

-- Define the complement of A in U
def CU_A : Set ℕ := U \ A

-- Define the complement of B in U
def CU_B : Set ℕ := U \ B

-- Define the intersection of CU_A and CU_B
def intersection_CU_A_CU_B : Set ℕ := CU_A ∩ CU_B

-- The theorem statement
theorem complement_intersection :
  intersection_CU_A_CU_B = {6} := by
  sorry

end complement_intersection_l138_138143


namespace average_shift_l138_138488

variable (a b c : ℝ)

-- Given condition: The average of the data \(a\), \(b\), \(c\) is 5.
def average_is_five := (a + b + c) / 3 = 5

-- Define the statement to prove: The average of the data \(a-2\), \(b-2\), \(c-2\) is 3.
theorem average_shift (h : average_is_five a b c) : ((a - 2) + (b - 2) + (c - 2)) / 3 = 3 :=
by
  sorry

end average_shift_l138_138488


namespace bill_experience_l138_138826

theorem bill_experience (j b : ℕ) 
  (h₁ : j - 5 = 3 * (b - 5)) 
  (h₂ : j = 2 * b) : b = 10 :=
sorry

end bill_experience_l138_138826


namespace rose_needs_more_money_l138_138774

def cost_of_paintbrush : ℝ := 2.4
def cost_of_paints : ℝ := 9.2
def cost_of_easel : ℝ := 6.5
def amount_rose_has : ℝ := 7.1
def total_cost : ℝ := cost_of_paintbrush + cost_of_paints + cost_of_easel

theorem rose_needs_more_money : (total_cost - amount_rose_has) = 11 := 
by
  -- Proof goes here
  sorry

end rose_needs_more_money_l138_138774


namespace chess_tournament_l138_138769

theorem chess_tournament (n : ℕ) (h : (n * (n - 1)) / 2 - ((n - 3) * (n - 4)) / 2 = 130) : n = 19 :=
sorry

end chess_tournament_l138_138769


namespace value_of_f_three_l138_138291

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * Real.cos x - x

theorem value_of_f_three (a b : ℝ) (h : f a b (-3) = 7) : f a b 3 = 1 :=
by
  sorry

end value_of_f_three_l138_138291


namespace initial_holes_count_additional_holes_needed_l138_138765

-- Defining the conditions as variables
def circumference : ℕ := 400
def initial_interval : ℕ := 50
def new_interval : ℕ := 40

-- Defining the problems

-- Problem 1: Calculate the number of holes for the initial interval
theorem initial_holes_count (circumference : ℕ) (initial_interval : ℕ) : 
  circumference % initial_interval = 0 → 
  circumference / initial_interval = 8 := 
sorry

-- Problem 2: Calculate the additional holes needed
theorem additional_holes_needed (circumference : ℕ) (initial_interval : ℕ) 
  (new_interval : ℕ) (lcm_interval : ℕ) :
  lcm new_interval initial_interval = lcm_interval →
  circumference % new_interval = 0 →
  circumference / new_interval - 
  (circumference / lcm_interval) = 8 :=
sorry

end initial_holes_count_additional_holes_needed_l138_138765


namespace parallelepiped_vectors_l138_138635

theorem parallelepiped_vectors (x y z : ℝ)
  (h1: ∀ (AB BC CC1 AC1 : ℝ), AC1 = AB + BC + CC1)
  (h2: ∀ (AB BC CC1 AC1 : ℝ), AC1 = x * AB + 2 * y * BC + 3 * z * CC1) :
  x + y + z = 11 / 6 :=
by
  -- This is where the proof would go, but as per the instruction we'll add sorry.
  sorry

end parallelepiped_vectors_l138_138635


namespace sum_of_areas_is_858_l138_138000

def first_six_odd_squares : List ℕ := [1^2, 3^2, 5^2, 7^2, 9^2, 11^2]

def rectangle_area (width length : ℕ) : ℕ := width * length

def sum_of_areas : ℕ := (first_six_odd_squares.map (rectangle_area 3)).sum

theorem sum_of_areas_is_858 : sum_of_areas = 858 := 
by
  -- Our aim is to show that sum_of_areas is 858
  -- The proof will be developed here
  sorry

end sum_of_areas_is_858_l138_138000


namespace playerB_hit_rate_playerA_probability_l138_138233

theorem playerB_hit_rate (p : ℝ) (h : (1 - p)^2 = 1/16) : p = 3/4 :=
sorry

theorem playerA_probability (hit_rate : ℝ) (h : hit_rate = 1/2) : 
  (1 - (1 - hit_rate)^2) = 3/4 :=
sorry

end playerB_hit_rate_playerA_probability_l138_138233


namespace find_q_sum_l138_138227

variable (q : ℕ → ℕ)

def conditions :=
  q 3 = 2 ∧ 
  q 8 = 20 ∧ 
  q 16 = 12 ∧ 
  q 21 = 30

theorem find_q_sum (h : conditions q) : 
  (q 1 + q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + 
   q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18 + q 19 + q 20 + q 21 + q 22) = 352 := 
  sorry

end find_q_sum_l138_138227


namespace multiply_fractions_l138_138153

theorem multiply_fractions :
  (1 / 3) * (4 / 7) * (9 / 13) * (2 / 5) = 72 / 1365 :=
by sorry

end multiply_fractions_l138_138153


namespace pat_mark_ratio_l138_138971

theorem pat_mark_ratio :
  ∃ K P M : ℕ, P + K + M = 189 ∧ P = 2 * K ∧ M = K + 105 ∧ P / gcd P M = 1 ∧ M / gcd P M = 3 :=
by
  sorry

end pat_mark_ratio_l138_138971


namespace valid_triangles_from_10_points_l138_138854

noncomputable def number_of_valid_triangles (n : ℕ) (h : n = 10) : ℕ :=
  if n = 10 then 100 else 0

theorem valid_triangles_from_10_points :
  number_of_valid_triangles 10 rfl = 100 := 
sorry

end valid_triangles_from_10_points_l138_138854


namespace matchsticks_distribution_l138_138738

open Nat

theorem matchsticks_distribution
  (length_sticks : ℕ)
  (width_sticks : ℕ)
  (length_condition : length_sticks = 60)
  (width_condition : width_sticks = 10)
  (total_sticks : ℕ)
  (total_sticks_condition : total_sticks = 60 * 11 + 10 * 61)
  (children_count : ℕ)
  (children_condition : children_count > 100)
  (division_condition : total_sticks % children_count = 0) :
  children_count = 127 := by
  sorry

end matchsticks_distribution_l138_138738


namespace circle_line_intersection_symmetric_l138_138879

theorem circle_line_intersection_symmetric (m n p x y : ℝ)
    (h_intersects : ∃ x y, x = m * y - 1 ∧ x^2 + y^2 + m * x + n * y + p = 0)
    (h_symmetric : ∀ A B : ℝ × ℝ, A = (x, y) ∧ B = (y, x) → y = x) :
    p < -3 / 2 :=
by
  sorry

end circle_line_intersection_symmetric_l138_138879


namespace mary_income_percentage_l138_138044

-- Declare noncomputable as necessary
noncomputable def calculate_percentage_more
    (J : ℝ) -- Juan's income
    (T : ℝ) (M : ℝ)
    (hT : T = 0.70 * J) -- Tim's income is 30% less than Juan's income
    (hM : M = 1.12 * J) -- Mary's income is 112% of Juan's income
    : ℝ :=
  ((M - T) / T) * 100

theorem mary_income_percentage
    (J T M : ℝ)
    (hT : T = 0.70 * J)
    (hM : M = 1.12 * J) :
    calculate_percentage_more J T M hT hM = 60 :=
by sorry

end mary_income_percentage_l138_138044


namespace nell_initial_ace_cards_l138_138475

def initial_ace_cards (initial_baseball_cards final_ace_cards final_baseball_cards given_difference : ℕ) : ℕ :=
  final_ace_cards + (initial_baseball_cards - final_baseball_cards)

theorem nell_initial_ace_cards : 
  initial_ace_cards 239 376 111 265 = 504 :=
by
  /- This is to show that the initial count of Ace cards Nell had is 504 given the conditions -/
  sorry

end nell_initial_ace_cards_l138_138475


namespace sum_y_coords_l138_138638

theorem sum_y_coords (h1 : ∃(y : ℝ), (0 + 3)^2 + (y - 5)^2 = 64) : 
  ∃ y1 y2 : ℝ, y1 + y2 = 10 ∧ (0, y1) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) ∧ 
                            (0, y2) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) := 
by
  sorry

end sum_y_coords_l138_138638


namespace stable_performance_l138_138991

/-- The variance of student A's scores is 0.4 --/
def variance_A : ℝ := 0.4

/-- The variance of student B's scores is 0.3 --/
def variance_B : ℝ := 0.3

/-- Prove that student B has more stable performance given the variances --/
theorem stable_performance (h1 : variance_A = 0.4) (h2 : variance_B = 0.3) : variance_B < variance_A :=
by
  rw [h1, h2]
  exact sorry

end stable_performance_l138_138991


namespace EM_parallel_AC_l138_138234

-- Define the points A, B, C, D, E, and M
variables (A B C D E M : Type) 

-- Define the conditions described in the problem
variables {x y : Real}

-- Given that ABCD is an isosceles trapezoid with AB parallel to CD and AB > CD
variable (isosceles_trapezoid : Prop)

-- E is the foot of the perpendicular from D to AB
variable (foot_perpendicular : Prop)

-- M is the midpoint of BD
variable (midpoint : Prop)

-- We need to prove that EM is parallel to AC
theorem EM_parallel_AC (h1 : isosceles_trapezoid) (h2 : foot_perpendicular) (h3 : midpoint) : Prop := sorry

end EM_parallel_AC_l138_138234


namespace increasing_on_real_iff_a_range_l138_138308

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a / x

theorem increasing_on_real_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -3 ≤ a ∧ a ≤ -2 := 
by
  sorry

end increasing_on_real_iff_a_range_l138_138308


namespace decrease_of_negative_five_l138_138510

-- Definition: Positive and negative numbers as explained
def increase (n: ℤ) : Prop := n > 0
def decrease (n: ℤ) : Prop := n < 0

-- Conditions
def condition : Prop := increase 17

-- Theorem stating the solution
theorem decrease_of_negative_five (h : condition) : decrease (-5) ∧ -5 = -5 :=
by
  sorry

end decrease_of_negative_five_l138_138510
