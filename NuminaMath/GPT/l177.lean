import Mathlib

namespace domain_of_function_l177_177265

theorem domain_of_function : {x : ℝ | 3 - 2 * x - x ^ 2 ≥ 0 } = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l177_177265


namespace solve_y_l177_177454

theorem solve_y : ∀ y : ℚ, (9 * y^2 + 8 * y - 2 = 0) ∧ (27 * y^2 + 62 * y - 8 = 0) → y = 1 / 9 :=
by
  intro y h
  cases h
  sorry

end solve_y_l177_177454


namespace find_flat_fee_l177_177041

def flat_fee_exists (f n : ℝ) : Prop :=
  f + n = 120 ∧ f + 4 * n = 255

theorem find_flat_fee : ∃ f n, flat_fee_exists f n ∧ f = 75 := by
  sorry

end find_flat_fee_l177_177041


namespace alice_numbers_l177_177386

theorem alice_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : x + y = 7) : (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) :=
by
  sorry

end alice_numbers_l177_177386


namespace find_f3_l177_177288

theorem find_f3 (a b : ℝ) (f : ℝ → ℝ)
  (h1 : f 1 = 3)
  (h2 : f 2 = 6)
  (h3 : ∀ x, f x = a * x^2 + b * x + 1) :
  f 3 = 10 :=
sorry

end find_f3_l177_177288


namespace patient_treatment_volume_l177_177379

noncomputable def total_treatment_volume : ℝ :=
  let drop_rate1 := 15     -- drops per minute for the first drip
  let ml_rate1 := 6 / 120  -- milliliters per drop for the first drip
  let drop_rate2 := 25     -- drops per minute for the second drip
  let ml_rate2 := 7.5 / 90 -- milliliters per drop for the second drip
  let total_time := 4 * 60 -- total minutes including breaks
  let break_time := 4 * 10 -- total break time in minutes
  let actual_time := total_time - break_time -- actual running time in minutes
  let total_drops1 := actual_time * drop_rate1
  let total_drops2 := actual_time * drop_rate2
  let volume1 := total_drops1 * ml_rate1
  let volume2 := total_drops2 * ml_rate2
  volume1 + volume2 -- total volume from both drips

theorem patient_treatment_volume : total_treatment_volume = 566.67 :=
  by
    -- Place the necessary calculation steps as assumptions or directly as one-liner
    sorry

end patient_treatment_volume_l177_177379


namespace smallest_AAB_l177_177669

theorem smallest_AAB (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) 
  (h : 10 * A + B = (110 * A + B) / 7) : 110 * A + B = 996 :=
by
  sorry

end smallest_AAB_l177_177669


namespace div_by_30_l177_177132

theorem div_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end div_by_30_l177_177132


namespace marian_returned_amount_l177_177601

theorem marian_returned_amount
  (B : ℕ) (G : ℕ) (H : ℕ) (N : ℕ)
  (hB : B = 126) (hG : G = 60) (hH : H = G / 2) (hN : N = 171) :
  (B + G + H - N) = 45 := 
by
  sorry

end marian_returned_amount_l177_177601


namespace find_angle_B_l177_177862

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l177_177862


namespace not_a_cube_l177_177183

theorem not_a_cube (a b : ℤ) : ¬ ∃ c : ℤ, a^3 + b^3 + 4 = c^3 := 
sorry

end not_a_cube_l177_177183


namespace difference_in_soda_bottles_l177_177776

def diet_soda_bottles : ℕ := 4
def regular_soda_bottles : ℕ := 83

theorem difference_in_soda_bottles :
  regular_soda_bottles - diet_soda_bottles = 79 :=
by
  sorry

end difference_in_soda_bottles_l177_177776


namespace find_n_of_permut_comb_eq_l177_177572

open Nat

theorem find_n_of_permut_comb_eq (n : Nat) (h : (n! / (n - 3)!) = 6 * (n! / (4! * (n - 4)!))) : n = 7 := by
  sorry

end find_n_of_permut_comb_eq_l177_177572


namespace expected_variance_replanted_seeds_l177_177276

theorem expected_variance_replanted_seeds:
  (p = 0.9) →
  (n = 1000) →
  (t = 2) →
  (q = 1 - p) →
  (E_X = n * q * t) →
  (Var_X = n * q * (1 - q) * t^2) →
  E_X = 200 ∧ Var_X = 360 := 
by
  intros p_eq n_eq t_eq q_eq E_X_eq Var_X_eq
  rw [p_eq, n_eq, t_eq, q_eq] at *
  simp at E_X_eq Var_X_eq
  exact ⟨E_X_eq, Var_X_eq⟩
  -- Remember that you need to replace 'E_X' and 'Var_X' by their actual values
  sorry -- actual simplifying and solving steps using Lean would replace this

end expected_variance_replanted_seeds_l177_177276


namespace probability_student_gets_at_least_12_correct_l177_177054

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

noncomputable def prob_at_least_12_correct : ℚ :=
  (finset.range 21).filter (≥ 12)
  .sum (λ k, binomial_prob 20 k (1/2 : ℚ))

theorem probability_student_gets_at_least_12_correct :
  prob_at_least_12_correct = 160466 / 1048576 :=
begin
  sorry
end

end probability_student_gets_at_least_12_correct_l177_177054


namespace find_total_students_l177_177749

theorem find_total_students (n : ℕ) : n < 550 ∧ n % 19 = 15 ∧ n % 17 = 10 → n = 509 :=
by 
  sorry

end find_total_students_l177_177749


namespace inequality_system_solution_l177_177431

theorem inequality_system_solution (a b : ℝ) (h : ∀ x : ℝ, x > -a → x > -b) : a ≥ b :=
by
  sorry

end inequality_system_solution_l177_177431


namespace sum_greater_than_3_l177_177606

theorem sum_greater_than_3 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a > a + b + c) : a + b + c > 3 :=
sorry

end sum_greater_than_3_l177_177606


namespace find_x_plus_z_l177_177805

theorem find_x_plus_z :
  ∃ (x y z : ℝ), 
  (x + y + z = 0) ∧
  (2016 * x + 2017 * y + 2018 * z = 0) ∧
  (2016^2 * x + 2017^2 * y + 2018^2 * z = 2018) ∧
  (x + z = 4036) :=
sorry

end find_x_plus_z_l177_177805


namespace fewest_number_of_students_l177_177718

theorem fewest_number_of_students :
  ∃ n : ℕ, n ≡ 3 [MOD 6] ∧ n ≡ 5 [MOD 8] ∧ n ≡ 7 [MOD 9] ∧ ∀ m : ℕ, (m ≡ 3 [MOD 6] ∧ m ≡ 5 [MOD 8] ∧ m ≡ 7 [MOD 9]) → m ≥ n := by
  sorry

end fewest_number_of_students_l177_177718


namespace work_completion_days_l177_177231

theorem work_completion_days (d : ℝ) : (1 / 15 + 1 / d = 1 / 11.25) → d = 45 := sorry

end work_completion_days_l177_177231


namespace min_visible_sum_of_4x4x4_cube_l177_177950

theorem min_visible_sum_of_4x4x4_cube (dice_capacity : ℕ) (opposite_sum : ℕ) (corner_dice edge_dice center_face_dice innermost_dice : ℕ) : 
  dice_capacity = 64 ∧ 
  opposite_sum = 7 ∧ 
  corner_dice = 8 ∧ 
  edge_dice = 24 ∧ 
  center_face_dice = 24 ∧ 
  innermost_dice = 8 → 
  ∃ min_sum, min_sum = 144 := by
  sorry

end min_visible_sum_of_4x4x4_cube_l177_177950


namespace inequality_1_solution_set_inequality_2_solution_set_l177_177328

theorem inequality_1_solution_set (x : ℝ) : 
  (2 + 3 * x - 2 * x^2 > 0) ↔ (-1/2 < x ∧ x < 2) := 
by sorry

theorem inequality_2_solution_set (x : ℝ) :
  (x * (3 - x) ≤ x * (x + 2) - 1) ↔ (x ≤ -1/2 ∨ x ≥ 1) :=
by sorry

end inequality_1_solution_set_inequality_2_solution_set_l177_177328


namespace smart_charging_piles_equation_l177_177581

-- defining conditions as constants
constant initial_piles : ℕ := 301
constant third_month_piles : ℕ := 500
constant growth_rate : ℝ 

-- Expressing the given mathematical proof problem
theorem smart_charging_piles_equation : 
  initial_piles * (1 + growth_rate)^2 = third_month_piles :=
sorry

end smart_charging_piles_equation_l177_177581


namespace smallest_common_multiple_5_6_l177_177934

theorem smallest_common_multiple_5_6 (n : ℕ) 
  (h_pos : 0 < n) 
  (h_5 : 5 ∣ n) 
  (h_6 : 6 ∣ n) :
  n = 30 :=
sorry

end smallest_common_multiple_5_6_l177_177934


namespace sum_of_perimeters_l177_177213

theorem sum_of_perimeters (x y : ℝ) (h₁ : x^2 + y^2 = 125) (h₂ : x^2 - y^2 = 65) : 4 * x + 4 * y = 60 := 
by
  sorry

end sum_of_perimeters_l177_177213


namespace fraction_multiplication_l177_177491

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l177_177491


namespace samantha_birth_year_l177_177920

theorem samantha_birth_year (first_kangaroo_year birth_year kangaroo_freq : ℕ)
  (h_first_kangaroo: first_kangaroo_year = 1991)
  (h_kangaroo_freq: kangaroo_freq = 1)
  (h_samantha_age: ∃ y, y = (first_kangaroo_year + 9 * kangaroo_freq) ∧ 2000 - 14 = y) :
  birth_year = 1986 :=
by sorry

end samantha_birth_year_l177_177920


namespace max_bk_at_k_l177_177399
open Nat Real

theorem max_bk_at_k :
  let B_k (k : ℕ) := (choose 2000 k) * (0.1 : ℝ) ^ k
  ∃ k : ℕ, (k = 181) ∧ (∀ m : ℕ, B_k m ≤ B_k k) :=
sorry

end max_bk_at_k_l177_177399


namespace trapezoid_area_l177_177406

def trapezoid_diagonals_and_height (AC BD h : ℕ) :=
  (AC = 17) ∧ (BD = 113) ∧ (h = 15)

theorem trapezoid_area (AC BD h : ℕ) (area1 area2 : ℕ) 
  (H : trapezoid_diagonals_and_height AC BD h) :
  (area1 = 900 ∨ area2 = 780) :=
by
  sorry

end trapezoid_area_l177_177406


namespace desired_cost_per_pound_l177_177039
-- Importing the necessary library

-- Defining the candy weights and their costs per pound
def weight1 : ℝ := 20
def cost_per_pound1 : ℝ := 8
def weight2 : ℝ := 40
def cost_per_pound2 : ℝ := 5

-- Defining the proof statement
theorem desired_cost_per_pound :
  let total_cost := (weight1 * cost_per_pound1 + weight2 * cost_per_pound2)
  let total_weight := (weight1 + weight2)
  let desired_cost := total_cost / total_weight
  desired_cost = 6 := sorry

end desired_cost_per_pound_l177_177039


namespace bounded_poly_constant_l177_177325

theorem bounded_poly_constant (P : Polynomial ℤ) (B : ℕ) (h_bounded : ∀ x : ℤ, abs (P.eval x) ≤ B) : 
  P.degree = 0 :=
sorry

end bounded_poly_constant_l177_177325


namespace series_solution_l177_177187

theorem series_solution (r : ℝ) (h : (r^3 - r^2 + (1 / 4) * r - 1 = 0) ∧ r > 0) :
  (∑' (n : ℕ), (n + 1) * r^(3 * (n + 1))) = 16 * r :=
by
  sorry

end series_solution_l177_177187


namespace distinct_real_roots_find_k_values_l177_177708

-- Question 1: Prove the equation has two distinct real roots
theorem distinct_real_roots (k : ℝ) : 
  (2 * k + 1) ^ 2 - 4 * (k ^ 2 + k) > 0 :=
  by sorry

-- Question 2: Find the values of k when triangle ABC is a right triangle
theorem find_k_values (k : ℝ) : 
  (k = 3 ∨ k = 12) ↔ 
  (∃ (AB AC : ℝ), 
    AB ≠ AC ∧ AB = k ∧ AC = k + 1 ∧ (AB^2 + AC^2 = 5^2 ∨ AC^2 + 5^2 = AB^2)) :=
  by sorry

end distinct_real_roots_find_k_values_l177_177708


namespace new_table_capacity_is_six_l177_177957

-- Definitions based on the conditions
def total_tables : ℕ := 40
def extra_new_tables : ℕ := 12
def total_customers : ℕ := 212
def original_table_capacity : ℕ := 4

-- Main statement to prove
theorem new_table_capacity_is_six (O N C : ℕ) 
  (h1 : O + N = total_tables)
  (h2 : N = O + extra_new_tables)
  (h3 : O * original_table_capacity + N * C = total_customers) :
  C = 6 :=
sorry

end new_table_capacity_is_six_l177_177957


namespace tim_final_soda_cans_l177_177002

-- Definitions based on given conditions
def initialSodaCans : ℕ := 22
def cansTakenByJeff : ℕ := 6
def remainingCans (t0 j : ℕ) : ℕ := t0 - j
def additionalCansBought (remaining : ℕ) : ℕ := remaining / 2

-- Function to calculate final number of soda cans
def finalSodaCans (t0 j : ℕ) : ℕ :=
  let remaining := remainingCans t0 j
  remaining + additionalCansBought remaining

-- Theorem to prove the final number of soda cans
theorem tim_final_soda_cans : finalSodaCans initialSodaCans cansTakenByJeff = 24 :=
by
  sorry

end tim_final_soda_cans_l177_177002


namespace product_odd_integers_lt_20_l177_177357

/--
The product of all odd positive integers strictly less than 20 is a positive number ending with the digit 5.
-/
theorem product_odd_integers_lt_20 :
  let nums := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
  let product := List.prod nums
  (product > 0) ∧ (product % 10 = 5) :=
by
  sorry

end product_odd_integers_lt_20_l177_177357


namespace probability_of_picking_letter_in_mathematics_l177_177166

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_picking_letter_in_mathematics :
  (unique_letters_in_mathematics.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l177_177166


namespace gunny_bag_capacity_l177_177199

def pounds_per_ton : ℝ := 2500
def ounces_per_pound : ℝ := 16
def packets : ℝ := 2000
def packet_weight_pounds : ℝ := 16
def packet_weight_ounces : ℝ := 4

theorem gunny_bag_capacity :
  (packets * (packet_weight_pounds + packet_weight_ounces / ounces_per_pound) / pounds_per_ton) = 13 := 
by
  sorry

end gunny_bag_capacity_l177_177199


namespace matt_total_points_l177_177602

variable (n2_successful_shots : Nat) (n3_successful_shots : Nat)

def total_points (n2 : Nat) (n3 : Nat) : Nat :=
  2 * n2 + 3 * n3

theorem matt_total_points :
  total_points 4 2 = 14 :=
by
  sorry

end matt_total_points_l177_177602


namespace ananthu_can_complete_work_in_45_days_l177_177966

def amit_work_rate : ℚ := 1 / 15

def time_amit_worked : ℚ := 3

def total_work : ℚ := 1

def total_days : ℚ := 39

noncomputable def ananthu_days (x : ℚ) : Prop :=
  let amit_work_done := time_amit_worked * amit_work_rate
  let remaining_work := total_work - amit_work_done
  let ananthu_work_rate := remaining_work / (total_days - time_amit_worked)
  1 /x = ananthu_work_rate

theorem ananthu_can_complete_work_in_45_days :
  ananthu_days 45 :=
by
  sorry

end ananthu_can_complete_work_in_45_days_l177_177966


namespace rhombus_shorter_diagonal_l177_177917

variable (d1 d2 : ℝ) (Area : ℝ)

def is_rhombus (Area : ℝ) (d1 d2 : ℝ) : Prop := Area = (d1 * d2) / 2

theorem rhombus_shorter_diagonal
  (h_d2 : d2 = 20)
  (h_Area : Area = 110)
  (h_rhombus : is_rhombus Area d1 d2) :
  d1 = 11 := by
  sorry

end rhombus_shorter_diagonal_l177_177917


namespace solution_set_of_inequality_l177_177155

variable {R : Type*} [LinearOrder R] [OrderedAddCommGroup R]

def odd_function (f : R → R) := ∀ x, f (-x) = -f x

def monotonic_increasing_on (f : R → R) (s : Set R) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : odd_function f)
  (h_mono_inc : monotonic_increasing_on f (Set.Ioi 0))
  (h_f_neg1 : f (-1) = 2) : 
  {x : ℝ | 0 < x ∧ f (x-1) + 2 ≤ 0 } = Set.Ioc 1 2 :=
by
  sorry

end solution_set_of_inequality_l177_177155


namespace discount_percentage_l177_177890

theorem discount_percentage (number_of_tshirts : ℕ) (cost_per_tshirt amount_paid : ℝ)
  (h1 : number_of_tshirts = 6)
  (h2 : cost_per_tshirt = 20)
  (h3 : amount_paid = 60) : 
  ((number_of_tshirts * cost_per_tshirt - amount_paid) / (number_of_tshirts * cost_per_tshirt) * 100) = 50 := by
  -- The proof will go here
  sorry

end discount_percentage_l177_177890


namespace angle_B_eq_3pi_over_10_l177_177885

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l177_177885


namespace option_b_correct_l177_177548

variables {m n : Line} {α β : Plane}

-- Define the conditions as per the problem.
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry

theorem option_b_correct (h1 : line_perpendicular_to_plane m α)
                         (h2 : line_perpendicular_to_plane n β)
                         (h3 : lines_perpendicular m n) :
                         plane_perpendicular_to_plane α β :=
sorry

end option_b_correct_l177_177548


namespace find_x_range_l177_177420

-- Given definition for a decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x > f y

-- The main theorem to prove
theorem find_x_range (f : ℝ → ℝ) (h_decreasing : is_decreasing f) :
  {x : ℝ | f (|1 / x|) < f 1} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end find_x_range_l177_177420


namespace find_possible_values_l177_177626

def real_number_y (y : ℝ) := (3 < y ∧ y < 4)

theorem find_possible_values (y : ℝ) (h : real_number_y y) : 
  42 < (y^2 + 7*y + 12) ∧ (y^2 + 7*y + 12) < 56 := 
sorry

end find_possible_values_l177_177626


namespace question1_geometric_sequence_question2_minimum_term_l177_177550

theorem question1_geometric_sequence (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  q = 0 →
  (a 1 = 1 / 2) →
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * (r ^ n)) →
  (p = 0 ∨ p = 1) :=
by sorry

theorem question2_minimum_term (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  p = 1 →
  (a 1 = 1 / 2) →
  (a 4 = min (min (a 1) (a 2)) (a 3)) →
  3 ≤ q ∧ q ≤ 27 / 4 :=
by sorry

end question1_geometric_sequence_question2_minimum_term_l177_177550


namespace final_state_of_marbles_after_operations_l177_177178

theorem final_state_of_marbles_after_operations :
  ∃ (b w : ℕ), b + w = 2 ∧ w = 2 ∧ (∀ n : ℕ, n % 2 = 0 → n = 100 - k * 2) :=
sorry

end final_state_of_marbles_after_operations_l177_177178


namespace find_angle_B_l177_177884

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l177_177884


namespace count_valid_choices_l177_177257

open Nat

def base4_representation (N : ℕ) : ℕ := 
  let a3 := N / 64 % 4
  let a2 := N / 16 % 4
  let a1 := N / 4 % 4
  let a0 := N % 4
  64 * a3 + 16 * a2 + 4 * a1 + a0

def base7_representation (N : ℕ) : ℕ := 
  let b3 := N / 343 % 7
  let b2 := N / 49 % 7
  let b1 := N / 7 % 7
  let b0 := N % 7
  343 * b3 + 49 * b2 + 7 * b1 + b0

def S (N : ℕ) : ℕ := base4_representation N + base7_representation N

def valid_choices (N : ℕ) : Prop := 
  (S N % 100) = (2 * N % 100)

theorem count_valid_choices : 
  ∃ (count : ℕ), count = 20 ∧ ∀ (N : ℕ), (N >= 1000 ∧ N < 10000) → valid_choices N ↔ (count = 20) :=
sorry

end count_valid_choices_l177_177257


namespace find_angle_B_l177_177876

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l177_177876


namespace cos_120_degrees_l177_177071

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l177_177071


namespace triangle_sides_inequality_l177_177309

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : a + b + c ≤ 2) :
  -3 < (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) ∧ 
  (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) < 3 :=
by sorry

end triangle_sides_inequality_l177_177309


namespace fraction_simplification_l177_177363

theorem fraction_simplification : (4 * 5) / 10 = 2 := 
by 
  sorry

end fraction_simplification_l177_177363


namespace julia_tulip_count_l177_177593

def tulip_count (tulips daisies : ℕ) : Prop :=
  3 * daisies = 7 * tulips

theorem julia_tulip_count : 
  ∃ t, tulip_count t 65 ∧ t = 28 := 
by
  sorry

end julia_tulip_count_l177_177593


namespace degrees_for_cherry_pie_l177_177435

theorem degrees_for_cherry_pie
  (n c a b : ℕ)
  (hc : c = 15)
  (ha : a = 10)
  (hb : b = 9)
  (hn : n = 48)
  (half_remaining_cherry : (n - (c + a + b)) / 2 = 7) :
  (7 / 48 : ℚ) * 360 = 52.5 := 
by sorry

end degrees_for_cherry_pie_l177_177435


namespace chessboard_no_single_black_square_l177_177147

theorem chessboard_no_single_black_square :
  (∀ (repaint : (Fin 8) × Bool → (Fin 8) × Bool), False) :=
by 
  sorry

end chessboard_no_single_black_square_l177_177147


namespace geometric_series_cubes_sum_l177_177519

theorem geometric_series_cubes_sum (b s : ℝ) (h : -1 < s ∧ s < 1) :
  ∑' n : ℕ, (b * s^n)^3 = b^3 / (1 - s^3) := 
sorry

end geometric_series_cubes_sum_l177_177519


namespace negation_of_exist_prop_l177_177479

theorem negation_of_exist_prop :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by {
  sorry
}

end negation_of_exist_prop_l177_177479


namespace altitude_length_l177_177697

noncomputable def length_of_altitude (l w : ℝ) : ℝ :=
  2 * l * w / Real.sqrt (l ^ 2 + w ^ 2)

theorem altitude_length (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ∃ h : ℝ, h = length_of_altitude l w := by
  sorry

end altitude_length_l177_177697


namespace xy_maximum_value_l177_177188

theorem xy_maximum_value (x y : ℝ) (h : 3 * (x^2 + y^2) = x + 2 * y) : x - 2 * y ≤ 2 / 3 :=
sorry

end xy_maximum_value_l177_177188


namespace cosine_120_eq_negative_half_l177_177117

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l177_177117


namespace tangent_line_at_one_extreme_points_and_inequality_l177_177900

noncomputable def f (x a : ℝ) := x^2 - 2*x + a * Real.log x

-- Question 1: Tangent Line
theorem tangent_line_at_one (x a : ℝ) (h_a : a = 2) (hx_pos : x > 0) :
    2*x - Real.log x - (2*x - Real.log 1 - 1) = 0 := by
  sorry

-- Question 2: Extreme Points and Inequality
theorem extreme_points_and_inequality (a x1 x2 : ℝ) (h1 : 2*x1^2 - 2*x1 + a = 0)
    (h2 : 2*x2^2 - 2*x2 + a = 0) (hx12 : x1 < x2) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
    0 < a ∧ a < 1/2 ∧ (f x1 a) / x2 > -3/2 - Real.log 2 := by
  sorry

end tangent_line_at_one_extreme_points_and_inequality_l177_177900


namespace cos_120_eq_neg_half_l177_177088

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l177_177088


namespace minimum_value_2a_plus_b_l177_177418

theorem minimum_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (1 / (a + 1)) + (2 / (b - 2)) = 1 / 2) : 2 * a + b ≥ 16 := 
sorry

end minimum_value_2a_plus_b_l177_177418


namespace bob_repay_l177_177059

theorem bob_repay {x : ℕ} (h : 50 + 10 * x >= 150) : x >= 10 :=
by
  sorry

end bob_repay_l177_177059


namespace arrangements_count_correct_l177_177804

noncomputable def count_arrangements : ℕ :=
  -- The total number of different arrangements of students A, B, C, D in 3 communities
  -- such that each community has at least one student, and A and B are not in the same community.
  sorry

theorem arrangements_count_correct : count_arrangements = 30 := by
  sorry

end arrangements_count_correct_l177_177804


namespace perfect_square_a_i_l177_177190

theorem perfect_square_a_i (a : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1) 
  (h3 : ∀ n, a (n + 2) = 18 * a (n + 1) - a n) :
  ∀ i, ∃ k, 5 * (a i) ^ 2 - 1 = k ^ 2 :=
by
  -- The proof is missing the skipped definitions from the problem and solution context
  sorry

end perfect_square_a_i_l177_177190


namespace weight_of_B_l177_177946

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

theorem weight_of_B :
  (A + B + C) / 3 = 45 → 
  (A + B) / 2 = 40 → 
  (B + C) / 2 = 43 → 
  B = 31 :=
by
  intros h1 h2 h3
  -- detailed proof steps omitted
  sorry

end weight_of_B_l177_177946


namespace cos_120_eq_neg_one_half_l177_177074

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l177_177074


namespace polyhedron_volume_l177_177474

-- Define the polyhedron and its properties
def polyhedron (P : Type) : Prop :=
∃ (C : Type), 
  (∀ (p : P) (e : ℝ), e = 2) ∧ 
  (∃ (octFaces triFaces : ℕ), octFaces = 6 ∧ triFaces = 8) ∧
  (∀ (vol : ℝ), vol = (56 + (112 * Real.sqrt 2) / 3))
  
-- A theorem stating the volume of the polyhedron
theorem polyhedron_volume : ∀ (P : Type), polyhedron P → ∃ (vol : ℝ), vol = 56 + (112 * Real.sqrt 2) / 3 :=
by
  intros P hP
  sorry

end polyhedron_volume_l177_177474


namespace max_gcd_is_2_l177_177629

-- Define the sequence
def a (n : ℕ) : ℕ := 101 + (n + 1)^2 + 3 * n

-- Define the gcd of consecutive terms
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_is_2 : ∀ n : ℕ, n > 0 → d n = 2 :=
by
  intros n hn
  dsimp [d]
  sorry

end max_gcd_is_2_l177_177629


namespace cos_120_degrees_eq_l177_177109

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l177_177109


namespace odds_against_y_winning_l177_177852

/- 
   Define the conditions: 
   odds_w: odds against W winning is 4:1
   odds_x: odds against X winning is 5:3
-/
def odds_w : ℚ := 4 / 1
def odds_x : ℚ := 5 / 3

/- 
   Calculate the odds against Y winning 
-/
theorem odds_against_y_winning : 
  (4 / (4 + 1)) + (5 / (5 + 3)) < 1 ∧
  (1 - ((4 / (4 + 1)) + (5 / (5 + 3)))) = 17 / 40 ∧
  ((1 - (17 / 40)) / (17 / 40)) = 23 / 17 := by
  sorry

end odds_against_y_winning_l177_177852


namespace gold_initial_amount_l177_177720

theorem gold_initial_amount :
  ∃ x : ℝ, x - (x / 2 * (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6)) = 1 ∧ x = 1.2 :=
by
  existsi 1.2
  sorry

end gold_initial_amount_l177_177720


namespace logarithmic_ratio_l177_177164

theorem logarithmic_ratio (m n : ℝ) (h1 : Real.log 2 = m) (h2 : Real.log 3 = n) :
  (Real.log 12) / (Real.log 15) = (2 * m + n) / (1 - m + n) := 
sorry

end logarithmic_ratio_l177_177164


namespace sum_of_differences_is_l177_177150

-- Definitions
def a (n : ℕ) : ℚ :=
  match n with
  | 1   => 1 - 1/3
  | 2   => 1/2 - 1/4
  | _  => 1/n - 1/(n + 2)

-- Theorem statement
theorem sum_of_differences_is (sum : ℚ) :
  (sum = (Finset.sum (Finset.range 98) (λ i, a (i + 1)))) → sum = 9701 / 9900 :=
by
  -- Proof obviously goes here, but we skip it according to instructions.
  intros
  sorry

end sum_of_differences_is_l177_177150


namespace boys_neither_sport_l177_177849

theorem boys_neither_sport (Total Boys B F BF N : ℕ) (H_total : Total = 22) (H_B : B = 13) (H_F : F = 15) (H_BF : BF = 18) :
    N = Total - (B + F - BF) :=
sorry

end boys_neither_sport_l177_177849


namespace fraction_identity_l177_177565

theorem fraction_identity (a b c : ℝ) (h1 : a + b + c > 0) (h2 : a + b - c > 0) (h3 : a + c - b > 0) (h4 : b + c - a > 0) 
  (h5 : (a+b+c)/(a+b-c) = 7) (h6 : (a+b+c)/(a+c-b) = 1.75) : (a+b+c)/(b+c-a) = 3.5 :=
by
  sorry

end fraction_identity_l177_177565


namespace intersection_points_zero_l177_177835

theorem intersection_points_zero (a b c: ℝ) (h1: b^2 = a * c) (h2: a * c > 0) : 
  ∀ x: ℝ, ¬ (a * x^2 + b * x + c = 0) := 
by 
  sorry

end intersection_points_zero_l177_177835


namespace factorize_expression_l177_177539

theorem factorize_expression (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  sorry

end factorize_expression_l177_177539


namespace product_of_terms_l177_177392

theorem product_of_terms :
  (∏ n in Finset.range 11 \ Finset.singleton 0, (1 - 1 / (n + 2)^2 : ℚ)) = 13 / 24 :=
by sorry

end product_of_terms_l177_177392


namespace fraction_simplification_l177_177364

theorem fraction_simplification : (4 * 5) / 10 = 2 := 
by 
  sorry

end fraction_simplification_l177_177364


namespace distance_between_stripes_l177_177251

theorem distance_between_stripes
  (curb_distance : ℝ) (length_curb : ℝ) (stripe_length : ℝ) (distance_stripes : ℝ)
  (h1 : curb_distance = 60)
  (h2 : length_curb = 20)
  (h3 : stripe_length = 50)
  (h4 : distance_stripes = (length_curb * curb_distance) / stripe_length) :
  distance_stripes = 24 :=
by
  sorry

end distance_between_stripes_l177_177251


namespace range_of_x_in_function_l177_177583

theorem range_of_x_in_function (x : ℝ) (h : x ≠ 8) : true := sorry

end range_of_x_in_function_l177_177583


namespace exists_point_at_distance_l177_177413

def Line : Type := sorry
def Point : Type := sorry
def distance (P Q : Point) : ℝ := sorry

variables (L : Line) (d : ℝ) (P : Point)

def is_at_distance (Q : Point) (L : Line) (d : ℝ) := ∃ Q, distance Q L = d

theorem exists_point_at_distance :
  ∃ Q : Point, is_at_distance Q L d :=
sorry

end exists_point_at_distance_l177_177413


namespace find_angle_B_l177_177881

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l177_177881


namespace number_of_valid_3_digit_numbers_l177_177825

def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_3_digit_numbers_count : ℕ :=
  let digits := [(4, 8), (8, 4), (6, 6)]
  digits.length * 9

theorem number_of_valid_3_digit_numbers : valid_3_digit_numbers_count = 27 :=
by
  sorry

end number_of_valid_3_digit_numbers_l177_177825


namespace division_problem_l177_177361

theorem division_problem : (4 * 5) / 10 = 2 :=
by sorry

end division_problem_l177_177361


namespace stream_current_speed_l177_177050

theorem stream_current_speed (r w : ℝ) (h1 : 18 / (r + w) + 4 = 18 / (r - w))
  (h2 : 18 / (1.5 * r + w) + 2 = 18 / (1.5 * r - w)) : w = 2.5 :=
by
  -- Translate the equations from the problem conditions directly.
  sorry

end stream_current_speed_l177_177050


namespace intersection_P_Q_l177_177724

-- Definitions and Conditions
variable (P Q : Set ℕ)
noncomputable def f (t : ℕ) : ℕ := t ^ 2
axiom hQ : Q = {1, 4}

-- Theorem to Prove
theorem intersection_P_Q (P : Set ℕ) (Q : Set ℕ) (hQ : Q = {1, 4})
  (hf : ∀ t ∈ P, f t ∈ Q) : P ∩ Q = {1} ∨ P ∩ Q = ∅ :=
sorry

end intersection_P_Q_l177_177724


namespace particle_reaches_4_2_in_8_seconds_l177_177664

/-- The number of ways for the particle to reach the point (4, 2) after 8 seconds
    starting from the origin (0, 0) by moving either right, left, up, or down by one unit
    at the end of each second is 448. --/
theorem particle_reaches_4_2_in_8_seconds :
  ∃ (moves : List (ℕ × ℕ)), 
  (moves.length = 8) ∧ 
  (∀ move ∈ moves, move = (1, 0) ∨ move = (-1, 0) ∨ move = (0, 1) ∨ move = (0, -1)) ∧ 
  (List.sum (List.map Prod.fst moves) = 4) ∧ 
  (List.sum (List.map Prod.snd moves) = 2) ∧ 
  (number_of_ways moves = 448) :=
sorry

end particle_reaches_4_2_in_8_seconds_l177_177664


namespace prove_P_plus_V_eq_zero_l177_177172

variable (P Q R S T U V : ℤ)

-- Conditions in Lean
def sequence_conditions (P Q R S T U V : ℤ) :=
  S = 7 ∧
  P + Q + R = 27 ∧
  Q + R + S = 27 ∧
  R + S + T = 27 ∧
  S + T + U = 27 ∧
  T + U + V = 27 ∧
  U + V + P = 27

-- Assertion that needs to be proved
theorem prove_P_plus_V_eq_zero (P Q R S T U V : ℤ) (h : sequence_conditions P Q R S T U V) : 
  P + V = 0 := by
  sorry

end prove_P_plus_V_eq_zero_l177_177172


namespace interval_increasing_l177_177139

open Real

noncomputable def interval_monotonic_increasing 
  (f : ℝ → ℝ)
  (a b : ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

noncomputable def f (x : ℝ) := 2 * sin (π / 6 - 2 * x)

theorem interval_increasing : interval_monotonic_increasing f (π / 3) (5 * π / 6) :=
sorry

end interval_increasing_l177_177139


namespace lambda_range_l177_177169

theorem lambda_range (λ : ℝ) (h : 0 < λ) :
  (∀ x : ℝ, 0 < x → e^(λ * x) - log x / λ ≥ 0) → λ ≥ 1 / Real.exp 1 :=
sorry

end lambda_range_l177_177169


namespace find_line_equation_l177_177542

-- define the condition of passing through the point (-3, -1)
def passes_through (x y : ℝ) (a b : ℝ) := (a = -3) ∧ (b = -1)

-- define the condition of being parallel to the line x - 3y - 1 = 0
def is_parallel (m n c : ℝ) := (m = 1) ∧ (n = -3)

-- theorem statement
theorem find_line_equation (a b : ℝ) (c : ℝ) :
  passes_through a b (-3) (-1) →
  is_parallel 1 (-3) c →
  (a - 3 * b + c = 0) :=
sorry

end find_line_equation_l177_177542


namespace part1_part2_l177_177412

-- Define the complex number z in terms of m
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

-- State the condition where z is a purely imaginary number
def purelyImaginary (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0 ∧ m^2 - 3 * m + 2 ≠ 0

-- State the condition where z is in the second quadrant.
def inSecondQuadrant (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 < 0 ∧ m^2 - 3 * m + 2 > 0

-- Part 1: Prove that m = -1/2 given that z is purely imaginary.
theorem part1 : purelyImaginary m → m = -1/2 :=
sorry

-- Part 2: Prove the range of m for z in the second quadrant.
theorem part2 : inSecondQuadrant m → -1/2 < m ∧ m < 1 :=
sorry

end part1_part2_l177_177412


namespace number_of_pieces_correct_l177_177901

-- Define the dimensions of the pan
def pan_length : ℕ := 30
def pan_width : ℕ := 24

-- Define the dimensions of each piece of brownie
def piece_length : ℕ := 3
def piece_width : ℕ := 2

-- Calculate the area of the pan
def pan_area : ℕ := pan_length * pan_width

-- Calculate the area of each piece of brownie
def piece_area : ℕ := piece_length * piece_width

-- The proof problem statement
theorem number_of_pieces_correct : (pan_area / piece_area) = 120 :=
by sorry

end number_of_pieces_correct_l177_177901


namespace cos_120_degrees_l177_177067

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l177_177067


namespace least_xy_value_l177_177415

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
sorry

end least_xy_value_l177_177415


namespace trigonometric_identity_l177_177547

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end trigonometric_identity_l177_177547


namespace pi_times_positive_volume_difference_l177_177527

theorem pi_times_positive_volume_difference :
  let r_A := 5 / Real.pi in
  let h_A := 8 in
  let V_A := Real.pi * r_A^2 * h_A in
  let r_B := 9 / (2 * Real.pi) in
  let h_B := 7 in
  let V_B := Real.pi * r_B^2 * h_B in
  Real.pi * abs (V_B - V_A) = 58.25 := 
by {
  sorry
}

end pi_times_positive_volume_difference_l177_177527


namespace cos_120_eq_neg_half_l177_177107

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l177_177107


namespace counterexample_statement_l177_177259

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n

theorem counterexample_statement (n : ℕ) : is_composite n ∧ (is_prime (n - 3) ∨ is_prime (n - 2)) ↔ n = 22 :=
by
  sorry

end counterexample_statement_l177_177259


namespace area_of_cross_section_l177_177923

noncomputable def area_cross_section (H α : ℝ) : ℝ :=
  let AC := 2 * H * Real.sqrt 3 * Real.tan (Real.pi / 2 - α)
  let MK := (H / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2)
  (1 / 2) * AC * MK

theorem area_of_cross_section (H α : ℝ) :
  area_cross_section H α = (H^2 * Real.sqrt 3 * Real.tan (Real.pi / 2 - α) / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2) :=
sorry

end area_of_cross_section_l177_177923


namespace dad_real_age_l177_177260

theorem dad_real_age (x : ℝ) (h : (5/7) * x = 35) : x = 49 :=
by
  sorry

end dad_real_age_l177_177260


namespace cosine_120_eq_negative_half_l177_177115

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l177_177115


namespace sum_of_coeffs_l177_177808

theorem sum_of_coeffs (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 2 * (0 : ℤ))^5 = a0)
  (h2 : (1 - 2 * (1 : ℤ))^5 = a0 + a1 + a2 + a3 + a4 + a5) :
  a1 + a2 + a3 + a4 + a5 = -2 := by
  sorry

end sum_of_coeffs_l177_177808


namespace quadratic_interval_solution_l177_177137

open Set

def quadratic_function (x : ℝ) : ℝ := x^2 + 5 * x + 6

theorem quadratic_interval_solution :
  {x : ℝ | 6 ≤ quadratic_function x ∧ quadratic_function x ≤ 12} = {x | -6 ≤ x ∧ x ≤ -5} ∪ {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end quadratic_interval_solution_l177_177137


namespace complex_z24_condition_l177_177995

open Complex

theorem complex_z24_condition (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (5 * π / 180)) : 
  z^24 + z⁻¹^24 = -1 := sorry

end complex_z24_condition_l177_177995


namespace sum_integers_neg40_to_60_l177_177938

theorem sum_integers_neg40_to_60 : 
  (Finset.sum (Finset.range (60 + 40 + 1)) (λ x => x - 40)) = 1010 := sorry

end sum_integers_neg40_to_60_l177_177938


namespace decreasing_function_condition_l177_177699

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (4 * a - 1) * x + 4 * a else a ^ x

theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y ≤ f a x) ↔ (1 / 7 ≤ a ∧ a < 1 / 4) :=
by
  sorry

end decreasing_function_condition_l177_177699


namespace smaller_number_is_476_l177_177645

theorem smaller_number_is_476 (x y : ℕ) 
  (h1 : y - x = 2395) 
  (h2 : y = 6 * x + 15) : 
  x = 476 := 
by 
  sorry

end smaller_number_is_476_l177_177645


namespace correct_analogical_reasoning_l177_177224

-- Definitions of the statements in the problem
def statement_A : Prop := ∀ (a b : ℝ), a * 3 = b * 3 → a = b → a * 0 = b * 0 → a = b
def statement_B : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → (a * b) * c = a * c * b * c
def statement_C : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → c ≠ 0 → (a + b) / c = a / c + b / c
def statement_D : Prop := ∀ (a b : ℝ) (n : ℕ), (a * b)^n = a^n * b^n → (a + b)^n = a^n + b^n

-- The theorem stating that option C is the only correct analogical reasoning
theorem correct_analogical_reasoning : statement_C ∧ ¬statement_A ∧ ¬statement_B ∧ ¬statement_D := by
  sorry

end correct_analogical_reasoning_l177_177224


namespace evaluate_f_difference_l177_177840

def f (x : ℝ) : ℝ := x^6 - 2 * x^4 + 7 * x

theorem evaluate_f_difference :
  f 3 - f (-3) = 42 := by
  sorry

end evaluate_f_difference_l177_177840


namespace dad_steps_90_l177_177647

/-- 
  Given:
  - When Dad takes 3 steps, Masha takes 5 steps.
  - When Masha takes 3 steps, Yasha takes 5 steps.
  - Masha and Yasha together made a total of 400 steps.

  Prove: 
  The number of steps that Dad took is 90.
-/
theorem dad_steps_90 (total_steps: ℕ) (masha_to_dad_ratio: ℕ) (yasha_to_masha_ratio: ℕ) (steps_masha_yasha: ℕ) (h1: masha_to_dad_ratio = 5) (h2: yasha_to_masha_ratio = 5) (h3: steps_masha_yasha = 400) :
  total_steps = 90 :=
by
  sorry

end dad_steps_90_l177_177647


namespace max_AMC_expression_l177_177313

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 24) :
  A * M * C + A * M + M * C + C * A ≤ 704 :=
sorry

end max_AMC_expression_l177_177313


namespace max_value_expr_l177_177896

theorem max_value_expr (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (hxyz : x + y + z = 3) :
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (x - y + z) ≤ 2187 / 216 :=
sorry

end max_value_expr_l177_177896


namespace regular_nonagon_interior_angle_l177_177765

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

def regular_polygon_interior_angle (n : ℕ) : ℝ := sum_of_interior_angles n / n

theorem regular_nonagon_interior_angle :
  regular_polygon_interior_angle 9 = 140 :=
by
  unfold regular_polygon_interior_angle
  unfold sum_of_interior_angles
  norm_num
  sorry

end regular_nonagon_interior_angle_l177_177765


namespace min_value_of_a_l177_177998

-- Defining the properties of the function f
variable {f : ℝ → ℝ}
variable (even_f : ∀ x, f x = f (-x))
variable (mono_f : ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Necessary condition involving f and a
variable {a : ℝ}
variable (a_condition : f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1)

-- Main statement proving that the minimum value of a is 1/2
theorem min_value_of_a : a = 1/2 :=
sorry

end min_value_of_a_l177_177998


namespace function_domain_exclusion_l177_177586

theorem function_domain_exclusion (x : ℝ) :
  (∃ y, y = 2 / (x - 8)) ↔ x ≠ 8 :=
sorry

end function_domain_exclusion_l177_177586


namespace proposition_relation_l177_177709

theorem proposition_relation :
  (∀ (x : ℝ), x < 3 → x < 5) ↔ (∀ (x : ℝ), x ≥ 5 → x ≥ 3) :=
by
  sorry

end proposition_relation_l177_177709


namespace a5_value_l177_177452

def sequence_sum (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem a5_value (a : ℕ → ℤ) (h : ∀ n : ℕ, 0 < n → sequence_sum n a = (1 / 2 : ℚ) * (a n : ℚ) + 1) :
  a 5 = 2 := by
  sorry

end a5_value_l177_177452


namespace angle_B_eq_3pi_over_10_l177_177886

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l177_177886


namespace minimum_value_of_fraction_l177_177165

theorem minimum_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (4 / a + 9 / b) ≥ 25 :=
by
  sorry

end minimum_value_of_fraction_l177_177165


namespace solve_for_x_l177_177610

noncomputable def x_solution : Real :=
  (Real.log2 3) / 2

theorem solve_for_x :
  ∀ x : Real, (2 ^ (8 ^ x) = 8 ^ (2 ^ x)) ↔ x = x_solution :=
by
  sorry

end solve_for_x_l177_177610


namespace cos_120_eq_neg_half_l177_177079

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177079


namespace circle_equation_l177_177208

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * y = 0

theorem circle_equation
  (x y : ℝ)
  (center_on_y_axis : ∃ r : ℝ, r > 0 ∧ x^2 + (y - r)^2 = r^2)
  (tangent_to_x_axis : ∃ r : ℝ, r > 0 ∧ y = r)
  (passes_through_point : x = 3 ∧ y = 1) :
  equation_of_circle x y :=
by
  sorry

end circle_equation_l177_177208


namespace xy_max_value_l177_177807

theorem xy_max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 12) :
  xy <= 9 := by
  sorry

end xy_max_value_l177_177807


namespace triangle_angle_B_l177_177871

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l177_177871


namespace multiple_choice_question_count_l177_177528

theorem multiple_choice_question_count (n : ℕ) : 
  (4 * 224 / (2^4 - 2) = 4^2) → n = 2 := 
by
  sorry

end multiple_choice_question_count_l177_177528


namespace average_remaining_two_numbers_l177_177512

theorem average_remaining_two_numbers 
    (a b c d e f : ℝ)
    (h1 : (a + b + c + d + e + f) / 6 = 3.95)
    (h2 : (a + b) / 2 = 4.4)
    (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 3.6 := 
sorry

end average_remaining_two_numbers_l177_177512


namespace rs_value_l177_177319

theorem rs_value (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 2) (h4 : r^4 + s^4 = 15 / 8) :
  r * s = (Real.sqrt 17) / 4 := 
sorry

end rs_value_l177_177319


namespace sequence_and_sum_problems_l177_177283

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n-1) * d) / 2

def geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := b * r^(n-1)

noncomputable def sum_geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := 
(if r = 1 then b * n
 else b * (r^n - 1) / (r - 1))

theorem sequence_and_sum_problems :
  (∀ n : ℕ, arithmetic_sequence 19 (-2) n = 21 - 2 * n) ∧
  (∀ n : ℕ, sum_arithmetic_sequence 19 (-2) n = 20 * n - n^2) ∧
  (∀ n : ℕ, ∃ a_n : ℤ, (geometric_sequence 1 3 n + (a_n - geometric_sequence 1 3 n) = 21 - 2 * n + 3^(n-1)) ∧
    sum_geometric_sequence 1 3 n = (sum_arithmetic_sequence 19 (-2) n + (3^n - 1) / 2))
:= by
  sorry

end sequence_and_sum_problems_l177_177283


namespace cos_120_eq_neg_half_l177_177099

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l177_177099


namespace weight_and_ratio_of_spheres_l177_177348

-- Defining constants and conditions
def surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2
def weight_dependent_on_surface_area (weight1 weight2 surface_area1 surface_area2 : ℝ) : Prop :=
  weight1 / surface_area1 = weight2 / surface_area2

-- Specific Properties
def weight_of_radius_015_weighs_8g : Prop :=
  weight_dependent_on_surface_area 8 (surface_area 0.15) (surface_area 0.3)

def weight_of_radius_03 : ℝ := 32
def weight_of_radius_06 : ℝ := 128

def weight_ratio (w1 w2 w3 : ℝ) (r : ℝ) : Prop :=
  (w1 : ℝ) / r = (w2 / (4 * r)) ∧ (w2 / (4 * r) = w3 / (16 * r))

-- Theorem to prove
theorem weight_and_ratio_of_spheres :
  weight_of_radius_015_weighs_8g ∧ 
  weight_dependent_on_surface_area 8 32 (surface_area 0.15) (surface_area 0.3) ∧
  weight_dependent_on_surface_area 8 128 (surface_area 0.15) (surface_area 0.6) ∧
  weight_ratio 8 32 128 8 :=
by sorry -- Proof is omitted

end weight_and_ratio_of_spheres_l177_177348


namespace fourth_hexagon_dots_l177_177973

   -- Define the number of dots in the first, second, and third hexagons
   def hexagon_dots (n : ℕ) : ℕ :=
     match n with
     | 1 => 1
     | 2 => 8
     | 3 => 22
     | 4 => 46
     | _ => 0

   -- State the theorem to be proved
   theorem fourth_hexagon_dots : hexagon_dots 4 = 46 :=
   by
     sorry
   
end fourth_hexagon_dots_l177_177973


namespace speed_of_train_l177_177383

theorem speed_of_train (length : ℝ) (time : ℝ) (conversion_factor : ℝ) (speed_kmh : ℝ) 
  (h1 : length = 240) (h2 : time = 16) (h3 : conversion_factor = 3.6) :
  speed_kmh = (length / time) * conversion_factor := 
sorry

end speed_of_train_l177_177383


namespace product_of_odd_and_even_is_odd_l177_177713

theorem product_of_odd_and_even_is_odd {f g : ℝ → ℝ} 
  (hf : ∀ x : ℝ, f (-x) = -f x)
  (hg : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x) * (g x) = -(f (-x) * g (-x)) :=
by
  sorry

end product_of_odd_and_even_is_odd_l177_177713


namespace volume_limit_l177_177446

open Real

noncomputable def volume_of_solid (n : ℕ) : ℝ :=
  let VC := π * ((n + 1)^2 / 2 - n^2 / 2)
  let vl := π * (1 / (2 * sqrt n) * ∫ x in n..n+1, (1 / (2 * sqrt n) * x + sqrt n / 2)^2)
  VC - vl

theorem volume_limit (V : ℕ → ℝ) (a b : ℝ) :
  (∀ n, V n = volume_of_solid n) →
  (a = 0 ∧ b = π / 6) →
  filter.tendsto (fun n => n^a * V n) filter.at_top (nhds b) :=
begin
  intros hV h_ab,
  rcases h_ab with ⟨ha, hb⟩, 
  simp only [ha, pow_zero, one_mul], 
  sorry,
end

end volume_limit_l177_177446


namespace max_total_weight_l177_177205

-- Definitions
def A_max_weight := 5
def E_max_weight := 2 * A_max_weight
def total_swallows := 90
def A_to_E_ratio := 2

-- Main theorem statement
theorem max_total_weight :
  ∃ A E, (A = A_to_E_ratio * E) ∧ (A + E = total_swallows) ∧ ((A * A_max_weight + E * E_max_weight) = 600) :=
  sorry

end max_total_weight_l177_177205


namespace area_half_l177_177408

theorem area_half (width height : ℝ) (h₁ : width = 25) (h₂ : height = 16) :
  (width * height) / 2 = 200 :=
by
  -- The formal proof is skipped here
  sorry

end area_half_l177_177408


namespace min_value_a_plus_b_plus_c_l177_177813

theorem min_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 9 * a + 4 * b = a * b * c) : a + b + c ≥ 10 :=
by
  sorry

end min_value_a_plus_b_plus_c_l177_177813


namespace common_positive_divisors_count_l177_177830

-- To use noncomputable functions
noncomputable theory

open Nat

-- Define the two numbers
def num1 : ℕ := 9240
def num2 : ℕ := 13860

-- Define their greatest common divisor
def gcd_val : ℕ := gcd num1 num2

-- State the prime factorization of the gcd (this can be proven or assumed as a given condition for cleaner code)
def prime_factors_gcd := [(2, 2), (3, 1), (7, 1), (11, 1)]

-- Given the prime factorization, calculate the number of divisors
def number_of_divisors : ℕ := 
  prime_factors_gcd.foldr (λ (factor : ℕ × ℕ) acc, acc * (factor.snd + 1)) 1

-- The final theorem stating the number of common positive divisors of num1 and num2
theorem common_positive_divisors_count : number_of_divisors = 24 := by {
  -- Here would go the proof, which is not required in this task
  sorry
}

end common_positive_divisors_count_l177_177830


namespace smallest_positive_period_of_f_f_of_2alpha_l177_177707

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos (x - (Real.pi / 3)) - Real.sin ((Real.pi / 2) - x)

theorem smallest_positive_period_of_f :
  ∀ x, f (x + (2 * Real.pi)) = f x :=
by
  sorry

theorem f_of_2alpha (α : ℝ) (h₀ : 0 < α) (h₁ : α < Real.pi / 2) (h₂ : f (α + Real.pi / 6) = 3 / 5) :
  f (2 * α) = (24 * Real.sqrt 3 - 7) / 50 :=
by
  sorry

end smallest_positive_period_of_f_f_of_2alpha_l177_177707


namespace angle_B_in_triangle_l177_177867

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l177_177867


namespace unique_zero_function_l177_177272

variable (f : ℝ → ℝ)

theorem unique_zero_function (h : ∀ x y : ℝ, f (x + y) = f x - f y) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end unique_zero_function_l177_177272


namespace carol_allowance_problem_l177_177537

open Real

theorem carol_allowance_problem (w : ℝ) 
  (fixed_allowance : ℝ := 20) 
  (extra_earnings_per_week : ℝ := 22.5) 
  (total_money : ℝ := 425) :
  fixed_allowance * w + extra_earnings_per_week * w = total_money → w = 10 :=
by
  intro h
  -- Proof skipped
  sorry

end carol_allowance_problem_l177_177537


namespace selling_price_correct_l177_177055

def meters_of_cloth : ℕ := 45
def profit_per_meter : ℝ := 12
def cost_price_per_meter : ℝ := 88
def total_selling_price : ℝ := 4500

theorem selling_price_correct :
  (cost_price_per_meter * meters_of_cloth) + (profit_per_meter * meters_of_cloth) = total_selling_price :=
by
  sorry

end selling_price_correct_l177_177055


namespace probability_and_relationship_l177_177027

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l177_177027


namespace value_added_to_each_number_is_12_l177_177306

theorem value_added_to_each_number_is_12
    (sum_original : ℕ)
    (sum_new : ℕ)
    (n : ℕ)
    (avg_original : ℕ)
    (avg_new : ℕ)
    (value_added : ℕ) :
  (n = 15) →
  (avg_original = 40) →
  (avg_new = 52) →
  (sum_original = n * avg_original) →
  (sum_new = n * avg_new) →
  (value_added = (sum_new - sum_original) / n) →
  value_added = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end value_added_to_each_number_is_12_l177_177306


namespace maximum_area_of_equilateral_triangle_in_rectangle_l177_177630

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  (953 * Real.sqrt 3) / 16

theorem maximum_area_of_equilateral_triangle_in_rectangle :
  ∀ (a b : ℕ), a = 13 → b = 14 → maxEquilateralTriangleArea a b = (953 * Real.sqrt 3) / 16 :=
by
  intros a b h₁ h₂
  rw [h₁, h₂]
  apply rfl

end maximum_area_of_equilateral_triangle_in_rectangle_l177_177630


namespace radius_of_circle_eq_zero_l177_177273

theorem radius_of_circle_eq_zero :
  ∀ x y: ℝ, (x^2 + 8 * x + y^2 - 10 * y + 41 = 0) → (0 : ℝ) = 0 :=
by
  intros x y h
  sorry

end radius_of_circle_eq_zero_l177_177273


namespace min_value_of_sequence_l177_177810

theorem min_value_of_sequence :
  ∃ (a : ℕ → ℤ), a 1 = 0 ∧ (∀ n : ℕ, n ≥ 2 → |a n| = |a (n - 1) + 1|) ∧ (a 1 + a 2 + a 3 + a 4 = -2) :=
by
  sorry

end min_value_of_sequence_l177_177810


namespace increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l177_177159

noncomputable def f (x : ℝ) := x ^ 2 * Real.exp x - Real.log x

theorem increasing_f_for_x_ge_1 : ∀ (x : ℝ), x ≥ 1 → ∀ y > x, f y > f x :=
by
  sorry

theorem f_gt_1_for_x_gt_0 : ∀ (x : ℝ), x > 0 → f x > 1 :=
by
  sorry

end increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l177_177159


namespace train_initial_speed_l177_177056

theorem train_initial_speed (x : ℝ) (h : 3 * 25 * (x / V + (2 * x / 20)) = 3 * x) : V = 50 :=
  by
  sorry

end train_initial_speed_l177_177056


namespace minimum_value_of_l177_177728

noncomputable def minimum_value (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem minimum_value_of (x y z : ℝ) (hxyz : x > 0 ∧ y > 0 ∧ z > 0) (h : 1/x + 1/y + 1/z = 9) :
  minimum_value x y z = 1 / 3456 := 
sorry

end minimum_value_of_l177_177728


namespace find_x_values_l177_177978

-- Defining the given condition as a function
def equation (x : ℝ) : Prop :=
  (4 / (Real.sqrt (x + 5) - 7)) +
  (3 / (Real.sqrt (x + 5) - 2)) +
  (6 / (Real.sqrt (x + 5) + 2)) +
  (9 / (Real.sqrt (x + 5) + 7)) = 0

-- Statement of the theorem in Lean
theorem find_x_values :
  equation ( -796 / 169) ∨ equation (383 / 22) :=
sorry

end find_x_values_l177_177978


namespace triangle_angle_B_l177_177870

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l177_177870


namespace geometric_sequence_sum_l177_177855

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 1 = 3)
    (h2 : a 4 = 24)
    (hn : ∀ n, a n = a 1 * q ^ (n - 1)) :
    (a 3 + a 4 + a 5 = 84) :=
by
  -- Proof will go here
  sorry

end geometric_sequence_sum_l177_177855


namespace find_remainder_l177_177726

theorem find_remainder (y : ℕ) (hy : 7 * y % 31 = 1) : (17 + 2 * y) % 31 = 4 :=
sorry

end find_remainder_l177_177726


namespace probability_below_8_l177_177960

theorem probability_below_8 (p10 p9 p8 : ℝ) (h1 : p10 = 0.20) (h2 : p9 = 0.30) (h3 : p8 = 0.10) : 
  1 - (p10 + p9 + p8) = 0.40 :=
by 
  rw [h1, h2, h3]
  sorry

end probability_below_8_l177_177960


namespace sequence_a_n_l177_177587

theorem sequence_a_n (a : ℤ) (h : (-1)^1 * 1 + a + (-1)^4 * 4 + a = 3 * ( (-1)^2 * 2 + a )) :
  a = -3 ∧ ((-1)^100 * 100 + a) = 97 :=
by
  sorry  -- proof is omitted

end sequence_a_n_l177_177587


namespace compute_expression_l177_177127

theorem compute_expression : 7^2 - 2 * 6 + (3^2 - 1) = 45 :=
by
  sorry

end compute_expression_l177_177127


namespace arithmetic_sequence_sum_first_nine_terms_l177_177994

noncomputable def geometric_sequence (n : ℕ) : ℝ := sorry

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := sorry

axiom b_seq : (n : ℕ) → Prop
  | 5 := geometric_sequence 5 = 2
  | _ := ∀ m, m < 5 ∨ m > 5 → geometric_sequence m ≠ 2

axiom a_seq : (n : ℕ) → Prop
  | 5 := arithmetic_sequence 5 = 2
  | _ := ∀ m, m < 5 ∨ m > 5 → arithmetic_sequence m ≠ 2

theorem arithmetic_sequence_sum_first_nine_terms :
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) (a_seq : ℕ → ℝ) 
  (h5 : a_seq 5 = 2) :
  a_seq 1 + a_seq 2 + a_seq 3 + a_seq 4 + a_seq 5 + a_seq 6 + a_seq 7 + a_seq 8 + a_seq 9 = 36 :=
by
  sorry

end arithmetic_sequence_sum_first_nine_terms_l177_177994


namespace probability_alice_after_bob_given_bob_before_45_is_9_over_16_l177_177782

noncomputable def probability_alice_after_bob_and_bob_before_45 : ℝ :=
  let total_area : ℝ := 0.5 * 60 * 60
  let restricted_area : ℝ := 0.5 * 45 * 45
  restricted_area / total_area

theorem probability_alice_after_bob_given_bob_before_45_is_9_over_16 :
  probability_alice_after_bob_and_bob_before_45 = 9 / 16 :=
sorry

end probability_alice_after_bob_given_bob_before_45_is_9_over_16_l177_177782


namespace geometric_series_sum_l177_177687

theorem geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 16383 / 49152 :=
by
  sorry

end geometric_series_sum_l177_177687


namespace cevians_concurrent_circumscribable_l177_177729

-- Define the problem
variables {A B C D X Y Z : Type}

-- Define concurrent cevians
def cevian_concurrent (A B C X Y Z D : Type) : Prop := true

-- Define circumscribable quadrilaterals
def circumscribable (A B C D : Type) : Prop := true

-- The theorem statement
theorem cevians_concurrent_circumscribable (h_conc: cevian_concurrent A B C X Y Z D) 
(h1: circumscribable D Y A Z) (h2: circumscribable D Z B X) : circumscribable D X C Y :=
sorry

end cevians_concurrent_circumscribable_l177_177729


namespace find_constants_l177_177277

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
if x < 3 then a * x^2 + b else 10 - 2 * x

theorem find_constants (a b : ℝ)
  (H : ∀ x, f a b (f a b x) = x) :
  a + b = 13 / 3 := by 
  sorry

end find_constants_l177_177277


namespace measure_of_angle_A_value_of_a_l177_177299

open Real

variables (a b c : ℝ)
variables (A B C : ℝ) [fact (0 < A)] [fact (A < π)] [fact (0 < B)] [fact (B < π)] [fact (0 < C)] [fact (C < π)]
variables (h_area : 0.5 * b * c * sin (A) = sqrt 3 / 4)
variables (h_cos_eq : 2 * cos A * (c * cos B + b * cos C) = a)
variables (h_constr : c^2 + a * b * cos C + a^2 = 4)

theorem measure_of_angle_A : A = π / 3 := sorry

theorem value_of_a (hA : A = π / 3) : a = sqrt 21 / 3 := sorry

end measure_of_angle_A_value_of_a_l177_177299


namespace min_value_of_ellipse_l177_177705

noncomputable def min_m_plus_n (a b : ℝ) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) : ℝ :=
(a ^ (2/3) + b ^ (2/3)) ^ (3/2)

theorem min_value_of_ellipse (m n a b : ℝ) (h1 : m > n) (h2 : n > 0) (h_ellipse : (a^2 / m^2) + (b^2 / n^2) = 1) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) :
  (m + n) = min_m_plus_n a b h_ab_nonzero h_abs_diff :=
sorry

end min_value_of_ellipse_l177_177705


namespace maximum_term_of_sequence_l177_177552

open Real

noncomputable def seq (n : ℕ) : ℝ := n / (n^2 + 81)

theorem maximum_term_of_sequence : ∃ n : ℕ, seq n = 1 / 18 ∧ ∀ k : ℕ, seq k ≤ 1 / 18 :=
by
  sorry

end maximum_term_of_sequence_l177_177552


namespace percentage_deducted_from_list_price_l177_177654

noncomputable def cost_price : ℝ := 47.50
noncomputable def list_price : ℝ := 65.97
noncomputable def selling_price : ℝ := 65.97
noncomputable def required_profit_percent : ℝ := 25

theorem percentage_deducted_from_list_price :
  let desired_selling_price := cost_price * (1 + required_profit_percent / 100)
  let discount_percentage := 100 * (1 - desired_selling_price / list_price)
  discount_percentage = 10.02 :=
by
  sorry

end percentage_deducted_from_list_price_l177_177654


namespace students_exceed_hamsters_l177_177969

-- Definitions corresponding to the problem conditions
def students_per_classroom : ℕ := 20
def hamsters_per_classroom : ℕ := 1
def number_of_classrooms : ℕ := 5

-- Lean 4 statement to express the problem
theorem students_exceed_hamsters :
  (students_per_classroom * number_of_classrooms) - (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end students_exceed_hamsters_l177_177969


namespace cos_120_eq_neg_half_l177_177083

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177083


namespace real_roots_iff_le_one_l177_177168

theorem real_roots_iff_le_one (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) → k ≤ 1 :=
by
  sorry

end real_roots_iff_le_one_l177_177168


namespace positive_m_for_one_root_l177_177291

theorem positive_m_for_one_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
by
  sorry

end positive_m_for_one_root_l177_177291


namespace solid_is_cone_l177_177430

-- Define what it means for a solid to have a given view as an isosceles triangle or a circle.
structure Solid :=
(front_view : ℝ → ℝ → Prop)
(left_view : ℝ → ℝ → Prop)
(top_view : ℝ → ℝ → Prop)

-- Definition of isosceles triangle view
def isosceles_triangle (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Definition of circle view with a center
def circle_with_center (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Define the solid that satisfies the conditions in the problem
def specified_solid (s : Solid) : Prop :=
  (∀ x y, s.front_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.left_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.top_view x y → circle_with_center x y)

-- Given proof problem statement
theorem solid_is_cone (s : Solid) (h : specified_solid s) : 
  ∃ cone, cone = s :=
sorry

end solid_is_cone_l177_177430


namespace villager4_truth_teller_l177_177268

def villager1_statement (liars : Finset ℕ) : Prop := liars = {0, 1, 2, 3}
def villager2_statement (liars : Finset ℕ) : Prop := liars.card = 1
def villager3_statement (liars : Finset ℕ) : Prop := liars.card = 2
def villager4_statement (liars : Finset ℕ) : Prop := 3 ∉ liars

theorem villager4_truth_teller (liars : Finset ℕ) :
  ¬ villager1_statement liars ∧
  ¬ villager2_statement liars ∧
  ¬ villager3_statement liars ∧
  villager4_statement liars ↔
  liars = {0, 1, 2} :=
by
  sorry

end villager4_truth_teller_l177_177268


namespace prob1_prob2_l177_177536

-- Problem 1
theorem prob1 (x y : ℝ) : 3 * x^2 * y * (-2 * x * y)^3 = -24 * x^5 * y^4 :=
sorry

-- Problem 2
theorem prob2 (x y : ℝ) : (5 * x + 2 * y) * (3 * x - 2 * y) = 15 * x^2 - 4 * x * y - 4 * y^2 :=
sorry

end prob1_prob2_l177_177536


namespace height_of_sarah_building_l177_177761

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

end height_of_sarah_building_l177_177761


namespace cos_120_eq_neg_half_l177_177094

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l177_177094


namespace sum_first_n_terms_l177_177848

-- Define the sequence a_n
def geom_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n, a (n + 1) = r * a n

-- Define the main conditions from the problem
axiom a7_cond (a : ℕ → ℕ) : a 7 = 8 * a 4
axiom arithmetic_seq_cond (a : ℕ → ℕ) : (1 / 2 : ℝ) * a 2 < (a 3 - 4) ∧ (a 3 - 4) < (a 4 - 12)

-- Define the sequences a_n and b_n using the conditions
def a_n (n : ℕ) : ℕ := 2^(n + 1)
def b_n (n : ℕ) : ℤ := (-1)^n * (Int.ofNat (n + 1))

-- Define the sum of the first n terms of b_n
noncomputable def T_n (n : ℕ) : ℤ :=
  (Finset.range n).sum b_n

-- Main theorem statement
theorem sum_first_n_terms (k : ℕ) : |T_n k| = 20 → k = 40 ∨ k = 37 :=
sorry

end sum_first_n_terms_l177_177848


namespace binomial_parameters_l177_177818

theorem binomial_parameters
  (n : ℕ) (p : ℚ)
  (hE : n * p = 12) (hD : n * p * (1 - p) = 2.4) :
  n = 15 ∧ p = 4 / 5 :=
by
  sorry

end binomial_parameters_l177_177818


namespace solve_for_r_l177_177202

theorem solve_for_r (r : ℝ) (h: (r + 9) / (r - 3) = (r - 2) / (r + 5)) : r = -39 / 19 :=
sorry

end solve_for_r_l177_177202


namespace relationship_P_Q_l177_177426

variable (a : ℝ)
variable (P : ℝ := Real.sqrt a + Real.sqrt (a + 5))
variable (Q : ℝ := Real.sqrt (a + 2) + Real.sqrt (a + 3))

theorem relationship_P_Q (h : 0 ≤ a) : P < Q :=
by
  sorry

end relationship_P_Q_l177_177426


namespace total_cost_smore_night_l177_177604

-- Define the costs per item
def cost_graham_cracker : ℝ := 0.10
def cost_marshmallow : ℝ := 0.15
def cost_chocolate : ℝ := 0.25
def cost_caramel_piece : ℝ := 0.20
def cost_toffee_piece : ℝ := 0.05

-- Calculate the cost for each ingredient per S'more
def cost_caramel : ℝ := 2 * cost_caramel_piece
def cost_toffee : ℝ := 4 * cost_toffee_piece

-- Total cost of one S'more
def cost_one_smore : ℝ :=
  cost_graham_cracker + cost_marshmallow + cost_chocolate + cost_caramel + cost_toffee

-- Number of people and S'mores per person
def num_people : ℕ := 8
def smores_per_person : ℕ := 3

-- Total number of S'mores
def total_smores : ℕ := num_people * smores_per_person

-- Total cost of all the S'mores
def total_cost : ℝ := total_smores * cost_one_smore

-- The final statement
theorem total_cost_smore_night : total_cost = 26.40 := 
  sorry

end total_cost_smore_night_l177_177604


namespace fraction_multiplication_l177_177495

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l177_177495


namespace area_inside_circle_outside_square_is_zero_l177_177249

theorem area_inside_circle_outside_square_is_zero 
  (side_length : ℝ) (circle_radius : ℝ)
  (h_square_side : side_length = 2) (h_circle_radius : circle_radius = 1) : 
  (π * circle_radius^2) - (side_length^2) = 0 := 
by 
  sorry

end area_inside_circle_outside_square_is_zero_l177_177249


namespace sum_of_perpendiculars_l177_177233

-- define the points on the rectangle
variables {A B C D P S R Q F : Type}

-- define rectangle ABCD and points P, S, R, Q, F
def is_rectangle (A B C D : Type) : Prop := sorry -- conditions for ABCD to be a rectangle
def point_on_segment (P A B: Type) : Prop := sorry -- P is a point on segment AB
def perpendicular (X Y Z : Type) : Prop := sorry -- definition for perpendicular between two segments
def length (X Y : Type) : ℝ := sorry -- definition for the length of a segment

-- Given conditions
axiom rect : is_rectangle A B C D
axiom p_on_ab : point_on_segment P A B
axiom ps_perp_bd : perpendicular P S D
axiom pr_perp_ac : perpendicular P R C
axiom af_perp_bd : perpendicular A F D
axiom pq_perp_af : perpendicular P Q F

-- Prove that PR + PS = AF
theorem sum_of_perpendiculars :
  length P R + length P S = length A F :=
sorry

end sum_of_perpendiculars_l177_177233


namespace video_game_cost_l177_177690

theorem video_game_cost
  (weekly_allowance1 : ℕ)
  (weeks1 : ℕ)
  (weekly_allowance2 : ℕ)
  (weeks2 : ℕ)
  (money_spent_on_clothes_fraction : ℚ)
  (remaining_money : ℕ)
  (allowance1 : weekly_allowance1 = 5)
  (duration1 : weeks1 = 8)
  (allowance2 : weekly_allowance2 = 6)
  (duration2 : weeks2 = 6)
  (money_spent_fraction : money_spent_on_clothes_fraction = 1/2)
  (remaining_money_condition : remaining_money = 3) :
  (weekly_allowance1 * weeks1 + weekly_allowance2 * weeks2) * (1 - money_spent_on_clothes_fraction) - remaining_money = 35 :=
by
  rw [allowance1, duration1, allowance2, duration2, money_spent_fraction, remaining_money_condition]
  -- Calculation steps are omitted; they can be filled in here.
  exact sorry

end video_game_cost_l177_177690


namespace find_fraction_l177_177561

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h1 : (a + b + c) / (a + b - c) = 7)
variable (h2 : (a + b + c) / (a + c - b) = 1.75)

theorem find_fraction : (a + b + c) / (b + c - a) = 3.5 := 
by {
  sorry
}

end find_fraction_l177_177561


namespace rectangle_width_l177_177048

theorem rectangle_width (L : ℚ) (A : ℚ) (W : ℚ) (hL : L = 3 / 5) (hA : A = 1 / 3) (hAW : A = L * W) : W = 5 / 9 :=
by
  subst hL
  subst hA
  simp at hAW
  sorry

end rectangle_width_l177_177048


namespace functional_equation_solution_l177_177541

theorem functional_equation_solution (f : ℕ → ℕ) 
  (H : ∀ a b : ℕ, f (f a + f b) = a + b) : 
  ∀ n : ℕ, f n = n := 
by
  sorry

end functional_equation_solution_l177_177541


namespace distance_travelled_first_hour_l177_177483

noncomputable def initial_distance (x : ℕ) : Prop :=
  let distance_travelled := (12 / 2) * (2 * x + (12 - 1) * 2)
  distance_travelled = 552

theorem distance_travelled_first_hour : ∃ x : ℕ, initial_distance x ∧ x = 35 :=
by
  use 35
  unfold initial_distance
  sorry

end distance_travelled_first_hour_l177_177483


namespace weekend_price_of_coat_l177_177742

-- Definitions based on conditions
def original_price : ℝ := 250
def sale_price_discount : ℝ := 0.4
def weekend_additional_discount : ℝ := 0.3

-- To prove the final weekend price
theorem weekend_price_of_coat :
  (original_price * (1 - sale_price_discount) * (1 - weekend_additional_discount)) = 105 := by
  sorry

end weekend_price_of_coat_l177_177742


namespace ratio_of_areas_of_squares_l177_177345

theorem ratio_of_areas_of_squares (a_side b_side : ℕ) (h_a : a_side = 36) (h_b : b_side = 42) : 
  (a_side ^ 2 : ℚ) / (b_side ^ 2 : ℚ) = 36 / 49 :=
by
  sorry

end ratio_of_areas_of_squares_l177_177345


namespace area_per_car_l177_177304

/-- Given the length and width of the parking lot, 
and the percentage of usable area, 
and the number of cars that can be parked,
prove that the area per car is as expected. -/
theorem area_per_car 
  (length width : ℝ) 
  (usable_percentage : ℝ) 
  (number_of_cars : ℕ) 
  (h_length : length = 400) 
  (h_width : width = 500) 
  (h_usable_percentage : usable_percentage = 0.80) 
  (h_number_of_cars : number_of_cars = 16000) :
  (length * width * usable_percentage) / number_of_cars = 10 :=
by
  sorry

end area_per_car_l177_177304


namespace total_listening_days_l177_177457

-- Definitions
variables {x y z t : ℕ}

-- Problem statement
theorem total_listening_days (x y z t : ℕ) : (x + y + z) * t = ((x + y + z) * t) :=
by sorry

end total_listening_days_l177_177457


namespace cos_120_eq_neg_half_l177_177086

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l177_177086


namespace distance_point_to_line_zero_or_four_l177_177613

theorem distance_point_to_line_zero_or_four {b : ℝ} 
(h : abs (b - 2) / Real.sqrt 2 = Real.sqrt 2) : 
b = 0 ∨ b = 4 := 
sorry

end distance_point_to_line_zero_or_four_l177_177613


namespace total_balloons_after_destruction_l177_177691

-- Define the initial numbers of balloons
def fredBalloons := 10.0
def samBalloons := 46.0
def destroyedBalloons := 16.0

-- Prove the total number of remaining balloons
theorem total_balloons_after_destruction : fredBalloons + samBalloons - destroyedBalloons = 40.0 :=
by
  sorry

end total_balloons_after_destruction_l177_177691


namespace find_k_l177_177907

noncomputable def a : ℚ := sorry -- Represents positive rational number a
noncomputable def b : ℚ := sorry -- Represents positive rational number b

def minimal_period (x : ℚ) : ℕ := sorry -- Function to determine minimal period of a rational number

-- Conditions as definitions
axiom h1 : minimal_period a = 30
axiom h2 : minimal_period b = 30
axiom h3 : minimal_period (a - b) = 15

-- Statement to prove smallest natural number k such that minimal period of (a + k * b) is 15
theorem find_k : ∃ k : ℕ, minimal_period (a + k * b) = 15 ∧ ∀ n < k, minimal_period (a + n * b) ≠ 15 :=
sorry

end find_k_l177_177907


namespace cos_120_eq_neg_half_l177_177122

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177122


namespace grey_eyes_black_hair_l177_177717

-- Definitions based on conditions
def num_students := 60
def num_black_hair := 36
def num_green_eyes_red_hair := 20
def num_grey_eyes := 24

-- Calculate number of students with red hair
def num_red_hair := num_students - num_black_hair

-- Calculate number of grey-eyed students with red hair
def num_grey_eyes_red_hair := num_red_hair - num_green_eyes_red_hair

-- Prove the number of grey-eyed students with black hair
theorem grey_eyes_black_hair:
  ∃ n, n = num_grey_eyes - num_grey_eyes_red_hair ∧ n = 20 :=
by
  sorry

end grey_eyes_black_hair_l177_177717


namespace sum_of_integers_is_27_24_or_20_l177_177343

theorem sum_of_integers_is_27_24_or_20 
    (x y : ℕ) 
    (h1 : 0 < x) 
    (h2 : 0 < y) 
    (h3 : x * y + x + y = 119) 
    (h4 : Nat.gcd x y = 1) 
    (h5 : x < 25) 
    (h6 : y < 25) 
    : x + y = 27 ∨ x + y = 24 ∨ x + y = 20 := 
sorry

end sum_of_integers_is_27_24_or_20_l177_177343


namespace reflection_line_slope_l177_177337

/-- Given two points (1, -2) and (7, 4), and the reflection line y = mx + b. 
    The image of (1, -2) under the reflection is (7, 4). Prove m + b = 4. -/
theorem reflection_line_slope (m b : ℝ)
    (h1: (∀ (x1 y1 x2 y2: ℝ), 
        (x1, y1) = (1, -2) → 
        (x2, y2) = (7, 4) → 
        (y2 - y1) / (x2 - x1) = 1)) 
    (h2: ∀ (x1 y1 x2 y2: ℝ),
        (x1, y1) = (1, -2) → 
        (x2, y2) = (7, 4) →
        (x1 + x2) / 2 = 4 ∧ (y1 + y2) / 2 = 1) 
    (h3: y = mx + b → m = -1 → (4, 1).1 = 4 ∧ (4, 1).2 = 1 → b = 5) : 
    m + b = 4 := by 
  -- No Proof Required
  sorry

end reflection_line_slope_l177_177337


namespace range_of_f_l177_177473

noncomputable def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ ({0, 1, 2, 3} : Finset ℕ), f x = y} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l177_177473


namespace positive_number_is_49_l177_177292

theorem positive_number_is_49 (a : ℝ) (x : ℝ) (h₁ : (3 - a) * (3 - a) = x) (h₂ : (2 * a + 1) * (2 * a + 1) = x) :
  x = 49 :=
sorry

end positive_number_is_49_l177_177292


namespace b8_expression_l177_177895

theorem b8_expression (a b : ℕ → ℚ)
  (ha0 : a 0 = 2)
  (hb0 : b 0 = 3)
  (ha : ∀ n, a (n + 1) = (a n) ^ 2 / (b n))
  (hb : ∀ n, b (n + 1) = (b n) ^ 2 / (a n)) :
  b 8 = 3 ^ 3281 / 2 ^ 3280 :=
by
  sorry

end b8_expression_l177_177895


namespace square_area_is_256_l177_177927

-- Definitions of the conditions
def rect_width : ℝ := 4
def rect_length : ℝ := 3 * rect_width
def side_of_square : ℝ := rect_length + rect_width

-- Proposition
theorem square_area_is_256 (rect_width : ℝ) (h1 : rect_width = 4) 
                           (rect_length : ℝ) (h2 : rect_length = 3 * rect_width) :
  side_of_square ^ 2 = 256 :=
by 
  sorry

end square_area_is_256_l177_177927


namespace distance_proof_l177_177525

theorem distance_proof (d : ℝ) (h1 : d < 6) (h2 : d > 5) (h3 : d > 4) : d ∈ Set.Ioo 5 6 :=
by
  sorry

end distance_proof_l177_177525


namespace find_root_l177_177685

theorem find_root (y : ℝ) (h : y - 9 / (y - 4) = 2 - 9 / (y - 4)) : y = 2 :=
by
  sorry

end find_root_l177_177685


namespace triangle_area_x_value_l177_177803

theorem triangle_area_x_value :
  ∃ x : ℝ, x > 0 ∧ 100 = (1 / 2) * x * (3 * x) ∧ x = 10 * Real.sqrt 6 / 3 :=
sorry

end triangle_area_x_value_l177_177803


namespace negation_equivalence_l177_177320

-- Declare the condition for real solutions of a quadratic equation
def has_real_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a * x + 1 = 0

-- Define the proposition p
def prop_p : Prop :=
  ∀ a : ℝ, a ≥ 0 → has_real_solutions a

-- Define the negation of p
def neg_prop_p : Prop :=
  ∃ a : ℝ, a ≥ 0 ∧ ¬ has_real_solutions a

-- The theorem stating the equivalence of p's negation to its formulated negation.
theorem negation_equivalence : neg_prop_p = ¬ prop_p := by
  sorry

end negation_equivalence_l177_177320


namespace turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l177_177797

-- Definitions based on conditions
def turnover_first_four_days : ℝ := 450
def turnover_fifth_day : ℝ := 0.12 * turnover_first_four_days
def total_turnover_five_days : ℝ := turnover_first_four_days + turnover_fifth_day

-- Proof statement for part 1
theorem turnover_five_days_eq_504 :
  total_turnover_five_days = 504 := 
sorry

-- Definitions and conditions for part 2
def turnover_february : ℝ := 350
def turnover_april : ℝ := total_turnover_five_days
def growth_rate (x : ℝ) : Prop := (1 + x)^2 * turnover_february = turnover_april

-- Proof statement for part 2
theorem monthly_growth_rate_eq_20_percent :
  ∃ x : ℝ, growth_rate x ∧ x = 0.2 := 
sorry

end turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l177_177797


namespace number_of_monkeys_l177_177747

theorem number_of_monkeys (X : ℕ) : 
  10 * 10 = 10 →
  1 * 1 = 1 →
  1 * 70 / 10 = 7 →
  (X / 7) = X / 7 :=
by
  intros h1 h2 h3
  sorry

end number_of_monkeys_l177_177747


namespace dolls_total_l177_177794

theorem dolls_total (dina_dolls ivy_dolls casey_dolls : ℕ) 
  (h1 : dina_dolls = 2 * ivy_dolls)
  (h2 : (2 / 3 : ℚ) * ivy_dolls = 20)
  (h3 : casey_dolls = 5 * 20) :
  dina_dolls + ivy_dolls + casey_dolls = 190 :=
by sorry

end dolls_total_l177_177794


namespace cos_120_eq_neg_half_l177_177106

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l177_177106


namespace number_of_three_digit_numbers_divisible_by_17_l177_177163

theorem number_of_three_digit_numbers_divisible_by_17 : 
  let k_min := Nat.ceil (100 / 17)
  let k_max := Nat.floor (999 / 17)
  ∃ n, 
    (n = k_max - k_min + 1) ∧ 
    (n = 53) := 
by
    sorry

end number_of_three_digit_numbers_divisible_by_17_l177_177163


namespace gasoline_tank_capacity_l177_177374

theorem gasoline_tank_capacity
  (initial_fill : ℝ) (final_fill : ℝ) (gallons_used : ℝ) (x : ℝ)
  (h1 : initial_fill = 3 / 4)
  (h2 : final_fill = 1 / 3)
  (h3 : gallons_used = 18)
  (h4 : initial_fill * x - final_fill * x = gallons_used) :
  x = 43 :=
by
  -- Skipping the proof
  sorry

end gasoline_tank_capacity_l177_177374


namespace basketball_weight_l177_177905

-- Definitions based on the given conditions
variables (b c : ℕ) -- weights of basketball and bicycle in pounds

-- Condition 1: Nine basketballs weigh the same as six bicycles
axiom condition1 : 9 * b = 6 * c

-- Condition 2: Four bicycles weigh a total of 120 pounds
axiom condition2 : 4 * c = 120

-- The proof statement we need to prove
theorem basketball_weight : b = 20 :=
by
  sorry

end basketball_weight_l177_177905


namespace min_radius_for_area_l177_177903

theorem min_radius_for_area (r : ℝ) (π : ℝ) (A : ℝ) (h1 : A = 314) (h2 : A = π * r^2) : r ≥ 10 :=
by
  sorry

end min_radius_for_area_l177_177903


namespace shpuntik_can_form_triangle_l177_177006

theorem shpuntik_can_form_triangle 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (hx : x1 + x2 + x3 = 1)
  (hy : y1 + y2 + y3 = 1)
  (infeasibility_vintik : x1 ≥ x2 + x3) :
  ∃ (a b c : ℝ), a + b + c = 1 ∧ a < b + c ∧ b < a + c ∧ c < a + b :=
sorry

end shpuntik_can_form_triangle_l177_177006


namespace problem_statement_l177_177806

def f : ℝ → ℝ
| x => if x > 0 then -Real.cos (π * x) else f (x + 1) + 1

theorem problem_statement : f (4/3) + f (-4/3) = 3 := 
sorry

end problem_statement_l177_177806


namespace number_of_roses_now_l177_177216

-- Given Conditions
def initial_roses : Nat := 7
def initial_orchids : Nat := 12
def current_orchids : Nat := 20
def orchids_more_than_roses : Nat := 9

-- Question to Prove: 
theorem number_of_roses_now :
  ∃ (R : Nat), (current_orchids = R + orchids_more_than_roses) ∧ (R = 11) :=
by {
  sorry
}

end number_of_roses_now_l177_177216


namespace multiplication_problem_l177_177427

-- Define the problem in Lean 4.
theorem multiplication_problem (a b : ℕ) (ha : a < 10) (hb : b < 10) 
  (h : (30 + a) * (10 * b + 4) = 126) : a + b = 7 :=
sorry

end multiplication_problem_l177_177427


namespace largest_perfect_square_factor_4410_l177_177763

theorem largest_perfect_square_factor_4410 : ∀ (n : ℕ), n = 441 → (∃ k : ℕ, k^2 ∣ 4410 ∧ ∀ m : ℕ, m^2 ∣ 4410 → m^2 ≤ k^2) := 
by
  sorry

end largest_perfect_square_factor_4410_l177_177763


namespace greatest_product_of_two_integers_sum_2006_l177_177010

theorem greatest_product_of_two_integers_sum_2006 :
  ∃ (x y : ℤ), x + y = 2006 ∧ x * y = 1006009 :=
by
  sorry

end greatest_product_of_two_integers_sum_2006_l177_177010


namespace sum_of_first_fifteen_terms_l177_177984

open Nat

-- Define the conditions
def a3 : ℝ := -5
def a5 : ℝ := 2.4

-- The arithmetic progression terms formula
def a (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- The sum of the first n terms formula
def Sn (a1 d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

-- The main theorem to prove
theorem sum_of_first_fifteen_terms :
  ∃ (a1 d : ℝ), (a a1 d 3 = a3) ∧ (a a1 d 5 = a5) ∧ (Sn a1 d 15 = 202.5) :=
sorry

end sum_of_first_fifteen_terms_l177_177984


namespace book_arrangement_l177_177962

theorem book_arrangement (total_books : ℕ) (geometry_books : ℕ) (number_theory_books : ℕ) (first_book_geometry : Prop)
  (h_total : total_books = 9)
  (h_geometry : geometry_books = 4)
  (h_number_theory : number_theory_books = 5)
  (h_first_geometry : first_book_geometry)
  : nat.choose 8 3 = 56 := 
by {
  -- Since we know total_books = geometry_books + number_theory_books and first_book_geometry holds,
  -- we just calculate the combination directly as in the problem statement:
  calc
  nat.choose 8 3 = 56 : by sorry -- skipping the proof step, as instructed.
}

end book_arrangement_l177_177962


namespace face_opposite_teal_is_blue_l177_177517

-- Definitions for the painting of the cube faces
def unique_colors := {"B", "Y", "O", "K", "T", "V"}

-- Views of the cube
def first_view := ("Y", "B", "O")
def second_view := ("Y", "K", "O")
def third_view := ("Y", "V", "O")

-- Face opposite to teal (T) should be proven to be blue (B)
theorem face_opposite_teal_is_blue
  (h1 : ∀ (c1 c2 c3 : String), (c1, c2, c3) ∈ {first_view, second_view, third_view} → c1 = "Y")
  (h2 : ∀ (c1 c2 c3 : String), (c1, c2, c3) ∈ {first_view, second_view, third_view} → c3 = "O")
  (h3 : {c2 | ∃ c1, ∃ c3, (c1, c2, c3) = first_view ∨ (c1, c2, c3) = second_view ∨ (c1, c2, c3) = third_view} = {"B", "K", "V"})
  : "B" = "Y" → "B" = "O" → "B" = "K" → "B" = "T" → "B" = "V" → false → ("B" = "B") :=
begin
  sorry
end

end face_opposite_teal_is_blue_l177_177517


namespace fraction_comparison_l177_177921

theorem fraction_comparison :
  let d := 0.33333333
  let f := (1 : ℚ) / 3
  f > d ∧ f - d = 1 / (3 * (10^8 : ℚ)) :=
by
  sorry

end fraction_comparison_l177_177921


namespace range_of_independent_variable_l177_177481

theorem range_of_independent_variable (x : ℝ) : (1 - x > 0) → x < 1 :=
by
  sorry

end range_of_independent_variable_l177_177481


namespace determine_x_l177_177702

theorem determine_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
by
  sorry

end determine_x_l177_177702


namespace fraction_identity_l177_177566

theorem fraction_identity (a b c : ℝ) (h1 : a + b + c > 0) (h2 : a + b - c > 0) (h3 : a + c - b > 0) (h4 : b + c - a > 0) 
  (h5 : (a+b+c)/(a+b-c) = 7) (h6 : (a+b+c)/(a+c-b) = 1.75) : (a+b+c)/(b+c-a) = 3.5 :=
by
  sorry

end fraction_identity_l177_177566


namespace solving_equation_l177_177589

theorem solving_equation (x : ℝ) : 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := 
by
  sorry

end solving_equation_l177_177589


namespace find_k4_l177_177281

theorem find_k4
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : ∃ r : ℝ, a_n 2^2 = a_n 1 * a_n 6)
  (h4 : a_n 1 = a_n k_1)
  (h5 : a_n 2 = a_n k_2)
  (h6 : a_n 6 = a_n k_3)
  (h_k1 : k_1 = 1)
  (h_k2 : k_2 = 2)
  (h_k3 : k_3 = 6) 
  : ∃ k_4 : ℕ, k_4 = 22 := sorry

end find_k4_l177_177281


namespace circle_area_above_line_l177_177358

-- Definition of the circle and the equation
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 48

-- The given line equation
def line_eq (y : ℝ) : Prop := y = 4

-- The target area above the line
def area_above_line : ℝ := 24 * Real.pi

-- The main theorem statement
theorem circle_area_above_line : ∀ x y : ℝ, circle_eq x y ∧ y > 4 → sorry := sorry

end circle_area_above_line_l177_177358


namespace student_allowance_l177_177228

theorem student_allowance (A : ℝ) (h1 : A * (2/5) = A - (A * (3/5)))
  (h2 : (A - (A * (2/5))) * (1/3) = ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) * (1/3))
  (h3 : ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) = 1.20) :
  A = 3.00 :=
by
  sorry

end student_allowance_l177_177228


namespace smallest_sector_angle_l177_177733

theorem smallest_sector_angle 
  (n : ℕ) (a1 : ℕ) (d : ℕ)
  (h1 : n = 18)
  (h2 : 360 = n * ((2 * a1 + (n - 1) * d) / 2))
  (h3 : ∀ i, 0 < i ∧ i ≤ 18 → ∃ k, 360 / 18 * k = i) :
  a1 = 3 :=
by sorry

end smallest_sector_angle_l177_177733


namespace cheating_percentage_l177_177248

theorem cheating_percentage (x : ℝ) :
  (∀ cost_price : ℝ, cost_price = 100 →
   let received_when_buying : ℝ := cost_price * (1 + x / 100)
   let given_when_selling : ℝ := cost_price * (1 - x / 100)
   let profit : ℝ := received_when_buying - given_when_selling
   let profit_percentage : ℝ := profit / cost_price
   profit_percentage = 2 / 9) →
  x = 22.22222222222222 := 
by
  sorry

end cheating_percentage_l177_177248


namespace total_baseball_fans_l177_177847

variable (Y M R : ℕ)

open Nat

theorem total_baseball_fans (h1 : 3 * M = 2 * Y) 
    (h2 : 4 * R = 5 * M) 
    (h3 : M = 96) : Y + M + R = 360 := by
  sorry

end total_baseball_fans_l177_177847


namespace a11_is_1_l177_177551

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Condition 1: The sum of the first n terms S_n satisfies S_n + S_m = S_{n+m}
axiom sum_condition (n m : ℕ) : S n + S m = S (n + m)

-- Condition 2: a_1 = 1
axiom a1_condition : a 1 = 1

-- Question: prove a_{11} = 1
theorem a11_is_1 : a 11 = 1 :=
sorry


end a11_is_1_l177_177551


namespace common_chord_of_circles_l177_177284

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x = y) :=
by
  intros x y h
  sorry

end common_chord_of_circles_l177_177284


namespace range_of_x_l177_177992

section
  variable (f : ℝ → ℝ)

  -- Conditions:
  -- 1. f is an even function
  def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

  -- 2. f is monotonically increasing on [0, +∞)
  def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

  -- Range of x
  def in_range (x : ℝ) : Prop := (1 : ℝ) / 3 < x ∧ x < (2 : ℝ) / 3

  -- Main statement
  theorem range_of_x (f_is_even : is_even f) (f_is_mono : mono_increasing_on_nonneg f) :
    ∀ x, f (2 * x - 1) < f ((1 : ℝ) / 3) ↔ in_range x := 
  by
    sorry
end

end range_of_x_l177_177992


namespace range_of_x_in_function_l177_177584

theorem range_of_x_in_function (x : ℝ) (h : x ≠ 8) : true := sorry

end range_of_x_in_function_l177_177584


namespace Tim_soda_cans_l177_177001

noncomputable def initial_cans : ℕ := 22
noncomputable def taken_cans : ℕ := 6
noncomputable def remaining_cans : ℕ := initial_cans - taken_cans
noncomputable def bought_cans : ℕ := remaining_cans / 2
noncomputable def final_cans : ℕ := remaining_cans + bought_cans

theorem Tim_soda_cans :
  final_cans = 24 :=
by
  sorry

end Tim_soda_cans_l177_177001


namespace problem_statement_l177_177507

noncomputable def increase_and_subtract (x p y : ℝ) : ℝ :=
  (x + p * x) - y

theorem problem_statement : increase_and_subtract 75 1.5 40 = 147.5 := by
  sorry

end problem_statement_l177_177507


namespace solution_of_inequality_l177_177983

noncomputable def solutionSet (a x : ℝ) : Set ℝ :=
  if a > 0 then {x | -a < x ∧ x < 3 * a}
  else if a < 0 then {x | 3 * a < x ∧ x < -a}
  else ∅

theorem solution_of_inequality (a x : ℝ) :
  (x^2 - 2 * a * x - 3 * a^2 < 0 ↔ x ∈ solutionSet a x) :=
sorry

end solution_of_inequality_l177_177983


namespace min_x2_plus_y2_l177_177706

noncomputable def min_val (x y : ℝ) : ℝ :=
  if h : (x + 1)^2 + y^2 = 1/4 then x^2 + y^2 else 0

theorem min_x2_plus_y2 : 
  ∃ x y : ℝ, (x + 1)^2 + y^2 = 1/4 ∧ x^2 + y^2 = 1/4 :=
by
  sorry

end min_x2_plus_y2_l177_177706


namespace solve_for_x_l177_177816

namespace proof_problem

-- Define the operation a * b = 4 * a * b
def star (a b : ℝ) : ℝ := 4 * a * b

-- Given condition rewritten in terms of the operation star
def equation (x : ℝ) : Prop := star x x + star 2 x - star 2 4 = 0

-- The statement we intend to prove
theorem solve_for_x (x : ℝ) : equation x → (x = 2 ∨ x = -4) :=
by
  -- Proof omitted
  sorry

end proof_problem

end solve_for_x_l177_177816


namespace radius_of_circle_l177_177953

variable (r M N : ℝ)

theorem radius_of_circle (h1 : M = Real.pi * r^2) 
  (h2 : N = 2 * Real.pi * r) 
  (h3 : M / N = 15) : 
  r = 30 :=
sorry

end radius_of_circle_l177_177953


namespace first_digit_base_8_of_725_is_1_l177_177506

-- Define conditions
def decimal_val := 725

-- Helper function to get the largest power of 8 less than the decimal value
def largest_power_base_eight (n : ℕ) : ℕ :=
  if 8^3 <= n then 8^3 else if 8^2 <= n then 8^2 else if 8^1 <= n then 8^1 else if 8^0 <= n then 8^0 else 0

-- The target theorem
theorem first_digit_base_8_of_725_is_1 : 
  (725 / largest_power_base_eight 725) = 1 :=
by 
  -- Proof goes here
  sorry

end first_digit_base_8_of_725_is_1_l177_177506


namespace volume_of_tetrahedron_PQRS_l177_177612

-- Definitions of the given conditions for the tetrahedron
def PQ := 6
def PR := 4
def PS := 5
def QR := 5
def QS := 6
def RS := 15 / 2  -- RS is (15 / 2), i.e., 7.5
def area_PQR := 12

noncomputable def volume_tetrahedron (PQ PR PS QR QS RS area_PQR : ℝ) : ℝ := 1 / 3 * area_PQR * 4

theorem volume_of_tetrahedron_PQRS :
  volume_tetrahedron PQ PR PS QR QS RS area_PQR = 16 :=
by sorry

end volume_of_tetrahedron_PQRS_l177_177612


namespace simplify_expression_l177_177508

theorem simplify_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := 
by 
  sorry 

end simplify_expression_l177_177508


namespace factors_of_2520_l177_177570

theorem factors_of_2520 : (∃ (factors : Finset ℕ), factors.card = 48 ∧ ∀ d, d ∈ factors ↔ d > 0 ∧ 2520 % d = 0) :=
sorry

end factors_of_2520_l177_177570


namespace option_d_is_correct_l177_177640

theorem option_d_is_correct : (-2 : ℤ) ^ 3 = -8 := by
  sorry

end option_d_is_correct_l177_177640


namespace Noemi_blackjack_loss_l177_177736

-- Define the conditions
def start_amount : ℕ := 1700
def end_amount : ℕ := 800
def roulette_loss : ℕ := 400

-- Define the total loss calculation
def total_loss : ℕ := start_amount - end_amount

-- Main theorem statement
theorem Noemi_blackjack_loss :
  ∃ (blackjack_loss : ℕ), blackjack_loss = total_loss - roulette_loss := 
by
  -- Start by calculating the total_loss
  let total_loss_eq := start_amount - end_amount
  -- The blackjack loss should be 900 - 400, which we claim to be 500
  use total_loss_eq - roulette_loss
  sorry

end Noemi_blackjack_loss_l177_177736


namespace fraction_value_l177_177560

theorem fraction_value
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (cond1 : (a + b + c) / (a + b - c) = 7)
  (cond2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 :=
by
  sorry

end fraction_value_l177_177560


namespace sum_of_integers_is_27_24_or_20_l177_177344

theorem sum_of_integers_is_27_24_or_20 
    (x y : ℕ) 
    (h1 : 0 < x) 
    (h2 : 0 < y) 
    (h3 : x * y + x + y = 119) 
    (h4 : Nat.gcd x y = 1) 
    (h5 : x < 25) 
    (h6 : y < 25) 
    : x + y = 27 ∨ x + y = 24 ∨ x + y = 20 := 
sorry

end sum_of_integers_is_27_24_or_20_l177_177344


namespace probability_eight_distinct_numbers_l177_177360

theorem probability_eight_distinct_numbers :
  let total_ways := 10^8
  let ways_distinct := (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3)
  (ways_distinct / total_ways : ℚ) = 18144 / 500000 := 
by
  sorry

end probability_eight_distinct_numbers_l177_177360


namespace John_to_floor_pushups_l177_177443

theorem John_to_floor_pushups:
  let days_per_week := 5
  let reps_per_day := 1
  let total_reps_per_stage := 15
  let stages := 3 -- number of stages: wall, high elevation, low elevation
  let total_days_needed := stages * total_reps_per_stage
  let total_weeks_needed := total_days_needed / days_per_week
  total_weeks_needed = 9 := by
  -- Here we will define the specifics of the proof later.
  sorry

end John_to_floor_pushups_l177_177443


namespace jar_marbles_difference_l177_177354

theorem jar_marbles_difference (a b : ℕ) (h1 : 9 * a = 9 * b) (h2 : 2 * a + b = 135) : 8 * b - 7 * a = 45 := by
  sorry

end jar_marbles_difference_l177_177354


namespace blocks_remaining_l177_177466

def initial_blocks : ℕ := 55
def blocks_eaten : ℕ := 29

theorem blocks_remaining : initial_blocks - blocks_eaten = 26 := by
  sorry

end blocks_remaining_l177_177466


namespace triangle_angle_side_cases_l177_177301

theorem triangle_angle_side_cases
  (b c : ℝ) (B : ℝ)
  (hb : b = 3)
  (hc : c = 3 * Real.sqrt 3)
  (hB : B = Real.pi / 6) :
  (∃ A C a, A = Real.pi / 2 ∧ C = Real.pi / 3 ∧ a = Real.sqrt 21) ∨
  (∃ A C a, A = Real.pi / 6 ∧ C = 2 * Real.pi / 3 ∧ a = 3) :=
by
  sorry

end triangle_angle_side_cases_l177_177301


namespace interior_angle_of_regular_nonagon_l177_177764

theorem interior_angle_of_regular_nonagon : 
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  (sum_of_interior_angles / n) = 140 := 
by
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  show sum_of_interior_angles / n = 140
  sorry

end interior_angle_of_regular_nonagon_l177_177764


namespace fraction_A_BC_l177_177952

-- Definitions for amounts A, B, C and the total T
variable (T : ℝ) (A : ℝ) (B : ℝ) (C : ℝ)

-- Given conditions
def conditions : Prop :=
  T = 300 ∧
  A = 120.00000000000001 ∧
  B = (6 / 9) * (A + C) ∧
  A + B + C = T

-- The fraction of the amount A gets compared to B and C together
def fraction (x : ℝ) : Prop :=
  A = x * (B + C)

-- The proof goal
theorem fraction_A_BC : conditions T A B C → fraction A B C (2 / 3) :=
by
  sorry

end fraction_A_BC_l177_177952


namespace cos_120_eq_neg_one_half_l177_177076

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l177_177076


namespace inequality_holds_l177_177544

variable {a b c : ℝ}

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
sorry

end inequality_holds_l177_177544


namespace rectangle_dimension_area_l177_177356

theorem rectangle_dimension_area (x : Real) 
  (h_dim1 : x + 3 > 0) 
  (h_dim2 : 3 * x - 2 > 0) :
  ((x + 3) * (3 * x - 2) = 9 * x + 1) ↔ x = (11 + Real.sqrt 205) / 6 := 
sorry

end rectangle_dimension_area_l177_177356


namespace cos_120_eq_neg_half_l177_177124

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177124


namespace gcd_polynomial_multiple_of_345_l177_177698

theorem gcd_polynomial_multiple_of_345 (b : ℕ) (h : ∃ k : ℕ, b = 345 * k) : 
  Nat.gcd (5 * b ^ 3 + 2 * b ^ 2 + 7 * b + 69) b = 69 := 
by
  sorry

end gcd_polynomial_multiple_of_345_l177_177698


namespace measure_angle_A_l177_177858

open Real.Angle

theorem measure_angle_A (B C A : Real.Angle) (hB : B = 18) (hC : C = 3 * B) (hSum : A + B + C = 180) :
  A = 108 :=
by sorry

end measure_angle_A_l177_177858


namespace distinct_right_angles_l177_177489

theorem distinct_right_angles (n : ℕ) (h : n > 0) : 
  ∃ (a b c d : ℕ), (a + b + c + d ≥ 4 * (Int.sqrt n)) ∧ (a * c ≥ n) ∧ (b * d ≥ n) :=
by sorry

end distinct_right_angles_l177_177489


namespace union_of_A_and_B_intersection_of_A_and_B_l177_177555

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 4 }
noncomputable def B : Set ℝ := { x | x > 3 ∨ x < 1 }

theorem union_of_A_and_B : A ∪ B = Set.univ :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = { x | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4) } :=
by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_l177_177555


namespace cosine_relationship_l177_177814

open Real

noncomputable def functional_relationship (x y : ℝ) : Prop :=
  y = -(4 / 5) * sqrt (1 - x ^ 2) + (3 / 5) * x

theorem cosine_relationship (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : cos (α + β) = - 4 / 5) (h6 : sin β = x) (h7 : cos α = y) (h8 : 4 / 5 < x) (h9 : x < 1) :
  functional_relationship x y :=
sorry

end cosine_relationship_l177_177814


namespace factorization_correct_l177_177403

-- Define the given expression
def expression (a b : ℝ) : ℝ := 9 * a^2 * b - b

-- Define the factorized form
def factorized_form (a b : ℝ) : ℝ := b * (3 * a + 1) * (3 * a - 1)

-- Theorem stating that the factorization is correct
theorem factorization_correct (a b : ℝ) : expression a b = factorized_form a b := by
  sorry

end factorization_correct_l177_177403


namespace shop_width_l177_177209

theorem shop_width 
  (monthly_rent : ℝ) 
  (shop_length : ℝ) 
  (annual_rent_per_sqft : ℝ) 
  (width : ℝ) 
  (monthly_rent_eq : monthly_rent = 2244) 
  (shop_length_eq : shop_length = 22) 
  (annual_rent_per_sqft_eq : annual_rent_per_sqft = 68) 
  (width_eq : width = 18) : 
  (12 * monthly_rent) / annual_rent_per_sqft / shop_length = width := 
by 
  sorry

end shop_width_l177_177209


namespace Jessica_cut_roses_l177_177485

theorem Jessica_cut_roses
  (initial_roses : ℕ) (initial_orchids : ℕ)
  (new_roses : ℕ) (new_orchids : ℕ)
  (cut_roses : ℕ) :
  initial_roses = 15 → initial_orchids = 62 →
  new_roses = 17 → new_orchids = 96 →
  new_roses = initial_roses + cut_roses →
  cut_roses = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h3] at h5
  linarith

end Jessica_cut_roses_l177_177485


namespace find_initial_milk_amount_l177_177017

-- Define the initial amount of milk as a variable in liters
variable (T : ℝ)

-- Given conditions
def consumed (T : ℝ) := 0.4 * T
def leftover (T : ℝ) := 0.69

-- The total milk at first was T if T = 0.69 / 0.6
theorem find_initial_milk_amount 
  (h1 : leftover T = 0.69)
  (h2 : consumed T = 0.4 * T) :
  T = 1.15 :=
by
  sorry

end find_initial_milk_amount_l177_177017


namespace remainder_of_n_plus_3255_l177_177012

theorem remainder_of_n_plus_3255 (n : ℤ) (h : n % 5 = 2) : (n + 3255) % 5 = 2 := 
by
  sorry

end remainder_of_n_plus_3255_l177_177012


namespace B_A_equals_expectedBA_l177_177898

noncomputable def MatrixA : Matrix (Fin 2) (Fin 2) ℝ := sorry
noncomputable def MatrixB : Matrix (Fin 2) (Fin 2) ℝ := sorry
def MatrixAB : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 1], ![-2, 4]]
def expectedBA : Matrix (Fin 2) (Fin 2) ℝ := ![![10, 2], ![-4, 8]]

theorem B_A_equals_expectedBA (A B : Matrix (Fin 2) (Fin 2) ℝ)
  (h1 : A + B = 2 * A * B)
  (h2 : A * B = MatrixAB) : 
  B * A = expectedBA := by
  sorry

end B_A_equals_expectedBA_l177_177898


namespace solve_y_l177_177793

theorem solve_y (y : ℝ) : (12 - y)^2 = 4 * y^2 ↔ y = 4 ∨ y = -12 := by
  sorry

end solve_y_l177_177793


namespace points_on_line_sufficient_but_not_necessary_l177_177649

open Nat

-- Define the sequence a_n
def sequence_a (n : ℕ) : ℕ := n + 1

-- Define a general arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) := ∀ n m : ℕ, n < m → a (m) - a (n) = (m - n) * (a 1 - a 0)

-- Define the condition that points (n, a_n), where n is a natural number, lie on the line y = x + 1
def points_on_line (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n) = n + 1

-- Prove that points_on_line is sufficient but not necessary for is_arithmetic_sequence
theorem points_on_line_sufficient_but_not_necessary :
  (∀ a : ℕ → ℕ, points_on_line a → is_arithmetic_sequence a)
  ∧ ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ ¬ points_on_line a := 
by 
  sorry

end points_on_line_sufficient_but_not_necessary_l177_177649


namespace part1_factorization_part2_factorization_l177_177741

-- Part 1
theorem part1_factorization (x : ℝ) :
  (x - 1) * (6 * x + 5) = 6 * x^2 - x - 5 :=
by {
  sorry
}

-- Part 2
theorem part2_factorization (x : ℝ) :
  (x - 1) * (x + 3) * (x - 2) = x^3 - 7 * x + 6 :=
by {
  sorry
}

end part1_factorization_part2_factorization_l177_177741


namespace vector_parallel_x_is_neg1_l177_177032

variables (a b : ℝ × ℝ)
variable (x : ℝ)

def vectors_parallel : Prop := 
  (a = (1, -1)) ∧ (b = (x, 1)) ∧ (a.1 * b.2 - a.2 * b.1 = 0)

theorem vector_parallel_x_is_neg1 (h : vectors_parallel a b x) : x = -1 :=
sorry

end vector_parallel_x_is_neg1_l177_177032


namespace find_sum_of_relatively_prime_integers_l177_177341

theorem find_sum_of_relatively_prime_integers :
  ∃ (x y : ℕ), x * y + x + y = 119 ∧ x < 25 ∧ y < 25 ∧ Nat.gcd x y = 1 ∧ x + y = 20 :=
by
  sorry

end find_sum_of_relatively_prime_integers_l177_177341


namespace complement_union_correct_l177_177192

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {2, 3, 4})
variable (hB : B = {1, 4})

theorem complement_union_correct :
  (compl A ∪ B) = {1, 4, 5} :=
by
  sorry

end complement_union_correct_l177_177192


namespace prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l177_177022

-- Define the conditions
def total_trips : ℕ := 500
def on_time_A : ℕ := 240
def not_on_time_A : ℕ := 20
def total_A : ℕ := on_time_A + not_on_time_A

def on_time_B : ℕ := 210
def not_on_time_B : ℕ := 30
def total_B : ℕ := on_time_B + not_on_time_B

def total_on_time : ℕ := on_time_A + on_time_B
def total_not_on_time : ℕ := not_on_time_A + not_on_time_B

-- Define the probabilities according to the given solution
def prob_A_on_time : ℚ := on_time_A / total_A
def prob_B_on_time : ℚ := on_time_B / total_B

-- Prove the estimated probabilities
theorem prob_A_correct : prob_A_on_time = 12 / 13 := sorry
theorem prob_B_correct : prob_B_on_time = 7 / 8 := sorry

-- Define the K^2 formula
def K_squared : ℚ :=
  total_trips * (on_time_A * not_on_time_B - on_time_B * not_on_time_A)^2 /
  ((total_A) * (total_B) * (total_on_time) * (total_not_on_time))

-- Prove the provided K^2 value and the conclusion
theorem K_squared_approx_correct (h : K_squared ≈ 3.205) : 3.205 > 2.706 := sorry
theorem punctuality_related_to_company : 3.205 > 2.706 → true := sorry

end prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l177_177022


namespace one_fourth_one_third_two_fifths_l177_177463

theorem one_fourth_one_third_two_fifths (N : ℝ)
  (h₁ : 0.40 * N = 300) :
  (1/4) * (1/3) * (2/5) * N = 25 := 
sorry

end one_fourth_one_third_two_fifths_l177_177463


namespace cos_120_eq_neg_half_l177_177081

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177081


namespace fraction_product_l177_177503

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l177_177503


namespace smallest_yellow_marbles_l177_177783

def total_marbles (n : ℕ) := n

def blue_marbles (n : ℕ) := n / 3

def red_marbles (n : ℕ) := n / 4

def green_marbles := 6

def yellow_marbles (n : ℕ) := n - (blue_marbles n + red_marbles n + green_marbles)

theorem smallest_yellow_marbles (n : ℕ) (hn : n % 12 = 0) (blue : blue_marbles n = n / 3)
  (red : red_marbles n = n / 4) (green : green_marbles = 6) :
  yellow_marbles n = 4 ↔ n = 24 :=
by sorry

end smallest_yellow_marbles_l177_177783


namespace correct_statements_l177_177157

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

def satisfies_condition (f : ℝ → ℝ) : Prop := 
  ∀ x, f (1 - x) + f (1 + x) = 0

def is_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (a + x)

theorem correct_statements (f : ℝ → ℝ) :
  is_even f →
  is_monotonically_increasing f (-1) 0 →
  satisfies_condition f →
  (f (-3) = 0 ∧
   is_monotonically_increasing f 1 2 ∧
   is_symmetric_about_line f 1) :=
by
  intros h_even h_mono h_cond
  sorry

end correct_statements_l177_177157


namespace angle_B_in_triangle_l177_177865

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l177_177865


namespace volume_ratio_l177_177303

variable (A B : ℝ)

theorem volume_ratio (h1 : (3 / 4) * A = (5 / 8) * B) :
  A / B = 5 / 6 :=
by
  sorry

end volume_ratio_l177_177303


namespace angle_between_vectors_is_90_degrees_l177_177162

open Real

variables (a b : ℝ)

def vector_a : ℝ × ℝ := (sin (15 * pi / 180), cos (15 * pi / 180))
def vector_b : ℝ × ℝ := (cos (15 * pi / 180), sin (15 * pi / 180))

def vector_add : ℝ × ℝ := (sin (15 * pi / 180) + cos (15 * pi / 180), sin (15 * pi / 180) + cos (15 * pi / 180))
def vector_sub : ℝ × ℝ := (sin (15 * pi / 180) - cos (15 * pi / 180), cos (15 * pi / 180) - sin (15 * pi / 180))

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem angle_between_vectors_is_90_degrees :
  dot_product vector_add vector_sub = 0 :=
sorry

end angle_between_vectors_is_90_degrees_l177_177162


namespace divisors_end_with_1_l177_177908

theorem divisors_end_with_1 (n : ℕ) (h : n > 0) :
  ∀ d : ℕ, d ∣ (10^(5^n) - 1) / 9 → d % 10 = 1 :=
sorry

end divisors_end_with_1_l177_177908


namespace perfect_rectangle_squares_l177_177778

theorem perfect_rectangle_squares (squares : Finset ℕ) 
  (h₁ : 9 ∈ squares) 
  (h₂ : 2 ∈ squares) 
  (h₃ : squares.card = 9) 
  (h₄ : ∀ x ∈ squares, ∃ y ∈ squares, x ≠ y ∧ (gcd x y = 1)) :
  squares = {2, 5, 7, 9, 16, 25, 28, 33, 36} := 
sorry

end perfect_rectangle_squares_l177_177778


namespace part1_part2_l177_177149

variables {a_n b_n : ℕ → ℤ} {k m : ℕ}

-- Part 1: Arithmetic Sequence
axiom a2_eq_3 : a_n 2 = 3
axiom S5_eq_25 : (5 * (2 * (a_n 1 + 2 * (a_n 1 + 1)) / 2)) = 25

-- Part 2: Geometric Sequence
axiom b1_eq_1 : b_n 1 = 1
axiom q_eq_3 : ∀ n, b_n n = 3^(n-1)

noncomputable def arithmetic_seq (n : ℕ) : ℤ :=
  2 * n - 1

theorem part1 : (a_n 2 + a_n 4) / 2 = 5 :=
  sorry

theorem part2 (k : ℕ) (hk : 0 < k) : ∃ m, b_n k = arithmetic_seq m ∧ m = (3^(k-1) + 1) / 2 :=
  sorry

end part1_part2_l177_177149


namespace soccer_balls_donated_l177_177047

def num_classes_per_school (elem_classes mid_classes : ℕ) : ℕ :=
  elem_classes + mid_classes

def total_classes (num_schools : ℕ) (classes_per_school : ℕ) : ℕ :=
  num_schools * classes_per_school

def total_soccer_balls (num_classes : ℕ) (balls_per_class : ℕ) : ℕ :=
  num_classes * balls_per_class

theorem soccer_balls_donated 
  (elem_classes mid_classes num_schools balls_per_class : ℕ) 
  (h_elem_classes : elem_classes = 4) 
  (h_mid_classes : mid_classes = 5) 
  (h_num_schools : num_schools = 2) 
  (h_balls_per_class : balls_per_class = 5) :
  total_soccer_balls (total_classes num_schools (num_classes_per_school elem_classes mid_classes)) balls_per_class = 90 :=
by
  sorry

end soccer_balls_donated_l177_177047


namespace net_gain_A_correct_l177_177196

-- Define initial values and transactions
def initial_cash_A : ℕ := 20000
def house_value : ℕ := 20000
def car_value : ℕ := 5000
def initial_cash_B : ℕ := 25000
def house_sale_price : ℕ := 21000
def car_sale_price : ℕ := 4500
def house_repurchase_price : ℕ := 19000
def car_depreciation : ℕ := 10
def car_repurchase_price : ℕ := 4050

-- Define the final cash calculations
def final_cash_A := initial_cash_A + house_sale_price + car_sale_price - house_repurchase_price - car_repurchase_price
def final_cash_B := initial_cash_B - house_sale_price - car_sale_price + house_repurchase_price + car_repurchase_price

-- Define the net gain calculations
def net_gain_A := final_cash_A - initial_cash_A
def net_gain_B := final_cash_B - initial_cash_B

-- Theorem to prove
theorem net_gain_A_correct : net_gain_A = 2000 :=
by 
  -- Definitions and calculations would go here
  sorry

end net_gain_A_correct_l177_177196


namespace salary_increase_l177_177255

theorem salary_increase (x : ℝ) 
  (h : ∀ s : ℕ, 1 ≤ s ∧ s ≤ 5 → ∃ p : ℝ, p = 7.50 + x * (s - 1))
  (h₁ : ∃ p₁ p₅ : ℝ, 1 ≤ 1 ∧ 5 ≤ 5 ∧ p₅ = p₁ + 1.25) :
  x = 0.3125 := sorry

end salary_increase_l177_177255


namespace x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l177_177712

-- Define the context and main statement
theorem x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta
  (θ : ℝ)
  (hθ₁ : 0 < θ)
  (hθ₂ : θ < (π / 2))
  {x : ℝ}
  (hx : x + 1 / x = 2 * Real.sin θ)
  (n : ℕ) (hn : 0 < n) :
  x^n + 1 / x^n = 2 * Real.sin (n * θ) :=
sorry

end x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l177_177712


namespace pencils_count_l177_177947

theorem pencils_count (P L : ℕ) (h₁ : 6 * P = 5 * L) (h₂ : L = P + 4) : L = 24 :=
by sorry

end pencils_count_l177_177947


namespace part_I_part_II_l177_177820

theorem part_I : 
  (∀ x : ℝ, |x - (2 : ℝ)| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) :=
  sorry

theorem part_II :
  (∀ a b c : ℝ, a - 2 * b + c = 2 → a^2 + b^2 + c^2 ≥ 2 / 3) :=
  sorry

end part_I_part_II_l177_177820


namespace cos_120_eq_neg_half_l177_177126

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177126


namespace time_for_one_essay_l177_177891

-- We need to define the times for questions and paragraphs first.

def time_per_short_answer_question := 3 -- in minutes
def time_per_paragraph := 15 -- in minutes
def total_homework_time := 4 -- in hours
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15

-- Now we need to state the total homework time and define the goal
def computed_homework_time :=
  (time_per_short_answer_question * num_short_answer_questions +
   time_per_paragraph * num_paragraphs) / 60 + num_essays * sorry -- time for one essay in hours

theorem time_for_one_essay :
  (total_homework_time = computed_homework_time) → sorry = 1 :=
by
  sorry

end time_for_one_essay_l177_177891


namespace number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l177_177696

theorem number_of_pentagonal_faces_is_12_more_than_heptagonal_faces
  (convex : Prop)
  (trihedral : Prop)
  (faces_have_5_6_or_7_sides : Prop)
  (V E F : ℕ)
  (a b c : ℕ)
  (euler : V - E + F = 2)
  (edges_def : E = (5 * a + 6 * b + 7 * c) / 2)
  (vertices_def : V = (5 * a + 6 * b + 7 * c) / 3) :
  a = c + 12 :=
  sorry

end number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l177_177696


namespace inscribed_square_area_l177_177250

theorem inscribed_square_area :
  (∃ (t : ℝ), (2*t)^2 = 4 * (t^2) ∧ ∀ (x y : ℝ), (x = t ∧ y = t ∨ x = -t ∧ y = t ∨ x = t ∧ y = -t ∨ x = -t ∧ y = -t) 
  → (x^2 / 4 + y^2 / 8 = 1) ) 
  → (∃ (a : ℝ), a = 32 / 3) := 
by
  sorry

end inscribed_square_area_l177_177250


namespace combined_age_l177_177201

-- Define the conditions given in the problem
def Hezekiah_age : Nat := 4
def Ryanne_age := Hezekiah_age + 7

-- The statement to prove
theorem combined_age : Ryanne_age + Hezekiah_age = 15 :=
by
  -- we would provide the proof here, but for now we'll skip it with 'sorry'
  sorry

end combined_age_l177_177201


namespace increasing_interval_iff_l177_177285

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 3 * x

def is_increasing (a : ℝ) : Prop :=
  ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ < f a x₂

theorem increasing_interval_iff (a : ℝ) (h : a ≠ 0) :
  is_increasing a ↔ a ∈ Set.Ioo (-(5/4)) 0 ∪ Set.Ioi 0 :=
sorry

end increasing_interval_iff_l177_177285


namespace probability_second_ball_black_l177_177373

-- Condition definitions
def total_balls := 10
def black_balls := 3
def white_balls := 7

-- Event definition
def total_ways_to_draw_two_balls := 10 * 9
def ways_to_draw_second_black := black_balls * 9

-- Theorem statement
theorem probability_second_ball_black :
  (ways_to_draw_second_black : ℚ) / total_ways_to_draw_two_balls = 3 / 10 :=
by 
  -- Placeholder for the actual proof
  sorry

end probability_second_ball_black_l177_177373


namespace percentage_problem_l177_177514

theorem percentage_problem 
  (number : ℕ)
  (h1 : number = 6400)
  (h2 : 5 * number / 100 = 20 * 650 / 100 + 190) : 
  20 = 20 :=
by 
  sorry

end percentage_problem_l177_177514


namespace second_term_geometric_sequence_l177_177753

-- Given conditions
def a3 : ℕ := 12
def a4 : ℕ := 18
def q := a4 / a3 -- common ratio

-- Geometric progression definition
noncomputable def a2 := a3 / q

-- Theorem to prove
theorem second_term_geometric_sequence : a2 = 8 :=
by
  -- proof not required
  sorry

end second_term_geometric_sequence_l177_177753


namespace length_of_field_l177_177367

-- Define the conditions and given facts.
def double_length (w l : ℝ) : Prop := l = 2 * w
def pond_area (l w : ℝ) : Prop := 49 = 1/8 * (l * w)

-- Define the main statement that incorporates the given conditions and expected result.
theorem length_of_field (w l : ℝ) (h1 : double_length w l) (h2 : pond_area l w) : l = 28 := by
  sorry

end length_of_field_l177_177367


namespace division_of_fractions_l177_177826

theorem division_of_fractions : (2 / 3) / (1 / 4) = (8 / 3) := by
  sorry

end division_of_fractions_l177_177826


namespace custom_op_evaluation_l177_177263

def custom_op (a b : ℤ) : ℤ := a * b - (a + b)

theorem custom_op_evaluation : custom_op 2 (-3) = -5 :=
by
sorry

end custom_op_evaluation_l177_177263


namespace bus_probabilities_and_chi_squared_l177_177023

noncomputable def prob_on_time_A : ℚ :=
12 / 13

noncomputable def prob_on_time_B : ℚ :=
7 / 8

noncomputable def chi_squared(K2 : ℚ) : Bool :=
K2 > 2.706

theorem bus_probabilities_and_chi_squared :
  prob_on_time_A = 240 / 260 ∧
  prob_on_time_B = 210 / 240 ∧
  chi_squared(3.205) = True :=
by
  -- proof steps will go here
  sorry

end bus_probabilities_and_chi_squared_l177_177023


namespace sequence_is_geometric_and_general_formula_l177_177811

theorem sequence_is_geometric_and_general_formula (a : ℕ → ℝ) (h0 : a 1 = 2 / 3)
  (h1 : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (a (n + 1) + 1)) :
  ∃ r : ℝ, (0 < r ∧ r < 1 ∧ (∀ n : ℕ, a (n + 1) = (2:ℝ)^n / (1 + (2:ℝ)^n)) ∧
  ∀ n : ℕ, (1 / a (n + 1) - 1) = (1 / 2) * (1 / a n - 1)) := sorry

end sequence_is_geometric_and_general_formula_l177_177811


namespace max_area_rectangle_l177_177384

-- Define the conditions using Lean
def is_rectangle (length width : ℕ) : Prop :=
  2 * (length + width) = 34

-- Define the problem as a theorem in Lean
theorem max_area_rectangle : ∃ (length width : ℕ), is_rectangle length width ∧ length * width = 72 :=
by
  sorry

end max_area_rectangle_l177_177384


namespace sqrt_D_irrational_l177_177310

open Real

theorem sqrt_D_irrational (a : ℤ) (D : ℝ) (hD : D = a^2 + (a + 2)^2 + (a^2 + (a + 2))^2) : ¬ ∃ m : ℤ, D = m^2 :=
by
  sorry

end sqrt_D_irrational_l177_177310


namespace common_divisors_count_l177_177827

-- Define the numbers
def num1 := 9240
def num2 := 13860

-- Define the gcd of the numbers
def gcdNum := Nat.gcd num1 num2

-- Prove the number of divisors of the gcd is 48
theorem common_divisors_count : (Nat.divisors gcdNum).card = 48 :=
by
  -- Normally we would provide a detailed proof here
  sorry

end common_divisors_count_l177_177827


namespace problem_proof_l177_177185

theorem problem_proof (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x^2 + A)^2 + B) - (B * (A * x^2 + B)^2 + A) = A^2 - B^2) :
  A^2 + B^2 = - (A * B) := 
sorry

end problem_proof_l177_177185


namespace isosceles_triangle_median_length_l177_177437

noncomputable def median_length (b h : ℝ) : ℝ :=
  let a := Real.sqrt ((b / 2) ^ 2 + h ^ 2)
  let m_a := Real.sqrt ((2 * a ^ 2 + 2 * b ^ 2 - a ^ 2) / 4)
  m_a

theorem isosceles_triangle_median_length :
  median_length 16 10 = Real.sqrt 146 :=
by
  sorry

end isosceles_triangle_median_length_l177_177437


namespace find_A_solution_l177_177404

theorem find_A_solution (A : ℝ) (h : 32 * A^3 = 42592) : A = 11 :=
sorry

end find_A_solution_l177_177404


namespace remaining_painting_time_l177_177777

-- Define the given conditions as Lean definitions
def total_rooms : ℕ := 9
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 5

-- Formulate the main theorem to prove the remaining time is 32 hours
theorem remaining_painting_time : 
  (total_rooms - rooms_painted) * hours_per_room = 32 := 
by 
  sorry

end remaining_painting_time_l177_177777


namespace min_x2_y2_eq_16_then_product_zero_l177_177432

theorem min_x2_y2_eq_16_then_product_zero
  (x y : ℝ)
  (h1 : ∃ x y : ℝ, (x^2 + y^2 = 16 ∧ ∀ a b : ℝ, a^2 + b^2 ≥ 16) ) :
  (x + 4) * (y - 4) = 0 := 
sorry

end min_x2_y2_eq_16_then_product_zero_l177_177432


namespace no_valid_k_values_l177_177389

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roots_are_primes (k : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 57 ∧ p * q = k

theorem no_valid_k_values : ∀ k : ℕ, ¬ roots_are_primes k := by
  sorry

end no_valid_k_values_l177_177389


namespace Mater_costs_10_percent_of_Lightning_l177_177902

-- Conditions
def price_Lightning : ℕ := 140000
def price_Sally : ℕ := 42000
def price_Mater : ℕ := price_Sally / 3

-- The theorem we want to prove
theorem Mater_costs_10_percent_of_Lightning :
  (price_Mater * 100 / price_Lightning) = 10 := 
by 
  sorry

end Mater_costs_10_percent_of_Lightning_l177_177902


namespace inverse_variation_solution_l177_177204

theorem inverse_variation_solution :
  ∀ (x y k : ℝ),
    (x * y^3 = k) →
    (∃ k, x = 8 ∧ y = 1 ∧ k = 8) →
    (y = 2 → x = 1) :=
by
  intros x y k h1 h2 hy2
  sorry

end inverse_variation_solution_l177_177204


namespace katie_speed_l177_177401

theorem katie_speed (eugene_speed : ℝ)
  (brianna_ratio : ℝ)
  (katie_ratio : ℝ)
  (h1 : eugene_speed = 4)
  (h2 : brianna_ratio = 2 / 3)
  (h3 : katie_ratio = 7 / 5) :
  katie_ratio * (brianna_ratio * eugene_speed) = 56 / 15 := 
by
  sorry

end katie_speed_l177_177401


namespace tetrahedron_volume_l177_177614

/-- Given a regular triangular pyramid (tetrahedron) with the following properties:
  - Distance from the midpoint of the height to a lateral face is 2.
  - Distance from the midpoint of the height to a lateral edge is √14.
  Prove that the volume of the pyramid is approximately 533.38.
-/
theorem tetrahedron_volume (d_face d_edge : ℝ) (volume : ℝ) (h1 : d_face = 2) (h2 : d_edge = Real.sqrt 14) :
  Abs (volume - 533.38) < 0.01 :=
by {
  sorry -- Proof will go here
}

end tetrahedron_volume_l177_177614


namespace compare_negative_sqrt_values_l177_177396

theorem compare_negative_sqrt_values : -3 * Real.sqrt 3 > -2 * Real.sqrt 7 := 
sorry

end compare_negative_sqrt_values_l177_177396


namespace planes_parallel_if_line_perpendicular_to_both_l177_177450

variables {Line Plane : Type}
variables (l : Line) (α β : Plane)

-- Assume we have a function parallel that checks if a line is parallel to a plane
-- and a function perpendicular that checks if a line is perpendicular to a plane. 
-- Also, we assume a function parallel_planes that checks if two planes are parallel.
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

theorem planes_parallel_if_line_perpendicular_to_both
  (h1 : perpendicular l α) (h2 : perpendicular l β) : parallel_planes α β :=
sorry

end planes_parallel_if_line_perpendicular_to_both_l177_177450


namespace angle_B_eq_3pi_over_10_l177_177888

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l177_177888


namespace michael_made_small_balls_l177_177904

def num_small_balls (total_bands : ℕ) (bands_per_small : ℕ) (bands_per_large : ℕ) (num_large : ℕ) : ℕ :=
  (total_bands - num_large * bands_per_large) / bands_per_small

theorem michael_made_small_balls :
  num_small_balls 5000 50 300 13 = 22 :=
by
  sorry

end michael_made_small_balls_l177_177904


namespace least_integer_remainder_condition_l177_177220

def is_least_integer_with_remainder_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ k ∈ [3, 4, 5, 6, 7, 10, 11], n % k = 1)

theorem least_integer_remainder_condition : ∃ (n : ℕ), is_least_integer_with_remainder_condition n ∧ n = 4621 :=
by
  -- The proof will go here.
  sorry

end least_integer_remainder_condition_l177_177220


namespace world_expo_visitors_l177_177670

noncomputable def per_person_cost (x : ℕ) : ℕ :=
  if x <= 30 then 120 else max (120 - 2 * (x - 30)) 90

theorem world_expo_visitors (x : ℕ) (h_cost : x * per_person_cost x = 4000) : x = 40 :=
by
  sorry

end world_expo_visitors_l177_177670


namespace investment_final_value_l177_177659

theorem investment_final_value 
  (original_investment : ℝ) 
  (increase_percentage : ℝ) 
  (original_investment_eq : original_investment = 12500)
  (increase_percentage_eq : increase_percentage = 2.15) : 
  original_investment * (1 + increase_percentage) = 39375 := 
by
  sorry

end investment_final_value_l177_177659


namespace find_y_l177_177223

theorem find_y (y : ℕ) (h : (2 * y) / 5 = 10) : y = 25 :=
sorry

end find_y_l177_177223


namespace son_age_18_l177_177019

theorem son_age_18 (F S : ℤ) (h1 : F = S + 20) (h2 : F + 2 = 2 * (S + 2)) : S = 18 :=
by
  sorry

end son_age_18_l177_177019


namespace ratio_of_arithmetic_sums_l177_177066

theorem ratio_of_arithmetic_sums : 
  let a1 := 4
  let d1 := 4
  let l1 := 48
  let a2 := 2
  let d2 := 3
  let l2 := 35
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let S1 := n1 * (a1 + l1) / 2
  let S2 := n2 * (a2 + l2) / 2
  let ratio := S1 / S2
  ratio = 52 / 37 := by sorry

end ratio_of_arithmetic_sums_l177_177066


namespace kite_area_eq_twenty_l177_177759

theorem kite_area_eq_twenty :
  let base := 10
  let height := 2
  let area_of_triangle := (1 / 2 : ℝ) * base * height
  let total_area := 2 * area_of_triangle
  total_area = 20 :=
by
  sorry

end kite_area_eq_twenty_l177_177759


namespace cosine_120_eq_negative_half_l177_177116

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l177_177116


namespace base_four_product_l177_177359

def base_four_to_decimal (n : ℕ) : ℕ :=
  -- definition to convert base 4 to decimal, skipping details for now
  sorry

def decimal_to_base_four (n : ℕ) : ℕ :=
  -- definition to convert decimal to base 4, skipping details for now
  sorry

theorem base_four_product : 
  base_four_to_decimal 212 * base_four_to_decimal 13 = base_four_to_decimal 10322 :=
sorry

end base_four_product_l177_177359


namespace sheila_hourly_wage_is_correct_l177_177324

-- Definitions based on conditions
def works_hours_per_day_mwf : ℕ := 8
def works_days_mwf : ℕ := 3
def works_hours_per_day_tt : ℕ := 6
def works_days_tt : ℕ := 2
def weekly_earnings : ℕ := 216

-- Total calculated hours based on the problem conditions
def total_weekly_hours : ℕ := (works_hours_per_day_mwf * works_days_mwf) + (works_hours_per_day_tt * works_days_tt)

-- Target wage per hour
def wage_per_hour : ℕ := weekly_earnings / total_weekly_hours

-- The theorem stating the proof problem
theorem sheila_hourly_wage_is_correct : wage_per_hour = 6 := by
  sorry

end sheila_hourly_wage_is_correct_l177_177324


namespace loga_increasing_loga_decreasing_l177_177488

noncomputable def loga (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem loga_increasing (a : ℝ) (h₁ : a > 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a x < loga a y := by
  sorry 

theorem loga_decreasing (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a y < loga a x := by
  sorry

end loga_increasing_loga_decreasing_l177_177488


namespace coefficient_of_expansion_l177_177576

theorem coefficient_of_expansion (m : ℝ) (h : m^3 * (Nat.choose 6 3) = -160) : m = -2 := by
  sorry

end coefficient_of_expansion_l177_177576


namespace b_amount_l177_177770

-- Define the conditions
def total_amount (a b : ℝ) : Prop := a + b = 1210
def fraction_condition (a b : ℝ) : Prop := (1/3) * a = (1/4) * b

-- Define the main theorem to prove B's amount
theorem b_amount (a b : ℝ) (h1 : total_amount a b) (h2 : fraction_condition a b) : b = 691.43 :=
sorry

end b_amount_l177_177770


namespace chef_cooked_additional_wings_l177_177376

def total_chicken_wings_needed (friends : ℕ) (wings_per_friend : ℕ) : ℕ :=
  friends * wings_per_friend

def additional_chicken_wings (total_needed : ℕ) (already_cooked : ℕ) : ℕ :=
  total_needed - already_cooked

theorem chef_cooked_additional_wings :
  let friends := 4
  let wings_per_friend := 4
  let already_cooked := 9
  additional_chicken_wings (total_chicken_wings_needed friends wings_per_friend) already_cooked = 7 := by
  sorry

end chef_cooked_additional_wings_l177_177376


namespace distance_proof_l177_177526

theorem distance_proof (d : ℝ) (h1 : d < 6) (h2 : d > 5) (h3 : d > 4) : d ∈ Set.Ioo 5 6 :=
by
  sorry

end distance_proof_l177_177526


namespace cos_120_eq_neg_half_l177_177091

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l177_177091


namespace correct_option_l177_177769

theorem correct_option : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := sorry

end correct_option_l177_177769


namespace smart_charging_piles_growth_l177_177580

noncomputable def a : ℕ := 301
noncomputable def b : ℕ := 500
variable (x : ℝ) -- Monthly average growth rate

theorem smart_charging_piles_growth :
  a * (1 + x) ^ 2 = b :=
by
  -- Proof should go here
  sorry

end smart_charging_piles_growth_l177_177580


namespace cos_120_eq_neg_half_l177_177092

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l177_177092


namespace solution_largest_a_exists_polynomial_l177_177800

def largest_a_exists_polynomial : Prop :=
  ∃ (P : ℝ → ℝ) (a b c d e : ℝ),
    (∀ x, P x = a * x^4 + b * x^3 + c * x^2 + d * x + e) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 1 → 0 ≤ P x ∧ P x ≤ 1) ∧
    a = 4

theorem solution_largest_a_exists_polynomial : largest_a_exists_polynomial :=
  sorry

end solution_largest_a_exists_polynomial_l177_177800


namespace find_x_modulo_l177_177801

theorem find_x_modulo (k : ℤ) : ∃ x : ℤ, x = 18 + 31 * k ∧ ((37 * x) % 31 = 15) := by
  sorry

end find_x_modulo_l177_177801


namespace total_population_is_700_l177_177755

-- Definitions for the problem conditions
def L : ℕ := 200
def P : ℕ := L / 2
def E : ℕ := (L + P) / 2
def Z : ℕ := E + P

-- Proof statement (with sorry)
theorem total_population_is_700 : L + P + E + Z = 700 :=
by
  sorry

end total_population_is_700_l177_177755


namespace fraction_computation_l177_177499

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l177_177499


namespace Alyssa_initial_puppies_l177_177965

theorem Alyssa_initial_puppies : 
  ∀ (a b c : ℕ), b = 7 → c = 5 → a = b + c → a = 12 := 
by
  intros a b c hb hc hab
  rw [hb, hc] at hab
  exact hab

end Alyssa_initial_puppies_l177_177965


namespace find_a_value_l177_177282

theorem find_a_value (a : ℝ) (x : ℝ) :
  (a + 1) * x^2 + (a^2 + 1) + 8 * x = 9 →
  a + 1 ≠ 0 →
  a^2 + 1 = 9 →
  a = 2 * Real.sqrt 2 :=
by
  intro h1 h2 h3
  sorry

end find_a_value_l177_177282


namespace cos_120_degrees_eq_l177_177113

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l177_177113


namespace daily_earnings_from_oil_refining_l177_177195

-- Definitions based on conditions
def daily_earnings_from_mining : ℝ := 3000000
def monthly_expenses : ℝ := 30000000
def fine : ℝ := 25600000
def profit_percentage : ℝ := 0.01
def months_in_year : ℝ := 12
def days_in_month : ℝ := 30

-- The question translated as a Lean theorem statement
theorem daily_earnings_from_oil_refining : ∃ O : ℝ, O = 5111111.11 ∧ 
  fine = profit_percentage * months_in_year * 
    (days_in_month * (daily_earnings_from_mining + O) - monthly_expenses) :=
sorry

end daily_earnings_from_oil_refining_l177_177195


namespace efficacy_rate_is_80_percent_l177_177439

-- Define the total number of people surveyed
def total_people : ℕ := 20

-- Define the number of people who find the new drug effective
def effective_people : ℕ := 16

-- Calculate the efficacy rate
def efficacy_rate (effective : ℕ) (total : ℕ) : ℚ := effective / total

-- The theorem to be proved
theorem efficacy_rate_is_80_percent : efficacy_rate effective_people total_people = 0.8 :=
by
  sorry

end efficacy_rate_is_80_percent_l177_177439


namespace total_toys_per_week_l177_177775

def toys_per_day := 1100
def working_days_per_week := 5

theorem total_toys_per_week : toys_per_day * working_days_per_week = 5500 :=
by
  sorry

end total_toys_per_week_l177_177775


namespace inequalities_no_solution_l177_177633

theorem inequalities_no_solution (x n : ℝ) (h1 : x ≤ 1) (h2 : x ≥ n) : n > 1 :=
sorry

end inequalities_no_solution_l177_177633


namespace intersection_A_B_l177_177429

def set_A : Set ℝ := {x : ℝ | |x| = x}
def set_B : Set ℝ := {x : ℝ | x^2 + x ≥ 0}
def set_intersection : Set ℝ := {x : ℝ | 0 ≤ x}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection :=
by
  sorry

-- You can verify if the Lean code builds successfully using Lean 4 environment.

end intersection_A_B_l177_177429


namespace usual_time_eight_l177_177370

/-- Define the parameters used in the problem -/
def usual_speed (S : ℝ) : ℝ := S
def usual_time (T : ℝ) : ℝ := T
def reduced_speed (S : ℝ) := 0.25 * S
def reduced_time (T : ℝ) := T + 24

/-- The main theorem that we need to prove -/
theorem usual_time_eight (S T : ℝ) 
  (h1 : usual_speed S = S)
  (h2 : usual_time T = T)
  (h3 : reduced_speed S = 0.25 * S)
  (h4 : reduced_time T = T + 24)
  (h5 : S / (0.25 * S) = (T + 24) / T) : T = 8 :=
by 
  sorry -- Proof omitted for brevity. Refers to the solution steps.


end usual_time_eight_l177_177370


namespace base_of_first_term_is_two_l177_177945

-- Define h as a positive integer
variable (h : ℕ) (a b c : ℕ)

-- Conditions
variables 
  (h_positive : h > 0)
  (divisor_225 : 225 ∣ h)
  (divisor_216 : 216 ∣ h)

-- Given h can be expressed as specified and a + b + c = 8
variable (h_expression : ∃ k : ℕ, h = k^a * 3^b * 5^c)
variable (sum_eight : a + b + c = 8)

-- Prove the base of the first term in the expression for h is 2.
theorem base_of_first_term_is_two : (∃ k : ℕ, k^a * 3^b * 5^c = h) → k = 2 :=
by 
  sorry

end base_of_first_term_is_two_l177_177945


namespace cricket_run_rate_l177_177774

theorem cricket_run_rate 
  (run_rate_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_played : ℕ)
  (remaining_overs : ℕ)
  (correct_run_rate : ℝ)
  (h1 : run_rate_10_overs = 3.6)
  (h2 : target_runs = 282)
  (h3 : overs_played = 10)
  (h4 : remaining_overs = 40)
  (h5 : correct_run_rate = 6.15) :
  (target_runs - run_rate_10_overs * overs_played) / remaining_overs = correct_run_rate :=
sorry

end cricket_run_rate_l177_177774


namespace division_problem_l177_177931

theorem division_problem : 96 / (8 / 4) = 48 := 
by {
  sorry
}

end division_problem_l177_177931


namespace odometer_reading_at_lunch_l177_177735

axiom odometer_start : ℝ
axiom miles_traveled : ℝ
axiom odometer_at_lunch : ℝ
axiom starting_reading : odometer_start = 212.3
axiom travel_distance : miles_traveled = 159.7
axiom at_lunch_reading : odometer_at_lunch = odometer_start + miles_traveled

theorem odometer_reading_at_lunch :
  odometer_at_lunch = 372.0 :=
  by
  sorry

end odometer_reading_at_lunch_l177_177735


namespace find_sum_indices_with_tastiness_l177_177688

noncomputable def chromatic_polynomial (G : Graph ℕ) : Polynomial ℕ := sorry

def tasty_graph (V E : ℕ) (G : Graph V) (k : ℕ) : Prop :=
  G.connected ∧ E = 2017 ∧ exists (C : list V), G.is_cycle_of_length k C

def tastiness (G : Graph ℕ) : ℕ :=
  (chromatic_polynomial G).coeffs.count is_odd

def minimal_tastiness (k_min k_max : ℕ) : ℕ :=
  finset.min' (finset.image tastiness (finset.filter (λ k, ∃ G, tasty_graph 2017 2017 G k) (finset.range (k_max - k_min + 1)))) sorry 

def sum_indices_with_tastiness (k_min k_max : ℕ) : ℕ :=
  finset.sum (finset.filter (λ k, ∃ G, tasty_graph 2017 2017 G k ∧ tastiness G = minimal_tastiness k_min k_max)
    (finset.range (k_max - k_min + 1))) id

theorem find_sum_indices_with_tastiness :
  sum_indices_with_tastiness 3 2017 = 2017 := 
  sorry

end find_sum_indices_with_tastiness_l177_177688


namespace find_cost_price_l177_177623

-- Condition 1: The owner charges his customer 15% more than the cost price.
def selling_price (C : Real) : Real := C * 1.15

-- Condition 2: A customer paid Rs. 8325 for the computer table.
def paid_amount : Real := 8325

-- Define the cost price and its expected value
def cost_price : Real := 7239.13

-- The theorem to prove that the cost price matches the expected value
theorem find_cost_price : 
  ∃ C : Real, selling_price C = paid_amount ∧ C = cost_price :=
by
  sorry

end find_cost_price_l177_177623


namespace minimum_value_of_expression_l177_177894

noncomputable def expr (a b c : ℝ) : ℝ := 8 * a^3 + 27 * b^3 + 64 * c^3 + 27 / (8 * a * b * c)

theorem minimum_value_of_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  expr a b c ≥ 18 * Real.sqrt 3 := 
by
  sorry

end minimum_value_of_expression_l177_177894


namespace cos_120_degrees_l177_177072

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l177_177072


namespace meadowbrook_total_not_74_l177_177857

theorem meadowbrook_total_not_74 (h c : ℕ) : 
  21 * h + 6 * c ≠ 74 := sorry

end meadowbrook_total_not_74_l177_177857


namespace renovation_costs_l177_177051

theorem renovation_costs :
  ∃ (x y : ℝ), 
    8 * x + 8 * y = 3520 ∧
    6 * x + 12 * y = 3480 ∧
    x = 300 ∧
    y = 140 ∧
    300 * 12 > 140 * 24 :=
by sorry

end renovation_costs_l177_177051


namespace relationship_of_y_coordinates_l177_177737

theorem relationship_of_y_coordinates (b y1 y2 y3 : ℝ):
  (y1 = 3 * -2.3 + b) → (y2 = 3 * -1.3 + b) → (y3 = 3 * 2.7 + b) → (y1 < y2 ∧ y2 < y3) := 
by 
  intros h1 h2 h3
  sorry

end relationship_of_y_coordinates_l177_177737


namespace min_sum_of_factors_of_144_is_neg_145_l177_177449

theorem min_sum_of_factors_of_144_is_neg_145 
  (a b : ℤ) 
  (h : a * b = 144) : 
  a + b ≥ -145 := 
sorry

end min_sum_of_factors_of_144_is_neg_145_l177_177449


namespace cos_120_eq_neg_half_l177_177101

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l177_177101


namespace sum_of_coefficients_l177_177393

theorem sum_of_coefficients (x y z : ℤ) (h : x = 1 ∧ y = 1 ∧ z = 1) :
    (x - 2 * y + 3 * z) ^ 12 = 4096 :=
by
  sorry

end sum_of_coefficients_l177_177393


namespace vector_dot_product_l177_177410

variables (a b : ℝ × ℝ)
variables (ha : a = (1, -1)) (hb : b = (-1, 2))

theorem vector_dot_product : 
  ((2 • a + b) • a) = -1 :=
by
  -- This is where the proof would go
  sorry

end vector_dot_product_l177_177410


namespace solve_x_l177_177817

-- Define the custom multiplication operation *
def custom_mul (a b : ℕ) : ℕ := 4 * a * b

-- Given that x * x + 2 * x - 2 * 4 = 0
def equation (x : ℕ) : Prop := custom_mul x x + 2 * x - 2 * 4 = 0

theorem solve_x (x : ℕ) (h : equation x) : x = 2 ∨ x = -4 := 
by 
  -- proof steps go here
  sorry

end solve_x_l177_177817


namespace technicians_count_l177_177295

theorem technicians_count {T R : ℕ} (h1 : T + R = 12) (h2 : 2 * T + R = 18) : T = 6 :=
sorry

end technicians_count_l177_177295


namespace find_value_of_a_l177_177387

-- Define variables and constants
variable (a : ℚ)
variable (b : ℚ := 3 * a)
variable (c : ℚ := 4 * b)
variable (d : ℚ := 6 * c)
variable (total : ℚ := 186)

-- State the theorem
theorem find_value_of_a (h : a + b + c + d = total) : a = 93 / 44 := by
  sorry

end find_value_of_a_l177_177387


namespace find_f2_plus_g2_l177_177153

variable (f g : ℝ → ℝ)

def even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x
def odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem find_f2_plus_g2 (hf : even_function f) (hg : odd_function g) (h : ∀ x, f x - g x = x^3 - 2 * x^2) :
  f 2 + g 2 = -16 :=
sorry

end find_f2_plus_g2_l177_177153


namespace ball_box_distribution_l177_177833

theorem ball_box_distribution : (∃ (f : Fin 4 → Fin 2), true) ∧ (∀ (f : Fin 4 → Fin 2), true) → ∃ (f : Fin 4 → Fin 2), true ∧ f = 16 :=
by sorry

end ball_box_distribution_l177_177833


namespace determine_range_of_a_l177_177821

noncomputable def f (x a : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

noncomputable def g (x a : ℝ) : ℝ := f x a - 2*x

theorem determine_range_of_a (a : ℝ) :
  (∀ x, g x a = 0 → (x = 2 ∨ x = -1 ∨ x = -2)) →
  (-1 ≤ a ∧ a < 2) :=
by
  intro h
  sorry

end determine_range_of_a_l177_177821


namespace value_of_b_plus_c_l177_177815

variable {a b c d : ℝ}

theorem value_of_b_plus_c (h1 : a + b = 4) (h2 : c + d = 5) (h3 : a + d = 2) : b + c = 7 :=
sorry

end value_of_b_plus_c_l177_177815


namespace julia_total_kids_l177_177181

def kidsMonday : ℕ := 7
def kidsTuesday : ℕ := 13
def kidsThursday : ℕ := 18
def kidsWednesdayCards : ℕ := 20
def kidsWednesdayHideAndSeek : ℕ := 11
def kidsWednesdayPuzzle : ℕ := 9
def kidsFridayBoardGame : ℕ := 15
def kidsFridayDrawingCompetition : ℕ := 12

theorem julia_total_kids : 
  kidsMonday + kidsTuesday + kidsThursday + kidsWednesdayCards + kidsWednesdayHideAndSeek + kidsWednesdayPuzzle + kidsFridayBoardGame + kidsFridayDrawingCompetition = 105 :=
by
  sorry

end julia_total_kids_l177_177181


namespace irrational_sqrt_2023_l177_177942

theorem irrational_sqrt_2023 (A B C D : ℝ) :
  A = -2023 → B = Real.sqrt 2023 → C = 0 → D = 1 / 2023 →
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ B = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ A = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ C = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ D = p / q) := 
by
  intro hA hB hC hD
  sorry

end irrational_sqrt_2023_l177_177942


namespace joe_out_of_money_after_one_month_worst_case_l177_177591

-- Define the initial amount Joe has
def initial_amount : ℝ := 240

-- Define Joe's monthly subscription cost
def subscription_cost : ℝ := 15

-- Define the range of prices for buying games
def min_game_cost : ℝ := 40
def max_game_cost : ℝ := 60

-- Define the range of prices for selling games
def min_resale_price : ℝ := 20
def max_resale_price : ℝ := 40

-- Define the maximum number of games Joe can purchase per month
def max_games_per_month : ℕ := 3

-- Prove that Joe will be out of money after 1 month in the worst-case scenario
theorem joe_out_of_money_after_one_month_worst_case :
  initial_amount - 
  (max_games_per_month * max_game_cost - max_games_per_month * min_resale_price + subscription_cost) < 0 :=
by
  sorry

end joe_out_of_money_after_one_month_worst_case_l177_177591


namespace window_treatments_cost_l177_177596

-- Define the costs and the number of windows
def cost_sheers : ℝ := 40.00
def cost_drapes : ℝ := 60.00
def number_of_windows : ℕ := 3

-- Define the total cost calculation
def total_cost := (cost_sheers + cost_drapes) * number_of_windows

-- State the theorem that needs to be proved
theorem window_treatments_cost : total_cost = 300.00 :=
by
  sorry

end window_treatments_cost_l177_177596


namespace length_reduction_percentage_to_maintain_area_l177_177339

theorem length_reduction_percentage_to_maintain_area
  (L W : ℝ)
  (new_width : ℝ := W * (1 + 28.2051282051282 / 100))
  (new_length : ℝ := L * (1 - 21.9512195121951 / 100))
  (original_area : ℝ := L * W) :
  original_area = new_length * new_width := by
  sorry

end length_reduction_percentage_to_maintain_area_l177_177339


namespace angle_measure_l177_177134

theorem angle_measure (x : ℝ) 
  (h1 : 90 - x = (2 / 5) * (180 - x)) :
  x = 30 :=
by
  sorry

end angle_measure_l177_177134


namespace sum_of_numbers_is_37_l177_177271

theorem sum_of_numbers_is_37 :
  ∃ (A B : ℕ), 
    1 ≤ A ∧ A ≤ 50 ∧ 1 ≤ B ∧ B ≤ 50 ∧ A ≠ B ∧
    (50 * B + A = k^2) ∧ Prime B ∧ B > 10 ∧
    A + B = 37 
  := by
    sorry

end sum_of_numbers_is_37_l177_177271


namespace relationship_x_y_l177_177694

theorem relationship_x_y (x y m : ℝ) (h1 : x + m = 4) (h2 : y - 5 = m) : x + y = 9 := 
by 
  sorry

end relationship_x_y_l177_177694


namespace common_divisors_count_l177_177828

-- Define the numbers
def num1 := 9240
def num2 := 13860

-- Define the gcd of the numbers
def gcdNum := Nat.gcd num1 num2

-- Prove the number of divisors of the gcd is 48
theorem common_divisors_count : (Nat.divisors gcdNum).card = 48 :=
by
  -- Normally we would provide a detailed proof here
  sorry

end common_divisors_count_l177_177828


namespace solution_set_of_inequality_l177_177346

theorem solution_set_of_inequality : 
  {x : ℝ | |x|^3 - 2 * x^2 - 4 * |x| + 3 < 0} = 
  { x : ℝ | -3 < x ∧ x < -1 } ∪ { x : ℝ | 1 < x ∧ x < 3 } := 
by
  sorry

end solution_set_of_inequality_l177_177346


namespace original_number_eq_0_000032_l177_177366

theorem original_number_eq_0_000032 (x : ℝ) (hx : 0 < x) 
  (h : 10^8 * x = 8 * (1 / x)) : x = 0.000032 :=
sorry

end original_number_eq_0_000032_l177_177366


namespace find_fraction_l177_177562

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h1 : (a + b + c) / (a + b - c) = 7)
variable (h2 : (a + b + c) / (a + c - b) = 1.75)

theorem find_fraction : (a + b + c) / (b + c - a) = 3.5 := 
by {
  sorry
}

end find_fraction_l177_177562


namespace largest_perfect_square_factor_4410_l177_177762

theorem largest_perfect_square_factor_4410 :
  ∀ (n : ℕ), n = 4410 → 
  ∃ (k : ℕ), k * k ∣ n ∧ ∀ (m : ℕ), m * m ∣ n → m * m ≤ k * k :=
by
  intro n hn
  have h4410 : ∃ p q r s : ℕ, 4410 = p * 3^2 * q * 7^2 ∧ p = 2 ∧ q = 5 ∧ r = 3 ∧ s = 7 :=
    ⟨2, 5, 3, 7, rfl, rfl, rfl, rfl⟩
  use 21
  constructor
  · rw hn
    norm_num
  · intros m hm
    norm_num  at hm
    sorry

end largest_perfect_square_factor_4410_l177_177762


namespace cos_120_eq_neg_half_l177_177084

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177084


namespace probability_increasing_function_correct_l177_177634

noncomputable def probability_increasing_function : ℚ :=
  let outcomes := { (m, n) | m ∈ finset.range 1 7 ∧ n ∈ finset.range 1 7 } in
  let valid_outcomes := { (m, n) ∈ outcomes | (n : ℚ) / (2 * m) ≤ 1 } in
  finset.card valid_outcomes / finset.card outcomes

theorem probability_increasing_function_correct :
  probability_increasing_function = 5 / 6 := by
  sorry

end probability_increasing_function_correct_l177_177634


namespace infinite_cubes_diff_3p1_infinite_cubes_diff_5q1_l177_177909

theorem infinite_cubes_diff_3p1 : 
  ∀ n : ℕ+, ∃ p : ℕ+, 3 * (n ^ 2 + n) + 1 = (n + 1) ^ 3 - n ^ 3 := 
by 
  intros n 
  use (n ^ 2 + n)
  sorry

theorem infinite_cubes_diff_5q1 :
  ∀ n : ℕ+, ∃ q : ℕ+, 5 * (15 * n ^ 2 + 3 * n) + 1 = (5 * n + 1) ^ 3 - (5 * n) ^ 3 := 
by 
  intros n 
  use (15 * n ^ 2 + 3 * n)
  sorry

end infinite_cubes_diff_3p1_infinite_cubes_diff_5q1_l177_177909


namespace charles_initial_bananas_l177_177225

theorem charles_initial_bananas (W C : ℕ) (h1 : W = 48) (h2 : C = C - 35 + W - 13) : C = 35 := by
  -- W = 48
  -- Charles loses 35 bananas
  -- Willie will have 13 bananas
  sorry

end charles_initial_bananas_l177_177225


namespace tan_sum_to_expression_l177_177145

theorem tan_sum_to_expression (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 :=
by 
  sorry

end tan_sum_to_expression_l177_177145


namespace part_1_part_2_part_3_l177_177278

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x - g x a

theorem part_1 (a : ℝ) : (F' := λ x, Real.exp x + Real.cos x - a) → 
                          (F' 0 = 0) → a = 2 :=
by sorry

theorem part_2 (a : ℝ) : (∀ x, 0 ≤ x → F x a ≥ 1) → a ≤ 2 :=
by sorry

theorem part_3 (x1 x2 : ℝ) : (a = 1 / 3) → 
                           (0 ≤ x1) → (0 ≤ x2) → 
                           (F x1 a - g x2 a) → (3 * (Real.exp x1 + Real.sin x1 - 1/3 * x1) ≥ 3) → 
                           x2 - x1 = 3 :=
by sorry

end part_1_part_2_part_3_l177_177278


namespace wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l177_177930

noncomputable def number_of_cubes (n : ℕ) : ℕ :=
  n + (n - 1) + n

noncomputable def painted_area (n : ℕ) : ℕ :=
  (5 * n) + (3 * (n + 1)) + (2 * (n - 2))

theorem wall_with_5_peaks_has_14_cubes : number_of_cubes 5 = 14 :=
  by sorry

theorem wall_with_2014_peaks_has_6041_cubes : number_of_cubes 2014 = 6041 :=
  by sorry

theorem painted_area_wall_with_2014_peaks : painted_area 2014 = 20139 :=
  by sorry

end wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l177_177930


namespace functional_equation_solution_l177_177977

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (f x + f y)) = f x + y) : ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equation_solution_l177_177977


namespace problem_l177_177925

def is_acute_angle (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_first_quadrant (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_second_quadrant (θ: ℝ) : Prop := θ > 90 ∧ θ < 180

def cond1 (θ: ℝ) : Prop := θ < 90 → is_acute_angle θ
def cond2 (θ: ℝ) : Prop := in_first_quadrant θ → θ ≥ 0
def cond3 (θ: ℝ) : Prop := is_acute_angle θ → in_first_quadrant θ
def cond4 (θ θ': ℝ) : Prop := in_second_quadrant θ → in_first_quadrant θ' → θ > θ'

theorem problem :
  (¬ ∃ θ, cond1 θ) ∧ (¬ ∃ θ, cond2 θ) ∧ (∃ θ, cond3 θ) ∧ (¬ ∃ θ θ', cond4 θ θ') →
  (number_of_correct_propositions = 1) :=
  by
    sorry

end problem_l177_177925


namespace GoldenRabbitCards_count_l177_177433

theorem GoldenRabbitCards_count :
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  golden_cards = 5904 :=
by
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  sorry

end GoldenRabbitCards_count_l177_177433


namespace man_speed_is_5_km_per_hr_l177_177240

def convert_minutes_to_hours (minutes: ℕ) : ℝ :=
  minutes / 60

def convert_meters_to_kilometers (meters: ℕ) : ℝ :=
  meters / 1000

def man_speed (distance_m: ℕ) (time_min: ℕ) : ℝ :=
  distance_m / 1000 / (time_min / 60)

theorem man_speed_is_5_km_per_hr : man_speed 1250 15 = 5 := by
  unfold man_speed convert_meters_to_kilometers convert_minutes_to_hours
  -- More steps in the proof would go here, but we use sorry to skip the proof.
  sorry

end man_speed_is_5_km_per_hr_l177_177240


namespace expected_value_X_probability_two_red_balls_from_B_l177_177719

noncomputable def count_combinations {α : Type*} (s : finset α) (k : ℕ) : ℕ :=
s.powerset.filter (λ t, t.card = k).card

-- Definitions based on the given conditions
def boxA := finset.of_list ["W1", "W2", "W3", "R1", "R2"]
def boxB := finset.of_list ["W4", "W5", "W6", "W7", "R3"]

def combinations (s : finset string) (n : ℕ) : finset (finset string) :=
s.powerset.filter (λ t, t.card = n)

def relevant_combinations (n : ℕ) : finset (finset string) :=
combinations boxA n

-- Lean statement for part 1
theorem expected_value_X :
  let p0 := (count_combinations (relevant_combinations 2).filter (λ x, x.filter (λ b, b[0] = 'W').card = 2) / count_combinations boxA 2)
      p1 := (count_combinations (relevant_combinations 2).filter (λ x, x.filter (λ b, b[0] = 'W').card = 1) / count_combinations boxA 2)
      p2 := (count_combinations (relevant_combinations 2).filter (λ x, x.filter (λ b, b[0] = 'R').card = 2) / count_combinations boxA 2) in
  0 * p0 + 1 * p1 + 2 * p2 = 4 / 5 := 
sorry

-- Lean statement for part 2
theorem probability_two_red_balls_from_B :
  let X0 := 0
      X1 := (count_combinations (combinations (boxB ∪ ["W1", "W2"]) 2) / count_combinations (boxB ∪ ["W1", "W2"]) 2)
      X2 := (count_combinations (combinations (boxB ∪ ["R1", "R2"]) 2) - 1 / count_combinations (boxB ∪ ["R1", "R2"]) 2) in
  3 / 10 * X0 + 6 / 10 * 1 / 21 + 1 / 10 * 1 / 7 = 3 / 70 :=
sorry

end expected_value_X_probability_two_red_balls_from_B_l177_177719


namespace horse_food_per_day_l177_177674

-- Given conditions
def sheep_count : ℕ := 48
def horse_food_total : ℕ := 12880
def sheep_horse_ratio : ℚ := 6 / 7

-- Definition of the number of horses based on the ratio
def horse_count : ℕ := (sheep_count * 7) / 6

-- Statement to prove: each horse needs 230 ounces of food per day
theorem horse_food_per_day : horse_food_total / horse_count = 230 := by
  -- proof here
  sorry

end horse_food_per_day_l177_177674


namespace lucas_investment_l177_177732

noncomputable def investment_amount (y : ℝ) : ℝ := 1500 - y

theorem lucas_investment :
  ∃ y : ℝ, (y * 1.04 + (investment_amount y) * 1.06 = 1584.50) ∧ y = 275 :=
by
  sorry

end lucas_investment_l177_177732


namespace activity_probability_l177_177545

noncomputable def total_basic_events : ℕ := 3^4
noncomputable def favorable_events : ℕ := Nat.choose 4 2 * Nat.factorial 3

theorem activity_probability :
  (favorable_events : ℚ) / total_basic_events = 4 / 9 :=
by
  sorry

end activity_probability_l177_177545


namespace cos_120_eq_neg_half_l177_177090

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l177_177090


namespace largest_4_digit_divisible_by_35_l177_177638

theorem largest_4_digit_divisible_by_35 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 35 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 35 = 0) → m ≤ n) ∧ n = 9985 := 
by sorry

end largest_4_digit_divisible_by_35_l177_177638


namespace count_perfect_squares_between_l177_177710

theorem count_perfect_squares_between :
  let n := 8
  let m := 70
  (m - n + 1) = 64 :=
by
  -- Definitions and step-by-step proof would go here.
  sorry

end count_perfect_squares_between_l177_177710


namespace prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l177_177021

-- Define the conditions
def total_trips : ℕ := 500
def on_time_A : ℕ := 240
def not_on_time_A : ℕ := 20
def total_A : ℕ := on_time_A + not_on_time_A

def on_time_B : ℕ := 210
def not_on_time_B : ℕ := 30
def total_B : ℕ := on_time_B + not_on_time_B

def total_on_time : ℕ := on_time_A + on_time_B
def total_not_on_time : ℕ := not_on_time_A + not_on_time_B

-- Define the probabilities according to the given solution
def prob_A_on_time : ℚ := on_time_A / total_A
def prob_B_on_time : ℚ := on_time_B / total_B

-- Prove the estimated probabilities
theorem prob_A_correct : prob_A_on_time = 12 / 13 := sorry
theorem prob_B_correct : prob_B_on_time = 7 / 8 := sorry

-- Define the K^2 formula
def K_squared : ℚ :=
  total_trips * (on_time_A * not_on_time_B - on_time_B * not_on_time_A)^2 /
  ((total_A) * (total_B) * (total_on_time) * (total_not_on_time))

-- Prove the provided K^2 value and the conclusion
theorem K_squared_approx_correct (h : K_squared ≈ 3.205) : 3.205 > 2.706 := sorry
theorem punctuality_related_to_company : 3.205 > 2.706 → true := sorry

end prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l177_177021


namespace even_function_value_at_2_l177_177477

theorem even_function_value_at_2 {a : ℝ} (h : ∀ x : ℝ, (x + 1) * (x - a) = (-x + 1) * (-x - a)) : 
  ((2 + 1) * (2 - a)) = 3 := by
  sorry

end even_function_value_at_2_l177_177477


namespace purely_imaginary_m_eq_neg_half_second_quadrant_m_range_l177_177411

noncomputable def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

-- Part (1)
theorem purely_imaginary_m_eq_neg_half (m : ℝ) (h : z m.imaginary = 0) : m = -1 / 2 :=
by
  sorry

-- Part (2)
theorem second_quadrant_m_range (m : ℝ) (h1 : z m.real < 0) (h2 : z m.imaginary > 0) : -1 / 2 < m ∧ m < 1 :=
by
  sorry

end purely_imaginary_m_eq_neg_half_second_quadrant_m_range_l177_177411


namespace sum_of_first_fifteen_terms_l177_177985

noncomputable def a₃ : ℝ := -5
noncomputable def a₅ : ℝ := 2.4
noncomputable def a₁ : ℝ := -12.4
noncomputable def d : ℝ := 3.7

noncomputable def S₁₅ : ℝ := 15 / 2 * (2 * a₁ + 14 * d)

theorem sum_of_first_fifteen_terms :
  S₁₅ = 202.5 := 
by
  sorry

end sum_of_first_fifteen_terms_l177_177985


namespace no_magpies_left_l177_177352

theorem no_magpies_left (initial_magpies killed_magpies : ℕ) (fly_away : Prop):
  initial_magpies = 40 → killed_magpies = 6 → fly_away → ∀ M : ℕ, M = 0 :=
by
  intro h0 h1 h2
  sorry

end no_magpies_left_l177_177352


namespace prob_and_relation_proof_l177_177031

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l177_177031


namespace width_at_bottom_l177_177333

-- Defining the given values and conditions
def top_width : ℝ := 14
def area : ℝ := 770
def depth : ℝ := 70

-- The proof problem
theorem width_at_bottom (b : ℝ) (h : area = (1/2) * (top_width + b) * depth) : b = 8 :=
by
  sorry

end width_at_bottom_l177_177333


namespace times_reaching_35m_l177_177475

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 30 * t

theorem times_reaching_35m :
  ∃ t1 t2 : ℝ, (abs (t1 - 1.57) < 0.01 ∧ abs (t2 - 4.55) < 0.01) ∧
               projectile_height t1 = 35 ∧ projectile_height t2 = 35 :=
by
  sorry

end times_reaching_35m_l177_177475


namespace no_solution_for_triples_l177_177136

theorem no_solution_for_triples :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (a * b + b * c = 66) ∧ (a * c + b * c = 35) :=
by {
  sorry
}

end no_solution_for_triples_l177_177136


namespace cos_120_eq_neg_half_l177_177121

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177121


namespace determine_values_of_x_l177_177727

variable (x : ℝ)

theorem determine_values_of_x (h1 : 1/x < 3) (h2 : 1/x > -4) : x > 1/3 ∨ x < -1/4 := 
  sorry


end determine_values_of_x_l177_177727


namespace three_digit_number_is_382_l177_177714

theorem three_digit_number_is_382 
  (x : ℕ) 
  (h1 : x >= 100 ∧ x < 1000) 
  (h2 : 7000 + x - (10 * x + 7) = 3555) : 
  x = 382 :=
by 
  sorry

end three_digit_number_is_382_l177_177714


namespace sixteen_powers_five_equals_four_power_ten_l177_177372

theorem sixteen_powers_five_equals_four_power_ten : 
  (16 * 16 * 16 * 16 * 16 = 4 ^ 10) :=
by
  sorry

end sixteen_powers_five_equals_four_power_ten_l177_177372


namespace regular_nonagon_interior_angle_l177_177766

theorem regular_nonagon_interior_angle : 
  let n := 9 in
  180 * (n - 2) / n = 140 :=
by 
  sorry

end regular_nonagon_interior_angle_l177_177766


namespace lisa_savings_l177_177193

-- Define the conditions
def originalPricePerNotebook : ℝ := 3
def numberOfNotebooks : ℕ := 8
def discountRate : ℝ := 0.30
def additionalDiscount : ℝ := 5

-- Define the total savings calculation
def calculateSavings (originalPricePerNotebook : ℝ) (numberOfNotebooks : ℕ) (discountRate : ℝ) (additionalDiscount : ℝ) : ℝ := 
  let totalPriceWithoutDiscount := originalPricePerNotebook * numberOfNotebooks
  let discountedPricePerNotebook := originalPricePerNotebook * (1 - discountRate)
  let totalPriceWith30PercentDiscount := discountedPricePerNotebook * numberOfNotebooks
  let totalPriceWithAllDiscounts := totalPriceWith30PercentDiscount - additionalDiscount
  totalPriceWithoutDiscount - totalPriceWithAllDiscounts

-- Theorem for the proof problem
theorem lisa_savings :
  calculateSavings originalPricePerNotebook numberOfNotebooks discountRate additionalDiscount = 12.20 :=
by
  -- Inserting the proof as sorry
  sorry

end lisa_savings_l177_177193


namespace advertisement_length_l177_177242

noncomputable def movie_length : ℕ := 90
noncomputable def replay_times : ℕ := 6
noncomputable def operation_time : ℕ := 660

theorem advertisement_length : ∃ A : ℕ, 90 * replay_times + 6 * A = operation_time ∧ A = 20 :=
by
  use 20
  sorry

end advertisement_length_l177_177242


namespace is_periodic_l177_177207

noncomputable def f : ℝ → ℝ := sorry

axiom domain (x : ℝ) : true
axiom not_eq_neg1_and_not_eq_0 (x : ℝ) : f x ≠ -1 ∧ f x ≠ 0
axiom functional_eq (x y : ℝ) : f (x - y) = - (f x / (1 + f y))

theorem is_periodic : ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end is_periodic_l177_177207


namespace value_of_a_minus_b_l177_177746

variable {R : Type} [Field R]

noncomputable def f (a b x : R) : R := a * x + b
noncomputable def g (x : R) : R := -2 * x + 7
noncomputable def h (a b x : R) : R := f a b (g x)

theorem value_of_a_minus_b (a b : R) (h_inv : R → R) 
  (h_def : ∀ x, h_inv x = x + 9)
  (h_eq : ∀ x, h a b x = x - 9) : 
  a - b = 5 := by
  sorry

end value_of_a_minus_b_l177_177746


namespace number_of_tires_l177_177635

theorem number_of_tires (n : ℕ)
  (repair_cost : ℕ → ℝ)
  (sales_tax : ℕ → ℝ)
  (total_cost : ℝ) :
  (∀ t, repair_cost t = 7) →
  (∀ t, sales_tax t = 0.5) →
  (total_cost = n * (repair_cost 0 + sales_tax 0)) →
  total_cost = 30 →
  n = 4 :=
by 
  sorry

end number_of_tires_l177_177635


namespace function_domain_exclusion_l177_177585

theorem function_domain_exclusion (x : ℝ) :
  (∃ y, y = 2 / (x - 8)) ↔ x ≠ 8 :=
sorry

end function_domain_exclusion_l177_177585


namespace john_protest_days_l177_177180

theorem john_protest_days (days1: ℕ) (days2: ℕ) (days3: ℕ): 
  days1 = 4 → 
  days2 = (days1 + (days1 / 4)) → 
  days3 = (days2 + (days2 / 2)) → 
  (days1 + days2 + days3) = 17 :=
by
  intros h1 h2 h3
  sorry

end john_protest_days_l177_177180


namespace fraction_identity_l177_177564

theorem fraction_identity (a b c : ℝ) (h1 : a + b + c > 0) (h2 : a + b - c > 0) (h3 : a + c - b > 0) (h4 : b + c - a > 0) 
  (h5 : (a+b+c)/(a+b-c) = 7) (h6 : (a+b+c)/(a+c-b) = 1.75) : (a+b+c)/(b+c-a) = 3.5 :=
by
  sorry

end fraction_identity_l177_177564


namespace incorrect_median_l177_177353

def data_set : List ℕ := [7, 11, 10, 11, 6, 14, 11, 10, 11, 9]

noncomputable def median (l : List ℕ) : ℚ := 
  let sorted := l.toArray.qsort (· ≤ ·) 
  if sorted.size % 2 = 0 then
    (sorted.get! (sorted.size / 2 - 1) + sorted.get! (sorted.size / 2)) / 2
  else
    sorted.get! (sorted.size / 2)

theorem incorrect_median :
  median data_set ≠ 10 := by
  sorry

end incorrect_median_l177_177353


namespace probability_same_color_is_3_over_13_l177_177515

def numGreenBalls : ℕ := 15
def numWhiteBalls : ℕ := 12
def totalBalls : ℕ := numGreenBalls + numWhiteBalls
def numDrawnBalls : ℕ := 3

noncomputable def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def probSameColor : ℚ := 
  (combinations numGreenBalls numDrawnBalls + combinations numWhiteBalls numDrawnBalls) / combinations totalBalls numDrawnBalls

theorem probability_same_color_is_3_over_13 : probSameColor = 3 / 13 := by
  sorry

end probability_same_color_is_3_over_13_l177_177515


namespace lanes_on_road_l177_177247

theorem lanes_on_road (num_lanes : ℕ)
  (h1 : ∀ trucks_per_lane cars_per_lane total_vehicles, 
          cars_per_lane = 2 * (trucks_per_lane * num_lanes) ∧
          trucks_per_lane = 60 ∧
          total_vehicles = num_lanes * (trucks_per_lane + cars_per_lane) ∧
          total_vehicles = 2160) :
  num_lanes = 12 :=
by
  sorry

end lanes_on_road_l177_177247


namespace prob_rain_at_most_3_days_in_may_l177_177625

noncomputable def prob_rain (n k: ℕ) (p: ℝ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def prob_rain_at_most_3_days (n: ℕ) (p: ℝ) : ℝ :=
  prob_rain n 0 p + prob_rain n 1 p + prob_rain n 2 p + prob_rain n 3 p

theorem prob_rain_at_most_3_days_in_may :
  prob_rain_at_most_3_days 31 (1/5) = 0.780 := 
sorry

end prob_rain_at_most_3_days_in_may_l177_177625


namespace find_formula_l177_177347

variable (x : ℕ) (y : ℕ)

theorem find_formula (h1: (x = 2 ∧ y = 10) ∨ (x = 3 ∧ y = 21) ∨ (x = 4 ∧ y = 38) ∨ (x = 5 ∧ y = 61) ∨ (x = 6 ∧ y = 90)) :
  y = 3 * x^2 - 2 * x + 2 :=
  sorry

end find_formula_l177_177347


namespace winning_strategy_l177_177350

theorem winning_strategy (n : ℕ) (take_stones : ℕ → Prop) :
  n = 13 ∧ (∀ k, (k = 1 ∨ k = 2) → take_stones k) →
  (take_stones 12 ∨ take_stones 9 ∨ take_stones 6 ∨ take_stones 3) :=
by sorry

end winning_strategy_l177_177350


namespace three_tenths_of_number_l177_177229

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 15) : (3/10) * x = 54 :=
by
  sorry

end three_tenths_of_number_l177_177229


namespace fraction_product_l177_177505

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l177_177505


namespace min_x_plus_9y_l177_177151

variable {x y : ℝ}

theorem min_x_plus_9y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / y = 1) : x + 9 * y ≥ 16 :=
  sorry

end min_x_plus_9y_l177_177151


namespace height_at_age_10_is_around_146_l177_177627

noncomputable def predicted_height (x : ℝ) : ℝ :=
  7.2 * x + 74

theorem height_at_age_10_is_around_146 :
  abs (predicted_height 10 - 146) < ε :=
by
  let ε := 10
  sorry

end height_at_age_10_is_around_146_l177_177627


namespace coloring_satisfies_conditions_l177_177851

/-- Define what it means for a point to be a lattice point -/
def is_lattice_point (x y : ℤ) : Prop := true

/-- Define the coloring function based on coordinates -/
def color (x y : ℤ) : Prop :=
  (x % 2 = 1 ∧ y % 2 = 1) ∨   -- white
  (x % 2 = 1 ∧ y % 2 = 0) ∨   -- black
  (x % 2 = 0)                 -- red (both (even even) and (even odd) are included)

/-- Proving the method of coloring lattice points satisfies the given conditions -/
theorem coloring_satisfies_conditions :
  (∀ x y : ℤ, is_lattice_point x y → 
    color x y ∧ 
    ∃ (A B C : ℤ × ℤ), 
      (is_lattice_point A.fst A.snd ∧ 
       is_lattice_point B.fst B.snd ∧ 
       is_lattice_point C.fst C.snd ∧ 
       color A.fst A.snd ∧ 
       color B.fst B.snd ∧ 
       color C.fst C.snd ∧
       ∃ D : ℤ × ℤ, 
         (is_lattice_point D.fst D.snd ∧ 
          color D.fst D.snd ∧ 
          D.fst = A.fst + C.fst - B.fst ∧ 
          D.snd = A.snd + C.snd - B.snd))) :=
sorry

end coloring_satisfies_conditions_l177_177851


namespace Tim_soda_cans_l177_177000

noncomputable def initial_cans : ℕ := 22
noncomputable def taken_cans : ℕ := 6
noncomputable def remaining_cans : ℕ := initial_cans - taken_cans
noncomputable def bought_cans : ℕ := remaining_cans / 2
noncomputable def final_cans : ℕ := remaining_cans + bought_cans

theorem Tim_soda_cans :
  final_cans = 24 :=
by
  sorry

end Tim_soda_cans_l177_177000


namespace greatest_possible_grapes_thrown_out_l177_177951

theorem greatest_possible_grapes_thrown_out (n : ℕ) : 
  n % 7 ≤ 6 := by 
  sorry

end greatest_possible_grapes_thrown_out_l177_177951


namespace matthew_ate_8_l177_177194

variable (M P A K : ℕ)

def kimberly_ate_5 : Prop := K = 5
def alvin_eggs : Prop := A = 2 * K - 1
def patrick_eggs : Prop := P = A / 2
def matthew_eggs : Prop := M = 2 * P

theorem matthew_ate_8 (M P A K : ℕ) (h1 : kimberly_ate_5 K) (h2 : alvin_eggs A K) (h3 : patrick_eggs P A) (h4 : matthew_eggs M P) : M = 8 := by
  sorry

end matthew_ate_8_l177_177194


namespace range_of_a_extrema_of_y_l177_177455

variable {a b c : ℝ}

def setA (a b c : ℝ) : Prop := a^2 - b * c - 8 * a + 7 = 0
def setB (a b c : ℝ) : Prop := b^2 + c^2 + b * c - b * a + b = 0

theorem range_of_a (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) : 1 ≤ a ∧ a ≤ 9 :=
sorry

theorem extrema_of_y (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) 
  (y : ℝ) 
  (hy1 : y = a * b + b * c + a * c)
  (hy2 : ∀ x y z : ℝ, setA x y z → setB x y z → y = x * y + y * z + x * z) : 
  y = 88 ∨ y = -56 :=
sorry

end range_of_a_extrema_of_y_l177_177455


namespace find_angle_B_l177_177880

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l177_177880


namespace hcf_of_two_numbers_l177_177618

noncomputable def number1 : ℕ := 414

noncomputable def lcm_factors : Set ℕ := {13, 18}

noncomputable def hcf (a b : ℕ) : ℕ := Nat.gcd a b

-- Statement to prove
theorem hcf_of_two_numbers (Y : ℕ) 
  (H : ℕ) 
  (lcm : ℕ) 
  (H_lcm_factors : lcm = H * 13 * 18)
  (H_lcm_prop : lcm = (number1 * Y) / H)
  (H_Y : Y = (H^2 * 13 * 18) / 414)
  : H = 23 := 
sorry

end hcf_of_two_numbers_l177_177618


namespace algebraic_expression_value_l177_177695

theorem algebraic_expression_value (x : ℝ) (h : x^2 - x - 1 = 0) : x^3 - 2*x + 1 = 2 :=
sorry

end algebraic_expression_value_l177_177695


namespace cosine_120_eq_negative_half_l177_177118

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l177_177118


namespace minimum_distance_from_circle_to_line_l177_177621

noncomputable def circle : set (ℝ × ℝ) := { p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 }

def line (p : ℝ × ℝ) : Prop := 3 * p.1 + 4 * p.2 + 8 = 0

theorem minimum_distance_from_circle_to_line :
  let center := (1, 1) in
  let radius := 1 in
  let distance := abs (3 * center.1 + 4 * center.2 + 8) / sqrt (3^2 + 4^2) in
  distance - radius = 2 :=
by
    let center := (1, 1)
    let radius := 1
    let distance := abs (3 * center.1 + 4 * center.2 + 8) / sqrt (3^2 + 4^2)
    have h : distance - radius = 2 := by sorry
    exact h

end minimum_distance_from_circle_to_line_l177_177621


namespace find_cd_minus_dd_base_d_l177_177330

namespace MathProof

variables (d C D : ℤ)

def digit_sum (C D : ℤ) (d : ℤ) : ℤ := d * C + D
def digit_sum_same (C : ℤ) (d : ℤ) : ℤ := d * C + C

theorem find_cd_minus_dd_base_d (h_d : d > 8) (h_eq : digit_sum C D d + digit_sum_same C d = d^2 + 8 * d + 4) :
  C - D = 1 :=
by
  sorry

end MathProof

end find_cd_minus_dd_base_d_l177_177330


namespace ram_initial_deposit_l177_177005

theorem ram_initial_deposit :
  ∃ P: ℝ, P + 100 = 1100 ∧ 1.20 * 1100 = 1320 ∧ P * 1.32 = 1320 ∧ P = 1000 :=
by
  existsi (1000 : ℝ)
  sorry

end ram_initial_deposit_l177_177005


namespace gcd_72_168_l177_177336

theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
by
  sorry

end gcd_72_168_l177_177336


namespace min_guests_l177_177064

/-- Problem statement:
Given:
1. The total food consumed by all guests is 319 pounds.
2. Each guest consumes no more than 1.5 pounds of meat, 0.3 pounds of vegetables, and 0.2 pounds of dessert.
3. Each guest has equal proportions of meat, vegetables, and dessert.

Prove:
The minimum number of guests such that the total food consumed is less than or equal to 319 pounds is 160.
-/
theorem min_guests (total_food : ℝ) (meat_per_guest : ℝ) (veg_per_guest : ℝ) (dessert_per_guest : ℝ) (G : ℕ) :
  total_food = 319 ∧ meat_per_guest ≤ 1.5 ∧ veg_per_guest ≤ 0.3 ∧ dessert_per_guest ≤ 0.2 ∧
  (meat_per_guest + veg_per_guest + dessert_per_guest = 2.0) →
  G = 160 :=
by
  intros h
  sorry

end min_guests_l177_177064


namespace find_avg_mpg_first_car_l177_177785

def avg_mpg_first_car (x : ℝ) : Prop :=
  let miles_per_month := 450 / 3
  let gallons_first_car := miles_per_month / x
  let gallons_second_car := miles_per_month / 10
  let gallons_third_car := miles_per_month / 15
  let total_gallons := 56 / 2
  gallons_first_car + gallons_second_car + gallons_third_car = total_gallons

theorem find_avg_mpg_first_car : avg_mpg_first_car 50 :=
  sorry

end find_avg_mpg_first_car_l177_177785


namespace savings_calculation_l177_177338

-- Define the conditions
def income := 17000
def ratio_income_expenditure := 5 / 4

-- Prove that the savings are Rs. 3400
theorem savings_calculation (h : income = 5 * 3400): (income - 4 * 3400) = 3400 :=
by sorry

end savings_calculation_l177_177338


namespace series_value_l177_177184

noncomputable def sum_series (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) : ℝ :=
∑' n : ℕ, (if h : n > 0 then
             1 / (((n - 1) * c - (n - 2) * b) * (n * c - (n - 1) * a))
           else 
             0)

theorem series_value (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) :
  sum_series a b c h_positivity h_order = 1 / ((c - a) * b) :=
by
  sorry

end series_value_l177_177184


namespace correct_system_of_equations_l177_177298

theorem correct_system_of_equations
  (x y : ℝ)
  (h1 : x + (1 / 2) * y = 50)
  (h2 : y + (2 / 3) * x = 50) :
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  exact ⟨h1, h2⟩

end correct_system_of_equations_l177_177298


namespace degree_of_vertex_angle_of_isosceles_triangle_l177_177289

theorem degree_of_vertex_angle_of_isosceles_triangle (exterior_angle : ℝ) (h_exterior_angle : exterior_angle = 40) : 
∃ vertex_angle : ℝ, vertex_angle = 140 :=
by 
  sorry

end degree_of_vertex_angle_of_isosceles_triangle_l177_177289


namespace max_f_max_ab_plus_bc_l177_177422

def f (x : ℝ) := |x - 3| - 2 * |x + 1|

theorem max_f : ∃ (m : ℝ), m = 4 ∧ (∀ x : ℝ, f x ≤ m) := 
  sorry

theorem max_ab_plus_bc (a b c : ℝ) : a > 0 ∧ b > 0 → a^2 + 2 * b^2 + c^2 = 4 → (ab + bc) ≤ 2 :=
  sorry

end max_f_max_ab_plus_bc_l177_177422


namespace bus_probabilities_and_chi_squared_l177_177025

noncomputable def prob_on_time_A : ℚ :=
12 / 13

noncomputable def prob_on_time_B : ℚ :=
7 / 8

noncomputable def chi_squared(K2 : ℚ) : Bool :=
K2 > 2.706

theorem bus_probabilities_and_chi_squared :
  prob_on_time_A = 240 / 260 ∧
  prob_on_time_B = 210 / 240 ∧
  chi_squared(3.205) = True :=
by
  -- proof steps will go here
  sorry

end bus_probabilities_and_chi_squared_l177_177025


namespace math_problem_l177_177128

noncomputable def a : ℝ := 3.67
noncomputable def b : ℝ := 4.83
noncomputable def c : ℝ := 2.57
noncomputable def d : ℝ := -0.12
noncomputable def x : ℝ := 7.25
noncomputable def y : ℝ := -0.55

theorem math_problem :
  (3 * a * (4 * b - 2 * y)^2) / (5 * c * d^3 * 0.5 * x) - (2 * x * y^3) / (a * b^2 * c) = -57.179729 := 
sorry

end math_problem_l177_177128


namespace original_mixture_acid_percent_l177_177768

-- Definitions of conditions as per the original problem
def original_acid_percentage (a w : ℕ) (h1 : 4 * a = a + w + 2) (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : Prop :=
  (a * 100) / (a + w) = 100 / 3

-- Main theorem statement
theorem original_mixture_acid_percent (a w : ℕ) 
  (h1 : 4 * a = a + w + 2)
  (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : original_acid_percentage a w h1 h2 :=
sorry

end original_mixture_acid_percent_l177_177768


namespace common_positive_divisors_count_l177_177829

-- To use noncomputable functions
noncomputable theory

open Nat

-- Define the two numbers
def num1 : ℕ := 9240
def num2 : ℕ := 13860

-- Define their greatest common divisor
def gcd_val : ℕ := gcd num1 num2

-- State the prime factorization of the gcd (this can be proven or assumed as a given condition for cleaner code)
def prime_factors_gcd := [(2, 2), (3, 1), (7, 1), (11, 1)]

-- Given the prime factorization, calculate the number of divisors
def number_of_divisors : ℕ := 
  prime_factors_gcd.foldr (λ (factor : ℕ × ℕ) acc, acc * (factor.snd + 1)) 1

-- The final theorem stating the number of common positive divisors of num1 and num2
theorem common_positive_divisors_count : number_of_divisors = 24 := by {
  -- Here would go the proof, which is not required in this task
  sorry
}

end common_positive_divisors_count_l177_177829


namespace find_b_l177_177453

-- Given conditions
def p (x : ℝ) : ℝ := 2 * x - 7
def q (x : ℝ) (b : ℝ) : ℝ := 3 * x - b

-- Assertion we need to prove
theorem find_b (b : ℝ) (h : p (q 3 b) = 3) : b = 4 := 
by
  sorry

end find_b_l177_177453


namespace eva_marks_difference_l177_177538

theorem eva_marks_difference 
    (m2 : ℕ) (a2 : ℕ) (s2 : ℕ) (total_marks : ℕ)
    (h_m2 : m2 = 80) (h_a2 : a2 = 90) (h_s2 : s2 = 90) (h_total_marks : total_marks = 485)
    (m1 a1 s1 : ℕ)
    (h_m1 : m1 = m2 + 10)
    (h_a1 : a1 = a2 - 15)
    (h_s1 : s1 = s2 - 1 / 3 * s2)
    (total_semesters : ℕ)
    (h_total_semesters : total_semesters = m1 + a1 + s1 + m2 + a2 + s2)
    : m1 = m2 + 10 := by
  sorry

end eva_marks_difference_l177_177538


namespace range_of_d_largest_S_n_l177_177678

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d a_1 : ℝ)

-- Conditions
axiom a_3_eq_12 : a_n 3 = 12
axiom S_12_pos : S_n 12 > 0
axiom S_13_neg : S_n 13 < 0
axiom arithmetic_sequence : ∀ n, a_n n = a_1 + (n - 1) * d
axiom sum_of_terms : ∀ n, S_n n = n * a_1 + (n * (n - 1)) / 2 * d

-- Problems
theorem range_of_d : -24/7 < d ∧ d < -3 := sorry

theorem largest_S_n : (∀ m, m > 0 ∧ m < 13 → (S_n 6 >= S_n m)) := sorry

end range_of_d_largest_S_n_l177_177678


namespace increasing_on_neg_reals_l177_177419

variable (f : ℝ → ℝ)

def even_function : Prop := ∀ x : ℝ, f (-x) = f x

def decreasing_on_pos_reals : Prop := ∀ x1 x2 : ℝ, (0 < x1 ∧ 0 < x2 ∧ x1 < x2) → f x1 > f x2

theorem increasing_on_neg_reals
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on_pos_reals f) :
  ∀ x1 x2 : ℝ, (x1 < 0 ∧ x2 < 0 ∧ x1 < x2) → f x1 < f x2 :=
by sorry

end increasing_on_neg_reals_l177_177419


namespace max_product_xyz_l177_177312

theorem max_product_xyz : ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 12 ∧ z ≤ 3 * x ∧ ∀ (a b c : ℕ), a + b + c = 12 → c ≤ 3 * a → 0 < a ∧ 0 < b ∧ 0 < c → a * b * c ≤ 48 :=
by
  sorry

end max_product_xyz_l177_177312


namespace min_value_of_objective_function_l177_177600

theorem min_value_of_objective_function : 
  ∃ (x y : ℝ), 
    (2 * x + y - 2 ≥ 0) ∧ 
    (x - 2 * y + 4 ≥ 0) ∧ 
    (x - 1 ≤ 0) ∧ 
    (∀ (u v: ℝ), 
      (2 * u + v - 2 ≥ 0) → 
      (u - 2 * v + 4 ≥ 0) → 
      (u - 1 ≤ 0) → 
      (3 * u + 2 * v ≥ 3)) :=
  sorry

end min_value_of_objective_function_l177_177600


namespace black_greater_than_gray_by_103_l177_177854

def a := 12
def b := 9
def c := 7
def d := 3

def area (side: ℕ) := side * side

def black_area_sum : ℕ := area a + area c
def gray_area_sum : ℕ := area b + area d

theorem black_greater_than_gray_by_103 :
  black_area_sum - gray_area_sum = 103 := by
  sorry

end black_greater_than_gray_by_103_l177_177854


namespace percent_increase_l177_177052

theorem percent_increase (new_value old_value : ℕ) (h_new : new_value = 480) (h_old : old_value = 320) :
  ((new_value - old_value) / old_value) * 100 = 50 := by
  sorry

end percent_increase_l177_177052


namespace p_and_not_q_l177_177553

def p : Prop :=
  ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≥ (1 / 3) ^ x

def q : Prop :=
  ∃ x : ℕ, x > 0 ∧ 2^x + 2^(1-x) = 2 * Real.sqrt 2

theorem p_and_not_q : p ∧ ¬q :=
by
  have h_p : p := sorry
  have h_not_q : ¬q := sorry
  exact ⟨h_p, h_not_q⟩

end p_and_not_q_l177_177553


namespace income_calculation_l177_177924

-- Define the conditions
def ratio (i e : ℕ) : Prop := 9 * e = 8 * i
def savings (i e : ℕ) : Prop := i - e = 4000

-- The theorem statement
theorem income_calculation (i e : ℕ) (h1 : ratio i e) (h2 : savings i e) : i = 36000 := by
  sorry

end income_calculation_l177_177924


namespace fraction_computation_l177_177500

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l177_177500


namespace prob_X_eq_2_ex_X_eq_4_l177_177809

noncomputable def binomial_distribution : Type := sorry

axiom binomial_6_2_3 : binomial_distribution := sorry

theorem prob_X_eq_2 : P(X = 2) = 20 / 243 := by
  sorry

theorem ex_X_eq_4 : E(X) = 4 := by
  sorry

end prob_X_eq_2_ex_X_eq_4_l177_177809


namespace betty_oranges_l177_177788

theorem betty_oranges (boxes: ℕ) (oranges_per_box: ℕ) (h1: boxes = 3) (h2: oranges_per_box = 8) : boxes * oranges_per_box = 24 :=
by
  -- proof omitted
  sorry

end betty_oranges_l177_177788


namespace height_difference_l177_177838

theorem height_difference (B_height A_height : ℝ) (h : A_height = 0.6 * B_height) :
  (B_height - A_height) / A_height * 100 = 66.67 := 
sorry

end height_difference_l177_177838


namespace part_a_solution_part_b_solution_l177_177510

-- Part (a) Statement in Lean 4
theorem part_a_solution (N : ℕ) (a b : ℕ) (h : N = a * 10^n + b * 10^(n-1)) :
  ∃ (m : ℕ), (N / 10 = m) -> m * 10 = N := sorry

-- Part (b) Statement in Lean 4
theorem part_b_solution (N : ℕ) (a b c : ℕ) (h : N = a * 10^n + b * 10^(n-1) + c * 10^(n-2)) :
  ∃ (m : ℕ), (N / 10^(n-1) = m) -> m * 10^(n-1) = N := sorry

end part_a_solution_part_b_solution_l177_177510


namespace exists_a_b_divisible_l177_177322

theorem exists_a_b_divisible (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := 
sorry

end exists_a_b_divisible_l177_177322


namespace find_angle_B_l177_177861

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l177_177861


namespace winning_strategy_l177_177191

noncomputable def winning_player (n : ℕ) (h : n ≥ 2) : String :=
if n = 2 ∨ n = 4 ∨ n = 8 then "Ariane" else "Bérénice"

theorem winning_strategy (n : ℕ) (h : n ≥ 2) :
  (winning_player n h = "Ariane" ↔ (n = 2 ∨ n = 4 ∨ n = 8)) ∧
  (winning_player n h = "Bérénice" ↔ ¬ (n = 2 ∨ n = 4 ∨ n = 8)) :=
sorry

end winning_strategy_l177_177191


namespace proof_problem_l177_177470

variable {a b c : ℝ}

theorem proof_problem (h_cond : 0 < a ∧ a < b ∧ b < c) : 
  a * c < b * c ∧ a + b < b + c ∧ c / a > c / b := by
  sorry

end proof_problem_l177_177470


namespace inequality_solution_l177_177611

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (2021 * (x ^ 10) - 1 ≥ 2020 * x) ↔ (x = 1) :=
sorry

end inequality_solution_l177_177611


namespace dot_product_a_b_l177_177568

-- Definitions for unit vectors e1 and e2 with given conditions
variables (e1 e2 : ℝ × ℝ)
variables (h_norm_e1 : e1.1^2 + e1.2^2 = 1) -- e1 is a unit vector
variables (h_norm_e2 : e2.1^2 + e2.2^2 = 1) -- e2 is a unit vector
variables (h_angle : e1.1 * e2.1 + e1.2 * e2.2 = -1 / 2) -- angle between e1 and e2 is 120 degrees

-- Definitions for vectors a and b
def a : ℝ × ℝ := (e1.1 + e2.1, e1.2 + e2.2)
def b : ℝ × ℝ := (e1.1 - 3 * e2.1, e1.2 - 3 * e2.2)

-- Theorem to prove
theorem dot_product_a_b : (a e1 e2) • (b e1 e2) = -1 :=
by
  sorry

end dot_product_a_b_l177_177568


namespace find_angle_B_l177_177863

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l177_177863


namespace cubic_polynomial_range_l177_177212

-- Define the conditions and the goal in Lean
theorem cubic_polynomial_range :
  ∀ x : ℝ, (x^2 - 5 * x + 6 < 0) → (41 < x^3 + 5 * x^2 + 6 * x + 1) ∧ (x^3 + 5 * x^2 + 6 * x + 1 < 91) :=
by
  intros x hx
  have h1 : 2 < x := sorry
  have h2 : x < 3 := sorry
  have h3 : (x^3 + 5 * x^2 + 6 * x + 1) > 41 := sorry
  have h4 : (x^3 + 5 * x^2 + 6 * x + 1) < 91 := sorry
  exact ⟨h3, h4⟩ 

end cubic_polynomial_range_l177_177212


namespace parabola_chord_length_eight_l177_177042

noncomputable def parabola_intersection_length (x1 x2: ℝ) (y1 y2: ℝ) : ℝ :=
  if x1 + x2 = 6 ∧ y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 then
    let A := (x1, y1)
    let B := (x2, y2)
    dist A B
  else
    0

theorem parabola_chord_length_eight :
  ∀ (x1 x2 y1 y2 : ℝ), (x1 + x2 = 6) → (y1^2 = 4 * x1) → (y2^2 = 4 * x2) →
  parabola_intersection_length x1 x2 y1 y2 = 8 :=
by
  -- proof goes here
  sorry

end parabola_chord_length_eight_l177_177042


namespace incorrect_average_calculated_initially_l177_177915

theorem incorrect_average_calculated_initially 
    (S : ℕ) 
    (h1 : (S + 75) / 10 = 51) 
    (h2 : (S + 25) = a) 
    : a / 10 = 46 :=
by
  sorry

end incorrect_average_calculated_initially_l177_177915


namespace expenditure_on_digging_l177_177683

noncomputable def volume_of_cylinder (r h : ℝ) := 
  Real.pi * r^2 * h

noncomputable def rate_per_cubic_meter (cost : ℝ) (r h : ℝ) : ℝ := 
  cost / (volume_of_cylinder r h)

theorem expenditure_on_digging (d h : ℝ) (cost : ℝ) (r : ℝ) (π : ℝ) (rate : ℝ)
  (h₀ : d = 3) (h₁ : h = 14) (h₂ : cost = 1682.32) (h₃ : r = d / 2) (h₄ : π = Real.pi) 
  : rate_per_cubic_meter cost r h = 17 := sorry

end expenditure_on_digging_l177_177683


namespace commute_time_abs_diff_l177_177665

theorem commute_time_abs_diff (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  |x - y| = 4 := by
  sorry

end commute_time_abs_diff_l177_177665


namespace apples_used_l177_177516

theorem apples_used (initial_apples remaining_apples : ℕ) (h_initial : initial_apples = 40) (h_remaining : remaining_apples = 39) : initial_apples - remaining_apples = 1 := 
by
  sorry

end apples_used_l177_177516


namespace find_fraction_l177_177563

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h1 : (a + b + c) / (a + b - c) = 7)
variable (h2 : (a + b + c) / (a + c - b) = 1.75)

theorem find_fraction : (a + b + c) / (b + c - a) = 3.5 := 
by {
  sorry
}

end find_fraction_l177_177563


namespace constant_remainder_l177_177130

def polynomial := (12 : ℚ) * (x^3) - (9 : ℚ) * (x^2) + b * x + (8 : ℚ)
def divisor_polynomial := (3 : ℚ) * (x^2) - (4 : ℚ) * x + (2 : ℚ)

theorem constant_remainder (b : ℚ) :
  (∃ r : ℚ, ∀ x : ℚ, (12 * (x^3) - 9 * (x^2) + b * x + 8) % (3 * (x^2) - 4 * x + 2) = r) ↔ b = -4 / 3 :=
by
  sorry

end constant_remainder_l177_177130


namespace fraction_multiplication_l177_177493

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l177_177493


namespace lines_slope_angle_l177_177616

theorem lines_slope_angle (m n : ℝ) (θ₁ θ₂ : ℝ)
  (h1 : L1 = fun x => m * x)
  (h2 : L2 = fun x => n * x)
  (h3 : θ₁ = 3 * θ₂)
  (h4 : m = 3 * n)
  (h5 : θ₂ ≠ 0) :
  m * n = 9 / 4 :=
by
  sorry

end lines_slope_angle_l177_177616


namespace gcf_3150_7350_l177_177932

theorem gcf_3150_7350 : Nat.gcd 3150 7350 = 525 := by
  sorry

end gcf_3150_7350_l177_177932


namespace Pyarelal_loss_is_1800_l177_177063

noncomputable def Ashok_and_Pyarelal_loss (P L : ℝ) : Prop :=
  let Ashok_cap := (1 / 9) * P
  let total_cap := P + Ashok_cap
  let Pyarelal_ratio := P / total_cap
  let total_loss := 2000
  let Pyarelal_loss := Pyarelal_ratio * total_loss
  Pyarelal_loss = 1800

theorem Pyarelal_loss_is_1800 (P : ℝ) (h1 : P > 0) (h2 : L = 2000) :
  Ashok_and_Pyarelal_loss P L := sorry

end Pyarelal_loss_is_1800_l177_177063


namespace automobile_travel_distance_l177_177530

theorem automobile_travel_distance (b s : ℝ) (h1 : s > 0) :
  let rate := (b / 8) / s  -- rate in meters per second
  let rate_km_per_min := rate * (1 / 1000) * 60  -- convert to kilometers per minute
  let time := 5  -- time in minutes
  rate_km_per_min * time = 3 * b / 80 / s := sorry

end automobile_travel_distance_l177_177530


namespace common_divisors_9240_13860_l177_177832

def num_divisors (n : ℕ) : ℕ :=
  -- function to calculate the number of divisors (implementation is not provided here)
  sorry

theorem common_divisors_9240_13860 :
  let d := Nat.gcd 9240 13860
  d = 924 → num_divisors d = 24 := by
  intros d gcd_eq
  rw [gcd_eq]
  sorry

end common_divisors_9240_13860_l177_177832


namespace total_carrots_l177_177305

-- Define constants for the number of carrots grown by each person
def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11
def Michael_carrots : ℕ := 37
def Taylor_carrots : ℕ := 24

-- The proof problem: Prove that the total number of carrots grown is 101
theorem total_carrots : Joan_carrots + Jessica_carrots + Michael_carrots + Taylor_carrots = 101 :=
by
  sorry

end total_carrots_l177_177305


namespace rowing_distance_l177_177043

theorem rowing_distance
  (v_still : ℝ) (v_current : ℝ) (time : ℝ)
  (h1 : v_still = 15) (h2 : v_current = 3) (h3 : time = 17.998560115190784) :
  (v_still + v_current) * 1000 / 3600 * time = 89.99280057595392 :=
by
  rw [h1, h2, h3] -- Apply the given conditions
  -- This will reduce to proving (15 + 3) * 1000 / 3600 * 17.998560115190784 = 89.99280057595392
  sorry

end rowing_distance_l177_177043


namespace eight_pow_n_over_three_eq_512_l177_177798

theorem eight_pow_n_over_three_eq_512 : 8^(9/3) = 512 :=
by
  -- sorry skips the proof
  sorry

end eight_pow_n_over_three_eq_512_l177_177798


namespace students_only_english_l177_177772

variable (total_students both_english_german enrolled_german: ℕ)

theorem students_only_english :
  total_students = 45 ∧ both_english_german = 12 ∧ enrolled_german = 22 ∧
  (∀ S E G B : ℕ, S = total_students ∧ B = both_english_german ∧ G = enrolled_german - B ∧
   (S = E + G + B) → E = 23) :=
by
  sorry

end students_only_english_l177_177772


namespace plates_to_remove_l177_177569

-- Definitions based on the problem conditions
def number_of_plates : ℕ := 38
def weight_per_plate : ℕ := 10
def acceptable_weight : ℕ := 320

-- Theorem to prove
theorem plates_to_remove (initial_weight := number_of_plates * weight_per_plate) 
  (excess_weight := initial_weight - acceptable_weight) 
  (plates_to_remove := excess_weight / weight_per_plate) :
  plates_to_remove = 6 :=
by
  sorry

end plates_to_remove_l177_177569


namespace magnitude_of_sum_l177_177286

variables (a b : ℝ × ℝ)
variables (h1 : a.1 * b.1 + a.2 * b.2 = 0)
variables (h2 : a = (4, 3))
variables (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1)

theorem magnitude_of_sum (a b : ℝ × ℝ) (h1 : a.1 * b.1 + a.2 * b.2 = 0) 
  (h2 : a = (4, 3)) (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1) : 
  (a.1 + 2 * b.1) ^ 2 + (a.2 + 2 * b.2) ^ 2 = 29 :=
by sorry

end magnitude_of_sum_l177_177286


namespace textbook_weight_l177_177179

theorem textbook_weight
  (w : ℝ)
  (bookcase_limit : ℝ := 80)
  (hardcover_books : ℕ := 70)
  (hardcover_weight_per_book : ℝ := 0.5)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (knick_knack_weight : ℝ := 6)
  (over_limit : ℝ := 33)
  (total_items_weight : ℝ := bookcase_limit + over_limit)
  (hardcover_total_weight : ℝ := hardcover_books * hardcover_weight_per_book)
  (knick_knack_total_weight : ℝ := knick_knacks * knick_knack_weight)
  (remaining_weight : ℝ := total_items_weight - (hardcover_total_weight + knick_knack_total_weight)) :
  remaining_weight = textbooks * 2 :=
by
  sorry

end textbook_weight_l177_177179


namespace rachel_total_homework_pages_l177_177607

-- Define the conditions
def math_homework_pages : Nat := 10
def additional_reading_pages : Nat := 3

-- Define the proof goal
def total_homework_pages (math_pages reading_extra : Nat) : Nat :=
  math_pages + (math_pages + reading_extra)

-- The final statement with the expected result
theorem rachel_total_homework_pages : total_homework_pages math_homework_pages additional_reading_pages = 23 :=
by
  sorry

end rachel_total_homework_pages_l177_177607


namespace zero_ending_of_A_l177_177214

theorem zero_ending_of_A (A : ℕ) (h : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c ∣ A ∧ a + b + c = 8 → a * b * c = 10) : 
  (10 ∣ A) ∧ ¬(100 ∣ A) :=
by
  sorry

end zero_ending_of_A_l177_177214


namespace negation_of_universal_statement_l177_177210

theorem negation_of_universal_statement :
  ¬(∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x - a = 0) ↔ ∃ a : ℝ, ∀ x : ℝ, ¬(x > 0 ∧ a * x^2 - 3 * x - a = 0) :=
by sorry

end negation_of_universal_statement_l177_177210


namespace intersect_range_k_l177_177619

theorem intersect_range_k : 
  ∀ k : ℝ, (∃ x y : ℝ, x^2 - (kx + 2)^2 = 6) ↔ 
  -Real.sqrt (5 / 3) < k ∧ k < Real.sqrt (5 / 3) := 
by sorry

end intersect_range_k_l177_177619


namespace first_batch_students_l177_177757

theorem first_batch_students 
  (x : ℕ) 
  (avg1 avg2 avg3 overall_avg : ℝ) 
  (n2 n3 : ℕ) 
  (h_avg1 : avg1 = 45) 
  (h_avg2 : avg2 = 55) 
  (h_avg3 : avg3 = 65) 
  (h_n2 : n2 = 50) 
  (h_n3 : n3 = 60) 
  (h_overall_avg : overall_avg = 56.333333333333336) 
  (h_eq : overall_avg = (45 * x + 55 * 50 + 65 * 60) / (x + 50 + 60)) 
  : x = 40 :=
sorry

end first_batch_students_l177_177757


namespace farm_horses_cows_ratio_l177_177198

variable (x y : ℕ)  -- x is the base variable related to the initial counts, y is the number of horses sold (and cows bought)

theorem farm_horses_cows_ratio (h1 : 4 * x / x = 4)
    (h2 : 13 * (x + y) = 7 * (4 * x - y))
    (h3 : 4 * x - y = (x + y) + 30) :
    y = 15 := sorry

end farm_horses_cows_ratio_l177_177198


namespace train_number_of_cars_l177_177182

theorem train_number_of_cars (lena_cars : ℕ) (time_counted : ℕ) (total_time : ℕ) 
  (cars_in_train : ℕ)
  (h1 : lena_cars = 8) 
  (h2 : time_counted = 15)
  (h3 : total_time = 210)
  (h4 : (8 / 15 : ℚ) * 210 = 112)
  : cars_in_train = 112 :=
sorry

end train_number_of_cars_l177_177182


namespace emma_correct_percentage_l177_177171

theorem emma_correct_percentage (t : ℕ) (lt : t > 0)
  (liam_correct_alone : ℝ := 0.70)
  (liam_overall_correct : ℝ := 0.82)
  (emma_correct_alone : ℝ := 0.85)
  (joint_error_rate : ℝ := 0.05)
  (liam_solved_together_correct : ℝ := liam_overall_correct * t - liam_correct_alone * (t / 2)) :
  (emma_correct_alone * (t / 2) + (1 - joint_error_rate) * liam_solved_together_correct) / t * 100 = 87.15 :=
by
  sorry

end emma_correct_percentage_l177_177171


namespace remainder_modulo_l177_177663

theorem remainder_modulo (N k q r : ℤ) (h1 : N = 1423 * k + 215) (h2 : N = 109 * q + r) : 
  (N - q ^ 2) % 109 = 106 := by
  sorry

end remainder_modulo_l177_177663


namespace cos_120_eq_neg_half_l177_177098

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l177_177098


namespace cos_120_eq_neg_half_l177_177097

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l177_177097


namespace caleb_double_burgers_count_l177_177511

theorem caleb_double_burgers_count
    (S D : ℕ)
    (cost_single cost_double total_hamburgers total_cost : ℝ)
    (h1 : cost_single = 1.00)
    (h2 : cost_double = 1.50)
    (h3 : total_hamburgers = 50)
    (h4 : total_cost = 66.50)
    (h5 : S + D = total_hamburgers)
    (h6 : cost_single * S + cost_double * D = total_cost) :
    D = 33 := 
sorry

end caleb_double_burgers_count_l177_177511


namespace ninth_term_of_geometric_sequence_l177_177375

theorem ninth_term_of_geometric_sequence (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) : a * r^8 = 19683 := by
  sorry

end ninth_term_of_geometric_sequence_l177_177375


namespace maximize_revenue_l177_177655

def revenue_function (p : ℝ) : ℝ :=
  p * (200 - 6 * p)

theorem maximize_revenue :
  ∃ (p : ℝ), (p ≤ 30) ∧ (∀ q : ℝ, (q ≤ 30) → revenue_function p ≥ revenue_function q) ∧ p = 50 / 3 :=
by
  sorry

end maximize_revenue_l177_177655


namespace cos_120_eq_neg_half_l177_177123

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177123


namespace find_m_l177_177650

theorem find_m (m : ℕ) (h₁ : 256 = 4^4) : (256 : ℝ)^(1/4) = (4 : ℝ)^m ↔ m = 1 :=
by
  sorry

end find_m_l177_177650


namespace ratio_of_areas_l177_177644

-- Defining the variables for sides of rectangles
variables {a b c d : ℝ}

-- Given conditions
axiom h1 : a / c = 4 / 5
axiom h2 : b / d = 4 / 5

-- Statement to prove the ratio of areas
theorem ratio_of_areas (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) : (a * b) / (c * d) = 16 / 25 :=
sorry

end ratio_of_areas_l177_177644


namespace average_score_in_5_matches_l177_177916

theorem average_score_in_5_matches 
  (avg1 avg2 : ℕ)
  (total_matches1 total_matches2 : ℕ)
  (h1 : avg1 = 27) 
  (h2 : avg2 = 32)
  (h3 : total_matches1 = 2) 
  (h4 : total_matches2 = 3) 
  : 
  (avg1 * total_matches1 + avg2 * total_matches2) / (total_matches1 + total_matches2) = 30 :=
by 
  sorry

end average_score_in_5_matches_l177_177916


namespace find_angle_B_l177_177883

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l177_177883


namespace greatest_integer_leq_l177_177269

theorem greatest_integer_leq (a b : ℝ) (ha : a = 5^150) (hb : b = 3^150) (c d : ℝ) (hc : c = 5^147) (hd : d = 3^147):
  ⌊ (a + b) / (c + d) ⌋ = 124 := 
sorry

end greatest_integer_leq_l177_177269


namespace fraction_value_l177_177558

theorem fraction_value
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (cond1 : (a + b + c) / (a + b - c) = 7)
  (cond2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 :=
by
  sorry

end fraction_value_l177_177558


namespace greatest_product_sum_2006_l177_177007

theorem greatest_product_sum_2006 :
  (∃ x y : ℤ, x + y = 2006 ∧ ∀ a b : ℤ, a + b = 2006 → a * b ≤ x * y) → 
  ∃ x y : ℤ, x + y = 2006 ∧ x * y = 1006009 :=
by sorry

end greatest_product_sum_2006_l177_177007


namespace exists_arithmetic_seq_perfect_powers_l177_177467

def is_perfect_power (x : ℕ) : Prop := ∃ (a k : ℕ), k > 1 ∧ x = a^k

theorem exists_arithmetic_seq_perfect_powers (n : ℕ) (hn : n > 1) :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → seq i = a + (i - 1) * d)
  ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → is_perfect_power (seq i)
  ∧ d ≠ 0 :=
sorry

end exists_arithmetic_seq_perfect_powers_l177_177467


namespace curling_teams_l177_177440

-- Define the problem conditions and state the theorem
theorem curling_teams (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
sorry

end curling_teams_l177_177440


namespace find_length_of_BC_l177_177177

-- Define the geometrical objects and lengths
variable {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variable (AB AC AM BC : ℝ)
variable (is_midpoint : Midpoint M B C)
variable (known_AB : AB = 7)
variable (known_AC : AC = 6)
variable (known_AM : AM = 4)

theorem find_length_of_BC : BC = Real.sqrt 106 := by
  sorry

end find_length_of_BC_l177_177177


namespace evaluate_expression_l177_177819

theorem evaluate_expression 
  (a c : ℝ)
  (h : a + c = 9) :
  (a * (-1)^2 + (-1) + c) = 8 := 
by 
  sorry

end evaluate_expression_l177_177819


namespace simplify_expression_evaluate_expression_l177_177327

-- Definitions for the first part
variable (a b : ℝ)

theorem simplify_expression (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a^(1/2) * b^(1/3)) * (a^(2/3) * b^(1/2)) / (1/3 * a^(1/6) * b^(5/6)) = 6 * a :=
by
  sorry

-- Definitions for the second part
theorem evaluate_expression :
  (9 / 16)^(1 / 2) + 10^(Real.log 9 / Real.log 10 - 2 * Real.log 2 / Real.log 10) + Real.log (4 * Real.exp 3) 
  - (Real.log 8 / Real.log 9) * (Real.log 33 / Real.log 4) = 7 / 2 :=
by 
  sorry

end simplify_expression_evaluate_expression_l177_177327


namespace fraction_product_l177_177502

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l177_177502


namespace cos_120_degrees_eq_l177_177111

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l177_177111


namespace measure_of_angle_B_l177_177300

theorem measure_of_angle_B (A B C a b c : ℝ) (h₁ : a = A.sin) (h₂ : b = B.sin) (h₃ : c = C.sin)
  (h₄ : (b - a) / (c + a) = c / (a + b)) :
  B = 2 * π / 3 :=
by
  sorry

end measure_of_angle_B_l177_177300


namespace value_of_a_plus_b_is_zero_l177_177716

noncomputable def sum_geometric_sequence (a b : ℝ) (n : ℕ) : ℝ :=
  a * 2^n + b

theorem value_of_a_plus_b_is_zero (a b : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = sum_geometric_sequence a b n) :
  a + b = 0 := 
sorry

end value_of_a_plus_b_is_zero_l177_177716


namespace tangent_line_at_pi_l177_177334

noncomputable def tangent_equation (x : ℝ) : ℝ := x * Real.sin x

theorem tangent_line_at_pi :
  let f := tangent_equation
  let f' := fun x => Real.sin x + x * Real.cos x
  let x : ℝ := Real.pi
  let y : ℝ := f x
  let slope : ℝ := f' x
  y + slope * x - Real.pi^2 = 0 :=
by
  -- This is where the proof would go
  sorry

end tangent_line_at_pi_l177_177334


namespace possible_value_of_b_l177_177993

theorem possible_value_of_b (a b : ℕ) (H1 : b ∣ (5 * a - 1)) (H2 : b ∣ (a - 10)) (H3 : ¬ b ∣ (3 * a + 5)) : 
  b = 49 :=
sorry

end possible_value_of_b_l177_177993


namespace javier_fraction_to_anna_zero_l177_177533

-- Variables
variable (l : ℕ) -- Lee's initial sticker count
variable (j : ℕ) -- Javier's initial sticker count
variable (a : ℕ) -- Anna's initial sticker count

-- Initial conditions
def conditions (l j a : ℕ) : Prop :=
  j = 4 * a ∧ a = 3 * l

-- Javier's final stickers count
def final_javier_stickers (ja : ℕ) (j : ℕ) : ℕ :=
  ja

-- Anna's final stickers count (af = final Anna's stickers)
def final_anna_stickers (af : ℕ) : ℕ :=
  af

-- Lee's final stickers count (lf = final Lee's stickers)
def final_lee_stickers (lf : ℕ) : ℕ :=
  lf

-- Final distribution requirements
def final_distribution (ja af lf : ℕ) : Prop :=
  ja = 2 * af ∧ ja = 3 * lf

-- Correct answer, fraction of stickers given to Anna
def fraction_given_to_anna (j ja : ℕ) : ℚ :=
  ((j - ja) : ℚ) / (j : ℚ)

-- Lean theorem statement to prove
theorem javier_fraction_to_anna_zero
  (l j a ja af lf : ℕ)
  (h_cond : conditions l j a)
  (h_final : final_distribution ja af lf) :
  fraction_given_to_anna j ja = 0 :=
by sorry

end javier_fraction_to_anna_zero_l177_177533


namespace exponent_multiplication_l177_177535

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l177_177535


namespace rectangle_perimeter_l177_177929

theorem rectangle_perimeter {a b c width : ℕ} (h₁: a = 15) (h₂: b = 20) (h₃: c = 25) (w : ℕ) (h₄: w = 5) :
  let area_triangle := (a * b) / 2
  let length := area_triangle / w
  let perimeter := 2 * (length + w)
  perimeter = 70 :=
by
  sorry

end rectangle_perimeter_l177_177929


namespace mike_needs_percentage_to_pass_l177_177314

theorem mike_needs_percentage_to_pass :
  ∀ (mike_score marks_short max_marks : ℕ),
  mike_score = 212 → marks_short = 22 → max_marks = 780 →
  ((mike_score + marks_short : ℕ) / (max_marks : ℕ) : ℚ) * 100 = 30 :=
by
  intros mike_score marks_short max_marks Hmike Hshort Hmax
  rw [Hmike, Hshort, Hmax]
  -- Proof will be filled out here
  sorry

end mike_needs_percentage_to_pass_l177_177314


namespace dolls_count_l177_177385

theorem dolls_count (lisa_dolls : ℕ) (vera_dolls : ℕ) (sophie_dolls : ℕ) (aida_dolls : ℕ)
  (h1 : vera_dolls = 2 * lisa_dolls)
  (h2 : sophie_dolls = 2 * vera_dolls)
  (h3 : aida_dolls = 2 * sophie_dolls)
  (hl : lisa_dolls = 20) :
  aida_dolls + sophie_dolls + vera_dolls + lisa_dolls = 300 :=
by
  sorry

end dolls_count_l177_177385


namespace bus_probabilities_and_chi_squared_l177_177024

noncomputable def prob_on_time_A : ℚ :=
12 / 13

noncomputable def prob_on_time_B : ℚ :=
7 / 8

noncomputable def chi_squared(K2 : ℚ) : Bool :=
K2 > 2.706

theorem bus_probabilities_and_chi_squared :
  prob_on_time_A = 240 / 260 ∧
  prob_on_time_B = 210 / 240 ∧
  chi_squared(3.205) = True :=
by
  -- proof steps will go here
  sorry

end bus_probabilities_and_chi_squared_l177_177024


namespace number_of_comedies_l177_177796

variables (T a : ℝ)
variables (dramas thrillers scifi comedies action_movies : ℝ)

-- Define the total number of movies rented
def total_movies_rented := T

-- Conditions
def comedies_percentage : ℝ := 0.48
def action_movies_percentage : ℝ := 0.16
def remaining_percentage : ℝ := 1 - comedies_percentage - action_movies_percentage

def dramas_count := 3 * a
def thrillers_count := 2 * dramas_count
def scifi_count := a

-- The equation representing the total rentals
noncomputable def total_rentals_equation : Prop := 
  T = comedies_percentage * T + action_movies_percentage * T + dramas_count + thrillers_count + scifi_count

-- Number of comedies in terms of T
noncomputable def comedies_in_terms_of_T : ℝ := comedies_percentage * T

-- To prove: 
-- Prove number of comedies in terms of a
theorem number_of_comedies (a : ℝ) : comedies_in_terms_of_T = (40 / 3) * a :=
by sorry

end number_of_comedies_l177_177796


namespace b_minus_d_sq_value_l177_177771

theorem b_minus_d_sq_value 
  (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3)
  (h3 : 2 * a - 3 * b + c + 4 * d = 17) :
  (b - d) ^ 2 = 25 :=
by
  sorry

end b_minus_d_sq_value_l177_177771


namespace man_speed_l177_177241

theorem man_speed (distance_meters time_minutes : ℝ) (h_distance : distance_meters = 1250) (h_time : time_minutes = 15) :
  (distance_meters / 1000) / (time_minutes / 60) = 5 :=
by
  sorry

end man_speed_l177_177241


namespace train_speed_first_part_l177_177964

theorem train_speed_first_part (x v : ℝ) (h1 : 0 < x) (h2 : 0 < v) 
  (h_avg_speed : (3 * x) / (x / v + 2 * x / 20) = 22.5) : v = 30 :=
sorry

end train_speed_first_part_l177_177964


namespace polynomial_divisibility_by_5_l177_177643

theorem polynomial_divisibility_by_5
  (a b c d : ℤ)
  (divisible : ∀ x : ℤ, 5 ∣ (a * x ^ 3 + b * x ^ 2 + c * x + d)) :
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d :=
sorry

end polynomial_divisibility_by_5_l177_177643


namespace special_fractions_distinct_integers_count_l177_177129

def is_special_fraction (a b : ℕ) : Prop := a + b = 20 ∧ a > 0 ∧ b > 0

def special_fractions : List ℚ :=
  List.filterMap (λ (p : ℕ × ℕ), if is_special_fraction p.1 p.2 then some (p.1 / (p.2 : ℚ)) else none)
    (List.product (List.range 20) (List.range 20))

def sum_of_three_special_fractions : List ℚ :=
  List.bind special_fractions (λ x, List.bind special_fractions (λ y, List.map (λ z, x + y + z) special_fractions))

def distinct_integers_from_special_fractions : Finset ℤ :=
  (List.filterMap (λ q, if q.den = 1 then some q.num else none) sum_of_three_special_fractions).toFinset

theorem special_fractions_distinct_integers_count : distinct_integers_from_special_fractions.card = 2 := 
  by
  sorry

end special_fractions_distinct_integers_count_l177_177129


namespace point_on_line_l177_177540

theorem point_on_line : 
  ∃ t : ℚ, (3 * t + 1 = 0) ∧ ((2 - 4) / (t - 1) = (7 - 4) / (3 - 1)) :=
by
  sorry

end point_on_line_l177_177540


namespace two_point_four_times_eight_point_two_l177_177675

theorem two_point_four_times_eight_point_two (x y z : ℝ) (hx : x = 2.4) (hy : y = 8.2) (hz : z = 4.8 + 5.2) :
  x * y * z = 2.4 * 8.2 * 10 ∧ abs (x * y * z - 200) < abs (x * y * z - 150) ∧
  abs (x * y * z - 200) < abs (x * y * z - 250) ∧
  abs (x * y * z - 200) < abs (x * y * z - 300) ∧
  abs (x * y * z - 200) < abs (x * y * z - 350) := by
  sorry

end two_point_four_times_eight_point_two_l177_177675


namespace count_parallelograms_392_l177_177958

-- Define the conditions in Lean
def is_lattice_point (x y : ℕ) : Prop :=
  ∃ q : ℕ, x = q ∧ y = q

def on_line_y_eq_x (x y : ℕ) : Prop :=
  y = x ∧ is_lattice_point x y

def on_line_y_eq_mx (x y : ℕ) (m : ℕ) : Prop :=
  y = m * x ∧ is_lattice_point x y ∧ m > 1

def area_parallelogram (q s m : ℕ) : ℕ :=
  (m - 1) * q * s

-- Define the target theorem
theorem count_parallelograms_392 :
  (∀ (q s m : ℕ),
    on_line_y_eq_x q q →
    on_line_y_eq_mx s (m * s) m →
    area_parallelogram q s m = 250000) →
  (∃! n : ℕ, n = 392) :=
sorry

end count_parallelograms_392_l177_177958


namespace min_distance_origin_to_line_l177_177997

theorem min_distance_origin_to_line 
  (x y : ℝ) 
  (h : x + y = 4) : 
  ∃ P : ℝ, P = 2 * Real.sqrt 2 ∧ 
    (∀ Q : ℝ, Q = Real.sqrt (x^2 + y^2) → P ≤ Q) :=
by
  sorry

end min_distance_origin_to_line_l177_177997


namespace non_defective_probability_l177_177520

-- Definitions based on conditions
def event_a := { p : ℙ // p ∉ event_b ∪ event_c }
def event_b : { p : ℙ // p ∈ event_b }
def event_c : { p : ℙ // p ∈ event_c }

-- Probabilities of events based on conditions
lemma prob_b : ℙ(event_b) = 0.03 := sorry 
lemma prob_c : ℙ(event_c) = 0.01 := sorry

-- Proof Statement
theorem non_defective_probability :
  ℙ(event_a) = 0.96 :=
by
  rw [event_a, compl_union]
  rw [prob_b, prob_c]
  norm_num
  -- Detailed proof omitted
  sorry


end non_defective_probability_l177_177520


namespace min_sum_of_distances_l177_177160

noncomputable def distance_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / sqrt (a^2 + b^2)

noncomputable def sum_of_distances (a : ℝ) : ℝ :=
  let P := (a^2, 2 * a)
  distance_point_to_line P 4 (-3) 6 + abs (P.1 + 1)

theorem min_sum_of_distances : ∃ a: ℝ, sum_of_distances a = 2 :=
by
  sorry

end min_sum_of_distances_l177_177160


namespace test_end_time_l177_177044

def start_time := 12 * 60 + 35  -- 12 hours 35 minutes in minutes
def duration := 4 * 60 + 50     -- 4 hours 50 minutes in minutes

theorem test_end_time : (start_time + duration) = 17 * 60 + 25 := by
  sorry

end test_end_time_l177_177044


namespace contractor_total_received_l177_177954

-- Define the conditions
def days_engaged : ℕ := 30
def daily_earnings : ℝ := 25
def fine_per_absence_day : ℝ := 7.50
def days_absent : ℕ := 4

-- Define the days worked based on conditions
def days_worked : ℕ := days_engaged - days_absent

-- Define the total earnings and total fines
def total_earnings : ℝ := days_worked * daily_earnings
def total_fines : ℝ := days_absent * fine_per_absence_day

-- Define the total amount received
def total_amount_received : ℝ := total_earnings - total_fines

-- State the theorem
theorem contractor_total_received :
  total_amount_received = 620 := 
by
  sorry

end contractor_total_received_l177_177954


namespace christopher_avg_speed_l177_177015

-- Definition of a palindrome (not required for this proof, but helpful for context)
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Given conditions
def initial_reading : ℕ := 12321
def final_reading : ℕ := 12421
def duration : ℕ := 4

-- Definition of average speed calculation
def average_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- Main theorem to prove
theorem christopher_avg_speed : average_speed (final_reading - initial_reading) duration = 25 :=
by
  sorry

end christopher_avg_speed_l177_177015


namespace find_angle_B_l177_177875

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l177_177875


namespace total_time_to_4864_and_back_l177_177230

variable (speed_boat : ℝ) (speed_stream : ℝ) (distance : ℝ)
variable (Sboat : speed_boat = 14) (Sstream : speed_stream = 1.2) (Dist : distance = 4864)

theorem total_time_to_4864_and_back :
  let speed_downstream := speed_boat + speed_stream
  let speed_upstream := speed_boat - speed_stream
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_time := time_downstream + time_upstream
  total_time = 700 :=
by
  sorry

end total_time_to_4864_and_back_l177_177230


namespace forty_percent_of_number_l177_177738

/--
Given that (1/4) * (1/3) * (2/5) * N = 30, prove that 0.40 * N = 360.
-/
theorem forty_percent_of_number {N : ℝ} (h : (1/4 : ℝ) * (1/3) * (2/5) * N = 30) : 0.40 * N = 360 := 
by
  sorry

end forty_percent_of_number_l177_177738


namespace final_answer_correct_l177_177991

-- Define the initial volume V0
def V0 := 1

-- Define the volume increment ratio for new tetrahedra
def volume_ratio := (1 : ℚ) / 27

-- Define the recursive volume increments
def ΔP1 := 4 * volume_ratio
def ΔP2 := 16 * volume_ratio
def ΔP3 := 64 * volume_ratio
def ΔP4 := 256 * volume_ratio

-- Define the total volume V4
def V4 := V0 + ΔP1 + ΔP2 + ΔP3 + ΔP4

-- The target volume as a rational number
def target_volume := 367 / 27

-- Define the fraction components
def m := 367
def n := 27

-- Define the final answer
def final_answer := m + n

-- Proof statement to verify the final answer
theorem final_answer_correct :
  V4 = target_volume ∧ (Nat.gcd m n = 1) ∧ final_answer = 394 :=
by
  -- The specifics of the proof are omitted
  sorry

end final_answer_correct_l177_177991


namespace exists_nat_with_digit_sum_1000_and_square_sum_1000000_l177_177680

-- Define a function to calculate the sum of digits in base-10
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem exists_nat_with_digit_sum_1000_and_square_sum_1000000 :
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = 1000000 :=
by
  sorry

end exists_nat_with_digit_sum_1000_and_square_sum_1000000_l177_177680


namespace total_arrangements_l177_177486

def count_arrangements : Nat :=
  let male_positions := 3
  let female_positions := 3
  let male_arrangements := Nat.factorial male_positions
  let female_arrangements := Nat.factorial (female_positions - 1)
  male_arrangements * female_arrangements / (male_positions - female_positions + 1)

theorem total_arrangements : count_arrangements = 36 := by
  sorry

end total_arrangements_l177_177486


namespace chandra_pairings_l177_177972

variable (bowls : ℕ) (glasses : ℕ)

theorem chandra_pairings : 
  bowls = 5 → 
  glasses = 4 → 
  bowls * glasses = 20 :=
by intros; 
    sorry

end chandra_pairings_l177_177972


namespace cos_120_eq_neg_half_l177_177082

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177082


namespace units_digit_37_pow_37_l177_177011

theorem units_digit_37_pow_37: (37^37) % 10 = 7 :=
by sorry

end units_digit_37_pow_37_l177_177011


namespace candy_distribution_l177_177016

theorem candy_distribution (A B : ℕ) (h1 : 7 * A = B + 12) (h2 : 3 * A = B - 20) : A + B = 52 :=
by {
  -- proof goes here
  sorry
}

end candy_distribution_l177_177016


namespace binom_divisible_by_prime_l177_177323

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (h1 : 1 ≤ k) (h2 : k ≤ p - 1) : p ∣ Nat.choose p k :=
sorry

end binom_divisible_by_prime_l177_177323


namespace max_banner_area_l177_177381

theorem max_banner_area (x y : ℕ) (cost_constraint : 330 * x + 450 * y ≤ 10000) : x * y ≤ 165 :=
by
  sorry

end max_banner_area_l177_177381


namespace number_of_possible_lengths_of_diagonal_l177_177689

theorem number_of_possible_lengths_of_diagonal :
  ∃ n : ℕ, n = 13 ∧
  (∀ y : ℕ, (5 ≤ y ∧ y ≤ 17) ↔ (y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9 ∨
   y = 10 ∨ y = 11 ∨ y = 12 ∨ y = 13 ∨ y = 14 ∨ y = 15 ∨ y = 16 ∨ y = 17)) :=
by
  exists 13
  sorry

end number_of_possible_lengths_of_diagonal_l177_177689


namespace find_multiplier_l177_177053

theorem find_multiplier (x : ℤ) : 
  30 * x - 138 = 102 ↔ x = 8 := 
by
  sorry

end find_multiplier_l177_177053


namespace no_integers_exist_l177_177948

theorem no_integers_exist :
  ¬ (∃ x y : ℤ, (x + 2019) * (x + 2020) + (x + 2020) * (x + 2021) + (x + 2019) * (x + 2021) = y^2) :=
by
  sorry

end no_integers_exist_l177_177948


namespace fencing_required_l177_177943

theorem fencing_required (L W : ℕ) (hL : L = 10) (hA : L * W = 600) : L + 2 * W = 130 :=
by
  sorry

end fencing_required_l177_177943


namespace no_such_function_exists_l177_177267

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, (f 0 > 0) ∧ (∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :=
by
  -- proof to be completed
  sorry

end no_such_function_exists_l177_177267


namespace tank_capacity_is_correct_l177_177382

-- Definition of the problem conditions
def initial_fraction := 1 / 3
def added_water := 180
def final_fraction := 2 / 3

-- Capacity of the tank
noncomputable def tank_capacity : ℕ := 540

-- Proof statement
theorem tank_capacity_is_correct (x : ℕ) :
  (initial_fraction * x + added_water = final_fraction * x) → x = tank_capacity := 
by
  -- This is where the proof would go
  sorry

end tank_capacity_is_correct_l177_177382


namespace speed_difference_valid_l177_177632

-- Definitions of the conditions
def speed (s : ℕ) : ℕ := s^2 + 2 * s

-- Theorem statement that needs to be proven
theorem speed_difference_valid : 
  (speed 5 - speed 3) = 20 :=
  sorry

end speed_difference_valid_l177_177632


namespace cos_120_eq_neg_half_l177_177125

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177125


namespace polynomials_equality_l177_177243

open Polynomial

variable {F : Type*} [Field F]

theorem polynomials_equality (P Q : Polynomial F) (h : ∀ x, P.eval (P.eval (P.eval x)) = Q.eval (Q.eval (Q.eval x)) ∧ P.eval (P.eval (P.eval x)) = Q.eval (P.eval (P.eval x))) : 
  P = Q := 
sorry

end polynomials_equality_l177_177243


namespace find_m_l177_177175

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {m : ℕ}

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def initial_condition (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def q_condition (q : ℝ) : Prop :=
  abs q ≠ 1

def a_m_condition (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a m = a 1 * a 2 * a 3 * a 4 * a 5

-- Theorem to prove
theorem find_m (h1 : geometric_sequence a q) (h2 : initial_condition a) (h3 : q_condition q) (h4 : a_m_condition a m) : m = 11 :=
  sorry

end find_m_l177_177175


namespace sum_of_squares_2222_l177_177133

theorem sum_of_squares_2222 :
  ∀ (N : ℕ), (∃ (k : ℕ), N = 2 * 10^k - 1) → (∀ (a b : ℤ), N = a^2 + b^2 ↔ N = 2) :=
by sorry

end sum_of_squares_2222_l177_177133


namespace cos_120_degrees_l177_177069

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l177_177069


namespace find_angle_B_l177_177864

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l177_177864


namespace smallest_number_l177_177967

theorem smallest_number (a b c d : ℝ) (h1 : a = -5) (h2 : b = 0) (h3 : c = 1/2) (h4 : d = Real.sqrt 2) : a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by
  sorry

end smallest_number_l177_177967


namespace sum_of_digits_decrease_by_10_percent_l177_177378

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum -- Assuming this method computes the sum of the digits

theorem sum_of_digits_decrease_by_10_percent :
  ∃ (n m : ℕ), m = 11 * n / 10 ∧ sum_of_digits m = 9 * sum_of_digits n / 10 :=
by
  sorry

end sum_of_digits_decrease_by_10_percent_l177_177378


namespace simplify_and_evaluate_expression_l177_177326

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1/2) : x^2 * (x - 1) - x * (x^2 + x - 1) = 0 := by
  sorry

end simplify_and_evaluate_expression_l177_177326


namespace prob_and_relation_proof_l177_177029

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l177_177029


namespace solution_interval_l177_177456

theorem solution_interval:
  ∃ x : ℝ, (x^3 = 2^(2-x)) ∧ 1 < x ∧ x < 2 :=
by
  sorry

end solution_interval_l177_177456


namespace polar_coordinates_of_2_neg2_l177_177624

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (ρ, θ)

theorem polar_coordinates_of_2_neg2 :
  polar_coordinates 2 (-2) = (2 * Real.sqrt 2, -Real.pi / 4) :=
by
  sorry

end polar_coordinates_of_2_neg2_l177_177624


namespace simplify_fraction_l177_177609

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + 4 * i) = -2 / 17 - (9 / 17) * i :=
by
  sorry

end simplify_fraction_l177_177609


namespace triangle_angle_B_l177_177874

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l177_177874


namespace fraction_computation_l177_177498

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l177_177498


namespace one_third_of_product_l177_177799

theorem one_third_of_product (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 4) : (1 / 3 : ℚ) * (a * b * c : ℕ) = 84 := by
  sorry

end one_third_of_product_l177_177799


namespace ratio_between_house_and_park_l177_177641

theorem ratio_between_house_and_park (w x y : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0)
    (h : y / w = x / w + (x + y) / (10 * w)) : x / y = 9 / 11 :=
by 
  sorry

end ratio_between_house_and_park_l177_177641


namespace sum_integers_neg40_to_60_l177_177936

theorem sum_integers_neg40_to_60 : (Finset.range (60 + 41)).sum (fun i => i - 40) = 1010 := by
  sorry

end sum_integers_neg40_to_60_l177_177936


namespace trajectory_equation_equation_of_line_l177_177692

-- Define the parabola and the trajectory
def parabola (x y : ℝ) := y^2 = 16 * x
def trajectory (x y : ℝ) := y^2 = 4 * x

-- Define the properties of the point P and the line l
def is_midpoint (P A B : ℝ × ℝ) :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line_through_point (x y k : ℝ) := 
  k * x + y = 1

-- Proof problem (Ⅰ): trajectory of the midpoints of segments perpendicular to the x-axis from points on the parabola
theorem trajectory_equation : ∀ (M : ℝ × ℝ), 
  (∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧ is_midpoint M P (P.1, 0)) → 
  trajectory M.1 M.2 :=
sorry

-- Proof problem (Ⅱ): equation of line l
theorem equation_of_line : ∀ (A B P : ℝ × ℝ), 
  trajectory A.1 A.2 → trajectory B.1 B.2 → 
  P = (3,2) → is_midpoint P A B → 
  ∃ k, line_through_point (A.1 - B.1) (A.2 - B.2) k :=
sorry

end trajectory_equation_equation_of_line_l177_177692


namespace cos_120_eq_neg_half_l177_177105

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l177_177105


namespace douglas_won_percentage_l177_177436

theorem douglas_won_percentage (p_X p_Y : ℝ) (r : ℝ) (V : ℝ) (h1 : p_X = 0.76) (h2 : p_Y = 0.4000000000000002) (h3 : r = 2) :
  (1.52 * V + 0.4000000000000002 * V) / (2 * V + V) * 100 = 64 := by
  sorry

end douglas_won_percentage_l177_177436


namespace exist_equal_success_rate_l177_177787

noncomputable def S : ℕ → ℝ := sorry -- Definition of S(N), the number of successful free throws

theorem exist_equal_success_rate (N1 N2 : ℕ) 
  (h1 : S N1 < 0.8 * N1) 
  (h2 : S N2 > 0.8 * N2) : 
  ∃ (N : ℕ), N1 ≤ N ∧ N ≤ N2 ∧ S N = 0.8 * N :=
sorry

end exist_equal_success_rate_l177_177787


namespace arccos_one_eq_zero_l177_177949

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l177_177949


namespace isosceles_trapezoid_height_l177_177135

theorem isosceles_trapezoid_height (S h : ℝ) (h_nonneg : 0 ≤ h) 
  (diag_perpendicular : S = (1 / 2) * h^2) : h = Real.sqrt S :=
by
  sorry

end isosceles_trapezoid_height_l177_177135


namespace three_at_five_l177_177573

def op_at (a b : ℤ) : ℤ := 3 * a - 3 * b

theorem three_at_five : op_at 3 5 = -6 :=
by
  sorry

end three_at_five_l177_177573


namespace never_2003_pieces_l177_177226

theorem never_2003_pieces :
  ¬∃ n : ℕ, (n = 5 + 4 * k) ∧ (n = 2003) :=
by
  sorry

end never_2003_pieces_l177_177226


namespace inverse_proportion_quadrants_l177_177843

theorem inverse_proportion_quadrants (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (k - 3) / x > 0) ∧ (x < 0 → (k - 3) / x < 0))) → k > 3 :=
by
  intros h
  sorry

end inverse_proportion_quadrants_l177_177843


namespace window_treatments_total_cost_l177_177598

def sheers_cost_per_pair := 40
def drapes_cost_per_pair := 60
def number_of_windows := 3

theorem window_treatments_total_cost :
  (number_of_windows * sheers_cost_per_pair) + (number_of_windows * drapes_cost_per_pair) = 300 :=
by 
  -- calculations omitted
  sorry

end window_treatments_total_cost_l177_177598


namespace max_gross_profit_price_l177_177239

def purchase_price : ℝ := 20
def Q (P : ℝ) : ℝ := 8300 - 170 * P - P^2
def L (P : ℝ) : ℝ := (8300 - 170 * P - P^2) * (P - 20)

theorem max_gross_profit_price : ∃ P : ℝ, (∀ x : ℝ, L x ≤ L P) ∧ P = 30 :=
by
  sorry

end max_gross_profit_price_l177_177239


namespace points_in_circle_l177_177739

theorem points_in_circle (points : Finset (ℝ × ℝ)) (h_card : points.card = 51) :
  ∃ (c : ℝ × ℝ), (Finset.filter (λ p, dist p c < 1 / 7) points).card ≥ 3 :=
by
  -- Problem conditions
  have conditions : true := true.intro
  -- Placeholder for proof, Lean requires a non-empty proof body
  sorry

end points_in_circle_l177_177739


namespace patanjali_distance_first_day_l177_177906

theorem patanjali_distance_first_day
  (h : ℕ)
  (H1 : 3 * h + 4 * (h - 1) + 4 * h = 62) :
  3 * h = 18 :=
by
  sorry

end patanjali_distance_first_day_l177_177906


namespace calculate_material_needed_l177_177657

theorem calculate_material_needed (area : ℝ) (pi_approx : ℝ) (extra_material : ℝ) (r : ℝ) (C : ℝ) : 
  area = 50.24 → pi_approx = 3.14 → extra_material = 4 → pi_approx * r ^ 2 = area → 
  C = 2 * pi_approx * r →
  C + extra_material = 29.12 :=
by
  intros h_area h_pi h_extra h_area_eq h_C_eq
  sorry

end calculate_material_needed_l177_177657


namespace diff_of_squares_value_l177_177939

theorem diff_of_squares_value :
  535^2 - 465^2 = 70000 :=
by sorry

end diff_of_squares_value_l177_177939


namespace slower_speed_is_10_l177_177046

-- Define the problem conditions
def walked_distance (faster_speed slower_speed actual_distance extra_distance : ℕ) : Prop :=
  actual_distance / slower_speed = (actual_distance + extra_distance) / faster_speed

-- Define main statement to prove
theorem slower_speed_is_10 (actual_distance : ℕ) (extra_distance : ℕ) (faster_speed : ℕ) (slower_speed : ℕ) :
  walked_distance faster_speed slower_speed actual_distance extra_distance ∧ 
  faster_speed = 15 ∧ extra_distance = 15 ∧ actual_distance = 30 → slower_speed = 10 :=
by
  intro h
  sorry

end slower_speed_is_10_l177_177046


namespace fraction_computation_l177_177497

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l177_177497


namespace integer_solution_inequalities_l177_177637

theorem integer_solution_inequalities (x : ℤ) (h1 : x + 12 > 14) (h2 : -3 * x > -9) : x = 2 :=
by
  sorry

end integer_solution_inequalities_l177_177637


namespace f_is_odd_and_periodic_l177_177599

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
axiom h2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)

theorem f_is_odd_and_periodic : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∃ T : ℝ, T = 40 ∧ ∀ x : ℝ, f (x + T) = f x) :=
by
  sorry

end f_is_odd_and_periodic_l177_177599


namespace arithmetic_sequence_sum_q_l177_177914

theorem arithmetic_sequence_sum_q (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 2) + a (n + 1) = 2 * a n)
  (hq : q ≠ 1) :
  S 5 = 11 :=
sorry

end arithmetic_sequence_sum_q_l177_177914


namespace finite_solutions_exists_l177_177740

variable (f : ℕ → ℝ)
variable (h1 : ∀ x : ℕ, f x > 0)
variable (h2 : Tendsto f atTop (𝓝 0))

theorem finite_solutions_exists : 
  ∃ N : ℕ, ∀ m n p : ℕ, f m + f n + f p = 1 → m ≤ N ∧ n ≤ N ∧ p ≤ N :=
sorry

end finite_solutions_exists_l177_177740


namespace triangle_angle_B_l177_177872

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l177_177872


namespace fish_swim_eastward_l177_177034

-- Define the conditions
variables (E : ℕ)
variable (total_fish_left : ℕ := 2870)
variable (fish_westward : ℕ := 1800)
variable (fish_north : ℕ := 500)
variable (fishwestward_not_caught : ℕ := fish_westward / 4)
variable (fishnorth_not_caught : ℕ := fish_north)
variable (fish_tobe_left_after_caught : ℕ := total_fish_left - fishwestward_not_caught - fishnorth_not_caught)

-- Define the theorem to prove
theorem fish_swim_eastward (h : 3 / 5 * E = fish_tobe_left_after_caught) : E = 3200 := 
by
  sorry

end fish_swim_eastward_l177_177034


namespace window_treatments_cost_l177_177595

-- Define the costs and the number of windows
def cost_sheers : ℝ := 40.00
def cost_drapes : ℝ := 60.00
def number_of_windows : ℕ := 3

-- Define the total cost calculation
def total_cost := (cost_sheers + cost_drapes) * number_of_windows

-- State the theorem that needs to be proved
theorem window_treatments_cost : total_cost = 300.00 :=
by
  sorry

end window_treatments_cost_l177_177595


namespace intersection_A_B_l177_177823

def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
def B : Set ℝ := { x : ℝ | x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l177_177823


namespace least_xy_value_l177_177417

theorem least_xy_value (x y : ℕ) (hposx : x > 0) (hposy : y > 0) (h : 1/x + 1/(3*y) = 1/8) :
  xy = 96 :=
by
  sorry

end least_xy_value_l177_177417


namespace cuboid_height_l177_177407

/-- Given a cuboid with surface area 2400 cm², length 15 cm, and breadth 10 cm,
    prove that the height is 42 cm. -/
theorem cuboid_height (SA l w : ℝ) (h : ℝ) : 
  SA = 2400 → l = 15 → w = 10 → 2 * (l * w + l * h + w * h) = SA → h = 42 :=
by
  intros hSA hl hw hformula
  sorry

end cuboid_height_l177_177407


namespace total_gas_consumed_l177_177795

def highway_consumption_rate : ℕ := 3
def city_consumption_rate : ℕ := 5

-- Distances driven each day
def day_1_highway_miles : ℕ := 200
def day_1_city_miles : ℕ := 300

def day_2_highway_miles : ℕ := 300
def day_2_city_miles : ℕ := 500

def day_3_highway_miles : ℕ := 150
def day_3_city_miles : ℕ := 350

-- Function to calculate the total consumption for a given day
def daily_consumption (highway_miles city_miles : ℕ) : ℕ :=
  (highway_miles * highway_consumption_rate) + (city_miles * city_consumption_rate)

-- Total consumption over three days
def total_consumption : ℕ :=
  (daily_consumption day_1_highway_miles day_1_city_miles) +
  (daily_consumption day_2_highway_miles day_2_city_miles) +
  (daily_consumption day_3_highway_miles day_3_city_miles)

-- Theorem stating the total consumption over the three days
theorem total_gas_consumed : total_consumption = 7700 := by
  sorry

end total_gas_consumed_l177_177795


namespace least_xy_value_l177_177416

theorem least_xy_value (x y : ℕ) (hposx : x > 0) (hposy : y > 0) (h : 1/x + 1/(3*y) = 1/8) :
  xy = 96 :=
by
  sorry

end least_xy_value_l177_177416


namespace find_angle_B_l177_177860

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l177_177860


namespace good_tipper_bill_amount_l177_177988

theorem good_tipper_bill_amount {B : ℝ} 
    (h₁ : 0.05 * B + 1/20 ≥ 0.20 * B) 
    (h₂ : 0.15 * B = 3.90) : 
    B = 26.00 := 
by 
  sorry

end good_tipper_bill_amount_l177_177988


namespace doubling_period_l177_177968

theorem doubling_period (initial_capacity: ℝ) (final_capacity: ℝ) (years: ℝ) (initial_year: ℝ) (final_year: ℝ) (doubling_period: ℝ) :
  initial_capacity = 0.4 → final_capacity = 4100 → years = (final_year - initial_year) →
  initial_year = 2000 → final_year = 2050 →
  2 ^ (years / doubling_period) * initial_capacity = final_capacity :=
by
  intros h_initial h_final h_years h_i_year h_f_year
  sorry

end doubling_period_l177_177968


namespace coordinate_plane_condition_l177_177297

theorem coordinate_plane_condition (a : ℝ) :
  a - 1 < 0 ∧ (3 * a + 1) / (a - 1) < 0 ↔ - (1 : ℝ)/3 < a ∧ a < 1 :=
by
  sorry

end coordinate_plane_condition_l177_177297


namespace find_m_l177_177812

open Nat

def is_arithmetic (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i < n - 1, a (i + 2) - a (i + 1) = a (i + 1) - a i
def is_geometric (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i ≥ n, a (i + 1) * a n = a i * a (n + 1)
def sum_prod_condition (a : ℕ → ℤ) (m : ℕ) : Prop := a m + a (m + 1) + a (m + 2) = a m * a (m + 1) * a (m + 2)

theorem find_m (a : ℕ → ℤ)
  (h1 : a 3 = -1)
  (h2 : a 7 = 4)
  (h3 : is_arithmetic a 6)
  (h4 : is_geometric a 5) :
  ∃ m : ℕ, m = 1 ∨ m = 3 ∧ sum_prod_condition a m := sorry

end find_m_l177_177812


namespace problem_statement_l177_177471

-- Define the problem parameters with the constraints
def numberOfWaysToDistributeBalls (totalBalls : Nat) (initialDistribution : List Nat) : Nat :=
  -- Compute the number of remaining balls after the initial distribution
  let remainingBalls := totalBalls - initialDistribution.foldl (· + ·) 0
  -- Use the stars and bars formula to compute the number of ways to distribute remaining balls
  Nat.choose (remainingBalls + initialDistribution.length - 1) (initialDistribution.length - 1)

-- The boxes are to be numbered 1, 2, and 3, and each must contain at least its number of balls
def answer : Nat := numberOfWaysToDistributeBalls 9 [1, 2, 3]

-- Statement of the theorem
theorem problem_statement : answer = 10 := by
  sorry

end problem_statement_l177_177471


namespace choir_students_min_l177_177038

/-- 
  Prove that the minimum number of students in the choir, where the number 
  of students must be a multiple of 9, 10, and 11, is 990. 
-/
theorem choir_students_min (n : ℕ) :
  (∃ n, n > 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) ∧ (∀ m, m > 0 ∧ m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → n ≤ m) → n = 990 :=
by
  sorry

end choir_students_min_l177_177038


namespace proportion_is_equation_l177_177959

/-- A proportion containing unknowns is an equation -/
theorem proportion_is_equation (P : Prop) (contains_equality_sign: Prop)
  (indicates_equality : Prop)
  (contains_unknowns : Prop) : (contains_equality_sign ∧ indicates_equality ∧ contains_unknowns ↔ True) := by
  sorry

end proportion_is_equation_l177_177959


namespace distance_to_nearest_town_l177_177523

theorem distance_to_nearest_town (d : ℝ) :
  ¬ (d ≥ 6) → ¬ (d ≤ 5) → ¬ (d ≤ 4) → (d > 5 ∧ d < 6) :=
by
  intro h1 h2 h3
  sorry

end distance_to_nearest_town_l177_177523


namespace max_profit_l177_177658

noncomputable def profit_A (x : ℕ) : ℝ := -↑x^2 + 21 * ↑x
noncomputable def profit_B (x : ℕ) : ℝ := 2 * ↑x
noncomputable def total_profit (x : ℕ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit : 
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 120 := sorry

end max_profit_l177_177658


namespace problem_area_triangle_PNT_l177_177910

noncomputable def area_triangle_PNT (PQ QR x : ℝ) : ℝ :=
  let PS := Real.sqrt (PQ^2 + QR^2)
  let PN := PS / 2
  let area := (PN * Real.sqrt (61 - x^2)) / 4
  area

theorem problem_area_triangle_PNT :
  ∀ (PQ QR : ℝ) (x : ℝ), PQ = 10 → QR = 12 → 0 ≤ x ∧ x ≤ 10 → area_triangle_PNT PQ QR x = 
  (Real.sqrt (244) * Real.sqrt (61 - x^2)) / 4 :=
by
  intros PQ QR x hPQ hQR hx
  sorry

end problem_area_triangle_PNT_l177_177910


namespace fill_time_eight_faucets_l177_177987

theorem fill_time_eight_faucets (r : ℝ) (h1 : 4 * r * 8 = 150) :
  8 * r * (50 / (8 * r)) * 60 = 80 := by
  sorry

end fill_time_eight_faucets_l177_177987


namespace worker_days_total_l177_177672

theorem worker_days_total
  (W I : ℕ)
  (hw : 20 * W - 3 * I = 280)
  (hi : I = 40) :
  W + I = 60 :=
by
  sorry

end worker_days_total_l177_177672


namespace concur_PS_QR_CI_l177_177189

variables {α : Type*} [euclidean_space α]

open euclidean_space

-- Variables representing the points and elements in the problem
variables (A B C P Q R S I : point α)
variable (k : circle α)

-- Hypothesis and conditions
hypothesis hI : is_incenter I (triangle.mk A B C)
hypothesis hk : k ∈ circle_passing_through A ∧ k ∈ circle_passing_through B
hypothesis hAP : ∃ AP, is_line_through AP A ∧ is_line_through AP I ∧ k.intersects AP (some_point_in k A I)
hypothesis hBQ : ∃ BQ, is_line_through BQ B ∧ is_line_through BQ I ∧ k.intersects BQ (some_point_in k B I)
hypothesis hAR : ∃ AR, is_line_through AR A ∧ is_line_through AR C ∧ k.intersects AR (some_point_in k A C)
hypothesis hBS : ∃ BS, is_line_through BS B ∧ is_line_through BS C ∧ k.intersects BS (some_point_in k B C)
hypothesis hDistinct : distinct_points [A, B, P, Q, R, S]

-- Additional geometric constraints
hypothesis hR : lies_on_segment R A C
hypothesis hS : lies_on_segment S B C

-- Theorem to be proved
theorem concur_PS_QR_CI :
  ∃ X, concurrent (line_through P S) (line_through Q R) (line_through C I) :=
begin
  sorry
end

end concur_PS_QR_CI_l177_177189


namespace points_on_octagon_boundary_l177_177681

def is_on_octagon_boundary (x y : ℝ) : Prop :=
  |x| + |y| + |x - 1| + |y - 1| = 4

theorem points_on_octagon_boundary :
  ∀ (x y : ℝ), is_on_octagon_boundary x y ↔ ((0 ≤ x ∧ x ≤ 1 ∧ (y = 2 ∨ y = -1)) ∨
                                             (0 ≤ y ∧ y ≤ 1 ∧ (x = 2 ∨ x = -1)) ∨
                                             (x ≥ 1 ∧ y ≥ 1 ∧ x + y = 3) ∨
                                             (x ≤ 1 ∧ y ≤ 1 ∧ x + y = 1) ∨
                                             (x ≥ 1 ∧ y ≤ -1 ∧ x + y = 1) ∨
                                             (x ≤ -1 ∧ y ≥ 1 ∧ x + y = 1) ∨
                                             (x ≤ -1 ∧ y ≤ 1 ∧ x + y = -1) ∨
                                             (x ≤ 1 ∧ y ≤ -1 ∧ x + y = -1)) :=
by
  sorry

end points_on_octagon_boundary_l177_177681


namespace sequence_a4_value_l177_177628

theorem sequence_a4_value :
  ∀ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + 3) → a 4 = 29 :=
by sorry

end sequence_a4_value_l177_177628


namespace log24_eq_2b_minus_a_l177_177546

variable (a b : ℝ)

-- given conditions
axiom log6_eq : Real.log 6 = a
axiom log12_eq : Real.log 12 = b

-- proof goal statement
theorem log24_eq_2b_minus_a : Real.log 24 = 2 * b - a :=
by
  sorry

end log24_eq_2b_minus_a_l177_177546


namespace infinite_slips_have_repeated_numbers_l177_177846

theorem infinite_slips_have_repeated_numbers
  (slips : Set ℕ) (h_inf_slips : slips.Infinite)
  (h_sub_infinite_imp_repeats : ∀ s : Set ℕ, s.Infinite → ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ x = y) :
  ∃ n : ℕ, {x ∈ slips | x = n}.Infinite :=
by sorry

end infinite_slips_have_repeated_numbers_l177_177846


namespace check_basis_l177_177784

structure Vector2D :=
  (x : ℤ)
  (y : ℤ)

def are_collinear (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y - v2.x * v1.y = 0

def can_be_basis (v1 v2 : Vector2D) : Prop :=
  ¬ are_collinear v1 v2

theorem check_basis :
  can_be_basis ⟨-1, 2⟩ ⟨5, 7⟩ ∧
  ¬ can_be_basis ⟨0, 0⟩ ⟨1, -2⟩ ∧
  ¬ can_be_basis ⟨3, 5⟩ ⟨6, 10⟩ ∧
  ¬ can_be_basis ⟨2, -3⟩ ⟨(1 : ℤ)/2, -(3 : ℤ)/4⟩ :=
by
  sorry

end check_basis_l177_177784


namespace ratio_S15_S5_l177_177156

variable {α : Type*} [LinearOrderedField α]

namespace ArithmeticSequence

def sum_of_first_n_terms (a : α) (d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

theorem ratio_S15_S5
  {a d : α}
  {S5 S10 S15 : α}
  (h1 : S5 = sum_of_first_n_terms a d 5)
  (h2 : S10 = sum_of_first_n_terms a d 10)
  (h3 : S15 = sum_of_first_n_terms a d 15)
  (h_ratio : S5 / S10 = 2 / 3) :
  S15 / S5 = 3 / 2 := 
sorry

end ArithmeticSequence

end ratio_S15_S5_l177_177156


namespace three_digit_number_unchanged_upside_down_l177_177252

theorem three_digit_number_unchanged_upside_down (n : ℕ) :
  (n >= 100 ∧ n <= 999) ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d = 0 ∨ d = 8) ->
  n = 888 ∨ n = 808 :=
by
  sorry

end three_digit_number_unchanged_upside_down_l177_177252


namespace cos_120_degrees_l177_177070

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l177_177070


namespace centroid_for_segment_pass_l177_177844

open TopologicalSpace

variable {α : Type}

structure triangle (α : Type) :=
(A B C : α)

variables {t : triangle α}
variables (A B C M N : α)
variables [HasVectorSpace (point α)]
variables [HasDiv Real (point α)]

def segment1 := segment_len B M / segment_len M A
def segment2 := segment_len C N / segment_len N A

theorem centroid_for_segment_pass
  (hAB : LiesOnSegment A B M)
  (hAC : LiesOnSegment A C N)
  (hCondition : segment1 + segment2 = 1):
  PassesThroughCentroid t M N :=
sorry

end centroid_for_segment_pass_l177_177844


namespace mutter_lagaan_payment_l177_177892

-- Conditions as definitions
def total_lagaan_collected : ℝ := 344000
def mutter_percentage_of_total_taxable_land : ℝ := 0.23255813953488372 / 100

-- Proof statement
theorem mutter_lagaan_payment : (mutter_percentage_of_total_taxable_land * total_lagaan_collected) = 800 := by
  sorry

end mutter_lagaan_payment_l177_177892


namespace oranges_count_l177_177660

def oranges_per_box : ℝ := 10
def boxes_per_day : ℝ := 2650
def total_oranges (x y : ℝ) : ℝ := x * y

theorem oranges_count :
  total_oranges oranges_per_box boxes_per_day = 26500 := 
  by sorry

end oranges_count_l177_177660


namespace actual_price_of_food_before_tax_and_tip_l177_177040

theorem actual_price_of_food_before_tax_and_tip 
  (total_paid : ℝ)
  (tip_percentage : ℝ)
  (tax_percentage : ℝ)
  (pre_tax_food_price : ℝ)
  (h1 : total_paid = 132)
  (h2 : tip_percentage = 0.20)
  (h3 : tax_percentage = 0.10)
  (h4 : total_paid = (1 + tip_percentage) * (1 + tax_percentage) * pre_tax_food_price) :
  pre_tax_food_price = 100 :=
by sorry

end actual_price_of_food_before_tax_and_tip_l177_177040


namespace g_of_5_l177_177922

noncomputable def g : ℝ → ℝ := sorry

theorem g_of_5 :
  (∀ x y : ℝ, x * g y = y * g x) →
  g 20 = 30 →
  g 5 = 7.5 :=
by
  intros h1 h2
  sorry

end g_of_5_l177_177922


namespace probability_of_odd_product_is_4_over_15_l177_177331

noncomputable def probability_odd_product_of_two_distinct_integers : ℚ :=
  let total_numbers := 15
  let total_ways := nat.choose total_numbers 2
  let odd_numbers := 8
  let odd_ways := nat.choose odd_numbers 2
  (odd_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_odd_product_is_4_over_15 :
  probability_odd_product_of_two_distinct_integers = 4 / 15 :=
sorry

end probability_of_odd_product_is_4_over_15_l177_177331


namespace cos_120_degrees_l177_177068

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l177_177068


namespace total_tin_in_new_alloy_l177_177513

-- Define the weights of alloy A and alloy B
def weightAlloyA : Float := 135
def weightAlloyB : Float := 145

-- Define the ratio of lead to tin in alloy A
def ratioLeadToTinA : Float := 3 / 5

-- Define the ratio of tin to copper in alloy B
def ratioTinToCopperB : Float := 2 / 3

-- Define the total parts for alloy A and alloy B
def totalPartsA : Float := 3 + 5
def totalPartsB : Float := 2 + 3

-- Define the fraction of tin in alloy A and alloy B
def fractionTinA : Float := 5 / totalPartsA
def fractionTinB : Float := 2 / totalPartsB

-- Calculate the amount of tin in alloy A and alloy B
def tinInAlloyA : Float := fractionTinA * weightAlloyA
def tinInAlloyB : Float := fractionTinB * weightAlloyB

-- Calculate the total amount of tin in the new alloy
def totalTinInNewAlloy : Float := tinInAlloyA + tinInAlloyB

-- The theorem to be proven
theorem total_tin_in_new_alloy : totalTinInNewAlloy = 142.375 := by
  sorry

end total_tin_in_new_alloy_l177_177513


namespace circle_area_l177_177219

theorem circle_area :
  let circle := {p : ℝ × ℝ | (p.fst - 8) ^ 2 + p.snd ^ 2 = 64}
  let line := {p : ℝ × ℝ | p.snd = 10 - p.fst}
  ∃ area : ℝ, 
    (area = 8 * Real.pi) ∧ 
    ∀ p : ℝ × ℝ, p ∈ circle → p.snd ≥ 0 → p ∈ line → p.snd ≥ 10 - p.fst →
  sorry := sorry

end circle_area_l177_177219


namespace travel_times_l177_177045

variable (t v1 v2 : ℝ)

def conditions := 
  (v1 * 2 = v2 * t) ∧ 
  (v2 * 4.5 = v1 * t)

theorem travel_times (h : conditions t v1 v2) : 
  t = 3 ∧ 
  (t + 2 = 5) ∧ 
  (t + 4.5 = 7.5) := by
  sorry

end travel_times_l177_177045


namespace problem1_problem2_l177_177554

-- Definitions and conditions:
def p (x : ℝ) : Prop := x^2 - 4 * x - 5 ≤ 0
def q (x m : ℝ) : Prop := (x^2 - 2 * x + 1 - m^2 ≤ 0) ∧ (m > 0)

-- Question (1) statement: Prove that if p is a sufficient condition for q, then m ≥ 4
theorem problem1 (p_implies_q : ∀ x : ℝ, p x → q x m) : m ≥ 4 := sorry

-- Question (2) statement: Prove that if m = 5 and p ∨ q is true but p ∧ q is false,
-- then the range of x is [-4, -1) ∪ (5, 6]
theorem problem2 (m_eq : m = 5) (p_or_q : ∃ x : ℝ, p x ∨ q x m) (p_and_not_q : ¬ (∃ x : ℝ, p x ∧ q x m)) :
  ∃ x : ℝ, (x < -1 ∧ -4 ≤ x) ∨ (5 < x ∧ x ≤ 6) := sorry

end problem1_problem2_l177_177554


namespace find_x_l177_177723

theorem find_x {x : ℝ} (hx : x^2 - 5 * x = -4) : x = 1 ∨ x = 4 :=
sorry

end find_x_l177_177723


namespace reduced_price_per_dozen_l177_177944

variables {P R : ℝ}

theorem reduced_price_per_dozen
  (H1 : R = 0.6 * P)
  (H2 : 40 / P - 40 / R = 64) :
  R = 3 := 
sorry

end reduced_price_per_dozen_l177_177944


namespace VIP_ticket_price_l177_177460

variable (total_savings : ℕ) 
variable (num_VIP_tickets : ℕ)
variable (num_regular_tickets : ℕ)
variable (price_per_regular_ticket : ℕ)
variable (remaining_savings : ℕ)

theorem VIP_ticket_price 
  (h1 : total_savings = 500)
  (h2 : num_VIP_tickets = 2)
  (h3 : num_regular_tickets = 3)
  (h4 : price_per_regular_ticket = 50)
  (h5 : remaining_savings = 150) :
  (total_savings - remaining_savings) - (num_regular_tickets * price_per_regular_ticket) = num_VIP_tickets * 100 := 
by
  sorry

end VIP_ticket_price_l177_177460


namespace maximum_distance_is_correct_l177_177238

-- Define the right trapezoid with the given side lengths and angle conditions
structure RightTrapezoid (AB CD : ℕ) where
  B_angle : ℝ
  D_angle : ℝ
  h_AB : AB = 200
  h_CD : CD = 100
  h_B_angle : B_angle = 90
  h_D_angle : D_angle = 45

-- Define the guards' walking condition and distance calculation
def max_distance_between_guards (T : RightTrapezoid 200 100) : ℝ :=
  let P := 400 + 100 * Real.sqrt 2
  let d := (400 + 100 * Real.sqrt 2) / 2
  222.1  -- Hard-coded according to the problem's correct answer for maximum distance

theorem maximum_distance_is_correct :
  ∀ (T : RightTrapezoid 200 100), max_distance_between_guards T = 222.1 := by
  sorry

end maximum_distance_is_correct_l177_177238


namespace angle_B_in_triangle_l177_177868

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l177_177868


namespace find_fourth_term_in_sequence_l177_177275

theorem find_fourth_term_in_sequence (x: ℤ) (h1: 86 - 8 = 78) (h2: 2 - 86 = -84) (h3: x - 2 = -90) (h4: -12 - x = 76):
  x = -88 :=
sorry

end find_fourth_term_in_sequence_l177_177275


namespace horse_food_per_day_l177_177673

theorem horse_food_per_day
  (ratio_sheep_horses : 6 / 7)
  (total_horse_food : 12880)
  (sheep_count : 48) :
  let horses_count := 7 * sheep_count / 6 in
  total_horse_food / horses_count = 230 := by
sorry

end horse_food_per_day_l177_177673


namespace find_ratio_l177_177057

def celsius_to_fahrenheit_ratio (ratio : ℝ) (c f : ℝ) : Prop :=
  f = ratio * c + 32

theorem find_ratio (ratio : ℝ) :
  (∀ c f, celsius_to_fahrenheit_ratio ratio c f ∧ ((f = 58) → (c = 14.444444444444445)) → f = 1.8 * c + 32) ∧ 
  (f - 32 = ratio * (c - 0)) ∧
  (c = 14.444444444444445 → f = 32 + 26) ∧
  (f = 58 → c = 14.444444444444445) ∧ 
  (ratio = 1.8)
  → ratio = 1.8 := 
sorry 


end find_ratio_l177_177057


namespace value_range_of_f_l177_177756

def f (x : ℝ) := 2 * x ^ 2 + 4 * x + 1

theorem value_range_of_f :
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 4 → (∃ y ∈ Set.Icc (-1 : ℝ) 49, f x = y) :=
by sorry

end value_range_of_f_l177_177756


namespace combine_like_terms_l177_177941

theorem combine_like_terms : ∀ (x y : ℝ), -2 * x * y^2 + 2 * x * y^2 = 0 :=
by
  intros
  sorry

end combine_like_terms_l177_177941


namespace triangle_area_of_integral_sides_with_perimeter_8_l177_177668

theorem triangle_area_of_integral_sides_with_perimeter_8 :
  ∃ (a b c : ℕ), a + b + c = 8 ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ 
  ∃ (area : ℝ), area = 2 * Real.sqrt 2 := by
  sorry

end triangle_area_of_integral_sides_with_perimeter_8_l177_177668


namespace quarterback_sacked_times_l177_177666

theorem quarterback_sacked_times
    (total_throws : ℕ)
    (no_pass_percentage : ℚ)
    (half_sacked : ℚ)
    (no_passes : ℕ)
    (sacks : ℕ) :
    total_throws = 80 →
    no_pass_percentage = 0.30 →
    half_sacked = 0.50 →
    no_passes = total_throws * no_pass_percentage →
    sacks = no_passes / 2 →
    sacks = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end quarterback_sacked_times_l177_177666


namespace tan_alpha_plus_pi_over_4_rational_expression_of_trig_l177_177144

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  Real.tan (α + Real.pi / 4) = -1 / 7 := 
by 
  sorry

theorem rational_expression_of_trig (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := 
by 
  sorry

end tan_alpha_plus_pi_over_4_rational_expression_of_trig_l177_177144


namespace bike_ride_time_good_l177_177256

theorem bike_ride_time_good (x : ℚ) :
  (20 * x + 12 * (8 - x) = 122) → x = 13 / 4 :=
by
  intro h
  sorry

end bike_ride_time_good_l177_177256


namespace similar_terms_solution_l177_177167

theorem similar_terms_solution
  (a b : ℝ)
  (m n x y : ℤ)
  (h1 : m - 1 = n - 2 * m)
  (h2 : m + n = 3 * m + n - 4)
  (h3 : m * x + (n - 2) * y = 24)
  (h4 : 2 * m * x + n * y = 46) :
  x = 9 ∧ y = 2 := by
  sorry

end similar_terms_solution_l177_177167


namespace smallest_n_for_coloring_l177_177543

theorem smallest_n_for_coloring (n : ℕ) : n = 4 :=
sorry

end smallest_n_for_coloring_l177_177543


namespace weight_of_5_moles_BaO_molar_concentration_BaO_l177_177676

-- Definitions based on conditions
def atomic_mass_Ba : ℝ := 137.33
def atomic_mass_O : ℝ := 16.00
def molar_mass_BaO : ℝ := atomic_mass_Ba + atomic_mass_O
def moles_BaO : ℝ := 5
def volume_solution : ℝ := 3

-- Theorem statements
theorem weight_of_5_moles_BaO : moles_BaO * molar_mass_BaO = 766.65 := by
  sorry

theorem molar_concentration_BaO : moles_BaO / volume_solution = 1.67 := by
  sorry

end weight_of_5_moles_BaO_molar_concentration_BaO_l177_177676


namespace tyson_one_point_count_l177_177355

def tyson_three_points := 3 * 15
def tyson_two_points := 2 * 12
def total_points := 75
def points_from_three_and_two := tyson_three_points + tyson_two_points

theorem tyson_one_point_count :
  ∃ n : ℕ, n % 2 = 0 ∧ (n = total_points - points_from_three_and_two) :=
sorry

end tyson_one_point_count_l177_177355


namespace company_production_l177_177235

theorem company_production (bottles_per_case number_of_cases total_bottles : ℕ)
  (h1 : bottles_per_case = 12)
  (h2 : number_of_cases = 10000)
  (h3 : total_bottles = number_of_cases * bottles_per_case) : 
  total_bottles = 120000 :=
by {
  -- Proof is omitted, add actual proof here
  sorry
}

end company_production_l177_177235


namespace ratio_sub_div_eq_l177_177142

theorem ratio_sub_div_eq 
  (a b : ℚ) 
  (h : a / b = 5 / 2) : 
  (a - b) / a = 3 / 5 := 
sorry

end ratio_sub_div_eq_l177_177142


namespace find_angle_B_l177_177879

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l177_177879


namespace quadratic_roots_l177_177468

theorem quadratic_roots (x : ℝ) : 
  (2 * x^2 - 4 * x - 5 = 0) ↔ 
  (x = (2 + Real.sqrt 14) / 2 ∨ x = (2 - Real.sqrt 14) / 2) :=
by
  sorry

end quadratic_roots_l177_177468


namespace sign_pyramid_combinations_l177_177792

-- Define a ±1 data type for easy reference
inductive Sign : Type
| pos : Sign
| neg : Sign

open Sign

/-- A function to determine the sign of a top cell in the pyramid structure given bottom row -/
noncomputable def pyramidTopSign (a b c d e : Sign) : Sign :=
  let ab := if a = b then pos else neg
  let bc := if b = c then pos else neg
  let cd := if c = d then pos else neg
  let de := if d = e then pos else neg
  let ab_bc := if ab = bc then pos else neg
  let bc_cd := if bc = cd then pos else neg
  let cd_de := if cd = de then pos else neg
  let ab_bc_bc_cd := if ab_bc = bc_cd then pos else neg
  let bc_cd_cd_de := if bc_cd = cd_de then pos else neg
  if ab_bc_bc_cd = bc_cd_cd_de then pos else neg

/-- Prove that there are exactly 16 ways to fill the bottom cells so all intermediate top cells are "+" -/
theorem sign_pyramid_combinations : 
  {l : List Sign // l.length = 5 ∧ pyramidTopSign (l.nth 0).get_or_else pos
                                         (l.nth 1).get_or_else pos 
                                         (l.nth 2).get_or_else pos 
                                         (l.nth 3).get_or_else pos 
                                         (l.nth 4).get_or_else pos = pos} 
  = 16 := sorry

end sign_pyramid_combinations_l177_177792


namespace division_problem_l177_177362

theorem division_problem : (4 * 5) / 10 = 2 :=
by sorry

end division_problem_l177_177362


namespace paving_stone_length_l177_177928

theorem paving_stone_length
  (length_courtyard : ℝ)
  (width_courtyard : ℝ)
  (num_paving_stones : ℝ)
  (width_paving_stone : ℝ)
  (total_area : ℝ := length_courtyard * width_courtyard)
  (area_per_paving_stone : ℝ := (total_area / num_paving_stones))
  (length_paving_stone : ℝ := (area_per_paving_stone / width_paving_stone)) :
  length_courtyard = 20 ∧
  width_courtyard = 16.5 ∧
  num_paving_stones = 66 ∧
  width_paving_stone = 2 →
  length_paving_stone = 2.5 :=
by {
   sorry
}

end paving_stone_length_l177_177928


namespace divisibility_by_six_l177_177140

theorem divisibility_by_six (a x: ℤ) : ∃ t: ℤ, x = 3 * t ∨ x = 3 * t - a^2 → 6 ∣ a * (x^3 + a^2 * x^2 + a^2 - 1) :=
by
  sorry

end divisibility_by_six_l177_177140


namespace objects_meeting_time_l177_177955

theorem objects_meeting_time 
  (initial_velocity : ℝ) (g : ℝ) (t_delay : ℕ) (t_meet : ℝ) 
  (hv : initial_velocity = 120) 
  (hg : g = 9.8) 
  (ht : t_delay = 5)
  : t_meet = 14.74 :=
sorry

end objects_meeting_time_l177_177955


namespace largest_five_digit_palindromic_number_l177_177148

theorem largest_five_digit_palindromic_number (a b c d e : ℕ)
  (h1 : ∃ a b c, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
                 ∃ d e, 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
                 (10001 * a + 1010 * b + 100 * c = 45 * (1001 * d + 110 * e))) :
  10001 * 5 + 1010 * 9 + 100 * 8 = 59895 :=
by
  sorry

end largest_five_digit_palindromic_number_l177_177148


namespace values_of_x_l177_177639

theorem values_of_x (x : ℝ) : (-2 < x ∧ x < 2) ↔ (x^2 < |x| + 2) := by
  sorry

end values_of_x_l177_177639


namespace measure_ADC_l177_177620

-- Definitions
def angle_measures (x y ADC : ℝ) : Prop :=
  2 * x + 60 + 2 * y = 180 ∧ x + y = 60 ∧ x + y + ADC = 180

-- Goal
theorem measure_ADC (x y ADC : ℝ) (h : angle_measures x y ADC) : ADC = 120 :=
by {
  -- Solution could go here, skipped for brevity
  sorry
}

end measure_ADC_l177_177620


namespace early_finish_hours_l177_177532

theorem early_finish_hours 
  (h : Nat) 
  (total_customers : Nat) 
  (num_workers : Nat := 3)
  (service_rate : Nat := 7) 
  (full_hours : Nat := 8)
  (total_customers_served : total_customers = 154) 
  (two_workers_hours : Nat := 2 * full_hours * service_rate) 
  (early_worker_customers : Nat := h * service_rate)
  (total_service : total_customers = two_workers_hours + early_worker_customers) : 
  h = 6 :=
by
  sorry

end early_finish_hours_l177_177532


namespace ashley_loan_least_months_l177_177254

theorem ashley_loan_least_months (t : ℕ) (principal : ℝ) (interest_rate : ℝ) (triple_principal : ℝ) : 
  principal = 1500 ∧ interest_rate = 0.06 ∧ triple_principal = 3 * principal → 
  1.06^t > triple_principal → t = 20 :=
by
  intro h h2
  sorry

end ashley_loan_least_months_l177_177254


namespace tree_initial_height_l177_177767

theorem tree_initial_height (H : ℝ) (C : ℝ) (P : H + 6 = (H + 4) + 1/4 * (H + 4) ∧ C = 1) : H = 4 :=
by
  let H := 4
  sorry

end tree_initial_height_l177_177767


namespace divides_expression_l177_177445

theorem divides_expression (n : ℕ) (h1 : n ≥ 3) 
  (h2 : Prime (4 * n + 1)) : (4 * n + 1) ∣ (n^(2 * n) - 1) :=
by
  sorry

end divides_expression_l177_177445


namespace speed_of_current_l177_177377

theorem speed_of_current (m c : ℝ) (h1 : m + c = 20) (h2 : m - c = 18) : c = 1 :=
by
  sorry

end speed_of_current_l177_177377


namespace wine_consumption_correct_l177_177577

-- Definitions based on conditions
def drank_after_first_pound : ℚ := 1
def drank_after_second_pound : ℚ := 1
def drank_after_third_pound : ℚ := 1 / 2
def drank_after_fourth_pound : ℚ := 1 / 4
def drank_after_fifth_pound : ℚ := 1 / 8
def drank_after_sixth_pound : ℚ := 1 / 16

-- Total wine consumption
def total_wine_consumption : ℚ :=
  drank_after_first_pound + drank_after_second_pound +
  drank_after_third_pound + drank_after_fourth_pound +
  drank_after_fifth_pound + drank_after_sixth_pound

-- Theorem statement
theorem wine_consumption_correct :
  total_wine_consumption = 47 / 16 :=
by
  sorry

end wine_consumption_correct_l177_177577


namespace sector_central_angle_l177_177704

theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : 1/2 * l * r = 1) : l / r = 2 := 
by
  sorry

end sector_central_angle_l177_177704


namespace common_divisors_9240_13860_l177_177831

def num_divisors (n : ℕ) : ℕ :=
  -- function to calculate the number of divisors (implementation is not provided here)
  sorry

theorem common_divisors_9240_13860 :
  let d := Nat.gcd 9240 13860
  d = 924 → num_divisors d = 24 := by
  intros d gcd_eq
  rw [gcd_eq]
  sorry

end common_divisors_9240_13860_l177_177831


namespace horse_grazing_area_l177_177956

noncomputable def grazing_area (radius : ℝ) : ℝ :=
  (1 / 4) * Real.pi * radius^2

theorem horse_grazing_area :
  let length := 46
  let width := 20
  let rope_length := 17
  rope_length <= length ∧ rope_length <= width →
  grazing_area rope_length = 72.25 * Real.pi :=
by
  sorry

end horse_grazing_area_l177_177956


namespace angles_in_interval_l177_177264

open Real

theorem angles_in_interval
    (θ : ℝ)
    (hθ : 0 ≤ θ ∧ θ ≤ 2 * π)
    (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 * sin θ - x * (2 - x) + (2 - x)^2 * cos θ > 0) :
  π / 12 < θ ∧ θ < 5 * π / 12 :=
by
  sorry

end angles_in_interval_l177_177264


namespace water_evaporation_l177_177484

theorem water_evaporation (m : ℝ) 
  (evaporation_day1 : m' = m * (0.1)) 
  (evaporation_day2 : m'' = (m * 0.9) * 0.1) 
  (total_evaporation : total = m' + m'')
  (water_added : 15 = total) 
  : m = 1500 / 19 := by
  sorry

end water_evaporation_l177_177484


namespace students_with_dog_and_cat_only_l177_177176

theorem students_with_dog_and_cat_only
  (U : Finset (ℕ)) -- Universe of students
  (D C B : Finset (ℕ)) -- Sets of students with dogs, cats, and birds respectively
  (hU : U.card = 50)
  (hD : D.card = 30)
  (hC : C.card = 35)
  (hB : B.card = 10)
  (hIntersection : (D ∩ C ∩ B).card = 5) :
  ((D ∩ C) \ B).card = 25 := 
sorry

end students_with_dog_and_cat_only_l177_177176


namespace value_of_a_plus_b_l177_177897

theorem value_of_a_plus_b 
  (a b : ℝ) 
  (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = a * x + b)
  (h₂ : ∀ x, g x = 3 * x - 6)
  (h₃ : ∀ x, g (f x) = 4 * x + 5) : 
  a + b = 5 :=
sorry

end value_of_a_plus_b_l177_177897


namespace solution_set_f_neg_x_l177_177557

noncomputable def f (a b x : Real) : Real := (a * x - 1) * (x - b)

theorem solution_set_f_neg_x (a b : Real) (h : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) : 
  ∀ x, f a b (-x) < 0 ↔ x < -3 ∨ x > 1 := 
by
  sorry

end solution_set_f_neg_x_l177_177557


namespace greatest_product_sum_2006_l177_177008

theorem greatest_product_sum_2006 :
  (∃ x y : ℤ, x + y = 2006 ∧ ∀ a b : ℤ, a + b = 2006 → a * b ≤ x * y) → 
  ∃ x y : ℤ, x + y = 2006 ∧ x * y = 1006009 :=
by sorry

end greatest_product_sum_2006_l177_177008


namespace fraction_product_l177_177504

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l177_177504


namespace determine_b_for_constant_remainder_l177_177131

theorem determine_b_for_constant_remainder (b : ℚ) :
  ∃ r : ℚ, ∀ x : ℚ,  (12 * x^3 - 9 * x^2 + b * x + 8) / (3 * x^2 - 4 * x + 2) = r ↔ b = -4 / 3 :=
by sorry

end determine_b_for_constant_remainder_l177_177131


namespace correct_expression_l177_177014

theorem correct_expression (a b : ℝ) : (a^2 * b)^3 = (a^6 * b^3) := 
by
sorry

end correct_expression_l177_177014


namespace no_solution_for_a_l177_177405

theorem no_solution_for_a {a : ℝ} :
  (a ∈ Set.Iic (-32) ∪ Set.Ici 0) →
  ¬ ∃ x : ℝ,  9 * |x - 4 * a| + |x - a^2| + 8 * x - 4 * a = 0 :=
by
  intro h
  sorry

end no_solution_for_a_l177_177405


namespace solve_for_s_l177_177203

-- Definition of the condition
def condition (s : ℝ) : Prop := (s - 60) / 3 = (6 - 3 * s) / 4

-- Theorem stating that if the condition holds, then s = 19.85
theorem solve_for_s (s : ℝ) : condition s → s = 19.85 := 
by {
  sorry -- Proof is skipped as per requirements
}

end solve_for_s_l177_177203


namespace prob_and_relation_proof_l177_177030

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l177_177030


namespace range_of_a_l177_177926

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, ¬ (x^2 - a * x + 1 ≤ 0)) ↔ -2 < a ∧ a < 2 := 
sorry

end range_of_a_l177_177926


namespace window_treatments_total_cost_l177_177597

def sheers_cost_per_pair := 40
def drapes_cost_per_pair := 60
def number_of_windows := 3

theorem window_treatments_total_cost :
  (number_of_windows * sheers_cost_per_pair) + (number_of_windows * drapes_cost_per_pair) = 300 :=
by 
  -- calculations omitted
  sorry

end window_treatments_total_cost_l177_177597


namespace job_completion_l177_177035

theorem job_completion (x y z : ℝ) 
  (h1 : 1/x + 1/y = 1/2) 
  (h2 : 1/y + 1/z = 1/4) 
  (h3 : 1/z + 1/x = 1/2.4) 
  (h4 : 1/x + 1/y + 1/z = 7/12) : 
  x = 3 := 
sorry

end job_completion_l177_177035


namespace project_completion_time_saving_l177_177244

/-- A theorem stating that if a project with initial and additional workforce configuration,
the project will be completed 10 days ahead of schedule. -/
theorem project_completion_time_saving
  (total_days : ℕ := 100)
  (initial_people : ℕ := 10)
  (initial_days : ℕ := 30)
  (initial_fraction : ℚ := 1 / 5)
  (additional_people : ℕ := 10)
  : (total_days - ((initial_days + (1 / (initial_people + additional_people * initial_fraction)) * (total_days * initial_fraction) / initial_fraction)) = 10) :=
sorry

end project_completion_time_saving_l177_177244


namespace complement_of_A_in_U_l177_177033

-- Define the universal set U
def U : Set ℕ := {2, 3, 4}

-- Define set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def C_U_A : Set ℕ := U \ A

-- Prove the complement of A in U is {4}
theorem complement_of_A_in_U : C_U_A = {4} := 
  by 
  sorry

end complement_of_A_in_U_l177_177033


namespace primes_with_no_sum_of_two_cubes_l177_177682

theorem primes_with_no_sum_of_two_cubes (p : ℕ) [Fact (Nat.Prime p)] :
  (∃ n : ℤ, ∀ x y : ℤ, x^3 + y^3 ≠ n % p) ↔ p = 7 :=
sorry

end primes_with_no_sum_of_two_cubes_l177_177682


namespace fraction_computation_l177_177496

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l177_177496


namespace pants_cost_l177_177578

theorem pants_cost (P : ℝ) : 
(80 + 3 * P + 300) * 0.90 = 558 → P = 80 :=
by
  sorry

end pants_cost_l177_177578


namespace original_price_hat_l177_177218

theorem original_price_hat 
  (x : ℝ)
  (discounted_price := x / 5)
  (final_price := discounted_price * 1.2)
  (h : final_price = 8) :
  x = 100 / 3 :=
by
  sorry

end original_price_hat_l177_177218


namespace _l177_177400

noncomputable def t_value_theorem (a b x d t y : ℕ) (h1 : a + b = x) (h2 : x + d = t) (h3 : t + a = y) (h4 : b + d + y = 16) : t = 8 :=
by sorry

end _l177_177400


namespace panteleimon_twos_l177_177434

-- Define the variables
variables (P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2 : ℕ)

-- Define the conditions
def conditions :=
  P_5 + P_4 + P_3 + P_2 = 20 ∧
  G_5 + G_4 + G_3 + G_2 = 20 ∧
  P_5 = G_4 ∧
  P_4 = G_3 ∧
  P_3 = G_2 ∧
  P_2 = G_5 ∧
  (5 * P_5 + 4 * P_4 + 3 * P_3 + 2 * P_2 = 5 * G_5 + 4 * G_4 + 3 * G_3 + 2 * G_2)

-- The proof goal
theorem panteleimon_twos (h : conditions P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2) : P_2 = 5 :=
sorry

end panteleimon_twos_l177_177434


namespace cos_120_eq_neg_half_l177_177080

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l177_177080


namespace ratio_A_to_B_investment_l177_177781

variable (A B C : Type) [Field A] [Field B] [Field C]
variable (investA investB investC profit total_profit : A) 

-- Conditions
axiom A_invests_some_times_as_B : ∃ n : A, investA = n * investB
axiom B_invests_two_thirds_of_C : investB = (2/3) * investC
axiom total_profit_statement : total_profit = 3300
axiom B_share_statement : profit = 600

-- Theorem: Ratio of A's investment to B's investment is 3:1
theorem ratio_A_to_B_investment : ∃ n : A, investA = 3 * investB :=
sorry

end ratio_A_to_B_investment_l177_177781


namespace last_four_digits_5_pow_2011_l177_177197

theorem last_four_digits_5_pow_2011 : 
  (5^2011 % 10000) = 8125 :=
by
  -- Definitions based on conditions in the problem
  have h5 : 5^5 % 10000 = 3125 := sorry
  have h6 : 5^6 % 10000 = 5625 := sorry
  have h7 : 5^7 % 10000 = 8125 := sorry
  
  -- Prove using periodicity and modular arithmetic
  sorry

end last_four_digits_5_pow_2011_l177_177197


namespace inequality_proof_l177_177996
open Nat

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (1 / b) = 1) (h4 : n > 0) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
by
  sorry

end inequality_proof_l177_177996


namespace cos_120_degrees_eq_l177_177110

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l177_177110


namespace balcony_more_than_orchestra_l177_177642

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 370) 
  (h2 : 12 * x + 8 * y = 3320) : y - x = 190 :=
sorry

end balcony_more_than_orchestra_l177_177642


namespace average_surfers_correct_l177_177913

-- Define the number of surfers for each day
def surfers_first_day : ℕ := 1500
def surfers_second_day : ℕ := surfers_first_day + 600
def surfers_third_day : ℕ := (2 / 5 : ℝ) * surfers_first_day

-- Average number of surfers
def average_surfers : ℝ := (surfers_first_day + surfers_second_day + surfers_third_day) / 3

theorem average_surfers_correct :
  average_surfers = 1400 := 
  by 
    sorry

end average_surfers_correct_l177_177913


namespace smallest_gcd_12a_20b_l177_177834

theorem smallest_gcd_12a_20b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 := sorry

end smallest_gcd_12a_20b_l177_177834


namespace cos_120_eq_neg_half_l177_177089

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l177_177089


namespace Megan_full_folders_l177_177734

def initial_files : ℕ := 256
def deleted_files : ℕ := 67
def files_per_folder : ℕ := 12

def remaining_files : ℕ := initial_files - deleted_files
def number_of_folders : ℕ := remaining_files / files_per_folder

theorem Megan_full_folders : number_of_folders = 15 := by
  sorry

end Megan_full_folders_l177_177734


namespace particles_tend_to_unit_circle_as_time_goes_to_infinity_l177_177138

variables {x y t : ℝ}

-- Condition: Velocity field as a function
def velocity_field (x y : ℝ) : ℝ × ℝ :=
  (y + 2*x - 2*x^3 - 2*x*y^2, -x)

-- Formal statement to prove: Particles tend towards the unit circle as t -> ∞.
theorem particles_tend_to_unit_circle_as_time_goes_to_infinity
  (f : ℝ × ℝ → ℝ × ℝ)
  (hf : ∀ (x y : ℝ), f (x, y) = velocity_field x y) :
  (∃ (R : ℝ), ∀ (x y t : ℝ), (t > 0) → ‖(x, y) - (0, 0)‖ < R → r = 1) :=
sorry

end particles_tend_to_unit_circle_as_time_goes_to_infinity_l177_177138


namespace find_angle_B_l177_177877

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l177_177877


namespace Ahmed_total_distance_traveled_l177_177522

/--
Ahmed stops one-quarter of the way to the store.
He continues for 12 km to reach the store.
Prove that the total distance Ahmed travels is 16 km.
-/
theorem Ahmed_total_distance_traveled
  (D : ℝ) (h1 : D > 0)  -- D is the total distance to the store, assumed to be positive
  (h_stop : D / 4 + 12 = D) : D = 16 := 
sorry

end Ahmed_total_distance_traveled_l177_177522


namespace find_m_l177_177365

theorem find_m (m : ℝ) :
  (m - 2013 = 0) → (m = 2013) ∧ (m - 1 ≠ 0) :=
by {
  sorry
}

end find_m_l177_177365


namespace solve_equation_parabola_equation_l177_177651

-- Part 1: Equation Solutions
theorem solve_equation {x : ℝ} :
  (x - 9) ^ 2 = 2 * (x - 9) ↔ x = 9 ∨ x = 11 := by
  sorry

-- Part 2: Expression of Parabola
theorem parabola_equation (a h k : ℝ) (vertex : (ℝ × ℝ)) (point: (ℝ × ℝ)) :
  vertex = (-3, 2) → point = (-1, -2) →
  a * (point.1 - h) ^ 2 + k = point.2 →
  (a = -1) → (h = -3) → (k = 2) →
  - x ^ 2 - 6 * x - 7 = a * (x + 3) ^ 2 + 2 := by
  sorry

end solve_equation_parabola_equation_l177_177651


namespace percentage_increase_in_rent_l177_177661

theorem percentage_increase_in_rent
  (avg_rent_per_person_before : ℝ)
  (num_friends : ℕ)
  (friend_original_rent : ℝ)
  (avg_rent_per_person_after : ℝ)
  (total_rent_before : ℝ := num_friends * avg_rent_per_person_before)
  (total_rent_after : ℝ := num_friends * avg_rent_per_person_after)
  (rent_increase : ℝ := total_rent_after - total_rent_before)
  (percentage_increase : ℝ := (rent_increase / friend_original_rent) * 100)
  (h1 : avg_rent_per_person_before = 800)
  (h2 : num_friends = 4)
  (h3 : friend_original_rent = 1400)
  (h4 : avg_rent_per_person_after = 870) :
  percentage_increase = 20 :=
by
  sorry

end percentage_increase_in_rent_l177_177661


namespace calculate_power_expression_l177_177395

theorem calculate_power_expression : 4 ^ 2009 * (-0.25) ^ 2008 - 1 = 3 := 
by
  -- steps and intermediate calculations go here
  sorry

end calculate_power_expression_l177_177395


namespace remainder_polynomial_l177_177280

theorem remainder_polynomial (n : ℕ) (hn : n ≥ 2) : 
  ∃ Q R : Polynomial ℤ, (R.degree < 2) ∧ (X^n = Q * (X^2 - 4 * X + 3) + R) ∧ 
                       (R = (Polynomial.C ((3^n - 1) / 2) * X + Polynomial.C ((3 - 3^n) / 2))) :=
by
  sorry

end remainder_polynomial_l177_177280


namespace probability_and_relationship_l177_177028

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l177_177028


namespace Jose_age_correct_l177_177307

variable (Jose Zack Inez : ℕ)

-- Define the conditions
axiom Inez_age : Inez = 15
axiom Zack_age : Zack = Inez + 3
axiom Jose_age : Jose = Zack - 4

-- The proof statement
theorem Jose_age_correct : Jose = 14 :=
by
  -- Proof will be filled in later
  sorry

end Jose_age_correct_l177_177307


namespace probability_and_relationship_l177_177026

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l177_177026


namespace aunt_may_milk_left_l177_177786

def morningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def eveningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def spoiledMilk (milkProduction : ℝ) (spoilageRate : ℝ) : ℝ :=
  milkProduction * spoilageRate

def freshMilk (totalMilk spoiledMilk : ℝ) : ℝ :=
  totalMilk - spoiledMilk

def soldMilk (freshMilk : ℝ) (saleRate : ℝ) : ℝ :=
  freshMilk * saleRate

def milkLeft (freshMilk soldMilk : ℝ) : ℝ :=
  freshMilk - soldMilk

noncomputable def totalMilkLeft (previousLeftover : ℝ) (morningLeft eveningLeft : ℝ) : ℝ :=
  previousLeftover + morningLeft + eveningLeft

theorem aunt_may_milk_left :
  let numCows := 5
  let numGoats := 4
  let numSheep := 10
  let cowMilkMorning := 13
  let goatMilkMorning := 0.5
  let sheepMilkMorning := 0.25
  let cowMilkEvening := 14
  let goatMilkEvening := 0.6
  let sheepMilkEvening := 0.2
  let morningSpoilageRate := 0.10
  let eveningSpoilageRate := 0.05
  let iceCreamSaleRate := 0.70
  let cheeseShopSaleRate := 0.80
  let previousLeftover := 15
  let morningMilk := morningMilkProduction numCows numGoats numSheep cowMilkMorning goatMilkMorning sheepMilkMorning
  let eveningMilk := eveningMilkProduction numCows numGoats numSheep cowMilkEvening goatMilkEvening sheepMilkEvening
  let morningSpoiled := spoiledMilk morningMilk morningSpoilageRate
  let eveningSpoiled := spoiledMilk eveningMilk eveningSpoilageRate
  let freshMorningMilk := freshMilk morningMilk morningSpoiled
  let freshEveningMilk := freshMilk eveningMilk eveningSpoiled
  let morningSold := soldMilk freshMorningMilk iceCreamSaleRate
  let eveningSold := soldMilk freshEveningMilk cheeseShopSaleRate
  let morningLeft := milkLeft freshMorningMilk morningSold
  let eveningLeft := milkLeft freshEveningMilk eveningSold
  totalMilkLeft previousLeftover morningLeft eveningLeft = 47.901 :=
by
  sorry

end aunt_may_milk_left_l177_177786


namespace tonya_payment_l177_177004

def original_balance : ℝ := 150.00
def new_balance : ℝ := 120.00

noncomputable def payment_amount : ℝ := original_balance - new_balance

theorem tonya_payment :
  payment_amount = 30.00 :=
by
  sorry

end tonya_payment_l177_177004


namespace original_pencil_count_l177_177062

-- Defining relevant constants and assumptions based on the problem conditions
def pencilsRemoved : ℕ := 4
def pencilsLeft : ℕ := 83

-- Theorem to prove the original number of pencils is 87
theorem original_pencil_count : pencilsLeft + pencilsRemoved = 87 := by
  sorry

end original_pencil_count_l177_177062


namespace find_range_of_a_l177_177842

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^3) / 3 - (a / 2) * x^2 + x + 1

def is_monotonically_decreasing_in (a : ℝ) (x : ℝ) : Prop := 
  ∀ s t : ℝ, (s ∈ Set.Ioo (3 / 2) 4) ∧ (t ∈ Set.Ioo (3 / 2) 4) ∧ s < t → 
  f a t ≤ f a s

theorem find_range_of_a :
  ∀ a : ℝ, is_monotonically_decreasing_in a x → 
  a ∈ Set.Ici (17/4)
:= sorry

end find_range_of_a_l177_177842


namespace sum_of_y_values_l177_177186

def g (x : ℚ) : ℚ := 2 * x^2 - x + 3

theorem sum_of_y_values (y1 y2 : ℚ) (hy : g (4 * y1) = 10 ∧ g (4 * y2) = 10) :
  y1 + y2 = 1 / 16 :=
sorry

end sum_of_y_values_l177_177186


namespace ratio_of_X_to_Y_l177_177850

theorem ratio_of_X_to_Y (total_respondents : ℕ) (preferred_X : ℕ)
    (h_total : total_respondents = 250)
    (h_X : preferred_X = 200) :
    preferred_X / (total_respondents - preferred_X) = 4 := by
  sorry

end ratio_of_X_to_Y_l177_177850


namespace min_value_inequality_l177_177451

open Real

theorem min_value_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 9) :
  ( (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ) ≥ 9 :=
sorry

end min_value_inequality_l177_177451


namespace problem_correct_l177_177371


def problem : ℕ := 101 * 101 - 99 * 99

theorem problem_correct : problem = 400 := by
  sorry

end problem_correct_l177_177371


namespace solution_set_abs_ineq_l177_177631

theorem solution_set_abs_ineq (x : ℝ) : abs (2 - x) ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end solution_set_abs_ineq_l177_177631


namespace yarn_cut_parts_l177_177652

-- Define the given conditions
def total_length : ℕ := 10
def crocheted_parts : ℕ := 3
def crocheted_length : ℕ := 6

-- The main problem statement
theorem yarn_cut_parts (total_length crocheted_parts crocheted_length : ℕ) (h1 : total_length = 10) (h2 : crocheted_parts = 3) (h3 : crocheted_length = 6) :
  (total_length / (crocheted_length / crocheted_parts)) = 5 :=
by
  sorry

end yarn_cut_parts_l177_177652


namespace symmetric_line_equation_l177_177476

theorem symmetric_line_equation (x y : ℝ) : 
  3 * x - 4 * y + 5 = 0 → (3 * x + 4 * y - 5 = 0) :=
by
sorry

end symmetric_line_equation_l177_177476


namespace calculate_fraction_l177_177970

variable (a b c d e f g h i j k l : ℚ)

def mixed_num_to_improper (a b : ℚ) := a * b + c

theorem calculate_fraction :
  (a = 2 ∧ b = 1/4 ∧ c = 0.25 ∧ d = 2 ∧ e = 3/4 ∧ f = 1/2 ∧ g = 2 ∧ h = 1/5 ∧ i = 2/5) →
  ((a * (4/1) + b + c) / (d * (4/1) + e - f) + ((2 * 0.5) / (g * (5/1) + h - i)) = 5/3) :=
by
  sorry

end calculate_fraction_l177_177970


namespace loci_of_square_view_l177_177266

-- Definitions based on the conditions in a)
def square (A B C D : Point) : Prop := -- Formalize what it means to be a square
sorry

def region1 (P : Point) (A B : Point) : Prop := -- Formalize the definition of region 1
sorry

def region2 (P : Point) (B C : Point) : Prop := -- Formalize the definition of region 2
sorry

-- Additional region definitions (3 through 9)
-- ...

def visible_side (P A B : Point) : Prop := -- Definition of a visible side from a point
sorry

def visible_diagonal (P A C : Point) : Prop := -- Definition of a visible diagonal from a point
sorry

def loci_of_angles (angle : ℝ) : Set Point := -- Definition of loci for a given angle
sorry

-- Main problem statement with the question and conditions as hypotheses
theorem loci_of_square_view (A B C D P : Point) (angle : ℝ) :
    square A B C D →
    (∀ P, (visible_side P A B ∨ visible_side P B C ∨ visible_side P C D ∨ visible_side P D A → 
             P ∈ loci_of_angles angle) ∧ 
         ((region1 P A B ∨ region2 P B C) → visible_diagonal P A C)) →
    -- Additional conditions here
    True :=
-- Prove that the loci is as described in the solution
sorry

end loci_of_square_view_l177_177266


namespace combined_average_age_l177_177206

theorem combined_average_age 
    (avgA : ℕ → ℕ → ℕ) -- defines the average function
    (avgA_cond : avgA 6 240 = 40) 
    (avgB : ℕ → ℕ → ℕ)
    (avgB_cond : avgB 4 100 = 25) 
    (combined_total_age : ℕ := 340) 
    (total_people : ℕ := 10) : avgA (total_people) (combined_total_age) = 34 := 
by
  sorry

end combined_average_age_l177_177206


namespace henrietta_has_three_bedrooms_l177_177425

theorem henrietta_has_three_bedrooms
  (living_room_walls_sqft : ℕ)
  (bedroom_walls_sqft : ℕ)
  (num_bedrooms : ℕ)
  (gallon_coverage_sqft : ℕ)
  (h1 : living_room_walls_sqft = 600)
  (h2 : bedroom_walls_sqft = 400)
  (h3 : gallon_coverage_sqft = 600)
  (h4 : num_bedrooms = 3) : 
  num_bedrooms = 3 :=
by
  exact h4

end henrietta_has_three_bedrooms_l177_177425


namespace trains_distance_l177_177646

theorem trains_distance (t x : ℝ) 
  (h1 : x = 20 * t)
  (h2 : x + 50 = 25 * t) : 
  x + (x + 50) = 450 := 
by 
  -- placeholder for the proof
  sorry

end trains_distance_l177_177646


namespace two_pow_ge_n_cubed_l177_177141

theorem two_pow_ge_n_cubed (n : ℕ) : 2^n ≥ n^3 ↔ n ≥ 10 := 
by sorry

end two_pow_ge_n_cubed_l177_177141


namespace find_number_l177_177036

-- Define the given condition
def number_div_property (num : ℝ) : Prop :=
  num / 0.3 = 7.3500000000000005

-- State the theorem to prove
theorem find_number (num : ℝ) (h : number_div_property num) : num = 2.205 :=
by sorry

end find_number_l177_177036


namespace f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l177_177335

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 / x - 1 else 2 / (-x) - 1

-- Assertion 1: Value of f(-1)
theorem f_neg_one : f (-1) = 1 := 
sorry

-- Assertion 2: f(x) is a decreasing function on (0, +∞)
theorem f_decreasing_on_positive : ∀ a b : ℝ, 0 < b → b < a → f (a) < f (b) := 
sorry

-- Assertion 3: Expression of the function when x < 0
theorem f_expression_on_negative (x : ℝ) (hx : x < 0) : f x = 2 / (-x) - 1 := 
sorry

end f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l177_177335


namespace brets_dinner_tip_calculation_l177_177258

/-
  We need to prove that the percentage of the tip Bret included is 20%, given the conditions.
-/

theorem brets_dinner_tip_calculation :
  let num_meals := 4
  let cost_per_meal := 12
  let num_appetizers := 2
  let cost_per_appetizer := 6
  let rush_fee := 5
  let total_cost := 77
  (total_cost - (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer + rush_fee))
  / (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer) * 100 = 20 :=
by
  sorry

end brets_dinner_tip_calculation_l177_177258


namespace carmen_rope_gcd_l177_177971

/-- Carmen has three ropes with lengths 48, 64, and 80 inches respectively.
    She needs to cut these ropes into pieces of equal length for a craft project,
    ensuring no rope is left unused.
    Prove that the greatest length in inches that each piece can have is 16. -/
theorem carmen_rope_gcd :
  Nat.gcd (Nat.gcd 48 64) 80 = 16 := by
  sorry

end carmen_rope_gcd_l177_177971


namespace draw_odds_l177_177170

theorem draw_odds (x : ℝ) (bet_Zubilo bet_Shaiba bet_Draw payout : ℝ) (h1 : bet_Zubilo = 3 * x) (h2 : bet_Shaiba = 2 * x) (h3 : payout = 6 * x) : 
  bet_Draw * 6 = payout :=
by
  sorry

end draw_odds_l177_177170


namespace angle_expr_correct_l177_177480

noncomputable def angle_expr : Real :=
  Real.cos (40 * Real.pi / 180) * Real.cos (160 * Real.pi / 180) +
  Real.sin (40 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)

theorem angle_expr_correct : angle_expr = -1 / 2 := 
by 
   sorry

end angle_expr_correct_l177_177480


namespace otimes_h_h_h_eq_h_l177_177974

variable (h : ℝ)

def otimes (x y : ℝ) : ℝ := x^3 - y

theorem otimes_h_h_h_eq_h : otimes h (otimes h h) = h := by
  -- Proof goes here, but is omitted
  sorry

end otimes_h_h_h_eq_h_l177_177974


namespace angle_B_side_b_l177_177588

variable (A B C a b c : ℝ)
variable (S : ℝ := 5 * Real.sqrt 3)

-- Conditions
variable (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
variable (h2 : 1/2 * a * c * Real.sin B = S)
variable (h3 : a = 5)

-- The two parts to prove
theorem angle_B (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B) : 
  B = π / 3 := 
  sorry

theorem side_b (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
  (h2 : 1/2 * a * c * Real.sin B = S) (h3 : a = 5) : 
  b = Real.sqrt 21 := 
  sorry

end angle_B_side_b_l177_177588


namespace banknotes_combination_l177_177215

theorem banknotes_combination (a b c d : ℕ) (h : a + b + c + d = 10) (h_val : 2000 * a + 1000 * b + 500 * c + 200 * d = 5000) :
  (a = 0 ∧ b = 0 ∧ c = 10 ∧ d = 0) ∨ 
  (a = 1 ∧ b = 0 ∧ c = 4 ∧ d = 5) ∨ 
  (a = 0 ∧ b = 3 ∧ c = 2 ∧ d = 5) :=
by
  sorry

end banknotes_combination_l177_177215


namespace ants_in_field_l177_177380

-- Defining constants
def width_feet : ℕ := 500
def length_feet : ℕ := 600
def ants_per_square_inch : ℕ := 4
def inches_per_foot : ℕ := 12

-- Converting dimensions from feet to inches
def width_inches : ℕ := width_feet * inches_per_foot
def length_inches : ℕ := length_feet * inches_per_foot

-- Calculating the area of the field in square inches
def field_area_square_inches : ℕ := width_inches * length_inches

-- Calculating the total number of ants
def total_ants : ℕ := ants_per_square_inch * field_area_square_inches

-- Theorem statement
theorem ants_in_field : total_ants = 172800000 :=
by
  -- Proof is skipped
  sorry

end ants_in_field_l177_177380


namespace Julie_hours_per_week_school_l177_177308

noncomputable def summer_rate : ℚ := 4500 / (36 * 10)

noncomputable def school_rate : ℚ := summer_rate * 1.10

noncomputable def total_school_hours_needed : ℚ := 9000 / school_rate

noncomputable def hours_per_week_school : ℚ := total_school_hours_needed / 40

theorem Julie_hours_per_week_school : hours_per_week_school = 16.36 := by
  sorry

end Julie_hours_per_week_school_l177_177308


namespace min_value_x_add_y_div_2_l177_177841

theorem min_value_x_add_y_div_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y - 2 * x - y = 0) :
  ∃ x y, 0 < x ∧ 0 < y ∧ (x * y - 2 * x - y = 0 ∧ x + y / 2 = 4) :=
sorry

end min_value_x_add_y_div_2_l177_177841


namespace jack_time_to_school_l177_177261

noncomputable def dave_speed : ℚ := 8000 -- cm/min
noncomputable def distance_to_school : ℚ := 160000 -- cm
noncomputable def jack_speed : ℚ := 7650 -- cm/min
noncomputable def jack_start_delay : ℚ := 10 -- min

theorem jack_time_to_school : (distance_to_school / jack_speed) - jack_start_delay = 10.92 :=
by
  sorry

end jack_time_to_school_l177_177261


namespace no_solution_for_given_m_l177_177940

theorem no_solution_for_given_m (x m : ℝ) (h1 : x ≠ 5) (h2 : x ≠ 8) :
  (∀ y : ℝ, (y - 2) / (y - 5) = (y - m) / (y - 8) → false) ↔ m = 5 :=
by
  sorry

end no_solution_for_given_m_l177_177940


namespace cos_120_eq_neg_half_l177_177085

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l177_177085


namespace total_number_of_squares_up_to_50th_ring_l177_177398

def number_of_squares_up_to_50th_ring : Nat :=
  let central_square := 1
  let sum_rings := (50 * (50 + 1)) * 4  -- Using the formula for arithmetic series sum where a = 8 and d = 8 and n = 50
  central_square + sum_rings

theorem total_number_of_squares_up_to_50th_ring : number_of_squares_up_to_50th_ring = 10201 :=
  by  -- This statement means we believe the theorem is true and will be proven.
    sorry                                                      -- Proof omitted, will need to fill this in later

end total_number_of_squares_up_to_50th_ring_l177_177398


namespace leah_coins_value_l177_177722

theorem leah_coins_value : 
  ∃ (p n : ℕ), p + n = 15 ∧ n + 1 = p ∧ 5 * n + 1 * p = 43 := 
by
  sorry

end leah_coins_value_l177_177722


namespace range_of_smallest_nonprime_with_condition_l177_177725

def smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 : ℕ :=
121

theorem range_of_smallest_nonprime_with_condition :
  120 < smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ∧ 
  smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ≤ 130 :=
by
  unfold smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10
  exact ⟨by norm_num, by norm_num⟩

end range_of_smallest_nonprime_with_condition_l177_177725


namespace total_pages_in_book_is_250_l177_177442

-- Definitions
def avg_pages_first_part := 36
def days_first_part := 3
def avg_pages_second_part := 44
def days_second_part := 3
def pages_last_day := 10

-- Calculate total pages
def total_pages := (days_first_part * avg_pages_first_part) + (days_second_part * avg_pages_second_part) + pages_last_day

-- Theorem statement
theorem total_pages_in_book_is_250 : total_pages = 250 := by
  sorry

end total_pages_in_book_is_250_l177_177442


namespace parabola_transformation_l177_177752

theorem parabola_transformation :
  (∀ x : ℝ, y = 2 * x^2 → y = 2 * (x-3)^2 - 1) := by
  sorry

end parabola_transformation_l177_177752


namespace quadratic_distinct_roots_l177_177290

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem quadratic_distinct_roots (k : ℝ) :
  (k ≠ 0) ∧ (1 > k) ↔ has_two_distinct_real_roots k (-6) 9 :=
by
  sorry

end quadratic_distinct_roots_l177_177290


namespace pascals_triangle_ratio_456_l177_177845

theorem pascals_triangle_ratio_456 (n : ℕ) :
  (∃ r : ℕ,
    (n.choose r * 5 = (n.choose (r + 1)) * 4) ∧
    ((n.choose (r + 1)) * 6 = (n.choose (r + 2)) * 5)) →
  n = 98 :=
sorry

end pascals_triangle_ratio_456_l177_177845


namespace find_height_l177_177049

-- Defining the known conditions
def length : ℝ := 3
def width : ℝ := 5
def cost_per_sqft : ℝ := 20
def total_cost : ℝ := 1240

-- Defining the unknown dimension as a variable
variable (height : ℝ)

-- Surface area formula for a rectangular tank
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

-- Given statement to prove that the height is 2 feet.
theorem find_height : surface_area length width height = total_cost / cost_per_sqft → height = 2 := by
  sorry

end find_height_l177_177049


namespace coordinates_of_M_l177_177703

-- Let M be a point in the 2D Cartesian plane
variable {x y : ℝ}

-- Definition of the conditions
def distance_from_x_axis (y : ℝ) : Prop := abs y = 1
def distance_from_y_axis (x : ℝ) : Prop := abs x = 2

-- Theorem to prove
theorem coordinates_of_M (hx : distance_from_y_axis x) (hy : distance_from_x_axis y) :
  (x = 2 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
sorry

end coordinates_of_M_l177_177703


namespace Fiona_Less_Than_Charles_l177_177748

noncomputable def percentDifference (a b : ℝ) : ℝ :=
  ((a - b) / a) * 100

theorem Fiona_Less_Than_Charles : percentDifference 600 (450 * 1.1) = 17.5 :=
by
  sorry

end Fiona_Less_Than_Charles_l177_177748


namespace ensure_two_of_each_l177_177018

theorem ensure_two_of_each {A B : ℕ} (hA : A = 10) (hB : B = 10) :
  ∃ n : ℕ, n = 12 ∧
  ∀ (extracted : ℕ → ℕ),
    (extracted 0 + extracted 1 = n) →
    (extracted 0 ≥ 2 ∧ extracted 1 ≥ 2) :=
by
  sorry

end ensure_two_of_each_l177_177018


namespace speed_difference_l177_177058

noncomputable def park_distance : ℝ := 10
noncomputable def kevin_time_hours : ℝ := 1 / 4
noncomputable def joel_time_hours : ℝ := 2

theorem speed_difference : (10 / kevin_time_hours) - (10 / joel_time_hours) = 35 := by
  sorry

end speed_difference_l177_177058


namespace simplify_and_evaluate_l177_177912

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2) :
  ( (1 + x) / (1 - x) / (x - (2 * x / (1 - x))) = - (Real.sqrt 2 + 2) / 2) :=
by
  rw [h]
  simp
  sorry

end simplify_and_evaluate_l177_177912


namespace smallest_positive_integer_l177_177933

theorem smallest_positive_integer (m n : ℤ) : ∃ k : ℕ, k > 0 ∧ (∃ m n : ℤ, k = 5013 * m + 111111 * n) ∧ k = 3 :=
by {
  sorry 
}

end smallest_positive_integer_l177_177933


namespace total_cost_of_vacation_l177_177369

noncomputable def total_cost (C : ℝ) : Prop :=
  let cost_per_person_three := C / 3
  let cost_per_person_four := C / 4
  cost_per_person_three - cost_per_person_four = 60

theorem total_cost_of_vacation (C : ℝ) (h : total_cost C) : C = 720 :=
  sorry

end total_cost_of_vacation_l177_177369


namespace find_a_range_l177_177152

open Set

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

theorem find_a_range :
  (p a ∧ ¬ q a ∨ ¬ p a ∧ q a) ↔ (-1 < a ∧ a ≤ 0 ∨ a ≥ 2) :=
by
  sorry

end find_a_range_l177_177152


namespace cosine_120_eq_negative_half_l177_177120

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l177_177120


namespace circumradius_relation_l177_177302

-- Definitions of the geometric constructs from the problem
open EuclideanGeometry

noncomputable def circumradius (A B C : Point) : Real := sorry

-- Given conditions
def angle_bisectors_intersect_at_point (A B C B1 C1 I : Point) : Prop := sorry
def line_intersects_circumcircle_at_points (B1 C1 : Point) (circumcircle : Circle) (M N : Point) : Prop := sorry

-- Main statement to prove
theorem circumradius_relation
  (A B C B1 C1 I M N : Point)
  (circumcircle : Circle)
  (h1 : angle_bisectors_intersect_at_point A B C B1 C1 I)
  (h2 : line_intersects_circumcircle_at_points B1 C1 circumcircle M N) :
  circumradius M I N = 2 * circumradius A B C :=
sorry

end circumradius_relation_l177_177302


namespace expected_number_of_matches_l177_177061

def xi (i : ℕ) (σ : list ℕ) : ℕ :=
  if σ.nth (i - 1) = some i then 1 else 0

theorem expected_number_of_matches :
  (1 / 720) * ∑ σ in (list.permutations [1, 2, 3, 4, 5, 6]), ∑ i in [1, 2, 3, 4, 5, 6], xi i σ = 1 :=
sorry

end expected_number_of_matches_l177_177061


namespace number_1973_occurrences_l177_177918

theorem number_1973_occurrences :
  ∀ n, Nat.Prime n →
  (∀ k, k ≤ n → ∃ a b c : ℕ, (∀ m, m ≤ k → (a + b = c))) →
  ∀ iter, iter = 1973 →
  EulerTotient (n - 1) = 1972 :=
by
  assume n h_prime h_seq iter h_iter
  sorry

end number_1973_occurrences_l177_177918


namespace faith_work_days_per_week_l177_177270

theorem faith_work_days_per_week 
  (hourly_wage : ℝ)
  (normal_hours_per_day : ℝ)
  (overtime_hours_per_day : ℝ)
  (weekly_earnings : ℝ)
  (overtime_rate_multiplier : ℝ) :
  hourly_wage = 13.50 → 
  normal_hours_per_day = 8 → 
  overtime_hours_per_day = 2 → 
  weekly_earnings = 675 →
  overtime_rate_multiplier = 1.5 →
  ∀ days_per_week : ℝ, days_per_week = 5 :=
sorry

end faith_work_days_per_week_l177_177270


namespace boat_stream_speeds_l177_177653

variable (x y : ℝ)

theorem boat_stream_speeds (h1 : 20 + x ≠ 0) (h2 : 40 - y ≠ 0) :
  380 = 7 * x + 13 * y ↔ 
  26 * (40 - y) = 14 * (20 + x) :=
by { sorry }

end boat_stream_speeds_l177_177653


namespace average_speed_l177_177963

theorem average_speed (x : ℝ) (h₀ : x > 0) : 
  let time1 := x / 90
  let time2 := 2 * x / 20
  let total_distance := 3 * x
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 27 := 
by
  sorry

end average_speed_l177_177963


namespace intersection_points_count_l177_177791

theorem intersection_points_count
  : (∀ n : ℤ, ∃ (x y : ℝ), (x - ⌊x⌋) ^ 2 + y ^ 2 = 2 * (x - ⌊x⌋) ∨ y = 1 / 3 * x) →
    (∃ count : ℕ, count = 12) :=
by
  sorry

end intersection_points_count_l177_177791


namespace range_of_a_l177_177329

variable {f : ℝ → ℝ} {a : ℝ}
open Real

-- Conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def f_positive_at_2 (f : ℝ → ℝ) : Prop := f 2 > 1
def f_value_at_2014 (f : ℝ → ℝ) (a : ℝ) : Prop := f 2014 = (a + 3) / (a - 3)

-- Proof Problem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : odd_function f)
  (h2 : periodic_function f 7)
  (h3 : f_positive_at_2 f)
  (h4 : f_value_at_2014 f a) :
  0 < a ∧ a < 3 :=
sorry

end range_of_a_l177_177329


namespace intersection_is_2_l177_177730

noncomputable def M : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def N : Set ℝ := {x | x^2 ≥ 2 * x}
noncomputable def intersection : Set ℝ := M ∩ N

theorem intersection_is_2 : intersection = {2} := by
  sorry

end intersection_is_2_l177_177730


namespace a_three_equals_35_l177_177999

-- Define the mathematical sequences and functions
def S (n : ℕ) : ℕ := 5 * n^2 + 10 * n

def a (n : ℕ) : ℕ := S (n + 1) - S n

-- The proposition we want to prove
theorem a_three_equals_35 : a 2 = 35 := by 
  sorry

end a_three_equals_35_l177_177999


namespace integer_part_M_is_4_l177_177279

-- Define the variables and conditions based on the problem statement
variable (a b c : ℝ)

-- This non-computable definition includes the main mathematical expression we need to evaluate
noncomputable def M (a b c : ℝ) := Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1)

-- The theorem we need to prove
theorem integer_part_M_is_4 (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 
  ⌊M a b c⌋ = 4 := 
by 
  sorry

end integer_part_M_is_4_l177_177279


namespace find_angle_B_l177_177882

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l177_177882


namespace ratio_of_wire_lengths_l177_177789

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

end ratio_of_wire_lengths_l177_177789


namespace triangle_perimeter_l177_177779

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

variable (a b c : ℝ)

theorem triangle_perimeter
  (h1 : 90 = (1/2) * 18 * b)
  (h2 : right_triangle 18 b c) :
  18 + b + c = 28 + 2 * Real.sqrt 106 :=
by
  sorry

end triangle_perimeter_l177_177779


namespace inequality_ln_l177_177464

theorem inequality_ln (x : ℝ) (h₁ : x > -1) (h₂ : x ≠ 0) :
    (2 * abs x) / (2 + x) < abs (Real.log (1 + x)) ∧ abs (Real.log (1 + x)) < (abs x) / Real.sqrt (1 + x) :=
by
  sorry

end inequality_ln_l177_177464


namespace find_roots_l177_177802

theorem find_roots (x : ℝ) : (x^2 + x = 0) ↔ (x = 0 ∨ x = -1) := 
by sorry

end find_roots_l177_177802


namespace angle_B_in_triangle_l177_177869

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l177_177869


namespace unique_solution_system_eqns_l177_177469

theorem unique_solution_system_eqns :
  ∃ (x y : ℝ), (2 * x - 3 * |y| = 1 ∧ |x| + 2 * y = 4 ∧ x = 2 ∧ y = 1) :=
sorry

end unique_solution_system_eqns_l177_177469


namespace probability_P_X_geq_5_l177_177990

-- Assumptions
variables {X : Type*} [MeasureTheory.ProbabilityMeasure X]

-- X follows normal distribution N(3, δ^2)
def is_normal_distribution (X : Type*) [MeasureTheory.ProbabilityMeasure X] : Prop :=
MeasureTheory.ProbabilityMeasure (fun x => PDFNormal x 3 δ^2)

-- Probability condition
axiom P_condition : MeasureTheory.ProbabilityMeasure.P (1 < X ∧ X ≤ 3) = 0.3

-- Proof statement
theorem probability_P_X_geq_5 (hX : is_normal_distribution X) : 
  MeasureTheory.ProbabilityMeasure.P (X ≥ 5) = 0.2 :=
sorry

end probability_P_X_geq_5_l177_177990


namespace simplified_result_l177_177013

theorem simplified_result (a b M : ℝ) (h1 : (2 * a) / (a ^ 2 - b ^ 2) - 1 / M = 1 / (a - b))
  (h2 : M - (a - b) = 2 * b) : (2 * a) / (a ^ 2 - b ^ 2) - 1 / (a - b) = 1 / (a + b) :=
by
  sorry

end simplified_result_l177_177013


namespace angle_measure_l177_177472

-- Define the problem conditions
def angle (x : ℝ) : Prop :=
  let complement := 3 * x + 6
  x + complement = 90

-- The theorem to prove
theorem angle_measure : ∃ x : ℝ, angle x ∧ x = 21 := 
sorry

end angle_measure_l177_177472


namespace least_xy_value_l177_177414

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
sorry

end least_xy_value_l177_177414


namespace free_cytosine_molecules_req_l177_177656

-- Definition of conditions
def DNA_base_pairs := 500
def AT_percentage := 34 / 100
def CG_percentage := 1 - AT_percentage

-- The total number of bases
def total_bases := 2 * DNA_base_pairs

-- The number of C or G bases
def CG_bases := total_bases * CG_percentage

-- Finally, the total number of free cytosine deoxyribonucleotide molecules 
def free_cytosine_molecules := 2 * CG_bases

-- Problem statement: Prove that the number of free cytosine deoxyribonucleotide molecules required is 1320
theorem free_cytosine_molecules_req : free_cytosine_molecules = 1320 :=
by
  -- conditions are defined, the proof is omitted
  sorry

end free_cytosine_molecules_req_l177_177656


namespace students_not_picked_l177_177758

/-- There are 36 students trying out for the school's trivia teams. 
If some of them didn't get picked and the rest were put into 3 groups with 9 students in each group,
prove that the number of students who didn't get picked is 9. -/

theorem students_not_picked (total_students groups students_per_group picked_students not_picked_students : ℕ)
    (h1 : total_students = 36)
    (h2 : groups = 3)
    (h3 : students_per_group = 9)
    (h4 : picked_students = groups * students_per_group)
    (h5 : not_picked_students = total_students - picked_students) :
    not_picked_students = 9 :=
by
  sorry

end students_not_picked_l177_177758


namespace angle_B_eq_3pi_over_10_l177_177889

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l177_177889


namespace determine_x_l177_177976

variables {m n x : ℝ}
variable (k : ℝ)
variable (Hmn : m ≠ 0 ∧ n ≠ 0)
variable (Hk : k = 5 * (m^2 - n^2))

theorem determine_x (H : (x + 2 * m)^2 - (x - 3 * n)^2 = k) : 
  x = (5 * m^2 - 9 * n^2) / (4 * m + 6 * n) := by
  sorry

end determine_x_l177_177976


namespace greatest_product_of_two_integers_sum_2006_l177_177009

theorem greatest_product_of_two_integers_sum_2006 :
  ∃ (x y : ℤ), x + y = 2006 ∧ x * y = 1006009 :=
by
  sorry

end greatest_product_of_two_integers_sum_2006_l177_177009


namespace seq_contains_divisible_term_l177_177465

open Nat

theorem seq_contains_divisible_term (n : ℕ) (h1 : odd n) (h2 : 1 < n) :
  ∃ k : ℕ, k < n ∧ n ∣ (2 ^ k - 1) := by
  sorry

end seq_contains_divisible_term_l177_177465


namespace hypotenuse_length_l177_177246

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 36) (h2 : 0.5 * a * b = 24) (h3 : a^2 + b^2 = c^2) :
  c = 50 / 3 :=
sorry

end hypotenuse_length_l177_177246


namespace find_side_b_l177_177859

theorem find_side_b
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 2 * Real.sin B = Real.sin A + Real.sin C)
  (h2 : Real.cos B = 3 / 5)
  (h3 : (1 / 2) * a * c * Real.sin B = 4) :
  b = 4 * Real.sqrt 6 / 3 := 
sorry

end find_side_b_l177_177859


namespace find_kn_l177_177979

theorem find_kn (k n : ℕ) (h : k * n^2 - k * n - n^2 + n = 94) : k = 48 ∧ n = 2 := 
by 
  sorry

end find_kn_l177_177979


namespace cost_per_blue_shirt_l177_177332

theorem cost_per_blue_shirt :
  let pto_spent := 2317
  let num_kindergarten := 101
  let cost_orange := 5.80
  let total_orange := num_kindergarten * cost_orange

  let num_first_grade := 113
  let cost_yellow := 5
  let total_yellow := num_first_grade * cost_yellow

  let num_third_grade := 108
  let cost_green := 5.25
  let total_green := num_third_grade * cost_green

  let total_other_shirts := total_orange + total_yellow + total_green
  let pto_spent_on_blue := pto_spent - total_other_shirts

  let num_second_grade := 107
  let cost_per_blue_shirt := pto_spent_on_blue / num_second_grade

  cost_per_blue_shirt = 5.60 :=
by
  sorry

end cost_per_blue_shirt_l177_177332


namespace quadratic_inverse_condition_l177_177318

theorem quadratic_inverse_condition : 
  (∀ x₁ x₂ : ℝ, (x₁ ≥ 2 ∧ x₂ ≥ 2 ∧ x₁ ≠ x₂) → (x₁^2 - 4*x₁ + 5 ≠ x₂^2 - 4*x₂ + 5)) :=
sorry

end quadratic_inverse_condition_l177_177318


namespace decagon_perimeter_l177_177679

theorem decagon_perimeter (num_sides : ℕ) (side_length : ℝ) (h_num_sides : num_sides = 10) (h_side_length : side_length = 3) : 
  (num_sides * side_length = 30) :=
by
  sorry

end decagon_perimeter_l177_177679


namespace exponents_mod_7_l177_177321

theorem exponents_mod_7 : (2222 ^ 5555 + 5555 ^ 2222) % 7 = 0 := 
by 
  -- sorries here because no proof is needed as stated
  sorry

end exponents_mod_7_l177_177321


namespace arithmetic_sequence_sum_l177_177745

theorem arithmetic_sequence_sum :
  ∃ (a_n : ℕ → ℝ) (d : ℝ), 
  (∀ n, a_n n = a_n 0 + n * d) ∧
  d > 0 ∧
  a_n 0 + a_n 1 + a_n 2 = 15 ∧
  a_n 0 * a_n 1 * a_n 2 = 80 →
  a_n 10 + a_n 11 + a_n 12 = 135 :=
by
  sorry

end arithmetic_sequence_sum_l177_177745


namespace factorization_correct_l177_177402

theorem factorization_correct (x : ℝ) :
  (x - 3) * (x - 1) * (x - 2) * (x + 4) + 24 = (x - 2) * (x + 3) * (x^2 + x - 8) := 
sorry

end factorization_correct_l177_177402


namespace cos_120_eq_neg_one_half_l177_177075

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l177_177075


namespace ben_chairs_in_10_days_l177_177388

def number_of_chairs (days hours_per_shift hours_rocking_chair hours_dining_chair hours_armchair : ℕ) : ℕ × ℕ × ℕ :=
  let rocking_chairs_per_day := hours_per_shift / hours_rocking_chair
  let remaining_hours_after_rocking_chairs := hours_per_shift % hours_rocking_chair
  let dining_chairs_per_day := remaining_hours_after_rocking_chairs / hours_dining_chair
  let remaining_hours_after_dining_chairs := remaining_hours_after_rocking_chairs % hours_dining_chair
  if remaining_hours_after_dining_chairs >= hours_armchair then
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, days * (remaining_hours_after_dining_chairs / hours_armchair))
  else
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, 0)

theorem ben_chairs_in_10_days :
  number_of_chairs 10 8 5 3 6 = (10, 10, 0) :=
by 
  sorry

end ben_chairs_in_10_days_l177_177388


namespace crayons_loss_difference_l177_177605

theorem crayons_loss_difference (crayons_given crayons_lost : ℕ) 
  (h_given : crayons_given = 90) 
  (h_lost : crayons_lost = 412) : 
  crayons_lost - crayons_given = 322 :=
by
  sorry

end crayons_loss_difference_l177_177605


namespace mono_increasing_intervals_l177_177262

noncomputable def f : ℝ → ℝ :=
by sorry

theorem mono_increasing_intervals (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_sym : ∀ x, f x = f (-2 - x))
  (h_decr1 : ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ -1 → f y ≤ f x) :
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f x ≤ f y) ∧
  (∀ x y, 3 ≤ x ∧ x < y ∧ y ≤ 4 → f x ≤ f y) :=
sorry

end mono_increasing_intervals_l177_177262


namespace cos_120_eq_neg_half_l177_177096

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l177_177096


namespace probability_rain_all_three_days_l177_177211

-- Define the probabilities as constant values
def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.5
def prob_rain_sunday : ℝ := 0.3
def prob_rain_sunday_given_fri_sat : ℝ := 0.6

-- Define the probability of raining all three days considering the conditional probabilities
def prob_rain_all_three_days : ℝ :=
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday_given_fri_sat

-- Prove that the probability of rain on all three days is 12%
theorem probability_rain_all_three_days : prob_rain_all_three_days = 0.12 :=
by
  sorry

end probability_rain_all_three_days_l177_177211


namespace cos_120_eq_neg_half_l177_177104

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l177_177104


namespace time_for_C_alone_to_finish_the_job_l177_177509

variable {A B C : ℝ} -- Declare work rates as real numbers

-- Define the conditions
axiom h1 : A + B = 1/15
axiom h2 : A + B + C = 1/10

-- Define the theorem to prove
theorem time_for_C_alone_to_finish_the_job : C = 1/30 :=
by
  apply sorry

end time_for_C_alone_to_finish_the_job_l177_177509


namespace arithmetic_sequence_first_term_l177_177582

theorem arithmetic_sequence_first_term :
  ∃ a₁ a₂ d : ℤ, a₂ = -5 ∧ d = 3 ∧ a₂ = a₁ + d ∧ a₁ = -8 :=
by
  sorry

end arithmetic_sequence_first_term_l177_177582


namespace fraction_multiplication_l177_177492

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l177_177492


namespace rectangle_area_l177_177961

theorem rectangle_area (side_length : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  side_length^2 = 64 → 
  rect_width = side_length →
  rect_length = 3 * rect_width →
  rect_width * rect_length = 192 := 
by
  intros h1 h2 h3
  sorry

end rectangle_area_l177_177961


namespace minimum_value_is_six_l177_177701

noncomputable def minimum_value (m n : ℝ) (h : m > 2 * n) : ℝ :=
  m + (4 * n ^ 2 - 2 * m * n + 9) / (m - 2 * n)

theorem minimum_value_is_six (m n : ℝ) (h : m > 2 * n) : minimum_value m n h = 6 := 
sorry

end minimum_value_is_six_l177_177701


namespace Monica_class_ratio_l177_177315

theorem Monica_class_ratio : 
  (20 + 25 + 25 + x + 28 + 28 = 136) → 
  (x = 10) → 
  (x / 20 = 1 / 2) :=
by 
  intros h h_x
  sorry

end Monica_class_ratio_l177_177315


namespace number_of_classmates_l177_177824

theorem number_of_classmates (total_apples : ℕ) (apples_per_classmate : ℕ) (people_in_class : ℕ) 
  (h1 : total_apples = 15) (h2 : apples_per_classmate = 5) (h3 : people_in_class = total_apples / apples_per_classmate) : 
  people_in_class = 3 :=
by sorry

end number_of_classmates_l177_177824


namespace inequality_l177_177899

theorem inequality (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) :=
by
  sorry

end inequality_l177_177899


namespace sum_integers_neg40_to_60_l177_177935

theorem sum_integers_neg40_to_60 : (Finset.range (60 + 41)).sum (fun i => i - 40) = 1010 := by
  sorry

end sum_integers_neg40_to_60_l177_177935


namespace c_n_monotonically_decreasing_l177_177567

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ)

theorem c_n_monotonically_decreasing 
    (h_a0 : a 0 = 0)
    (h_b : ∀ n ≥ 1, b n = a n - a (n - 1))
    (h_c : ∀ n ≥ 1, c n = a n / n)
    (h_bn_decrease : ∀ n ≥ 1, b n ≥ b (n + 1)) : 
    ∀ n ≥ 2, c n ≤ c (n - 1) := 
by
  sorry

end c_n_monotonically_decreasing_l177_177567


namespace travel_time_comparison_l177_177592

theorem travel_time_comparison
  (v : ℝ) -- speed during the first trip
  (t1 : ℝ) (t2 : ℝ)
  (h1 : t1 = 80 / v) -- time for the first trip
  (h2 : t2 = 100 / v) -- time for the second trip
  : t2 = 1.25 * t1 :=
by
  sorry

end travel_time_comparison_l177_177592


namespace seating_arrangement_ways_l177_177773

open Nat

theorem seating_arrangement_ways : 
  let boys := 4
      girls := 1
  in (2 * boys - 1) * factorial boys = 120 := by
  let boys := 4
  let girls := 1
  have h1 : (2 * boys - 1) = 5 := by norm_num
  have h2 : factorial boys = 24 := by norm_num
  calc
      (2 * boys - 1) * factorial boys
      = 5 * 24 : by rw [h1, h2]
  ... = 120  : by norm_num

end seating_arrangement_ways_l177_177773


namespace volume_of_regular_tetrahedron_l177_177615

noncomputable def volume_of_tetrahedron (a H : ℝ) : ℝ :=
  (a^2 * H) / (6 * Real.sqrt 2)

theorem volume_of_regular_tetrahedron
  (d_face : ℝ)
  (d_edge : ℝ)
  (h : Real.sqrt 14 = d_edge)
  (h1 : 2 = d_face)
  (volume_approx : ℝ) :
  ∃ a H, (d_face = Real.sqrt ((H / 2)^2 + (a * Real.sqrt 3 / 6)^2) ∧ 
          d_edge = Real.sqrt ((H / 2)^2 + (a / (2 * Real.sqrt 3))^2) ∧ 
          Real.sqrt (volume_of_tetrahedron a H) = 533.38) :=
  sorry

end volume_of_regular_tetrahedron_l177_177615


namespace rectangle_ratio_l177_177409

theorem rectangle_ratio (s x y : ℝ) 
  (h1 : 4 * (x * y) + s^2 = 9 * s^2)
  (h2 : x + s = 3 * s)
  (h3 : s + 2 * y = 3 * s) :
  x / y = 2 :=
by
  sorry

end rectangle_ratio_l177_177409


namespace least_n_for_reducible_fraction_l177_177980

theorem least_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, n - 13 = 71 * k) ∧ n = 84 := by
  sorry

end least_n_for_reducible_fraction_l177_177980


namespace find_min_value_l177_177981

noncomputable def expression (x : ℝ) : ℝ :=
  (Real.sin x ^ 8 + Real.cos x ^ 8 + 2) / (Real.sin x ^ 6 + Real.cos x ^ 6 + 2)

theorem find_min_value : ∃ x : ℝ, expression x = 5 / 4 :=
sorry

end find_min_value_l177_177981


namespace greatest_four_digit_p_l177_177518

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def reverse_digits (n : ℕ) : ℕ := 
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1
def is_divisible_by (a b : ℕ) : Prop := b ∣ a

-- Proof problem
theorem greatest_four_digit_p (p : ℕ) (q : ℕ) 
    (hp1 : is_four_digit p)
    (hp2 : q = reverse_digits p)
    (hp3 : is_four_digit q)
    (hp4 : is_divisible_by p 63)
    (hp5 : is_divisible_by q 63)
    (hp6 : is_divisible_by p 19) :
  p = 5985 :=
sorry

end greatest_four_digit_p_l177_177518


namespace sum_integers_neg40_to_60_l177_177937

theorem sum_integers_neg40_to_60 : 
  (Finset.sum (Finset.range (60 + 40 + 1)) (λ x => x - 40)) = 1010 := sorry

end sum_integers_neg40_to_60_l177_177937


namespace flight_duration_sum_l177_177444

theorem flight_duration_sum (h m : ℕ) (h_hours : h = 11) (m_minutes : m = 45) (time_limit : 0 < m ∧ m < 60) :
  h + m = 56 :=
by
  sorry

end flight_duration_sum_l177_177444


namespace isosceles_triangle_angle_between_vectors_l177_177174

theorem isosceles_triangle_angle_between_vectors 
  (α β γ : ℝ) 
  (h1: α + β + γ = 180)
  (h2: α = 120) 
  (h3: β = γ):
  180 - β = 150 :=
sorry

end isosceles_triangle_angle_between_vectors_l177_177174


namespace collinear_vectors_triangle_C_sum_of_sides_l177_177421

theorem collinear_vectors_triangle_C
  (A B C a b c : ℝ)
  (habc : a, b, c)
  (h1 : (2*b - a) * cos C = c * cos A)
  (h2 : a / sin A = b / sin B = c / sin C)
  (h3 : sin (A + C) = sin B) :
  C = π / 3 :=
sorry

theorem sum_of_sides
  (A B C a b c : ℝ)
  (hC : C = π / 3)
  (hc : c = sqrt 3)
  (harea : (1/2) * a * b * sin (π / 3) = sqrt 3 / 2) :
  a + b = 3 :=
sorry

end collinear_vectors_triangle_C_sum_of_sides_l177_177421


namespace cos_120_degrees_eq_l177_177114

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l177_177114


namespace find_a_plus_b_l177_177839

theorem find_a_plus_b (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : a^2 - b^4 = 2009) : a + b = 47 :=
by
  sorry

end find_a_plus_b_l177_177839


namespace triangle_BC_length_l177_177441

theorem triangle_BC_length (A : ℝ) (AC : ℝ) (S : ℝ) (BC : ℝ)
  (h1 : A = 60) (h2 : AC = 16) (h3 : S = 220 * Real.sqrt 3) :
  BC = 49 :=
by
  sorry

end triangle_BC_length_l177_177441


namespace cos_120_eq_neg_half_l177_177108

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l177_177108


namespace relationship_among_a_b_c_l177_177693

noncomputable def a : ℝ := Real.log (7 / 2) / Real.log 3
noncomputable def b : ℝ := (1 / 4)^(1 / 3)
noncomputable def c : ℝ := -Real.log 5 / Real.log 3

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_among_a_b_c_l177_177693


namespace area_of_frame_l177_177349

def width : ℚ := 81 / 4
def depth : ℚ := 148 / 9
def area (w d : ℚ) : ℚ := w * d

theorem area_of_frame : area width depth = 333 := by
  sorry

end area_of_frame_l177_177349


namespace number_of_balls_sold_l177_177462

-- Definitions from conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 120
def loss : ℕ := 5 * cost_price_per_ball

-- Mathematically equivalent proof statement
theorem number_of_balls_sold (n : ℕ) (h : n * cost_price_per_ball - selling_price = loss) : n = 11 :=
  sorry

end number_of_balls_sold_l177_177462


namespace leaves_falling_every_day_l177_177534

-- Definitions of the conditions
def roof_capacity := 500 -- in pounds
def leaves_per_pound := 1000 -- number of leaves per pound
def collapse_time := 5000 -- in days

-- Function to calculate the number of leaves falling each day
def leaves_per_day (roof_capacity : Nat) (leaves_per_pound : Nat) (collapse_time : Nat) : Nat :=
  (roof_capacity * leaves_per_pound) / collapse_time

-- Theorem stating the expected result
theorem leaves_falling_every_day :
  leaves_per_day roof_capacity leaves_per_pound collapse_time = 100 :=
by
  sorry

end leaves_falling_every_day_l177_177534


namespace sum_cyc_geq_one_l177_177911

theorem sum_cyc_geq_one (a b c : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hcond : a * b + b * c + c * a = a * b * c) :
  (a^4 / (b * (b^4 + c^3)) + b^4 / (c * (c^3 + a^4)) + c^4 / (a * (a^4 + b^3))) ≥ 1 :=
sorry

end sum_cyc_geq_one_l177_177911


namespace angle_B_eq_3pi_over_10_l177_177887

theorem angle_B_eq_3pi_over_10
  (a b c A B : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (C_eq : ∠ C = π / 5)
  (h_tri : ∠ A + ∠ B + ∠ C = π)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < ∠ A)
  (hB : 0 < ∠ B)
  (C_pos : 0 < ∠ C)
  (C_lt_pi : ∠ C < π) :
  B = 3 * π / 10 :=
sorry

end angle_B_eq_3pi_over_10_l177_177887


namespace length_of_AB_is_correct_l177_177684

noncomputable def hyperbola_length_of_AB : ℝ :=
  let a := 1
  let b := sqrt 3
  let c := sqrt (a^2 + b^2)
  let right_focus := (c, 0)
  let asymptote1 := λ x : ℝ, b * x / a
  let asymptote2 := λ x : ℝ, -b * x / a
  let y1 := asymptote1 (2 : ℝ)
  let y2 := asymptote2 (2 : ℝ)
  |y1 - y2|

-- Theorem stating that the length of AB is 4√3
theorem length_of_AB_is_correct : hyperbola_length_of_AB = 4 * sqrt 3 :=
sorry

end length_of_AB_is_correct_l177_177684


namespace impossible_to_achieve_desired_piles_l177_177490

def initial_piles : List ℕ := [51, 49, 5]

def desired_piles : List ℕ := [52, 48, 5]

def combine_piles (x y : ℕ) : ℕ := x + y

def divide_pile (x : ℕ) (h : x % 2 = 0) : List ℕ := [x / 2, x / 2]

theorem impossible_to_achieve_desired_piles :
  ∀ (piles : List ℕ), 
    (piles = initial_piles) →
    (∀ (p : List ℕ), 
      (p = desired_piles) → 
      False) :=
sorry

end impossible_to_achieve_desired_piles_l177_177490


namespace parabola_intersects_once_compare_y_values_l177_177423

noncomputable def parabola (x : ℝ) (m : ℝ) : ℝ := -2 * x^2 + 4 * x + m

theorem parabola_intersects_once (m : ℝ) : 
  ∃ x, parabola x m = 0 ↔ m = -2 := 
by 
  sorry

theorem compare_y_values (x1 x2 m : ℝ) (h1 : x1 > x2) (h2 : x2 > 2) : 
  parabola x1 m < parabola x2 m :=
by 
  sorry

end parabola_intersects_once_compare_y_values_l177_177423


namespace cost_of_each_muffin_l177_177594

-- Define the cost of juice
def juice_cost : ℝ := 1.45

-- Define the total cost paid by Kevin
def total_cost : ℝ := 3.70

-- Assume the cost of each muffin
def muffin_cost (M : ℝ) : Prop := 
  3 * M + juice_cost = total_cost

-- The theorem we aim to prove
theorem cost_of_each_muffin : muffin_cost 0.75 :=
by
  -- Here the proof would go
  sorry

end cost_of_each_muffin_l177_177594


namespace total_cost_is_correct_l177_177662

def num_children : ℕ := 5
def daring_children : ℕ := 3
def ferris_wheel_cost_per_child : ℕ := 5
def merry_go_round_cost_per_child : ℕ := 3
def ice_cream_cones_per_child : ℕ := 2
def ice_cream_cost_per_cone : ℕ := 8

def total_spent_on_ferris_wheel : ℕ := daring_children * ferris_wheel_cost_per_child
def total_spent_on_merry_go_round : ℕ := num_children * merry_go_round_cost_per_child
def total_spent_on_ice_cream : ℕ := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone

def total_spent : ℕ := total_spent_on_ferris_wheel + total_spent_on_merry_go_round + total_spent_on_ice_cream

theorem total_cost_is_correct : total_spent = 110 := by
  sorry

end total_cost_is_correct_l177_177662


namespace beta_greater_than_alpha_l177_177556

theorem beta_greater_than_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) (h5 : Real.sin (α + β) = 2 * Real.sin α) : β > α := 
sorry

end beta_greater_than_alpha_l177_177556


namespace monotonic_increasing_interval_l177_177622

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x + 6)

theorem monotonic_increasing_interval : 
  ∀ x y : ℝ, x < y → y < 1 → f x < f y :=
by
  sorry

end monotonic_increasing_interval_l177_177622


namespace regular_dodecahedron_edges_l177_177571

-- Define a regular dodecahedron as a type
inductive RegularDodecahedron : Type
| mk : RegularDodecahedron

-- Define a function that returns the number of edges for a regular dodecahedron
def numberOfEdges (d : RegularDodecahedron) : Nat :=
  30

-- The mathematical statement to be proved
theorem regular_dodecahedron_edges (d : RegularDodecahedron) : numberOfEdges d = 30 := by
  sorry

end regular_dodecahedron_edges_l177_177571


namespace math_problem_l177_177743

theorem math_problem (a b : ℝ) 
  (h1 : a^2 - 3*a*b + 2*b^2 + a - b = 0)
  (h2 : a^2 - 2*a*b + b^2 - 5*a + 7*b = 0) :
  a*b - 12*a + 15*b = 0 :=
by
  sorry

end math_problem_l177_177743


namespace point_coordinates_l177_177853

theorem point_coordinates (m : ℝ) 
  (h1 : dist (0 : ℝ) (Real.sqrt m) = 4) : 
  (-m, Real.sqrt m) = (-16, 4) := 
by
  -- The proof will use the conditions and solve for m to find the coordinates
  sorry

end point_coordinates_l177_177853


namespace cos_120_eq_neg_half_l177_177100

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l177_177100


namespace find_xy_l177_177667

theorem find_xy (x y : ℝ) (π_ne_zero : Real.pi ≠ 0) (h1 : 4 * (x + 2) = 6 * x) (h2 : 6 * x = 2 * Real.pi * y) : x = 4 ∧ y = 12 / Real.pi :=
by
  sorry

end find_xy_l177_177667


namespace minimum_degree_of_g_l177_177711

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := sorry

theorem minimum_degree_of_g :
  (5 * f - 3 * g = h) →
  (Polynomial.degree f = 10) →
  (Polynomial.degree h = 11) →
  (Polynomial.degree g = 11) :=
sorry

end minimum_degree_of_g_l177_177711


namespace distance_to_nearest_town_l177_177524

theorem distance_to_nearest_town (d : ℝ) :
  ¬ (d ≥ 6) → ¬ (d ≤ 5) → ¬ (d ≤ 4) → (d > 5 ∧ d < 6) :=
by
  intro h1 h2 h3
  sorry

end distance_to_nearest_town_l177_177524


namespace complex_ab_value_l177_177311

open Complex

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h : i = Complex.I) (h₁ : (a + b * i) * (3 + i) = 10 + 10 * i) : a * b = 8 := 
by
  sorry

end complex_ab_value_l177_177311


namespace x_y_square_sum_l177_177549

theorem x_y_square_sum (x y : ℝ) (h1 : x - y = -1) (h2 : x * y = 1 / 2) : x^2 + y^2 = 2 := 
by 
  sorry

end x_y_square_sum_l177_177549


namespace find_m_value_l177_177822

noncomputable def m_value : ℝ := -sqrt 3 / 2

theorem find_m_value : 
  let f : ℝ → ℝ  := cos in
  let x1 := π / 2 in
  let x2 := 3 * π / 2 in
  (0 < x1) →
  (x1 < 2 * π) →
  (0 < x2) →
  (x2 < 2 * π) →
  f x1 = 0 →
  f x2 = 0 →
  ∃ x3 x4 : ℝ, x1 < x3 ∧ x3 < x4 ∧ x4 < x2 ∧ 
              2 * x3 = x1 + x4 ∧
              2 * x4 - x3 = x2 →
  ∀ y : ℝ, f y = m_value → y = x3 ∨ y = x4 :=
by
  sorry

end find_m_value_l177_177822


namespace lines_through_same_quadrants_l177_177856

theorem lines_through_same_quadrants (k b : ℝ) (hk : k ≠ 0):
    ∃ n, n ≥ 7 ∧ ∀ (f : Fin n → ℝ × ℝ), ∃ (i j : Fin n), i ≠ j ∧ 
    ((f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0) :=
by sorry

end lines_through_same_quadrants_l177_177856


namespace cos_120_eq_neg_one_half_l177_177078

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l177_177078


namespace prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l177_177020

-- Define the conditions
def total_trips : ℕ := 500
def on_time_A : ℕ := 240
def not_on_time_A : ℕ := 20
def total_A : ℕ := on_time_A + not_on_time_A

def on_time_B : ℕ := 210
def not_on_time_B : ℕ := 30
def total_B : ℕ := on_time_B + not_on_time_B

def total_on_time : ℕ := on_time_A + on_time_B
def total_not_on_time : ℕ := not_on_time_A + not_on_time_B

-- Define the probabilities according to the given solution
def prob_A_on_time : ℚ := on_time_A / total_A
def prob_B_on_time : ℚ := on_time_B / total_B

-- Prove the estimated probabilities
theorem prob_A_correct : prob_A_on_time = 12 / 13 := sorry
theorem prob_B_correct : prob_B_on_time = 7 / 8 := sorry

-- Define the K^2 formula
def K_squared : ℚ :=
  total_trips * (on_time_A * not_on_time_B - on_time_B * not_on_time_A)^2 /
  ((total_A) * (total_B) * (total_on_time) * (total_not_on_time))

-- Prove the provided K^2 value and the conclusion
theorem K_squared_approx_correct (h : K_squared ≈ 3.205) : 3.205 > 2.706 := sorry
theorem punctuality_related_to_company : 3.205 > 2.706 → true := sorry

end prob_A_correct_prob_B_correct_K_squared_approx_correct_punctuality_related_to_company_l177_177020


namespace cos_120_eq_neg_one_half_l177_177077

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l177_177077


namespace contrapositive_equiv_l177_177919

variable (x : Type)

theorem contrapositive_equiv (Q R : x → Prop) :
  (∀ x, Q x → R x) ↔ (∀ x, ¬ (R x) → ¬ (Q x)) :=
by
  sorry

end contrapositive_equiv_l177_177919


namespace solution_set_inequality_l177_177154

theorem solution_set_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x, f x + (deriv^[2] f) x < 1) (h_f0 : f 0 = 2018) :
  ∀ x, x > 0 → f x < 2017 * Real.exp (-x) + 1 :=
by
  sorry

end solution_set_inequality_l177_177154


namespace find_a_l177_177700

theorem find_a (a : ℝ) (i : ℂ) (hi : i = Complex.I) (z : ℂ) (hz : z = a + i) (h : z^2 + z = 1 - 3 * Complex.I) :
  a = -2 :=
by {
  sorry
}

end find_a_l177_177700


namespace insurance_covers_80_percent_l177_177721

-- Definitions from the problem conditions
def cost_per_aid : ℕ := 2500
def num_aids : ℕ := 2
def johns_payment : ℕ := 1000

-- Total cost of hearing aids
def total_cost : ℕ := cost_per_aid * num_aids

-- Insurance payment
def insurance_payment : ℕ := total_cost - johns_payment

-- The theorem to prove
theorem insurance_covers_80_percent :
  (insurance_payment * 100 / total_cost) = 80 :=
by
  sorry

end insurance_covers_80_percent_l177_177721


namespace juan_more_marbles_l177_177397

theorem juan_more_marbles (connie_marbles : ℕ) (juan_marbles : ℕ) (h1 : connie_marbles = 323) (h2 : juan_marbles = 498) :
  juan_marbles - connie_marbles = 175 :=
by
  -- Proof goes here
  sorry

end juan_more_marbles_l177_177397


namespace percentage_error_equals_l177_177780

noncomputable def correct_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7/8 : ℚ) * 8
  let denom := (3/10 : ℚ) - (1/8 : ℚ)
  num / denom

noncomputable def incorrect_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7 / 8 : ℚ) * 8
  num * (3/5 : ℚ)

def percentage_error (correct incorrect : ℚ) : ℚ :=
  abs (correct - incorrect) / correct * 100

theorem percentage_error_equals :
  percentage_error correct_fraction_calc incorrect_fraction_calc = 89.47 :=
by
  sorry

end percentage_error_equals_l177_177780


namespace intersection_P_Q_l177_177447

noncomputable def P : Set ℝ := { x | x < 1 }
noncomputable def Q : Set ℝ := { x | x^2 < 4 }

theorem intersection_P_Q :
  P ∩ Q = { x | -2 < x ∧ x < 1 } :=
by 
  sorry

end intersection_P_Q_l177_177447


namespace tim_final_soda_cans_l177_177003

-- Definitions based on given conditions
def initialSodaCans : ℕ := 22
def cansTakenByJeff : ℕ := 6
def remainingCans (t0 j : ℕ) : ℕ := t0 - j
def additionalCansBought (remaining : ℕ) : ℕ := remaining / 2

-- Function to calculate final number of soda cans
def finalSodaCans (t0 j : ℕ) : ℕ :=
  let remaining := remainingCans t0 j
  remaining + additionalCansBought remaining

-- Theorem to prove the final number of soda cans
theorem tim_final_soda_cans : finalSodaCans initialSodaCans cansTakenByJeff = 24 :=
by
  sorry

end tim_final_soda_cans_l177_177003


namespace find_point_A_l177_177245

-- Define the point -3, 4
def pointP : ℝ × ℝ := (-3, 4)

-- Define the point 0, 2
def pointB : ℝ × ℝ := (0, 2)

-- Define the coordinates of point A
def pointA (x : ℝ) : ℝ × ℝ := (x, 0)

-- The hypothesis using the condition derived from the problem
def ray_reflection_condition (x : ℝ) : Prop :=
  4 / (x + 3) = -2 / x

-- The main theorem we need to prove that the coordinates of point A are (-1, 0)
theorem find_point_A :
  ∃ x : ℝ, ray_reflection_condition x ∧ pointA x = (-1, 0) :=
sorry

end find_point_A_l177_177245


namespace cos_120_eq_neg_half_l177_177093

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l177_177093


namespace determine_remainder_l177_177574

theorem determine_remainder (a b c : ℕ) (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (H1 : (a + 2 * b + 3 * c) % 7 = 1) 
  (H2 : (2 * a + 3 * b + c) % 7 = 2) 
  (H3 : (3 * a + b + 2 * c) % 7 = 1) : 
  (a * b * c) % 7 = 0 := 
sorry

end determine_remainder_l177_177574


namespace class_scores_mean_l177_177461

theorem class_scores_mean 
  (F S : ℕ) (Rf Rs : ℚ)
  (hF : F = 90)
  (hS : S = 75)
  (hRatio : Rf / Rs = 2 / 3) :
  (F * (2/3 * Rs) + S * Rs) / (2/3 * Rs + Rs) = 81 := by
    sorry

end class_scores_mean_l177_177461


namespace triangle_angle_B_l177_177873

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l177_177873


namespace student_range_exact_student_count_l177_177037

-- Definitions for the conditions
def retail_price (x : ℕ) : ℕ := 240
def wholesale_price (x : ℕ) : ℕ := 260 / (x + 60)

def student_conditions (x : ℕ) : Prop := (x < 250) ∧ (x + 60 ≥ 250)
def wholesale_retail_equation (a : ℕ) : Prop := (240^2 / a) * 240 = (260 / (a+60)) * 288

-- Proofs of the required statements
theorem student_range (x : ℕ) (hc : student_conditions x) : 190 ≤ x ∧ x < 250 :=
by {
  sorry
}

theorem exact_student_count (a : ℕ) (heq : wholesale_retail_equation a) : a = 200 :=
by {
  sorry
}

end student_range_exact_student_count_l177_177037


namespace find_angle_B_l177_177878

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l177_177878


namespace find_C_l177_177671

variable (A B C : ℚ)

def condition1 := A + B + C = 350
def condition2 := A + C = 200
def condition3 := B + C = 350

theorem find_C : condition1 A B C → condition2 A C → condition3 B C → C = 200 :=
by
  sorry

end find_C_l177_177671


namespace cos_120_degrees_eq_l177_177112

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l177_177112


namespace remainder_div_x_plus_1_l177_177222

noncomputable def polynomial1 : Polynomial ℝ := Polynomial.X ^ 11 - 1

theorem remainder_div_x_plus_1 :
  Polynomial.eval (-1) polynomial1 = -2 := by
  sorry

end remainder_div_x_plus_1_l177_177222


namespace angle_B_in_triangle_l177_177866

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l177_177866


namespace jindra_initial_dice_count_l177_177590

-- Given conditions about the dice stacking
def number_of_dice_per_layer : ℕ := 36
def layers_stacked_completely : ℕ := 6
def dice_received : ℕ := 18

-- We need to prove that the initial number of dice Jindra had is 234
theorem jindra_initial_dice_count : 
    (layers_stacked_completely * number_of_dice_per_layer + dice_received) = 234 :=
    by 
        sorry

end jindra_initial_dice_count_l177_177590


namespace fraction_multiplication_l177_177494

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l177_177494


namespace maximal_x2009_l177_177227

theorem maximal_x2009 (x : ℕ → ℝ) 
    (h_seq : ∀ n, x n - 2 * x (n + 1) + x (n + 2) ≤ 0)
    (h_x0 : x 0 = 1)
    (h_x20 : x 20 = 9)
    (h_x200 : x 200 = 6) :
    x 2009 ≤ 6 :=
sorry

end maximal_x2009_l177_177227


namespace solve_for_x_l177_177221

noncomputable def is_satisfied (x : ℝ) : Prop :=
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 2

theorem solve_for_x :
  ∀ x : ℝ, 0 < x → x ≠ 1 ↔ is_satisfied x := by
  sorry

end solve_for_x_l177_177221


namespace inscribed_sphere_tetrahedron_volume_l177_177715

theorem inscribed_sphere_tetrahedron_volume
  (R : ℝ) (S1 S2 S3 S4 : ℝ) :
  ∃ V : ℝ, V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end inscribed_sphere_tetrahedron_volume_l177_177715


namespace gain_percentage_l177_177390

theorem gain_percentage (selling_price gain : ℕ) (h_sp : selling_price = 110) (h_gain : gain = 10) :
  (gain * 100) / (selling_price - gain) = 10 :=
by
  sorry

end gain_percentage_l177_177390


namespace cos_120_eq_neg_half_l177_177102

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l177_177102


namespace expression_meaningful_l177_177217

theorem expression_meaningful (x : ℝ) : 
  (x - 1 ≠ 0 ∧ true) ↔ x ≠ 1 := 
sorry

end expression_meaningful_l177_177217


namespace cos_120_eq_neg_half_l177_177103

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l177_177103


namespace inequality_proof_l177_177424

variable (a b : ℝ)

theorem inequality_proof (h1 : -1 < b) (h2 : b < 0) (h3 : a < 0) : 
  (a * b > a * b^2) ∧ (a * b^2 > a) := 
by
  sorry

end inequality_proof_l177_177424


namespace equality_of_areas_l177_177316

theorem equality_of_areas (d : ℝ) :
  (∀ d : ℝ, (1/2) * d * 3 = 9 / 2 → d = 3) ↔ d = 3 :=
by
  sorry

end equality_of_areas_l177_177316


namespace number_line_steps_l177_177317

theorem number_line_steps (n : ℕ) (total_distance : ℕ) (steps_to_x : ℕ) (x : ℕ)
  (h1 : total_distance = 32)
  (h2 : n = 8)
  (h3 : steps_to_x = 6)
  (h4 : x = (total_distance / n) * steps_to_x) :
  x = 24 := 
sorry

end number_line_steps_l177_177317


namespace unique_line_equation_l177_177351

theorem unique_line_equation
  (k : ℝ)
  (m b : ℝ)
  (h1 : |(k^2 + 4*k + 3) - (m*k + b)| = 4)
  (h2 : 2*m + b = 8)
  (h3 : b ≠ 0) :
  (m = 6 ∧ b = -4) :=
by
  sorry

end unique_line_equation_l177_177351


namespace marble_group_l177_177237

theorem marble_group (x : ℕ) (h1 : 144 % x = 0) (h2 : 144 % (x + 2) = (144 / x) - 1) : x = 16 :=
sorry

end marble_group_l177_177237


namespace eight_digit_number_divisibility_l177_177448

theorem eight_digit_number_divisibility (a b c d : ℕ) (Z : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
(h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : d ≤ 9) (hZ : Z = 1001 * (1000 * a + 100 * b + 10 * c + d)) : 
  10001 ∣ Z := 
  by sorry

end eight_digit_number_divisibility_l177_177448


namespace range_of_m_l177_177146

theorem range_of_m {x m : ℝ} 
  (α : 2 / (x + 1) > 1) 
  (β : m ≤ x ∧ x ≤ 2) 
  (suff_condition : ∀ x, (2 / (x + 1) > 1) → (m ≤ x ∧ x ≤ 2)) :
  m ≤ -1 :=
sorry

end range_of_m_l177_177146


namespace add_100ml_water_l177_177234

theorem add_100ml_water 
    (current_volume : ℕ) 
    (current_water_percentage : ℝ) 
    (desired_water_percentage : ℝ) 
    (current_water_volume : ℝ) 
    (x : ℝ) :
    current_volume = 300 →
    current_water_percentage = 0.60 →
    desired_water_percentage = 0.70 →
    current_water_volume = 0.60 * 300 →
    180 + x = 0.70 * (300 + x) →
    x = 100 := 
sorry

end add_100ml_water_l177_177234


namespace find_radius_of_tangent_circles_l177_177487

noncomputable def radius_of_tangent_circles : ℝ :=
  let ellipse_eq : ℝ → ℝ → Prop := λ x y, x^2 + 6 * y^2 = 8
  let circle_eq : ℝ → ℝ → ℝ → Prop := λ r x y, (x - r)^2 + y^2 = r^2
  let discriminant_zero : ℝ → Prop := λ r, (12 * r)^2 - 4 * 5 * 8 = 0
  if h : discriminant_zero ((√10) / 3) then (√10) / 3 else 0

theorem find_radius_of_tangent_circles :
  ∃ r : ℝ, (r = radius_of_tangent_circles) ∧ r = ((√10) / 3) :=
begin
  use ((√10) / 3),
  split,
  { unfold radius_of_tangent_circles,
    simp,
    norm_num,
    exact if_pos (by norm_num : discriminant_zero ((√10) / 3)) },
  { refl }
end

end find_radius_of_tangent_circles_l177_177487


namespace number_of_distinct_arrangements_l177_177790

-- Given conditions: There are 7 items and we need to choose 4 out of these 7.
def binomial_coefficient (n k : ℕ) : ℕ :=
  (n.choose k)

-- Given condition: Calculate the number of sequences of arranging 4 selected items.
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- The statement in Lean 4 to prove that the number of distinct arrangements is 840.
theorem number_of_distinct_arrangements : binomial_coefficient 7 4 * factorial 4 = 840 :=
by
  sorry

end number_of_distinct_arrangements_l177_177790


namespace tractor_brigades_l177_177200
noncomputable def brigade_plowing : Prop :=
∃ x y : ℝ,
  x * y = 240 ∧
  (x + 3) * (y + 2) = 324 ∧
  x > 20 ∧
  (x + 3) > 20 ∧
  x = 24 ∧
  (x + 3) = 27

theorem tractor_brigades:
  brigade_plowing :=
sorry

end tractor_brigades_l177_177200


namespace price_for_70_cans_is_correct_l177_177368

def regular_price_per_can : ℝ := 0.55
def discount_percentage : ℝ := 0.25
def purchase_quantity : ℕ := 70

def discount_per_can : ℝ := discount_percentage * regular_price_per_can
def discounted_price_per_can : ℝ := regular_price_per_can - discount_per_can

def price_for_72_cans : ℝ := 72 * discounted_price_per_can
def price_for_2_cans : ℝ := 2 * discounted_price_per_can

def final_price_for_70_cans : ℝ := price_for_72_cans - price_for_2_cans

theorem price_for_70_cans_is_correct
    (regular_price_per_can : ℝ := 0.55)
    (discount_percentage : ℝ := 0.25)
    (purchase_quantity : ℕ := 70)
    (disc_per_can : ℝ := discount_percentage * regular_price_per_can)
    (disc_price_per_can : ℝ := regular_price_per_can - disc_per_can)
    (price_72_cans : ℝ := 72 * disc_price_per_can)
    (price_2_cans : ℝ := 2 * disc_price_per_can):
    final_price_for_70_cans = 28.875 :=
by
  sorry

end price_for_70_cans_is_correct_l177_177368


namespace MN_squared_l177_177173

theorem MN_squared (PQ QR RS SP : ℝ) (h1 : PQ = 15) (h2 : QR = 15) (h3 : RS = 20) (h4 : SP = 20) (angle_S : ℝ) (h5 : angle_S = 90)
(M N: ℝ) (Midpoint_M : M = (QR / 2)) (Midpoint_N : N = (SP / 2)) : 
MN^2 = 100 := by
  sorry

end MN_squared_l177_177173


namespace sum_of_divisors_45_l177_177686

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun i => n % i = 0) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_45 : sum_of_divisors 45 = 78 := 
  sorry

end sum_of_divisors_45_l177_177686


namespace original_price_of_tshirt_l177_177750

theorem original_price_of_tshirt :
  ∀ (P : ℝ), 
    (∀ discount quantity_sold revenue : ℝ, discount = 8 ∧ quantity_sold = 130 ∧ revenue = 5590 ∧
      revenue = quantity_sold * (P - discount)) → P = 51 := 
by
  intros P
  intro h
  sorry

end original_price_of_tshirt_l177_177750


namespace soccer_league_equation_l177_177294

noncomputable def equation_represents_soccer_league (x : ℕ) : Prop :=
  ∀ x : ℕ, (x * (x - 1)) / 2 = 50

theorem soccer_league_equation (x : ℕ) (h : equation_represents_soccer_league x) :
  (x * (x - 1)) / 2 = 50 :=
  by sorry

end soccer_league_equation_l177_177294


namespace customers_added_l177_177253

theorem customers_added (x : ℕ) (h : 29 + x = 49) : x = 20 := by
  sorry

end customers_added_l177_177253


namespace f_divides_f_2k_plus_1_f_coprime_f_multiple_l177_177287

noncomputable def f (g n : ℕ) : ℕ := g ^ n + 1

theorem f_divides_f_2k_plus_1 (g : ℕ) (k n : ℕ) :
  f g n ∣ f g ((2 * k + 1) * n) :=
by sorry

theorem f_coprime_f_multiple (g n : ℕ) :
  Nat.Coprime (f g n) (f g (2 * n)) ∧
  Nat.Coprime (f g n) (f g (4 * n)) ∧
  Nat.Coprime (f g n) (f g (6 * n)) :=
by sorry

end f_divides_f_2k_plus_1_f_coprime_f_multiple_l177_177287


namespace interval_contains_root_l177_177751

theorem interval_contains_root :
  (∃ c, (0 < c ∧ c < 1) ∧ (2^c + c - 2 = 0) ∧ 
        (∀ x1 x2, x1 < x2 → 2^x1 + x1 - 2 < 2^x2 + x2 - 2) ∧ 
        (0 < 1) ∧ 
        ((2^0 + 0 - 2) = -1) ∧ 
        ((2^1 + 1 - 2) = 1)) := 
by 
  sorry

end interval_contains_root_l177_177751


namespace gym_guest_count_l177_177065

theorem gym_guest_count (G : ℕ) (H1 : ∀ G, 0 < G → ∀ G, G * 5.7 = 285 ∧ G = 50) : G = 50 :=
by
  sorry

end gym_guest_count_l177_177065


namespace solve_abs_eq_linear_l177_177744

theorem solve_abs_eq_linear (x : ℝ) (h : |2 * x - 4| = x + 3) : x = 7 :=
sorry

end solve_abs_eq_linear_l177_177744


namespace remainder_x150_l177_177982

theorem remainder_x150 (x : ℝ) : 
  ∃ r : ℝ, ∃ q : ℝ, x^150 = q * (x - 1)^3 + 11175*x^2 - 22200*x + 11026 := 
by
  sorry

end remainder_x150_l177_177982


namespace ratio_sub_div_eq_l177_177143

theorem ratio_sub_div_eq 
  (a b : ℚ) 
  (h : a / b = 5 / 2) : 
  (a - b) / a = 3 / 5 := 
sorry

end ratio_sub_div_eq_l177_177143


namespace parallelogram_area_l177_177760

theorem parallelogram_area (b h : ℝ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end parallelogram_area_l177_177760


namespace morgan_first_sat_score_l177_177459

theorem morgan_first_sat_score (x : ℝ) (h : 1.10 * x = 1100) : x = 1000 :=
sorry

end morgan_first_sat_score_l177_177459


namespace isosceles_triangle_perimeter_l177_177579

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

end isosceles_triangle_perimeter_l177_177579


namespace system_of_equations_solution_l177_177482

theorem system_of_equations_solution :
  ∃ (a b : ℤ), (2 * (2 : ℤ) + b = a ∧ (2 : ℤ) + b = 3 ∧ a = 5 ∧ b = 1) :=
by
  sorry

end system_of_equations_solution_l177_177482


namespace cos_120_eq_neg_one_half_l177_177073

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l177_177073


namespace arithmetic_progression_sum_15_terms_l177_177986

theorem arithmetic_progression_sum_15_terms (a₃ a₅ : ℝ) (h₁ : a₃ = -5) (h₂ : a₅ = 2.4) : 
  let d := (a₅ - a₃) / 2 in
  let a₁ := a₃ - 2 * d in
  (15 / 2) * (2 * a₁ + 14 * d) = 202.5 :=
by
  sorry

end arithmetic_progression_sum_15_terms_l177_177986


namespace sandy_paint_area_l177_177608

-- Define the dimensions of the wall
def wall_height : ℕ := 10
def wall_length : ℕ := 15

-- Define the dimensions of the decorative region
def deco_height : ℕ := 3
def deco_length : ℕ := 5

-- Calculate the areas and prove the required area to paint
theorem sandy_paint_area :
  wall_height * wall_length - deco_height * deco_length = 135 := by
  sorry

end sandy_paint_area_l177_177608


namespace geometric_sequence_fifth_term_l177_177754

variables (a r : ℝ) (h1 : a * r ^ 2 = 12 / 5) (h2 : a * r ^ 6 = 48)

theorem geometric_sequence_fifth_term : a * r ^ 4 = 12 / 5 := by
  sorry

end geometric_sequence_fifth_term_l177_177754


namespace problem_solution_l177_177575
open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.foldl (· + ·) 0

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_seq : ℕ → ℕ → ℕ
| 0, n => f n
| (k+1), n => f (f_seq k n)

theorem problem_solution :
  f_seq 2016 9 = 8 :=
sorry

end problem_solution_l177_177575


namespace not_enough_evidence_to_show_relationship_l177_177636

noncomputable def isEvidenceToShowRelationship (table : Array (Array Nat)) : Prop :=
  ∃ evidence : Bool, ¬evidence

theorem not_enough_evidence_to_show_relationship :
  isEvidenceToShowRelationship #[#[5, 15, 20], #[40, 10, 50], #[45, 25, 70]] :=
sorry 

end not_enough_evidence_to_show_relationship_l177_177636


namespace find_sum_of_relatively_prime_integers_l177_177342

theorem find_sum_of_relatively_prime_integers :
  ∃ (x y : ℕ), x * y + x + y = 119 ∧ x < 25 ∧ y < 25 ∧ Nat.gcd x y = 1 ∧ x + y = 20 :=
by
  sorry

end find_sum_of_relatively_prime_integers_l177_177342


namespace problem_1a_problem_1b_l177_177158

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log x - x

theorem problem_1a (a : ℝ) : 
  a = 1 → (0 < x ∧ x < 1 → HasDerivAt (f a) (1 / x - 1) x ∧ (f a)' x > 0) ∧ 
          (x > 1 → HasDerivAt (f a) (1 / x - 1) x ∧ (f a)' x < 0) := 
by
  sorry

theorem problem_1b (a : ℝ) (x : ℝ) (h : 0 < x ∧ x ≤ 2) : 
  a > 0 → (
    (a < 2 → ∃ c ∈ (0, -- sorry
end 끝
 1), HasDerivAt (f a) (1 / x - 1) c ∧ (f a)' c > 0 ∧ ∀ b ∈ (c, 2], HasDerivAt (f a) (1 / x - 1) b ∧ (f a)' b < 0 ∧ f a x ≤ f a a) ∨
    (a ≥ 2 → ∀ d ∈ (0, 2], HasDerivAt (f a) (1 / x - 1) d ∧ (f a)' d > 0 ∧ f a x ≤ f a 2) 
  ∧ (
    (a < 2 → f a (a) = a * Real.log a - a) ∨ 
    (a ≥ 2 → f a (2) = 2 * Real.log 2 - 2)
) := 
by
  sorry

end problem_1a_problem_1b_l177_177158


namespace top_layer_blocks_l177_177236

theorem top_layer_blocks (x : Nat) (h : x + 3 * x + 9 * x + 27 * x = 40) : x = 1 :=
by
  sorry

end top_layer_blocks_l177_177236


namespace option_B_proof_option_C_proof_l177_177731

-- Definitions and sequences
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Statement of the problem

theorem option_B_proof (A B : ℝ) :
  (∀ n : ℕ, S n = A * (n : ℝ)^2 + B * n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d := 
sorry

theorem option_C_proof :
  (∀ n : ℕ, S n = 1 - (-1)^n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end option_B_proof_option_C_proof_l177_177731


namespace football_game_spectators_l177_177293

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ) (h1 : total_wristbands = 234) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 117 := by
  sorry

end football_game_spectators_l177_177293


namespace total_ceilings_to_paint_l177_177458

theorem total_ceilings_to_paint (ceilings_painted_this_week : ℕ) 
                                (ceilings_painted_next_week : ℕ)
                                (ceilings_left_to_paint : ℕ) 
                                (h1 : ceilings_painted_this_week = 12) 
                                (h2 : ceilings_painted_next_week = ceilings_painted_this_week / 4) 
                                (h3 : ceilings_left_to_paint = 13) : 
    ceilings_painted_this_week + ceilings_painted_next_week + ceilings_left_to_paint = 28 :=
by
  sorry

end total_ceilings_to_paint_l177_177458


namespace cos_120_eq_neg_half_l177_177087

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l177_177087


namespace infinite_series_sum_l177_177677

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) * (n + 1) + 2 * (n + 1) + 1) / ((n + 1) * (n + 2) * (n + 3) * (n + 4))) 
  = 7 / 6 := 
by
  sorry

end infinite_series_sum_l177_177677


namespace f_45_g_10_l177_177478

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_condition1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom g_condition2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x + y) = g x + g y
axiom f_15 : f 15 = 10
axiom g_5 : g 5 = 3

theorem f_45 : f 45 = 10 / 3 := sorry
theorem g_10 : g 10 = 6 := sorry

end f_45_g_10_l177_177478


namespace find_x_solution_l177_177837

theorem find_x_solution
  (x y z : ℤ)
  (h1 : 4 * x + y + z = 80)
  (h2 : 2 * x - y - z = 40)
  (h3 : 3 * x + y - z = 20) :
  x = 20 :=
by
  -- Proof steps go here...
  sorry

end find_x_solution_l177_177837


namespace smallest_positive_x_l177_177274

theorem smallest_positive_x (x : ℕ) (h : 42 * x + 9 ≡ 3 [MOD 15]) : x = 2 :=
sorry

end smallest_positive_x_l177_177274


namespace DE_value_l177_177438

theorem DE_value {AG GF FC HJ DE : ℝ} (h1 : AG = 2) (h2 : GF = 13) 
  (h3 : FC = 1) (h4 : HJ = 7) : DE = 2 * Real.sqrt 22 :=
sorry

end DE_value_l177_177438


namespace f_of_2_l177_177232

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_value : f (-2) = 11)

-- The theorem we want to prove
theorem f_of_2 : f 2 = -11 :=
by 
  sorry

end f_of_2_l177_177232


namespace cos_120_eq_neg_half_l177_177095

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l177_177095


namespace find_m_l177_177161

theorem find_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 3, 2*m - 1}) (hB: B = {3, m^2}) (h_subset: B ⊆ A) : m = 1 :=
by
  sorry

end find_m_l177_177161


namespace problem_statement_l177_177060

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem problem_statement :
  ¬ is_pythagorean_triple 2 3 4 ∧ 
  is_pythagorean_triple 3 4 5 ∧ 
  is_pythagorean_triple 6 8 10 ∧ 
  is_pythagorean_triple 5 12 13 :=
by 
  constructor
  sorry
  constructor
  sorry
  constructor
  sorry
  sorry

end problem_statement_l177_177060


namespace missing_coin_value_l177_177603

-- Definitions based on the conditions
def value_of_dime := 10 -- Value of 1 dime in cents
def value_of_nickel := 5 -- Value of 1 nickel in cents
def num_dimes := 1
def num_nickels := 2
def total_value_found := 45 -- Total value found in cents

-- Statement to prove the missing coin's value
theorem missing_coin_value : 
  (total_value_found - (num_dimes * value_of_dime + num_nickels * value_of_nickel)) = 25 := 
by
  sorry

end missing_coin_value_l177_177603


namespace fraction_value_l177_177559

theorem fraction_value
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (cond1 : (a + b + c) / (a + b - c) = 7)
  (cond2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 :=
by
  sorry

end fraction_value_l177_177559


namespace min_value_arith_seq_l177_177529

noncomputable def S_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem min_value_arith_seq : ∀ n : ℕ, n > 0 → 2 * S_n 2 = (n + 1) * 2 → (n = 4 → (2 * S_n n + 13) / n = 33 / 4) :=
by
  intros n hn hS2 hn_eq_4
  sorry

end min_value_arith_seq_l177_177529


namespace odd_n_cubed_plus_23n_divisibility_l177_177648

theorem odd_n_cubed_plus_23n_divisibility (n : ℤ) (h1 : n % 2 = 1) : (n^3 + 23 * n) % 24 = 0 := 
by 
  sorry

end odd_n_cubed_plus_23n_divisibility_l177_177648


namespace exponent_multiplication_l177_177394

theorem exponent_multiplication :
  (-1 / 2 : ℝ) ^ 2022 * (2 : ℝ) ^ 2023 = 2 :=
by sorry

end exponent_multiplication_l177_177394


namespace min_vertical_segment_length_l177_177617

noncomputable def minVerticalSegLength : ℤ → ℝ 
| x => abs (2 * abs x + x^2 + 4 * x + 1)

theorem min_vertical_segment_length :
  ∀ x : ℤ, minVerticalSegLength x = 1 ↔  x = 0 := 
by
  intros x
  sorry

end min_vertical_segment_length_l177_177617


namespace transform_equation_l177_177836

theorem transform_equation (x y : ℝ) (h : y = x + x⁻¹) :
  x^4 + x^3 - 5 * x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 7) = 0 := 
sorry

end transform_equation_l177_177836


namespace cosine_120_eq_negative_half_l177_177119

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l177_177119


namespace max_rocket_height_l177_177521

-- Define the quadratic function representing the rocket's height
def rocket_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

-- State the maximum height problem
theorem max_rocket_height : ∃ t : ℝ, rocket_height t = 175 ∧ ∀ t' : ℝ, rocket_height t' ≤ 175 :=
by
  use 2.5
  sorry -- The proof will show that the maximum height is 175 meters at time t = 2.5 seconds

end max_rocket_height_l177_177521


namespace inequality_must_hold_l177_177428

theorem inequality_must_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by {
  sorry
}

end inequality_must_hold_l177_177428


namespace mutter_paid_correct_amount_l177_177893

def total_lagaan_collected : ℝ := 344000
def mutter_land_percentage : ℝ := 0.0023255813953488372
def mutter_lagaan_paid : ℝ := 800

theorem mutter_paid_correct_amount : 
  mutter_lagaan_paid = total_lagaan_collected * mutter_land_percentage := by
  sorry

end mutter_paid_correct_amount_l177_177893


namespace fraction_product_l177_177501

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l177_177501


namespace limit_calculation_l177_177391

theorem limit_calculation :
  tendsto (λ n : ℕ, (2 * n^2 - 1 : ℝ) / (n^2 + n + 1)) at_top (𝓝 2) :=
by { sorry }

end limit_calculation_l177_177391


namespace range_F_l177_177975

-- Define the function and its critical points
def F (x : ℝ) : ℝ := |2 * x + 4| - |x - 2|

theorem range_F : ∀ y : ℝ, y ∈ Set.range F ↔ -4 ≤ y := by
  sorry

end range_F_l177_177975


namespace isosceles_triangle_perimeter_l177_177531

theorem isosceles_triangle_perimeter (a b : ℕ) (h₀ : a = 3 ∨ a = 4) (h₁ : b = 3 ∨ b = 4) (h₂ : a ≠ b) :
  (a = 3 ∧ b = 4 ∧ 4 ∈ [b]) ∨ (a = 4 ∧ b = 3 ∧ 4 ∈ [a]) → 
  (a + a + b = 10) ∨ (a + b + b = 11) :=
by
  sorry

end isosceles_triangle_perimeter_l177_177531


namespace percentage_passed_both_subjects_l177_177296

def failed_H : ℝ := 0.35
def failed_E : ℝ := 0.45
def failed_HE : ℝ := 0.20

theorem percentage_passed_both_subjects :
  (100 - (failed_H * 100 + failed_E * 100 - failed_HE * 100)) = 40 := 
by
  sorry

end percentage_passed_both_subjects_l177_177296


namespace min_distance_sum_l177_177989

open Real EuclideanGeometry

-- Define the parabola y^2 = 4x
noncomputable def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1 

-- Define the fixed point M
def M : ℝ × ℝ := (2, 3)

-- Define the line l: x = -1
def line_l (P : ℝ × ℝ) : ℝ := abs (P.1 + 1)

-- Define the distance from point P to point M
def distance_to_M (P : ℝ × ℝ) : ℝ := dist P M

-- Define the distance from point P to line l
def distance_to_line (P : ℝ × ℝ) := line_l P 

-- Define the sum of distances
def sum_of_distances (P : ℝ × ℝ) : ℝ := distance_to_M P + distance_to_line P

-- Prove the minimum value of the sum of distances
theorem min_distance_sum : ∃ P, parabola P ∧ sum_of_distances P = sqrt 10 := sorry

end min_distance_sum_l177_177989


namespace interval_length_l177_177340

theorem interval_length (c d : ℝ) (h : (d - 5) / 3 - (c - 5) / 3 = 15) : d - c = 45 :=
sorry

end interval_length_l177_177340
