import Mathlib

namespace solve_system_l1346_134645

open Classical

theorem solve_system : ∃ t : ℝ, ∀ (x y z : ℝ), 
  (x^2 - 9 * y^2 = 0 ∧ x + y + z = 0) ↔ 
  (x = 3 * t ∧ y = t ∧ z = -4 * t) 
  ∨ (x = -3 * t ∧ y = t ∧ z = 2 * t) := 
by 
  sorry

end solve_system_l1346_134645


namespace james_gave_away_one_bag_l1346_134644

theorem james_gave_away_one_bag (initial_marbles : ℕ) (bags : ℕ) (marbles_left : ℕ) (h1 : initial_marbles = 28) (h2 : bags = 4) (h3 : marbles_left = 21) : (initial_marbles / bags) = (initial_marbles - marbles_left) / (initial_marbles / bags) :=
by
  sorry

end james_gave_away_one_bag_l1346_134644


namespace sin_seven_pi_over_six_l1346_134664

theorem sin_seven_pi_over_six :
  Real.sin (7 * Real.pi / 6) = - 1 / 2 :=
by
  sorry

end sin_seven_pi_over_six_l1346_134664


namespace exterior_angle_BAC_eq_162_l1346_134629

noncomputable def measure_of_angle_BAC : ℝ := 360 - 108 - 90

theorem exterior_angle_BAC_eq_162 :
  measure_of_angle_BAC = 162 := by
  sorry

end exterior_angle_BAC_eq_162_l1346_134629


namespace initial_percent_l1346_134662

theorem initial_percent (x : ℝ) :
  (x / 100) * (5 / 100) = 60 / 100 → x = 1200 := 
by 
  sorry

end initial_percent_l1346_134662


namespace ron_spending_increase_l1346_134697

variable (P Q : ℝ) -- initial price and quantity
variable (X : ℝ)   -- intended percentage increase in spending

theorem ron_spending_increase :
  (1 + X / 100) * P * Q = 1.25 * P * (0.92 * Q) →
  X = 15 := 
by
  sorry

end ron_spending_increase_l1346_134697


namespace quad_to_square_l1346_134613

theorem quad_to_square (a b z : ℝ)
  (h_dim : a = 9) 
  (h_dim2 : b = 16) 
  (h_area : a * b = z * z) :
  z = 12 :=
by
  -- Proof outline would go here, but let's skip the actual proof for this definition.
  sorry

end quad_to_square_l1346_134613


namespace base_price_lowered_percentage_l1346_134682

theorem base_price_lowered_percentage (P : ℝ) (new_price final_price : ℝ) (x : ℝ)
    (h1 : new_price = P - (x / 100) * P)
    (h2 : final_price = 0.9 * new_price)
    (h3 : final_price = P - (14.5 / 100) * P) :
    x = 5 :=
  sorry

end base_price_lowered_percentage_l1346_134682


namespace boys_in_class_l1346_134615

theorem boys_in_class (total_students : ℕ) (fraction_girls : ℝ) (fraction_girls_eq : fraction_girls = 1 / 4) (total_students_eq : total_students = 160) :
  (total_students - fraction_girls * total_students = 120) :=
by
  rw [fraction_girls_eq, total_students_eq]
  -- Here, additional lines proving the steps would follow, but we use sorry for completeness.
  sorry

end boys_in_class_l1346_134615


namespace intersection_eq_l1346_134637

noncomputable def A : Set ℕ := {1, 2, 3, 4}
noncomputable def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_eq : A ∩ B = {2, 3, 4} := 
by
  sorry

end intersection_eq_l1346_134637


namespace find_b_l1346_134654

noncomputable def angle_B : ℝ := 60
noncomputable def c : ℝ := 8
noncomputable def diff_b_a (b a : ℝ) : Prop := b - a = 4

theorem find_b (b a : ℝ) (h₁ : angle_B = 60) (h₂ : c = 8) (h₃ : diff_b_a b a) :
  b = 7 :=
sorry

end find_b_l1346_134654


namespace least_number_condition_l1346_134624

-- Define the set of divisors as a constant
def divisors : Set ℕ := {1, 2, 3, 4, 5, 6, 8, 15}

-- Define the least number that satisfies the condition
def least_number : ℕ := 125

-- The theorem stating that the least number 125 leaves a remainder of 5 when divided by the given set of numbers
theorem least_number_condition : ∀ d ∈ divisors, least_number % d = 5 :=
by
  sorry

end least_number_condition_l1346_134624


namespace remaining_download_time_l1346_134684

-- Define the relevant quantities
def total_size : ℝ := 1250
def downloaded : ℝ := 310
def download_speed : ℝ := 2.5

-- State the theorem
theorem remaining_download_time : (total_size - downloaded) / download_speed = 376 := by
  -- Proof will be filled in here
  sorry

end remaining_download_time_l1346_134684


namespace student_weight_loss_l1346_134604

variables (S R L : ℕ)

theorem student_weight_loss :
  S = 75 ∧ S + R = 110 ∧ S - L = 2 * R → L = 5 :=
by
  sorry

end student_weight_loss_l1346_134604


namespace find_m_from_power_function_l1346_134687

theorem find_m_from_power_function :
  (∃ a : ℝ, (2 : ℝ) ^ a = (Real.sqrt 2) / 2) →
  (∃ m : ℝ, (m : ℝ) ^ (-1 / 2 : ℝ) = 2) →
  ∃ m : ℝ, m = 1 / 4 :=
by
  intro h1 h2
  sorry

end find_m_from_power_function_l1346_134687


namespace sum_of_mixed_numbers_is_between_18_and_19_l1346_134602

theorem sum_of_mixed_numbers_is_between_18_and_19 :
  let a := 2 + 3 / 8;
  let b := 4 + 1 / 3;
  let c := 5 + 2 / 21;
  let d := 6 + 1 / 11;
  18 < a + b + c + d ∧ a + b + c + d < 19 :=
by
  sorry

end sum_of_mixed_numbers_is_between_18_and_19_l1346_134602


namespace exists_abc_l1346_134611

theorem exists_abc (n k : ℕ) (hn : n > 20) (hk : k > 1) (hdiv : k^2 ∣ n) : 
  ∃ (a b c : ℕ), n = a * b + b * c + c * a :=
by
  sorry

end exists_abc_l1346_134611


namespace alice_pints_wednesday_l1346_134601

-- Initial conditions
def pints_sunday : ℕ := 4
def pints_monday : ℕ := 3 * pints_sunday
def pints_tuesday : ℕ := pints_monday / 3
def total_pints_before_return : ℕ := pints_sunday + pints_monday + pints_tuesday
def pints_returned_wednesday : ℕ := pints_tuesday / 2
def pints_wednesday : ℕ := total_pints_before_return - pints_returned_wednesday

-- The proof statement
theorem alice_pints_wednesday : pints_wednesday = 18 :=
by
  sorry

end alice_pints_wednesday_l1346_134601


namespace find_first_term_of_geometric_progression_l1346_134676

theorem find_first_term_of_geometric_progression
  (a_2 : ℝ) (a_3 : ℝ) (a_1 : ℝ) (q : ℝ)
  (h1 : a_2 = a_1 * q)
  (h2 : a_3 = a_1 * q^2)
  (h3 : a_2 = 5)
  (h4 : a_3 = 1) : a_1 = 25 :=
by
  sorry

end find_first_term_of_geometric_progression_l1346_134676


namespace complex_magnitude_l1346_134619

theorem complex_magnitude (z : ℂ) (i_unit : ℂ := Complex.I) 
  (h : (z - i_unit) * i_unit = 2 + i_unit) : Complex.abs z = Real.sqrt 5 := 
by
  sorry

end complex_magnitude_l1346_134619


namespace isosceles_triangle_perimeter_l1346_134671

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6) (h2 : b = 13) 
  (triangle_inequality : b + b > a) : 
  (2 * b + a) = 32 := by
  sorry

end isosceles_triangle_perimeter_l1346_134671


namespace proof_problem_l1346_134636

theorem proof_problem (p q : Prop) : (p ∧ q) ↔ ¬ (¬ p ∨ ¬ q) :=
sorry

end proof_problem_l1346_134636


namespace total_sodas_bought_l1346_134677

-- Condition 1: Number of sodas they drank
def sodas_drank : ℕ := 3

-- Condition 2: Number of extra sodas Robin had
def sodas_extras : ℕ := 8

-- Mathematical equivalence we want to prove: Total number of sodas bought by Robin
theorem total_sodas_bought : sodas_drank + sodas_extras = 11 := by
  sorry

end total_sodas_bought_l1346_134677


namespace find_y_l1346_134628

def angle_at_W (RWQ RWT QWR TWQ : ℝ) :=  RWQ + RWT + QWR + TWQ

theorem find_y 
  (RWQ RWT QWR TWQ : ℝ)
  (h1 : RWQ = 90) 
  (h2 : RWT = 3 * y)
  (h3 : QWR = y)
  (h4 : TWQ = 90) 
  (h_sum : angle_at_W RWQ RWT QWR TWQ = 360)  
  : y = 67.5 :=
by
  sorry

end find_y_l1346_134628


namespace parabola_directrix_l1346_134605

theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) (O D : ℝ × ℝ) :
  A ≠ B →
  O = (0, 0) →
  D = (1, 2) →
  (∃ k, k = ((2:ℝ) - 0) / ((1:ℝ) - 0) ∧ k = 2) →
  (∃ k, k = - 1 / 2) →
  (∀ x y, y^2 = 2 * p * x) →
  p = 5 / 2 →
  O.1 * A.1 + O.2 * A.2 = 0 →
  O.1 * B.1 + O.2 * B.2 = 0 →
  A.1 * B.1 + A.2 * B.2 = 0 →
  (∃ k, (y - 2) = k * (x - 1) ∧ (A.1 * B.1) = 25 ∧ (A.1 + B.1) = 10 + 8 * p) →
  ∃ dir_eq, dir_eq = -5 / 4 :=
by
  sorry

end parabola_directrix_l1346_134605


namespace andre_flowers_given_l1346_134638

variable (initialFlowers totalFlowers flowersGiven : ℕ)

theorem andre_flowers_given (h1 : initialFlowers = 67) (h2 : totalFlowers = 90) :
  flowersGiven = totalFlowers - initialFlowers → flowersGiven = 23 :=
by
  intro h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end andre_flowers_given_l1346_134638


namespace roots_fourth_pow_sum_l1346_134675

theorem roots_fourth_pow_sum :
  (∃ p q r : ℂ, (∀ z, (z = p ∨ z = q ∨ z = r) ↔ z^3 - z^2 + 2*z - 3 = 0) ∧ p^4 + q^4 + r^4 = 13) := by
sorry

end roots_fourth_pow_sum_l1346_134675


namespace sin_330_eq_neg_half_l1346_134618

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by sorry

end sin_330_eq_neg_half_l1346_134618


namespace simplify_abs_value_l1346_134661

theorem simplify_abs_value : abs (- 5 ^ 2 + 6) = 19 := by
  sorry

end simplify_abs_value_l1346_134661


namespace find_monic_cubic_polynomial_with_root_l1346_134699

-- Define the monic cubic polynomial
def Q (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 6

-- Define the root condition we need to prove
theorem find_monic_cubic_polynomial_with_root (a : ℝ) (ha : a = (5 : ℝ)^(1/3) + 1) : Q a = 0 :=
by
  -- Proof goes here (omitted)
  sorry

end find_monic_cubic_polynomial_with_root_l1346_134699


namespace PQ_R_exist_l1346_134603

theorem PQ_R_exist :
  ∃ P Q R : ℚ, 
    (P = -3/5) ∧ (Q = -1) ∧ (R = 13/5) ∧
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
    (x^2 - 10)/((x - 1)*(x - 4)*(x - 6)) = P/(x - 1) + Q/(x - 4) + R/(x - 6)) :=
by
  sorry

end PQ_R_exist_l1346_134603


namespace slope_angle_of_line_l1346_134647

theorem slope_angle_of_line (α : ℝ) (hα : 0 ≤ α ∧ α < 180) 
    (slope_eq_tan : Real.tan α = 1) : α = 45 :=
by
  sorry

end slope_angle_of_line_l1346_134647


namespace leonardo_nap_duration_l1346_134631

theorem leonardo_nap_duration (h : (1 : ℝ) / 5 * 60 = 12) : (1 / 5 : ℝ) * 60 = 12 :=
by 
  exact h

end leonardo_nap_duration_l1346_134631


namespace find_d_l1346_134630

theorem find_d (a b c d : ℝ) (hac : 0 < a) (hbc : 0 < b) (hcc : 0 < c) (hdc : 0 < d)
  (oscillates : ∀ x, -2 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 4) :
  d = 1 :=
sorry

end find_d_l1346_134630


namespace n_squared_plus_n_plus_1_is_perfect_square_l1346_134643

theorem n_squared_plus_n_plus_1_is_perfect_square (n : ℕ) :
  (∃ k : ℕ, n^2 + n + 1 = k^2) ↔ n = 0 :=
by
  sorry

end n_squared_plus_n_plus_1_is_perfect_square_l1346_134643


namespace least_m_for_no_real_roots_l1346_134681

theorem least_m_for_no_real_roots : ∃ (m : ℤ), (∀ (x : ℝ), 3 * x * (m * x + 6) - 2 * x^2 + 8 ≠ 0) ∧ m = 4 := 
sorry

end least_m_for_no_real_roots_l1346_134681


namespace scalene_triangle_smallest_angle_sum_l1346_134620

theorem scalene_triangle_smallest_angle_sum :
  ∀ (A B C : ℝ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A = 45 ∧ C = 135 → (∃ x y : ℝ, x = y ∧ x = 45 ∧ y = 45 ∧ x + y = 90) :=
by
  intros A B C h
  sorry

end scalene_triangle_smallest_angle_sum_l1346_134620


namespace projectile_reaches_40_at_first_time_l1346_134653

theorem projectile_reaches_40_at_first_time : ∃ t : ℝ, 0 < t ∧ (40 = -16 * t^2 + 64 * t) ∧ (∀ t' : ℝ, 0 < t' ∧ t' < t → ¬ (40 = -16 * t'^2 + 64 * t')) ∧ t = 0.8 :=
by
  sorry

end projectile_reaches_40_at_first_time_l1346_134653


namespace range_of_a_same_side_of_line_l1346_134641

theorem range_of_a_same_side_of_line 
  {P Q : ℝ × ℝ} 
  (hP : P = (3, -1)) 
  (hQ : Q = (-1, 2)) 
  (h_side : (3 * a - 3) * (-a + 3) > 0) : 
  a > 1 ∧ a < 3 := 
by 
  sorry

end range_of_a_same_side_of_line_l1346_134641


namespace find_x4_plus_y4_l1346_134692

theorem find_x4_plus_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^4 + y^4 = 135.5 :=
by
  sorry

end find_x4_plus_y4_l1346_134692


namespace average_first_14_even_numbers_l1346_134607

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun x => 2 * (x + 1))

theorem average_first_14_even_numbers :
  let even_nums := first_n_even_numbers 14
  (even_nums.sum / even_nums.length = 15) :=
by
  sorry

end average_first_14_even_numbers_l1346_134607


namespace min_red_hair_students_l1346_134673

theorem min_red_hair_students (B N R : ℕ) 
  (h1 : B + N + R = 50)
  (h2 : N ≥ B - 1)
  (h3 : R ≥ N - 1) :
  R = 17 := sorry

end min_red_hair_students_l1346_134673


namespace total_gold_value_l1346_134626

def legacy_bars : ℕ := 5
def aleena_bars : ℕ := legacy_bars - 2
def value_per_bar : ℕ := 2200
def total_bars : ℕ := legacy_bars + aleena_bars
def total_value : ℕ := total_bars * value_per_bar

theorem total_gold_value : total_value = 17600 :=
by
  -- Begin proof
  sorry

end total_gold_value_l1346_134626


namespace function_values_l1346_134632

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * x^2 + c

theorem function_values (a b c : ℝ) : 
  f a b c 1 = 1 ∧ f a b c (-1) = 1 := 
by
  sorry

end function_values_l1346_134632


namespace number_of_multiples_of_10_lt_200_l1346_134698

theorem number_of_multiples_of_10_lt_200 : 
  ∃ n, (∀ k, (1 ≤ k) → (k < 20) → k * 10 < 200) ∧ n = 19 := 
by
  sorry

end number_of_multiples_of_10_lt_200_l1346_134698


namespace cos_value_in_second_quadrant_l1346_134667

variable (a : ℝ)
variables (h1 : π/2 < a ∧ a < π) (h2 : Real.sin a = 5/13)

theorem cos_value_in_second_quadrant : Real.cos a = -12/13 :=
  sorry

end cos_value_in_second_quadrant_l1346_134667


namespace stratified_sampling_size_l1346_134658

theorem stratified_sampling_size (a_ratio b_ratio c_ratio : ℕ) (total_items_A : ℕ) (h_ratio : a_ratio + b_ratio + c_ratio = 10)
  (h_A_ratio : a_ratio = 2) (h_B_ratio : b_ratio = 3) (h_C_ratio : c_ratio = 5) (items_A : total_items_A = 20) : 
  ∃ n : ℕ, n = total_items_A * 5 := 
by {
  -- The proof should go here. Since we only need the statement:
  sorry
}

end stratified_sampling_size_l1346_134658


namespace domain_of_f_l1346_134639

noncomputable def f (x : ℝ) : ℝ := (5 * x - 2) / Real.sqrt (x^2 - 3 * x - 4)

theorem domain_of_f :
  {x : ℝ | ∃ (f_x : ℝ), f x = f_x} = {x : ℝ | (x < -1) ∨ (x > 4)} :=
by
  sorry

end domain_of_f_l1346_134639


namespace readers_both_l1346_134606

-- Define the given conditions
def total_readers : ℕ := 250
def readers_S : ℕ := 180
def readers_L : ℕ := 88

-- Define the proof statement
theorem readers_both : (readers_S + readers_L - total_readers = 18) :=
by
  -- Proof is omitted
  sorry

end readers_both_l1346_134606


namespace arccos_sin_eq_pi_div_two_sub_1_72_l1346_134600

theorem arccos_sin_eq_pi_div_two_sub_1_72 :
  Real.arccos (Real.sin 8) = Real.pi / 2 - 1.72 :=
sorry

end arccos_sin_eq_pi_div_two_sub_1_72_l1346_134600


namespace average_of_combined_results_l1346_134691

theorem average_of_combined_results {avg1 avg2 n1 n2 : ℝ} (h1 : avg1 = 28) (h2 : avg2 = 55) (h3 : n1 = 55) (h4 : n2 = 28) :
  ((n1 * avg1) + (n2 * avg2)) / (n1 + n2) = 37.11 :=
by sorry

end average_of_combined_results_l1346_134691


namespace problem_statement_l1346_134623

noncomputable def f (x : ℝ) : ℝ := x + 1
noncomputable def g (x : ℝ) : ℝ := -x + 1
noncomputable def h (x : ℝ) : ℝ := f x * g x

theorem problem_statement :
  (h (-x) = h x) :=
by
  sorry

end problem_statement_l1346_134623


namespace money_equations_l1346_134635

theorem money_equations (x y : ℝ) (h1 : x + (1 / 2) * y = 50) (h2 : y + (2 / 3) * x = 50) :
  x + (1 / 2) * y = 50 ∧ y + (2 / 3) * x = 50 :=
by
  exact ⟨h1, h2⟩

-- Please note that by stating the theorem this way, we have restated the conditions and conclusion
-- in Lean 4. The proof uses the given conditions directly without the need for intermediate steps.

end money_equations_l1346_134635


namespace oil_truck_radius_l1346_134669

theorem oil_truck_radius
  (r_stationary : ℝ) (h_stationary : ℝ) (h_drop : ℝ) 
  (h_truck : ℝ)
  (V_pumped : ℝ) (π : ℝ) (r_truck : ℝ) :
  r_stationary = 100 → h_stationary = 25 → h_drop = 0.064 → h_truck = 10 →
  V_pumped = π * r_stationary^2 * h_drop →
  V_pumped = π * r_truck^2 * h_truck →
  r_truck = 8 := 
by 
  intros r_stationary_eq h_stationary_eq h_drop_eq h_truck_eq V_pumped_eq1 V_pumped_eq2
  sorry

end oil_truck_radius_l1346_134669


namespace original_number_l1346_134640

variable (x : ℝ)

theorem original_number (h1 : x - x / 10 = 37.35) : x = 41.5 := by
  sorry

end original_number_l1346_134640


namespace problem1_problem2_problem3_problem4_l1346_134612

theorem problem1 : (-4.7 : ℝ) + 0.9 = -3.8 := by
  sorry

theorem problem2 : (- (1 / 2) : ℝ) - (-(1 / 3)) = -(1 / 6) := by
  sorry

theorem problem3 : (- (10 / 9) : ℝ) * (- (6 / 10)) = (2 / 3) := by
  sorry

theorem problem4 : (0 : ℝ) * (-5) = 0 := by
  sorry

end problem1_problem2_problem3_problem4_l1346_134612


namespace central_cell_value_l1346_134655

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l1346_134655


namespace angle_measure_F_l1346_134696

theorem angle_measure_F (D E F : ℝ) 
  (h1 : D = 75) 
  (h2 : E = 4 * F - 15) 
  (h3 : D + E + F = 180) : 
  F = 24 := 
sorry

end angle_measure_F_l1346_134696


namespace total_sweaters_l1346_134622

-- Define the conditions
def washes_per_load : ℕ := 9
def total_shirts : ℕ := 19
def total_loads : ℕ := 3

-- Define the total_sweaters theorem to prove Nancy had to wash 9 sweaters
theorem total_sweaters {n : ℕ} (h1 : washes_per_load = 9) (h2 : total_shirts = 19) (h3 : total_loads = 3) : n = 9 :=
by
  sorry

end total_sweaters_l1346_134622


namespace total_savings_percentage_l1346_134656

theorem total_savings_percentage :
  let coat_price := 100
  let hat_price := 50
  let shoes_price := 75
  let coat_discount := 0.30
  let hat_discount := 0.40
  let shoes_discount := 0.25
  let original_total := coat_price + hat_price + shoes_price
  let coat_savings := coat_price * coat_discount
  let hat_savings := hat_price * hat_discount
  let shoes_savings := shoes_price * shoes_discount
  let total_savings := coat_savings + hat_savings + shoes_savings
  let savings_percentage := (total_savings / original_total) * 100
  savings_percentage = 30.556 :=
by
  sorry

end total_savings_percentage_l1346_134656


namespace median_and_mode_l1346_134634

open Set

variable (data_set : List ℝ)
variable (mean : ℝ)

noncomputable def median (l : List ℝ) : ℝ := sorry -- Define medial function
noncomputable def mode (l : List ℝ) : ℝ := sorry -- Define mode function

theorem median_and_mode (x : ℝ) (mean_set : (3 + x + 4 + 5 + 8) / 5 = 5) :
  data_set = [3, 4, 5, 5, 8] ∧ median data_set = 5 ∧ mode data_set = 5 :=
by
  have hx : x = 5 := sorry
  have hdata_set : data_set = [3, 4, 5, 5, 8] := sorry
  have hmedian : median data_set = 5 := sorry
  have hmode : mode data_set = 5 := sorry
  exact ⟨hdata_set, hmedian, hmode⟩

end median_and_mode_l1346_134634


namespace intersection_infinite_l1346_134617

-- Define the equations of the curves
def curve1 (x y : ℝ) : Prop := 2 * x^2 - x * y - y^2 - x - 2 * y - 1 = 0
def curve2 (x y : ℝ) : Prop := 3 * x^2 - 4 * x * y + y^2 - 3 * x + y = 0

-- Theorem statement
theorem intersection_infinite : ∃ (f : ℝ → ℝ), ∀ x, curve1 x (f x) ∧ curve2 x (f x) :=
sorry

end intersection_infinite_l1346_134617


namespace problem_statement_l1346_134679

variable (X Y : ℝ)

theorem problem_statement
  (h1 : 0.18 * X = 0.54 * 1200)
  (h2 : X = 4 * Y) :
  X = 3600 ∧ Y = 900 := by
  sorry

end problem_statement_l1346_134679


namespace triangle_inequality_sides_l1346_134621

theorem triangle_inequality_sides {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : b + c > a) (triangle_ineq3 : c + a > b) : 
  |(a / b) + (b / c) + (c / a) - (b / a) - (c / b) - (a / c)| < 1 :=
  sorry

end triangle_inequality_sides_l1346_134621


namespace randy_blocks_left_l1346_134672

theorem randy_blocks_left 
  (initial_blocks : ℕ := 78)
  (used_blocks : ℕ := 19)
  (given_blocks : ℕ := 25)
  (bought_blocks : ℕ := 36)
  (sets_from_sister : ℕ := 3)
  (blocks_per_set : ℕ := 12) :
  (initial_blocks - used_blocks - given_blocks + bought_blocks + (sets_from_sister * blocks_per_set)) / 2 = 53 := 
by
  sorry

end randy_blocks_left_l1346_134672


namespace proof_height_difference_l1346_134608

noncomputable def height_in_inches_between_ruby_and_xavier : Prop :=
  let janet_height_inches := 62.75
  let inch_to_cm := 2.54
  let janet_height_cm := janet_height_inches * inch_to_cm
  let charlene_height := 1.5 * janet_height_cm
  let pablo_height := charlene_height + 1.85 * 100
  let ruby_height := pablo_height - 0.5
  let xavier_height := charlene_height + 2.13 * 100 - 97.75
  let paul_height := ruby_height + 50
  let height_diff_cm := xavier_height - ruby_height
  let height_diff_inches := height_diff_cm / inch_to_cm
  height_diff_inches = -18.78

theorem proof_height_difference :
  height_in_inches_between_ruby_and_xavier :=
by
  sorry

end proof_height_difference_l1346_134608


namespace A_investment_l1346_134694

theorem A_investment (x : ℝ) (hx : 0 < x) :
  (∃ a b c d e : ℝ,
    a = x ∧ b = 12 ∧ c = 200 ∧ d = 6 ∧ e = 60 ∧ 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    ((a * b) / (a * b + c * d)) * 100 = e)
  → x = 150 :=
by
  sorry

end A_investment_l1346_134694


namespace range_of_x_l1346_134648

noncomputable def function_domain (x : ℝ) : Prop :=
x + 2 > 0 ∧ x ≠ 1

theorem range_of_x {x : ℝ} (h : function_domain x) : x > -2 ∧ x ≠ 1 :=
by
  sorry

end range_of_x_l1346_134648


namespace value_of_fraction_l1346_134665

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = (-4 - (-1)) / (4 - 1)

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
  b2 * b2 = (-4) * (-1) ∧ b2 < 0

theorem value_of_fraction (a1 a2 b2 : ℝ)
  (h1 : arithmetic_sequence a1 a2)
  (h2 : geometric_sequence b2) :
  (a2 - a1) / b2 = 1 / 2 :=
by
  sorry

end value_of_fraction_l1346_134665


namespace intersection_complement_l1346_134660

def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}
def compl_U_N : Set ℕ := {x ∈ U | x ∉ N}

theorem intersection_complement :
  M ∩ compl_U_N = {4} :=
by
  have h1 : compl_U_N = {2, 4, 8} := by sorry
  have h2 : M ∩ compl_U_N = {4} := by sorry
  exact h2

end intersection_complement_l1346_134660


namespace minimum_toys_to_add_l1346_134657

theorem minimum_toys_to_add {T : ℤ} (k m n : ℤ) (h1 : T = 12 * k + 3) (h2 : T = 18 * m + 3) 
  (h3 : T = 36 * n + 3) : 
  ∃ x : ℤ, (T + x) % 7 = 0 ∧ x = 4 :=
sorry

end minimum_toys_to_add_l1346_134657


namespace tan_double_angle_tan_angle_add_pi_div_4_l1346_134633

theorem tan_double_angle (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

theorem tan_angle_add_pi_div_4 (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α + Real.pi / 4) = -7 :=
by
  sorry

end tan_double_angle_tan_angle_add_pi_div_4_l1346_134633


namespace remainder_43_pow_43_plus_43_mod_44_l1346_134649

theorem remainder_43_pow_43_plus_43_mod_44 :
  let n := 43
  let m := 44
  (n^43 + n) % m = 42 :=
by 
  let n := 43
  let m := 44
  sorry

end remainder_43_pow_43_plus_43_mod_44_l1346_134649


namespace find_z_l1346_134670

-- Define the given angles
def angle_ABC : ℝ := 95
def angle_BAC : ℝ := 65

-- Define the angle sum property for triangle ABC
def angle_sum_triangle_ABC (a b : ℝ) : ℝ := 180 - (a + b)

-- Define the angle DCE as equal to angle BCA
def angle_DCE : ℝ := angle_sum_triangle_ABC angle_ABC angle_BAC

-- Define the angle sum property for right triangle CDE
def z (dce : ℝ) : ℝ := 90 - dce

-- State the theorem to be proved
theorem find_z : z angle_DCE = 70 :=
by
  -- Statement for proof is provided
  sorry

end find_z_l1346_134670


namespace unique_solution_c_eq_one_l1346_134688

theorem unique_solution_c_eq_one (b c : ℝ) (hb : b > 0) 
  (h_unique_solution : ∃ x : ℝ, x^2 + (b + 1/b) * x + c = 0 ∧ 
  ∀ y : ℝ, y^2 + (b + 1/b) * y + c = 0 → y = x) : c = 1 :=
by
  sorry

end unique_solution_c_eq_one_l1346_134688


namespace geometric_sequence_sum_l1346_134651

theorem geometric_sequence_sum (S : ℕ → ℚ) (n : ℕ) 
  (hS_n : S n = 54) 
  (hS_2n : S (2 * n) = 60) 
  : S (3 * n) = 60 + 2 / 3 := 
sorry

end geometric_sequence_sum_l1346_134651


namespace hannah_age_l1346_134689

-- Define the constants and conditions
variables (E F G H : ℕ)
axiom h₁ : E = F - 4
axiom h₂ : F = G + 6
axiom h₃ : H = G + 2
axiom h₄ : E = 15

-- Prove that Hannah is 15 years old
theorem hannah_age : H = 15 :=
by sorry

end hannah_age_l1346_134689


namespace nina_money_l1346_134610

theorem nina_money (W : ℝ) (h1 : W > 0) (h2 : 10 * W = 14 * (W - 1)) : 10 * W = 35 := by
  sorry

end nina_money_l1346_134610


namespace minji_total_water_intake_l1346_134642

variable (morning_water : ℝ)
variable (afternoon_water : ℝ)

theorem minji_total_water_intake (h_morning : morning_water = 0.26) (h_afternoon : afternoon_water = 0.37):
  morning_water + afternoon_water = 0.63 :=
sorry

end minji_total_water_intake_l1346_134642


namespace average_ducks_l1346_134614

theorem average_ducks (a e k : ℕ) 
  (h1 : a = 2 * e) 
  (h2 : e = k - 45) 
  (h3 : a = 30) :
  (a + e + k) / 3 = 35 :=
by
  sorry

end average_ducks_l1346_134614


namespace find_number_l1346_134674

theorem find_number (x : ℤ) (h : 3 * (3 * x) = 18) : x = 2 := 
sorry

end find_number_l1346_134674


namespace three_students_with_A_l1346_134652

-- Define the statements of the students
variables (Eliza Fiona George Harry : Prop)

-- Conditions based on the problem statement
axiom Fiona_implies_Eliza : Fiona → Eliza
axiom George_implies_Fiona : George → Fiona
axiom Harry_implies_George : Harry → George

-- There are exactly three students who scored an A
theorem three_students_with_A (hE : Bool) : 
  (Eliza = false) → (Fiona = true) → (George = true) → (Harry = true) :=
by
  sorry

end three_students_with_A_l1346_134652


namespace planA_charge_for_8_minutes_eq_48_cents_l1346_134693

theorem planA_charge_for_8_minutes_eq_48_cents
  (X : ℝ)
  (hA : ∀ t : ℝ, t ≤ 8 → X = X)
  (hB : ∀ t : ℝ, 6 * 0.08 = 0.48)
  (hEqual : 6 * 0.08 = X) :
  X = 0.48 := by
  sorry

end planA_charge_for_8_minutes_eq_48_cents_l1346_134693


namespace value_increase_factor_l1346_134646

theorem value_increase_factor (P S : ℝ) (frac F : ℝ) (hP : P = 200) (hS : S = 240) (hfrac : frac = 0.40) :
  frac * (P * F) = S -> F = 3 := by
  sorry

end value_increase_factor_l1346_134646


namespace charity_donation_correct_l1346_134683

-- Define each donation series for Suzanne, Maria, and James
def suzanne_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 10
  | (n+1)  => 2 * suzanne_donation_per_km n

def maria_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 15
  | (n+1)  => 1.5 * maria_donation_per_km n

def james_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 20
  | (n+1)  => 2 * james_donation_per_km n

-- Total donations after 5 kilometers
def total_donation_suzanne : ℝ := (List.range 5).map suzanne_donation_per_km |>.sum
def total_donation_maria : ℝ := (List.range 5).map maria_donation_per_km |>.sum
def total_donation_james : ℝ := (List.range 5).map james_donation_per_km |>.sum

def total_donation_charity : ℝ :=
  total_donation_suzanne + total_donation_maria + total_donation_james

-- Statement to be proven
theorem charity_donation_correct : total_donation_charity = 1127.81 := by
  sorry

end charity_donation_correct_l1346_134683


namespace abs_inequality_holds_l1346_134659

theorem abs_inequality_holds (m x : ℝ) (h : -1 ≤ m ∧ m ≤ 6) : 
  |x - 2| + |x + 4| ≥ m^2 - 5 * m :=
sorry

end abs_inequality_holds_l1346_134659


namespace exists_special_number_divisible_by_1991_l1346_134627

theorem exists_special_number_divisible_by_1991 :
  ∃ (N : ℤ) (n : ℕ), n > 2 ∧ (N % 1991 = 0) ∧ 
  (∃ a b x : ℕ, N = 10 ^ (n + 1) * a + 10 ^ n * x + 9 * 10 ^ (n - 1) + b) :=
sorry

end exists_special_number_divisible_by_1991_l1346_134627


namespace find_y_of_equations_l1346_134690

theorem find_y_of_equations (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 1 + 1 / y) (h2 : y = 2 + 1 / x) : 
  y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3 :=
by
  sorry

end find_y_of_equations_l1346_134690


namespace f_14_52_l1346_134668

def f : ℕ × ℕ → ℕ := sorry

axiom f_xx (x : ℕ) : f (x, x) = x
axiom f_symm (x y : ℕ) : f (x, y) = f (y, x)
axiom f_eq (x y : ℕ) : (x + y) * f (x, y) = y * f (x, x + y)

theorem f_14_52 : f (14, 52) = 364 := sorry

end f_14_52_l1346_134668


namespace min_value_inequality_l1346_134686

theorem min_value_inequality (a b c : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h : a + b + c = 2) : 
  (1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a)) ≥ 27 / 8 :=
sorry

end min_value_inequality_l1346_134686


namespace total_amount_pqr_l1346_134666

theorem total_amount_pqr (p q r : ℕ) (T : ℕ) 
  (hr : r = 2 / 3 * (T - r))
  (hr_value : r = 1600) : 
  T = 4000 :=
by
  sorry

end total_amount_pqr_l1346_134666


namespace area_triangle_BFC_l1346_134695

-- Definitions based on conditions
def Rectangle (A B C D : Type) (AB BC CD DA : ℝ) := AB = 5 ∧ BC = 12 ∧ CD = 5 ∧ DA = 12

def PointOnDiagonal (F A C : Type) := True  -- Simplified definition as being on the diagonal
def Perpendicular (B F A C : Type) := True  -- Simplified definition as being perpendicular

-- Main theorem statement
theorem area_triangle_BFC 
  (A B C D F : Type)
  (rectangle_ABCD : Rectangle A B C D 5 12 5 12)
  (F_on_AC : PointOnDiagonal F A C)
  (BF_perpendicular_AC : Perpendicular B F A C) :
  ∃ (area : ℝ), area = 30 :=
sorry

end area_triangle_BFC_l1346_134695


namespace fourth_vs_third_difference_l1346_134685

def first_competitor_distance : ℕ := 22

def second_competitor_distance : ℕ := first_competitor_distance + 1

def third_competitor_distance : ℕ := second_competitor_distance - 2

def fourth_competitor_distance : ℕ := 24

theorem fourth_vs_third_difference : 
  fourth_competitor_distance - third_competitor_distance = 3 := by
  sorry

end fourth_vs_third_difference_l1346_134685


namespace Marissa_sunflower_height_l1346_134609

-- Define the necessary conditions
def sister_height_feet : ℕ := 4
def sister_height_inches : ℕ := 3
def extra_sunflower_height : ℕ := 21
def inches_per_foot : ℕ := 12

-- Calculate the total height of the sister in inches
def sister_total_height_inch : ℕ := (sister_height_feet * inches_per_foot) + sister_height_inches

-- Calculate the sunflower height in inches
def sunflower_height_inch : ℕ := sister_total_height_inch + extra_sunflower_height

-- Convert the sunflower height to feet
def sunflower_height_feet : ℕ := sunflower_height_inch / inches_per_foot

-- The theorem we want to prove
theorem Marissa_sunflower_height : sunflower_height_feet = 6 := by
  sorry

end Marissa_sunflower_height_l1346_134609


namespace surveys_on_tuesday_l1346_134625

theorem surveys_on_tuesday
  (num_surveys_monday: ℕ) -- number of surveys Bart completed on Monday
  (earnings_monday: ℕ) -- earning per survey on Monday
  (total_earnings: ℕ) -- total earnings over the two days
  (earnings_per_survey: ℕ) -- earnings Bart gets per survey
  (monday_earnings_eq : earnings_monday = num_surveys_monday * earnings_per_survey)
  (total_earnings_eq : total_earnings = earnings_monday + (8 : ℕ))
  (earnings_per_survey_eq : earnings_per_survey = 2)
  : ((8 : ℕ) / earnings_per_survey = 4) := sorry

end surveys_on_tuesday_l1346_134625


namespace axis_of_symmetry_parabola_l1346_134678

theorem axis_of_symmetry_parabola (x y : ℝ) :
  y = - (1 / 8) * x^2 → y = 2 :=
sorry

end axis_of_symmetry_parabola_l1346_134678


namespace determine_number_of_20_pound_boxes_l1346_134650

variable (numBoxes : ℕ) (avgWeight : ℕ) (x : ℕ) (y : ℕ)

theorem determine_number_of_20_pound_boxes 
  (h1 : numBoxes = 30) 
  (h2 : avgWeight = 18) 
  (h3 : x + y = 30) 
  (h4 : 10 * x + 20 * y = 540) : 
  y = 24 :=
  by
  sorry

end determine_number_of_20_pound_boxes_l1346_134650


namespace part_1_part_2_l1346_134616

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part_1 (a : ℝ) (h : ∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) : a = 2 :=
sorry

theorem part_2 (a : ℝ) (h : a = 2) : ∀ m, (∀ x, f (3 * x) a + f (x + 3) a ≥ m) ↔ m ≤ 5 / 3 :=
sorry

end part_1_part_2_l1346_134616


namespace candle_height_comparison_l1346_134663

def first_candle_height (t : ℝ) : ℝ := 10 - 2 * t
def second_candle_height (t : ℝ) : ℝ := 8 - 2 * t

theorem candle_height_comparison (t : ℝ) :
  first_candle_height t = 3 * second_candle_height t → t = 3.5 :=
by
  -- the main proof steps would be here
  sorry

end candle_height_comparison_l1346_134663


namespace find_x_for_abs_expression_zero_l1346_134680

theorem find_x_for_abs_expression_zero (x : ℚ) : |5 * x - 2| = 0 → x = 2 / 5 := by
  sorry

end find_x_for_abs_expression_zero_l1346_134680
