import Mathlib

namespace additional_length_of_track_l1157_115752

theorem additional_length_of_track (rise : ℝ) (grade1 grade2 : ℝ) (h_rise : rise = 800) (h_grade1 : grade1 = 0.04) (h_grade2 : grade2 = 0.02) :
  (rise / grade2) - (rise / grade1) = 20000 :=
by
  sorry

end additional_length_of_track_l1157_115752


namespace xyz_leq_36_l1157_115775

theorem xyz_leq_36 {x y z : ℝ} 
    (hx0 : x > 0) (hy0 : y > 0) (hz0 : z > 0) 
    (hx2 : x ≤ 2) (hy3 : y ≤ 3) 
    (hxyz_sum : x + y + z = 11) : 
    x * y * z ≤ 36 := 
by
  sorry

end xyz_leq_36_l1157_115775


namespace isosceles_triangle_perimeter_l1157_115792

-- Definitions for the side lengths
def side_a (x : ℝ) := 4 * x - 2
def side_b (x : ℝ) := x + 1
def side_c (x : ℝ) := 15 - 6 * x

-- Main theorem statement
theorem isosceles_triangle_perimeter (x : ℝ) (h1 : side_a x = side_b x ∨ side_a x = side_c x ∨ side_b x = side_c x) :
  (side_a x + side_b x + side_c x = 12.3) :=
  sorry

end isosceles_triangle_perimeter_l1157_115792


namespace factor_x6_minus_64_l1157_115713

theorem factor_x6_minus_64 :
  ∀ x : ℝ, (x^6 - 64) = (x-2) * (x+2) * (x^4 + 4*x^2 + 16) :=
by
  sorry

end factor_x6_minus_64_l1157_115713


namespace triangle_inequality_sum_zero_l1157_115757

theorem triangle_inequality_sum_zero (a b c p q r : ℝ) (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) (hpqr : p + q + r = 0) : a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
by 
  sorry

end triangle_inequality_sum_zero_l1157_115757


namespace prove_correct_options_l1157_115797

theorem prove_correct_options (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 2) :
  (min (((1 : ℝ) / x) + (1 / y)) = 2) ∧
  (max (x * y) = 1) ∧
  (min (x^2 + y^2) = 2) ∧
  (max (x * (y + 1)) = (9 / 4)) :=
by
  sorry

end prove_correct_options_l1157_115797


namespace symmetric_points_y_axis_l1157_115795

theorem symmetric_points_y_axis (a b : ℤ) 
  (h1 : a + 1 = 2) 
  (h2 : b + 2 = 3) : 
  a + b = 2 :=
by
  sorry

end symmetric_points_y_axis_l1157_115795


namespace inequality_solution_solution_set_l1157_115763

noncomputable def f (x a : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

theorem inequality_solution (a : ℝ) : 
  f 1 a > 0 ↔ 3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3 :=
by sorry

theorem solution_set (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 3 → f x a > b) ∧ (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x a = b) ↔ 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ b = -3 :=
by sorry

end inequality_solution_solution_set_l1157_115763


namespace diff_lines_not_parallel_perpendicular_same_plane_l1157_115787

-- Variables
variables (m n : Type) (α β : Type)

-- Conditions
-- m and n are different lines, which we can assume as different types (or elements of some type).
-- α and β are different planes, which we can assume as different types (or elements of some type).
-- There exist definitions for parallel and perpendicular relationships between lines and planes.

def areParallel (x y : Type) : Prop := sorry
def arePerpendicularToSamePlane (x y : Type) : Prop := sorry

-- Theorem Statement
theorem diff_lines_not_parallel_perpendicular_same_plane
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : ¬ areParallel m n) :
  ¬ arePerpendicularToSamePlane m n :=
sorry

end diff_lines_not_parallel_perpendicular_same_plane_l1157_115787


namespace parabola_constant_term_l1157_115799

theorem parabola_constant_term :
  ∃ b c : ℝ, (∀ x : ℝ, (x = 2 → 3 = x^2 + b * x + c) ∧ (x = 4 → 3 = x^2 + b * x + c)) → c = 11 :=
by
  sorry

end parabola_constant_term_l1157_115799


namespace find_a_sq_plus_b_sq_l1157_115783

-- Variables and conditions
variables (a b : ℝ)
-- Conditions from the problem
axiom h1 : a - b = 3
axiom h2 : a * b = 9

-- The proof statement
theorem find_a_sq_plus_b_sq (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 9) : a^2 + b^2 = 27 :=
by {
  sorry
}

end find_a_sq_plus_b_sq_l1157_115783


namespace find_f_value_l1157_115768

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_value (α : ℝ) (h : f 3 α = Real.sqrt 3) : f (1 / 4) α = 1 / 2 :=
by
  sorry

end find_f_value_l1157_115768


namespace divides_three_and_eleven_l1157_115765

theorem divides_three_and_eleven (n : ℕ) (h : n ≥ 1) : (n ∣ 3^n + 1 ∧ n ∣ 11^n + 1) ↔ (n = 1 ∨ n = 2) := by
  sorry

end divides_three_and_eleven_l1157_115765


namespace arithmetic_sum_S8_l1157_115781

theorem arithmetic_sum_S8 (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, S (n + 1) - S n = S 1 - S 0)
  (h_positive : ∀ n, S n > 0)
  (h_S4 : S 4 = 10)
  (h_S12 : S 12 = 130) : 
  S 8 = 40 :=
sorry

end arithmetic_sum_S8_l1157_115781


namespace smallest_prime_dividing_sum_l1157_115769

theorem smallest_prime_dividing_sum (a b : ℕ) (h₁ : a = 7^15) (h₂ : b = 9^17) (h₃ : a % 2 = 1) (h₄ : b % 2 = 1) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (a + b) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (a + b)) → q ≥ p := by
  sorry

end smallest_prime_dividing_sum_l1157_115769


namespace percent_non_sugar_l1157_115758

-- Definitions based on the conditions in the problem.
def pie_weight : ℕ := 200
def sugar_weight : ℕ := 50

-- Statement of the proof problem.
theorem percent_non_sugar : ((pie_weight - sugar_weight) * 100) / pie_weight = 75 :=
by
  sorry

end percent_non_sugar_l1157_115758


namespace gwen_walked_time_l1157_115789

-- Definition of given conditions
def time_jogged : ℕ := 15
def ratio_jogged_to_walked (j w : ℕ) : Prop := j * 3 = w * 5

-- Definition to state the exact time walked with given ratio
theorem gwen_walked_time (j w : ℕ) (h1 : j = time_jogged) (h2 : ratio_jogged_to_walked j w) : w = 9 :=
by
  sorry

end gwen_walked_time_l1157_115789


namespace avg_writing_speed_l1157_115705

theorem avg_writing_speed 
  (words1 hours1 words2 hours2 : ℕ)
  (h_words1 : words1 = 30000)
  (h_hours1 : hours1 = 60)
  (h_words2 : words2 = 50000)
  (h_hours2 : hours2 = 100) :
  (words1 + words2) / (hours1 + hours2) = 500 :=
by {
  sorry
}

end avg_writing_speed_l1157_115705


namespace standard_eq_of_tangent_circle_l1157_115726

-- Define the center and tangent condition of the circle
def center : ℝ × ℝ := (1, 2)
def tangent_to_x_axis (r : ℝ) : Prop := r = center.snd

-- The standard equation of the circle given the center and radius
def standard_eq_circle (h k r : ℝ) : Prop := ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement to prove the standard equation of the circle
theorem standard_eq_of_tangent_circle : 
  ∃ r, tangent_to_x_axis r ∧ standard_eq_circle 1 2 r := 
by 
  sorry

end standard_eq_of_tangent_circle_l1157_115726


namespace value_of_x_l1157_115773

theorem value_of_x (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := 
by 
  sorry

end value_of_x_l1157_115773


namespace remove_denominators_l1157_115780

theorem remove_denominators (x : ℝ) : (1 / 2 - (x - 1) / 3 = 1) → (3 - 2 * (x - 1) = 6) :=
by
  intro h
  sorry

end remove_denominators_l1157_115780


namespace expression_for_f_l1157_115761

variable {R : Type*} [CommRing R]

def f (x : R) : R := sorry

theorem expression_for_f (x : R) :
  (f (x-1) = x^2 + 4*x - 5) → (f x = x^2 + 6*x) := by
  sorry

end expression_for_f_l1157_115761


namespace exists_even_among_pythagorean_triplet_l1157_115798

theorem exists_even_among_pythagorean_triplet (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ x, (x = a ∨ x = b ∨ x = c) ∧ x % 2 = 0 :=
sorry

end exists_even_among_pythagorean_triplet_l1157_115798


namespace cubic_expression_l1157_115756

theorem cubic_expression {x : ℝ} (h : x + (1/x) = 5) : x^3 + (1/x^3) = 110 := 
by
  sorry

end cubic_expression_l1157_115756


namespace coeff_sum_eq_minus_243_l1157_115776

theorem coeff_sum_eq_minus_243 (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x y : ℝ, (x - 2 * y) ^ 5 = a * (x + 2 * y) ^ 5 + a₁ * (x + 2 * y)^4 * y + a₂ * (x + 2 * y)^3 * y^2 
             + a₃ * (x + 2 * y)^2 * y^3 + a₄ * (x + 2 * y) * y^4 + a₅ * y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -243 :=
by
  intros h
  sorry

end coeff_sum_eq_minus_243_l1157_115776


namespace necessary_sufficient_condition_l1157_115733

noncomputable def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + (4 - 2 * a)

theorem necessary_sufficient_condition (a : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) : 
  (∀ (x : ℝ), f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end necessary_sufficient_condition_l1157_115733


namespace range_of_m_l1157_115746

noncomputable def f (x m : ℝ) : ℝ := |x^2 - 4| + x^2 + m * x

theorem range_of_m 
  (f_has_two_distinct_zeros : ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < 3 ∧ f a m = 0 ∧ f b m = 0) :
  -14 / 3 < m ∧ m < -2 :=
sorry

end range_of_m_l1157_115746


namespace esperanzas_tax_ratio_l1157_115718

theorem esperanzas_tax_ratio :
  let rent := 600
  let food_expenses := (3 / 5) * rent
  let mortgage_bill := 3 * food_expenses
  let savings := 2000
  let gross_salary := 4840
  let total_expenses := rent + food_expenses + mortgage_bill + savings
  let taxes := gross_salary - total_expenses
  (taxes / savings) = (2 / 5) := by
  sorry

end esperanzas_tax_ratio_l1157_115718


namespace gcd_2720_1530_l1157_115782

theorem gcd_2720_1530 : Nat.gcd 2720 1530 = 170 := by
  sorry

end gcd_2720_1530_l1157_115782


namespace find_a_l1157_115720

theorem find_a : 
  ∃ a : ℝ, (a > 0) ∧ (1 / Real.logb 5 a + 1 / Real.logb 6 a + 1 / Real.logb 7 a = 1) ∧ a = 210 :=
by
  sorry

end find_a_l1157_115720


namespace find_extrema_l1157_115794

noncomputable def y (x : ℝ) := (Real.sin (3 * x))^2

theorem find_extrema : 
  ∃ (x : ℝ), (0 < x ∧ x < 0.6) ∧ (∀ ε > 0, ε < 0.6 - x → y (x + ε) ≤ y x ∧ y (x - ε) ≤ y x) ∧ x = Real.pi / 6 :=
by
  sorry

end find_extrema_l1157_115794


namespace train_speed_conversion_l1157_115755

theorem train_speed_conversion (s_mps : ℝ) (h : s_mps = 30.002399999999998) : 
  s_mps * 3.6 = 108.01 :=
by
  sorry

end train_speed_conversion_l1157_115755


namespace inequality_subtraction_l1157_115754

theorem inequality_subtraction {a b c : ℝ} (h : a > b) : a - c > b - c := 
sorry

end inequality_subtraction_l1157_115754


namespace fraction_product_simplified_l1157_115735

theorem fraction_product_simplified:
  (2 / 9 : ℚ) * (5 / 8 : ℚ) = 5 / 36 :=
by {
  sorry
}

end fraction_product_simplified_l1157_115735


namespace distance_midpoint_AD_to_BC_l1157_115770

variable (AC BC BD : ℕ)
variable (perpendicular : Prop)
variable (d : ℝ)

theorem distance_midpoint_AD_to_BC
  (h1 : AC = 6)
  (h2 : BC = 5)
  (h3 : BD = 3)
  (h4 : perpendicular) :
  d = Real.sqrt 5 + 2 := by
  sorry

end distance_midpoint_AD_to_BC_l1157_115770


namespace total_plums_correct_l1157_115796

/-- Each picked number of plums. -/
def melanie_picked := 4
def dan_picked := 9
def sally_picked := 3
def ben_picked := 2 * (melanie_picked + dan_picked)
def sally_ate := 2

/-- The total number of plums picked in the end. -/
def total_plums_picked :=
  melanie_picked + dan_picked + sally_picked + ben_picked - sally_ate

theorem total_plums_correct : total_plums_picked = 40 := by
  sorry

end total_plums_correct_l1157_115796


namespace find_a7_l1157_115767

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l1157_115767


namespace incorrect_value_of_observation_l1157_115700

theorem incorrect_value_of_observation
  (mean_initial : ℝ) (n : ℕ) (sum_initial: ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mean_corrected : ℝ)
  (h1 : mean_initial = 36) 
  (h2 : n = 50) 
  (h3 : sum_initial = n * mean_initial) 
  (h4 : correct_value = 45) 
  (h5 : mean_corrected = 36.5) 
  (sum_corrected : ℝ) 
  (h6 : sum_corrected = n * mean_corrected) : 
  incorrect_value = 20 := 
by 
  sorry

end incorrect_value_of_observation_l1157_115700


namespace left_square_side_length_l1157_115785

theorem left_square_side_length 
  (x y z : ℝ)
  (H1 : y = x + 17)
  (H2 : z = x + 11)
  (H3 : x + y + z = 52) : 
  x = 8 := by
  sorry

end left_square_side_length_l1157_115785


namespace max_cookies_l1157_115711

-- Definitions for the conditions
def John_money : ℕ := 2475
def cookie_cost : ℕ := 225

-- Statement of the problem
theorem max_cookies (x : ℕ) : cookie_cost * x ≤ John_money → x ≤ 11 :=
sorry

end max_cookies_l1157_115711


namespace max_area_circle_eq_l1157_115719

theorem max_area_circle_eq (m : ℝ) :
  (x y : ℝ) → (x - 1) ^ 2 + (y + m) ^ 2 = -(m - 3) ^ 2 + 1 → 
  (∃ (r : ℝ), (r = (1 : ℝ)) ∧ (m = 3) ∧ ((x - 1) ^ 2 + (y + 3) ^ 2 = 1)) :=
by
  sorry

end max_area_circle_eq_l1157_115719


namespace right_angle_triangle_exists_l1157_115740

theorem right_angle_triangle_exists (color : ℤ × ℤ → ℕ) (H1 : ∀ c : ℕ, ∃ p : ℤ × ℤ, color p = c) : 
  ∃ (A B C : ℤ × ℤ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (color A ≠ color B ∧ color B ≠ color C ∧ color C ≠ color A) ∧
  ((A.1 = B.1 ∧ B.2 = C.2 ∧ A.1 - C.1 = A.2 - B.2) ∨ (A.2 = B.2 ∧ B.1 = C.1 ∧ A.1 - B.1 = A.2 - C.2)) :=
sorry

end right_angle_triangle_exists_l1157_115740


namespace solve_for_a_l1157_115729

theorem solve_for_a (a x : ℝ) (h₁ : 2 * x - 3 = 5 * x - 2 * a) (h₂ : x = 1) : a = 3 :=
by
  sorry

end solve_for_a_l1157_115729


namespace intersecting_lines_k_value_l1157_115730

theorem intersecting_lines_k_value (k : ℝ) : 
  (∃ x y : ℝ, y = 7 * x + 5 ∧ y = -3 * x - 35 ∧ y = 4 * x + k) → k = -7 :=
by
  sorry

end intersecting_lines_k_value_l1157_115730


namespace option_A_incorrect_l1157_115723

theorem option_A_incorrect {a b m : ℤ} (h : am = bm) : m = 0 ∨ a = b :=
by sorry

end option_A_incorrect_l1157_115723


namespace initial_pipes_num_l1157_115788

variable {n : ℕ}

theorem initial_pipes_num (h1 : ∀ t : ℕ, (n * t = 8) → n = 3) (h2 : ∀ t : ℕ, (2 * t = 12) → n = 3) : n = 3 := 
by 
  sorry

end initial_pipes_num_l1157_115788


namespace remainder_of_n_mod_5_l1157_115766

theorem remainder_of_n_mod_5
  (n : Nat)
  (h1 : n^2 ≡ 4 [MOD 5])
  (h2 : n^3 ≡ 2 [MOD 5]) :
  n ≡ 3 [MOD 5] :=
sorry

end remainder_of_n_mod_5_l1157_115766


namespace focus_parabola_l1157_115753

theorem focus_parabola (x : ℝ) (y : ℝ): (y = 8 * x^2) → (0, 1 / 32) = (0, 1 / 32) :=
by
  intro h
  sorry

end focus_parabola_l1157_115753


namespace find_y_l1157_115760

-- Define the known values and the proportion relation
variable (x y : ℝ)
variable (h1 : 0.75 / x = y / 7)
variable (h2 : x = 1.05)

theorem find_y : y = 5 :=
by
sorry

end find_y_l1157_115760


namespace rabbit_hid_carrots_l1157_115725

theorem rabbit_hid_carrots (h_r h_f : ℕ) (x : ℕ)
  (rabbit_holes : 5 * h_r = x) 
  (fox_holes : 7 * h_f = x)
  (holes_relation : h_r = h_f + 6) :
  x = 105 :=
by
  sorry

end rabbit_hid_carrots_l1157_115725


namespace question1_question2_l1157_115741

-- Definitions:
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Question 1 Statement:
theorem question1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := by
  sorry

-- Question 2 Statement:
theorem question2 (a : ℝ) (h : A ∪ B a = A) : a > 3 := by
  sorry

end question1_question2_l1157_115741


namespace min_b_l1157_115771

-- Definitions
def S (n : ℕ) : ℤ := 2^n - 1
def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2^(n-1)
def b (n : ℕ) : ℤ := (a n)^2 - 7 * (a n) + 6

-- Theorem
theorem min_b : ∃ n : ℕ, (b n = -6) :=
sorry

end min_b_l1157_115771


namespace max_vehicles_div_by_100_l1157_115744

noncomputable def max_vehicles_passing_sensor (n : ℕ) : ℕ :=
  2 * (20000 * n / (5 + 10 * n))

theorem max_vehicles_div_by_100 : 
  (∀ n : ℕ, (n > 0) → (∃ M : ℕ, M = max_vehicles_passing_sensor n ∧ M / 100 = 40)) :=
sorry

end max_vehicles_div_by_100_l1157_115744


namespace greatest_possible_x_l1157_115706

theorem greatest_possible_x (x : ℕ) (h : x^4 / x^2 < 18) : x ≤ 4 :=
sorry

end greatest_possible_x_l1157_115706


namespace scalene_triangle_geometric_progression_l1157_115786

theorem scalene_triangle_geometric_progression :
  ∀ (q : ℝ), q ≠ 0 → 
  (∀ b : ℝ, b > 0 → b + q * b > q^2 * b ∧ q * b + q^2 * b > b ∧ b + q^2 * b > q * b) → 
  ¬((0.5 < q ∧ q < 1.7) ∨ q = 2.0) → false :=
by
  intros q hq_ne_zero hq hq_interval
  sorry

end scalene_triangle_geometric_progression_l1157_115786


namespace minimal_moves_for_7_disks_l1157_115764

/-- Mathematical model of the Tower of Hanoi problem with special rules --/
def tower_of_hanoi_moves (n : ℕ) : ℚ :=
  if n = 7 then 23 / 4 else sorry

/-- Proof problem for the minimal number of moves required to transfer all seven disks to rod C --/
theorem minimal_moves_for_7_disks : tower_of_hanoi_moves 7 = 23 / 4 := 
  sorry

end minimal_moves_for_7_disks_l1157_115764


namespace slices_ratio_l1157_115731

theorem slices_ratio (total_slices : ℕ) (hawaiian_slices : ℕ) (cheese_slices : ℕ) 
  (dean_hawaiian_eaten : ℕ) (frank_hawaiian_eaten : ℕ) (sammy_cheese_eaten : ℕ)
  (total_leftover : ℕ) (hawaiian_leftover : ℕ) (cheese_leftover : ℕ)
  (H1 : total_slices = 12)
  (H2 : hawaiian_slices = 12)
  (H3 : cheese_slices = 12)
  (H4 : dean_hawaiian_eaten = 6)
  (H5 : frank_hawaiian_eaten = 3)
  (H6 : total_leftover = 11)
  (H7 : hawaiian_leftover = hawaiian_slices - dean_hawaiian_eaten - frank_hawaiian_eaten)
  (H8 : cheese_leftover = total_leftover - hawaiian_leftover)
  (H9 : sammy_cheese_eaten = cheese_slices - cheese_leftover)
  : sammy_cheese_eaten / cheese_slices = 1 / 3 :=
by sorry

end slices_ratio_l1157_115731


namespace inequality_am_gm_l1157_115745

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (x + z) + (2 * z^2) / (x + y) ≥ x + y + z :=
by
  sorry

end inequality_am_gm_l1157_115745


namespace no_nat_numbers_satisfy_lcm_eq_l1157_115707

theorem no_nat_numbers_satisfy_lcm_eq (n m : ℕ) :
  ¬ (Nat.lcm (n^2) m + Nat.lcm n (m^2) = 2019) :=
sorry

end no_nat_numbers_satisfy_lcm_eq_l1157_115707


namespace mass_percentage_H_in_CaH₂_l1157_115779

def atomic_mass_Ca : ℝ := 40.08
def atomic_mass_H : ℝ := 1.008
def molar_mass_CaH₂ : ℝ := atomic_mass_Ca + 2 * atomic_mass_H

theorem mass_percentage_H_in_CaH₂ :
  (2 * atomic_mass_H / molar_mass_CaH₂) * 100 = 4.79 := 
by
  -- Skipping the detailed proof for now
  sorry

end mass_percentage_H_in_CaH₂_l1157_115779


namespace gold_coins_percentage_is_35_l1157_115737

-- Define the conditions: percentage of beads and percentage of silver coins
def percent_beads : ℝ := 0.30
def percent_silver_coins : ℝ := 0.50

-- Definition of the percentage of all objects that are gold coins
def percent_gold_coins (percent_beads percent_silver_coins : ℝ) : ℝ :=
  (1 - percent_beads) * (1 - percent_silver_coins)

-- The statement that we need to prove:
theorem gold_coins_percentage_is_35 :
  percent_gold_coins percent_beads percent_silver_coins = 0.35 :=
  by
    unfold percent_gold_coins percent_beads percent_silver_coins
    sorry

end gold_coins_percentage_is_35_l1157_115737


namespace employed_males_percent_l1157_115728

variable (population : ℝ) (percent_employed : ℝ) (percent_employed_females : ℝ)

theorem employed_males_percent :
  percent_employed = 120 →
  percent_employed_females = 33.33333333333333 →
  2 / 3 * percent_employed = 80 :=
by
  intros h1 h2
  sorry

end employed_males_percent_l1157_115728


namespace real_roots_exist_l1157_115747

noncomputable def cubic_equation (x : ℝ) := x^3 - x^2 - 2*x + 1

theorem real_roots_exist : ∃ (a b : ℝ), 
  cubic_equation a = 0 ∧ cubic_equation b = 0 ∧ a - a * b = 1 := 
by
  sorry

end real_roots_exist_l1157_115747


namespace find_notebooks_l1157_115751

theorem find_notebooks (S N : ℕ) (h1 : N = 4 * S + 3) (h2 : N + 6 = 5 * S) : N = 39 := 
by
  sorry 

end find_notebooks_l1157_115751


namespace books_left_l1157_115716

variable (initialBooks : ℕ) (soldBooks : ℕ) (remainingBooks : ℕ)

-- Conditions
def initial_conditions := initialBooks = 136 ∧ soldBooks = 109

-- Question: Proving the remaining books after the sale
theorem books_left (initial_conditions : initialBooks = 136 ∧ soldBooks = 109) : remainingBooks = 27 :=
by
  cases initial_conditions
  sorry

end books_left_l1157_115716


namespace apples_ratio_l1157_115793

theorem apples_ratio (bonnie_apples samuel_extra_apples samuel_left_over samuel_total_pies : ℕ) 
  (h_bonnie : bonnie_apples = 8)
  (h_samuel_extra : samuel_extra_apples = 20)
  (h_samuel_left_over : samuel_left_over = 10)
  (h_pie_ratio : samuel_total_pies = (8 + 20) / 7) :
  (28 - samuel_total_pies - 10) / 28 = 1 / 2 := 
by
  sorry

end apples_ratio_l1157_115793


namespace even_sum_exactly_one_even_l1157_115734

theorem even_sum_exactly_one_even (a b c : ℕ) (h : (a + b + c) % 2 = 0) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  sorry

end even_sum_exactly_one_even_l1157_115734


namespace length_of_PW_l1157_115748

-- Given variables
variables (CD WX DP PX : ℝ) (CW : ℝ)

-- Condition 1: CD is parallel to WX
axiom h1 : true -- Parallelism is given as part of the problem

-- Condition 2: CW = 60 units
axiom h2 : CW = 60

-- Condition 3: DP = 18 units
axiom h3 : DP = 18

-- Condition 4: PX = 36 units
axiom h4 : PX = 36

-- Question/Answer: Prove that the length of PW = 40 units
theorem length_of_PW (PW CP : ℝ) (h5 : CP = PW / 2) (h6 : CW = CP + PW) : PW = 40 :=
by sorry

end length_of_PW_l1157_115748


namespace paint_cost_for_flag_l1157_115759

noncomputable def flag_width : ℕ := 12
noncomputable def flag_height : ℕ := 10
noncomputable def paint_cost_per_quart : ℝ := 3.5
noncomputable def coverage_per_quart : ℕ := 4

theorem paint_cost_for_flag : (flag_width * flag_height * 2 / coverage_per_quart : ℝ) * paint_cost_per_quart = 210 := by
  sorry

end paint_cost_for_flag_l1157_115759


namespace find_number_l1157_115709

theorem find_number (x : ℝ) (h : 0.3 * x - (1 / 3) * (0.3 * x) = 36) : x = 180 :=
sorry

end find_number_l1157_115709


namespace num_valid_combinations_l1157_115715

-- Definitions based on the conditions
def num_herbs := 4
def num_gems := 6
def num_incompatible_gems := 3
def num_incompatible_herbs := 2

-- Statement to be proved
theorem num_valid_combinations :
  (num_herbs * num_gems) - (num_incompatible_gems * num_incompatible_herbs) = 18 :=
by
  sorry

end num_valid_combinations_l1157_115715


namespace arccos_half_eq_pi_over_three_l1157_115732

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l1157_115732


namespace quadratic_radical_type_l1157_115791

-- Problem statement: Given that sqrt(2a + 1) is a simplest quadratic radical and the same type as sqrt(48), prove that a = 1.

theorem quadratic_radical_type (a : ℝ) (h1 : ((2 * a) + 1) = 3) : a = 1 :=
by
  sorry

end quadratic_radical_type_l1157_115791


namespace connie_total_markers_l1157_115727

theorem connie_total_markers : 2315 + 1028 = 3343 :=
by
  sorry

end connie_total_markers_l1157_115727


namespace cafeteria_extra_fruit_l1157_115777

theorem cafeteria_extra_fruit 
    (red_apples : ℕ)
    (green_apples : ℕ)
    (students : ℕ)
    (total_apples := red_apples + green_apples)
    (apples_taken := students)
    (extra_apples := total_apples - apples_taken)
    (h1 : red_apples = 42)
    (h2 : green_apples = 7)
    (h3 : students = 9) :
    extra_apples = 40 := 
by 
  sorry

end cafeteria_extra_fruit_l1157_115777


namespace negation_proposition_l1157_115750

theorem negation_proposition (x : ℝ) : ¬(∀ x, x > 0 → x^2 > 0) ↔ ∃ x, x > 0 ∧ x^2 ≤ 0 :=
by
  sorry

end negation_proposition_l1157_115750


namespace total_blue_balloons_l1157_115742

theorem total_blue_balloons (Joan_balloons : ℕ) (Melanie_balloons : ℕ) (Alex_balloons : ℕ) 
  (hJoan : Joan_balloons = 60) (hMelanie : Melanie_balloons = 85) (hAlex : Alex_balloons = 37) :
  Joan_balloons + Melanie_balloons + Alex_balloons = 182 :=
by
  sorry

end total_blue_balloons_l1157_115742


namespace negation_equiv_l1157_115772

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop := 
  (is_even a ∧ ¬is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ is_even c)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop := 
  (is_even a ∧ is_even b) ∨ 
  (is_even a ∧ is_even c) ∨ 
  (is_even b ∧ is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ ¬is_even c)
  
theorem negation_equiv (a b c : ℕ) : 
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c := 
sorry

end negation_equiv_l1157_115772


namespace smallest_possible_sum_l1157_115721

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_ne : x ≠ y) (h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 50 :=
sorry

end smallest_possible_sum_l1157_115721


namespace primes_count_l1157_115714

open Int

theorem primes_count (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ r s : ℤ, ∀ x : ℤ, (x^3 - x + 2) % p = ((x - r)^2 * (x - s)) % p := 
  by
    sorry

end primes_count_l1157_115714


namespace find_number_to_add_l1157_115778

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l1157_115778


namespace solve_for_x_l1157_115717

-- Define the operation
def triangle (a b : ℝ) : ℝ := 2 * a - b

-- Define the necessary conditions and the goal
theorem solve_for_x :
  (∀ (a b : ℝ), triangle a b = 2 * a - b) →
  (∃ x : ℝ, triangle x (triangle 1 3) = 2) →
  ∃ x : ℝ, x = 1 / 2 :=
by 
  intros h_main h_eqn
  -- We can skip the proof part as requested.
  sorry

end solve_for_x_l1157_115717


namespace find_x_l1157_115712

theorem find_x (n : ℕ) (hn : n % 2 = 1) (hpf : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 * p2 * p3 = 9^n - 1 ∧ [p1, p2, p3].contains 61) :
  9^n - 1 = 59048 :=
by
  sorry

end find_x_l1157_115712


namespace crt_solution_l1157_115784

/-- Congruences from the conditions -/
def congruences : Prop :=
  ∃ x : ℤ, 
    (x % 2 = 1) ∧
    (x % 3 = 2) ∧
    (x % 5 = 3) ∧
    (x % 7 = 4)

/-- The target result from the Chinese Remainder Theorem -/
def target_result : Prop :=
  ∃ x : ℤ, 
    (x % 210 = 53)

/-- The proof problem stating that the given conditions imply the target result -/
theorem crt_solution : congruences → target_result :=
by
  sorry

end crt_solution_l1157_115784


namespace range_of_f_neg2_l1157_115749

def quadratic_fn (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_neg2 (a b : ℝ) (h1 : 1 ≤ quadratic_fn a b (-1) ∧ quadratic_fn a b (-1) ≤ 2)
    (h2 : 2 ≤ quadratic_fn a b 1 ∧ quadratic_fn a b 1 ≤ 4) :
    3 ≤ quadratic_fn a b (-2) ∧ quadratic_fn a b (-2) ≤ 12 :=
sorry

end range_of_f_neg2_l1157_115749


namespace find_p_l1157_115708

-- Lean 4 definitions corresponding to the conditions
variables {p a b x0 y0 : ℝ} (hp : p > 0) (ha : a > 0) (hb : b > 0) (hx0 : x0 ≠ 0)
variables (hA : (y0^2 = 2 * p * x0) ∧ ((x0 / a)^2 - (y0 / b)^2 = 1))
variables (h_dist : x0 + x0 = p^2)
variables (h_ecc : (5^.half) = sqrt 5)

-- The proof problem
theorem find_p :
  p = 1 :=
by
  sorry

end find_p_l1157_115708


namespace diamonds_in_G15_l1157_115739

theorem diamonds_in_G15 (G : ℕ → ℕ) 
  (h₁ : G 1 = 3)
  (h₂ : ∀ n, n ≥ 2 → G (n + 1) = 3 * (2 * (n - 1) + 3) - 3 ) :
  G 15 = 90 := sorry

end diamonds_in_G15_l1157_115739


namespace find_f_l1157_115743

variable (f : ℝ → ℝ)

open Function

theorem find_f (h : ∀ x: ℝ, f (3 * x + 2) = 9 * x + 8) : ∀ x: ℝ, f x = 3 * x + 2 := 
sorry

end find_f_l1157_115743


namespace increasing_function_solution_l1157_115722

noncomputable def solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y

theorem increasing_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y)
  ∧ (∀ x y : ℝ, x < y → f x < f y)
  → ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = 1 / (a * x) :=
by {
  sorry
}

end increasing_function_solution_l1157_115722


namespace slope_of_tangent_line_at_zero_l1157_115702

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 :=
by
  sorry 

end slope_of_tangent_line_at_zero_l1157_115702


namespace value_of_y_l1157_115736

theorem value_of_y (x y : ℕ) (h1 : x % y = 6) (h2 : (x : ℝ) / y = 6.12) : y = 50 :=
sorry

end value_of_y_l1157_115736


namespace rotary_club_eggs_needed_l1157_115724

theorem rotary_club_eggs_needed 
  (small_children_tickets : ℕ := 53)
  (older_children_tickets : ℕ := 35)
  (adult_tickets : ℕ := 75)
  (senior_tickets : ℕ := 37)
  (waste_percentage : ℝ := 0.03)
  (extra_omelets : ℕ := 25)
  (eggs_per_extra_omelet : ℝ := 2.5) :
  53 * 1 + 35 * 2 + 75 * 3 + 37 * 4 + 
  Nat.ceil (waste_percentage * (53 * 1 + 35 * 2 + 75 * 3 + 37 * 4)) + 
  Nat.ceil (extra_omelets * eggs_per_extra_omelet) = 574 := 
by 
  sorry

end rotary_club_eggs_needed_l1157_115724


namespace allan_initial_balloons_l1157_115790

theorem allan_initial_balloons (jake_balloons allan_bought_more allan_total_balloons : ℕ) 
  (h1 : jake_balloons = 4)
  (h2 : allan_bought_more = 3)
  (h3 : allan_total_balloons = 8) :
  ∃ (allan_initial_balloons : ℕ), allan_total_balloons = allan_initial_balloons + allan_bought_more ∧ allan_initial_balloons = 5 := 
by
  sorry

end allan_initial_balloons_l1157_115790


namespace december_revenue_times_average_l1157_115762

def revenue_in_december_is_multiple_of_average_revenue (R_N R_J R_D : ℝ) : Prop :=
  R_N = (3/5) * R_D ∧    -- Condition: November's revenue is 3/5 of December's revenue
  R_J = (1/3) * R_N ∧    -- Condition: January's revenue is 1/3 of November's revenue
  R_D = 2.5 * ((R_N + R_J) / 2)   -- Question: December's revenue is 2.5 times the average of November's and January's revenue

theorem december_revenue_times_average (R_N R_J R_D : ℝ) :
  revenue_in_december_is_multiple_of_average_revenue R_N R_J R_D :=
by
  -- adding sorry to skip the proof
  sorry

end december_revenue_times_average_l1157_115762


namespace prob_two_red_balls_consecutively_without_replacement_l1157_115701

def numOfRedBalls : ℕ := 3
def totalNumOfBalls : ℕ := 8

theorem prob_two_red_balls_consecutively_without_replacement :
  (numOfRedBalls / totalNumOfBalls) * ((numOfRedBalls - 1) / (totalNumOfBalls - 1)) = 3 / 28 :=
by
  sorry

end prob_two_red_balls_consecutively_without_replacement_l1157_115701


namespace alex_money_left_l1157_115703

noncomputable def alex_main_income : ℝ := 900
noncomputable def alex_side_income : ℝ := 300
noncomputable def main_job_tax_rate : ℝ := 0.15
noncomputable def side_job_tax_rate : ℝ := 0.20
noncomputable def water_bill : ℝ := 75
noncomputable def main_job_tithe_rate : ℝ := 0.10
noncomputable def side_job_tithe_rate : ℝ := 0.15
noncomputable def grocery_expense : ℝ := 150
noncomputable def transportation_expense : ℝ := 50

theorem alex_money_left :
  let main_income_after_tax := alex_main_income * (1 - main_job_tax_rate)
  let side_income_after_tax := alex_side_income * (1 - side_job_tax_rate)
  let total_income_after_tax := main_income_after_tax + side_income_after_tax
  let main_tithe := alex_main_income * main_job_tithe_rate
  let side_tithe := alex_side_income * side_job_tithe_rate
  let total_tithe := main_tithe + side_tithe
  let total_deductions := water_bill + grocery_expense + transportation_expense + total_tithe
  let money_left := total_income_after_tax - total_deductions
  money_left = 595 :=
by
  -- Proof goes here
  sorry

end alex_money_left_l1157_115703


namespace solution_to_inequality_l1157_115710

theorem solution_to_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : 1 / x > 1 :=
by
  sorry

end solution_to_inequality_l1157_115710


namespace problem1_l1157_115704

def setA : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def setB (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 2 * m - 2}

theorem problem1 (m : ℝ) : 
  (∀ x, x ∈ setA → x ∈ setB m) ∧ ¬(∀ x, x ∈ setA ↔ x ∈ setB m) → 3 ≤ m :=
sorry

end problem1_l1157_115704


namespace find_g_five_l1157_115738

theorem find_g_five 
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : g 0 = 1) : g 5 = Real.exp 5 :=
sorry

end find_g_five_l1157_115738


namespace cost_of_27_pounds_l1157_115774

def rate_per_pound : ℝ := 1
def weight_pounds : ℝ := 27

theorem cost_of_27_pounds :
  weight_pounds * rate_per_pound = 27 := 
by 
  -- sorry placeholder indicates that the proof is not provided
  sorry

end cost_of_27_pounds_l1157_115774
