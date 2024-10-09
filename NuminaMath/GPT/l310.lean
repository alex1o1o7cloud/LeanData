import Mathlib

namespace prove_ab_leq_one_l310_31071

theorem prove_ab_leq_one (a b : ℝ) (h : (a + b + a) * (a + b + b) = 9) : ab ≤ 1 := 
by
  sorry

end prove_ab_leq_one_l310_31071


namespace solve_for_y_l310_31038

-- Define the variables and conditions
variable (y : ℝ)
variable (h_pos : y > 0)
variable (h_seq : (4 + y^2 = 2 * y^2 ∧ y^2 + 25 = 2 * y^2))

-- State the theorem
theorem solve_for_y : y = Real.sqrt 14.5 :=
by sorry

end solve_for_y_l310_31038


namespace tailwind_speed_rate_of_change_of_ground_speed_l310_31015

-- Define constants and variables
variables (Vp Vw : ℝ) (altitude Vg1 Vg2 : ℝ)

-- Define conditions
def conditions := Vg1 = Vp + Vw ∧ altitude = 10000 ∧ Vg1 = 460 ∧
                  Vg2 = Vp - Vw ∧ altitude = 5000 ∧ Vg2 = 310

-- Define theorems to prove
theorem tailwind_speed (Vp Vw : ℝ) (altitude Vg1 Vg2 : ℝ) :
  conditions Vp Vw altitude Vg1 Vg2 → Vw = 75 :=
by
  sorry

theorem rate_of_change_of_ground_speed (altitude1 altitude2 Vg1 Vg2 : ℝ) :
  altitude1 = 10000 → altitude2 = 5000 → Vg1 = 460 → Vg2 = 310 →
  (Vg2 - Vg1) / (altitude2 - altitude1) = 0.03 :=
by
  sorry

end tailwind_speed_rate_of_change_of_ground_speed_l310_31015


namespace cos_pi_plus_alpha_l310_31073

theorem cos_pi_plus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π + α) = - 1 / 3 :=
by
  sorry

end cos_pi_plus_alpha_l310_31073


namespace mean_of_combined_sets_l310_31069

theorem mean_of_combined_sets (mean_set1 : ℝ) (mean_set2 : ℝ) (n1 : ℕ) (n2 : ℕ)
  (h1 : mean_set1 = 15) (h2 : mean_set2 = 27) (h3 : n1 = 7) (h4 : n2 = 8) :
  (mean_set1 * n1 + mean_set2 * n2) / (n1 + n2) = 21.4 := 
sorry

end mean_of_combined_sets_l310_31069


namespace polygon_diagonals_twice_sides_l310_31078

theorem polygon_diagonals_twice_sides
  (n : ℕ)
  (h : n * (n - 3) / 2 = 2 * n) :
  n = 7 :=
sorry

end polygon_diagonals_twice_sides_l310_31078


namespace base7_to_base10_l310_31030

theorem base7_to_base10 (a b c d e : ℕ) (h : 45321 = a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0)
  (ha : a = 4) (hb : b = 5) (hc : c = 3) (hd : d = 2) (he : e = 1) : 
  a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0 = 11481 := 
by 
  sorry

end base7_to_base10_l310_31030


namespace team_lineup_count_l310_31062

theorem team_lineup_count (total_members specialized_kickers remaining_players : ℕ) 
  (captain_assignments : specialized_kickers = 2) 
  (available_members : total_members = 20) 
  (choose_players : remaining_players = 8) : 
  (2 * (Nat.choose 19 remaining_players)) = 151164 := 
by
  sorry

end team_lineup_count_l310_31062


namespace percentage_error_square_area_l310_31080

theorem percentage_error_square_area (s : ℝ) (h : s > 0) :
  let s' := (1.02 * s)
  let actual_area := s^2
  let measured_area := s'^2
  let error_area := measured_area - actual_area
  let percentage_error := (error_area / actual_area) * 100
  percentage_error = 4.04 := 
sorry

end percentage_error_square_area_l310_31080


namespace brick_length_proof_l310_31095

-- Definitions based on conditions
def courtyard_length_m : ℝ := 18
def courtyard_width_m : ℝ := 16
def brick_width_cm : ℝ := 10
def total_bricks : ℝ := 14400

-- Conversion factors
def sqm_to_sqcm (area_sqm : ℝ) : ℝ := area_sqm * 10000
def courtyard_area_cm2 : ℝ := sqm_to_sqcm (courtyard_length_m * courtyard_width_m)

-- The proof statement
theorem brick_length_proof :
  (∀ (L : ℝ), courtyard_area_cm2 = total_bricks * (L * brick_width_cm)) → 
  (∃ (L : ℝ), L = 20) :=
by
  intro h
  sorry

end brick_length_proof_l310_31095


namespace rowing_time_l310_31049

def man_speed_still := 10.0
def river_speed := 1.2
def total_distance := 9.856

def upstream_speed := man_speed_still - river_speed
def downstream_speed := man_speed_still + river_speed

def one_way_distance := total_distance / 2
def time_upstream := one_way_distance / upstream_speed
def time_downstream := one_way_distance / downstream_speed

theorem rowing_time :
  time_upstream + time_downstream = 1 :=
by
  sorry

end rowing_time_l310_31049


namespace equal_mass_piles_l310_31058

theorem equal_mass_piles (n : ℕ) (hn : n > 3) (hn_mod : n % 3 = 0 ∨ n % 3 = 2) : 
  ∃ A B C : Finset ℕ, A ∪ B ∪ C = {i | i ∈ Finset.range (n + 1)} ∧
  Disjoint A B ∧ Disjoint A C ∧ Disjoint B C ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id :=
sorry

end equal_mass_piles_l310_31058


namespace books_per_day_l310_31083

-- Define the condition: Mrs. Hilt reads 15 books in 3 days.
def reads_books_in_days (total_books : ℕ) (days : ℕ) : Prop :=
  total_books = 15 ∧ days = 3

-- Define the theorem to prove that Mrs. Hilt reads 5 books per day.
theorem books_per_day (total_books : ℕ) (days : ℕ) (h : reads_books_in_days total_books days) : total_books / days = 5 :=
by
  -- Stub proof
  sorry

end books_per_day_l310_31083


namespace relationship_y1_y2_l310_31068

theorem relationship_y1_y2 (y1 y2 : ℝ) (m : ℝ) (h_m : m ≠ 0) 
  (hA : y1 = m * (-2) + 4) (hB : 3 = m * 1 + 4) (hC : y2 = m * 3 + 4) : y1 > y2 :=
by
  sorry

end relationship_y1_y2_l310_31068


namespace consumption_increased_by_27_91_percent_l310_31011
noncomputable def percentage_increase_in_consumption (T C : ℝ) : ℝ :=
  let new_tax_rate := 0.86 * T
  let new_revenue_effect := 1.1000000000000085
  let cons_percentage_increase (P : ℝ) := (new_tax_rate * (C * (1 + P))) = new_revenue_effect * (T * C)
  let P_solution := 0.2790697674418605
  if cons_percentage_increase P_solution then P_solution * 100 else 0

-- The statement we are proving
theorem consumption_increased_by_27_91_percent (T C : ℝ) (hT : 0 < T) (hC : 0 < C) :
  percentage_increase_in_consumption T C = 27.91 :=
by
  sorry

end consumption_increased_by_27_91_percent_l310_31011


namespace compute_modulo_l310_31056

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l310_31056


namespace min_value_quadratic_l310_31081

theorem min_value_quadratic : 
  ∀ x : ℝ, (4 * x^2 - 12 * x + 9) ≥ 0 :=
by
  sorry

end min_value_quadratic_l310_31081


namespace sum_of_adjacent_to_7_l310_31090

/-- Define the divisors of 245, excluding 1 -/
def divisors245 : Set ℕ := {5, 7, 35, 49, 245}

/-- Define the adjacency condition to ensure every pair of adjacent integers has a common factor greater than 1 -/
def adjacency_condition (a b : ℕ) : Prop := (a ≠ b) ∨ (Nat.gcd a b > 1)

/-- Prove the sum of the two integers adjacent to 7 in the given condition is 294. -/
theorem sum_of_adjacent_to_7 (d1 d2 : ℕ) (h1 : d1 ∈ divisors245) (h2 : d2 ∈ divisors245) 
    (adj1 : adjacency_condition 7 d1) (adj2 : adjacency_condition 7 d2) : 
    d1 + d2 = 294 := 
sorry

end sum_of_adjacent_to_7_l310_31090


namespace no_values_satisfy_equation_l310_31002

-- Define the sum of the digits function S
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the sum of digits of the sum of the digits function S(S(n))
noncomputable def sum_of_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (sum_of_digits n)

-- Theorem statement about the number of n satisfying n + S(n) + S(S(n)) = 2099
theorem no_values_satisfy_equation :
  (∃ n : ℕ, n > 0 ∧ n + sum_of_digits n + sum_of_sum_of_digits n = 2099) ↔ False := sorry

end no_values_satisfy_equation_l310_31002


namespace largest_integer_among_four_l310_31097

theorem largest_integer_among_four 
  (x y z w : ℤ)
  (h1 : x + y + z = 234)
  (h2 : x + y + w = 255)
  (h3 : x + z + w = 271)
  (h4 : y + z + w = 198) :
  max x (max y (max z w)) = 121 := 
by
  -- This is a placeholder for the actual proof
  sorry

end largest_integer_among_four_l310_31097


namespace exists_polynomial_for_divisors_l310_31020

open Polynomial

theorem exists_polynomial_for_divisors (n : ℕ) :
  (∃ P : ℤ[X], ∀ d : ℕ, d ∣ n → P.eval (d : ℤ) = (n / d : ℤ)^2) ↔
  (Nat.Prime n ∨ n = 1 ∨ n = 6) := by
  sorry

end exists_polynomial_for_divisors_l310_31020


namespace right_triangle_hypotenuse_l310_31036

theorem right_triangle_hypotenuse
  (a b c : ℝ)
  (h₀ : a = 24)
  (h₁ : a^2 + b^2 + c^2 = 2500)
  (h₂ : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end right_triangle_hypotenuse_l310_31036


namespace coplanar_AD_eq_linear_combination_l310_31075

-- Define the points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨4, 1, 3⟩
def B : Point3D := ⟨2, 3, 1⟩
def C : Point3D := ⟨3, 7, -5⟩
def D : Point3D := ⟨11, -1, 3⟩

-- Define the vectors
def vector (P Q : Point3D) : Point3D := ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def AB := vector A B
def AC := vector A C
def AD := vector A D

-- Coplanar definition: AD = λ AB + μ AC
theorem coplanar_AD_eq_linear_combination (lambda mu : ℝ) :
  AD = ⟨lambda * 2 + mu * (-1), lambda * (-2) + mu * 6, lambda * (-2) + mu * (-8)⟩ :=
sorry

end coplanar_AD_eq_linear_combination_l310_31075


namespace discount_rate_l310_31026

variable (P P_b P_s D : ℝ)

-- Conditions
variable (h1 : P_s = 1.24 * P)
variable (h2 : P_s = 1.55 * P_b)
variable (h3 : P_b = P * (1 - D))

theorem discount_rate :
  D = 0.2 :=
by
  sorry

end discount_rate_l310_31026


namespace candidate_knows_Excel_and_willing_nights_l310_31031

variable (PExcel PXNight : ℝ)
variable (H1 : PExcel = 0.20) (H2 : PXNight = 0.30)

theorem candidate_knows_Excel_and_willing_nights : (PExcel * PXNight) = 0.06 :=
by
  rw [H1, H2]
  norm_num

end candidate_knows_Excel_and_willing_nights_l310_31031


namespace area_of_circle_l310_31028

def circle_area (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y = 1

theorem area_of_circle : ∃ (area : ℝ), area = 6 * Real.pi :=
by sorry

end area_of_circle_l310_31028


namespace line_parallel_unique_a_l310_31043

theorem line_parallel_unique_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + a + 3 = 0 → x + (a + 1)*y + 4 = 0) → a = -2 :=
  by
  sorry

end line_parallel_unique_a_l310_31043


namespace right_triangle_leg_lengths_l310_31091

theorem right_triangle_leg_lengths (a b c : ℕ) (h : a ^ 2 + b ^ 2 = c ^ 2) (h1: c = 17) (h2: a + (c - b) = 17) (h3: b + (c - a) = 17) : a = 8 ∧ b = 15 :=
by {
  sorry
}

end right_triangle_leg_lengths_l310_31091


namespace fraction_equality_implies_equality_l310_31012

theorem fraction_equality_implies_equality (a b c : ℝ) (hc : c ≠ 0) :
  (a / c = b / c) → (a = b) :=
by {
  sorry
}

end fraction_equality_implies_equality_l310_31012


namespace r_n_m_smallest_m_for_r_2006_l310_31041

def euler_totient (n : ℕ) : ℕ := 
  n * (1 - (1 / 2)) * (1 - (1 / 17)) * (1 - (1 / 59))

def r (n m : ℕ) : ℕ :=
  m * euler_totient n

theorem r_n_m (n m : ℕ) : r n m = m * euler_totient n := 
  by sorry

theorem smallest_m_for_r_2006 (n m : ℕ) (h : n = 2006) (h2 : r n m = 841 * 928) : 
  ∃ m, r n m = 841^2 := 
  by sorry

end r_n_m_smallest_m_for_r_2006_l310_31041


namespace find_weekly_allowance_l310_31089

noncomputable def weekly_allowance (A : ℝ) : Prop :=
  let spent_at_arcade := (3/5) * A
  let remaining_after_arcade := A - spent_at_arcade
  let spent_at_toy_store := (1/3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store
  remaining_after_toy_store = 1.20

theorem find_weekly_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 4.50 := 
  sorry

end find_weekly_allowance_l310_31089


namespace vehicles_traveled_l310_31008

theorem vehicles_traveled (V : ℕ)
  (h1 : 40 * V = 800 * 100000000) : 
  V = 2000000000 := 
sorry

end vehicles_traveled_l310_31008


namespace problem_sum_congruent_mod_11_l310_31052

theorem problem_sum_congruent_mod_11 : 
  (2 + 333 + 5555 + 77777 + 999999 + 11111111 + 222222222) % 11 = 3 := 
by
  -- Proof needed here
  sorry

end problem_sum_congruent_mod_11_l310_31052


namespace usual_travel_time_l310_31051

theorem usual_travel_time
  (S : ℝ) (T : ℝ) 
  (h0 : S > 0)
  (h1 : (S / T) = (4 / 5 * S / (T + 6))) : 
  T = 30 :=
by sorry

end usual_travel_time_l310_31051


namespace four_students_same_acquaintances_l310_31054

theorem four_students_same_acquaintances
  (students : Finset ℕ)
  (acquainted : ∀ s ∈ students, (students \ {s}).card ≥ 68)
  (count : students.card = 102) :
  ∃ n, ∃ cnt, cnt ≥ 4 ∧ (∃ S, S ⊆ students ∧ S.card = cnt ∧ ∀ x ∈ S, (students \ {x}).card = n) :=
sorry

end four_students_same_acquaintances_l310_31054


namespace max_a_plus_2b_l310_31067

theorem max_a_plus_2b (a b : ℝ) (h : a^2 + 2 * b^2 = 1) : a + 2 * b ≤ Real.sqrt 3 := 
sorry

end max_a_plus_2b_l310_31067


namespace john_purchased_large_bottles_l310_31027

noncomputable def large_bottle_cost : ℝ := 1.75
noncomputable def small_bottle_cost : ℝ := 1.35
noncomputable def num_small_bottles : ℝ := 690
noncomputable def avg_price_paid : ℝ := 1.6163438256658595
noncomputable def total_small_cost : ℝ := num_small_bottles * small_bottle_cost
noncomputable def total_cost (L : ℝ) : ℝ := large_bottle_cost * L + total_small_cost
noncomputable def total_bottles (L : ℝ) : ℝ := L + num_small_bottles

theorem john_purchased_large_bottles : ∃ L : ℝ, 
  (total_cost L / total_bottles L = avg_price_paid) ∧ 
  (L = 1380) := 
sorry

end john_purchased_large_bottles_l310_31027


namespace polygon_sides_l310_31047

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_l310_31047


namespace smaller_solution_quadratic_equation_l310_31072

theorem smaller_solution_quadratic_equation :
  (∀ x : ℝ, x^2 + 7 * x - 30 = 0 → x = -10 ∨ x = 3) → -10 = min (-10) 3 :=
by
  sorry

end smaller_solution_quadratic_equation_l310_31072


namespace jill_first_show_length_l310_31029

theorem jill_first_show_length : 
  ∃ (x : ℕ), (x + 4 * x = 150) ∧ (x = 30) :=
sorry

end jill_first_show_length_l310_31029


namespace number_multiplies_p_plus_1_l310_31065

theorem number_multiplies_p_plus_1 (p q x : ℕ) 
  (hp : 1 < p) (hq : 1 < q)
  (hEq : x * (p + 1) = 25 * (q + 1))
  (hSum : p + q = 40) :
  x = 325 :=
sorry

end number_multiplies_p_plus_1_l310_31065


namespace pipe_B_filling_time_l310_31077

theorem pipe_B_filling_time (T_B : ℝ) 
  (A_filling_time : ℝ := 10) 
  (combined_filling_time: ℝ := 20/3)
  (A_rate : ℝ := 1 / A_filling_time)
  (combined_rate : ℝ := 1 / combined_filling_time) : 
  1 / T_B = combined_rate - A_rate → T_B = 20 := by 
  sorry

end pipe_B_filling_time_l310_31077


namespace find_smallest_even_number_l310_31022

theorem find_smallest_even_number (x : ℕ) (h1 : 
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14)) = 424) : 
  x = 46 := 
by
  sorry

end find_smallest_even_number_l310_31022


namespace m_eq_half_l310_31003

theorem m_eq_half (m : ℝ) (h1 : m > 0) (h2 : ∀ x, (0 < x ∧ x < m) → (x * (x - 1) < 0))
  (h3 : ∃ x, (0 < x ∧ x < 1) ∧ ¬(0 < x ∧ x < m)) : m = 1 / 2 :=
sorry

end m_eq_half_l310_31003


namespace max_value_of_sequence_l310_31046

theorem max_value_of_sequence :
  ∃ a : ℕ → ℕ, (∀ i, 1 ≤ i ∧ i ≤ 101 → 0 < a i) →
              (∀ i, 1 ≤ i ∧ i < 101 → (a i + 1) % a (i + 1) = 0) →
              (a 102 = a 1) →
              (∀ n, (1 ≤ n ∧ n ≤ 101) → a n ≤ 201) :=
by
  sorry

end max_value_of_sequence_l310_31046


namespace initial_deposit_l310_31032

variable (P R : ℝ)

theorem initial_deposit (h1 : P + (P * R * 3) / 100 = 11200)
                       (h2 : P + (P * (R + 2) * 3) / 100 = 11680) :
  P = 8000 :=
by
  sorry

end initial_deposit_l310_31032


namespace area_of_triangle_from_line_l310_31050

-- Define the conditions provided in the problem
def line_eq (B : ℝ) (x y : ℝ) := B * x + 9 * y = 18
def B_val := (36 : ℝ)

theorem area_of_triangle_from_line (B : ℝ) (hB : B = B_val) : 
  (∃ C : ℝ, C = 1 / 2) := by
  sorry

end area_of_triangle_from_line_l310_31050


namespace geometric_sequence_common_ratio_l310_31000

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {S : ℕ → ℝ} (q : ℝ) 
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  
  (h_condition : ∀ n : ℕ+, S (2 * n) / S n < 5) :
  0 < q ∧ q ≤ 1 :=
sorry

end geometric_sequence_common_ratio_l310_31000


namespace smallest_t_circle_sin_l310_31074

theorem smallest_t_circle_sin (t : ℝ) (h0 : 0 ≤ t) (h : ∀ θ, 0 ≤ θ ∧ θ ≤ t → ∃ k : ℤ, θ = (π/2 + 2 * π * k) ∨ θ = (3 * π / 2 + 2 * π * k)) : t = π :=
by {
  sorry
}

end smallest_t_circle_sin_l310_31074


namespace circles_condition_l310_31004

noncomputable def circles_intersect_at (p1 p2 : ℝ × ℝ) (m c : ℝ) : Prop :=
  p1 = (1, 3) ∧ p2 = (m, 1) ∧ (∃ (x y : ℝ), (x - y + c / 2 = 0) ∧ 
    (p1.1 - x)^2 + (p1.2 - y)^2 = (p2.1 - x)^2 + (p2.2 - y)^2)

theorem circles_condition (m c : ℝ) (h : circles_intersect_at (1, 3) (m, 1) m c) : m + c = 3 :=
sorry

end circles_condition_l310_31004


namespace minimum_value_expr_l310_31006

theorem minimum_value_expr (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  (1 + (1 / m)) * (1 + (1 / n)) = 9 :=
sorry

end minimum_value_expr_l310_31006


namespace contestant_score_l310_31059

theorem contestant_score (highest_score lowest_score : ℕ) (average_score : ℕ)
  (h_hs : highest_score = 86)
  (h_ls : lowest_score = 45)
  (h_avg : average_score = 76) :
  (76 * 9 - 86 - 45) / 7 = 79 := 
by 
  sorry

end contestant_score_l310_31059


namespace solution_set_inequality_l310_31010

noncomputable def solution_set (x : ℝ) : Prop :=
  (2 * x - 1) / (x + 2) > 1

theorem solution_set_inequality :
  { x : ℝ | solution_set x } = { x : ℝ | x < -2 ∨ x > 3 } := by
  sorry

end solution_set_inequality_l310_31010


namespace find_larger_number_l310_31045

theorem find_larger_number 
  (x y : ℤ)
  (h1 : x + y = 37)
  (h2 : x - y = 5) : max x y = 21 := 
sorry

end find_larger_number_l310_31045


namespace lcm_12_18_l310_31087

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l310_31087


namespace part_one_solution_set_part_two_lower_bound_l310_31053

def f (x a b : ℝ) : ℝ := abs (x - a) + abs (x + b)

-- Part (I)
theorem part_one_solution_set (a b x : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (f x a b ≤ 5) ↔ -3 ≤ x ∧ x ≤ 2 := by
  rw [h1, h2]
  sorry

-- Part (II)
theorem part_two_lower_bound (a b x : ℝ) (h : a > 0) (h' : b > 0) (h'' : a + 4 * b = 2 * a * b) :
  f x a b ≥ 9 / 2 := by
  sorry

end part_one_solution_set_part_two_lower_bound_l310_31053


namespace behemoth_and_rita_finish_ice_cream_l310_31024

theorem behemoth_and_rita_finish_ice_cream (x y : ℝ) (h : 3 * x + 2 * y = 1) : 3 * (x + y) ≥ 1 :=
by
  sorry

end behemoth_and_rita_finish_ice_cream_l310_31024


namespace part1_part2_part3_l310_31033

-- Part 1: Proving a₁ for given a₃, p, and q
theorem part1 (a : ℕ → ℝ) (p q : ℝ) (h1 : p = (1/2)) (h2 : q = 2) 
  (h3 : a 3 = 41 / 20) (h4 : ∀ n, a (n + 1) = p * a n + q / a n) :
  a 1 = 1 ∨ a 1 = 4 := 
sorry

-- Part 2: Finding the sum Sₙ of the first n terms given a₁ and p * q = 0
theorem part2 (a : ℕ → ℝ) (p q : ℝ) (h1 : a 1 = 5) (h2 : p * q = 0) 
  (h3 : ∀ n, a (n + 1) = p * a n + q / a n) 
  (S : ℕ → ℝ) (n : ℕ) :
    S n = (25 * n + q * n + q - 25) / 10 ∨ 
    S n = (25 * n + q * n) / 10 ∨ 
    S n = (5 * (p^n - 1)) / (p - 1) ∨ 
    S n = 5 * n :=
sorry

-- Part 3: Proving the range of p given a₁, q and that the sequence is monotonically decreasing
theorem part3 (a : ℕ → ℝ) (p q : ℝ) (h1 : a 1 = 2) (h2 : q = 1) 
  (h3 : ∀ n, a (n + 1) = p * a n + q / a n) 
  (h4 : ∀ n, a (n + 1) < a n) :
  1/2 < p ∧ p < 3/4 :=
sorry

end part1_part2_part3_l310_31033


namespace fraction_of_students_paired_l310_31017

theorem fraction_of_students_paired {t s : ℕ} 
  (h1 : t / 4 = s / 3) : 
  (t / 4 + s / 3) / (t + s) = 2 / 7 := by sorry

end fraction_of_students_paired_l310_31017


namespace problem_solution_l310_31093

theorem problem_solution
  (a b c : ℝ)
  (habc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
by
  sorry

end problem_solution_l310_31093


namespace max_ab_l310_31019

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 8) : 
  ab ≤ 8 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 8 ∧ ab = 8 :=
by
  sorry

end max_ab_l310_31019


namespace calculate_fraction_l310_31088

theorem calculate_fraction : (10^9 + 10^6) / (3 * 10^4) = 100100 / 3 := by
  sorry

end calculate_fraction_l310_31088


namespace largest_negative_integer_l310_31007

theorem largest_negative_integer :
  ∃ (n : ℤ), (∀ m : ℤ, m < 0 → m ≤ n) ∧ n = -1 := by
  sorry

end largest_negative_integer_l310_31007


namespace incorrect_option_B_l310_31009

-- Definitions of the given conditions
def optionA (a : ℝ) : Prop := (8 * a = 8 * a)
def optionB (a : ℝ) : Prop := (a - (0.08 * a) = 8 * a)
def optionC (a : ℝ) : Prop := (8 * a = 8 * a)
def optionD (a : ℝ) : Prop := (a * 8 = 8 * a)

-- The statement to be proved
theorem incorrect_option_B (a : ℝ) : 
  optionA a ∧ ¬optionB a ∧ optionC a ∧ optionD a := 
by
  sorry

end incorrect_option_B_l310_31009


namespace prod_of_extrema_l310_31057

noncomputable def f (x k : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem prod_of_extrema (k : ℝ) (h : ∀ x : ℝ, f x k ≥ 0 ∧ f x k ≤ 1 + (k - 1) / 3) :
  (∀ x : ℝ, f x k ≤ (k + 2) / 3) ∧ (∀ x : ℝ, f x k ≥ 1) → 
  (∃ φ ψ : ℝ, φ = 1 ∧ ψ = (k + 2) / 3 ∧ ∀ x y : ℝ, f x k = φ → f y k = ψ) → 
  (∃ φ ψ : ℝ, φ * ψ = (k + 2) / 3) :=
sorry

end prod_of_extrema_l310_31057


namespace students_taking_history_but_not_statistics_l310_31039

-- Definitions based on conditions
def T : Nat := 150
def H : Nat := 58
def S : Nat := 42
def H_union_S : Nat := 95

-- Statement to prove
theorem students_taking_history_but_not_statistics : H - (H + S - H_union_S) = 53 :=
by
  sorry

end students_taking_history_but_not_statistics_l310_31039


namespace range_of_t_l310_31066

theorem range_of_t (a b c : ℝ) (t : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_inequality : ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → (1 / a^2) + (4 / b^2) + (t / c^2) ≥ 0) :
  t ≥ -9 :=
sorry

end range_of_t_l310_31066


namespace initial_yards_lost_l310_31064

theorem initial_yards_lost (x : ℤ) (h : -x + 7 = 2) : x = 5 := by
  sorry

end initial_yards_lost_l310_31064


namespace contrapositive_proposition_l310_31013

theorem contrapositive_proposition :
  (∀ x : ℝ, (x^2 < 4 → -2 < x ∧ x < 2)) ↔ (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 2 → x^2 ≥ 4)) :=
by
  sorry

end contrapositive_proposition_l310_31013


namespace angle_PQR_correct_l310_31096

-- Define the points and angles
variables {R P Q S : Type*}
variables (angle_RSQ angle_QSP angle_RQS angle_PQS : ℝ)

-- Define the conditions
def condition1 : Prop := true  -- RSP is a straight line implicitly means angle_RSQ + angle_QSP = 180
def condition2 : Prop := angle_QSP = 70
def condition3 (RS SQ : Type*) : Prop := true  -- Triangle RSQ is isosceles with RS = SQ
def condition4 (PS SQ : Type*) : Prop := true  -- Triangle PSQ is isosceles with PS = SQ

-- Define the isosceles triangle properties
def angle_RSQ_def : ℝ := 180 - angle_QSP
def angle_RQS_def : ℝ := 0.5 * (180 - angle_RSQ)
def angle_PQS_def : ℝ := 0.5 * (180 - angle_QSP)

-- Prove the main statement
theorem angle_PQR_correct : 
  (angle_RSQ = 110) →
  (angle_RQS = 35) →
  (angle_PQS = 55) →
  (angle_PQR : ℝ) = angle_PQS + angle_RQS :=
sorry

end angle_PQR_correct_l310_31096


namespace problem_solution_l310_31025

theorem problem_solution :
  ∀ (a b c d : ℝ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
    (a^2 = 7 ∨ a^2 = 8) →
    (b^2 = 7 ∨ b^2 = 8) →
    (c^2 = 7 ∨ c^2 = 8) →
    (d^2 = 7 ∨ d^2 = 8) →
    a^2 + b^2 + c^2 + d^2 = 30 :=
by sorry

end problem_solution_l310_31025


namespace yeast_counting_procedure_l310_31099

def yeast_counting_conditions (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool) : Prop :=
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true

theorem yeast_counting_procedure :
  ∀ (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool),
  yeast_counting_conditions counting_method shake_test_tube_needed dilution_needed →
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true :=
by
  intros counting_method shake_test_tube_needed dilution_needed h_condition
  exact h_condition

end yeast_counting_procedure_l310_31099


namespace value_of_k_l310_31084

theorem value_of_k :
  (∀ x : ℝ, x ^ 2 - x - 2 > 0 → 2 * x ^ 2 + (5 + 2 * k) * x + 5 * k < 0 → x = -2) ↔ -3 ≤ k ∧ k < 2 :=
sorry

end value_of_k_l310_31084


namespace find_remainder_l310_31040

noncomputable def remainder_expr_division (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) : ℂ :=
  1 - β

theorem find_remainder (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) :
  ∃ r, (x^45 + x^34 + x^23 + x^12 + 1) % (x^4 + x^3 + x^2 + x + 1) = r ∧ r = remainder_expr_division β hβ :=
sorry

end find_remainder_l310_31040


namespace rectangle_area_l310_31086

theorem rectangle_area (x w : ℝ) (h₁ : 3 * w = 3 * w) (h₂ : x^2 = 9 * w^2 + w^2) : 
  (3 * w) * w = (3 / 10) * x^2 := 
by
  sorry

end rectangle_area_l310_31086


namespace problem_statement_l310_31061

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - 2 / x^2 + a / x

theorem problem_statement (a : ℝ) (k : ℝ) : 
  0 < a ∧ a ≤ 4 →
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 →
  |f x1 a - f x2 a| > k * |x1 - x2|) ↔
  k ≤ 2 - a^3 / 108 :=
by
  sorry

end problem_statement_l310_31061


namespace balance_scale_cereal_l310_31082

def scales_are_balanced (left_pan : ℕ) (right_pan : ℕ) : Prop :=
  left_pan = right_pan

theorem balance_scale_cereal (inaccurate_scales : ℕ → ℕ → Prop)
  (cereal : ℕ)
  (correct_weight : ℕ) :
  (∀ left_pan right_pan, inaccurate_scales left_pan right_pan → left_pan = right_pan) →
  (cereal / 2 = 1) →
  true :=
  sorry

end balance_scale_cereal_l310_31082


namespace greatest_integer_2e_minus_5_l310_31060

noncomputable def e : ℝ := 2.718

theorem greatest_integer_2e_minus_5 : ⌊2 * e - 5⌋ = 0 :=
by
  -- This is a placeholder for the actual proof. 
  sorry

end greatest_integer_2e_minus_5_l310_31060


namespace rahul_savings_is_correct_l310_31048

def Rahul_Savings_Problem : Prop :=
  ∃ (NSC PPF : ℝ), 
    (1/3) * NSC = (1/2) * PPF ∧ 
    NSC + PPF = 180000 ∧ 
    PPF = 72000

theorem rahul_savings_is_correct : Rahul_Savings_Problem :=
  sorry

end rahul_savings_is_correct_l310_31048


namespace music_player_winner_l310_31035

theorem music_player_winner (n : ℕ) (h1 : ∀ k, k % n = 0 → k = 35) (h2 : 35 % 7 = 0) (h3 : 35 % n = 0) (h4 : n ≠ 1) (h5 : n ≠ 7) (h6 : n ≠ 35) : n = 5 := 
sorry

end music_player_winner_l310_31035


namespace value_three_in_range_of_g_l310_31094

theorem value_three_in_range_of_g (a : ℝ) : ∀ (a : ℝ), ∃ (x : ℝ), x^2 + a * x + 1 = 3 :=
by
  sorry

end value_three_in_range_of_g_l310_31094


namespace angle_between_vectors_l310_31092

noncomputable def vec_a : ℝ × ℝ := (-2 * Real.sqrt 3, 2)
noncomputable def vec_b : ℝ × ℝ := (1, - Real.sqrt 3)

-- Define magnitudes
noncomputable def mag_a : ℝ := Real.sqrt ((-2 * Real.sqrt 3) ^ 2 + 2^2)
noncomputable def mag_b : ℝ := Real.sqrt (1^2 + (- Real.sqrt 3) ^ 2)

-- Define the dot product
noncomputable def dot_product : ℝ := (-2 * Real.sqrt 3) * 1 + 2 * (- Real.sqrt 3)

-- Define cosine of the angle theta
-- We use mag_a and mag_b defined above
noncomputable def cos_theta : ℝ := dot_product / (mag_a * mag_b)

-- Define the angle theta, within the range [0, π]
noncomputable def theta : ℝ := Real.arccos cos_theta

-- The expected result is θ = 5π / 6
theorem angle_between_vectors : theta = (5 * Real.pi) / 6 :=
by
  sorry

end angle_between_vectors_l310_31092


namespace floating_time_l310_31016

theorem floating_time (boat_with_current: ℝ) (boat_against_current: ℝ) (distance: ℝ) (time: ℝ) : 
boat_with_current = 28 ∧ boat_against_current = 24 ∧ distance = 20 ∧ 
time = distance / ((boat_with_current - boat_against_current) / 2) → 
time = 10 := by
  sorry

end floating_time_l310_31016


namespace find_f_729_l310_31042

variable (f : ℕ+ → ℕ+) -- Define the function f on the positive integers.

-- Conditions of the problem.
axiom h1 : ∀ n : ℕ+, f (f n) = 3 * n
axiom h2 : ∀ n : ℕ+, f (3 * n + 1) = 3 * n + 2 

-- Proof statement.
theorem find_f_729 : f 729 = 729 :=
by
  sorry -- Placeholder for the proof.

end find_f_729_l310_31042


namespace AgathaAdditionalAccessories_l310_31037

def AgathaBudget : ℕ := 250
def Frame : ℕ := 85
def FrontWheel : ℕ := 35
def RearWheel : ℕ := 40
def Seat : ℕ := 25
def HandlebarTape : ℕ := 15
def WaterBottleCage : ℕ := 10
def BikeLock : ℕ := 20
def FutureExpenses : ℕ := 10

theorem AgathaAdditionalAccessories :
  AgathaBudget - (Frame + FrontWheel + RearWheel + Seat + HandlebarTape + WaterBottleCage + BikeLock + FutureExpenses) = 10 := by
  sorry

end AgathaAdditionalAccessories_l310_31037


namespace Katie_marble_count_l310_31085

theorem Katie_marble_count :
  ∀ (pink_marbles orange_marbles purple_marbles total_marbles : ℕ),
  pink_marbles = 13 →
  orange_marbles = pink_marbles - 9 →
  purple_marbles = 4 * orange_marbles →
  total_marbles = pink_marbles + orange_marbles + purple_marbles →
  total_marbles = 33 :=
by
  intros pink_marbles orange_marbles purple_marbles total_marbles
  intros hpink horange hpurple htotal
  sorry

end Katie_marble_count_l310_31085


namespace students_in_diligence_before_transfer_l310_31070

theorem students_in_diligence_before_transfer (D I P : ℕ)
  (h_total : D + I + P = 75)
  (h_equal : D + 2 = I - 2 + 3 ∧ D + 2 = P - 3) :
  D = 23 :=
by
  sorry

end students_in_diligence_before_transfer_l310_31070


namespace price_per_butterfly_l310_31034

theorem price_per_butterfly (jars : ℕ) (caterpillars_per_jar : ℕ) (fail_percentage : ℝ) (total_money : ℝ) (price : ℝ) :
  jars = 4 →
  caterpillars_per_jar = 10 →
  fail_percentage = 0.40 →
  total_money = 72 →
  price = 3 :=
by
  intros h_jars h_caterpillars h_fail_percentage h_total_money
  -- Full proof here
  sorry

end price_per_butterfly_l310_31034


namespace geometric_series_S6_value_l310_31079

theorem geometric_series_S6_value (S : ℕ → ℝ) (S3 : S 3 = 3) (S9_minus_S6 : S 9 - S 6 = 12) : 
  S 6 = 9 :=
by
  sorry

end geometric_series_S6_value_l310_31079


namespace complex_sum_is_2_l310_31063

theorem complex_sum_is_2 
  (a b c d e f : ℂ) 
  (hb : b = 4) 
  (he : e = 2 * (-a - c)) 
  (hr : a + c + e = 0) 
  (hi : b + d + f = 6) 
  : d + f = 2 := 
  by
  sorry

end complex_sum_is_2_l310_31063


namespace tiger_initial_leaps_behind_l310_31014

theorem tiger_initial_leaps_behind (tiger_leap_distance deer_leap_distance tiger_leaps_per_minute deer_leaps_per_minute total_distance_to_catch initial_leaps_behind : ℕ) 
  (h1 : tiger_leap_distance = 8) 
  (h2 : deer_leap_distance = 5) 
  (h3 : tiger_leaps_per_minute = 5) 
  (h4 : deer_leaps_per_minute = 4) 
  (h5 : total_distance_to_catch = 800) :
  initial_leaps_behind = 40 := 
by
  -- Leaving proof body incomplete as it is not required
  sorry

end tiger_initial_leaps_behind_l310_31014


namespace sonny_received_45_boxes_l310_31023

def cookies_received (cookies_given_brother : ℕ) (cookies_given_sister : ℕ) (cookies_given_cousin : ℕ) (cookies_left : ℕ) : ℕ :=
  cookies_given_brother + cookies_given_sister + cookies_given_cousin + cookies_left

theorem sonny_received_45_boxes :
  cookies_received 12 9 7 17 = 45 :=
by
  sorry

end sonny_received_45_boxes_l310_31023


namespace primes_unique_l310_31001

-- Let's define that p, q, r are prime numbers, and define the main conditions.
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem primes_unique (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (div1 : (p^4 - 1) % (q * r) = 0)
  (div2 : (q^4 - 1) % (p * r) = 0)
  (div3 : (r^4 - 1) % (p * q) = 0) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ 
  (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end primes_unique_l310_31001


namespace part1_part2_part3_l310_31044

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  abs (x^2 - 1) + x^2 + k * x

theorem part1 (h : 2 = 2) :
  (f (- (1 + Real.sqrt 3) /2) 2 = 0) ∧ (f (-1/2) 2 = 0) := by
  sorry

theorem part2 (h_alpha : 0 < α) (h_beta : α < β) (h_beta2 : β < 2) (h_f_alpha : f α k = 0) (h_f_beta : f β k = 0) :
  -7/2 < k ∧ k < -1 := by
  sorry

theorem part3 (h_alpha : 0 < α) (h_alpha1 : α ≤ 1) (h_beta1 : 1 < β) (h_beta2 : β < 2) (h1 : k = - 1 / α) (h2 : 2 * β^2 + k * β - 1 = 0) :
  1/α + 1/β < 4 := by
  sorry

end part1_part2_part3_l310_31044


namespace knights_and_liars_solution_l310_31005

-- Definitions of each person's statement as predicates
def person1_statement (liar : ℕ → Prop) : Prop := liar 2 ∧ liar 3 ∧ liar 4 ∧ liar 5 ∧ liar 6
def person2_statement (liar : ℕ → Prop) : Prop := liar 1 ∧ ∀ i, i ≠ 1 → ¬ liar i
def person3_statement (liar : ℕ → Prop) : Prop := liar 4 ∧ liar 5 ∧ liar 6 ∧ ¬ liar 3 ∧ ¬ liar 2 ∧ ¬ liar 1
def person4_statement (liar : ℕ → Prop) : Prop := liar 1 ∧ liar 2 ∧ liar 3 ∧ ∀ i, i > 3 → ¬ liar i
def person5_statement (liar : ℕ → Prop) : Prop := liar 6 ∧ ∀ i, i ≠ 6 → ¬ liar i
def person6_statement (liar : ℕ → Prop) : Prop := liar 5 ∧ ∀ i, i ≠ 5 → ¬ liar i

-- Definition of a knight and a liar
def is_knight (statement : Prop) : Prop := statement
def is_liar (statement : Prop) : Prop := ¬ statement

-- Defining the theorem
theorem knights_and_liars_solution (knight liar : ℕ → Prop) : 
  is_liar (person1_statement liar) ∧ 
  is_knight (person2_statement liar) ∧ 
  is_liar (person3_statement liar) ∧ 
  is_liar (person4_statement liar) ∧ 
  is_knight (person5_statement liar) ∧ 
  is_liar (person6_statement liar) :=
by
  sorry

end knights_and_liars_solution_l310_31005


namespace find_constant_c_l310_31098

theorem find_constant_c (c : ℝ) :
  (∀ x y : ℝ, x + y = c ∧ y - (2 + 5) / 2 = x - (8 + 11) / 2) →
  (c = 13) :=
by
  sorry

end find_constant_c_l310_31098


namespace unique_solution_inequality_l310_31021

theorem unique_solution_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a * x + a ∧ x^2 - a * x + a ≤ 1) → a = 2 :=
by
  sorry

end unique_solution_inequality_l310_31021


namespace part1_part2_l310_31055

theorem part1 (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a * b > 0) : a + b = 8 ∨ a + b = -8 :=
sorry

theorem part2 (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h4 : |a + b| = a + b) : a - b = 4 ∨ a - b = 8 :=
sorry

end part1_part2_l310_31055


namespace inequality_condition_l310_31018

theorem inequality_condition {a b x y : ℝ} (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) : 
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (a^2 / x) + (b^2 / y) = ((a + b)^2 / (x + y)) ↔ (x / y) = (a / b) :=
sorry

end inequality_condition_l310_31018


namespace num_members_in_league_l310_31076

-- Definitions based on conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def shorts_cost : ℕ := tshirt_cost
def total_cost_per_member : ℕ := 2 * (sock_cost + tshirt_cost + shorts_cost)
def total_league_cost : ℕ := 4719

-- Theorem statement
theorem num_members_in_league : (total_league_cost / total_cost_per_member) = 74 :=
by
  sorry

end num_members_in_league_l310_31076
