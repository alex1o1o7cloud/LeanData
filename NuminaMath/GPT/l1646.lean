import Mathlib

namespace NUMINAMATH_GPT_second_acid_solution_percentage_l1646_164617

-- Definitions of the problem conditions
def P : ℝ := 75
def V₁ : ℝ := 4
def C₁ : ℝ := 0.60
def V₂ : ℝ := 20
def C₂ : ℝ := 0.72

/-
Given that 4 liters of a 60% acid solution are mixed with a certain volume of another acid solution
to get 20 liters of 72% solution, prove that the percentage of the second acid solution must be 75%.
-/
theorem second_acid_solution_percentage
  (x : ℝ) -- volume of the second acid solution
  (P_percent : ℝ := P) -- percentage of the second acid solution
  (h1 : V₁ + x = V₂) -- condition on volume
  (h2 : C₁ * V₁ + (P_percent / 100) * x = C₂ * V₂) -- condition on acid content
  : P_percent = P := 
by
  -- Moving forward with proof the lean proof
  sorry

end NUMINAMATH_GPT_second_acid_solution_percentage_l1646_164617


namespace NUMINAMATH_GPT_value_of_a2_l1646_164669

theorem value_of_a2 
  (a1 a2 a3 : ℝ)
  (h_seq : ∃ d : ℝ, (-8) = -8 + d * 0 ∧ a1 = -8 + d * 1 ∧ 
                     a2 = -8 + d * 2 ∧ a3 = -8 + d * 3 ∧ 
                     10 = -8 + d * 4) :
  a2 = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_a2_l1646_164669


namespace NUMINAMATH_GPT_stratified_sampling_l1646_164603

theorem stratified_sampling
  (ratio_first : ℕ)
  (ratio_second : ℕ)
  (ratio_third : ℕ)
  (sample_size : ℕ)
  (h_ratio : ratio_first = 3 ∧ ratio_second = 4 ∧ ratio_third = 3)
  (h_sample_size : sample_size = 50) :
  (ratio_second * sample_size) / (ratio_first + ratio_second + ratio_third) = 20 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l1646_164603


namespace NUMINAMATH_GPT_number_of_correct_propositions_l1646_164666

theorem number_of_correct_propositions : 
    (∀ a b : ℝ, a < b → ¬ (a^2 < b^2)) ∧ 
    (∀ a : ℝ, (∀ x : ℝ, |x + 1| + |x - 1| ≥ a ↔ a ≤ 2)) ∧ 
    (¬ (∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0) → 
    1 = 1 := 
by
  sorry

end NUMINAMATH_GPT_number_of_correct_propositions_l1646_164666


namespace NUMINAMATH_GPT_part1_part2_l1646_164608

-- Define the conditions p and q
def p (a x : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := (x - 2) * (x - 4) < 0 ∧ (x - 3) * (x - 5) > 0

-- Problem Part 1: Prove that if a = 1 and p ∧ q is true, then 2 < x < 3
theorem part1 (x : ℝ) : p 1 x ∧ q x → 2 < x ∧ x < 3 :=
by
  intro h
  sorry

-- Problem Part 2: Prove that if p is a necessary but not sufficient condition for q, then 1 ≤ a ≤ 2
theorem part2 (a : ℝ) : (∀ x, q x → p a x) ∧ (∃ x, p a x ∧ ¬q x) → 1 ≤ a ∧ a ≤ 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_part1_part2_l1646_164608


namespace NUMINAMATH_GPT_modulo_residue_l1646_164601

theorem modulo_residue:
  (247 + 5 * 40 + 7 * 143 + 4 * (2^3 - 1)) % 13 = 7 :=
by
  sorry

end NUMINAMATH_GPT_modulo_residue_l1646_164601


namespace NUMINAMATH_GPT_interest_rate_calc_l1646_164625

theorem interest_rate_calc
  (P : ℝ) (A : ℝ) (T : ℝ) (SI : ℝ := A - P)
  (R : ℝ := (SI * 100) / (P * T))
  (hP : P = 750)
  (hA : A = 950)
  (hT : T = 5) :
  R = 5.33 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_calc_l1646_164625


namespace NUMINAMATH_GPT_amount_received_by_sam_l1646_164642

def P : ℝ := 15000
def r : ℝ := 0.10
def n : ℝ := 2
def t : ℝ := 1

noncomputable def compoundInterest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem amount_received_by_sam : compoundInterest P r n t = 16537.50 := by
  sorry

end NUMINAMATH_GPT_amount_received_by_sam_l1646_164642


namespace NUMINAMATH_GPT_shaded_area_l1646_164627

-- Define the points as per the problem
structure Point where
  x : ℝ
  y : ℝ

@[simp]
def A : Point := ⟨0, 0⟩
@[simp]
def B : Point := ⟨0, 7⟩
@[simp]
def C : Point := ⟨7, 7⟩
@[simp]
def D : Point := ⟨7, 0⟩
@[simp]
def E : Point := ⟨7, 0⟩
@[simp]
def F : Point := ⟨14, 0⟩
@[simp]
def G : Point := ⟨10.5, 7⟩

-- Define function for area of a triangle given three points
def triangle_area (P Q R : Point) : ℝ :=
  0.5 * abs ((P.x - R.x) * (Q.y - P.y) - (P.x - Q.x) * (R.y - P.y))

-- The theorem stating the area of the shaded region
theorem shaded_area : triangle_area D G H - triangle_area D E H = 24.5 := by
  sorry

end NUMINAMATH_GPT_shaded_area_l1646_164627


namespace NUMINAMATH_GPT_power_function_quadrant_IV_l1646_164651

theorem power_function_quadrant_IV (a : ℝ) (h : a ∈ ({-1, 1/2, 2, 3} : Set ℝ)) :
  ∀ x : ℝ, x * x^a ≠ -x * (-x^a) := sorry

end NUMINAMATH_GPT_power_function_quadrant_IV_l1646_164651


namespace NUMINAMATH_GPT_inequality_proof_l1646_164668

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 4) : 
  1 / a + 4 / b ≥ 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1646_164668


namespace NUMINAMATH_GPT_volume_of_pyramid_l1646_164655

variables (a b c : ℝ)

def triangle_face1 (a b : ℝ) : Prop := 1/2 * a * b = 1.5
def triangle_face2 (b c : ℝ) : Prop := 1/2 * b * c = 2
def triangle_face3 (c a : ℝ) : Prop := 1/2 * c * a = 6

theorem volume_of_pyramid (h1 : triangle_face1 a b) (h2 : triangle_face2 b c) (h3 : triangle_face3 c a) :
  1/3 * a * b * c = 2 :=
sorry

end NUMINAMATH_GPT_volume_of_pyramid_l1646_164655


namespace NUMINAMATH_GPT_find_distance_from_origin_l1646_164675

-- Define the conditions as functions
def point_distance_from_x_axis (y : ℝ) : Prop := abs y = 15
def distance_from_point (x y : ℝ) (x₀ y₀ : ℝ) (d : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = d^2

-- Define the proof problem
theorem find_distance_from_origin (x y : ℝ) (n : ℝ) (hx : x = 2 + Real.sqrt 105) (hy : point_distance_from_x_axis y) (hx_gt : x > 2) (hdist : distance_from_point x y 2 7 13) :
  n = Real.sqrt (334 + 4 * Real.sqrt 105) :=
sorry

end NUMINAMATH_GPT_find_distance_from_origin_l1646_164675


namespace NUMINAMATH_GPT_yen_to_usd_conversion_l1646_164637

theorem yen_to_usd_conversion
  (cost_of_souvenir : ℕ)
  (service_charge : ℕ)
  (conversion_rate : ℕ)
  (total_cost_in_yen : ℕ)
  (usd_equivalent : ℚ)
  (h1 : cost_of_souvenir = 340)
  (h2 : service_charge = 25)
  (h3 : conversion_rate = 115)
  (h4 : total_cost_in_yen = cost_of_souvenir + service_charge)
  (h5 : usd_equivalent = (total_cost_in_yen : ℚ) / conversion_rate) :
  total_cost_in_yen = 365 ∧ usd_equivalent = 3.17 :=
by
  sorry

end NUMINAMATH_GPT_yen_to_usd_conversion_l1646_164637


namespace NUMINAMATH_GPT_volume_in_cubic_yards_l1646_164604

-- Define the conditions
def volume_in_cubic_feet : ℕ := 162
def cubic_feet_per_cubic_yard : ℕ := 27

-- Problem statement in Lean 4
theorem volume_in_cubic_yards : volume_in_cubic_feet / cubic_feet_per_cubic_yard = 6 := 
  by
    sorry

end NUMINAMATH_GPT_volume_in_cubic_yards_l1646_164604


namespace NUMINAMATH_GPT_fraction_pow_zero_l1646_164649

theorem fraction_pow_zero
  (a : ℤ) (b : ℤ)
  (h_a : a = -325123789)
  (h_b : b = 59672384757348)
  (h_nonzero_num : a ≠ 0)
  (h_nonzero_denom : b ≠ 0) :
  (a / b : ℚ) ^ 0 = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_pow_zero_l1646_164649


namespace NUMINAMATH_GPT_AB_plus_C_eq_neg8_l1646_164696

theorem AB_plus_C_eq_neg8 (A B C : ℤ) (g : ℝ → ℝ)
(hf : ∀ x > 3, g x > 0.5)
(heq : ∀ x, g x = x^2 / (A * x^2 + B * x + C))
(hasymp_vert : ∀ x, (A * (x + 3) * (x - 2) = 0 → x = -3 ∨ x = 2))
(hasymp_horiz : (1 : ℝ) / (A : ℝ) < 1) :
A + B + C = -8 :=
sorry

end NUMINAMATH_GPT_AB_plus_C_eq_neg8_l1646_164696


namespace NUMINAMATH_GPT_find_B_l1646_164680

def is_prime_203B21 (B : ℕ) : Prop :=
  2 ≤ B ∧ B < 10 ∧ Prime (200000 + 3000 + 100 * B + 20 + 1)

theorem find_B : ∃ B, is_prime_203B21 B ∧ ∀ B', is_prime_203B21 B' → B' = 5 := by
  sorry

end NUMINAMATH_GPT_find_B_l1646_164680


namespace NUMINAMATH_GPT_find_a2016_l1646_164683

theorem find_a2016 (S : ℕ → ℕ)
  (a : ℕ → ℤ)
  (h₁ : S 1 = 6)
  (h₂ : S 2 = 4)
  (h₃ : ∀ n, S n > 0)
  (h₄ : ∀ n, (S (2 * n - 1))^2 = S (2 * n) * S (2 * n + 2))
  (h₅ : ∀ n, 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1))
  : a 2016 = -1009 := 
  sorry

end NUMINAMATH_GPT_find_a2016_l1646_164683


namespace NUMINAMATH_GPT_solution_x_y_l1646_164609

theorem solution_x_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
    x^4 - 6 * x^2 + 1 = 7 * 2^y ↔ (x = 3 ∧ y = 2) :=
by {
    sorry
}

end NUMINAMATH_GPT_solution_x_y_l1646_164609


namespace NUMINAMATH_GPT_sum_eq_neg_20_div_3_l1646_164682
-- Import the necessary libraries

-- The main theoretical statement
theorem sum_eq_neg_20_div_3
    (a b c d : ℝ)
    (h : a + 2 = b + 4 ∧ b + 4 = c + 6 ∧ c + 6 = d + 8 ∧ d + 8 = a + b + c + d + 10) :
    a + b + c + d = -20 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_eq_neg_20_div_3_l1646_164682


namespace NUMINAMATH_GPT_value_of_a_l1646_164678

-- Definitions based on conditions
def cond1 (a : ℝ) := |a| - 1 = 0
def cond2 (a : ℝ) := a + 1 ≠ 0

-- The main proof problem
theorem value_of_a (a : ℝ) : (cond1 a ∧ cond2 a) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1646_164678


namespace NUMINAMATH_GPT_exists_power_of_two_with_consecutive_zeros_l1646_164610

theorem exists_power_of_two_with_consecutive_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ, ∃ a b : ℕ, ∃ m : ℕ, 2^n = a * 10^(m + k) + b ∧ 10^(k - 1) ≤ b ∧ b < 10^k ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 :=
sorry

end NUMINAMATH_GPT_exists_power_of_two_with_consecutive_zeros_l1646_164610


namespace NUMINAMATH_GPT_pictures_vertical_l1646_164692

theorem pictures_vertical (V H X : ℕ) (h1 : V + H + X = 30) (h2 : H = 15) (h3 : X = 5) : V = 10 := 
by 
  sorry

end NUMINAMATH_GPT_pictures_vertical_l1646_164692


namespace NUMINAMATH_GPT_avg_people_moving_to_florida_per_hour_l1646_164654

theorem avg_people_moving_to_florida_per_hour (people : ℕ) (days : ℕ) (hours_per_day : ℕ) 
  (h1 : people = 3000) (h2 : days = 5) (h3 : hours_per_day = 24) : 
  people / (days * hours_per_day) = 25 := by
  sorry

end NUMINAMATH_GPT_avg_people_moving_to_florida_per_hour_l1646_164654


namespace NUMINAMATH_GPT_least_froods_l1646_164645

theorem least_froods (n : ℕ) :
  (∃ n, n ≥ 1 ∧ (n * (n + 1)) / 2 > 20 * n) → (∃ n, n = 40) :=
by {
  sorry
}

end NUMINAMATH_GPT_least_froods_l1646_164645


namespace NUMINAMATH_GPT_locus_of_points_l1646_164656

-- Define points A and B
variable {A B : (ℝ × ℝ)}
-- Define constant d
variable {d : ℝ}

-- Definition of the distances
def distance_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem locus_of_points (A B : (ℝ × ℝ)) (d : ℝ) :
  ∀ M : (ℝ × ℝ), distance_sq M A - distance_sq M B = d ↔ 
  ∃ x : ℝ, ∃ y : ℝ, (M.1, M.2) = (x, y) ∧ 
  x = ((B.1 - A.1)^2 + d) / (2 * (B.1 - A.1)) :=
by
  sorry

end NUMINAMATH_GPT_locus_of_points_l1646_164656


namespace NUMINAMATH_GPT_greatest_integer_value_l1646_164630

theorem greatest_integer_value (x : ℤ) (h : 3 * |x| + 4 ≤ 19) : x ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_value_l1646_164630


namespace NUMINAMATH_GPT_fifth_term_of_geometric_sequence_l1646_164640

theorem fifth_term_of_geometric_sequence
  (a r : ℝ)
  (h1 : a * r^2 = 16)
  (h2 : a * r^6 = 2) : a * r^4 = 8 :=
sorry

end NUMINAMATH_GPT_fifth_term_of_geometric_sequence_l1646_164640


namespace NUMINAMATH_GPT_compute_f3_l1646_164629

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 4*n + 3 else 2*n + 1

theorem compute_f3 : f (f (f 3)) = 99 :=
by
  sorry

end NUMINAMATH_GPT_compute_f3_l1646_164629


namespace NUMINAMATH_GPT_olympic_rings_area_l1646_164613

theorem olympic_rings_area (d R r: ℝ) 
  (hyp_d : d = 12 * Real.sqrt 2) 
  (hyp_R : R = 11) 
  (hyp_r : r = 9) 
  (overlap_area : ∀ (i j : ℕ), i ≠ j → 592 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54): 
  592.0 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54 := 
by sorry

end NUMINAMATH_GPT_olympic_rings_area_l1646_164613


namespace NUMINAMATH_GPT_train_speed_without_stoppages_l1646_164623

theorem train_speed_without_stoppages 
  (distance_with_stoppages : ℝ)
  (avg_speed_with_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (distance_without_stoppages : ℝ)
  (avg_speed_without_stoppages : ℝ) :
  avg_speed_with_stoppages = 200 → 
  stoppage_time_per_hour = 20 / 60 →
  distance_without_stoppages = distance_with_stoppages * avg_speed_without_stoppages →
  distance_with_stoppages = avg_speed_with_stoppages →
  avg_speed_without_stoppages == 300 := 
by
  intros
  sorry

end NUMINAMATH_GPT_train_speed_without_stoppages_l1646_164623


namespace NUMINAMATH_GPT_travel_time_K_l1646_164658

/-
Given that:
1. K's speed is x miles per hour.
2. M's speed is x - 1 miles per hour.
3. K takes 1 hour less than M to travel 60 miles (i.e., 60/x hours).
Prove that K's time to travel 60 miles is 6 hours.
-/
theorem travel_time_K (x : ℝ)
  (h1 : x > 0)
  (h2 : x ≠ 1)
  (h3 : 60 / (x - 1) - 60 / x = 1) :
  60 / x = 6 :=
sorry

end NUMINAMATH_GPT_travel_time_K_l1646_164658


namespace NUMINAMATH_GPT_solve_bank_account_problem_l1646_164677

noncomputable def bank_account_problem : Prop :=
  ∃ (A E Z : ℝ),
    A > E ∧
    Z > A ∧
    A - E = (1/12) * (A + E) ∧
    Z - A = (1/10) * (Z + A) ∧
    1.10 * A = 1.20 * E + 20 ∧
    1.10 * A + 30 = 1.15 * Z ∧
    E = 2000 / 23

theorem solve_bank_account_problem : bank_account_problem :=
sorry

end NUMINAMATH_GPT_solve_bank_account_problem_l1646_164677


namespace NUMINAMATH_GPT_find_side_b_l1646_164695

theorem find_side_b (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3)
    (hC : C = Real.pi / 3) (hA : A = Real.pi / 6) (hB : B = Real.pi / 2) : b = 4 := by
  sorry

end NUMINAMATH_GPT_find_side_b_l1646_164695


namespace NUMINAMATH_GPT_sqrt_12_estimate_l1646_164681

theorem sqrt_12_estimate : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end NUMINAMATH_GPT_sqrt_12_estimate_l1646_164681


namespace NUMINAMATH_GPT_heating_time_l1646_164611

def T_initial: ℝ := 20
def T_final: ℝ := 100
def rate: ℝ := 5

theorem heating_time : (T_final - T_initial) / rate = 16 := by
  sorry

end NUMINAMATH_GPT_heating_time_l1646_164611


namespace NUMINAMATH_GPT_stratified_sampling_l1646_164660

-- Definitions
def total_staff : ℕ := 150
def senior_titles : ℕ := 45
def intermediate_titles : ℕ := 90
def clerks : ℕ := 15
def sample_size : ℕ := 10

-- Ratios for stratified sampling
def senior_sample : ℕ := (senior_titles * sample_size) / total_staff
def intermediate_sample : ℕ := (intermediate_titles * sample_size) / total_staff
def clerks_sample : ℕ := (clerks * sample_size) / total_staff

-- Theorem statement
theorem stratified_sampling :
  senior_sample = 3 ∧ intermediate_sample = 6 ∧ clerks_sample = 1 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l1646_164660


namespace NUMINAMATH_GPT_maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l1646_164676

-- Part (a): One blue cube
theorem maximum_amount_one_blue_cube : 
  ∃ (B : ℕ → ℚ) (P : ℕ → ℕ), (B 1 = 2) ∧ (∀ m > 1, B m = 2^m / P m) ∧ (P 1 = 1) ∧ (∀ m > 1, P m = m) ∧ B 100 = 2^100 / 100 :=
by
  sorry

-- Part (b): Exactly n blue cubes
theorem maximum_amount_n_blue_cubes (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 100) : 
  ∃ (B : ℕ × ℕ → ℚ) (P : ℕ × ℕ → ℕ), (B (1, 0) = 2) ∧ (B (1, 1) = 2) ∧ (∀ m > 1, B (m, 0) = 2^m) ∧ (P (1, 0) = 1) ∧ (P (1, 1) = 1) ∧ (∀ m > 1, P (m, 0) = 1) ∧ B (100, n) = 2^100 / Nat.choose 100 n :=
by
  sorry

end NUMINAMATH_GPT_maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l1646_164676


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1646_164691

noncomputable def U : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2}
noncomputable def A : Set ℝ := {x | x < 1 ∨ x > 3}
noncomputable def B : Set ℝ := {x | x < 1 ∨ x > 2}

theorem problem1 : A ∩ B = {x | x < 1 ∨ x > 3} := 
  sorry

theorem problem2 : A ∩ (U \ B) = ∅ := 
  sorry

theorem problem3 : U \ (A ∪ B) = {1, 2} := 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1646_164691


namespace NUMINAMATH_GPT_parabola_directrix_distance_l1646_164693

theorem parabola_directrix_distance (a : ℝ) : 
  (abs (a / 4 + 1) = 2) → (a = -12 ∨ a = 4) := 
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_distance_l1646_164693


namespace NUMINAMATH_GPT_smallest_t_for_sine_polar_circle_l1646_164662

theorem smallest_t_for_sine_polar_circle :
  ∃ t : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t) → ∃ r : ℝ, r = Real.sin θ) ∧
           (∀ θ : ℝ, (θ = t) → ∃ r : ℝ, r = 0) ∧
           (∀ t' : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t') → ∃ r : ℝ, r = Real.sin θ) →
                       (∀ θ : ℝ, (θ = t') → ∃ r : ℝ, r = 0) → t' ≥ t) :=
by
  sorry

end NUMINAMATH_GPT_smallest_t_for_sine_polar_circle_l1646_164662


namespace NUMINAMATH_GPT_cost_of_meatballs_is_five_l1646_164648

-- Define the conditions
def cost_of_pasta : ℕ := 1
def cost_of_sauce : ℕ := 2
def total_cost_of_meal (servings : ℕ) (cost_per_serving : ℕ) : ℕ := servings * cost_per_serving

-- Define the cost of meatballs calculation
def cost_of_meatballs (total_cost pasta_cost sauce_cost : ℕ) : ℕ :=
  total_cost - pasta_cost - sauce_cost

-- State the theorem we want to prove
theorem cost_of_meatballs_is_five :
  cost_of_meatballs (total_cost_of_meal 8 1) cost_of_pasta cost_of_sauce = 5 :=
by
  -- This part will include the proof steps
  sorry

end NUMINAMATH_GPT_cost_of_meatballs_is_five_l1646_164648


namespace NUMINAMATH_GPT_Dad_steps_l1646_164600

variable (d m y : ℕ)

-- Conditions
def condition_1 : Prop := d = 3 → m = 5
def condition_2 : Prop := m = 3 → y = 5
def condition_3 : Prop := m + y = 400

-- Question and Answer
theorem Dad_steps : condition_1 d m → condition_2 m y → condition_3 m y → d = 90 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Dad_steps_l1646_164600


namespace NUMINAMATH_GPT_Yankees_to_Mets_ratio_l1646_164679

theorem Yankees_to_Mets_ratio : 
  ∀ (Y M R : ℕ), M = 88 → (M + R + Y = 330) → (4 * R = 5 * M) → (Y : ℚ) / M = 3 / 2 :=
by
  intros Y M R hm htotal hratio
  sorry

end NUMINAMATH_GPT_Yankees_to_Mets_ratio_l1646_164679


namespace NUMINAMATH_GPT_initial_deposit_l1646_164673

theorem initial_deposit (P R : ℝ) (h1 : 8400 = P + (P * R * 2) / 100) (h2 : 8760 = P + (P * (R + 4) * 2) / 100) : 
  P = 2250 :=
  sorry

end NUMINAMATH_GPT_initial_deposit_l1646_164673


namespace NUMINAMATH_GPT_factorize_expression_l1646_164667

variable (a x : ℝ)

theorem factorize_expression : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l1646_164667


namespace NUMINAMATH_GPT_q_minus_r_max_value_l1646_164636

theorem q_minus_r_max_value :
  ∃ (q r : ℕ), 1073 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q - r = 31 :=
sorry

end NUMINAMATH_GPT_q_minus_r_max_value_l1646_164636


namespace NUMINAMATH_GPT_max_volume_prism_l1646_164626

theorem max_volume_prism (a b h : ℝ) (V : ℝ) 
  (h1 : a * h + b * h + a * b = 32) : 
  V = a * b * h → V ≤ 128 * Real.sqrt 3 / 3 := 
by
  sorry

end NUMINAMATH_GPT_max_volume_prism_l1646_164626


namespace NUMINAMATH_GPT_no_solution_values_l1646_164688

theorem no_solution_values (m : ℝ) :
  (∀ x : ℝ, x ≠ 5 → x ≠ -5 → (1 / (x - 5) + m / (x + 5) ≠ (m + 5) / (x^2 - 25))) ↔
  m = -1 ∨ m = 5 ∨ m = -5 / 11 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_values_l1646_164688


namespace NUMINAMATH_GPT_college_student_ticket_cost_l1646_164685

theorem college_student_ticket_cost 
    (total_visitors : ℕ)
    (nyc_residents: ℕ)
    (college_students_nyc: ℕ)
    (total_money_received : ℕ) :
    total_visitors = 200 →
    nyc_residents = total_visitors / 2 →
    college_students_nyc = (nyc_residents * 30) / 100 →
    total_money_received = 120 →
    (total_money_received / college_students_nyc) = 4 := 
sorry

end NUMINAMATH_GPT_college_student_ticket_cost_l1646_164685


namespace NUMINAMATH_GPT_students_taking_both_courses_l1646_164646

theorem students_taking_both_courses (total_students students_french students_german students_neither both_courses : ℕ) 
(h1 : total_students = 94) 
(h2 : students_french = 41) 
(h3 : students_german = 22) 
(h4 : students_neither = 40) 
(h5 : total_students = students_french + students_german - both_courses + students_neither) :
both_courses = 9 :=
by
  -- sorry can be replaced with the actual proof if necessary
  sorry

end NUMINAMATH_GPT_students_taking_both_courses_l1646_164646


namespace NUMINAMATH_GPT_sum_of_integers_l1646_164639

theorem sum_of_integers (a b c : ℤ) (h1 : a = (1 / 3) * (b + c)) (h2 : b = (1 / 5) * (a + c)) (h3 : c = 35) : a + b + c = 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1646_164639


namespace NUMINAMATH_GPT_arith_seq_sum_7_8_9_l1646_164694

noncomputable def S_n (a : Nat → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n.succ).sum a

def arith_seq (a : Nat → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

theorem arith_seq_sum_7_8_9 (a : Nat → ℝ) (h_arith : arith_seq a)
    (h_S3 : S_n a 3 = 8) (h_S6 : S_n a 6 = 7) : 
  (a 7 + a 8 + a 9) = 1 / 8 := 
  sorry

end NUMINAMATH_GPT_arith_seq_sum_7_8_9_l1646_164694


namespace NUMINAMATH_GPT_problem_solution_l1646_164631

theorem problem_solution :
  -20 + 7 * (8 - 2 / 2) = 29 :=
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l1646_164631


namespace NUMINAMATH_GPT_marlon_goals_l1646_164647

theorem marlon_goals :
  ∃ g : ℝ,
    (∀ p f : ℝ, p + f = 40 → g = 0.4 * p + 0.5 * f) → g = 20 :=
by
  sorry

end NUMINAMATH_GPT_marlon_goals_l1646_164647


namespace NUMINAMATH_GPT_product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l1646_164674

theorem product_two_smallest_one_digit_primes_and_largest_three_digit_prime :
  2 * 3 * 997 = 5982 :=
by
  sorry

end NUMINAMATH_GPT_product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l1646_164674


namespace NUMINAMATH_GPT_find_v_l1646_164632

theorem find_v (v : ℝ) (h : (v - v / 3) - ((v - v / 3) / 3) = 4) : v = 9 := 
by 
  sorry

end NUMINAMATH_GPT_find_v_l1646_164632


namespace NUMINAMATH_GPT_betty_cookies_brownies_l1646_164659

theorem betty_cookies_brownies :
  let initial_cookies := 60
  let initial_brownies := 10
  let cookies_per_day := 3
  let brownies_per_day := 1
  let days := 7
  let remaining_cookies := initial_cookies - cookies_per_day * days
  let remaining_brownies := initial_brownies - brownies_per_day * days
  remaining_cookies - remaining_brownies = 36 :=
by
  sorry

end NUMINAMATH_GPT_betty_cookies_brownies_l1646_164659


namespace NUMINAMATH_GPT_price_each_clock_is_correct_l1646_164661

-- Definitions based on the conditions
def numberOfDolls := 3
def numberOfClocks := 2
def numberOfGlasses := 5
def pricePerDoll := 5
def pricePerGlass := 4
def totalCost := 40
def profit := 25

-- The total revenue from selling dolls and glasses
def revenueFromDolls := numberOfDolls * pricePerDoll
def revenueFromGlasses := numberOfGlasses * pricePerGlass
def totalRevenueNeeded := totalCost + profit
def revenueFromDollsAndGlasses := revenueFromDolls + revenueFromGlasses

-- The required revenue from clocks
def revenueFromClocks := totalRevenueNeeded - revenueFromDollsAndGlasses

-- The price per clock
def pricePerClock := revenueFromClocks / numberOfClocks

-- Statement to prove
theorem price_each_clock_is_correct : pricePerClock = 15 := sorry

end NUMINAMATH_GPT_price_each_clock_is_correct_l1646_164661


namespace NUMINAMATH_GPT_square_diagonal_length_l1646_164650

theorem square_diagonal_length (rect_length rect_width : ℝ) 
  (h1 : rect_length = 45) 
  (h2 : rect_width = 40) 
  (rect_area := rect_length * rect_width) 
  (square_area := rect_area) 
  (side_length := Real.sqrt square_area) 
  (diagonal := side_length * Real.sqrt 2) :
  diagonal = 60 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_square_diagonal_length_l1646_164650


namespace NUMINAMATH_GPT_length_of_best_day_l1646_164697

theorem length_of_best_day
  (len_raise_the_roof : Nat)
  (len_rap_battle : Nat)
  (len_best_day : Nat)
  (total_ride_duration : Nat)
  (playlist_count : Nat)
  (total_songs_length : Nat)
  (h_len_raise_the_roof : len_raise_the_roof = 2)
  (h_len_rap_battle : len_rap_battle = 3)
  (h_total_ride_duration : total_ride_duration = 40)
  (h_playlist_count : playlist_count = 5)
  (h_total_songs_length : len_raise_the_roof + len_rap_battle + len_best_day = total_songs_length)
  (h_playlist_length : total_ride_duration / playlist_count = total_songs_length) :
  len_best_day = 3 := 
sorry

end NUMINAMATH_GPT_length_of_best_day_l1646_164697


namespace NUMINAMATH_GPT_no_nat_numbers_satisfy_l1646_164670

theorem no_nat_numbers_satisfy (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k := 
sorry

end NUMINAMATH_GPT_no_nat_numbers_satisfy_l1646_164670


namespace NUMINAMATH_GPT_smallest_n_product_exceeds_l1646_164633

theorem smallest_n_product_exceeds (n : ℕ) : (5 : ℝ) ^ (n * (n + 1) / 14) > 1000 ↔ n = 7 :=
by sorry

end NUMINAMATH_GPT_smallest_n_product_exceeds_l1646_164633


namespace NUMINAMATH_GPT_largest_initial_number_l1646_164634

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end NUMINAMATH_GPT_largest_initial_number_l1646_164634


namespace NUMINAMATH_GPT_prob_B_win_correct_l1646_164620

-- Define the probabilities for player A winning and a draw
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.4

-- Define the total probability of all outcomes
def total_prob : ℝ := 1

-- Define the probability of player B winning
def prob_B_win : ℝ := total_prob - prob_A_win - prob_draw

-- Proof problem: Prove that the probability of player B winning is 0.3
theorem prob_B_win_correct : prob_B_win = 0.3 :=
by
  -- The proof would go here, but we use sorry to skip it for now.
  sorry

end NUMINAMATH_GPT_prob_B_win_correct_l1646_164620


namespace NUMINAMATH_GPT_arithmetic_sequence_count_l1646_164663

-- Define the initial conditions
def a1 : ℤ := -3
def d : ℤ := 3
def an : ℤ := 45

-- Proposition stating the number of terms n in the arithmetic sequence
theorem arithmetic_sequence_count :
  ∃ n : ℕ, an = a1 + (n - 1) * d ∧ n = 17 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_count_l1646_164663


namespace NUMINAMATH_GPT_cylinder_volume_l1646_164638

variables (a : ℝ) (π_ne_zero : π ≠ 0) (two_ne_zero : 2 ≠ 0) 

theorem cylinder_volume (h1 : ∃ (h r : ℝ), (2 * π * r = 2 * a ∧ h = a) 
                        ∨ (2 * π * r = a ∧ h = 2 * a)) :
  (∃ (V : ℝ), V = a^3 / π) ∨ (∃ (V : ℝ), V = a^3 / (2 * π)) :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_l1646_164638


namespace NUMINAMATH_GPT_percentage_less_than_l1646_164665

theorem percentage_less_than (P T J : ℝ) 
  (h1 : T = 0.9375 * P) 
  (h2 : J = 0.8 * T) 
  : (P - J) / P * 100 = 25 := 
by
  sorry

end NUMINAMATH_GPT_percentage_less_than_l1646_164665


namespace NUMINAMATH_GPT_least_value_of_x_l1646_164615

theorem least_value_of_x (x : ℝ) : (4 * x^2 + 8 * x + 3 = 1) → (-1 ≤ x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_least_value_of_x_l1646_164615


namespace NUMINAMATH_GPT_h_value_l1646_164672

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 7

theorem h_value :
  ∃ (h : ℝ → ℝ), (h 0 = 7)
  ∧ (∃ (a b c : ℝ), (f a = 0) ∧ (f b = 0) ∧ (f c = 0) ∧ (h (-8) = (1/49) * (-8 - a^3) * (-8 - b^3) * (-8 - c^3))) 
  ∧ h (-8) = -1813 := by
  sorry

end NUMINAMATH_GPT_h_value_l1646_164672


namespace NUMINAMATH_GPT_probability_same_unit_l1646_164657

theorem probability_same_unit
  (units : ℕ) (people : ℕ) (same_unit_cases total_cases : ℕ)
  (h_units : units = 4)
  (h_people : people = 2)
  (h_total_cases : total_cases = units * units)
  (h_same_unit_cases : same_unit_cases = units) :
  (same_unit_cases :  ℝ) / total_cases = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_probability_same_unit_l1646_164657


namespace NUMINAMATH_GPT_sum_of_numbers_l1646_164614

/-- Given three numbers in the ratio 1:2:5, with the sum of their squares being 4320,
prove that the sum of the numbers is 96. -/

theorem sum_of_numbers (x : ℝ) (h1 : (x:ℝ) = x) (h2 : 2 * x = 2 * x) (h3 : 5 * x = 5 * x) 
  (h4 : x^2 + (2 * x)^2 + (5 * x)^2 = 4320) :
  x + 2 * x + 5 * x = 96 := 
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1646_164614


namespace NUMINAMATH_GPT_Carmen_candle_burn_time_l1646_164698

theorem Carmen_candle_burn_time
  (night_to_last_candle_first_scenario : ℕ := 8)
  (hours_per_night_second_scenario : ℕ := 2)
  (nights_second_scenario : ℕ := 24)
  (candles_second_scenario : ℕ := 6) :
  ∃ T : ℕ, (night_to_last_candle_first_scenario * T = hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) ∧ T = 1 :=
by
  let T := (hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) / night_to_last_candle_first_scenario
  have : T = 1 := by sorry
  use T
  exact ⟨ by sorry, this⟩

end NUMINAMATH_GPT_Carmen_candle_burn_time_l1646_164698


namespace NUMINAMATH_GPT_ratio_of_area_l1646_164664

noncomputable def area_of_triangle_ratio (AB CD height : ℝ) (h : CD = 2 * AB) : ℝ :=
  let ABCD_area := (AB + CD) * height / 2
  let EAB_area := ABCD_area / 3
  EAB_area / ABCD_area

theorem ratio_of_area (AB CD : ℝ) (height : ℝ) (h1 : AB = 10) (h2 : CD = 20) (h3 : height = 5) : 
  area_of_triangle_ratio AB CD height (by rw [h1, h2]; ring) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_area_l1646_164664


namespace NUMINAMATH_GPT_solutions_of_quadratic_l1646_164699

theorem solutions_of_quadratic (c : ℝ) (h : ∀ α β : ℝ, 
  (α^2 - 3*α + c = 0 ∧ β^2 - 3*β + c = 0) → 
  ( (-α)^2 + 3*(-α) - c = 0 ∨ (-β)^2 + 3*(-β) - c = 0 ) ) :
  ∃ α β : ℝ, (α = 0 ∧ β = 3) ∨ (α = 3 ∧ β = 0) :=
by
  sorry

end NUMINAMATH_GPT_solutions_of_quadratic_l1646_164699


namespace NUMINAMATH_GPT_age_6_not_child_l1646_164622

-- Definition and assumptions based on the conditions
def billboard_number : ℕ := 5353
def mr_smith_age : ℕ := 53
def children_ages : List ℕ := [1, 2, 3, 4, 5, 7, 8, 9, 10, 11] -- Excluding age 6

-- The theorem to prove that the age 6 is not one of Mr. Smith's children's ages.
theorem age_6_not_child :
  (billboard_number ≡ 53 * 101 [MOD 10^4]) ∧
  (∀ age ∈ children_ages, billboard_number % age = 0) ∧
  oldest_child_age = 11 → ¬(6 ∈ children_ages) :=
sorry

end NUMINAMATH_GPT_age_6_not_child_l1646_164622


namespace NUMINAMATH_GPT_value_of_n_l1646_164653

theorem value_of_n : ∃ (n : ℕ), 6 * 8 * 3 * n = Nat.factorial 8 ∧ n = 280 :=
by
  use 280
  sorry

end NUMINAMATH_GPT_value_of_n_l1646_164653


namespace NUMINAMATH_GPT_garden_yield_l1646_164652

theorem garden_yield
  (steps_length : ℕ)
  (steps_width : ℕ)
  (step_to_feet : ℕ → ℝ)
  (yield_per_sqft : ℝ)
  (h1 : steps_length = 18)
  (h2 : steps_width = 25)
  (h3 : ∀ n : ℕ, step_to_feet n = n * 2.5)
  (h4 : yield_per_sqft = 2 / 3)
  : (step_to_feet steps_length * step_to_feet steps_width) * yield_per_sqft = 1875 :=
by
  sorry

end NUMINAMATH_GPT_garden_yield_l1646_164652


namespace NUMINAMATH_GPT_prime_ge_7_p2_sub1_div_by_30_l1646_164644

theorem prime_ge_7_p2_sub1_div_by_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) :=
sorry

end NUMINAMATH_GPT_prime_ge_7_p2_sub1_div_by_30_l1646_164644


namespace NUMINAMATH_GPT_factorize_expr_l1646_164643

noncomputable def example_expr (x : ℝ) : ℝ := 2 * x^2 - 4 * x

theorem factorize_expr (x : ℝ) : example_expr x = 2 * x * (x - 2) := by
  sorry

end NUMINAMATH_GPT_factorize_expr_l1646_164643


namespace NUMINAMATH_GPT_frances_towels_weight_in_ounces_l1646_164628

theorem frances_towels_weight_in_ounces (Mary_towels Frances_towels : ℕ) (Mary_weight Frances_weight : ℝ) (total_weight : ℝ) :
  Mary_towels = 24 ∧ Mary_towels = 4 * Frances_towels ∧ total_weight = Mary_weight + Frances_weight →
  Frances_weight * 16 = 240 :=
by
  sorry

end NUMINAMATH_GPT_frances_towels_weight_in_ounces_l1646_164628


namespace NUMINAMATH_GPT_product_of_slopes_hyperbola_l1646_164686

theorem product_of_slopes_hyperbola (a b x0 y0 : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : (x0, y0) ≠ (-a, 0)) (h4 : (x0, y0) ≠ (a, 0)) 
(h5 : x0^2 / a^2 - y0^2 / b^2 = 1) : 
(y0 / (x0 + a) * (y0 / (x0 - a)) = b^2 / a^2) :=
sorry

end NUMINAMATH_GPT_product_of_slopes_hyperbola_l1646_164686


namespace NUMINAMATH_GPT_sales_volume_function_max_profit_min_boxes_for_2000_profit_l1646_164612

-- Definitions and conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def init_boxes_sold : ℝ := 250
def price_increase_effect : ℝ := 10
def max_selling_price : ℝ := 38

-- Question 1: Find functional relationship between daily sales volume y and selling price per box x
theorem sales_volume_function (x : ℝ) (hx : x ≥ min_selling_price) :
  ∃ y, y = -10 * x + 500 := by
  sorry

-- Question 2: Find the price per box to maximize daily sales profit and the maximum profit
theorem max_profit (x : ℝ) (hx : x = 35) :
  ∃ P, P = -10 * (x-20) * (x) + 500 * (x-20) := by
  sorry

-- Question 3: Determine min boxes sold to make at least 2000 yuan given price does not exceed 38 yuan
theorem min_boxes_for_2000_profit (x : ℝ) (hx : min_selling_price ≤ x ∧ x ≤ max_selling_price ∧ 
                             -10 * (x-20) * (-10 * x + 500) ≥ 2000) :
  ∃ y, y = -10 * x + 500 ∧ y ≥ 120 := by
  sorry

end NUMINAMATH_GPT_sales_volume_function_max_profit_min_boxes_for_2000_profit_l1646_164612


namespace NUMINAMATH_GPT_at_least_one_angle_not_less_than_sixty_l1646_164619

theorem at_least_one_angle_not_less_than_sixty (A B C : ℝ)
  (hABC_sum : A + B + C = 180)
  (hA : A < 60)
  (hB : B < 60)
  (hC : C < 60) : false :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_angle_not_less_than_sixty_l1646_164619


namespace NUMINAMATH_GPT_retail_price_per_book_l1646_164621

theorem retail_price_per_book (n r w : ℝ)
  (h1 : r * n = 48)
  (h2 : w = r - 2)
  (h3 : w * (n + 4) = 48) :
  r = 6 := by
  sorry

end NUMINAMATH_GPT_retail_price_per_book_l1646_164621


namespace NUMINAMATH_GPT_smallest_value_3a_plus_1_l1646_164605

theorem smallest_value_3a_plus_1 
  (a : ℝ)
  (h : 8 * a^2 + 9 * a + 6 = 2) : 
  ∃ (b : ℝ), b = 3 * a + 1 ∧ b = -2 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_value_3a_plus_1_l1646_164605


namespace NUMINAMATH_GPT_overall_average_mark_l1646_164606

theorem overall_average_mark :
  let n1 := 70
  let mean1 := 50
  let n2 := 35
  let mean2 := 60
  let n3 := 45
  let mean3 := 55
  let n4 := 42
  let mean4 := 45
  (n1 * mean1 + n2 * mean2 + n3 * mean3 + n4 * mean4 : ℝ) / (n1 + n2 + n3 + n4) = 51.89 := 
by {
  sorry
}

end NUMINAMATH_GPT_overall_average_mark_l1646_164606


namespace NUMINAMATH_GPT_isosceles_triangle_count_l1646_164635

theorem isosceles_triangle_count : 
  ∃ (count : ℕ), count = 6 ∧ 
  ∀ (a b c : ℕ), a + b + c = 25 → 
  (a = b ∨ a = c ∨ b = c) → 
  a ≠ b ∨ c ≠ b ∨ a ≠ c → 
  ∃ (x y z : ℕ), x = a ∧ y = b ∧ z = c := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_count_l1646_164635


namespace NUMINAMATH_GPT_general_term_formula_l1646_164671

theorem general_term_formula (n : ℕ) :
  ∀ (S : ℕ → ℝ), (∀ k : ℕ, S k = 1 - 2^k) → 
  (∀ a : ℕ → ℝ, a 1 = (S 1) ∧ (∀ m : ℕ, m > 1 → a m = S m - S (m - 1)) → 
  a n = -2 ^ (n - 1)) :=
by
  intro S hS a ha
  sorry

end NUMINAMATH_GPT_general_term_formula_l1646_164671


namespace NUMINAMATH_GPT_count_complex_numbers_l1646_164641

theorem count_complex_numbers (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : a + b ≤ 5) : 
  ∃ n, n = 10 := 
by
  sorry

end NUMINAMATH_GPT_count_complex_numbers_l1646_164641


namespace NUMINAMATH_GPT_greatest_integer_gcd_6_l1646_164687

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_gcd_6_l1646_164687


namespace NUMINAMATH_GPT_translate_upwards_one_unit_l1646_164689

theorem translate_upwards_one_unit (x y : ℝ) : (y = 2 * x) → (y + 1 = 2 * x + 1) := 
by sorry

end NUMINAMATH_GPT_translate_upwards_one_unit_l1646_164689


namespace NUMINAMATH_GPT_find_M_l1646_164616

theorem find_M (M : ℕ) (h1 : M > 0) (h2 : M < 10) : 
  5 ∣ (1989^M + M^1989) ↔ M = 1 ∨ M = 4 := by
  sorry

end NUMINAMATH_GPT_find_M_l1646_164616


namespace NUMINAMATH_GPT_circle_condition_l1646_164684

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + m = 0) → m < 1 := 
by
  sorry

end NUMINAMATH_GPT_circle_condition_l1646_164684


namespace NUMINAMATH_GPT_part1_real_roots_part2_specific_roots_l1646_164618

-- Part 1: Real roots condition
theorem part1_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 + (2 * m - 1) * x + m^2 = 0) : m ≤ 1/4 :=
by sorry

-- Part 2: Specific roots condition
theorem part2_specific_roots (m : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 + (2 * m - 1) * x1 + m^2 = 0) 
  (h2 : x2^2 + (2 * m - 1) * x2 + m^2 = 0) 
  (h3 : x1 * x2 + x1 + x2 = 4) : m = -1 :=
by sorry

end NUMINAMATH_GPT_part1_real_roots_part2_specific_roots_l1646_164618


namespace NUMINAMATH_GPT_smallest_x_plus_y_l1646_164624

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end NUMINAMATH_GPT_smallest_x_plus_y_l1646_164624


namespace NUMINAMATH_GPT_min_value_a_b_l1646_164607

variable (a b : ℝ)

theorem min_value_a_b (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) : 
  a + b ≥ 2 * (Real.sqrt 2 + 1) :=
sorry

end NUMINAMATH_GPT_min_value_a_b_l1646_164607


namespace NUMINAMATH_GPT_both_A_and_B_are_Gnomes_l1646_164690

inductive Inhabitant
| Elf
| Gnome

open Inhabitant

def lies_about_gold (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def tells_truth_about_others (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def A_statement : Prop := ∀ i : Inhabitant, lies_about_gold i → i = Gnome
def B_statement : Prop := ∀ i : Inhabitant, tells_truth_about_others i → i = Gnome

theorem both_A_and_B_are_Gnomes (A_statement_true : A_statement) (B_statement_true : B_statement) :
  ∀ i : Inhabitant, (lies_about_gold i ∧ tells_truth_about_others i) → i = Gnome :=
by
  sorry

end NUMINAMATH_GPT_both_A_and_B_are_Gnomes_l1646_164690


namespace NUMINAMATH_GPT_instantaneous_velocity_at_3_l1646_164602

-- Definitions based on the conditions.
def displacement (t : ℝ) := 2 * t ^ 3

-- The statement to prove.
theorem instantaneous_velocity_at_3 : (deriv displacement 3) = 54 := by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_3_l1646_164602
