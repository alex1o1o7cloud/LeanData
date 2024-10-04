import Mathlib

namespace polygon_sides_l772_772292

theorem polygon_sides (n : ℕ) 
  (h : 3240 = 180 * (n - 2) - (360)) : n = 22 := 
by 
  sorry

end polygon_sides_l772_772292


namespace patternD_cannot_form_pyramid_l772_772170

-- Define the patterns
inductive Pattern
| A
| B
| C
| D

-- Define the condition for folding into a pyramid with a square base
def canFormPyramidWithSquareBase (p : Pattern) : Prop :=
  p = Pattern.A ∨ p = Pattern.B ∨ p = Pattern.C

-- Goal: Prove that Pattern D cannot be folded into a pyramid with a square base
theorem patternD_cannot_form_pyramid : ¬ canFormPyramidWithSquareBase Pattern.D :=
by
  -- Need to provide the proof here
  sorry

end patternD_cannot_form_pyramid_l772_772170


namespace reeya_average_score_l772_772419

theorem reeya_average_score :
  let scores := [65, 67, 76, 80, 95]
  let average := (List.sum scores : ℝ) / (List.length scores : ℝ)
  average = 76.6 :=
by {
  let scores := [65, 67, 76, 80, 95]
  let average := (List.sum scores : ℝ) / (List.length scores : ℝ)
  have h : List.sum scores = 383 := by simp
  have l : List.length scores = 5 := by simp
  rw [h, l],
  norm_num,
  sorry
}

end reeya_average_score_l772_772419


namespace ice_cream_scoops_combination_l772_772555

theorem ice_cream_scoops_combination :
  let n := 5 in
  let k := 3 in
  (Nat.choose (n + k - 1) (k - 1)) = 21 :=
by
  let n := 5
  let k := 3
  have h : Nat.choose (n + k - 1) (k - 1) = Nat.choose 7 2 := by rfl
  rw [h]
  norm_num -- Computes Nat.choose 7 2 directly
  done

end ice_cream_scoops_combination_l772_772555


namespace smallest_number_l772_772924

theorem smallest_number:
  ∃ n : ℕ, (∀ d ∈ [12, 16, 18, 21, 28, 35, 39], (n - 7) % d = 0) ∧ n = 65527 :=
by
  sorry

end smallest_number_l772_772924


namespace change_in_max_value_l772_772092

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem change_in_max_value (a b c : ℝ) (h1 : -b^2 / (4 * (a + 1)) + c = -b^2 / (4 * a) + c + 27 / 2)
  (h2 : -b^2 / (4 * (a - 4)) + c = -b^2 / (4 * a) + c - 9) :
  -b^2 / (4 * (a - 2)) + c = -b^2 / (4 * a) + c - 27 / 4 :=
by
  sorry

end change_in_max_value_l772_772092


namespace maximize_distance_difference_l772_772479

noncomputable def Point := ℝ × ℝ × ℝ

structure Plane :=
  (point : Point)
  (normal : Point)

structure Line :=
  (point1 : Point)
  (point2 : Point)

def symmetric_point (B : Point) (α : Plane) : Point := 
  let (Bx, By, Bz) := B
  let (alpha_point, alpha_normal) := (α.point, α.normal)
  -- The computation for symmetric point (details omitted for brevity)
  sorry

def intersection_point (L : Line) (α : Plane) : Option Point := 
  -- The computation for intersection point (details omitted for brevity)
  sorry

axiom dist : Point → Point → ℝ

theorem maximize_distance_difference 
  (A B : Point) (α : Plane) (h_opposite_sides : ¬ α.contains A ∧ ¬ α.contains B) :
  let B' := symmetric_point B α in
  ∃ M : Point, (M = intersection_point ⟨A, B'⟩ α) ∧ 
              (∀ N ∈ α, |dist A N - dist B N| ≤ |dist A M - dist B M|):
  ∃ M, sorry

end maximize_distance_difference_l772_772479


namespace decreasing_range_of_a_l772_772045

def quadratic_function (a x : ℝ) : ℝ :=
  a*x^2 + (a-3)*x + 1

def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≥ f y

theorem decreasing_range_of_a:
  ∀ a : ℝ, is_decreasing_on (quadratic_function a) (set.Ici (-1)) ↔ -3 ≤ a ∧ a < 0 :=
by sorry

end decreasing_range_of_a_l772_772045


namespace term_100_is_14_l772_772059

def sequence_term (n : ℕ) : ℕ := 
  let k := Nat.find (λ k, n ≤ k * (k + 1) / 2)
  k

theorem term_100_is_14 : sequence_term 100 = 14 := by
  sorry

end term_100_is_14_l772_772059


namespace exists_int_leq_abs_diff_sqrt_l772_772832

theorem exists_int_leq_abs_diff_sqrt (x : ℝ) (hx : x ≥ 1/2) : ∃ n : ℤ, |x - n^2| ≤ (sqrt (x - 1/4)) :=
by
  sorry

end exists_int_leq_abs_diff_sqrt_l772_772832


namespace m_divisible_by_p_l772_772416

theorem m_divisible_by_p 
  (m n : ℕ) 
  (p : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_gt_two : p > 2)
  (h_rational_repr : (m : ℚ) / n = ∑ k in Finset.range (p - 1) + 1, (1 : ℚ) / k) : 
  p ∣ m :=
sorry

end m_divisible_by_p_l772_772416


namespace greatest_m_dividing_20_fact_by_10_pow_l772_772509

theorem greatest_m_dividing_20_fact_by_10_pow (m : ℕ) :
  (∃ m, (∃ f : ℕ, 2^f * 5^f ∣ 20!) ∧ 10^m ∣ 20!) ↔ m = 4 :=
by
  sorry

end greatest_m_dividing_20_fact_by_10_pow_l772_772509


namespace oil_leakage_calculation_l772_772969

def total_oil_leaked : ℕ := 11687
def oil_leaked_while_worked : ℕ := 5165
def oil_leaked_before_work : ℕ := 6522

theorem oil_leakage_calculation :
  oil_leaked_before_work = total_oil_leaked - oil_leaked_while_work :=
sorry

end oil_leakage_calculation_l772_772969


namespace General_Formula_l772_772236

noncomputable def a_n : ℕ → ℕ
| n => 2*n + 1

def S_n (n : ℕ) : ℕ := n * (n + 2)

def b_n (n : ℕ) : ℚ := 1 / (n^2 + 2*n)

def T_n (n : ℕ) : ℚ := (1 : ℚ) / 2 * (1 - 1 / 3) + (1 / 2 - 1 / 4) + (1 / 3 - 1 / 5) +
  ∑ k in finset.range (n - 2), (1 / (k + 2) - 1 / (k + 4))

theorem General_Formula (n : ℕ) : 
  (∀ n, a_n n = 2 * n + 1) ∧ 
  (∀ n, S_n n = n^2 + 2 * n) ∧ 
  (∀ n, T_n n = 3/4 - 1/(2*n + 2) - 1/(2*n + 4)) :=
by
  sorry

end General_Formula_l772_772236


namespace problem_statement_l772_772229

variable {α : ℝ}

def f (α : ℝ) : ℝ := 
  (sin (π - α) * cos (α - π / 2) * cos (π + α)) / 
  (sin (π / 2 + α) * cos (π / 2 + α) * tan (3 * π + α))

theorem problem_statement 
  (h1 : α ∈ set.Ioo (π) (3*π/2)) 
  (h2 : sin (π + α) = 1 / 3) : 
  f α = - (2 * real.sqrt 2) / 3 := 
sorry

end problem_statement_l772_772229


namespace find_positive_integral_solution_l772_772194

theorem find_positive_integral_solution :
  ∃ n : ℕ, n > 0 ∧ (n - 1) * 101 = (n + 1) * 100 := by
sorry

end find_positive_integral_solution_l772_772194


namespace quadrant_of_angle_l772_772619

theorem quadrant_of_angle 
  (θ : ℝ) 
  (h : sin θ + cos θ = 2023/2024) : 
  (π/2 < θ ∧ θ < π) ∨ (3*π/2 < θ ∧ θ < 2*π) := 
sorry

end quadrant_of_angle_l772_772619


namespace p_scale_measurement_l772_772101

theorem p_scale_measurement (a b P S : ℝ) (h1 : 30 = 6 * a + b) (h2 : 60 = 24 * a + b) (h3 : 100 = a * P + b) : P = 48 :=
by
  sorry

end p_scale_measurement_l772_772101


namespace find_angle_C_find_sin_A_plus_sin_B_l772_772785

open Real

namespace TriangleProblem

variables (a b c : ℝ) (A B C : ℝ)

def sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  c^2 = a^2 + b^2 + a * b

def given_c (c : ℝ) : Prop :=
  c = 4 * sqrt 7

def perimeter (a b c : ℝ) : Prop :=
  a + b + c = 12 + 4 * sqrt 7

theorem find_angle_C (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C) : 
  C = 2 * pi / 3 :=
sorry

theorem find_sin_A_plus_sin_B (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C)
  (h2 : given_c c)
  (h3 : perimeter a b c) : 
  sin A + sin B = 3 * sqrt 21 / 28 :=
sorry

end TriangleProblem

end find_angle_C_find_sin_A_plus_sin_B_l772_772785


namespace problem1_problem2_problem3_problem4_l772_772165

-- Problem 1
theorem problem1 : (-10 + (-5) - (-18)) = 3 := 
by
  sorry

-- Problem 2
theorem problem2 : (-80 * (-(4 / 5)) / (abs 16)) = -4 := 
by 
  sorry

-- Problem 3
theorem problem3 : ((1/2 - 5/9 + 5/6 - 7/12) * (-36)) = -7 := 
by 
  sorry

-- Problem 4
theorem problem4 : (- 3^2 * (-1/3)^2 +(-2)^2 / (- (2/3))^3) = -29 / 27 :=
by 
  sorry

end problem1_problem2_problem3_problem4_l772_772165


namespace max_x_minus_y_l772_772652

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l772_772652


namespace boys_from_clay_l772_772364

theorem boys_from_clay (total_students jonas_students clay_students pine_students total_boys total_girls : ℕ)
  (jonas_girls pine_boys : ℕ) 
  (H1 : total_students = 150)
  (H2 : jonas_students = 50)
  (H3 : clay_students = 60)
  (H4 : pine_students = 40)
  (H5 : total_boys = 80)
  (H6 : total_girls = 70)
  (H7 : jonas_girls = 30)
  (H8 : pine_boys = 15):
  ∃ (clay_boys : ℕ), clay_boys = 45 :=
by
  have jonas_boys : ℕ := jonas_students - jonas_girls
  have boys_from_clay := total_boys - pine_boys - jonas_boys
  exact ⟨boys_from_clay, by sorry⟩

end boys_from_clay_l772_772364


namespace diameter_of_C_l772_772575

theorem diameter_of_C
    (diameter_D : ℝ)
    (ratio_shaded_area_to_area_C : ℝ) 
    (h_d : diameter_D = 20)
    (h_ratio : ratio_shaded_area_to_area_C = 4) :
    (2 * real.sqrt 5) * 4 = 4 * real.sqrt 5 :=
by sorry

end diameter_of_C_l772_772575


namespace max_x_minus_y_l772_772647

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772647


namespace gcd_g50_g52_l772_772386

def g (x : ℤ) := x^2 - 2*x + 2022

theorem gcd_g50_g52 : Int.gcd (g 50) (g 52) = 2 := by
  sorry

end gcd_g50_g52_l772_772386


namespace sum_of_cubes_mod_5_l772_772572

theorem sum_of_cubes_mod_5 :
  (∑ i in Finset.range 51, i^3) % 5 = 0 :=
by
  sorry

end sum_of_cubes_mod_5_l772_772572


namespace series_sum_to_4_l772_772748

theorem series_sum_to_4 (x : ℝ) (hx : ∑' n : ℕ, (n + 1) * x^n = 4) : x = 1 / 2 := 
sorry

end series_sum_to_4_l772_772748


namespace sum_of_angles_of_sin_eq_pi_over_4_l772_772278

theorem sum_of_angles_of_sin_eq_pi_over_4 (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hα_sin : sin α = sqrt 5 / 5)
  (hβ_sin : sin β = sqrt 10 / 10) 
: α + β = π / 4 := 
sorry

end sum_of_angles_of_sin_eq_pi_over_4_l772_772278


namespace points_in_circle_l772_772763

theorem points_in_circle (n : ℕ) (h : nat.choose n 4 = 126) : n = 9 := 
sorry

end points_in_circle_l772_772763


namespace number_of_female_students_l772_772982

theorem number_of_female_students (n : ℕ) (h1 : 42 > 0) (h2 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → called_students k = 2 + k) (h3 : called_students n = 21) : 
  42 - n = 23 :=
by
  sorry

end number_of_female_students_l772_772982


namespace necessary_condition_for_minus1_lt_x_lt_3_l772_772447

-- Definitions for the conditions involving the interval
def necessary_but_not_sufficient (A B : Set ℝ) : Prop :=
  ∀ x, B x → A x ∧ ∃ x', A x' ∧ ¬B x'

theorem necessary_condition_for_minus1_lt_x_lt_3 :
  necessary_but_not_sufficient (λ x, -2 < x ∧ x < 4) (λ x, -1 < x ∧ x < 3) :=
begin
  sorry
end

end necessary_condition_for_minus1_lt_x_lt_3_l772_772447


namespace feeding_ways_correct_l772_772549

def total_feeding_ways : Nat :=
  (5 * 6 * (5 * 4 * 3 * 2 * 1)^2)

theorem feeding_ways_correct :
  total_feeding_ways = 432000 :=
by
  -- Proof is omitted here
  sorry

end feeding_ways_correct_l772_772549


namespace parents_years_in_america_before_aziz_birth_l772_772562

noncomputable def aziz_birth_year (current_year : ℕ) (aziz_age : ℕ) : ℕ :=
  current_year - aziz_age

noncomputable def years_parents_in_america_before_aziz_birth (arrival_year : ℕ) (aziz_birth_year : ℕ) : ℕ :=
  aziz_birth_year - arrival_year

theorem parents_years_in_america_before_aziz_birth 
  (current_year : ℕ := 2021) 
  (aziz_age : ℕ := 36) 
  (arrival_year : ℕ := 1982) 
  (expected_years : ℕ := 3) :
  years_parents_in_america_before_aziz_birth arrival_year (aziz_birth_year current_year aziz_age) = expected_years :=
by 
  sorry

end parents_years_in_america_before_aziz_birth_l772_772562


namespace coefficient_x_squared_in_expansion_l772_772435

theorem coefficient_x_squared_in_expansion : 
  (x - x⁻¹)^6 = ∑ n in range 7, 
    (λ r, (-1)^r * (binom 6 r) * x^(6 - 2 * r)) n & r = 2 :=
  by
  sorry

end coefficient_x_squared_in_expansion_l772_772435


namespace probability_three_heads_in_four_tosses_l772_772490

theorem probability_three_heads_in_four_tosses :
  let p := (1 / 2 : ℚ) in
  let n := 4 in
  let r := 3 in
  let comb := (nat.factorial n) / ((nat.factorial r) * (nat.factorial (n - r))) in
  let probability := comb * (p ^ r) * (p ^ (n - r)) in
  probability = (1 / 4 : ℚ) :=
by
  sorry

end probability_three_heads_in_four_tosses_l772_772490


namespace simplify_expression_l772_772838

theorem simplify_expression (m : ℤ) : 
  ((7 * m + 3) - 3 * m * 2) * 4 + (5 - 2 / 4) * (8 * m - 12) = 40 * m - 42 :=
by 
  sorry

end simplify_expression_l772_772838


namespace patty_coins_value_l772_772410

theorem patty_coins_value (n d q : ℕ) (h₁ : n + d + q = 30) (h₂ : 5 * n + 15 * d - 20 * q = 120) : 
  5 * n + 10 * d + 25 * q = 315 := by
sorry

end patty_coins_value_l772_772410


namespace roots_triple_relation_l772_772612

theorem roots_triple_relation (p q r α β : ℝ) (h1 : α + β = -q / p) (h2 : α * β = r / p) (h3 : β = 3 * α) :
  3 * q ^ 2 = 16 * p * r :=
sorry

end roots_triple_relation_l772_772612


namespace elasticity_ratio_is_correct_l772_772159

-- Definitions of the given elasticities
def e_OGBR_QN : ℝ := 1.27
def e_OGBR_PN : ℝ := 0.76

-- Theorem stating the ratio of elasticities equals 1.7
theorem elasticity_ratio_is_correct : (e_OGBR_QN / e_OGBR_PN) = 1.7 := sorry

end elasticity_ratio_is_correct_l772_772159


namespace smallest_number_of_marbles_l772_772558

-- Define the conditions
variables (r w b g n : ℕ)
def valid_total (r w b g n : ℕ) := r + w + b + g = n
def valid_probability_4r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w) * (r * (r - 1) * (r - 2) / 6)
def valid_probability_1w3r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w * b * (r * (r - 1) / 2))
def valid_probability_1w1b2r (r w b g n : ℕ) := w * b * (r * (r - 1) / 2) = w * b * g * r

theorem smallest_number_of_marbles :
  ∃ n r w b g, valid_total r w b g n ∧
  valid_probability_4r r w b g n ∧
  valid_probability_1w3r r w b g n ∧
  valid_probability_1w1b2r r w b g n ∧ 
  n = 21 :=
  sorry

end smallest_number_of_marbles_l772_772558


namespace floor_sqrt_2_perfect_square_infinite_l772_772014

theorem floor_sqrt_2_perfect_square_infinite :
  ∃ (S : Set ℕ), (∀ n ∈ S, ∃ k : ℕ, nat.floor (real.sqrt 2 * n) = k^2) ∧ S.Infinite :=
by
  sorry

end floor_sqrt_2_perfect_square_infinite_l772_772014


namespace largest_divisor_of_Q_l772_772137

def eight_sided_die : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def product_of_visible_numbers (visible_numbers : finset ℕ) : ℕ :=
  visible_numbers.prod id

theorem largest_divisor_of_Q :
  ∀ visible_numbers ∈ eight_sided_die.ssubsets (7),
    ∃ Q : ℕ, Q = product_of_visible_numbers visible_numbers ∧ Q % 192 = 0 :=
begin
  sorry  -- Proof is omitted as per instructions
end

end largest_divisor_of_Q_l772_772137


namespace max_value_of_f_l772_772381

open Real

noncomputable def f (θ : ℝ) : ℝ :=
  sin (θ / 2) * (1 + cos θ)

theorem max_value_of_f : 
  (∃ θ : ℝ, 0 < θ ∧ θ < π ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π → f θ' ≤ f θ) ∧ f θ = 4 * sqrt 3 / 9) := 
by
  sorry

end max_value_of_f_l772_772381


namespace father_l772_772527

-- Define the variables
variables (F S : ℕ)

-- Define the conditions
def condition1 : Prop := F = 4 * S
def condition2 : Prop := F + 20 = 2 * (S + 20)
def condition3 : Prop := S = 10

-- Statement of the problem
theorem father's_age (h1 : condition1 F S) (h2 : condition2 F S) (h3 : condition3 S) : F = 40 :=
by sorry

end father_l772_772527


namespace num_sets_union_l772_772113

theorem num_sets_union : 
  let A := {1, 2}
  in (∃ B : set ℕ, A ∪ B = {1, 2, 3}).to_finset.card = 4 :=
by
  sorry

end num_sets_union_l772_772113


namespace second_player_winning_n_gon_l772_772892

def n_gon_game (n : ℕ) := ∀ (first_player_moves : list ℕ), ∃ (second_player_moves : list ℕ), 
  second_player_has_winning_strategy n second_player_moves

def second_player_wins_for (n : ℕ) : Prop := n = 4

theorem second_player_winning_n_gon :
  ∃ n, second_player_wins_for n :=
by 
  use 4
  sorry

end second_player_winning_n_gon_l772_772892


namespace value_of_expression_l772_772624

theorem value_of_expression (x y : ℕ) (h : x * 5 = y * 4) : (x + y) / (x - y) = -9 := 
by
  -- Omitted proof
  sorry

end value_of_expression_l772_772624


namespace oil_leak_before_fix_l772_772971

theorem oil_leak_before_fix (total_leak : ℕ) (leak_during_fix : ℕ) 
    (total_leak_eq : total_leak = 11687) (leak_during_fix_eq : leak_during_fix = 5165) :
    total_leak - leak_during_fix = 6522 :=
by 
  rw [total_leak_eq, leak_during_fix_eq]
  simp
  sorry

end oil_leak_before_fix_l772_772971


namespace runners_meet_at_starting_point_runners_meet_at_starting_point_l772_772507

-- Define the conditions and the question as a proof statement
theorem runners_meet_at_starting_point :
    let time1 := 2 * 60 -- seconds
    let time2 := 4 * 60 -- seconds
    let time3 := (11 / 2) * 60 -- seconds
    let lcm_seconds := Nat.lcm (Nat.lcm time1 time2) time3
    lcm_seconds / 60 = 44 :=
by
    -- Use sorry to skip the proof steps
    sorry

-- Now, we define the main theorem using the above conditions.

theorem runners_meet_at_starting_point : 
    (Nat.lcm (Nat.lcm (2 * 60) (4 * 60)) ((11 / 2) * 60)) / 60 = 44 :=
by
    -- Use sorry to skip the proof steps
    sorry

end runners_meet_at_starting_point_runners_meet_at_starting_point_l772_772507


namespace point_B_in_third_quadrant_l772_772318

theorem point_B_in_third_quadrant (x y : ℝ) (hx : x < 0) (hy : y < 1) :
    (y - 1 < 0) ∧ (x < 0) :=
by
  sorry  -- proof to be filled

end point_B_in_third_quadrant_l772_772318


namespace pentagon_parallelograms_exist_l772_772328

open EuclideanGeometry

noncomputable def exists_point_P (A B C D E : Point) (A1 A2 A3 A4 A5 : Point) : Prop :=
  ∃ P : Point,
    (vector A1 A2 = vector P A3) ∧
    (vector A P = vector A5 A4)

theorem pentagon_parallelograms_exist (A B C D E : Point)
  (A1 : midpoint A B)
  (A2 : midpoint B C)
  (A3 : midpoint C D)
  (A4 : midpoint D E)
  (A5 : midpoint E A) :
  exists_point_P A B C D E A1 A2 A3 A4 A5 :=
sorry

end pentagon_parallelograms_exist_l772_772328


namespace at_least_one_small_area_l772_772245

open Real

-- Define the area of a triangle
def triangle_area (A B C : Vect{3}) : ℝ :=
  let a := (B - A).magnitude
  let b := (C - A).magnitude
  let c := (C - B).magnitude
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Given vertices of triangle ABC
variables {A B C P1 P2 P3 P4 : Vect{3}}

-- Conditions: P1, P2, P3, P4 lie on sides of triangle ABC
variables (hP1 : ∃ t1 : ℝ, 0 ≤ t1 ∧ t1 ≤ 1 ∧ P1 = t1 • B + (1 - t1) • C)
variables (hP2 : ∃ t2 : ℝ, 0 ≤ t2 ∧ t2 ≤ 1 ∧ P2 = t2 • A + (1 - t2) • C)
variables (hP3 : ∃ t3 : ℝ, 0 ≤ t3 ∧ t3 ≤ 1 ∧ P3 = t3 • A + (1 - t3) • B)
variables (hP4 : ∃ t4 : ℝ, 0 ≤ t4 ∧ t4 ≤ 1 ∧ P4 = t4 • A + (1 - t4) • C)

theorem at_least_one_small_area :
  let S_ABC := triangle_area A B C in
  ∃ (i j k : fin 4),
  i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
  triangle_area (subsets.mkfin4 [P1, P2, P3, P4] i) (subsets.mkfin4 [P1, P2, P3, P4] j) (subsets.mkfin4 [P1, P2, P3, P4] k) ≤ (1 / 4) * S_ABC :=
by sorry

end at_least_one_small_area_l772_772245


namespace minimize_time_theta_l772_772148

theorem minimize_time_theta (α θ : ℝ) (h1 : 0 < α) (h2 : α < 90) (h3 : θ = α / 2) : 
  θ = α / 2 :=
by
  sorry

end minimize_time_theta_l772_772148


namespace largest_prime_factor_9689_l772_772603

theorem largest_prime_factor_9689 : ∃ p, nat.prime p ∧ nat.factors 9689 = [5, 23, p] ∧ ∀ q, nat.prime q ∧ q ∈ nat.factors 9689 → q ≤ p :=
by
  use 83
  split
  sorry -- Proof of primality of 83
  split
  sorry -- Proof that the factors of 9689 are [5, 23, 83]
  sorry -- Proof that 83 is the largest prime factor among them

end largest_prime_factor_9689_l772_772603


namespace a_c3_b3_equiv_zero_l772_772720

-- Definitions based on conditions
def cubic_eq_has_geom_progression_roots (a b c : ℝ) :=
  ∃ d q : ℝ, d ≠ 0 ∧ q ≠ 0 ∧ d + d * q + d * q^2 = -a ∧
    d^2 * q * (1 + q + q^2) = b ∧
    d^3 * q^3 = -c

-- Main theorem to prove
theorem a_c3_b3_equiv_zero (a b c : ℝ) :
  cubic_eq_has_geom_progression_roots a b c → a^3 * c - b^3 = 0 :=
by
  sorry

end a_c3_b3_equiv_zero_l772_772720


namespace max_x_minus_y_l772_772646

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772646


namespace zero_point_sufficient_not_necessary_l772_772713

theorem zero_point_sufficient_not_necessary (a : ℝ) (h : a < -3) :
  (∃ x ∈ Icc (-1 : ℝ) 2, a * x + 3 = 0) ↔ (∃ x, a * x + 3 = 0) ∧ ¬ (∀ x ∈ Icc (-1 : ℝ) 2, a * x + 3 = 0) :=
by
  sorry

end zero_point_sufficient_not_necessary_l772_772713


namespace travel_ways_l772_772614

theorem travel_ways (m n : ℕ) (h1 : m = 3) (h2 : n = 4) : m * n = 12 := by
  rw [h1, h2]
  norm_num
  sorry

end travel_ways_l772_772614


namespace triangle_side_eq_median_l772_772339

theorem triangle_side_eq_median (A B C : Type) (a b c : ℝ) (hAB : a = 2) (hAC : b = 3) (hBC_eq_median : c = (2 * (Real.sqrt (13 / 10)))) :
  c = (Real.sqrt 130) / 5 := by
  sorry

end triangle_side_eq_median_l772_772339


namespace find_eccentricity_of_ellipse_l772_772262

theorem find_eccentricity_of_ellipse
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (c : ℝ)
  (P Q : ℝ × ℝ)
  (hp1 : P.1 = c)
  (hp2 : P.2 = (sqrt 2 / 2) * c)
  (hq1 : Q.1 = -c)
  (hq2 : Q.2 = -(sqrt 2 / 2) * c)
  (line_intersects_ellipse : (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ (Q.1^2 / a^2 + Q.2^2 / b^2 = 1))
  (foci : P.1 = c ∧ Q.1 = -c) :
  (c = a * (sqrt (1 - (b^2 / a^2)))) -> 
  (sqrt ((a^2 - b^2) / a^2) = sqrt 2 / 2) :=
by sorry

end find_eccentricity_of_ellipse_l772_772262


namespace impossible_column_sums_equal_l772_772369

theorem impossible_column_sums_equal (n : ℕ) (h : n < 12) :
  ¬ (∃ k : ℕ, k ≠ 0 ∧ k ∣ n) :=
by
  have h₁ : ¬ (∃ (k : ℕ), k ≠ 0 ∧ 12 ∣ n),
  {
    intros,
    have h₀:  k = 12,
    {
      sorry -- detailed logic to show only possible k is 12
    },
    linarith -- contradiction with h: n < 12
  },
  sorry -- combining impossibility of k > 0 and k ∣ n 

end impossible_column_sums_equal_l772_772369


namespace oil_leakage_calculation_l772_772968

def total_oil_leaked : ℕ := 11687
def oil_leaked_while_worked : ℕ := 5165
def oil_leaked_before_work : ℕ := 6522

theorem oil_leakage_calculation :
  oil_leaked_before_work = total_oil_leaked - oil_leaked_while_work :=
sorry

end oil_leakage_calculation_l772_772968


namespace geometric_sequence_properties_l772_772756

variable {a : ℕ → Real}
variable {q : Real}

noncomputable def geometricSequence (a : ℕ → Real) := ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (hq : 0 < q) 
  (h : geometricSequence a) 
  (ha2_4 : a 2 + a 4 = 3) 
  (ha3_5 : a 3 * a 5 = 1) :
  q = Real.sqrt 2 / 2 ∧ ∀ n, a n = 2 ^ (n + 2) / 2 :=
by
  sorry

end geometric_sequence_properties_l772_772756


namespace sum_of_possible_values_l772_772060

noncomputable theory

def square_area (x : ℝ) : ℝ := (x - 4) ^ 2
def rectangle_area (x : ℝ) : ℝ := (x - 5) * (x + 6)

theorem sum_of_possible_values (x : ℝ) (hx : rectangle_area x = 3 * square_area x) :
  x = 6.5 ∨ x = 6 → (6.5 + 6 = 12.5) :=
by
  intro h
  exact (if hx1 : x = 6.5 then by { subst x, norm_num }) (if hx2 : x = 6 then by { subst x, norm_num }) sorry

end sum_of_possible_values_l772_772060


namespace arrangement_count_l772_772178

-- Define the problem conditions as hypotheses
def teacher := true
def boys := fin 4
def girls := fin 2
def total_people := 1 + 4 + 2 = 7

theorem arrangement_count (h1 : girls_are_adjacent) (h2 : boys_not_adjacent) 
    (h3 : teacher_not_middle ∧ girlA_not_left) : 
    total_arrangements 7 teacher boys girls = 3720 :=
sorry

end arrangement_count_l772_772178


namespace sequence_proof_l772_772636

-- Definitions based on given conditions
def is_arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sequence_condition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, (∑ i in (finset.range n), 1 / (a i * a (i + 1))) = n / (2 * n + 4)

def b_n (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (-1) ^ n * a n * a (n + 1)

-- Definition of the specific arithmetic sequence
def specific_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a n = n + 1

-- The specific conditions of the problem
def problem_conditions (a : ℕ → ℕ) (d : ℕ) : Prop :=
  is_arithmetic_seq a d ∧ d > 0 ∧ sequence_condition a

-- The sum S_20 of the first 20 terms of b_n
def S_20 (a : ℕ → ℕ) : ℕ :=
  (finset.range 20).sum (b_n a)

-- Statement to prove
theorem sequence_proof :
  ∃ a d : ℕ, problem_conditions a d ∧ specific_arithmetic_sequence a ∧ S_20 a = 240 :=
sorry

end sequence_proof_l772_772636


namespace area_CDE_proof_l772_772048

-- Defining the points and conditions
structure Triangle :=
  (A B C D E : Point)
  (right_angle_C : ∠ A C B = 90)
  (isosceles_ACB : dist A C = dist B C)
  (area_ABC : ½ * dist A C * dist B C = 12.5)
  (trisection_D_E: Rint (D ∈ Segment A B) ∧ Rint (E ∈ Segment A B) ∧
    ∠ A C D = 30 ∧ ∠ D C B = 30)

noncomputable def area_CDE {A B C D E : Point} (H : Triangle A B C D E) : ℝ :=
  sorry -- Will be proven

theorem area_CDE_proof (H : Triangle) : area_CDE H = (50 - 25 * √3) / 2 :=
  sorry

end area_CDE_proof_l772_772048


namespace BD_FX_ME_concurrent_l772_772760

-- Define the given geometric setup and conditions as Lean types and properties
variables {B C F A D E M X : Type} [MetricSpace B] [MetricSpace C] [MetricSpace F]
          [MetricSpace A] [MetricSpace D] [MetricSpace E] [MetricSpace M] [MetricSpace X] 

-- Define the specific conditions in the problem
def triangle_BCF (B C F : Type) [MetricSpace B] [MetricSpace C] [MetricSpace F] : Prop :=
  ∃ (is_right_angle : ∠ B C F = 90)

def point_A_on_CF (A F B C : Type) [MetricSpace A] [MetricSpace F] [MetricSpace B] [MetricSpace C] : Prop :=
  FA = FB ∧ F between A and C

def point_D_properties (D A C : Type) [MetricSpace D] [MetricSpace A] [MetricSpace C] : Prop :=
  DA = DC ∧ ∠BAC bisects ∠DAB

def point_E_properties (E A D : Type) [MetricSpace E] [MetricSpace A] [MetricSpace D] : Prop :=
  EA = ED ∧ AD bisects ∠EAC

def midpoint_M_of_CF (M C F : Type) [MetricSpace M] [MetricSpace C] [MetricSpace F] : Prop :=
  M is the midpoint of segment CF

def quadrilateral_AMXE_parallelogram (A M X E : Type) [MetricSpace A] [MetricSpace M] [MetricSpace X] [MetricSpace E] : Prop :=
  AM || EX ∧ AE || MX

-- Prove that the lines BD, FX, and ME are concurrent
theorem BD_FX_ME_concurrent 
  (B C F A D E M X : Type)
  [H1: triangle_BCF B C F]
  [H2: point_A_on_CF A F B C] 
  [H3; point_D_properties D A C] 
  [H4: point_E_properties E A D] 
  [H5: midpoint_M_of_CF M C F] 
  [H6: quadrilateral_AMXE_parallelogram A M X E] :
  concurrent_lines B D F X M E := sorry

end BD_FX_ME_concurrent_l772_772760


namespace rhombus_new_perimeter_l772_772859

theorem rhombus_new_perimeter (d1 d2 : ℝ) (scale : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 24) (h_scale : scale = 0.5) : 
  4 * (scale * (Real.sqrt ((d1/2)^2 + (d2/2)^2))) = 26 := 
by
  sorry

end rhombus_new_perimeter_l772_772859


namespace dissection_possible_l772_772897

theorem dissection_possible :
  ∃ (pieces: ℕ), pieces = 7 ∧
  (∀ (a b c d e f g: ℚ),
    a = 169 ∧ b = 1225 ∧
    c = 289 ∧ d = 529 ∧ e = 576 ∧
    a + b = c + d + e →
    pieces = 7 ∧
    (∀ (cut: ℕ → ℕ → bool), 
      (∃ (x y z: ℕ),
        (cut x y = tt → cut y z = tt →
          a + b = c + d + e ∧
          cut x y = tt ∧ cut y z = tt)))))
:= by
  -- Each step of the proof would normally be filled out here,
  -- but we are omitting the proof itself as instructed.
  sorry

end dissection_possible_l772_772897


namespace digit_of_n_l772_772483

theorem digit_of_n (n : ℕ) (h : n ≥ 3) (prime_S : Nat.Prime (∑ i in Finset.range (2*n - 2), n^(i+1) - 4)) : 
  n % 10 = 5 :=
sorry

end digit_of_n_l772_772483


namespace problem_l772_772617

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

noncomputable def f_deriv (x : ℝ) : ℝ := - (1 / x^2) * Real.cos x - (1 / x) * Real.sin x

theorem problem (h_pi_ne_zero : Real.pi ≠ 0) (h_pi_div_two_ne_zero : Real.pi / 2 ≠ 0) :
  f Real.pi + f_deriv (Real.pi / 2) = -3 / Real.pi  := by
  sorry

end problem_l772_772617


namespace triangle_side_length_l772_772344

theorem triangle_side_length (A B C : Type*) [inner_product_space ℝ (A × B)]
  (AB AC BC : ℝ) (h1 : AB = 2)
  (h2 : AC = 3) (h3 : BC = sqrt(5.2)) :
  ACS = sqrt(5.2) :=
sorry

end triangle_side_length_l772_772344


namespace pages_diff_l772_772820

theorem pages_diff (finished_fraction : ℚ) (total_pages finished_pages remaining_pages difference : ℕ) 
    (h1 : finished_fraction = 2/3) 
    (h2 : total_pages = 60) 
    (h3 : finished_pages = (finished_fraction * total_pages).toNat) 
    (h4 : remaining_pages = (total_pages - finished_pages))
    (h5 : difference = finished_pages - remaining_pages) : 
    difference = 20 := 
by 
    sorry

end pages_diff_l772_772820


namespace min_I_F_l772_772366

noncomputable def F : set (ℝ → ℝ) :=
  {f | continuous f ∧ ∀ x, real.exp (f x) + f x ≥ x + 1}

noncomputable def I (f : ℝ → ℝ) : ℝ :=
  ∫ x in set.Icc 0 real.exp 1 f x

theorem min_I_F : ∃ f ∈ F, I f = 3 / 2 :=
sorry

end min_I_F_l772_772366


namespace no_rational_solution_5x2_plus_3y2_eq_1_l772_772023

theorem no_rational_solution_5x2_plus_3y2_eq_1 :
  ¬ ∃ (x y : ℚ), 5 * x^2 + 3 * y^2 = 1 := 
sorry

end no_rational_solution_5x2_plus_3y2_eq_1_l772_772023


namespace misha_students_count_l772_772808

theorem misha_students_count :
  (∀ n : ℕ, n = 60 → (exists better worse : ℕ, better = n - 1 ∧  worse = n - 1)) →
  (∀ n : ℕ, n = 60 → (better + worse + 1 = 119)) :=
by
  sorry

end misha_students_count_l772_772808


namespace ellipse_x_intercept_l772_772967

theorem ellipse_x_intercept
  (foci1 foci2 : ℝ × ℝ)
  (intercept1 : ℝ × ℝ)
  (constant_sum : ℝ)
  (foci1_eq : foci1 = (0, -3))
  (foci2_eq : foci2 = (4, 0))
  (intercept1_eq : intercept1 = (0, 0))
  (constant_sum_eq : constant_sum = 7) :
  ∃ x : ℝ, (x, 0) = (56 / 11, 0) ∧ 
  (Real.sqrt ((x - 0)^2 + (-3 - 0)^2) + Real.sqrt ((x - 4)^2) = constant_sum ∧ 
  Real.sqrt ((0 - 0)^2 + (-3 - 0)^2) + Real.sqrt ((0 - 4)^2) = constant_sum) :=
by
  use 56 / 11
  split
  · rfl
  split
  · sorry
  · sorry

end ellipse_x_intercept_l772_772967


namespace dryer_cost_is_490_l772_772957

-- Definitions of the conditions given in the problem
def total_cost := 1200
def washer_cost (dryer_cost : ℕ) := dryer_cost + 220

-- Mathematical statement to be proved
theorem dryer_cost_is_490 : ∃ (dryer_cost : ℕ), total_cost = dryer_cost + washer_cost dryer_cost ∧ dryer_cost = 490 :=
by
  use 490
  constructor
  sorry -- This is where the proof would go
  reflexivity

end dryer_cost_is_490_l772_772957


namespace total_cans_in_display_l772_772039

theorem total_cans_in_display :
  let a := 30, d := -3 in
  let a_n n := a + (n - 1) * d in
  let n := 10 in
  ( ∑ i in finset.range(n), a_n (i + 1) ) * 2 = 310 :=
by
  sorry

end total_cans_in_display_l772_772039


namespace cross_section_area_l772_772786

theorem cross_section_area :
  let a := 8 * Real.sqrt 2,
      h := 12,
      b := 2,
      h_0 := 3 in
  let AB := a,
      SO := h,
      KL := b,
      KK_1 := h_0 in
  let CO := a / Real.sqrt 2,
      CF := CO / 4,
      GR := b / 2,
      LE := CO - CF - GR,
      tan_alpha := h_0 / LE,
      OH := (h * tan_alpha) / (Real.sqrt (1 + tan_alpha ^ 2)) in
  let BD := (2 * a),
      S_section := (BD * OH) / 2 in
  S_section = 64 * Real.sqrt 34 / 7 :=
sorry

end cross_section_area_l772_772786


namespace sum_of_squares_of_roots_l772_772978

theorem sum_of_squares_of_roots (a b : ℝ) (x₁ x₂ : ℝ)
  (h₁ : x₁^2 - (3 * a + b) * x₁ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0)
  (h₂ : x₂^2 - (3 * a + b) * x₂ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0) :
  x₁^2 + x₂^2 = 5 * (a^2 + b^2) := 
by
  sorry

end sum_of_squares_of_roots_l772_772978


namespace max_sara_tie_fraction_l772_772309

theorem max_sara_tie_fraction :
  let max_wins := 2 / 5
  let sara_wins := 1 / 4
  let postponed_fraction := 1 / 20
  let total_wins := max_wins + (sara_wins * (5 / 5))
  let non_postponed_fraction := 1 - postponed_fraction
  let win_ratio_non_postponed := total_wins * (20 / 19)
  let tie_fraction := 1 - win_ratio_non_postponed
  in tie_fraction = 6 / 19 :=
by
  sorry

end max_sara_tie_fraction_l772_772309


namespace max_x_minus_y_l772_772664

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l772_772664


namespace sum_of_cosines_dihedral_angles_l772_772065

-- Define the conditions of the problem
def sum_of_plane_angles_trihederal (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Define the problem statement
theorem sum_of_cosines_dihedral_angles (α β γ : ℝ) (d1 d2 d3 : ℝ)
  (h : sum_of_plane_angles_trihederal α β γ) : 
  d1 + d2 + d3 = 1 :=
  sorry

end sum_of_cosines_dihedral_angles_l772_772065


namespace female_managers_count_l772_772298

-- Definitions for the problem statement

def total_female_employees : ℕ := 500
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Problem parameters
variable (E M FM : ℕ) -- E: total employees, M: male employees, FM: female managers

-- Conditions
def total_employees_eq : Prop := E = M + total_female_employees
def total_managers_eq : Prop := fraction_of_managers * E = fraction_of_male_managers * M + FM

-- The statement we want to prove
theorem female_managers_count (h1 : total_employees_eq E M) (h2 : total_managers_eq E M FM) : FM = 200 :=
by
  -- to be proven
  sorry

end female_managers_count_l772_772298


namespace higher_percentage_rate_l772_772938

theorem higher_percentage_rate
  (principal : ℝ) (years : ℝ) (interest12 : ℝ) (interest_diff : ℝ) (higher_interest : ℝ)
  (h1 : principal = 14000) (h2 : years = 2) (h3 : interest12 = 3360)
  (h4 : interest_diff = 840) (h5 : higher_interest = interest12 + interest_diff) :
  higher_interest = 4200 → 
  ∃ r : ℝ, r = 15 :=
by
  intros h
  rw [h5, h3, h4] at h
  simp at h
  use 15
  sorry

end higher_percentage_rate_l772_772938


namespace identify_solid_as_frustum_l772_772066

-- Define the conditions
variables (S : Type) [solid S]
variables (top_view circular_view : set S) (front_view side_view trapezoidal_view : set S)

-- Define the problem statement
theorem identify_solid_as_frustum
  (h_top_view : top_view = circular_view)
  (h_bottom_view : bottom_view = circular_view)
  (h_front_view : front_view = trapezoidal_view)
  (h_side_view : side_view = trapezoidal_view) :
  solid_is_frustum S :=
sorry

end identify_solid_as_frustum_l772_772066


namespace arcsin_one_half_eq_pi_six_l772_772986

theorem arcsin_one_half_eq_pi_six :
  Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l772_772986


namespace solve_for_xy_l772_772841

theorem solve_for_xy (x y : ℝ) 
  (h1 : 0.05 * x + 0.07 * (30 + x) = 14.9)
  (h2 : 0.03 * y - 5.6 = 0.07 * x) : 
  x = 106.67 ∧ y = 435.567 := 
  by 
  sorry

end solve_for_xy_l772_772841


namespace sum_13_gt_0_l772_772064

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

axiom a7_gt_0 : 0 < a_n 7
axiom a8_lt_0 : a_n 8 < 0

theorem sum_13_gt_0 : S_n 13 > 0 :=
sorry

end sum_13_gt_0_l772_772064


namespace exists_longest_continuous_path_l772_772184

-- Define the grid and its properties
structure Grid :=
  (points : Finset (ℕ × ℕ))  -- Represents points on a grid (finite set of integer coordinate pairs)

-- Define a path as a list of points
structure Path (G : Grid) :=
  (nodes : List (ℕ × ℕ))  -- A list of points defining the path
  (continuous : ∀ (i : ℕ), i < nodes.length - 1 → (nodes.get? i).isSome → (nodes.get? (i + 1)).isSome)
  (no_intersect : ∀ (i j : ℕ), i ≠ j → nodes.get? i = nodes.get? j → False)

-- Define the main statement to be proven in Lean
theorem exists_longest_continuous_path (G : Grid) : 
  ∃ (P : Path G), ∀ (Q : Path G), P.nodes.length ≥ Q.nodes.length := sorry

end exists_longest_continuous_path_l772_772184


namespace teresa_total_marks_l772_772430

-- Define Teresa's scores in different subjects
def science_score := 70
def music_score := 80
def social_studies_score := 85
def physics_score := music_score / 2
def mathematics_actual_score := 90
def mathematics_adjusted_score := mathematics_actual_score * 0.75

-- Calculate the total marks considering the adjusted mathematics score
def total_marks := science_score + music_score + social_studies_score + physics_score + mathematics_adjusted_score

theorem teresa_total_marks :
  total_marks = 342.5 :=
by
  sorry

end teresa_total_marks_l772_772430


namespace percentage_of_number_l772_772817

variable (N P : ℝ)

theorem percentage_of_number 
  (h₁ : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) 
  (h₂ : (P / 100) * N = 120) : 
  P = 40 := 
by 
  sorry

end percentage_of_number_l772_772817


namespace same_terminal_side_l772_772393

theorem same_terminal_side (k : ℤ) : 
  ((2 * k + 1) * 180) % 360 = ((4 * k + 1) * 180) % 360 ∨ ((2 * k + 1) * 180) % 360 = ((4 * k - 1) * 180) % 360 := 
sorry

end same_terminal_side_l772_772393


namespace f_1984_and_f_1985_l772_772000

namespace Proof

variable {N M : Type} [AddMonoid M] [Zero M] (f : ℕ → M)

-- Conditions
axiom f_10 : f 10 = 0
axiom f_last_digit_3 {n : ℕ} : (n % 10 = 3) → f n = 0
axiom f_mn (m n : ℕ) : f (m * n) = f m + f n

-- Prove f(1984) = 0 and f(1985) = 0
theorem f_1984_and_f_1985 : f 1984 = 0 ∧ f 1985 = 0 :=
by
  sorry

end Proof

end f_1984_and_f_1985_l772_772000


namespace sec_240_eq_neg2_tan_240_ne_2tan_120_l772_772569

theorem sec_240_eq_neg2 : real.sec (240 * real.pi / 180) = -2 :=
by
  -- Definitions and conditions will be included here
  sorry

theorem tan_240_ne_2tan_120 : real.tan (240 * real.pi / 180) ≠ 2 * real.tan (120 * real.pi / 180) :=
by
  -- Definitions and conditions will be included here
  sorry

end sec_240_eq_neg2_tan_240_ne_2tan_120_l772_772569


namespace complex_conjugate_quadrant_l772_772736

theorem complex_conjugate_quadrant (z : ℂ) 
  (h : complex.det (λ i j, if i = 0 then (if j = 0 then z else 1 + 2 * complex.I) else (if j = 0 then 1 - complex.I else 1 + complex.I)) = 0) : 
  (∃ a b : ℝ, z = a + b * complex.I ∧ 0 < a ∧ 0 < b) ↔ ∃ w : ℂ, w = complex.conj z ∧ 0 < w.re ∧ 0 < w.im :=
by
  sorry

end complex_conjugate_quadrant_l772_772736


namespace triangle_area_ratio_l772_772858

variables {A B C D O P Q : Type}
variables [convex_quad : ConvexQuadrilateral A B C D]
variables [diagonals_intersect_at O A B C D]
variables {area : Π {X Y Z : Type}, ℝ}

theorem triangle_area_ratio (A B C D O P Q : Type)
  [ConvexQuadrilateral A B C D]
  [DiagonalsIntersectAt O A B C D]
  (Area : Π {T U V : Type}, ℝ) :
  (Area {A O P} / Area {B O Q}) = (Area {A C P} / Area {B D Q}) * (Area {A B D} / Area {A B C}) :=
sorry

end triangle_area_ratio_l772_772858


namespace rebecca_perm_charge_l772_772417

theorem rebecca_perm_charge :
  ∀ (P : ℕ), (4 * 30 + 2 * 60 - 2 * 10 + P + 50 = 310) -> P = 40 :=
by
  intros P h
  sorry

end rebecca_perm_charge_l772_772417


namespace inclination_angle_range_l772_772457

theorem inclination_angle_range (α : ℝ) (h : 0 ≤ α ∧ α < π) :
  ∃ (θ : ℝ), θ ∈ [0, π] ∧
  (∀ x y : ℝ, x * Real.cos α + sqrt 3 * y + 2 = 0 ↔ 
   α ∈ [0, π/6] ∪ [5*π/6, π)) :=
sorry

end inclination_angle_range_l772_772457


namespace cost_of_10_apples_l772_772787

-- Define the price for 10 apples as a variable
noncomputable def price_10_apples (P : ℝ) : ℝ := P

-- Theorem stating that the cost for 10 apples is the provided price
theorem cost_of_10_apples (P : ℝ) : price_10_apples P = P :=
  by
    sorry

end cost_of_10_apples_l772_772787


namespace find_a_f_greater_than_1_l772_772231

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) := x^2 * Real.exp x - a * Real.log x

-- Condition: Slope at x = 1 is 3e - 1
theorem find_a (a : ℝ) (h : deriv (fun x => f x a) 1 = 3 * Real.exp 1 - 1) : a = 1 := sorry

-- Given a = 1
theorem f_greater_than_1 (x : ℝ) (hx : x > 0) : f x 1 > 1 := sorry

end find_a_f_greater_than_1_l772_772231


namespace largest_inscribed_rectangle_area_l772_772353

theorem largest_inscribed_rectangle_area : 
  ∀ (width length : ℝ) (a b : ℝ), 
  width = 8 → length = 12 → 
  (a = (8 / Real.sqrt 3) ∧ b = 2 * a) → 
  (area : ℝ) = (12 * (8 - a)) → 
  area = (96 - 32 * Real.sqrt 3) :=
by
  intros width length a b hw hl htr harea
  sorry

end largest_inscribed_rectangle_area_l772_772353


namespace ekaterina_wins_game_l772_772877

theorem ekaterina_wins_game : 
  ∀ (points : List ℕ) 
  (h_distinct: points.nodup)
  (h_length: points.length = 100),
  (∃ common_interior_point : ℤ,
   (∀ (triangles : List (List ℕ)),
    nodup_triangles triangles →
    ∀ t ∈ triangles, t.length = 3 ∧ common_interior_point ∈ ∆ t)) →
    optimal_play_winner points = ekaterina :=
by
  sorry

-- Auxiliary definitions to express game rules in Lean
def nodup_triangles := ∀ triangles : List (List ℕ), triangles.nodup
def ∆ (t : List ℕ) : set ℝ
def optimal_play_winner (points: List ℕ) : player
def ekaterina : player

end ekaterina_wins_game_l772_772877


namespace ways_to_make_50_cents_without_dimes_or_quarters_l772_772276

theorem ways_to_make_50_cents_without_dimes_or_quarters : 
  ∃ (n : ℕ), n = 1024 := 
by
  let num_ways := (2 ^ 10)
  existsi num_ways
  sorry

end ways_to_make_50_cents_without_dimes_or_quarters_l772_772276


namespace driving_time_l772_772958

-- Define the given conditions as functions and constants in Lean.
def time_to_airport : ℕ := 10
def wait_time : ℕ := 20
def time_on_airplane (D : ℕ) : ℕ := D / 3
def time_from_airplane_to_interview : ℕ := 10
def airplane_is_faster_by : ℕ := 90

-- Lean statement to prove the driving time given the conditions
theorem driving_time : ∃ (D : ℕ), D = time_to_airport + wait_time + time_on_airplane D + time_from_airplane_to_interview + airplane_is_faster_by :=
by
  let D := 195
  have : D = time_to_airport + wait_time + time_on_airplane D + time_from_airplane_to_interview + airplane_is_faster_by := sorry
  use D
  exact this

end driving_time_l772_772958


namespace find_d_l772_772867

variable (t : ℝ)
abbreviation Vector2 := ℝ × ℝ

def parametrized_line (t : ℝ) (d : Vector2) := ((2 : ℝ), 0) + t • d

theorem find_d (x y : ℝ) (d : Vector2) :
  y = (2 * x - 4) / 3 →
  (∃ t : ℝ, (x, y) = ((2 : ℝ), 0) + t • d ∧
             (x ≥ 2) ∧
             dist (x, y) (2, 0) = t) →
  d = (3 / Real.sqrt 13, 2 / Real.sqrt 13) :=
by
  sorry

end find_d_l772_772867


namespace maximum_value_of_x_minus_y_l772_772707

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l772_772707


namespace probability_of_real_roots_is_correct_l772_772531

open Real

def has_real_roots (m : ℝ) : Prop :=
  2 * m^2 - 8 ≥ 0 

def favorable_set : Set ℝ := {m | has_real_roots m}

def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_of_real_roots : ℝ :=
  interval_length (-4) (-2) + interval_length 2 3 / interval_length (-4) 3

theorem probability_of_real_roots_is_correct : probability_of_real_roots = 3 / 7 :=
by
  sorry

end probability_of_real_roots_is_correct_l772_772531


namespace remaining_tickets_l772_772974

def initial_tickets : ℝ := 49.0
def lost_tickets : ℝ := 6.0
def spent_tickets : ℝ := 25.0

theorem remaining_tickets : initial_tickets - lost_tickets - spent_tickets = 18.0 := by
  sorry

end remaining_tickets_l772_772974


namespace cheeseburger_cost_l772_772359

-- Definitions for given conditions
def milkshake_price : ℝ := 5
def cheese_fries_price : ℝ := 8
def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money := jim_money + cousin_money
def spending_percentage : ℝ := 0.80
def total_spent := spending_percentage * combined_money
def number_of_milkshakes : ℝ := 2
def number_of_cheeseburgers : ℝ := 2

-- Prove the cost of one cheeseburger
theorem cheeseburger_cost : (total_spent - (number_of_milkshakes * milkshake_price) - cheese_fries_price) / number_of_cheeseburgers = 3 :=
by
  sorry

end cheeseburger_cost_l772_772359


namespace triangle_median_equal_bc_l772_772336

-- Let \( ABC \) be a triangle, \( AB = 2 \), \( AC = 3 \), and the median from \( A \) to \( BC \) has the same length as \( BC \).
theorem triangle_median_equal_bc (A B C M : Type) (AB AC BC AM : ℝ) 
  (hAB : AB = 2) (hAC : AC = 3) 
  (hMedian : BC = AM) (hM : M = midpoint B C) :
  BC = real.sqrt (26 / 5) :=
by sorry

end triangle_median_equal_bc_l772_772336


namespace cathy_wins_probability_l772_772147

theorem cathy_wins_probability : 
  (∑' (n : ℕ), (1 / 6 : ℚ)^3 * (5 / 6)^(3 * n)) = 1 / 91 
:= by sorry

end cathy_wins_probability_l772_772147


namespace max_difference_value_l772_772677

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l772_772677


namespace prove_f_l772_772281

-- Define the condition given in the problem
def condition (f : ℝ → ℝ) := ∀ x : ℝ, 2 * f(x) - f(-x) = 3 * x + 1

-- State the theorem to be proved
theorem prove_f (f : ℝ → ℝ) (h: condition f) : ∀ x : ℝ, f(x) = x + 1 :=
by 
  sorry

end prove_f_l772_772281


namespace condition_necessary_but_not_sufficient_l772_772927

variable (a b : ℝ)

theorem condition_necessary_but_not_sufficient (h : a ≠ 1 ∨ b ≠ 2) : (a + b ≠ 3) ∧ ¬(a + b ≠ 3 → a ≠ 1 ∨ b ≠ 2) :=
by
  --Proof will go here
  sorry

end condition_necessary_but_not_sufficient_l772_772927


namespace kim_driving_speed_l772_772363

open Nat
open Real

noncomputable def driving_speed (distance there distance_back time_spent traveling_time total_time: ℝ) : ℝ :=
  (distance + distance_back) / traveling_time

theorem kim_driving_speed:
  ∀ (distance there distance_back time_spent traveling_time total_time: ℝ),
  distance = 30 →
  distance_back = 30 * 1.20 →
  total_time = 2 →
  time_spent = 0.5 →
  traveling_time = total_time - time_spent →
  driving_speed distance there distance_back time_spent traveling_time total_time = 44 :=
by
  intros
  simp only [driving_speed]
  sorry

end kim_driving_speed_l772_772363


namespace solve_integral_equation_l772_772843

def integral_equation_solution (ϕ : ℝ → ℝ) : Prop :=
  ∀ x > 0, (1 / Real.sqrt (π * x)) * ∫ t in 0..∞, Real.exp (- t^2 / (4 * x)) * ϕ t = 1

theorem solve_integral_equation : integral_equation_solution (fun x => 1) :=
by
  sorry

end solve_integral_equation_l772_772843


namespace washer_stack_height_l772_772951

theorem washer_stack_height :
  ∀ (n: ℕ) (d thick : ℕ → ℝ),
    (d 0 = 10) ∧ (thick 0 = 1) ∧ (∀ k, d (k + 1) = d k - 0.5) ∧ (∀ k, thick (k + 1) = thick k - 0.1) ∧ 
    (∃ m, d m = 2) → ∑ i in finset.range 17, thick i = 11.9 :=
begin
  -- sorry placeholder for the proof
  sorry
end

end washer_stack_height_l772_772951


namespace parabola_point_x_correct_l772_772630

noncomputable def parabola_point_x : ℝ :=
  let y_sq := (4 : ℝ) in
  let p    := (1 : ℝ) in
  let focus := (1, 0) in
  let distance (M : ℝ × ℝ) (F : ℝ × ℝ) : ℝ :=
    real.sqrt ((M.1 - F.1)^2 + M.2^2) in
  let parabola (M : ℝ × ℝ) :=
    M.2^2 = 4 * M.1 in
  let on_parabola (M : ℝ × ℝ) :=
    parabola M ∧ distance M focus = 4 in
  let M := (λ x : ℝ, (x, real.sqrt (4 * x))) in
  classical.some (exists.intro 3 (by
    have : focus = (1 : ℝ, 0 : ℝ) := rfl
    unfold distance parabola,
    existsi (3, real.sqrt (4 * 3)),
    calc distance (3, real.sqrt (4 * 3)) focus
        = real.sqrt ((3 - 1)^2 + (real.sqrt (4 * 3))^2) : rfl
    ... = real.sqrt (4 + 12) : by simp
    ... = real.sqrt 16 : rfl
    ... = 4 : real.sqrt_eq_rfl.mpr rfl,
    unfold on_parabola,
    constructor; simp [parabola]; sorry))

theorem parabola_point_x_correct : parabola_point_x = 3 :=
sorry

end parabola_point_x_correct_l772_772630


namespace range_of_length_TS_l772_772716

noncomputable def point_on_circle (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y - 2) ^ 2 = 1

noncomputable def rotated_point (x y : ℝ) : ℝ × ℝ :=
  (-y, x)

noncomputable def reflected_point (x y : ℝ) : ℝ × ℝ :=
  (6 - x, -y)

noncomputable def length_TS (x y : ℝ) : ℝ :=
  let (s1, s2) := rotated_point x y in
  let (t1, t2) := reflected_point x y in
  Real.sqrt ((t1 - s1) ^ 2 + (t2 - s2) ^ 2)

theorem range_of_length_TS {x y : ℝ} :
  point_on_circle x y →
  √2 * (√26 - 1) ≤ length_TS x y ∧ length_TS x y ≤ √2 * (√26 + 1) :=
by
  sorry

end range_of_length_TS_l772_772716


namespace cannot_form_isosceles_triangle_l772_772876

theorem cannot_form_isosceles_triangle :
  ¬ ∃ (sticks : Finset ℕ) (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧
  a + b > c ∧ a + c > b ∧ b + c > a ∧ -- Triangle inequality
  (a = b ∨ b = c ∨ a = c) ∧ -- Isosceles condition
  sticks ⊆ {1, 2, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9} := sorry

end cannot_form_isosceles_triangle_l772_772876


namespace books_sold_over_summer_l772_772446

theorem books_sold_over_summer (n l t : ℕ) (h1 : n = 37835) (h2 : l = 143) (h3 : t = 271) : 
  t - l = 128 :=
by
  sorry

end books_sold_over_summer_l772_772446


namespace eq_root_condition_l772_772252

theorem eq_root_condition (k : ℝ) 
    (h_discriminant : -4 * k + 5 ≥ 0)
    (h_roots : ∃ x1 x2 : ℝ, 
        (x1 + x2 = 1 - 2 * k) ∧ 
        (x1 * x2 = k^2 - 1) ∧ 
        (x1^2 + x2^2 = 16 + x1 * x2)) :
    k = -2 :=
sorry

end eq_root_condition_l772_772252


namespace common_chord_l772_772043

theorem common_chord (circle1 circle2 : ℝ × ℝ → Prop)
  (h1 : ∀ x y, circle1 (x, y) ↔ x^2 + y^2 + 2 * x = 0)
  (h2 : ∀ x y, circle2 (x, y) ↔ x^2 + y^2 - 4 * y = 0) :
  ∀ x y, circle1 (x, y) ∧ circle2 (x, y) ↔ x + 2 * y = 0 := 
by
  sorry

end common_chord_l772_772043


namespace find_B_and_dot_product_l772_772759

-- Define triangle sides and condition
variables {a b c : ℝ} {A B C : ℝ}
axiom in_triangle_ABC : (2 * a + c) * cos B + b * cos C = 0

-- Given values and conditions
def a_val : ℝ := 3
def area_ABC : ℝ := (3 * real.sqrt 3) / 2

-- Prove the two-part statement
theorem find_B_and_dot_product :
  (B = 2 * real.pi / 3) ∧
  (a = a_val → area_ABC = (1 / 2) * a * c * real.sin B → 
  ∃ c_value : ℝ, (c = c_value) ∧ c_value = 2 ∧ 
  (∃ dot_product_value : ℝ, \overrightarrow{AB} \cdot \overrightarrow{BC} = dot_product_value ∧ dot_product_value = 3)) :=
sorry

end find_B_and_dot_product_l772_772759


namespace probability_all_same_color_l772_772559

def color_blocks :=
  (Ang_blocks : Fin 6 → Fin 6) × (Ben_blocks : Fin 6 → Fin 6) × (Jasmin_blocks : Fin 6 → Fin 6)

theorem probability_all_same_color :
  let boxes : Fin 5 → color_blocks → Fin 5 := λ i cb =>
    (cb.1 i, cb.2 i, cb.3 i)
  ∧ ∀ i j, (fin_sum (boxes j).val + fin_sum (boxes j).val + fin_sum (boxes j).val) ≤ 4
  ∧ ∀ i j, fin_sum (boxes j).val ≤ 2
  ∧ ∀ i j, fin_sum (boxes j).val ≤ 2
  ∧ ∀ i j, fin_sum (boxes j).val ≤ 2
  ∧ (∀ j, (cb.1 j = cb.2 j ∧ cb.2 j = cb.3 j))
  ∧ (∀ j, (cb.1 j = cb.2 j ∧ cb.2 j = cb.3 j)),
  let probability := 
    (1 - 5 * (6 / 252 ^ 3))
  in
  probability = 1.86891 * 10 ^ (-6) := by
  sorry

end probability_all_same_color_l772_772559


namespace probability_same_number_of_multiples_l772_772563

theorem probability_same_number_of_multiples (n : ℕ) (hn : n < 500 ∧ n > 0):
  (n % 20 = 0) → (n % 30 = 0) → n % 60 = 0 → 
  ((∃ k : ℕ, k ≤ 500 ∧ (k % 20 = 0)) →
   (∃ k : ℕ, k ≤ 500 ∧ (k % 30 = 0)) →
   (∃ k : ℕ, k ≤ 500 ∧ (k % 60 = 0))) → 
  ∃ p : ℚ, p = 1 / 50 := 
begin
  intros h1 h2 h3 h_mult20 h_mult30 h_mult60,
  sorry
end

end probability_same_number_of_multiples_l772_772563


namespace zero_of_f_l772_772582

def f (x : ℝ) := Real.exp x + x - 2

theorem zero_of_f :
  (∃ c : ℝ, f c = 0) ∧ (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (f 0 = -1) ∧ (f 1 = Real.exp 1 + 1 - 2) →
  ∃! x : ℝ, f x = 0 := by
  sorry

end zero_of_f_l772_772582


namespace Zelda_probability_success_l772_772498

variable (P : ℝ → ℝ)
variable (X Y Z : ℝ)

theorem Zelda_probability_success :
  P X = 1/3 ∧ P Y = 1/2 ∧ (P X) * (P Y) * (1 - P Z) = 0.0625 → P Z = 0.625 :=
by
  sorry

end Zelda_probability_success_l772_772498


namespace smallest_integer_gt_sqrt3_l772_772462

theorem smallest_integer_gt_sqrt3 : ∃ (n : ℤ), n = 2 ∧ (√3 < n) ∧ (∀ m : ℤ, m < 2 → √3 ≥ m) :=
by
  sorry

end smallest_integer_gt_sqrt3_l772_772462


namespace arc_length_one_radian_l772_772058

-- Given definitions and conditions
def radius : ℝ := 6370
def angle : ℝ := 1

-- Arc length formula
def arc_length (R α : ℝ) : ℝ := R * α

-- Statement to prove
theorem arc_length_one_radian : arc_length radius angle = 6370 := 
by 
  -- Proof goes here
  sorry

end arc_length_one_radian_l772_772058


namespace number_of_ordered_triples_l772_772212

-- Definitions for the conditions
def a_is (a : ℕ) : Prop :=
  ∃ j k l : ℕ, a = 2^j * 3^k * 5^l

def b_is (b : ℕ) : Prop :=
  ∃ m n o : ℕ, b = 2^m * 3^n * 5^o

def c_is (c : ℕ) : Prop :=
  ∃ p q r : ℕ, c = 2^p * 3^q * 5^r

-- Definition for lcm
def lcm (x y : ℕ) : ℕ := sorry -- lcm can be defined by its properties in a formal proof

-- Theorem statement
theorem number_of_ordered_triples (a b c : ℕ) : 
  a_is a → b_is b → c_is c → 
  lcm a b = 1200 → lcm b c = 1800 → lcm c a = 2400 → 
  (∃! t : ℕ, t = 1) :=
sorry

end number_of_ordered_triples_l772_772212


namespace negation_universal_proposition_l772_772737

theorem negation_universal_proposition : 
  (¬ ∀ x : ℝ, x^2 - x < 0) = ∃ x : ℝ, x^2 - x ≥ 0 :=
by
  sorry

end negation_universal_proposition_l772_772737


namespace smallest_positive_root_range_l772_772172

noncomputable def smallest_positive_root (b2 b1 b0 : ℝ) (h2 : |b2| < 3) (h1 : |b1| < 3) (h0 : |b0| < 3) : ℝ :=
if s : ∃ x : ℝ, x > 0 ∧ x^3 + b2 * x^2 + b1 * x + b0 = 0 then 
  classical.some s 
else 
  0

theorem smallest_positive_root_range :
  ∀ (b2 b1 b0 : ℝ),
    |b2| < 3 → 
    |b1| < 3 → 
    |b0| < 3 →
    (∃ x : ℝ, x > 0 ∧ x^3 + b2 * x^2 + b1 * x + b0 = 0) →
    let s := smallest_positive_root b2 b1 b0 in
    ∃ s, (3/2 : ℝ) < s ∧ s < 2 :=
by
  sorry

end smallest_positive_root_range_l772_772172


namespace sufficient_but_not_necessary_condition_l772_772242

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (-2 ≤ x ∧ x ≤ 2) → (x ≤ a))
  → (∃ x : ℝ, (x ≤ a ∧ ¬((-2 ≤ x ∧ x ≤ 2))))
  → (a ≥ 2) :=
by
  intros h1 h2
  sorry

end sufficient_but_not_necessary_condition_l772_772242


namespace cricket_run_rate_l772_772504

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target_runs : ℝ) (first_overs : ℝ) (remaining_overs : ℝ):
  run_rate_first_10_overs = 6.2 → 
  target_runs = 282 →
  first_overs = 10 →
  remaining_overs = 40 →
  (target_runs - run_rate_first_10_overs * first_overs) / remaining_overs = 5.5 :=
by
  intros h1 h2 h3 h4
  -- Insert proof here
  sorry

end cricket_run_rate_l772_772504


namespace math_problem_l772_772815

theorem math_problem (a b : ℕ) (x y : ℚ) (h1 : a = 10) (h2 : b = 11) (h3 : x = 1.11) (h4 : y = 1.01) :
  ∃ k : ℕ, k * y = 2.02 ∧ (a * x + b * y - k * y = 20.19) :=
by {
  sorry
}

end math_problem_l772_772815


namespace find_x_and_angle_l772_772239

theorem find_x_and_angle 
  (A P : set ℝ^3) 
  (α : set ℝ^3)
  (PA : ℝ × ℝ × ℝ)
  (x : ℝ) 
  (hA_in_alpha : A ⊆ α)
  (hP_not_in_alpha : ¬ P ⊆ α)
  (hPA_components : PA = (-real.sqrt 3 / 2, 1 / 2, x))
  (hx_pos : x > 0)
  (hPA_norm : ∥PA∥ = real.sqrt 3)
  (n : ℝ × ℝ × ℝ)
  (hn_components : n = (0, -1 / 2, -real.sqrt 2)) :
  (x = real.sqrt 2) ∧ (let θ := real.arcsin (real.sqrt 3 / 2) in θ = real.pi / 3) := 
by 
  sorry

end find_x_and_angle_l772_772239


namespace trapezium_side_length_l772_772200

theorem trapezium_side_length (a b h A x : ℝ) 
  (ha : a = 20) (hh : h = 15) (hA : A = 285) 
  (h_formula : A = 1 / 2 * (a + b) * h) : 
  b = 18 :=
by
  sorry

end trapezium_side_length_l772_772200


namespace piravena_total_round_trip_cost_l772_772171

noncomputable def piravena_round_trip_cost : ℝ :=
  let distance_AB := 4000
  let bus_cost_per_km := 0.20
  let flight_cost_per_km := 0.12
  let flight_booking_fee := 120
  let flight_cost := distance_AB * flight_cost_per_km + flight_booking_fee
  let bus_cost := distance_AB * bus_cost_per_km
  flight_cost + bus_cost

theorem piravena_total_round_trip_cost : piravena_round_trip_cost = 1400 := by
  -- Problem conditions for reference:
  -- distance_AC = 3000
  -- distance_AB = 4000
  -- bus_cost_per_km = 0.20
  -- flight_cost_per_km = 0.12
  -- flight_booking_fee = 120
  -- Piravena decides to fly from A to B but returns by bus
  sorry

end piravena_total_round_trip_cost_l772_772171


namespace tan_addition_identity_l772_772189

-- Define the angles involved as degrees (converted to radians)
def angle_12_deg : ℝ := real.pi * 12 / 180
def angle_18_deg : ℝ := real.pi * 18 / 180

-- Lean statement for the proof problem
theorem tan_addition_identity :
  (real.tan angle_12_deg + real.tan angle_18_deg) / (1 - real.tan angle_12_deg * real.tan angle_18_deg) = real.tan (angle_12_deg + angle_18_deg) :=
sorry

end tan_addition_identity_l772_772189


namespace eq_of_divisibility_condition_l772_772004

theorem eq_of_divisibility_condition (a b : ℕ) (h : ∃ᶠ n in Filter.atTop, (a^n + b^n) ∣ (a^(n+1) + b^(n+1))) : a = b :=
sorry

end eq_of_divisibility_condition_l772_772004


namespace interval_of_monotonicity_range_of_f_for_a_eq_neg1_sum_g_inequality_l772_772256

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * x * Real.log x

def g (x : ℝ) (a : ℝ) : ℝ := (f x a) / x - 1

theorem interval_of_monotonicity 
  (a : ℝ) (h : a ≤ 0) : 
  (a = 0 -> ∀ x > 0, f' x 0 > 0) ∧ 
  (a < 0 -> ∀ x > 0, (x ≤ Real.exp ((1 - a)/a) -> f' x a < 0) ∧ 
  (x > Real.exp ((1 - a)/a) -> f' x a > 0)) := 
sorry

theorem range_of_f_for_a_eq_neg1 :
  ∀ x ∈ Set.Icc (Real.exp (-Real.exp 1)) Real.exp 1, f x (-1) ∈ Set.Icc (-1 / Real.exp 2^2) (2 * Real.exp 1) :=
sorry

theorem sum_g_inequality :
  ∀ n : ℕ, 2 ≤ n -> ∑ k in Finset.range n, 1 / g k (-1) > (3 * n^2 - n - 2) / (n * (n + 1)) :=
sorry

end interval_of_monotonicity_range_of_f_for_a_eq_neg1_sum_g_inequality_l772_772256


namespace find_d_l772_772153

/-- Given the sine function of the form y = a * sin(b * x + c) + d, 
    with positive constants a, b, c, and d. If the function oscillates between 4 and -2, 
    then the vertical shift d is equal to 1. -/
theorem find_d (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
(h_oscillates : ∀ x, -2 ≤ a * real.sin (b * x + c) + d ∧ a * real.sin (b * x + c) + d ≤ 4) 
: d = 1 :=
by {
  sorry
}

end find_d_l772_772153


namespace sum_of_g35_l772_772110

theorem sum_of_g35 (g : ℤ → ℤ) (h1 : ∀ x y : ℤ, g(x) + g(y) = g(x + y) - x * y) (h2 : g(23) = 0) :
  g(35) = 210 :=
sorry

end sum_of_g35_l772_772110


namespace orthogonal_projection_is_orthocenter_construct_equidistant_points_l772_772875

noncomputable def plane (V : Type _) [InnerProductSpace ℝ V] :=
  {p : Submodule ℝ V // ∃ (x y : V), IsBasis ℝ ![x, y] ∧ span ℝ ({x, y} : Set V) = p}

noncomputable def triangle {V : Type _} [InnerProductSpace ℝ V] (p : plane V) :=
  {A B C : V // A ∈ p.1 ∧ B ∈ p.1 ∧ C ∈ p.1}

noncomputable def orthocenter {V : Type _} [InnerProductSpace ℝ V] (T : triangle V) : V := sorry

noncomputable def projection {V : Type _} [InnerProductSpace ℝ V] (O : V) (p : plane V) : V := sorry

noncomputable def equidistant_points {V : Type _} [InnerProductSpace ℝ V]
  (O : V) (x y z : V) (d : ℝ) : Set V := sorry

theorem orthogonal_projection_is_orthocenter {V : Type _} [InnerProductSpace ℝ V]
  (O : V) (p : plane V) (T : triangle p) :
  projection O p = orthocenter T := sorry

theorem construct_equidistant_points {V : Type _} [InnerProductSpace ℝ V]
  (O : V) (x y z : V) (d : ℝ) (p : plane V) (A B C M N P : V)
  (hx : A = O + d • x) (hy : B = O + d • y) (hz : C = O + d • z)
  (hA : A ∈ p.1) (hB : B ∈ p.1) (hC : C ∈ p.1) :
  {M, N, P} = equidistant_points O x y z d := sorry

end orthogonal_projection_is_orthocenter_construct_equidistant_points_l772_772875


namespace probability_inequality_l772_772803

noncomputable def Xi : Type := ℝ

axiom normal_distribution (μ σ : ℝ) : Xi → Prop

theorem probability_inequality (σ : ℝ) (h₁ : ∀ ξ, normal_distribution 2 σ ξ)
    (h₂ : P (λ ξ, ξ > 4) = 0.1) : P (λ ξ, ξ < 0) = 0.1 :=
by
  sorry

end probability_inequality_l772_772803


namespace distinct_roots_iff_l772_772580

def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 + |x1| = 2 * Real.sqrt (3 + 2*a*x1 - 4*a)) ∧
                       (x2 + |x2| = 2 * Real.sqrt (3 + 2*a*x2 - 4*a))

theorem distinct_roots_iff (a : ℝ) :
  has_two_distinct_roots a ↔ (a ∈ Set.Ioo 0 (3 / 4 : ℝ) ∨ 3 < a) :=
sorry

end distinct_roots_iff_l772_772580


namespace cristine_initial_lemons_l772_772173

theorem cristine_initial_lemons (L : ℕ) (h : (3 / 4 : ℚ) * L = 9) : L = 12 :=
sorry

end cristine_initial_lemons_l772_772173


namespace number_of_teams_with_phd_l772_772922

-- Define the total number of engineers
def total_engineers : ℕ := 8

-- Define the number of PhD engineers
def phd_engineers : ℕ := 3

-- Define the number of non-PhD engineers
def non_phd_engineers : ℕ := 5

-- Define a function for the binomial coefficient (combinations)
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Prove the total number of different teams that can be chosen
theorem number_of_teams_with_phd : C(phd_engineers, 1) * C(non_phd_engineers, 2) +
                                   C(phd_engineers, 2) * C(non_phd_engineers, 1) +
                                   C(phd_engineers, 3) = 46 := by
  sorry

end number_of_teams_with_phd_l772_772922


namespace measure_of_side_XY_l772_772890

theorem measure_of_side_XY 
  (a b c : ℝ) 
  (Area : ℝ)
  (h1 : a = 30)
  (h2 : b = 60)
  (h3 : c = 90)
  (h4 : a + b + c = 180)
  (h_area : Area = 36)
  : (∀ (XY YZ XZ : ℝ), XY = 4.56) :=
by
  sorry

end measure_of_side_XY_l772_772890


namespace team_C_games_played_is_60_l772_772429

variable (C_games D_games : ℕ)

-- Condition 1: Team C has won 3/4 of its games.
def WinRatioC : ℚ := 3 / 4

-- Condition 2: Team D has won 2/3 of its games.
def WinRatioD : ℚ := 2 / 3

-- Condition 3: For every 4 games team C wins, team D wins 3 games more than team C.
def WinDifference (c_wins: ℕ) (d_wins: ℕ) : Prop :=
  ∃ k : ℕ, d_wins = c_wins + 3 * k ∧ c_wins = 4 * k

-- Condition 4: Team C has played 12 fewer games than team D.
def GamesDifference : Prop := C_games + 12 = D_games

-- The main statement
theorem team_C_games_played_is_60 (h1 : WinRatioC = 3 / 4) 
                                  (h2 : WinRatioD = 2 / 3) 
                                  (h3 : ∃ (c_wins d_wins : ℕ), WinDifference c_wins d_wins)
                                  (h4 : GamesDifference) :
  C_games = 60 :=
sorry

end team_C_games_played_is_60_l772_772429


namespace L_shaped_solid_surface_area_l772_772221

-- Definitions of the conditions
def unit_cube : Type := PUnit

def L_shaped_solid : Type := 
  List.unit_cube    -- base layer of 8 cubes
  × List.unit_cube  -- second layer of 6 cubes

def base_layer (s : L_shaped_solid) : List unit_cube := s.1

def second_layer (s : L_shaped_solid) : List unit_cube := s.2

-- Area calculation
noncomputable def surface_area (s : L_shaped_solid) : ℕ :=
  let base_area_top := 4  -- Top of the first four cubes in the base layer
  let base_area_front_back := 16  -- Front and back of base layer
  let base_area_sides := 2  -- Sides of base layer
  let top_layer_top := 6  -- Top of the second layer
  let top_layer_bottom := 2  -- Bottom of the first two cubes in the second layer
  let top_layer_front_back := 12  -- Front and back of the second layer
  let top_layer_sides := 2  -- Sides of the second layer
  base_area_top + base_area_front_back + base_area_sides +
  top_layer_top + top_layer_bottom + top_layer_front_back + top_layer_sides

-- Theorem statement
theorem L_shaped_solid_surface_area (s : L_shaped_solid) :
  surface_area(s) = 44 := by
  sorry

end L_shaped_solid_surface_area_l772_772221


namespace change_in_max_value_l772_772093

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem change_in_max_value (a b c : ℝ) (h1 : -b^2 / (4 * (a + 1)) + c = -b^2 / (4 * a) + c + 27 / 2)
  (h2 : -b^2 / (4 * (a - 4)) + c = -b^2 / (4 * a) + c - 9) :
  -b^2 / (4 * (a - 2)) + c = -b^2 / (4 * a) + c - 27 / 4 :=
by
  sorry

end change_in_max_value_l772_772093


namespace trapezoid_area_242_l772_772332

-- Variables and assumptions
variables {A B C D E : Type}
variables [EuclideanSpace A B C D E]
variables (area_ABE : ℝ) (area_CDE : ℝ)

-- Given conditions
def given_conditions (area_ABE : ℝ) (area_CDE : ℝ) :=
  area_ABE = 72 ∧ area_CDE = 50

-- Proof problem to show the total area of trapezoid ABCD is 242
theorem trapezoid_area_242 (h : given_conditions area_ABE area_CDE) : ∃ area_trapezoid : ℝ, area_trapezoid = 242 :=
by
  sorry

end trapezoid_area_242_l772_772332


namespace update_year_l772_772524

def a (n : ℕ) : ℕ :=
  if n ≤ 7 then 2 * n + 2 else 16 * (5 / 4) ^ (n - 7)

noncomputable def S (n : ℕ) : ℕ :=
  if n ≤ 7 then n^2 + 3 * n else 80 * ((5 / 4) ^ (n - 7)) - 10

noncomputable def avg_maintenance_cost (n : ℕ) : ℚ :=
  (S n : ℚ) / n

theorem update_year (n : ℕ) (h : avg_maintenance_cost n > 12) : n = 9 :=
  by
  sorry

end update_year_l772_772524


namespace tan_theta_solution_l772_772226

theorem tan_theta_solution (θ : ℝ) (h : 2 * Real.sin θ = 1 + Real.cos θ) :
  Real.tan θ = 0 ∨ Real.tan θ = 4 / 3 :=
sorry

end tan_theta_solution_l772_772226


namespace number_of_different_groups_of_two_marbles_l772_772474

-- Definitions for the marbles
inductive Marble
| Red | Green | Blue | Yellow | Yellow | White | White

-- Define the list of marbles
def tomsMarbles : List Marble := 
  [Marble.Red, Marble.Green, Marble.Blue, Marble.Yellow, Marble.Yellow, Marble.White, Marble.White]

-- Prove the number of different groups of two marbles
theorem number_of_different_groups_of_two_marbles (m : List Marble) (h : m = tomsMarbles) :
  (Finset.card (Finset.image (λ m1 m2, if m1 ≤ m2 then (m1, m2) else (m2, m1)) (Finset.of_list m).product (Finset.of_list m))) = 12 :=
sorry

end number_of_different_groups_of_two_marbles_l772_772474


namespace expected_value_correct_l772_772481

noncomputable def expected_value_of_total_roll : ℝ :=
  let probability_all_different := 120 / 216
  let probability_at_least_two_same := 1 - probability_all_different
  let expected_value_single_roll := (1 + 2 + 3 + 4 + 5 + 6) / 6
  let expected_sum_three_dice := 3 * expected_value_single_roll
  let expected_value_when_all_different := 10.5
  let expected_value_when_at_least_two_same := sorry -- Placeholder for the exact calculation
  assume_unrolling := sorry -- Modeling the recursive nature accurately

  -- Approximate total expected value combining the scenarios
  23.625

theorem expected_value_correct :
  expected_value_of_total_roll = 23.625 :=
  sorry

end expected_value_correct_l772_772481


namespace part1_part2_l772_772513

-- Part (1) statement
theorem part1 {x : ℝ} : (|x - 1| + |x + 2| >= 5) ↔ (x <= -3 ∨ x >= 2) := 
sorry

-- Part (2) statement
theorem part2 (a : ℝ) : (∀ x : ℝ, (|a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3)) → a = -3 :=
sorry

end part1_part2_l772_772513


namespace total_accidents_l772_772593

theorem total_accidents (t : ℕ) (h1 : every_n_seconds car_collision 3 t)
                         (h2 : every_n_seconds big_crash 7 t)
                         (h3 : every_n_seconds multi_vehicle_pileup 15 t)
                         (h4 : every_n_seconds massive_accident 25 t)
                         (h5 : t = 240) : 
   total_accidents t = 139 := 
by 
  sorry

end total_accidents_l772_772593


namespace coefficient_x3_expansion_l772_772631

noncomputable def normal_mean (μ : ℝ) (σ : ℝ) : ℝ := μ
noncomputable def normal_variance (μ σ : ℝ) : ℝ := σ^2
noncomputable def normal_distribution (X : ℝ) (μ σ : ℝ) := 
  ∀ x : ℝ, P(X ≤ x) = ∫ t in -∞..x, (1/(σ * sqrt (2 * π))) * exp (-(t-μ)^2 / (2*σ^2))

theorem coefficient_x3_expansion {X : ℝ} {a : ℝ} (hX : normal_distribution X 2 3) 
  (hProb : P(X ≤ 1) = P(X ≥ a)) : 
  a = 3 ∧ (∑ k in finset.range 6, choose 5 k * 3^(5-k) * (-1)^k) * 
  (∑ l in finset.range 3, choose 2 l * (6)^(2-l)) == 1620 :=
by
  sorry

end coefficient_x3_expansion_l772_772631


namespace skew_lines_cannot_project_to_two_points_l772_772080

noncomputable def skew_lines (l1 l2 : ℝ → ℝ^3) : Prop :=
(∀ t₁ t₂, l1 t₁ - l2 t₂ ≠ 0) ∧ ¬ ∃ t₁ t₂, l1 t₁ = l2 t₂

theorem skew_lines_cannot_project_to_two_points (l1 l2 : ℝ → ℝ^3) (h1 : skew_lines l1 l2) 
: ¬ ∃ p1 p2 : ℝ^2, ∀ t1 t2, (project (l1 t1) = p1) ∧ (project (l2 t2) = p2) := sorry

end skew_lines_cannot_project_to_two_points_l772_772080


namespace red_marbles_l772_772567

theorem red_marbles (R B : ℕ) (h₁ : B = R + 24) (h₂ : B = 5 * R) : R = 6 := by
  sorry

end red_marbles_l772_772567


namespace ratio_speed_l772_772960

variable (L : ℝ) (vp vc t : ℝ) 

-- Conditions
axiom (H1 : 0 < L)
axiom (H2 : vp = (2 / 5) * L / t)
axiom (H3 : vc = L / t)

-- Proof Target: the ratio of the speed of the car to the speed of the pedestrian is 5.
theorem ratio_speed (H1: L > 0)  (H2 : vp = (2 / 5) * L / t) (H3 : vc = L / t) : vc / vp = 5 := 
sorry

end ratio_speed_l772_772960


namespace proof_problem_l772_772234

noncomputable def LineEquation (t θ : ℝ) (hθ : 0 < θ ∧ θ < π) : Set (ℝ × ℝ) :=
{p | p.1 = t * sin θ ∧ p.2 = 2 + t * cos θ}

noncomputable def CurveEquation (ρ θ : ℝ) (hθ : 0 < θ ∧ θ < π) : Prop :=
ρ * cos θ ^ 2 = 8 * sin θ

noncomputable def GeneralLineEquation (x y θ : ℝ) (hθ : 0 < θ ∧ θ < π) : Prop :=
x * cos θ - y * sin θ + 2 * cos θ = 0

noncomputable def RectangularCurveEquation (x y : ℝ) : Prop :=
x ^ 2 = 8 * y

theorem proof_problem (t θ ρ : ℝ) (x y : ℝ) (hθ : 0 < θ ∧ θ < π) :
  (LineEquation t θ hθ) → (CurveEquation ρ θ hθ) →
  (GeneralLineEquation x y θ hθ) ∧ (RectangularCurveEquation x y) ∧
  (∃ A B, x = A ∧ y = B ∧ |A - B| = 8) :=
by
  sorry

end proof_problem_l772_772234


namespace incorrect_conclusion_d_l772_772220

theorem incorrect_conclusion_d
    (x y : ℕ → ℕ)
    (h_total_wins_losses : ∑ i in finset.range 4, x i = 6 ∧ ∑ i in finset.range 4, y i = 6)
    (h_individual_wins_losses : ∀ i ∈ finset.range 4, x i + y i = 3) :
    (¬ (x 0 ^ 2 + x 1 ^ 2 + x 2 ^ 2 + x 3 ^ 2 = (3 - y 0) ^ 2 + (3 - y 1) ^ 2 + (3 - y 2) ^ 2 + (3 - y 3) ^ 2)).
Proof :=
  sorry

end incorrect_conclusion_d_l772_772220


namespace unique_property_p_l772_772108

def property_P (n : ℕ) : Prop :=
  ∃ (k : ℕ) (n_1 n_2 ⋯ n_k : ℕ), 4 ≤ n_1 ∧ 4 ≤ n_2 ∧ ⋯ ∧ 4 ≤ n_k ∧
    n = (n_1 * n_2 * ⋯ * n_k) ∧
    n = 2^((1 / (k^k * (n_1 - 1) * (n_2 - 1) * ⋯ * (n_k - 1))) - 1)

theorem unique_property_p : ∀ (n : ℕ), property_P n → n = 7 :=
by
  sorry

end unique_property_p_l772_772108


namespace find_b_l772_772735

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x b : ℝ) : ℝ := Real.log x + b

theorem find_b (b : ℝ) :
  (∃ (x1 x2 : ℝ), (f x1 = x2⁻¹) ∧ (1 - x1 = 0) ∧ (Real.log x2 + b - 1 = 0)) →
  b = 2 :=
by
  sorry

end find_b_l772_772735


namespace troll_problem_l772_772188

theorem troll_problem (T : ℕ) (h : 6 + T + T / 2 = 33) : 4 * 6 - T = 6 :=
by sorry

end troll_problem_l772_772188


namespace intersections_of_absolute_value_functions_l772_772180

theorem intersections_of_absolute_value_functions : 
  (∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|4 * x + 3|) → ∃ (x y : ℝ), (x = -1 ∧ y = 1) ∧ ¬(∃ (x' y' : ℝ), y' = |3 * x' + 4| ∧ y' = -|4 * x' + 3| ∧ (x' ≠ -1 ∨ y' ≠ 1)) :=
by
  sorry

end intersections_of_absolute_value_functions_l772_772180


namespace area_of_triangle_PQR_l772_772054

-- Definition of the initial point P
def P : ℝ × ℝ := (2, 5)

-- Definition of point Q as the reflection of P over the x-axis
def Q : ℝ × ℝ := (2, -5)

-- Definition of point R as the reflection of Q over the line y = x
def R : ℝ × ℝ := (-5, 2)

-- Function to calculate the area of a triangle given three points (x1, y1), (x2, y2), (x3, y3)
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs

-- Prove that the area of triangle PQR is 35
theorem area_of_triangle_PQR : triangle_area P.1 P.2 Q.1 Q.2 R.1 R.2 = 35 := by
  -- Here we skip the proof, which would involve calculating the area step by step
  sorry

end area_of_triangle_PQR_l772_772054


namespace expenditure_july_l772_772146

def avg_expenditure_jan_to_jun : ℝ := 4200
def expenditure_january : ℝ := 1200
def avg_expenditure_feb_to_jul : ℝ := 4250

theorem expenditure_july 
  (avg_expenditure_jan_to_jun : ℝ) 
  (expenditure_january : ℝ) 
  (avg_expenditure_feb_to_jul : ℝ) :
  let expenditure_feb_to_jun := 6 * avg_expenditure_jan_to_jun - expenditure_january,
      expenditure_feb_to_jul := 6 * avg_expenditure_feb_to_jul in
  expenditure_feb_to_jul - expenditure_feb_to_jun = 1500 :=
by
  sorry

end expenditure_july_l772_772146


namespace main_theorem_l772_772730

-- Define the initially given function
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := sin (ω * x + φ) - cos (ω * x + φ)

-- Define the main proof theorem with given conditions
theorem main_theorem (ω φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) (h3 : 0 < ω)
  (h4 : ∀ x : ℝ, f x ω φ = f (-x) ω φ) 
  (h5 : ∀ a b : ℝ, 0 < a → b > 0 → a = π / b → ω = 2) :
  (∀ x : ℝ, f x ω φ = 2 * cos (2 * x)) 
  ∧ 
  (finally_g : ∀ x : ℝ, g x = 2 * cos (x / 2 - (π / 4))) 
  ∧ 
  (∀ k : ℤ, [ (4:ℝ) * k * π + π, 4 * k * π + 3 * π ] ⊆ decreasing_intervals_for_g) :=
sorry

end main_theorem_l772_772730


namespace compare_sizes_l772_772984

theorem compare_sizes :
  (- (3 / 4) < | - (7 / 5) |) :=
by 
  sorry

end compare_sizes_l772_772984


namespace books_ratio_l772_772550

-- Definitions based on the conditions
def Alyssa_books : Nat := 36
def Nancy_books : Nat := 252

-- Statement to prove
theorem books_ratio :
  (Nancy_books / Alyssa_books) = 7 := 
sorry

end books_ratio_l772_772550


namespace not_possible_to_replace_coefficients_l772_772811

theorem not_possible_to_replace_coefficients (n : ℕ) (h : n > 100) :
  ¬ ∃ (nums : fin (3 * n) → ℕ), ∀ (i : fin n),
  ∃ (a b c : ℕ), 
  nums (⟨3 * i.val⟩ : fin (3 * n)) = a ∧
  nums (⟨3 * i.val + 1⟩ : fin (3 * n)) = b ∧
  nums (⟨3 * i.val + 2⟩ : fin (3 * n)) = c ∧
  ∃ (x1 x2 : ℤ), x1 ≠ x2 ∧
  a * x1^2 + b * x1 + c = 0 ∧
  a * x2^2 + b * x2 + c = 0 :=
sorry

end not_possible_to_replace_coefficients_l772_772811


namespace triangle_ratio_values_l772_772216

theorem triangle_ratio_values (O A B C A1 B1 C1 : Point)
  (AO : Line)
  (BO : Line)
  (CO : Line)
  (AO_A1 : Segment AO A1)
  (BO_B1 : Segment BO B1)
  (CO_C1 : Segment CO C1)
  (h1 : ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    (∃ (n1 n2 n3 : ℝ),
      n1 = (b + c) / a ∧ 
      n2 = (a + c) / b ∧
      n3 = (a + b) / c ∧
      (n1 = 1 ∧ n2 = 2 ∧ n3 = 5) ∨
      (n1 = 2 ∧ n2 = 2 ∧ n3 = 2) ∨
      (n1 = 1 ∧ n2 = 3 ∧ n3 = 3) ∨
      (n1 = 2 ∧ n2 = 5 ∧ n3 = 1) ∨
      (n1 = 3 ∧ n2 = 3 ∧ n3 = 1) ∨
      (n1 = 5 ∧ n2 = 1 ∧ n3 = 2))) :
    true := 
by {
  sorry
}

end triangle_ratio_values_l772_772216


namespace A_inter_B_empty_l772_772267

open Set

def real_univ : Set ℝ := univ

def A : Set ℝ := {x | sqrt (x - 2) ≤ 0}

def B : Set ℝ := {x | 10^(x^2 - 2) = 10^x}

def A_inter_B_complement (A B : Set ℝ) : Set ℝ := A ∩ Bᶜ

theorem A_inter_B_empty : A_inter_B_complement A B = ∅ := 
by
  sorry

end A_inter_B_empty_l772_772267


namespace find_fraction_l772_772502

theorem find_fraction (n d : ℕ) (h1 : n / (d + 1) = 1 / 2) (h2 : (n + 1) / d = 1) : n / d = 2 / 3 := 
by 
  sorry

end find_fraction_l772_772502


namespace max_difference_value_l772_772682

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l772_772682


namespace max_x_minus_y_l772_772673

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772673


namespace triangle_side_length_l772_772348

theorem triangle_side_length (A B C M : Point)
  (hAB : dist A B = 2)
  (hAC : dist A C = 3)
  (hMidM : M = midpoint B C)
  (hAM_BC : dist A M = dist B C) :
  dist B C = Real.sqrt (78) / 3 :=
by
  sorry

end triangle_side_length_l772_772348


namespace triangle_side_eq_median_l772_772337

theorem triangle_side_eq_median (A B C : Type) (a b c : ℝ) (hAB : a = 2) (hAC : b = 3) (hBC_eq_median : c = (2 * (Real.sqrt (13 / 10)))) :
  c = (Real.sqrt 130) / 5 := by
  sorry

end triangle_side_eq_median_l772_772337


namespace Jill_has_5_peaches_l772_772033

-- Define the variables and their relationships
variables (S Jl Jk : ℕ)

-- Declare the conditions as assumptions
axiom Steven_has_14_peaches : S = 14
axiom Jake_has_6_fewer_peaches_than_Steven : Jk = S - 6
axiom Jake_has_3_more_peaches_than_Jill : Jk = Jl + 3

-- Define the theorem to prove Jill has 5 peaches
theorem Jill_has_5_peaches (S Jk Jl : ℕ) 
  (h1 : S = 14) 
  (h2 : Jk = S - 6)
  (h3 : Jk = Jl + 3) : 
  Jl = 5 := 
by
  sorry

end Jill_has_5_peaches_l772_772033


namespace admission_methods_count_l772_772932

theorem admission_methods_count :
  let students := 4
  let universities := 3
  let condition := (∃ (f : Fin students → Fin universities), ∀ u : Fin universities, ∃ s : Fin students, f s = u)
  condition → (count_permutations students universities = 36) := by
  intros students universities condition
  sorry

end admission_methods_count_l772_772932


namespace hypotenuse_length_l772_772768

theorem hypotenuse_length (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 1450) (h2 : c^2 = a^2 + b^2) : 
  c = Real.sqrt 725 :=
by
  sorry

end hypotenuse_length_l772_772768


namespace mens_wages_l772_772933

theorem mens_wages
  (M : ℝ) (WW : ℝ) (B : ℝ)
  (h1 : 5 * M = WW)
  (h2 : WW = 8 * B)
  (h3 : 5 * M + WW + 8 * B = 60) :
  5 * M = 30 :=
by
  sorry

end mens_wages_l772_772933


namespace option_a_correct_option_b_incorrect_option_c_correct_option_d_correct_correct_choices_l772_772729

noncomputable def f (x : ℝ) (a : ℝ) := Real.log (x + 1) - a * Real.sin x

theorem option_a_correct (a : ℝ) (h_a : a < 0) :
  ∀ x : ℝ, 0 < x ∧ x < Real.pi → deriv (λ x : ℝ, f x a) x < deriv (λ x : ℝ, f x a) (x + 1) :=
sorry

theorem option_b_incorrect :
  ¬(∀ x : ℝ, f x 0 ≤ 2 * x) :=
sorry

theorem option_c_correct :
  ∃ x : ℝ, -1 < x ∧ x < Real.pi / 2 ∧ ∀ y : ℝ, -1 < y ∧ y < Real.pi / 2 → deriv (λ x : ℝ, f x 1) x ≥ deriv (λ x : ℝ, f x 1) y :=
sorry

theorem option_d_correct :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 1 = 0 ∧ f x2 1 = 0 :=
sorry

theorem correct_choices :
  (option_a_correct a h_a) ∧ ¬option_b_incorrect ∧ option_c_correct ∧ option_d_correct :=
sorry

end option_a_correct_option_b_incorrect_option_c_correct_option_d_correct_correct_choices_l772_772729


namespace johns_number_l772_772360

theorem johns_number : 
  ∃ n, (200 ∣ n) ∧ (18 ∣ n) ∧ (1000 ≤ n) ∧ (n ≤ 2500) ∧ (n = 1800) :=
begin
  use 1800,
  split,
  { exact dvd_refl 200, },
  split,
  { exact dvd_refl 18, },
  split,
  { exact le_refl 1000, },
  split,
  { exact le_refl 2500, },
  { refl, },
end

end johns_number_l772_772360


namespace percentage_first_team_supporters_l772_772585

theorem percentage_first_team_supporters 
  (total_audience : ℕ) 
  (percentage_second_team_supporters : ℝ) 
  (non_supporters : ℕ) 
  (total_audience_eq : total_audience = 50)
  (percentage_second_team_supporters_eq : percentage_second_team_supporters = 0.34)
  (non_supporters_eq : non_supporters = 3) 
  : (100 * (total_audience - (percentage_second_team_supporters * total_audience).to_nat - non_supporters) / total_audience) = 60 := 
by 
  sorry

end percentage_first_team_supporters_l772_772585


namespace purely_imaginary_real_part_eq_zero_l772_772250

theorem purely_imaginary_real_part_eq_zero (a : ℝ) : 
  ∀ (z : ℂ), z = complex.mk (a - 1) 1 → complex.re z = 0 → a = 1 :=
by
  intro z
  intro hz
  intro hz_real
  have H : real_of_complex z = a - 1 := by sorry
  have H_zero : real_of_complex z = 0 := hz_real
  have Eq : a - 1 = 0 := by sorry
  have A_1 : a = 1 := by sorry
  exact A_1

end purely_imaginary_real_part_eq_zero_l772_772250


namespace suitable_investigation_is_electricity_consumption_l772_772493

def investigations : Type :=
  | popularity_of_product
  | viewership_ratings
  | explosive_power_test
  | electricity_consumption

def suitable_for_census (inv : investigations) : Prop :=
  match inv with
  | investigations.electricity_consumption => True
  | _ => False

theorem suitable_investigation_is_electricity_consumption :
  suitable_for_census investigations.electricity_consumption :=
by 
  -- Complete proof skipped
  sorry

end suitable_investigation_is_electricity_consumption_l772_772493


namespace sin_pi_plus_alpha_l772_772711

/-- Given that \(\sin \left(\frac{\pi}{2}+\alpha \right) = \frac{3}{5}\)
    and \(\alpha \in (0, \frac{\pi}{2})\),
    prove that \(\sin(\pi + \alpha) = -\frac{4}{5}\). -/
theorem sin_pi_plus_alpha (α : ℝ) (h1 : Real.sin (Real.pi / 2 + α) = 3 / 5)
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (Real.pi + α) = -4 / 5 := 
  sorry

end sin_pi_plus_alpha_l772_772711


namespace polygon_sides_l772_772465

theorem polygon_sides (s : ℕ) (h : 180 * (s - 2) = 720) : s = 6 :=
by
  sorry

end polygon_sides_l772_772465


namespace maximum_value_of_x_minus_y_l772_772708

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l772_772708


namespace length_width_difference_l772_772538

noncomputable def width : ℝ := Real.sqrt (588 / 8)
noncomputable def length : ℝ := 4 * width
noncomputable def difference : ℝ := length - width

theorem length_width_difference : difference = 25.722 := by
  sorry

end length_width_difference_l772_772538


namespace minimum_m_value_l772_772618

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem minimum_m_value (a b m : ℝ) (h : ∀ x ∈ set.Icc (-1:ℝ) 1, |f x a b| ≤ m) : m ≥ 1 / 2 :=
by
  have h1 : |f (-1) a b| + |f 1 a b| + 2 * |f 0 a b| ≥ 2 := sorry
  have h2 : 4 * m ≥ 2 := sorry
  have h3 : m ≥ 1 / 2 := by
    linarith [h2]
  exact h3

end minimum_m_value_l772_772618


namespace average_of_remaining_two_l772_772919

theorem average_of_remaining_two (a1 a2 a3 a4 a5 a6 : ℝ)
    (h_avg6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
    (h_avg2_1 : (a1 + a2) / 2 = 3.4)
    (h_avg2_2 : (a3 + a4) / 2 = 3.85) :
    (a5 + a6) / 2 = 4.6 := 
sorry

end average_of_remaining_two_l772_772919


namespace prob_heads_removed_correct_l772_772961

noncomputable def prob_heads_removed
  (initial_heads : 1 = 1)
  (bill_first_flip_heads : Prop)
  (bill_second_flip_heads : Prop)
  (carl_removes_coin : Prop)
  (alice_sees_two_heads : Prop)
  : ℚ := 
  (3 : ℚ) / 5

theorem prob_heads_removed_correct : 
  ∀ (initial_heads : 1 = 1) 
    (bill_first_flip_heads bill_second_flip_heads 
     carl_removes_coin alice_sees_two_heads : Prop),
  alice_sees_two_heads →
  prob_heads_removed initial_heads bill_first_flip_heads bill_second_flip_heads carl_removes_coin alice_sees_two_heads = 3 / 5 :=
by
  intros
  sorry

end prob_heads_removed_correct_l772_772961


namespace smallest_period_value_of_a_for_period_2019_l772_772215

def sequence_fn (a : ℕ) : ℕ → ℕ 
| n :=
    if n ≤ 2 then 1 + 1009 * n
    else if n ≤ 1010 then 2021 - n 
    else if n ≤ 1011 then 3031 - 2 * n
    else 2020 - n

def is_periodic (a : ℕ) (k : ℕ) : Prop :=
  ∀ n, sequence_fn a (n + k) = sequence_fn a n

theorem smallest_period {a : ℕ} (h : 1 ≤ a ∧ a ≤ 2019) : ∃ k, is_periodic a k ∧ (∀ m, m < k → ¬ is_periodic a m) ∧ k = 2019 :=
sorry

theorem value_of_a_for_period_2019 {a : ℕ} (h : 1 ≤ a ∧ a ≤ 2019) :
  (∃ k, is_periodic a k ∧ 2 < k ∧ k % 2 = 1 ∧ k = 2019) ↔ (a ∈ [1, 2019]) :=
sorry

end smallest_period_value_of_a_for_period_2019_l772_772215


namespace pencil_length_after_sharpening_l772_772354

theorem pencil_length_after_sharpening (original_length sharpened_length: ℕ) : original_length = 31 → sharpened_length = 17 → original_length - sharpened_length = 14 :=
by 
  intros h1 h2
  rw [h1, h2]
  rfl

end pencil_length_after_sharpening_l772_772354


namespace cos_minus_sin_l772_772224

theorem cos_minus_sin (α : ℝ) (hα1 : (π / 4) < α) (hα2 : α < (π / 2)) (h : sin (2 * α) = 24 / 25) : cos α - sin α = -1 / 5 :=
by
  sorry

end cos_minus_sin_l772_772224


namespace supply_lasts_for_8_months_l772_772788

-- Define the conditions
def pills_per_supply : ℕ := 120
def days_per_pill : ℕ := 2
def days_per_month : ℕ := 30

-- Define the function to calculate the duration in days
def supply_duration_in_days (pills : ℕ) (days_per_pill : ℕ) : ℕ :=
  pills * days_per_pill

-- Define the function to convert days to months
def days_to_months (days : ℕ) (days_per_month : ℕ) : ℕ :=
  days / days_per_month

-- Main statement to prove
theorem supply_lasts_for_8_months :
  days_to_months (supply_duration_in_days pills_per_supply days_per_pill) days_per_month = 8 :=
by
  sorry

end supply_lasts_for_8_months_l772_772788


namespace soccer_ball_cost_l772_772515

theorem soccer_ball_cost (F S : ℝ) 
  (h1 : 3 * F + S = 155) 
  (h2 : 2 * F + 3 * S = 220) :
  S = 50 := 
sorry

end soccer_ball_cost_l772_772515


namespace total_boxes_is_4575_l772_772070

-- Define the number of boxes in each warehouse
def num_boxes_in_warehouse_A (x : ℕ) := x
def num_boxes_in_warehouse_B (x : ℕ) := 3 * x
def num_boxes_in_warehouse_C (x : ℕ) := (3 * x) / 2 + 100
def num_boxes_in_warehouse_D (x : ℕ) := 2 * ((3 * x) / 2 + 100) - 50
def num_boxes_in_warehouse_E (x : ℕ) := x + (2 * ((3 * x) / 2 + 100) - 50) - 200

-- Define the condition that warehouse B has 300 more boxes than warehouse E
def condition_B_E (x : ℕ) := 3 * x = num_boxes_in_warehouse_E x + 300

-- Define the total number of boxes calculation
def total_boxes (x : ℕ) := 
    num_boxes_in_warehouse_A x +
    num_boxes_in_warehouse_B x +
    num_boxes_in_warehouse_C x +
    num_boxes_in_warehouse_D x +
    num_boxes_in_warehouse_E x

-- The statement of the problem
theorem total_boxes_is_4575 (x : ℕ) (h : condition_B_E x) : total_boxes x = 4575 :=
by
    sorry

end total_boxes_is_4575_l772_772070


namespace negation_of_existence_proposition_l772_772448

theorem negation_of_existence_proposition :
  ¬ (∃ x : ℝ, x^2 + 2*x - 8 = 0) ↔ ∀ x : ℝ, x^2 + 2*x - 8 ≠ 0 := by
  sorry

end negation_of_existence_proposition_l772_772448


namespace num_terms_in_polynomial_l772_772181

theorem num_terms_in_polynomial (x y z : ℝ):
  (let num_terms := (finset.range (2011)).sum (λ a, if (even a) then (2011 - a) else 0)
   in num_terms = 1006 * 1006) := 
begin
  sorry
end

end num_terms_in_polynomial_l772_772181


namespace smallest_x_for_multiple_l772_772899

theorem smallest_x_for_multiple (x : ℕ) (h720 : 720 = 2^4 * 3^2 * 5) (h1250 : 1250 = 2 * 5^4) : 
  (∃ x, (x > 0) ∧ (1250 ∣ (720 * x))) → x = 125 :=
by
  sorry

end smallest_x_for_multiple_l772_772899


namespace garden_strawberry_yield_l772_772573

-- Definitions from the conditions
def garden_length : ℝ := 10
def garden_width : ℝ := 15
def plants_per_sq_ft : ℝ := 5
def strawberries_per_plant : ℝ := 12

-- Expected total number of strawberries
def expected_strawberries : ℝ := 9000

-- Proof statement
theorem garden_strawberry_yield : 
  (garden_length * garden_width * plants_per_sq_ft * strawberries_per_plant) = expected_strawberries :=
by sorry

end garden_strawberry_yield_l772_772573


namespace professionals_work_days_l772_772361

theorem professionals_work_days (cost_per_hour_1 cost_per_hour_2 hours_per_day total_cost : ℝ) (h_cost1: cost_per_hour_1 = 15) (h_cost2: cost_per_hour_2 = 15) (h_hours: hours_per_day = 6) (h_total: total_cost = 1260) : (∃ d : ℝ, total_cost = d * hours_per_day * (cost_per_hour_1 + cost_per_hour_2) ∧ d = 7) :=
by
  use 7
  rw [h_cost1, h_cost2, h_hours, h_total]
  simp
  sorry

end professionals_work_days_l772_772361


namespace ratio_of_areas_eq_nine_sixteenth_l772_772460

-- Definitions based on conditions
def side_length_C : ℝ := 45
def side_length_D : ℝ := 60
def area (s : ℝ) : ℝ := s * s

-- Theorem stating the desired proof problem
theorem ratio_of_areas_eq_nine_sixteenth :
  (area side_length_C) / (area side_length_D) = 9 / 16 :=
by
  sorry

end ratio_of_areas_eq_nine_sixteenth_l772_772460


namespace cistern_fill_time_l772_772917

theorem cistern_fill_time (Rf Re : ℝ) (Rf_fills : Rf = 1 / 4) (Re_empties : Re = 1 / 6) : 
  let Rnet := Rf - Re in
  let T := 1 / Rnet in
  T = 12 :=
by
  have Rf := Rf_fills
  have Re := Re_empties
  sorry

end cistern_fill_time_l772_772917


namespace find_maaza_l772_772121

/-- Let M be the number of liters of Maaza. Given that -/
def liters_of_Maaza (M x : ℕ) : Prop :=
  let totalCans := (M + 144 + 368) / x
  gcd 144 368 = x ∧ totalCans = 261

/-- Prove that M = 3664 under the given conditions -/
theorem find_maaza (M x : ℕ) (h : liters_of_Maaza M x) : M = 3664 :=
by
  have 144 ≤ M + 144 + 368 := sorry
  have 368 ≤ M + 144 + 368 := sorry
  have 261 * x = M + 144 + 368 := sorry
  have x = 16 := sorry
  have M + 512 = 261 * 16 := sorry
  have M = 4176 - 512 := sorry
  exact eq.refl 3664

end find_maaza_l772_772121


namespace expenditure_july_l772_772145

def avg_expenditure_jan_to_jun : ℝ := 4200
def expenditure_january : ℝ := 1200
def avg_expenditure_feb_to_jul : ℝ := 4250

theorem expenditure_july 
  (avg_expenditure_jan_to_jun : ℝ) 
  (expenditure_january : ℝ) 
  (avg_expenditure_feb_to_jul : ℝ) :
  let expenditure_feb_to_jun := 6 * avg_expenditure_jan_to_jun - expenditure_january,
      expenditure_feb_to_jul := 6 * avg_expenditure_feb_to_jul in
  expenditure_feb_to_jul - expenditure_feb_to_jun = 1500 :=
by
  sorry

end expenditure_july_l772_772145


namespace primes_not_sum_of_two_composites_l772_772398

def is_prime (n : ℕ) : Prop :=
nat.prime n

def is_composite (n : ℕ) : Prop :=
n ≠ 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def cannot_be_sum_of_two_composites (p : ℕ) : Prop :=
¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b

theorem primes_not_sum_of_two_composites :
  [2, 3, 5, 7, 11] = (list.filter (λ p, is_prime p ∧ cannot_be_sum_of_two_composites p) [2,3,5,7,11]) :=
by sorry

end primes_not_sum_of_two_composites_l772_772398


namespace tan_11_25_eq_sqrt_a_sub_sqrt_b_add_sqrt_c_sub_d_l772_772057

theorem tan_11_25_eq_sqrt_a_sub_sqrt_b_add_sqrt_c_sub_d :
  ∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ tan 11.25° = real.sqrt a - real.sqrt b + real.sqrt c - d ∧ a + b + c + d = 4 :=
sorry

end tan_11_25_eq_sqrt_a_sub_sqrt_b_add_sqrt_c_sub_d_l772_772057


namespace shaded_region_area_l772_772542

noncomputable def area_of_shaded_region (beta : ℝ) (cos_beta_rational : ∀ beta, (cos beta = 3/5)) (h1 : 0 < beta) (h2 : beta < π/2) : ℝ :=
  if h3 : cos beta = 3 / 5 then 12 / 7 else 0

theorem shaded_region_area (beta : ℝ) (cos_beta_rational : cos beta = 3 / 5) (h1 : 0 < beta) (h2 : beta < π/2) :
    area_of_shaded_region beta cos_beta_rational h1 h2 = 12 / 7 := 
  by
  sorry

end shaded_region_area_l772_772542


namespace max_x_minus_y_l772_772674

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772674


namespace ellipse_eq_and_line_eq_l772_772638

theorem ellipse_eq_and_line_eq
  (e : ℝ) (a b c xC yC: ℝ)
  (h_e : e = (Real.sqrt 3 / 2))
  (h_a : a = 2)
  (h_c : c = Real.sqrt 3)
  (h_b : b = Real.sqrt (a^2 - c^2))
  (h_ellipse : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 = 1))
  (h_C_on_G : xC^2 / 4 + yC^2 = 1)
  (h_diameter_condition : ∀ (B : ℝ × ℝ), B = (0, 1) →
    ((2 * xC - yC + 1 = 0) →
    (xC = 0 ∧ yC = 1) ∨ (xC = -16 / 17 ∧ yC = -15 / 17)))
  : (∀ x y, (y = 2*x + 1) ↔ (x + 2*y - 2 = 0 ∨ 3*x - 10*y - 6 = 0)) :=
by
  sorry

end ellipse_eq_and_line_eq_l772_772638


namespace tank_length_l772_772955

-- Definitions and assumptions from the problem
def width : ℝ := 12
def depth : ℝ := 6
def cost_per_sq_meter_paise : ℝ := 55
def total_cost_rupees : ℝ := 409.20
def total_cost_paise : ℝ := total_cost_rupees * 100
def total_area_plastered_sq_m : ℝ := total_cost_paise / cost_per_sq_meter_paise

-- Main theorem to prove
theorem tank_length
  (L : ℝ)
  (h : total_area_plastered_sq_m = 2 * (L * depth) + 2 * (width * depth) + (L * width))
  : L = 25 :=
sorry

end tank_length_l772_772955


namespace symmetric_circle_eq_l772_772726

theorem symmetric_circle_eq :
  let C1 := fun x y => (x - 3) ^ 2 + (y + 1) ^ 2 = 1
  let ref_line := fun x y => 2 * x - y - 2 = 0
  ∃ C2 : ℝ → ℝ → Prop, (∀ x y, C2 x y ↔ (x + 1) ^ 2 + (y - 1) ^ 2 = 1) ∧
                       (symmetric_about_line C1 ref_line C2) :=
by
  -- Definitions and conditions
  let C1 := fun x y => (x - 3) ^ 2 + (y + 1) ^ 2 = 1
  let ref_line := fun x y => 2 * x - y - 2 = 0
  -- Existence and equation of C2
  use (fun x y => (x + 1) ^ 2 + (y - 1) ^ 2 = 1)
  -- Proof of equivalence and symmetry
  split
  -- C2 equation is (x + 1) ^ 2 + (y - 1) ^ 2 = 1
  intro x y
  exact Iff.rfl
  -- Symmetry about the line
  sorry

end symmetric_circle_eq_l772_772726


namespace turnBackDifference_game_sum_l772_772775

-- Define the initial sequence and the operation function
def turnBackDifferenceGame (m n : ℚ) : List ℚ :=
list.iterate 2023 step [-m, -n]

-- Define the step function as described in the problem
def step : List ℚ -> List ℚ
| [] => []
| [x] => [x]
| (x :: y :: zs) => (x :: y :: (y - x) :: zs)

-- Prove that the sum of the polynomials obtained after the 2023rd operation is -2n
theorem turnBackDifference_game_sum (m n : ℚ) : list.sum (turnBackDifferenceGame m n) = -2 * n := by
  sorry

end turnBackDifference_game_sum_l772_772775


namespace max_x_minus_y_l772_772648

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l772_772648


namespace ancient_chinese_wine_problem_l772_772780

theorem ancient_chinese_wine_problem:
  ∃ x: ℝ, 10 * x + 3 * (5 - x) = 30 :=
by
  sorry

end ancient_chinese_wine_problem_l772_772780


namespace sphere_radius_l772_772540

noncomputable def cube_edge : ℝ := 8 -- Edge length of the cube is 8 inches
noncomputable def cube_volume : ℝ := cube_edge ^ 3 -- Volume of the cube
noncomputable def sphere_volume : ℝ := cube_volume / 2 -- Volume of the sphere is half the volume of the cube

theorem sphere_radius (r : ℝ) : 
  (4 / 3) * Real.pi * r ^ 3 = sphere_volume → 
  r = Real.cbrt (192 / Real.pi) := 
by 
  sorry

end sphere_radius_l772_772540


namespace max_min_quadratic_l772_772264

theorem max_min_quadratic (a b c : ℝ) (h1 : f(x) = a * x^2 + b * x + c) 
  (h2 : a + b + c = 1) (h3 : ∃ x₁ x₂, f(x₁) = 0 ∧ f(x₂) = 0) :
  max (min a (min b c)) = 1 / 4 :=
by 
  sorry

end max_min_quadratic_l772_772264


namespace no_positive_integer_solution_l772_772844

theorem no_positive_integer_solution (a b c d : ℕ) (h1 : a^2 + b^2 = c^2 - d^2) (h2 : a * b = c * d) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : false := 
by 
  sorry

end no_positive_integer_solution_l772_772844


namespace exists_irrational_between_zero_and_three_l772_772908

theorem exists_irrational_between_zero_and_three : ∃ x : ℝ, (0 < x) ∧ (x < 3) ∧ irrational x :=
sorry

end exists_irrational_between_zero_and_three_l772_772908


namespace find_a_2011_l772_772632

-- Definitions based on the conditions
def sequence (a : ℕ → ℤ) := (∀ n : ℕ, a n + a (n + 1) = 0)

noncomputable def a : ℕ → ℤ := sorry

-- Conditions translating from the problem
axiom a_201 : a 201 = 2
axiom sequence_property : sequence a

-- Final proof statement
theorem find_a_2011 : a 2011 = 2 :=
by 
  sorry

end find_a_2011_l772_772632


namespace triangles_similar_l772_772888

-- Given Conditions as Definitions
variable {α : Type*} [EuclideanGeometry α]

structure TriangleInscribedInCircle (ω : Circle α) (A B C : Point α) : Prop := 
(inscribed : ω.inscribed_triangle A B C)

structure TangentLinesMeet (ω : Circle α) (B C T : Point α) : Prop :=
(tangent_line_B : ω.tangent B)
(tangent_line_C : ω.tangent C)
(meet_at_T : tangent_line_B ∩ tangent_line_C = T)

structure PerpendicularOnRay (A S T : Point α) : Prop :=
(perpendicular : ∠AS T = 90)

structure PointsOnRay (S T B1 C1 : Point α) : Prop :=
(B1T_eq_BT : distance B1 T = distance B T)
(C1T_eq_CT : distance C1 T = distance C T)

-- Define the main theorem
theorem triangles_similar
  {ω : Circle α} {A B C T S B1 C1 : Point α} 
  (h₁ : TriangleInscribedInCircle ω A B C) 
  (h₂ : TangentLinesMeet ω B C T)
  (h₃ : PerpendicularOnRay A S T)
  (h₄ : PointsOnRay S T B1 C1) :
  similar_triangle A B C A B1 C1 := 
sorry

end triangles_similar_l772_772888


namespace solve_for_x_l772_772491

theorem solve_for_x (x : ℝ) (h : (sqrt x) ^ 3 = 100) : x = 10^(4 / 3) :=
sorry

end solve_for_x_l772_772491


namespace radius_ratio_l772_772124

theorem radius_ratio (V₁ V₂ : ℝ) (hV₁ : V₁ = 432 * Real.pi) (hV₂ : V₂ = 108 * Real.pi) : 
  (∃ (r₁ r₂ : ℝ), V₁ = (4/3) * Real.pi * r₁^3 ∧ V₂ = (4/3) * Real.pi * r₂^3) →
  ∃ k : ℝ, k = r₂ / r₁ ∧ k = 1 / 2^(2/3) := 
by
  sorry

end radius_ratio_l772_772124


namespace no_right_triangle_l772_772494

theorem no_right_triangle (a b c : ℝ) (h₁ : a = Real.sqrt 3) (h₂ : b = 2) (h₃ : c = Real.sqrt 5) : 
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end no_right_triangle_l772_772494


namespace fraction_simplification_proof_l772_772423

noncomputable def simplify_fraction : Prop :=
  let num := Real.sqrt (3 + Real.sqrt 5)
  let den := Real.cbrt ((4 * Real.sqrt 2 - 2 * Real.sqrt 10) ^ 2)
  (num / den) = (Real.sqrt 2 / 2) * (Real.sqrt 5 - 2)

theorem fraction_simplification_proof : simplify_fraction :=
  by sorry

end fraction_simplification_proof_l772_772423


namespace mary_weight_gain_back_l772_772402

theorem mary_weight_gain_back : 
  ∀ (W0 Wf ΔW_initial ΔW_final: ℕ), 
  W0 = 99 ∧ 
  ΔW_initial = 12 ∧ 
  Wf = 81 ∧ 
  Wf = W0 - ΔW_initial + 2 * ΔW_initial - 3 * ΔW_initial + ΔW_final → 
  ΔW_final = 6 :=
by
  intros ?
  sorry

end mary_weight_gain_back_l772_772402


namespace f_increasing_on_interval_l772_772442

def f (x : ℝ) : ℝ := log (1/2 : ℝ) (x^2 - 2*x - 3)

theorem f_increasing_on_interval :
  ∀ x y : ℝ, x < -1 ∧ y < -1 ∧ x < y → f x < f y :=
by
  sorry

end f_increasing_on_interval_l772_772442


namespace longest_side_of_garden_l772_772400

theorem longest_side_of_garden (l w : ℝ) (h1 : 2 * l + 2 * w = 225) (h2 : l * w = 8 * 225) :
  l = 93.175 ∨ w = 93.175 :=
by
  sorry

end longest_side_of_garden_l772_772400


namespace has_exactly_one_zero_iff_l772_772044

def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 0 then log 2 x else -2^x + a

theorem has_exactly_one_zero_iff (a : ℝ) :
  (∃! x : ℝ, f x a = 0) ↔ a < 0 :=
sorry

end has_exactly_one_zero_iff_l772_772044


namespace ashton_remaining_items_l772_772561

variables (pencil_boxes : ℕ) (pens_boxes : ℕ) (pencils_per_box : ℕ) (pens_per_box : ℕ)
          (given_pencils_brother : ℕ) (distributed_pencils_friends : ℕ)
          (distributed_pens_friends : ℕ)

def total_initial_pencils := 3 * 14
def total_initial_pens := 2 * 10

def remaining_pencils := total_initial_pencils - 6 - 12
def remaining_pens := total_initial_pens - 8
def remaining_items := remaining_pencils + remaining_pens

theorem ashton_remaining_items : remaining_items = 36 :=
sorry

end ashton_remaining_items_l772_772561


namespace cannot_achieve_90_cents_l772_772499

theorem cannot_achieve_90_cents :
  ∀ (p n d q : ℕ),        -- p: number of pennies, n: number of nickels, d: number of dimes, q: number of quarters
  (p + n + d + q = 6) →   -- exactly six coins chosen
  (p ≤ 4 ∧ n ≤ 4 ∧ d ≤ 4 ∧ q ≤ 4) →  -- no more than four of each kind of coin
  (p + 5 * n + 10 * d + 25 * q ≠ 90) -- total value should not equal 90 cents
:= by
  sorry

end cannot_achieve_90_cents_l772_772499


namespace tet_lateral_surface_area_max_l772_772721

noncomputable def sphere_radius := 3
noncomputable def max_lateral_surface_area := 18

theorem tet_lateral_surface_area_max (PA PB PC : ℝ) 
  (mutually_perpendicular : PA ≠ 0 ∧ PB ≠ 0 ∧ PC ≠ 0 ∧ PA ⟂ PB ∧ PB ⟂ PC ∧ PA ⟂ PC)
  (on_sphere : PA + PB + PC = sphere_radius * sqrt 3 ) :
  max_lateral_surface_area ≤ 18 :=
by
  sorry

end tet_lateral_surface_area_max_l772_772721


namespace arith_seq_frac_l772_772792

noncomputable def arith_seq_conditions (a_n b_n : ℕ → ℕ) (T S : ℕ → ℕ) :=
  (∀ n, T n = (n * (2 * a (n - 1))) + a n) ∧ 
  (∀ n, S n = (n * (2 * b (n - 1))) + b n) ∧
  (∀ n, S n / T n = n / (2 * n - 1))

theorem arith_seq_frac (a_n b_n : ℕ → ℕ) (T S : ℕ → ℕ)
  (h : arith_seq_conditions a_n b_n T S) :
  a_n 6 / b_n 6 = 11 / 21 := sorry

end arith_seq_frac_l772_772792


namespace expr_divisible_by_120_l772_772825

theorem expr_divisible_by_120 (m : ℕ) : 120 ∣ (m^5 - 5 * m^3 + 4 * m) :=
sorry

end expr_divisible_by_120_l772_772825


namespace savannah_wrapped_first_roll_l772_772019

theorem savannah_wrapped_first_roll :
  ∀ (total_gifts rolls roll1_gifts roll2_gifts roll3_gifts : ℕ),
    total_gifts = 12 →
    rolls = 3 →
    roll1_gifts + roll2_gifts + roll3_gifts = total_gifts →
    roll2_gifts = 5 →
    roll3_gifts = 4 →
    roll1_gifts = 3 :=
by
  intros total_gifts rolls roll1_gifts roll2_gifts roll3_gifts
  assume h1 h2 h3 h4 h5
  sorry

end savannah_wrapped_first_roll_l772_772019


namespace max_x_minus_y_l772_772672

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772672


namespace vertex_of_parabola_l772_772445

theorem vertex_of_parabola (a b c : ℝ)
  (h1 : 4 * a - 2 * b + c = 8)
  (h2 : 16 * a + 4 * b + c = 8)
  (h3 : 49 * a + 7 * b + c = 15) :
  let x_vertex := (λ a b c : ℝ, -b / (2 * a)) in
  x_vertex a b c = 1 :=
by
  sorry

end vertex_of_parabola_l772_772445


namespace percent_literate_females_l772_772005

theorem percent_literate_females 
  (inhabitants : ℕ)
  (male_percent : ℕ)
  (male_literate_percent : ℕ)
  (total_literate_percent : ℕ)
  (H1 : inhabitants = 1000)
  (H2 : male_percent = 60)
  (H3 : male_literate_percent = 20)
  (H4 : total_literate_percent = 25) :
  let males := male_percent * inhabitants / 100,
      females := inhabitants - males,
      literate_males := male_literate_percent * males / 100,
      total_literate := total_literate_percent * inhabitants / 100,
      literate_females := total_literate - literate_males,
      percent_literate_females := literate_females * 100 / females
  in percent_literate_females = 32.5 :=
by
  sorry

end percent_literate_females_l772_772005


namespace number_of_sheep_equals_27_l772_772149

variables (A Y S H : ℕ)

-- Conditions
def ratio_sh : Prop := ∃ S H, S / H = 3 / 7
def twice_adult_young : Prop := A = 2 * Y
def adult_food : Prop := ∀ (A : ℕ), A * 230 ≤ 12880
def young_food : Prop := ∀ (Y : ℕ), Y * 150 ≤ 12880
def total_food : Prop := (A * 230 + Y * 150) = 12880
def total_horses : Prop := H = A + Y
def ratio_sheep_to_horses : Prop := S / H = 3 / 7

-- Problem statement
theorem number_of_sheep_equals_27 :
  (ratio_sh) ∧ (twice_adult_young) ∧ (total_food) ∧ (total_horses) ∧ (ratio_sheep_to_horses) → S = 27 := 
by
  sorry

end number_of_sheep_equals_27_l772_772149


namespace problem_1_problem_2_l772_772395

def findSmaller (a b : ℝ) : ℝ := if a < b then a else b
def findLarger (a b : ℝ) : ℝ := if a > b then a else b

theorem problem_1 : findSmaller (-5) (-0.5) + findLarger (-4) 2 = -3 :=
by
  sorry

theorem problem_2 : findSmaller 1 (-3) + findLarger (-5) (findSmaller (-2) (-7)) = -8 :=
by
  sorry

end problem_1_problem_2_l772_772395


namespace average_of_divisibles_by_4_l772_772501

theorem average_of_divisibles_by_4 (a b : ℕ) (h₁ : 6 ≤ a) (h₂ : b ≤ 38) (h₃ : a % 4 = 0) (h₄ : b % 4 = 0) :
  let L := [n | n ← List.range (b-a+1), n % 4 = 0, a ≤ n ∧ n ≤ b]
  (L.sum / L.length) = 22 :=
by
  have L := [8, 12, 16, 20, 24, 28, 32, 36]
  have len_L : L length = 8 := by rfl
  have sum_L : L.sum = 176 := by rfl
  have average := L.sum / L.length
  show average = 22 from sorry

end average_of_divisibles_by_4_l772_772501


namespace gcd_determinant_l772_772277

theorem gcd_determinant (a b : ℤ) (h : Int.gcd a b = 1) :
  Int.gcd (a + b) (a^2 + b^2 - a * b) = 1 ∨ Int.gcd (a + b) (a^2 + b^2 - a * b) = 3 :=
sorry

end gcd_determinant_l772_772277


namespace distance_to_origin_l772_772088

noncomputable def calculate_distance : ℝ :=
  let z := ((1 - complex.i) * (1 + complex.i)) / complex.i in
  complex.abs z

theorem distance_to_origin :
  calculate_distance = 2 :=
by
  sorry

end distance_to_origin_l772_772088


namespace root_in_neg_one_to_zero_l772_772431

noncomputable def f (x : ℝ) : ℝ := x^3 + x + 1

theorem root_in_neg_one_to_zero : ∃ x ∈ Ioo (-1:ℝ) 0, f x = 0 :=
by {
  -- Proof omitted.
  sorry
}

end root_in_neg_one_to_zero_l772_772431


namespace point_with_at_most_three_closest_l772_772627

theorem point_with_at_most_three_closest {α : Type*} [metric_space α] 
  (S : finset α) (hS : S.finite) :
  ∃ P ∈ S, ∀ P' ∈ S \ {P}, 3 < ∑ Q ∈ S, (dist P Q = dist P' Q) → false :=
sorry

end point_with_at_most_three_closest_l772_772627


namespace sqrt_sin_cos_expression_l772_772915

theorem sqrt_sin_cos_expression (α β : ℝ) : 
  Real.sqrt ((1 - Real.sin α * Real.sin β)^2 - (Real.cos α * Real.cos β)^2) = |Real.sin α - Real.sin β| :=
sorry

end sqrt_sin_cos_expression_l772_772915


namespace distance_from_origin_to_midpoint_l772_772485

theorem distance_from_origin_to_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 10) → (y1 = 20) → (x2 = -10) → (y2 = -20) → 
  dist (0 : ℝ × ℝ) ((x1 + x2) / 2, (y1 + y2) / 2) = 0 := 
by
  intros x1 y1 x2 y2 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- remaining proof goes here
  sorry

end distance_from_origin_to_midpoint_l772_772485


namespace monotonic_intervals_distinct_solutions_l772_772258

-- Define the function f(x)
def f (a x : ℝ) : ℝ := (1 / 3) * x^3 - a * x + 1

-- Define the derivative f'(x)
def f_prime (a x : ℝ) : ℝ := x^2 - a

-- Define the conditions for monotonic intervals
theorem monotonic_intervals (a : ℝ) :
  (∀ x : ℝ, (a > 0 ∧ ((x < -real.sqrt a ∨ x > real.sqrt a) → f_prime a x > 0) ∧ (-real.sqrt a < x ∧ x < real.sqrt a → f_prime a x < 0))) ∨
  (a ≤ 0 ∧ ∀ x : ℝ, f_prime a x ≥ 0) :=
sorry

-- Define the condition for three distinct solutions of f(x) = 0
theorem distinct_solutions (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) ↔
  a > real.sqrt (3 / 2) ^ (2 / 3) :=
sorry

end monotonic_intervals_distinct_solutions_l772_772258


namespace temperature_conversion_l772_772084

theorem temperature_conversion (F : ℝ) (C : ℝ) : 
  F = 95 → 
  C = (F - 32) * 5 / 9 → 
  C = 35 := by
  intro hF hC
  sorry

end temperature_conversion_l772_772084


namespace appropriate_sampling_methods_l772_772076

-- Conditions for the first survey
structure Population1 where
  high_income_families : Nat
  middle_income_families : Nat
  low_income_families : Nat
  total : Nat := high_income_families + middle_income_families + low_income_families

def survey1_population : Population1 :=
  { high_income_families := 125,
    middle_income_families := 200,
    low_income_families := 95
  }

-- Condition for the second survey
structure Population2 where
  art_specialized_students : Nat

def survey2_population : Population2 :=
  { art_specialized_students := 5 }

-- The main statement to prove
theorem appropriate_sampling_methods :
  (survey1_population.total >= 100 → stratified_sampling_for_survey1) ∧ 
  (survey2_population.art_specialized_students >= 3 → simple_random_sampling_for_survey2) :=
  sorry

end appropriate_sampling_methods_l772_772076


namespace ordering_of_a_b_c_l772_772616

noncomputable def a := 1 / Real.exp 1
noncomputable def b := Real.log 3 / 3
noncomputable def c := Real.log 4 / 4

-- We need to prove that the ordering is a > b > c.

theorem ordering_of_a_b_c : a > b ∧ b > c :=
by 
  sorry

end ordering_of_a_b_c_l772_772616


namespace males_who_dont_listen_equal_105_l772_772952

theorem males_who_dont_listen_equal_105
  (total_listeners : ℕ)
  (females_listen : ℕ)
  (males_dont_listen : ℕ)
  (total_dont_listen : ℕ)
  (H1 : total_listeners = 160)
  (H2 : females_listen = 75)
  (H3 : males_dont_listen = 105)
  (H4 : total_dont_listen = 200) :
  males_dont_listen = 105 :=
by
  rw [H3]
  -- sorry marker is for the uncompleted proof
  sorry

end males_who_dont_listen_equal_105_l772_772952


namespace log_conversion_k_l772_772375

theorem log_conversion_k (y k : ℝ) (h₁ : log 8 5 = y) (h₂ : log 2 81 = k * y) : 
  k = 12 * (log 2 3) / (log 2 5) :=
by
  sorry

end log_conversion_k_l772_772375


namespace baron_munchausen_theorem_l772_772975

theorem baron_munchausen_theorem :
  ∀ (triangle : Type) (center : triangle → Prop),
  (∀ A B C : triangle, EquilateralTriangle A B C) → 
  (∀ (start : triangle), 
    reflects_off_edges start 
    ∧ passes_through_point start center 3 
    ∧ three_directions start center 
    → returns_to_start start
  ) :=
begin
  sorry
end

end baron_munchausen_theorem_l772_772975


namespace total_resistance_l772_772772

theorem total_resistance (x y z : ℝ) (R_parallel r : ℝ)
    (hx : x = 3)
    (hy : y = 6)
    (hz : z = 4)
    (hR_parallel : 1 / R_parallel = 1 / x + 1 / y)
    (hr : r = R_parallel + z) :
    r = 6 := by
  sorry

end total_resistance_l772_772772


namespace female_managers_count_l772_772295

def E : ℕ -- total number of employees E
def M : ℕ := E - 500 -- number of male employees (M = E - 500)
def total_managers : ℕ := (2/5) * E -- total number of managers ((2/5)E)
def male_managers : ℕ := (2/5) * M -- number of male managers ((2/5)M)
def female_managers : ℕ := total_managers - male_managers -- number of female managers (total_managers - male_managers)
def company_total_managers: E : ℕ → total_managers : ℕ→ female_ubalnce_constraints: female_managers
theorem female_managers_count : female_managers = 200 := sorry

end female_managers_count_l772_772295


namespace maximum_value_of_x_minus_y_l772_772706

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l772_772706


namespace hyperbola_eq_no_line_exists_l772_772260

-- Defining the hyperbola M and parabola in Lean
structure Hyperbola (a b : ℝ) :=
(eq_Hyperbola : ∀ x y, y^2 / a^2 - x^2 / b^2 = 1)

structure Parabola :=
(eq_Parabola : ∀ x y, x^2 = 16 * y)

-- Conditions
variables {a b : ℝ} (ha : a > 0) (hb : b > 0)
variables (focus_same : ∀ M P, M = Parabola.focus P ∧ P = Hyperbola.focus M)

-- Given Conditions
def length_imaginary_axis (M : Hyperbola a b) : ℝ := 2 * b

lemma imaginary_axis_length_eq_4 (M : Hyperbola a b) : length_imaginary_axis M = 4 :=
sorry

-- Proof Problem 1: Equation of M
theorem hyperbola_eq (M : Hyperbola a b) (h1 : length_imaginary_axis M = 4) (h2: sqrt(a^2 + b^2) = 4) :
  M = Hyperbola 12 4 :=
sorry

-- Proof Problem 2: Existence and Slope of Line l
theorem no_line_exists (M : Hyperbola 12 4) (P : ℝ × ℝ) (hP : P = (1,2)) :
  ∀ l, ¬(∃ A B : ℝ × ℝ, Line.l_intersect_Hyperbola l M A B ∧ midpoint A B = P) :=
sorry

end hyperbola_eq_no_line_exists_l772_772260


namespace particle_probability_l772_772944

theorem particle_probability (n k : ℕ) (h : k > 0) :
  let total_sequences := 2^(2 * n + k),
      valid_sequences := 2 * Nat.factorial (2 * n + k) / (Nat.factorial n * Nat.factorial (n + k)) in
  valid_sequences / total_sequences = 
    (2 * Nat.factorial (2 * n + k)) / (Nat.factorial n * Nat.factorial (n + k) * 2^(2 * n + k)) :=
by
  sorry

end particle_probability_l772_772944


namespace least_multiple_of_17_gt_450_l772_772487

def least_multiple_gt (n x : ℕ) (k : ℕ) : Prop :=
  k * n > x ∧ ∀ m : ℕ, m * n > x → m ≥ k

theorem least_multiple_of_17_gt_450 : ∃ k : ℕ, least_multiple_gt 17 450 k :=
by
  use 27
  sorry

end least_multiple_of_17_gt_450_l772_772487


namespace math_proof_problem_l772_772225

noncomputable def proof_problem (x m : ℝ) : Prop :=
  (∀ x m, (x ≠ 0) → m = 2 / x) ∧
  (∀ x m, (2*x - m - 4 = 0) → x = 1 ∧ m = 2) ∧
  (x ∈ Icc (-1:ℝ) 1) ∧ 
  (∀ (x : ℝ), (x^2 - 2 * x + 1 = 2 * x + m) → m < x^2 - 4 * x + 1 ∧ x ∈ Icc (-1) 1)

theorem math_proof_problem (x m : ℝ) (h₁: ∀ x m, (x ≠ 0) → m = 2 / x)
    (h₂: ∀ x m, (2*x - m - 4 = 0) → x = 1 ∧ m = 2)
    (h₃: x ∈ Icc (-1:ℝ) 1)
    (h₄: ∀ (x : ℝ), (x^2 - 2 * x + 1 = 2 * x + m) → m < x^2 - 4 * x + 1 ∧ x ∈ Icc (-1) 1) : proof_problem x m :=
  sorry

end math_proof_problem_l772_772225


namespace minimum_knights_required_l772_772031

def combat_skill_Bedevir : ℝ := 1

def combat_skill_opponent (n : ℕ) : ℝ := 1 / (2^(n + 1) - 1)

def probability_Bedevir_wins (n : ℕ) : ℝ :=
  combat_skill_Bedevir / (combat_skill_Bedevir + combat_skill_opponent n)

theorem minimum_knights_required (n : ℕ) (hn : ∀ k < n, probability_Bedevir_wins k ≥ 1 / 2) :
  n = 1 :=
sorry

end minimum_knights_required_l772_772031


namespace only_one_relatively_prime_l772_772020

-- Define the sequence as described in the problem
def sequence (n : ℕ) : ℕ := 2^n + 3^n + 6^n - 1

-- State the theorem that 1 is the only positive integer coprime with all terms of sequence
theorem only_one_relatively_prime {n : ℕ} (d : ℕ) (h : ∀ n, Nat.coprime d (sequence n)) : d = 1 :=
by
  sorry

end only_one_relatively_prime_l772_772020


namespace factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l772_772594

-- Problem 1
theorem factorize_x_squared_minus_4 (x : ℝ) :
  x^2 - 4 = (x + 2) * (x - 2) :=
by { 
  sorry
}

-- Problem 2
theorem factorize_2mx_squared_minus_4mx_plus_2m (x m : ℝ) :
  2 * m * x^2 - 4 * m * x + 2 * m = 2 * m * (x - 1)^2 :=
by { 
  sorry
}

-- Problem 3
theorem factorize_y_quad (y : ℝ) :
  (y^2 - 1)^2 - 6 * (y^2 - 1) + 9 = (y + 2)^2 * (y - 2)^2 :=
by { 
  sorry
}

end factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l772_772594


namespace area_of_shaded_region_l772_772782

theorem area_of_shaded_region :
  let inner_square_side_length := 3
  let triangle_base := 2
  let triangle_height := 1
  let number_of_triangles := 8
  let area_inner_square := inner_square_side_length * inner_square_side_length
  let area_one_triangle := (1/2) * triangle_base * triangle_height
  let total_area_triangles := number_of_triangles * area_one_triangle
  let total_area_shaded := area_inner_square + total_area_triangles
  total_area_shaded = 17 :=
sorry

end area_of_shaded_region_l772_772782


namespace root_in_interval_l772_772896

noncomputable def f (x : ℝ) := x^3 + x - 1

theorem root_in_interval : ∃ r ∈ Ioo 0.6 0.7, f r = 0 := 
by
  have h1 : f 0.6 < 0 := by norm_num1
  have h2 : 0 < f 0.7 := by norm_num1
  apply IntermediateValueTheorem (f 0.6) (f 0.7) h1 h2 0.6 0.7 _ _
  sorry

end root_in_interval_l772_772896


namespace students_with_all_three_pets_l772_772761

theorem students_with_all_three_pets :
  ∀ (total_students dog_owners cat_owners other_pet_owners no_pet_students dog_only cat_only other_only x y w z : ℕ),
    total_students = 40 →
    dog_owners = 20 →
    cat_owners = 10 →
    other_pet_owners = 8 →
    no_pet_students = 5 →
    dog_only = 15 →
    cat_only = 4 →
    other_only = 5 →
    dog_only + y + w + z = dog_owners →
    cat_only + x + w + z = cat_owners →
    other_only + y + x + z = other_pet_owners →
    dog_only + cat_only + other_only + y + x + w + 2 * z = total_students - no_pet_students →
    z = 1 :=
by
  intros total_students dog_owners cat_owners other_pet_owners no_pet_students dog_only cat_only other_only x y w z
  intros h_total_students h_dog_owners h_cat_owners h_other_pet_owners h_no_pet_students
  intros h_dog_only h_cat_only h_other_only h_eq1 h_eq2 h_eq3 h_eq_total_pet_owners
  have h1 : y + x + z = 3,
  { sorry },
  have h2 : y + w + z = 5,
  { sorry },
  have h3 : x + w + z = 6,
  { sorry },
  have h4 : y + x + w + 2 * z = 16,
  { sorry },
  have := calc
    y + x + w + 2 * z = 16 : h4
    (y + w + z) + (x + w + z) - (w + z) + (y + x + z) = 16 : h4, sorry,
  Sorry := sorry,
  exact sorry

end students_with_all_three_pets_l772_772761


namespace log_5_3200_to_integer_l772_772484

noncomputable def log_5_3200 : ℝ := Real.logb 5 3200

theorem log_5_3200_to_integer : Int.round (log_5_3200) = 5 :=
by
  -- Proof will be skipped with sorry; the goal is to ensure this statement builds correctly.
  sorry

end log_5_3200_to_integer_l772_772484


namespace grid_representation_l772_772302

-- Definitions corresponding to the conditions
def grid_11x1 : Prop := (1 = 1)
def broken_lines_5 (n : ℕ) : Prop := n = 8
def length_each_5 : Prop := 5 = 5
def broken_lines_8 (m : ℕ) : Prop := m = 5
def length_each_8 : Prop := 8 = 8

-- The proposition to prove
theorem grid_representation : grid_11x1 →
  (broken_lines_5 8 → length_each_5 → True) ∧
  (broken_lines_8 5 → length_each_8 → False) :=
by
  intros _ h1 h2
  split
  · intros _ _
    exact trivial
  · intros _ _
    exact false.elim sorry

end grid_representation_l772_772302


namespace expression_calculates_to_l772_772163

noncomputable def mixed_number : ℚ := 3 + 3 / 4

noncomputable def decimal_to_fraction : ℚ := 2 / 10

noncomputable def given_expression : ℚ := ((mixed_number * decimal_to_fraction) / 135) * 5.4

theorem expression_calculates_to : given_expression = 0.03 := by
  sorry

end expression_calculates_to_l772_772163


namespace maximum_area_of_triangle_PIE_l772_772981

open Real

noncomputable def area_triangle_PIE_max (P I E : Point) (PLUM : set Point) : ℝ :=
  if h : (centered_inscribed_circle PLUM P I E ∧ collinear_points U I E) then some (max_area_triangle_PIE PLUM) else 0

theorem maximum_area_of_triangle_PIE 
  (PLUM : set Point) 
  (ω : InscribedCircle PLUM)
  (I E : Point) 
  (h1 : CollinearPoints U I E) 
  (h2 : OnCircle ω I E) 
  : 
  area_triangle_PIE_max (point 2 2) I E PLUM = 1 / 4 :=
sorry

end maximum_area_of_triangle_PIE_l772_772981


namespace num_square_free_odds_l772_772557

noncomputable def is_square_free (m : ℕ) : Prop :=
  ∀ n : ℕ, n^2 ∣ m → n = 1

noncomputable def count_square_free_odds : ℕ :=
  (199 - 1) / 2 - (11 + 4 + 2 + 1 + 1 + 1)

theorem num_square_free_odds : count_square_free_odds = 79 := by
  sorry

end num_square_free_odds_l772_772557


namespace angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l772_772574

variable (a b c A B C : ℝ)

-- Condition 1
def cond1 : Prop := b / a = (Real.cos B + 1) / (Real.sqrt 3 * Real.sin A)

-- Condition 2
def cond2 : Prop := 2 * b * Real.sin A = a * Real.tan B

-- Condition 3
def cond3 : Prop := (c - a = b * Real.cos A - a * Real.cos B)

-- Angle B and area of the triangle for Condition 1
theorem angle_B_cond1 (h : cond1 a b A B) : B = π / 3 := sorry

theorem area_range_cond1 (h : cond1 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 2
theorem angle_B_cond2 (h : cond2 a b A B) : B = π / 3 := sorry

theorem area_range_cond2 (h : cond2 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 3
theorem angle_B_cond3 (h : cond3 a b c A B) : B = π / 3 := sorry

theorem area_range_cond3 (h : cond3 a b c A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

end angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l772_772574


namespace eq_represents_two_lines_l772_772753

theorem eq_represents_two_lines (m : ℝ) :
  (∃ (a b c d : ℝ), x^2 - my^2 + 2x + 2y = (a*x + b*y + c) * (d*x + e*y + f)) → m = 1 :=
sorry

end eq_represents_two_lines_l772_772753


namespace arithmetic_sequence_a5_l772_772781

variable (a : ℕ → ℝ)

-- Conditions translated to Lean definitions
def cond1 : Prop := a 3 = 7
def cond2 : Prop := a 9 = 19

-- Theorem statement that needs to be proved
theorem arithmetic_sequence_a5 (h1 : cond1 a) (h2 : cond2 a) : a 5 = 11 :=
sorry

end arithmetic_sequence_a5_l772_772781


namespace geometric_sequence_general_term_sequence_sum_of_bn_l772_772232

theorem geometric_sequence_general_term :
  ∃ (a : ℕ → ℝ), (a 1 = 2) ∧ (∀ n, a (n + 2) / a (n + 1) = (a 2 / a 1)^n) :=
  sorry

theorem sequence_sum_of_bn (n : ℕ) :
  let a : ℕ → ℝ := λ n, 2^n in
  let S : ℕ → ℝ := λ n, 2^(n + 1) - 2 in
  let b : ℕ → ℝ := λ n, a (n + 1) / (S n * S (n + 1)) in
  let T : ℕ → ℝ := λ n, (∑ i in (finset.range n), b i) in
  T n = 1 / 2 - 1 / (2^(n + 2) - 2) :=
  sorry

end geometric_sequence_general_term_sequence_sum_of_bn_l772_772232


namespace exists_nat_numbers_with_conditions_l772_772994

def count_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).filter (λ d, n % d = 0).length

def sum_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).filter (λ d, n % d = 0).sum

theorem exists_nat_numbers_with_conditions :
  ∃ x y : ℕ, 
    count_divisors x = count_divisors y ∧
    x > y ∧
    sum_divisors x < sum_divisors y :=
by {
  let x := 38,
  let y := 39,
  have h1 : count_divisors x = count_divisors y := by {
    unfold count_divisors,
    sorry
  },
  have h2 : x > y := by {
    sorry
  },
  have h3 : sum_divisors x < sum_divisors y := by {
    unfold sum_divisors,
    sorry
  },
  exact ⟨x, y, h1, h2, h3⟩,
}

end exists_nat_numbers_with_conditions_l772_772994


namespace find_common_ratio_sum_arithmetic_sequence_l772_772712

-- Conditions
variable {a : ℕ → ℝ}   -- a_n is a numeric sequence
variable (S : ℕ → ℝ)   -- S_n is the sum of the first n terms
variable {q : ℝ}       -- q is the common ratio
variable (k : ℕ)

-- Given: a_n is a geometric sequence with common ratio q, q ≠ 1, q ≠ 0
variable (h_geometric : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (h_q_ne_zero : q ≠ 0)

-- Given: S_n = a_1 * (1 - q^n) / (1 - q) when q ≠ 1 and q ≠ 0
variable (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))

-- Given: a_5, a_3, a_4 form an arithmetic sequence, so 2a_3 = a_5 + a_4
variable (h_arithmetic : 2 * a 3 = a 5 + a 4)

-- Prove part 1: common ratio q is -2
theorem find_common_ratio (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : 2 * a 3 = a 5 + a 4) 
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0) : q = -2 :=
sorry

-- Prove part 2: S_(k+2), S_k, S_(k+1) form an arithmetic sequence
theorem sum_arithmetic_sequence (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0)
  (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))
  (k : ℕ) : S (k + 2) + S k = 2 * S (k + 1) :=
sorry

end find_common_ratio_sum_arithmetic_sequence_l772_772712


namespace find_n_l772_772500

-- Definitions based on conditions
def a := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7
def b (n : ℕ) := 2 * n

-- Theorem stating the problem
theorem find_n (n : ℕ) (h : a^2 - (b n)^2 = 0) : n = 10 :=
by sorry

end find_n_l772_772500


namespace find_constant_l772_772294

variable (constant : ℝ)

theorem find_constant (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 1 - 2 * t)
  (h2 : y = constant * t - 2)
  (h3 : x = y) : constant = 2 :=
by
  sorry

end find_constant_l772_772294


namespace angle_equality_l772_772323

variables {A B C D E F P Q: Type*}
variables [convex_quadrilateral A B C D]
variables [point_on_segment F A D]
variables [point_on_segment E B C]
variables [ratio AF FD BE EC AB CD]

theorem angle_equality : 
  let P := intersection (extension_segment E F) (line_through A B),
      Q := intersection (extension_segment E F) (line_through C D)
  in ∠B P E = ∠C Q E :=
sorry

end angle_equality_l772_772323


namespace max_omega_l772_772422

theorem max_omega (ω : ℝ) (hω : 0 < ω) (g : ℝ → ℝ) :
  (∀ x : ℝ, g x = 2 * sin (ω * x)) →
  (∀ x : ℝ, -π / 6 ≤ x ∧ x ≤ π / 4 → (differentiable_at ℝ g x ∧ 0 < deriv g x)) →
  ω ≤ 2 :=
by
  sorry

end max_omega_l772_772422


namespace arithmetic_series_sum_l772_772862

theorem arithmetic_series_sum (k : ℤ) : 
  let a₁ := k^2 + k + 1 
  let n := 2 * k + 3 
  let d := 1 
  let aₙ := a₁ + (n - 1) * d 
  let S_n := n / 2 * (a₁ + aₙ)
  S_n = 2 * k^3 + 7 * k^2 + 10 * k + 6 := 
by {
  sorry
}

end arithmetic_series_sum_l772_772862


namespace total_tickets_l772_772144

theorem total_tickets (A C total_tickets total_cost : ℕ) 
  (adult_ticket_cost : ℕ := 8) (child_ticket_cost : ℕ := 5) 
  (total_cost_paid : ℕ := 201) (child_tickets_count : ℕ := 21) 
  (ticket_cost_eqn : 8 * A + 5 * 21 = 201) 
  (adult_tickets_count : A = total_cost_paid - (child_ticket_cost * child_tickets_count) / adult_ticket_cost) :
  total_tickets = A + child_tickets_count :=
sorry

end total_tickets_l772_772144


namespace triangle_side_length_l772_772346

theorem triangle_side_length (A B C M : Point)
  (hAB : dist A B = 2)
  (hAC : dist A C = 3)
  (hMidM : M = midpoint B C)
  (hAM_BC : dist A M = dist B C) :
  dist B C = Real.sqrt (78) / 3 :=
by
  sorry

end triangle_side_length_l772_772346


namespace fraction_of_evaporated_water_l772_772139

namespace evaporation_problem

def initial_volume : ℝ := 119.99999999999996
def initial_salt_concentration : ℝ := 0.20
def final_salt_concentration : ℝ := 1 / 3
def added_water : ℝ := 8
def added_salt : ℝ := 16

theorem fraction_of_evaporated_water :
  let w := 143.99999999999996 - 120 in
  w / initial_volume = 1 / 5 :=
by
  sorry

end evaporation_problem

end fraction_of_evaporated_water_l772_772139


namespace robot_minimal_moves_l772_772083

def move : Type := char

def count_moves (moves : List move) (dir : move) : Nat :=
  moves.countP (fun x => x = dir)

def net_moves (moves : List move) (dir1 dir2 : move) : Int :=
  (count_moves moves dir1 : Int) - (count_moves moves dir2 : Int)

def target_reached_in_minimal_moves (moves : List move) : Nat :=
  Int.natAbs (net_moves moves 'E' 'W') + Int.natAbs (net_moves moves 'N' 'S')

theorem robot_minimal_moves :
  target_reached_in_minimal_moves ['E', 'E', 'N', 'E', 'N', 'N', 'W', 'N', 'E', 'N', 'E', 'S', 'E', 'E', 'E', 'E', 'S', 'S', 'S', 'W', 'N'] = 10 :=
by
  sorry

end robot_minimal_moves_l772_772083


namespace fibonacci_odd_index_not_divisible_by_4k_plus_3_l772_772823

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_odd_index_not_divisible_by_4k_plus_3 (n k : ℕ) (p : ℕ) (h : p = 4 * k + 3) : ¬ (p ∣ fibonacci (2 * n - 1)) :=
by
  sorry

end fibonacci_odd_index_not_divisible_by_4k_plus_3_l772_772823


namespace alien_home_50_people_l772_772966

/-- The alien abducted 200 people. -/
def total_abducted : ℕ := 200

/-- The alien returned 80% of the abducted people. -/
def percent_returned : ℚ := 0.80

/-- The number of people the alien took to another planet after returning. -/
def taken_to_another_planet : ℕ := 10

/-- The number of people returned to Earth. -/
def people_returned : ℕ := (percent_returned * total_abducted).natAbs

/-- The number of people the alien initially kept. -/
def kept_initially : ℕ := total_abducted - people_returned

/-- The total number of people the alien took to his home planet. -/
def total_taken_home : ℕ := kept_initially + taken_to_another_planet

/-- The proof statement to be validated. -/
theorem alien_home_50_people : total_taken_home = 50 :=
by sorry

end alien_home_50_people_l772_772966


namespace range_of_a_l772_772751

theorem range_of_a (a : ℝ) : (2 * a - 1)^0 = 1 → a ≠ 1 / 2 := 
by 
  intro h,
  have h1 : (2 * a - 1 = 0) → False := by 
    intro h_eq_zero,
    exact (zero_pow zero_lt_one).symm ▸ h_eq_zero,
  by_contradiction h2,
  exact h1 h2.1

end range_of_a_l772_772751


namespace sum_of_unit_vectors_is_zero_pairwise_opposite_l772_772063

open_locale classical

noncomputable def is_unit_vector (v : ℝ^3) : Prop := ∥v∥ = 1

theorem sum_of_unit_vectors_is_zero_pairwise_opposite
  (v1 v2 v3 v4 : ℝ^3)
  (h1 : is_unit_vector v1)
  (h2 : is_unit_vector v2)
  (h3 : is_unit_vector v3)
  (h4 : is_unit_vector v4)
  (h_sum : v1 + v2 + v3 + v4 = 0) :
  ∃ a b c d : ℝ^3,
  (a + b = 0) ∧ (c + d = 0) ∧ 
  ({v1, v2, v3, v4} = {a, b, c, d} ∨ 
   {v1, v2, v3, v4} = {a, c, b, d} ∨ 
   {v1, v2, v3, v4} = {a, d, b, c} ∨ 
   {v1, v2, v3, v4} = {c, a, b, d} ∨ 
   {v1, v2, v3, v4} = {c, b, a, d} ∨ 
   {v1, v2, v3, v4} = {c, d, a, b} ∨ 
   {v1, v2, v3, v4} = {d, a, b, c} ∨ 
   {v1, v2, v3, v4} = {d, b, a, c} ∨ 
   {v1, v2, v3, v4} = {d, c, a, b}) := 
sorry

end sum_of_unit_vectors_is_zero_pairwise_opposite_l772_772063


namespace opposite_of_negative_a_is_a_l772_772496

-- Define the problem:
theorem opposite_of_negative_a_is_a (a : ℝ) : -(-a) = a :=
by 
  sorry

end opposite_of_negative_a_is_a_l772_772496


namespace sequence_formula_l772_772739

noncomputable def a : ℕ → ℕ
| 0       := 0  -- This is a placeholder index, since our sequence starts from a_1 = 1
| 1       := 1
| (n + 1) := 2 * a n + 1

theorem sequence_formula (n : ℕ) : a (n + 1) = 2^(n + 1) - 1 :=
by induction n with k hk
   -- Base case
   case zero { simp [a] }
   -- Inductive step
   case succ {
     simp [a],
     rw [hk, pow_succ],
     ring,
   }

end sequence_formula_l772_772739


namespace min_value_of_a4_l772_772907

def is_distinct {α : Type*} [DecidableEq α] (l : List α) : Prop := l.Nodup

theorem min_value_of_a4 :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ), 
    a2 = a1 + a5 ∧
    a3 = a2 + a6 ∧
    a4 = a3 + a7 ∧
    a7 = a6 + a9 ∧
    a6 = a5 + a8 ∧
    a9 = a8 + a10 ∧
    is_distinct [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] ∧
    a4 = 20 :=
begin
  sorry
end

end min_value_of_a4_l772_772907


namespace min_stamps_to_meet_postage_l772_772132

def stamp_units := {0.4, 0.8, 1.5}
def postage_requirement : ℝ := 10.2

theorem min_stamps_to_meet_postage : ∃ n, n = 8 ∧
  (∃ a b c : ℕ, a * 0.4 + b * 0.8 + c * 1.5 = postage_requirement ∧ a + b + c = n) :=
by sorry

end min_stamps_to_meet_postage_l772_772132


namespace prime_divisor_gte_11_l772_772798

theorem prime_divisor_gte_11 (n : ℤ) (h1 : 10 < n) (h2 : ∀ (d : ℕ), d ∈ digits 10 n → d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9) : 
  ∃ p : ℕ, prime p ∧ p ≥ 11 ∧ p ∣ n :=
by sorry

end prime_divisor_gte_11_l772_772798


namespace abs_g_eq_l772_772865

def g (x : ℝ) : ℝ :=
if -4 ≤ x ∧ x < -1 then x^2 - 4
else if -1 ≤ x ∧ x < 2 then -2 * x - 1
else if 2 ≤ x ∧ x ≤ 4 then x - 2
else 0

def abs_g (x : ℝ) : ℝ :=
if -4 ≤ x ∧ x < -1 then x^2 - 4
else if -1 ≤ x ∧ x < 2 then 2 * x + 1
else if 2 ≤ x ∧ x ≤ 4 then x - 2
else 0

theorem abs_g_eq : ∀ x : ℝ, |g x| = abs_g x :=
by
  intro x
  sorry

end abs_g_eq_l772_772865


namespace channels_taken_away_l772_772790

theorem channels_taken_away (X : ℕ) : 
  (150 - X + 12 - 10 + 8 + 7 = 147) -> X = 20 :=
by
  sorry

end channels_taken_away_l772_772790


namespace smallest_x_l772_772425

theorem smallest_x (x : ℚ) (h : 7 * (4 * x^2 + 4 * x + 5) = x * (4 * x - 35)) : 
  x = -5/3 ∨ x = -7/8 := by
  sorry

end smallest_x_l772_772425


namespace sqrt_sqrt4_16_eq_1_4_l772_772976

noncomputable def sqrt_to_nearest_tenth (x : ℝ) : ℝ :=
  Float.ceil (x * 10) / 10

theorem sqrt_sqrt4_16_eq_1_4 :
  sqrt_to_nearest_tenth (Real.sqrt (Real.sqrt (16))) = 1.4 :=
by
  have h1 : (16 : ℝ) = 2^4 := by norm_num
  have h2 : Real.sqrt (Real.sqrt 16) = Real.sqrt 2 := by rw [Real.sqrt_sqrt h1, Real.sqrt_sqrt2_pow4]
  suffices Real.sqrt 2 ≈ 1.414 by
    sorry
  -- Here approximation to 1.414 is required to prove the rounding.
  sorry

end sqrt_sqrt4_16_eq_1_4_l772_772976


namespace beehive_paths_count_l772_772520

theorem beehive_paths_count :
  let cells := 10 in
  let start_cell := 1 in
  let end_cell := 10 in
  (∀ (path : fin cells → fin cells), 
    -- Condition 1: Start at cell 1
    path 0 = start_cell ∧ 
    -- Condition 2: End at cell 10
    path (cells-1) = end_cell ∧ 
    -- Condition 3: Move only to adjacent cells 
    (∀ (i : fin (cells-1)), abs (path i.succ - path i) = 1) ∧ 
    -- Condition 4: Pass through each cell exactly once
    (∀ (i j : fin cells), i ≠ j → path i ≠ path j)) ->
  -- Conclusion: There are exactly 12 such paths
  12 := sorry

end beehive_paths_count_l772_772520


namespace brett_red_marbles_l772_772564

variables (r b : ℕ)

-- Define the conditions
axiom h1 : b = r + 24
axiom h2 : b = 5 * r

theorem brett_red_marbles : r = 6 :=
by
  sorry

end brett_red_marbles_l772_772564


namespace vanessa_points_l772_772306

theorem vanessa_points (total_points : ℕ) (num_other_players : ℕ) (avg_points_other : ℕ) 
  (h1 : total_points = 65) (h2 : num_other_players = 7) (h3 : avg_points_other = 5) :
  ∃ vp : ℕ, vp = 30 :=
by
  sorry

end vanessa_points_l772_772306


namespace find_a_l772_772042

theorem find_a (a : ℤ) (h : ∃ x1 x2 : ℤ, (x - x1) * (x - x2) = (x - a) * (x - 8) - 1) : a = 8 :=
sorry

end find_a_l772_772042


namespace complement_intersection_l772_772230

def U : Set ℤ := {1, 2, 3, 4, 5}
def P : Set ℤ := {2, 4}
def Q : Set ℤ := {1, 3, 4, 6}
def C_U_P : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_intersection :
  (C_U_P ∩ Q) = {1, 3} :=
by sorry

end complement_intersection_l772_772230


namespace sin_double_alpha_l772_772257

def f (x : ℝ) : ℝ := cos^2 (x / 2) - sin (x / 2) * cos (x / 2) - 1 / 2

theorem sin_double_alpha (α : ℝ) (h : f α = 3 * sqrt 2 / 10) : 
  sin (2 * α) = 7 / 25 := 
by
  sorry

end sin_double_alpha_l772_772257


namespace peyton_score_l772_772407

-- Conditions as definitions
def total_students := 15
def average_score_excluding_peyton := 80
def average_score_including_peyton := 81

-- The proof problem statement
theorem peyton_score : 
  (let S14 := 14 * average_score_excluding_peyton in
   let S15 := total_students * average_score_including_peyton in
   let P := S15 - S14 in
   P = 95) :=
sorry

end peyton_score_l772_772407


namespace product_of_C_l772_772804

noncomputable def A (m : ℝ) : Set ℝ := {1, 2, m}
noncomputable def B (m : ℝ) : Set ℝ := {a^2 | a ∈ A m}
noncomputable def C (m : ℝ) : Set ℝ := A m ∪ B m

theorem product_of_C (m : ℝ) (h_sum : (∑ c in C m, c) = 6) :
  (∏ c in C m, c) = -8 :=
sorry

end product_of_C_l772_772804


namespace original_plan_trees_average_l772_772311

-- Definitions based on conditions
def original_trees_per_day (x : ℕ) := x
def increased_trees_per_day (x : ℕ) := x + 5
def time_to_plant_60_trees (x : ℕ) := 60 / (x + 5)
def time_to_plant_45_trees (x : ℕ) := 45 / x

-- The main theorem we need to prove
theorem original_plan_trees_average : ∃ x : ℕ, time_to_plant_60_trees x = time_to_plant_45_trees x ∧ x = 15 :=
by
  -- Placeholder for the proof
  sorry

end original_plan_trees_average_l772_772311


namespace maximum_value_of_x_minus_y_l772_772705

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l772_772705


namespace _l772_772973

noncomputable def problem_problem (A B C E F D P : Type*) (circ_AEF: Type*) (circ_ABC: Type*) [right_triangle : triangle A B C]
  (h1 : E ∈ AC) (h2 : F ∈ AB) (h3 : BE ∩ CF = D) (h4 : P ∈ (circ_AEF ∩ circ_ABC))
  (circ_AEF : circumcircle (triangle A E F)) (circ_ABC : circumcircle (triangle A B C)) : Prop :=
  ∃ AP PD,
  AP ⊥ PD

noncomputable def main_theorem : problem_problem sorry sorry sorry sorry sorry sorry :=
sorry

#print axioms main_theorem

end _l772_772973


namespace smallest_possible_b_l772_772852

theorem smallest_possible_b 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a - b = 8) 
  (h4 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  b = 4 := 
sorry

end smallest_possible_b_l772_772852


namespace sin_pi_over_3_minus_2theta_l772_772623

theorem sin_pi_over_3_minus_2theta (θ : ℝ) (h : tan (θ + π / 12) = 2) :
  sin (π / 3 - 2 * θ) = -3 / 5 :=
by
  sorry

end sin_pi_over_3_minus_2theta_l772_772623


namespace area_intersection_eq_pi_l772_772732

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

def M : set (ℝ × ℝ) := { p | f p.1 + f p.2 ≤ 0 }
def N : set (ℝ × ℝ) := { p | f p.1 - f p.2 ≥ 0 }

theorem area_intersection_eq_pi : 
  let intersection := M ∩ N in
  ∃ area, area = π ∧ 
  (∀ s : set (ℝ × ℝ), s = intersection → measure_theory.measure_space.volume.measure_of s = area) :=
  sorry

end area_intersection_eq_pi_l772_772732


namespace max_difference_value_l772_772679

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l772_772679


namespace nyusha_candies_l772_772840

theorem nyusha_candies (K E N B : ℕ) (h1 : K + E + N + B = 86)
    (h2 : K + E = 53) (h3 : K ≥ 5) (h4 : E ≥ 5)
    (h5 : N ≥ 5) (h6 : B ≥ 5) (h7 : N > K) (h8 : N > E) (h9 : N > B) :
    N = 28 :=
begin
  sorry
end

end nyusha_candies_l772_772840


namespace elasticity_ratio_l772_772162

theorem elasticity_ratio (e_QN e_PN : ℝ) (h1 : e_QN = 1.27) (h2 : e_PN = 0.76) : 
  (e_QN / e_PN) ≈ 1.7 :=
by
  rw [h1, h2]
  -- prove the statement using the given conditions
  sorry

end elasticity_ratio_l772_772162


namespace find_r_l772_772248

noncomputable def radius_of_tangent_circles
  (a b c : ℝ)
  (h1 : a = 13)
  (h2 : b = 14)
  (h3 : c = 15)
  (tangent_O1_AB : O1 ∈ tangent AB)
  (tangent_O1_AC : O1 ∈ tangent AC)
  (tangent_O2_BA : O2 ∈ tangent BA)
  (tangent_O2_BC : O2 ∈ tangent BC)
  (tangent_O3_CB : O3 ∈ tangent CB)
  (tangent_O3_CA : O3 ∈ tangent CA)
  (tangent_O_O1 : O ∈ tangent O1)
  (tangent_O_O2 : O ∈ tangent O2)
  (tangent_O_O3 : O ∈ tangent O3) : ℝ :=
r

-- We then show that:
theorem find_r : radius_of_tangent_circles 13 14 15 = 260 / 129 :=
sorry

end find_r_l772_772248


namespace find_x_l772_772744

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (8, 1/2 * x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : vector_a x = (8, 1/2 * x)) 
(h3 : vector_b x = (x, 1)) 
(h4 : ∀ k : ℝ, (vector_a x).1 = k * (vector_b x).1 ∧ 
                       (vector_a x).2 = k * (vector_b x).2) : 
                       x = 4 := sorry

end find_x_l772_772744


namespace find_f_when_x_lt_0_l772_772715

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_defined (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 2 * x

theorem find_f_when_x_lt_0 (f : ℝ → ℝ) (h_odd : odd_function f) (h_defined : f_defined f) :
  ∀ x < 0, f x = -x^2 - 2 * x :=
by
  sorry

end find_f_when_x_lt_0_l772_772715


namespace range_of_t_l772_772253

noncomputable section

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 3^x
else if 1 < x ∧ x ≤ 3 then (9 / 2) - (3 / 2) * x
else 0

theorem range_of_t (t : ℝ) (h1 : 0 ≤ t ∧ t ≤ 1) (h2 : (0 ≤ f (f t)) ∧ (f (f t) ≤ 1)) :
  (real.log (7 / 3) / real.log 3) ≤ t ∧ t ≤ 1 :=
sorry

end range_of_t_l772_772253


namespace geometric_sequence_problem_l772_772301

noncomputable def a3_a4_a5_sum : ℕ :=
  let a_1 := 3
  let q := Nat.sqrt(21 / a_1).natAbs
  let a_2 := a_1 * q
  let a_3 := a_2 * q
  let a_4 := a_3 * q
  let a_5 := a_4 * q
  a_3 + a_4 + a_5

theorem geometric_sequence_problem (a₁ q : ℝ) (h_pos : 0 < a₁) (h_a₁ : a₁ = 3) (h_sum_first_three : a₁ + a₁ * q + a₁ * q^2 = 21) :
  a₁ * q^2 + a₁ * q^3 + a₁ * q^4 = 84 :=
by
  rw [h_a₁] at h_sum_first_three
  have h_q : q = 2 := by sorry
  rw [h_a₁, h_q]
  sorry

end geometric_sequence_problem_l772_772301


namespace domain_of_f_x_squared_plus_3_l772_772244

theorem domain_of_f_x_squared_plus_3 (f : ℝ → ℝ) (h : ∀ x, 3 ≤ x → x ≤ 6 → ∃ y, y = f(x + 1)) :
  ∀ x, (-2 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ 2) → ∃ y, y = f(x^2 + 3) :=
by
  intro x
  intro hx
  cases hx
  case inl hx_neg_interval =>
    let x_neg_interval := hx_neg_interval
    sorry
  case inr hx_pos_interval =>
    let x_pos_interval := hx_pos_interval
    sorry

end domain_of_f_x_squared_plus_3_l772_772244


namespace eval_expression_l772_772164

theorem eval_expression : |real.sqrt 3 - 3| - real.sqrt 16 + real.cos (real.pi / 6) + (1 / 3) ^ 0 = - (real.sqrt 3) / 2 := by
  sorry

end eval_expression_l772_772164


namespace sphere_radius_correct_l772_772541
-- Import all necessary libraries

-- Define conditions
def sphere_shadow_length : ℝ := 8
def stick_height : ℝ := 1.5
def stick_shadow_length : ℝ := 1
def tan_theta_stick := stick_height / stick_shadow_length  -- This implicitly defines the angle θ

-- Define the unknown radius of the sphere
noncomputable def radius_of_sphere : ℝ :=
  let r := sorry in r

-- State the theorem/problem to prove
theorem sphere_radius_correct :
  tan_theta_stick = radius_of_sphere / (sphere_shadow_length - radius_of_sphere) →
  radius_of_sphere = 4.8 :=
sorry

end sphere_radius_correct_l772_772541


namespace triangle_side_length_l772_772343

theorem triangle_side_length (A B C : Type*) [inner_product_space ℝ (A × B)]
  (AB AC BC : ℝ) (h1 : AB = 2)
  (h2 : AC = 3) (h3 : BC = sqrt(5.2)) :
  ACS = sqrt(5.2) :=
sorry

end triangle_side_length_l772_772343


namespace intersection_ratios_l772_772511

variables {α β : ℝ} {A B C D K N L M P : Type}
variables [ConvexQuadrilateral A B C D]
variables [OnSides K N L M A B C D]
variables (h1 : (∃ (a : ℝ), a > 0 ∧ a = α ∧ a * KB = AK))
variables (h2 : (∃ (d : ℝ), d > 0 ∧ d = α ∧ d * LC = DL))
variables (h3 : (∃ (a : ℝ), a > 0 ∧ a = β ∧ a * MP = AM))
variables (h4 : (∃ (b : ℝ), b > 0 ∧ b = β ∧ b * NC = BN))

theorem intersection_ratios :
  (∃ P : Type, (∃ (pₖ : ℝ), pₖ = α ∧ pₖ * KL = PK) ∧ (∃ (pₘ : ℝ), pₘ = β ∧ pₘ * MN = PM)) := sorry

end intersection_ratios_l772_772511


namespace coords_A2022_l772_772778

structure Point where
  x : Int
  y : Int

def companion (P : Point) : Point :=
  ⟨-P.y + 1, P.x + 1⟩

def sequence : ℕ → Point 
| 0 => ⟨2, 4⟩
| n + 1 => companion (sequence n)

theorem coords_A2022 : sequence 2022 = ⟨-3, 3⟩ := 
  sorry

end coords_A2022_l772_772778


namespace solve_inequality_l772_772261

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - (a + 2) * x + 2 < 0

-- Prove the solution sets for different values of a
theorem solve_inequality :
  ∀ (a : ℝ),
    (a = -1 → {x : ℝ | inequality a x} = {x | x < -2 ∨ x > 1}) ∧
    (a = 0 → {x : ℝ | inequality a x} = {x | x > 1}) ∧
    (a < 0 → {x : ℝ | inequality a x} = {x | x < 2 / a ∨ x > 1}) ∧
    (0 < a ∧ a < 2 → {x : ℝ | inequality a x} = {x | 1 < x ∧ x < 2 / a}) ∧
    (a = 2 → {x : ℝ | inequality a x} = ∅) ∧
    (a > 2 → {x : ℝ | inequality a x} = {x | 2 / a < x ∧ x < 1}) :=
by sorry

end solve_inequality_l772_772261


namespace count_sweet_in_1_to_60_l772_772405

def sequence_step (n: ℕ): ℕ := if n ≤ 30 then 3 * n else n - 15

def is_sweet (G: ℕ): Prop :=
  ∀ n, (sequence_step^[n] G) ≠ 18

def count_sweet_numbers (n m: ℕ): ℕ :=
  (Finset.range (m + 1)).filter (λ x, is_sweet x).card

theorem count_sweet_in_1_to_60: count_sweet_numbers 1 60 = 44 := by sorry

end count_sweet_in_1_to_60_l772_772405


namespace intersection_point_coincides_l772_772415

-- Define the problem conditions and statement
structure TangentialQuadrilateral (A B C D : Type) :=
  (inscribed_circle : exists E F G K : Type, ∀ (X : TangentialQuadrilateral), 
      tangent_to_circle E A B ∧ tangent_to_circle F B C ∧ 
      tangent_to_circle G C D ∧ tangent_to_circle K D A)

def intersection_coincides {A B C D E F G K : Type} 
  (h1 : TangentialQuadrilateral A B C D)
  (h2 : PointsOfTangency E F G K A B C D)
  (M : Type) : Prop :=
  intersection (diagonal A C) (diagonal E G) = 
  intersection (diagonal E F) (diagonal G K)

theorem intersection_point_coincides 
  {A B C D E F G K : Type}
  (h_quad : TangentialQuadrilateral A B C D)
  (h_tangency : PointsOfTangency E F G K A B C D)
  (M : Type) : Prop :=
  intersection_coincides h_quad h_tangency M
  
#eval intersection_point_coincides

end intersection_point_coincides_l772_772415


namespace maritza_study_time_l772_772401

def num_multiple_choice_questions : ℕ := 30
def num_fill_in_blank_questions : ℕ := 30
def num_essay_questions : ℕ := 30
def time_per_multiple_choice_min : ℕ := 15
def time_per_fill_in_blank_min : ℕ := 25
def time_per_essay_min : ℕ := 45
def minutes_per_hour : ℕ := 60

theorem maritza_study_time :
  (num_multiple_choice_questions * time_per_multiple_choice_min +
   num_fill_in_blank_questions * time_per_fill_in_blank_min +
   num_essay_questions * time_per_essay_min) / minutes_per_hour = 42.5 := sorry

end maritza_study_time_l772_772401


namespace probability_of_graduate_degree_l772_772921

-- Define the conditions as Lean statements
variable (k m : ℕ)
variable (G := 1 * k) 
variable (C := 2 * m) 
variable (N1 := 8 * k) -- from the ratio G:N = 1:8
variable (N2 := 3 * m) -- from the ratio C:N = 2:3

-- Least common multiple (LCM) of 8 and 3 is 24
-- Therefore, determine specific values for G, C, and N
-- Given these updates from solution steps we set:
def G_scaled : ℕ := 3
def C_scaled : ℕ := 16
def N_scaled : ℕ := 24

-- Total number of college graduates
def total_college_graduates : ℕ := G_scaled + C_scaled

-- Probability q of picking a college graduate with a graduate degree
def q : ℚ := G_scaled / total_college_graduates

-- Lean proof statement for equivalence
theorem probability_of_graduate_degree : 
  q = 3 / 19 := by
sorry

end probability_of_graduate_degree_l772_772921


namespace sin_identity_l772_772621

noncomputable def θ : Real := sorry  -- θ is an unspecified angle in this context.

axiom tan_condition : Real.tan (θ + Real.pi / 12) = 2

theorem sin_identity : sin(Real.pi / 3 - 2 * θ) = -3 / 5 := by
  -- This is where the proof would go
  sorry

end sin_identity_l772_772621


namespace sum_of_squared_residuals_l772_772738

theorem sum_of_squared_residuals : 
  let f : ℝ → ℝ := λ x, 2 * x + 1
  let data_points := [(2, 4.9), (3, 7.1), (4, 9.1)]
  let residuals := data_points.map (λ p, p.2 - f p.1)
  let squared_residuals := residuals.map (λ e, e ^ 2)
  (squared_residuals.sum = 0.03) := 
by {
  let f : ℝ → ℝ := λ x, 2 * x + 1
  let data_points := [(2, 4.9), (3, 7.1), (4, 9.1)]
  let residuals := data_points.map (λ p, p.2 - f p.1)
  let squared_residuals := residuals.map (λ e, e ^ 2)
  have : squared_residuals.sum = 0.03 := sorry,
  exact this
}

end sum_of_squared_residuals_l772_772738


namespace count_four_digit_numbers_with_thousands_digit_one_l772_772273

theorem count_four_digit_numbers_with_thousands_digit_one : 
  ∃ N : ℕ, N = 1000 ∧ (∀ n : ℕ, 1000 ≤ n ∧ n < 2000 → (n / 1000 = 1)) :=
sorry

end count_four_digit_numbers_with_thousands_digit_one_l772_772273


namespace robert_books_read_l772_772421

variable (reading_speed : ℕ) (time_hours : ℕ) (pages_per_book1 pages_per_book2 : ℕ)

theorem robert_books_read
  (h_speed : reading_speed = 120)
  (h_time : time_hours = 6)
  (h_book1 : pages_per_book1 = 240)
  (h_book2 : pages_per_book2 = 360) :
  ∃ (book_count1 book_count2 : ℕ), book_count1 + book_count2 = 5 :=
by
  have h_books_read : ∀ pages_per_book, pages_per_book / reading_speed = pages_per_book1 / 120 → pages_per_book / 120 = 2 → (6 / 2) = 3 → ∀ pages_per_book, pages_per_book / reading_speed = pages_per_book2 / 120 → pages_per_book / 120 = 3 → (6 / 3) = 2 → 5 = 2 + 3 := sorry
  exact ⟨3, 2, by
    apply h_books_read;
    sorry
  ⟩

end robert_books_read_l772_772421


namespace largest_three_digit_geometric_sequence_l772_772089

-- Definitions based on conditions
def is_three_digit_integer (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def digits_distinct (n : ℕ) : Prop := 
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃
def geometric_sequence (n : ℕ) : Prop :=
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ != 0 ∧ d₂ != 0  ∧ d₃ != 0 ∧ 
  (∃ r: ℚ, d₂ = d₁ * r ∧ d₃ = d₂ * r)

-- Theorem statement
theorem largest_three_digit_geometric_sequence : 
  ∃ n : ℕ, is_three_digit_integer n ∧ digits_distinct n ∧ geometric_sequence n ∧ n = 964 :=
sorry

end largest_three_digit_geometric_sequence_l772_772089


namespace range_sqrt_x2_y2_l772_772053

noncomputable def dist_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

open Real

theorem range_sqrt_x2_y2 (x y : ℝ) :
  (x + y = 10) → (-5 ≤ x - y ∧ x - y ≤ 5) → 
  (∃ a b : ℝ, set.Icc a b = set.range (λ (x y : ℝ), sqrt (x^2 + y^2))):=
by
  intros h1 h2
  let d := dist_point_to_line 0 0 1 1 (-10)
  have hd : d = 10 / sqrt 2, sorry
  let min_dist := 5 * sqrt 2
  let max_dist := (5 * sqrt 10) / 2
  use [min_dist, max_dist]
  sorry -- Detailed proof omitted

end range_sqrt_x2_y2_l772_772053


namespace Linda_snakes_l772_772397

theorem Linda_snakes (snakes : Type) (Green Smart CanDance CanSing : snakes → Prop)
  (h1 : ∀ s, Smart s → CanDance s)  -- All smart snakes can dance
  (h2 : ∀ s, Green s → ¬ CanSing s)  -- No green snake can sing
  (h3 : ∀ s, ¬ CanSing s → ¬ CanDance s)  -- All snakes that cannot sing also cannot dance
  : ∀ s, Smart s → ¬ Green s :=      -- Smart snakes are not green
by
  intro s hs
  apply Classical.by_contradiction
  intro hgs
  have hns : ¬ CanSing s :=
    h2 s hgs
  have hnd : ¬ CanDance s :=
    h3 s hns
  apply hnd
  apply h1 s hs

end Linda_snakes_l772_772397


namespace total_number_of_triangles_is_24_l772_772988

-- Define the vertices of the square
variables {A B C D O M N P Q : Type}

-- Define the properties of the square
variable (square : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ {A, B, C, D}.pairwise (≠))
variable (diagonals : AC ∩ BD = {O})
variable (midpoints : M ≠ N ∧ N ≠ P ∧ P ≠ Q ∧ Q ≠ M)
variable (circle_inscribed : circle M N P Q ∧ touches M ∧ touches N ∧ touches P ∧ touches Q)

-- Mathematical proof statement
theorem total_number_of_triangles_is_24 
  (h_square : square)
  (h_diagonals : diagonals)
  (h_midpoints : midpoints)
  (h_circle_inscribed : circle_inscribed) : 
  total_triangles = 24 := 
  sorry

end total_number_of_triangles_is_24_l772_772988


namespace betty_additional_money_needed_l772_772154

def wallet_cost : ℝ := 100
def betty_savings : ℝ := wallet_cost / 2
def parents_contribution : ℝ := 15
def grandparents_contribution : ℝ := 2 * parents_contribution

def total_money : ℝ := betty_savings + parents_contribution + grandparents_contribution
def amount_needed : ℝ := wallet_cost - total_money

theorem betty_additional_money_needed : amount_needed = 5 := by
  sorry

end betty_additional_money_needed_l772_772154


namespace strictly_increasing_intervals_l772_772601

def is_increasing_interval (k : ℤ) : set ℝ :=
  {x : ℝ | (2 * k * real.pi - real.pi / 3) ≤ x ∧ x ≤ (2 * k * real.pi + 2 * real.pi / 3)}

def f (x : ℝ) : ℝ :=
  sqrt 3 * real.cos (x - real.pi / 2) + real.cos (real.pi - x)

theorem strictly_increasing_intervals :
  ∀ k : ℤ, ∀ x1 x2 : ℝ,
    x1 ∈ is_increasing_interval k → x2 ∈ is_increasing_interval k → x1 < x2 → f x1 < f x2 :=
sorry

end strictly_increasing_intervals_l772_772601


namespace family_soft_tacos_l772_772953

-- Conditions
variable (price_soft_taco : ℕ) (price_hard_taco : ℕ) (num_hard_tacos_family : ℕ)
variable (num_other_customers : ℕ) (soft_tacos_per_customer : ℕ) (total_amount : ℕ)

-- Set the specific values based on given conditions
def price_soft_taco := 2
def price_hard_taco := 5
def num_hard_tacos_family := 4
def num_other_customers := 10
def soft_tacos_per_customer := 2
def total_amount := 66

-- Definition of the proof problem
theorem family_soft_tacos : 
  (total_amount = num_hard_tacos_family * price_hard_taco + 
  num_other_customers * (soft_tacos_per_customer * price_soft_taco) + 
  soft_tacos_family * price_soft_taco) → 
  soft_tacos_family = 3 := 
by
  sorry

end family_soft_tacos_l772_772953


namespace mutually_exclusive_both_miss_hitting_at_least_once_l772_772846

open Finset

variable {α : Type} [DecidableEq α]

/-- Definitions of the shooting events -/
def hitting_at_least_once (shots: Finset α) : Prop := (shots.card > 0)
def both_miss (shots: Finset α) : Prop := (shots.card = 0)

/-- Mutually exclusive events -/
def mutually_exclusive (A B : Finset α → Prop) : Prop :=
  ∀ shots, ¬ (A shots ∧ B shots)

-- Proof statement
theorem mutually_exclusive_both_miss_hitting_at_least_once :
  mutually_exclusive hitting_at_least_once both_miss :=
by
  intros shots
  unfold mutually_exclusive hitting_at_least_once both_miss
  sorry

end mutually_exclusive_both_miss_hitting_at_least_once_l772_772846


namespace determine_specialty_l772_772075

variables 
  (Peter_is_mathematician Sergey_is_physicist Roman_is_physicist : Prop)
  (Peter_is_chemist Sergey_is_mathematician Roman_is_chemist : Prop)

-- Conditions
axiom cond1 : Peter_is_mathematician → ¬ Sergey_is_physicist
axiom cond2 : ¬ Roman_is_physicist → Peter_is_mathematician
axiom cond3 : ¬ Sergey_is_mathematician → Roman_is_chemist

theorem determine_specialty 
  (h1 : ¬ Roman_is_physicist)
: Peter_is_chemist ∧ Sergey_is_mathematician ∧ Roman_is_physicist := 
by sorry

end determine_specialty_l772_772075


namespace erwan_total_expenditure_l772_772591

theorem erwan_total_expenditure
  (shoe_price : ℤ) (shoe_discount : ℤ)
  (shirt_price : ℤ) (num_shirts : ℕ)
  (additional_discount : ℤ) :
  shoe_price = 200 →
  shoe_discount = 30 →
  shirt_price = 80 →
  num_shirts = 2 →
  additional_discount = 5 →
  let discounted_shoe_price := shoe_price - (shoe_price * shoe_discount / 100) in
  let shirt_total_price := num_shirts * shirt_price in
  let total_price_before_additional_discount := discounted_shoe_price + shirt_total_price in
  let total_additional_discount := total_price_before_additional_discount * additional_discount / 100 in
  let final_price := total_price_before_additional_discount - total_additional_discount in
  final_price = 285 :=
sorry

end erwan_total_expenditure_l772_772591


namespace power_equation_l772_772928

theorem power_equation (m : ℤ) (h : 16 = 2 ^ 4) : (16 : ℝ) ^ (3 / 4) = (2 : ℝ) ^ (m : ℝ) → m = 3 := by
  intros
  sorry

end power_equation_l772_772928


namespace factorial_divisible_by_power_of_3_l772_772284

theorem factorial_divisible_by_power_of_3 : 
  (∃ k : ℕ, k = 10 ∧ (3^k ∣ (25!))) :=
sorry

end factorial_divisible_by_power_of_3_l772_772284


namespace triangle_abo_is_right_isosceles_l772_772330

-- Define the polar to rectangular conversion function
def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

-- Define the given points in polar coordinates
def point_A_polar := (-2 : ℝ, -Real.pi / 2)
def point_B_polar := (Real.sqrt 2, 3 * Real.pi / 4)
def point_O := (0 : ℝ, 0 : ℝ)

-- Convert the points from polar to rectangular coordinates
def point_A_rect := polar_to_rect point_A_polar.1 point_A_polar.2
def point_B_rect := polar_to_rect point_B_polar.1 point_B_polar.2

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define the distances
def AC := distance point_A_rect point_O
def AB := distance point_A_rect point_B_rect
def BC := distance point_B_rect point_O

-- The main theorem statement
theorem triangle_abo_is_right_isosceles :
  AC ^ 2 = AB ^ 2 + BC ^ 2 ∧ 
  AB = BC ∧ 
  ∃ (angle_ABC : ℝ), angle_ABC = Real.pi / 2 :=
by
  -- Proof goes here
  sorry

end triangle_abo_is_right_isosceles_l772_772330


namespace find_sets_l772_772213

theorem find_sets (A B : Set ℕ) :
  A ∩ B = {1, 2, 3} ∧ A ∪ B = {1, 2, 3, 4, 5} →
    (A = {1, 2, 3} ∧ B = {1, 2, 3, 4, 5}) ∨
    (A = {1, 2, 3, 4, 5} ∧ B = {1, 2, 3}) ∨
    (A = {1, 2, 3, 4} ∧ B = {1, 2, 3, 5}) ∨
    (A = {1, 2, 3, 5} ∧ B = {1, 2, 3, 4}) :=
by
  sorry

end find_sets_l772_772213


namespace repeating_decimal_to_fraction_l772_772191

theorem repeating_decimal_to_fraction :
  (0.3 + 0.206) = (5057 / 9990) :=
sorry

end repeating_decimal_to_fraction_l772_772191


namespace ad_minus_bc_divisible_by_2017_l772_772874

theorem ad_minus_bc_divisible_by_2017 
  (a b c d n : ℕ) 
  (h1 : (a * n + b) % 2017 = 0) 
  (h2 : (c * n + d) % 2017 = 0) : 
  (a * d - b * c) % 2017 = 0 :=
sorry

end ad_minus_bc_divisible_by_2017_l772_772874


namespace max_x_minus_y_l772_772649

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l772_772649


namespace greatest_divisor_l772_772599

theorem greatest_divisor (d : ℕ) (h1 : 4351 % d = 8) (h2 : 5161 % d = 10) : d = 1 :=
by
  -- Proof goes here
  sorry

end greatest_divisor_l772_772599


namespace max_x_minus_y_l772_772639

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772639


namespace star_polygon_angle_condition_l772_772009

theorem star_polygon_angle_condition (n : ℕ) (α β : ℝ) 
  (h1 : α = β - 15)  -h2 : n > 2 :
  n = 24 :=
by 
  have ext_angle_sum := 360
  have exts_contributed := n * α + n * β
  have eqn := n * (β - α) = 360
  rw [h1] at eqn
  linarith

end star_polygon_angle_condition_l772_772009


namespace probability_at_least_four_same_value_l772_772607

theorem probability_at_least_four_same_value : 
  let dice := 5
  let faces := 6
  let target_probability := 13 / 648
  at_least_four_same_value dice faces = target_probability :=
sorry

end probability_at_least_four_same_value_l772_772607


namespace exradii_sum_l772_772413

-- Definitions of the exradii, circumradius, and inradius
variables {a b c r R ra rb rc : ℝ}
variables {s : ℝ} -- semi-perimeter of the triangle

-- Conditions
def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

-- Main theorem
theorem exradii_sum (r_a r_b r_c R r : ℝ) :
  r_a + r_b + r_c = 4 * R + r :=
sorry

end exradii_sum_l772_772413


namespace find_f_100_l772_772246

noncomputable def f (x : ℝ) := sorry
noncomputable def g (x : ℝ) := sorry

theorem find_f_100 :
  (f : ℝ → ℝ) ∧ (g : ℝ → ℝ) → 
  (∀ x : ℝ, (f (g x) = x) ∧ (g (f x) = x)) →
  f 10 = 1 →
  f 100 = 2 := 
by
  intros h_inv h_point
  sorry

end find_f_100_l772_772246


namespace sequence_property_l772_772949

variable (u : ℕ → ℝ)

-- Conditions of the problem
def condition1 := u 1 = 1 / 2
def condition2 (n : ℕ) := ∑ i in Finset.range (n + 1), u (i + 1) = (n + 1)^2 * u (n + 1)

-- Theorem to prove
theorem sequence_property (n : ℕ) (h1 : condition1 u) (h2 : ∀ n, condition2 u n) :
  u n = 1 / (n * (n + 1)) :=
begin
  sorry
end

end sequence_property_l772_772949


namespace find_xy_l772_772578

noncomputable def positive_real_numbers (x y : ℝ) :=
  x > 0 ∧ y > 0

theorem find_xy (x y : ℝ) (h1 : positive_real_numbers x y) (h2 : x^2 + y^2 = 2) (h3 : x^4 + y^4 = 15 / 8) :
  x * y = real.sqrt (17) / 4 := 
by
  sorry

end find_xy_l772_772578


namespace circumcircle_passes_midpoint_l772_772779

noncomputable def midpoint (A B : ℝ) : ℝ := (A + B) / 2

theorem circumcircle_passes_midpoint (A B C A1 B1 C1 P Q : ℝ) 
  (h_abc_acute : ∠A + ∠B + ∠C = 180 ∧ ∠A < 90 ∧ ∠B < 90 ∧ ∠C < 90)
  (h_ab_lt_ac : AB < AC)
  (h_ac_lt_bc : AC < BC)
  (h_a1 : A1 = foot_of_altitude A B C)
  (h_b1 : B1 = foot_of_altitude B A C)
  (h_c1 : C1 = foot_of_altitude C A B)
  (h_p : P = reflect_over_line C1 BB1)
  (h_q : Q = reflect_over_line B1 CC1) :
  let F := midpoint B C in
  concyclic A1 P Q F :=
sorry

end circumcircle_passes_midpoint_l772_772779


namespace correct_statements_02_and_03_l772_772965

-- Definitions for conditions in the problem
def f1 (x : ℝ) : ℝ := if x = 0 then 1 else 1
def g (x : ℝ) : ℝ := 1

def f2 (x : ℝ) : ℝ := 1

def f3 (x : ℝ) : ℝ := x
def f4 (t : ℝ) : ℝ := t

def same_domain_and_range (f g : ℝ → ℝ) : Prop :=
  (∀ x, f x = g x) 

-- Problem statement
theorem correct_statements_02_and_03 : (f1 = g ∧ f2 1 = f2 0 ∧ same_domain_and_range f3 f4) :=
by
  -- Statement ①: f1 equals g
  have h1 : f1 ≠ g := by sorry,
  
  -- Statement ②: Given f2, then f2 1 equals f2 0
  have h2 : f2 1 = f2 0 := by sorry,

  -- Statement ③: f3 and f4 have the same expression and domain
  have h3 : same_domain_and_range f3 f4 := by sorry,

  -- Statement ④: f3 and f4 are not the same as they have different expressions
  have h4 : ¬ same_domain_and_range (λ x, x) (λ x, x + 1) := by sorry,
  
  -- Combining all correct statements
  exact ⟨h2, h3⟩ 

end correct_statements_02_and_03_l772_772965


namespace diagonal_bisector_l772_772805

-- Definitions for points, lines, and their relationships.
variables {Point Line : Type}

-- Define lines e and f intersecting at points A, B, C, and D.
variables (e f : Line) (A B C D : Point)

-- Condition 1: Intersection at specific points.
axiom intersect_e : ∃ (O : Point), A ≠ B ∧ C ≠ D ∧ (A ∈ e ∧ B ∈ e) ∧ (C ∈ f ∧ D ∈ f)

-- Condition 2: Equal lengths of segments AB and CD.
axiom equal_segments : dist A B = dist C D

-- Condition 3: Drawing lines parallel to segments AC and BD.
axiom parallel_through_B : ∃ (g : Line), (B ∈ g) ∧ ∀ (P Q : Point), (P ∈ g ∧ Q ∈ g) → (dist P Q = dist A C)
axiom parallel_through_D : ∃ (h : Line), (D ∈ h) ∧ ∀ (P Q : Point), (P ∈ h ∧ Q ∈ h) → (dist P Q = dist A C)
axiom parallel_through_A : ∃ (i : Line), (A ∈ i) ∧ ∀ (P Q : Point), (P ∈ i ∧ Q ∈ i) → (dist P Q = dist B D)
axiom parallel_through_C : ∃ (j : Line), (C ∈ j) ∧ ∀ (P Q : Point), (P ∈ j ∧ Q ∈ j) → (dist P Q = dist B D)

-- Definition of M and N as the intersections of these parallel lines.
variables (M N : Point)
axiom M_def : (∃ (a d : Line), (B ∈ a) ∧ (D ∈ d) ∧ (∀ (P Q : Point), (P ∈ a ∧ Q ∈ a) → (dist P Q = dist A C)) ∧ (∀ (P Q : Point), (P ∈ d ∧ Q ∈ d) → (dist P Q = dist A C)) ∧ (M ∈ a ∧ M ∈ d))
axiom N_def : (∃ (b c : Line), (A ∈ c) ∧ (C ∈ b) ∧ (∀ (P Q : Point), (P ∈ c ∧ Q ∈ c) → (dist P Q = dist B D)) ∧ (∀ (P Q : Point), (P ∈ b ∧ Q ∈ b) → (dist P Q = dist B D)) ∧ (N ∈ b ∧ N ∈ c))

-- Theorem to prove: One of the diagonals MN lies on the angle bisector of the angles formed by e and f.
theorem diagonal_bisector : ∃ (l m : Line), 
  (M ∈ l ∧ N ∈ l ∧ l ⊆ (angle_bisector e f)) ∨ (M ∈ m ∧ N ∈ m ∧ m ⊆ (angle_bisector e f)) :=
sorry -- Proof not required.

end diagonal_bisector_l772_772805


namespace sine_translation_equivalence_l772_772866

theorem sine_translation_equivalence :
  ∀ x, 3 * sin (x - (π / 3) + (π / 3)) = 3 * sin x :=
by
  intro x
  simp
  sorry

end sine_translation_equivalence_l772_772866


namespace max_value_x_minus_y_l772_772687

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l772_772687


namespace meaningful_fraction_condition_l772_772077

theorem meaningful_fraction_condition (x : ℝ) : (x - 1 ≠ 0) → (x ≠ 1) :=
by
  intro h
  exact h

end meaningful_fraction_condition_l772_772077


namespace point_A_location_fuel_consumption_times_passed_gas_station_l772_772937

-- Define the travel records as a list of integers
def travelRecords : List ℤ := [+10, -8, +6, -13, +7, -12, +3, -1]

-- Define the fuel consumption rate
def fuelConsumptionRate : ℝ := 0.05

-- Define the location of the gas station relative to the guard post
def gasStationLocation : ℤ := 6

-- Prove that point A is located 8 kilometers west of the guard post
theorem point_A_location :
  let finalPosition := travelRecords.sum
  finalPosition = -8 :=
by
  sorry

-- Prove the fuel consumption during the patrol
theorem fuel_consumption :
  let totalDistance := travelRecords.map (Int.natAbs ∘ Int.toNat).sum
  totalDistance = 60 ∧ (totalDistance:ℝ) * fuelConsumptionRate = 3 :=
by
  sorry

-- Prove the number of times the police officer passed the gas station
theorem times_passed_gas_station :
  let positions := travelRecords.scanl (· + ·) 0
  positions.filter (λ x, x = gasStationLocation).length = 4 :=
by
  sorry

end point_A_location_fuel_consumption_times_passed_gas_station_l772_772937


namespace num_correct_propositions_l772_772861

theorem num_correct_propositions (t : ℝ) :
  (1 < t ∧ t < 4 → ¬(4 - t = t - 1)) ∧ -- Proposition ① is incorrect
  ((t < 1 ∨ t > 4) → (4 - t) * (t - 1) < 0) ∧ -- Proposition ② is correct
  (∀t, ∃x y, x^2/(4-t) + y^2/(t-1) = 1 → ¬(4 - t = t - 1)) ∧ -- Proposition ③ is incorrect
  (1 < t ∧ t < 5/2 → 4 - t > t - 1 ∧ t - 1 > 0) -- Proposition ④ is correct
  → 2 := -- The number of correct propositions is 2
sorry

end num_correct_propositions_l772_772861


namespace union_of_sets_l772_772266

def M : Set Int := { -1, 0, 1 }
def N : Set Int := { 0, 1, 2 }

theorem union_of_sets : M ∪ N = { -1, 0, 1, 2 } := by
  sorry

end union_of_sets_l772_772266


namespace max_area_of_region_tangent_to_line_l772_772305

noncomputable def max_area_of_region (radii : List ℕ) : ℝ :=
  let areas := radii.map (fun r => Float.pi * r ^ 2)
  areas.sum - 16 * Float.pi -- subtract overlap area

theorem max_area_of_region_tangent_to_line :
  max_area_of_region [2, 4, 6, 8, 10] = 184 * Float.pi :=
by
  sorry

end max_area_of_region_tangent_to_line_l772_772305


namespace sin_identity_l772_772620

noncomputable def θ : Real := sorry  -- θ is an unspecified angle in this context.

axiom tan_condition : Real.tan (θ + Real.pi / 12) = 2

theorem sin_identity : sin(Real.pi / 3 - 2 * θ) = -3 / 5 := by
  -- This is where the proof would go
  sorry

end sin_identity_l772_772620


namespace planes_formed_by_pairwise_parallel_lines_l772_772758

-- Definition of lines being pairwise parallel
def pairwise_parallel (l1 l2 l3 : affine_plane) : Prop :=
  (l1 ∥ l2) ∧ (l2 ∥ l3) ∧ (l1 ∥ l3)

-- Number of planes that can be determined by any two of the three lines
def num_planes_determined (l1 l2 l3 : affine_plane) : ℕ :=
  if l1 = l2 ∧ l2 = l3 then 1 else 3

-- Statement to prove
theorem planes_formed_by_pairwise_parallel_lines (l1 l2 l3 : affine_plane)
  (h : pairwise_parallel l1 l2 l3) :
  num_planes_determined l1 l2 l3 = 1 ∨ num_planes_determined l1 l2 l3 = 3 :=
by
  sorry

end planes_formed_by_pairwise_parallel_lines_l772_772758


namespace sqrt_subtraction_specific_x_expression_l772_772930

-- Define the first problem
theorem sqrt_subtraction (h : Real.sqrt 9 - Real.sqrt 2 * 32 * 62 = 1) : True := by
  sorry

-- Define the second problem
theorem specific_x_expression (h : ∀ (x : ℝ), x + x⁻¹ = 3 → 0 < x → x^((3 : ℝ) / (2 : ℝ)) + x^(-(3 : ℝ) / (2 : ℝ)) = 2 * Real.sqrt 5) : True := by
  sorry

end sqrt_subtraction_specific_x_expression_l772_772930


namespace max_x_minus_y_l772_772650

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l772_772650


namespace max_value_x_minus_y_l772_772690

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l772_772690


namespace probability_product_divisible_by_5_approx_0_488_l772_772885

def num_integers : ℕ := 2020
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

noncomputable def probability_divisible_by_5 (p : ℝ) : Prop :=
  ∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ num_integers ∧ 1 ≤ y ∧ y ≤ num_integers ∧ 1 ≤ z ∧ z ≤ num_integers ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  (divisible_by_5 x ∨ divisible_by_5 y ∨ divisible_by_5 z) ∧
  p ≈ 0.488

theorem probability_product_divisible_by_5_approx_0_488 : 
  probability_divisible_by_5 0.488 :=
sorry

end probability_product_divisible_by_5_approx_0_488_l772_772885


namespace sin_pi_over_3_minus_2theta_l772_772622

theorem sin_pi_over_3_minus_2theta (θ : ℝ) (h : tan (θ + π / 12) = 2) :
  sin (π / 3 - 2 * θ) = -3 / 5 :=
by
  sorry

end sin_pi_over_3_minus_2theta_l772_772622


namespace integer_solution_l772_772087

theorem integer_solution (n : ℤ) (h1 : n + 15 > 16) (h2 : -3 * n > -9) : n = 2 :=
by
  sorry

end integer_solution_l772_772087


namespace sum_of_values_x_l772_772605

-- Define the problem conditions
def cos_squared (θ : ℝ) : ℝ := real.cos θ ^ 2
def in_degrees (θ : ℝ) : Prop := 150 < θ ∧ θ < 250

-- Define the main statement of the problem
theorem sum_of_values_x (x : ℝ) (hx : in_degrees x) :
  (cos_squared (3 * real.to_radians x) + cos_squared (7 * real.to_radians x)) =
  6 * (cos_squared (4 * real.to_radians x) * cos_squared (2 * real.to_radians x)) →
  ∃ x1 x2, in_degrees x1 ∧ in_degrees x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 420 :=
sorry

end sum_of_values_x_l772_772605


namespace opposite_2024_eq_neg_2024_l772_772451

def opposite (n : ℤ) : ℤ := -n

theorem opposite_2024_eq_neg_2024 : opposite 2024 = -2024 :=
by
  sorry

end opposite_2024_eq_neg_2024_l772_772451


namespace max_primes_arithmetic_sequence_l772_772356

theorem max_primes_arithmetic_sequence (d : ℕ) (h : d = 12) : ∃ n, n ≤ 5 ∧ 
  ∀ (s : ℕ → ℕ), (∀ m, s m is_prime) ∧ (∀ k m, s k = s m + k * d) → n = 5 :=
begin
  sorry,
end

end max_primes_arithmetic_sequence_l772_772356


namespace triangle_side_length_l772_772342

theorem triangle_side_length (A B C : Type*) [inner_product_space ℝ (A × B)]
  (AB AC BC : ℝ) (h1 : AB = 2)
  (h2 : AC = 3) (h3 : BC = sqrt(5.2)) :
  ACS = sqrt(5.2) :=
sorry

end triangle_side_length_l772_772342


namespace smallest_b_value_l772_772850

theorem smallest_b_value (a b : ℕ) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : b = 3 := sorry

end smallest_b_value_l772_772850


namespace least_area_of_triangle_DEF_l772_772463

noncomputable theory

open Complex

def area_of_triangle_DEF : ℂ → ℂ := λ z,
  let s := 2 * (2 * sqrt 3) * sin(π/12) in
  (1/2) * s^2 * sin(π/6)

theorem least_area_of_triangle_DEF :
  let z := (2 * sqrt 3 : ℂ) * exp(2 * 0 * π * I / 12) in
  let z₁ := (2 * sqrt 3 : ℂ) * exp(2 * 1 * π * I / 12) in
  let z₂ := (2 * sqrt 3 : ℂ) * exp(2 * 2 * π * I / 12) in
  area_of_triangle_DEF z = (12 * sin (π/12)^2 * sin (π/6)) / 2 :=
by
  sorry

end least_area_of_triangle_DEF_l772_772463


namespace find_tan_Z_l772_772349

def cot (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem find_tan_Z (X Y Z : ℝ)
  (hXZ : cot X * cot Z = 2)
  (hYZ : cot Y * cot Z = 1 / 32)
  (sum_angles : X + Y + Z = Real.pi) :
  Real.tan Z = 8 + 5 * Real.sqrt 19 :=
by
  sorry

end find_tan_Z_l772_772349


namespace zucchini_pounds_l772_772830

theorem zucchini_pounds :
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let quarts := 4
  let cost_per_quart := 10.00
  let total_cost := quarts * cost_per_quart
  let cost_of_eggplants := eggplants_pounds * eggplants_cost_per_pound
  let cost_of_tomatoes := tomatoes_pounds * tomatoes_cost_per_pound
  let cost_of_onions := onions_pounds * onions_cost_per_pound
  let cost_of_basil := basil_pounds * (basil_cost_per_half_pound * 2)
  let other_ingredients_cost := cost_of_eggplants + cost_of_tomatoes + cost_of_onions + cost_of_basil
  let cost_of_zucchini := total_cost - other_ingredients_cost
  let zucchini_cost_per_pound := 2.00
  let pounds_of_zucchini := cost_of_zucchini / zucchini_cost_per_pound
  pounds_of_zucchini = 4 :=
by
  sorry

end zucchini_pounds_l772_772830


namespace dihedral_angle_greater_than_90_l772_772434

theorem dihedral_angle_greater_than_90 
  (a b : ℝ) 
  (h_perp : b ≠ 0) 
  (h_non_neg : a > 0 ∧ b > 0) : 
  ∀ θ, θ ∈ dihedral_angle (plane_through_points (0, a, -b) (a, a, -b)) (plane_through_points (a, 0, -b) (a, a, -b)) → θ > π / 2 :=
sorry

end dihedral_angle_greater_than_90_l772_772434


namespace symmetric_points_l772_772286

-- Let points P and Q be symmetric about the origin
variables (m n : ℤ)
axiom symmetry_condition : (m, 4) = (- (-2), -n)

theorem symmetric_points :
  m = 2 ∧ n = -4 := 
  by {
    sorry
  }

end symmetric_points_l772_772286


namespace percentage_discount_l772_772358

theorem percentage_discount
  (P : ℝ) (N : ℝ) (total_price_after_tax : ℝ) (D : ℝ)
  (h1 : P = 45)
  (h2 : N = 10)
  (h3 : total_price_after_tax = 396)
  : D = 20 :=
by
  -- Define the discounted price per pair
  let discounted_price := P - (D / 100 * P)

  -- The total cost before tax
  let total_cost := N * discounted_price

  -- The total cost after 10% tax is applied
  let total_cost_after_tax := total_cost * 1.10

  -- Given that the total cost after tax equals 396
  have h4 : total_cost_after_tax = 396 := h3
  
  -- Now we will prove that D = 20
  sorry

end percentage_discount_l772_772358


namespace square_perimeter_l772_772317

theorem square_perimeter (d : ℝ) (h : d = 2 * real.sqrt 2) : ∃ p, p = 8 :=
by
  sorry

end square_perimeter_l772_772317


namespace isosceles_triangle_of_conditions_l772_772373

theorem isosceles_triangle_of_conditions (A B C P : Point) 
  (h1 : inside_triangle P A B C)
  (h2 : ∠ P B A = 10)
  (h3 : ∠ B A P = 20)
  (h4 : ∠ P C B = 30)
  (h5 : ∠ C B P = 40) : 
  length A B = length A C := 
sorry

end isosceles_triangle_of_conditions_l772_772373


namespace bases_for_final_digit_one_l772_772992

noncomputable def numberOfBases : ℕ :=
  (Finset.filter (λ b => ((625 - 1) % b = 0)) (Finset.range 11)).card - 
  (Finset.filter (λ b => b ≤ 2) (Finset.range 11)).card

theorem bases_for_final_digit_one : numberOfBases = 4 :=
by sorry

end bases_for_final_digit_one_l772_772992


namespace parallelogram_height_l772_772767

variable (base height area : ℝ)
variable (h_eq_diag : base = 30)
variable (h_eq_area : area = 600)

theorem parallelogram_height :
  (height = 20) ↔ (base * height = area) :=
by
  sorry

end parallelogram_height_l772_772767


namespace magnitude_of_c_l772_772741

def vector_a : ℝ × ℝ := (1, -1)
def vector_b : ℝ × ℝ := (2, 1)
def vector_c : ℝ × ℝ := (2 * (1, -1).1 + (2, 1).1, 2 * (1, -1).2 + (2, 1).2)

def magnitude (v : ℝ × ℝ) : ℝ := 
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_c : magnitude vector_c = real.sqrt 17 := by
  sorry

end magnitude_of_c_l772_772741


namespace range_of_a_l772_772794

theorem range_of_a (a : ℝ) (h1 : 1 < a) :
  (∀ x : ℝ, x ∈ set.Icc a (2 * a) →
    (∃ y : ℝ, y ∈ set.Icc a (a^2) ∧ real.log x / real.log a + real.log y / real.log a = 3)) → 
  2 ≤ a :=
by
  sorry

end range_of_a_l772_772794


namespace max_x_minus_y_l772_772669

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772669


namespace expectedDistanceFromCenterOfCircle_l772_772945

noncomputable def radius : ℝ := 1
noncomputable def expected_distance (R : ℝ) : ℝ :=
  ∫ x in Icc (0 : ℝ) R, 2 * x^2

theorem expectedDistanceFromCenterOfCircle : expected_distance radius = 2 / 3 :=
  sorry

end expectedDistanceFromCenterOfCircle_l772_772945


namespace december_25_is_thursday_l772_772997

theorem december_25_is_thursday (thanksgiving : ℕ) (h : thanksgiving = 27) :
  (∀ n, n % 7 = 0 → n + thanksgiving = 25 → n / 7 = 4) :=
by
  sorry

end december_25_is_thursday_l772_772997


namespace min_reciprocal_sum_l772_772241

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : log 2 3 = (log 2 x + log 2 y) / 2) : (1/x + 1/y) ≥ 2/3 :=
sorry

end min_reciprocal_sum_l772_772241


namespace sqrt_three_irrational_and_in_range_l772_772910

theorem sqrt_three_irrational_and_in_range : irrational (sqrt 3) ∧ 0 < sqrt 3 ∧ sqrt 3 < 3 := 
by 
  sorry

end sqrt_three_irrational_and_in_range_l772_772910


namespace betty_additional_money_needed_l772_772155

def wallet_cost : ℝ := 100
def betty_savings : ℝ := wallet_cost / 2
def parents_contribution : ℝ := 15
def grandparents_contribution : ℝ := 2 * parents_contribution

def total_money : ℝ := betty_savings + parents_contribution + grandparents_contribution
def amount_needed : ℝ := wallet_cost - total_money

theorem betty_additional_money_needed : amount_needed = 5 := by
  sorry

end betty_additional_money_needed_l772_772155


namespace sum_totient_equals_n_l772_772011

open Nat

theorem sum_totient_equals_n (n : ℕ) : n = ∑ d in (finset.divisors n).val, φ d := 
sorry

end sum_totient_equals_n_l772_772011


namespace line_AP_cuts_CD_at_midpoint_l772_772382

-- Convex pentagon and its properties
variables (A B C D E P M : Type)
variables [geometry : geometry A B C D E P M]

-- Angles in the pentagon
variables (∠BAC = ∠CAD) (∠CAD = ∠DAE)
variables (∠CBA = ∠DCA) (∠DCA = ∠EDA)

-- Intersection point
axiom intersection (BD CE : Type) : P = line.intersection BD CE

-- Line AP cuts segment CD at midpoint M
theorem line_AP_cuts_CD_at_midpoint :
  (convex A B C D E) ∧ (∠BAC = ∠CAD ∧ ∠CAD = ∠DAE) ∧ (∠CBA = ∠DCA ∧ ∠DCA = ∠EDA) →
  line (AP) ∩ segment (CD) = M →
  midpoint M C D :=
by sorry

end line_AP_cuts_CD_at_midpoint_l772_772382


namespace smaller_hexagon_area_fraction_l772_772868

theorem smaller_hexagon_area_fraction 
  (ABCDEF : Hexagon) 
  (H1 : regular_hexagon ABCDEF) 
  (H2 : mid_hexagon ABCDEF)
  : area (mid_hexagon ABCDEF) = (3/4) * area ABCDEF := 
sorry

end smaller_hexagon_area_fraction_l772_772868


namespace distance_calculation_correct_l772_772168

-- Define the necessary vectors
def a := ⟨3, -2⟩ : ℝ × ℝ
def b := ⟨4, -6⟩ : ℝ × ℝ
def d := ⟨2, -5⟩ : ℝ × ℝ

-- Function to calculate the distance between the two given parallel lines
noncomputable def distance_parallel_lines : ℝ :=
  let v := (a.1 - b.1, a.2 - b.2) in
  let dot_vd := (v.1 * d.1 + v.2 * d.2) in
  let dot_dd := (d.1 * d.1 + d.2 * d.2) in
  let projection := ( (dot_vd / dot_dd) * d.1, (dot_vd / dot_dd) * d.2 ) in
  let c := (b.1 + projection.1, b.2 + projection.2) in
  real.sqrt ((a.1 - c.1)^2 + (a.2 - c.2)^2)

theorem distance_calculation_correct :
  distance_parallel_lines = 31 / 29 :=
sorry

end distance_calculation_correct_l772_772168


namespace workshop_technicians_l772_772037

/-- The average salary of all the workers in a workshop is Rs. 8000. The average 
salary of some technicians is Rs. 20000 and the average salary of the rest is 
Rs. 6000. The total number of workers in the workshop is 49. -/
theorem workshop_technicians 
  (average_salary_all : ℕ)
  (average_salary_technicians : ℕ)
  (average_salary_rest : ℕ)
  (total_workers : ℕ)
  (total_average_salary : average_salary_all = 8000)
  (technicians_average_salary : average_salary_technicians = 20000)
  (rest_average_salary : average_salary_rest = 6000)
  (total_number_of_workers : total_workers = 49) :
  ∃ (T : ℕ), 
  let R := total_workers - T in
  average_salary_technicians * T + average_salary_rest * R = average_salary_all * total_workers 
  ∧ T = 7 :=
by
  obtain ⟨T, R⟩ := exists_unique (@exists_unique_of_exists_of_unique (λ T, (average_salary_technicians * T + average_salary_rest * (total_workers - T) = average_salary_all * total_workers ∧ T = 7)))
  sorry

end workshop_technicians_l772_772037


namespace largest_number_l772_772096

def number_A : ℝ := 7.25678
noncomputable def number_B : ℝ := 7.256777777777...
noncomputable def number_C : ℝ := 7.257676767676...
noncomputable def number_D : ℝ := 7.275675675675...
noncomputable def number_E : ℝ := 7.275627562756...

theorem largest_number : number_C > number_A ∧ number_C > number_B ∧ number_C > number_D ∧ number_C > number_E :=
by
  sorry

end largest_number_l772_772096


namespace triangle_cos_X_and_XZ_l772_772350

-- Define the triangle context with the given conditions
structure Triangle :=
  (X Y Z : Type)
  (YZ : ℝ)
  (sin_Z : ℝ)

-- Given conditions for the problem
axiom angle_Y_right (T : Triangle) : T.Y = 90
axiom sin_Z_value (T : Triangle) : T.sin_Z = 3 / 5
axiom YZ_value (T : Triangle) : T.YZ = 10

-- Define the Lean theorem to prove the correct answers
theorem triangle_cos_X_and_XZ (T : Triangle) 
  (hY : angle_Y_right T)
  (hsinZ : sin_Z_value T)
  (hYZ : YZ_value T) : 
  (cos 90 - (asin (T.sin_Z)) = 3 / 5) ∧ 
  (T.YZ * T.sin_Z = 6) := 
  by 
  sorry

end triangle_cos_X_and_XZ_l772_772350


namespace am_gm_inequality_example_am_gm_inequality_equality_condition_l772_772370

theorem am_gm_inequality_example (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x + y^3) * (x^3 + y) ≥ 4 * x^2 * y^2 :=
sorry

theorem am_gm_inequality_equality_condition (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  ((x + y^3) * (x^3 + y) = 4 * x^2 * y^2) ↔ (x = 0 ∧ y = 0 ∨ x = 1 ∧ y = 1) :=
sorry

end am_gm_inequality_example_am_gm_inequality_equality_condition_l772_772370


namespace problem1_problem2_l772_772512

-- Problem 1: Prove the given mathematical expression evaluates to -1 + √3
theorem problem1 : (- (1 / 2)) ^ (-1:ℤ) + real.sqrt 2 * real.sqrt 6 - (real.pi - 3) ^ (0:ℤ) + abs (real.sqrt 3 - 2) = -1 + real.sqrt 3 :=
by
  sorry

-- Problem 2: Prove the given fraction simplifies to x / (x-1) for the given conditions
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) : (x^2 - 1) / (x + 1) / ((x^2 - 2 * x + 1) / (x^2 - x)) = x / (x - 1) :=
by
  sorry

end problem1_problem2_l772_772512


namespace maximum_value_of_x_minus_y_l772_772703

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l772_772703


namespace ratio_DE_EC_l772_772135

-- Define the rhombus ABCD with given conditions
def rhombus (A B C D : Point) : Prop := 
  angle A B D = 60 * (π / 180) ∧ 
  distance A B = distance A D ∧ 
  distance A D = distance C D ∧
  distance B C = distance A D

-- Define conditions for pyramid
variables 
  (A B C D S E : Point)
  (a : ℝ)

-- Define specific distances
def pyramid_conditions : Prop :=
  rhombus A B C D ∧
  distance S A = distance S C ∧
  distance S D = distance S B ∧ 
  distance S B = a
  
-- Define the ratio condition that needs to be proved
theorem ratio_DE_EC (h : pyramid_conditions A B C D S E a) :
  distance D E / distance E C = 2 / 3 :=
sorry

end ratio_DE_EC_l772_772135


namespace trapezium_parallel_side_length_l772_772203

theorem trapezium_parallel_side_length (a h area x : ℝ) (h1 : a = 20) (h2 : h = 15) (h3 : area = 285) :
  area = 1/2 * (a + x) * h → x = 18 :=
by
  -- placeholder for the proof
  sorry

end trapezium_parallel_side_length_l772_772203


namespace usual_time_is_20_l772_772508

variable (R T : ℝ) -- Usual rate and usual time

-- Given conditions
variables (rate_incr : R / (5/4 * R) = T / (T - 4))

-- Prove that the usual time is 20 minutes
theorem usual_time_is_20 (h : rate_incr) : T = 20 :=
by
  sorry

end usual_time_is_20_l772_772508


namespace new_area_of_parallelogram_l772_772570

-- Definitions for the problem
def base := 20 -- base of the parallelogram in feet
def height := 4 -- height of the parallelogram in feet
def area_of_removed_square := 4 -- area of the square to be removed in square feet

-- Statement to prove
theorem new_area_of_parallelogram : 
  let initial_area := base * height,
      new_area := initial_area - area_of_removed_square 
  in new_area = 76 := 
by 
  -- Initial area calculation
  let initial_area := base * height,
  -- New area calculation
  let new_area := initial_area - area_of_removed_square 
  show new_area = 76 from 
    sorry

end new_area_of_parallelogram_l772_772570


namespace number_of_pens_each_student_gets_l772_772071

theorem number_of_pens_each_student_gets 
    (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ)
    (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) :
  (total_pens / Nat.gcd total_pens total_pencils) = 11 :=
by
  sorry

end number_of_pens_each_student_gets_l772_772071


namespace angle_bisector_square_l772_772826

-- Define the required entities and their properties
variable {A B C M K : Type}
variables [Metric_Space A] [Metric_Space B] [Metric_Space C] [Metric_Space M] [Metric_Space K]

-- Definitions of points A, B, C, M and triangle ABC with angle bisector AM
def is_triangle (A B C : Type) := true
def is_angle_bisector (A B C M : Type) := true

-- Define distances
variables {AB AM MB AC MC AK : ℝ}

-- Define the conditions
axiom distance_AB_AC: AB = dist A B * dist A C
axiom distance_AM_MB_MC: AB * AC = AM * AM + MB * MC

-- Lean theorem statement
theorem angle_bisector_square (A B C M K : Type) [Metric_Space A] [Metric_Space B] [Metric_Space C] [Metric_Space M] [Metric_Space K] 
  (h_triangle : is_triangle A B C) (h_bisector : is_angle_bisector A B C M)
  (dist_AB_AC : AB = dist A B * dist A C) (dist_AM_MB_MC : AB * AC = AM * AM + MB * MC) : 
  AM * AM = AB * AC - MB * MC := 
by {
  sorry 
}

end angle_bisector_square_l772_772826


namespace problem_c_neither_odd_nor_even_l772_772964

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_neither_odd_nor_even (f : ℝ → ℝ) : Prop := ¬ is_odd f ∧ ¬ is_even f

theorem problem_c_neither_odd_nor_even : is_neither_odd_nor_even (λ x : ℝ, x + Real.cos x) := sorry

end problem_c_neither_odd_nor_even_l772_772964


namespace fraction_initially_filled_l772_772954

theorem fraction_initially_filled (x : ℕ) :
  ∃ (x : ℕ), 
    x + 15 + (15 + 5) = 100 ∧ 
    (x : ℚ) / 100 = 13 / 20 :=
by
  sorry

end fraction_initially_filled_l772_772954


namespace square_perimeter_l772_772316

theorem square_perimeter (d : ℝ) (h : d = 2 * real.sqrt 2) : ∃ p, p = 8 :=
by
  sorry

end square_perimeter_l772_772316


namespace cos_165_is_correct_l772_772929

noncomputable def cos_165_eq : Prop :=
  cos (Real.pi * (165 / 180)) = -((Real.sqrt 6 + Real.sqrt 2) / 4)

theorem cos_165_is_correct : cos_165_eq := by
  sorry

end cos_165_is_correct_l772_772929


namespace isosceles_triangle_sides_l772_772291

theorem isosceles_triangle_sides (a b c : ℕ) (h₁ : a + b + c = 10) (h₂ : (a = b ∨ b = c ∨ a = c)) 
  (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) : 
  (a = 3 ∧ b = 3 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 4) := 
by
  sorry

end isosceles_triangle_sides_l772_772291


namespace sum_of_wins_in_tournament_l772_772769

theorem sum_of_wins_in_tournament :
  let n := 15
  let total_matches := n * (n-1) / 2
  (∀ i, 1 ≤ i ∧ i ≤ n → ℕ) →
  ∑ i in finset.range n, (λ i, ℕ) i = total_matches :=
by sorry

end sum_of_wins_in_tournament_l772_772769


namespace asymptotes_of_hyperbola_l772_772243

open Real

variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (c : ℝ)
variable (focus : c = sqrt (a^2 + b^2))
variable (reflection_condition : ∃ x₀ y₀ : ℝ, (x₀, y₀) = (-c, 0) ∧ (y₀ = 1/3 * (x₀ + c)))

noncomputable def equation_of_asymptotes (a b : ℝ) : String :=
  "y = ± (" ++ toString (sqrt (6) / 2) ++ ") x"

theorem asymptotes_of_hyperbola 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (c : ℝ) 
  (focus : c = sqrt (a^2 + b^2))
  (reflection_condition : ∃ x₀ y₀ : ℝ, (x₀, y₀) = (-c, 0) ∧ (y₀ = 1/3 * (x₀ + c))) : 
  equation_of_asymptotes a b = "y = ± " ++ toString (sqrt (6) / 2) ++ " x" := 
  sorry

end asymptotes_of_hyperbola_l772_772243


namespace number_of_male_students_l772_772036

noncomputable def avg_all : ℝ := 90
noncomputable def avg_male : ℝ := 84
noncomputable def avg_female : ℝ := 92
noncomputable def count_female : ℕ := 24

theorem number_of_male_students (M : ℕ) (T : ℕ) :
  avg_all * (M + count_female) = avg_male * M + avg_female * count_female →
  T = M + count_female →
  M = 8 :=
by
  intro h_avg h_count
  sorry

end number_of_male_students_l772_772036


namespace nat_numbers_equal_if_divisible_l772_772001

theorem nat_numbers_equal_if_divisible
  (a b : ℕ)
  (h : ∀ n : ℕ, ∃ m : ℕ, n ≠ m → (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end nat_numbers_equal_if_divisible_l772_772001


namespace geometric_sequence_common_ratio_l772_772300

noncomputable def common_ratio (a : ℕ → ℝ) (q : ℝ) (positive : ∀ n, a n > 0) : Prop :=
  a 2 + a 1 = a 1 * q^2 ∧ q = (1 + Real.sqrt 5) / 2

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (positive : ∀ n, a n > 0)
  (geometric : ∀ n, a (n + 1) = a n * q)
  (arithmetic : a 2 + a 1 = 2 * (a 3 / 2)) : 
  q = (1 + Real.sqrt 5) / 2 :=
by {
  sorry
}

end geometric_sequence_common_ratio_l772_772300


namespace angle_trisection_l772_772534

open_locale classical
open_locale big_operators

-- Definitions and assumptions for the problem
variables {C : Type*} [circle C] 
variables {K I N M L P Q : point C}

-- Conditions given in the problem
axiom outside_circle (hC : C K)
axiom tangents_from_point (hKI : tangent K C I) (hKN : tangent K C N)
axiom point_on_extension (hM : line K N M)
axiom circumcircle_intersects (hP : (circumcircle K L M) ∩ C = {P})
axiom foot_perpendicular (hQ : perpendicular_from_point N L M Q)

-- Theorem statement: Prove that ∠MPQ = 2∠KML
theorem angle_trisection (h : conditions) : ∠MPQ = 2 * ∠KML :=
sorry

end angle_trisection_l772_772534


namespace point_in_second_quadrant_l772_772055

def i : ℂ := complex.I
def z : ℂ := i + i^2
def point : ℝ × ℝ := (z.re, z.im)

theorem point_in_second_quadrant (z_eq : i + i^2 = z) : 
  (-1, 1) = point → 
  point.1 < 0 ∧ point.2 > 0 :=
by {
  -- proof goes here
  sorry
}

end point_in_second_quadrant_l772_772055


namespace total_eggs_l772_772882

theorem total_eggs (students : ℕ) (eggs_per_student : ℕ) (h1 : students = 7) (h2 : eggs_per_student = 8) :
  students * eggs_per_student = 56 :=
by
  sorry

end total_eggs_l772_772882


namespace least_possible_b_l772_772848

theorem least_possible_b (a b : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (h_a_factors : ∃ k, a = p^k ∧ k + 1 = 3) (h_b_factors : ∃ m, b = p^m ∧ m + 1 = a) (h_divisible : b % a = 0) : 
  b = 8 := 
by 
  sorry

end least_possible_b_l772_772848


namespace maximum_value_of_x_minus_y_l772_772710

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l772_772710


namespace find_m_value_l772_772626

theorem find_m_value : 
  ∃ m : ℝ, (∀ x y : ℝ, (x ^ 2 + y ^ 2 + 2 * x - 4 * y = 0) → (x, y) = (-1, 2))
  ∧ (3 * (-1: ℝ) + 2 + m = 0) :=
by
  use 1
  split
  sorry

end find_m_value_l772_772626


namespace final_result_l772_772118

noncomputable def given_conditions 
  (R : ℝ) (d : ℝ) (a : ℝ) (b : ℝ) (h1 : a > b) : 
  ∃ circle O, 
  ∃ l : affine_subspace ℝ point, 
  ∃ H : point,
  line l ∧ -- l is a line
  circle.center = 0 ∧  -- Assuming circle center at origin for simplicity
  dist circle.center l = d ∧ -- Distance from center to line l
  ∃ P Q : point, 
    dist P H = a ∧
    dist Q H = b ∧
  2008 ∃ chords : (Σ i : fin 2008, segment),
  is_chord_parallel_to_line chords.2 l ∧
  divides_in_equal_segments chords.2 CD

theorem final_result 
  (R d a b : ℝ) 
  (h1 : a > b)
  (h_cond : given_conditions R d a b h1) :
  (1 / 2008) * (∑ i in range 1 2009, 
    let P_A_i := dist P (A_i) in
    let P_B_i := dist P (B_i) in
    let Q_A_i := dist Q (A_i) in
    let Q_B_i := dist Q (B_i) in
    P_A_i^2 + P_B_i^2 + Q_A_i^2 + Q_B_i^2)
  = 2 * a^2 + 2 * b^2 + 4 * d^2 + 4 * R^2 := 
sorry

end final_result_l772_772118


namespace shaded_square_area_l772_772123

theorem shaded_square_area (a b s : ℝ) (h : a * b = 40) :
  ∃ s, s^2 = 2500 / 441 :=
by
  sorry

end shaded_square_area_l772_772123


namespace triangle_area_ADJ_l772_772819

noncomputable def parallelogram_area := 48
noncomputable def midpoint (a b p: ℝ × ℝ) : Prop := p = (a + b) / 2
noncomputable def point_j (c d j: ℝ × ℝ) : Prop := (j - c).norm = (j - d).norm

theorem triangle_area_ADJ
  (A B C D P Q J: ℝ × ℝ)
  (H_parallelogram : parallelogram_area = 48)
  (H_midpoints_AB : midpoint A B P)
  (H_midpoints_CD : midpoint C D Q)
  (H_J : point_j C D J):
  let area_ADJ := area_of_triangle A D J in 
  area_ADJ = 36 :=
sorry

end triangle_area_ADJ_l772_772819


namespace water_formation_l772_772197

noncomputable def NaHCO3_reaction := "NaHCO3" + "HC2H3O2" → "NaC2H3O2" + "CO2" + "H2O"

def moles_NaHCO3 : ℕ := 3
def moles_HC2H3O2 : ℕ := 3

theorem water_formation (hn : moles_NaHCO3 = 3) (ha : moles_HC2H3O2 = 3) :
  ∃ h2o_moles : ℕ, h2o_moles = 3 :=
by {
  use 3,
  sorry
}

end water_formation_l772_772197


namespace simplify_factorial_expression_l772_772835

theorem simplify_factorial_expression : (13.factorial / (10.factorial + 3 * 9.factorial) = 1320) :=
by
  sorry

end simplify_factorial_expression_l772_772835


namespace max_x_minus_y_l772_772662

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l772_772662


namespace trapezium_side_length_l772_772198

theorem trapezium_side_length (a b h A x : ℝ) 
  (ha : a = 20) (hh : h = 15) (hA : A = 285) 
  (h_formula : A = 1 / 2 * (a + b) * h) : 
  b = 18 :=
by
  sorry

end trapezium_side_length_l772_772198


namespace maximum_value_of_x_minus_y_l772_772702

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l772_772702


namespace triangle_median_equal_bc_l772_772334

-- Let \( ABC \) be a triangle, \( AB = 2 \), \( AC = 3 \), and the median from \( A \) to \( BC \) has the same length as \( BC \).
theorem triangle_median_equal_bc (A B C M : Type) (AB AC BC AM : ℝ) 
  (hAB : AB = 2) (hAC : AC = 3) 
  (hMedian : BC = AM) (hM : M = midpoint B C) :
  BC = real.sqrt (26 / 5) :=
by sorry

end triangle_median_equal_bc_l772_772334


namespace proof_problem_l772_772237

variables {a : ℕ → ℕ} {b : ℕ → ℕ} {S : ℕ → ℕ}

-- Given conditions
def a_condition := a 2 = 2
def line_condition := ∃ x y, (a 4 = x) ∧ (a 6 = y) ∧ x + 2 * y - 16 = 0

-- Questions converted to proof problems
def general_term_a (n : ℕ) : Prop := a n = n
def sum_b (n : ℕ) : Prop := S n = (n * (n + 1)) / 2 + 2^(n + 1) - 2

theorem proof_problem (n : ℕ) : 
  a_condition → line_condition → (general_term_a n) ∧ (sum_b n) :=
by sorry

end proof_problem_l772_772237


namespace question1_question1_odd_question2_question3_l772_772814

/-- Define f(m, n) as the absolute difference between the areas of black and white regions
    within the given right-angled triangle with legs m and n lying along the edges of unit squares. -/
def f (m n : ℕ) : ℚ :=
  let S1 (m n : ℕ) : ℚ := sorry -- area of black regions (implementation omitted)
  let S2 (m n : ℕ) : ℚ := sorry -- area of white regions (implementation omitted)
  abs (S1 m n - S2 m n)

/-- 1. Proving f(m, n) = 0 if m, n are both even, 
        and f(m, n) = 1/2 if m, n are both odd -/
theorem question1 (m n : ℕ) (h1 : even m) (h2 : even n) : f m n = 0 := sorry

theorem question1_odd (m n : ℕ) (h1 : odd m) (h2 : odd n) : f m n = 1 / 2 := sorry

/-- 2. Proving f(m, n) ≤ 1/2 max {m, n} for all m and n -/
theorem question2 (m n : ℕ) : f m n ≤ (1 / 2 : ℚ) * max m n := sorry

/-- 3. Proving that there does not exist a constant c such that f(m, n) < c for all m and n -/
theorem question3 (c : ℚ) : ¬(∀ m n : ℕ, f m n < c) := sorry

end question1_question1_odd_question2_question3_l772_772814


namespace cosine_angle_between_diagonals_of_prism_l772_772950

noncomputable def cosine_angle_between_diagonals_prism (a V : ℝ) : ℝ :=
  V^2 / (V^2 + a^6)

theorem cosine_angle_between_diagonals_of_prism (a V : ℝ) (hV : V > 0) (ha : a > 0) :
  let c := cosine_angle_between_diagonals_prism a V in
  c = V^2 / (V^2 + a^6) :=
by
  sorry

end cosine_angle_between_diagonals_of_prism_l772_772950


namespace right_triangle_acute_angle_ratio_l772_772308

theorem right_triangle_acute_angle_ratio (A B : ℝ) (h_ratio : A / B = 5 / 4) (h_sum : A + B = 90) :
  min A B = 40 :=
by
  -- Conditions are provided
  sorry

end right_triangle_acute_angle_ratio_l772_772308


namespace relationship_among_a_b_and_c_l772_772799

def a : ℝ := 2 ^ 0.3
def b : ℝ := 0.3 ^ 2
def c : ℝ := Real.log 0.3 / Real.log 2

theorem relationship_among_a_b_and_c : c < b ∧ b < a := 
by 
  sorry

end relationship_among_a_b_and_c_l772_772799


namespace finite_good_set_either_Snr_or_4_elements_l772_772085

noncomputable def is_good_set (S : set ℝ) : Prop :=
∀ x y ∈ S, (x + y) ∈ S ∨ (|x - y|) ∈ S

noncomputable def S (n : ℕ) (r : ℝ) : set ℝ :=
{0} ∪ (set.image (λ k, k * r) (finset.range (n + 1)))

theorem finite_good_set_either_Snr_or_4_elements (A : set ℝ) :
  set.finite A → A ≠ {0} → is_good_set A →
  (∃ n : ℕ, ∃ r : ℝ, r > 0 ∧ A = S n r) ∨ (A.to_finset.card = 4) :=
sorry

end finite_good_set_either_Snr_or_4_elements_l772_772085


namespace coffee_bags_per_week_l772_772996

def bags_morning : Nat := 3
def bags_afternoon : Nat := 3 * bags_morning
def bags_evening : Nat := 2 * bags_morning
def bags_per_day : Nat := bags_morning + bags_afternoon + bags_evening
def days_per_week : Nat := 7

theorem coffee_bags_per_week : bags_per_day * days_per_week = 126 := by
  sorry

end coffee_bags_per_week_l772_772996


namespace find_second_number_l772_772433

theorem find_second_number (x : ℝ) (h : (20 + x + 60) / 3 = (10 + 70 + 16) / 3 + 8) : x = 40 :=
sorry

end find_second_number_l772_772433


namespace steers_cows_unique_solution_l772_772906

-- Definition of the problem
def steers_and_cows_problem (s c : ℕ) : Prop :=
  25 * s + 26 * c = 1000 ∧ s > 0 ∧ c > 0

-- The theorem statement to be proved
theorem steers_cows_unique_solution :
  ∃! (s c : ℕ), steers_and_cows_problem s c ∧ c > s :=
sorry

end steers_cows_unique_solution_l772_772906


namespace actual_percent_change_is_minus_ten_l772_772807

-- Define the last year's revenue as R (arbitrary positive value)
variable {R : ℝ} (hR : R > 0)

-- Define the projected revenue as 1.20 times last year's revenue
def projected_revenue := 1.20 * R

-- Define the actual revenue as 0.75 times the projected revenue
def actual_revenue := 0.75 * projected_revenue hR

-- Define the percent change function
def percent_change := ((actual_revenue hR - R) / R) * 100

-- State the theorem we aim to prove
theorem actual_percent_change_is_minus_ten :
  percent_change hR = -10 :=
by
  -- proof goes here
  sorry

end actual_percent_change_is_minus_ten_l772_772807


namespace hat_cost_l772_772895

theorem hat_cost (total_hats blue_hat_cost green_hat_cost green_hats : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_hat_cost = 6)
  (h3 : green_hat_cost = 7)
  (h4 : green_hats = 20) :
  (total_hats - green_hats) * blue_hat_cost + green_hats * green_hat_cost = 530 := 
by sorry

end hat_cost_l772_772895


namespace triangle_side_eq_median_l772_772340

theorem triangle_side_eq_median (A B C : Type) (a b c : ℝ) (hAB : a = 2) (hAC : b = 3) (hBC_eq_median : c = (2 * (Real.sqrt (13 / 10)))) :
  c = (Real.sqrt 130) / 5 := by
  sorry

end triangle_side_eq_median_l772_772340


namespace square_area_is_81_l772_772453

def square_perimeter (s : ℕ) : ℕ := 4 * s
def square_area (s : ℕ) : ℕ := s * s

theorem square_area_is_81 (s : ℕ) (h : square_perimeter s = 36) : square_area s = 81 :=
by {
  sorry
}

end square_area_is_81_l772_772453


namespace sum_of_numbers_l772_772326

theorem sum_of_numbers (a b : ℕ) (h : a + 4 * b = 30) : a + b = 12 :=
sorry

end sum_of_numbers_l772_772326


namespace sum_of_legs_of_larger_triangle_is_50_l772_772894

theorem sum_of_legs_of_larger_triangle_is_50 :
  ∀ (area_small area_large shorter_leg_small : ℝ),
    area_small = 8 → area_large = 200 → shorter_leg_small = 2 →
    let ratio := area_large / area_small;
        scale := Real.sqrt ratio;
        longer_leg_small := 8 / (1 / 2 * shorter_leg_small);
        shorter_leg_large := scale * shorter_leg_small;
        longer_leg_large := scale * longer_leg_small in
    shorter_leg_large + longer_leg_large = 50 :=
by
  intros area_small area_large shorter_leg_small h_area_small h_area_large h_shorter_leg_small
  simp [h_area_small, h_area_large, h_shorter_leg_small]
  let ratio := 200 / 8
  let scale := Real.sqrt ratio
  let longer_leg_small := 8 / (1 / 2 * 2)
  let shorter_leg_large := scale * 2
  let longer_leg_large := scale * longer_leg_small
  sorry

end sum_of_legs_of_larger_triangle_is_50_l772_772894


namespace parallelepiped_analogy_l772_772904

-- Define plane figures and the concept of analogy for a parallelepiped 
-- (specifically here as a parallelogram) in space
inductive PlaneFigure where
  | triangle
  | parallelogram
  | trapezoid
  | rectangle

open PlaneFigure

/-- 
  Given the properties and definitions of a parallelepiped and plane figures,
  we want to show that the appropriate analogy for a parallelepiped in space
  is a parallelogram.
-/
theorem parallelepiped_analogy : 
  (analogy : PlaneFigure) = parallelogram :=
sorry

end parallelepiped_analogy_l772_772904


namespace product_sin_eq_one_eighth_l772_772987

theorem product_sin_eq_one_eighth (h1 : Real.sin (3 * Real.pi / 8) = Real.cos (Real.pi / 8))
                                  (h2 : Real.sin (Real.pi / 8) = Real.cos (3 * Real.pi / 8)) :
  ((1 - Real.sin (Real.pi / 8)) * (1 - Real.sin (3 * Real.pi / 8)) * 
   (1 + Real.sin (Real.pi / 8)) * (1 + Real.sin (3 * Real.pi / 8)) = 1 / 8) :=
by {
  sorry
}

end product_sin_eq_one_eighth_l772_772987


namespace oz_words_lost_l772_772327

theorem oz_words_lost (letters : Fin 64) (forbidden_letter : Fin 64) (h_forbidden : forbidden_letter.val = 6) : 
  let one_letter_words := 64 
  let two_letter_words := 64 * 64
  let one_letter_lost := if letters = forbidden_letter then 1 else 0
  let two_letter_lost := (if letters = forbidden_letter then 64 else 0) + (if letters = forbidden_letter then 64 else 0) 
  1 + two_letter_lost = 129 :=
by
  sorry

end oz_words_lost_l772_772327


namespace trapezium_area_correct_l772_772597

def a : ℚ := 20  -- Length of the first parallel side
def b : ℚ := 18  -- Length of the second parallel side
def h : ℚ := 20  -- Distance (height) between the parallel sides

def trapezium_area (a b h : ℚ) : ℚ :=
  (1/2) * (a + b) * h

theorem trapezium_area_correct : trapezium_area a b h = 380 := 
  by
    sorry  -- Proof goes here

end trapezium_area_correct_l772_772597


namespace angle_Q_is_72_degrees_l772_772025

-- Define the context and conditions
def regular_decagon_angles_sum (n : ℕ) : ℕ := 180 * (n - 2)

def one_angle_of_regular_decagon (n : ℕ) := (regular_decagon_angles_sum n) / n

def reflex_angle (angle : ℕ) := 360 - angle

-- Define the problem as a theorem
theorem angle_Q_is_72_degrees :
  let n := 10 in 
  let angle_EFG := one_angle_of_regular_decagon n in 
  let angle_EFR := 180 - angle_EFG in
  let angle_RAJ := angle_EFR in 
  let reflex_angle_E := reflex_angle angle_EFG in 
  let angle_sum_quad_AEFQ := 360 in 
  angle_sum_quad_AEFQ - angle_RAJ - reflex_angle_E - angle_EFR = 72 := sorry

end angle_Q_is_72_degrees_l772_772025


namespace max_value_x_minus_y_l772_772689

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l772_772689


namespace betty_needs_more_money_l772_772156

-- Define the variables and conditions
def wallet_cost : ℕ := 100
def parents_gift : ℕ := 15
def grandparents_gift : ℕ := parents_gift * 2
def initial_betty_savings : ℕ := wallet_cost / 2
def total_savings : ℕ := initial_betty_savings + parents_gift + grandparents_gift

-- Prove that Betty needs 5 more dollars to buy the wallet
theorem betty_needs_more_money : total_savings + 5 = wallet_cost :=
by
  sorry

end betty_needs_more_money_l772_772156


namespace lines_through_point_tangent_to_parabola_l772_772584

theorem lines_through_point_tangent_to_parabola :
  let P := (0 : ℝ, 1 : ℝ)
  ∃ (L : list (ℝ × ℝ → Prop)), L.length = 3 ∧
  ∀ (l ∈ L), intersect_at_one_point l (λ (x y : ℝ), y^2 = 4 * x ∧ l (x, y))

/- 
Givens:
  - Point P at (0, 1)
  - Parabola y^2 = 4x
To Prove:
  - There are exactly 3 distinct lines intersecting the parabola at exactly one point
-/
:= sorry

end lines_through_point_tangent_to_parabola_l772_772584


namespace range_of_a_l772_772615

def A (a : ℝ) : Set ℝ := {x | x - 1 > a^2}
def B (a : ℝ) : Set ℝ := {x | x - 4 < 2a}

theorem range_of_a (a : ℝ) (h : A a ∩ B a ≠ ∅) : -1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l772_772615


namespace angle_BCO_l772_772889

theorem angle_BCO (A B C O : Type) [EuclideanGeometry] 
  (h_circle : IsCircumscribedAboutTriangle (triangle A B C) (circle O))
  (h_incenter : Incenter O (triangle A B C))
  (angle_BAC : angle BAC = 75)
  (angle_ACB : angle ACB = 55) :
  angle BCO = 25 :=
sorry

end angle_BCO_l772_772889


namespace sum_last_two_digits_l772_772900

theorem sum_last_two_digits (n : ℕ) (h1 : n = 20) : (9^n + 11^n) % 100 = 1 :=
by
  sorry

end sum_last_two_digits_l772_772900


namespace points_in_quadrants_l772_772872

theorem points_in_quadrants :
  ∀ (x y : ℝ), (y > 3 * x) → (y > 5 - 2 * x) → ((0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)) :=
by
  intros x y h1 h2
  sorry

end points_in_quadrants_l772_772872


namespace second_number_9th_group_l772_772546

structure EmployeeSampling where
  num_employees : ℕ
  groups : list (list ℕ)
  sampling_interval : ℕ

noncomputable def systematic_sampling (num_employees : ℕ) (num_groups : ℕ) : EmployeeSampling :=
  let interval := num_employees / num_groups
  let groups := (list.range num_employees).grouped interval
  ⟨num_employees, groups, interval⟩

theorem second_number_9th_group {e : EmployeeSampling}
  (h_num_employees : e.num_employees = 200)
  (h_groups : e.groups = list.range 200 |> list.map (list.range 5 ∘ (* 5)) |> list.map (list.map nat.succ))
  (h_sampling_interval : e.sampling_interval = 5)
  {second_number_5th_group : nat}
  (h_second_number_5th_group : second_number_5th_group = 22) :
  (e.groups.nth 8).iget.nth 1.iget = 42 := 
sorry

end second_number_9th_group_l772_772546


namespace max_x_minus_y_l772_772668

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772668


namespace abs_sum_floor_sin_l772_772177

theorem abs_sum_floor_sin :
  let S := (∑ k in Finset.range 360, Int.floor (2013 * Real.sin (k * Real.pi / 180)))
  in abs S = 178 :=
by
  let S := (∑ k in Finset.range 360, Int.floor (2013 * Real.sin (k * Real.pi / 180)),
  sorry

end abs_sum_floor_sin_l772_772177


namespace max_rooks_on_chessboard_l772_772602

theorem max_rooks_on_chessboard (n : ℕ) : ∃ M : ℕ, M = 4 * n ∧ 
  ∀ configuration : fin (3*n) → fin (3*n) → bool, 
  (∀ i j, configuration i j = tt → ∃! (k l : fin (3*n)), configuration k l = tt ∧ (i = k ∨ j = l)) → ∑ i j, if configuration i j = tt then 1 else 0 ≤ M :=
sorry

end max_rooks_on_chessboard_l772_772602


namespace axis_of_symmetry_and_vertex_l772_772856

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

theorem axis_of_symmetry_and_vertex :
  (∃ (a : ℝ), (f a = -2 * (a - 1)^2 + 3) ∧ a = 1) ∧ ∃ v, (v = (1, 3) ∧ ∀ x, f x = -2 * (x - 1)^2 + 3) :=
sorry

end axis_of_symmetry_and_vertex_l772_772856


namespace max_x_minus_y_l772_772651

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l772_772651


namespace angle_equiv_l772_772034

theorem angle_equiv (k : ℤ) : ∃ k : ℤ, -463 = k * 360 + 257 :=
by
  existsi -2
  norm_num
  sorry

end angle_equiv_l772_772034


namespace percentage_less_than_m_add_d_l772_772100

def symmetric_about_mean (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P m - x = P m + x

def within_one_stdev (P : ℝ → ℝ) (m d : ℝ) : Prop :=
  P m - d = 0.68 ∧ P m + d = 0.68

theorem percentage_less_than_m_add_d 
  (P : ℝ → ℝ) (m d : ℝ) 
  (symm : symmetric_about_mean P m)
  (within_stdev : within_one_stdev P m d) : 
  ∃ f, f = 0.84 :=
by
  sorry

end percentage_less_than_m_add_d_l772_772100


namespace range_of_m_l772_772628

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + (4 : ℝ)
noncomputable def g (x m : ℝ) : ℝ := x * (f x + m * x - 5)

theorem range_of_m (m : ℝ) :
  (∃ (h : (∀ x, f x = x^2 - 4 * x + 4) ∧ (∀ x, g x m = x * (f x + m * x - 5)) ∧ (¬is_monotone_on g (2 : ℝ) (3 : ℝ))), 
    (-1 / 3 : ℝ) < m ∧ m < (5 / 4 : ℝ)) :=
sorry

end range_of_m_l772_772628


namespace solve_problem_l772_772818

noncomputable def problem_statement : Prop :=
  ∀ (tons_to_pounds : ℕ) 
    (packet_weight_pounds : ℕ) 
    (packet_weight_ounces : ℕ)
    (num_packets : ℕ)
    (bag_capacity_tons : ℕ)
    (X : ℕ),
    tons_to_pounds = 2300 →
    packet_weight_pounds = 16 →
    packet_weight_ounces = 4 →
    num_packets = 1840 →
    bag_capacity_tons = 13 →
    X = (packet_weight_ounces * bag_capacity_tons * tons_to_pounds) / 
        ((bag_capacity_tons * tons_to_pounds) - (num_packets * packet_weight_pounds)) →
    X = 16

theorem solve_problem : problem_statement :=
by sorry

end solve_problem_l772_772818


namespace concentration_of_concentrated_kola_is_correct_l772_772934

noncomputable def concentration_of_concentrated_kola_added 
  (initial_volume : ℝ) (initial_pct_sugar : ℝ)
  (sugar_added : ℝ) (water_added : ℝ)
  (required_pct_sugar : ℝ) (new_sugar_volume : ℝ) : ℝ :=
  let initial_sugar := initial_volume * initial_pct_sugar / 100
  let total_sugar := initial_sugar + sugar_added
  let new_total_volume := initial_volume + sugar_added + water_added
  let total_volume_with_kola := new_total_volume + (new_sugar_volume / required_pct_sugar * 100 - total_sugar) / (100 / required_pct_sugar - 1)
  total_volume_with_kola - new_total_volume

noncomputable def problem_kola : ℝ :=
  concentration_of_concentrated_kola_added 340 7 3.2 10 7.5 27

theorem concentration_of_concentrated_kola_is_correct : 
  problem_kola = 6.8 :=
by
  unfold problem_kola concentration_of_concentrated_kola_added
  sorry

end concentration_of_concentrated_kola_is_correct_l772_772934


namespace bob_speed_before_construction_l772_772158

theorem bob_speed_before_construction:
  ∀ (v : ℝ),
    (1.5 * v + 2 * 45 = 180) →
    v = 60 :=
by
  intros v h
  sorry

end bob_speed_before_construction_l772_772158


namespace ratio_proof_l772_772901

-- Given conditions
variables (r y x : ℝ)
hypothesis (cond_r : r = 17.5)
hypothesis (cond_y : y = 10)
hypothesis (ratio : r / 1 = x / y)

-- Claim to prove
theorem ratio_proof : x = 175 :=
by
  have h₁ : r = 17.5 := cond_r
  have h₂ : y = 10 := cond_y
  have h₃ : r / 1 = x / y := ratio
  sorry

end ratio_proof_l772_772901


namespace diff_by_three_l772_772365

theorem diff_by_three (A : Finset ℕ) (hA_card : A.card = 16) (hA_range : ∀ x ∈ A, x ≤ 106) (hA_diff : ∀ {x y}, x ∈ A → y ∈ A → (x ≠ y) → (|x - y| ≠ 6) 
  ∧ (|x - y| ≠ 9) ∧ (|x - y| ≠ 12) ∧ (|x - y| ≠ 15) ∧ (|x - y| ≠ 18) ∧ (|x - y| ≠ 21)) : 
  ∃ x y ∈ A, x ≠ y ∧ |x - y| = 3 := 
sorry

end diff_by_three_l772_772365


namespace max_x_minus_y_l772_772643

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772643


namespace money_problem_l772_772142

theorem money_problem
  (A B C : ℕ)
  (h1 : A + B + C = 450)
  (h2 : B + C = 350)
  (h3 : C = 100) :
  A + C = 200 :=
by
  sorry

end money_problem_l772_772142


namespace statement_A_statement_B_statement_C_statement_D_final_problem_l772_772905

variables {a b c : Vector ℝ 2}

-- Statement A
theorem statement_A (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) 
  (h₄ : a ∥ b) (h₅ : b ∥ c) : a ∥ c :=
by sorry

-- Statement B
theorem statement_B (h₁ : a ≠ 0) (h₂ : a ∥ b) 
  : ∃! λ : ℝ, b = λ • a :=
by sorry

-- Statement C
theorem statement_C (h₁ : c ≠ 0) (h₂ : a ⬝ c = b ⬝ c) 
  : ¬ (a = b) :=
by sorry

-- Statement D
theorem statement_D (h₁ : ∥a - b∥ = 4) (h₂ : a ⬝ b = 1) 
  : ¬ (∥a + b∥ = sqrt 5) :=
by sorry

-- We combine the Theorems for the conclusive proof problem
theorem final_problem : 
  (statement_A a_ne_0 b_ne_0 c_ne_0 ab_parallel bc_parallel) ∧ 
  (statement_B a_ne_0 ab_parallel) ∧ 
  (statement_C c_ne_0 a_dot_c_eq_b_dot_c) ∧ 
  (statement_D a_minus_b_norm_eq_4 a_dot_b_eq_1) :=
by sorry

end statement_A_statement_B_statement_C_statement_D_final_problem_l772_772905


namespace impossible_tiling_of_convex_ngon_l772_772074

def convex_ngon (M : Type) (n : ℕ) : Prop :=
  convex_polygon M ∧ polygon_sides M = n

def parquet (tile : Type) : Prop :=
  ∀ (R : ℝ), ∃ (tiles : set tile), covers_circle_of_radius R tiles ∧ no_gaps tiles ∧ no_overlaps tiles

theorem impossible_tiling_of_convex_ngon (M : Type) (n : ℕ) (h_convex_ngon : convex_ngon M n) (h_n_ge_7 : n ≥ 7) :
  ¬ parquet M :=
by
  sorry

end impossible_tiling_of_convex_ngon_l772_772074


namespace no_integer_pair_2006_l772_772219

theorem no_integer_pair_2006 : ∀ (x y : ℤ), x^2 - y^2 ≠ 2006 := by
  sorry

end no_integer_pair_2006_l772_772219


namespace min_modulus_of_m_l772_772290

theorem min_modulus_of_m (m : ℂ) (x : ℝ) 
  (h : (1 + 2*complex.i) * x^2 + m * x + (1 - 2*complex.i) = 0) : |m| ≥ 2 :=
sorry

end min_modulus_of_m_l772_772290


namespace class7_E_student_count_l772_772143

variables (M F n : ℕ)
variables (total_students : ℕ)
variables (students_not_interested : ℕ)

-- Conditions
def condition1 : Prop := n = 20 / 100 * M
def condition2 : Prop := n = 25 / 100 * F
def condition3 : Prop := students_not_interested = 2
def condition4 : Prop := 20 < total_students ∧ total_students < 30

-- Definition of the total number of students
def total_students_def : Prop := total_students = (M + F - n) + students_not_interested

-- Problem: Prove the total number of students in the class is 26
theorem class7_E_student_count :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ total_students_def → total_students = 26 :=
by
  sorry

end class7_E_student_count_l772_772143


namespace oil_drop_probability_l772_772587

theorem oil_drop_probability :
  let r_circle := 1 -- radius of the circle in cm
  let side_square := 0.5 -- side length of the square in cm
  let area_circle := π * r_circle^2
  let area_square := side_square * side_square
  (area_square / area_circle) = 1 / (4 * π) :=
by
  sorry

end oil_drop_probability_l772_772587


namespace solve_equation_l772_772842

theorem solve_equation (x : ℝ) (h₁ : x ≠ -11) (h₂ : x ≠ -5) (h₃ : x ≠ -12) (h₄ : x ≠ -4) :
  (1 / (x + 11) + 1 / (x + 5) = 1 / (x + 12) + 1 / (x + 4)) ↔ x = -8 :=
by
  sorry

end solve_equation_l772_772842


namespace simplify_fraction_l772_772836

theorem simplify_fraction : 
  (13.factorial : ℚ) / (10.factorial + 3 * 9.factorial) = 44 / 3 := by
  sorry

end simplify_fraction_l772_772836


namespace find_constants_l772_772598

theorem find_constants
  (a_1 a_2 : ℚ)
  (h1 : 3 * a_1 - 3 * a_2 = 0)
  (h2 : 4 * a_1 + 7 * a_2 = 5) :
  a_1 = 5 / 11 ∧ a_2 = 5 / 11 :=
by
  sorry

end find_constants_l772_772598


namespace distinct_arrays_with_48_chairs_l772_772119

theorem distinct_arrays_with_48_chairs : 
  ∃ n : ℕ, n = 8 ∧ (
    ∀ (r c : ℕ), r * c = 48 → r ≥ 2 → c ≥ 2 → 
    (
      (r = 2 ∧ c = 24) ∨ 
      (r = 3 ∧ c = 16) ∨ 
      (r = 4 ∧ c = 12) ∨ 
      (r = 6 ∧ c = 8) ∨ 
      (r = 8 ∧ c = 6) ∨ 
      (r = 12 ∧ c = 4) ∨ 
      (r = 16 ∧ c = 3) ∨ 
      (r = 24 ∧ c = 2)
    )
  ) :=
begin
  -- proof will go here
  sorry
end

end distinct_arrays_with_48_chairs_l772_772119


namespace max_value_change_l772_772095

open Real

variables (f : ℝ → ℝ) (x : ℝ)

-- Conditions
def condition1 : Prop := ∀ f, (∃ M1 : ℝ, ∀ x : ℝ, f(x) ≤ M1 ∧ ∃ a, a + x ^ 2 = f ⟹ f(M1 + x ^ 2) - M1 = 27 / 2)
def condition2 : Prop := ∀ f, (∃ M2 : ℝ, ∀ x : ℝ, f(x) ≤ M2 ∧ ∃ b, b - 4 * x ^ 2 = f ⟹ f(M2 - 4 * x ^ 2) - M2 = -9)

-- Statement to prove
theorem max_value_change (f : ℝ → ℝ) 
  (h1 : condition1 f) 
  (h2 : condition2 f) : 
  ∃ C : ℝ, ∀ x : ℝ, C = - 27 / 4 ∧ ∃ c, c - 2 * x ^ 2 = f ⟹ f (C - 2 * x ^ 2) = f C :=
sorry

end max_value_change_l772_772095


namespace trapezium_parallel_side_length_l772_772202

theorem trapezium_parallel_side_length (a h area x : ℝ) (h1 : a = 20) (h2 : h = 15) (h3 : area = 285) :
  area = 1/2 * (a + x) * h → x = 18 :=
by
  -- placeholder for the proof
  sorry

end trapezium_parallel_side_length_l772_772202


namespace trapezium_other_side_length_l772_772205

theorem trapezium_other_side_length (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ) :
  side1 = 20 ∧ distance = 15 ∧ area = 285 → side2 = 18 :=
by
  intro h
  sorry

end trapezium_other_side_length_l772_772205


namespace num_valid_four_digit_numbers_l772_772745

theorem num_valid_four_digit_numbers : 
  {N : ℕ // 1000 ≤ N ∧ N < 10000 ∧ ∃ a x : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 100 ≤ x ∧ x < 1000 ∧ N = 1000 * a + x ∧ N = 5 * x} = 3 :=
by
  sorry

end num_valid_four_digit_numbers_l772_772745


namespace weight_conversion_l772_772067

theorem weight_conversion (kg_in_pounds : ℝ) (weight_in_kg : ℝ) :
  kg_in_pounds = 0.9072 / 2 →
  weight_in_kg = 350 →
  Int.round (weight_in_kg / kg_in_pounds) = 772 :=
by
  intro h1 h2
  sorry

end weight_conversion_l772_772067


namespace maximum_value_of_x_minus_y_l772_772699

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l772_772699


namespace remainder_1490_div_29_is_11_l772_772600

theorem remainder_1490_div_29_is_11 :
  ∃ (G R2 : ℕ), G = 29 ∧ (1255 % G = 8) ∧ (1490 % G = R2) ∧ R2 = 11 := 
by
  let G := 29
  let R2 := 11
  use G
  use R2
  split
  . rfl
  . split
    . exact Nat.mod_eq_of_lt (show 1255 % G = 8 by norm_num) (show 8 < 29 by norm_num)
    . split
      . exact Nat.mod_eq_of_lt (show 1490 % G = 11 by norm_num) (show 11 < 29 by norm_num)
      . rfl

end remainder_1490_div_29_is_11_l772_772600


namespace envelopes_initial_count_l772_772167

noncomputable def initialEnvelopes (given_per_friend : ℕ) (friends : ℕ) (left : ℕ) : ℕ :=
  given_per_friend * friends + left

theorem envelopes_initial_count
  (given_per_friend : ℕ) (friends : ℕ) (left : ℕ)
  (h_given_per_friend : given_per_friend = 3)
  (h_friends : friends = 5)
  (h_left : left = 22) :
  initialEnvelopes given_per_friend friends left = 37 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end envelopes_initial_count_l772_772167


namespace simplify_expression1_simplify_expression2_l772_772030

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D F : V)

-- Problem 1:
theorem simplify_expression1 : 
  (D - C) + (C - B) + (B - A) = D - A := 
sorry

-- Problem 2:
theorem simplify_expression2 : 
  (B - A) + (F - D) + (D - C) + (C - B) + (A - F) = 0 := 
sorry

end simplify_expression1_simplify_expression2_l772_772030


namespace trapezium_side_length_l772_772199

theorem trapezium_side_length (a b h A x : ℝ) 
  (ha : a = 20) (hh : h = 15) (hA : A = 285) 
  (h_formula : A = 1 / 2 * (a + b) * h) : 
  b = 18 :=
by
  sorry

end trapezium_side_length_l772_772199


namespace tank_overflow_time_l772_772106

theorem tank_overflow_time :
  let rateA := 2 -- tanks per hour
  let rateB := 1 -- tank per hour
  let t := 1 / rateA + 1 / rateB -- combined time for both pipes to fill the tank
  ∀ (t_open : ℝ), t_open = 15 / 60 → -- time when both pipes are open in hours
  (t_open * (rateA + rateB) + (1/4) * rateB = 1) → -- condition to fill the tank
  t_open + (15 / 60) = 1 / 2 → -- total time accounting for the last 15 minutes
  t_open = 15 / 60 -- time in hours when both pipes are open
  t_open + (15 / 60) = 1 / 2 :=
by
  sorry

end tank_overflow_time_l772_772106


namespace min_max_abs_poly_eq_zero_l772_772193

theorem min_max_abs_poly_eq_zero : 
  (∀y : ℝ, ∃x (hx : 0 ≤ x ∧ x ≤ 2), max (abs (x^2 - x * y + x)) = 0) :=
by sorry

end min_max_abs_poly_eq_zero_l772_772193


namespace side_length_of_square_land_l772_772914

theorem side_length_of_square_land (A : ℝ) (h : A = 625) (hsquare : ∃ s : ℝ, s^2 = A) : ∃ s : ℝ, s = 25 :=
by
  have hsquare' : ∃ s : ℝ, s^2 = 625 := by
    use 25
    norm_num
  assumption_mod_cast
  sorry

end side_length_of_square_land_l772_772914


namespace initial_buckets_correct_l772_772006

-- Define the conditions as variables
def total_buckets : ℝ := 9.8
def added_buckets : ℝ := 8.8
def initial_buckets : ℝ := total_buckets - added_buckets

-- The theorem to prove the initial amount of water is 1 bucket
theorem initial_buckets_correct : initial_buckets = 1 := 
by
  sorry

end initial_buckets_correct_l772_772006


namespace intersection_C1_C2_is_M_max_distance_A_B_C2_C3_l772_772777

noncomputable def curve_1 (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sin α ^ 2)

noncomputable def curve_2 (x y : ℝ) : Prop :=
  x + y + 1 = 0

noncomputable def curve_3 (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 1

theorem intersection_C1_C2_is_M :
  ∃ α : ℝ, curve_1 α = (-1, 0) ∧ curve_2 (-1) 0 :=
sorry

theorem max_distance_A_B_C2_C3 :
  ∀ A : ℝ × ℝ, ∀ B : ℝ × ℝ,
  (curve_2 A.1 A.2) → (curve_3 B.1 B.2) →
  ∥A - B∥ ≤ Real.sqrt 2 + 1 :=
sorry

end intersection_C1_C2_is_M_max_distance_A_B_C2_C3_l772_772777


namespace arc_length_of_unit_circle_with_30_degrees_angle_l772_772310

open Real

-- Define the conditions.
def unit_circle_radius : ℝ := 1
def central_angle_degrees : ℝ := 30
def central_angle_radians : ℝ := central_angle_degrees / 180 * π

-- Define the arc length formula in radians.
def arc_length (r : ℝ) (θ : ℝ) : ℝ := r * θ

-- The theorem we need to prove.
theorem arc_length_of_unit_circle_with_30_degrees_angle :
  arc_length unit_circle_radius central_angle_radians = π / 6 :=
by
  -- Proof goes here.
  sorry

end arc_length_of_unit_circle_with_30_degrees_angle_l772_772310


namespace product_of_coprime_numbers_l772_772455

variable {a b c : ℕ}

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem product_of_coprime_numbers (h1 : coprime a b) (h2 : a * b = c) : Nat.lcm a b = c := by
  sorry

end product_of_coprime_numbers_l772_772455


namespace find_divisor_l772_772505

theorem find_divisor (d : ℕ) : 15 = (d * 4) + 3 → d = 3 := by
  intros h
  have h1 : 15 - 3 = 4 * d := by
    linarith
  have h2 : 12 = 4 * d := by
    linarith
  have h3 : d = 3 := by
    linarith
  exact h3

end find_divisor_l772_772505


namespace erwan_total_spending_l772_772588

-- Definitions
def price_of_shoes : ℝ := 200
def discount_on_shoes : ℝ := 30 / 100
def price_of_shirt : ℝ := 80
def quantity_of_shirts : ℕ := 2
def additional_discount : ℝ := 5 / 100

-- Calculation of discounted prices
def discounted_price_shoes : ℝ :=
  price_of_shoes * (1 - discount_on_shoes)

def total_price_shirts : ℝ :=
  price_of_shirt * quantity_of_shirts

def initial_total_cost : ℝ :=
  discounted_price_shoes + total_price_shirts

def checkout_discount : ℝ :=
  initial_total_cost * additional_discount

def final_total_cost : ℝ :=
  initial_total_cost - checkout_discount

-- Proof statement
theorem erwan_total_spending : final_total_cost = 285 :=
by
  sorry

end erwan_total_spending_l772_772588


namespace part1_part2_l772_772743

variables {k : ℝ} {a b : ℝ} {x t : ℝ}

-- Given conditions
def norm_a (a : vector ℝ 3) : Prop := ∥a∥ = 1
def norm_b (b : vector ℝ 3) : Prop := ∥b∥ = 1
def cond_eq (a b : vector ℝ 3) (k : ℝ) : Prop :=
  ∥k • a + b∥ = real.sqrt 3 * ∥a - k • b∥
def f (k : ℝ) : ℝ := (k^2 + 1) / (4 * k)

-- Problem statement part 1
theorem part1 (a b : vector ℝ 3) (k : ℝ) (h1 : norm_a a) (h2 : norm_b b) (h3 : cond_eq a b k) (h4 : k > 0) :
  a ⬝ b = (k^2 + 1) / (4 * k) := sorry

-- Problem statement part 2
def inequality (x t k : ℝ) : ℝ :=
  x^2 - 2*t*x - 1/2

theorem part2 {x : ℝ} (h : ∀ k (t : ℝ), k > 0 → t ∈ set.Icc (-1:ℝ) 1 → f k ≥ inequality x t k) :
  1 - real.sqrt 2 ≤ x ∧ x ≤ real.sqrt 2 - 1 := sorry

end part1_part2_l772_772743


namespace boxes_with_nothing_l772_772399

theorem boxes_with_nothing (h_total : 15 = total_boxes)
    (h_pencils : 9 = pencil_boxes)
    (h_pens : 5 = pen_boxes)
    (h_both_pens_and_pencils : 3 = both_pen_and_pencil_boxes)
    (h_markers : 4 = marker_boxes)
    (h_both_markers_and_pencils : 2 = both_marker_and_pencil_boxes)
    (h_no_markers_and_pens : no_marker_and_pen_boxes = 0)
    (h_no_all_three_items : no_all_three_items = 0) :
    ∃ (neither_boxes : ℕ), neither_boxes = 2 :=
by
  sorry

end boxes_with_nothing_l772_772399


namespace sequence_is_increasing_l772_772235

theorem sequence_is_increasing :
  ∀ n m : ℕ, n < m → (1 - 2 / (n + 1) : ℝ) < (1 - 2 / (m + 1) : ℝ) :=
by
  intro n m hnm
  have : (2 : ℝ) / (n + 1) > 2 / (m + 1) :=
    sorry
  linarith [this]

end sequence_is_increasing_l772_772235


namespace odd_function_monotone_function_range_of_k_l772_772394

-- Conditions
axiom additivity (f : ℝ → ℝ) : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom positivity (f : ℝ → ℝ) : ∀ x : ℝ, x > 0 → f(x) > 0

open Real

-- Proof statements
theorem odd_function (f : ℝ → ℝ) [hf : ∀ x y : ℝ, f x + y = f x + f y]
  [hp : ∀ x : ℝ, x > 0 → f x > 0] : ∀ x : ℝ, f(-x) = -f(x) := by sorry

theorem monotone_function (f : ℝ → ℝ) [hf : ∀ x y : ℝ, f x + y = f x + f y] 
  [hp : ∀ x : ℝ, x > 0 → f(x) > 0] : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) := by sorry

theorem range_of_k (f : ℝ → ℝ) [hf : ∀ x y : ℝ, f x + y = f x + f y] 
  [hp : ∀ x : ℝ, x > 0 → f x > 0] : 
  (∀ x : ℝ, f(k * 3^x) + f(3^x - 9^x - 2) < 0) → k < 2 * sqrt 2 - 1 := by sorry

end odd_function_monotone_function_range_of_k_l772_772394


namespace exist_circle_tangent_to_three_circles_l772_772883

variable (h1 k1 r1 h2 k2 r2 h3 k3 r3 h k r : ℝ)

def condition1 : Prop := (h - h1)^2 + (k - k1)^2 = (r + r1)^2
def condition2 : Prop := (h - h2)^2 + (k - k2)^2 = (r + r2)^2
def condition3 : Prop := (h - h3)^2 + (k - k3)^2 = (r + r3)^2

theorem exist_circle_tangent_to_three_circles : 
  ∃ (h k r : ℝ), condition1 h1 k1 r1 h k r ∧ condition2 h2 k2 r2 h k r ∧ condition3 h3 k3 r3 h k r :=
by
  sorry

end exist_circle_tangent_to_three_circles_l772_772883


namespace quadrilateral_area_l772_772581

open Real

/-- Define vertex coordinates of the quadrilateral -/
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (4, 0)
def vertex3 : ℝ × ℝ := (4, 3)
def vertex4 : ℝ × ℝ := (2, 5)

/-- Calculate the area using Shoelace Theorem -/
def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (1 / 2) * abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

theorem quadrilateral_area : shoelace_area vertex1 vertex2 vertex3 vertex4 = 13 := by
  sorry

end quadrilateral_area_l772_772581


namespace spectators_count_l772_772299

theorem spectators_count (total_wristbands : ℕ) (wristbands_per_person : ℕ) (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : (total_wristbands / wristbands_per_person = 125) :=
by
  sorry

end spectators_count_l772_772299


namespace triangle_side_length_l772_772341

theorem triangle_side_length (A B C : Type*) [inner_product_space ℝ (A × B)]
  (AB AC BC : ℝ) (h1 : AB = 2)
  (h2 : AC = 3) (h3 : BC = sqrt(5.2)) :
  ACS = sqrt(5.2) :=
sorry

end triangle_side_length_l772_772341


namespace distance_A_B_l772_772571

-- Define the points A and B with their coordinates
def A : ℝ × ℝ := (-3, 1)
def B : ℝ × ℝ := (6, -4)

-- The distance function between two points in 2D space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The theorem stating the distance between points A and B
theorem distance_A_B : distance A B = real.sqrt 106 :=
  sorry

end distance_A_B_l772_772571


namespace speed_limit_inequality_l772_772947

theorem speed_limit_inequality (v : ℝ) : (v ≤ 40) :=
sorry

end speed_limit_inequality_l772_772947


namespace simplify_fraction_l772_772837

theorem simplify_fraction : 
  (13.factorial : ℚ) / (10.factorial + 3 * 9.factorial) = 44 / 3 := by
  sorry

end simplify_fraction_l772_772837


namespace part_a_part_b_part_c_l772_772918

-- Definition of log
def log2 (x : ℝ) : ℝ := if x > 0 then Real.log x / Real.log 2 else 0

-- Part (a) proof statement
theorem part_a : ∃ (a b : ℝ), log2 a * log2 b = log2 (a * b) := sorry

-- Part (b) proof statement
theorem part_b : ∃ (a b : ℝ), log2 a + log2 b = log2 (a + b) := sorry

-- Part (c) proof statement
theorem part_c : ¬∃ (a b : ℝ), (log2 a * log2 b = log2 (a * b)) ∧ (log2 a + log2 b = log2 (a + b)) := sorry

end part_a_part_b_part_c_l772_772918


namespace preston_high_school_teachers_l772_772822

theorem preston_high_school_teachers 
  (num_students : ℕ)
  (classes_per_student : ℕ)
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (teachers_per_class : ℕ)
  (H : num_students = 1500)
  (C : classes_per_student = 6)
  (T : classes_per_teacher = 5)
  (S : students_per_class = 30)
  (P : teachers_per_class = 1) : 
  (num_students * classes_per_student / students_per_class / classes_per_teacher = 60) :=
by sorry

end preston_high_school_teachers_l772_772822


namespace milk_production_l772_772849

variable (x y z : ℕ)
variable (h1 : ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0)
variable (initial_milk : ∀ x y z, (x : ℝ) * (y : ℝ) = (z : ℝ))

-- If the initial rate per cow per day is (y / (x * z))
-- and this rate increases by 10%
-- Then for 2x cows over 3z days, the total milk production
-- is expected to be 6.6 times y gallons.

theorem milk_production (x y z : ℕ) (h1 : ∀ x y z, (x > 0) ∧ (y > 0) ∧ (z > 0)) : 
  (2 * x) * (1.1 * (y / (x * z))) * (3 * z) = 6.6 * y := 
by
  sorry

end milk_production_l772_772849


namespace find_a_l772_772727

open Real

def ellipse (x y a : ℝ) : Prop := x^2 / 6 + y^2 / (a^2) = 1
def hyperbola (x y a : ℝ) : Prop := x^2 / a - y^2 / 4 = 1

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, ellipse x y a → hyperbola x y a → true) → a = 1 :=
by 
  sorry

end find_a_l772_772727


namespace find_a_l772_772251

noncomputable def f (x : ℝ) := Real.sin x
def tangent_slope := Real.cos (Real.pi / 3)
def line_slope (a : ℝ) := -a

theorem find_a (a : ℝ) (h_perpend : tangent_slope * line_slope a = -1) : a = 2 :=
by
  -- We need to justify the preconditions and conclude the proof here.
  sorry

end find_a_l772_772251


namespace expected_value_of_8_sided_die_l772_772122

noncomputable def expected_value_winnings_8_sided_die : ℚ :=
  let probabilities := [1, 1, 1, 1, 1, 1, 1, 1].map (λ x => (1 : ℚ) / 8)
  let winnings := [0, 0, 0, 0, 2, 4, 6, 8]
  let expected_value := (List.zipWith (*) probabilities winnings).sum
  expected_value

theorem expected_value_of_8_sided_die : expected_value_winnings_8_sided_die = 2.5 :=
by
  sorry

end expected_value_of_8_sided_die_l772_772122


namespace problem_l772_772255

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

def interval : set ℝ := set.Icc (3 * Real.pi / 4) Real.pi

theorem problem (x : ℝ) (h1 : x ∈ interval) : 0 < f'.deriv x :=
by
  sorry

end problem_l772_772255


namespace count_integer_solutions_l772_772275

theorem count_integer_solutions : 
  ∃ (s : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ s) ↔ (x^3 + y^2 = 2*y + 1)) ∧ 
  s.card = 3 := 
by
  sorry

end count_integer_solutions_l772_772275


namespace every_positive_integer_appears_in_sequence_l772_772459

/-- The sequence {a_n} is defined as follows:
  1. a_1 is any positive integer.
  2. For any integer n ≥ 1, a_{n+1} is the smallest positive integer that is coprime with
     the sum of the first n terms (∑_{i=1}^n a_i) and not equal to any of a_1, a_2, ..., a_n.
 
    We are to prove: every positive integer appears in the sequence {a_n}.
 -/
noncomputable def sequence (a : ℕ → ℕ) := 
  ∃ (a_1 : ℕ), 0 < a_1 ∧ 
  (∀ n ≥ 1, ∃ mn, mn = ((λ s, ∀ x > 0, (s < x ∧ ∀ m, m < n → a m ≠ x ∧ (Nat.gcd x s = 1))) ((finset.range n).sum a)) (finset.range n).sum a)

theorem every_positive_integer_appears_in_sequence :
  (∀ m : ℕ, ∃ n : ℕ, ∃ a : ℕ → ℕ, sequence a ∧ a n = m) :=
begin
  sorry
end

end every_positive_integer_appears_in_sequence_l772_772459


namespace fox_initial_coins_l772_772476

theorem fox_initial_coins :
  ∃ (x : ℕ), ∀ (c1 c2 c3 : ℕ),
    c1 = 3 * x - 50 ∧
    c2 = 3 * c1 - 50 ∧
    c3 = 3 * c2 - 50 ∧
    3 * c3 - 50 = 20 →
    x = 25 :=
by
  sorry

end fox_initial_coins_l772_772476


namespace option_D_correct_l772_772552

noncomputable def y1 (x : ℝ) : ℝ := 1 / x
noncomputable def y2 (x : ℝ) : ℝ := x^2
noncomputable def y3 (x : ℝ) : ℝ := (1 / 2)^x
noncomputable def y4 (x : ℝ) : ℝ := 1 / x^2

theorem option_D_correct :
  (∀ x : ℝ, y4 x = y4 (-x)) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → y4 x₁ > y4 x₂) :=
by
  sorry

end option_D_correct_l772_772552


namespace boarding_students_percentage_change_l772_772936

theorem boarding_students_percentage_change (x : ℕ) :
  let x2009 := x * 120 / 100 in
  let x2010 := x2009 * 80 / 100 in
  x2010 = x * 96 / 100 :=
by
  sorry

end boarding_students_percentage_change_l772_772936


namespace caps_difference_l772_772018

theorem caps_difference (Billie_caps Sammy_caps : ℕ) (Janine_caps := 3 * Billie_caps)
  (Billie_has : Billie_caps = 2) (Sammy_has : Sammy_caps = 8) :
  Sammy_caps - Janine_caps = 2 := by
  -- proof goes here
  sorry

end caps_difference_l772_772018


namespace trapezium_parallel_side_length_l772_772201

theorem trapezium_parallel_side_length (a h area x : ℝ) (h1 : a = 20) (h2 : h = 15) (h3 : area = 285) :
  area = 1/2 * (a + x) * h → x = 18 :=
by
  -- placeholder for the proof
  sorry

end trapezium_parallel_side_length_l772_772201


namespace selling_price_of_mixture_l772_772052

noncomputable def selling_price_per_pound (weight1 weight2 price1 price2 total_weight : ℝ) : ℝ :=
  (weight1 * price1 + weight2 * price2) / total_weight

theorem selling_price_of_mixture :
  selling_price_per_pound 20 10 2.95 3.10 30 = 3.00 :=
by
  -- Skipping the proof part
  sorry

end selling_price_of_mixture_l772_772052


namespace tompkins_routes_count_l772_772408

/-- 
There are four islands labeled A, B, C, and D.
Nine bridges span the canals between these islands.
The degrees of the nodes representing the islands are:
- Node A: 3 bridges
- Node B: 5 bridges
- Node C: 3 bridges
- Node D: 3 bridges
Tompkins must cross each bridge exactly once before reaching his destination.
Prove that the number of distinct routes Tompkins can take is 132.
-/
theorem tompkins_routes_count :
  let G := {A B C D : Type},
  let E := {⟨A, B⟩, ⟨A, C⟩, ⟨A, D⟩, ⟨B, C⟩, ⟨B, D⟩, ⟨C, D⟩},
  EulerianPath.count G E = 132 :=
sorry

end tompkins_routes_count_l772_772408


namespace angle_congruence_l772_772321

variables (A B C D E F P Q : Type)
variables [ConvexQuad A B C D] [OnSegment F A D] [OnSegment E B C]
variables [eq_ratios: (AF / FD) = (BE / EC) = (AB / CD)]
variables [ExtendIntersects EF P A B] [ExtendIntersects EF Q C D]

def equal_angles (X Y Z : Type) := ∠ X Y E = ∠ X Z E

theorem angle_congruence :
  equal_angles B P E C Q E :=
sorry

end angle_congruence_l772_772321


namespace range_of_a_l772_772040

noncomputable def f (x a : ℝ) : ℝ := log x + a

theorem range_of_a (x₀ a : ℝ) (h1 : 0 < x₀) (h2 : x₀ < 1) (h3 : 1 / x₀ = log x₀ + a) : a > 1 :=
by
  sorry

end range_of_a_l772_772040


namespace maximum_value_of_x_minus_y_l772_772701

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l772_772701


namespace diagonal_bisect_l772_772438

variables {A B C D O : Type}

-- Definitions and given conditions
def is_convex_quadrilateral (A B C D : Type) : Prop := sorry
def diagonals_intersect (A B C D O : Type) : Prop := sorry
def sum_areas_equal (A B C D O : Type) : Prop :=
  let area (X Y Z : Type) := sorry -- some area function
  area A O B + area C O D = area B O C + area D O A

-- The statement to prove
theorem diagonal_bisect (A B C D O : Type)
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_intersect A B C D O)
  (h3 : sum_areas_equal A B C D O) :
  (A = C) ∨ (B = D) :=
sorry

end diagonal_bisect_l772_772438


namespace select_4_non_coplanar_points_from_tetrahedron_l772_772450

theorem select_4_non_coplanar_points_from_tetrahedron :
  let points := 10
  let totalWays := Nat.choose points 4
  let sameFaceWays := 4 * Nat.choose 6 4
  let sameEdgeWay := 6
  let parallelogramWay := 3
  totalWays - sameFaceWays - sameEdgeWay - parallelogramWay = 141 :=
by sorry

end select_4_non_coplanar_points_from_tetrahedron_l772_772450


namespace triangle_side_length_l772_772347

theorem triangle_side_length (A B C M : Point)
  (hAB : dist A B = 2)
  (hAC : dist A C = 3)
  (hMidM : M = midpoint B C)
  (hAM_BC : dist A M = dist B C) :
  dist B C = Real.sqrt (78) / 3 :=
by
  sorry

end triangle_side_length_l772_772347


namespace number_of_coprime_integers_less_than_14_l772_772993

def coprime_with_14 (a : ℕ) : Prop :=
  Nat.gcd 14 a = 1

theorem number_of_coprime_integers_less_than_14 :
  {a : ℕ | a < 14 ∧ coprime_with_14 a}.to_finset.card = 6 :=
by
  sorry

end number_of_coprime_integers_less_than_14_l772_772993


namespace find_second_dimension_l772_772943

-- Definitions for the conditions given in the problem:
def length_square_cut : ℕ := 8
def other_dimension : ℕ := 52
def volume_box : ℕ := 5760

-- The mathematical statement we need to prove:
theorem find_second_dimension (w : ℕ) :
  (w - 2 * length_square_cut) * (other_dimension - 2 * length_square_cut) * length_square_cut = volume_box →
  w = 36 :=
begin
  sorry
end

end find_second_dimension_l772_772943


namespace range_of_alpha_range_of_x_plus_y_l772_772776

-- Definition for Question (I)
theorem range_of_alpha (α : ℝ) (h₁ : 0 ≤ α ∧ α < π)
  (P : ℝ×ℝ := (-1, 0))
  (h₂ : ∃ t, let x := -1 + t * Real.cos α,
               let y := t * Real.sin α,
               x^2 + y^2 - 6*x + 1 = 0) :
  α ∈ set.Icc 0 (π / 4) ∪ set.Icc (3 * π / 4) π :=
sorry

-- Definition for Question (II)
theorem range_of_x_plus_y (M : ℝ×ℝ)
  (h₁ : let x := M.1,
            let y := M.2,
            (x - 3)^2 + y^2 = 8) :
  M.1 + M.2 ∈ set.Icc (-1 : ℝ) 7 :=
sorry

end range_of_alpha_range_of_x_plus_y_l772_772776


namespace max_knights_cannot_be_all_liars_l772_772813

-- Define the conditions of the problem
structure Student :=
  (is_knight : Bool)
  (statement : String)

-- Define the function to check the truthfulness of statements
def is_truthful (s : Student) (conditions : List Student) : Bool :=
  -- Define how to check the statement based on conditions
  sorry

-- The maximum number of knights
theorem max_knights (N : ℕ) (students : List Student) (cond : ∀ s ∈ students, is_truthful s students = true ↔ s.is_knight) :
  ∃ M, M = N := by
  sorry

-- The school cannot be made up entirely of liars
theorem cannot_be_all_liars (N : ℕ) (students : List Student) (cond : ∀ s ∈ students, ¬is_truthful s students) :
  false := by
  sorry

end max_knights_cannot_be_all_liars_l772_772813


namespace square_perimeter_l772_772315

theorem square_perimeter (x : ℝ) (h : x * x + x * x = (2 * Real.sqrt 2) * (2 * Real.sqrt 2)) :
    4 * x = 8 :=
by
  sorry

end square_perimeter_l772_772315


namespace sum_of_arithmetic_sequence_is_constant_l772_772217

def is_constant (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, S n = c

theorem sum_of_arithmetic_sequence_is_constant
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 2 + a 6 + a 10 = a 1 + d + a 1 + 5 * d + a 1 + 9 * d)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  is_constant 11 a S :=
by
  sorry

end sum_of_arithmetic_sequence_is_constant_l772_772217


namespace max_value_x_minus_y_l772_772686

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l772_772686


namespace radius_ratio_l772_772125

theorem radius_ratio (V₁ V₂ : ℝ) (hV₁ : V₁ = 432 * Real.pi) (hV₂ : V₂ = 108 * Real.pi) : 
  (∃ (r₁ r₂ : ℝ), V₁ = (4/3) * Real.pi * r₁^3 ∧ V₂ = (4/3) * Real.pi * r₂^3) →
  ∃ k : ℝ, k = r₂ / r₁ ∧ k = 1 / 2^(2/3) := 
by
  sorry

end radius_ratio_l772_772125


namespace train_crossing_time_l772_772140

/-- 
Prove that the time it takes for a train traveling at 90 kmph with a length of 100.008 meters to cross a pole is 4.00032 seconds.
-/
theorem train_crossing_time (speed_kmph : ℝ) (length_meters : ℝ) : 
  speed_kmph = 90 → length_meters = 100.008 → (length_meters / (speed_kmph * (1000 / 3600))) = 4.00032 :=
by
  intros h1 h2
  sorry

end train_crossing_time_l772_772140


namespace max_difference_value_l772_772681

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l772_772681


namespace max_value_x_minus_y_l772_772685

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l772_772685


namespace sum_squares_l772_772466

theorem sum_squares (a b c : ℝ) (h1 : a + b + c = 22) (h2 : a * b + b * c + c * a = 116) : 
  (a^2 + b^2 + c^2 = 252) :=
by
  sorry

end sum_squares_l772_772466


namespace cyclic_quadrilateral_tangent_bisector_equivalence_l772_772133

theorem cyclic_quadrilateral_tangent_bisector_equivalence
  (A B C D : Point)
  (circ : Circle)
  (hA : OnCircle A circ) (hB : OnCircle B circ)
  (hC : OnCircle C circ) (hD : OnCircle D circ)
  (M N K L : Point)
  (hM : isTangentIntersection A C M)
  (hN : isTangentIntersection B D N)
  (hK : isAngleBisectorIntersection A C K)
  (hL : isAngleBisectorIntersection B D L)
  (lineBD : Line)
  (lineAC : Line)
  (hlineBD : onLine B lineBD ∧ onLine D lineBD)
  (hlineAC : onLine A lineAC ∧ onLine C lineAC)
  (lengths_eq : length A B * length C D = length A D * length B C) :
  (onLine M lineBD ↔ onLine N lineAC) ∧ 
  (onLine M lineBD ↔ onLine K lineBD) ∧
  (onLine M lineBD ↔ onLine L lineAC) := sorry

end cyclic_quadrilateral_tangent_bisector_equivalence_l772_772133


namespace perimeter_difference_l772_772576

-- Define the dimensions of the two figures
def width1 : ℕ := 6
def height1 : ℕ := 3
def width2 : ℕ := 6
def height2 : ℕ := 2

-- Define the perimeters of the two figures
def perimeter1 : ℕ := 2 * (width1 + height1)
def perimeter2 : ℕ := 2 * (width2 + height2)

-- Prove the positive difference in perimeters is 2 units
theorem perimeter_difference : (perimeter1 - perimeter2) = 2 := by
  sorry

end perimeter_difference_l772_772576


namespace find_polynomial_l772_772596

noncomputable def polynomial_condition (f : Polynomial ℤ) (M : ℕ) : Prop :=
  ∀ (n : ℕ), n ≥ M → ((f.eval (2^n) - 2^(f.eval (n))) / f.eval n) ∈ ℤ

theorem find_polynomial :
  ∃ f : Polynomial ℤ, Monic f ∧ ∃ M : ℕ, polynomial_condition f M
  ∧ (f = Polynomial.Coeff 1 1) :=
sorry

end find_polynomial_l772_772596


namespace smallest_intersection_value_l772_772443

theorem smallest_intersection_value (a b : ℝ) (f g : ℝ → ℝ)
    (Hf : ∀ x, f x = x^4 - 6 * x^3 + 11 * x^2 - 6 * x + a)
    (Hg : ∀ x, g x = x + b)
    (Hinter : ∀ x, f x = g x → true):
  ∃ x₀, x₀ = 0 :=
by
  intros
  -- Further steps would involve proving roots and conditions stated but omitted here.
  sorry

end smallest_intersection_value_l772_772443


namespace a_100_value_l772_772783

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0     => 0    -- using 0-index for convenience
| (n+1) => a n + 4

-- Prove the value of the 100th term in the sequence
theorem a_100_value : a 100 = 397 := 
by {
  -- proof would go here
  sorry
}

end a_100_value_l772_772783


namespace find_ellipse_eq_product_of_tangent_slopes_l772_772637

variables {a b : ℝ} {x y x0 y0 : ℝ}

-- Given conditions
def ellipse (a b : ℝ) := a > 0 ∧ b > 0 ∧ a > b ∧ (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → y = 1 ∧ y = 3 / 2)

def eccentricity (a b : ℝ) := b = (1 / 2) * a

def passes_through (x y : ℝ) := x = 1 ∧ y = 3 / 2

-- Part 1: Prove the equation of the ellipse
theorem find_ellipse_eq (a b : ℝ) (h_ellipse : ellipse a b) (h_eccentricity : eccentricity a b) (h_point : passes_through 1 (3/2)) :
    (x^2) / 4 + (y^2) / 3 = 1 :=
sorry

-- Circle equation definition
def circle (x y : ℝ) := x^2 + y^2 = 7

-- Part 2: Prove the product of the slopes of the tangent lines is constant
theorem product_of_tangent_slopes (P : ℝ × ℝ) (h_circle : circle P.1 P.2) : 
    ∀ k1 k2 : ℝ, (4 - P.1^2) * k1^2 + 6 * P.1 * P.2 * k1 + 3 - P.2^2 = 0 → 
    (4 - P.1^2) * k2^2 + 6 * P.1 * P.2 * k2 + 3 - P.2^2 = 0 → k1 * k2 = -1 :=
sorry

end find_ellipse_eq_product_of_tangent_slopes_l772_772637


namespace jacks_walking_rate_l772_772503

theorem jacks_walking_rate :
  let distance := 8
  let time_in_minutes := 1 * 60 + 15
  let time := time_in_minutes / 60.0
  let rate := distance / time
  rate = 6.4 :=
by
  sorry

end jacks_walking_rate_l772_772503


namespace log_inequality_l772_772579

theorem log_inequality (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ 1) (h4 : y ≠ 1) :
    (Real.log y / Real.log x + Real.log x / Real.log y > 2) →
    (x ≠ y ∧ ((x > 1 ∧ y > 1) ∨ (x < 1 ∧ y < 1))) :=
by
    sorry

end log_inequality_l772_772579


namespace possible_ticket_values_l772_772544

open Int Nat

theorem possible_ticket_values (x : ℕ) (h1 : ∃ n m : ℕ, n * x = 72 ∧ m * x = 90) : 
  (∃ divs : Finset ℕ, divs.card = 6 ∧ ∀ d ∈ divs, d ∣ gcd 72 90) :=
by
  have gcd_72_90 : gcd 72 90 = 18 := by
    calc
      gcd 72 90 = 18 := by sorry -- Detailed steps of Euclidean algorithm are skipped
  use {1, 2, 3, 6, 9, 18}
  split
  · exact Finset.card_insert_of_not_mem sorry sorry sorry -- by counting needed divisors
  · intros d hd
    simp only [Finset.mem_insert, Finset.mem_singleton] at hd
    rcases hd with rfl | rfl | rfl | rfl | rfl | rfl
    · exact dvd_refl 18  -- for each divisor in the set, check it divides 18
  
  done -- Sorry is used to complete the sketch

end possible_ticket_values_l772_772544


namespace probability_problem_l772_772530

-- Define the events A, B, C, D, and E and their corresponding probabilities
variable {A B C D E : Prop}
variable {P : Prop → ℝ}
variable hA : P A = 0.20
variable hB : P B = 0.10
variable hC : P C = 0.15
variable hD : P D = 0.25
variable hE : P E = 0.30

-- Define the main theorem to prove the required probabilities given the conditions
theorem probability_problem :
  (P (A ∨ C) = 0.35) ∧
  (P (B ∨ E) = 0.40) ∧
  (P (A ∨ D) = 0.45) ∧
  (P (A ∨ B ∨ C) = 0.45) ∧
  (P (D ∨ E) = 0.55) := by
  sorry -- Proof to be filled

end probability_problem_l772_772530


namespace eccentricity_range_l772_772718

-- Definitions and conditions for the problem
def is_ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (ab_relation : b < a) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 → abs x < a)

def obtuse_angle_condition (x0 y0 a b e : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (ab_relation : b < a)
  (H : x0^2 / a^2 + y0^2 / b^2 = 1) (c : ℝ) (c_val : c = a * e) : Prop :=
  ((-c - x0) * (c - x0) + (-y0) * (-y0)) < 0

-- The main statement we need to prove
theorem eccentricity_range (a b c e : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (ab_relation : b < a)
  (ecc_relation : c = a * e) (ecc_pos : 0 < e) (ecc_upper : e < 1)
  (obtuse_condition : ∃ (x0 y0 : ℝ), x0^2 + y0^2 = b^2 ∧ ¬obtuse_angle_condition x0 y0 a b e a_pos b_pos ab_relation (x0^2 / a^2 + y0^2 / b^2 = 1) c ecc_relation) :
  (sqrt 2 / 2) < e ∧ e < 1 :=
sorry

end eccentricity_range_l772_772718


namespace urn_final_contents_possible_l772_772352

def initial_white_marbles := 150
def initial_black_marbles := 50

def operation1 (white black : ℕ) : ℕ × ℕ :=
  (white + 3, black - 2)

def operation2 (white black : ℕ) : ℕ × ℕ :=
  (white - 2, black + 1)

def operation3 (white black : ℕ) : ℕ × ℕ :=
  (white - 2, black)

theorem urn_final_contents_possible : 
  ∃ (n1 n2 n3 : ℕ), operation1_n n1 (operation2_n n2 (operation3_n n3 (initial_white_marbles, initial_black_marbles))) = (148, 2) :=
sorry

end urn_final_contents_possible_l772_772352


namespace drum_Y_full_capacity_l772_772185

variable (C : ℝ) (hC : 0 < C)

def drum_X_oil := (1 / 2) * C
def drum_Y_capacity := 2 * C
def drum_Y_oil := (1 / 4) * drum_Y_capacity
def drum_Z_oil := (1 / 3) * (3 * C)

theorem drum_Y_full_capacity :
  drum_Y_oil + drum_X_oil + drum_Z_oil = drum_Y_capacity :=
by
  -- the statement of the theorem
  sorry

end drum_Y_full_capacity_l772_772185


namespace max_x_minus_y_l772_772667

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772667


namespace maximum_value_of_x_minus_y_l772_772709

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l772_772709


namespace find_C_l772_772959

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 320) : 
  C = 20 := 
by 
  sorry

end find_C_l772_772959


namespace root_in_1_2_l772_772114

noncomputable def f (x : ℝ) : ℝ := x - 2 + Real.log x

theorem root_in_1_2 : ∃ c ∈ set.Ioo (1 : ℝ) 2, f c = 0 :=
by
  have h1 : f 1 < 0 := by simp [f, Real.log_one]
  have h2 : f 2 > 0 := by simp [f, Real.log]
  apply exists_between
  exact lt_of_lt_of_le h1 (le_of_lt h2)
  sorry

end root_in_1_2_l772_772114


namespace angle_equality_l772_772439

variables (A B C D O E F : Type) [parallelogram A B C D]
variables (circumcircle_a_o_e : circle A B O) (circumcircle_d_o_e : circle D O E)
variables (intersect_e : intersects circumcircle_a_o_e AD E)
variables (intersect_f : intersects circumcircle_d_o_e BE F)

theorem angle_equality :
  angle B C A = angle F C D :=
sorry

end angle_equality_l772_772439


namespace max_marks_paper_one_l772_772523

theorem max_marks_paper_one (M : ℝ) : 
  (0.42 * M = 64) → (M = 152) :=
by
  sorry

end max_marks_paper_one_l772_772523


namespace tangent_line_at_x0_l772_772985

noncomputable def curve (x : ℝ) := (x ^ 2 - 3 * x + 3) / 3
def x0 : ℝ := 3

theorem tangent_line_at_x0 (x : ℝ) : 
    let y0 := curve x0 in
    let slope := (deriv curve) x0 in
    y = slope * (x - x0) + y0 :=
by
    sorry

end tangent_line_at_x0_l772_772985


namespace probability_product_greater_than_zero_l772_772893

open ProbabilityTheory

-- Define the interval, probabilities, and the final probability
noncomputable def interval := Set.Icc (-30 : ℝ) 15
noncomputable def probability_pos := (15 / 45 : ℝ)
noncomputable def probability_neg := (30 / 45 : ℝ)
noncomputable def probability_product_gt_zero := (probability_pos ^ 2) + (probability_neg ^ 2)

-- Lean 4 statement for the proof
theorem probability_product_greater_than_zero :
  probability_product_gt_zero = (5 / 9 : ℝ) :=
by
  sorry

end probability_product_greater_than_zero_l772_772893


namespace angle_of_inclination_l772_772263

open Real

theorem angle_of_inclination : 
  let x (t : ℝ) := 1 - (1/2) * t
  let y (t : ℝ) := (sqrt 3 / 2) * t
  ∃ θ : ℝ, atan (-sqrt 3) = θ ∧ degree θ = 120 :=
by
  sorry

end angle_of_inclination_l772_772263


namespace find_p_from_conditions_l772_772115

section problem

variables {A B : ℝ × ℝ} -- points of intersection of ellipse and parabola
variable {O : ℝ × ℝ} (O := (0, 0)) -- origin

-- Ellipse and parabola definitions
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def parabola (x y : ℝ) (p : ℝ) : Prop := x^2 = 2 * p * y

-- Conditions definitions
def on_ellipse (pt : ℝ × ℝ) : Prop := ellipse pt.1 pt.2
def on_parabola (pt : ℝ × ℝ) (p : ℝ) : Prop := parabola pt.1 pt.2 p
def intersection (pt : ℝ × ℝ) (p : ℝ) : Prop := on_ellipse pt ∧ on_parabola pt p

-- Point definitions
variable {N : ℝ × ℝ} (N := (0, 13 / 2)) -- point N for the second part

-- Proposition for first part: circumcenter on ellipse
def circumcenter_on_ellipse (p : ℝ) : Prop :=
∀ A B : ℝ × ℝ, intersection A p → intersection B p →
  (∃ C : ℝ × ℝ, C = circumcenter O A B ∧ on_ellipse C ∧ C = (0, 1)) → p = (7 - sqrt(13)) / 6
noncomputable def circumcenter (O A B : ℝ × ℝ) : ℝ × ℝ := sorry

-- Proposition for second part: circumcircle passing through N
def circumcircle_passing_through_N (p : ℝ) : Prop :=
∀ A B : ℝ × ℝ, intersection A p → intersection B p →
  (∃ O : ℝ × ℝ, O = (0, 0)) →
  (∃ r : ℝ, r = circumradius O A B ∧ (∃ N : ℝ × ℝ, N = (0, 13 / 2))) → p = 3
noncomputable def circumradius (O A B : ℝ × ℝ) : ℝ := sorry

-- Proof problem combining the parts
theorem find_p_from_conditions :
  ∃ p : ℝ, (circumcenter_on_ellipse p ∨ circumcircle_passing_through_N p)
:= sorry

end problem

end find_p_from_conditions_l772_772115


namespace BK_invariant_l772_772169

-- Definitions for points, triangles and circles
variable {α : Type*} [metric_space α] [normed_group α] 

structure Point (α : Type*) :=
(x : ℝ) (y : ℝ)

-- Triangle ABC
structure Triangle (α : Type*) :=
(A B C : Point α)

structure Circle (α : Type*) :=
(center : Point α) (radius : ℝ)

def center_of_circumcircle_of_triangle (A B M : Point α) : Point α := sorry

def points_on_circle (c : Circle α) (P1 P2 : Point α) : Prop := sorry

-- The task is to prove that the line BK is the same for any point M on BC
theorem BK_invariant (ABC : Triangle α) (M : Point α) (BC_segment : list (Point α))
  (O : Point α) (k : Circle α) (conditions : Point α → Prop)
  (H1 : M ∈ BC_segment)
  (H2 : O = center_of_circumcircle_of_triangle ABC.A ABC.B M)
  (H3 : ∃ k, k.center ∈ line_of_points (ABC.B :: ABC.C :: []))
  (H4 : points_on_circle k ABC.A M)
  (K : Point α)
  (H5 : K ∈ line_of_points (M :: O :: []))
  (H6 : K ∉ M) : 
  ∀ M ∈ BC_segment, ∃ BK : Line α, line_of_points [ABC.B, K] = BK := 
sorry

end BK_invariant_l772_772169


namespace max_x_minus_y_l772_772645

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772645


namespace probability_of_red_pen_l772_772469

theorem probability_of_red_pen :
  let colors := ["red", "yellow", "blue", "green", "purple"] in
  let total_outcomes := Nat.choose 5 2 in
  let favorable_outcomes := Nat.choose 1 1 * Nat.choose 4 1 in
  (favorable_outcomes / total_outcomes : ℚ) = 2 / 5 :=
by
  let colors := ["red", "yellow", "blue", "green", "purple"]
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 1 1 * Nat.choose 4 1
  have : (total_outcomes : ℚ) = 10 := by sorry
  have : (favorable_outcomes : ℚ) = 4 := by sorry
  calc 
    (favorable_outcomes / total_outcomes : ℚ)
        = (4 / 10 : ℚ)        : by sorry
    ... = (2 / 5 : ℚ)         : by sorry

end probability_of_red_pen_l772_772469


namespace final_result_approx_l772_772207

def place_value_7_in_856973 : ℤ := 7 * 10000
def face_value_7_in_856973 : ℤ := 7

def diff_place_face_value_7 : ℤ := place_value_7_in_856973 - face_value_7_in_856973

def sum_digits_489237 : ℤ := 4 + 8 + 9 + 2 + 3 + 7

def multiplied_result : ℤ := diff_place_face_value_7 * sum_digits_489237

def place_value_3_in_734201 : ℤ := 3 * 1000
def face_value_3_in_734201 : ℤ := 3

def diff_place_face_value_3 : ℤ := place_value_3_in_734201 - face_value_3_in_734201

def final_result : ℤ := multiplied_result / diff_place_face_value_3

theorem final_result_approx :
  abs (final_result.toReal - 770.43) < 1e-2 := by
  sorry

end final_result_approx_l772_772207


namespace math_problem_proof_l772_772488

theorem math_problem_proof :
  let a := 3⁻¹
  let b := 7^3
  let c := a + b - 2
  let d := c⁻¹
  let e := d * 7
  e = 21 / 1024 :=
by
  sorry

end math_problem_proof_l772_772488


namespace shrink_ray_coffee_l772_772127

theorem shrink_ray_coffee (num_cups : ℕ) (ounces_per_cup : ℕ) (shrink_factor : ℝ) 
  (h1 : num_cups = 5) 
  (h2 : ounces_per_cup = 8) 
  (h3 : shrink_factor = 0.5) 
  : num_cups * ounces_per_cup * shrink_factor = 20 :=
by
  rw [h1, h2, h3]
  simp
  norm_num

end shrink_ray_coffee_l772_772127


namespace max_difference_value_l772_772680

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l772_772680


namespace find_b10_l772_772377

def sequence (b : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n

theorem find_b10 (b : ℕ → ℤ) (h_seq : sequence b) (h_b1 : b 1 = 2) : b 10 = 110 :=
  sorry

end find_b10_l772_772377


namespace sum_of_smallest_multiples_l772_772384

def smallest_two_digit_multiple_of_5 := 10
def smallest_three_digit_multiple_of_7 := 105

theorem sum_of_smallest_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end sum_of_smallest_multiples_l772_772384


namespace find_n_l772_772208

-- Definition of the conditions
def condition1 (n : ℤ) : Prop := 0 ≤ n ∧ n ≤ 6
def condition2 (n : ℤ) : Prop := n ≡ -7845 [MOD 7]

-- Final proof problem statement
theorem find_n (n : ℤ) (h1 : condition1 n) (h2 : condition2 n) : n = 2 :=
by
  sorry

end find_n_l772_772208


namespace map_upper_half_disk_to_upper_half_plane_l772_772595

noncomputable def map_function (z : ℂ) : ℂ :=
  ((1 + z) / (1 - z)) ^ 2

theorem map_upper_half_disk_to_upper_half_plane (z : ℂ) (H : |z| < 1 ∧ 0 < z.im) :
  0 < (map_function z).im := 
sorry

end map_upper_half_disk_to_upper_half_plane_l772_772595


namespace fixed_point_of_function_l772_772444

def f (a : ℝ) (x : ℝ) := a^(x-1) + 4

theorem fixed_point_of_function (a : ℝ) : f a 1 = 5 :=
by
  unfold f
  sorry

end fixed_point_of_function_l772_772444


namespace coffee_shrinkage_l772_772129

theorem coffee_shrinkage :
  let initial_volume_per_cup := 8
  let shrink_factor := 0.5
  let number_of_cups := 5
  let final_volume_per_cup := initial_volume_per_cup * shrink_factor
  let total_remaining_coffee := final_volume_per_cup * number_of_cups
  total_remaining_coffee = 20 :=
by
  -- This is where the steps of the solution would go.
  -- We'll put a sorry here to indicate omission of proof.
  sorry

end coffee_shrinkage_l772_772129


namespace length_of_train_is_475_l772_772545

-- Conditions
def train_speed_kmph : ℝ := 90  -- Train speed in km/h
def crossing_time_s : ℝ := 30   -- Time to cross the bridge in seconds
def bridge_length_m : ℝ := 275  -- Length of the bridge in meters

-- Convert speed to meters per second
def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600

-- Distance traveled by the train
def distance_traveled : ℝ := train_speed_mps * crossing_time_s

-- Length of the train
def train_length : ℝ := distance_traveled - bridge_length_m

-- The theorem proving the length of the train is 475 meters
theorem length_of_train_is_475 : train_length = 475 := by
  sorry

end length_of_train_is_475_l772_772545


namespace cone_height_approx_l772_772120

-- Given Conditions
variable (V : Real) (r h : Real)
variable (cone_volume : V = 20000 * Real.pi)
variable (vertex_angle : h = r * Real.sqrt 2)
variable (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)

-- Theorem to prove
theorem cone_height_approx : h ≈ 43.8 :=
by
  -- Here is where the proof would go, currently omitted.
  sorry

end cone_height_approx_l772_772120


namespace Correct_Statement_l772_772097

theorem Correct_Statement : 
  (∀ x : ℝ, 7 * x = 4 * x - 3 → 7 * x - 4 * x = -3) ∧
  (∀ x : ℝ, (2 * x - 1) / 3 = 1 + (x - 3) / 2 → 2 * (2 * x - 1) = 6 + 3 * (x - 3)) ∧
  (∀ x : ℝ, 2 * (2 * x - 1) - 3 * (x - 3) = 1 → 4 * x - 2 - 3 * x + 9 = 1) ∧
  (∀ x : ℝ, 2 * (x + 1) = x + 7 → x = 5) :=
by
  sorry

end Correct_Statement_l772_772097


namespace sum_of_decimals_is_fraction_l772_772192

def decimal_to_fraction_sum : ℚ :=
  (1 / 10) + (2 / 100) + (3 / 1000) + (4 / 10000) + (5 / 100000) + (6 / 1000000) + (7 / 10000000)

theorem sum_of_decimals_is_fraction :
  decimal_to_fraction_sum = 1234567 / 10000000 :=
by sorry

end sum_of_decimals_is_fraction_l772_772192


namespace line_passing_through_incircle_centers_perpendicular_to_angle_bisector_l772_772871

-- Definitions of the geometric entities involved.
variables {A B C M K O1 O2 : Type} [EuclideanGeometry]

-- Conditions provided in the problem statement.
def perpendicular_bisector_AC_intersects_BC_at_M (A B C M : Point) : Prop :=
is_perpendicular_bisector (segment A C) (segment B C) M

def angle_bisector_AMB_intersects_circumcircle_ABC_at_K (A B C M K : Point) : Prop :=
∃ (Ω : Circle), is_circumcircle Ω (triangle A B C) ∧
intersects Ω (line (angle_bisector (∠AMB))) K

-- Final proof goal.
theorem line_passing_through_incircle_centers_perpendicular_to_angle_bisector (A B C M K O1 O2 : Point)
  (h1 : perpendicular_bisector_AC_intersects_BC_at_M A B C M)
  (h2 : angle_bisector_AMB_intersects_circumcircle_ABC_at_K A B C M K)
  (incenter_AKM : center_of_incircle (triangle A K M) = O1)
  (incenter_BKM : center_of_incircle (triangle B K M) = O2) :
  is_perpendicular (line (segment O1 O2)) (angle_bisector (∠AKB)) :=
sorry

end line_passing_through_incircle_centers_perpendicular_to_angle_bisector_l772_772871


namespace red_marbles_l772_772566

theorem red_marbles (R B : ℕ) (h₁ : B = R + 24) (h₂ : B = 5 * R) : R = 6 := by
  sorry

end red_marbles_l772_772566


namespace central_angle_of_sector_with_area_one_l772_772784

theorem central_angle_of_sector_with_area_one (θ : ℝ):
  (1 / 2) * θ = 1 → θ = 2 :=
by
  sorry

end central_angle_of_sector_with_area_one_l772_772784


namespace catalan_recurrence_l772_772833

noncomputable def catalan : ℕ → ℕ
| 0       := 1
| (n + 1) := ∑ k in Finset.range (n+1), catalan k * catalan (n - k)

theorem catalan_recurrence : ∀ n : ℕ, 
  catalan (n + 1) = ∑ k in Finset.range (n+1), (catalan k) * (catalan (n - k)) :=
by
  sorry

end catalan_recurrence_l772_772833


namespace jellybeans_Diana_box_l772_772218

def jellybeans_in_box (volume_ratio : ℕ) (jellybeans_Bert : ℕ) : ℕ := 
  volume_ratio * jellybeans_Bert

theorem jellybeans_Diana_box (jellybeans_Bert : ℕ) (mult1 mult2 mult3 : ℕ) 
  (h_jellybeans_Bert : jellybeans_Bert = 150) 
  (h_mult1 : mult1 = 3) 
  (h_mult2 : mult2 = 2)
  (h_mult3 : mult3 = 4) : 
  jellybeans_in_box (mult1 * mult2 * mult3) jellybeans_Bert = 3600 :=
by
  unfold jellybeans_in_box
  rw [h_jellybeans_Bert, h_mult1, h_mult2, h_mult3]
  norm_num
  sorry

end jellybeans_Diana_box_l772_772218


namespace problem_solution_correct_l772_772240

-- Define the arithmetic sequence a_n
def a_n (n : ℕ) : ℕ := 2 * n - 1

-- Define the geometric sequence b_n
def b_n (n : ℕ) : ℕ := 2^(n - 1)

-- Define the sequence c_n
def c_n (n : ℕ) : ℕ := a_n n * b_n n

-- Define the sum of the first n terms of c_n, S_n
noncomputable def S_n (n : ℕ) : ℕ := 3 + (2 * n - 3) * 2^n

-- Prove that the derived terms are correct given the conditions
theorem problem_solution_correct :
  (∀ n, a_n n = 2 * n - 1) ∧
  (∀ n, b_n n = 2^(n - 1)) ∧
  (∀ n, c_n n = a_n n * b_n n) ∧
  (∀ n, ∑ i in finset.range n, c_n (i + 1) = S_n n) := by
  sorry

end problem_solution_correct_l772_772240


namespace range_a_decreasing_l772_772728

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (1 - a) * x + a else (a - 3) * x^2 + 2

theorem range_a_decreasing :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) → (2 ≤ a ∧ a < 3) :=
begin
  -- placeholder for any steps in the proof
  sorry
end

end range_a_decreasing_l772_772728


namespace simplify_and_evaluate_expression_l772_772028

theorem simplify_and_evaluate_expression (x y : ℝ) (h_x : x = -2) (h_y : y = 1) :
  (((2 * x - (1/2) * y)^2 - ((-y + 2 * x) * (2 * x + y)) + y * (x^2 * y - (5/4) * y)) / x) = -4 :=
by
  sorry

end simplify_and_evaluate_expression_l772_772028


namespace max_x_minus_y_l772_772642

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772642


namespace integral_value_l772_772977

def pi_div_4 : Real := Real.pi / 4
def arcsin_2_over_sqrt_5 : Real := Real.arcsin(2 / Real.sqrt 5)
def integrand (x : Real) : Real := (4 * Real.tan x - 5) / (4 * Real.cos x ^ 2 - Real.sin (2 * x) + 1)

theorem integral_value :
  ∫ x in pi_div_4..arcsin_2_over_sqrt_5, integrand x = 
  2 * Real.log (5 / 4) - (1 / 2) * Real.arctan (1 / 2) :=
by
  sorry

end integral_value_l772_772977


namespace oil_leak_before_fix_l772_772970

theorem oil_leak_before_fix (total_leak : ℕ) (leak_during_fix : ℕ) 
    (total_leak_eq : total_leak = 11687) (leak_during_fix_eq : leak_during_fix = 5165) :
    total_leak - leak_during_fix = 6522 :=
by 
  rw [total_leak_eq, leak_during_fix_eq]
  simp
  sorry

end oil_leak_before_fix_l772_772970


namespace five_digit_number_unique_nonzero_l772_772765

theorem five_digit_number_unique_nonzero (a b c d e : ℕ) (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) (h3 : (100 * a + 10 * b + c) * 7 = 100 * c + 10 * d + e) : a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 4 ∧ e = 6 :=
by
  sorry

end five_digit_number_unique_nonzero_l772_772765


namespace max_difference_value_l772_772683

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l772_772683


namespace conic_section_eccentricity_l772_772724

theorem conic_section_eccentricity (a : ℝ) (h : a^2 = 9) :
  let e := if a = 3 then (Real.sqrt 3 / 3) else (Real.sqrt 10 / 2) in
  e = (Real.sqrt 3 / 3) ∨ e = (Real.sqrt 10 / 2) :=
by
  sorry

end conic_section_eccentricity_l772_772724


namespace tan_cos_simplify_l772_772839

theorem tan_cos_simplify :
  (tan (20 * Real.pi / 180) + tan (30 * Real.pi / 180) + tan (60 * Real.pi / 180) + tan (70 * Real.pi / 180)) / cos (10 * Real.pi / 180)
  = (Real.sqrt 3 + 2) * Real.sqrt 3 / 2 :=
by
  sorry

end tan_cos_simplify_l772_772839


namespace distinct_pairs_solution_l772_772179

noncomputable def number_of_distinct_pairs : ℕ :=
  let S := {p : ℝ × ℝ | let (x, y) := p; x = 2 * x^2 + y^2 ∧ y = 3 * x * y} in
  S.toFinset.card

theorem distinct_pairs_solution : number_of_distinct_pairs = 4 :=
by
  sorry

end distinct_pairs_solution_l772_772179


namespace M_ends_in_two_zeros_iff_l772_772367

theorem M_ends_in_two_zeros_iff (n : ℕ) (h : n > 0) : 
  (1^n + 2^n + 3^n + 4^n) % 100 = 0 ↔ n % 4 = 3 :=
by sorry

end M_ends_in_two_zeros_iff_l772_772367


namespace prime_dates_in_2008_l772_772810

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_prime_date (month day : ℕ) : Prop := is_prime month ∧ is_prime day

noncomputable def prime_dates_2008 : ℕ :=
  let prime_months := [2, 3, 5, 7, 11]
  let prime_days_31 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let prime_days_30 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_days_29 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  prime_months.foldl (λ acc month => 
    acc + match month with
      | 2 => List.length prime_days_29
      | 3 | 5 | 7 => List.length prime_days_31
      | 11 => List.length prime_days_30
      | _ => 0
    ) 0

theorem prime_dates_in_2008 : 
  prime_dates_2008 = 53 :=
  sorry

end prime_dates_in_2008_l772_772810


namespace remainder_of_3_pow_800_mod_17_l772_772903

theorem remainder_of_3_pow_800_mod_17 :
    (3 ^ 800) % 17 = 1 :=
by
    sorry

end remainder_of_3_pow_800_mod_17_l772_772903


namespace polynomial_remainder_l772_772374

theorem polynomial_remainder (Q : ℕ → ℕ) (h₁ : Q 21 = 105) (h₂ : Q 105 = 21) :
  ∃ c d : ℕ, (∀ x, Q x = (x - 21) * (x - 105) * some_polynomial x + c * x + d) ∧ c = -1 ∧ d = 126 :=
by
  sorry

end polynomial_remainder_l772_772374


namespace parallelogram_to_kite_l772_772174

theorem parallelogram_to_kite (A B C D E O : Point) (h_parallelogram : is_parallelogram A B C D) 
  (h_symmetric : is_symmetric A E C with respect to AC) 
  (h_midpoint_o : is_midpoint O (A, B) (E, C))
  (h_intersection : intersect AE BC O) : 
  can_reassemble_to_kite (cut_points A O) :=
begin
  sorry
end

end parallelogram_to_kite_l772_772174


namespace find_x_l772_772750

-- Definition of the problem
def infinite_series (x : ℝ) : ℝ := ∑' n : ℕ, (n + 1) * x^n

-- Given condition
axiom condition : infinite_series x = 4

-- Statement to prove
theorem find_x : (∃ x : ℝ, infinite_series x = 4) → x = 1/2 := by
  sorry

end find_x_l772_772750


namespace gas_volume_at_12_l772_772610

variable (VolumeTemperature : ℕ → ℕ) -- a function representing the volume of gas at a given temperature 

axiom condition1 : ∀ t : ℕ, VolumeTemperature (t + 4) = VolumeTemperature t + 5

axiom condition2 : VolumeTemperature 28 = 35

theorem gas_volume_at_12 :
  VolumeTemperature 12 = 15 := 
sorry

end gas_volume_at_12_l772_772610


namespace people_off_second_eq_8_l772_772884

-- Initial number of people on the bus
def initial_people := 50

-- People who got off at the first stop
def people_off_first := 15

-- People who got on at the second stop
def people_on_second := 2

-- People who got off at the second stop (unknown, let's call it x)
variable (x : ℕ)

-- People who got off at the third stop
def people_off_third := 4

-- People who got on at the third stop
def people_on_third := 3

-- Number of people on the bus after the third stop
def people_after_third := 28

-- Equation formed by given conditions
def equation := initial_people - people_off_first - x + people_on_second - people_off_third + people_on_third = people_after_third

-- Goal: Prove the equation with given conditions results in x = 8
theorem people_off_second_eq_8 : equation x → x = 8 := by
  sorry

end people_off_second_eq_8_l772_772884


namespace find_n_l772_772086

theorem find_n :
  ∃ n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2222 ≡ n [MOD 9] ∧ n = 1 := 
sorry

end find_n_l772_772086


namespace compare_exponentiations_l772_772983

theorem compare_exponentiations :
  0 < (0.5 : ℝ) ∧ (0.5 : ℝ) < 1 → (0.5 : ℝ)^(-2) > (0.5 : ℝ)^(-0.8) :=
by
  sorry

end compare_exponentiations_l772_772983


namespace fruit_platter_has_thirty_fruits_l772_772007

-- Define the conditions
def at_least_five_apples (g_apple r_apple y_apple : ℕ) : Prop :=
  g_apple + r_apple + y_apple ≥ 5

def at_most_five_oranges (r_orange y_orange : ℕ) : Prop :=
  r_orange + y_orange ≤ 5

def kiwi_grape_constraints (g_kiwi p_grape : ℕ) : Prop :=
  g_kiwi + p_grape ≥ 8 ∧ g_kiwi + p_grape ≤ 12 ∧ g_kiwi = p_grape

def at_least_one_each_grape (g_grape p_grape : ℕ) : Prop :=
  g_grape ≥ 1 ∧ p_grape ≥ 1

-- The final statement to prove
theorem fruit_platter_has_thirty_fruits :
  ∃ (g_apple r_apple y_apple r_orange y_orange g_kiwi p_grape g_grape : ℕ),
    at_least_five_apples g_apple r_apple y_apple ∧
    at_most_five_oranges r_orange y_orange ∧
    kiwi_grape_constraints g_kiwi p_grape ∧
    at_least_one_each_grape g_grape p_grape ∧
    g_apple + r_apple + y_apple + r_orange + y_orange + g_kiwi + p_grape + g_grape = 30 :=
sorry

end fruit_platter_has_thirty_fruits_l772_772007


namespace polynomial_remainder_l772_772489

theorem polynomial_remainder (x : ℤ) : (x + 1) ∣ (x^15 + 1) ↔ x = -1 := sorry

end polynomial_remainder_l772_772489


namespace valid_arrangements_formula_l772_772272

def valid_arrangements (n : ℕ) : ℕ :=
  if n = 1 then 3
  else if n = 2 then 8
  else 2 * (valid_arrangements (n - 2)) + 2 * (valid_arrangements (n - 1))

-- Prove that valid_arrangements follows the formula derived in the solution
theorem valid_arrangements_formula (n : ℕ) :
  valid_arrangements n =
  (let A := (2 + Real.sqrt 3) / (2 * Real.sqrt 3)
   let B := (-2 + Real.sqrt 3) / (2 * Real.sqrt 3)
   A * (1 + Real.sqrt 3) ^ n + B * (1 - Real.sqrt 3) ^ n) :=
by
  -- Proof omitted
  sorry

end valid_arrangements_formula_l772_772272


namespace remaining_marbles_l772_772406

theorem remaining_marbles (initial_marbles : ℕ) (num_customers : ℕ) (marble_range : List ℕ)
  (h_initial : initial_marbles = 2500)
  (h_customers : num_customers = 50)
  (h_range : marble_range = List.range' 1 50)
  (disjoint_range : ∀ (a b : ℕ), a ∈ marble_range → b ∈ marble_range → a ≠ b → a + b ≤ 50) :
  initial_marbles - (num_customers * (50 + 1) / 2) = 1225 :=
by
  sorry

end remaining_marbles_l772_772406


namespace javier_visit_sequences_l772_772355

theorem javier_visit_sequences :
  (finset.perm (finset.mk₀ ["A", "B", "C", "D", "E", "S", "S"])) / (multiset.card (multiset.replicate 2 "S")!) = 360 :=
by
  sorry

end javier_visit_sequences_l772_772355


namespace sequence_a2007_l772_772556

theorem sequence_a2007 :
  ∀ (a : ℕ → ℝ), (a 0 = 1) →
  (∀ (n : ℕ), a (n + 2) = 6 * a n - a (n + 1)) →
  a 2007 = 2 ^ 2007 :=
by {
  intros a h₀ hrec,
  sorry
}

end sequence_a2007_l772_772556


namespace find_dividend_l772_772560

theorem find_dividend (partial_product : ℕ) (remainder : ℕ) (divisor quotient : ℕ) :
  partial_product = 2015 → 
  remainder = 0 →
  divisor = 105 → 
  quotient = 197 → 
  divisor * quotient + remainder = partial_product → 
  partial_product * 10 = 20685 :=
by {
  -- Proof skipped
  sorry
}

end find_dividend_l772_772560


namespace bea_earns_more_than_dawn_l772_772152

noncomputable def bea_price_per_glass := 25
noncomputable def dawn_price_per_glass := 28
noncomputable def bea_glasses_sold := 10
noncomputable def dawn_glasses_sold := 8

def bea_earnings : Nat :=
bea_glasses_sold * bea_price_per_glass

def dawn_earnings : Nat :=
dawn_glasses_sold * dawn_price_per_glass

theorem bea_earns_more_than_dawn :
  (bea_earnings - dawn_earnings) = 26 :=
by
  sorry

end bea_earns_more_than_dawn_l772_772152


namespace shrink_ray_coffee_l772_772128

theorem shrink_ray_coffee (num_cups : ℕ) (ounces_per_cup : ℕ) (shrink_factor : ℝ) 
  (h1 : num_cups = 5) 
  (h2 : ounces_per_cup = 8) 
  (h3 : shrink_factor = 0.5) 
  : num_cups * ounces_per_cup * shrink_factor = 20 :=
by
  rw [h1, h2, h3]
  simp
  norm_num

end shrink_ray_coffee_l772_772128


namespace wheat_bread_served_l772_772409

noncomputable def total_bread_served : ℝ := 0.6
noncomputable def white_bread_served : ℝ := 0.4

theorem wheat_bread_served : total_bread_served - white_bread_served = 0.2 :=
by
  sorry

end wheat_bread_served_l772_772409


namespace boxes_ball_coloring_l772_772998

noncomputable def smallest_color_number : Nat :=
  23

theorem boxes_ball_coloring :
  ∀ (S : Fin 8 → Finset (Fin smallest_color_number)),
  (∀ i, S i.card = 6) ∧ 
  (∀ i j, i ≠ j → S i ∩ S j = ∅) → 
  (∃ n, n = smallest_color_number) :=
by
  intros S h
  use smallest_color_number
  sorry

end boxes_ball_coloring_l772_772998


namespace collinearity_condition_not_sufficient_condition_necessary_but_not_sufficient_condition_l772_772436

noncomputable def collinear_points (A B C D : Type) [has_vsub A] : Prop :=
  ∃ (u : ℝ), ∀ (x y : A), y - x = u * (B - A)

noncomputable def collinear_vectors (A B C D : Type) [has_vsub A] : Prop :=
  ∃ (v : ℝ), ∀ (x y : A), y - x = v * (D - C)

theorem collinearity_condition (A B C D : Type) [has_vsub A] :
  collinear_points A B C D → collinear_vectors A B C D :=
sorry

theorem not_sufficient_condition (A B C D : Type) [has_vsub A] :
  collinear_vectors A B C D → collinear_points A B C D :=
sorry

theorem necessary_but_not_sufficient_condition (A B C D : Type) [has_vsub A] :
  collinear_points A B C D ↔ collinear_vectors A B C D ∧ ¬collinear_vectors A B C D :=
begin
  split,
  { intro h,
    split,
    { apply collinearity_condition,
      exact h },
    { intro,
      apply not_sufficient_condition,
      exact h } },
  { intro h,
    cases h with hc hn,
    apply collinearity_condition,
    exact hc }
end

end collinearity_condition_not_sufficient_condition_necessary_but_not_sufficient_condition_l772_772436


namespace maximum_value_of_x_minus_y_l772_772693

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l772_772693


namespace shortest_distance_polar_coordinates_l772_772329

noncomputable def polar_coord_A := (2 : ℝ, Real.pi / 2)
def line_l (ρ θ : ℝ) := ρ * Real.cos θ + ρ * Real.sin θ = 0
def point_B_on_line_l (ρ θ : ℝ) := 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ line_l ρ θ

theorem shortest_distance_polar_coordinates :
  ∃ (ρ θ : ℝ), point_B_on_line_l ρ θ ∧ (ρ = Real.sqrt 2) ∧ (θ = 3 * Real.pi / 4) :=
sorry

end shortest_distance_polar_coordinates_l772_772329


namespace min_surveyed_consumers_l772_772543

theorem min_surveyed_consumers (N : ℕ) 
    (h10 : ∃ k : ℕ, N = 10 * k)
    (h30 : ∃ l : ℕ, N = 10 * l) 
    (h40 : ∃ m : ℕ, N = 5 * m) : 
    N = 10 :=
by
  sorry

end min_surveyed_consumers_l772_772543


namespace trapezium_other_side_length_l772_772204

theorem trapezium_other_side_length (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ) :
  side1 = 20 ∧ distance = 15 ∧ area = 285 → side2 = 18 :=
by
  intro h
  sorry

end trapezium_other_side_length_l772_772204


namespace system_of_equations_solution_l772_772112

theorem system_of_equations_solution (x y : ℚ) :
  (x / 3 + y / 4 = 4 ∧ 2 * x - 3 * y = 12) → (x = 10 ∧ y = 8 / 3) :=
by
  sorry

end system_of_equations_solution_l772_772112


namespace angle_Q_is_72_degrees_l772_772024

-- Define the context and conditions
def regular_decagon_angles_sum (n : ℕ) : ℕ := 180 * (n - 2)

def one_angle_of_regular_decagon (n : ℕ) := (regular_decagon_angles_sum n) / n

def reflex_angle (angle : ℕ) := 360 - angle

-- Define the problem as a theorem
theorem angle_Q_is_72_degrees :
  let n := 10 in 
  let angle_EFG := one_angle_of_regular_decagon n in 
  let angle_EFR := 180 - angle_EFG in
  let angle_RAJ := angle_EFR in 
  let reflex_angle_E := reflex_angle angle_EFG in 
  let angle_sum_quad_AEFQ := 360 in 
  angle_sum_quad_AEFQ - angle_RAJ - reflex_angle_E - angle_EFR = 72 := sorry

end angle_Q_is_72_degrees_l772_772024


namespace maximum_value_of_x_minus_y_l772_772696

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l772_772696


namespace triangle_median_equal_bc_l772_772333

-- Let \( ABC \) be a triangle, \( AB = 2 \), \( AC = 3 \), and the median from \( A \) to \( BC \) has the same length as \( BC \).
theorem triangle_median_equal_bc (A B C M : Type) (AB AC BC AM : ℝ) 
  (hAB : AB = 2) (hAC : AC = 3) 
  (hMedian : BC = AM) (hM : M = midpoint B C) :
  BC = real.sqrt (26 / 5) :=
by sorry

end triangle_median_equal_bc_l772_772333


namespace cannot_bisect_segment_with_ruler_l772_772824

noncomputable def projective_transformation (A B M : Point) : Point :=
  -- This definition will use an unspecified projective transformation that leaves A and B invariant
  sorry

theorem cannot_bisect_segment_with_ruler (A B : Point) (method : Point -> Point -> Point) :
  (forall (phi : Point -> Point), phi A = A -> phi B = B -> phi (method A B) ≠ method A B) ->
  ¬ (exists (M : Point), method A B = M) := by
  sorry

end cannot_bisect_segment_with_ruler_l772_772824


namespace max_difference_value_l772_772676

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l772_772676


namespace min_books_borrowed_l772_772764

theorem min_books_borrowed 
    (h1 : 12 * 1 = 12) 
    (h2 : 10 * 2 = 20) 
    (h3 : 2 = 2) 
    (h4 : 32 = 32) 
    (h5 : (32 * 2 = 64))
    (h6 : ∀ x, x ≤ 11) :
    ∃ (x : ℕ), (8 * x = 32) ∧ x ≤ 11 := 
  sorry

end min_books_borrowed_l772_772764


namespace general_term_formula_sum_of_first_2n_terms_l772_772740

-- Definitions and Conditions
def S (n : ℕ) : ℕ := (n^2 + n) / 2
def a (n : ℕ) : ℕ := n
def b (n : ℕ) : ℕ := 2^n + (-1)^n * n
def T (n : ℕ) : ℕ := (∑ k in Finset.range (2*n), b (k + 1))

-- Lean Statements for Proof Problems
theorem general_term_formula :
  ∀ n : ℕ, S n = (∑ k in Finset.range n, a (k + 1)) :=
sorry

theorem sum_of_first_2n_terms (n : ℕ) :
  T n = 2^(2*n + 1) + n - 2 :=
sorry

end general_term_formula_sum_of_first_2n_terms_l772_772740


namespace seq_1000_eq_495_l772_772926

def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else
  let S := ∑ i in Finset.range (n - 1), seq (i + 1) in
  Int.floor (Real.sqrt S)

theorem seq_1000_eq_495 : seq 1000 = 495 := by
  sorry

end seq_1000_eq_495_l772_772926


namespace volume_of_solid_of_revolution_l772_772873

theorem volume_of_solid_of_revolution (a : ℝ) (α : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) :
  let V := 2 * Real.pi * a^3 * Real.sin(α) * Real.sin(α / 2)
  V = 2 * Real.pi * a^3 * Real.sin(α) * Real.sin(α / 2) :=
by
  -- Let V be the volume of the solid of revolution
  let V := 2 * Real.pi * a^3 * Real.sin(α) * Real.sin(α / 2)
  -- Remember to assign the correct value to V
  exact rfl

end volume_of_solid_of_revolution_l772_772873


namespace find_n_l772_772082

open Nat

def digits (n : ℕ) : List ℕ :=
  n.digits 10

def uses_once (ns : List ℕ) (ds : List ℕ) : Prop :=
  ∀ d ∈ ds, (ns.count d = 1)

def is_3_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_5_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem find_n : 
  ∃ n : ℕ, 22 ≤ n ∧ n < 32 ∧ 
  uses_once (digits (n^2) ++ digits (n^3)) [1,2,3,4,5,6,7,8] ∧
  is_3_digit (n^2) ∧
  is_5_digit (n^3) ∧
  n = 24 :=
by
  sorry

end find_n_l772_772082


namespace spider_socks_gloves_shoes_permutations_l772_772136

theorem spider_socks_gloves_shoes_permutations :
  let num_legs := 8
  let num_items_per_leg := 3
  let total_items := num_legs * num_items_per_leg
  -- Constraints: glove before shoe, sock before shoe
  -- Total valid orders
  (nat.fact total_items) / (2 ^ num_legs) = (nat.factorial 24) / (2 ^ 8) :=
sorry

end spider_socks_gloves_shoes_permutations_l772_772136


namespace negation_of_exists_l772_772449

theorem negation_of_exists (h : ¬ (∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0)) : ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
by
  sorry

end negation_of_exists_l772_772449


namespace speed_of_boat_in_still_water_l772_772923

variable (b s : ℝ) -- Speed of the boat in still water and speed of the stream

-- Condition 1: The boat goes 9 km along the stream in 1 hour
def boat_along_stream := b + s = 9

-- Condition 2: The boat goes 5 km against the stream in 1 hour
def boat_against_stream := b - s = 5

-- Theorem to prove: The speed of the boat in still water is 7 km/hr
theorem speed_of_boat_in_still_water : boat_along_stream b s → boat_against_stream b s → b = 7 := 
by
  sorry

end speed_of_boat_in_still_water_l772_772923


namespace sum_of_areas_l772_772186

theorem sum_of_areas (T1_area S1_area : ℝ) (T2_factor S2_factor : ℝ) :
  T1_area = 9 → S1_area = 25 →
  T2_factor = 4 → S2_factor = 2 →
  let T2_area := T1_area / T2_factor in
  let T3_area := T2_area / T2_factor in
  let S2_area := S1_area / S2_factor in
  let S3_area := S2_area / S2_factor in
  T3_area + S3_area = 6.8125 :=
by {
  intros T1_area_eq S1_area_eq T2_factor_eq S2_factor_eq,
  let T2_area := T1_area / T2_factor,
  let T3_area := T2_area / T2_factor,
  let S2_area := S1_area / S2_factor,
  let S3_area := S2_area / S2_factor,
  have T2_area_eq : T2_area = 9 / 4,
  rw T1_area_eq,
  rw ← T2_factor_eq,
  norm_num,
  have T3_area_eq : T3_area = (9 / 4) / 4,
  rw T2_area_eq,
  rw ← T2_factor_eq,
  norm_num,
  have S2_area_eq : S2_area = 25 / 2,
  rw S1_area_eq,
  rw ← S2_factor_eq,
  norm_num,
  have S3_area_eq : S3_area = (25 / 2) / 2,
  rw S2_area_eq,
  rw ← S2_factor_eq,
  norm_num,
  have : T3_area + S3_area = 6.8125,
  rw [T3_area_eq, S3_area_eq],
  norm_num,
  exact this,
  sorry
}

end sum_of_areas_l772_772186


namespace arithmetic_sequence_property_l772_772771

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
                                     (h2 : a 3 + a 11 = 40) :
  a 6 - a 7 + a 8 = 20 :=
by
  sorry

end arithmetic_sequence_property_l772_772771


namespace inequality_solution_l772_772734

-- Definition of the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) / (x + b)

-- Hypotheses
variables {a b : ℝ}

-- Given conditions
axiom h1 : (a < 0)
axiom h2 : (1 / a = -1)
axiom h3 : (-b = 3)

-- Desired inequality to solve
def g (x : ℝ) : ℝ := (2 * x - 1) / (-2 * x - 3)
def solution_set (x : ℝ) : Prop := (x > 1/2) ∨ (x < -3/2)

-- The theorem to prove
theorem inequality_solution :
    ∀ x : ℝ, g x < 0 ↔ solution_set x :=
by sorry

end inequality_solution_l772_772734


namespace lines_intersect_at_point_l772_772126

-- Define the first line
def line1 (t : ℚ) : ℚ × ℚ :=
  (2 + 3 * t, 2 - 4 * t)

-- Define the second line
def line2 (u : ℚ) : ℚ × ℚ :=
  (4 + 5 * u, -10 + 3 * u)

-- Define the point of intersection
def intersection_point : ℚ × ℚ :=
  (184 / 11, -194 / 11)

-- The theorem to prove the intersection points are equal
theorem lines_intersect_at_point :
  ∃ (t u : ℚ), line1 t = intersection_point ∧ line2 u = intersection_point :=
by {
  use (54 / 11),
  use (28 / 11),
  split;
  calc
    line1 (54 / 11) = intersection_point : by sorry
    line2 (28 / 11) = intersection_point : by sorry
}

end lines_intersect_at_point_l772_772126


namespace time_in_1876_minutes_from_6AM_is_116PM_l772_772857

def minutesToTime (startTime : Nat) (minutesToAdd : Nat) : Nat × Nat :=
  let totalMinutes := startTime + minutesToAdd
  let totalHours := totalMinutes / 60
  let remainderMinutes := totalMinutes % 60
  let resultHours := (totalHours % 24)
  (resultHours, remainderMinutes)

theorem time_in_1876_minutes_from_6AM_is_116PM :
  minutesToTime (6 * 60) 1876 = (13, 16) :=
  sorry

end time_in_1876_minutes_from_6AM_is_116PM_l772_772857


namespace triangle_side_length_l772_772345

theorem triangle_side_length (A B C M : Point)
  (hAB : dist A B = 2)
  (hAC : dist A C = 3)
  (hMidM : M = midpoint B C)
  (hAM_BC : dist A M = dist B C) :
  dist B C = Real.sqrt (78) / 3 :=
by
  sorry

end triangle_side_length_l772_772345


namespace max_difference_value_l772_772678

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l772_772678


namespace find_x_value_l772_772864

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x)
  else Real.log x * Real.log 81

theorem find_x_value (x : ℝ) (h : f x = 1 / 4) : x = 3 :=
sorry

end find_x_value_l772_772864


namespace equal_area_triangles_l772_772383

variables {A B C D E F O : Type} [AffineTriangle A B C]
  (AD BE CF : Line) -- Altitudes
  (OA OB OC OD OE OF : Segment) -- Segments

-- Conditions
variable (h1 : AcuteTriangle A B C)
variable (h2 : IsAltitude AD)
variable (h3 : IsAltitude BE)
variable (h4 : IsAltitude CF)
variable (h5 : IsCircumcenter O A B C)

-- Question
theorem equal_area_triangles (h6 : Contains OA A) 
  (h7 : Contains OA O)
  (h8 : Contains OF F)
  (h9 : Contains OF O)
  (h10 : Contains OB B) 
  (h11 : Contains OB O)
  (h12 : Contains OD D)
  (h13 : Contains OD O)
  (h14 : Contains OC C) 
  (h15 : Contains OC O)
  (h16 : Contains OE E)
  (h17 : Contains OE O) :
  Area (Triangle A O E) = Area (Triangle B O D) ∧ 
  Area (Triangle B O F) = Area (Triangle C O E) ∧
  Area (Triangle C O D) = Area (Triangle A O F) := 
sorry

end equal_area_triangles_l772_772383


namespace measure_angle_BDC_l772_772396

-- Definitions based on given problem conditions.
variable {A B C D : Type} [add_group A] [add_group B] [add_group C] [add_group D]

-- Given conditions:
def is_right_triangle (A B C : Type) (BAC : A) : Prop := BAC = 90

def exterior_angle_bisectors_meet_at (B C D : Type) (point : D) : Prop := sorry

-- Theorem statement:
theorem measure_angle_BDC (A B C D : Type) (BAC : A) (angle_BDC : D) 
  (h1 : is_right_triangle A B C BAC) 
  (h2 : exterior_angle_bisectors_meet_at B C D angle_BDC) :
  angle_BDC = 45 := sorry

end measure_angle_BDC_l772_772396


namespace triangle_proof_l772_772801

-- Defining the properties of triangle ABC
variable (A B C : Type) -- A, B, and C are points
variable [P : PointType A B C] -- Make use of a PointType class for triangle properties
variable (AB AC BC : ℝ) -- Define side lengths as real numbers

-- Introduction of the lengths
def side_AB : AB = 5 := sorry
def side_BC : BC = 4 := sorry

-- Definitions of angles at respective points
variable (angle_ABC angle_BAC angle_ACB : ℝ)

-- Propose statements
-- Statement (a)
def isosceles_triangle (AC : ℝ) (hAC : AC = 4) : angle_ABC > angle_BAC := sorry
-- Statement (c)
def degenerate_triangle (AC : ℝ) (hAC : AC = 2) : angle_ABC < angle_ACB := sorry
-- Proving statements a and c are incorrect
theorem triangle_proof : triangle_angles AB 5 BC 4 AC 4 ∧ isosceles_triangle AB BC AC → False ∧ 
                         triangle_angles AB 5 BC 4 AC 2 ∧ degenerate_triangle AB BC AC → False := 
begin
  sorry
end

end triangle_proof_l772_772801


namespace ellipse_through_points_eccentricity_range_circle_existence_l772_772800

noncomputable def ellipse_eq (a b x y : ℝ) :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_through_points (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (ellipse_eq a b 2 (Real.sqrt 2)) ∧ (ellipse_eq a b (Real.sqrt 6) 1) ↔
  (a = Real.sqrt 8) ∧ (b = Real.sqrt 4) :=
  sorry

theorem eccentricity_range (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a > b) :
  (∃ (M : ℝ × ℝ), ellipse_eq a b M.1 M.2 ∧ ((M.1 + c) * (M.1 - c) + M.2^2 = 0)) →
  (c / a ≥ Real.sqrt 2 / 2) ∧ (c / a < 1) :=
  sorry

theorem circle_existence (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hellipse_eq : ellipse_eq a b 2 (Real.sqrt 2) ∧ ellipse_eq a b (Real.sqrt 6) 1) :
  ∃ r : ℝ, ∀ (A B : ℝ × ℝ), (A.1^2 + A.2^2 = r^2) → (B.1^2 + B.2^2 = r^2) →
  (ellipse_eq a b A.1 A.2) ∧ (ellipse_eq a b B.1 B.2) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 ∈ set.Icc (4 * Real.sqrt 6 / 3) (2 * Real.sqrt 3)) :=
  sorry

end ellipse_through_points_eccentricity_range_circle_existence_l772_772800


namespace ratio_volume_tetrahedrons_l772_772717

-- Let’s define the information and conditions given in the problem
variables (A B C D P Q R S : Type)
variables [affine_space ℝ A B C D P Q R S]

-- Ratios conditions
def PA_PB : ℝ := 1
def AS_SC : ℝ := 1
def BQ_QD : ℝ := 1 / 2
def CR_RD : ℝ := 1 / 2

-- The statement to prove
theorem ratio_volume_tetrahedrons (V1 V2 VABCD : ℝ) 
  (H1 : ∃ t : ℝ, t ∈ {PA_PB, AS_SC, BQ_QD, CR_RD}) 
  (H2 : V1 = (1 / 6 + 1 / 12 + 1 / 9) * VABCD)
  (H3 : V2 = VABCD - V1) : 
  V1 / V2 = 13 / 23 := by
  sorry

end ratio_volume_tetrahedrons_l772_772717


namespace smallest_four_digit_number_meeting_conditions_l772_772898

-- Define the conditions
def four_digit_integer_with_all_different_digits (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
  ∀ d1 d2, d1 ∈ digits → d2 ∈ digits → d1 ≠ d2

def digits_greater_than_one (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
  ∀ d, d ∈ digits → d > 1

def divisible_by_each_digit (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
  ∀ d, d ∈ digits → n % d = 0

-- Main proof statement
theorem smallest_four_digit_number_meeting_conditions : ∃ n : ℕ, 
  four_digit_integer_with_all_different_digits n ∧
  digits_greater_than_one n ∧
  divisible_by_each_digit n ∧
  n = 3246 :=
by sorry

end smallest_four_digit_number_meeting_conditions_l772_772898


namespace quotient_of_division_l772_772041

theorem quotient_of_division (L S Q : ℕ) (h1 : L - S = 2500) (h2 : L = 2982) (h3 : L = Q * S + 15) : Q = 6 := 
sorry

end quotient_of_division_l772_772041


namespace sqrt_nested_ineq_l772_772828

theorem sqrt_nested_ineq (n : ℕ) (h_pos : 0 < n) : 
  (sqrt (2 * sqrt (3 * sqrt (4 * ... * sqrt (n))))) < 3 := 
sorry

end sqrt_nested_ineq_l772_772828


namespace solve_price_per_litre_second_oil_l772_772116

variable (P : ℝ)

def price_per_litre_second_oil :=
  10 * 55 + 5 * P = 15 * 58.67

theorem solve_price_per_litre_second_oil (h : price_per_litre_second_oil P) : P = 66.01 :=
  by
  sorry

end solve_price_per_litre_second_oil_l772_772116


namespace marbles_problem_l772_772081

theorem marbles_problem (initial_marble_tyrone : ℕ) (initial_marble_eric : ℕ) (x : ℝ)
  (h1 : initial_marble_tyrone = 125)
  (h2 : initial_marble_eric = 25)
  (h3 : initial_marble_tyrone - x = 3 * (initial_marble_eric + x)) :
  x = 12.5 := 
sorry

end marbles_problem_l772_772081


namespace product_of_two_numbers_l772_772440

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 340) : x * y = 97.9450625 :=
by
  sorry

end product_of_two_numbers_l772_772440


namespace smallest_possible_b_l772_772853

theorem smallest_possible_b 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a - b = 8) 
  (h4 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  b = 4 := 
sorry

end smallest_possible_b_l772_772853


namespace positional_relationship_b_c_l772_772629

variable (a b c : Line) (α β : Plane) (l : Line)
variables (proj_α : ∀ (x : Point), Point) (proj_β : ∀ (x : Point), Point)
variables (A B C : Point)

-- Given conditions
axiom Hαβ : α ∩ β = l
axiom Hα : ¬(a ⊆ α)
axiom Hβ : ¬(a ⊆ β)
axiom Ha_α : ∀ x, proj_α (point_on_line a x) = point_on_line b x
axiom Ha_β : ∀ x, proj_β (point_on_line a x) = point_on_line c x

-- Proof statement
theorem positional_relationship_b_c : (∃ (P : Point), P ∈ b ∧ P ∈ c) ∨ (∃ (v : Vector), parallel b c v) ∨ skew b c :=
sorry

end positional_relationship_b_c_l772_772629


namespace prove_f_when_x_positive_l772_772176

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := if x < 0 then x^2 - x else -x^2 - x

theorem prove_f_when_x_positive (x : ℝ) (h : x > 0) :
  odd_function f ∧ (∀ x : ℝ, x < 0 → f x = x^2 - x) → f x = -x^2 - x :=
by
  sorry

end prove_f_when_x_positive_l772_772176


namespace sqrt_three_irrational_and_in_range_l772_772911

theorem sqrt_three_irrational_and_in_range : irrational (sqrt 3) ∧ 0 < sqrt 3 ∧ sqrt 3 < 3 := 
by 
  sorry

end sqrt_three_irrational_and_in_range_l772_772911


namespace sitio_proof_l772_772995

theorem sitio_proof :
  (∃ t : ℝ, t = 4 + 7 + 12 ∧ 
    (∃ f : ℝ, 
      (∃ s : ℝ, s = 6 + 5 + 10 ∧ t = 23 ∧ f = 23 - s) ∧
      f = 2) ∧
    (∃ cost_per_hectare : ℝ, cost_per_hectare = 2420 / (4 + 12) ∧ 
      (∃ saci_spent : ℝ, saci_spent = 6 * cost_per_hectare ∧ saci_spent = 1320))) :=
by sorry

end sitio_proof_l772_772995


namespace isosceles_triangle_angle_l772_772461

theorem isosceles_triangle_angle (x : ℝ) (hx : x ∈ Ioo 0 (π/2)) :
  (sin x = sin (π / 8)) ∧ (sin x = sin (5 * π / 8)) ∧ (3 * x = π / 4) → x = π / 4 :=
begin
  assume h,
  have h1 : sin x = sin (π / 4),
  { rw h.1 at h.2, assumption, },
  sorry
end

end isosceles_triangle_angle_l772_772461


namespace smallest_k_for_n_points_in_unit_square_l772_772368

noncomputable def smallest_k (n : ℕ) : ℕ :=
2 * n + 2

theorem smallest_k_for_n_points_in_unit_square (n : ℕ) (S : set (ℝ × ℝ)) 
  (hS : S ⊆ { p : ℝ × ℝ | 0 < p.1 ∧ p.1 < 1 ∧ 0 < p.2 ∧ p.2 < 1 } ∧ S.finite ∧ S.card = n) :
  ∃ (k : ℕ) (rectangles : set (set (ℝ × ℝ))), k = smallest_k n ∧
    (∀ r ∈ rectangles, ∀ p ∈ r, 0 < p.1 ∧ p.1 < 1 ∧ 0 < p.2 ∧ p.2 < 1) ∧
    (∀ p ∈ S, ∀ r ∈ rectangles, p ∉ interior r) ∧
    (∀ p : ℝ × ℝ, (0 < p.1 ∧ p.1 < 1 ∧ 0 < p.2 ∧ p.2 < 1) → 
      (p ∈ ⋃ r ∈ rectangles, interior r ∨ p ∈ S)) := sorry

end smallest_k_for_n_points_in_unit_square_l772_772368


namespace inverse_g_neg1_l772_772385

noncomputable def g (c d x : ℝ) : ℝ := 1 / (c * x + d)

theorem inverse_g_neg1 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ y : ℝ, g c d y = -1 ∧ y = (-1 - d) / c := 
by
  unfold g
  sorry

end inverse_g_neg1_l772_772385


namespace quadratic_has_two_distinct_real_roots_l772_772755

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ((k - 1) * x^2 + 2 * x - 2 = 0) → (1 / 2 < k ∧ k ≠ 1) :=
sorry

end quadratic_has_two_distinct_real_roots_l772_772755


namespace correct_transformation_l772_772492

theorem correct_transformation (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end correct_transformation_l772_772492


namespace brett_red_marbles_l772_772565

variables (r b : ℕ)

-- Define the conditions
axiom h1 : b = r + 24
axiom h2 : b = 5 * r

theorem brett_red_marbles : r = 6 :=
by
  sorry

end brett_red_marbles_l772_772565


namespace probability_diamond_first_and_ace_or_king_second_l772_772891

-- Define the condition of the combined deck consisting of two standard decks (104 cards total)
def two_standard_decks := 104

-- Define the number of diamonds, aces, and kings in the combined deck
def number_of_diamonds := 26
def number_of_aces := 8
def number_of_kings := 8

-- Define the events for drawing cards
def first_card_is_diamond := (number_of_diamonds : ℕ) / (two_standard_decks : ℕ)
def second_card_is_ace_or_king_if_first_is_not_ace_or_king :=
  (16 / 103 : ℚ) -- 16 = 8 (aces) + 8 (kings)
def second_card_is_ace_or_king_if_first_is_ace_or_king :=
  (15 / 103 : ℚ) -- 15 = 7 (remaining aces) + 7 (remaining kings) + 1 (remaining ace or king of the same suit)

-- Define the probabilities of the combined event
def probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king :=
  (22 / 104) * (16 / 103)
def probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king :=
  (4 / 104) * (15 / 103)

-- Define the total probability combining both events
noncomputable def total_probability :=
  probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king +
  probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king

-- Theorem stating the desired probability result
theorem probability_diamond_first_and_ace_or_king_second :
  total_probability = (103 / 2678 : ℚ) :=
sorry

end probability_diamond_first_and_ace_or_king_second_l772_772891


namespace nested_inverse_value_l772_772854

def f (x : ℝ) : ℝ := 5 * x + 6

noncomputable def f_inv (y : ℝ) : ℝ := (y - 6) / 5

theorem nested_inverse_value :
  f_inv (f_inv 16) = -4/5 :=
by
  sorry

end nested_inverse_value_l772_772854


namespace distance_point_focus_l772_772752

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_point_focus (M : ℝ × ℝ)
  (h1 : M.2^2 = 3 * M.1)
  (h2 : distance M (0, 0) = 2) :
  distance M (3/4, 0) = 7/4 :=
sorry

end distance_point_focus_l772_772752


namespace car_X_average_speed_l772_772980

theorem car_X_average_speed:
  ∃ (V_x : ℝ), 
  let T_y := 1.2 in -- Time delay in hours before Car Y starts traveling
  let V_y := 42 in -- Speed of Car Y in miles per hour
  let D := 210 in -- Distance traveled by Car X from the time Car Y starts until both cars stop
  let T_x := (D / V_y + T_y) in -- Time for Car X to travel, accounting for the delay of Car Y
  D = V_x * T_x → 
  V_x = 33.87 := 
sorry

end car_X_average_speed_l772_772980


namespace simplify_eval_expr_l772_772029

noncomputable def a : ℝ := (Real.sqrt 2) + 1
noncomputable def b : ℝ := (Real.sqrt 2) - 1

theorem simplify_eval_expr (a b : ℝ) (ha : a = (Real.sqrt 2) + 1) (hb : b = (Real.sqrt 2) - 1) : 
  (a^2 - b^2) / a / (a + (2 * a * b + b^2) / a) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_eval_expr_l772_772029


namespace maximum_value_sqrt_expression_l772_772380

theorem maximum_value_sqrt_expression (x y z : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : x + y + z = 5) :
  √(2*x + 1) + √(2*y + 1) + √(2*z + 1) ≤ √39 :=
by sorry

end maximum_value_sqrt_expression_l772_772380


namespace max_difference_value_l772_772675

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l772_772675


namespace rational_roots_of_polynomial_with_integer_coefficients_are_integers_l772_772458

theorem rational_roots_of_polynomial_with_integer_coefficients_are_integers 
  {n : ℕ} {a : Fin n → ℤ} (x : ℚ) :
  (eval x (∑ i in (Fin.range n), (a i) * x^((Fin.range n).to_finset.val.find i).val)) = 0 → x ∈ ℤ :=
by
  sorry

end rational_roots_of_polynomial_with_integer_coefficients_are_integers_l772_772458


namespace part1_part1_axis_part2_max_part2_min_l772_772733

noncomputable def f (x : ℝ) : ℝ := 
  Math.cos x * Math.sin (x + Real.pi / 3) - Real.sqrt 3 * (Math.cos x)^2 + Real.sqrt 3 / 4 - 1

theorem part1 : (∀ x : ℝ, f(x + Real.pi) = f(x)) := 
by sorry

theorem part1_axis : 
  ∃ k : ℤ, (∀ x : ℝ, f x = f (x + Real.pi)) → 
  (2 * x - Real.pi / 3) = 0 → (x = 5 * Real.pi / 12 + k * Real.pi) := 
by sorry

theorem part2_max : 
  is_max_on f (x : ℝ) (x = Real.pi / 4) ∧ f (Real.pi / 4) = -3 / 4 := 
by sorry

theorem part2_min : 
  is_min_on f (x : ℝ) (x = - Real.pi / 12) ∧ f (- Real.pi / 12) = -3 / 2 := 
by sorry

end part1_part1_axis_part2_max_part2_min_l772_772733


namespace find_m_l772_772269

-- Define the given equations of the lines
def line1 (m : ℝ) : ℝ × ℝ → Prop := fun p => (3 + m) * p.1 - 4 * p.2 = 5 - 3 * m
def line2 : ℝ × ℝ → Prop := fun p => 2 * p.1 - p.2 = 8

-- Define the condition for parallel lines based on the given equations
def are_parallel (m : ℝ) : Prop := (3 + m) / 4 = 2

-- The main theorem stating the value of m
theorem find_m (m : ℝ) (h1 : ∀ p : ℝ × ℝ, line1 m p) (h2 : ∀ p : ℝ × ℝ, line2 p) (h_parallel : are_parallel m) : m = 5 :=
sorry

end find_m_l772_772269


namespace min_num_people_max_num_people_l772_772913

-- Define the conditions as hypotheses
variables (num_students : ℕ)
variables (num_junior_students : ℕ)
variables (num_teachers : ℕ)
variables (num_table_tennis_enthusiasts : ℕ)
variables (num_basketball_enthusiasts : ℕ)

-- The given values in the problem
def num_students := 6
def num_junior_students := 4
def num_teachers := 2
def num_table_tennis_enthusiasts := 5
def num_basketball_enthusiasts := 2

-- Prove the minimum number of people interviewed is 8
theorem min_num_people : 
  num_students + num_teachers = 8 := by sorry 

-- Prove the maximum number of people interviewed is 15
theorem max_num_people : 
  num_students - num_junior_students + num_teachers + num_table_tennis_enthusiasts + 
  num_basketball_enthusiasts + num_junior_students = 15 := by sorry

end min_num_people_max_num_people_l772_772913


namespace maximum_value_of_x_minus_y_l772_772700

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l772_772700


namespace sin_double_angle_computation_l772_772723

theorem sin_double_angle_computation (x y : ℝ) (h : x^2 + y^2 ≠ 0) (hx : x = -2) (hy : y = 1) :
  ∃ α : ℝ, let r := real.sqrt (x^2 + y^2) in 
  (real.sin (2 * real.atan2 y x) = 2 * (y / r) * (x / r)) ∧
  (1 / (2 * (y / r) * (x / r)) = -5 / 4) :=
by
  sorry

end sin_double_angle_computation_l772_772723


namespace seating_arrangements_l772_772072

def total_seats_front := 11
def total_seats_back := 12
def middle_seats_front := 3

def number_of_arrangements := 334

theorem seating_arrangements: 
  (total_seats_front - middle_seats_front) * (total_seats_front - middle_seats_front - 1) / 2 +
  (total_seats_back * (total_seats_back - 1)) / 2 +
  (total_seats_front - middle_seats_front) * total_seats_back +
  total_seats_back * (total_seats_front - middle_seats_front) = number_of_arrangements := 
sorry

end seating_arrangements_l772_772072


namespace cost_of_single_room_l772_772151

theorem cost_of_single_room
  (total_rooms : ℕ)
  (double_rooms : ℕ)
  (cost_double_room : ℕ)
  (revenue_total : ℕ)
  (cost_single_room : ℕ)
  (H1 : total_rooms = 260)
  (H2 : double_rooms = 196)
  (H3 : cost_double_room = 60)
  (H4 : revenue_total = 14000)
  (H5 : revenue_total = (total_rooms - double_rooms) * cost_single_room + double_rooms * cost_double_room)
  : cost_single_room = 35 :=
sorry

end cost_of_single_room_l772_772151


namespace sin_130_eq_sin_50_l772_772554

theorem sin_130_eq_sin_50 : sin (130 * real.pi / 180) = sin (50 * real.pi / 180) :=
by 
  have h : rad_to_deg (real.pi) = 180 := sorry
  rw [←real.sin_sub_add_eq_sin],
  rw [h],
  rw [sub_add_eq],
  rw [real.sin_add (50 * real.pi / 180) (real.pi / 2)],
  simp,
  exact sin_coe_pi_mul_half

end sin_130_eq_sin_50_l772_772554


namespace find_m_l772_772270

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, real.sqrt 3)

-- Define the dot product calculation
def dot_product (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2)

-- The main statement to prove
theorem find_m (m : ℝ) (h : dot_product (1, m) (3, real.sqrt 3) = 6) : m = real.sqrt 3 :=
sorry

end find_m_l772_772270


namespace triangle_sum_l772_772427

-- Define the triangle operation
def triangle (a b c : ℕ) : ℕ := a + b + c

-- State the theorem
theorem triangle_sum :
  triangle 2 4 3 + triangle 1 6 5 = 21 :=
by
  sorry

end triangle_sum_l772_772427


namespace puppy_weight_is_3_8_l772_772536

noncomputable def puppy_weight_problem (p s l : ℝ) : Prop :=
  p + 2 * s + l = 38 ∧
  p + l = 3 * s ∧
  p + 2 * s = l

theorem puppy_weight_is_3_8 :
  ∃ p s l : ℝ, puppy_weight_problem p s l ∧ p = 3.8 :=
by
  sorry

end puppy_weight_is_3_8_l772_772536


namespace probability_of_forming_word_l772_772470

theorem probability_of_forming_word (cards : Fin 8 → String) (h : ∀ i, i ∈ finset.univ → cards i ∈ ["З", "О", "О", "Л", "О", "Г", "И", "Я"]) :
  ((nat.factorial 3) / (nat.factorial 8)) = 1 / 6720 :=
by
  sorry

end probability_of_forming_word_l772_772470


namespace c_alone_finishes_in_60_days_l772_772099

-- Definitions for rates of work
variables (A B C : ℝ)

-- The conditions given in the problem
-- A and B together can finish the job in 15 days
def condition1 : Prop := A + B = 1 / 15
-- A, B, and C together can finish the job in 12 days
def condition2 : Prop := A + B + C = 1 / 12

-- The statement to prove: C alone can finish the job in 60 days
theorem c_alone_finishes_in_60_days 
  (h1 : condition1 A B) 
  (h2 : condition2 A B C) : 
  (1 / C) = 60 :=
by
  sorry

end c_alone_finishes_in_60_days_l772_772099


namespace sqrt_two_half_not_in_interval_l772_772021

theorem sqrt_two_half_not_in_interval
    (p q : ℕ)
    (h_coprime : Nat.coprime p q)
    (h_q_pos : 0 < q)
    (h_pq_interval : 0 < (p : ℝ) / q ∧ (p : ℝ) / q < 1) :
    ∀ δ : ℝ, δ ∈ interval (p : ℝ) / q - 1 / (4 * q^2) (p : ℝ) / q + 1 / (4 * q^2) 
        → abs δ ≥ 1 → (δ ≠ √2 / 2) :=
sorry

end sqrt_two_half_not_in_interval_l772_772021


namespace ratio_PR_QS_l772_772821

noncomputable def PQ : ℝ := 3
noncomputable def QR : ℝ := 7
noncomputable def PS : ℝ := 18

def PR := PQ + QR
def QS := PS - PQ

theorem ratio_PR_QS (h₁ : PQ = 3) (h₂ : QR = 7) (h₃ : PS = 18) :
  PR / QS = (2 : ℝ) / (3 : ℝ) :=
by
  sorry

end ratio_PR_QS_l772_772821


namespace solutions_h_eq_4_l772_772391

def h (x : ℝ) : ℝ :=
if x < 0 then 4 * x + 8 else 3 * x - 12

theorem solutions_h_eq_4 : {x : ℝ | h x = 4} = {-1, 16 / 3} :=
by
  sorry

end solutions_h_eq_4_l772_772391


namespace max_x_minus_y_l772_772671

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772671


namespace parabola_vertex_sum_l772_772117

variable (a b c : ℝ)

def parabola_eq (x y : ℝ) : Prop :=
  x = a * y^2 + b * y + c

def vertex (v : ℝ × ℝ) : Prop :=
  v = (-3, 2)

def passes_through (p : ℝ × ℝ) : Prop :=
  p = (-1, 0)

theorem parabola_vertex_sum :
  ∀ (a b c : ℝ),
  (∃ v : ℝ × ℝ, vertex v) ∧
  (∃ p : ℝ × ℝ, passes_through p) →
  a + b + c = -7/2 :=
by
  intros a b c
  intro conditions
  sorry

end parabola_vertex_sum_l772_772117


namespace triangle_side_eq_median_l772_772338

theorem triangle_side_eq_median (A B C : Type) (a b c : ℝ) (hAB : a = 2) (hAC : b = 3) (hBC_eq_median : c = (2 * (Real.sqrt (13 / 10)))) :
  c = (Real.sqrt 130) / 5 := by
  sorry

end triangle_side_eq_median_l772_772338


namespace visited_iceland_l772_772766

variable (total : ℕ) (visitedNorway : ℕ) (visitedBoth : ℕ) (visitedNeither : ℕ)

theorem visited_iceland (h_total : total = 50)
                        (h_visited_norway : visitedNorway = 23)
                        (h_visited_both : visitedBoth = 21)
                        (h_visited_neither : visitedNeither = 23) :
                        (total - (visitedNorway - visitedBoth + visitedNeither) = 25) :=
  sorry

end visited_iceland_l772_772766


namespace exists_irrational_between_zero_and_three_l772_772909

theorem exists_irrational_between_zero_and_three : ∃ x : ℝ, (0 < x) ∧ (x < 3) ∧ irrational x :=
sorry

end exists_irrational_between_zero_and_three_l772_772909


namespace sum_of_polynomials_l772_772047

theorem sum_of_polynomials (p q : ℝ → ℝ) 
  (hq : q 2 = 2) 
  (hp : p 3 = 3) 
  (hq_quadratic : ∃ a b c, ∀ x, q x = a * x^2 + b * x + c) 
  (hp_linear : ∃ a b, ∀ x, p x = a * x + b)
  (hq_factors : ∃ a b, q x = a * x * (x - 1) + b): 
  ∀ x, p x + q x = x^2 :=
begin
  sorry -- Proof goes here
end

end sum_of_polynomials_l772_772047


namespace max_value_x_minus_y_l772_772691

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l772_772691


namespace quadratic_real_root_count_l772_772754

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b * b - 4 * a * c

theorem quadratic_real_root_count: 
  (∃ k : ℕ, discriminant (k-2) (-2) 1 > 0 ∧ k ≠ 2) ↔ fintype.card { k : ℕ | discriminant (k-2) (-2) 1 > 0 ∧ k ≠ 2 } = 2 :=
by 
  sorry

end quadratic_real_root_count_l772_772754


namespace intersection_points_count_l772_772714

noncomputable def f : ℝ → ℝ :=
  λ x, if x % 2 = x then x^3 - x else (x % 2)^3 - (x % 2)

theorem intersection_points_count :
  let period := 2 in
  let intervals := (0,1) ∪ (1,1 + period) ∪ (1 + period, 2 + period) in
  finset.card (finset.filter (λ x, f x= 0) (finset.range (nat.floor period + 1))) = 6 + 1 :=
begin
  sorry
end

end intersection_points_count_l772_772714


namespace concyclic_projections_of_concyclic_quad_l772_772796

variables {A B C D A' B' C' D' : Type*}

def are_concyclic (p1 p2 p3 p4: Type*) : Prop :=
  sorry -- Assume we have a definition for concyclic property of points

def are_orthogonal_projection (x y : Type*) (l : Type*) : Type* :=
  sorry -- Assume we have a definition for orthogonal projection of a point on line

theorem concyclic_projections_of_concyclic_quad
  (hABCD : are_concyclic A B C D)
  (hA'_proj : are_orthogonal_projection A A' (BD))
  (hC'_proj : are_orthogonal_projection C C' (BD))
  (hB'_proj : are_orthogonal_projection B B' (AC))
  (hD'_proj : are_orthogonal_projection D D' (AC)) :
  are_concyclic A' B' C' D' :=
sorry

end concyclic_projections_of_concyclic_quad_l772_772796


namespace point_B_coordinates_l772_772411

theorem point_B_coordinates {A B : (ℝ × ℝ)} (hA : A = (2, -1)) (hB : B = (A.1 - 3, A.2 + 4)) :
  B = (-1, 3) :=
by
  intro A
  intro B
  simp [hA, hB]
  sorry

end point_B_coordinates_l772_772411


namespace opposite_2024_eq_neg_2024_l772_772452

def opposite (n : ℤ) : ℤ := -n

theorem opposite_2024_eq_neg_2024 : opposite 2024 = -2024 :=
by
  sorry

end opposite_2024_eq_neg_2024_l772_772452


namespace trig_equation_solution_l772_772845

theorem trig_equation_solution (x : ℝ) (h : -π ≤ x ∧ x ≤ π) : 
    cos (sin x) = sin (cos (x / 3)) → x = 0 :=
sorry

end trig_equation_solution_l772_772845


namespace betty_needs_more_money_l772_772157

-- Define the variables and conditions
def wallet_cost : ℕ := 100
def parents_gift : ℕ := 15
def grandparents_gift : ℕ := parents_gift * 2
def initial_betty_savings : ℕ := wallet_cost / 2
def total_savings : ℕ := initial_betty_savings + parents_gift + grandparents_gift

-- Prove that Betty needs 5 more dollars to buy the wallet
theorem betty_needs_more_money : total_savings + 5 = wallet_cost :=
by
  sorry

end betty_needs_more_money_l772_772157


namespace sodium_hydride_reaction_with_water_l772_772274

-- Define the basic chemical reactions
def first_reaction := "NaH + H2O -> NaOH + H2"
def second_reaction := "2 NaH + CO2 -> Na2CO3 + 2 H2"

-- Define the number of moles of reactants and products
def moles_NaH_needed (H2O_moles : ℕ) : ℕ :=
  H2O_moles

-- Theorem stating the amount of Sodium hydride required to react with 1 mole of Water
theorem sodium_hydride_reaction_with_water : moles_NaH_needed 1 = 1 :=
  begin
    sorry
  end

end sodium_hydride_reaction_with_water_l772_772274


namespace integral_calculation_l772_772107

noncomputable def integral_result (x : ℝ) : ℝ := 
x^2 - Real.log (Real.abs x) + (1 / 9) * Real.log (Real.abs (x - 4)) - (1 / 9) * Real.log (Real.abs (x + 5)) + C

theorem integral_calculation :
  ∃ C : ℝ, ∫ (λ x : ℝ, (2 * x^4 + 2 * x^3 - 41 * x^2 + 20) / (x * (x - 4) * (x + 5))) dx = integral_result x :=
by sorry

end integral_calculation_l772_772107


namespace part1_proof_part2_proof_l772_772979

-- Part (1): Proving the equality
theorem part1_proof : 
  (1 / Real.sqrt 0.04) + ((1 / Real.sqrt 27)^(1 / 3)) + ((Real.sqrt 2 + 1)^(-1)) - (2^(1 / 2)) + ((-2)^0) = 8 := 
by sorry

-- Part (2): Proving the equality
theorem part2_proof : 
  (2 / 5 * Real.log 32) + (Real.log 50) + Real.sqrt((Real.log 3)^2 - (Real.log 9) + 1) - (Real.log (2 / 3)) = 3 := 
by sorry

end part1_proof_part2_proof_l772_772979


namespace find_length_of_field_l772_772946

variables (L : ℝ) -- Length of the field
variables (width_field : ℝ := 55) -- Width of the field, given as 55 meters.
variables (width_path : ℝ := 2.5) -- Width of the path around the field, given as 2.5 meters.
variables (area_path : ℝ := 1200) -- Area of the path, given as 1200 square meters.

theorem find_length_of_field
  (h : area_path = (L + 2 * width_path) * (width_field + 2 * width_path) - L * width_field)
  : L = 180 :=
by sorry

end find_length_of_field_l772_772946


namespace income_of_fourth_member_l772_772879

theorem income_of_fourth_member 
  (num_members : ℕ)
  (avg_income : ℝ)
  (income1 income2 income3 : ℝ)
  (total_income : ℝ) :
  num_members = 4 →
  avg_income = 10000 →
  income1 = 8000 →
  income2 = 15000 →
  income3 = 6000 →
  total_income = avg_income * num_members →
  (total_income - (income1 + income2 + income3)) = 11000 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  sorry

end income_of_fourth_member_l772_772879


namespace area_of_midpoint_quadrilateral_l772_772432

theorem area_of_midpoint_quadrilateral : 
  ∀ (d1 d2 : ℝ) (angle : ℝ), 
  d1 = 6 → d2 = 8 → angle = π/4 → 
  let s1 := d1 / 2 in 
  let s2 := d2 / 2 in 
  let area := s1 * s2 * Real.sin angle in 
  area = 6 * Real.sqrt 2 :=
by
  intros d1 d2 angle h1 h2 h3
  let s1 := d1 / 2
  let s2 := d2 / 2
  let area := s1 * s2 * Real.sin angle
  have area_eq: area = 6 * Real.sqrt 2 := sorry
  exact area_eq

end area_of_midpoint_quadrilateral_l772_772432


namespace solve_radius_area_circumcircle_l772_772526

noncomputable def radius_area_circumcircle (P N Q : Point) (Ω : Circle) (R : ℝ) (S : ℝ) : Prop :=
  let PN_mid := midpoint P N
  let QN_mid := midpoint Q N
  let C := arc_midpoint P N (not_contains Ω Q)
  let D := arc_midpoint Q N (not_contains Ω P)
  
  is_isosceles_acute_triangle P N Q ∧
  is_circumcircle Ω P N Q ∧
  distance C (line P N) = 4 ∧
  distance D (line Q N) = 0.4 →
  R = 5 ∧
  S = (192 * real.sqrt 6) / 25

theorem solve_radius_area_circumcircle :
  ∀ (P N Q Ω : Type) (R S : ℝ),
  radius_area_circumcircle P N Q Ω R S := sorry

end solve_radius_area_circumcircle_l772_772526


namespace polar_coordinates_equivalence_l772_772722

theorem polar_coordinates_equivalence :
  ∀ (ρ θ1 θ2 : ℝ), θ1 = π / 3 ∧ θ2 = -5 * π / 3 →
  (ρ = 5) → 
  (ρ * Real.cos θ1 = ρ * Real.cos θ2 ∧ ρ * Real.sin θ1 = ρ * Real.sin θ2) :=
by
  sorry

end polar_coordinates_equivalence_l772_772722


namespace problem_part_I_problem_part_II_l772_772265

noncomputable def a : ℕ → ℤ
| 0     := 0 -- This is unused, placeholding for the Lean definition start from 1
| 1     := 1
| 2     := 3
| (n+3) := 3 * a (n + 2) - 2 * a (n + 1)

def a_geometric (n : ℕ) : Prop :=
  a (n + 2) - a (n + 1) = 2 * (a (n + 1) - a n)

def a_general_term (n : ℕ) : ℤ :=
  2^n - 1

def b (n : ℕ) : ℝ :=
  2 * real.log (a n + 1) / real.log 4

def sum_reciprocal_b_squared_minus_1 (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, 1 / (b k ^ 2 - 1))

theorem problem_part_I (n : ℕ) : a_geometric n ∧ ∀ n, a n = a_general_term n := 
by sorry

theorem problem_part_II (n : ℕ) : sum_reciprocal_b_squared_minus_1 n < 1 / 2 :=
by sorry

end problem_part_I_problem_part_II_l772_772265


namespace average_round_trip_time_l772_772816

def speed (day : String) : ℝ :=
  match day with
  | "Sunday"    => 2.5
  | "Monday"    => 8
  | "Tuesday"   => 11
  | "Thursday"  => 9
  | "Saturday"  => 3
  | _           => 0 -- default, should not be used

def distance (day : String) : ℝ :=
  match day with
  | "Sunday"    => 3.5
  | "Monday"    => 3
  | "Tuesday"   => 4.5
  | "Thursday"  => 2.8
  | "Saturday"  => 4.2
  | _           => 0 -- default, should not be used

def time_one_way (day : String) : ℝ :=
  distance day / speed day

def time_round_trip (day : String) : ℝ :=
  2 * time_one_way day

def round_trip_times : List ℝ :=
  [time_round_trip "Sunday", time_round_trip "Monday", time_round_trip "Tuesday", time_round_trip "Thursday", time_round_trip "Saturday"]

def average_time (times : List ℝ) : ℝ :=
  (times.foldl (· + ·) 0) / times.length

theorem average_round_trip_time : average_time (round_trip_times.map (· * 60)) = 93.4848 := by
  sorry

end average_round_trip_time_l772_772816


namespace instantaneous_velocity_at_t2_l772_772948

def robot_motion_eq (t : ℝ) : ℝ :=
  t + 3 / t

def velocity_eq (t : ℝ) : ℝ :=
  (deriv robot_motion_eq) t

theorem instantaneous_velocity_at_t2 :
  velocity_eq 2 = 13 / 4 :=
by
  sorry

end instantaneous_velocity_at_t2_l772_772948


namespace simplified_expression_correct_l772_772424

def simplify_expression (x : ℝ) : ℝ :=
  4 * (x ^ 2 - 5 * x) - 5 * (2 * x ^ 2 + 3 * x)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = -6 * x ^ 2 - 35 * x :=
by
  sorry

end simplified_expression_correct_l772_772424


namespace tangent_chord_length_problem_l772_772478

noncomputable def circle_tangent_chord_length (r1 r2 r3 : ℝ) (m n p : ℕ) : ℝ :=
  if h : m > 0 ∧ p > 0 ∧ Nat.gcd m p = 1 ∧ ¬ ∃ k : ℕ, k^2 ∣ n then
    let c := m.to_nat in
    (m * Real.sqrt n) / p
  else 0

theorem tangent_chord_length_problem : 
    ∃ (C1 C2 C3 : ℝ) (m n p ∈ ℕ) (m' n' p' ∈ ℕ), 
    C1 = 6 ∧ C2 = 12 ∧ C1 + C2 = C3 - C2 ∧ m' = 144 ∧ n' = 26 ∧ p' = 5 ∧ m' + n' + p' = 175 :=
begin
  use [6, 12, 36, 144, 26, 5],
  split, simp,
  split, simp,
  split, simp [Nat.Guard.get],
  split, simp [Nat.Guard.get],
  split, simp [Nat.Guard.get],
  exact by norm_num,
end

end tangent_chord_length_problem_l772_772478


namespace cuckoo_sounds_from_10_to_16_l772_772940

-- Define a function for the cuckoo sounds per hour considering the clock
def cuckoo_sounds (h : ℕ) : ℕ :=
  if h ≤ 12 then h else h - 12

-- Define the total number of cuckoo sounds from 10:00 to 16:00
def total_cuckoo_sounds : ℕ :=
  (List.range' 10 (16 - 10 + 1)).map cuckoo_sounds |>.sum

theorem cuckoo_sounds_from_10_to_16 : total_cuckoo_sounds = 43 := by
  sorry

end cuckoo_sounds_from_10_to_16_l772_772940


namespace max_x_minus_y_l772_772656

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l772_772656


namespace not_divisible_by_x2_x_1_l772_772454

-- Definitions based on conditions
def P (x : ℂ) (n : ℕ) : ℂ := x^(2 * n) + 1 + (x + 1)^(2 * n)
def divisor (x : ℂ) : ℂ := x^2 + x + 1
def possible_n : List ℕ := [17, 20, 21, 64, 65]

theorem not_divisible_by_x2_x_1 (n : ℕ) (hn : n ∈ possible_n) :
  ¬ ∃ p : ℂ[X], P (C ℂ p) n = divisor p := by
    sorry

end not_divisible_by_x2_x_1_l772_772454


namespace coffee_shrinkage_l772_772130

theorem coffee_shrinkage :
  let initial_volume_per_cup := 8
  let shrink_factor := 0.5
  let number_of_cups := 5
  let final_volume_per_cup := initial_volume_per_cup * shrink_factor
  let total_remaining_coffee := final_volume_per_cup * number_of_cups
  total_remaining_coffee = 20 :=
by
  -- This is where the steps of the solution would go.
  -- We'll put a sorry here to indicate omission of proof.
  sorry

end coffee_shrinkage_l772_772130


namespace least_possible_n_l772_772941

noncomputable def d (n : ℕ) := 105 * n - 90

theorem least_possible_n :
  ∀ n : ℕ, d n > 0 → (45 - (d n + 90) / n = 150) → n ≥ 2 :=
by
  sorry

end least_possible_n_l772_772941


namespace loan_percentage_correct_l772_772475

-- Define the parameters and conditions of the problem
def house_initial_value : ℕ := 100000
def house_increase_percentage : ℝ := 0.25
def new_house_cost : ℕ := 500000
def loan_percentage : ℝ := 75.0

-- Define the theorem we want to prove
theorem loan_percentage_correct :
  let increase_value := house_initial_value * house_increase_percentage
  let sale_price := house_initial_value + increase_value
  let loan_amount := new_house_cost - sale_price
  let loan_percentage_computed := (loan_amount / new_house_cost) * 100
  loan_percentage_computed = loan_percentage :=
by
  -- Proof placeholder
  sorry

end loan_percentage_correct_l772_772475


namespace max_x_minus_y_l772_772665

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l772_772665


namespace max_x_minus_y_l772_772654

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l772_772654


namespace max_x_minus_y_l772_772653

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l772_772653


namespace borrowing_needed_l772_772062

def state_policies (purchase_price sell_price : ℕ) :=
  let deed_tax := 0.04 * 350000 in
  let business_tax := 0.05 * sell_price in
  let income_tax := 0.20 * (sell_price - purchase_price) in
  let total_taxes := deed_tax + business_tax + income_tax in
  let actual_income := sell_price - income_tax - business_tax in
  let new_house_price := 350000 + deed_tax in
  let borrowing_amount := new_house_price - actual_income in
  borrowing_amount = 140500

theorem borrowing_needed (purchase_price: ℕ) (sell_price : ℕ) (personal_income_tax: ℕ) (new_house_price: ℕ) :
  purchase_price = 180000 ∧ personal_income_tax = 14000 ∧ new_house_price = 350000 →
  state_policies purchase_price sell_price = 140500 :=
by
  sorry

end borrowing_needed_l772_772062


namespace exists_subset_sum_divisible_by_2008_l772_772012

theorem exists_subset_sum_divisible_by_2008 (a : Fin 2008 → ℤ) :
  ∃ (I : Finset (Fin 2008)), I.nonempty ∧ (∑ i in I, a i) % 2008 = 0 :=
sorry

end exists_subset_sum_divisible_by_2008_l772_772012


namespace quadrilateral_area_is_114_5_l772_772989

noncomputable def area_of_quadrilateral_114_5 
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) : ℝ :=
  114.5

theorem quadrilateral_area_is_114_5
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) :
  area_of_quadrilateral_114_5 AB BC CD AD angle_ABC h1 h2 h3 h4 h5 = 114.5 :=
sorry

end quadrilateral_area_is_114_5_l772_772989


namespace centroid_of_triangle_OKD_on_circumcircle_COD_l772_772109

open EuclideanGeometry

-- Definition of the problem and conditions
variables {A B C D K M O S : Point}

-- Conditions
def Rectangle (A B C D : Point) : Prop := ∃ O, IsRect A B C D ∧ Center A B C D O
def CircumcirclePt (K : Point) : Prop := OnCircumcircle K A B C D
def IntersectSegment (K M : Point) : Prop := ∃ M, OnLineSegment C K M ∧ OnLineSegment A D M
def RatioAM_MD (M : Point) : Prop := Ratio A M D 2
def RectangleCenter (O : Point) : Prop := IsCenter A B C D O

-- The mathematically equivalent proof problem in Lean 4
theorem centroid_of_triangle_OKD_on_circumcircle_COD 
    (h1 : Rectangle A B C D) 
    (h2 : CircumcirclePt K) 
    (h3 : IntersectSegment K M) 
    (h4 : RatioAM_MD M) 
    (h5 : RectangleCenter O) : 
    OnCircumcircle (Centroid O K D) C O D :=
by
  sorry

end centroid_of_triangle_OKD_on_circumcircle_COD_l772_772109


namespace minimum_length_AX_l772_772608

theorem minimum_length_AX (ABC : Triangle) (X : Point) 
  (h_acute : ABC.acute)
  (h_side_AB : ABC.side_length AB = 13) 
  (h_side_BC : ABC.side_length BC = 14) 
  (h_side_CA : ABC.side_length CA = 15) 
  (h_angle_sum : ∠ ABX + ∠ ACX = ∠ CBX + ∠ BCX) :
  AX.min_length = sqrt(171) := 
sorry

end minimum_length_AX_l772_772608


namespace wire_cut_probability_l772_772141

theorem wire_cut_probability (x y : ℝ) (hx : 80 ≥ x ∧ x ≥ 20) (hy : 80 ≥ y ∧ y ≥ 20) (hxy : 80 ≥ 80 - x - y ∧ 80 - x - y ≥ 20) : 
  (area (event := {segments : ℝ × ℝ | let (x, y) := segments in 80 ≥ x ∧ x ≥ 20 ∧ 80 ≥ y ∧ y ≥ 20 ∧ 80 ≥ 80 - x - y ∧ 80 - x - y ≥ 20 })) / (totalArea := 6400) = 1/32 :=
sorry

end wire_cut_probability_l772_772141


namespace compute_Z_value_l772_772280

def operation_Z (c d : ℕ) : ℤ := c^2 - 3 * c * d + d^2

theorem compute_Z_value : operation_Z 4 3 = -11 := by
  sorry

end compute_Z_value_l772_772280


namespace perimeter_triangle_QRS_l772_772325

-- Definitions for the conditions
def right_angle_triangle (P Q R : Type) (angleP : angle P = 90) : Prop := 
  sqrt (P Q ^ 2 + P R ^ 2) = Q R

def length_PR : ℝ := 12
def length_SQ : ℝ := 11
def length_SR : ℝ := 13

-- Main theorem to be proven
theorem perimeter_triangle_QRS (P Q R S : Type) 
  (h1 : right_angle_triangle P Q R) 
  (h2 : PR = 12) 
  (h3 : SQ = 11) 
  (h4 : SR = 13) :
  perimeter_triangle_QRS = 44 := 
sorry

end perimeter_triangle_QRS_l772_772325


namespace probability_of_two_hits_in_three_shots_l772_772537

noncomputable def probabilities (P_A : ℝ) (P_not_A : ℝ) (independent_shots : ℕ → Prop) : ℝ :=
  let P := P_A * P_A * P_not_A + P_A * P_not_A * P_A + P_not_A * P_A * P_A
  P

theorem probability_of_two_hits_in_three_shots :
  let P_A := 0.8
  let P_not_A := 1 - P_A
  let independent_shots := λ (n : ℕ), true
  probabilities P_A P_not_A independent_shots = 0.384 :=
by
  -- The proof will be written here
  sorry

end probability_of_two_hits_in_three_shots_l772_772537


namespace sum_of_first_5n_l772_772757

theorem sum_of_first_5n (n : ℕ) (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 504) :
  (5 * n * (5 * n + 1)) / 2 = 1035 :=
sorry

end sum_of_first_5n_l772_772757


namespace maximum_value_of_x_minus_y_l772_772694

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l772_772694


namespace number_of_correct_statements_l772_772313

-- Define basic properties and relations
variable (m n : Type) [line m] [line n]
variable (α β : Type) [plane α] [plane β]

-- Assumptions
axiom m_perp_α : m ⊥ α
axiom n_parallel_α : n ∥ α
axiom m_parallel_n : m ∥ n
axiom n_perp_β : n ⊥ β
axiom m_parallel_α : m ∥ α
axiom m_cap_n : m ∩ n = ∅
axiom m_parallel_β : m ∥ β
axiom n_parallel_β : n ∥ β

-- The proof statement
theorem number_of_correct_statements : 
  (¬ (m_perp_α ∧ n_parallel_α) ∧ 
  ((m_parallel_n ∧ n_parallel_α) → m_parallel_α) ∧ 
  ((m_parallel_n ∧ n_perp_β ∧ m_parallel_α) → α ⊥ β) ∧ 
  ((m_cap_n ∧ m_parallel_α ∧ m_parallel_β ∧ n_parallel_α ∧ n_parallel_β) → α ∥ β)) = 3 := 
sorry

end number_of_correct_statements_l772_772313


namespace tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l772_772731

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - 1 - x - a * x ^ 2

theorem tangent_line_eqn_when_a_zero :
  (∀ x, y = f 0 x → y - (Real.exp 1 - 2) = (Real.exp 1 - 1) * (x - 1)) :=
sorry

theorem min_value_f_when_a_zero :
  (∀ x : ℝ, f 0 x >= f 0 0) := 
sorry

theorem range_of_a_for_x_ge_zero (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f a x ≥ 0) → (a ≤ 1/2) :=
sorry

theorem exp_x_ln_x_plus_one_gt_x_sq (x : ℝ) :
  x > 0 → ((Real.exp x - 1) * Real.log (x + 1) > x ^ 2) :=
sorry

end tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l772_772731


namespace max_value_x_minus_y_l772_772684

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l772_772684


namespace angle_Q_is_72_degrees_l772_772026

-- Define the regular decagon and the internal angle
def regular_decagon (A B C D E F G H I J : Type) :=
  ∀ (n : ℕ), (n < 10) → true -- This is a placeholder definition

def internal_angle_regular_decagon := 144

-- The proposition to prove
theorem angle_Q_is_72_degrees (A B C D E F G H I J Q : Type)
  (h_regular : regular_decagon A B C D E F G H I J)
  (h_internal_angle : internal_angle_regular_decagon = 144):
  Q = 72 := 
  sorry

end angle_Q_is_72_degrees_l772_772026


namespace problem_l772_772495

-- Define what it means to be a factor or divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a
def is_divisor (a b : ℕ) : Prop := a ∣ b

-- The specific problem conditions
def statement_A := is_factor 4 28
def statement_B := is_divisor 19 209 ∧ ¬ is_divisor 19 57
def statement_C := ¬ is_divisor 30 90 ∧ ¬ is_divisor 30 76
def statement_D := is_divisor 14 28 ∧ ¬ is_divisor 14 56
def statement_E := is_factor 9 162

-- The proof problem
theorem problem : statement_A ∧ ¬statement_B ∧ ¬statement_C ∧ ¬statement_D ∧ statement_E :=
by 
  -- You would normally provide the proof here
  sorry

end problem_l772_772495


namespace math_inequality_proof_l772_772390

theorem math_inequality_proof
  (a : ℕ → ℝ) (n : ℕ) (hn : n > 1) (hp : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0) :
  (∏ i in Finset.range (n + 1), (1 + a i)) < 
  (1 + (∑ i in Finset.range (n + 1), a i) + 
  ∑ i in Finset.range (n + 1), (∑ j in Finset.range (i + 1), (a j) ^ i / nat.factorial i)) :=
sorry

end math_inequality_proof_l772_772390


namespace consecutive_page_sum_l772_772456

theorem consecutive_page_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 285 := by
  sorry

end consecutive_page_sum_l772_772456


namespace unique_solution_of_abc_l772_772238

theorem unique_solution_of_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_lt_ab_c : a < b) (h_lt_b_c: b < c) (h_eq_abc : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 :=
by {
  -- Proof skipped, only the statement is provided.
  sorry
}

end unique_solution_of_abc_l772_772238


namespace compare_factorials_l772_772288

noncomputable def factorial_iterate : ℕ → ℕ → ℕ
| 0, n := n
| (succ k), n := factorial_iterate k (Nat.factorial n)

theorem compare_factorials : 
  let A := factorial_iterate 99 100
  let B := factorial_iterate 100 99
  in A < B :=
sorry

end compare_factorials_l772_772288


namespace total_hatchlings_l772_772477

section TurtleHatchlings

/-- Conditions: turtle species specifics -/
def clutchA : ℕ := 25
def hatchRateA : ℝ := 0.40
def turtlesA : ℕ := 3

def clutchB : ℕ := 20
def hatchRateB : ℝ := 0.30
def turtlesB : ℕ := 6

def clutchC : ℕ := 10
def hatchRateC : ℝ := 0.50
def turtlesC : ℕ := 4

def clutchD : ℕ := 15
def hatchRateD : ℝ := 0.35
def turtlesD : ℕ := 5

/-- Calculate expected hatchlings per clutch -/
def hatchlingsPerClutch (clutch: ℕ) (hatchRate: ℝ) := (clutch: ℝ) * hatchRate

/-- Calculate total hatchlings for a species -/
def totalHatchlings (clutch: ℕ) (hatchRate: ℝ) (turtles: ℕ) := turtles * (hatchlingsPerClutch clutch hatchRate).toNat

/-- Prove that the total number of hatchlings equals 111 -/
theorem total_hatchlings : 
  (totalHatchlings clutchA hatchRateA turtlesA) +
  (totalHatchlings clutchB hatchRateB turtlesB) +
  (totalHatchlings clutchC hatchRateC turtlesC) +
  (totalHatchlings clutchD hatchRateD turtlesD) = 111 :=
by
  sorry

end TurtleHatchlings

end total_hatchlings_l772_772477


namespace sum_of_angles_l772_772412

-- Define points A, B, C, M, N on a circle and their respective arc measures
variables {A B M N C : Type} [circle (A B M N C)]
def arcBM : ℝ := 60
def arcMN : ℝ := 24

-- Define angles R and S using the inscribed angle theorem
def angleR : ℝ := (arcBM + arcMN) / 2
def angleS : ℝ := (arcBM - arcMN) / 2

-- Lean 4 statement to prove the sum of angles R and S is 60 degrees
theorem sum_of_angles (A B M N C : Type) [circle (A B M N C)] :
  angleR + angleS = 60 :=
by
  -- Conditions from the problem statement
  have arcMC : ℝ := arcBM + arcMN
  have arcBN : ℝ := arcBM - arcMN
  have angle_R : ℝ := (arcMC / 2) := rfl
  have angle_S : ℝ := (arcBN / 2) := rfl
  -- Sum of angles
  calc
  angleR + angleS
    = (arcMC / 2) + (arcBN / 2) : by simp [angleR, angleS]
    = (84 / 2) + (36 / 2) : by simp [arcMC, arcBN]
    = 42 + 18 : by norm_num
    = 60 : by norm_num

end sum_of_angles_l772_772412


namespace max_x_minus_y_l772_772659

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l772_772659


namespace probability_multiple_of_3_or_7_l772_772051

theorem probability_multiple_of_3_or_7 :
  let cards := Finset.range 50
  let multiples_of_3 := cards.filter (λ n, (n + 1) % 3 = 0)
  let multiples_of_7 := cards.filter (λ n, (n + 1) % 7 = 0)
  let multiples_of_3_and_7 := cards.filter (λ n, (n + 1) % 21 = 0)
  (multiples_of_3.card + multiples_of_7.card - multiples_of_3_and_7.card) / 50 = 21 / 50 :=
by
  sorry

end probability_multiple_of_3_or_7_l772_772051


namespace lcm_subtract100_correct_l772_772486

noncomputable def lcm1364_884_subtract_100 : ℕ :=
  let a := 1364
  let b := 884
  let lcm_ab := Nat.lcm a b
  lcm_ab - 100

theorem lcm_subtract100_correct : lcm1364_884_subtract_100 = 1509692 := by
  sorry

end lcm_subtract100_correct_l772_772486


namespace total_area_of_folded_blankets_l772_772175

-- Define the initial conditions
def initial_area : ℕ := 8 * 8
def folds : ℕ := 4
def num_blankets : ℕ := 3

-- Define the hypothesis about folding
def folded_area (initial_area : ℕ) (folds : ℕ) : ℕ :=
  initial_area / (2 ^ folds)

-- The total area of all folded blankets
def total_folded_area (initial_area : ℕ) (folds : ℕ) (num_blankets : ℕ) : ℕ :=
  num_blankets * folded_area initial_area folds

-- The theorem we want to prove
theorem total_area_of_folded_blankets : total_folded_area initial_area folds num_blankets = 12 := by
  sorry

end total_area_of_folded_blankets_l772_772175


namespace total_wages_l772_772517

theorem total_wages (A_days : ℕ) (B_days : ℕ) (A_share : ℕ) (A_total : ℕ) : 
  A_days = 10 → 
  B_days = 15 → 
  A_share = 1860 → 
  A_total = 3 → 
  (A_share / A_total) * (A_total + (A_total * 2 / 3)) = 3100 :=
by {
  intros h1 h2 h3 h4,
  have A_one_day_work : ℚ := 1 / 10,
  have B_one_day_work : ℚ := 1 / 15,
  have together_one_day_work : ℚ := A_one_day_work + B_one_day_work,
  have ratio_work : ℚ := (1 / 10) / (1 / 15),
  have A_parts : ℚ := 3,
  have B_parts : ℚ := 2,
  have total_parts : ℚ := A_parts + B_parts,
  have per_part_value : ℚ := A_share / A_parts,
  have total_wages : ℚ := per_part_value * total_parts,
  rw [h1, h2, h3, h4],
  exact (620 : ℚ) * 5 = 3100
}

end total_wages_l772_772517


namespace sphere_touches_plane_at_circumcenter_l772_772634

theorem sphere_touches_plane_at_circumcenter {S x y z : Point} (A B C D E F : Point)
  (P : Plane) (T : Sphere) :
  is_trihedral_angle S x y z → 
  P.does_not_pass_through S → 
  P.cuts S x S y S z at A B C →
  triangle_congruent D A B S A B → 
  triangle_congruent E B C S B C → 
  triangle_congruent F C A S C A → 
  T.inside_trihedral S x y z → 
  T.not_inside_tetrahedron S A B C → 
  T.touches_plane_containing S A B → 
  T.touches_plane_containing S B C → 
  T.touches_plane_containing S C A →
  touches_at_circumcenter_of_triangle T P D E F 
:=
sorry

end sphere_touches_plane_at_circumcenter_l772_772634


namespace expressions_equal_iff_l772_772182

theorem expressions_equal_iff (a b c: ℝ) : a + 2 * b * c = (a + b) * (a + 2 * c) ↔ a + b + 2 * c = 0 :=
by 
  sorry

end expressions_equal_iff_l772_772182


namespace problem_statement_l772_772551

noncomputable def is_quadratic_eq (eq : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x: ℝ, eq x = a * x ^ 2 + b * x + c)

theorem problem_statement :
  ∀ eqA eqB eqC eqD : (ℝ → ℝ),
  eqA = (λ x, 3*(x-1)^2 - 2*(x-1)) →
  eqB = (λ x, abs 3 * x + 22 * x - 1) →
  eqC = (λ x, ax * x ^ 2 + bx * x + cx) →
  eqD = (λ x, (x - 1) * (x + 1) - (x ^ 2 + 2 * x)) →
  is_quadratic_eq eqA ∧ ¬is_quadratic_eq eqB ∧ ¬is_quadratic_eq eqD :=
by
  intros eqA eqB eqC eqD hA hB hC hD
  sorry

end problem_statement_l772_772551


namespace part_I_part_II_i_part_II_ii_l772_772719

open Real

def set_A (f : ℝ → ℝ) : Prop := ∃ x0 : ℝ, f (x0 + 1) + f x0 = f 1

theorem part_I : set_A (λ x => x⁻¹) :=
sorry

noncomputable def g (x a b : ℝ) : ℝ := log ((2^x + a) / b)

theorem part_II_i {a : ℝ} (h : ∀ x, g x a 1 ∈ set_A) : 0 ≤ a ∧ a < 2 :=
sorry

theorem part_II_ii {b : ℝ} (h : ∀ a ∈ Ioo 0 2, ∀ x, g x a b ∈ set_A) : b = 1 :=
sorry

end part_I_part_II_i_part_II_ii_l772_772719


namespace monotonic_decreasing_interval_l772_772050

def is_monotonic_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ ≥ f x₂

def func (x : ℝ) : ℝ :=
  (sqrt 3 - tan x) / (1 + sqrt 3 * tan x)

theorem monotonic_decreasing_interval :
  ∃ (k : ℤ), is_monotonic_decreasing func {x : ℝ | k * π - π / 6 < x ∧ x < k * π + 5 * π / 6} :=
sorry

end monotonic_decreasing_interval_l772_772050


namespace coprime_integer_pairs_sum_285_l772_772746

theorem coprime_integer_pairs_sum_285 : 
  (∃ s : Finset (ℕ × ℕ), 
    ∀ p ∈ s, p.1 + p.2 = 285 ∧ Nat.gcd p.1 p.2 = 1 ∧ s.card = 72) := sorry

end coprime_integer_pairs_sum_285_l772_772746


namespace max_x_minus_y_l772_772657

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l772_772657


namespace triangle_proof_l772_772802

-- Defining the properties of triangle ABC
variable (A B C : Type) -- A, B, and C are points
variable [P : PointType A B C] -- Make use of a PointType class for triangle properties
variable (AB AC BC : ℝ) -- Define side lengths as real numbers

-- Introduction of the lengths
def side_AB : AB = 5 := sorry
def side_BC : BC = 4 := sorry

-- Definitions of angles at respective points
variable (angle_ABC angle_BAC angle_ACB : ℝ)

-- Propose statements
-- Statement (a)
def isosceles_triangle (AC : ℝ) (hAC : AC = 4) : angle_ABC > angle_BAC := sorry
-- Statement (c)
def degenerate_triangle (AC : ℝ) (hAC : AC = 2) : angle_ABC < angle_ACB := sorry
-- Proving statements a and c are incorrect
theorem triangle_proof : triangle_angles AB 5 BC 4 AC 4 ∧ isosceles_triangle AB BC AC → False ∧ 
                         triangle_angles AB 5 BC 4 AC 2 ∧ degenerate_triangle AB BC AC → False := 
begin
  sorry
end

end triangle_proof_l772_772802


namespace pyramid_edges_l772_772073

-- Definitions based on conditions
variables (V E F : ℕ)
def prism_condition := V + E + F = 50
def euler_formula := V - E + F = 2

-- The theorem we want to prove regarding the pyramid
theorem pyramid_edges (V F : ℕ) (h_prism : prism_condition V (V + F - 2) F) (h_euler : euler_formula V (V + F - 2) F) :
  let B := (V + F - 2) / 3 in
  2 * B = 16 := sorry

end pyramid_edges_l772_772073


namespace centroid_property_l772_772372

open Real

theorem centroid_property
  {α β γ : ℝ} (h1 : α ≠ 0) (h2 : β ≠ 0) (h3 : γ ≠ 0)
  (h4 : (1 / α) + (1 / β) + (1 / γ) = 1) :
  let p := α / 3,
      q := β / 3,
      r := γ / 3 in
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 9 :=
by
  sorry

end centroid_property_l772_772372


namespace convex_polygon_cover_with_circle_l772_772013

-- Definitions
variables {V : Type} [EuclideanSpace V] 

def is_convex_polygon (vertices : List V) : Prop := 
  sorry -- assume the definition for a convex polygon

def circle_through_three_vertices (A B C : V) : Circle := 
  sorry -- assume the definitions for the circle through three vertices

def covers_polygon (C : Circle) (polygon : List V) : Prop :=
  sorry -- assume the definition that the circle covers the polygon

-- Main theorem
theorem convex_polygon_cover_with_circle {polygon : List V} 
  (h_convex : is_convex_polygon polygon) :
  ∃ (A B C : V) (h_three_consecutive : (A, B, C) ∈ zip3 polygon (tail polygon) (tail (tail polygon))),
    covers_polygon (circle_through_three_vertices A B C) polygon :=
begin
  sorry
end

end convex_polygon_cover_with_circle_l772_772013


namespace black_dogs_count_l772_772068

def number_of_brown_dogs := 20
def number_of_white_dogs := 10
def total_number_of_dogs := 45
def number_of_black_dogs := total_number_of_dogs - (number_of_brown_dogs + number_of_white_dogs)

theorem black_dogs_count : number_of_black_dogs = 15 := by
  sorry

end black_dogs_count_l772_772068


namespace ellipse_focal_distance_l772_772441

-- Define the problem conditions
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 5) + (y^2 / 4) = 1

-- Prove the focal distance is 2
theorem ellipse_focal_distance : ∀ {x y : ℝ}, ellipse_eq x y → 2 * (sqrt (5 - 4)) = 2 :=
by
  intro x y h
  have a_sq : ℝ := 5
  have b_sq : ℝ := 4
  have c_sq : ℝ := a_sq - b_sq
  have c : ℝ := sqrt c_sq
  calc
    2 * c = 2 * sqrt 1 := by rw [a_sq, b_sq, sq_sub]
    ... = 2 * 1 := by rw sqrt_one
    ... = 2 := by norm_num

end ellipse_focal_distance_l772_772441


namespace maria_carrots_l772_772806

theorem maria_carrots :
  ∀ (picked initially thrownOut moreCarrots totalLeft : ℕ),
    initially = 48 →
    thrownOut = 11 →
    totalLeft = 52 →
    moreCarrots = totalLeft - (initially - thrownOut) →
    moreCarrots = 15 :=
by
  intros
  sorry

end maria_carrots_l772_772806


namespace largest_unique_digits_multiple_of_9_remainder_l772_772371

theorem largest_unique_digits_multiple_of_9_remainder :
  ∃ (M : ℕ), (∀ i j, 0 ≤ i ∧ i < 10 → 0 ≤ j ∧ j < 10 → i ≠ j → digit_in_pos i M ∧ digit_in_pos j M) ∧ 
             (∑ d in (digits M), d) % 9 = 0 ∧
             M = max M ∧ 
             M % 1000 = 210 :=
sorry

end largest_unique_digits_multiple_of_9_remainder_l772_772371


namespace find_n_l772_772211

theorem find_n (n : ℕ) (S : ℕ) (h1 : S = n * (n + 1) / 2)
  (h2 : ∃ a : ℕ, a > 0 ∧ a < 10 ∧ S = 111 * a) : n = 36 :=
sorry

end find_n_l772_772211


namespace max_prime_sums_is_seven_l772_772870

-- Define the permutation of integers from 1 to 10
def isPermutation (l : List ℕ) : Prop :=
  l.perm [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the sums S_i
def sums (l : List ℕ) : List ℕ :=
  l.tails.foldr (λ tail acc, (acc.headI + tail.headI) :: acc) [0]

-- Define the primality check
def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Define the maximum number of prime sums
def maxPrimeSums (l : List ℕ) : ℕ :=
  (sums l).countp isPrime

theorem max_prime_sums_is_seven :
  ∃ l : List ℕ, isPermutation l ∧ maxPrimeSums l = 7 := 
sorry

end max_prime_sums_is_seven_l772_772870


namespace nat_numbers_equal_if_divisible_l772_772002

theorem nat_numbers_equal_if_divisible
  (a b : ℕ)
  (h : ∀ n : ℕ, ∃ m : ℕ, n ≠ m → (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end nat_numbers_equal_if_divisible_l772_772002


namespace power_sum_eq_nine_l772_772797

theorem power_sum_eq_nine {m n p q : ℕ} (h : ∀ x > 0, (x + 1)^m / x^n - 1 = (x + 1)^p / x^q) :
  (m^2 + 2 * n + p)^(2 * q) = 9 :=
sorry

end power_sum_eq_nine_l772_772797


namespace max_5_cent_coins_l772_772091

theorem max_5_cent_coins :
  ∃ (x y z : ℕ), 
  x + y + z = 25 ∧ 
  x + 2*y + 5*z = 60 ∧
  (∀ y' z' : ℕ, y' + 4*z' = 35 → z' ≤ 8) ∧
  y + 4*z = 35 ∧ z = 8 := 
sorry

end max_5_cent_coins_l772_772091


namespace option_d_true_l772_772283

theorem option_d_true (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpr_qr : p * r < q * r) : 1 > q / p :=
sorry

end option_d_true_l772_772283


namespace sum_angles_multiple_pi_l772_772017

open Real

-- Define the parabola as a function, its focus, and axis
def parabola (x : ℝ) : ℝ := x^2
def focus : ℝ × ℝ := (0, 0.25)
def axis : ℝ → ℝ := id -- y-axis

-- Given conditions
variables (P : ℝ × ℝ)
hypothesis three_distinct_normals : ∃ k1 k2 k3 : ℝ, k1 ≠ k2 ∧ k2 ≠ k3 ∧ k3 ≠ k1 ∧
  (∃ (f : ℝ × ℝ → ℝ), ∀ k, f (P.1, P.2) = 0)

-- Define function to calculate angle with the axis
noncomputable def angle_with_axis (θ : ℝ) : ℝ := θ

-- Prove that the sum is a multiple of π
theorem sum_angles_multiple_pi (L : ℝ → ℝ) :
  ∃ n : ℤ, ∑ (θ : ℝ) in {k1, k2, k3}, angle_with_axis θ - angle_with_axis (atan ((P.2 - focus.2) / (P.1 - focus.1))) = n * π :=
by sorry

end sum_angles_multiple_pi_l772_772017


namespace average_weight_of_class_equal_l772_772069

theorem average_weight_of_class_equal :
  let students_A := 50
      avg_weight_A := 50
      students_B := 40
      avg_weight_B := 70
      students_C := 30
      total_weight_C := 2400
      total_students := students_A + students_B + students_C
      total_weight := (students_A * avg_weight_A) + (students_B * avg_weight_B) + total_weight_C
      avg_weight := total_weight / total_students
  in avg_weight = 64.17 :=
by {
  sorry
}

end average_weight_of_class_equal_l772_772069


namespace unfair_die_sum_odd_l772_772902

theorem unfair_die_sum_odd :
  let p := 1 / 6 in
  let p_even := 5 * p in
  let p_odd := p in
  (p_even * p_odd + p_odd * p_even) = 5 / 18 :=
by
  sorry

end unfair_die_sum_odd_l772_772902


namespace exercise_proof_problem_l772_772999

-- Definitions based on the conditions
def parametric_equation_line (l : ℝ → ℝ × ℝ) (t : ℝ) (α : ℝ) : Prop :=
  l t = (1 + t * Real.cos α, 2 + t * Real.sin α)

def polar_equation_circle (C : ℝ* → Prop) (ρ θ : ℝ) : Prop :=
  ∃ θ, ρ = 6 * Real.sin θ ∧ C (ρ, θ)

def cartesian_equation_circle (C_cart : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, C_cart (x, y) ↔ x^2 + (y - 3)^2 = 9

def distance_min_value (P A B : (ℝ × ℝ)) : ℝ :=
  Real.sqrt (32 - 4 * Real.sin (2 * (Real.atan (P.2 - A.2) (P.1 - A.1))))

-- The proof problem
theorem exercise_proof_problem
  (l : ℝ → ℝ × ℝ) (C : ℝ* → Prop) (C_cart : ℝ × ℝ → Prop) (P A B : (ℝ × ℝ))
  (α : ℝ) (t : ℝ) :
  (∀ t, parametric_equation_line l t α) →
  (∀ ρ θ, polar_equation_circle C ρ θ) →
  (∀ P, P = (1, 2)) →
  (∃ t₁ t₂, ∀ x y, C_cart (x, y) ↔ x^2 + (y - 3)^2 = 9 ∧
    ∃ t₁ t₂, t₁ + t₂ = -2*(Real.cos α - Real.sin α) ∧ t₁ * t₂ = -7 ∧
    distance_min_value P A B = 2 * Real.sqrt 7) :=
by 
  intro parametric_eq polar_eq P_eq cartesian_eq intersection_eq_min_value;
  sorry

end exercise_proof_problem_l772_772999


namespace min_max_value_l772_772510

variable {ι : Type} [Fintype ι] {f : ι → (ℝ → ℝ)}

theorem min_max_value :
  (∃ (F : Fin 2017 → (ℝ → ℝ)), 
      ∀ (x : Fin 2017 → ℝ), 
        (∀ i, 0 ≤ x i ∧ x i ≤ 1) → 
        F 1 (x 1) + F 2 (x 2) + ... + F 2017 (x 2017) - x 1 * x 2 * ... * x 2017 = \max_{0 \le x_1, \ldots, x_{2017} \le 1} \left| F_1( x_{1}) + \ldots + F_{2017}(x_{ 2017}) - x_{1} x_{2} \ldots x_{2017} \right| := 
    ∃ (F : ℕ → (ℝ → ℝ)), (∀ i, 1 ≤ i ∧ i ≤ 2017 ∧ ∀ x, 0 ≤ x ∧ x ≤ 1),
    ∃ x : ℕ → ℝ, (∀ i, 1 ≤ i ∧ i ≤ 2017 ∧ 0 ≤ x i ∧ x i ≤ 1) ∧
    ( ∃ S : ℝ, 
      (S = F 1 (0) + F 2 (0) + ... + F 2017 (0) )
        + (2016 * (F 1 (1) + F 2 (1) + ... + F 2017 (1) - 1))
        + ⌊ 1.0 := \frac{1008}{2017})
      ∧
      (S_2 = F (0)+ F_{2000} + ... )

end min_max_value_l772_772510


namespace total_sleep_per_week_l772_772473

namespace TotalSleep

def hours_sleep_wd (days: Nat) : Nat := 6 * days
def hours_sleep_wknd (days: Nat) : Nat := 10 * days

theorem total_sleep_per_week : 
  hours_sleep_wd 5 + hours_sleep_wknd 2 = 50 := by
  sorry

end TotalSleep

end total_sleep_per_week_l772_772473


namespace percentage_of_green_ducks_smaller_pond_l772_772303

-- Definitions of the conditions
def num_ducks_smaller_pond : ℕ := 30
def num_ducks_larger_pond : ℕ := 50
def percentage_green_larger_pond : ℕ := 12
def percentage_green_total : ℕ := 15
def total_ducks : ℕ := num_ducks_smaller_pond + num_ducks_larger_pond

-- Calculation of the number of green ducks
def num_green_larger_pond := percentage_green_larger_pond * num_ducks_larger_pond / 100
def num_green_total := percentage_green_total * total_ducks / 100

-- Define the percentage of green ducks in the smaller pond
def percentage_green_smaller_pond (x : ℕ) :=
  x * num_ducks_smaller_pond / 100 + num_green_larger_pond = num_green_total

-- The theorem to be proven
theorem percentage_of_green_ducks_smaller_pond : percentage_green_smaller_pond 20 :=
  sorry

end percentage_of_green_ducks_smaller_pond_l772_772303


namespace estimate_students_above_120_l772_772319

noncomputable def normal_distribution (μ : ℝ) (σ : ℝ) :=
  distribNormal μ σ

def num_students_above_120 (students : ℕ) (μ σ : ℝ) (hσ : σ > 0) (P_range : ℝ) (hP_range : P_range = 0.8) : ℕ := 
  let proportion_above_120 := (1 - P_range) / 2
  let num_above_120 := proportion_above_120 * students
  num_above_120.toNat

theorem estimate_students_above_120 :
  ∀ (students : ℕ) (μ σ : ℝ) (hσ : σ > 0) (P_range : ℝ) (hP_range : P_range = 0.8),
  students = 780 → μ = 90 →
  num_students_above_120 students μ σ hσ P_range hP_range = 78 := 
by intros students μ σ hσ P_range hP_range h_students h_μ
   simp [num_students_above_120, h_students, h_μ, hP_range]
   sorry

end estimate_students_above_120_l772_772319


namespace find_p8_l772_772378

noncomputable def p (x : ℝ) : ℝ :=
  x + (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem find_p8 :
  ∀ x : ℝ, p(1) = 1 ∧ p(2) = 2 ∧ p(3) = 3 ∧ p(4) = 4 ∧ p(5) = 5 ∧ p(6) = 6 → p(8) = 5048 :=
by
  intro x
  sorry

end find_p8_l772_772378


namespace incorrect_projection_on_D1D_BB1_l772_772324

variables (A A1 B B1 C C1 D D1 E F : Point)

-- Axioms based on the conditions given in the problem
axiom trisection_A1_A (hEA : dist A1 E = dist E A) (hE2A1E : dist A E = 2 * dist A1 E)
axiom trisection_C1_C (hFC : dist C1 F = dist F C) (hF2C1F : dist C F = 2 * dist C1 F)

-- The main statement: The projection of the section on face D1 D B B1 is not a line segment
theorem incorrect_projection_on_D1D_BB1 : ¬ is_line_segment (projection_of_section_on_face {B, E, F} D1 D B B1) :=
sorry

end incorrect_projection_on_D1D_BB1_l772_772324


namespace trihedral_angle_sum_trihedral_angle_inequality_l772_772104

theorem trihedral_angle_sum (A B C S : Point)
  (angle_BSC : Real)
  (angle_CSA : Real)
  (angle_ASB : Real)
  (h1 : is_trihedral_angle S A B C angle_BSC angle_CSA angle_ASB) :
  angle_BSC + angle_CSA + angle_ASB < 2 * Real.pi := 
  sorry

theorem trihedral_angle_inequality (A B C S : Point)
  (angle_BSC : Real)
  (angle_CSA : Real)
  (angle_ASB : Real)
  (h1 : is_trihedral_angle S A B C angle_BSC angle_CSA angle_ASB) :
  angle_BSC < angle_ASB + angle_CSA :=
  sorry

end trihedral_angle_sum_trihedral_angle_inequality_l772_772104


namespace correct_percentage_is_correct_l772_772920

noncomputable def percentage_of_correct_answers (correct incorrect : ℕ) : ℝ :=
  (correct : ℝ) / ((correct + incorrect) : ℝ) * 100

theorem correct_percentage_is_correct {correct incorrect : ℕ}
  (h_correct : correct = 35) (h_incorrect : incorrect = 13) :
  Real.round (percentage_of_correct_answers correct incorrect * 100) / 100 = 72.92 :=
by
  -- sorry is used to skip the proof
  sorry

end correct_percentage_is_correct_l772_772920


namespace megan_seashells_l772_772403

theorem megan_seashells : ∃ (n_curr_seashells : ℕ), n_curr_seashells + 6 = 25 ∧ n_curr_seashells = 19 :=
by
  use 19
  split
  . exact rfl
  . exact rfl

end megan_seashells_l772_772403


namespace triangle_median_equal_bc_l772_772335

-- Let \( ABC \) be a triangle, \( AB = 2 \), \( AC = 3 \), and the median from \( A \) to \( BC \) has the same length as \( BC \).
theorem triangle_median_equal_bc (A B C M : Type) (AB AC BC AM : ℝ) 
  (hAB : AB = 2) (hAC : AC = 3) 
  (hMedian : BC = AM) (hM : M = midpoint B C) :
  BC = real.sqrt (26 / 5) :=
by sorry

end triangle_median_equal_bc_l772_772335


namespace maximum_value_of_x_minus_y_l772_772704

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l772_772704


namespace divisibility_rules_l772_772196

def base6_digits (N : ℕ) : List ℕ := sorry  -- Constructs the list of digits of N in base-6 representation

def divisible_by (N : ℕ) (d : ℕ) : Prop :=
  match d with
  | 2 => List.head (base6_digits N) ∈ [0, 2, 4]
  | 3 => List.head (base6_digits N) ∈ [0, 3]
  | 4 => let last_two_digits := List.take 2 (base6_digits N)
         (last_two_digits.head + 6 * last_two_digits.tail.head) % 4 = 0
  | 5 => (List.foldr (λ x acc, x + acc) 0 (base6_digits N)) % 5 = 0
  | 6 => List.head (base6_digits N) = 0
  | 7 => (List.foldr (λ x acc, x - acc) 0 (List.zipWith HasNeg.neg (base6_digits N) (List.range (List.length (base6_digits N))))) % 7 = 0
  | 8 => let last_three_digits := List.take 3 (base6_digits N)
         (last_three_digits.head + 6 * last_three_digits.tail.head + 36 * last_three_digits.tail.tail.head) % 8 = 0
  | 9 => let last_two_digits := List.take 2 (base6_digits N)
         (last_two_digits.head + 6 * last_two_digits.tail.head) % 9 = 0
  | 10 => (List.foldr (λ x acc, x + acc) 0 (base6_digits N)) % 5 = 0 ∧ List.head (base6_digits N) ∈ [0, 2, 4]
  | _ => false

theorem divisibility_rules (N : ℕ) (d : ℕ) (h_d : d ≤ 10) : divisible_by N d := sorry

end divisibility_rules_l772_772196


namespace equation_of_line_ab_l772_772529

open Real

theorem equation_of_line_ab (x1 y1 x2 y2 : ℝ) :
  let ellipse : set (ℝ × ℝ) := {p | (p.1)^2 / 4 + (p.2)^2 / 3 = 1} in
  (1, 1) ∈ ellipse →
  (x1, y1) ∈ ellipse →
  (x2, y2) ∈ ellipse →
  x1 + x2 = 2 →
  y1 + y2 = 2 →
  3 * 1 + 4 * 1 - 7 = 0 :=
by
  intro ellipse hM hA hB hsum_x hsum_y
  sorry

end equation_of_line_ab_l772_772529


namespace max_x_minus_y_l772_772641

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772641


namespace elasticity_ratio_l772_772161

theorem elasticity_ratio (e_QN e_PN : ℝ) (h1 : e_QN = 1.27) (h2 : e_PN = 0.76) : 
  (e_QN / e_PN) ≈ 1.7 :=
by
  rw [h1, h2]
  -- prove the statement using the given conditions
  sorry

end elasticity_ratio_l772_772161


namespace polynomial_primes_to_primes_l772_772131

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem polynomial_primes_to_primes (f : ℕ → ℕ) (h_poly : ∀ x, is_polynomial f x) : 
  (∀ x, f(x) > 0) ->  
  (∀ p, is_prime p → is_prime (f p)) -> 
  (∀ p, is_prime p → is_prime (f p)) :=
by
  intro h_positive h_primes
  sorry

end polynomial_primes_to_primes_l772_772131


namespace P_symmetry_l772_772056

noncomputable def P : ℕ → ℝ → ℝ → ℝ
| 1, _, _ => 1
| (n + 2), x, y => (x + y - 1) * (y + 1) * P (n + 1) x (y + 2) + (y - y^2) * P (n + 1) x y

theorem P_symmetry (n : ℕ) (x y : ℝ) : P n x y = P n y x := 
sorry

end P_symmetry_l772_772056


namespace quad_area_FDBG_l772_772078

open Real

noncomputable def area_quad_FDBG (AB AC area_ABC : ℝ) : ℝ :=
  let AD := AB / 2
  let AE := AC / 2
  let area_ADE := area_ABC / 4
  let x := 2 * area_ABC / (AB * AC)
  let sin_A := x
  let hyp_ratio := sin_A / (area_ABC / AC)
  let factor := hyp_ratio / 2
  let area_AFG := factor * area_ADE
  area_ABC - area_ADE - 2 * area_AFG

theorem quad_area_FDBG (AB AC area_ABC : ℝ) (hAB : AB = 60) (hAC : AC = 15) (harea : area_ABC = 180) :
  area_quad_FDBG AB AC area_ABC = 117 := by
  sorry

end quad_area_FDBG_l772_772078


namespace snowdrift_depth_first_day_l772_772521

theorem snowdrift_depth_first_day (final_depth fourth_day_snow third_day_snow : ℝ) (half_melt : ℝ) :
  final_depth = 34 →
  fourth_day_snow = 18 →
  third_day_snow = 6 →
  half_melt * 2 = 20 →
  (final_depth - fourth_day_snow - third_day_snow) * 2 = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  have h5 : (34 - 18 - 6) = 10 := sorry
  rw h5 at h4
  exact h4

end snowdrift_depth_first_day_l772_772521


namespace length_MN_in_trapezoid_l772_772038

theorem length_MN_in_trapezoid (a b : ℝ) (h₀ : a > b) :
  let K := (a / 2, 0)
  let M := intersection_point (line_through (a / 2, 0) (0, b)) (line_through (0, 0) (a, b))
  let N := intersection_point (line_through (a / 2, 0) (a, b)) (line_through (0, b) (a, 0))
  MN = distance M N :=
  MN = (a * b) / (a + 2 * b) :=
begin
  sorry
end

end length_MN_in_trapezoid_l772_772038


namespace rectangle_path_length_l772_772418

theorem rectangle_path_length :
  let d := sqrt 34
  in total_path_length (rectangle_rotation_path 3 5 ( (rotate 180 (3, 5)) then rotate 270 ((new_position_after 180 (rotate 180 (3, 5))) (3, 5))) ) = 5 * π * d := 
sorry

end rectangle_path_length_l772_772418


namespace solve_for_x_l772_772032

theorem solve_for_x (x : ℚ) : (2/5 : ℚ) - (1/4 : ℚ) = 1/x → x = 20/3 :=
by
  intro h
  sorry

end solve_for_x_l772_772032


namespace max_x_minus_y_l772_772644

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772644


namespace find_integers_l772_772195

theorem find_integers (x : ℤ) (h₁ : x ≠ 3) (h₂ : (x - 3) ∣ (x ^ 3 - 3)) :
  x = -21 ∨ x = -9 ∨ x = -5 ∨ x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 5 ∨
  x = 7 ∨ x = 9 ∨ x = 11 ∨ x = 15 ∨ x = 27 :=
sorry

end find_integers_l772_772195


namespace exists_abc_eq_2015_min_abc_value_l772_772482

open Nat

-- Condition definitions
def gcdCond1 (a b c : ℕ) : Prop := gcd (gcd a b) c = 1
def gcdCond2 (a b c : ℕ) : Prop := gcd a (b + c) > 1
def gcdCond3 (a b c : ℕ) : Prop := gcd b (c + a) > 1
def gcdCond4 (a b c : ℕ) : Prop := gcd c (a + b) > 1

-- Part (a): Prove the existence of a, b, c such that a + b + c = 2015
theorem exists_abc_eq_2015 : ∃ (a b c : ℕ), gcdCond1 a b c ∧ gcdCond2 a b c ∧ gcdCond3 a b c ∧ gcdCond4 a b c ∧ a + b + c = 2015 := sorry

-- Part (b): Prove the minimum possible value of a + b + c
theorem min_abc_value : ∃ (a b c : ℕ), gcdCond1 a b c ∧ gcdCond2 a b c ∧ gcdCond3 a b c ∧ gcdCond4 a b c ∧ a + b + c = 30 := sorry

end exists_abc_eq_2015_min_abc_value_l772_772482


namespace tricycles_count_l772_772972

theorem tricycles_count {s t : Nat} (h1 : s + t = 10) (h2 : 2 * s + 3 * t = 26) : t = 6 :=
sorry

end tricycles_count_l772_772972


namespace probability_two_girls_l772_772939

theorem probability_two_girls (total_members boys girls : ℕ) (total_members_eq : total_members = 15) (boys_eq : boys = 9) (girls_eq : girls = 6) :
  (∃ (chosen_prob : ℚ), chosen_prob = (∑ s : Finset (Fin 15), (s.card = 2 ∧ ∀ i ∈ s, i < 6)) / (∑ s : Finset (Fin 15), s.card = 2) ∧ chosen_prob = 1 / 7) :=
by
  have total_pairs := (total_members * (total_members - 1)) / 2
  have girl_pairs := (girls * (girls - 1)) / 2
  have prob := (girl_pairs : ℚ) / (total_pairs : ℚ)
  use prob
  split
  sorry
  exact rfl

end probability_two_girls_l772_772939


namespace probability_all_switches_on_is_correct_l772_772098

-- Mechanical declaration of the problem
structure SwitchState :=
  (state : Fin 2003 → Bool)

noncomputable def probability_all_on (initial : SwitchState) : ℚ :=
  let satisfying_confs := 2
  let total_confs := 2 ^ 2003
  let p := satisfying_confs / total_confs
  p

-- Definition of the term we want to prove
theorem probability_all_switches_on_is_correct :
  ∀ (initial : SwitchState), probability_all_on initial = 1 / 2 ^ 2002 :=
  sorry

end probability_all_switches_on_is_correct_l772_772098


namespace square_perimeter_l772_772314

theorem square_perimeter (x : ℝ) (h : x * x + x * x = (2 * Real.sqrt 2) * (2 * Real.sqrt 2)) :
    4 * x = 8 :=
by
  sorry

end square_perimeter_l772_772314


namespace sin_120_eq_sqrt3_div_2_l772_772468

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l772_772468


namespace find_value_divide_subtract_l772_772532

theorem find_value_divide_subtract :
  (Number = 8 * 156 + 2) → 
  (CorrectQuotient = Number / 5) → 
  (Value = CorrectQuotient - 3) → 
  Value = 247 :=
by
  intros h1 h2 h3
  sorry

end find_value_divide_subtract_l772_772532


namespace sum_non_palindrome_integers_exactly_three_steps_l772_772613

def reverse (n : ℕ) : ℕ :=
  n.toString.reverse.toNat

def is_palindrome (n : ℕ) : Prop :=
  n = reverse n

def takes_three_steps_to_palindrome (n : ℕ) : Prop :=
  ¬is_palindrome n ∧
  let n1 := n + reverse n in
  ¬is_palindrome n1 ∧
  let n2 := n1 + reverse n1 in
  ¬is_palindrome n2 ∧
  is_palindrome (n2 + reverse n2)

def is_non_palindrome_between_10_and_200 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 200 ∧ ¬is_palindrome n

theorem sum_non_palindrome_integers_exactly_three_steps : 
  (Finset.filter 
    (λ n => is_non_palindrome_between_10_and_200 n ∧ takes_three_steps_to_palindrome n) 
    (Finset.range 200)).sum (λ x => (x : ℕ)) = 642 :=
by
  sorry

end sum_non_palindrome_integers_exactly_three_steps_l772_772613


namespace vector_perpendicular_solution_l772_772271

noncomputable def a (m : ℝ) : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (3, -2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_perpendicular_solution (m : ℝ) (h : dot_product (a m + b) b = 0) : m = 8 := by
  sorry

end vector_perpendicular_solution_l772_772271


namespace course_students_l772_772304

noncomputable def total_students (n : ℕ) : Prop := 
  let a := 2 * n / 9
  let b := 3 * n / 11
  let c := 1 * n / 5
  let d := 2 * n / 15
  let e := 1 * n / 10
  let f := 12 
  (a + b + c + d + e + f = n) ∧ (a, b, c, d, e, f ≥ 0)

theorem course_students : total_students 931 :=
sorry

end course_students_l772_772304


namespace series_sum_to_4_l772_772747

theorem series_sum_to_4 (x : ℝ) (hx : ∑' n : ℕ, (n + 1) * x^n = 4) : x = 1 / 2 := 
sorry

end series_sum_to_4_l772_772747


namespace x_quad_greater_l772_772831

theorem x_quad_greater (x : ℝ) : x^4 > x - 1/2 :=
sorry

end x_quad_greater_l772_772831


namespace M_intersection_N_eq_N_l772_772254

def M := { x : ℝ | x < 4 }
def N := { x : ℝ | x ≤ -2 }

theorem M_intersection_N_eq_N : M ∩ N = N :=
by
  sorry

end M_intersection_N_eq_N_l772_772254


namespace number_of_true_propositions_l772_772863

-- Define the propositions
def proposition1 := ∀ (k : ℝ) (A : ℝ), (k = 9) → (A * k^2 = 81 * A)
def proposition2 := ∀ (r : ℝ) (θ : ℝ), (0 < r ∧ 0 < θ) → 
  (chord_length r θ = chord_length r θ)
def proposition3 := ∀ (a b : ℝ), (a = b ∧ a ⟂ b) → ∃ q : quadrilateral, (q.is_square)
def proposition4 := ∀ (P : Point) (l : Line), (¬ P ∈ l) → ∃! m : Line, (P ∈ m ∧ Parallel l m)

-- The number of true propositions is 2
theorem number_of_true_propositions : 
  [proposition1, proposition2, proposition3, proposition4].count(true) = 2 := 
sorry

end number_of_true_propositions_l772_772863


namespace monotonic_decreasing_interval_l772_772210

def f (x : ℝ) : ℝ := 3 + x * Real.log x

theorem monotonic_decreasing_interval :
  ∃ a b : ℝ, a = 0 ∧ b = 1 / Real.exp 1 ∧ (∀ x : ℝ, x ∈ Set.Ioo a b → (f' x) < 0) :=
by
  sorry

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

end monotonic_decreasing_interval_l772_772210


namespace area_of_rectangle_PQRS_l772_772312

def P : ℝ × ℝ := (1, -3)
def Q : ℝ × ℝ := (101, 17)
def y_S : ℤ := -13
def S : ℝ × ℝ := (3, y_S)

theorem area_of_rectangle_PQRS :
  let PQ := real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2),
      PS := real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2)
  in PQ * PS = 1010 :=
by
  -- The proofs are skipped here.
  sorry

end area_of_rectangle_PQRS_l772_772312


namespace percentage_increase_is_50_l772_772516

def initialNumber := 80
def finalNumber := 120

theorem percentage_increase_is_50 : ((finalNumber - initialNumber) / initialNumber : ℝ) * 100 = 50 := 
by 
  sorry

end percentage_increase_is_50_l772_772516


namespace find_x_l772_772749

-- Definition of the problem
def infinite_series (x : ℝ) : ℝ := ∑' n : ℕ, (n + 1) * x^n

-- Given condition
axiom condition : infinite_series x = 4

-- Statement to prove
theorem find_x : (∃ x : ℝ, infinite_series x = 4) → x = 1/2 := by
  sorry

end find_x_l772_772749


namespace sum_of_prime_f_values_is_zero_l772_772609

noncomputable def f (n : ℕ) : ℤ :=
  n^4 - 400*n^2 + 600

theorem sum_of_prime_f_values_is_zero :
  ∑ n in finset.range (n + 1), if (nat.prime (f n)) then (f n) else 0 = 0 :=
sorry

end sum_of_prime_f_values_is_zero_l772_772609


namespace increasing_function_k_l772_772046

open Real

theorem increasing_function_k (k : ℝ) : 
  (∀ x > 0, deriv (λ x, (log x) / x - k * x) x > 0) ↔ k ≤ -1 / (2 * exp 3) :=
by
  sorry

end increasing_function_k_l772_772046


namespace ball_arrangement_count_l772_772878

theorem ball_arrangement_count :
  let total_balls := 9
  let red_balls := 2
  let yellow_balls := 3
  let white_balls := 4
  let arrangements := Nat.factorial total_balls / (Nat.factorial red_balls * Nat.factorial yellow_balls * Nat.factorial white_balls)
  arrangements = 1260 :=
by
  sorry

end ball_arrangement_count_l772_772878


namespace projection_of_b_on_c_l772_772742

variables (a b c : ℝ × ℝ)
variables (m : ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def norm_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2
def projection (v u : ℝ × ℝ) : ℝ × ℝ := 
  let scalar := (dot_product v u) / (norm_squared u)
  in (scalar * u.1, scalar * u.2)

theorem projection_of_b_on_c :
  let a := (2, m)
  let b := (-1, 2)
  let c := (a.1 - b.1, a.2 - b.2)
  (dot_product a b = 0) →
  m = 1 →
  projection b c = (-√10 / 2, √10 / 2) :=
sorry

end projection_of_b_on_c_l772_772742


namespace max_value_x_minus_y_l772_772688

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l772_772688


namespace sum_x_coords_common_points_l772_772111

theorem sum_x_coords_common_points : 
  ∃ (x_coords : Finset ℕ), 
    (∀ x ∈ x_coords, (∃ y, y ≡ 5 * x + 2 [MOD 16] ∧ y ≡ 11 * x + 12 [MOD 16])) ∧
    x_coords.sum id = 10 := 
by
  -- Define the conditions
  let cond := λ x, ∃ y, y ≡ 5 * x + 2 [MOD 16] ∧ y ≡ 11 * x + 12 [MOD 16]

  -- Derive the values of x that satisfy the conditions
  have sol : Finset ℕ := {1, 9}

  -- Prove these are the only solutions within the range 0 ≤ x < 16
  have all_sols : ∀ x, x < 16 → cond x ↔ x ∈ sol := sorry

  -- Prove the sum of the x-coordinates
  use sol
  split
  { intros x hx
    rw [all_sols x _] at hx
    exact hx
  }
  { rw [Finset.sum_eq_add_sum_compl, Finset.sum_singleton, Finset.sum_singleton, add_zero]
    refl
  }
  sorry

end sum_x_coords_common_points_l772_772111


namespace limit_fraction_pow_zero_of_arith_geo_seq_l772_772725

theorem limit_fraction_pow_zero_of_arith_geo_seq (a c : ℝ) (h_arith : a + c = 2) (h_geo : a^2 * c^2 = 1) (h_neq : a ≠ c) :
  (∃ L, L = 0 ∧ (∀ ε > 0, ∃ N, ∀ n ≥ N, | ( (a + c) / (a^2 + c^2) )^n - L | < ε)) :=
by
  use 0
  intro ε hε
  use 1
  intro n hn
  simp
  sorry

end limit_fraction_pow_zero_of_arith_geo_seq_l772_772725


namespace kelly_games_l772_772362

theorem kelly_games (initial_games : ℕ) (desired_games : ℕ) : initial_games = 50 → desired_games = 35 → (initial_games - desired_games) = 15 :=
begin
  intros h1 h2,
  rw [h1, h2],
  exact rfl,
end

end kelly_games_l772_772362


namespace triangle_inequality_l772_772351

theorem triangle_inequality (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a < b + c) (h₄ : b < a + c) (h₅ : c < a + b) :
    let s := (a + b + c) / 2 in
    1 / (s - a) + 1 / (s - b) + 1 / (s - c) ≥ 2 * (1 / a + 1 / b + 1 / c) :=
by
  sorry

end triangle_inequality_l772_772351


namespace reading_time_difference_l772_772912

theorem reading_time_difference :
  let xanthia_reading_speed := 100 -- pages per hour
  let molly_reading_speed := 50 -- pages per hour
  let book_pages := 225
  let xanthia_time := book_pages / xanthia_reading_speed
  let molly_time := book_pages / molly_reading_speed
  let difference_in_hours := molly_time - xanthia_time
  let difference_in_minutes := difference_in_hours * 60
  difference_in_minutes = 135 := by
  sorry

end reading_time_difference_l772_772912


namespace twenty_fifty_yuan_bills_unique_l772_772138

noncomputable def twenty_fifty_yuan_bills (x y : ℕ) : Prop :=
  x + y = 260 ∧ 20 * x + 50 * y = 100 * 100

theorem twenty_fifty_yuan_bills_unique (x y : ℕ) (h : twenty_fifty_yuan_bills x y) :
  x = 100 ∧ y = 160 :=
by
  sorry

end twenty_fifty_yuan_bills_unique_l772_772138


namespace f_comp_g_eq_g_comp_f_has_solution_l772_772389

variable {R : Type*} [Field R]

def f (a b x : R) : R := a * x + b
def g (c d x : R) : R := c * x ^ 2 + d

theorem f_comp_g_eq_g_comp_f_has_solution (a b c d : R) :
  (∃ x : R, f a b (g c d x) = g c d (f a b x)) ↔ (c = 0 ∨ a * b = 0) ∧ (a * d - c * b ^ 2 + b - d = 0) := by
  sorry

end f_comp_g_eq_g_comp_f_has_solution_l772_772389


namespace trapezium_other_side_length_l772_772206

theorem trapezium_other_side_length (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ) :
  side1 = 20 ∧ distance = 15 ∧ area = 285 → side2 = 18 :=
by
  intro h
  sorry

end trapezium_other_side_length_l772_772206


namespace max_value_x_minus_y_l772_772692

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l772_772692


namespace chipmunk_families_went_away_l772_772592

theorem chipmunk_families_went_away :
  ∀ (total_families left_families went_away_families : ℕ),
  total_families = 86 →
  left_families = 21 →
  went_away_families = total_families - left_families →
  went_away_families = 65 :=
by
  intros total_families left_families went_away_families ht hl hw
  rw [ht, hl] at hw
  exact hw

end chipmunk_families_went_away_l772_772592


namespace sum_of_angles_of_sixth_roots_l772_772061

theorem sum_of_angles_of_sixth_roots (z : ℂ) (r : ℂ → ℝ) (θ : ℂ → ℝ) 
  (h1 : z^6 = 64 * complex.I)
  (h2 : ∀ k, k = z → r k > 0 ∧ 0 ≤ θ k ∧ θ k < 360) :
  ∑ k in finset.range 6, θ (z * (complex.cis (k * (π/3)))) = 990 :=
by
  sorry

end sum_of_angles_of_sixth_roots_l772_772061


namespace Q_difference_l772_772233

def Q (x n : ℕ) : ℕ :=
  (Finset.range (10^n)).sum (λ k => x / (k + 1))

theorem Q_difference (n : ℕ) : 
  Q (10^n) n - Q (10^n - 1) n = (n + 1)^2 :=
by
  sorry

end Q_difference_l772_772233


namespace number_of_female_democrats_l772_772506

variables (F M D_f : ℕ)

def total_participants := F + M = 660
def female_democrats := D_f = F / 2
def male_democrats := (F / 2) + (M / 4) = 220

theorem number_of_female_democrats 
  (h1 : total_participants F M) 
  (h2 : female_democrats F D_f) 
  (h3 : male_democrats F M) : 
  D_f = 110 := by
  sorry

end number_of_female_democrats_l772_772506


namespace range_of_m_l772_772285

noncomputable def quadratic_polynomial (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + m^2 - 2

theorem range_of_m (m : ℝ) (h1 : ∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ quadratic_polynomial m x1 = 0 ∧ quadratic_polynomial m x2 = 0) :
  0 < m ∧ m < 1 :=
sorry

end range_of_m_l772_772285


namespace primes_digit_sum_difference_l772_772268

def is_prime (a : ℕ) : Prop := Nat.Prime a

def sum_digits (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

theorem primes_digit_sum_difference (p q r : ℕ) (n : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (hneq : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (hpqr : p * q * r = 1899 * 10^n + 962) :
  (sum_digits p + sum_digits q + sum_digits r - sum_digits (p * q * r) = 8) := 
sorry

end primes_digit_sum_difference_l772_772268


namespace proof_problem_l772_772228

variable (a b : ℝ)

-- Given Conditions
def func (x : ℝ) : ℝ := a * Real.log x + b * x + 1

-- Given that the derivative at x = 1 is equal to 2
axiom slope_condition : (a / 1 + b = 2)

-- Proof problem
theorem proof_problem 
  (h1 : a + b = 2) :
  (a * b <= 1) ∧
  (a^2 + b^2 >= 2) ∧
  (3^a + 3^b >= 6) :=
by
  sorry

end proof_problem_l772_772228


namespace circle_equation_l772_772464

theorem circle_equation (x y : ℝ) (h_eq : x = 0) (k_eq : y = -2) (r_eq : y = 4) :
  (x - 0)^2 + (y - (-2))^2 = 16 := 
by
  sorry

end circle_equation_l772_772464


namespace correct_result_l772_772809

def multiply_and_add_round (x y z : ℝ) : ℝ :=
  let result := (x * y) + z
  (result * 100).round / 100

theorem correct_result :
  multiply_and_add_round 53.463 12.9873 10.253 = 705.02 :=
by
  sorry

end correct_result_l772_772809


namespace num_ways_to_assign_positions_l772_772606

-- Define the constants for the problem
def num_contestants : Nat := 5
def num_positions : Nat := 3

-- Define the main theorem to state the number of ways to assign positions
theorem num_ways_to_assign_positions :
  (5.choose(3)) * 3! = 60 :=
by
  sorry

end num_ways_to_assign_positions_l772_772606


namespace sine_double_angle_l772_772227

theorem sine_double_angle (theta : ℝ) (h : Real.tan (theta + Real.pi / 4) = 2) : Real.sin (2 * theta) = 3 / 5 :=
sorry

end sine_double_angle_l772_772227


namespace weekly_goal_of_cans_l772_772497

theorem weekly_goal_of_cans (initial_cans : ℕ) (increment_per_day : ℕ) (days : ℕ) : 
  initial_cans = 20 → increment_per_day = 5 → days = 5 → (∑ i in Finset.range days, initial_cans + i * increment_per_day) = 150 := 
by 
  sorry

end weekly_goal_of_cans_l772_772497


namespace batsman_average_after_17th_inning_l772_772916

theorem batsman_average_after_17th_inning (A : ℝ) :
  let total_runs_after_17_innings := 16 * A + 87
  let new_average := total_runs_after_17_innings / 17
  new_average = A + 3 → 
  (A + 3) = 39 :=
by
  sorry

end batsman_average_after_17th_inning_l772_772916


namespace exists_y_less_than_half_p_l772_772388

theorem exists_y_less_than_half_p (p : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) :
  ∃ (y : ℕ), y < p / 2 ∧ ∀ (a b : ℕ), p * y + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by sorry

end exists_y_less_than_half_p_l772_772388


namespace total_amount_paid_l772_772522

/-- Conditions -/
def days_in_may : Nat := 31
def rate_per_day : ℚ := 0.5
def days_book1_borrowed : Nat := 20
def days_book2_borrowed : Nat := 31
def days_book3_borrowed : Nat := 31

/-- Question and Proof -/
theorem total_amount_paid : rate_per_day * (days_book1_borrowed + days_book2_borrowed + days_book3_borrowed) = 41 := by
  sorry

end total_amount_paid_l772_772522


namespace part_a_solution_exists_l772_772426

theorem part_a_solution_exists : ∃ (x y : ℕ), x^2 - y^2 = 31 ∧ x = 16 ∧ y = 15 := 
by 
  sorry

end part_a_solution_exists_l772_772426


namespace remainder_when_sum_divided_by_5_l772_772279

/-- Reinterpreting the same conditions and question: -/
theorem remainder_when_sum_divided_by_5 (a b c : ℕ) 
    (ha : a < 5) (hb : b < 5) (hc : c < 5) 
    (h1 : a * b * c % 5 = 1) 
    (h2 : 3 * c % 5 = 2)
    (h3 : 4 * b % 5 = (3 + b) % 5): 
    (a + b + c) % 5 = 4 := 
sorry

end remainder_when_sum_divided_by_5_l772_772279


namespace maximum_value_of_x_minus_y_l772_772697

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l772_772697


namespace baseball_speed_after_collision_l772_772519

-- Define the given masses and initial velocities
def m : ℝ := 0.2 -- mass of the baseball in kg
def M : ℝ := 0.5 -- mass of the basketball in kg
def v_baseball : ℝ := -4.0 -- speed of the baseball before collision in m/s (downward)
def v_basketball : ℝ := 4.0 -- speed of the basketball before collision in m/s (upward)
def V_prime : ℝ := 0 -- speed of the basketball after collision in m/s (at rest)

-- Define the initial and final momentum
def p_initial : ℝ := M * v_basketball + m * v_baseball
def p_final (v_prime : ℝ) : ℝ := M * V_prime + m * v_prime

-- The goal is to prove that the speed of the baseball after collision is 6.0 m/s
theorem baseball_speed_after_collision : 
  ∃ v_prime : ℝ, p_initial = p_final v_prime ∧ v_prime = 6.0 :=
by
  -- Declare the target speed after collision
  let v_prime := 6.0
  -- Calculate the initial momentum
  have h1 : p_initial = 1.2 := by 
    unfold p_initial v_baseball v_basketball M m
    norm_num
  -- Calculate the final momentum with the target speed
  have h2 : p_final v_prime = 1.2 := by 
    unfold p_final V_prime m
    norm_num
  -- Conclude the proof
  existsi v_prime
  constructor
  case left => exact h1
  case right => exact rfl
  sorry

end baseball_speed_after_collision_l772_772519


namespace shortest_hair_l772_772789

def hair_lengths (Junseop_cm : ℝ) (Junseop_mm : ℝ) (Taehun_cm : ℝ) (Hayul_cm : ℝ) : ℝ :=
  let Junseop_total_cm := Junseop_cm + Junseop_mm / 10
  min Taehun_cm (min Junseop_total_cm Hayul_cm)

theorem shortest_hair :
  hair_lengths 9 8 8.9 9.3 = 8.9 :=
by
  -- converting 8 millimeters to 0.8 centimeters and comparing the lengths to find the minimum
  sorry

end shortest_hair_l772_772789


namespace function_decreasing_on_interval_l772_772414

theorem function_decreasing_on_interval (f : ℝ → ℝ) (h_def : ∀ x, f x = 2 * x / (x - 1)) :
  ∀ x1 x2 > 1, x1 > x2 → f x1 < f x2 := 
by
  sorry

end function_decreasing_on_interval_l772_772414


namespace maximum_value_of_x_minus_y_l772_772698

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l772_772698


namespace sin_a_2022_l772_772259

noncomputable def a : ℕ → ℕ 
| 0     := 1 
| (n+1) := (1 + 1 / (n + 1)) * a n + 1 / (n + 1)

theorem sin_a_2022 :
  sin (π * a 2022 / 3) = -√3 / 2 :=
by 
  sorry

end sin_a_2022_l772_772259


namespace sequence_general_term_l772_772249

theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, a n = S n - S (n-1) :=
by
  -- The proof will be filled in here
  sorry

end sequence_general_term_l772_772249


namespace lines_perpendicular_to_same_line_are_parallel_l772_772010

-- Definitions for the lines and plane
def line (α : Type) := α → α → Prop
def plane (α : Type) := set (line α)

-- Definitions for perpendicularity and parallelism
def perpendicular {α : Type} (l1 l2 : line α) : Prop := sorry -- Assuming the definition is provided
def parallel {α : Type} (l1 l2 : line α) : Prop := sorry -- Assuming the definition is provided

-- The main statement
theorem lines_perpendicular_to_same_line_are_parallel {α : Type} (P : plane α) (l m n : line α) :
  (l ∈ P ∧ m ∈ P ∧ n ∈ P) →
  (perpendicular l n ∧ perpendicular m n) →
  parallel l m :=
by sorry

end lines_perpendicular_to_same_line_are_parallel_l772_772010


namespace half_recipe_flour_half_recipe_flour_mixed_l772_772134

theorem half_recipe_flour (flour : ℚ) (h : flour = 9/2) : flour / 2 = 9 / 4 := by
  rw [h]
  norm_num
  sorry

theorem half_recipe_flour_mixed (flour : ℚ) (h : flour = 9/2) : (flour / 2) = 2 + 1 / 4 := by
  rw [h]
  norm_num
  sorry

end half_recipe_flour_half_recipe_flour_mixed_l772_772134


namespace binary_calculation_l772_772568

theorem binary_calculation :
  let b1 := 0b110110
  let b2 := 0b101110
  let b3 := 0b100
  let expected_result := 0b11100011110
  ((b1 * b2) / b3) = expected_result := by
  sorry

end binary_calculation_l772_772568


namespace T_8_is_85_l772_772376

def isValidString (s : String) : Prop :=
  (∀ i : Nat, 2 ≤ s.data.enumFrom(i).take(3).sum (fun c => if c = '1' then 1 else 0))

def T (n : Nat) : Finset String :=
  {s | s.length = n ∧ isValidString s}

def B1 : ℕ → ℕ
| 1 => 1
| n+1 => B1(n) + B2(n)

def B2 : ℕ → ℕ
| 2 => 1
| n+1 => B1(n)

def Tn (n : ℕ) : ℕ := B1(n) + B2(n)

theorem T_8_is_85 : Tn 8 = 85 :=
by
  sorry

end T_8_is_85_l772_772376


namespace problem_conditions_max_min_values_l772_772287

theorem problem_conditions_max_min_values (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_eq : 2 * m + n = 1) : 
  (∀ x y, x > 0 → y > 0 → 2 * x + y = 1 → x * y ≤ 1/8) ∧ (∀ x y, x > 0 → y > 0 → 2 * x + y = 1 → 4 * x^2 + y^2 ≥ 1/2) :=
by {
  split,
  sorry,
  sorry,
}

end problem_conditions_max_min_values_l772_772287


namespace geoff_needed_more_votes_to_win_l772_772307

-- Definitions based on the conditions
def total_votes : ℕ := 6000
def percent_to_fraction (p : ℕ) : ℚ := p / 100
def geoff_percent : ℚ := percent_to_fraction 1
def win_percent : ℚ := percent_to_fraction 51

-- Specific values derived from the conditions
def geoff_votes : ℚ := geoff_percent * total_votes
def win_votes : ℚ := win_percent * total_votes + 1

-- The theorem we intend to prove
theorem geoff_needed_more_votes_to_win :
  (win_votes - geoff_votes) = 3001 := by
  sorry

end geoff_needed_more_votes_to_win_l772_772307


namespace eq_of_divisibility_condition_l772_772003

theorem eq_of_divisibility_condition (a b : ℕ) (h : ∃ᶠ n in Filter.atTop, (a^n + b^n) ∣ (a^(n+1) + b^(n+1))) : a = b :=
sorry

end eq_of_divisibility_condition_l772_772003


namespace max_x_minus_y_l772_772661

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l772_772661


namespace sum_of_people_equals_sum_of_houses_l772_772812

-- Definitions based on the conditions provided.
def number_of_people_in_houses {n : ℕ} (a : ℕ → ℕ) := ∑ i in Finset.range n, a i

-- Indicator function for b_i
def indicator (p : Prop) [decidable p] : ℕ := if p then 1 else 0

-- Summing the indicator functions
def number_of_houses_with_at_least {n : ℕ} (a : ℕ → ℕ) (i : ℕ) := ∑ k in Finset.range i, 
  ∑ j in Finset.range n, indicator (a j ≥ k)

-- Final theorem statement
theorem sum_of_people_equals_sum_of_houses
  (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ)
  (h : ∀ k, b k = ∑ j in Finset.range n, indicator (a j ≥ k)) :
  number_of_people_in_houses a = ∑ k in Finset.range (∞), b k :=
sorry

end sum_of_people_equals_sum_of_houses_l772_772812


namespace ratio_of_kendra_to_sam_l772_772829

-- Define variables for ages
variables {S U K: ℕ}

-- Define the conditions from the problem
def sam_is_twice_as_old_as_sue := S = 2 * U
def kendra_age := K = 18
def total_age_in_three_years := S + U + K + 9 = 36

-- The proof problem
theorem ratio_of_kendra_to_sam
  (h1: sam_is_twice_as_old_as_sue)
  (h2: kendra_age)
  (h3: total_age_in_three_years) :
  K / S = 3 :=
by
  -- We prove the statement here
  sorry

end ratio_of_kendra_to_sam_l772_772829


namespace derivative_at_one_l772_772437

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_at_one : deriv f 1 = -1 := sorry

end derivative_at_one_l772_772437


namespace friendly_triangle_angle_l772_772990

theorem friendly_triangle_angle (α : ℝ) (β : ℝ) (γ : ℝ) (hα12β : α = 2 * β) (h_sum : α + β + γ = 180) :
    (α = 42 ∨ α = 84 ∨ α = 92) ∧ (42 = β ∨ 42 = γ) := 
sorry

end friendly_triangle_angle_l772_772990


namespace second_train_speed_l772_772480

theorem second_train_speed (length_train1 length_train2 : ℕ) (speed_train1 : ℝ) (time_to_clear : ℝ)
    (H_length1 : length_train1 = 100) 
    (H_length2 : length_train2 = 160) 
    (H_speed_train1 : speed_train1 = 42)
    (H_time : time_to_clear = 12.998960083193344) : 
    let total_length := length_train1 + length_train2 in 
    let time_in_hours := time_to_clear / 3600 in 
    let V_rel := total_length / 1000 / time_in_hours in 
    let speed_train2 := V_rel - speed_train1 in 
    speed_train2 = 30 :=
by
    sorry

end second_train_speed_l772_772480


namespace max_x_minus_y_l772_772640

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772640


namespace correct_quadratic_eqn_l772_772773

-- Definitions based on the conditions
def sum_of_roots (α β : ℝ) : ℝ := α + β
def product_of_roots (α β : ℝ) : ℝ := α * β

-- Given the conditions:
axiom first_student_roots : sum_of_roots 3 7 = 10
axiom second_student_roots : product_of_roots 5 (-1) = -5

-- We need to prove the correct quadratic equation is x^2 - 10x - 5 = 0
theorem correct_quadratic_eqn : ∃ b c : ℝ, b = -10 ∧ c = -5 ∧ (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x^2 - 10x - 5 = 0) := 
by 
  existsi (-10 : ℝ)
  existsi (-5 : ℝ)
  split
  { refl }
  split
  { refl }
  intro x
  transitivity
  { norm_num }
  { sorry }

end correct_quadratic_eqn_l772_772773


namespace unique_passenger_counts_l772_772150

def train_frequencies : Nat × Nat × Nat := (6, 4, 3)
def train_passengers_leaving : Nat × Nat × Nat := (200, 300, 150)
def train_passengers_taking : Nat × Nat × Nat := (320, 400, 280)
def trains_per_hour (freq : Nat) : Nat := 60 / freq

def total_passengers_leaving : Nat :=
  let t1 := (trains_per_hour 10) * 200
  let t2 := (trains_per_hour 15) * 300
  let t3 := (trains_per_hour 20) * 150
  t1 + t2 + t3

def total_passengers_taking : Nat :=
  let t1 := (trains_per_hour 10) * 320
  let t2 := (trains_per_hour 15) * 400
  let t3 := (trains_per_hour 20) * 280
  t1 + t2 + t3

theorem unique_passenger_counts :
  total_passengers_leaving = 2850 ∧ total_passengers_taking = 4360 := by
  sorry

end unique_passenger_counts_l772_772150


namespace problem_solution_l772_772635

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
axiom condition1 : ∃ d : ℝ, d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d
axiom condition2 : a 3 + a 7 = 20
axiom condition3 : a 2 = real.sqrt (a 1 * a 4)

-- General formula
def a_n (n : ℕ) : ℝ := 2 * n

-- Sequence b_n and its sum T_n
def b_n (n : ℕ) : ℝ := 2^(a n) + 4 / (a n * a (n + 1))
def T_n (n : ℕ) : ℝ := (∑ k in finset.range n, b k)

-- Statement to prove
theorem problem_solution :
  (∀ n, a n = a_n n) ∧
  (∀ n, T_n n = (4 / 3 * (4^n - 1)) + (n / (n + 1))) :=
by
  sorry

end problem_solution_l772_772635


namespace divisor_of_1025_l772_772090

theorem divisor_of_1025 : ∃ k : ℕ, 41 * k = 1025 :=
  sorry

end divisor_of_1025_l772_772090


namespace maximum_value_of_x_minus_y_l772_772695

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l772_772695


namespace average_speed_round_trip_l772_772103

theorem average_speed_round_trip :
  ∀ (D : ℝ), D > 0 → 
  let speed_PQ := 50 in
  let speed_QP := speed_PQ + 0.5 * speed_PQ in
  let time_PQ := D / speed_PQ in
  let time_QP := D / speed_QP in
  let total_distance := 2 * D in
  let total_time := time_PQ + time_QP in 
  (total_distance / total_time) = 60 :=
by
  intros D hD speed_PQ speed_QP time_PQ time_QP total_distance total_time
  sorry

end average_speed_round_trip_l772_772103


namespace airport_distance_l772_772577

-- Variables and conditions
variable (d t : ℝ)

-- Initially drives 45 miles in the first hour
def first_leg_distance := 45

-- Time taken to travel t hours more at 45 mph
def first_leg_time := 1.0 
def lateness := 0.75

-- Speed increased by 20 mph after the first hour
def increased_speed := 65

-- He arrives 15 minutes early
def earliness := 0.25

-- Total distance equation when continuing at the same speed
def continued_speed_equation : Prop := d = 45 * (t + lateness)

-- Total distance equation when increasing speed
def increased_speed_equation : Prop := (d - 45) = 65 * (t - earliness)

-- Main Theorem to prove
theorem airport_distance :
  (continued_speed_equation d t ∧ increased_speed_equation d t) → d = 61.875 := 
by
  sorry

end airport_distance_l772_772577


namespace angle_Q_is_72_degrees_l772_772027

-- Define the regular decagon and the internal angle
def regular_decagon (A B C D E F G H I J : Type) :=
  ∀ (n : ℕ), (n < 10) → true -- This is a placeholder definition

def internal_angle_regular_decagon := 144

-- The proposition to prove
theorem angle_Q_is_72_degrees (A B C D E F G H I J Q : Type)
  (h_regular : regular_decagon A B C D E F G H I J)
  (h_internal_angle : internal_angle_regular_decagon = 144):
  Q = 72 := 
  sorry

end angle_Q_is_72_degrees_l772_772027


namespace right_triangle_XYZ_properties_l772_772774

noncomputable def XY := 15 / Real.tan (Real.pi * 50 / 180) -- Using radians for trigonometric function
noncomputable def XZ := Real.sqrt (XY^2 + 15^2)

theorem right_triangle_XYZ_properties :
  ∀ (XY XZ : ℝ), 
  XY = 15 / Real.tan (Real.pi * 50 / 180) ∧ 
  XZ = Real.sqrt (XY^2 + 15^2) -> 
  XY ≈ 12.59 ∧ XZ ≈ 19.59 :=
by
  sorry

end right_triangle_XYZ_properties_l772_772774


namespace sin_sub_halfpi_eq_neg_cos_cos_add_halfpi_eq_sin_l772_772827

variable (α : ℝ)

theorem sin_sub_halfpi_eq_neg_cos (α : ℝ) : sin (3 * π / 2 - α) = -cos α := sorry

theorem cos_add_halfpi_eq_sin (α : ℝ) : cos (3 * π / 2 + α) = sin α := sorry

end sin_sub_halfpi_eq_neg_cos_cos_add_halfpi_eq_sin_l772_772827


namespace simplify_factorial_expression_l772_772834

theorem simplify_factorial_expression : (13.factorial / (10.factorial + 3 * 9.factorial) = 1320) :=
by
  sorry

end simplify_factorial_expression_l772_772834


namespace find_side_length_of_cube_l772_772548

theorem find_side_length_of_cube (n : ℕ) :
  (4 * n^2 = (1/3) * 6 * n^3) -> n = 2 :=
by
  sorry

end find_side_length_of_cube_l772_772548


namespace employed_males_percent_l772_772331

variable (population X: ℝ)
variable (employment_ratio: population > 0)
variable (employed_ratio: population = 0.64 * X)
variable (employed_female_ratio: population = 0.28125 * employed_ratio)

theorem employed_males_percent (population = 1): (1 - (28.125/100)) * 0.64 = 0.4596 :=
by
  sorry

end employed_males_percent_l772_772331


namespace fixed_point_trajectory_straight_line_l772_772539

theorem fixed_point_trajectory_straight_line :
  ∀ (O : Point) (R : ℝ) 
  (C₁ C₂ : Circle) (P : Point),
  C₁.radius = 2 * C₂.radius → 
  (∃ (tracePoint : Circle → Point → Path), 
    (circle_rolls_without_slipping C₁ C₂ tracePoint)) →
    trajectory P (C₂.roll_inside_circle C₁) = straight_line_segment O :=
sorry

end fixed_point_trajectory_straight_line_l772_772539


namespace length_of_AD_l772_772887

noncomputable theory

variable (A B C O D : Type) [IsTriangle A B C]
variables (b : ℝ) (α : ℝ)
variables (is_isosceles : AB = BC)
variables (angle_ABC : ∠ABC = α)
variables (center_O : is_circumcenter O A B C)
variables (line_through_AO : is_line_through O A D)
variables (intersect_D : intersect_point BC D)

theorem length_of_AD :
  AD = b * sin α / sin (3 * α / 2) :=
sorry

end length_of_AD_l772_772887


namespace mode_of_data_set_l772_772049

def data_set : List ℤ := [-1, 1, 2, 2, 3]

theorem mode_of_data_set : mode data_set = 2 := by sorry

end mode_of_data_set_l772_772049


namespace max_x_minus_y_l772_772658

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l772_772658


namespace max_x_minus_y_l772_772655

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l772_772655


namespace minimum_positive_period_l772_772869

theorem minimum_positive_period (x : ℝ) : 
    ∃ T > 0, ∀ x, 3 * Real.cos(2 / 5 * x - Real.pi / 6) = 3 * Real.cos(2 / 5 * (x + T) - Real.pi / 6) :=
by sorry

end minimum_positive_period_l772_772869


namespace alexandre_wins_game_l772_772387

theorem alexandre_wins_game (n : ℕ) (h : n % 2 = 0 ∧ n > 3) :
  (∃ strategy : (ℕ → ℕ → Prop), 
  (∀ Mona_turn : ℕ → ℕ, ∃ Alexandre_turn : ℕ → ℕ, 
    (∀ k, strategy k Mona_turn Alexandre_turn) 
    → ¬(∃ i, (Mona_turn i + Mona_turn (i+1) + Mona_turn (i+2)) % 3 = 0))) :=
sorry

end alexandre_wins_game_l772_772387


namespace intersection_result_union_complements_result_l772_772793

open Set

variable (U M N : Set ℝ)

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem intersection_result : M ∩ N = {x | 1 ≤ x ∧ x < 5} :=
by sorry

theorem union_complements_result : (U \ M) ∪ (U \ N) = {x | x < 1 ∨ x ≥ 5} :=
by sorry

end intersection_result_union_complements_result_l772_772793


namespace shoe_length_size_15_l772_772102

theorem shoe_length_size_15 : 
  ∀ (length : ℕ → ℝ), 
    (∀ n, 8 ≤ n ∧ n ≤ 17 → length (n + 1) = length n + 1 / 4) → 
    length 17 = (1 + 0.10) * length 8 →
    length 15 = 24.25 :=
by
  intro length h_increase h_largest
  sorry

end shoe_length_size_15_l772_772102


namespace female_managers_count_l772_772296

def E : ℕ -- total number of employees E
def M : ℕ := E - 500 -- number of male employees (M = E - 500)
def total_managers : ℕ := (2/5) * E -- total number of managers ((2/5)E)
def male_managers : ℕ := (2/5) * M -- number of male managers ((2/5)M)
def female_managers : ℕ := total_managers - male_managers -- number of female managers (total_managers - male_managers)
def company_total_managers: E : ℕ → total_managers : ℕ→ female_ubalnce_constraints: female_managers
theorem female_managers_count : female_managers = 200 := sorry

end female_managers_count_l772_772296


namespace binomial_coefficient_formula_l772_772611

theorem binomial_coefficient_formula (n k : ℕ) (h : 0 ≤ k ∧ k ≤ n) : nat.choose n k = nat.factorial n / (nat.factorial (n - k) * nat.factorial k) :=
by sorry

end binomial_coefficient_formula_l772_772611


namespace no_more_beverages_needed_l772_772015

namespace HydrationPlan

def daily_water_need := 9
def daily_juice_need := 5
def daily_soda_need := 3
def days := 60

def total_water_needed := daily_water_need * days
def total_juice_needed := daily_juice_need * days
def total_soda_needed := daily_soda_need * days

def water_already_have := 617
def juice_already_have := 350
def soda_already_have := 215

theorem no_more_beverages_needed :
  (water_already_have >= total_water_needed) ∧ 
  (juice_already_have >= total_juice_needed) ∧ 
  (soda_already_have >= total_soda_needed) :=
by 
  -- proof goes here
  sorry

end HydrationPlan

end no_more_beverages_needed_l772_772015


namespace compare_abc_l772_772795

noncomputable def a : ℝ := 4 ^ (Real.log 2 / Real.log 3)
noncomputable def b : ℝ := 4 ^ (Real.log 6 / (2 * Real.log 3))
noncomputable def c : ℝ := 2 ^ (Real.sqrt 5)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end compare_abc_l772_772795


namespace max_x_minus_y_l772_772666

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772666


namespace area_of_complex_polygon_l772_772886

-- Defining the problem
def area_of_polygon (side1 side2 side3 : ℝ) (rot1 rot2 : ℝ) : ℝ :=
  -- This is a placeholder definition.
  -- In a complete proof, here we would calculate the area based on the input conditions.
  sorry

-- Main theorem statement
theorem area_of_complex_polygon :
  area_of_polygon 4 5 6 (π / 4) (-π / 6) = 72 :=
by sorry

end area_of_complex_polygon_l772_772886


namespace double_series_sum_eq_one_l772_772190

theorem double_series_sum_eq_one :
  (∑' m : ℕ, ∑' n : ℕ, (1 / (m+1) * ((m+n+3)⁻¹))) = 1 :=
sorry

end double_series_sum_eq_one_l772_772190


namespace total_profit_l772_772535

-- Definitions based on conditions
def P_x := 3 * k
def P_y := 2 * k

theorem total_profit (k : ℕ) (h1 : P_x = 3 * k) (h2 : P_y = 2 * k) (h3 : P_x - P_y = 140) :
  P_x + P_y = 700 := 
by
  sorry

end total_profit_l772_772535


namespace max_value_of_y_l772_772293

noncomputable def max_y (x y : ℝ) : ℝ := 
if h : 0 < x ∧ 0 < y ∧ x * log (y / x) - y * exp x + x * (x + 1) ≥ 0 then y else 0

theorem max_value_of_y : ∀ (x y : ℝ), 
  0 < x ∧ 0 < y ∧ x * log (y / x) - y * exp x + x * (x + 1) ≥ 0 → 
  y ≤ 1 / exp 1 := 
by 
  intros x y h
  sorry

end max_value_of_y_l772_772293


namespace erwan_total_expenditure_l772_772590

theorem erwan_total_expenditure
  (shoe_price : ℤ) (shoe_discount : ℤ)
  (shirt_price : ℤ) (num_shirts : ℕ)
  (additional_discount : ℤ) :
  shoe_price = 200 →
  shoe_discount = 30 →
  shirt_price = 80 →
  num_shirts = 2 →
  additional_discount = 5 →
  let discounted_shoe_price := shoe_price - (shoe_price * shoe_discount / 100) in
  let shirt_total_price := num_shirts * shirt_price in
  let total_price_before_additional_discount := discounted_shoe_price + shirt_total_price in
  let total_additional_discount := total_price_before_additional_discount * additional_discount / 100 in
  let final_price := total_price_before_additional_discount - total_additional_discount in
  final_price = 285 :=
sorry

end erwan_total_expenditure_l772_772590


namespace max_value_change_l772_772094

open Real

variables (f : ℝ → ℝ) (x : ℝ)

-- Conditions
def condition1 : Prop := ∀ f, (∃ M1 : ℝ, ∀ x : ℝ, f(x) ≤ M1 ∧ ∃ a, a + x ^ 2 = f ⟹ f(M1 + x ^ 2) - M1 = 27 / 2)
def condition2 : Prop := ∀ f, (∃ M2 : ℝ, ∀ x : ℝ, f(x) ≤ M2 ∧ ∃ b, b - 4 * x ^ 2 = f ⟹ f(M2 - 4 * x ^ 2) - M2 = -9)

-- Statement to prove
theorem max_value_change (f : ℝ → ℝ) 
  (h1 : condition1 f) 
  (h2 : condition2 f) : 
  ∃ C : ℝ, ∀ x : ℝ, C = - 27 / 4 ∧ ∃ c, c - 2 * x ^ 2 = f ⟹ f (C - 2 * x ^ 2) = f C :=
sorry

end max_value_change_l772_772094


namespace min_triangle_area_l772_772633

variable {θ0 : ℝ} (hθ0 : θ0 ≠ π / 2)

def l (ρ : ℝ) : Prop := θ = θ0
def C (θ : ℝ) (ρ : ℝ) : Prop := ρ * (Real.sin θ) ^ 2 = 4 * Real.cos θ
def l' (ρ : ℝ) : Prop := θ = θ0 + π / 2

theorem min_triangle_area (hθ0 : θ0 ≠ π / 2) :
    ∃ M N : ℝ × ℝ, M ≠ (0, 0) ∧ N ≠ (0, 0) ∧
    C M.1 M.2 ∧ C N.1 N.2 ∧
    l M.2 ∧ l' N.2 ∧
    ∃ OM ON : ℝ, OM = |M.2| ∧ ON = |N.2| ∧
    (OM * ON / 2 = 16) := sorry

end min_triangle_area_l772_772633


namespace worker_number_in_40th_segment_l772_772547

   theorem worker_number_in_40th_segment :
     (total_staff segments : ℕ) (staff_per_segment starting_point segment_index : ℕ) 
     (h1 : total_staff = 620)
     (h2 : segments = 62)
     (h3 : staff_per_segment = total_staff / segments)
     (h4 : starting_point = 4)
     (h5 : segment_index = 40) :
     starting_point + (segment_index - 1) * staff_per_segment = 394 :=
   by
     rw [h1, h2, h3, h4, h5]
     sorry
   
end worker_number_in_40th_segment_l772_772547


namespace complex_inequality_l772_772625

theorem complex_inequality 
  (z1 z2 z3 : ℂ)
  (h : z1^2 + z2^2 > -z3^2) : 
  z1^2 + z2^2 + z3^2 > 0 :=
sorry

end complex_inequality_l772_772625


namespace largest_number_among_options_l772_772553

theorem largest_number_among_options :
  let a := -3
  let b := 0
  let c := 2
  let d := | -1 |
in max (max a b) (max c d) = c :=
by
  sorry

end largest_number_among_options_l772_772553


namespace smallest_b_value_l772_772851

theorem smallest_b_value (a b : ℕ) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : b = 3 := sorry

end smallest_b_value_l772_772851


namespace largest_integer_k_real_roots_l772_772289

theorem largest_integer_k_real_roots :
  ∃ (k : ℤ), ∀ x : ℝ, (x * (k * x + 1) - x ^ 2 + 3 = 0 → 
  (let a := (k : ℝ) - 1, b := 1, c := 3 in
   b ^ 2 - 4 * a * c ≥ 0)) ∧
  (k = 1) := sorry

end largest_integer_k_real_roots_l772_772289


namespace wire_length_15_rounds_l772_772209

theorem wire_length_15_rounds
  (A : ℕ)
  (hA : A = 69696) :
  let s := Nat.sqrt A in
  let P := 4 * s in
  let L := 15 * P in
  L = 15840 :=
by
  sorry

end wire_length_15_rounds_l772_772209


namespace all_valid_a_divisible_l772_772991

def divisible_by_12321 (a k : ℤ) : Prop :=
  (a ^ k + 1) % 12321 = 0

def valid_a (a : ℤ) : Prop :=
  a % 111 ∈ ({11, 41, 62, 65, 77, 95, 101, 104, 110} : set ℤ)

theorem all_valid_a_divisible : ∀ a : ℤ, 
  (∃ k : ℤ, divisible_by_12321 a k) ↔ valid_a a :=
by
  sorry

end all_valid_a_divisible_l772_772991


namespace original_price_of_house_l772_772016

theorem original_price_of_house (P : ℝ) 
  (h1 : P * 0.56 = 56000) : P = 100000 :=
sorry

end original_price_of_house_l772_772016


namespace possible_sums_of_digits_l772_772514

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def all_digits_nonzero (A : ℕ) : Prop :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def reverse_number (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  1000 * d + 100 * c + 10 * b + a

def sum_of_digits (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a + b + c + d

theorem possible_sums_of_digits (A B : ℕ) 
  (h_four_digit : is_four_digit_number A) 
  (h_nonzero_digits : all_digits_nonzero A) 
  (h_reverse : B = reverse_number A) 
  (h_divisible : (A + B) % 109 = 0) : 
  sum_of_digits A = 14 ∨ sum_of_digits A = 23 ∨ sum_of_digits A = 28 := 
sorry

end possible_sums_of_digits_l772_772514


namespace no_solution_except_eq_l772_772518

theorem no_solution_except_eq (A B: ℝ) (x y: ℝ): (A = 0 ∨ (B / A) ∈ Ioo (-2 : ℝ) 0) → (Ax + B * ⌊x⌋ = Ay + B * ⌊y⌋ → x = y) :=
sorry

end no_solution_except_eq_l772_772518


namespace angle_congruence_l772_772320

variables (A B C D E F P Q : Type)
variables [ConvexQuad A B C D] [OnSegment F A D] [OnSegment E B C]
variables [eq_ratios: (AF / FD) = (BE / EC) = (AB / CD)]
variables [ExtendIntersects EF P A B] [ExtendIntersects EF Q C D]

def equal_angles (X Y Z : Type) := ∠ X Y E = ∠ X Z E

theorem angle_congruence :
  equal_angles B P E C Q E :=
sorry

end angle_congruence_l772_772320


namespace trains_passing_time_l772_772925

-- Define the lengths of the trains
def train_length : ℕ := 190

-- Define the speeds of the trains in km/h
def speed_train1_kmh : ℕ := 65
def speed_train2_kmh : ℕ := 50

-- Convert speeds from km/h to m/s
def speed_train1_ms : ℝ := (speed_train1_kmh * 1000) / 3600
def speed_train2_ms : ℝ := (speed_train2_kmh * 1000) / 3600

-- Calculate relative speed in m/s
def relative_speed_ms : ℝ := speed_train1_ms + speed_train2_ms

-- Calculate total length to be covered for trains to pass each other
def total_length : ℕ := 2 * train_length

-- Calculate the time it takes for the trains to pass each other completely
noncomputable def passing_time : ℝ := total_length / relative_speed_ms

theorem trains_passing_time : passing_time ≈ 11.89 := 
by sorry

end trains_passing_time_l772_772925


namespace max_common_elements_in_sequences_l772_772847

-- Define an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ a1 d : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 2024 → a n = a1 + (n - 1) * d)

-- Define a geometric sequence
def is_geometric_seq (b : ℕ → ℕ) : Prop :=
  ∃ b1 r : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 2024 → b n = b1 * r^(n - 1))

-- Define the problem
theorem max_common_elements_in_sequences (a b : ℕ → ℕ) 
  (ha : is_arithmetic_seq a) (hb : is_geometric_seq b) : 
  ∃ k : ℕ, k ≤ 11 ∧ ∀ x, (x ∈ set_of (λ n, 1 ≤ n ∧ n ≤ 2024 ∧ a n = b n)) ↔ k = 11 :=
sorry

end max_common_elements_in_sequences_l772_772847


namespace remainder_div_by_13_l772_772525

-- Define conditions
variable (N : ℕ)
variable (k : ℕ)

-- Given condition
def condition := N = 39 * k + 19

-- Goal statement
theorem remainder_div_by_13 (h : condition N k) : N % 13 = 6 :=
sorry

end remainder_div_by_13_l772_772525


namespace largest_sum_is_sum3_l772_772166

-- Definitions of the individual sums given in the conditions
def sum1 : ℚ := (1/4 : ℚ) + (1/5 : ℚ) * (1/2 : ℚ)
def sum2 : ℚ := (1/4 : ℚ) - (1/6 : ℚ)
def sum3 : ℚ := (1/4 : ℚ) + (1/3 : ℚ) * (1/2 : ℚ)
def sum4 : ℚ := (1/4 : ℚ) - (1/8 : ℚ)
def sum5 : ℚ := (1/4 : ℚ) + (1/7 : ℚ) * (1/2 : ℚ)

-- Theorem to prove that sum3 is the largest
theorem largest_sum_is_sum3 : sum3 = (5/12 : ℚ) ∧ sum3 > sum1 ∧ sum3 > sum2 ∧ sum3 > sum4 ∧ sum3 > sum5 := 
by 
  -- The proof would go here
  sorry

end largest_sum_is_sum3_l772_772166


namespace tangency_point_proof_l772_772604

noncomputable def pointOfTangency : ℝ × ℝ :=
  (-(7 / 2), -(15 / 2))

theorem tangency_point_proof :
  ∃ x y : ℝ, 
  y = x^2 + 8 * x + 15 ∧ 
  x = y^2 + 16 * y + 63 ∧ 
  (x, y) = pointOfTangency :=
by
  use (-7 / 2)
  use (-15 / 2)
  split; sorry  -- First goal: y = x^2 + 8*x + 15
  split; sorry  -- Second goal: x = y^2 + 16*y + 63
                -- Third goal is automatically true by the definition of pointOfTangency

end tangency_point_proof_l772_772604


namespace avg_weight_of_class_l772_772880

def A_students : Nat := 36
def B_students : Nat := 44
def C_students : Nat := 50
def D_students : Nat := 30

def A_avg_weight : ℝ := 40
def B_avg_weight : ℝ := 35
def C_avg_weight : ℝ := 42
def D_avg_weight : ℝ := 38

def A_additional_students : Nat := 5
def A_additional_weight : ℝ := 10

def B_reduced_students : Nat := 7
def B_reduced_weight : ℝ := 8

noncomputable def total_weight_class : ℝ :=
  (A_students * A_avg_weight + A_additional_students * A_additional_weight) +
  (B_students * B_avg_weight - B_reduced_students * B_reduced_weight) +
  (C_students * C_avg_weight) +
  (D_students * D_avg_weight)

noncomputable def total_students_class : Nat :=
  A_students + B_students + C_students + D_students

noncomputable def avg_weight_class : ℝ :=
  total_weight_class / total_students_class

theorem avg_weight_of_class :
  avg_weight_class = 38.84 := by
    sorry

end avg_weight_of_class_l772_772880


namespace find_eel_fat_l772_772942

def herring_fat : ℕ := 40
def pike_extra_fat : ℕ := 10
def fish_count : ℕ := 40
def total_fat_served : ℕ := 3600

variable (E : ℕ)

theorem find_eel_fat (E_oz : E) :
  (fish_count * herring_fat) + (fish_count * E) + (fish_count * (E + pike_extra_fat)) = total_fat_served → 
  E = 20 :=
by
  sorry

end find_eel_fat_l772_772942


namespace max_pairs_k_l772_772223

theorem max_pairs_k:
  let k := 899 in
  ∃ (a b : Fin 3001 → ℕ → ℕ) 
    (distinct : ∀ i j, i ≠ j → (a i b ≠ a j b ∧ a i b ≠ a j b)) 
    (distinct_sums : ∀ i j, i ≠ j → a i + b i ≠ a j + b j) 
    (pairs_less_than_3000 : ∀ i, a i < b i)  
    (sum_le_3000 : ∀ i, a i + b i ≤ 3000), 
  True := sorry

end max_pairs_k_l772_772223


namespace find_smaller_integer_l772_772079

theorem find_smaller_integer : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ y = x + 8 ∧ x * y = 80 ∧ x = 2 :=
by
  sorry

end find_smaller_integer_l772_772079


namespace prize_winners_l772_772770

theorem prize_winners (total_people : ℕ) (percent_envelope : ℝ) (percent_win : ℝ) 
  (h_total : total_people = 100) (h_percent_envelope : percent_envelope = 0.40) 
  (h_percent_win : percent_win = 0.20) : 
  (percent_win * (percent_envelope * total_people)) = 8 := by
  sorry

end prize_winners_l772_772770


namespace ratio_rounded_to_nearest_tenth_l772_772187

theorem ratio_rounded_to_nearest_tenth : 
  (Float.round (11 / 16 : Float) * 10) / 10 = 0.7 :=
by
  -- sorry is used because the proof steps are not required in this task.
  sorry

end ratio_rounded_to_nearest_tenth_l772_772187


namespace num_ballpoint_pens_l772_772881

-- Define the total number of school supplies
def total_school_supplies : ℕ := 60

-- Define the number of pencils
def num_pencils : ℕ := 5

-- Define the number of notebooks
def num_notebooks : ℕ := 10

-- Define the number of erasers
def num_erasers : ℕ := 32

-- Define the number of ballpoint pens and prove it equals 13
theorem num_ballpoint_pens : total_school_supplies - (num_pencils + num_notebooks + num_erasers) = 13 :=
by
sorry

end num_ballpoint_pens_l772_772881


namespace least_money_l772_772962

-- Declare the individuals
variable Alice Bob Charlie Dana Eve : ℝ

-- Declare the conditions from the problem
theorem least_money :
  (Alice ≠ Bob) ∧ (Alice ≠ Charlie) ∧ (Alice ≠ Dana) ∧ (Alice ≠ Eve) ∧
  (Bob ≠ Charlie) ∧ (Bob ≠ Dana) ∧ (Bob ≠ Eve) ∧
  (Charlie ≠ Dana) ∧ (Charlie ≠ Eve) ∧ (Dana ≠ Eve) ∧
  (∀ x, x ≠ Charlie → Charlie > x) ∧
  (Bob > Alice) ∧ (Dana > Alice) ∧ (Eve > Alice) ∧ (Eve < Bob)
  → Alice < Bob ∧ Alice < Charlie ∧ Alice < Dana ∧ Alice < Eve := 
by sorry

end least_money_l772_772962


namespace consecutive_integers_exist_l772_772183

def good (n : ℕ) : Prop :=
∃ (k : ℕ) (a : ℕ → ℕ), 
  (∀ i j, 1 ≤ i → i < j → j ≤ k → a i < a j) ∧ 
  (∀ i j i' j', 1 ≤ i → i < j → j ≤ k → 1 ≤ i' → i' < j' → j' ≤ k → a i + a j = a i' + a j' → i = i' ∧ j = j') ∧ 
  (∃ (t : ℕ), ∀ m, 0 ≤ m → m < n → ∃ i j, 1 ≤ i → i < j → j ≤ k → a i + a j = t + m)

theorem consecutive_integers_exist (n : ℕ) (h : n = 1000) : good n :=
sorry

end consecutive_integers_exist_l772_772183


namespace female_managers_count_l772_772297

-- Definitions for the problem statement

def total_female_employees : ℕ := 500
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Problem parameters
variable (E M FM : ℕ) -- E: total employees, M: male employees, FM: female managers

-- Conditions
def total_employees_eq : Prop := E = M + total_female_employees
def total_managers_eq : Prop := fraction_of_managers * E = fraction_of_male_managers * M + FM

-- The statement we want to prove
theorem female_managers_count (h1 : total_employees_eq E M) (h2 : total_managers_eq E M FM) : FM = 200 :=
by
  -- to be proven
  sorry

end female_managers_count_l772_772297


namespace find_m_integral_eq_zero_l772_772214

theorem find_m_integral_eq_zero :
  ∃ m : ℝ, (∫ x in 0..1, x^2 + m * x) = 0 ∧ m = -(2 / 3) :=
by
  sorry

end find_m_integral_eq_zero_l772_772214


namespace propositions_verification_l772_772963

/- Definitions and conditions based on the problem -/
def converse_statement (x y : ℝ) : Prop := (x * y = 1) ↔ (x = 1/y)

def negation_statement : Prop := ∃ (T₁ T₂ : Triangle), congruent T₁ T₂ ∧ perimeter T₁ ≠ perimeter T₂

def contrapositive_statement (b : ℝ) : Prop := (∃ x : ℝ, x^2 - 2 * b * x + b^2 + b = 0) ↔ (b > -1)

def sine_equation_statement : Prop := ∃ (α β : ℝ), sin (α + β) = sin α + sin β

/- Lean theorem stating our claims -/
theorem propositions_verification :
  (∀ x y : ℝ, converse_statement x y) ∧
  ¬ negation_statement ∧
  (∀ b : ℝ, contrapositive_statement b) ∧
  sine_equation_statement :=
by
  -- Proof skipped
  sorry

end propositions_verification_l772_772963


namespace jaya_rank_from_bottom_l772_772357

theorem jaya_rank_from_bottom (total_students : ℕ) (top_rank : ℕ) (bottom_rank : ℕ) 
    (H1 : total_students = 53) (H2 : top_rank = 5) : bottom_rank = 50 :=
begin
  sorry,
end

end jaya_rank_from_bottom_l772_772357


namespace number_of_students_l772_772035

theorem number_of_students (n S : ℕ) 
  (h1 : S = 15 * n) 
  (h2 : (S + 36) / (n + 1) = 16) : 
  n = 20 :=
by 
  sorry

end number_of_students_l772_772035


namespace robert_coin_arrangement_l772_772420

variables (G S : ℕ) (engraved : bool) (vertical_stack : list ℕ)

-- Robert's conditions
def robert_conditions := (G = 5) ∧
                         (S = 5) ∧
                         (engraved = true) ∧
                         (vertical_stack.length = 10)

-- The arrangement count problem
def possible_arrangements (G S : ℕ) : ℕ :=
  if robert_conditions then 2772 else 0

-- Theorem statement
theorem robert_coin_arrangement :
  possible_arrangements 5 5 = 2772 :=
by { unfold possible_arrangements, rw if_pos, apply robert_conditions; sorry }

end robert_coin_arrangement_l772_772420


namespace fruit_transport_max_profit_l772_772586

noncomputable def profit (x y z : ℕ) : ℤ := 7 * 1200 * x + 6 * 1800 * y + 5 * 1500 * z

theorem fruit_transport :
  ∃ (x y z : ℕ), 
    7 * x + 6 * y + 5 * z = 120 ∧
    x + y + z = 20 ∧
    x ≥ 3 ∧ y ≥ 3 ∧ z ≥ 3 :=
begin
  /- Proof will involve checking integer values for x, y, z to satisfy the conditions. -/
  sorry
end

theorem max_profit :
  ∃ (x y z : ℕ), 
    7 * x + 6 * y + 5 * z = 120 ∧
    x + y + z = 20 ∧
    x ≥ 3 ∧ y ≥ 3 ∧ z ≥ 3 ∧
    ∀ (x' y' z' : ℕ),
      7 * x' + 6 * y' + 5 * z' = 120 ∧ x' + y' + z' = 20 ∧ x' ≥ 3 ∧ y' ≥ 3 ∧ z' ≥ 3 →
      profit x y z ≥ profit x' y' z' :=
begin
  /- Proof will involve checking integer values for x, y, z to meet the given conditions and maximize profit. -/
  sorry
end

end fruit_transport_max_profit_l772_772586


namespace volleyball_practice_start_time_l772_772428

def homework_time := 1 * 60 + 59  -- convert 1:59 p.m. to minutes since 12:00 p.m.
def homework_duration := 96        -- duration in minutes
def buffer_time := 25              -- time between finishing homework and practice
def practice_start_time := 4 * 60  -- convert 4:00 p.m. to minutes since 12:00 p.m.

theorem volleyball_practice_start_time :
  homework_time + homework_duration + buffer_time = practice_start_time := 
by
  sorry

end volleyball_practice_start_time_l772_772428


namespace max_x_minus_y_l772_772663

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l772_772663


namespace must_divide_p_l772_772379

theorem must_divide_p (p q r s : ℕ) 
  (hpq : Nat.gcd p q = 45)
  (hqr : Nat.gcd q r = 75)
  (hrs : Nat.gcd r s = 90)
  (hspt : 150 < Nat.gcd s p)
  (hspb : Nat.gcd s p < 200) : 10 ∣ p := by
  sorry

end must_divide_p_l772_772379


namespace elasticity_ratio_is_correct_l772_772160

-- Definitions of the given elasticities
def e_OGBR_QN : ℝ := 1.27
def e_OGBR_PN : ℝ := 0.76

-- Theorem stating the ratio of elasticities equals 1.7
theorem elasticity_ratio_is_correct : (e_OGBR_QN / e_OGBR_PN) = 1.7 := sorry

end elasticity_ratio_is_correct_l772_772160


namespace length_of_goods_train_l772_772528

theorem length_of_goods_train 
  (speed_kmph : ℝ) (platform_length : ℝ) (time_sec : ℝ) (train_length : ℝ) 
  (h1 : speed_kmph = 72)
  (h2 : platform_length = 270) 
  (h3 : time_sec = 26) 
  (h4 : train_length = (speed_kmph * 1000 / 3600 * time_sec) - platform_length)
  : train_length = 250 := 
  by
    sorry

end length_of_goods_train_l772_772528


namespace difference_of_remainders_is_2310_l772_772583

theorem difference_of_remainders_is_2310 :
  let k := 2 * 3 * 5 * 7 * 11
  in let n₁ := 2 + k
  in let n₂ := 2 + 2 * k
  in n₂ - n₁ = 2310 :=
by
  let k := 2 * 3 * 5 * 7 * 11
  let n₁ := 2 + k
  let n₂ := 2 + 2 * k
  sorry

end difference_of_remainders_is_2310_l772_772583


namespace domain_tan_3x_sub_pi_over_4_l772_772860

noncomputable def domain_of_f : Set ℝ :=
  {x : ℝ | ∀ k : ℤ, x ≠ (k * Real.pi) / 3 + Real.pi / 4}

theorem domain_tan_3x_sub_pi_over_4 :
  ∀ x : ℝ, x ∈ domain_of_f ↔ ∀ k : ℤ, x ≠ (k * Real.pi) / 3 + Real.pi / 4 :=
by
  intro x
  sorry

end domain_tan_3x_sub_pi_over_4_l772_772860


namespace expand_and_simplify_powers_of_two_one_more_than_cube_l772_772931

-- Part (i) statement
theorem expand_and_simplify : ∀ x : ℝ, (x + 1) * (x^2 - x + 1) = x^3 + 1 :=
by
  intro x
  sorry

-- Part (ii) statement
theorem powers_of_two_one_more_than_cube :
  ∀ n : ℕ, 2^n = (∃ k : ℕ, 2^n = k^3 + 1) ↔ (n = 0 ∨ n = 1) :=
by
  intro n
  sorry

end expand_and_simplify_powers_of_two_one_more_than_cube_l772_772931


namespace mass_copper_added_correct_l772_772533

variables (initial_mass total_mass-copper_correct added_copper_correct: ℝ) -- declaring the variables

def is_alloy(src_alloy: ℝ):= src_alloy = 36 

def initial_copper(src: ℝ):= src * 0.45

def new_mass_with_copper(init_c:item ℝ, added_item ℝ):= init_c + added_item

def total_mass_alloy(new_item: ℝ,added: ℝ ):= new_item + added

def percentage_of_copper (mass_copper: ℝ, mass_alloy: ℝ) := mass_copper / mass_alloy

theorem mass_copper_added_correct (initial_mass : ℝ)(added_copper_correct: ℝ)(correct_mass: ℝ):
is_alloy initial_mass→added_copper_correct = 13.5 → correct_mass = initial_mass +added_copper_correct→ percentage_of_copper correct_mass (36 +13.5) = .6:=
by
intro h1
intro h2
intro h3
sorry

end mass_copper_added_correct_l772_772533


namespace average_speed_of_the_car_l772_772935

variable (s1 s2 s3 s4 s5 : ℝ)
variable (t : ℝ)
variable (total_distance : ℝ)
variable (average_speed : ℝ)

-- Define the given conditions
def conditions : Prop :=
  s1 = 70 ∧ s2 = 90 ∧ s3 = 80 ∧ s4 = 100 ∧ s5 = 60 ∧ t = 5 ∧
  total_distance = s1 + s2 + s3 + s4 + s5 ∧
  average_speed = total_distance / t

-- State the theorem
theorem average_speed_of_the_car :
  conditions s1 s2 s3 s4 s5 t total_distance average_speed → average_speed = 80 :=
by
  intro h
  cases h with h_speeds h_rest
  cases h_rest with h_t h_rest2
  cases h_rest2 with h_td h_avg_spd
  sorry

end average_speed_of_the_car_l772_772935


namespace select_m_minus_one_vectors_l772_772022

open Classical

variables {M : ℕ} (hM : M ≥ 2) (v : Fin M → ℝ^3)

noncomputable def unit_vector (i : Fin M) : Prop :=
  ‖v i‖ = 1

theorem select_m_minus_one_vectors (H : ∀ i, unit_vector v i) : 
  ∃ (s : Fin (M-1) → Fin M), 
    (Finset.univ.image s).card = M-1 ∧ 
    ‖(Finset.univ.image (s ∘ Fin.ofNat)).sum (λ i, v i)‖ ≥ 1 :=
sorry

end select_m_minus_one_vectors_l772_772022


namespace min_sum_x_y_l772_772282

open Real

theorem min_sum_x_y (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : 2 * x + 8 * y - x * y = 0) : 
  x + y = 18 :=
begin
  -- the actual proof steps will be written here
  -- however, as per instructions, we're skipping the proof
  sorry
end

end min_sum_x_y_l772_772282


namespace exactly_2_std_devs_less_than_mean_l772_772855

noncomputable def mean : ℝ := 14.5
noncomputable def std_dev : ℝ := 1.5
noncomputable def value : ℝ := mean - 2 * std_dev

theorem exactly_2_std_devs_less_than_mean : value = 11.5 := by
  sorry

end exactly_2_std_devs_less_than_mean_l772_772855


namespace overall_ranking_l772_772956

-- Define the given conditions
def total_participants := 99
def rank_number_theory := 16
def rank_combinatorics := 30
def rank_geometry := 23
def exams := ["geometry", "number_theory", "combinatorics"]
def final_ranking_strategy := "sum_of_scores"

-- Given: best possible rank and worst possible rank should be the same in this specific problem (from solution steps).
def best_possible_rank := 67
def worst_possible_rank := 67

-- Mathematically prove that 100 * best possible rank + worst possible rank = 167
theorem overall_ranking :
  100 * best_possible_rank + worst_possible_rank = 167 :=
by {
  -- Add the "sorry" here to skip the proof, as required:
  sorry
}

end overall_ranking_l772_772956


namespace erwan_total_spending_l772_772589

-- Definitions
def price_of_shoes : ℝ := 200
def discount_on_shoes : ℝ := 30 / 100
def price_of_shirt : ℝ := 80
def quantity_of_shirts : ℕ := 2
def additional_discount : ℝ := 5 / 100

-- Calculation of discounted prices
def discounted_price_shoes : ℝ :=
  price_of_shoes * (1 - discount_on_shoes)

def total_price_shirts : ℝ :=
  price_of_shirt * quantity_of_shirts

def initial_total_cost : ℝ :=
  discounted_price_shoes + total_price_shirts

def checkout_discount : ℝ :=
  initial_total_cost * additional_discount

def final_total_cost : ℝ :=
  initial_total_cost - checkout_discount

-- Proof statement
theorem erwan_total_spending : final_total_cost = 285 :=
by
  sorry

end erwan_total_spending_l772_772589


namespace second_number_is_correct_l772_772467

theorem second_number_is_correct :
  ∀ (x y z : ℝ), 
    x + y + z = 120 ∧ 
    x = (3 / 4) * y ∧ 
    z = (9 / 7) * y -> 
      y = 39.53 :=
by
  intros x y z h
  cases h with h_sum h_rest
  cases h_rest with h_ratio1 h_ratio2
  sorry

end second_number_is_correct_l772_772467


namespace angle_equality_l772_772322

variables {A B C D E F P Q: Type*}
variables [convex_quadrilateral A B C D]
variables [point_on_segment F A D]
variables [point_on_segment E B C]
variables [ratio AF FD BE EC AB CD]

theorem angle_equality : 
  let P := intersection (extension_segment E F) (line_through A B),
      Q := intersection (extension_segment E F) (line_through C D)
  in ∠B P E = ∠C Q E :=
sorry

end angle_equality_l772_772322


namespace math_books_probability_l772_772404

theorem math_books_probability :
  let boxes := [3, 4, 5]
  let total_books := 12
  let math_books := 3
  let total_ways := Nat.choose total_books 3 * Nat.choose (total_books - 3) 4 * Nat.choose (total_books - 7) 5
  let favorable_ways := Nat.choose 9 0 * Nat.choose 9 4 * Nat.choose 5 5 + Nat.choose 9 1 * Nat.choose 8 3 * Nat.choose 5 5 + Nat.choose 9 2 * Nat.choose 7 3 * Nat.choose 4 4
  let probability := rational.mk favorable_ways total_ways
  probability = rational.mk 3 44 → 3 + 44 = 47 := by
  sorry

end math_books_probability_l772_772404


namespace find_page_number_l772_772762

theorem find_page_number (n p : ℕ) (h1 : (n * (n + 1)) / 2 + 2 * p = 2046) : p = 15 :=
sorry

end find_page_number_l772_772762


namespace points_on_circle_l772_772791

open EuclideanGeometry

variables {A B C T U R S : Point}
variables (triangle : Triangle A B C)
variables (hT : Collinear A B T)
variables (hU : Collinear A C U)
variables (h1 : Distance B T = Distance C U)
variables (hR : Collinear A B R)
variables (hS : Collinear A C S)
variables (h2 : Distance A S = Distance A T)
variables (h3 : Distance A R = Distance A U)

theorem points_on_circle (triangle : Triangle A B C)
  (hT : Collinear A B T) (hU : Collinear A C U) (h1 : Distance B T = Distance C U)
  (hR : Collinear A B R) (hS : Collinear A C S)
  (h2 : Distance A S = Distance A T) (h3 : Distance A R = Distance A U) :
  ∃ (O : Point), IsOnCircumcircle O R S T U ∧ IsOnCircumcircle O A B C :=
by
  sorry

end points_on_circle_l772_772791


namespace tan_alpha_plus_pi_l772_772392

theorem tan_alpha_plus_pi (α : ℝ) (hα1 : 0 < α) (hα2 : α < π) (hcos : cos (π - α) = 1 / 3) : 
  tan (α + π) = -2 * real.sqrt 2 := 
  sorry

end tan_alpha_plus_pi_l772_772392


namespace radius_equals_side_l772_772008

-- Definitions based on the conditions
def is_equilateral_triangle (T : Type) := -- To be filled with the exact mathematical definitions
sorry

def has_six_identical_circles (T : Type) (radius : ℝ) := -- Define the identical circles and tangency conditions
sorry

def radius_of_inscribed_circle (T : Type) (r : ℝ) := -- Define the radius of the inscribed circle
sorry

def side_of_inscribed_decagon (T : Type) (a : ℝ) := -- Define the regular decagon within the circles
sorry

-- The statement to prove
theorem radius_equals_side (T : Type) (r a : ℝ) (triangle : is_equilateral_triangle T)
  (circles_property : has_six_identical_circles T r) (inscribed_circle : radius_of_inscribed_circle T r)
  (inscribed_decagon : side_of_inscribed_decagon T a) : r = a :=
by
  sorry

end radius_equals_side_l772_772008


namespace equilateral_triangle_l772_772247

noncomputable theory
open Real

def hyperbola (x : ℝ) : ℝ := 1 / x

def A (a : ℝ) (h : a > 0) : ℝ × ℝ := (a, hyperbola a)

def B (a : ℝ) (h : a > 0) : ℝ × ℝ := (-a, -hyperbola a)

def AB_dist (a : ℝ) (h : a > 0) : ℝ :=
  dist (A a h) (B a h)

def circle_eq (a : ℝ) (h : a > 0) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - (hyperbola a))^2 = (AB_dist a h / 2)^2

def intersection_points (a : ℝ) (h : a > 0) : set (ℝ × ℝ) :=
  { p | circle_eq a h p.1 p.2 ∧ p.2 = hyperbola p.1 }

theorem equilateral_triangle (a : ℝ) (h : a > 0) :
  ∃ P Q R : ℝ × ℝ, 
  P ∈ intersection_points a h ∧
  Q ∈ intersection_points a h ∧
  R ∈ intersection_points a h ∧
  (equilateral_triangle P Q R) :=
sorry

end equilateral_triangle_l772_772247


namespace max_x_minus_y_l772_772670

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l772_772670


namespace marcella_shoes_l772_772105

theorem marcella_shoes :
  ∀ (original_pairs lost_shoes : ℕ), original_pairs = 27 → lost_shoes = 9 → 
  ∃ (remaining_pairs : ℕ), remaining_pairs = 18 ∧ remaining_pairs ≤ original_pairs - lost_shoes / 2 :=
by
  intros original_pairs lost_shoes h1 h2
  use 18
  constructor
  . exact rfl
  . sorry

end marcella_shoes_l772_772105


namespace not_possible_triangle_l772_772222

theorem not_possible_triangle (a : Real) : ¬(a + a > 2 * a ∧ a + 2 * a > a ∧ 2 * a + a > a) := 
by
  have h1 : a + a > 2 * a → False := by sorry
  have h2 : a + 2 * a > a := by linarith
  have h3 : 2 * a + a > a := by linarith
  exact h1

end not_possible_triangle_l772_772222


namespace largest_cube_edge_from_cone_l772_772472

theorem largest_cube_edge_from_cone : 
  ∀ (s : ℝ), 
  (s = 2) → 
  ∃ (x : ℝ), x = 3 * Real.sqrt 2 - 2 * Real.sqrt 3 :=
by
  sorry

end largest_cube_edge_from_cone_l772_772472


namespace max_x_minus_y_l772_772660

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l772_772660


namespace XO_probability_l772_772471

open Classical
noncomputable theory

/-- There are four tiles marked X, and three tiles marked O.
The seven tiles are randomly arranged in a row.
What is the probability that the two outermost positions in the arrangement are occupied by X
and the middle one by O? -/
def probability_XO_arrangement : ℚ :=
  let total_arrangements := Nat.choose 7 4 in
  let favorable_arrangements := Nat.choose 4 2 in
  favorable_arrangements / total_arrangements

theorem XO_probability:
  probability_XO_arrangement = 6 / 35 :=
by
  sorry

end XO_probability_l772_772471
