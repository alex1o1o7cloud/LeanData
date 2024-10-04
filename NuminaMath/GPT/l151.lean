import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Totient
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Point
import Mathlib.LinearAlgebra.Finrank
import Mathlib.Probability.Basic
import Mathlib.SAlgebra.Module.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Trigonometry.Basic

namespace sec_150_eq_l151_151396

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151396


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151557

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151557


namespace parabola_curve_l151_151774

theorem parabola_curve :
  (∀ (x y : ℝ), sqrt ((x - 2)^2 + y^2) = abs (3 * x - 4 * y + 2) / 5) →
  (∃ (F : ℝ × ℝ) (d : ℝ → ℝ × ℝ → ℝ), (F = (2, 0)) ∧ (d = λ x y, 3 * x - 4 * y + 2) ∧ 
  (∀ P : ℝ × ℝ, sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = abs ((d P.1 P.2) / 5)) →
  ∀ P : ℝ × ℝ, sqrt ((P.1 - 2)^2 + P.2^2) = abs (3 * P.1 - 4 * P.2 + 2) / 5) :=
begin
  sorry
end

end parabola_curve_l151_151774


namespace total_number_of_squares_l151_151778

theorem total_number_of_squares (n : ℕ) (h : n = 12) : 
  ∃ t, t = 17 :=
by
  -- The proof is omitted here
  sorry

end total_number_of_squares_l151_151778


namespace lcm_18_60_is_180_l151_151829

theorem lcm_18_60_is_180 : Nat.lcm 18 60 = 180 := 
  sorry

end lcm_18_60_is_180_l151_151829


namespace extra_days_per_grade_below_b_l151_151898

theorem extra_days_per_grade_below_b :
  ∀ (total_days lying_days grades_below_B : ℕ), 
  total_days = 26 → lying_days = 14 → grades_below_B = 4 → 
  (total_days - lying_days) / grades_below_B = 3 :=
by
  -- conditions and steps of the proof will be here
  sorry

end extra_days_per_grade_below_b_l151_151898


namespace tan_double_angle_l151_151942

theorem tan_double_angle (α : ℝ) (h : tan (2 * α) = 4 / 3) :
  tan α = -2 ∨ tan α = 1 / 2 :=
sorry

end tan_double_angle_l151_151942


namespace num_valid_integers_M_l151_151021

theorem num_valid_integers_M :
  { M : ℕ | M < 2000 ∧
            ∃ ks : finset ℕ,
              (ks.card = 3 ∧
               ∀ k ∈ ks, ∃ m ≥ 1, M = k * (2 * m + k - 1)) 
  }.card = 10 :=
sorry

end num_valid_integers_M_l151_151021


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151453

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151453


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151380

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151380


namespace reconstruct_polygon_l151_151612

theorem reconstruct_polygon {n : ℕ} (h2013 : n = 2013) 
  (A B C D E : ℝ × ℝ) (h_reg : ∃ (P : ℕ → ℝ × ℝ), 
  (∀ i, P i = (P 0).rot (2 * π * i / n)) ∧ (P 0 = A) ∧ (P 1 = B) ∧ (P 2 = C) ∧ (P 3 = D) ∧ (P 4 = E)) :
  ∃ (P : ℕ → ℝ × ℝ), (∀ i, P i = (P 0).rot (2 * π * i / n)) :=
by
  sorry

end reconstruct_polygon_l151_151612


namespace cone_base_radius_l151_151293

theorem cone_base_radius 
  (R : ℝ) (θ : ℝ) (C_cone : ℝ)
  (hR : R = 4)
  (hθ : θ = 120)
  (hC_cone : C_cone = (1 / 3) * 2 * real.pi * R) :
  ∃ r : ℝ, 2 * real.pi * r = C_cone ∧ r = 4 / 3 :=
by {
  use (4 / 3),
  split,
  { -- Proof of 2 * π * r = C_cone
    rw [hR, hC_cone],
    simp,
    -- Continue proving the equality
    sorry
  },
  { -- Proof of r = 4 / 3
    simp,
    -- Continue proving the equality
    sorry
  }
}

end cone_base_radius_l151_151293


namespace david_pushups_more_than_zachary_l151_151264

theorem david_pushups_more_than_zachary :
  ∀ (Z D J : ℕ), Z = 51 → J = 69 → J = D - 4 → D = Z + 22 :=
by
  intros Z D J hZ hJ hJD
  sorry

end david_pushups_more_than_zachary_l151_151264


namespace probability_x_gt_4y_l151_151128

theorem probability_x_gt_4y :
  let rect := setOf (λ p : ℝ × ℝ, 0 ≤ p.1 ∧ p.1 ≤ 2010 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2011)
  let A := setOf (λ p : ℝ × ℝ, 0 ≤ p.1 ∧ p.1 ≤ 2010 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1/4 * p.1)
  P(rect : ℝ) (A : ℝ) = 505506.25 / 4043010 := sorry

end probability_x_gt_4y_l151_151128


namespace necessary_but_not_sufficient_l151_151997

variables (A B : Prop)

theorem necessary_but_not_sufficient 
  (h1 : ¬ B → ¬ A)  -- Condition: ¬ B → ¬ A is true
  (h2 : ¬ (¬ A → ¬ B))  -- Condition: ¬ A → ¬ B is false
  : (A → B) ∧ ¬ (B → A) := -- Conclusion: A → B and not (B → A)
by
  -- Proof is not required, so we place sorry
  sorry

end necessary_but_not_sufficient_l151_151997


namespace no_preimage_k_l151_151978

noncomputable def f (x : ℝ) : ℝ := log (2 - x) / log (1 / 2) - (1 / 3)^x

theorem no_preimage_k (k : ℝ) : 
  (∀ x ∈ set.Icc 0 1, f x ≠ k) ↔ k ∈ set.Iio (-2) ∪ set.Ioi (-1 / 3) := 
sorry

end no_preimage_k_l151_151978


namespace volume_region_zero_l151_151258

noncomputable def volume_of_intersection (x y z : ℝ) : ℝ :=
if ∀ (x y z : ℝ), |x| ≤ 1 ∧ |y| ≤ 1 ∧ |z| ≤ 1 ∧ |z - 2| ≤ 1
then 0 else sorry

theorem volume_region_zero :
  volume_of_intersection = 0 :=
sorry

end volume_region_zero_l151_151258


namespace remainder_of_7_pow_145_mod_12_l151_151826

theorem remainder_of_7_pow_145_mod_12 : (7 ^ 145) % 12 = 7 :=
by
  sorry

end remainder_of_7_pow_145_mod_12_l151_151826


namespace problem1_inequality_problem2_range_l151_151668

noncomputable def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 1)

theorem problem1_inequality (x : ℝ) : (f x ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := 
sorry

theorem problem2_range (m : ℝ) (h : -1 ≤ m ∧ m ≤ 3) : (∀ x : ℝ, f x ≥ m^2 - 2m) :=
sorry

end problem1_inequality_problem2_range_l151_151668


namespace sec_150_eq_l151_151464

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151464


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151575

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151575


namespace probability_last_two_digits_less_than_five_l151_151865

theorem probability_last_two_digits_less_than_five : 
  let total_digits := 10
  let valid_digits := 5
  let probability_one_digit_less_than_five := (valid_digits : ℚ) / total_digits
  in (probability_one_digit_less_than_five * probability_one_digit_less_than_five = 1 / 4) :=
by 
  let total_digits := 10
  let valid_digits := 5
  let probability_one_digit_less_than_five := (valid_digits : ℚ) / total_digits
  show (probability_one_digit_less_than_five * probability_one_digit_less_than_five = 1 / 4)
  sorry

end probability_last_two_digits_less_than_five_l151_151865


namespace no_real_satisfies_absolute_value_equation_l151_151990

theorem no_real_satisfies_absolute_value_equation :
  ∀ x : ℝ, ¬ (|x - 2| = |x - 1| + |x - 5|) :=
by
  sorry

end no_real_satisfies_absolute_value_equation_l151_151990


namespace number_of_children_l151_151152

theorem number_of_children 
  (A C : ℕ) 
  (h1 : A + C = 201) 
  (h2 : 8 * A + 4 * C = 964) : 
  C = 161 := 
sorry

end number_of_children_l151_151152


namespace calculate_q_l151_151763

theorem calculate_q (a b p q : ℝ) (ha : a ≠ b) :
  -- Condition: Positive difference conditions for roots
  (2 * |a - b| = real.sqrt (a^2 - 4 * b) ∧
  real.sqrt (a^2 - 4 * b) = real.sqrt (b^2 - 4 * a) ∧
  2 * |a - b| = real.sqrt (b^2 - 4 * a)) →
  -- Condition: a + b = -4
  (a + b = -4) →
  -- Conclusion to prove
  q = (16 / 5) :=
by sorry

end calculate_q_l151_151763


namespace analytical_expression_years_to_eight_times_height_l151_151853

section Problem

def f (A : ℝ) (a b t : ℝ) (x : ℕ) : ℝ :=
  9 * A / (a + b * t^x)

theorem analytical_expression (A : ℝ) : 
  let t := 2^(-2/3) in
  f A 1 8 t x = 9 * A / (1 + 8 * t^x) :=
by
  have t := 2^(-2/3)
  have a : ℝ := 1
  have b : ℝ := 8
  sorry

theorem years_to_eight_times_height (A : ℝ) : 
  let t := 2^(-2/3) in 
  let n := 9 in 
  f A 1 8 t 9 = 8 * A :=
by
  have t := 2^(-2/3)
  have a : ℝ := 1
  have b : ℝ := 8
  have n : ℕ := 9
  sorry

end Problem

end analytical_expression_years_to_eight_times_height_l151_151853


namespace angle_B_and_area_l151_151040

open Real

-- Definitions of the conditions
variables (A B C a b c : ℝ)
variables (tan_a : tan A = sqrt 2 / 2)
variables (c_val : c = sqrt 3)
variables (order : A < B ∧ B < C)
variables (side_relation : a^2 + c^2 - b^2 = ac)

-- Prove angle B and the area of triangle ABC
theorem angle_B_and_area (B_val : B = π / 3) : B = π / 3 ∧ (1/2 * a * c_val * sin B = 3 / 10 * (3 * sqrt 2 - sqrt 3)) :=
sorry

end angle_B_and_area_l151_151040


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151444

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151444


namespace coefficient_x4_in_expansion_l151_151228

theorem coefficient_x4_in_expansion : 
  let a := (1:ℝ)
  let b := -3 * (x^2)
  let n := 6
  (∑ k in Finset.range (n + 1), (Nat.choose n k) * (a ^ (n - k)) * (b ^ k)) = 135 * x^4
 :=
by
  sorry

end coefficient_x4_in_expansion_l151_151228


namespace class_average_score_l151_151044

theorem class_average_score (n_boys n_girls : ℕ) (avg_score_boys avg_score_girls : ℕ) 
  (h_nb : n_boys = 12)
  (h_ng : n_girls = 4)
  (h_ab : avg_score_boys = 84)
  (h_ag : avg_score_girls = 92) : 
  (n_boys * avg_score_boys + n_girls * avg_score_girls) / (n_boys + n_girls) = 86 := 
by 
  sorry

end class_average_score_l151_151044


namespace find_x_coordinate_range_of_C_l151_151645

/-- Given an isosceles triangle ABC with base points A and B on the hyperbola 
  x^2 / 6 - y^2 / 3 = 1, and vertex C on the x-axis, prove that the range of the 
  x-coordinate t of vertex C, given that the midpoint M(x₀, y₀) with x₀ > sqrt(6), 
  is (3/2 * sqrt(6), +∞) --/
theorem find_x_coordinate_range_of_C
  (A B : Point)
  (hA : A ∈ hyperbola)
  (hB : B ∈ hyperbola)
  (h_midpoint : Midpoint(A, B) = M)
  (h_Mx₀ : M.x > Real.sqrt 6) :
  ∃ t : ℝ, t > 3/2 * Real.sqrt(6) :=
sorry

definition Point := (ℝ, ℝ)

definition hyperbola (P : Point) : Prop :=
  (P.1^2 / 6 - P.2^2 / 3 = 1)

definition Midpoint (A B : Point) : Point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

definition M (x₀ y₀ : ℝ) :=
  (x₀, y₀)

end find_x_coordinate_range_of_C_l151_151645


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151569

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151569


namespace tangent_line_at_1_max_value_and_range_l151_151002

noncomputable def f (x : ℝ) (m : ℝ) (hm : m > 0) : ℝ := log x - m * x

theorem tangent_line_at_1 (hf : ∀ x : ℝ, f x 1 = log x - x) : tangent_line (1 : ℝ) (f 1 1) = -1 :=
by sorry

theorem max_value_and_range (m : ℝ) (hm : m > 0) : 
  (∀ x : ℝ, f x m = log x - m * x) →
  ((g : ℝ) (hf : f (1 / m) m = -log m - 1) → 
  g > m - 2 → 0 < m ∧ m < 1) :=
by sorry

end tangent_line_at_1_max_value_and_range_l151_151002


namespace find_infinite_subsequence_a₀_l151_151846

theorem find_infinite_subsequence_a₀ (a_0 : ℕ) (h₀ : 1 < a_0):
  (∃ (A : ℕ), ∀ᶠ (n : ℕ) in Filter.atTop, a_n = A) ↔ a_0 % 3 = 0 :=
by
  sorry

where
  a_n : ℕ → ℕ
  a_n 0 := a_0
  a_n (n+1) := if isInt (sqrt (a_n n)) then sqrt (a_n n) else a_n n + 3

end find_infinite_subsequence_a₀_l151_151846


namespace problem_solution_l151_151959

noncomputable def a_sequence : ℕ → ℝ
| 1     := 4041
| (n+1) := ∏ i in finset.range n, (1 + a_sequence i.succ)

lemma a_sequence_pos : ∀ n, a_sequence n > 0 := sorry

lemma sum_condition_lt : ∀ n : ℕ, (∑ k in finset.range n, 1 / (1 + a_sequence (k + 1))) < (1 / 2021) := sorry

lemma sum_condition_gt : ∀ c : ℝ, c < 1 / 2021 → ∃ n : ℕ, (∑ k in finset.range n, 1 / (1 + a_sequence (k + 1))) > c := sorry

theorem problem_solution : 
  ∃ a > 0, (∀ n : ℕ, (∑ k in finset.range n, 1 / (1 + a_sequence (k + 1))) < (1 / 2021)) ∧ 
             (∀ c : ℝ, c < 1 / 2021 → ∃ n : ℕ, (∑ k in finset.range n, 1 / (1 + a)) > c) := 
begin
  refine ⟨4041, by norm_num, sum_condition_lt, sum_condition_gt⟩,
end

end problem_solution_l151_151959


namespace dave_paid_more_l151_151353

-- Definitions based on conditions in the problem statement
def total_pizza_cost : ℕ := 11  -- Total cost of the pizza in dollars
def num_slices : ℕ := 8  -- Total number of slices in the pizza
def plain_pizza_cost : ℕ := 8  -- Cost of the plain pizza in dollars
def anchovies_cost : ℕ := 2  -- Extra cost of adding anchovies in dollars
def mushrooms_cost : ℕ := 1  -- Extra cost of adding mushrooms in dollars
def dave_slices : ℕ := 7  -- Number of slices Dave ate
def doug_slices : ℕ := 1  -- Number of slices Doug ate
def doug_payment : ℕ := 1  -- Amount Doug paid in dollars
def dave_payment : ℕ := total_pizza_cost - doug_payment  -- Amount Dave paid in dollars

-- Prove that Dave paid 9 dollars more than Doug
theorem dave_paid_more : dave_payment - doug_payment = 9 := by
  -- Proof to be filled in
  sorry

end dave_paid_more_l151_151353


namespace find_integers_l151_151923

theorem find_integers (a b m n : ℕ) (r : ℕ) (h_coprime : Nat.coprime m n)
  (h_eq : (a^2 + b^2)^m = (a * b)^n) (h_pos_int : a > 0 ∧ b > 0 ∧ m > 0 ∧ n > 0) :
  ∃ r : ℕ, a = 2^r ∧ b = 2^r ∧ m = 2 * r ∧ n = 2 * r + 1 := 
sorry

end find_integers_l151_151923


namespace sec_150_eq_l151_151395

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151395


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151556

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151556


namespace sin_c_eq_tan_b_find_side_length_c_l151_151041

-- (1) Prove that sinC = tanB
theorem sin_c_eq_tan_b {a b c : ℝ} {C : ℝ} (h1 : a / b = 1 + Real.cos C) : 
  Real.sin C = Real.tan B := by
  sorry

-- (2) If given conditions, find the value of c
theorem find_side_length_c {a b c : ℝ} {B C : ℝ} 
  (h1 : Real.cos B = 2 * Real.sqrt 7 / 7)
  (h2 : 0 < C ∧ C < Real.pi / 2)
  (h3 : 1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) 
  : c = Real.sqrt 7 := by
  sorry

end sin_c_eq_tan_b_find_side_length_c_l151_151041


namespace movie_replay_count_l151_151306

def movie_length_hours : ℝ := 1.5
def advertisement_length_minutes : ℝ := 20
def theater_operating_hours : ℝ := 11

theorem movie_replay_count :
  let movie_length_minutes := movie_length_hours * 60
  let total_showing_time_minutes := movie_length_minutes + advertisement_length_minutes
  let operating_time_minutes := theater_operating_hours * 60
  (operating_time_minutes / total_showing_time_minutes) = 6 :=
by
  sorry

end movie_replay_count_l151_151306


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151456

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151456


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151434

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151434


namespace sum_of_integer_roots_l151_151687

theorem sum_of_integer_roots (a : ℤ) (h_neq_one : a ≠ 1) (h_int_sol : ∃ x : ℤ, a * x - 3 = a^2 + 2 * a + x) : 
  ∃ x : ℤ,  sum (roots : set ℤ, roots a x) = 16 :=
begin
  sorry
end

end sum_of_integer_roots_l151_151687


namespace range_of_a_second_quadrant_l151_151032

def complex_quadrant (z : ℂ) : ℕ :=
  if z.re < 0 ∧ z.im > 0 then 2 else 0

theorem range_of_a_second_quadrant (a : ℝ) :
  complex_quadrant ((1 - complex.i) * (a - complex.i)) = 2 → a < -1 :=
by
  sorry

end range_of_a_second_quadrant_l151_151032


namespace sec_150_eq_l151_151468

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151468


namespace total_money_spent_l151_151282

noncomputable def total_expenditure (A : ℝ) : ℝ :=
  let person1_8_expenditure := 8 * 12
  let person9_expenditure := A + 8
  person1_8_expenditure + person9_expenditure

theorem total_money_spent :
  (∃ A : ℝ, total_expenditure A = 9 * A ∧ A = 13) →
  total_expenditure 13 = 117 :=
by
  intro h
  sorry

end total_money_spent_l151_151282


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151577

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151577


namespace sec_150_eq_l151_151469

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151469


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151439

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151439


namespace sec_150_eq_l151_151394

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151394


namespace find_a_n_l151_151789

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ a 2 = 3 ∧
(∀ m ≥ 1, a (2*m + 1) = a (2*m) + a (2*m-1)) ∧
(∀ m ≥ 2, a (2*m) = a (2*m-1) + 2 * a (2*m-2))

theorem find_a_n (a : ℕ → ℝ) (h : sequence a) (n : ℕ):
  (n % 2 = 1 → a n = (4 + Real.sqrt 2) / 4 * (2 + Real.sqrt 2)^(n-1) + (4 * Real.sqrt 2) / 4 * (2 - Real.sqrt 2)^(n-1)) ∧
  (n % 2 = 0 → a n = (2 * Real.sqrt 2 + 1) / 4 * (2 + Real.sqrt 2)^n - (2 * Real.sqrt 2 - 1) / 4 * (2 - Real.sqrt 2)^n) :=
sorry

end find_a_n_l151_151789


namespace seven_digit_palindrome_count_l151_151680

def digits : Multiset ℕ := [2, 2, 3, 3, 5, 5, 5]

def is_palindrome (n : List ℕ) : Prop :=
  n = List.reverse n

def seven_digit_palindrome (n : List ℕ) : Prop :=
  n.length = 7 ∧ is_palindrome n

noncomputable def count_palindromes (digits : Multiset ℕ) : ℕ :=
  (digits.permutations.filter (λ l, seven_digit_palindrome l)).length

theorem seven_digit_palindrome_count : count_palindromes digits = 6 :=
sorry

end seven_digit_palindrome_count_l151_151680


namespace max_f_value_l151_151108

noncomputable def f (A B x y : ℝ) : ℝ :=
  min x (min (A / y) (y + B / x))

theorem max_f_value (A B : ℝ) (hA : 0 < A) (hB : 0 < B) :
  ∃ M, (∀ x y : ℝ, 0 < x → 0 < y → f A B x y ≤ M) ∧ M = sqrt (A + B) :=
by
  sorry

end max_f_value_l151_151108


namespace op_7_8_eq_19_over_3_l151_151998

def op (a b : ℝ) : ℝ := (5 * a - 2 * b) / 3

theorem op_7_8_eq_19_over_3 : op 7 8 = 19 / 3 := by
  sorry

end op_7_8_eq_19_over_3_l151_151998


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151515

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151515


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151534

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151534


namespace definite_integral_value_l151_151799

-- Definitions based on the conditions
def integral_sqrt_1_minus_x_sq : ℝ := ∫ x in (0:ℝ)..(1:ℝ), real.sqrt (1 - x^2)
def integral_x : ℝ := ∫ x in (0:ℝ)..(1:ℝ), x

-- The main theorem to be proved
theorem definite_integral_value :
  integral_sqrt_1_minus_x_sq = π / 4 ∧
  integral_x = 1 / 2 →
  (∫ x in (0:ℝ)..(-1:ℝ), real.sqrt (1 - x^2) + x) = π / 4 + 1 / 2 := by
sorry

end definite_integral_value_l151_151799


namespace correct_union_l151_151982

universe u

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2}
def C_I (A : Set ℕ) : Set ℕ := {x ∈ I | x ∉ A}

-- Theorem statement
theorem correct_union : B ∪ C_I A = {2, 4, 5} :=
by
  sorry

end correct_union_l151_151982


namespace circle_M_equation_l151_151963

noncomputable theory

-- Given points A and B
def A := (Real.sqrt 2, -Real.sqrt 2)
def B := (10 : ℝ, 4 : ℝ)

-- Line equation condition for the center of the circle
def center_lies_on_line_y_eq_x (center : ℝ × ℝ) : Prop := center.snd = center.fst

-- Standard form equation of the circle
def circle_equation (center : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) : Prop := 
  (p.fst - center.fst)^2 + (p.snd - center.snd)^2 = r^2

-- Given conditions for the center and radius
def center := (4 : ℝ, 4 : ℝ)
def radius := 6

-- Line m passes through (0, -4)
def point_on_line_m := ( (0 : ℝ), (-4 : ℝ) )

-- Equations for line m
def line_m_eq1 (p : ℝ × ℝ) : Prop := p.fst = 0
def line_m_eq2 (k : ℝ) (p : ℝ × ℝ) : Prop := p.snd = k * p.fst - 4

-- Equation of the circle M
theorem circle_M_equation : 
  (circle_equation center radius A) ∧ (circle_equation center radius B) ∧ center_lies_on_line_y_eq_x center 
  → ∃ k : ℝ, (∀ p : ℝ × ℝ, line_m_eq1 p ∨ line_m_eq2 k p) :=
by intros; sorry

end circle_M_equation_l151_151963


namespace school_fee_correct_l151_151740

-- Definitions
def mother_fifty_bills : ℕ := 1
def mother_twenty_bills : ℕ := 2
def mother_ten_bills : ℕ := 3

def father_fifty_bills : ℕ := 4
def father_twenty_bills : ℕ := 1
def father_ten_bills : ℕ := 1

def total_fifty_bills : ℕ := mother_fifty_bills + father_fifty_bills
def total_twenty_bills : ℕ := mother_twenty_bills + father_twenty_bills
def total_ten_bills : ℕ := mother_ten_bills + father_ten_bills

def value_fifty_bills : ℕ := 50 * total_fifty_bills
def value_twenty_bills : ℕ := 20 * total_twenty_bills
def value_ten_bills : ℕ := 10 * total_ten_bills

-- Theorem
theorem school_fee_correct :
  value_fifty_bills + value_twenty_bills + value_ten_bills = 350 :=
by
  sorry

end school_fee_correct_l151_151740


namespace consecutive_non_carry_pairs_l151_151620

def no_carry (n : ℕ) : Prop :=
  let units_digit := n % 10
  let hundreds_digit := (n / 100) % 10
  let next_units_digit := (n + 1) % 10
  let next_hundreds_digit := ((n + 1) / 100) % 10
  next_units_digit ≠ 0 ∧ (units_digit ≠ 9) ∧ (hundreds_digit ≠ 9 ∨ next_hundreds_digit ≠ 0)

def count_no_carry_pairs (l u : ℕ) : ℕ :=
  (List.range (u - l)).map (λ n => l + n).filter (λ n => no_carry n).length

theorem consecutive_non_carry_pairs :
  count_no_carry_pairs 1500 2500 = 991 := by
  sorry

end consecutive_non_carry_pairs_l151_151620


namespace volume_of_tetrahedron_450_sqrt_2_l151_151920

/-- Define the conditions -/
variables {A B C D : Type}
variables (PQR : Triangle A B C)
variables (QRS : Triangle B C D)
variables (BC : LineSegment B C)

/-- Areas of triangular faces ABC and BCD and length of BC -/
variable (area_ABC : ℝ)
variable (area_BCD : ℝ)
variable (length_BC : ℝ)

/-- The angle between faces ABC and BCD -/
variable (angle_ABC_BCD : ℝ)

/-- Assume the given conditions -/
variables (area_ABC_cond : area_ABC = 150)
variables (area_BCD_cond : area_BCD = 90)
variables (length_BC_cond : length_BC = 10)
variables (angle_ABC_BCD_cond : angle_ABC_BCD = π / 4)

/-- Define the volume of the tetrahedron -/
def volume_tetrahedron (area_ABC area_BCD length_BC angle_ABC_BCD : ℝ) : ℝ := 
  (1 / 3) * area_ABC * (sqrt 2 * 9)

/-- The goal: the volume of tetrahedron ABCD is 450√2 under the given conditions -/
theorem volume_of_tetrahedron_450_sqrt_2 :
  volume_tetrahedron area_ABC area_BCD length_BC angle_ABC_BCD = 450 * sqrt 2 := by
  sorry

end volume_of_tetrahedron_450_sqrt_2_l151_151920


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151390

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151390


namespace range_of_a_l151_151653

-- Define the hyperbola and necessary parameters
def hyperbola (x y a : ℝ) : Prop :=
  x^2 - (y^2)/(a^2) = 1

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = sqrt 7 * x - 4

-- Define the condition for the circle intersecting the line and the hyperbola conditions
theorem range_of_a (a : ℝ) (h : a > 0) (circ_inter_line : ∃ (x y : ℝ), line x y ∧ x^2 + y^2 = 1 + a^2) : 
  a > 1 := sorry

end range_of_a_l151_151653


namespace ninth_day_skate_time_l151_151937

-- Define the conditions
def first_4_days_skate_time : ℕ := 4 * 70
def second_4_days_skate_time : ℕ := 4 * 100
def total_days : ℕ := 9
def average_minutes_per_day : ℕ := 100

-- Define the theorem stating that Gage must skate 220 minutes on the ninth day to meet the average
theorem ninth_day_skate_time : 
  let total_minutes_needed := total_days * average_minutes_per_day
  let current_skate_time := first_4_days_skate_time + second_4_days_skate_time
  total_minutes_needed - current_skate_time = 220 := 
by
  -- Placeholder for the proof
  sorry

end ninth_day_skate_time_l151_151937


namespace ratio_EF_FD_l151_151059

-- Define the geometrical entities and conditions
variables (A B C D E F : Type) [Geometry A] [Geometry B] [Geometry C] [Geometry D] [Geometry E] [Geometry F]

-- Given conditions
def angles_45 (A B F: Type) [Geometry A] [Geometry B] [Geometry F] : Prop :=
  angle A B F = 45 ∧ angle F B C = 45

def square_ACDE (A C D E : Type) [Geometry A] [Geometry C] [Geometry D] [Geometry E] : Prop :=
  is_square A C D E

def AB_BC (A B C : Type) [Geometry A] [Geometry B] [Geometry C] : Prop :=
  segment_length A B = 2 / 3 * segment_length B C

-- The goal
theorem ratio_EF_FD {A B C D E F : Type} [Geometry A] [Geometry B] [Geometry C] [Geometry D] [Geometry E] [Geometry F] :
  angles_45 A B F ∧ square_ACDE A C D E ∧ AB_BC A B C →
  segment_ratio E F F D = 3 / 2 :=
sorry

end ratio_EF_FD_l151_151059


namespace gnome_stone_distribution_l151_151019

theorem gnome_stone_distribution : 
  ∀ (gloin oin train : ℕ), gloin + oin + train = 70 ∧ gloin ≥ 10 ∧ oin ≥ 10 ∧ train ≥ 10 ↔ 
  ∃ (ways_to_divide : ℕ), ways_to_divide = 946 :=
by
  intro gloin oin train
  have fact_sum : gloin + oin + train = 70 := sorry
  have min_gloin : gloin ≥ 10 := sorry
  have min_oin : oin ≥ 10 := sorry
  have min_train : train ≥ 10 := sorry
  use 946
  split
  { intro h
    exact sorry }
  { intro h
    split
    { exact sorry }
    { split
      { exact sorry }
      { exact sorry } } }

end gnome_stone_distribution_l151_151019


namespace vector_coordinates_standard_basis_l151_151983

theorem vector_coordinates_standard_basis :
  ∀ (m : ℝ × ℝ × ℝ) (a b c: ℝ × ℝ × ℝ),
  m = (8, 6, 4) →
  a = (1, 1, 0) →
  b = (0, 1, 1) →
  c = (1, 0, 1) →
  let i := (1, 0, 0),
      j := (0, 1, 0),
      k := (0, 0, 1),
      in
  (8 * a.1 + 6 * b.1 + 4 * c.1,
   8 * a.2 + 6 * b.2 + 4 * c.2,
   8 * a.3 + 6 * b.3 + 4 * c.3) = (12, 14, 10) := 
begin
  intros m a b c hm ha hb hc,
  rw [ha, hb, hc],
  simp,
end

end vector_coordinates_standard_basis_l151_151983


namespace kaleb_initial_books_l151_151075

def initial_books (sold_books bought_books final_books : ℕ) : ℕ := 
  sold_books - bought_books + final_books

theorem kaleb_initial_books :
  initial_books 17 (-7) 24 = 34 := 
by 
  -- use the definition of initial_books
  sorry

end kaleb_initial_books_l151_151075


namespace additional_grazing_area_l151_151862

/-- A rope of which a calf is tied is increased from 12 m to 23 m.
    How much additional grassy ground shall it graze? -/
theorem additional_grazing_area :
  let r1 := 12
  let r2 := 23
  let area1 := Real.pi * (r1:ℝ)^2
  let area2 := Real.pi * (r2:ℝ)^2
  area2 - area1 = 385 * Real.pi := by
  let r1 : ℝ := 12
  let r2 : ℝ := 23
  let area1 := Real.pi * r1^2
  let area2 := Real.pi * r2^2
  calc area2 - area1 = Real.pi * r2^2 - Real.pi * r1^2 : by sorry
                   ... = 385 * Real.pi              : by sorry

end additional_grazing_area_l151_151862


namespace andy_questions_wrong_l151_151329

variables (a b c d : ℕ)

-- Given conditions
def condition1 : Prop := a + b = c + d
def condition2 : Prop := a + d = b + c + 6
def condition3 : Prop := c = 7

-- The theorem to prove
theorem andy_questions_wrong (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 c) : a = 10 :=
by
  sorry

end andy_questions_wrong_l151_151329


namespace solve_sqrt_equation_l151_151597

theorem solve_sqrt_equation (x : ℝ) (hx : real.sqrt (62 - 3 * x) + real.sqrt (38 + 3 * x) = 5) : 
  x = 43 / 3 :=
sorry

end solve_sqrt_equation_l151_151597


namespace coefficient_of_x4_in_expansion_l151_151233

theorem coefficient_of_x4_in_expansion (x : ℤ) :
  let a := 3
  let b := 2
  let n := 8
  let k := 4
  (finset.sum (finset.range (n + 1)) (λ r, binomial n r * a^r * b^(n-r) * x^r) = 
  ∑ r in finset.range (n + 1), binomial n r * a^r * b^(n - r) * x^r)

  ∑ r in finset.range (n + 1), 
    if r = k then 
      binomial n r * a^r * b^(n-r)
    else 
      0 = 90720
:= 
by
  sorry

end coefficient_of_x4_in_expansion_l151_151233


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151565

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151565


namespace part_a_part_b_l151_151132

variables (a b c p r S : ℝ)

-- Conditions
def semi_perimeter := (a + b + c) / 2
def area_from_semi_perimeter := sqrt (p * (p - a) * (p - b) * (p - c))
def area_from_inradius := p * r

-- Assumptions based on conditions
axiom semi_perimeter_def : p = semi_perimeter a b c
axiom area_from_semi_perimeter_def : S = area_from_semi_perimeter p a b c
axiom area_from_inradius_def : S = area_from_inradius p r

-- Theorem to prove part (a)
theorem part_a : 3 * sqrt 3 * r^2 ≤ S ∧ S ≤ p^2 / (3 * sqrt 3) :=
by sorry

-- Theorem to prove part (b)
theorem part_b : S ≤ (a^2 + b^2 + c^2) / (4 * sqrt 3) :=
by sorry

end part_a_part_b_l151_151132


namespace sec_150_eq_l151_151474

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151474


namespace tom_total_payment_l151_151211

/-- Problem statement:
Tom purchased 8 kg of apples at the rate of 70 per kg, 
9 kg of mangoes at the rate of 90 per kg, 
and 5 kg of grapes at a rate of 150 per kg. 
The shopkeeper offered him a discount of 10% on the total amount.
Prove that the total amount Tom ends up paying to the shopkeeper is 1908.
-/
theorem tom_total_payment :
  let cost_apples := 8 * 70,
      cost_mangoes := 9 * 90,
      cost_grapes := 5 * 150,
      total_cost_before_discount := cost_apples + cost_mangoes + cost_grapes,
      discount := total_cost_before_discount * 10 / 100,
      total_amount_after_discount := total_cost_before_discount - discount
  in total_amount_after_discount = 1908 :=
by
  sorry

end tom_total_payment_l151_151211


namespace inequality_solution_set_maximum_value_expression_l151_151005

def gaussian_function (x : ℝ) : ℤ := int.floor x

theorem inequality_solution_set :
  {x : ℝ | 1 ≤ x ∧ x < 4 ∧ (gaussian_function x) / (gaussian_function x - 4) < 0} =
  set.Ico 1 4 :=
sorry

theorem maximum_value_expression (x : ℝ) (h : x > 0) :
  let gauss_x := gaussian_function x in
  ∃ (m : ℝ), (gauss_x / (gauss_x ^ 2 + 4)) = m ∧ m = 1 / 4 :=
sorry

end inequality_solution_set_maximum_value_expression_l151_151005


namespace tiled_floor_area_correct_garden_area_correct_seating_area_correct_l151_151311

noncomputable def length_room : ℝ := 20
noncomputable def width_room : ℝ := 12
noncomputable def width_veranda : ℝ := 2
noncomputable def length_pool : ℝ := 15
noncomputable def width_pool : ℝ := 6

noncomputable def area (length width : ℝ) : ℝ := length * width

noncomputable def area_room : ℝ := area length_room width_room
noncomputable def area_pool : ℝ := area length_pool width_pool
noncomputable def area_tiled_floor : ℝ := area_room - area_pool

noncomputable def total_length : ℝ := length_room + 2 * width_veranda
noncomputable def total_width : ℝ := width_room + 2 * width_veranda
noncomputable def area_total : ℝ := area total_length total_width
noncomputable def area_veranda : ℝ := area_total - area_room
noncomputable def area_garden : ℝ := area_veranda / 2
noncomputable def area_seating : ℝ := area_veranda / 2

theorem tiled_floor_area_correct : area_tiled_floor = 150 := by
  sorry

theorem garden_area_correct : area_garden = 72 := by
  sorry

theorem seating_area_correct : area_seating = 72 := by
  sorry

end tiled_floor_area_correct_garden_area_correct_seating_area_correct_l151_151311


namespace incorrect_props_l151_151324

-- Define propositions and their correctness.
def prop1 (L1 L2 L3 : Type) [LinearOrder L1] [LinearOrder L2] : Prop :=
  ∀ l1 l2 l3 : L1, l1 ∥ l2 → l2 ∥ l3 → l1 ∥ l3

def prop2 (L1 L2 L3 : Type) [LinearOrder L1] [LinearOrder L2] : Prop :=
  ∀ l1 l2 l3 : L1, l1 ⟂ l2 → l1 ⟂ l3 → l2 ∥ l3

def prop3 (P1 P2 P3 : Type) [PartialOrder P1] [PartialOrder P2] : Prop :=
  ∀ p1 p2 p3 : P1, p1 ∥ p2 → p2 ∥ p3 → p1 ∥ p3

def prop4 (P1 P2 P3 : Type) [PartialOrder P1] [PartialOrder P2] : Prop :=
  ∀ p1 p2 p3 : P1, p1 ⟂ p2 → p1 ⟂ p3 → p2 ∥ p3

-- Prove that propositions (2) and (4) are incorrect.
theorem incorrect_props (L1 L2 L3 : Type) [LinearOrder L1] [LinearOrder L2] [LinearOrder L3] :
  ¬ prop2 L1 L2 L3 ∧ ¬ prop4 P1 P2 P3 :=
sorry

end incorrect_props_l151_151324


namespace part1_part2_l151_151649

noncomputable def f (x : Real) : Real :=
  (Real.cos (x + Real.pi / 12)) ^ 2

noncomputable def g (x : Real) : Real :=
  1 + 1 / 2 * Real.sin (2 * x)

noncomputable def h (x : Real) : Real :=
  f x + g x

def is_axis_of_symmetry (x0 : Real) : Prop :=
  ∃ k : Int, 2 * x0 + Real.pi / 6 = k * Real.pi

theorem part1 (x0 : Real) (hx0 : is_axis_of_symmetry x0) : 
  g x0 = 3 / 4 ∨ g x0 = 5 / 4 :=
  sorry

theorem part2 (k : Int) :
  ∀ x : Real, k * Real.pi - 5 * Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 12 → 
  (h x - h (x - Real.epsilon)) / Real.epsilon > 0 :=
  sorry

end part1_part2_l151_151649


namespace range_of_k_for_equation_l151_151164

theorem range_of_k_for_equation (k : ℝ) :
  (∃ x ∈ Icc (0 : ℝ) 1, k * 4^x - k * 2^(x + 1) + 6 * (k - 5) = 0) →
  k ∈ Icc (5 : ℝ) 6 :=
by
  sorry

end range_of_k_for_equation_l151_151164


namespace sec_150_l151_151486

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151486


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151564

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151564


namespace consecutive_numbers_bases_l151_151058

theorem consecutive_numbers_bases (A B : ℕ) (h1 : B = A + 1 ∨ B = A - 1) (h2 : 132.base_repr A + 43.base_repr B = 69.base_repr (A + B)) :
  A + B = 13 :=
  sorry

end consecutive_numbers_bases_l151_151058


namespace number_of_nonnegative_real_x_sqrt_integer_l151_151619

-- Define the conditions
def is_integer (n : ℝ) : Prop := ∃ k : ℤ, n = k

-- Problem statement
theorem number_of_nonnegative_real_x_sqrt_integer (x : ℝ) (hx : x ≥ 0) :
  (is_integer (sqrt (225 - (x ^ (1/3))))) → (∃ n : ℤ, 0 ≤ n ∧ n ≤ 15) :=
sorry

end number_of_nonnegative_real_x_sqrt_integer_l151_151619


namespace number_of_multiples_of_1001_l151_151684

theorem number_of_multiples_of_1001 :
  let S := { (i, j) | 0 ≤ i ∧ i < j ∧ j ≤ 149 ∧ ∃ k, 10^j - 10^i = 1001 * k } in
  S.finite →
  S.card = 1752 :=
by
  sorry

end number_of_multiples_of_1001_l151_151684


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151374

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151374


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151554

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151554


namespace initial_treasure_amount_l151_151759

theorem initial_treasure_amount 
  (T : ℚ)
  (h₁ : T * (1 - 1/13) * (1 - 1/17) = 150) : 
  T = 172 + 21/32 :=
sorry

end initial_treasure_amount_l151_151759


namespace radius_increased_by_100_percent_l151_151154

noncomputable def radius_increase (r : ℝ) (A : ℝ) (A' : ℝ) (r' : ℝ) : Prop :=
    (A = π * r^2) ∧ (A' = 4 * A) ∧ (A' = π * r'^2) ∧ (r' = 2 * r)

theorem radius_increased_by_100_percent (r : ℝ) (A : ℝ) (A' : ℝ) (r' : ℝ) :
   (radius_increase r A A' r') → (r' = 2 * r) :=
by
  intro h
  have h1 : 4 * (π * r^2) = π * r'^2 := sorry
  have h2 : r'^2 = 4 * r^2 := sorry
  have h3 : r' = 2 * r := sorry
  exact h3

end radius_increased_by_100_percent_l151_151154


namespace range_of_a_monotonically_decreasing_l151_151781

/-- Let \( g(x) = ax^3 + 2(1-a)x^2 - 3ax \). Given that \( g(x) \) is monotonically decreasing in the interval 
\(\left( -\infty, \frac{a}{3} \right)\), find the range of values for \( a \). -/
theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (a ≤ -1 ∨ a = 0) ↔ ∀ x ∈ Iio (a / 3), 3*a*x^2 + 4*(1-a)*x - 3*a ≤ 0 :=
sorry

end range_of_a_monotonically_decreasing_l151_151781


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151520

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151520


namespace probability_of_region_F_l151_151320

theorem probability_of_region_F
  (pD pE pG pF : ℚ)
  (hD : pD = 3/8)
  (hE : pE = 1/4)
  (hG : pG = 1/8)
  (hSum : pD + pE + pF + pG = 1) : pF = 1/4 :=
by
  -- we can perform the steps as mentioned in the solution without actually executing them
  sorry

end probability_of_region_F_l151_151320


namespace number_of_correct_vector_expressions_l151_151704

theorem number_of_correct_vector_expressions :
  let complex_expr1 := ∀ (z1 z2 : ℂ), abs (z1 + z2) ≤ abs z1 + abs z2
  let complex_expr2 := ∀ (z1 z2 : ℂ), abs (z1 * z2) = abs z1 * abs z2
  let complex_expr3 := ∀ (z1 z2 z3 : ℂ), (z1 * z2) * z3 = z1 * (z2 * z3)
  let vector_expr1 := ∀ (a b : ℝ × ℝ × ℝ), abs (a.1 + b.1, a.2 + b.2, a.3 + b.3) ≤ abs a + abs b
  let vector_expr2 := ∀ (a b : ℝ × ℝ × ℝ), abs (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = abs a * abs b
  let vector_expr3 := ∀ (a b c : ℝ × ℝ × ℝ), (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) * c = a * (b.1 * c + b.2 * c + b.3 * c)
  in
  (complex_expr1 ∧ complex_expr2 ∧ complex_expr3) →
  (vector_expr1 ∧ ¬vector_expr2 ∧ ¬vector_expr3) →
  (true = 1)
:= 
  by 
  sorry

end number_of_correct_vector_expressions_l151_151704


namespace smallest_solution_l151_151609

def equation (x : ℝ) : ℝ := (3*x)/(x-3) + (3*x^2 - 27)/x

theorem smallest_solution : ∃ x : ℝ, equation x = 14 ∧ x = (7 - Real.sqrt 76) / 3 := 
by {
  -- proof steps go here
  sorry
}

end smallest_solution_l151_151609


namespace circle_area_l151_151129

/-
Points A = (4, 15) and B = (10, 13) lie on circle ω in the plane. 
The tangent lines to ω at A and B intersect at a point on the y-axis.
-/
open Real

def A : ℝ × ℝ := (4, 15)
def B : ℝ × ℝ := (10, 13)
def intersectsYaxis (point : ℝ × ℝ) := point.1 = 0

/-
Prove that the area of circle ω is 100π / 49.
-/
theorem circle_area {ω : Type} [metric_space ω] [proper_space ω] 
(hA : A ∈ ω) (hB : B ∈ ω) (h_tangent_at_y_axis: ∃ P : ℝ × ℝ, intersectsYaxis P ∧ is_tangent_line_at ω A P ∧ is_tangent_line_at ω B P) :
  ∃ (R : ℝ), R ^ 2 * π = 100 * π / 49 :=
sorry

end circle_area_l151_151129


namespace number_of_integers_a_satisfying_conditions_l151_151999

def has_two_integer_solutions (a : ℤ) : Prop :=
∀ x : ℤ, 
  (6 * x - 5 ≥ a) ∧ ((x < 4 : ℚ) ∧ (x ≥ (a + 5) / 6 : ℚ))

def has_positive_solution (a : ℤ) : Prop :=
∃ y : ℚ, ((4 * y - 3 * a = 2 * (y - 3)) ∧ (y > 0))

theorem number_of_integers_a_satisfying_conditions :
  (finset.card (finset.filter (λ a, has_two_integer_solutions a ∧ has_positive_solution a) (finset.range 8))) = 5 := sorry

end number_of_integers_a_satisfying_conditions_l151_151999


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151516

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151516


namespace arithmetic_sequence_sum_l151_151958

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℤ) (S_n : ℕ → ℤ),
  (∀ n : ℕ, S_n n = (n * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) / 2) →
  S_n 17 = 170 →
  a_n 7 + a_n 8 + a_n 12 = 30 := 
by
  sorry

end arithmetic_sequence_sum_l151_151958


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151508

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151508


namespace perimeter_of_resulting_figure_l151_151080

-- Define the perimeters of the squares
def perimeter_small_square : ℕ := 40
def perimeter_large_square : ℕ := 100

-- Define the side lengths of the squares
def side_length_small_square := perimeter_small_square / 4
def side_length_large_square := perimeter_large_square / 4

-- Define the total perimeter of the uncombined squares
def total_perimeter_uncombined := perimeter_small_square + perimeter_large_square

-- Define the shared side length
def shared_side_length := side_length_small_square

-- Define the perimeter after considering the shared side
def resulting_perimeter := total_perimeter_uncombined - 2 * shared_side_length

-- Prove that the resulting perimeter is 120 cm
theorem perimeter_of_resulting_figure : resulting_perimeter = 120 := by
  sorry

end perimeter_of_resulting_figure_l151_151080


namespace possible_single_lamp_positions_l151_151207

-- Define the grid size
def grid_size : Nat := 5

-- Define what it means for a lamp to change its state
def toggle (grid : Array (Array Bool)) (i j : Nat) : Array (Array Bool) := 
  let toggle_lamp (lamp : Bool) := not lamp
  let mutate (grid : Array (Array Bool)) (i j : Nat) := 
    (grid.set! i (grid[i].set! j (toggle_lamp (grid[i][j]))))
  mutate (mutate (mutate (mutate (mutate grid i j) (i + 1) j) (i - 1) j) i (j + 1)) i (j - 1)

-- Define the initial state (all lamps off)
def initial_grid : Array (Array Bool) := 
  Array.mkArray grid_size (Array.mkArray grid_size false)

-- Function to activate a set of lamps
def activate_lamps (grid : Array (Array Bool)) (positions : List (Nat × Nat)) : Array (Array Bool) :=
  positions.foldl (λ g p, toggle g p.fst p.snd) grid

-- Specify the positions as the result
theorem possible_single_lamp_positions (positions : List (Nat × Nat)) :
  let final_grid := activate_lamps initial_grid positions
  ((∃ i j, final_grid[i][j] = true) ∧ 
   (∀ i1 j1 i2 j2, (i1 ≠ i2 ∨ j1 ≠ j2) → final_grid[i1][j1] = false ∧ final_grid[i2][j2] = false)) → 
   ∃! pos ∈ [(3,3), (2,2), (2,4), (4,2), (4,4)], 
   final_grid[pos.fst][pos.snd] = true := sorry

end possible_single_lamp_positions_l151_151207


namespace hyperbola_equation_l151_151702

theorem hyperbola_equation 
  (h k a c : ℝ)
  (center_cond : (h, k) = (3, -1))
  (vertex_cond : a = abs (2 - (-1)))
  (focus_cond : c = abs (7 - (-1)))
  (b : ℝ)
  (b_square : c^2 = a^2 + b^2) :
  h + k + a + b = 5 + Real.sqrt 55 := 
by
  -- Prove that given the conditions, the value of h + k + a + b is 5 + √55.
  sorry

end hyperbola_equation_l151_151702


namespace lavinias_son_older_than_daughter_l151_151718

def katies_daughter_age := 12
def lavinias_daughter_age := katies_daughter_age - 10
def lavinias_son_age := 2 * katies_daughter_age

theorem lavinias_son_older_than_daughter :
  lavinias_son_age - lavinias_daughter_age = 22 :=
by
  sorry

end lavinias_son_older_than_daughter_l151_151718


namespace composite_square_perimeter_l151_151087

theorem composite_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let s1 := p1 / 4
  let s2 := p2 / 4
  (p1 + p2 - 2 * s1) = 120 := 
by
  -- proof goes here
  sorry

end composite_square_perimeter_l151_151087


namespace remainder_2503_div_28_l151_151256

theorem remainder_2503_div_28 : 2503 % 28 = 11 := 
by
  -- The proof goes here
  sorry

end remainder_2503_div_28_l151_151256


namespace maximize_minimize_BP_CQ_cos_alpha_l151_151952

-- Definitions for the conditions
variables (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (α r : ℝ)
variables (a b c : ℝ) (triangle_ABC : ∀ {x y z : ℝ}, x + y > z → x + z > y → y + z > x)
variables (circle_center_A_diameter_2r : dist A P = r ∧ dist A Q = r)

-- Problem statement in Lean 4
theorem maximize_minimize_BP_CQ_cos_alpha :
    (BP : ℝ) (CQ : ℝ) (cos_α : ℝ) →
    ∃ (P Q : Type), 
    (BP * CQ * cos_α = max (BP * CQ * 1) (BP * CQ * -1)) ∨ 
    (BP * CQ * cos_α = min (BP * CQ * 1) (BP * CQ * -1)) :=
sorry

end maximize_minimize_BP_CQ_cos_alpha_l151_151952


namespace smallest_possible_x_l151_151832

/-- Proof problem: When x is divided by 6, 7, and 8, remainders of 5, 6, and 7 (respectively) are obtained. 
We need to show that the smallest possible positive integer value of x is 167. -/
theorem smallest_possible_x (x : ℕ) (h1 : x % 6 = 5) (h2 : x % 7 = 6) (h3 : x % 8 = 7) : x = 167 :=
by 
  sorry

end smallest_possible_x_l151_151832


namespace remainder_calculation_l151_151698

theorem remainder_calculation :
  ∃ remainder, 690 = 36 * 19 + remainder ∧ remainder = 6 :=
by
  use 6
  split
  sorry
  refl

end remainder_calculation_l151_151698


namespace books_left_over_l151_151046

theorem books_left_over (boxes : ℕ) (books_per_box_initial : ℕ) (books_per_box_new: ℕ) (total_books : ℕ) :
  boxes = 1500 →
  books_per_box_initial = 45 →
  books_per_box_new = 47 →
  total_books = boxes * books_per_box_initial →
  (total_books % books_per_box_new) = 8 :=
by intros; sorry

end books_left_over_l151_151046


namespace ice_cream_scoops_permutations_l151_151139

theorem ice_cream_scoops_permutations :
  ∀ (scoops : Finset String), scoops.card = 5 → 
    scoops = {'vanilla', 'chocolate', 'strawberry', 'cherry', 'mango'} → 
    (scoops.to_list.permutations.length = 120) :=
by
  intros scoops h1 h2
  sorry

end ice_cream_scoops_permutations_l151_151139


namespace equilateral_triangle_of_circle_and_hyperbola_l151_151290

variables {x0 : ℝ}

noncomputable def circle_radius : ℝ := 2 * real.sqrt (x0^2 + x0⁻²)

noncomputable def circle_center : ℝ × ℝ := (x0, x0⁻¹)

def hyperbola (x y : ℝ) : Prop := x * y = 1

def circle (x y : ℝ) : Prop :=
  let (c_x, c_y) := circle_center in
  (c_x - x)^2 + (c_y - y)^2 = circle_radius^2

variables {A B C : ℝ × ℝ}

-- Condition: A, B, C are intersection points of the circle and the hyperbola
def intersection_points : Prop :=
  hyperbola A.1 A.2 ∧ circle A.1 A.2 ∧
  hyperbola B.1 B.2 ∧ circle B.1 B.2 ∧
  hyperbola C.1 C.2 ∧ circle C.1 C.2 ∧
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Point (-x0, -x0⁻¹) is also an intersection point
def special_intersection_point : ℝ × ℝ := (-x0, -x0⁻¹)

theorem equilateral_triangle_of_circle_and_hyperbola
  (hA : hyperbola A.1 A.2) (hA_circ : circle A.1 A.2)
  (hB : hyperbola B.1 B.2) (hB_circ : circle B.1 B.2)
  (hC : hyperbola C.1 C.2) (hC_circ : circle C.1 C.2)
  (h_ne_AB : A ≠ B) (h_ne_BC : B ≠ C) (h_ne_CA : C ≠ A)
  (h_special : hyperbola special_intersection_point.1 special_intersection_point.2) 
  (h_special_circ : circle special_intersection_point.1 special_intersection_point.2) :
  (triangle_is_equilateral : A ≠ B ∧ B ≠ C ∧ C ≠ A) :=
sorry

end equilateral_triangle_of_circle_and_hyperbola_l151_151290


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151413

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151413


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151499

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151499


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151450

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151450


namespace sec_150_l151_151478

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151478


namespace selling_price_before_brokerage_l151_151270

theorem selling_price_before_brokerage (cash_realized : ℝ) (brokerage_rate : ℝ) :
  cash_realized = 101.25 → brokerage_rate = 1 / 4 → 
  let P := cash_realized / (1 - brokerage_rate / 100) in
  P = 101.50 :=
by
  intro h1 h2
  let P := cash_realized / (1 - brokerage_rate / 100)
  sorry

end selling_price_before_brokerage_l151_151270


namespace f_monotonic_intervals_g_not_below_f_inequality_holds_l151_151970

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem f_monotonic_intervals :
  ∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < 1 / 2 → f x < f (x + 1)) ∧ 
    (1 / 2 < x ∧ x < 1 → f x > f (x + 1)) ∧ 
    (1 < x → f x < f (x + 1)) :=
sorry

theorem g_not_below_f :
  ∀ x : ℝ, 0 < x → f x < g x :=
sorry

theorem inequality_holds (n : ℕ) : (2 * n + 1)^2 > 4 * Real.log (Nat.factorial n) :=
sorry

end f_monotonic_intervals_g_not_below_f_inequality_holds_l151_151970


namespace donut_selection_l151_151750

theorem donut_selection :
  ∃ (ways : ℕ), ways = Nat.choose 8 3 ∧ ways = 56 :=
by
  sorry

end donut_selection_l151_151750


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151593

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151593


namespace hall_width_is_correct_l151_151045

def width_of_hall (L : ℝ) (C : ℝ) (T : ℝ) : ℝ :=
  T / C / L

theorem hall_width_is_correct :
  ∀ (L : ℝ) (C : ℝ) (T : ℝ),
  L = 20 → C = 10 → T = 9500 → width_of_hall L C T = 47.5 :=
by 
  intros L C T hL hC hT 
  unfold width_of_hall
  rw [hL, hC, hT]
  norm_num

end hall_width_is_correct_l151_151045


namespace length_FB_equals_3_l151_151057

theorem length_FB_equals_3
  (A B C D G F : EuclideanGeometry.Point)
  (s : EuclideanGeometry.Square)
  (h_ABCD : EuclideanGeometry.ShapeIsSquare ABCD 8) 
  (h_G : EuclideanGeometry.Midpoint BC G)
  (h_fold : EuclideanGeometry.PointOnLineSegment A G = F)
  (h_AF_GF : EuclideanGeometry.Distance A F = EuclideanGeometry.Distance G F): 
  EuclideanGeometry.Distance F B = 3 :=
by
  sorry

end length_FB_equals_3_l151_151057


namespace triangle_ABC_area_l151_151310

open Real

/-- Define the vertices of the triangle and the rectangle dimensions as constants. -/
def A : (ℝ × ℝ) := (0, 2)
def B : (ℝ × ℝ) := (6, 0)
def C : (ℝ × ℝ) := (3, 7)
def rect_width : ℝ := 6
def rect_height : ℝ := 7

/-- Define the formula to compute the area of a triangle given its vertices. -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2)) + (p2.1 * (p3.2 - p1.2)) + (p3.1 * (p1.2 - p2.2))) / 2

theorem triangle_ABC_area :
  triangle_area A B C = 18 :=
sorry

end triangle_ABC_area_l151_151310


namespace passes_through_quadrants_l151_151301

-- Define the linear function y = kx + b with the given conditions
def linear_function (x : ℝ) : ℝ := -x + 5

-- Define the proof problem
theorem passes_through_quadrants : 
  ∀ x : ℝ, (linear_function x > 0 → x < 5) ∧ (linear_function x = 0 → x = 5) ∧ (linear_function x < 0 → x > 5): 
by
  intro x
  sorry

end passes_through_quadrants_l151_151301


namespace gear_alignment_l151_151222

def gears_exist_rotation_alignment (n : ℕ) (gear_teeth : ℕ) (ground_pairs : ℕ) : Prop :=
  ground_pairs = 6 ∧ gear_teeth = 32 ∧
  ∃ (k : ℕ), k = n^2 - n + 1 ∧
  (31 = n^2 - n + 1) ∧
  (30 = n * (n - 1)) ∧ 
  (31 - 30 = 1)

theorem gear_alignment :
  gears_exist_rotation_alignment 6 32 6 :=
by
  simp [gears_exist_rotation_alignment]
  split
  . refl
  split
  . refl
  use 31
  split 
  . refl
  split
  . refl
  split
  . refl
  . refl
  sorry

end gear_alignment_l151_151222


namespace incorrect_formula_l151_151291

-- The conditions of the problem
def students : ℕ := 60
def class_president : ℕ := 1
def vice_president : ℕ := 1
def selected_students : ℕ := 5

-- The question: Which formula is incorrect given the conditions?
theorem incorrect_formula :
  let A := choose 2 1 * choose 59 4 in
  let B := choose 60 5 - choose 58 5 in
  let C := choose 2 1 * choose 59 4 - choose 2 2 * choose 58 3 in
  let D := choose 2 1 * choose 58 4 + choose 2 2 * choose 58 3 in
  A ≠ (B ∨ C ∨ D) :=
sorry -- Prove that A is incorrect under the given conditions

end incorrect_formula_l151_151291


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151585

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151585


namespace parabola_translation_l151_151214

theorem parabola_translation :
  ∀ (x y : ℝ), (y = x^2 - 2x + 4) → (∃ x' y' : ℝ, y' = (x' - 2)^2 + 6 ∧ x' = x + 1 ∧ y' = y + 3) → y = x^2 - 4x + 10 :=
by
  intro x y
  intro h_eq
  intro ⟨x', y', h_y', h_x', h_y''⟩
  rw [h_eq, h_y', h_x', h_y'']
  sorry

end parabola_translation_l151_151214


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151587

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151587


namespace limit_seq_a_l151_151673

def seq_a : ℕ+ → ℝ
| ⟨n, n_pos⟩ := 
  if n ≤ 4 then -n else real.sqrt (n^2 - 4 * n) - n

theorem limit_seq_a : filter.tendsto seq_a filter.at_top (nhds (-2)) :=
sorry

end limit_seq_a_l151_151673


namespace equidistant_point_is_circumcenter_l151_151192

noncomputable def equidistant_point (Δ : Triangle) :=
  ∃ (P : Point), is_equidistant_to_vertices P Δ

theorem equidistant_point_is_circumcenter (Δ : Triangle) :
  ∃ (P : Point), is_circumcenter P Δ :=
by
  -- Proof goes here (not required)
  sorry

end equidistant_point_is_circumcenter_l151_151192


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151389

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151389


namespace least_integer_value_l151_151823

theorem least_integer_value (x : ℤ) (h : |3 * x + 10| ≤ 25) : x = -11 :=
begin
  sorry
end

end least_integer_value_l151_151823


namespace sec_150_l151_151491

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151491


namespace remainder_of_sum_l151_151931

theorem remainder_of_sum (h1 : 9375 % 5 = 0) (h2 : 9376 % 5 = 1) (h3 : 9377 % 5 = 2) (h4 : 9378 % 5 = 3) :
  (9375 + 9376 + 9377 + 9378) % 5 = 1 :=
by
  sorry

end remainder_of_sum_l151_151931


namespace trajectory_is_parabola_l151_151775

theorem trajectory_is_parabola (P : ℝ × ℝ) (h : abs (real.dist P (2, 0) - abs (P.1 + 4) / real.sqrt (1^2)) = 2) : 
  ∃ a b c : ℝ, P.2 = a * (P.1^2) + b * P.1 + c :=
sorry

end trajectory_is_parabola_l151_151775


namespace num_such_functions_l151_151981

def A : finset ℕ := {1, 2, 3}

def satisfiesCondition (f : ℕ → ℕ) : Prop := ∀ x ∈ A, f(f(x)) >= x

theorem num_such_functions : 
  {f ∈ (A → A) | satisfiesCondition f}.to_finset.card = 13 :=
sorry

end num_such_functions_l151_151981


namespace Shawn_scored_6_points_l151_151742

theorem Shawn_scored_6_points
  (points_per_basket : ℤ)
  (matthew_points : ℤ)
  (total_baskets : ℤ)
  (h1 : points_per_basket = 3)
  (h2 : matthew_points = 9)
  (h3 : total_baskets = 5)
  : (∃ shawn_points : ℤ, shawn_points = 6) :=
by
  sorry

end Shawn_scored_6_points_l151_151742


namespace log_value_l151_151028

theorem log_value (y : ℝ) (h : y = (Real.log 3 / Real.log 9)^(Real.log 9 / Real.log 3)) : Real.log 4 y = -1 :=
by {
  sorry
}

end log_value_l151_151028


namespace cans_left_l151_151140

theorem cans_left (bags_saturday bags_sunday cans_per_bag bags_given larger_cans : ℕ) 
  (bags_saturday_eq : bags_saturday = 3)
  (bags_sunday_eq : bags_sunday = 4)
  (cans_per_bag_eq : cans_per_bag = 9)
  (bags_given_eq : bags_given = 2)
  (larger_cans_eq : larger_cans = 2) :
  (bags_saturday * cans_per_bag + bags_sunday * cans_per_bag - bags_given * cans_per_bag + larger_cans * 2) = 49 := 
by
  rw [bags_saturday_eq, bags_sunday_eq, cans_per_bag_eq, bags_given_eq, larger_cans_eq]
  sorry

end cans_left_l151_151140


namespace real_root_fraction_l151_151776

theorem real_root_fraction (a b : ℝ) 
  (h_cond_a : a^4 - 7 * a - 3 = 0) 
  (h_cond_b : b^4 - 7 * b - 3 = 0)
  (h_order : a > b) : 
  (a - b) / (a^4 - b^4) = 1 / 7 := 
sorry

end real_root_fraction_l151_151776


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151511

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151511


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151501

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151501


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151366

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151366


namespace ab_squared_equals_eight_l151_151166

variable (a b : ℝ)

def func_y (x : ℝ) : ℝ := (a * cos x + b * sin x) * cos x

theorem ab_squared_equals_eight
  (h_max : ∀ x : ℝ, func_y a b x ≤ 2)
  (h_min : ∀ x : ℝ, func_y a b x ≥ -1) :
  (a * b) ^ 2 = 8 :=
by
  sorry

end ab_squared_equals_eight_l151_151166


namespace find_p_l151_151960

theorem find_p (x y : ℝ) (h : | x - 1 / 2 | + real.sqrt (y^2 - 1) = 0) : | x | + | y | = 3 / 2 := 
sorry

end find_p_l151_151960


namespace students_taking_single_subject_l151_151800

theorem students_taking_single_subject (geometry_both : ℕ) (total_geometry : ℕ) (only_biology : ℕ) : 
  geometry_both = 15 → 
  total_geometry = 40 → 
  only_biology = 20 → 
  (total_geometry - geometry_both) + only_biology = 45 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end students_taking_single_subject_l151_151800


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151416

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151416


namespace part1_part2_part3_l151_151667

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * (x + 1)

theorem part1 {a : ℝ} : (∀ x : ℝ, f x a = 0 → a = 1) :=
by sorry

theorem part2 {a : ℝ} (h : 0 ≤ a ∧ a ≤ 1) : ∀ x : ℝ, f x a ≥ 0 :=
by sorry

theorem part3 (n : ℕ) (h : 0 < n) : (finset.prod (finset.range n) λ i => 1 + 1 / (2^(i + 1)) < Real.exp 1) :=
by sorry

end part1_part2_part3_l151_151667


namespace cassidy_posters_l151_151897

theorem cassidy_posters (p_two_years_ago : ℕ) (p_double : ℕ) (p_current : ℕ) (p_added : ℕ) 
    (h1 : p_two_years_ago = 14) 
    (h2 : p_double = 2 * p_two_years_ago)
    (h3 : p_current = 22)
    (h4 : p_added = p_double - p_current) : 
    p_added = 6 := 
by
  sorry

end cassidy_posters_l151_151897


namespace determine_a_l151_151969

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if h : x = 3 then a else 2 / |x - 3|

theorem determine_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ 3 ∧ x2 ≠ 3 ∧ (f x1 a - 4 = 0) ∧ (f x2 a - 4 = 0) ∧ f 3 a - 4 = 0) →
  a = 4 :=
by
  sorry

end determine_a_l151_151969


namespace problem1_solution_l151_151149

theorem problem1_solution (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 16)
  (h2 : 5 * x - 6 * y = 33) : 
  x = 6 ∧ y = -1 / 2 := 
  by
  sorry

end problem1_solution_l151_151149


namespace simplify_and_evaluate_expression_l151_151144

variables (a b : ℚ)

theorem simplify_and_evaluate_expression : 
  (4 * (a^2 - 2 * a * b) - (3 * a^2 - 5 * a * b + 1)) = 5 :=
by
  let a := -2
  let b := (1 : ℚ) / 3
  sorry

end simplify_and_evaluate_expression_l151_151144


namespace cone_base_radius_l151_151296

-- Definitions for conditions
def sector_angle : ℝ := 120
def sector_radius : ℝ := 4

-- Theorem to prove
theorem cone_base_radius (r : ℝ) : 
  let arc_length := (sector_angle / 360) * (2 * Real.pi * sector_radius)
  let base_circumference := 2 * Real.pi * r
  arc_length = base_circumference →
  r = 4 / 3 :=
by
  intro h
  have h1 : arc_length = (120 / 360) * (2 * Real.pi * 4), from rfl
  rw h1 at h
  sorry -- skipping proof

end cone_base_radius_l151_151296


namespace composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l151_151280

theorem composite_10201_base_gt_2 (x : ℕ) (hx : x > 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + 2*x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base (x : ℕ) (hx : x ≥ 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base_any_x (x : ℕ) (hx : x ≥ 1) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

end composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l151_151280


namespace faster_whale_turn_back_time_l151_151820

-- Define the conditions and variables
variables (t₀ tₘ tₜ : ℝ) (v₁ v₂ : ℝ)

-- Assume initial separation time is t₀ = 8.25 (8:15 AM in hours)
-- Assume meeting time tₘ = 10 (10 AM in hours)
-- Assume speeds v₁, v₂ are 6 (km/h) and 10 (km/h) respectively

def separation_time := t₀ + tₜ
def meeting_time := tₘ
def initial_speeds := (v₁ = 6) ∧ (v₂ = 10)
def time_to_turn_back (tt tb: ℝ) := tt + tb = tₘ - t₀
def distance_covered := (v₁ * (tₘ - t₀)) = (v₂ * tb) + (v₂ * (tt - tb))

theorem faster_whale_turn_back_time :
  ∀ (t₀ tₘ tₜ v₁ v₂ : ℝ),
    (t₀ = 8.25) ∧ (tₘ = 10) ∧ (v₁ = 6) ∧ (v₂ = 10) →
    (∃ tb tt : ℝ, (9.85 - tb) = 9.51 ∧  
      time_to_turn_back tt tb ∧
      distance_covered) :=
by
  intros t₀ tₘ tₜ v₁ v₂ h,
  -- The proof steps would go here, but for now we use sorry to skip the proof
  sorry

end faster_whale_turn_back_time_l151_151820


namespace maximum_value_of_trig_expr_l151_151727

   theorem maximum_value_of_trig_expr (a b c : ℝ) : 
     ∃ (θ : ℝ), |a * cos θ + b * sin θ + c * (sin θ / cos θ)| ≤ sqrt (a^2 + b^2 + c^2) :=
   sorry
   
end maximum_value_of_trig_expr_l151_151727


namespace age_difference_l151_151716

variable (K_age L_d_age L_s_age : ℕ)

def condition1 : Prop := L_d_age = K_age - 10
def condition2 : Prop := L_s_age = 2 * K_age
def condition3 : Prop := K_age = 12

theorem age_difference : condition1 → condition2 → condition3 → L_s_age - L_d_age = 22 := by
  intros h1 h2 h3
  rw [h3] at h1
  rw [h3] at h2
  simp at h1
  simp at h2
  rw [h1, h2]
  norm_num
  sorry

end age_difference_l151_151716


namespace opposite_of_neg2_l151_151171

theorem opposite_of_neg2 : ∃ y : ℤ, -2 + y = 0 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end opposite_of_neg2_l151_151171


namespace ratio_of_angles_l151_151876

theorem ratio_of_angles 
  (A B C O E : Type) -- Points
  (circ : Circle O) -- Circle with center O
  (triangle : Triangle A B C)
  (is_inscribed : triangle.is_inscribed_in circ)
  (arc1 : Arc circ B A 140)
  (arc2 : Arc circ C B 60)
  (is_acute : triangle.is_acute)
  (E_on_minor_arc : E ∈ (circ.minor_arc A C))
  (perpendicular : ∠ O E ⟂ AC) 
  : ∠OBE / ∠BAC = 4 / 3 := 
sorry

end ratio_of_angles_l151_151876


namespace remainder_of_7_pow_145_mod_12_l151_151827

theorem remainder_of_7_pow_145_mod_12 : (7 ^ 145) % 12 = 7 :=
by
  sorry

end remainder_of_7_pow_145_mod_12_l151_151827


namespace eccentricity_of_hyperbola_l151_151007

noncomputable def hyperbola_eccentricity (a b : ℝ) (P Q F1 F2 : ℝ × ℝ) : ℝ :=
if h₁ : a > 0 ∧ b > 0 ∧ PF2 - PF1 = QF2 ∧ QF1 - QF2 = 2 * a ∧ QF1 = 4 * a then
  sqrt 7
else
  0

theorem eccentricity_of_hyperbola (a b : ℝ) (P Q F1 F2 : ℝ × ℝ) (h : a > 0) (h_cond1 : PF2 - PF1 = QF2) (h_cond2 : QF1 - QF2 = 2 * a) (h_cond3 : QF1 = 4 * a) :
  hyperbola_eccentricity a b P Q F1 F2 = sqrt 7 :=
by sorry

end eccentricity_of_hyperbola_l151_151007


namespace resulting_perimeter_l151_151083

theorem resulting_perimeter (p1 p2 : ℕ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let a := p1 / 4 in
  let b := p2 / 4 in
  p1 + p2 - 2 * a = 120 :=
by
  sorry

end resulting_perimeter_l151_151083


namespace sec_150_l151_151484

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151484


namespace value_of_a_l151_151012

theorem value_of_a (a : ℝ) (A : set ℝ) (B : set ℝ) 
  (hA : A = {-1, 0, a})
  (hB : B = {0, real.sqrt a})
  (h_subset : B ⊆ A) : a = 1 :=
by 
  sorry

end value_of_a_l151_151012


namespace triangle_inscribed_circle_area_l151_151869

noncomputable def circle_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * Real.pi)

noncomputable def triangle_area (r : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin (Real.pi / 2) + Real.sin (2 * Real.pi / 3) + Real.sin (5 * Real.pi / 6))

theorem triangle_inscribed_circle_area (a b c : ℝ) (h : a + b + c = 24) :
  ∀ (r : ℝ) (h_r : r = circle_radius 24),
  triangle_area r = 72 / Real.pi^2 * (Real.sqrt 3 + 1) :=
by
  intro r h_r
  rw [h_r, circle_radius, triangle_area]
  sorry

end triangle_inscribed_circle_area_l151_151869


namespace perimeter_of_resulting_figure_l151_151078

-- Define the perimeters of the squares
def perimeter_small_square : ℕ := 40
def perimeter_large_square : ℕ := 100

-- Define the side lengths of the squares
def side_length_small_square := perimeter_small_square / 4
def side_length_large_square := perimeter_large_square / 4

-- Define the total perimeter of the uncombined squares
def total_perimeter_uncombined := perimeter_small_square + perimeter_large_square

-- Define the shared side length
def shared_side_length := side_length_small_square

-- Define the perimeter after considering the shared side
def resulting_perimeter := total_perimeter_uncombined - 2 * shared_side_length

-- Prove that the resulting perimeter is 120 cm
theorem perimeter_of_resulting_figure : resulting_perimeter = 120 := by
  sorry

end perimeter_of_resulting_figure_l151_151078


namespace problem1_problem2_problem3_problem4_l151_151341

-- Problem 1
theorem problem1 : 8 - (-4) - (+3) - 5 = 4 := 
by simp only [neg_neg, sub_eq_add_neg, add_assoc]; exact rfl

-- Problem 2
theorem problem2 : - (9 / 10) / (2 / 5) * (- (5 / 2)) = 9 / 8 :=
by linarith

-- Problem 3
theorem problem3 : ((1 / 2 - 5 / 9 + 5 / 6 - 7 / 12) * (-36)) + (-3)^2 = 2 :=
by linarith

-- Problem 4
theorem problem4 : - 1^2 + 16 / (-2)^3 * (-4) = 7 :=
by linarith

end problem1_problem2_problem3_problem4_l151_151341


namespace sec_150_eq_l151_151401

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151401


namespace cost_of_article_l151_151029

theorem cost_of_article (C: ℝ) (G: ℝ) (h1: 380 = C + G) (h2: 420 = C + G + 0.05 * C) : C = 800 :=
by
  sorry

end cost_of_article_l151_151029


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151361

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151361


namespace range_of_k_l151_151691

def f (x : ℝ) : ℝ := x^3 - 12*x

def not_monotonic_on_I (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), k - 1 < x₁ ∧ x₁ < k + 1 ∧ k - 1 < x₂ ∧ x₂ < k + 1 ∧ x₁ ≠ x₂ ∧ (f x₁ - f x₂) * (x₁ - x₂) < 0

theorem range_of_k (k : ℝ) : not_monotonic_on_I k ↔ (k > -3 ∧ k < -1) ∨ (k > 1 ∧ k < 3) :=
sorry

end range_of_k_l151_151691


namespace sum_of_roots_gt_two_l151_151972

noncomputable def f : ℝ → ℝ := λ x => Real.log x - x + 1

theorem sum_of_roots_gt_two (m : ℝ) (x1 x2 : ℝ) (hx1 : f x1 = m) (hx2 : f x2 = m) (hne : x1 ≠ x2) : x1 + x2 > 2 := by
  sorry

end sum_of_roots_gt_two_l151_151972


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151383

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151383


namespace max_size_subs_no_sum_n_eq_l151_151092

/-!
  Prove that the maximum size of a subset S of the set {1, 2, ..., k}
  such that no n distinct elements of S add up to m is 
  k - floor (m / n - (n - 1) / 2), given the conditions 
  1 < n ≤ m - 1 ≤ k for integers k, m, n.
-/

noncomputable def maxSubsetWithoutSum (k m n : ℕ) (h1 : 1 < n) (h2 : n ≤ m - 1) (h3 : m - 1 ≤ k) : ℕ :=
  k - (m / n - (n - 1) / 2)

theorem max_size_subs_no_sum_n_eq (k m n : ℕ) (h1 : 1 < n) (h2 : n ≤ m - 1) (h3 : m - 1 ≤ k) :
  ∃ (S : Finset ℕ), (∀ (T : Finset ℕ), (T ⊆ S ∧ T.card = n) → ¬(T.sum = m)) ∧ S.card = maxSubsetWithoutSum k m n h1 h2 h3 :=
begin
  sorry
end

end max_size_subs_no_sum_n_eq_l151_151092


namespace sequence_term_is_square_l151_151753

noncomputable def sequence_term (n : ℕ) : ℕ :=
  let part1 := (10 ^ (n + 1) - 1) / 9
  let part2 := (10 ^ (2 * n + 2) - 10 ^ (n + 1)) / 9
  1 + 4 * part1 + 4 * part2

theorem sequence_term_is_square (n : ℕ) : ∃ k : ℕ, k^2 = sequence_term n :=
by
  sorry

end sequence_term_is_square_l151_151753


namespace resulting_perimeter_l151_151082

theorem resulting_perimeter (p1 p2 : ℕ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let a := p1 / 4 in
  let b := p2 / 4 in
  p1 + p2 - 2 * a = 120 :=
by
  sorry

end resulting_perimeter_l151_151082


namespace non_attacking_knight_count_l151_151994

def knight_moves (pos : ℕ × ℕ) : Finset (ℕ × ℕ) :=
  {(pos.1 + 2, pos.2 + 1), (pos.1 + 2, pos.2 - 1),
   (pos.1 - 2, pos.2 + 1), (pos.1 - 2, pos.2 - 1),
   (pos.1 + 1, pos.2 + 2), (pos.1 + 1, pos.2 - 2),
   (pos.1 - 1, pos.2 + 2), (pos.1 - 1, pos.2 - 2)}.filter
  (λ p, 1 ≤ p.1 ∧ p.1 ≤ 8 ∧ 1 ≤ p.2 ∧ p.2 ≤ 8)

def knight_attacks : Finset (ℕ × ℕ) → Finset (ℕ × ℕ) :=
  Finset.bUnion knight_moves

def non_attacking_knight_placements : Finset (ℕ × ℕ) × Finset (ℕ × ℕ) :=
  (Finset.product (Finset.range 8.succ).product (Finset.range 8.succ))

theorem non_attacking_knight_count :
  non_attacking_knight_placements.card = 3696 :=
by sorry

end non_attacking_knight_count_l151_151994


namespace problem_l151_151665

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos (x / 2))^2 - Real.sqrt 3 * Real.sin x 

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem problem 
  (a : ℝ) (h_in_quadrant_II : (π / 2) < a ∧ a < π)
  (h_f_a_minus_pi_over_3 : f (a - π / 3) = 1 / 3) :
  (is_periodic f (2 * π)) ∧
  (∀ x : ℝ, f x ≥ -1 ∧ f x ≤ 3) ∧
  (cos a = -1 / 3) →
  (cos 2a / (1 + cos 2a - sin 2a) = (1 - 2 * Real.sqrt 2) / 2) :=
by
  sorry

end problem_l151_151665


namespace sec_150_l151_151489

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151489


namespace plan_3_smallest_reduction_l151_151287

def price_reduction_plan_1 (x a b : ℝ) : ℝ :=
  (x * (1 - a / 100)) * (1 - b / 100)

def price_reduction_plan_2 (x a b : ℝ) : ℝ :=
  (x * (1 - (a + b) / 200)) * (1 - a / 100)

def price_reduction_plan_3 (x a b : ℝ) : ℝ :=
  (x * (1 - (a + b) / 200)) * (1 - (a + b) / 200)

def price_reduction_plan_4 (x a b : ℝ) : ℝ :=
  x * (1 - (a + b) / 100)

theorem plan_3_smallest_reduction (x a b : ℝ) (h : a > b) :
  price_reduction_plan_3 x a b < price_reduction_plan_1 x a b ∧
  price_reduction_plan_3 x a b < price_reduction_plan_2 x a b ∧
  price_reduction_plan_3 x a b < price_reduction_plan_4 x a b :=
sorry

end plan_3_smallest_reduction_l151_151287


namespace find_a_l151_151965

noncomputable def polynomial := (x + a) ^ 2 * (x - 1) ^ 3

theorem find_a (a : ℝ) (h : polynomial.coeff(4) = 1) : a = 2 :=
by sorry

end find_a_l151_151965


namespace coefficient_of_x4_in_expansion_l151_151244

noncomputable def problem_statement : ℕ :=
  let n := 8
  let a := 2
  let b := 3
  let k := 4
  binomial n k * (b ^ k) * (a ^ (n - k))

theorem coefficient_of_x4_in_expansion :
  problem_statement = 90720 :=
by
  sorry

end coefficient_of_x4_in_expansion_l151_151244


namespace number_of_n_satisfying_conditions_l151_151930

theorem number_of_n_satisfying_conditions : 
  {n : ℕ | n ≤ 800 ∧ (∃ k : ℕ, 240 * n = k^3)}.finite.card = 4 := 
sorry

end number_of_n_satisfying_conditions_l151_151930


namespace overall_average_score_l151_151157

variables (average_male average_female sum_male sum_female total_sum : ℕ)
variables (count_male count_female total_count : ℕ)

def average_score (sum : ℕ) (count : ℕ) : ℕ := sum / count

theorem overall_average_score
  (average_male : ℕ := 84)
  (count_male : ℕ := 8)
  (average_female : ℕ := 92)
  (count_female : ℕ := 24)
  (sum_male : ℕ := count_male * average_male)
  (sum_female : ℕ := count_female * average_female)
  (total_sum : ℕ := sum_male + sum_female)
  (total_count : ℕ := count_male + count_female) :
  average_score total_sum total_count = 90 := 
sorry

end overall_average_score_l151_151157


namespace sum_first_8_terms_l151_151644

variable {α : Type*} [LinearOrderedField α]

-- Define the arithmetic sequence
def arithmetic_sequence (a_1 d : α) (n : ℕ) : α := a_1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a_1 d : α) (n : ℕ) : α :=
  (n * (2 * a_1 + (n - 1) * d)) / 2

-- Define the given condition
variable (a_1 d : α)
variable (h : arithmetic_sequence a_1 d 3 = 20 - arithmetic_sequence a_1 d 6)

-- Statement of the problem
theorem sum_first_8_terms : sum_arithmetic_sequence a_1 d 8 = 80 :=
by
  sorry

end sum_first_8_terms_l151_151644


namespace coefficient_of_x_in_binomial_expansion_l151_151706

theorem coefficient_of_x_in_binomial_expansion :
  let c := polynomial.coeff (polynomial.expand 5 (2 * polynomial.X ^ 2 - polynomial.X⁻¹)) 1 in
  c = -40 :=
by
  sorry

end coefficient_of_x_in_binomial_expansion_l151_151706


namespace problem_proof_l151_151953

noncomputable def a_n (n : ℕ) : ℝ :=
  2 * n + 8

noncomputable def b_n (n : ℕ) : ℝ :=
  2^(n-1)

noncomputable def C_n (n : ℕ) : ℝ :=
  min (a_n n) (b_n n)

noncomputable def S_n : ℕ → ℝ
| 0     := 0
| (n+1) :=
  if h : n + 1 ≤ 5 then
    (4^n - 1) / 3
  else
    4 / 3 * (n + 1)^3 + 18 * (n + 1)^2 + 242 / 3 * (n + 1) - 679

theorem problem_proof (n : ℕ) : 
  a_n n = 2 * n + 8 ∧ 
  b_n n = 2^(n-1) ∧ 
  S_n n = 
    (if h : n ≤ 5 then (4^n - 1) / 3 else 4 / 3 * n^3 + 18 * n^2 + 242 / 3 * n - 679) :=
by
  sorry

end problem_proof_l151_151953


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151529

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151529


namespace problem_1_problem_2_l151_151634

variable (x y : ℝ)
noncomputable def x_val : ℝ := 2 + Real.sqrt 3
noncomputable def y_val : ℝ := 2 - Real.sqrt 3

theorem problem_1 :
  3 * x_val^2 + 5 * x_val * y_val + 3 * y_val^2 = 47 := sorry

theorem problem_2 :
  Real.sqrt (x_val / y_val) + Real.sqrt (y_val / x_val) = 4 := sorry

end problem_1_problem_2_l151_151634


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151584

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151584


namespace opposite_of_neg_two_is_two_l151_151177

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l151_151177


namespace angle_bisector_through_midpoint_l151_151637

/-- Definition of midpoint for clarity -/
def is_midpoint {α : Type*} [add_comm_group α] (M : α) (A B : α) : Prop :=
  M = (A + B) / 2

/-- Definition of angle bisector for clarity -/
def angle_bisector (P A B M : Affine λ0) : Prop :=
  -- To be defined properly based on your affine geometry library and model
  sorry

/-- Main theorem -/
theorem angle_bisector_through_midpoint
  (A B C D M : Affine λ0)
  (h_convex : convex_quadrilateral A B C D)
  (h_ad_eq_ab_cd : dist A D = dist A B + dist C D)
  (h_angle_bisector_A : angle_bisector A M B (midpoint B C))
  (h_midpoint_BC : is_midpoint M B C) :
  angle_bisector D M C (midpoint B C) := sorry


end angle_bisector_through_midpoint_l151_151637


namespace sec_150_eq_l151_151462

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151462


namespace sec_150_eq_l151_151391

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151391


namespace graph_y_eq_f_x_plus_2_is_A_l151_151669

def f (x : ℝ) : ℝ :=
if h1 : -3 ≤ x ∧ x ≤ 0 then -2 - x
else if h2 : 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
else if h3 : 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
else 0

def y_translation (h : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
λ x, h x + k

theorem graph_y_eq_f_x_plus_2_is_A :
  ∀ p : ℝ × ℝ, ((p.2 = y_translation f 2 p.1) ↔ (p ∈ set_of_a_translated_vertical_by_2_units ↑graph_of_f_and_is_A)) :=
sorry

end graph_y_eq_f_x_plus_2_is_A_l151_151669


namespace opposite_of_neg_two_l151_151180

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l151_151180


namespace smallest_x_mod_conditions_l151_151259

theorem smallest_x_mod_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ x % 7 = 6 ∧ x = 209 := by
  sorry

end smallest_x_mod_conditions_l151_151259


namespace length_of_AB_l151_151703

-- Definitions of the lengths and the given conditions
variables (AB CD : ℝ)
variables (ratio_area : AB / CD = 5 / 2)
variables (sum_length : AB + CD = 280)

-- The theorem stating the goal to prove
theorem length_of_AB (h : ℝ) (h_nonneg : 0 ≤ h) (Hratio : ratio_area) (Hsum : sum_length) : AB = 200 :=
by
  -- proof goes here
  sorry

end length_of_AB_l151_151703


namespace opposite_of_neg_two_l151_151189

theorem opposite_of_neg_two : ∀ x : ℤ, (-2 + x = 0) → (x = 2) :=
begin
  assume x hx,
  sorry

end opposite_of_neg_two_l151_151189


namespace find_m_l151_151004

noncomputable def g (x : ℝ) (m : ℝ) : ℝ :=
  2 * sqrt 3 * real.sin x * real.cos x + 2 * (real.cos x)^2 + m

theorem find_m :
  ∃ m : ℝ, ∀ x ∈ set.Icc 0 (real.pi / 2), g x m ≤ 6 ∧ ∃ y ∈ set.Icc 0 (real.pi / 2), g y m = 6 :=
sorry

end find_m_l151_151004


namespace regular_hexagon_product_one_l151_151861

noncomputable def regular_hexagon_product (Q : ℕ → ℂ) : ℂ :=
  (Q 1) * (Q 2) * (Q 3) * (Q 4) * (Q 5) * (Q 6)

theorem regular_hexagon_product_one 
  (Q : ℕ → ℂ) 
  (hex : ∀ n, (n ≥ 1 ∧ n ≤ 6) → Q n ∈ (λ {z : ℂ}, z^6 = 1)) 
  (Q1_pos : Q 1 = 1)
  (Q4_neg : Q 4 = -1) :
  regular_hexagon_product Q = 1 :=
begin
  sorry
end

end regular_hexagon_product_one_l151_151861


namespace symbol_opposite_bullet_is_Delta_l151_151805

-- Definitions corresponding to the given conditions
def cube_face_symbols : list char := ['\u25A1', '\u25B3', '\u25E6', '\u2022', '+', 'O']

def view1 : list char := ['\u25A1', '\u25B3', 'O']
def view2 : list char := ['\u2022', '\u25B2', 'O']
def view3 : list char := ['+', '\u25B2', '\u25A1']

-- Theorem statement
theorem symbol_opposite_bullet_is_Delta :
  ∀ (cube_face_symbols view1 view2 view3 : list char),
  cube_face_symbols = ['\u25A1', '\u25B3', '\u25E6', '\u2022', '+', 'O'] →
  view1 = ['\u25A1', '\u25B3', 'O'] →
  view2 = ['\u2022', '\u25B2', 'O'] →
  view3 = ['+', '\u25B2', '\u25A1'] →
  exists face, face = '\u25B2' ∧ (face ∈ cube_face_symbols) ∧ ¬(face ∈ ['\u25A1', 'O', '\u2022'] ∨ (face ∈ ['+', '\u2022', 'O']) ) := 
sorry

end symbol_opposite_bullet_is_Delta_l151_151805


namespace frequency_of_middle_group_l151_151699

theorem frequency_of_middle_group
    (num_rectangles : ℕ)
    (middle_area : ℝ)
    (other_areas_sum : ℝ)
    (sample_size : ℕ)
    (total_area_norm : ℝ)
    (h1 : num_rectangles = 11)
    (h2 : middle_area = other_areas_sum)
    (h3 : sample_size = 160)
    (h4 : middle_area + other_areas_sum = total_area_norm)
    (h5 : total_area_norm = 1):
    160 * (middle_area / total_area_norm) = 80 :=
by
  sorry

end frequency_of_middle_group_l151_151699


namespace opposite_of_neg_two_l151_151185

theorem opposite_of_neg_two : ∀ x : ℤ, (-2 + x = 0) → (x = 2) :=
begin
  assume x hx,
  sorry

end opposite_of_neg_two_l151_151185


namespace equations_of_median_and_altitude_l151_151642

def midpoint (p1 p2 : ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def slope (p1 p2 : ℝ × ℝ) :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem equations_of_median_and_altitude :
  let A := (-5, 0) in
  let B := (4, -4) in
  let C := (0, 2) in
  let D := midpoint B C in
  let k_AD := slope A D in
  let k_BC := slope B C in
  let k_altitude := -1 / k_BC in
  let median_equation := (λ x y : ℝ, x + 7 * y + 5 = 0) in
  let altitude_equation := (λ x y : ℝ, 2 * x - 3 * y + 10 = 0) in
  median_equation = (λ x y : ℝ, x + 7 * y + 5 = 0) ∧
  altitude_equation = (λ x y : ℝ, 2 * x - 3 * y + 10 = 0) := sorry

end equations_of_median_and_altitude_l151_151642


namespace find_vals_of_a_b_l151_151974

noncomputable theory

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x + a / x + b

theorem find_vals_of_a_b (a b : ℝ) (x : ℝ) (hx : x ≠ 0)
  (tangent_eq : ∀ y, y = -2 * x + 5) :
  f 1 a b = 1 + a + b ∧ f' 1 a = 1 - a ∧ a - b = 4 :=
by
  sorry

end find_vals_of_a_b_l151_151974


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151371

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151371


namespace sec_150_l151_151476

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151476


namespace sin_cos_inequality_l151_151143

variable (x : ℝ) (n : ℕ)
noncomputable def s := Real.sin x
noncomputable def c := Real.cos x

theorem sin_cos_inequality (h : s^2 + c^2 = 1) : 
  (Real.sin (2 * x))^n + (s^n - c^n)^2 ≤ 1 := 
sorry

end sin_cos_inequality_l151_151143


namespace sec_150_l151_151481

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151481


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151359

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151359


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151531

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151531


namespace pyramid_volume_l151_151100

theorem pyramid_volume (a b c d : ℝ)
  (h1 : b = a - 2)
  (h2 : c = a - 4)
  (h3 : ∀ x y z, (x - y) * (y - z) + (y - z) * (z - x) + (z - x) * (x - y) = 0)
  :
  ∀ AD BD CD, AD * AD = a * a - d * d → BD * BD = (a - 2) * (a - 2) - d * d → CD * CD = (a - 4) * (a - 4) - d * d → 
  √(a^2 - d^2) = AD → 
  ∃ V, V = (1 / 3) * (a^2 - d^2) * d := 
  begin
    sorry,
  end

end pyramid_volume_l151_151100


namespace coefficient_x4_in_expansion_l151_151239

theorem coefficient_x4_in_expansion : 
  (∑ k in Finset.range (9), (Nat.choose 8 k) * (3 : ℤ)^k * (2 : ℤ)^(8-k) * (X : ℤ[X])^k).coeff 4 = 90720 :=
by
  sorry

end coefficient_x4_in_expansion_l151_151239


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151370

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151370


namespace find_number_of_blue_balls_l151_151883

-- Define the conditions: number of green and blue balls in each urn and the probability condition.
def num_green_balls_urn1 := 3
def num_blue_balls_urn1 := 5
def num_green_balls_urn2 := 9
def draws_same_color_probability := 0.55

-- Define the total number of balls in each urn
def total_balls_urn1 := num_green_balls_urn1 + num_blue_balls_urn1
def total_balls_urn2 (N : ℕ) := num_green_balls_urn2 + N

-- Probability definitions
def probability_both_green (N : ℕ) := (num_green_balls_urn1 / total_balls_urn1) * (num_green_balls_urn2 / total_balls_urn2 N)
def probability_both_blue (N : ℕ) := (num_blue_balls_urn1 / total_balls_urn1) * (N / total_balls_urn2 N)

-- The main goal: Prove that the required N satisfies the condition for the given probabilities.
theorem find_number_of_blue_balls (N : ℕ) (h : probability_both_green N + probability_both_blue N = draws_same_color_probability) : N = 21 := by
  -- The proof part will be filled in here.
  sorry

end find_number_of_blue_balls_l151_151883


namespace opposite_of_neg2_l151_151174

theorem opposite_of_neg2 : ∃ y : ℤ, -2 + y = 0 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end opposite_of_neg2_l151_151174


namespace complex_roots_right_triangle_l151_151735

noncomputable def root_condition 
  (a b z1 z2 : ℂ) : Prop := z1^2 + a * z1 * z2 + b * z2 = 0

noncomputable def right_triangle_condition
  (z1 z2 : ℂ) : Prop := z2 = z1 * complex.I

theorem complex_roots_right_triangle 
  (a b z1 z2 : ℂ) 
  (h1: root_condition a b z1 z2) 
  (h2: right_triangle_condition z1 z2) :
  (a^2 / b) = 2 := sorry

end complex_roots_right_triangle_l151_151735


namespace number_of_schools_is_29_l151_151914

variables {rank_alex rank_briana rank_charlie rank_dana : ℕ} 
          {schools : ℕ}

-- Define the conditions
def high_school_sends_four_students : Prop := ∀ (s : ℕ), s ∈ {rank_alex, rank_briana, rank_charlie, rank_dana} → rank_alex < 40
def alex_median_score : Prop := true -- Placeholder, exact condition requires median but related to specifics in the problem
def highest_score_team : Prop := rank_alex > rank_briana ∧ rank_alex > rank_charlie ∧ rank_alex > rank_dana
def teammates_ranks : Prop := rank_briana = 40 ∧ rank_charlie = 75 ∧ rank_dana = 90

-- Using these conditions, assert the proof problem
theorem number_of_schools_is_29
  (h1 : high_school_sends_four_students)
  (h2 : alex_median_score)
  (h3 : highest_score_team)
  (h4 : teammates_ranks)
  (rank_alex : ℕ) : schools = 29 := sorry

end number_of_schools_is_29_l151_151914


namespace fraction_comparison_l151_151911

theorem fraction_comparison : 
  (1 / (Real.sqrt 2 - 1)) < (Real.sqrt 3 + 1) :=
sorry

end fraction_comparison_l151_151911


namespace distinct_points_common_to_ellipses_l151_151169

theorem distinct_points_common_to_ellipses : 
  let ellipse1 := (x y : ℝ) → x^2 + 9 * y^2 = 9,
      ellipse2 := (x y : ℝ) → x^2 + 4 * y^2 = 4 in
  ∃! p : ℝ × ℝ, ellipse1 p.1 p.2 ∧ ellipse2 p.1 p.2 :=
begin
  sorry -- Proof goes here
end

end distinct_points_common_to_ellipses_l151_151169


namespace real_solution_count_l151_151024

theorem real_solution_count :
  {x : ℝ | |x - 2| = |x - 1| + |x - 4|}.finite.to_finset.card = 1 :=
by
  sorry

end real_solution_count_l151_151024


namespace number_of_sets_satisfying_condition_l151_151603

theorem number_of_sets_satisfying_condition : 
  (∃ A : set ℕ, {0, 1} ∪ A = {0, 1}) → ∃ n : ℕ, n = 4 := 
by
  sorry

end number_of_sets_satisfying_condition_l151_151603


namespace general_term_a_sum_of_bn_l151_151641

-- Part 1: General term formula for the sequence {a_n}
theorem general_term_a (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : ∀ (n : ℕ), n > 0 → S n = 1 - a n) :
    a = λ n, (n - 2) / 2 := sorry

-- Part 2: Sum of the first n terms of the sequence {b_n}
theorem sum_of_bn (b T : ℕ → ℝ) (h2 : ∀ (n: ℕ), n > 0 → b n = n * 2^n) :
    (∀ n > 0, T n = ∑ i in finset.range n, b i) →
    ∀ n > 0, T n = (n - 1) * 2^(n+1) + 2 := sorry

end general_term_a_sum_of_bn_l151_151641


namespace original_selling_price_l151_151336

variable (P : ℝ)

def SP1 := 1.10 * P
def P_new := 0.90 * P
def SP2 := 1.17 * P
def price_diff := SP2 - SP1

theorem original_selling_price : price_diff = 49 → SP1 = 770 :=
by
  sorry

end original_selling_price_l151_151336


namespace rectangle_error_percent_deficit_l151_151053

theorem rectangle_error_percent_deficit (L W : ℝ) (p : ℝ) 
    (h1 : L > 0) (h2 : W > 0)
    (h3 : 1.05 * (1 - p) = 1.008) :
    p = 0.04 :=
by
  sorry

end rectangle_error_percent_deficit_l151_151053


namespace quadratic_roots_complex_non_real_l151_151905

theorem quadratic_roots_complex_non_real :
  let a := 4
  let b := -2 * Real.sqrt 2
  let c := 3
  Discriminant := b^2 - 4 * a * c
  Discriminant < 0 →
  ∃ z1 z2 : ℂ, z1 ≠ z2 ∧ (z1.re = z2.re ∧ z1.im = -z2.im) :=
sorry

end quadratic_roots_complex_non_real_l151_151905


namespace vector_AP_formula_l151_151067

variable (A B C M N P : Type)
variable [add_group A] [vector_space ℝ A] [add_group B] [vector_space ℝ B] [add_group C]
variable [has_add P] [has_vsub P A] [vector_space ℝ P]
variable (a b : P)

-- Definitions for midpoint and vectors
def midpoint (X Y : P) : P := (X + Y) / (2 : ℝ)

-- Conditions
variable (CA CB AB : P)
variable (M : P := midpoint CA CB)
variable (N : P := midpoint AB CA)
variable (CN AM : P)

-- Given conditions
axiom CA_def : CA = a
axiom CB_def : CB = b
axiom M_def : M = midpoint CA CB
axiom N_def : N = midpoint AB CA
axiom CN_AM_intersect_at_P : CN + AM = P

-- Proof goal
theorem vector_AP_formula :
  \overrightarrow{AP} = (1/3) * b - (2/3) * a :=
sorry

end vector_AP_formula_l151_151067


namespace compare_trig_functions_l151_151941

theorem compare_trig_functions :
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c :=
by
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  sorry

end compare_trig_functions_l151_151941


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151542

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151542


namespace equidistant_point_intersection_of_perpendicular_bisectors_l151_151194

/-
Prove that the point equidistant from all three vertices of a triangle 
is the intersection of the perpendicular bisectors of the sides of the triangle,
given that a point equidistant from the endpoints of a segment 
lies on the perpendicular bisector of that segment.
-/
theorem equidistant_point_intersection_of_perpendicular_bisectors 
  {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (triangle : Triangle A B C) :
  ∃ P,
    (∀ v ∈ {triangle.v1, triangle.v2, triangle.v3}, dist P v = dist P (triangle.v1)) ∧
    (P ∈ perpendicular_bisector (segment triangle.v1 triangle.v2)) ∧
    (P ∈ perpendicular_bisector (segment triangle.v2 triangle.v3)) ∧
    (P ∈ perpendicular_bisector (segment triangle.v3 triangle.v1)) :=
sorry

end equidistant_point_intersection_of_perpendicular_bisectors_l151_151194


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151377

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151377


namespace count_int_values_not_satisfying_ineq_l151_151615

theorem count_int_values_not_satisfying_ineq :
  ∃ (s : Finset ℤ), (∀ x ∈ s, 3 * x^2 + 14 * x + 8 ≤ 17) ∧ (s.card = 10) :=
by
  sorry

end count_int_values_not_satisfying_ineq_l151_151615


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151539

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151539


namespace odd_if_and_only_if_m_even_l151_151734

variables (o n m : ℕ)

theorem odd_if_and_only_if_m_even
  (h_o_odd : o % 2 = 1) :
  ((o^3 + n*o + m) % 2 = 1) ↔ (m % 2 = 0) :=
sorry

end odd_if_and_only_if_m_even_l151_151734


namespace intersection_complement_U_B_l151_151117

open Set

-- Assume the universal set U, and the sets A and B
def U := {-2, -1, 0, 1, 2, 3}
def A := {2, 3}
def B := {-1, 0}

-- Define the complement of B in the universe U
def complement_U_B := U \ B

theorem intersection_complement_U_B :
  A ∩ complement_U_B = {2, 3} := by
  sorry

end intersection_complement_U_B_l151_151117


namespace sum_abs_b_i_l151_151621

noncomputable def P (x : ℝ) : ℝ :=
  1 - (2 / 5) * x + (1 / 5) * x^3

noncomputable def Q (x : ℝ) : ℝ :=
  P(x) * P(x^2) * P(x^4) * P(x^6) * P(x^8)

theorem sum_abs_b_i :
  (∑ i in Finset.range 36, |(Q (x : ℝ)).coeff i|) = (32 / 3125) :=
by
  sorry

end sum_abs_b_i_l151_151621


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151533

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151533


namespace progress_reach_10000_regress_after_23_months_l151_151274

theorem progress_reach_10000_regress_after_23_months
  (progress_rate regress_rate : ℝ)
  (lg2 lg3: ℝ)
  (h_progress : progress_rate = 1.2)
  (h_regress : regress_rate = 0.8)
  (h_lg2 : lg2 ≈ 0.3010)
  (h_lg3 : lg3 ≈ 0.4771) :
  (∃ x : ℕ, (progress_rate / regress_rate) ^ x = 10000 ∧ x ≈ 23) :=
by
  sorry

end progress_reach_10000_regress_after_23_months_l151_151274


namespace parabola_translation_l151_151215

theorem parabola_translation :
  ∀ (x y : ℝ), (y = x^2 - 2x + 4) → (∃ x' y' : ℝ, y' = (x' - 2)^2 + 6 ∧ x' = x + 1 ∧ y' = y + 3) → y = x^2 - 4x + 10 :=
by
  intro x y
  intro h_eq
  intro ⟨x', y', h_y', h_x', h_y''⟩
  rw [h_eq, h_y', h_x', h_y'']
  sorry

end parabola_translation_l151_151215


namespace problem_proof_l151_151939

theorem problem_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = ab → a + 4 * b = 9) ∧
  (a + b = 1 → ∀ a b,  2^a + 2^(b + 1) ≥ 4) ∧
  (a + b = ab → 1 / a^2 + 2 / b^2 = 2 / 3) ∧
  (a + b = 1 → ∀ a b,  2 * a / (a + b^2) + b / (a^2 + b) = (2 * Real.sqrt 3 / 3) + 1) :=
by
  sorry

end problem_proof_l151_151939


namespace price_increase_percentage_l151_151265
noncomputable def manufacturing_cost : ℝ := 100
noncomputable def retail_price (cost : ℝ) : ℝ := cost * 1.4
noncomputable def customer_price (retail_price : ℝ) : ℝ := retail_price * 1.4

theorem price_increase_percentage :
  let cost := manufacturing_cost in
  let retail := retail_price cost in
  let customer := customer_price retail in
  (customer - cost) / cost * 100 = 96 := by
    sorry

end price_increase_percentage_l151_151265


namespace power_function_value_at_16_l151_151784

theorem power_function_value_at_16 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f(x) = x^a) (h2 : f(4) = 2) : f(16) = 4 :=
by {
  sorry
}

end power_function_value_at_16_l151_151784


namespace largest_apartment_size_l151_151331

theorem largest_apartment_size (rate_per_square_foot : ℝ) (affordable_amount : ℝ) : 
  rate_per_square_foot = 1.25 → affordable_amount = 750 → ∃ (s : ℝ), s = 600 :=
by
  intros h_rate h_amount
  use 600
  rw [h_rate, h_amount]
  have h : 1.25 * 600 = 750 := by norm_num
  exact h.symm

end largest_apartment_size_l151_151331


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151422

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151422


namespace eggs_found_at_club_house_l151_151690

theorem eggs_found_at_club_house (E_park E_townhall total_eggs : ℕ) 
  (h_park : E_park = 25) 
  (h_townhall : E_townhall = 15)
  (h_total : total_eggs = 80) :
  ∀ E_club, E_club + E_park + E_townhall = total_eggs → E_club = 40 :=
by
  intros E_club h_eq
  rw [h_park, h_townhall, h_total] at h_eq
  simp [h_eq]
  sorry

end eggs_found_at_club_house_l151_151690


namespace intersection_of_M_and_N_l151_151738

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_of_M_and_N_l151_151738


namespace probability_of_same_type_is_correct_l151_151355

noncomputable def total_socks : ℕ := 12 + 10 + 6
noncomputable def ways_to_pick_any_3_socks : ℕ := Nat.choose total_socks 3
noncomputable def ways_to_pick_3_black_socks : ℕ := Nat.choose 12 3
noncomputable def ways_to_pick_3_white_socks : ℕ := Nat.choose 10 3
noncomputable def ways_to_pick_3_striped_socks : ℕ := Nat.choose 6 3
noncomputable def ways_to_pick_3_same_type : ℕ := ways_to_pick_3_black_socks + ways_to_pick_3_white_socks + ways_to_pick_3_striped_socks
noncomputable def probability_same_type : ℚ := ways_to_pick_3_same_type / ways_to_pick_any_3_socks

theorem probability_of_same_type_is_correct :
  probability_same_type = 60 / 546 :=
by
  sorry

end probability_of_same_type_is_correct_l151_151355


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151517

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151517


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151576

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151576


namespace compare_powers_l151_151275

theorem compare_powers :
  let a := 5 ^ 140
  let b := 3 ^ 210
  let c := 2 ^ 280
  c < a ∧ a < b := by
  -- Proof omitted
  sorry

end compare_powers_l151_151275


namespace james_initial_gallons_l151_151710

/-- 
  James has some gallons of milk. He drank 13 ounces of the milk. 
  There are 128 ounces in a gallon. James has 371 ounces of milk left.
  Prove that James initially had 3 gallons of milk.
-/
theorem james_initial_gallons (o_drank o_per_gallon o_left : ℕ) (h1 : o_drank = 13) (h2 : o_per_gallon = 128) (h3 : o_left = 371) : 
  (o_left + o_drank) / o_per_gallon = 3 :=
by 
  rw [h1, h2, h3]
  sorry -- proof steps are omitted

end james_initial_gallons_l151_151710


namespace median_group_estimate_households_below_178_l151_151814

def HouseholdDistribution : Type :=
  List (Nat × Nat × Nat)

def distribution : HouseholdDistribution :=
  [ (1, 8, 93), (50),
    (2, 93, 178), (100),
    (3, 178, 263), (34),
    (4, 263, 348), (11),
    (5, 348, 433), (1),
    (6, 433, 518), (1),
    (7, 518, 603), (2),
    (8, 603, 688), (1) ]

def sample_size : Nat := 200

def total_households : Nat := 10000

theorem median_group (dist : HouseholdDistribution) (n : Nat) (g : Nat) :
  dist = distribution → n = sample_size → g = 2 → 
  (find_median_group dist n = g) :=
by
  sorry

theorem estimate_households_below_178 (dist : HouseholdDistribution) (n m : Nat) (estimate : Nat) :
  dist = distribution → n = sample_size → m = total_households → estimate = 7500 → 
  (estimate_below_178 dist n m = estimate) :=
by
  sorry

end median_group_estimate_households_below_178_l151_151814


namespace cars_given_by_mum_and_dad_l151_151748

-- Define the conditions given in the problem
def initial_cars : ℕ := 150
def final_cars : ℕ := 196
def cars_by_auntie : ℕ := 6
def cars_more_than_uncle : ℕ := 1
def cars_given_by_family (uncle : ℕ) (grandpa : ℕ) (auntie : ℕ) : ℕ :=
  uncle + grandpa + auntie

-- Prove the required statement
theorem cars_given_by_mum_and_dad :
  ∃ (uncle grandpa : ℕ), grandpa = 2 * uncle ∧ auntie = uncle + cars_more_than_uncle ∧ 
    auntie = cars_by_auntie ∧
    final_cars - initial_cars - cars_given_by_family uncle grandpa auntie = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end cars_given_by_mum_and_dad_l151_151748


namespace toms_speed_is_correct_l151_151743

def max_speed : ℝ := 5
def lila_speed : ℝ := (4 / 5) * max_speed
def tom_speed : ℝ := (6 / 7) * lila_speed

theorem toms_speed_is_correct : tom_speed = 24 / 7 := 
by 
  sorry

end toms_speed_is_correct_l151_151743


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151368

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151368


namespace relationship_among_y_points_l151_151672

theorem relationship_among_y_points :
  ∀ (y₁ y₂ y₃ : ℝ),
  (∃ (y₁ y₂ y₃ : ℝ), 
  y₁ = 3 * (-1)^2 + 6 * (-1) + 12 ∧
  y₂ = 3 * (-3)^2 + 6 * (-3) + 12 ∧
  y₃ = 3 * 2^2 + 6 * 2 + 12)
  → y₃ > y₂ ∧ y₂ > y₁ :=
by
  intros y₁ y₂ y₃ h
  cases h with y₁' h₁
  cases h₁ with y₂' h₂
  cases h₂ with y₃' [h₃₁ h₃₂ h₃₃]
  sorry

end relationship_among_y_points_l151_151672


namespace compound_interest_rate_l151_151925

theorem compound_interest_rate 
  (P : ℝ) (CI : ℝ) (t : ℝ) (n : ℕ) 
  (H_P : P = 12000) 
  (H_CI : CI = 4663.5) 
  (H_t : t = 2 + 4/12) 
  (H_n : n = 1) : 
  let A := P + CI in 
  let r := (A / P)^(1/(n * t)) - 1 in 
  r ≈ 0.1505 :=
by 
  -- the proof will go here
  sorry

end compound_interest_rate_l151_151925


namespace repeating_decimal_fraction_equivalence_l151_151225

noncomputable def repeating_decimal_to_fraction : ℚ :=
⟨24, 55, by norm_num [Rat.zero_lt], by norm_num [gcd_eq_zero_iff, gcd_eq_right_iff], by norm_num⟩

theorem repeating_decimal_fraction_equivalence : (0.4 * 10 ^ 2 + 36/99 : ℝ) = (24/55 : ℝ) :=
sorry

end repeating_decimal_fraction_equivalence_l151_151225


namespace opposite_of_neg2_l151_151173

theorem opposite_of_neg2 : ∃ y : ℤ, -2 + y = 0 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end opposite_of_neg2_l151_151173


namespace composite_square_perimeter_l151_151086

theorem composite_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let s1 := p1 / 4
  let s2 := p2 / 4
  (p1 + p2 - 2 * s1) = 120 := 
by
  -- proof goes here
  sorry

end composite_square_perimeter_l151_151086


namespace projection_a_on_b_l151_151629

-- Declare the vectors a and b
def a : Real × Real × Real := (2, 3, 1)
def b : Real × Real × Real := (1, -2, -2)

-- Define the dot product function for 3D vectors
def dot_product (u v : Real × Real × Real) : Real :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the magnitude (norm) function for 3D vectors
def magnitude (v : Real × Real × Real) : Real :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Define the projection function of a vector u onto a vector v
def projection (u v : Real × Real × Real) : Real × Real × Real :=
  let dot_uv := dot_product u v
  let mag_v_squared := (magnitude v)^2
  (dot_uv / mag_v_squared * v.1, dot_uv / mag_v_squared * v.2, dot_uv / mag_v_squared * v.3)

-- State the problem formally: proving the projection of vector a onto vector b
theorem projection_a_on_b :
  projection a b = (-2 / 3) * b := by
  -- Proof will be here
  sorry -- Skip the proof

end projection_a_on_b_l151_151629


namespace molecular_weight_l151_151255

noncomputable def molecular_weight_of_one_mole : ℕ → ℝ :=
  fun n => if n = 1 then 78 else n * 78

theorem molecular_weight (n: ℕ) (hn: n > 0) (condition: ∃ k: ℕ, k = 4 ∧ 312 = k * 78) :
  molecular_weight_of_one_mole n = 78 * n :=
by
  sorry

end molecular_weight_l151_151255


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151436

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151436


namespace solve_system_l151_151197

theorem solve_system : ∃ (x y : ℤ), 
  2010 * x - 2011 * y = 2009 ∧
  2009 * x - 2008 * y = 2010 ∧
  x = 2 ∧ y = 1 :=
by
  existsi 2
  existsi 1
  split
  repeat { sorry }

end solve_system_l151_151197


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151513

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151513


namespace triangle_inscribed_circle_area_l151_151870

noncomputable def circle_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * Real.pi)

noncomputable def triangle_area (r : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin (Real.pi / 2) + Real.sin (2 * Real.pi / 3) + Real.sin (5 * Real.pi / 6))

theorem triangle_inscribed_circle_area (a b c : ℝ) (h : a + b + c = 24) :
  ∀ (r : ℝ) (h_r : r = circle_radius 24),
  triangle_area r = 72 / Real.pi^2 * (Real.sqrt 3 + 1) :=
by
  intro r h_r
  rw [h_r, circle_radius, triangle_area]
  sorry

end triangle_inscribed_circle_area_l151_151870


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151376

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151376


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151506

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151506


namespace half_subset_even_l151_151614

def is_half_subset (A B : Finset ℕ) : Prop :=
  2 * B.sum id = A.sum id

theorem half_subset_even {n : ℕ} (A : Finset ℕ) 
  (h : A.sum id > 0) : 
  ∃ k : ℕ, 2 * k = (A.subsets.filter (is_half_subset A)).card := 
sorry

end half_subset_even_l151_151614


namespace joe_spent_on_fruits_l151_151712

theorem joe_spent_on_fruits (total_money amount_left : ℝ) (spent_on_chocolates : ℝ)
  (h1 : total_money = 450)
  (h2 : spent_on_chocolates = (1/9) * total_money)
  (h3 : amount_left = 220)
  : (total_money - spent_on_chocolates - amount_left) / total_money = 2 / 5 :=
by
  sorry

end joe_spent_on_fruits_l151_151712


namespace students_not_yet_pictured_l151_151802

def students_in_class : ℕ := 24
def students_before_lunch : ℕ := students_in_class / 3
def students_after_lunch_before_gym : ℕ := 10
def total_students_pictures_taken : ℕ := students_before_lunch + students_after_lunch_before_gym

theorem students_not_yet_pictured : total_students_pictures_taken = 18 → students_in_class - total_students_pictures_taken = 6 := by
  intros h
  rw [h]
  rfl

end students_not_yet_pictured_l151_151802


namespace composite_square_perimeter_l151_151088

theorem composite_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let s1 := p1 / 4
  let s2 := p2 / 4
  (p1 + p2 - 2 * s1) = 120 := 
by
  -- proof goes here
  sorry

end composite_square_perimeter_l151_151088


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151590

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151590


namespace find_angle_AMC_l151_151127

open Real

variables (A B C C1 C2 A2 M : ℝ²)
variables (hABC : B ≠ C)
variables (h_right : ∃ (angle_ABC : ℝ), angle_ABC = 90)
variables (h_CC1 : ∃ (BC_length : ℝ), dist B C = dist C C1)
variables (h_AC2 : ∃ (AC2_length : ℝ), dist A C2 = dist A C1)
variables (h_CA2 : ∃ (CA2_length : ℝ), dist C A2 = dist C A1)
variables (hM : M = midpoint A2 C2)

theorem find_angle_AMC : ∠ A M C = 135 := 
sorry

end find_angle_AMC_l151_151127


namespace projection_of_a_onto_b_l151_151686

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Given conditions
def mag_a_eq_5 := ‖a‖ = 5
def mag_b_eq_sqrt3 := ‖b‖ = Real.sqrt 3
def dot_ab_eq_neg2 := a ⬝ b = -2

theorem projection_of_a_onto_b :
  ∀ (a b : EuclideanSpace ℝ (Fin 2)),
    ‖a‖ = 5 →
    ‖b‖ = Real.sqrt 3 →
    a ⬝ b = -2 →
    (a ⬝ b) / ‖b‖ = - (2 * Real.sqrt 3) / 3 :=
by
  intros a b h1 h2 h3
  sorry

end projection_of_a_onto_b_l151_151686


namespace smallest_integer_is_10_l151_151807

noncomputable def smallest_integer (a b c : ℕ) : ℕ :=
  if h : (a + b + c = 90) ∧ (2 * b = 3 * a) ∧ (5 * a = 2 * c)
  then a
  else 0

theorem smallest_integer_is_10 (a b c : ℕ) (h₁ : a + b + c = 90) (h₂ : 2 * b = 3 * a) (h₃ : 5 * a = 2 * c) : 
  smallest_integer a b c = 10 :=
sorry

end smallest_integer_is_10_l151_151807


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151388

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151388


namespace arun_expected_profit_percentage_l151_151332

-- Define the quantities and prices involved
def wheat1_weight := 30 -- kg
def wheat1_price_per_kg := 11.50 -- Rs per kg
def wheat2_weight := 20 -- kg
def wheat2_price_per_kg := 14.25 -- Rs per kg
def selling_price_per_kg := 17.01 -- Rs per kg

-- Calculate total quantities
def total_wheat_weight := wheat1_weight + wheat2_weight
def total_cost := (wheat1_weight * wheat1_price_per_kg) + (wheat2_weight * wheat2_price_per_kg)
def cost_price_per_kg := total_cost / total_wheat_weight
def total_selling_price := total_wheat_weight * selling_price_per_kg
def profit := total_selling_price - total_cost
def percentage_of_profit := (profit / total_cost) * 100

-- Statement proving the expected profit percentage
theorem arun_expected_profit_percentage : percentage_of_profit ≈ 35 := by
  sorry

end arun_expected_profit_percentage_l151_151332


namespace cyclic_quadrilateral_adfe_l151_151055

variables {A B C D E F : Type*}
variables (P Q R S T : Type*)
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
variables [EuclideanGeometry D] [EuclideanGeometry E] [EuclideanGeometry F]

theorem cyclic_quadrilateral_adfe :
  ∀ {A B C D E F : Type*},
  ∀ {ABAC : A} {BCBD : B} {CECA : C},
  ∀ {ABC : EuclideanGeometry ABC},
    EuclideanGeometry.acute ABC →
    (D ∈ AB ∧ E ∈ AC) →
    (F ∈ Intersection (BE, CD)) →
    (BC ^ 2 = BD * BA + CE * CA) →
    cyclic_quadrilateral A D F E :=
begin
  intros,
  sorry,
end

end cyclic_quadrilateral_adfe_l151_151055


namespace existence_of_extrema_l151_151202

noncomputable def x (n : ℕ) : ℕ := |n - 3|
noncomputable def y (n : ℕ) : ℤ := (-1) ^ n
noncomputable def z (n : ℕ) : ℤ := (-1) ^ n - |n - 5|
noncomputable def t (n : ℕ) : ℕ := 2 ^ (|n - 2| + 1)
noncomputable def v (n : ℕ) : ℤ := (-1) ^ n * n
noncomputable def w (n : ℕ) : ℕ := |2 ^ n - n ^ 2|

theorem existence_of_extrema :
  (∃ n, x n = 0) ∧ (¬ ∃ n, ∃ m, x n < x m) ∧
  (∀ k, (y k = -1) ∨ (y k = 1)) ∧
  (¬ ∃ n, ∃ m, z n < z m) ∧ (¬ ∃ n, ∃ m, z m < z n) ∧
  (∃ n, t n = 2) ∧ (¬ ∃ n, ∃ m, t n < t m) ∧
  (¬ ∃ n, ∃ m, v n < v m) ∧ (¬ ∃ n, ∃ m, v m < v n) ∧
  (∃ n, w n = 0) ∧ (¬ ∃ n, ∃ m, w n < w m) := 
by 
  sorry

end existence_of_extrema_l151_151202


namespace enclosed_area_l151_151769

noncomputable def area_between_curves : ℝ :=
  ∫ x in 0..1, sqrt x - x^2

theorem enclosed_area :
  area_between_curves = 1 / 3 :=
by
  sorry

end enclosed_area_l151_151769


namespace injective_of_comp_injective_surjective_of_comp_surjective_l151_151107

section FunctionProperties

variables {X Y V : Type} (f : X → Y) (g : Y → V)

-- Proof for part (i) if g ∘ f is injective, then f is injective
theorem injective_of_comp_injective (h : Function.Injective (g ∘ f)) : Function.Injective f :=
  sorry

-- Proof for part (ii) if g ∘ f is surjective, then g is surjective
theorem surjective_of_comp_surjective (h : Function.Surjective (g ∘ f)) : Function.Surjective g :=
  sorry

end FunctionProperties

end injective_of_comp_injective_surjective_of_comp_surjective_l151_151107


namespace opposite_of_neg_two_l151_151183

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l151_151183


namespace limit_sequence_l151_151844

open Real

theorem limit_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs ((sqrt (2 / 7)) + (sqrt[4] (2 + n^5) - sqrt (2 * n^3 + 3)) / ((n + sin n) * sqrt (7 * n))) < ε := by
  sorry

end limit_sequence_l151_151844


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151551

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151551


namespace Frank_work_hours_l151_151936

def hoursWorked (h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday : Nat) : Nat :=
  h_monday + h_tuesday + h_wednesday + h_thursday + h_friday + h_saturday

theorem Frank_work_hours
  (h_monday : Nat := 8)
  (h_tuesday : Nat := 10)
  (h_wednesday : Nat := 7)
  (h_thursday : Nat := 9)
  (h_friday : Nat := 6)
  (h_saturday : Nat := 4) :
  hoursWorked h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday = 44 :=
by
  unfold hoursWorked
  sorry

end Frank_work_hours_l151_151936


namespace digit_square_mul_9_l151_151765

theorem digit_square_mul_9 (square : ℕ) (h : square ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  9 ∣ (734600 + square * 10 + 1) ↔ (square = 6 ∨ square = 9) :=
by
  -- The detailed proof is omitted.
  sorry

end digit_square_mul_9_l151_151765


namespace parabola_directrix_l151_151777

theorem parabola_directrix (x y : ℝ) (h : y = 8 * x^2) : y = -1 / 32 :=
sorry

end parabola_directrix_l151_151777


namespace sec_150_eq_l151_151398

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151398


namespace equilateral_triangle_side_squared_l151_151880

theorem equilateral_triangle_side_squared (A B C : ℝ × ℝ) (R : ℝ) (side_squared : ℝ) :
  (R = 3) →
  (A = (0, 3)) →
  (A.x^2 + A.y^2 = R^2) →
  (B.x^2 + B.y^2 = R^2) →
  (C.x^2 + C.y^2 = R^2) →
  (A.x = 0) →
  (A.y = R) →
  (B.y = C.y) →
  (B.x = -C.x) →
  (side_squared = (A.x - B.x)^2 + (A.y - B.y)^2) →
  side_squared = 10 :=
sorry

end equilateral_triangle_side_squared_l151_151880


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151572

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151572


namespace find_f_neg3_l151_151730

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if h : x > 0 then x * (1 - x) else -x * (1 + x)

theorem find_f_neg3 :
  is_odd_function f →
  (∀ x, x > 0 → f x = x * (1 - x)) →
  f (-3) = 6 :=
by
  intros h_odd h_condition
  sorry

end find_f_neg3_l151_151730


namespace sum_of_zeros_transformed_parabola_eq_14_l151_151783

noncomputable def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

theorem sum_of_zeros_transformed_parabola_eq_14 : 
  let p := 8
  let q := 6
  p + q = 14 :=
by
  have h1 : (x - 7)^2 = 1 -> (x - 7) = 1 \/ (x - 7) = -1 := sorry
  have h2 : transformed_parabola 8 = 0 := by apply h1; sorry
  have h3 : transformed_parabola 6 = 0 := by apply h1; sorry
  sorry

end sum_of_zeros_transformed_parabola_eq_14_l151_151783


namespace find_beta_minus_alpha_l151_151678

noncomputable def a (α : ℝ) : ℝ × ℝ := (real.sqrt 2 * real.cos α, real.sqrt 2 * real.sin α)
noncomputable def b (β : ℝ) : ℝ × ℝ := (2 * real.cos β, 2 * real.sin β)
noncomputable def sub_ab (α β : ℝ) : ℝ × ℝ :=
  let a_val := a α
  let b_val := b β
  (b_val.1 - a_val.1, b_val.2 - a_val.2)

def perp (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0

theorem find_beta_minus_alpha (α β : ℝ) (h₁ : real.pi / 6 ≤ α) (h₂ : α < real.pi / 2)
  (h₃ : real.pi / 2 < β) (h4 : β ≤ 5 * real.pi / 6) 
  (h5 : perp (a α) (sub_ab α β)) :
  β - α = real.pi / 4 :=
sorry

end find_beta_minus_alpha_l151_151678


namespace points_per_enemy_l151_151051

-- Definitions: total enemies, enemies not destroyed, points earned
def total_enemies : ℕ := 11
def enemies_not_destroyed : ℕ := 3
def points_earned : ℕ := 72

-- To prove: points per enemy
theorem points_per_enemy : points_earned / (total_enemies - enemies_not_destroyed) = 9 := 
by
  sorry

end points_per_enemy_l151_151051


namespace solution_exists_iff_divisor_form_l151_151097

theorem solution_exists_iff_divisor_form (n : ℕ) (hn_pos : 0 < n) (hn_odd : n % 2 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 4 * x * y = n * (x + y)) ↔
    (∃ k : ℕ, n % (4 * k + 3) = 0) :=
by
  sorry

end solution_exists_iff_divisor_form_l151_151097


namespace school_fee_correct_l151_151739

-- Definitions
def mother_fifty_bills : ℕ := 1
def mother_twenty_bills : ℕ := 2
def mother_ten_bills : ℕ := 3

def father_fifty_bills : ℕ := 4
def father_twenty_bills : ℕ := 1
def father_ten_bills : ℕ := 1

def total_fifty_bills : ℕ := mother_fifty_bills + father_fifty_bills
def total_twenty_bills : ℕ := mother_twenty_bills + father_twenty_bills
def total_ten_bills : ℕ := mother_ten_bills + father_ten_bills

def value_fifty_bills : ℕ := 50 * total_fifty_bills
def value_twenty_bills : ℕ := 20 * total_twenty_bills
def value_ten_bills : ℕ := 10 * total_ten_bills

-- Theorem
theorem school_fee_correct :
  value_fifty_bills + value_twenty_bills + value_ten_bills = 350 :=
by
  sorry

end school_fee_correct_l151_151739


namespace diamond_problem_l151_151622

def diamond (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

theorem diamond_problem (a b c d : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = -24) (h4 : d = -7) :
  diamond (diamond a b) (diamond c d) = 25 * Real.sqrt 2 :=
by
  rw [h1, h2, h3, h4]
  -- Check definition of the diamond
  sorry

end diamond_problem_l151_151622


namespace log_multiplication_l151_151339

theorem log_multiplication : log 2 9 * log 3 8 = 6 := by
  sorry

end log_multiplication_l151_151339


namespace coefficient_of_x4_in_expansion_l151_151243

noncomputable def problem_statement : ℕ :=
  let n := 8
  let a := 2
  let b := 3
  let k := 4
  binomial n k * (b ^ k) * (a ^ (n - k))

theorem coefficient_of_x4_in_expansion :
  problem_statement = 90720 :=
by
  sorry

end coefficient_of_x4_in_expansion_l151_151243


namespace rearrange_circles_sums13_l151_151795

def isSum13 (a b c d x y z w : ℕ) : Prop :=
  (a + 4 + b = 13) ∧ (b + 2 + d = 13) ∧ (d + 1 + c = 13) ∧ (c + 3 + a = 13)

theorem rearrange_circles_sums13 : 
  ∃ (a b c d x y z w : ℕ), 
  a = 4 ∧ b = 5 ∧ c = 6 ∧ d = 6 ∧ 
  a + b = 9 ∧ b + z = 11 ∧ z + c = 12 ∧ c + a = 10 ∧ 
  isSum13 a b c d x y z w :=
by {
  sorry
}

end rearrange_circles_sums13_l151_151795


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151427

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151427


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151507

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151507


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151435

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151435


namespace triangle_side_a_l151_151050

noncomputable theory

open Real

theorem triangle_side_a {A B C : ℝ} {a b c: ℝ} (h₀ : c = sqrt 2) (h₁ : b = sqrt 6) (h₂ : B = 120) :
  a = sqrt 2 :=
sorry

end triangle_side_a_l151_151050


namespace integer_sqrt_225_minus_cbrt_x_count_l151_151616

theorem integer_sqrt_225_minus_cbrt_x_count :
  (∃ n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ ∃ x : ℝ, x ≥ 0 ∧ sqrt (225 - cbrt x) = n) → 16 :=
by
  sorry

end integer_sqrt_225_minus_cbrt_x_count_l151_151616


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151543

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151543


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151429

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151429


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151588

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151588


namespace area_of_triangle_is_2_l151_151695

-- Define the conditions of the problem
variable (a b c : ℝ)
variable (A B C : ℝ)  -- Angles in radians

-- Conditions for the triangle ABC
variable (sin_A : ℝ) (sin_C : ℝ)
variable (c2sinA_eq_5sinC : c^2 * sin_A = 5 * sin_C)
variable (a_plus_c_squared_eq_16_plus_b_squared : (a + c)^2 = 16 + b^2)
variable (ac_eq_5 : a * c = 5)
variable (cos_B : ℝ)
variable (sin_B : ℝ)

-- Sine and Cosine law results
variable (cos_B_def : cos_B = (a^2 + c^2 - b^2) / (2 * a * c))
variable (sin_B_def : sin_B = Real.sqrt (1 - cos_B^2))

-- Area of the triangle
noncomputable def area_triangle_ABC := (1/2) * a * c * sin_B

-- Theorem to prove the area
theorem area_of_triangle_is_2 :
  area_triangle_ABC a c sin_B = 2 :=
by
  rw [area_triangle_ABC]
  sorry

end area_of_triangle_is_2_l151_151695


namespace tangent_line_of_curve_at_point_l151_151927

noncomputable def tangent_line_equation (f : ℝ → ℝ) (f' : ℝ → ℝ) (x₀ y₀ : ℝ) := 
  ∀ x y : ℝ, y - y₀ = f' x₀ * (x - x₀)

theorem tangent_line_of_curve_at_point : 
  let f := λ x : ℝ, x^2 + 3 * x + 1
  let f' := λ x : ℝ, 2 * x + 3
  tangent_line_equation f f' 0 1 → 3 * x - y + 1 = 0 :=
by
  sorry

end tangent_line_of_curve_at_point_l151_151927


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151410

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151410


namespace vector_proj_problems_l151_151677

noncomputable theory
open_locale big_operators
open Finset

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def vector_sub (a b : Point3D) : Point3D :=
⟨a.x - b.x, a.y - b.y, a.z - b.z⟩

def dot_product (v1 v2 : Point3D) : ℝ :=
(v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)

def magnitude (v : Point3D) : ℝ :=
real.sqrt ((v.x ^ 2) + (v.y ^ 2) + (v.z ^ 2))

def cos_angle (v1 v2 : Point3D) : ℝ :=
(dot_product v1 v2) / ((magnitude v1) * (magnitude v2))

def distance_point_to_line (a b c : Point3D) : ℝ :=
let ab := vector_sub b a,
    ac := vector_sub c a,
    bc := vector_sub c b,
    height := magnitude ac * real.sqrt (1 - (cos_angle ac bc) ^ 2) in
    height / magnitude bc

theorem vector_proj_problems :
  let A := Point3D.mk (-1) 2 1,
      B := Point3D.mk 1 3 1,
      C := Point3D.mk (-2) 4 2 in
  (dot_product (vector_sub B A) (vector_sub C A) = 0) ∧
  (dot_product (Point3D.mk 1 (-2) (-5)) (vector_sub C A) ≠ 0) ∧
  (cos_angle (vector_sub C A) (vector_sub C B) = real.sqrt 66 / 11) ∧
  (distance_point_to_line A B C = real.sqrt 330 / 11) :=
by
  sorry

end vector_proj_problems_l151_151677


namespace max_perpendicular_intersections_l151_151635

theorem max_perpendicular_intersections
  (P : Fin 5 → ℝ × ℝ)
  (h1 : ∀ i j, i ≠ j → ¬(∃ k l m n: Fin 5, (P i, P j) = (P k, P l) ∨ (P i, P j) = (P m, P n) ∧ k ≠ m ∧ l ≠ n))
  (h2: ∀ i j, i ≠ j → ¬(P i.1 = P j.1 ∧ P i.2 = P j.2)) :
  ∃ n, n = 25 := 
begin 
  sorry
end

end max_perpendicular_intersections_l151_151635


namespace solution_l151_151031

noncomputable def condition {x : ℝ} : Prop := |2 - x| = 2 + |x|

theorem solution (x : ℝ) (h : condition x) : |2 - x| = 2 - x :=
sorry

end solution_l151_151031


namespace sequence_a_sequence_b_sequence_c_l151_151987

noncomputable def a (n : ℕ) : ℕ := n + n⁻¹
noncomputable def b (n : ℕ) : ℕ := n
noncomputable def c (n : ℕ) : ℕ := if n = 1 then 5 else 3 * n + 1

theorem sequence_a (n : ℕ) : (a n - b n) * b n = 1 ∧ 1 / a n = b n / (n^2 + 1) :=
sorry

theorem sequence_b (n : ℕ) : (a n = n + 1 / n) ∧ (b n = n) :=
sorry

theorem sequence_c (n : ℕ) : 
  (∀ n, c n = if n = 1 then 5 else 3 * n + 1) →
  (S n = 1/2 * (3 * n^2 + 5 * n + 2)) :=
sorry

end sequence_a_sequence_b_sequence_c_l151_151987


namespace average_age_is_correct_l151_151868

-- Define the conditions
def num_men : ℕ := 6
def num_women : ℕ := 9
def average_age_men : ℕ := 57
def average_age_women : ℕ := 52
def total_age_men : ℕ := num_men * average_age_men
def total_age_women : ℕ := num_women * average_age_women
def total_age : ℕ := total_age_men + total_age_women
def total_people : ℕ := num_men + num_women
def average_age_group : ℕ := total_age / total_people

-- The proof will require showing average_age_group is 54, left as sorry.
theorem average_age_is_correct : average_age_group = 54 := sorry

end average_age_is_correct_l151_151868


namespace sqrt_equality_l151_151901

theorem sqrt_equality :
  Real.sqrt ((18: ℝ) * (17: ℝ) * (16: ℝ) * (15: ℝ) + 1) = 271 :=
by
  sorry

end sqrt_equality_l151_151901


namespace negative_reciprocal_slopes_minimum_area_conjecture_minimum_area_l151_151010

-- Definition of the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Point M and its symmetric point N
def M := (1 : ℝ, 0 : ℝ)
def N := (-1 : ℝ, 0 : ℝ)

-- Line l passing through M and intersecting the parabola at points A and B
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Definition of points A and B on the parabola and satisfying line_l equation
def point_A (k x1 y1 : ℝ) : Prop := line_l k x1 y1 ∧ parabola x1 y1
def point_B (k x2 y2 : ℝ) : Prop := line_l k x2 y2 ∧ parabola x2 y2

-- Proof of the first question
theorem negative_reciprocal_slopes (k x1 y1 x2 y2 : ℝ) 
  (hA : point_A k x1 y1) (hB : point_B k x2 y2) : 
  (y1 / (x1 + 1)) = -(y2 / (x2 + 1)) := 
sorry

-- Proof of the second question
theorem minimum_area (k x1 y1 x2 y2 : ℝ) 
  (hA : point_A k x1 y1) (hB : point_B k x2 y2) 
  (h_symmetric : (x1 + x2) = 2) 
  (h_product : x1 * x2 = 1) : 
  abs (y1 - y2) >= 4 := 
sorry

-- Conjecture for the third question
theorem conjecture_minimum_area (m k x1 y1 x2 y2 : ℝ) 
  (hm : m > 0) (hm1 : m ≠ 1) 
  (hA : point_A k x1 y1) (hB : point_B k x2 y2) 
  (h_M : (1/x1, 0)) (h_symmetric_m : x1 = 2 - x2) : 
  abs (y1 - y2) >= 4 * m * sqrt m := 
sorry

end negative_reciprocal_slopes_minimum_area_conjecture_minimum_area_l151_151010


namespace prob_parabola_line_no_intersection_l151_151102

theorem prob_parabola_line_no_intersection :
  let P (x : ℝ) := x^2
  let Q := (10, 36)
  ∃ r s : ℝ, (∀ m : ℝ, r < m ∧ m < s → ¬ ∃ x : ℝ, P x = m * x + b)
  where b = Q.2 - m * Q.1
  → r + s = 40 :=
by
  intros P Q r s hm 
  sorry

end prob_parabola_line_no_intersection_l151_151102


namespace student_19_in_sample_l151_151322

-- Definitions based on conditions
def total_students := 52
def sample_size := 4
def sampling_interval := 13

def selected_students := [6, 32, 45]

-- The theorem to prove
theorem student_19_in_sample : 19 ∈ selected_students ∨ ∃ k : ℕ, 13 * k + 6 = 19 :=
by
  sorry

end student_19_in_sample_l151_151322


namespace sec_150_l151_151488

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151488


namespace sum_d_k_squared_eq_385_l151_151908

def d (k : ℕ) : ℝ := k + (1 / (3 * k + (1 / (3 * k + (1 / (3 * k + ...)))))

theorem sum_d_k_squared_eq_385 : ∑ k in Finset.range 10, (d k)^2 = 385 := 
by sorry

end sum_d_k_squared_eq_385_l151_151908


namespace part1_part2_part3_l151_151003

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

theorem part1 {a : ℝ} (h : 0 < a) :
  Deriv (λ x, f a x) 2 = 2 → a = 4 := sorry

theorem part2 {a : ℝ} (h : 0 < a) (x : ℝ) (hx : 0 < x) :
  f(a, x) ≥ a * (1 - (1/x)) := sorry

theorem part3 {a : ℝ} (h : 0 < a) :
  (∀ x, 1 < x ∧ x < Real.exp 1 → f(a, x) / (x - 1) > 1) → a ≥ Real.exp 1 - 1 := sorry

end part1_part2_part3_l151_151003


namespace range_of_x0_l151_151946

-- Define the circle equation
def Circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line equation
def Line (x y : ℝ) : Prop := x + y = 6

-- Define the main theorem
theorem range_of_x0 (x0 : ℝ) :
  (∃ y0, Line x0 y0 ∧ 
            (∃ Bx By Cx Cy, 
                Circle Bx By ∧ Circle Cx Cy ∧ 
                (let A := (x0, 6 - x0) 
                in (angleBAC A (Bx, By) (Cx, Cy) = 60) ))) 
  → 1 ≤ x0 ∧ x0 ≤ 5 :=
begin
  sorry,
end

end range_of_x0_l151_151946


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151424

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151424


namespace points_concyclic_l151_151949

-- Defining the points and the line
noncomputable theory

variables {A B M N : Point} {e : Line}

-- Defining the conditions
def on_one_side (A B : Point) (e : Line) : Prop := sorry  -- Assume definition of being on one side
def minimizes_AM_MB (A B M : Point) (e : Line) : Prop := sorry  -- Assume definition for minimizing AM + MB
def equidistant_AN_BN (A B N : Point) (e : Line) : Prop := sorry  -- Assume definition for AN=BN

-- Problem statement: points A, B, M, N are concyclic under given conditions
theorem points_concyclic (A B M N : Point) (e : Line)
  (h1 : on_one_side A B e)
  (h2 : minimizes_AM_MB A B M e)
  (h3 : equidistant_AN_BN A B N e) :
  concyclic A B M N :=
sorry

end points_concyclic_l151_151949


namespace man_son_age_ratio_l151_151305

-- Define the present age of the son
def son_age_present : ℕ := 22

-- Define the present age of the man based on the son's age
def man_age_present : ℕ := son_age_present + 24

-- Define the son's age in two years
def son_age_future : ℕ := son_age_present + 2

-- Define the man's age in two years
def man_age_future : ℕ := man_age_present + 2

-- Prove the ratio of the man's age to the son's age in two years is 2:1
theorem man_son_age_ratio : man_age_future / son_age_future = 2 := by
  sorry

end man_son_age_ratio_l151_151305


namespace find_m_l151_151640

theorem find_m (m : ℝ) (α : ℝ) (h1 : P(m, 2) is_on_terminal_side_of α) (h2 : sin α = 1/3) : 
  m = 4 * real.sqrt 2 ∨ m = -4 * real.sqrt 2 := 
sorry

end find_m_l151_151640


namespace find_f_expression_find_a_range_l151_151950

-- Define the function f(x) that satisfies f(f(x)) = x + 4
def f (x : ℝ) : ℝ := x + 2

-- Define the function g(x) = (1 - a) * x² - x
def g (a x : ℝ) : ℝ := (1 - a) * x * x - x

-- Prove the expression for f(x)
theorem find_f_expression : ∀ x : ℝ, f(f(x)) = x + 4 :=
by {
  intro x,
  sorry
}

-- Prove the range of values for a
theorem find_a_range (h : ∀ x1 ∈ Set.Icc (1/4 : ℝ) 4, ∃ x2 ∈ Set.Icc (-3 : ℝ) (1/3), g a x1 ≥ f x2) :
  a ≤ (3/4) :=
by {
  sorry
}

end find_f_expression_find_a_range_l151_151950


namespace max_sum_a_b_l151_151027

theorem max_sum_a_b (a b : ℝ) (h : a^2 - a*b + b^2 = 1) : a + b ≤ 2 := 
by sorry

end max_sum_a_b_l151_151027


namespace right_handed_players_count_l151_151747

noncomputable def total_players : ℕ := 70
noncomputable def throwers : ℕ := 28
noncomputable def non_throwers : ℕ := total_players - throwers
noncomputable def left_handed_non_throwers : ℕ := non_throwers / 3
noncomputable def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
noncomputable def total_right_handed_players : ℕ := throwers + right_handed_non_throwers

theorem right_handed_players_count :
  total_right_handed_players = 56 :=
begin
  sorry
end

end right_handed_players_count_l151_151747


namespace f_le_g_l151_151632

noncomputable def f (n : ℕ) : ℚ := 
  if n = 0 then 0 else (finset.sum (finset.range n) (λ k, 1 / (k + 1) ^ 2))

noncomputable def g (n : ℕ) : ℚ := 
  if n = 0 then 0 else 1 / 2 * (3 - 1 / n ^ 2)

theorem f_le_g : ∀ n : ℕ, n > 0 → (f n) ≤ (g n) :=
by
  sorry

end f_le_g_l151_151632


namespace infinite_series_sum_of_inscribed_circles_l151_151288

theorem infinite_series_sum_of_inscribed_circles (r : ℝ) (h : 0 < r) :
  let A₁ := π * r^2 in
  let s := r * real.sqrt 2 in
  let A₂ := π * (r * real.sqrt 2 / 2)^2 in
  let T_n := ∑ i in finset.range n, A₁ * (1 / 2)^i in
  ∃ L, has_sum (λ i, A₁ * (1 / 2)^i) L ∧ L = 2 * π * r^2 :=
begin
  let A₁ := π * r^2,
  let s := r * real.sqrt 2,
  let r₂ := r * real.sqrt 2 / 2,
  let A₂ := π * r₂^2,
  let T_n := ∑ i in finset.range n, A₁ * (1 / 2)^i,
  use 2 * π * r^2,
  split,
  { apply has_sum_geometric,
    { norm_num,
      exact one_div_pos.mpr two_pos },
    { norm_num } },
  { norm_num }
end

end infinite_series_sum_of_inscribed_circles_l151_151288


namespace resulting_perimeter_l151_151081

theorem resulting_perimeter (p1 p2 : ℕ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let a := p1 / 4 in
  let b := p2 / 4 in
  p1 + p2 - 2 * a = 120 :=
by
  sorry

end resulting_perimeter_l151_151081


namespace gcd_multiple_less_than_120_l151_151250

theorem gcd_multiple_less_than_120 (n : ℕ) (h1 : n < 120) (h2 : n % 10 = 0) (h3 : n % 15 = 0) : n ≤ 90 :=
by {
  sorry
}

end gcd_multiple_less_than_120_l151_151250


namespace correct_operation_only_A_l151_151261

theorem correct_operation_only_A (a : ℝ) : 
  (((2 * a^2)^3 = 8 * a^6) ∧ ¬((3 * a)^2 = 6 * a^2) ∧ ¬(a^5 + a^5 = 2 * a^{10}) ∧ ¬(3 * a^2 * a^3 = 3 * a^6)) :=
by sorry

end correct_operation_only_A_l151_151261


namespace germs_per_dish_l151_151707

theorem germs_per_dish (total_germs : ℕ) (total_dishes : ℕ) 
  (h_germs : total_germs = 0.037 * 10^5) 
  (h_dishes : total_dishes = 74000 * 10^(-3)) : 
  (total_germs / total_dishes = 50) := 
by
  sorry

end germs_per_dish_l151_151707


namespace incorrect_statements_l151_151008

-- Define the line equation
def line_eq (m x y : ℝ) : Prop :=
  (m - 2) * x + (m + 1) * y - 3 = 0

-- Statement B: Incorrect slope calculation for m = 1/2
def statement_b_incorrect (m : ℝ) : Prop :=
  m = 1/2 → slope_of_line l ≠ 3 * π / 4

-- Placeholder for slope calculation function
def slope_of_line (l : ℝ × ℝ → Prop) : ℝ := sorry

-- Statement C: Incorrect symmetric line equation for m = 1
def statement_c_incorrect (m : ℝ) : Prop :=
  m = 1 → symmetric_line_eq (line_eq m) ≠ x + 2 * y - 3 = 0

-- Placeholder for symmetric line equation function
def symmetric_line_eq (l : ℝ × ℝ → Prop) : ℝ × ℝ → Prop := sorry

-- Statement D: Incorrect maximum distance from point P(2, 4) to line l
def statement_d_incorrect (m : ℝ) : Prop :=
  ∀ m, max_distance_to_point (line_eq m) (2, 4) ≠ 3 * sqrt 2

-- Placeholder for maximum distance calculation function
def max_distance_to_point (l : ℝ × ℝ → Prop) (p : ℝ × ℝ) : ℝ := sorry

-- Final theorem statement
theorem incorrect_statements (m : ℝ) :
  statement_b_incorrect m ∧ statement_c_incorrect m ∧ statement_d_incorrect m :=
by
  sorry

end incorrect_statements_l151_151008


namespace minimal_value_abs_a_minus_b_l151_151109

theorem minimal_value_abs_a_minus_b (a b : ℕ) (h : a * b - 4 * a + 7 * b = 679) : 
  |a - b| = 37 :=
sorry

end minimal_value_abs_a_minus_b_l151_151109


namespace simplify_and_evaluate_expr_l151_151754

noncomputable def a : ℝ := 3 + Real.sqrt 5
noncomputable def b : ℝ := 3 - Real.sqrt 5

theorem simplify_and_evaluate_expr : 
  (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end simplify_and_evaluate_expr_l151_151754


namespace ways_to_paint_2_to_10_is_96_l151_151993

open Function

-- Definitions for the problem conditions

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d => d ∣ n ∧ d ≠ n) (List.range (n + 1))

def consecutive_different_color (color : ℕ → ℕ) : Prop :=
  ∀ n, 2 ≤ n → n < 10 → color n ≠ color (n + 1)

def different_color_from_divisors (color : ℕ → ℕ) : Prop :=
  ∀ n, 2 ≤ n → n ≤ 10 → ∀ d, d ∈ proper_divisors n → color n ≠ color d

-- Main theorem

theorem ways_to_paint_2_to_10_is_96 :
  ∃ (color : ℕ → ℕ), (∀ n, 2 ≤ n → n ≤ 10 → color n ∈ {1, 2, 3}) ∧
    consecutive_different_color color ∧ different_color_from_divisors color ∧
    (cardinal.mk (sigma (λ color_n: (ℕ → ℕ), (∀ n, 2 ≤ n → n ≤ 10 → color_n n ∈ {1, 2, 3}) ∧
      consecutive_different_color color_n ∧ different_color_from_divisors color_n))) = 96 :=
sorry

end ways_to_paint_2_to_10_is_96_l151_151993


namespace greatest_common_multiple_of_10_and_15_lt_120_l151_151253

theorem greatest_common_multiple_of_10_and_15_lt_120 : 
  ∃ (m : ℕ), lcm 10 15 = 30 ∧ m ∈ {i | i < 120 ∧ ∃ (k : ℕ), i = k * 30} ∧ m = 90 := 
sorry

end greatest_common_multiple_of_10_and_15_lt_120_l151_151253


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151568

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151568


namespace symmetric_plane_midpoint_or_symmetry_plane_l151_151906

theorem symmetric_plane_midpoint_or_symmetry_plane
    (P : Type) [convex_polyhedron P] (H : ∃ (plane : P), symmetric_wrt_plane plane) :
  ∃ (plane : P), (is_midpoint_plane plane P) ∨ (is_symmetry_plane_for_polyhedral_angle plane P) :=
by
sory

end symmetric_plane_midpoint_or_symmetry_plane_l151_151906


namespace eggs_ordered_l151_151218

theorem eggs_ordered (E : ℕ) (h1 : E > 0) (h_crepes : E * 1 / 4 = E / 4)
                     (h_cupcakes : 2 / 3 * (3 / 4 * E) = 1 / 2 * E)
                     (h_left : (3 / 4 * E - 2 / 3 * (3 / 4 * E)) = 9) :
  E = 18 := by
  sorry

end eggs_ordered_l151_151218


namespace zacks_friends_l151_151838

theorem zacks_friends (initial_marbles : ℕ) (marbles_kept : ℕ) (marbles_per_friend : ℕ) 
  (h_initial : initial_marbles = 65) (h_kept : marbles_kept = 5) 
  (h_per_friend : marbles_per_friend = 20) : (initial_marbles - marbles_kept) / marbles_per_friend = 3 :=
by
  sorry

end zacks_friends_l151_151838


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151562

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151562


namespace motorcycle_materials_cost_l151_151854

theorem motorcycle_materials_cost 
  (car_material_cost : ℕ) (cars_per_month : ℕ) (car_sale_price : ℕ)
  (motorcycles_per_month : ℕ) (motorcycle_sale_price : ℕ)
  (additional_profit : ℕ) :
  car_material_cost = 100 →
  cars_per_month = 4 →
  car_sale_price = 50 →
  motorcycles_per_month = 8 →
  motorcycle_sale_price = 50 →
  additional_profit = 50 →
  car_material_cost + additional_profit = 250 := by
  sorry

end motorcycle_materials_cost_l151_151854


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151446

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151446


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151592

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151592


namespace smallest_solution_l151_151610

def equation (x : ℝ) : ℝ := (3*x)/(x-3) + (3*x^2 - 27)/x

theorem smallest_solution : ∃ x : ℝ, equation x = 14 ∧ x = (7 - Real.sqrt 76) / 3 := 
by {
  -- proof steps go here
  sorry
}

end smallest_solution_l151_151610


namespace power_subtraction_l151_151917

theorem power_subtraction (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : (a^b)^b - (b^a)^a = 42792577 :=
by
  rw [h_a, h_b]
  have h1 : (3^4)^4 = 43046721 := by sorry
  have h2 : (4^3)^3 = 262144 := by sorry
  rw [h1, h2]
  simp

end power_subtraction_l151_151917


namespace difference_between_two_numbers_l151_151796

theorem difference_between_two_numbers : 
  ∃ (a b : ℕ),
    (a + b = 21780) ∧
    (a % 5 = 0) ∧
    ((a / 10) = b) ∧
    (a - b = 17825) :=
sorry

end difference_between_two_numbers_l151_151796


namespace coefficient_of_x4_in_expansion_l151_151236

theorem coefficient_of_x4_in_expansion (x : ℤ) :
  let a := 3
  let b := 2
  let n := 8
  let k := 4
  (finset.sum (finset.range (n + 1)) (λ r, binomial n r * a^r * b^(n-r) * x^r) = 
  ∑ r in finset.range (n + 1), binomial n r * a^r * b^(n - r) * x^r)

  ∑ r in finset.range (n + 1), 
    if r = k then 
      binomial n r * a^r * b^(n-r)
    else 
      0 = 90720
:= 
by
  sorry

end coefficient_of_x4_in_expansion_l151_151236


namespace perpendicular_condition_l151_151984

def vector := (ℝ × ℝ)

def a : vector := (1, 2)
def b : vector := (1, 0)
def c : vector := (4, -3)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def lambda_val : ℝ := 1 / 2

theorem perpendicular_condition : ∀ (λ : ℝ), dot_product (a.1 + λ * b.1, a.2 + λ * b.2) c = 0 → λ = lambda_val :=
  by
    assume λ h,
    sorry

end perpendicular_condition_l151_151984


namespace min_S6_minus_S4_l151_151725

variable {a₁ a₂ q : ℝ} (h1 : q > 1) (h2 : (q^2 - 1) * (a₁ + a₂) = 3)

theorem min_S6_minus_S4 : 
  ∃ (a₁ a₂ q : ℝ), q > 1 ∧ (q^2 - 1) * (a₁ + a₂) = 3 ∧ (q^4 * (a₁ + a₂) - (a₁ + a₂ + a₂ * q + a₂ * q^2) = 12) := sorry

end min_S6_minus_S4_l151_151725


namespace ratio_of_areas_l151_151162

theorem ratio_of_areas (d1 d2 : ℝ) (h1 : d1 = 4 * Real.sqrt 2) (h2 : d2 = 8) :
  let s1 := d1 / Real.sqrt 2,
      s2 := d2 / Real.sqrt 2,
      A1 := s1^2,
      A2 := s2^2
  in A2 / A1 = 2 :=
by
  sorry

end ratio_of_areas_l151_151162


namespace time_relationship_l151_151793

-- Definitions
def V_b : ℝ := 20
def V_s : ℝ := 6
def D_down : ℝ := 26
def D_up : ℝ := 14

-- Calculations
def V_down : ℝ := V_b + V_s
def V_up : ℝ := V_b - V_s
def T_down : ℝ := D_down / V_down
def T_up : ℝ := D_up / V_up

-- Theorem Statement
theorem time_relationship : T_down = T_up := by
  -- Proof here, which will show that T_down = T_up
  sorry

end time_relationship_l151_151793


namespace king_middle_school_teachers_l151_151713

theorem king_middle_school_teachers 
    (students : ℕ)
    (classes_per_student : ℕ)
    (normal_class_size : ℕ)
    (special_classes : ℕ)
    (special_class_size : ℕ)
    (classes_per_teacher : ℕ)
    (H1 : students = 1500)
    (H2 : classes_per_student = 5)
    (H3 : normal_class_size = 30)
    (H4 : special_classes = 10)
    (H5 : special_class_size = 15)
    (H6 : classes_per_teacher = 3) : 
    ∃ teachers : ℕ, teachers = 85 :=
by
  sorry

end king_middle_school_teachers_l151_151713


namespace six_digit_numbers_same_parity_l151_151025

theorem six_digit_numbers_same_parity : 
  let evens := [2, 4, 6, 8]; let odds := [1, 3, 5, 7, 9] in
  (4 * 5^5) + (5 * 5^5) = 9 * 5^5 :=
by 
  sorry

end six_digit_numbers_same_parity_l151_151025


namespace product_identity_l151_151354

variable (x y : ℝ)

theorem product_identity :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end product_identity_l151_151354


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151586

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151586


namespace coefficient_x4_in_expansion_l151_151229

theorem coefficient_x4_in_expansion : 
  (∃ (c : ℤ), c = (choose 8 4) * 3^4 * 2^4 ∧ c = 90720) := 
by
  use (choose 8 4) * 3^4 * 2^4
  split
  sorry
  sorry

end coefficient_x4_in_expansion_l151_151229


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151502

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151502


namespace prank_combinations_l151_151333

theorem prank_combinations : 
  let monday := 1 in
  let tuesday := 2 in
  let wednesday := 3 in
  let thursday := 2 + 3 + 1 * 3 in
  let friday := 1 in
  monday * tuesday * wednesday * thursday * friday = 36 :=
by sorry

end prank_combinations_l151_151333


namespace solution_set_interval_l151_151729

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x > f y

noncomputable def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y - 1

theorem solution_set_interval (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (h_equation : satisfies_equation f) :
  set_of (λ x, 0 < x ∧ f (Real.log x / Real.log 2 - 1) > 1) = {x : ℝ | 2 < x ∧ x < 4} :=
sorry

end solution_set_interval_l151_151729


namespace inequality_holds_l151_151962

noncomputable theory

open Real Nat

theorem inequality_holds (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : 1 / a + 1 / b = 1) (n : ℕ) :
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry

end inequality_holds_l151_151962


namespace tens_digit_equiv_cycle_of_6_l151_151830

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_equiv_cycle_of_6 : tens_digit (6 ^ 22) = 3 := 
by {
  -- Define the cases based on the given problem
  have h1 : tens_digit (6 ^ 1) = 0 := by norm_num,
  have h2 : tens_digit (6 ^ 2) = 3 := by norm_num,
  have h3 : tens_digit (6 ^ 3) = 1 := by norm_num,
  have h4 : tens_digit (6 ^ 4) = 9 := by norm_num,
  have h5 : tens_digit (6 ^ 5) = 7 := by norm_num,
  have h6 : tens_digit (6 ^ 6) = 6 := by norm_num,
  
  -- Notice and define the cycle
  let digit_cycle := [0, 3, 1, 9, 7, 6],
  let k := 22,
  let cycle_length := 5,
  
  -- Determine the remainder when k divided by cycle_length
  let r := k % cycle_length,
  
  -- Assign correct digit in the cycle
  cases r with
  | 0 => exact h1
  | 1 => exact h2
  | 2 => exact h2
  | 3 => exact h3
  | 4 => exact h4
  | 5 => exact h5
]

end tens_digit_equiv_cycle_of_6_l151_151830


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151384

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151384


namespace inequality_proof_l151_151094

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  a^a * b^b + a^b * b^a ≤ 1 :=
  sorry

end inequality_proof_l151_151094


namespace count_1988_in_S_1988_eq_phi_1988_l151_151788

def seq (n : ℕ) : List ℕ :=
  match n with
  | 1 => [1, 1]
  | 2 => [1, 2, 1]
  | m + 3 =>
    let prevSeq := seq (m + 2)
    List.bind (List.zip prevSeq (List.tail prevSeq)) (fun p => [p.1, p.1 + p.2]) ++ [List.last prevSeq 1]

def count_occurrences (n a : ℕ) : ℕ :=
  List.count a (seq n)

def phi_1988 : ℕ :=
  Nat.totient 1988

theorem count_1988_in_S_1988_eq_phi_1988 :
  count_occurrences 1988 1988 = phi_1988 := by
  sorry

end count_1988_in_S_1988_eq_phi_1988_l151_151788


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151558

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151558


namespace assignments_needed_for_40_points_l151_151857

theorem assignments_needed_for_40_points : 
  let points1 := 7 * 3 in
  let points2 := 7 * 4 in
  let points3 := 7 * 5 in
  let points4 := 7 * 6 in
  let points5 := 7 * 7 in
  let points6 := 5 * 8 in
  points1 + points2 + points3 + points4 + points5 + points6 = 215 :=
by
  let points1 := 7 * 3
  let points2 := 7 * 4
  let points3 := 7 * 5
  let points4 := 7 * 6
  let points5 := 7 * 7
  let points6 := 5 * 8
  show points1 + points2 + points3 + points4 + points5 + points6 = 215
  sorry

end assignments_needed_for_40_points_l151_151857


namespace empty_to_occupied_ratio_of_spheres_in_cylinder_package_l151_151885

theorem empty_to_occupied_ratio_of_spheres_in_cylinder_package
  (R : ℝ) 
  (volume_sphere : ℝ)
  (volume_cylinder : ℝ)
  (sphere_occupies_fraction : ∀ R : ℝ, volume_sphere = (2 / 3) * volume_cylinder) 
  (num_spheres : ℕ) 
  (h_num_spheres : num_spheres = 5) :
  (num_spheres : ℝ) * volume_sphere = (5 * (2 / 3) * π * R^3) → 
  volume_sphere = (4 / 3) * π * R^3 → 
  volume_cylinder = 2 * π * R^3 → 
  (volume_cylinder - volume_sphere) / volume_sphere = 1 / 2 := by 
  sorry

end empty_to_occupied_ratio_of_spheres_in_cylinder_package_l151_151885


namespace monotonic_intervals_b_zero_range_of_b_l151_151975

def f (a b x : ℝ) : ℝ := 2 * a * x + b * x - 1 - 2 * log x

theorem monotonic_intervals_b_zero (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, deriv (f a 0) x < 0) ∧ 
  (a > 0 → ∀ x > 0, (x < 1 / a → deriv (f a 0) x < 0) ∧ (x > 1 / a → deriv (f a 0) x > 0)) :=
sorry

theorem range_of_b : ∀ a ∈ set.Icc 1 3, ∀ x ∈ set.Ioi 0, f a 0 x ≥ 2 * 0 * x - 3 → ∀ b, b ∈ set.Iic (2 - 2 / real.exp 2) :=
sorry

end monotonic_intervals_b_zero_range_of_b_l151_151975


namespace odd_non_empty_subsets_count_l151_151022

theorem odd_non_empty_subsets_count :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let oddNumbers := {n | n ∈ S ∧ (n % 2 = 1)}
  card ({T | T ⊆ oddNumbers ∧ T ≠ ∅}) = 31 :=
by
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let oddNumbers := {n | n ∈ S ∧ (n % 2 = 1)}
  sorry

end odd_non_empty_subsets_count_l151_151022


namespace coefficient_x4_in_expansion_l151_151238

theorem coefficient_x4_in_expansion : 
  (∑ k in Finset.range (9), (Nat.choose 8 k) * (3 : ℤ)^k * (2 : ℤ)^(8-k) * (X : ℤ[X])^k).coeff 4 = 90720 :=
by
  sorry

end coefficient_x4_in_expansion_l151_151238


namespace circle_symmetric_two_points_l151_151945

theorem circle_symmetric_two_points (m : ℝ) :
    (∃ (x y : ℝ), (x^2 + y^2 + m * x - 4 = 0) ∧ (∃ A B : ℝ × ℝ, A ≠ B ∧ A.1 - A.2 + 3 = 0 ∧ B.1 - B.2 + 3 = 0) →
    ∃ (m : ℝ), m = 6 := 
begin
  sorry
end

end circle_symmetric_two_points_l151_151945


namespace opposite_of_neg_two_l151_151182

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l151_151182


namespace circumcircle_centers_parallel_l151_151879

variable (A B C D M O1 O2 : Type)
variable [Geometry A B C D M O1 O2]

-- Assume an acute triangle ABC with AB < AC
axiom acute_triangle (ABC : Triangle) (h1 : AB < AC) : acute_triangle ABC

-- The bisector of ∠BAC intersects BC at D
axiom angle_bisector (h2 : bisects ∠BAC at D)

-- M is the midpoint of BC
axiom midpoint (M : Point) (h3 : midpoint M BC)

-- O1 is the circumcenter of ΔABC and O2 is the circumcenter of ΔADM
axiom circumcenter (O1 : Point) (O2 : Point) (h4: circumcenter O1 ABC) (h5: circumcenter O2 ADM)

-- Prove that the line through the centers of the circumcircles of ΔABC and ΔADM is parallel to AD
theorem circumcircle_centers_parallel (ABC : Triangle) (ADM : Triangle) (h1 : acute_triangle ABC) (h2: bisects ∠BAC at D) (h3: midpoint M BC) (h4: circumcenter O1 ABC) (h5: circumcenter O2 ADM) :
  parallel (line_through_centers O1 O2) AD := 
sorry

end circumcircle_centers_parallel_l151_151879


namespace sum_cis_angles_l151_151794

noncomputable def complex.cis (θ : ℝ) := Complex.exp (Complex.I * θ)

theorem sum_cis_angles :
  (complex.cis (80 * Real.pi / 180) + complex.cis (88 * Real.pi / 180) + complex.cis (96 * Real.pi / 180) + 
  complex.cis (104 * Real.pi / 180) + complex.cis (112 * Real.pi / 180) + complex.cis (120 * Real.pi / 180) + 
  complex.cis (128 * Real.pi / 180)) = r * complex.cis (104 * Real.pi / 180) := 
sorry

end sum_cis_angles_l151_151794


namespace cone_base_radius_l151_151294

theorem cone_base_radius 
  (R : ℝ) (θ : ℝ) (C_cone : ℝ)
  (hR : R = 4)
  (hθ : θ = 120)
  (hC_cone : C_cone = (1 / 3) * 2 * real.pi * R) :
  ∃ r : ℝ, 2 * real.pi * r = C_cone ∧ r = 4 / 3 :=
by {
  use (4 / 3),
  split,
  { -- Proof of 2 * π * r = C_cone
    rw [hR, hC_cone],
    simp,
    -- Continue proving the equality
    sorry
  },
  { -- Proof of r = 4 / 3
    simp,
    -- Continue proving the equality
    sorry
  }
}

end cone_base_radius_l151_151294


namespace opposite_of_neg_two_l151_151187

theorem opposite_of_neg_two : ∀ x : ℤ, (-2 + x = 0) → (x = 2) :=
begin
  assume x hx,
  sorry

end opposite_of_neg_two_l151_151187


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151535

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151535


namespace inscribed_circle_radius_l151_151701

theorem inscribed_circle_radius 
(ABC : Triangle) (CD : Line) 
(h_right : ABC.is_right_triangle (∠C = 90°)) 
(h_perp : CD ⊥ AB) 
(r1 r2 : ℝ) (h_r1 : r1 = 0.6) (h_r2 : r2 = 0.8) 
(r : ℝ) 
(h_1 : ABC.inscribed_radius = r) :
r = 1 :=
sorry

end inscribed_circle_radius_l151_151701


namespace ball_counts_after_199_turns_l151_151803

def initial_ball_count : list ℕ := [9, 5, 3, 2, 1]

def redistribute_balls (counts : list ℕ) : list ℕ :=
  let min_idx := counts.index_of (counts.minimum sorry) in
  counts.map_with_index (λ i n => if i = min_idx then n + counts.length - 1 else n - 1)

def ball_counts_after_turns (n : ℕ) (counts : list ℕ) : list ℕ :=
  (list.range n).foldl (λ c _ => redistribute_balls c) counts

theorem ball_counts_after_199_turns :
  ball_counts_after_turns 199 initial_ball_count = [5, 6, 4, 3, 2] :=
sorry

end ball_counts_after_199_turns_l151_151803


namespace sequence_formula_l151_151663

theorem sequence_formula (a : ℕ → ℤ)
  (h₁ : a 1 = 1)
  (h₂ : a 2 = -3)
  (h₃ : a 3 = 5)
  (h₄ : a 4 = -7)
  (h₅ : a 5 = 9) :
  ∀ n : ℕ, a n = (-1)^(n+1) * (2 * n - 1) :=
by
  sorry

end sequence_formula_l151_151663


namespace opposite_of_neg2_l151_151170

theorem opposite_of_neg2 : ∃ y : ℤ, -2 + y = 0 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end opposite_of_neg2_l151_151170


namespace sec_150_eq_l151_151407

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151407


namespace find_angle_BCD_l151_151762

variables {A B C D : Type*} [euclidean_geometry A B C D]
variables (AB BC AC BD : ℝ)
variables (α β γ δ : ℝ)
variables (convex_quadrilateral : convex_quadrilateral ABCD)
variables (angle_ABD angle_CBD : ℝ) 

def angle_BCD : ℝ := 80

theorem find_angle_BCD : 
  convex_quadrilateral ABCD → 
  AB = BC → 
  AC = BD → 
  angle_ABD = 80 →
  angle_CBD = 20 → 
  angle_BCD = 80 := 
  sorry

end find_angle_BCD_l151_151762


namespace inscribed_sphere_radius_l151_151069

-- Define the distances from points X and Y to the faces of the tetrahedron
variable (X_AB X_AD X_AC X_BC : ℝ)
variable (Y_AB Y_AD Y_AC Y_BC : ℝ)

-- Setting the given distances in the problem
axiom dist_X_AB : X_AB = 14
axiom dist_X_AD : X_AD = 11
axiom dist_X_AC : X_AC = 29
axiom dist_X_BC : X_BC = 8

axiom dist_Y_AB : Y_AB = 15
axiom dist_Y_AD : Y_AD = 13
axiom dist_Y_AC : Y_AC = 25
axiom dist_Y_BC : Y_BC = 11

-- The theorem to prove that the radius of the inscribed sphere of the tetrahedron is 17
theorem inscribed_sphere_radius : 
  ∃ r : ℝ, r = 17 ∧ 
  (∀ (d_X_AB d_X_AD d_X_AC d_X_BC d_Y_AB d_Y_AD d_Y_AC d_Y_BC: ℝ),
    d_X_AB = 14 ∧ d_X_AD = 11 ∧ d_X_AC = 29 ∧ d_X_BC = 8 ∧
    d_Y_AB = 15 ∧ d_Y_AD = 13 ∧ d_Y_AC = 25 ∧ d_Y_BC = 11 → 
    r = 17) :=
sorry

end inscribed_sphere_radius_l151_151069


namespace coefficient_x4_in_expansion_l151_151230

theorem coefficient_x4_in_expansion : 
  (∃ (c : ℤ), c = (choose 8 4) * 3^4 * 2^4 ∧ c = 90720) := 
by
  use (choose 8 4) * 3^4 * 2^4
  split
  sorry
  sorry

end coefficient_x4_in_expansion_l151_151230


namespace maximize_volume_l151_151851

-- Definitions for surface area (S), radius (r), and volume (V)
def S := arbitrary ℝ  -- Fixed surface area constant
def r := arbitrary ℝ  -- Radius 

noncomputable def V (r : ℝ) (S : ℝ) := (1 / 2) * S * r - π * r^3

-- Condition for the range of r and the solution for maximizing r
def r_max (S : ℝ) := (Real.sqrt (6 * π * S)) / (6 * π)
def r_range (r : ℝ) (S : ℝ) := 0 < r ∧ r < (Real.sqrt (2 * π * S)) / (2 * π)

theorem maximize_volume (r : ℝ) (S : ℝ) (h_cond : r_range r S) : 
  r = r_max S → V r S = V (r_max S) S :=
by
  intro h
  sorry

end maximize_volume_l151_151851


namespace reynald_volleyballs_l151_151138

def total_balls : ℕ := 145
def soccer_balls : ℕ := 20
def basketballs : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls
def baseballs : ℕ := soccer_balls + 10
def volleyballs : ℕ := total_balls - (soccer_balls + basketballs + tennis_balls + baseballs)

theorem reynald_volleyballs : volleyballs = 30 :=
by
  sorry

end reynald_volleyballs_l151_151138


namespace correct_operation_l151_151837

theorem correct_operation (a m : ℝ) :
  ¬(a^5 / a^10 = a^2) ∧ 
  (-2 * a^3)^2 = 4 * a^6 ∧ 
  ¬((1 / (2 * m)) - (1 / m) = (1 / m)) ∧ 
  ¬(a^4 + a^3 = a^7) :=
by
  sorry

end correct_operation_l151_151837


namespace emily_lives_lost_l151_151916

variable (L : ℕ)
variable (initial_lives : ℕ) (extra_lives : ℕ) (final_lives : ℕ)

-- Conditions based on the problem statement
axiom initial_lives_def : initial_lives = 42
axiom extra_lives_def : extra_lives = 24
axiom final_lives_def : final_lives = 41

-- Mathematically equivalent proof statement
theorem emily_lives_lost : initial_lives - L + extra_lives = final_lives → L = 25 := by
  sorry

end emily_lives_lost_l151_151916


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151455

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151455


namespace john_new_weekly_earnings_l151_151074

theorem john_new_weekly_earnings :
  let original_earnings : ℝ := 40
  let percentage_increase : ℝ := 37.5 / 100
  let raise_amount : ℝ := original_earnings * percentage_increase
  let new_weekly_earnings : ℝ := original_earnings + raise_amount
  new_weekly_earnings = 55 := 
by
  sorry

end john_new_weekly_earnings_l151_151074


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151382

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151382


namespace probability_of_red_then_blue_is_correct_l151_151856

noncomputable def probability_red_then_blue : ℚ :=
  let total_marbles := 5 + 4 + 12 + 2
  let prob_red := 5 / total_marbles
  let remaining_marbles := total_marbles - 1
  let prob_blue_given_red := 2 / remaining_marbles
  prob_red * prob_blue_given_red

theorem probability_of_red_then_blue_is_correct :
  probability_red_then_blue = 5 / 253 := 
by 
  sorry

end probability_of_red_then_blue_is_correct_l151_151856


namespace probability_sum_12_with_octahedral_dice_l151_151278

open Function

theorem probability_sum_12_with_octahedral_dice : 
  let dice_faces := {1, 2, 3, 4, 5, 6, 7, 8},
      all_possible_rolls := dice_faces ×ˢ dice_faces,
      successful_rolls := {roll ∈ all_possible_rolls | (roll.1 + roll.2 = 12)},
      probability := successful_rolls.card / all_possible_rolls.card 
  in probability = (5 : ℚ) / 64 :=
by
  let dice_faces := {1, 2, 3, 4, 5, 6, 7, 8}
  let all_possible_rolls := dice_faces ×ˢ dice_faces
  let successful_rolls := {roll ∈ all_possible_rolls | (roll.1 + roll.2 = 12)}
  let probability := successful_rolls.card / all_possible_rolls.card 
  have : all_possible_rolls.card = 64 := sorry  -- Total outcomes from 8 faces on each die
  have : successful_rolls.card = 5 := sorry      -- Successful pairs adding up to 12
  have : probability = (5 : ℚ) / 64 := sorry     -- Probability calculation
  exact this

end probability_sum_12_with_octahedral_dice_l151_151278


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151526

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151526


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151417

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151417


namespace negative_expression_l151_151323

noncomputable section

-- Define the expressions
def A := -( -3 : ℝ)
def B := -( (-3) ^ 3 : ℝ )
def C := ( -3 : ℝ ) ^ 2
def D := -( abs( -3 : ℝ ))

-- Prove that D is the only negative one
theorem negative_expression :
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D < 0 :=
by
  -- Expressions evaluation
  have hA : A = 3 := by sorry
  have hB : B = 27 := by sorry
  have hC : C = 9 := by sorry
  have hD : D = -3 := by sorry
  
  -- Prove the inequalities
  exact ⟨by rw [hA]; exact zero_lt_three, by rw [hB]; exact zero_lt_twentyseven, by rw [hC]; exact zero_lt_nine, by rw [hD]; exact neg_three_lt_zero⟩

end negative_expression_l151_151323


namespace carmichael_lt_100000_count_eq_16_l151_151989

def is_carmichael (n : ℕ) : Prop :=
  n > 1 ∧
  ¬ nat.prime n ∧
  nat.odd n ∧
  ∀ a : ℕ, 1 < a ∧ a < n ∧ nat.coprime a n → n ∣ (a ^ (n - 1) - 1)

theorem carmichael_lt_100000_count_eq_16 :
  { n : ℕ | is_carmichael n ∧ n < 100000 }.card = 16 := sorry

end carmichael_lt_100000_count_eq_16_l151_151989


namespace basketball_team_total_players_l151_151285

theorem basketball_team_total_players (total_points : ℕ) (min_points : ℕ) (max_points : ℕ) (team_size : ℕ)
  (h1 : total_points = 100)
  (h2 : min_points = 7)
  (h3 : max_points = 23)
  (h4 : ∀ (n : ℕ), n ≥ min_points)
  (h5 : max_points = 23)
  : team_size = 12 :=
sorry

end basketball_team_total_players_l151_151285


namespace tangent_line_min_length_and_coordinates_l151_151705

def curve_C1 (α : ℝ) : Prop := (λ (x y : ℝ), x = real.sqrt 2 + real.cos α ∧ y = real.sqrt 2 + real.sin α)

def curve_C1_cartesian : Prop :=
∀ x y : ℝ, (x - real.sqrt 2)^2 + (y - real.sqrt 2)^2 = 1

def curve_C2_polar (θ : ℝ) : Prop := 
∀ ρ : ℝ, ρ = 8 / (real.sin (θ + real.pi / 4))

def curve_C2_cartesian : Prop :=
∀ x y : ℝ, x + y = 8 * real.sqrt 2

def minimum_tangent_length (P : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
P = (4 * real.sqrt 2, 4 * real.sqrt 2) ∧ Q ∈ set_of (curve_C1 (8, real.pi / 4)) ∧ dist P Q = real.sqrt 35

def polar_coordinates_P : Prop :=
P = (8, real.pi / 4)

theorem tangent_line_min_length_and_coordinates :
  ∃ P Q, minimum_tangent_length P Q ∧ polar_coordinates_P P :=
sorry

end tangent_line_min_length_and_coordinates_l151_151705


namespace sec_150_l151_151477

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151477


namespace four_divides_n_l151_151111

variable {n : ℕ}
variable {a : ℕ → ℤ}

-- Function definition ensuring a_i ∈ {-1, 1} 
def is_in_neg1_to_1 (a : ℕ → ℤ) (n : ℕ) : Prop := 
∀i, 1 ≤ i ∧ i ≤ n → a i = 1 ∨ a i = -1

-- Sum of products condition
def sum_of_products_zero (a : ℕ → ℤ) (n : ℕ) : Prop :=
∑ i in Finset.range n, a i * a ((i+1) % n) = 0

theorem four_divides_n
  (h_in_set : is_in_neg1_to_1 a n)
  (h_sum_zero : sum_of_products_zero a n) : 4 ∣ n := sorry

end four_divides_n_l151_151111


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151441

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151441


namespace find_p_q_r_l151_151161

noncomputable def polynomials_root (p q r : ℕ) (h : p > 0 ∧ q > 0 ∧ r > 0) : Prop :=
∃ x : ℝ, x = (real.cbrt p + real.cbrt q + 1) / r ∧ 27 * x^3 - 4 * x^2 - 4 * x - 1 = 0

theorem find_p_q_r :
  (∃ (p q r : ℕ), p > 0 ∧ q > 0 ∧ r > 0 ∧ (polynomials_root p q r) ∧ p + q + r = 12) :=
sorry

end find_p_q_r_l151_151161


namespace proof_problem_l151_151061

-- Definitions of sequence terms and their properties
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ a 4 = 16 ∧ ∀ n, a n = 2^n

-- Definition for the sum of the first n terms of the sequence
noncomputable def sum_of_sequence (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n + 1) - 2

-- Definition for the transformed sequence b_n = log_2 a_n
def transformed_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ n, b n = Nat.log2 (a n)

-- Definition for the sum T_n related to b_n
noncomputable def sum_of_transformed_sequence (T : ℕ → ℚ) (b : ℕ → ℕ) : Prop :=
  ∀ n, T n = 1 - 1 / (n + 1)

theorem proof_problem :
  (∃ a : ℕ → ℕ, geometric_sequence a) ∧
  (∃ S : ℕ → ℕ, sum_of_sequence S) ∧
  (∃ (a b : ℕ → ℕ), geometric_sequence a ∧ transformed_sequence a b ∧
   (∃ T : ℕ → ℚ, sum_of_transformed_sequence T b)) :=
by {
  -- Definitions and proofs will go here
  sorry
}

end proof_problem_l151_151061


namespace selection_sets_count_l151_151639

noncomputable def number_of_ways (n : ℕ) : ℕ :=
  (2 * n)! * 2^(n^2)

theorem selection_sets_count (n : ℕ) : 
  ∃ f : (Fin (n+1) × Fin (n+1) → Finset (Fin (2 * n))), 
  (∀ i j, (i.1 + j.1 : ℕ) = (f i j).card) ∧ 
  (∀ (i j k l : Fin (n+1)),
    (i.1 ≤ k.1 ∧ j.1 ≤ l.1) → f i j ⊆ f k l) ∧
  ∃ g : (Fin (n+1) → Fin (n+1) → Finset (Fin (2 * n))),
    (∀ i j k l, (i, j) = (k, l) → g i j = f k l)
  :=
begin
  use λ (ij : Fin (n+1) × Fin (n+1)), (Finset.range (ij.1 + ij.2)),
  split,
  { -- First condition to prove
    intros i j,
    simp,
  },
  split,
  { -- Second condition to prove
    intros i j k l h,
    cases h,
    sorry,
  },
  { -- Provided correct answer
    sorry,
  }
end

end selection_sets_count_l151_151639


namespace isosceles_triangle_perimeter_l151_151882

theorem isosceles_triangle_perimeter
  (a b c : ℝ )
  (ha : a = 20)
  (hb : b = 20)
  (hc : c = (2/5) * 20)
  (triangle_ineq1 : a ≤ b + c)
  (triangle_ineq2 : b ≤ a + c)
  (triangle_ineq3 : c ≤ a + b) :
  a + b + c = 48 := by
  sorry

end isosceles_triangle_perimeter_l151_151882


namespace ratio_new_circumference_to_original_diameter_l151_151693

-- Define the problem conditions
variables (r k : ℝ) (hk : k > 0)

-- Define the Lean theorem to express the proof problem
theorem ratio_new_circumference_to_original_diameter (r k : ℝ) (hk : k > 0) :
  (π * (1 + k / r)) = (2 * π * (r + k)) / (2 * r) :=
by {
  -- Placeholder proof, to be filled in
  sorry
}

end ratio_new_circumference_to_original_diameter_l151_151693


namespace quadratic_has_two_distinct_real_roots_l151_151791

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  let Δ := m^2 + 20 in Δ > 0 :=
by
  let Δ := m^2 + 20
  show Δ > 0
  sorry

end quadratic_has_two_distinct_real_roots_l151_151791


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151509

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151509


namespace exist_pos_integers_for_O_in_triangle_l151_151733

theorem exist_pos_integers_for_O_in_triangle 
  (A B C O : EuclideanSpace ℝ (Fin 2)) 
  (h : ∃ (α β : ℝ) (a b c : ℝ), α > 0 ∧ β > 0 ∧ (a = α ∧ b = β ∧ c = 1) ∧ (α • A + β • B + c • C = (0 : EuclideanSpace ℝ (Fin 2)))) :
  ∃ (p q r : ℕ), (p > 0 ∧ q > 0 ∧ r > 0) ∧ 
    ∥ (p • A + q • B + r • C : EuclideanSpace ℝ (Fin 2)) ∥ < 1/2007 := 
by
  sorry

end exist_pos_integers_for_O_in_triangle_l151_151733


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151438

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151438


namespace inequality_non_empty_solution_set_l151_151792

theorem inequality_non_empty_solution_set (a : ℝ) : ∃ x : ℝ, ax^2 - (a-2)*x - 2 ≤ 0 :=
sorry

end inequality_non_empty_solution_set_l151_151792


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151559

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151559


namespace conjugate_of_fraction_l151_151957

theorem conjugate_of_fraction :
  ∀ (i : ℂ), i = complex.I → conj ( (1 + i) / (i^3)) = -1 - i := by
sorr

end conjugate_of_fraction_l151_151957


namespace reynald_volleyballs_l151_151137

def total_balls : ℕ := 145
def soccer_balls : ℕ := 20
def basketballs : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls
def baseballs : ℕ := soccer_balls + 10
def volleyballs : ℕ := total_balls - (soccer_balls + basketballs + tennis_balls + baseballs)

theorem reynald_volleyballs : volleyballs = 30 :=
by
  sorry

end reynald_volleyballs_l151_151137


namespace power_mod_equiv_l151_151824

theorem power_mod_equiv :
  7 ^ 145 % 12 = 7 % 12 :=
by
  -- Here the solution would go
  sorry

end power_mod_equiv_l151_151824


namespace football_problem_l151_151933

-- Definitions based on conditions
def total_balls (x y : Nat) : Prop := x + y = 200
def total_cost (x y : Nat) : Prop := 80 * x + 60 * y = 14400
def football_A_profit_per_ball : Nat := 96 - 80
def football_B_profit_per_ball : Nat := 81 - 60
def total_profit (x y : Nat) : Nat :=
  football_A_profit_per_ball * x + football_B_profit_per_ball * y

-- Lean statement proving the conditions lead to the solution
theorem football_problem
  (x y : Nat)
  (h1 : total_balls x y)
  (h2 : total_cost x y)
  (h3 : x = 120)
  (h4 : y = 80) :
  total_profit x y = 3600 := by
  sorry

end football_problem_l151_151933


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151500

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151500


namespace min_cost_yogurt_l151_151913

theorem min_cost_yogurt (cost_per_box : ℕ) (boxes : ℕ) (promotion : ℕ → ℕ) (cost : ℕ) :
  cost_per_box = 4 → 
  boxes = 10 → 
  promotion 3 = 2 → 
  cost = 28 := 
by {
  -- The proof will go here
  sorry
}

end min_cost_yogurt_l151_151913


namespace coefficient_x4_in_expansion_l151_151231

theorem coefficient_x4_in_expansion : 
  (∃ (c : ℤ), c = (choose 8 4) * 3^4 * 2^4 ∧ c = 90720) := 
by
  use (choose 8 4) * 3^4 * 2^4
  split
  sorry
  sorry

end coefficient_x4_in_expansion_l151_151231


namespace volume_of_pyramid_is_correct_l151_151873

-- Declare the base and height as parameters
variables (length width height : ℝ)
-- Assumptions
axiom base_lengths : length = 1/3 ∧ width = 2/3
axiom pyramid_height : height = 1/2

-- Define the base area and volume of the pyramid
def base_area : ℝ := length * width
def volume_of_pyramid : ℝ := (1/3) * base_area * height

-- The theorem statement
theorem volume_of_pyramid_is_correct :
  base_lengths → pyramid_height → volume_of_pyramid length width height = 1/27 :=
by
  intros
  -- Lean will require proofs for these assumptions to be complete, but this is the statement required
  sorry

end volume_of_pyramid_is_correct_l151_151873


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151522

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151522


namespace pounds_per_ton_l151_151123

theorem pounds_per_ton (packet_count : ℕ) (packet_weight_pounds : ℚ) (packet_weight_ounces : ℚ) (ounces_per_pound : ℚ) (total_weight_tons : ℚ) (total_weight_pounds : ℚ) :
  packet_count = 1760 →
  packet_weight_pounds = 16 →
  packet_weight_ounces = 4 →
  ounces_per_pound = 16 →
  total_weight_tons = 13 →
  total_weight_pounds = (packet_count * (packet_weight_pounds + (packet_weight_ounces / ounces_per_pound))) →
  total_weight_pounds / total_weight_tons = 2200 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pounds_per_ton_l151_151123


namespace Melies_initial_money_l151_151746

theorem Melies_initial_money (costPerKg : ℕ) (meatKg : ℕ) (moneyLeft : ℕ) (totalCost : ℕ) (initialMoney : ℕ) :
  costPerKg = 82 →
  meatKg = 2 →
  moneyLeft = 16 →
  totalCost = meatKg * costPerKg →
  initialMoney = totalCost + moneyLeft →
  initialMoney = 180 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end Melies_initial_money_l151_151746


namespace even_product_probability_l151_151760

theorem even_product_probability :
  let spinnerA := {1, 2, 3, 4, 5}
  let spinnerB := {1, 2, 4}
  (∀ n ∈ spinnerA, 1 ≤ n ∧ n ≤ 5) ∧ (∀ m ∈ spinnerB, 1 ≤ m ∧ m ≤ 4) →
  (∀ n ∈ spinnerA, ∀ m ∈ spinnerB, (1/5 : ℚ) * (1/3 : ℚ) = 1/15) →
  let even_prob := 1 - (3 / 15) in
  even_prob = (4 / 5 : ℚ) := 
sorry

end even_product_probability_l151_151760


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151414

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151414


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151578

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151578


namespace butterfat_calculation_l151_151020

theorem butterfat_calculation :
  ∀ (m₁ m₂ : ℝ) (x : ℝ), 
  m₁ = 20 → 
  m₂ = 8 → 
  (20 + 8) * 0.2 = 20 * 0.1 + 8 * (x / 100) → 
  x = 45 :=
by 
  intros m₁ m₂ x h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  linarith

end butterfat_calculation_l151_151020


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151493

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151493


namespace length_of_bridge_is_correct_l151_151786

-- Define the variables
def train_length : ℕ := 250
def train_speed_kmh : ℕ := 72
def crossing_time : ℕ := 30

-- Define the conversion rate from km/hr to m/s
def conversion_rate : ℝ := 1000 / 3600

-- The train speed in m/s
def train_speed_ms : ℝ := train_speed_kmh * conversion_rate

-- Distance covered by the train in 30 seconds
def total_distance : ℝ := train_speed_ms * crossing_time

-- Length of the bridge
def bridge_length : ℝ := total_distance - train_length

-- The proof statement
theorem length_of_bridge_is_correct : bridge_length = 350 := by
  sorry

end length_of_bridge_is_correct_l151_151786


namespace digits_satisfy_sqrt_l151_151600

theorem digits_satisfy_sqrt (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  (b = 0 ∧ a = 0) ∨ (b = 3 ∧ a = 1) ∨ (b = 6 ∧ a = 4) ∨ (b = 9 ∧ a = 9) ↔ b^2 = 9 * a :=
by
  sorry

end digits_satisfy_sqrt_l151_151600


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151409

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151409


namespace alice_wins_probability_l151_151877

/-- 
Alice has a coin that lands heads with a probability of 1/4.
Bob has a coin that lands heads with a probability of 3/7.
Alice and Bob alternately toss their coins until someone gets a head.
Alice goes first.
Alice wins if she gets heads on an odd-numbered turn.
Prove that the probability that Alice wins the game is 7/19.
-/
theorem alice_wins_probability :
  let p_head_Alice := (1:ℚ) / 4,
      p_head_Bob := (3:ℚ) / 7,
      p_tail_Both := (3:ℚ / 4) * ((4:ℚ) / 7),
      p_Alice_wins_3rd := p_tail_Both * (1:ℚ / 4)
  in 
    1 / 4 + 
    p_tail_Both * p_Alice_wins_3rd /
    (1 - p_tail_Both * p_Alice_wins_3rd) = (7:ℚ) / 19 :=
sorry

end alice_wins_probability_l151_151877


namespace citizen_income_l151_151907

noncomputable def income (I : ℝ) : Prop :=
  let P := 0.11 * 40000
  let A := I - 40000
  P + 0.20 * A = 8000

theorem citizen_income (I : ℝ) (h : income I) : I = 58000 := 
by
  -- proof steps go here
  sorry

end citizen_income_l151_151907


namespace man_son_age_ratio_l151_151304

-- Define the present age of the son
def son_age_present : ℕ := 22

-- Define the present age of the man based on the son's age
def man_age_present : ℕ := son_age_present + 24

-- Define the son's age in two years
def son_age_future : ℕ := son_age_present + 2

-- Define the man's age in two years
def man_age_future : ℕ := man_age_present + 2

-- Prove the ratio of the man's age to the son's age in two years is 2:1
theorem man_son_age_ratio : man_age_future / son_age_future = 2 := by
  sorry

end man_son_age_ratio_l151_151304


namespace find_g_l151_151272

theorem find_g (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x+1) = 3 - 2 * x) (h2 : ∀ x : ℝ, f (g x) = 6 * x - 3) : 
  ∀ x : ℝ, g x = 4 - 3 * x := 
by
  sorry

end find_g_l151_151272


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151358

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151358


namespace sqrt_eq_2_then_square_eq_16_l151_151026

theorem sqrt_eq_2_then_square_eq_16 (x : ℝ) (h : sqrt (x + 2) = 2) : (x + 2)^2 = 16 := by
  sorry

end sqrt_eq_2_then_square_eq_16_l151_151026


namespace degree_of_polynomial_l151_151245

theorem degree_of_polynomial :
  polynomial.degree ((X^3 + X + 1)^5 * (X^4 - 1)^2 * (X - 1)^3) = 26 :=
by
  sorry

end degree_of_polynomial_l151_151245


namespace bryden_receives_10_dollars_l151_151292

theorem bryden_receives_10_dollars 
  (collector_rate : ℝ := 5)
  (num_quarters : ℝ := 4)
  (face_value_per_quarter : ℝ := 0.50) :
  collector_rate * num_quarters * face_value_per_quarter = 10 :=
by
  sorry

end bryden_receives_10_dollars_l151_151292


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151571

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151571


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151381

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151381


namespace option_D_correct_l151_151106

variables (Line : Type) (Plane : Type)
variables (parallel : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop)
variables (perpendicular_planes : Plane → Plane → Prop)

theorem option_D_correct (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → perpendicular_planes α β :=
sorry

end option_D_correct_l151_151106


namespace distinct_complex_numbers_no_solution_l151_151131

theorem distinct_complex_numbers_no_solution :
  ¬∃ (a b c d : ℂ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (a^3 - b * c * d = b^3 - c * d * a) ∧ 
  (b^3 - c * d * a = c^3 - d * a * b) ∧ 
  (c^3 - d * a * b = d^3 - a * b * c) := 
by {
  sorry
}

end distinct_complex_numbers_no_solution_l151_151131


namespace sec_150_eq_l151_151392

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151392


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151415

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151415


namespace has_two_real_roots_l151_151345

theorem has_two_real_roots : ∀ (x : ℝ), (x - real.sqrt (2 * x + 6) = 2) ↔ (x = 3 + real.sqrt 11 ∨ x = 3 - real.sqrt 11) :=
by
  sorry

end has_two_real_roots_l151_151345


namespace problem_l151_151968

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
if x < 1 then -x^3 + x^2 + b * x + c else a * Real.log x

theorem problem
  (a b c : ℝ)
  (h1 : f (-1) a b c = 2)
  (h2 : Deriv (f _ a b c) (-1) = -5)
  (h3 : ∀ x y : ℝ, x - 5 * y + 1 = 0 → tangent_perpendicular (x, y) (-1, f (-1) a b c)) :
  b = 0 ∧ c = 0 ∧ ∀ a : ℝ, (f (1:ℝ) a 0 0).maximum_on [-1, Real.exp 1] = max a 2 := by
sorry

end problem_l151_151968


namespace fractional_inspection_l151_151715

theorem fractional_inspection:
  ∃ (J E A : ℝ),
  J + E + A = 1 ∧
  0.005 * J + 0.007 * E + 0.012 * A = 0.01 :=
by
  sorry

end fractional_inspection_l151_151715


namespace length_of_stone_slab_l151_151281

theorem length_of_stone_slab 
  (num_slabs : ℕ) 
  (total_area : ℝ) 
  (h_num_slabs : num_slabs = 30) 
  (h_total_area : total_area = 50.7): 
  ∃ l : ℝ, l = 1.3 ∧ l * l * num_slabs = total_area := 
by 
  sorry

end length_of_stone_slab_l151_151281


namespace max_tangential_quadrilaterals_l151_151112

noncomputable def largest_k_with_inscribed_circle (n : ℕ) (h : n ≥ 5) : ℕ :=
  ⌊ n / 2 ⌋

theorem max_tangential_quadrilaterals (n : ℕ) (h : n ≥ 5) :
  ∃ k, (k = largest_k_with_inscribed_circle n h) ∧ ∃ (A : Fin n → Point), 
  (convex_poly n A) ∧ (count_tang_quads n A = k) :=
sorry

end max_tangential_quadrilaterals_l151_151112


namespace articles_selling_price_eq_cost_price_of_50_articles_l151_151033

theorem articles_selling_price_eq_cost_price_of_50_articles (C S : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * S) (h2 : S = 2 * C) : N = 25 := by
  sorry

end articles_selling_price_eq_cost_price_of_50_articles_l151_151033


namespace complement_M_inter_N_eq_l151_151011
noncomputable theory

-- Define U, M, and N based on the conditions
def U : Set ℝ := {x | True}

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

def N : Set ℝ := {y | ∃ x : ℝ, y = log (3 - x)}

-- Define the complement of M in U
def complement_M : Set ℝ := {y | y ≤ 0}

-- Statement of the problem
theorem complement_M_inter_N_eq : (complement_M ∩ N) = {y | y ≤ 0} := by
  sorry

end complement_M_inter_N_eq_l151_151011


namespace sec_150_eq_l151_151466

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151466


namespace devon_rotation_correct_l151_151895

-- Define the conditions
def carla_rotation (angle : ℕ) : ℕ :=
  angle % 360  -- Simplifies Carla's rotation to within 0 to 359 degrees

def devon_rotation (carla_effective : ℕ) : ℕ :=
  360 - carla_effective  -- Calculates Devon's rotation x

-- Main theorem statement to prove
theorem devon_rotation_correct :
  let carla_effective := carla_rotation 550 in
  let x := devon_rotation carla_effective in
  x < 360 ∧ x = 170 :=
by
  sorry

end devon_rotation_correct_l151_151895


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151448

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151448


namespace smallest_solution_l151_151607

def equation (x : ℝ) := (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 14

theorem smallest_solution :
  ∀ x : ℝ, equation x → x = (3 - Real.sqrt 333) / 6 :=
sorry

end smallest_solution_l151_151607


namespace quadratic_pyramid_cut_correct_l151_151771

noncomputable def quadratic_pyramid_cut (a b c : ℝ) : ℝ :=
  have volume_equal_p : a = 6 ∧ b = 8 ∧ c = 12 := by
    -- The question constraints
    sorry
  let m := Real.sqrt ((4 * (c ^ 2)) - (a ^ 2 + b ^ 2)) / 2
  let x := m / (Real.cbrt 2)
  x

theorem quadratic_pyramid_cut_correct : quadratic_pyramid_cut 6 8 12 ≈ 8.658 := by
  -- Skip the proof
  sorry

end quadratic_pyramid_cut_correct_l151_151771


namespace sum_of_distinct_x_l151_151902

def g (x : ℝ) : ℝ := x^2 / 4 + x - 2

theorem sum_of_distinct_x (x : ℝ) (gx := g x) (ggx := g (g x)) :
  ∃ xs : set ℝ, (∀ x ∈ xs, g (g (g x)) = -1) ∧ ∑ x in xs.to_finset, x = -8 := by
sorry

end sum_of_distinct_x_l151_151902


namespace age_difference_l151_151717

variable (K_age L_d_age L_s_age : ℕ)

def condition1 : Prop := L_d_age = K_age - 10
def condition2 : Prop := L_s_age = 2 * K_age
def condition3 : Prop := K_age = 12

theorem age_difference : condition1 → condition2 → condition3 → L_s_age - L_d_age = 22 := by
  intros h1 h2 h3
  rw [h3] at h1
  rw [h3] at h2
  simp at h1
  simp at h2
  rw [h1, h2]
  norm_num
  sorry

end age_difference_l151_151717


namespace infinite_rational_points_l151_151023

theorem infinite_rational_points (x y : ℚ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y ≤ 6) :
  ∃ S : set (ℚ × ℚ), set.infinite S ∧ (∀ p ∈ S, (0 < p.1 ∧ 0 < p.2 ∧ p.1 + 2 * p.2 ≤ 6)) :=
by 
  sorry

end infinite_rational_points_l151_151023


namespace slope_of_tangent_at_one_l151_151195

theorem slope_of_tangent_at_one :
  let y := λ (x : ℝ), (1/2) * x^2 - 2 in
  let y' := λ (x : ℝ), x in
  y' 1 = 1 :=
by sorry

end slope_of_tangent_at_one_l151_151195


namespace count_one_full_numbers_l151_151342

/-
Definitions of conditions
1. A digit can be either 0, 1, or 2.
2. Out of every two consecutive digits, at least one of them is 1.
3. Numbers cannot start with the digit 0.
-/

def is_one_full (digits : List ℕ) : Prop :=
  (∀ d ∈ digits, d = 0 ∨ d = 1 ∨ d = 2) ∧
  (∀ i < digits.length - 1, digits.get! i = 1 ∨ digits.get! (i + 1) = 1) ∧
  (digits.head! ≠ 0)

def one_full_count (n : ℕ) : ℕ :=
  if h : n ≥ 2 then 2 ^ n else 0

theorem count_one_full_numbers (n : ℕ) (h : n ≥ 2) :
  one_full_count n = 2 ^ n :=
by
  rw [one_full_count]; simp [h]
  sorry

end count_one_full_numbers_l151_151342


namespace sector_area_l151_151159

theorem sector_area (α : ℝ) (l : ℝ) (r : ℝ) :
  α = 2 ∧ l = 4 ∧ l = α * r → (1 / 2) * α * r^2 = 1 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h3] at h2
  have r_eq_1 : r = 1 := by linarith
  rw r_eq_1 at *
  simp
  exact h1

end sector_area_l151_151159


namespace functional_equation_solution_l151_151724

noncomputable def candidate_function_1 : ℝ → ℝ := λ x, 2 - x
noncomputable def candidate_function_2 : ℝ → ℝ := λ x, x

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x + f(x + y)) + f(xy) = x + f(x + y) + y * f(x)) →
  (f = candidate_function_1 ∨ f = candidate_function_2) :=
by {
  sorry
}

end functional_equation_solution_l151_151724


namespace BM_eq_KQ_l151_151070

variables {A B C M K N Q : Type*}
variables [PlaneGeometry A] [PlaneGeometry B] [PlaneGeometry C] [PlaneGeometry M] [PlaneGeometry K] [PlaneGeometry N] [PlaneGeometry Q]
variables (triangle_ABC : Triangle A B C)
variables (M_inside_ABC : Point M triangle_ABC)
variables (K_on_BC : Point K (side B C triangle_ABC))
variables (MK_parallel_AB : Parallel M K A B)
variables (circle_MKC : Circle M K C)
variables (N_on_circle_MKC : Point N (circle_MKC.ac_intersect_other A C))
variables (circle_MNA : Circle M N A)
variables (Q_on_circle_MNA : Point Q (circle_MNA.ab_intersect_other A B))

theorem BM_eq_KQ : BM = KQ := sorry

end BM_eq_KQ_l151_151070


namespace sec_150_eq_l151_151397

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151397


namespace age_ratio_proof_l151_151303

-- Define the ages
def sonAge := 22
def manAge := sonAge + 24

-- Define the ratio computation statement
def ageRatioInTwoYears : ℚ := 
  let sonAgeInTwoYears := sonAge + 2
  let manAgeInTwoYears := manAge + 2
  manAgeInTwoYears / sonAgeInTwoYears

-- The theorem to prove
theorem age_ratio_proof : ageRatioInTwoYears = 2 :=
by
  sorry

end age_ratio_proof_l151_151303


namespace minimum_number_of_small_pipes_l151_151874

-- Define the conditions
def radius_large : ℝ := 6
def radius_small : ℝ := 1.5
def height : ℝ := 1  -- assuming a unit length for simplicity

-- Define the volume function for cylindrical pipes
noncomputable def volume (r : ℝ) (h : ℝ) : ℝ := π * r ^ 2 * h

-- Prove the volume ratio
theorem minimum_number_of_small_pipes :
  (volume radius_large height) / (volume radius_small height) = 16 :=
by
  sorry

end minimum_number_of_small_pipes_l151_151874


namespace workout_total_weight_l151_151319

def total_chest_weight (w r : ℕ) : ℕ :=
w * r

def total_back_weight (w r : ℕ) : ℕ :=
w * r

def total_leg_weight (w r : ℕ) : ℕ :=
w * r

def grand_total_weight (chest back leg : ℕ) : ℕ :=
chest + back + leg

theorem workout_total_weight :
  total_chest_weight 90 8 + total_back_weight 70 10 + total_leg_weight 130 6 = 2200 := 
by
  unfold total_chest_weight
  unfold total_back_weight
  unfold total_leg_weight
  unfold grand_total_weight
  sorry

end workout_total_weight_l151_151319


namespace circle_through_points_line_intersects_circle_segment_length_n_l151_151006

-- Define the interpolation points
def M := (3 : ℝ, 0 : ℝ)
def N := (1 : ℝ, 0 : ℝ)
def P := (0 : ℝ, 3 : ℝ)

-- Define the circle
def circle_eq (x y a b r : ℝ) := (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2

-- Prove the circle goes through points M, N, and P
theorem circle_through_points :
  ∃ (a b r : ℝ), 
    circle_eq 3 0 a b r ∧ 
    circle_eq 1 0 a b r ∧
    circle_eq 0 3 a b r ∧
    (a = 2) ∧ (b = 2) ∧ (r = sqrt 5) :=
sorry

-- Prove the intersection with line and segment length condition
theorem line_intersects_circle_segment_length_n :
  let c_eq := circle_eq,
      line_eq := fun (x y n : ℝ) => x - y + n = 0 in
  ∃ (n : ℝ),
    let d := abs (2 - 2) / sqrt 2 in
    (sqrt 5) ^ 2 = d ^ 2 + (4 / 2) ^ 2 ∧
    (n = sqrt 2 ∨ n = -sqrt 2) :=
sorry

end circle_through_points_line_intersects_circle_segment_length_n_l151_151006


namespace find_x_distance_traveled_l151_151321

theorem find_x_distance_traveled :
  ∃ x : ℝ, let final_coords := (-2, x - 2 * Real.sqrt 3) in
           (0 - (-2))^2 + (x - 2 * Real.sqrt 3 - 0)^2 = 2^2 ∧ x = 2 * Real.sqrt 3 :=
by
  sorry

end find_x_distance_traveled_l151_151321


namespace stratified_sampling_example_l151_151852

-- Define the conditions as parameters for the theorem
variables (h : ℕ) (m : ℕ) (k : ℕ)

-- State the theorem:
theorem stratified_sampling_example
    (h_eq : h = 3500)
    (m_eq : m = 1500)
    (k_eq : k = 30):
    let n := (m + h) // m * k in
    n = 100 :=
by
  sorry

end stratified_sampling_example_l151_151852


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151432

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151432


namespace greatest_prime_factor_expression_l151_151842

theorem greatest_prime_factor_expression : 
  let expr := (factorial 11 * factorial 10 + factorial 10 * factorial 9) / 111 in
  nat.greatest_prime_factor expr = 7 :=
by
  -- Define the expression
  let expr := (factorial 11 * factorial 10 + factorial 10 * factorial 9) / 111;
  sorry

end greatest_prime_factor_expression_l151_151842


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151418

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151418


namespace combination_15_5_l151_151205

theorem combination_15_5 : 
  ∀ (n r : ℕ), n = 15 → r = 5 → n.choose r = 3003 :=
by
  intro n r h1 h2
  rw [h1, h2]
  exact Nat.choose_eq_factorial_div_factorial (by norm_num)

end combination_15_5_l151_151205


namespace triangle_area_inscribed_circle_l151_151872

noncomputable def arc_length1 : ℝ := 6
noncomputable def arc_length2 : ℝ := 8
noncomputable def arc_length3 : ℝ := 10

noncomputable def circumference : ℝ := arc_length1 + arc_length2 + arc_length3
noncomputable def radius : ℝ := circumference / (2 * Real.pi)
noncomputable def angle_sum : ℝ := 360
noncomputable def angle1 : ℝ := 90 * Real.pi / 180
noncomputable def angle2 : ℝ := 120 * Real.pi / 180
noncomputable def angle3 : ℝ := 150 * Real.pi / 180

theorem triangle_area_inscribed_circle :
  let r := radius in
  let sin_a1 := Real.sin angle1 in
  let sin_a2 := Real.sin angle2 in
  let sin_a3 := Real.sin angle3 in
  (1 / 2) * r^2 * (sin_a1 + sin_a2 + sin_a3) = (72 * (1 + Real.sqrt 3)) / (Real.pi^2) := by
  sorry

end triangle_area_inscribed_circle_l151_151872


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151437

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151437


namespace range_of_a_l151_151064

noncomputable def operation (x y : ℝ) := x * (1 - y)

theorem range_of_a
  (a : ℝ)
  (hx : ∀ x : ℝ, operation (x - a) (x + a) < 1) :
  -1/2 < a ∧ a < 3/2 := by
  sorry

end range_of_a_l151_151064


namespace winning_votes_l151_151208

theorem winning_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 312) : 0.62 * V = 806 :=
by
  -- The proof should be written here, but we'll skip it as per the instructions.
  sorry

end winning_votes_l151_151208


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151504

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151504


namespace cos_B_in_triangle_l151_151037

theorem cos_B_in_triangle (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = Real.pi) : 
  Real.cos B = 1 / 2 :=
sorry

end cos_B_in_triangle_l151_151037


namespace cone_base_radius_l151_151295

-- Definitions for conditions
def sector_angle : ℝ := 120
def sector_radius : ℝ := 4

-- Theorem to prove
theorem cone_base_radius (r : ℝ) : 
  let arc_length := (sector_angle / 360) * (2 * Real.pi * sector_radius)
  let base_circumference := 2 * Real.pi * r
  arc_length = base_circumference →
  r = 4 / 3 :=
by
  intro h
  have h1 : arc_length = (120 / 360) * (2 * Real.pi * 4), from rfl
  rw h1 at h
  sorry -- skipping proof

end cone_base_radius_l151_151295


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151547

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151547


namespace sec_150_eq_l151_151404

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151404


namespace opposite_of_neg_two_l151_151186

theorem opposite_of_neg_two : ∀ x : ℤ, (-2 + x = 0) → (x = 2) :=
begin
  assume x hx,
  sorry

end opposite_of_neg_two_l151_151186


namespace lavinias_son_older_than_daughter_l151_151719

def katies_daughter_age := 12
def lavinias_daughter_age := katies_daughter_age - 10
def lavinias_son_age := 2 * katies_daughter_age

theorem lavinias_son_older_than_daughter :
  lavinias_son_age - lavinias_daughter_age = 22 :=
by
  sorry

end lavinias_son_older_than_daughter_l151_151719


namespace incorrect_judgment_D_l151_151835

-- Definitions based on given conditions
variable (a b m : ℝ) (p q : Prop)

-- Lean statements for the given propositions
def optionA := (a * m ^ 2 < b * m ^ 2) → (a < b)
def optionB := ¬ (∀ x : ℝ, x^3 - x^2 - 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 - 1 > 0
def optionC := (¬p ∧ ¬q) → ¬ (p ∧ q)
def optionD_wrong := (x : ℝ) → (¬ (x = 1 ∨ x = -1)) → (x^2 ≠ 1)

-- The proof that the judgment in option D is incorrect
theorem incorrect_judgment_D (x : ℝ) : optionD_wrong x ↔ (¬ (x = 1 ∧ x ≠ -1)) := sorry

end incorrect_judgment_D_l151_151835


namespace ellipse_equation_midpoint_condition_l151_151661

noncomputable def ellipse_params (a b : ℝ) (ha : a > b) (hb : b > 0) : Prop :=
∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

noncomputable def circle_params (b : ℝ) (hb : b > 0) : Prop :=
∀ x y : ℝ, x^2 + y^2 = b^2

noncomputable def points_on_ray (x y : ℝ) (h : x ≥ 0) : Prop :=
y = x

noncomputable def distances (OA OB : ℝ) : Prop :=
OA = 2 * (√10) / 5 ∧ OB = 1

theorem ellipse_equation (a b : ℝ) (ha : a > b) (hb : b > 0) (OA OB : ℝ) (h_distances : distances OA OB) :
  ellipse_params a b ha hb :=
sorry

theorem midpoint_condition (a b k : ℝ) (ha : a > b) (hb : b > 0) (area : ℝ) (h_area : area = 1) (x0 y0 : ℝ) (h_midpoint : x0 = -4*k*t/(1 + 4*k^2) ∧ y0 = t/(1 + 4*k^2)) :
  x0^2 + 4*y0^2 = 2 :=
sorry

end ellipse_equation_midpoint_condition_l151_151661


namespace points_leq_half_in_eq_triangle_l151_151126

theorem points_leq_half_in_eq_triangle (α : Type*) 
  [metric_space α] (triangle : set (point α))
  (h1 : equilateral triangle ∧ ∀ p ∈ triangle, ∃ A B C : point α,
         dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) 
  (points : finset (point α))
  (h2 : finset.card points = 5 ∧ ∀ p ∈ points, p ∈ triangle) :
  ∃ p q ∈ points, p ≠ q ∧ dist p q ≤ 0.5 :=
sorry

end points_leq_half_in_eq_triangle_l151_151126


namespace find_a_l151_151651

noncomputable def set_A : Set ℤ := { x | x^2 + x - 6 = 0 }
noncomputable def set_B (a : ℤ) : Set ℤ := { x | a * x + 1 = 0 }
def possible_values (a : ℤ) : Prop :=
  ∀ x, x ∈ set_B(a) → x ∈ set_A

theorem find_a (a : ℤ) :
  possible_values a ↔ (a = 0 ∨ a = 1/3 ∨ a = -1/2) :=
sorry

end find_a_l151_151651


namespace intersection_M_N_l151_151013

open Set

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 0 < x}
def intersection := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l151_151013


namespace find_rotation_angle_l151_151660

noncomputable def rotation_transform_matrix : Matrix (Fin 2) (Fin 2) ℝ := 
  ![\[-(Real.sqrt 3) / 2, -1 / 2\], \[1 / 2, -(Real.sqrt 3) / 2\]]

theorem find_rotation_angle (θ : ℝ) (hθ : θ ∈ Icc 0 (2 * Real.pi)) :
  cos θ = -(Real.sqrt 3) / 2 ∧ sin θ = 1 / 2 → θ = 5 * Real.pi / 6 :=
by
  intro hcos_sin
  obtain ⟨hcos, hsin⟩ := hcos_sin
  -- the next proof step is omitted in context to only write the statement as per user's instructions.
  sorry

end find_rotation_angle_l151_151660


namespace transformed_polynomial_roots_l151_151030

theorem transformed_polynomial_roots (a b c d : ℝ) 
  (h1 : a + b + c + d = 0)
  (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h3 : a * b * c * d ≠ 0)
  (h4 : Polynomial.eval a (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h5 : Polynomial.eval b (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h6 : Polynomial.eval c (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h7 : Polynomial.eval d (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0):
  Polynomial.eval (-2 / d^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / c^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / b^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / a^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 :=
sorry

end transformed_polynomial_roots_l151_151030


namespace determine_polygon_is_isosceles_l151_151347

def line1 (x : ℝ) : ℝ := 4 * x + 3
def line2 (x : ℝ) : ℝ := -4 * x + 3
def line3 (y : ℝ) : ℝ := -3

theorem determine_polygon_is_isosceles:
  (∃ (x1 x2 x3 : ℝ) (y1 y2 y3 : ℝ),
    line1 x1 = y1 ∧ line2 x1 = y1 ∧  -- Intersection of y = 4x + 3 and y = -4x + 3
    line1 x2 = y2 ∧ line3 y2 = -3 ∧  -- Intersection of y = 4x + 3 and y = -3
    line2 x3 = y3 ∧ line3 y3 = -3 ∧  -- Intersection of y = -4x + 3 and y = -3
    y1 = 3 ∧ x1 = 0 ∧                -- Point (0, 3)
    y2 = -3 ∧ x2 = -3/2 ∧            -- Point (-3/2, -3)
    y3 = -3 ∧ x3 = 3/2)              -- Point (3/2, -3)
  → IsoscelesTriangle) :=
sorry

end determine_polygon_is_isosceles_l151_151347


namespace greatest_common_multiple_of_10_and_15_lt_120_l151_151252

theorem greatest_common_multiple_of_10_and_15_lt_120 : 
  ∃ (m : ℕ), lcm 10 15 = 30 ∧ m ∈ {i | i < 120 ∧ ∃ (k : ℕ), i = k * 30} ∧ m = 90 := 
sorry

end greatest_common_multiple_of_10_and_15_lt_120_l151_151252


namespace sec_150_eq_l151_151472

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151472


namespace pipeA_fills_tank_in_56_minutes_l151_151125

-- Define the relevant variables and conditions.
variable (t : ℕ) -- Time for Pipe A to fill the tank in minutes

-- Condition: Pipe B fills the tank 7 times faster than Pipe A
def pipeB_time (t : ℕ) := t / 7

-- Combined rate of Pipe A and Pipe B filling the tank in 7 minutes
def combined_rate (t : ℕ) := (1 / t) + (1 / pipeB_time t)

-- Given the combined rate fills the tank in 7 minutes
def combined_rate_equals (t : ℕ) := combined_rate t = 1 / 7

-- The proof statement
theorem pipeA_fills_tank_in_56_minutes (t : ℕ) (h : combined_rate_equals t) : t = 56 :=
sorry

end pipeA_fills_tank_in_56_minutes_l151_151125


namespace carpet_length_is_9_l151_151328

noncomputable def carpet_length (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) : ℝ :=
  living_room_area * coverage / width

theorem carpet_length_is_9 (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) (length := carpet_length width living_room_area coverage) :
    width = 4 → living_room_area = 48 → coverage = 0.75 → length = 9 := by
  intros
  sorry

end carpet_length_is_9_l151_151328


namespace first_player_win_except_1x1_l151_151708

theorem first_player_win_except_1x1 (m n : ℕ) (h : m ≥ 1 ∧ n ≥ 1) :
  (m = 1 ∧ n = 1) ∨
  (∃ f : fin m → fin n → Prop, ∃ g : fin m → fin n → Prop,
    (∀ i j, f i j → g i j) ∧
    (∀ i j, g i j → f i j) ∧
    (∀ i j, i ≠ j → ¬(f i j ∧ g j i)) ∧
    (∃ i, (∀ j, ¬g j i) ∧ (∃ j, f i j))) :=
by {
  sorry
}

end first_player_win_except_1x1_l151_151708


namespace conclusion_l151_151977

-- Assuming U is the universal set and Predicates represent Mems, Ens, and Veens
variable (U : Type)
variable (Mem : U → Prop)
variable (En : U → Prop)
variable (Veen : U → Prop)

-- Hypotheses
variable (h1 : ∀ x, Mem x → En x)          -- Hypothesis I: All Mems are Ens
variable (h2 : ∀ x, En x → ¬Veen x)        -- Hypothesis II: No Ens are Veens

-- To be proven
theorem conclusion (x : U) : (Mem x → ¬Veen x) ∧ (Mem x → ¬Veen x) := sorry

end conclusion_l151_151977


namespace prove_statements_l151_151648

def vector3 := (ℝ × ℝ × ℝ)

def O : vector3 := (0, 0, 0)
def A : vector3 := (4, 3, 0)
def B : vector3 := (-3, 0, 4)
def C : vector3 := (5, 6, 4)

def dot_product (v1 v2 : vector3) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : vector3) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def cosine_angle (v1 v2 : vector3) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

def point_to_line_distance (p a b : vector3) : ℝ :=
  let ap := (p.1 - a.1, p.2 - a.2, p.3 - a.3)
  let ab := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let ab_mag := magnitude ab
  (magnitude (ap.1 - (dot_product ap ab / ab_mag ^ 2) * ab.1,
              ap.2 - (dot_product ap ab / ab_mag ^ 2) * ab.2,
              ap.3 - (dot_product ap ab / ab_mag ^ 2) * ab.3)) / ab_mag

def are_coplanar (a b c d : vector3) : Prop :=
  let ab := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let ac := (c.1 - a.1, c.2 - a.2, c.3 - a.3)
  let ad := (d.1 - a.1, d.2 - a.2, d.3 - a.3)
  let determinant := ab.1 * (ac.2 * ad.3 - ac.3 * ad.2) -
                     ab.2 * (ac.1 * ad.3 - ac.3 * ad.1) +
                     ab.3 * (ac.1 * ad.2 - ac.2 * ad.1)
  determinant = 0

theorem prove_statements :
  dot_product A B = -12 ∧
  cosine_angle A B = -12 / 25 ∧
  point_to_line_distance O B C ≠ real.sqrt 5 ∧
  are_coplanar O A B C :=
by 
  sorry

end prove_statements_l151_151648


namespace sum_reciprocals_bound_l151_151636

-- Defining the sequence according to the given problem
def sequence (a : ℝ) (a0 : ℝ) (a1 : ℝ) (n : ℕ) (a_seq : ℕ → ℝ) : ℕ → ℝ
| 0 := a0
| 1 := a
| (n + 2) := ((a_seq (n + 1))^2 / (a_seq n)^2 - 2) * (a_seq (n + 1))

-- Defining the sum of reciprocals of the sequence up to index k
def sum_reciprocals (a_seq : ℕ → ℝ) (k : ℕ) : ℝ :=
∑ i in Finset.range (k + 1), 1 / a_seq i

-- Statement of the theorem
theorem sum_reciprocals_bound (a : ℝ) (k : ℕ) : 
  a > 2 → 
  (sequence a 1 a).sum_reciprocals k < 1 / 2 * (2 + a - Real.sqrt (a^2 - 4)) :=
by sorry

end sum_reciprocals_bound_l151_151636


namespace football_league_games_l151_151153

-- Definitions based on the problem conditions
def num_divisions : ℕ := 2
def num_teams_per_division : ℕ := 9
def num_intra_division_games_per_pair : ℕ := 2
def num_inter_division_games_per_pair : ℕ := 2

-- Lean 4 statement for the theorem
theorem football_league_games : 
  num_divisions = 2 →
  num_teams_per_division = 9 →
  num_intra_division_games_per_pair = 2 →
  num_inter_division_games_per_pair = 2 →
  let intra_division_games := (num_teams_per_division - 1) * num_intra_division_games_per_pair * num_teams_per_division in
  let intra_division_total := num_divisions * intra_division_games in
  let inter_division_games :=
    (num_teams_per_division * num_inter_division_games_per_pair) * num_teams_per_division in
  let inter_division_total := inter_division_games in
  intra_division_total + inter_division_total = 450 :=
by
  intros _ _ _ _
  let intra_division_games := (num_teams_per_division - 1) * num_intra_division_games_per_pair * num_teams_per_division
  let intra_division_total := num_divisions * intra_division_games
  let inter_division_games :=
    (num_teams_per_division * num_inter_division_games_per_pair) * num_teams_per_division
  let inter_division_total := inter_division_games
  show intra_division_total + inter_division_total = 450
  sorry

end football_league_games_l151_151153


namespace opposite_sides_of_line_l151_151689

theorem opposite_sides_of_line (m : ℝ) 
  (ha : (m + 0 - 1) * (2 + m - 1) < 0): 
  -1 < m ∧ m < 1 :=
sorry

end opposite_sides_of_line_l151_151689


namespace number_of_correct_statements_l151_151884

theorem number_of_correct_statements 
  (h1 : ∀ xs : List ℝ, ∀ c : ℝ, (xs.map (fun x => x - c)).Mean ≠ xs.Mean ∨ 
        (xs.map (fun x => x - c)).Variance = xs.Variance)
  (h2 : ¬ stratified_sampling 50 (fun _ => true))
  (h3 : let X := NormalDist 3 1 in 
        P (fun x => 2 ≤ x ∧ x ≤ 4) 0.6826 → P (fun x => x > 4) 0.1587)
  (h4 : stratified_sample_correct 350 7 250 5 150 3 15) :
  number_of_correct_statements [h1, h2, h3, h4] = 2 := by 
    sorry

end number_of_correct_statements_l151_151884


namespace sec_150_l151_151492

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151492


namespace tangent_line_eqn_range_of_a_l151_151973

-- Define the function f(x) with the parameter a
def f (a : ℝ) (x : ℝ) := x^2 - 2 * (a + 1) * x + 2 * a * (Real.log x)

-- Condition: a > 0
variable {a : ℝ} (ha : a > 0)

-- Tangent line problem statement
theorem tangent_line_eqn :
  f 1 1 = -3 ∧ Deriv (f 1) 1 = 0 → ∀ y, y = -3 :=
  by sorry

-- Range of a problem statement
theorem range_of_a :
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 1 → f a x ≤ 0) ↔ a ≥ (Real.exp 2 - 2 * Real.exp 1) / (2 * Real.exp 1 - 2) :=
  by sorry

end tangent_line_eqn_range_of_a_l151_151973


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151563

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151563


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151512

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151512


namespace correct_inequality_l151_151834

theorem correct_inequality : -2 < (-1)^3 ∧ (-1)^3 < (-0.6)^2 := by
  -- Conditions derived from the problem
  have h1 : (-1)^3 = -1 := by norm_num
  have h2 : (-0.6)^2 = 0.36 := by norm_num
  
  -- Prove the inequality parts
  have part1 : -2 < -1 := by linarith
  have part2 : -1 < 0.36 := by linarith
  
  -- Combine the parts into the final inequality
  exact ⟨part1, part2⟩

end correct_inequality_l151_151834


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151447

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151447


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151567

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151567


namespace find_monic_polynomial_roots_l151_151736

theorem find_monic_polynomial_roots (r1 r2 r3 : ℝ) :
  (Polynomial.map (algebraMap ℤ ℝ) (Polynomial.X ^ 3 - 3 * Polynomial.X ^ 2 + 5)).is_root r1 ∧
  (Polynomial.map (algebraMap ℤ ℝ) (Polynomial.X ^ 3 - 3 * Polynomial.X ^ 2 + 5)).is_root r2 ∧
  (Polynomial.map (algebraMap ℤ ℝ) (Polynomial.X ^ 3 - 3 * Polynomial.X ^ 2 + 5)).is_root r3 →
  (Polynomial.map (algebraMap ℤ ℝ) (Polynomial.X ^ 3 - 9 * Polynomial.X ^ 2 + 135)).is_root (3 * r1) ∧
  (Polynomial.map (algebraMap ℤ ℝ) (Polynomial.X ^ 3 - 9 * Polynomial.X ^ 2 + 135)).is_root (3 * r2) ∧
  (Polynomial.map (algebraMap ℤ ℝ) (Polynomial.X ^ 3 - 9 * Polynomial.X ^ 2 + 135)).is_root (3 * r3) :=
by
  sorry

end find_monic_polynomial_roots_l151_151736


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151428

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151428


namespace sum_of_sequence_2019_l151_151167

noncomputable def a (n : ℕ) : ℝ := n * (Real.sin (n * Real.pi / 2) + Real.cos (n * Real.pi / 2))

noncomputable def sequence_sum (n : ℕ) : ℝ := (Finset.range n).sum (λ i, a (i + 1))

theorem sum_of_sequence_2019 : sequence_sum 2019 = -2020 := 
by
  -- skip the proof
  sorry

end sum_of_sequence_2019_l151_151167


namespace solve_trig_equation_l151_151147

theorem solve_trig_equation (x : ℝ) :
  0 ≤ x ∧ x < 2 * π ∧ (sin x + cos x = cos (2 * x) / (1 - 2 * sin x)) ↔
  x = 0 ∨ x = 3 * π / 4 ∨ x = 3 * π / 2 ∨ x = 7 * π / 4 :=
sorry

end solve_trig_equation_l151_151147


namespace find_moles_AgNO3_l151_151604

-- Given conditions
variables (a : ℝ) (NaOH : ℝ) (AgOH : ℝ)

-- Let the number of moles of NaOH be 3 and the number of moles of AgOH formed be 3
axiom NaOH_eq : NaOH = 3
axiom AgOH_eq : AgOH = 3

-- Define a condition that models the balanced chemical equation (1:1 molar ratio)
axiom reaction_eq : AgOH = NaOH

-- We need to prove that the number of moles of AgNO3 combined (a) is 3
theorem find_moles_AgNO3 : a = 3 :=
by
  rw [NaOH_eq, AgOH_eq, reaction_eq]
  sorry

end find_moles_AgNO3_l151_151604


namespace y_value_l151_151613

theorem y_value (x y : ℤ) (h1 : list.sorted (≤) [2, 5, x, 10, y]) (h2 : x = 7) (h3 : (2 + 5 + x + 10 + y) / 5 = 8) : y = 16 :=
by {
  sorry
}

end y_value_l151_151613


namespace line_equation_point_slope_l151_151782

theorem line_equation_point_slope (A : Point) (m : ℝ) (hA : A = (1, 1)) (hm : m = -3) :
  ∃ (A B C : ℝ), A = 3 ∧ B = 1 ∧ C = -4 ∧ (∀ x y, y - 1 = -3 * (x - 1) → A * x + B * y + C = 0) := 
by
  use 3, 1, -4
  split; [refl, split; [refl, split; [refl, intros x y h, rw h, ring]]]

end line_equation_point_slope_l151_151782


namespace find_q_eqn_l151_151150

noncomputable def q (x : ℝ) : ℝ := x^3 - 186/13 * x^2 + 1836/13 * x - 108

theorem find_q_eqn (q : ℝ → ℝ) (h1 : monic q ∧ degree q = 3) 
  (h2 : q (3 - 2 * Complex.I) = 0)
  (h3 : q 0 = -108) :
  q = λ x, x^3 - (186 / 13) * x^2 + (1836 / 13) * x - 108 :=
by {
  sorry
}

end find_q_eqn_l151_151150


namespace union_of_sets_l151_151676

open Set

theorem union_of_sets :
  ∀ (P Q : Set ℕ), P = {1, 2} → Q = {2, 3} → P ∪ Q = {1, 2, 3} :=
by
  intros P Q hP hQ
  rw [hP, hQ]
  exact sorry

end union_of_sets_l151_151676


namespace rectangle_area_l151_151289

-- Conditions
def radius : ℝ := 6
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def ratio_length_to_width : ℝ := 3

-- Given the ratio of the length to the width is 3:1
def length : ℝ := ratio_length_to_width * width

-- Theorem stating the area of the rectangle
theorem rectangle_area :
  let area := length * width
  area = 432 := by
    sorry

end rectangle_area_l151_151289


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151431

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151431


namespace inequality_solution_set_l151_151035

theorem inequality_solution_set 
  (a b c : ℝ)
  (h1 : ∀ x, -1 < x ∧ x < 2 → ax^2 + bx + c > 0) :
  (∀ x, 0 < x ∧ x < 3 → a(x^2 + 1) + b(x - 1) + c > 2ax) :=
by
  sorry

end inequality_solution_set_l151_151035


namespace sec_150_eq_l151_151473

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151473


namespace solution_correct_l151_151596

noncomputable def solution_set : Set ℝ :=
  {x | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)}

theorem solution_correct (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2))) < 1 / 4 ↔ (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x) :=
by sorry

end solution_correct_l151_151596


namespace number_of_violinists_l151_151752

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

end number_of_violinists_l151_151752


namespace a_1_eq_3_recurrence_relation_a_2014_eq_4_l151_151063

-- Define the sequence {a_n} with the given conditions
def a : ℕ → ℝ
| 0       := 3
| (n + 1) := 2 / (a n - 2) + 2

-- Prove that a₁ = 3
theorem a_1_eq_3 : a 0 = 3 := rfl

-- Prove the given recurrence relation
theorem recurrence_relation (n : ℕ) : (a (n + 1) - 2) * (a n - 2) = 2 := by
sorry

-- Main theorem: Prove that a_{2014} = 4
theorem a_2014_eq_4 : a 2013 = 4 := by
sorry

end a_1_eq_3_recurrence_relation_a_2014_eq_4_l151_151063


namespace current_price_l151_151714

-- Definitions for the conditions
def original_price : ℝ := 0.45
def saved_amount : ℝ := 30 * original_price
def models_now : ℕ := 27

-- Goal: Prove that the current price per model is $0.50
theorem current_price (original_price models_now : ℝ) (saved_amount : ℝ) :
  saved_amount = 13.5 ∧ models_now = 27 → saved_amount / models_now = 0.50 :=
by
  intro h
  cases h with h₁ h₂
  rw h₁
  rw h₂
  norm_num
  sorry -- Proof placeholder

end current_price_l151_151714


namespace odd_multiples_digit_5_l151_151071

def all_fives (n : ℕ) : Prop :=
  n.digits 10 = List.repeat 5 n.digits.length

theorem odd_multiples_digit_5 (n : ℤ) (odd_n : n % 2 = 1) :
  ¬ ∃ k : ℕ, all_fives (k * n.natAbs) :=
by
  sorry

end odd_multiples_digit_5_l151_151071


namespace draw_five_segments_create_shapes_l151_151314

theorem draw_five_segments_create_shapes :
  ∃ (p : ℝ²) (pts : fin 5 → ℝ²) (segments : list (ℝ² × ℝ²)),
    is_circle p ∧ 
    (∀ i, pts i ∈ (circle_points p)) ∧ 
    (segments.length = 5) ∧ 
    (∀ seg ∈ segments, ∃ i j, seg = (pts i, pts j)) ∧ 
    (has_region_set_with_shapes p segments 1 pentagon 2 quadrilateral) :=
sorry

end draw_five_segments_create_shapes_l151_151314


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151538

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151538


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151443

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151443


namespace distribute_balls_equally_l151_151801

/--
Given 20 different colors of balls, each with at least 10 balls, and a total of 800 balls. 
Each box contains at least 10 balls and all balls in a box are of the same color.
Prove that there exists a way to distribute these boxes among 20 students 
so that each student receives the same number of balls.
-/
theorem distribute_balls_equally (colors : ℕ) (balls_per_color : ℕ → ℕ) 
  (total_balls : ℕ) (students : ℕ) (boxes : ℕ → ℕ) :
  colors = 20 →
  (∀ c, balls_per_color c ≥ 10) →
  (∑ c in finset.range colors, balls_per_color c) = 800 →
  (∀ b, boxes b ≥ 10) →
  students = 20 →
  ∃ (distribution : ℕ → ℕ), 
    (∀ s, (∑ b in finset.range (boxes s), boxes b) = (total_balls / students)) :=
begin
  -- Sorry filled to skip proof details
  sorry,
end

end distribute_balls_equally_l151_151801


namespace number_of_possible_schedules_l151_151756

-- Define the six teams
inductive Team : Type
| A | B | C | D | E | F

open Team

-- Define the function to get the number of different schedules possible
noncomputable def number_of_schedules : ℕ := 70

-- Define the theorem statement
theorem number_of_possible_schedules (teams : Finset Team) (play_games : Team → Finset Team) (h : teams.card = 6) 
  (h2 : ∀ t ∈ teams, (play_games t).card = 3 ∧ ∀ t' ∈ (play_games t), t ≠ t') : 
  number_of_schedules = 70 :=
by sorry

end number_of_possible_schedules_l151_151756


namespace measure_of_angle_AIE_l151_151709

-- Defining the problem conditions
variable (ABC : Type)
variable [triangle ABC] (AD BE CF : segment) (I : point)
variable [angleBisector AD] [angleBisector BE] [angleBisector CF] (A B C : point)
variable (angleACB : angle) (angleACB_value : angleACB = 38)

-- Definition of the angles involved
def angleBAC := ∠A B C
def angleABC := ∠A B C
def angleAIE := ∠A I E

-- The proof statement
theorem measure_of_angle_AIE : angleAIE = 71 :=
by
  sorry

end measure_of_angle_AIE_l151_151709


namespace find_b_squared_l151_151780

-- Definitions and conditions
def ellipse_foci_coincide (b : ℝ) : Prop :=
  ∃ (c : ℝ), c = (Real.sqrt 41) / 2 ∧
  (∀ x y, x ^ 2 / 25 + y ^ 2 / b ^ 2 = 1 → (x, y) = (±c, 0))

def hyperbola_foci_coincide : Prop :=
  ∃ (c : ℝ), c = (Real.sqrt 41) / 2 ∧
  (∀ x y, x ^ 2 / 100 - y ^ 2 / 64 = 1 / 16 → (x, y) = (±c, 0))

-- Theorem to prove
theorem find_b_squared (b : ℝ) (h_ellipse : ellipse_foci_coincide b) (h_hyperbola : hyperbola_foci_coincide) :
  b ^ 2 = 14.75 :=
sorry

end find_b_squared_l151_151780


namespace solution_set_for_inequality_l151_151196

theorem solution_set_for_inequality : 
  { x : ℝ | x * (x - 1) < 2 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_for_inequality_l151_151196


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151498

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151498


namespace inequality_correct_l151_151940

theorem inequality_correct (a b : ℝ) (h : a - |b| > 0) : a + b > 0 :=
sorry

end inequality_correct_l151_151940


namespace round_robin_tournament_l151_151048

open Finset

noncomputable def calculateSets (n : ℕ) : ℕ :=
  2 * (n * (n - 1) * (n - 2) / 6) / 2

theorem round_robin_tournament :
  ∀ (n : ℕ), (∀ (A B C : ℤ), 1 ≤ A ∧ A < n ∧ 1 ≤ B ∧ B < n ∧ 1 ≤ C ∧ C < n ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A) →
    ∀ (games_won games_lost : ℕ), 
    (∀ (team : ℕ), team < n → (games_won = 12 ∧ games_lost = 8)) →
    (n = 21) →
    (calculateSets n = 665) := by sorry

end round_robin_tournament_l151_151048


namespace cost_of_cheaper_feed_l151_151810

theorem cost_of_cheaper_feed (C : ℝ) 
  (h1 : 35 * 0.36 = 12.6)
  (h2 : 18 * 0.53 = 9.54)
  (h3 : 17 * C + 9.54 = 12.6) :
  C = 0.18 := sorry

end cost_of_cheaper_feed_l151_151810


namespace sec_150_eq_l151_151400

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151400


namespace parametric_eq_and_max_value_l151_151009

noncomputable def polar_eq (θ : ℝ) (ρ : ℝ) : Prop :=
  ρ = 2 * (sin θ + cos θ + 1/ρ)

theorem parametric_eq_and_max_value :
  (∀ (θ : ℝ), ∃ (x y : ℝ), x = 1 + 2 * cos θ ∧ y = 1 + 2 * sin θ ∧
                         polar_eq θ (real.sqrt (x^2 + y^2))) ∧
  (∀ (P : ℝ × ℝ), ((∃ θ : ℝ, P = (1 + 2 * cos θ, 1 + 2 * sin θ)) →
    3 * P.1 + 4 * P.2 ≤ 17)) :=
by
  sorry

end parametric_eq_and_max_value_l151_151009


namespace convert_speed_l151_151919

theorem convert_speed : 
  ∀ (v_km_per_hr : ℚ) (conversion_factor : ℚ),
  v_km_per_hr = 126 ∧ conversion_factor = 0.277778 → 
  (v_km_per_hr * conversion_factor).round = 35 := 
by 
  intro v_km_per_hr conversion_factor 
  intro h
  cases h with h1 h2
  -- This is a placeholder for the actual proof
  sorry

end convert_speed_l151_151919


namespace find_ab_range_of_c_l151_151666

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem find_ab {c : ℝ} (h : ∀ x, f a b c x has an extremum at x = 2) 
  (h_val : f a b c 2 = c - 16) : a = 1 ∧ b = -12 := by
sorry

theorem range_of_c {c : ℝ} (h_zeros : ∀ x, ∃ c, (λ x, x^3 - 12x + c) x = 0) 
  (h3_zeros : \# (roots (λ x, x^3 - 12x + c)) = 3) : -16 < c ∧ c < 16 := by
sorry

end find_ab_range_of_c_l151_151666


namespace angle_bisector_second_quadrant_set_l151_151694

theorem angle_bisector_second_quadrant_set (α : ℝ) :
  (α = 3 * Real.pi / 4 + 2 * k * Real.pi ∀ k : ℤ) ↔
  (∃ (k : ℤ), α = 3 * Real.pi / 4 + 2 * k * Real.pi) := by
  sorry

end angle_bisector_second_quadrant_set_l151_151694


namespace remainder_of_b_mod_13_l151_151728

theorem remainder_of_b_mod_13:
  (∃ b : ℕ, b ≡ (2⁻¹ + 3⁻¹ + 5⁻¹)⁻¹ [MOD 13]) → 
  ∃ b : ℕ, b % 13 = 6 :=
begin
  sorry
end

end remainder_of_b_mod_13_l151_151728


namespace pump_filling_time_without_leak_l151_151309

/-- 
A pump can fill a tank with water in a certain time \( t \). Because of a leak, it took \( 2 \frac{1}{8} \) hours to fill the tank. 
The leak can drain all the water of the tank in 34 hours.
Prove that the pump alone takes 2 hours to fill the tank.
-/
theorem pump_filling_time_without_leak
  (h1 : (1:ℝ) / 2 + 1 / 8 = 8 / 17)
  (h2 : (1:ℝ) / 34)
  : (2:ℝ) := 
sorry

end pump_filling_time_without_leak_l151_151309


namespace chuck_play_area_l151_151900

-- Define the constants and conditions
def shed_length : ℝ := 3
def shed_width : ℝ := 4
def leash_length : ℝ := 4

-- Define the main area calculation
def main_area : ℝ := (3/4) * Real.pi * (leash_length ^ 2)

-- Define the additional area calculation
def additional_area : ℝ := (1/4) * Real.pi * (1 ^ 2)

-- Define the total area
def total_area : ℝ := main_area + additional_area

-- The theorem we want to prove
theorem chuck_play_area : total_area = (49/4) * Real.pi :=
by
  sorry

end chuck_play_area_l151_151900


namespace equidistant_point_intersection_of_perpendicular_bisectors_l151_151193

/-
Prove that the point equidistant from all three vertices of a triangle 
is the intersection of the perpendicular bisectors of the sides of the triangle,
given that a point equidistant from the endpoints of a segment 
lies on the perpendicular bisector of that segment.
-/
theorem equidistant_point_intersection_of_perpendicular_bisectors 
  {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (triangle : Triangle A B C) :
  ∃ P,
    (∀ v ∈ {triangle.v1, triangle.v2, triangle.v3}, dist P v = dist P (triangle.v1)) ∧
    (P ∈ perpendicular_bisector (segment triangle.v1 triangle.v2)) ∧
    (P ∈ perpendicular_bisector (segment triangle.v2 triangle.v3)) ∧
    (P ∈ perpendicular_bisector (segment triangle.v3 triangle.v1)) :=
sorry

end equidistant_point_intersection_of_perpendicular_bisectors_l151_151193


namespace gcd_multiple_less_than_120_l151_151249

theorem gcd_multiple_less_than_120 (n : ℕ) (h1 : n < 120) (h2 : n % 10 = 0) (h3 : n % 15 = 0) : n ≤ 90 :=
by {
  sorry
}

end gcd_multiple_less_than_120_l151_151249


namespace find_a1_geometric_sequence_l151_151966

theorem find_a1_geometric_sequence (a₁ q : ℝ) (h1 : q ≠ 1) 
    (h2 : a₁ * (1 - q^3) / (1 - q) = 7)
    (h3 : a₁ * (1 - q^6) / (1 - q) = 63) :
    a₁ = 1 :=
by
  sorry

end find_a1_geometric_sequence_l151_151966


namespace angle_BEC_l151_151764

theorem angle_BEC (A B C E : Point) :
  congruent ⟨A, B, C⟩ ⟨A, B, E⟩ ∧ (AB = AC ∧ AC = AE) ∧ angle B A C = 30° ⟹ 
  angle B E C = 150° :=
begin
  sorry,
end

end angle_BEC_l151_151764


namespace problem1_problem2_l151_151887

variables {A B C D E H O G K : Point}
variables {BC : Line} (BC_perp_AD : Perpendicular AD BC)

-- Definitions based on problem conditions
def is_circumcenter (O : Point) (Δ : Triangle) : Prop := -- Define circumcenter condition
sorry

def is_orthocenter (H : Point) (Δ : Triangle) : Prop := -- Define orthocenter condition
sorry

def is_midpoint (G : Point) (A H : Point) : Prop := -- Define midpoint condition
sorry

def on_segment (K : Point) (G H : Point) : Prop := -- Define on segment condition
sorry

def extend_KO (O : Point) (K : Point) : Line := -- Define extension of KO to intersection
sorry

def intersects (ℓ : Line) (AB : Line) (E : Point) : Prop := -- Define intersection condition
sorry

variables
(Δ : Triangle) -- Acute triangle ABC
(h1 : is_circumcenter O Δ)
(h2 : is_orthocenter H Δ)
(h3 : BC_perp_AD)
(h4 : is_midpoint G A H)
(h5 : on_segment K G H)
(h6 : extend_KO O K = ℓKO)
(h7 : intersects ℓKO AB E)
(h8 : GK = HD)

-- Target problem statements
theorem problem1 : Parallel EK BC := 
sorry

theorem problem2 : Perpendicular GE GC :=
sorry

end problem1_problem2_l151_151887


namespace opposite_of_neg_two_is_two_l151_151178

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l151_151178


namespace sec_150_eq_l151_151471

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151471


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151505

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151505


namespace cos_product_2pi_over_7_l151_151839

theorem cos_product_2pi_over_7 :
  cos (2 * pi / 7) * cos (4 * pi / 7) * cos (8 * pi / 7) = 1 / 8 :=
by
  sorry

end cos_product_2pi_over_7_l151_151839


namespace range_of_m_l151_151650

def PropositionP (m : ℝ) : Prop := m < 1 / 2

def PropositionQ (m : ℝ) : Prop := m > 2

theorem range_of_m (m : ℝ) :
  (PropositionP m ∧ ¬PropositionQ m) ∨ (¬PropositionP m ∧ PropositionQ m) ↔ m ∈ Set.Ioo (2 : ℝ) (Float.Infinity) ∪ Set.Ioo (Float.NegInfinity) (1 / 2) :=
by
  sorry

end range_of_m_l151_151650


namespace sqrt_defined_iff_l151_151210

theorem sqrt_defined_iff (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_defined_iff_l151_151210


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151449

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151449


namespace find_beta_l151_151843

variables (R1 R2 : ℝ) (alpha beta : ℝ)

-- Constants
def radius_ratio := R2 = 2 * R1
def angle_alpha := alpha = 70 * (π / 180) -- converting degrees to radians
def cos70 := real.cos angle_alpha

-- Expected result
def expected_cos_beta := 1 - cos70
def beta_value := beta ≈ real.arccos expected_cos_beta

theorem find_beta (h1 : radius_ratio) (h2 : angle_alpha) : beta_value :=
sorry

end find_beta_l151_151843


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151525

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151525


namespace question1_effective_purification_16days_question2_min_mass_optimal_purification_l151_151299

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then x^2 / 16 + 2
else if x > 4 then (x + 14) / (2 * x - 2)
else 0

-- Effective Purification Conditions
def effective_purification (m : ℝ) (x : ℝ) : Prop := m * f x ≥ 4

-- Optimal Purification Conditions
def optimal_purification (m : ℝ) (x : ℝ) : Prop := 4 ≤ m * f x ∧ m * f x ≤ 10

-- Proof for Question 1
theorem question1_effective_purification_16days (x : ℝ) (hx : 0 < x ∧ x ≤ 16) :
  effective_purification 4 x :=
by sorry

-- Finding Minimum m for Optimal Purification within 7 days
theorem question2_min_mass_optimal_purification :
  ∃ m : ℝ, (16 / 7 ≤ m ∧ m ≤ 10 / 3) ∧ ∀ (x : ℝ), (0 < x ∧ x ≤ 7) → optimal_purification m x :=
by sorry

end question1_effective_purification_16days_question2_min_mass_optimal_purification_l151_151299


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151421

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151421


namespace range_of_b_l151_151000

-- Define the function f
def f (x b : ℝ) : ℝ := -x^2 + b * Real.log (x + 1)

-- Define the derivative f'
def f_prime (x b : ℝ) : ℝ := -2 * x + b / (x + 1)

-- Define the problem statement
theorem range_of_b (b : ℝ) (h : ∀ x ≥ 0, f_prime x b ≤ 0) : b ∈ set.Iic 0 := by
  -- The proof is omitted
  sorry

end range_of_b_l151_151000


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151550

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151550


namespace angle_between_MN_AD_is_90_l151_151047

noncomputable def vec (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def angle_between (u v : ℝ × ℝ × ℝ) : ℝ :=
  real.acos ((dot_product u v) / (norm u * norm v))

def A : ℝ × ℝ × ℝ := vec 0 0 0
def B : ℝ × ℝ × ℝ := vec 1 0 0
def C : ℝ × ℝ × ℝ := vec 0 1 0
def D : ℝ × ℝ × ℝ := vec 0 0 1

def M : ℝ × ℝ × ℝ := vec (1/2) 0 0
def N : ℝ × ℝ × ℝ := vec 0 0 (1/2)

def AD : ℝ × ℝ × ℝ := vec 0 0 1

theorem angle_between_MN_AD_is_90 :
  angle_between (vec ((N.1 - M.1)) ((N.2 - M.2)) ((N.3 - M.3))) AD = real.pi / 2 :=
sorry

end angle_between_MN_AD_is_90_l151_151047


namespace runners_speed_ratio_l151_151818

/-- Two runners, 20 miles apart, start at the same time, aiming to meet. 
    If they run in the same direction, they meet in 5 hours. 
    If they run towards each other, they meet in 1 hour.
    Prove that the ratio of the speed of the faster runner to the slower runner is 3/2. -/
theorem runners_speed_ratio (v1 v2 : ℝ) (h1 : v1 > v2)
  (h2 : 20 = 5 * (v1 - v2)) 
  (h3 : 20 = (v1 + v2)) : 
  v1 / v2 = 3 / 2 :=
sorry

end runners_speed_ratio_l151_151818


namespace sec_150_eq_l151_151402

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151402


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151580

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151580


namespace find_total_cows_l151_151700

-- Define the conditions given in the problem
def ducks_legs (D : ℕ) : ℕ := 2 * D
def cows_legs (C : ℕ) : ℕ := 4 * C
def total_legs (D C : ℕ) : ℕ := ducks_legs D + cows_legs C
def total_heads (D C : ℕ) : ℕ := D + C

-- State the problem in Lean 4
theorem find_total_cows (D C : ℕ) (h : total_legs D C = 2 * total_heads D C + 32) : C = 16 :=
sorry

end find_total_cows_l151_151700


namespace count_three_digit_numbers_with_4_l151_151992

theorem count_three_digit_numbers_with_4 : 
  (∑ n in { x | 100 ≤ x ∧ x ≤ 999 ∧ (x / 100 = 4 ∨ (x % 100) / 10 = 4 ∨ x % 10 = 4) }, 1) = 271 := 
by
  sorry

end count_three_digit_numbers_with_4_l151_151992


namespace count_distinct_factors_l151_151682

theorem count_distinct_factors :
  let n := 2^8 * 5^3 * 7^2 in
  ∃ (a b c : ℕ), a = 9 ∧ b = 4 ∧ c = 3 ∧ a * b * c = 108 :=
by
  let n := 2^8 * 5^3 * 7^2
  existsi 9, 4, 3
  simp
  sorry

end count_distinct_factors_l151_151682


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151503

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151503


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151379

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151379


namespace problem_statement_l151_151956

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def T : Set ℝ := {x | x < 2}

def set_otimes (A B : Set ℝ) : Set ℝ := {x | x ∈ (A ∪ B) ∧ x ∉ (A ∩ B)}

theorem problem_statement : set_otimes M T = {x | x < -1 ∨ (2 ≤ x ∧ x ≤ 4)} :=
by sorry

end problem_statement_l151_151956


namespace resulting_perimeter_l151_151084

theorem resulting_perimeter (p1 p2 : ℕ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let a := p1 / 4 in
  let b := p2 / 4 in
  p1 + p2 - 2 * a = 120 :=
by
  sorry

end resulting_perimeter_l151_151084


namespace g_neg1_eq_3_l151_151943

-- Definition of odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

-- Given conditions
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)
variable (h_odd : is_odd f)
variable (h_g_def : ∀ x, g(x) = f(x) + 2)
variable (h_g_1 : g 1 = 1)

-- The statement to prove
theorem g_neg1_eq_3 : g (-1) = 3 :=
sorry

end g_neg1_eq_3_l151_151943


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151544

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151544


namespace find_constants_l151_151889

-- Define the conditions of the problem
def y (a b c : ℝ) (x : ℝ) : ℝ := a * Real.cos (b * x + c)

theorem find_constants (a b c : ℝ) (h1 : ∀ x, y a b c x ≤ 3)
  (h2 : y a b c (-π / 4) = 3) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) :
  a = 3 ∧ b = 1 ∧ c = π / 4 :=
sorry

end find_constants_l151_151889


namespace polynomial_degree_equality_l151_151095

theorem polynomial_degree_equality
    (m n : ℕ) (m_gt2 : m > 2) (n_gt2 : n > 2)
    (A B : Polynomial ℂ) (A_nonconstant : ¬A.Coefficients.empty) (B_nonconstant : ¬B.Coefficients.empty)
    (A_deg_gt1_or_B_deg_gt1 : A.degree > 1 ∨ B.degree > 1)
    (deg_Am_Bn_lt_min_mn : (A^m - B^n).degree < min m n) :
  A^m = B^n := 
sorry

end polynomial_degree_equality_l151_151095


namespace snowball_total_distance_l151_151866

noncomputable def total_distance (a1 d n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem snowball_total_distance :
  total_distance 6 5 25 = 1650 := by
  sorry

end snowball_total_distance_l151_151866


namespace unique_positive_integer_n_l151_151343

theorem unique_positive_integer_n (n : ℕ) (h : 3 * 2^3 + 4 * 2^4 + 5 * 2^5 + ∑ (k : ℕ) in finset.range (n - 4), (k + 6) * 2^(k + 6) = 2^(n + 11)) : n = 1025 :=
sorry

end unique_positive_integer_n_l151_151343


namespace min_max_difference_l151_151732

theorem min_max_difference (x y k : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) :
    let expr := (|kx + y| / (|kx| + |y|))
    let m := min expr
    let M := max expr
    M - m = 1 := 
begin
    -- the proof will go here
    sorry
end

end min_max_difference_l151_151732


namespace feb_03_2013_nine_day_l151_151768

-- Definitions of the main dates involved
def dec_21_2012 : Nat := 0  -- Assuming day 0 is Dec 21, 2012
def feb_03_2013 : Nat := 45  -- 45 days after Dec 21, 2012

-- Definition to determine the Nine-day period
def nine_day_period (x : Nat) : (Nat × Nat) :=
  let q := x / 9
  let r := x % 9
  (q + 1, r + 1)

-- Theorem we want to prove
theorem feb_03_2013_nine_day : nine_day_period feb_03_2013 = (5, 9) :=
by
  sorry

end feb_03_2013_nine_day_l151_151768


namespace Lauryn_earnings_l151_151090

variables (L : ℝ)

theorem Lauryn_earnings (h1 : 0.70 * L + L = 3400) : L = 2000 :=
sorry

end Lauryn_earnings_l151_151090


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151425

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151425


namespace sec_150_eq_l151_151470

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151470


namespace hexagon_divided_iff_N_geq_4_l151_151934

theorem hexagon_divided_iff_N_geq_4 (n : ℕ) (h : n > 0) : 
  (∃ (T : Type) (divide : Π (H : Hexagon), Hexagon -> set T -> Prop), divide H (hex T) ↔ n ≥ 4) :=
sorry

end hexagon_divided_iff_N_geq_4_l151_151934


namespace coefficient_of_x4_in_expansion_l151_151234

theorem coefficient_of_x4_in_expansion (x : ℤ) :
  let a := 3
  let b := 2
  let n := 8
  let k := 4
  (finset.sum (finset.range (n + 1)) (λ r, binomial n r * a^r * b^(n-r) * x^r) = 
  ∑ r in finset.range (n + 1), binomial n r * a^r * b^(n - r) * x^r)

  ∑ r in finset.range (n + 1), 
    if r = k then 
      binomial n r * a^r * b^(n-r)
    else 
      0 = 90720
:= 
by
  sorry

end coefficient_of_x4_in_expansion_l151_151234


namespace girls_not_playing_soccer_l151_151841

theorem girls_not_playing_soccer (total_students boys total_playing_soccer : ℕ)
  (percentage_boys_playing_soccer : ℝ)
  (h1 : total_students = 450)
  (h2 : boys = 320)
  (h3 : total_playing_soccer = 250)
  (h4 : percentage_boys_playing_soccer = 0.86) :
  (total_students - boys - (total_playing_soccer - (percentage_boys_playing_soccer * total_playing_soccer).to_nat)) = 95 :=
by
  sorry

end girls_not_playing_soccer_l151_151841


namespace probability_of_drawing_red_ball_l151_151327

def num_yellow_balls := 2
def num_red_balls := 3
def total_balls := num_yellow_balls + num_red_balls

theorem probability_of_drawing_red_ball : total_balls > 0 → (num_red_balls / total_balls) = 3 / 5 :=
by 
  intros h
  have h_total : total_balls = 5 := by norm_num
  rw h_total
  norm_num
  sorry

end probability_of_drawing_red_ball_l151_151327


namespace sec_150_eq_l151_151461

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151461


namespace integer_sqrt_225_minus_cbrt_x_count_l151_151617

theorem integer_sqrt_225_minus_cbrt_x_count :
  (∃ n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ ∃ x : ℝ, x ≥ 0 ∧ sqrt (225 - cbrt x) = n) → 16 :=
by
  sorry

end integer_sqrt_225_minus_cbrt_x_count_l151_151617


namespace total_height_of_rings_l151_151867

theorem total_height_of_rings : 
  let top_diameter := 30
  let top_height := 8
  let bottom_diameter := 10
  let diameter_step := 2
  let height_step := 0.5
  let num_rings := (top_diameter - bottom_diameter) / diameter_step + 1
  let height (n : ℕ) := top_height - n * height_step
  let total_height := ∑ i in Finset.range num_rings, height i
  in total_height = 60 :=
by
  sorry

end total_height_of_rings_l151_151867


namespace shawn_and_karen_time_l151_151267

-- Defining the work rates of Shawn and Karen
def shawn_rate : ℝ := 1 / 18
def karen_rate : ℝ := 1 / 12

-- Combined work rate
def combined_rate : ℝ := shawn_rate + karen_rate

-- Theorem stating the time it takes for Shawn and Karen to paint the house together
theorem shawn_and_karen_time : combined_rate ≠ 0 → (1 / combined_rate = 7.2) :=
by
  -- We'll simply state the proof without solving it here
  sorry

end shawn_and_karen_time_l151_151267


namespace find_largest_number_l151_151955

theorem find_largest_number
  (a b c d : ℕ)
  (h1 : a + b + c = 222)
  (h2 : a + b + d = 208)
  (h3 : a + c + d = 197)
  (h4 : b + c + d = 180) :
  max a (max b (max c d)) = 89 :=
by
  sorry

end find_largest_number_l151_151955


namespace license_plate_increase_l151_151198

-- definitions from conditions
def old_plates_count : ℕ := 26 ^ 2 * 10 ^ 3
def new_plates_count : ℕ := 26 ^ 4 * 10 ^ 2

-- theorem stating the increase in the number of license plates
theorem license_plate_increase : 
  (new_plates_count : ℚ) / (old_plates_count : ℚ) = 26 ^ 2 / 10 :=
by
  sorry

end license_plate_increase_l151_151198


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151532

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151532


namespace percentage_increase_in_gross_revenue_l151_151266

theorem percentage_increase_in_gross_revenue 
  (P R : ℝ) 
  (hP : P > 0) 
  (hR : R > 0) 
  (new_price : ℝ := 0.80 * P) 
  (new_quantity : ℝ := 1.60 * R) : 
  (new_price * new_quantity - P * R) / (P * R) * 100 = 28 := 
by
  sorry

end percentage_increase_in_gross_revenue_l151_151266


namespace dot_product_of_a_and_b_l151_151988

-- Given conditions translated into Lean definitions
def a : ℝ × ℝ := (2, 1)
def b_mag : ℝ := Real.sqrt 3
def a_plus_b_mag : ℝ := 4

-- Norm (magnitude) of vectors using Euclidean norm definition
noncomputable def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- The dot product function
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- The goal: to prove a ⋅ b = 4 under the given conditions.
theorem dot_product_of_a_and_b (b : ℝ × ℝ) (hb_mag : norm b = b_mag) (h_a_plus_b : norm (a.1 + b.1, a.2 + b.2) = a_plus_b_mag) :
  dot a b = 4 := by
  sorry

end dot_product_of_a_and_b_l151_151988


namespace equation_of_ellipse_equation_of_line_l151_151967

noncomputable def semi_major_axis := 2
noncomputable def semi_minor_axis := 1
noncomputable def semi_focal_distance := Real.sqrt 3
noncomputable def eccentricity := Real.sqrt 3 / 2
noncomputable def midpoint_distance := Real.sqrt 5 / 2
noncomputable def area_of_triangle := Real.sqrt 3 / 2

theorem equation_of_ellipse 
  (a b c : ℝ) 
  (h_maj: a = semi_major_axis) 
  (h_min: b = semi_minor_axis) 
  (h_foc: c = semi_focal_distance) 
  (h_ecc: c / a = eccentricity) :
  (x y : ℝ) → (x^2 / a^2) + (y^2 / b^2) = 1 :=
by
  sorry

theorem equation_of_line 
  (h_line : (x, y : ℝ) → y = k * (x + 1)) 
  (h_max_area : area_of_triangle) :
  (x : ℝ) → x = -1 :=
by
  sorry

end equation_of_ellipse_equation_of_line_l151_151967


namespace sec_150_eq_l151_151393

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151393


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151553

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151553


namespace problem_I_problem_II_l151_151658
noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := -1 + 2 * a n
noncomputable def an (n : ℕ) : ℕ := 2^(n-1)
noncomputable def bn (n : ℕ) : ℝ := Real.log2 (an (n+1))
noncomputable def Tn (n : ℕ) : ℝ := (n * (n + 1)) / 2
noncomputable def H (n : ℕ) : ℝ := ∑ k in Finset.range (n+1), 1 / Tn k

-- Statement for (I): Prove that the general term formula for {a_n} is 2^(n-1) given S_n = -1 + 2a_n
theorem problem_I (a : ℕ → ℝ) (n : ℕ) :
  Sn a n = -1 + 2 * a n →
  ∀ n, a n = an n :=
sorry

-- Statement for (II): Prove that the sum of the inverse of T_n is 2n / (n+1)
theorem problem_II (n : ℕ) :
  H n = 2 * (1 - 1 / (n + 1)) →
  (∑ k in Finset.range n, 1 / Tn k) = 2 * n / (n + 1) :=
sorry

end problem_I_problem_II_l151_151658


namespace intelligent_test_failure_prob_maximize_failure_prob_production_improvement_needed_l151_151679

-- Definitions to formalize the conditions
def prob_safe : ℚ := 49 / 50
def prob_energy : ℚ := 48 / 49
def prob_perf : ℚ := 47 / 48
def prob_fail_manual (p : ℚ) : Prop := 0 < p ∧ p < 1

-- Definitions for the main probabilities
def prob_pass_intelligent : ℚ := prob_safe * prob_energy * prob_perf
def prob_fail_intelligent : ℚ := 1 - prob_pass_intelligent

-- Formalize Question 1
theorem intelligent_test_failure_prob :
  prob_fail_intelligent = 3 / 50 :=
begin
  -- proof omitted
  sorry
end

-- Formalize Question 2
def f (p : ℚ) : ℚ := 50 * p * (1 - p) ^ 49

theorem maximize_failure_prob (p : ℚ) (h : prob_fail_manual p) :
  argmax f p = 1 / 50 :=
begin
  -- proof omitted
  sorry
end

-- Formalize Question 3
def overall_pass_rate (p : ℚ) : ℚ :=
  prob_pass_intelligent * (1 - p)

theorem production_improvement_needed (p : ℚ) (h : prob_fail_manual p) :
  p = 1 / 50 → overall_pass_rate p < 93 / 100 :=
begin
  -- proof omitted
  sorry
end

end intelligent_test_failure_prob_maximize_failure_prob_production_improvement_needed_l151_151679


namespace Liu_Bei_vs_Zhang_Fei_daily_consumption_l151_151284

theorem Liu_Bei_vs_Zhang_Fei_daily_consumption :
  (let lb_daily := 1 / 5 in
   let zf_daily := 1 / 4 in
   let difference := |zf_daily - lb_daily| in
   let percentage := 100 * (difference / zf_daily)
   in percentage = 52) :=
by
  sorry

end Liu_Bei_vs_Zhang_Fei_daily_consumption_l151_151284


namespace chess_team_arrangements_l151_151766

def num_arrangements (boys girls : Nat) (alice_end : Bool) (boys_adjacent : Bool) : Nat :=
  if alice_end && boys_adjacent then 
    2 * 4 * 2 * 6
  else 
    0

theorem chess_team_arrangements :
  num_arrangements 2 4 true true = 96 :=
by 
  simp [num_arrangements]
  sorry

end chess_team_arrangements_l151_151766


namespace projection_a_on_b_l151_151630

-- Declare the vectors a and b
def a : Real × Real × Real := (2, 3, 1)
def b : Real × Real × Real := (1, -2, -2)

-- Define the dot product function for 3D vectors
def dot_product (u v : Real × Real × Real) : Real :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the magnitude (norm) function for 3D vectors
def magnitude (v : Real × Real × Real) : Real :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Define the projection function of a vector u onto a vector v
def projection (u v : Real × Real × Real) : Real × Real × Real :=
  let dot_uv := dot_product u v
  let mag_v_squared := (magnitude v)^2
  (dot_uv / mag_v_squared * v.1, dot_uv / mag_v_squared * v.2, dot_uv / mag_v_squared * v.3)

-- State the problem formally: proving the projection of vector a onto vector b
theorem projection_a_on_b :
  projection a b = (-2 / 3) * b := by
  -- Proof will be here
  sorry -- Skip the proof

end projection_a_on_b_l151_151630


namespace probability_non_positive_product_l151_151221

-- Definition of the set S
def S : Set ℤ := {-6, -3, 0, 2, 5, 9}

-- Define the function to select two different elements from the set and calculate the probability of non-positive product.
theorem probability_non_positive_product : 
  (∑ x in S, ∑ y in S, if x ≠ y ∧ x * y ≤ 0 then 1 else 0).toRat / 
  (∑ x in S, ∑ y in S, if x ≠ y then 1 else 0).toRat = 11 / 15 :=
by
  sorry

end probability_non_positive_product_l151_151221


namespace power_mod_equiv_l151_151825

theorem power_mod_equiv :
  7 ^ 145 % 12 = 7 % 12 :=
by
  -- Here the solution would go
  sorry

end power_mod_equiv_l151_151825


namespace opposite_of_neg_two_is_two_l151_151179

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l151_151179


namespace ratio_area_perimeter_equilateral_triangle_l151_151351

theorem ratio_area_perimeter_equilateral_triangle (s : ℝ) (h : s = 6) :
  let area := (math.sqrt 3 / 4) * s^2,
      perimeter := 3 * s
  in area / perimeter = math.sqrt 3 / 2 :=
by
  sorry

end ratio_area_perimeter_equilateral_triangle_l151_151351


namespace sec_150_eq_l151_151475

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151475


namespace coefficient_of_x4_in_expansion_l151_151241

noncomputable def problem_statement : ℕ :=
  let n := 8
  let a := 2
  let b := 3
  let k := 4
  binomial n k * (b ^ k) * (a ^ (n - k))

theorem coefficient_of_x4_in_expansion :
  problem_statement = 90720 :=
by
  sorry

end coefficient_of_x4_in_expansion_l151_151241


namespace integer_pairs_count_l151_151337

/-- 
Prove that the number of ordered pairs of integers (x, y) such that 
10 ≤ x < y ≤ 90 and i^x + i^y is real is 505.
-/
theorem integer_pairs_count : 
  (∃ n : ℕ, (∀ x y : ℤ, (10 ≤ x ∧ x < y ∧ y ≤ 90) ∧ (i ^ x + i ^ y) ∈ ℝ → (x, y) ∈ ℤ × ℤ) ∧ n = 505) :=
by
  sorry

end integer_pairs_count_l151_151337


namespace wrapping_paper_amount_l151_151896

theorem wrapping_paper_amount (x : ℝ) (h : x + (3/4) * x + (x + (3/4) * x) = 7) : x = 2 :=
by
  sorry

end wrapping_paper_amount_l151_151896


namespace sec_150_eq_l151_151467

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151467


namespace find_solution_set_l151_151924

noncomputable def is_solution (x : ℝ) : Prop :=
(1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < 1 / 4

theorem find_solution_set :
  { x : ℝ | is_solution x } = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end find_solution_set_l151_151924


namespace horner_rule_correct_l151_151223

theorem horner_rule_correct (n : ℕ) (a : Fin (n+1) → ℝ) (x0 : ℝ) : 
  let P : ℝ → ℝ := λ x, ∑ i in Fin.range (n+1), a i * x ^ (n - i) in
  let horner_eval : ℝ → ℝ := 
    λ x, 
      (Fin.reverseRange (n + 1)).foldl (λ (v : ℝ) (i : Fin (n + 1)), v * x + a i) 0 in
  P x0 = horner_eval x0 := 
  sorry

end horner_rule_correct_l151_151223


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151367

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151367


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151549

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151549


namespace larger_cross_section_distance_l151_151220

theorem larger_cross_section_distance
  (h_area1 : ℝ)
  (h_area2 : ℝ)
  (dist_planes : ℝ)
  (h_area1_val : h_area1 = 256 * Real.sqrt 2)
  (h_area2_val : h_area2 = 576 * Real.sqrt 2)
  (dist_planes_val : dist_planes = 10) :
  ∃ h : ℝ, h = 30 :=
by
  sorry

end larger_cross_section_distance_l151_151220


namespace equilateral_triangle_on_square_l151_151881

theorem equilateral_triangle_on_square
  (x : ℝ) (h_triangle : ∀ (A B C : ℝ), ∃ (A B C : (ℝ × ℝ)),
    A ∈ square_side 1 ∧ B ∈ square_side 1 ∧ C ∈ square_side 1 ∧ 
    dist A B = x ∧ dist B C = x ∧ dist C A = x) :
  1 ≤ x ∧ x ≤ (Real.sqrt 6 - Real.sqrt 2) :=
sorry

end equilateral_triangle_on_square_l151_151881


namespace find_point_P_l151_151662

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

def F : ℝ × ℝ := (Real.sqrt 3, 0)

theorem find_point_P :
  ∃ (p : ℝ), p > 0 ∧ (∀ A B : ℝ × ℝ, A ≠ B ∧
    (ellipse A.1 A.2) ∧ (ellipse B.1 B.2) ∧
    (A.1 - F.1) * (B.2 - F.2) = (A.2 - F.2) * (B.1 - F.1) → 
    (let P := (p, 0) in
      (A.1 - P.1) * (B.2 - P.2) = (A.2 - P.2) * (B.1 - P.1) ∧
      (A.1 - P.1) * (F.2 - P.2) = (A.2 - P.2) * (F.1 - P.1))
  ) ∧
  p = 2 :=
sorry

end find_point_P_l151_151662


namespace angle_SPU_l151_151066

-- Define the given conditions for the triangle and angles
def triangle (P Q R : Type) : Prop := ∃ P Q R : Type, true

def angle_PRQ := 40
def angle_QRP := 30

-- Define the point S as the foot of the perpendicular from P to RQ
def foot_of_perpendicular (P R Q S : Type) : Prop := ∃ S : Type, true

-- Define T as the center of the circle circumscribed about triangle PQR
def circumcenter (P Q R T : Type) : Prop := ∃ T : Type, true

-- Define U as the other end of the diameter going through P
def diameter_other_end (P U : Type) : Prop := ∃ U : Type, true

-- Assert the desired angle SPU == 70 degrees
theorem angle_SPU (P Q R S T U : Type) 
  (h1 : triangle P Q R) 
  (h2 : foot_of_perpendicular P R Q S) 
  (h3 : circumcenter P Q R T) 
  (h4 : diameter_other_end P U) : 
  angle_SPU = 70 :=
sorry

end angle_SPU_l151_151066


namespace coefficient_of_x3_in_binom_expansion_l151_151056

theorem coefficient_of_x3_in_binom_expansion :
  let term := (x^2 - (2 / x))^6,
      general_term (r : ℕ) := binomial 6 r * ((-2)^r) * (x)^(12 - 3 * r)
  in ∃ r : ℕ, (12 - 3 * r = 3) ∧ (r = 3) ∧ (coefficient_of general_term 3 = -160) := by
  sorry

end coefficient_of_x3_in_binom_expansion_l151_151056


namespace sum_t_result_l151_151892

noncomputable def t (x : ℝ) (a_1 a_2 a_3 a_4 : ℝ) : ℝ :=
  cos (5 * x) + a_4 * cos (4 * x) + a_3 * cos (3 * x) + a_2 * cos (2 * x) + a_1 * cos x

theorem sum_t_result (a_1 a_2 a_3 a_4 : ℝ) : 
  t 0 a_1 a_2 a_3 a_4 - t (π / 5) a_1 a_2 a_3 a_4 + t (2 * π / 5) a_1 a_2 a_3 a_4 - t (3 * π / 5) a_1 a_2 a_3 a_4 +
  t (4 * π / 5) a_1 a_2 a_3 a_4 - t (5 * π / 5) a_1 a_2 a_3 a_4 + t (6 * π / 5) a_1 a_2 a_3 a_4 - t (7 * π / 5) a_1 a_2 a_3 a_4 +
  t (8 * π / 5) a_1 a_2 a_3 a_4 - t (9 * π / 5) a_1 a_2 a_3 a_4 = 6 := 
  sorry

end sum_t_result_l151_151892


namespace max_transportable_weight_by_five_trucks_l151_151224

-- Define the problem's conditions in Lean.
def five_trucks := 5
def truck_capacity := 3 -- in tons
def max_part_weight := 1 -- in tons
def max_transportable_weight := 11.25 -- in tons

-- Lean theorem statement
theorem max_transportable_weight_by_five_trucks :
  ∀ (n trucks: ℕ) (capacity max_weight: ℝ) (cargo: Fin n → ℝ),
    n = five_trucks →
    capacity = truck_capacity →
    (∀ i, cargo i ≤ max_part_weight) →
    (∑ i, cargo i ≤ n * capacity) →
    ∑ i, cargo i = max_transportable_weight :=
sorry

end max_transportable_weight_by_five_trucks_l151_151224


namespace sec_150_eq_l151_151403

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151403


namespace smallest_possible_x_l151_151833

/-- Proof problem: When x is divided by 6, 7, and 8, remainders of 5, 6, and 7 (respectively) are obtained. 
We need to show that the smallest possible positive integer value of x is 167. -/
theorem smallest_possible_x (x : ℕ) (h1 : x % 6 = 5) (h2 : x % 7 = 6) (h3 : x % 8 = 7) : x = 167 :=
by 
  sorry

end smallest_possible_x_l151_151833


namespace distance_between_points_l151_151926

theorem distance_between_points : 
  let p1 := (0, 24)
  let p2 := (10, 0)
  dist p1 p2 = 26 := 
by
  sorry

end distance_between_points_l151_151926


namespace rectangular_region_area_l151_151860

-- Definitions based on conditions
variable (w : ℝ) -- length of the shorter sides
variable (l : ℝ) -- length of the longer side
variable (total_fence_length : ℝ) -- total length of the fence

-- Given conditions as hypotheses
theorem rectangular_region_area
  (h1 : l = 2 * w) -- The length of the side opposite the wall is twice the length of each of the other two fenced sides
  (h2 : w + w + l = total_fence_length) -- The total length of the fence is 40 feet
  (h3 : total_fence_length = 40) -- total fence length of 40 feet
: (w * l) = 200 := -- The area of the rectangular region is 200 square feet
sorry

end rectangular_region_area_l151_151860


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151385

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151385


namespace sec_150_l151_151485

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151485


namespace triangles_side_relation_l151_151986

variable (a b c a' b' c' : ℝ)
variable (A B C A' B' C' : ℝ)

-- Conditions from the problem
variable (angle_B : B = B')
variable (angle_A_sum : A + A' = 180)

theorem triangles_side_relation
(angle_B : B = B') :
A + A' = 180 → 
a * a' = b * b' + c * c' :=
by
  intro angle_A_sum,
  sorry

end triangles_side_relation_l151_151986


namespace total_rainfall_l151_151890

theorem total_rainfall (rain_monday rain_tuesday rain_wednesday : ℝ) :
    rain_monday = 0.17 ∧ rain_tuesday = 0.42 ∧ rain_wednesday = 0.08 →
    rain_monday + rain_tuesday + rain_wednesday = 0.67 :=
by
  intro h
  cases h with hrain_monday hrest
  cases hrest with hrain_tuesday hrain_wednesday
  rw [hrain_monday, hrain_tuesday, hrain_wednesday]
  norm_num

end total_rainfall_l151_151890


namespace diorama_factor_l151_151330

theorem diorama_factor (P B factor : ℕ) (h1 : P + B = 67) (h2 : B = P * factor - 5) (h3 : B = 49) : factor = 3 :=
by
  sorry

end diorama_factor_l151_151330


namespace nina_saving_ratio_l151_151119

theorem nina_saving_ratio (game_cost : ℝ) (tax_rate : ℝ) (allowance : ℝ) (weeks : ℕ) 
  (h_game_cost : game_cost = 50)
  (h_tax_rate : tax_rate = 0.10)
  (h_allowance : allowance = 10)
  (h_weeks : weeks = 11) :
  let total_cost := game_cost * (1 + tax_rate),
      total_savings := allowance * weeks,
      weekly_savings := allowance
  in weekly_savings / allowance = 1 :=
by sorry

end nina_saving_ratio_l151_151119


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151451

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151451


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151540

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151540


namespace volleyballs_count_l151_151135

theorem volleyballs_count 
  (total_balls soccer_balls : ℕ)
  (basketballs tennis_balls baseballs volleyballs : ℕ) 
  (h_total : total_balls = 145) 
  (h_soccer : soccer_balls = 20) 
  (h_basketballs : basketballs = soccer_balls + 5)
  (h_tennis : tennis_balls = 2 * soccer_balls) 
  (h_baseballs : baseballs = soccer_balls + 10) 
  (h_specific_total : soccer_balls + basketballs + tennis_balls + baseballs = 115): 
  volleyballs = 30 := 
by 
  have h_specific_balls : soccer_balls + basketballs + tennis_balls + baseballs = 115 :=
    h_specific_total
  have total_basketballs : basketballs = 25 :=
    by rw [h_basketballs, h_soccer]; refl
  have total_tennis_balls : tennis_balls = 40 :=
    by rw [h_tennis, h_soccer]; refl
  have total_baseballs : baseballs = 30 :=
    by rw [h_baseballs, h_soccer]; refl
  have total_specific_balls : 20 + 25 + 40 + 30 = 115 :=
    by norm_num
  have volleyballs_to_find : volleyballs = 145 - 115 :=
    by rw [h_total]; exact rfl
  sorry

end volleyballs_count_l151_151135


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151536

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151536


namespace smallest_solution_l151_151608

def equation (x : ℝ) := (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 14

theorem smallest_solution :
  ∀ x : ℝ, equation x → x = (3 - Real.sqrt 333) / 6 :=
sorry

end smallest_solution_l151_151608


namespace set_union_example_l151_151014

theorem set_union_example (x : ℕ) (M N : Set ℕ) (h1 : M = {0, x}) (h2 : N = {1, 2}) (h3 : M ∩ N = {2}) :
  M ∪ N = {0, 1, 2} := by
  sorry

end set_union_example_l151_151014


namespace tan_theta_l151_151652

theorem tan_theta:
  ∀ (θ : ℝ), 0 < θ ∧ θ < π ∧ sin θ + cos θ = 1 / 5 → tan θ = - (4 / 3) :=
by
  intro θ h
  sorry

end tan_theta_l151_151652


namespace greatest_lcm_less_than_120_l151_151248

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b
noncomputable def multiples (x limit : ℕ) : List ℕ := List.range (limit / x) |>.map (λ n => x * (n + 1))

theorem greatest_lcm_less_than_120 :  GCM_of_10_and_15_lt_120 = 90
  where
    GCM_of_10_and_15_lt_120 : ℕ := match (multiples (lcm 10 15) 120) with
                                     | [] => 0
                                     | xs => xs.maximum'.getD 0 :=
  by
  apply sorry

end greatest_lcm_less_than_120_l151_151248


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151545

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151545


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151497

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151497


namespace AB_not_selected_together_l151_151995

open Finset

-- Define the five points
def points : Finset ℕ := {0, 1, 2, 3, 4}

-- Define the specific points A and B
def A := 0
def B := 1

-- Define the event of selecting 3 out of 5 points
def selections := points.powerset.filter (λ s, s.card = 3)

-- Define the event that A and B are selected together
def AB_selected := selections.filter (λ s, A ∈ s ∧ B ∈ s)

-- Total number of selections
def total_selections := selections.card

-- Number of selections where A and B are selected together
def AB_selections := AB_selected.card

-- Probability calculation
def probability_AB_not_selected_together := 1 - (AB_selections / total_selections : ℚ)

-- Theorem statement
theorem AB_not_selected_together : 
  probability_AB_not_selected_together = 7 / 10 := 
sorry

end AB_not_selected_together_l151_151995


namespace min_shirts_to_save_money_l151_151875

theorem min_shirts_to_save_money :
  let acme_cost (x : ℕ) := 75 + 12 * x
  let gamma_cost (x : ℕ) := 18 * x
  ∀ x : ℕ, acme_cost x < gamma_cost x → x ≥ 13 := 
by
  intros
  sorry

end min_shirts_to_save_money_l151_151875


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151419

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151419


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151527

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151527


namespace calculate_area_bounded_figure_l151_151145

noncomputable def area_of_bounded_figure (R : ℝ) : ℝ :=
  (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi)

theorem calculate_area_bounded_figure (R : ℝ) :
  ∀ r, r = (R / 3) → area_of_bounded_figure R = (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi) :=
by
  intros r hr
  subst hr
  exact rfl

end calculate_area_bounded_figure_l151_151145


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151560

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151560


namespace ray_initial_cents_l151_151134

theorem ray_initial_cents :
  ∀ (initial_cents : ℕ), 
    (∃ (peter_cents : ℕ), 
      peter_cents = 30 ∧
      ∃ (randi_cents : ℕ),
        randi_cents = 2 * peter_cents ∧
        randi_cents = peter_cents + 60 ∧
        peter_cents + randi_cents = initial_cents
    ) →
    initial_cents = 90 := 
by
    intros initial_cents h
    obtain ⟨peter_cents, hp, ⟨randi_cents, hr1, hr2, hr3⟩⟩ := h
    sorry

end ray_initial_cents_l151_151134


namespace roots_of_polynomial_l151_151103

noncomputable def omega : ℂ := complex.exp (complex.I * π * 2 / 5)

theorem roots_of_polynomial :
  ∃ (P : polynomial ℂ), (P = polynomial.X^4 - polynomial.X^3 + polynomial.X^2 - polynomial.X + 1) ∧ 
  (P.eval (omega) = 0) ∧ 
  (P.eval (omega^3) = 0) ∧ 
  (P.eval (omega^7) = 0) ∧ 
  (P.eval (omega^9) = 0) :=
by
  -- Define ω as a complex number satisfying ω^5 = -1
  have hω : omega^5 = -1 := by
    -- Sorry placeholder
    sorry

  -- Define the polynomial P(x) = x^4 - x^3 + x^2 - x + 1
  let P : polynomial ℂ := polynomial.X^4 - polynomial.X^3 + polynomial.X^2 - polynomial.X + 1

  -- Prove that P(ω) = 0
  have hroot1 : P.eval omega = 0 := by
    -- Sorry placeholder
    sorry

  -- Prove that P(ω^3) = 0
  have hroot2 : P.eval (omega^3) = 0 := by
    -- Sorry placeholder
    sorry

  -- Prove that P(ω^7) = 0
  have hroot3 : P.eval (omega^7) = 0 := by
    -- Sorry placeholder
    sorry

  -- Prove that P(ω^9) = 0
  have hroot4 : P.eval (omega^9) = 0 := by
    -- Sorry placeholder
    sorry

  -- Existential quantification of the polynomial P with the given roots
  exact ⟨P, rfl, hroot1, hroot2, hroot3, hroot4⟩

end roots_of_polynomial_l151_151103


namespace bus_arrives_on_time_at_least_4_times_l151_151316

theorem bus_arrives_on_time_at_least_4_times (p : ℝ) (n : ℕ) (k : ℕ) (P : ℝ) :
  p = 0.8 → n = 5 → k = 4 → P = 0.74 →
  (∑ i in finset.range (k + 1), (nat.choose n i) * (p ^ i) * ((1 - p) ^ (n - i))) = P :=
by
  intros
  sorry

end bus_arrives_on_time_at_least_4_times_l151_151316


namespace distinct_terms_count_l151_151350

theorem distinct_terms_count (a b : ℕ) : 
    let x := (a + 5 * b) in 
    let y := (a - 5 * b) in 
    let expression := (x * y)^3 in 
    let z := a^2 - 25 * b^2 in
    let expansion := (a^2 - 25 * b^2)^6 in 
  true := 
by sorry

end distinct_terms_count_l151_151350


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151519

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151519


namespace smallest_integer_is_nine_l151_151809

theorem smallest_integer_is_nine 
  (a b c : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : a + b + c = 90) 
  (h3 : (a:ℝ)/b = 2/3) 
  (h4 : (b:ℝ)/c = 3/5) : 
  a = 9 :=
by 
  sorry

end smallest_integer_is_nine_l151_151809


namespace eval_expression_l151_151891

theorem eval_expression : (-1)^45 + 2^(3^2 + 5^2 - 4^2) = 262143 := by
  sorry

end eval_expression_l151_151891


namespace find_p_l151_151269

theorem find_p (m n p : ℝ) :
  m = (n / 7) - (2 / 5) →
  m + p = ((n + 21) / 7) - (2 / 5) →
  p = 3 := by
  sorry

end find_p_l151_151269


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151372

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151372


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151373

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151373


namespace sec_150_l151_151490

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151490


namespace opposite_of_neg_two_l151_151181

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l151_151181


namespace complement_intersection_l151_151015

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  ((U \ A) ∩ B) = {0} :=
by
  sorry

end complement_intersection_l151_151015


namespace triangle_right_circle_l151_151104

theorem triangle_right_circle (A B C D : Point) (hB_right : ∠ABC = 90°) 
(h_circle_diameter : Circle (Segment BC) meets AC at D) 
(hAD : AD = 3) (hBD : BD = 6) : CD = 12 :=
by 
  sorry

end triangle_right_circle_l151_151104


namespace sqrt_subtraction_l151_151340

theorem sqrt_subtraction : sqrt 8 - sqrt 2 = sqrt 2 :=
  sorry

end sqrt_subtraction_l151_151340


namespace inequality_am_gm_l151_151113

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (3 * x * y + y * z) / (x^2 + y^2 + z^2) ≤ sqrt 10 / 2 :=
sorry

end inequality_am_gm_l151_151113


namespace find_slope_of_parallel_line_l151_151168

-- Define the condition that line1 is parallel to line2.
def lines_parallel (k : ℝ) : Prop :=
  k = -3

-- The theorem that proves the condition given.
theorem find_slope_of_parallel_line (k : ℝ) (h : lines_parallel k) : k = -3 :=
by
  exact h

end find_slope_of_parallel_line_l151_151168


namespace locus_of_M_is_ellipse_l151_151954

theorem locus_of_M_is_ellipse :
  ∀ (a b : ℝ) (M : ℝ × ℝ),
  a > b → b > 0 → (∃ x y : ℝ, 
  (M = (x, y)) ∧ 
  ∃ (P : ℝ × ℝ),
  (∃ x0 y0 : ℝ, P = (x0, y0) ∧ (x0^2 / a^2 + y0^2 / b^2 = 1)) ∧ 
  P ≠ (a, 0) ∧ P ≠ (-a, 0) ∧
  (∃ t : ℝ, t = (x^2 + y^2 - a^2) / (2 * y)) ∧ 
  (∃ x0 y0 : ℝ, 
    x0 = -x ∧ 
    y0 = 2 * t - y ∧
    x0^2 / a^2 + y0^2 / b^2 = 1)) →
  ∃ (x y : ℝ),
  M = (x, y) ∧ 
  (x^2 / a^2 + y^2 / (a^4 / b^2) = 1) := 
sorry

end locus_of_M_is_ellipse_l151_151954


namespace triangle_area_l151_151036

theorem triangle_area (b c : ℝ) (angleC : ℝ) (S : ℝ) 
  (hb : b = 1) (hc : c = √3) (hC : angleC = (2/3) * Real.pi) :
  S = (√3) / 4 := 
sorry

end triangle_area_l151_151036


namespace area_of_sector_l151_151893

-- Define the radius and central angle
def R : ℝ := 1
def α : ℝ := (2 * Real.pi) / 3

-- Formula for the area of the sector
def S (R α : ℝ) : ℝ := (1 / 2) * α * R^2

-- Statement that the computed area is equal to the expected area
theorem area_of_sector (S (R α) = (Real.pi) / 3) : 
  S R α = Real.pi / 3 := sorry

end area_of_sector_l151_151893


namespace math_problem_l151_151110

noncomputable def proof_statement : Prop :=
  ∃ (a b m : ℝ),
    0 < a ∧ 0 < b ∧ 0 < m ∧
    (5 = m^2 * ((a^2 / b^2) + (b^2 / a^2)) + m * (a/b + b/a)) ∧
    m = (-1 + Real.sqrt 21) / 2

theorem math_problem : proof_statement :=
  sorry

end math_problem_l151_151110


namespace equidistant_point_is_circumcenter_l151_151191

noncomputable def equidistant_point (Δ : Triangle) :=
  ∃ (P : Point), is_equidistant_to_vertices P Δ

theorem equidistant_point_is_circumcenter (Δ : Triangle) :
  ∃ (P : Point), is_circumcenter P Δ :=
by
  -- Proof goes here (not required)
  sorry

end equidistant_point_is_circumcenter_l151_151191


namespace beta_sum_l151_151348

noncomputable def Q (x : ℂ) : ℂ := (1 + x + x^2 + ⋯ + x^20)^2 - x^19

def zero_form (z : ℂ) (r : ℝ) (β : ℝ) : Prop :=
  z = r * (Complex.cos (2 * Real.pi * β) + Complex.sin (2 * Real.pi * β) * Complex.I)

def β_values (βs : List ℝ) : Prop :=
  βs = [1/21, 1/20, 2/21, 1/10, 3/21]

theorem beta_sum : ∀ (βs : List ℝ), 
  (∀ k ∈ [1,2,3,...,42], ∃ r k, zero_form (Q k) r β k ∧ 0 < β k ∧ β k < 1) 
  ∧ β_values βs → βs.sum = 71 / 210 := 
  by 
  intros βs h1 h2 
  sorry

end beta_sum_l151_151348


namespace triangle_area_inscribed_circle_l151_151871

noncomputable def arc_length1 : ℝ := 6
noncomputable def arc_length2 : ℝ := 8
noncomputable def arc_length3 : ℝ := 10

noncomputable def circumference : ℝ := arc_length1 + arc_length2 + arc_length3
noncomputable def radius : ℝ := circumference / (2 * Real.pi)
noncomputable def angle_sum : ℝ := 360
noncomputable def angle1 : ℝ := 90 * Real.pi / 180
noncomputable def angle2 : ℝ := 120 * Real.pi / 180
noncomputable def angle3 : ℝ := 150 * Real.pi / 180

theorem triangle_area_inscribed_circle :
  let r := radius in
  let sin_a1 := Real.sin angle1 in
  let sin_a2 := Real.sin angle2 in
  let sin_a3 := Real.sin angle3 in
  (1 / 2) * r^2 * (sin_a1 + sin_a2 + sin_a3) = (72 * (1 + Real.sqrt 3)) / (Real.pi^2) := by
  sorry

end triangle_area_inscribed_circle_l151_151871


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151570

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151570


namespace length_CD_l151_151787

-- Given conditions as per the problem
def radius : ℝ := 4
def volume_region : ℝ := 384 * Real.pi

-- Definition of the volume formula for the region around the line segment CD.
def volume_hemispheres (r : ℝ) : ℝ := 2 * (2 / 3) * Real.pi * r^3
def volume_cylinder (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h
def total_volume (r : ℝ) (h : ℝ) : ℝ := volume_hemispheres r + volume_cylinder r h

-- The proof statement
theorem length_CD (h : ℝ) : h = 19 :=
by
  have radius_def : radius = 4 := rfl,
  have vol_def : volume_region = 384 * Real.pi := rfl,
  have vol_hemispheres : volume_hemispheres radius = 2 * (2 / 3) * Real.pi * radius^3 :=
    by rw [radius_def]; simp [volume_hemispheres, radius],
  have vol_cylinder : volume_cylinder radius h = volume_region - volume_hemispheres radius :=
    by rw [vol_def, vol_hemispheres]; simp [volume_cylinder, total_volume, volume_region, radius],
  sorry -- Proof placeholder

end length_CD_l151_151787


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151386

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151386


namespace largest_solution_log_eq_l151_151346

theorem largest_solution_log_eq (x : ℝ) (h : log (5 : ℝ) / log (5 * x^2) + log (5 : ℝ) / log (25 * x^3) = -1) : 
  1 / x ^ 10 = 0.00001 :=
by
  sorry

end largest_solution_log_eq_l151_151346


namespace mileage_per_gallon_l151_151745

-- Define the conditions
def miles_driven : ℝ := 100
def gallons_used : ℝ := 5

-- Define the question as a theorem to be proven
theorem mileage_per_gallon : (miles_driven / gallons_used) = 20 := by
  sorry

end mileage_per_gallon_l151_151745


namespace average_of_other_25_results_l151_151156

theorem average_of_other_25_results
  (avg_first_45 : ℝ)
  (avg_all_70 : ℝ)
  (num_first_results : ℕ)
  (num_other_results : ℕ)
  (total_results : ℕ)
  (sum_first_45 : ℝ)
  (sum_all_70 : ℝ) :
  avg_first_45 = 25 → 
  avg_all_70 = 32.142857142857146 → 
  num_first_results = 45 → 
  num_other_results = 25 → 
  total_results = 70 → 
  sum_first_45 = num_first_results * avg_first_45 → 
  sum_all_70 = total_results * avg_all_70 →
  (25 * 45 + 25 * avg_first_45 = sum_all_70) →
  45 :=
by
  sorry

end average_of_other_25_results_l151_151156


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151548

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151548


namespace min_expression_min_expression_achieve_l151_151602

theorem min_expression (x : ℝ) (hx : 0 < x) : 
  (x^2 + 8 * x + 64 / x^3) ≥ 28 :=
sorry

theorem min_expression_achieve (x : ℝ) (hx : x = 2): 
  (x^2 + 8 * x + 64 / x^3) = 28 :=
sorry

end min_expression_min_expression_achieve_l151_151602


namespace smaller_number_is_four_l151_151797

theorem smaller_number_is_four (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 10) : y = 4 :=
by
  sorry

end smaller_number_is_four_l151_151797


namespace number_of_common_terms_l151_151674

theorem number_of_common_terms :
  (finset.range 36).filter (λ n, ∃ m, m < 21 ∧ 4 * n - 1 = 7 * m - 5).card = 5 :=
sorry

end number_of_common_terms_l151_151674


namespace alex_height_l151_151850

theorem alex_height
  (tree_height: ℚ) (tree_shadow: ℚ) (alex_shadow_in_inches: ℚ)
  (h_tree: tree_height = 50)
  (h_shadow_tree: tree_shadow = 25)
  (h_shadow_alex: alex_shadow_in_inches = 20) :
  ∃ alex_height_in_feet: ℚ, alex_height_in_feet = 10 / 3 :=
by
  sorry

end alex_height_l151_151850


namespace magnitude_of_2a_minus_b_eq_sqrt_21_l151_151659

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Definitions and conditions
def angle_between_a_b : Real.Angle := Real.Angle.pi / 3 -- 60 degrees in radians
def norm_a : ℝ := 2
def norm_b : ℝ := 5

-- Vector norms and properties
def norm_a_def : ∥a∥ = norm_a := by sorry
def norm_b_def : ∥b∥ = norm_b := by sorry
def angle_between_a_b_def : Real.Angle.cos angle_between_a_b = 1/2 := by sorry

-- Propose
theorem magnitude_of_2a_minus_b_eq_sqrt_21 :
  |2 • a - b| = Real.sqrt 21 :=
by
  sorry

end magnitude_of_2a_minus_b_eq_sqrt_21_l151_151659


namespace even_n_equals_identical_numbers_l151_151121

theorem even_n_equals_identical_numbers (n : ℕ) (h1 : n ≥ 2) : 
  (∃ f : ℕ → ℕ, (∀ a b, f a = f b + f b) ∧ n % 2 = 0) :=
sorry


end even_n_equals_identical_numbers_l151_151121


namespace composite_square_perimeter_l151_151085

theorem composite_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let s1 := p1 / 4
  let s2 := p2 / 4
  (p1 + p2 - 2 * s1) = 120 := 
by
  -- proof goes here
  sorry

end composite_square_perimeter_l151_151085


namespace y_work_days_eq_10_l151_151271

-- Definitions based on the given conditions
variables (W : ℝ)
def x_rate : ℝ := W / 24
def y_rate : ℝ := W / 16
variables (d : ℝ)
def y_work := d * y_rate
def x_remaining_work := 9 * x_rate

-- The main problem
theorem y_work_days_eq_10 (W : ℝ) (d : ℝ) :
  d * (W / 16) + 9 * (W / 24) = W → d = 10 := by
  intro h
  sorry

end y_work_days_eq_10_l151_151271


namespace coefficient_x4_in_expansion_l151_151240

theorem coefficient_x4_in_expansion : 
  (∑ k in Finset.range (9), (Nat.choose 8 k) * (3 : ℤ)^k * (2 : ℤ)^(8-k) * (X : ℤ[X])^k).coeff 4 = 90720 :=
by
  sorry

end coefficient_x4_in_expansion_l151_151240


namespace sum_le_S10_implies_S19_nonnegative_l151_151643

noncomputable def arithmetic_sequence (a n d: ℤ) : ℕ → ℤ 
| 0     := a
| (n+1) := arithmetic_sequence (a + d) n 

noncomputable def sum_of_first_n_terms (a d: ℤ) (n: ℕ) : ℤ :=
(n * (2 * a + (n - 1) * d)) / 2

theorem sum_le_S10_implies_S19_nonnegative {a d : ℤ} (h_d : d ≠ 0):
(∀ n : ℕ, sum_of_first_n_terms a d n ≤ sum_of_first_n_terms a d 10) → 
sum_of_first_n_terms a d 19 ≥ 0 := sorry

end sum_le_S10_implies_S19_nonnegative_l151_151643


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151426

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151426


namespace dealership_sales_l151_151888

theorem dealership_sales (sports_cars : ℕ) (sedans : ℕ) (trucks : ℕ) 
  (h1 : sports_cars = 36)
  (h2 : (3 : ℤ) * sedans = 5 * sports_cars)
  (h3 : (3 : ℤ) * trucks = 4 * sports_cars) :
  sedans = 60 ∧ trucks = 48 := 
sorry

end dealership_sales_l151_151888


namespace find_a99_l151_151979

def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n ≥ 2, a n - a (n-1) = n + 1

theorem find_a99 (a : ℕ → ℕ) (h : seq a) : a 99 = 5049 :=
by
  have : seq a := h
  sorry

end find_a99_l151_151979


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151583

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151583


namespace area_of_region_l151_151821

theorem area_of_region : ∃ A, (∀ x y : ℝ, x^2 + y^2 + 6*x - 4*y = 12 → A = 25 * Real.pi) :=
by
  -- Completing the square and identifying the circle
  -- We verify that the given equation represents a circle
  existsi (25 * Real.pi)
  intros x y h
  sorry

end area_of_region_l151_151821


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151408

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151408


namespace median_room_number_of_remaining_28_participants_l151_151334

open List

def remaining_rooms : List ℕ := [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30]

theorem median_room_number_of_remaining_28_participants : 
  List.median remaining_rooms = 15 :=
by
  sorry

end median_room_number_of_remaining_28_participants_l151_151334


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151411

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151411


namespace min_tangent_length_is_sqrt_5_l151_151723

-- Define the circle equation center and radius
def circle_center : ℝ × ℝ := (-3, 2)
def circle_radius : ℝ := 2

-- Define the line y = -1
def line_y_eq_negative_one (x : ℝ) : ℝ := -1

-- Define the distance function between a point and the line y = -1
def distance_to_line (p : ℝ × ℝ) : ℝ := |p.snd + 1|

-- Define the point lying on the line y = -1
def point_on_line (P : ℝ × ℝ) := P.snd = -1

-- Define the minimum tangent length function
def tangent_length (d r : ℝ) : ℝ := Real.sqrt (d ^ 2 - r ^ 2)

-- Define the minimum tangent length from a point on the line y = -1 to the circle
noncomputable def min_tangent_length_to_circle : ℝ :=
  tangent_length (distance_to_line circle_center) circle_radius

-- The proof statement 
theorem min_tangent_length_is_sqrt_5 : 
  min_tangent_length_to_circle = Real.sqrt 5 := 
by 
  sorry

end min_tangent_length_is_sqrt_5_l151_151723


namespace opposite_of_neg2_l151_151172

theorem opposite_of_neg2 : ∃ y : ℤ, -2 + y = 0 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end opposite_of_neg2_l151_151172


namespace roots_of_quadratic_eqn_l151_151114

theorem roots_of_quadratic_eqn (a b : ℚ) (m r : ℚ)
  (hyp1 : a * b = 6)
  (hyp2 : ∃ p, (a + 1/b) * (b + 1/a) = r) :
  r = 49 / 6 :=
by 
  -- Solution proof goes here
  sorry

end roots_of_quadratic_eqn_l151_151114


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151552

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151552


namespace find_circle_eq_l151_151646

noncomputable def circle_eq (x : ℝ) (y : ℝ) (r : ℝ) : Prop :=
  (x - 1) ^ 2 + y ^ 2 = r

noncomputable def is_tangent_to (line_eq : ℝ → ℝ → Prop) (circle_center : ℝ × ℝ) (circle_radius : ℝ) :=
  ∀ (x y : ℝ), line_eq x y → (circle_center.fst - x) ^ 2 + (circle_center.snd - y) ^ 2 = circle_radius ^ 2

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd - p2.snd)^2)

theorem find_circle_eq (x y r : ℝ) :
  circle_eq x y 1 / 2 →
  is_tangent_to (λ x y, y = x) (x, y) r →
  is_tangent_to (λ x y, y = -x) (x, y) r →
  distance (1, 0) (x, y) = 3 →
  (circle_eq x y 8 ∨
   circle_eq (x + 2) y 2 ∨
   circle_eq x (y - 2 * real.sqrt 2) 4 ∨
   circle_eq x (y + 2 * real.sqrt 2) 4) :=
sorry

end find_circle_eq_l151_151646


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151494

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151494


namespace finite_operations_to_single_subset_l151_151864

theorem finite_operations_to_single_subset {α : Type*} {n : ℕ} (s : Finset α) (h_card : s.card = 2^n)
  (partitions : Finset (Finset α)) (h_disjoint : ∀ ⦃a b⦄, a ∈ partitions → b ∈ partitions → a ≠ b → a ∩ b = ∅)
  (h_union : partitions.sup id = s) :
  ∃ (ops : ℕ), ∃ (result : Finset α), result = s := 
begin
  -- This statement asserts the existence of a finite number of operations
  -- that result in one subset equivalent to the entire original set s.
  sorry
end

end finite_operations_to_single_subset_l151_151864


namespace find_n_l151_151685

theorem find_n (n : ℕ) (h : 7^(2*n) = (1/7)^(n-12)) : n = 4 :=
sorry

end find_n_l151_151685


namespace quadratic_equation_statements_l151_151935

theorem quadratic_equation_statements 
    (a b c : ℝ) (h₀ : a ≠ 0) :
    (a + b + c = 0 → IsRoot (fun x => a*x^2 + b*x + c) 1) ∧
    ((∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a*x1^2 + c = 0 ∧ a*x2^2 + c = 0) → (¬ ∃ x : ℝ, a*x^2 + b*x + c = 0)) ∧
    (∀ x1 x2 : ℝ, x1 ≠ x2 ∧ a*x1^2 + b*x1 + c = 0 ∧ a*x2^2 + b*x2 + c = 0 → 
        ∃ y1 y2 : ℝ, y1 = 1 / x1 ∧ y2 = 1 / x2 ∧ c*y1^2 + b*y1 + a = 0 ∧ c*y2^2 + b*y2 + a = 0) ∧
    (∀ x0 : ℝ, a*x0^2 + b*x0 + c = 0 → b^2 - 4*a*c = (2*a*x0 + b)^2) :=
by 
  sorry

end quadratic_equation_statements_l151_151935


namespace center_on_PQ_l151_151190

open EuclideanGeometry

noncomputable def acuteTriangle (A B C : Point) : Prop := ∠ B A C < π / 2 ∧ ∠ A B C < π / 2 ∧ ∠ B C A < π / 2

noncomputable def perpendicularBisector (p1 p2 : Point) : Line := sorry -- Define this properly

noncomputable def circumcenter (A B C : Point) : Point := sorry -- Define this properly

noncomputable def circumcircle (A B C : Point) : Circle := sorry -- Define this properly

noncomputable def intersect (c1 c2 : Circle) : set Point := sorry -- Define this properly

theorem center_on_PQ
  (A B C B1 B2 C1 C2 P Q : Point)
  (hAcute: acuteTriangle A B C)
  (hPbAC: perpendicularBisector A C = LineThrough B1 B2)
  (hPbAB: perpendicularBisector A B = LineThrough C1 C2)
  (hPB1B2: intersection (circumcircle B B1 B2) (circumcircle C C1 C2) = {P, Q})
  : let O := circumcenter A B C in Collinear O P Q :=
sorry

end center_on_PQ_l151_151190


namespace expected_profit_l151_151804

namespace DailyLottery

/-- Definitions for the problem -/

def ticket_cost : ℝ := 2
def first_prize : ℝ := 100
def second_prize : ℝ := 10
def prob_first_prize : ℝ := 0.001
def prob_second_prize : ℝ := 0.1
def prob_no_prize : ℝ := 1 - prob_first_prize - prob_second_prize

/-- Expected profit calculation as a theorem -/

theorem expected_profit :
  (first_prize * prob_first_prize + second_prize * prob_second_prize + 0 * prob_no_prize) - ticket_cost = -0.9 :=
by
  sorry

end DailyLottery

end expected_profit_l151_151804


namespace sphere_volume_l151_151201

theorem sphere_volume (h : 4 * π * r^2 = 256 * π) : (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end sphere_volume_l151_151201


namespace portions_left_to_complete_l151_151054

-- Defining the total number of each type of media in the collection
def total_books := 22
def total_movies := 10
def total_video_games := 8
def total_audiobooks := 15

-- Defining the number of each type of media that has been completed
def books_read := 12
def half_books_read := 2 / 2
def movies_watched := 6
def half_movies_watched := 1 / 2
def video_games_played := 3
def audiobooks_listened := 7

-- Total number of portions in the collection
def total_portions : ℕ := total_books + total_movies + total_video_games + total_audiobooks

-- Number of portions completed
def completed_books := books_read - half_books_read
def completed_movies := movies_watched - half_movies_watched
def completed_video_games := video_games_played
def completed_audiobooks := audiobooks_listened

def total_completed_portions : ℕ := completed_books + completed_movies + completed_video_games + completed_audiobooks

-- Defining the theorem to be proven
theorem portions_left_to_complete :
  total_portions - total_completed_portions = 28.5 :=
by
  sorry

end portions_left_to_complete_l151_151054


namespace correct_division_result_l151_151089

theorem correct_division_result (M : ℕ) (h1 : M / 37 = 9.684469) : 
  M / 37 = 9 + 648649 / 1000000 :=
sorry

end correct_division_result_l151_151089


namespace pablo_popsicles_max_l151_151124

def greatest_popsicles (single_popsicle_cost three_popsicle_box_cost five_popsicle_box_cost : ℕ) (budget : ℕ) : ℕ :=
  if budget < five_popsicle_box_cost then
    budget // single_popsicle_cost -- buy single popsicles only
  else if budget < five_popsicle_box_cost + three_popsicle_box_cost then
    (budget // five_popsicle_box_cost) * 5 + (budget % five_popsicle_box_cost) // three_popsicle_box_cost * 3
  else
    (budget // five_popsicle_box_cost) * 5 + (budget % five_popsicle_box_cost) // three_popsicle_box_cost * 3 + (budget % five_popsicle_box_cost % three_popsicle_box_cost) // single_popsicle_cost

theorem pablo_popsicles_max : greatest_popsicles 1 2 3 8 = 13 :=
by
  -- Define the costs for single popsicle, 3-popsicle box, 5-popsicle box
  let single_popsicle_cost := 1
  let three_popsicle_box_cost := 2
  let five_popsicle_box_cost := 3
  -- Define budget
  let budget := 8
  -- Greatest number of popsicles Pablo can buy
  have h : greatest_popsicles single_popsicle_cost three_popsicle_box_cost five_popsicle_box_cost budget = 13 := sorry
  exact h

end pablo_popsicles_max_l151_151124


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151452

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151452


namespace derivative_of_y_l151_151845

variable (x : ℝ)

def y : ℝ :=
  8 * Real.sin (Real.cot 3) + (1 / 5) * (Real.sin (5 * x))^2 / Real.cos (10 * x)

theorem derivative_of_y :
  deriv y x = Real.tan (10 * x) / (5 * Real.cos (10 * x)) :=
by
  sorry

end derivative_of_y_l151_151845


namespace hyperbola_s_squared_zero_l151_151300

open Real

theorem hyperbola_s_squared_zero :
  ∃ s : ℝ, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ (x y : ℝ), 
  ((x, y) = (-2, 3) ∨ (x, y) = (0, -1) ∨ (x, y) = (s, 1)) → (y^2 / a^2 - x^2 / b^2 = 1))
  ) → s ^ 2 = 0 :=
by
  sorry

end hyperbola_s_squared_zero_l151_151300


namespace min_value_of_a_b_c_l151_151655

variable (a b c : ℕ)
variable (x1 x2 : ℝ)

axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a * x1^2 + b * x1 + c = 0
axiom h5 : a * x2^2 + b * x2 + c = 0
axiom h6 : |x1| < 1/3
axiom h7 : |x2| < 1/3

theorem min_value_of_a_b_c : a + b + c = 25 :=
by
  sorry

end min_value_of_a_b_c_l151_151655


namespace correct_sqrt_operation_l151_151836

theorem correct_sqrt_operation :
  ¬ (sqrt 3 + sqrt 3 = 3) ∧
  ¬ (4 * sqrt 5 - sqrt 5 = 4) ∧
  ¬ (sqrt 32 / sqrt 8 = 4) ∧
  (sqrt 3 * sqrt 2 = sqrt 6) :=
by
  sorry

end correct_sqrt_operation_l151_151836


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151412

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151412


namespace number_of_nonnegative_real_x_sqrt_integer_l151_151618

-- Define the conditions
def is_integer (n : ℝ) : Prop := ∃ k : ℤ, n = k

-- Problem statement
theorem number_of_nonnegative_real_x_sqrt_integer (x : ℝ) (hx : x ≥ 0) :
  (is_integer (sqrt (225 - (x ^ (1/3))))) → (∃ n : ℤ, 0 ≤ n ∧ n ≤ 15) :=
sorry

end number_of_nonnegative_real_x_sqrt_integer_l151_151618


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151530

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151530


namespace sample_size_is_150_l151_151863

theorem sample_size_is_150 
  (classes : ℕ) (students_per_class : ℕ) (selected_students : ℕ)
  (h1 : classes = 40) (h2 : students_per_class = 50) (h3 : selected_students = 150)
  : selected_students = 150 :=
sorry

end sample_size_is_150_l151_151863


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151454

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151454


namespace directrix_of_parabola_l151_151601

theorem directrix_of_parabola (x y : ℝ) :
  (∀ x y : ℝ, y = -x^2 -> y = -a ∧ 4a = -1 -> y = (1 / 4)) :=
sorry

end directrix_of_parabola_l151_151601


namespace gcd_multiple_less_than_120_l151_151251

theorem gcd_multiple_less_than_120 (n : ℕ) (h1 : n < 120) (h2 : n % 10 = 0) (h3 : n % 15 = 0) : n ≤ 90 :=
by {
  sorry
}

end gcd_multiple_less_than_120_l151_151251


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151375

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151375


namespace lindy_distance_l151_151072

theorem lindy_distance
  (d : ℝ) (v_j : ℝ) (v_c : ℝ) (v_l : ℝ) (t : ℝ)
  (h1 : d = 270)
  (h2 : v_j = 4)
  (h3 : v_c = 5)
  (h4 : v_l = 8)
  (h_time : t = d / (v_j + v_c)) :
  v_l * t = 240 := by
  sorry

end lindy_distance_l151_151072


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151442

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151442


namespace apple_tunnel_cut_l151_151325

noncomputable section

open Classical

/-- Given an apple in the shape of a ball with radius 31 mm and a tunnel dug by a worm 
with total length 61 mm, prove that it is possible to cut the apple with a straight 
slice through the center so that one of the two halves does not contain any part of the tunnel. -/
theorem apple_tunnel_cut :
  ∀ (apple : Ball ℝ (31 : ℝ)), ∀ (tunnel : ℝ → Ball ℝ (31 : ℝ)),
  ( ∀ s, (0 ≤ s ∧ s ≤ 61 → tunnel s ∈ apple)) → 
  ∃ (π : AffineSubspace ℝ (EuclideanSpace ℝ (fin 3))),
  (∀ (p : EuclideanSpace ℝ (fin 3)), p ∈ tunnel '' Set.Icc 0 61 → (p ∉ π)) :=
by 
  sorry

end apple_tunnel_cut_l151_151325


namespace paintable_integer_sum_l151_151915

theorem paintable_integer_sum (e f g : ℕ) (h_pos : e > 0 ∧ f > 0 ∧ g > 0) 
                             (h_e : e = 3) (h_f : f = 3) (h_g : g = 3) 
                             (h_unique : ∀ n, (n % e = 0 ↔ n % f ≠ 0 ∧ n % g ≠ 0) ∨
                                              (n % f = 0 ↔ n % e ≠ 0 ∧ n % g ≠ 0) ∨
                                              (n % g = 0 ↔ n % e ≠ 0 ∧ n % f ≠ 0)) : 
                             ∑ e = 3 → ∑ f = 3 → ∑ g = 3 → 100 * e + 10 * f + g = 333 :=
by 
  sorry

end paintable_integer_sum_l151_151915


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151420

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151420


namespace triangle_area_l151_151038

theorem triangle_area (B : Real) (AB AC : Real) 
  (hB : B = Real.pi / 6) 
  (hAB : AB = 2 * Real.sqrt 3)
  (hAC : AC = 2) : 
  let area := 1 / 2 * AB * AC * Real.sin B
  area = 2 * Real.sqrt 3 := by
  sorry

end triangle_area_l151_151038


namespace bernstein_criterion_l151_151855

theorem bernstein_criterion (φ : ℝ → ℝ) (μ : MeasureTheory.Measure ℝ) :
  (∀ n ≥ 0, ∀ λ > 0, (-1)^n * (deriv^[n] φ) λ ≥ 0)
  ∧ (∃ lim_φ0 : ℝ, Filter.Tendsto φ (Filter.NhdsWithin 0 (Set.Ioi 0)) (𝓝 lim_φ0) ∧ lim_φ0 = 1)
  ↔ (∃ μ : MeasureTheory.Measure ℝ,
      ∀ λ > 0, φ λ = ∫ x in Set.Ici 0, exp (-λ * x) ∂μ) :=
sorry

end bernstein_criterion_l151_151855


namespace perimeter_of_given_rectangle_area_of_given_rectangle_approx_l151_151209

noncomputable def perimeter_of_rectangle (length width : ℝ) : ℝ :=
2 * length + 2 * width

noncomputable def area_of_rectangle (length width : ℝ) : ℝ :=
length * width

theorem perimeter_of_given_rectangle :
  perimeter_of_rectangle (Real.sqrt 128) (Real.sqrt 75) = 16 * Real.sqrt 2 + 10 * Real.sqrt 3 := by
    sorry

theorem area_of_given_rectangle_approx :
  area_of_rectangle (Real.sqrt 128) (Real.sqrt 75) ≈ 96 := by
    have approx_sqrt6 : Real.sqrt 6 ≈ 2.4 := by
    sorry

end perimeter_of_given_rectangle_area_of_given_rectangle_approx_l151_151209


namespace min_safe_squares_is_1_l151_151749

-- Define the size of the chessboard
def chessboard_width : ℕ := 6
def chessboard_height : ℕ := 6

-- Define the number of rooks
def num_rooks : ℕ := 9

-- Define the condition that a rook can attack its entire row and column
def rook_attacks (r : ℕ) (c : ℕ) : set (ℕ × ℕ) := 
  { (r', c') | r = r' ∨ c = c' }

-- The set of all possible positions on the chessboard
def board_positions : set (ℕ × ℕ) := 
  { (r, c) | r < chessboard_height ∧ c < chessboard_width }

-- Define the remaining "safe" squares after placing 9 rooks
noncomputable def remaining_safe_squares (positions : set (ℕ × ℕ)) : ℕ := 
  (board_positions.diff (positions.flat_map (λ pos, rook_attacks pos.1 pos.2))).card

-- Define the proof statement we want to prove
theorem min_safe_squares_is_1 (rook_positions : set (ℕ × ℕ)) 
  (h_rooks : rook_positions.card = num_rooks) : 
  remaining_safe_squares rook_positions = 1 :=
sorry

end min_safe_squares_is_1_l151_151749


namespace mutually_exclusive_not_exhaustive_l151_151944

-- Assume bag contains 2 black balls (B) and 2 white balls (W)
def bag : List String := ["B", "B", "W", "W"]

-- Draw 2 balls from the bag
def draw (s : List String) : List (List String) :=
  s.combinations 2

-- Event: exactly one black ball
def exactly_one_black (drawn : List String) : Prop :=
  drawn.count "B" = 1

-- Event: exactly two white balls
def exactly_two_white (drawn : List String) : Prop :=
  drawn.count "W" = 2

-- The proof statement
theorem mutually_exclusive_not_exhaustive :
  ∃ (events : List (List String)), 
    (∀ d ∈ events, (exactly_one_black d) ∨ (exactly_two_white d)) ∧
    (∀ d1 d2 ∈ events, d1 ≠ d2 → ¬(exactly_one_black d1 ∧ exactly_two_white d2)) ∧
    ¬((∀ d ∈ events, (exactly_one_black d) ∨ (exactly_two_white d)) ∧
      (∀ d ∈ draw bag, (exactly_one_black d) ∨ (exactly_two_white d))) :=
sorry

end mutually_exclusive_not_exhaustive_l151_151944


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151496

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151496


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151573

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151573


namespace train_speed_km_hr_l151_151317

def train_length : ℝ := 500 -- length of the train in meters
def bridge_length : ℝ := 200 -- length of the bridge in meters
def time : ℝ := 60 -- time to cross the bridge in seconds
def total_distance : ℝ := train_length + bridge_length -- total distance traveled
def speed_m_s : ℝ := total_distance / time -- speed in meters per second
def conversion_factor : ℝ := 3.6 -- conversion factor from m/s to km/hr
def speed_km_hr : ℝ := speed_m_s * conversion_factor -- speed in km/hr

theorem train_speed_km_hr :
  speed_km_hr = 42.012 :=
  sorry

end train_speed_km_hr_l151_151317


namespace workout_days_l151_151912

theorem workout_days (n : ℕ) (squats : ℕ → ℕ) 
  (h1 : squats 1 = 30)
  (h2 : ∀ k, squats (k + 1) = squats k + 5)
  (h3 : squats 4 = 45) :
  n = 4 :=
sorry

end workout_days_l151_151912


namespace a_n_less_than_3_l151_151638

-- Define the function f
def f (x : ℝ) : ℝ := -(1 / 3) * x ^ 2 + 2 * x

-- Define the sequence a_n
def a : ℕ → ℝ
| 0       := 1
| (n + 1) := f (a n)

-- Prove the conjecture using mathematical induction
theorem a_n_less_than_3 : ∀ n : ℕ, a n < 3 :=
by
  intros n
  induction n with
  | zero => sorry
  | succ n ih => sorry

end a_n_less_than_3_l151_151638


namespace trigonometric_identity_l151_151627

theorem trigonometric_identity
  (x : ℝ)
  (h_tan : Real.tan x = -1/2) :
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 :=
sorry

end trigonometric_identity_l151_151627


namespace exists_irrational_between_one_and_four_l151_151262

theorem exists_irrational_between_one_and_four : ∃ x : ℝ, 1 < x ∧ x < 4 ∧ irrational x := 
begin
  use real.sqrt 2,
  split,
  { norm_num,
    linarith [real.sqrt 2_pos] },
  split,
  { norm_num,
    linarith [real.sqrt 2_lt_four] },
  { exact real.sqrt_not_rational 2 0},
end

/-- sqrt(2) is greater than 1 -/
lemma real.sqrt 2_pos : 1 < real.sqrt 2 := sorry

/-- sqrt(2) is less than 4 -/
lemma real.sqrt 2_lt_four : real.sqrt 2 < 4 := sorry

/-- sqrt(2) is not a rational number -/
lemma real.sqrt_not_rational (n : ℕ) (k : ℕ) : ¬ rational (real.sqrt 2) := sorry

end exists_irrational_between_one_and_four_l151_151262


namespace number_of_blue_pencils_l151_151042

/-- In Carl's pencil case there are nine pencils. At least one of the pencils is blue.
In any group of four pencils, at least two have the same color. In any group of five pencils,
at most three have the same color. How many pencils are blue?
-/
theorem number_of_blue_pencils :
  ∃ (n : ℕ), n = 9 ∧
  ∃ (b : ℕ), b ≥ 1 ∧
  (∀ (x y z w : ℕ), x + y + z + w = 4 → x = y ∨ x = z ∨ x = w ∨ y = z ∨ y = w ∨ z = w) ∧
  (∀ (x y z w v : ℕ), x + y + z + w + v = 5 → x ≠ y ∨ x ≠ z ∨ x ≠ w ∨ x ≠ v ∨ y ≠ z ∨ y ≠ w ∨ y ≠ v ∨ z ≠ w ∨ z ≠ v ∨ w ≠ v) ∧
  b = 3 :=
begin
  sorry
end

end number_of_blue_pencils_l151_151042


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151555

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151555


namespace number_of_equilateral_triangles_l151_151318

-- Defining the problems and conditions:
def triangular_lattice_points : list (℣ctor ℝ) :=
  [ (1,0), (0.5, math.sqrt 3 / 2), (-0.5, math.sqrt 3 / 2), (-1,0), (-0.5, - math.sqrt 3 / 2), (0.5, - math.sqrt 3 / 2), (0,0) ]

def is_equilateral_triangle (a b c : ℤ ) : Prop := 
  dist a b = 1 ∧ dist b c = 1 ∧ dist c a = 1

-- The statement to prove the number of equilateral triangles formed is 12
theorem number_of_equilateral_triangles: 
  (∃ (T : finset (finset ℤ )), 
     T.card = 12 ∧ 
     (∀ t ∈ T, is_equilateral_triangle t)) :=
sorry

end number_of_equilateral_triangles_l151_151318


namespace max_area_of_cone_section_l151_151692

noncomputable def maximum_section_area (radius_sector : ℝ) (central_angle : ℝ) : ℝ :=
  let l := radius_sector
  let r := (central_angle / (2 * π)) * radius_sector in
  let a := (2 * sqrt (2)) in
  (1 / 2) * a * sqrt (l ^ 2 - (a ^ 2 / 4))

theorem max_area_of_cone_section :
  maximum_section_area 2 (5 * π / 3) = 2 :=
sorry

end max_area_of_cone_section_l151_151692


namespace sec_150_l151_151483

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151483


namespace translated_parabola_l151_151213

-- Define the initial quadratic equation
def initial_parabola (x : ℝ) : ℝ := x^2 - 2 * x + 4

-- Define the translation up by 3 units function
def translate_up (y : ℝ) (k : ℝ) : ℝ := y + k

-- Define the translation right by 1 unit function
def translate_right (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- The resulting equation after the specified translations
theorem translated_parabola:
  (translate_right (λ x, translate_up (initial_parabola x) 3) 1) = (λ x, x^2 - 4 * x + 10) :=
by
  -- Initializing lean to accept the theorem without an actual proof yet
  sorry

end translated_parabola_l151_151213


namespace find_b_l151_151611

theorem find_b 
  (b : ℝ)
  (h_pos : 0 < b)
  (h_geom_sequence : ∃ r : ℝ, 10 * r = b ∧ b * r = 2 / 3) :
  b = 2 * Real.sqrt 15 / 3 :=
by
  sorry

end find_b_l151_151611


namespace distance_from_pole_to_line_l151_151062

theorem distance_from_pole_to_line (rho theta : ℝ) (h : rho * real.cos theta = 1) :
  real.sqrt ((1 - 0)^2 + (0 - 0)^2) = 1 :=
by
  simp [h]
  sorry

end distance_from_pole_to_line_l151_151062


namespace min_value_of_expr_l151_151654

theorem min_value_of_expr {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = (1 / a) + (1 / b)) :
  ∃ x : ℝ, x = (1 / a) + (2 / b) ∧ x = 2 * Real.sqrt 2 :=
sorry

end min_value_of_expr_l151_151654


namespace fifth_sample_number_is_328_l151_151297

theorem fifth_sample_number_is_328 :
  let parts := [33, 21, 18, 34, 29, 78, 64, 56, 07, 32, 52, 42, 06, 44, 38, 12, 23, 43, 56, 77,
                35, 78, 90, 56, 42, 84, 42, 12, 53, 31, 34, 57, 86, 07, 36, 25, 30, 07, 32, 85,
                23, 45, 78, 89, 07, 23, 68, 96, 08, 04, 32, 56, 78, 08, 43, 67, 89, 53, 55, 77,
                34, 89, 94, 83, 75, 22, 53, 55, 78, 32, 45, 77, 89, 23, 45] in
  (get_5th_sample parts = 328) :=
sorry

/--
  Define a function to get the 5th sample number.
  We start scanning from the 6th number on the 5th row.
--/
def get_5th_sample (parts : List ℕ) : ℕ := 
  let rows := parts.splitOn 50
  let relevant_numbers := rows.drop 1 |> List.join
  let numbers := relevant_numbers.drop 5
  let samples := numbers.filter (fun x => x <= 700 && x > 0)
  samples.get! 4 -- The 5th (0-indexed) number is the 5th sample

end fifth_sample_number_is_328_l151_151297


namespace categorize_numbers_correct_l151_151356

noncomputable def given_numbers : List ℚ :=
  [ -8, -0.275, 22 / 7, 0, 10, -1.4040040004, -1 / 3, -2, Real.pi / 3, 0.5 ]

def positive_numbers (nums : List ℚ) : List ℚ :=
  nums.filter (λ x => x > 0)

def irrational_numbers (nums : List ℚ) : List ℚ :=
  nums.filter (λ x => ¬ is_rat x)

def integers (nums : List ℚ) : List ℚ :=
  nums.filter (λ x => ∃ n : ℤ, ↑n = x)

def negative_fractions (nums : List ℚ) : List ℚ :=
  nums.filter (λ x => x < 0 ∧ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

theorem categorize_numbers_correct :
  positive_numbers given_numbers = [22 / 7, 10, Real.pi / 3, 0.5] ∧
  irrational_numbers given_numbers = [-1.4040040004, Real.pi / 3] ∧
  integers given_numbers = [-8, 0, 10, -2] ∧
  negative_fractions given_numbers = [-0.275, -1 / 3] :=
by
  sorry

end categorize_numbers_correct_l151_151356


namespace booth_makes_50_per_day_on_popcorn_l151_151286

-- Define the conditions as provided
def daily_popcorn_revenue (P : ℝ) : Prop :=
  let cotton_candy_revenue := 3 * P
  let total_days := 5
  let rent := 30
  let ingredients := 75
  let total_expenses := rent + ingredients
  let profit := 895
  let total_revenue_before_expenses := profit + total_expenses
  total_revenue_before_expenses = 20 * P 

theorem booth_makes_50_per_day_on_popcorn : daily_popcorn_revenue 50 :=
  by sorry

end booth_makes_50_per_day_on_popcorn_l151_151286


namespace solution_set_transformation_l151_151670

variables (a b c α β : ℝ) (h_root : (α : ℝ) > 0)

open Set

def quadratic_inequality (x : ℝ) : Prop :=
  a * x^2 + b * x + c > 0

def transformed_inequality (x : ℝ) : Prop :=
  c * x^2 + b * x + a < 0

theorem solution_set_transformation :
  (∀ x, quadratic_inequality a b c x ↔ (α < x ∧ x < β)) →
  (∃ α β : ℝ, α > 0 ∧ (∀ x, transformed_inequality c b a x ↔ (x < 1/β ∨ x > 1/α))) :=
by
  sorry

end solution_set_transformation_l151_151670


namespace lambs_traded_for_goat_l151_151741

-- Definitions for the given conditions
def initial_lambs : ℕ := 6
def babies_per_lamb : ℕ := 2 -- each of 2 lambs had 2 babies
def extra_babies : ℕ := 2 * babies_per_lamb
def extra_lambs : ℕ := 7
def current_lambs : ℕ := 14

-- Proof statement for the number of lambs traded
theorem lambs_traded_for_goat : initial_lambs + extra_babies + extra_lambs - current_lambs = 3 :=
by
  sorry

end lambs_traded_for_goat_l151_151741


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151591

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151591


namespace conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l151_151849

theorem conversion1 : 4 * 60 + 35 = 275 := by
  sorry

theorem conversion2 : 4 * 1000 + 35 = 4035 := by
  sorry

theorem conversion3_minutes : 678 / 60 = 11 := by
  sorry

theorem conversion3_seconds : 678 % 60 = 18 := by
  sorry

theorem conversion4 : 120000 / 10000 = 12 := by
  sorry

end conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l151_151849


namespace quadrilateral_sum_of_opposite_sides_equal_l151_151219

-- Assume the existence of two circles touching each other externally
variables (circle1 circle2 : Circle)
-- Assume the existence of points corresponding to tangency
variables (A B C D : Point) 
-- Assume tangents form a quadrilateral with given points of tangency
variables (AB_tangent BC_tangent CD_tangent DA_tangent : Tangent)

-- Lean statement to prove the sum of opposite sides in a quadrilateral is equal
theorem quadrilateral_sum_of_opposite_sides_equal
  (touching_externally : circle1.touching_externally circle2)
  (tangents_drawn : AB_tangent.drawn_from_external_point circle1 circle2 ∧
                    BC_tangent.drawn_from_external_point circle1 circle2 ∧
                    CD_tangent.drawn_from_external_point circle1 circle2 ∧
                    DA_tangent.drawn_from_external_point circle1 circle2)
  (tangency_points : AB_tangent.tangent_points = (A, B) ∧ 
                      BC_tangent.tangent_points = (B, C) ∧ 
                      CD_tangent.tangent_points = (C, D) ∧ 
                      DA_tangent.tangent_points = (D, A))
  (quadrilateral : Quadrilateral (A, B, C, D)) :
  length(BC_tangent) + length(DA_tangent) = length(AB_tangent) + length(CD_tangent) :=
sorry

end quadrilateral_sum_of_opposite_sides_equal_l151_151219


namespace solve_quad_1_solve_quad_2_l151_151148

theorem solve_quad_1 :
  ∀ (x : ℝ), x^2 - 5 * x - 6 = 0 ↔ x = 6 ∨ x = -1 := by
  sorry

theorem solve_quad_2 :
  ∀ (x : ℝ), (x + 1) * (x - 1) + x * (x + 2) = 7 + 6 * x ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

end solve_quad_1_solve_quad_2_l151_151148


namespace parallel_planes_l151_151878

variables {Point Line Plane : Type}
variables (P : Point) (l m : Line) (α β : Plane)
variables (skew_lines : ¬∃ P, (P ∈ l) ∧ (P ∈ m));
variables (contains_m : ∀ P, P ∈ m → P ∈ α)
variables (parallel_m_β : ∀ P Q : Point, (P ∈ m) → (Q ∈ β) → ((P ≠ Q) ∧ (∃ P', (P' ∈ α) ∧ (P' ∈ β) → ∀ R : Point, R ∈ α ↔ R ∈ β)))
variables (contains_l : ∀ P, P ∈ l → P ∈ β)
variables (parallel_l_α : ∀ P Q : Point, (P ∈ l) → (Q ∈ α) → ((P ≠ Q) ∧ (∃ P', (P' ∈ α) ∧ (P' ∈ β) → ∀ R : Point, R ∈ α ↔ R ∈ β)))

theorem parallel_planes (skew_lines : ¬∃ P, (P ∈ l) ∧ (P ∈ m)) (contains_m : ∀ P, P ∈ m → P ∈ α) (parallel_m_β : ∀ P Q : Point, (P ∈ m) → (Q ∈ β) → ((P ≠ Q) ∧ (∃ P', (P' ∈ α) ∧ (P' ∈ β) → ∀ R : Point, R ∈ α ↔ R ∈ β))) (contains_l : ∀ P, P ∈ l → P ∈ β) (parallel_l_α : ∀ P Q : Point, (P ∈ l) → (Q ∈ α) → ((P ≠ Q) ∧ (∃ P', (P' ∈ α) ∧ (P' ∈ β) → ∀ R : Point, R ∈ α ↔ R ∈ β))) : α ∥ β :=
sorry

end parallel_planes_l151_151878


namespace verify_propositions_l151_151904

def prop1 : Prop :=
  ¬ (∀ x : ℝ, cos x > 0) = ∃ x : ℝ, cos x ≤ 0

def prop2 (a b c : Type) [LinearOrder a] : Prop :=
  (a ∥ b ↔ a ⊥ c ∧ b ⊥ c)

def prop3 (A B : ℝ) : Prop :=
  (A > B → sin A > sin B) → ¬ (sin A > sin B → A > B)

def prop4 : Prop :=
  ∃ (x1 y1 x2 y2: ℝ), (x2 - x1 = x1 ∧ y2 - y1 = y1)

def validPropositions : List ℕ :=
  [1, 4]

theorem verify_propositions 
  (a b c : Type) [LinearOrder a]
  (A B : ℝ)
  (x1 y1 x2 y2: ℝ) 
  : prop1 ∧ ¬ prop2 a b c ∧ ¬ prop3 A B ∧ prop4 
    ↔ validPropositions = [1, 4] := 
by 
  sorry

end verify_propositions_l151_151904


namespace find_fifth_term_l151_151980

-- Define the sequence based on the given conditions
def sequence (n : ℕ) : ℤ :=
  if n = 1 then 3
  else if n = 2 then 6
  else sequence (n - 1) - sequence (n - 2)

-- Define the theorem to be proved
theorem find_fifth_term : sequence 5 = -6 := sorry

end find_fifth_term_l151_151980


namespace larger_circle_radius_proof_l151_151779

noncomputable theory

-- Definitions of the conditions in the form of Lean constants
def small_circle_radius : ℝ := 1
def identical_circle_radius : ℝ := 1 + Real.sqrt 2

-- Define the radius of the larger concentric circle based on the given conditions
def larger_circle_radius : ℝ :=
  small_circle_radius + 2 * identical_circle_radius

-- The goal is to prove that the radius of the larger concentric circle is 3 + 2√2
theorem larger_circle_radius_proof :
  larger_circle_radius = 3 + 2 * Real.sqrt 2 :=
by
  -- Proof will be here
  sorry

end larger_circle_radius_proof_l151_151779


namespace prime_solution_exists_l151_151951

theorem prime_solution_exists (p : ℕ) (hp : Nat.Prime p) : ∃ x y z : ℤ, x^2 + y^2 + (p:ℤ) * z = 2003 := 
by 
  sorry

end prime_solution_exists_l151_151951


namespace translated_parabola_l151_151212

-- Define the initial quadratic equation
def initial_parabola (x : ℝ) : ℝ := x^2 - 2 * x + 4

-- Define the translation up by 3 units function
def translate_up (y : ℝ) (k : ℝ) : ℝ := y + k

-- Define the translation right by 1 unit function
def translate_right (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- The resulting equation after the specified translations
theorem translated_parabola:
  (translate_right (λ x, translate_up (initial_parabola x) 3) 1) = (λ x, x^2 - 4 * x + 10) :=
by
  -- Initializing lean to accept the theorem without an actual proof yet
  sorry

end translated_parabola_l151_151212


namespace min_value_of_expression_l151_151115

/-- Given that x is a positive real number,
    the minimum value of 9 * x^7 + 4 * x^(-6) is 13. -/
theorem min_value_of_expression (x : ℝ) (h : 0 < x) : 
  ∃ y, (∀ z, 0 < z → 9 * z^7 + 4 * z^(-6) ≥ y) ∧ y = 9 * x^7 + 4 * x^(-6) :=
sorry

end min_value_of_expression_l151_151115


namespace min_value_of_a_for_inverse_l151_151812

theorem min_value_of_a_for_inverse (a : ℝ) : 
  (∀ x y : ℝ, x ≥ a → y ≥ a → (x^2 + 4*x ≤ y^2 + 4*y ↔ x ≤ y)) → a = -2 :=
by
  sorry

end min_value_of_a_for_inverse_l151_151812


namespace problem_I_problem_II_problem_III_l151_151938

theorem problem_I (a : ℕ → ℤ) : 
  (1 - 2) ^ 7 = ∑ i in Finset.range 8, a i * 1 ^ i := sorry

theorem problem_II (a : ℕ → ℤ) :
  (1 - 2 * 0) ^ 7 = ∑ i in Finset.range 8, a i * 0 ^ i ∧
  (1 - 2 * -1) ^ 7 = 2187 :=
sorry

theorem problem_III :
  (∑ k in Finset.range 8, Nat.choose 7 k) = 2 ^ 7 := 
by simp [Nat.sum_choose_symmetric, Nat.pow_succ] ; exact rfl

end problem_I_problem_II_problem_III_l151_151938


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151524

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151524


namespace shelves_used_l151_151315

def initial_books : Nat := 87
def sold_books : Nat := 33
def books_per_shelf : Nat := 6

theorem shelves_used :
  (initial_books - sold_books) / books_per_shelf = 9 := by
  sorry

end shelves_used_l151_151315


namespace Mary_works_hours_on_Tuesday_and_Thursday_l151_151118

theorem Mary_works_hours_on_Tuesday_and_Thursday 
  (h_mon_wed_fri : ∀ (d : ℕ), d = 3 → 9 * d = 27)
  (weekly_earnings : ℕ)
  (hourly_rate : ℕ)
  (weekly_hours_mon_wed_fri : ℕ)
  (tue_thu_hours : ℕ) :
  weekly_earnings = 407 →
  hourly_rate = 11 →
  weekly_hours_mon_wed_fri = 9 * 3 →
  weekly_earnings - weekly_hours_mon_wed_fri * hourly_rate = tue_thu_hours * hourly_rate →
  tue_thu_hours = 10 :=
by
  intros hearnings hrate hweek hsub
  sorry

end Mary_works_hours_on_Tuesday_and_Thursday_l151_151118


namespace complex_in_second_quadrant_l151_151720

def euler_identity (x : ℝ) : ℂ := complex.exp (x * complex.I)

theorem complex_in_second_quadrant : euler_identity (2 * Real.pi / 3) = -1 / 2 + complex.I * (Real.sqrt 3 / 2) →
  (-1 / 2 + complex.I * (Real.sqrt 3 / 2)).re < 0 ∧
  (-1 / 2 + complex.I * (Real.sqrt 3 / 2)).im > 0 :=
by
  sorry

end complex_in_second_quadrant_l151_151720


namespace ellipse_area_l151_151043

/-- 
In a certain ellipse, the endpoints of the major axis are (1, 6) and (21, 6). 
Also, the ellipse passes through the point (19, 9). Prove that the area of the ellipse is 50π. 
-/
theorem ellipse_area : 
  let a := 10
  let b := 5 
  let center := (11, 6)
  let endpoints_major := [(1, 6), (21, 6)]
  let point_on_ellipse := (19, 9)
  ∀ x y, ((x - 11)^2 / a^2) + ((y - 6)^2 / b^2) = 1 → 
    (x, y) = (19, 9) →  -- given point on the ellipse
    (endpoints_major = [(1, 6), (21, 6)]) →  -- given endpoints of the major axis
    50 * Real.pi = π * a * b := 
by
  sorry

end ellipse_area_l151_151043


namespace smallest_n_for_sum_and_squares_l151_151623

theorem smallest_n_for_sum_and_squares :
  ∃ (n : ℕ), (∀ (x : fin n → ℝ), (∀ i, -1 < x i ∧ x i < 1) →
  ∑ i, x i = 0 →
  ∑ i, (x i)^2 = 42) ∧ (∀ m, m < n → ¬(∃ (y : fin m → ℝ), 
  (∀ j, -1 < y j ∧ y j < 1) ∧ 
  ∑ j, y j = 0 ∧ 
  ∑ j, (y j)^2 = 42)) :=
sorry

end smallest_n_for_sum_and_squares_l151_151623


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151579

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151579


namespace coefficient_of_x4_in_expansion_l151_151242

noncomputable def problem_statement : ℕ :=
  let n := 8
  let a := 2
  let b := 3
  let k := 4
  binomial n k * (b ^ k) * (a ^ (n - k))

theorem coefficient_of_x4_in_expansion :
  problem_statement = 90720 :=
by
  sorry

end coefficient_of_x4_in_expansion_l151_151242


namespace most_reasonable_sampling_method_is_stratified_l151_151313

def Grade := Type
def Students := Type
def Survey := {students: Students // students ∈ Grade}

constants (G10 G11 G12 : Grade) (sampling : Survey → Students)

-- The condition: School plans to sample a proportionate number of students from Grade 10, Grade 11, and Grade 12
axiom sampling_proportionate : 
  ∀ (g : Grade), g ∈ {G10, G11, G12} → ∃ (s : Students), sampling s = g

-- Theorem statement proving the most reasonable sampling method.
theorem most_reasonable_sampling_method_is_stratified : 
  ( ∀ (g : Grade), g ∈ {G10, G11, G12} → ∃ (s : Students), sampling s = g ) → 
  "stratified_sampling" :=
begin
  -- Proof skipped
  sorry
end

end most_reasonable_sampling_method_is_stratified_l151_151313


namespace max_good_pairs_l151_151722

open Locale BigOperators

def is_good_pair (n : ℕ) (colors : Fin (2 * n - 1) → Bool) (i j : Fin (2 * n - 1)) : Prop :=
  i.val ≤ j.val ∧ (∑ k in Finset.Ico i j.succ, if colors k then 1 else 0) % 2 = 1

theorem max_good_pairs (n : ℕ) (hpos : 0 < n) :
  ∃ (colors : Fin (2 * n - 1) → Bool), 
    (∀ (i j : Fin (2 * n - 1)), is_good_pair n colors i j → ∃ (p q : ℕ), p + q = n^2) :=
sorry

end max_good_pairs_l151_151722


namespace equal_sets_of_conditions_l151_151163

theorem equal_sets_of_conditions
  (a b c : ℝ)
  (h_diff: a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_set_equality : {a + b, b + c, c + a} = {a * b, b * c, c * a}) :
  {a, b, c} = {a^2 - 2, b^2 - 2, c^2 - 2} :=
sorry

end equal_sets_of_conditions_l151_151163


namespace power_of_power_eq_512_l151_151226

theorem power_of_power_eq_512 : (2^3)^3 = 512 := by
  sorry

end power_of_power_eq_512_l151_151226


namespace sum_of_coefficients_l151_151737

-- Define the sequence and the recurrence relation
def sequence (v : ℕ → ℤ) : Prop :=
  v 1 = 7 ∧ ∀ n : ℕ, v (n + 1) - v n = 6 * n - 1

-- Define the polynomial form of v_n and the sum of its coefficients
def polynomial_form (v : ℕ → ℤ) : Prop :=
  ∃ A B C : ℤ, ∀ n : ℕ, v n = A * n ^ 2 + B * n + C ∧ (A + B + C = 7)
  
-- The property we want to prove
theorem sum_of_coefficients (v : ℕ → ℤ) (h : sequence v) : polynomial_form v :=
  sorry

end sum_of_coefficients_l151_151737


namespace greatest_common_multiple_of_10_and_15_lt_120_l151_151254

theorem greatest_common_multiple_of_10_and_15_lt_120 : 
  ∃ (m : ℕ), lcm 10 15 = 30 ∧ m ∈ {i | i < 120 ∧ ∃ (k : ℕ), i = k * 30} ∧ m = 90 := 
sorry

end greatest_common_multiple_of_10_and_15_lt_120_l151_151254


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151378

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151378


namespace opposite_of_neg_two_l151_151188

theorem opposite_of_neg_two : ∀ x : ℤ, (-2 + x = 0) → (x = 2) :=
begin
  assume x hx,
  sorry

end opposite_of_neg_two_l151_151188


namespace generating_function_T_generating_function_U_l151_151928

noncomputable def L (x z : ℂ) : ℂ := (2 - 2 * x * z) / (1 - 2 * x * z + z^2)
noncomputable def F (x z : ℂ) : ℂ := z / (1 - x * z - z^2)
noncomputable def Tn (x : ℂ) (n : ℕ) : ℂ := (1 / 2) * complex.I.pow n * L (2 * complex.I * x) (-complex.I * z)
noncomputable def Un (x : ℂ) (n : ℕ) : ℂ := complex.I.pow (n + 1) * F (2 * complex.I * x) (-complex.I * z)

noncomputable def FT (x z : ℂ) : ℂ := ∑' n, Tn x n * z ^ n
noncomputable def FU (x z : ℂ) : ℂ := ∑' n, Un x n * z ^ n

theorem generating_function_T (x z : ℂ) : FT x z = (1 - x * z) / (1 - 2 * x * z + z^2) :=
sorry

theorem generating_function_U (x z : ℂ) : FU x z = 1 / (1 - 2 * x * z + z^2) :=
sorry

end generating_function_T_generating_function_U_l151_151928


namespace problem_a_problem_b_l151_151847

-- Problem (a): Prove that (1 + 1/x)(1 + 1/y) ≥ 9 given x > 0, y > 0, and x + y = 1
theorem problem_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 := sorry

-- Problem (b): Prove that 0 < u + v - uv < 1 given 0 < u < 1 and 0 < v < 1
theorem problem_b (u v : ℝ) (hu : 0 < u) (hu1 : u < 1) (hv : 0 < v) (hv1 : v < 1) : 
  0 < u + v - u * v ∧ u + v - u * v < 1 := sorry

end problem_a_problem_b_l151_151847


namespace sec_150_l151_151482

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151482


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151433

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151433


namespace requires_irish_sea_l151_151307

def PathSegment := String
def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def path_segments : List PathSegment := ["Segment1", "Irish Sea", "Segment2", ..., "Segment18"]

def valid_path (segments: List PathSegment) : Prop :=
  is_even segments.length ∧ "Irish Sea" ∈ segments

theorem requires_irish_sea :
  ∀ (segments : List PathSegment), valid_path segments → "Irish Sea" ∈ segments :=
by
  intros segments h
  sorry

end requires_irish_sea_l151_151307


namespace coefficient_of_x4_in_expansion_l151_151235

theorem coefficient_of_x4_in_expansion (x : ℤ) :
  let a := 3
  let b := 2
  let n := 8
  let k := 4
  (finset.sum (finset.range (n + 1)) (λ r, binomial n r * a^r * b^(n-r) * x^r) = 
  ∑ r in finset.range (n + 1), binomial n r * a^r * b^(n - r) * x^r)

  ∑ r in finset.range (n + 1), 
    if r = k then 
      binomial n r * a^r * b^(n-r)
    else 
      0 = 90720
:= 
by
  sorry

end coefficient_of_x4_in_expansion_l151_151235


namespace find_CB_l151_151626

theorem find_CB
  (O K M N C B : Point)
  (R α b : ℝ)
  (h_outside_circle : K ∉ circle O R)
  (h_tangent_MK : tangent MK K circle O R)
  (h_tangent_NK : tangent NK K circle O R)
  (h_tangency_M : tangency_point M MK circle O R)
  (h_tangency_N : tangency_point N NK circle O R)
  (h_chord_MN : chord M N)
  (h_C_on_MN : point_on_segment C M N)
  (h_MC_lt_CN : MC < CN)
  (h_perpendicular : orthogonal_to_segment C OC NK B)
  : CB = cot (α / 2) * sqrt (R^2 + b^2 - 2 * R * b * cos (α / 2)) :=
sorry

end find_CB_l151_151626


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151369

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151369


namespace sec_150_eq_l151_151399

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151399


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151362

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151362


namespace vertices_after_2007_operations_l151_151903

def initial_vertices : List Char := ['A', 'B', 'C', 'D']

def rotate_90_clockwise (vertices : List Char) : List Char :=
  [vertices[3], vertices[0], vertices[1], vertices[2]]

def reflect_vertical (vertices : List Char) : List Char :=
  [vertices[2], vertices[1], vertices[0], vertices[3]]

def reflect_horizontal (vertices : List Char) : List Char :=
  [vertices[3], vertices[2], vertices[1], vertices[0]]

def perform_transformations (vertices : List Char) (n : Nat) : List Char :=
  let sequence := [rotate_90_clockwise, reflect_vertical, reflect_horizontal]
  (List.foldl (λ v, id) vertices (List.range n |>.map (λ i, sequence[i % 3])))

theorem vertices_after_2007_operations : perform_transformations initial_vertices 2007 = ['D', 'C', 'B', 'A'] :=
sorry

end vertices_after_2007_operations_l151_151903


namespace perimeter_of_resulting_figure_l151_151079

-- Define the perimeters of the squares
def perimeter_small_square : ℕ := 40
def perimeter_large_square : ℕ := 100

-- Define the side lengths of the squares
def side_length_small_square := perimeter_small_square / 4
def side_length_large_square := perimeter_large_square / 4

-- Define the total perimeter of the uncombined squares
def total_perimeter_uncombined := perimeter_small_square + perimeter_large_square

-- Define the shared side length
def shared_side_length := side_length_small_square

-- Define the perimeter after considering the shared side
def resulting_perimeter := total_perimeter_uncombined - 2 * shared_side_length

-- Prove that the resulting perimeter is 120 cm
theorem perimeter_of_resulting_figure : resulting_perimeter = 120 := by
  sorry

end perimeter_of_resulting_figure_l151_151079


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151521

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151521


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151594

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151594


namespace matrix_product_nonzero_l151_151098

noncomputable def A_i (n : ℕ) (i : ℕ) [1 ≤ i ∧ i ≤ k] : Matrix (Fin n) (Fin n) ℝ := sorry

theorem matrix_product_nonzero {n k : ℕ} (h : n > k) :
  ∀ (A_i : Fin k → Matrix (Fin n) (Fin n) ℝ),
    (∀ i, rank (A_i i) = n - 1) →
    (A_i 0 ⬝ A_i 1 ⬝ ... ⬝ A_i (Fin k).last) ≠ 0 := sorry

end matrix_product_nonzero_l151_151098


namespace cost_of_football_correct_l151_151744

-- We define the variables for the costs
def total_amount_spent : ℝ := 20.52
def cost_of_marbles : ℝ := 9.05
def cost_of_baseball : ℝ := 6.52
def cost_of_football : ℝ := total_amount_spent - cost_of_marbles - cost_of_baseball

-- We now state what needs to be proven: that Mike spent $4.95 on the football.
theorem cost_of_football_correct : cost_of_football = 4.95 := by
  sorry

end cost_of_football_correct_l151_151744


namespace power_cycle_i_pow_2012_l151_151279

-- Define the imaginary unit i as a complex number
def i : ℂ := Complex.I

-- Define the periodic properties of i
theorem power_cycle (n : ℕ) : Complex := 
  match n % 4 with
  | 0 => 1
  | 1 => i
  | 2 => -1
  | 3 => -i
  | _ => 0 -- this case should never happen

-- Using the periodic properties
theorem i_pow_2012 : (i ^ 2012) = 1 := by
  sorry

end power_cycle_i_pow_2012_l151_151279


namespace limsup_ge_e_l151_151268

theorem limsup_ge_e (a : ℕ → ℝ) (h_pos : ∀ (n : ℕ), 0 < a n) :
  limsup (λ (n : ℕ), (a 1 + a (n + 1)) / a n)^n ≥ real.exp 1 :=
by sorry

end limsup_ge_e_l151_151268


namespace sec_150_eq_l151_151460

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151460


namespace simplify_cubicroot_1600_l151_151848

theorem simplify_cubicroot_1600 : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c^3 * d = 1600) ∧ (c + d = 102) := 
by 
  sorry

end simplify_cubicroot_1600_l151_151848


namespace min_value_of_k_l151_151790

def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ n : ℕ, a + b = n^2 ∧ a ≠ b

theorem min_value_of_k : ∃ (k : ℕ), (∀ (partition : Finset (Finset ℕ)), 
  (partition.card = k ∧ (∀ (subset : Finset ℕ), subset ∈ partition → subset ⊆ M) ∧ 
  (∀ {a b : ℕ}, a ∈ M → b ∈ M → satisfies_condition a b → 
  (∀ subset ∈ partition, a ∈ subset → b ∉ subset))) 
  → k = 3) :=
sorry

end min_value_of_k_l151_151790


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151430

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151430


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151561

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151561


namespace quadratic_has_distinct_real_roots_l151_151034

theorem quadratic_has_distinct_real_roots :
  ∃ (a c : ℝ), a = -1 ∧ c = 3 ∧ ( 9 - 4 * a * c > 0 ) := by
  use [-1, 3]
  split
  · rfl
  split
  · rfl
  · sorry

end quadratic_has_distinct_real_roots_l151_151034


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151582

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151582


namespace find_x_for_f_eq_10_l151_151165

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 1 else -2 * x

theorem find_x_for_f_eq_10 (x : ℝ) (h : f x = 10) : x = -3 :=
by
  sorry

end find_x_for_f_eq_10_l151_151165


namespace double_single_count_l151_151283

def is_double_single (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  let d2 := (n / 10) % 10; let d3 := n % 10; let d1 := n / 100
  in d1 = d2 ∧ d1 ≠ d3

theorem double_single_count : {n : ℕ | is_double_single n}.to_finset.card = 81 := 
by sorry

end double_single_count_l151_151283


namespace davids_profit_l151_151312

-- Definitions of conditions
def weight_of_rice : ℝ := 50
def cost_of_rice : ℝ := 50
def selling_price_per_kg : ℝ := 1.20

-- Theorem stating the expected profit
theorem davids_profit : 
  (selling_price_per_kg * weight_of_rice) - cost_of_rice = 10 := 
by 
  -- Proofs are omitted.
  sorry

end davids_profit_l151_151312


namespace curve_represents_parabola_l151_151599

theorem curve_represents_parabola (r θ : ℝ) (h : r = 3 * Real.tan θ * Real.sec θ) : 
  ∃ a b c, ∀ x y : ℝ, (x^2 = 3 * y) → (r * Real.cos(θ)^2 = 3 * y) :=
sorry

end curve_represents_parabola_l151_151599


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151458

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151458


namespace count_integers_absolute_value_less_than_4pi_l151_151683

theorem count_integers_absolute_value_less_than_4pi : 
  let pi := Real.pi in
  finset.card ({ x : ℤ | |(x : ℝ)| < 4 * pi }) = 25 := 
by
  sorry

end count_integers_absolute_value_less_than_4pi_l151_151683


namespace initial_shares_bought_l151_151816

variable (x : ℕ) -- x is the number of shares Tom initially bought

-- Conditions:
def initial_cost_per_share : ℕ := 3
def num_shares_sold : ℕ := 10
def selling_price_per_share : ℕ := 4
def doubled_value_per_remaining_share : ℕ := 2 * initial_cost_per_share
def total_profit : ℤ := 40

-- Proving the number of shares initially bought
theorem initial_shares_bought (h : num_shares_sold * selling_price_per_share - x * initial_cost_per_share = total_profit) :
  x = 10 := by sorry

end initial_shares_bought_l151_151816


namespace proof_problem_l151_151886

-- Define the centroid property and circumcircle intersection
variables {A B C G A' B' C' : Type}
variables [IsCentroid G A B C] [IntersectsCircumcircle A G A'] [IntersectsCircumcircle B G B'] [IntersectsCircumcircle C G C']

-- Theorem stating the three proofs.
theorem proof_problem :
  (∃ G A' B' C', IsCentroid G A B C ∧ IntersectsCircumcircle A G A' ∧
   IntersectsCircumcircle B G B' ∧ IntersectsCircumcircle C G C') →
  ( (AG / GA' + BG / GB' + CG / GC') = 3 ∧
    (GA' / GA + GB' / BG + GC' / CG) ≥ 3 ∧
    (AG / GA' ≤ 1 ∨ BG / GB' ≤ 1 ∨ CG / GC' ≤ 1) ) :=
by
  sorry

end proof_problem_l151_151886


namespace complex_division_identity_l151_151276

def i : ℂ := complex.I

theorem complex_division_identity :
  (i / (1 + i)) = (1 / 2 + (1 / 2) * i) :=
by
  sorry

end complex_division_identity_l151_151276


namespace find_T_div_12_l151_151894

def k_pretty (n k : ℕ) : Prop :=
  (∀ d, d ∣ n → ∃ l : ℕ, l ∣ k ∧ l ∣ d ∧ l ∣ (n/d) ∧ nat.gcd (n/k) l = 1)

def twelve_pretty (n : ℕ) : Prop :=
  k_pretty n 12 ∧ nat.divisors n = 12

def sum_of_twelve_pretty (T : ℕ) : Prop :=
  T = ∑ n in finset.filter (λ n, twelve_pretty n) (finset.range 2023), n

theorem find_T_div_12 :
  sum_of_twelve_pretty T →
  T = 801
  → T / 12 = 66.75 :=
sorry

end find_T_div_12_l151_151894


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151387

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151387


namespace sec_150_eq_l151_151465

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151465


namespace rationalize_cube_root_sum_l151_151133

theorem rationalize_cube_root_sum :
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  A + B + C + D = 51 :=
by
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  have step1 : (a^3 = 5) := by sorry
  have step2 : (b^3 = 3) := by sorry
  have denom_eq : denom = 2 := by sorry
  have frac_simp : fraction = (A^(1/3) + B^(1/3) + C^(1/3)) / D := by sorry
  show A + B + C + D = 51
  sorry

end rationalize_cube_root_sum_l151_151133


namespace coefficient_x4_in_expansion_l151_151237

theorem coefficient_x4_in_expansion : 
  (∑ k in Finset.range (9), (Nat.choose 8 k) * (3 : ℤ)^k * (2 : ℤ)^(8-k) * (X : ℤ[X])^k).coeff 4 = 90720 :=
by
  sorry

end coefficient_x4_in_expansion_l151_151237


namespace medal_allocation_l151_151711

-- Define the participants
inductive Participant
| Jiri
| Vit
| Ota

open Participant

-- Define the medals
inductive Medal
| Gold
| Silver
| Bronze

open Medal

-- Define a structure to capture each person's statement
structure Statements :=
  (Jiri : Prop)
  (Vit : Prop)
  (Ota : Prop)

-- Define the condition based on their statements
def statements (m : Participant → Medal) : Statements :=
  {
    Jiri := m Ota = Gold,
    Vit := m Ota = Silver,
    Ota := (m Ota ≠ Gold ∧ m Ota ≠ Silver)
  }

-- Define the condition for truth-telling and lying based on medals
def truths_and_lies (m : Participant → Medal) (s : Statements) : Prop :=
  (m Jiri = Gold → s.Jiri) ∧ (m Jiri = Bronze → ¬ s.Jiri) ∧
  (m Vit = Gold → s.Vit) ∧ (m Vit = Bronze → ¬ s.Vit) ∧
  (m Ota = Gold → s.Ota) ∧ (m Ota = Bronze → ¬ s.Ota)

-- Define the final theorem to be proven
theorem medal_allocation : 
  ∃ (m : Participant → Medal), 
    truths_and_lies m (statements m) ∧ 
    m Vit = Gold ∧ 
    m Ota = Silver ∧ 
    m Jiri = Bronze := 
sorry

end medal_allocation_l151_151711


namespace common_point_on_circle_l151_151344

theorem common_point_on_circle (p q : ℝ) (h_intersects_axes : ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 ≠ 0) ∧ (x2 ≠ 0) ∧ (x1 + x2 = -2 * p) ∧ (x1 * x2 = q)) :
  let center := (-p, q / 2)
  let radius := Math.sqrt (p^2 + (q^2 / 4))
  (0, 1) ∈ set_of (λ ⟨x, y⟩, (x + p)^2 + (y - q / 2)^2 = p^2 + (q^2 / 4)) :=
begin
  sorry
end

end common_point_on_circle_l151_151344


namespace solve_system_of_inequalities_simplified_expression_correct_l151_151757

variable (x : ℝ)

def system_of_inequalities (x : ℝ) : Prop :=
  3 * x + 6 ≥ 5 * (x - 2) ∧ (x - 5) / 2 - (4 * x - 3) / 3 < 1

def solution_set : Set ℝ := {x | -3 < x ∧ x ≤ 8}

theorem solve_system_of_inequalities :
  ∀ x, system_of_inequalities x → x ∈ solution_set := 
by
  intro x h
  sorry

def simplify_expr (x : ℝ) : ℝ :=
  (1 - 2 / (x - 1)) / ((x^2 - 6 * x + 9) / (x - 1))

def simplified_expr (x : ℝ) : ℝ := 1 / (x - 3)

theorem simplified_expression_correct :
  ∀ x, x ≠ 1 ∧ x ≠ 3 → simplify_expr x = simplified_expr x :=
by
  intro x h
  sorry

example : simplify_expr 0 = -1 / 3 :=
by
  sorry

example : simplify_expr 1 = simplified_expr 1 :=
by
  have : simplify_expr 1 = 0 / 0, by sorry   -- Representing undefined expression
  sorry

example : simplify_expr 2 = -1 :=
by
  sorry

example : simplify_expr 3 = simplified_expr 3 :=
by
  have : simplify_expr 3 = 0 / 0, by sorry   -- Representing undefined expression
  sorry

end solve_system_of_inequalities_simplified_expression_correct_l151_151757


namespace simplify_expression_l151_151831

theorem simplify_expression (x : ℝ) (h : x ≤ 2) : 
  (Real.sqrt (x^2 - 4*x + 4) - Real.sqrt (x^2 - 6*x + 9)) = -1 :=
by 
  sorry

end simplify_expression_l151_151831


namespace maximum_bc_range_f_theta_l151_151039

theorem maximum_bc (a b c θ : ℝ) (h1 : a = 4) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b * (Real.sin θ) = 8) :
  b * c ≤ 16 :=
sorry

theorem range_f_theta (θ : ℝ) (f : ℝ → ℝ) (h1 : ∀ θ, f θ = sqrt 3 * Real.sin (2 * θ) + Real.cos (2 * θ) - 1)
  (h2 : 0 < θ ∧ θ ≤ Real.pi / 3) :
  set.range f = set.Icc 0 1 :=
sorry

end maximum_bc_range_f_theta_l151_151039


namespace six_dollar_three_eq_eighteen_l151_151909

-- Define the notation for the operation
notation a " \$ " b => 4 * a - 2 * b

-- The statement to prove
theorem six_dollar_three_eq_eighteen : 6 \$ 3 = 18 :=
by
  -- The steps of applying the definition and performing arithmetic will be filled in the proof
  sorry

end six_dollar_three_eq_eighteen_l151_151909


namespace sum_of_fractions_l151_151664

-- Define the function f
def f (x : ℝ) : ℝ := 4^x / (4^x + 2)

-- State the theorem
theorem sum_of_fractions :
  (List.range 1000).sum (λ n => f ((n + 1) / 1001)) = 500 :=
sorry

end sum_of_fractions_l151_151664


namespace find_extreme_values_find_min_m_l151_151971

-- Part (Ⅰ) proof problem
theorem find_extreme_values :
  ∃ x_max x_min,
    (x_max = 1/2 ∧ f(x_max) = 3/4 - ln 2) ∧
    (x_min = 1 ∧ f(x_min) = 0) := by
  /- Define the function -/
  let f : ℝ → ℝ := λ x, ln x + x^2 - 3*x + 2
  /- Provide the proof steps -/
  sorry

-- Part (Ⅱ) proof problem
theorem find_min_m {m : ℝ} :
  (∃ a b, a ≠ b ∧ g(a) = 0 ∧ g(b) = 0) → m ≥ 3 := by
  /- Define the functions -/
  let f : ℝ → ℝ := λ x, ln x + x^2 - 3*x + 2
  let g : ℝ → ℝ := λ x, f(x) + (3 - m) * x
  /- Provide the proof steps -/
  sorry

end find_extreme_values_find_min_m_l151_151971


namespace sqrt_inequality_satisfied_l151_151308

theorem sqrt_inequality_satisfied (x : ℝ) (hx : x > 0) : sqrt (3 * x) < 5 * x ↔ x > 1 / 15 := 
sorry

end sqrt_inequality_satisfied_l151_151308


namespace smallest_integer_is_nine_l151_151808

theorem smallest_integer_is_nine 
  (a b c : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : a + b + c = 90) 
  (h3 : (a:ℝ)/b = 2/3) 
  (h4 : (b:ℝ)/c = 3/5) : 
  a = 9 :=
by 
  sorry

end smallest_integer_is_nine_l151_151808


namespace area_quadrilateral_FGCD_l151_151065

theorem area_quadrilateral_FGCD
  (AB CD AD BC : ℝ)
  (h_parallel: AB = 10 ∧ CD = 26)
  (h_altitude: AD - BC = 15)
  (E F: ℝ)
  (h_E: E = (1 / 2) * AD)
  (h_F: F = (1 / 2) * BC)
  (G: ℝ)
  (h_G: G = (2 / 3) * AD):

  let DG := (1 / 3) * AD in
  let DE := (1 / 2) * AD in
  let EG := DE - DG in
  let FG := ((AB + CD) / 2) - EG in
  let area := (1 / 2) * (CD + FG) * AD in
  area = 311.25 :=
begin
  sorry
end

end area_quadrilateral_FGCD_l151_151065


namespace evaluate_expression_l151_151203

theorem evaluate_expression : (1:ℤ)^10 + (-1:ℤ)^8 + (-1:ℤ)^7 + (1:ℤ)^5 = 2 := by
  sorry

end evaluate_expression_l151_151203


namespace skyler_song_difference_l151_151146

variable (X : ℕ)
variable (total_songs : ℕ := 80)
variable (top10_hits : ℕ := 25)
variable (unreleased_songs : ℕ := top10_hits - 5)
variable (top100_hits : ℕ := X)

theorem skyler_song_difference :
  total_songs = top10_hits + top100_hits + unreleased_songs →
  top100_hits - top10_hits = 10 :=
by
  intro h
  rw [total_songs, top10_hits, unreleased_songs] at h
  rw [Nat.add_sub_of_le (by decide : 20 ≤ 25)] at h
  norm_num at h
  linarith

end skyler_song_difference_l151_151146


namespace discriminant_of_quadratic_equation_l151_151822

noncomputable def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic_equation : discriminant 5 (-11) (-18) = 481 := by
  sorry

end discriminant_of_quadratic_equation_l151_151822


namespace ratio_rounded_to_nearest_tenth_l151_151697

theorem ratio_rounded_to_nearest_tenth : 
  Float.round (12 / 17 : ℝ) 1 = 0.7 :=
sorry

end ratio_rounded_to_nearest_tenth_l151_151697


namespace least_value_of_a_plus_b_l151_151105

def a_and_b (a b : ℕ) : Prop :=
  (Nat.gcd (a + b) 330 = 1) ∧ 
  (a^a % b^b = 0) ∧ 
  (¬ (a % b = 0))

theorem least_value_of_a_plus_b :
  ∃ (a b : ℕ), a_and_b a b ∧ a + b = 105 :=
sorry

end least_value_of_a_plus_b_l151_151105


namespace ratio_celeste_bianca_l151_151335

-- Definitions based on given conditions
def bianca_hours : ℝ := 12.5
def celest_hours (x : ℝ) : ℝ := 12.5 * x
def mcclain_hours (x : ℝ) : ℝ := 12.5 * x - 8.5

-- The total time worked in hours
def total_hours : ℝ := 54

-- The ratio to prove
def celeste_bianca_ratio : ℝ := 2

-- The proof statement
theorem ratio_celeste_bianca (x : ℝ) (hx :  12.5 + 12.5 * x + (12.5 * x - 8.5) = total_hours) :
  celest_hours 2 / bianca_hours = celeste_bianca_ratio :=
by
  sorry

end ratio_celeste_bianca_l151_151335


namespace trajectory_is_line_segment_l151_151017

-- Definitions for the fixed points
def F1 : ℝ × ℝ := (5, 0)
def F2 : ℝ × ℝ := (-5, 0)

-- Definition of the distance function between two points
def dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

-- Condition for the moving point M
def on_trajectory (M : ℝ × ℝ) : Prop := dist M F1 + dist M F2 = 10

-- The problem statement to be proved
theorem trajectory_is_line_segment (M : ℝ × ℝ) (h : on_trajectory M) : 
    ∃ x : ℝ, (x = 0 → M = F1) ∧ (x = 5 → M = F2) ∧ 
    (0 ≤ x ∧ x ≤ 5 → M = ((5 - x) * F1.1 / 5, 0)) :=
sorry

end trajectory_is_line_segment_l151_151017


namespace area_triangle_ABC_is_300_l151_151227

noncomputable def area_of_triangle_ABC (A B C D : Point) (AC DC AB : ℝ) (condition1 : AC = 25) (condition2 : DC = 24) (condition3 : AB = 25) (angleD_right : ∠D is_right_angle) : ℝ :=
  let BD := 24 in -- Given BD as height for right triangle ABD.
  1 / 2 * AB * BD

theorem area_triangle_ABC_is_300 (A B C D : Point) (h_coplanar : coplanar A B C D) 
  (h_angleD_right : ∠D = 90) (h_AC : dist A C = 25) (h_DC : dist D C = 24)
  (h_AB : dist A B = 25) :
  area_of_triangle_ABC A B C D 25 24 25 h_AC h_DC h_AB h_angleD_right = 300 := 
  sorry  -- Proof omitted.

end area_triangle_ABC_is_300_l151_151227


namespace area_of_parallelogram_l151_151598

theorem area_of_parallelogram (base height : ℕ) (h1 : base = 28) (h2 : height = 32) :
  base * height = 896 :=
by
  rw [h1, h2]
  norm_num
  done

end area_of_parallelogram_l151_151598


namespace max_non_intersecting_chords_l151_151751

theorem max_non_intersecting_chords (n : ℕ) (colors : ℕ) (points : Fin n) :
  n = 2006 → colors = 17 → (∃ c : set (Fin n), c.card = 118 ∧ ∃ p : set (Fin n), p.card = 118) →
  ∃ k : ℕ, k = 117 ∧ ∃ chords, (∀ (chord : set (Fin n)), chord.card = 2 → ∀ p₁ p₂ ∈ chord, p₁ ≠ p₂ → ∃ color, chord ⊆ color) ∧ (∀ chord₁ chord₂, chord₁ ≠ chord₂ → chord₁ ∩ chord₂ = ∅) :=
begin
  intros hn hcolors hpoints,
  use [117, _],
  sorry
end

end max_non_intersecting_chords_l151_151751


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151546

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151546


namespace seq_a_eq_seq_b_l151_151096

-- Define auxiliary functions for sequences a and b.
def seq_a (n : ℕ) (epsilon : ℕ → ℕ) : ℕ → ℕ
| 0 => 1
| 1 => 7
| i + 2 => if epsilon i = 0 then 2 * seq_a i + 3 * seq_a (i + 1) else 3 * seq_a i + seq_a (i + 1)

def seq_b (n : ℕ) (epsilon : ℕ → ℕ) : ℕ → ℕ
| 0 => 1
| 1 => 7
| i + 2 => if epsilon (n - i - 2) = 0 then 2 * seq_b i + 3 * seq_b (i + 1) else 3 * seq_b i + seq_b (i + 1)

-- Theorem stating that seq_a and seq_b are equal at n.
theorem seq_a_eq_seq_b (n : ℕ) (epsilon : ℕ → ℕ) (hn_pos : 0 < n) (hepsilon_range : ∀ i, i < n - 1 → epsilon i = 0 ∨ epsilon i = 1) :
  seq_a n epsilon n = seq_b n epsilon n := 
sorry

end seq_a_eq_seq_b_l151_151096


namespace repetend_of_five_over_seventeen_l151_151932

theorem repetend_of_five_over_seventeen :
  ∃ s : String, s = "294117" ∧ (∃ (f : ℕ → ℕ), ∀ n, decimal_of_fraction 5 17 (n + 1) = some (f n)) :=
sorry

end repetend_of_five_over_seventeen_l151_151932


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151457

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151457


namespace range_of_a_l151_151018

theorem range_of_a (a : ℝ) (θ : ℝ) :
  (0 < θ ∧ θ < π / 12) →
  let l1 := ∀ x, y = x,
      l2 := ∀ x, ax - y = 0 in
  a = tan θ →
  a ∈ (set.Ioo (real.sqrt 3 / 3) 1 ∪ set.Ioo 1 (real.sqrt 3)) :=
sorry

end range_of_a_l151_151018


namespace average_class_weight_l151_151206

theorem average_class_weight :
  let students_A := 50
  let weight_A := 60
  let students_B := 60
  let weight_B := 80
  let students_C := 70
  let weight_C := 75
  let students_D := 80
  let weight_D := 85
  let total_students := students_A + students_B + students_C + students_D
  let total_weight := students_A * weight_A + students_B * weight_B + students_C * weight_C + students_D * weight_D
  (total_weight / total_students : ℝ) = 76.35 :=
by
  sorry

end average_class_weight_l151_151206


namespace average_of_a_and_b_l151_151770

theorem average_of_a_and_b (a b c : ℝ) 
  (h₁ : (b + c) / 2 = 90)
  (h₂ : c - a = 90) :
  (a + b) / 2 = 45 :=
sorry

end average_of_a_and_b_l151_151770


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151566

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151566


namespace ex_l151_151217

noncomputable def equilateral_triangle_area (A B C S : Point) (r : ℝ) :=
  (equilateral A B C ∧ incircle_center A B C S ∧ dist B S = r) →
  (r = 10 * (Real.sqrt 3) / 3) →
  circle_area (10 * (Real.sqrt 3) / 3) = (100 * Real.pi) / 3

-- definition for the circle area given radius
def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Proof term omitted here
theorem ex.equilateral_triangle_area (A B C S : Point) (r : ℝ) :
  (equilateral A B C ∧ incircle_center A B C S ∧ dist B S = r) →
  (r = 10 * (Real.sqrt 3) / 3) →
  circle_area (10 * (Real.sqrt 3) / 3) = (100 * Real.pi) / 3 :=
  by sorry

end ex_l151_151217


namespace probability_sum_greater_than_six_l151_151817

theorem probability_sum_greater_than_six : 
  let dice_faces := {1, 2, 3, 4, 5, 6}
  let total_possibilities := 6 * 6   -- Total number of outcomes when two dice are tossed
  let favorable_combinations := 6 + 5 + 4 + 3 + 2 + 1
  let probability_sum_six_or_less := favorable_combinations.to_rat / total_possibilities.to_rat
  let probability_sum_greater_than_six := 1 - probability_sum_six_or_less
  probability_sum_greater_than_six = 7 / 12 := 
by {
  sorry
}

end probability_sum_greater_than_six_l151_151817


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151518

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151518


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151574

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151574


namespace sum_digits_2_2005_times_5_2007_times_3_l151_151257

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem sum_digits_2_2005_times_5_2007_times_3 : 
  sum_of_digits (2^2005 * 5^2007 * 3) = 12 := 
by 
  sorry

end sum_digits_2_2005_times_5_2007_times_3_l151_151257


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151581

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151581


namespace magnitude_power_eight_of_z_l151_151595

open Complex

noncomputable def z : ℂ := (2/3 : ℚ) + (5/6 : ℚ) * Complex.I

theorem magnitude_power_eight_of_z : |z^8| = 2825761 / 1679616 := by
  sorry

end magnitude_power_eight_of_z_l151_151595


namespace calculate_volume_of_pyramid_l151_151656

open EuclideanGeometry Real

-- Definitions
def point (α : Type*) := α × α × α
def prism (A B C A₁ B₁ C₁ : point ℝ) := true -- A regular triangular prism A B C - A_{1} B_{1} C_{1}

def center (P : point ℝ) (A₁ B₁ C₁ : point ℝ) := true -- P is the center of the base A_{1} B_{1} C_{1}

def perpendicular (plane : point ℝ → point ℝ → point ℝ) (A P : point ℝ) := true -- plane B C D is perpendicular to A P

def intersects_edge (D : point ℝ) (A A₁ : point ℝ) := true -- and intersects edge A A_{1} at D

def AA₁_equals (A A₁ : point ℝ) := (2 : ℝ) -- Given: A A_{1} = 2

def AB_equals_half (AB : ℝ) := (1 : ℝ) -- Given: 2AB = 2

theorem calculate_volume_of_pyramid :
  ∀ (P A A₁ B₁ C₁ B C D : point ℝ),
    prism A B C A₁ B₁ C₁ →
    center P A₁ B₁ C₁ →
    perpendicular (λ B C D, (B, C, D)) A P →
    intersects_edge D A A₁ →
    AA₁_equals A A₁ →
    AB_equals_half (dist A B / 2) →
    volume_of_pyramid P A D B C = (sqrt 3 / 48) := 
sorry

end calculate_volume_of_pyramid_l151_151656


namespace calculate_cost_per_square_meter_l151_151859

-- Define the dimensions of the lawn
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 60

-- Define the width of the roads
def road_width : ℝ := 10

-- Define the areas of the two roads
def area_road_parallel_to_length : ℝ := road_width * lawn_breadth
def area_road_parallel_to_breadth : ℝ := road_width * lawn_length

-- Define the overlap area
def overlap_area : ℝ := road_width * road_width

-- Calculate the total area of the roads taking into account the overlap
def total_area_of_roads : ℝ := (area_road_parallel_to_length + area_road_parallel_to_breadth) - overlap_area

-- Define the total cost to travel the roads
def total_cost : ℝ := 2600

-- Define the cost per square meter
def cost_per_square_meter : ℝ := total_cost / total_area_of_roads

-- Prove the cost per square meter
theorem calculate_cost_per_square_meter : cost_per_square_meter = 2 := by
  -- skip the proof
  sorry

end calculate_cost_per_square_meter_l151_151859


namespace num_persons_working_l151_151758

-- Defining the conditions of the problem
def work_rate (days : ℝ) : ℝ := 1 / days

def first_person_work_rate : ℝ := work_rate 24
def second_person_work_rate : ℝ := work_rate 12
def combined_work_rate : ℝ := first_person_work_rate + second_person_work_rate
def some_persons_work_rate : ℝ := work_rate 8

-- The theorem we want to prove
theorem num_persons_working : 
  combined_work_rate = some_persons_work_rate → 2 = 2 :=
by
  sorry

end num_persons_working_l151_151758


namespace customerPaidPercentGreater_l151_151298

-- Definitions for the conditions
def costOfManufacture (C : ℝ) : ℝ := C
def designerPrice (C : ℝ) : ℝ := C * 1.40
def retailerTaxedPrice (C : ℝ) : ℝ := (C * 1.40) * 1.05
def customerInitialPrice (C : ℝ) : ℝ := ((C * 1.40) * 1.05) * 1.10
def customerFinalPrice (C : ℝ) : ℝ := (((C * 1.40) * 1.05) * 1.10) * 0.90

-- The theorem statement
theorem customerPaidPercentGreater (C : ℝ) (hC : 0 < C) : 
    (customerFinalPrice C - costOfManufacture C) / costOfManufacture C * 100 = 45.53 := by 
  sorry

end customerPaidPercentGreater_l151_151298


namespace identity_map_a_plus_b_l151_151726

theorem identity_map_a_plus_b (a b : ℝ) (h : ∀ x ∈ ({-1, b / a, 1} : Set ℝ), x ∈ ({a, b, b - a} : Set ℝ)) : a + b = -1 ∨ a + b = 1 :=
by
  sorry

end identity_map_a_plus_b_l151_151726


namespace coeff_x4_in_expansion_l151_151772

theorem coeff_x4_in_expansion : 
  coeff_of_x4_in_expansion (x - (1 / (3 * x)))^8 = -56 := 
sorry

end coeff_x4_in_expansion_l151_151772


namespace sec_150_eq_neg_2_sqrt3_over_3_l151_151440

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l151_151440


namespace smoking_confidence_not_probability_l151_151352

def smoking_related_to_lung_disease := true
def confidence_level : ℝ := 0.99
def smoker_has_lung_disease (smokes : Bool) : Prop := smokes → false

theorem smoking_confidence_not_probability :
  (confidence_level = 0.99) ∧ smoking_related_to_lung_disease →
  ¬ (∀ smokes, smokes → smoker_has_lung_disease smokes) :=
by
  intro h
  sorry

end smoking_confidence_not_probability_l151_151352


namespace opposite_of_neg_two_is_two_l151_151176

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l151_151176


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151423

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151423


namespace set_intersection_equivalence_l151_151675

def setA (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def setB (x : ℝ) : Prop := 2^(x - 1) >= 1
def intersection (x : ℝ) : Prop := 1 <= x ∧ x < 3

theorem set_intersection_equivalence : 
  ∀ x, (setA x ∧ setB x) ↔ intersection x := by
  sorry

end set_intersection_equivalence_l151_151675


namespace find_all_functions_l151_151922

theorem find_all_functions (f : ℕ → ℕ) 
  (h1 : ∀ m : ℕ, (f^[2000] m) = f m) 
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n / f (Nat.gcd m n)) 
  (h3 : ∀ m : ℕ, f m = 1 ↔ m = 1) : 
  ∃ G : (ℕ → ℕ), ∃ H : ℕ → ℕ, ∃ I : ℕ → ℕ, ∀ m : ℕ, 
    let prime_factors := (λ m, list, if nat.exists_prime_and_dvd m then list.map (prime_to_power m) (nat.factorization m) else []) in 
    let f_of_prime_factors := λ pf, list.map (λ (p, exp), 
            (G p)^ (H exp * I p)) pf in
    f m = f_of_prime_factors (prime_factors m) in sorry

end find_all_functions_l151_151922


namespace exists_non_integer_solution_l151_151091

noncomputable def q (x y : ℝ) (b1 b2 : ℝ) : ℝ := 
  b1 * x + b2 * y - b1 * x^3 - b2 * y^3

theorem exists_non_integer_solution (b1 b2 : ℝ) (r s : ℝ) :
  q 0 0 b1 b2 = 0 ∧ q 1 0 b1 b2 = 0 ∧ q (-1) 0 b1 b2 = 0 ∧ 
  q 0 1 b1 b2 = 0 ∧ q 0 (-1) b1 b2 = 0 ∧ q 1 1 b1 b2 = 0 ∧ 
  q (-2) 1 b1 b2 = 0 ∧ q 3 (-1) b1 b2 = 0 → 
    b1 = 0 ∧ b2 = 0 → 
      q r s b1 b2 = 0 ∧ r ≠ floor r ∧ s ≠ floor s :=
sorry

end exists_non_integer_solution_l151_151091


namespace age_ratio_proof_l151_151302

-- Define the ages
def sonAge := 22
def manAge := sonAge + 24

-- Define the ratio computation statement
def ageRatioInTwoYears : ℚ := 
  let sonAgeInTwoYears := sonAge + 2
  let manAgeInTwoYears := manAge + 2
  manAgeInTwoYears / sonAgeInTwoYears

-- The theorem to prove
theorem age_ratio_proof : ageRatioInTwoYears = 2 :=
by
  sorry

end age_ratio_proof_l151_151302


namespace volleyballs_count_l151_151136

theorem volleyballs_count 
  (total_balls soccer_balls : ℕ)
  (basketballs tennis_balls baseballs volleyballs : ℕ) 
  (h_total : total_balls = 145) 
  (h_soccer : soccer_balls = 20) 
  (h_basketballs : basketballs = soccer_balls + 5)
  (h_tennis : tennis_balls = 2 * soccer_balls) 
  (h_baseballs : baseballs = soccer_balls + 10) 
  (h_specific_total : soccer_balls + basketballs + tennis_balls + baseballs = 115): 
  volleyballs = 30 := 
by 
  have h_specific_balls : soccer_balls + basketballs + tennis_balls + baseballs = 115 :=
    h_specific_total
  have total_basketballs : basketballs = 25 :=
    by rw [h_basketballs, h_soccer]; refl
  have total_tennis_balls : tennis_balls = 40 :=
    by rw [h_tennis, h_soccer]; refl
  have total_baseballs : baseballs = 30 :=
    by rw [h_baseballs, h_soccer]; refl
  have total_specific_balls : 20 + 25 + 40 + 30 = 115 :=
    by norm_num
  have volleyballs_to_find : volleyballs = 145 - 115 :=
    by rw [h_total]; exact rfl
  sorry

end volleyballs_count_l151_151136


namespace angle_PQR_is_correct_l151_151101

noncomputable def P : EuclideanSpace ℝ (Fin 3) := ![2, 3, -1]
noncomputable def Q : EuclideanSpace ℝ (Fin 3) := ![5, 3, -4]
noncomputable def R : EuclideanSpace ℝ (Fin 3) := ![3, 2, -5]

noncomputable def dist (A B : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  Real.sqrt (((B - A) ^ 2).sum)

noncomputable def PQ := dist P Q
noncomputable def PR := dist P R
noncomputable def QR := dist Q R

noncomputable def cosAnglePQR : ℝ :=
  ((PQ ^ 2 + PR ^ 2 - QR ^ 2) / (2 * PQ * PR))

noncomputable def anglePQR : ℝ :=
  Real.arccos cosAnglePQR

theorem angle_PQR_is_correct :
  anglePQR = Real.arccos (5 / 6) :=
by
  sorry

end angle_PQR_is_correct_l151_151101


namespace find_g_of_given_conditions_l151_151947

theorem find_g_of_given_conditions :
  (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 3 * x - 1) (h₂ : ∀ x, f (g x) = 2 * x + 3) :
  g = λ x, (2 / 3) * x + (4 / 3) :=
by
  sorry

end find_g_of_given_conditions_l151_151947


namespace greatest_lcm_less_than_120_l151_151246

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b
noncomputable def multiples (x limit : ℕ) : List ℕ := List.range (limit / x) |>.map (λ n => x * (n + 1))

theorem greatest_lcm_less_than_120 :  GCM_of_10_and_15_lt_120 = 90
  where
    GCM_of_10_and_15_lt_120 : ℕ := match (multiples (lcm 10 15) 120) with
                                     | [] => 0
                                     | xs => xs.maximum'.getD 0 :=
  by
  apply sorry

end greatest_lcm_less_than_120_l151_151246


namespace geometric_sequence_angles_count_l151_151910

theorem geometric_sequence_angles_count : 
  ∃ n ∈ ℕ, n = 4 ∧ ∀ θ ∈ (set.Icc 0 (2 * Real.pi)),
    (θ ≠ (Real.pi / 2) ∧ θ ≠ Real.pi ∧ θ ≠ (3 * Real.pi / 2)) →
    (∃ (a b c : ℝ),
      (a = Real.sin θ ∨ a = Real.cos θ ∨ a = Real.cot θ) ∧
      (b = Real.sin θ ∨ b = Real.cos θ ∨ b = Real.cot θ) ∧
      (c = Real.sin θ ∨ c = Real.cos θ ∨ c = Real.cot θ) ∧
      (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
      (a * c = b * b)) :=
by
  sorry

end geometric_sequence_angles_count_l151_151910


namespace modulus_of_complex_number_l151_151628

theorem modulus_of_complex_number
  (x y : ℝ) 
  (h : (1 + complex.I) * x + y * complex.I = (1 + 3 * complex.I) * complex.I) :
  complex.abs (x + y * complex.I) = 5 :=
sorry -- Proof is skipped

end modulus_of_complex_number_l151_151628


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151523

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151523


namespace AKIL_is_rhombus_proof_l151_151721

open Triangle

variables {A B C I S T K L : Point}

def AKIL_is_rhombus (h : acute_triangle A B C) 
  (hI : incenter I A B C) 
  (hS : ray_intersect BI I S (circumcircle A B C))
  (hT : ray_intersect CI I T (circumcircle A B C))
  (hK : seg_intersect ST AB K)
  (hL : seg_intersect ST AC L) : Prop :=
  rhombus A K I L

theorem AKIL_is_rhombus_proof (h : acute_triangle A B C) 
  (hI : incenter I A B C) 
  (hS : ray_intersect BI I S (circumcircle A B C))
  (hT : ray_intersect CI I T (circumcircle A B C))
  (hK : seg_intersect ST AB K)
  (hL : seg_intersect ST AC L) : AKIL_is_rhombus h hI hS hT hK hL :=
sorry

end AKIL_is_rhombus_proof_l151_151721


namespace compare_abc_l151_151631

noncomputable def a : ℝ := 0.9 ^ 1.5
noncomputable def b : ℝ := Real.logBase 2 0.9
noncomputable def c : ℝ := Real.logBase 0.3 0.2

theorem compare_abc :
  c > a ∧ a > b :=
by
  sorry

end compare_abc_l151_151631


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151528

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151528


namespace find_common_difference_of_arithmetic_sequence_l151_151964

noncomputable def arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (a_3_value : ℝ) (S_4_value : ℝ) : Prop :=
  ( ∀ n, a n = a 0 + (n - 1) * (a 1 - a 0) ) ∧
  ( ∀ n, S n = (n * (2 * (a 0) + (n - 1) * (a 1 - a 0))) / 2 ) ∧
  a 3 = a_3_value ∧
  S 4 = S_4_value ∧
  (a 1 - a 0) = 2

theorem find_common_difference_of_arithmetic_sequence :
  arithmetic_sequence_common_difference (λ n, 0) (λ n, 0) 10 36 :=
by
  unfold arithmetic_sequence_common_difference
  -- Definitions from conditions
  have h1 : ∀ n, (0 : ℕ → ℝ) n = 0 + (n - 1) * (0 - 0) :=
    λ _, by simp
  have h2 : ∀ n, (λ n, (n * (2 * (0 : ℝ) + (n - 1) * (0 - 0))) / 2) n = (λ n, 0) n :=
    λ _, by simp
  -- Specific values of a and S
  have h3 : (λ n, 0) 3 = 10 := by simp
  have h4 : (λ n, 0) 4 = 36 := by simp
  -- Common difference proof
  have h5 : (0 - 0) = 2 := by simp
  exact ⟨h1, h2, h3, h4, h5⟩

end find_common_difference_of_arithmetic_sequence_l151_151964


namespace series_sum_eq_one_half_l151_151921

noncomputable def series_sum : ℝ :=
  ∑' n, (n ^ 3 + 2 * n ^ 2 - n - 1) / (n + 3)! 

theorem series_sum_eq_one_half : series_sum = 1 / 2 := by
  sorry

end series_sum_eq_one_half_l151_151921


namespace evaluate_expression_at_x_eq_3_l151_151918

theorem evaluate_expression_at_x_eq_3 :
  (3 ^ 3) ^ (3 ^ 3) = 7625597484987 := by
  sorry

end evaluate_expression_at_x_eq_3_l151_151918


namespace place_the_numbers_l151_151052

def circles_are_properly_labeled : Prop :=
  ∃ (A B C D : ℕ), 
    A ∈ {6, 7, 8, 9} ∧ B ∈ {6, 7, 8, 9} ∧ C ∈ {6, 7, 8, 9} ∧ D ∈ {6, 7, 8, 9} ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
    (A + C + 3 + 4 = B + 5 + 2 + 3 ∧ A + C + 3 + 4 = 5 + D + 2 + 4) ∧
    (A = 6 ∧ B = 8 ∧ C = 7 ∧ D = 9)

theorem place_the_numbers : circles_are_properly_labeled :=
by
  unfold circles_are_properly_labeled
  sorry

end place_the_numbers_l151_151052


namespace kaleb_initial_books_l151_151076

def initial_books (sold_books bought_books final_books : ℕ) : ℕ := 
  sold_books - bought_books + final_books

theorem kaleb_initial_books :
  initial_books 17 (-7) 24 = 34 := 
by 
  -- use the definition of initial_books
  sorry

end kaleb_initial_books_l151_151076


namespace sum_of_squares_geometric_sequence_l151_151199

theorem sum_of_squares_geometric_sequence (n : ℕ) (a : ℕ → ℕ) 
  (h : ∀ n, (finset.range n).sum (λ i, a i) = (2^n - 1)) :
  (finset.range n).sum (λ i, (a i)^2) = (4^n - 1) / 3 :=
sorry

end sum_of_squares_geometric_sequence_l151_151199


namespace opposite_of_neg_two_is_two_l151_151175

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l151_151175


namespace not_diamond_0_eq_2x_l151_151349

def diamond (x y : ℝ) : ℝ := x + y - |x - y|

theorem not_diamond_0_eq_2x : ¬ (∀ x : ℝ, diamond x 0 = 2 * x) :=
begin
  intro h,
  have h_neg : diamond (-1) 0 = 2 * (-1), by exact h (-1),
  simp [diamond, abs_neg] at h_neg,
  linarith,
end

end not_diamond_0_eq_2x_l151_151349


namespace circles_tangent_internally_l151_151016

theorem circles_tangent_internally (a : ℝ) :
  (∃ (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ),
    C1 = (a, 0) ∧ r1 = 6 ∧
    C2 = (0, 2) ∧ r2 = 2 ∧
    ∀ (x y : ℝ), 
      ((x - a)^2 + y^2 = 36) ∧ (x^2 + (y - 2)^2 = 4) ∧
      (√(a^2 + 4) = r1 - r2)) → 
  (a = 2 * √3 ∨ a = -2 * √3) :=
begin
  sorry
end

end circles_tangent_internally_l151_151016


namespace number_of_digits_right_of_decimal_l151_151681

theorem number_of_digits_right_of_decimal :
  (∀ (n : ℕ), (\frac (5 ^ 8) (10 ^ 6 * 3125) = n / 10^6) → 6) :=
sorry

end number_of_digits_right_of_decimal_l151_151681


namespace angle_CPD_tangent_semicircles_l151_151060

theorem angle_CPD_tangent_semicircles :
  ∀ (P C D S R T : Point) (O1 O2 : Point),
    (tangent_to_semicircle P C S O1) →
    (tangent_to_semicircle P D T O2) →
    (S, R, T is_collinear) →
    (arc_angle O1 S A = 70) →
    (arc_angle O2 T B = 45) →
    (angle CPD D P = 115) := by
sorry

end angle_CPD_tangent_semicircles_l151_151060


namespace zoe_can_ensure_two_turns_l151_151263

theorem zoe_can_ensure_two_turns (n : ℕ) (h : n = 14) : ∃ k : ℕ, k > 0 ∧ (k <= n) :=
begin
  sorry
end

end zoe_can_ensure_two_turns_l151_151263


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151495

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151495


namespace number_of_correct_assertions_l151_151151

-- Definitions of conditions
def perpendicular (a b : Type) : Prop := 
  -- Placeholder definition
  sorry

def skew (a b : Type) : Prop := 
  -- Placeholder definition
  sorry

def intersect (a b : Type) : Prop := 
  -- Placeholder definition
  sorry

def coplanar (a b : Type) : Prop := 
  -- Placeholder definition
  sorry

def parallel (a b : Type) : Prop := 
  -- Placeholder definition
  sorry

-- Assertions
def assertion_1 (a b c : Type) : Prop :=
  perpendicular a b ∧ perpendicular b c → perpendicular a c

def assertion_2 (a b c : Type) : Prop :=
  skew a b ∧ skew b c → skew a c

def assertion_3 (a b c : Type) : Prop :=
  intersect a b ∧ intersect b c → intersect a c

def assertion_4 (a b c : Type) : Prop :=
  coplanar a b ∧ coplanar b c → coplanar a c

def assertion_5 (a b c : Type) : Prop :=
  parallel a b ∧ parallel b c → parallel a c

-- The theorem to be proved
theorem number_of_correct_assertions (a b c : Type) :
  (assertion_1 a b c = false) ∧ 
  (assertion_2 a b c = false) ∧ 
  (assertion_3 a b c = false) ∧ 
  (assertion_4 a b c = false) ∧ 
  (assertion_5 a b c = true) → 
  1 := 
sorry

end number_of_correct_assertions_l151_151151


namespace triangle_PB_PC_length_l151_151216

noncomputable def PA := 12 : ℝ
noncomputable def PB := 8 : ℝ
noncomputable def PC := Real.sqrt 464

theorem triangle_PB_PC_length :
  ∀ (A B C P : ℝ) (h_right_angle: ∠ BPC = Real.pi / 2) (h_triangle : true),
  PA = 12 → PB = 8 → P ≠ B → P ≠ C → 
  PC = Real.sqrt 464 :=
by
  sorry

end triangle_PB_PC_length_l151_151216


namespace find_x_if_vectors_parallel_l151_151985

/--
Given the vectors a = (2 * x + 1, 3) and b = (2 - x, 1), if a is parallel to b, 
then x must be equal to 1.
-/
theorem find_x_if_vectors_parallel (x : ℝ) :
  let a := (2 * x + 1, 3)
  let b := (2 - x, 1)
  (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) → x = 1 :=
by
  sorry

end find_x_if_vectors_parallel_l151_151985


namespace necessary_but_not_sufficient_l151_151647

variables {x y : ℝ}

def p := (x - 1) * (y - 2) = 0
def q := (x - 1) ^ 2 + (y - 2) ^ 2 = 0

theorem necessary_but_not_sufficient : (p → q) ∧ ¬(q → p) :=
by {
  unfold p q,
  split,
  {
    intros h,
    cases h,
    sorry
  },
  {
    intro hf,
    sorry
  }
}

end necessary_but_not_sufficient_l151_151647


namespace correct_quotient_l151_151840

theorem correct_quotient (D : ℕ) (Q : ℕ) (h1 : D = 21 * Q) (h2 : D = 12 * 49) : Q = 28 := 
by
  sorry

end correct_quotient_l151_151840


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151537

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151537


namespace exists_unique_i_l151_151731

theorem exists_unique_i (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) 
  (a : ℤ) (ha1 : 2 ≤ a) (ha2 : a ≤ p - 2) : 
  ∃! (i : ℤ), 2 ≤ i ∧ i ≤ p - 2 ∧ (i * a) % p = 1 ∧ Nat.gcd (i.natAbs) (a.natAbs) = 1 :=
sorry

end exists_unique_i_l151_151731


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151363

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151363


namespace triangle_concurrency_l151_151068

open EuclideanGeometry

theorem triangle_concurrency (A B C K L M R P Q : Point) 
  (hABC: is_triangle A B C)
  (hK: is_circumcircle_point A B C K)
  (hL: is_circumcircle_point A B C L)
  (hM: is_circumcircle_point A B C M)
  (hR: is_on_line_segment R A B)
  (hRP_parallel_AK: parallel R P A K)
  (hBP_perp_BL: perp B P B L)
  (hRQ_parallel_BL: parallel R Q B L)
  (hAQ_perp_AK: perp A Q A K) :
  concurrent (line_through K P) (line_through L Q) (line_through M R) :=
sorry

end triangle_concurrency_l151_151068


namespace opposite_of_neg_two_l151_151184

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l151_151184


namespace smallest_area_triangle_OAB_l151_151828

noncomputable def smallest_triangle_area : ℝ :=
  let f : ℝ → ℝ := λ x, 2 * |x| - x + 1
  let area (a b : ℝ) : ℝ := (|a - b|) / 2
  let min_area (a b : ℝ) : ℝ := (2 / ((1 - ((a - b)/a)) * (((a - b)/a) + 3)))
  0.5 -- smallest area is 0.5

theorem smallest_area_triangle_OAB :
  ∀(a b : ℝ), 
    (f a = a + 1) → 
    (f b = -3 * b + 1) → 
    (a ≠ b) →
    ((min_area a b) = 0.5) :=
by
  intro a b
  intro ha hb hab
  sorry

end smallest_area_triangle_OAB_l151_151828


namespace exponent_of_second_term_l151_151798

theorem exponent_of_second_term (e : ℝ) (exponent : ℝ) 
  (h1 : e = 35)
  (h2 : (1/5)^e * (1/4)^exponent = 1/(2*(10)^35)) :
  exponent = 17.5 :=
by
  sorry

end exponent_of_second_term_l151_151798


namespace simplify_and_evaluate_l151_151755

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 + 1 / a) / ((a^2 - 1) / a) = (Real.sqrt 2 / 2) :=
by
  sorry

end simplify_and_evaluate_l151_151755


namespace sec_150_l151_151480

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151480


namespace find_y_from_expression_l151_151338

theorem find_y_from_expression :
  ∀ y : ℕ, 2^10 + 2^10 + 2^10 + 2^10 = 4^y → y = 6 :=
by
  sorry

end find_y_from_expression_l151_151338


namespace probability_of_sum_distances_less_than_2_is_two_thirds_l151_151625

noncomputable def probability_distances_less_than_2 : ℝ :=
  let A := 0
  let B := 1
  let C := 2
  let D := 3
  let AD := Icc A D
  let MN := Icc (A + 0.5) (D - 0.5)
  let probability := (MN.length / AD.length : ℝ)
  probability

theorem probability_of_sum_distances_less_than_2_is_two_thirds :
  probability_distances_less_than_2 = (2 / 3 : ℝ) :=
by
  -- proof omitted
  sorry

end probability_of_sum_distances_less_than_2_is_two_thirds_l151_151625


namespace solve_reflection_problem_l151_151858

noncomputable def reflection_problem : Prop :=
  ∃ (A B C D E : Type), 
    (∀ (poly : Type) (line_of_symmetry : poly → poly) (reflection : poly → poly),
      (line_of_symmetry (A) = A) ∧ 
      (reflection (A) ≠ A) ∧ 
      (line_of_symmetry (B) = B) ∧ 
      (reflection (B) = B) ∧ 
      (line_of_symmetry (C) = C) ∧ 
      (reflection (C) ≠ C) ∧ 
      (line_of_symmetry (D) = D) ∧
      (reflection (D) ≠ D) ∧ 
      (line_of_symmetry (E) = E) ∧ 
      (reflection (E) ≠ E) → 
      reflection poly = C)

theorem solve_reflection_problem : reflection_problem :=
  sorry

end solve_reflection_problem_l151_151858


namespace Emily_subtract_59_l151_151811

theorem Emily_subtract_59 : (30 - 1) ^ 2 = 30 ^ 2 - 59 := by
  sorry

end Emily_subtract_59_l151_151811


namespace sec_150_eq_l151_151406

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151406


namespace perimeter_of_resulting_figure_l151_151077

-- Define the perimeters of the squares
def perimeter_small_square : ℕ := 40
def perimeter_large_square : ℕ := 100

-- Define the side lengths of the squares
def side_length_small_square := perimeter_small_square / 4
def side_length_large_square := perimeter_large_square / 4

-- Define the total perimeter of the uncombined squares
def total_perimeter_uncombined := perimeter_small_square + perimeter_large_square

-- Define the shared side length
def shared_side_length := side_length_small_square

-- Define the perimeter after considering the shared side
def resulting_perimeter := total_perimeter_uncombined - 2 * shared_side_length

-- Prove that the resulting perimeter is 120 cm
theorem perimeter_of_resulting_figure : resulting_perimeter = 120 := by
  sorry

end perimeter_of_resulting_figure_l151_151077


namespace inequality_proof_l151_151099

theorem inequality_proof 
  {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ} (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) (h₆ : x₆ > 0) :
  (x₂ / x₁)^5 + (x₄ / x₂)^5 + (x₆ / x₃)^5 + (x₁ / x₄)^5 + (x₃ / x₅)^5 + (x₅ / x₆)^5 ≥ 
  (x₁ / x₂) + (x₂ / x₄) + (x₃ / x₆) + (x₄ / x₁) + (x₅ / x₃) + (x₆ / x₅) := 
  sorry

end inequality_proof_l151_151099


namespace transformed_graph_correct_l151_151785

-- Defining the original function
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

-- Defining the resulting function after transformations
def g (x : ℝ) : ℝ := -2 * x^2 - 4 * x

-- Theorem statement
theorem transformed_graph_correct :
  ∀ x : ℝ, 
  let rotated_f := -(f x) in
  let translated_f := rotated_f + 3 in
  translated_f = g x := 
by
  sorry

end transformed_graph_correct_l151_151785


namespace fraction_transformation_half_l151_151688

theorem fraction_transformation_half (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  ((2 * a + 2 * b) / (4 * a^2 + 4 * b^2)) = (1 / 2) * ((a + b) / (a^2 + b^2)) :=
by sorry

end fraction_transformation_half_l151_151688


namespace first_train_length_l151_151819

noncomputable def length_of_first_train (speed1_kmph speed2_kmph length2_m time_s : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * (5 / 18)
  let speed2_mps := speed2_kmph * (5 / 18)
  let relative_speed_mps := speed1_mps + speed2_mps
  (time_s * relative_speed_mps) - length2_m

theorem first_train_length :
  length_of_first_train 60 40 120 6.119510439164867 ≈ 49.99 :=
by
  sorry

end first_train_length_l151_151819


namespace coefficient_x4_in_expansion_l151_151232

theorem coefficient_x4_in_expansion : 
  (∃ (c : ℤ), c = (choose 8 4) * 3^4 * 2^4 ∧ c = 90720) := 
by
  use (choose 8 4) * 3^4 * 2^4
  split
  sorry
  sorry

end coefficient_x4_in_expansion_l151_151232


namespace sec_150_eq_neg_two_sqrt_three_over_three_l151_151445

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l151_151445


namespace tape_mounting_cost_correct_l151_151773

-- Define the given conditions as Lean definitions
def os_overhead_cost : ℝ := 1.07
def cost_per_millisecond : ℝ := 0.023
def total_cost : ℝ := 40.92
def runtime_seconds : ℝ := 1.5

-- Define the required target cost for mounting a data tape
def cost_of_data_tape : ℝ := 5.35

-- Prove that the cost of mounting a data tape is correct given the conditions
theorem tape_mounting_cost_correct :
  let computer_time_cost := cost_per_millisecond * (runtime_seconds * 1000)
  let total_cost_computed := os_overhead_cost + computer_time_cost
  cost_of_data_tape = total_cost - total_cost_computed := by
{
  sorry
}

end tape_mounting_cost_correct_l151_151773


namespace smallest_value_of_w3_plus_z3_l151_151633

-- Define complex numbers w and z
variables {w z : ℂ}

-- The given conditions
def conditions : Prop := 
  |w + z| = 2 ∧ |w^2 + z^2| = 10

-- The statement to be proved
theorem smallest_value_of_w3_plus_z3 (hc : conditions) : |w^3 + z^3| = 26 := 
  sorry

end smallest_value_of_w3_plus_z3_l151_151633


namespace smallest_N_constant_l151_151606

-- Define the property to be proven
theorem smallest_N_constant (a b c : ℝ) 
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) (h₄ : k = 0):
  (a^2 + b^2 + k) / c^2 > 1 / 2 :=
by
  sorry

end smallest_N_constant_l151_151606


namespace find_avg_temp_monday_to_thursday_l151_151158

def avg_temp_monday_to_thursday (T_M : ℝ) (T_Tu : ℝ) (T_W : ℝ) (T_Th : ℝ) (T_F : ℝ) :=
  (T_M + T_Tu + T_W + T_Th) / 4

def avg_temp_tuesday_to_friday (T_Tu : ℝ) (T_W : ℝ) (T_Th : ℝ) (T_F : ℝ) :=
  (T_Tu + T_W + T_Th + T_F) / 4

theorem find_avg_temp_monday_to_thursday :
  ∀ T : Type, ∀ (avg_T : ℝ) (T_Tu T_W T_Th T_F : ℝ),
    avg_temp_tuesday_to_friday T_Tu T_W T_Th T_F = 46 →
    T_F = 35 → 
    (T : ℝ),
    avg_temp_monday_to_thursday T 43 T_Tu T_W T_Th = 48 :=
by {
  intros,
  sorry
}

end find_avg_temp_monday_to_thursday_l151_151158


namespace additive_inverse_commutativity_l151_151696

section
  variable {R : Type} [Ring R] (h : ∀ x : R, x ^ 2 = x)

  theorem additive_inverse (x : R) : -x = x := by
    sorry

  theorem commutativity (x y : R) : x * y = y * x := by
    sorry
end

end additive_inverse_commutativity_l151_151696


namespace sec_150_l151_151487

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151487


namespace greatest_lcm_less_than_120_l151_151247

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b
noncomputable def multiples (x limit : ℕ) : List ℕ := List.range (limit / x) |>.map (λ n => x * (n + 1))

theorem greatest_lcm_less_than_120 :  GCM_of_10_and_15_lt_120 = 90
  where
    GCM_of_10_and_15_lt_120 : ℕ := match (multiples (lcm 10 15) 120) with
                                     | [] => 0
                                     | xs => xs.maximum'.getD 0 :=
  by
  apply sorry

end greatest_lcm_less_than_120_l151_151247


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151364

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151364


namespace chuck_play_area_l151_151899

-- Define the constants and conditions
def shed_length : ℝ := 3
def shed_width : ℝ := 4
def leash_length : ℝ := 4

-- Define the main area calculation
def main_area : ℝ := (3/4) * Real.pi * (leash_length ^ 2)

-- Define the additional area calculation
def additional_area : ℝ := (1/4) * Real.pi * (1 ^ 2)

-- Define the total area
def total_area : ℝ := main_area + additional_area

-- The theorem we want to prove
theorem chuck_play_area : total_area = (49/4) * Real.pi :=
by
  sorry

end chuck_play_area_l151_151899


namespace exponent_equation_l151_151996

theorem exponent_equation (m n : ℝ) (h₁ : 3^m = 5) (h₂ : 3^n = 6) : 3^(m + n) = 30 :=
sorry

end exponent_equation_l151_151996


namespace non_nilpotent_matrices_sum_k_ne_zero_l151_151260

variable {R : Type*} [comm_ring R] [nontrivial R]
variable {n m : ℕ} (A : fin m → matrix (fin n) (fin n) R)

theorem non_nilpotent_matrices_sum_k_ne_zero (hn : 2 ≤ n) (hm : 2 ≤ m)
    (hA : ¬ ∀ i, is_nilpotent (A i)) :
    ∃ k > 0, ∑ i : fin m, (A i)^k ≠ 0 := 
sorry

end non_nilpotent_matrices_sum_k_ne_zero_l151_151260


namespace max_students_seating_l151_151326

theorem max_students_seating (numRows : ℕ) (initialSeats : ℕ) (increaseSeats : ℕ) 
  (h_numRows : numRows = 15) 
  (h_initialSeats : initialSeats = 18) 
  (h_increaseSeats : increaseSeats = 2) : 
  let n_i := λ i, initialSeats + increaseSeats * i,
      max_students_in_row := λ n, (n + 1) / 2,
      total_students := ∑ i in Finset.range numRows, max_students_in_row (n_i i)
  in total_students = 240 :=
by
  {
    -- specify conditions and constraints in Lean
    sorry
  }

end max_students_seating_l151_151326


namespace equation_of_ellipse_C_range_TF_PQ_l151_151671

-- Conditions 
def parabola : Set (ℝ × ℝ) := { p | p.2 ^ 2 = 4 * p.1 }
def ellipse : Set (ℝ × ℝ) (a b : ℝ) := { p | p.1^2 / a^2 + p.2^2 / b^2 = 1 }
def F : ℝ × ℝ := (1, 0)
def a := 2
def b := sqrt 3
def ellipse_C : Set (ℝ × ℝ) := { p | p.1 ^ 2 / 4 + p.2 ^ 2 / 3 = 1}
def B : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p ∈ (parabola ∩ ellipse_C)}
def d_BF : ℝ := (2 / 3)

-- Questions as Lean definitions
theorem equation_of_ellipse_C :
  ∃ a b, B = (ellipse_C ∩ parabola) ∧ F ∈ ellipse_C ∧ |B - F| = d_BF → ellipse_C = ellipse a b :=
sorry

theorem range_TF_PQ :
  ∃ (F : ℝ × ℝ), ∀ (m : ℝ), 
  let t := sqrt (m^2 + 1) in 
  (t > 1) → ∀ (T PQ : ℝ), T ∈ {p | p.1 = 4} ∧ PQ ∈ ellipse_C →
  (3/4) * t + (1/4) * (1/t) ≥ 1 :=
sorry

end equation_of_ellipse_C_range_TF_PQ_l151_151671


namespace greatest_divisor_same_remainder_l151_151929

theorem greatest_divisor_same_remainder (a b c : ℕ) (d1 d2 d3 : ℕ) (h1 : a = 41) (h2 : b = 71) (h3 : c = 113)
(hd1 : d1 = b - a) (hd2 : d2 = c - b) (hd3 : d3 = c - a) :
  Nat.gcd (Nat.gcd d1 d2) d3 = 6 :=
by
  -- some computation here which we are skipping
  sorry

end greatest_divisor_same_remainder_l151_151929


namespace teaching_arrangements_l151_151120

-- Define the conditions
structure Conditions :=
  (teach_A : ℕ)
  (teach_B : ℕ)
  (teach_C : ℕ)
  (teach_D : ℕ)
  (max_teach_AB : ∀ t, t = teach_A ∨ t = teach_B → t ≤ 2)
  (max_teach_CD : ∀ t, t = teach_C ∨ t = teach_D → t ≤ 1)
  (total_periods : ℕ)
  (teachers_per_period : ℕ)

-- Constants and assumptions
def problem_conditions : Conditions := {
  teach_A := 2,
  teach_B := 2,
  teach_C := 1,
  teach_D := 1,
  max_teach_AB := by sorry,
  max_teach_CD := by sorry,
  total_periods := 2,
  teachers_per_period := 2
}

-- Define the proof goal
theorem teaching_arrangements (c : Conditions) :
  c = problem_conditions → ∃ arrangements, arrangements = 19 :=
by
  sorry

end teaching_arrangements_l151_151120


namespace abs_neg_reciprocal_of_two_l151_151767

theorem abs_neg_reciprocal_of_two : 
  | - (1 / 2) | = | (1 / 2) | :=
by {
  sorry
}

end abs_neg_reciprocal_of_two_l151_151767


namespace P_on_euler_circle_l151_151093

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def perpendicular_foot (A B C : Point) : Point := sorry
noncomputable def perpendicular_bisector_intersection (D E : Point) : Point := sorry
noncomputable def perpendicular_from_point_to_line (A B : Point) (l : Line) : Point := sorry
noncomputable def euler_circle (A B C : Point) : Circle := sorry
noncomputable def is_on_circle (P : Point) (C : Circle) : Prop := sorry

variables {A B C P D E : Point}

-- Assumptions and conditions
axiom H1 : D = midpoint B C
axiom H2 : E = perpendicular_foot A B C
axiom H3 : P = perpendicular_bisector_intersection D E
axiom H4 : P = perpendicular_from_point_to_line D (angle_bisector A (line_through B C))
axiom H5 : ¬ (A = B)
axiom H6 : ¬ (A = C)

-- Prove that point P lies on the Euler circle of triangle ABC
theorem P_on_euler_circle (A B C : Point) (H1 : midpoint B C) (H2 : perpendicular_foot A B C) (H3 : perpendicular_bisector_intersection D E) (H4 : perpendicular_from_point_to_line D (angle_bisector A (line_through B C))) (H5 : ¬ (A = B)) (H6 : ¬ (A = C)) : 
is_on_circle P (euler_circle A B C) :=
sorry

end P_on_euler_circle_l151_151093


namespace proposal_spreading_problem_l151_151813

theorem proposal_spreading_problem (n : ℕ) : 1 + n + n^2 = 1641 := 
sorry

end proposal_spreading_problem_l151_151813


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151510

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151510


namespace equal_segments_of_isosceles_triangle_l151_151122

open Triangle

structure IsoscelesTriangle :=
  (A B C : Point)
  (AB_eq_AC : dist A B = dist A C)
  (X_mid_BC : midpoint X B C)

structure PointsOnSides :=
  (P Q : Point)
  (P_on_AB : OnSegment P A B)
  (Q_on_AC : OnSegment Q A C)

def AnglesAtMidpoint (P Q X B C : Point) : Prop :=
  (∠ P X B = ∠ Q X C)

theorem equal_segments_of_isosceles_triangle
  (T : IsoscelesTriangle) 
  (Pts : PointsOnSides)
  (H : AnglesAtMidpoint Pts.P Pts.Q T.X T.B T.C) :
  dist T.B Pts.Q = dist T.C Pts.P := 
sorry

end equal_segments_of_isosceles_triangle_l151_151122


namespace prob_C_at_fifth_drink_l151_151049

def transition_Matrix : Matrix (Fin 4) (Fin 4) ℚ :=
  ![![0, 1/3, 1/3, 0],
    ![1/2, 0, 1/3, 1/2],
    ![1/2, 1/3, 0, 1/2],
    ![0, 1/3, 1/3, 0]]

/-- Initial state -/
def initial_state : Fin 4 → ℚ
| 0 => 1  -- Starting at A
| _ => 0  -- All other pubs are zero

noncomputable def state_after_n_drinks (n : ℕ) : Fin 4 → ℚ :=
  (transition_Matrix ^ n).vecMul initial_state

theorem prob_C_at_fifth_drink : state_after_n_drinks 5 2 = 55 / 162 :=
by skip -- Proof to be completed

end prob_C_at_fifth_drink_l151_151049


namespace dissection_possible_l151_151624

theorem dissection_possible (k : ℝ) (h1 : k > 0) : 
  (∃ (P Q : set (ℝ × ℝ)), is_similar P Q ∧ ¬is_congruent P Q ∧ 
  is_dissection (1, k) (P ∪ Q) (P ∩ Q)) ↔ k ≠ 1 :=
begin
  sorry
end

end dissection_possible_l151_151624


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151365

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151365


namespace sec_150_l151_151479

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l151_151479


namespace quadratic_roots_sign_range_l151_151657

theorem quadratic_roots_sign_range (a : ℝ) (h1 : a ≠ 0)
  (h2 : (2^2 - 4 * a * 1) ≥ 0)
  (h3 : (1 / a) > 0) :
  0 < a ∧ a ≤ 1 :=
begin
  sorry
end

end quadratic_roots_sign_range_l151_151657


namespace polynomial_no_integer_roots_l151_151116

theorem polynomial_no_integer_roots
  (f : ℤ → ℤ)
  (a_0 a_1 a_2 ... a_n : ℤ)
  (n : ℕ)
  (h_f : ∀ x, f x = a_0 * x ^ n + a_1 * x ^ (n - 1) + a_2 * x ^ (n - 2) + ... + a_n)
  (exists_odd_alpha : ∃ α : ℤ, α % 2 = 1 ∧ f α % 2 = 1)
  (exists_even_beta : ∃ β : ℤ, β % 2 = 0 ∧ f β % 2 = 1) :
  ∀ x : ℤ, f x ≠ 0 := sorry

end polynomial_no_integer_roots_l151_151116


namespace sec_150_eq_l151_151459

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151459


namespace inequality_proof_l151_151142

variables (x y z : ℝ)

theorem inequality_proof (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  2 ≤ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ∧ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ≤ (1 + x) * (1 + y) * (1 + z) :=
by
  sorry

end inequality_proof_l151_151142


namespace sec_150_eq_neg_two_div_sqrt_three_l151_151514

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l151_151514


namespace sec_150_eq_l151_151463

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l151_151463


namespace sec_150_eq_neg_2_sqrt3_div_3_l151_151541

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l151_151541


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151360

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151360


namespace probability_two_congestion_days_is_correct_l151_151815

-- Define the condition for traffic congestion probability
def prob_traffic_congestion_per_day := 0.4

-- Define the mapping of integers representing traffic congestion
def is_traffic (x : ℕ) : Prop := x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4

-- Define the list of simulated data
def simulated_data := [[8, 0, 7], [0, 6, 6], [1, 2, 3], [9, 2, 3], [4, 7, 1], 
                       [5, 3, 2], [7, 1, 2], [2, 6, 9], [5, 0, 7], [7, 5, 2], 
                       [4, 4, 3], [2, 7, 7], [3, 0, 3], [9, 2, 7], [7, 5, 6], 
                       [3, 6, 8], [8, 4, 0], [4, 1, 3], [7, 3, 0], [0, 8, 6]]

-- Function to check if exactly two days have traffic congestion
def exactly_two_traffic_days (days : List ℕ) : Prop :=
  (countp is_traffic days) = 2

-- Calculate the probability
def prob_exactly_two_traffic_days (data : List (List ℕ)) : ℝ :=
  (countp exactly_two_traffic_days data).to_real / data.length.to_real

-- Statement to be proved
theorem probability_two_congestion_days_is_correct :
  prob_exactly_two_traffic_days simulated_data = 0.25 :=
by sorry

end probability_two_congestion_days_is_correct_l151_151815


namespace perimeter_of_region_l151_151155

theorem perimeter_of_region
  (area_of_region : ℕ)
  (h1 : area_of_region = 144)
  (num_squares : ℕ)
  (h2 : num_squares = 4)
  (sides_per_square : ℕ)
  (h3 : sides_per_square = 4)
  (side_length : ℕ)
  (h4 : side_length * side_length = area_of_region / num_squares)
  (total_sides : ℕ)
  (h5 : total_sides = 2 * (2 + 2))
  (perimeter : ℕ)
  (h6 : perimeter = total_sides * side_length)
  : perimeter = 48 :=
begin
  sorry
end

end perimeter_of_region_l151_151155


namespace sarah_scores_arithmetic_mean_l151_151141

theorem sarah_scores_arithmetic_mean :
    let scores := [85, 90, 78, 92, 87, 81, 93]
    let n := scores.length
    let mean := (scores.sum : ℝ) / n
    Real.round mean = 87 := by
  sorry

end sarah_scores_arithmetic_mean_l151_151141


namespace smallest_integer_is_10_l151_151806

noncomputable def smallest_integer (a b c : ℕ) : ℕ :=
  if h : (a + b + c = 90) ∧ (2 * b = 3 * a) ∧ (5 * a = 2 * c)
  then a
  else 0

theorem smallest_integer_is_10 (a b c : ℕ) (h₁ : a + b + c = 90) (h₂ : 2 * b = 3 * a) (h₃ : 5 * a = 2 * c) : 
  smallest_integer a b c = 10 :=
sorry

end smallest_integer_is_10_l151_151806


namespace ammonia_moles_l151_151605

-- Definitions corresponding to the given conditions
def moles_KOH : ℚ := 3
def moles_NH4I : ℚ := 3

def balanced_equation (n_KOH n_NH4I : ℚ) : ℚ :=
  if n_KOH = n_NH4I then n_KOH else 0

-- Proof problem: Prove that the reaction produces 3 moles of NH3
theorem ammonia_moles (n_KOH n_NH4I : ℚ) (h1 : n_KOH = moles_KOH) (h2 : n_NH4I = moles_NH4I) :
  balanced_equation n_KOH n_NH4I = 3 :=
by 
  -- proof here 
  sorry

end ammonia_moles_l151_151605


namespace chord_length_intersection_l151_151160

open Real

-- Define a circle with center (1, 3) and radius √10
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 10

-- Define a line in the form of 4x - 3y = 0
def line (x y : ℝ) : Prop := 4 * x - 3 * y = 0

-- Define the chord length problem
theorem chord_length_intersection :
  let c := 1 in
  let r := sqrt 10 in
  let d := 1 in
  let l := 2 * sqrt (r^2 - d^2) in
  l = 6 := by
  -- Use calculation to substitute the definitions and prove the theorem
  sorry

end chord_length_intersection_l151_151160


namespace f_neither_odd_nor_even_num_solutions_f_eq_0_l151_151948

noncomputable def f : ℝ → ℝ := sorry

axiom f_symm1 : ∀ x : ℝ, f (2 + x) = f (2 - x)
axiom f_symm2 : ∀ x : ℝ, f (7 + x) = f (7 - x)
axiom f_vals : f 1 = 0 ∧ f 3 = 0

theorem f_neither_odd_nor_even :
  ¬ (∀ x : ℝ, f x = f (-x)) ∧ ¬ (∀ x : ℝ, f x = -f (-x)) := sorry

theorem num_solutions_f_eq_0 : 
  ∃ n : ℕ, n = 802 ∧ (∑ x in Icc (-2005 : ℝ) 2005, if f x == 0 then 1 else 0) = n := sorry

end f_neither_odd_nor_even_num_solutions_f_eq_0_l151_151948


namespace min_tests_required_l151_151273

namespace BallTests

-- Define the parameters of the problem.
def numBalls : Nat := 99
def numCopperBalls : Nat := 50
def numZincBalls : Nat := 49
def testBallPairs (n : Nat) : Prop := n ≥ 98

-- The main theorem stating that at least 98 tests are required to distinguish the balls.
theorem min_tests_required : ∀ (n : Nat), (numBalls = 99) → (numCopperBalls = 50) → (numZincBalls = 49) → testBallPairs n :=
begin
  intros n numBalls_eq numCopperBalls_eq numZincBalls_eq,
  -- Proof goes here
  sorry
end

end BallTests

end min_tests_required_l151_151273


namespace solution_exists_l151_151204

noncomputable def find_p_q : Prop :=
  ∃ p q : ℕ, (p^q - q^p = 1927) ∧ (p = 2611) ∧ (q = 11)

theorem solution_exists : find_p_q :=
sorry

end solution_exists_l151_151204


namespace triangle_problem_l151_151277

theorem triangle_problem (perimeter_XYZ : ℝ) (angle_XZY : ℝ) (radius_O : ℝ)
  (O_on_XZ : ∃ O : ℝ, O ∈ set.interval 0 120) (tangent_ZY_YX : ∃ O : ℝ, ∀ ZY YX : ℝ, tangent ZY O ∧ tangent YX O)
  (h1 : perimeter_XYZ = 120)
  (h2 : angle_XZY = 90)
  (h3 : radius_O = 15) :
  let OY := 5 / 2 in ∃ p q : ℕ, nat.gcd p q = 1 ∧ OY = p / q ∧ p + q = 7 := by
  sorry

end triangle_problem_l151_151277


namespace find_m_n_l151_151961

theorem find_m_n 
  (a b c d m n : ℕ) 
  (h₁ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₂ : a + b + c + d = m^2)
  (h₃ : a = max (max a b) (max c d) ∨ b = max (max a b) (max c d) ∨ c = max (max a b) (max c d) ∨ d = max (max a b) (max c d))
  (h₄ : exists k, k^2 = max (max a b) (max c d))
  : m = 9 ∧ n = 6 :=
by
  -- Proof omitted
  sorry

end find_m_n_l151_151961


namespace no_solutions_sinx_eq_sin_sinx_l151_151991

open Real

theorem no_solutions_sinx_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ arcsin 0.9) : ¬ (sin x = sin (sin x)) :=
by
  sorry

end no_solutions_sinx_eq_sin_sinx_l151_151991


namespace sec_150_eq_l151_151405

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l151_151405


namespace sec_150_eq_neg_two_sqrt_three_div_three_l151_151357

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l151_151357


namespace first_term_geometric_progression_l151_151200

theorem first_term_geometric_progression (S : ℝ) (sum_first_two_terms : ℝ) (a : ℝ) (r : ℝ) :
  S = 8 → sum_first_two_terms = 5 →
  (a = 8 * (1 - (Real.sqrt 6) / 4)) ∨ (a = 8 * (1 + (Real.sqrt 6) / 4)) :=
by
  sorry

end first_term_geometric_progression_l151_151200


namespace Jacks_100th_number_is_156_l151_151073

-- Define a function that checks if a number contains the digit 2 or 9
def contains_two_or_nine (n : Nat) : Bool :=
  n.digits 10 |> List.any (λ d => d = 2 ∨ d = 9)

-- Define a function that generate the list of valid numbers
def valid_numbers : List Nat :=
  List.filter (λ n => ¬contains_two_or_nine n) (List.range 200) -- we take a range beyond 100 to ensure we cover the required cases

-- Define the 100th valid number Jack writes
def Jacks_100th_valid_number := valid_numbers.get 99 -- 0-indexed, so 99 is the 100th

theorem Jacks_100th_number_is_156 :
  Jacks_100th_valid_number = 156 :=
by 
  -- The proof would go here
  sorry

end Jacks_100th_number_is_156_l151_151073


namespace sum_first_10_log2_terms_l151_151976

theorem sum_first_10_log2_terms (a : ℕ → ℝ) (a4_eq_2 : a 4 = 2) (a7_eq_16 : a 7 = 16) :
  (∑ n in Finset.range 10, log 2 (a (n + 1))) = 25 :=
sorry

end sum_first_10_log2_terms_l151_151976


namespace min_value_of_c_l151_151130

theorem min_value_of_c (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 1503)
  (h4 : ∀ x y, (2 * x + y = 2008) ∧ (y = |x - a| + |x - b| + |x - c|) → (x, y) = (b, 2008 - 2 * b)):
  c = 496 :=
sorry

end min_value_of_c_l151_151130


namespace math_problem_l151_151001

-- Conditions
def f (a b x : ℝ) : ℝ := (2 * a * x + b) / (x^2 + 1)
def f_odd (a b : ℝ) : Prop := ∀ x : ℝ, f a b x = -f a b (-x)
def f_at_half (a b : ℝ) : Prop := f a b (1/2) = 4 / 5
def f_range_inequality (a b m x t : ℝ) : Prop := f a b x ≤ m^2 - 5 * m * t - 5

-- Proof statement
theorem math_problem (a b : ℝ) :
  (f_odd a b) → 
  (f_at_half a b) → 
  (a = 1 ∧ b = 0) ∧
  (∀ x y : ℝ, x ∈ Icc (-1 : ℝ) 1 → y ∈ Icc (-1 : ℝ) 1 → x < y → f 1 0 x < f 1 0 y) ∧ 
  (∀ m : ℝ, (∀ x t : ℝ, x ∈ Icc (-1 : ℝ) 1 → t ∈ Icc (-1 : ℝ) 1 → f_range_inequality 1 0 m x t) → 
    (m ∈ Icc (6 : ℝ) (⊤ : ℝ) ∨ m ∈ Icc (⊥ : ℝ) (-6))) :=
by {
  intros,
  split,
  -- First part: proof for a = 1 and b = 0
  sorry,
  split,
  -- Second part: proof for monotonicity on [-1, 1]
  sorry,
  -- Third part: proof for range of m
  sorry,
}

end math_problem_l151_151001


namespace sec_150_eq_neg_2_sqrt_3_div_3_l151_151589

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l151_151589


namespace james_played_five_rounds_l151_151761

variables (R : ℕ) -- Number of rounds played
variables (points_per_correct : ℕ) (bonus_points : ℕ) (questions_per_round : ℕ) (total_points : ℕ)
variables (points_missed_round : ℕ)

/-- 
  Students earn 2 points for each correct answer during a quiz bowl. 
  A student is awarded an additional 4 points bonus if all questions in a round are correctly answered.
  Each round consists of 5 questions.
  James only missed one question and got 66 points.
  Prove James played 5 rounds.
-/
theorem james_played_five_rounds
  (points_per_correct := 2)
  (bonus_points := 4)
  (questions_per_round := 5)
  (total_points := 66)
  (points_missed_round := (2 * (questions_per_round - 1))) :
  R = 5 :=
begin
  have max_points_per_round := points_per_correct * questions_per_round + bonus_points,
  have points_no_bonus := points_per_correct * (questions_per_round - 1),
  have total_remaining_points := total_points - points_no_bonus,
  have rounds_with_full_points := total_remaining_points / max_points_per_round,
  have total_rounds := rounds_with_full_points + 1,
  exact eq.symm (nat.succ pred R = total_rounds)
end

end james_played_five_rounds_l151_151761
