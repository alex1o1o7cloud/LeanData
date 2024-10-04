import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Floor
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Distributions
import Mathlib.Data.Probability.ProbabilitySpace
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.NumberTheory.Digits
import Mathlib.NumberTheory.ModularArithmetic
import Mathlib.NumberTheory.Totient
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Statistics.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace correct_propositions_l602_602319

-- Definitions for the conditions
variables (α β : Type) -- Representing planes
variable (l m : Type) -- Representing lines

-- Propositions as Boolean statements
def proposition1 (h1 : α ∥ β) (h2 : l ⊆ α) : Prop := l ∥ β
def proposition2 (h1 : α ∥ β) (h2 : l ⊥ α) : Prop := l ⊥ β
def proposition3 (h1 : l ∥ α) (h2 : m ⊆ α) : Prop := l ∥ m
def proposition4 (h1 : α ⊥ β) (h2 : α ∩ β = l) (h3 : m ⊆ α) (h4 : m ⊥ l) : Prop := m ⊥ β

-- Lean statement for the theorem with the answers identified true or false
theorem correct_propositions (h1 : α ∥ β) (h2 : l ⊆ α) (h3 : l ⊥ α) (h4 : l ∥ α) (h5 : m ⊆ α) (h6 : α ⊥ β) (h7 : α ∩ β = l) (h8 : m ⊥ l) :
  (proposition1 h1 h2) ∧
  (proposition2 h1 h3) ∧
  ¬ (proposition3 h4 h5) ∧
  (proposition4 h6 h7 h5 h8) :=
by -- Proof placeholder
   sorry

end correct_propositions_l602_602319


namespace nth_monomial_correct_l602_602619

-- Definitions of the sequence of monomials

def coeff (n : ℕ) : ℕ := 3 * n + 2
def exponent (n : ℕ) : ℕ := n

def nth_monomial (n : ℕ) (a : ℕ) : ℕ := (coeff n) * (a ^ (exponent n))

-- Theorem statement
theorem nth_monomial_correct (n : ℕ) (a : ℕ) : nth_monomial n a = (3 * n + 2) * (a ^ n) :=
by
  sorry

end nth_monomial_correct_l602_602619


namespace solution_set_f_1_minus_x_l602_602331

noncomputable def problem :=
  ∀ (f : ℝ → ℝ),
  (∀ x, f (-x) = -f (x+1)) → 
  (∀ x_1 x_2, x_1 ≠ x_2 → (x_1 - x_2) * (f x_1 - f x_2) < 0) →
  {x : ℝ | f (1 - x) < 0} = set.Iio 0

theorem solution_set_f_1_minus_x (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f (x+1))
  (h_decreasing : ∀ x_1 x_2, x_1 ≠ x_2 → (x_1 - x_2) * (f x_1 - f x_2) < 0) :
  {x : ℝ | f (1 - x) < 0} = set.Iio 0 :=
sorry

end solution_set_f_1_minus_x_l602_602331


namespace max_p_of_chessboard_cutting_l602_602651

theorem max_p_of_chessboard_cutting :
  ∃ (p : ℕ), p = 7 ∧ 
  (∃ (a : Fin p → ℕ), (∑ i:Fin p, a i = 32) ∧ 
  (∀ i j : Fin p, i < j → a i < a j)) ∧ 
  (∀ i, (i < p / 2 → a i = (nat.succ i) * 2 - 1) ∧ 
  (p / 2 ≤ i → a i < 32)) :=
begin
  sorry
end

end max_p_of_chessboard_cutting_l602_602651


namespace total_ants_employed_l602_602081

theorem total_ants_employed
  (M_r : ℕ) (M_b : ℕ) (M_y : ℕ)
  (T_r : ℕ) (T_b : ℕ) (T_g : ℕ)
  (A_r : ℕ) (A_b : ℕ) (A_blue : ℕ)
  (black_ants_percent: ℕ -> Nat) :
  M_r = 413 ∧ M_b = 487 ∧ M_y = 360 ∧
  T_r = 356 ∧ T_b = 518 ∧ T_g = 250 ∧
  A_r = 298 ∧ A_b = 392 ∧ A_blue = 200 ∧
  black_ants_percent M_b = (M_b * 25 / 100).toNat →
  let total := M_r + M_b + M_y + T_r + T_b + T_g + A_r + A_b + A_blue - black_ants_percent M_b in
  total = 3153 :=
by {
  sorry
}

# Check the theorem to ensure it builds: total_ants_employed

end total_ants_employed_l602_602081


namespace unit_vector_norm_equal_l602_602711

variables (a b : EuclideanSpace ℝ (Fin 2)) -- assuming 2D Euclidean space for simplicity

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ‖v‖ = 1

theorem unit_vector_norm_equal {a b : EuclideanSpace ℝ (Fin 2)}
  (ha : is_unit_vector a) (hb : is_unit_vector b) : ‖a‖ = ‖b‖ :=
by 
  sorry

end unit_vector_norm_equal_l602_602711


namespace no_solution_for_factorial_product_l602_602268

theorem no_solution_for_factorial_product : ¬ ∃ (n : ℕ), ∃ (a : ℕ), a ≥ 2 ∧ a + n - 2 > n ∧ n! = (a + n - 2)! / a! := 
by
  sorry

end no_solution_for_factorial_product_l602_602268


namespace least_common_multiple_9_12_15_l602_602977

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l602_602977


namespace find_k_l602_602095

noncomputable def intersection_point (x y : ℝ) :=
  (4 * x - 2 = y) ∧ (-3 * x + 9 = y) ∧ (2 * x + (8 / 7) = y)

theorem find_k : ∃ k : ℝ, k = 8 / 7 ∧ 
  ∃ x y : ℝ, (4 * x - 2 = y) ∧ (-3 * x + 9 = y) ∧ (2 * x + k = y) :=
by {
  use 8 / 7,
  existsi [11 / 7, 30 / 7],
  split,
  { linarith, },
  split,
  { linarith, },
  linarith,
}

end find_k_l602_602095


namespace range_of_a10_l602_602810

variable {a d : ℝ} -- Define our variables

def arithmetic_sequence (n : ℕ) : ℝ := a + (↑n - 1) * d

-- Conditions given in the problem
def condition_1 := arithmetic_sequence 2 ≤ 7
def condition_2 := arithmetic_sequence 6 ≥ 9

-- The main theorem which encapsulates our proof problem
theorem range_of_a10 (h1 : condition_1) (h2 : condition_2) : 11 < arithmetic_sequence 10 := by
  sorry -- proof will go here

end range_of_a10_l602_602810


namespace find_original_price_l602_602015

noncomputable def original_price_per_bottle (P : ℝ) : Prop :=
  let discounted_price := 0.80 * P
  let final_price_per_bottle := discounted_price - 2.00
  3 * final_price_per_bottle = 30

theorem find_original_price : ∃ P : ℝ, original_price_per_bottle P ∧ P = 15 :=
by
  sorry

end find_original_price_l602_602015


namespace angle_BC_proof_l602_602021

-- Definitions of given facts
variables {A B C D E : Type}
variables (ABC : Triangle A B C)
variables (angleA : angle A = 60) (AB_ne_AC : side AB ≠ side AC)
variable (AD_bisector : AngleBisector A D)
variable (E_on_e : Perpendicular E (AngleBisector A D) A)
variable (BE_len : side BE = side AB + side AC)

-- Proof obligations
theorem angle_BC_proof (ABC : Triangle A B C)
  (angleA : angle A = 60)
  (AB_ne_AC : side AB ≠ side AC)
  (AD_bisector : AngleBisector A D)
  (E_on_e : Perpendicular E (AngleBisector A D) A)
  (BE_len : side BE = side AB + side AC) :
  angle B = 80 ∧ angle C = 40 :=
by
  sorry

end angle_BC_proof_l602_602021


namespace heather_starts_24_minutes_after_stacy_l602_602493

theorem heather_starts_24_minutes_after_stacy :
  ∀ (distance_between : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) (heather_distance : ℝ),
    distance_between = 10 →
    heather_speed = 5 →
    stacy_speed = heather_speed + 1 →
    heather_distance = 3.4545454545454546 →
    60 * ((heather_distance / heather_speed) - ((distance_between - heather_distance) / stacy_speed)) = -24 :=
by
  sorry

end heather_starts_24_minutes_after_stacy_l602_602493


namespace triangle_AFE_angles_l602_602396

theorem triangle_AFE_angles :
  ∀ (A B C D F E : Type) 
  [geometry_type A B C D F E]
  (h_right_isosceles_ABC : is_right_isosceles_triangle A B C)
  (h_AC_eq_BC : dist A C = dist B C)
  (h_mid_D : midpoint D A B)
  (h_mid_F : midpoint F B C)
  (h_on_ray_E_DC : on_ray D C E)
  (h_AF_eq_FE : dist A F = dist F E),
  (angles_of_triangle A F E = (45°, 45°, 90°)) :=
by
  sorry

end triangle_AFE_angles_l602_602396


namespace exist_subset_complex_magnitude_at_least_one_sixth_l602_602843

theorem exist_subset_complex_magnitude_at_least_one_sixth
  (n : ℕ)
  (z : ℕ → ℂ)
  (h_sum_mag : (Finset.range n).sum (λ k, complex.abs (z k)) = 1) :
  ∃ (s : Finset ℕ), s.sum (λ k, complex.abs (z k)) ≥ 1 / 6 :=
sorry

end exist_subset_complex_magnitude_at_least_one_sixth_l602_602843


namespace triangle_external_angle_properties_l602_602803

theorem triangle_external_angle_properties (A B C : ℝ) (hA : 0 < A ∧ A < 180) (hB : 0 < B ∧ B < 180) (hC : 0 < C ∧ C < 180) (h_sum : A + B + C = 180) :
  (∃ E1 E2 E3, E1 + E2 + E3 = 360 ∧ E1 > 90 ∧ E2 > 90 ∧ E3 <= 90) :=
by
  sorry

end triangle_external_angle_properties_l602_602803


namespace incorrect_statement_l602_602274

noncomputable def systolic_pressures : List ℝ := [151, 148, 140, 139, 140, 136, 140]
noncomputable def diastolic_pressures : List ℝ := [90, 92, 88, 88, 90, 80, 88]

def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get (l.length / 2) sorry

def mode (l : List ℝ) : ℝ :=
  (l.groupBy id).map (λ g => (g.head!, g.length)).maxBy (λ p => p.snd).fst

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem incorrect_statement :
  median systolic_pressures = 140 ∧
  mode diastolic_pressures = 88 ∧
  mean systolic_pressures = 142 ∧
  variance diastolic_pressures = 88 / 7 ∧
  (∀ statement : char, statement = 'A') :=
by 
  sorry

end incorrect_statement_l602_602274


namespace sum_non_empty_subsets_no_real_roots_in_interval_l602_602571

-- Prove that the sum of the elements of all non-empty subsets of set {1,2,3,...,10} is 28160
theorem sum_non_empty_subsets :
  let A := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  (finset.sum (finset.filter (λ s, s.nonempty) (finset.powerset A)) (λ s, finset.sum s id)) = 28160 := sorry

-- Prove that there are no real roots of the equation log₂(log₃(x)) = log₃(log₂(x)) in the interval [2, 3]
theorem no_real_roots_in_interval :
  ¬ ∃ x, 2 ≤ x ∧ x ≤ 3 ∧ real.log 2 (real.log 3 x) = real.log 3 (real.log 2 x) := sorry

end sum_non_empty_subsets_no_real_roots_in_interval_l602_602571


namespace total_votes_polled_diff_votes_B_C_l602_602802

-- Candidates A, B, and C receive portions of the votes.
constants (V : ℕ) -- total number of votes

-- Conditions given in the problem
axiom candidate_A_votes : 0.42 * V
axiom candidate_B_votes : 0.37 * V
axiom candidate_A_margin_over_B : 0.05 * V = 650

-- Goals to prove
theorem total_votes_polled : V = 13000 :=
by
  have h1 : 0.05 * V = 650 := candidate_A_margin_over_B
  /- Continue with necessary steps to prove V = 13000, skipping steps -/
  sorry

theorem diff_votes_B_C (hV : V = 13000) : 0.37 * V - 0.21 * V = 2080 :=
by
   rw hV
   /- Calculate the difference and verify it is 2080, skipping steps -/
   sorry
  

end total_votes_polled_diff_votes_B_C_l602_602802


namespace sum_of_two_numbers_l602_602943

theorem sum_of_two_numbers (a b S : ℤ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 7) = 3 * S + 36 :=
by
  sorry

end sum_of_two_numbers_l602_602943


namespace slope_of_secant_line_l602_602272

theorem slope_of_secant_line : 
  let f := (λ x: ℝ, 2^x) in
  let x1 := 0 in
  let y1 := f x1 in
  let x2 := 1 in
  let y2 := f x2 in
  (x1 = 0) → (y1 = 1) → (x2 = 1) → (y2 = 2) → (y2 - y1) / (x2 - x1) = 1 :=
by
  intros f x1 y1 x2 y2 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact sorry

end slope_of_secant_line_l602_602272


namespace variance_transformation_l602_602336

theorem variance_transformation (a_1 a_2 a_3 : ℝ) (h : (1 / 3) * ((a_1 - ((a_1 + a_2 + a_3) / 3))^2 + (a_2 - ((a_1 + a_2 + a_3) / 3))^2 + (a_3 - ((a_1 + a_2 + a_3) / 3))^2) = 1) :
  (1 / 3) * ((3 * a_1 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_2 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_3 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2) = 9 := by 
  sorry

end variance_transformation_l602_602336


namespace greatest_prime_factor_of_221_l602_602970

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greatest_prime_factor (n : ℕ) (p : ℕ) : Prop := 
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q → q ∣ n → q ≤ p

theorem greatest_prime_factor_of_221 : greatest_prime_factor 221 17 := by
  sorry

end greatest_prime_factor_of_221_l602_602970


namespace square_perimeter_l602_602600

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end square_perimeter_l602_602600


namespace determine_tomorrow_l602_602877

-- Defining the days of the week
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open DayOfWeek

-- Defining a function to add a certain number of days to a given day
def addDays (start_day : DayOfWeek) (n : Nat) : DayOfWeek :=
  match start_day, n % 7 with
  | Monday, 0 => Monday
  | Monday, 1 => Tuesday
  | Monday, 2 => Wednesday
  | Monday, 3 => Thursday
  | Monday, 4 => Friday
  | Monday, 5 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 0 => Tuesday
  | Tuesday, 1 => Wednesday
  | Tuesday, 2 => Thursday
  | Tuesday, 3 => Friday
  | Tuesday, 4 => Saturday
  | Tuesday, 5 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 0 => Wednesday
  | Wednesday, 1 => Thursday
  | Wednesday, 2 => Friday
  | Wednesday, 3 => Saturday
  | Wednesday, 4 => Sunday
  | Wednesday, 5 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 0 => Thursday
  | Thursday, 1 => Friday
  | Thursday, 2 => Saturday
  | Thursday, 3 => Sunday
  | Thursday, 4 => Monday
  | Thursday, 5 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 0 => Friday
  | Friday, 1 => Saturday
  | Friday, 2 => Sunday
  | Friday, 3 => Monday
  | Friday, 4 => Tuesday
  | Friday, 5 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 0 => Saturday
  | Saturday, 1 => Sunday
  | Saturday, 2 => Monday
  | Saturday, 3 => Tuesday
  | Saturday, 4 => Wednesday
  | Saturday, 5 => Thursday
  | Saturday, 6 => Friday
  | Sunday, 0 => Sunday
  | Sunday, 1 => Monday
  | Sunday, 2 => Tuesday
  | Sunday, 3 => Wednesday
  | Sunday, 4 => Thursday
  | Sunday, 5 => Friday
  | Sunday, 6 => Saturday

-- Conditions
axiom condition : Monday = addDays x 5

-- Find the day of the week tomorrow
theorem determine_tomorrow (x : DayOfWeek) : addDays (addDays x 2) 1 = Saturday := sorry

end determine_tomorrow_l602_602877


namespace value_of_expression_l602_602372

theorem value_of_expression (a : ℝ) (h : 10 * a^2 + 3 * a + 2 = 5) : 
  3 * a + 2 = (31 + 3 * Real.sqrt 129) / 20 :=
by sorry

end value_of_expression_l602_602372


namespace value_of_expression_when_x_is_neg2_l602_602147

theorem value_of_expression_when_x_is_neg2 : 
  ∀ (x : ℤ), x = -2 → (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_when_x_is_neg2_l602_602147


namespace smallest_D_for_inequality_l602_602681

theorem smallest_D_for_inequality :
  ∃ D : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ D * (x + y + z)) ∧ 
           D = -Real.sqrt (72 / 11) :=
by
  sorry

end smallest_D_for_inequality_l602_602681


namespace tangent_line_at_zero_l602_602507

def f (x : ℝ) : ℝ := Math.cos x + Real.exp x

theorem tangent_line_at_zero :
  ∃ m b, (∀ x, f(x) = m * x + b) ∧ (x * -1 + y = b) :=
by
  use 1, -2
  sorry

end tangent_line_at_zero_l602_602507


namespace find_a_f_increasing_l602_602040

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x) / a + a / (Real.exp x)

-- Define the conditions as lemmas
lemma condition_even_function (a : ℝ) (h : a > 0) (x : ℝ) : f a (-x) = f a x := 
  by
  unfold f
  simp
  sorry -- This is the symmetry proof

-- Main proof statements
theorem find_a (h : ∀ x : ℝ, f a (-x) = f a x) (h_pos : a > 0) : a = 1 :=
  by
  sorry -- Proof that a = 1 based on given conditions

theorem f_increasing (a : ℝ) (h : a = 1) (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx12 : x1 < x2) : 
  f a x1 < f a x2 :=
  by
  sorry -- Proof that the function is increasing on (0, +∞)

end find_a_f_increasing_l602_602040


namespace no_term_of_sequence_is_sum_of_three_7th_powers_l602_602519

def sequence (n : ℕ) : ℤ := 
  if n = 1 then 8 else
  if n = 2 then 20 else
  sequence (n-1) ^ 2 + 12 * sequence (n-1) * sequence (n-2) + sequence (n-1) + 11 * sequence (n-2)

theorem no_term_of_sequence_is_sum_of_three_7th_powers :
  ∀ n : ℕ, ¬ ∃ (x y z : ℤ), sequence n = x^7 + y^7 + z^7 :=
by sorry

end no_term_of_sequence_is_sum_of_three_7th_powers_l602_602519


namespace min_f_g_zeros_factorial_inequality_l602_602725

-- Problem (1)
def f (x : ℝ) : ℝ := x - Real.log x - 1

theorem min_f : ∀ x > 0, (x - Real.log x - 1 >= 0) :=
sorry

-- Problem (2)
def g (x : ℝ) : ℝ := (x - 1) * Real.log x - x - 1

theorem g_zeros : ∀ x, x > 0 → g x = 0 → ∃ (x1 x2 : ℝ), (x1 > 0 ∧ x2 > 0 
                      ∧ g x1 = 0 ∧ g x2 = 0 ∧ x1 * x2 = 1) :=
sorry

-- Problem (3)
theorem factorial_inequality : ∀ n : ℕ, n > 0 → Real.sqrt n! > (n + 1 : ℝ) / Real.exp 1 :=
sorry

end min_f_g_zeros_factorial_inequality_l602_602725


namespace tomorrow_is_saturday_l602_602871

noncomputable def day_before_yesterday : string := "Wednesday"
noncomputable def today : string := "Friday"
noncomputable def tomorrow : string := "Saturday"

theorem tomorrow_is_saturday (dby : string) (tod : string) (tom : string) 
  (h1 : dby = "Wednesday") (h2 : tod = "Friday") (h3 : tom = "Saturday")
  (h_cond : "Monday" = dby + 5) : 
  tom = "Saturday" := 
sorry

end tomorrow_is_saturday_l602_602871


namespace find_g1_gneg1_l602_602780

variables {f g : ℝ → ℝ}

theorem find_g1_gneg1 (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
                      (h2 : f (-2) = f 1 ∧ f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end find_g1_gneg1_l602_602780


namespace determine_tomorrow_l602_602875

-- Defining the days of the week
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open DayOfWeek

-- Defining a function to add a certain number of days to a given day
def addDays (start_day : DayOfWeek) (n : Nat) : DayOfWeek :=
  match start_day, n % 7 with
  | Monday, 0 => Monday
  | Monday, 1 => Tuesday
  | Monday, 2 => Wednesday
  | Monday, 3 => Thursday
  | Monday, 4 => Friday
  | Monday, 5 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 0 => Tuesday
  | Tuesday, 1 => Wednesday
  | Tuesday, 2 => Thursday
  | Tuesday, 3 => Friday
  | Tuesday, 4 => Saturday
  | Tuesday, 5 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 0 => Wednesday
  | Wednesday, 1 => Thursday
  | Wednesday, 2 => Friday
  | Wednesday, 3 => Saturday
  | Wednesday, 4 => Sunday
  | Wednesday, 5 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 0 => Thursday
  | Thursday, 1 => Friday
  | Thursday, 2 => Saturday
  | Thursday, 3 => Sunday
  | Thursday, 4 => Monday
  | Thursday, 5 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 0 => Friday
  | Friday, 1 => Saturday
  | Friday, 2 => Sunday
  | Friday, 3 => Monday
  | Friday, 4 => Tuesday
  | Friday, 5 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 0 => Saturday
  | Saturday, 1 => Sunday
  | Saturday, 2 => Monday
  | Saturday, 3 => Tuesday
  | Saturday, 4 => Wednesday
  | Saturday, 5 => Thursday
  | Saturday, 6 => Friday
  | Sunday, 0 => Sunday
  | Sunday, 1 => Monday
  | Sunday, 2 => Tuesday
  | Sunday, 3 => Wednesday
  | Sunday, 4 => Thursday
  | Sunday, 5 => Friday
  | Sunday, 6 => Saturday

-- Conditions
axiom condition : Monday = addDays x 5

-- Find the day of the week tomorrow
theorem determine_tomorrow (x : DayOfWeek) : addDays (addDays x 2) 1 = Saturday := sorry

end determine_tomorrow_l602_602875


namespace car_p_less_hours_l602_602107

theorem car_p_less_hours (distance : ℕ) (speed_r : ℕ) (speed_p : ℕ) (time_r : ℕ) (time_p : ℕ) (h1 : distance = 600) (h2 : speed_r = 50) (h3 : speed_p = speed_r + 10) (h4 : time_r = distance / speed_r) (h5 : time_p = distance / speed_p) : time_r - time_p = 2 := 
by
  sorry

end car_p_less_hours_l602_602107


namespace minimum_norm_a_add_tb_collinear_a_sub_tb_with_c_l602_602357

-- Given vectors a = (-3, 2), b = (2, 1), and c = (3, -1)
def vector_a : (ℝ × ℝ) := (-3, 2)
def vector_b : (ℝ × ℝ) := (2, 1)
def vector_c : (ℝ × ℝ) := (3, -1)

-- Problem 1: Prove the minimum value of ‖a + t * b‖ is 7 * sqrt(5) / 5 when t = 4 / 5
theorem minimum_norm_a_add_tb :
  ∀ t : ℝ, ∥ (vector_a.1 + t * vector_b.1, vector_a.2 + t * vector_b.2) ∥ ≥ 7 * real.sqrt 5 / 5 ∧
    (∥ (vector_a.1 + (4 / 5) * vector_b.1, vector_a.2 + (4 / 5) * vector_b.2) ∥ = 7 * real.sqrt 5 / 5) :=
sorry

-- Problem 2: If a - t * b is collinear with c, prove t = 3 / 5
theorem collinear_a_sub_tb_with_c :
  ∀ t : ℝ, (vector_a.1 - t * vector_b.1) * vector_c.2 - (vector_a.2 - t * vector_b.2) * vector_c.1 = 0 ↔ t = 3 / 5 :=
sorry

end minimum_norm_a_add_tb_collinear_a_sub_tb_with_c_l602_602357


namespace points_opposite_side_of_line_l602_602784

theorem points_opposite_side_of_line :
  (∀ a : ℝ, ((2 * 2 - 3 * 1 + a) * (2 * 4 - 3 * 3 + a) < 0) ↔ -1 < a ∧ a < 1) :=
by sorry

end points_opposite_side_of_line_l602_602784


namespace no_natural_number_with_d_eq_10_l602_602894

def D (n : ℕ) : Finset ℕ :=
  {m ∈ Finset.range n.succ | ∃ d > 1, d ∣ n ∧ d ∣ m}

def d (n : ℕ) : ℕ :=
  (D n).card

theorem no_natural_number_with_d_eq_10 : ¬ ∃ n : ℕ, d n = 10 := 
by 
  sorry

end no_natural_number_with_d_eq_10_l602_602894


namespace num_even_digit_four_digit_integers_l602_602180

theorem num_even_digit_four_digit_integers :
  let even_digits := {0, 2, 4, 6, 8} in
  let first_digit_choices := {2, 4, 6, 8} in
  let remaining_digit_choices := even_digits in
  ∃ num : ℕ, num = 4 * 5 * 5 * 5 ∧ num = 500 :=
by
  let even_digits := {0, 2, 4, 6, 8}
  let first_digit_choices := {2, 4, 6, 8}
  let remaining_digit_choices := even_digits
  use 4 * 5 * 5 * 5
  split
  · sorry
  · sorry

end num_even_digit_four_digit_integers_l602_602180


namespace joshua_needs_more_cents_l602_602828

-- Definitions of inputs
def cost_of_pen_dollars : ℕ := 6
def joshua_money_dollars : ℕ := 5
def borrowed_cents : ℕ := 68

-- Convert dollar amounts to cents
def dollar_to_cents (d : ℕ) : ℕ := d * 100

def cost_of_pen_cents := dollar_to_cents cost_of_pen_dollars
def joshua_money_cents := dollar_to_cents joshua_money_dollars

-- Total amount Joshua has in cents
def total_cents := joshua_money_cents + borrowed_cents

-- Calculation of the required amount
def needed_cents := cost_of_pen_cents - total_cents

theorem joshua_needs_more_cents : needed_cents = 32 := by 
  sorry

end joshua_needs_more_cents_l602_602828


namespace probability_of_symmetric_line_l602_602796

noncomputable def probability_symmetric_line :=
  let total_points := 121
  let remaining_points := 119
  let symmetric_points := 40
  symmetric_points / remaining_points

theorem probability_of_symmetric_line :
  probability_symmetric_line = 40 / 119 := 
by sorry

end probability_of_symmetric_line_l602_602796


namespace find_a_l602_602333

theorem find_a (a x : ℝ) (h1 : 3 * x + 2 * a = 2) (h2 : x = 1) : a = -1/2 :=
by
  sorry

end find_a_l602_602333


namespace decreasing_function_l602_602625

-- Define the functions
noncomputable def fA (x : ℝ) : ℝ := 3^x
noncomputable def fB (x : ℝ) : ℝ := Real.logb 0.5 x
noncomputable def fC (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fD (x : ℝ) : ℝ := 1/x

-- Define the domains
def domainA : Set ℝ := Set.univ
def domainB : Set ℝ := {x | x > 0}
def domainC : Set ℝ := {x | x ≥ 0}
def domainD : Set ℝ := {x | x < 0} ∪ {x | x > 0}

-- Prove that fB is the only decreasing function in its domain
theorem decreasing_function:
  (∀ x y, x ∈ domainA → y ∈ domainA → x < y → fA x > fA y) = false ∧
  (∀ x y, x ∈ domainB → y ∈ domainB → x < y → fB x > fB y) ∧
  (∀ x y, x ∈ domainC → y ∈ domainC → x < y → fC x > fC y) = false ∧
  (∀ x y, x ∈ domainD → y ∈ domainD → x < y → fD x > fD y) = false :=
  sorry

end decreasing_function_l602_602625


namespace leopards_arrangement_l602_602450

theorem leopards_arrangement :
  let total_leopards := 9
  let ends_leopards := 2
  let middle_leopard := 1
  let remaining_leopards := total_leopards - ends_leopards - middle_leopard
  (2 * 1 * (Nat.factorial remaining_leopards) = 1440) := by
  sorry

end leopards_arrangement_l602_602450


namespace area_tangency_triangle_correct_l602_602515

noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def area_tangency_triangle (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c in
  (2 * (s - a) * (s - b) * (s - c)) / (a * b * c) * (Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem area_tangency_triangle_correct (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  area_tangency_triangle a b c = 
    (2 * (semiperimeter a b c - a) * (semiperimeter a b c - b) * (semiperimeter a b c - c)) / 
      (a * b * c) * 
      Real.sqrt (semiperimeter a b c * (semiperimeter a b c - a) * (semiperimeter a b c - b) * (semiperimeter a b c - c)) := 
  by
  sorry

end area_tangency_triangle_correct_l602_602515


namespace meaningful_expression_range_l602_602751

theorem meaningful_expression_range (x : ℝ) : (∃ y : ℝ, y = (1 / (Real.sqrt (x - 2)))) ↔ (x > 2) := 
sorry

end meaningful_expression_range_l602_602751


namespace birthday_next_tuesday_l602_602045

-- Define the concept of a year being leap or normal
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

theorem birthday_next_tuesday :
  let advance_year := λ (year_day : ℕ × ℕ), -- (year, current_weekday)
    let (year, weekday) := year_day in
    let next_weekday := if is_leap_year year then weekday + 2 else weekday + 1 in
    (year + 1, next_weekday % 7) in
  let december_3_2012 := (2012, 5) in -- 5 represents Friday
  nat.iterate advance_year 9 december_3_2012 = (2021, 2) -- 2 represents Tuesday
:= by
  sorry

end birthday_next_tuesday_l602_602045


namespace curtains_length_needed_l602_602261

def room_height_feet : ℕ := 8
def additional_material_inches : ℕ := 5

def height_in_inches : ℕ := room_height_feet * 12

def total_length_curtains : ℕ := height_in_inches + additional_material_inches

theorem curtains_length_needed : total_length_curtains = 101 := by
  sorry

end curtains_length_needed_l602_602261


namespace combination_addition_l602_602643

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem combination_addition :
  combination 13 11 + 3 = 81 :=
by
  sorry

end combination_addition_l602_602643


namespace lulu_savings_l602_602070

theorem lulu_savings (L : ℝ) (h1 : ∑ (x : ℝ) in {L, 5 * L, (5 / 3) * L}, x = 46) : L = 4.76 := 
by
  -- The proof is not required, so we use 'sorry'
  sorry

end lulu_savings_l602_602070


namespace sin_cos_identity_l602_602757

theorem sin_cos_identity (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602757


namespace lcm_9_12_15_l602_602996

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l602_602996


namespace cartesian_equations_and_minimum_distance_l602_602318

section

-- Define the conditions
def circleM_parametric (θ : ℝ) : ℝ × ℝ := 
  (5 * Real.sqrt 3 / 2 + 2 * Real.cos θ, 7 / 2 + 2 * Real.sin θ)

-- Circle N center in Cartesian coordinates
def centerN : ℝ × ℝ := (Real.sqrt 3 / 2, 3 / 2)

-- The equation of the circle N in Cartesian coordinates and its radius
def circleN (p : ℝ × ℝ) : Prop := 
  (p.1 - centerN.1)^2 + (p.2 - centerN.2)^2 = 1

theorem cartesian_equations_and_minimum_distance :
  (∀ x y θ, circleM_parametric θ = (x, y) ↔ ((x - 5 * Real.sqrt 3 / 2)^2 + (y - 7 / 2)^2 = 4)) ∧
  (∀ p, circleN p ↔ ((p.1 - Real.sqrt 3 / 2)^2 + (p.2 - 3 / 2)^2 = 1)) ∧
  (∃ P Q, (circleM_parametric P.2 = P ∧ circleN Q) → 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 1) :=
begin
  sorry
end

end

end cartesian_equations_and_minimum_distance_l602_602318


namespace real_solutions_system_l602_602644

theorem real_solutions_system :
  let system (x y z w : ℝ) := (x = z + w + z * w * x) ∧
                              (y = w + x + w * x * y) ∧
                              (z = x + y + x * y * z) ∧
                              (w = y + z + y * z * w)
  in
  ∀ x y z w : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 ∧ (x * y * z * w = 1) ∧ system x y z w ↔ x = 15 :=
by sorry

end real_solutions_system_l602_602644


namespace sixth_number_in_sequence_l602_602279

noncomputable def alternating_sequence_square : ℕ → ℤ
| (2 * n) + 1 := -(100 - n)^2
| (2 * n) := (100 - n)^2

theorem sixth_number_in_sequence : alternating_sequence_square 6 = 9025 := 
by
  simp [alternating_sequence_square]
  have : (100 - 5)^2 = 9025 := by norm_num
  exact this

end sixth_number_in_sequence_l602_602279


namespace distance_polar_coordinates_l602_602026

variables (r₁ θ₁ r₂ θ₂ : ℝ)
variables (h₁ : r₁ = 5) (h₂ : r₂ = 7) (hθ : θ₁ - θ₂ = Real.pi / 3)

theorem distance_polar_coordinates :
  let A := (r₁, θ₁),
      B := (r₂, θ₂),
      AB := Real.sqrt (r₁^2 + r₂^2 - 2 * r₁ * r₂ * Real.cos (θ₁ - θ₂))
  in AB = Real.sqrt 39 :=
by
  -- To be proved
  sorry

end distance_polar_coordinates_l602_602026


namespace sine_subtraction_formula_simplify_expression_l602_602487

-- Define the sine subtraction formula as a condition
theorem sine_subtraction_formula (a b : ℝ) : 
    real.sin (a - b) = real.sin a * real.cos b - real.cos a * real.sin b := by
  sorry

-- Prove the given expression simplifies to sin(x)
theorem simplify_expression (x y : ℝ) :
    real.sin (x + y) * real.cos y - real.cos (x + y) * real.sin y = real.sin x := by
  have h : real.sin ((x + y) - y) = real.sin (x + y) * real.cos y - real.cos (x + y) * real.sin y := by
    exact sine_subtraction_formula (x + y) y
  rw [sub_self y, h]
  simp
  sorry

end sine_subtraction_formula_simplify_expression_l602_602487


namespace cone_water_fill_ratio_l602_602193

theorem cone_water_fill_ratio (h r : ℝ) :
  let V := (1/3) * π * r^2 * h in
  let V_water := (1/3) * π * (2/3 * r)^2 * (2/3 * h) in
  (V_water / V : ℝ) ≈ 0.2963 :=
by
  sorry

end cone_water_fill_ratio_l602_602193


namespace total_legs_in_farm_l602_602110

theorem total_legs_in_farm (total_animals : ℕ) (total_cows : ℕ) (cow_legs : ℕ) (duck_legs : ℕ) 
  (h_total_animals : total_animals = 15) (h_total_cows : total_cows = 6) 
  (h_cow_legs : cow_legs = 4) (h_duck_legs : duck_legs = 2) :
  total_cows * cow_legs + (total_animals - total_cows) * duck_legs = 42 :=
by
  sorry

end total_legs_in_farm_l602_602110


namespace find_k_l602_602257

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (k : ℕ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_k (h_seq : arithmetic_sequence a d) (h_d_nonzero : d ≠ 0) (h_a1 : a 1 = 9 * d) (h_geom_mean : a k = real.sqrt (a 1 * a (2 * k))) :
  k = 4 := 
sorry

end find_k_l602_602257


namespace how_many_leaves_blew_away_l602_602568

theorem how_many_leaves_blew_away (original_leaves : ℕ) (leaves_left : ℕ) 
  (h_original : original_leaves = 356) (h_left : leaves_left = 112) :
  original_leaves - leaves_left = 244 :=
by
  -- Conditions given in the problem
  have h1 : 356 - 112 = 244 := by norm_num,
  rw [h_original, h_left],
  apply h1

end how_many_leaves_blew_away_l602_602568


namespace lemonade_water_needed_l602_602955

noncomputable theory
open Real

-- Definitions based on conditions
def parts_water : ℚ := 7
def parts_juice : ℚ := 1
def total_parts : ℚ := parts_water + parts_juice
def gallon_in_quarts : ℚ := 4
def part_size_in_quarts : ℚ := gallon_in_quarts / total_parts

-- Expected amount of water
def water_needed : ℚ := parts_water * part_size_in_quarts

-- Theorem to prove
theorem lemonade_water_needed :
  water_needed = 7 / 2 :=
by
  sorry

end lemonade_water_needed_l602_602955


namespace estimate_height_160_to_170_l602_602275

theorem estimate_height_160_to_170 :
  -- Conditions
  (total_students : ℕ) = 1500 →
  (height_160_165 : ℝ) = 0.01 →
  (height_165_170 : ℝ) = 0.05 →
  -- Statement to prove
  let width_160_165 := 165 - 160 in
  let width_165_170 := 170 - 165 in
  let proportion_160_165 := height_160_165 * width_160_165 in
  let proportion_165_170 := height_165_170 * width_165_170 in
  let total_proportion := proportion_160_165 + proportion_165_170 in
  let estimated_students := total_students * total_proportion in
  estimated_students = 450 :=
by
  intros
  sorry

end estimate_height_160_to_170_l602_602275


namespace find_f1_l602_602698

def f (x : ℝ) (f_1' : ℝ) : ℝ := 2 * x * f_1' + Real.log x

theorem find_f1'_value (x : ℝ) : 
  ∀ (f_1' : ℝ), (deriv (f x) x).eval 1 = -1 := by
  sorry

end find_f1_l602_602698


namespace right_triangle_area_l602_602618

theorem right_triangle_area
    (h : ∀ {a b c : ℕ}, a^2 + b^2 = c^2 → c = 13 → a = 5 ∨ b = 5)
    (hypotenuse : ℕ)
    (leg : ℕ)
    (hypotenuse_eq : hypotenuse = 13)
    (leg_eq : leg = 5) : ∃ (area: ℕ), area = 30 :=
by
  -- The proof will go here.
  sorry

end right_triangle_area_l602_602618


namespace pyramid_volume_l602_602395

-- Definitions of the conditions
def AB : ℝ := 4
def BC : ℝ := 2
def CG : ℝ := 5

-- Point N is the centroid of triangle FGH
def height_N_from_base : ℝ := 2 / 3 * CG

-- Area of the base BCFE
def area_BCFE : ℝ := BC * 4

-- Volume of the pyramid with base BCFE and apex N
def volume_base_height (base height : ℝ) : ℝ := 1 / 3 * base * height

-- Main theorem to prove
theorem pyramid_volume : volume_base_height area_BCFE height_N_from_base = 80 / 9 := by
  sorry

end pyramid_volume_l602_602395


namespace harmonic_mean_2_3_6_l602_602250

theorem harmonic_mean_2_3_6 : 
  let a := 2
  let b := 3
  let c := 6
  (3 / ((1 / a) + (1 / b) + (1 / c)) / 3) = 3 :=
by
  let a := 2
  let b := 3
  let c := 6
  have hrecip := (1 / a) + (1 / b) + (1 / c)
  have hmean := 3 / hrecip
  show hmean = 3, from
  sorry

end harmonic_mean_2_3_6_l602_602250


namespace least_number_to_add_to_246835_l602_602148

-- Define relevant conditions and computations
def lcm_of_169_and_289 : ℕ := Nat.lcm 169 289
def remainder_246835_mod_lcm : ℕ := 246835 % lcm_of_169_and_289
def least_number_to_add : ℕ := lcm_of_169_and_289 - remainder_246835_mod_lcm

-- The theorem statement
theorem least_number_to_add_to_246835 : least_number_to_add = 52 :=
by
  sorry

end least_number_to_add_to_246835_l602_602148


namespace tetrahedron_ratio_inspace_l602_602413

noncomputable def equilateral_tetrahedron :=
  {A B C D : Type* | ∀ (e : {A B C D}), (length e = 1)}

theorem tetrahedron_ratio_inspace (A B C D : equilateral_tetrahedron) 
  (M : midpoint B C)
  (O : center_of_circumscribed_sphere A B C D) : 
  ∃ (AO OM : ℝ), AO = (√6 / 4) ∧ OM = (√6 / 12) ∧ 
  (AO / OM = 3) :=
begin
  sorry
end

end tetrahedron_ratio_inspace_l602_602413


namespace slices_with_both_proof_l602_602237

-- Define the conditions: total slices, slices with at least one topping, slices with pepperoni, slices with mushrooms
variables (total_slices slices_with_pepperoni slices_with_mushrooms : ℕ)
variables (slices_with_both : ℕ)

-- Given conditions
def pizza_slices_conditions :=
  total_slices = 18 ∧
  slices_with_pepperoni = 10 ∧
  slices_with_mushrooms = 10 ∧
  ∀ s, s ∈ {slices_with_pepperoni, slices_with_mushrooms, slices_with_both} → s ≥ (0 : ℕ)

-- The proof problem
theorem slices_with_both_proof : pizza_slices_conditions total_slices slices_with_pepperoni slices_with_mushrooms slices_with_both →
  slices_with_both = 2 :=
begin
  sorry
end

end slices_with_both_proof_l602_602237


namespace tomorrow_is_Saturday_l602_602855
noncomputable theory

def day_of_week := ℕ
def Monday : day_of_week := 0
def Tuesday : day_of_week := 1
def Wednesday : day_of_week := 2
def Thursday : day_of_week := 3
def Friday : day_of_week := 4
def Saturday : day_of_week := 5
def Sunday : day_of_week := 6

def days_after (d : day_of_week) (n : ℕ) : day_of_week := (d + n) % 7

-- The condition: Monday is five days after the day before yesterday.
def day_before_yesterday := Wednesday
def today := days_after day_before_yesterday 2
def tomorrow := days_after today 1

theorem tomorrow_is_Saturday (h: days_after day_before_yesterday 5 = Monday) : tomorrow = Saturday := 
by {
  sorry
}

end tomorrow_is_Saturday_l602_602855


namespace find_x_l602_602945

noncomputable def inv_cubicroot (y x : ℝ) : ℝ := y * x^(1/3)

theorem find_x (x y : ℝ) (h1 : ∃ k, inv_cubicroot 2 8 = k) (h2 : y = 8) : x = 1 / 8 :=
by
  sorry

end find_x_l602_602945


namespace area_of_quadrilateral_ABCD_l602_602815

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem area_of_quadrilateral_ABCD :
  let AB := 15 * sqrt 2
  let BE := 15 * sqrt 2
  let BC := 7.5 * sqrt 2
  let CE := 7.5 * sqrt 6
  let CD := 7.5 * sqrt 2
  let DE := 7.5 * sqrt 6
  (1/2 * AB * BE) + (1/2 * BC * CE) + (1/2 * CD * DE) = 225 + 112.5 * sqrt 12 :=
by
  sorry

end area_of_quadrilateral_ABCD_l602_602815


namespace simplify_trig_identity_l602_602485

theorem simplify_trig_identity (x y : ℝ) : 
  sin (x + y) * cos y - cos (x + y) * sin y = sin x := 
by
  sorry

end simplify_trig_identity_l602_602485


namespace min_diameter_of_system_l602_602001

noncomputable def min_diameter (n : ℕ) (points : Finset (ℝ × ℝ)) : ℝ :=
  if ∀ (p1 p2 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → dist p1 p2 ≥ 1 then
    points.sup (λ p1, points.sup (λ p2, dist p1 p2))
  else
    0 -- This 0 case should actually never happen given the proper conditions.

theorem min_diameter_of_system (n : ℕ) (points : Finset (ℝ × ℝ)) :
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 →
  (∀ (p1 p2 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → dist p1 p2 ≥ 1) →
  min_diameter n points = sqrt 2 :=
by
  sorry

end min_diameter_of_system_l602_602001


namespace eight_square_root_pattern_l602_602811

theorem eight_square_root_pattern : 
  ∀ (n : ℕ), (8 * real.sqrt (8 / n) = real.sqrt (8 * 8 / n)) → n = 63 := 
by
  intro n

  -- Assuming the pattern x * sqrt(x / (x^2 - 1))
  have pattern : ∀ x : ℕ, x * real.sqrt (x / (x^2 - 1)) = real.sqrt (x * (x / (x^2 - 1))) :=
    by
      intro x
      sorry

  -- Check that 8 follows the pattern
  specialize pattern 8
  rw [pattern] at *

  -- Given that 8 * real.sqrt(8 / n) = real.sqrt (8 * 8 / n)
  intro h,
  
  -- Solve for n
  have h1 : 8 * 8 / n = 64 / n := by norm_num
  rw [h1] at h
  sorry

end eight_square_root_pattern_l602_602811


namespace minimize_y_l602_602349

noncomputable def y (x a b : ℝ) : ℝ := 2 * (x - a)^2 + 3 * (x - b)^2

theorem minimize_y (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) ∧ x = (2 * a + 3 * b) / 5 :=
sorry

end minimize_y_l602_602349


namespace sequence_inequality_l602_602038

def f (x: ℝ) := x / (x + 1)

def a : ℕ → ℝ
| 0 => 1 / 2
| 1 => 3 / 4
| (n + 2) => f (a n) + f (a (n + 1))

theorem sequence_inequality (n : ℕ) : 
   1 ≤ n → f(3 * 2^(n-1)) ≤ a (2 * n) ∧ a (2 * n) ≤ f(3 * 2^(2 * n - 2)) :=
by
  sorry

end sequence_inequality_l602_602038


namespace square_perimeter_l602_602592

theorem square_perimeter (area : ℝ) (h : area = 625) :
  ∃ p : ℝ, p = 4 * real.sqrt area ∧ p = 100 :=
by
  sorry

end square_perimeter_l602_602592


namespace determine_coin_types_l602_602109

structure Coin (weight : ℕ)

theorem determine_coin_types :
  ∀ (coins : List Coin), (length coins = 100) →
  (∃ g s c : Coin, Coin.weight g = 3 ∧ Coin.weight s = 2 ∧ Coin.weight c = 1) →
  ∃ types : List ℕ, (length types = 100) ∧
  (∀ (i : ℕ) (H : i < 100), Coin.weight (nth_le coins i H) = nth_le types i H) →
  ∃ (n : ℕ), n ≤ 100 :=
by
  sorry

end determine_coin_types_l602_602109


namespace no_adjacent_same_color_l602_602530

-- Definitions:
def is_adjacent (p1 p2 : ℕ) : Prop := p1 + 1 = p2 ∨ p2 + 1 = p1
def is_valid_arrangement (arr : list ℕ) : Prop := 
  ∀ i, i < arr.length - 1 → ¬is_adjacent arr[i] arr[i + 1]
def total_valid_arrangements : ℕ := 48

-- Problem statement:
theorem no_adjacent_same_color : 
  ∃ (arrangements : list (list ℕ)), 
    (length arrangements = total_valid_arrangements) ∧ 
    (∀ arr, arr ∈ arrangements → is_valid_arrangement arr) :=
sorry

end no_adjacent_same_color_l602_602530


namespace sum_of_infinite_geometric_series_l602_602302

theorem sum_of_infinite_geometric_series :
  let a := (2 / 3 : ℝ),
      S := ∑' n : ℕ, a^(n+1)
  in S = 2 := by
sorry

end sum_of_infinite_geometric_series_l602_602302


namespace log_sum_calculation_exponent_calculation_l602_602570

theorem log_sum_calculation :
  (\log_{2.5} 6.25) + (\log 10 0.01) + (Real.log (Real.sqrt Real.exp)) - (2 ^ (1 + \log_{2} 3)) = -\frac{11}{2} := by
  sorry

theorem exponent_calculation :
  (64^(-1/3)) - (-\frac{3 * Real.sqrt 2}{2})^0 + (2^(-3))^(4/3) + (16^(-0.75)) = -\frac{9}{16} := by
  sorry

end log_sum_calculation_exponent_calculation_l602_602570


namespace area_of_gray_region_is_96π_l602_602130

noncomputable def area_gray_region (d_small : ℝ) (r_ratio : ℝ) : ℝ :=
  let r_small := d_small / 2
  let r_large := r_ratio * r_small
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  area_large - area_small

theorem area_of_gray_region_is_96π :
  ∀ (d_small : ℝ) (r_ratio : ℝ), d_small = 4 → r_ratio = 5 → area_gray_region d_small r_ratio = 96 * π :=
by
  intros d_small r_ratio h1 h2
  have : d_small = 4 := h1
  have : r_ratio = 5 := h2
  sorry

end area_of_gray_region_is_96π_l602_602130


namespace ellipse_equation_l602_602382

theorem ellipse_equation (a b : ℝ) (e_he : Real.sqrt (1^2 + 1^2) = 2)
  (h_center : (0, 1) ∈ set_of (λ p : ℝ × ℝ, p.snd^2 - p.fst^2 = 1))
  (h_ecc : (Real.sqrt (a^2 - b^2) / a) * (1 / Real.sqrt (1^2 + 1^2)) = 1) :
  (a = Real.sqrt 2 ∧ b = 1) → ((x y : ℝ), x^2 / 2 + y^2 = 1 :=
by
  sorry

end ellipse_equation_l602_602382


namespace erica_pie_percentage_l602_602664

theorem erica_pie_percentage (a c : ℚ) (ha : a = 1/5) (hc : c = 3/4) : 
  (a + c) * 100 = 95 := 
sorry

end erica_pie_percentage_l602_602664


namespace pentagon_projection_identity_l602_602520

noncomputable def pentagon_projections (M N : ℝ^2) (a : Fin 5 → ℝ^2 → Prop) : Prop :=
  let M_i := λ i : Fin 5, classical.some (exists_unique_proj M (a i))
  let N_i := λ i : Fin 5, classical.some (exists_unique_proj N (a i))
  2 * (∑ i, (M_i i - N_i i)) = 5 * (M - N)

-- Assuming the existence and uniqueness of projections in the plane
axiom exists_unique_proj (P : ℝ^2) (line : ℝ^2 → Prop) : ∃! (Q : ℝ^2), line Q ∧ perp (P - Q) line

-- The main theorem to prove
theorem pentagon_projection_identity (M N : ℝ^2) (a : Fin 5 → ℝ^2 → Prop)
  (h_regular_pentagon : ∀ i, is_regular_pentagon (a i))
  : pentagon_projections M N a :=
sorry

end pentagon_projection_identity_l602_602520


namespace value_of_expression_when_x_is_neg2_l602_602146

theorem value_of_expression_when_x_is_neg2 : 
  ∀ (x : ℤ), x = -2 → (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_when_x_is_neg2_l602_602146


namespace problem_sum_K_l602_602840

-- Define K(x) as the sum of totient functions
def K (x : ℕ) : ℕ := ∑ i in Finset.range x, Nat.totient i

-- Statement of the problem in lean
theorem problem_sum_K :
  (K 100) + (K (100 / 2)) + (K (100 / 3)) + (K (100 / 4)) + (K (100 / 5)) +
  (K (100 / 6)) + (K (100 / 7)) + (K (100 / 8)) + (K (100 / 9)) + (K (100 / 10)) +
  (K (100 / 11)) + (K (100 / 12)) + (K (100 / 13)) + (K (100 / 14)) + (K (100 / 15)) +
  (K (100 / 16)) + (K (100 / 17)) + (K (100 / 18)) + (K (100 / 19)) + (K (100 / 20)) +
  (K (100 / 21)) + (K (100 / 22)) + (K (100 / 23)) + (K (100 / 24)) + (K (100 / 25)) +
  (K (100 / 26)) + (K (100 / 27)) + (K (100 / 28)) + (K (100 / 29)) + (K (100 / 30)) +
  (K (100 / 31)) + (K (100 / 32)) + (K (100 / 33)) + (K (100 / 34)) + (K (100 / 35)) +
  (K (100 / 36)) + (K (100 / 37)) + (K (100 / 38)) + (K (100 / 39)) + (K (100 / 40)) +
  (K (100 / 41)) + (K (100 / 42)) + (K (100 / 43)) + (K (100 / 44)) + (K (100 / 45)) +
  (K (100 / 46)) + (K (100 / 47)) + (K (100 / 48)) + (K (100 / 49)) + (K (100 / 50)) +
  (K (100 / 51)) + (K (100 / 52)) + (K (100 / 53)) + (K (100 / 54)) + (K (100 / 55)) +
  (K (100 / 56)) + (K (100 / 57)) + (K (100 / 58)) + (K (100 / 59)) + (K (100 / 60)) +
  (K (100 / 61)) + (K (100 / 62)) + (K (100 / 63)) + (K (100 / 64)) + (K (100 / 65)) +
  (K (100 / 66)) + (K (100 / 67)) + (K (100 / 68)) + (K (100 / 69)) + (K (100 / 70)) +
  (K (100 / 71)) + (K (100 / 72)) + (K (100 / 73)) + (K (100 / 74)) + (K (100 / 75)) +
  (K (100 / 76)) + (K (100 / 77)) + (K (100 / 78)) + (K (100 / 79)) + (K (100 / 80)) +
  (K (100 / 81)) + (K (100 / 82)) + (K (100 / 83)) + (K (100 / 84)) + (K (100 / 85)) +
  (K (100 / 86)) + (K (100 / 87)) + (K (100 / 88)) + (K (100 / 89)) + (K (100 / 90)) +
  (K (100 / 91)) + (K (100 / 92)) + (K (100 / 93)) + (K (100 / 94)) + (K (100 / 95)) +
  (K (100 / 96)) + (K (100 / 97)) + (K (100 / 98)) + (K (100 / 99)) + (K 1) = 9801 := sorry

end problem_sum_K_l602_602840


namespace triangle_PQR_min_perimeter_l602_602959

theorem triangle_PQR_min_perimeter (PQ PR QR : ℕ) (QJ : ℕ) 
  (hPQ_PR : PQ = PR) (hQJ_10 : QJ = 10) (h_pos_QR : 0 < QR) :
  QR * 2 + PQ * 2 = 96 :=
  sorry

end triangle_PQR_min_perimeter_l602_602959


namespace max_d_l602_602671

/--
Find the maximum value of d in the 7-digit multiples of 22 of the form 55d22ee
where d and e are digits (0-9), the last digit e is even, and the alternating
sum condition for divisibility by 11 is satisfied.
-/
theorem max_d (d e : ℕ) (h₀ : e ∈ {0, 2, 4, 6, 8}) (h₁ : 10 - d - 2 * e ≡ 0 [MOD 11]) :
  d ≤ 6 :=
by sorry

end max_d_l602_602671


namespace cost_small_and_large_puzzle_l602_602583

-- Define the cost of a large puzzle L and the cost equation for large and small puzzles
def cost_large_puzzle : ℤ := 15

def cost_equation (S : ℤ) : Prop := cost_large_puzzle + 3 * S = 39

-- Theorem to prove the total cost of a small puzzle and a large puzzle together
theorem cost_small_and_large_puzzle : ∃ S : ℤ, cost_equation S ∧ (S + cost_large_puzzle = 23) :=
by
  sorry

end cost_small_and_large_puzzle_l602_602583


namespace sin_cos_value_l602_602768

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end sin_cos_value_l602_602768


namespace log_sqrt5_500_sqrt5_eq_log5_2_plus_7_l602_602276

noncomputable def log_sqrt5_500_sqrt5 : Real :=
  Real.logBase (Real.sqrt 5) (500 * Real.sqrt 5)

theorem log_sqrt5_500_sqrt5_eq_log5_2_plus_7 :
  log_sqrt5_500_sqrt5 = Real.logBase 5 2 + 7 := by
  sorry

end log_sqrt5_500_sqrt5_eq_log5_2_plus_7_l602_602276


namespace math_bonanza_2020_2_4_l602_602370

theorem math_bonanza_2020_2_4 :
  let S := (∑ k in Finset.range 1000, ((k + 1) / (k : ℝ)) + ((k : ℝ) / (k + 1))) in
  ∃ (m n : ℕ), nat.coprime m n ∧ S = m / n ∧ m + n = 2004000 :=
by
  let S := (∑ k in Finset.range 1000, ((k + 1) / (k : ℝ)) + ((k : ℝ) / (k + 1)))
  existsi 2002999
  existsi 1001
  split
  · exact nat.coprime_of_dvd ((decimal 2002999).nat_abs.gcd_eq_one (decimal 1001).nat_abs)
  split
  · sorry -- Proof of S = 2002999 / 1001
  · rfl -- Proof of 2002999 + 1001 = 2004000

end math_bonanza_2020_2_4_l602_602370


namespace election_total_votes_l602_602401

theorem election_total_votes (V: ℝ) (valid_votes: ℝ) (candidate_votes: ℝ) (invalid_rate: ℝ) (candidate_rate: ℝ) :
  candidate_rate = 0.75 →
  invalid_rate = 0.15 →
  candidate_votes = 357000 →
  valid_votes = (1 - invalid_rate) * V →
  candidate_votes = candidate_rate * valid_votes →
  V = 560000 :=
by
  intros candidate_rate_eq invalid_rate_eq candidate_votes_eq valid_votes_eq equation
  sorry

end election_total_votes_l602_602401


namespace focus_of_parabola_is_3_0_l602_602082

def parabola_equation (x y : ℝ) : Prop := y^2 = 12 * x

def is_focus (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), parabola_equation x y ∧ p = (3, 0)

theorem focus_of_parabola_is_3_0 :
  ∀ (x y : ℝ), parabola_equation x y → is_focus (3, 0) :=
by
  intros x y h
  unfold is_focus
  use [x, y]
  split
  . exact h
  . refl
  sorry

end focus_of_parabola_is_3_0_l602_602082


namespace peter_satisfied_probability_expected_satisfied_men_l602_602173

variable (numMen : ℕ) (numWomen : ℕ) (totalPeople : ℕ)
variable (peterSatisfiedProb : ℚ) (expectedSatisfiedMen : ℚ)

-- Conditions
def conditions_holds : Prop :=
  numMen = 50 ∧ numWomen = 50 ∧ totalPeople = 100 ∧ peterSatisfiedProb = 25 / 33 ∧ expectedSatisfiedMen = 1250 / 33

-- Prove the probability that Peter Ivanovich is satisfied.
theorem peter_satisfied_probability : conditions_holds → peterSatisfiedProb = 25 / 33 := by
  sorry

-- Prove the expected number of satisfied men.
theorem expected_satisfied_men : conditions_holds → expectedSatisfiedMen = 1250 / 33 := by
  sorry

end peter_satisfied_probability_expected_satisfied_men_l602_602173


namespace inequality_ab_ab2_a_l602_602693

theorem inequality_ab_ab2_a (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_ab_ab2_a_l602_602693


namespace theodore_total_monthly_earning_l602_602528

def total_earnings (stone_statues: Nat) (wooden_statues: Nat) (cost_stone: Nat) (cost_wood: Nat) (tax_rate: Rat) : Rat :=
  let pre_tax_earnings := stone_statues * cost_stone + wooden_statues * cost_wood
  let tax := tax_rate * pre_tax_earnings
  pre_tax_earnings - tax

theorem theodore_total_monthly_earning : total_earnings 10 20 20 5 0.10 = 270 :=
by
  sorry

end theodore_total_monthly_earning_l602_602528


namespace monotonic_f_deriv_nonneg_l602_602779

theorem monotonic_f_deriv_nonneg (k : ℝ) :
  (∀ x : ℝ, (1 / 2) < x → k - 1 / x ≥ 0) ↔ k ≥ 2 :=
by sorry

end monotonic_f_deriv_nonneg_l602_602779


namespace lucille_paint_cans_needed_l602_602846

theorem lucille_paint_cans_needed :
  let wall1_area := 3 * 2
  let wall2_area := 3 * 2
  let wall3_area := 5 * 2
  let wall4_area := 4 * 2
  let total_area := wall1_area + wall2_area + wall3_area + wall4_area
  let coverage_per_can := 2
  let cans_needed := total_area / coverage_per_can
  cans_needed = 15 := 
by 
  sorry

end lucille_paint_cans_needed_l602_602846


namespace cyclic_quad_diagonal_ratio_l602_602017

def are_cyclic (A B C D : Point) : Prop := -- Definition that points A, B, C, D are on a circle (cyclic quadrilateral)
  -- Use a placeholder for now as precise definitions aren't provided
  sorry

/-- Theorem: Given four points A, B, C, and D on a circle with respective segment lengths,
    and the diagonals AC and BD intersecting at P, the ratio AP/CP is 2/5. -/
theorem cyclic_quad_diagonal_ratio (A B C D P : Point)
  (h_cyclic: are_cyclic A B C D)
  (h_ab: distance A B = 3)
  (h_bc: distance B C = 5)
  (h_cd: distance C D = 6)
  (h_da: distance D A = 4)
  (h_intersect: are_intersecting_diagonals A C B D P) :
  (distance A P) / (distance C P) = 2 / 5 :=
sorry

end cyclic_quad_diagonal_ratio_l602_602017


namespace volume_of_pyramid_l602_602798

theorem volume_of_pyramid
  (AB BC CG : ℝ)
  (hAB : AB = 4)
  (hBC : BC = 2)
  (hCG : CG = 3)
  (P : ℝ × ℝ × ℝ)
  (hP : P = (2, 0, 1.5)) :
  let base_area := AB * CG,
      height := 1.5 
  in
  (1 / 3) * base_area * height = 6 :=
by
  -- Define base_area and height for clarity
  let base_area := AB * CG
  let height := 1.5
  -- Calculate volume
  have volume : (1 / 3) * base_area * height = 6,
  {
    calc (1 / 3) * base_area * height 
        = (1 / 3) * (4 * 3) * 1.5 : by rw [hAB, hCG]
    ... = (1 / 3) * 12 * 1.5 : by ring
    ... = 6 : by norm_num,
  }
  exact volume

end volume_of_pyramid_l602_602798


namespace tomorrow_is_saturday_l602_602873

noncomputable def day_before_yesterday : string := "Wednesday"
noncomputable def today : string := "Friday"
noncomputable def tomorrow : string := "Saturday"

theorem tomorrow_is_saturday (dby : string) (tod : string) (tom : string) 
  (h1 : dby = "Wednesday") (h2 : tod = "Friday") (h3 : tom = "Saturday")
  (h_cond : "Monday" = dby + 5) : 
  tom = "Saturday" := 
sorry

end tomorrow_is_saturday_l602_602873


namespace susy_initial_followers_l602_602901

theorem susy_initial_followers 
  (initial_sarah : ℕ := 50)
  (sarah_gain : ℕ)
  (susy_gain1 susy_gain2 susy_gain3 : ℕ := 40)
  (total_followers_max : ℕ := 180)
  (gain2_eq_half_gain1 : susy_gain2 = susy_gain1 / 2)
  (gain3_eq_half_gain2 : susy_gain3 = susy_gain2 / 2) :
  let initial_susy := 110 in
  susy_gain1 + susy_gain2 + susy_gain3 = 70 →
  total_followers_max = initial_susy + 70 →
  ∃ initial_susy : ℕ, initial_susy = 110 :=
by
  intros h1 h2
  use 110
  sorry

end susy_initial_followers_l602_602901


namespace fraction_simplification_l602_602543

theorem fraction_simplification : (145^2 - 121^2) / 24 = 266 := by
  sorry

end fraction_simplification_l602_602543


namespace normal_line_at_point_l602_602565

noncomputable def curve (x : ℝ) : ℝ := (4 * x - x ^ 2) / 4

theorem normal_line_at_point (x0 : ℝ) (h : x0 = 2) :
  ∃ (L : ℝ → ℝ), ∀ (x : ℝ), L x = (2 : ℝ) :=
by
  sorry

end normal_line_at_point_l602_602565


namespace find_roots_calculate_abcdd_sum_l602_602289

noncomputable def rootForm (x : ℝ) : Prop :=
  ∃ a b c d : ℝ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ ∀ p : ℝ, p ∣ d → ¬ ∃ q : ℝ, p = q * q) ∧
    (x = -a + Real.sqrt(b + c * Real.sqrt(d)) ∨ x = -a - Real.sqrt(b + c * Real.sqrt(d)) ∨ 
     x = -a + Real.sqrt(b - c * Real.sqrt(d)) ∨ x = -a - Real.sqrt(b - c * Real.sqrt(d)))

theorem find_roots : ∀ x : ℝ, 
  (1 / x + 1 / (x + 4) - 1 / (x + 6) - 1 / (x + 10) - 1 / (x + 12) + 1 / (x + 16) = 0) ↔ 
  (rootForm x) :=
sorry

theorem calculate_abcdd_sum :
  (∃ a b c d : ℝ, rootForm (-a + Real.sqrt(b + c * Real.sqrt(d))) ∧
   rootForm (-a - Real.sqrt(b + c * Real.sqrt(d))) ∧
   rootForm (-a + Real.sqrt(b - c * Real.sqrt(d))) ∧
   rootForm (-a - Real.sqrt(b - c * Real.sqrt(d))) ∧
   (a + b + c + d = 102)) :=
sorry

end find_roots_calculate_abcdd_sum_l602_602289


namespace square_perimeter_l602_602607

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end square_perimeter_l602_602607


namespace square_perimeter_l602_602595

noncomputable def side_length_of_square_with_area (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def perimeter_of_square_with_side (side : ℝ) : ℝ :=
  4 * side

theorem square_perimeter {area : ℝ} (h_area : area = 625) :
  perimeter_of_square_with_side (side_length_of_square_with_area area) = 100 :=
by
  have h_side_length : side_length_of_square_with_area area = 25 := by
    rw [side_length_of_square_with_area, real.sqrt, h_area]
    norm_num
  rw [perimeter_of_square_with_side, h_side_length]
  norm_num
  sorry

end square_perimeter_l602_602595


namespace ratio_of_girls_more_than_boys_l602_602050

theorem ratio_of_girls_more_than_boys 
  (B : ℕ := 50) 
  (P : ℕ := 123) 
  (driver_assistant_teacher := 3) 
  (h : P = driver_assistant_teacher + B + (P - driver_assistant_teacher - B)) : 
  (P - driver_assistant_teacher - B) - B = 21 → 
  (P - driver_assistant_teacher - B) % B = 21 / 50 := 
sorry

end ratio_of_girls_more_than_boys_l602_602050


namespace wall_passing_pattern_l602_602813

theorem wall_passing_pattern (n : ℕ) : (8^2 - 1 = n) → (8 * sqrt (8 / n) = sqrt (8 * 8 / n)) :=
by
  intro h
  rw h
  sorry

end wall_passing_pattern_l602_602813


namespace domain_log_function_l602_602506

theorem domain_log_function : 
  (∀ x, (∃ y : ℝ, f x = log y) ↔ -x + 4 > 0) ↔ set.univ \ {4} = set.Iio 4 :=
by sorry

end domain_log_function_l602_602506


namespace two_digit_number_is_91_l602_602209

/-- A positive two-digit number is odd and is a multiple of 13.
    The product of its digits is a perfect square.
    What is this two-digit number? -/
theorem two_digit_number_is_91 (M : ℕ) (h1 : M > 9) (h2 : M < 100) (h3 : M % 2 = 1) (h4 : M % 13 = 0) (h5 : ∃ n : ℕ, n * n = (M / 10) * (M % 10)) :
  M = 91 :=
sorry

end two_digit_number_is_91_l602_602209


namespace rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l602_602404

-- Define the problem conditions: number of ways to place two same-color rooks that do not attack each other.
def num_ways_rooks : ℕ := 1568
theorem rooks_non_attacking : ∃ (n : ℕ), n = num_ways_rooks := by
  sorry

-- Define the problem conditions: number of ways to place two same-color kings that do not attack each other.
def num_ways_kings : ℕ := 1806
theorem kings_non_attacking : ∃ (n : ℕ), n = num_ways_kings := by
  sorry

-- Define the problem conditions: number of ways to place two same-color bishops that do not attack each other.
def num_ways_bishops : ℕ := 1736
theorem bishops_non_attacking : ∃ (n : ℕ), n = num_ways_bishops := by
  sorry

-- Define the problem conditions: number of ways to place two same-color knights that do not attack each other.
def num_ways_knights : ℕ := 1848
theorem knights_non_attacking : ∃ (n : ℕ), n = num_ways_knights := by
  sorry

-- Define the problem conditions: number of ways to place two same-color queens that do not attack each other.
def num_ways_queens : ℕ := 1288
theorem queens_non_attacking : ∃ (n : ℕ), n = num_ways_queens := by
  sorry

end rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l602_602404


namespace evaluate_M_l602_602667

noncomputable def M : ℝ := 
  (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)

theorem evaluate_M : M = (1 + Real.sqrt 3 + Real.sqrt 5 + 3 * Real.sqrt 2) / 3 :=
by
  sorry

end evaluate_M_l602_602667


namespace two_digit_number_satisfies_conditions_l602_602212

theorem two_digit_number_satisfies_conditions :
  ∃ N : ℕ, (N > 0) ∧ (N < 100) ∧ (N % 2 = 1) ∧ (N % 13 = 0) ∧ (∃ a b : ℕ, N = 10 * a + b ∧ (a * b) = (k : ℕ) * k) ∧ (N = 91) :=
by
  sorry

end two_digit_number_satisfies_conditions_l602_602212


namespace pentagon_diagonal_intersection_l602_602445

theorem pentagon_diagonal_intersection
  (ABCDE : ConvexPentagon)
  (F : Point)
  (h_AF_BD_intersect : Intersect AC BD F)
  (S : ℝ)
  (h_S : S = ∠AFB + ∠DFE)
  (S' : ℝ)
  (h_S' : S' = ∠BAC + ∠CAD)
  : (S / S') = 2 :=
by
  sorry

end pentagon_diagonal_intersection_l602_602445


namespace sum_binom_equality_l602_602640

theorem sum_binom_equality : 
  (\sum k in Finset.range 51, (-1)^k * Nat.choose 101 (2 * k)) = -2^50 := by
  sorry

end sum_binom_equality_l602_602640


namespace line_through_point_l602_602090

theorem line_through_point (b : ℚ) :
  (∃ x y,
    (x = 3) ∧ (y = -7) ∧ (b * x + (b - 1) * y = b + 3))
  → (b = 4 / 5) :=
begin
  sorry
end
 
end line_through_point_l602_602090


namespace zeros_of_f_l602_602947

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 2

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 :=
by {
  sorry
}

end zeros_of_f_l602_602947


namespace value_of_q_l602_602427

theorem value_of_q (m p q a b : ℝ) 
  (h₁ : a * b = 6) 
  (h₂ : (a + 1 / b) * (b + 1 / a) = q): 
  q = 49 / 6 := 
sorry

end value_of_q_l602_602427


namespace range_of_function_l602_602099

noncomputable def f (x : ℝ) : ℝ := (2 * Real.exp x - 1) / (Real.exp x + 2)

theorem range_of_function :
  set.range f = set.Ioo (-1/2 : ℝ) 2 :=
by
  sorry

end range_of_function_l602_602099


namespace largest_prime_divisor_test_l602_602956

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def max_in_range : ℕ := 550

def sqrt_max : ℝ := Real.sqrt max_in_range

noncomputable def primes_upto_sqrt_max : List ℕ :=
List.filter is_prime (List.range (Nat.floor sqrt_max + 1))

noncomputable def largest_prime_upto_sqrt_max : ℕ :=
List.maximum primes_upto_sqrt_max

theorem largest_prime_divisor_test :
  largest_prime_upto_sqrt_max = 23 :=
by
  sorry

end largest_prime_divisor_test_l602_602956


namespace general_term_formula_l602_602942

def sequence_sum (n : ℕ) : ℕ := 3 * n^2 - 2 * n

def general_term (n : ℕ) : ℕ := if n = 0 then 0 else 6 * n - 5

theorem general_term_formula (n : ℕ) (h : n > 0) :
  general_term n = sequence_sum n - sequence_sum (n - 1) := by
  sorry

end general_term_formula_l602_602942


namespace square_perimeter_l602_602613

theorem square_perimeter (area : ℝ) (h : area = 625) : 
  let s := Real.sqrt area in
  (4 * s) = 100 :=
by
  let s := Real.sqrt area
  have hs : s = 25 := by sorry
  calc
    (4 * s) = 4 * 25 : by rw hs
          ... = 100   : by norm_num

end square_perimeter_l602_602613


namespace Meadow_sells_each_diaper_for_5_l602_602452

-- Define the conditions as constants
def boxes_per_week := 30
def packs_per_box := 40
def diapers_per_pack := 160
def total_revenue := 960000

-- Calculate total packs and total diapers
def total_packs := boxes_per_week * packs_per_box
def total_diapers := total_packs * diapers_per_pack

-- The target price per diaper
def price_per_diaper := total_revenue / total_diapers

-- Statement of the proof theorem
theorem Meadow_sells_each_diaper_for_5 : price_per_diaper = 5 := by
  sorry

end Meadow_sells_each_diaper_for_5_l602_602452


namespace famous_teacher_positive_correlation_l602_602371

theorem famous_teacher_positive_correlation (famous_teacher_produces_outstanding_students: Prop) :
  famous_teacher_produces_outstanding_students → "Positive correlation between the fame of the teacher and the excellence of the students" := 
sorry

end famous_teacher_positive_correlation_l602_602371


namespace number_solution_l602_602459

theorem number_solution (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
by
  -- The proof is omitted according to the instructions
  sorry

end number_solution_l602_602459


namespace simplify_trig_identity_l602_602480

theorem simplify_trig_identity (x y : ℝ) : 
  sin (x + y) * cos y - cos (x + y) * sin y = sin x :=
by 
  sorry

end simplify_trig_identity_l602_602480


namespace number_of_teams_l602_602071

theorem number_of_teams (x : ℕ) (h : 21 = x * (x - 1) / 2) : 
  ∃ x : ℕ, 21 = x * (x - 1) / 2 :=
by
  use x
  exact h

end number_of_teams_l602_602071


namespace coefficient_ab2cm3_l602_602326

noncomputable def integral_sinx : ℝ :=
  3 * (∫ x in 0..Real.pi, Real.sin x)

theorem coefficient_ab2cm3 (a b c : ℝ) :
  let m := integral_sinx in
  m = 6 → 
  ∑ i in Range (m+1), 
    (binom (nat.pred m) i * a ^ (m - i) * (2 * b - 3 * c) ^ i = -6480 * (ab^2)*(c ^ (m-3)) :
  sorry

end coefficient_ab2cm3_l602_602326


namespace last_number_is_four_l602_602880

theorem last_number_is_four (a b c d e last_number : ℕ) (h_counts : a = 6 ∧ b = 12 ∧ c = 1 ∧ d = 12 ∧ e = 7)
    (h_mean : (a + b + c + d + e + last_number) / 6 = 7) : last_number = 4 := 
sorry

end last_number_is_four_l602_602880


namespace part1_part2_l602_602845

def p (a : ℝ) : Prop := 3 ^ a ≤ 9
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 3 * (3 - a) * x + 9 ≥ 0
def r (a : ℝ) : Prop := p a ∧ q a
def t (a : ℝ) (m : ℕ) : Prop := a < m ∨ a > m + 1 / 2
def not_t (a : ℝ) (m : ℕ) : Prop := m ≤ a ∧ a ≤ m + 1 / 2

theorem part1 (a : ℝ) : r a ↔ (1 ≤ a ∧ a ≤ 2) := 
by sorry -- Proof not required

theorem part2 (m : ℕ) (h : ∀ a : ℝ, r a → not_t a m) : m = 1 := 
by sorry -- Proof not required

end part1_part2_l602_602845


namespace speed_against_current_l602_602155

theorem speed_against_current (V_m V_c : ℝ) (h1 : V_m + V_c = 20) (h2 : V_c = 1) :
  V_m - V_c = 18 :=
by
  sorry

end speed_against_current_l602_602155


namespace topsoil_cost_l602_602958

theorem topsoil_cost (cost_per_cubic_foot : ℕ) (cubic_yard_to_cubic_foot : ℕ) (volume_in_cubic_yards : ℕ) :
  cost_per_cubic_foot = 8 →
  cubic_yard_to_cubic_foot = 27 →
  volume_in_cubic_yards = 3 →
  volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 648 :=
by
  intros h1 h2 h3
  sorry

end topsoil_cost_l602_602958


namespace roots_of_equation_l602_602933

theorem roots_of_equation (x : ℝ) : ((x - 5) ^ 2 = 2 * (x - 5)) ↔ (x = 5 ∨ x = 7) := by
sorry

end roots_of_equation_l602_602933


namespace cannot_determine_start_month_l602_602461

variables (preparedUntilAugust3rd : Prop) (preparedForAugust : Prop)

theorem cannot_determine_start_month (h1 : preparedUntilAugust3rd) (h2 : preparedForAugust) : 
  ∃ m, true → ¬ (∃ month, true) := sorry

end cannot_determine_start_month_l602_602461


namespace sin_cos_identity_l602_602764

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602764


namespace geometric_sequence_sum_ratio_l602_602447

-- The geometric sequence sum conditions
def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

-- The condition S_{10} : S_{5} = 1 : 2
def ratio_condition (a : ℕ → ℝ) : Prop :=
  geometric_sum a 10 / geometric_sum a 5 = 1 / 2

-- Prove that S_{15} : S_{5} = 3 : 4 given the ratio_condition
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (h : ratio_condition a) : 
  geometric_sum a 15 / geometric_sum a 5 = 3 / 4 :=
sorry

end geometric_sequence_sum_ratio_l602_602447


namespace erica_pie_fraction_as_percentage_l602_602666

theorem erica_pie_fraction_as_percentage (apple_pie_fraction : ℚ) (cherry_pie_fraction : ℚ) 
  (h1 : apple_pie_fraction = 1 / 5) 
  (h2 : cherry_pie_fraction = 3 / 4) 
  (common_denominator : ℚ := 20) : 
  (apple_pie_fraction + cherry_pie_fraction) * 100 = 95 :=
by
  sorry

end erica_pie_fraction_as_percentage_l602_602666


namespace total_shingles_needed_l602_602918

-- Defining the dimensions of the house and the porch
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_length : ℝ := 6
def porch_width : ℝ := 4.5

-- The goal is to prove that the total area of the shingles needed is 232 square feet
theorem total_shingles_needed :
  (house_length * house_width) + (porch_length * porch_width) = 232 := by
  sorry

end total_shingles_needed_l602_602918


namespace combined_height_of_trees_is_correct_l602_602419

noncomputable def original_height_of_trees 
  (h1_current : ℝ) (h1_growth_rate : ℝ)
  (h2_current : ℝ) (h2_growth_rate : ℝ)
  (h3_current : ℝ) (h3_growth_rate : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  let h1 := h1_current / (1 + h1_growth_rate)
  let h2 := h2_current / (1 + h2_growth_rate)
  let h3 := h3_current / (1 + h3_growth_rate)
  (h1 + h2 + h3) / conversion_rate

theorem combined_height_of_trees_is_correct :
  original_height_of_trees 240 0.70 300 0.50 180 0.60 12 = 37.81 :=
by
  sorry

end combined_height_of_trees_is_correct_l602_602419


namespace least_common_multiple_9_12_15_l602_602974

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l602_602974


namespace remaining_days_temperature_l602_602911

theorem remaining_days_temperature (avg_temp : ℕ) (d1 d2 d3 d4 d5 : ℕ) :
  avg_temp = 60 →
  d1 = 40 →
  d2 = 40 →
  d3 = 40 →
  d4 = 80 →
  d5 = 80 →
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  total_temp - known_temp = 140 := 
by
  intros _ _ _ _ _ _
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  sorry

end remaining_days_temperature_l602_602911


namespace find_birthday_stickers_l602_602847

variable (initial_stickers : ℕ) (bought_stickers : ℕ) (given_away_stickers : ℕ) (used_stickers : ℕ) (remaining_stickers : ℕ) (birthday_stickers : ℕ)

axiom initial_stickers_value : initial_stickers = 20
axiom bought_stickers_value : bought_stickers = 12
axiom given_away_stickers_value : given_away_stickers = 5
axiom used_stickers_value : used_stickers = 8
axiom remaining_stickers_value : remaining_stickers = 39

theorem find_birthday_stickers : birthday_stickers = 20 :=
by
  have total_initial := initial_stickers + bought_stickers -- 32
  have total_stickers_before_giving_away := total_initial + birthday_stickers
  have total_given_away := given_away_stickers + used_stickers -- 13
  have equation := total_stickers_before_giving_away - total_given_away = remaining_stickers
  rw [initial_stickers_value, bought_stickers_value, given_away_stickers_value, used_stickers_value, remaining_stickers_value] at equation
  calc
    birthday_stickers = 20 := sorry

end find_birthday_stickers_l602_602847


namespace concyclic_AEKF_l602_602631

noncomputable def point : Type := sorry

variables {A B C D E F P K : point}
variables {AC AB : point → point → Prop}
variables (inside_triangle : point → point → point → point → Prop) 
variables (circumcircle : point → point → point → point → Prop) 
variables (symmetric_about : point → point → point → Prop)

axioms
  (h_in_triangle : inside_triangle D A B C)
  (h_E_on_AC : AC A E)
  (h_F_on_AB : AB A F)
  (h_DE_eq_DC : DE = DC)
  (h_DF_eq_DB : DF = DB)
  (h_AP_perp_AD : AP ⟂ AD ∧ circumcircle A B C P)
  (h_K_symmetric : symmetric_about K P A)

theorem concyclic_AEKF :
  concyclic A E K F :=
begin
  sorry
end

end concyclic_AEKF_l602_602631


namespace maple_tree_total_l602_602948

-- Conditions
def initial_maple_trees : ℕ := 53
def trees_planted_today : ℕ := 11

-- Theorem to prove the result
theorem maple_tree_total : initial_maple_trees + trees_planted_today = 64 := by
  sorry

end maple_tree_total_l602_602948


namespace assistant_professors_charts_l602_602245

theorem assistant_professors_charts (A B C : ℕ) (h1 : 2 * A + B = 10) (h2 : A + B * C = 11) (h3 : A + B = 7) : C = 2 :=
by
  sorry

end assistant_professors_charts_l602_602245


namespace hypotenuse_length_l602_602900

noncomputable def side_lengths_to_hypotenuse (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_length 
  (AB BC : ℝ) 
  (h1 : Real.sqrt (AB * BC) = 8) 
  (h2 : (1 / 2) * AB * BC = 48) :
  side_lengths_to_hypotenuse AB BC = 4 * Real.sqrt 13 :=
by
  sorry

end hypotenuse_length_l602_602900


namespace arithmetic_sequence_problem_l602_602899

-- Define the arithmetic sequence and related sum functions
def a_n (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

def S (a1 d : ℤ) (n : ℕ) : ℤ :=
  (a1 + a_n a1 d n) * n / 2

-- Problem statement: proving a_5 = -1 given the conditions
theorem arithmetic_sequence_problem :
  (∃ (a1 d : ℕ), S a1 d 2 = S a1 d 6 ∧ a_n a1 d 4 = 1) → a_n a1 d 5 = -1 :=
by
  -- Assume the statement and then skip the proof
  sorry

end arithmetic_sequence_problem_l602_602899


namespace inverse_equilateral_is_equilateral_l602_602088

theorem inverse_equilateral_is_equilateral (T : Type) [triangle T] :
  (∀ (t : T), (∃ a b c : ℝ, a = b ∧ b = c ∧ c = a) → (∠A t = ∠B t ∧ ∠B t = ∠C t ∧ ∠C t = ∠A t)) →
  (∀ (t : T), (∠A t = ∠B t ∧ ∠B t = ∠C t ∧ ∠C t = ∠A t) → (∃ a b c : ℝ, a = b ∧ b = c ∧ c = a)) :=
sorry

end inverse_equilateral_is_equilateral_l602_602088


namespace ratio_of_areas_l602_602405

variable {A B C P Q R : Type*}
variable [Collinear A B R] [Collinear B C P] [Collinear C A Q]
variable {L K : ℝ}
variable {t : ℝ} (ht : 0 < t ∧ t < 1)
variable (hP: ∃ (t : ℝ), Ratios BC P = t / (1 - t))
variable (hQ: ∃ (t : ℝ), Ratios CA Q = t / (1 - t))
variable (hR: ∃ (t : ℝ), Ratios AB R = t / (1 - t))
variable (hArea : ∃ K, Area (ΔAPB) (ΔBQC) (ΔCQA) = K)
variable (hTriangle : ∃ L, Area (ΔABC) = L)

theorem ratio_of_areas (ht : t ∈ Ioo 0 1) (hArea: Area (ΔAPB) (ΔBQC) (ΔCQA) = K) 
      (hTriangle: Area (ΔABC) = L) : 
  K / L = t^2 - t + 1 :=
sorry

end ratio_of_areas_l602_602405


namespace find_birthday_stickers_l602_602848

variable (initial_stickers : ℕ) (bought_stickers : ℕ) (given_away_stickers : ℕ) (used_stickers : ℕ) (remaining_stickers : ℕ) (birthday_stickers : ℕ)

axiom initial_stickers_value : initial_stickers = 20
axiom bought_stickers_value : bought_stickers = 12
axiom given_away_stickers_value : given_away_stickers = 5
axiom used_stickers_value : used_stickers = 8
axiom remaining_stickers_value : remaining_stickers = 39

theorem find_birthday_stickers : birthday_stickers = 20 :=
by
  have total_initial := initial_stickers + bought_stickers -- 32
  have total_stickers_before_giving_away := total_initial + birthday_stickers
  have total_given_away := given_away_stickers + used_stickers -- 13
  have equation := total_stickers_before_giving_away - total_given_away = remaining_stickers
  rw [initial_stickers_value, bought_stickers_value, given_away_stickers_value, used_stickers_value, remaining_stickers_value] at equation
  calc
    birthday_stickers = 20 := sorry

end find_birthday_stickers_l602_602848


namespace tank_to_bucket_ratio_l602_602218

-- Define the parameters and conditions
variables (T B : ℝ)
-- Conditions
variables (H1 : T > 0) (H2 : B > 0)
variables (H3 : \(\frac{3}{5} * T\) = initial amount of water in the tank)
variables (H4 : \(\frac{1}{4} * B\) = initial amount of water in the bucket)
variables (H5 : \(\frac{1}{2} * B\) = after transferring, bucket is half full)
variables (H6 : \(\frac{2}{3} * T\) = remaining water in tank)

-- Theorem statement
theorem tank_to_bucket_ratio :
  \(\frac{T}{B} = \frac{15}{4}\)
  sorry

end tank_to_bucket_ratio_l602_602218


namespace birthday_stickers_l602_602850

def initial_stickers := 20
def bought_stickers := 12
def given_away_stickers := 5
def used_stickers := 8
def stickers_left := 39

theorem birthday_stickers : 
  let total_stickers_before_giving := stickers_left + given_away_stickers + used_stickers in
  let total_stickers_after_buying := initial_stickers + bought_stickers in
  total_stickers_before_giving - total_stickers_after_buying = 20 :=
by
  sorry

end birthday_stickers_l602_602850


namespace cube_not_divisible_into_distinct_smaller_cubes_l602_602888

theorem cube_not_divisible_into_distinct_smaller_cubes :
  ¬ ∃ (cubes : set ℝ³), (∀ c ∈ cubes, is_cube c) ∧ (⋃₀ cubes = original_cube) ∧ (∀ p q ∈ cubes, p ≠ q → p ≠ q) :=
sorry

end cube_not_divisible_into_distinct_smaller_cubes_l602_602888


namespace number_of_squares_l602_602073

theorem number_of_squares (total_streetlights squares_streetlights unused_streetlights : ℕ) 
  (h1 : total_streetlights = 200) 
  (h2 : squares_streetlights = 12) 
  (h3 : unused_streetlights = 20) : 
  (∃ S : ℕ, total_streetlights = squares_streetlights * S + unused_streetlights ∧ S = 15) :=
by
  sorry

end number_of_squares_l602_602073


namespace erica_pie_fraction_as_percentage_l602_602665

theorem erica_pie_fraction_as_percentage (apple_pie_fraction : ℚ) (cherry_pie_fraction : ℚ) 
  (h1 : apple_pie_fraction = 1 / 5) 
  (h2 : cherry_pie_fraction = 3 / 4) 
  (common_denominator : ℚ := 20) : 
  (apple_pie_fraction + cherry_pie_fraction) * 100 = 95 :=
by
  sorry

end erica_pie_fraction_as_percentage_l602_602665


namespace peter_satisfied_probability_expected_satisfied_men_l602_602172

variable (numMen : ℕ) (numWomen : ℕ) (totalPeople : ℕ)
variable (peterSatisfiedProb : ℚ) (expectedSatisfiedMen : ℚ)

-- Conditions
def conditions_holds : Prop :=
  numMen = 50 ∧ numWomen = 50 ∧ totalPeople = 100 ∧ peterSatisfiedProb = 25 / 33 ∧ expectedSatisfiedMen = 1250 / 33

-- Prove the probability that Peter Ivanovich is satisfied.
theorem peter_satisfied_probability : conditions_holds → peterSatisfiedProb = 25 / 33 := by
  sorry

-- Prove the expected number of satisfied men.
theorem expected_satisfied_men : conditions_holds → expectedSatisfiedMen = 1250 / 33 := by
  sorry

end peter_satisfied_probability_expected_satisfied_men_l602_602172


namespace find_BE_l602_602824

noncomputable def triangle_AB (A B C : Type) [MetricSpace A] (a b c : ℝ) (hab : dist A B = a) (hbc : dist B C = b) (hac : dist A C = c) :=
  dist A B = a ∧ dist B C = b ∧ dist A C = c

noncomputable def point_D_on_BC (B C D : Type) [MetricSpace B] (c d : ℝ) (hcd : dist C D = c) :=
  dist C D = c

noncomputable def point_E_midpoint_BC (B C E : Type) [MetricSpace B] [MetricSpace C] [MetricSpace E] :=
  dist B E = dist E C ∧ dist B C = 2 * dist B E

theorem find_BE
  (A B C D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (hABC : triangle_AB A B C 15 17 16)
  (hD_on_BC : point_D_on_BC B C D 7)
  (hE_midpoint_BC : point_E_midpoint_BC B C E) :
  dist B E = 8.5 := by
    sorry

end find_BE_l602_602824


namespace part_a_part_b_l602_602162

-- Define the setup
def total_people := 100
def total_men := 50
def total_women := 50

-- Peter Ivanovich's position and neighbor relations
def pi_satisfied_prob : ℚ := 25 / 33

-- Expected number of satisfied men
def expected_satisfied_men : ℚ := 1250 / 33

-- Lean statements for the problems

-- Part (a): Prove Peter Ivanovich's satisfaction probability
theorem part_a (total_people = 100) (total_men = 50) (total_women = 50) : 
  pi_satisfied_prob = 25 / 33 := 
sorry

-- Part (b): Expected number of satisfied men
theorem part_b (total_people = 100) (total_men = 50) (total_women = 50) : 
  expected_satisfied_men = 1250 / 33 := 
sorry

end part_a_part_b_l602_602162


namespace tangent_line_b_value_l602_602718

theorem tangent_line_b_value (b : ℝ) : 
  (∃ pt : ℝ × ℝ, (pt.1)^2 + (pt.2)^2 = 25 ∧ pt.1 - pt.2 + b = 0)
  ↔ b = 5 * Real.sqrt 2 ∨ b = -5 * Real.sqrt 2 :=
by
  sorry

end tangent_line_b_value_l602_602718


namespace distribute_pure_alcohol_to_achieve_50_percent_l602_602554

theorem distribute_pure_alcohol_to_achieve_50_percent :
  ∀ (A B C : ℝ),
  (A = 8 * 0.25) → 
  (B = 10 * 0.40) → 
  (C = 6 * 0.30) →
  (2 + A) / 10 = 0.50 ∧ 
  (1 + B) / 11 = 0.50 ∧ 
  (1.2 + C) / 7.2 = 0.50 :=
begin
  sorry
end

end distribute_pure_alcohol_to_achieve_50_percent_l602_602554


namespace exists_m_not_rel_prime_l602_602000

theorem exists_m_not_rel_prime :
  ∃ m : ℤ, let a := 100 + 101 * m, b := 101 - 100 * m in Int.gcd a b ≠ 1 :=
by 
  sorry

end exists_m_not_rel_prime_l602_602000


namespace tom_games_value_l602_602117

/--
  Tom bought his games for $200. The value tripled and he sold 40% of them.
  Prove that Tom sold the games for $240.
-/
theorem tom_games_value : (price: ℤ) -> (tripled_value: ℤ) -> (percentage_sold: ℤ) -> (sold_value: ℤ) ->
  price = 200 -> tripled_value = 3 * price -> percentage_sold = 40 -> sold_value = tripled_value * percentage_sold / 100 -> sold_value = 240 :=
by
  assume price tripled_value percentage_sold sold_value
  assume h1: price = 200
  assume h2: tripled_value = 3 * price
  assume h3: percentage_sold = 40
  assume h4: sold_value = tripled_value * percentage_sold / 100
  sorry

end tom_games_value_l602_602117


namespace factorize_expression_l602_602154

variable (a b c d : ℝ)

theorem factorize_expression : 
  (a^2 - b^2) * c^2 - (a^2 - b^2) * d^2 = (a + b) * (a - b) * (c + d) * (c - d) := 
by 
  sorry

end factorize_expression_l602_602154


namespace lcm_of_9_12_15_is_180_l602_602987

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l602_602987


namespace prob_all_three_siblings_selected_l602_602533

noncomputable def prob_selection_X : ℚ := 1/3
noncomputable def prob_selection_Y : ℚ := 2/5
noncomputable def prob_selection_Z : ℚ := 1/4

def at_most_two_siblings_selected (X_selected Y_selected Z_selected : Bool) : Prop :=
  (X_selected.to_nat + Y_selected.to_nat + Z_selected.to_nat) ≤ 2

theorem prob_all_three_siblings_selected :
  ∀ (X_selected Y_selected Z_selected : Bool),
    X_selected ∧ Y_selected ∧ Z_selected → at_most_two_siblings_selected X_selected Y_selected Z_selected → false :=
by
  intros X_selected Y_selected Z_selected h1 h2
  sorry

end prob_all_three_siblings_selected_l602_602533


namespace jimin_and_seokjin_total_l602_602546

def Jimin_coins := (5 * 100) + (1 * 50)
def Seokjin_coins := (2 * 100) + (7 * 10)
def total_coins := Jimin_coins + Seokjin_coins

theorem jimin_and_seokjin_total : total_coins = 820 :=
by
  sorry

end jimin_and_seokjin_total_l602_602546


namespace problem_part_a_problem_part_b_l602_602168

noncomputable def probability_peter_satisfied : ℚ := 25 / 33

noncomputable def expected_number_satisfied_men : ℚ := 1250 / 33

theorem problem_part_a (total_men total_women : ℕ) (h1 : total_men = 50) (h2 : total_women = 50):
  let peter_satisfied := probability_peter_satisfied in
  let prob := λ m w, 1 - ((m / (m + w - 1)) * ((m - 1) / (m + w - 2))) in
  peter_satisfied = prob (total_men - 1) total_women := 
by {
  dsimp [peter_satisfied, prob],
  rw [h1, h2],
  unfold_coes,
  norm_num,
  sorry
}

theorem problem_part_b (total_men total_women : ℕ) (h1 : total_men = 50) (h2 : total_women = 50):
  let satisfied_probability := probability_peter_satisfied in
  let expected_satisfied_men := expected_number_satisfied_men in
  expected_satisfied_men = total_men * satisfied_probability := 
by {
  dsimp [satisfied_probability, expected_satisfied_men],
  rw [h1, h2],
  unfold_coes,
  norm_num,
  sorry
}

end problem_part_a_problem_part_b_l602_602168


namespace student_tickets_count_l602_602685

-- Define the relevant conditions
def total_tickets := 821
def total_revenue := 1933
def student_ticket_price := 2
def nonstudent_ticket_price := 3

-- Define the variables representing the number of tickets sold
variable (S N : ℕ)

-- State the problem as a Lean theorem
theorem student_tickets_count :
  (S + N = total_tickets) ∧ (student_ticket_price * S + nonstudent_ticket_price * N = total_revenue) → S = 530 :=
by
  intros h
  cases h with h1 h2
  -- Proof part to be filled by the user
  sorry

end student_tickets_count_l602_602685


namespace copper_alloy_proof_l602_602529

variable (x p : ℝ)

theorem copper_alloy_proof
  (copper_content1 copper_content2 weight1 weight2 total_weight : ℝ)
  (h1 : weight1 = 3)
  (h2 : copper_content1 = 0.4)
  (h3 : weight2 = 7)
  (h4 : copper_content2 = 0.3)
  (h5 : total_weight = 8)
  (h6 : 1 ≤ x ∧ x ≤ 3)
  (h7 : p = 100 * (copper_content1 * x + copper_content2 * (total_weight - x)) / total_weight) :
  31.25 ≤ p ∧ p ≤ 33.75 := 
  sorry

end copper_alloy_proof_l602_602529


namespace train_speed_l602_602621

-- Defining the known conditions
def length_of_train : ℝ := 500 -- meters
def time_to_cross : ℝ := 10 -- seconds
def speed_of_man_kmh : ℝ := 5 -- km/h

-- Convert man's speed from km/h to m/s
def speed_of_man_ms : ℝ := (speed_of_man_kmh * 1000) / 3600

-- Define the relative speed
def relative_speed (speed_of_train : ℝ) : ℝ := speed_of_train + speed_of_man_ms

-- Define the equation for time taken to cross
def time_eq (speed_of_train : ℝ) : Prop := time_to_cross = length_of_train / (relative_speed speed_of_train)

-- Define the speed of the train in m/s
def speed_of_train_ms : ℝ := 50 - speed_of_man_ms

-- Convert train's speed to km/h
def speed_of_train_kmh : ℝ := speed_of_train_ms * 3.6

-- Formalizing the theorem to prove that speed of the train is approximately 175 km/h
theorem train_speed : abs (speed_of_train_kmh - 175) < 0.01 :=
by sorry

end train_speed_l602_602621


namespace sequence_involving_constants_l602_602736

theorem sequence_involving_constants (n : ℕ) (hn : n > 0) :
  ∑ i in Finset.range n, 1 / (i + (i * (i + 1))) = (3 * n^2 + 5 * n) / (4 * (n + 1) * (n + 2)) :=
by sorry

end sequence_involving_constants_l602_602736


namespace tomorrow_is_saturday_l602_602862

def day := ℕ   -- Represent days as natural numbers for simplicity
def Monday := 0  -- Let's denote Monday as day 0 (Monday)
def one_week := 7  -- One week consists of 7 days

noncomputable def day_of_week (n : day) : day :=
  n % one_week  -- Calculate the day of the week based on modulo 7

theorem tomorrow_is_saturday
  (x : day)  -- the day before yesterday
  (hx : day_of_week (x + 5) = day_of_week Monday)  -- Monday is 5 days after the day before yesterday
  (today : day)  -- today
  (hy : day_of_week today = day_of_week (x + 2))  -- Today is 2 days after the day before yesterday
  : day_of_week (today + 1) = day_of_week 5 :=   -- Tomorrow will be Saturday (since Saturday is day 5)
by sorry

end tomorrow_is_saturday_l602_602862


namespace women_meet_time_l602_602124

theorem women_meet_time
  (track_length : ℕ)
  (speed1_kmh : ℕ)
  (speed2_kmh : ℕ)
  (start_same_point : true)
  (opposite_directions : true) :
  track_length = 1800 ∧ speed1_kmh = 10 ∧ speed2_kmh = 20 → 
    (∃ time_to_meet : ℚ, time_to_meet ≈ 216) :=
by
  sorry

end women_meet_time_l602_602124


namespace min_value_of_f_on_interval_l602_602340

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem min_value_of_f_on_interval :
  ∃ x ∈ set.Icc (0 : ℝ) (3 : ℝ), f x = 0 ∧
  ∀ y ∈ set.Icc (0 : ℝ) (3 : ℝ), f y ≥ f x := by
sorrow

end min_value_of_f_on_interval_l602_602340


namespace tangent_line_eq_l602_602915

open Real

def f (x : ℝ) : ℝ := cos (2 * x)
def x0 : ℝ := π / 4
def y0 : ℝ := 0

theorem tangent_line_eq : 4 * x0 + 2 * y0 - π = 0 :=
by
  -- the proof would come here, but we'll omit it with sorry
  sorry

end tangent_line_eq_l602_602915


namespace train_duration_l602_602503

-- Define the departure and arrival times using hours and minutes
structure Time :=
(hours : ℕ)
(minutes : ℕ)

-- Definition of the departure and arrival times
def departure := Time.mk 17 48
def arrival := Time.mk 7 23

-- Function to calculate the duration between two times assuming they are within a 24-hour window
def calculate_duration (start : Time) (end : Time) : Time :=
  if end.hours >= start.hours then
    let total_minutes := (end.hours * 60 + end.minutes) - (start.hours * 60 + start.minutes)
    Time.mk (total_minutes / 60) (total_minutes % 60)
  else
    let minutes_to_midnight := (24 * 60) - (start.hours * 60 + start.minutes)
    let minutes_from_midnight := end.hours * 60 + end.minutes
    let total_minutes := minutes_to_midnight + minutes_from_midnight
    Time.mk (total_minutes / 60) (total_minutes % 60)

-- The theorem to be proved
theorem train_duration :
  calculate_duration departure arrival = Time.mk 13 35 :=
by
  sorry

end train_duration_l602_602503


namespace tom_games_value_l602_602118

/--
  Tom bought his games for $200. The value tripled and he sold 40% of them.
  Prove that Tom sold the games for $240.
-/
theorem tom_games_value : (price: ℤ) -> (tripled_value: ℤ) -> (percentage_sold: ℤ) -> (sold_value: ℤ) ->
  price = 200 -> tripled_value = 3 * price -> percentage_sold = 40 -> sold_value = tripled_value * percentage_sold / 100 -> sold_value = 240 :=
by
  assume price tripled_value percentage_sold sold_value
  assume h1: price = 200
  assume h2: tripled_value = 3 * price
  assume h3: percentage_sold = 40
  assume h4: sold_value = tripled_value * percentage_sold / 100
  sorry

end tom_games_value_l602_602118


namespace machine_work_time_today_l602_602242

theorem machine_work_time_today :
  let shirts_today := 40
  let pants_today := 50
  let shirt_rate := 5
  let pant_rate := 3
  let time_for_shirts := shirts_today / shirt_rate
  let time_for_pants := pants_today / pant_rate
  time_for_shirts + time_for_pants = 24.67 :=
by
  sorry

end machine_work_time_today_l602_602242


namespace problem_part_a_problem_part_b_l602_602166

noncomputable def probability_peter_satisfied : ℚ := 25 / 33

noncomputable def expected_number_satisfied_men : ℚ := 1250 / 33

theorem problem_part_a (total_men total_women : ℕ) (h1 : total_men = 50) (h2 : total_women = 50):
  let peter_satisfied := probability_peter_satisfied in
  let prob := λ m w, 1 - ((m / (m + w - 1)) * ((m - 1) / (m + w - 2))) in
  peter_satisfied = prob (total_men - 1) total_women := 
by {
  dsimp [peter_satisfied, prob],
  rw [h1, h2],
  unfold_coes,
  norm_num,
  sorry
}

theorem problem_part_b (total_men total_women : ℕ) (h1 : total_men = 50) (h2 : total_women = 50):
  let satisfied_probability := probability_peter_satisfied in
  let expected_satisfied_men := expected_number_satisfied_men in
  expected_satisfied_men = total_men * satisfied_probability := 
by {
  dsimp [satisfied_probability, expected_satisfied_men],
  rw [h1, h2],
  unfold_coes,
  norm_num,
  sorry
}

end problem_part_a_problem_part_b_l602_602166


namespace evenness_oddness_range_of_a_l602_602724

variables (a x : ℝ)

def f (a x : ℝ) : ℝ := (a * 2^x - 1) / (2^x + 1)

-- Proof of evenness: if a = -1, then f is even
theorem evenness (h : a = -1) : f a x = f a (-x) :=
sorry

-- Proof of oddness: if a = 1, then f is odd
theorem oddness (h : a = 1) : f a x = -f a (-x) :=
sorry

-- Proof of range for a given 1 <= f(x) <= 3 for any x >= 1
theorem range_of_a (h : ∀ (x : ℝ), x ≥ 1 → 1 ≤ f a x ∧ f a x ≤ 3) : 2 ≤ a ∧ a ≤ 3 :=
sorry

end evenness_oddness_range_of_a_l602_602724


namespace visible_radius_sphere_is_10_l602_602584

-- Definitions of given conditions
def shadow_distance_sphere : ℝ := 15
def stick_height : ℝ := 2
def stick_shadow_length : ℝ := 3

-- Visible radius to be proven
def visible_radius (r : ℝ) : Prop := 
  tan (stick_height / stick_shadow_length) = tan (r / shadow_distance_sphere)

-- Proof statement equivalent to the mathematical problem
theorem visible_radius_sphere_is_10 : visible_radius 10 :=
by
  sorry

end visible_radius_sphere_is_10_l602_602584


namespace lcm_of_9_12_15_is_180_l602_602986

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l602_602986


namespace theodore_total_monthly_earnings_l602_602525

-- Define the conditions
def stone_statues_per_month := 10
def wooden_statues_per_month := 20
def cost_per_stone_statue := 20
def cost_per_wooden_statue := 5
def tax_rate := 0.10

-- Calculate earnings from stone and wooden statues
def earnings_from_stone_statues := stone_statues_per_month * cost_per_stone_statue
def earnings_from_wooden_statues := wooden_statues_per_month * cost_per_wooden_statue

-- Total earnings before taxes
def total_earnings := earnings_from_stone_statues + earnings_from_wooden_statues

-- Taxes paid
def taxes_paid := total_earnings * tax_rate

-- Total earnings after taxes
def total_earnings_after_taxes := total_earnings - taxes_paid

-- Theorem stating the total earnings after taxes
theorem theodore_total_monthly_earnings : total_earnings_after_taxes = 270 := sorry

end theodore_total_monthly_earnings_l602_602525


namespace lcm_of_9_12_15_is_180_l602_602990

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l602_602990


namespace lcm_9_12_15_l602_602992

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l602_602992


namespace line_intersects_x_axis_at_l602_602199

theorem line_intersects_x_axis_at {x y : ℤ} (h₁ : (7, 3) : ℤ × ℤ) (h₂ : (3, 7) : ℤ × ℤ) :
  ∃ (p : ℤ × ℤ), p.1 = 10 ∧ p.2 = 0 :=
by
  sorry

end line_intersects_x_axis_at_l602_602199


namespace max_sum_of_arithmetic_seq_l602_602322

variable {α : Type*} [linear_ordered_field α]

def is_arithmetic_sequence (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n * (a 0 + a (n - 1))) / 2

theorem max_sum_of_arithmetic_seq
  (a : ℕ → α)
  (h_arith : is_arithmetic_sequence a)
  (h_a3 : a 2 = 7)
  (h_a1_a7 : a 0 + a 6 = 10) :
  ∃ n : ℕ, n = 6 ∧ ∀ m : ℕ, sum_first_n_terms a n ≥ sum_first_n_terms a m :=
sorry

end max_sum_of_arithmetic_seq_l602_602322


namespace slope_of_tangent_line_to_circle_at_point_l602_602138

-- Definitions of the slope calculation and perpendicularity conditions
def slope (p1 p2 : (ℝ × ℝ)) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

def negative_reciprocal (m : ℝ) : ℝ :=
  -1 / m

def is_tangent_line_slope (p1 p2 : (ℝ × ℝ)) (p3 : (ℝ × ℝ)) (expected_slope : ℝ) : Prop :=
  let radius_slope := slope p1 p2
  let tangent_slope := negative_reciprocal radius_slope
  tangent_slope = expected_slope

-- Theorem statement
theorem slope_of_tangent_line_to_circle_at_point :
  is_tangent_line_slope (2, 5) (8, 3) (8, 3) 3 :=
sorry

end slope_of_tangent_line_to_circle_at_point_l602_602138


namespace sin_C_in_right_angle_triangle_l602_602411

theorem sin_C_in_right_angle_triangle (A B C : ℝ) (hABC : A + B + C = 180) (hA : A = 90) (cosB : ℝ) (hcosB : cos B = 3/5) :
  sin C = 3 / 5 :=
by
  sorry

end sin_C_in_right_angle_triangle_l602_602411


namespace dot_product_bc_values_l602_602788

open Real

/-- Unit vectors in the plane satisfying certain dot product constraints. --/
variables {a b c : Vector (Fin 2) Real}
variable (ha : ‖a‖ = 1)
variable (hb : ‖b‖ = 1)
variable (hc : ‖c‖ = 1)
variable (hab : abs (dot a b) = 1 / 2)
variable (hac : abs (dot a c) = sqrt 3 / 2)

theorem dot_product_bc_values :
  ∃ (values : Finset ℝ), 
    values = { -sqrt 3 / 2, 0, sqrt 3 / 2 } ∧ values = { dot b c } :=
sorry

end dot_product_bc_values_l602_602788


namespace jerry_total_mean_l602_602009

def receivedFromAunt : ℕ := 9
def receivedFromUncle : ℕ := 9
def receivedFromBestFriends : List ℕ := [22, 23, 22, 22]
def receivedFromSister : ℕ := 7

def totalAmountReceived : ℕ :=
  receivedFromAunt + receivedFromUncle +
  receivedFromBestFriends.sum + receivedFromSister

def totalNumberOfGifts : ℕ :=
  1 + 1 + receivedFromBestFriends.length + 1

def meanAmountReceived : ℚ :=
  totalAmountReceived / totalNumberOfGifts

theorem jerry_total_mean :
  meanAmountReceived = 16.29 := by
sorry

end jerry_total_mean_l602_602009


namespace count_ways_to_find_8_integers_l602_602747

theorem count_ways_to_find_8_integers : 
  (∃ (a : Fin 8 → Fin 8), (∀ i j, i ≤ j → a i ≤ a j)) → (nat.choose 15 7 = 6435) :=
begin
  intro h,
  sorry
end

end count_ways_to_find_8_integers_l602_602747


namespace constant_term_of_bella_l602_602648

noncomputable def anna_polynomial : Polynomial ℝ :=
  Polynomial.monic (Polynomial.C 3 + Polynomial.X + Polynomial.X^2)

noncomputable def bella_polynomial : Polynomial ℝ :=
  Polynomial.monic (Polynomial.C 3 + Polynomial.X + Polynomial.X^2)

noncomputable def product_polynomial : Polynomial ℝ :=
  anna_polynomial * bella_polynomial

theorem constant_term_of_bella :
  anna_polynomial.monic ∧ bella_polynomial.monic ∧
  anna_polynomial.coeff 3 = bella_polynomial.coeff 3 ∧
  anna_polynomial.coeff 2 = bella_polynomial.coeff 2 ∧
  product_polynomial = Polynomial.monic (Polynomial.C 9 + 8 * Polynomial.X + 10 * Polynomial.X^2 + 
                                          10 * Polynomial.X^3 + 5 * Polynomial.X^4 + 
                                          2 * Polynomial.X^5 + Polynomial.X^6) →
  bella_polynomial.coeff 0 = 3 := 
sorry

end constant_term_of_bella_l602_602648


namespace find_magnitude_of_QR_l602_602662

noncomputable def equilateral_triangle (a : ℝ) := 
∀ (ABC : Type) [metric_space ABC]
(triangle : ABC → ABC → ABC → Prop)
(a : ℝ),
(triangle A B C) ∧ distance A B = a ∧ distance B C = a ∧ distance C A = a

noncomputable def tangent_circle (Q : Type) (A B : Type) := 
(A B Π [metric_space Q],
∀ (C : Q),
(triangle : A → A → A → Prop)
(a : ℝ),
(triangle A B C) ∧ distance C A = a ∧ distance C B = a)

theorem find_magnitude_of_QR
  (DEF : Type) [metric_space DEF]
  (a : ℝ) (D E F Q R : DEF)
  (equilateral_triangle a) 
  (tangent_circle Q D E)
  (tangent_circle R D F) :
  distance Q R = 9 * real.sqrt 3 := 
sorry

end find_magnitude_of_QR_l602_602662


namespace lcm_of_9_12_15_is_180_l602_602991

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l602_602991


namespace gcd_product_eq_gcd_l602_602037

theorem gcd_product_eq_gcd {a b c : ℤ} (hab : Int.gcd a b = 1) : Int.gcd a (b * c) = Int.gcd a c := 
by 
  sorry

end gcd_product_eq_gcd_l602_602037


namespace problem1_problem2_l602_602704

-- Definitions of the polynomials A and B
def A (x y : ℝ) := x^2 + x * y + 3 * y
def B (x y : ℝ) := x^2 - x * y

-- Problem 1 Statement: 
theorem problem1 (x y : ℝ) (h : (x - 2)^2 + |y + 5| = 0) : 2 * (A x y) - (B x y) = -56 := by
  sorry

-- Problem 2 Statement:
theorem problem2 (x : ℝ) (h : ∀ y, 2 * (A x y) - (B x y) = 0) : x = -2 := by
  sorry

end problem1_problem2_l602_602704


namespace inequality_proof_l602_602890

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
    (((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2) ≥ 9 / 2 := 
by
  sorry

end inequality_proof_l602_602890


namespace range_of_f_l602_602106

def floor_function (x : ℝ) : ℤ :=
  Int.ofNat ⌊x⌋

def f (x : ℝ) : ℤ :=
  floor_function (x + 1/x)

theorem range_of_f :
  ∀ x ∈ set.Ici (1/3) ∩ set.Iio 5, 
  ∃ n ∈ ({2, 3, 4, 5} : set ℤ), f x = n := by
  sorry

end range_of_f_l602_602106


namespace cubic_root_sum_l602_602036

theorem cubic_root_sum (a b : ℝ) (h1 : is_root (λ x : ℂ, x^3 + ↑a * x + ↑b) (1 + complex.i * real.sqrt 7)) : a + b = 20 :=
sorry

end cubic_root_sum_l602_602036


namespace simple_interest_rate_l602_602634

/-- Prove that given Principal (P) = 750, Amount (A) = 900, and Time (T) = 5 years,
    the rate (R) such that the Simple Interest formula holds is 4 percent. -/
theorem simple_interest_rate :
  ∀ (P A T : ℕ) (R : ℕ),
    P = 750 → 
    A = 900 → 
    T = 5 → 
    A = P + (P * R * T / 100) →
    R = 4 :=
by
  intros P A T R hP hA hT h_si
  sorry

end simple_interest_rate_l602_602634


namespace curtain_length_correct_l602_602263

-- Define the problem conditions in Lean
def room_height_feet : ℝ := 8
def feet_to_inches : ℝ := 12
def additional_material_inches : ℝ := 5

-- Define the target length of the curtains
def curtain_length_inches : ℝ :=
  (room_height_feet * feet_to_inches) + additional_material_inches

-- Statement to prove the length of the curtains is 101 inches.
theorem curtain_length_correct :
  curtain_length_inches = 101 := by
  sorry

end curtain_length_correct_l602_602263


namespace basement_pump_time_l602_602187

/-- Prove the time required to pump out all the water from the basement given the conditions. -/
theorem basement_pump_time 
  (length width : ℝ) 
  (depth_pumps_time : ℝ) 
  (num_pumps pump_rate : ℝ) 
  (cubic_to_gal : ℝ) : 
  length = 30 → 
  width = 40 → 
  depth_pumps_time = 12 → 
  depth_pumps_time / 12 = 1 → 
  num_pumps = 4 → 
  pump_rate = 10 → 
  cubic_to_gal = 7.5 → 
  ((length * width * (depth_pumps_time / 12)) * cubic_to_gal) / (num_pumps * pump_rate) = 225 :=
by
  intro length width depth_pumps_time num_pumps pump_rate cubic_to_gal
  intro length_eq width_eq depth_eq foot_conversion_eq num_pumps_eq pump_rate_eq cubic_to_gal_eq
  sorry

end basement_pump_time_l602_602187


namespace find_lengths_and_perimeter_l602_602161

-- Assume the standard Euclidean plane geometry
axiom parallelogram (A B C D : Point) : Prop
axiom circle (Omega : Circle) (diameter : ℤ)
axiom intersection_point (M : Point) (A B C D : Point) : Prop
axiom arc_length_ratio (Omega : Circle) (A E B M : Point) (ratio : ℤ)
axiom segment_length (E M : Point) (length : ℤ)

-- Given conditions
variables {A B C D E K M : Point}
variable Omega : Circle

-- Axioms
axiom parallelogram_ABCD : parallelogram A B C D
axiom Omega_circumscribes_ABM : circle Omega 5
axiom M_is_intersection_point : intersection_point M A C B D
axiom Omega_intersects_ray_CB_at_E : E ∈ ray C B
axiom Omega_intersects_segment_AD_at_K : K ∈ segment D A
axiom length_of_arc_AE_is_twice_BM : arc_length_ratio Omega A E B M 2
axiom length_of_EM_is_4 : segment_length E M 4

-- Required to prove
axiom BC_length : lengthSegment B C = 5
axiom BK_length : lengthSegment B K = 24 / 5
axiom perimeter_AKM : perimeter A K M = 42 / 5

theorem find_lengths_and_perimeter :
  lengthSegment B C = 5 ∧
  lengthSegment B K = 24 / 5 ∧
  perimeter A K M = 42 / 5 :=
sorry

end find_lengths_and_perimeter_l602_602161


namespace sin_870_equals_half_l602_602255

theorem sin_870_equals_half : Real.sin (870 * Real.pi / 180) = 1 / 2 :=
by
  -- Here, we know that reducing the angle by multiples of 360 degrees yields the equivalent angle
  have h1 : 870 = 2 * 360 + 150 := by norm_num
  have h2 : Real.sin (870 * Real.pi / 180) = Real.sin (150 * Real.pi / 180) := 
    by rw [← Real.sin_add (2 * 360 * Real.pi / 180) (150 * Real.pi / 180), Real.sin_2π, zero_add]
  -- Knowing sin(150 degrees) = 1/2 from trigonometric identities
  have h3 : Real.sin (150 * Real.pi / 180) = 1 / 2 := by norm_num
  -- Combining these facts together
  rw [h2, h3]
  norm_num

end sin_870_equals_half_l602_602255


namespace pens_pencils_ratio_l602_602951

theorem pens_pencils_ratio (pencils pens : ℕ) (h1 : pencils = 30) (h2 : pencils = pens + 5) :
  pens.toRat / pencils.toRat = (5 : ℚ) / 6 := 
by
  sorry

end pens_pencils_ratio_l602_602951


namespace pencil_distribution_l602_602114

theorem pencil_distribution : ∃ ways : ℕ, ways = 6 ∧ ∀ (pencils : ℕ) (friends : ℕ) (min_pencils_each : ℕ), 
  pencils = 8 → friends = 3 → min_pencils_each = 2 →
  (ways = (Nat.choose (pencils - friends * min_pencils_each + friends - 1) (friends - 1))) := 
by 
  use 6
  intros pencils friends min_pencils_each hpencils hfriends hmin_pencils_each
  rw [hpencils, hfriends, hmin_pencils_each]
  simp
  sorry

end pencil_distribution_l602_602114


namespace james_has_43_oreos_l602_602006

def james_oreos (jordan : ℕ) : ℕ := 7 + 4 * jordan

theorem james_has_43_oreos (jordan : ℕ) (total : ℕ) (h1 : total = jordan + james_oreos jordan) (h2 : total = 52) : james_oreos jordan = 43 :=
by
  sorry

end james_has_43_oreos_l602_602006


namespace one_point_one_billion_scientific_notation_l602_602499

theorem one_point_one_billion_scientific_notation :
  ∃ (n : ℝ), n = 1.1 * 10^9 ∧ scientific_notation 1.1e9 n :=
sorry

end one_point_one_billion_scientific_notation_l602_602499


namespace lcm_9_12_15_l602_602997

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l602_602997


namespace box_interior_surface_area_l602_602492

-- Defining the conditions
def original_length := 30
def original_width := 20
def corner_length := 5
def num_corners := 4

-- Defining the area calculations based on given dimensions and removed corners
def original_area := original_length * original_width
def area_one_corner := corner_length * corner_length
def total_area_removed := num_corners * area_one_corner
def remaining_area := original_area - total_area_removed

-- Statement to prove
theorem box_interior_surface_area :
  remaining_area = 500 :=
by 
  sorry

end box_interior_surface_area_l602_602492


namespace reflection_correct_l602_602029

noncomputable def reflection_matrix := 
  matrix (fin 3) (fin 3) ℚ

def normal_vector : vector ℚ 3 := ⟨[2, -1, 1], sorry⟩

def reflection_plane_matrix : reflection_matrix :=
  !![-1/3, 2/3, 4/3; 
     4/3, 2/3, -1/3; 
     4/3, -1/3, 2/3]

theorem reflection_correct (u : vector ℚ 3) :
  reflection_plane_matrix.mul_vec u =
  (λ (v : vector ℚ 3), v - 2 * (v.dot_product normal_vector / normal_vector.dot_product normal_vector • normal_vector)) u := sorry

end reflection_correct_l602_602029


namespace tomorrow_is_saturday_l602_602863

def day := ℕ   -- Represent days as natural numbers for simplicity
def Monday := 0  -- Let's denote Monday as day 0 (Monday)
def one_week := 7  -- One week consists of 7 days

noncomputable def day_of_week (n : day) : day :=
  n % one_week  -- Calculate the day of the week based on modulo 7

theorem tomorrow_is_saturday
  (x : day)  -- the day before yesterday
  (hx : day_of_week (x + 5) = day_of_week Monday)  -- Monday is 5 days after the day before yesterday
  (today : day)  -- today
  (hy : day_of_week today = day_of_week (x + 2))  -- Today is 2 days after the day before yesterday
  : day_of_week (today + 1) = day_of_week 5 :=   -- Tomorrow will be Saturday (since Saturday is day 5)
by sorry

end tomorrow_is_saturday_l602_602863


namespace product_of_real_values_r_l602_602299

noncomputable def product_of_real_r : ℚ :=
if (∃ r : ℚ, ∀ x : ℚ, x ≠ 0 → (1 / (3 * x) = (r - x) / 8) = true) then 
  (-32 / 3 : ℚ) 
else 
  0

theorem product_of_real_values_r : product_of_real_r = -32 / 3 :=
by sorry

end product_of_real_values_r_l602_602299


namespace square_perimeter_l602_602603

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end square_perimeter_l602_602603


namespace prime_factor_difference_l602_602833

def is_digit (d : ℕ) : Prop := d >= 0 ∧ d <= 9

theorem prime_factor_difference (A B C : ℕ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (h_neq : A ≠ C) :
  ∃ (p : ℕ), p = 11 ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) :=
by
  have h1 : 100 * A + 10 * B + C - (100 * C + 10 * B + A) = 99 * (A - C),
  { calc
      100 * A + 10 * B + C - (100 * C + 10 * B + A)
          = 100 * (A - C) + (C - A) : by { ring }
      ... = 99 * (A - C)           : by { ring }, },
  use 11,
  split,
  { refl, },
  {
    rw h1,
    exact dvd.intro (A - C) rfl,
  }

end prime_factor_difference_l602_602833


namespace one_point_one_billion_in_scientific_notation_l602_602501

noncomputable def one_point_one_billion : ℝ := 1.1 * 10^9

theorem one_point_one_billion_in_scientific_notation :
  1.1 * 10^9 = 1100000000 :=
by
  sorry

end one_point_one_billion_in_scientific_notation_l602_602501


namespace find_d_l602_602806

-- Given conditions
def initial_point := (126 : ℝ, 21 : ℝ)
def possible_points : List (ℝ × ℝ) := [(126, 0), (105, 0), (111, 0)]

-- Proof goal
theorem find_d : ∃ d : ℝ, (d, 0) ∈ possible_points ∧ d = 111 := by
  sorry

end find_d_l602_602806


namespace simplify_eval_l602_602065

variable (x : ℝ)
def expr := 8 * x^2 - (x - 2) * (3 * x + 1) - 2 * (x + 1) * (x - 1)

theorem simplify_eval (h : x = -2) : expr x = 6 := by
  sorry

end simplify_eval_l602_602065


namespace probability_of_product_multiple_of_4_l602_602495

open Finset

noncomputable def count_pairs_with_product_multiple_of_4 : ℕ :=
let nums := Icc 1 20 in
let pairs := (nums.product nums).filter (λ p, p.1 < p.2) in
(pairs.filter (λ p, ∃ m n : ℕ, p.1 = 2 * m ∧ p.2 = 2 * n) ∨
              (∃ k : ℕ, p.1 = 4 * k ∨ p.2 = 4 * k)).card

noncomputable def total_pairs : ℕ :=
((Icc 1 20).product (Icc 1 20)).filter (λ p, p.1 < p.2).card

theorem probability_of_product_multiple_of_4 :
  (count_pairs_with_product_multiple_of_4.to_rat / total_pairs.to_rat) = 9 / 38 :=
sorry

end probability_of_product_multiple_of_4_l602_602495


namespace square_perimeter_l602_602602

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end square_perimeter_l602_602602


namespace percent_increase_combined_cost_l602_602420

theorem percent_increase_combined_cost :
  let original_bicycle_cost := 150
  let original_helmet_cost := 35
  let original_gloves_cost := 20
  let new_bicycle_cost := original_bicycle_cost * (1 + 0.10)
  let new_helmet_cost := original_helmet_cost * (1 + 0.15)
  let new_gloves_cost := original_gloves_cost * (1 + 0.20)
  let original_total_cost := original_bicycle_cost + original_helmet_cost + original_gloves_cost
  let new_total_cost := new_bicycle_cost + new_helmet_cost + new_gloves_cost
  let total_increase := new_total_cost - original_total_cost
  let percent_increase := (total_increase / original_total_cost) * 100
  percent_increase = 11.83
  :=
by
  unfold let new_bicycle_cost new_helmet_cost new_gloves_cost original_total_cost new_total_cost total_increase percent_increase
  sorry

end percent_increase_combined_cost_l602_602420


namespace longest_interval_between_friday_13ths_l602_602381

theorem longest_interval_between_friday_13ths
  (friday_the_13th : ℕ → ℕ → Prop)
  (at_least_once_per_year : ∀ year, ∃ month, friday_the_13th year month)
  (friday_occurs : ℕ) :
  ∃ (interval : ℕ), interval = 14 :=
by
  sorry

end longest_interval_between_friday_13ths_l602_602381


namespace find_X_l602_602534

theorem find_X :
  ∃ X : ℝ, (3 / 4) * (1 / 8) * X = (1 / 2) * (1 / 4) * 120 ∧ X = 160 :=
by
  use 160
  -- Simplify the original equation
  have h1 : (1 / 2) * (1 / 4) * 120 = 15 := by norm_num
  have h2 : (3 / 4) * (1 / 8) * 160 = (3 / 32) * 160 := by norm_num
  -- Simplify (3 / 32) * 160
  have h3 : (3 / 32) * 160 = 15 := by norm_num
  exact ⟨h3.symm ▸ h1, rfl⟩

end find_X_l602_602534


namespace sphere_eq_l602_602823

theorem sphere_eq (x y z x0 y0 z0 r : ℝ) :
  (x - x0)^2 + (y - y0)^2 + (z - z0)^2 = r^2 ↔ 
  (∃ p : ℝ × ℝ × ℝ, p = (x0, y0, z0) ∧ ∃ r : ℝ, r > 0 ∧
  ∀ q : ℝ × ℝ × ℝ, (q.1 - p.1)^2 + (q.2 - p.2)^2 + (q.3 - p.3)^2 = r^2) := 
sorry

end sphere_eq_l602_602823


namespace number_of_sequences_remainder_l602_602678

theorem number_of_sequences_remainder :
  (binomial 689 15) % 1000 = 689 :=
sorry

end number_of_sequences_remainder_l602_602678


namespace projection_is_correct_l602_602680

-- Define vectors u and v
def u : ℝ × ℝ × ℝ := (3, -1, 2)
def v : ℝ × ℝ × ℝ := (1, 4, -2)

-- Projection function definition
def proj (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
  let dot_vv := v.1 * v.1 + v.2 * v.2 + v.3 * v.3 in
  let scalar := dot_uv / dot_vv in
  (scalar * v.1, scalar * v.2, scalar * v.3)

-- The main theorem that needs to be proved
theorem projection_is_correct : proj u v = ( -5 / 21, -20 / 21, 10 / 21) := by
  sorry

end projection_is_correct_l602_602680


namespace triangle_similarity_l602_602844

variables {A B C : ℝ} -- Angles of triangle ABC
variables {s r : ℝ} -- semiperimeter and inradius of the triangle ABC

-- Define the condition using Lean's notation and syntax
def condition (s r : ℝ) (A B C : ℝ) : Prop :=
  (cot (A / 2))^2 + (2 * (cot (B / 2)))^2 + (3 * (cot (C / 2)))^2 = (6 * s / (7 * r))^2

-- The theorem statement based on the derived correct answer
theorem triangle_similarity (A B C s r : ℝ) (h : condition s r A B C) :
  ∃ (T : Triangle), T.similar_to (triangle.mk 13 40 45) := sorry

end triangle_similarity_l602_602844


namespace least_common_multiple_9_12_15_l602_602972

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l602_602972


namespace sin_cos_identity_l602_602761

theorem sin_cos_identity (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602761


namespace find_k_l602_602032

variables (a b d : EuclideanSpace (Fin 3) ℝ)
variables (k : ℝ)

-- Conditions
axiom unit_vect1 : ∥a∥ = 1
axiom unit_vect2 : ∥b∥ = 1
axiom unit_vect3 : ∥d∥ = 1
axiom orth_ab : inner a b = 0
axiom orth_ad : inner a d = 0
axiom angle_bd : real.angle b d = real.pi / 3

-- Question statement in Lean
theorem find_k : a = k (b ×ᵥ d) → k = (2 * real.sqrt 3) / 3 ∨ k = -(2 * real.sqrt 3) / 3 :=
sorry

end find_k_l602_602032


namespace gray_region_area_is_96pi_l602_602132

noncomputable def smaller_circle_diameter : ℝ := 4

noncomputable def smaller_circle_radius : ℝ := smaller_circle_diameter / 2

noncomputable def larger_circle_radius : ℝ := 5 * smaller_circle_radius

noncomputable def area_of_larger_circle : ℝ := Real.pi * (larger_circle_radius ^ 2)

noncomputable def area_of_smaller_circle : ℝ := Real.pi * (smaller_circle_radius ^ 2)

noncomputable def area_of_gray_region : ℝ := area_of_larger_circle - area_of_smaller_circle

theorem gray_region_area_is_96pi : area_of_gray_region = 96 * Real.pi := by
  sorry

end gray_region_area_is_96pi_l602_602132


namespace range_of_a_outside_circle_l602_602719

  variable (a : ℝ)

  def point_outside_circle (a : ℝ) : Prop :=
    let x := a
    let y := 2
    let distance_sqr := (x - a) ^ 2 + (y - 3 / 2) ^ 2
    let r_sqr := 1 / 4
    distance_sqr > r_sqr

  theorem range_of_a_outside_circle {a : ℝ} (h : point_outside_circle a) :
      2 < a ∧ a < 9 / 4 := sorry
  
end range_of_a_outside_circle_l602_602719


namespace Lizzie_winning_strategy_l602_602448

theorem Lizzie_winning_strategy (n : ℕ) (h₁ : ¬ (n % 3 = 0)) : ∃ strategy : (ℕ × list ℕ) → list ℕ, 
  ∀ start : list ℕ, (start = list.replicate n 2) → 
  (∀ gameState : list ℕ, ∀ turn : ℕ, strategy (turn, gameState) = list (if gameState = list.replicate gameState.length 0 then gameState else sorry)) :=
sorry

end Lizzie_winning_strategy_l602_602448


namespace hyperbola_equation_l602_602731

theorem hyperbola_equation {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0)
    (hfocal : 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5)
    (hslope : b / a = 1 / 8) :
    (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) ↔ (x^2 / a^2 - y^2 / b^2 = 1)) :=
by
  -- Goals and conditions to handle proof
  sorry

end hyperbola_equation_l602_602731


namespace range_a_l602_602431

open Set Real

-- Define the predicate p: real number x satisfies x^2 - 4ax + 3a^2 < 0, where a < 0
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

-- Define the predicate q: real number x satisfies x^2 - x - 6 ≤ 0, or x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the complement sets
def not_p_set (a : ℝ) : Set ℝ := {x | ¬p x a}
def not_q_set : Set ℝ := {x | ¬q x}

-- Define p as necessary but not sufficient condition for q
def necessary_but_not_sufficient (a : ℝ) : Prop := 
  (not_q_set ⊆ not_p_set a) ∧ ¬(not_p_set a ⊆ not_q_set)

-- The main theorem to prove
theorem range_a : {a : ℝ | necessary_but_not_sufficient a} = {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4} :=
by
  sorry

end range_a_l602_602431


namespace lcm_9_12_15_l602_602980

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l602_602980


namespace combinatorial_expression_value_l602_602639

theorem combinatorial_expression_value (t : ℕ) (h1 : 0 ≤ 11 - 2 * t) (h2 : 11 - 2 * t ≤ 5 * t) (h3 : 0 ≤ 2 * t - 2) (h4 : 2 * t - 2 ≤ 11 - 3 * t) (h_t : t = 2) :
  (Nat.choose (5 * t) (11 - 2 * t) - Nat.perm (11 - 3 * t) (2 * t - 2) = 100) :=
by
  -- These are the provided conditions
  have h1 : 0 ≤ 11 - 2 * 2 := by norm_num
  have h2 : 11 - 2 * 2 ≤ 5 * 2 := by norm_num
  have h3 : 0 ≤ 2 * 2 - 2 := by norm_num
  have h4 : 2 * 2 - 2 ≤ 11 - 3 * 2 := by norm_num
  -- It is given that t = 2
  have h_t : t = 2 := by rfl
  sorry

end combinatorial_expression_value_l602_602639


namespace max_b4_b6_l602_602935

-- Define the properties of the sequence {a_n}
variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Condition 1
axiom a_prop : ∀ n : ℕ, 1 ≤ n → a n - a (n + 1) = a n * a (n + 1)

-- Condition 2
axiom b_prop : ∀ n : ℕ, 1 ≤ n → b n = 1 / a n

-- Condition 3
axiom b_sum : ∑ i in finset.range 9, b (i + 1) = 90

-- Conclusion: Show that the maximum value of b_4 * b_6 is 100
theorem max_b4_b6 : ∃ b4 b6, b 4 = b4 ∧ b 6 = b6 ∧ b4 * b6 = 100 :=
sorry

end max_b4_b6_l602_602935


namespace relative_error_comparison_l602_602241

-- Conditions
def length_first := 50
def error_first := 0.05
def length_second := 500
def error_second := 1

-- Definitions of relative errors
def relative_error (length : ℝ) (error : ℝ) : ℝ := (error / length) * 100

-- Definition of the relative errors
def relative_error_first := relative_error length_first error_first
def relative_error_second := relative_error length_second error_second

-- Theorem statement
theorem relative_error_comparison : relative_error_second > relative_error_first :=
by
  sorry

end relative_error_comparison_l602_602241


namespace reggie_left_money_l602_602469

def price_per_book : ℝ := 12
def number_of_books : ℝ := 5
def discount_books : ℝ := 0.10
def game_price : ℝ := 45
def water_bottle_price : ℝ := 10
def price_per_snack : ℝ := 3
def number_of_snacks : ℝ := 3
def bundle_cost : ℝ := 20
def tax_rate : ℝ := 0.12
def initial_amount : ℝ := 200

def total_after_shopping_spree : ℝ :=
  let book_cost := number_of_books * price_per_book
  let discounted_book_cost := book_cost * (1 - discount_books)
  let snack_cost := number_of_snacks * price_per_snack
  let subtotal := discounted_book_cost + game_price + water_bottle_price + snack_cost + bundle_cost
  let total_with_tax := subtotal * (1 + tax_rate)
  initial_amount - total_with_tax

theorem reggie_left_money : total_after_shopping_spree = 45.44 :=
by
  rw [total_after_shopping_spree]
  let book_cost := number_of_books * price_per_book
  let discounted_book_cost := book_cost * (1 - discount_books)
  let snack_cost := number_of_snacks * price_per_snack
  let subtotal := discounted_book_cost + game_price + water_bottle_price + snack_cost + bundle_cost
  let total_with_tax := subtotal * (1 + tax_rate)
  have h1 : book_cost = 60 := by norm_num
  have h2 : discounted_book_cost = 54 := by norm_num
  have h3 : snack_cost = 9 := by norm_num
  have h4 : subtotal = 138 := by norm_num
  have h5 : total_with_tax = 154.56 := by norm_num
  calc
    initial_amount - total_with_tax
      = 200 - 154.56 := by rw h5
      = 45.44 := by norm_num

end reggie_left_money_l602_602469


namespace theodore_total_monthly_earning_l602_602527

def total_earnings (stone_statues: Nat) (wooden_statues: Nat) (cost_stone: Nat) (cost_wood: Nat) (tax_rate: Rat) : Rat :=
  let pre_tax_earnings := stone_statues * cost_stone + wooden_statues * cost_wood
  let tax := tax_rate * pre_tax_earnings
  pre_tax_earnings - tax

theorem theodore_total_monthly_earning : total_earnings 10 20 20 5 0.10 = 270 :=
by
  sorry

end theodore_total_monthly_earning_l602_602527


namespace new_person_weight_increase_avg_l602_602912

theorem new_person_weight_increase_avg
  (W : ℝ) -- total weight of the original 20 people
  (new_person_weight : ℝ) -- weight of the new person
  (h1 : (W - 80 + new_person_weight) = W + 20 * 15) -- condition given in the problem
  : new_person_weight = 380 := 
sorry

end new_person_weight_increase_avg_l602_602912


namespace sine_squared_sum_identity_l602_602556

variable (n : ℕ) (α : ℝ)
hypothesis (hα : α ≠ k * π) -- Ensuring α is not a multiple of π

theorem sine_squared_sum_identity :
  (∑ k in finset.range n, sin (k * α) ^ 2) =
  n / 2 - (cos ((n + 1) * α) * sin (n * α)) / (2 * sin α) :=
sorry

end sine_squared_sum_identity_l602_602556


namespace unit_vectors_equal_magnitude_l602_602710

variable {ℝ : Type*}
variable [normed_group ℝ] [normed_space ℝ ℝ]

theorem unit_vectors_equal_magnitude
    (a b : ℝ)
    (unit_a : ‖a‖ = 1)
    (unit_b : ‖b‖ = 1) :
    ‖a‖ = ‖b‖ := 
sorry

end unit_vectors_equal_magnitude_l602_602710


namespace karl_total_income_is_53_l602_602014

noncomputable def compute_income (tshirt_price pant_price skirt_price sold_tshirts sold_pants sold_skirts sold_refurbished_tshirts: ℕ) : ℝ :=
  let tshirt_income := 2 * tshirt_price
  let pant_income := sold_pants * pant_price
  let skirt_income := sold_skirts * skirt_price
  let refurbished_tshirt_price := (tshirt_price : ℝ) / 2
  let refurbished_tshirt_income := sold_refurbished_tshirts * refurbished_tshirt_price
  tshirt_income + pant_income + skirt_income + refurbished_tshirt_income

theorem karl_total_income_is_53 : compute_income 5 4 6 2 1 4 6 = 53 := by
  sorry

end karl_total_income_is_53_l602_602014


namespace length_of_train_l602_602223

/-- The length of the train given that:
* it is traveling at 45 km/hr
* it crosses a bridge in 30 seconds
* the bridge is 265 meters long
is 110 meters.
-/
theorem length_of_train (speed : ℕ) (time : ℕ) (bridge_length : ℕ) (train_length : ℕ)
  (h1 : speed = 45) (h2 : time = 30) (h3 : bridge_length = 265) :
  (train_length : ℕ) = 110 :=
begin
  sorry
end

end length_of_train_l602_602223


namespace max_modulus_z_add_1_plus_sqrt3i_l602_602312

open Complex

theorem max_modulus_z_add_1_plus_sqrt3i (z : ℂ) (hz : Complex.abs z = 1) : 
  ∃ (M : ℝ), M = 3 ∧ ∀ (z : ℂ), Complex.abs z = 1 → Complex.abs (z + 1 + Complex.I * Real.sqrt 3) ≤ M :=
by
  use 3
  split
  exact rfl
  intros z hz
  have : Complex.abs (z + 1 + Complex.I * Real.sqrt 3) ≤ Complex.abs z + Complex.abs (1 + Complex.I * Real.sqrt 3) := Complex.abs_add _ _
  have : Complex.abs z = 1 := hz
  rw [this] at this
  have : Complex.abs (1 + Complex.I * Real.sqrt 3) = 2 :=
    calc
    Complex.abs (1 + Complex.I * Real.sqrt 3) = Real.sqrt (1 ^ 2 + (Real.sqrt 3) ^ 2) : by simp [Complex.abs]
    ... = Real.sqrt 4 : by norm_num
    ... = 2 : Real.sqrt_eq_rfl
  rw [this] at this
  show Complex.abs (z + 1 + Complex.I * Real.sqrt 3) ≤ 3
  sorry

end max_modulus_z_add_1_plus_sqrt3i_l602_602312


namespace diagonal_of_rectangular_plate_l602_602103

theorem diagonal_of_rectangular_plate (h C : ℝ) (h_cylinder : h = 8) (C_base : C = 6) : 
  (Real.sqrt (C^2 + h^2) = 10) :=
by
  rw [h_cylinder, C_base]
  have : 100 = 10 * 10 := by norm_num
  calc
    Real.sqrt (6^2 + 8^2) = Real.sqrt 100 := by norm_num
    ... = 10 := by rw [this, Real.sqrt_eq_rpow, real.rpow_self 2 10]; norm_num

end diagonal_of_rectangular_plate_l602_602103


namespace quadratic_root_sqrt_2010_2009_l602_602672

theorem quadratic_root_sqrt_2010_2009 :
  (∃ (a b : ℤ), a = 0 ∧ b = -(2010 + 2 * Real.sqrt 2009) ∧
  ∀ (x : ℝ), x^2 + (a : ℝ) * x + (b : ℝ) = 0 → x = Real.sqrt (2010 + 2 * Real.sqrt 2009) ∨ x = -Real.sqrt (2010 + 2 * Real.sqrt 2009)) :=
sorry

end quadratic_root_sqrt_2010_2009_l602_602672


namespace jane_savings_l602_602582

noncomputable def cost_promotion_A (price: ℝ) : ℝ :=
  price + (price / 2)

noncomputable def cost_promotion_B (price: ℝ) : ℝ :=
  price + (price - (price * 0.25))

theorem jane_savings (price : ℝ) (h_price_pos : 0 < price) : 
  cost_promotion_B price - cost_promotion_A price = 12.5 :=
by
  let price := 50
  unfold cost_promotion_A
  unfold cost_promotion_B
  norm_num
  sorry

end jane_savings_l602_602582


namespace binomial_expansion_propositions_l602_602468

theorem binomial_expansion_propositions (x : ℝ) (hx : x ≠ 0) :
  (∑ i in finset.range 2005, nat.binomial 2005 i * (real.sqrt x) ^ i * (-1) ^ (2005 - i)) = 1 ∧
  (-nat.binomial 2005 5 * x ^ 1000 ≠ nat.binomial 2005 6 * x ^ 1999) ∧
  (-1 : ℝ) /∈ (finset.range 2006).image (λ i, nat.binomial 2005 i * (real.sqrt x) ^ i * (-1) ^ (2005 - i)) ∧
  ((real.sqrt 100 - 1) ^ 2005 % 100 = 49) :=
sorry

end binomial_expansion_propositions_l602_602468


namespace percentage_increase_in_side_of_square_l602_602937

theorem percentage_increase_in_side_of_square (p : ℝ) : 
  (1 + p / 100) ^ 2 = 1.3225 → 
  p = 15 :=
by
  sorry

end percentage_increase_in_side_of_square_l602_602937


namespace truth_values_of_p_and_q_l602_602749

variable {p q : Prop}

theorem truth_values_of_p_and_q (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by
  sorry

end truth_values_of_p_and_q_l602_602749


namespace div_by_2_iff_div_by_3_l602_602830

def A : ℕ → ℕ
| 0       := 1
| 1       := 0
| 2       := 2
| (n + 3) := A n + 2 * (list.sum (list.map A (list.range (n + 1))))

theorem div_by_2_iff_div_by_3 (n : ℕ) : (A n % 2 = 0) ↔ (A n % 3 = 0) :=
sorry

end div_by_2_iff_div_by_3_l602_602830


namespace trapezoid_base_length_l602_602532

-- Definitions from the conditions
def trapezoid_area (a b h : ℕ) : ℕ := (1 / 2) * (a + b) * h

theorem trapezoid_base_length (b : ℕ) (h : ℕ) (a : ℕ) (A : ℕ) (H_area : A = 222) (H_upper_side : a = 23) (H_height : h = 12) :
  A = trapezoid_area a b h ↔ b = 14 :=
by sorry

end trapezoid_base_length_l602_602532


namespace switches_in_position_A_after_1000_steps_l602_602553

-- Define the set of switches and their labels
def switches (id : ℕ) : Prop :=
  ∃ x y z : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧ id = 2^x * 3^y * 7^z

-- Define initial positions
def initial_position (id : ℕ) : char :=
  if switches id then 'A' else '_'

-- Define the toggle function
def toggle (pos : char) : char :=
  if pos = 'A' then 'B'
  else if pos = 'B' then 'C'
  else 'A'

-- Define the process over 1000 steps
def final_positions : ℕ → ℕ → char := sorry

-- Define the predicate for the number of switches in position A
def count_switches_in_A : ℕ := sorry

theorem switches_in_position_A_after_1000_steps : count_switches_in_A = 660 := sorry

end switches_in_position_A_after_1000_steps_l602_602553


namespace simplify_evaluate_expr_l602_602895

noncomputable def expr (x : ℝ) : ℝ := 
  ( ( (x^2 - 3) / (x + 2) - x + 2 ) / ( (x^2 - 4) / (x^2 + 4*x + 4) ) )

theorem simplify_evaluate_expr : 
  expr (Real.sqrt 2 + 1) = Real.sqrt 2 + 1 := by
  sorry

end simplify_evaluate_expr_l602_602895


namespace domain_of_f_max_value_condition_l602_602344

noncomputable def f (x : ℝ) : ℝ := sqrt (4 - x) + sqrt (x - 1)

-- Prove domain is [1, 4]
theorem domain_of_f : ∀ x, 1 ≤ x ∧ x ≤ 4 ↔ f x ≥ 0 :=
begin
  sorry
end

-- Prove necessary and sufficient condition for maximum value
theorem max_value_condition (a : ℝ) : 
  f_has_maximum_on [a, a+1) ↔ (3 / 2 < a ∧ a ≤ 3) :=
begin
  sorry
end

end domain_of_f_max_value_condition_l602_602344


namespace ages_correct_l602_602402

variables (Rehana_age Phoebe_age Jacob_age Xander_age : ℕ)

theorem ages_correct
  (h1 : Rehana_age = 25)
  (h2 : Rehana_age + 5 = 3 * (Phoebe_age + 5))
  (h3 : Jacob_age = 3 * Phoebe_age / 5)
  (h4 : Xander_age = Rehana_age + Jacob_age - 4) : 
  Rehana_age = 25 ∧ Phoebe_age = 5 ∧ Jacob_age = 3 ∧ Xander_age = 24 :=
by
  sorry

end ages_correct_l602_602402


namespace flag_design_l602_602259

/-- Given three colors and a flag with three horizontal stripes where no adjacent stripes can be the 
same color, there are exactly 12 different possible flags. -/
theorem flag_design {colors : Finset ℕ} (h_colors : colors.card = 3) : 
  ∃ n : ℕ, n = 12 ∧ (∃ f : ℕ → ℕ, (∀ i, f i ∈ colors) ∧ (∀ i < 2, f i ≠ f (i + 1))) :=
sorry

end flag_design_l602_602259


namespace value_of_expression_l602_602144

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x + 4)^2 = 4 :=
by
  rw [h]
  norm_num
  sorry

end value_of_expression_l602_602144


namespace sphere_property_l602_602820

/-- Definition: In the plane, a circle has the property that a line passing through the tangent point and 
perpendicular to the tangent line must pass through the center of the circle. -/
def circle_property (C O P : Type) [MetricSpace C] [MetricSpace O] [MetricSpace P]
  (circle : C) (center : O) (tangent_point : P) (tangent_line : Line) (line_through_point : Line) : Prop :=
  line_through_point ∩ tangent_point ∧ ⟂ line_through_point tangent_line → passes_through line_through_point center

/-- Proof problem: By analogy, a sphere in space has the property that
a plane passing through the tangent point and perpendicular to the tangent plane must pass through the center of the sphere. -/
theorem sphere_property (S O P : Type) [MetricSpace S] [MetricSpace O] [MetricSpace P]
  (sphere : S) (center : O) (tangent_point : P) (tangent_plane : Plane) (plane_through_point : Plane) :
  circle_property sphere center tangent_point tangent_plane plane_through_point :=
begin
  sorry
end

end sphere_property_l602_602820


namespace incircle_tangency_point_l602_602700

open Set

variable {F1 F2 M N P : Point}

-- Assume a point structure to represent points in a Euclidean plane
structure Point := (x : ℝ) (y : ℝ)

-- Definition: the hyperbola with given foci and vertices
def is_hyperbola (F1 F2 M N : Point) : Set Point :=
  {P | ∃ a b, (a = dist P F1 - dist P F2 ∧ a = dist M F1 - dist M F2 ∧ b = -a)
          ∨ (a = dist P F2 - dist P F1 ∧ a = dist N F2 - dist N F1 ∧ b = -a)}

-- Given the hyperbola and the point P on the hyperbola
variable (hyp : P ∈ is_hyperbola F1 F2 M N)

-- Proving the point of tangency assertion for the triangle P F1 F2
theorem incircle_tangency_point : 
  ∀ P ∈ is_hyperbola F1 F2 M N, 
  ∃ G, (G = M ∨ G = N) ∧ tangent G (incircle P F1 F2) F1 F2 :=
sorry

end incircle_tangency_point_l602_602700


namespace triangle_inequality_angle_bisector_l602_602061

theorem triangle_inequality_angle_bisector
  (AB AC BC BD CD : ℝ)
  (AB_length : AB = 8)
  (CD_length : CD = 4)
  (angle_bisector : BD / CD = AB / AC) :
  let m := 4, n := 16 in m + n = 20 :=
by
  let x := AC,
  have h₁ : 2 < x ∧ x < 16 := sorry,
  have h₂ : x > 4 := sorry,
  sorry

end triangle_inequality_angle_bisector_l602_602061


namespace problem_1261_1399_equivalence_l602_602059

theorem problem_1261_1399_equivalence
  (A' B A B' C E' E : Point)
  (p q : ℝ)
  (γ : ℝ)
  (ext_triangle : A'CB' is extended to parallelogram A'CB'E')
  (bisects : Line e bisects ∠A'E'B')
  (eq_segments : B'E' = CA' = p = q - (q - p) = CB' - CE = B'E)
  (isosceles : Triangle EB'E' is isosceles with ∠B'EE' = γ / 2)
  (coincides : EE' coincides with e)
  : AB = A'B = AB' :=
by sorry

end problem_1261_1399_equivalence_l602_602059


namespace count_alternate_seating_ways_l602_602403

/-!
# The number of ways to seat 3 boys and 3 girls alternately

We are given:
1. There are 3 boys.
2. There are 3 girls.
3. They must be seated alternately in a row.

We need to prove that the total number of ways to seat the boys and girls alternately is 72.
-/

theorem count_alternate_seating_ways : 
  ∃ (boys girls : ℕ), boys = 3 ∧ girls = 3 ∧ 
  (number_of_ways boys girls = 72) :=
begin
  let boys := 3,
  let girls := 3,
  have hboys : boys = 3 := rfl,
  have hgirls : girls = 3 := rfl,
  have hways : number_of_ways boys girls = 72, sorry,
  use [boys, girls],
  exact ⟨hboys, hgirls, hways⟩,
end

end count_alternate_seating_ways_l602_602403


namespace RachelFurnitureAssemblyTime_l602_602060

/-- Rachel bought seven new chairs and three new tables for her house.
    She spent four minutes on each piece of furniture putting it together.
    Prove that it took her 40 minutes to finish putting together all the furniture. -/
theorem RachelFurnitureAssemblyTime :
  let chairs := 7
  let tables := 3
  let time_per_piece := 4
  let total_time := (chairs + tables) * time_per_piece
  total_time = 40 := by
    sorry

end RachelFurnitureAssemblyTime_l602_602060


namespace amoebas_after_ten_days_l602_602628

def amoeba_split_fun (n : Nat) : Nat := 3^n

theorem amoebas_after_ten_days : amoeba_split_fun 10 = 59049 := by
  have h : 3 ^ 10 = 59049 := by norm_num
  exact h

end amoebas_after_ten_days_l602_602628


namespace dot_product_parallel_a_b_l602_602740

noncomputable def a : ℝ × ℝ := (-1, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Definition of parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v2 = (k * v1.1, k * v1.2)

-- Given conditions and result to prove
theorem dot_product_parallel_a_b : ∀ (x : ℝ), parallel a (b x) → x = -2 → (a.1 * (b x).1 + a.2 * (b x).2) = -4 := 
by
  intros x h_parallel h_x
  subst h_x
  sorry

end dot_product_parallel_a_b_l602_602740


namespace find_a_for_minimum_at_one_l602_602348

variable {a : ℝ}

def f (x : ℝ) : ℝ := x * (x + a) ^ 2

theorem find_a_for_minimum_at_one (h : ∀ x, deriv f x = 3 * x^2 + 4 * a * x + a^2) 
  (min_cond : deriv f 1 = 0) : a = -1 :=
by
  sorry

end find_a_for_minimum_at_one_l602_602348


namespace probability_y_gt_1_l602_602410

-- Define the set M
def M : Set ℝ := { x : ℝ | 0 < x ∧ x ≤ 4 }

-- Define the function y = log2(x)
def y (x : ℝ) : ℝ := log x / log 2

-- Define the condition "y > 1"
def condition (x : ℝ) : Prop := y x > 1

-- The theorem stating the probability
theorem probability_y_gt_1 : 
  (∃! x : ℝ, x ∈ M ∧ condition x) → (measure_theory.measure_space.volume (M ∩ {x | condition x})) / (measure_theory.measure_space.volume M) = 1 / 2 :=
by
  -- measure_theory.measure_space.volume represents the length of the interval
  sorry

end probability_y_gt_1_l602_602410


namespace find_p_l602_602735

-- Assume the parametric equations and conditions specified in the problem.
noncomputable def parabola_eqns (p t : ℝ) (M E F : ℝ × ℝ) :=
  ∃ m : ℝ,
    (M = (6, m)) ∧
    (E = (-p / 2, m)) ∧
    (F = (p / 2, 0)) ∧
    (m^2 = 6 * p) ∧
    (|E.1 - F.1|^2 + |E.2 - F.2|^2 = |F.1 - M.1|^2 + |F.2 - M.2|^2) ∧
    (|F.1 - M.1|^2 + |F.2 - M.2|^2 = (F.1 + p / 2)^2 + (F.2 - m)^2)

theorem find_p {p t : ℝ} {M E F : ℝ × ℝ} (h : parabola_eqns p t M E F) : p = 4 :=
by
  sorry

end find_p_l602_602735


namespace number_of_true_propositions_l602_602111

-- Definitions of the propositions
def converse (P Q : Prop) : Prop := Q → P
def contrapositive (P Q : Prop) : Prop := not Q → not P
def negation (P : Prop) : Prop := not P

-- Proposition Definitions
def prop1 := converse (x + y = 0) (x + y = -y + -x)
def prop2 := contrapositive (a > b) (a^2 > b^2)
def prop3 := negation (x ≤ -3 → x^2 + x - 6 > 0)

-- Assertion to check the number of true propositions
theorem number_of_true_propositions : (prop1 ∧ not prop2 ∧ not prop3) ∨ (not prop1 ∧ prop2 ∧ not prop3) ∨ (not prop1 ∧ not prop2 ∧ prop3) := sorry

end number_of_true_propositions_l602_602111


namespace S6_value_l602_602178

noncomputable def x : ℝ := sorry
def S (m : ℕ) : ℝ := x^m + x^(-m)

axiom h_x : x + x⁻¹ = 5

theorem S6_value : S 6 = 12077 :=
by sorry

end S6_value_l602_602178


namespace james_has_43_oreos_l602_602007

def james_oreos (jordan : ℕ) : ℕ := 7 + 4 * jordan

theorem james_has_43_oreos (jordan : ℕ) (total : ℕ) (h1 : total = jordan + james_oreos jordan) (h2 : total = 52) : james_oreos jordan = 43 :=
by
  sorry

end james_has_43_oreos_l602_602007


namespace parabola_hyperbola_coincide_directrix_l602_602406

noncomputable def parabola_directrix (p : ℝ) : ℝ := -p / 2
noncomputable def hyperbola_directrix : ℝ := -3 / 2

theorem parabola_hyperbola_coincide_directrix (p : ℝ) (hp : 0 < p) 
  (h_eq : parabola_directrix p = hyperbola_directrix) : p = 3 :=
by
  have hp_directrix : parabola_directrix p = -p / 2 := rfl
  have h_directrix : hyperbola_directrix = -3 / 2 := rfl
  rw [hp_directrix, h_directrix] at h_eq
  sorry

end parabola_hyperbola_coincide_directrix_l602_602406


namespace least_common_multiple_9_12_15_l602_602976

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l602_602976


namespace tomorrow_is_saturday_l602_602864

def day := ℕ   -- Represent days as natural numbers for simplicity
def Monday := 0  -- Let's denote Monday as day 0 (Monday)
def one_week := 7  -- One week consists of 7 days

noncomputable def day_of_week (n : day) : day :=
  n % one_week  -- Calculate the day of the week based on modulo 7

theorem tomorrow_is_saturday
  (x : day)  -- the day before yesterday
  (hx : day_of_week (x + 5) = day_of_week Monday)  -- Monday is 5 days after the day before yesterday
  (today : day)  -- today
  (hy : day_of_week today = day_of_week (x + 2))  -- Today is 2 days after the day before yesterday
  : day_of_week (today + 1) = day_of_week 5 :=   -- Tomorrow will be Saturday (since Saturday is day 5)
by sorry

end tomorrow_is_saturday_l602_602864


namespace arithmetic_series_sum_relation_l602_602892

theorem arithmetic_series_sum_relation (a₁ d : ℝ) (n : ℕ) :
  let S (k : ℕ) := k * (a₁ + (k - 1) * d / 2) in
  S (2 * n) = S n + (1 / 3) * S (3 * n) := by
  -- Definitions
  let S := λk : ℕ, k * (a₁ + (k - 1) * d / 2)
  sorry

end arithmetic_series_sum_relation_l602_602892


namespace remainder_division_1614_254_eq_90_l602_602505

theorem remainder_division_1614_254_eq_90 :
  ∀ (x : ℕ) (R : ℕ),
    1614 - x = 1360 →
    x * 6 + R = 1614 →
    0 ≤ R →
    R < x →
    R = 90 := 
by
  intros x R h_diff h_div h_nonneg h_lt
  sorry

end remainder_division_1614_254_eq_90_l602_602505


namespace fraction_of_earths_surface_habitable_for_humans_l602_602380

theorem fraction_of_earths_surface_habitable_for_humans :
  (1 / 3 ≠ 0) → (3 / 4 ≠ 0) → (2 / 3 ≠ 0) →
  let land_fraction := 2 / 3,
      inhabitable_land_fraction := 3 / 4,
      inhabitable_surface_fraction := inhabitable_land_fraction * land_fraction
  in inhabitable_surface_fraction = 1 / 2 :=
by
  intros h1 h2 h3
  let land_fraction := 2 / 3
  let inhabitable_land_fraction := 3 / 4
  let inhabitable_surface_fraction := inhabitable_land_fraction * land_fraction
  sorry

end fraction_of_earths_surface_habitable_for_humans_l602_602380


namespace round_trip_return_speed_l602_602577

-- Defining the conditions as variables
def distance_AB : ℝ := 150
def speed_AB : ℝ := 60
def average_speed_round_trip : ℝ := 50

-- The function that calculates the return speed given the conditions
def return_speed (d r v : ℝ) : ℝ := 2 * d / ((d / speed_AB) + (d / r))

-- The main theorem statement
theorem round_trip_return_speed :
  return_speed distance_AB r average_speed_round_trip = r -> r = 42.857 :=
sorry

end round_trip_return_speed_l602_602577


namespace two_digit_number_is_91_l602_602210

/-- A positive two-digit number is odd and is a multiple of 13.
    The product of its digits is a perfect square.
    What is this two-digit number? -/
theorem two_digit_number_is_91 (M : ℕ) (h1 : M > 9) (h2 : M < 100) (h3 : M % 2 = 1) (h4 : M % 13 = 0) (h5 : ∃ n : ℕ, n * n = (M / 10) * (M % 10)) :
  M = 91 :=
sorry

end two_digit_number_is_91_l602_602210


namespace midpoints_collinear_l602_602439

noncomputable def midpoint (P Q : Point) : Point := sorry
noncomputable def collinear (P Q R : Point) : Prop := sorry
noncomputable def intersection (ℓ₁ ℓ₂ : Line) : Point := sorry

variables (A B C D E F : Point)
variables (ℓ₁ ℓ₂ ℓ₃ ℓ₄ : Line)

-- Conditions
-- C and D are on the semicircle with diameter [A B]
axiom on_semicircle : is_on_semicircle A B C ∧ is_on_semicircle A B D

-- ℓ₁ is line (A C) and ℓ₂ is line (B D)
axiom line_ℓ₁ : ℓ₁ = line_through A C
axiom line_ℓ₂ : ℓ₂ = line_through B D

-- ℓ₃ is line (A D) and ℓ₄ is line (B C)
axiom line_ℓ₃ : ℓ₃ = line_through A D
axiom line_ℓ₄ : ℓ₄ = line_through B C

-- F is the intersection of lines (A C) and (B D)
axiom F_intersection : F = intersection ℓ₁ ℓ₂

-- E is the intersection of lines (A D) and (B C)
axiom E_intersection : E = intersection ℓ₃ ℓ₄

-- Conclusion
theorem midpoints_collinear :
  let O := midpoint A B in
  let M := midpoint C D in
  let N := midpoint E F in
  collinear O M N :=
sorry

end midpoints_collinear_l602_602439


namespace product_of_real_values_r_l602_602300

noncomputable def product_of_real_r : ℚ :=
if (∃ r : ℚ, ∀ x : ℚ, x ≠ 0 → (1 / (3 * x) = (r - x) / 8) = true) then 
  (-32 / 3 : ℚ) 
else 
  0

theorem product_of_real_values_r : product_of_real_r = -32 / 3 :=
by sorry

end product_of_real_values_r_l602_602300


namespace time_for_c_l602_602156

theorem time_for_c (a b work_completion: ℝ) (ha : a = 16) (hb : b = 6) (habc : work_completion = 3.2) : 
  (12 : ℝ) = 
  (48 * work_completion - 48) / 4 := 
sorry

end time_for_c_l602_602156


namespace equilateral_triangles_count_l602_602513

-- Definitions for conditions
def line1 (k : ℤ) : ℝ → ℝ := λ x, k
def line2 (k : ℤ) : ℝ → ℝ := λ x, (√3) * x + 3*k
def line3 (k : ℤ) : ℝ → ℝ := λ x, - (√3) * x + 3*k

def intersect_triangles_count (min_k max_k : ℤ) : ℕ := sorry

-- Condition stating the range of k values
def valid_k (k : ℤ) : Prop := -12 ≤ k ∧ k ≤ 12

-- Main theorem stating the proof problem
theorem equilateral_triangles_count :
  (∀ k, valid_k k → 
       intersect_triangles_count (-12) (12) = 5256) := sorry

end equilateral_triangles_count_l602_602513


namespace count_ordered_pairs_satisfying_equation_l602_602360

theorem count_ordered_pairs_satisfying_equation :
  ({p : ℤ × ℤ | p.1 ^ 4 + p.2 ^ 2 = 4 * p.2}.to_finset.card = 2) :=
by { sorry }

end count_ordered_pairs_satisfying_equation_l602_602360


namespace sin_range_l602_602930

theorem sin_range (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2)) : 
  Set.range (fun x => Real.sin x) = Set.Icc (1/2 : ℝ) 1 :=
sorry

end sin_range_l602_602930


namespace parallelogram_properties_l602_602660

noncomputable def perimeter (x y : ℤ) : ℝ :=
  2 * (5 + Real.sqrt ((x - 7) ^ 2 + (y - 3) ^ 2))

noncomputable def area (x y : ℤ) : ℝ :=
  5 * abs (y - 3)

theorem parallelogram_properties (x y : ℤ) (hx : x = 7) (hy : y = 7) :
  (perimeter x y + area x y) = 38 :=
by
  simp [perimeter, area, hx, hy]
  sorry

end parallelogram_properties_l602_602660


namespace factorial_trailing_zeros_l602_602375

theorem factorial_trailing_zeros :
  ∃ (S : Finset ℕ), (∀ m ∈ S, 1 ≤ m ∧ m ≤ 30) ∧ (S.card = 24) ∧ (∀ m ∈ S, 
    ∃ n : ℕ, ∃ k : ℕ,  n ≥ k * 5 ∧ n ≤ (k + 1) * 5 - 1 ∧ 
      m = (n / 5) + (n / 25) + (n / 125) ∧ ((n / 5) % 5 = 0)) :=
sorry

end factorial_trailing_zeros_l602_602375


namespace slope_of_perpendicular_line_l602_602541

theorem slope_of_perpendicular_line (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3) (h2 : y1 = -2) (h3 : x2 = 1) (h4 : y2 = 4) :
  let m := ((y2 - y1) / (x2 - x1)) in
  let m_perpendicular := -1 / m in
  m_perpendicular = 1 / 3 :=
by
  sorry

end slope_of_perpendicular_line_l602_602541


namespace AssignmentSymbol_is_C_l602_602547

def FlowchartSymbol (Option : Char) : Prop :=
  match Option with
  | 'A' => "start and end symbols"
  | 'B' => "decision symbol"
  | 'C' => "assignment and calculation symbol"
  | 'D' => "input/output symbol"
  | _   => "unknown symbol"

theorem AssignmentSymbol_is_C :
  (FlowchartSymbol 'A' = "start and end symbols") →
  (FlowchartSymbol 'B' = "decision symbol") →
  (FlowchartSymbol 'C' = "assignment and calculation symbol") →
  (FlowchartSymbol 'D' = "input/output symbol") →
  (FlowchartSymbol 'C' = "assignment and calculation symbol") := by
  intros hA hB hC hD
  exact hC

end AssignmentSymbol_is_C_l602_602547


namespace tan_theta_eq_l602_602692

theorem tan_theta_eq (θ : ℝ) (h : Real.tan θ = -3) :
  ( (sin θ + 2 * cos θ) / (cos θ - 3 * sin θ) = -1 / 10 ) ∧ 
  ( sin θ ^ 2 - sin θ * cos θ = 6 / 5 ) :=
by
  sorry

end tan_theta_eq_l602_602692


namespace train_length_is_110_l602_602220

-- Definitions based on conditions
def speed_kmh : ℝ := 45  -- Speed of the train in km/hr
def crossing_time : ℝ := 30  -- Time to cross the bridge in seconds
def bridge_length : ℝ := 265  -- Length of the bridge in meters

-- Convert speed from km/hr to m/s
def speed_ms : ℝ := speed_kmh * 1000 / 3600

-- Define the total distance covered in terms of meters
def total_distance : ℝ := speed_ms * crossing_time

-- Define the length of the train
def train_length : ℝ := total_distance - bridge_length

-- Theorem stating that the length of the train is 110 meters
theorem train_length_is_110 : train_length = 110 :=
  sorry

end train_length_is_110_l602_602220


namespace at_least_3_speaking_l602_602786

noncomputable def probability_speaking : ℚ := 1 / 3

theorem at_least_3_speaking (n : ℕ) (speaking : ℕ → Prop) : 
  n = 6 →
  (∀ k, speaking k ↔ k ≤ n ∧ k ≥ 0) →
  (∀ k, ProbabilitySpace.Probability (fun _ => speaking k) = probability_speaking) →
  ProbabilitySpace.Probability (fun _ => (∑ k in finset.range 7, k * (ProbabilitySpace.Probability (fun _ => speaking k))) ≥ 3) = 233 / 729 :=
by sorry

end at_least_3_speaking_l602_602786


namespace rational_function_at_x_4_has_value_l602_602510

noncomputable def p (k : ℝ) (x : ℝ) := k * (x - 6) * (x - 2)
noncomputable def q (x : ℝ) := (x - 6) * (x + 1)
noncomputable def f (k : ℝ) (x : ℝ) := p k x / q x

theorem rational_function_at_x_4_has_value (k : ℝ) (h_vert_asymptote : ∃ k, q (-1) = 0)
    (h_horz_asymptote : f k → at_top = -3) (h_hole : q 6 = 0)
    (h_intersect : p k 2 = 0) : f k 4 = -6 / 5 := by
  sorry

end rational_function_at_x_4_has_value_l602_602510


namespace negation_proposition_l602_602929

theorem negation_proposition :
  ∃ (a : ℝ) (n : ℕ), n > 0 ∧ a ≠ n ∧ a * n = 2 * n :=
sorry

end negation_proposition_l602_602929


namespace find_t_l602_602738

variables {a b : EuclideanSpace ℝ}
variables (t : ℝ)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
variables (angle_ab : real.angle a b = real.angle.pi_div_three)

def dot_product (u v : EuclideanSpace ℝ) : ℝ := 
  u.toBasis.toMatrix.vecMul v.toBasis.toMatrix

def vector_c := 2 • a - 3 • b
def vector_d := a + t • b

noncomputable def is_perpendicular (u v : EuclideanSpace ℝ) : Prop := 
  dot_product u v = 0

theorem find_t 
  (h_perpendicular: is_perpendicular vector_c vector_d) 
  (h_cos: dot_product a b = 1 / 2) : 
  t = 1 / 4 := 
sorry

end find_t_l602_602738


namespace parabola_properties_l602_602969

-- Define the parabola function as y = x^2 + px + q
def parabola (p q : ℝ) (x : ℝ) : ℝ := x^2 + p * x + q

-- Prove the properties of parabolas for varying p and q.
theorem parabola_properties (p q p' q' : ℝ) :
  (∀ x : ℝ, parabola p q x = x^2 + p * x + q) ∧
  (∀ x : ℝ, parabola p' q' x = x^2 + p' * x + q') →
  (∀ x : ℝ, ( ∃ k h : ℝ, parabola p q x = (x + h)^2 + k ) ∧ 
               ( ∃ k' h' : ℝ, parabola p' q' x = (x + h')^2 + k' ) ) ∧
  (∀ x : ℝ, h = -p / 2 ∧ k = q - p^2 / 4 ) ∧
  (∀ x : ℝ, h' = -p' / 2 ∧ k' = q' - p'^2 / 4 ) ∧
  (∀ x : ℝ, (h, k) ≠ (h', k') → parabola p q x ≠ parabola p' q' x) ∧
  (∀ x : ℝ, h = h' ∧ k = k' → parabola p q x = parabola p' q' x) :=
by
  sorry

end parabola_properties_l602_602969


namespace tomorrow_is_saturday_l602_602865

noncomputable def day_of_the_week := ℕ

def Monday : day_of_the_week := 0
def Tuesday : day_of_the_week := 1
def Wednesday : day_of_the_week := 2
def Thursday : day_of_the_week := 3
def Friday : day_of_the_week := 4
def Saturday : day_of_the_week := 5
def Sunday : day_of_the_week := 6

def day_before_yesterday (today : day_of_the_week) : day_of_the_week :=
(today + 5) % 7

theorem tomorrow_is_saturday (today : day_of_the_week) :
  (day_before_yesterday today + 5) % 7 = Monday →
  ((today + 1) % 7 = Saturday) :=
by
  sorry

end tomorrow_is_saturday_l602_602865


namespace marble_probability_green_l602_602531

theorem marble_probability_green :
  let P_white := 1 / 6 in
  let P_red_or_blue := 0.6333333333333333 in
  (1 - P_white - P_red_or_blue) = 0.2 :=
by
  let P_white := 1 / 6
  let P_red_or_blue := 0.6333333333333333
  have h1 : 1 - P_white - P_red_or_blue = 0.2 := by sorry
  exact h1

end marble_probability_green_l602_602531


namespace trig_expression_evaluation_l602_602280

theorem trig_expression_evaluation 
  (cos_10 : ℝ := real.cos (10 * real.pi / 180))
  (sin_170 : ℝ := real.sin (170 * real.pi / 180)) :
  (real.sqrt 3 / cos_10 - 1 / sin_170) = -4 := 
by
  sorry

end trig_expression_evaluation_l602_602280


namespace cody_increases_steps_by_1000_l602_602253

theorem cody_increases_steps_by_1000 (x : ℕ) 
  (initial_steps : ℕ := 7000)
  (steps_logged_in_four_weeks : ℕ := 70000)
  (goal_steps : ℕ := 100000)
  (remaining_steps : ℕ := 30000)
  (condition : 1000 + 7 * (1 + 2 + 3) * x = 70000 → x = 1000) : x = 1000 :=
by
  sorry

end cody_increases_steps_by_1000_l602_602253


namespace find_a_if_lines_are_perpendicular_l602_602089

noncomputable def perpendicular_lines {a : ℝ} : Prop :=
  (3 * a) * (a - 1) + (-1) * 1 = 0

theorem find_a_if_lines_are_perpendicular :
  ∀ {a : ℝ}, perpendicular_lines → (a = 1 ∨ a = -1) :=
by 
  intros a h₁
  sorry

end find_a_if_lines_are_perpendicular_l602_602089


namespace find_f_of_3_l602_602777

-- Define the function f(t) such that f(2x + 1) = x^2 - 2x + 1
def f : ℝ → ℝ := λ t, ( (t/2) - 2 )^2

-- Prove that f(3) = 0
theorem find_f_of_3 : f 3 = 0 :=
by
  sorry

end find_f_of_3_l602_602777


namespace boys_in_art_class_l602_602101

noncomputable def number_of_boys (ratio_girls_to_boys : ℕ × ℕ) (total_students : ℕ) : ℕ :=
  let (g, b) := ratio_girls_to_boys
  let k := total_students / (g + b)
  b * k

theorem boys_in_art_class (h : number_of_boys (4, 3) 35 = 15) : true := 
  sorry

end boys_in_art_class_l602_602101


namespace least_common_multiple_9_12_15_l602_602975

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l602_602975


namespace find_m_l602_602355

variable (m : ℝ)

-- Definitions of the vectors
def AB : ℝ × ℝ := (m + 3, 2 * m + 1)
def CD : ℝ × ℝ := (m + 3, -5)

-- Definition of perpendicular vectors, dot product is zero
def perp (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_m (h : perp (AB m) (CD m)) : m = 2 := by
  sorry

end find_m_l602_602355


namespace area_triangle_POQ_l602_602805

open EuclideanGeometry

variables (A B C D P N Q M O: Point)
variables [is_square ABCD] [lies_on P AB] [bisects DP BC N] [lies_on Q BA] [bisects CQ AD M] [intersects DP CQ O]
variable (h_area : area ABCD = 192)

theorem area_triangle_POQ : area_triangle POQ = 216 :=
sorry

end area_triangle_POQ_l602_602805


namespace ratio_BD_BO_l602_602466

theorem ratio_BD_BO :
  ∀ (O A C B D : Type) [MetricSpace O] [MetricSpace A] [MetricSpace C] [MetricSpace B] [MetricSpace D],
    (circle_centered_at O A) ∧ (circle_centered_at O C) ∧
    (tangent_at B A O) ∧ (tangent_at B C O) ∧
    (isosceles_triangle B A C) ∧
    (angle_B_ABC B = 100) ∧
    (circle_intersects_BO O B D) →
    (BD_ratio_BO B D O = 1 - real.sin (40 * real.pi / 180)) :=
by
  intros O A C B D MetricSpace_O MetricSpace_A MetricSpace_C MetricSpace_B MetricSpace_D
    CIRCLE_OA CIRCLE_OC TANGENT_BA TANGENT_BC ISO_TRIANGLE ABCA_GE100 INTERSECT_BD
  sorry

end ratio_BD_BO_l602_602466


namespace trey_turtles_l602_602122

theorem trey_turtles (Kr K T : ℕ) (h_Kr : Kr = 12) (h_K : K = 1/4 * Kr) (h_total : T + K + Kr = 30) :
  T = 5 * K := 
by
  subst h_Kr
  simp at h_K
  rw h_K at h_total
  sorry

end trey_turtles_l602_602122


namespace product_of_real_values_r_l602_602297

theorem product_of_real_values_r {x r : ℝ} (h : x ≠ 0) (heq : (1 / (3 * x)) = ((r - x) / 8)) :
  (∃! x : ℝ, 24 * x^2 - 8 * r * x + 24 = 0) →
  r = 6 ∨ r = -6 ∧ (r * -r) = -36 :=
by
  sorry

end product_of_real_values_r_l602_602297


namespace f_f_x_eq_1_f_g_x_g_f_x_g_g_x_l602_602566

def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then 1 else 0

def g (x : ℝ) : ℝ :=
  if |x| ≤ 1 then 2 - x^2 else 2

theorem f_f_x_eq_1 : ∀ x : ℝ, f (f x) = 1 := by
  sorry

theorem f_g_x : ∀ x : ℝ, f (g x) = (if x = 1 ∨ x = -1 then 1 else 0) := by
  sorry

theorem g_f_x : ∀ x : ℝ, g (f x) = (if |x| ≤ 1 then 1 else 2) := by
  sorry

theorem g_g_x : ∀ x : ℝ, g (g x) = (if x = 1 ∨ x = -1 then 1 else 2) := by
  sorry

end f_f_x_eq_1_f_g_x_g_f_x_g_g_x_l602_602566


namespace tomorrow_is_saturday_l602_602868

noncomputable def day_of_the_week := ℕ

def Monday : day_of_the_week := 0
def Tuesday : day_of_the_week := 1
def Wednesday : day_of_the_week := 2
def Thursday : day_of_the_week := 3
def Friday : day_of_the_week := 4
def Saturday : day_of_the_week := 5
def Sunday : day_of_the_week := 6

def day_before_yesterday (today : day_of_the_week) : day_of_the_week :=
(today + 5) % 7

theorem tomorrow_is_saturday (today : day_of_the_week) :
  (day_before_yesterday today + 5) % 7 = Monday →
  ((today + 1) % 7 = Saturday) :=
by
  sorry

end tomorrow_is_saturday_l602_602868


namespace m_plus_n_value_l602_602063

noncomputable def triangle_conditions : Prop :=
  ∃ (A B C D : Point) (AB AC AD BC BD CD : ℝ),
  AB = 8 ∧ CD = 4 ∧
  (is_angle_bisector A B C D) ∧
  ∀ AC, AC ∈ set.Ioo 4 16
  ∧ ∃ m n : ℝ, set.Ioo m n = set.Ioo 4 16

theorem m_plus_n_value :
  triangle_conditions → 
  ∃ (m n : ℝ), (m + n = 20) :=
by 
  intro h
  cases h with A h
  cases h with B h
  cases h with C h
  cases h with D h
  cases h with AB h
  cases h with AC h
  cases h with AD h
  cases h with BC h
  cases h with BD h
  cases h with CD h
  cases h with h_AB h
  cases h with h_CD h
  cases h with h_angle_bisector h
  cases h with h_in_interval h
  cases h with m h
  cases h with n h_interval_eq
  existsi m
  existsi n
  rw h_interval_eq
  exact 8 + 12
  sorry

end m_plus_n_value_l602_602063


namespace geometric_sum_common_ratio_l602_602940

theorem geometric_sum_common_ratio (a₁ a₂ : ℕ) (q : ℕ) (S₃ : ℕ)
  (h1 : S₃ = a₁ + 3 * a₂)
  (h2: S₃ = a₁ * (1 + q + q^2)) :
  q = 2 :=
by
  sorry

end geometric_sum_common_ratio_l602_602940


namespace trace_shape_segments_l602_602437

theorem trace_shape_segments {n : ℕ} (h : n ≥ 5) (O A B : Point) (OAB : triangle O A B) (X Y Z : Point) (XYZ : triangle X Y Z) 
  (hXYZOAB : congruent_triangles XYZ OAB) (Y_side : ∀ {P}, on_side n P → on_side n Y) 
  (Z_side : ∀ {P}, on_side n P → on_side n Z) : 
  traces_X_in_ngon X Y Z n O = segments n (OB * (1 / (cos (real.pi / n)) - 1)) :=
begin
  -- Proof goes here
  sorry
end

end trace_shape_segments_l602_602437


namespace find_varphi_l602_602511

theorem find_varphi :
  ∀ (x : ℝ) (ϕ : ℝ),
  (∀ y, y = cos (2 * x + ϕ) ∨ y = sin (2 * x + π / 3) →
    (cos (2 * (x + π / 4) + ϕ) = sin (2 * x + π / 3))) →
  ϕ = 5 * π / 6 :=
by
  intro x ϕ h
  sorry

end find_varphi_l602_602511


namespace sin_cos_identity_l602_602759

theorem sin_cos_identity (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602759


namespace part_a_part_b_l602_602163

-- Define the setup
def total_people := 100
def total_men := 50
def total_women := 50

-- Peter Ivanovich's position and neighbor relations
def pi_satisfied_prob : ℚ := 25 / 33

-- Expected number of satisfied men
def expected_satisfied_men : ℚ := 1250 / 33

-- Lean statements for the problems

-- Part (a): Prove Peter Ivanovich's satisfaction probability
theorem part_a (total_people = 100) (total_men = 50) (total_women = 50) : 
  pi_satisfied_prob = 25 / 33 := 
sorry

-- Part (b): Expected number of satisfied men
theorem part_b (total_people = 100) (total_men = 50) (total_women = 50) : 
  expected_satisfied_men = 1250 / 33 := 
sorry

end part_a_part_b_l602_602163


namespace square_perimeter_l602_602608

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end square_perimeter_l602_602608


namespace determine_x_l602_602266

theorem determine_x :
  ∃ x : ℕ, 4 * 5^(x - 1) = 7812500 ∧ x = 10 :=
begin
  use 10,
  split,
  {
    calc 4 * 5^(10 - 1) = 4 * 5^9 : by norm_num
                 ...    = 4 * 1953125 : by norm_num
                 ...    = 7812500 : by norm_num,
  },
  exact rfl,
end

end determine_x_l602_602266


namespace area_of_triangle_ABC_l602_602775

theorem area_of_triangle_ABC (CD : ℝ) (angle_A : ℝ) 
  (h_CD : CD = real.sqrt 3) 
  (h_angle_A : angle_A = real.pi / 6) : 
  ∃ (area : ℝ), area = 2 * real.sqrt 3 :=
by
  sorry

end area_of_triangle_ABC_l602_602775


namespace sum_even_integers_l602_602542

theorem sum_even_integers : 
  ∑ k in finset.filter (λ n, n % 2 = 0) (finset.range (701 - 400)) (λ n, n + 400) = 83050 :=
by sorry

end sum_even_integers_l602_602542


namespace prime_factors_of_69_l602_602098

theorem prime_factors_of_69 
  (prime : ℕ → Prop)
  (is_prime : ∀ n, prime n ↔ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ 
                        n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23)
  (x y : ℕ)
  (h1 : 15 < 69)
  (h2 : 69 < 70)
  (h3 : prime y)
  (h4 : 13 < y)
  (h5 : y < 25)
  (h6 : 69 = x * y)
  : prime x ∧ x = 3 := 
sorry

end prime_factors_of_69_l602_602098


namespace compute_9h_l602_602921

noncomputable def log3 : ℝ := log 3
noncomputable def log5 : ℝ := log 5

def log_base (base x : ℝ) : ℝ := log x / log base

def a : ℝ := log_base 3 125
def b : ℝ := log_base 5 64
def h : ℝ := sqrt (a * a + b * b)

theorem compute_9h : 9^h = 5^15.6 :=
sorry

end compute_9h_l602_602921


namespace lcm_9_12_15_l602_602993

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l602_602993


namespace max_tied_teams_for_most_wins_l602_602397

theorem max_tied_teams_for_most_wins 
  (n : ℕ) 
  (h₀ : n = 6)
  (total_games : ℕ := n * (n - 1) / 2)
  (game_result : Π (i j : ℕ), i ≠ j → (0 = 1 → false) ∨ (1 = 1))
  (rank_by_wins : ℕ → ℕ) : true := sorry

end max_tied_teams_for_most_wins_l602_602397


namespace train_speed_in_kmph_l602_602219

theorem train_speed_in_kmph (length_in_m : ℝ) (time_in_s : ℝ) (length_in_m_eq : length_in_m = 800.064) (time_in_s_eq : time_in_s = 18) : 
  (length_in_m / 1000) / (time_in_s / 3600) = 160.0128 :=
by
  rw [length_in_m_eq, time_in_s_eq]
  /-
  To convert length in meters to kilometers, divide by 1000.
  To convert time in seconds to hours, divide by 3600.
  The speed is then computed by dividing the converted length by the converted time.
  -/
  sorry

end train_speed_in_kmph_l602_602219


namespace bus_people_difference_l602_602952

theorem bus_people_difference 
  (initial : ℕ) (got_off : ℕ) (got_on : ℕ) (current : ℕ) 
  (h_initial : initial = 35)
  (h_got_off : got_off = 18)
  (h_got_on : got_on = 15)
  (h_current : current = initial - got_off + got_on) :
  initial - current = 3 := by
  sorry

end bus_people_difference_l602_602952


namespace colin_skip_speed_l602_602254

variables (B T C : ℝ)

def Bruce_speed := 1
def Tony_speed := 2 * Bruce_speed
def Brandon_speed := 2 / 3 -- (To reflect the solved Brandon's speed from the steps)
def Colin_speed := 6 * Brandon_speed

theorem colin_skip_speed :
  Colin_speed = 4 :=
by
  unfold Colin_speed Brandon_speed Tony_speed Bruce_speed
  sorry

end colin_skip_speed_l602_602254


namespace largest_removable_columns_l602_602882

theorem largest_removable_columns (m n k : ℕ) (h_mn : m ≤ n)
  (h_unique_rows : ∀ C : Fin n → Fin m → bool, ∀ i j : Fin m, i ≠ j → ∃ c : Fin n, C c i ≠ C c j) :
  k = n - m + 1 :=
sorry

end largest_removable_columns_l602_602882


namespace gcd_9_18_36_and_lcm_9_18_36_l602_602438

def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcd_9_18_36_and_lcm_9_18_36 :
  let A := gcd (gcd 9 18) 36
  let B := lcm (lcm 9 18) 36
  A + B = 45 := by
  sorry

end gcd_9_18_36_and_lcm_9_18_36_l602_602438


namespace specific_value_correct_l602_602116

noncomputable def specific_value (x : ℝ) : ℝ :=
  (3 / 5) * (x ^ 2)

theorem specific_value_correct :
  specific_value 14.500000000000002 = 126.15000000000002 :=
by
  sorry

end specific_value_correct_l602_602116


namespace volume_of_prism_l602_602617

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 10) (hwh : w * h = 15) (hlh : l * h = 18) :
  l * w * h = 30 * Real.sqrt 3 :=
by
  sorry

end volume_of_prism_l602_602617


namespace one_point_one_billion_in_scientific_notation_l602_602500

noncomputable def one_point_one_billion : ℝ := 1.1 * 10^9

theorem one_point_one_billion_in_scientific_notation :
  1.1 * 10^9 = 1100000000 :=
by
  sorry

end one_point_one_billion_in_scientific_notation_l602_602500


namespace average_after_discarding_l602_602564

theorem average_after_discarding (avg50 : ℝ) (sum50 : ℝ) (sum_discarded : ℝ) (n_remaining : ℕ) (new_avg : ℝ) :
  avg50 = 44 → sum50 = avg50 * 50 → sum_discarded = 45 + 55 → n_remaining = 50 - 2 →
  new_avg = (sum50 - sum_discarded) / n_remaining → new_avg = 43.75 :=
by
  intros h_avg50 h_sum50 h_sum_discarded h_n_remaining h_new_avg
  rw [h_avg50] at h_sum50
  rw [h_sum50, h_sum_discarded, h_n_remaining] at h_new_avg
  linarith

end average_after_discarding_l602_602564


namespace precision_of_6_7e6_l602_602944

theorem precision_of_6_7e6 :
  let x := 6.7e6 in
  ∃ s, (to_digits s x = [6,7,0,0,0,0,0]) ∧ s = 10^6 ∧ last_significant_digit_place s = 10^5 :=
by
  sorry

end precision_of_6_7e6_l602_602944


namespace last_two_digits_of_quotient_l602_602920

noncomputable def greatest_integer_not_exceeding (x : ℝ) : ℤ := ⌊x⌋

theorem last_two_digits_of_quotient :
  let a : ℤ := 10 ^ 93
  let b : ℤ := 10 ^ 31 + 3
  let x : ℤ := greatest_integer_not_exceeding (a / b : ℝ)
  (x % 100) = 8 :=
by
  sorry

end last_two_digits_of_quotient_l602_602920


namespace square_perimeter_l602_602594

theorem square_perimeter (area : ℝ) (h : area = 625) :
  ∃ p : ℝ, p = 4 * real.sqrt area ∧ p = 100 :=
by
  sorry

end square_perimeter_l602_602594


namespace tomorrow_is_saturday_l602_602870

noncomputable def day_before_yesterday : string := "Wednesday"
noncomputable def today : string := "Friday"
noncomputable def tomorrow : string := "Saturday"

theorem tomorrow_is_saturday (dby : string) (tod : string) (tom : string) 
  (h1 : dby = "Wednesday") (h2 : tod = "Friday") (h3 : tom = "Saturday")
  (h_cond : "Monday" = dby + 5) : 
  tom = "Saturday" := 
sorry

end tomorrow_is_saturday_l602_602870


namespace largest_real_solution_sum_l602_602269

theorem largest_real_solution_sum (d e f : ℕ) (x : ℝ) (h : d = 13 ∧ e = 61 ∧ f = 0) : 
  (∃ d e f : ℕ, d + e + f = 74) ↔ 
  (n : ℝ) * n = (x - d)^2 ∧ 
  (∀ x : ℝ, 
    (4 / (x - 4)) + (6 / (x - 6)) + (18 / (x - 18)) + (20 / (x - 20)) = x^2 - 13 * x - 6 → 
    n = x) :=
sorry

end largest_real_solution_sum_l602_602269


namespace diagonal_AC_length_l602_602616

-- Define the dimensions of the rectangular prism
def length : ℝ := 12
def width : ℝ := 15
def height : ℝ := 8

-- Define the length of the diagonal AC
noncomputable def diagonal_AC : ℝ := Real.sqrt (height^2 + (Real.sqrt (length^2 + width^2))^2)

-- The theorem to be proved
theorem diagonal_AC_length : diagonal_AC = Real.sqrt 433 := by
  -- Proof skipped
  sorry

end diagonal_AC_length_l602_602616


namespace decreasing_function_condition_l602_602412

theorem decreasing_function_condition (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, x ≤ 3 → deriv f x ≤ 0) ↔ (m ≥ 1) :=
by 
  sorry

end decreasing_function_condition_l602_602412


namespace total_people_going_to_zoo_and_amusement_park_l602_602183

theorem total_people_going_to_zoo_and_amusement_park :
  (7.0 * 45.0) + (5.0 * 56.0) = 595.0 :=
by
  sorry

end total_people_going_to_zoo_and_amusement_park_l602_602183


namespace integral_sqrt_circle_l602_602277

theorem integral_sqrt_circle :
  (∫ x in 2..3, real.sqrt (1 - (x - 3)^2)) = (real.pi / 4) :=
by
  sorry

end integral_sqrt_circle_l602_602277


namespace next_smallest_abundant_after_12_l602_602294

def is_abundant (n : ℕ) : Prop :=
  ∑ i in (Finset.range n).filter (λ d, n % d = 0), i > n

theorem next_smallest_abundant_after_12 : ∃ n, n > 12 ∧ is_abundant n ∧ ∀ m, 12 < m < n → ¬ is_abundant m :=
by
  use 18
  -- Proof that 18 is abundant goes here
  sorry

end next_smallest_abundant_after_12_l602_602294


namespace sum_of_coefficients_P_l602_602841

-- Define the polynomial P(x) as described in the conditions.
noncomputable def P (x : ℚ) : ℚ :=
  ∏ (r : ℚ) in finset.range (1000 - 2 + 1) + 2, (x^2 - 1 / r)

-- The main theorem we want to prove
theorem sum_of_coefficients_P : P 1 = 1/1000 :=
by
  -- We skip the proof for now
  sorry

end sum_of_coefficients_P_l602_602841


namespace soap_cost_in_two_years_l602_602661

theorem soap_cost_in_two_years (bars_per_month : ℕ) (cost_per_bar : ℕ) (months_in_year: ℕ) (years : ℕ) :
  bars_per_month = 1 → 
  cost_per_bar = 4 → 
  months_in_year = 12 → 
  years = 2 → 
  (bars_per_month * months_in_year * years * cost_per_bar) = 96 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  simp
  sorry

end soap_cost_in_two_years_l602_602661


namespace point_in_fourth_quadrant_l602_602383

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a^2 + 1 > 0) (h2 : -1 - b^2 < 0) : 
  (a^2 + 1 > 0 ∧ -1 - b^2 < 0) ∧ (0 < a^2 + 1) ∧ (-1 - b^2 < 0) :=
by
  sorry

end point_in_fourth_quadrant_l602_602383


namespace total_students_in_fifth_grade_l602_602046

theorem total_students_in_fifth_grade :
  let c1 := 42 in
  let c2 := (42 * 6) / 7 in
  let c3 := (c2 * 5) / 6 in
  let c4 := c3 * 1.2 in
  c1 + c2 + c3 + c4 = 144 :=
by
  let c1 := 42
  let c2 := (42 * 6) / 7
  let c3 := (c2 * 5) / 6
  let c4 := c3 * 1.2
  have h1 : c2 = 36 := by sorry
  have h2 : c3 = 30 := by sorry
  have h3 : c4 = 36 := by sorry
  calc
    c1 + c2 + c3 + c4 = 42 + 36 + 30 + 36 : by sorry
                  ... = 144 : by sorry

end total_students_in_fifth_grade_l602_602046


namespace two_digit_number_is_91_l602_602208

/-- A positive two-digit number is odd and is a multiple of 13.
    The product of its digits is a perfect square.
    What is this two-digit number? -/
theorem two_digit_number_is_91 (M : ℕ) (h1 : M > 9) (h2 : M < 100) (h3 : M % 2 = 1) (h4 : M % 13 = 0) (h5 : ∃ n : ℕ, n * n = (M / 10) * (M % 10)) :
  M = 91 :=
sorry

end two_digit_number_is_91_l602_602208


namespace amount_on_table_A_l602_602069

-- Definitions based on conditions
variables (A B C : ℝ)
variables (h1 : B = 2 * C)
variables (h2 : C = A + 20)
variables (h3 : A + B + C = 220)

-- Theorem statement
theorem amount_on_table_A : A = 40 :=
by
  -- This is expected to be filled in with the proof steps, but we skip it with 'sorry'
  sorry

end amount_on_table_A_l602_602069


namespace solve_for_x_l602_602285

noncomputable def x : ℝ := 64

theorem solve_for_x (x : ℝ) (h : 9^(Real.log 8 x) = 81) : x = 64 := by
  sorry

end solve_for_x_l602_602285


namespace path_count_correct_l602_602646

-- Define the points A, B, C, D
def Point : Type := ℕ

-- Define the specific points A, B, C, D
def A : Point := 0
def B : Point := 1
def C : Point := 2
def D : Point := 3

-- Define paths between points
def paths (p1 p2 : Point) : ℕ :=
  if (p1 = A ∧ p2 = B) ∨ (p1 = B ∧ p2 = D) ∨ (p1 = D ∧ p2 = C) then 2
  else if p1 = A ∧ p2 = C then 1
  else 0

-- Define the problem statement in terms of a proof
theorem path_count_correct :
  paths A C = 9 :=
by
  sorry

end path_count_correct_l602_602646


namespace find_position_of_2017_l602_602243

theorem find_position_of_2017 :
  ∃ (row col : ℕ), row = 45 ∧ col = 81 ∧ 2017 = (row - 1)^2 + col :=
by
  sorry

end find_position_of_2017_l602_602243


namespace mean_difference_l602_602075

theorem mean_difference (T : ℝ) : 
  let mean_actual := (T + 120000) / 500 in
  let mean_incorrect := (T + 1200000) / 500 in
  mean_incorrect - mean_actual = 2160 :=
by
  sorry

end mean_difference_l602_602075


namespace minimum_value_inverse_sum_l602_602732

theorem minimum_value_inverse_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∃ (minimum : ℝ), minimum = 4 ∧ (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → (1 / x + 1 / y) ≥ minimum) :=
by
  use 4
  split
  { refl }
  sorry

end minimum_value_inverse_sum_l602_602732


namespace infinite_product_tends_to_zero_l602_602652

def seq (a : ℕ → ℝ) : Prop :=
  a 0 = 1/3 ∧ ∀ n, a (n+1) = 1 + 2 * (a n - 1)^2

theorem infinite_product_tends_to_zero (a : ℕ → ℝ) (h : seq a) :
  tendsto (λ n, ∏ i in finset.range (n+1), a i) at_top (𝓝 0) :=
sorry

end infinite_product_tends_to_zero_l602_602652


namespace compound_interest_equals_l602_602374

-- Definitions coming directly from the conditions:
def principal : ℝ := 8000
def simple_interest : ℝ := 800
def years : ℕ := 2
def rate_simple_interest (P SI T : ℝ) : ℝ := (SI * 100) / (P * T)

-- Derived Rate from given conditions:
def annual_rate : ℝ := rate_simple_interest principal simple_interest years

-- Compound Interest for given duration and rate:
def compound_interest (P R : ℝ) (T : ℕ) : ℝ := 
  P * ((1 + R / 100) ^ T - 1)

-- Final Property to Prove:
theorem compound_interest_equals :
  compound_interest principal annual_rate years = 820 :=
by
  -- Proof is skipped with sorry.
  sorry

end compound_interest_equals_l602_602374


namespace equilateral_triangle_l602_602883

theorem equilateral_triangle
  (A B C : Type)
  (angle_A : ℝ)
  (side_BC : ℝ)
  (perimeter : ℝ)
  (h1 : angle_A = 60)
  (h2 : side_BC = 1/3 * perimeter)
  (side_AB : ℝ)
  (side_AC : ℝ)
  (h3 : perimeter = side_BC + side_AB + side_AC) :
  (side_AB = side_BC) ∧ (side_AC = side_BC) :=
by
  sorry

end equilateral_triangle_l602_602883


namespace simplify_trig_identity_l602_602481

theorem simplify_trig_identity (x y : ℝ) : 
  sin (x + y) * cos y - cos (x + y) * sin y = sin x :=
by 
  sorry

end simplify_trig_identity_l602_602481


namespace piecewise_eq_solution_count_l602_602679

theorem piecewise_eq_solution_count :
  (∃ f : ℝ → ℝ, ∀ x, f x = (if x ≤ 150 then x / 150 - sin x else -x / 150 - sin x) ∧ 
  ∀ g : ℝ → ℝ, ∀ I : set ℝ, I = Icc (-300) 300 ∧ 
  (∃ y, count_root f (-300) 300 y = 91)) :=
sorry

end piecewise_eq_solution_count_l602_602679


namespace problem_statement_l602_602831

-- Defining the real numbers and the hypothesis
variables {a b c x y z : ℝ}
variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 31 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

-- State the theorem
theorem problem_statement : 
  (a / (a - 17) + b / (b - 31) + c / (c - 53) = 1) :=
by
  sorry

end problem_statement_l602_602831


namespace tom_sold_price_l602_602119

noncomputable def original_price : ℝ := 200
noncomputable def tripled_price (price : ℝ) : ℝ := 3 * price
noncomputable def sold_price (price : ℝ) : ℝ := 0.4 * price

theorem tom_sold_price : sold_price (tripled_price original_price) = 240 := 
by
  sorry

end tom_sold_price_l602_602119


namespace dwayne_yearly_earnings_l602_602635

variables (D B : ℝ)
variable h1 : B = D + 450
variable h2 : D + B = 3450

theorem dwayne_yearly_earnings (D B : ℝ) (h1 : B = D + 450) (h2 : D + B = 3450) : D = 1500 :=
by 
  sorry

end dwayne_yearly_earnings_l602_602635


namespace white_tiles_in_square_l602_602559

theorem white_tiles_in_square :
  ∀ (n : ℕ), (n * n = 81) → (n ^ 2 - (2 * n - 1)) = 6480 :=
by
  intro n
  intro hn
  sorry

end white_tiles_in_square_l602_602559


namespace decrease_is_75_86_percent_l602_602817

noncomputable def decrease_percent (x y z : ℝ) : ℝ :=
  let x' := 0.8 * x
  let y' := 0.75 * y
  let z' := 0.9 * z
  let original_value := x^2 * y^3 * z
  let new_value := (x')^2 * (y')^3 * z'
  let decrease_value := original_value - new_value
  decrease_value / original_value

theorem decrease_is_75_86_percent (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  decrease_percent x y z = 0.7586 :=
sorry

end decrease_is_75_86_percent_l602_602817


namespace problem_part_a_problem_part_b_l602_602167

noncomputable def probability_peter_satisfied : ℚ := 25 / 33

noncomputable def expected_number_satisfied_men : ℚ := 1250 / 33

theorem problem_part_a (total_men total_women : ℕ) (h1 : total_men = 50) (h2 : total_women = 50):
  let peter_satisfied := probability_peter_satisfied in
  let prob := λ m w, 1 - ((m / (m + w - 1)) * ((m - 1) / (m + w - 2))) in
  peter_satisfied = prob (total_men - 1) total_women := 
by {
  dsimp [peter_satisfied, prob],
  rw [h1, h2],
  unfold_coes,
  norm_num,
  sorry
}

theorem problem_part_b (total_men total_women : ℕ) (h1 : total_men = 50) (h2 : total_women = 50):
  let satisfied_probability := probability_peter_satisfied in
  let expected_satisfied_men := expected_number_satisfied_men in
  expected_satisfied_men = total_men * satisfied_probability := 
by {
  dsimp [satisfied_probability, expected_satisfied_men],
  rw [h1, h2],
  unfold_coes,
  norm_num,
  sorry
}

end problem_part_a_problem_part_b_l602_602167


namespace area_of_right_triangle_l602_602044

def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let v1 := (B.1 - A.1, B.2 - A.2)
  let v2 := (C.1 - A.1, C.2 - A.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def is_hypotenuse_length (A B : ℝ × ℝ) (length : ℝ) : Prop :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = length

def median_line (A B G : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_right_triangle :
  ∀ (A B C : ℝ × ℝ),
    is_right_triangle A B C →
    is_hypotenuse_length A B 50 →
    median_line ((A.1 + C.1) / 2, (A.2 + C.2) / 2) ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ((0, 0)) →
    area_of_triangle A B C = 1250 :=
by
  intros A B C h_right h_hypotenuse h_median
  sorry

end area_of_right_triangle_l602_602044


namespace area_of_triangle_ABC_l602_602391

noncomputable def triangle_area 
  (a b c: ℝ) 
  (A B C: ℝ) 
  (angle_sum: A + B + C = Real.pi) 
  (arithmetic_sequence: A + C = 2 * B) 
  (sides: a = 1 ∧ b = Real.sqrt 3) 
  (sine_rule: a / Real.sin A = b / Real.sin B) : Real :=
  (1 / 2) * a * b

theorem area_of_triangle_ABC : 
  ∃ (a b c: ℝ) (A B C: ℝ), 
  A + B + C = Real.pi ∧ 
  A + C = 2 * B ∧ 
  a = 1 ∧ 
  b = Real.sqrt 3 ∧ 
  (∃ (sine_rule: a / Real.sin A = b / Real.sin B), 
  triangle_area a b c A B C (by sorry) (by sorry) (by sorry) sine_rule = Real.sqrt 3 / 2) :=
begin
  use [1, Real.sqrt 3, 1, Real.pi / 6, Real.pi / 3, Real.pi / 2],
  split,
  { sorry },
  split,
  { sorry },
  split,
  { refl },
  split,
  { refl },
  use (by sorry),
  sorry,
end

end area_of_triangle_ABC_l602_602391


namespace train_length_is_110_l602_602222

-- Definitions based on conditions
def speed_kmh : ℝ := 45  -- Speed of the train in km/hr
def crossing_time : ℝ := 30  -- Time to cross the bridge in seconds
def bridge_length : ℝ := 265  -- Length of the bridge in meters

-- Convert speed from km/hr to m/s
def speed_ms : ℝ := speed_kmh * 1000 / 3600

-- Define the total distance covered in terms of meters
def total_distance : ℝ := speed_ms * crossing_time

-- Define the length of the train
def train_length : ℝ := total_distance - bridge_length

-- Theorem stating that the length of the train is 110 meters
theorem train_length_is_110 : train_length = 110 :=
  sorry

end train_length_is_110_l602_602222


namespace one_point_one_billion_scientific_notation_l602_602498

theorem one_point_one_billion_scientific_notation :
  ∃ (n : ℝ), n = 1.1 * 10^9 ∧ scientific_notation 1.1e9 n :=
sorry

end one_point_one_billion_scientific_notation_l602_602498


namespace common_tangent_slope_l602_602782

theorem common_tangent_slope (a m : ℝ) : 
  ((∃ a, ∃ m, l = (2 * a) ∧ l = (3 * m^2) ∧ a^2 = 2 * m^3) → (l = 0 ∨ l = 64 / 27)) := 
sorry

end common_tangent_slope_l602_602782


namespace solve_sqrt_eq_l602_602673

theorem solve_sqrt_eq (z : ℚ) (h : Real.sqrt (5 - 4 * z) = 10) : z = -95 / 4 := by
  sorry

end solve_sqrt_eq_l602_602673


namespace triangle_inequality_angle_bisector_l602_602062

theorem triangle_inequality_angle_bisector
  (AB AC BC BD CD : ℝ)
  (AB_length : AB = 8)
  (CD_length : CD = 4)
  (angle_bisector : BD / CD = AB / AC) :
  let m := 4, n := 16 in m + n = 20 :=
by
  let x := AC,
  have h₁ : 2 < x ∧ x < 16 := sorry,
  have h₂ : x > 4 := sorry,
  sorry

end triangle_inequality_angle_bisector_l602_602062


namespace square_perimeter_l602_602605

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end square_perimeter_l602_602605


namespace shortest_distance_l602_602580

-- The initial position of the cowboy.
def initial_position : ℝ × ℝ := (-2, -6)

-- The position of the cabin relative to the cowboy's initial position.
def cabin_position : ℝ × ℝ := (10, -15)

-- The equation of the stream flowing due northeast.
def stream_equation : ℝ → ℝ := id  -- y = x

-- Function to calculate the distance between two points (x1, y1) and (x2, y2).
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Calculate the reflection point of C over y = x.
def reflection_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Main proof statement: shortest distance the cowboy can travel.
theorem shortest_distance : distance initial_position (reflection_point initial_position) +
                            distance (reflection_point initial_position) cabin_position = 8 +
                            Real.sqrt 545 :=
by
  sorry

end shortest_distance_l602_602580


namespace exists_nonzero_integers_l602_602705

theorem exists_nonzero_integers (m n : ℕ) (hm : 2 ≤ m) (hn : 2 ≤ n)
  (a : ℕ → ℕ) (h_non_multiple : ∀ i : ℕ, i < n → ¬ m ^ (n - 1) ∣ a i) :
  ∃ (e : ℕ → ℕ), (∀ i, i < n → e i ≠ 0 ∧ e i < m) ∧ 
    (m ^ n ∣ (Σ i, e i * a i)) :=
by
  sorry

end exists_nonzero_integers_l602_602705


namespace peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l602_602174

-- Define the context of the problem
def total_people := 100
def men := 50
def women := 50

-- Define Peter Ivanovich being satisfied
def satisfies_peter_ivanovich := (women / (total_people - 1)) * ((women - 1) / (total_people - 2)) 

-- Define the probability that Peter Ivanovich is satisfied
theorem peter_ivanovich_satisfied_probability :
  satisfies_peter_ivanovich = 25 / 33 := 
sorry

-- Define the expected number of satisfied men
def expected_satisfied_men := men * (25 / 33)

-- Prove the expected number of satisfied men
theorem expected_satisfied_men_value :
  expected_satisfied_men = 1250 / 33 :=
sorry

end peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l602_602174


namespace cheese_loss_ratio_l602_602884

/- Define the conditions -/
structure CheeseShapes where
  pentagonalPrismVertices : Nat := 10
  squarePyramidVertices : Nat := 5

def volumeEaten (vertices : Nat) : ℝ :=
  vertices * (4 * π / 3)

/- Statement of the problem -/
theorem cheese_loss_ratio
  (shapes : CheeseShapes)
  (V_prism : volumeEaten shapes.pentagonalPrismVertices = (40 * π / 3))
  (V_pyramid : volumeEaten shapes.squarePyramidVertices = (20 * π / 3)) :
  volumeEaten shapes.pentagonalPrismVertices = 2 * volumeEaten shapes.squarePyramidVertices := by
  sorry

end cheese_loss_ratio_l602_602884


namespace chocolate_cost_is_120_l602_602960

noncomputable def candy_cost : ℕ := 530
noncomputable def num_candies : ℕ := 12
noncomputable def cost_difference : ℕ := 5400
noncomputable def num_chocolates : ℕ := 8

theorem chocolate_cost_is_120 :
  let total_candy_cost := num_candies * candy_cost in
  let total_chocolate_cost := total_candy_cost - cost_difference in
  total_chocolate_cost / num_chocolates = 120 :=
by
  sorry

end chocolate_cost_is_120_l602_602960


namespace percent_of_b_is_40_l602_602377

variables {a c b : ℝ}

def c_is_14_percent_of_a := c = 0.14 * a
def b_is_35_percent_of_a := b = 0.35 * a
def x_percent_of_b := c = (40 / 100) * b

theorem percent_of_b_is_40
  (h1 : c_is_14_percent_of_a)
  (h2 : b_is_35_percent_of_a) :
  x_percent_of_b :=
by
  sorry -- skipping the proof part as per instructions

end percent_of_b_is_40_l602_602377


namespace javier_average_hits_per_game_l602_602008

theorem javier_average_hits_per_game (total_games_first_part : ℕ) (average_hits_first_part : ℕ) 
  (remaining_games : ℕ) (average_hits_remaining : ℕ) : 
  total_games_first_part = 20 → average_hits_first_part = 2 → 
  remaining_games = 10 → average_hits_remaining = 5 →
  (total_games_first_part * average_hits_first_part + 
  remaining_games * average_hits_remaining) /
  (total_games_first_part + remaining_games) = 3 := 
by intros h1 h2 h3 h4;
   sorry

end javier_average_hits_per_game_l602_602008


namespace smaller_angle_of_parallelogram_l602_602389

theorem smaller_angle_of_parallelogram (x : ℝ) (h : x + 3 * x = 180) : x = 45 :=
sorry

end smaller_angle_of_parallelogram_l602_602389


namespace mutually_exclusive_events_l602_602186

theorem mutually_exclusive_events (red white black : ℕ) (h_red : red = 3) (h_white : white = 2) (h_black : black = 1) :
  ∀ events : set (set (ℕ × ℕ)),
    (events = { {(r, w) | r + w = 2 ∧ (r > 0 ∨ w > 0)}, 
                {(r, w) | r = 1 ∧ w = 0 ∧ (1 ≤ r ∧ r < 3)} }) →
    ∀ ev1 ev2 ∈ events, ev1 ≠ ev2 →
    (ev1 ∩ ev2 = ∅)  -- mutually exclusive
    → (∀ s ∈ ev1, ¬(s ∈ ev2) ∧ ∀ t ∈ ev2, ¬(t ∈ ev1)) -- not contradictory (i.e., they don't contradict each other logically)
    → sorry :=
by intros;
   apply sorry

end mutually_exclusive_events_l602_602186


namespace remainder_g10_div_g_l602_602035

-- Conditions/Definitions
def g (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1
def g10 (x : ℝ) : ℝ := (g (x^10))

-- Theorem/Question
theorem remainder_g10_div_g : (g10 x) % (g x) = 6 :=
by
  sorry

end remainder_g10_div_g_l602_602035


namespace sqrt_evaluation_l602_602641

theorem sqrt_evaluation : sqrt 9 + sqrt 25 - sqrt (1/4) = 15 / 2 := by
  sorry

end sqrt_evaluation_l602_602641


namespace problem_part_a_problem_part_b_l602_602169

noncomputable def probability_peter_satisfied : ℚ := 25 / 33

noncomputable def expected_number_satisfied_men : ℚ := 1250 / 33

theorem problem_part_a (total_men total_women : ℕ) (h1 : total_men = 50) (h2 : total_women = 50):
  let peter_satisfied := probability_peter_satisfied in
  let prob := λ m w, 1 - ((m / (m + w - 1)) * ((m - 1) / (m + w - 2))) in
  peter_satisfied = prob (total_men - 1) total_women := 
by {
  dsimp [peter_satisfied, prob],
  rw [h1, h2],
  unfold_coes,
  norm_num,
  sorry
}

theorem problem_part_b (total_men total_women : ℕ) (h1 : total_men = 50) (h2 : total_women = 50):
  let satisfied_probability := probability_peter_satisfied in
  let expected_satisfied_men := expected_number_satisfied_men in
  expected_satisfied_men = total_men * satisfied_probability := 
by {
  dsimp [satisfied_probability, expected_satisfied_men],
  rw [h1, h2],
  unfold_coes,
  norm_num,
  sorry
}

end problem_part_a_problem_part_b_l602_602169


namespace range_sin_interval_l602_602516

theorem range_sin_interval : 
  let f := fun x : ℝ => Real.sin x in
  let a := -Real.pi / 4 in
  let b := 3 * Real.pi / 4 in
  set.range (λ x, f x) (set.Icc a b) = set.Icc (-(Real.sqrt 2) / 2) 1 := 
sorry

end range_sin_interval_l602_602516


namespace flower_seedlings_pots_l602_602191

theorem flower_seedlings_pots (x y z : ℕ) :
  (1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z) →
  (x + y + z = 16) →
  (2 * x + 4 * y + 10 * z = 50) →
  (x = 10 ∨ x = 13) :=
by
  intros h1 h2 h3
  sorry

end flower_seedlings_pots_l602_602191


namespace father_children_problem_l602_602074

theorem father_children_problem {F C n : ℕ} 
  (hF_C : F = C) 
  (sum_ages_after_15_years : C + 15 * n = 2 * (F + 15)) 
  (father_age : F = 75) : 
  n = 7 :=
by
  sorry

end father_children_problem_l602_602074


namespace square_perimeter_l602_602612

theorem square_perimeter (area : ℝ) (h : area = 625) : 
  let s := Real.sqrt area in
  (4 * s) = 100 :=
by
  let s := Real.sqrt area
  have hs : s = 25 := by sorry
  calc
    (4 * s) = 4 * 25 : by rw hs
          ... = 100   : by norm_num

end square_perimeter_l602_602612


namespace area_of_hexagon_l602_602028

open Classical

noncomputable def hexagonArea (A B C D E F : Type) (mid_AB mid_CD mid_EF : Type) (area_JKL : ℝ) : ℝ :=
if is_regular_hexagon : Prop := True then
  if JKL_triangle_area : area_JKL = 100 then
    area_hexagon = 400
  else
    sorry
else
  sorry

theorem area_of_hexagon (A B C D E F : Type) (mid_AB mid_CD mid_EF : Type) (area_JKL : ℝ)
  (h1 : RegularHexagon A B C D E F)
  (h2 : Midpoints A B mid_AB ∧ Midpoints C D mid_CD ∧ Midpoints E F mid_EF)
  (h3 : TriangleArea mid_AB mid_CD mid_EF = 100) : 
  HexagonArea A B C D E F = 400 :=
begin
  sorry
end

end area_of_hexagon_l602_602028


namespace find_sum_12_terms_of_sequence_l602_602801

variable {a : ℕ → ℕ}

def geometric_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

def is_periodic_sequence (a : ℕ → ℕ) (period : ℕ) : Prop :=
  ∀ n : ℕ, a n = a (n + period)

noncomputable def given_sequence : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => (given_sequence n * given_sequence (n + 1) / 4) -- This should ensure periodic sequence of period 3 given a common product of 8 and simplifying the product equation.

theorem find_sum_12_terms_of_sequence :
  geometric_sequence given_sequence 8 ∧ given_sequence 0 = 1 ∧ given_sequence 1 = 2 →
  (Finset.range 12).sum given_sequence = 28 :=
by
  sorry

end find_sum_12_terms_of_sequence_l602_602801


namespace probability_of_3_rainy_days_l602_602785

noncomputable def rainy_probability : ℝ :=
  let n := 4
  let k := 3
  let p := 0.5
  binomial_pdf p n k

theorem probability_of_3_rainy_days : rainy_probability = 0.25 := by
  sorry

end probability_of_3_rainy_days_l602_602785


namespace lcm_9_12_15_l602_602994

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l602_602994


namespace power_function_C_power_function_D_l602_602149

theorem power_function_C (a : ℝ) (h : a > 0) :
  (0 : ℝ)^a = 0 ∧ (1 : ℝ)^a = 1 :=
by
  split
  · exact zero_rpow h
  · exact one_rpow _

theorem power_function_D (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) :
  (sqrt x1 + sqrt x2) / 2 ≤ sqrt ((x1 + x2) / 2) :=
by
  have key : 0 ≤ (sqrt x1 - sqrt x2)^2 := sq_nonneg _
  rw [sub_sq, add_sq, two_mul, sq_sqrt h1, sq_sqrt h2] at key
  linarith

#print axioms power_function_C
#print axioms power_function_D

end power_function_C_power_function_D_l602_602149


namespace number_of_squares_is_70_l602_602615

def countDifferentSquares (m n : ℕ) : ℕ :=
  (∑ k in Finset.range (min m n + 1), (m - k) * (n - k))

theorem number_of_squares_is_70 : countDifferentSquares 5 6 = 70 :=
  by
    sorry

end number_of_squares_is_70_l602_602615


namespace minimum_diagonal_of_rectangle_l602_602589

theorem minimum_diagonal_of_rectangle (P Q R S : Point) (d : ℝ) (hPQRSP : rectangle P Q R S) (h_perimeter : ∀ p q r s : ℝ, 2 * (p + r) = 24) (h_PQ : PQ = 7) :
  PR = sqrt 74 :=
by
  sorry

end minimum_diagonal_of_rectangle_l602_602589


namespace range_a_l602_602432

open Set Real

-- Define the predicate p: real number x satisfies x^2 - 4ax + 3a^2 < 0, where a < 0
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

-- Define the predicate q: real number x satisfies x^2 - x - 6 ≤ 0, or x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the complement sets
def not_p_set (a : ℝ) : Set ℝ := {x | ¬p x a}
def not_q_set : Set ℝ := {x | ¬q x}

-- Define p as necessary but not sufficient condition for q
def necessary_but_not_sufficient (a : ℝ) : Prop := 
  (not_q_set ⊆ not_p_set a) ∧ ¬(not_p_set a ⊆ not_q_set)

-- The main theorem to prove
theorem range_a : {a : ℝ | necessary_but_not_sufficient a} = {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4} :=
by
  sorry

end range_a_l602_602432


namespace find_largest_beta_l602_602818

theorem find_largest_beta (α : ℝ) (r : ℕ → ℝ) (C : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < 1)
  (h3 : ∀ n, ∀ m ≠ n, dist (r n) (r m) ≥ (r n) ^ α)
  (h4 : ∀ n, r n ≤ r (n + 1)) 
  (h5 : ∀ n, r n ≥ C * n ^ (1 / (2 * (1 - α)))) :
  ∀ β, (∃ C > 0, ∀ n, r n ≥ C * n ^ β) → β ≤ 1 / (2 * (1 - α)) :=
sorry

end find_largest_beta_l602_602818


namespace find_valid_numbers_l602_602683

def is_distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

def digits_in_range (a b c : ℕ) : Prop := 
  ∀ x, x ∈ {a, b, c} → 0 ≤ x ∧ x ≤ 9

def num_from_digits (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

def arithmetic_mean_condition (a b c : ℕ) : Prop :=
  num_from_digits a b c = (num_from_digits b c a + num_from_digits c a b) / 2

theorem find_valid_numbers (a b c : ℕ) :
  digits_in_range a b c → is_distinct a b c → arithmetic_mean_condition a b c →
  (num_from_digits a b c = 592 ∨ num_from_digits a b c = 481 ∨ 
   num_from_digits a b c = 370 ∨ num_from_digits a b c = 629 ∨ 
   num_from_digits a b c = 518 ∨ num_from_digits a b c = 407) :=
sorry

end find_valid_numbers_l602_602683


namespace quadrilateral_parallelogram_and_single_point_exists_l602_602408

-- Define the setup for the quadrilateral ABCD and the point P
variables {A B C D P : Type} [Plane A] [Plane B] [Plane C] [Plane D] [Plane P]
variable (ABCD : ConvexQuadrilateral A B C D)

-- Define the areas of the triangles
variables (areaABP areaBCP areaCDP areaDAP : ℝ)
variable (S : ℝ)

-- Conditions given in the problem
axiom equal_areas : 
  areaABP = areaBCP ∧ 
  areaBCP = areaCDP ∧ 
  areaCDP = areaDAP ∧ 
  areaABP = S

-- Prove that ABCD is a parallelogram and there is exactly one such point P
theorem quadrilateral_parallelogram_and_single_point_exists 
  (h : areaABP = S ∧ areaBCP = S ∧ areaCDP = S ∧ areaDAP = S) : 
  isParallelogram ABCD ∧ ∃! P, P ∈ planeABCD ∧ 
  (area (triangle A B P) = S ∧ area (triangle B C P) = S ∧ 
   area (triangle C D P) = S ∧ area (triangle D A P) = S) :=
sorry

end quadrilateral_parallelogram_and_single_point_exists_l602_602408


namespace length_of_train_l602_602226

def speed_km_per_hr := 45
def time_to_cross_secs := 30
def bridge_length_meters := 265

theorem length_of_train :
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600 in
  let total_distance := speed_m_per_s * time_to_cross_secs in
  let train_length := total_distance - bridge_length_meters in
  train_length = 110 :=
by
  sorry

end length_of_train_l602_602226


namespace value_preserving_interval_of_g_l602_602716

noncomputable def g (x : ℝ) (m : ℝ) : ℝ :=
  x + m - Real.log x

theorem value_preserving_interval_of_g
  (m : ℝ)
  (h_increasing : ∀ x, x ∈ Set.Ici 2 → 1 - 1 / x > 0)
  (h_range : ∀ y, y ∈ Set.Ici 2): 
  (2 + m - Real.log 2 = 2) → 
  m = Real.log 2 :=
by 
  sorry

end value_preserving_interval_of_g_l602_602716


namespace find_constants_l602_602290

variables {A B C x : ℝ}

theorem find_constants (h : (A = 6) ∧ (B = -5) ∧ (C = 5)) :
  (x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1) :=
by sorry

end find_constants_l602_602290


namespace polygon_is_isosceles_triangle_l602_602368

noncomputable def is_isosceles_triangle (p1 p2 p3 : (ℝ × ℝ)) : Prop :=
  let dist := λ (a b : (ℝ × ℝ)), Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)
  in (dist p1 p2 = dist p1 p3 ∨ dist p1 p2 = dist p2 p3 ∨ dist p1 p3 = dist p2 p3)

theorem polygon_is_isosceles_triangle : 
  ∃ (A B C : (ℝ × ℝ)), 
  A = (0, 3) ∧ B = (1/2, 4) ∧ C = (-1/2, 4) ∧ is_isosceles_triangle A B C := by
  sorry

end polygon_is_isosceles_triangle_l602_602368


namespace sine_subtraction_formula_simplify_expression_l602_602488

-- Define the sine subtraction formula as a condition
theorem sine_subtraction_formula (a b : ℝ) : 
    real.sin (a - b) = real.sin a * real.cos b - real.cos a * real.sin b := by
  sorry

-- Prove the given expression simplifies to sin(x)
theorem simplify_expression (x y : ℝ) :
    real.sin (x + y) * real.cos y - real.cos (x + y) * real.sin y = real.sin x := by
  have h : real.sin ((x + y) - y) = real.sin (x + y) * real.cos y - real.cos (x + y) * real.sin y := by
    exact sine_subtraction_formula (x + y) y
  rw [sub_self y, h]
  simp
  sorry

end sine_subtraction_formula_simplify_expression_l602_602488


namespace erica_pie_percentage_l602_602663

theorem erica_pie_percentage (a c : ℚ) (ha : a = 1/5) (hc : c = 3/4) : 
  (a + c) * 100 = 95 := 
sorry

end erica_pie_percentage_l602_602663


namespace hueys_pizza_problem_l602_602366

noncomputable def side_length_small_pizza : ℝ :=
  let s := (60 / 20) * 144 / (3 + 1.5) - 16 / 3
  sqrt s

theorem hueys_pizza_problem :
  ∃ s : ℝ, s ≈ 8.16 ∧ s = side_length_small_pizza :=
by
  use side_length_small_pizza
  split
  . apply real.sqrt_eq_approx
    norm_num
  . rfl

end hueys_pizza_problem_l602_602366


namespace b_plus_d_value_l602_602799

noncomputable def calculate_b_plus_d : ℝ :=
  let r_c := 15
  let h_c := 30
  let cone_base_to_apex := Real.sqrt (r_c^2 + h_c^2)
  let sphere_radius_expression := λ b d, b * Real.sqrt d - b
  let r := 7.5 * (Real.sqrt 5 - 1)
  let b := 7.5
  let d := 5
  b + d

theorem b_plus_d_value :
  calculate_b_plus_d = 12.5 := by {
    sorry
  }

end b_plus_d_value_l602_602799


namespace distinct_arrangements_count_l602_602586

theorem distinct_arrangements_count : 
  let women := 12
  let men := 3
  let slots := women + men - 1 -- because one slot is occupied by Mr. X
  let remaining_stools := men - 1 -- because one stool is occupied by Mr. X
  ∃ (count : ℕ), count = nat.choose slots remaining_stools ∧ count = 91 :=
by
  let women := 12
  let men := 3
  let slots := women + men - 1
  let remaining_stools := men - 1
  existsi nat.choose slots remaining_stools
  split
  . rfl
  . sorry

end distinct_arrangements_count_l602_602586


namespace unit_vector_parallel_to_2a_minus_3b_l602_602739

def vector (x y : ℝ) := (x, y)

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
(u.1 + v.1, u.2 + v.2)

def vector_scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
(c * v.1, c * v.2)

def vector_mag (v : ℝ × ℝ) : ℝ :=
real.sqrt (v.1 * v.1 + v.2 * v.2)

def vector_unit (v : ℝ × ℝ) : ℝ × ℝ :=
(v.1 / vector_mag v, v.2 / vector_mag v)

theorem unit_vector_parallel_to_2a_minus_3b :
  let a := vector 5 4 in
  let b := vector 3 2 in
  let v := vector_add (vector_scalar_mul 2 a) (vector_scalar_mul (-3) b) in
  let u := vector_unit v in
  u = (1 / real.sqrt 5, 2 / real.sqrt 5) ∨ u = (-1 / real.sqrt 5, -2 / real.sqrt 5) :=
sorry

end unit_vector_parallel_to_2a_minus_3b_l602_602739


namespace monotonic_intervals_harmonic_log_inequality_l602_602729

noncomputable theory
open Real

-- Define the function f(x) as given
def f (x : ℝ) (a : ℝ) : ℝ := log x - (a*x - 1) / x

-- Define the statement for monotonic intervals
theorem monotonic_intervals (a : ℝ) :
  (∀ x > 1, 0 < (f x a)) ∧ (∀ x, 0 < x < 1 → 0 > (f x a)) :=
sorry

-- Define the harmonic function H(n) and its bounds
def H (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), (1 : ℝ) / (k + 1)

-- Prove the inequality for the harmonic sum and natural logarithm
theorem harmonic_log_inequality (n : ℕ) (h : 0 < n) :
  H n - 1 < log (n + 1) ∧ log (n + 1) < 1 + H n :=
sorry

end monotonic_intervals_harmonic_log_inequality_l602_602729


namespace willy_stuffed_animals_l602_602549

theorem willy_stuffed_animals :
  ∀ (initial mom dad : ℕ), 
  initial = 10 → mom = 2 → dad = 3 →
  let after_mom := initial + mom in
  let after_dad := after_mom + (dad * after_mom) in
  after_dad = 48 :=
by
  intros initial mom dad h1 h2 h3 after_mom after_dad
  sorry

end willy_stuffed_animals_l602_602549


namespace sum_ab_eq_five_l602_602932

theorem sum_ab_eq_five (a b : ℕ) (h : (∃ (ab : ℕ), ab = a * 10 + b ∧ 3 / 13 = ab / 100)) : a + b = 5 :=
sorry

end sum_ab_eq_five_l602_602932


namespace exists_sequence_l602_602893

theorem exists_sequence (x : ℕ → ℕ) : 
  -- Every positive integer appears exactly once
  (∀ n : ℕ, ∃ m : ℕ, x m = n) ∧ 
  -- For every n, the partial sum is divisible by n^n
  (∀ n : ℕ, (∑ i in Finset.range (n+1), x i) % (n+1)^(n+1) = 0) :=
by sorry

end exists_sequence_l602_602893


namespace probability_of_letter_in_MATHEMATICAL_l602_602774

theorem probability_of_letter_in_MATHEMATICAL :
  let unique_letters_in_WORD := 8
  let total_letters := 26
  ∃ p : ℚ, p = unique_letters_in_WORD / total_letters ∧ p = 4 / 13 :=
by
  let unique_letters_in_WORD := 8
  let total_letters := 26
  have h : unique_letters_in_WORD / total_letters = 4 / 13,
  { calc
    (unique_letters_in_WORD / total_letters : ℚ)
        = 8 / 26 : by sorry
    ... = (4 : ℚ) / 13 : by sorry },
  use 4 / 13,
  exact ⟨rfl, h⟩

end probability_of_letter_in_MATHEMATICAL_l602_602774


namespace vasya_time_upward_l602_602743

noncomputable def time_to_run_up_down := 
  let length := 1
  let time_no_escalator := 6
  let time_downward := 13.5
  let x := 1 / (1 / 6 + 1 / (6 / 2)) -- Determine x from the 6 minutes equation
  let y := 1 / (13.5 * (1 / (1 / x + y) + 1 / (1 / (x / 2 - y)))) -- Solve for y using the 13.5 minutes equation
  (1 / (x - y) + 1 / (x / 2 + y)) * 60 -- Calculate time for upward case in seconds

theorem vasya_time_upward : time_to_run_up_down = 324 := 
  sorry

end vasya_time_upward_l602_602743


namespace find_z_l602_602653

variables (z : ℂ)

def det2x2 (a b c d : ℂ) : ℂ :=
  a * d - b * c

theorem find_z (h : det2x2 z complex.I 1 complex.I = 1 + complex.I) : 
  z = 2 - complex.I :=
by
  sorry

end find_z_l602_602653


namespace cost_of_paint_per_kg_l602_602504

/-- The cost of painting one square foot is Rs. 50. -/
theorem cost_of_paint_per_kg (side_length : ℝ) (cost_total : ℝ) (coverage_per_kg : ℝ) (total_surface_area : ℝ) (total_paint_needed : ℝ) (cost_per_kg : ℝ) 
  (h1 : side_length = 20)
  (h2 : cost_total = 6000)
  (h3 : coverage_per_kg = 20)
  (h4 : total_surface_area = 6 * side_length^2)
  (h5 : total_paint_needed = total_surface_area / coverage_per_kg)
  (h6 : cost_per_kg = cost_total / total_paint_needed) :
  cost_per_kg = 50 :=
sorry

end cost_of_paint_per_kg_l602_602504


namespace Jia_wins_l602_602967

def game_winner : Prop :=
  ∀ (n: ℕ) (Jia_turn: bool), 
    (n = 2 → Jia_turn = tt) →  -- Jia starts first with n initialized to 2
    (∀ d, d > 0 → d < n → n % d = 0 → d > n → false) →  -- d is a positive divisor of n less than n
    (n > 19891989 → false) →  -- Players lose if they write a number greater than 19891989
    (n < 19891989 → Jia_turn = tt)  -- Jia will always ensure Yi to write a number greater than 19891989 

theorem Jia_wins : game_winner := sorry

end Jia_wins_l602_602967


namespace inequality_solution_set_l602_602645

theorem inequality_solution_set {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_inc : ∀ {x y : ℝ}, 0 < x → x < y → f x ≤ f y)
  (h_value : f 1 = 0) :
  {x | (f x - f (-x)) / x ≤ 0} = {x | -1 ≤ x ∧ x < 0} ∪ {x | 0 < x ∧ x ≤ 1} :=
by
  sorry


end inequality_solution_set_l602_602645


namespace total_convertibles_count_l602_602239

variable (Vehicles Speedsters Roadsters Cruisers : ℕ)

theorem total_convertibles_count
  (h1 : 2/5 * Vehicles = Speedsters)
  (h2 : 3/10 * Vehicles = Roadsters)
  (h3 : Speedsters + Roadsters + Cruisers = Vehicles)
  (h4 : 4/5 * Speedsters = SpeedsterConvertibles)
  (h5 : 2/3 * Roadsters = RoadsterConvertibles)
  (h6 : 1/4 * Cruisers = CruiserConvertibles)
  (h7 : Vehicles - Speedsters = 60) :
  SpeedsterConvertibles + RoadsterConvertibles = 52 := by
  sorry

end total_convertibles_count_l602_602239


namespace interval_of_decrease_l602_602677

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem interval_of_decrease : { x : ℝ | -2 < x ∧ x < 0 } ⊆ { x : ℝ | ∃ (y : ℝ), has_deriv_at f y x ∧ 0 < y } → 
{ x : ℝ | -2 < x ∧ x < 0 } = { x : ℝ | f' x < 0 } :=
by 
  intro h
  sorry

end interval_of_decrease_l602_602677


namespace vertices_after_removal_l602_602215

theorem vertices_after_removal (a b : ℕ) (h₁ : a = 5) (h₂ : b = 2) : 
  let initial_vertices := 8
  let removed_vertices := initial_vertices
  let new_vertices := 8 * 9
  let final_vertices := new_vertices - removed_vertices
  final_vertices = 64 :=
by
  sorry

end vertices_after_removal_l602_602215


namespace product_of_integers_l602_602561

theorem product_of_integers (x y : ℕ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 40) : x * y = 99 :=
by {
  sorry
}

end product_of_integers_l602_602561


namespace peter_satisfied_probability_expected_satisfied_men_l602_602171

variable (numMen : ℕ) (numWomen : ℕ) (totalPeople : ℕ)
variable (peterSatisfiedProb : ℚ) (expectedSatisfiedMen : ℚ)

-- Conditions
def conditions_holds : Prop :=
  numMen = 50 ∧ numWomen = 50 ∧ totalPeople = 100 ∧ peterSatisfiedProb = 25 / 33 ∧ expectedSatisfiedMen = 1250 / 33

-- Prove the probability that Peter Ivanovich is satisfied.
theorem peter_satisfied_probability : conditions_holds → peterSatisfiedProb = 25 / 33 := by
  sorry

-- Prove the expected number of satisfied men.
theorem expected_satisfied_men : conditions_holds → expectedSatisfiedMen = 1250 / 33 := by
  sorry

end peter_satisfied_probability_expected_satisfied_men_l602_602171


namespace circle_representation_l602_602384

theorem circle_representation (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2 * m * x + 2 * m^2 + 2 * m - 3 = 0) ↔ m ∈ Set.Ioo (-3 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end circle_representation_l602_602384


namespace fixed_point_of_function_l602_602545

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^(2-2) - 3) = -2 :=
by
  sorry

end fixed_point_of_function_l602_602545


namespace find_number_l602_602457

theorem find_number (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
sorry

end find_number_l602_602457


namespace margo_walks_total_distance_l602_602449

theorem margo_walks_total_distance :
  let time_to_house := 15
  let time_to_return := 25
  let total_time_minutes := time_to_house + time_to_return
  let total_time_hours := (total_time_minutes : ℝ) / 60
  let avg_rate := 3  -- units: miles per hour
  (avg_rate * total_time_hours = 2) := 
sorry

end margo_walks_total_distance_l602_602449


namespace length_of_XY_l602_602407

theorem length_of_XY :
  ∀ (O A B : Point) (X Y : Point),
    is_sector O A B 45 ∧
    is_perpendicular (line O Y) (line A B) ∧
    intersects_at (line O Y) (line A B) X ∧
    distance O A = 12 ∧
    distance O B = 12 →
    distance X Y ≈ 0.9132 := by
  sorry

end length_of_XY_l602_602407


namespace find_k_l602_602309

noncomputable def k : ℝ :=
  let a := (2, 1)
  let b := (3, -2)
  let c k := (a.1 + k * b.1, a.2 + k * b.2)
  let d := (a.1 - b.1, a.2 - b.2)
  if h : (c k).1 * d.1 + (c k).2 * d.2 = 0 then k else 0

theorem find_k : k = 1 / 3 := by
  let a := (2, 1)
  let b := (3, -2)
  let d := (a.1 - b.1, a.2 - b.2)
  let c k := (a.1 + k * b.1, a.2 + k * b.2)
  have h1 : d = (-1, 3) := by simp [a, b]
  have h2 : c (1 / 3) = (1, -1 / 3) := by simp [a, b]
  have h3 : (1, -1 / 3).1 * (-1, 3).1 + (1, -1 / 3).2 * (-1, 3).2 = 0 := 
    by simp [c, d, a, b]; norm_num 
  show k = (1 / 3) from 
    by simp [h3]
  sorry

end find_k_l602_602309


namespace questionnaires_from_unit_D_l602_602881

variable (A B C D : ℕ)
variable h1 : A + B + C + D = 300
variable h2 : B = 60
variable h3 : 2 * B = A + C
variable h4 : B + B + B + 2 * (D - B) / 3 = D

theorem questionnaires_from_unit_D :
  D = 120 :=
by
  sorry

end questionnaires_from_unit_D_l602_602881


namespace cross_product_correct_l602_602291

open Matrix
open Real

-- Define the vectors u and v
def u : ℝ^3 := ![1, 2, 4]
def v : ℝ^3 := ![-3, 5, 6]

-- Define the expected cross product result
def w : ℝ^3 := ![-8, -18, 11]

-- The statement of the problem
theorem cross_product_correct :
  u × v = w :=
by sorry

end cross_product_correct_l602_602291


namespace fraction_sum_l602_602151

theorem fraction_sum : ((10 : ℚ) / 9 + (9 : ℚ) / 10 = 2.0 + (0.1 + 0.1 / 9)) :=
by sorry

end fraction_sum_l602_602151


namespace distinct_integer_sum_count_l602_602642

def is_special_fraction (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 18

def special_fractions : List (ℕ × ℕ) :=
  List.filter (λ (ab : ℕ × ℕ), is_special_fraction ab.1 ab.2)
    (List.product (List.range 18) (List.range 18))

def fractions_as_rationals : List ℚ :=
  special_fractions.map (λ (ab : ℕ × ℕ), (ab.1 : ℚ) / (ab.2 : ℚ))

def integer_sums (l : List ℚ) : List ℚ :=
  List.bind l (λ x, l.map (λ y, x + y))

def distinct_integer_sums : List ℚ :=
  (integer_sums fractions_as_rationals).filter (λ q, q.den = 1)

#eval distinct_integer_sums.length

theorem distinct_integer_sum_count : distinct_integer_sums.length = 3 := 
  sorry

end distinct_integer_sum_count_l602_602642


namespace conic_section_parabola_l602_602369

theorem conic_section_parabola (x y : ℝ) : 
  abs (x - 3) = sqrt ((x + 1)^2 + (y - 4)^2) → ∃ k a b, (y - a)^2 = k * (x - b) := 
sorry

end conic_section_parabola_l602_602369


namespace length_of_train_l602_602224

/-- The length of the train given that:
* it is traveling at 45 km/hr
* it crosses a bridge in 30 seconds
* the bridge is 265 meters long
is 110 meters.
-/
theorem length_of_train (speed : ℕ) (time : ℕ) (bridge_length : ℕ) (train_length : ℕ)
  (h1 : speed = 45) (h2 : time = 30) (h3 : bridge_length = 265) :
  (train_length : ℕ) = 110 :=
begin
  sorry
end

end length_of_train_l602_602224


namespace circumscribed_sphere_surface_area_l602_602337

-- Given Conditions
def area_rectangle (x y : ℝ) : Prop := x * y = 8
def minimized_perimeter (x y : ℝ) : Prop := 2 * (x + y) = 8 * sqrt 2

-- The theorem to be proven
theorem circumscribed_sphere_surface_area (x y : ℝ) 
  (h1 : area_rectangle x y) 
  (h2 : minimized_perimeter x y) : 
  (4 * real.pi * (2 ^ 2)) = 16 * real.pi := 
sorry

end circumscribed_sphere_surface_area_l602_602337


namespace concurent_lines_l602_602019

theorem concurent_lines 
    (A B C A' B' C' L M N P Q R : Type)
    [IncidencePlane A B C] [IncidencePlane A' B' C']
    (h1 : coplanar A B C A' B' C')
    (h2 : parallel AL BC) (h3 : parallel A'L B'C')
    (h4 : parallel BM CA) (h5 : parallel B'M C'A')
    (h6 : parallel CN AB) (h7 : parallel C'N A'B')
    (h8 : intersects BC B'C' P)
    (h9 : intersects CA C'A' Q)
    (h10 : intersects AB A'B' R)
    : concurrent PL QM RN :=
sorry

end concurent_lines_l602_602019


namespace minimum_travel_time_l602_602214

def cars := 31
def car_length := 5 -- in meters
def tunnel_length := 2725 -- in meters
def max_speed := 25 -- in m/s
def distance_between_cars (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then 20
  else if 12 < x ∧ x ≤ 25 then (1 / 6) * x^2 + (1 / 3) * x
  else 0

def y (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then 3480 / x
  else if 12 < x ∧ x ≤ 25 then 5 * x + 2880 / x + 10
  else 0

theorem minimum_travel_time :
  ∃ (x : ℝ), (0 < x ∧ x ≤ max_speed) ∧ y x = 250 :=
sorry

end minimum_travel_time_l602_602214


namespace range_of_a_l602_602776

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 5^x = a + 3) → a > -3 :=
by
  sorry

end range_of_a_l602_602776


namespace percentage_girls_not_attended_college_l602_602232

-- Definitions based on given conditions
def total_boys : ℕ := 300
def total_girls : ℕ := 240
def percent_boys_not_attended_college : ℚ := 0.30
def percent_class_attended_college : ℚ := 0.70

-- The goal is to prove that the percentage of girls who did not attend college is 30%
theorem percentage_girls_not_attended_college 
  (total_boys : ℕ)
  (total_girls : ℕ)
  (percent_boys_not_attended_college : ℚ)
  (percent_class_attended_college : ℚ)
  (total_students := total_boys + total_girls)
  (boys_not_attended := percent_boys_not_attended_college * total_boys)
  (students_attended := percent_class_attended_college * total_students)
  (students_not_attended := total_students - students_attended)
  (girls_not_attended := students_not_attended - boys_not_attended) :
  (girls_not_attended / total_girls) * 100 = 30 := 
  sorry

end percentage_girls_not_attended_college_l602_602232


namespace committee_with_at_least_one_boy_and_girl_l602_602922

def total_members : ℕ := 40
def boys : ℕ := 18
def girls : ℕ := 22
def committee_size : ℕ := 6

def total_committees : ℕ := Nat.choose total_members committee_size
def all_boys_committees : ℕ := Nat.choose boys committee_size
def all_girls_committees : ℕ := Nat.choose girls committee_size

def complement_probability := (all_boys_committees + all_girls_committees : ℚ) / total_committees
def desired_probability := 1 - complement_probability

theorem committee_with_at_least_one_boy_and_girl :
  desired_probability = (2913683 : ℚ) / 3838380 := 
sorry

end committee_with_at_least_one_boy_and_girl_l602_602922


namespace find_x_l602_602283

theorem find_x (x : ℝ) (h : 9^(Real.log x / Real.log 8) = 81) : x = 64 := by
  have h₁ : 9 = 3^2 := by norm_num
  have h₂ : 81 = 9^2 := by norm_num
  rw [h₁] at h
  rw [h₂] at h
  have log_identity : (Real.log x / Real.log 8) = 2 := by sorry
  have x_value : x = 8^2 := by sorry
  rw [x_value]
  norm_num
  sorry

end find_x_l602_602283


namespace Kate_has_223_pennies_l602_602829

-- Definition of the conditions
variables (J K : ℕ)
variable (h1 : J = 388)
variable (h2 : J = K + 165)

-- Prove the question equals the answer
theorem Kate_has_223_pennies : K = 223 :=
by
  sorry

end Kate_has_223_pennies_l602_602829


namespace cone_base_radius_condition_l602_602715

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

noncomputable def cone_volume (r l : ℝ) : ℝ := (1/3) * π * r^2 * l

theorem cone_base_radius_condition {r : ℝ} :
  let radius_cylinder := 2 in
  let height_cylinder := (2 * Real.sqrt 3) / 3 in
  let slant_height_cone := Real.sqrt 3 * r in
  cylinder_volume radius_cylinder height_cylinder = cone_volume r slant_height_cone → r = 2 :=
by
  intros _ _ _ h
  sorry

end cone_base_radius_condition_l602_602715


namespace value_of_expression_l602_602139

theorem value_of_expression (x : ℝ) (hx : x = -2) : (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_l602_602139


namespace abs_sum_inequality_for_all_x_l602_602306

theorem abs_sum_inequality_for_all_x (m : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ (m ≤ 3) :=
by
  sorry

end abs_sum_inequality_for_all_x_l602_602306


namespace beads_triangle_l602_602555

theorem beads_triangle (n : ℕ) (h : n = 289) 
  (labels : list ℕ) (hl : (list.range' (n + 1) (6 * n)).perm labels) :
  ∃ (a b c : ℕ), (a ∈ labels) ∧ (b ∈ labels) ∧ (c ∈ labels) ∧ (a ≤ b) ∧ (b ≤ c) ∧ 
  (a + b > c) ∧ (list.nth_le labels i _ = a) ∧ (list.nth_le labels (i+1) _ = b) ∧ 
  (list.nth_le labels (i+2) _ = c) := sorry

end beads_triangle_l602_602555


namespace tomorrow_is_saturday_l602_602866

noncomputable def day_of_the_week := ℕ

def Monday : day_of_the_week := 0
def Tuesday : day_of_the_week := 1
def Wednesday : day_of_the_week := 2
def Thursday : day_of_the_week := 3
def Friday : day_of_the_week := 4
def Saturday : day_of_the_week := 5
def Sunday : day_of_the_week := 6

def day_before_yesterday (today : day_of_the_week) : day_of_the_week :=
(today + 5) % 7

theorem tomorrow_is_saturday (today : day_of_the_week) :
  (day_before_yesterday today + 5) % 7 = Monday →
  ((today + 1) % 7 = Saturday) :=
by
  sorry

end tomorrow_is_saturday_l602_602866


namespace fixed_point_of_function_l602_602544

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^(2-2) - 3) = -2 :=
by
  sorry

end fixed_point_of_function_l602_602544


namespace symmetric_difference_card_eq_18_l602_602789

-- Definitions and conditions
variables (x y : Set ℤ) -- x and y are sets of integers
variable h_x_card : x.finite ∧ x.toFinset.card = 12 -- x consists of 12 integers
variable h_y_card : y.finite ∧ y.toFinset.card = 18 -- y consists of 18 integers
variable h_intersection_card : (x ∩ y).finite ∧ (x ∩ y).toFinset.card = 6 -- 6 integers are in both x and y

-- The theorem statement
theorem symmetric_difference_card_eq_18 : (x ∆ y).toFinset.card = 18 := 
by 
  sorry

end symmetric_difference_card_eq_18_l602_602789


namespace find_k_l602_602158

theorem find_k
  (t k : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : t = 20) :
  k = 68 := 
by
  sorry

end find_k_l602_602158


namespace divisor_equation_solution_l602_602842

-- Define the number of positive divisors function d(n)
def num_divisors (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n.sqrt + 1).natAbs

-- Define the problem statement
theorem divisor_equation_solution (n : ℕ) (h : 0 < n) :
  (num_divisors (n^3) = n) ↔ (n = 1 ∨ n = 28 ∨ n = 40) := 
by 
  sorry

end divisor_equation_solution_l602_602842


namespace combinatorics_problem_l602_602748

open Classical

variables {I : Type} [Nonempty I] [Fintype I]
variables (f g : I → ℤ)

def m : ℤ := ∑ x y, if f x = g y then 1 else 0
def n : ℤ := ∑ x y, if f x = f y then 1 else 0
def k : ℤ := ∑ x y, if g x = g y then 1 else 0

theorem combinatorics_problem (hf : ∀ x, f x ∈ ℤ) (hg : ∀ x, g x ∈ ℤ) :
  2 * m f g ≤ n f + k g := 
sorry

end combinatorics_problem_l602_602748


namespace part_a_part_b_l602_602165

-- Define the setup
def total_people := 100
def total_men := 50
def total_women := 50

-- Peter Ivanovich's position and neighbor relations
def pi_satisfied_prob : ℚ := 25 / 33

-- Expected number of satisfied men
def expected_satisfied_men : ℚ := 1250 / 33

-- Lean statements for the problems

-- Part (a): Prove Peter Ivanovich's satisfaction probability
theorem part_a (total_people = 100) (total_men = 50) (total_women = 50) : 
  pi_satisfied_prob = 25 / 33 := 
sorry

-- Part (b): Expected number of satisfied men
theorem part_b (total_people = 100) (total_men = 50) (total_women = 50) : 
  expected_satisfied_men = 1250 / 33 := 
sorry

end part_a_part_b_l602_602165


namespace probability_A_not_lose_l602_602575

theorem probability_A_not_lose (p_win p_draw : ℝ) (h_win : p_win = 0.3) (h_draw : p_draw = 0.5) :
  (p_win + p_draw = 0.8) :=
by
  rw [h_win, h_draw]
  norm_num

end probability_A_not_lose_l602_602575


namespace smaller_angle_of_parallelogram_l602_602390

theorem smaller_angle_of_parallelogram (x : ℝ) (h : x + 3 * x = 180) : x = 45 :=
sorry

end smaller_angle_of_parallelogram_l602_602390


namespace range_of_MF_plus_MN_l602_602351

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

theorem range_of_MF_plus_MN (M : ℝ × ℝ) (N : ℝ × ℝ) (F : ℝ × ℝ) (hM : point_on_parabola M.1 M.2) (hN : N = (2, 2)) (hF : F = (1, 0)) :
  ∃ y : ℝ, y ≥ 3 ∧ ∀ MF MN : ℝ, MF = abs (M.1 - F.1) + abs (M.2 - F.2) ∧ MN = abs (M.1 - N.1) + abs (M.2 - N.2) → MF + MN = y :=
sorry

end range_of_MF_plus_MN_l602_602351


namespace hyperbola_eccentricity_l602_602702

/-- Given a quadrilateral formed by the two foci and the two endpoints of the 
conjugate axis of a hyperbola C, one of its internal angles is 60 degrees. 
Prove that the eccentricity of the hyperbola C is sqrt(6)/2. -/
theorem hyperbola_eccentricity (F1 F2 B1 B2 : ℝ×ℝ) (C : set (ℝ×ℝ)) 
  (h1 : hyperbola C) 
  (h2 : is_focus F1 C) 
  (h3 : is_focus F2 C) 
  (h4 : is_endpoint_conjugate_axis B1 C) 
  (h5 : is_endpoint_conjugate_axis B2 C) 
  (h6 : internal_angle F1 B1 F2 B2 = 60) :
  eccentricity C = real.sqrt 3 / 2 :=
sorry

end hyperbola_eccentricity_l602_602702


namespace lcm_9_12_15_l602_602998

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l602_602998


namespace expected_value_of_multiples_of_6_is_50_div_3_l602_602443

noncomputable def x : ℕ → ℕ := sorry

noncomputable def S (n : ℕ) : ℕ :=
finset.sum (finset.range n) x

noncomputable def expected_multiples_of_6 : ℚ :=
100 * (1/6)

theorem expected_value_of_multiples_of_6_is_50_div_3 :
  (100 * (1/6) : ℚ) = 50 / 3 :=
sorry

end expected_value_of_multiples_of_6_is_50_div_3_l602_602443


namespace evaluate_expression_l602_602265

def sequence (i : ℕ) : ℕ :=
  if i = 1 then 2
  else if i = 2 then 4
  else if i = 3 then 6
  else 2 * (List.prod (List.map sequence (List.range (i - 1)))) - 2

def product_of_sequence_upto (n : ℕ) : ℕ :=
  List.prod (List.map sequence (List.range n))

def sum_of_squares_sequence_upto (n : ℕ) : ℕ :=
  List.sum (List.map (λ i, (sequence i)^2) (List.range n))

theorem evaluate_expression : 
  product_of_sequence_upto 7 - sum_of_squares_sequence_upto 7 ≈ -1.5049293766319112 * 10^23 :=
by
  sorry

end evaluate_expression_l602_602265


namespace union_of_A_B_complement_and_intersection_range_of_a_l602_602737

namespace MathProof

variable {α : Type*}

def A : Set ℝ := { x | 2 < x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | 5 - a < x ∧ x < a }

theorem union_of_A_B : A ∪ B = { x | 2 < x ∧ x < 10 } :=
by
  sorry

theorem complement_and_intersection (x : ℝ) : 
  (x ∉ A ∨ x = ∞) ∧ B x → (7 ≤ x ∧ x < 10) :=
by
  sorry

theorem range_of_a (a : ℝ) (h : C a ⊆ B) : a ≤ 3 :=
by
  sorry

end MathProof

end union_of_A_B_complement_and_intersection_range_of_a_l602_602737


namespace area_quadrilateral_eq_sum_l602_602825

variables {A B C a b c : Point}
variables {Aa Bb Cc : Line}

-- Assume Triangle structure and parallel lines
variables (h_triangle : Triangle A B C)
variables (h_Aa_parallel_bc : Parallel Aa (BC : Line))
variables (h_Bb_parallel_ca : Parallel Bb (CA : Line))
variables (h_Cc_parallel_ab : Parallel Cc (AB : Line))

-- Assume intersection points
variables (h_intersect_a : Intersect Aa BC a)
variables (h_intersect_b : Intersect Bb CA b)
variables (h_intersect_c : Intersect Cc AB c)

theorem area_quadrilateral_eq_sum {Aa Bb Cc : ℝ} :
  Aa * Bb = Aa * Cc + Bb * Cc :=
sorry

end area_quadrilateral_eq_sum_l602_602825


namespace method_one_cost_eq_300_method_two_cost_eq_300_method_one_more_cost_effective_l602_602189

noncomputable def method_one_cost (x : ℕ) : ℕ := 120 + 10 * x

noncomputable def method_two_cost (x : ℕ) : ℕ := 15 * x

theorem method_one_cost_eq_300 (x : ℕ) : method_one_cost x = 300 ↔ x = 18 :=
by sorry

theorem method_two_cost_eq_300 (x : ℕ) : method_two_cost x = 300 ↔ x = 20 :=
by sorry

theorem method_one_more_cost_effective (x : ℕ) :
  x ≥ 40 → method_one_cost x < method_two_cost x :=
by sorry

end method_one_cost_eq_300_method_two_cost_eq_300_method_one_more_cost_effective_l602_602189


namespace soccer_match_water_consumed_l602_602462

theorem soccer_match_water_consumed:
  ∀ (W D : ℕ),
  let initial_water := 12,
      initial_soda  := 34,
      remaining_total := 30,
      total_consumed := initial_water + initial_soda - remaining_total
  in
  (D = 3 * W) ∧ (W + D = total_consumed) → W = 4 := by
sorry

end soccer_match_water_consumed_l602_602462


namespace smallest_angle_l602_602256

theorem smallest_angle (x : ℝ) (h : Real.Tan (6 * x) = (Real.Sin x - Real.Cos x) / (Real.Sin x + Real.Cos x)) :
  x = 6.42857 :=
sorry

end smallest_angle_l602_602256


namespace total_students_at_competition_l602_602473

theorem total_students_at_competition:
  let quantum_students := 90 in
  let schrodinger_students := (2/3:ℚ) * quantum_students in
  let einstein_students := (4/9:ℚ) * schrodinger_students in
  let newton_students := (5/12:ℚ) * einstein_students in
  let galileo_students := (11/20:ℚ) * newton_students in
  let pascal_students := (13/50:ℚ) * galileo_students in
  let first_six_schools_total := quantum_students + schrodinger_students + einstein_students + newton_students + galileo_students + pascal_students in
  let faraday_students := 4 * first_six_schools_total in
  first_six_schools_total + faraday_students = 980 :=
by
  sorry

end total_students_at_competition_l602_602473


namespace pairs_equality_l602_602627

-- Define all the pairs as given in the problem.
def pairA_1 : ℤ := - (2^7)
def pairA_2 : ℤ := (-2)^7
def pairB_1 : ℤ := - (3^2)
def pairB_2 : ℤ := (-3)^2
def pairC_1 : ℤ := -3 * (2^3)
def pairC_2 : ℤ := - (3^2) * 2
def pairD_1 : ℤ := -((-3)^2)
def pairD_2 : ℤ := -((-2)^3)

-- The problem statement.
theorem pairs_equality :
  pairA_1 = pairA_2 ∧ ¬ (pairB_1 = pairB_2) ∧ ¬ (pairC_1 = pairC_2) ∧ ¬ (pairD_1 = pairD_2) := by
  sorry

end pairs_equality_l602_602627


namespace sum_of_cubes_three_consecutive_divisible_by_three_l602_602058

theorem sum_of_cubes_three_consecutive_divisible_by_three (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 3 = 0 := 
by 
  sorry

end sum_of_cubes_three_consecutive_divisible_by_three_l602_602058


namespace covered_squares_l602_602190

/-- Given a 10x10 checkerboard with each square having side length D,
    and a disc of diameter D centered at (5D, 4D), 
    the number of completely covered squares is 20. -/

theorem covered_squares (D : ℝ) : 
  let checkerboard_side := 10 * D,
      disc_radius := D / 2,
      disc_center := (5 * D, 4 * D) in
  ∃ n : ℕ, n = 20 ∧ ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ 10 → 
    let square_center := (i * D + D / 2, j * D + D / 2),
        distance := (square_center.1 - disc_center.1)^2 + (square_center.2 - disc_center.2)^2 in
    distance ≤ (disc_radius)^2 → true := sorry

end covered_squares_l602_602190


namespace A_square_or_cube_neg_identity_l602_602838

open Matrix

theorem A_square_or_cube_neg_identity (A : Matrix (Fin 2) (Fin 2) ℚ)
  (n : ℕ) (hn_nonzero : n ≠ 0) (hA_pow_n : A ^ n = -(1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  A ^ 2 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) ∨ A ^ 3 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

end A_square_or_cube_neg_identity_l602_602838


namespace max_S8_l602_602316

-- Definitions and conditions
def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → (|a (n + 1) - a n| = n) ∧ (a n ≤ (n - 1) / 2)

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, a i

-- Main theorem
theorem max_S8 : ∀ (a : ℕ → ℤ), 
sequence a →
a 1 = 0 →
(sum_of_first_n_terms a 8) = -4 :=
sorry

end max_S8_l602_602316


namespace expected_number_of_own_hats_l602_602574

-- Define the number of people
def num_people : ℕ := 2015

-- Define the expectation based on the problem description
noncomputable def expected_hats (n : ℕ) : ℝ := 1

-- The main theorem representing the problem statement
theorem expected_number_of_own_hats : expected_hats num_people = 1 := sorry

end expected_number_of_own_hats_l602_602574


namespace count_desired_multiples_l602_602364

-- Define the condition of being a positive integer not exceeding 200
def is_valid (n : ℕ) : Prop := n > 0 ∧ n ≤ 200

-- Define the condition of being a multiple of 2
def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

-- Define the condition of being a multiple of 5
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

-- Define the condition of being a multiple of 6
def is_multiple_of_6 (n : ℕ) : Prop := n % 6 = 0

-- Define the condition of satisfying the problem's main criteria
def is_desired_multiple (n : ℕ) : Prop := 
  is_valid n ∧ (is_multiple_of_2 n ∨ is_multiple_of_5 n) ∧ ¬is_multiple_of_6 n

-- Problem statement: the number of desired multiples
theorem count_desired_multiples : 
  (finset.filter is_desired_multiple (finset.range 201)).card = 87 := 
sorry

end count_desired_multiples_l602_602364


namespace correct_flowchart_requirement_l602_602086

def flowchart_requirement (option : String) : Prop := 
  option = "From left to right, from top to bottom" ∨
  option = "From right to left, from top to bottom" ∨
  option = "From left to right, from bottom to top" ∨
  option = "From right to left, from bottom to top"

theorem correct_flowchart_requirement : 
  (∀ option, flowchart_requirement option → option = "From left to right, from top to bottom") :=
by
  sorry

end correct_flowchart_requirement_l602_602086


namespace tomorrow_is_Saturday_l602_602859
noncomputable theory

def day_of_week := ℕ
def Monday : day_of_week := 0
def Tuesday : day_of_week := 1
def Wednesday : day_of_week := 2
def Thursday : day_of_week := 3
def Friday : day_of_week := 4
def Saturday : day_of_week := 5
def Sunday : day_of_week := 6

def days_after (d : day_of_week) (n : ℕ) : day_of_week := (d + n) % 7

-- The condition: Monday is five days after the day before yesterday.
def day_before_yesterday := Wednesday
def today := days_after day_before_yesterday 2
def tomorrow := days_after today 1

theorem tomorrow_is_Saturday (h: days_after day_before_yesterday 5 = Monday) : tomorrow = Saturday := 
by {
  sorry
}

end tomorrow_is_Saturday_l602_602859


namespace tomorrow_is_saturday_l602_602867

noncomputable def day_of_the_week := ℕ

def Monday : day_of_the_week := 0
def Tuesday : day_of_the_week := 1
def Wednesday : day_of_the_week := 2
def Thursday : day_of_the_week := 3
def Friday : day_of_the_week := 4
def Saturday : day_of_the_week := 5
def Sunday : day_of_the_week := 6

def day_before_yesterday (today : day_of_the_week) : day_of_the_week :=
(today + 5) % 7

theorem tomorrow_is_saturday (today : day_of_the_week) :
  (day_before_yesterday today + 5) % 7 = Monday →
  ((today + 1) % 7 = Saturday) :=
by
  sorry

end tomorrow_is_saturday_l602_602867


namespace possible_values_of_varphi_l602_602387

theorem possible_values_of_varphi (phi : ℝ) :
  (∀ x : ℝ, 
    sin (2 * (x + π / 6) + phi) = sin (2 * (-x) + π / 3 + phi)) →
  phi = π / 6 :=
begin
  intro h,
  -- Lean proof would be written here
  sorry
end

end possible_values_of_varphi_l602_602387


namespace additional_people_needed_l602_602684

theorem additional_people_needed (h₁ : ∀ p h : ℕ, (p * h = 40)) (h₂ : 5 * 8 = 40) : 7 - 5 = 2 :=
by
  sorry

end additional_people_needed_l602_602684


namespace karl_total_income_is_53_l602_602013

noncomputable def compute_income (tshirt_price pant_price skirt_price sold_tshirts sold_pants sold_skirts sold_refurbished_tshirts: ℕ) : ℝ :=
  let tshirt_income := 2 * tshirt_price
  let pant_income := sold_pants * pant_price
  let skirt_income := sold_skirts * skirt_price
  let refurbished_tshirt_price := (tshirt_price : ℝ) / 2
  let refurbished_tshirt_income := sold_refurbished_tshirts * refurbished_tshirt_price
  tshirt_income + pant_income + skirt_income + refurbished_tshirt_income

theorem karl_total_income_is_53 : compute_income 5 4 6 2 1 4 6 = 53 := by
  sorry

end karl_total_income_is_53_l602_602013


namespace avg_interview_score_higher_recruit_B_based_on_comprehensive_score_l602_602961

-- Define the initial interview scores of A and B
def scores_A : List ℤ := [93, 91, 80, 92, 95, 89, 88, 97, 95, 93]
def scores_B : List ℤ := [90, 92, 88, 92, 90, 90, 84, 96, 94, 92]

-- Define the written test scores of A and B
def written_test_A : ℤ := 92
def written_test_B : ℤ := 94

-- Define the weighted ratio of written test score to interview score
def weighted_ratio_written : ℚ := 6 / 10
def weighted_ratio_interview : ℚ := 4 / 10

-- Define the average function to calculate the average score after removing the highest and lowest scores
noncomputable def avg_scores_after_removal (scores : List ℤ) : ℚ :=
  let sorted_scores := List.sort (≤) scores
  let trimmed_scores := List.drop 1 (List.take (sorted_scores.length - 1) sorted_scores)
  (trimmed_scores.sum : ℚ) / trimmed_scores.length

-- Define the interview scores after removal for A and B
noncomputable def interview_score_A : ℚ := avg_scores_after_removal scores_A
noncomputable def interview_score_B : ℚ := avg_scores_after_removal scores_B

-- Define the comprehensive scores for A and B using the weighted ratio
noncomputable def comprehensive_score_A : ℚ :=
  (written_test_A : ℚ) * weighted_ratio_written + interview_score_A * weighted_ratio_interview

noncomputable def comprehensive_score_B : ℚ :=
  (written_test_B : ℚ) * weighted_ratio_written + interview_score_B * weighted_ratio_interview

-- Proof goal: The average interview score of A is higher than B
theorem avg_interview_score_higher :
  interview_score_A > interview_score_B :=
by
  sorry

-- Proof goal: Based on the comprehensive scores, B should be recruited
theorem recruit_B_based_on_comprehensive_score :
  comprehensive_score_B > comprehensive_score_A :=
by
  sorry

end avg_interview_score_higher_recruit_B_based_on_comprehensive_score_l602_602961


namespace line_tangent_ln_curve_l602_602733

theorem line_tangent_ln_curve (b : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (ln x = (1/2) * x + b) ∧ ((1 / x) = 1 / 2)) → b = ln 2 - 1 :=
by
  intro h
  sorry

end line_tangent_ln_curve_l602_602733


namespace triangle_perimeters_sum_l602_602464

theorem triangle_perimeters_sum :
  ∃ (t : ℕ),
    (∀ (A B C D : Type) (x y : ℕ), 
      (AB = 7 ∧ BC = 17 ∧ AD = x ∧ CD = x ∧ BD = y ∧ x^2 - y^2 = 240) →
      t = 114) :=
sorry

end triangle_perimeters_sum_l602_602464


namespace niraek_donut_hole_count_l602_602247

noncomputable def surface_area (r : ℕ) : ℝ := 4 * real.pi * (r ^ 2)

theorem niraek_donut_hole_count :
  let A := surface_area 7
  let B := surface_area 9
  let C := surface_area 11
  let D := surface_area 13
  let lcm_areas := nat.lcm (nat.lcm (nat.lcm (to_nat A) (to_nat B)) (to_nat C)) (to_nat D)
  lcm_areas / (to_nat A) = 371293 := 
by {
  sorry
}

end niraek_donut_hole_count_l602_602247


namespace total_shingles_needed_l602_602919

-- Defining the dimensions of the house and the porch
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_length : ℝ := 6
def porch_width : ℝ := 4.5

-- The goal is to prove that the total area of the shingles needed is 232 square feet
theorem total_shingles_needed :
  (house_length * house_width) + (porch_length * porch_width) = 232 := by
  sorry

end total_shingles_needed_l602_602919


namespace part1_part2_l602_602027

open Set

def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem part1 (a : ℝ) : (A a ∩ B = A a ∪ B) → a = 5 :=
by
  sorry

theorem part2 (a : ℝ) : (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -2 :=
by
  sorry

end part1_part2_l602_602027


namespace incorrect_subset_l602_602150

variables {α β : Type*} (a b : α)

theorem incorrect_subset : ¬ ({(a, b)} ⊆ {a, b}) :=
by {
  intro h,
  have ha : (a, b) ∈ {(a, b)} := set.mem_singleton (a, b),
  have hsub := h ha,
  cases hsub,
  { exact false.elim (set.not_mem_singleton b hsub) },
  { exact false.elim (set.not_mem_singleton a hsub) }
}

end incorrect_subset_l602_602150


namespace blood_expiration_date_l602_602230

theorem blood_expiration_date :
  let seconds_in_a_day : ℕ := 86400 in
  let seconds_in_eight_factorial : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 in
  (seconds_in_eight_factorial < seconds_in_a_day) →
  ∃ t : ℕ, t = 11 * 60 * 60 + 13 * 60 → expiry_date = "February 1, 11:13 PM" :=
by {
  sorry
}

end blood_expiration_date_l602_602230


namespace find_n_l602_602288

noncomputable def satisfies_condition (n d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ) : Prop :=
  1 = d₁ ∧ d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ d₄ < d₅ ∧ d₅ < d₆ ∧ d₆ < d₇ ∧ d₇ < n ∧
  (∀ d, d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n → n % d = 0) ∧
  (∀ d, n % d = 0 → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n)

theorem find_n (n : ℕ) : (∃ d₁ d₂ d₃ d₄ d₅ d₆ d₇, satisfies_condition n d₁ d₂ d₃ d₄ d₅ d₆ d₇ ∧ n = d₆^2 + d₇^2 - 1) → (n = 144 ∨ n = 1984) :=
  by
  sorry

end find_n_l602_602288


namespace train_speed_calculation_l602_602229

open Real

noncomputable def train_speed_in_kmph (V : ℝ) : ℝ := V * 3.6

theorem train_speed_calculation (L V : ℝ) (h1 : L = 16 * V) (h2 : L + 280 = 30 * V) :
  train_speed_in_kmph V = 72 :=
by
  sorry

end train_speed_calculation_l602_602229


namespace least_common_multiple_9_12_15_l602_602973

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l602_602973


namespace moles_of_C6H5CH3_formed_l602_602674

-- Stoichiometry of the reaction
def balanced_reaction (C6H6 CH4 C6H5CH3 H2 : ℝ) : Prop :=
  C6H6 + CH4 = C6H5CH3 + H2

-- Given conditions
def reaction_conditions (initial_CH4 : ℝ) (initial_C6H6 final_C6H5CH3 final_H2 : ℝ) : Prop :=
  balanced_reaction initial_C6H6 initial_CH4 final_C6H5CH3 final_H2 ∧ initial_CH4 = 3 ∧ final_H2 = 3

-- Theorem to prove
theorem moles_of_C6H5CH3_formed (initial_CH4 final_C6H5CH3 : ℝ) : reaction_conditions initial_CH4 3 final_C6H5CH3 3 → final_C6H5CH3 = 3 :=
by
  intros h
  sorry

end moles_of_C6H5CH3_formed_l602_602674


namespace n_is_perfect_square_l602_602826

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k ^ 2

theorem n_is_perfect_square (a b c d : ℤ) (h : a + b + c + d = 0) : 
  is_perfect_square ((ab - cd) * (bc - ad) * (ca - bd)) := 
  sorry

end n_is_perfect_square_l602_602826


namespace correct_average_l602_602077

theorem correct_average 
  (avg : ℕ) (num_count : ℕ) (incorrect_num : ℕ) (correct_num : ℕ) 
  (sum_except : ℕ) (x : ℕ) :
  avg = 16 →
  num_count = 10 →
  incorrect_num = 25 →
  correct_num = 35 →
  sum_except = 102 →
  (x = (correct_num * num_count + correct_num - incorrect_num - sum_except)) →
  let correct_total_sum := sum_except + correct_num + x in 
  (correct_total_sum / num_count = 17) :=
by
  intros
  sorry

end correct_average_l602_602077


namespace pascal_triangle_42nd_number_l602_602128

theorem pascal_triangle_42nd_number :
  nat.choose 45 41 = 148995 :=
by sorry

end pascal_triangle_42nd_number_l602_602128


namespace range_of_m_l602_602778

theorem range_of_m (m : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = 4 * x / (x^2 + 1)) :
  (∀ x ∈ (m, 2 * m + 1), ∀ y ∈ (m, 2 * m + 1), x < y → f x ≤ f y) ↔ m ∈ Ioo (-1 : ℝ) 0 ∪ Icc (-1 : ℝ) 0 := 
by
  sorry

end range_of_m_l602_602778


namespace band_length_cylinders_l602_602194

/-- Consider two cylinders of radii 12 and 36 
which are held tangent to each other with a tight band.
Let m, k, and n be positive integers such that the length of the band is of 
the form m * sqrt k + n * pi, where k is not divisible by the square of any prime.
Given these conditions, we prove that m + k + n = 107. -/
theorem band_length_cylinders :
  ∃ m k n : ℕ, 
  let r1 := 12 in let r2 := 36 in
  let band_length := 48 * Real.sqrt 3 + 56 * Real.pi in
  m * Real.sqrt k + n * Real.pi = band_length ∧
  m > 0 ∧ k > 0 ∧ n > 0 ∧
  (∀ p : ℕ, Prime p → ¬ (p * p ∣ k)) ∧
  m + k + n = 107 :=
  sorry

end band_length_cylinders_l602_602194


namespace max_value_of_f_l602_602341

def f (x : ℝ) : ℝ := 3 * real.sqrt (4 - x) + 4 * real.sqrt (x - 3)

theorem max_value_of_f :
  ∃ x (h : 3 ≤ x ∧ x ≤ 4), f x = 5 :=
sorry

end max_value_of_f_l602_602341


namespace night_crew_fraction_l602_602246

theorem night_crew_fraction (D N : ℝ) (B : ℝ) 
  (h1 : ∀ d, d = D → ∀ n, n = N → ∀ b, b = B → (n * (3/4) * b) = (3/4) * (d * b) / 3)
  (h2 : ∀ t, t = (D * B + (N * (3/4) * B)) → (D * B) / t = 2 / 3) :
  N / D = 2 / 3 :=
by
  sorry

end night_crew_fraction_l602_602246


namespace sequence_bijective_l602_602852

def sequence (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, (∑ i in Finset.range n, a i + a n) % (n + 1) = 0)
  ∧ ∀ m n : ℕ, m ≠ n → a m ≠ a n

theorem sequence_bijective (a : ℕ → ℕ) (h : sequence a) :
  ∀ z : ℤ, ∃ ! i : ℕ, (a i - i : ℤ) = z :=
sorry

end sequence_bijective_l602_602852


namespace sum_of_values_eq_neg4_l602_602444

def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x - 3
else x / 3 + 1

theorem sum_of_values_eq_neg4 :
  (∑ x in {x : ℝ | f x = 1}, x) = -4 := by
  sorry

end sum_of_values_eq_neg4_l602_602444


namespace find_hyperbola_equation_hyperbola_equation_l602_602292

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) := (x^2 / 2) - y^2 = 1

-- Define the new hyperbola with unknown constant m
def new_hyperbola (x y m : ℝ) := (x^2 / (m * 2)) - (y^2 / m) = 1

variable (m : ℝ)

-- The point (2, 0)
def point_on_hyperbola (x y : ℝ) := x = 2 ∧ y = 0

theorem find_hyperbola_equation (h : ∀ (x y : ℝ), point_on_hyperbola x y → new_hyperbola x y m) :
  m = 2 :=
    sorry

theorem hyperbola_equation :
  ∀ (x y : ℝ), (x = 2 ∧ y = 0) → (x^2 / 4 - y^2 / 2 = 1) :=
    sorry

end find_hyperbola_equation_hyperbola_equation_l602_602292


namespace sequence_S15_is_211_l602_602409

theorem sequence_S15_is_211 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2)
  (h3 : ∀ n > 1, S (n + 1) + S (n - 1) = 2 * (S n + S 1)) :
  S 15 = 211 := 
sorry

end sequence_S15_is_211_l602_602409


namespace unit_vectors_equal_magnitude_l602_602709

variable {ℝ : Type*}
variable [normed_group ℝ] [normed_space ℝ ℝ]

theorem unit_vectors_equal_magnitude
    (a b : ℝ)
    (unit_a : ‖a‖ = 1)
    (unit_b : ‖b‖ = 1) :
    ‖a‖ = ‖b‖ := 
sorry

end unit_vectors_equal_magnitude_l602_602709


namespace average_speed_calculation_l602_602002

theorem average_speed_calculation :
  (100 : ℝ) / (1 + 15 / 60 : ℝ) = 80 :=
by
  -- Converting the given time 1 hour and 15 minutes to hours
  have time_in_hours : ℝ := 1 + 15 / 60
  -- Establishing the total distance traveled
  have distance_in_kilometers : ℝ := 100
  -- Calculate average speed
  let avg_speed := distance_in_kilometers / time_in_hours
  -- Check if the calculated average speed is 80 km/h
  show avg_speed = 80
  sorry

end average_speed_calculation_l602_602002


namespace lunch_break_duration_l602_602886

-- Definitions based on the conditions
variables (p h1 h2 L : ℝ)
-- Monday equation
def monday_eq : Prop := (9 - L/60) * (p + h1 + h2) = 0.55
-- Tuesday equation
def tuesday_eq : Prop := (7 - L/60) * (p + h2) = 0.35
-- Wednesday equation
def wednesday_eq : Prop := (5 - L/60) * (p + h1 + h2) = 0.25
-- Thursday equation
def thursday_eq : Prop := (4 - L/60) * p = 0.15

-- Combine all conditions
def all_conditions : Prop :=
  monday_eq p h1 h2 L ∧ tuesday_eq p h2 L ∧ wednesday_eq p h1 h2 L ∧ thursday_eq p L

-- Proof that the lunch break duration is 60 minutes
theorem lunch_break_duration : all_conditions p h1 h2 L → L = 60 :=
by
  sorry

end lunch_break_duration_l602_602886


namespace verify_a_l602_602335

def terminal_side_of_angle (θ : ℝ) (x y : ℝ) := 
  (x, y) = (2 * (Real.sin (Real.pi / 8))^2 - 1, y) && Real.sin θ = 2 * Real.sqrt 3 * Real.sin (13 * Real.pi / 12) * Real.cos (Real.pi / 12)

theorem verify_a (θ a : ℝ) (h1 : terminal_side_of_angle θ (-Real.sqrt 2 / 2) a) :
  a = -Real.sqrt 6 / 2 := 
sorry

end verify_a_l602_602335


namespace lcm_9_12_15_l602_602984

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l602_602984


namespace geometric_sequence_sum_l602_602426

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h1 : a 1 + a 2 + a 3 = 7)
  (h2 : a 2 + a 3 + a 4 = 14) :
  a 4 + a 5 + a 6 = 56 :=
sorry

end geometric_sequence_sum_l602_602426


namespace curtains_length_needed_l602_602260

def room_height_feet : ℕ := 8
def additional_material_inches : ℕ := 5

def height_in_inches : ℕ := room_height_feet * 12

def total_length_curtains : ℕ := height_in_inches + additional_material_inches

theorem curtains_length_needed : total_length_curtains = 101 := by
  sorry

end curtains_length_needed_l602_602260


namespace tomorrow_is_saturday_l602_602869

noncomputable def day_of_the_week := ℕ

def Monday : day_of_the_week := 0
def Tuesday : day_of_the_week := 1
def Wednesday : day_of_the_week := 2
def Thursday : day_of_the_week := 3
def Friday : day_of_the_week := 4
def Saturday : day_of_the_week := 5
def Sunday : day_of_the_week := 6

def day_before_yesterday (today : day_of_the_week) : day_of_the_week :=
(today + 5) % 7

theorem tomorrow_is_saturday (today : day_of_the_week) :
  (day_before_yesterday today + 5) % 7 = Monday →
  ((today + 1) % 7 = Saturday) :=
by
  sorry

end tomorrow_is_saturday_l602_602869


namespace perpendicular_line_plane_l602_602031

-- Definitions for the conditions:
variables (α β : Plane) (l : Line)

-- Stating the conditions:
axiom l_perp_alpha : Perpendicular l α
axiom alpha_parallel_beta : Parallel α β

-- Stating the theorem to be proved:
theorem perpendicular_line_plane (α β : Plane) (l : Line)
  (l_perp_alpha : Perpendicular l α)
  (alpha_parallel_beta : Parallel α β) : Perpendicular l β :=
sorry

end perpendicular_line_plane_l602_602031


namespace sine_subtraction_formula_simplify_expression_l602_602486

-- Define the sine subtraction formula as a condition
theorem sine_subtraction_formula (a b : ℝ) : 
    real.sin (a - b) = real.sin a * real.cos b - real.cos a * real.sin b := by
  sorry

-- Prove the given expression simplifies to sin(x)
theorem simplify_expression (x y : ℝ) :
    real.sin (x + y) * real.cos y - real.cos (x + y) * real.sin y = real.sin x := by
  have h : real.sin ((x + y) - y) = real.sin (x + y) * real.cos y - real.cos (x + y) * real.sin y := by
    exact sine_subtraction_formula (x + y) y
  rw [sub_self y, h]
  simp
  sorry

end sine_subtraction_formula_simplify_expression_l602_602486


namespace value_of_expression_l602_602140

theorem value_of_expression (x : ℝ) (hx : x = -2) : (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_l602_602140


namespace qualified_probability_l602_602535

def defect_rate_1 : ℝ := 0.03
def defect_rate_2 : ℝ := 0.05
def process_independent : Prop := true  -- This represents that the processes are independent

theorem qualified_probability :
  let P_A := 1 - defect_rate_1 in
  let P_B := 1 - defect_rate_2 in
  process_independent → P_A * P_B = 0.9215 :=
by
  intros
  sorry

end qualified_probability_l602_602535


namespace square_perimeter_l602_602614

theorem square_perimeter (area : ℝ) (h : area = 625) : 
  let s := Real.sqrt area in
  (4 * s) = 100 :=
by
  let s := Real.sqrt area
  have hs : s = 25 := by sorry
  calc
    (4 * s) = 4 * 25 : by rw hs
          ... = 100   : by norm_num

end square_perimeter_l602_602614


namespace lcm_9_12_15_l602_602979

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l602_602979


namespace count_multiples_2_or_5_but_not_6_l602_602362

theorem count_multiples_2_or_5_but_not_6 : 
  {n : ℕ | n ≤ 200 ∧ (n % 2 = 0 ∨ n % 5 = 0) ∧ n % 6 ≠ 0}.card = 87 := 
  sorry

end count_multiples_2_or_5_but_not_6_l602_602362


namespace find_m_value_l602_602807

open Real

theorem find_m_value (m : ℝ) (h1 : m ≠ -2)
  (h2 : ∃ y' l, (∀ x y, y = 2 * x - m / x → y' = deriv (λ x, 2 * x - m / x) x → l = tangent_line y' x) 
  (h3 : ∃ y_intercept x_intercept, y_intercept + x_intercept = 12) :
  m = -3 ∨ m = -4 := by
sorry

end find_m_value_l602_602807


namespace statements_truth_l602_602548

theorem statements_truth :
  (5 ∣ 45) ∧ (¬ (19 ∣ 209 ∧ ¬ (19 ∣ 57))) ∧ (¬ (¬ (28 ∣ 84) ∧ ¬ (28 ∣ 71))) ∧
  ((14 ∣ 42 ∧ ¬ (14 ∣ 63))) ∧ (9 ∣ 180) :=
by {
  -- 5 is a factor of 45
  have H1 : 5 ∣ 45, by sorry,
  -- 19 is a divisor of 209 but also of 57, so the statement is false
  have H2 : ¬ (19 ∣ 209 ∧ ¬ (19 ∣ 57)), by sorry,
  -- 28 is a divisor of 84, so the statement is false
  have H3 : ¬ (¬ (28 ∣ 84) ∧ ¬ (28 ∣ 71)), by sorry,
  -- 14 is a divisor of 42 but not of 63
  have H4 : 14 ∣ 42 ∧ ¬ (14 ∣ 63), by sorry,
  -- 9 is a factor of 180
  have H5 : 9 ∣ 180, by sorry,
  exact ⟨H1, H2, H3, H4, H5⟩
}

end statements_truth_l602_602548


namespace square_perimeter_l602_602591

theorem square_perimeter (area : ℝ) (h : area = 625) :
  ∃ p : ℝ, p = 4 * real.sqrt area ∧ p = 100 :=
by
  sorry

end square_perimeter_l602_602591


namespace sequence_problems_l602_602315

noncomputable def sequenceSum (f : ℕ → ℝ) (n : ℕ) : ℝ := (finset.range n).sum (λ i => f i)

theorem sequence_problems (a_n b_n : ℕ → ℝ) (S_n : ℕ → ℝ) :
  -- Given the conditions
  (∀ n : ℕ, S_n n = sequenceSum a_n (n + 1)) ∧
  (∀ n : ℕ, S_n n = 2 * a_n n - 2) ∧
  (∀ n : ℕ, b_n n = n) ∧
  -- Prove the sequences and sum formula
  (∀ n : ℕ, a_n n = 2^n) ∧
  (∀ n : ℕ, b_n n = n) ∧
  (∀ n : ℕ, sequenceSum (λ n => a_n n * b_n n) n = (n - 1) * 2^(n + 1) + 2) :=
by
  sorry

end sequence_problems_l602_602315


namespace manu_wins_probability_l602_602629

def prob_manu_wins : ℚ :=
  let a := (1/2) ^ 5
  let r := (1/2) ^ 4
  a / (1 - r)

theorem manu_wins_probability : prob_manu_wins = 1 / 30 :=
  by
  -- here we would have the proof steps
  sorry

end manu_wins_probability_l602_602629


namespace lcm_of_9_12_15_is_180_l602_602985

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l602_602985


namespace intersection_high_probability_l602_602962

-- Definitions
def parabola (a b : ℤ) (x : ℝ) : ℝ := x^2 + (a : ℝ) * x + (b : ℝ)
def line (c d : ℤ) (x : ℝ) : ℝ := (c : ℝ) * x + (d : ℝ)

def quadratic_discriminant (a b c d : ℤ) : ℝ := (a - c : ℝ)^2 - 4 * (b - d : ℝ)

-- Event E: The quadratic equation has at least one real root (i.e., its discriminant is non-negative)
def E (a b c d : ℤ) : Prop := quadratic_discriminant a b c d ≥ 0

-- Probability measure on the set of integers from 1 to 6
noncomputable def uniform_measure : measure ℤ := sorry  -- We assume a uniform probability measure on the given range

-- Mathematical statement
theorem intersection_high_probability : 
  (probability (set_of (λ (abcd : ℤ × ℤ × ℤ × ℤ), E abcd.1 abcd.2.1 abcd.2.2.1 abcd.2.2.2))) > 5/6 :=
sorry

end intersection_high_probability_l602_602962


namespace min_cost_to_1981_impossible_to_1982_l602_602463

-- Define the operations cost
def multiply_by_3_cost : ℕ := 5
def add_4_cost : ℕ := 2

-- Define the starting point
def start_number : ℕ := 1

-- Define the target number
def target_number_1981 : ℕ := 1981
def target_number_1982 : ℕ := 1982

-- Statement of the problems in Lean 4
theorem min_cost_to_1981 : ∃ (C : ℕ), min_cost start_number target_number_1981 = C := sorry

theorem impossible_to_1982 : ¬ (∃ (C : ℕ), min_cost start_number target_number_1982 = C) := sorry

end min_cost_to_1981_impossible_to_1982_l602_602463


namespace find_range_of_a_l602_602433

noncomputable def value_range_for_a : Set ℝ := {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4}

theorem find_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0)  ∧
  (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧
  (¬ (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0) → ¬ (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0))
  → a ∈ value_range_for_a :=
sorry

end find_range_of_a_l602_602433


namespace peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l602_602176

-- Define the context of the problem
def total_people := 100
def men := 50
def women := 50

-- Define Peter Ivanovich being satisfied
def satisfies_peter_ivanovich := (women / (total_people - 1)) * ((women - 1) / (total_people - 2)) 

-- Define the probability that Peter Ivanovich is satisfied
theorem peter_ivanovich_satisfied_probability :
  satisfies_peter_ivanovich = 25 / 33 := 
sorry

-- Define the expected number of satisfied men
def expected_satisfied_men := men * (25 / 33)

-- Prove the expected number of satisfied men
theorem expected_satisfied_men_value :
  expected_satisfied_men = 1250 / 33 :=
sorry

end peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l602_602176


namespace correct_average_l602_602133

noncomputable def numbers : List ℕ := [12, 13, 14, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem correct_average :
  let sum := numbers.sum,
      count := numbers.length,
      incorrect_avg := 858.5454545454545,
      correct_avg := sum / count in
  correct_avg = 125781 :=
by
  -- Based on the conditions and calculations given in the problem
  sorry

end correct_average_l602_602133


namespace min_value_expression_l602_602436

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 8) : 
  (x + 3 * y) * (y + 3 * z) * (3 * x * z + 1) ≥ 72 :=
sorry

end min_value_expression_l602_602436


namespace rainy_days_l602_602123

theorem rainy_days
  (rain_on_first_day : ℕ) (rain_on_second_day : ℕ) (rain_on_third_day : ℕ) (sum_of_first_two_days : ℕ)
  (h1 : rain_on_first_day = 4)
  (h2 : rain_on_second_day = 5 * rain_on_first_day)
  (h3 : sum_of_first_two_days = rain_on_first_day + rain_on_second_day)
  (h4 : rain_on_third_day = sum_of_first_two_days - 6) :
  rain_on_third_day = 18 :=
by
  sorry

end rainy_days_l602_602123


namespace chord_length_l602_602794

variable (r ρ : ℝ)

-- Assuming r > 0 and ρ > 0.
def valid_rho (r ρ : ℝ) :=  ρ > 0 ∧ r > ρ

theorem chord_length (h : valid_rho r ρ) : 
  let AB := (2 * r * ρ) / (r - ρ) in
  AB = (2 * r * ρ) / (r - ρ) :=
by
  sorry

end chord_length_l602_602794


namespace tomorrow_is_saturday_l602_602861

def day := ℕ   -- Represent days as natural numbers for simplicity
def Monday := 0  -- Let's denote Monday as day 0 (Monday)
def one_week := 7  -- One week consists of 7 days

noncomputable def day_of_week (n : day) : day :=
  n % one_week  -- Calculate the day of the week based on modulo 7

theorem tomorrow_is_saturday
  (x : day)  -- the day before yesterday
  (hx : day_of_week (x + 5) = day_of_week Monday)  -- Monday is 5 days after the day before yesterday
  (today : day)  -- today
  (hy : day_of_week today = day_of_week (x + 2))  -- Today is 2 days after the day before yesterday
  : day_of_week (today + 1) = day_of_week 5 :=   -- Tomorrow will be Saturday (since Saturday is day 5)
by sorry

end tomorrow_is_saturday_l602_602861


namespace polyhedron_face_vertex_edge_l602_602030

def faces (Γ : ℕ → ℕ) : ℕ := sorry
def vertices (B : ℕ → ℕ) : ℕ := sorry

theorem polyhedron_face_vertex_edge
  (Γ : ℕ → ℕ)
  (B : ℕ → ℕ)
  (P : ℕ)
  (h_sum : (∑ h in Icc 3 Nat.infinity, h * B h) = 2 * P)
  (k_sum : (∑ k in Icc 3 Nat.infinity, k * Γ k) = 2 * P)
  : (∑ h in Icc 3 Nat.infinity, h * B h) = (∑ k in Icc 3 Nat.infinity, k * Γ k) :=
by
  sorry

end polyhedron_face_vertex_edge_l602_602030


namespace find_n_arctan_sum_eq_pi_over_4_l602_602296

theorem find_n_arctan_sum_eq_pi_over_4 :
  ∃ n : ℕ+, arctan (1/3) + arctan (1/4) + arctan (1/6) + arctan (1/(n : ℝ)) = (π / 4) ∧ n = 57 :=
begin
  use 57,
  repeat { norm_num, sorry },
end

end find_n_arctan_sum_eq_pi_over_4_l602_602296


namespace pie_eating_contest_l602_602115

theorem pie_eating_contest :
  let a := 5 / 6
  let b := 7 / 8
  let c := 2 / 3
  let max_pie := max a (max b c)
  let min_pie := min a (min b c)
  max_pie - min_pie = 5 / 24 :=
by
  sorry

end pie_eating_contest_l602_602115


namespace chocolate_game_winner_l602_602965

theorem chocolate_game_winner (m n : ℕ) (h_m : m = 6) (h_n : n = 8) :
  (∃ k : ℕ, (48 - 1) - 2 * k = 0) ↔ true :=
by
  sorry

end chocolate_game_winner_l602_602965


namespace geom_sequence_sum_l602_602033

-- Defining a geometric sequence with given conditions
def geometric_sequence := {a : ℕ → ℝ // ∃ q, a 2 = 2 ∧ a 5 = 1 / 4 ∧ ∀ n, a (n + 1) = a n * q}

-- Defining the sum to be proved
noncomputable def sum_formula (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (32 / 3) * (1 - (1 / (4 ^ n)))

-- Statement of the theorem
theorem geom_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) :
  (∀ n, a (n + 1) = a n * q) ∧ a 2 = 2 ∧ a 5 = 1 / 4 →
  (∑ i in finset.range n, a i * a (i + 1)) =
  sum_formula a n :=
sorry

end geom_sequence_sum_l602_602033


namespace center_of_symmetry_smallest_positive_period_interval_of_monotonic_increase_minimum_value_in_interval_l602_602730

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2*x) - 2 * sin x ^ 2

theorem center_of_symmetry (k : ℤ) : 
  ∃ c : ℝ, f (c - π / 12 + k * π / 2) = f (-c - π / 12 + k * π / 2) := sorry

theorem smallest_positive_period : 
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem interval_of_monotonic_increase (k : ℤ) : 
  ∀ x, ( -π/3 + k*π ≤ x ∧ x ≤ π/6 + k*π ) → deriv f x > 0 := sorry

theorem minimum_value_in_interval : 
  ( ∀ x ∈ set.Icc ( -π/2:ℝ ) 0, f x ≥ -3 ) ∧
  ( ∃ x ∈ set.Icc ( -π/2:ℝ ) 0, f x = -3 ∧ x = -π/3 ) := sorry

end center_of_symmetry_smallest_positive_period_interval_of_monotonic_increase_minimum_value_in_interval_l602_602730


namespace tomorrow_is_saturday_l602_602874

noncomputable def day_before_yesterday : string := "Wednesday"
noncomputable def today : string := "Friday"
noncomputable def tomorrow : string := "Saturday"

theorem tomorrow_is_saturday (dby : string) (tod : string) (tom : string) 
  (h1 : dby = "Wednesday") (h2 : tod = "Friday") (h3 : tom = "Saturday")
  (h_cond : "Monday" = dby + 5) : 
  tom = "Saturday" := 
sorry

end tomorrow_is_saturday_l602_602874


namespace principal_amount_is_26_l602_602104

-- Define the conditions
def rate : Real := 0.07
def time : Real := 6
def simple_interest : Real := 10.92

-- Define the simple interest formula
def simple_interest_formula (P R T : Real) : Real := P * R * T

-- State the theorem to prove
theorem principal_amount_is_26 : 
  ∃ (P : Real), simple_interest_formula P rate time = simple_interest ∧ P = 26 :=
by
  sorry

end principal_amount_is_26_l602_602104


namespace find_third_number_l602_602137

theorem find_third_number (N : ℤ) :
  (1274 % 12 = 2) ∧ (1275 % 12 = 3) ∧ (1285 % 12 = 1) ∧ ((1274 * 1275 * N * 1285) % 12 = 6) →
  N % 12 = 1 :=
by
  sorry

end find_third_number_l602_602137


namespace curtain_length_correct_l602_602262

-- Define the problem conditions in Lean
def room_height_feet : ℝ := 8
def feet_to_inches : ℝ := 12
def additional_material_inches : ℝ := 5

-- Define the target length of the curtains
def curtain_length_inches : ℝ :=
  (room_height_feet * feet_to_inches) + additional_material_inches

-- Statement to prove the length of the curtains is 101 inches.
theorem curtain_length_correct :
  curtain_length_inches = 101 := by
  sorry

end curtain_length_correct_l602_602262


namespace tan_ratio_sum_l602_602435

open Real

-- Definitions based on the conditions in a)
def tan_sum (x y : ℝ) : Prop := (tan x + tan y = 3)
def sec_prod (x y : ℝ) : Prop := (sec x * sec y = 5)

-- The statement to be proved
theorem tan_ratio_sum (x y : ℝ) (h1 : tan_sum x y) (h2 : sec_prod x y) : 
  (tan x / tan y + tan y / tan x) = 223 := 
sorry

end tan_ratio_sum_l602_602435


namespace pricePerRedStamp_l602_602853

namespace StampCollection

-- Definitions for the conditions
def totalRedStamps : ℕ := 20
def soldRedStamps : ℕ := 20
def totalBlueStamps : ℕ := 80
def soldBlueStamps : ℕ := 80
def pricePerBlueStamp : ℝ := 0.8
def totalYellowStamps : ℕ := 7
def pricePerYellowStamp : ℝ := 2
def totalTargetEarnings : ℝ := 100

-- Derived definitions from conditions
def earningsFromBlueStamps : ℝ := soldBlueStamps * pricePerBlueStamp
def earningsFromYellowStamps : ℝ := totalYellowStamps * pricePerYellowStamp
def earningsRequiredFromRedStamps : ℝ := totalTargetEarnings - (earningsFromBlueStamps + earningsFromYellowStamps)

-- The statement asserting the main proof obligation
theorem pricePerRedStamp :
  (earningsRequiredFromRedStamps / soldRedStamps) = 1.10 :=
sorry

end StampCollection

end pricePerRedStamp_l602_602853


namespace percentage_increase_in_ear_piercing_l602_602416

def cost_of_nose_piercing : ℕ := 20
def noses_pierced : ℕ := 6
def ears_pierced : ℕ := 9
def total_amount_made : ℕ := 390

def cost_of_ear_piercing : ℕ := (total_amount_made - (noses_pierced * cost_of_nose_piercing)) / ears_pierced

def percentage_increase (original new : ℕ) : ℚ := ((new - original : ℚ) / original) * 100

theorem percentage_increase_in_ear_piercing : 
  percentage_increase cost_of_nose_piercing cost_of_ear_piercing = 50 := 
by 
  sorry

end percentage_increase_in_ear_piercing_l602_602416


namespace max_tan_A_l602_602708

-- The problem setup
variables {A B : ℝ}
variable (h1 : A ∈ Ioo 0 (π / 2))
variable (h2 : B ∈ Ioo 0 (π / 2))
variable (h3 : sin A / sin B = sin (A + B))

-- The required maximum value of tan A
theorem max_tan_A : tan A ≤ 4 / 3 :=
by
  sorry

end max_tan_A_l602_602708


namespace sin_cos_product_l602_602752

-- Define the problem's main claim
theorem sin_cos_product (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 :=
by
  have h1 : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := Real.sin_square_add_cos_square x
  sorry

end sin_cos_product_l602_602752


namespace min_smallest_abs_sum_l602_602655

noncomputable def smallest_abs_sum {a b c d : ℤ} (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_cond : (λ (M : Matrix (Fin 2) (Fin 2) ℤ), M.mul M = Matrix.diagonal (λ _, 9))
    (Matrix.of ![![a, b], ![c, d]])) : ℤ :=
|a| + |b| + |c| + |d|

theorem min_smallest_abs_sum {a b c d : ℤ} (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_cond : (λ (M : Matrix (Fin 2) (Fin 2) ℤ), M.mul M = Matrix.diagonal (λ _, 9))
    (Matrix.of ![![a, b], ![c, d]])) : smallest_abs_sum h h_cond = 8 :=
sorry

end min_smallest_abs_sum_l602_602655


namespace count_desired_multiples_l602_602363

-- Define the condition of being a positive integer not exceeding 200
def is_valid (n : ℕ) : Prop := n > 0 ∧ n ≤ 200

-- Define the condition of being a multiple of 2
def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

-- Define the condition of being a multiple of 5
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

-- Define the condition of being a multiple of 6
def is_multiple_of_6 (n : ℕ) : Prop := n % 6 = 0

-- Define the condition of satisfying the problem's main criteria
def is_desired_multiple (n : ℕ) : Prop := 
  is_valid n ∧ (is_multiple_of_2 n ∨ is_multiple_of_5 n) ∧ ¬is_multiple_of_6 n

-- Problem statement: the number of desired multiples
theorem count_desired_multiples : 
  (finset.filter is_desired_multiple (finset.range 201)).card = 87 := 
sorry

end count_desired_multiples_l602_602363


namespace two_co_presidents_probability_l602_602800

-- Define the conditions
structure Club :=
(number_of_students : ℕ)
(co_presidents : finset ℕ)

def Club1 : Club := ⟨6, {0, 1, 2}⟩
def Club2 : Club := ⟨9, {0, 1, 2}⟩
def Club3 : Club := ⟨10, {0, 1, 2}⟩
def Club4 : Club := ⟨12, {0, 1, 2}⟩

def Clubs := [Club1, Club2, Club3, Club4]

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

noncomputable def probability_of_two_co_presidents (club : Club) : ℚ :=
  if club.number_of_students < 4 then 0 else
  let n := club.number_of_students in
  let k := club.co_presidents.card in
  (binomial k 2 * binomial (n - k) 2 : ℚ) / binomial n 4

-- Compute the final probability considering club selection
noncomputable def final_probability : ℚ :=
  (probability_of_two_co_presidents Club1 +
   probability_of_two_co_presidents Club2 +
   probability_of_two_co_presidents Club3 +
   probability_of_two_co_presidents Club4) / 4

-- Prove the desired probability
theorem two_co_presidents_probability : 
  final_probability = 14753 / 40000 := 
sorry

end two_co_presidents_probability_l602_602800


namespace sum_geometric_sequence_l602_602941

variable {α : Type*} [LinearOrderedField α]

theorem sum_geometric_sequence {S : ℕ → α} {n : ℕ} (h1 : S n = 3) (h2 : S (3 * n) = 21) :
    S (2 * n) = 9 := 
sorry

end sum_geometric_sequence_l602_602941


namespace square_perimeter_l602_602597

noncomputable def side_length_of_square_with_area (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def perimeter_of_square_with_side (side : ℝ) : ℝ :=
  4 * side

theorem square_perimeter {area : ℝ} (h_area : area = 625) :
  perimeter_of_square_with_side (side_length_of_square_with_area area) = 100 :=
by
  have h_side_length : side_length_of_square_with_area area = 25 := by
    rw [side_length_of_square_with_area, real.sqrt, h_area]
    norm_num
  rw [perimeter_of_square_with_side, h_side_length]
  norm_num
  sorry

end square_perimeter_l602_602597


namespace trains_crossing_time_l602_602538

noncomputable def length_first_train : ℝ := 700
noncomputable def speed_first_train : ℝ := 120 * 1000 / 3600
noncomputable def length_second_train : ℝ := 1000
noncomputable def speed_second_train : ℝ := 80 * 1000 / 3600
noncomputable def combined_length : ℝ := length_first_train + length_second_train
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train

theorem trains_crossing_time : combined_length / relative_speed ≈ 30.58 := by
  sorry

end trains_crossing_time_l602_602538


namespace square_perimeter_l602_602606

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end square_perimeter_l602_602606


namespace polynomial_characterization_l602_602286

theorem polynomial_characterization (P : ℝ[X]) : P 2017 = 2016 ∧ ∀ x : ℝ, (P.eval x + 1)^2 = P.eval (x^2 + 1) → P = (λ x, x - 1) :=
by
  sorry

end polynomial_characterization_l602_602286


namespace sin_cos_value_l602_602770

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end sin_cos_value_l602_602770


namespace max_value_when_a_neg4_num_roots_f_eq_zero_l602_602347

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log x + x^2

theorem max_value_when_a_neg4 : 
  ∃ x ∈ Set.Icc (1 : ℝ) Real.exp, ∀ y ∈ Set.Icc (1 : ℝ) Real.exp, (f (-4) y ≤ f (-4) x) ∧ x = Real.exp ∧ f (-4) x = Real.exp^2 - 4 :=
sorry

theorem num_roots_f_eq_zero (a : ℝ) :
  (0 ≤ a → ∀ x ∈ Set.Icc (1 : ℝ) Real.exp, f a x ≠ 0 ∧ (∀ x ∈ Set.Icc (1 : ℝ) Real.exp, ∃ (y : ℝ), ¬(f a y = 0))) ∧
  (-2 ≤ a ∧ a < 0 → ∀ x ∈ Set.Icc (1 : ℝ) Real.exp, f a x ≠ 0) ∧
  (a ≤ -2 * Real.exp^2 → ∃ x ∈ Set.Icc (1 : ℝ) Real.exp, f a x = 0 ∧ (∀ y ∈ Set.Icc (1 : ℝ) Real.exp, y ≠ x → f a y ≠ 0)) ∧
  (-2 * Real.exp^2 < a ∧ a < -2 → 
    ((-2 * Real.exp < a ∧ a < -2 → ∀ x ∈ Set.Icc (1 : ℝ) Real.exp, ¬(f a x = 0)) ∧
    (a = -2 * Real.exp → ∃ x ∈ Set.Icc (1 : ℝ) Real.exp, f a x = 0 ∧ (∀ y ∈ Set.Icc (1 : ℝ) Real.exp, y ≠ x → f a y ≠ 0)) ∧
    (-Real.exp^2 ≤ a ∧ a < -2 * Real.exp → ∃ x ∈ Set.Icc (1 : ℝ) Real.exp, f a x = 0 ∧ 
      (∃ z ≠ x, z ∈ Set.Icc (1 : ℝ) Real.exp ∧ f a z = 0))) ∧
    (-2 * Real.exp^2 < a ∧ a < -Real.exp^2 → ∃ x ∈ Set.Icc (1 : ℝ) Real.exp, f a x = 0 ∧
      ∀ y ∈ Set.Icc (1 : ℝ) Real.exp, y ≠ x → f a y ≠ 0))
sorry

end max_value_when_a_neg4_num_roots_f_eq_zero_l602_602347


namespace computation_h_5_0_l602_602835

noncomputable def f (x : ℝ) := Real.sin (Real.pi * x)
noncomputable def g (x : ℝ) := Real.cos (Real.pi * x)
noncomputable def h1 (x : ℝ) := g (f x)
noncomputable def h : ℕ → ℝ → ℝ
| 1, x := h1 x
| (n+1), x := h1 (h n x)

theorem computation_h_5_0.5 : h 5 0.5 = -1 :=
sorry

end computation_h_5_0_l602_602835


namespace product_of_real_values_r_l602_602298

theorem product_of_real_values_r {x r : ℝ} (h : x ≠ 0) (heq : (1 / (3 * x)) = ((r - x) / 8)) :
  (∃! x : ℝ, 24 * x^2 - 8 * r * x + 24 = 0) →
  r = 6 ∨ r = -6 ∧ (r * -r) = -36 :=
by
  sorry

end product_of_real_values_r_l602_602298


namespace square_perimeter_l602_602604

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end square_perimeter_l602_602604


namespace valid_votes_other_candidate_l602_602400

theorem valid_votes_other_candidate (total_votes : ℕ) (percentage_invalid : ℝ) (percentage_received_first : ℝ) 
    (h_total_votes : total_votes = 7500)
    (h_percentage_invalid : percentage_invalid = 20)
    (h_percentage_received_first : percentage_received_first = 55) : 
    (valid_votes_other_candidate : ℕ) = 
    total_votes * (1 - (percentage_invalid / 100)) * (1 - (percentage_received_first / 100)) := 
begin
    sorry
end

end valid_votes_other_candidate_l602_602400


namespace smallest_uv_non_factor_of_48_l602_602963

theorem smallest_uv_non_factor_of_48 :
  ∃ (u v : ℕ) (hu : u ∣ 48) (hv : v ∣ 48), u ≠ v ∧ ¬ (u * v ∣ 48) ∧ u * v = 18 :=
sorry

end smallest_uv_non_factor_of_48_l602_602963


namespace largest_area_Q_l602_602887

noncomputable def pointP := (0, 0, sqrt 3)
noncomputable def planeA := {p : (ℝ × ℝ × ℝ) | p.2 = 0}
noncomputable def regionQ := {p : (ℝ × ℝ × ℝ) | 1 ≤ (p.1^2 + p.3^2) ∧ (p.1^2 + p.3^2) ≤ 9}

theorem largest_area_Q : 
  ∃ Q : set (ℝ × ℝ × ℝ), Q = regionQ ∧ (∀ p ∈ Q, ∃ l, line_through_point_plane l pointP planeA ∧ angle_between_line_plane l planeA ∈ Icc (real.pi / 6) (real.pi / 3)) → 
  ∫⁻ x in Q, 1 ≤ 8 * real.pi := sorry

end largest_area_Q_l602_602887


namespace point_in_second_quadrant_l602_602809

def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

theorem point_in_second_quadrant : is_in_second_quadrant (-2) 3 :=
by
  sorry

end point_in_second_quadrant_l602_602809


namespace find_b_l602_602092

theorem find_b (b : ℚ) (H : ∃ x y : ℚ, x = 3 ∧ y = -7 ∧ b * x + (b - 1) * y = b + 3) : 
  b = 4 / 5 := 
by
  sorry

end find_b_l602_602092


namespace concyclic_and_common_tangent_intersection_l602_602018

variables {ℝ : Type*} [linear_ordered_field ℝ] [real_plane ℝ]

noncomputable def is_concyclic (K L M N : ℝ) : Prop :=
∃ ω : circle ℝ, K ∈ ω ∧ L ∈ ω ∧ M ∈ ω ∧ N ∈ ω

noncomputable def common_tangent_point (P Q : circle ℝ) (K L M N : ℝ) : Prop :=
∃ H : ℝ, is_intersection_point_of_tangents H P Q ∧ ∀ pt ∈ {K, L, M, N}, pt ∈ circle ℝ H -- H should be the intersection point of tangents

theorem concyclic_and_common_tangent_intersection {A B C D K L M N : ℝ} 
  (d : line ℝ) (a b c d : tangent ℝ) 
  (P Q : circle ℝ) :
  collinear A B C D ∧ segment AB = segment CD ∧
  P through {A, B} ∧
  Q through {C, D} ∧
  intersection_point a c K ∧ intersection_point a d L ∧
  intersection_point b c M ∧ intersection_point b d N →
  (is_concyclic K L M N) ∧ (common_tangent_point P Q K L M N) :=
sorry

end concyclic_and_common_tangent_intersection_l602_602018


namespace length_of_AB_l602_602792

-- Define the triangle ABC with the given conditions
variables {A B C : Type} [normed_group A] [normed_space ℝ A]
variables {a b c : A}
variable (T : triangle A B C)
variable (hypot : right_angle a)
variable (tanC : real.tan (angle a c b) = 3)
variable (BC : dist b c = 30)

-- Theorem statement to determine the length of side AB
theorem length_of_AB : dist a b = 9 * real.sqrt 10 :=
by
  -- setup theorem proof skeleton (this part is done for you, but you don't need the actual proof)
  sorry

end length_of_AB_l602_602792


namespace log_eq_condition_pq_l602_602633

theorem log_eq_condition_pq :
  ∀ (p q : ℝ), p > 0 → q > 0 → (Real.log p + Real.log q = Real.log (2 * p + q)) → p = 3 ∧ q = 3 :=
by
  intros p q hp hq hlog
  sorry

end log_eq_condition_pq_l602_602633


namespace abe_spending_on_wages_l602_602623

theorem abe_spending_on_wages
    (B : ℕ) (food_pct : ℚ) (supplies_pct : ℚ)
    (hB : B = 3000)
    (h_food_pct : food_pct = 1 / 3)
    (h_supplies_pct : supplies_pct = 1 / 4) :
    B - (B * food_pct).natAbs - (B * supplies_pct).natAbs = 1250 := by
  sorry

end abe_spending_on_wages_l602_602623


namespace question1_question2_question3_question4_question5_main_l602_602048

variable {a x y : ℝ}

theorem question1 : a * (x - y) = a * x - a * y := by sorry
theorem question2 : a ^ (x - y) ≠ a ^ x - a ^ y := by sorry
theorem question3 : log (x - y) ≠ log x - log y := by sorry
theorem question4 : log x / log y ≠ log x - log y := by sorry
theorem question5 : a * (x * y) ≠ a * x * a * y := by sorry

theorem main : 
  (a * (x - y) = a * x - a * y) ∧ 
  (a ^ (x - y) ≠ a ^ x - a ^ y) ∧
  (log (x - y) ≠ log x - log y) ∧ 
  (log x / log y ≠ log x - log y) ∧
  (a * (x * y) ≠ a * x * a * y) :=
by
  apply And.intro question1
  apply And.intro question2
  apply And.intro question3
  apply And.intro question4
  exact question5

end question1_question2_question3_question4_question5_main_l602_602048


namespace line_through_points_l602_602388

theorem line_through_points (a b : ℝ) (h1 : 3 = a * 2 + b) (h2 : 19 = a * 6 + b) :
  a - b = 9 :=
sorry

end line_through_points_l602_602388


namespace gray_region_area_is_96pi_l602_602131

noncomputable def smaller_circle_diameter : ℝ := 4

noncomputable def smaller_circle_radius : ℝ := smaller_circle_diameter / 2

noncomputable def larger_circle_radius : ℝ := 5 * smaller_circle_radius

noncomputable def area_of_larger_circle : ℝ := Real.pi * (larger_circle_radius ^ 2)

noncomputable def area_of_smaller_circle : ℝ := Real.pi * (smaller_circle_radius ^ 2)

noncomputable def area_of_gray_region : ℝ := area_of_larger_circle - area_of_smaller_circle

theorem gray_region_area_is_96pi : area_of_gray_region = 96 * Real.pi := by
  sorry

end gray_region_area_is_96pi_l602_602131


namespace vandermonde_identity_anti_vandermonde_identity_l602_602039

open Nat

-- Define binomial coefficient following the convention \(\binom{n}{k}=0\) for \(k > n\)
def binom : ℕ → ℕ → ℕ
| n, k => if k > n then 0 else nat.choose n k

theorem vandermonde_identity (a b n : ℕ) :
  ∑ k in finset.range (n + 1), binom a k * binom b (n - k) = binom (a + b) n := sorry

theorem anti_vandermonde_identity (a b n : ℕ) :
  ∑ k in finset.range (n + 1), binom k a * binom (n - k) b = binom (n + 1) (a + b + 1) := sorry

end vandermonde_identity_anti_vandermonde_identity_l602_602039


namespace clock_hands_angle_120_degrees_l602_602303

def angleBetweenHands (hour minute : ℕ) : ℝ :=
  let h_angle := (hour % 12 + minute / 60.0) * 30.0
  let m_angle := minute * 6.0
  let angle := abs (h_angle - m_angle)
  if angle > 180 then 360 - angle else angle

theorem clock_hands_angle_120_degrees : ∃ (t : ℕ), t = 8 * 60 + 22 ∧ angleBetweenHands 8 t = 120 := by
  sorry

end clock_hands_angle_120_degrees_l602_602303


namespace reservoir_capacity_l602_602233

-- Definitions based on the conditions
def storm_deposit : ℚ := 120 * 10^9
def final_full_percentage : ℚ := 0.85
def initial_full_percentage : ℚ := 0.55
variable (C : ℚ) -- total capacity of the reservoir in gallons

-- The statement we want to prove
theorem reservoir_capacity :
  final_full_percentage * C - initial_full_percentage * C = storm_deposit →
  C = 400 * 10^9
:= by
  sorry

end reservoir_capacity_l602_602233


namespace find_x_l602_602706

theorem find_x (x : ℕ) : 3 * 2^x + 5 * 2^x = 2048 → x = 8 := by
  sorry

end find_x_l602_602706


namespace jed_change_each_bill_value_l602_602415

theorem jed_change_each_bill_value (n_games : ℕ) (cost_per_game : ℕ) (total_money : ℕ) (change_bills : ℕ) 
  (total_cost : n_games * cost_per_game = 90)
  (total_paid : total_money = 100)
  (total_change : total_money - total_change_calc = 10)
  (num_bills : change_bills = 2) : 
  change_each_bill = 5 :=
by
  -- Conditions provided
  have total_spent : total_cost = n_games * cost_per_game := sorry
  have total_received_change : total_change = total_money - total_spent := sorry
  have total_received_change_eq : total_received_change = 10 := sorry
  have each_bill_value : change_each_bill = total_received_change / change_bills := sorry
  -- Conclude the theorem
  exact each_bill_value

end jed_change_each_bill_value_l602_602415


namespace pool_houses_count_l602_602393

-- Definitions based on conditions
def total_houses : ℕ := 65
def num_garage : ℕ := 50
def num_both : ℕ := 35
def num_neither : ℕ := 10
def num_pool : ℕ := total_houses - num_garage - num_neither + num_both

theorem pool_houses_count :
  num_pool = 40 := by
  -- Simplified form of the problem expressed in Lean 4 theorem statement.
  sorry

end pool_houses_count_l602_602393


namespace solve_system_of_equations_l602_602898

theorem solve_system_of_equations (x y z : ℝ) :
  (x = (y + 1) / (3 * y - 5)) ∧ (y = (3 * z - 2) / (2 * z - 3)) ∧ (z = (3 * x - 1) / (x - 1))
  ↔ (x = 0 ∧ y = -1 ∧ z = 1) ∨ (x = 3 ∧ y = 2 ∧ z = 4) :=
by
  -- Definitions and conditions used in the given system of equations
  def eq1 := x = (y + 1) / (3 * y - 5)
  def eq2 := y = (3 * z - 2) / (2 * z - 3)
  def eq3 := z = (3 * x - 1) / (x - 1)
  sorry


end solve_system_of_equations_l602_602898


namespace find_initial_books_each_l602_602472

variable (x : ℝ)
variable (sandy_books : ℝ := x)
variable (tim_books : ℝ := 2 * x + 33)
variable (benny_books : ℝ := 3 * x - 24)
variable (total_books : ℝ := 100)

theorem find_initial_books_each :
  sandy_books + tim_books + benny_books = total_books → x = 91 / 6 := by
  sorry

end find_initial_books_each_l602_602472


namespace markov_inequality_l602_602327

variables {Ω : Type*} {X : Ω → ℝ} {n : ℕ} {p : ℝ}
  (G : MeasureTheory.Measure Ω)
  [MeasureTheory.ProbabilityMeasure G]
  [MeasureTheory.HasFiniteIntegral X]

theorem markov_inequality
  (X_nonneg : ∀ ω, 0 ≤ X ω)
  (a : ℝ)
  (a_pos : 0 < a) :
  MeasureTheory.measure (λ ω, X ω ≥ a) ≤ (MeasureTheory.expectation X) / a :=
by
  sorry

end markov_inequality_l602_602327


namespace mary_sugar_salt_difference_l602_602451

def problem_statement (flour_recipe sugar_recipe salt_recipe flour_added : ℕ) : ℕ :=
sugar_recipe - salt_recipe

theorem mary_sugar_salt_difference :
  ∀ (flour_recipe sugar_recipe salt_recipe flour_added : ℕ),
  flour_recipe = 6 →
  sugar_recipe = 8 →
  salt_recipe = 7 →
  flour_added = 5 →
  problem_statement flour_recipe sugar_recipe salt_recipe flour_added = 1 :=
by
  intros
  rw [H1, H2, H3]
  exact rfl

end mary_sugar_salt_difference_l602_602451


namespace exists_irrational_between_1_and_5_l602_602153

theorem exists_irrational_between_1_and_5 : ∃ (x : ℝ), irrational x ∧ 1 < x ∧ x < 5 := 
sorry

end exists_irrational_between_1_and_5_l602_602153


namespace remaining_days_temperature_l602_602909

theorem remaining_days_temperature :
  let avg_temp := 60
  let total_days := 7
  let temp_day1 := 40
  let temp_day2 := 40
  let temp_day3 := 40
  let temp_day4 := 80
  let temp_day5 := 80
  let total_temp := avg_temp * total_days
  let temp_first_five_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5
  total_temp - temp_first_five_days = 140 :=
by
  -- proof is omitted
  sorry

end remaining_days_temperature_l602_602909


namespace row_time_ratio_l602_602523

-- Define the speed of the boat in still water
def V_b : ℝ := 36

-- Define the speed of the stream
def V_s : ℝ := 12

-- Define the effective speed upstream
def V_up : ℝ := V_b - V_s

-- Define the effective speed downstream
def V_down : ℝ := V_b + V_s

-- Define the ratio of time taken to row upstream to downstream
def time_ratio : ℝ := V_down / V_up

-- Assert the ratio is 2:1
theorem row_time_ratio : time_ratio = 2 :=
  by
    -- Proof is omitted
    sorry

end row_time_ratio_l602_602523


namespace area_of_quadrilateral_is_correct_l602_602338

noncomputable def circle_area_quadrilateral : ℝ :=
  let center_C := (3 : ℝ, 4 : ℝ)
  let radius := 5
  let point_P := (2 : ℝ, 6 : ℝ)
  let longest_chord := 2 * radius
  let distance_center_P := Real.sqrt ((3 - 2)^2 + (4 - 6)^2)
  let shortest_chord := 2 * Real.sqrt (radius^2 - distance_center_P^2)
  let area_ABCD := (1 / 2) * longest_chord * shortest_chord
  area_ABCD

theorem area_of_quadrilateral_is_correct :
  circle_area_quadrilateral = 20 * Real.sqrt 5 := sorry

end area_of_quadrilateral_is_correct_l602_602338


namespace tom_sold_price_l602_602120

noncomputable def original_price : ℝ := 200
noncomputable def tripled_price (price : ℝ) : ℝ := 3 * price
noncomputable def sold_price (price : ℝ) : ℝ := 0.4 * price

theorem tom_sold_price : sold_price (tripled_price original_price) = 240 := 
by
  sorry

end tom_sold_price_l602_602120


namespace moles_of_NH4SO4_formed_l602_602295

noncomputable theory

-- Define the balanced chemical equation
def balanced_eq (n_NH3 n_H2SO4 n_NH4SO4 : ℕ) : Prop :=
  2 * n_NH3 + n_H2SO4 = n_NH4SO4

-- Define the number of moles of reactants
def moles_of_reactants (n_NH3 n_H2SO4 : ℕ) : Prop :=
  n_NH3 = 4 ∧ n_H2SO4 = 2

-- Main theorem to prove
theorem moles_of_NH4SO4_formed (n_NH3 n_H2SO4 n_NH4SO4 : ℕ)
  (h_reactants : moles_of_reactants n_NH3 n_H2SO4)
  (h_balanced : balanced_eq n_NH3 n_H2SO4 n_NH4SO4) :
  n_NH4SO4 = 2 :=
sorry

end moles_of_NH4SO4_formed_l602_602295


namespace sin_cos_identity_l602_602762

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602762


namespace shop_width_l602_602924

theorem shop_width
  (monthly_rent : ℝ)
  (length_shop : ℝ)
  (annual_rent_per_sqft : ℝ)
  (annual_rent : ℝ := monthly_rent * 12)
  (area_shop : ℝ := annual_rent / annual_rent_per_sqft)
  (width_shop : ℝ := area_shop / length_shop) :
  monthly_rent = 1440 → length_shop = 18 → annual_rent_per_sqft = 48 → width_shop = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end shop_width_l602_602924


namespace number_of_zeros_l602_602926

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x - 3

theorem number_of_zeros (b : ℝ) : 
  ∃ x₁ x₂ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ x₁ ≠ x₂ := by
  sorry

end number_of_zeros_l602_602926


namespace second_smallest_sum_of_cubes_twice_l602_602442

theorem second_smallest_sum_of_cubes_twice : 
  ∃ n : ℕ, (∃ (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0), a^3 + b^3 = n ∧ c^3 + d^3 = n ∧ (a, b) ≠ (c, d) ∧ (a, b) ≠ (d, c)) 
         ∧ 
         ∀ n' : ℕ, (n' < n →  ∃ (a' b' c' d' : ℕ) (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0) (hd' : d' > 0), a'^3 + b'^3 = n' ∧ c'^3 + d'^3 = n' ∧ (a', b') ≠ (c', d') ∧ (a', b') ≠ (d', c')) 
             → 
             (n' = 1729)) 
  ∧ n = 4104 :=
sorry

end second_smallest_sum_of_cubes_twice_l602_602442


namespace exists_integer_n_for_divisors_num_divisors_n2_minus_4_l602_602572

-- Define necessary conditions and theorems

-- Part (a)
theorem exists_integer_n_for_divisors (k : ℕ) (h : k = 1 ∨ k = 2 ∨ k = 3) :
    ∃ (n : ℤ), nat.num_divisors (n^2 - k) = 10 := sorry

-- Part (b)
theorem num_divisors_n2_minus_4 (n : ℤ) :
    ¬(nat.num_divisors (n^2 - 4) = 10) := sorry

end exists_integer_n_for_divisors_num_divisors_n2_minus_4_l602_602572


namespace mary_cut_10_roses_l602_602953

-- Define the initial and final number of roses
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses cut as the difference between final and initial
def roses_cut : ℕ :=
  final_roses - initial_roses

-- Theorem stating the number of roses cut by Mary
theorem mary_cut_10_roses : roses_cut = 10 := by
  sorry

end mary_cut_10_roses_l602_602953


namespace integral_of_x_l602_602668

variable (f : ℝ → ℝ := id)

theorem integral_of_x :
  ∫ x in 0..1, f x = 1 / 2 :=
by
  have h : f = λ x, x := by simp [f]
  rw [h]
  simp
  sorry

end integral_of_x_l602_602668


namespace sin_cos_identity_l602_602765

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602765


namespace hamburger_combinations_l602_602358

def number_of_condiments := 8
def condiment_combinations := 2 ^ number_of_condiments
def number_of_meat_patties := 4
def total_hamburgers := number_of_meat_patties * condiment_combinations

theorem hamburger_combinations :
  total_hamburgers = 1024 :=
by
  sorry

end hamburger_combinations_l602_602358


namespace unique_zero_of_g_l602_602728

theorem unique_zero_of_g :
  ∃! x : ℝ, 1 < x ∧ x < 2 ∧ (g x = 0)
    where g (x : ℝ) := Real.log x - 1/x := by
  sorry

end unique_zero_of_g_l602_602728


namespace min_speed_to_catch_up_l602_602904

theorem min_speed_to_catch_up (x : ℝ) :
  let speed_A := 30
  let travel_time_A := 48 / 60
  let messenger_speed := 72
  let team_speed := 30
  let catch_up_time := 25 / 60
  let distance_A := speed_A * travel_time_A
  let messenger_catch_up_time := distance_A / messenger_speed
  let additional_team_distance := team_speed * messenger_catch_up_time
  x ≥ 54 :=
  by
  -- The distance A has traveled in 48 minutes (24 kilometers)
  have distance_A_eq : distance_A = 24, by sorry
  -- The time the messenger takes to catch up with A (1/3 hour)
  have messenger_catch_up_time_eq : messenger_catch_up_time = 1 / 3, by sorry
  -- The additional distance the team travels while the messenger catches up with A (10 kilometers)
  have team_additional_distance_eq : additional_team_distance = 10, by sorry
  -- The minimum speed required for A to catch up with the team in 25 minutes
  have min_speed : (catch_up_time * (x - team_speed)) ≥ additional_team_distance ↔ x ≥ 54, by sorry
  exact min_speed.1 rfl

end min_speed_to_catch_up_l602_602904


namespace proof_probability_sum_16_l602_602417

noncomputable def probability_sum_16 : ℚ :=
  let coin_sides := {5, 15}
  let die_sides := {1, 2, 3, 4, 5, 6}
  let favorable_coin_side := 15
  let favorable_die_side := 1
  
  -- Probability of flipping 15 on the coin
  let prob_coin := if favorable_coin_side ∈ coin_sides then (1 : ℚ) / coin_sides.card else 0
  
  -- Probability of rolling 1 on the die
  let prob_die := if favorable_die_side ∈ die_sides then (1 : ℚ) / die_sides.card else 0

  -- The combined probability to get the sum of 16
  prob_coin * prob_die

theorem proof_probability_sum_16 : probability_sum_16 = 1 / 12 := by
  -- Sketch of proof: There are 2 sides on the coin and 6 sides on the die.
  -- The probability of flipping a 15 is 1/2 and rolling a 1 is 1/6.
  -- Hence, the combined probability is (1/2) * (1/6) = 1/12.
  sorry

end proof_probability_sum_16_l602_602417


namespace trisectable_angles_l602_602152

theorem trisectable_angles :
  ∃ (angles : set ℝ), 
    (∀ α ∈ angles, ∃ β, 3 * β = α ∧ β ∈ angles) ∧
    ({180, 135, 90, 45, 22.5} ⊆ angles) :=
by
  sorry

end trisectable_angles_l602_602152


namespace find_adult_tickets_l602_602827

noncomputable def number_of_adult_tickets :=
  ∃ (A C : ℕ), 
    A + C = 7 ∧
    21 * A + 14 * C = 119 ∧
    A = 3

theorem find_adult_tickets : number_of_adult_tickets :=
by 
  existsi 3
  existsi 4
  simp
  sorry -- the full proof would be developed here

end find_adult_tickets_l602_602827


namespace find_a_l602_602313

noncomputable def curve := λ (a : ℝ) (x : ℝ), a * x^2 + 1/3

def hyperbola_asymptotes (x : ℝ) : set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = (2/3) * x ∨ p.2 = -(2/3) * x}

def hyperbola (x y : ℝ) : Prop := (x^2) / 9 - (y^2) / 4 = 1

theorem find_a (a : ℝ) : (∀ x : ℝ, (curve a x, curve a (-x)) ∈ (hyperbola_asymptotes x)) → a = 1/3 := 
by
  intros h
  sorry

end find_a_l602_602313


namespace inequality_holds_for_all_x_l602_602690

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (x^2 + m * x - 1) / (2 * x^2 - 2 * x + 3) < 1) ↔ -6 < m ∧ m < 2 := 
sorry -- Proof to be provided

end inequality_holds_for_all_x_l602_602690


namespace find_dividend_l602_602394

theorem find_dividend (R D Q V : ℤ) (hR : R = 5) (hD1 : D = 3 * Q) (hD2 : D = 3 * R + 3) : V = D * Q + R → V = 113 :=
by 
  sorry

end find_dividend_l602_602394


namespace term_50th_of_sequence_l602_602650

def contains_digit_four (n : ℕ) : Prop :=
  n.digits 10 |>.contains 4

def sequence_special : List ℕ :=
  (List.range 2000).filter (λ n, n % 4 = 0 ∧ contains_digit_four n)

theorem term_50th_of_sequence : sequence_special.get? 49 = some 448 :=
by
  sorry

end term_50th_of_sequence_l602_602650


namespace total_interest_at_end_of_tenth_year_l602_602938

-- Definitions and conditions based on the problem
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100
def trebled (P : ℝ) : ℝ := 3 * P

-- The main goal: Prove that the total interest at the end of the tenth year is Rs. 650.
theorem total_interest_at_end_of_tenth_year (P R : ℝ) 
    (h1 : simple_interest P R 10 = 1000)
    (h2 : simple_interest (trebled P) R 5 = 150) : 
    simple_interest P R 5 + simple_interest (trebled P) R 5 = 650 := 
by 
    sorry

end total_interest_at_end_of_tenth_year_l602_602938


namespace dosage_range_l602_602514

theorem dosage_range (d : ℝ) (h : 60 ≤ d ∧ d ≤ 120) : 15 ≤ (d / 4) ∧ (d / 4) ≤ 30 :=
by
  sorry

end dosage_range_l602_602514


namespace find_smallest_q_l602_602430

noncomputable def smallest_q (s : ℝ) (h_s1 : s > 0) (h_s2 : s < 1 / 2000) :=
  let p := ((8 : ℝ) + s) ^ 4 in
  let q := 8 in
  q

theorem find_smallest_q (s : ℝ) (h_s1 : s > 0) (h_s2 : s < 1 / 2000) :
  smallest_q s h_s1 h_s2 = 8 :=
sorry

end find_smallest_q_l602_602430


namespace angle_bisectors_intersection_on_midline_l602_602467

variables {A B C D M N E : Type*}
variables [InCircleQuadrilateral A B C D]
variables [IntersectionOfDiagonals E A C B D]
variables [Midpoint M A C]
variables [Midpoint N B D]

theorem angle_bisectors_intersection_on_midline
  (inscribed : ℝ) :
  ∃ P, angle_bisectors_intersection P A B C D ∧ lies_on_line P M N :=
sorry

end angle_bisectors_intersection_on_midline_l602_602467


namespace tomorrow_is_Saturday_l602_602857
noncomputable theory

def day_of_week := ℕ
def Monday : day_of_week := 0
def Tuesday : day_of_week := 1
def Wednesday : day_of_week := 2
def Thursday : day_of_week := 3
def Friday : day_of_week := 4
def Saturday : day_of_week := 5
def Sunday : day_of_week := 6

def days_after (d : day_of_week) (n : ℕ) : day_of_week := (d + n) % 7

-- The condition: Monday is five days after the day before yesterday.
def day_before_yesterday := Wednesday
def today := days_after day_before_yesterday 2
def tomorrow := days_after today 1

theorem tomorrow_is_Saturday (h: days_after day_before_yesterday 5 = Monday) : tomorrow = Saturday := 
by {
  sorry
}

end tomorrow_is_Saturday_l602_602857


namespace conditional_probability_problem_l602_602691

-- Definitions of the problem:
variable {Ω : Type*}
variable choiceSpace : set (Ω × Ω) := { (a, b) | a ≠ b ∧ a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} }
variable eventA : set (Ω × Ω) := { (a, b) | (a + b).even }
variable eventB : set (Ω × Ω) := { (a, b) | a % 2 = 0 ∧ b % 2 = 0 }

-- Lean statement of the proof problem:
theorem conditional_probability_problem : 
  probability_space.cond_prob choiceSpace eventA eventB = 1/4 := 
sorry

end conditional_probability_problem_l602_602691


namespace relationship_among_m_n_k_l602_602385

theorem relationship_among_m_n_k :
  (¬ ∃ x : ℝ, |2 * x - 3| + m = 0) → 
  (∃! x: ℝ, |3 * x - 4| + n = 0) → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |4 * x₁ - 5| + k = 0 ∧ |4 * x₂ - 5| + k = 0) →
  (m > n ∧ n > k) :=
by
  intros h1 h2 h3
  -- Proof part will be added here
  sorry

end relationship_among_m_n_k_l602_602385


namespace average_speed_l602_602105

-- Define the conditions given in the problem
def distance_first_hour : ℕ := 50 -- distance traveled in the first hour
def distance_second_hour : ℕ := 60 -- distance traveled in the second hour
def total_distance : ℕ := distance_first_hour + distance_second_hour -- total distance traveled

-- Define the total time
def total_time : ℕ := 2 -- total time in hours

-- The problem statement: proving the average speed
theorem average_speed : total_distance / total_time = 55 := by
  unfold total_distance total_time
  sorry

end average_speed_l602_602105


namespace volume_of_water_in_cylindrical_tank_l602_602195

-- Definitions based on conditions
def radius : ℝ := 5
def height : ℝ := 10
def depth : ℝ := 3

-- Defining the expected volume
def expected_volume : ℝ := 150 * real.pi - 180 * real.sqrt 3

-- Statement that requires proof
theorem volume_of_water_in_cylindrical_tank : 
  volume_of_water radius height depth = expected_volume :=
sorry

end volume_of_water_in_cylindrical_tank_l602_602195


namespace simplify_fraction_l602_602477

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l602_602477


namespace mean_score_classes_is_82_l602_602047

theorem mean_score_classes_is_82
  (F S : ℕ)
  (f s : ℕ)
  (hF : F = 90)
  (hS : S = 75)
  (hf_ratio : f * 6 = s * 5)
  (hf_total : f + s = 66) :
  ((F * f + S * s) / (f + s) : ℚ) = 82 :=
by
  sorry

end mean_score_classes_is_82_l602_602047


namespace math_city_intersections_l602_602851

theorem math_city_intersections :
  ∀ (n : ℕ), n = 12 → (∑ k in Finset.range (n - 1), k) = 66 :=
by
  intros n hn
  rw hn
  have h : ∑ k in Finset.range (12 - 1), k = (11 * 12) / 2 := by sorry
  rw h
  norm_num

end math_city_intersections_l602_602851


namespace instantaneous_velocity_at_t4_is_6_l602_602084

open Real

-- Define the displacement function s
def s (t : ℝ) : ℝ := 4 - 2 * t + t^2

-- Define the derivative of s
def ds_dt (t : ℝ) : ℝ := deriv s t

-- The theorem to prove that the derivative of s at t = 4 is equal to 6
theorem instantaneous_velocity_at_t4_is_6 : ds_dt 4 = 6 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end instantaneous_velocity_at_t4_is_6_l602_602084


namespace vector_addition_and_scalar_multiplication_l602_602356

-- Specify the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

-- Define the theorem we want to prove
theorem vector_addition_and_scalar_multiplication :
  a + 2 • b = (-3, 4) :=
sorry

end vector_addition_and_scalar_multiplication_l602_602356


namespace measure_of_angle_B_minimum_dot_product_l602_602791

variables {A B C : ℝ} {a b c : ℝ}
variable (m : ℝ × ℝ)
variable (n : ℝ × ℝ)

-- Given conditions for the triangle
def triangle_condition (a b c : ℝ) := 
  a^2 + c^2 = b^2 + a * c

-- Calculate the measure of angle B
theorem measure_of_angle_B (h : triangle_condition a b c) : B = π / 3 := by
  sorry

-- Given vectors m and n
def vector_m : ℝ × ℝ := (-3, -1)
def vector_n (A : ℝ) : ℝ × ℝ := (Real.sin A, Real.cos (2 * A))

-- Compute the minimum value of the dot product
theorem minimum_dot_product (A : ℝ) (hA : 0 < A ∧ A < 2 * π / 3) :
  ∃ sin_A, 
  Real.sin A = sin_A ∧ vector_m.1 * sin_A + vector_m.2 * (Real.cos (2 * A)) = -17 / 8 := by
  use 3 / 4
  sorry

end measure_of_angle_B_minimum_dot_product_l602_602791


namespace solution_set_of_inequality_l602_602717

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_2 : f 2 = 1 / 2
axiom f_prime_lt_exp : ∀ x : ℝ, deriv f x < Real.exp x

theorem solution_set_of_inequality :
  {x : ℝ | f x < Real.exp x - 1 / 2} = {x : ℝ | 0 < x} :=
by
  sorry

end solution_set_of_inequality_l602_602717


namespace friend_l602_602414

def hives := 5
def honey_per_hive := 20
def jar_capacity := 0.5
def james_jars := 100

theorem friend's_jars_ratio :
  let total_honey := hives * honey_per_hive in
  let total_jars := total_honey / jar_capacity in
  let friends_jars := total_jars - james_jars in
  (friends_jars / total_jars) = 1 / 2 :=
by
  sorry

end friend_l602_602414


namespace find_p_for_ellipse_focus_l602_602885

theorem find_p_for_ellipse_focus (p : ℝ) :
  (∀ F : ℝ × ℝ, F = (real.sqrt 3, 0) →
    ∀ (A B : ℝ × ℝ),
      (A.1 ^ 2 / 4 + A.2 ^ 2 = 1) →
      (B.1 ^ 2 / 4 + B.2 ^ 2 = 1) →
      ((A.1 - p) * (B.1 - p) = -(A.2 * B.2)) →
      (angle (real.sqrt 3, 0) A (p, 0) = angle (real.sqrt 3, 0) B (p, 0))) →
  p = 2 + real.sqrt 3 :=
begin
  sorry
end

end find_p_for_ellipse_focus_l602_602885


namespace not_all_on_same_branch_find_coordinates_of_QR_l602_602647

-- Definitions for problem 1
def is_on_hyperbola (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1

def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop := 
  let d1 := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
  let d2 := (Q.1 - R.1)^2 + (Q.2 - R.2)^2
  let d3 := (P.1 - R.1)^2 + (P.2 - R.2)^2
  d1 = d2 ∧ d2 = d3

-- Problem 1: Prove P, Q, R cannot all lie on the same branch C1 of the hyperbola
theorem not_all_on_same_branch (P Q R : ℝ × ℝ) :
  is_on_hyperbola P ∧ is_on_hyperbola Q ∧ is_on_hyperbola R →
  is_equilateral_triangle P Q R →
  ¬(P.1 > 0 ∧ Q.1 > 0 ∧ R.1 > 0) ∧ ¬(P.1 < 0 ∧ Q.1 < 0 ∧ R.1 < 0) :=
sorry

-- Definitions for problem 2
def point_on_C2 (P : ℝ × ℝ) : Prop := P = (-1, -1)

-- Problem 2: Find coordinates of Q and R given P(-1, -1) on C2
theorem find_coordinates_of_QR (P Q R : ℝ × ℝ) :
  point_on_C2 P →
  is_on_hyperbola Q ∧ is_on_hyperbola R →
  is_equilateral_triangle P Q R →
  Q = (2 - real.sqrt 3, 2 + real.sqrt 3) ∧ R = (2 + real.sqrt 3, 2 - real.sqrt 3) :=
sorry

end not_all_on_same_branch_find_coordinates_of_QR_l602_602647


namespace parallel_alarm_reliability_l602_602518

theorem parallel_alarm_reliability (r : ℝ) (h : r = 0.90) : (1 - (1 - r) * (1 - r)) = 0.99 :=
by
  rw h
  calc
    (1 - (1 - 0.9) * (1 - 0.9)) = 1 - 0.1 * 0.1 : by norm_num
                             ... = 1 - 0.01      : by norm_num
                             ... = 0.99          : by norm_num

end parallel_alarm_reliability_l602_602518


namespace base_conversion_321_base5_to_base7_l602_602258

theorem base_conversion_321_base5_to_base7 : 
  ∃ n : ℕ, (n = 3 * 5^2 + 2 * 5^1 + 1 * 5^0) ∧ (nat.to_digits 7 n = [1, 5, 2]) := 
by
  sorry

end base_conversion_321_base5_to_base7_l602_602258


namespace square_perimeter_l602_602599

noncomputable def side_length_of_square_with_area (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def perimeter_of_square_with_side (side : ℝ) : ℝ :=
  4 * side

theorem square_perimeter {area : ℝ} (h_area : area = 625) :
  perimeter_of_square_with_side (side_length_of_square_with_area area) = 100 :=
by
  have h_side_length : side_length_of_square_with_area area = 25 := by
    rw [side_length_of_square_with_area, real.sqrt, h_area]
    norm_num
  rw [perimeter_of_square_with_side, h_side_length]
  norm_num
  sorry

end square_perimeter_l602_602599


namespace m_plus_n_value_l602_602064

noncomputable def triangle_conditions : Prop :=
  ∃ (A B C D : Point) (AB AC AD BC BD CD : ℝ),
  AB = 8 ∧ CD = 4 ∧
  (is_angle_bisector A B C D) ∧
  ∀ AC, AC ∈ set.Ioo 4 16
  ∧ ∃ m n : ℝ, set.Ioo m n = set.Ioo 4 16

theorem m_plus_n_value :
  triangle_conditions → 
  ∃ (m n : ℝ), (m + n = 20) :=
by 
  intro h
  cases h with A h
  cases h with B h
  cases h with C h
  cases h with D h
  cases h with AB h
  cases h with AC h
  cases h with AD h
  cases h with BC h
  cases h with BD h
  cases h with CD h
  cases h with h_AB h
  cases h with h_CD h
  cases h with h_angle_bisector h
  cases h with h_in_interval h
  cases h with m h
  cases h with n h_interval_eq
  existsi m
  existsi n
  rw h_interval_eq
  exact 8 + 12
  sorry

end m_plus_n_value_l602_602064


namespace sqrt_ab_lt_avg_l602_602697

theorem sqrt_ab_lt_avg (a b : ℝ) (h1 : a > b) (h2 : b > 0) : sqrt (a * b) < (a + b) / 2 := 
sorry

end sqrt_ab_lt_avg_l602_602697


namespace abs_neg_2000_l602_602179

theorem abs_neg_2000 : abs (-2000) = 2000 := by
  sorry

end abs_neg_2000_l602_602179


namespace total_amount_earned_l602_602216

-- Conditions
def avg_price_pair_rackets : ℝ := 9.8
def num_pairs_sold : ℕ := 60

-- Proof statement
theorem total_amount_earned :
  avg_price_pair_rackets * num_pairs_sold = 588 := by
    sorry

end total_amount_earned_l602_602216


namespace error_percentage_90_44_l602_602157

variable (S : ℝ)
variable (h : S > 0)
def percentageErrorInArea := 100 * ((1.38 * S)^2 - S^2) / S^2

theorem error_percentage_90_44 (S : ℝ) (h : S > 0) : percentageErrorInArea S = 90.44 := by
  sorry

end error_percentage_90_44_l602_602157


namespace certain_time_in_seconds_l602_602573

theorem certain_time_in_seconds
  (ratio : ℕ) (minutes : ℕ) (time_in_minutes : ℕ) (seconds_in_a_minute : ℕ)
  (h_ratio : ratio = 8)
  (h_minutes : minutes = 4)
  (h_time : time_in_minutes = minutes)
  (h_conversion : seconds_in_a_minute = 60) :
  time_in_minutes * seconds_in_a_minute = 240 :=
by
  sorry

end certain_time_in_seconds_l602_602573


namespace theodore_total_monthly_earnings_l602_602526

-- Define the conditions
def stone_statues_per_month := 10
def wooden_statues_per_month := 20
def cost_per_stone_statue := 20
def cost_per_wooden_statue := 5
def tax_rate := 0.10

-- Calculate earnings from stone and wooden statues
def earnings_from_stone_statues := stone_statues_per_month * cost_per_stone_statue
def earnings_from_wooden_statues := wooden_statues_per_month * cost_per_wooden_statue

-- Total earnings before taxes
def total_earnings := earnings_from_stone_statues + earnings_from_wooden_statues

-- Taxes paid
def taxes_paid := total_earnings * tax_rate

-- Total earnings after taxes
def total_earnings_after_taxes := total_earnings - taxes_paid

-- Theorem stating the total earnings after taxes
theorem theodore_total_monthly_earnings : total_earnings_after_taxes = 270 := sorry

end theodore_total_monthly_earnings_l602_602526


namespace cost_per_bag_l602_602954

theorem cost_per_bag
  (friends : ℕ)
  (payment_per_friend : ℕ)
  (total_bags : ℕ)
  (total_cost : ℕ)
  (h1 : friends = 3)
  (h2 : payment_per_friend = 5)
  (h3 : total_bags = 5)
  (h4 : total_cost = friends * payment_per_friend) :
  total_cost / total_bags = 3 :=
by {
  sorry
}

end cost_per_bag_l602_602954


namespace cube_probability_l602_602581

theorem cube_probability :
  let total_cubes := 5 * 5 * 5,
      three_faces := 4,
      no_faces := 3 * 3 * 3,
      total_pairs := (total_cubes * (total_cubes - 1)) / 2,
      favorable_pairs := three_faces * no_faces in
  (favorable_pairs : ℚ) / total_pairs = 9 / 646 :=
sorry

end cube_probability_l602_602581


namespace intersection_complement_correct_l602_602043

open Set

-- Declare the sets as Lean constants
def U : Set ℕ := {x | x > 0 ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

-- Lean statement for the proof problem
theorem intersection_complement_correct : S ∩ (U \ T) = {1, 2, 4} :=
by {sory}

end intersection_complement_correct_l602_602043


namespace f_odd_and_period_pi_l602_602509

def f (x : ℝ) : ℝ := 2 * sin x * cos x

theorem f_odd_and_period_pi : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + π) = f x) :=
by
  -- We need to show that f is odd and has period π
  sorry

end f_odd_and_period_pi_l602_602509


namespace total_seating_arrangements_l602_602304

theorem total_seating_arrangements : 
  ∀ (n k : ℕ), n = 5 → k = 3 → 
  let comb := Nat.choose n k in
  let perm1 := Nat.factorial k in
  let perm2 := Nat.factorial (n - k) in
  2 * comb * perm1 * perm2 = 240 :=
by
  intros n k hn hk comb perm1 perm2
  have h1 : comb = Nat.choose n k := by rfl
  have h2 : perm1 = Nat.factorial k := by rfl
  have h3 : perm2 = Nat.factorial (n - k) := by rfl
  rw [hn, hk, Nat.choose, Nat.factorial]
  norm_num
  sorry

end total_seating_arrangements_l602_602304


namespace solution_set_to_coeff_properties_l602_602720

theorem solution_set_to_coeff_properties 
  (a b c : ℝ) 
  (h : ∀ x, (2 < x ∧ x < 3) → ax^2 + bx + c > 0) 
  : 
  (a < 0) 
  ∧ (b * c < 0) 
  ∧ (b + c = a) :=
sorry

end solution_set_to_coeff_properties_l602_602720


namespace smallest_6_digit_binary_palindrome_exists_l602_602654

-- Define what it means to be a palindrome in a given base
def is_palindrome {α : Type} [Inhabited α] [PartialOrder α] (f : ℕ → α) (n : ℕ) : Prop :=
  ∀ i, i < n → f i = f (n - i - 1)

-- Define the 6-digit palindromes in base 2
def is_6_digit_binary_palindrome (n : ℕ) : Prop :=
  n < 64 ∧ is_palindrome (λ i, (bitvec.bit i (bitvec.of_nat 6 n)).to_bool) 6

-- Define the 4-digit palindromes in another base
def is_4_digit_palindrome_in_base (n base : ℕ) : Prop :=
  n < base^4 ∧ ∃ f, (λ n, nat.digits base n) n = (λ i, nat.digits base n) (4 - i - 1) ∧ is_palindrome (λ i, (nat.digits base n).get i) 4

-- Statement to be proved
theorem smallest_6_digit_binary_palindrome_exists : ∃ n : ℕ, is_6_digit_binary_palindrome n ∧ ∃ base : ℕ, is_4_digit_palindrome_in_base n base ∧ n = 45 :=
sorry

end smallest_6_digit_binary_palindrome_exists_l602_602654


namespace integering_matrix_preserves_sums_l602_602686

-- Statements and conditions
theorem integering_matrix_preserves_sums {m n : ℕ} (p : Fin m → Fin n → ℝ) :
  let a (i : Fin m) := ∑ j, p i j,
      b (j : Fin n) := ∑ i, p i j in
  ∃ p', a' b',
    (a' : Fin m → ℤ) = (λ i, ⌊a i⌋) ∧
    (b' : Fin n → ℤ) = (λ j, ⌊b j⌋) ∧
    (p' : Fin m → Fin n → ℤ) = (λ i j, ⌊p i j⌋) ∧
    (∀ i, a' i = ∑ j, p' i j) ∧
    (∀ j, b' j = ∑ i, p' i j) :=
begin
  sorry
end

end integering_matrix_preserves_sums_l602_602686


namespace two_digit_number_satisfies_conditions_l602_602213

theorem two_digit_number_satisfies_conditions :
  ∃ N : ℕ, (N > 0) ∧ (N < 100) ∧ (N % 2 = 1) ∧ (N % 13 = 0) ∧ (∃ a b : ℕ, N = 10 * a + b ∧ (a * b) = (k : ℕ) * k) ∧ (N = 91) :=
by
  sorry

end two_digit_number_satisfies_conditions_l602_602213


namespace find_p_l602_602688

noncomputable def b (n : ℕ) (h : n ≥ 4) := ((n + 2) ^ 2 : ℚ) / (n * (n^3 - 1))

theorem find_p : 
  (∏ n in Finset.range 96 \ Finset.range 3, b (n + 4) (by norm_num)) = (1457 / Nat.factorial 98) :=
by
  sorry

end find_p_l602_602688


namespace pencils_per_student_l602_602112

-- Define the number of pens
def numberOfPens : ℕ := 1001

-- Define the number of pencils
def numberOfPencils : ℕ := 910

-- Define the maximum number of students
def maxNumberOfStudents : ℕ := 91

-- Using the given conditions, prove that each student gets 10 pencils
theorem pencils_per_student :
  (numberOfPencils / maxNumberOfStudents) = 10 :=
by sorry

end pencils_per_student_l602_602112


namespace smallest_positive_value_l602_602772

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℝ), k = 2 ∧ k = (↑(a - b) / ↑(a + b) + ↑(a + b) / ↑(a - b)) :=
sorry

end smallest_positive_value_l602_602772


namespace son_age_next_year_l602_602455

-- Definitions based on the given conditions
def my_current_age : ℕ := 35
def son_current_age : ℕ := my_current_age / 5

-- Theorem statement to prove the answer
theorem son_age_next_year : son_current_age + 1 = 8 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end son_age_next_year_l602_602455


namespace number_solution_l602_602460

theorem number_solution (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
by
  -- The proof is omitted according to the instructions
  sorry

end number_solution_l602_602460


namespace can_divide_into_any_number_of_similar_triangles_l602_602622

theorem can_divide_into_any_number_of_similar_triangles 
  (ABC : Triangle) 
  (h1 : ∃ T1 T2 T3 : Triangle, Similar T1 ABC ∧ Similar T2 ABC ∧ Similar T3 ABC ∧ (ABC is_divided_into T1 T2 T3)) :
  ∀ n : ℕ, ∃ triangles : Fin n → Triangle, (∀ i : Fin n, Similar (triangles i) ABC) ∧ (ABC is_divided_into (triangles)) :=
by
  sorry

end can_divide_into_any_number_of_similar_triangles_l602_602622


namespace lcm_9_12_15_l602_602981

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l602_602981


namespace range_of_a_l602_602321

variable (a : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≥ 0)

def proposition_q : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + (2 - a) = 0

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : a ≤ -2 ∨ a = 1 :=
  sorry

end range_of_a_l602_602321


namespace determine_constants_l602_602656

theorem determine_constants
  (C D : ℝ)
  (h1 : 3 * C + D = 7)
  (h2 : 4 * C - 2 * D = -15) :
  C = -0.1 ∧ D = 7.3 :=
by
  sorry

end determine_constants_l602_602656


namespace smallest_same_terminal_1000_l602_602939

def has_same_terminal_side (theta phi : ℝ) : Prop :=
  ∃ n : ℤ, theta = phi + n * 360

theorem smallest_same_terminal_1000 : ∀ θ : ℝ,
  θ ≥ 0 → θ < 360 → has_same_terminal_side θ 1000 → θ = 280 :=
by
  sorry

end smallest_same_terminal_1000_l602_602939


namespace line_passes_quadrants_l602_602781

theorem line_passes_quadrants (a b c : ℝ) (h : ∀ x y : ℝ, (x * y > 0 → ax + by + c ≠ 0) ∧ (x * y = 0 → ax + by + c ≠ 0) ∧ (x * y < 0 → ax + by + c ≠ 0)) :
  (a * b < 0) ∧ (b * c < 0) :=
sorry

end line_passes_quadrants_l602_602781


namespace problem_part1_problem_part2_l602_602726

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + (Real.pi / 3))

theorem problem_part1 :
  (f (Real.pi / 12) = 3) ∧ (f (7 * Real.pi / 12) = -3) :=
begin
  split;
  -- Fill in the details of the proof here
  sorry,
  sorry,
end

theorem problem_part2 (m : ℝ) :
  (Exists (fun h : ℝ → ℝ, (∀ x ∈ Icc (-Real.pi / 3) (Real.pi / 6), h x = 2 * f x + 1 - m) ∧
  (0 = h (-Real.pi / 3) ∨ 0 = h (Real.pi / 6)))) →
  (1 + 3 * Real.sqrt 3 ≤ m ∧ m < 7) :=
begin
  intros h_zeros,
  -- Fill in the details of the proof here
  sorry,
end

end problem_part1_problem_part2_l602_602726


namespace parabola_standard_equation_l602_602521

theorem parabola_standard_equation (x y : ℝ) : 
  (3 * x - 4 * y - 12 = 0) →
  (y = 0 → x = 4 ∨ y = -3 → x = 0) →
  (y^2 = 16 * x ∨ x^2 = -12 * y) :=
by
  intros h_line h_intersect
  sorry

end parabola_standard_equation_l602_602521


namespace riley_outside_fraction_l602_602011

theorem riley_outside_fraction
  (awake_jonsey : ℚ := 2 / 3)
  (jonsey_outside_fraction : ℚ := 1 / 2)
  (awake_riley : ℚ := 3 / 4)
  (total_inside_time : ℚ := 10)
  (hours_per_day : ℕ := 24) :
  let jonsey_inside_time := 1 / 3 * hours_per_day
  let riley_inside_time := (1 - (8 / 9)) * (3 / 4) * hours_per_day
  jonsey_inside_time + riley_inside_time = total_inside_time :=
by
  sorry

end riley_outside_fraction_l602_602011


namespace minimum_additional_squares_to_symmetry_l602_602819

-- Define the type for coordinates in the grid
structure Coord where
  x : Nat
  y : Nat

-- Define the conditions
def initial_shaded_squares : List Coord := [
  ⟨2, 4⟩, ⟨3, 2⟩, ⟨5, 1⟩, ⟨1, 4⟩
]

def grid_size : Coord := ⟨6, 5⟩

def vertical_line_of_symmetry : Nat := 3 -- between columns 3 and 4
def horizontal_line_of_symmetry : Nat := 2 -- between rows 2 and 3

-- Define reflection across lines of symmetry
def reflect_vertical (c : Coord) : Coord :=
  ⟨2 * vertical_line_of_symmetry - c.x, c.y⟩

def reflect_horizontal (c : Coord) : Coord :=
  ⟨c.x, 2 * horizontal_line_of_symmetry - c.y⟩

def reflect_both (c : Coord) : Coord :=
  reflect_vertical (reflect_horizontal c)

-- Define the theorem
theorem minimum_additional_squares_to_symmetry :
  ∃ (additional_squares : Nat), additional_squares = 5 := 
sorry

end minimum_additional_squares_to_symmetry_l602_602819


namespace letter_at_position_2533_l602_602160

theorem letter_at_position_2533 :
  (repeats := "ABCCCDDDD".data) →
  (pattern_length := repeats.length) →
  (pos := 2533 % pattern_length) →
  (letter := repeats.nth pos) =
  some 'C' :=
by
  sorry

end letter_at_position_2533_l602_602160


namespace greatest_five_digit_multiple_of_6_l602_602539

-- Define the digits set
def digits := {4, 5, 7, 8, 9}

-- Define the five-digit number using the digits
def number := 97548

-- Define conditions
def isFiveDigits (n : ℕ) : Prop := n / 10000 > 0 ∧ n / 100000 = 0
def isMultipleOf6 (n : ℕ) : Prop := n % 6 = 0
def usesAllDigitsOnce (n : ℕ) : Prop :=
  let d := [n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  ∧ d.toSet = digits
  ∧ d.nodup

-- Prove that 97548 is the greatest possible five-digit multiple of 6 using each of 4, 5, 7, 8, and 9 exactly once.
theorem greatest_five_digit_multiple_of_6 : 
  isFiveDigits number ∧ 
  isMultipleOf6 number ∧ 
  usesAllDigitsOnce number ∧ 
  ∀ n, isFiveDigits n ∧ isMultipleOf6 n ∧ usesAllDigitsOnce n → n ≤ number :=
sorry

end greatest_five_digit_multiple_of_6_l602_602539


namespace methanol_oxidation_product_l602_602453

-- Definition of methanol and methanal as types
def Methanol : Type := { c : String // c = "CH3OH" }
def Methanal : Type := { c : String // c = "HCHO" }

-- Definition that Methanol is a primary alcohol
def IsPrimaryAlcohol (alcohol : Type) : Prop := 
  -- This would be a property describing what it means to be a primary alcohol
  sorry

-- Defining the oxidation reaction where a primary alcohol forms an aldehyde
def OxidationProduct (alcohol : Type) [IsPrimaryAlcohol alcohol] : Type := Methanal

-- Now, the statement we need to prove
theorem methanol_oxidation_product : 
  ∀ (m : Methanol), OxidationProduct Methanol = Methanal := 
by 
  sorry

end methanol_oxidation_product_l602_602453


namespace Bob_walking_rate_l602_602052

theorem Bob_walking_rate
  (distance_XY : ℕ)
  (yolanda_rate : ℕ → ℕ)
  (bob_start_time : ℕ)
  (bob_distance_walked : ℕ)
  (bob_meeting_time : ℕ → ℕ)
  (bob_rate : ℕ → ℕ) :
  distance_XY = 17 →
  yolanda_rate 1 = 3 →
  bob_start_time = 1 →
  bob_distance_walked = 8 →
  (bob_meeting_time t = t - 1) →
  bob_rate bob_meeting_time = 3 :=
by
-- conditions and definitions
let yolanda_total_distance := (yolanda_rate bob_meeting_time + 3)
let yolanda_time := yolanda_total_distance / 3
let bob_time := yolanda_time - 1 
let bob_rate_calc := bob_distance_walked / bob_time

-- final proof step is skipped with sorry
have h: bob_rate bob_meeting_time = bob_rate_calc := sorry
a_exact h -- Return the finding

end Bob_walking_rate_l602_602052


namespace simplify_evaluate_expression_l602_602490

theorem simplify_evaluate_expression (a b : ℚ) (h1 : a = -2) (h2 : b = 1/5) :
    2 * a * b^2 - (6 * a^3 * b + 2 * (a * b^2 - (1/2) * a^3 * b)) = 8 := 
by
  sorry

end simplify_evaluate_expression_l602_602490


namespace interesting_numbers_below_1000_l602_602196

-- Define a proper divisor
def is_proper_divisor (n d : ℕ) : Prop :=
  d ≠ 1 ∧ d ≠ n ∧ n % d = 0

-- Define if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

-- Define if a number is prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define an interesting number
def is_interesting (n : ℕ) : Prop :=
  ∃ p q, is_prime p ∧ is_perfect_square q ∧ is_proper_divisor n p ∧ is_proper_divisor n (q * q) ∧ is_perfect_square (p + q * q)

-- Count interesting numbers less than or equal to 1000
def count_interesting_numbers (limit : ℕ) : ℕ :=
  Nat.card (SetOf (fun n => n ≤ limit ∧ is_interesting n))

theorem interesting_numbers_below_1000 : count_interesting_numbers 1000 = 70 :=
  sorry

end interesting_numbers_below_1000_l602_602196


namespace lcm_9_12_15_l602_602983

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l602_602983


namespace flagpoleSolution_l602_602658

noncomputable def flagpoleProblem : ℚ :=
  let field : ℝ × ℝ := (100, 100)
  let center : ℝ × ℝ := (50, 50)
  let distance_flag (x y : ℝ) : ℝ := real.sqrt ((x - 50)^2 + (y - 50)^2)
  let distance_fence (x y : ℝ) : ℝ := min (min x (100 - x)) (min y (100 - y))
  let condition (x y : ℝ) : Prop := distance_fence x y < distance_flag x y
  let prob : ℚ := (3 - real.sqrt 2) / 2
  in 8

theorem flagpoleSolution : flagpoleProblem = 8 := 
by
  sorry

end flagpoleSolution_l602_602658


namespace percentage_employed_females_is_16_l602_602793

/- 
  In Town X, the population is divided into three age groups: 18-34, 35-54, and 55+.
  For each age group, the percentage of the employed population is 64%, and the percentage of employed males is 48%.
  We need to prove that the percentage of employed females in each age group is 16%.
-/

theorem percentage_employed_females_is_16
  (percentage_employed_population : ℝ)
  (percentage_employed_males : ℝ)
  (h1 : percentage_employed_population = 0.64)
  (h2 : percentage_employed_males = 0.48) :
  percentage_employed_population - percentage_employed_males = 0.16 :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end percentage_employed_females_is_16_l602_602793


namespace square_perimeter_l602_602590

theorem square_perimeter (area : ℝ) (h : area = 625) :
  ∃ p : ℝ, p = 4 * real.sqrt area ∧ p = 100 :=
by
  sorry

end square_perimeter_l602_602590


namespace susan_probability_exactly_three_blue_marbles_l602_602497

open ProbabilityTheory

noncomputable def probability_blue_marbles (n_blue n_red : ℕ) (total_trials drawn_blue : ℕ) : ℚ :=
  let total_marbles := n_blue + n_red
  let prob_blue := (n_blue : ℚ) / total_marbles
  let prob_red := (n_red : ℚ) / total_marbles
  let n_comb := Nat.choose total_trials drawn_blue
  (n_comb : ℚ) * (prob_blue ^ drawn_blue) * (prob_red ^ (total_trials - drawn_blue))

theorem susan_probability_exactly_three_blue_marbles :
  probability_blue_marbles 8 7 7 3 = 35 * (1225621 / 171140625) :=
by
  sorry

end susan_probability_exactly_three_blue_marbles_l602_602497


namespace area_of_region_enclosed_by_graph_l602_602675

noncomputable def area_of_enclosed_region : ℝ :=
  let x1 := 41.67
  let x2 := 62.5
  let y1 := 8.33
  let y2 := -8.33
  0.5 * (x2 - x1) * (y1 - y2)

theorem area_of_region_enclosed_by_graph :
  area_of_enclosed_region = 173.28 :=
sorry

end area_of_region_enclosed_by_graph_l602_602675


namespace Ali_winning_strategy_l602_602023

def Ali_and_Mohammad_game (m n : ℕ) (a : Fin m → ℕ) : Prop :=
∃ (k l : ℕ), k > 0 ∧ l > 0 ∧ (∃ p : ℕ, Nat.Prime p ∧ m = p^k ∧ n = p^l)

theorem Ali_winning_strategy (m n : ℕ) (a : Fin m → ℕ) :
  Ali_and_Mohammad_game m n a :=
sorry

end Ali_winning_strategy_l602_602023


namespace zero_point_in_interval_l602_602946

noncomputable def f (x : ℝ) : ℝ := log x / log 2 + x - 4

theorem zero_point_in_interval :
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) ∧
  f 2 < 0 ∧
  f 3 > 0 →
  ∃ c : ℝ, c ∈ Set.Ioo 2 3 ∧ f c = 0 :=
by
  intro h
  sorry

end zero_point_in_interval_l602_602946


namespace scientific_notation_189100_l602_602244

  theorem scientific_notation_189100 :
    (∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 189100 = a * 10^n) ∧ (∃ (a : ℝ) (n : ℤ), a = 1.891 ∧ n = 5) :=
  by {
    sorry
  }
  
end scientific_notation_189100_l602_602244


namespace gracie_height_is_56_l602_602741

theorem gracie_height_is_56 
  (gracie_height grayson_height griffin_height : ℕ)
  (h1 : gracie_height < grayson_height)
  (h2 : grayson_height = griffin_height + 2)
  (h3 : griffin_height = 61)
  (h4 : gracie_height = 56) :
  gracie_height = 56 :=
by {
  exact h4,
}

end gracie_height_is_56_l602_602741


namespace inequality_solution_l602_602301

theorem inequality_solution (x : ℝ) :
  2^(x^2 + 2*x - 4) ≤ 1/2 ↔ -3 ≤ x ∧ x ≤ 1 :=
by sorry

end inequality_solution_l602_602301


namespace isosceles_triangle_exists_l602_602024

-- Define the conditions
variable {n : Nat} (h_n : n ≥ 3)
variable {vertices : Finset (Fin (4 * n + 1))} (h_colored : vertices.card = 2 * n)

-- Main statement: There exist three coloured vertices that form an isosceles triangle
theorem isosceles_triangle_exists (h_n : n ≥ 3) (h_colored : vertices.card = 2 * n) :
  ∃ (a b c : Fin (4 * n + 1)), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∈ vertices ∧ b ∈ vertices ∧ c ∈ vertices ∧
  ((dist a b = dist b c) ∨ (dist b c = dist c a) ∨ (dist c a = dist a b)) :=
sorry

end isosceles_triangle_exists_l602_602024


namespace derivative_of_f_l602_602386

-- Definitions based on conditions
def f (x : ℝ) : ℝ := 3^x + sin (2 * x)

-- Statement of the problem to be proved
theorem derivative_of_f : 
  ∀ (x : ℝ), deriv f x = 3^x * Real.log 3 + 2 * cos (2 * x) := 
by
  sorry

end derivative_of_f_l602_602386


namespace triangle_ABC_area_l602_602307

variable A : ℝ × ℝ := (0, 0)
variable B : ℝ × ℝ := (2, 0.5)
variable C : ℝ × ℝ := (2, 2)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_ABC_area : triangle_area A B C = 2.5 :=
  sorry

end triangle_ABC_area_l602_602307


namespace value_of_expression_l602_602141

theorem value_of_expression (x : ℝ) (hx : x = -2) : (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_l602_602141


namespace coin_flip_prob_l602_602068

theorem coin_flip_prob : 
  let outcomes := 2^5 in
  let success_cases := 8 in
  (success_cases / outcomes) = 1 / 4 :=
by
  sorry

end coin_flip_prob_l602_602068


namespace convex_k_gons_count_l602_602252

noncomputable def number_of_convex_k_gons (n k : ℕ) : ℕ :=
  if h : n ≥ 2 * k then
    n * Nat.factorial (n - k - 1) / (k * Nat.factorial k * Nat.factorial (n - 2 * k))
  else
    0

theorem convex_k_gons_count (n k : ℕ) (h : n ≥ 2 * k) :
  number_of_convex_k_gons n k = n * Nat.factorial (n - k - 1) / (k * Nat.factorial k * Nat.factorial (n - 2 * k)) :=
by
  sorry

end convex_k_gons_count_l602_602252


namespace ages_of_residents_l602_602579

theorem ages_of_residents (a b c : ℕ)
  (h1 : a * b * c = 1296)
  (h2 : a + b + c = 91)
  (h3 : ∀ x y z : ℕ, x * y * z = 1296 → x + y + z = 91 → (x < 80 ∧ y < 80 ∧ z < 80) → (x = 1 ∧ y = 18 ∧ z = 72)) :
  (a = 1 ∧ b = 18 ∧ c = 72 ∨ a = 1 ∧ b = 72 ∧ c = 18 ∨ a = 18 ∧ b = 1 ∧ c = 72 ∨ a = 18 ∧ b = 72 ∧ c = 1 ∨ a = 72 ∧ b = 1 ∧ c = 18 ∨ a = 72 ∧ b = 18 ∧ c = 1) :=
by
  sorry

end ages_of_residents_l602_602579


namespace distinct_odd_lists_count_l602_602744

theorem distinct_odd_lists_count : 
  ∃ (l : List (List ℕ)), 
  (∀ (a b c d : ℕ), a < b → b < c → c < d → a + b + c + d = 24 → 
  a ∈ l.head ∧ b ∈ l.head ∧ c ∈ l.head ∧ d ∈ l.head ∧ 
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ 
  c % 2 = 1 ∧ d % 2 = 1) ∧
  l.length = 5 :=
by
  sorry

end distinct_odd_lists_count_l602_602744


namespace smallest_positive_period_of_f_is_pi_monotonically_increasing_interval_of_f_range_of_f_when_x_in_0_to_pi_over_2_l602_602350

noncomputable def f (x : ℝ) := 2 * sin x * sin (x + π / 6)

theorem smallest_positive_period_of_f_is_pi : 
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = π  := sorry

theorem monotonically_increasing_interval_of_f (k : ℤ) : 
  ∃ a b, ∀ x ∈ Icc a b, ∀ y ∈ Icc a b, x ≤ y → f x ≤ f y :=
begin
  use [(-π / 12 + k * π), (5 * π / 12 + k * π)],
  sorry
end

theorem range_of_f_when_x_in_0_to_pi_over_2 :
  ∃ a b, ∀ y, y ∈ (set.range (λ x : Icc 0 (π/2), f x)) ↔ y ∈ Icc a b :=
begin
  use [0, (1 + sqrt 3 / 2)],
  sorry
end

end smallest_positive_period_of_f_is_pi_monotonically_increasing_interval_of_f_range_of_f_when_x_in_0_to_pi_over_2_l602_602350


namespace lcm_9_12_15_l602_602999

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l602_602999


namespace problem_a_2008_l602_602821

def a : ℕ → ℤ
| 0       := 0
| 1       := -1
| 2       := 2
| (n + 3) := a (n + 2) + a (n + 1)

theorem problem_a_2008 : a 2008 = 1 :=
by 
-- Proof goes here
sorry

end problem_a_2008_l602_602821


namespace function_defined_for_all_reals_except_3_l602_602100

def function_defined (x : ℝ) : Prop :=
  1 / (x - 3) ≠ y

theorem function_defined_for_all_reals_except_3 (x : ℝ) : 
  function_defined x ↔ x ≠ 3 :=
by
  sorry

end function_defined_for_all_reals_except_3_l602_602100


namespace triangle_properties_l602_602936

-- Define the sides of the triangle
def side1 : ℕ := 8
def side2 : ℕ := 15
def hypotenuse : ℕ := 17

-- Using the Pythagorean theorem to assert it is a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Calculate the area of the right triangle
def triangle_area (a b : ℕ) : ℕ :=
  (a * b) / 2

-- Calculate the perimeter of the triangle
def triangle_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_properties :
  let a := side1
  let b := side2
  let c := hypotenuse
  is_right_triangle a b c →
  triangle_area a b = 60 ∧ triangle_perimeter a b c = 40 := by
  intros h
  sorry

end triangle_properties_l602_602936


namespace tank_capacities_l602_602108

theorem tank_capacities (x y z : ℕ) 
  (h1 : x + y + z = 1620)
  (h2 : z = x + y / 5) 
  (h3 : z = y + x / 3) :
  x = 540 ∧ y = 450 ∧ z = 630 := 
by 
  sorry

end tank_capacities_l602_602108


namespace solution_eqn_l602_602491

noncomputable def problem_statement (x : ℝ) : Prop :=
  1/2 * |cos (2 * x) - 1/2| = cos (3 * x) ^ 2 + cos (x) * cos (3 * x)

theorem solution_eqn (x : ℝ) (k : ℤ) : 
  (problem_statement x) → 
  (∃ k : ℤ, x = (π / 6) + (k * π / 2)) ∨ (∃ k : ℤ, x = -(π / 6) + (k * π / 2)) :=
by
  sorry

end solution_eqn_l602_602491


namespace find_other_number_l602_602502

noncomputable def HCF : ℕ := 14
noncomputable def LCM : ℕ := 396
noncomputable def one_number : ℕ := 154
noncomputable def product_of_numbers : ℕ := HCF * LCM

theorem find_other_number (other_number : ℕ) :
  HCF * LCM = one_number * other_number → other_number = 36 :=
by
  sorry

end find_other_number_l602_602502


namespace min_perimeter_of_triangle_l602_602399

theorem min_perimeter_of_triangle (x : ℕ) (h1 : 52 + 76 > x) (h2 : 52 + x > 76) (h3 : 76 + x > 52) : 
  52 + 76 + x = 153 :=
begin
  -- We assume and introduce the hypothesis that x is an integer, greater than 24 and less than 128.
  let h4 : 24 < x := sorry,
  let h5 : x < 128 := sorry,

  -- Applying the hypothesis to find the minimum integer value of x.
  have min_x : x = 25 := sorry,

  -- Calculating the minimum perimeter.
  calc
    52 + 76 + x = 153 : by
    { rw min_x },
end

end min_perimeter_of_triangle_l602_602399


namespace train_crossing_time_l602_602198

/-- The time it takes for a train to cross a platform given the speed of the train in km/hr,
    the length of the train in meters, and the length of the platform in meters. -/
theorem train_crossing_time (speed_kmh : ℕ) (train_length : ℕ) (platform_length : ℕ) :
  speed_kmh = 72 ∧ train_length = 240 ∧ platform_length = 280 → 
  (train_length + platform_length) / (speed_kmh * 5 / 18) = 26 :=
by
  -- We define the values given in the problem
  assume h : speed_kmh = 72 ∧ train_length = 240 ∧ platform_length = 280
  change (speed_kmh * 5 / 18) = (72 * 5 / 18) with 20 -- Converting speed to m/s
  sorry

end train_crossing_time_l602_602198


namespace sum_of_digits_18_to_21_l602_602522

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_18_to_21 : 
  (sum_digits 18 + sum_digits 19 + sum_digits 20 + sum_digits 21) = 24 := 
by 
  sorry

end sum_of_digits_18_to_21_l602_602522


namespace translation_eq_l602_602087

def f (x : ℝ) : ℝ := (Real.sin x)^(4 : ℝ) + (Real.cos x)^(4 : ℝ)

def g (x : ℝ) : ℝ := f (x + Real.pi / 8)

theorem translation_eq :
  g = λ x, (3 / 4 : ℝ) - (1 / 4) * Real.sin (4 * x) :=
by 
  sorry

end translation_eq_l602_602087


namespace square_perimeter_l602_602593

theorem square_perimeter (area : ℝ) (h : area = 625) :
  ∃ p : ℝ, p = 4 * real.sqrt area ∧ p = 100 :=
by
  sorry

end square_perimeter_l602_602593


namespace relationship_m_gt_n_l602_602324

variable (a : ℝ) (m n : ℝ)
variable (f : ℝ → ℝ)

-- Given conditions
-- Condition 1: $a= \frac { \sqrt {2}+1}{2}$
def condition1 : a = (Real.sqrt 2 + 1) / 2 := by sorry

-- Condition 2: $f(x)=\log _{a}x$
def condition2 : f = λ x, Real.log x / Real.log a := by sorry

-- Condition 3: $f(m) > f(n)$
def condition3 : f m > f n := by sorry

-- Question: Prove $m > n$
theorem relationship_m_gt_n (hc1 : condition1) (hc2 : condition2) (hc3 : condition3) : m > n := by
  sorry

end relationship_m_gt_n_l602_602324


namespace exist_lambdas_divisible_by_3_l602_602034

theorem exist_lambdas_divisible_by_3 
  (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h1 : a1 > 0) (h2 : a2 > 0) (h3 : a3 > 0) 
  (h4 : b1 > 0) (h5 : b2 > 0) (h6 : b3 > 0) :
  ∃ λ1 λ2 λ3 ∈ {0, 1, 2}, 
    (λ1 * a1 + λ2 * a2 + λ3 * a3) % 3 = 0 ∧ 
    (λ1 * b1 + λ2 * b2 + λ3 * b3) % 3 = 0 :=
sorry

end exist_lambdas_divisible_by_3_l602_602034


namespace taxes_are_135_l602_602790

def gross_pay : ℕ := 450
def net_pay : ℕ := 315
def taxes_paid (G N: ℕ) : ℕ := G - N

theorem taxes_are_135 : taxes_paid gross_pay net_pay = 135 := by
  sorry

end taxes_are_135_l602_602790


namespace inequality1_summation_inequality_l602_602181

-- Problem 1
theorem inequality1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  1 / (1 + x) ≥ 1 / (1 + y) - (1 / (1 + y)^2) * (x - y) := 
sorry

-- Problem 2
theorem summation_inequality (n : ℕ) : 
  ∑ k in Finset.range (n + 1), (Nat.choose n k) * (3^k / (3^k + 1)) ≥ (3^n * 2^n) / (3^n + 2^n) := 
sorry

end inequality1_summation_inequality_l602_602181


namespace determine_tomorrow_l602_602878

-- Defining the days of the week
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open DayOfWeek

-- Defining a function to add a certain number of days to a given day
def addDays (start_day : DayOfWeek) (n : Nat) : DayOfWeek :=
  match start_day, n % 7 with
  | Monday, 0 => Monday
  | Monday, 1 => Tuesday
  | Monday, 2 => Wednesday
  | Monday, 3 => Thursday
  | Monday, 4 => Friday
  | Monday, 5 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 0 => Tuesday
  | Tuesday, 1 => Wednesday
  | Tuesday, 2 => Thursday
  | Tuesday, 3 => Friday
  | Tuesday, 4 => Saturday
  | Tuesday, 5 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 0 => Wednesday
  | Wednesday, 1 => Thursday
  | Wednesday, 2 => Friday
  | Wednesday, 3 => Saturday
  | Wednesday, 4 => Sunday
  | Wednesday, 5 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 0 => Thursday
  | Thursday, 1 => Friday
  | Thursday, 2 => Saturday
  | Thursday, 3 => Sunday
  | Thursday, 4 => Monday
  | Thursday, 5 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 0 => Friday
  | Friday, 1 => Saturday
  | Friday, 2 => Sunday
  | Friday, 3 => Monday
  | Friday, 4 => Tuesday
  | Friday, 5 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 0 => Saturday
  | Saturday, 1 => Sunday
  | Saturday, 2 => Monday
  | Saturday, 3 => Tuesday
  | Saturday, 4 => Wednesday
  | Saturday, 5 => Thursday
  | Saturday, 6 => Friday
  | Sunday, 0 => Sunday
  | Sunday, 1 => Monday
  | Sunday, 2 => Tuesday
  | Sunday, 3 => Wednesday
  | Sunday, 4 => Thursday
  | Sunday, 5 => Friday
  | Sunday, 6 => Saturday

-- Conditions
axiom condition : Monday = addDays x 5

-- Find the day of the week tomorrow
theorem determine_tomorrow (x : DayOfWeek) : addDays (addDays x 2) 1 = Saturday := sorry

end determine_tomorrow_l602_602878


namespace initial_number_of_men_l602_602240

theorem initial_number_of_men
  (road_length : ℕ)
  (total_days : ℕ)
  (completed_length : ℕ)
  (completed_days : ℕ)
  (extra_men : ℕ)
  (remaining_length : ℕ)
  (remaining_days : ℕ) :
  road_length = 10 →
  total_days = 300 →
  completed_length = 2 →
  completed_days = 100 →
  extra_men = 30 →
  remaining_length = 10 - completed_length →
  remaining_days = total_days - completed_days →
  let M := (remaining_length * total_days + remaining_days) / total_days in
  let required_men := M - 30 in
  required_men = 30 :=
by
  sorry

end initial_number_of_men_l602_602240


namespace johns_original_earnings_l602_602010

variable (x : ℝ)

-- Conditions
def postRaiseEarnings := 60
def percentageIncrease := 0.50

-- Statement to prove
theorem johns_original_earnings :
  (x + percentageIncrease * x = postRaiseEarnings) -> x = 40 := by
  sorry

end johns_original_earnings_l602_602010


namespace stuffed_animals_total_l602_602551

-- Define the initial number of stuffed animals, the gifts, and the multiplier
def initial_stuffed_animals : ℕ := 10
def mom_gift : ℕ := 2
def dad_multiplier : ℕ := 3

-- The total number of stuffed animals
def total_stuffed_animals : ℕ :=
  let after_moms_gift := initial_stuffed_animals + mom_gift in
  let dads_gift := after_moms_gift * dad_multiplier in
  after_moms_gift + dads_gift

-- Statement to be proven
theorem stuffed_animals_total : total_stuffed_animals = 48 :=
  sorry

end stuffed_animals_total_l602_602551


namespace find_probability_of_rain_l602_602669

def probability_of_rain (f : ℝ) : ℝ := 1 / 9

theorem find_probability_of_rain : 
  (∀ p_rain : ℝ, 
    (p_rain + ((p_rain / (p_rain+1)) - ((p_rain^2)/(p_rain+1)))) = 0.2) → 
  probability_of_rain 0.2 = 1 / 9 :=
by
  sorry

end find_probability_of_rain_l602_602669


namespace least_common_multiple_9_12_15_l602_602971

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l602_602971


namespace stuffed_animals_total_l602_602552

-- Define the initial number of stuffed animals, the gifts, and the multiplier
def initial_stuffed_animals : ℕ := 10
def mom_gift : ℕ := 2
def dad_multiplier : ℕ := 3

-- The total number of stuffed animals
def total_stuffed_animals : ℕ :=
  let after_moms_gift := initial_stuffed_animals + mom_gift in
  let dads_gift := after_moms_gift * dad_multiplier in
  after_moms_gift + dads_gift

-- Statement to be proven
theorem stuffed_animals_total : total_stuffed_animals = 48 :=
  sorry

end stuffed_animals_total_l602_602552


namespace find_m_l602_602085

theorem find_m (
  a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_ecc : sqrt 3 = b / a)
  (h_a : a = 1)
  (h_line_intersects_hyperbola : ∀ (x y m: ℝ), (x - y + m = 0) → x^2 - y^2 / 2 = 1 → x = m ∧ y = 2 * m)
  (h_midpoint_on_circle : ∀ (m : ℝ), (m, 2 * m) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 5}) :
  m = 1 ∨ m = -1 :=
by
  sorry

end find_m_l602_602085


namespace sin_cos_product_l602_602754

-- Define the problem's main claim
theorem sin_cos_product (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 :=
by
  have h1 : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := Real.sin_square_add_cos_square x
  sorry

end sin_cos_product_l602_602754


namespace tomorrow_is_Saturday_l602_602856
noncomputable theory

def day_of_week := ℕ
def Monday : day_of_week := 0
def Tuesday : day_of_week := 1
def Wednesday : day_of_week := 2
def Thursday : day_of_week := 3
def Friday : day_of_week := 4
def Saturday : day_of_week := 5
def Sunday : day_of_week := 6

def days_after (d : day_of_week) (n : ℕ) : day_of_week := (d + n) % 7

-- The condition: Monday is five days after the day before yesterday.
def day_before_yesterday := Wednesday
def today := days_after day_before_yesterday 2
def tomorrow := days_after today 1

theorem tomorrow_is_Saturday (h: days_after day_before_yesterday 5 = Monday) : tomorrow = Saturday := 
by {
  sorry
}

end tomorrow_is_Saturday_l602_602856


namespace count_valid_integer_n_l602_602689

theorem count_valid_integer_n :
  {n : ℤ | ∃ k : ℤ, 7200 * (3 ^ n) * (5 ^ (-n)) = k}.to_finset.card = 3 :=
begin
  sorry
end

end count_valid_integer_n_l602_602689


namespace line_intersects_x_axis_at_l602_602200

theorem line_intersects_x_axis_at {x y : ℤ} (h₁ : (7, 3) : ℤ × ℤ) (h₂ : (3, 7) : ℤ × ℤ) :
  ∃ (p : ℤ × ℤ), p.1 = 10 ∧ p.2 = 0 :=
by
  sorry

end line_intersects_x_axis_at_l602_602200


namespace tom_dimes_count_l602_602121

def originalDimes := 15
def dimesFromDad := 33
def dimesSpent := 11

theorem tom_dimes_count : originalDimes + dimesFromDad - dimesSpent = 37 := by
  sorry

end tom_dimes_count_l602_602121


namespace lcm_9_12_15_l602_602995

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l602_602995


namespace cos_2alpha_value_l602_602373

theorem cos_2alpha_value (α : ℝ) (h1 : 2 * cos (2 * α) = sin (α - π / 4))
  (h2 : π / 2 < α ∧ α < π) :
  cos (2 * α) = sqrt 15 / 8 :=
by
  sorry

end cos_2alpha_value_l602_602373


namespace tomorrow_is_saturday_l602_602860

def day := ℕ   -- Represent days as natural numbers for simplicity
def Monday := 0  -- Let's denote Monday as day 0 (Monday)
def one_week := 7  -- One week consists of 7 days

noncomputable def day_of_week (n : day) : day :=
  n % one_week  -- Calculate the day of the week based on modulo 7

theorem tomorrow_is_saturday
  (x : day)  -- the day before yesterday
  (hx : day_of_week (x + 5) = day_of_week Monday)  -- Monday is 5 days after the day before yesterday
  (today : day)  -- today
  (hy : day_of_week today = day_of_week (x + 2))  -- Today is 2 days after the day before yesterday
  : day_of_week (today + 1) = day_of_week 5 :=   -- Tomorrow will be Saturday (since Saturday is day 5)
by sorry

end tomorrow_is_saturday_l602_602860


namespace square_perimeter_l602_602601

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end square_perimeter_l602_602601


namespace total_visitors_three_days_l602_602624

def V_Rachel := 92
def V_prev_day := 419
def V_day_before_prev := 103

theorem total_visitors_three_days : V_Rachel + V_prev_day + V_day_before_prev = 614 := 
by sorry

end total_visitors_three_days_l602_602624


namespace solve_quadratic_inequality_l602_602682

open Set

theorem solve_quadratic_inequality (x : ℝ) :
  ((1 / 2 - x) * (x - 1 / 3) > 0) ↔ (1 / 3 < x ∧ x < 1 / 2) :=
by sorry

end solve_quadratic_inequality_l602_602682


namespace lcm_of_9_12_15_is_180_l602_602988

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l602_602988


namespace tangent_line_value_sum_l602_602512

theorem tangent_line_value_sum
  (f : ℝ → ℝ) 
  (hf : differentiable ℝ f) 
  (tangent_eq : ∀ x, f x = -x + 8 ↔ x = 5) : 
  f 5 + f' (5) = 2 :=
sorry

end tangent_line_value_sum_l602_602512


namespace point_on_line_l602_602159

theorem point_on_line (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 4 = 2 * (n + k) + 5) : k = 2 := by
  sorry

end point_on_line_l602_602159


namespace cube_root_of_expression_l602_602334

theorem cube_root_of_expression (a b : ℕ)
  (h1 : sqrt (2 * a - 1) = 3 ∨ sqrt (2 * a - 1) = -3)
  (h2 : sqrt (3 * a + b) = 4) :
  real.cbrt (5 * a + 2 * b) = 3 := sorry

end cube_root_of_expression_l602_602334


namespace largest_circle_in_black_squares_l602_602699

theorem largest_circle_in_black_squares:
  ∃ (r : ℝ), r = sqrt 10 / 2 :=
sorry

end largest_circle_in_black_squares_l602_602699


namespace combined_cost_price_l602_602964

-- Conditions
def cost_price_A (P Q : ℝ) : ℝ := (P + Q) / 2
def cost_price_B (P Q : ℝ) : ℝ := (P + Q) / 2

theorem combined_cost_price 
  (P_A : ℝ) (Q_A : ℝ) (P_B : ℝ) (Q_B : ℝ)
  (price_A : cost_price_A P_A Q_A = 90)
  (price_B : cost_price_B P_B Q_B = 150) :
  (cost_price_A P_A Q_A) + (cost_price_B P_B Q_B) = 240 :=
  by
    rw [price_A, price_B]
    norm_num
    sorry

end combined_cost_price_l602_602964


namespace circular_seating_count_l602_602185

theorem circular_seating_count :
  let D := 5 -- Number of Democrats
  let R := 5 -- Number of Republicans
  let total_politicians := D + R -- Total number of politicians
  let linear_arrangements := Nat.factorial total_politicians -- Total linear arrangements
  let unique_circular_arrangements := linear_arrangements / total_politicians -- Adjusting for circular rotations
  unique_circular_arrangements = 362880 :=
by
  sorry

end circular_seating_count_l602_602185


namespace odd_multiple_of_9_implies_multiple_of_3_l602_602569

-- Define an odd number that is a multiple of 9
def odd_multiple_of_nine (m : ℤ) : Prop := 9 * m % 2 = 1

-- Define multiples of 3 and 9
def multiple_of_three (n : ℤ) : Prop := ∃ k : ℤ, n = 3 * k
def multiple_of_nine (n : ℤ) : Prop := ∃ k : ℤ, n = 9 * k

-- The main statement
theorem odd_multiple_of_9_implies_multiple_of_3 (n : ℤ) 
  (h1 : ∀ n, multiple_of_nine n → multiple_of_three n)
  (h2 : odd_multiple_of_nine n ∧ multiple_of_nine n) : 
  multiple_of_three n :=
by sorry

end odd_multiple_of_9_implies_multiple_of_3_l602_602569


namespace correct_area_flat_surface_l602_602188

noncomputable def area_of_flat_surface (r h : ℝ) (theta : ℝ) : ℝ :=
  let sector_area := (theta / 360) * π * r^2
  let triangle_area := (1 / 2) * r^2 * Real.sin (theta * π / 180)
  sector_area - triangle_area

theorem correct_area_flat_surface :
  area_of_flat_surface 8 10 150 = (1456 / 9) * π :=
by
  sorry

end correct_area_flat_surface_l602_602188


namespace number_of_digits_of_50_8_8_3_11_2_10_4_l602_602560

theorem number_of_digits_of_50_8_8_3_11_2_10_4 : (number_of_digits (50 ^ 8 * 8 ^ 3 * 11 ^ 2 * 10 ^ 4)) = 21 := by
  sorry

end number_of_digits_of_50_8_8_3_11_2_10_4_l602_602560


namespace segment_length_parametric_l602_602734

theorem segment_length_parametric (x0 y0 a b t1 t2 : ℝ) :
  let A := (x0 + a * t1, y0 + b * t1)
  let B := (x0 + a * t2, y0 + b * t2)
  dist A B = sqrt (a^2 + b^2) * abs (t1 - t2) :=
by
  sorry

end segment_length_parametric_l602_602734


namespace simplify_trig_identity_l602_602483

theorem simplify_trig_identity (x y : ℝ) : 
  sin (x + y) * cos y - cos (x + y) * sin y = sin x := 
by
  sorry

end simplify_trig_identity_l602_602483


namespace find_special_number_l602_602207

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def is_multiple_of_13 (n : ℕ) : Prop := n % 13 = 0
def digits_product_is_square (n : ℕ) : Prop :=
  let digits := (Nat.digits 10 n) in
  let product := List.prod digits in
  ∃ m : ℕ, m * m = product

theorem find_special_number : ∃ N : ℕ,
  0 < N ∧ -- N is positive
  is_two_digit N ∧ -- N is a two-digit number
  is_odd N ∧ -- N is odd
  is_multiple_of_13 N ∧ -- N is a multiple of 13
  digits_product_is_square N := -- The product of its digits is a perfect square
begin
  -- Proof omitted
  sorry
end

end find_special_number_l602_602207


namespace complex_expression_equals_2i_l602_602913

def complex_expression : ℂ :=
  (3 + 2 * complex.I) / (2 - 3 * complex.I) - (3 - 2 * complex.I) / (2 + 3 * complex.I)

theorem complex_expression_equals_2i : complex_expression = 2 * complex.I := 
  sorry

end complex_expression_equals_2i_l602_602913


namespace smallest_x_value_l602_602750

theorem smallest_x_value : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (3 : ℚ) / 4 = y / (250 + x) ∧ x = 2 := by
  sorry

end smallest_x_value_l602_602750


namespace line_through_points_intersects_x_axis_l602_602202

theorem line_through_points_intersects_x_axis :
  let P1 := (7, 3)
  let P2 := (3, 7)
  line_intersects_x_axis_at P1 P2 = (10, 0) :=
by
  sorry

def line_intersects_x_axis_at (P1 P2 : ℤ × ℤ) : ℤ × ℤ :=
  let ⟨x1, y1⟩ := P1;
  let ⟨x2, y2⟩ := P2;
  if y1 = y2 then (y1, 0) else
  let m := (y2 - y1) / (x2 - x1);
  let c := y1 - m * x1;
  (c, 0)

end line_through_points_intersects_x_axis_l602_602202


namespace percent_pension_participation_l602_602659

-- Define the conditions provided
def total_first_shift_members : ℕ := 60
def total_second_shift_members : ℕ := 50
def total_third_shift_members : ℕ := 40

def first_shift_pension_percentage : ℚ := 20 / 100
def second_shift_pension_percentage : ℚ := 40 / 100
def third_shift_pension_percentage : ℚ := 10 / 100

-- Calculate participation in the pension program for each shift
def first_shift_pension_members := total_first_shift_members * first_shift_pension_percentage
def second_shift_pension_members := total_second_shift_members * second_shift_pension_percentage
def third_shift_pension_members := total_third_shift_members * third_shift_pension_percentage

-- Calculate total participation in the pension program and total number of workers
def total_pension_members := first_shift_pension_members + second_shift_pension_members + third_shift_pension_members
def total_workers := total_first_shift_members + total_second_shift_members + total_third_shift_members

-- Lean proof statement
theorem percent_pension_participation : (total_pension_members / total_workers * 100) = 24 := by
  sorry

end percent_pension_participation_l602_602659


namespace ab_bounds_l602_602421

noncomputable def find_ab (S : set (EuclideanSpace (fin 2))) (hS : S.finite) (h_card : S.card = 2017) : ℝ × ℝ := 
  let R := (∃ c ∈ S, ∀ p ∈ S, dist p c ≤ sorry) in -- The radius of the smallest circle containing all points in S
  let D := (∃ p₁ p₂ ∈ S, p₁ ≠ p₂ ∧ dist p₁ p₂ = sorry) in -- The longest distance between two points in S
  (real.sqrt 3, 2)

theorem ab_bounds (S : set (EuclideanSpace (fin 2))) (hS : S.finite) (h_card : S.card = 2017) :
  find_ab S hS h_card = (real.sqrt 3, 2) :=
sorry

end ab_bounds_l602_602421


namespace polynomial_roots_correct_l602_602896

theorem polynomial_roots_correct :
  ∀ x : ℂ, (x^4 + 4 * x^3 * complex.sqrt 3 + 6 * x^2 * 3 + 4 * x * 3 * complex.sqrt 3 + 9 + x^2 + 3 = 0) ↔
  (x = -3 * complex.sqrt 3 ∨ x = 0 ∨ x = (-3 * complex.sqrt 3 + complex.sqrt 3 * complex.I) / 2 ∨ x = (-3 * complex.sqrt 3 - complex.sqrt 3 * complex.I) / 2) :=
by
  sorry

end polynomial_roots_correct_l602_602896


namespace roses_remaining_correct_l602_602496

noncomputable def roses_initial : ℕ := 36
noncomputable def roses_given_away : ℕ := roses_initial / 2
noncomputable def roses_in_vase_initial : ℕ := roses_initial - roses_given_away
noncomputable def roses_wilted : ℕ := roses_in_vase_initial / 3
noncomputable def roses_remaining : ℕ := roses_in_vase_initial - roses_wilted

theorem roses_remaining_correct : roses_remaining = 12 := by
  let roses_initial := 36
  let roses_given_away := roses_initial / 2
  let roses_in_vase_initial := roses_initial - roses_given_away
  let roses_wilted := roses_in_vase_initial / 3
  let roses_remaining := roses_in_vase_initial - roses_wilted
  have h1 : roses_in_vase_initial = 18 := by
    rw [roses_initial, roses_given_away]
    norm_num
  have h2 : roses_wilted = 6 := by
    rw [h1]
    norm_num [roses_in_vase_initial]
  show roses_remaining = 12, by
    rw [h1, h2]
    norm_num [roses_wilted, roses_in_vase_initial]

end roses_remaining_correct_l602_602496


namespace pairs_intersections_l602_602311

theorem pairs_intersections (P : Finset (ℝ × ℝ)) (hP : P.card = 22) (hNoThreeCollinear : ∀ (A B C : ℝ × ℝ), A ∈ P → B ∈ P → C ∈ P → ¬Collinear ℝ {A, B, C}) :
  ∃ (pairings : Finset ((ℝ × ℝ) × (ℝ × ℝ))) (intersections : Finset (ℝ × ℝ)),
  pairings.card = 11 ∧ intersections.card ≥ 5 ∧
  ∀ (p : (ℝ × ℝ) × (ℝ × ℝ)), p ∈ pairings → p.1 ∈ P ∧ p.2 ∈ P ∧
  ∀ (I J : (ℝ × ℝ) × (ℝ × ℝ)), I ∈ pairings → J ∈ pairings → I ≠ J → SegmentsIntersect I J -> SegmentsIntersectAt I J ∈ intersections :=
by
  sorry

end pairs_intersections_l602_602311


namespace min_x_div_y_l602_602310

theorem min_x_div_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + y = 2) : ∃c: ℝ, c = 1 ∧ ∀(a: ℝ), x = a → y = 1 → a/y ≥ c :=
by
  sorry

end min_x_div_y_l602_602310


namespace agent_commission_l602_602238

-- Define the conditions
def commission_rate : ℝ := 2.5 / 100
def total_sales : ℝ := 840

-- Prove that the commission calculated is Rs. 21
theorem agent_commission :
  total_sales * commission_rate = 21 :=
by
  sorry

end agent_commission_l602_602238


namespace value_of_expression_l602_602143

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x + 4)^2 = 4 :=
by
  rw [h]
  norm_num
  sorry

end value_of_expression_l602_602143


namespace david_chemistry_marks_l602_602264

def David_marks : ℕ := 67

theorem david_chemistry_marks :
  ∃ (english math physics biology chemistry : ℝ),
  english = 74 ∧
  math = 65 ∧
  physics = 82 ∧
  biology = 90 ∧
  (english + math + physics + biology + chemistry) / 5 = 75.6 ∧
  chemistry = David_marks :=
begin
  sorry
end

end david_chemistry_marks_l602_602264


namespace sequence_solution_equiv_l602_602022

noncomputable def sequence (m : ℕ) : ℕ → ℤ
| 0     := 0
| 1     := m
| (n+2) := m^2 * sequence m (n+1) - sequence m n

theorem sequence_solution_equiv (a b m : ℤ) (h: 0 < m) :
  (a^2 + b^2) / (a * b + 1) = m^2 ↔ ∃ n : ℕ, a = sequence m n ∧ b = sequence m (n+1) :=
sorry

end sequence_solution_equiv_l602_602022


namespace cost_to_feed_turtles_l602_602902

-- Define the conditions
def ounces_per_half_pound : ℝ := 1 
def total_weight_turtles : ℝ := 30
def food_per_half_pound : ℝ := 0.5
def ounces_per_jar : ℝ := 15
def cost_per_jar : ℝ := 2

-- Define the statement to prove
theorem cost_to_feed_turtles : (total_weight_turtles / food_per_half_pound) / ounces_per_jar * cost_per_jar = 8 := by
  sorry

end cost_to_feed_turtles_l602_602902


namespace area_comparison_l602_602251

noncomputable def area_difference_decagon (s : ℝ) : ℝ := 
  let R := s / (2 * Real.sin (Real.pi / 10))
  let r := s / (2 * Real.tan (Real.pi / 10))
  Real.pi * (R^2 - r^2)

noncomputable def area_difference_nonagon (s : ℝ) : ℝ := 
  let R := s / (2 * Real.sin (Real.pi / 9))
  let r := s / (2 * Real.tan (Real.pi / 9))
  Real.pi * (R^2 - r^2)

theorem area_comparison :
  (area_difference_decagon 3 > area_difference_nonagon 3) :=
sorry

end area_comparison_l602_602251


namespace find_number_l602_602458

theorem find_number (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
sorry

end find_number_l602_602458


namespace mark_charged_more_hours_than_kate_l602_602563

theorem mark_charged_more_hours_than_kate :
  ∃ K : ℝ,
  let P := 2 * K in
  let M := 3 * P in
  K + P + M = 180 →
  M - K = 100 :=
by
  sorry

end mark_charged_more_hours_than_kate_l602_602563


namespace simplify_fraction_l602_602474

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l602_602474


namespace arc_length_polar_curve_l602_602637

noncomputable def arc_length (rho : ℝ → ℝ) (phi0 phi1 : ℝ) : ℝ :=
  ∫ φ in phi0..phi1, Real.sqrt ((rho φ)^2 + (Deriv.deriv rho φ)^2)

theorem arc_length_polar_curve :
  arc_length (λ φ, 3 * φ) 0 (4 / 3) = (10 / 3) + (3 / 2) * Real.log 3 := by
  sorry

end arc_length_polar_curve_l602_602637


namespace angle_KPL_135_deg_l602_602465

noncomputable theory

-- Define the points and the relevant conditions
variables {A B C D K M L P : Type} [inner_product_space ℝ Type] 
  -- Assuming we have a square ABCD
  (square_ABCD : ∀ (X : Type), X ∈ ({A, B, C, D} : set Type) → ∀ (Y : Type), Y ∈ ({A, B, C, D} : set Type) → 
    dist X Y = ∥outer_product_space.norm_sq X - outer_product_space.norm_sq Y∥)
  -- Point K is on side AB
  (K_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ K = (A + t • (B - A)))
  -- Point M is on side CD
  (M_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (C + t • (D - C)))
  -- Point L is on diagonal AC such that ML = KL
  (L_on_AC_eqdist : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ L = (A + t • (C - A)) ∧ dist M L = dist K L)
  -- Point P is the intersection point of the segments MK and BD
  (P_inter_MK_BD : ∃ tᵢ u : ℝ, tᵢ ∈ Icc 0 1 ∧ u ∈ Icc 0 1 ∧ P = ((1 - tᵢ) • M + tᵢ • K) ∧ P = (u • D + (1 - u) • B))

-- The theorem statement
theorem angle_KPL_135_deg 
  (h_square : square_ABCD Type)
  (h_K : K_on_AB)
  (h_M : M_on_CD)
  (h_L : L_on_AC_eqdist)
  (h_P : P_inter_MK_BD) :
  ∠ K P L = 135 :=
sorry

end angle_KPL_135_deg_l602_602465


namespace problem_part_I_problem_part_II_l602_602042

noncomputable theory
open Classical

-- Definitions for the functions f and h
def f (x : ℝ) (m : ℝ) : ℝ := x^2 - m * log x
def h (x : ℝ) (a : ℝ) : ℝ := x^2 - x + a
def g (x : ℝ) (m : ℝ) (a : ℝ) : ℝ := f x m - h x a

-- Main Theorems to be proved
theorem problem_part_I (m : ℝ) :
  (∀ x : ℝ, 1 < x → f x m ≥ h x 0) → m ≤ Real.exp 1 :=
sorry

theorem problem_part_II (a : ℝ) :
  (∃ (x1 x2 : ℝ), 1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 3 ∧ g x1 2 a = 0 ∧ g x2 2 a = 0) → 
  2 - 2 * Real.log 2 < a ∧ a ≤ 3 - 2 * Real.log 3 :=
sorry

end problem_part_I_problem_part_II_l602_602042


namespace tg_plus_ctg_l602_602670

theorem tg_plus_ctg (x : ℝ) (h : 1 / Real.cos x - 1 / Real.sin x = Real.sqrt 15) :
  Real.tan x + (1 / Real.tan x) = -3 ∨ Real.tan x + (1 / Real.tan x) = 5 :=
sorry

end tg_plus_ctg_l602_602670


namespace max_regions_divided_l602_602135

def sum_nat (n : ℕ) : ℕ := (n * (n + 1)) / 2

def R : ℕ → ℕ
| 0       := 1
| (n + 1) := R n + 2 * n

theorem max_regions_divided (n : ℕ) : R n = n * (n - 1) + 2 := by
  -- Inductive base case
  cases n with
  | zero => 
    sorry
  | succ n' =>
    induction n' with
    | zero =>
      sorry
    | succ m' =>
      sorry

end max_regions_divided_l602_602135


namespace math_problem_l602_602695

variables {x y : ℝ}

theorem math_problem (h1 : x + y = 6) (h2 : x * y = 5) :
  (2 / x + 2 / y = 12 / 5) ∧ ((x - y) ^ 2 = 16) ∧ (x ^ 2 + y ^ 2 = 26) :=
by
  sorry

end math_problem_l602_602695


namespace travel_rate_on_foot_l602_602203

theorem travel_rate_on_foot
  (total_distance : ℝ)
  (total_time : ℝ)
  (distance_on_foot : ℝ)
  (rate_on_bicycle : ℝ)
  (rate_on_foot : ℝ) :
  total_distance = 80 ∧ total_time = 7 ∧ distance_on_foot = 32 ∧ rate_on_bicycle = 16 →
  rate_on_foot = 8 := by
  sorry

end travel_rate_on_foot_l602_602203


namespace roger_reading_time_l602_602470

theorem roger_reading_time :
    ∀ (total_books books_first_week books_per_week : ℕ), 
    total_books = 70 → 
    books_first_week = 5 → 
    books_per_week = 7 → 
    (1 + (total_books - books_first_week + books_per_week - 1) / books_per_week) = 11 :=
by
  intros total_books books_first_week books_per_week h1 h2 h3
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end roger_reading_time_l602_602470


namespace shingles_needed_l602_602916

structure Dimensions where
  length : ℝ
  width : ℝ

def area (d : Dimensions) : ℝ :=
  d.length * d.width

def houseDimensions : Dimensions := { length := 20.5, width := 10 }
def porchDimensions : Dimensions := { length := 6, width := 4.5 }

def totalArea (d1 d2 : Dimensions) : ℝ :=
  area d1 + area d2

theorem shingles_needed :
  totalArea houseDimensions porchDimensions = 232 :=
by
  simp [totalArea, area, houseDimensions, porchDimensions]
  norm_num
  sorry

end shingles_needed_l602_602916


namespace percentage_decrease_is_correct_l602_602097

variable (P : ℝ)

-- Condition 1: After the first year, the price increased by 30%
def price_after_first_year : ℝ := 1.30 * P

-- Condition 2: At the end of the 2-year period, the price of the painting is 110.5% of the original price
def price_after_second_year : ℝ := 1.105 * P

-- Condition 3: Let D be the percentage decrease during the second year
def D : ℝ := 0.15

-- Goal: Prove that the percentage decrease during the second year is 15%
theorem percentage_decrease_is_correct : 
  1.30 * P - D * 1.30 * P = 1.105 * P → D = 0.15 :=
by
  sorry

end percentage_decrease_is_correct_l602_602097


namespace domain_of_f_max_value_condition_l602_602343

noncomputable def f (x : ℝ) : ℝ := sqrt (4 - x) + sqrt (x - 1)

-- Prove domain is [1, 4]
theorem domain_of_f : ∀ x, 1 ≤ x ∧ x ≤ 4 ↔ f x ≥ 0 :=
begin
  sorry
end

-- Prove necessary and sufficient condition for maximum value
theorem max_value_condition (a : ℝ) : 
  f_has_maximum_on [a, a+1) ↔ (3 / 2 < a ∧ a ≤ 3) :=
begin
  sorry
end

end domain_of_f_max_value_condition_l602_602343


namespace primes_dividing_sequence_l602_602968

def a_n (n : ℕ) : ℕ := 2 * 10^(n + 1) + 19

def is_prime (p : ℕ) := Nat.Prime p

theorem primes_dividing_sequence :
  {p : ℕ | is_prime p ∧ p ≤ 19 ∧ ∃ n ≥ 1, p ∣ a_n n} = {3, 7, 13, 17} :=
by
  sorry

end primes_dividing_sequence_l602_602968


namespace value_of_expression_l602_602142

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x + 4)^2 = 4 :=
by
  rw [h]
  norm_num
  sorry

end value_of_expression_l602_602142


namespace groups_of_four_on_plane_l602_602632

-- Define the points in the tetrahedron
inductive Point
| vertex : Point
| midpoint : Point

noncomputable def points : List Point :=
  [Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.midpoint,
   Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.vertex]

-- Condition: all 10 points are either vertices or midpoints of the edges of a tetrahedron 
def points_condition : ∀ p ∈ points, p = Point.vertex ∨ p = Point.midpoint := sorry

-- Function to count unique groups of four points lying on the same plane
noncomputable def count_groups : ℕ :=
  33  -- Given as the correct answer in the problem

-- Proof problem stating the count of groups
theorem groups_of_four_on_plane : count_groups = 33 :=
by 
  sorry -- Proof omitted

end groups_of_four_on_plane_l602_602632


namespace number_of_bedrooms_l602_602891

theorem number_of_bedrooms 
  (total_area : ℕ) 
  (bedroom_length : ℕ) 
  (bedroom_width : ℕ) 
  (num_bathrooms : ℕ) 
  (bathroom_length : ℕ) 
  (bathroom_width : ℕ) 
  (kitchen_area : ℕ) 
  (living_area : ℕ) 
  (same_area : kitchen_area = living_area)
  (total_area_eq : total_area = 1110)
  (bedroom_area_eq : bedroom_length * bedroom_width = 121)
  (num_bathrooms_eq : num_bathrooms = 2)
  (bathroom_area_eq : bathroom_length * bathroom_width = 48)
  (kitchen_area_eq : kitchen_area = 265) :
  bedroom_length * bedroom_width * 4 = total_area - (num_bathrooms * bathroom_area_eq + kitchen_area + living_area) :=
sorry

end number_of_bedrooms_l602_602891


namespace length_of_train_l602_602228

def speed_km_per_hr := 45
def time_to_cross_secs := 30
def bridge_length_meters := 265

theorem length_of_train :
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600 in
  let total_distance := speed_m_per_s * time_to_cross_secs in
  let train_length := total_distance - bridge_length_meters in
  train_length = 110 :=
by
  sorry

end length_of_train_l602_602228


namespace sin_cos_product_l602_602753

-- Define the problem's main claim
theorem sin_cos_product (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 :=
by
  have h1 : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := Real.sin_square_add_cos_square x
  sorry

end sin_cos_product_l602_602753


namespace degree_of_polynomial_l602_602249

theorem degree_of_polynomial :
  degree ((X^3 + 1)^5 * (X^4 + 1)^2) = 23 :=
sorry

end degree_of_polynomial_l602_602249


namespace remaining_days_temperature_l602_602910

theorem remaining_days_temperature (avg_temp : ℕ) (d1 d2 d3 d4 d5 : ℕ) :
  avg_temp = 60 →
  d1 = 40 →
  d2 = 40 →
  d3 = 40 →
  d4 = 80 →
  d5 = 80 →
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  total_temp - known_temp = 140 := 
by
  intros _ _ _ _ _ _
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  sorry

end remaining_days_temperature_l602_602910


namespace sin_cos_product_l602_602756

-- Define the problem's main claim
theorem sin_cos_product (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 :=
by
  have h1 : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := Real.sin_square_add_cos_square x
  sorry

end sin_cos_product_l602_602756


namespace tomorrow_is_saturday_l602_602872

noncomputable def day_before_yesterday : string := "Wednesday"
noncomputable def today : string := "Friday"
noncomputable def tomorrow : string := "Saturday"

theorem tomorrow_is_saturday (dby : string) (tod : string) (tom : string) 
  (h1 : dby = "Wednesday") (h2 : tod = "Friday") (h3 : tom = "Saturday")
  (h_cond : "Monday" = dby + 5) : 
  tom = "Saturday" := 
sorry

end tomorrow_is_saturday_l602_602872


namespace transformed_coordinates_l602_602927

-- Define initial point
def initial_point : ℝ × ℝ × ℝ := (2, 3, -1)

-- Define rotation matrix for 90 degrees rotation about z-axis
def rotation_z_90 : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ := λ p,
  let (x, y, z) := p in
  (0 * x + -1 * y, 1 * x + 0 * y, z)

-- Define reflection through the xz-plane
def reflection_xz : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ := λ p,
  let (x, y, z) := p in
  (x, -y, z)

-- Define reflection through the yz-plane
def reflection_yz : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ := λ p,
  let (x, y, z) := p in
  (-x, y, z)

-- Proof statement
theorem transformed_coordinates :
  let p1 := rotation_z_90 initial_point
  let p2 := reflection_xz p1
  let p3 := reflection_yz p2
  p3 = (3, -2, -1) :=
by
  have h1 : rotation_z_90 (2, 3, -1) = (-3, 2, -1) := by sorry
  have h2 : reflection_xz (-3, 2, -1) = (-3, -2, -1) := by sorry
  have h3 : reflection_yz (-3, -2, -1) = (3, -2, -1) := by sorry
  show let p1 := rotation_z_90 initial_point
       let p2 := reflection_xz p1
       let p3 := reflection_yz p2
       p3 = (3, -2, -1)
  from by
    rw [h1, h2, h3]
    exact rfl

end transformed_coordinates_l602_602927


namespace factory_initial_cars_l602_602197

theorem factory_initial_cars (x : ℕ) 
  (h1 : 5 * (0.5 * (x - 50)) = 375) : x = 200 :=
sorry

end factory_initial_cars_l602_602197


namespace edge_maximal_tutte_condition_l602_602742
-- This statement requires the Mathlib library to define graphs and necessary graph properties.

-- Definition of a graph, edge-maximal graph, and the Tutte condition

structure Graph (V : Type) :=
  (E : finset (V × V))

def is_edge_maximal (G : Graph V) : Prop :=
  ∀ G' : Graph V, (G ⊆ G' ∧ 1 vertex_added G G') → has_1_factor G'

def does_not_have_1_factor (G : Graph V) : Prop :=
  ¬ has_1_factor G

def soddisfa_tutte_condition (G : Graph V) : Prop :=
  ∀ S ⊆ V(G), q_num (G - S) ≤ S.card

-- Main statement for the problem: 
theorem edge_maximal_tutte_condition (G : Graph V) :
  is_edge_maximal G → does_not_have_1_factor G ↔ soddisfa_tutte_condition G := by
  sorry

end edge_maximal_tutte_condition_l602_602742


namespace num_proper_subsets_of_A_l602_602096

open Set

def A : Finset ℕ := {2, 3}

theorem num_proper_subsets_of_A : (A.powerset \ {A, ∅}).card = 3 := by
  sorry

end num_proper_subsets_of_A_l602_602096


namespace determine_tomorrow_l602_602876

-- Defining the days of the week
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open DayOfWeek

-- Defining a function to add a certain number of days to a given day
def addDays (start_day : DayOfWeek) (n : Nat) : DayOfWeek :=
  match start_day, n % 7 with
  | Monday, 0 => Monday
  | Monday, 1 => Tuesday
  | Monday, 2 => Wednesday
  | Monday, 3 => Thursday
  | Monday, 4 => Friday
  | Monday, 5 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 0 => Tuesday
  | Tuesday, 1 => Wednesday
  | Tuesday, 2 => Thursday
  | Tuesday, 3 => Friday
  | Tuesday, 4 => Saturday
  | Tuesday, 5 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 0 => Wednesday
  | Wednesday, 1 => Thursday
  | Wednesday, 2 => Friday
  | Wednesday, 3 => Saturday
  | Wednesday, 4 => Sunday
  | Wednesday, 5 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 0 => Thursday
  | Thursday, 1 => Friday
  | Thursday, 2 => Saturday
  | Thursday, 3 => Sunday
  | Thursday, 4 => Monday
  | Thursday, 5 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 0 => Friday
  | Friday, 1 => Saturday
  | Friday, 2 => Sunday
  | Friday, 3 => Monday
  | Friday, 4 => Tuesday
  | Friday, 5 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 0 => Saturday
  | Saturday, 1 => Sunday
  | Saturday, 2 => Monday
  | Saturday, 3 => Tuesday
  | Saturday, 4 => Wednesday
  | Saturday, 5 => Thursday
  | Saturday, 6 => Friday
  | Sunday, 0 => Sunday
  | Sunday, 1 => Monday
  | Sunday, 2 => Tuesday
  | Sunday, 3 => Wednesday
  | Sunday, 4 => Thursday
  | Sunday, 5 => Friday
  | Sunday, 6 => Saturday

-- Conditions
axiom condition : Monday = addDays x 5

-- Find the day of the week tomorrow
theorem determine_tomorrow (x : DayOfWeek) : addDays (addDays x 2) 1 = Saturday := sorry

end determine_tomorrow_l602_602876


namespace minimum_value_of_f_l602_602540

def f (x : ℝ) : ℝ := |x - 4| + |x + 6| + |x - 5|

theorem minimum_value_of_f :
  ∃ x : ℝ, (x = -6 ∧ f (-6) = 1) ∧ ∀ y : ℝ, f y ≥ 1 :=
by
  sorry

end minimum_value_of_f_l602_602540


namespace solve_equation1_solve_equation2_l602_602067

-- Problem for Equation (1)
theorem solve_equation1 (x : ℝ) : x * (x - 6) = 2 * (x - 8) → x = 4 := by
  sorry

-- Problem for Equation (2)
theorem solve_equation2 (x : ℝ) : (2 * x - 1)^2 + 3 * (2 * x - 1) + 2 = 0 → x = 0 ∨ x = -1 / 2 := by
  sorry

end solve_equation1_solve_equation2_l602_602067


namespace lions_win_at_least_three_of_five_l602_602903

open Nat

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n, k => binom (n - 1) (k - 1) + binom (n - 1) k

theorem lions_win_at_least_three_of_five :
  let n := 5
  let p := 0.5
  let probability (k : ℕ) : ℝ := binom n k * (p^k) * ((1-p)^(n-k))
  (probability 3 + probability 4 + probability 5) = 1/2 :=
by
  sorry

end lions_win_at_least_three_of_five_l602_602903


namespace small_triangle_perimeter_l602_602854

theorem small_triangle_perimeter (P : ℕ) (P₁ : ℕ) (P₂ : ℕ) (P₃ : ℕ)
  (h₁ : P = 11) (h₂ : P₁ = 5) (h₃ : P₂ = 7) (h₄ : P₃ = 9) :
  (P₁ + P₂ + P₃) - P = 10 :=
by
  sorry

end small_triangle_perimeter_l602_602854


namespace square_perimeter_l602_602598

noncomputable def side_length_of_square_with_area (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def perimeter_of_square_with_side (side : ℝ) : ℝ :=
  4 * side

theorem square_perimeter {area : ℝ} (h_area : area = 625) :
  perimeter_of_square_with_side (side_length_of_square_with_area area) = 100 :=
by
  have h_side_length : side_length_of_square_with_area area = 25 := by
    rw [side_length_of_square_with_area, real.sqrt, h_area]
    norm_num
  rw [perimeter_of_square_with_side, h_side_length]
  norm_num
  sorry

end square_perimeter_l602_602598


namespace ducks_in_the_lake_l602_602113

theorem ducks_in_the_lake (original_ducks new_ducks : ℕ) (h1 : original_ducks = 13) (h2 : new_ducks = 20) : original_ducks + new_ducks = 33 :=
by {
  rw [h1, h2],
  exact rfl,
  sorry
}

end ducks_in_the_lake_l602_602113


namespace simplify_fraction_l602_602475

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l602_602475


namespace find_length_AB_l602_602094

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line y = x - 1
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection length |AB|
noncomputable def length_AB (x1 x2 : ℝ) : ℝ := x1 + x2 + 2

-- Main theorem statement
theorem find_length_AB (x1 x2 : ℝ)
  (h₁ : parabola x1 (x1 - 1))
  (h₂ : parabola x2 (x2 - 1))
  (hx : x1 + x2 = 6) :
  length_AB x1 x2 = 8 := sorry

end find_length_AB_l602_602094


namespace expected_number_of_games_l602_602966
noncomputable def probability_of_A_winning (g : ℕ) : ℚ := 2 / 3
noncomputable def probability_of_B_winning (g : ℕ) : ℚ := 1 / 3
noncomputable def expected_games: ℚ := 266 / 81

theorem expected_number_of_games 
  (match_ends : ∀ g : ℕ, (∃ p1 p2 : ℕ, (p1 = g ∧ p2 = 0) ∨ (p1 = 0 ∧ p2 = g))) 
  (independent_outcomes : ∀ g1 g2 : ℕ, g1 ≠ g2 → probability_of_A_winning g1 * probability_of_A_winning g2 = (2 / 3) * (2 / 3) ∧ probability_of_B_winning g1 * probability_of_B_winning g2 = (1 / 3) * (1 / 3)) :
  (expected_games = 266 / 81) := 
sorry

end expected_number_of_games_l602_602966


namespace polar_to_rectangular_l602_602649

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 6) (h₂ : θ = Real.pi / 2) :
  (r * Real.cos θ, r * Real.sin θ) = (0, 6) :=
by
  sorry

end polar_to_rectangular_l602_602649


namespace sin_cos_value_l602_602771

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end sin_cos_value_l602_602771


namespace find_AB_l602_602804

-- Defining the given values and their relationships
def angleA : ℝ := 90
def angleC : ℝ := 35
def BC : ℝ := 12

-- Computing angle B
def angleB : ℝ := 180 - angleA - angleC

-- Defining the side AB using the tangent function and given data
def AB : ℝ := BC * Real.tan angleB

-- Our goal is to prove that AB equals approximately 17.1
theorem find_AB :
  AB = 12 * Real.tan (55 * Real.pi / 180) := sorry

end find_AB_l602_602804


namespace find_circle_eq_l602_602578

noncomputable def circle_eqn_passing_through_PACB 
  (P : ℝ × ℝ) 
  (C : ℝ × ℝ) 
  (circle1 : ℝ → ℝ → ℝ) 
  (AQ BC: Prop) 
  : Prop :=
  P = (-1, -1) ∧ C = (1, 1) ∧ 
  (∀ x y : ℝ, circle1 x y = (x - 1) ^ 2 + (y - 1) ^ 2 - 1 = 0) ∧
  AQ ∧ BC → (∀ x y : ℝ, x ^ 2 + y ^ 2 = 2)

theorem find_circle_eq 
  : circle_eqn_passing_through_PACB 
      (-1, -1) 
      (1, 1) 
      (λ x y, (x - 1) ^ 2 + (y - 1) ^ 2 - 1) 
      (λ PQ CA, true)  -- placeholder for AQ condition
      (λ PQ CB, true)  -- placeholder for BC condition 
:= sorry

end find_circle_eq_l602_602578


namespace Var_fg_leq_l602_602687

variable (f g : ℝ → ℝ)
variable [Continuous f] [Continuous g]

def mu (f : ℝ → ℝ) : ℝ :=
  ∫ x in 0..1, f x

def Var (f : ℝ → ℝ) : ℝ :=
  ∫ x in 0..1, (f x - mu f) ^ 2

def M (f : ℝ → ℝ) : ℝ :=
  ⨆ x in Icc 0 1, |f x|

theorem Var_fg_leq :
  Var (λ x, f x * g x) ≤ 2 * Var f * (M g) ^ 2 + 2 * Var g * (M f) ^ 2 :=
by
  sorry

end Var_fg_leq_l602_602687


namespace simplify_fraction_l602_602476

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l602_602476


namespace probability_of_special_full_house_is_correct_l602_602398

def total_number_of_ways_to_choose_five_cards : ℕ :=
  Nat.binom 52 5

def number_of_successful_outcomes_special_full_house : ℕ :=
  13 * (Nat.binom 4 3) * 4 * (Nat.binom 12 2)

def probability_special_full_house : ℚ :=
  number_of_successful_outcomes_special_full_house / total_number_of_ways_to_choose_five_cards

theorem probability_of_special_full_house_is_correct :
  probability_special_full_house = 7 / 1335 := by
  sorry

end probability_of_special_full_house_is_correct_l602_602398


namespace diana_shopping_for_newborns_l602_602273

-- Define the conditions
def num_toddlers : ℕ := 6
def num_teenagers : ℕ := 5 * num_toddlers
def total_children : ℕ := 40

-- Define the problem statement
theorem diana_shopping_for_newborns : (total_children - (num_toddlers + num_teenagers)) = 4 := by
  sorry

end diana_shopping_for_newborns_l602_602273


namespace hard_to_determine_position_l602_602620

-- Define seat coordinates
structure SeatNumber where
  row : Nat
  col : Nat

-- Given seat number (2, 4)
def given_seat : SeatNumber := { row := 2, col := 4 }

-- Theorem stating it's hard to determine the position based on given_seat
theorem hard_to_determine_position : ∀ s : SeatNumber, (s = given_seat) → "Hard to determine" :=
sorry

end hard_to_determine_position_l602_602620


namespace coefficient_of_monomial_l602_602080

-- Definition of the monomial
def monomial : ℕ → ℕ → ℕ → ℤ := λ a b c, a * (b + c) -- defining a monomial with coefficients and exponents

-- Given monomial
def given_monomial : ℤ := monomial 3 2 1 -- representing 3 * x^2 * y with x² = 2, y = 1

-- Theorem: Coefficient of the given monomial
theorem coefficient_of_monomial :
  given_monomial = 3 := by
  sorry

end coefficient_of_monomial_l602_602080


namespace num_triangles_with_longest_side_11_l602_602365

def is_valid_triangle (x y : ℕ) : Prop :=
  x + y > 11 ∧ x ≤ y ∧ y ≤ 11

def count_triangles (n : ℕ) : ℕ :=
  (finset.range n).sum (λ x, (finset.range n).count (λ y, is_valid_triangle x.succ y.succ))

theorem num_triangles_with_longest_side_11 : count_triangles 12 = 36 :=
  sorry

end num_triangles_with_longest_side_11_l602_602365


namespace number_of_k_lt_sum_div_alpha_l602_602703

def sequence := list ℕ

def m_k (a : sequence) (k : ℕ) : ℝ :=
  finset.sup (finset.range k) (λ l, (list.sum (list.drop (k - l + 1) a)) / l)

theorem number_of_k_lt_sum_div_alpha (a : sequence) (α : ℝ) (hα : α > 0) : 
  (finset.card (finset.filter (λ k, m_k a k > α) (finset.range (list.length a))))
  < (list.sum a) / α :=
sorry

end number_of_k_lt_sum_div_alpha_l602_602703


namespace concurrency_of_perpendiculars_l602_602839

variable {A B C O H: Point}
variable {AD BE CF YZ ZX XY P Q R A' D': Line}

-- Conditions
variable {acute_scalene_triangle : triangle ABC}
variable {circumcenter_O : is_circumcenter O ABC}
variable {altitude_AD : is_altitude AD A B C}
variable {altitude_BE : is_altitude BE B A C}
variable {altitude_CF : is_altitude CF C A B}
variable {midpoint_X : is_midpoint X AD}
variable {midpoint_Y : is_midpoint Y BE}
variable {midpoint_Z : is_midpoint Z CF}
variable {line_P : intersect AD YZ P}
variable {line_Q : intersect BE ZX Q}
variable {line_R : intersect CF XY R}
variable {line_A' : intersect YZ BC A'}
variable {line_D' : intersect QR EF D'}

theorem concurrency_of_perpendiculars :
  concurrent_perpendiculars A B C O QR RP PQ A'D' := 
sorry

end concurrency_of_perpendiculars_l602_602839


namespace square_perimeter_l602_602610

theorem square_perimeter (area : ℝ) (h : area = 625) : 
  let s := Real.sqrt area in
  (4 * s) = 100 :=
by
  let s := Real.sqrt area
  have hs : s = 25 := by sorry
  calc
    (4 * s) = 4 * 25 : by rw hs
          ... = 100   : by norm_num

end square_perimeter_l602_602610


namespace length_of_train_l602_602227

def speed_km_per_hr := 45
def time_to_cross_secs := 30
def bridge_length_meters := 265

theorem length_of_train :
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600 in
  let total_distance := speed_m_per_s * time_to_cross_secs in
  let train_length := total_distance - bridge_length_meters in
  train_length = 110 :=
by
  sorry

end length_of_train_l602_602227


namespace value_of_m_l602_602423

theorem value_of_m (a b m : ℚ) (h1 : 2 * a = m) (h2 : 5 * b = m) (h3 : a + b = 2) : m = 20 / 7 :=
by
  sorry

end value_of_m_l602_602423


namespace units_digit_of_j_squared_plus_3_power_j_l602_602428

def j : ℕ := 19^2 + 3^10

theorem units_digit_of_j_squared_plus_3_power_j :
  ((j^2 + 3^j) % 10) = 3 :=
by
  sorry

end units_digit_of_j_squared_plus_3_power_j_l602_602428


namespace leila_total_cakes_l602_602016

def cakes_monday : ℕ := 6
def cakes_friday : ℕ := 9
def cakes_saturday : ℕ := 3 * cakes_monday
def total_cakes : ℕ := cakes_monday + cakes_friday + cakes_saturday

theorem leila_total_cakes : total_cakes = 33 := by
  sorry

end leila_total_cakes_l602_602016


namespace binomial_sum_alternating_even_indices_l602_602638

open Real

theorem binomial_sum_alternating_even_indices :
  ∑ k in Finset.range(51), ((-1) ^ k) * Nat.choose 101 (2 * k) = -2 ^ 50 := by
  sorry

end binomial_sum_alternating_even_indices_l602_602638


namespace square_perimeter_l602_602609

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end square_perimeter_l602_602609


namespace pradeep_failed_marks_l602_602056

theorem pradeep_failed_marks
    (total_marks : ℕ)
    (obtained_marks : ℕ)
    (pass_percentage : ℕ)
    (pass_marks : ℕ)
    (fail_marks : ℕ)
    (total_marks_eq : total_marks = 2075)
    (obtained_marks_eq : obtained_marks = 390)
    (pass_percentage_eq : pass_percentage = 20)
    (pass_marks_eq : pass_marks = (pass_percentage * total_marks) / 100)
    (fail_marks_eq : fail_marks = pass_marks - obtained_marks) :
    fail_marks = 25 :=
by
  rw [total_marks_eq, obtained_marks_eq, pass_percentage_eq] at *
  sorry

end pradeep_failed_marks_l602_602056


namespace area_of_gray_region_is_96π_l602_602129

noncomputable def area_gray_region (d_small : ℝ) (r_ratio : ℝ) : ℝ :=
  let r_small := d_small / 2
  let r_large := r_ratio * r_small
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  area_large - area_small

theorem area_of_gray_region_is_96π :
  ∀ (d_small : ℝ) (r_ratio : ℝ), d_small = 4 → r_ratio = 5 → area_gray_region d_small r_ratio = 96 * π :=
by
  intros d_small r_ratio h1 h2
  have : d_small = 4 := h1
  have : r_ratio = 5 := h2
  sorry

end area_of_gray_region_is_96π_l602_602129


namespace sequence_term_1000_l602_602392

theorem sequence_term_1000 :
  ∃ (a : ℕ → ℤ), a 1 = 2007 ∧ a 2 = 2008 ∧ (∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) ∧ a 1000 = 2340 := 
by
  sorry

end sequence_term_1000_l602_602392


namespace line_through_points_intersects_x_axis_l602_602201

theorem line_through_points_intersects_x_axis :
  let P1 := (7, 3)
  let P2 := (3, 7)
  line_intersects_x_axis_at P1 P2 = (10, 0) :=
by
  sorry

def line_intersects_x_axis_at (P1 P2 : ℤ × ℤ) : ℤ × ℤ :=
  let ⟨x1, y1⟩ := P1;
  let ⟨x2, y2⟩ := P2;
  if y1 = y2 then (y1, 0) else
  let m := (y2 - y1) / (x2 - x1);
  let c := y1 - m * x1;
  (c, 0)

end line_through_points_intersects_x_axis_l602_602201


namespace complement_intersection_l602_602308

open Set

theorem complement_intersection {x : ℝ} :
  (x ∉ {x | -2 ≤ x ∧ x ≤ 2}) ∧ (x < 1) ↔ (x < -2) := 
by
  sorry

end complement_intersection_l602_602308


namespace min_AB_CD_value_l602_602783

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def AB_CD (AC BD CB : vector) : ℝ :=
  let AB := (CB.1 + AC.1, CB.2 + AC.2)
  let CD := (CB.1 + BD.1, CB.2 + BD.2)
  dot_product AB CD

theorem min_AB_CD_value : ∀ (AC BD : vector), AC = (1, 2) → BD = (-2, 2) → 
  ∃ CB : vector, AB_CD AC BD CB = -9 / 4 :=
by
  intros AC BD hAC hBD
  sorry

end min_AB_CD_value_l602_602783


namespace find_p_q_r_l602_602713

def f (x : ℝ) : ℝ := x^2 + 2*x + 2
def g (x p q r : ℝ) : ℝ := x^3 + 2*x^2 + 6*p*x + 4*q*x + r

noncomputable def roots_sum_f := -2
noncomputable def roots_product_f := 2

theorem find_p_q_r (p q r : ℝ) (h1 : ∀ x, f x = 0 → g x p q r = 0) :
  (p + q) * r = 0 :=
sorry

end find_p_q_r_l602_602713


namespace sum_of_inverse_terms_l602_602332

variable {a_n : ℕ → ℝ}

/-- Given a geometric sequence {a_n} with first term 1 and sum S_n of the first n terms.
    If 28S_3 = S_6, prove that the sum of the first 5 terms of the sequence {1 / a_n} is 121 / 81. -/
theorem sum_of_inverse_terms (q : ℝ)
  (h1 : a_n 0 = 1)
  (hq : ∀ n, a_n (n + 1) = q * a_n n)
  (h_sum : ∀ n, S_n n = (1 - q^(n + 1)) / (1 - q))
  (condition : 28 * S_n 2 = S_n 5) :
  (∑ i in Finset.range 5, 1 / a_n i) = 121 / 81 :=
by
  sorry

end sum_of_inverse_terms_l602_602332


namespace probability_three_divisible_by_3_l602_602234

theorem probability_three_divisible_by_3 :
  let probability_divisible_by_3 := 1 / 3 in
  let probability_not_divisible_by_3 := 2 / 3 in
  let n := 5 in
  let k := 3 in
  nat.choose n k * (probability_divisible_by_3 ^ k) * (probability_not_divisible_by_3 ^ (n - k)) = 40 / 243 :=
by sorry

end probability_three_divisible_by_3_l602_602234


namespace number_of_valid_triplets_l602_602446

theorem number_of_valid_triplets (a b c : ℕ) (h_b : b = 2023) (h_rel : a * c = 2023 * 2023) (h_order : a ≤ b ∧ b ≤ c) : 
  ∃ n, n = 7 ∧ ∃ triples : fin n → (ℕ × ℕ × ℕ), ∀ i, let (a', b', c') := triples i in (a' ≤ b' ∧ b' ≤ c' ∧ b' = 2023 ∧ a' * c' = 2023 * 2023) :=
sorry

end number_of_valid_triplets_l602_602446


namespace unit_vector_norm_equal_l602_602712

variables (a b : EuclideanSpace ℝ (Fin 2)) -- assuming 2D Euclidean space for simplicity

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ‖v‖ = 1

theorem unit_vector_norm_equal {a b : EuclideanSpace ℝ (Fin 2)}
  (ha : is_unit_vector a) (hb : is_unit_vector b) : ‖a‖ = ‖b‖ :=
by 
  sorry

end unit_vector_norm_equal_l602_602712


namespace probability_one_solves_l602_602328

theorem probability_one_solves :
  let pA := 0.8
  let pB := 0.7
  (pA * (1 - pB) + pB * (1 - pA)) = 0.38 :=
by
  sorry

end probability_one_solves_l602_602328


namespace sin_cos_identity_l602_602760

theorem sin_cos_identity (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602760


namespace student_marks_l602_602217

theorem student_marks (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) (student_marks : ℕ) : 
  (passing_percentage = 30) → (failed_by = 40) → (max_marks = 400) → 
  student_marks = (max_marks * passing_percentage / 100 - failed_by) → 
  student_marks = 80 :=
by {
  sorry
}

end student_marks_l602_602217


namespace remaining_days_temperature_l602_602908

theorem remaining_days_temperature :
  let avg_temp := 60
  let total_days := 7
  let temp_day1 := 40
  let temp_day2 := 40
  let temp_day3 := 40
  let temp_day4 := 80
  let temp_day5 := 80
  let total_temp := avg_temp * total_days
  let temp_first_five_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5
  total_temp - temp_first_five_days = 140 :=
by
  -- proof is omitted
  sorry

end remaining_days_temperature_l602_602908


namespace log_expression_evaluation_l602_602278

theorem log_expression_evaluation (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : y ≠ 1) :
  (log (x^2) / log (y^3)) * (log (y) / log (x^4)) * (log (x^5) / log (y^2)) * (log (y^2) / log (x^5)) * (log (x^4) / log (y))
  = (1 / 3) * log y x :=
by
  sorry

end log_expression_evaluation_l602_602278


namespace square_perimeter_l602_602611

theorem square_perimeter (area : ℝ) (h : area = 625) : 
  let s := Real.sqrt area in
  (4 * s) = 100 :=
by
  let s := Real.sqrt area
  have hs : s = 25 := by sorry
  calc
    (4 * s) = 4 * 25 : by rw hs
          ... = 100   : by norm_num

end square_perimeter_l602_602611


namespace line_through_point_l602_602091

theorem line_through_point (b : ℚ) :
  (∃ x y,
    (x = 3) ∧ (y = -7) ∧ (b * x + (b - 1) * y = b + 3))
  → (b = 4 / 5) :=
begin
  sorry
end
 
end line_through_point_l602_602091


namespace find_a_l602_602342

def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x else x + 1

theorem find_a (a : ℝ) (h : f a + f 2 = 0) : a = -5 :=
by
  sorry

end find_a_l602_602342


namespace FerrisWheelCostIsSix_l602_602630

structure AmusementPark where
  roller_coaster_cost : ℕ
  log_ride_cost : ℕ
  initial_tickets : ℕ
  additional_tickets_needed : ℕ

def ferris_wheel_cost (a : AmusementPark) : ℕ :=
  let total_needed := a.initial_tickets + a.additional_tickets_needed
  let total_ride_cost := a.roller_coaster_cost + a.log_ride_cost
  total_needed - total_ride_cost

theorem FerrisWheelCostIsSix (a : AmusementPark) 
  (h₁ : a.roller_coaster_cost = 5)
  (h₂ : a.log_ride_cost = 7)
  (h₃ : a.initial_tickets = 2)
  (h₄ : a.additional_tickets_needed = 16) :
  ferris_wheel_cost a = 6 :=
by
  -- proof omitted
  sorry

end FerrisWheelCostIsSix_l602_602630


namespace new_price_correct_l602_602928

-- Define the conditions
def original_price : ℝ := 6
def reduction_percentage : ℝ := 19.999999999999996 / 100
def new_price (P : ℝ) : Prop :=
  (original_price * 1) = (0.8 * P)

-- Statement to prove that the new price is Rs. 7.5
theorem new_price_correct : ∃ P, new_price P ∧ P = 7.5 :=
by
  sorry

end new_price_correct_l602_602928


namespace choose_three_points_l602_602051

-- The type representing a point in the plane
structure Point :=
(x : ℝ)
(y : ℝ)

-- Definitions of collinear and concyclic conditions
def collinear (p1 p2 p3 : Point) : Prop :=
(p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

def concyclic (p1 p2 p3 p4 : Point) : Prop :=
let d = determinant in -- This is a placeholder for the actual determinant definition
d [ [ 1, p1.x, p1.y, p1.x^2 + p1.y^2 ],
    [ 1, p2.x, p2.y, p2.x^2 + p2.y^2 ],
    [ 1, p3.x, p3.y, p3.x^2 + p3.y^2 ],
    [ 1, p4.x, p4.y, p4.x^2 + p4.y^2 ]] = 0

-- Main theorem
theorem choose_three_points (points : list Point) (n : ℕ) :
  points.length = 2 * n + 3 →
  (∀ p1 p2 p3 : Point, p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬ collinear p1 p2 p3) →
  (∀ p1 p2 p3 p4 : Point, p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → ¬ concyclic p1 p2 p3 p4) →
  ∃ (a b c : Point) (rest : list Point), 
    a ∈ points ∧ b ∈ points ∧ c ∈ points ∧ 
    (list.erase (list.erase (list.erase points a) b) c) = rest ∧
    (∃ inside outside : list Point, 
       inside.length = n ∧ 
       outside.length = n ∧ 
       is_circumcircle a b c inside outside) :=
begin
  sorry
end

end choose_three_points_l602_602051


namespace graph_connected_l602_602889

noncomputable def ceil (x : ℝ) : ℕ := sorry

variables (G : SimpleGraph (Fin n)) (n : ℕ) [Fintype (Fin n)]

-- Condition: Each vertex has a degree of at least ⌈(n-1)/2⌉
def degree_condition (G : SimpleGraph (Fin n)) (n : ℕ) : Prop :=
  ∀ v : Fin n, (G.degree v) ≥ ceil ((n - 1 : ℝ) / 2)

-- The conclusion we need to prove is that the graph is connected.
theorem graph_connected (G : SimpleGraph (Fin n)) (n : ℕ) [Fintype (Fin n)] 
  (h: degree_condition G n) : G.Connected :=
sorry

end graph_connected_l602_602889


namespace choose_team_no_twins_l602_602934

theorem choose_team_no_twins (total_players twins excluded total_ways restricted_ways : ℕ) 
  (h1 : total_players = 16)
  (h2 : twins = 2)
  (h3 : excluded = 14)
  (h4 : total_ways = nat.choose 16 5)
  (h5 : restricted_ways = nat.choose 14 3)
  (h6 : 4004 = total_ways - restricted_ways) : 
  total_ways - restricted_ways = 4004 :=
by
  rw [total_ways, restricted_ways, h4, h5]
  exact h6
  sorry

end choose_team_no_twins_l602_602934


namespace lcm_9_12_15_l602_602982

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l602_602982


namespace product_gcd_lcm_4_12_l602_602136

open Nat

theorem product_gcd_lcm_4_12 : gcd 4 12 * lcm 4 12 = 48 := by
  sorry

end product_gcd_lcm_4_12_l602_602136


namespace domain_of_f_max_value_condition_l602_602345

def f (x : ℝ) : ℝ := real.sqrt (4 - x) + real.sqrt (x - 1)

theorem domain_of_f :
  {x : ℝ | 1 ≤ x ∧ x ≤ 4} = set_of (λ x, 1 ≤ x ∧ x ≤ 4) :=
sorry

theorem max_value_condition {a : ℝ} (h : a ∈ set.Icc (3 / 2) 3) :
  f (a + 1) ≤ f a :=
sorry

end domain_of_f_max_value_condition_l602_602345


namespace sin_cos_identity_l602_602763

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602763


namespace function_order_l602_602330

noncomputable def f : ℝ → ℝ := sorry

theorem function_order (a b c : ℝ) (h1 : a = f(2017)) (h2 : b = f(2016)) (h3 : c = f(2015))
  (symmetry : ∀ x : ℝ, f(-x + 1) = f(x + 1))
  (periodicity : ∀ x : ℝ, f(x + 2) = -f(x))
  (decreasing : ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 < x2 → x2 ≤ 3 → (f(x2) - f(x1)) * (x2 - x1) < 0) :
  a > b ∧ b > c :=
by
  sorry

end function_order_l602_602330


namespace problem_conditions_l602_602721

variable (A B : ℝ)
def S (n : ℕ) : ℝ := A * n^2 + B * n
def a (n : ℕ) : ℝ := S n - S (n - 1)
def sequence_sum (n : ℕ) (T : ℕ → (ℕ → ℝ) → ℝ) : ℝ := T n (λ k, 1 / (S k + 5 * k))

theorem problem_conditions :
  (S 1 = -3) → (S 2 = -4) → 
  (∃ a, ∀ n, a n = 2 * n - 5) →
  (∀ n, S n - S 2 ≥ 0) →
  (∀ n, sequence_sum n (λ k f, ∑ i in finset.range k, f i) < 1) →
  true := sorry

end problem_conditions_l602_602721


namespace second_number_deduction_l602_602076

theorem second_number_deduction
  (x : ℝ)
  (h1 : (10 * 16 = 10 * x + (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)))
  (h2 : 2.5 + (x+1 - y) + 6.5 + 8.5 + 10.5 + 12.5 + 14.5 + 16.5 + 18.5 + 20.5 = 115)
  : y = 8 :=
by
  -- This is where the proof would go, but we'll leave it as 'sorry' for now.
  sorry

end second_number_deduction_l602_602076


namespace no_two_digit_multiples_of_3_5_7_l602_602745

theorem no_two_digit_multiples_of_3_5_7 : ∀ n : ℕ, 10 ≤ n ∧ n < 100 → ¬ (3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) := 
by
  intro n
  intro h
  intro h_div
  sorry

end no_two_digit_multiples_of_3_5_7_l602_602745


namespace proof_problem_l602_602808

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 1

-- Define the circle C with center (h, k) and radius r
def circle_eq (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Define condition of line that intersects the circle C at points A and B
def line_eq (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Condition: OA ⊥ OB
def perpendicular_cond (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem stating the proof problem
theorem proof_problem :
  (∃ (h k r : ℝ),
    circle_eq h k r 3 1 ∧
    circle_eq h k r 5 0 ∧
    circle_eq h k r 1 0 ∧
    h = 3 ∧ k = 1 ∧ r = 3) ∧
    (∃ (a : ℝ),
      (∀ (x1 y1 x2 y2 : ℝ),
        line_eq a x1 y1 ∧
        circle_eq 3 1 3 x1 y1 ∧
        line_eq a x2 y2 ∧
        circle_eq 3 1 3 x2 y2 → 
        perpendicular_cond x1 y1 x2 y2) →
      a = -1) :=
by
  sorry

end proof_problem_l602_602808


namespace lcm_9_12_15_l602_602978

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l602_602978


namespace simplify_trig_identity_l602_602482

theorem simplify_trig_identity (x y : ℝ) : 
  sin (x + y) * cos y - cos (x + y) * sin y = sin x := 
by
  sorry

end simplify_trig_identity_l602_602482


namespace derivative_of_curve_tangent_line_at_one_l602_602723

-- Definition of the curve
def curve (x : ℝ) : ℝ := x^3 + 5 * x^2 + 3 * x

-- Part 1: Prove the derivative of the curve
theorem derivative_of_curve (x : ℝ) :
  deriv curve x = 3 * x^2 + 10 * x + 3 :=
sorry

-- Part 2: Prove the equation of the tangent line at x = 1
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), a = 16 ∧ b = -1 ∧ c = -7 ∧
  ∀ (x y : ℝ), curve 1 = 9 → y - 9 = 16 * (x - 1) → a * x + b * y + c = 0 :=
sorry

end derivative_of_curve_tangent_line_at_one_l602_602723


namespace y_coord_range_of_C_l602_602320

-- Given points A(0,2), B, and C on the parabola y² = x + 4
-- such that AB ⊥ BC, we aim to prove the y-coordinate of C.

theorem y_coord_range_of_C (x0 : ℝ) (m : ℝ) :
  -- Points B and C lie on the parabola y² = x + 4
  let B_y := m^2 - 4
  let C_y := x0^2 - 4
  B ∈ set_of (λ p : ℝ × ℝ, (p.snd^2 = p.fst + 4) ∧ (p.fst = B_y)) ∧
  C ∈ set_of (λ p : ℝ × ℝ, (p.snd^2 = p.fst + 4) ∧ (p.fst = C_y)) →
  -- Points A(0,2) and B(m^2 - 4, m)
  let A := (0, 2)
  let slope_AB := (m - 2) / (m^2 - 4)
  -- Points B(m^2 - 4, m) and C(x0^2 - 4, x0)
  let slope_BC := (x0 - m) / (x0^2 - m^2)
  -- AB ⊥ BC
  slope_AB * slope_BC = -1 →
  -- Conclusion, the range of the y-coordinates of C
  x0 ≤ 2 - 2 * Real.sqrt 2 ∨ x0 ≥ 2 + 2 * Real.sqrt 2 :=
begin
  sorry
end

end y_coord_range_of_C_l602_602320


namespace sin_cos_value_l602_602769

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end sin_cos_value_l602_602769


namespace inequality_not_always_true_l602_602323

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) : ¬ (∀ c, (a - b) / c > 0) := 
sorry

end inequality_not_always_true_l602_602323


namespace minimum_value_of_m_l602_602836

variable (x : ℝ)

-- Definition of m as per the problem's condition
def m (x : ℝ) : ℝ := x + if x >= 1 then x - 1 else 1 - x

-- Statement of the theorem
theorem minimum_value_of_m : ∃ y, (∀ x, m x >= y) ∧ (∀ x, y = m x → x < 1 ∨ x = 1) := by
  use 1
  sorry

end minimum_value_of_m_l602_602836


namespace peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l602_602175

-- Define the context of the problem
def total_people := 100
def men := 50
def women := 50

-- Define Peter Ivanovich being satisfied
def satisfies_peter_ivanovich := (women / (total_people - 1)) * ((women - 1) / (total_people - 2)) 

-- Define the probability that Peter Ivanovich is satisfied
theorem peter_ivanovich_satisfied_probability :
  satisfies_peter_ivanovich = 25 / 33 := 
sorry

-- Define the expected number of satisfied men
def expected_satisfied_men := men * (25 / 33)

-- Prove the expected number of satisfied men
theorem expected_satisfied_men_value :
  expected_satisfied_men = 1250 / 33 :=
sorry

end peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l602_602175


namespace rectangular_coordinates_transformation_l602_602587

noncomputable theory

variables (ρ θ φ : ℝ)
variables (coords : ℝ × ℝ × ℝ)

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

def derived_spherical_coordinates : Prop :=
  spherical_to_rectangular ρ θ φ = (3, -3, 2 * real.sqrt 2)

def new_spherical_coordinates : ℝ × ℝ × ℝ :=
  spherical_to_rectangular ρ (θ + real.pi / 2) (-φ)

theorem rectangular_coordinates_transformation
  (h1 : derived_spherical_coordinates ρ θ φ) :
  new_spherical_coordinates ρ θ φ = (-3, -3, 2 * real.sqrt 2) :=
sorry

end rectangular_coordinates_transformation_l602_602587


namespace sine_subtraction_formula_simplify_expression_l602_602489

-- Define the sine subtraction formula as a condition
theorem sine_subtraction_formula (a b : ℝ) : 
    real.sin (a - b) = real.sin a * real.cos b - real.cos a * real.sin b := by
  sorry

-- Prove the given expression simplifies to sin(x)
theorem simplify_expression (x y : ℝ) :
    real.sin (x + y) * real.cos y - real.cos (x + y) * real.sin y = real.sin x := by
  have h : real.sin ((x + y) - y) = real.sin (x + y) * real.cos y - real.cos (x + y) * real.sin y := by
    exact sine_subtraction_formula (x + y) y
  rw [sub_self y, h]
  simp
  sorry

end sine_subtraction_formula_simplify_expression_l602_602489


namespace max_min_abs_z_plus_2_plus_5i_l602_602714

theorem max_min_abs_z_plus_2_plus_5i (z : ℂ) (hz : complex.abs (z - 2) = 1) :
  ∃ L U : ℝ, L = real.sqrt 41 - 1 ∧ U = real.sqrt 41 + 1 ∧
  ∀ w, complex.abs (z - 2) = 1 → L ≤ complex.abs (z + 2 + 5 * complex.I) ∧ complex.abs (z + 2 + 5 * complex.I) ≤ U :=
begin
  sorry
end

end max_min_abs_z_plus_2_plus_5i_l602_602714


namespace simplify_trig_identity_l602_602479

theorem simplify_trig_identity (x y : ℝ) : 
  sin (x + y) * cos y - cos (x + y) * sin y = sin x :=
by 
  sorry

end simplify_trig_identity_l602_602479


namespace floor_of_s_l602_602441

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x + 5 * Real.tan x

theorem floor_of_s :
  let s := Inf {x : ℝ | x > 0 ∧ g x = 0} in
  ⌊s⌋ = 3 :=
by
  let s := Inf {x : ℝ | x > 0 ∧ g x = 0}
  have hs : s > 0 := sorry
  have hs_pos : g s = 0 := sorry
  have s_in_range : s ∈ Icc π (5 * π / 4) := sorry
  exact sorry

end floor_of_s_l602_602441


namespace disjoint_subsets_with_equal_sum_l602_602832

variable {α : Type*} [Fintype α] [DecidableEq α]

-- Assume A is a finite set of n positive integers with the sum constraint.
variable (A : Finset ℕ)
variable (n : ℕ)
variable (hA_card : A.card = n)
variable (h_sum : ∑ a in A, a < 2^n - 1)

theorem disjoint_subsets_with_equal_sum :
  ∃ (B C : Finset ℕ), B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ (∑ b in B, b = ∑ c in C, c) :=
sorry

end disjoint_subsets_with_equal_sum_l602_602832


namespace stddev_of_data_l602_602079

variable {R : Type} [RealField R]

def average (l : List R) : R := l.sum / (l.length : R)

def variance (l : List R) : R := (l.map (λ x, (x - average l) ^ 2)).sum / (l.length : R)

def standard_deviation (l : List R) : R := Real.sqrt (variance l)

theorem stddev_of_data {x : R} (h_avg : average [1, 3, 2, 5, x] = 3) : standard_deviation [1, 3, 2, 5, x] = Real.sqrt 2 := by
  have h1 : x = 4 := by
    sorry
  rw [h1]
  have h2 : variance [1, 3, 2, 5, 4] = 2 := by
    sorry
  rw [variance]
  sorry

end stddev_of_data_l602_602079


namespace find_magnitude_l602_602696

theorem find_magnitude (i z : ℂ) (hi : i * i = -1) : 
  (i ^ 2023) * z = (1 / 2) - (1 / 2) * i → |z - i| = (Real.sqrt 2) / 2 :=
by
  intros h
  sorry

end find_magnitude_l602_602696


namespace unique_function_satisfying_conditions_l602_602267

theorem unique_function_satisfying_conditions :
  ∀ (f : ℝ → ℝ), 
    (∀ x : ℝ, f x ≥ 0) → 
    (∀ x : ℝ, f (x^2) = f x ^ 2 - 2 * x * f x) →
    (∀ x : ℝ, f (-x) = f (x - 1)) → 
    (∀ x y : ℝ, 1 < x → x < y → f x < f y) →
    (∀ x : ℝ, f x = x^2 + x + 1) :=
by
  -- formal proof would go here
  sorry

end unique_function_satisfying_conditions_l602_602267


namespace value_of_f_l602_602378

def f (a b c : ℝ) : ℝ := (c + a) / (c - b)

theorem value_of_f : f 1 (-2) (-3) = 2 := by
  -- Add proof here
  sorry

end value_of_f_l602_602378


namespace sin_cos_value_l602_602767

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end sin_cos_value_l602_602767


namespace cos_phi_sum_num_denom_l602_602795

theorem cos_phi_sum_num_denom (r : ℝ) (h1 : (3^2 = 2 * r^2 * (1 - cos φ))) (h2 : 2 * r^2 ≠ 0) :
  let φ_cos := 1 - 9 / (2 * r^2) in
  φ_cos = 7 / 8 → φ_cos.num + φ_cos.denom = 15 :=
by {
  sorry
}

end cos_phi_sum_num_denom_l602_602795


namespace ellipse_hyperbola_intersection_l602_602707

theorem ellipse_hyperbola_intersection (a1 b1 a2 b2 : ℝ) (h1 : a1 > b1) (h2 : b1 > 0) (h3 : a2 > 0) (h4 : b2 > 0)
    (h5 : ∃ P : ℝ × ℝ, P ∈ { p : ℝ × ℝ | p.1^2 / a1^2 + p.2^2 / b1^2 = 1 } ∧ P ∈ { p : ℝ × ℝ | p.1^2 / a2^2 - p.2^2 / b2^2 = 1 })
    (h6 : ∀ (F1 F2 : ℝ × ℝ), is_focus F1 a1 b1 ∧ is_focus F2 a1 b1 ∧ is_focus F1 a2 b2 ∧ is_focus F2 a2 b2 ->
          angle F1 (classical.some h5) F2 = π / 3) :
  b1 / b2 = sqrt 3 := sorry

end ellipse_hyperbola_intersection_l602_602707


namespace inverse_function_correct_l602_602134

noncomputable def f (x : ℝ) : ℝ := 3 - 7 * x

noncomputable def g (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_function_correct : ∀ x : ℝ, f (g x) = x ∧ g (f x) = x :=
by
  intro x
  sorry

end inverse_function_correct_l602_602134


namespace sum_of_roots_eq_l602_602083

theorem sum_of_roots_eq {x_1 x_2 y_1 y_2 z_1 z_2 : ℝ}
  (h1 : x_2 - x_1 = 1)
  (h2 : y_2 - y_1 = 2)
  (h3 : z_2 - z_1 = 3) :
  x_1 + y_1 + z_2 = x_2 + y_2 + z_1 :=
begin
  sorry
end

end sum_of_roots_eq_l602_602083


namespace exists_non_planar_regular_n_segment_chain_l602_602192

theorem exists_non_planar_regular_n_segment_chain (N : ℕ) (h : N > 5) : 
  ∃ chain : list (euclidean_space 3), 
    (regular_chain chain ∧ length chain = N ∧ ¬ is_planar chain) :=
sorry

-- Definition of regular_chain
def regular_chain (chain : list (euclidean_space 3)) : Prop :=
  ∀ (i j : ℕ) (h1 : i < length chain) (h2 : j < length chain), 
    (segment_length (chain.nth i) = segment_length (chain.nth j)) ∧ 
    (angle (chain.nth i) (chain.nth (i + 1)) = angle (chain.nth j) (chain.nth (j + 1)))

-- Supporting Definitions (Assuming they exist or will be defined)
def segment_length (p q : euclidean_space 3) : ℝ := 
  dist p q

def angle (p q r : euclidean_space 3) : ℝ :=
  -- Assume some appropriate definition here
  0

def is_planar (chain : list (euclidean_space 3)) : Prop :=
  -- Definition to check if the shape formed by the chain lies in a single plane.
  false

end exists_non_planar_regular_n_segment_chain_l602_602192


namespace trigonometric_functions_unit_circle_l602_602722

theorem trigonometric_functions_unit_circle (x y : ℝ) (r : ℝ) (h₁ : r = 1) (hx : x = -5 / 13) (hy : y = 12 / 13) :
  sin (atan2 y x) = 12 / 13 ∧ tan (atan2 y x) = -12 / 5 :=
by {
  sorry
}

end trigonometric_functions_unit_circle_l602_602722


namespace equal_sets_l602_602236

theorem equal_sets :
  (¬ (3^2 = 2^3)) ∧
  (¬ (-((3 * 2)^2) = -3 * 2^2)) ∧
  (¬ (-|2^3| = | -2^3 |)) ∧
  (-2^3 = (-2)^3) :=
by
  -- condition for set A
  have h1 : 3^2 ≠ 2^3 := by sorry,
  -- condition for set B
  have h2 : -((3 * 2)^2) ≠ -3 * 2^2 := by sorry,
  -- condition for set C
  have h3 : -|2^3| ≠ | -2^3 | := by sorry,
  -- condition for set D
  have h4 : -2^3 = (-2)^3 := by sorry,
  exact ⟨h1, h2, h3, h4⟩

end equal_sets_l602_602236


namespace solve_for_x_l602_602284

noncomputable def x : ℝ := 64

theorem solve_for_x (x : ℝ) (h : 9^(Real.log 8 x) = 81) : x = 64 := by
  sorry

end solve_for_x_l602_602284


namespace solve_price_per_bottle_l602_602003

noncomputable theory

def price_per_bottle : ℝ :=
  let bottles := 80
  let cans := 140
  let total_money := 15
  let price_per_can := 0.05
  (total_money - (cans * price_per_can)) / bottles

theorem solve_price_per_bottle : price_per_bottle = 0.10 := by
  sorry

end solve_price_per_bottle_l602_602003


namespace probability_real_roots_l602_602787

theorem probability_real_roots (m : ℝ) (h₀ : 0 ≤ m) (h₆ : m ≤ 6) : 
  Pr (x^2 - mx + 4 = 0 has real roots) = 1 / 3 := 
by 
sorry

end probability_real_roots_l602_602787


namespace expected_winnings_is_0_25_l602_602585

def prob_heads : ℚ := 3 / 8
def prob_tails : ℚ := 1 / 4
def prob_edge  : ℚ := 1 / 8
def prob_disappear : ℚ := 1 / 4

def winnings_heads : ℚ := 2
def winnings_tails : ℚ := 5
def winnings_edge  : ℚ := -2
def winnings_disappear : ℚ := -6

def expected_winnings : ℚ := 
  prob_heads * winnings_heads +
  prob_tails * winnings_tails +
  prob_edge  * winnings_edge +
  prob_disappear * winnings_disappear

theorem expected_winnings_is_0_25 : expected_winnings = 0.25 := by
  sorry

end expected_winnings_is_0_25_l602_602585


namespace simplify_trig_identity_l602_602478

theorem simplify_trig_identity (x y : ℝ) : 
  sin (x + y) * cos y - cos (x + y) * sin y = sin x :=
by 
  sorry

end simplify_trig_identity_l602_602478


namespace eight_square_root_pattern_l602_602812

theorem eight_square_root_pattern : 
  ∀ (n : ℕ), (8 * real.sqrt (8 / n) = real.sqrt (8 * 8 / n)) → n = 63 := 
by
  intro n

  -- Assuming the pattern x * sqrt(x / (x^2 - 1))
  have pattern : ∀ x : ℕ, x * real.sqrt (x / (x^2 - 1)) = real.sqrt (x * (x / (x^2 - 1))) :=
    by
      intro x
      sorry

  -- Check that 8 follows the pattern
  specialize pattern 8
  rw [pattern] at *

  -- Given that 8 * real.sqrt(8 / n) = real.sqrt (8 * 8 / n)
  intro h,
  
  -- Solve for n
  have h1 : 8 * 8 / n = 64 / n := by norm_num
  rw [h1] at h
  sorry

end eight_square_root_pattern_l602_602812


namespace hexagon_parallelogram_connected_l602_602422

open Classical

variable {Point : Type}

-- Define points A, B, C, D, E, and F.
variables (A B C D E F : Point)

-- A function that determines whether two segments are connected
def connected (X Y Z T : Point) : Prop :=
  ∃ O : Point, (O = O) ∧ (O = O) ∧ (O = O) ∧ (O = O)

-- Conditions: AB and CE are connected, BD and EF are connected
variable (h1 : connected A B C E) (h2 : connected B D E F)

-- Goal: Show that ACDF is a parallelogram and BC and AE are connected
theorem hexagon_parallelogram_connected :
  (∃ P Q R S : Point, A = P ∧ C = Q ∧ D = R ∧ F = S ∧ ∀ O, connected A C D F)
  ∧ connected B C A E :=
begin
  sorry
end

end hexagon_parallelogram_connected_l602_602422


namespace tomorrow_is_Saturday_l602_602858
noncomputable theory

def day_of_week := ℕ
def Monday : day_of_week := 0
def Tuesday : day_of_week := 1
def Wednesday : day_of_week := 2
def Thursday : day_of_week := 3
def Friday : day_of_week := 4
def Saturday : day_of_week := 5
def Sunday : day_of_week := 6

def days_after (d : day_of_week) (n : ℕ) : day_of_week := (d + n) % 7

-- The condition: Monday is five days after the day before yesterday.
def day_before_yesterday := Wednesday
def today := days_after day_before_yesterday 2
def tomorrow := days_after today 1

theorem tomorrow_is_Saturday (h: days_after day_before_yesterday 5 = Monday) : tomorrow = Saturday := 
by {
  sorry
}

end tomorrow_is_Saturday_l602_602858


namespace distinct_integers_l602_602270

theorem distinct_integers (S : Finset ℕ) (f : ℕ → ℕ) (h1 : ∀ n ∈ Finset.range 500, S.contains (f (n + 1)) = true) :
  f = λ n, Nat.floor (n^2 / 500) ∧ S.card = 376 :=
by
  sorry

end distinct_integers_l602_602270


namespace simplify_trig_identity_l602_602484

theorem simplify_trig_identity (x y : ℝ) : 
  sin (x + y) * cos y - cos (x + y) * sin y = sin x := 
by
  sorry

end simplify_trig_identity_l602_602484


namespace inequality_1_inequality_2_inequality_3_l602_602897

variable (x : ℝ)

theorem inequality_1 (h : 2 * x^2 - 3 * x + 1 ≥ 0) : x ≤ 1 / 2 ∨ x ≥ 1 := 
  sorry

theorem inequality_2 (h : x^2 - 2 * x - 3 < 0) : -1 < x ∧ x < 3 := 
  sorry

theorem inequality_3 (h : -3 * x^2 + 5 * x - 2 > 0) : 2 / 3 < x ∧ x < 1 := 
  sorry

end inequality_1_inequality_2_inequality_3_l602_602897


namespace length_of_AB_in_30_60_90_triangle_l602_602281

theorem length_of_AB_in_30_60_90_triangle 
  (A B C : Type) 
  [triangle A B C] 
  (angle_BAC : angle A B C = 30) 
  (length_AC : AC = 12) : AB = 8 * sqrt 3 := 
begin
  sorry
end

end length_of_AB_in_30_60_90_triangle_l602_602281


namespace loss_percentage_is_ten_l602_602231

variable (CP SP SP_new : ℝ)  -- introduce the cost price, selling price, and new selling price as variables

theorem loss_percentage_is_ten
  (h1 : CP = 2000)
  (h2 : SP_new = CP + 80)
  (h3 : SP_new = SP + 280)
  (h4 : SP = CP - (L / 100 * CP)) : L = 10 :=
by
  -- proof goes here
  sorry

end loss_percentage_is_ten_l602_602231


namespace central_projections_are_1_2_5_l602_602626

/-- Definition of whether an item forms a central projection -/
inductive forms_central_projection : ℕ → Prop
| searchlight : forms_central_projection 1
| car_light : forms_central_projection 2
| desk_lamp : forms_central_projection 5

/-- Theorem: Identifying items that form central projections -/
theorem central_projections_are_1_2_5 :
  ∀ x, x = 1 ∨ x = 2 ∨ x = 5 ↔ forms_central_projection x :=
begin
  intro x,
  split,
  {
    intro hx,
    cases hx,
    { exact forms_central_projection.searchlight },
    cases hx,
    { exact forms_central_projection.car_light },
    { exact forms_central_projection.desk_lamp },
  },
  {
    intro hx,
    cases hx,
    { left, refl },
    cases hx,
    { right, left, refl },
    { right, right, refl },
  }
end

end central_projections_are_1_2_5_l602_602626


namespace find_number_l602_602558

theorem find_number (x : ℤ) 
  (h1 : 3 * (2 * x + 9) = 51) : x = 4 := 
by 
  sorry

end find_number_l602_602558


namespace peter_satisfied_probability_expected_satisfied_men_l602_602170

variable (numMen : ℕ) (numWomen : ℕ) (totalPeople : ℕ)
variable (peterSatisfiedProb : ℚ) (expectedSatisfiedMen : ℚ)

-- Conditions
def conditions_holds : Prop :=
  numMen = 50 ∧ numWomen = 50 ∧ totalPeople = 100 ∧ peterSatisfiedProb = 25 / 33 ∧ expectedSatisfiedMen = 1250 / 33

-- Prove the probability that Peter Ivanovich is satisfied.
theorem peter_satisfied_probability : conditions_holds → peterSatisfiedProb = 25 / 33 := by
  sorry

-- Prove the expected number of satisfied men.
theorem expected_satisfied_men : conditions_holds → expectedSatisfiedMen = 1250 / 33 := by
  sorry

end peter_satisfied_probability_expected_satisfied_men_l602_602170


namespace willy_stuffed_animals_l602_602550

theorem willy_stuffed_animals :
  ∀ (initial mom dad : ℕ), 
  initial = 10 → mom = 2 → dad = 3 →
  let after_mom := initial + mom in
  let after_dad := after_mom + (dad * after_mom) in
  after_dad = 48 :=
by
  intros initial mom dad h1 h2 h3 after_mom after_dad
  sorry

end willy_stuffed_animals_l602_602550


namespace find_range_of_a_l602_602434

noncomputable def value_range_for_a : Set ℝ := {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4}

theorem find_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0)  ∧
  (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧
  (¬ (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0) → ¬ (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0))
  → a ∈ value_range_for_a :=
sorry

end find_range_of_a_l602_602434


namespace average_salary_before_manager_l602_602906

-- Definitions of the conditions
def average_salary_20_employees : ℝ := _
def manager_salary : ℝ := 12000
def average_salary_increase : ℝ := 500

-- Statement of the problem
theorem average_salary_before_manager (A : ℝ) (hA1 : average_salary_20_employees = A)
  (hA2 : 21 * (A + average_salary_increase) = 20 * A + manager_salary) :
  A = 1500 :=
begin
  sorry
end

end average_salary_before_manager_l602_602906


namespace trisha_money_left_l602_602537

theorem trisha_money_left
    (meat cost: ℕ) (chicken_cost: ℕ) (veggies_cost: ℕ) (eggs_cost: ℕ) (dog_food_cost: ℕ) 
    (initial_money: ℕ) (total_spent: ℕ) (money_left: ℕ) :
    meat_cost = 17 →
    chicken_cost = 22 →
    veggies_cost = 43 →
    eggs_cost = 5 →
    dog_food_cost = 45 →
    initial_money = 167 →
    total_spent = meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost →
    money_left = initial_money - total_spent →
    money_left = 35 :=
by
    intros
    sorry

end trisha_money_left_l602_602537


namespace peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l602_602177

-- Define the context of the problem
def total_people := 100
def men := 50
def women := 50

-- Define Peter Ivanovich being satisfied
def satisfies_peter_ivanovich := (women / (total_people - 1)) * ((women - 1) / (total_people - 2)) 

-- Define the probability that Peter Ivanovich is satisfied
theorem peter_ivanovich_satisfied_probability :
  satisfies_peter_ivanovich = 25 / 33 := 
sorry

-- Define the expected number of satisfied men
def expected_satisfied_men := men * (25 / 33)

-- Prove the expected number of satisfied men
theorem expected_satisfied_men_value :
  expected_satisfied_men = 1250 / 33 :=
sorry

end peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l602_602177


namespace sum_of_sequence_l602_602822

-- Definitions for the sequence and the condition
axiom seq (a : ℕ → ℝ) : Prop
axiom condition (a : ℕ → ℝ) : Prop := ∀ n, 1 ≤ n → a n + a (n + 1) + a (n + 2) = (real.sqrt 2) ^ n

-- Statement on the sum of the first 9 terms
noncomputable def sum_of_first_9_terms (a : ℕ → ℝ) : ℝ := 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9

theorem sum_of_sequence (a : ℕ → ℝ) (h : condition a) : 
  sum_of_first_9_terms a = 4 + 9 * real.sqrt 2 :=
sorry

end sum_of_sequence_l602_602822


namespace train_crossing_time_l602_602562

theorem train_crossing_time :
  ∀ (length speed : ℕ), 
  (length = 20 ∧ speed = 144) → 
  (length / (speed * 1000 / 3600) = 0.5) := 
begin 
  intros length speed h,
  cases h with h_length h_speed,
  rw h_length,
  rw h_speed,
  norm_num,
  sorry
end

end train_crossing_time_l602_602562


namespace product_units_digit_and_sign_l602_602127

def odd_neg_ints (n : ℤ) : Prop := n < 0 ∧ n % 2 ≠ 0
def gt_neg_2018 (n : ℤ) : Prop := n > -2018
def exclude_every_third_starting_neg2017 (n : ℤ) : Prop := (∃ (k : ℤ), n = -2017 + 3 * k)

noncomputable def product_of_odd_neg_ints_altered_seq : ℤ :=
  ∏ i in (finset.Ico (-2018) 0).filter (λ x, odd_neg_ints x ∧ gt_neg_2018 x ∧ ¬exclude_every_third_starting_neg2017 x), i

theorem product_units_digit_and_sign :
  product_of_odd_neg_ints_altered_seq > 0 ∧ (product_of_odd_neg_ints_altered_seq % 10 = 1) :=
sorry

end product_units_digit_and_sign_l602_602127


namespace two_digit_number_satisfies_conditions_l602_602211

theorem two_digit_number_satisfies_conditions :
  ∃ N : ℕ, (N > 0) ∧ (N < 100) ∧ (N % 2 = 1) ∧ (N % 13 = 0) ∧ (∃ a b : ℕ, N = 10 * a + b ∧ (a * b) = (k : ℕ) * k) ∧ (N = 91) :=
by
  sorry

end two_digit_number_satisfies_conditions_l602_602211


namespace distance_between_planes_l602_602676

noncomputable def plane1 := {p : ℝ × ℝ × ℝ | p.1 - 3 * p.2 + 3 * p.3 = 8}
noncomputable def plane2 := {p : ℝ × ℝ × ℝ | 2 * p.1 - 6 * p.2 + 6 * p.3 = 2}

noncomputable def distance (A B C D : ℝ) (p : ℝ × ℝ × ℝ) : ℝ :=
  (abs (A * p.1 + B * p.2 + C * p.3 + D)) / (sqrt (A^2 + B^2 + C^2))

theorem distance_between_planes : 
  let p : ℝ × ℝ × ℝ := (1, 0, 0) in
  distance 1 (-3) 3 (-8) p = 7 * sqrt 19 / 19 :=
by sorry

end distance_between_planes_l602_602676


namespace sets_equal_sufficient_condition_l602_602352

variable (a : ℝ)

-- Define sets A and B
def A (x : ℝ) : Prop := 0 < a * x + 1 ∧ a * x + 1 ≤ 5
def B (x : ℝ) : Prop := -1/2 < x ∧ x ≤ 2

-- Statement for Part 1: Sets A and B can be equal if and only if a = 2
theorem sets_equal (h : ∀ x, A a x ↔ B x) : a = 2 :=
sorry

-- Statement for Part 2: Proposition p ⇒ q holds if and only if a > 2 or a < -8
theorem sufficient_condition (h : ∀ x, A a x → B x) (h_neq : ∃ x, B x ∧ ¬A a x) : a > 2 ∨ a < -8 :=
sorry

end sets_equal_sufficient_condition_l602_602352


namespace count_multiples_2_or_5_but_not_6_l602_602361

theorem count_multiples_2_or_5_but_not_6 : 
  {n : ℕ | n ≤ 200 ∧ (n % 2 = 0 ∨ n % 5 = 0) ∧ n % 6 ≠ 0}.card = 87 := 
  sorry

end count_multiples_2_or_5_but_not_6_l602_602361


namespace part_a_part_b_l602_602164

-- Define the setup
def total_people := 100
def total_men := 50
def total_women := 50

-- Peter Ivanovich's position and neighbor relations
def pi_satisfied_prob : ℚ := 25 / 33

-- Expected number of satisfied men
def expected_satisfied_men : ℚ := 1250 / 33

-- Lean statements for the problems

-- Part (a): Prove Peter Ivanovich's satisfaction probability
theorem part_a (total_people = 100) (total_men = 50) (total_women = 50) : 
  pi_satisfied_prob = 25 / 33 := 
sorry

-- Part (b): Expected number of satisfied men
theorem part_b (total_people = 100) (total_men = 50) (total_women = 50) : 
  expected_satisfied_men = 1250 / 33 := 
sorry

end part_a_part_b_l602_602164


namespace incorrect_reasoning_l602_602517

noncomputable def is_exponential_function (a : ℝ) (hx : a > 0 ∧ a ≠ 1) : Prop :=
∀ x : ℝ, real.pow a x = y

theorem incorrect_reasoning (a : ℝ) (h1 : a = 1/3) :
  ∀ x : ℝ, y = (1/3)^x ∧ (0 < 1/3 ∧ 1/3 < 1) → ¬(∀ x : ℝ, (1/3)^x > 0) :=
by
  sorry

end incorrect_reasoning_l602_602517


namespace triangle_centroid_angle_inequality_l602_602020

theorem triangle_centroid_angle_inequality
  (A B C : Point)
  (hBC_CA_AB : dist B C > dist C A ∧ dist C A > dist A B)
  (G : Point)
  (hG_centroid : is_centroid G A B C) :
  ∠GCA + ∠GBC < ∠ BAC ∧ ∠ BAC < ∠GAC + ∠GBA :=
by 
  sorry

end triangle_centroid_angle_inequality_l602_602020


namespace find_c_plus_d_l602_602834

theorem find_c_plus_d (c d : ℝ) (h : Polynomial.eval (2 + Complex.I * Real.sqrt 2 : ℂ) (X^3 + C c * X + C d : Polynomial ℂ) = 0) : c + d = 14 := sorry

end find_c_plus_d_l602_602834


namespace minimal_maximal_arrangement_l602_602126

noncomputable def gamma_piece : ℕ := 3
def chessboard_size : ℕ := 8
def sub_square_size : ℕ := 2
def total_squares := chessboard_size * chessboard_size
def total_sub_squares := (chessboard_size / sub_square_size) * (chessboard_size / sub_square_size)

theorem minimal_maximal_arrangement :
  ∀ (gamma_count : ℕ),
    (gamma_count * gamma_piece + total_sub_squares = total_squares)
    → (gamma_count = 16) :=
by
  intro gamma_count
  assume h : gamma_count * gamma_piece + total_sub_squares = total_squares
  sorry

end minimal_maximal_arrangement_l602_602126


namespace function_changes_sign_75_times_l602_602746

-- Define the function f(x) as described in the problem
def f (x : ℝ) : ℝ := (List.range 2009).foldr (λ k acc, acc * Math.cos (x / (k + 1))) 1

-- Define the interval [0, 2009π/2]
def interval : Set ℝ := Set.Icc 0 (2009 * Real.pi / 2)

-- Define the function that counts the number of sign changes in a given function over an interval
def count_sign_changes (f : ℝ → ℝ) (s : Set ℝ) : ℕ := sorry

-- The statement we want to prove
theorem function_changes_sign_75_times :
  count_sign_changes f interval = 75 := sorry

end function_changes_sign_75_times_l602_602746


namespace symmetric_Bernoulli_walk_equality_in_distribution_l602_602567

-- Definitions based on given conditions
def S (k : ℕ) (ξ : ℕ → ℤ) (n : ℕ) : ℤ :=
  if k = 0 then 0
  else (List.sum (List.map ξ (List.range (k + 1))))

def M (n : ℕ) (S : ℕ → ℤ) : ℤ := 
  List.foldl max 0 (List.map S (List.range n))

def m (n : ℕ) (S : ℕ → ℤ) : ℤ := 
  List.foldl min 0 (List.map S (List.range n))

-- Random variable definitions, Bernoulli distribution setup
noncomputable def ξ (i : ℕ) : ℤ := 
  if Classical.choose (∃ξ : ℤ, ξ = 1 ∨ ξ = -1) i = 1 then 1 else -1

-- The theorem that needs to be proved
theorem symmetric_Bernoulli_walk_equality_in_distribution (n : ℕ) (S : ℕ → ℤ) (ξ : ℕ → ℤ) : 
  (M n S, -m n S, -S n) ∼ (-m n S, M n S, S n) ∧
  (M n S, -m n S, -S n) ∼ (M n S - S n, S n - m n S, S n) :=
sorry

end symmetric_Bernoulli_walk_equality_in_distribution_l602_602567


namespace option_A_is_positive_l602_602773

variable {x : ℝ}

theorem option_A_is_positive (hx : x < 0) : - (x / (2 * |x|)) > 0 := 
by
  have h_abs_x : |x| = -x := abs_of_nonpos (le_of_lt hx)
  calc
    - (x / (2 * |x|)) = - (x / (2 * (-x))) : by rw h_abs_x
                   ... = - (x / (-2 * x))  : by rw neg_mul_eq_mul_neg
                   ... = - (-1 / 2) : by rw [neg_div_neg_eq_div, one_div_mul_cancel hx.ne.symm]
                   ... = 1 / 2 : by rw neg_neg
  have h_pos : (1 / 2 : ℝ) > 0 := by norm_num
  exact h_pos

end option_A_is_positive_l602_602773


namespace square_perimeter_l602_602596

noncomputable def side_length_of_square_with_area (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def perimeter_of_square_with_side (side : ℝ) : ℝ :=
  4 * side

theorem square_perimeter {area : ℝ} (h_area : area = 625) :
  perimeter_of_square_with_side (side_length_of_square_with_area area) = 100 :=
by
  have h_side_length : side_length_of_square_with_area area = 25 := by
    rw [side_length_of_square_with_area, real.sqrt, h_area]
    norm_num
  rw [perimeter_of_square_with_side, h_side_length]
  norm_num
  sorry

end square_perimeter_l602_602596


namespace unique_solution_inequality_num_real_a_l602_602314

noncomputable def discriminant (a : ℝ) : ℝ :=
  (2*a)^2 - 4*(3*a - 2)

theorem unique_solution_inequality :
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) ↔ ∃! a : ℝ, (discriminant a = 0) :=
sorry

theorem num_real_a :
  ∃! a : ℝ, (unique_solution_inequality) :=
sorry

#reduce 2

end unique_solution_inequality_num_real_a_l602_602314


namespace last_number_with_35_zeros_l602_602471

def count_zeros (n : Nat) : Nat :=
  if n = 0 then 1
  else if n < 10 then 0
  else count_zeros (n / 10) + count_zeros (n % 10)

def total_zeros_written (upto : Nat) : Nat :=
  (List.range (upto + 1)).foldl (λ acc n => acc + count_zeros n) 0

theorem last_number_with_35_zeros : ∃ n, total_zeros_written n = 35 ∧ ∀ m, m > n → total_zeros_written m ≠ 35 :=
by
  let x := 204
  have h1 : total_zeros_written x = 35 := sorry
  have h2 : ∀ m, m > x → total_zeros_written m ≠ 35 := sorry
  existsi x
  exact ⟨h1, h2⟩

end last_number_with_35_zeros_l602_602471


namespace logarithmic_sum_inequality_l602_602727

open Real

theorem logarithmic_sum_inequality (n : ℕ) (h : 2 ≤ n) : 
  (∑ k in finset.range n \ finset.range 1, (log (k + 1) : ℝ) / (k + 2)) < (n * (n - 1) / 4 : ℝ) :=
sorry

end logarithmic_sum_inequality_l602_602727


namespace lines_perpendicular_to_plane_are_parallel_l602_602701

-- Definitions for lines and planes
variables {Point : Type} [linear_order Point] {Line Plane : Type}
variables (a b : Line) (α : Plane)

-- Relations: parallelism and perpendicularity, and subset relation
variables [has_parallel Line Plane] [has_perp Line Plane] [has_subset Line Plane]

-- Stating the theorem: If a is perpendicular to α and b is perpendicular to α, then a is parallel to b
theorem lines_perpendicular_to_plane_are_parallel (h1 : a ⊥ α) (h2 : b ⊥ α) : a ∥ b :=
sorry

end lines_perpendicular_to_plane_are_parallel_l602_602701


namespace find_b_l602_602093

theorem find_b (b : ℚ) (H : ∃ x y : ℚ, x = 3 ∧ y = -7 ∧ b * x + (b - 1) * y = b + 3) : 
  b = 4 / 5 := 
by
  sorry

end find_b_l602_602093


namespace sin_cos_identity_l602_602766

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602766


namespace wall_passing_pattern_l602_602814

theorem wall_passing_pattern (n : ℕ) : (8^2 - 1 = n) → (8 * sqrt (8 / n) = sqrt (8 * 8 / n)) :=
by
  intro h
  rw h
  sorry

end wall_passing_pattern_l602_602814


namespace fib_10_eq_55_l602_602072

def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem fib_10_eq_55 : fib 10 = 55 := by
  sorry

end fib_10_eq_55_l602_602072


namespace common_chord_passes_through_orthocenter_l602_602055

noncomputable def triangle_circles_common_chord 
  (A B C M N : Point) 
  (hM : M ∈ line_through A B)
  (hN : N ∈ line_through A C) 
  (hCM : Circle := circle_diameter_through C M)
  (hBN : Circle := circle_diameter_through B N)
  : Prop :=
  let O := orthocenter A B C in
  chord_pass_through_intersection hCM hBN O

/- Proof goes here -/
theorem common_chord_passes_through_orthocenter 
  (A B C M N : Point) 
  (hM : M ∈ line_through A B)
  (hN : N ∈ line_through A C) 
  : triangle_circles_common_chord A B C M N hM hN :=
sorry

end common_chord_passes_through_orthocenter_l602_602055


namespace algae_doubling_l602_602905

theorem algae_doubling {d : ℕ} (h1 : ∀ n, covered (n + 1) = 2 * covered n) (h2 : covered 15 = 1) :
  d = 11 ↔ (free d ≥ 0.9 ∧ free (d + 1) < 0.9) :=
by
  let free_days_ago := λ n, free 15 / 2 ^ (15 - n)
  have free_day_11 : free_days_ago 11 = free 11 := sorry
  have free_day_12 : free_days_ago 12 = free 12 := sorry
  have : free_days_ago 11 ≥ 0.9 := by sorry
  have : free_days_ago 12 < 0.9 := by sorry
  exact (free 11 = free_days_ago 11 ∧ free 12 = free_days_ago 12) ▸ ⟨by sorry, by sorry⟩

end algae_doubling_l602_602905


namespace profit_expression_max_profit_result_l602_602494

-- Definitions based on the problem conditions
def yield (x : ℝ) : ℝ := (16/5) - (4 / (x + 1))

def fertilizer_cost (x : ℝ) : ℝ := x

def additional_cost (x : ℝ) : ℝ := 3 * x

def market_price : ℝ := 25

-- Define the profit function based on the given expressions
def profit (x : ℝ) : ℝ := 80 - (100 / (x + 1)) - 4 * x

-- Assertion 1: The analytical expression of the profit function
theorem profit_expression (x : ℝ) (h : 0 < x ∧ x ≤ 500) :
  profit(x) = 80 - 100 / (x + 1) - 4 * x := 
by sorry

-- Assertion 2: The fertilizer cost per acre that yields the maximum profit
theorem max_profit_result (h : 0 < 4 ∧ 4 ≤ 500) :
  profit(4) = 44 :=
by sorry

end profit_expression_max_profit_result_l602_602494


namespace proposition_p_is_necessary_but_not_sufficient_condition_for_q_l602_602057

-- Definitions of propositions p and q
def p : Prop := ∃ l C, (line_has_exactly_one_common_point_with_parabola l C)
def q : Prop := ∃ l C, (line_is_tangent_to_parabola l C)

-- The statement to prove
def proof_problem : Prop :=
  (p → q) ∧ ¬(q → p)

-- Assuming the conditions in the solution
variable (l : Line) (C : Parabola)
variable (line_has_exactly_one_common_point_with_parabola : Line → Parabola → Prop)
variable (line_is_tangent_to_parabola : Line → Parabola → Prop)

-- Statement in Lean 4
theorem proposition_p_is_necessary_but_not_sufficient_condition_for_q :
  proof_problem :=
by
  sorry

end proposition_p_is_necessary_but_not_sufficient_condition_for_q_l602_602057


namespace lcm_of_9_12_15_is_180_l602_602989

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l602_602989


namespace tangent_min_perimeter_l602_602125

noncomputable def Point : Type := {
  x : ℚ,
  y : ℚ
}

def Angle (A K L : Point) : Prop := sorry
def Inside (M : Point) (A K L : Point) : Prop := sorry
def Circle (S : Point → Prop) : Prop := sorry
def Inscribed (S : Point → Prop) (K A L : Point) : Prop := sorry
def LargerInscribedCircle (S : Point → Prop) (M : Point) (K A L : Point) : Prop := sorry
def Tangent (M : Point) (S : Point → Prop) (t : Point → Point → Prop) : Prop := sorry
def Triangle (M B C : Point) : Prop := sorry
def Perimeter (M B C : Point) : ℚ := sorry
def SmallestPerimeter (M : Point) (K A L : Point) := sorry

theorem tangent_min_perimeter 
  (K A L M : Point)
  (h1 : Inside M K A L)
  (S : Point → Prop)
  (h2 : LargerInscribedCircle S M K A L)
  (t : Point → Point → Prop)
  (h3 : Tangent M S t):
  SmallestPerimeter M K A L :=
sorry

end tangent_min_perimeter_l602_602125


namespace find_x_l602_602282

theorem find_x (x : ℝ) (h : 9^(Real.log x / Real.log 8) = 81) : x = 64 := by
  have h₁ : 9 = 3^2 := by norm_num
  have h₂ : 81 = 9^2 := by norm_num
  rw [h₁] at h
  rw [h₂] at h
  have log_identity : (Real.log x / Real.log 8) = 2 := by sorry
  have x_value : x = 8^2 := by sorry
  rw [x_value]
  norm_num
  sorry

end find_x_l602_602282


namespace rectangle_area_expression_l602_602931

theorem rectangle_area_expression {d x : ℝ} (h : d^2 = 29 * x^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = (10 / 29) :=
by {
 sorry
}

end rectangle_area_expression_l602_602931


namespace collinear_incenter_circumcenter_O_l602_602354

structure Triangle (α : Type) := 
(A B C : α)

structure Circle (α : Type) :=
(center : α)
(radius : ℝ)

variable {α : Type}

variables (O : α) (ABC : Triangle α) (circles : List (Circle α))

noncomputable def incenter (t : Triangle α) : α := sorry
noncomputable def circumcenter (t : Triangle α) : α := sorry

axiom identical_circles (l : List (Circle α)) : l.all (λ c, c.radius = l.head!.radius) -- All circles have identical radii
axiom common_point (l : List (Circle α)) (P : α) : l.all (λ c, P ∈ {p | distance c.center p = c.radius}) -- Common intersection point O
axiom tangent_to_sides (t : Triangle α) (c : Circle α) : sorry -- Each circle is tangent to two sides of the triangle

theorem collinear_incenter_circumcenter_O 
  (h_identical : identical_circles circles)
  (h_common_point : common_point circles O)
  (h_tangent : ∀ c ∈ circles, tangent_to_sides ABC c) :
  collinear {incenter ABC, circumcenter ABC, O} := 
sorry

end collinear_incenter_circumcenter_O_l602_602354


namespace train_length_is_110_l602_602221

-- Definitions based on conditions
def speed_kmh : ℝ := 45  -- Speed of the train in km/hr
def crossing_time : ℝ := 30  -- Time to cross the bridge in seconds
def bridge_length : ℝ := 265  -- Length of the bridge in meters

-- Convert speed from km/hr to m/s
def speed_ms : ℝ := speed_kmh * 1000 / 3600

-- Define the total distance covered in terms of meters
def total_distance : ℝ := speed_ms * crossing_time

-- Define the length of the train
def train_length : ℝ := total_distance - bridge_length

-- Theorem stating that the length of the train is 110 meters
theorem train_length_is_110 : train_length = 110 :=
  sorry

end train_length_is_110_l602_602221


namespace sin_cos_product_l602_602755

-- Define the problem's main claim
theorem sin_cos_product (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 :=
by
  have h1 : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := Real.sin_square_add_cos_square x
  sorry

end sin_cos_product_l602_602755


namespace find_special_number_l602_602206

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def is_multiple_of_13 (n : ℕ) : Prop := n % 13 = 0
def digits_product_is_square (n : ℕ) : Prop :=
  let digits := (Nat.digits 10 n) in
  let product := List.prod digits in
  ∃ m : ℕ, m * m = product

theorem find_special_number : ∃ N : ℕ,
  0 < N ∧ -- N is positive
  is_two_digit N ∧ -- N is a two-digit number
  is_odd N ∧ -- N is odd
  is_multiple_of_13 N ∧ -- N is a multiple of 13
  digits_product_is_square N := -- The product of its digits is a perfect square
begin
  -- Proof omitted
  sorry
end

end find_special_number_l602_602206


namespace number_of_circles_l602_602325

theorem number_of_circles :
  let a_values := {3, 4, 6}
  let b_values := {1, 2, 7, 8}
  let r_values := {8, 9}
  (∀ a ∈ a_values, ∀ b ∈ b_values, ∀ r ∈ r_values, (exists x y, (x - a) ^ 2 + (y - b)^ 2 = r^2)) → 
  (3 * 4 * 2 = 24) :=
by
  let a_values := {3, 4, 6}
  let b_values := {1, 2, 7, 8}
  let r_values := {8, 9}
  sorry

end number_of_circles_l602_602325


namespace sum_of_digits_y_coord_l602_602424

noncomputable def coordinates_of_A_B_C (a b c : ℝ) : Prop :=
distinct_points a b c ∧ vertical_line a b ∧ right_triangle_area_2016 a b c

theorem sum_of_digits_y_coord (a b c : ℝ) (h : coordinates_of_A_B_C a b c) : 
  sum_of_digits (y_coord_of_C c) = 0 :=
by
  sorry

-- Definitions of helper terms used above
def distinct_points (a b c : ℝ) : Prop := a ≠ c ∧ b ≠ c ∧ a ≠ b
def vertical_line (a b : ℝ) : Prop := true -- line AB perpendicular to x-axis
def right_triangle_area_2016 (a b c : ℝ) : Prop := 0.5 * |a^2 - b^2| * |a - c| = 2016
def y_coord_of_C (c : ℝ) : ℝ := 0
def sum_of_digits (n : ℝ) : ℝ := 0 -- y-coordinate of C is 0, sum of digits is 0

end sum_of_digits_y_coord_l602_602424


namespace intersection_correct_l602_602353

def M := {1, 2, 3} : Set ℕ
def N := {1, 2} : Set ℕ

theorem intersection_correct : M ∩ N = {1, 2} := by
  sorry

end intersection_correct_l602_602353


namespace find_positive_int_sol_l602_602287

theorem find_positive_int_sol (a b c d n : ℕ) (h1 : n > 1) (h2 : a ≤ b) (h3 : b ≤ c) :
  ((n^a + n^b + n^c = n^d) ↔ 
  ((a = b ∧ b = c - 1 ∧ c = d - 1 ∧ n = 2) ∨ 
  (a = b ∧ b = c ∧ c = d - 1 ∧ n = 3))) :=
  sorry

end find_positive_int_sol_l602_602287


namespace jamesOreos_count_l602_602005

noncomputable def jamesOreos (jordanOreos : ℕ) : ℕ := 4 * jordanOreos + 7

theorem jamesOreos_count (J : ℕ) (h1 : J + jamesOreos J = 52) : jamesOreos J = 43 :=
by
  sorry

end jamesOreos_count_l602_602005


namespace length_of_train_l602_602225

/-- The length of the train given that:
* it is traveling at 45 km/hr
* it crosses a bridge in 30 seconds
* the bridge is 265 meters long
is 110 meters.
-/
theorem length_of_train (speed : ℕ) (time : ℕ) (bridge_length : ℕ) (train_length : ℕ)
  (h1 : speed = 45) (h2 : time = 30) (h3 : bridge_length = 265) :
  (train_length : ℕ) = 110 :=
begin
  sorry
end

end length_of_train_l602_602225


namespace find_q_minus_p_l602_602837

variable {p q : ℕ} (hpq : p > 0 ∧ q > 0)
variable (hfrac1 : 4 * q < 7 * p)
variable (hfrac2 : 7 * p < 12 * q)
variable (hminq : ∀ r r' : ℕ, r > 0 ∧ r' > 0 → (4 * r' < 7 * r) ∧ (7 * r < 12 * r') → r ≥ q)

theorem find_q_minus_p (hpq : p = 11) (hq : q = 19) : q - p = 8 := by
  rw [hq, hpq]
  norm_num

end find_q_minus_p_l602_602837


namespace triangle_obtuse_of_exterior_angle_lt_interior_l602_602376

theorem triangle_obtuse_of_exterior_angle_lt_interior (T : Triangle)
  (ext_angle : ∀ A B C : Point, ∃ α β γ : ℝ, sum_of_angles_eq_180 A B C α β γ ∧ α > γ) : T.is_obtuse :=
  sorry

end triangle_obtuse_of_exterior_angle_lt_interior_l602_602376


namespace shingles_needed_l602_602917

structure Dimensions where
  length : ℝ
  width : ℝ

def area (d : Dimensions) : ℝ :=
  d.length * d.width

def houseDimensions : Dimensions := { length := 20.5, width := 10 }
def porchDimensions : Dimensions := { length := 6, width := 4.5 }

def totalArea (d1 d2 : Dimensions) : ℝ :=
  area d1 + area d2

theorem shingles_needed :
  totalArea houseDimensions porchDimensions = 232 :=
by
  simp [totalArea, area, houseDimensions, porchDimensions]
  norm_num
  sorry

end shingles_needed_l602_602917


namespace goose_eggs_l602_602049

theorem goose_eggs (E : ℕ) (hatch_rate : ℚ) (month_survival_rate : ℚ) 
  (year_survival_fraction : ℚ) (survived_first_year : ℕ)
  (hatched_single_per_egg : ∀ n, n = E ∧ (1 / 4 : ℚ) * (4 / 5 : ℚ) * (3 / 5 : ℚ) n = 120 → n = 1000) :
  E = 1000 :=
sorry

end goose_eggs_l602_602049


namespace birthday_stickers_l602_602849

def initial_stickers := 20
def bought_stickers := 12
def given_away_stickers := 5
def used_stickers := 8
def stickers_left := 39

theorem birthday_stickers : 
  let total_stickers_before_giving := stickers_left + given_away_stickers + used_stickers in
  let total_stickers_after_buying := initial_stickers + bought_stickers in
  total_stickers_before_giving - total_stickers_after_buying = 20 :=
by
  sorry

end birthday_stickers_l602_602849


namespace percentage_of_women_not_speak_french_is_correct_l602_602182

noncomputable def total_employees : ℕ := 100
noncomputable def percentage_men : ℤ := 45
noncomputable def percentage_men_speak_french : ℤ := 60
noncomputable def percentage_employees_speak_french : ℤ := 40

noncomputable def number_of_men : ℕ := (total_employees * percentage_men) / 100
noncomputable def number_of_men_speak_french : ℕ := (number_of_men * percentage_men_speak_french) / 100
noncomputable def number_of_employees_speak_french : ℕ := (total_employees * percentage_employees_speak_french) / 100

noncomputable def number_of_women : ℕ := total_employees - number_of_men
noncomputable def number_of_women_speak_french : ℕ := number_of_employees_speak_french - number_of_men_speak_french
noncomputable def number_of_women_not_speak_french : ℕ := number_of_women - number_of_women_speak_french

noncomputable def percentage_women_not_speak_french : ℚ :=
  (number_of_women_not_speak_french / number_of_women : ℚ) * 100

theorem percentage_of_women_not_speak_french_is_correct :
  percentage_women_not_speak_french = 76.36 :=
by
  sorry

end percentage_of_women_not_speak_french_is_correct_l602_602182


namespace parallelogram_perimeter_l602_602204

-- Definitions based on the conditions
def is_equilateral_triangle (a b c : ℝ) : Prop :=
  a = b ∧ b = c

def is_small_parallelogram (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.2 = p3.2 ∧ p2.2 = p4.2

noncomputable def count_equilateral_triangles (s : ℝ) : ℕ :=
  92

noncomputable def possible_side_lengths (n : ℕ) : list (ℝ × ℝ) :=
  if n = 46 then [(46, 1), (23, 2)] else []

-- Proof statement
theorem parallelogram_perimeter {s : ℝ} (n : ℕ) (h : s = 1) (t : count_equilateral_triangles s = 92) :
  (possible_side_lengths n = [(46, 1), (23, 2)] → (n = 46)) →
  (possible_side_lengths n = [(23, 2)] → (n = 46)) →
  (n = 46 → (list.prod (possible_side_lengths n)).1 = 94 ∨ (list.prod (possible_side_lengths n)).1 = 50) :=
begin
  sorry
end

end parallelogram_perimeter_l602_602204


namespace sum_fifth_to_seventh_terms_arith_seq_l602_602914

theorem sum_fifth_to_seventh_terms_arith_seq (a d : ℤ)
  (h1 : a + 7 * d = 16) (h2 : a + 8 * d = 22) (h3 : a + 9 * d = 28) :
  (a + 4 * d) + (a + 5 * d) + (a + 6 * d) = 12 :=
by
  sorry

end sum_fifth_to_seventh_terms_arith_seq_l602_602914


namespace strongLinearRelationship_calculateRegressionEquation_expectedConsumption_actualConsumptionAnalysis_l602_602066

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def correlationCoefficient (xs ys : List ℝ) (xbar ybar : ℝ) : ℝ :=
  (List.zipWith (λ x y => (x - xbar) * (y - ybar)) xs ys).sum / 
  (Real.sqrt ((xs.map (λ x => (x - xbar) ^ 2)).sum) *
   Real.sqrt ((ys.map (λ y => (y - ybar) ^ 2)).sum))

noncomputable def regressionCoeff (xs ys : List ℝ) (xbar ybar : ℝ) : ℝ :=
  ((List.zipWith (λ x y => (x - xbar) * (y - ybar)) xs ys).sum) /
  ((xs.map (λ x => (x - xbar) ^ 2)).sum)

noncomputable def regressionIntercept (ybar b xbar : ℝ) : ℝ :=
  ybar - b * xbar

noncomputable def regressionEquation (b a x : ℝ) : ℝ :=
  b * x + a

theorem strongLinearRelationship (xs ys : List ℝ) (xbar ybar : ℝ) (r : ℝ) :
  correlationCoefficient xs ys xbar ybar = r →
  |r| > 0.75 :=
by
  intros h
  rw [h]
  sorry

theorem calculateRegressionEquation 
  (xs ys : List ℝ) (xbar ybar : ℝ) (b a : ℝ)
  (h1 : regressionCoeff xs ys xbar ybar = b)
  (h2 : regressionIntercept ybar b xbar = a) :
  ∀ x, regressionEquation b a x = 3.45 * x + 0.75 :=
by
  intros x
  rw [h1, h2]
  sorry

theorem expectedConsumption
  (x : ℝ) (expected : ℝ)
  (h : regressionEquation 3.45 0.75 x = expected) :
  expected = 35.25 :=
by
  rw [h]
  sorry

theorem actualConsumptionAnalysis
  (estimated actual : ℝ) :
  estimated = 35.25 → actual = 30 →
  |actual - estimated| / estimated > 0.1 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end strongLinearRelationship_calculateRegressionEquation_expectedConsumption_actualConsumptionAnalysis_l602_602066


namespace find_b_l602_602339

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 4) : b = 4 :=
sorry

end find_b_l602_602339


namespace no_valid_subset_exists_l602_602657

-- Define the conditions where M is a subset of positive integers
def is_valid_subset (M : Set ℕ) : Prop :=
  ∀ r : ℚ, 0 < r ∧ r < 1 → ∃! S : Finset ℕ, (S ⊆ M ∧ (∑ s in S, (1 : ℚ) / s) = r)

theorem no_valid_subset_exists : ¬ ∃ M : Set ℕ, is_valid_subset M :=
sorry

end no_valid_subset_exists_l602_602657


namespace early_time_l602_602576

noncomputable def speed1 : ℝ := 5 -- km/hr
noncomputable def timeLate : ℝ := 5 / 60 -- convert minutes to hours
noncomputable def speed2 : ℝ := 10 -- km/hr
noncomputable def distance : ℝ := 2.5 -- km

theorem early_time (speed1 speed2 distance : ℝ) (timeLate : ℝ) :
  (distance / speed1 - timeLate) * 60 - (distance / speed2) * 60 = 10 :=
by
  sorry

end early_time_l602_602576


namespace coefficient_x3_in_expansion_l602_602816

theorem coefficient_x3_in_expansion :
  let coeff (n k : ℕ) := (Nat.choose n k) * (-1)^k in
  coeff 5 3 + coeff 6 3 + coeff 7 3 + coeff 8 3 = -121 :=
by
  sorry

end coefficient_x3_in_expansion_l602_602816


namespace problem_statement_l602_602425

open Real

noncomputable def a (m n : ℕ) [h1: 0 < m] [h2: 0 < n] : ℝ :=
  (m^(m+1) + n^(n+1)) / (m^m + n^n)

theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  let a := (m^(m+1) + n^(n+1)) / (m^m + n^n) in
  a^m + a^n ≥ ↑m^m + ↑n^n :=
by
  sorry

end problem_statement_l602_602425


namespace murtha_pebble_collection_l602_602454

theorem murtha_pebble_collection : 
  let pebbles_on_first_day := 2
  let pebbles_on_day (n : ℕ) : ℕ := if n = 1 then 2 else 3 * (n - 1) + 2
  let total_pebbles (n : ℕ) : ℕ := ∑ i in finset.range(n+1), pebbles_on_day(i)
  in total_pebbles 15 = 345 := by 
  sorry

end murtha_pebble_collection_l602_602454


namespace min_value_of_sum_l602_602329

theorem min_value_of_sum (x y : ℝ) (h1 : x + 4 * y = 2 * x * y) (h2 : 0 < x) (h3 : 0 < y) : 
  x + y ≥ 9 / 2 :=
sorry

end min_value_of_sum_l602_602329


namespace unique_value_of_a_l602_602508

theorem unique_value_of_a :
  ∃! a : ℝ, ∃ x : ℝ, (x ^ 2 + a * x + 1 = 0) ∧ (x ^ 2 - x - a = 0) :=
begin
   sorry
end

end unique_value_of_a_l602_602508


namespace keats_library_percentage_increase_l602_602418

theorem keats_library_percentage_increase :
  let total_books_A := 8000
  let total_books_B := 10000
  let total_books_C := 12000
  let initial_bio_A := 0.20 * total_books_A
  let initial_bio_B := 0.25 * total_books_B
  let initial_bio_C := 0.28 * total_books_C
  let total_initial_bio := initial_bio_A + initial_bio_B + initial_bio_C
  let final_bio_A := 0.32 * total_books_A
  let final_bio_B := 0.35 * total_books_B
  let final_bio_C := 0.40 * total_books_C
  --
  let total_final_bio := final_bio_A + final_bio_B + final_bio_C
  let increase_in_bio := total_final_bio - total_initial_bio
  let percentage_increase := (increase_in_bio / total_initial_bio) * 100
  --
  percentage_increase = 45.58 := 
by
  sorry

end keats_library_percentage_increase_l602_602418


namespace value_of_expression_when_x_is_neg2_l602_602145

theorem value_of_expression_when_x_is_neg2 : 
  ∀ (x : ℤ), x = -2 → (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_when_x_is_neg2_l602_602145


namespace total_marbles_in_bag_l602_602367

theorem total_marbles_in_bag (r b g y total_marbles : ℕ) 
  (h_ratio : r : b : g : y = 2 : 4 : 3 : 1) 
  (h_green_marbles : g = 24) 
  (h_total_marbles_eq : g * 10 / 3 = total_marbles) : 
  total_marbles = 80 := 
by 
  sorry

end total_marbles_in_bag_l602_602367


namespace arrange_abc_l602_602694

noncomputable def a := Real.log 3 / Real.log 0.2
noncomputable def b := 2⁻¹
noncomputable def c := Real.sin (Real.pi / 5)

theorem arrange_abc :
  a < b ∧ b < c :=
by
  have ha : a = Real.log 3 / Real.log 0.2 := by rfl
  have hb : b = 2⁻¹ := by rfl
  have hc : c = Real.sin (Real.pi / 5) := by rfl
  sorry

end arrange_abc_l602_602694


namespace num_repeating_decimals_l602_602305

theorem num_repeating_decimals : ∃ n : ℕ, n = 39 ∧ (∀ m : ℕ, (1 ≤ m ∧ m ≤ 50) → 
  let d := m + 1 in 
  ¬(d = 2 ∨ d = 4 ∨ d = 5 ∨ d = 8 ∨ d = 10 ∨ d = 16 ∨ d = 20 ∨ d = 25 ∨ d = 32 ∨ d = 40 ∨ d = 50) →
  ¬∀ (p : ℕ), (p ∣ d) → p = 2 ∨ p = 5) :=
by
  sorry

end num_repeating_decimals_l602_602305


namespace jamesOreos_count_l602_602004

noncomputable def jamesOreos (jordanOreos : ℕ) : ℕ := 4 * jordanOreos + 7

theorem jamesOreos_count (J : ℕ) (h1 : J + jamesOreos J = 52) : jamesOreos J = 43 :=
by
  sorry

end jamesOreos_count_l602_602004


namespace seats_per_bus_l602_602102

-- Conditions
def total_students : ℕ := 180
def total_buses : ℕ := 3

-- Theorem Statement
theorem seats_per_bus : (total_students / total_buses) = 60 := 
by 
  sorry

end seats_per_bus_l602_602102


namespace problem_f_2_eq_neg_2_l602_602440

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2_eq_neg_2 
  (hf : ∀ (x y : ℝ), f(f(x) + y) = f(x) - f(f(y) + f(-x)) + x) : 
  f 2 = -2 := 
sorry

end problem_f_2_eq_neg_2_l602_602440


namespace trig_identity_l602_602524

theorem trig_identity :
  cos (π / 12) * cos (π / 6) - sin (π / 12) * sin (π / 6) = sqrt 2 / 2 :=
by
  sorry

end trig_identity_l602_602524


namespace sqrt_inequality_l602_602235

theorem sqrt_inequality (x : ℝ) (hx : x > 0) : sqrt x + 1 / sqrt x ≥ 2 :=
sorry

end sqrt_inequality_l602_602235


namespace determine_tomorrow_l602_602879

-- Defining the days of the week
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open DayOfWeek

-- Defining a function to add a certain number of days to a given day
def addDays (start_day : DayOfWeek) (n : Nat) : DayOfWeek :=
  match start_day, n % 7 with
  | Monday, 0 => Monday
  | Monday, 1 => Tuesday
  | Monday, 2 => Wednesday
  | Monday, 3 => Thursday
  | Monday, 4 => Friday
  | Monday, 5 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 0 => Tuesday
  | Tuesday, 1 => Wednesday
  | Tuesday, 2 => Thursday
  | Tuesday, 3 => Friday
  | Tuesday, 4 => Saturday
  | Tuesday, 5 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 0 => Wednesday
  | Wednesday, 1 => Thursday
  | Wednesday, 2 => Friday
  | Wednesday, 3 => Saturday
  | Wednesday, 4 => Sunday
  | Wednesday, 5 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 0 => Thursday
  | Thursday, 1 => Friday
  | Thursday, 2 => Saturday
  | Thursday, 3 => Sunday
  | Thursday, 4 => Monday
  | Thursday, 5 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 0 => Friday
  | Friday, 1 => Saturday
  | Friday, 2 => Sunday
  | Friday, 3 => Monday
  | Friday, 4 => Tuesday
  | Friday, 5 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 0 => Saturday
  | Saturday, 1 => Sunday
  | Saturday, 2 => Monday
  | Saturday, 3 => Tuesday
  | Saturday, 4 => Wednesday
  | Saturday, 5 => Thursday
  | Saturday, 6 => Friday
  | Sunday, 0 => Sunday
  | Sunday, 1 => Monday
  | Sunday, 2 => Tuesday
  | Sunday, 3 => Wednesday
  | Sunday, 4 => Thursday
  | Sunday, 5 => Friday
  | Sunday, 6 => Saturday

-- Conditions
axiom condition : Monday = addDays x 5

-- Find the day of the week tomorrow
theorem determine_tomorrow (x : DayOfWeek) : addDays (addDays x 2) 1 = Saturday := sorry

end determine_tomorrow_l602_602879


namespace rectangle_area_increase_l602_602053

variable (l w : ℝ)

theorem rectangle_area_increase :
  let l' := 1.3 * l
      w' := 1.15 * w
      A := l * w
      A' := l' * w'
  in (A' - A) / A * 100 = 49.5 := by
  sorry

end rectangle_area_increase_l602_602053


namespace average_of_remaining_two_nums_l602_602078

theorem average_of_remaining_two_nums (S S4 : ℕ) (h1 : S / 6 = 8) (h2 : S4 / 4 = 5) :
  ((S - S4) / 2 = 14) :=
by 
  sorry

end average_of_remaining_two_nums_l602_602078


namespace arithmetic_sequence_sum_l602_602379

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + m) = a n + m * (a 1 - a 0)

theorem arithmetic_sequence_sum
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 5 + a 6 + a 7 = 15) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  sorry

end arithmetic_sequence_sum_l602_602379


namespace Bo_needs_to_learn_per_day_l602_602248

theorem Bo_needs_to_learn_per_day
  (total_flashcards : ℕ)
  (known_percentage : ℚ)
  (days_to_learn : ℕ)
  (h1 : total_flashcards = 800)
  (h2 : known_percentage = 0.20)
  (h3 : days_to_learn = 40) : 
  total_flashcards * (1 - known_percentage) / days_to_learn = 16 := 
by
  sorry

end Bo_needs_to_learn_per_day_l602_602248


namespace domain_of_f_max_value_condition_l602_602346

def f (x : ℝ) : ℝ := real.sqrt (4 - x) + real.sqrt (x - 1)

theorem domain_of_f :
  {x : ℝ | 1 ≤ x ∧ x ≤ 4} = set_of (λ x, 1 ≤ x ∧ x ≤ 4) :=
sorry

theorem max_value_condition {a : ℝ} (h : a ∈ set.Icc (3 / 2) 3) :
  f (a + 1) ≤ f a :=
sorry

end domain_of_f_max_value_condition_l602_602346


namespace find_special_number_l602_602205

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def is_multiple_of_13 (n : ℕ) : Prop := n % 13 = 0
def digits_product_is_square (n : ℕ) : Prop :=
  let digits := (Nat.digits 10 n) in
  let product := List.prod digits in
  ∃ m : ℕ, m * m = product

theorem find_special_number : ∃ N : ℕ,
  0 < N ∧ -- N is positive
  is_two_digit N ∧ -- N is a two-digit number
  is_odd N ∧ -- N is odd
  is_multiple_of_13 N ∧ -- N is a multiple of 13
  digits_product_is_square N := -- The product of its digits is a perfect square
begin
  -- Proof omitted
  sorry
end

end find_special_number_l602_602205


namespace number_of_small_gardens_l602_602456

def totalSeeds : ℕ := 85
def tomatoSeeds : ℕ := 42
def capsicumSeeds : ℕ := 26
def cucumberSeeds : ℕ := 17

def plantedTomatoSeeds : ℕ := 24
def plantedCucumberSeeds : ℕ := 17

def remainingTomatoSeeds : ℕ := tomatoSeeds - plantedTomatoSeeds
def remainingCapsicumSeeds : ℕ := capsicumSeeds
def remainingCucumberSeeds : ℕ := cucumberSeeds - plantedCucumberSeeds

def seedsInSmallGardenTomato : ℕ := 2
def seedsInSmallGardenCapsicum : ℕ := 1
def seedsInSmallGardenCucumber : ℕ := 1

theorem number_of_small_gardens : (remainingTomatoSeeds / seedsInSmallGardenTomato = 9) :=
by 
  sorry

end number_of_small_gardens_l602_602456


namespace compute_expression_l602_602636

theorem compute_expression : 
  let c := (3 + 2 * complex.I)
  let d := (2 - 3 * complex.I)
  3 * c + 4 * d = (17 - 6 * complex.I) :=
by
  let c := (3 + 2 * complex.I)
  let d := (2 - 3 * complex.I)
  sorry

end compute_expression_l602_602636


namespace average_of_first_12_results_l602_602907

theorem average_of_first_12_results
  (average_25_results : ℝ)
  (average_last_12_results : ℝ)
  (result_13th : ℝ)
  (total_results : ℕ)
  (num_first_12 : ℕ)
  (num_last_12 : ℕ)
  (total_sum : ℝ)
  (sum_first_12 : ℝ)
  (sum_last_12 : ℝ)
  (A : ℝ)
  (h1 : average_25_results = 24)
  (h2 : average_last_12_results = 17)
  (h3 : result_13th = 228)
  (h4 : total_results = 25)
  (h5 : num_first_12 = 12)
  (h6 : num_last_12 = 12)
  (h7 : total_sum = average_25_results * total_results)
  (h8 : sum_last_12 = average_last_12_results * num_last_12)
  (h9 : total_sum = sum_first_12 + result_13th + sum_last_12)
  (h10 : sum_first_12 = A * num_first_12) :
  A = 14 :=
by
  sorry

end average_of_first_12_results_l602_602907


namespace number_of_lines_passing_through_four_points_l602_602359

-- Defining the three-dimensional points and conditions
structure Point3D where
  x : ℕ
  y : ℕ
  z : ℕ
  h1 : 1 ≤ x ∧ x ≤ 5
  h2 : 1 ≤ y ∧ y ≤ 5
  h3 : 1 ≤ z ∧ z ≤ 5

-- Define a valid line passing through four distinct points (Readonly accessors for the conditions)
def valid_line (p1 p2 p3 p4 : Point3D) : Prop := 
  sorry -- Define conditions for points to be collinear and distinct

-- Main theorem statement
theorem number_of_lines_passing_through_four_points : 
  ∃ (lines : ℕ), lines = 150 :=
sorry

end number_of_lines_passing_through_four_points_l602_602359


namespace congruent_rectangles_in_6x6_l602_602184

theorem congruent_rectangles_in_6x6 :
  ∃ r₁ r₂ : ℕ × ℕ,
  r₁ ∈ ({(l * w) | l w : ℕ, l * w ≤ 36 ∧ l + w = 6}) ∧
  r₂ ∈ ({(l * w) | l w : ℕ, l * w ≤ 36 ∧ l + w = 6}) ∧
  r₁ ≠ r₂ := 
sorry

end congruent_rectangles_in_6x6_l602_602184


namespace flower_beds_l602_602054

theorem flower_beds (seeds_per_bed total_seeds flower_beds : ℕ) 
  (h1 : seeds_per_bed = 10) (h2 : total_seeds = 60) : 
  flower_beds = total_seeds / seeds_per_bed := by
  rw [h1, h2]
  sorry

end flower_beds_l602_602054


namespace product_of_solutions_l602_602271

theorem product_of_solutions (x : ℝ) :
  let a := -2
  let b := -8
  let c := -49
  ∀ x₁ x₂, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) → 
  x₁ * x₂ = 49/2 :=
sorry

end product_of_solutions_l602_602271


namespace eccentricity_of_ellipse_l602_602317

theorem eccentricity_of_ellipse
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (c : ℝ) (h3 : c = a * sqrt (1 - b^2 / a^2)) :
  let e := c / a in e = sqrt 5 / 5 :=
by
  sorry

end eccentricity_of_ellipse_l602_602317


namespace sin_cos_identity_l602_602758

theorem sin_cos_identity (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l602_602758


namespace find_A_when_B_is_largest_l602_602950

theorem find_A_when_B_is_largest :
  ∃ A : ℕ, ∃ B : ℕ, A = 17 * 25 + B ∧ B < 17 ∧ B = 16 ∧ A = 441 :=
by
  sorry

end find_A_when_B_is_largest_l602_602950


namespace cos_angle_BOC_l602_602588

theorem cos_angle_BOC (ABCD : Type) [AffineSpace Choice (Vec2 ABCD)] (O B C : Choice) (AC BD : ℝ) 
  (hAC : AC = 15) (hBD : BD = 40) (inter_diag : ∃ O ∈ ABCD, O = Midpoint AC ∧ O = Midpoint BD) :
  ∃ (cos_BOC : ℝ), cos_BOC = 1/2 :=
by
  sorry

end cos_angle_BOC_l602_602588


namespace more_advantageous_to_cut_log6_l602_602949

-- Definition of logs
def log_length := ℕ

-- The lengths of the two different types of logs
def log6 : log_length := 6
def log7 : log_length := 7

-- The total required length in meters
def total_length := 42

-- Number of cuts required to cut a log of given length into 1-meter pieces
def cuts_required (log_length : ℕ) (total_length : ℕ) : ℕ :=
  (total_length / log_length) * (log_length - 1)

-- The statement to prove
theorem more_advantageous_to_cut_log6 :
  cuts_required log6 total_length < cuts_required log7 total_length :=
by
  sorry

end more_advantageous_to_cut_log6_l602_602949


namespace monotonic_increasing_interval_l602_602923

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 - 2 * x - 3)

theorem monotonic_increasing_interval : 
  ∀ x y : ℝ, 3 ≤ x → x ≤ y → f x ≤ f y :=
by sorry

end monotonic_increasing_interval_l602_602923


namespace number_of_true_statements_l602_602429

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem number_of_true_statements (n : ℕ) :
  let s1 := reciprocal 4 + reciprocal 8 ≠ reciprocal 12
  let s2 := reciprocal 9 - reciprocal 3 ≠ reciprocal 6
  let s3 := reciprocal 5 * reciprocal 10 = reciprocal 50
  let s4 := reciprocal 16 / reciprocal 4 = reciprocal 4
  (cond s1 1 0) + (cond s2 1 0) + (cond s3 1 0) + (cond s4 1 0) = 2 := by
  sorry

end number_of_true_statements_l602_602429


namespace no_distributive_laws_hold_l602_602025

def avg (a b : ℝ) : ℝ := (a + b) / 3

theorem no_distributive_laws_hold (x y z : ℝ) :
  ¬ (avg x (y + z) = avg (avg x y) (avg x z)) ∧
  ¬ (x + avg y z = avg (x + y) (x + z)) ∧
  ¬ (avg x (avg y z) = avg (avg x y) (avg x z)) :=
by
  sorry

end no_distributive_laws_hold_l602_602025


namespace relationship_among_a_b_c_l602_602041

-- Definitions based on conditions
def a : ℝ := 0.6^(0.6)
def b : ℝ := 0.6^(1.5)
def c : ℝ := 1.5^(0.6)

-- Lean statement to prove the required relationship
theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l602_602041


namespace find_last_three_digits_of_9_pow_107_l602_602293

theorem find_last_three_digits_of_9_pow_107 : (9 ^ 107) % 1000 = 969 := 
by 
  sorry

end find_last_three_digits_of_9_pow_107_l602_602293


namespace total_expenditure_correct_l602_602797

def length : ℝ := 50
def width : ℝ := 30
def cost_per_square_meter : ℝ := 100

def area (L W : ℝ) : ℝ := L * W
def total_expenditure (A C : ℝ) : ℝ := A * C

theorem total_expenditure_correct :
  total_expenditure (area length width) cost_per_square_meter = 150000 := by
  sorry

end total_expenditure_correct_l602_602797


namespace juan_number_is_532_l602_602012

theorem juan_number_is_532 (j : ℕ) (k : ℕ) (h₁ : Maria's number = 10^(k+1) + 10 * j + 1) 
(h₂ : Maria's number - j = 14789) : j = 532 := by
  sorry

end juan_number_is_532_l602_602012


namespace dispatch_plans_l602_602957

theorem dispatch_plans (students : Finset ℕ) (h : students.card = 6) :
  ∃ (plans : Finset (Finset ℕ)), plans.card = 180 :=
by
  sorry

end dispatch_plans_l602_602957


namespace negation_of_universal_l602_602925

theorem negation_of_universal:
  ¬(∀ x : ℝ, (0 < x ∧ x < (π / 2)) → x > Real.sin x) ↔
  ∃ x : ℝ, (0 < x ∧ x < (π / 2)) ∧ x ≤ Real.sin x := by
  sorry

end negation_of_universal_l602_602925


namespace fraction_sum_proof_l602_602557

theorem fraction_sum_proof :
    (19 / ((2^3 - 1) * (3^3 - 1)) + 
     37 / ((3^3 - 1) * (4^3 - 1)) + 
     61 / ((4^3 - 1) * (5^3 - 1)) + 
     91 / ((5^3 - 1) * (6^3 - 1))) = (208 / 1505) :=
by
  -- Proof goes here
  sorry

end fraction_sum_proof_l602_602557


namespace MN_value_l602_602536

theorem MN_value (PQ PR QR : ℕ) (h_mid : ℝ) : 
  PQ = 24 → PR = 26 → QR = 30 → 
  let MN := (QR : ℝ) / 2 in
  (∃ (a b : ℕ), a / b = MN ∧ Nat.coprime a b ∧ a + b = 16) :=
by
-- Given data
intro hPQ hPR hQR
rw [hPQ, hPR, hQR]
use 15, 1
split
case : 
    rw [Nat.cast_succ]
case _: 
    exact Nat.coprime_one_left 15
case _: 
    norm_num

end MN_value_l602_602536
