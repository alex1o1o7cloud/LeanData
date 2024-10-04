import Data.Finset.Basic
import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Parity
import Mathlib.Analysis.Calculus.Circle
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.GCD
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Prob
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.NumberTheory.
import Mathlib.Probability.ProbabilityMassFunc
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Algebra.Order.Basic

namespace op_4_neg3_eq_neg28_l75_75317

def op (x y : Int) : Int := x * (y + 2) + 2 * x * y

theorem op_4_neg3_eq_neg28 : op 4 (-3) = -28 := by
  sorry

end op_4_neg3_eq_neg28_l75_75317


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75742

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75742


namespace choose_15_4_l75_75958

/-- The number of ways to choose 4 captains from a team of 15 people is 1365. -/
theorem choose_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end choose_15_4_l75_75958


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75735

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75735


namespace choose_4_captains_from_15_l75_75967

def num_ways_to_choose_captains : ℕ := 15.choose 4

theorem choose_4_captains_from_15 : num_ways_to_choose_captains = 1365 := by
  sorry

end choose_4_captains_from_15_l75_75967


namespace area_triangle_BRS_l75_75617

-- Given conditions
structure Point (α : Type*) := (x : α) (y : α)

def B := Point.mk 3 4

-- Assume lines y = m1 x + c1 and y = m2 x + c2 with the perpendicular condition
variables {m₁ m₂ c₁ c₂ : ℝ}
axiom perp_lines : m₁ * m₂ = -1
axiom intercept_sum : c₁ + c₂ = B.x

-- Goal: area of triangle BRS = 9
theorem area_triangle_BRS (h1 : perp_lines) (h2 : intercept_sum) : 
  let R := Point.mk 0 c₁,
      S := Point.mk 0 c₂ in
  let area := 1/2 * abs (c₁ - c₂) * B.x in
  area = 9 := by 
  sorry

end area_triangle_BRS_l75_75617


namespace juniors_score_l75_75940

theorem juniors_score (n : ℕ) (juniors seniors : ℕ) (score_total : ℕ) :
  juniors = 0.2 * (n : ℕ) →
  seniors = 0.8 * (n : ℕ) →
  (score_total : ℕ) = 85 * (n : ℕ) →
  (∀ j, j ∈ juniors → j = (j : ℕ)) →
  (seniors_avg_score : ℕ) = 84 →
  juniors_avg_score juniors seniors score_total seniors_avg_score = 89 :=
by
  sorry

end juniors_score_l75_75940


namespace largest_binom_coeff_l75_75183

theorem largest_binom_coeff (n : ℕ) :
  (nat.choose 10 3) + (nat.choose 10 4) = nat.choose 11 n → n = 7 :=
by sorry

end largest_binom_coeff_l75_75183


namespace faye_age_l75_75845

def ages (C D E F : ℕ) :=
  D = E - 2 ∧
  E = C + 3 ∧
  F = C + 4 ∧
  D = 15

theorem faye_age (C D E F : ℕ) (h : ages C D E F) : F = 18 :=
by
  unfold ages at h
  sorry

end faye_age_l75_75845


namespace part1_part2_l75_75409

noncomputable def z (b : ℝ) : ℂ := 3 + b * complex.I

theorem part1 (b : ℝ) (hb1 : (1 + 3 * complex.I) * z b = 3 + b * (complex.I) → 3 - 3 * b = 0) : z 1 = 3 + complex.I :=
sorry

noncomputable def w : ℂ := (3 + complex.I) / (2 + complex.I)

theorem part2 : complex.abs w = real.sqrt 2 :=
sorry

end part1_part2_l75_75409


namespace roots_modulus_less_than_one_l75_75985

theorem roots_modulus_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ x, x^2 + A * x + B = 0 → |x| < 1)
  (h2 : ∀ x, x^2 + C * x + D = 0 → |x| < 1) :
  ∀ x, x^2 + (A + C) / 2 * x + (B + D) / 2 = 0 → |x| < 1 :=
by
  sorry

end roots_modulus_less_than_one_l75_75985


namespace arithmetic_mean_missing_digit_l75_75582

theorem arithmetic_mean_missing_digit :
  let S := {8, 88, 888, 8888, 88888} in
  let M := 17777 in
  ¬ ↑('8') ∈ (finset.digits M) :=
begin
  sorry
end

end arithmetic_mean_missing_digit_l75_75582


namespace even_product_of_differences_l75_75062

variables {a : Fin 7 → ℤ} {b : Fin 7 → ℤ}

def is_permutation_of (a b : Fin 7 → ℤ) : Prop :=
  ∃σ : Equiv.Perm (Fin 7), ∀ i, b i = a (σ i)

theorem even_product_of_differences (h : is_permutation_of a b) :
  Even (∏ i, (a i - b i)) :=
  sorry

end even_product_of_differences_l75_75062


namespace votes_for_candidate_a_correct_l75_75944

noncomputable def total_votes : ℝ := 900000

def percentage_invalid_votes : ℝ := 20 / 100

def percentage_valid_votes : ℝ := 1 - percentage_invalid_votes

noncomputable def valid_votes : ℝ := total_votes * percentage_valid_votes

def percentage_candidate_a : ℝ := 55 / 100

noncomputable def votes_for_candidate_a : ℝ := valid_votes * percentage_candidate_a

theorem votes_for_candidate_a_correct :
  votes_for_candidate_a = 396000 := by
  sorry

end votes_for_candidate_a_correct_l75_75944


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75649

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75649


namespace four_digit_number_count_l75_75427

theorem four_digit_number_count : 
  ∀ (digits : list ℕ), 
    digits = [2, 0, 2, 5] → 
    (∀ n ∈ digits, n ≠ 0 → n = 2 ∨ n = 5) → 
    ∃! (count : ℕ), count = 6 :=
by
  sorry

end four_digit_number_count_l75_75427


namespace sameTypeTerm_l75_75811

variable (a b : ℝ) -- Assume a and b are real numbers 

-- Definitions for each term in the conditions
def term1 : ℝ := 2 * a * b^2
def term2 : ℝ := -a^2 * b
def term3 : ℝ := -2 * a * b
def term4 : ℝ := 5 * a^2

-- The term we are comparing against
def compareTerm : ℝ := 3 * a^2 * b

-- The condition we want to prove
theorem sameTypeTerm : term2 = compareTerm :=
  sorry


end sameTypeTerm_l75_75811


namespace sum_of_integers_whose_squares_end_with_25_l75_75713

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75713


namespace triangle_ratio_l75_75532

theorem triangle_ratio (A B C D : Type) [RealVectorSpace A] 
  [RealVectorSpace B] [RealVectorSpace C] [RealVectorSpace D] 
  (h1 : angle B = 90) 
  (h2 : dist A D = dist D C) 
  (h3 : dist B D = dist B C) :
  dist A B / dist B C = sqrt 3 := 
  sorry

end triangle_ratio_l75_75532


namespace sum_of_squares_ends_in_25_is_644_l75_75693

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75693


namespace sum_of_first_n_terms_l75_75916

theorem sum_of_first_n_terms (n : ℕ) (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h1 : ∀ n, a n = 2 / (n * (n + 1))) 
  (h2 : S n = ∑ i in finset.range (n + 1), a i) : 
  S n = 2 * n / (n + 1) :=
sorry

end sum_of_first_n_terms_l75_75916


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75744

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75744


namespace largest_int_n_binom_l75_75177

theorem largest_int_n_binom (
  h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n
) : n = 7 := by
  sorry

end largest_int_n_binom_l75_75177


namespace EventB_is_random_l75_75753

-- Define the events A, B, C, and D as propositions
def EventA : Prop := ∀ (x : ℕ), true -- A coin thrown will fall due to gravity (certain event)
def EventB : Prop := ∃ (n : ℕ), n > 0 -- Hitting the target with a score of 10 points (random event)
def EventC : Prop := ∀ (x : ℕ), true -- The sun rises from the east (certain event)
def EventD : Prop := ∀ (x : ℕ), false -- Horse runs at 70 meters per second (impossible event)

-- Prove that EventB is random, we can use a custom predicate for random events
def is_random_event (e : Prop) : Prop := (∃ (n : ℕ), n > 1) ∧ ¬ ∀ (x : ℕ), e

-- Main statement
theorem EventB_is_random :
  is_random_event EventB :=
by sorry -- The proof will be written here

end EventB_is_random_l75_75753


namespace solve_for_x_l75_75573

theorem solve_for_x : (2 / 5 : ℚ) - (1 / 7) = 1 / (35 / 9) :=
by
  sorry

end solve_for_x_l75_75573


namespace evaluate_expression_l75_75340

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l75_75340


namespace least_n_divisible_by_25_and_7_l75_75767

theorem least_n_divisible_by_25_and_7 (n : ℕ) (h1 : n > 1) (h2 : n % 25 = 1) (h3 : n % 7 = 1) : n = 126 :=
by
  sorry

end least_n_divisible_by_25_and_7_l75_75767


namespace smallest_positive_x_max_value_l75_75313

theorem smallest_positive_x_max_value : 
  ∃ x : ℝ, f(x) = (sin (x / 4) + sin (x / 7)) ∧ 
  0 < x ∧ 
  (∀ y : ℝ, f(y) = (sin (y / 4) + sin (y / 7)) → y ≤ x) ∧ 
  x = 17190 :=
sorry

end smallest_positive_x_max_value_l75_75313


namespace ratio_songs_kept_to_deleted_l75_75166

theorem ratio_songs_kept_to_deleted (initial_songs deleted_songs kept_songs : ℕ) 
  (h_initial : initial_songs = 54) (h_deleted : deleted_songs = 9) (h_kept : kept_songs = initial_songs - deleted_songs) :
  (kept_songs : ℚ) / (deleted_songs : ℚ) = 5 / 1 :=
by
  sorry

end ratio_songs_kept_to_deleted_l75_75166


namespace line_circle_intersection_l75_75594

theorem line_circle_intersection {k : ℝ} :
  (∃ M N : ℝ × ℝ, let x := M.1, y := M.2, x' := N.1, y' := N.2 in 
  y = k * x + 3 ∧ 
  y' = k * x' + 3 ∧ 
  (x - 2)^2 + (y - 3)^2 = 4 ∧ 
  (x' - 2)^2 + (y' - 3)^2 = 4 ∧ 
  (x - x')^2 + (y - y')^2 = 12) → 
  k = √3 / 3 ∨ k = -√3 / 3 :=
sorry

end line_circle_intersection_l75_75594


namespace least_number_to_add_l75_75356

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) : n = 228712 ∧ d = 30 ∧ r = n % d → d - r = 8 :=
begin
  intros h,
  cases h with hn hd,
  cases hd with hd hr,
  rw hn at hr,
  exact hr
end

end least_number_to_add_l75_75356


namespace combined_tax_rate_l75_75048

theorem combined_tax_rate (john_income ingrid_income samantha_income : ℝ) 
  (john_tax_rate ingrid_tax_rate samantha_tax_rate : ℝ) 
  (h_john_income : john_income = 56000)
  (h_ingrid_income : ingrid_income = 72000)
  (h_samantha_income : samantha_income = 45000)
  (h_john_tax_rate : john_tax_rate = 0.30)
  (h_ingrid_tax_rate : ingrid_tax_rate = 0.40)
  (h_samantha_tax_rate : samantha_tax_rate = 0.35) :
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income + 
  samantha_tax_rate * samantha_income) / 
  (john_income + ingrid_income + samantha_income) * 100 ≈ 35.46 := 
by 
  let combined_tax := (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income + samantha_tax_rate * samantha_income)
  let total_income := (john_income + ingrid_income + samantha_income)
  have h_combined_tax : combined_tax = 61350, sorry
  have h_total_income : total_income = 173000, sorry
  have h_combined_rate : (combined_tax / total_income) * 100 = 35.454795, sorry
  show (combined_tax / total_income) * 100 ≈ 35.46, from sorry

end combined_tax_rate_l75_75048


namespace kate_candy_l75_75302

variable (K : ℕ)
variable (R : ℕ) (B : ℕ) (M : ℕ)

-- Define the conditions
def robert_pieces := R = K + 2
def mary_pieces := M = R + 2
def bill_pieces := B = M - 6
def total_pieces := K + R + M + B = 20

-- The theorem to prove
theorem kate_candy :
  ∃ (K : ℕ), robert_pieces K R ∧ mary_pieces R M ∧ bill_pieces M B ∧ total_pieces K R M B ∧ K = 4 :=
sorry

end kate_candy_l75_75302


namespace least_positive_integer_to_multiple_of_5_l75_75228

theorem least_positive_integer_to_multiple_of_5 (n : ℕ) (h₁ : n = 725) :
  ∃ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 ∧ ∀ k : ℕ, (k > 0 ∧ (725 + k) % 5 = 0) → m ≤ k :=
begin
  use 5,
  sorry
end

end least_positive_integer_to_multiple_of_5_l75_75228


namespace total_sum_lent_l75_75278

theorem total_sum_lent (x : ℝ) (h₁ : 1656 * 5 / 100 * 3 = x * 3 / 100 * 8) : x + 1656 = 2691 :=
by
  have : 1656 * 15 = x * 24 := by
    calc
      1656 * 15 = 1656 * 5 * 3 : by ring
      ... = (1656 * 5 * 3 / 100) * 100 : by field_simp
      ... = (x * 3 * 8 / 100) * 100 : by rw [h₁]
      ... = x * 24 : by ring
  calc
    x + 1656
        = 1656 * 15 / 24 + 1656 : by rw [this]; field_simp
    ... = 24840 / 24 + 1656 : by ring
    ... = 1035 + 1656 : by norm_num
    ... = 2691 : by norm_num

end total_sum_lent_l75_75278


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75732

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75732


namespace complex_multiplication_in_a_plus_bi_form_l75_75348

theorem complex_multiplication_in_a_plus_bi_form : (3 - 4*complex.I) * (-4 + 6*complex.I) = 12 + 34*complex.I :=
by sorry

end complex_multiplication_in_a_plus_bi_form_l75_75348


namespace dilation_0_minus_2i_to_neg3_minus_14i_l75_75122

open Complex

def dilation_centered (z_center z zk : ℂ) (factor : ℝ) : ℂ :=
  z_center + factor * (zk - z_center)

theorem dilation_0_minus_2i_to_neg3_minus_14i :
  dilation_centered (1 + 2 * I) (0 - 2 * I) (1 + 2 * I) 4 = -3 - 14 * I :=
by
  sorry

end dilation_0_minus_2i_to_neg3_minus_14i_l75_75122


namespace geometric_sequence_a4_l75_75501

theorem geometric_sequence_a4 :
    ∀ (a : ℕ → ℝ) (n : ℕ), 
    a 1 = 2 → 
    (∀ n : ℕ, a (n + 1) = 3 * a n) → 
    a 4 = 54 :=
by
  sorry

end geometric_sequence_a4_l75_75501


namespace tan_pi_over_7_eq_cos_sin_form_l75_75843

theorem tan_pi_over_7_eq_cos_sin_form (n : ℤ) (h : 0 ≤ n ∧ n ≤ 13) :
  (n = 5) ↔
  (complex.exp (complex.I * (π / 7)) = complex.cos ((2 * n * π) / 14) + complex.I * complex.sin ((2 * n * π) / 14)) :=
by sorry

end tan_pi_over_7_eq_cos_sin_form_l75_75843


namespace sum_of_squares_ending_in_25_l75_75723

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75723


namespace triangle_isosceles_l75_75442

variable {A B C : ℝ} 

theorem triangle_isosceles (h: 2 * cos B * sin A = sin C) : A = B ∨ B = C ∨ C = A := 
sorry

end triangle_isosceles_l75_75442


namespace AM2_plus_CN2_eq_MN2_l75_75453

open_locale classical

-- Definitions of points and properties for the right-angled triangle ABC
variables {A B C M N O : Type*}
variables [noncomputable_space A] [noncomputable_space B] [noncomputable_space C]
variables [noncomputable_space M] [noncomputable_space N] [noncomputable_space O]

-- Defining right-angled triangle ABC with O as the midpoint of AC
structure RightAngledTriangle (A B C O : Type*) :=
  (right_angle : angle A B C = 90)
  (midpoint : midpoint A C O)

-- Point M is on segment AB, N is on segment BC, and MON forms a right angle
variables (triangle : RightAngledTriangle A B C O)
variables (M_on_AB : segment A B M)
variables (N_on_BC : segment B C N)
variables (right_angle_MON : angle M O N = 90)

-- The theorem to be proved
theorem AM2_plus_CN2_eq_MN2
  (h1 : triangle.right_angle)
  (h2 : triangle.midpoint)
  (h3 : M_on_AB)
  (h4 : N_on_BC)
  (h5 : right_angle_MON) :
  (dist A M)^2 + (dist C N)^2 = (dist M N)^2 :=
sorry

end AM2_plus_CN2_eq_MN2_l75_75453


namespace circular_garden_area_l75_75273

theorem circular_garden_area (w : ℝ) (h1 : 2 * (2 * w - 3) + 2 * w = 84) : 
  let r := w / 2 in
  let A := π * r^2 in
  A = 56.25 * π :=
by 
  sorry

end circular_garden_area_l75_75273


namespace sum_of_two_digit_numbers_ending_in_25_l75_75661

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75661


namespace find_angle_C_l75_75935

noncomputable def angle_C (b c B : ℝ) : ℝ :=
  if h : b = sqrt 2 ∧ c = 1 ∧ B = (45 : ℝ) then 30 else 0

theorem find_angle_C : ∀ (b c B : ℝ), b = sqrt 2 → c = 1 → B = (45 : ℝ) → angle_C b c B = 30 := by
  -- Place proof here
  intros b c B hb hc hB
  have h : b = sqrt 2 ∧ c = 1 ∧ B = (45 : ℝ) := ⟨hb, hc, hB⟩
  rw [angle_C, dif_pos h]
  exact rfl

-- This line is to suppress a warning about angle_C being noncomputable
#lint except noncomputable-definition def

end find_angle_C_l75_75935


namespace sum_of_squares_ending_in_25_l75_75721

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75721


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75738

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75738


namespace largest_int_n_binom_l75_75181

theorem largest_int_n_binom (
  h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n
) : n = 7 := by
  sorry

end largest_int_n_binom_l75_75181


namespace functional_eq_constant_l75_75098

theorem functional_eq_constant (f : ℝ → ℝ) 
  (h_cont : continuous f)
  (h_eq : ∀ x y : ℝ, (0 < x ∧ 0 < y) → 
    (f(x) - f(y)) * (f((x + y) / 2) - f(Real.sqrt (x * y)) = 0)) :
  ∃ c : ℝ, ∀ x : ℝ, 0 < x → f(x) = c := 
by
  sorry

end functional_eq_constant_l75_75098


namespace points_not_all_odd_distance_l75_75570

open Real

theorem points_not_all_odd_distance (p : Fin 4 → ℝ × ℝ) : ∃ i j : Fin 4, i ≠ j ∧ ¬ Odd (dist (p i) (p j)) := 
by
  sorry

end points_not_all_odd_distance_l75_75570


namespace car_miles_per_tankful_on_highway_l75_75785

theorem car_miles_per_tankful_on_highway
  (miles_city : ℕ)
  (diff_mpg : ℕ)
  (mpg_city : ℕ)
  (mpg_highway : ℕ)
  (tank_size : ℤ)
  (miles_highway : ℕ)
  (h1 : miles_city = 336)
  (h2 : diff_mpg = 6)
  (h3 : mpg_city = 9)
  (h4 : mpg_highway = mpg_city + diff_mpg)
  (h5 : tank_size = miles_city / mpg_city)
  (h6 : miles_highway = mpg_highway * (tank_size : ℕ)) :
  miles_highway = 560 :=
by 
  sorry

end car_miles_per_tankful_on_highway_l75_75785


namespace minimum_bird_species_l75_75458

theorem minimum_bird_species (total_birds : ℕ) (h : total_birds = 2021) :
  ∃ (min_species : ℕ), min_species = 1011 ∧ 
  (∀ (species_array : array total_birds ℕ),
   ∀ i j : fin total_birds, 
   species_array[i] = species_array[j] → ((i ≠ j) →
   (abs (i - j) mod 2 = 0))) :=
sorry

end minimum_bird_species_l75_75458


namespace annual_population_increase_l75_75581

theorem annual_population_increase (P A : ℝ) (t n : ℕ) (r : ℝ) 
  (hP : P = 10000)
  (hA : A = 12100)
  (ht : t = 2)
  (hn : n = 1)
  (h_eq : A = P * (1 + r / n) ^ (n * t)) :
  r = 0.1 :=
by
  have h1 : 12100 = 10000 * (1 + r)^2,
  { rw [hP, hA, ht, hn] at h_eq,
    exact h_eq, },
  have h2 : 1.21 = (1 + r)^2,
  { rw ← (div_eq_iff (10000 : ℝ).ne_zero) at h1,
    exact (eq_div_iff (10000 : ℝ).ne_zero).mp h1, },
  have h3 : 1 + r = real.sqrt 1.21,
  { exact eq_of_pow_eq_pow 2 (sqrt_nonneg _) h2 (sqrt_nonneg _), },
  have h4 : real.sqrt 1.21 = 1.1,
  { sorry, }, -- sqrt computation
  have h5 : 1 + r = 1.1,
  { exact h4.symm ▸ h3, },
  exact (add_eq_iff_eq_sub'.mp h5).symm

end annual_population_increase_l75_75581


namespace find_m_and_tan_alpha_l75_75380

theorem find_m_and_tan_alpha (m : ℝ) (h1 : cos α = -1/3) (h2 : α = P m 1) :
  m = -sqrt 2 / 4 ∧ tan α = 2 * sqrt 2 := by
  sorry

end find_m_and_tan_alpha_l75_75380


namespace slope_range_l75_75891

variables (x y k : ℝ)

theorem slope_range :
  (2 ≤ x ∧ x ≤ 3) ∧ (y = -2 * x + 8) ∧ (k = -3 * y / (2 * x)) →
  -3 ≤ k ∧ k ≤ -1 :=
by
  sorry

end slope_range_l75_75891


namespace find_p_l75_75165

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_p (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (h1 : p + q = r + 2) (h2 : 1 < p) (h3 : p < q) :
  p = 2 := 
sorry

end find_p_l75_75165


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75734

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75734


namespace quadratic_has_distinct_real_roots_l75_75143

theorem quadratic_has_distinct_real_roots :
  let a := 3
  let b := -2
  let c := -1
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  let a := 3
  let b := -2
  let c := -1
  let Δ := b^2 - 4 * a * c
  have discriminant : Δ = b^2 - 4 * a * c := rfl
  rw discriminant
  dsimp [a, b, c]
  linarith

end quadratic_has_distinct_real_roots_l75_75143


namespace find_a2017_l75_75998

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (a1 : ℝ := -2017)
variable (d : ℝ := 2)

-- Define the arithmetic sequence and sum of the first n terms
def arithmetic_seq (n : ℕ) : ℝ := a1 + (n - 1) * d
def sum_arithmetic_seq (n : ℕ) : ℝ := n * a1 + (n * (n - 1)) / 2 * d

-- Given conditions
axiom h1 : a 1 = a1
axiom h2 : (S 2007) / 2007 - (S 2005) / 2005 = 2

-- Proof goal
theorem find_a2017 : a 2017 = 2015 := 
by
  -- Show steps of the proof
  sorry

end find_a2017_l75_75998


namespace charlotte_flour_cost_l75_75310

noncomputable def flour_cost 
  (flour_sugar_eggs_butter_cost blueberry_cost cherry_cost total_cost : ℝ)
  (blueberry_weight oz_per_lb blueberry_cost_per_container cherry_weight cherry_cost_per_bag : ℝ)
  (additional_cost : ℝ) : ℝ :=
  total_cost - (blueberry_cost + additional_cost)

theorem charlotte_flour_cost :
  flour_cost 2.5 13.5 14 18 3 16 2.25 4 14 2.5 = 2 :=
by
  unfold flour_cost
  sorry

end charlotte_flour_cost_l75_75310


namespace fuel_needed_to_empty_l75_75794

theorem fuel_needed_to_empty (x : ℝ) 
  (h1 : (3/4) * x - (1/3) * x = 15) :
  (1/3) * x = 12 :=
by 
-- Proving the result
sorry

end fuel_needed_to_empty_l75_75794


namespace example_theorem_l75_75109

noncomputable def sum_of_real_solutions (a b : ℝ) (ha : a > 1) (hb : b > 0): ℝ :=
  sqrt ((2 * a + 1 + sqrt (4 * a + 1 + 4 * b)) / 2)

theorem example_theorem (a b : ℝ) (ha : a > 1) (hb : b > 0) :
  ∑ x in (finset.filter (λ x, (sqrt (a - sqrt (a + b + x)) = x)) (finset.range 100)), x = sum_of_real_solutions a b ha hb := 
sorry

end example_theorem_l75_75109


namespace super_rare_snake_cost_multiple_l75_75509

noncomputable def price_of_regular_snake : ℕ := 250
noncomputable def total_money_obtained : ℕ := 2250
noncomputable def number_of_snakes : ℕ := 3
noncomputable def eggs_per_snake : ℕ := 2

theorem super_rare_snake_cost_multiple :
  (total_money_obtained - (number_of_snakes * eggs_per_snake - 1) * price_of_regular_snake) / price_of_regular_snake = 4 :=
by
  sorry

end super_rare_snake_cost_multiple_l75_75509


namespace complex_addition_l75_75907

variable (a b : ℝ) (i : ℂ)

theorem complex_addition :
  (1 + 2 * complex.i) * complex.i = a + b * complex.i →
  a + b = -1 := by
  assume h: (1 + 2 * complex.i) * complex.i = a + b * complex.i
  have h1 : ((1 + 2 * complex.i) * complex.i).re = a := sorry -- Implicitly uses that re of (1 * complex.i + 2 * (complex.i * complex.i)) = a
  have h2 : ((1 + 2 * complex.i) * complex.i).im = b := sorry -- Implicitly uses that im of (1 * complex.i + 2 * (complex.i * complex.i)) = b
  exact sorry

end complex_addition_l75_75907


namespace parametric_number_of_pyramid_image_l75_75547

/-- Given a triangular pyramid where one of the lateral faces is perpendicular to the base
    plane and both the lateral face and the base are equilateral triangles, 
    prove that the parametric number of the image of the pyramid 
    using quadrilateral SABC is 5. -/
theorem parametric_number_of_pyramid_image 
  (pyramid : Type)
  (base : pyramid → set (euclidean_space ℝ 3))
  (lateral_face : pyramid → set (euclidean_space ℝ 3))
  (S A B C : euclidean_space ℝ 3)
  (is_equilateral_triangle : ∀ T : set (euclidean_space ℝ 3), T ∈ {base, lateral_face} → equilateral_triangle T) 
  (is_perpendicular : latera_face → base → Prop)
  (is_perpendicular (lateral_face pyramid) (base pyramid))
  (image_quad  : set (euclidean_space ℝ 3)) :
  parametric_number (image_quad (S, A, B, C)) = 5 :=
sorry

end parametric_number_of_pyramid_image_l75_75547


namespace ninth_term_arithmetic_sequence_l75_75455

theorem ninth_term_arithmetic_sequence :
  ∃ (a d : ℤ), (a + 2 * d = 5 ∧ a + 5 * d = 17) ∧ (a + 8 * d = 29) := 
by
  sorry

end ninth_term_arithmetic_sequence_l75_75455


namespace geometric_sequence_a4_value_l75_75929

theorem geometric_sequence_a4_value 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h1 : a 1 + (2 / 3) * a 2 = 3) 
  (h2 : a 4^2 = (1 / 9) * a 3 * a 7) 
  :
  a 4 = 27 :=
sorry

end geometric_sequence_a4_value_l75_75929


namespace hyperbola_a_solution_l75_75917

noncomputable def hyperbola_a_value (a : ℝ) : Prop :=
  (a > 0) ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / 2) = 1) ∧ (∃ e : ℝ, e = 2)

theorem hyperbola_a_solution : ∃ a : ℝ, hyperbola_a_value a ∧ a = (Real.sqrt 6) / 3 :=
  by
    sorry

end hyperbola_a_solution_l75_75917


namespace probability_of_digit_2_in_3_div_11_eq_one_half_l75_75092

theorem probability_of_digit_2_in_3_div_11_eq_one_half :
  let decimal_rep := "0.272727..."
  let repeating_block := "27"
  let block_length := String.length repeating_block
  let num_of_2s := String.count repeating_block '2'
  ∃ n, decimal_rep = "0." ++ String.repeat repeating_block n ∧
  (num_of_2s / block_length : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_digit_2_in_3_div_11_eq_one_half_l75_75092


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75737

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75737


namespace bird_species_min_l75_75488

theorem bird_species_min (total_birds : ℕ) (h_total_birds : total_birds = 2021)
  (h_even_between : ∀ (species : Sort*) (a b : species), (a ≠ b) → even (nat.dist a b)) :
  ∃ species_num : ℕ, species_num = 1011 :=
by
  sorry

end bird_species_min_l75_75488


namespace smallest_positive_value_l75_75321

-- Define the expressions
def exprA : ℝ := 20 - 4 * Real.sqrt 14
def exprB : ℝ := 4 * Real.sqrt 14 - 20
def exprC : ℝ := 25 - 6 * Real.sqrt 15
def exprD : ℝ := 60 - 12 * Real.sqrt 30
def exprE : ℝ := 12 * Real.sqrt 30 - 60

-- Statement of the theorem
theorem smallest_positive_value :
  exprC = 25 - 6 * Real.sqrt 15 ∧
  (∀ x ∈ {exprA, exprB, exprC, exprD, exprE}, 0 < x → 25 - 6 * Real.sqrt 15 ≤ x) :=
sorry

end smallest_positive_value_l75_75321


namespace b_must_be_one_l75_75155

theorem b_must_be_one (a b : ℝ) (h1 : a + b - a * b = 1) (h2 : ∀ n : ℤ, a ≠ n) : b = 1 :=
sorry

end b_must_be_one_l75_75155


namespace roots_sum_l75_75533

theorem roots_sum (p q r : ℝ) (h1 : ∀ x : ℝ, Polynomial.eval x (Polynomial.Coeff 3 - 4 * Polynomial.Coeff 2 + Polynomial.Coeff 1 + Polynomial.Coeff 0 = 0)) :
  (p / (q * r + 1) + q / (p * r + 1) + r / (p * q + 1)) = -8.43 := 
sorry

end roots_sum_l75_75533


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75646

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75646


namespace min_num_groups_l75_75135

theorem min_num_groups : 
  (∀ g ∈ groups, (∀ a b ∈ g, (a + b) % 6 = 0)) → 
  ∃ g₀ g₃ rest, 
    g₀ ∪ g₃ ∪ rest = (finset.range 600).filter (λ n, n + 1) ∧ 
    (∀ a b ∈ g₀, (a + b) % 6 = 0) ∧ 
    (∀ a b ∈ g₃, (a + b) % 6 = 0) ∧ 
    (∀ g ∈ rest, (∀ a b ∈ g, (a + b) % 6 = 0) ∧ |g| = 2) ∧
    g₀.card = (600 / 6) ∧ 
    g₃.card = (600 / 6) ∧ 
    rest.card = 200 * 2 → 
    g₀.card + g₃.card + rest.card / 2 = 202 :=
by sorry

end min_num_groups_l75_75135


namespace non_invited_classmates_eq_l75_75027

variable (students : Finset ℕ) -- Represents the set of all students as numbers from 0 to 24
variable (direct_friends_of_lucas : Finset ℕ) -- Represents Lucas's direct friends as indices
variable (common_course_friends : Finset (ℕ × ℕ)) -- Represents pairs of students who share at least two common courses

-- Total number of students in the class
def total_students : ℕ := students.card

-- Number of direct friends Lucas has
def lucas_direct_friends_count : ℕ := direct_friends_of_lucas.card

-- Number of friends through shared courses each of Lucas's friends has
def friends_through_courses_count : ℕ := 
  direct_friends_of_lucas.sum (λ friend, (common_course_friends.filter (λ pair, pair.1 = friend ∨ pair.2 = friend)).card)

-- Total number of unique invited classmates
def total_invited : ℕ :=
  (direct_friends_of_lucas ∪ (Finset.bUnion direct_friends_of_lucas
    (λ friend, common_course_friends.filter (λ pair, pair.1 = friend ∨ pair.2 = friend).image Prod.fst ∪
    common_course_friends.filter (λ pair, pair.1 = friend ∨ pair.2 = friend).image Prod.snd))).card

-- Final count of non-invited classmates
def non_invited_classmates (total_students : ℕ) (total_invited : ℕ) : ℕ :=
  total_students - total_invited

theorem non_invited_classmates_eq :
  total_students students = 25 → 
  lucas_direct_friends_count direct_friends_of_lucas = 5 → 
  (friends_through_courses_count students direct_friends_of_lucas common_course_friends) = 10 → 
  non_invited_classmates 25 total_invited = 10 :=
by
  intros
  sorry -- Proof to be completed


end non_invited_classmates_eq_l75_75027


namespace complex_problem_l75_75017

noncomputable def z : ℂ := 4 - 3 * complex.i

theorem complex_problem :
  (z * complex.conj(z)) / complex.abs(z) = 5 := sorry

end complex_problem_l75_75017


namespace sum_of_squares_ends_in_25_is_644_l75_75692

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75692


namespace lcm_of_2_4_5_6_l75_75231

theorem lcm_of_2_4_5_6 : Nat.lcm (Nat.lcm (Nat.lcm 2 4) 5) 6 = 60 :=
by
  sorry

end lcm_of_2_4_5_6_l75_75231


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75642

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75642


namespace half_product_two_consecutive_integers_mod_3_l75_75099

theorem half_product_two_consecutive_integers_mod_3 (A : ℤ) : 
  (A * (A + 1) / 2) % 3 = 0 ∨ (A * (A + 1) / 2) % 3 = 1 :=
sorry

end half_product_two_consecutive_integers_mod_3_l75_75099


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75656

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75656


namespace sum_of_squares_of_roots_l75_75827

theorem sum_of_squares_of_roots :
  let r1, r2, r3, r4 := roots_of_polynomial x^4 + 6 * x^3 + 11 * x^2 + 6 * x + 1 in
  (r1 + r2 + r3 + r4 = -6) →
  (r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4 = 11) →
  (r1^2 + r2^2 + r3^2 + r4^2 = 14) :=
by
  sorry

end sum_of_squares_of_roots_l75_75827


namespace stratified_sampling_of_students_l75_75796

theorem stratified_sampling_of_students
  (total_students: ℕ)
  (num_freshmen: ℕ)
  (num_sophomores: ℕ)
  (num_juniors: ℕ)
  (sample_size: ℕ)
  (stratified_sampling : sample_size = 50 ∧ total_students = 2000 ∧ num_freshmen = 800 ∧ num_sophomores = 600 ∧ num_juniors = 600) :
  (num_freshmen_sampled: ℕ × num_sophomores_sampled: ℕ × num_juniors_sampled: ℕ) = (20, 15, 15) :=
sorry

end stratified_sampling_of_students_l75_75796


namespace evaluate_expression_l75_75339

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l75_75339


namespace p100_one_l75_75140

noncomputable def p : ℕ → (ℝ → ℝ)
| 0     := λ x, 1
| (n+1) := by
  let pn1 := λ n x, (n + 1) * p n (x + 1)
  let p := λ n x, (0 : ℝ) -- assume this exists as a placeholder
  sorry  -- should define a proper integral that fit this place

theorem p100_one :
  let p_100_1 := p 100 1 in
  (p_100_1 = 101 ^ 99) :=
by sorry

end p100_one_l75_75140


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75650

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75650


namespace sum_of_squares_ending_in_25_l75_75722

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75722


namespace sequence_strictly_decreasing_l75_75295

noncomputable theory
open_locale classical

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, (n ≥ 2018) → (∃ x, P n x = 0 ∧ a (n + 1) = x ∧ (∀ y, P n y = 0 → x ≤ y))

def P (n : ℕ) (x : ℝ) : ℝ :=
  (x ^ (2 * n)) + (∑ i in finset.range n, a i * (x ^ (2 * (n - i) - 2)))

theorem sequence_strictly_decreasing (a : ℕ → ℝ) (h : sequence a) :
  ∃ N, ∀ n, (n ≥ N) → (a (n + 1) < a n) :=
begin
  use 2019,
  intros n hn,
  specialize h (n - 1),
  have hn' : n - 1 ≥ 2018, from nat.sub_le_right_iff_le_add_one.2 hn,
  specialize h hn',
  obtain ⟨x, hx₁, hx₂, hx₃⟩ := h,
  have h₄ := hx₃ (a n),
  rw hx₂ at h₄,
  exact h₄ (hx₁.mpr rfl).symm,
end

end sequence_strictly_decreasing_l75_75295


namespace total_savings_l75_75512

theorem total_savings :
  let josiah_daily := 0.25 
  let josiah_days := 24 
  let leah_daily := 0.50 
  let leah_days := 20 
  let megan_multiplier := 2
  let megan_days := 12 
  let josiah_savings := josiah_daily * josiah_days 
  let leah_savings := leah_daily * leah_days 
  let megan_daily := megan_multiplier * leah_daily 
  let megan_savings := megan_daily * megan_days 
  let total_savings := josiah_savings + leah_savings + megan_savings 
  total_savings = 28 :=
by
  sorry

end total_savings_l75_75512


namespace triangle_IJK_right_angle_at_I_l75_75071

-- Lean definitions of the geometric concepts.
variables {P Q R I J K : Type*} [EuclideanGeometry P Q R I J K]

-- Main theorem statement
theorem triangle_IJK_right_angle_at_I
  (ABC_triangle : Triangle ABC)
  (Gamma_incircle : Incircle ABC)
  (GammaA_excircle : Excircle ABC A)
  (touch_incircle_BC_I : Gamma_incircle.Touches(BC, I))
  (touch_excircle_BC_J : GammaA_excircle.Touches(BC, J))
  (intersection_incircle_AJ_K : Intersects_AJ_Closest(ABC, Gamma_incircle, AJ, K, A)) :
  RightAngle (I J K) :=
sorry

end triangle_IJK_right_angle_at_I_l75_75071


namespace square_side_length_distance_l75_75516

theorem square_side_length_distance (S : square) (side_length : ℝ) (h_S : side_length = 2) (a b c : ℕ) (h_gcd : Int.gcd a b = 1 ∧ Int.gcd b c = 1 ∧ Int.gcd a c = 1):
  (a = 16) -> (b = 1) -> (c = 16) -> 
  probability_distance (S, side_length, 2) (≥ sqrt 2 / 2) = (a - b * pi) / c :=
by
  sorry

end square_side_length_distance_l75_75516


namespace sum_of_two_digit_numbers_ending_in_25_l75_75634

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75634


namespace symmetry_of_circle_l75_75906

theorem symmetry_of_circle {b : ℝ} :
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 1 → (y = x + b)) ↔ b = 1 :=
begin
  sorry,
end

end symmetry_of_circle_l75_75906


namespace weight_per_trombone_player_l75_75033

theorem weight_per_trombone_player:
  ∀ (wt_trumpet wt_clarinet wt_tuba wt_drum players_trumpet players_clarinet players_trombone players_tuba players_drum total_weight : ℕ),
    wt_trumpet = 5 →
    wt_clarinet = 5 →
    wt_tuba = 20 →
    wt_drum = 15 →
    players_trumpet = 6 →
    players_clarinet = 9 →
    players_trombone = 8 →
    players_tuba = 3 →
    players_drum = 2 →
    total_weight = 245 →
    wt_trumpet * players_trumpet + wt_clarinet * players_clarinet + wt_tuba * players_tuba + wt_drum * players_drum +
    (players_trombone * wt_trombone) = total_weight →
    wt_trombone = 10 := 
by
  intros 
      wt_trumpet wt_clarinet wt_tuba wt_drum 
      players_trumpet players_clarinet players_trombone 
      players_tuba players_drum total_weight 
      h1 h2 h3 h4 h5 h6 h7 h8 h9 h10,
  sorry

end weight_per_trombone_player_l75_75033


namespace man_salary_problem_l75_75267

-- Define the problem in Lean 4
theorem man_salary_problem (S : ℝ) :
  (1/3 * S) + (1/4 * S) + (1/5 * S) + 1760 = S → 
  S = 8123.08 :=
sorry

end man_salary_problem_l75_75267


namespace least_positive_integer_to_multiple_of_5_l75_75217

theorem least_positive_integer_to_multiple_of_5 : ∃ (n : ℕ), n > 0 ∧ (725 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m :=
by
  sorry

end least_positive_integer_to_multiple_of_5_l75_75217


namespace bird_species_min_l75_75491

theorem bird_species_min (total_birds : ℕ) (h_total_birds : total_birds = 2021)
  (h_even_between : ∀ (species : Sort*) (a b : species), (a ≠ b) → even (nat.dist a b)) :
  ∃ species_num : ℕ, species_num = 1011 :=
by
  sorry

end bird_species_min_l75_75491


namespace find_x_l75_75861

theorem find_x 
  (x : ℝ)
  (h : 3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * x * 0.5)) = 2800.0000000000005) : 
  x = 0.225 := 
sorry

end find_x_l75_75861


namespace evaluate_expression_l75_75342

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l75_75342


namespace minimum_species_count_l75_75464

theorem minimum_species_count {n : ℕ} (h_n : n = 2021) 
  (h_cond : ∀ i j k : ℕ, i < j ∧ j < k → 
    birds i = birds k → birds j ≠ birds i → (j - i - 1) % 2 = 1 ∧ (k - j - 1) % 2 = 1) : 
  ∃ s : ℕ, s ≥ 1011 :=
begin
  sorry
end

end minimum_species_count_l75_75464


namespace find_n_modulo_l75_75354

theorem find_n_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 11) (h3 : n ≡ 15827 [ZMOD 12]) : n = 11 :=
by
  sorry

end find_n_modulo_l75_75354


namespace number_of_solutions_l75_75854

theorem number_of_solutions (x y : ℕ) : (3 * x + 2 * y = 1001) → ∃! (n : ℕ), n = 167 := by
  sorry

end number_of_solutions_l75_75854


namespace num_subsets_sum_not_11_l75_75926

theorem num_subsets_sum_not_11 : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let valid_subset (sub : Finset ℕ) : Prop := 
    sub.card = 5 ∧ ∀ x y ∈ sub, x ≠ y → x + y ≠ 11
  (Finset.filter valid_subset (Finset.powerset S)).card = 32 :=
sorry

end num_subsets_sum_not_11_l75_75926


namespace swimming_pool_volume_l75_75004

def radius (diameter : ℝ) : ℝ := diameter / 2

def volume_of_cylinder (r : ℝ) (h : ℝ) : ℝ := π * r^2 * h

theorem swimming_pool_volume :
  let diameter := 20
  let depth := 5
  let r := radius diameter
  volume_of_cylinder r depth = 500 * π :=
by
  sorry -- Proof will be provided here

end swimming_pool_volume_l75_75004


namespace stella_toilet_paper_packs_l75_75094

-- Define the relevant constants/conditions
def rolls_per_bathroom_per_day : Nat := 1
def number_of_bathrooms : Nat := 6
def days_per_week : Nat := 7
def weeks : Nat := 4
def rolls_per_pack : Nat := 12

-- Theorem statement
theorem stella_toilet_paper_packs :
  (rolls_per_bathroom_per_day * number_of_bathrooms * days_per_week * weeks) / rolls_per_pack = 14 :=
by
  sorry

end stella_toilet_paper_packs_l75_75094


namespace math_proof_l75_75895
noncomputable theory

-- Definition of given problem
variables (α : ℝ)

-- Condition 1: α is an acute angle
def is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2 

-- Condition 2: given tan(π/4 + α) = 2
def tan_condition (α : ℝ) : Prop := Real.tan (π / 4 + α) = 2

-- Question 1: tan α = 1/3
def tan_alpha (α : ℝ) : ℝ := Real.tan α

-- Question 2: The given expression
def complex_expr (α : ℝ) : ℝ := 
  (√2 * Real.sin (2 * α + π / 4) * Real.cos α - Real.sin α) / Real.cos (2 * α)

-- The main proof problem
theorem math_proof : is_acute α ∧ tan_condition α → tan_alpha α = 1 / 3 ∧ complex_expr α = (2 / 5) * √10 := by
  sorry

end math_proof_l75_75895


namespace increasing_denominator_keeps_fraction_unchanged_l75_75611

theorem increasing_denominator_keeps_fraction_unchanged :
  ∀ (a b increase_numerator increase_denominator : ℕ),
    let new_numerator := a + increase_numerator in
    let new_denominator := b + increase_denominator in
    a = 3 →
    b = 16 →
    increase_numerator = 6 →
    increase_denominator = 32 →
    (a : ℚ) / b = (new_numerator : ℚ) / new_denominator :=
begin
  intros a b increase_numerator increase_denominator,
  let new_numerator := a + increase_numerator,
  let new_denominator := b + increase_denominator,
  assume ha hb hin hincr,
  rw [ha, hb, hin, hincr],
  norm_cast,
  simp,
end

end increasing_denominator_keeps_fraction_unchanged_l75_75611


namespace sum_of_sequence_l75_75276

noncomputable def sequence : ℕ → ℚ
| 0     := 0
| 1     := 2
| 2     := 3
| (n+3) := (1/4) * (sequence (n+2)) + (1/5) * (sequence (n+1))

theorem sum_of_sequence : (∑' n : ℕ, sequence n) = 90 / 11 := by
  sorry

end sum_of_sequence_l75_75276


namespace cyclic_quadrilaterals_BEXF_DXFC_l75_75886

-- Define the given elements and conditions
variable {A B C M D E F X : Type}
variable [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
variable [Midpoint M B C]
variable [OnSegment D A C]
variable [OnSegment E A B]
variable (h1 : dist M B = dist M C)
variable (h2 : dist M E = dist M B)
variable (h3 : dist M D = dist M B)
variable [AngleBisector A B C F]
variable [AngleBisector M E D X]

-- Prove the quadrilaterals are cyclic
theorem cyclic_quadrilaterals_BEXF_DXFC : Cyclic B E X F ∧ Cyclic D X F C := sorry

end cyclic_quadrilaterals_BEXF_DXFC_l75_75886


namespace minimum_species_l75_75481

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l75_75481


namespace average_price_of_towels_l75_75280

theorem average_price_of_towels :
  let total_cost := 2350
  let total_towels := 10
  total_cost / total_towels = 235 :=
by
  sorry

end average_price_of_towels_l75_75280


namespace eval_expr_ceil_floor_l75_75337

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l75_75337


namespace circle_arrangement_at_least_factorial_l75_75088

open Finset

theorem circle_arrangement_at_least_factorial (n : ℕ) (n_pos : 0 < n)
  (a : Fin n → ℝ) (h₀ : (∑ i : Fin n, a i) = 0) (h₁ : ∀ i : Fin n, a i ≠ 0):
  ∃ σ : Perm (Fin n), (∑ i in range n, ∑ j in range (i + 1), a (σ (Fin.mk j sorry))) ≥ 0 :=
sorry

end circle_arrangement_at_least_factorial_l75_75088


namespace louie_last_match_goals_l75_75030

-- Define the conditions as given in the problem statement.
variables (L : ℕ) -- Number of goals Louie scored in the last match.

-- Conditions
axiom louie_previous_goals : 40 -- Louie scored 40 goals in previous matches.
axiom brother_games : 3 * 50 = 150 -- Louie's brother played 150 games.
axiom total_goals : 40 + L + 150 * (2 * L) = 1244 -- Total goals scored by both brothers is 1244.

-- Statement to prove
theorem louie_last_match_goals : L = 4 :=
by
  sorry -- Proof to be filled in.

end louie_last_match_goals_l75_75030


namespace grade_swap_l75_75593

theorem grade_swap (a b : ℕ) (h_a : 0 < a) (h_a_b : a < 100) (h_b : 0 < b) (h_b_a : b < 100) :
  ∃ (scale_factor : ℕ), true := 
begin
  -- Define the original grades as real numbers out of 100
  let grade_a := (a : ℝ) / 100,
  let grade_b := (b : ℝ) / 100,

  -- Let the academic department change the grades such that Petya's grade becomes b and Vasya's grade becomes a
  sorry,
end 

end grade_swap_l75_75593


namespace solve_for_a_and_b_range_of_f_when_x_lt_zero_l75_75913

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1 + a * (2 ^ x)) / (2 ^ x + b)

theorem solve_for_a_and_b (a b : ℝ) :
  f a b 1 = 3 ∧
  f a b (-1) = -3 →
  a = 1 ∧ b = -1 :=
by
  sorry

theorem range_of_f_when_x_lt_zero (x : ℝ) :
  ∀ x < 0, f 1 (-1) x < -1 :=
by 
  sorry

end solve_for_a_and_b_range_of_f_when_x_lt_zero_l75_75913


namespace double_persons_half_work_l75_75766

theorem double_persons_half_work :
  (∀ (n : ℕ) (d : ℕ), d = 12 → (2 * n) * (d / 2) = n * 3) :=
by
  sorry

end double_persons_half_work_l75_75766


namespace large_screen_width_l75_75798

theorem large_screen_width
  (height width diagonal : ℝ) 
  (ratio : height / width = 3 / 4)
  (hypotenuse : real.sqrt (height^2 + width^2) = diagonal)
  (diagonal_value : diagonal = 36) : 
  width = 28.8 :=
by
  sorry

end large_screen_width_l75_75798


namespace geometric_series_sum_1_geometric_series_sum_2_geometric_series_sum_3_l75_75565

theorem geometric_series_sum_1 (n : ℕ) : 
  (∑ i in range (n+1), 2^i) = 2^(n+1) - 1 :=
by sorry

theorem geometric_series_sum_2 : 
  (∑ i in range 11, 2^i) = 2^11 - 1 :=
by sorry

theorem geometric_series_sum_3 : 
  (∑ i in range 2017, 3^i) = (3^2017 - 1) / 2 :=
by sorry

end geometric_series_sum_1_geometric_series_sum_2_geometric_series_sum_3_l75_75565


namespace a_n_formula_S_n_formula_l75_75982

-- Given Definitions as per conditions 
def geometric_sequence (N : ℕ) (seq : ℕ → ℝ) : Prop :=
  seq 0 = 1 ∧ seq (N + 1) = 100 ∧ ∀ n, n > 0 ∧ n < N + 1 → seq n = 1 * (100 ^ ((n:ℝ)/(N+1:ℝ)))

def product_terms (N : ℕ) (seq : ℕ → ℝ) : ℝ :=
  ∏ i in Finset.range (N+2), seq i

def a_n (N : ℕ) : ℝ := real.log10 (product_terms N (λ n => 1 * (100 ^ ((n:ℝ)/(N+1:ℝ)))))

def b_n (N : ℕ) : ℝ := (a_n N - 1) * 2^(N - 1)

def S_n (N : ℕ) : ℝ := ∑ k in Finset.range N, b_n k

-- Theorem Statements
theorem a_n_formula (N : ℕ) (h : N ≥ 1) : a_n N = N + 2 := sorry

theorem S_n_formula (N : ℕ) (h : N ≥ 1) : S_n N = -2 - (N + 1) * 2^N := sorry

end a_n_formula_S_n_formula_l75_75982


namespace quintuple_correct_l75_75392

noncomputable def quintuple (A' B' C' D' E' : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ :=
  (1/31, 1/31, 2/31, 4/31, 8/31)

theorem quintuple_correct (A' B' C' D' E' : ℝ) :
  ∃ (p q r s t : ℝ), 
    (p, q, r, s, t) = quintuple A' B' C' D' E' ∧ 
    ∀ (A B C D E : ℝ),
      B = 1/2 * A + 1/2 * A' →
      C = 1/2 * B + 1/2 * B' →
      D = 1/2 * C + 1/2 * C' →
      E = 1/2 * D + 1/2 * D' →
      E = p * A' + q * B' + r * C' + s * D' + t * E' :=
begin
  sorry
end

end quintuple_correct_l75_75392


namespace hyperbola_eccentricity_l75_75899

-- Define the context/conditions
noncomputable def hyperbola_vertex_to_asymptote_distance (a b e : ℝ) : Prop :=
  (2 = b / e)

noncomputable def hyperbola_focus_to_asymptote_distance (a b e : ℝ) : Prop :=
  (6 = b)

-- Define the main theorem to prove the eccentricity
theorem hyperbola_eccentricity (a b e : ℝ) (h1 : hyperbola_vertex_to_asymptote_distance a b e) (h2 : hyperbola_focus_to_asymptote_distance a b e) : 
  e = 3 := 
sorry 

end hyperbola_eccentricity_l75_75899


namespace largest_binom_coeff_l75_75182

theorem largest_binom_coeff (n : ℕ) :
  (nat.choose 10 3) + (nat.choose 10 4) = nat.choose 11 n → n = 7 :=
by sorry

end largest_binom_coeff_l75_75182


namespace sum_even_102_to_600_l75_75603

def sum_first_50_even : ℕ := 2550
def sum_even_602_to_700 : ℕ := 32550

theorem sum_even_102_to_600 : sum_even_602_to_700 - sum_first_50_even = 30000 :=
by
  -- The given sum of the first 50 positive even integers is 2550
  have h1 : sum_first_50_even = 2550 := by rfl
  
  -- The given sum of the even integers from 602 to 700 inclusive is 32550
  have h2 : sum_even_602_to_700 = 32550 := by rfl
  
  -- Therefore, the sum of the even integers from 102 to 600 is:
  have h3 : sum_even_602_to_700 - sum_first_50_even = 32550 - 2550 := by
    rw [h1, h2]
  
  -- Calculate the result
  exact h3

end sum_even_102_to_600_l75_75603


namespace sum_of_geometric_sequence_l75_75367

noncomputable def geometric_sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence (a₁ q : ℝ) (n : ℕ) 
  (h1 : a₁ + a₁ * q^3 = 10) 
  (h2 : a₁ * q + a₁ * q^4 = 20) : 
  geometric_sequence_sum a₁ q n = (10 / 9) * (2^n - 1) :=
by 
  sorry

end sum_of_geometric_sequence_l75_75367


namespace ratio_of_heights_l75_75078

def min_height := 140
def brother_height := 180
def grow_needed := 20

def mary_height := min_height - grow_needed
def height_ratio := mary_height / brother_height

theorem ratio_of_heights : height_ratio = (2 / 3) := 
  sorry

end ratio_of_heights_l75_75078


namespace toilet_paper_packs_needed_l75_75095

-- Definitions based on conditions
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def weeks : ℕ := 4
def rolls_per_pack : ℕ := 12
def daily_stock : ℕ := 1

-- The main theorem statement
theorem toilet_paper_packs_needed : 
  (bathrooms * days_per_week * weeks) / rolls_per_pack = 14 := by
sorry

end toilet_paper_packs_needed_l75_75095


namespace descending_order_a_b_c_l75_75870

noncomputable def a : ℝ := 0.4 ^ (-0.5)
noncomputable def b : ℝ := 0.5 ^ 0.5
noncomputable def c : ℝ := Real.log 2 / Real.log 0.2

theorem descending_order_a_b_c : a > b ∧ b > c := by
  have ha : a = 0.4 ^ (-0.5) := rfl
  have hb : b = 0.5 ^ 0.5 := rfl
  have hc : c = Real.log 2 / Real.log 0.2 := rfl
  sorry

end descending_order_a_b_c_l75_75870


namespace sum_of_bn_sequence_range_of_m_l75_75911

-- Problem 1: Sum of sequence b_n
theorem sum_of_bn_sequence (n : ℕ) (h : n > 0) : 
  let m := (Real.sqrt 2) / 2
  let f := (fun x : ℝ => log m x)
  let a (n : ℕ) := m^(2 * n)
  let b (n : ℕ) := a n * f (a n)
  Sn = (finset.range n).sum (λ k, b (k+1)) = - (n + 2) * (1/2)^(n-1) + 4 :=
sorry

-- Problem 2: Range of values for m
theorem range_of_m (m: ℝ) (n : ℕ) (a : ℝ) (ha : 0 < a) (h1: m*a = n) : 
  0 < m ∧ m < (Real.sqrt 2) / 2 :=
sorry

end sum_of_bn_sequence_range_of_m_l75_75911


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75748

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75748


namespace euler_complex_plane_l75_75326

theorem euler_complex_plane :
  let i : ℂ := complex.I in
  let z := complex.exp (8 * real.pi / 3 * i) in
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end euler_complex_plane_l75_75326


namespace four_digit_integers_with_thousands_digit_7_l75_75925

theorem four_digit_integers_with_thousands_digit_7 : 
  (number_of_integers : ℕ) 
  (h1 : ∀ n : ℕ, 7000 ≤ n ∧ n ≤ 7999 → ∃ k : ℕ, n = 7000 + k) 
  (h2 : ∀ a b c : ℕ, a ∈ finset.range 10 ∧ b ∈ finset.range 10 ∧ c ∈ finset.range 10
   → number_of_integers = 10 * 10 * 10) : 
  number_of_integers = 1000 := 
by 
  sorry

end four_digit_integers_with_thousands_digit_7_l75_75925


namespace sum_of_squares_ends_in_25_is_644_l75_75688

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75688


namespace intersection_point_on_line_centers_l75_75921

-- Define circles and centers
variables {O₁ O₂ : Type*} [metric_space O₁] [metric_space O₂]

-- Suppose there are two circles externally tangent to each other
variables (circle₁ : set O₁) (circle₂ : set O₂)
(h₀₁ : is_circle O₁ circle₁)
(h₀₂ : is_circle O₂ circle₂)
(tangent: externally_tangent O₁ O₂)

-- Suppose there are tangent lines
variables (A B : O₁) (C D : O₂)
(h₁ : is_tangent A circle₁)
(h₂ : is_tangent C circle₂)
(h₃ : internal_tangent B circle₁)
(h₄ : internal_tangent D circle₂)

-- Suppose we have defined the lines and intersections
variables (AB CD : set (O₁ × O₂))
(h_int_AB : intersects AB (line A B))
(h_int_CD : intersects CD (line C D))
(def_K : ∃ K, point_of_intersection AB CD K)

-- Goal: Prove the intersection point lies on the line joining the centers
theorem intersection_point_on_line_centers
: ∀ (K : O₁ × O₂),
  point_of_intersection AB CD K →
  lies_on_line K (line O₁ O₂) :=
sorry -- Proof is omitted

end intersection_point_on_line_centers_l75_75921


namespace largest_n_binom_sum_l75_75194

open Nat

theorem largest_n_binom_sum (n : ℕ) 
  (h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n) : n = 7 :=
by 
  sorry

end largest_n_binom_sum_l75_75194


namespace Randy_biscuits_l75_75560

theorem Randy_biscuits
  (biscuits_initial : ℕ)
  (father_gift : ℕ)
  (mother_gift : ℕ)
  (brother_eat : ℕ) :
  biscuits_initial = 32 →
  father_gift = 13 →
  mother_gift = 15 →
  brother_eat = 20 →
  biscuits_initial + father_gift + mother_gift - brother_eat = 40 :=
by
  intros h_initial h_father h_mother h_brother
  rw [h_initial, h_father, h_mother, h_brother]
  norm_num
  sorry

end Randy_biscuits_l75_75560


namespace max_theta_correct_l75_75896

noncomputable def max_theta : ℝ :=
  let A := (0, 2, 0)
  let B := (0, 0, 0)
  let D := (2, 0, 0)
  let D₁ := (2, 0, 2)
  let T := (x, y, z) in
  let θ := Real.arctan (2 / 2) in
  if sqrt ((x - 0)^2 + (y - 2)^2 + (z - 0)^2) = 2 * sqrt (x^2 + y^2 + z^2)
  then θ
  else 0

theorem max_theta_correct
  (T : ℝ × ℝ × ℝ)
  (hT: T ∈ {t | sqrt ((t.1 - 0)^2 + (t.2 - 2)^2 + (t.3 - 0)^2) = 2 * sqrt (t.1^2 + t.2^2 + t.3^2)})
  : max_theta = π / 4 :=
sorry

end max_theta_correct_l75_75896


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75672

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75672


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75694

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75694


namespace bird_species_min_l75_75487

theorem bird_species_min (total_birds : ℕ) (h_total_birds : total_birds = 2021)
  (h_even_between : ∀ (species : Sort*) (a b : species), (a ≠ b) → even (nat.dist a b)) :
  ∃ species_num : ℕ, species_num = 1011 :=
by
  sorry

end bird_species_min_l75_75487


namespace correct_statement_b_l75_75375

open Set 

variables {Point Line Plane : Type}
variable (m n : Line)
variable (α : Plane)
variable (perpendicular_to_plane : Line → Plane → Prop) 
variable (parallel_to_plane : Line → Plane → Prop)
variable (is_subline_of_plane : Line → Plane → Prop)
variable (perpendicular_to_line : Line → Line → Prop)

theorem correct_statement_b (hm : perpendicular_to_plane m α) (hn : is_subline_of_plane n α) : perpendicular_to_line m n :=
sorry

end correct_statement_b_l75_75375


namespace ceil_floor_eq_zero_l75_75329

theorem ceil_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (- (7 / 3)) = 0) :=
by
  sorry

end ceil_floor_eq_zero_l75_75329


namespace ribbon_fraction_per_box_l75_75047

theorem ribbon_fraction_per_box 
  (total_ribbon_used : ℚ)
  (number_of_boxes : ℕ)
  (h1 : total_ribbon_used = 5/8)
  (h2 : number_of_boxes = 5) :
  (total_ribbon_used / number_of_boxes = 1/8) :=
by
  sorry

end ribbon_fraction_per_box_l75_75047


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75731

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75731


namespace correct_calculation_l75_75751

theorem correct_calculation :
  ¬(sqrt 9 = ±sqrt 3) ∧
  ¬(sqrt ((-2)^2) = -2) ∧
  ¬(-sqrt 16 = 4) ∧
  (-∛(-8) = 2) :=
by
  -- Define the evaluations:
  have sqrt_9_eq_3 : sqrt 9 = 3 := by sorry
  have sqrt_neg2_sq_eq_2 : sqrt ((-2)^2) = 2 := by sorry
  have sqrt_16_eq_4 : sqrt 16 = 4 := by sorry
  have cube_root_neg8_eq_neg2 : ∛(-8) = -2 := by sorry

  -- Proof of each condition:
  split
  case left => exact ne_of_lt (by sorry)
  split
  case left => exact ne_of_lt (by sorry)
  split
  case left => exact ne_of_lt (by sorry)
  exact eq.symm (by exact_congr cube_root_neg8_eq_neg2.symm ‹_›)

  sorry

end correct_calculation_l75_75751


namespace median_length_is_sqrt17_l75_75159

-- Define the vertices of the triangle
def A : ℝ × ℝ × ℝ := (2, 0, 3)
def B : ℝ × ℝ × ℝ := (-5, 2, 1)
def C : ℝ × ℝ × ℝ := (3, 2, 1)

-- Function to calculate the midpoint of two points in 3D
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Function to calculate the distance between two points in 3D
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

-- Define the midpoint D of segment BC
def D : ℝ × ℝ × ℝ := midpoint B C

-- Lean statement for proving the length of the median AD
theorem median_length_is_sqrt17 : distance A D = Real.sqrt 17 := 
by {
  -- The proof goes here
  sorry
}

end median_length_is_sqrt17_l75_75159


namespace largest_n_binom_sum_l75_75195

open Nat

theorem largest_n_binom_sum (n : ℕ) 
  (h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n) : n = 7 :=
by 
  sorry

end largest_n_binom_sum_l75_75195


namespace largest_binom_coeff_l75_75187

theorem largest_binom_coeff (n : ℕ) :
  (nat.choose 10 3) + (nat.choose 10 4) = nat.choose 11 n → n = 7 :=
by sorry

end largest_binom_coeff_l75_75187


namespace zero_digit_count_in_pages_l75_75781

theorem zero_digit_count_in_pages : let pages := list.range' 1 900
  pages.countp (λ n, n.digits 10).any (λ d, d = 0) = 189 :=
by
  sorry

end zero_digit_count_in_pages_l75_75781


namespace min_species_needed_l75_75470

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l75_75470


namespace probability_xi_leq_0_l75_75904

noncomputable def xi : ℝ → ℝ := sorry -- Define the random variable ξ, which is normally distributed.

axiom normal_dist {μ δ : ℝ} (Hδ_pos : δ > 0) : 
  ∀ x : ℝ, xi x ∼ NormalDist.mk μ δ

axiom probability_condition : (ProbabilityMassFunc xi).probability_event (λ x, x ≤ 4) = 0.84

theorem probability_xi_leq_0 (μ δ : ℝ) (Hδ_pos : δ > 0) (Hμ : μ = 2) :
  (ProbabilityMassFunc xi).probability_event (λ x, x ≤ 0) = 0.16 := by
begin
  -- main proof using the given conditions
  have H1 : (ProbabilityMassFunc xi).probability_event (λ x, x ≤ 4) = 0.84,
  from probability_condition,
  have H2 : (ProbabilityMassFunc xi).probability_event (λ x, x ≥ 4) = 1 - 0.84,
  from by rw [sub_self],
  have symmetry_property : (ProbabilityMassFunc xi).probability_event (λ x, x ≤ 0) = 
                          (ProbabilityMassFunc xi).probability_event (λ x, x ≥ 4),
  from sorry,
  rw [H2, symmetry_property],
  exact 0.16,
end

end probability_xi_leq_0_l75_75904


namespace choose_4_from_15_l75_75949

theorem choose_4_from_15 : Nat.choose 15 4 = 1365 := by
  sorry

end choose_4_from_15_l75_75949


namespace sum_of_integers_whose_squares_end_with_25_l75_75715

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75715


namespace max_value_of_trig_function_l75_75131

theorem max_value_of_trig_function :
  (∀ x : ℝ, y = sin x - (1 / 2) * cos x → y ≤ sqrt 5 / 2) ∧ (∃ x : ℝ, y = sin x - (1 / 2) * cos x ∧ y = sqrt 5 / 2) :=
sorry

end max_value_of_trig_function_l75_75131


namespace car_rental_cost_per_mile_l75_75254

theorem car_rental_cost_per_mile (daily_rate budget miles_driven : ℝ) 
    (h_daily_rate : daily_rate = 30) 
    (h_budget : budget = 75) 
    (h_miles_driven : miles_driven = 250) : 
    let x := (budget - daily_rate) / miles_driven in
    x = 0.18 :=
by
    sorry

end car_rental_cost_per_mile_l75_75254


namespace ornithological_park_species_l75_75476

/-- In an ornithological park, there are 2021 birds arranged in a row.
Each pair of birds of the same species has an even number of birds between them.
Prove that the smallest number of bird species is 1011. -/
theorem ornithological_park_species (n : ℕ) (h1 : n = 2021) 
  (h2 : ∀ s : ℕ, s ∈ {1..n} → (∀ x y : ℕ, x < y ∧ x ≠ y → (∀ z : ℕ, z ∈ ({x, y} : set ℕ) → even (y - x - 1))) ) 
  : s ≥ 1011 :=
sorry

end ornithological_park_species_l75_75476


namespace combinations_count_l75_75541

def colorChoices := 4
def decorationChoices := 3
def methodChoices := 3

theorem combinations_count : colorChoices * decorationChoices * methodChoices = 36 := by
  sorry

end combinations_count_l75_75541


namespace angle_QCB_eq_angle_BAC_l75_75022

theorem angle_QCB_eq_angle_BAC
  (A B C M D E P Q : Type)
  [Midpoint M A C]   -- M is midpoint of AC
  (MD_parallel_AB : Parallel M D A B)  -- MD parallel to AB
  (MD_tangent_A : IsTangent M D A (Circumcircle (triangle A B C)))  -- MD meets tangent at A to circumcircle of ABC at D
  [Midpoint A D E]  -- A is the midpoint of DE
  (P_circumcircle_ABE : OnCircumcircle P (triangle A B E) A C)  -- P = (ABE) ∩ AC
  (Q_circumcircle_ADP_DM : OnCircumcircle Q (triangle A D P) M D)  -- Q = (ADP) ∩ DM
  : InteriorAngle Q C B = InteriorAngle B A C := sorry

end angle_QCB_eq_angle_BAC_l75_75022


namespace total_amount_paid_l75_75990

-- Definitions of the conditions
def cost_earbuds : ℝ := 200
def tax_rate : ℝ := 0.15

-- Statement to prove
theorem total_amount_paid : (cost_earbuds + (cost_earbuds * tax_rate)) = 230 := sorry

end total_amount_paid_l75_75990


namespace sum_of_squares_ending_in_25_l75_75718

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75718


namespace distance_formula_barycentric_l75_75518

variable (α1 β1 γ1 α2 β2 γ2 : ℝ)
variable (A B C : ℝ)
variable (S : ℝ)
noncomputable def S_A := 2 * S * Real.cot A
noncomputable def S_B := 2 * S * Real.cot B
noncomputable def S_C := 2 * S * Real.cot C

theorem distance_formula_barycentric :
  let M_N_sq := S_A * (α1 - α2)^2 + S_B * (β1 - β2)^2 + S_C * (γ1 - γ2)^2 
  in M_N_sq = (S_A * (α1 - α2)^2 + S_B * (β1 - β2)^2 + S_C * (γ1 - γ2)^2) :=
sorry

end distance_formula_barycentric_l75_75518


namespace solve_logarithmic_equation_l75_75366

theorem solve_logarithmic_equation :
    (log 10 (3 * (68 / 3 : ℝ)^2 - 24 * (68 / 3 : ℝ)) = 3) ∧ 
    (log 10 (3 * (-44 / 3 : ℝ)^2 - 24 * (-44 / 3 : ℝ)) = 3) :=
by
    sorry

end solve_logarithmic_equation_l75_75366


namespace problem_statement_l75_75291

/-- Problem statement conditions --/

-- Condition 1: The locus of points in a plane whose sum of distances from two fixed points F1 and F2 is a constant.
def is_ellipse (F1 F2 : Point) (c : ℝ) : Prop :=
  c > dist F1 F2

-- Condition 2: Three non-collinear vectors
structure Vectors3D (α : Type u) [Field α] [AddCommGroup α] [Module α] :=
  (e1 e2 e3 : α)
  (non_collinear : ¬ collinear e1 e2 e3)

-- Condition 2: The existence of a unique set of real numbers for linear combination of vectors
def exists_unique_linear_combination {α : Type u} [Field α] [AddCommGroup α] [Module α] 
  (v1 v2 v3 a : α) : Prop :=
  ∃ (λ1 λ2 λ3 : α), a = λ1 • v1 + λ2 • v2 + λ3 • v3 ∧ 
  ∀ (μ1 μ2 μ3 : α), a = μ1 • v1 + μ2 • v2 + μ3 • v3 → (λ1, λ2, λ3) = (μ1, μ2, μ3)

-- Condition 3: Sufficient and necessary conditions
def sufficient_but_not_necessary (p q : Prop) : Prop :=
  (p → q) ∧ ¬ (q → p)

def necessary_but_not_sufficient (p q : Prop) : Prop :=
  (q → p) ∧ ¬ (p → q)

/-- Proof to be done: Prop3 is true, others are false --/

theorem problem_statement :
  ¬ is_ellipse F1 F2 c ∧ -- Condition (1) is false
  ¬ exists_unique_linear_combination e1 e2 e3 a ∧ -- Condition (2) is false
  sufficient_but_not_necessary p q → 
  necessary_but_not_sufficient p q := -- Condition (3) is true
begin
  -- Proof steps here (omitted)
  sorry
end

end problem_statement_l75_75291


namespace helens_mother_brought_101_l75_75002

-- Define the conditions
def total_hotdogs : ℕ := 480
def dylan_mother_hotdogs : ℕ := 379
def helens_mother_hotdogs := total_hotdogs - dylan_mother_hotdogs

-- Theorem statement: Prove that the number of hotdogs Helen's mother brought is 101
theorem helens_mother_brought_101 : helens_mother_hotdogs = 101 :=
by
  sorry

end helens_mother_brought_101_l75_75002


namespace volleyball_tournament_l75_75971

theorem volleyball_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end volleyball_tournament_l75_75971


namespace necessary_but_not_sufficient_l75_75036

variable {a b c : ℝ}

theorem necessary_but_not_sufficient (h1 : b^2 - 4 * a * c ≥ 0) (h2 : a * c > 0) (h3 : a * b < 0) : 
  ¬∀ r1 r2 : ℝ, (r1 = (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) ∧ (r2 = (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) → r1 > 0 ∧ r2 > 0 :=
sorry

end necessary_but_not_sufficient_l75_75036


namespace least_positive_integer_to_multiple_of_5_l75_75219

theorem least_positive_integer_to_multiple_of_5 : ∃ (n : ℕ), n > 0 ∧ (725 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m :=
by
  sorry

end least_positive_integer_to_multiple_of_5_l75_75219


namespace tangent_line_at_zero_exp_geq_x_plus_one_l75_75776

-- Define the exponential function
def f (x : ℝ) : ℝ := Real.exp x

-- Question 1: The equation of the tangent line to f(x) at x = 0 is y = x + 1
theorem tangent_line_at_zero : 
  ∀ x, (x - 0) * f' 0 + f 0 = x + 1 := by
  sorry

-- Question 2: Prove the inequality e^x ≥ x + 1 for all x ∈ ℝ
theorem exp_geq_x_plus_one (x : ℝ) : 
  f x ≥ x + 1 := by
  sorry

end tangent_line_at_zero_exp_geq_x_plus_one_l75_75776


namespace correct_statements_l75_75884

-- Define the conditions
variables {O B C P : Type}
variables {λ μ : ℝ}
variables [InnerProductSpace ℝ Type]

-- Define point and vector relationships
variable (OP OC OB : ℝ)
hypothesis h1 : ∀ (O B C P : Type) (λ μ : ℝ), OP = λ * OB + μ * OC
hypothesis h_λ1 : λ + μ = 1
hypothesis h_λ_pos : λ > 0
hypothesis h_λ_neg : λ < 0
hypothesis h_λμ_gt : λ + μ > 1
hypothesis h_λμ_lt : λ + μ < 1

-- Lean statement to prove the correct statements
theorem correct_statements (h1 : ∀ (O B C P : Type) (λ μ : ℝ), OP = λ * OB + μ * OC)
  (h_λ1 : λ + μ = 1) (h_λ_neg : λ < 0) (h_λμ_gt : λ + μ > 1) : 
  (statement_02 : Point P is on the extension line of segment BC) ∧
  (statement_03 : Point P is outside triangle OBC) := 
by sorry

end correct_statements_l75_75884


namespace solution_set_of_inequality_l75_75351

theorem solution_set_of_inequality :
  { x : ℝ | x ≠ 5 ∧ (x * (x + 1)) / ((x - 5) ^ 3) ≥ 25 } = 
  { x : ℝ | x ≤ 5 / 3 } ∪ { x : ℝ | x > 5 } := by
  sorry

end solution_set_of_inequality_l75_75351


namespace K_position_KF_length_l75_75515

noncomputable def side_length := 5
constant (A B C D E F L K : Type) 
constant hexagon : regular_hexagon A B C D E F side_length
constant L_intersection : is_intersection_point_of_diagonals C E D F L
constant vector_LK : (FB - 3 * AB) = LK
constant segment_length_KF := sorry

theorem K_position : is_outside_of_hexagon K :=
sorry

theorem KF_length : segment_length_KF = (5 * sqrt 3) / 3 :=
sorry

end K_position_KF_length_l75_75515


namespace population_increase_l75_75452

-- Define the problem conditions
def average_birth_rate := (6 + 10) / 2 / 2  -- the average number of births per second
def average_death_rate := (4 + 8) / 2 / 2  -- the average number of deaths per second
def net_migration_day := 500  -- net migration inflow during the day
def net_migration_night := -300  -- net migration outflow during the night

-- Define the number of seconds in a day
def seconds_in_a_day := 24 * 3600

-- Define the net increase due to births and deaths
def net_increase_births_deaths := (average_birth_rate - average_death_rate) * seconds_in_a_day

-- Define the total net migration
def total_net_migration := net_migration_day + net_migration_night

-- Define the total population net increase
def total_population_net_increase :=
  net_increase_births_deaths + total_net_migration

-- The theorem to be proved
theorem population_increase (h₁ : average_birth_rate = 4)
                           (h₂ : average_death_rate = 3)
                           (h₃ : seconds_in_a_day = 86400) :
  total_population_net_increase = 86600 := by
  sorry

end population_increase_l75_75452


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75703

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75703


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75741

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75741


namespace best_fitting_model_l75_75451

/-- Four models with different coefficients of determination -/
def model1_R2 : ℝ := 0.98
def model2_R2 : ℝ := 0.80
def model3_R2 : ℝ := 0.50
def model4_R2 : ℝ := 0.25

/-- Prove that Model 1 has the best fitting effect among the given models -/
theorem best_fitting_model :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by {sorry}

end best_fitting_model_l75_75451


namespace b_must_be_one_l75_75156

theorem b_must_be_one (a b : ℝ) (h1 : a + b - a * b = 1) (h2 : ∀ n : ℤ, a ≠ n) : b = 1 :=
sorry

end b_must_be_one_l75_75156


namespace largest_int_n_binom_l75_75179

theorem largest_int_n_binom (
  h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n
) : n = 7 := by
  sorry

end largest_int_n_binom_l75_75179


namespace distance_center_to_line_l75_75124

-- Define the center of the circle derived from the polar equation
def circle_center : ℝ × ℝ := (2, 0)

-- Define the equation of the line in Cartesian coordinates
def line_eq (p : ℝ × ℝ) : Prop := p.1 - p.2 = 0

-- Define the formula for distance from a point to a line
def distance_from_point_to_line (C : ℝ × ℝ) : ℝ :=
  abs (C.1 - C.2) / real.sqrt 2

-- Define the main theorem to prove the distance
theorem distance_center_to_line : distance_from_point_to_line circle_center = real.sqrt 2 := by
  -- Proof is omitted
  sorry

end distance_center_to_line_l75_75124


namespace probability_of_digit_two_in_decimal_rep_3_div_11_l75_75089

theorem probability_of_digit_two_in_decimal_rep_3_div_11 :
  ∃ d : ℚ, d = 3 / 11 ∧ ∀ n : ℕ, n > 0 → probability_of_digit d n 2 = 1 / 2 := by
  sorry

end probability_of_digit_two_in_decimal_rep_3_div_11_l75_75089


namespace find_c_l75_75608

open Real

theorem find_c (a b c d : ℕ) (M : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1) (hM : M ≠ 1) :
  (M ^ (1 / a) * (M ^ (1 / b) * (M ^ (1 / c) * (M ^ (1 / d))))) ^ (1 / a * b * c * d) = (M ^ 37) ^ (1 / 48) →
  c = 2 :=
by
  sorry

end find_c_l75_75608


namespace power_function_direct_proportionality_inverse_proportionality_quadratic_function_l75_75909

-- The function definition
def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(-5 * m - 3)

-- Proof problems

-- 1) When does f(x) become a power function?
theorem power_function (m : ℝ) : (m = 2 ∨ m = -1) ↔ ∀ x, ∃ k, f m x = k * x :=
sorry

-- 2) When does f(x) become a direct proportionality function?
theorem direct_proportionality (m : ℝ) : (m = -4/5) ↔ ∀ x, ∃ k, f m x = k * x :=
sorry

-- 3) When does f(x) become an inverse proportionality function?
theorem inverse_proportionality (m : ℝ) : (m = -2/5) ↔ ∀ x, ∃ k, f m x = k / x :=
sorry

-- 4) When does f(x) become a quadratic function?
theorem quadratic_function (m : ℝ) : (m = -1) ↔ ∀ x, f m x = x^2 :=
sorry

end power_function_direct_proportionality_inverse_proportionality_quadratic_function_l75_75909


namespace total_amount_paid_l75_75989

-- Definitions of the conditions
def cost_earbuds : ℝ := 200
def tax_rate : ℝ := 0.15

-- Statement to prove
theorem total_amount_paid : (cost_earbuds + (cost_earbuds * tax_rate)) = 230 := sorry

end total_amount_paid_l75_75989


namespace ornithological_park_species_l75_75478

/-- In an ornithological park, there are 2021 birds arranged in a row.
Each pair of birds of the same species has an even number of birds between them.
Prove that the smallest number of bird species is 1011. -/
theorem ornithological_park_species (n : ℕ) (h1 : n = 2021) 
  (h2 : ∀ s : ℕ, s ∈ {1..n} → (∀ x y : ℕ, x < y ∧ x ≠ y → (∀ z : ℕ, z ∈ ({x, y} : set ℕ) → even (y - x - 1))) ) 
  : s ≥ 1011 :=
sorry

end ornithological_park_species_l75_75478


namespace ceil_floor_eq_zero_l75_75328

theorem ceil_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (- (7 / 3)) = 0) :=
by
  sorry

end ceil_floor_eq_zero_l75_75328


namespace price_reduction_for_desired_profit_l75_75256

def profit_per_piece (x : ℝ) : ℝ := 40 - x
def pieces_sold_per_day (x : ℝ) : ℝ := 20 + 2 * x

theorem price_reduction_for_desired_profit (x : ℝ) :
  (profit_per_piece x) * (pieces_sold_per_day x) = 1200 ↔ (x = 10 ∨ x = 20) := by
  sorry

end price_reduction_for_desired_profit_l75_75256


namespace conditional_probability_l75_75770

noncomputable def P {α : Type*} (p : α → Prop) [measurable_space α] [probability_measure α] : ℝ := sorry

variables {n m l k : ℕ}
variables {p : ℝ} (h_p : 0 < p ∧ p < 1)
variables (ξ : ℕ → ℕ) 
variables (H_ξ : ∀ i, i.i.d. (bernoulli p) (ξ i)) 

-- Define partial sum S_k
def S : ℕ → ℕ
| 0     := 0
| (k+1) := S k + ξ (k+1)

-- Define binomial coefficients
noncomputable def C (n k : ℕ) : ℕ := nat.choose n k

theorem conditional_probability (H_k : S m = k) (H_l : S n = l) :
  P (λ ω, S m = k ∧ S n = l) / P (λ ω, S n = l) = 
  (C m k) * (C (n - m) (l - k)) / (C n l) := 
sorry

end conditional_probability_l75_75770


namespace repaint_chessboard_possible_l75_75079

def can_repaint_chessboard (n : ℕ) (initial : ℕ → ℕ → Prop) (corner_recolor : ℕ → ℕ → Prop) : Prop :=
∀ (initial_state : ℕ → ℕ → bool), (∃ (operations : list (ℕ × ℕ)), ∀ (i j : ℕ),
     (i < n) → (j < n) → final_color i j = initial_color i j ∘ operation (corner_recolor i j)) :=
  true

theorem repaint_chessboard_possible : can_repaint_chessboard 8 initial corner_recolor :=
sorry

end repaint_chessboard_possible_l75_75079


namespace probability_X_leq_1_in_interval_l75_75554

noncomputable def probability_leq_one : ℝ := 
  let a : ℝ := -2
  let b : ℝ := 3
  let x : ℝ := 1 in
  (x - a) / (b - a)

theorem probability_X_leq_1_in_interval : probability_leq_one = 3/5 := by
  sorry

end probability_X_leq_1_in_interval_l75_75554


namespace tallest_player_height_correct_l75_75154

-- Define the height of the shortest player
def shortest_player_height : ℝ := 68.25

-- Define the height difference between the tallest and shortest player
def height_difference : ℝ := 9.5

-- Define the height of the tallest player based on the conditions
def tallest_player_height : ℝ :=
  shortest_player_height + height_difference

-- Theorem statement
theorem tallest_player_height_correct : tallest_player_height = 77.75 := by
  sorry

end tallest_player_height_correct_l75_75154


namespace sum_of_squares_ends_in_25_is_644_l75_75689

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75689


namespace John_walked_miles_to_park_l75_75511

theorem John_walked_miles_to_park :
  ∀ (total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles : ℕ),
    total_skateboarded_miles = 24 →
    skateboarded_first_leg = 10 →
    skateboarded_return_leg = 10 →
    total_skateboarded_miles = skateboarded_first_leg + skateboarded_return_leg + walked_miles →
    walked_miles = 4 :=
by
  intros total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles
  intro h1 h2 h3 h4
  sorry

end John_walked_miles_to_park_l75_75511


namespace tetrahedron_segments_intersection_l75_75942

-- Statement for Lean 4 according to the problem
theorem tetrahedron_segments_intersection 
  (V : Type) [EuclideanSpace V] (T : Tetrahedron V)
  (A B C D : V) (T.has_vertices A B C D) :
  ∃ O : V,
    (∀ (X ∈ [A, B, C, D]), 
      let X1 := centroid (T.opposite_face X)
      in collinear [X, X1, O] ∧ dist X O = 3 * dist O X1)
  ∧ 
    (∀ (E1 E2 E3 E4 : V) (T.edges.include [E1, E2, E3, E4]),
      let M1 := midpoint E1 E2,
          M2 := midpoint E3 E4
      in collinear [M1, M2, O] ∧ dist M1 O = dist O M2) := 
sorry

end tetrahedron_segments_intersection_l75_75942


namespace tangency_condition_maximum_OA_plus_OB_l75_75976

-- Define the curve C: ρ = 2a cos θ where a > 0
def curve_C (a : ℝ) (a_pos : 0 < a) (θ : ℝ) : ℝ := 2 * a * real.cos θ

-- Define the line l: ρ cos (θ - π/3) = 3/2
def line_l (θ : ℝ) : ℝ := real.cos (θ - real.pi / 3) * (3 / 2)

theorem tangency_condition (a : ℝ) (a_pos : 0 < a) :
  (∃ θ, curve_C a a_pos θ = line_l θ) → a = 1 :=
sorry

-- Define the polar coordinates |OA| and |OB|
def OA (a : ℝ) (θ : ℝ) : ℝ := 2 * real.cos θ

def OB (a : ℝ) (θ : ℝ) : ℝ := 2 * real.cos (θ + real.pi / 3)

-- Define the sum of |OA| and |OB|
def OA_plus_OB (a : ℝ) (θ : ℝ) : ℝ := OA a θ + OB a θ

-- Assuming a = 1, find the maximum value of |OA| + |OB| given ∠AOB = π/3
theorem maximum_OA_plus_OB :
  (∃ θ, real.cos (θ + real.pi / 6) = 1) → ∀ (a : ℝ), (a = 1) → ∀ (θ : ℝ), OA_plus_OB a θ = 2 * real.sqrt 3 :=
sorry

end tangency_condition_maximum_OA_plus_OB_l75_75976


namespace length_base_bc_l75_75504

theorem length_base_bc {A B C D : Type} [Inhabited A]
  (AB AC : ℕ)
  (BD : ℕ → ℕ → ℕ → ℕ) -- function for the median on AC
  (perimeter1 perimeter2 : ℕ)
  (h1 : AB = AC)
  (h2 : perimeter1 = 24 ∨ perimeter2 = 30)
  (AD CD : ℕ) :
  (AD = CD ∧ (∃ ab ad cd, ab + ad = perimeter1 ∧ cd + ad = perimeter2 ∧ ((AB = 2 * AD ∧ BC = 30 - CD) ∨ (AB = 2 * AD ∧ BC = 24 - CD)))) →
  (BC = 22 ∨ BC = 14) := 
sorry

end length_base_bc_l75_75504


namespace complex_quadrant_l75_75881

def i : ℂ := complex.I
def z : ℂ := i * (1 + 2 * i)

theorem complex_quadrant :
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_quadrant_l75_75881


namespace parabola_vertex_y_coord_l75_75364

theorem parabola_vertex_y_coord (a b c x y : ℝ) (h : a = 2 ∧ b = 16 ∧ c = 35 ∧ y = a*x^2 + b*x + c ∧ x = -b / (2 * a)) : y = 3 :=
by
  sorry

end parabola_vertex_y_coord_l75_75364


namespace trapezoid_area_is_400_l75_75305

noncomputable def area_of_one_trapezoid
  (base1 base2 non_parallel_side side_length_large_square area_large_square side_length_small_square : ℝ)
  (h1 : base1 = 30) (h2 : base2 = 50) (h3 : non_parallel_side > 0)
  (h4 : side_length_large_square ^ 2 = area_large_square) 
  (h5 : area_large_square = 2500) 
  (h6 : side_length_small_square = 30)
  (h7 : area_large_square - (side_length_small_square ^ 2) = 1600):
  ℝ :=
  (area_large_square - (side_length_small_square ^ 2)) / 4

theorem trapezoid_area_is_400 :
  area_of_one_trapezoid 30 50 1 50 2500 30 30 = 400 := 
  by sorry

end trapezoid_area_is_400_l75_75305


namespace choose_4_from_15_l75_75945

theorem choose_4_from_15 : Nat.choose 15 4 = 1365 := by
  sorry

end choose_4_from_15_l75_75945


namespace increasing_function_on_pos_infty_only_A_l75_75234

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

def function_A := λ x : ℝ, x^2 + 1
def function_B := λ x : ℝ, 3 - 2 * x
def function_C := λ x : ℝ, 1 / x
def function_D := λ x : ℝ, -x^2 + 1

theorem increasing_function_on_pos_infty_only_A :
  is_increasing_on_interval function_A 0 ⊤ ∧
  ¬ is_increasing_on_interval function_B 0 ⊤ ∧
  ¬ is_increasing_on_interval function_C 0 ⊤ ∧
  ¬ is_increasing_on_interval function_D 0 ⊤ :=
by
  sorry

end increasing_function_on_pos_infty_only_A_l75_75234


namespace abs_gt_1_not_sufficient_nor_necessary_l75_75774

theorem abs_gt_1_not_sufficient_nor_necessary (a : ℝ) :
  ¬((|a| > 1) → (a > 0)) ∧ ¬((a > 0) → (|a| > 1)) :=
by
  sorry

end abs_gt_1_not_sufficient_nor_necessary_l75_75774


namespace greatest_number_of_dimes_l75_75568

-- Definitions according to the conditions in a)
def total_value_in_cents : ℤ := 485
def dime_value_in_cents : ℤ := 10
def nickel_value_in_cents : ℤ := 5

-- The proof problem in Lean 4
theorem greatest_number_of_dimes : 
  ∃ (d : ℤ), (dime_value_in_cents * d + nickel_value_in_cents * d = total_value_in_cents) ∧ d = 32 := 
by
  sorry

end greatest_number_of_dimes_l75_75568


namespace mapping_preserves_parabola_and_tangent_lines_perpendicular_at_origin_l75_75863

theorem mapping_preserves_parabola_and_tangent_lines_perpendicular_at_origin
  (a b c d : ℝ) (h_abcd : (a, b, c, d) ≠ (1, 0, 0, 1))
  (k : ℝ) (h_k_nonzero : k ≠ 0)
  (preserves_parabola : ∀ (x y : ℝ), y = x^2 - x + k → (c * x + d * y = (a * x + b * y)^2 - (a * x + b * y) + k)) :
  a = -1 ∧ b = 0 ∧ c = 2 ∧ d = 1 ∧ k = 1/2 ∧ 
  (∀ a : ℝ, a = ± (1 / Real.sqrt 2) → (x, y) = (a, 1 - a / Real.sqrt 2) → 
    let A := (a, a^2 - a + k)
    let A' := (-a, a^2 + a + k)
    (2 * A.1 - 1) * (-2 * A'.1 - 1) = -1) :=
sorry

end mapping_preserves_parabola_and_tangent_lines_perpendicular_at_origin_l75_75863


namespace sum_of_two_digit_numbers_ending_in_25_l75_75631

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75631


namespace shortest_distance_from_parabola_to_line_l75_75531

theorem shortest_distance_from_parabola_to_line :
  let A (a : ℝ) := (a, a^2 - 6*a + 14)
  let distance_to_line (a : ℝ) : ℝ := abs(-a^2 + 7*a - 19) / real.sqrt 2
  ∃ a : ℝ, distance_to_line a = real.sqrt 15 / 2 := sorry

end shortest_distance_from_parabola_to_line_l75_75531


namespace min_groups_divisible_by_six_l75_75137

def group_divisible_by_six : Prop :=
  ∃ (groups : Finset (Finset ℕ)), 
    ∀ g ∈ groups, 
      (∀ x y ∈ g, (x + y) % 6 = 0) ∧ 
      (Finset.bUnion groups id = Finset.range 600) ∧ 
      (∀ g1 g2 ∈ groups, g1 ≠ g2 → g1 ∩ g2 = ∅) ∧ 
      Finset.card groups = 202

theorem min_groups_divisible_by_six : group_divisible_by_six :=
sorry

end min_groups_divisible_by_six_l75_75137


namespace cube_surface_area_with_holes_l75_75808

theorem cube_surface_area_with_holes :
  let cube_edge := 5
  let hole_side := 2
  (total_surface_area cube_edge hole_side = 258)
:= by
  let cube_edge := 5
  let hole_side := 2

  -- Define the functions inline to ensure clean dependencies.
  def face_area (edge : ℕ) := edge * edge
  def num_faces := 6
  def original_surface_area (edge : ℕ) := num_faces * (face_area edge)
  def num_holes := 3
  def hole_area (side : ℕ) := side * side
  def area_removed_by_holes (side : ℕ) := num_holes * (hole_area side)
  def exposed_area (cube_edge hole_side : ℕ) := num_holes * 4 * (cube_edge * hole_side)
  def total_surface_area (cube_edge hole_side : ℕ) := 
    (original_surface_area cube_edge) 
    - (area_removed_by_holes hole_side) 
    + (exposed_area cube_edge hole_side)

  show (total_surface_area cube_edge hole_side = 258), by
  sorry -- The proof goes here.

end cube_surface_area_with_holes_l75_75808


namespace sum_of_two_digit_numbers_ending_in_25_l75_75628

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75628


namespace flea_can_visit_all_nat_numbers_l75_75294

def flea_jumps (k : ℕ) : ℤ := 2^k + 1

theorem flea_can_visit_all_nat_numbers :
  ∃ (f : ℕ → bool), (∀ n : ℕ, ∃ k : ℕ, ∃ m : ℤ, m = if f k then flea_jumps k else -flea_jumps k) → true :=
by
  sorry

end flea_can_visit_all_nat_numbers_l75_75294


namespace movie_of_the_year_min_lists_l75_75768

theorem movie_of_the_year_min_lists (members : ℕ) (fraction : ℚ) (h_members : members = 775) (h_fraction : fraction = 1 / 4) :
  let min_lists := ⌈fraction * members⌉ in
  min_lists = 194 :=
by
  sorry

end movie_of_the_year_min_lists_l75_75768


namespace find_n_expansion_l75_75864

theorem find_n_expansion : 
  (∃ n : ℕ, 4^n + 2^n = 1056) → n = 5 :=
by sorry

end find_n_expansion_l75_75864


namespace math_problem_l75_75395

variable (a b : ℝ)

def unequal_reals (a b : ℝ) : Prop := a ≠ b
def cuberoot_eq (a b : ℝ) : Prop := (∛a + ∛b)^3 = a^2 * b^2

theorem math_problem (ha : unequal_reals a b) (hb: cuberoot_eq a b) :
  (3 * a + 1) * (3 * b + 1) - 3 * a^2 * b^2 = 1 := 
sorry

end math_problem_l75_75395


namespace cos_angle_proof_l75_75494

-- Definitions of the conditions
variable {A B C D E F : Point}
variable {a b x y : ℝ}
variable [ordered_field ℝ]

-- Condition 1: ABCD is a rectangle (sides AB, BC, CD, DA are perpendicular with equal opposite sides)
def is_rectangle (A B C D : Point) : Prop :=
  (A.dist B = C.dist D) ∧ (B.dist C = D.dist A) ∧ (A.dist C^2 + B.dist D^2 = (A.dist D)^2 + (B.dist C)^2)

-- Points E and F on sides BC and CD respectively
def on_side (E : Point) (B C : Point) : Prop := ∃ k ∈ Icc (0:ℝ) 1, E = k • B + (1-k) • C
def on_side (F : Point) (C D : Point) : Prop := ∃ k ∈ Icc (0:ℝ) 1, F = k • C + (1-k) • D

-- Equal distances AE and AF
def equal_distance (A E F : Point) : Prop := A.dist E = A.dist F

-- Area of triangles
def area_triangle (A B C : Point) : ℝ := (1 / 2) * abs((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

-- Given areas
def given_areas : Prop :=
  area_triangle A D F = 2 ∧
  area_triangle A B E = 4 ∧
  area_triangle C E F = 8

-- Define the angle and the given cosine of the angle
def angle_AEF : ℝ := real.angle A E F
def cos_angle (A E F : Point) : ℝ := real.cos (angle_AEF A E F)

theorem cos_angle_proof : 
  is_rectangle A B C D ∧ 
  on_side E B C ∧ 
  on_side F C D ∧ 
  equal_distance A E F ∧ 
  given_areas →
  7 * cos_angle A E F = 3 := by 
  sorry

end cos_angle_proof_l75_75494


namespace point_placement_in_square_l75_75277

theorem point_placement_in_square :
  let side_length : ℝ := 5
      distances : set ℝ := {1, 2, 3, 4}
  in ∃ (pts : set (ℝ × ℝ)), 
        (∀ p ∈ pts, ∀ d ∈ distances, ∃ a b c d : ℝ, 
            a = d_dist_to_side p side_length ∧
            b = d_dist_to_side p side_length ∧
            c = d_dist_to_side p side_length ∧
            d = d_dist_to_side p side_length ∧
            insert a (insert b (insert c {d})) = distances) ∧
       pts.card = 8 :=
by
  let side_length : ℝ := 5
      distances : set ℝ := {1, 2, 3, 4}
  sorry

end point_placement_in_square_l75_75277


namespace least_integer_to_add_l75_75213

theorem least_integer_to_add (n : ℕ) (h : n = 725) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 :=
by
  use 5
  split
  · exact Nat.lt_succ_self 4
  · rw [h]
    norm_num
    sorry

end least_integer_to_add_l75_75213


namespace rectangle_covered_by_checkered_figure_l75_75788

noncomputable def checkered_figure_covering 
  (m n : ℕ) 
  (F : Type) 
  (position : F → ℕ × ℕ → Prop) 
  (rotate : F → F) 
  (pos_sum_property : ∀ (rect_fill : ℕ → ℕ → ℕ), 
    (∑ i j, rect_fill i j) > 0 → 
    ∃ f : F, ∃ i j : ℕ, position f (i, j) ∧ (∑ i j, if position f (i, j) then rect_fill i j else 0) > 0) 
  : Prop :=
∃ d : F → ℚ, 
  (∀ (i j : ℕ), ∑ f : F, if position f (i, j) then d f else 0 = 1) ∧ 
  (∀ f : F, 0 ≤ d f)

theorem rectangle_covered_by_checkered_figure 
  {m n : ℕ} 
  {F : Type} 
  {position : F → ℕ × ℕ → Prop} 
  {rotate : F → F}
  (pos_sum_property : ∀ (rect_fill : ℕ → ℕ → ℕ), 
    (∑ i j, rect_fill i j) > 0 → 
    ∃ f : F, ∃ i j : ℕ, position f (i, j) ∧ (∑ i j, if position f (i, j) then rect_fill i j else 0) > 0) 
  : checkered_figure_covering m n F position rotate pos_sum_property :=
sorry

end rectangle_covered_by_checkered_figure_l75_75788


namespace probability_at_least_four_girls_l75_75045

noncomputable def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_at_least_four_girls
  (n : ℕ)
  (p : ℝ)
  (q : ℝ)
  (h_pq : p + q = 1)
  (h_p : p = 0.55)
  (h_q : q = 0.45)
  (h_n : n = 7) :
  (binomial_probability n 4 p) + (binomial_probability n 5 p) + (binomial_probability n 6 p) + (binomial_probability n 7 p) = 0.59197745 :=
sorry

end probability_at_least_four_girls_l75_75045


namespace randy_biscuits_l75_75561

theorem randy_biscuits (initial_biscuits father_gift mother_gift brother_ate : ℕ) : 
  (initial_biscuits = 32) →
  (father_gift = 13) →
  (mother_gift = 15) →
  (brother_ate = 20) →
  initial_biscuits + father_gift + mother_gift - brother_ate = 40 := by
  sorry

end randy_biscuits_l75_75561


namespace eval_ceil_floor_sum_l75_75346

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l75_75346


namespace propositions_correct_l75_75528

-- Definitions for the mathematical entities involved
variable (l m : Type) -- Lines l and m
variable (a : Type) -- Plane a
variable [Nonempty l] [Nonempty m] -- Lines are non-empty
variable [Nonempty a] -- Plane is non-empty
variable (perp : l → a → Prop) -- Perpendicularity relation
variable (subset : m → a → Prop) -- Subset relation
variable (parallel : l → m → Prop) -- Parallel relation

-- Statements of the propositions
def proposition1 (l a : l) (m : m) := (perp l a) ∧ (subset m a) → (perp l m)
def proposition2 (l : l) (a : a) (m : m) := (perp l a) ∧ (parallel l m) → (perp m a)
def proposition3 (l : l) (a : a) (m : m) := (parallel l a) ∧ (subset m a) → (parallel l m)
def proposition4 (l : l) (m : m) (a : a) := (parallel l a) ∧ (parallel m a) → (parallel l m)

-- The main statement to prove
theorem propositions_correct : 
  (proposition1 l a m) ∧ 
  (proposition2 l a m) ∧ 
  ¬ (proposition3 l a m) ∧
  ¬ (proposition4 l a m) := 
by
  sorry -- Proof goes here

end propositions_correct_l75_75528


namespace sum_of_integers_whose_squares_end_with_25_l75_75707

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75707


namespace law_firm_associates_l75_75244

theorem law_firm_associates :
  (percent_second_year percent_not_first_year : ℝ) 
  (h_percent_second_year : percent_second_year = 0.25)
  (h_percent_not_first_year : percent_not_first_year = 0.75) :
  1.0 - percent_second_year - (1.0 - percent_not_first_year) = 0.5 := 
by
  sorry

end law_firm_associates_l75_75244


namespace consecutive_integers_250th_digit_is_five_l75_75437

def sequence_250th_digit : ℕ :=
  let sequence := String.intercalate "" (List.map toString (List.range' 1 (150 + 1)).reverse)
  sequence.get 249 -- 0-indexed, so the 250th digit is at index 249

theorem consecutive_integers_250th_digit_is_five :
  sequence_250th_digit = '5' :=
sorry

end consecutive_integers_250th_digit_is_five_l75_75437


namespace sum_of_two_digit_numbers_ending_in_25_l75_75635

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75635


namespace sum_of_x_coords_of_solutions_l75_75857

theorem sum_of_x_coords_of_solutions :
  (∑ x in {x | |x^2 - 4 * x + 3| = 25 / 4 - x}, x) = 6 := 
by
sorry

end sum_of_x_coords_of_solutions_l75_75857


namespace area_of_interior_triangle_l75_75408

theorem area_of_interior_triangle
  (leg1_sq : ℝ)
  (leg2_sq : ℝ)
  (hyp_sq : ℝ)
  (h_leg1_sq : leg1_sq = 225)
  (h_leg2_sq : leg2_sq = 64)
  (h_hyp_sq : hyp_sq = 256)
  (h_right_triangle : leg1_sq + leg2_sq = hyp_sq) :
  let leg1 := Real.sqrt leg1_sq
  let leg2 := Real.sqrt leg2_sq
  in (1 / 2) * leg1 * leg2 = 60 :=
by
  have h_leg1 : ℝ := Real.sqrt 225
  have h_leg2 : ℝ := Real.sqrt 64
  show (1 / 2) * h_leg1 * h_leg2 = 60
  sorry

end area_of_interior_triangle_l75_75408


namespace const_seq_is_arithmetic_not_geometric_l75_75145

-- Define the sequence
def const_seq (n : ℕ) : ℕ := 0

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, seq (n + 1) = seq n + d

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = seq n * r

-- The proof statement
theorem const_seq_is_arithmetic_not_geometric :
  is_arithmetic_sequence const_seq ∧ ¬ is_geometric_sequence const_seq :=
by
  sorry

end const_seq_is_arithmetic_not_geometric_l75_75145


namespace smallest_sum_of_factors_of_8_l75_75127

theorem smallest_sum_of_factors_of_8! :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a * b * c * d = Nat.factorial 8 ∧ a + b + c + d = 102 :=
sorry

end smallest_sum_of_factors_of_8_l75_75127


namespace max_min_ab_bc_ca_l75_75524

theorem max_min_ab_bc_ca (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_sum : a + b + c = 12) (h_sum_ab_bc_ca : ab + bc + ca = 32) :
    ∃ m, m = min (ab) (min (bc) (ca)) ∧ m ≤ 4 :=
begin
  sorry
end

end max_min_ab_bc_ca_l75_75524


namespace monochromatic_triangle_33_l75_75875

noncomputable theory
open_locale classical

-- Define the problem as a Lean theorem
theorem monochromatic_triangle_33 : ∃ n : ℕ, (∀ (P : fin 9 → Prop), 
  (∀ i, P i) → 
  (∀ f : fin 36 → bool, 
    (∃ (T : finset (fin 36)), T.card = n ∧ (∀ e ∈ T, f e = tt) ∧ (∃ t : finset (fin 36), t ⊆ T ∧ t.card = 3))) →
    n = 33) :=
begin
  sorry
end

end monochromatic_triangle_33_l75_75875


namespace find_symmetric_point_l75_75038

noncomputable def point_symmetric_with_respect_to_plane
  (A : ℝ × ℝ × ℝ) (B : ℝ × ℝ × ℝ) (xOy : ℝ × ℝ × ℝ → Prop) : Prop :=
  A.1 = B.1 ∧ A.2 = B.2 ∧ A.3 = -B.3

theorem find_symmetric_point :
  ∀ (A B : ℝ × ℝ × ℝ), A = (1, 2, 3) →
    point_symmetric_with_respect_to_plane A B (λ (p : ℝ × ℝ × ℝ), p.3 = 0) →
    B = (1, 2, -3) :=
by
  intros A B hA hSym
  sorry

end find_symmetric_point_l75_75038


namespace students_participation_l75_75108

theorem students_participation (like_food dislike_food : ℕ) (h1 : like_food = 383) (h2 : dislike_food = 431) :
  like_food + dislike_food = 814 :=
by {
  rw [h1, h2],
  norm_num,
  }

end students_participation_l75_75108


namespace count_elements_in_T_l75_75999

theorem count_elements_in_T :
  let T := {n: ℕ | 1 < n ∧ ∃ k, (10^20 - 1) = n * k }
  (∀ (n : ℕ), n ∈ T ↔ 1 < n ∧ (10^20 - 1) % n = 0) →
  Nat.Prime 999001 →
  ∃ D, D = (Nat.divisors  (10^20 - 1)).card →
  T.card = D - 1 :=
by
  sorry

end count_elements_in_T_l75_75999


namespace inequality_solution_l75_75147

open Set

theorem inequality_solution (x : ℝ) : (1 - 7 / (2 * x - 1) < 0) ↔ (1 / 2 < x ∧ x < 4) := 
by
  sorry

end inequality_solution_l75_75147


namespace solve_lambda_l75_75923

noncomputable def vector_a : ℝ × ℝ := (2, 1)
noncomputable def vector_b : ℝ × ℝ := (-3, 2)

theorem solve_lambda :
  let ab_sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2),
      two_a_minus_lambda_b := 
        (2 * vector_a.1 + 3 * vector_b.1, 2 * vector_a.2 - λ * vector_b.2) in
  ab_sum.1 * two_a_minus_lambda_b.1 + ab_sum.2 * two_a_minus_lambda_b.2 = 0 →
  λ = 2 / 9 :=
by 
  sorry

end solve_lambda_l75_75923


namespace total_cranes_folded_l75_75429

-- Definitions based on conditions
def hyerinCranesPerDay : ℕ := 16
def hyerinDays : ℕ := 7
def taeyeongCranesPerDay : ℕ := 25
def taeyeongDays : ℕ := 6

-- Definition of total number of cranes folded by Hyerin and Taeyeong
def totalCranes : ℕ :=
  (hyerinCranesPerDay * hyerinDays) + (taeyeongCranesPerDay * taeyeongDays)

-- Proof statement
theorem total_cranes_folded : totalCranes = 262 := by 
  sorry

end total_cranes_folded_l75_75429


namespace zoey_finishes_on_friday_l75_75240

theorem zoey_finishes_on_friday :
  let total_days := (20 * 21) / 2 in
  (total_days % 7 = 0) ∧ (total_days ≡ 0 [MOD 7]) →
  true := 
by
  let total_days := (20 * 21) / 2
  have : total_days % 7 = 0 := sorry
  have : total_days ≡ 0 [MOD 7] := sorry
  trivial

end zoey_finishes_on_friday_l75_75240


namespace calculate_sum_l75_75303

theorem calculate_sum (E : ℝ → ℝ) (h₃ : E 3 = 5) (h₄ : E 4 = 5) : E 3 + E 4 = 10 :=
by {
  -- Summing the given values
  calc
    E 3 + E 4 = 5 + 5   : by rw [h₃, h₄]
          ... = 10      : by norm_num
}

end calculate_sum_l75_75303


namespace R_xdotx_eq_partial_K_x_t2_R_dotxx_eq_partial_K_x_t1_l75_75552

variable (X : ℝ → ℝ) [IsRandomFunction X]
variable (K_x : ℝ × ℝ → ℝ) 
variable (R_xdotx : ℝ × ℝ → ℝ)
variable (R_dotxx : ℝ × ℝ → ℝ)
variable (t1 t2 : ℝ)

-- The statements for the partial derivatives of the correlation functions.
axiom partial_derivative_K_x_t2
  : (∂ K_x / ∂ (Prod.snd)) = (t1, t2) → ∂ K_x / ∂ (Prod.snd) (t1, t2)

axiom partial_derivative_K_x_t1
  : (∂ K_x / ∂ (Prod.fst)) = (t1, t2) → ∂ K_x / ∂ (Prod.fst) (t1, t2)

-- Proofs that need to be shown:
theorem R_xdotx_eq_partial_K_x_t2 
  : R_xdotx = ∂ K_x / ∂ (Prod.snd) (t1, t2) :=
sorry

theorem R_dotxx_eq_partial_K_x_t1 
  : R_dotxx = ∂ K_x / ∂ (Prod.fst) (t1, t2) :=
sorry

end R_xdotx_eq_partial_K_x_t2_R_dotxx_eq_partial_K_x_t1_l75_75552


namespace container_weight_l75_75779

-- Definition of the problem conditions
def weight_of_copper_bar : ℕ := 90
def weight_of_steel_bar := weight_of_copper_bar + 20
def weight_of_tin_bar := weight_of_steel_bar / 2

-- Formal statement to be proven
theorem container_weight (n : ℕ) (h1 : weight_of_steel_bar = 2 * weight_of_tin_bar)
  (h2 : weight_of_steel_bar = weight_of_copper_bar + 20)
  (h3 : weight_of_copper_bar = 90) :
  20 * (weight_of_copper_bar + weight_of_steel_bar + weight_of_tin_bar) = 5100 := 
by sorry

end container_weight_l75_75779


namespace number_of_ways_to_choose_4_captains_from_15_l75_75951

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end number_of_ways_to_choose_4_captains_from_15_l75_75951


namespace geometric_sequence_n_l75_75499

theorem geometric_sequence_n (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 * a 2 * a 3 = 4) 
  (h2 : a 4 * a 5 * a 6 = 12) 
  (h3 : a (n-1) * a n * a (n+1) = 324) : 
  n = 14 := 
  sorry

end geometric_sequence_n_l75_75499


namespace least_number_of_workers_needed_l75_75835

variables (total_days : ℕ) (days_passed : ℕ) (initial_workers : ℕ) (percent_complete : ℕ)
variables (work_rate_per_worker : ℕ) (least_workers_needed : ℕ)

-- Given conditions
def conditions : Prop :=
  total_days = 40 ∧
  days_passed = 10 ∧
  initial_workers = 10 ∧
  percent_complete = 40 ∧
  work_rate_per_worker = (percent_complete / days_passed) / initial_workers

-- Prove least number of workers needed
theorem least_number_of_workers_needed (h : conditions) : least_workers_needed = 5 :=
sorry

end least_number_of_workers_needed_l75_75835


namespace problem_statement_l75_75393

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3}

def operation (A B : Set ℕ) : Set ℕ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a + b}

theorem problem_statement : 
  let AB := operation A B in
  (∀ (max_elem : ℕ), max_elem ∈ AB → max_elem = 6) ∧
  (∃ (count : ℕ), count = 2 ^ (Set.card AB) - 1 ∧ count = 15) :=
by
  sorry

end problem_statement_l75_75393


namespace four_digit_number_count_l75_75426

theorem four_digit_number_count : 
  ∀ (digits : list ℕ), 
    digits = [2, 0, 2, 5] → 
    (∀ n ∈ digits, n ≠ 0 → n = 2 ∨ n = 5) → 
    ∃! (count : ℕ), count = 6 :=
by
  sorry

end four_digit_number_count_l75_75426


namespace cylindrical_coordinates_of_point_l75_75837

noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x = 0 then
             if y > 0 then Real.pi / 2 else 3 * Real.pi / 2
           else if x > 0 then Real.atan (y / x)
           else Real.atan (y / x) + Real.pi
  in (r, θ, z)

theorem cylindrical_coordinates_of_point :
  rectangular_to_cylindrical 7 (-7) 4 = (7 * Real.sqrt 2, 7 * Real.pi / 4, 4) :=
by
  sorry

end cylindrical_coordinates_of_point_l75_75837


namespace simplify_expr_l75_75107

theorem simplify_expr (x : ℝ) : 
  2 * x * (4 * x ^ 3 - 3 * x + 1) - 7 * (x ^ 3 - x ^ 2 + 3 * x - 4) = 
  8 * x ^ 4 - 7 * x ^ 3 + x ^ 2 - 19 * x + 28 := 
by
  sorry

end simplify_expr_l75_75107


namespace tourist_total_value_l75_75806

theorem tourist_total_value
    (tax_rate : ℝ)
    (V : ℝ)
    (tax_paid : ℝ)
    (exempt_amount : ℝ) :
    exempt_amount = 600 ∧
    tax_rate = 0.07 ∧
    tax_paid = 78.4 →
    (tax_rate * (V - exempt_amount) = tax_paid) →
    V = 1720 :=
by
  intros h1 h2
  have h_exempt : exempt_amount = 600 := h1.left
  have h_tax_rate : tax_rate = 0.07 := h1.right.left
  have h_tax_paid : tax_paid = 78.4 := h1.right.right
  sorry

end tourist_total_value_l75_75806


namespace min_species_needed_l75_75469

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l75_75469


namespace linear_in_x_y_l75_75019

theorem linear_in_x_y (n m : ℤ) (x y : ℝ) (h : (n - 1) * x ^ (n ^ 2) - 3 * y ^ (m - 2023) = 6) 
  (h_linear : (n ^ 2 = 1 ∧ m - 2023 = 1) ∨ False) : n ^ m = 1 := 
by
  cases h_linear
  · obtain ⟨hn, hm⟩ := h_linear
    have h_n : n = -1 := by
      rw [hn] at h
      sorry
    have h_m : m = 2024 := by
      rw [hm] at h
      sorry
    rw [h_n, h_m]
    apply pow_eq_one_iff_even.mpr
    sorry
  · exact False.elim h_linear

end linear_in_x_y_l75_75019


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75641

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75641


namespace find_number_l75_75021

theorem find_number (x : ℚ) (h : (3 * x / 2) + 6 = 11) : x = 10 / 3 :=
sorry

end find_number_l75_75021


namespace tangent_line_at_0_max_min_values_in_interval_l75_75914

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_at_0 : 
  let p : ℝ × ℝ := (0, f 0) in 
  ∀ (x : ℝ), (x = p.1) → (f p.1 = p.2) → (Real.deriv f p.1 = 0) → (x, f x) = (p.1, p.2) ∨ (f x = 1) :=
by sorry

theorem max_min_values_in_interval : (1, -Real.pi / 2) ∈ 
  ((λ x, f x) '' set.Icc 0 (Real.pi/2)) ∧
  ∃ (x : ℝ), x ∈ set.Icc 0 (Real.pi/2) ∧
  f x = 1 ∧ 
  f 0 = 1 ∧ 
  ∃ (x : ℝ), 
  x ∈ set.Icc 0 (Real.pi/2) ∧
  f x = -Real.pi / 2 := 
by sorry

end tangent_line_at_0_max_min_values_in_interval_l75_75914


namespace number_of_ways_to_choose_4_captains_from_15_l75_75953

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end number_of_ways_to_choose_4_captains_from_15_l75_75953


namespace sum_of_x_coordinates_correct_l75_75860
-- Import required Lean libraries

-- Define the conditions
noncomputable def equation1 (x : ℝ) : ℝ := abs (x^2 - 4 * x + 3)
noncomputable def equation2 (x : ℝ) : ℝ := 25 / 4 - x

-- Define the proof problem
theorem sum_of_x_coordinates_correct : 
  let s := {x : ℝ | equation1 x = equation2 x} in 
  (∑ x in s, x) = 11 / 2 :=
by
  sorry

end sum_of_x_coordinates_correct_l75_75860


namespace sum_of_squares_of_consecutive_integers_l75_75601

theorem sum_of_squares_of_consecutive_integers (b : ℕ) (h : (b-1) * b * (b+1) = 12 * ((b-1) + b + (b+1))) : 
  (b - 1) * (b - 1) + b * b + (b + 1) * (b + 1) = 110 := 
by sorry

end sum_of_squares_of_consecutive_integers_l75_75601


namespace intersection_M_N_l75_75578

-- Define the sets M and N according to the conditions given in the problem
def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

-- State the theorem to prove the intersection of M and N
theorem intersection_M_N : (M ∩ N) = {0, 1} := 
  sorry

end intersection_M_N_l75_75578


namespace scientific_notation_of_seventy_one_million_three_hundred_thousand_l75_75115

theorem scientific_notation_of_seventy_one_million_three_hundred_thousand :
    scientific_notation 71300000 = "7.13 * 10^7" :=
by
  sorry

end scientific_notation_of_seventy_one_million_three_hundred_thousand_l75_75115


namespace function_characterization_l75_75840

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ m n, Int.gcd m n ∣ f m + f n

theorem function_characterization (f : ℤ → ℤ) :
  satisfies_condition f → ∃ k : ℤ, ∀ n : ℤ, f n = k * n :=
sorry

end function_characterization_l75_75840


namespace bisector_line_equation_l75_75495

def point (x y : ℝ) := (x, y)

def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
(p2.1 - p1.1, p2.2 - p1.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
real.sqrt (v.1 * v.1 + v.2 * v.2)

def unit_vector (v : ℝ × ℝ) : ℝ × ℝ :=
let m := magnitude v in (v.1 / m, v.2 / m)

def bisector_vector (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
let u1 := unit_vector v1 in
let u2 := unit_vector v2 in
(u1.1 + u2.1, u1.2 + u2.2)

def slope (v : ℝ × ℝ) : ℝ :=
v.2 / v.1

def line_equation (p : ℝ × ℝ) (m : ℝ) : string :=
if m = 1 then "x - y = 0" else "other"

theorem bisector_line_equation :
  let A := point -1 -2
  let B := point 2 2
  let C := point -2 -1 in
  let BA := vector B A in
  let BC := vector B C in
  let bisector := bisector_vector BA BC in
  let m := slope bisector in
  line_equation B m = "x - y = 0" :=
by
  let A := point -1 -2
  let B := point 2 2
  let C := point -2 -1
  let BA := vector B A
  let BC := vector B C
  let bisector := bisector_vector BA BC
  let m := slope bisector
  sorry

end bisector_line_equation_l75_75495


namespace verify_f_and_g_l75_75756

-- Define f(n) and g(n) given the conditions
def f (n : ℕ) : ℕ := n - 3
def g (n : ℕ) : ℕ := n - 1 - ((n + 2) / 3)

-- Main theorem statement for verification
theorem verify_f_and_g (n : ℕ) (h : n ≥ 4) : f(n) = n - 3 ∧ g(n) = n - 1 - ((n + 2) / 3) :=
by {
  sorry -- proof will be added later
}

end verify_f_and_g_l75_75756


namespace functional_equation_l75_75350

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ ({0, 1} : Set ℝ) then 0 else (x^3 - 2*x) / (2*x*(x - 1))

theorem functional_equation (c : ℝ) :
  (∀ x : ℝ, x ≠ 1 → f(x) + f(1/(1-x)) = x) ∧
  (f(0) = c) ∧ (f(1) = -c) :=
by
  unfold f
  split_ifs
  sorry

end functional_equation_l75_75350


namespace geometric_arithmetic_sum_l75_75882

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a 1 * (a 1) ^ n

noncomputable def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, b (n + 1) - b n = b 2 - b 1

theorem geometric_arithmetic_sum :
  (∃ a : ℕ → ℝ, geometric_sequence a ∧ a 1 = 1 ∧ 
    (let b (n : ℕ) := if n = 0 then 4 * a 2 else if n = 1 then 2 * a 3 else a 4 in 
     arithmetic_sequence b) ∧ a 2 + a 3 + a 4 = 14) :=
begin
  sorry
end

end geometric_arithmetic_sum_l75_75882


namespace molecular_weight_of_7_moles_KBrO3_l75_75626

def potassium_atomic_weight : ℝ := 39.10
def bromine_atomic_weight : ℝ := 79.90
def oxygen_atomic_weight : ℝ := 16.00
def oxygen_atoms_in_KBrO3 : ℝ := 3

def KBrO3_molecular_weight : ℝ := 
  potassium_atomic_weight + bromine_atomic_weight + (oxygen_atomic_weight * oxygen_atoms_in_KBrO3)

def moles := 7

theorem molecular_weight_of_7_moles_KBrO3 : KBrO3_molecular_weight * moles = 1169.00 := 
by {
  -- The proof would be here, but it is omitted as instructed.
  sorry
}

end molecular_weight_of_7_moles_KBrO3_l75_75626


namespace find_legs_of_triangle_l75_75270

noncomputable def right_triangle_legs (AD BD : ℝ) (h_eqdist : ℝ) (h_hypotenuse : ℝ) : Prop :=
  ∃ (AC BC : ℝ), ∠ACB = 90 ∧ AD + BD = h_hypotenuse ∧ 
  AD = h_eqdist ∧ BD = h_eqdist ∧
  AC = 42 ∧ BC = 56

theorem find_legs_of_triangle : right_triangle_legs 30 40 70 :=
sorry

end find_legs_of_triangle_l75_75270


namespace least_positive_integer_to_multiple_of_5_l75_75227

theorem least_positive_integer_to_multiple_of_5 (n : ℕ) (h₁ : n = 725) :
  ∃ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 ∧ ∀ k : ℕ, (k > 0 ∧ (725 + k) % 5 = 0) → m ≤ k :=
begin
  use 5,
  sorry
end

end least_positive_integer_to_multiple_of_5_l75_75227


namespace ornithological_park_species_l75_75474

/-- In an ornithological park, there are 2021 birds arranged in a row.
Each pair of birds of the same species has an even number of birds between them.
Prove that the smallest number of bird species is 1011. -/
theorem ornithological_park_species (n : ℕ) (h1 : n = 2021) 
  (h2 : ∀ s : ℕ, s ∈ {1..n} → (∀ x y : ℕ, x < y ∧ x ≠ y → (∀ z : ℕ, z ∈ ({x, y} : set ℕ) → even (y - x - 1))) ) 
  : s ≥ 1011 :=
sorry

end ornithological_park_species_l75_75474


namespace ratio_of_floors_l75_75304

-- Define the number of floors of each building
def floors_building_A := 4
def floors_building_B := 4 + 9
def floors_building_C := 59

-- Prove the ratio of floors in Building C to Building B
theorem ratio_of_floors :
  floors_building_C / floors_building_B = 59 / 13 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_floors_l75_75304


namespace annualInterestRateCalc_l75_75293

noncomputable def annualInterestRateSecondCertificate
    (initialInvestment : ℝ) (initialRate : ℝ) (finalAmount : ℝ) : ℝ :=
    let firstPeriodAmount := initialInvestment * (1 + initialRate / 400)
    let equation := firstPeriodAmount * (1 + (24 : ℝ) / 400)
    calc
    initialInvestment * 1.02 * (1 + (24 : ℝ) / 400) = finalAmount : 
    by
        sorry /* The proof steps */

theorem annualInterestRateCalc :
    ∀ (initialInvestment finalAmount : ℝ)
      (initialRate s : ℝ),
    initialInvestment = 12000 -> initialRate = 8 -> finalAmount = 12980 -> s = 24 ->
    annualInterestRateSecondCertificate initialInvestment initialRate finalAmount = 12980 :=
by
    intros
    unfold annualInterestRateSecondCertificate
    calc
    initialInvestment * 1.02 * (1 + s / 400) = finalAmount :
    by
        -- Skipping the computational proof steps
        sorry

end annualInterestRateCalc_l75_75293


namespace washington_high_students_l75_75150
-- Import the necessary library

-- Define the general conditions
def student_teacher_ratio : ℚ := 27.5 -- ratio of students to one teacher
def number_of_teachers : ℕ := 42 -- number of teachers

-- Define a predicate that states the number of students is as expected
def number_of_students (ratio : ℚ) (teachers : ℕ) : ℕ := 
  (ratio * (teachers : ℚ)).to_nat

-- The theorem that we need to prove
theorem washington_high_students : 
  number_of_students student_teacher_ratio number_of_teachers = 1155 :=
sorry

end washington_high_students_l75_75150


namespace train_speed_proof_l75_75618

-- Definitions based on conditions
def length_train1 : ℝ := 1.10
def length_train2 : ℝ := 0.9
def crossing_time_in_seconds : ℝ := 40

-- Conversion from seconds to hours
def crossing_time_in_hours : ℝ := crossing_time_in_seconds / 3600

-- Total distance to be covered is the sum of the lengths of the trains
def total_length : ℝ := length_train1 + length_train2

-- Define the speed v in km/hr
def relative_speed (v : ℝ) : ℝ := 2 * v

-- Define the speed of each train based on the solution provided
def speed_of_each_train : ℝ := 2.00 / crossing_time_in_hours / 2

-- The proof statement
theorem train_speed_proof :
  (relative_speed (speed_of_each_train) = total_length / crossing_time_in_hours) ->
  speed_of_each_train = 90 :=
by
  sorry

end train_speed_proof_l75_75618


namespace bird_species_min_l75_75489

theorem bird_species_min (total_birds : ℕ) (h_total_birds : total_birds = 2021)
  (h_even_between : ∀ (species : Sort*) (a b : species), (a ≠ b) → even (nat.dist a b)) :
  ∃ species_num : ℕ, species_num = 1011 :=
by
  sorry

end bird_species_min_l75_75489


namespace range_of_a_l75_75900

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x : ℝ, f (-x) = f x) -- f is even
  (h2 : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) -- f is monotonically increasing on [0, +∞)
  (h3 : f (real.log 2 a) + f (real.log (1/2) a) < 2 * f 1) : 
  ∃ (a : ℝ), (1/2 < a ∧ a < 2) := sorry

end range_of_a_l75_75900


namespace pizza_area_increases_l75_75435

noncomputable def percent_increase (old_area new_area : ℝ) : ℝ := ((new_area - old_area) / old_area) * 100

theorem pizza_area_increases (π r_s : ℝ) :
  let r_m := 1.25 * r_s;
      r_l := 1.875 * r_s;
      A_s := π * r_s^2;
      A_m := π * (1.25 * r_s)^2;
      A_l := π * (1.875 * r_s)^2 in
  percent_increase A_s A_m = 56.25 ∧
  percent_increase A_m A_l = 195.3125 ∧
  percent_increase A_s A_l = 251.5625 :=
by {
  sorry
}

end pizza_area_increases_l75_75435


namespace domain_and_range_h_l75_75526

noncomputable def f : ℝ → ℝ := sorry
def domain_f : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def range_f : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2}
def h (x : ℝ) : ℝ := 2 - f (x - 1)
def domain_h : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}
def range_h : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2}

theorem domain_and_range_h :
  (domain_h = {x | 2 ≤ x ∧ x ≤ 4}) ∧
  (range_h = {y | 0 ≤ y ∧ y ≤ 2}) :=
sorry

end domain_and_range_h_l75_75526


namespace largest_divisor_of_seven_consecutive_odd_numbers_l75_75621

theorem largest_divisor_of_seven_consecutive_odd_numbers (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ d, d = 45 ∧ ∀ k, k ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) → k ≤ 45 :=
sorry

end largest_divisor_of_seven_consecutive_odd_numbers_l75_75621


namespace find_expression_l75_75534

noncomputable def complex_numbers := ℂ

theorem find_expression (x y z : complex_numbers) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
(h4 : x + y + z = 10) 
(h5 : (x - y)^2 + (x - z)^2 + (y - z)^2 + 6 = x * y * z) :
  (x^3 + y^3 + z^3 - 3 * x * y * z) / (x * y * z) = 5 - 30 / (x * y * z) := 
sorry

end find_expression_l75_75534


namespace number_of_leap_years_l75_75292

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0) ∧ (year % 100 ≠ 0 ∨ year % 400 = 0)

def count_leap_years (years : List ℕ) : ℕ :=
  years.countp is_leap_year

theorem number_of_leap_years :
  count_leap_years [1964, 1978, 1996, 2001, 2100] = 2 :=
by
  sorry

end number_of_leap_years_l75_75292


namespace tangent_line_equation_extreme_values_l75_75915

section

variables (a b : ℝ)
noncomputable def f (x : ℝ) := x^3 + a*x^2 + b*x + 1
noncomputable def f' (x : ℝ) := 3*x^2 + 2*a*x + b
noncomputable def g (x : ℝ) := f a b x * exp (-x)

-- given conditions
axiom h₁ : f' a b 1 = 2 * a
axiom h₂ : f' a b 2 = -b

-- statement for problem 1 (tangent line at x = 1 is 6x + 2y - 1 = 0)
theorem tangent_line_equation : 6 * 1 + 2 * f a b 1 - 1 = 0 :=
sorry

-- extreme values of g (x = 0 is minimum, g(0) = -3; x = 3 is maximum, g(3) = 15 * exp(-3))
theorem extreme_values :
  g a b 0 = -3 ∧ g a b 3 = 15 * exp (-3) :=
sorry

end

end tangent_line_equation_extreme_values_l75_75915


namespace arithmetic_geometric_mean_inequality_l75_75102

open BigOperators

noncomputable def A (a : Fin n → ℝ) : ℝ := (Finset.univ.sum a) / n

noncomputable def G (a : Fin n → ℝ) : ℝ := (Finset.univ.prod a) ^ (1 / n)

theorem arithmetic_geometric_mean_inequality (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) : A a ≥ G a :=
  sorry

end arithmetic_geometric_mean_inequality_l75_75102


namespace kite_area_l75_75866

theorem kite_area :
  let vertices := [(0, 7), (4, 10), (8, 7), (4, 0)]
  let base := 10 -- Corresponding to the horizontal distance between (0,7) and (8,7) or (4,0)
  let height := 6 -- Corresponding to the vertical distance between (4,10) and (4,0)
  2 * (1 / 2 * base * height) = 60 :=
by
  let base := 10
  let height := 6
  calc
    2 * (1 / 2 * base * height) = 2 * (1 / 2 * 10 * 6) : by rfl
                          ... = 2 * 30 : by rfl
                          ... = 60 : by rfl

end kite_area_l75_75866


namespace difference_of_two_numbers_l75_75119

theorem difference_of_two_numbers
  (L : ℕ) (S : ℕ) 
  (hL : L = 1596) 
  (hS : 6 * S + 15 = 1596) : 
  L - S = 1333 := 
by
  sorry

end difference_of_two_numbers_l75_75119


namespace bird_species_min_l75_75490

theorem bird_species_min (total_birds : ℕ) (h_total_birds : total_birds = 2021)
  (h_even_between : ∀ (species : Sort*) (a b : species), (a ≠ b) → even (nat.dist a b)) :
  ∃ species_num : ℕ, species_num = 1011 :=
by
  sorry

end bird_species_min_l75_75490


namespace max_value_f_min_value_a_l75_75910

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x)^2

theorem max_value_f :
  ∀ x, f x ≤ 2 ∧ (∃ k : ℤ, x = k * Real.pi - Real.pi / 6) → f x = 2 :=
by { sorry }

variables {A B C a b c : ℝ}

noncomputable def f' (x : ℝ) : ℝ := Real.cos (2 * x +  Real.pi / 3) + 1

theorem min_value_a
  (h1 : f' (B + C) = 3/2)
  (h2 : b + c = 2)
  (h3 : A + B + C = Real.pi)
  (h4 : Real.cos A = 1/2) :
  ∃ a, ∀ b c, a^2 = b^2 + c^2 - 2 * b * c * Real.cos A ∧ a ≥ 1 :=
by { sorry }

end max_value_f_min_value_a_l75_75910


namespace min_value_expression_l75_75853

noncomputable def expression (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_expression : ∃ x : ℝ, expression x = -6480.25 :=
sorry

end min_value_expression_l75_75853


namespace minimum_species_l75_75483

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l75_75483


namespace sum_of_integers_whose_squares_end_with_25_l75_75711

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75711


namespace geometry_problem_l75_75889

noncomputable def pointOnCircumcircle (A B C P : Point) : Prop := sorry
noncomputable def lineIntersection (L1 L2 : Line) (P : Point) : Prop := sorry
noncomputable def perpendicularBisector (A B : Point) (L : Line) : Prop := sorry

theorem geometry_problem
    (A B C P E F J K: Point)
    (h1: pointOnCircumcircle A B C P)
    (h2: lineIntersection (line C P) (line A B) E)
    (h3: lineIntersection (line B P) (line A C) F)
    (h4: perpendicularBisector A C (line J))
    (h5: lineIntersection (line J) (line A B) J)
    (h6: perpendicularBisector A B (line K))
    (h7: lineIntersection (line K) (line A C) K)
    : (dist C E)^2 / (dist B F)^2 = (dist A J) * (dist J E) / (dist A K) * (dist K F) := 
by 
  sorry

end geometry_problem_l75_75889


namespace opposite_face_of_x_l75_75830

theorem opposite_face_of_x 
    (A D F B E x : Prop) 
    (h1 : x → (A ∧ D ∧ F))
    (h2 : x → B)
    (h3 : E → D ∧ ¬x) : B := 
sorry

end opposite_face_of_x_l75_75830


namespace true_inverse_proposition_is_option_D_l75_75754

def vertical_angles_equal (α β : Type) [Group α] [Group β] : Prop :=
  sorry  -- Placeholder: Define the actual proposition for vertical angles

def greater_than_implies_square_greater (a b : ℝ) : Prop :=
  a > b → a^2 > b^2

def congruent_triangles_equal_areas (T1 T2 : Type) [Triangle T1] [Triangle T2] : Prop :=
  congruent T1 T2 → area T1 = area T2

def parallel_lines_corresponding_angles (l1 l2 : Line) : Prop :=
  parallel l1 l2 → corresponding_angles_equal l1 l2

theorem true_inverse_proposition_is_option_D (α β : Type) [Group α] [Group β] 
  (a b : ℝ) (T1 T2 : Type) [Triangle T1] [Triangle T2] (l1 l2 : Line) : Prop :=
  (¬ vertical_angles_equal α β) ∧
  (¬ greater_than_implies_square_greater a b) ∧
  (¬ congruent_triangles_equal_areas T1 T2) ∧
  (parallel_lines_corresponding_angles l1 l2)

end true_inverse_proposition_is_option_D_l75_75754


namespace parabola_focus_axis_l75_75883

-- Conditions from the problem
variables (p x₀ : ℝ)
axiom parabola_cond : y^2 = 2 * p * x
axiom focus_cond : p > 0
axiom distance_P_F : abs (x₀ - p / 2) = 10
axiom parabola_pts : (2 * p * x₀ = 36) ∧ (abs (6) = 6)

-- Question: Prove p = 2 or p = 18
theorem parabola_focus_axis (h₁ : p > 0)
  (h₂ : abs (x₀ + p / 2) = 10)
  (h₃ : 2 * p * x₀ = 36) : p = 2 ∨ p = 18 :=
by 
  sorry

end parabola_focus_axis_l75_75883


namespace price_reduction_correct_l75_75238

noncomputable def percentage_reduction (x : ℝ) : Prop :=
  (5000 * (1 - x)^2 = 4050)

theorem price_reduction_correct {x : ℝ} (h : percentage_reduction x) : x = 0.1 :=
by
  -- proof is omitted, so we use sorry
  sorry

end price_reduction_correct_l75_75238


namespace general_term_and_sum_lt_l75_75386

-- Definitions of the conditions
variable (a : ℕ → ℚ) (S : ℕ → ℚ) (b : ℕ → ℚ)

hypothesis h1 : ∀ n, S n = ∑ i in Finset.range n, a (i+1)
hypothesis h2 : ∀ n, b n = 1 / S n
hypothesis h3 : a 2 * b 2 = 5 / 8
hypothesis h4 : S 5 = 35 / 2

theorem general_term_and_sum_lt {n : ℕ} :
  (∀ k, a k = (2 * k + 1) / 2) ∧ 
  (∀ k, b k = 2 / (k * (k + 2))) ∧
  ∑ i in Finset.range (n + 1), b (i + 1) < 3 / 2 :=
by
  sorry

end general_term_and_sum_lt_l75_75386


namespace M_subsetneq_P_l75_75074

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

theorem M_subsetneq_P : M ⊂ P :=
by sorry

end M_subsetneq_P_l75_75074


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75640

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75640


namespace calculate_discount_l75_75510

theorem calculate_discount
  (original_cost : ℝ)
  (amount_spent : ℝ)
  (h1 : original_cost = 35.00)
  (h2 : amount_spent = 18.00) :
  original_cost - amount_spent = 17.00 :=
by
  sorry

end calculate_discount_l75_75510


namespace angle_in_third_quadrant_l75_75014

theorem angle_in_third_quadrant (α : Real) :
  sin (2 * α) > 0 → cos α < 0 → (π / 2 < α ∧ α < π) :=
by
  intros h1 h2
  have h_cos_neg := h2
  have h_sin2_pos := h1
  sorry

end angle_in_third_quadrant_l75_75014


namespace min_value_expr_l75_75376

theorem min_value_expr (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (m : ℝ), m = sqrt (2 - y) + sqrt (5 - 2 * x) 
        ∧ ∀ (a b : ℝ), a^2 + b^2 = 4 → sqrt (2 - b) + sqrt (5 - 2 * a) ≥ m 
        ∧ m = sqrt 5 :=
by
  -- Proof goes here
  sorry

end min_value_expr_l75_75376


namespace largest_divisor_of_seven_consecutive_odd_numbers_l75_75622

theorem largest_divisor_of_seven_consecutive_odd_numbers (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ d, d = 45 ∧ ∀ k, k ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) → k ≤ 45 :=
sorry

end largest_divisor_of_seven_consecutive_odd_numbers_l75_75622


namespace smallest_sum_of_factors_of_8_l75_75126

theorem smallest_sum_of_factors_of_8! :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a * b * c * d = Nat.factorial 8 ∧ a + b + c + d = 102 :=
sorry

end smallest_sum_of_factors_of_8_l75_75126


namespace problem_given_conditions_l75_75887

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem problem_given_conditions {S a : ℕ → ℝ} (a : ℕ → ℝ) :
  (arithmetic_sequence a (-3)) →
  (S 13 = -26) →
  (a 9 = 4) →
  (∀ n, a n = 31 - 3 * n) ∧ 
  (∀ n, ∑ k in Finset.range n, a (2 * k + 1) = -3 * n^2 + 34 * n) :=
begin
  intros seq sum_cond a9_cond,
  split,
  {
    intro n,
    -- Proof of the general term of the sequence
    sorry,
  },
  {
    intro n,
    -- Proof of the sum of series
    sorry,
  }
end

end problem_given_conditions_l75_75887


namespace augustus_makes_3_milkshakes_l75_75815

def augMilkshakePerHour (A : ℕ) (Luna : ℕ) (hours : ℕ) (totalMilkshakes : ℕ) : Prop :=
  (A + Luna) * hours = totalMilkshakes

theorem augustus_makes_3_milkshakes :
  augMilkshakePerHour 3 7 8 80 :=
by
  -- We assume the proof here
  sorry

end augustus_makes_3_milkshakes_l75_75815


namespace nine_a_plus_a_plus_nine_l75_75362

theorem nine_a_plus_a_plus_nine (A : Nat) (hA : 0 < A) : 
  10 * A + 9 = 9 * A + (A + 9) := 
by 
  sorry

end nine_a_plus_a_plus_nine_l75_75362


namespace sally_combinations_l75_75567

theorem sally_combinations :
  let wall_colors := 4
  let flooring_types := 3
  wall_colors * flooring_types = 12 := by
  sorry

end sally_combinations_l75_75567


namespace ceil_floor_eq_zero_l75_75331

theorem ceil_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (- (7 / 3)) = 0) :=
by
  sorry

end ceil_floor_eq_zero_l75_75331


namespace sum_exponents_outside_radical_l75_75572

noncomputable def problem_statement (a b c : ℝ) : ℝ :=
  (√[4] (48 * a^5 * b^8 * c^14))

theorem sum_exponents_outside_radical (a b c : ℝ) :
  let simplified_form := 2 * a * b^2 * c^3 * (√[4] (3 * a * c^2))
  (simplified_form = 2 * a * b^2 * c^3 * (√[4] (3 * a * c^2))) →
  1 + 2 + 3 = 6 :=
by
  intro h
  sorry

end sum_exponents_outside_radical_l75_75572


namespace find_integer_pairs_l75_75849

theorem find_integer_pairs (x y : ℤ) (h : x^3 - y^3 = 2 * x * y + 8) : 
  (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) := 
by {
  sorry
}

end find_integer_pairs_l75_75849


namespace max_sum_x_y_under_condition_l75_75535

-- Define the conditions
variables (x y : ℝ)

-- State the problem and what needs to be proven
theorem max_sum_x_y_under_condition : 
  (3 * (x^2 + y^2) = x - y) → (x + y) ≤ (1 / Real.sqrt 2) :=
by
  sorry

end max_sum_x_y_under_condition_l75_75535


namespace alternating_geometric_sum_eq_neg684_l75_75828

-- Define the alternating geometric series
def alternatingGeometricSum (n : ℕ) : ℤ :=
  ∑ k in finset.range n, (-1)^(k+1) * (2^k)

-- The statement of the problem
theorem alternating_geometric_sum_eq_neg684 : alternatingGeometricSum 10 = -684 := by
  sorry

end alternating_geometric_sum_eq_neg684_l75_75828


namespace toilet_paper_packs_needed_l75_75096

-- Definitions based on conditions
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def weeks : ℕ := 4
def rolls_per_pack : ℕ := 12
def daily_stock : ℕ := 1

-- The main theorem statement
theorem toilet_paper_packs_needed : 
  (bathrooms * days_per_week * weeks) / rolls_per_pack = 14 := by
sorry

end toilet_paper_packs_needed_l75_75096


namespace proof_problem_l75_75266

noncomputable def problem (A B C I M N K L : Point) : Prop :=
  let line_through_I := line_passing_through I
  let meets_AB_at_M := line_meeting_side line_through_I AB M
  let meets_BC_at_N := line_meeting_side line_through_I BC N
  let acute_triangle_BMN := acute_triangle B M N
  let chosen_points_on_AC := chosen_points K L AC
  let angle_ILA_eq_IMB := angle_eq (angle I L A) (angle I M B)
  let angle_KCI_eq_INB := angle_eq (angle K C I) (angle I N B)
  let equality_to_prove := AM + KL + CN = AC
  in meets_AB_at_M ∧ meets_BC_at_N ∧ acute_triangle_BMN ∧ chosen_points_on_AC ∧ angle_ILA_eq_IMB ∧ angle_KCI_eq_INB → equality_to_prove

theorem proof_problem {A B C I M N K L : Point} : problem A B C I M N K L := sorry

end proof_problem_l75_75266


namespace sum_of_squares_ending_in_25_l75_75724

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75724


namespace percent_increase_l75_75301

variable (P : ℝ)

theorem percent_increase (h1 : 1.30 * P = P * 1.30) (h2 : 1.50 * P = P * 1.50) :
  ((1.50 * P - 1.30 * P) / (1.30 * P)) * 100 ≈ 15.38 :=
by
  sorry

end percent_increase_l75_75301


namespace largest_integer_binom_eq_l75_75203

theorem largest_integer_binom_eq :
  ∃ n : ℕ, (n = 7) ∧ (binom 11 n = binom 10 3 + binom 10 4) :=
by sorry

end largest_integer_binom_eq_l75_75203


namespace function_range_l75_75363

def f (x : ℝ) : ℝ := Real.arctan (Real.sqrt (6 * (Real.log (λ z => 1/z) (0.25)⁻¹ (λ z => (2:ℝ) * z) (dec_trivial : -((1:ℝ)/16) < sin x / (sin x + 15) ≤ (1:ℝ)/16)))) 

theorem function_range : ∀ x : ℝ, 0 < f x ∧ f x ≤ Real.pi / 3 :=
sorry

end function_range_l75_75363


namespace dilation_0_minus_2i_to_neg3_minus_14i_l75_75123

open Complex

def dilation_centered (z_center z zk : ℂ) (factor : ℝ) : ℂ :=
  z_center + factor * (zk - z_center)

theorem dilation_0_minus_2i_to_neg3_minus_14i :
  dilation_centered (1 + 2 * I) (0 - 2 * I) (1 + 2 * I) 4 = -3 - 14 * I :=
by
  sorry

end dilation_0_minus_2i_to_neg3_minus_14i_l75_75123


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75701

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75701


namespace greatest_n_for_segments_l75_75995
open Nat

def coprime (a b : ℕ) : Prop := gcd a b = 1

theorem greatest_n_for_segments 
  (n : ℕ) 
  (h : 2 ≤ n) 
  (positioning : ∀ a b : ℕ, (1 ≤ a ∧ a ≤ n) ∧ (1 ≤ b ∧ b ≤ n) → a ≠ b → coprime a b → ¬coprime a b → (p_s : ℕ) → (d_s : ℕ) → p_s ≤ abs (d_s)) :
  n ≤ 11 :=
sorry

end greatest_n_for_segments_l75_75995


namespace minimum_species_count_l75_75467

theorem minimum_species_count {n : ℕ} (h_n : n = 2021) 
  (h_cond : ∀ i j k : ℕ, i < j ∧ j < k → 
    birds i = birds k → birds j ≠ birds i → (j - i - 1) % 2 = 1 ∧ (k - j - 1) % 2 = 1) : 
  ∃ s : ℕ, s ≥ 1011 :=
begin
  sorry
end

end minimum_species_count_l75_75467


namespace max_distance_from_point_to_line_l75_75378

noncomputable def point : Type := ℝ × ℝ

def P : point := (-2, 0)

def line (λ : ℝ) : point → Prop := 
  λ ⟨x,y⟩ => (1 + 3 * λ) * x + (1 + 2 * λ) * y - (2 + 5 * λ) = 0

def maxDistance (P : point) (l : (ℝ → point → Prop)) : ℝ := 
  let Q := (1, 1)  -- since we solve the system and find Q (1,1)
  (Real.sqrt ((-2 - 1)^2 + (0 - 1)^2))

theorem max_distance_from_point_to_line : 
  ∀ λ : ℝ, ∃ Q : point, (line λ Q) → maxDistance P (line λ) = Real.sqrt 10 :=
by
  sorry

end max_distance_from_point_to_line_l75_75378


namespace probability_calculator_l75_75283

def probabilities_equal (num_players : Nat) (coins : Nat) (rounds : Nat) (prob : Rat) : Prop :=
  num_players = 4 ∧ coins = 5 ∧ rounds = 5 ∧ prob = 1 / 3200000

def initial_conditions (num_players coins rounds : Nat) (prob : Rat) : Prop :=
  num_players = 4 ∧ coins = 5 ∧ rounds = 5 

theorem probability_calculator (num_players coins rounds : Nat) :
  initial_conditions num_players coins rounds 1 / 3200000 →
  probabilities_equal num_players coins rounds (1 / 3200000) :=
by
  intros h
  cases h with hn hc hr
  sorry

end probability_calculator_l75_75283


namespace balance_scale_weights_part_a_balance_scale_weights_part_b_l75_75606

-- Part (a)
theorem balance_scale_weights_part_a (w : List ℕ) (h : w = List.range (90 + 1) \ List.range 1) :
  ¬ ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

-- Part (b)
theorem balance_scale_weights_part_b (w : List ℕ) (h : w = List.range (99 + 1) \ List.range 1) :
  ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

end balance_scale_weights_part_a_balance_scale_weights_part_b_l75_75606


namespace min_num_groups_l75_75136

theorem min_num_groups : 
  (∀ g ∈ groups, (∀ a b ∈ g, (a + b) % 6 = 0)) → 
  ∃ g₀ g₃ rest, 
    g₀ ∪ g₃ ∪ rest = (finset.range 600).filter (λ n, n + 1) ∧ 
    (∀ a b ∈ g₀, (a + b) % 6 = 0) ∧ 
    (∀ a b ∈ g₃, (a + b) % 6 = 0) ∧ 
    (∀ g ∈ rest, (∀ a b ∈ g, (a + b) % 6 = 0) ∧ |g| = 2) ∧
    g₀.card = (600 / 6) ∧ 
    g₃.card = (600 / 6) ∧ 
    rest.card = 200 * 2 → 
    g₀.card + g₃.card + rest.card / 2 = 202 :=
by sorry

end min_num_groups_l75_75136


namespace b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l75_75576

variable (a b : ℕ)

-- Conditions
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k
def is_multiple_of_10 (n : ℕ) : Prop := ∃ k : ℕ, n = 10 * k

-- Given conditions in the problem
axiom h_a : is_multiple_of_5 a
axiom h_b : is_multiple_of_10 b

-- Statements to be proved
theorem b_is_multiple_of_5 : is_multiple_of_5 b :=
sorry

theorem a_plus_b_is_multiple_of_5 : is_multiple_of_5 (a + b) :=
sorry

end b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l75_75576


namespace rectangle_cut_l75_75838

def dimensions_ratio (x y : ℕ) : Prop := ∃ (r : ℚ), x = r * y

theorem rectangle_cut (k m n : ℕ) (hk : ℝ) (hm : ℝ) (hn : ℝ) 
  (h1 : k + m + n = 10) 
  (h2 : k * 9 / 10 = hk)
  (h3 : m * 9 / 10 = hm)
  (h4 : n * 9 / 10 = hn)
  (h5 : hk + hm + hn = 9) :
  ∃ (k' m' n' : ℕ), 
    dimensions_ratio k k' ∧ 
    dimensions_ratio m m' ∧
    dimensions_ratio n n' ∧
    k ≠ m ∧ m ≠ n ∧ k ≠ n :=
sorry

end rectangle_cut_l75_75838


namespace terminal_side_third_quadrant_l75_75405

theorem terminal_side_third_quadrant (α : ℝ) (k : ℤ) 
  (hα : (π / 2) + 2 * k * π < α ∧ α < π + 2 * k * π) : 
  ¬(π + 2 * k * π < α / 3 ∧ α / 3 < (3 / 2) * π + 2 * k * π) :=
by
  sorry

end terminal_side_third_quadrant_l75_75405


namespace triangle_median_angle_l75_75979

theorem triangle_median_angle (A B C D : Type*) [IsTriangle A B C]
  (h1 : IsMedian B D A C)
  (h2 : length B D = length A C / 2) :
  angle B = 90 := 
by 
   sorry

end triangle_median_angle_l75_75979


namespace fraction_students_study_japanese_l75_75299

theorem fraction_students_study_japanese (J S : ℕ) (h1 : S = 3 * J) 
(h2 : ∃ k : ℕ, k = (1/3 : ℚ) * S) (h3 : ∃ l : ℕ, l = (3/4 : ℚ) * J) :
  (∃ f : ℚ, f = ((1/3 : ℚ) * S + (3/4 : ℚ) * J) / (S + J) ∧ f = 7/16) :=
by
  sorry

end fraction_students_study_japanese_l75_75299


namespace lottery_probability_l75_75449

-- Definitions of the conditions
def MegaBallCount : ℕ := 30
def WinnerBallCount : ℕ := 50
def WinningCombinationCount : ℕ := Nat.choose 50 6

-- Main theorem
theorem lottery_probability :
  (1 / MegaBallCount : ℚ) * (1 / WinningCombinationCount : ℚ) = 1 / 477621000 := by
  -- Computation of the binomial coefficient
  have binom_50_6 : Nat.choose 50 6 = 15890700 := by
    rw [Nat.choose_eq_factorial_div_factorial]
    norm_num
  sorry

end lottery_probability_l75_75449


namespace number_of_ways_to_choose_4_captains_from_15_l75_75950

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end number_of_ways_to_choose_4_captains_from_15_l75_75950


namespace problem_statement_l75_75118

open Real

def lines_perpendicular (a : ℝ) : Prop :=
  let line1 := a * x + (2 - a) * y = 0 in
  let line2 := x - a * y = 1 in
  line1 ∧ line2

def is_sufficient_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, lines_perpendicular a → 
  (a = 1 → (x = 0 ∨ y = 0))

theorem problem_statement :
  is_sufficient_condition 1 :=
sorry

end problem_statement_l75_75118


namespace abc_geq_expression_l75_75396

variable (a b c : ℝ) -- Define variables a, b, c as real numbers
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) -- Define conditions of a, b, c being positive

theorem abc_geq_expression : 
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
by 
  sorry -- Proof goes here

end abc_geq_expression_l75_75396


namespace men_absent_l75_75795

theorem men_absent (x : ℕ) :
  let original_men := 42
  let original_days := 17
  let remaining_days := 21 
  let total_work := original_men * original_days
  let remaining_men_work := (original_men - x) * remaining_days 
  total_work = remaining_men_work →
  x = 8 :=
by
  intros
  let total_work := 42 * 17
  let remaining_men_work := (42 - x) * 21
  have h : total_work = remaining_men_work := ‹total_work = remaining_men_work›
  sorry

end men_absent_l75_75795


namespace max_value_of_fraction_l75_75505

noncomputable def triangle_sides_max_value (a b c : ℝ) : ℝ :=
  (1 / (b + 1)) + (9 / (a + 9))

variables {a b c : ℝ} (A B C : ℝ) 
hypothesis h1: a > 0
hypothesis h2: b > 0
hypothesis h3: c > 0
hypothesis h4: a = 2 * b - c
hypothesis h5: 1/2 * a * b * sin C = sqrt 3
hypothesis h6: cos C / cos B = c / (2*a - b)

theorem max_value_of_fraction : 
  ∃ a b c, h1 ∧ h2 ∧ h3 ∧ h4 ∧ h5 ∧ h6 ∧ triangle_sides_max_value a b c = 3 / 5 :=
begin
  sorry
end

end max_value_of_fraction_l75_75505


namespace third_player_always_loses_l75_75592

structure Game :=
  (k : ℕ) -- the number of layers in the hexagonal grid
  (turns : ℕ) -- the total number of turns possible

inductive Player
| Ali
| Shayan
| Sajad

def can_collaborate_to_make_third_player_lose (g : Game) : Prop :=
  ∃ (strategy : Player → ℕ → (ℕ × ℕ)),  -- strategy as a function that takes a Player and turn number to a position on the grid
    ∀ n,  -- for all turns
      n % 3 ≠ 2 →  -- if it's Ali's or Shayan's turn
        strategy Player.Ali n = strategy Player.Shayan n -- players Ali and Shayan use a coordinated strategy

theorem third_player_always_loses : 
  ∀ (g : Game),
  ∃ (strategy : Player → ℕ → (ℕ × ℕ)),
    ∀ n, n % 3 ≠ 2 → strategy Player.Ali n = strategy Player.Shayan n :=
by
  intro g
  exists λ p n, (n, n) -- dummy strategy, needs refinement
  intro n h
  sorry

end third_player_always_loses_l75_75592


namespace average_subset_size_l75_75545

theorem average_subset_size (n : ℕ) : 
  let S := {k | k ∈ Finset.range (n + 1)}
  let subsets := {t | t ⊆ S}
  (∑ i in Finset.range (n + 1), (1 : ℚ) / 2).toReal = n / 2 :=
by
  let S := {k | k ∈ Finset.range (n + 1)}
  let subsets := {t | t ⊆ S}
  sorry

end average_subset_size_l75_75545


namespace find_max_pqr_l75_75058

open Matrix

-- Definitions for our specific problem
def B (p q r : ℤ) : Matrix (Fin 2) (Fin 2) ℚ :=
  (1 / 7 : ℚ) • ![![(-5 : ℤ), (p : ℤ)],![(q : ℤ), (r : ℤ)]]

-- Define the condition
def condition (p q r : ℤ) (B : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  B * B = 1

-- Final goal to prove
theorem find_max_pqr : ∃ (p q r : ℤ), condition p q r (B p q r) ∧ (p + q + r) = 30 := by
  sorry

end find_max_pqr_l75_75058


namespace tan_theta_value_l75_75163

noncomputable def tan_theta_satisfies : ℝ :=
  if θ : ℝ, 0 < θ ∧ θ < π / 4 ∧ (Real.tan θ + Real.tan (2 * θ) + Real.tan (3 * θ) = 0) then Real.tan θ else 0

theorem tan_theta_value (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4) (h : Real.tan θ + Real.tan (2 * θ) + Real.tan (3 * θ) = 0) :
     tan_theta_satisfies = 1 / Real.sqrt 2 :=
  sorry

end tan_theta_value_l75_75163


namespace det_Q_is_zero_l75_75521

-- Define the vector
def v : ℝ × ℝ := (3, -5)

-- Define the projection matrix Q
def Q : Matrix (Fin 2) (Fin 2) ℝ :=
  let c := v.1 * v.1 + v.2 * v.2
  (1 / c) • Matrix.of ![
    ![v.1 * v.1, v.1 * v.2],
    ![v.2 * v.1, v.2 * v.2]
  ]

-- Statement to prove that det Q = 0
theorem det_Q_is_zero : det Q = 0 :=
by
  sorry

end det_Q_is_zero_l75_75521


namespace minimum_bird_species_l75_75459

theorem minimum_bird_species (total_birds : ℕ) (h : total_birds = 2021) :
  ∃ (min_species : ℕ), min_species = 1011 ∧ 
  (∀ (species_array : array total_birds ℕ),
   ∀ i j : fin total_birds, 
   species_array[i] = species_array[j] → ((i ≠ j) →
   (abs (i - j) mod 2 = 0))) :=
sorry

end minimum_bird_species_l75_75459


namespace john_paid_after_tax_l75_75988

-- Definitions based on problem conditions
def original_cost : ℝ := 200
def tax_rate : ℝ := 0.15

-- Definition of the tax amount
def tax_amount : ℝ := tax_rate * original_cost

-- Definition of the total amount paid
def total_amount_paid : ℝ := original_cost + tax_amount

-- Theorem statement for the proof
theorem john_paid_after_tax : total_amount_paid = 230 := by
  sorry

end john_paid_after_tax_l75_75988


namespace length_limit_l75_75799

-- Define the partial sums of the two infinite geometric series
def first_series_sum (n : ℕ) : ℝ := 2 * (1 - (1 / 3)^n) / (1 - 1 / 3)

def second_series_sum (n : ℕ) : ℝ := (1 / 3) * (1 - (1 / 3)^n) / (1 - 1 / 3) * (Float.sqrt 3)

-- Define the total length of the line given by the infinite series
noncomputable def total_length (n : ℕ) : ℝ := first_series_sum n + second_series_sum n

-- Define the limit of the series as n -> ∞
noncomputable def limit_length : ℝ := 3 + (Float.sqrt 3) / 2

-- Main theorem: the limit of the length of the line when the growth process continues forever
theorem length_limit : ∀ n, total_length n = limit_length :=
by
  sorry

end length_limit_l75_75799


namespace choose_4_captains_from_15_l75_75968

def num_ways_to_choose_captains : ℕ := 15.choose 4

theorem choose_4_captains_from_15 : num_ways_to_choose_captains = 1365 := by
  sorry

end choose_4_captains_from_15_l75_75968


namespace largest_integer_n_l75_75852

theorem largest_integer_n (n : ℕ) : (∀ x : ℝ, sin x ^ n + cos x ^ n ≥ 1 / n) → n ≤ 8 :=
by sorry

end largest_integer_n_l75_75852


namespace smallest_n_exists_l75_75750

theorem smallest_n_exists :
  ∃ (a1 a2 a3 a4 a5 : ℤ), a1 + a2 + a3 + a4 + a5 = 1990 ∧ a1 * a2 * a3 * a4 * a5 = 1990 :=
sorry

end smallest_n_exists_l75_75750


namespace average_snowfall_per_hour_l75_75937

theorem average_snowfall_per_hour (total_snowfall : ℕ) (hours_per_week : ℕ) (total_snowfall_eq : total_snowfall = 210) (hours_per_week_eq : hours_per_week = 7 * 24) : 
  total_snowfall / hours_per_week = 5 / 4 :=
by
  -- skip the proof
  sorry

end average_snowfall_per_hour_l75_75937


namespace log_simplified_l75_75232

-- Define the logarithmic expressions using Lean's log base and the given problem's conditions.
def log_base_2 := Real.log 2
def log16_32 := Real.log (2^5) / Real.log (2^4)
def log16_half := Real.log (2^(-1)) / Real.log (2^4)

-- The proof statement, expressing the simplified division.
theorem log_simplified : log16_32 / log16_half = -5 := by
  sorry

end log_simplified_l75_75232


namespace ceil_floor_eq_zero_l75_75330

theorem ceil_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (- (7 / 3)) = 0) :=
by
  sorry

end ceil_floor_eq_zero_l75_75330


namespace largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75210

open Nat

theorem largest_n_binom_10_3_10_4_eq_binom_11_n :
  ∃ n, (n ≤ 11) ∧ (∑ i in finset.range 11, if i = 4 then 1 else 0) = 1 ∧
  (n = 7) := by
sorry

end largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75210


namespace sophomore_sample_count_correct_l75_75787

-- Define the constants given in the problem
def total_students := 4500
def freshman_students := 1200
def sophomore_students := 1500
def senior_students := 1800
def total_sample_size := 600

-- Define the required calculation
def proportion_sophomores := (sophomore_students: ℝ) / (total_students: ℝ)
def expected_sophomore_sample := proportion_sophomores * (total_sample_size: ℝ)

-- The proof statement
theorem sophomore_sample_count_correct : expected_sophomore_sample = 200 := by
  sorry

end sophomore_sample_count_correct_l75_75787


namespace swimming_pool_volume_l75_75003

def radius (diameter : ℝ) : ℝ := diameter / 2

def volume_of_cylinder (r : ℝ) (h : ℝ) : ℝ := π * r^2 * h

theorem swimming_pool_volume :
  let diameter := 20
  let depth := 5
  let r := radius diameter
  volume_of_cylinder r depth = 500 * π :=
by
  sorry -- Proof will be provided here

end swimming_pool_volume_l75_75003


namespace ellipse_eq_line_eq_l75_75402

-- Definition of the ellipse
def isEllipse (a b : ℝ) (x y : ℝ) := (x^2) / (a^2) + (y^2) / (b^2) = 1
-- Definition that given point O, F, and M
def isIsoscelesRight (O M F : ℝ × ℝ) := 
  let (ox, oy) := O in
  let (mx, my) := M in
  let (fx, fy) := F in
  (ox = 0 ∧ oy = 0) ∧ (mx = 0 ∧ my = 1) ∧ (fx = 1 ∧ fy = 0) ∧ ((fx - ox)^2 + (fy - oy)^2 = (mx - ox)^2 + (my - oy)^2)

-- Proof Problem 1: Equation of the ellipse
theorem ellipse_eq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (H : isIsoscelesRight (0,0) (0,1) (1,0)) : 
  isEllipse (√2) 1 = λ x y, (x^2) / 2 + (y^2) / 1 = 1 :=
sorry

-- Proof Problem 2: Equation of the line
theorem line_eq (l : ℝ → ℝ) (H : isEllipse (√2) 1) : 
  ∃ m, l = λ x, x - 4/3 :=
sorry

end ellipse_eq_line_eq_l75_75402


namespace simplify_fraction_addition_l75_75105

theorem simplify_fraction_addition : (3 : ℚ) / 462 + 13 / 42 = 73 / 231 :=
by
  sorry

end simplify_fraction_addition_l75_75105


namespace rowboat_time_ratio_l75_75157

theorem rowboat_time_ratio
  (V_b : ℝ) (V_s : ℝ) (D : ℝ)
  (h1: V_b = 24)
  (h2: V_s = 8)
  (h3: D > 0) : 
  (T_up / T_down) = (2 : ℝ) :=
by
  let V_up := V_b - V_s,
  let V_down := V_b + V_s,
  have T_up := D / V_up,
  have T_down := D / V_down,
  have h4: V_up = 16 := by linarith,
  have h5: V_down = 32 := by linarith,
  rw [h4, h5],
  field_simp,
  ring,
  sorry

end rowboat_time_ratio_l75_75157


namespace expected_mixed_pairs_expected_attempts_expected_attempts_approx_l75_75445

-- Define the conditions for the problem:
variable (n : ℕ)
noncomputable def harmonicNumber (n : ℕ) : ℝ := ∑ k in finset.range (n+1), 1 / (k+1 : ℝ)

-- Define the statements for the expected number of mixed pairs in part a)
theorem expected_mixed_pairs :
  (n : ℕ) → (n > 0) → ℝ :=
  λ n hn, (n^2 : ℝ) / (2 * n - 1)

-- Define the statements for the expected number of attempts in part b)
theorem expected_attempts :
  (n : ℕ) → (n > 0) → ℝ :=
  λ n hn, 2 * n - harmonicNumber n

-- Using the definitions, we can state the expected number of attempts approximately:
theorem expected_attempts_approx :
  (n : ℕ) → (n > 0) → ℝ :=
  λ n hn, 2 * n - real.log n - real.eulerMascheroni_constant

end expected_mixed_pairs_expected_attempts_expected_attempts_approx_l75_75445


namespace intercept_sum_eq_26_l75_75248

theorem intercept_sum_eq_26 :
  (∃ (x0 y0 : ℤ), 
    (5 * x0 ≡ 2 [MOD 25]) ∧ 
    (3 * y0 + 2 ≡ 0 [MOD 25]) ∧ 
    (0 ≤ x0 ∧ x0 < 25) ∧ 
    (0 ≤ y0 ∧ y0 < 25) ∧ 
    (x0 + y0 = 26)) := by
  sorry

end intercept_sum_eq_26_l75_75248


namespace largest_integer_binom_eq_l75_75200

theorem largest_integer_binom_eq :
  ∃ n : ℕ, (n = 7) ∧ (binom 11 n = binom 10 3 + binom 10 4) :=
by sorry

end largest_integer_binom_eq_l75_75200


namespace mutually_exclusive_union_probability_l75_75930

variable (Ω : Type) [ProbabilitySpace Ω]
variables (A B : Event Ω)

theorem mutually_exclusive_union_probability
  (h1 : disjoint A B) 
  (h2 : ProbabilityMeasure.probability A = 0.5)
  (h3 : ProbabilityMeasure.probability B = 0.3) :
  ProbabilityMeasure.probability (A ∪ B) = 0.8 :=
by
  sorry

end mutually_exclusive_union_probability_l75_75930


namespace isosceles_triangle_base_length_l75_75902

theorem isosceles_triangle_base_length (P Q : ℕ) (x y : ℕ) (hP : P = 15) (hQ : Q = 12) (hPerimeter : 2 * x + y = 27) 
      (hCondition : (y = P ∧ (1 / 2) * x + x = P) ∨ (y = Q ∧ (1 / 2) * x + x = Q)) : 
  y = 7 ∨ y = 11 :=
sorry

end isosceles_triangle_base_length_l75_75902


namespace sum_of_products_l75_75401

variable {a : Fin 10 → ℕ}

def valid_sequence (a : Fin 10 → ℕ) : Prop :=
  (∀ i, a i > 0 ∧ a i ≤ 21) ∧ (∀ i j, i ≠ j → a i + a j ≠ 21)

theorem sum_of_products :
  (∑ (s : Finset (Fin 10 → ℕ)) in (Finset.univ.filter valid_sequence),
     ∏ i, s.val i
  ) = (11! * 21^10) :=
sorry

end sum_of_products_l75_75401


namespace find_number_l75_75268

theorem find_number (x : ℝ) (h : x / 0.04 = 25) : x = 1 := 
by 
  -- the steps for solving this will be provided here
  sorry

end find_number_l75_75268


namespace calculate_expression_l75_75823

theorem calculate_expression :
  [81^(-0.25) + ((3^3 / 8)^(-1 / 3))]^(1 / 2) + (1 / 2) * log 4 - log (1 / 5) = 2 :=
sorry

end calculate_expression_l75_75823


namespace yeast_culturing_correct_operation_l75_75235

theorem yeast_culturing_correct_operation :
  (∀ (c1 c2 c3 c4 c5 : Prop),
    c1 ∧ c2 ∧ c3 ∧ c4 ∧ c5 → 
    (c1 → (c2 → (c3 → (c4 → (c5 → "To facilitate the counting of yeast, the culture medium in the later stages of culturing should be diluted before counting")))))) :=
begin
  intros c1 c2 c3 c4 c5 h,
  exact "To facilitate the counting of yeast, the culture medium in the later stages of culturing should be diluted before counting", -- the conclusion that follows from the given conditions
end

end yeast_culturing_correct_operation_l75_75235


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75682

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75682


namespace smallest_positive_m_l75_75612

def f (x : ℝ) : ℝ := (3/2) * Real.cos (2 * x) + (Real.sqrt 3 / 2) * Real.sin (2 * x)

def g (m x : ℝ) : ℝ := f (x + m)

theorem smallest_positive_m (m : ℝ) :
  (∀ x, g m (-x) = g m x) → (m > 0) → (∃ k : ℤ, m = (Real.pi / 12) + k * (Real.pi / 4)) :=
by
  intros hsymm mpos
  -- proof goes here
  sorry

end smallest_positive_m_l75_75612


namespace problem_proof_l75_75500

open Real

-- Definitions and conditions from the problem
def C1 (x y : ℝ) : Prop := (x + sqrt 3) ^ 2 + y ^ 2 = 4

def C2 (θ : ℝ) : ℝ × ℝ :=
  let x := 2 + 2 * cos θ
  let y := 2 * sin θ
  (x, y)

def C3 (θ : ℝ) : Prop := θ = π / 3

-- Proof goal
theorem problem_proof :
  (∀ ρ θ, C1 (ρ * cos θ - sqrt 3) (ρ * sin θ) ↔ ρ^2 + 2*sqrt 3 * ρ * cos θ - 1 = 0) ∧
  (∀ θ, let (x, y) := C2 θ in (x - 2)^2 + y^2 = 4) ∧
  ∃ A B : ℝ × ℝ, (let O := (0 : ℝ, 0 : ℝ) in 
      A = O ∧
      dist A B = 2 ∧
      dist A (sqrt 3, 0) = sqrt 3 ∧ 
      ∠ O A (sqrt 3, 0) = 2 * π / 3 ∧
      ∃ S, S = 3 / 2) := sorry

end problem_proof_l75_75500


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75745

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75745


namespace inequality_not_hold_l75_75431

theorem inequality_not_hold (x y : ℝ) (h : x > y) : ¬ (1 - x > 1 - y) :=
by
  -- condition and given statements
  sorry

end inequality_not_hold_l75_75431


namespace original_number_l75_75546

theorem original_number (y : ℚ) (h : 1 - (1 / y) = 5 / 4) : y = -4 :=
sorry

end original_number_l75_75546


namespace hydrangea_cost_l75_75104

/--
Simon wanted to buy flowers that his mom could plant for Mother's Day. The garden center was offering 10% off all purchases. He bought 5 pansies at $2.50 each, one hydrangea, and 5 petunias that cost $1.00 each. He paid with a $50 bill and received $23 back as change. Prove that the cost of the hydrangea is $12.50.
-/
theorem hydrangea_cost
  (pansy_price : ℝ)
  (pansy_count : ℕ)
  (petunia_price : ℝ)
  (petunia_count : ℕ)
  (hydrangea_cost : ℝ)
  (total_cost : ℝ)
  (discount_rate : ℝ)
  (cash_given : ℝ)
  (change_received : ℝ)
  (total_with_discount : ℝ)
  (original_price : ℝ)
  (total_price_paid : ℝ)
  (final_cost_hydrangea : ℝ)
  (h1 : ∀n price, (5 : ℕ) = n → ($2.50 : ℝ) = price → pansy_count = n ∧ pansy_price = price = true)
  (h2 : ∀n price, (5 : ℕ) = n → ($1.00 : ℝ) = price → petunia_count = n ∧ petunia_price = price = true)
  (h3 : total_cost = $17.50)
  (h4 : discount_rate = 0.10)
  (h5 : cash_given = $50)
  (h6 : change_received = $23)
  (h7 : total_with_discount = $27)
  (h8 : original_price - (discount_rate * original_price) = total_with_discount)
  (h9 : original_price = $30)
  (h10 : total_price_paid = $30)
  (h11 : final_cost_hydrangea = total_price_paid - total_cost)
  : final_cost_hydrangea = $12.50 :=
sorry

end hydrangea_cost_l75_75104


namespace fruit_bowl_apples_l75_75253

theorem fruit_bowl_apples (A : ℕ) (total_oranges initial_oranges remaining_oranges : ℕ) (percentage_apples : ℝ) :
  total_oranges = 20 →
  initial_oranges = total_oranges →
  remaining_oranges = initial_oranges - 14 →
  percentage_apples = 0.70 →
  percentage_apples * (A + remaining_oranges) = A →
  A = 14 :=
by 
  intro h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end fruit_bowl_apples_l75_75253


namespace sum_of_integers_whose_squares_end_with_25_l75_75710

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75710


namespace millet_proportion_exceeds_sixty_l75_75323

-- Definitions based on conditions in the problem
def quart_seed (millet_percentage : ℝ) := 0.3
def consume_rate (millet_rate other_rate : ℝ) := (0.7, 0.5)

-- State the problem: Prove that on Wednesday (Day 4), the proportion of millet seeds exceeds 60%.
theorem millet_proportion_exceeds_sixty :
  let initial_seed := 1 -- initial quart of seed mix
  let daily_addition := (0.3, 0.7) -- millet and other seeds added daily
  let consumption := (0.7, 0.5) -- consumption rates of millet and other seeds respectively
  let day1_millet := 0.3 * initial_seed + daily_addition.1
  let day2_millet := 0.7 * day1_millet + daily_addition.1
  let day3_millet := 0.7 * day2_millet + daily_addition.1
  let day4_millet := 0.7 * day3_millet + daily_addition.1
  let day1_other := 0.7 * initial_seed
  let day2_other := 0.5 * day1_other + daily_addition.2
  let day3_other := 0.5 * day2_other + daily_addition.2
  let day4_other := 0.5 * day3_other + daily_addition.2
  (day4_millet / (day4_millet + day4_other)) * 100 > 60 :=
by
  let initial_seed := 1
  let daily_addition := (0.3, 0.7)
  let consumption := (0.7, 0.5)
  let day1_millet := 0.3 * initial_seed
  let day2_millet := 0.7 * day1_millet + daily_addition.1
  let day3_millet := 0.7 * day2_millet + daily_addition.1
  let day4_millet := 0.7 * day3_millet + daily_addition.1
  let day1_other := 0.7 * initial_seed
  let day2_other := 0.5 * day1_other + daily_addition.2
  let day3_other := 0.5 * day2_other + daily_addition.2
  let day4_other := 0.5 * day3_other + daily_addition.2
  have h : (day4_millet / (day4_millet + day4_other)) * 100 > 60 := sorry
  exact h

end millet_proportion_exceeds_sixty_l75_75323


namespace largest_integer_binom_eq_l75_75204

theorem largest_integer_binom_eq :
  ∃ n : ℕ, (n = 7) ∧ (binom 11 n = binom 10 3 + binom 10 4) :=
by sorry

end largest_integer_binom_eq_l75_75204


namespace find_a_l75_75586

-- Define the expansion of the given expression and the condition on the constant term
def binomial_expansion_constant_term (a : ℝ) : ℝ :=
  let T := λ r, (binomial 5 r) * a^r * (5 - 5*r)/2
  T 1

theorem find_a (a : ℝ) (h : binomial_expansion_constant_term a = 10) : a = 2 :=
  by 
    sorry

end find_a_l75_75586


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75746

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75746


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75730

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75730


namespace find_length_CE_l75_75980

-- Define the main variables and parameters of the problem
variables (A B C E F : Type)
variables (AC AB CB : ℝ)
variables (mid : A → B → F)

-- Given conditions as Lean definitions
def is_bisector_of_external_angle (l : A → B) (C : Type) : Prop :=
  sorry -- As definition for actual bisector as per external angle C.

def is_parallel (l1 l2 : A → B) : Prop :=
  sorry -- As definition for l1 parallel to l2.

-- Midpoint definition
def midpoint (A B : Type) : Type :=
  ∃ F, mid A B F 


-- Main theorem to prove CE = 11/2
theorem find_length_CE (AC AB CB : ℝ) (h1 : AC > AB) 
  (h2 : is_bisector_of_external_angle l C) 
  (h3 : is_parallel (A → midpoint A B) l)
  (h4 : AC = 7)
  (h5 : CB = 4) :
  CE = 11/2 :=
sorry

end find_length_CE_l75_75980


namespace find_range_of_m_l75_75867
  
theorem find_range_of_m (m : ℝ) : (0.7^1.3)^m < (1.3^0.7)^m ↔ m > 0 := 
by 
  sorry

end find_range_of_m_l75_75867


namespace Randy_biscuits_l75_75559

theorem Randy_biscuits
  (biscuits_initial : ℕ)
  (father_gift : ℕ)
  (mother_gift : ℕ)
  (brother_eat : ℕ) :
  biscuits_initial = 32 →
  father_gift = 13 →
  mother_gift = 15 →
  brother_eat = 20 →
  biscuits_initial + father_gift + mother_gift - brother_eat = 40 :=
by
  intros h_initial h_father h_mother h_brother
  rw [h_initial, h_father, h_mother, h_brother]
  norm_num
  sorry

end Randy_biscuits_l75_75559


namespace max_vertices_l75_75324

def vertices_condition (x y z : ℕ) : Prop :=
  x + y + z = 30

def no_same_line_or_side (a b c : ℕ × ℕ × ℕ → Prop) : Prop :=
  ∀ v₁ v₂, v₁ ≠ v₂ → (a v₁.1 v₂.1 ∨ b v₁.2 v₂.2 ∨ c v₁.3 v₂.3)

theorem max_vertices (n : ℕ) :
  (∃ (V : Finset (ℕ × ℕ × ℕ)),
    (∀ (v ∈ V), vertices_condition v.1 v.2 v.3) ∧
    no_same_line_or_side (λ x₁ x₂, x₁ ≠ x₂) (λ y₁ y₂, y₁ ≠ y₂) (λ z₁ z₂, z₁ ≠ z₂) ∧
    V.card = n) →
  n ≤ 21 :=
sorry

end max_vertices_l75_75324


namespace probability_two_red_balls_l75_75758

open Nat

theorem probability_two_red_balls 
  (total_red : Nat := 5)
  (total_blue : Nat := 6)
  (total_green : Nat := 4)
  (total_balls : Nat := total_red + total_blue + total_green)  
  (num_ways_to_pick_2_in_15 : Nat := choose total_balls 2)
  (num_ways_to_pick_2_red : Nat := choose total_red 2) :
  num_ways_to_pick_2_red.to_rat / num_ways_to_pick_2_in_15.to_rat = 2 / 21 := by
  sorry

end probability_two_red_balls_l75_75758


namespace range_of_f_l75_75872

theorem range_of_f (x a : ℝ) (hx : 0 ≤ x) (hx' : x < 1) (h : (2 - a) * Real.exp a = x * (2 + a)) :
  ∃ r, r = (f x = a^2 * Real.exp a / (Real.exp a - (a+1) * x)) ∧ (2 < f x ∧ f x ≤ 4) := sorry

end range_of_f_l75_75872


namespace condition_for_purely_imaginary_l75_75016

theorem condition_for_purely_imaginary (m : ℝ) : 
  (m = 1) ↔ (∃ z : ℂ, z = m^2 * (1 + complex.I) + m * (complex.I - 1) ∧ (z.re = 0 ∧ z.im ≠ 0)) :=
by sorry

end condition_for_purely_imaginary_l75_75016


namespace complex_square_eq_two_i_l75_75973

theorem complex_square_eq_two_i :
  let z : ℂ := (1 : ℂ) + (1 : ℂ) * complex.I in
  z^2 = 2 * complex.I :=
by
  sorry

end complex_square_eq_two_i_l75_75973


namespace recover_original_sequence_l75_75943

theorem recover_original_sequence :
  ∃ (a d : ℤ),
    [a, a + d, a + 2 * d, a + 3 * d, a + 4 * d, a + 5 * d] = [113, 125, 137, 149, 161, 173] :=
by
  sorry

end recover_original_sequence_l75_75943


namespace min_PE_PC_eq_sqrt13_l75_75028

variables (A B C D E P : Point)
variables (BE CE : ℝ)
variables (square : ∀ {X Y Z W : Point}, is_square X Y Z W)
variables (BC_side : ∀ {X Y : Point}, is_segment X Y ∧ X = B ∧ Y = C)
variables (BD_line : ∀ {X Y : Point}, is_line X Y ∧ X = B ∧ Y = D)

noncomputable def minimum_PE_PC (P : Point) : ℝ :=
  if (is_on_line B D P) 
  then distance P E + distance P C
  else -1  -- can be replaced with any sentinel value for non-valid P
  
theorem min_PE_PC_eq_sqrt13 :
  BE = 2 → CE = 1 →
  square A B C D →
  BC_side B C →
  BD_line B D →
  ∃ P, (is_on_line B D P) ∧ minimum_PE_PC E P = sqrt 13 :=
by
  intros
  sorry

end min_PE_PC_eq_sqrt13_l75_75028


namespace sum_of_squares_ending_in_25_l75_75717

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75717


namespace largest_int_n_binom_l75_75180

theorem largest_int_n_binom (
  h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n
) : n = 7 := by
  sorry

end largest_int_n_binom_l75_75180


namespace uv_in_triangle_l75_75040

noncomputable def triangle (X Y Z : Point) (XY XZ YZ : ℝ)
  (XY_eq : XY = 150) (XZ_eq : XZ = 130) (YZ_eq : YZ = 140)
  (T : Point) (S : Point)  -- Points where angle bisectors intersect opposite sides
  (U : Point) (V : Point)  -- Feet of perpendiculars from Z to line segments YS and XT
  (UV : ℝ) : Prop :=
  ∃ (X Y Z T S U V : Point), 
    XY = 150 ∧ XZ = 130 ∧ YZ = 140 ∧
    T_on_YZ : lies_on T YZ ∧
    S_on_XZ : lies_on S XZ ∧
    U_foot : perpendicular_from Z YS U ∧
    V_foot : perpendicular_from Z XT V ∧
    XY_segment_eq : UV = 60

theorem uv_in_triangle :
  ∀ (X Y Z T S U V : Point) (XY XZ YZ UV : ℝ),
  triangle X Y Z XY XZ YZ 150 130 140 T S U V UV →
  UV = 60 :=
by
  intros X Y Z T S U V XY XZ YZ UV h
  exact sorry

end uv_in_triangle_l75_75040


namespace train_speed_is_correct_l75_75279

noncomputable def train_speed (L : ℝ) (t : ℝ) (V_man_kmhr : ℝ) : ℝ :=
  let V_man_ms := V_man_kmhr * 1000 / 3600
  let V_relative := L / t
  let V_train_ms := V_relative - V_man_ms
  V_train_ms * 3600 / 1000

theorem train_speed_is_correct : 
  train_speed 275 14.998800095992323 6 ≈ 60.006 := by
  sorry

end train_speed_is_correct_l75_75279


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75647

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75647


namespace coefficient_of_x5y2_in_expansion_l75_75584

theorem coefficient_of_x5y2_in_expansion 
  :  polynomial.coeff (polynomial.expand (x^2 + x + y)^5) (5, 2) = 30 :=
sorry

end coefficient_of_x5y2_in_expansion_l75_75584


namespace purely_imaginary_complex_number_l75_75932

theorem purely_imaginary_complex_number (a : ℝ) :
  (∃ b : ℝ, (a^2 - 3 * a + 2) = 0 ∧ a ≠ 1) → a = 2 :=
by
  sorry

end purely_imaginary_complex_number_l75_75932


namespace students_second_scenario_l75_75018

def total_students (R : ℕ) : ℕ := 5 * R + 6
def effective_students (R : ℕ) : ℕ := 6 * (R - 3)
def filled_rows (R : ℕ) : ℕ := R - 3
def students_per_row := 6

theorem students_second_scenario:
  ∀ (R : ℕ), R = 24 → total_students R = effective_students R → students_per_row = 6
:= by
  intro R h_eq h_total_eq_effective
  -- Insert proof steps here
  sorry

end students_second_scenario_l75_75018


namespace ratio_surface_area_l75_75260

noncomputable def side_length (a : ℝ) := a
noncomputable def radius (R : ℝ) := R

theorem ratio_surface_area (a R : ℝ) (h : a^3 = (4/3) * Real.pi * R^3) : 
  (6 * a^2) / (4 * Real.pi * R^2) = (3 * (6 / Real.pi)) :=
by sorry

end ratio_surface_area_l75_75260


namespace min_value_of_f_monotonic_increasing_interval_of_g_l75_75412

open Real

def f (x : ℝ) : ℝ := (1 / 2) * cos x ^ 2 + (sqrt 3 / 2) * sin x * cos x

theorem min_value_of_f :
  ∃ x : ℝ, f x = -1 / 4 ∧ (∃ k : ℤ, x = k * π - π / 3) := 
sorry

def g (x : ℝ) : ℝ := f (x - π / 6)

theorem monotonic_increasing_interval_of_g :
  ∃ k : ℤ, -π / 4 + k * π < x ∧ x < π / 4 + k * π :=
sorry

end min_value_of_f_monotonic_increasing_interval_of_g_l75_75412


namespace choose_4_from_15_l75_75946

theorem choose_4_from_15 : Nat.choose 15 4 = 1365 := by
  sorry

end choose_4_from_15_l75_75946


namespace sum_of_squares_ends_in_25_is_644_l75_75685

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75685


namespace hyperbola_properties_l75_75404

-- Define parameters and variables
variables {x y : ℝ}

-- Define the standard form of the hyperbola equation
def standard_hyperbola_eq : Prop :=
  (x ^ 2 / 36) - (y ^ 2 / 64) = 1

-- Define the asymptote equations
def asymptotes : Prop :=
  ∀ x, (y = (4/3) * x) ∨ (y = -(4/3) * x)

-- Define the eccentricity of the hyperbola
def eccentricity : ℝ := 5/3

-- State the theorem
theorem hyperbola_properties (h : standard_hyperbola_eq):
  asymptotes ∧ eccentricity = 5/3 :=
sorry

end hyperbola_properties_l75_75404


namespace minimum_species_l75_75480

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l75_75480


namespace sum_of_two_digit_numbers_ending_in_25_l75_75666

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75666


namespace tickets_sold_l75_75285

def advanced_purchase_tickets := ℕ
def door_purchase_tickets := ℕ

variable (A D : ℕ)

theorem tickets_sold :
  (A + D = 140) →
  (8 * A + 14 * D = 1720) →
  A = 40 :=
by
  intros h1 h2
  sorry

end tickets_sold_l75_75285


namespace max_area_right_triangle_l75_75874

def right_triangle_max_area (l : ℝ) (p : ℝ) (h : ℝ) : ℝ :=
  l + p + h

noncomputable def maximal_area (x y : ℝ) : ℝ :=
  (1/2) * x * y

theorem max_area_right_triangle (x y : ℝ) (h : ℝ) (hp : h = Real.sqrt (x^2 + y^2)) (hp2: x + y + h = 60) :
  maximal_area 30 30 = 450 :=
by
  sorry

end max_area_right_triangle_l75_75874


namespace top8_median_l75_75024

/--
  Given 15 participants in a speech contest and their respective scores,
  prove that to determine if a participant ranks in the top 8, they need
  to know their own score and the median of all scores.
-/
theorem top8_median (scores : List ℝ) (h_len : scores.length = 15) (own_score : ℝ) :
  (∃ m, median scores = m ∧ own_score > m) ∨ (∃ m, median scores = m ∧ own_score = m) ∨ (∃ m, median scores = m ∧ own_score < m) ↔
  ((∃ m, median scores = m ∧ own_score ≥ m) ↔ ∃ n ≤ 15, n < 8 ∧ ∃ sorted_scores, (sorted_scores = scores.sort) ∧ (sorted_scores = own_score)) :=
sorry

end top8_median_l75_75024


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75727

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75727


namespace unique_positive_solution_l75_75361

noncomputable def num_positive_solutions (f : ℝ → ℝ) : ℕ :=
  if h : ∃ x ∈ Icc (-1.0 : ℝ) 1.0, f x = x ∧ 0 < x
  then 1
  else 0

theorem unique_positive_solution : num_positive_solutions (λ x, cos (arcsin (cot (arccos x)))) = 1 :=
sorry

end unique_positive_solution_l75_75361


namespace randy_biscuits_l75_75563

theorem randy_biscuits (initial_biscuits father_gift mother_gift brother_ate : ℕ) : 
  (initial_biscuits = 32) →
  (father_gift = 13) →
  (mother_gift = 15) →
  (brother_ate = 20) →
  initial_biscuits + father_gift + mother_gift - brother_ate = 40 := by
  sorry

end randy_biscuits_l75_75563


namespace rainfall_in_2011_l75_75443

-- Define the parameters
def avg_rainfall_2010 : ℝ := 37.2
def increase_from_2010_to_2011 : ℝ := 1.8
def months_in_a_year : ℕ := 12

-- Define the total rainfall in 2011
def total_rainfall_2011 : ℝ := 468

-- Prove that the total rainfall in Driptown in 2011 is 468 mm
theorem rainfall_in_2011 :
  avg_rainfall_2010 + increase_from_2010_to_2011 = 39.0 → 
  12 * (avg_rainfall_2010 + increase_from_2010_to_2011) = total_rainfall_2011 :=
by sorry

end rainfall_in_2011_l75_75443


namespace final_volume_solution_l75_75986

variables (V2 V12 V_final : ℝ)

-- Given conditions
def V2_percent_solution (V2 : ℝ) := true
def V12_percent_solution (V12 : ℝ) := V12 = 18
def mixture_equation (V2 V12 V_final : ℝ) := 0.02 * V2 + 0.12 * V12 = 0.05 * V_final
def total_volume (V2 V12 V_final : ℝ) := V_final = V2 + V12

theorem final_volume_solution (V2 V_final : ℝ) (hV2: V2_percent_solution V2)
    (hV12 : V12_percent_solution V12) (h_mix : mixture_equation V2 V12 V_final)
    (h_total : total_volume V2 V12 V_final) : V_final = 60 :=
sorry

end final_volume_solution_l75_75986


namespace jay_used_zero_fraction_of_gallon_of_paint_l75_75322

theorem jay_used_zero_fraction_of_gallon_of_paint
    (dexter_used : ℝ := 3/8)
    (gallon_in_liters : ℝ := 4)
    (paint_left_liters : ℝ := 4) :
    dexter_used = 3/8 ∧ gallon_in_liters = 4 ∧ paint_left_liters = 4 →
    ∃ jay_used : ℝ, jay_used = 0 :=
by
  sorry

end jay_used_zero_fraction_of_gallon_of_paint_l75_75322


namespace articles_produced_l75_75928

theorem articles_produced (a b c p q r : Nat) (h : a * b * c = abc) : p * q * r = pqr := sorry

end articles_produced_l75_75928


namespace largest_int_n_binom_l75_75178

theorem largest_int_n_binom (
  h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n
) : n = 7 := by
  sorry

end largest_int_n_binom_l75_75178


namespace sum_of_two_digit_numbers_ending_in_25_l75_75633

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75633


namespace find_b_l75_75607

variable (a b : ℝ^3)
variable (k : ℝ)
variable (v : ℝ^3 := ![2, -1, 1])
variable (w : ℝ^3 := ![8, 4, -3])
variable (b_res : ℝ^3 := ![5, 11/2, -9/2])

def is_parallel_to (x y : ℝ^3) : Prop := ∃ k : ℝ, x = k • y
def is_orthogonal_to (x y : ℝ^3) : Prop := dot_product x y = 0

theorem find_b (ha : is_parallel_to a v)
  (hab : a + b = w)
  (hb : is_orthogonal_to b v) :
  b = b_res := sorry

end find_b_l75_75607


namespace part1_proof_part2_proof_part3_proof_l75_75032

-- Definitions and conditions for part 1
def P (a : ℤ) : ℤ × ℤ := (-3 * a - 4, 2 + a)
def part1_condition (a : ℤ) : Prop := (2 + a = 0)
def part1_answer : ℤ × ℤ := (2, 0)

-- Definitions and conditions for part 2
def Q : ℤ × ℤ := (5, 8)
def part2_condition (a : ℤ) : Prop := (-3 * a - 4 = 5)
def part2_answer : ℤ × ℤ := (5, -1)

-- Definitions and conditions for part 3
def part3_condition (a : ℤ) : Prop := 
  (-3 * a - 4 + 2 + a = 0) ∧ (-3 * a - 4 < 0 ∧ 2 + a > 0) -- Second quadrant
def part3_answer (a : ℤ) : ℤ := (a ^ 2023 + 2023)

-- Lean statements for proofs

theorem part1_proof (a : ℤ) (h : part1_condition a) : P a = part1_answer :=
by sorry

theorem part2_proof (a : ℤ) (h : part2_condition a) : P a = part2_answer :=
by sorry

theorem part3_proof (a : ℤ) (h : part3_condition a) : part3_answer a = 2022 :=
by sorry

end part1_proof_part2_proof_part3_proof_l75_75032


namespace problem1_problem2_l75_75824

theorem problem1 : 
  (5 / 7 : ℚ) * (-14 / 3) / (5 / 3) = -2 := 
by 
  sorry

theorem problem2 : 
  (-15 / 7 : ℚ) / (-6 / 5) * (-7 / 5) = -5 / 2 := 
by 
  sorry

end problem1_problem2_l75_75824


namespace quadratic_equation_roots_l75_75406

theorem quadratic_equation_roots (m n : ℝ) 
  (h_sum : m + n = -3) 
  (h_prod : m * n = 1) 
  (h_equation : m^2 + 3 * m + 1 = 0) :
  (3 * m + 1) / (m^3 * n) = -1 := 
by sorry

end quadratic_equation_roots_l75_75406


namespace min_value_fraction_sum_l75_75893

theorem min_value_fraction_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (∃ x : ℝ, x = (1 / a + 4 / b) ∧ x = 9 / 4) :=
by
  sorry

end min_value_fraction_sum_l75_75893


namespace circle_radius_in_quadrilateral_lt_3_l75_75790

theorem circle_radius_in_quadrilateral_lt_3 :
  ∀ a b c d r : ℝ, 
  a = 3 → 
  b = 6 → 
  c = 5 → 
  d = 8 → 
  let s := (a + b + c + d) / 2 in
  let S := real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) in
  r = S / s →
  r < 3 := by
  intros a b c d r ha hb hc hd s S hr
  simp [ha, hb, hc, hd] at s
  simp [s, real.sqrt] at S
  simp [S, hr]
  sorry

end circle_radius_in_quadrilateral_lt_3_l75_75790


namespace bananas_per_friend_l75_75619

-- Define constants and conditions
def totalBananas : Nat := 40
def totalFriends : Nat := 40

-- Define the main theorem to prove
theorem bananas_per_friend : totalBananas / totalFriends = 1 := by
  sorry

end bananas_per_friend_l75_75619


namespace total_pay_is_880_l75_75614

theorem total_pay_is_880 (X_pay Y_pay : ℝ) 
  (hY : Y_pay = 400)
  (hX : X_pay = 1.2 * Y_pay):
  X_pay + Y_pay = 880 :=
by
  sorry

end total_pay_is_880_l75_75614


namespace minimum_bird_species_l75_75456

theorem minimum_bird_species (total_birds : ℕ) (h : total_birds = 2021) :
  ∃ (min_species : ℕ), min_species = 1011 ∧ 
  (∀ (species_array : array total_birds ℕ),
   ∀ i j : fin total_birds, 
   species_array[i] = species_array[j] → ((i ≠ j) →
   (abs (i - j) mod 2 = 0))) :=
sorry

end minimum_bird_species_l75_75456


namespace sum_of_integers_whose_squares_end_with_25_l75_75714

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75714


namespace choose_4_from_15_is_1365_l75_75963

theorem choose_4_from_15_is_1365 : nat.choose 15 4 = 1365 :=
by
  sorry

end choose_4_from_15_is_1365_l75_75963


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75648

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75648


namespace largest_n_binom_sum_l75_75197

open Nat

theorem largest_n_binom_sum (n : ℕ) 
  (h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n) : n = 7 :=
by 
  sorry

end largest_n_binom_sum_l75_75197


namespace rationalize_denominator_cube_root_l75_75564

theorem rationalize_denominator_cube_root : 
  (∛(4 / 9) : ℝ) = (∛324) / 9 :=
by
  sorry

end rationalize_denominator_cube_root_l75_75564


namespace tallest_player_height_l75_75152

theorem tallest_player_height (shortest_player tallest_player : ℝ) (height_diff : ℝ)
  (h1 : shortest_player = 68.25)
  (h2 : height_diff = 9.5)
  (h3 : tallest_player = shortest_player + height_diff) :
  tallest_player = 77.75 :=
by {
  rw [h1, h2] at h3,
  exact h3,
  sorry
}

end tallest_player_height_l75_75152


namespace range_of_m_l75_75908

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h1 : y1 = (1 + 2 * m) / x1) (h2 : y2 = (1 + 2 * m) / x2)
    (hx : x1 < 0 ∧ 0 < x2) (hy : y1 < y2) : m > -1 / 2 :=
sorry

end range_of_m_l75_75908


namespace fraction_inequality_l75_75931

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < (b + m) / (a + m) := 
sorry

end fraction_inequality_l75_75931


namespace simplify_and_evaluate_l75_75106

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = 1/25) (h2 : y = -25) :
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = -3 :=
by
  sorry

end simplify_and_evaluate_l75_75106


namespace julie_same_hours_september_october_l75_75513

-- Define Julie's hourly rates and work hours
def rate_mowing : ℝ := 4
def rate_weeding : ℝ := 8
def september_mowing_hours : ℕ := 25
def september_weeding_hours : ℕ := 3
def total_earnings_september_october : ℤ := 248

-- Define Julie's earnings for each activity and total earnings for September
def september_earnings_mowing : ℝ := september_mowing_hours * rate_mowing
def september_earnings_weeding : ℝ := september_weeding_hours * rate_weeding
def september_total_earnings : ℝ := september_earnings_mowing + september_earnings_weeding

-- Define earnings in October
def october_earnings : ℝ := total_earnings_september_october - september_total_earnings

-- Define the theorem to prove Julie worked the same number of hours in October as in September
theorem julie_same_hours_september_october :
  october_earnings = september_total_earnings :=
by
  sorry

end julie_same_hours_september_october_l75_75513


namespace sum_of_squares_ending_in_25_l75_75720

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75720


namespace altitude_intersection_inequality_l75_75580

theorem altitude_intersection_inequality
  (A B C H B1 C1 O_b O_c : Point)
  (acute_triangle : ∠BAC < π / 2 ∧ ∠ABC < π / 2 ∧ ∠BCA < π / 2)
  (altitude_B1 : line B B1 ⊥ line C A)
  (altitude_C1 : line C C1 ⊥ line A B)
  (intersection_H : ∃ H, line B B1 ∩ line C C1 = {H})
  (circle_O_b : ∃ M, midpoint (segment B H) M ∧ circle O_b A C1 M)
  (circle_O_c : ∃ M', midpoint (segment C H) M' ∧ circle O_c A B1 M') :
  distance B1 O_b + distance C1 O_c > distance B C :=
sorry

end altitude_intersection_inequality_l75_75580


namespace gain_percent_correct_l75_75241

def gain_percent (cp sp : ℕ) : ℕ :=
  ((sp - cp) * 100) / cp

theorem gain_percent_correct (cp sp : ℕ) (h_cp : cp = 1000) (h_sp : sp = 1080) :
  gain_percent cp sp = 8 :=
by
  rw [h_cp, h_sp]
  simp [gain_percent]
  norm_num
  sorry

end gain_percent_correct_l75_75241


namespace residue_5_pow_1234_mod_13_l75_75627

theorem residue_5_pow_1234_mod_13 : ∃ k : ℤ, 5^1234 = 13 * k + 12 :=
by
  sorry

end residue_5_pow_1234_mod_13_l75_75627


namespace min_species_needed_l75_75471

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l75_75471


namespace necklace_arrangements_l75_75492

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem necklace_arrangements (n : ℕ) (hn : n = 7) : 
  (factorial n) / (n * 2) = 360 :=
by 
  rw [hn]
  rw [←nat.fact_eq_factorial]
  rw [nat.factorial]
  apply sorry

end necklace_arrangements_l75_75492


namespace range_of_a_l75_75373

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : Real.log a 3 < 1) : a ∈ (Set.Ioo 0 1 ∪ Set.Ioi 3) :=
sorry

end range_of_a_l75_75373


namespace same_side_of_sign_l75_75772

-- Define the motion equations for the tourists
def S1 (t : ℝ) : ℝ := sqrt (1 + 6 * t) - 1

def S2 (t : ℝ) : ℝ :=
  if t >= 1/6 then 6 * (t - 1/6) else 0

-- The times at which the tourists reach the sign
def t1 := 4/3
def t2 := 1/2

-- Define the intervals when the tourists are on the same side of the sign
def same_side_intervals (t : ℝ) : Prop :=
  (t >= 0 ∧ t <= 1/2) ∨ (t >= 4/3)

theorem same_side_of_sign (t : ℝ) :
  same_side_intervals t ↔ (S1 t = 2 ∨ S2 t = 2) ∨ (S1 t <= -2 ∧ S2 t <= -2) :=
sorry

end same_side_of_sign_l75_75772


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75644

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75644


namespace stack_of_crates_count_l75_75168

open Finset

theorem stack_of_crates_count : 
  (∃ a b c : ℕ, 2 * a + 4 * b + 5 * c = 40 ∧ a + b + c = 12 ∧ 
    choose 12 a * choose (12 - a) b = 495 + 792 + 924) :=
by
  let configurations := [
    ⟨4, 8, 0⟩,
    ⟨5, 5, 2⟩,
    ⟨6, 2, 4⟩
  ]
  sorry

end stack_of_crates_count_l75_75168


namespace arithmetic_mean_of_integers_from_neg5_to_6_l75_75175

def intRange := List.range' (-5) (12)

def sumIntRange (l : List Int) : Int :=
  l.foldl (fun acc x => acc + x) 0

theorem arithmetic_mean_of_integers_from_neg5_to_6 : 
  (sumIntRange intRange / intRange.length.toReal) = 0.5 := by
  sorry

end arithmetic_mean_of_integers_from_neg5_to_6_l75_75175


namespace alice_more_than_half_sum_l75_75246

-- Conditions
def row_of_fifty_coins (denominations : List ℤ) : Prop :=
  denominations.length = 50 ∧ (List.sum denominations) % 2 = 1

def alice_starts (denominations : List ℤ) : Prop := True
def bob_follows (denominations : List ℤ) : Prop := True
def alternating_selection (denominations : List ℤ) : Prop := True

-- Question/Proof Goal
theorem alice_more_than_half_sum (denominations : List ℤ) 
  (h1 : row_of_fifty_coins denominations)
  (h2 : alice_starts denominations)
  (h3 : bob_follows denominations)
  (h4 : alternating_selection denominations) :
  ∃ s_A : ℤ, s_A > (List.sum denominations) / 2 ∧ s_A ≤ List.sum denominations :=
sorry

end alice_more_than_half_sum_l75_75246


namespace ham_cheese_bread_percentage_l75_75610

noncomputable def bread_cost_yen := 1000 -- cost in yen
noncomputable def ham_cost_per_oz_eur := 10 -- cost per ounce in euros
noncomputable def total_ham_oz := 2 -- total ham in ounces
noncomputable def ham_discount := 0.1 -- 10% discount on ham
noncomputable def cake_cost_kr := 100 -- cost in kroner
noncomputable def cake_discount := 0.2 -- 20% discount on cake
noncomputable def cheese_cost_per_lb := 75 -- cost per pound in dollars
noncomputable def total_cheese_lb := 1.5 -- total cheese in pounds

noncomputable def yen_to_usd := 0.010 -- exchange rate yen to USD
noncomputable def eur_to_usd := 1.1 -- exchange rate euro to USD
noncomputable def kr_to_usd := 0.15 -- exchange rate krone to USD

noncomputable def calculate_cost_in_usd : ℝ :=
  (bread_cost_yen * yen_to_usd) +
  ((ham_cost_per_oz_eur * eur_to_usd) * total_ham_oz * (1 - ham_discount)) +
  ((cake_cost_kr * kr_to_usd) * (1 - cake_discount)) +
  (cheese_cost_per_lb * total_cheese_lb)

noncomputable def ham_cost_in_usd := (ham_cost_per_oz_eur * eur_to_usd) * total_ham_oz * (1 - ham_discount)
noncomputable def cheese_cost_in_usd := cheese_cost_per_lb * total_cheese_lb
noncomputable def bread_cost_in_usd := bread_cost_yen * yen_to_usd

noncomputable def percentage_of_total_cost (item_cost : ℝ) : ℝ :=
  (item_cost / calculate_cost_in_usd) * 100

theorem ham_cheese_bread_percentage : percentage_of_total_cost ham_cost_in_usd + percentage_of_total_cost cheese_cost_in_usd + percentage_of_total_cost bread_cost_in_usd = 92.21 := by
  sorry

#print ham_cheese_bread_percentage

end ham_cheese_bread_percentage_l75_75610


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75675

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75675


namespace expression_value_l75_75230

theorem expression_value (a b : ℤ) (ha : a = 4) (hb : b = -5) : -a^2 - b^2 + a * b + b = -66 :=
by {
  rw [ha, hb],
  norm_num,
  sorry
}

end expression_value_l75_75230


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75658

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75658


namespace cone_base_radius_l75_75275

theorem cone_base_radius (R : ℝ) (theta : ℝ) (radius : ℝ) (hR : R = 30) (hTheta : theta = 120) :
    2 * Real.pi * radius = (theta / 360) * 2 * Real.pi * R → radius = 10 :=
by
  intros h
  sorry

end cone_base_radius_l75_75275


namespace digit_sum_2001_not_perfect_square_l75_75825

theorem digit_sum_2001_not_perfect_square (n : ℕ) (h : (n.digits 10).sum = 2001) : ¬ ∃ k : ℕ, n = k * k := 
sorry

end digit_sum_2001_not_perfect_square_l75_75825


namespace find_numbers_l75_75604

theorem find_numbers (x y : ℝ) (hx : y = 10 * x) (hsum : x + y = 13.5927) : x = 1.2357 ∧ y = 12.357 :=
by {
  -- We state the conditions directly in the assumptions.
  have h1 : 11 * x = 13.5927 := by sorry,
  have h2 : x = 1.2357 := by sorry,
  have h3 : y = 12.357 := by sorry,
  exact ⟨h2, h3⟩,
}

end find_numbers_l75_75604


namespace sum_of_two_digit_numbers_ending_in_25_l75_75665

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75665


namespace c_10_is_177_l75_75839

def seq (c : ℕ → ℕ) : Prop :=
  c 1 = 1 ∧
  c 2 = 3 ∧
  ∀ n ≥ 2, c (n + 1) = c n + c (n - 1) + 1

theorem c_10_is_177 (c : ℕ → ℕ) (h : seq c) : c 10 = 177 :=
by {
  unfold seq at h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  sorry
}

end c_10_is_177_l75_75839


namespace smallest_positive_integer_n_l75_75051

noncomputable def minimum_edges (V : Finset (Fin 2019)) (E : Finset (Finset (Fin 2019))) : ℕ :=
  Inf {n | ∃ C : Finset (Finset (Finset (Fin 2019))), C.card = 908 ∧ ∀ c ∈ C, (c.card = 2 ∧ c ∈ E) ∧ 
    ∃ v : Fin 2019, ∀ e ∈ c, v ∈ e ∧ (∀ c1 c2 ∈ C, c1 ≠ c2 → c1 ∩ c2 = ∅)}

theorem smallest_positive_integer_n {V : Finset (Fin 2019)}
  (hV : ∀ p1 p2 p3 p4 ∈ V, ¬(∃ a : ℝ, ∃ b : ℝ, ∃ c : ℝ, ∃ d : ℝ, 
    ∀ p : Fin 2019, (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 → a * (p1.1) + b * (p2.1) + c * (p3.1) + d = 0)))
  (E : Finset (Finset (Fin 2019)))
  (hE : ∀ e ∈ E, e.card = 2)
  : minimum_edges V E = 2795 := 
sorry

end smallest_positive_integer_n_l75_75051


namespace Randy_biscuits_l75_75558

theorem Randy_biscuits
  (biscuits_initial : ℕ)
  (father_gift : ℕ)
  (mother_gift : ℕ)
  (brother_eat : ℕ) :
  biscuits_initial = 32 →
  father_gift = 13 →
  mother_gift = 15 →
  brother_eat = 20 →
  biscuits_initial + father_gift + mother_gift - brother_eat = 40 :=
by
  intros h_initial h_father h_mother h_brother
  rw [h_initial, h_father, h_mother, h_brother]
  norm_num
  sorry

end Randy_biscuits_l75_75558


namespace f_at_neg_2_l75_75591

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + x^2 + b * x + 2

-- Given the condition
def f_at_2_eq_3 (a b : ℝ) : Prop := f 2 a b = 3

-- Prove the value of f(-2)
theorem f_at_neg_2 (a b : ℝ) (h : f_at_2_eq_3 a b) : f (-2) a b = 1 :=
sorry

end f_at_neg_2_l75_75591


namespace correct_statement_l75_75236

-- Definitions
def is_monomial (f : ℝ → ℝ) : Prop := ∃ (c : ℝ) (n : ℕ), ∀ x, f x = c * x^n

def is_binomial_of_degree (f : ℝ → ℝ) (d : ℕ) : Prop := (∃ a b n1 n2 : ℝ, (f(x) = a * x^n1 + b * x^n2) ∧ (max n1 n2 = d))

-- Given conditions
axiom A : ¬ ∀ f, ∃ c : ℤ, ∀ x, is_integer (f(x) = c * x^n)
axiom B : is_monomial (λ x, π)
axiom C : ¬ is_binomial_of_degree (λ x, x^4 + 2 * x^3) 7
axiom D : ¬ is_monomial (λ x, (3 * x - 1) / 5)

-- Proof statement
theorem correct_statement : B := by sorry

end correct_statement_l75_75236


namespace sum_of_squares_ends_in_25_is_644_l75_75687

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75687


namespace exists_monochromatic_right_triangle_l75_75060

-- Define the equilateral triangle ABC
structure EquilateralTriangle :=
(A B C : Point)
(eq_sides : dist A B = dist B C ∧ dist B C = dist C A)

-- Define set G such that G is the set of all points on the segments AB, BC, CA
def setG (ABC : EquilateralTriangle) : Set Point :=
  {p | ∃ t ∈ {(ABC.A, ABC.B), (ABC.B, ABC.C), (ABC.C, ABC.A)}, p ∈ segment t.1 t.2}

-- Define a coloring function such that some points in G are black and others are white
def coloring (ABC : EquilateralTriangle) : Point → Prop :=
  sorry  -- We define function later

-- Define inscribed right triangle condition
def isRightTriangle (Δ : Triangle) : Prop :=
  Δ.angle1 = π / 2 ∨ Δ.angle2 = π / 2 ∨ Δ.angle3 = π / 2

-- Define monochromatic condition
def isMonochromatic (Δ : Triangle) (color : Point → Prop) : Prop :=
  color Δ.vert1 ∧ color Δ.vert2 ∧ color Δ.vert3

-- Main theorem statement
theorem exists_monochromatic_right_triangle
  (ABC : EquilateralTriangle)
  (color : Point → Prop)
  (h_color : ∀ p, p ∈ setG ABC → color p ∨ ¬ color p)
  : ∃ Δ : Triangle, (Δ ∈ inscribed_triangles ABC) ∧ isRightTriangle Δ ∧ isMonochromatic Δ color :=
sorry

end exists_monochromatic_right_triangle_l75_75060


namespace measure_angle_FAO_l75_75972

-- Define basic geometrical entities and properties
def point := Type
def angle := ℝ -- angles in degrees
def length := ℝ

-- Define the problem conditions as hypotheses
variables (CD : length) (O : point) (A : point) (E : point) (F : point) (AF OD : length)
variable (is_diameter : ∀ (C D : point), CD = 2 * OD)
variable (on_extension_DO_past_D : ∀ (D : point), A is on the extension of DO past D)
variable (on_semircle : E on semicircle with center O)
variable (intersection_AF : F is intersection of line segment AF with semicircle distinct from E)
variable (AF_twice_OD : AF = 2 * OD)
variable (angle_EOD_30 : angle EOD = 30)

-- Define the measure of ∠FAO
variable (FAO_angle : angle)

-- The theorem to be proven
theorem measure_angle_FAO : FAO_angle = 10 :=
sorry

end measure_angle_FAO_l75_75972


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75643

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75643


namespace range_of_a_l75_75871

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, x ≥ 1 → f x = log x - a * x^2 + a) →
  (∀ x, x ≥ 1 → f x ≤ 0) →
  a ≥ 1 / 2 :=
by sorry

end range_of_a_l75_75871


namespace golden_residue_modulo_nnn_l75_75514

-- Define positive integers and gcd
variables (m n a : ℕ)

-- Assume conditions
def golden_residue (n k : ℕ) (a : ℕ) : Prop := (nat.gcd a n = 1 ∧ ∃ x : ℕ, (x ^ x) % n = a)

-- Definition of the original problem's conditional statement
theorem golden_residue_modulo_nnn {n a : ℕ} (hn_pos : 0 < n) (ha_pos : 0 < a)
  (hg_residue_n : golden_residue (n ^ n) a) :
  golden_residue (n ^ (n ^ n)) a :=
sorry

end golden_residue_modulo_nnn_l75_75514


namespace swimming_pool_volume_l75_75005

def diameter : ℝ := 20
def depth : ℝ := 5
noncomputable def volume : ℝ := π * (diameter / 2)^2 * depth

theorem swimming_pool_volume : volume = 500 * π :=
by
  sorry

end swimming_pool_volume_l75_75005


namespace total_turtles_after_one_month_l75_75548

structure TurtleData where
  initial_turtles_owen : ℕ   -- Owen initially bred 21 turtles
  initial_turtles_johanna : ℕ -- Johanna has 5 fewer turtles than Owen
  G : ℕ                    -- Growth rate of Owen's turtles
  X : ℕ                    -- Number of turtles Liam has
  Y : ℕ                    -- Percentage of turtles Liam will contribute
  deriving DecidableEq, Repr

theorem total_turtles_after_one_month (data : TurtleData) : 
  data.initial_turtles_owen = 21 →
  data.initial_turtles_johanna = 16 →
  data.G = 2 →  -- Assuming G is doubled, can be generalized for other values if needed
  ∃ total_turtles : ℕ, 
    total_turtles = (42 * data.G) + 8 + ((data.Y / 100) * data.X) :=
by
  intros ho hj hG 
  use (42 * data.G + 8 + (data.Y / 100) * data.X)
  sorry


end total_turtles_after_one_month_l75_75548


namespace max_min_f_g_critical_points_count_min_k_value_l75_75413

open Real

-- Definition for f(x) and its conditions
def f (m x : ℝ) : ℝ := (m - x) * exp x

-- Definition for g(x) and its conditions
def g (m x : ℝ) : ℝ := x * f m x

-- Statement (Ⅰ): Prove maximum and minimum values for f(x) when m=2 on the interval [-1, 2]
theorem max_min_f (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) : 
  let f2 := f 2 x in 
  f2 ≤ exp 1 ∧ f2 ≥ 0 := 
sorry

-- Statement (Ⅱ): Prove that g(x) has exactly 2 critical points when m=2
theorem g_critical_points_count (m x : ℝ) : 
  let g' := ((m - 2 * x) + (m * x - x^2)) * exp x in 
  ∃ a b : ℝ, a ≠ b ∧ g' = 0 :=
sorry

-- Statement (Ⅲ): Prove the minimum value of k such that ke^x ≥ x + 2 for all x ∈ [-2, 1]
theorem min_k_value (k : ℝ) : 
  (∀ x ∈ Icc (-2 : ℝ) (1 : ℝ), k * exp x ≥ x + 2) ↔ k ≥ exp 1 :=
sorry

end max_min_f_g_critical_points_count_min_k_value_l75_75413


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75739

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75739


namespace det_E_l75_75519

open Matrix

def E : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

theorem det_E : det E = 25 := by
  sorry

end det_E_l75_75519


namespace incorrect_transformation_l75_75233

theorem incorrect_transformation (a b c : ℝ) (h1 : a = b) (h2 : c = 0) : ¬(a / c = b / c) :=
by
  sorry

end incorrect_transformation_l75_75233


namespace ellipse_eqn_constant_dot_product_l75_75097

-- Define the ellipse and the conditions
def is_on_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Point M on the ellipse
def point_M_on_ellipse (a b : ℝ) : Prop :=
  is_on_ellipse (real.sqrt 2) 1 a b

-- Condition for the sum of distances from a point to the foci of the ellipse
def sum_distances_to_foci (x y a b : ℝ) : ℝ :=
  let f := real.sqrt (a^2 - b^2) in
  real.sqrt ((x - f)^2 + y^2) + real.sqrt ((x + f)^2 + y^2)

-- Define the equation of the ellipse
noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  1 = 1 → is_on_ellipse 1 0 a b = (1 / 5 = 1 / (a^2)) ∧ (3 / 5 = 3 / (b^2)))

-- Prove 1: The equation of the ellipse is x^2/5 + 3y^2/5 = 1
theorem ellipse_eqn :
  ∃ a b : ℝ, point_M_on_ellipse a b ∧ (sum_distances_to_foci (real.sqrt 2) 1 a b = 2 * real.sqrt 5) ∧ ellipse_equation a b :=
sorry

-- Prove 2: The dot product is constant 
theorem constant_dot_product :
  ∀ (a b : ℝ) (k : ℝ),
  point_M_on_ellipse a b →
  sum_distances_to_foci (real.sqrt 2) 1 a b = 2 * real.sqrt 5 →
  (ellipse_equation a b) →
  (∃ x1 y1 x2 y2: ℝ,
    y1 = k * (x1 + 1) ∧ is_on_ellipse x1 y1 a b ∧
    y2 = k * (x2 + 1) ∧ is_on_ellipse x2 y2 a b →
    ((x1 + 7/3) * (x2 + 7/3) + y1 * y2) = 4 / 9) :=
sorry

end ellipse_eqn_constant_dot_product_l75_75097


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75740

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75740


namespace largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75209

open Nat

theorem largest_n_binom_10_3_10_4_eq_binom_11_n :
  ∃ n, (n ≤ 11) ∧ (∑ i in finset.range 11, if i = 4 then 1 else 0) = 1 ∧
  (n = 7) := by
sorry

end largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75209


namespace reynalds_total_balls_l75_75103

theorem reynalds_total_balls : 
  let soccer_balls := 20 in
  let basketballs := soccer_balls + 5 in
  let tennis_balls := 2 * soccer_balls in
  let baseballs := soccer_balls + 10 in
  let volleyballs := 30 in
  let total_balls := soccer_balls + basketballs + tennis_balls + baseballs + volleyballs in
  total_balls = 145 :=
by
  sorry

end reynalds_total_balls_l75_75103


namespace total_percentage_saved_is_11_l75_75797

theorem total_percentage_saved_is_11 :
  let saved_on_dress := 4.25
  let saved_on_skirt := 3.75
  let spent_on_dress := 38.00
  let spent_on_skirt := 27.00
  let original_price_dress := spent_on_dress + saved_on_dress
  let original_price_skirt := spent_on_skirt + saved_on_skirt
  let total_original_price := original_price_dress + original_price_skirt
  let total_spent := spent_on_dress + spent_on_skirt
  let total_saved := saved_on_dress + saved_on_skirt
  let total_percentage_saved := (total_saved / total_original_price) * 100
  total_percentage_saved ≈ 11 := sorry

end total_percentage_saved_is_11_l75_75797


namespace triangle_ABC_area_l75_75035

-- Define the base and height of the triangle
def base : ℕ := 12
def height : ℕ := 9

-- Define the area calculation function for a triangle
def triangle_area (base height : ℕ) : ℕ := (base * height) / 2

-- The theorem that states the area of triangle ABC is equal to 54 given the base and height
theorem triangle_ABC_area : triangle_area base height = 54 := by
  -- Proof is omitted with sorry
  sorry

end triangle_ABC_area_l75_75035


namespace circle_C_equation_minimum_PQ_dot_MQ_lines_OP_parallel_AB_l75_75878

-- Definitions

def circle_M (x y : ℝ) (r : ℝ) : Prop := (x + 2) ^ 2 + (y + 2) ^ 2 = r ^ 2

def symmetric_about_line (x y : ℝ) : Prop := x + y + 2 = 0

def circle_C (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 2

def passes_through_P (x y : ℝ) : Prop := (1, 1) = (x, y)

-- Theorems

theorem circle_C_equation (x y : ℝ) :
  (passes_through_P x y) ∧ (circle_M.symmetry_about_line x y (circle_C_eq (x ≥ 0) x y)) →
  circle_C x y :=
sorry

theorem minimum_PQ_dot_MQ (x y : ℝ) :
  circle_C x y →
  ∃ Qx Qy : ℝ, (minimum (P_to_Q_dot_M_to_Q Qx Qy)) :=
sorry

theorem lines_OP_parallel_AB :
  ∃ k : ℝ, slope (PA) k ∧ slope (PB) (-k) ∧ (parallel (OP) (AB)) :=
sorry

end circle_C_equation_minimum_PQ_dot_MQ_lines_OP_parallel_AB_l75_75878


namespace sum_of_two_digit_numbers_ending_in_25_l75_75670

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75670


namespace farthest_vertex_square_l75_75574

noncomputable def center_pos : ℝ × ℝ := (5, 5)
noncomputable def area_square : ℝ := 16
noncomputable def dilation_center : ℝ × ℝ := (3, 3)
noncomputable def scale_factor : ℝ := 3

theorem farthest_vertex_square :
  let c := center_pos,
      a := area_square,
      d_center := dilation_center,
      k := scale_factor in
  (∃ E' F' G' H' : ℝ × ℝ,
    let side := Real.sqrt a,
        E := (c.1 - side / 2, c.2 + side / 2),
        F := (c.1 - side / 2, c.2 - side / 2),
        G := (c.1 + side / 2, c.2 + side / 2),
        H := (c.1 + side / 2, c.2 - side / 2),
        dilated := λ P, d_center + k • (P - d_center) in
    E' = dilated E ∧
    F' = dilated F ∧
    G' = dilated G ∧
    H' = dilated H ∧
    G' = (15, 15) ∧
    ∀ Q ∈ {E', F', G', H'}, Real.sqrt (Q.1^2 + Q.2^2) ≤ Real.sqrt (15^2 + 15^2)) :=
sorry

end farthest_vertex_square_l75_75574


namespace choose_4_from_15_is_1365_l75_75961

theorem choose_4_from_15_is_1365 : nat.choose 15 4 = 1365 :=
by
  sorry

end choose_4_from_15_is_1365_l75_75961


namespace scientists_opinions_converge_to_same_l75_75792

/-- Define the structure for the scientists' opinions and friendships. -/
structure Conference :=
  (n : ℕ)
  (n_ge_3 : 3 ≤ n)
  (friends : Fin n → Fin n → Prop)
  (mutual_friends : ∀ i j, friends i j ↔ friends j i)
  (self_friendship : ∀ i, ¬ friends i i)
  (partition_property : ∀ g1 g2 : Finset (Fin n), (g1.nonempty ∧ g2.nonempty ∧ g1 ∪ g2 = Finset.univ) →
    (∃ i1 i2, i1 ∈ g1 ∧ i2 ∈ g1 ∧ friends i1 i2) ∧ 
    (∃ j1 j2, j1 ∈ g1 ∧ j2 ∈ g2 ∧ friends j1 j2))

/-- Define the update function for the scientists' opinions. -/
def update_opinion (friends : Fin n → Fin n → Prop) (opinions : Fin n → ℕ) 
  (i : Fin n) : ℕ :=
  let friend_opinions := (Finset.filter (friends i) Finset.univ).image opinions
  in friend_opinions.to_finset.sum / (friend_opinions.card |> Nat.div_up 1 1)

/-- Define the theorem to prove the convergence of scientists' opinions. -/
theorem scientists_opinions_converge_to_same (conf : Conference) 
  (initial_opinions : Fin n → ℕ) : 
  ∃ t : ℕ, ∀ i j : Fin n, update_opinion conf.friends (λ k, (fin.rec_on t initial_opinions)) i = update_opinion conf.friends (λ k, (fin.rec_on t initial_opinions)) j :=
sorry

end scientists_opinions_converge_to_same_l75_75792


namespace least_integer_to_add_l75_75216

theorem least_integer_to_add (n : ℕ) (h : n = 725) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 :=
by
  use 5
  split
  · exact Nat.lt_succ_self 4
  · rw [h]
    norm_num
    sorry

end least_integer_to_add_l75_75216


namespace how_many_oxen_c_put_l75_75282

variables (oxen_a oxen_b months_a months_b rent total_rent c_share x : ℕ)
variable (H : 10 * 7 = oxen_a)
variable (H1 : 12 * 5 = oxen_b)
variable (H2 : 3 * x = months_a)
variable (H3 : 70 + 60 + 3 * x = months_b)
variable (H4 : 280 = total_rent)
variable (H5 : 72 = c_share)

theorem how_many_oxen_c_put : x = 15 :=
  sorry

end how_many_oxen_c_put_l75_75282


namespace find_a3_geometric_sequence_max_value_l75_75383

noncomputable def sequence (a : ℕ → ℝ) := 
  ∀ n ≥ 2, (a n = (a n.succ + a (n - 1)) / 2 ∨ a n.succ = (a n + a (n - 1)) / 2) ∧
            (∀ n ≥ 2, a n ≥ 0)

theorem find_a3 (a : ℕ → ℝ) (h_seq : sequence a) :
  a 1 = 5 → a 2 = 3 → a 4 = 2 → a 3 = 1 ∨ a 3 = 4 :=
sorry

theorem geometric_sequence (a : ℕ → ℝ) (h_seq : sequence a) :
  a 1 = 0 → a 4 = 0 → a 7 = 0 → a 2 > 0 → a 5 > 0 → a 8 > 0 →
  ∃ q, q = 1 / 4 ∧ a 5 = a 2 * q ∧ a 8 = a 5 * q :=
sorry

theorem max_value (a : ℕ → ℝ) (h_seq : sequence a) :
  a 1 = 1 → a 2 = 2 →
  ∃ r s t, 2 < r ∧ r < s ∧ s < t ∧ a r = 0 ∧ a s = 0 ∧ a t = 0 →
  a (r+1) + a (s+1) + a (t+1) ≤ 21 / 64 :=
sorry

end find_a3_geometric_sequence_max_value_l75_75383


namespace triangle_congruence_l75_75371

open_locale euclidean_geometry

variables {A B C A' B' C' : Point ℝ}
variables (O : Point ℝ)

noncomputable def distance (A B : Point ℝ) := euclidean_distance A B

axiom side_lengths :
  distance A B = Real.sqrt 7 ∧ distance B C = Real.sqrt 13 ∧ distance C A = Real.sqrt 19

axiom circle_conditions :
  distance A A' = 1/3 ∧ distance B B' = 2/3 ∧ distance C C' = 1

theorem triangle_congruence :
  ∃ (A' B' C' : Point ℝ), 
  distance A' B' = distance A B ∧ distance B' C' = distance B C ∧ distance C' A' = distance C A :=
by {
  sorry
}

end triangle_congruence_l75_75371


namespace arithmetic_mean_of_p_and_q_l75_75583

variable (p q r : ℝ)

theorem arithmetic_mean_of_p_and_q
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22)
  (h3 : r - p = 24) :
  (p + q) / 2 = 10 :=
by
  sorry

end arithmetic_mean_of_p_and_q_l75_75583


namespace smallest_number_is_490000_l75_75598

noncomputable def smallest_number_with_75_divisors (a : ℕ) : Prop :=
  (35 ∣ a) ∧ (Nat.divisors a).length = 75

theorem smallest_number_is_490000 : ∃ a : ℕ, smallest_number_with_75_divisors a ∧ a = 490000 := by
  sorry

end smallest_number_is_490000_l75_75598


namespace last_person_is_Dee_l75_75575

-- Define the students in a circle
inductive Student
| Alice
| Brad
| Carl
| Dee
| Eve
| Fay
| Gus
| Hal

open Student

-- Define the counting and elimination rules
def eliminated (n : ℕ) : Prop :=
  (n % 5 = 0) ∨ (n % 7 = 0) ∨ (n.to_digits.contains 5) ∨ (n.to_digits.contains 7)

-- Define the function to determine the last person standing
noncomputable def lastPersonStanding : Student :=
sorry

-- Problem statement in Lean
theorem last_person_is_Dee :
  lastPersonStanding = Dee :=
sorry

end last_person_is_Dee_l75_75575


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75657

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75657


namespace min_value_of_f_l75_75877

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x + 5

theorem min_value_of_f :
  ∃ x ∈ Icc 0 2, f x = 1/2 :=
sorry

end min_value_of_f_l75_75877


namespace value_of_a_l75_75398

noncomputable theory
open Classical

theorem value_of_a (a : ℝ) (x : ℝ) (h : x ≠ 0) (h_term : ∑ k in finset.range (6 + 1), ((binom 6 k) * (a * (sqrt x))^(6 - k) * ((- 1)^(k) * (1 / (sqrt x))^k)) = -160) : a = 2 :=
sorry

end value_of_a_l75_75398


namespace largest_binom_coeff_l75_75184

theorem largest_binom_coeff (n : ℕ) :
  (nat.choose 10 3) + (nat.choose 10 4) = nat.choose 11 n → n = 7 :=
by sorry

end largest_binom_coeff_l75_75184


namespace evaluate_transformed_expression_at_2_l75_75315

-- Define the original transformation of the expression
def transform (x : ℝ) : ℝ := (2 * x - 2) / (x + 2)

-- Define the expression after replacement
def transformed_expression (x : ℝ) : ℝ := (2 * transform x - 2) / (transform x + 2)

-- Evaluate the transformed expression at x = 2
theorem evaluate_transformed_expression_at_2 : transformed_expression 2 = -2 / 5 :=
by
  -- Proof will be provided here in a real Lean environment
  sorry

end evaluate_transformed_expression_at_2_l75_75315


namespace least_integer_to_add_l75_75214

theorem least_integer_to_add (n : ℕ) (h : n = 725) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 :=
by
  use 5
  split
  · exact Nat.lt_succ_self 4
  · rw [h]
    norm_num
    sorry

end least_integer_to_add_l75_75214


namespace karl_sum_of_coefficients_l75_75050

theorem karl_sum_of_coefficients :
  let polynomials := {P : ℤ[x] // P.monic ∧ 17 ∈ P.roots ∧ ∀ r ∈ P.roots, |17 - r| ≤ 17},
  ∑ P in polynomials, P.coeff 0 + P.coeff 1 + P.coeff 2 = 8960 :=
sorry

end karl_sum_of_coefficients_l75_75050


namespace trig_inequality_l75_75101

theorem trig_inequality (x : ℝ) : 
  -1/4 ≤ 5 * (cos x)^2 - 5 * (cos x)^4 + 5 * (sin x) * (cos x) + 1 ∧
  5 * (cos x)^2 - 5 * (cos x)^4 + 5 * (sin x) * (cos x) + 1 ≤ 19/4 :=
by
  sorry

end trig_inequality_l75_75101


namespace rectangle_width_decrease_l75_75934

theorem rectangle_width_decrease (a b : ℝ) (p x : ℝ) 
  (hp : p ≥ 0) (hx : x ≥ 0)
  (area_eq : a * b = (a * (1 + p / 100)) * (b * (1 - x / 100))) :
  x = (100 * p) / (100 + p) := 
by
  sorry

end rectangle_width_decrease_l75_75934


namespace bike_to_tractor_ratio_l75_75117

variable (speed_of_bike speed_of_tractor speed_of_car : ℝ)

def problem_conditions : Prop :=
  let ratio_car_bike := 9 / 5 in
  let dist_tractor := 575 in
  let time_tractor := 23 in
  let dist_car := 450 in
  let time_car := 5 in
  speed_of_car = dist_car / time_car ∧
  speed_of_tractor = dist_tractor / time_tractor ∧ 
  speed_of_car = ratio_car_bike * speed_of_bike ∧
  ∃ k : ℝ, speed_of_bike = k * speed_of_tractor

theorem bike_to_tractor_ratio (h : problem_conditions speed_of_bike speed_of_tractor speed_of_car) : 
  speed_of_bike / speed_of_tractor = 2 := by
  sorry

end bike_to_tractor_ratio_l75_75117


namespace max_value_of_b_l75_75053

-- Definitions and Assumptions
def is_origin (O : ℝ × ℝ) := O = (0, 0)
def line_eq (b : ℝ) (p : ℝ × ℝ) := p.2 = p.1 + b
def circle_eq (p : ℝ × ℝ) := p.1^2 + p.2^2 = 4

-- Conditions
variables {O : ℝ × ℝ} {b : ℝ} {P1 P2 : ℝ × ℝ}

-- Origin is defined
axiom origin_is_O : is_origin O

-- Line intersecting circle at two distinct points
axiom line_intersects_circle :
  ∃ (P1 P2 : ℝ × ℝ), P1 ≠ P2 ∧ line_eq b P1 ∧ line_eq b P2 ∧ circle_eq P1 ∧ circle_eq P2

-- Inequality condition
axiom inequality_condition :
  ∥P1 - P2∥ ≥ ∥(P1 - O) + (P2 - O)∥

-- The statement to prove the maximum value of b
theorem max_value_of_b : b ≤ 2 :=
sorry -- Proof placeholder

end max_value_of_b_l75_75053


namespace largest_divisor_of_consecutive_odd_product_l75_75624

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : n > 0) :
  315 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) := 
sorry

end largest_divisor_of_consecutive_odd_product_l75_75624


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75655

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75655


namespace number_of_ways_to_choose_4_captains_from_15_l75_75954

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end number_of_ways_to_choose_4_captains_from_15_l75_75954


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75729

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75729


namespace smaller_mold_radius_l75_75263

-- Define a constant for the volume of the large hemisphere
def volume_large_hemisphere : ℝ := (2 / 3) * Real.pi * (1^3)

-- Define the volume function for a hemisphere given its radius
def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * Real.pi * (r^3)

-- Define the condition of total volume for 27 smaller hemispheres being equal to the large hemisphere volume
theorem smaller_mold_radius :
  ∃ r : ℝ, 27 * volume_hemisphere(r) = volume_large_hemisphere ∧ r = 1 / 3 := 
by
  sorry

end smaller_mold_radius_l75_75263


namespace alloy_problem_solution_l75_75974

theorem alloy_problem_solution (x y k n : ℝ) (H_weight : k * 4 * x + n * 3 * y = 10)
    (H_ratio : (kx + ny)/(k * 3 * x + n * 2 * y) = 3/7) :
    k * 4 * x = 4 :=
by
  -- Proof to be provided
  sorry

end alloy_problem_solution_l75_75974


namespace cos_difference_l75_75571

variable (c d : ℝ)

def cos_54 := cos (54 * Real.pi / 180)
def cos_18 := cos (18 * Real.pi / 180)

theorem cos_difference : c = cos_54 → d = cos_18 → cos_54 - cos_18 = 1 / 2 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry  -- Proof to be constructed

end cos_difference_l75_75571


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75654

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75654


namespace fraction_RP_PQ_eq_3_2_l75_75039

-- Define the entities and proof parameters
variables (X Y Z P Q N R : Type) [normed_space ℝ X] [normed_space ℝ Y] [normed_space ℝ Z]
  (dist_XY : dist X Y = 15)
  (dist_XZ : dist X Z = 20)
  (N_midpoint_YZ : N = midpoint ℝ Y Z)
  (P_on_XZ : P ∈ line_segment ℝ X Z)
  (Q_on_XY : Q ∈ line_segment ℝ X Y)
  (XP_eq_3XQ : dist X P = 3 * dist X Q)
  (R_intersection_PQ_XN : R ∈ line_segment ℝ P Q ∧ R ∈ line_segment ℝ X N)

-- The theorem statement
theorem fraction_RP_PQ_eq_3_2 : ∀ (X Y Z P Q N R : Type) [normed_space ℝ X] [normed_space ℝ Y] [normed_space ℝ Z]
  (dist_XY : dist X Y = 15)
  (dist_XZ : dist X Z = 20)
  (N_midpoint_YZ : N = midpoint ℝ Y Z)
  (P_on_XZ : P ∈ line_segment ℝ X Z)
  (Q_on_XY : Q ∈ line_segment ℝ X Y)
  (XP_eq_3XQ : dist X P = 3 * dist X Q)
  (R_intersection_PQ_XN : R ∈ line_segment ℝ P Q ∧ R ∈ line_segment ℝ X N),
  dist R P / dist P Q = 3 / 2 :=
by {
  sorry
}

end fraction_RP_PQ_eq_3_2_l75_75039


namespace price_of_turban_l75_75762

theorem price_of_turban {T : ℝ}:
  (3 / 4) * (90 + T) = 45 + T → T = 90 :=
by
  intro h
  have : 67.5 + (3 / 4) * T = 45 + T := by linarith
  sorry

end price_of_turban_l75_75762


namespace opposite_neg_pos_six_l75_75139

def opposite (n : Int) : Int := -n

theorem opposite_neg_pos_six : opposite (-(+6)) = 6 := by
  sorry

end opposite_neg_pos_six_l75_75139


namespace imaginary_part_l75_75012

theorem imaginary_part (x y : ℝ) (h : (x - Complex.i) * Complex.i = y + 2 * Complex.i) :
  Complex.imaginary (x + y * Complex.i) = 1 :=
sorry

end imaginary_part_l75_75012


namespace f_decreasing_on_interval_l75_75588

def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

-- Define the interval for which we want to prove monotonicity.
def I : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The main theorem statement
theorem f_decreasing_on_interval : 
  ∀ x y ∈ I, x < y → f y < f x :=
sorry

end f_decreasing_on_interval_l75_75588


namespace analytical_expression_of_f_range_of_m_l75_75420

-- Define the vectors and function with given conditions
def a (ω x : ℝ) : ℝ × ℝ := (Real.sqrt 3, Real.cos (2 * ω * x))
def b (ω x : ℝ) : ℝ × ℝ := (Real.sin (2 * ω * x), 1)
def f (ω x : ℝ) : ℝ := a ω x.1 * b ω x.1 + a ω x.2 * b ω x.2

-- The first problem: prove the analytical expression of f(x)
theorem analytical_expression_of_f (ω : ℝ) (hω : ω > 0) (T : ℝ) (hT : f ω x + y = 2 * y => Periodic f T) :
  (f 1 x = 2 * Real.sin (2 * x + (Real.pi / 6))) := sorry

-- The second problem: prove the range of m
theorem range_of_m (m : ℝ) (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi / 2) (h2 : f 1 x + m ≤ 3) :
  (m ≤ 1) := sorry

end analytical_expression_of_f_range_of_m_l75_75420


namespace evaluate_expression_l75_75327

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem evaluate_expression : (factorial (factorial 4)) / factorial 4 = factorial 23 :=
by sorry

end evaluate_expression_l75_75327


namespace max_days_for_same_shift_l75_75264

open BigOperators

-- We define the given conditions
def nurses : ℕ := 15
def shifts_per_day : ℕ := 24 / 8
noncomputable def total_pairs : ℕ := (nurses.choose 2)

-- The main statement to prove
theorem max_days_for_same_shift : 
  35 = total_pairs / shifts_per_day := by
  sorry

end max_days_for_same_shift_l75_75264


namespace infinite_real_values_of_a_for_equal_roots_l75_75833

theorem infinite_real_values_of_a_for_equal_roots
  (a b : ℝ)
  (h : (b^2 - 4 * a * b = 0)) :
  ∃ a : ℝ, true :=
begin
  sorry
end

end infinite_real_values_of_a_for_equal_roots_l75_75833


namespace secant_segment_length_l75_75615

theorem secant_segment_length (R r : ℝ) :
    ∃ x, (2 * x) = (1 / 2) * (sqrt ((14 * R * r - R^2 - r^2) / 3)) :=
by
  sorry

end secant_segment_length_l75_75615


namespace least_positive_integer_to_multiple_of_5_l75_75226

theorem least_positive_integer_to_multiple_of_5 (n : ℕ) (h₁ : n = 725) :
  ∃ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 ∧ ∀ k : ℕ, (k > 0 ∧ (725 + k) % 5 = 0) → m ≤ k :=
begin
  use 5,
  sorry
end

end least_positive_integer_to_multiple_of_5_l75_75226


namespace least_positive_integer_k_l75_75212

theorem least_positive_integer_k (n : ℕ):
  n = 101 → 
  let total_diagonals := (n * (n - 3)) / 2 in
  total_diagonals = 4949 → 
  ∃ k : ℕ, k = 4900 ∧ (∀ d : (fin 4949) → ℚ, 
    ∑ i in finset.range k, d i ≥ ∑ i in finset.range (4949 - k), d (k + i)) :=
begin
  assume n hn,
  assume total_diagonals ht,
  use 4900,
  split,
  { refl },
  { assume d,
    sorry
  }
end

end least_positive_integer_k_l75_75212


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75679

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75679


namespace minimum_value_of_f_l75_75844

def f (x : ℝ) : ℝ := 2 * cos x * cos x + sin x + 3

theorem minimum_value_of_f : ∃ x : ℝ, f x = 2 :=
by
  sorry

end minimum_value_of_f_l75_75844


namespace count_squares_containing_shaded_l75_75975

theorem count_squares_containing_shaded {m n : ℕ} (grid : matrix (fin m) (fin n) ℕ) (shaded : fin m × fin n) :
  m = 5 → n = 5 → shaded = (3, 3) → ∃ count, count = 14 :=
by
  intros hm hn hshaded
  have h1 : 1 = 1 := rfl
  have h2 : 4 = 4 := rfl
  have h3 : 4 = 4 := rfl
  have h4 : 4 = 4 := rfl
  have h5 : 1 = 1 := rfl
  use (1 + 4 + 4 + 4 + 1)
  rw [h1, h2, h3, h4, h5]
  norm_num
  exact h4

end count_squares_containing_shaded_l75_75975


namespace page_problem_insufficient_information_l75_75549

theorem page_problem_insufficient_information
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (x y : ℕ)
  (O E : ℕ)
  (h1 : total_problems = 450)
  (h2 : finished_problems = 185)
  (h3 : remaining_pages = 15)
  (h4 : O + E = remaining_pages)
  (h5 : O * x + E * y = total_problems - finished_problems) :
  ∀ (x y : ℕ), O * x + E * y = 265 → x = x ∧ y = y :=
by
  sorry

end page_problem_insufficient_information_l75_75549


namespace apples_to_sell_for_profit_l75_75542

def cost_price_per_apple : ℝ := 1 / 4
def selling_price_per_apple : ℝ := 2 / 5
def profit_per_apple : ℝ := selling_price_per_apple - cost_price_per_apple
def profit_target : ℝ := 15

theorem apples_to_sell_for_profit (x : ℝ) :
  profit_per_apple * x = profit_target ↔ x = 100 :=
by 
  sorry

end apples_to_sell_for_profit_l75_75542


namespace equation_of_line_l_minimizing_chord_length_l75_75265

noncomputable def minimizeChordLength (l : Set (ℝ×ℝ)) : Prop :=
  ∃ (A B : ℝ × ℝ), l A ∧ l B ∧ (A ≠ B) ∧ ((A.1 + 1) ^ 2 + (A.2 - 2) ^ 2 = 5) ∧
    ((B.1 + 1) ^ 2 + (B.2 - 2) ^ 2 = 5) ∧
    ∀ l', ((l' = SetOf λ (p : ℝ × ℝ), p.1 - p.2 + 5 = 0) → False) →
    ∃ (C D : ℝ × ℝ), l' C ∧ l' D ∧ (C ≠ D) ∧ ((C.1 + 1) ^ 2 + (D.2 - 2) ^ 2 = 5) ∧
      ((D.1 + 1) ^ 2 + (C.2 - 2) ^ 2 = 5) ∧ dist (C, D) < dist (A, B)

theorem equation_of_line_l_minimizing_chord_length : 
  minimizeChordLength (SetOf λ (p : ℝ × ℝ), p.1 - p.2 + 5 = 0) := 
  sorry

end equation_of_line_l_minimizing_chord_length_l75_75265


namespace V3_value_l75_75821

noncomputable def polynomial_horner_value (c : List ℤ) (x : ℤ) : ℤ :=
c.foldr (λ a acc, a + x * acc) 0

theorem V3_value :
  polynomial_horner_value [2, 5, 6, 23, -8, 10, -3] 2 = 71 := by
  sorry

end V3_value_l75_75821


namespace problem_solution_set_l75_75403

theorem problem_solution_set (a b : ℝ) (h : ∀ x, (1 < x ∧ x < 2) ↔ ax^2 + x + b > 0) : a + b = -1 :=
sorry

end problem_solution_set_l75_75403


namespace no_more_than_50_distinct_plants_no_exactly_50_distinct_plants_l75_75025

-- Part (a)
theorem no_more_than_50_distinct_plants (P : Fin 51 → Fin 100 → Bool) (h_diff : ∀ i j, i ≠ j → Nat.count (λ k, P i k ≠ P j k) ≥ 51) :
  false :=
sorry

-- Part (b)
theorem no_exactly_50_distinct_plants (P : Fin 50 → Fin 101 → Bool) (h_diff : ∀ i j, i ≠ j → Nat.count (λ k, P i k ≠ P j k) ≥ 52) :
  false :=
sorry

end no_more_than_50_distinct_plants_no_exactly_50_distinct_plants_l75_75025


namespace find_x_l75_75400

theorem find_x (x : ℝ) (M : ℝ) (hM : M = max {1, 2, 3, x}) (hSum : M = 1 + 2 + 3 + x) : x = -3 :=
sorry

end find_x_l75_75400


namespace table_ratio_l75_75274

theorem table_ratio (L W : ℝ) (h1 : L * W = 128) (h2 : L + 2 * W = 32) : L / W = 2 :=
by
  sorry

end table_ratio_l75_75274


namespace area_of_G₁G₂G₃_eq_five_l75_75054

-- Defining the geometric terms and conditions
variables {Point : Type*} [metric_space Point] [inner_product_space ℝ Point]
variables (A B C P G₁ G₂ G₃ : Point)

def is_centroid (X Y Z : Point) (G : Point) : Prop :=
  ∃ u v, u + v + G = (1 : ℝ) / 3 * (X + Y + Z) -- Definition of centroid in barycentric coordinates

-- Defining the conditions
axiom point_in_triangle (P : Point) (A B C : Point) : Prop :=
  ∃ u v w, u + v + w = 1 ∧ 0 < u ∧ 0 < v ∧ 0 < w ∧ u = P.1 ∧ v = P.2 ∧ w = P.3

axiom area_of_triangle_ABC (A B C : Point) : ℝ := 45

-- The theorem that needs to be proven
theorem area_of_G₁G₂G₃_eq_five
  (h1 : point_in_triangle P A B C)
  (h2 : is_centroid P B C G₁)
  (h3 : is_centroid P C A G₂)
  (h4 : is_centroid P A B G₃)
  (h5 : area_of_triangle_ABC A B C = 45) :
  area (triangle G₁ G₂ G₃) = 5 := 
sorry

end area_of_G₁G₂G₃_eq_five_l75_75054


namespace name_tags_distribution_l75_75605

open Nat

-- Define the statement of the problem
theorem name_tags_distribution (n : ℕ) : 
  let total_distributions := factorial n in
  let special_distributions := total_distributions - 4^n + n * (3^(n-1) + 2^(n-1) + 1) in
  special_distributions = total_distributions - ∑ k in range (n+1), (-1)^k * choose n k * derangements (n - k) :=
sorry

end name_tags_distribution_l75_75605


namespace abc_eq_efg_l75_75073

-- Define the points where the incircle touches the sides of triangle ABC
variables (A B C E F G P : Type) [EuclideanSpace ℝ (Fin 2)]

-- Define the distances from P to the sides of the triangle
variables (a b c e f g : ℝ)

-- Define the conditions
axiom incircle_touches : ∃ (E F G : EucCirclePoint), touches_incircle_to_sides_triangle ABC E F G
axiom distances_to_sides : ∃ (a b c : ℝ), distances_from_P_to_sides P ABC a b c
axiom distances_to_lines : ∃ (e f g : ℝ), distances_from_P_to_lines P FG EG EF e f g

-- Define the theorem/lemma we need to prove
theorem abc_eq_efg (h1: ∃ E F G : EucCirclePoint, touches_incircle_to_sides_triangle ABC E F G)
                    (h2: ∃ a b c : ℝ, distances_from_P_to_sides P ABC a b c)
                    (h3: ∃ e f g : ℝ, distances_from_P_to_lines P FG EG EF e f g) :
                    a * b * c = e * f * g :=
by sorry

end abc_eq_efg_l75_75073


namespace perpendicular_vectors_X_value_l75_75537

open Real

-- Define vectors a and b, and their perpendicularity condition
def vector_a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def vector_b : ℝ × ℝ := (1, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The theorem statement
theorem perpendicular_vectors_X_value (x : ℝ) 
  (h : dot_product (vector_a x) vector_b = 0) : 
    x = -2 / 3 :=
by sorry

end perpendicular_vectors_X_value_l75_75537


namespace total_goals_in_five_matches_l75_75793

-- Define average goals and total goals
def avg_goals (n : ℕ) (total_goals : ℕ) : ℚ := total_goals / n

variable (A : ℚ)

theorem total_goals_in_five_matches (h1 : 3 + avg_goals 4 (A * 4) > avg_goals 4 (A * 4)) :
  3 + (avg_goals 4 (A * 4)) = avg_goals 5 (4 * A + 3) + 0.2 →
  4 * 2 + 3 = 11 :=
by
  sorry

end total_goals_in_five_matches_l75_75793


namespace cost_of_green_pill_l75_75817

-- Let the cost of a green pill be g and the cost of a pink pill be p
variables (g p : ℕ)
-- Beth takes two green pills and one pink pill each day
-- A green pill costs twice as much as a pink pill
-- The total cost for the pills over three weeks (21 days) is $945

theorem cost_of_green_pill : 
  (2 * g + p) * 21 = 945 ∧ g = 2 * p → g = 18 :=
by
  sorry

end cost_of_green_pill_l75_75817


namespace intersection_A_B_l75_75527

-- Define the sets A and the function f
def A : Set ℤ := {-2, 0, 2}
def f (x : ℤ) : ℤ := |x|

-- Define the set B as the image of A under the function f
def B : Set ℤ := {b | ∃ a ∈ A, f a = b}

-- State the property that every element in B has a pre-image in A
axiom B_has_preimage : ∀ b ∈ B, ∃ a ∈ A, f a = b

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {0, 2} :=
by sorry

end intersection_A_B_l75_75527


namespace average_speed_second_part_l75_75782

theorem average_speed_second_part (total_distance : ℕ) (speed_40 : ℕ) (total_time : ℝ) (distance_40 : ℕ) (distance_remaining : ℕ) (time_remaining : ℝ) (avg_speed : ℝ) :
  total_distance = 250 →
  speed_40 = 40 →
  total_time = 5.2 →
  distance_40 = 124 →
  distance_remaining = total_distance - distance_40 →
  time_remaining = total_time - (distance_40 / speed_40) →
  avg_speed = distance_remaining / time_remaining →
  avg_speed = 60 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  simp [h1, h2, h3, h4] at h5 h6
  rw [h5, h6]
  norm_num
  exact h7
  sorry

end average_speed_second_part_l75_75782


namespace choose_4_captains_from_15_l75_75965

def num_ways_to_choose_captains : ℕ := 15.choose 4

theorem choose_4_captains_from_15 : num_ways_to_choose_captains = 1365 := by
  sorry

end choose_4_captains_from_15_l75_75965


namespace problem_statement_l75_75388

-- Definitions for the conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def functional_eq (f : ℝ → ℝ) : Prop := ∀ x, f (3 / 2 - x) = f x

def f_of_neg_two (f : ℝ → ℝ) : Prop := f (-2) = -3

def seq_a : ℕ → ℝ
| 1 := -1
| n := 2 * seq_a (n - 1) - 1

def seq_S (n : ℕ) : ℝ := 2 * seq_a n + n

-- Final theorem to prove
theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_func_eq : functional_eq f) 
  (h_f_of_neg_two : f_of_neg_two f) :
  f (seq_a 5) + f (seq_a 6) = 3 := 
sorry

end problem_statement_l75_75388


namespace tournament_prizes_l75_75454

theorem tournament_prizes :
  let outcomes := 2 in
  let matches := 5 in
  outcomes ^ matches = 32 := by
-- This is just a statement, so we add a proof placeholder.
  sorry

end tournament_prizes_l75_75454


namespace Martian_calendar_holidays_l75_75496

/-- 
Prove that the minister can declare certain days as holidays such that:
1. At least 2/7 of the number of days in each week are holidays.
2. The total number of holidays over the entire period of 2019 weeks does not exceed half of the total days.
-/
theorem Martian_calendar_holidays :
  ∃ (holiday : Fin 7 → Bool), 
    (∀ (week_days : Nat), (7 ≤ week_days ∧ week_days ≤ 13) → ((∑ i in Finset.finRange 7, if holiday i then 1 else 0) ≥ 2 * week_days / 7)) ∧
    (let total_days := 2019 * 7 in ∑ i in Finset.finRange total_days, if holiday (i % 7) then 1 else 0 <= total_days / 2) :=
sorry

end Martian_calendar_holidays_l75_75496


namespace expression_even_nat_l75_75551

theorem expression_even_nat (m n : ℕ) : 
  2 ∣ (5 * m + n + 1) * (3 * m - n + 4) := 
sorry

end expression_even_nat_l75_75551


namespace construct_half_and_third_l75_75379

-- Given a line segment AB, we want to construct segments half the length and a third of AB using only a compass.
theorem construct_half_and_third (A B : Point) (segment_AB : LineSegment A B) :
  ∃ F H : Point, F ∈ segment_AB ∧ H ∈ segment_AB ∧ 
    dist A F = (1 / 2) * dist A B ∧ dist A H = (1 / 3) * dist A B :=
sorry

end construct_half_and_third_l75_75379


namespace T_simplified_l75_75056

-- Define the polynomial expression T
def T (x : ℝ) : ℝ := (x-2)^4 - 4*(x-2)^3 + 6*(x-2)^2 - 4*(x-2) + 1

-- Prove that T simplifies to (x-3)^4
theorem T_simplified (x : ℝ) : T x = (x - 3)^4 := by
  sorry

end T_simplified_l75_75056


namespace largest_n_binom_sum_l75_75198

open Nat

theorem largest_n_binom_sum (n : ℕ) 
  (h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n) : n = 7 :=
by 
  sorry

end largest_n_binom_sum_l75_75198


namespace optionD_is_quadratic_l75_75752

variable (x : ℝ)

-- Original equation in Option D
def optionDOriginal := (x^2 + 2 * x = 2 * x^2 - 1)

-- Rearranged form of Option D's equation
def optionDRearranged := (-x^2 + 2 * x + 1 = 0)

theorem optionD_is_quadratic : optionDOriginal x → optionDRearranged x :=
by
  intro h
  -- The proof steps would go here, but we use sorry to skip it
  sorry

end optionD_is_quadratic_l75_75752


namespace largest_n_binom_sum_l75_75196

open Nat

theorem largest_n_binom_sum (n : ℕ) 
  (h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n) : n = 7 :=
by 
  sorry

end largest_n_binom_sum_l75_75196


namespace sum_of_two_digit_numbers_ending_in_25_l75_75664

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75664


namespace general_admission_price_l75_75803

theorem general_admission_price :
  ∃ x : ℝ,
    ∃ G V : ℕ,
      VIP_price = 45 ∧ Total_tickets_sold = 320 ∧ Total_revenue = 7500 ∧ VIP_tickets_less = 276 ∧
      G + V = Total_tickets_sold ∧ V = G - VIP_tickets_less ∧ 45 * V + x * G = Total_revenue ∧ x = 21.85 :=
sorry

end general_admission_price_l75_75803


namespace athena_spent_correct_amount_l75_75034

-- Define the conditions
def num_sandwiches : ℕ := 3
def price_per_sandwich : ℝ := 3
def num_drinks : ℕ := 2
def price_per_drink : ℝ := 2.5

-- Define the total cost as per the given conditions
def total_cost : ℝ :=
  (num_sandwiches * price_per_sandwich) + (num_drinks * price_per_drink)

-- The theorem that states the problem and asserts the correct answer
theorem athena_spent_correct_amount : total_cost = 14 := 
  by
    sorry

end athena_spent_correct_amount_l75_75034


namespace max_value_f_l75_75596

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (real.sqrt 3) * real.sin x + real.cos x

-- State the theorem that the maximum value of the function is 2
theorem max_value_f : ∃ x : ℝ, f x = 2 := sorry

end max_value_f_l75_75596


namespace average_percentage_l75_75243

theorem average_percentage (n1 n2 : ℕ) (avg1 avg2 : ℕ) (h1 : n1 = 15) (h2 : n2 = 10) (h3 : avg1 = 70) (h4 : avg2 = 95) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 80 :=
by
  simp [h1, h2, h3, h4]
  sorry

end average_percentage_l75_75243


namespace proof_problem_l75_75083

def f (a b c : ℕ) : ℕ :=
  a * 100 + b * 10 + c

def special_op (a b c : ℕ) : ℕ :=
  f (a * b) (b * c / 10) (b * c % 10)

theorem proof_problem :
  special_op 5 7 4 - special_op 7 4 5 = 708 := 
    sorry

end proof_problem_l75_75083


namespace garden_perimeter_is_56_l75_75229

-- Define the conditions
def garden_width : ℕ := 12
def playground_length : ℕ := 16
def playground_width : ℕ := 12
def playground_area : ℕ := playground_length * playground_width
def garden_length : ℕ := playground_area / garden_width
def garden_perimeter : ℕ := 2 * (garden_length + garden_width)

-- Statement to prove
theorem garden_perimeter_is_56 :
  garden_perimeter = 56 := by
sorry

end garden_perimeter_is_56_l75_75229


namespace annual_income_of_A_l75_75597

-- Define the variables
def C_m := 15000

-- Define the monthly income of B
def B_m := C_m + 0.12 * C_m

-- Define the monthly income of A
def A_m := (5 / 2) * B_m

-- Define the annual income of A
def A_a := 12 * A_m

-- The theorem that we need to prove
theorem annual_income_of_A : A_a = 504000 := by
  sorry

end annual_income_of_A_l75_75597


namespace least_positive_integer_to_multiple_of_5_l75_75218

theorem least_positive_integer_to_multiple_of_5 : ∃ (n : ℕ), n > 0 ∧ (725 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m :=
by
  sorry

end least_positive_integer_to_multiple_of_5_l75_75218


namespace Yoe_speed_is_40_mph_l75_75114

theorem Yoe_speed_is_40_mph
    (Teena_speed : ℝ)
    (Teena_behind : ℝ)
    (Time : ℝ)
    (Teena_ahead : ℝ)
    (H1 : Teena_speed = 55)
    (H2 : Teena_behind = 7.5)
    (H3 : Time = 1.5)
    (H4 : Teena_ahead = 15):
    let Yoe_speed := (Teena_speed * Time - (Teena_behind + Teena_ahead)) / Time in Yoe_speed = 40 :=
by
    sorry

end Yoe_speed_is_40_mph_l75_75114


namespace sum_of_two_digit_numbers_ending_in_25_l75_75637

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75637


namespace distance_A_B_on_curve_C_is_4_l75_75407

open Real

theorem distance_A_B_on_curve_C_is_4 :
  ∀ (ρ1 ρ2 : ℝ), 
  (∃ (π_3 : ℝ), ρ1 = 4 * sin π_3 ∧ π_3 = (π/3)) →
  (∃ (5π_6 : ℝ), ρ2 = 4 * sin 5π_6 ∧ 5π_6 = (5*π/6)) →
  dist (ρ1 * cos (π/3), 2 + ρ1 * sin (π/3)) (ρ2 * cos (5*π/6), 2 + ρ2 * sin (5*π/6)) = 4 :=
by
  intros ρ1 ρ2 h1 h2
  sorry

end distance_A_B_on_curve_C_is_4_l75_75407


namespace smallest_positive_solution_of_trig_eqn_l75_75855

theorem smallest_positive_solution_of_trig_eqn :
  ∃ x > 0, tan (3 * x) + tan (4 * x) = sec (4 * x) ∧ x = π / 17 :=
by
  sorry

end smallest_positive_solution_of_trig_eqn_l75_75855


namespace mike_spending_l75_75543

noncomputable def marbles_cost : ℝ := 9.05
noncomputable def football_cost : ℝ := 4.95
noncomputable def baseball_cost : ℝ := 6.52

noncomputable def toy_car_original_cost : ℝ := 6.50
noncomputable def toy_car_discount : ℝ := 0.20
noncomputable def toy_car_discounted_cost : ℝ := toy_car_original_cost * (1 - toy_car_discount)

noncomputable def puzzle_cost : ℝ := 3.25
noncomputable def puzzle_total_cost : ℝ := puzzle_cost -- 'buy one get one free' condition

noncomputable def action_figure_original_cost : ℝ := 15.00
noncomputable def action_figure_discounted_cost : ℝ := 10.50

noncomputable def total_cost : ℝ := marbles_cost + football_cost + baseball_cost + toy_car_discounted_cost + puzzle_total_cost + action_figure_discounted_cost

theorem mike_spending : total_cost = 39.47 := by
  sorry

end mike_spending_l75_75543


namespace exists_infinite_circles_l75_75065

-- Definitions and conditions
variables {α : Type*} [metric_space α] [normed_ring α] [normed_algebra ℝ α] 
variables {A B C : α} {Ω : set α}

axiom triangle_vertices_in_circle : ∀ (A B C : α), (A ∈ Ω) ∧ (B ∈ Ω) ∧ (C ∈ Ω)

theorem exists_infinite_circles (A B C : α) (Ω : set α) 
  (hABC_in_Ω : (A ∈ Ω) ∧ (B ∈ Ω) ∧ (C ∈ Ω)) :
  ∃ (ω : set α), (ω ⊆ Ω) ∧ (A ∈ ω) ∧ (B ∈ ω) ∧ (C ∈ interior ω) :=
sorry

end exists_infinite_circles_l75_75065


namespace minimum_bird_species_l75_75461

theorem minimum_bird_species (total_birds : ℕ) (h : total_birds = 2021) :
  ∃ (min_species : ℕ), min_species = 1011 ∧ 
  (∀ (species_array : array total_birds ℕ),
   ∀ i j : fin total_birds, 
   species_array[i] = species_array[j] → ((i ≠ j) →
   (abs (i - j) mod 2 = 0))) :=
sorry

end minimum_bird_species_l75_75461


namespace min_species_needed_l75_75468

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l75_75468


namespace largest_binom_coeff_l75_75185

theorem largest_binom_coeff (n : ℕ) :
  (nat.choose 10 3) + (nat.choose 10 4) = nat.choose 11 n → n = 7 :=
by sorry

end largest_binom_coeff_l75_75185


namespace dilation_transform_l75_75120

theorem dilation_transform (z c : ℂ) (k : ℝ) (h₀ : z = 0 - 2 * complex.I) (h₁: c = 1 + 2 * complex.I) (h₂ : k = 4) :
  (z - c = k * (0 - 2 * complex.I - c)) → (z = -3 - 14 * complex.I) :=
by
  intro h
  have h₃ : z - (1 + 2 * complex.I) = 4 * ((0 - 2 * complex.I) - (1 + 2 * complex.I)), by rw [h₀, h₁, h₂],
  have h₄ : z - (1 + 2 * complex.I) = -4 - 16 * complex.I, by simp [h₃],
  have h₅ : z = -4 - 16 * complex.I + 1 + 2 * complex.I, by rwa [h₄],
  have h₆ : z = -3 - 14 * complex.I, by simp [h₅],
  exact h₆

end dilation_transform_l75_75120


namespace pyramid_volume_l75_75566

theorem pyramid_volume (VW WX VZ : ℝ) (h1 : VW = 10) (h2 : WX = 5) (h3 : VZ = 8)
  (h_perp1 : ∀ (V W Z : ℝ), V ≠ W → V ≠ Z → Z ≠ W → W = 0 ∧ Z = 0)
  (h_perp2 : ∀ (V W X : ℝ), V ≠ W → V ≠ X → X ≠ W → W = 0 ∧ X = 0) :
  let area_base := VW * WX
  let height := VZ
  let volume := 1 / 3 * area_base * height
  volume = 400 / 3 := by
  sorry

end pyramid_volume_l75_75566


namespace sum_first_15_constant_l75_75020

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

-- Define the sum of specific terms in the arithmetic sequence
def specific_sum (a d : ℝ) : ℝ :=
  arithmetic_sequence a d 2 + arithmetic_sequence a d 10 + arithmetic_sequence a d 12

-- Condition: The sum a_2 + a_10 + a_12 is a given constant
axiom h : ∃ c : ℝ, specific_sum a d = c

-- We want to prove that S_15 is also a constant
theorem sum_first_15_constant (a d : ℝ) : ∃ k : ℝ, sum_arithmetic_sequence a d 15 = k :=
  sorry

end sum_first_15_constant_l75_75020


namespace car_B_distance_from_A_when_A_reaches_B_l75_75826

noncomputable def distance_AB : ℝ := 100

def car_A_continues_to_B_after_meeting (v1 : ℝ) (t : ℝ) : Prop :=
  t = 4 / v1

def car_B_continues_to_A_after_meeting (v2 : ℝ) (t : ℝ) : Prop :=
  t = 1 / v2

theorem car_B_distance_from_A_when_A_reaches_B (v1 v2 : ℝ) (t1 t2 : ℝ) :
  distance_AB = 100 ∧ car_A_continues_to_B_after_meeting v1 t1 ∧ car_B_continues_to_A_after_meeting v2 t2 ∧
  t1 = 4 ∧ t2 = 1 ∧ v2 = 2 * v1 →
  100 ∈ set.range (λ t, v2 * t) :=
by
  sorry

end car_B_distance_from_A_when_A_reaches_B_l75_75826


namespace value_of_expression_l75_75763

theorem value_of_expression (x y : ℤ) (hx : x = -5) (hy : y = 8) : 2 * (x - y) ^ 2 - x * y = 378 :=
by
  rw [hx, hy]
  -- The proof goes here.
  sorry

end value_of_expression_l75_75763


namespace min_species_needed_l75_75473

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l75_75473


namespace license_plate_combinations_l75_75430

theorem license_plate_combinations : 
  let letters := 26 
  let letters_and_digits := 36 
  let middle_character_choices := 2
  3 * letters * letters_and_digits * middle_character_choices = 1872 :=
by
  sorry

end license_plate_combinations_l75_75430


namespace train_pass_bridge_time_l75_75760

theorem train_pass_bridge_time
  (train_length : ℕ)
  (bridge_length : ℕ)
  (train_speed_kmh : ℕ)
  (convert_kmh_to_ms : ℕ → ℝ := λ kmh, kmh * 1000 / 3600)
  (total_distance : ℕ := train_length + bridge_length)
  (train_speed_ms : ℝ := convert_kmh_to_ms train_speed_kmh)
  (time_to_pass : ℝ := total_distance / train_speed_ms) :
  train_length = 360 → 
  bridge_length = 140 → 
  train_speed_kmh = 75 →
  time_to_pass ≈ 24 :=
by {
  sorry
}

end train_pass_bridge_time_l75_75760


namespace ellipse_standard_equation_l75_75888

-- Definitions of the conditions
def isEllipse (c : ℝ × ℝ) (f1 f2 : ℝ × ℝ) (P : ℝ × ℝ) (d : ℝ) : Prop :=
  let (x, y) := P
  let c1 := (x - fst f1)^2 + (y - snd f1)^2
  let c2 := (x - fst f2)^2 + (y - snd f2)^2
  c1 + c2 = d^2

def onEllipse (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Statement of the problem
theorem ellipse_standard_equation : 
  ∀ (a b c : ℝ) (f1 f2 : ℝ × ℝ) (P : ℝ × ℝ),
  f1 = (-6, 0) →
  f2 = (6, 0) →
  P = (3 * Real.sqrt 2, 4) →
  isEllipse (0, 0) f1 f2 P 12 →
  2 * a = 12 →
  a = 6 →
  b^2 = 32 →
  onEllipse a b P :=
by
  intros a b c f1 f2 P h_f1 h_f2 h_P h_isEllipse h_2a h_a h_b2
  -- Proof omitted: not required as per instructions
  sorry

end ellipse_standard_equation_l75_75888


namespace choose_4_from_15_is_1365_l75_75960

theorem choose_4_from_15_is_1365 : nat.choose 15 4 = 1365 :=
by
  sorry

end choose_4_from_15_is_1365_l75_75960


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75660

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75660


namespace compare_binary_digits_l75_75312

def numDigits_base2 (n : ℕ) : ℕ :=
  (Nat.log2 n) + 1

theorem compare_binary_digits :
  numDigits_base2 1600 - numDigits_base2 400 = 2 := by
  sorry

end compare_binary_digits_l75_75312


namespace max_distance_circle_to_line_l75_75595

open Real

theorem max_distance_circle_to_line :
  let circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y + 1 = 0
  let line_eq (x y : ℝ) := x - y = 2
  ∃ (M : ℝ), (∀ x y, circle_eq x y → ∀ (d : ℝ), (line_eq x y → M ≤ d)) ∧ M = sqrt 2 + 1 :=
by
  sorry

end max_distance_circle_to_line_l75_75595


namespace largest_circle_area_l75_75800

theorem largest_circle_area (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) :
  ∃ r : ℝ, (2 * π * r = 60) ∧ (π * r ^ 2 = 900 / π) := 
sorry

end largest_circle_area_l75_75800


namespace alpha_beta_value_l75_75013

variable (α β : ℝ)

def quadratic (x : ℝ) := x^2 + 2 * x - 2005

axiom roots_quadratic_eq : quadratic α = 0 ∧ quadratic β = 0

theorem alpha_beta_value :
  α^2 + 3 * α + β = 2003 :=
by sorry

end alpha_beta_value_l75_75013


namespace area_of_triangle_AMN_l75_75042

theorem area_of_triangle_AMN (A B C M N : Type) [Point A] [Point B] [Point C] [Midpoint M B C] [Midpoint N A C]
  (h_area_ABC : area_triangle A B C = 180) : area_triangle A M N = 45 :=
by
  sorry

end area_of_triangle_AMN_l75_75042


namespace angle_ALB_acute_l75_75994

variables {A B C D L : Type}
variables [TangentialQuadrilateral A B C D]
variables [Intersect AC BD L]

-- Given that AB = CD > BC
axiom AB_CD_eq : AB = CD
axiom AB_BC_gt : AB > BC

theorem angle_ALB_acute : ∠ALB < 90 := 
sorry

end angle_ALB_acute_l75_75994


namespace plane_zero_l75_75992

-- Define the real-valued function on the plane
variable {R : Type*} [LinearOrderedField R]
variable {Point : Type*}
variable (f : Point → R)

-- Condition: For every square ABCD, the sum of f at the corners is zero
def condition (A B C D : Point) : Prop :=
  ∀ (A B C D : Point), (/* ABCD forms a square */) → f A + f B + f C + f D = 0

-- Problem statement: Proving f(P) = 0 for all points P in the plane
theorem plane_zero (H : ∀ (A B C D : Point), (/* ABCD forms a square */) → f A + f B + f C + f D = 0) (P : Point) : 
  f P = 0 :=
by
  sorry

end plane_zero_l75_75992


namespace product_units_tens_not_divisible_by_5_l75_75812

-- Define the list of four-digit numbers
def numbers : List ℕ := [4750, 4760, 4775, 4785, 4790]

-- Define a function to check if a number is divisible by 5
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Define a function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Define a function to extract the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Statement: The product of the units digit and the tens digit of the number
-- that is not divisible by 5 in the list is 0
theorem product_units_tens_not_divisible_by_5 : 
  ∃ n ∈ numbers, ¬divisible_by_5 n ∧ (units_digit n * tens_digit n = 0) :=
by sorry

end product_units_tens_not_divisible_by_5_l75_75812


namespace Isabel_initial_flowers_l75_75984

-- Constants for conditions
def b := 7  -- Number of bouquets after wilting
def fw := 10  -- Number of wilted flowers
def n := 8  -- Number of flowers in each bouquet

-- Theorem statement
theorem Isabel_initial_flowers (h1 : b = 7) (h2 : fw = 10) (h3 : n = 8) : 
  (b * n + fw = 66) := by
  sorry

end Isabel_initial_flowers_l75_75984


namespace sum_of_two_digit_numbers_ending_in_25_l75_75662

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75662


namespace sa_to_austin_passes_l75_75818

theorem sa_to_austin_passes (austin_sa_interval : ℕ) 
                            (sa_austin_interval : ℕ) 
                            (travel_time : ℕ)
                            (austin_first_bus : ℕ)
                            (sa_first_bus : ℕ)
                            (austin_start : ℕ) 
                            (sa_start : ℕ) 
                            (count : ℕ) :
  austin_sa_interval = 2 ∧ sa_austin_interval = 1 ∧ travel_time = 6 ∧ 
  austin_first_bus = 6 ∧ sa_first_bus = 6 ∧ 
  austin_start = 0 ∧ sa_start = 30 →
  count = 2 :=
begin
  sorry
end

end sa_to_austin_passes_l75_75818


namespace increasing_interval_f_l75_75873

def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem increasing_interval_f :
  ∀ x : ℝ, 0 < x ∧ x < π → 0 < x ∧ x < π / 4 → 0 < f x :=
by
  intros x hx1 hx2
  sorry

end increasing_interval_f_l75_75873


namespace even_numbers_count_l75_75170

theorem even_numbers_count : 
  let digits := {0, 1, 2, 3, 4}
  have ∀ d, d ∈ digits → even d ↔ d = 0 ∨ d = 2 ∨ d = 4 ∧ d ≠ 0 ∧ d ≠ 2 ∧ d ≠ 4
  have no_repeated_digits {0, 1, 2, 3, 4} ––
let units_digits := {0, 2, 4} → True := 
  163 := sorry

end even_numbers_count_l75_75170


namespace inflection_point_symmetry_sum_l75_75865

noncomputable def f (x : ℝ) : ℝ := (1 / 3)*x^3 - (1 / 2)*x^2 + 3*x - (5 / 12)

theorem inflection_point_symmetry_sum :
  (∑ i in Finset.range 2014, f (i.succ / 2015)) = 2014 :=
sorry

end inflection_point_symmetry_sum_l75_75865


namespace expansion_15th_term_l75_75365

noncomputable def binom : ℕ → ℕ → ℤ
| n, k := (nat.choose n k).to_int

theorem expansion_15th_term : binom 20 14 * (-1)^6 * (complex.I)^14 = -38760 := 
begin
  sorry
end

end expansion_15th_term_l75_75365


namespace coeff_x3_term_sum_of_coeffs_l75_75585

theorem coeff_x3_term (x: ℝ) : 
  (polynomial.expand ℝ (polynomial.X + (2 : ℝ)) * (polynomial.expand ℝ (polynomial.X + (1 : ℝ)) ^ 6)).coeff 3 = 55 := 
by sorry

theorem sum_of_coeffs (x: ℝ) :
  (polynomial.expand ℝ (polynomial.X + (2 : ℝ)) * (polynomial.expand ℝ (polynomial.X + (1 : ℝ)) ^ 6)).eval 1 = 192 :=
by sorry

end coeff_x3_term_sum_of_coeffs_l75_75585


namespace nancy_initial_bottle_caps_l75_75081

theorem nancy_initial_bottle_caps (found additional_bottle_caps: ℕ) (total_bottle_caps: ℕ) (h1: additional_bottle_caps = 88) (h2: total_bottle_caps = 179) : 
  (total_bottle_caps - additional_bottle_caps) = 91 :=
by
  sorry

end nancy_initial_bottle_caps_l75_75081


namespace transformation_y1_transformation_y2_l75_75385

theorem transformation_y1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ 4 * x ∧ 4 * x ≤ 4 := 
by 
    intros
    split
    {
        linarith,
    }
    {
        linarith,
    }

theorem transformation_y2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → -4 ≤ 5 * x - 4 ∧ 5 * x - 4 ≤ 1 :=
by 
    intros
    split
    {
        linarith,
    }
    {
        linarith,
    }

end transformation_y1_transformation_y2_l75_75385


namespace joy_quadrilateral_rod_problem_l75_75991

-- Defining the specific rods and the condition for forming a quadrilateral.
def rods_on_table : List ℕ := [4, 10, 22]
def rod_lengths : List ℕ := List.range 40
def remaining_rods : List ℕ := rod_lengths.diff rods_on_table

noncomputable def valid_rod_count : ℕ :=
  (remaining_rods.filter (λ d, 8 < d ∧ d < 36)).length

theorem joy_quadrilateral_rod_problem : valid_rod_count = 25 :=
by sorry

end joy_quadrilateral_rod_problem_l75_75991


namespace sum_of_integers_whose_squares_end_with_25_l75_75706

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75706


namespace problem1_problem2_l75_75912

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- First proof problem: f(a) - f(a + 1) = -2a - 1 
theorem problem1 (a : ℝ) : f(a) - f(a + 1) = -2a - 1 :=
by sorry

-- Second proof problem: values of x for which f(x) = x + 3
theorem problem2 (x : ℝ) : (f(x) = x + 3) ↔ (x = -1 ∨ x = 2) :=
by sorry

end problem1_problem2_l75_75912


namespace choose_4_from_15_is_1365_l75_75962

theorem choose_4_from_15_is_1365 : nat.choose 15 4 = 1365 :=
by
  sorry

end choose_4_from_15_is_1365_l75_75962


namespace max_cos2_sin_expression_l75_75359

noncomputable def max_expression_value : ℝ :=
  let f (θ : ℝ) := (Real.cos (θ / 2))^2 * (1 - Real.sin θ)
  Real.max (f 0) (Real.max (f (Real.pi / 4)) (f (Real.pi / 2)))

theorem max_cos2_sin_expression :
  ∃ θ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 ∧
  (∀ θ', 0 ≤ θ' ∧ θ' ≤ Real.pi / 2 → (Real.cos (θ' / 2))^2 * (1 - Real.sin θ') ≤ 1) ∧
  (Real.cos (θ / 2))^2 * (1 - Real.sin θ) = 1 :=
  by
  use [0]
  split; linarith
  split; linarith
  split
  assume θ' hθ'
  have h_bound : (Real.cos (θ' / 2))^2 ≥ 0 := by exact Real.cos_sq_nonneg (θ' / 2)
  have h_expr : 0 ≤ (1 - Real.sin θ') := by exact sub_nonneg.mpr (Real.sin_le_one θ')
  exact (mul_le_of_le_one_left h_bound h_expr)
  sorry

end max_cos2_sin_expression_l75_75359


namespace least_k_9_l75_75569

open Nat

noncomputable def u : ℕ → ℝ
| 0     => 1 / 3
| (n+1) => 3 * u n - 3 * (u n) * (u n)

def M : ℝ := 0.5

def acceptable_error (n : ℕ): Prop := abs (u n - M) ≤ 1 / 2 ^ 500

theorem least_k_9 : ∃ k, 0 ≤ k ∧ acceptable_error k ∧ ∀ j, (0 ≤ j ∧ j < k) → ¬acceptable_error j ∧ k = 9 := by
  sorry

end least_k_9_l75_75569


namespace scientific_notation_l75_75579

/-- Express 1,650,000,000 in scientific notation keeping three significant figures. -/
theorem scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ a ∧ a < 10) ∧ 1650000000 = a * 10^n ∧ a = 1.65 ∧ n = 9 :=
begin
  sorry
end

end scientific_notation_l75_75579


namespace correct_answer_l75_75112

variables (A B C : Type) -- Types for the teachers
variables (X Y Z : Type) -- Types for the schools
variables (Mathematics Latin Music : Type) -- Types for the subjects

-- Conditions as per the problem
variable (teacher_works : A → X ∨ Y ∨ Z) -- Each teacher works in one of the schools
variables (A_teaches_not_Mathematics : ∀ a : A, ¬ teaches a Mathematics)
variables (B_works_not_Z : ∀ b : B, ¬ works b Z)
variable (Z_teaches_Music : ∀ z : Z, teaches z Music)
variable (X_teaches_not_Latin : ∀ x : X, ¬ teaches x Latin)
variables (B_teaches_not_Mathematics : ∀ b : B, ¬ teaches b Mathematics)

-- The proof goals
theorem correct_answer (h1: (∃ b : B, teaches b Latin) ∧ (works_in Y) b) :
  (∃ b : B, teaches b Latin) ∧ (works_in Y) b :=
begin
  sorry
end

end correct_answer_l75_75112


namespace volume_of_earth_dug_out_l75_75791

noncomputable def volume_of_well (d h : ℝ) : ℝ :=
  let r := d / 2 in
  π * r^2 * h

theorem volume_of_earth_dug_out :
  volume_of_well 2 8 ≈ 25.13272 := by
  sorry

end volume_of_earth_dug_out_l75_75791


namespace proposition_true_iff_l75_75590

theorem proposition_true_iff :
  (∀ x y : ℝ, (xy = 1 → x = 1 / y ∧ y = 1 / x) → (x = 1 / y ∧ y = 1 / x → xy = 1)) ∧
  (∀ (A B : Set ℝ), (A ∩ B = B → A ⊆ B) → (A ⊆ B → A ∩ B = B)) ∧
  (∀ m : ℝ, (m > 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0) → (¬(∃ x : ℝ, x^2 - 2 * x + m = 0) → m ≤ 1)) :=
by
  sorry

end proposition_true_iff_l75_75590


namespace most_likely_outcome_is_draw_l75_75749

noncomputable def prob_A_win : ℝ := 0.3
noncomputable def prob_A_not_lose : ℝ := 0.7
noncomputable def prob_draw : ℝ := prob_A_not_lose - prob_A_win

theorem most_likely_outcome_is_draw :
  prob_draw = 0.4 ∧ prob_draw > prob_A_win ∧ prob_draw > (1 - prob_A_not_lose) :=
by
  -- proof goes here
  sorry

end most_likely_outcome_is_draw_l75_75749


namespace alex_initial_jelly_beans_l75_75287

variable (initial : ℕ)
variable (eaten : ℕ := 6)
variable (pile_weight : ℕ := 10)
variable (piles : ℕ := 3)

theorem alex_initial_jelly_beans :
  (initial - eaten = pile_weight * piles) → initial = 36 :=
by
  -- proof will be provided here
  sorry

end alex_initial_jelly_beans_l75_75287


namespace sum_first_21_terms_l75_75144

open Nat

/-- The sequence {a} satisfies the given conditions. -/
def seq_a : ℕ → ℕ
| 0 := 1
| n := if n % 2 = 1 then seq_a (n - 1) + (n / 2) + 1 else seq_a (n - 2) - (n / 2) + 1

theorem sum_first_21_terms :
  (finset.range 21).sum seq_a = 66 :=
sorry

end sum_first_21_terms_l75_75144


namespace jillian_largest_apartment_l75_75813

noncomputable def largest_apartment_size (budget : ℝ) (rate : ℝ) : ℝ :=
  budget / rate

theorem jillian_largest_apartment : largest_apartment_size 720 1.20 = 600 := by
  sorry

end jillian_largest_apartment_l75_75813


namespace least_positive_integer_to_multiple_of_5_l75_75225

theorem least_positive_integer_to_multiple_of_5 (n : ℕ) (h₁ : n = 725) :
  ∃ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 ∧ ∀ k : ℕ, (k > 0 ∧ (725 + k) % 5 = 0) → m ≤ k :=
begin
  use 5,
  sorry
end

end least_positive_integer_to_multiple_of_5_l75_75225


namespace group_members_lcm_l75_75805

theorem group_members_lcm : ∃ n, (n % 8 = 0) ∧ (n % 10 = 0) ∧ (n % 12 = 0) ∧ (n % 15 = 0) ∧ n = 120 :=
by
  existsi 120
  split
  -- n is a multiple of 8
  rw [Nat.mod_eq_zero_of_dvd (dvd_of_mul_left_dvd (dvd_lcm_left 8 10 12 15))] <|> rw [Nat.dvd_iff_mod_eq_zero]
  -- n is a multiple of 10
  split
  rw [Nat.mod_eq_zero_of_dvd (dvd_of_mul_right_dvd (dvd_lcm_left 10 12 15))] <|> rw [Nat.dvd_iff_mod_eq_zero]
  -- n is a multiple of 12
  split
  rw [Nat.mod_eq_zero_of_dvd (dvd_lcm_left 12 15)] <|> rw [Nat.dvd_iff_mod_eq_zero]
  -- n is a multiple of 15
  split
  rw [Nat.mod_eq_zero_of_dvd (dvd_lcm_right 12 15)] <|> rw [Nat.dvd_iff_mod_eq_zero]
  -- n = 120
  rfl
  sorry

end group_members_lcm_l75_75805


namespace sum_of_squares_ends_in_25_is_644_l75_75683

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75683


namespace det_Q_is_zero_l75_75520

-- Define the vector
def v : ℝ × ℝ := (3, -5)

-- Define the projection matrix Q
def Q : Matrix (Fin 2) (Fin 2) ℝ :=
  let c := v.1 * v.1 + v.2 * v.2
  (1 / c) • Matrix.of ![
    ![v.1 * v.1, v.1 * v.2],
    ![v.2 * v.1, v.2 * v.2]
  ]

-- Statement to prove that det Q = 0
theorem det_Q_is_zero : det Q = 0 :=
by
  sorry

end det_Q_is_zero_l75_75520


namespace eqn_neither_direct_nor_inverse_l75_75842

/-- Determine which one of the given equations y is neither directly nor inversely proportional to x. -/
theorem eqn_neither_direct_nor_inverse (x y : ℝ) :
  (x + 2 * y = 5 ∨ 4 * x * y = 12 ∨ x = 3 * y + 2 ∨ 2 * x + 3 * y = 15 ∨ x / y = 2) →
  (y ≠ k * x ∧ y * x ≠ k ∀ k : ℝ) → (2 * x + 3 * y = 15) :=
sorry

end eqn_neither_direct_nor_inverse_l75_75842


namespace distinct_colorings_eq_51_l75_75446

open Finset

-- Define the cyclic group C5
def C5 : Finset (Equiv.Perm (Fin 5)) := {
  { Equiv.refl _ },
  { Equiv.Perm.cycleExactlyFive (Fin 5) },
  { Equiv.Perm.cycleExactlyThree (Fin 5) },
  { Equiv.Perm.cycleThirtySix (Fin 5) },
  { Equiv.Perm.cycleTwelve (Fin 5) }
}

-- Define the number of fixed points of a permutation in C5
def fixed_points (σ : Equiv.Perm (Fin 5)) : Nat :=
  (Fin 5).filter (λ i, σ i = i).card

-- Define the number of valid colorings fixed by a permutation
def valid_colorings (σ : Equiv.Perm (Fin 5)) : Nat :=
  if σ = Equiv.refl _ then 3^5 else if σ = Equiv.Perm.cycleExactlyFive _ then 3 else if σ = Equiv.Perm.cycleExactlyThree _ ||
  σ = Equiv.Perm.cycleThirtySix _ then 9 else 3

-- Define Burnside's lemma application to this problem
def distinct_colorings : Nat :=
  (C5.sum valid_colorings) / 5

theorem distinct_colorings_eq_51 : distinct_colorings = 51 := by
  sorry

end distinct_colorings_eq_51_l75_75446


namespace green_fish_always_15_l75_75085

def total_fish (T : ℕ) : Prop :=
∃ (O B G : ℕ),
B = T / 2 ∧
O = B - 15 ∧
T = B + O + G ∧
G = 15

theorem green_fish_always_15 (T : ℕ) : total_fish T → ∃ G, G = 15 :=
by
  intro h
  sorry

end green_fish_always_15_l75_75085


namespace probability_of_digit_two_in_decimal_rep_3_div_11_l75_75090

theorem probability_of_digit_two_in_decimal_rep_3_div_11 :
  ∃ d : ℚ, d = 3 / 11 ∧ ∀ n : ℕ, n > 0 → probability_of_digit d n 2 = 1 / 2 := by
  sorry

end probability_of_digit_two_in_decimal_rep_3_div_11_l75_75090


namespace solve_for_b_l75_75436

theorem solve_for_b (n : ℝ) (b : ℝ) (h1 : n = 2 ^ 0.15) (h2 : n ^ b = 64) : b = 40 :=
sorry

end solve_for_b_l75_75436


namespace number_with_150_quarters_is_37_point_5_l75_75423

theorem number_with_150_quarters_is_37_point_5 (n : ℝ) (h : n / (1/4) = 150) : n = 37.5 := 
by 
  sorry

end number_with_150_quarters_is_37_point_5_l75_75423


namespace PERU_TST_IMO_2006_Q2_l75_75778

theorem PERU_TST_IMO_2006_Q2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ∀ n : ℕ, 0 < n → ⌊a * ⌊b * n⌋⌋ = n - 1) :
  ∃ a b : ℝ, (0 < a) ∧ (a < 1) ∧ irrational a ∧ b = 1/a ∧ ∀ n : ℕ, 0 < n → ⌊a * ⌊b * n⌋⌋ = n - 1 :=
begin
  sorry
end

end PERU_TST_IMO_2006_Q2_l75_75778


namespace rectangle_diagonal_point_problem_l75_75550

theorem rectangle_diagonal_point_problem
  (A B C D P O1 O2 : Point)
  (ABCD_rect : Rectangle A B C D)
  (AB_length : dist A B = 15)
  (BC_length : dist B C = 8)
  (P_on_AC : P ∈ Segment A C)
  (AP_gt_CP : dist A P > dist C P)
  (circumcenter_1 : IsCircumcenter O1 A B P)
  (circumcenter_2 : IsCircumcenter O2 C D P)
  (angle_90 : angle O1 P O2 = 90) :
  dist A P = 8.5 + (Real.sqrt 2) / 2 :=
sorry

end rectangle_diagonal_point_problem_l75_75550


namespace ratio_of_terms_l75_75997

open Nat

variables {a b : ℕ → ℝ} {S T : ℕ → ℝ}

def S (n : ℕ) : ℝ := (n / 2 : ℝ) * (2 * a 1 + (n - 1) * a 2)
def T (n : ℕ) : ℝ := (n / 2 : ℝ) * (2 * b 1 + (n - 1) * b 2)

theorem ratio_of_terms (h : ∀ n : ℕ, S n / T n = n / (2 * n + 1)) :
  a 5 / b 5 = 9 / 19 :=
by
  sorry

end ratio_of_terms_l75_75997


namespace greatest_value_of_x_l75_75851

theorem greatest_value_of_x (x : ℝ) : 
  (∃ (M : ℝ), (∀ y : ℝ, (y ^ 2 - 14 * y + 45 <= 0) → y <= M) ∧ (M ^ 2 - 14 * M + 45 <= 0)) ↔ M = 9 :=
by
  sorry

end greatest_value_of_x_l75_75851


namespace minimum_species_count_l75_75466

theorem minimum_species_count {n : ℕ} (h_n : n = 2021) 
  (h_cond : ∀ i j k : ℕ, i < j ∧ j < k → 
    birds i = birds k → birds j ≠ birds i → (j - i - 1) % 2 = 1 ∧ (k - j - 1) % 2 = 1) : 
  ∃ s : ℕ, s ≥ 1011 :=
begin
  sorry
end

end minimum_species_count_l75_75466


namespace evaluate_fg_2_l75_75063

def f(x : ℝ) : ℝ := (5 * x^2 + 7 * x + 12) / (x^2 - x + 2)
def g(x : ℝ) : ℝ := x - 2

theorem evaluate_fg_2 : f(g(2)) + g(f(2)) = 15.5 :=
by
  sorry

end evaluate_fg_2_l75_75063


namespace sum_of_integers_whose_squares_end_with_25_l75_75712

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75712


namespace sum_first_8_terms_of_sequence_l75_75885

theorem sum_first_8_terms_of_sequence :
  (∃ (a : ℕ → ℝ), (∀ n, 3 * a (n + 1) + a n = 0) ∧ (a 3 = 4 / 9) ∧ (finset.sum (finset.range 8) a = 3 * (1 - 3 ^ (-8)))) :=
sorry

end sum_first_8_terms_of_sequence_l75_75885


namespace chess_tournament_l75_75847

theorem chess_tournament (totalPlayers totalGames : ℕ) : totalPlayers = 50 ∧ totalGames = 61 ∧
  (∀ player, player.games ∈ {2, 3}) ∧
  (∀ (p1 p2 : Player), p1 ≠ p2 → ¬(p1 playedAgainst p2)) ∧
  let x := 22 in
  let y := totalPlayers - x in
  2 * totalGames = 3 * x + 2 * y →
  ∃ (p1 p2 : Player), p1.playedGames = 3 ∧ p2.playedGames = 3 ∧ p1 playedAgainst p2 :=
by
  sorry

end chess_tournament_l75_75847


namespace a_b_work_days_l75_75757

-- Definitions:
def work_days_a_b_together := 40
def work_days_a_alone := 12
def remaining_work_days_with_a := 9

-- Statement to be proven:
theorem a_b_work_days (x : ℕ) 
  (h1 : ∀ W : ℕ, W / work_days_a_b_together + remaining_work_days_with_a * (W / work_days_a_alone) = W) :
  x = 10 :=
sorry

end a_b_work_days_l75_75757


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75747

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75747


namespace chord_central_angle_l75_75789

-- Given that a chord divides the circumference of a circle in the ratio 5:7
-- Prove that the central angle opposite this chord can be either 75° or 105°
theorem chord_central_angle (x : ℝ) (h : 5 * x + 7 * x = 180) :
  5 * x = 75 ∨ 7 * x = 105 :=
sorry

end chord_central_angle_l75_75789


namespace sum_of_squares_ends_in_25_is_644_l75_75684

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75684


namespace projection_matrix_determinant_zero_l75_75522

open Matrix

theorem projection_matrix_determinant_zero :
  let a := 3
  let b := -5
  let Q := Scalar 1 / (a^2 + b^2) • !![![(a * a), (a * b)], ![(a * b), (b * b)]]
  det Q = 0 :=
by
  let a := 3
  let b := -5
  let Q := Scalar 1 / (a^2 + b^2) • !![![(a * a), (a * b)], ![(a * b), (b * b)]]
  show det Q = 0
  sorry

end projection_matrix_determinant_zero_l75_75522


namespace calculation_correct_l75_75172

theorem calculation_correct :
  15 * ( (1/3 : ℚ) + (1/4) + (1/6) )⁻¹ = 20 := sorry

end calculation_correct_l75_75172


namespace eval_ceil_floor_sum_l75_75347

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l75_75347


namespace food_to_rent_ratio_l75_75087

-- Define the conditions
def rent : ℝ := 600
def savings : ℝ := 2000
def taxes : ℝ := (2 / 5) * savings
def salary : ℝ := 4840

-- Denote the amount spent on food as F
variables (F : ℝ)

-- Define the mortgage as three times the food expenses
def mortgage : ℝ := 3 * F

-- Define the total expenses
def total_expenses : ℝ := rent + F + mortgage + savings + taxes

-- State the problem: Prove the ratio of food expenses to rent is 3:5
theorem food_to_rent_ratio : 
  salary = total_expenses → (F / rent) = 3 / 5 :=
by
  intros h
  sorry

end food_to_rent_ratio_l75_75087


namespace same_side_of_sign_l75_75771

-- Define the motion equations for the tourists
def S1 (t : ℝ) : ℝ := sqrt (1 + 6 * t) - 1

def S2 (t : ℝ) : ℝ :=
  if t >= 1/6 then 6 * (t - 1/6) else 0

-- The times at which the tourists reach the sign
def t1 := 4/3
def t2 := 1/2

-- Define the intervals when the tourists are on the same side of the sign
def same_side_intervals (t : ℝ) : Prop :=
  (t >= 0 ∧ t <= 1/2) ∨ (t >= 4/3)

theorem same_side_of_sign (t : ℝ) :
  same_side_intervals t ↔ (S1 t = 2 ∨ S2 t = 2) ∨ (S1 t <= -2 ∧ S2 t <= -2) :=
sorry

end same_side_of_sign_l75_75771


namespace arithmetic_sequence_geometric_condition_l75_75387

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + 3) 
  (h_geom : (a 1 + 6) ^ 2 = a 1 * (a 1 + 9)) : 
  a 2 = -9 :=
sorry

end arithmetic_sequence_geometric_condition_l75_75387


namespace total_distance_center_ball_l75_75252

-- Define the given conditions
def diameter : ℝ := 6
def radius := diameter / 2

def R1 : ℝ := 120
def R2 : ℝ := 50
def R3 : ℝ := 70

def adjusted_R1 := R1 - radius
def adjusted_R2 := R2 + radius
def adjusted_R3 := R3 - radius

-- Prove the total distance traveled by the center of the ball
theorem total_distance_center_ball :
  (adjusted_R1 + adjusted_R2 + adjusted_R3) * Real.pi = 237 * Real.pi :=
by
  sorry

end total_distance_center_ball_l75_75252


namespace unattainable_y_value_l75_75862

theorem unattainable_y_value :
  ∀ (y x : ℝ), (y = (1 - x) / (2 * x^2 + 3 * x + 4)) → (∀ x, 2 * x^2 + 3 * x + 4 ≠ 0) → y ≠ 0 :=
by
  intros y x h1 h2
  -- Proof to be provided
  sorry

end unattainable_y_value_l75_75862


namespace bisector_length_l75_75357

variables {a b : ℝ}

theorem bisector_length (ha : a > 0) (hb : b > 0) :
  let CK := (CK : ℝ) in CK = (a * b * Real.sqrt 2) / (a + b) :=
sorry

end bisector_length_l75_75357


namespace largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75207

open Nat

theorem largest_n_binom_10_3_10_4_eq_binom_11_n :
  ∃ n, (n ≤ 11) ∧ (∑ i in finset.range 11, if i = 4 then 1 else 0) = 1 ∧
  (n = 7) := by
sorry

end largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75207


namespace last_four_digits_of_2_pow_1965_l75_75355

theorem last_four_digits_of_2_pow_1965 :
  ∃ x : ℕ, x < 10000 ∧ 2^1965 % 10000 = x ∧ x = 3125 :=
by
  have h1 : 2^1965 % 16 = 0 := by sorry
  have h2 : 2^1965 % 625 = 239 := by sorry
  have h3 : x ≡ 0 [MOD 16] := h1
  have h4 : x ≡ 239 [MOD 625] := h2
  use 3125
  split
  { norm_num }
  split
  { rw [← Nat.modeq_iff_modeq_of_dvd 3125 0 16, ← Nat.modeq_iff_modeq_of_dvd 3125 239 625],
    apply @Nat.ChineseRemainder 0 239 16 625, 
    exact dvd_refl 16, 
    exact dvd_refl 625 }
  { norm_num }

end last_four_digits_of_2_pow_1965_l75_75355


namespace inequality_solution_l75_75769

theorem inequality_solution (x : ℝ) : 
  -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 3 ≤ x ∧ x < 4 → 
  (x + 6 ≥ 0) ∧ (x + 1 > 0) ∧ (5 - x > 0) ∧ (x ≠ 0) ∧ (x ≠ 1) ∧ (x ≠ 4) ∧
  ( (x - 3) / ((x - 1) * (4 - x)) ≥ 0 ) :=
sorry

end inequality_solution_l75_75769


namespace walking_running_ratio_l75_75269

theorem walking_running_ratio (d_w d_r : ℝ) (h1 : d_w / 4 + d_r / 8 = 3) (h2 : d_w + d_r = 16) :
  d_w / d_r = 1 := by
  sorry

end walking_running_ratio_l75_75269


namespace choose_15_4_l75_75955

/-- The number of ways to choose 4 captains from a team of 15 people is 1365. -/
theorem choose_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end choose_15_4_l75_75955


namespace count_four_digit_numbers_with_digit_sum_4_l75_75249

theorem count_four_digit_numbers_with_digit_sum_4 : 
  ∃ n : ℕ, (∀ (x1 x2 x3 x4 : ℕ), 
    x1 + x2 + x3 + x4 = 4 ∧ x1 ≥ 1 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0 →
    n = 20) :=
sorry

end count_four_digit_numbers_with_digit_sum_4_l75_75249


namespace zero_in_interval_l75_75775

def f (x : ℝ) : ℝ := -x^3 - 3 * x + 5

theorem zero_in_interval :
  (∀ x y, x < y → f x > f y) →
  f 1 > 0 →
  f 2 < 0 →
  ∃ c, c ∈ set.Ioo 1 2 ∧ f c = 0 := by
  intros h_mono h_f1 h_f2
  sorry

end zero_in_interval_l75_75775


namespace least_pos_int_for_multiple_of_5_l75_75224

theorem least_pos_int_for_multiple_of_5 (n : ℕ) (h1 : n = 725) : ∃ x : ℕ, x > 0 ∧ (725 + x) % 5 = 0 ∧ x = 5 :=
by
  sorry

end least_pos_int_for_multiple_of_5_l75_75224


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75652

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75652


namespace choose_15_4_l75_75957

/-- The number of ways to choose 4 captains from a team of 15 people is 1365. -/
theorem choose_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end choose_15_4_l75_75957


namespace sum_of_two_digit_numbers_ending_in_25_l75_75636

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75636


namespace eval_expr_ceil_floor_l75_75333

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l75_75333


namespace choose_15_4_l75_75959

/-- The number of ways to choose 4 captains from a team of 15 people is 1365. -/
theorem choose_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end choose_15_4_l75_75959


namespace hyperbola_eccentricity_l75_75416

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (F₁ F₂ P : ℝ × ℝ)
  (h_hyperbola_eq : ∀ {x y : ℝ}, (x, y) = P → x^2 / a^2 - y^2 / b^2 = 1)
  (h_distance : dist P F₁ = 3 * dist P F₂) :
  1 < (real.sqrt (1 + (b^2 / a^2))) ∧ (real.sqrt (1 + (b^2 / a^2))) ≤ 2 :=
sorry

end hyperbola_eccentricity_l75_75416


namespace ticTacToeWinningDiagonals_l75_75447

-- Define the tic-tac-toe board and the conditions
def ticTacToeBoard : Type := Fin 3 × Fin 3
inductive Player | X | O

def isWinningDiagonal (board : ticTacToeBoard → Option Player) : Prop :=
  (board (0, 0) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 2) = some Player.O) ∨
  (board (0, 2) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 0) = some Player.O)

-- Define the main problem statement
theorem ticTacToeWinningDiagonals : ∃ (n : ℕ), n = 40 :=
  sorry

end ticTacToeWinningDiagonals_l75_75447


namespace polygon_area_l75_75271

theorem polygon_area (sides : ℕ) (perimeter : ℝ) (area : ℝ)
  (h_sides : sides = 36)
  (h_perimeter : perimeter = 72)
  (h_congruent : ∀ (i j : ℕ), i ≠ j → congruent (side_length i) (side_length j))
  (h_perpendicular : ∀ (i : ℕ), perpendicular (side_length i) (side_length ((i + 1) % sides))):
  area = 144 :=
sorry

end polygon_area_l75_75271


namespace eval_ceil_floor_sum_l75_75343

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l75_75343


namespace solve_equation_l75_75148

theorem solve_equation : ∀ x : ℝ, (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1) → x = 5 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l75_75148


namespace maximum_area_of_triangle_l75_75609

noncomputable def maximum_triangle_area
  (s1 s2 s3 : set (ℝ × ℝ))
  (h1 : metric.bounded s1)
  (h2 : metric.bounded s2)
  (h3 : metric.bounded s3)
  (h4 : (∃ X, X ∈ s1 ∧ X ∈ s2 ∧ X ∈ s3))
  (A B C : ℝ × ℝ)
  (hA : A ∈ s1 ∨ A ∈ s2 ∨ A ∈ s3)
  (hB : B ∈ s1 ∨ B ∈ s2 ∨ B ∈ s3)
  (hC : C ∈ s1 ∨ C ∈ s2 ∨ C ∈ s3)
  : ℝ :=
  let area := (3 * Real.sqrt 3) / 2 in
  area

theorem maximum_area_of_triangle
  (s1 s2 s3 : set (ℝ × ℝ))
  (h1 : metric.bounded s1)
  (h2 : metric.bounded s2)
  (h3 : metric.bounded s3)
  (h4 : (∃ X, X ∈ s1 ∧ X ∈ s2 ∧ X ∈ s3))
  (A B C : ℝ × ℝ)
  (hA : A ∈ s1 ∨ A ∈ s2 ∨ A ∈ s3)
  (hB : B ∈ s1 ∨ B ∈ s2 ∨ B ∈ s3)
  (hC : C ∈ s1 ∨ C ∈ s2 ∨ C ∈ s3)
  : (3 * Real.sqrt 3) / 2 = maximum_triangle_area s1 s2 s3 h1 h2 h3 h4 A B C hA hB hC :=
  by
    sorry

end maximum_area_of_triangle_l75_75609


namespace _l75_75834

open Int

def lcm_range (m n : ℕ) : ℕ :=
  Nat.foldl lcm 1 (Nat.Ico m (n + 1))

noncomputable theorem fraction_lcm :
  let A := lcm_range 15 25
  let B := lcm A (lcm_range 26 45)
  B / A = 3 * 37 * 41 * 43 :=
by
  sorry

end _l75_75834


namespace choose_4_from_15_is_1365_l75_75964

theorem choose_4_from_15_is_1365 : nat.choose 15 4 = 1365 :=
by
  sorry

end choose_4_from_15_is_1365_l75_75964


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75651

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75651


namespace min_consecutive_points_contains_three_colors_l75_75773

-- Define the problem conditions
constant P : ℕ → ℕ
constant color : ℕ → ℕ
constant num_points : ℕ := 100
constant num_colors : ℕ := 5

-- Define the properties
axiom points_range : ∀ n, 1 ≤ P n ∧ P n ≤ num_points
axiom color_range : ∀ n, 1 ≤ color n ∧ color n ≤ num_colors
axiom each_color_used_at_least_once : ∀ c, 1 ≤ c ∧ c ≤ num_colors → ∃ n, color n = c

-- The theorem to be proved
theorem min_consecutive_points_contains_three_colors :
  ∃ t, (∀ coloring, ∃ i, (P (i+1) = P i + 1) ∧
           (color (P i) ≠ color (P (i+1)) ∧ color (P (i+1)) ≠ color (P (i+2)) ∧ color (P i) ≠ color (P (i+2)))) → 
           t = 26 :=
sorry

end min_consecutive_points_contains_three_colors_l75_75773


namespace code_master_secret_codes_l75_75037

-- Define the number of colors and slots
def num_colors : ℕ := 8
def num_slots : ℕ := 5
def black : ℕ := 1
def non_black_colors : ℕ := 7

-- Define the restriction: Black can only be in the first two slots
def color_restriction (slots_with_black : ℕ) : Prop :=
  slots_with_black ∈ {0, 1, 2}

-- Define the function to calculate the number of codes with restrictions
noncomputable def restricted_combinations : ℕ :=
  (num_colors ^ 2) * (non_black_colors ^ 3)

-- Prove that the number of different secret codes is 21952 given the conditions
theorem code_master_secret_codes : restricted_combinations = 21952 :=
by
  sorry

end code_master_secret_codes_l75_75037


namespace not_divisible_by_3_or_4_l75_75069

theorem not_divisible_by_3_or_4 (n : ℤ) : 
  ¬ (n^2 + 1) % 3 = 0 ∧ ¬ (n^2 + 1) % 4 = 0 := 
by
  sorry

end not_divisible_by_3_or_4_l75_75069


namespace highest_pow2_divides_l75_75820

def v2 (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.find (λ k, n % (Nat.pow 2 k) ≠ 0) - 1

theorem highest_pow2_divides (a b : ℕ) (ha : a = 17) (hb : b = 13) :
  let n := a^5 - b^5
  v2 n = 2 := by
  sorry

end highest_pow2_divides_l75_75820


namespace solution_exists_l75_75765

def problem (p q r s : ℝ) : Prop :=
  p + q + r + s = 12000 ∧
  r = (2 / 3) * (p + q) ∧
  s = (1 / 4) * (p + q)

theorem solution_exists : ∃ (p q r s : ℝ), problem p q r s :=
by
  use [6260.87, 0, 4173.91, 1565.22]
  sorry

end solution_exists_l75_75765


namespace neither_coffee_nor_tea_l75_75300

theorem neither_coffee_nor_tea (total_businesspeople coffee_drinkers tea_drinkers both_drinkers : ℕ) 
    (h_total : total_businesspeople = 35)
    (h_coffee : coffee_drinkers = 18)
    (h_tea : tea_drinkers = 15)
    (h_both : both_drinkers = 6) :
    (total_businesspeople - (coffee_drinkers + tea_drinkers - both_drinkers)) = 8 := 
by
  sorry

end neither_coffee_nor_tea_l75_75300


namespace green_leaves_remaining_l75_75160

theorem green_leaves_remaining (initial_leaves_per_plant : ℕ) (num_plants : ℕ) (loss_fraction : ℚ) 
  (h₀ : initial_leaves_per_plant = 18)
  (h₁ : num_plants = 3)
  (h₂ : loss_fraction = 1 / 3) :
  (initial_leaves_per_plant - (initial_leaves_per_plant * loss_fraction).toNat) * num_plants = 36 :=
sorry

end green_leaves_remaining_l75_75160


namespace units_digit_of_N_is_8_l75_75055

def product_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens * units

def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens + units

theorem units_digit_of_N_is_8 (N : ℕ) (hN_range : 10 ≤ N ∧ N < 100)
    (hN_eq : N = product_of_digits N * sum_of_digits N) : N % 10 = 8 :=
sorry

end units_digit_of_N_is_8_l75_75055


namespace tangent_line_slope_condition_l75_75414

noncomputable def exp_function (x m : ℝ) : ℝ := exp x - m * x + 1

theorem tangent_line_slope_condition (m : ℝ) :
  (¬∃ s : ℝ, exp s - m = -1 / exp 1) → m ≤ 1 / exp 1 :=
begin
  sorry
end

end tangent_line_slope_condition_l75_75414


namespace every_positive_integer_displayable_l75_75783

-- Definitions based on the conditions of the problem
def flip_switch_up (n : ℕ) : ℕ := n + 1
def flip_switch_down (n : ℕ) : ℕ := n - 1
def press_red_button (n : ℕ) : ℕ := n * 3
def press_yellow_button (n : ℕ) : ℕ := if n % 3 = 0 then n / 3 else n
def press_green_button (n : ℕ) : ℕ := n * 5
def press_blue_button (n : ℕ) : ℕ := if n % 5 = 0 then n / 5 else n

-- Prove that every positive integer can appear on the calculator display
theorem every_positive_integer_displayable : ∀ n : ℕ, n > 0 → 
  ∃ m : ℕ, m = n ∧
    (m = flip_switch_up m ∨ m = flip_switch_down m ∨ 
     m = press_red_button m ∨ m = press_yellow_button m ∨ 
     m = press_green_button m ∨ m = press_blue_button m) := 
sorry

end every_positive_integer_displayable_l75_75783


namespace container_weight_l75_75780

-- Definition of the problem conditions
def weight_of_copper_bar : ℕ := 90
def weight_of_steel_bar := weight_of_copper_bar + 20
def weight_of_tin_bar := weight_of_steel_bar / 2

-- Formal statement to be proven
theorem container_weight (n : ℕ) (h1 : weight_of_steel_bar = 2 * weight_of_tin_bar)
  (h2 : weight_of_steel_bar = weight_of_copper_bar + 20)
  (h3 : weight_of_copper_bar = 90) :
  20 * (weight_of_copper_bar + weight_of_steel_bar + weight_of_tin_bar) = 5100 := 
by sorry

end container_weight_l75_75780


namespace find_pension_in_terms_l75_75281

def pension_proportionality_constant (a b p q : ℕ) (k x : ℝ) : Prop :=
  (k * (x + 2 * a) ^ 2 = k * x ^ 2 + 2 * p) ∧
  (k * (x + 3 * b) ^ 2 = k * x ^ 2 + 3 * q)

theorem find_pension_in_terms (a b p q : ℕ) (kx : ℝ) :
  (pension_proportionality_constant a b p q kx x) → 
  kx = -((bp - 2aq) / (6 * ab)) :=
begin
  sorry
end

end find_pension_in_terms_l75_75281


namespace sum_of_two_digit_numbers_ending_in_25_l75_75669

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75669


namespace corresponds_to_location_of_neg_x_squared_l75_75434

theorem corresponds_to_location_of_neg_x_squared (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : 
  letter = 'c' :=
begin
  -- Definitions of necessary terms and their ranges
  have h3 : 0 < x^2, from by {
    apply pow_pos_of_pos _ 2,
    exact lt_of_lt_of_le (lt_add_one x) h2,
  },
  have h4 : x^2 < 1, from (pow_lt_one x (_root_.abs x)),
  have h5 : -1 < - (x^2), from neg_lt_neg h4,
  have h6 : - (x^2) < 0, from neg_lt_neg h3,
  -- Conclusion based on the given number line and comparisons
  exact (by analyze_number_line_standard_form h1 h2 h5 h6),
end

end corresponds_to_location_of_neg_x_squared_l75_75434


namespace no_solution_of_fractional_equation_l75_75110

theorem no_solution_of_fractional_equation (x : ℝ) : ¬ (x - 8) / (x - 7) - 8 = 1 / (7 - x) := 
sorry

end no_solution_of_fractional_equation_l75_75110


namespace gena_profit_l75_75247

def is_profitable (p1 x1 p2 x2 : ℝ) : Prop := 
  p1 > 0 ∧ p2 > 0 ∧ 
  (4 * x1 - 3 * p1 - 44 = 0) ∧ 
  (p2^2 - 12 * p2 + x2^2 - 8 * x2 + 43 = 0) ∧
  let d := (Real.sqrt ((x1 - x2)^2 + (p1 - p2)^2)) in
  d = 6.2 ∧ (x1 + x2 - p1 - p2 > 0) ∧ (x1 + x2 - p1 - p2 = 13080 / 1000)

theorem gena_profit (p1 x1 p2 x2 : ℝ) :
  is_profitable p1 x1 p2 x2 := 
sorry

end gena_profit_l75_75247


namespace sum_of_two_digit_numbers_ending_in_25_l75_75668

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75668


namespace translate_line_up_l75_75286

theorem translate_line_up (x y : ℝ) (h : y = 2 * x - 3) : y + 6 = 2 * x + 3 :=
by sorry

end translate_line_up_l75_75286


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75673

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75673


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75696

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75696


namespace trigonometric_range_l75_75142

noncomputable def trigonometric_range_function (x : ℝ) : ℝ :=
  (sin x * cos x) / (1 + sin x - cos x)

theorem trigonometric_range :
  (∀ x : ℝ, sin x - cos x ≠ -1) →
  set.range trigonometric_range_function = 
    (set.Icc (-(sqrt 2 + 1) / 2) (-1) \ { -1 }) ∪ (set.Icc (-1) ((sqrt 2 - 1) / 2) \ { -1 }) :=
begin
  sorry,
end

end trigonometric_range_l75_75142


namespace four_digit_numbers_formed_2025_l75_75424

theorem four_digit_numbers_formed_2025 : 
  (let digits := [2, 0, 2, 5] in
  let valid_digits := [2, 5] in
  let count_permutations (d : ℕ) :=
    if d = 2 then (3! / 1!) else (3! / 2!) in
  list.sum (list.map count_permutations valid_digits) = 6 ) :=
sorry

end four_digit_numbers_formed_2025_l75_75424


namespace exists_smallest_subtriangle_l75_75113

variable {A B C K L M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
    [IsTriangle A B C]
    (K_on_BC : OnSegment K B C)
    (L_on_CA : OnSegment L C A)
    (M_on_AB : OnSegment M A B)

theorem exists_smallest_subtriangle :
  ∃ Δ ∈ {AML, BKM, CLK}, Area(Δ) ≤ (1/4 : ℝ) * Area(ABC) := sorry

end exists_smallest_subtriangle_l75_75113


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75695

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75695


namespace ornithological_park_species_l75_75477

/-- In an ornithological park, there are 2021 birds arranged in a row.
Each pair of birds of the same species has an even number of birds between them.
Prove that the smallest number of bird species is 1011. -/
theorem ornithological_park_species (n : ℕ) (h1 : n = 2021) 
  (h2 : ∀ s : ℕ, s ∈ {1..n} → (∀ x y : ℕ, x < y ∧ x ≠ y → (∀ z : ℕ, z ∈ ({x, y} : set ℕ) → even (y - x - 1))) ) 
  : s ≥ 1011 :=
sorry

end ornithological_park_species_l75_75477


namespace cos_squared_eq_one_l75_75111

theorem cos_squared_eq_one (x : ℝ) (k : ℤ) :
  cos(x)^2 + cos(2*x)^2 + cos(3*x)^2 = 1 ↔
    (∃ k : ℤ, x = k * π + π / 2) ∨
    (∃ k : ℤ, x = k * π / 2 + π / 4) ∨
    (∃ k : ℤ, x = k * π / 3 + π / 6) := by
  sorry

end cos_squared_eq_one_l75_75111


namespace area_transformation_l75_75057

-- Define the region T with given area
def area_T : ℝ := 9

-- Define the transformation matrix
def M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![3, 1],
  ![4, 5]
]

-- State the theorem
theorem area_transformation (T T' : Set (Fin 2 → ℝ)) (h₁ : MeasureTheory.MeasureSpace.volume T = 9) :
  let detM := M.det in
  MeasureTheory.MeasureSpace.volume T' = detM * area_T := 
by
  have detM := (3 * 5 - 1 * 4) -- Calculate determinant
  have h₂ : detM = 11 := rfl
  rw [h₂, h₁]
  sorry

end area_transformation_l75_75057


namespace binomial_sum_divisible_by_p_squared_l75_75433

theorem binomial_sum_divisible_by_p_squared {p k : ℕ} (hp : Nat.Prime p) (hp_gt3 : p > 3) (hk : k = (2 * p) / 3) :
  (Finset.sum (Finset.range (k + 1)) (λ i, Nat.choose p i)) % (p * p) = 0 := by
  sorry

end binomial_sum_divisible_by_p_squared_l75_75433


namespace randy_biscuits_l75_75556

theorem randy_biscuits : 
  let initial := 32 
  let father_gift := 13 
  let mother_gift := 15 
  let brother_ate := 20 
  let total := initial + father_gift + mother_gift 
  total - brother_ate = 40 := 
by
  let initial := 32 
  let father_gift := 13 
  let mother_gift := 15 
  let brother_ate := 20 
  let total := initial + father_gift + mother_gift 
  show total - brother_ate = 40 from sorry

end randy_biscuits_l75_75556


namespace friends_count_l75_75262

-- Define the given conditions
def initial_chicken_wings := 2
def additional_chicken_wings := 25
def chicken_wings_per_person := 3

-- Define the total number of chicken wings
def total_chicken_wings := initial_chicken_wings + additional_chicken_wings

-- Define the target number of friends in the group
def number_of_friends := total_chicken_wings / chicken_wings_per_person

-- The theorem stating that the number of friends is 9
theorem friends_count : number_of_friends = 9 := by
  sorry

end friends_count_l75_75262


namespace sum_of_squares_mod_five_l75_75525

theorem sum_of_squares_mod_five (b : Fin 100 → ℕ)
  (h1 : StrictMono b)
  (h2 : (∑ i, b i) = 123456) :
  (∑ i, (b i) ^ 2) % 5 = 1 := 
sorry

end sum_of_squares_mod_five_l75_75525


namespace largest_binom_n_l75_75188

theorem largest_binom_n : ∃ (n : ℕ), 
  n ≤ 11 ∧ 
  (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 n ∧ 
  ∀ m (h : m ≤ 11), (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 m → m ≤ n 
  :=
begin
  use 7,
  sorry
end

end largest_binom_n_l75_75188


namespace stddev_of_data_set_is_2_l75_75309

def dataSet : List ℝ := [5, 7, 7, 8, 10, 11]

def mean (data : List ℝ) : ℝ :=
  (data.sum) / (data.length)

def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (λ x => (x - m) ^ 2)).sum / (data.length)

def stddev (data : List ℝ) : ℝ :=
  real.sqrt (variance data)

theorem stddev_of_data_set_is_2 : stddev dataSet = 2 := by
  sorry

end stddev_of_data_set_is_2_l75_75309


namespace least_positive_integer_to_multiple_of_5_l75_75220

theorem least_positive_integer_to_multiple_of_5 : ∃ (n : ℕ), n > 0 ∧ (725 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m :=
by
  sorry

end least_positive_integer_to_multiple_of_5_l75_75220


namespace helen_baked_yesterday_l75_75421

theorem helen_baked_yesterday 
  (c_morning : ℕ) 
  (c_day_before_yesterday : ℕ) 
  (c_till_last_night : ℕ) :
  c_morning = 270 → c_day_before_yesterday = 419 → c_till_last_night = 450 → 
  (c_morning + c_day_before_yesterday - c_till_last_night = 239) :=
by
  intros h_morning h_day_before h_last_night
  rw [h_morning, h_day_before, h_last_night]
  exact calc
    270 + 419 - 450 = 689 - 450 : by rw add_comm -- 270 + 419 = 689
    ...           = 239 : by rw sub_comm -- 689 - 450 = 239
    

end helen_baked_yesterday_l75_75421


namespace max_collinear_points_l75_75381

theorem max_collinear_points (k : ℕ) (h : k > 0) :
  ∃ m : ℕ, (m = 2 * k) -> 
  (∀ (points : Finset (EuclideanSpace ℝ (2))) (h_size : points.card = 3 * k),
      (∀ l : AffineSubspace ℝ (EuclideanSpace ℝ 2), l ≤ ⊤ → l.direction.dim = 1 → points.filter (λ p, p ∈ l).card ≤ m) ->
      ∃ (triangle_sets : Finset (Finset (EuclideanSpace ℝ 2))), 
        triangle_sets.card = k ∧
        (∀ t ∈ triangle_sets, t.card = 3 ∧ 
         ∃ a b c: EuclideanSpace ℝ 2, a ∈ t ∧ b ∈ t ∧ c ∈ t ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
           ¬AffineSubspace ℝ 2 (AffineSpan ℝ (Finset.insert a (Finset.insert b {c}))) ≤ ⊤)) :=
sorry

end max_collinear_points_l75_75381


namespace max_value_of_f_max_value_of_f_at_pi_div_2_g_period_g_center_of_symmetry_sin_2alpha_value_l75_75922

-- Defining vectors and function f(x)
def a (x : ℝ) : ℝ × ℝ := (-Real.cos (2 * x), 2)
def b (x : ℝ) : ℝ × ℝ := (2, 2 - Real.sqrt 3 * Real.sin (2 * x))
def f (x : ℝ) : ℝ := let a_x := a x in let b_x := b x in a_x.1 * b_x.1 + a_x.2 * b_x.2 - 4

-- Problem (I): Maximum value of f(x) over [0, π/2]
theorem max_value_of_f : ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ 2 :=
sorry

-- Point where maximum is attained
theorem max_value_of_f_at_pi_div_2 : f (Real.pi / 2) = 2 :=
sorry

-- Problem (II): Properties of g(x)
def g (x : ℝ) : ℝ := -2 * Real.cos x

-- Smallest positive period of g(x)
theorem g_period : ∀ x, g (x + 2 * Real.pi) = g x :=
sorry

-- Center of symmetry of g(x)
theorem g_center_of_symmetry (k : ℤ) : g (k * Real.pi + Real.pi / 2) = 0 :=
sorry

-- Problem (III): Value of sin(2α) given f(α) = -1 and α in (π/4, π/2)
theorem sin_2alpha_value (α : ℝ) (h1: α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2)) (h2: f α = -1) : 
  Real.sin (2 * α) = (Real.sqrt 15 + Real.sqrt 3) / 8 :=
sorry

end max_value_of_f_max_value_of_f_at_pi_div_2_g_period_g_center_of_symmetry_sin_2alpha_value_l75_75922


namespace probability_approaches_one_third_l75_75161

noncomputable def probability_of_odd (n : ℕ) : ℝ :=
if n = 1 then 1 / 2 else (1 / 4) * probability_of_odd (n - 1) + (1 / 4)

theorem probability_approaches_one_third : 
  tendsto (λ n, probability_of_odd n) at_top (𝓝 (1 / 3)) :=
sorry

end probability_approaches_one_third_l75_75161


namespace average_sales_correct_l75_75116

def jan_sales : ℝ := 150
def feb_sales : ℝ := 120
def mar_sales : ℝ := 80
def apr_sales : ℝ := 100
def may_sales : ℝ := 90
def jun_sales : ℝ := 130
def total_months : ℝ := 6
def total_sales : ℝ := jan_sales + feb_sales + mar_sales + apr_sales + may_sales + jun_sales
def average_sales : ℝ := total_sales / total_months

theorem average_sales_correct : average_sales = 111.67 :=
by
  unfold average_sales total_sales jan_sales feb_sales mar_sales apr_sales may_sales jun_sales total_months
  norm_num
  sorry

end average_sales_correct_l75_75116


namespace swimming_pool_volume_l75_75006

def diameter : ℝ := 20
def depth : ℝ := 5
noncomputable def volume : ℝ := π * (diameter / 2)^2 * depth

theorem swimming_pool_volume : volume = 500 * π :=
by
  sorry

end swimming_pool_volume_l75_75006


namespace randy_biscuits_l75_75562

theorem randy_biscuits (initial_biscuits father_gift mother_gift brother_ate : ℕ) : 
  (initial_biscuits = 32) →
  (father_gift = 13) →
  (mother_gift = 15) →
  (brother_ate = 20) →
  initial_biscuits + father_gift + mother_gift - brother_ate = 40 := by
  sorry

end randy_biscuits_l75_75562


namespace largest_binom_n_l75_75190

theorem largest_binom_n : ∃ (n : ℕ), 
  n ≤ 11 ∧ 
  (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 n ∧ 
  ∀ m (h : m ≤ 11), (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 m → m ≤ n 
  :=
begin
  use 7,
  sorry
end

end largest_binom_n_l75_75190


namespace minimum_species_count_l75_75465

theorem minimum_species_count {n : ℕ} (h_n : n = 2021) 
  (h_cond : ∀ i j k : ℕ, i < j ∧ j < k → 
    birds i = birds k → birds j ≠ birds i → (j - i - 1) % 2 = 1 ∧ (k - j - 1) % 2 = 1) : 
  ∃ s : ℕ, s ≥ 1011 :=
begin
  sorry
end

end minimum_species_count_l75_75465


namespace sum_ЛЯ_ФУ_l75_75259

noncomputable def ЛЯЛЯЛЯ := 10101 * ЛЯ
noncomputable def ФУФУФУ := 10101 * ФУ
noncomputable def ГГЫГЫЫР := 10101 * (ЛЯ + ФУ)

theorem sum_ЛЯ_ФУ (ЛЯ ФУ : ℕ) :
  (∀ (ЛЯ ФУ ГГЫ Ы Р : ℕ),
    ЛЯЛЯЛЯ + ФУФУФУ = ГГЫГЫЫР ∧
    ЛЯ ≠ ФУ ∧ ЛЯ ≠ ГГЫ ∧ ЛЯ ≠ Ы ∧ ЛЯ ≠ Р ∧
    ФУ ≠ ГГЫ ∧ ФУ ≠ Ы ∧ ФУ ≠ Р ∧
    ГГЫ ≠ Ы ∧ ГГЫ ≠ Р ∧ Ы ≠ Р) →
  ЛЯ + ФУ = 109 :=
by
  sorry

end sum_ЛЯ_ФУ_l75_75259


namespace bird_species_min_l75_75486

theorem bird_species_min (total_birds : ℕ) (h_total_birds : total_birds = 2021)
  (h_even_between : ∀ (species : Sort*) (a b : species), (a ≠ b) → even (nat.dist a b)) :
  ∃ species_num : ℕ, species_num = 1011 :=
by
  sorry

end bird_species_min_l75_75486


namespace FH_perp_EG_l75_75297

-- Define the points and their relationship as described in the condition
variable (A B E F C G D H : Type) 
variable [HasRatio A B E F C G D H] -- Assuming a typeclass to handle ratios between points on lines
variable (AE EB AF FB DG GC DH HC AD BC: ℝ)

def conditions (A B E F C G D H : Type)
  [HasRatio A B E F C G D H]
  (AE EB AF FB DG GC DH HC AD BC: ℝ) : Prop :=
  (AE / EB = ED / BC) ∧
  (AF / FB = AD / BC) ∧
  (DG / GC = AD / BC) ∧
  (DH / HC = AD / BC)

-- Create a theorem that corresponds to the problem statement
theorem FH_perp_EG
  (A B E F C G D H : Type)
  [HasRatio A B E F C G D H]
  (AE EB AF FB DG GC DH HC AD BC: ℝ)
  (h1 : conditions A B E F C G D H AE EB AF FB DG GC DH HC AD BC) :
  ∃ FH EG, Perpendicular FH EG := 
sorry

end FH_perp_EG_l75_75297


namespace complex_expression_evaluation_l75_75822

theorem complex_expression_evaluation :
  (complex.mk 0 (-2 * real.sqrt 3 + 1 * complex.i.real) /
      complex.mk 0 (1 + 2 * real.sqrt 3 * complex.i.real)) +
  ((complex.mk real.sqrt 2 0 /. complex.mk 1 (-1)).^2017) =
  complex.mk (real.sqrt 2 / 2) ((real.sqrt 2 / 2) + 1) :=
  sorry

end complex_expression_evaluation_l75_75822


namespace car_distance_proof_l75_75255

variable (D T : ℝ)
variable (h1 : D = 70 * T)
variable (h2 : D = 105 * (T - 0.5))

theorem car_distance_proof (h1 : D = 70 * T) (h2 : D = 105 * (T - 0.5)) : D = 105 := by
  -- Begin the proof here
  sorry

end car_distance_proof_l75_75255


namespace second_mechanic_hours_l75_75616

theorem second_mechanic_hours (x y : ℕ) (h1 : 45 * x + 85 * y = 1100) (h2 : x + y = 20) : y = 5 :=
by
  sorry

end second_mechanic_hours_l75_75616


namespace least_pos_int_for_multiple_of_5_l75_75221

theorem least_pos_int_for_multiple_of_5 (n : ℕ) (h1 : n = 725) : ∃ x : ℕ, x > 0 ∧ (725 + x) % 5 = 0 ∧ x = 5 :=
by
  sorry

end least_pos_int_for_multiple_of_5_l75_75221


namespace false_statement_D_l75_75237

theorem false_statement_D :
  ¬ (∀ {α β : ℝ}, α = β → (true → true → true → α = β ↔ α = β)) :=
by
  sorry

end false_statement_D_l75_75237


namespace number_of_mixed_edge_triangles_l75_75158

open Real

-- Definitions representing each condition from the problem
def vertices : List (ℝ → ℝ) := [λ θ, cos θ, λ θ, cos (2 * θ), λ θ, cos (3 * θ), λ θ, cos (4 * θ), λ θ, cos (5 * θ), λ θ, cos (6 * θ)]

def is_blue_line (f g : ℝ → ℝ) : Prop :=
∃ (P : ℝ → ℝ), ∀ θ, g θ = P (f θ)

def blue_lines : List ((ℝ → ℝ) × (ℝ → ℝ)) :=
[(λ θ, cos θ, λ θ, cos (2 * θ)), (λ θ, cos θ, λ θ, cos (3 * θ)), (λ θ, cos θ, λ θ, cos (4 * θ)), 
 (λ θ, cos θ, λ θ, cos (5 * θ)), (λ θ, cos θ, λ θ, cos (6 * θ)), (λ θ, cos (2 * θ), λ θ, cos (4 * θ)),
 (λ θ, cos (2 * θ), λ θ, cos (6 * θ)), (λ θ, cos (3 * θ), λ θ, cos (6 * θ))]

-- A triangle is determined by 3 vertices
def is_triangle (a b c : ℝ → ℝ) : Prop :=
a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Check at least one blue and one red edge in a triangle
def has_mixed_edges (a b c : ℝ → ℝ) : Prop :=
(is_blue_line a b ∨ is_blue_line a c ∨ is_blue_line b c) ∧ ¬(is_blue_line a b ∧ is_blue_line a c ∧ is_blue_line b c) 

-- The problem statement
theorem number_of_mixed_edge_triangles : 
  ∃! (n : ℕ), (n = 14) :=
sorry

end number_of_mixed_edge_triangles_l75_75158


namespace tangent_line_slope_at_point_l75_75438

theorem tangent_line_slope_at_point (a : ℝ) (f : ℝ → ℝ) (f_deriv : ℝ → ℝ)
  (h₀ : f = λ x, x - a * Real.log x)
  (h₁ : f 1 = 1)
  (h₂ : f_deriv = λ x, 1 - a / x)
  (h₃ : f_deriv 1 = 0) : a = 1 :=
sorry

end tangent_line_slope_at_point_l75_75438


namespace compute_product_l75_75829

theorem compute_product : (100 - 5) * (100 + 5) = 9975 := by
  sorry

end compute_product_l75_75829


namespace value_of_squared_difference_l75_75441

theorem value_of_squared_difference (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) :
  (x - y)^2 = 9 :=
by
  sorry

end value_of_squared_difference_l75_75441


namespace Maria_ate_2_cookies_l75_75539

theorem Maria_ate_2_cookies : 
  ∀ (initial_cookies given_to_friend given_to_family remaining_after_eating : ℕ),
  initial_cookies = 19 →
  given_to_friend = 5 →
  given_to_family = (initial_cookies - given_to_friend) / 2 →
  remaining_after_eating = initial_cookies - given_to_friend - given_to_family - 2 →
  remaining_after_eating = 5 →
  2 = 2 := by
  intros
  sorry

end Maria_ate_2_cookies_l75_75539


namespace angle_conversion_l75_75836

theorem angle_conversion : (1 : ℝ) * (π / 180) * (-225) = - (5 * π / 4) :=
by
  sorry

end angle_conversion_l75_75836


namespace min_groups_divisible_by_six_l75_75138

def group_divisible_by_six : Prop :=
  ∃ (groups : Finset (Finset ℕ)), 
    ∀ g ∈ groups, 
      (∀ x y ∈ g, (x + y) % 6 = 0) ∧ 
      (Finset.bUnion groups id = Finset.range 600) ∧ 
      (∀ g1 g2 ∈ groups, g1 ≠ g2 → g1 ∩ g2 = ∅) ∧ 
      Finset.card groups = 202

theorem min_groups_divisible_by_six : group_divisible_by_six :=
sorry

end min_groups_divisible_by_six_l75_75138


namespace total_cost_of_harrys_order_l75_75924

def cost_of_large_pizza : ℝ := 14
def cost_per_topping : ℝ := 2
def number_of_pizzas : ℕ := 2
def number_of_toppings_per_pizza : ℕ := 3
def tip_percentage : ℝ := 0.25

theorem total_cost_of_harrys_order : 
  let cost_per_pizza := cost_of_large_pizza + cost_of_toppings number_of_toppings_per_pizza * cost_per_topping,
      total_pizza_cost := number_of_pizzas * cost_per_pizza,
      tip := total_pizza_cost * tip_percentage,
      total_cost := total_pizza_cost + tip
  in total_cost = 50 := 
sorry

end total_cost_of_harrys_order_l75_75924


namespace emily_101st_card_is_10_of_Hearts_l75_75325

def number_sequence : List String := ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
def suit_sequence : List String := ["Hearts", "Diamonds", "Clubs", "Spades"]

-- Function to get the number of a specific card
def card_number (n : ℕ) : String :=
  number_sequence.get! (n % number_sequence.length)

-- Function to get the suit of a specific card
def card_suit (n : ℕ) : String :=
  suit_sequence.get! ((n / suit_sequence.length) % suit_sequence.length)

-- Definition to state the question and the answer
def emily_card (n : ℕ) : String := card_number n ++ " of " ++ card_suit n

-- Proving that the 101st card is "10 of Hearts"
theorem emily_101st_card_is_10_of_Hearts : emily_card 100 = "10 of Hearts" :=
by {
  sorry
}

end emily_101st_card_is_10_of_Hearts_l75_75325


namespace feed_correct_parakeets_l75_75298

variable (parakeets1 parakeets2 cups1 cups2 days : ℕ)

-- Initial conditions
def condition1 := (cups1 = 60) ∧ (days = 5)
def condition2 := (cups2 = 30) ∧ (parakeets2 = 10) ∧ (days = 5)

-- The task to prove: 60 cups can feed 20 parakeets for 5 days under given conditions
theorem feed_correct_parakeets
  (h1 : condition1)
  (h2 : condition2)
  : parakeets1 = 20 := by
  sorry

end feed_correct_parakeets_l75_75298


namespace value_of_coupon_l75_75001

theorem value_of_coupon (price_per_bag : ℝ) (oz_per_bag : ℕ) (cost_per_serving_with_coupon : ℝ) (total_servings : ℕ) :
  price_per_bag = 25 → oz_per_bag = 40 → cost_per_serving_with_coupon = 0.50 → total_servings = 40 →
  (price_per_bag - (cost_per_serving_with_coupon * total_servings)) = 5 :=
by 
  intros hpb hob hcpwcs hts
  sorry

end value_of_coupon_l75_75001


namespace tree_edges_count_l75_75314

structure Graph (V E : Type _) :=
(vertices : finset V)
(edges : finset E)
(incident : E → fin (2 : Nat) → V)

variables {V E : Type _} (G : Graph V E) (n : ℕ)

noncomputable def is_connected (G : Graph V E) : Prop :=
  ∀ v1 v2 : V, v1 ∈ G.vertices → v2 ∈ G.vertices → (∃ p : list (E × bool), is_path G v1 v2 p)

def has_no_cycles (G : Graph V E) : Prop :=
  ¬ ∃ v1 v2 : V, v1 ≠ v2 ∧ ∃ (p1 p2 : list (E × bool)), is_path G v1 v2 p1 ∧ is_path G v1 v2 p2 ∧ p1 ≠ p2

def unique_path (G : Graph V E) : Prop :=
  ∀ v1 v2 : V, v1 ≠ v2 → ∃! p : list (E × bool), is_path G v1 v2 p

noncomputable def is_tree (G : Graph V E) : Prop :=
  is_connected G ∧ has_no_cycles G

theorem tree_edges_count (h : is_tree G) (hv : G.vertices.card = n) : G.edges.card = n - 1 :=
sorry

end tree_edges_count_l75_75314


namespace point_with_min_distance_sum_to_60_points_l75_75389

-- Define a circle of radius 1
def circle (r : Real) (c : Real × Real) := { p : Real × Real // (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 }

-- Define the point set on the circle
def point_set_on_circle (P : Fin₆₀ → circle 1 (0, 0)) := True

theorem point_with_min_distance_sum_to_60_points (P : Fin₆₀ → circle 1 (0, 0)) : 
  ∃ p : circle 1 (0, 0), ∑ i, dist P i p ≤ 80 := 
sorry

end point_with_min_distance_sum_to_60_points_l75_75389


namespace correct_conversion_l75_75810

theorem correct_conversion (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (x / y)^(-3 / 4) = 4 * (y / x)^3 := 
sorry

end correct_conversion_l75_75810


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75677

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75677


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75674

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75674


namespace cartons_being_considered_l75_75786

-- Definitions based on conditions
def packs_per_box : ℕ := 10
def boxes_per_carton : ℕ := 12
def price_per_pack : ℕ := 1
def total_cost : ℕ := 1440

-- Calculate total cost per carton
def cost_per_carton : ℕ := boxes_per_carton * packs_per_box * price_per_pack

-- Formulate the main theorem
theorem cartons_being_considered : (total_cost / cost_per_carton) = 12 :=
by
  -- The relevant steps would go here, but we're only providing the statement
  sorry

end cartons_being_considered_l75_75786


namespace largest_integer_binom_eq_l75_75202

theorem largest_integer_binom_eq :
  ∃ n : ℕ, (n = 7) ∧ (binom 11 n = binom 10 3 + binom 10 4) :=
by sorry

end largest_integer_binom_eq_l75_75202


namespace arithmetic_expression_l75_75349

theorem arithmetic_expression :
  4 * 6 * 8 + 24 / 4 - 2^3 = 190 := by
  sorry

end arithmetic_expression_l75_75349


namespace area_ratio_centroids_l75_75996

structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

structure Quadrilateral :=
(A B C D : Point)

def centroid (t : Triangle) : Point :=
{ x := (t.A.x + t.B.x + t.C.x) / 3,
  y := (t.A.y + t.B.y + t.C.y) / 3 }

def area (q : Quadrilateral) : ℝ := 
  abs ((q.A.x * (q.B.y - q.C.y) + q.B.x * (q.C.y - q.A.y) + q.C.x * (q.A.y - q.B.y) +
  q.D.x * (q.A.y - q.D.y)) / 2)

theorem area_ratio_centroids (P Q R S : Point) :
  let PQRS : Quadrilateral := {A := P, B := Q, C := R, D := S}
  let H_P := centroid {A := Q, B := R, C := S}
  let H_Q := centroid {A := R, B := S, C := P}
  let H_R := centroid {A := S, B := P, C := Q}
  let H_S := centroid {A := P, B := Q, C := R}
  let H_PHQHRHS : Quadrilateral := {A := H_P, B := H_Q, C := H_R, D := H_S}
  in area H_PHQHRHS / area PQRS = 1 / 9 :=
sorry

end area_ratio_centroids_l75_75996


namespace largest_binom_n_l75_75189

theorem largest_binom_n : ∃ (n : ℕ), 
  n ≤ 11 ∧ 
  (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 n ∧ 
  ∀ m (h : m ≤ 11), (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 m → m ≤ n 
  :=
begin
  use 7,
  sorry
end

end largest_binom_n_l75_75189


namespace problem_statement_l75_75577

theorem problem_statement (x : ℤ) (y : ℝ) (h : y = 0.5) : 
  (⌈x + y⌉ - ⌊x + y⌋ = 1) ∧ (⌈x + y⌉ - (x + y) = 0.5) := 
by 
  sorry

end problem_statement_l75_75577


namespace range_of_a_l75_75876

section
  variable {x a : ℝ}

  -- Define set A
  def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }

  -- Define set B
  def setB (a : ℝ) : Set ℝ := 
    { x | (2*x + a ≤ 0) ∧ (x^2 - 2*(a + 7)*x + 5 ≤ 0)}

  -- The proof problem statement
  theorem range_of_a (a : ℝ) : 
    (setA ⊆ setB a) ↔ (-4 ≤ a ∧ a ≤ -2) :=
  sorry
end

end range_of_a_l75_75876


namespace min_sum_of_factors_l75_75129

theorem min_sum_of_factors (a b c d : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h : a * b * c * d = nat.factorial 8) : 
  a + b + c + d ≥ 61 := 
sorry

end min_sum_of_factors_l75_75129


namespace min_species_needed_l75_75472

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l75_75472


namespace angle_B_pi_div_3_l75_75936

theorem angle_B_pi_div_3
  {A B C : ℝ} {a b c : ℝ}
  (h_sin_ratio : sin A / sin B = 5 / 7 ∧ sin B / sin C = 7 / 8) 
  (hABC : A + B + C = π) 
  (h_sides : a = 5 * sin A ∧ b = 7 * sin B ∧ c = 8 * sin C):
  B = π / 3 := 
sorry

end angle_B_pi_div_3_l75_75936


namespace largest_binom_n_l75_75193

theorem largest_binom_n : ∃ (n : ℕ), 
  n ≤ 11 ∧ 
  (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 n ∧ 
  ∀ m (h : m ≤ 11), (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 m → m ≤ n 
  :=
begin
  use 7,
  sorry
end

end largest_binom_n_l75_75193


namespace largest_prime_divisor_of_expression_l75_75307

theorem largest_prime_divisor_of_expression :
  ∃ p, (prime p) ∧ (p ∣ (12! + 13! + 17)) ∧ 
       (∀ q, (prime q) ∧ (q ∣ (12! + 13! + 17)) → q ≤ p) ∧ p = 17 :=
by
  sorry

end largest_prime_divisor_of_expression_l75_75307


namespace find_a_b_find_tangent_line_l75_75536

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := 2 * x ^ 3 + 3 * a * x ^ 2 + 3 * b * x + 8

-- Define the derivative of the function f(x)
def f' (a b x : ℝ) : ℝ := 6 * x ^ 2 + 6 * a * x + 3 * b

-- Define the conditions for extreme values at x=1 and x=2
def extreme_conditions (a b : ℝ) : Prop :=
  f' a b 1 = 0 ∧ f' a b 2 = 0

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) (h : extreme_conditions a b) : a = -3 ∧ b = 4 :=
by sorry

-- Find the equation of the tangent line at x=0
def tangent_equation (a b : ℝ) (x y : ℝ) : Prop :=
  12 * x - y + 8 = 0

-- Prove the equation of the tangent line
theorem find_tangent_line (a b : ℝ) (h : extreme_conditions a b) : tangent_equation a b 0 8 :=
by sorry

end find_a_b_find_tangent_line_l75_75536


namespace hypotenuse_length_l75_75146

theorem hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = Real.sqrt 5) (h₂ : b = Real.sqrt 12) : c = Real.sqrt 17 :=
by
  -- Proof not required, hence skipped with 'sorry'
  sorry

end hypotenuse_length_l75_75146


namespace votes_for_winner_l75_75164

-- Definitions of the conditions
def total_votes (winner_votes loser_votes : ℕ) : ℕ := winner_votes + loser_votes
def winner_percentage : ℝ := 0.62
def loser_percentage : ℝ := 0.38
def victory_margin : ℕ := 360

theorem votes_for_winner (V : ℕ) (H1 : V = (360 : ℕ) / 0.24) : (winner_percentage * V : ℝ) = 930 :=
by
  sorry

end votes_for_winner_l75_75164


namespace lg_property_l75_75439

noncomputable def seq (x : ℕ → ℝ) := ∀ n : ℕ, x (n + 1) = 10 * x n

theorem lg_property (x : ℕ → ℝ) (h1 : seq x) (h2 : ∑ i in finset.range 100, x (i + 1) = 100) :
  Real.log10 (∑ i in finset.range 100, x (i + 101)) = 102 :=
  sorry

end lg_property_l75_75439


namespace inequality_solution_subset_l75_75390

theorem inequality_solution_subset (x : ℝ) : (1 < x) → (x < 3) → x = 2 :=
by
  intros h1 h2
  rational
  have h : 1 < 2 ↔ 2 < 3 := sorry -- intermediate steps skipped
  sorry

end inequality_solution_subset_l75_75390


namespace limbs_polished_per_girl_is_one_l75_75261

def PolishedLimbsPerGirl
  (num_girls : ℕ) 
  (fingers_per_limb : ℕ)
  (total_fingers_polished : ℕ) : ℕ :=
  show ℕ, from total_fingers_polished / (num_girls * fingers_per_limb)

theorem limbs_polished_per_girl_is_one
  (num_girls : ℕ)
  (fingers_per_limb : ℕ)
  (total_fingers_polished : ℕ)
  (hnum_girls : num_girls = 8)
  (hfingers_per_limb : fingers_per_limb = 5)
  (htotal_fingers_polished : total_fingers_polished = 40) :
  PolishedLimbsPerGirl num_girls fingers_per_limb total_fingers_polished = 1 := by
  rw [PolishedLimbsPerGirl, hnum_girls, hfingers_per_limb, htotal_fingers_polished]
  sorry

end limbs_polished_per_girl_is_one_l75_75261


namespace trenton_commission_rate_l75_75167

noncomputable def commission_rate (fixed_earnings : ℕ) (goal : ℕ) (sales : ℕ) : ℚ :=
  ((goal - fixed_earnings : ℤ) / (sales : ℤ)) * 100

theorem trenton_commission_rate :
  commission_rate 190 500 7750 = 4 := 
  by
  sorry

end trenton_commission_rate_l75_75167


namespace primes_partition_golden_l75_75171

-- Definitions related to the problem
def bin_rep (n: ℕ) : ℕ → bool := sorry

def circ (m n : ℕ) : Prop :=
  bin_rep m n = true ∨ bin_rep n m = true

def bullet (m n : ℕ) : Prop :=
  ¬ circ m n

def golden (A : Set ℕ) : Prop :=
  ∀ (U V : Set ℕ), U ⊆ A → V ⊆ A → U.Finite → V.Finite → U.Nonempty → V.Nonempty → U ∩ V = ∅ →
    ∃ z ∈ A, (∀ x ∈ U, circ z x) ∧ (∀ y ∈ V, bullet z y)

-- The main theorem
theorem primes_partition_golden (P : ℕ → Prop) [prime_numbers : ∀ n, P n ↔ Prime n] 
  (P_set : Set ℕ) (partition : Finset (Set ℕ)) (hP : ∀ p, P p → p ∈ P_set)
  (h_partition : Finset.sup partition id = P_set)
  (h_disjoint : ∀ s t ∈ partition, s ≠ t → s ∩ t = ∅) :
  ∃ (S ∈ partition), golden S :=
sorry

end primes_partition_golden_l75_75171


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75733

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75733


namespace add_base_12_l75_75284

theorem add_base_12 :
  let a := 5*12^2 + 1*12^1 + 8*12^0
  let b := 2*12^2 + 7*12^1 + 6*12^0
  let result := 7*12^2 + 9*12^1 + 2*12^0
  a + b = result :=
by
  -- Placeholder for the actual proof
  sorry

end add_base_12_l75_75284


namespace Tristan_study_l75_75031

noncomputable def hours_studied (monday tuesday wednesday thursday friday target total : ℝ) : Prop :=
  thursday = tuesday / 2 ∧ 
  friday = 2 * monday ∧ 
  total = monday + tuesday + wednesday + thursday + friday ∧ 
  target = 41.5 ∧ 
  total = 25.5

theorem Tristan_study (h : hours_studied 4 5 6 2.5 8 41.5 25.5) :
  let remaining_hours := 41.5 - 25.5,
  let saturday := remaining_hours / 2
  in saturday = 8 :=
by
  sorry

end Tristan_study_l75_75031


namespace rectangles_containment_l75_75391

theorem rectangles_containment :
  (∀ (rects: ℕ → ℕ × ℕ), ∃ (i j : ℕ), i ≠ j ∧ 
  (rects i).1 ≤ (rects j).1 ∧ (rects i).2 ≤ (rects j).2) 
  ∨ 
  (rects j).1 ≤ (rects i).1 ∧ (rects j).2 ≤ (rects i).2 :=
sorry

end rectangles_containment_l75_75391


namespace overall_percentage_gain_l75_75802

def percentageChange (original final : ℝ) : ℝ :=
  ((final - original) / original) * 100

theorem overall_percentage_gain :
  let original_price := 100
  let raised_price := original_price + (original_price * 36 / 100)
  let discount_price1 := raised_price - (raised_price * 10 / 100)
  let final_price := discount_price1 - (discount_price1 * 15 / 100)
  percentageChange original_price final_price = 4.04 :=
by
  let original_price := 100
  let raised_price := original_price + (original_price * 36 / 100)
  let discount_price1 := raised_price - (raised_price * 10 / 100)
  let final_price := discount_price1 - (discount_price1 * 15 / 100)
  have h1 : percentageChange original_price final_price = 4.04, from sorry
  exact h1

end overall_percentage_gain_l75_75802


namespace sum_of_integers_whose_squares_end_with_25_l75_75709

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75709


namespace sum_of_two_digit_numbers_ending_in_25_l75_75667

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75667


namespace choose_4_from_15_l75_75947

theorem choose_4_from_15 : Nat.choose 15 4 = 1365 := by
  sorry

end choose_4_from_15_l75_75947


namespace digit_507_in_decimal_expansion_of_15_37_eq_5_l75_75174

theorem digit_507_in_decimal_expansion_of_15_37_eq_5 :
  (∃ s : String, s = "405" ∧ ∀ n : ℕ, (n % 3 = 0 → (s[(n/3 % s.length) % s.length] = '5'))) :=
by
  sorry

end digit_507_in_decimal_expansion_of_15_37_eq_5_l75_75174


namespace sum_of_two_digit_numbers_ending_in_25_l75_75663

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75663


namespace coefficients_sum_l75_75011

theorem coefficients_sum :
  let A := 3
  let B := 14
  let C := 18
  let D := 19
  let E := 30
  A + B + C + D + E = 84 := by
  sorry

end coefficients_sum_l75_75011


namespace dual_expressions_reciprocal_rationalize_expression_solve_sqrt_equation_l75_75319

-- Problem (1)
theorem dual_expressions_reciprocal : 
  let a := 2
  let b := 3
  let e1 := a + Real.sqrt b
  let e2 := a - Real.sqrt b
  e1 * e2 = 1 -> (e1 * e2 = 1) := 
by sorry

-- Problem (2)
theorem rationalize_expression :
  let x := 1 / (Real.sqrt 5 - 2)
  let y := 1 / (Real.sqrt 5 + 2)
  (x - y) / (x^2 * y + x * y^2) = (2 * Real.sqrt 5) / 5 :=
by sorry

-- Problem (3)
theorem solve_sqrt_equation {x t : ℝ} (h : sqrt (24 - x) - sqrt (8 - x) = 2) :
  let t := sqrt (24 - x) + sqrt (8 - x)
  t = 8 -> x = -1 :=
by sorry

end dual_expressions_reciprocal_rationalize_expression_solve_sqrt_equation_l75_75319


namespace evaluate_expression_l75_75338

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l75_75338


namespace minimum_species_count_l75_75463

theorem minimum_species_count {n : ℕ} (h_n : n = 2021) 
  (h_cond : ∀ i j k : ℕ, i < j ∧ j < k → 
    birds i = birds k → birds j ≠ birds i → (j - i - 1) % 2 = 1 ∧ (k - j - 1) % 2 = 1) : 
  ∃ s : ℕ, s ≥ 1011 :=
begin
  sorry
end

end minimum_species_count_l75_75463


namespace box_volume_possible_l75_75801

theorem box_volume_possible (x : ℕ) (hx : 30 * x^3 = 240) : ∃ x : ℕ, 2*x ≠ 0 ∧ 3*x ≠ 0 ∧ 5*x ≠ 0 ∧ 30 * x^3 = 240 :=
begin
  use 2,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { exact hx }
end

end box_volume_possible_l75_75801


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75697

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75697


namespace sum_of_two_digit_numbers_ending_in_25_l75_75629

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75629


namespace range_of_a_l75_75918

open Classical

variable (a : ℝ)

def p := a ≤ -2 ∨ a ≥ 2
def q := a ≥ -10

theorem range_of_a (h₁ : p a ∨ q a) (h₂ : ¬ (p a ∧ q a)) : a ∈ set.Ioo (-∞) (-10) ∪ set.Ioo (-2) 2 :=
by
  sorry

end range_of_a_l75_75918


namespace randy_biscuits_l75_75555

theorem randy_biscuits : 
  let initial := 32 
  let father_gift := 13 
  let mother_gift := 15 
  let brother_ate := 20 
  let total := initial + father_gift + mother_gift 
  total - brother_ate = 40 := 
by
  let initial := 32 
  let father_gift := 13 
  let mother_gift := 15 
  let brother_ate := 20 
  let total := initial + father_gift + mother_gift 
  show total - brother_ate = 40 from sorry

end randy_biscuits_l75_75555


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75698

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75698


namespace minimum_bird_species_l75_75460

theorem minimum_bird_species (total_birds : ℕ) (h : total_birds = 2021) :
  ∃ (min_species : ℕ), min_species = 1011 ∧ 
  (∀ (species_array : array total_birds ℕ),
   ∀ i j : fin total_birds, 
   species_array[i] = species_array[j] → ((i ≠ j) →
   (abs (i - j) mod 2 = 0))) :=
sorry

end minimum_bird_species_l75_75460


namespace people_pay_taxes_every_week_l75_75251

theorem people_pay_taxes_every_week (num_people_daily : ℕ) (tax_exempt_percentage : ℝ) (days_in_week : ℕ) :
  tax_exempt_percentage = 0.06 -> num_people_daily = 1000 -> days_in_week = 7 ->
  let paying_customers_daily := (1 - tax_exempt_percentage) * num_people_daily
  let paying_customers_weekly := paying_customers_daily * days_in_week
  paying_customers_weekly = 6580 :=
by
  intros h_exempt_percent h_daily h_days
  rw [h_exempt_percent, h_daily, h_days]
  let paying_customers_daily := (1 - 0.06) * 1000
  let paying_customers_weekly := paying_customers_daily * 7
  have : paying_customers_daily = 940 := by linarith
  rw [this]
  have : paying_customers_weekly = 6580 := by linarith
  exact this

end people_pay_taxes_every_week_l75_75251


namespace quadratic_cost_function_maximum_profit_at_23_minimum_average_cost_per_ton_production_at_20_lowest_average_cost_l75_75809

noncomputable def cost_function (x : ℝ) : ℝ :=
  (1 / 10) * (x - 15)^2 + 17.5

def profit_function (x : ℝ) : ℝ :=
  1.6 * x - cost_function x

def average_cost_per_ton (x : ℝ) : ℝ :=
  cost_function x / x

theorem quadratic_cost_function :
    ∀ x ∈ set.Icc 10 25, cost_function x = (1 / 10) * (x - 15)^2 + 17.5 :=
begin
  sorry,
end

theorem maximum_profit_at_23 :
    23 ∈ set.Icc 10 25 ∧ ∀ x ∈ set.Icc 10 25, profit_function x ≤ profit_function 23 :=
begin
  sorry,
end

theorem minimum_average_cost_per_ton :
    ∀ x ∈ set.Icc 10 25, average_cost_per_ton x ≥ 1 :=
begin
  sorry,
end

theorem production_at_20_lowest_average_cost :
  20 ∈ set.Icc 10 25 ∧ average_cost_per_ton 20 = 1 :=
begin
  sorry,
end

end quadratic_cost_function_maximum_profit_at_23_minimum_average_cost_per_ton_production_at_20_lowest_average_cost_l75_75809


namespace find_common_difference_and_sum_l75_75498

variable (a : ℕ → ℝ)

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) = a n + d

def forms_geometric_sequence (a1 a2 a5 : ℝ) : Prop :=
a2 ^ 2 = a1 * a5 ∧ a1 ≠ a2 ∧ a2 ≠ a5

-- Problem statement
theorem find_common_difference_and_sum (d : ℝ) (a1 a2 a5 : ℝ) (n : ℕ) (a : ℕ → ℝ) :
  a1 = 1 →
  forms_geometric_sequence a1 a2 a5 →
  is_arithmetic_sequence a d →
  a 1 = a1 →
  a 2 = a2 →
  a 5 = a5 →
  d = 2 ∧ (∑ i in Finset.range n, 1 / (a i * a (i + 1))) = n / (2 * n + 1) :=
by
  intros h_a1 h_geom h_arith h_a1_ h_a2 h_a5
  sorry

end find_common_difference_and_sum_l75_75498


namespace rectangle_area_l75_75933

theorem rectangle_area
  (L B : ℕ)
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) : L * B = 2030 :=
sorry

end rectangle_area_l75_75933


namespace least_integer_to_add_l75_75215

theorem least_integer_to_add (n : ℕ) (h : n = 725) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 :=
by
  use 5
  split
  · exact Nat.lt_succ_self 4
  · rw [h]
    norm_num
    sorry

end least_integer_to_add_l75_75215


namespace totalGames_l75_75044

-- Define Jerry's original number of video games
def originalGames : ℕ := 7

-- Define the number of video games Jerry received for his birthday
def birthdayGames : ℕ := 2

-- Statement: Prove that the total number of games Jerry has now is 9
theorem totalGames : originalGames + birthdayGames = 9 := by
  sorry

end totalGames_l75_75044


namespace unique_difference_count_l75_75007

open Finset

noncomputable def count_unique_differences (S : Finset ℕ) : ℕ :=
  (S.product S).filter (λ p, (p.1 ≠ p.2) ∧ (p.1 > p.2)).image (λ p, p.1 - p.2) |> toFinset |> card

theorem unique_difference_count : 
  let S := ({2, 3, 5, 7, 11, 13} : Finset ℕ) in
  count_unique_differences S = 10 :=
by
  let S := ({2, 3, 5, 7, 11, 13} : Finset ℕ)
  have h : count_unique_differences S = 10 := sorry
  exact h

end unique_difference_count_l75_75007


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75678

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75678


namespace circle_diameter_l75_75257

theorem circle_diameter (A : ℝ) (h : A = Real.pi) : ∃ d : ℝ, d = 2 := by
  let r : ℝ := Real.sqrt 1
  have hr : r = 1 := Real.sqrt_eq_iff_root_eq.mpr ⟨le_rfl, (one_mul 1).symm⟩
  let d := 2 * r
  use d
  rw hr
  norm_num

end circle_diameter_l75_75257


namespace sum_of_squares_ending_in_25_l75_75725

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75725


namespace P_correct_l75_75067

-- Let a, b, c be the roots of the cubic x^3 + 4x^2 + 6x + 9 = 0
noncomputable def roots_of_cubic : List ℂ := -- We use complex numbers since roots might be complex
[(Root ɪ₄ (Polynomial.Cubic X^3 + 4*X^2 + 6*X + 9))]

-- Define the polynomial P(x) such that P(a) = b + c, P(b) = a + c, P(c) = a + b and P(a + b + c) = -20
def P (x : ℂ) := 20*x^3 + 79*x^2 + 114*x + 176

-- Define the statement to be proved
theorem P_correct (a b c : ℂ) (h_roots : (Polynomial.Cubic X^3 + 4*X^2 + 6*X + 9).is_root a ∧ 
                                        (Polynomial.Cubic X^3 + 4*X^2 + 6*X + 9).is_root b ∧ 
                                        (Polynomial.Cubic X^3 + 4*X^2 + 6*X + 9).is_root c):
  P(a) = b + c ∧ P(b) = a + c ∧ P(c) = a + b ∧ P(a + b + c) = -20 := 
by
  sorry

end P_correct_l75_75067


namespace exponential_comparison_l75_75369

theorem exponential_comparison (a b : ℝ) (h1 : 1 < a) (h2 : a < b) :
  let m := a^(b-1)
  let n := b^(a-1)
  in m > n :=
by
  sorry

end exponential_comparison_l75_75369


namespace arithmetic_sequence_10th_term_log_l75_75134

theorem arithmetic_sequence_10th_term_log :
  ∀ (a b : ℕ) (B: ℕ), log (a^4 * b^6) + 3 * 8 * B = log (b^(82 * B)) :=
by { sorry }

end arithmetic_sequence_10th_term_log_l75_75134


namespace sum_of_squares_ending_in_25_l75_75726

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75726


namespace train_pass_time_l75_75981

theorem train_pass_time (L : ℝ) (S : ℝ) (hL : L = 50) (hS : S = 10) : 
  (L / S) = 5 :=
by
  rw [hL, hS]
  norm_num
  sorry

end train_pass_time_l75_75981


namespace product_of_two_is_perfect_square_l75_75777

theorem product_of_two_is_perfect_square :
  ∀ (numbers : Fin 17 → ℕ), 
    (∀ i, ∀ p > 7, ¬ p.Prime ∨ ¬ p ∣ numbers i) → 
    ∃ i j, i ≠ j ∧ ∃ k, numbers i * numbers j = k^2 := 
begin
  sorry
end

end product_of_two_is_perfect_square_l75_75777


namespace Jennifer_money_left_l75_75764

def money : ℝ := 180

def sandwich_cost : ℝ := (1/5) * money

def museum_ticket_cost : ℝ := (1/6) * money

def book_cost : ℝ := (1/2) * money

def total_spent : ℝ := sandwich_cost + museum_ticket_cost + book_cost

def money_left (money total_spent : ℝ) : ℝ := money - total_spent

theorem Jennifer_money_left : money_left money total_spent = 24 := by
  sorry

end Jennifer_money_left_l75_75764


namespace B_four_v_equals_l75_75059

noncomputable def matrix : Type := Matrix (Fin 2) (Fin 2) ℝ
noncomputable def vector : Type := Fin 2 → ℝ

def B : matrix := sorry

def v : vector := λ i, if i = 0 then 7 else -3

theorem B_four_v_equals :
  (∀ (B : matrix),
  B.mulVec v = (λ i, if i = 0 then 21 else -9)) → 
  (Matrix.pow B 4).mulVec v = 
  (λ i, if i = 0 then 567 else -243) := by
  intro h
  apply sorry

end B_four_v_equals_l75_75059


namespace probability_of_digit_2_in_3_div_11_eq_one_half_l75_75091

theorem probability_of_digit_2_in_3_div_11_eq_one_half :
  let decimal_rep := "0.272727..."
  let repeating_block := "27"
  let block_length := String.length repeating_block
  let num_of_2s := String.count repeating_block '2'
  ∃ n, decimal_rep = "0." ++ String.repeat repeating_block n ∧
  (num_of_2s / block_length : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_digit_2_in_3_div_11_eq_one_half_l75_75091


namespace count_integers_within_range_l75_75008

theorem count_integers_within_range : 
  ∃ (count : ℕ), count = 57 ∧ ∀ n : ℤ, -5.5 * Real.pi ≤ n ∧ n ≤ 12.5 * Real.pi → n ≥ -17 ∧ n ≤ 39 :=
by
  sorry

end count_integers_within_range_l75_75008


namespace choose_15_4_l75_75956

/-- The number of ways to choose 4 captains from a team of 15 people is 1365. -/
theorem choose_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end choose_15_4_l75_75956


namespace projection_matrix_determinant_zero_l75_75523

open Matrix

theorem projection_matrix_determinant_zero :
  let a := 3
  let b := -5
  let Q := Scalar 1 / (a^2 + b^2) • !![![(a * a), (a * b)], ![(a * b), (b * b)]]
  det Q = 0 :=
by
  let a := 3
  let b := -5
  let Q := Scalar 1 / (a^2 + b^2) • !![![(a * a), (a * b)], ![(a * b), (b * b)]]
  show det Q = 0
  sorry

end projection_matrix_determinant_zero_l75_75523


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75659

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75659


namespace max_true_statements_at_most_three_l75_75530

open Real

theorem max_true_statements_at_most_three (x : ℝ) : 
  let S1 := (0 < x^3) ∧ (x^3 < 1)
  let S2 := (x^2 > 1)
  let S3 := (-1 < x) ∧ (x < 0)
  let S4 := (0 < x) ∧ (x < 1)
  let S5 := (0 < x - x^3) ∧ (x - x^3 < 1)
  ∃ n ≤ 3, (S1 → n = n + 1) ∧ (S2 → n = n + 1) ∧ (S3 → n = n + 1) ∧ (S4 → n = n + 1) ∧ (S5 → n = n + 1) := sorry

end max_true_statements_at_most_three_l75_75530


namespace math_problem_proof_l75_75374

noncomputable def math_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 4 / a + 2 / b) : Prop :=
  (a + 2 * b >= 4) ∧
  (let log2_a := Real.log a / Real.log 2;
       log2_b := Real.log b / Real.log 2 in
   (a - 1) * (b - 1) > 0 → 
   (1 / log2_a + 3 / log2_b) = 4 + 2 * Real.sqrt 3)

theorem math_problem_proof (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a + 2 * b = 4 / a + 2 / b)
  (h4 : (a - 1) * (b - 1) > 0)
  : math_problem a b h1 h2 h3 :=
by 
  split
  { sorry }
  { sorry }

end math_problem_proof_l75_75374


namespace equilateral_triangle_condition_l75_75023

theorem equilateral_triangle_condition (A B C : Type) [triangle A B C] (h1 : angle A = 60°) :
  is_equilateral_triangle A B C ↔ AB = AC := 
by
  sorry

end equilateral_triangle_condition_l75_75023


namespace unique_zero_of_f_l75_75133

noncomputable def f (x : ℝ) : ℝ := 2^x + x^3 - 2

theorem unique_zero_of_f :
  (StrictMono f) ∧ (f 0 < 0) ∧ (f 1 > 0) → ∃! x : ℝ, f x = 0 :=
by {
  intro h,
  sorry
}

end unique_zero_of_f_l75_75133


namespace sum_of_integers_whose_squares_end_with_25_l75_75705

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75705


namespace monthly_vs_annual_compounding_difference_l75_75540

theorem monthly_vs_annual_compounding_difference :
  let P := 8000
  let r := 0.10
  let n := 5
  let m := 12
  let annual_amount := P * (1 + r) ^ n
  let monthly_amount := P * (1 + r / m) ^ (m * n)
  let difference := (monthly_amount - annual_amount)
  abs (difference - 292) < 0.01 :=
by {
  sorry
}

end monthly_vs_annual_compounding_difference_l75_75540


namespace sum_of_two_digit_integers_with_square_ending_in_25_l75_75653

theorem sum_of_two_digit_integers_with_square_ending_in_25 :
  ∑ n in Finset.filter (λ n => (n % 100) ^ 2 % 100 == 25) (Finset.Icc 10 99), n = 495 := by
  sorry

end sum_of_two_digit_integers_with_square_ending_in_25_l75_75653


namespace equation_of_line_intersecting_ellipse_l75_75399

theorem equation_of_line_intersecting_ellipse (x1 x2 y1 y2 : ℝ)
  (h1 : x1 + x2 = 4)
  (h2 : y1 + y2 = 2)
  (h3 : (x1^2) / 16 + (y1^2) / 12 = 1)
  (h4 : (x2^2) / 16 + (y2^2) / 12 = 1) :
  ∃ l : ℝ → ℝ → Prop, l = (λ x y, 3 * x + 2 * y - 8 = 0) :=
by 
  sorry

end equation_of_line_intersecting_ellipse_l75_75399


namespace exists_set_of_2011_integers_l75_75841

theorem exists_set_of_2011_integers : ∃ S : set ℕ, S.card = 2011 ∧ (∀ m n ∈ S, m ≠ n → |m - n| = Nat.gcd m n) :=
by sorry

end exists_set_of_2011_integers_l75_75841


namespace choose_4_captains_from_15_l75_75966

def num_ways_to_choose_captains : ℕ := 15.choose 4

theorem choose_4_captains_from_15 : num_ways_to_choose_captains = 1365 := by
  sorry

end choose_4_captains_from_15_l75_75966


namespace ceil_minus_y_eq_one_minus_frac_l75_75064

theorem ceil_minus_y_eq_one_minus_frac (x : ℝ) (y : ℝ) (h1 : y = 2 * x) (h2 : ⌈y⌉ - ⌊y⌋ = 1) :
  let f := y - ⌊y⌋ in ⌈y⌉ - y = 1 - f :=
sorry

end ceil_minus_y_eq_one_minus_frac_l75_75064


namespace all_hobbits_if_nine_inhabitants_five_goblins_and_five_elves_if_ten_inhabitants_l75_75444

-- Definitions of tribes
inductive Tribe
| elf
| goblin
| hobbit

-- Definition for the round table setup
structure RoundTable where
  inhabitants : List Tribe
  repeatPhrase : (inhabitants : List Tribe) → Prop

-- Conditions
def areElvesAlwaysTruthTellers : Prop := ∀ p : Tribe, p = Tribe.elf → true
def areGoblinsAlwaysLiars : Prop := ∀ p : Tribe, p = Tribe.goblin → false
def areHobbitsAlternating : Prop := ∀ (p q : Tribe), p = q → (p = Tribe.hobbit ↔ ¬(q = Tribe.hobbit))

-- Lean statement for the problem with 9 inhabitants
theorem all_hobbits_if_nine_inhabitants (inhabitants : List Tribe)
  (h1 : inhabitants.length = 9)
  (h2 : RoundTable.repeatPhrase inhabitants)
  (h3 : areElvesAlwaysTruthTellers)
  (h4 : areGoblinsAlwaysLiars)
  (h5 : areHobbitsAlternating) :
  ∀ p ∈ inhabitants, p = Tribe.hobbit :=
by
  sorry

-- Lean statement for the problem with 10 inhabitants
theorem five_goblins_and_five_elves_if_ten_inhabitants (inhabitants : List Tribe)
  (h1 : inhabitants.length = 10)
  (h2 : RoundTable.repeatPhrase inhabitants)
  (h3 : areElvesAlwaysTruthTellers)
  (h4 : areGoblinsAlwaysLiars)
  (h5 : areHobbitsAlternating) :
  (inhabitants.count Tribe.goblin = 5) ∧ (inhabitants.count Tribe.elf = 5) :=
by
  sorry

end all_hobbits_if_nine_inhabitants_five_goblins_and_five_elves_if_ten_inhabitants_l75_75444


namespace ellipse_focal_length_l75_75411

theorem ellipse_focal_length :
  (let a := Real.sqrt 32; let b := Real.sqrt 23 in
   let c := Real.sqrt (a^2 - b^2) in
   2 * c = 6) :=
by
  let a := Real.sqrt 32
  let b := Real.sqrt 23
  let c := Real.sqrt (a^2 - b^2)
  have h : c = 3, sorry
  show 2 * c = 6, by rw [h]; norm_num

end ellipse_focal_length_l75_75411


namespace limit_S_n_l75_75417

noncomputable def a_n (n : ℕ+) : ℝ :=
  if n = 1 ∨ n = 2 then 1 / n else (1 / 2) ^ n

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n.succ, a_n ⟨i, Nat.succ_pos' i⟩

theorem limit_S_n : tendsto (λ n : ℕ, S_n n) at_top (𝓝 (7 / 4)) :=
  by
    sorry

end limit_S_n_l75_75417


namespace sequence_problem_l75_75382

open Nat

theorem sequence_problem 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, S n = 2 * a n - 1)
  (h2 : ∀ n, b 1 = a 1 ∧ b 4 = a 3) 
  (h3 : ∀ n, a n = 2^(n - 1))
  (h4 : ∀ n, b n = n)
  (h5 : ∀ n, c n = 2 / (a n) - 1 / (b n * b (n + 1))) :
  ∀ n, T n = 4 * (1 - 1 / 2^n) - n / (n + 1) :=
sorry

end sequence_problem_l75_75382


namespace number_of_ways_to_choose_4_captains_from_15_l75_75952

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end number_of_ways_to_choose_4_captains_from_15_l75_75952


namespace max_min_le_sqrt2_l75_75070

theorem max_min_le_sqrt2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    max (min x (min (y + 1/x) (1/y))) ≤ Real.sqrt 2 :=
sorry

end max_min_le_sqrt2_l75_75070


namespace PQRS_cyclic_iff_X_on_altitude_l75_75993

-- Definitions of points and geometric configurations

variables {α : Type*} [EuclideanSpace α]
variables {A B C X Y P Q R S : α}

-- Assumptions for the problem setup
variables (h_triangle : ∃ (A B C : α), Triangle A B C)
variables (h_X_in_triangle : ∀ (A B C X : α), InsideTriangle A B C X)
variables (h_AXY : LineThrough A X Y ∧ IntersectsAt Y (LineSegment BC))
variables (h_perpendiculars :
  Perpendicular Y P (LineSegment CA) ∧
  Perpendicular Y Q (LineSegment CX) ∧
  Perpendicular Y R (LineSegment BX) ∧
  Perpendicular Y S (LineSegment BA))

-- The necessary and sufficient condition we need to prove
theorem PQRS_cyclic_iff_X_on_altitude : 
  CyclicQuadrilateral P Q R S ↔ LiesOnAltitude X A B C :=
sorry

end PQRS_cyclic_iff_X_on_altitude_l75_75993


namespace fraction_is_expected_l75_75066

open Finset

def set_of_subsets (n : ℕ) := powerset (range n)

def is_good_subset (S : Finset (Finset ℕ)) : Prop :=
  ∀ (A B : Finset ℕ), A ∈ S → B ∈ S → ((A \ B) ∪ (B \ A)) ∈ S

def count_good_subsets (n low high : ℕ) : ℚ := 
  let good_subsets := { S : Finset (Finset ℕ) | is_good_subset S ∧ S.card ≥ low ∧ S.card ≤ high } in
  (good_subsets.card : ℚ) / (2 ^ (2 ^ n) : ℚ)

noncomputable def fraction_good_subsets_2015_3015 : ℚ := 
  count_good_subsets 23 2015 3015

theorem fraction_is_expected :
  fraction_good_subsets_2015_3015  = (18839183877670041942218307147122500601235 : ℚ) / (4769168484048619242209570178451249271212 : ℚ) :=
sorry

end fraction_is_expected_l75_75066


namespace sum_of_ceiling_sqrt_l75_75846

theorem sum_of_ceiling_sqrt :
  (∑ n in finset.range (40 - 10 + 1), ⌈real.sqrt (n + 10)⌉) = 170 := by
  sorry

end sum_of_ceiling_sqrt_l75_75846


namespace fraction_of_ponies_with_horseshoes_l75_75272

variable (P H : ℕ)
variable (F : ℚ)

theorem fraction_of_ponies_with_horseshoes 
  (h1 : H = P + 3)
  (h2 : P + H = 163)
  (h3 : (5/8 : ℚ) * F * P = 5) :
  F = 1/10 :=
  sorry

end fraction_of_ponies_with_horseshoes_l75_75272


namespace eval_expr_ceil_floor_l75_75336

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l75_75336


namespace largest_binom_coeff_l75_75186

theorem largest_binom_coeff (n : ℕ) :
  (nat.choose 10 3) + (nat.choose 10 4) = nat.choose 11 n → n = 7 :=
by sorry

end largest_binom_coeff_l75_75186


namespace minimum_species_l75_75485

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l75_75485


namespace real_solutions_count_eq_l75_75320

theorem real_solutions_count_eq :
  let eq := ∀ x : ℝ, 3 * x^2 - 40 * (floor (x^2)) + 100 = 0 in
  let solutions := { x : ℝ | eq x } in
  solutions.finite ∧ solutions.to_finset.card = /* computed value here */ :=
by
  sorry

end real_solutions_count_eq_l75_75320


namespace project_completion_time_l75_75508

theorem project_completion_time :
  let team_a_rate := 1 / 40
  let team_b_rate := 1 / 60
  let initial_days := 20
  let initial_work := initial_days * team_a_rate
  let total_work := 1
  let remaining_work := total_work - initial_work
  let combined_rate := team_a_rate + team_b_rate
  let days_together := remaining_work / combined_rate
  in days_together = 12 := by
  sorry

end project_completion_time_l75_75508


namespace dilation_transform_l75_75121

theorem dilation_transform (z c : ℂ) (k : ℝ) (h₀ : z = 0 - 2 * complex.I) (h₁: c = 1 + 2 * complex.I) (h₂ : k = 4) :
  (z - c = k * (0 - 2 * complex.I - c)) → (z = -3 - 14 * complex.I) :=
by
  intro h
  have h₃ : z - (1 + 2 * complex.I) = 4 * ((0 - 2 * complex.I) - (1 + 2 * complex.I)), by rw [h₀, h₁, h₂],
  have h₄ : z - (1 + 2 * complex.I) = -4 - 16 * complex.I, by simp [h₃],
  have h₅ : z = -4 - 16 * complex.I + 1 + 2 * complex.I, by rwa [h₄],
  have h₆ : z = -3 - 14 * complex.I, by simp [h₅],
  exact h₆

end dilation_transform_l75_75121


namespace largest_divisor_of_consecutive_odd_product_l75_75623

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : n > 0) :
  315 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) := 
sorry

end largest_divisor_of_consecutive_odd_product_l75_75623


namespace range_of_k_l75_75892

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x > k → 3 / (x + 1) < 1) ∧ ¬(∀ x : ℝ, 3 / (x + 1) < 1 → x > k) ↔ k ∈ set.Ici 2 :=
by
  sorry

end range_of_k_l75_75892


namespace modulo_residue_addition_l75_75308

theorem modulo_residue_addition : 
  (368 + 3 * 78 + 8 * 242 + 6 * 22) % 11 = 8 := 
by
  have h1 : 368 % 11 = 5 := by sorry
  have h2 : 78 % 11 = 1 := by sorry
  have h3 : 242 % 11 = 0 := by sorry
  have h4 : 22 % 11 = 0 := by sorry
  sorry

end modulo_residue_addition_l75_75308


namespace distance_midpoints_is_sqrt5_over2_l75_75086

noncomputable def distance_between_midpoints 
  (a b c d : ℝ) : ℝ :=
  let m := (a + c) / 2
  let n := (b + d) / 2
  let mp_x := m - 1/2
  let mp_y := n + 1
  sqrt ((mp_x - m) ^ 2 + (mp_y - n) ^ 2)

theorem distance_midpoints_is_sqrt5_over2
  (a b c d : ℝ) : 
  distance_between_midpoints a b c d = sqrt 5 / 2 :=
by
  sorry

end distance_midpoints_is_sqrt5_over2_l75_75086


namespace k_tasty_exists_l75_75620

def is_k_tasty (n k : ℕ) : Prop :=
  ∃ (a : Fin (n + 1) → Fin (n + 1)), 
    (∀ (i : ℕ) (hi : i < n), |(a (Fin.ofNat (i + 1))).val - (a (Fin.ofNat i)).val| ∈ {k, k + 1}) ∧
    ∀ (i : Fin (n + 1)), (∃ (j : ℕ), j < n + 1 ∧ a i = Fin.ofNat j)

theorem k_tasty_exists (k : ℕ) (hk : k > 0) : 
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → is_k_tasty n k := 
by 
  sorry

end k_tasty_exists_l75_75620


namespace least_pos_int_for_multiple_of_5_l75_75223

theorem least_pos_int_for_multiple_of_5 (n : ℕ) (h1 : n = 725) : ∃ x : ℕ, x > 0 ∧ (725 + x) % 5 = 0 ∧ x = 5 :=
by
  sorry

end least_pos_int_for_multiple_of_5_l75_75223


namespace central_angle_of_sector_l75_75903

theorem central_angle_of_sector (r A : ℝ) (h₁ : r = 4) (h₂ : A = 4) :
  (1 / 2) * r^2 * (1 / 4) = A :=
by
  sorry

end central_angle_of_sector_l75_75903


namespace probability_of_B_l75_75898

noncomputable def A : set Ω := {}
noncomputable def B : set Ω := {}

-- Given conditions
axiom P : set Ω → ℝ
axiom mutually_exclusive : P (A ∩ B) = 0
axiom PA : P A = 0.25
axiom PA_union_B : P (A ∪ B) = 0.78

-- The proof goal
theorem probability_of_B : P B = 0.53 :=
by
  sorry

end probability_of_B_l75_75898


namespace stella_toilet_paper_packs_l75_75093

-- Define the relevant constants/conditions
def rolls_per_bathroom_per_day : Nat := 1
def number_of_bathrooms : Nat := 6
def days_per_week : Nat := 7
def weeks : Nat := 4
def rolls_per_pack : Nat := 12

-- Theorem statement
theorem stella_toilet_paper_packs :
  (rolls_per_bathroom_per_day * number_of_bathrooms * days_per_week * weeks) / rolls_per_pack = 14 :=
by
  sorry

end stella_toilet_paper_packs_l75_75093


namespace fraction_of_selected_color_films_l75_75759

variables (x y : ℕ) (y_is_nonzero : y ≠ 0) (x_is_nonzero : x ≠ 0)

theorem fraction_of_selected_color_films (h : x * y ≠ 0) :
  let selected_bw := (y / x * (20 * x)) / 100 in
  let selected_color := 6 * y in
  let total_selected := selected_bw + selected_color in
  (6 * y) / total_selected = 6 / 31 := by
  sorry

end fraction_of_selected_color_films_l75_75759


namespace tallest_player_height_correct_l75_75153

-- Define the height of the shortest player
def shortest_player_height : ℝ := 68.25

-- Define the height difference between the tallest and shortest player
def height_difference : ℝ := 9.5

-- Define the height of the tallest player based on the conditions
def tallest_player_height : ℝ :=
  shortest_player_height + height_difference

-- Theorem statement
theorem tallest_player_height_correct : tallest_player_height = 77.75 := by
  sorry

end tallest_player_height_correct_l75_75153


namespace function_eq_square_l75_75848

theorem function_eq_square (f : ℕ → ℕ) :
  (∀ m n : ℕ, 0 < m → 0 < n → f(m) + f(n) - m * n ≠ 0 ∧ (f(m) + f(n) - m * n) ∣ (m * f(m) + n * f(n))) → (∀ n : ℕ, 0 < n → f(n) = n ^ 2) :=
by
  sorry

end function_eq_square_l75_75848


namespace sum_of_squares_ends_in_25_is_644_l75_75686

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75686


namespace value_of_a_l75_75015

theorem value_of_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h1 : a^b = b^a) (h2 : b = 9 * a) : 
  a = Real.root 4 3 :=
by
  sorry

end value_of_a_l75_75015


namespace largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75208

open Nat

theorem largest_n_binom_10_3_10_4_eq_binom_11_n :
  ∃ n, (n ≤ 11) ∧ (∑ i in finset.range 11, if i = 4 then 1 else 0) = 1 ∧
  (n = 7) := by
sorry

end largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75208


namespace quadratic_intersection_with_x_axis_l75_75587

theorem quadratic_intersection_with_x_axis :
  ∃ x : ℝ, (x^2 - 4*x + 4 = 0) ∧ (x = 2) ∧ (x, 0) = (2, 0) :=
sorry

end quadratic_intersection_with_x_axis_l75_75587


namespace cost_price_of_article_l75_75076

theorem cost_price_of_article :
  ∃ (CP : ℝ), 
    let MP := 1.15 * CP in
    let SP := 0.7439613526570048 * MP in
    SP = 462 ∧ CP = 540 :=
by
  sorry

end cost_price_of_article_l75_75076


namespace sum_G_from_2_to_20_l75_75368

def G (n : ℕ) : ℕ := 
  if n % 4 = 1 then 2 * n - 2 
  else if n % 4 = 3 ∨ n % 2 = 0 then 2 * n 
  else if n % 4 = 0 then 2 * n + 2 
  else 0

theorem sum_G_from_2_to_20 : (∑ n in Finset.range (21 - 2 + 1), G (n + 2)) = 456 := 
  sorry

end sum_G_from_2_to_20_l75_75368


namespace logarithm_relationship_l75_75370

variable (a b c x : ℝ) (p q r y : ℝ)

theorem logarithm_relationship (h₁ : log a / p = log b / q)
                               (h₂ : log b / q = log c / r)
                               (h₃ : log c / r = log x)
                               (hx : x ≠ 1)
                               (h₄ : b^3 / (a^2 * c) = x^y) :
  y = 3*q - 2*p - r := by
  sorry

end logarithm_relationship_l75_75370


namespace sum_of_squares_ending_in_25_l75_75719

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75719


namespace series_sum_100_smallest_a_b_c_l75_75316

theorem series_sum_100 :
  ∑ k in Finset.range 100 + 1, (-1)^k * (k^3 + k + 1) / (k.factorial) = (10001 / 100.factorial) - 1 :=
sorry

theorem smallest_a_b_c :
  (∑ k in Finset.range 100 + 1, (-1)^k * (k^3 + k + 1) / (k.factorial) = (10001 / 100.factorial) - 1) ->
  ∃ (a b c : ℕ), a = 10001 ∧ b = 100 ∧ c = 1 ∧ a + b + c = 10102 :=
by
  intro hyp
  use 10001, 100, 1
  simp [hyp]
  norm_num
  sorry

end series_sum_100_smallest_a_b_c_l75_75316


namespace largest_number_of_students_awarded_extra_credit_l75_75080

theorem largest_number_of_students_awarded_extra_credit
  (total_students : ℕ)
  (total_students_eq : total_students = 107)
  (mean : ℕ → ℚ)
  (scores : fin total_students → ℚ)
  (mean_scores_eq : mean total_students = (finset.univ.sum' scores / total_students : ℚ))
  : ∃ n, n ≤ total_students ∧ (∀ (k : fin total_students), scores k > mean total_students ↔ ↑k < n) ∧ n = 106 := 
sorry

end largest_number_of_students_awarded_extra_credit_l75_75080


namespace probability_phone_booth_l75_75162

theorem probability_phone_booth (P : ℕ → ℚ) (h : ∀ n, P(n) = if n = 0 then P 0 else if 1 ≤ n ∧ n ≤ 6 then (1/2)^n * P 0 else 0)
  (h_sum : ∑ n in finset.range 7, P n = 1) : 
  P 0 = 64 / 127 :=
by
  sorry

end probability_phone_booth_l75_75162


namespace sum_of_two_digit_numbers_ending_in_25_l75_75630

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75630


namespace number_of_teachers_sampled_from_C_l75_75938

def num_teachers_A : ℕ := 180
def num_teachers_B : ℕ := 270
def num_teachers_C : ℕ := 90
def total_teachers : ℕ := num_teachers_A + num_teachers_B + num_teachers_C
def total_sampled_teachers : ℕ := 60
def proportion_C : ℚ := num_teachers_C / total_teachers
def sampled_teachers_C : ℕ := nat.floor (total_sampled_teachers * proportion_C)

theorem number_of_teachers_sampled_from_C : sampled_teachers_C = 10 :=
by {
  sorry
}

end number_of_teachers_sampled_from_C_l75_75938


namespace four_digit_numbers_formed_2025_l75_75425

theorem four_digit_numbers_formed_2025 : 
  (let digits := [2, 0, 2, 5] in
  let valid_digits := [2, 5] in
  let count_permutations (d : ℕ) :=
    if d = 2 then (3! / 1!) else (3! / 2!) in
  list.sum (list.map count_permutations valid_digits) = 6 ) :=
sorry

end four_digit_numbers_formed_2025_l75_75425


namespace apple_tree_bears_fruit_in_7_years_l75_75075

def age_planted : ℕ := 4
def age_eats : ℕ := 11
def time_to_bear_fruit : ℕ := age_eats - age_planted

theorem apple_tree_bears_fruit_in_7_years :
  time_to_bear_fruit = 7 :=
by
  sorry

end apple_tree_bears_fruit_in_7_years_l75_75075


namespace min_value_l75_75832

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin (x / 2018) + (2019 ^ x - 1) / (2019 ^ x + 1)

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : f (2 * a) + f (b - 4) = 0) :
  2 * a + b = 4 → (1 / a + 2 / b) = 2 :=
by sorry

end min_value_l75_75832


namespace equivalent_angle_l75_75353

theorem equivalent_angle (θ : ℝ) : 
  (∃ k : ℤ, θ = k * 360 + 257) ↔ θ = -463 ∨ (∃ k : ℤ, θ = k * 360 + 257) :=
by
  sorry

end equivalent_angle_l75_75353


namespace sum_of_squares_ending_in_25_l75_75716

theorem sum_of_squares_ending_in_25 : 
  let two_digit_nums_with_square_ending_25 := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n^2 % 100 = 25)},
  sum := ∑ n in two_digit_nums_with_square_ending_25, n,
  sum = 495 :=
by
  sorry

end sum_of_squares_ending_in_25_l75_75716


namespace solve_ax_plus_b_l75_75029

theorem solve_ax_plus_b (a b : ℝ) : 
  (if a ≠ 0 then "unique solution, x = -b / a"
   else if b ≠ 0 then "no solution"
   else "infinitely many solutions") = "A conditional control structure should be adopted" :=
sorry

end solve_ax_plus_b_l75_75029


namespace part_one_part_two_l75_75450

-- Define vectors a, b, and c
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Define the first proof problem
theorem part_one (m n : ℝ) :
  a = m • b + n • c → m = 5 / 9 ∧ n = 8 / 9 := sorry

-- Define the second proof problem
theorem part_two (k : ℝ) :
  (a + k • c) ∥ (2 • b - a) → k = -16 / 13 := sorry

end part_one_part_two_l75_75450


namespace perpendicular_segment_length_l75_75977

theorem perpendicular_segment_length 
  (A B C D E F G H : Type)
  (AD BE CF : ℕ)
  (h1 : AD = 12)
  (h2 : BE = 8)
  (h3 : CF = 30)
  (h4 : G = (A + B + C) / 3) -- Here centered representation of centroid
  (h5 : GH = (D + E + F) / 3) -- Here defined that the distance GH
  : GH = 50 / 3 := sorry

end perpendicular_segment_length_l75_75977


namespace sum_of_two_digit_numbers_ending_in_25_l75_75671

theorem sum_of_two_digit_numbers_ending_in_25 : 
  (∑ n in {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 5}, n) = 495 :=
  by sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75671


namespace lisa_additional_marbles_l75_75538

theorem lisa_additional_marbles (n : ℕ) (f : ℕ) (m : ℕ) (current_marbles : ℕ) : 
  n = 12 ∧ f = n ∧ m = (n * (n + 1)) / 2 ∧ current_marbles = 34 → 
  m - current_marbles = 44 :=
by
  intros
  sorry

end lisa_additional_marbles_l75_75538


namespace suff_but_not_nec_l75_75890

-- Definition of proposition p
def p (m : ℝ) : Prop := m = -1

-- Definition of proposition q
def q (m : ℝ) : Prop := 
  let line1 := fun (x y : ℝ) => x - y = 0
  let line2 := fun (x y : ℝ) => x + (m^2) * y = 0
  ∀ (x1 y1 x2 y2 : ℝ), line1 x1 y1 → line2 x2 y2 → (x1 = x2 → y1 = -y2)

-- The proof problem
theorem suff_but_not_nec (m : ℝ) : p m → q m ∧ (q m → m = -1 ∨ m = 1) :=
sorry

end suff_but_not_nec_l75_75890


namespace largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75206

open Nat

theorem largest_n_binom_10_3_10_4_eq_binom_11_n :
  ∃ n, (n ≤ 11) ∧ (∑ i in finset.range 11, if i = 4 then 1 else 0) = 1 ∧
  (n = 7) := by
sorry

end largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75206


namespace no_positive_integer_solution_l75_75428

theorem no_positive_integer_solution (x : ℕ) (hx : 0 < x) : ¬ (| x + 4 | < x) :=
sorry

end no_positive_integer_solution_l75_75428


namespace no_functions_satisfy_inequality_l75_75360

theorem no_functions_satisfy_inequality :
  ∀ (f : ℝ → ℝ), (∀ x y z : ℝ, f (x * y) + f (x * z) - f (x) * f (y * z) > 1) → false :=
by
  assume f h,
  have h0 := h 0 0 0,
  have h_sq : (f 0 - 1)^2 < 0 := by
    calc
      (f 0 - 1)^2
      = f 0^2 - 2 * f 0 * 1 + 1 : by ring
      ... < 0 : by sorry, -- use h0 here to show the contradiction
  exact lt_irrefl _ h_sq

end no_functions_satisfy_inequality_l75_75360


namespace intersection_single_point_l75_75602

def A (x y : ℝ) := x^2 + y^2 = 4
def B (x y : ℝ) (r : ℝ) := (x - 3)^2 + (y - 4)^2 = r^2

theorem intersection_single_point (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, A p.1 p.2 ∧ B p.1 p.2 r) → r = 3 :=
by
  apply sorry -- Proof goes here

end intersection_single_point_l75_75602


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75704

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75704


namespace is_tangent_l75_75072

variable (α : Type) [EuclideanGeometry α] -- Assuming Euclidean Geometry context in Lean

open EuclideanGeometry

-- Define the isosceles triangle ABC with AC = BC
variable (A B C D E : α)
variable (h_iso : AC = BC)
variable (h_circumcircle : is_circumcircle k (triangle ABC))
variable (h_D_on_arc : D ∈ (shorter_arc B C k) ∧ D ≠ B ∧ D ≠ C)
variable (h_E_intersection : E ∈ (line CD) ∩ (line AB))

-- Prove that line BC is tangent to the circumcircle of triangle BDE at point B
theorem is_tangent (h_tangent : tangent (line BC) (circumcircle (triangle BDE)) B) : Prop :=
sorry

end is_tangent_l75_75072


namespace fraction_of_larger_sphere_volume_l75_75506

-- Definitions related to the problem conditions
def larger_sphere_radius_from_unit_spheres_touching : ℝ :=
  let r := (Real.sqrt 6) / 2 in
  r + 1

def larger_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * R^3

def unit_sphere_volume : ℝ :=
  (4 / 3) * Real.pi * 1^3

def combined_unit_spheres_volume : ℝ :=
  4 * unit_sphere_volume

def volume_fraction (unit_spheres_vol : ℝ) (larger_sphere_vol : ℝ) : ℝ :=
  unit_spheres_vol / larger_sphere_vol

-- Lean 4 statement to prove the given question and conditions
theorem fraction_of_larger_sphere_volume :
  volume_fraction combined_unit_spheres_volume (larger_sphere_volume larger_sphere_radius_from_unit_spheres_touching) = 0.36 :=
by
  sorry

end fraction_of_larger_sphere_volume_l75_75506


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75700

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75700


namespace evaluate_expression_l75_75341

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l75_75341


namespace alice_bob_meet_after_six_turns_l75_75288

/-
Alice and Bob play a game involving a circle whose circumference
is divided by 12 equally-spaced points. The points are numbered
clockwise, from 1 to 12. Both start on point 12. Alice moves clockwise
and Bob, counterclockwise. In a turn of the game, Alice moves 5 points 
clockwise and Bob moves 9 points counterclockwise. The game ends when they stop on
the same point. 
-/
theorem alice_bob_meet_after_six_turns (k : ℕ) :
  (5 * k) % 12 = (12 - (9 * k) % 12) % 12 -> k = 6 :=
by
  sorry

end alice_bob_meet_after_six_turns_l75_75288


namespace triangle_angle_bisector_l75_75807

open Real EuclideanGeometry

noncomputable def a : ℝ := sorry
noncomputable def c : ℝ := sorry
def a_plus_c : ℝ := sorry

theorem triangle_angle_bisector (a c : ℝ) (a_plus_c : ℝ) : 
  let A := (1, 4) 
  let B := (5, -3) 
  let C := (10, 2) 
  let eqn := λ x y : ℝ, a * x + 2 * y + c = 0
in
  ∃ (a c : ℝ), eqn = 0 ∧ a + c = a_plus_c := 
sorry

end triangle_angle_bisector_l75_75807


namespace arithmetic_sequence_problem_l75_75497

theorem arithmetic_sequence_problem : 
  ∀ (a : ℕ → ℕ) (d : ℕ), 
  a 1 = 1 →
  (a 3 + a 4 + a 5 + a 6 = 20) →
  a 8 = 9 :=
by
  intros a d h₁ h₂
  -- We skip the proof, leaving a placeholder.
  sorry

end arithmetic_sequence_problem_l75_75497


namespace min_value_of_frac_expr_l75_75372

theorem min_value_of_frac_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (3 / a) + (2 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end min_value_of_frac_expr_l75_75372


namespace parabola_standard_form_l75_75149

-- Definitions
def directrix (p : ℝ × ℝ → Prop) : Prop :=
  ∀ y, p (3, y)

def standard_form (p : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, p (x, y) ↔ y ^ 2 = -12 * x

-- Theorem statement
theorem parabola_standard_form :
  ∃ p : ℝ × ℝ → Prop, directrix p → standard_form p :=
begin
  sorry
end

end parabola_standard_form_l75_75149


namespace a7_b7_equals_29_l75_75082

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry

def cond1 := a + b = 1
def cond2 := a^2 + b^2 = 3
def cond3 := a^3 + b^3 = 4
def cond4 := a^4 + b^4 = 7
def cond5 := a^5 + b^5 = 11

theorem a7_b7_equals_29 : cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 → a^7 + b^7 = 29 :=
by
  sorry

end a7_b7_equals_29_l75_75082


namespace smallest_positive_angle_l75_75397

open Real

theorem smallest_positive_angle (α : ℝ) 
  (h1 : (sin (2 * π / 3), cos (2 * π / 3)) = (sin α, cos α)) :
  α = 11 * π / 6 :=
by sorry

end smallest_positive_angle_l75_75397


namespace locus_of_point_M_l75_75879

theorem locus_of_point_M (O A : Point) (R : ℝ) (M : Point)
    (hA_outside_circle : dist O A > R)
    (h_tangents_intersect_at_M : ∃ B : Point, is_tangent (circle O R) A B ∧ tangent_intersects_in_M A B M) :
    ∃ L : Line, is_perpendicular L (line_through O A) ∧ M ∈ L :=
by
  sorry

end locus_of_point_M_l75_75879


namespace choose_4_captains_from_15_l75_75969

def num_ways_to_choose_captains : ℕ := 15.choose 4

theorem choose_4_captains_from_15 : num_ways_to_choose_captains = 1365 := by
  sorry

end choose_4_captains_from_15_l75_75969


namespace trajectory_closes_l75_75384

theorem trajectory_closes (ABC : Triangle) (M : Point) 
  (h0 : M ∈ ABC.interior)
  (h1 : ∀ (P Q : Point), P ∈ Triangle.side BC ∧ Q ∈ Triangle.side CA → P ≠ Q → P.moves_parallel_to BC.to Q)
  (h2 : ∀ (P Q : Point), P ∈ Triangle.side CA ∧ Q ∈ Triangle.side BC → P ≠ Q → P.moves_parallel_to CA.to Q)
  (h3 : ∀ (P Q : Point), P ∈ Triangle.side AB ∧ Q ∈ Triangle.side BC → P ≠ Q → P.moves_parallel_to AB.to Q)
  (h4 : ∀ (P Q : Point), P ∈ Triangle.side BC ∧ Q ∈ Triangle.side AB → P ≠ Q → P.moves_parallel_to BC.to Q)
  (h5 : ∀ (P Q : Point), P ∈ Triangle.side AB ∧ Q ∈ Triangle.side CA → P ≠ Q → P.moves_parallel_to AB.to Q)
  (h6 : ∀ (P Q : Point), P ∈ Triangle.side CA ∧ Q ∈ Triangle.side AB → P ≠ Q → P.moves_parallel_to CA.to Q) :
  ∃ N : ℕ, let trajectory := λ k : ℕ, move_point_k_times M ABC k ∈ M → trajectory N = M := sorry

end trajectory_closes_l75_75384


namespace pascal_even_rows_perfect_squares_l75_75010

theorem pascal_even_rows_perfect_squares :
  (Nat.filter (λ n => n ≤ 30 ∧ n > 1 ∧ ∃ k, n = 2^k ∧ Nat.sqrt (2^n) = Nat.sqrt (2^n) ) (List.range 31)).length = 4 := 
sorry

end pascal_even_rows_perfect_squares_l75_75010


namespace complement_intersection_l75_75419

def U : Set ℤ := Set.univ
def A : Set ℤ := {1, 2}
def B : Set ℤ := {3, 4}

-- A ∪ B should equal {1, 2, 3, 4}
axiom AUeq : A ∪ B = {1, 2, 3, 4}

theorem complement_intersection : (U \ A) ∩ B = {3, 4} :=
by
  sorry

end complement_intersection_l75_75419


namespace find_matrix_N_l75_75358

-- Define a 3x3 matrix type
def mat3 := matrix (fin 3) (fin 3) ℝ

-- Define a condition that matrix N multiplied by any vector v is equal to -7 times that vector
def condition (N : mat3) : Prop :=
  ∀ (v : vector ℝ 3), matrix.mul_vec N v = -7 • v

-- State the main theorem
theorem find_matrix_N (N : mat3) : condition N → N = matrix.scalar (fin 3) (-7) :=
by sorry

end find_matrix_N_l75_75358


namespace divide_grid3x6_into_LShapes_l75_75970

-- Define the L-shaped piece which covers exactly three cells
structure LShape (α : Type) [HasZero α] :=
  (a b c : α)
  (h : a + b + c = 3)

-- Define a 3x6 grid
def grid3x6 := Fin (3 * 6)

-- Statement of the problem
theorem divide_grid3x6_into_LShapes :
  ∃ (ways : ℕ), ways = 8 :=
begin
  sorry
end

end divide_grid3x6_into_LShapes_l75_75970


namespace garden_perimeter_is_48_l75_75804

def square_garden_perimeter (pond_area garden_remaining_area : ℕ) : ℕ :=
  let garden_area := pond_area + garden_remaining_area
  let side_length := Int.natAbs (Int.sqrt garden_area)
  4 * side_length

theorem garden_perimeter_is_48 :
  square_garden_perimeter 20 124 = 48 :=
  by
  sorry

end garden_perimeter_is_48_l75_75804


namespace find_A_find_b_c_l75_75868

variable (a b c A B C : ℝ)
hypothesis (h1 : ∀ {A B C : ℕ} (a b c : ℝ), a, b, c are the sides opposite to angles A, B, C in  \triangle ABC respectively)
hypothesis (h2 : ∀ (a b c A B C: ℝ), c = a * Real.sin C - c * Real.cos A)

theorem find_A (h1) (h2)  : A = Real.pi / 2 := by
  sorry

variable (area : ℝ)
hypothesis (h3 : area = 2)
hypothesis (h4 : ∀ (a : ℕ),  a = 2 ) 

theorem find_b_c (h1) (h2) (h3) (h4)  : (b = 2) ∧ (c = 2) := by
  sorry

end find_A_find_b_c_l75_75868


namespace Bob_wins_2012_grid_Alice_wins_2011_grid_l75_75761

-- Part (a): $2012 \times 2012$ grid
theorem Bob_wins_2012_grid :
  ∀ (grid : ℕ × ℕ), grid = (2012, 2012) →
  ∃ strategy_bob : (ℕ × ℕ) → (ℕ × ℕ), winning_strategy strategy_bob :=
sorry

-- Part (b): $2011 \times 2011$ grid
theorem Alice_wins_2011_grid :
  ∀ (grid : ℕ × ℕ), grid = (2011, 2011) →
  ∃ strategy_alice : (ℕ × ℕ) → (ℕ × ℕ), winning_strategy strategy_alice :=
sorry

end Bob_wins_2012_grid_Alice_wins_2011_grid_l75_75761


namespace shaded_area_correct_l75_75169

-- Definitions of the given conditions
def first_rectangle_length : ℕ := 8
def first_rectangle_width : ℕ := 5
def second_rectangle_length : ℕ := 4
def second_rectangle_width : ℕ := 9
def overlapping_area : ℕ := 3

def first_rectangle_area := first_rectangle_length * first_rectangle_width
def second_rectangle_area := second_rectangle_length * second_rectangle_width

-- Problem statement in Lean 4
theorem shaded_area_correct :
  first_rectangle_area + second_rectangle_area - overlapping_area = 73 :=
by
  -- The proof is skipped
  sorry

end shaded_area_correct_l75_75169


namespace arithmetic_sequence_solution_l75_75517

variable {α : Type}

/-- Define an arithmetic sequence with common difference not equal to zero. -/
structure ArithmeticSequence (α : Type) :=
(a_n : ℕ → α)
(d : α)
(hd_ne_zero : d ≠ 0)

noncomputable def general_formula_a (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

noncomputable def sum_S (a1 : ℤ) (d : ℤ) (n : ℕ): ℤ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (h : d ≠ 0)

-- Given conditions
variable (h1 : a 3 = sum_S (a 1) d 5)
variable (h2 : a 2 * a 4 = sum_S (a 1) d 4)

-- Definitions to appear directly from the conditions
def a_n := λ n : ℕ, 2 * n - 6
def S_n := λ n: ℕ, n^2 - 5 * n

-- Lean 4 statement for the proof problem
theorem arithmetic_sequence_solution :
  (∀ n : ℕ, a n = a_n n) ∧
  (∀ n : ℕ, S_n n > a_n n → n ≥ 7) := 
sorry

end arithmetic_sequence_solution_l75_75517


namespace average_speed_round_trip_l75_75784

theorem average_speed_round_trip 
  (D : ℝ)
  (h : D > 0) 
  (speed_to_sf : ℝ := 63)
  (time_to_sf : ℝ := D / speed_to_sf) 
  (time_from_sf : ℝ := 2 * time_to_sf) :
  let total_distance := 2 * D
  let total_time := time_to_sf + time_from_sf
  average_speed := total_distance / total_time
  average_speed = 42 :=
by
  -- Definitions based on problem statement
  let time_to_sf := D / 63
  let time_from_sf := 2 * time_to_sf
  let total_time := time_to_sf + time_from_sf
  let total_distance := 2 * D
  let average_speed := total_distance / total_time
  
  -- Proof of the average speed for the round trip
  have : time_to_sf = D / 63 := rfl
  have : time_from_sf = 2 * (D / 63) := by rw [time_to_sf, mul_assoc, mul_div_cancel' _ (by norm_num : 63 ≠ 0)]
  have : total_time = (D / 63) + 2 * (D / 63) := by rw [time_to_sf, time_from_sf]
  have : total_time = 3 * (D / 63) := by ring
  have : total_time = D / 21 := by rw [← mul_div_assoc, mul_div_cancel' _ (by norm_num : 3 ≠ 0)]
  have : total_distance = 2 * D := rfl
  have : average_speed = (2 * D) / (D / 21) := by rw [total_distance, total_time]
  have : average_speed = 2 * D * (21 / D) := by rw [div_mul_eq_mul_div, div_eq_mul_one_div]
  have : average_speed = 2 * 21 := by rw [mul_comm (21 : ℝ), mul_assoc, div_mul_cancel _ (by linarith)]
  exact eq.symm (by norm_num),
  exact h

end average_speed_round_trip_l75_75784


namespace largest_n_binom_sum_l75_75199

open Nat

theorem largest_n_binom_sum (n : ℕ) 
  (h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n) : n = 7 :=
by 
  sorry

end largest_n_binom_sum_l75_75199


namespace sheets_required_l75_75239

open Real

noncomputable def plank_area (width length : ℕ) : ℕ :=
  width * length * 100 * 100 -- converting meters to centimeters and then calculating area

noncomputable def colored_paper_area (width height : ℕ) : ℕ :=
  width * height

noncomputable def number_of_sheets (plank_width plank_length paper_width paper_height : ℕ) : ℕ :=
  (plank_area plank_width plank_length) / (colored_paper_area paper_width paper_height)

theorem sheets_required :
  number_of_sheets 6 4 60 20 = 200 :=
by
  simp [number_of_sheets, plank_area, colored_paper_area]
  norm_num
  sorry

end sheets_required_l75_75239


namespace find_length_PQ_l75_75503

variable {A B C D P Q F : Point}

-- Definitions for Trapezoid and Parallelism Conditions
def is_trapezoid (A B C D : Point) : Prop :=
  ∃ (BC AD : Line), parallel BC AD ∧ (BC = 800) ∧ (AD = 1600)

def is_midpoint (P : Point) (BC : Line) : Prop :=
  distance P (endpoint1 BC) = distance P (endpoint2 BC)

def is_midpoint' (Q : Point) (AD : Line) : Prop :=
  distance Q (endpoint1 AD) = distance Q (endpoint2 AD)

-- Definitions for angles
def angle_A (A B F : Point) : Prop := angle A == 45
def angle_D (D B F : Point) : Prop := angle D == 45

-- Length PQ
def segment_PQ (P Q : Point) : ℝ := distance P Q

-- The theorem to be proved
theorem find_length_PQ :
  is_trapezoid A B C D →
  is_midpoint P BC →
  is_midpoint' Q AD →
  angle_A A B F →
  angle_D D B F →
  segment_PQ P Q = 400 := by
  sorry

end find_length_PQ_l75_75503


namespace john_paid_after_tax_l75_75987

-- Definitions based on problem conditions
def original_cost : ℝ := 200
def tax_rate : ℝ := 0.15

-- Definition of the tax amount
def tax_amount : ℝ := tax_rate * original_cost

-- Definition of the total amount paid
def total_amount_paid : ℝ := original_cost + tax_amount

-- Theorem statement for the proof
theorem john_paid_after_tax : total_amount_paid = 230 := by
  sorry

end john_paid_after_tax_l75_75987


namespace sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75743

theorem sum_of_two_digit_numbers_with_squares_ending_in_25 :
  let n_values := {n | ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 10 * a + 5} in
  (∑ n in n_values, n) = 500 :=
by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_25_l75_75743


namespace ceil_floor_eq_zero_l75_75332

theorem ceil_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (- (7 / 3)) = 0) :=
by
  sorry

end ceil_floor_eq_zero_l75_75332


namespace minimum_species_l75_75482

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l75_75482


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75699

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75699


namespace monotonic_increasing_intervals_l75_75132

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x + 1

theorem monotonic_increasing_intervals :
  (∀ x, x < 1 → (f' x > 0)) ∧ (∀ x, x > 2 → (f' x > 0)) :=
sorry

end monotonic_increasing_intervals_l75_75132


namespace find_m_in_triangle_DEF_l75_75978

theorem find_m_in_triangle_DEF
  (DE DF EF : ℝ)
  (h1 : DE = 5)
  (h2 : DF = 13)
  (h3 : EF = 12)
  (G_mid : ∃ G, (G = (EF / 2)) ∧ (DG = m * sqrt(2))) :
  m = (sqrt(122)) / 2 := 
sorry

end find_m_in_triangle_DEF_l75_75978


namespace angle_difference_l75_75026

-- Define the conditions
variables (A B : ℝ) 

def is_parallelogram := A + B = 180
def smaller_angle := A = 70
def larger_angle := B = 180 - 70

-- State the theorem to be proved
theorem angle_difference (A B : ℝ) (h1 : is_parallelogram A B) (h2 : smaller_angle A) : B - A = 40 := by
  sorry

end angle_difference_l75_75026


namespace find_triplets_l75_75352

noncomputable def phi (t : ℝ) : ℝ := 2 * t^3 + t - 2

theorem find_triplets (x y z : ℝ) (h1 : x^5 = phi y) (h2 : y^5 = phi z) (h3 : z^5 = phi x) :
  ∃ r : ℝ, (x = r ∧ y = r ∧ z = r) ∧ (r^5 = phi r) :=
by
  sorry

end find_triplets_l75_75352


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75728

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75728


namespace berengere_contribution_l75_75816

def cost_of_cake : ℝ := 8 -- The cake costs 8 euros.
def emily_usd : ℝ := 10 -- Emily has a ten-dollar bill.
def exchange_rate : ℝ := 1 / 1.10 -- Exchange rate is 1 euro = 1.10 USD.

theorem berengere_contribution : ∀ (cost_of_cake emily_usd exchange_rate : ℝ), 
  cost_of_cake = 8 → 
  emily_usd = 10 → 
  exchange_rate = 1 / 1.10 → 
  ∃ contribution : ℝ, contribution = 0 :=
by
  intros cost_of_cake emily_usd exchange_rate hcost hemily hexrate
  have emily_euros := (10 : ℝ) * (1 / 1.10)
  have berengere_needed := (8 : ℝ) - emily_euros
  use 0
  sorry -- Proof to be filled in.

end berengere_contribution_l75_75816


namespace no_difference_410_l75_75250

theorem no_difference_410 (n : ℕ) (R L a : ℕ) (h1 : R + L = 300)
  (h2 : L = 300 - R)
  (h3 : a ≤ 2 * R)
  (h4 : n = L + a)  :
  ¬ (n = 410) :=
by
  sorry

end no_difference_410_l75_75250


namespace subrectangles_areas_l75_75502

variable {A B C D P : Type}

variables (A B C D : Point) (P : Point)
variable [Rectangle A B C D]
variable [Inside P A B C D]

theorem subrectangles_areas (h : unit_area A B C D) :
  ∃(R1 R2 R3 R4: Rectangle), (has_vertex R1 A ∨ has_vertex R2 A ∨ has_vertex R3 A ∨ has_vertex R4 A 
    ∨ has_vertex R1 C ∨ has_vertex R2 C ∨ has_vertex R3 C ∨ has_vertex R4 C)
    → (∀ i ∈ {R1, R2, R3, R4}, area i ≤ 1/4) := 
sorry

end subrectangles_areas_l75_75502


namespace sum_of_two_digit_integers_with_square_ending_25_l75_75702

theorem sum_of_two_digit_integers_with_square_ending_25 :
  ∑ a in (Finset.range 9).map (λ k, k + 1), (10 * a + 5) = 495 :=
by
  sorry

end sum_of_two_digit_integers_with_square_ending_25_l75_75702


namespace largest_4_digit_divisible_by_98_l75_75306

theorem largest_4_digit_divisible_by_98 :
  ∃ n, (n ≤ 9999 ∧ 9999 < n + 98) ∧ 98 ∣ n :=
sorry

end largest_4_digit_divisible_by_98_l75_75306


namespace exists_x_inequality_l75_75141

theorem exists_x_inequality (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * a * x + 9 < 0) ↔ a < -2 ∨ a > 2 :=
by
  sorry

end exists_x_inequality_l75_75141


namespace constant_term_of_binomial_expansion_l75_75869

theorem constant_term_of_binomial_expansion :
  let a := ∫ x in (0:ℝ)..(2:ℝ), (1 - 2*x) in
  (a = ∫ x in (0:ℝ)..(2:ℝ), (1 - 2*x)) → 
  (let expr := (1/2 * x^2 + a / x) ^ 6 in
    ∃ c : ℤ, natDegreePolynomialxnTerm expr 0 = 60) :=
by { sorry }

end constant_term_of_binomial_expansion_l75_75869


namespace geometric_arithmetic_problem_l75_75905

variable {a1 a2 a3 b1 b2 b3 : ℝ}

-- Define arithmetic sequence property
def is_arithmetic_sequence (a1 a2 a3 : ℝ) :=
  (a2 - a1) = (a3 - a2)

-- Define geometric sequence property
def is_geometric_sequence (b1 b2 b3 : ℝ) :=
  (b2 / b1) = (b3 / b2)

-- Define specific sequences 1, a1, a2, a3, 9 and -9, b1, b2, b3, -1
def arithmetic_seq := is_arithmetic_sequence 1 a1 a2 ∧ is_arithmetic_sequence a1 a2 a3 ∧ is_arithmetic_sequence a2 a3 9
def geometric_seq := is_geometric_sequence (-9) b1 b2 ∧ is_geometric_sequence b1 b2 b3 ∧ is_geometric_sequence b2 b3 (-1)

theorem geometric_arithmetic_problem (h_arith : arithmetic_seq) (h_geom : geometric_seq) : 
  (b2 / (a1 + a3) = -3 / 10) := 
by 
  sorry

end geometric_arithmetic_problem_l75_75905


namespace sufficient_boxes_with_material_958_l75_75983

-- Definitions for the given problem conditions
def sufficient_material (surface_area : ℕ) (volume : ℕ) : Prop :=
  ∃ (l w h : ℕ), l * w * h ≥ volume ∧ 2 * (l * w + l * h + w * h) = surface_area

-- The actual theorem to be proved
theorem sufficient_boxes_with_material_958 :
  sufficient_material 958 1995 :=
by
  use [11, 13, 14]
  simp
  split
  { norm_num }
  { norm_num }
  sorry

end sufficient_boxes_with_material_958_l75_75983


namespace largest_binom_n_l75_75191

theorem largest_binom_n : ∃ (n : ℕ), 
  n ≤ 11 ∧ 
  (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 n ∧ 
  ∀ m (h : m ≤ 11), (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 m → m ≤ n 
  :=
begin
  use 7,
  sorry
end

end largest_binom_n_l75_75191


namespace eval_expr_ceil_floor_l75_75335

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l75_75335


namespace largest_binom_n_l75_75192

theorem largest_binom_n : ∃ (n : ℕ), 
  n ≤ 11 ∧ 
  (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 n ∧ 
  ∀ m (h : m ≤ 11), (∑ k in {3, 4}, Nat.choose 10 k) = Nat.choose 11 m → m ≤ n 
  :=
begin
  use 7,
  sorry
end

end largest_binom_n_l75_75192


namespace largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75211

open Nat

theorem largest_n_binom_10_3_10_4_eq_binom_11_n :
  ∃ n, (n ≤ 11) ∧ (∑ i in finset.range 11, if i = 4 then 1 else 0) = 1 ∧
  (n = 7) := by
sorry

end largest_n_binom_10_3_10_4_eq_binom_11_n_l75_75211


namespace max_prime_factors_of_p_l75_75529

noncomputable def p : ℕ := sorry
noncomputable def q : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry
def gcd (a b : ℕ) : ℕ := sorry
def lcm (a b : ℕ) : ℕ := sorry
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

axiom positive_integers (p q : ℕ) : 0 < p ∧ 0 < q
axiom gcd_divisible_by_5_distinct_primes : num_distinct_prime_factors (gcd p q) = 5
axiom lcm_divisible_by_20_distinct_primes : num_distinct_prime_factors (lcm p q) = 20
axiom p_has_fewer_primes_than_q : num_distinct_prime_factors p < num_distinct_prime_factors q

theorem max_prime_factors_of_p : num_distinct_prime_factors p ≤ 12 :=
by sorry

end max_prime_factors_of_p_l75_75529


namespace larger_triangle_perimeter_correct_l75_75831

-- Define an isosceles triangle with sides 15 cm, 30 cm, 30 cm
structure Triangle :=
  (a b c : ℕ) 
  (is_isosceles : a = b ∨ b = c ∨ c = a)

noncomputable def original_triangle : Triangle := 
{ a := 15,
  b := 30,
  c := 30,
  is_isosceles := Or.inr rfl }

-- Define the perimeter calculation for a similar triangle with a scale factor
noncomputable def scale_factor (smallest_side₁ smallest_side₂ : ℕ) :=
  (smallest_side₂ / smallest_side₁ : ℝ)

noncomputable def larger_triangle_perimeter (original : Triangle) (smallest_side_larger : ℕ) : ℕ :=
  let s := scale_factor original.a smallest_side_larger in
  let side_a := original.a * s in
  let side_b := original.b * s in
  let side_c := original.c * s in
  (side_a + side_b + side_c).toNat

-- The main statement to prove
theorem larger_triangle_perimeter_correct : 
  larger_triangle_perimeter original_triangle 75 = 375 :=
by simp [original_triangle, larger_triangle_perimeter, scale_factor]; sorry

end larger_triangle_perimeter_correct_l75_75831


namespace largest_int_n_binom_l75_75176

theorem largest_int_n_binom (
  h : nat.choose 10 3 + nat.choose 10 4 = nat.choose 11 n
) : n = 7 := by
  sorry

end largest_int_n_binom_l75_75176


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75645

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75645


namespace sum_of_integers_whose_squares_end_with_25_l75_75708

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_25 (n : ℕ) : Prop :=
  (n * n) % 100 = 25

theorem sum_of_integers_whose_squares_end_with_25 :
  (∑ n in (Finset.filter (λ n, is_two_digit_positive_integer n ∧ ends_with_25 n) (Finset.range 100)), n) = 495 :=
by
  -- This is the proof part which is being omitted
  sorry

end sum_of_integers_whose_squares_end_with_25_l75_75708


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75681

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75681


namespace find_m_of_trigonometric_roots_l75_75410

theorem find_m_of_trigonometric_roots :
  ∀ (θ : ℝ), 0 < θ ∧ θ < 2 * Real.pi →
  (∃ m : ℝ, (∀ x : ℝ, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 → (x = Real.sin θ ∨ x = Real.cos θ)) → m = Real.sqrt 3) :=
by
  intro θ hθ
  existsi Real.sqrt 3
  intro Hroots
  have h1 : Real.sin θ + Real.cos θ = (Real.sqrt 3 + 1) / 2,
  {
    sorry
  }
  have h2 : Real.sin θ * Real.cos θ = Real.sqrt 3 / 2,
  {
    sorry
  }
  cases Hroots with Hroot1 Hroot2
  simp only [sub_add_cancel] at Hroot2
  linarith
  sorry

end find_m_of_trigonometric_roots_l75_75410


namespace tallest_player_height_l75_75151

theorem tallest_player_height (shortest_player tallest_player : ℝ) (height_diff : ℝ)
  (h1 : shortest_player = 68.25)
  (h2 : height_diff = 9.5)
  (h3 : tallest_player = shortest_player + height_diff) :
  tallest_player = 77.75 :=
by {
  rw [h1, h2] at h3,
  exact h3,
  sorry
}

end tallest_player_height_l75_75151


namespace sum_of_squares_ends_in_25_is_644_l75_75691

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75691


namespace max_rooks_no_attack_l75_75625

theorem max_rooks_no_attack (n : ℕ) (h1 : n = 8)
  : ∃(rooks : Fin n → Fin n → Fin n → Prop), 
    (∀ x y z, rooks x y z → x + y + z ≡ 0 [MOD n]) ∧
    (forall x y, ∃ z, rooks x y z) ∧
    (∀ x₁ x₂ y₁ y₂ z₁ z₂, (x₁ = x₂ ∧ y₁ = y₂ ∧ rooks x₁ y₁ z₁ ∧ rooks x₂ y₂ z₂) → z₁ = z₂ ∧ (x₁ = x₂ ∧ y₁ = y₂) ∨ ((x₁ ≠ x₂) ∨ (y₁ ≠ y₂)) ) ∧
    ((∑ x y z, ite (rooks x y z) 1 0) = 64) :=
begin
  sorry
end

end max_rooks_no_attack_l75_75625


namespace max_true_statements_l75_75061

theorem max_true_statements (a b : ℝ) (ha: a < 0) (hb: b < 0) : 
  let s1 := (1 / a < 1 / b)
  let s2 := (a ^ 2 > b ^ 2)
  let s3 := (Real.sqrt a < Real.sqrt b)
  let s4 := (a ^ 3 > b ^ 3)
  let s5 := (a < 0)
  let s6 := (b < 0)
  count (λ s, s) [s1, s2, s3, s4, s5, s6] ≤ 3 :=
by
  sorry

end max_true_statements_l75_75061


namespace bananas_distribution_l75_75077

theorem bananas_distribution (a : Fin 8 → ℕ) :
  (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ (∑ i, a i = 40) → 
  ∃ (b : Fin 8 → ℕ), {b 0, b 1, b 2, b 3, b 4, b 5, b 6, b 7} = {1, 2, 3, 4, 6, 7, 8, 9} :=
by
  sorry

end bananas_distribution_l75_75077


namespace set_equality_l75_75394

open Set

variable (A : Set ℕ)

theorem set_equality (h1 : {1, 3} ⊆ A) (h2 : {1, 3} ∪ A = {1, 3, 5}) : A = {1, 3, 5} :=
sorry

end set_equality_l75_75394


namespace angle_between_vectors_60_degrees_l75_75901

variables (a b : EuclideanSpace ℝ 3)

def magnitudes (a b : EuclideanSpace ℝ 3) :=   
  ∥a∥ = 2 ∧ ∥b∥ = 1

def dot_product_condition (a b : EuclideanSpace ℝ 3) := 
  (a ⬝ (a - 2 • b)) = 2

theorem angle_between_vectors_60_degrees (a b : EuclideanSpace ℝ 3)
  (h₁ : magnitudes a b) 
  (h₂ : dot_product_condition a b) :
  real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥)) = real.pi / 3 :=
sorry

end angle_between_vectors_60_degrees_l75_75901


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75680

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75680


namespace john_total_distance_l75_75046

theorem john_total_distance :
  let mph1 := 55 in
  let time1 := 2 in
  let mph2 := 45 in
  let time2 := 1 in
  let mph3 := 15 in
  let time3 := 1.5 in
  let mph4 := 3 in
  let time4 := 0.5 in
  mph1 * time1 + mph2 * time2 + mph3 * time3 + mph4 * time4 = 179 :=
by {
  sorry
}

end john_total_distance_l75_75046


namespace ornithological_park_species_l75_75475

/-- In an ornithological park, there are 2021 birds arranged in a row.
Each pair of birds of the same species has an even number of birds between them.
Prove that the smallest number of bird species is 1011. -/
theorem ornithological_park_species (n : ℕ) (h1 : n = 2021) 
  (h2 : ∀ s : ℕ, s ∈ {1..n} → (∀ x y : ℕ, x < y ∧ x ≠ y → (∀ z : ℕ, z ∈ ({x, y} : set ℕ) → even (y - x - 1))) ) 
  : s ≥ 1011 :=
sorry

end ornithological_park_species_l75_75475


namespace largest_integer_binom_eq_l75_75201

theorem largest_integer_binom_eq :
  ∃ n : ℕ, (n = 7) ∧ (binom 11 n = binom 10 3 + binom 10 4) :=
by sorry

end largest_integer_binom_eq_l75_75201


namespace count_fractions_in_set_l75_75290

theorem count_fractions_in_set : 
  {0, -2, 5, (1 / 4 : ℚ), -3 / 10, real.pi / 2}.to_finset.filter (λ x, ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)) .card = 2 := 
sorry

end count_fractions_in_set_l75_75290


namespace sequence_sum_l75_75919

-- Assume the sum of first n terms of the sequence {a_n} is given by S_n = n^2 + n + 1
def S (n : ℕ) : ℕ := n^2 + n + 1

-- The sequence a_8 + a_9 + a_10 + a_11 + a_12 is what we want to prove equals 100.
theorem sequence_sum : S 12 - S 7 = 100 :=
by
  sorry

end sequence_sum_l75_75919


namespace choose_4_from_15_l75_75948

theorem choose_4_from_15 : Nat.choose 15 4 = 1365 := by
  sorry

end choose_4_from_15_l75_75948


namespace sequence_eventually_constant_l75_75052

open Nat Real

noncomputable def eventually_constant_sequence {c : ℝ} (c_pos : 0 < c) (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n > 0 → ∃ k : ℕ, (2^(a 1) + 2^(a 2) + ... + 2^(a n)) / n = k) ∧
  (∀ n : ℕ, n > 0 → 2^(a n) ≤ c * n)
  
theorem sequence_eventually_constant
  (c : ℝ) (hc : 0 < c) (a : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → ∃ k : ℕ, (2^(a 1) + 2^(a 2) + ... + 2^(a n)) / n = k)
  (h2 : ∀ n : ℕ, n > 0 → 2^(a n) ≤ c * n) :
  ∃ N : ℕ, ∀ m n : ℕ, m ≥ N ∧ n ≥ N → a m = a n :=
sorry

end sequence_eventually_constant_l75_75052


namespace sqrt_equiv_c_d_l75_75819

noncomputable def c : ℤ := 3
noncomputable def d : ℤ := 375

theorem sqrt_equiv_c_d : ∀ (x y : ℤ), x = 3^5 ∧ y = 5^3 → (∃ c d : ℤ, (c = 3 ∧ d = 375 ∧ x * y = c^4 * d))
    ∧ c + d = 378 := by sorry

end sqrt_equiv_c_d_l75_75819


namespace sum_of_x_coordinates_correct_l75_75859
-- Import required Lean libraries

-- Define the conditions
noncomputable def equation1 (x : ℝ) : ℝ := abs (x^2 - 4 * x + 3)
noncomputable def equation2 (x : ℝ) : ℝ := 25 / 4 - x

-- Define the proof problem
theorem sum_of_x_coordinates_correct : 
  let s := {x : ℝ | equation1 x = equation2 x} in 
  (∑ x in s, x) = 11 / 2 :=
by
  sorry

end sum_of_x_coordinates_correct_l75_75859


namespace shaded_area_l75_75258

variable (r : ℝ)

def is_right_angle_isosceles_triangle (a b : ℝ) :=
  a = b

theorem shaded_area (h : ∃ t : Triangle, is_right_angle_isosceles_triangle t.side1 t.side2 ∧ 2 * r = t.hypotenuse)
: shaded_area = r^2 := sorry

end shaded_area_l75_75258


namespace sum_of_squares_ends_in_25_is_644_l75_75690

-- Definitions based on conditions:
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop :=
  n % 100 = d

-- Proof statement:
theorem sum_of_squares_ends_in_25_is_644 :
  (∑ n in Finset.filter (λ n : ℕ, is_two_digit_number n ∧ ends_in (n^2) 25)
    (Finset.range 100)) = 644 :=
by
  sorry

end sum_of_squares_ends_in_25_is_644_l75_75690


namespace range_of_a_l75_75130

theorem range_of_a (a : ℝ) (h : ∀ x ∈ Icc 0 1, deriv (λ x, a * x^2 - (2*a - 1) * x + 3) x ≥ 0) : a ≤ 1/2 :=
sorry

end range_of_a_l75_75130


namespace sum_two_digit_numbers_ending_in_25_sq_l75_75639

theorem sum_two_digit_numbers_ending_in_25_sq : 
  (∑ n in Finset.filter (λ n : ℕ, ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ b = 5 ∧ (n^2 % 100 = 25)) (Finset.range 100), n) =
  495 :=
by
  sorry

end sum_two_digit_numbers_ending_in_25_sq_l75_75639


namespace final_state_probability_l75_75941

-- Initial conditions
def initial_state := (3, 1, 1)

-- Definition of a state transition given the new rule
structure state :=
  (raashan : ℕ)
  (sylvia : ℕ)
  (ted : ℕ)

def transition (s : state) : state :=
  sorry -- Placeholder for the transition function

-- Define the event of interest
def exact_state (s : state) : Prop :=
  s.raashan = 2 ∧ s.sylvia = 2 ∧ s.ted = 2

-- Define the probability calculation after 2023 transitions
noncomputable def probability_final_state : ℚ :=
  sorry -- Placeholder for the probabilistic calculation

-- The theorem to prove
theorem final_state_probability : probability_final_state = 1 / 6 :=
  sorry

end final_state_probability_l75_75941


namespace sin_sum_identity_l75_75100

theorem sin_sum_identity (n : ℕ) (α : ℝ) (hα : α = π / (2^(n + 1) - 1)) :
  (∑ i in (finset.range n).map ⟨λ k, 2^(k+1), λ a b h, by simp⟩, 1 / (Real.sin (i * α))) = 1 / (Real.sin α) :=
sorry

end sin_sum_identity_l75_75100


namespace find_complex_number_l75_75377

theorem find_complex_number (z : ℂ) (h1 : abs z = 1) (h2 : (3 + 4 * complex.I) * z = complex.I * (im ((3 + 4 * complex.I) * z))) :
  z = - 4 / 5 - 3 / 5 * complex.I ∨ z = 4 / 5 + 3 / 5 * complex.I :=
by
  sorry

end find_complex_number_l75_75377


namespace sum_p_q_eq_21_l75_75920

theorem sum_p_q_eq_21 (p q : ℤ) :
  {x | x^2 + 6 * x - q = 0} ∩ {x | x^2 - p * x + 6 = 0} = {2} → p + q = 21 :=
by
  sorry

end sum_p_q_eq_21_l75_75920


namespace eval_ceil_floor_sum_l75_75344

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l75_75344


namespace nth_inequality_l75_75084

theorem nth_inequality (n : ℕ) :
  (1 + ∑ k in finset.range(n + 1), 1 / (k + 2)^2 < (2 * n + 1) / (n + 1)) :=
sorry

end nth_inequality_l75_75084


namespace time_jran_l75_75049

variable (D : ℕ) (S : ℕ)

theorem time_jran (hD: D = 80) (hS : S = 10) : D / S = 8 := 
  sorry

end time_jran_l75_75049


namespace eval_ceil_floor_sum_l75_75345

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l75_75345


namespace sum_of_two_digit_integers_whose_squares_end_in_25_l75_75676

/--
Prove that the sum of all two-digit positive integers \( n \) 
such that \( n^2 \) ends with 25 is equal to 495.
-/
theorem sum_of_two_digit_integers_whose_squares_end_in_25 
  : (∑ n in finset.filter (λ n : ℕ, (10 ≤ n ∧ n ≤ 99) ∧ (n^2 % 100 = 25)) (finset.range 100)) = 495 :=
sorry

end sum_of_two_digit_integers_whose_squares_end_in_25_l75_75676


namespace alyssa_photos_vacation_l75_75289

theorem alyssa_photos_vacation
  (pages_first_section : ℕ)
  (photos_per_page_first_section : ℕ)
  (pages_second_section : ℕ)
  (photos_per_page_second_section : ℕ)
  (pages_total : ℕ)
  (photos_per_page_remaining : ℕ)
  (pages_remaining : ℕ)
  (h_total_pages : pages_first_section + pages_second_section + pages_remaining = pages_total)
  (h_photos_first_section : photos_per_page_first_section = 3)
  (h_photos_second_section : photos_per_page_second_section = 4)
  (h_pages_first_section : pages_first_section = 10)
  (h_pages_second_section : pages_second_section = 10)
  (h_photos_remaining : photos_per_page_remaining = 3)
  (h_pages_total : pages_total = 30)
  (h_pages_remaining : pages_remaining = 10) :
  pages_first_section * photos_per_page_first_section +
  pages_second_section * photos_per_page_second_section +
  pages_remaining * photos_per_page_remaining = 100 := by
sorry

end alyssa_photos_vacation_l75_75289


namespace total_cost_of_books_l75_75927

variable (C1 C2 SP1 SP2 : ℝ)

-- Conditions
def book_cost_at_loss := C1 = 210
def loss_15_percent := SP1 = C1 - 0.15 * C1
def same_selling_price := SP2 = SP1
def gain_19_percent := SP2 = C2 + 0.19 * C2

-- Theorem to be proved
theorem total_cost_of_books (h1 : book_cost_at_loss) 
                               (h2 : loss_15_percent) 
                               (h3 : same_selling_price) 
                               (h4 : gain_19_percent) : 
  C1 + C2 = 360 := 
sorry

end total_cost_of_books_l75_75927


namespace impossible_reverse_swap_l75_75599

theorem impossible_reverse_swap :
  let start_sequence := list.range' 1 100,
      target_sequence := list.range' 1 100 |> list.reverse
  in (∀ i j, (i < j) → (j - i = 2) → (list.swap start_sequence i j) ≠ target_sequence) :=
by
  sorry

end impossible_reverse_swap_l75_75599


namespace sum_of_two_digit_ints_with_squares_ending_in_25_l75_75736

/-- The sum of all two-digit positive integers whose squares end with the digits 25 is 495. -/
theorem sum_of_two_digit_ints_with_squares_ending_in_25 :
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95] in
  n_values.sum = 495 :=
by
  let n_values : List ℕ := [15, 25, 35, 45, 55, 65, 75, 85, 95]
  have h : n_values.sum = 495 := sorry
  exact h

end sum_of_two_digit_ints_with_squares_ending_in_25_l75_75736


namespace randy_biscuits_l75_75557

theorem randy_biscuits : 
  let initial := 32 
  let father_gift := 13 
  let mother_gift := 15 
  let brother_ate := 20 
  let total := initial + father_gift + mother_gift 
  total - brother_ate = 40 := 
by
  let initial := 32 
  let father_gift := 13 
  let mother_gift := 15 
  let brother_ate := 20 
  let total := initial + father_gift + mother_gift 
  show total - brother_ate = 40 from sorry

end randy_biscuits_l75_75557


namespace least_pos_int_for_multiple_of_5_l75_75222

theorem least_pos_int_for_multiple_of_5 (n : ℕ) (h1 : n = 725) : ∃ x : ℕ, x > 0 ∧ (725 + x) % 5 = 0 ∧ x = 5 :=
by
  sorry

end least_pos_int_for_multiple_of_5_l75_75222


namespace minimum_species_l75_75484

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l75_75484


namespace sum_base6_eq_3153_l75_75856

noncomputable def base6_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n => 
    let d := n % 10 in
    let r := n / 10 in
    d + 6 * base6_to_base10 r

theorem sum_base6_eq_3153 :
 ∀ a1 a2 a3 a4 : ℕ,
  base6_to_base10 a1 = 310 →
  base6_to_base10 a2 = 250 →
  base6_to_base10 a3 = 121 →
  base6_to_base10 a4 = 36 →
  let sum := a1 + a2 + a3 + a4 in
  let base6_sum := base6_to_base10 sum in
  base6_sum = 3153
:= by
  intros
  sorry

end sum_base6_eq_3153_l75_75856


namespace min_sum_of_factors_l75_75128

theorem min_sum_of_factors (a b c d : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h : a * b * c * d = nat.factorial 8) : 
  a + b + c + d ≥ 61 := 
sorry

end min_sum_of_factors_l75_75128


namespace candy_distribution_impossible_l75_75422

theorem candy_distribution_impossible :
  ∀ (candies : Fin 6 → ℕ),
  (candies 0 = 0 ∧ candies 1 = 1 ∧ candies 2 = 0 ∧ candies 3 = 0 ∧ candies 4 = 0 ∧ candies 5 = 1) →
  (∀ t, ∃ i, (i < 6) ∧ candies ((i+t)%6) = candies ((i+t+1)%6)) →
  ∃ (i : Fin 6), candies i ≠ candies ((i + 1) % 6) :=
by
  sorry

end candy_distribution_impossible_l75_75422


namespace balloons_given_away_correct_l75_75311

variable (initial_balloons : ℕ)
variable (balloons_floated_away : ℕ)
variable (balloons_grabbed : ℕ)
variable (balloons_remaining : ℕ)
variable (balloons_given_away : ℕ)

theorem balloons_given_away_correct :
  (initial_balloons - balloons_floated_away + balloons_grabbed - balloons_remaining = balloons_given_away) :=
by
  assume
    h₁ : initial_balloons = 50,
    h₂ : balloons_floated_away = 12,
    h₃ : balloons_grabbed = 11,
    h₄ : balloons_remaining = 39,
    h₅ : balloons_given_away = 10,
  sorry

end balloons_given_away_correct_l75_75311


namespace count_of_nonzero_complex_numbers_forming_equilateral_triangle_l75_75009

/-- 
There are exactly 2 nonzero complex numbers z such that the points 0, 
z, and z^2 form the vertices of an equilateral triangle in the complex plane.
-/
theorem count_of_nonzero_complex_numbers_forming_equilateral_triangle :
  ∃! z1 z2 : ℂ, z1 ≠ 0 ∧ z2 ≠ 0 ∧ z1 ≠ z2 ∧ 
                   (∃ z : ℂ, z ≠ 0 ∧
                   0, z, and z^2 form an equilateral triangle. sorry

end count_of_nonzero_complex_numbers_forming_equilateral_triangle_l75_75009


namespace sum_of_two_digit_numbers_ending_in_25_l75_75632

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75632


namespace symmetric_line_equation_l75_75125

-- Definitions of the given conditions.
def original_line_equation (x y : ℝ) : Prop := 2 * x + 3 * y + 6 = 0
def line_of_symmetry (x y : ℝ) : Prop := y = x

-- The theorem statement to prove:
theorem symmetric_line_equation (x y : ℝ) : original_line_equation y x ↔ (3 * x + 2 * y + 6 = 0) :=
sorry

end symmetric_line_equation_l75_75125


namespace area_of_set_A_l75_75894

noncomputable def area_of_A : ℝ :=
  π

theorem area_of_set_A :
  (∀ (a b : ℝ), ¬ ∃ x : ℝ, (x ^ 2 - 2 * a * x + 1 = 2 * b * (a - x))) →
  (set.univ ∩ {p : ℝ × ℝ | (p.1^2 + (p.2)^2 < 1)}).measure = π :=
by
  intro h
  sorry

end area_of_set_A_l75_75894


namespace largest_integer_binom_eq_l75_75205

theorem largest_integer_binom_eq :
  ∃ n : ℕ, (n = 7) ∧ (binom 11 n = binom 10 3 + binom 10 4) :=
by sorry

end largest_integer_binom_eq_l75_75205


namespace largest_area_of_equilateral_triangle_l75_75600

theorem largest_area_of_equilateral_triangle (P : ℝ) (h : P = 30) :
  let s := P / 3 in let A := (sqrt 3 / 4) * s^2 in A = 25 * sqrt 3 :=
by
  sorry

end largest_area_of_equilateral_triangle_l75_75600


namespace arnel_kept_fifty_pencils_l75_75814

theorem arnel_kept_fifty_pencils
    (num_boxes : ℕ) (pencils_each_box : ℕ) (friends : ℕ) (pencils_each_friend : ℕ) (total_pencils : ℕ)
    (boxes_pencils : ℕ) (friends_pencils : ℕ) :
    num_boxes = 10 →
    pencils_each_box = 5 →
    friends = 5 →
    pencils_each_friend = 8 →
    friends_pencils = friends * pencils_each_friend →
    boxes_pencils = num_boxes * pencils_each_box →
    total_pencils = boxes_pencils + friends_pencils →
    (total_pencils - friends_pencils) = 50 :=
by
    sorry

end arnel_kept_fifty_pencils_l75_75814


namespace average_of_possible_values_l75_75432

theorem average_of_possible_values 
  (x : ℝ)
  (h : Real.sqrt (2 * x^2 + 5) = Real.sqrt 25) : 
  (x = Real.sqrt 10 ∨ x = -Real.sqrt 10) → (Real.sqrt 10 + (-Real.sqrt 10)) / 2 = 0 :=
by
  sorry

end average_of_possible_values_l75_75432


namespace sum_of_two_digit_numbers_ending_in_25_l75_75638

theorem sum_of_two_digit_numbers_ending_in_25 : 
  ∑ n in (Finset.range 100).filter (λ n, n ≥ 10 ∧ n ≤ 99 ∧ (n^2 % 100 = 25)), n = 495 := 
by
  sorry

end sum_of_two_digit_numbers_ending_in_25_l75_75638


namespace female_voters_percentage_is_correct_l75_75939

def percentage_of_population_that_are_female_voters
  (female_percentage : ℝ)
  (voter_percentage_of_females : ℝ) : ℝ :=
  female_percentage * voter_percentage_of_females * 100

theorem female_voters_percentage_is_correct :
  percentage_of_population_that_are_female_voters 0.52 0.4 = 20.8 := by
  sorry

end female_voters_percentage_is_correct_l75_75939


namespace find_interest_rate_l75_75242

-- Define the conditions
def A2 : ℝ := 17640
def A3 : ℝ := 21168
variable (P : ℝ) (r : ℝ)

/-- The rate of interest given the conditions on compound interest amounts -/
theorem find_interest_rate :
  (A2 = P * (1 + r)^2) →
  (A3 = P * (1 + r)^3) →
  r ≈ 0.19977 := by
    intros h1 h2
    sorry

end find_interest_rate_l75_75242


namespace product_less_than_50_l75_75507

theorem product_less_than_50 :
  ( ∏ k in (finset.filter (λ x, x % 2 = 0) (finset.range 1 2019)), (1 + 1/k)) < 50 :=
sorry

end product_less_than_50_l75_75507


namespace equation_represents_two_intersecting_lines_l75_75755

theorem equation_represents_two_intersecting_lines :
  (∀ x y : ℝ, x^3 * (x + y - 2) = y^3 * (x + y - 2) ↔
    (x = y ∨ y = 2 - x)) :=
by sorry

end equation_represents_two_intersecting_lines_l75_75755


namespace largest_polygon_area_l75_75448

structure Polygon :=
(unit_squares : Nat)
(right_triangles : Nat)

def area (p : Polygon) : ℝ :=
p.unit_squares + 0.5 * p.right_triangles

def polygon_A : Polygon := { unit_squares := 6, right_triangles := 2 }
def polygon_B : Polygon := { unit_squares := 7, right_triangles := 1 }
def polygon_C : Polygon := { unit_squares := 8, right_triangles := 0 }
def polygon_D : Polygon := { unit_squares := 5, right_triangles := 4 }
def polygon_E : Polygon := { unit_squares := 6, right_triangles := 2 }

theorem largest_polygon_area :
  max (area polygon_A) (max (area polygon_B) (max (area polygon_C) (max (area polygon_D) (area polygon_E)))) = area polygon_C :=
by
  sorry

end largest_polygon_area_l75_75448


namespace P_never_77_l75_75553

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_never_77 (x y : ℤ) : P x y ≠ 77 := sorry

end P_never_77_l75_75553


namespace determine_a_l75_75897

theorem determine_a :
  ∀ (a : ℝ), (a ≠ 0) →
  (let l1 := λ x y : ℝ, x + a*y + 1 = 0,
       l2 := λ x y : ℝ, x - 2*y + 2 = 0,
       slope_l2 := 1/2,
       slope_l1 := -1/a
   in slope_l1 * slope_l2 = -1) →
  a = 1/2 :=
  by
  intro a h_a_ne_zero h_slope_condition
  sorry

end determine_a_l75_75897


namespace labeling_edges_complete_graph_condition_l75_75043

open Nat

theorem labeling_edges_complete_graph_condition (n : ℕ) : 
  (∃ (f : Fin (binom n 2) → Fin (binom n 3) → ℕ), 
  ∀ (a b c : Fin (binom n 3)), 
  (gcd (f a) (f c)) ∣ f b) → n ≤ 3 :=
sorry

end labeling_edges_complete_graph_condition_l75_75043


namespace minimum_bird_species_l75_75457

theorem minimum_bird_species (total_birds : ℕ) (h : total_birds = 2021) :
  ∃ (min_species : ℕ), min_species = 1011 ∧ 
  (∀ (species_array : array total_birds ℕ),
   ∀ i j : fin total_birds, 
   species_array[i] = species_array[j] → ((i ≠ j) →
   (abs (i - j) mod 2 = 0))) :=
sorry

end minimum_bird_species_l75_75457


namespace eval_expr_ceil_floor_l75_75334

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l75_75334


namespace investment_ratio_l75_75245

variable (x : ℝ)
variable (p q t : ℝ)

theorem investment_ratio (h1 : 7 * p = 5 * q) (h2 : (7 * p * 8) / (5 * q * t) = 7 / 10) : t = 16 :=
by
  sorry

end investment_ratio_l75_75245


namespace distance_to_lightning_strike_l75_75544

def speed_of_sound : ℤ := 1100    -- feet per second
def time_delay : ℤ := 15          -- seconds
def feet_per_mile : ℤ := 5280     -- conversion factor from feet to miles

theorem distance_to_lightning_strike : 
  let distance_in_feet := speed_of_sound * time_delay in
  let distance_in_miles := distance_in_feet / feet_per_mile in
  let rounded_distance := Float.round (distance_in_miles : Float) * 2 / 2 in 
  rounded_distance = 3 := 
by
  sorry

end distance_to_lightning_strike_l75_75544


namespace determine_m_l75_75418

theorem determine_m (x y m : ℝ) :
  (3 * x - y = 4 * m + 1) ∧ (x + y = 2 * m - 5) ∧ (x - y = 4) → m = 1 :=
by sorry

end determine_m_l75_75418


namespace smallest_denominator_of_sum_of_irreducible_fractions_l75_75589

theorem smallest_denominator_of_sum_of_irreducible_fractions :
  ∀ (a b : ℕ),
  Nat.Coprime a 600 → Nat.Coprime b 700 →
  (∃ c d : ℕ, Nat.Coprime c d ∧ d < 168 ∧ (7 * a + 6 * b) / Nat.gcd (7 * a + 6 * b) 4200 = c / d) →
  False :=
by
  sorry

end smallest_denominator_of_sum_of_irreducible_fractions_l75_75589


namespace find_m_if_extraneous_root_l75_75440

theorem find_m_if_extraneous_root :
  (∃ x : ℝ, x = 2 ∧ (∀ z : ℝ, z ≠ 2 → (m / (z-2) - 2*z / (2-z) = 1)) ∧ m = -4) :=
sorry

end find_m_if_extraneous_root_l75_75440


namespace sum_of_x_coords_of_solutions_l75_75858

theorem sum_of_x_coords_of_solutions :
  (∑ x in {x | |x^2 - 4 * x + 3| = 25 / 4 - x}, x) = 6 := 
by
sorry

end sum_of_x_coords_of_solutions_l75_75858


namespace minimum_species_count_l75_75462

theorem minimum_species_count {n : ℕ} (h_n : n = 2021) 
  (h_cond : ∀ i j k : ℕ, i < j ∧ j < k → 
    birds i = birds k → birds j ≠ birds i → (j - i - 1) % 2 = 1 ∧ (k - j - 1) % 2 = 1) : 
  ∃ s : ℕ, s ≥ 1011 :=
begin
  sorry
end

end minimum_species_count_l75_75462


namespace cyclists_travel_time_l75_75613

theorem cyclists_travel_time:
  ∀ (x y t : ℝ),
  (t = x - 4) ∧ (t = y - 9) →
  x = 10 ∧ y = 15 :=
by
  intros x y t h,
  cases h with h1 h2,
  have hxy : x = y - 5 := by linarith,
  have hyx_eq : 4 = (y - 9) * (y - 9) / 9 := by
    rw [h1, h2],
    field_simp,
    ring,
  have hyx : 36 = (y - 9) ^ 2 := by linarith,
  have h_y : y = 15 := by
    apply eq_of_sqr_eq_sqr,
    exact hyx,
    exact real.sqrt_neq_zero_of_mem_Ioc,
  have h_x : x = 10 := by linarith,
  exact ⟨h_x, h_y⟩,
  sorry,

end cyclists_travel_time_l75_75613


namespace imaginary_part_of_z_l75_75880

noncomputable def z (x : ℂ) : ℂ := x * (1 - 2 * complex.I) / (|(3 + 4 * complex.I)|)

theorem imaginary_part_of_z (x : ℂ) (h : x * (1 - 2 * complex.I) = |3 + 4 * complex.I|) : 
  complex.im x = 2 := 
  sorry

end imaginary_part_of_z_l75_75880


namespace calc_f_2005_2007_zero_l75_75318

variable {R : Type} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

def periodic_function (f : R → R) (p : R) : Prop :=
  ∀ x, f (x + p) = f x

theorem calc_f_2005_2007_zero
  {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_period : periodic_function f 4) :
  f 2005 + f 2006 + f 2007 = 0 :=
sorry

end calc_f_2005_2007_zero_l75_75318


namespace gracie_joe_distance_l75_75000

noncomputable def distance_between_complex_points (z1 z2 : Complex) : Real :=
  Complex.abs (z1 - z2)

theorem gracie_joe_distance :
  let gracie := -2 + 2 * Complex.i
  let joe := 2 + 3 * Complex.i
  distance_between_complex_points gracie joe = Real.sqrt 17 :=
by
  let gracie := -2 + 2 * Complex.i
  let joe := 2 + 3 * Complex.i
  sorry

end gracie_joe_distance_l75_75000


namespace ornithological_park_species_l75_75479

/-- In an ornithological park, there are 2021 birds arranged in a row.
Each pair of birds of the same species has an even number of birds between them.
Prove that the smallest number of bird species is 1011. -/
theorem ornithological_park_species (n : ℕ) (h1 : n = 2021) 
  (h2 : ∀ s : ℕ, s ∈ {1..n} → (∀ x y : ℕ, x < y ∧ x ≠ y → (∀ z : ℕ, z ∈ ({x, y} : set ℕ) → even (y - x - 1))) ) 
  : s ≥ 1011 :=
sorry

end ornithological_park_species_l75_75479


namespace verify_line_l75_75850

def is_line_equation (θ : Real) (b : Real) (x y : Real) : Prop :=
  let k := Math.tan θ
  y = k * x + b

theorem verify_line :
  let θ := 135 * Real.pi / 180   -- 135 degrees in radians
  let b := 5
  is_line_equation θ b 2 3 ∧ ¬ is_line_equation θ b 2 (-3) :=
by
  let θ := 135 * Real.pi / 180
  let b := 5
  constructor
  · simp [is_line_equation, Real.tan, θ, b]
  · simp [is_line_equation, Real.tan, θ, b]
  sorry

end verify_line_l75_75850


namespace rectangle_area_percentage_change_l75_75173

theorem rectangle_area_percentage_change
  (L W : ℝ) (X Y : ℝ) (hL_nonzero : L ≠ 0) (hW_nonzero : W ≠ 0)
  (h_perimeter : 2 * L + 2 * W = 2 * L * (1 + X / 100) + 2 * W * (1 - Y / 100)) :
  let A := L * W in
  let A' := L * (1 + X / 100) * W * (1 - Y / 100) in
  ((A' - A) / A) * 100 = X - Y - (X * Y / 100) :=
by
  sorry

end rectangle_area_percentage_change_l75_75173


namespace ratio_YQ_QN_l75_75041

variables {X Y Z Q M N : Type*} [inner_product_space ℝ (Y Z Q M N)]
variables (XY XZ YZ : ℝ) (M N Q : Y Z Q M N)
variables (angle_bisector : Prop) (ratio : ℝ)

-- Given conditions
axiom triangle_XYZ : (XY = 8) ∧ (XZ = 13) ∧ (YZ = 10)
axiom angle_bisector_XM_YN : angle_bisector

-- Problem to be solved
theorem ratio_YQ_QN : triangle_XYZ ∧ angle_bisector_XM_YN → ratio = (18 / 13) :=
by sorry

end ratio_YQ_QN_l75_75041


namespace range_of_a_part1_range_of_a_part2_l75_75415

noncomputable def f (a x : ℝ) : ℝ :=
  a * sin x - (1 / 2) * cos (2 * x) + a - (3 / a) + (1 / 2)

theorem range_of_a_part1 (a : ℝ) (h1 : a ≠ 0) :
  (∀ (x : ℝ), f a x ≤ 0) → (0 < a ∧ a ≤ 1.5) :=
sorry

theorem range_of_a_part2 (a : ℝ) (h1 : a ≠ 0) :
  (a ≥ 2 ∧ ∃ (x : ℝ), f a x ≤ 0) → (0 < a ∧ a ≤ 3) :=
sorry

end range_of_a_part1_range_of_a_part2_l75_75415


namespace median_tire_mileage_l75_75493

/-
Problem: In order to understand the performance of a certain type of tire,
8 tires were randomly selected for testing. The maximum mileage numbers 
(in units of 1,000 km) were: 96, 112, 97, 108, 99, 104, 86, 98. 
What is their median?
-/

noncomputable def tire_mileage : List ℝ := [96, 112, 97, 108, 99, 104, 86, 98]

theorem median_tire_mileage : List.median tire_mileage = 98.5 :=
by sorry

end median_tire_mileage_l75_75493


namespace num_distinct_remainders_of_prime_squared_mod_120_l75_75296

theorem num_distinct_remainders_of_prime_squared_mod_120:
  ∀ p : ℕ, Prime p → p > 5 → (p^2 % 120 = 1 ∨ p^2 % 120 = 49) := 
sorry

end num_distinct_remainders_of_prime_squared_mod_120_l75_75296


namespace integer_not_in_range_of_f_l75_75068

noncomputable def f (x : ℝ) : ℤ :=
  if x > -1 then ⌈1 / (x + 1)⌉ else ⌊1 / (x + 1)⌋

theorem integer_not_in_range_of_f :
  ¬ ∃ x : ℝ, x ≠ -1 ∧ f x = 0 :=
by
  sorry

end integer_not_in_range_of_f_l75_75068
