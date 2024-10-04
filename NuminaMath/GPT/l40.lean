import Mathlib
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.LinearEquiv
import Mathlib.Algebra.Prime
import Mathlib.Algebra.Trigonometry.Identities
import Mathlib.Analysis.Geometry.Euclidean.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Clique
import Mathlib.Combinatorics.Combination
import Mathlib.Combinatorics.CombinatorialStructures
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Combinations
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.Inference
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Ring
import Mathlib.Topology.Algebra.Order
import data.real.basic

namespace digits_difference_l40_40492

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end digits_difference_l40_40492


namespace different_lock_settings_l40_40713

theorem different_lock_settings:
  let digits := finset.range 10 in
  finset.card { (d1, d2, d3) ∈ digits × digits × digits | d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 } = 720 :=
by sorry

end different_lock_settings_l40_40713


namespace find_number_l40_40005

variables (n : ℝ)

-- Condition: a certain number divided by 14.5 equals 173.
def condition_1 (n : ℝ) : Prop := n / 14.5 = 173

-- Condition: 29.94 ÷ 1.45 = 17.3.
def condition_2 : Prop := 29.94 / 1.45 = 17.3

-- Theorem: Prove that the number is 2508.5 given the conditions.
theorem find_number (h1 : condition_1 n) (h2 : condition_2) : n = 2508.5 :=
by 
  sorry

end find_number_l40_40005


namespace pair_with_15_is_47_l40_40133

theorem pair_with_15_is_47 (numbers : Set ℕ) (k : ℕ) 
  (h : numbers = {49, 29, 9, 40, 22, 15, 53, 33, 13, 47}) 
  (pair_sum_eq_k : ∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → (a, b) ≠ (15, 15) → a + b = k) : 
  ∃ (k : ℕ), 15 + 47 = k := 
sorry

end pair_with_15_is_47_l40_40133


namespace six_digit_numbers_with_zero_count_l40_40908

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40908


namespace smallest_integer_with_eight_factors_l40_40159

theorem smallest_integer_with_eight_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (card (factors n) == 8) ∧ (∀ m : ℕ, (0 < m) ∧ (card (factors m) == 8) → n ≤ m) :=
sorry

end smallest_integer_with_eight_factors_l40_40159


namespace six_digit_numbers_with_zero_l40_40876

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40876


namespace total_wire_length_l40_40119

theorem total_wire_length : 
  ∀ (d : ℝ) (h₁ h₂ : ℝ), d = 20 → h₁ = 12 → h₂ = 20 →
  let horizontal_distance := d / 2 in
  let wire_between_tops := sqrt (d^2 + (h₂ - h₁)^2) in
  let wire_12_to_midpoint := sqrt (horizontal_distance^2 + h₁^2) in
  let wire_20_to_midpoint := sqrt (horizontal_distance^2 + h₂^2) in
  wire_between_tops + wire_12_to_midpoint + wire_20_to_midpoint = sqrt 464 + sqrt 244 + sqrt 500 :=
by 
  intros d h₁ h₂ hd h₁_12 h₂_20
  simp [hd, h₁_12, h₂_20, show 20 / 2 = 10 by norm_num, show 20 - 12 = 8 by norm_num]
  sorry

end total_wire_length_l40_40119


namespace knitting_time_is_correct_l40_40430

-- Definitions of the conditions
def time_per_hat : ℕ := 2
def time_per_scarf : ℕ := 3
def time_per_mitten : ℕ := 1
def time_per_sock : ℕ := 3 / 2 -- fractional time in hours
def time_per_sweater : ℕ := 6
def number_of_grandchildren : ℕ := 3

-- Compute total time for one complete outfit
def time_per_outfit : ℕ := time_per_hat + time_per_scarf + (time_per_mitten * 2) + (time_per_sock * 2) + time_per_sweater

-- Compute total time for all outfits
def total_knitting_time : ℕ := number_of_grandchildren * time_per_outfit

-- Prove that total knitting time is 48 hours
theorem knitting_time_is_correct : total_knitting_time = 48 := by
  unfold total_knitting_time time_per_outfit
  norm_num
  sorry

end knitting_time_is_correct_l40_40430


namespace find_A_d_minus_B_d_l40_40479

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end find_A_d_minus_B_d_l40_40479


namespace all_primes_appear_in_sequence_l40_40669

-- Defining the sequence p
def p : ℕ → ℕ
| 0 => 0 -- Not used as the sequence starts from p_1
| 1 => 2
| (n + 1) => Nat.find (λ k => k.prime ∧ (n * ∏ i in Finset.range n, p (i + 1) ^ Nat.factorial (i + 1) + 1) % k = 0)

-- Theorem statement: All primes appear in the sequence
theorem all_primes_appear_in_sequence : ∀ p', p'.prime → ∃ n, p n = p' :=
sorry

end all_primes_appear_in_sequence_l40_40669


namespace count_digit_seven_is_correct_l40_40123

-- We define a lean function that counts the occurrences of digit 7 in the range [1, 999].
def count_digit_seven_in_range (start end_ : Nat) (d : Nat) : Nat :=
  (List.range' start (end_ + 1 - start)).map (λ n =>
    (if (n / 100) = d then 1 else 0) + 
    (if (n / 10 % 10) = d then 1 else 0) + 
    (if (n % 10) = d then 1 else 0) 
  ).sum

-- We then assert the proof statement that the count of the digit 7 in the range [1, 999] is 280.
theorem count_digit_seven_is_correct : count_digit_seven_in_range 1 999 7 = 280 :=
  sorry

end count_digit_seven_is_correct_l40_40123


namespace smallest_prime_divisor_of_sum_l40_40625

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end smallest_prime_divisor_of_sum_l40_40625


namespace porter_previous_painting_price_l40_40086

-- definitions from the conditions
def most_recent_sale : ℕ := 44000

-- definitions for the problem statement
def sale_equation (P : ℕ) : Prop :=
  most_recent_sale = 5 * P - 1000

theorem porter_previous_painting_price (P : ℕ) (h : sale_equation P) : P = 9000 :=
by {
  sorry
}

end porter_previous_painting_price_l40_40086


namespace average_hit_targets_value_average_hit_targets_ge_half_l40_40267

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end average_hit_targets_value_average_hit_targets_ge_half_l40_40267


namespace six_digit_numbers_with_zero_l40_40997

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l40_40997


namespace train_passes_man_in_50_seconds_l40_40220

-- Definitions of conditions
def train_speed_kmh : ℝ := 54
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
def length_of_platform_m : ℝ := 300.024
def time_train_passes_platform_s : ℝ := 50
def length_of_train_m : ℝ := train_speed_ms * time_train_passes_platform_s

-- Statement of the problem
theorem train_passes_man_in_50_seconds :
  train_speed_kmh = 54 →
  train_speed_ms = 15 →
  length_of_platform_m = 300.024 →
  time_train_passes_platform_s = 50 →
  length_of_train_m = 750 →
  length_of_train_m / train_speed_ms = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end train_passes_man_in_50_seconds_l40_40220


namespace zeros_in_decimal_representation_l40_40657

theorem zeros_in_decimal_representation :
  let d := (7 : ℚ) / 8000 in
  ∃ n : ℕ, d = 875 / (10 ^ 6) ∧ n = 3 :=
by
  let d := (7 : ℚ) / 8000
  use 3
  constructor
  . norm_num [d]
  . rfl

end zeros_in_decimal_representation_l40_40657


namespace product_of_juicy_is_juicy_l40_40090

def juicy (n : ℕ) : Prop :=
  ∃ (l : List ℕ), (∀ (d ∈ l), 1 ≤ d ∧ d ≤ n) ∧ List.sum (l.map (λ d => 1/d)) = 1

theorem product_of_juicy_is_juicy (j1 j2 : ℕ) (hj1 : juicy j1) (hj2 : juicy j2) : juicy (j1 * j2) :=
by
  sorry

end product_of_juicy_is_juicy_l40_40090


namespace int_count_div_factorial_condition_l40_40795

theorem int_count_div_factorial_condition :
  (finset.range 60).filter (λ n : ℕ, n > 0 ∧ (nat.factorial (n^2+2)) % (nat.factorial n)^(n+2) = 0).card = 5 :=
begin
  sorry
end

end int_count_div_factorial_condition_l40_40795


namespace triangle_inequality_l40_40058

variables {P Q A B C : Point}
variables {a b c PA QA PB QB PC QC : ℝ}

-- Given conditions on the side lengths and distances
axiom ABC_triangle_sides : a = distance B C ∧ b = distance C A ∧ c = distance A B
axiom P_Q_in_plane : ∃ P Q : Point, True  -- P and Q are any points in the plane

-- Distances from points P and Q to vertices A, B, and C
axiom distances_P_Q : PA = distance P A ∧ QA = distance Q A ∧ 
                      PB = distance P B ∧ QB = distance Q B ∧ 
                      PC = distance P C ∧ QC = distance Q C

-- Statement of the inequality to be proven
theorem triangle_inequality :
  a * PA * QA + b * PB * QB + c * PC * QC ≥ a * b * c :=
sorry

end triangle_inequality_l40_40058


namespace zeros_between_decimal_and_first_nonzero_digit_l40_40652

theorem zeros_between_decimal_and_first_nonzero_digit : 
  let frac : ℚ := 7 / 8000 in
  let exp : ℤ := 6 - nat_digits (875 : ℕ) in
  exp = 3 :=
by sorry

end zeros_between_decimal_and_first_nonzero_digit_l40_40652


namespace remove_two_vertices_eliminate_triangles_l40_40409

variables {V : Type} [Fintype V] (G : SimpleGraph V)

noncomputable def has_no_5_clique (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 5 → ¬ s.pairwise (G.adj)

noncomputable def every_pair_of_triangles_shares_vertex (G : SimpleGraph V) : Prop :=
  ∀ (T₁ T₂ : Finset V), T₁.card = 3 → T₂.card = 3 → G.is_triangle T₁ → G.is_triangle T₂ → (T₁ ∩ T₂).card ≥ 1

theorem remove_two_vertices_eliminate_triangles :
  (has_no_5_clique G) →
  (every_pair_of_triangles_shares_vertex G) →
  ∃ (X : Finset V), X.card = 2 ∧ ∀ (H : SimpleGraph V), (H = G.delete_vertices X) → ∀ (T : Finset V), T.card = 3 → ¬ H.is_triangle T :=
begin
  intros h1 h2,
  sorry
end

end remove_two_vertices_eliminate_triangles_l40_40409


namespace six_digit_numbers_with_zero_l40_40870

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40870


namespace total_frogs_in_ponds_l40_40565

def pondA_frogs := 32
def pondB_frogs := pondA_frogs / 2

theorem total_frogs_in_ponds : pondA_frogs + pondB_frogs = 48 := by
  sorry

end total_frogs_in_ponds_l40_40565


namespace coffee_break_l40_40575

variable {n : ℕ}
variables {participants : Finset ℕ} (hparticipants : participants.card = 14)
variables (left : Finset ℕ)
variables (stayed : Finset ℕ) (hstayed : stayed.card = 14 - 2 * left.card)

-- The overall proof problem statement:
theorem coffee_break (h : ∀ x ∈ stayed, ∃! y, (y ∈ left ∧ adjacent x y participants)) :
  left.card = 6 ∨ left.card = 8 ∨ left.card = 10 ∨ left.card = 12 := 
sorry

end coffee_break_l40_40575


namespace x_values_l40_40821

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 1 / 3) :
  x = 3 ∨ x = 4 :=
by
  sorry

end x_values_l40_40821


namespace product_of_complex_roots_l40_40331

open Complex

noncomputable def product_of_values (z : ℂ) : Prop :=
  z^2 + 2 * z = conj z ∧ z ≠ conj z

theorem product_of_complex_roots : 
  ∀ (z : ℂ), product_of_values z → {|z|^2, 3} :=
by
  intro z h
  sorry

end product_of_complex_roots_l40_40331


namespace greatest_m_for_factorial_division_l40_40155

theorem greatest_m_for_factorial_division :
  (∃ (m : ℕ), (20! / 10 ^ m).nat_cast = 4) ∧ m = 4 :=
begin
  sorry
end

end greatest_m_for_factorial_division_l40_40155


namespace leading_coefficient_of_g_is_one_div_sixteen_l40_40066

noncomputable def f (α x : ℝ) : ℝ := (x / 2)^α / (x - 1)

noncomputable def g (α : ℝ) : ℝ := 
  let deriv4 := (deriv^[4] (λ x => f α x))
  deriv4 2

theorem leading_coefficient_of_g_is_one_div_sixteen :
  leading_coeff (polynomial.of_fn (λ α => g α)) = 1 / 16 :=
sorry

end leading_coefficient_of_g_is_one_div_sixteen_l40_40066


namespace six_digit_numbers_with_zero_l40_40884

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40884


namespace seq_sin_product_l40_40137

variable {n : ℕ}
variable (a : ℕ → ℝ)

-- Define the sequence a_n
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = Real.pi / 6 ∧ ∀ n, a (n + 1) = Real.arctan (Real.sec (a n))

-- Define the product of sine values condition
def sin_product (a : ℕ → ℝ) (m : ℕ) : Prop :=
  (List.prod (List.map (λ k, Real.sin (a k)) (List.range (m + 1)))) = 1 / 100

theorem seq_sin_product:
  seq a → sin_product a 3333 :=
by
  -- Seq definition: a_1 = π/6 and a_(n+1) = arctan(sec(a_n))
  intro h1
  -- Product of sin values up to m = 3333 yields 1/100
  have h2 : (List.prod (List.map (λ k, Real.sin (a k)) (List.range (3334)))) = 1 / 100 := sorry
  exact h2

end seq_sin_product_l40_40137


namespace area_of_circle_l40_40530

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l40_40530


namespace circle_area_l40_40525

theorem circle_area (θ : ℝ) : 
  let r := 3 * Real.cos θ - 4 * Real.sin θ in
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let center := (3 / 2 : ℝ, -2 : ℝ) in
  let radius := 5 / 2 in
  (x - center.1)^2 + (y - center.2)^2 = radius^2 →
  ∃ A : ℝ, A = (radius^2) * Real.pi ∧ A = 25 * Real.pi / 4 := 
by sorry

end circle_area_l40_40525


namespace six_digit_numbers_with_zero_l40_40930

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40930


namespace six_digit_numbers_with_at_least_one_zero_l40_40965

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40965


namespace Cathy_and_Chris_worked_months_l40_40742

theorem Cathy_and_Chris_worked_months (Cathy_hours : ℕ) (weekly_hours : ℕ) (weeks_in_month : ℕ) (extra_weekly_hours : ℕ) (weeks_for_Chris_sick : ℕ) : 
  Cathy_hours = 180 →
  weekly_hours = 20 →
  weeks_in_month = 4 →
  extra_weekly_hours = weekly_hours →
  weeks_for_Chris_sick = 1 →
  (Cathy_hours - extra_weekly_hours * weeks_for_Chris_sick) / weekly_hours / weeks_in_month = (2 : ℕ) :=
by
  intros hCathy_hours hweekly_hours hweeks_in_month hextra_weekly_hours hweeks_for_Chris_sick
  rw [hCathy_hours, hweekly_hours, hweeks_in_month, hextra_weekly_hours, hweeks_for_Chris_sick]
  norm_num
  sorry

end Cathy_and_Chris_worked_months_l40_40742


namespace num_zeros_in_decimal_rep_l40_40648

theorem num_zeros_in_decimal_rep (a b : ℕ) (h₁ : a = 7) (h₂ : b = 8000) :
  num_zeros_between_decimal_and_first_nonzero (a / b : ℚ) = 2 :=
sorry

end num_zeros_in_decimal_rep_l40_40648


namespace knitting_time_total_l40_40433

-- Define knitting times for each item
def hat_knitting_time : ℕ := 2
def scarf_knitting_time : ℕ := 3
def mitten_knitting_time : ℕ := 1
def sock_knitting_time : ℕ := 3 / 2
def sweater_knitting_time : ℕ := 6

-- Define the number of grandchildren
def grandchildren_count : ℕ := 3

-- Total knitting time calculation
theorem knitting_time_total : 
  hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time = 16 ∧ 
  (hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time) * grandchildren_count = 48 :=
by 
  sorry

end knitting_time_total_l40_40433


namespace largest_smallest_adjacent_l40_40227

variable {A B C D E : Type} [Pentagon A B C D E]

-- The theorem to be proved: the largest angle is adjacent to the smallest angle in the equilateral convex pentagon
theorem largest_smallest_adjacent (h1 : equilateral A B C D E) (h2 : all_angles_distinct A B C D E):
  ∃ (X Y : Type) [Adjacent X Y], 
    (largest_angle A B C D E X) ∧ (smallest_angle A B C D E Y) :=
sorry

end largest_smallest_adjacent_l40_40227


namespace smallest_delicious_l40_40467

-- Definitions and conditions
def delicious (A : ℤ) : Prop :=
  ∃ B : ℤ, (∑ k in finset.range (B - A + 1), A + k) = 2024

-- Proof statement
theorem smallest_delicious : ∀ A : ℤ, delicious A → A = -2023 :=
by sorry

end smallest_delicious_l40_40467


namespace combination_permutation_42_l40_40363

-- Defining the terms for combination and permutation
def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) := n.factorial / (n - k).factorial

theorem combination_permutation_42 (n : ℕ) (h : combination n 2 * permutation 2 2 = 42) :
    n.factorial / (3.factorial * (n - 3).factorial) = 35 := by
  sorry

end combination_permutation_42_l40_40363


namespace prod_of_roots_eq_zero_l40_40756

theorem prod_of_roots_eq_zero :
  let P (x : ℂ) := ∏ k in (finset.range 15).map (λ n, n+1), (x - exp (2 * π * complex.I * (k:ℂ) / 17))
  ∧ let Q (x : ℂ) := ∏ j in (finset.range 12).map (λ n, n+1), (x - exp (2 * π * complex.I * (j:ℂ) / 13))
  in (∏ k in (finset.range 15).map (λ n, n+1), ∏ j in (finset.range 12).map (λ n, n+1), 
    (exp (2 * π * j * complex.I / 13) - exp (2 * π * k * complex.I / 17))) = 0 :=
by
  intro P Q
  sorry

end prod_of_roots_eq_zero_l40_40756


namespace abs_neg_two_l40_40505

def abs (x : ℤ) : ℤ :=
  if x ≥ 0 then x else -x

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end abs_neg_two_l40_40505


namespace maximize_sum_of_sides_l40_40014

theorem maximize_sum_of_sides (a b c : ℝ) (A B C : ℝ) 
  (h_b : b = 2) (h_B : B = (Real.pi / 3)) (h_law_of_cosines : b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) :
  a + c ≤ 4 :=
by
  sorry

end maximize_sum_of_sides_l40_40014


namespace ratio_of_sums_l40_40364

theorem ratio_of_sums (a b c d : ℚ) (h1 : b / a = 3) (h2 : d / b = 4) (h3 : c = (a + b) / 2) :
  (a + b + c) / (b + c + d) = 8 / 17 :=
by
  sorry

end ratio_of_sums_l40_40364


namespace distinct_values_ab_plus_c_l40_40114

theorem distinct_values_ab_plus_c 
  (a b c : ℕ) 
  (ha : a ∈ {1, 2, 3, 4, 5})
  (hb : b ∈ {1, 2, 3, 4, 5})
  (hc : c ∈ {1, 2, 3, 4, 5})
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_odd : (a * b + c) % 2 = 1) : 
  ∃! n, n = 9 := 
sorry

end distinct_values_ab_plus_c_l40_40114


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40921

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40921


namespace average_marks_b_c_d_e_l40_40117

-- Definitions for the conditions
variables (a b c d e : ℕ)
variable average : ℕ
variable total_marks : ℕ

-- Given conditions as hypotheses
variables (h1 : (a + b + c) / 3 = 48)
variables (h2 : (a + b + c + d) / 4 = 47)
variables (h3 : e = d + 3)
variables (h4 : a = 43)

-- Statement to prove
theorem average_marks_b_c_d_e : (b + c + d + e) / 4 = 48 :=
sorry

end average_marks_b_c_d_e_l40_40117


namespace six_digit_numbers_with_at_least_one_zero_l40_40855

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40855


namespace circle_C_exists_or_not_l40_40246

noncomputable def circle_radius (a : ℝ) := 2 * abs a

def tangent_to_y_axis_condition (C : ℕ → ℝ → Prop) := 
  ∀ a > 0, C (2 * a) a

def intersection_condition (C : ℕ → ℝ → Prop) := 
  ∃ a > 0, circle_radius(a) = 2 * abs(√3)

def passes_through_points_condition (C : ℕ → ℝ → Prop) := 
  C 4 1 ∧ C 2 3

def tangent_line_condition (C : ℕ → ℝ → Prop) := 
  ∀ a b : ℝ, line_distance (2a, a) (x-2y-1=0) = circle_radius(a) + 1 

def externally_tangent_to_circle_condition (C : ℕ → ℝ → Prop) := 
  ∀ a > 0, (2a - 2)^2 + (a - 2)^2 = 1 + circle_radius(a)

def exists_circle (C : ℕ → ℝ → Prop) :=
  ∃ x y, y = (1/2) * x ∧ X x y

axiom line_distance {x1 y1 b : ℝ} := 
  (x1 + b)/√5 = (y1)(2a) - (a1))

theorem circle_C_exists_or_not :
  (exists_circle (λ x y, (x-2)^2 + (y-1)^2 = 4) ∨ ¬exists_circle) :=
by sorry

end circle_C_exists_or_not_l40_40246


namespace avg_distance_is_600_l40_40438

-- Assuming Mickey runs half as many times as Johnny
def num_laps_johnny := 4
def num_laps_mickey := num_laps_johnny / 2
def lap_distance := 200

-- Calculating distances
def distance_johnny := num_laps_johnny * lap_distance
def distance_mickey := num_laps_mickey * lap_distance

-- Total distance run by both Mickey and Johnny
def total_distance := distance_johnny + distance_mickey

-- Average distance run by Johnny and Mickey
def avg_distance := total_distance / 2

-- Prove that the average distance run by Johnny and Mickey is 600 meters
theorem avg_distance_is_600 : avg_distance = 600 :=
by
  sorry

end avg_distance_is_600_l40_40438


namespace find_y_six_l40_40703

theorem find_y_six (y : ℝ) (hy : 0 < y)
  (h : real.cbrt (2 - y^3) + real.cbrt (2 + y^3) = 2) : y^6 = 116 / 27 :=
sorry

end find_y_six_l40_40703


namespace part1_solution_set_part2_range_of_m_l40_40779

def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part1_solution_set (x : ℝ) : (f x 3 >= 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2_range_of_m (m : ℝ) (x : ℝ) : 
 (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
by sorry

end part1_solution_set_part2_range_of_m_l40_40779


namespace coffee_break_l40_40591

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end coffee_break_l40_40591


namespace total_frogs_in_both_ponds_l40_40568

noncomputable def total_frogs_combined : Nat :=
let frogs_in_pond_a : Nat := 32
let frogs_in_pond_b : Nat := frogs_in_pond_a / 2
frogs_in_pond_a + frogs_in_pond_b

theorem total_frogs_in_both_ponds :
  total_frogs_combined = 48 := by
  sorry

end total_frogs_in_both_ponds_l40_40568


namespace maximize_angle_BVM_l40_40406

variables {M B V G : Point}
variable {k : Circle}

-- Assume M is the center of circle k
axiom center_of_circle : M = k.center

-- Assume B is a point inside the circle k
axiom B_inside_k : dist M B < k.radius

-- Assume V is a point on the circle k
axiom V_on_k : dist M V = k.radius

-- Assume VG and MB are the relevant lines being referred to in the problem
axiom VG_line : Line (V, G)
axiom MB_line : Line (M, B)

-- Prove that the angle BVM is maximal when VG is perpendicular to MB
theorem maximize_angle_BVM (angle_BVM : Real) :
  angle_BVM = max ?x | (angle x B M) -> is_perpendicular VG_line MB_line :=
sorry

end maximize_angle_BVM_l40_40406


namespace skateboard_travel_and_graffiti_area_l40_40708

theorem skateboard_travel_and_graffiti_area:
  let a₁ := 8 -- starting distance in feet
  let d := 10 -- common difference in feet
  let g₁ := 2 -- initial visible area in square feet
  let r := 2 -- doubling ratio for graffiti area
  ∃ total_distance total_graffiti_area,
  (total_distance = 10 * (a₁ + (a₁ + 19 * d))) /\ -- sum of arithmetic series formula
  (total_graffiti_area = g₁ * (r^20 - 1) / (r - 1)) -- sum of geometric series formula
  :=
  ∃ total_distance total_graffiti_area,
  (total_distance = 2060) /\
  (total_graffiti_area = 2^21 - 2) :=
begin
  sorry
end

end skateboard_travel_and_graffiti_area_l40_40708


namespace largest_prime_factor_18_to_4_plus_12_to_5_minus_6_to_6_l40_40240

theorem largest_prime_factor_18_to_4_plus_12_to_5_minus_6_to_6 :
  ∃ p : ℕ, nat.prime p ∧ p ≥ 2 ∧ (∀ q : ℕ, nat.prime q ∧ q ∣ (18^4 + 12^5 - 6^6) → q ≤ p) ∧ p = 11 :=
sorry

end largest_prime_factor_18_to_4_plus_12_to_5_minus_6_to_6_l40_40240


namespace codys_grandmother_age_l40_40748

theorem codys_grandmother_age (cody_age : ℕ) (grandmother_factor : ℕ) (h1 : cody_age = 14) (h2 : grandmother_factor = 6) :
  grandmother_factor * cody_age = 84 :=
by
  sorry

end codys_grandmother_age_l40_40748


namespace total_savings_l40_40076

def weekly_savings : ℕ := 15
def weeks_per_cycle : ℕ := 60
def number_of_cycles : ℕ := 5

theorem total_savings :
  (weekly_savings * weeks_per_cycle) * number_of_cycles = 4500 := 
sorry

end total_savings_l40_40076


namespace marbles_percentage_l40_40190

def solid_color_other_than_yellow (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) : ℚ :=
  solid_color_percent - solid_yellow_percent

theorem marbles_percentage (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) :
  solid_color_percent = 90 / 100 →
  solid_yellow_percent = 5 / 100 →
  solid_color_other_than_yellow total_marbles solid_color_percent solid_yellow_percent = 85 / 100 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num
  sorry

end marbles_percentage_l40_40190


namespace find_num_alligators_l40_40039

-- We define the conditions as given in the problem
def journey_to_delta_hours : ℕ := 4
def extra_hours : ℕ := 2
def combined_time_alligators_walked : ℕ := 46

-- We define the hypothesis in terms of Lean variables
def num_alligators_traveled_with_Paul (A : ℕ) : Prop :=
  (journey_to_delta_hours + (journey_to_delta_hours + extra_hours) * A) = combined_time_alligators_walked

-- Now the theorem statement where we prove that the number of alligators (A) is 7
theorem find_num_alligators :
  ∃ A : ℕ, num_alligators_traveled_with_Paul A ∧ A = 7 :=
by
  existsi 7
  unfold num_alligators_traveled_with_Paul
  simp
  sorry -- this is where the actual proof would go

end find_num_alligators_l40_40039


namespace prod_eq_one_l40_40753

noncomputable def P (x : ℂ) : ℂ :=
  ∏ k in finset.range 15, (x - complex.exp (2 * real.pi * complex.I * k / 17))

noncomputable def Q (x : ℂ) : ℂ :=
  ∏ j in finset.range 12, (x - complex.exp (2 * real.pi * complex.I * j / 13))

theorem prod_eq_one :
  (∏ k in finset.range 15, ∏ j in finset.range 12, (complex.exp (2 * real.pi * complex.I * j / 13) - complex.exp (2 * real.pi * complex.I * k / 17))) = 1 := 
by
  sorry

end prod_eq_one_l40_40753


namespace average_distance_is_600_l40_40443

-- Definitions based on the given conditions
def distance_around_block := 200
def johnny_rounds := 4
def mickey_rounds := johnny_rounds / 2

-- The calculated distances
def johnny_distance := johnny_rounds * distance_around_block
def mickey_distance := mickey_rounds * distance_around_block

-- The average distance computation
def average_distance := (johnny_distance + mickey_distance) / 2

-- The theorem to prove that the average distance is 600 meters
theorem average_distance_is_600 : average_distance = 600 := by sorry

end average_distance_is_600_l40_40443


namespace sum_interior_ninth_row_l40_40035

-- Define Pascal's Triangle and the specific conditions
def pascal_sum (n : ℕ) : ℕ := 2^(n - 1)

def sum_interior_numbers (n : ℕ) : ℕ := pascal_sum n - 2

theorem sum_interior_ninth_row :
  sum_interior_numbers 9 = 254 := 
by {
  sorry
}

end sum_interior_ninth_row_l40_40035


namespace smallest_prime_divisor_of_n_is_2_l40_40635

def a := 3^19
def b := 11^13
def n := a + b

theorem smallest_prime_divisor_of_n_is_2 : nat.prime_divisors n = {2} :=
by
  -- The proof is not needed as per instructions.
  -- We'll add a placeholder sorry to complete the statement.
  sorry

end smallest_prime_divisor_of_n_is_2_l40_40635


namespace part1_part2_l40_40850

variable (U : Set ℝ)

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m}

theorem part1 (m : ℝ) (hm : m = 3) :
  (A U ∪ B m = {x | -2 ≤ x ∧ x < 6}) ∧ (A U ∩ B m = ∅) :=
by 
  sorry

theorem part2 (m : ℝ) :
  (A U ∩ B m = ∅) → (m ≤ 1 ∨ 3 ≤ m) :=
by 
  sorry

end part1_part2_l40_40850


namespace six_digit_numbers_with_at_least_one_zero_l40_40966

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40966


namespace right_triangle_area_l40_40771

-- Define points A, B, and C
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := -4, y := 8 }
def B : Point := { x := -9, y := 3 }
def C : Point := { x := 0,  y := 12 }

-- Define the function to calculate the area of the triangle formed by three points
def triangle_area (A B C : Point) : ℚ :=
  (1 / 2 : ℚ) * (abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)).toRat)

-- The theorem to prove
theorem right_triangle_area : triangle_area A B C = 18 :=
  sorry

end right_triangle_area_l40_40771


namespace initial_fee_calculation_l40_40041

/-- Assume the following conditions:
      - additional_charge: The additional charge per 2/5 mile (in dollars)
      - distance_traveled: The distance traveled in the trip (in miles)
      - total_trip_charge: The total charge for the trip (in dollars)
      
    Prove that the initial fee is $2.25.
-/
theorem initial_fee_calculation
    (additional_charge : ℝ := 0.3)
    (segment_length : ℝ := 2/5)
    (distance_traveled : ℝ := 3.6)
    (total_trip_charge : ℝ := 4.95) :
    (let number_of_segments := distance_traveled / segment_length in
     let total_distance_charge := number_of_segments * additional_charge in
     let initial_fee := total_trip_charge - total_distance_charge in
     initial_fee = 2.25) :=
by
  sorry

end initial_fee_calculation_l40_40041


namespace correct_tan_values_l40_40802

noncomputable def verify_tan_sum (a b : ℝ) (θ : ℝ) : Prop :=
  sin (2 * θ) = a ∧ cos (2 * θ) = b ∧ 0 < θ ∧ θ < (Real.pi / 4) →
  (tan (θ + (Real.pi / 4)) = (1 + a) / b ∨ 
   tan (θ + (Real.pi / 4)) = b / (1 - a) ∨  
   tan (θ + (Real.pi / 4)) = (a - b + 1) / (a + b - 1))

theorem correct_tan_values (a b θ : ℝ) : verify_tan_sum a b θ :=
by
  sorry

end correct_tan_values_l40_40802


namespace f_is_odd_l40_40397

def f (x : ℝ) : ℝ := 1 / x

theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f x = -f (-x) :=
by 
  intros x hx
  unfold f 
  rw [neg_div]
  sorry

end f_is_odd_l40_40397


namespace total_profit_is_42000_l40_40663

noncomputable def total_profit (I_B T_B : ℝ) :=
  let I_A := 3 * I_B
  let T_A := 2 * T_B
  let profit_B := I_B * T_B
  let profit_A := I_A * T_A
  profit_A + profit_B

theorem total_profit_is_42000
  (I_B T_B : ℝ)
  (h1 : I_A = 3 * I_B)
  (h2 : T_A = 2 * T_B)
  (h3 : I_B * T_B = 6000) :
  total_profit I_B T_B = 42000 := by
  sorry

end total_profit_is_42000_l40_40663


namespace find_f_log_log_3_value_l40_40183

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x - b * Real.logb 3 (Real.sqrt (x*x + 1) - x) + 1

theorem find_f_log_log_3_value
  (a b : ℝ)
  (h1 : f a b (Real.log 10 / Real.log 3) = 5) :
  f a b (-Real.log 10 / Real.log 3) = -3 :=
  sorry

end find_f_log_log_3_value_l40_40183


namespace simplify_expression_l40_40471

variable (a : Real)

theorem simplify_expression : (-2 * a) * a - (-2 * a)^2 = -6 * a^2 := by
  sorry

end simplify_expression_l40_40471


namespace area_circle_l40_40534

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l40_40534


namespace circle_area_polar_eq_l40_40540

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l40_40540


namespace abs_neg_two_l40_40506

def abs (x : ℤ) : ℤ :=
  if x ≥ 0 then x else -x

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end abs_neg_two_l40_40506


namespace max_area_of_n_points_l40_40185

theorem max_area_of_n_points (n : ℕ) (h : n > 3) (P : Fin n → ℝ × ℝ)
  (triangle_area : ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ i → 
    triangle_area (P i) (P j) (P k) ≤ 1) :
  ∃ (A B C : ℝ × ℝ), 
    (∀ (p : Fin n), inside_triangle (P p) A B C) ∧ 
    triangle_area A B C ≤ 4 := 
sorry

-- Definitions for triangle_area and inside_triangle would need to be provided

end max_area_of_n_points_l40_40185


namespace rationalize_fraction_sum_l40_40094

theorem rationalize_fraction_sum :
  ∃ (A B C D : ℤ),
    D > 0 ∧
    ¬ ∃ p : ℕ, p.prime ∧ p^2 ∣ B ∧
    Int.gcd A C D = 1 ∧
    A * 8 + C = 21 - 7 * 3 ∧
    A + B + C + D = 23 :=
sorry

end rationalize_fraction_sum_l40_40094


namespace index_of_150th_negative_term_l40_40769

open Real

def sequence_b (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), cos k

theorem index_of_150th_negative_term :
  ∃ n, sequence_b n < 0 ∧ (∃ m, m = 150 ∧ n = floor (2 * π * m)) := sorry

end index_of_150th_negative_term_l40_40769


namespace probability_YW_correct_l40_40608

noncomputable def probability_YW_greater_than_six_sqrt_three (XY YZ XZ YW : ℝ) : ℝ :=
  if H : XY = 12 ∧ YZ = 6 ∧ XZ = 6 * Real.sqrt 3 then 
    if YW > 6 * Real.sqrt 3 then (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3
    else 0
  else 0

theorem probability_YW_correct : probability_YW_greater_than_six_sqrt_three 12 6 (6 * Real.sqrt 3) (6 * Real.sqrt 3) = (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3 :=
sorry

end probability_YW_correct_l40_40608


namespace monthly_price_reduction_rate_l40_40025

-- Let's define the given conditions
def initial_price_March : ℝ := 23000
def price_in_May : ℝ := 16000

-- Define the monthly average price reduction rate
variable (x : ℝ)

-- Define the statement to be proven
theorem monthly_price_reduction_rate :
  23 * (1 - x) ^ 2 = 16 :=
sorry

end monthly_price_reduction_rate_l40_40025


namespace sequence_general_term_l40_40428

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 1 else 
  let a_n_minus_2 := sequence (n - 2) in
  let a_n_minus_1 := sequence (n - 1) in
  let left := Real.sqrt (Real.ofNat (sequence n * a_n_minus_2)) in
  let right := Real.sqrt (Real.ofNat (a_n_minus_1 * a_n_minus_2)) + 2 * Real.ofNat a_n_minus_1 in
  Nat.floor ((left - right)^4)

theorem sequence_general_term (n : ℕ) (hn : n ≥ 2) :
  sequence n = (∏ k in Finset.range (n + 1), 2 ^ k - 1) ^ 2 :=
sorry

end sequence_general_term_l40_40428


namespace circle_center_and_radius_l40_40311

theorem circle_center_and_radius :
  ∃ (h k r : ℝ), (λ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y - 4 = 0) = (λ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2) :=
begin
  use [2, -1, 3],
  sorry
end

end circle_center_and_radius_l40_40311


namespace relationship_between_a_b_c_l40_40803

theorem relationship_between_a_b_c :
  let a := Real.log 0.7 / Real.log 1.4
  let b := 1.4 ^ 0.7
  let c := 0.7 ^ 1.4
  a < c ∧ c < b := by
  sorry

end relationship_between_a_b_c_l40_40803


namespace problem_part_1_problem_part_2_l40_40347

noncomputable def A : Set ℝ := { x | 2 ≤ 2^x ∧ 2^x ≤ 8 }
noncomputable def B : Set ℝ := { x | x > 2 }
noncomputable def C (a : ℝ) : Set ℝ := { x | 1 < x ∧ x < a }
def U : Set ℝ := Set.univ

theorem problem_part_1 :
  ((Set.univ \ B) ∪ A) = { x | x ≤ 3 } :=
sorry

theorem problem_part_2 (a : ℝ) (h : C a ⊆ A) :
  a ∈ Iic 3 :=
sorry

end problem_part_1_problem_part_2_l40_40347


namespace parabola_equation_and_k_value_l40_40810

-- Definitions and conditions
def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.2^2 = 2 * p * p.1}
def focus : ℝ × ℝ := (2, 0)
def directrix (p : ℝ) : set (ℝ × ℝ) := { b | b.1 = -p/2 }
def distance (a b : ℝ × ℝ) := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Point P on the parabola
def P : ℝ × ℝ := (4, m : ℝ)

-- Distance between P and the focus
#check distance P focus = 6

def line (k : ℝ) : set (ℝ × ℝ) := { l | l.2 = k * l.1 - 2 }
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def intersects (A B : ℝ × ℝ) := A ∩ B

-- Proving the equations and the value of k
theorem parabola_equation_and_k_value (m k : ℝ)
  (hP : P ∈ parabola p)
  (h_eq_dist : distance P focus = 6)
  (h_intersects : ∃ A B, intersects (parabola p) (line k) = ({A, B} : set (ℝ × ℝ)))
  (h_midpoint : (midpoint A B).1 = 2) :
  (parabola p = { p | p.2^2 = 8 * p.1}) ∧ (k = 2) :=
  sorry

end parabola_equation_and_k_value_l40_40810


namespace six_digit_numbers_with_at_least_one_zero_l40_40863

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40863


namespace problem_l40_40484

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end problem_l40_40484


namespace rate_percent_calculation_l40_40676

theorem rate_percent_calculation (SI P T : ℝ) (R : ℝ) : SI = 640 ∧ P = 4000 ∧ T = 2 → SI = P * R * T / 100 → R = 8 :=
by
  intros
  sorry

end rate_percent_calculation_l40_40676


namespace f_monotonic_intervals_f_extreme_values_l40_40340

def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Monotonicity intervals
theorem f_monotonic_intervals (x : ℝ) : 
  (x < -2 → deriv f x > 0) ∧ 
  (-2 < x ∧ x < 2 → deriv f x < 0) ∧ 
  (2 < x → deriv f x > 0) := 
sorry

-- Extreme values
theorem f_extreme_values :
  f (-2) = 16 ∧ f (2) = -16 :=
sorry

end f_monotonic_intervals_f_extreme_values_l40_40340


namespace trigonometric_inequality_l40_40462

theorem trigonometric_inequality (x : ℝ) : 
  -1/3 ≤ (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ∧
  (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ≤ 3 := 
sorry

end trigonometric_inequality_l40_40462


namespace triangle_areas_equal_l40_40717

variable {A B C D O : Type}

-- A trapezoid divided by its diagonals into four triangles

variable (trapezoid : Prop)
variable (diagonals_intersect : ∀ (A B C D O : Prop), trapezoid → (∃ O. O ∈ A ∩ C) ∧ (O ∈ B ∩ D))

-- declare the areas of triangles BOA and COD
variable {S_BOA S_COD : ℝ}

noncomputable def triangle_area_equal (BO CO DO AO: ℝ) (angle_BOA angle_COD: ℝ) : Prop :=
    S_BOA = (1/2) * BO * (AO/CO) * CO * sin angle_BOA ∧
    S_COD = (1/2) * CO * (DO/BO) * BO * sin angle_COD

theorem triangle_areas_equal 
    (h_eq_sin : ∀angle_BOA angle_COD, angle_BOA = angle_COD → sin angle_BOA = sin angle_COD) :
  trapezoid → 
  diagonals_intersect A B C D O →
  let k := AO / CO in
  let k' := DO / BO in
  triangle_area_equal BO CO DO AO angle_BOA angle_COD →
  S_BOA = S_COD :=
by 
  intros
  sorry

end triangle_areas_equal_l40_40717


namespace six_digit_numbers_with_zero_count_l40_40896

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40896


namespace abs_neg_two_l40_40500

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end abs_neg_two_l40_40500


namespace domain_of_tan_arcsin_xsq_l40_40255

noncomputable def domain_f (x : ℝ) : Prop :=
  x ≠ 1 ∧ x ≠ -1 ∧ -1 ≤ x ∧ x ≤ 1

theorem domain_of_tan_arcsin_xsq :
  ∀ x : ℝ, -1 < x ∧ x < 1 ↔ domain_f x := 
sorry

end domain_of_tan_arcsin_xsq_l40_40255


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40918

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40918


namespace wall_building_days_l40_40671

-- Define the initial conditions as Lean definitions
def persons1 : ℕ := 18
def length1 : ℕ := 140
def days1 : ℕ := 42

def persons2 : ℕ := 30
def length2 : ℕ := 100

-- To prove that under given conditions, 30 persons will take 18 days to build a 100-meter wall
theorem wall_building_days (persons1 : ℕ) (length1 : ℕ) (days1 : ℕ) (persons2 : ℕ) (length2 : ℕ) :
  persons1 = 18 →
  length1 = 140 →
  days1 = 42 →
  persons2 = 30 →
  length2 = 100 →
  days1 * persons1 * length2 / (length1 * persons2) = 18 := 
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  exact Decidable.mul_nonneg (by norm_num) (by norm_num)

end wall_building_days_l40_40671


namespace ball_hits_ground_time_l40_40124

-- Define the height function
def height (t : ℝ) : ℝ := -4.9 * t^2 + 4 * t + 10

-- State the problem: time when height of the ball is zero
theorem ball_hits_ground_time :
  ∃ t : ℝ, height t = 0 ∧ t > 0 ∧ t = 20 / 7 :=
by
  sorry

end ball_hits_ground_time_l40_40124


namespace six_digit_numbers_with_zero_l40_40982

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40982


namespace decompose_polynomial_to_quadratic_l40_40060

noncomputable def factorize_polynomial (t : ℝ) : Prop :=
  ∀ x : ℝ, (
    if |t| ≥ 2 then 
      (x^4 + t * x^2 + 1 = (x^2 + (t + sqrt (t^2 - 4)) / 2) * (x^2 + (t - sqrt (t^2 - 4)) / 2))
    else if t ≤ 2 then 
      (x^4 + t * x^2 + 1 = (x^2 + x * sqrt (-t + 2) + 1) * (x^2 - x * sqrt (-t + 2) + 1))
    else -- t ≤ -2
      (x^4 + t * x^2 + 1 = (x^2 + x * sqrt (-t - 2) - 1) * (x^2 - x * sqrt (-t - 2) - 1))
  )

theorem decompose_polynomial_to_quadratic (t : ℝ) : factorize_polynomial t :=
by sorry

end decompose_polynomial_to_quadratic_l40_40060


namespace coffee_break_participants_l40_40578

theorem coffee_break_participants (n : ℕ) (h1 : n = 14) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 6 ∧ n - 2 * k > 0 ∧ n - 2 * k < n)
    (h3 : ∀ m, m < n → (∃ k, m = n - 2 * k → m ∈ {6, 8, 10, 12})): 
    ∃ m, n - 2 * m ∈ {6, 8, 10, 12} := 
by
  use 6
  use 8
  use 10
  use 12
  sorry

end coffee_break_participants_l40_40578


namespace six_digit_numbers_with_zero_l40_40996

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l40_40996


namespace find_A_d_minus_B_d_l40_40489

-- Definitions for the proof problem
variables {d A B : ℕ}
variables {ad bd : ℤ} -- Representing A_d and B_d in ℤ for arithmetic operations

-- Conditions
constants (base_check : d > 7)
constants (digit_check : A < d ∧ B < d)
constants (encoded_check : d^1 * A + d^0 * B + d^2 * A + d^1 * A = 1 * d^2 + 7 * d^1 + 2 * d^0)

-- The theorem to prove
theorem find_A_d_minus_B_d : A - B = 5 :=
good sorry

end find_A_d_minus_B_d_l40_40489


namespace parallel_lines_in_triangle_l40_40812

theorem parallel_lines_in_triangle (A B C X Y L : Point) (h1 : Collinear A X B)
  (h2 : Collinear B Y C) (h3 : AX = BY) (h4 : Cyclic A X Y C)
  (h5 : AngleBisector B A C L) : Parallel X L B C :=
by
  sorry

end parallel_lines_in_triangle_l40_40812


namespace max_modulus_l40_40836

noncomputable def max_value (z1 z2 : ℂ) : ℝ :=
  complex.abs ((z1 + 1) ^ 2 * (z1 - 2))

theorem max_modulus (z1 z2 : ℂ) (h1 : complex.abs z2 = 4)
  (h2 : 4 * z1 ^ 2 - 2 * z1 * z2 + z2 ^ 2 = 0) : 
  max_value z1 z2 ≤ 6 * real.sqrt 6 :=
begin
  sorry
end

end max_modulus_l40_40836


namespace circle_area_l40_40543

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l40_40543


namespace train_pass_time_correct_l40_40715

def time_to_pass_man (length_of_train : ℝ) (speed_of_train_kmhr : ℝ) (speed_of_man_kmhr : ℝ) : ℝ :=
  let speed_of_train_ms := speed_of_train_kmhr * (5 / 18)
  let speed_of_man_ms := speed_of_man_kmhr * (5 / 18)
  let relative_speed_ms := speed_of_train_ms + speed_of_man_ms
  length_of_train / relative_speed_ms

theorem train_pass_time_correct :
  time_to_pass_man 250 60 10 ≈ 12.85 :=
by
  sorry

end train_pass_time_correct_l40_40715


namespace team_with_at_least_one_girl_l40_40108

noncomputable def choose (n m : ℕ) := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem team_with_at_least_one_girl (total_boys total_girls select : ℕ) (h_boys : total_boys = 5) (h_girls : total_girls = 5) (h_select : select = 3) :
  (choose (total_boys + total_girls) select) - (choose total_boys select) = 110 := 
by
  sorry

end team_with_at_least_one_girl_l40_40108


namespace chess_group_players_l40_40564

theorem chess_group_players : ∃ n : ℕ, n * (n - 1) / 2 = 105 ∧ n = 15 :=
by
  have h : ∃ n : ℕ, n * (n - 1) / 2 = 105 := sorry
  obtain ⟨n, hn⟩ := h
  use n
  apply and.intro hn
  have hn' : n = 15 := sorry
  exact hn'

end chess_group_players_l40_40564


namespace smallest_prime_divisor_of_sum_l40_40639

theorem smallest_prime_divisor_of_sum (h1 : Nat.odd (Nat.pow 3 19))
                                      (h2 : Nat.odd (Nat.pow 11 13))
                                      (h3 : ∀ a b : Nat, Nat.odd a → Nat.odd b → Nat.even (a + b)) :
  Nat.minFac (Nat.pow 3 19 + Nat.pow 11 13) = 2 :=
by
  -- placeholder proof
  sorry

end smallest_prime_divisor_of_sum_l40_40639


namespace abs_ineq_range_k_l40_40002

theorem abs_ineq_range_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| > k) → k < 4 :=
by
  sorry

end abs_ineq_range_k_l40_40002


namespace test_total_points_l40_40219

def total_points (total_problems comp_problems : ℕ) (points_comp points_word : ℕ) : ℕ :=
  let word_problems := total_problems - comp_problems
  (comp_problems * points_comp) + (word_problems * points_word)

theorem test_total_points :
  total_points 30 20 3 5 = 110 := by
  sorry

end test_total_points_l40_40219


namespace six_digit_numbers_with_at_least_one_zero_l40_40972

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40972


namespace total_sheets_of_paper_l40_40429

variables (sheets_in_desk : ℕ) (sheets_in_backpack : ℕ)

theorem total_sheets_of_paper (h1 : sheets_in_desk = 50) (h2 : sheets_in_backpack = 41) :
  sheets_in_desk + sheets_in_backpack = 91 :=
by {
  rw [h1, h2],
  exact Nat.add_comm 50 41,
  norm_num,
}

end total_sheets_of_paper_l40_40429


namespace car_dealership_sales_l40_40776

theorem car_dealership_sales (x y : ℕ) (a b : ℕ) :
  (a = 145 ∨ a = 150) →
  (1 * x + 3 * y = 65) →
  (4 * x + 5 * y = 155) →
  (145 ≤ 20 * b + 15 * (8 - b) ∧ 20 * b + 15 * (8 - b) ≤ 153) →
  (x = 20 ∧ y = 15 ∧ ((b = 5 ∧ a = 145) ∨ (b = 6 ∧ a = 150))) :=
begin
  intros h_price_bounds h_week1 h_week2 h_cost_bounds,
  have h_price : x = 20 ∧ y = 15, sorry,
  use h_price,
  have h_plans : (b = 5 ∧ a = 145) ∨ (b = 6 ∧ a = 150), sorry,
  exact ⟨h_price, h_plans⟩,
end

end car_dealership_sales_l40_40776


namespace number_of_16_tuples_l40_40788

theorem number_of_16_tuples :
  let S (x : Fin 16 → ℤ) :=
    (∀ i, x i = 1 ∨ x i = -1) ∧
    (∀ r : Fin 15, 0 ≤ (Finset.range (r.1 + 1)).sum x ∧ (Finset.range (r.1 + 1)).sum x < 4) ∧
    ((Finset.range 10).sum x = 4)
  in
  (∃ (x : Fin 16 → ℤ), S x) →

  (let a_16 : ℚ :=
    1 / Real.sqrt 5 * (
      (3 + Real.sqrt 5) ^ 14 / 2 ^ 14 - 
      (3 - Real.sqrt 5) ^ 14 / 2 ^ 14)
  in
  ∃ (x : Fin 16 → ℤ), S x ∧ (a_16 = 1 / Real.sqrt 5 * ((3 + Real.sqrt 5) / 2) ^ 14 - ((3 - Real.sqrt 5) / 2) ^ 14)
) :=
by
  sorry

end number_of_16_tuples_l40_40788


namespace angle_A44_A45_A43_is_100_l40_40413

noncomputable def midpoint (A B : Point) : Point := sorry -- Definition of midpoint

def isosceles_triangle (A1 A2 A3 : Point) (angle_A1 : ℝ) : Prop :=
  A2 ≠ A3 ∧ A3 ≠ A1 ∧ A1 ≠ A2 ∧ angle_A1 = 100 ∧ dist A2 A3 = dist A3 A1

def midpoint_sequence (A : ℕ → Point) : Prop :=
  ∀ n, A (n + 3) = midpoint (A n) (A (n + 2))

theorem angle_A44_A45_A43_is_100 
  (A : ℕ → Point) (A1 A2 A3 : Point)
  (h_iso : isosceles_triangle A1 A2 A3 100)
  (h_sequence : midpoint_sequence A)
  : measure_angle (A 44) (A 45) (A 43) = 100 :=
sorry

end angle_A44_A45_A43_is_100_l40_40413


namespace eq_triangle_dot_product_l40_40021

theorem eq_triangle_dot_product (ABC : Triangle) (side_length : ℝ) (E : Point)
  (h₁ : ABC.equilateral) (h₂ : side_length = 2)
  (h₃ : E ∈ ABC.AC ∧ vector_to E ABC.A = (1 / 3) • vector_to ABC.C ABC.A) :
  let BE := vector_to E ABC.A - vector_to ABC.B ABC.A,
      BC := vector_to ABC.B ABC.A - vector_to ABC.C ABC.A
  in BE ⋅ BC = (8 / 3) := sorry

end eq_triangle_dot_product_l40_40021


namespace six_digit_numbers_with_zero_l40_40983

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40983


namespace sum_of_vertices_l40_40165

theorem sum_of_vertices (n : ℕ) (h1 : 6 * n + 12 * n = 216) : 8 * n = 96 :=
by
  -- Proof is omitted intentionally
  sorry

end sum_of_vertices_l40_40165


namespace rationalize_denominator_l40_40099

theorem rationalize_denominator :
  ∃ (A B C D : ℤ), 
    D > 0 ∧
    ¬ (∃ p, prime p ∧ p^2 ∣ B) ∧
    Int.gcd (Int.gcd A C) D = 1 ∧ 
    A * B * (3 + Real.sqrt 8) + C * B = 7 ∧ 
    D = (3 + Real.sqrt 8) * (3 - Real.sqrt 8) ∧ 
    A + B + C + D = 23 :=
sorry

end rationalize_denominator_l40_40099


namespace six_digit_numbers_with_zero_l40_40960

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40960


namespace isosceles_right_triangle_ratio_l40_40023

-- Define the conditions of the problem.
variables {a c : ℝ}
hypothesis1 : c = real.sqrt 2 * a

-- Define the theorem to prove the exact value of the ratio 2a/c
theorem isosceles_right_triangle_ratio : c = real.sqrt 2 * a → 2 * a / c = real.sqrt 2 :=
λ h, by {
  -- Assuming the hypothesis
  have h1 : c = real.sqrt 2 * a := h,
  -- sorry allows us to skip the proof
  sorry
}

end isosceles_right_triangle_ratio_l40_40023


namespace find_number_l40_40012

theorem find_number (x : ℤ) (N : ℤ) (h1 : ∃ N, 2.134 * 10 ^ x < N) (h2 : x ≤ 4) : 
  N = 21341 :=
by
  sorry

end find_number_l40_40012


namespace larger_integer_l40_40151

-- Definitions based on the given conditions
def two_integers (x : ℤ) (y : ℤ) :=
  y = 4 * x ∧ (x + 12) * 2 = y

-- Statement of the problem
theorem larger_integer (x : ℤ) (y : ℤ) (h : two_integers x y) : y = 48 :=
by sorry

end larger_integer_l40_40151


namespace print_shop_x_charge_l40_40298

theorem print_shop_x_charge (
  (y_charge: ℝ) (y_total_charge: ℝ) (y_more_than_x: ℝ)) :
  (y_charge = 1.70) →
  (y_total_charge = 70 * y_charge) →
  (y_total_charge = 35 + y_more_than_x) →
  (y_more_than_x / 70 = 1.20) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end print_shop_x_charge_l40_40298


namespace find_x_l40_40806

theorem find_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end find_x_l40_40806


namespace six_digit_numbers_with_at_least_one_zero_l40_40969

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40969


namespace fish_remaining_correct_l40_40767

def guppies := 225
def angelfish := 175
def tiger_sharks := 200
def oscar_fish := 140
def discus_fish := 120

def guppies_sold := 3/5 * guppies
def angelfish_sold := 3/7 * angelfish
def tiger_sharks_sold := 1/4 * tiger_sharks
def oscar_fish_sold := 1/2 * oscar_fish
def discus_fish_sold := 2/3 * discus_fish

def guppies_remaining := guppies - guppies_sold
def angelfish_remaining := angelfish - angelfish_sold
def tiger_sharks_remaining := tiger_sharks - tiger_sharks_sold
def oscar_fish_remaining := oscar_fish - oscar_fish_sold
def discus_fish_remaining := discus_fish - discus_fish_sold

def total_remaining_fish := guppies_remaining + angelfish_remaining + tiger_sharks_remaining + oscar_fish_remaining + discus_fish_remaining

theorem fish_remaining_correct : total_remaining_fish = 450 := 
by 
  -- insert the necessary steps of the proof here
  sorry

end fish_remaining_correct_l40_40767


namespace smallest_n_10_l40_40157

noncomputable def smallest_n_gt_5000 (n : ℕ) : Prop := 
  (∏ k in finset.range(n + 1), 3^(2 * k / 11 : ℝ)) > 5000

theorem smallest_n_10 : (∃ n : ℕ, smallest_n_gt_5000 n) ∧ ∀ m < 10, ¬ smallest_n_gt_5000 m :=
begin
  split,
  { use 10, sorry },
  { intros m hm, sorry }
end

end smallest_n_10_l40_40157


namespace school_enrollment_l40_40132

theorem school_enrollment
  (X Y : ℝ)
  (h1 : X + Y = 4000)
  (h2 : 1.07 * X > X)
  (h3 : 1.03 * Y > Y)
  (h4 : 0.07 * X - 0.03 * Y = 40) :
  Y = 2400 :=
by
  -- problem reduction
  sorry

end school_enrollment_l40_40132


namespace mark_parking_tickets_l40_40734

theorem mark_parking_tickets (total_tickets : ℕ) (same_speeding_tickets : ℕ) (mark_parking_mult_sarah : ℕ) (sarah_speeding_tickets : ℕ) (mark_speeding_tickets : ℕ) (sarah_parking_tickets : ℕ) :
  total_tickets = 24 →
  mark_parking_mult_sarah = 2 →
  mark_speeding_tickets = same_speeding_tickets →
  sarah_speeding_tickets = same_speeding_tickets →
  same_speeding_tickets = 6 →
  sarah_parking_tickets = (total_tickets - 2 * same_speeding_tickets) / 3 →
  2 * sarah_parking_tickets = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw h1 at h6
  rw h5 at h6
  rw h2 at h6
  sorry

end mark_parking_tickets_l40_40734


namespace triangles_in_S_are_isosceles_and_similar_l40_40420

noncomputable def inradius (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Type := sorry
noncomputable def point_of_tangency (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Type := sorry

structure Triangle :=
(A B C : Type)
[inA: MetricSpace A]
[inB: MetricSpace B]
[inC: MetricSpace C]
(P Q R : Type)
[pTang : point_of_tangency A B C]
(inRad : inradius A B C)
(condition : 5 * (1 / (d P A) + 1 / (d Q B) + 1 / (d R C)) - 3 / (min (d P A) (min (d Q B) (d R C))) = 6 / (inRad A B C))

def is_isosceles_and_similar (S : Set Triangle) : Prop :=
∀ t₁ t₂ ∈ S, (isosceles t₁) ∧ (isosceles t₂) ∧ (similar t₁ t₂)

theorem triangles_in_S_are_isosceles_and_similar (S : Set Triangle) 
  (h : ∀ t ∈ S, 5 * (1 / (d t.P t.A) + 1 / (d t.Q t.B) + 1 / (d t.R t.C)) - 3 / (min (d t.P t.A) (min (d t.Q t.B) (d t.R t.C))) = 6 / (t.inRad t.A t.B t.C)) :
  is_isosceles_and_similar S := 
sorry

end triangles_in_S_are_isosceles_and_similar_l40_40420


namespace problem1_l40_40001

theorem problem1 (x y : ℝ) (h1 : x * (x + y) = 27) (h2 : y * (x + y) = 54) : (x + y)^2 = 81 := 
by
  sorry

end problem1_l40_40001


namespace profit_percentage_B_is_25_l40_40706

-- Define the necessary variables and conditions
variables (A B C : Type)
variable (cost_price_A : ℝ) -- Cost price of the bicycle for A
variable (profit_percentage_A : ℝ) -- Profit percentage of A
variable (selling_price_C : ℝ) -- Selling price to C

-- Conditions given in the problem
axiom cost_price_A_def : cost_price_A = 150
axiom profit_percentage_A_def : profit_percentage_A = 20 / 100
axiom selling_price_C_def : selling_price_C = 225

-- Formula for calculating profit
def profit (cp sp : ℝ) : ℝ := sp - cp

-- Formula for calculating selling price
def selling_price (cp profit_percent : ℝ) : ℝ := cp + (profit_percent * cp)

-- Formula for calculating profit percentage
def profit_percentage (profit cp : ℝ) : ℝ := (profit / cp) * 100

-- Main proof statement to be proven
theorem profit_percentage_B_is_25 :
  let cost_price_B := selling_price cost_price_A profit_percentage_A in
  let profit_B := profit cost_price_B selling_price_C in
  profit_percentage profit_B cost_price_B = 25 :=
by
  sorry

end profit_percentage_B_is_25_l40_40706


namespace smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40629

/-- 
  Problem statement: Prove that the smallest prime divisor of 3^19 + 11^13 is 2, 
  given the conditions:
   - 3^19 is odd
   - 11^13 is odd
   - The sum of two odd numbers is even
-/

theorem smallest_prime_divisor_of_3_pow_19_plus_11_pow_13 :
  ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ (3^19 + 11^13) := 
by
  sorry

end smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40629


namespace coffee_break_participants_l40_40577

theorem coffee_break_participants (n : ℕ) (h1 : n = 14) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 6 ∧ n - 2 * k > 0 ∧ n - 2 * k < n)
    (h3 : ∀ m, m < n → (∃ k, m = n - 2 * k → m ∈ {6, 8, 10, 12})): 
    ∃ m, n - 2 * m ∈ {6, 8, 10, 12} := 
by
  use 6
  use 8
  use 10
  use 12
  sorry

end coffee_break_participants_l40_40577


namespace fish_swim_westward_l40_40474

theorem fish_swim_westward :
  ∃ W : ℕ, 
    let eastward := 3200,
        north := 500,
        caught_east := 2 * eastward / 5,
        caught_west := 3 * W / 4,
        total_fish := W + eastward + north,
        fish_left := 2870
    in total_fish - (caught_east + caught_west) = fish_left ∧ W = 1800 :=
by
  -- The proof is omitted as it is not required.
  sorry

end fish_swim_westward_l40_40474


namespace interest_rate_l40_40122

theorem interest_rate (P T R : ℝ) (SI CI : ℝ) (difference : ℝ)
  (hP : P = 1700)
  (hT : T = 1)
  (hdiff : difference = 4.25)
  (hSI : SI = P * R * T / 100)
  (hCI : CI = P * ((1 + R / 200)^2 - 1))
  (hDiff : CI - SI = difference) : 
  R = 10 := sorry

end interest_rate_l40_40122


namespace gym_monthly_revenue_l40_40697

theorem gym_monthly_revenue (members_per_month_fee : ℕ) (num_members : ℕ) 
  (h1 : members_per_month_fee = 18 * 2) 
  (h2 : num_members = 300) : 
  num_members * members_per_month_fee = 10800 := 
by 
  -- calculation rationale goes here
  sorry

end gym_monthly_revenue_l40_40697


namespace least_possible_value_l40_40411

theorem least_possible_value (T : Finset ℕ) 
  (h1 : ∀ x ∈ T, x ≤ 15)
  (h2 : T.card = 5)
  (h3 : ∀ c d ∈ T, c < d → d ≠ c * c) :
  ∃ x ∈ T, x = 2 :=
by
  sorry

end least_possible_value_l40_40411


namespace six_digit_numbers_with_zero_count_l40_40895

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40895


namespace quadrant_of_2019_angle_l40_40772

def quadrant_of_angle (θ : ℝ) : ℕ :=
  if θ % 360 < 90 then 1
  else if θ % 360 < 180 then 2
  else if θ % 360 < 270 then 3
  else 4

theorem quadrant_of_2019_angle : quadrant_of_angle 2019 = 3 := by
  sorry

end quadrant_of_2019_angle_l40_40772


namespace circles_have_three_common_tangent_l40_40851

noncomputable def circle1 (p : ℝ × ℝ) : Prop := p.1 ^ 2 + p.2 ^ 2 = 1
noncomputable def circle2 (a : ℝ) (p : ℝ × ℝ) : Prop := (p.1 + 4) ^ 2 + (p.2 - a) ^ 2 = 25

theorem circles_have_three_common_tangent (a : ℝ) :
  (∃ t, tangent_line t circle1 ∧ tangent_line t (circle2 a)) → a = 2 * real.sqrt 5 ∨ a = -2 * real.sqrt 5 :=
sorry

end circles_have_three_common_tangent_l40_40851


namespace cost_per_bag_l40_40273

theorem cost_per_bag (C : ℝ)
  (total_bags : ℕ := 20)
  (price_per_bag_original : ℝ := 6)
  (sold_original : ℕ := 15)
  (price_per_bag_discounted : ℝ := 4)
  (sold_discounted : ℕ := 5)
  (net_profit : ℝ := 50) :
  sold_original * price_per_bag_original + sold_discounted * price_per_bag_discounted - net_profit = total_bags * C →
  C = 3 :=
by
  intros h
  sorry

end cost_per_bag_l40_40273


namespace six_digit_numbers_with_at_least_one_zero_l40_40856

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40856


namespace largest_coefficient_term_in_expansion_l40_40379

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_coefficient_term_in_expansion :
  ∃ (T : ℕ × ℤ × ℕ), 
  (2 : ℤ) ^ (14 - 1) = 8192 ∧ 
  T = (binom 14 4, 2 ^ 10, 4) ∧ 
  ∀ (k : ℕ), 
    (binom 14 k * (2 ^ (14 - k))) ≤ (binom 14 4 * 2 ^ 10) :=
sorry

end largest_coefficient_term_in_expansion_l40_40379


namespace part_II_l40_40842

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 + (a - 1) * x - Real.log x

theorem part_II (a : ℝ) (h : a > 0) :
  ∀ x > 0, f a x ≥ 2 - (3 / (2 * a)) :=
sorry

end part_II_l40_40842


namespace inequality_solution_set_l40_40290

theorem inequality_solution_set :
  {x : ℝ | |x - 2| > ∫ (t : ℝ) in 0..1, 2 * t} = {x : ℝ | x < 1 ∨ x > 3} := by
  sorry

end inequality_solution_set_l40_40290


namespace max_accepted_ages_l40_40673

noncomputable def maximum_accepted_applicant_ages (avg_age : ℝ) (std_dev : ℝ) : ℕ :=
  let lower_bound := avg_age - std_dev
  let upper_bound := avg_age + std_dev
  (upper_bound.floor.to_nat - lower_bound.ceil.to_nat + 1)

theorem max_accepted_ages :
  maximum_accepted_applicant_ages 31 8 = 17 :=
by
  -- Proof omitted
  sorry

end max_accepted_ages_l40_40673


namespace exists_special_collection_l40_40613

def is_better (a b : ℝ × ℝ × ℝ) : Prop :=
  (a.1 > b.1 ∧ a.2 > b.2) ∨
  (a.2 > b.2 ∧ a.3 > b.3) ∨
  (a.1 > b.1 ∧ a.3 > b.3)

def is_special (t : ℝ × ℝ × ℝ) : Prop :=
  t.1 ≥ 0 ∧ t.2 ≥ 0 ∧ t.3 ≥ 0 ∧ t.1 + t.2 + t.3 = 1

theorem exists_special_collection (n : ℕ) :
  (∃ S : Finset (ℝ × ℝ × ℝ), S.card = n ∧ ∀ t, is_special t → (∃ s ∈ S, is_better s t)) ↔ n ≥ 3 :=
by
  sorry

end exists_special_collection_l40_40613


namespace train_passing_through_tunnel_l40_40222

theorem train_passing_through_tunnel :
  let train_length : ℝ := 300
  let tunnel_length : ℝ := 1200
  let speed_in_kmh : ℝ := 54
  let speed_in_mps : ℝ := speed_in_kmh * (1000 / 3600)
  let total_distance : ℝ := train_length + tunnel_length
  let time : ℝ := total_distance / speed_in_mps
  time = 100 :=
by
  sorry

end train_passing_through_tunnel_l40_40222


namespace sally_balances_unspent_l40_40106

def spending_limit (gold platinum silver diamond : ℝ) := 
  (platinum = 2 * gold) ∧
  (silver = 2.5 * gold) ∧
  (diamond = 3 * gold)

def initial_balances (gold platinum silver diamond : ℝ) :=
  (gold = 1/3 * gold) ∧
  (platinum = 1/7 * (2 * gold)) ∧
  (silver = 3/5 * (2.5 * gold)) ∧
  (diamond = 2/9 * (3 * gold))

def transfer_balances (gold platinum silver diamond : ℝ) :=
  let new_platinum := (2 * gold) / 7 + 1 / 3 * gold in
  let new_silver := 1.5 * gold / 2 in
  let new_diamond := 2 * gold / 3 + 0.75 * gold in
  (new_platinum = 13 / 21 * gold) ∧
  (new_silver = 0.75 * gold) ∧
  (new_diamond = 17 / 12 * gold)

def unspent_portions (gold platinum silver diamond : ℝ) := 
  (gold / gold = 1) ∧
  ((2 * gold - 13 / 21 * gold) / (2 * gold) = 29 / 42) ∧
  ((2.5 * gold - 0.75 * gold) / (2.5 * gold) = 7 / 10) ∧
  ((3 * gold - 17 / 12 * gold) / (3 * gold) = 19 / 36)

theorem sally_balances_unspent (gold platinum silver diamond : ℝ) :
  spending_limit gold platinum silver diamond →
  initial_balances gold platinum silver diamond →
  transfer_balances gold platinum silver diamond →
  unspent_portions gold platinum silver diamond :=
by
  intro sl ib tb
  split 
  sorry
  sorry
  sorry
  sorry

end sally_balances_unspent_l40_40106


namespace participants_coffee_l40_40599

theorem participants_coffee (n : ℕ) (h_n : n = 14) (h_not_all : 0 < n - 2 * k) (h_not_all2 : n - 2 * k < n)
(h_neighbors : ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧ (remaining i → leaving j) ∧ (leaving i → remaining j)) :
  ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 6 ∧ (n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12) :=
by {
  sorry
}

end participants_coffee_l40_40599


namespace participants_coffee_l40_40600

theorem participants_coffee (n : ℕ) (h_n : n = 14) (h_not_all : 0 < n - 2 * k) (h_not_all2 : n - 2 * k < n)
(h_neighbors : ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧ (remaining i → leaving j) ∧ (leaving i → remaining j)) :
  ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 6 ∧ (n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12) :=
by {
  sorry
}

end participants_coffee_l40_40600


namespace six_digit_numbers_with_zero_count_l40_40901

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40901


namespace diana_hours_on_TT_l40_40258

variables (diana_hours_per_day_on_MWF : ℕ)
          (diana_hours_per_day_on_TT : ℕ)
          (diana_weekly_earnings : ℕ)
          (diana_hourly_rate : ℕ)

def diana_total_MWF_hours := 3 * diana_hours_per_day_on_MWF
def diana_total_weekly_hours := diana_weekly_earnings / diana_hourly_rate
def total_hours_TT := diana_total_weekly_hours - diana_total_MWF_hours

theorem diana_hours_on_TT :
    diana_hours_per_day_on_MWF = 10 →
    diana_weekly_earnings = 1800 →
    diana_hourly_rate = 30 →
    total_hours_TT diana_hours_per_day_on_MWF diana_hours_per_day_on_TT diana_weekly_earnings diana_hourly_rate = 30 := 
by
  intros
  sorry

end diana_hours_on_TT_l40_40258


namespace area_circle_l40_40532

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l40_40532


namespace count_valid_4_digit_numbers_l40_40356

-- Define the sets for the digit constraints
def first_digit_set := {1, 4, 5, 6}
def last_digit_set := {3, 6, 9}

-- Define the constraints for the 4-digit number
def valid_first_two_digits (d1 d2 : ℕ) : Prop := d1 ∈ first_digit_set ∧ d2 ∈ first_digit_set
def valid_last_two_digits (d3 d4 : ℕ) : Prop := d3 ∈ last_digit_set ∧ d4 ∈ last_digit_set ∧ d3 ≠ d4
def valid_sum_second_third_digit (d2 d3 : ℕ) : Prop := (d2 + d3) % 2 = 0

-- Main statement translating to the mathematical proof
theorem count_valid_4_digit_numbers :
  ∃ n : ℕ, n = 96 ∧
    (∀ (d1 d2 d3 d4 : ℕ), 
      (valid_first_two_digits d1 d2 
      ∧ valid_last_two_digits d3 d4 
      ∧ valid_sum_second_third_digit d2 d3) → true) := by
  -- Proof steps to be added
  sorry

end count_valid_4_digit_numbers_l40_40356


namespace second_smallest_palindromic_prime_200_l40_40550

-- Definition of a palindrome number in the 200s
def in_200s (n : ℕ) : Prop :=
  200 ≤ n ∧ n < 300

def is_palindrome_200 (n : ℕ) : Prop :=
  in_200s n ∧ (n.mod 10 = n / 100)

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m < n → m > 1 → ¬(n % m = 0)

-- The smallest palindromic prime in the 200s
def smallest_palindromic_prime_200 : ℕ := 202

-- Exists a number which is palindromic prime in the 200s and greater than 202, which is the second smallest palindromic prime
theorem second_smallest_palindromic_prime_200 :
  ∃ n : ℕ, is_palindrome_200 n ∧ is_prime n ∧ n > smallest_palindromic_prime_200 ∧
  ∀ m : ℕ, is_palindrome_200 m ∧ is_prime m ∧ m > smallest_palindromic_prime_200 → n ≤ m :=
  exists.intro 232 sorry

end second_smallest_palindromic_prime_200_l40_40550


namespace sum_of_areas_eq_108_l40_40563

theorem sum_of_areas_eq_108 :
  let rect_area (length : ℕ) : ℕ := 3 * length in
  let areas := [rect_area 2, rect_area 5, rect_area 10, rect_area 17, rect_area 26] in
  (areas.filter (λ x, x % 10 = 0)).sum = 108 :=
by
  sorry

end sum_of_areas_eq_108_l40_40563


namespace median_of_heights_is_correct_l40_40777

def heights : List ℝ := [1.72, 1.78, 1.75, 1.80, 1.69, 1.77]

theorem median_of_heights_is_correct :
  median heights = 1.76 :=
by
  -- proof goes here
  sorry

end median_of_heights_is_correct_l40_40777


namespace ellipse_eccentricity_l40_40136

-- Define the problem conditions
variables (a b c : ℝ)
variables (h1 : a > b) (h2 : b > 0)
variables (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 - 1 = 0)
variables (P : ℝ × ℝ) (hP : P.1^2 / a^2 + P.2^2 / b^2 = 1)
variables (O : ℝ × ℝ) (hO : O = (0, 0))
variables (A : ℝ × ℝ) (hA : A = (a, 0))
variables (angle_POA : ∀ P : ℝ × ℝ, P.1 ≠ 0 → ∠P O A = 60)
variables (perp_OP_AP : ∀ P : ℝ × ℝ, P.1 ≠ 0 → ∠P O A = 90)

-- Prove that the eccentricity of the ellipse is 2√5 / 5
theorem ellipse_eccentricity : 
  let e := c / a in 
  e = 2 * real.sqrt 5 / 5 :=
sorry

end ellipse_eccentricity_l40_40136


namespace rank_three_balls_l40_40562

def minimum_weighings_to_rank_balls (n : ℕ) : ℕ :=
  (n-1) * n / 2

theorem rank_three_balls (n : ℕ) (h₁ : n = 3) :
  minimum_weighings_to_rank_balls n = 3 :=
by
  rw [h₁]
  unfold minimum_weighings_to_rank_balls
  norm_num
  sorry

end rank_three_balls_l40_40562


namespace remainder_of_4_pow_a_div_10_l40_40665

theorem remainder_of_4_pow_a_div_10 (a : ℕ) (h1 : a > 0) (h2 : a % 2 = 0) :
  (4 ^ a) % 10 = 6 :=
by sorry

end remainder_of_4_pow_a_div_10_l40_40665


namespace expected_value_r3_l40_40251

noncomputable def expected_value_single_die : ℚ := (1+2+3+4+5+6) / 6

def r1 (n : ℕ) : ℚ := n * expected_value_single_die
def r2 (n : ℕ) : ℚ := r1 n * expected_value_single_die
def r3 (n : ℕ) : ℚ := r2 n * expected_value_single_die

theorem expected_value_r3 {n : ℕ} (h : n = 8) : r3 n = 343 := by
  rw [r3, r2, r1, h]
  norm_num
  sorry

end expected_value_r3_l40_40251


namespace house_orderings_count_l40_40610

theorem house_orderings_count :
  let houses := Finset (List ℕ)
  let orderings := houses.filter (λ l : List ℕ, 
    ∃ (O R B Y G: ℕ), 
      (O < R) ∧
      (B < Y) ∧
      (abs (l.indexOf B - l.indexOf Y) ≠ 1) ∧
      (G < B ∧ G < Y) ∧
      (l.nodup ∧ 
      (l = [O, R, B, Y, G] ∨
      l = [O, B, R, Y, G] ∨
      l = [O, R, G, B, Y] ∨
      l = [O, G, R, B, Y] ∨
      l = [B, O, R, Y, G] ∨
      l = [B, O, G, R, Y]))
  in orderings.card = 6 := sorry

end house_orderings_count_l40_40610


namespace max_value_of_g_range_of_a_sum_inequality_l40_40837

-- Proof 1: Maximum value of g(x) when b = 1
theorem max_value_of_g {x : ℝ} (h : b = 1) (hx : x > -1) : g x = (x / (1 + x)) - log (1 + x) → g 0 = 0 :=
sorry

-- Proof 2: Range of a such that f(x) ≤ 0 for all x in [0, +∞)
theorem range_of_a (a : ℝ) (hx : ∀ x ≥ 0, ln (1 + x) - a * x ≤ 0) : 1 ≤ a :=
sorry

-- Proof 3: Prove the inequality for the sum
theorem sum_inequality (n : ℕ) : (∑ i in range (1+n), (i/(i^2+1)) - log n ≤ 1/2) :=
sorry

end max_value_of_g_range_of_a_sum_inequality_l40_40837


namespace part1_part2_l40_40819
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then exp (x - 2) - 2 * exp 1 else exp (-x - 2) - 2 * exp 1

theorem part1 (x : ℝ) (hx : x < 0) : f x = exp (-x - 2) - 2 * exp 1 := by
  simp [f, hx]
  sorry

theorem part2 (a : ℝ) (h : f a + f 3 < 0) : -(3 + real.log 3) < a ∧ a < 3 + real.log 3 := by
  sorry

end part1_part2_l40_40819


namespace area_of_region_S_is_correct_l40_40705

noncomputable def area_of_inverted_region (d : ℝ) : ℝ :=
  if h : d = 1.5 then 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi else 0

theorem area_of_region_S_is_correct :
  area_of_inverted_region 1.5 = 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi := 
by 
  sorry

end area_of_region_S_is_correct_l40_40705


namespace rationalize_denominator_l40_40101

theorem rationalize_denominator :
  ∃ (A B C D : ℤ), 
    D > 0 ∧
    ¬ (∃ p, prime p ∧ p^2 ∣ B) ∧
    Int.gcd (Int.gcd A C) D = 1 ∧ 
    A * B * (3 + Real.sqrt 8) + C * B = 7 ∧ 
    D = (3 + Real.sqrt 8) * (3 - Real.sqrt 8) ∧ 
    A + B + C + D = 23 :=
sorry

end rationalize_denominator_l40_40101


namespace tan_theta_condition_l40_40121

open Real

theorem tan_theta_condition (k : ℤ) : 
  (∃ θ : ℝ, θ = 2 * k * π + π / 4 ∧ tan θ = 1) ∧ ¬ (∀ θ : ℝ, tan θ = 1 → ∃ k : ℤ, θ = 2 * k * π + π / 4) :=
by sorry

end tan_theta_condition_l40_40121


namespace six_digit_numbers_with_zero_l40_40933

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40933


namespace no_infinite_m_f_eq_f_succ_l40_40794

-- Definition of the function f(n)
def f (n : ℕ) : ℕ :=
  -- number of subsets of {1, 2, ..., n} whose sum is n
  -- Note: Here we define it abstractly as it's non-trivial to fully implement.
  sorry

-- Statement of the problem
theorem no_infinite_m_f_eq_f_succ :
  ¬ (∃ᶠ m, f(m) = f(m+1)) :=
sorry

end no_infinite_m_f_eq_f_succ_l40_40794


namespace six_digit_numbers_with_at_least_one_zero_l40_40976

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40976


namespace routes_A_to_B_on_3x2_grid_l40_40357

def routes_from_A_to_B : ℕ := 10

/-- Prove the number of different routes from point A to point B on a 3x2 grid --/
theorem routes_A_to_B_on_3x2_grid : routes_from_A_to_B = (nat.choose 5 2) := by
  sorry

end routes_A_to_B_on_3x2_grid_l40_40357


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40938

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40938


namespace outer_perimeter_of_fence_l40_40472

/-
This definition captures the conditions provided in the problem:
1. Sixteen 6-inch wide square posts.
2. Posts are evenly spaced with 4 feet between adjacent posts.
3. The posts enclose a square field.

From these conditions, we aim to prove the outer perimeter of the square fence equals 56 feet.
-/
theorem outer_perimeter_of_fence
  (num_posts : ℕ) (post_width_inch : ℤ) (gap_length_ft : ℤ)
  (is_square_field : bool)
  (h_posts : num_posts = 16)
  (h_post_width : post_width_inch = 6)
  (h_gap_length : gap_length_ft = 4)
  (h_field : is_square_field = true) :
  (4 * (((num_posts / 4) * (post_width_inch / 12)) + ((num_posts / 4 - 1) * gap_length_ft))) = 56 := sorry

end outer_perimeter_of_fence_l40_40472


namespace six_digit_numbers_with_zero_l40_40880

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40880


namespace smallest_prime_divisor_of_sum_l40_40619

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end smallest_prime_divisor_of_sum_l40_40619


namespace toucan_count_correct_l40_40182

def initial_toucans : ℕ := 2
def toucans_joined : ℕ := 1
def total_toucans : ℕ := initial_toucans + toucans_joined

theorem toucan_count_correct : total_toucans = 3 := by
  sorry

end toucan_count_correct_l40_40182


namespace combine_like_terms_l40_40749

theorem combine_like_terms (x y : ℝ) : 3 * x * y - 6 * x * y + (-2) * x * y = -5 * x * y := by
  sorry

end combine_like_terms_l40_40749


namespace ellipse_eccentricity_l40_40727

-- Definitions and conditions
def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((1, 0), (3, 0))
def origin_pass (p : ℝ × ℝ) : Prop := p = (0, 0)

-- Eccentricity function using given conditions shall be computed and validated
theorem ellipse_eccentricity : 
  (exists (a c : ℝ), (a = 2) ∧ (c = 1) ∧ (c / a = 1 / 2) ∧ ∀ (x : ℝ × ℝ), origin_pass x → x = (0, 0)) →
  ∃ e : ℝ, e = 1 / 2 := 
by
  intros h,
  sorry

end ellipse_eccentricity_l40_40727


namespace difference_max_min_change_l40_40236

noncomputable def initial_percentages : ℝ × ℝ × ℝ := (0.40, 0.40, 0.20)
noncomputable def final_percentages : ℝ × ℝ × ℝ := (0.60, 0.30, 0.10)

theorem difference_max_min_change :
  let x_max := 0.40 in  -- Maximum change in percentage
  let x_min := 0.30 in  -- Minimum change in percentage
  x_max - x_min = 0.10 := sorry

end difference_max_min_change_l40_40236


namespace parabola_eq_4x_line_passes_through_fixed_point_minimum_area_quadrilateral_l40_40344

-- Parabola definition
def parabola (x y : ℝ) (p : ℝ) : Prop := y^2 = 2 * p * x

-- Condition: p > 0 and focus at (1, 0) implies the equation of the parabola is y^2 = 4x
theorem parabola_eq_4x (p : ℝ) (hp : 0 < p) (focus_eq : (1:ℝ) = p / 2) :
  ∀ x y, parabola x y 2 ↔ y^2 = 4 * x :=
by sorry

-- Condition: OA • OB = -4
def OA (A : ℝ × ℝ) : ℝ × ℝ := A
def OB (B : ℝ × ℝ) : ℝ × ℝ := B

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- A and B are on the parabola and dot product equals -4 implies the line passes through (2, 0)
theorem line_passes_through_fixed_point
  (A B : ℝ × ℝ) (hA : ∃ x y, parabola x y 2 ∧ A = (x, y)) (hB : ∃ x y, parabola x y 2 ∧ B = (x, y))
  (dot_eq : dot_product (OA A) (OB B) = -4) :
  ∃ T : ℝ × ℝ, T = (2, 0) :=
by sorry

-- Minimum area of quadrilateral AMBN
theorem minimum_area_quadrilateral (A B M N : ℝ × ℝ)
  (hA : ∃ x1 y1, parabola x1 y1 2 ∧ A = (x1, y1))
  (hB : ∃ x2 y2, parabola x2 y2 2 ∧ B = (x2, y2))
  (hM : ∃ x3 y3, parabola x3 y3 2 ∧ M = (x3, y3))
  (hN : ∃ x4 y4, parabola x4 y4 2 ∧ N = (x4, y4))
  (ht : ∀ x1 y1 x2 y2, dot_product (OA A) (OB B) = -4 ∧ line_passes_through_fixed_point A B hA hB dot_eq) :
  8 * real.sqrt (2 + (1 + real.sqrt (5))) = 48 :=
by sorry

end parabola_eq_4x_line_passes_through_fixed_point_minimum_area_quadrilateral_l40_40344


namespace biased_coin_probability_l40_40694

theorem biased_coin_probability (x : ℝ) (h1 : x < 1/2) (h2 : (6.choose 2) * x^2 * (1 - x)^4 = 1 / 8) : 
  x = 1 / 5 :=
by
  -- proof here
  sorry

end biased_coin_probability_l40_40694


namespace six_digit_numbers_with_zero_l40_40962

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40962


namespace measure_six_pints_l40_40111
-- Importing the necessary library

-- Defining the problem conditions
def total_wine : ℕ := 12
def capacity_8_pint_vessel : ℕ := 8
def capacity_5_pint_vessel : ℕ := 5

-- The problem to prove: it is possible to measure 6 pints into the 8-pint container
theorem measure_six_pints :
  ∃ (n : ℕ), n = 6 ∧ n ≤ capacity_8_pint_vessel := 
sorry

end measure_six_pints_l40_40111


namespace ratio_new_circumference_to_new_diameter_l40_40693

-- Define original and new dimensions of the circle
def original_diameter (r : ℝ) : ℝ := 2 * r
def new_diameter (r : ℝ) : ℝ := 2 * r + 4

-- Define the new circumference based on the adjusted diameter
def new_circumference (r : ℝ) : ℝ := 2 * Real.pi * (r + 2)

-- Prove the ratio of the new circumference to the new diameter is π
theorem ratio_new_circumference_to_new_diameter (r : ℝ) : 
  (new_circumference r) / (new_diameter r) = Real.pi := by
  sorry

end ratio_new_circumference_to_new_diameter_l40_40693


namespace acute_triangle_isosceles_base_angles_l40_40461

theorem acute_triangle_isosceles_base_angles {A B C A1 B1 F : Point} 
  (hABC : acute_triangle A B C)
  (hA1 : perpendicular A1 A C B) 
  (hB1 : perpendicular B1 B C A)
  (hF : midpoint F A B) :
  let γ := angle A C B in 
  isosceles_triangle F A1 B1 ∧ 
  angle F A1 B1 = γ := 
sorry

end acute_triangle_isosceles_base_angles_l40_40461


namespace mle_binomial_distribution_l40_40790

variables (m n : ℕ) (x : ℕ → ℕ)

def likelihood (p : ℝ) : ℝ :=
  Real.log (∏ i in Finset.range n, Nat.choose m (x i) * p^(x i) * (1 - p)^(m - x i))

def mle_estimate (m n : ℕ) (x : ℕ → ℕ) : ℝ :=
  ∑ i in Finset.range n, x i / (n * m)

theorem mle_binomial_distribution (m n : ℕ) (x : ℕ → ℕ) :
  p = mle_estimate m n x :=
sorry

end mle_binomial_distribution_l40_40790


namespace equation_of_lamps_l40_40422

theorem equation_of_lamps (n k : ℕ) (N M : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : k ≥ n) (h4 : (k - n) % 2 = 0) : 
  N = 2^(k - n) * M := 
sorry

end equation_of_lamps_l40_40422


namespace pebble_difference_l40_40741

-- Definitions and conditions
variables (x : ℚ) -- we use rational numbers for exact division
def Candy := 2 * x
def Lance := 5 * x
def Sandy := 4 * x
def condition1 := Lance = Candy + 10

-- Theorem statement
theorem pebble_difference (h : condition1) : Lance + Sandy - Candy = 30 :=
sorry

end pebble_difference_l40_40741


namespace f_lt_g_range_of_a_l40_40426

noncomputable def f (x : ℝ) := |x + 3|
noncomputable def g (x : ℝ) := |2x - 1|

theorem f_lt_g (x : ℝ) : f x < g x → (x < -2/3 ∨ x > 4) :=
sorry

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * f x + g x > a * x + 4) → a ∈ Icc (-1 : ℝ) 4 :=
sorry

end f_lt_g_range_of_a_l40_40426


namespace rationalize_fraction_sum_l40_40093

theorem rationalize_fraction_sum :
  ∃ (A B C D : ℤ),
    D > 0 ∧
    ¬ ∃ p : ℕ, p.prime ∧ p^2 ∣ B ∧
    Int.gcd A C D = 1 ∧
    A * 8 + C = 21 - 7 * 3 ∧
    A + B + C + D = 23 :=
sorry

end rationalize_fraction_sum_l40_40093


namespace volume_of_cuboid_box_l40_40445

-- Define the conditions
def bottom_area : ℝ := 14 -- bottom area in square centimeters
def height : ℝ := 13 -- height in centimeters

-- Define the volume function
def volume (base_area height : ℝ) : ℝ := base_area * height

-- State the theorem
theorem volume_of_cuboid_box :
  volume bottom_area height = 182 :=
by
  sorry -- Proof is left as an exercise

end volume_of_cuboid_box_l40_40445


namespace x9_is_1_l40_40047

noncomputable def a : ℝ := real.sqrt 2013 / real.sqrt 2014
def x (n : ℕ) : ℝ := real.floor ((n + 1) * a) - real.floor (n * a)

theorem x9_is_1 : x 9 = 1 :=
by 
  -- Placeholder for the proof
  sorry

end x9_is_1_l40_40047


namespace right_triangle_area_l40_40282

-- Define the conditions a = 4/3 * b and a = 2/3 * c
variable (a b c : ℝ)
hypothesis h1 : a = (4 / 3) * b
hypothesis h2 : a = (2 / 3) * c

-- Define the theorem stating that the area of the right triangle is 2/3
theorem right_triangle_area : (1 / 2) * a * b = (2 / 3) := by
  sorry

end right_triangle_area_l40_40282


namespace six_digit_numbers_with_zero_l40_40951

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40951


namespace nonagon_diagonals_l40_40700

-- Define nonagon and its properties
def is_nonagon (n : ℕ) : Prop := n = 9
def has_parallel_sides (n : ℕ) : Prop := n = 9 ∧ true

-- Define the formula for calculating diagonals in a convex polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The main theorem statement
theorem nonagon_diagonals :
  ∀ (n : ℕ), is_nonagon n → has_parallel_sides n → diagonals n = 27 :=  by 
  intros n hn _ 
  rw [is_nonagon] at hn
  rw [hn]
  sorry

end nonagon_diagonals_l40_40700


namespace six_digit_numbers_with_zero_l40_40958

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40958


namespace total_frogs_in_ponds_l40_40566

def pondA_frogs := 32
def pondB_frogs := pondA_frogs / 2

theorem total_frogs_in_ponds : pondA_frogs + pondB_frogs = 48 := by
  sorry

end total_frogs_in_ponds_l40_40566


namespace Quadrilateral_Angles_Equal_l40_40459

theorem Quadrilateral_Angles_Equal
  (A B C D : Point)
  (AB CD BC DA : Line)
  (P : Quadrilateral A B C D)
  (h1 : P.OppositeSidesEqual AB CD BC DA)
  (h2 : P.AdjacentAnglesEqual ∡BAD ∡CDA) :
  ∡ABC = ∡CDA := 
    sorry

end Quadrilateral_Angles_Equal_l40_40459


namespace area_distance_relationship_l40_40033

theorem area_distance_relationship (R d S_1 : ℝ) (triangle_ABC : Σa b c : ℝ, a = b ∧ b = c ∧ c = a)
  (P_inside_circle : ∃ P : ℝ × ℝ, true) :
  d^2 + (4 * S_1 / Real.sqrt 3) = R^2 :=
sorry

end area_distance_relationship_l40_40033


namespace abs_neg_two_l40_40501

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end abs_neg_two_l40_40501


namespace max_project_teams_l40_40209

theorem max_project_teams (n : ℕ) :
  (∃ students : Finset (Fin 14),
     ∃ teams : Finset (Fin n),
     (∀ t ∈ teams, (t.val : Finset (Fin 14)).card = 6) ∧
     (∀ s ∈ students, ∃ t₁ t₂ ∈ teams, s ∈ t₁.val ∧ s ∈ t₂.val) ∧
     (∀ t₁ t₂ ∈ teams, t₁ ≠ t₂ → (t₁.val ∩ t₂.val).card ≤ 2))
     → n ≤ 7 :=
begin
  sorry
end

end max_project_teams_l40_40209


namespace six_digit_numbers_with_zero_count_l40_40898

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40898


namespace relationship_between_length_and_width_l40_40174

theorem relationship_between_length_and_width 
  (x y : ℝ) (h : 2 * (x + y) = 20) : y = 10 - x := 
by
  sorry

end relationship_between_length_and_width_l40_40174


namespace tenth_term_of_sequence_l40_40249

theorem tenth_term_of_sequence : 
  let a_1 := 3
  let d := 6 
  let n := 10 
  (a_1 + (n-1) * d) = 57 := by
  sorry

end tenth_term_of_sequence_l40_40249


namespace perpendicular_vectors_find_a_l40_40355

theorem perpendicular_vectors_find_a
  (a : ℝ)
  (m : ℝ × ℝ := (1, 2))
  (n : ℝ × ℝ := (a, -1))
  (h : m.1 * n.1 + m.2 * n.2 = 0) :
  a = 2 := 
sorry

end perpendicular_vectors_find_a_l40_40355


namespace smallest_angle_of_triangle_l40_40549

theorem smallest_angle_of_triangle (y : ℝ) (h : 40 + 70 + y = 180) : 
  ∃ smallest_angle : ℝ, smallest_angle = 40 ∧ smallest_angle = min 40 (min 70 y) := 
by
  use 40
  sorry

end smallest_angle_of_triangle_l40_40549


namespace inequality_may_not_hold_l40_40000

theorem inequality_may_not_hold (m n : ℝ) (h : m > n) : ¬ (m^2 > n^2) :=
by
  -- Leaving the proof out according to the instructions.
  sorry

end inequality_may_not_hold_l40_40000


namespace find_dihedral_angle_PAD_M_l40_40392

variables {P A B C D N M : Point}
variables {PAD ABCD DAN PC : Plane}

-- Conditions
structure QuadrilateralPyramid (P A B C D : Point) : Prop :=
(face_PAD_equilateral : equilateral_triangle P A D)
(face_PAD_perpendicular_base : is_perpendicular PAD ABCD)
(base_ABCD_rhombus : rhombus A B C D)
(base_side_length : side_length (A B C D) = 2)
(base_angle_60 : angle A B D = 60)
(N_midpoint_PB : midpoint N P B)
(DAN_intersects_PC_at_M : intersects_at DAN PC M)

-- Question
theorem find_dihedral_angle_PAD_M (P A B C D N M : Point)
  (h : QuadrilateralPyramid P A B C D) :
  dihedral_angle P A D M = 45 :=
sorry

end find_dihedral_angle_PAD_M_l40_40392


namespace not_all_same_after_turns_l40_40248

theorem not_all_same_after_turns (n : ℕ) (m : ℕ) (board : Array (Array ℤ)) :
  (∀ i j, i < n → j < m → 
    let neighbours_avg := (λ (i j : ℕ), 
      let neighbours := [if i > 0 then board[i - 1][j] else 0,
        if i < n - 1 then board[i + 1][j] else 0,
        if j > 0 then board[i][j - 1] else 0,
        if j < m - 1 then board[i][j + 1] else 0].filterMap id
      in match neighbours with
      | [] => 0
      | ns => ns.sum / ns.length)
    in board[i][j] = neighbours_avg i j) → False :=
begin
  sorry
end

end not_all_same_after_turns_l40_40248


namespace height_at_10inches_l40_40210

theorem height_at_10inches 
  (a : ℚ)
  (h : 20 = (- (4 / 125) * 25 ^ 2 + 20))
  (span_eq : 50 = 50)
  (height_eq : 20 = 20)
  (y_eq : ∀ x : ℚ, - (4 / 125) * x ^ 2 + 20 = 16.8) :
  (- (4 / 125) * 10 ^ 2 + 20) = 16.8 :=
by
  sorry

end height_at_10inches_l40_40210


namespace six_digit_numbers_with_zero_l40_40881

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40881


namespace problem_solution_l40_40164

open Real

theorem problem_solution (x : ℝ) :
  (\frac{4 * x^4 - 20 * x^2 + 18}{x^4 - 5 * x^2 + 4} < 3) ↔
  (x ∈ Set.Ioo (-2) (-sqrt 3) ∪ Set.Ioo (sqrt 3) 2 ∪ Set.Ioo (-sqrt 2) (-1) ∪ Set.Ioo 1 (sqrt 2)) :=
  sorry

end problem_solution_l40_40164


namespace A_can_force_B_to_lose_l40_40152

noncomputable def game_on_8x8_grid :=
  ∀ B : ℕ, (A : ℕ → option (fin 64 × fin 64)) → (B_moves : ℕ → option (fin 64)) → Prop

def A_wins (A B : ℕ) : Prop :=
  ∃ k, ∀ n, n < k → game_on_8x8_grid B A (λ m, if m < n then B_moves m else none)

theorem A_can_force_B_to_lose :
  A_wins 8 :=
sorry

end A_can_force_B_to_lose_l40_40152


namespace six_digit_numbers_with_zero_l40_40935

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40935


namespace framing_required_l40_40682

/- 
  Problem: A 5-inch by 7-inch picture is enlarged by quadrupling its dimensions.
  A 3-inch-wide border is then placed around each side of the enlarged picture.
  What is the minimum number of linear feet of framing that must be purchased
  to go around the perimeter of the border?
-/
def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

theorem framing_required : (2 * ((original_width * enlargement_factor + 2 * border_width) + (original_height * enlargement_factor + 2 * border_width))) / 12 = 10 :=
by
  sorry

end framing_required_l40_40682


namespace central_angle_of_sector_l40_40828

theorem central_angle_of_sector :
  ∃ (α : ℝ), (radius l : ℝ), 
    2 * radius + l = 10 ∧ (1 / 2) * l * radius = 4 ∧ α = l / radius ∧ α = 1 / 2 :=
sorry

end central_angle_of_sector_l40_40828


namespace ambiguous_dates_in_year_l40_40028

def is_ambiguous_date (m d : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 12 ∧ m ≠ d

theorem ambiguous_dates_in_year :
  ∃ n : ℕ, n = 132 ∧ (∀ m d : ℕ, is_ambiguous_date m d → n = 132) :=
sorry

end ambiguous_dates_in_year_l40_40028


namespace mark_parking_tickets_eq_l40_40732

def total_tickets : ℕ := 24
def sarah_speeding_tickets : ℕ := 6
def mark_speeding_tickets : ℕ := 6
def sarah_parking_tickets (S : ℕ) := S
def mark_parking_tickets (S : ℕ) := 2 * S
def total_traffic_tickets (S : ℕ) := S + 2 * S + sarah_speeding_tickets + mark_speeding_tickets

theorem mark_parking_tickets_eq (S : ℕ) (h1 : total_traffic_tickets S = total_tickets)
  (h2 : sarah_speeding_tickets = 6) (h3 : mark_speeding_tickets = 6) :
  mark_parking_tickets S = 8 :=
sorry

end mark_parking_tickets_eq_l40_40732


namespace six_digit_numbers_with_zero_l40_40954

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40954


namespace circumference_of_post_l40_40711

theorem circumference_of_post
  (spiral_path : Type)
  (circuit_height : ℝ := 4)
  (post_height : ℝ := 16)
  (traveled_distance : ℝ := 8):
  (∃ (C : ℝ), 2 * C = traveled_distance) →
  ∃ (C : ℝ), C = 4 :=
by
  intros hC
  obtain ⟨C, h⟩ := hC
  use C
  rw [← h, mul_comm]
  exact mul_div_cancel_left _ zero_lt_two

end circumference_of_post_l40_40711


namespace six_digit_numbers_with_zero_l40_40888

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40888


namespace scientific_notation_932700_l40_40720

theorem scientific_notation_932700 : 932700 = 9.327 * 10^5 :=
sorry

end scientific_notation_932700_l40_40720


namespace calculate_product_sum_l40_40721

-- Define the structure for points, lengths and perpendicularity in the plane
variables {A B C P Q R H : Type}
variables (triangle : ∀ {A B C : Type}, Type) 
variables (altitude : ∀ {A B : Type}, Type)

-- Given conditions
variables (HP : ℝ := 3) (HQ : ℝ := 4) (HR : ℝ := 6)

-- Define that point H is the orthocenter of triangle ABC
variables (orthocenter_H : ∀ {A B C : Type}, Type)

-- Define altitudes intersect at orthocenter, lengths and perpendicularities
variables (is_altitude_AP : altitude A P)
variables (is_altitude_BQ : altitude B Q)
variables (is_perpendicular_CR : altitude C R)
variables (intersect_at_H : A P B Q H)

-- Assertion to prove the required expression evaluates to 11
theorem calculate_product_sum :
  ( (BP : ℝ) * (PC : ℝ) ) - ( (AQ : ℝ) * (QC : ℝ) ) + ( (CR : ℝ) * (RA : ℝ) ) = 11 := by
  sorry

end calculate_product_sum_l40_40721


namespace determine_OP_squared_l40_40196

-- Define the given conditions
variable (O P : Point) -- Points: center O and intersection point P
variable (r : ℝ) (AB CD : ℝ) (E F : Point) -- radius, lengths of chords, midpoints of chords
variable (OE OF : ℝ) -- Distances from center to midpoints of chords
variable (EF : ℝ) -- Distance between midpoints
variable (OP : ℝ) -- Distance from center to intersection point

-- Conditions as given
axiom circle_radius : r = 30
axiom chord_AB_length : AB = 40
axiom chord_CD_length : CD = 14
axiom distance_midpoints : EF = 15
axiom distance_OE : OE = 20
axiom distance_OF : OF = 29

-- The proof problem: determine that OP^2 = 733 given the conditions
theorem determine_OP_squared :
  OP^2 = 733 :=
sorry

end determine_OP_squared_l40_40196


namespace sharpened_pencil_length_l40_40038

theorem sharpened_pencil_length (length_before length_after sharpened: ℝ)
  (h_before : length_before = 31.25)
  (h_after : length_after = 14.75)
  (h_sharpened : sharpened = length_before - length_after) :
  sharpened = 16.5 :=
by
  rw [h_before, h_after] at h_sharpened
  exact h_sharpened

end sharpened_pencil_length_l40_40038


namespace average_distance_is_600_l40_40444

-- Definitions based on the given conditions
def distance_around_block := 200
def johnny_rounds := 4
def mickey_rounds := johnny_rounds / 2

-- The calculated distances
def johnny_distance := johnny_rounds * distance_around_block
def mickey_distance := mickey_rounds * distance_around_block

-- The average distance computation
def average_distance := (johnny_distance + mickey_distance) / 2

-- The theorem to prove that the average distance is 600 meters
theorem average_distance_is_600 : average_distance = 600 := by sorry

end average_distance_is_600_l40_40444


namespace expected_hit_targets_expected_hit_targets_not_less_than_half_l40_40261

-- Part (a): The expected number of hit targets
theorem expected_hit_targets (n : ℕ) (h : n ≠ 0) :
  E (number_of_hit_targets n) = n * (1 - (1 - (1 / n)) ^ n) :=
sorry

-- Part (b): The expected number of hit targets cannot be less than n / 2
theorem expected_hit_targets_not_less_than_half (n : ℕ) (h : n ≠ 0) :
  n * (1 - (1 - (1 / n)) ^ n) ≥ n / 2 :=
sorry

end expected_hit_targets_expected_hit_targets_not_less_than_half_l40_40261


namespace smallest_prime_divisor_of_sum_l40_40628

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end smallest_prime_divisor_of_sum_l40_40628


namespace main_theorem_l40_40451

variable (a α : ℝ)
def A : ℝ × ℝ := (-a, 0)
def B : ℝ × ℝ := (a, 0)
def C : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (a * Real.cos α, a * Real.sin α)
def N : ℝ × ℝ := (-a * Real.sin α, a * Real.cos α)
def P : ℝ × ℝ := (a * (Real.cos α - Real.sin α), 0)

def dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def PM := dist P M
def PN := dist P N
def PC := dist P C

theorem main_theorem : |PM - PN| / PC = Real.sqrt 2 :=
by
  sorry

end main_theorem_l40_40451


namespace no_solution_bills_l40_40036

theorem no_solution_bills (x y z : ℕ) (h1 : x + y + z = 10) (h2 : x + 3 * y + 5 * z = 25) : false :=
by
  sorry

end no_solution_bills_l40_40036


namespace six_digit_numbers_with_at_least_one_zero_l40_40977

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40977


namespace minimum_value_M_sub_N_l40_40127

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)

theorem minimum_value_M_sub_N :
  ∃ (t : ℝ), 
  let M := Real.sup (Set.image f (Set.interval t (t + Real.pi / 4))),
      N := Real.inf (Set.image f (Set.interval t (t + Real.pi / 4)))
  in M - N = 2 - Real.sqrt 2 :=
sorry

end minimum_value_M_sub_N_l40_40127


namespace participants_coffee_l40_40585

theorem participants_coffee (n : ℕ) (h1 : n = 14) (h2 : 0 < (14 - 2 * (n / 2)) < 14) : 
  ∃ (k : ℕ), k ∈ {6, 8, 10, 12} ∧ k = (14 - 2 * (n / 2)) :=
by sorry

end participants_coffee_l40_40585


namespace problem_inequality_l40_40309

variable (x y z : ℝ)

theorem problem_inequality (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := by
  sorry

end problem_inequality_l40_40309


namespace six_digit_numbers_with_zero_count_l40_40897

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40897


namespace product_of_roots_of_unity_l40_40759

theorem product_of_roots_of_unity : 
  ∏ k in Finset.range 1 16, ∏ j in Finset.range 1 13, (Complex.exp(2 * Real.pi * j * Complex.I / 13) - Complex.exp(2 * Real.pi * k * Complex.I / 17)) = 0 :=
sorry

end product_of_roots_of_unity_l40_40759


namespace langsley_commute_time_l40_40079

theorem langsley_commute_time (first_bus: ℕ) (first_wait: ℕ) (second_bus: ℕ) (second_wait: ℕ) (third_bus: ℕ) (total_time: ℕ)
  (h1: first_bus = 40)
  (h2: first_wait = 10)
  (h3: second_bus = 50)
  (h4: second_wait = 15)
  (h5: third_bus = 95)
  (h6: total_time = first_bus + first_wait + second_bus + second_wait + third_bus) :
  total_time = 210 := 
by 
  sorry

end langsley_commute_time_l40_40079


namespace geometric_sum_six_l40_40555

theorem geometric_sum_six (a r : ℚ) (n : ℕ) 
  (hn₁ : a = 1/4) 
  (hn₂ : r = 1/2) 
  (hS: a * (1 - r^n) / (1 - r) = 63/128) : 
  n = 6 :=
by
  -- Statement to be Proven
  rw [hn₁, hn₂] at hS
  sorry

end geometric_sum_six_l40_40555


namespace probability_nonnegative_value_of_f_l40_40814

def f (x : ℝ) : ℝ := 3 * (Real.sin (2 * x - Real.pi / 6))

theorem probability_nonnegative_value_of_f :
  let I := Set.Icc (-Real.pi / 4) (2 * Real.pi / 3)
  (Set.intervalIntegral (λ x, indicator I (λ x, 1)) (fun x => if f x ≥ 0 then 1 else 0) I) / (Set.intervalIntegral (λ x, indicator I (λ x, 1)) (λ x, 1) I) = 6 / 11 :=
sorry

end probability_nonnegative_value_of_f_l40_40814


namespace even_rows_in_pascal_triangle_l40_40770

theorem even_rows_in_pascal_triangle :
  (List.filter (λ n, ∀ k : ℕ, k > 0 ∧ k < n → binom n k % 2 = 0) (List.range' 2 24)).length = 4 :=
by
  sorry

end even_rows_in_pascal_triangle_l40_40770


namespace find_f_prime_one_l40_40844

noncomputable def f : ℝ → ℝ := λ x => Real.log x - (4 / 3) * x^2 + 3 * x + 2

theorem find_f_prime_one : 
  let f_prime := λ x, 1 / x - 2 * (4 / 3) * x + 3 in
  f_prime 1 = 4 / 3 :=
by
  sorry

end find_f_prime_one_l40_40844


namespace smallest_prime_divisor_of_sum_l40_40640

theorem smallest_prime_divisor_of_sum (h1 : Nat.odd (Nat.pow 3 19))
                                      (h2 : Nat.odd (Nat.pow 11 13))
                                      (h3 : ∀ a b : Nat, Nat.odd a → Nat.odd b → Nat.even (a + b)) :
  Nat.minFac (Nat.pow 3 19 + Nat.pow 11 13) = 2 :=
by
  -- placeholder proof
  sorry

end smallest_prime_divisor_of_sum_l40_40640


namespace overall_profit_percentage_is_correct_l40_40215

structure Item :=
  (units_sold : ℕ)
  (selling_price_per_unit : ℕ)
  (cost_price_per_unit : ℕ)

def profit_percentage (items : List Item) : ℝ :=
  let total_cost_price := items.foldl (λ acc item => acc + item.units_sold * item.cost_price_per_unit) 0
  let total_selling_price := items.foldl (λ acc item => acc + item.units_sold * item.selling_price_per_unit) 0
  let profit := total_selling_price - total_cost_price
  (profit.toFloat / total_cost_price.toFloat) * 100

def item_A := Item.mk 5 500 450
def item_B := Item.mk 10 1500 1300
def item_C := Item.mk 8 900 800

theorem overall_profit_percentage_is_correct :
  profit_percentage [item_A, item_B, item_C] ≈ 14.09 :=
by
  sorry

end overall_profit_percentage_is_correct_l40_40215


namespace find_line_equation_and_check_area_l40_40082

noncomputable def nine_circles_packed : set (ℕ × ℕ) :=
{ (x, y) | x < 3 ∧ y < 3 }

def circle_center_coordinates (x y : ℕ) (h : (x, y) ∈ nine_circles_packed) : ℝ × ℝ :=
(((x : ℝ) + 0.5), ((y : ℝ) + 0.5))

def S : set (ℝ × ℝ) :=
ball (0.5, 0.5) 0.5 ∪ ball (1.5, 0.5) 0.5 ∪ ball (2.5, 0.5) 0.5 ∪
ball (0.5, 1.5) 0.5 ∪ ball (1.5, 1.5) 0.5 ∪ ball (2.5, 1.5) 0.5 ∪
ball (0.5, 2.5) 0.5 ∪ ball (1.5, 2.5) 0.5 ∪ ball (2.5, 2.5) 0.5

def line_m (x : ℝ) : ℝ := 4 * x - 3

theorem find_line_equation_and_check_area :
  ∃ (a b c : ℤ), gcd a b c = 1 ∧ (∀ (x y : ℝ), y = line_m x -> 4 * x - y = 3) ∧ 
  (4^2 + (-1)^2 + 3^2 = 26) ∧ 
  (let S1 := {p : ℝ × ℝ | p.2 ≤ line_m p.1} in let S2 := {p : ℝ × ℝ | line_m p.1 < p.2} in 
   S1 ∪ S2 = S ∧ disjoint S1 S2 ∧ area (S1 ∩ S) = area (S2 ∩ S)) :=
begin
  sorry
end

end find_line_equation_and_check_area_l40_40082


namespace exponential_identity_l40_40372

theorem exponential_identity (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h1 : a * 2^b = 8) (h2 : a^b = 2) : a^(Real.log2 a) * 2^(b^2) = 128 := by
  sorry

end exponential_identity_l40_40372


namespace form_regular_octagon_l40_40679

def concentric_squares_form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) : Prop :=
  ∀ (p : ℂ), ∃ (h₃ : ∀ (pvertices : ℤ → ℂ), -- vertices of the smaller square
                ∀ (lperpendiculars : ℤ → ℂ), -- perpendicular line segments
                true), -- additional conditions representing the perpendicular lines construction
    -- proving that the formed shape is a regular octagon:
    true -- Placeholder for actual condition/check for regular octagon

theorem form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) :
  concentric_squares_form_regular_octagon a b h₀ h₁ h₂ :=
by sorry

end form_regular_octagon_l40_40679


namespace sum_divisible_by_7_l40_40156

theorem sum_divisible_by_7 : (∑ i in Finset.range 21, i) % 7 = 0 := by
  sorry

end sum_divisible_by_7_l40_40156


namespace six_digit_numbers_with_at_least_one_zero_l40_40862

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40862


namespace m_is_perfect_square_l40_40044

theorem m_is_perfect_square
  (m n k : ℕ) 
  (h1 : 0 < m) 
  (h2 : 0 < n) 
  (h3 : 0 < k) 
  (h4 : 1 + m + n * Real.sqrt 3 = (2 + Real.sqrt 3) ^ (2 * k + 1)) : 
  ∃ a : ℕ, m = a ^ 2 :=
by 
  sorry

end m_is_perfect_square_l40_40044


namespace routes_from_A_to_B_l40_40359

theorem routes_from_A_to_B (n_r n_d : ℕ) (n_r_eq : n_r = 3) (n_d_eq : n_d = 2) :
  nat.choose (n_r + n_d) n_r = 10 :=
by
  rw [n_r_eq, n_d_eq]
  exact nat.choose_succ_succ 3 2

end routes_from_A_to_B_l40_40359


namespace perpendicular_vectors_lambda_l40_40305

def a := (-3, 2)
def b := (-1, 0)
def dot_product (v1 : ℝ × ℝ) (v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors_lambda :
  let a := a
  let b := b
  let l := -1 / 7
  dot_product 
    (λ v1, l * v1.1 + b.1, l * v1.2 + b.2) 
    ((a.1 - 2 * b.1), (a.2 - 2 * b.2)) = 0 :=
by 
  sorry

end perpendicular_vectors_lambda_l40_40305


namespace hour_hand_pointing_correctly_l40_40730

/-- Define the given conditions -/
def angle_between_hands := 130
def minute_hand_position := 40

/-- Define the correct answer based on conditions -/
def correct_hour_hand_position := 3 + (40 / 60 : ℝ)

/-- The goal is to prove that the hour hand's position matches the calculated correct position -/
theorem hour_hand_pointing_correctly
  (angle_hand_minute: ℝ)
  (minute_at_position: ℕ)
  (h1 : angle_hand_minute = 130)
  (h2 : minute_at_position = 40) :
  minute_at_position / 60 + (angle_hand_minute + 130) / 360  = correct_hour_hand_position :=
by {
  sorry
}

end hour_hand_pointing_correctly_l40_40730


namespace car_travel_distance_l40_40361

theorem car_travel_distance :
  ∀ (train_speed : ℝ) (fraction : ℝ) (time_minutes : ℝ) (car_speed : ℝ) (distance : ℝ),
  train_speed = 90 →
  fraction = 5 / 6 →
  time_minutes = 30 →
  car_speed = fraction * train_speed →
  distance = car_speed * (time_minutes / 60) →
  distance = 37.5 :=
by
  intros train_speed fraction time_minutes car_speed distance
  intros h_train_speed h_fraction h_time_minutes h_car_speed h_distance
  sorry

end car_travel_distance_l40_40361


namespace prod_of_roots_eq_zero_l40_40755

theorem prod_of_roots_eq_zero :
  let P (x : ℂ) := ∏ k in (finset.range 15).map (λ n, n+1), (x - exp (2 * π * complex.I * (k:ℂ) / 17))
  ∧ let Q (x : ℂ) := ∏ j in (finset.range 12).map (λ n, n+1), (x - exp (2 * π * complex.I * (j:ℂ) / 13))
  in (∏ k in (finset.range 15).map (λ n, n+1), ∏ j in (finset.range 12).map (λ n, n+1), 
    (exp (2 * π * j * complex.I / 13) - exp (2 * π * k * complex.I / 17))) = 0 :=
by
  intro P Q
  sorry

end prod_of_roots_eq_zero_l40_40755


namespace coffee_break_l40_40571

variable {n : ℕ}
variables {participants : Finset ℕ} (hparticipants : participants.card = 14)
variables (left : Finset ℕ)
variables (stayed : Finset ℕ) (hstayed : stayed.card = 14 - 2 * left.card)

-- The overall proof problem statement:
theorem coffee_break (h : ∀ x ∈ stayed, ∃! y, (y ∈ left ∧ adjacent x y participants)) :
  left.card = 6 ∨ left.card = 8 ∨ left.card = 10 ∨ left.card = 12 := 
sorry

end coffee_break_l40_40571


namespace power_function_decreasing_l40_40377

theorem power_function_decreasing (m : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 0 < x → f x = (m^2 + m - 11) * x^(m - 1))
  (hm : m^2 + m - 11 > 0)
  (hm' : m - 1 < 0)
  (hx : 0 < 1):
  f (-1) = -1 := by 
sorry

end power_function_decreasing_l40_40377


namespace find_a1_a2_general_formula_perpendicular_vectors_l40_40296

section math_problem

variable {ℕ : Type} [Nat]

-- Define the sequence a_n and S_n
def a (n : ℕ) : ℝ := sorry -- depending on your desired sequence definition
def S (n : ℕ) : ℝ := (finset.range n).sum a

-- Conditions given in the problem
axiom pos_sequence : ∀ n : ℕ, a n > 0
axiom main_equation : ∀ n : ℕ, a n ^ 2 = 4 * S n - 2 * a n - 1

-- Questions transformed into Lean theorems to be proved

-- Part (1) of the problem
theorem find_a1_a2 : a 1 = 1 ∧ a 2 = 3 :=
sorry

-- Part (2) of the problem
theorem general_formula : ∀ n : ℕ, a n = 2 * n - 1 :=
sorry

-- Part (3) of the problem
theorem perpendicular_vectors : ∃ (m n : ℕ), (2 * a (n + 2), m) ⬝ (- a (n + 5), 3 + a n) = 0 :=
sorry

end math_problem

end find_a1_a2_general_formula_perpendicular_vectors_l40_40296


namespace bisection_method_applicable_l40_40167

-- Define the functions corresponding to each equation
def f₁ (x : ℝ) : ℝ := Real.log x + x
def f₂ (x : ℝ) : ℝ := Real.exp x - 3 * x
def f₃ (x : ℝ) : ℝ := x^3 - 3 * x + 1
def f₄ (x : ℝ) : ℝ := 4 * x^2 - 4 * Real.sqrt 5 * x + 5

-- Define the intervals and continuity requirement
theorem bisection_method_applicable :
  (∃ (a b : ℝ), f₁ a * f₁ b < 0 ∧ ∀ (x : ℝ), a ≤ x ∧ x ≤ b → ContinuousAt f₁ x) ∧
  (∃ (a b : ℝ), f₂ a * f₂ b < 0 ∧ ∀ (x : ℝ), a ≤ x ∧ x ≤ b → ContinuousAt f₂ x) ∧
  (∃ (a b : ℝ), f₃ a * f₃ b < 0 ∧ ∀ (x : ℝ), a ≤ x ∧ x ≤ b → ContinuousAt f₃ x) :=
by sorry

end bisection_method_applicable_l40_40167


namespace greatest_integer_value_of_y_l40_40286

theorem greatest_integer_value_of_y :
  ∃ (y : ℤ), (3 * |y| + 6 < 24) ∧ (∀ z : ℤ, 3 * |z| + 6 < 24 → z ≤ y) :=
begin
  sorry
end

end greatest_integer_value_of_y_l40_40286


namespace quadratic_prime_roots_l40_40239

theorem quadratic_prime_roots (k : ℕ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p + q = 101 → p * q = k → False :=
by
  sorry

end quadratic_prime_roots_l40_40239


namespace independent_events_l40_40150

def first_coin_heads : Prop := true
def second_coin_heads : Prop := true
def both_coins_same_face : Prop := (first_coin_heads ∧ second_coin_heads) ∨ (¬first_coin_heads ∧ ¬second_coin_heads)

theorem independent_events (A B C : Prop) (P : Prop → ℝ) [independent_A_B : P(A ∧ B) = P(A) * P(B)]
  [independent_B_C : P(B ∧ C) = P(B) * P(C)]
  [independent_A_C : P(A ∧ C) = P(A) * P(C)] :
  (P(A ∧ B) = P(A) * P(B)) ∧ (P(B ∧ C) = P(B) * P(C)) ∧ (P(A ∧ C) = P(A) * P(C)) :=
by
  sorry

end independent_events_l40_40150


namespace total_weight_of_family_l40_40130

theorem total_weight_of_family (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 40) : M + D + C = 160 :=
sorry

end total_weight_of_family_l40_40130


namespace evaluate_exponent_product_l40_40276

theorem evaluate_exponent_product : 3^5 * 4^5 = 248832 := by
  have h1 : 12^5 = 248832 := sorry
  have h2 : 3 * 4 = 12 := rfl
  rw [←h2, pow_mul],
  exact h1

end evaluate_exponent_product_l40_40276


namespace chord_length_hyperbola_asymptote_circle_intersection_l40_40829

theorem chord_length_hyperbola_asymptote_circle_intersection {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (a : ℝ) * (b : ℝ) = 2 * (a^2)) (h₄ : dist (2, 3) (line.mk (2, 1) 2) = 1 / sqrt 5) : 
  let A : point ℝ := ...
  let B : point ℝ := ...
  dist A B = 4 * sqrt 5 / 5 :=
sorry

end chord_length_hyperbola_asymptote_circle_intersection_l40_40829


namespace final_cost_l40_40465

-- Definitions of initial conditions
def initial_cart_total : ℝ := 54.00
def discounted_item_original_price : ℝ := 20.00
def discount_rate1 : ℝ := 0.20
def coupon_rate : ℝ := 0.10

-- Prove the final cost after applying discounts
theorem final_cost (initial_cart_total discounted_item_original_price discount_rate1 coupon_rate : ℝ) :
  let discounted_price := discounted_item_original_price * (1 - discount_rate1)
  let total_after_first_discount := initial_cart_total - discounted_price
  let final_total := total_after_first_discount * (1 - coupon_rate)
  final_total = 45.00 :=
by 
  sorry

end final_cost_l40_40465


namespace sum_of_real_solutions_l40_40292

theorem sum_of_real_solutions :
  (∃ x : ℝ, (x - 3) / (x^2 + 5 * x + 3) = (x - 7) / (x^2 - 8 * x + 7)) →
  (sum (roots ((x - 3) / (x^2 + 5 * x + 3) = (x - 7) / (x^2 - 8 * x + 7))) = -7) := sorry

end sum_of_real_solutions_l40_40292


namespace find_range_g_l40_40791

noncomputable def g (x : ℝ) : ℝ := (Real.sin x)^6 - 3 * Real.sin x * Real.cos x + (Real.cos x)^6

theorem find_range_g :
  let g := λ x : ℝ, (Real.sin x)^6 - 3 * Real.sin x * Real.cos x + (Real.cos x)^6
  ∃ a b : ℝ, ∀ x : ℝ, a ≤ g(x) ∧ g(x) ≤ b ∧ a = 0 ∧ b = 7/4 :=
by {
  sorry
}

end find_range_g_l40_40791


namespace transfer_people_eq_l40_40197

theorem transfer_people_eq : ∃ x : ℕ, 22 + x = 2 * (26 - x) := 
by 
  -- hypothesis and equation statement
  sorry

end transfer_people_eq_l40_40197


namespace right_triangle_area_l40_40281

-- Define the conditions a = 4/3 * b and a = 2/3 * c
variable (a b c : ℝ)
hypothesis h1 : a = (4 / 3) * b
hypothesis h2 : a = (2 / 3) * c

-- Define the theorem stating that the area of the right triangle is 2/3
theorem right_triangle_area : (1 / 2) * a * b = (2 / 3) := by
  sorry

end right_triangle_area_l40_40281


namespace constant_term_expansion_l40_40389

-- Defining the binomial theorem term
noncomputable def binomial_coeff (n k : ℕ) : ℕ := 
  Nat.choose n k

-- The general term of the binomial expansion (2sqrt(x) - 1/x)^6
noncomputable def general_term (r : ℕ) (x : ℝ) : ℝ :=
  binomial_coeff 6 r * (-1)^r * (2^(6-r)) * x^((6 - 3 * r) / 2)

-- Problem statement: Show that the constant term in the expansion is 240
theorem constant_term_expansion :
  (∃ r : ℕ, (6 - 3 * r) / 2 = 0 ∧ 
            general_term r arbitrary = 240) :=
sorry

end constant_term_expansion_l40_40389


namespace six_digit_numbers_with_zero_l40_40929

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40929


namespace average_hit_targets_value_average_hit_targets_ge_half_l40_40268

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end average_hit_targets_value_average_hit_targets_ge_half_l40_40268


namespace determine_x_l40_40289

theorem determine_x (x : ℝ) : (16^(x-2) / 2^(x-2) = 32^(3*x)) → x = -1/2 := by
  sorry

end determine_x_l40_40289


namespace bus_length_l40_40685

theorem bus_length (d : ℕ) (t1 : ℕ) (t2 : ℕ) (h_d : d = 12000) (h_t1 : t1 = 300) (h_t2 : t2 = 5) : 
  d / t1 * t2 = 200 :=
by
  rw [h_d, h_t1, h_t2]
  norm_num

end bus_length_l40_40685


namespace find_AM_length_l40_40383

-- Define the right triangle shapes and properties
variables {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variables (h d : ℝ) (AM BM : ℝ)

-- Conditions specified in the problem
-- 1. Triangle ABC is a right triangle with AC as the hypotenuse
-- 2. Point M is on hypotenuse AC such that AM = x.
-- 3. BM is perpendicular to AC.
-- 4. BM equals AM (BM = x).
-- 5. AB = h
-- 6. BC = d

theorem find_AM_length (h : ℝ) (d : ℝ) (x : ℝ) 
  (h_pythagoras : h^2 = (h - x)^2 + d^2) :
  x = (h^2 - d^2) / (2 * h) :=
by
  sorry

end find_AM_length_l40_40383


namespace prod_eq_one_l40_40751

noncomputable def P (x : ℂ) : ℂ :=
  ∏ k in finset.range 15, (x - complex.exp (2 * real.pi * complex.I * k / 17))

noncomputable def Q (x : ℂ) : ℂ :=
  ∏ j in finset.range 12, (x - complex.exp (2 * real.pi * complex.I * j / 13))

theorem prod_eq_one :
  (∏ k in finset.range 15, ∏ j in finset.range 12, (complex.exp (2 * real.pi * complex.I * j / 13) - complex.exp (2 * real.pi * complex.I * k / 17))) = 1 := 
by
  sorry

end prod_eq_one_l40_40751


namespace prime_sum_1_to_50_l40_40068

theorem prime_sum_1_to_50 :
  let primes := { p : ℕ | p.prime ∧ 1 ≤ p ∧ p ≤ 50 }
  in (2 ∈ primes) ∧ (47 ∈ primes) ∧ (∀ p ∈ primes, p = 2 ∨ p = 47)
   → 2 + 47 = 49 := 
by
  intros _ h_smallest_prime h_largest_prime _
  exact eq.refl 49

end prime_sum_1_to_50_l40_40068


namespace solve_inequality_l40_40839

def f (x : ℝ) : ℝ := |x + 1| - |x - 3|

theorem solve_inequality : ∀ x : ℝ, |f x| ≤ 4 :=
by
  intro x
  sorry

end solve_inequality_l40_40839


namespace find_m_from_root_l40_40321

theorem find_m_from_root (m : ℝ) : (x : ℝ) = 1 → x^2 + m * x + 2 = 0 → m = -3 :=
by
  sorry

end find_m_from_root_l40_40321


namespace sale_first_month_l40_40696

/-- Given the sales in the second, third, and fourth month and the average sale over four months,
prove that the sale in the first month is Rs. 2500. -/
theorem sale_first_month
  (sale_second_month : ℝ)
  (sale_third_month : ℝ)
  (sale_fourth_month : ℝ)
  (average_sale : ℝ)
  (num_months : ℕ)
  (total_sales : ℝ := average_sale * num_months)
  (sale_first_month : ℝ := total_sales - (sale_second_month + sale_third_month + sale_fourth_month)) :
  sale_first_month = 2500 :=
by
  -- Given conditions
  let sale_second_month : ℝ := 4000
  let sale_third_month : ℝ := 3540
  let sale_fourth_month : ℝ := 1520
  let average_sale : ℝ := 2890
  let num_months : ℕ := 4
  have h1 : total_sales = average_sale * num_months := rfl
  have total_sales : ℝ := 11560  -- 2890 * 4
  have sale_first_month : ℝ := 11560 - (4000 + 3540 + 1520)

  -- The proof to show the first month's sale is 2500
  have : sale_first_month = 2500 := rfl
  exact this


end sale_first_month_l40_40696


namespace six_digit_numbers_with_zero_l40_40875

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40875


namespace truck_speed_difference_l40_40707

variables
  (t_paved t_dirt : ℝ) -- time on paved and dirt roads respectively
  (d_total : ℝ) -- total distance of the trip
  (v_dirt : ℝ) -- speed on the dirt road

-- Assign values as per the conditions
noncomputable def t_paved_val : ℝ := 2
noncomputable def t_dirt_val : ℝ := 3
noncomputable def d_total_val : ℝ := 200
noncomputable def v_dirt_val : ℝ := 32.0

theorem truck_speed_difference :
  let d_dirt := v_dirt_val * t_dirt_val in
  let d_paved := d_total_val - d_dirt in
  let v_paved := d_paved / t_paved_val in
  v_paved - v_dirt_val = 20 :=
by
  let d_dirt := v_dirt_val * t_dirt_val
  let d_paved := d_total_val - d_dirt
  let v_paved := d_paved / t_paved_val
  have h : v_paved - v_dirt_val = 20 := by sorry
  exact h

end truck_speed_difference_l40_40707


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40917

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40917


namespace average_distance_is_600_l40_40441

-- Definitions based on the conditions
def one_lap_distance : ℕ := 200
def johnny_lap_count : ℕ := 4
def mickey_lap_count : ℕ := johnny_lap_count / 2

-- Total distances run by Johnny and Mickey
def johnny_distance : ℕ := johnny_lap_count * one_lap_distance
def mickey_distance : ℕ := mickey_lap_count * one_lap_distance

-- Sum of distances
def total_distance : ℕ := johnny_distance + mickey_distance

-- Number of people
def number_of_people : ℕ := 2

-- Average distance calculation
def average_distance : ℕ := total_distance / number_of_people

-- The theorem to prove: the average distance run by Johnny and Mickey
theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end average_distance_is_600_l40_40441


namespace edric_days_per_week_l40_40272

variable (monthly_salary : ℝ) (hours_per_day : ℝ) (hourly_rate : ℝ) (weeks_per_month : ℝ)
variable (days_per_week : ℝ)

-- Defining the conditions
def monthly_salary_condition : Prop := monthly_salary = 576
def hours_per_day_condition : Prop := hours_per_day = 8
def hourly_rate_condition : Prop := hourly_rate = 3
def weeks_per_month_condition : Prop := weeks_per_month = 4

-- Correct answer
def correct_answer : Prop := days_per_week = 6

-- Proof problem statement
theorem edric_days_per_week :
  monthly_salary_condition monthly_salary ∧
  hours_per_day_condition hours_per_day ∧
  hourly_rate_condition hourly_rate ∧
  weeks_per_month_condition weeks_per_month →
  correct_answer days_per_week :=
by
  sorry

end edric_days_per_week_l40_40272


namespace problem_solution_l40_40088

def prop_p (a b c : ℝ) : Prop := a < b → a * c^2 < b * c^2

def prop_q : Prop := ∃ x : ℝ, x^2 - x + 1 ≤ 0

theorem problem_solution : (p ∨ ¬q) := sorry

end problem_solution_l40_40088


namespace total_savings_l40_40078

def weekly_savings : ℕ := 15
def weeks_per_cycle : ℕ := 60
def number_of_cycles : ℕ := 5

theorem total_savings :
  (weekly_savings * weeks_per_cycle) * number_of_cycles = 4500 := 
sorry

end total_savings_l40_40078


namespace incorrect_statement_D_l40_40172

-- Define conditions
def condition_A (Q : Type) [Quadrilateral Q] :=
  (∀ (q : Q), (∀ (s1 s2 : Side Q), opposite Sides s1 s2 → length s1 = length s2) → Parallelogram Q)

def condition_B (Q : Type) [Quadrilateral Q] :=
  (∀ (q : Q), (∀ (i : Angle Q), interior i → equal_angle i 90) → Rectangle Q)

def condition_C (Q : Type) [Quadrilateral Q] :=
  (∀ (q : Q), (∀ (s : Side Q), equal_sides s) → Rhombus Q)

def condition_D (Q : Type) [Quadrilateral Q] :=
  (∀ (q : Q), (perpendicular_and_bisecting_diagonals Q) → Square Q)

-- Correct answer: Statement D is incorrect
theorem incorrect_statement_D (Q : Type) [Quadrilateral Q] :
  (condition_A Q) →
  (condition_B Q) →
  (condition_C Q) →
  ¬ (condition_D Q) :=
sorry

end incorrect_statement_D_l40_40172


namespace quadratic_root_m_value_l40_40320

theorem quadratic_root_m_value (m : ℝ) (x : ℝ) (h : x = 1) (hx : x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end quadratic_root_m_value_l40_40320


namespace andy_demerits_for_joke_l40_40729

def max_demerits := 50
def demerits_late_per_instance := 2
def instances_late := 6
def remaining_demerits := 23
def total_demerits := max_demerits - remaining_demerits
def demerits_late := demerits_late_per_instance * instances_late
def demerits_joke := total_demerits - demerits_late

theorem andy_demerits_for_joke : demerits_joke = 15 := by
  sorry

end andy_demerits_for_joke_l40_40729


namespace six_digit_numbers_with_zero_l40_40964

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40964


namespace concyclic_points_l40_40677

noncomputable def midpoint (S : Point) (A B : Point) := (dist S A = dist S B) ∧ (S not_on (segment A B))

theorem concyclic_points
  (Γ : Circle)
  (C D S A B P Q : Point)
  (h1 : C ∈ Γ)
  (h2 : D ∈ Γ)
  (h3 : midpoint S C D)
  (h4 : S ≠ A ∧ S ≠ B)
  (h5 : A ∈ Γ ∧ B ∈ Γ)
  (h6 : S ∉ arc C D) -- S is not on the arc that contains A and B
  (h7 : line_through A S ≠ ∅ ∧ point_on_line P (line_through A S))
  (h8 : line_through B S ≠ ∅ ∧ point_on_line Q (line_through B S))
  (h9 : P ∈ line_through C D)
  (h10 : Q ∈ line_through C D) :
  concyclic {A, P, B, Q} :=
  sorry

end concyclic_points_l40_40677


namespace scrabble_champions_l40_40391

noncomputable def num_champions : Nat := 25
noncomputable def male_percentage : Nat := 40
noncomputable def bearded_percentage : Nat := 40
noncomputable def bearded_bald_percentage : Nat := 60
noncomputable def non_bearded_bald_percentage : Nat := 30

theorem scrabble_champions :
  let male_champions := (male_percentage * num_champions) / 100
  let bearded_champions := (bearded_percentage * male_champions) / 100
  let bearded_bald_champions := (bearded_bald_percentage * bearded_champions) / 100
  let bearded_hair_champions := bearded_champions - bearded_bald_champions
  let non_bearded_champions := male_champions - bearded_champions
  let non_bearded_bald_champions := (non_bearded_bald_percentage * non_bearded_champions) / 100
  let non_bearded_hair_champions := non_bearded_champions - non_bearded_bald_champions
  bearded_bald_champions = 2 ∧ 
  bearded_hair_champions = 2 ∧ 
  non_bearded_bald_champions = 1 ∧ 
  non_bearded_hair_champions = 5 :=
by
  sorry

end scrabble_champions_l40_40391


namespace problem1_problem2_problem3_l40_40425

-- Proof Problem 1
theorem problem1 (f : ℝ → ℝ) (h : ∀ x, f x = log (x + 2)) :
  {x | f (1/x) > 1} = Ioo 0 (1/8) :=
sorry

-- Proof Problem 2
theorem problem2 (f : ℝ → ℝ) (λ : ℝ)
  (h1 : f 0 = 1)
  (h2 : ∃ x ∈ Icc 2 3, f x = (1/real.sqrt 2)^x + λ) :
  λ ∈ Icc (log 12 - 1/2) (log 13 - real.sqrt 2 / 4) :=
sorry

-- Proof Problem 3
theorem problem3 (f : ℝ → ℝ) (x : ℝ)
  (h1 : f 98 = 2)
  (h2 : ∀ n : ℕ, f (real.cos (2^n * x)) < log 2) :
  ∃ k : ℤ, x = (2*k + 2/3)*π ∨ x = (2*k + 4/3)*π :=
sorry

end problem1_problem2_problem3_l40_40425


namespace circle_area_l40_40522

theorem circle_area (θ : ℝ) : 
  let r := 3 * Real.cos θ - 4 * Real.sin θ in
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let center := (3 / 2 : ℝ, -2 : ℝ) in
  let radius := 5 / 2 in
  (x - center.1)^2 + (y - center.2)^2 = radius^2 →
  ∃ A : ℝ, A = (radius^2) * Real.pi ∧ A = 25 * Real.pi / 4 := 
by sorry

end circle_area_l40_40522


namespace zeros_between_decimal_point_and_first_nonzero_digit_l40_40653

theorem zeros_between_decimal_point_and_first_nonzero_digit :
  let n := (7 : ℚ) / 8000 in
  decimal_zeros_between_decimal_point_and_first_nonzero_digit n = 3 :=
by
  sorry

end zeros_between_decimal_point_and_first_nonzero_digit_l40_40653


namespace solve_problem_l40_40387

-- Define the parametric equations of the line
def line (t : ℝ) : ℝ × ℝ := (-(3 / 5) * t + 2, (4 / 5) * t)

-- Define the polar equation of the circle
def polar_circle (theta : ℝ) (a : ℝ) : ℝ := a * Real.sin theta

-- The Cartesian equation of the line
def cartesian_line : Prop :=
  ∀ x y : ℝ, (∃ t : ℝ, x = -(3 / 5) * t + 2 ∧ y = (4 / 5) * t) ↔ 4 * x + 3 * y - 8 = 0

-- The Cartesian equation of the circle
def cartesian_circle (a : ℝ) : Prop :=
  ∀ x y : ℝ, (∃ theta : ℝ, x^2 + y^2 - a * y = 0) ↔ (a ≠ 0 → x^2 + (y - a / 2)^2 = (a^2 / 4))

-- The value of 'a' given the chord length condition
def value_of_a (a : ℝ) : Prop :=
  let r := a / 2
  let d := abs ((3 * a / 2 - 8) / 5)
  let chord_length := sqrt 3 * r 
  2 * sqrt (r^2 - d^2) = chord_length → a = 32 ∨ a = 32 / 11

-- The main theorem
theorem solve_problem :
  cartesian_line ∧ (∀ a : ℝ, cartesian_circle a ∧ value_of_a a) :=
by sorry

end solve_problem_l40_40387


namespace area_calculation_l40_40032

noncomputable def area_of_right_triangle : ℝ :=
  let s := (1 + Real.sqrt 3) / 2 in
  (1 / 2) * s * (s + 1)

theorem area_calculation :
  let area := area_of_right_triangle in
  area = (3 / 4) + (Real.sqrt 3 / 2) :=
by
  sorry

end area_calculation_l40_40032


namespace six_digit_numbers_with_zero_l40_40952

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40952


namespace value_of_7th_term_l40_40205

noncomputable def arithmetic_sequence_a1_d_n (a1 d n a7 : ℝ) : Prop := 
  ((5 * a1 + 10 * d = 68) ∧ 
   (5 * (a1 + (n - 1) * d) - 10 * d = 292) ∧
   (n / 2 * (2 * a1 + (n - 1) * d) = 234) ∧ 
   (a1 + 6 * d = a7))

theorem value_of_7th_term (a1 d n a7 : ℝ) : 
  arithmetic_sequence_a1_d_n a1 d n 18 := 
by
  simp [arithmetic_sequence_a1_d_n]
  sorry

end value_of_7th_term_l40_40205


namespace probability_interval_0_2a_l40_40016

noncomputable def normalDist (μ σ : ℝ) := sorry

variables (a σ : ℝ)
variables (h₀ : a > 0) (h₁ : σ > 0)
variables (ξ : ℝ → ℝ)
variables (ξ_dist : ξ = normalDist a σ^2)
variables (prob_interval_a : ∀ x, x > 0 → x < a → P(x) = 0.3)

theorem probability_interval_0_2a :
  P(0 < ξ < 2 * a) = 0.6 :=
sorry

end probability_interval_0_2a_l40_40016


namespace cos_alpha_plus_cos_beta_inequality_l40_40310

theorem cos_alpha_plus_cos_beta_inequality (α β : ℝ) (hα : 0 < α) (hα₂ : α < π / 2) (hβ : 0 < β) (hβ₂ : β < π / 2) :
  cos α + cos β + sqrt 2 * sin α * sin β ≤ 3 * sqrt 2 / 2 := 
sorry

end cos_alpha_plus_cos_beta_inequality_l40_40310


namespace six_digit_numbers_with_at_least_one_zero_l40_40968

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40968


namespace vertex_of_parabola_l40_40154

theorem vertex_of_parabola :
  ∃ (x y : ℝ), y^2 - 8*x + 6*y + 17 = 0 ∧ (x, y) = (1, -3) :=
by
  use 1, -3
  sorry

end vertex_of_parabola_l40_40154


namespace average_distance_is_600_l40_40439

-- Definitions based on the conditions
def one_lap_distance : ℕ := 200
def johnny_lap_count : ℕ := 4
def mickey_lap_count : ℕ := johnny_lap_count / 2

-- Total distances run by Johnny and Mickey
def johnny_distance : ℕ := johnny_lap_count * one_lap_distance
def mickey_distance : ℕ := mickey_lap_count * one_lap_distance

-- Sum of distances
def total_distance : ℕ := johnny_distance + mickey_distance

-- Number of people
def number_of_people : ℕ := 2

-- Average distance calculation
def average_distance : ℕ := total_distance / number_of_people

-- The theorem to prove: the average distance run by Johnny and Mickey
theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end average_distance_is_600_l40_40439


namespace parabola_point_distance_condition_l40_40375

theorem parabola_point_distance_condition :
  ∃ y0 : ℝ, (∃ x0 : ℝ, x0^2 = 28 * y0) ∧ (real.sqrt ((x0 - 0)^2 + (y0 - 7)^2) = 3 * y0) → y0 = 7 / 2 :=
begin
  sorry
end

end parabola_point_distance_condition_l40_40375


namespace min_value_of_expression_l40_40087

theorem min_value_of_expression (a b : ℕ) (h1 : 0 < a) (h2 : a < 8) (h3 : 0 < b) (h4 : b < 8) : 
  2 * a - a * b ≥ -35 :=
by
  sorry

end min_value_of_expression_l40_40087


namespace abs_neg_two_is_two_l40_40504

def absolute_value (x : ℝ) : ℝ := if x < 0 then -x else x

theorem abs_neg_two_is_two : absolute_value (-2) = 2 :=
by
  sorry

end abs_neg_two_is_two_l40_40504


namespace coffee_break_l40_40589

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end coffee_break_l40_40589


namespace cost_equivalence_min_sets_of_A_l40_40194

noncomputable def cost_of_B := 120
noncomputable def cost_of_A := cost_of_B + 30

theorem cost_equivalence (x : ℕ) :
  (1200 / (x + 30) = 960 / x) → x = 120 :=
by
  sorry

theorem min_sets_of_A :
  ∀ m : ℕ, (150 * m + 120 * (20 - m) ≥ 2800) ↔ m ≥ 14 :=
by
  sorry

end cost_equivalence_min_sets_of_A_l40_40194


namespace rationalized_sum_l40_40096

theorem rationalized_sum : 
  ∃ A B C D : ℤ, 
    (D > 0) ∧ 
    (∀ p : ℕ, prime p → p * p ∣ B → false) ∧ 
    (Int.gcd A (Int.gcd C D) = 1) ∧ 
    (A * (Int.ofNat B).sqrt + C) / D = 7 * (3 - (Int.ofNat B).sqrt) / (3 + (Int.ofNat B).sqrt) ∧ 
    A + B + C + D = 23 := sorry

end rationalized_sum_l40_40096


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40948

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40948


namespace sum_reciprocal_b_eq_2015_div_2016_l40_40768

def average_reciprocal(n : ℕ) (p : Fin n → ℝ) : ℝ :=
  n / (p.sum (λ i, p i))

def a (n : ℕ) : ℕ :=
  match n with
  | 1       => 3
  | (n+1) => 4 * (n + 1) - 1

def b (n : ℕ) : ℕ :=
  (a n + 1) / 4

theorem sum_reciprocal_b_eq_2015_div_2016 :
  (∑ k in Finset.range 2015, 1 / (b k * b (k + 1))) = 2015 / 2016 :=
  sorry

end sum_reciprocal_b_eq_2015_div_2016_l40_40768


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40940

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40940


namespace question_eq_three_l40_40825

open Equiv Function

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
variables {a b G P Q : V} {m n : ℝ}

-- Conditions
def point_is_centroid (G : V) (a b : V) : Prop :=
  G = (1/3 : ℝ) • (a + b)

def line_intersects (P Q G a b : V) (m n : ℝ) : Prop :=
  P = m • a ∧ Q = n • b ∧ G ∈ line_through P Q

-- Prove that
theorem question_eq_three
  (h_centroid : point_is_centroid G a b)
  (h_intersects : line_intersects P Q G a b m n) :
  1/m + 1/n = 3 := 
sorry

end question_eq_three_l40_40825


namespace sum_odd_numbers_l40_40333

-- Define the given conditions as separate definitions
def equation1 : Prop := 2^2 - 1^2 = 3
def equation2 : Prop := 3^2 - 2^2 = 5
def equation3 : Prop := 4^2 - 3^2 = 7

-- Define the pattern for the nth equation
def nth_equation (n : ℕ) : Prop := (n + 1)^2 - n^2 = 2 * n + 1

-- State the main proof problem
theorem sum_odd_numbers : 
    (∀ {n : ℕ}, nth_equation n) → 
    equation1 → equation2 → equation3 → 
    1 + 3 + 5 + 7 + ... + 2005 + 2007 = 1008016 := 
by
  -- Proof to be completed
  sorry

end sum_odd_numbers_l40_40333


namespace abs_neg_two_is_two_l40_40502

def absolute_value (x : ℝ) : ℝ := if x < 0 then -x else x

theorem abs_neg_two_is_two : absolute_value (-2) = 2 :=
by
  sorry

end abs_neg_two_is_two_l40_40502


namespace six_digit_numbers_with_zero_l40_40989

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40989


namespace six_digit_numbers_with_zero_l40_40985

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40985


namespace triangle_equilibrium_or_relation_l40_40395

theorem triangle_equilibrium_or_relation
  (a b c : ℝ) 
  (h : ∃ α β γ : angle, α + β + γ = π ∧ (sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)) = (2 * b * c * cos (α / 2)) / (b + c)) :
  (a = b ∧ b = c ∧ c = a) ∨ (a^2 = 2 * (b + c)^2) :=
sorry

end triangle_equilibrium_or_relation_l40_40395


namespace coffee_participants_l40_40601

noncomputable def participants_went_for_coffee (total participants : ℕ) (stay : ℕ) : Prop :=
  (0 < stay) ∧ (stay < 14) ∧ (stay % 2 = 0) ∧ ∃ k : ℕ, stay = 2 * k

theorem coffee_participants :
  ∀ (total participants : ℕ), total participants = 14 →
  (∀ participant, ((∃ k : ℕ, stay = 2 * k) → (0 < stay) ∧ (stay < 14))) →
  participants_went_for_coffee total_participants (14 - stay) =
  (stay = 12) ∨ (stay = 10) ∨ (stay = 8) ∨ (stay = 6) :=
by
  sorry

end coffee_participants_l40_40601


namespace participants_coffee_l40_40587

theorem participants_coffee (n : ℕ) (h1 : n = 14) (h2 : 0 < (14 - 2 * (n / 2)) < 14) : 
  ∃ (k : ℕ), k ∈ {6, 8, 10, 12} ∧ k = (14 - 2 * (n / 2)) :=
by sorry

end participants_coffee_l40_40587


namespace problem_l40_40378

variable (p q : Prop)

theorem problem (h₁ : ¬ p) (h₂ : ¬ (p ∧ q)) : ¬ (p ∨ q) := sorry

end problem_l40_40378


namespace smallest_prime_divisor_of_sum_l40_40621

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end smallest_prime_divisor_of_sum_l40_40621


namespace period_preperiod_le_totient_l40_40424

theorem period_preperiod_le_totient (m n : ℕ) (h_coprime : Nat.coprime m n) :
  let φ := Nat.totient n
  in (length_preperiod m n + length_period m n) ≤ φ := 
sorry

end period_preperiod_le_totient_l40_40424


namespace smallest_prime_divisor_of_sum_l40_40623

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end smallest_prime_divisor_of_sum_l40_40623


namespace find_fivefold_f_of_one_plus_i_l40_40297

def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 - 1

theorem find_fivefold_f_of_one_plus_i :
  f (f (f (f (f (1 + complex.i))))) = -48 + 249984 * complex.i :=
by
  sorry

end find_fivefold_f_of_one_plus_i_l40_40297


namespace complement_union_l40_40349

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

theorem complement_union (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (hU : U = univ) 
(hA : A = {x : ℝ | 0 < x}) 
(hB : B = {x : ℝ | -3 < x ∧ x < 1}) : 
compl (A ∪ B) = {x : ℝ | x ≤ -3} :=
by
  sorry

end complement_union_l40_40349


namespace last_digit_of_expression_l40_40004

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_expression (n : ℕ) : last_digit (n ^ 9999 - n ^ 5555) = 0 :=
by
  sorry

end last_digit_of_expression_l40_40004


namespace sum_of_cubes_l40_40371

theorem sum_of_cubes (x y : ℝ) (hx : x + y = 10) (hxy : x * y = 12) : x^3 + y^3 = 640 := 
by
  sorry

end sum_of_cubes_l40_40371


namespace harmonic_mean_ordered_pairs_l40_40300

theorem harmonic_mean_ordered_pairs :
  ∃ n : ℕ, n = 23 ∧ ∀ (a b : ℕ), 
    0 < a ∧ 0 < b ∧ a < b ∧ (2 * a * b = 2 ^ 24 * (a + b)) → n = 23 :=
by sorry

end harmonic_mean_ordered_pairs_l40_40300


namespace smallest_period_f_l40_40140

noncomputable def smallest_positive_period (f : ℝ → ℝ) := 
inf {T : ℝ | T > 0 ∧ ∀ x, f(x + T) = f(x)}

def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + cos (2 * x)

theorem smallest_period_f : smallest_positive_period f = π :=
sorry

end smallest_period_f_l40_40140


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40943

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40943


namespace MarthaEndBlocks_l40_40104

theorem MarthaEndBlocks (start_blocks found_blocks total_blocks : ℕ) 
  (h₁ : start_blocks = 11)
  (h₂ : found_blocks = 129) : 
  total_blocks = 140 :=
by
  sorry

end MarthaEndBlocks_l40_40104


namespace triangle_angle_determinant_zero_l40_40421

theorem triangle_angle_determinant_zero (θ φ ψ : ℝ) (h : θ + φ + ψ = Real.pi) : 
  Matrix.det !![![Real.cos θ, Real.sin θ, 1], ![Real.cos φ, Real.sin φ, 1], ![Real.cos ψ, Real.sin ψ, 1]] = 0 :=
by 
  sorry

end triangle_angle_determinant_zero_l40_40421


namespace average_hit_targets_formula_average_hit_targets_ge_half_l40_40264

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

end average_hit_targets_formula_average_hit_targets_ge_half_l40_40264


namespace xiaoming_score_is_88_l40_40147

theorem xiaoming_score_is_88 (s₁ s₂ s₃ s₄ s₅ : ℕ) (hx : multiset.Nodup [83, 86, 88, 91, 93]) 
  (hmed : multiset.median ([83, 86, 88, 91, 93] ++ [s₆]) = s₆)
  (hmode : multiset.mode ([83, 86, 88, 91, 93] ++ [s₆]) = s₆) : 
  s₆ = 88 := 
by { sorry }

end xiaoming_score_is_88_l40_40147


namespace smallest_prime_divisor_of_n_is_2_l40_40637

def a := 3^19
def b := 11^13
def n := a + b

theorem smallest_prime_divisor_of_n_is_2 : nat.prime_divisors n = {2} :=
by
  -- The proof is not needed as per instructions.
  -- We'll add a placeholder sorry to complete the statement.
  sorry

end smallest_prime_divisor_of_n_is_2_l40_40637


namespace total_age_proof_l40_40494

noncomputable def total_age : ℕ :=
  let susan := 15
  let arthur := susan + 2
  let bob := 11
  let tom := bob - 3
  let emily := susan / 2
  let david := (arthur + tom + emily) / 3
  susan + arthur + tom + bob + emily + david

theorem total_age_proof : total_age = 70 := by
  unfold total_age
  sorry

end total_age_proof_l40_40494


namespace karl_tshirt_price_l40_40403

theorem karl_tshirt_price:
  ∃ T : ℝ, (2 * T + 4 + 4 * 6 + 6 * (T / 2) = 53) ∧ T = 5 :=
by
  use 5
  have h1: 2 * 5 + 4 + 4 * 6 + 6 * (5 / 2) = 53 :=
    by norm_num
  split
  · exact h1
  · refl

end karl_tshirt_price_l40_40403


namespace product_of_roots_of_unity_l40_40758

theorem product_of_roots_of_unity : 
  ∏ k in Finset.range 1 16, ∏ j in Finset.range 1 13, (Complex.exp(2 * Real.pi * j * Complex.I / 13) - Complex.exp(2 * Real.pi * k * Complex.I / 17)) = 0 :=
sorry

end product_of_roots_of_unity_l40_40758


namespace method_of_continued_proportion_eq_euclidean_algorithm_gcd_l40_40513

/--
The Method of continued proportion used during the Song and Yuan dynasties
is the same as the Euclidean algorithm in terms of finding the greatest common
divisor (GCD) of two numbers.
-/
theorem method_of_continued_proportion_eq_euclidean_algorithm_gcd : 
  ∀ (a b : ℕ), method_of_continued_proportion a b = gcd a b := 
by
  sorry

end method_of_continued_proportion_eq_euclidean_algorithm_gcd_l40_40513


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40949

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40949


namespace sequence_sum_property_l40_40022

open Nat

-- Definitions based on problem conditions
def seq_incr (a : ℕ → ℕ) := ∀ n, a n < a (n + 1)

def seq_divisor (a : ℕ → ℕ) := ∀ n ≥ 2002, a n ∣ (∑ i in range (n-1), a i)

-- The main theorem to prove
theorem sequence_sum_property (a : ℕ → ℕ) (N : ℕ) (h_incr : seq_incr a) (h_div : seq_divisor a) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 1) = (∑ i in range n, a i) :=
by
  sorry

end sequence_sum_property_l40_40022


namespace range_of_a_l40_40339

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x + x^2 else x - x^2

theorem range_of_a (a : ℝ) : (∀ x, -1/2 ≤ x ∧ x ≤ 1/2 → f (x^2 + 1) > f (a * x)) ↔ -5/2 < a ∧ a < 5/2 := 
sorry

end range_of_a_l40_40339


namespace abs_neg_two_is_two_l40_40503

def absolute_value (x : ℝ) : ℝ := if x < 0 then -x else x

theorem abs_neg_two_is_two : absolute_value (-2) = 2 :=
by
  sorry

end abs_neg_two_is_two_l40_40503


namespace six_digit_numbers_with_zero_l40_40953

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40953


namespace max_edges_no_mono_chromatic_cycles_l40_40288

theorem max_edges_no_mono_chromatic_cycles (V : Finset ℕ) (E : ℕ)
  (hV : V.card = 60)
  (hE : E = V.card)
  (colors : Finset (V × V) → ℕ)
  (coloring : ∀ e ∈ Finset.powersetLen 2 V, colors e = 0 ∨ colors e = 1)
  (no_mono_3_cycle : ∀ (u v w : V), u ≠ v → v ≠ w → u ≠ w → ¬(colors {u, v} = colors {v, w} ∧ colors {v, w} = colors {u, w}))
  (no_mono_5_cycle : ∀ (u v w x y : V), u ≠ v → v ≠ w → w ≠ x → x ≠ y → u ≠ y →
    ¬(colors {u, v} = colors {v, w} ∧ colors {v, w} = colors {w, x} ∧ colors {w, x} = colors {x, y} ∧ colors {x, y} = colors {u, y})) :
  E ≤ 1350 := by
  sorry

end max_edges_no_mono_chromatic_cycles_l40_40288


namespace work_done_in_five_days_l40_40176

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 11
def work_rate_B : ℚ := 1 / 5
def work_rate_C : ℚ := 1 / 55

-- Define the work done in a cycle of 2 days
def work_one_cycle : ℚ := (work_rate_A + work_rate_B) + (work_rate_A + work_rate_C)

-- The total work needed to be done is 1
def total_work : ℚ := 1

-- The number of days in a cycle of 2 days
def days_per_cycle : ℕ := 2

-- Proving that the work will be done in exactly 5 days
theorem work_done_in_five_days :
  ∃ n : ℕ, n = 5 →
  n * (work_rate_A + work_rate_B) + (n-1) * (work_rate_A + work_rate_C) = total_work :=
by
  -- Sorry to skip the detailed proof steps
  sorry

end work_done_in_five_days_l40_40176


namespace greatest_possible_difference_l40_40558

theorem greatest_possible_difference (d1 d2 : ℕ) (h1 : d1 = 0 ∨ d1 = 5) (h2 : d2 = 0 ∨ d2 = 5) : ∃ d_max, d_max = 5 :=
by 
  -- Let us consider the units digits that make 740 to 745 multiples of 5, i.e., 0 and 5.
  use 5
  sorry

end greatest_possible_difference_l40_40558


namespace path_count_l40_40798

/-- Paths from (0,0) to (m,n) such that at every intermediate point (a, b), a < b --/
theorem path_count (m n : ℕ) (hmn : m < n) : 
  (∃ (N : ℕ), N = (nat.factorial (m+n-1)) / ((nat.factorial m) * (nat.factorial n)) * (n - m)) := 
sorry

end path_count_l40_40798


namespace first_plan_minutes_l40_40212

noncomputable def minutes_in_first_plan : ℝ :=
  if 50 + (2500 - x) * 0.35 = 75 + (2500 - 1000) * 0.45 then
    500
  else
    sorry

theorem first_plan_minutes :
  (50 + (2500 - minutes_in_first_plan) * 0.35 = 75 + (2500 - 1000) * 0.45) → 
  minutes_in_first_plan = 500 :=
by
  unfold minutes_in_first_plan
  sorry

end first_plan_minutes_l40_40212


namespace six_digit_numbers_with_at_least_one_zero_l40_40978

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40978


namespace six_digit_numbers_with_at_least_one_zero_l40_40973

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40973


namespace intersection_P_T_l40_40345

def P (x : ℝ) : Prop := |x| > 2
def T (x : ℝ) : Prop := 3^x > 1
def inter (x : ℝ) : Prop := x > 2

theorem intersection_P_T :
  {x : ℝ | P x} ∩ {x : ℝ | T x} = {x : ℝ | inter x} :=
by
  sorry

end intersection_P_T_l40_40345


namespace circle_area_l40_40526

theorem circle_area (θ : ℝ) : 
  let r := 3 * Real.cos θ - 4 * Real.sin θ in
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let center := (3 / 2 : ℝ, -2 : ℝ) in
  let radius := 5 / 2 in
  (x - center.1)^2 + (y - center.2)^2 = radius^2 →
  ∃ A : ℝ, A = (radius^2) * Real.pi ∧ A = 25 * Real.pi / 4 := 
by sorry

end circle_area_l40_40526


namespace smallest_integer_with_eight_factors_l40_40161

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, ∃ p q : ℕ, ∃ a b : ℕ,
    (prime p) ∧ (prime q) ∧ (p ≠ q) ∧ (n = p^a * q^b) ∧ ((a + 1) * (b + 1) = 8) ∧ (n = 24) :=
by 
  sorry

end smallest_integer_with_eight_factors_l40_40161


namespace tallest_giraffe_height_l40_40557

theorem tallest_giraffe_height :
  ∃ (height : ℕ), height = 96 ∧ (height = 68 + 28) := by
  sorry

end tallest_giraffe_height_l40_40557


namespace percentage_left_due_to_fear_is_15_l40_40188

-- Definitions based on the conditions
def initial_population : ℕ := 4400
def died_percentage : ℝ := 0.05
def remaining_population : ℕ := 3553

-- Intermediate calculations
def num_died : ℕ := (died_percentage * initial_population).toNat
def after_bombardment : ℕ := initial_population - num_died
def num_left_due_to_fear : ℕ := after_bombardment - remaining_population
def fear_percentage : ℝ := (num_left_due_to_fear.to_nat / after_bombardment) * 100

-- The theorem statement to prove the percentage
theorem percentage_left_due_to_fear_is_15 : fear_percentage = 15 := by
  sorry

end percentage_left_due_to_fear_is_15_l40_40188


namespace smallest_positive_period_l40_40845

theorem smallest_positive_period :
  let f : ℝ → ℝ := λ x, (Real.sqrt 2) * sin (2 * x + (π / 4)) in
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') :=
begin
  sorry,
end

end smallest_positive_period_l40_40845


namespace smallest_period_f_range_f_on_interval_l40_40336

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos (x - π / 3)

theorem smallest_period_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π :=
sorry

theorem range_f_on_interval :
  ∃ a b : ℝ, (∀ x ∈ set.Icc 0 (π / 2), f x ∈ set.Icc a b) ∧ a = 0 ∧ b = (2 + sqrt 3) / 2 :=
sorry

end smallest_period_f_range_f_on_interval_l40_40336


namespace cuboid_edge_lengths_l40_40517

theorem cuboid_edge_lengths (a b c : ℕ) (S V : ℕ) :
  (S = 2 * (a * b + b * c + c * a)) ∧ (V = a * b * c) ∧ (V = S) ∧ 
  (∃ d : ℕ, d = Int.sqrt (a^2 + b^2 + c^2)) →
  (∃ a b c : ℕ, a = 4 ∧ b = 8 ∧ c = 8) :=
by
  sorry

end cuboid_edge_lengths_l40_40517


namespace distance_travelled_downstream_l40_40553

-- Given conditions
def speed_boat_still_water : ℝ := 18 -- km/hr
def speed_current : ℝ := 4 -- km/hr
def travel_time_minutes : ℝ := 14 -- minutes

-- Effective speed and time in hours
def effective_speed_downstream := speed_boat_still_water + speed_current
def travel_time_hours := travel_time_minutes / 60 -- conversion from minutes to hours

-- Statement to prove
theorem distance_travelled_downstream : effective_speed_downstream * travel_time_hours = 5.1326 := 
by 
  sorry

end distance_travelled_downstream_l40_40553


namespace value_of_a8_l40_40317

section ArithmeticSequence

variables (a : ℕ → ℝ)

-- Condition: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- Given condition: a_7 + a_8 + a_9 = 21
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 7 + a 8 + a 9 = 21

-- Theorem to prove: a_8 = 7
theorem value_of_a8 (a : ℕ → ℝ) [is_arithmetic_sequence a] [sum_condition a] : a 8 = 7 := 
sorry

end ArithmeticSequence

end value_of_a8_l40_40317


namespace chrystal_vehicle_speed_l40_40746

theorem chrystal_vehicle_speed
  (V : ℝ)
  (h1 : ∀ (V : ℝ), 0 ≤ V)
  (ascendSpeed : V / 2 = V * 0.5)
  (descendSpeed : 1.2 * V = V + 0.2 * V)
  (distAscend : 60)
  (distDescend : 72)
  (timeTotal : 6)
  (timeEq : (60 / (V / 2)) + (72 / (1.2 * V)) = 6) :
  V = 30 :=
by
  -- proof to be filled
  sorry

end chrystal_vehicle_speed_l40_40746


namespace find_p5_l40_40417

-- Definitions based on conditions from the problem
def p (x : ℝ) : ℝ :=
  x^4 - 10 * x^3 + 35 * x^2 - 50 * x + 18  -- this construction ensures it's a quartic monic polynomial satisfying provided conditions

-- The main theorem we want to prove
theorem find_p5 :
  p 1 = 3 ∧ p 2 = 7 ∧ p 3 = 13 ∧ p 4 = 21 → p 5 = 51 :=
by
  -- The proof will be inserted here later
  sorry

end find_p5_l40_40417


namespace number_of_free_ranging_chickens_is_105_l40_40560

namespace ChickenProblem

-- Conditions as definitions
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def free_ranging_chickens : ℕ := 2 * run_chickens - 4
def total_coop_run_chickens : ℕ := coop_chickens + run_chickens

-- The ratio condition
def ratio_condition : Prop :=
  (coop_chickens + run_chickens) * 5 = free_ranging_chickens * 2

-- Proof Statement
theorem number_of_free_ranging_chickens_is_105 :
  free_ranging_chickens = 105 :=
by {
  sorry
}

end ChickenProblem

end number_of_free_ranging_chickens_is_105_l40_40560


namespace bisection_method_applicable_l40_40166

-- Define the functions corresponding to each equation
def f₁ (x : ℝ) : ℝ := Real.log x + x
def f₂ (x : ℝ) : ℝ := Real.exp x - 3 * x
def f₃ (x : ℝ) : ℝ := x^3 - 3 * x + 1
def f₄ (x : ℝ) : ℝ := 4 * x^2 - 4 * Real.sqrt 5 * x + 5

-- Define the intervals and continuity requirement
theorem bisection_method_applicable :
  (∃ (a b : ℝ), f₁ a * f₁ b < 0 ∧ ∀ (x : ℝ), a ≤ x ∧ x ≤ b → ContinuousAt f₁ x) ∧
  (∃ (a b : ℝ), f₂ a * f₂ b < 0 ∧ ∀ (x : ℝ), a ≤ x ∧ x ≤ b → ContinuousAt f₂ x) ∧
  (∃ (a b : ℝ), f₃ a * f₃ b < 0 ∧ ∀ (x : ℝ), a ≤ x ∧ x ≤ b → ContinuousAt f₃ x) :=
by sorry

end bisection_method_applicable_l40_40166


namespace find_k_5_l40_40477

def h (x : ℝ) : ℝ := 5 / (3 - x)
def h_inv (x : ℝ) : ℝ := 3 - 5 / x
def k (x : ℝ) : ℝ := 1 / (h_inv x) + 8

theorem find_k_5 : k 5 = 8.5 :=
by
  sorry

end find_k_5_l40_40477


namespace arithmetic_sequence_sum_l40_40367

open Nat

theorem arithmetic_sequence_sum :
  let a1 := 1
  let a96 := 1996
  let d := 21
  let n := 96
  let s := n * (2 * a1 + (n - 1) * d) / 2
   in s = 95856 :=
by
  -- Definitions
  let a1 := 1
  let a96 := 1996
  let d := 21
  let n := 96
  let s := n * (2 * a1 + (n - 1) * d) / 2
  show s = 95856
  sorry

end arithmetic_sequence_sum_l40_40367


namespace six_digit_numbers_with_zero_l40_40926

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40926


namespace six_digit_numbers_with_at_least_one_zero_l40_40974

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40974


namespace mileage_per_gallon_l40_40081

-- Definitions for the conditions
def total_distance_to_grandma (d : ℕ) : Prop := d = 100
def gallons_to_grandma (g : ℕ) : Prop := g = 5

-- The statement to be proved
theorem mileage_per_gallon :
  ∀ (d g m : ℕ), total_distance_to_grandma d → gallons_to_grandma g → m = d / g → m = 20 :=
sorry

end mileage_per_gallon_l40_40081


namespace problem1_problem2_l40_40245

-- Problem 1
theorem problem1 : 2023^2 - 2024 * 2022 = 1 :=
sorry

-- Problem 2
variables (a b c : ℝ)
theorem problem2 : 5 * a^2 * b^3 * (-1/10 * a * b^3 * c) / (1/2 * a * b^2)^3 = -4 * c :=
sorry

end problem1_problem2_l40_40245


namespace binomial_expansion_coefficient_l40_40007

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the problem in the statement
theorem binomial_expansion_coefficient {n : ℕ} (h : binom n 2 * 9 = 54) : n = 4 :=
begin
  sorry
end

end binomial_expansion_coefficient_l40_40007


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40910

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40910


namespace coffee_break_participants_l40_40582

theorem coffee_break_participants (n : ℕ) (h1 : n = 14) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 6 ∧ n - 2 * k > 0 ∧ n - 2 * k < n)
    (h3 : ∀ m, m < n → (∃ k, m = n - 2 * k → m ∈ {6, 8, 10, 12})): 
    ∃ m, n - 2 * m ∈ {6, 8, 10, 12} := 
by
  use 6
  use 8
  use 10
  use 12
  sorry

end coffee_break_participants_l40_40582


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40913

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40913


namespace expected_hit_targets_expected_hit_targets_not_less_than_half_l40_40259

-- Part (a): The expected number of hit targets
theorem expected_hit_targets (n : ℕ) (h : n ≠ 0) :
  E (number_of_hit_targets n) = n * (1 - (1 - (1 / n)) ^ n) :=
sorry

-- Part (b): The expected number of hit targets cannot be less than n / 2
theorem expected_hit_targets_not_less_than_half (n : ℕ) (h : n ≠ 0) :
  n * (1 - (1 - (1 / n)) ^ n) ≥ n / 2 :=
sorry

end expected_hit_targets_expected_hit_targets_not_less_than_half_l40_40259


namespace midpoint_length_l40_40374

theorem midpoint_length (A B C : Point) (h : midpoint C A B) (hAC : distance A C = 2.5) : distance A B = 5 :=
by
  sorry

end midpoint_length_l40_40374


namespace find_added_number_l40_40013

theorem find_added_number (R X : ℕ) (hR : R = 45) (h : 2 * (2 * R + X) = 188) : X = 4 :=
by 
  -- We would normally provide the proof here
  sorry  -- We skip the proof as per the instructions

end find_added_number_l40_40013


namespace a_seq_b_sum_l40_40314

noncomputable def a : ℕ → ℕ
| 0     := 0  -- base case for natural numbers in Lean, not actually used
| (n+1) := 2*n + 1

-- The first condition for the sequence a_n
theorem a_seq (k : ℕ) (hk : k > 0) : (∑ i in range k.succ, a (i + 1)) = k * (k + 1) :=
sorry

-- Defining b_n by recurrence relation
noncomputable def b : ℕ → ℝ
| 0     := 0  -- again base case not used, just for natural numbers
| (n+1) := if n = 0 then 1 else -1^(n + 1) * (1 / (n+1)) * ∏ i in range n, (n - i) / (i+1)

-- The requirement for the sum of b_n's
theorem b_sum (n : ℕ) (hn : n ≥ 2) : (∑ i in range n.succ, b (i + 1)) = 1 / n :=
sorry

end a_seq_b_sum_l40_40314


namespace max_m_value_correct_l40_40312

open Real

noncomputable def max_m_value {k : ℝ} (hk : k > 0) : ℝ :=
  let a := sqrt 2 / 2 * k^(-1/4)
  a

theorem max_m_value_correct {k : ℝ} (hk : k > 0) :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (let m := min a (b / (k * a^2 + b^2))
   in m ≤ sqrt 2 / 2 * k^(-1/4)) :=
  sorry

end max_m_value_correct_l40_40312


namespace final_cost_l40_40466

-- Definitions of initial conditions
def initial_cart_total : ℝ := 54.00
def discounted_item_original_price : ℝ := 20.00
def discount_rate1 : ℝ := 0.20
def coupon_rate : ℝ := 0.10

-- Prove the final cost after applying discounts
theorem final_cost (initial_cart_total discounted_item_original_price discount_rate1 coupon_rate : ℝ) :
  let discounted_price := discounted_item_original_price * (1 - discount_rate1)
  let total_after_first_discount := initial_cart_total - discounted_price
  let final_total := total_after_first_discount * (1 - coupon_rate)
  final_total = 45.00 :=
by 
  sorry

end final_cost_l40_40466


namespace ice_palace_steps_l40_40452

theorem ice_palace_steps (time_for_20_steps total_time : ℕ) (h1 : time_for_20_steps = 120) (h2 : total_time = 180) : 
  total_time * 20 / time_for_20_steps = 30 := by
  have time_per_step : ℕ := time_for_20_steps / 20
  have total_steps : ℕ := total_time / time_per_step
  sorry

end ice_palace_steps_l40_40452


namespace six_digit_numbers_with_zero_l40_40981

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40981


namespace num_zeros_in_decimal_rep_l40_40645

theorem num_zeros_in_decimal_rep (a b : ℕ) (h₁ : a = 7) (h₂ : b = 8000) :
  num_zeros_between_decimal_and_first_nonzero (a / b : ℚ) = 2 :=
sorry

end num_zeros_in_decimal_rep_l40_40645


namespace routes_A_to_B_on_3x2_grid_l40_40358

def routes_from_A_to_B : ℕ := 10

/-- Prove the number of different routes from point A to point B on a 3x2 grid --/
theorem routes_A_to_B_on_3x2_grid : routes_from_A_to_B = (nat.choose 5 2) := by
  sorry

end routes_A_to_B_on_3x2_grid_l40_40358


namespace six_digit_numbers_with_at_least_one_zero_l40_40860

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40860


namespace participants_coffee_l40_40597

theorem participants_coffee (n : ℕ) (h_n : n = 14) (h_not_all : 0 < n - 2 * k) (h_not_all2 : n - 2 * k < n)
(h_neighbors : ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧ (remaining i → leaving j) ∧ (leaving i → remaining j)) :
  ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 6 ∧ (n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12) :=
by {
  sorry
}

end participants_coffee_l40_40597


namespace no_limit_elements_in_set_with_zero_average_l40_40570

open Set

theorem no_limit_elements_in_set_with_zero_average 
  (s : Set ℝ) 
  (h_avg : (∑ x in s, x) = 0) 
  (h_nonneg : ∃ x ∈ s, x ≥ 0) :
  ∃ n : ℕ, (card s) = n :=
sorry

end no_limit_elements_in_set_with_zero_average_l40_40570


namespace zeros_between_decimal_and_first_nonzero_digit_l40_40651

theorem zeros_between_decimal_and_first_nonzero_digit : 
  let frac : ℚ := 7 / 8000 in
  let exp : ℤ := 6 - nat_digits (875 : ℕ) in
  exp = 3 :=
by sorry

end zeros_between_decimal_and_first_nonzero_digit_l40_40651


namespace domain_of_f_l40_40285

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - real.sqrt (2 - real.sqrt (3 - x)))

theorem domain_of_f :
  ∀ x, (3 - x ≥ 0) ∧ (2 - real.sqrt (3 - x) ≥ 0) ∧ (1 - real.sqrt (2 - real.sqrt (3 - x)) ≥ 0) ↔ x ∈ set.Icc (-1 : ℝ) 2 :=
by
  sorry

end domain_of_f_l40_40285


namespace value_of_x_l40_40393

theorem value_of_x {x y z w v : ℝ} 
  (h1 : y * x = 3)
  (h2 : z = 3)
  (h3 : w = z * y)
  (h4 : v = w * z)
  (h5 : v = 18)
  (h6 : w = 6) :
  x = 3 / 2 :=
by
  sorry

end value_of_x_l40_40393


namespace smallest_prime_divisor_of_sum_l40_40624

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end smallest_prime_divisor_of_sum_l40_40624


namespace locus_of_X_l40_40085

-- Defining points and triangles
variables (A B C D E X M : Type) [EuclideanGeometry A B C D E X M]

-- Conditions
axiom equal_segments_AD_CE : ∀ (A B C D E : Type), seg_eq A D C E
axiom midpoint_X_DE : ∀ (D E X : Type), midpoint D E X
axiom midpoint_M_AC : ∀ (A C M : Type), midpoint A C M

-- Theorem stating the proof problem
theorem locus_of_X (A B C D E X M : Type) [EuclideanGeometry A B C D E X M] 
  (h1 : seg_eq A D C E) 
  (h2 : midpoint D E X) 
  (h3 : midpoint A C M) :
  ∃ l : Line, passes_through l M ∧ parallel l (angle_bisector B) :=
sorry

end locus_of_X_l40_40085


namespace surface_area_of_circumscribed_sphere_l40_40813

theorem surface_area_of_circumscribed_sphere (PA PB PC : ℝ) 
  (h1 : PA = 2) 
  (h2 : PB = sqrt 3) 
  (h3 : PC = 3) 
  (h4 : PA * PB * PC ≠ 0) 
  (h5 : PA * PA + PB * PB + PC * PC = 4 * 4) :
  4 * Real.pi * (4 / 2) * (4 / 2) = 16 * Real.pi := 
by 
  sorry

end surface_area_of_circumscribed_sphere_l40_40813


namespace find_A_d_minus_B_d_l40_40480

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end find_A_d_minus_B_d_l40_40480


namespace participants_coffee_l40_40583

theorem participants_coffee (n : ℕ) (h1 : n = 14) (h2 : 0 < (14 - 2 * (n / 2)) < 14) : 
  ∃ (k : ℕ), k ∈ {6, 8, 10, 12} ∧ k = (14 - 2 * (n / 2)) :=
by sorry

end participants_coffee_l40_40583


namespace six_digit_numbers_with_zero_l40_40955

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40955


namespace problem_l40_40482

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end problem_l40_40482


namespace alice_total_distance_l40_40226

-- Define the points in the problem
def point1 := (-3 : ℝ, 6 : ℝ)
def point2 := (1 : ℝ, 1 : ℝ)
def point3 := (6 : ℝ, -3 : ℝ)

-- Function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Calculate the total distance Alice must travel
def totalDistance : ℝ :=
  distance point1 point2 + distance point2 point3

-- The statement to be proved
theorem alice_total_distance : totalDistance = 2 * Real.sqrt 41 := by
  sorry

end alice_total_distance_l40_40226


namespace num_zeros_in_decimal_rep_l40_40646

theorem num_zeros_in_decimal_rep (a b : ℕ) (h₁ : a = 7) (h₂ : b = 8000) :
  num_zeros_between_decimal_and_first_nonzero (a / b : ℚ) = 2 :=
sorry

end num_zeros_in_decimal_rep_l40_40646


namespace andrei_wins_if_and_only_if_irreducible_fraction_l40_40233

theorem andrei_wins_if_and_only_if_irreducible_fraction (p : ℚ) :
  (∃ (m n : ℕ), p = m / 2^n ∧ nat.coprime m (2^n) ∧ 0 ≤ m ∧ m ≤ 2^n) ↔ 
  (∀ d : ℚ, (d > 0) → (∃ n : ℕ, ∀ dir : bool, 
    (if dir then p + d else p - d) (mod 1) = 0 ∨ (if dir then p + d else p - d) (mod 1) = 1)) :=
begin
  sorry
end

end andrei_wins_if_and_only_if_irreducible_fraction_l40_40233


namespace sum_of_fourth_powers_39_l40_40293

theorem sum_of_fourth_powers_39 :
  let n := 39 in
  n * (n + 1) * (2 * n + 1) * (3 * n ^ 2 + 3 * n - 1) / 30 = 19215272 :=
by
  -- Machine verification of the formula steps can be included here
  sorry

end sum_of_fourth_powers_39_l40_40293


namespace smallest_positive_angle_l40_40792

-- Definitions based on conditions in the problem
def sin_60 := Real.sin (60 * Real.pi / 180)
def cos_42 := Real.cos (42 * Real.pi / 180)
def sin_12 := Real.sin (12 * Real.pi / 180)
def cos_6 := Real.cos (6 * Real.pi / 180)

-- The statement of the problem
theorem smallest_positive_angle :
  ∃ θ : ℝ, θ > 0 ∧ θ < 90 ∧ Real.cos (θ * Real.pi / 180) = sin_60 + cos_42 - sin_12 - cos_6 ∧ θ = 66 :=
by
  sorry

end smallest_positive_angle_l40_40792


namespace smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40633

/-- 
  Problem statement: Prove that the smallest prime divisor of 3^19 + 11^13 is 2, 
  given the conditions:
   - 3^19 is odd
   - 11^13 is odd
   - The sum of two odd numbers is even
-/

theorem smallest_prime_divisor_of_3_pow_19_plus_11_pow_13 :
  ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ (3^19 + 11^13) := 
by
  sorry

end smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40633


namespace sum_of_smallest_and_largest_prime_l40_40070

theorem sum_of_smallest_and_largest_prime :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in
  (primes.head! + primes.reverse.head! = 49) := 
by
  sorry

end sum_of_smallest_and_largest_prime_l40_40070


namespace calculation_correct_l40_40737

theorem calculation_correct :
  ((1 / 3) ^ 6 / (2 / 5) ^ (-4) + 1 / 2) = (455657 / 911250) :=
sorry

end calculation_correct_l40_40737


namespace six_digit_numbers_with_zero_count_l40_40900

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40900


namespace arithmetic_sequence_sum_l40_40163

theorem arithmetic_sequence_sum (x y : ℕ) (h₀: ∃ (n : ℕ), x = 3 + n * 4) (h₁: ∃ (m : ℕ), y = 3 + m * 4) (h₂: y = 31 - 4) (h₃: x = y - 4) : x + y = 50 := by
  sorry

end arithmetic_sequence_sum_l40_40163


namespace remaining_bottles_after_2_days_l40_40203

-- Definitions based on the conditions:
def initial_bottles : ℕ := 24
def fraction_first_day : ℚ := 1 / 3
def fraction_second_day : ℚ := 1 / 2

-- Theorem statement proving the remaining number of bottles after 2 days
theorem remaining_bottles_after_2_days : 
    (initial_bottles - initial_bottles * fraction_first_day) - 
    ((initial_bottles - initial_bottles * fraction_first_day) * fraction_second_day) = 8 := 
by 
    -- Skipping the proof
    sorry

end remaining_bottles_after_2_days_l40_40203


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40912

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40912


namespace trapezoid_area_l40_40508

variable (a b : ℝ) (h1 : a > b)

theorem trapezoid_area (h2 : ∃ (angle1 angle2 : ℝ), angle1 = 30 ∧ angle2 = 45) : 
  (1/4) * ((a^2 - b^2) * (Real.sqrt 3 - 1)) = 
    ((1/2) * (a + b) * ((b - a) * (Real.sqrt 3 - 1) / 2)) := 
sorry

end trapezoid_area_l40_40508


namespace divide_11_items_into_groups_l40_40385

theorem divide_11_items_into_groups : 
  (finset.card (pow finset.univ 11).powerset.filter (λ s, 3 ≤ s.card ∧ s.card ≤ 8)) / 2 = 957 :=
by
  sorry

end divide_11_items_into_groups_l40_40385


namespace percentage_increase_l40_40750

theorem percentage_increase (employees_dec : ℝ) (employees_jan : ℝ) (inc : ℝ) (percentage : ℝ) :
  employees_dec = 470 →
  employees_jan = 408.7 →
  inc = employees_dec - employees_jan →
  percentage = (inc / employees_jan) * 100 →
  percentage = 15 := 
sorry

end percentage_increase_l40_40750


namespace cody_games_remaining_l40_40247

-- Definitions based on the conditions
def initial_games : ℕ := 9
def games_given_away : ℕ := 4

-- Theorem statement
theorem cody_games_remaining : initial_games - games_given_away = 5 :=
by sorry

end cody_games_remaining_l40_40247


namespace routes_from_A_to_B_l40_40360

theorem routes_from_A_to_B (n_r n_d : ℕ) (n_r_eq : n_r = 3) (n_d_eq : n_d = 2) :
  nat.choose (n_r + n_d) n_r = 10 :=
by
  rw [n_r_eq, n_d_eq]
  exact nat.choose_succ_succ 3 2

end routes_from_A_to_B_l40_40360


namespace six_digit_numbers_with_zero_l40_40893

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40893


namespace bellas_score_l40_40447

theorem bellas_score (sum_19 : ℕ) (sum_20 : ℕ) (avg_19 : ℕ) (avg_20 : ℕ) (n_19 : ℕ) (n_20 : ℕ) :
  avg_19 = 82 → avg_20 = 85 → n_19 = 19 → n_20 = 20 → sum_19 = n_19 * avg_19 → sum_20 = n_20 * avg_20 →
  sum_20 - sum_19 = 142 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end bellas_score_l40_40447


namespace walls_per_person_l40_40722

theorem walls_per_person (people : ℕ) (rooms : ℕ) (r4_walls r5_walls : ℕ) (total_walls : ℕ) (walls_each_person : ℕ)
  (h1 : people = 5)
  (h2 : rooms = 9)
  (h3 : r4_walls = 5 * 4)
  (h4 : r5_walls = 4 * 5)
  (h5 : total_walls = r4_walls + r5_walls)
  (h6 : walls_each_person = total_walls / people) :
  walls_each_person = 8 := by
  sorry

end walls_per_person_l40_40722


namespace coffee_participants_l40_40604

noncomputable def participants_went_for_coffee (total participants : ℕ) (stay : ℕ) : Prop :=
  (0 < stay) ∧ (stay < 14) ∧ (stay % 2 = 0) ∧ ∃ k : ℕ, stay = 2 * k

theorem coffee_participants :
  ∀ (total participants : ℕ), total participants = 14 →
  (∀ participant, ((∃ k : ℕ, stay = 2 * k) → (0 < stay) ∧ (stay < 14))) →
  participants_went_for_coffee total_participants (14 - stay) =
  (stay = 12) ∨ (stay = 10) ∨ (stay = 8) ∨ (stay = 6) :=
by
  sorry

end coffee_participants_l40_40604


namespace calculation_correct_l40_40271

def grid_coloring_probability : ℚ := 591 / 1024

theorem calculation_correct : (m + n = 1615) ↔ (∃ m n : ℕ, m + n = 1615 ∧ gcd m n = 1 ∧ grid_coloring_probability = m / n) := sorry

end calculation_correct_l40_40271


namespace average_hit_targets_formula_average_hit_targets_ge_half_l40_40263

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

end average_hit_targets_formula_average_hit_targets_ge_half_l40_40263


namespace monica_total_savings_l40_40074

theorem monica_total_savings :
  ∀ (weekly_saving : ℤ) (weeks_per_cycle : ℤ) (cycles : ℤ),
    weekly_saving = 15 →
    weeks_per_cycle = 60 →
    cycles = 5 →
    weekly_saving * weeks_per_cycle * cycles = 4500 :=
by
  intros weekly_saving weeks_per_cycle cycles
  sorry

end monica_total_savings_l40_40074


namespace orange_juice_fraction_l40_40609

def pitcher1_capacity : ℕ := 500
def pitcher2_capacity : ℕ := 700
def pitcher1_juice_fraction : ℚ := 1 / 2
def pitcher2_juice_fraction : ℚ := 3 / 5

theorem orange_juice_fraction :
  let pitcher1_juice := pitcher1_capacity * pitcher1_juice_fraction in
  let pitcher2_juice := pitcher2_capacity * pitcher2_juice_fraction in
  let total_juice := pitcher1_juice + pitcher2_juice in
  let total_volume := pitcher1_capacity + pitcher2_capacity in
  total_juice / total_volume = 67 / 120 :=
by
  sorry

end orange_juice_fraction_l40_40609


namespace solid_triangle_front_view_l40_40186

def is_triangle_front_view (solid : ℕ) : Prop :=
  solid = 1 ∨ solid = 2 ∨ solid = 3 ∨ solid = 5

theorem solid_triangle_front_view (s : ℕ) (h : s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 4 ∨ s = 5 ∨ s = 6):
  is_triangle_front_view s ↔ (s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 5) :=
by
  sorry

end solid_triangle_front_view_l40_40186


namespace sum_of_first_ten_superb_equals_1399_l40_40252

-- Define a superb number
def is_superb (n : ℕ) : Prop := 
  let divisors := (list.filter (λ x : ℕ, x ≠ n ∧ n % x = 0) (list.range (n+1)))
  list.prod divisors = n

-- Define the sum of the first ten superb numbers
def sum_first_ten_superb : ℕ :=
  (list.filter is_superb (list.range 10000)).take 10).sum

theorem sum_of_first_ten_superb_equals_1399 : sum_first_ten_superb = 1399 := 
  sorry

end sum_of_first_ten_superb_equals_1399_l40_40252


namespace total_window_area_is_correct_l40_40718

open Real

-- Define the given conditions
def radius_semicircle : ℝ := 50
def length_rectangle_m : ℝ := 1.5
def width_rectangle_m : ℝ := 1
def length_rectangle_cm : ℝ := length_rectangle_m * 100
def width_rectangle_cm : ℝ := width_rectangle_m * 100

-- Define total area calculation
def area_rectangle : ℝ := length_rectangle_cm * width_rectangle_cm
def area_semicircle : ℝ := (1/2) * π * (radius_semicircle ^ 2)
def total_area : ℝ := area_rectangle + area_semicircle

-- The theorem to be proved
theorem total_window_area_is_correct :
  total_area ≈ 18926.9875 := sorry

end total_window_area_is_correct_l40_40718


namespace will_buy_5_toys_l40_40173

theorem will_buy_5_toys (initial_money spent_money toy_cost money_left toys : ℕ) 
  (h1 : initial_money = 57) 
  (h2 : spent_money = 27) 
  (h3 : toy_cost = 6) 
  (h4 : money_left = initial_money - spent_money) 
  (h5 : toys = money_left / toy_cost) : 
  toys = 5 := 
by
  sorry

end will_buy_5_toys_l40_40173


namespace six_digit_numbers_with_zero_l40_40871

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40871


namespace bacterium_probability_l40_40145

noncomputable def probability_bacterium_in_small_cup
  (total_volume : ℚ) (small_cup_volume : ℚ) (contains_bacterium : Bool) : ℚ :=
if contains_bacterium then small_cup_volume / total_volume else 0

theorem bacterium_probability
  (total_volume : ℚ) (small_cup_volume : ℚ) (bacterium_present : Bool) :
  total_volume = 2 ∧ small_cup_volume = 0.1 ∧ bacterium_present = true →
  probability_bacterium_in_small_cup 2 0.1 true = 0.05 :=
by
  intros h
  sorry

end bacterium_probability_l40_40145


namespace six_digit_numbers_with_zero_l40_40984

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40984


namespace six_digit_numbers_with_at_least_one_zero_l40_40859

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40859


namespace first_candidate_percentage_l40_40384

-- Define the conditions
def total_votes : ℕ := 7500
def invalid_vote_percentage : ℝ := 0.20
def other_candidate_votes : ℕ := 2700

-- Define the total valid votes
def total_valid_votes : ℕ := (total_votes * (1 - invalid_vote_percentage)).toInt

-- Define the votes for the first candidate
def first_candidate_votes : ℕ := total_valid_votes - other_candidate_votes

-- State the proof problem
theorem first_candidate_percentage :
  (first_candidate_votes.toFloat / total_valid_votes.toFloat) * 100 = 55 := by
  sorry

end first_candidate_percentage_l40_40384


namespace value_of_y1_minus_y2_l40_40128

-- Definitions based on the conditions
noncomputable def ellipse_eqn : ℝ → ℝ → Prop := λ x y, (x^2 / 25) + (y^2 / 16) = 1

def foci_1 : ℝ × ℝ := (-3, 0)
def foci_2 : ℝ × ℝ := (3, 0)

def chord_AB (x1 y1 x2 y2 : ℝ) : Prop := ellipse_eqn x1 y1 ∧ ellipse_eqn x2 y2

def radius_of_inscribed_circle : ℝ := 1 / 2

-- The main theorem statement
theorem value_of_y1_minus_y2 {x1 y1 x2 y2 : ℝ} 
    (hAB : chord_AB x1 y1 x2 y2) 
    (hcircumference : 4 * radius_of_inscribed_circle = π)
    : | y1 - y2 | = 5 / 3 := sorry

end value_of_y1_minus_y2_l40_40128


namespace measure_of_alpha_l40_40617

theorem measure_of_alpha
  (A B D α : ℝ)
  (hA : A = 50)
  (hB : B = 150)
  (hD : D = 140)
  (quadrilateral_sum : A + B + D + α = 360) : α = 20 :=
by
  rw [hA, hB, hD] at quadrilateral_sum
  sorry

end measure_of_alpha_l40_40617


namespace average_hit_targets_formula_average_hit_targets_ge_half_l40_40265

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

end average_hit_targets_formula_average_hit_targets_ge_half_l40_40265


namespace prod_of_roots_eq_zero_l40_40754

theorem prod_of_roots_eq_zero :
  let P (x : ℂ) := ∏ k in (finset.range 15).map (λ n, n+1), (x - exp (2 * π * complex.I * (k:ℂ) / 17))
  ∧ let Q (x : ℂ) := ∏ j in (finset.range 12).map (λ n, n+1), (x - exp (2 * π * complex.I * (j:ℂ) / 13))
  in (∏ k in (finset.range 15).map (λ n, n+1), ∏ j in (finset.range 12).map (λ n, n+1), 
    (exp (2 * π * j * complex.I / 13) - exp (2 * π * k * complex.I / 17))) = 0 :=
by
  intro P Q
  sorry

end prod_of_roots_eq_zero_l40_40754


namespace inequality_a_inequality_b_l40_40089

variable (ABCD : Type) [Tetrahedron ABCD]
variable (P : ℝ) (Δ Π : ℝ) 

theorem inequality_a (h1 : is_tetrahedron ABCD) (h2 : ∀ x, facet_area x = Δ / 4):
  P ≥ (sqrt (2 * sqrt 3 * Δ)) / 4 :=
sorry

theorem inequality_b (h1 : is_tetrahedron ABCD) (h2 : ∀ x, volume x = Π):
  P ≥ (sqrt 6 * Π) / 12 :=
sorry

end inequality_a_inequality_b_l40_40089


namespace gcd_fib_1960_1988_l40_40116

noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem gcd_fib_1960_1988 : Nat.gcd (fib 1960) (fib 1988) = 317811 := by
  sorry

end gcd_fib_1960_1988_l40_40116


namespace area_of_shaded_region_l40_40184

theorem area_of_shaded_region (EH GH : ℝ) (hEH : EH = 5) (hGH : GH = 12) :
  let EG := Real.sqrt (EH^2 + GH^2)
  let radius := EG
  let circle_area := Real.pi * radius^2
  let half_circle_area := (1/2) * circle_area
  let rectangle_area := EH * GH
  let shaded_region_area := half_circle_area - rectangle_area
in shaded_region_area = (169 * Real.pi) / 2 - 60 :=
by
  sorry

end area_of_shaded_region_l40_40184


namespace six_digit_numbers_with_zero_l40_40925

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40925


namespace max_mappings_l40_40346

def A : set ℕ := {x, y}
def B : set ℕ := {0, 1}

theorem max_mappings (A B : set ℕ) (hA : A = {x, y}) (hB : B = {0, 1}) :
  (finset.univ : finset (A → B)).card = 4 :=
by sorry

end max_mappings_l40_40346


namespace sum_of_fourth_powers_of_solutions_l40_40793

theorem sum_of_fourth_powers_of_solutions (x y : ℝ)
  (h : |x^2 - 2 * x + 1/1004| = 1/1004 ∨ |y^2 - 2 * y + 1/1004| = 1/1004) :
  x^4 + y^4 = 20160427280144 / 12600263001 :=
sorry

end sum_of_fourth_powers_of_solutions_l40_40793


namespace distance_from_point_to_focus_l40_40847

theorem distance_from_point_to_focus (x0 : ℝ) (h1 : (2 * Real.sqrt 3)^2 = 4 * x0) :
    x0 + 1 = 4 := by
  sorry

end distance_from_point_to_focus_l40_40847


namespace six_digit_numbers_with_zero_l40_40886

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40886


namespace min_value_PB_AB_l40_40027

def x_alpha (α : ℝ) := 1 + Real.cos α
def y_alpha (α : ℝ) := Real.sin α

def parametric_curve_C (x y α : ℝ) : Prop :=
  x = x_alpha α ∧ y = y_alpha α

def polar_line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + π / 4) = 2 * Real.sqrt 2

def point_P : ℝ × ℝ := (-2, 2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_value_PB_AB (x y α ρ θ : ℝ) :
  parametric_curve_C x y α →
  polar_line_l ρ θ →
  (∃ (A B : ℝ × ℝ), A = (x, y) ∧ B = (ρ * Real.cos θ, ρ * Real.sin θ) ∧
    distance point_P B + distance A B = Real.sqrt 37 - 1) :=
sorry

end min_value_PB_AB_l40_40027


namespace g_symmetric_l40_40521

theorem g_symmetric (g : ℝ → ℝ) (h₀ : ∀ x, x ≠ 0 → (g x + 3 * g (1 / x) = 4 * x ^ 2)) : 
  ∀ x : ℝ, x ≠ 0 → g x = g (-x) :=
by 
  sorry

end g_symmetric_l40_40521


namespace intersection_of_A_and_B_l40_40318

variable A : Set ℤ := {-1, 0, 1, 2}
variable B : Set ℤ := {x | x^2 ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} :=
by
  sorry

end intersection_of_A_and_B_l40_40318


namespace b_positive_l40_40843

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := (a * x + b) / x

-- State the problem condition
def is_monotonically_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y ≤ f x

-- Define the domain (0, +∞)
def positive_real : Set ℝ := { x | 0 < x }

-- The theorem to prove
theorem b_positive (a b : ℝ) (h : is_monotonically_decreasing_on (f a b) positive_real) : b > 0 :=
  sorry

end b_positive_l40_40843


namespace coffee_break_l40_40572

variable {n : ℕ}
variables {participants : Finset ℕ} (hparticipants : participants.card = 14)
variables (left : Finset ℕ)
variables (stayed : Finset ℕ) (hstayed : stayed.card = 14 - 2 * left.card)

-- The overall proof problem statement:
theorem coffee_break (h : ∀ x ∈ stayed, ∃! y, (y ∈ left ∧ adjacent x y participants)) :
  left.card = 6 ∨ left.card = 8 ∨ left.card = 10 ∨ left.card = 12 := 
sorry

end coffee_break_l40_40572


namespace six_digit_numbers_with_at_least_one_zero_l40_40857

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40857


namespace bottles_remaining_after_2_days_l40_40201

theorem bottles_remaining_after_2_days :
  ∀ (initial_bottles : ℕ), initial_bottles = 24 →
  let bottles_first_day := initial_bottles - initial_bottles / 3 in
  let bottles_after_first_day := initial_bottles - bottles_first_day in
  let bottles_second_day := bottles_after_first_day / 2 in
  let bottles_remaining := bottles_after_first_day - bottles_second_day in
  bottles_remaining = 8 :=
by
  intros initial_bottles h_init
  let bottles_first_day := initial_bottles / 3
  let bottles_after_first_day := initial_bottles - bottles_first_day
  let bottles_second_day := bottles_after_first_day / 2
  let bottles_remaining := bottles_after_first_day - bottles_second_day
  have h_init_val : initial_bottles = 24 := h_init
  rw h_init_val at *
  calc
    bottles_first_day = 8 : by sorry
    bottles_after_first_day = 24 - 8 : by sorry
    _ = 16 : by sorry
    bottles_second_day = 16 / 2 : by sorry
    _ = 8 : by sorry
    bottles_remaining = 16 - 8 : by sorry
    _ = 8 : by sorry

end bottles_remaining_after_2_days_l40_40201


namespace dp_kl_perpendicular_l40_40453

noncomputable def square_side_length := ℝ

structure Point (α : Type) := 
  (x : α)
  (y : α)

structure Square (α : Type) := 
  (A B C D : Point α)
  (side_length : α)

variables {α : Type} [LinearOrderedField α] [DecidableEq α]

def point_K_L_intersection (S : Square α) (K L : Point α) (hK : K ∈ S.AB) (hL : L ∈ S.BC) (h: K.distance_to B = L.distance_to C) : Point α := sorry

def slope (P1 P2 : Point α) : α := (P2.y - P1.y) / (P2.x - P1.x)

theorem dp_kl_perpendicular {a : α} (S : Square α) (K L : Point α) (P : Point α)
  (h1 : S.side_length = a) 
  (hK_condition : K.x ∈ set.Icc 0 a)
  (hL_condition : L.y ∈ set.Icc 0 a)
  (hKOnAB : K.y = 0)
  (hLOnBC : L.x = a)
  (hKL : K.distance_to B = L.distance_to C)
  (hP : P = point_K_L_intersection S K L hKOnAB hLOnBC hKL) :
  slope S.D P * slope K L = -1 :=
sorry

end dp_kl_perpendicular_l40_40453


namespace find_number_l40_40661

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end find_number_l40_40661


namespace original_faculty_size_l40_40454

theorem original_faculty_size (F : ℝ) (h1 : F * 0.85 * 0.80 = 195) : F = 287 :=
by
  sorry

end original_faculty_size_l40_40454


namespace six_digit_numbers_with_at_least_one_zero_l40_40971

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40971


namespace determine_k_for_circle_l40_40257

theorem determine_k_for_circle (x y k : ℝ) (h : x^2 + 14*x + y^2 + 8*y - k = 0) (r : ℝ) :
  r = 5 → k = 40 :=
by
  intros radius_eq_five
  sorry

end determine_k_for_circle_l40_40257


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40946

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40946


namespace cosine_identity_l40_40822

theorem cosine_identity (α : ℝ) (h : sin α = 1 / 3) : cos (π / 4 + α) * cos (π / 4 - α) = 7 / 18 :=
by
  sorry

end cosine_identity_l40_40822


namespace probability_of_all_co_captains_l40_40561

def team_sizes : List ℕ := [6, 8, 9, 10]

def captains_per_team : ℕ := 3

noncomputable def probability_all_co_captains (s : ℕ) : ℚ :=
  1 / (Nat.choose s 3 : ℚ)

noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * 
  (probability_all_co_captains 6 + 
   probability_all_co_captains 8 +
   probability_all_co_captains 9 +
   probability_all_co_captains 10)

theorem probability_of_all_co_captains : total_probability = 1 / 84 :=
  sorry

end probability_of_all_co_captains_l40_40561


namespace range_of_quadratic_l40_40366

theorem range_of_quadratic (a b c : ℝ) (h_domain : ∀ x, -1 ≤ x → x ≤ 2 → true) (h_pos : a > 0) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → ax^2 + bx + c ∈ 
    set.Icc (min (min (-b^2 / (4 * a) + c) (a - b + c)) (4 * a + 2 * b + c))
             (max (a - b + c) (4 * a + 2 * b + c))) := by
  sorry

end range_of_quadratic_l40_40366


namespace six_digit_numbers_with_at_least_one_zero_l40_40853

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40853


namespace zeros_between_decimal_point_and_first_nonzero_digit_l40_40656

theorem zeros_between_decimal_point_and_first_nonzero_digit :
  let n := (7 : ℚ) / 8000 in
  decimal_zeros_between_decimal_point_and_first_nonzero_digit n = 3 :=
by
  sorry

end zeros_between_decimal_point_and_first_nonzero_digit_l40_40656


namespace participants_coffee_l40_40584

theorem participants_coffee (n : ℕ) (h1 : n = 14) (h2 : 0 < (14 - 2 * (n / 2)) < 14) : 
  ∃ (k : ℕ), k ∈ {6, 8, 10, 12} ∧ k = (14 - 2 * (n / 2)) :=
by sorry

end participants_coffee_l40_40584


namespace solution_set_f_l40_40833

noncomputable def f (x : ℝ) : ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(x) = f(-x)

def decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ∈ I → x2 ∈ I → x1 < x2 → f(x1) > f(x2)

theorem solution_set_f (h_even : is_even f)
    (h_decr : decreasing_on f {x | x ≤ 0}) :
    {x | x - 1 < 3} ∩ {x | x - 1 > 0} = {x | 1 < x ∧ x < 4} :=
by
  sorry

end solution_set_f_l40_40833


namespace coefficient_x2_in_expansion_l40_40509

theorem coefficient_x2_in_expansion : 
  ∃ c : ℤ, c = 60 ∧ (∀ x : ℝ, (1 - 2 * x)^6 = ∑ i in Finset.range 7, (binomial 6 i) * (-2)^i * x^i) := by 
  sorry

end coefficient_x2_in_expansion_l40_40509


namespace circle_area_l40_40524

theorem circle_area (θ : ℝ) : 
  let r := 3 * Real.cos θ - 4 * Real.sin θ in
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let center := (3 / 2 : ℝ, -2 : ℝ) in
  let radius := 5 / 2 in
  (x - center.1)^2 + (y - center.2)^2 = radius^2 →
  ∃ A : ℝ, A = (radius^2) * Real.pi ∧ A = 25 * Real.pi / 4 := 
by sorry

end circle_area_l40_40524


namespace number_of_ways_to_pick_shoes_l40_40797

theorem number_of_ways_to_pick_shoes (pairs : Finset (Finset (Fin 2))) (h : pairs.card = 4) :
  (∑ pair in pairs, pair.card) = 8 → 
  -- There are 2 shoes per pair and 4 pairs in total
  ∃! n, n = 16 := 
begin
  sorry
end

end number_of_ways_to_pick_shoes_l40_40797


namespace six_digit_numbers_with_zero_l40_40932

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40932


namespace vrox_staffing_ways_l40_40800

theorem vrox_staffing_ways :
  ∃ n, (nat.choose 21 7) * (nat.factorial 7) = n ∧ n = 3_315_480_000 := by
  sorry

end vrox_staffing_ways_l40_40800


namespace ladder_slide_out_l40_40191

noncomputable def ladderBaseSlideOut (L b1 y z : ℝ) : ℝ :=
  let init_height := sqrt (L^2 - b1^2)
  let new_height := init_height - y
  let new_base := sqrt (L^2 - new_height^2)
  new_base - b1

theorem ladder_slide_out (L b1 y : ℝ) (h1 : L = 30) (h2 : b1 = 8) (h3 : y = 6) : ladderBaseSlideOut L b1 y = 10 := by
  -- conditions
  have hL : L = 30 := h1
  have hb1 : b1 = 8 := h2
  have hy : y = 6 := h3
  sorry

end ladder_slide_out_l40_40191


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40947

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40947


namespace zeros_between_decimal_and_first_nonzero_digit_l40_40649

theorem zeros_between_decimal_and_first_nonzero_digit : 
  let frac : ℚ := 7 / 8000 in
  let exp : ℤ := 6 - nat_digits (875 : ℕ) in
  exp = 3 :=
by sorry

end zeros_between_decimal_and_first_nonzero_digit_l40_40649


namespace area_of_semicircle_l40_40681

theorem area_of_semicircle (w h : ℝ) (π : ℝ) (A : ℝ) 
  (h_w : w = 1) (h_h : h = 3) (h_geom : A = w * h):
  (1 / 2 * (π * (h / sqrt 2) ^ 2)) = 9 * π / 4 := 
by
  sorry

end area_of_semicircle_l40_40681


namespace intersection_complement_A_B_l40_40849

open Set

variable (A : Set ℕ) (B : Set ℝ) (complementB : Set ℝ)

def A_set : Set ℕ := { x | 0 ≤ x ∧ x ≤ 5 }
def B_set : Set ℝ := { x | 2 - x < 0 }
def complement_B : Set ℝ := { x | x ≤ 2 }

theorem intersection_complement_A_B : 
  (A_set ∩ (complement_B : Set ℝ) : Set ℕ) = {0, 1, 2} := by
  sorry

end intersection_complement_A_B_l40_40849


namespace rectangle_properties_l40_40009
-- Importing the entire Mathlib library

-- Define the conditions
def length : ℝ := sqrt 6 + 2 * sqrt 5
def width : ℝ := 2 * sqrt 6 - sqrt 5

-- Define the results to prove
def perimeter : ℝ := 6 * sqrt 6 + 2 * sqrt 5
def area : ℝ := 2 + 3 * sqrt 30

-- Stating the problem to prove
theorem rectangle_properties :
  (2 * (length + width) = perimeter) ∧ (length * width = area) :=
by
  -- Lean proof will be inserted here
  sorry

end rectangle_properties_l40_40009


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40937

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40937


namespace parameter_interval_l40_40796

noncomputable theory
open_locale classical

theorem parameter_interval
  (a : ℝ)
  (hx : ∀ x : ℝ, (x - 5) * real.sqrt (a * x + 2 * x - x^2 - 2 * a) ≤ 0)
  (h_difference : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 - x2).abs = 6) :
  a ∈ set.Iic (-4) ∪ set.Icc 8 11 :=
sorry

end parameter_interval_l40_40796


namespace graph_not_passing_through_origin_l40_40376

theorem graph_not_passing_through_origin (m : ℝ) (h : 3 * m^2 - 2 * m ≠ 0) : m = -(1 / 3) :=
sorry

end graph_not_passing_through_origin_l40_40376


namespace six_digit_numbers_with_zero_l40_40890

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40890


namespace six_digit_numbers_with_zero_count_l40_40902

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40902


namespace range_of_a_l40_40773

-- Given equation is: x^2 + y^2 + ax + 2ay + 2a^2 + a - 1 = 0

def represents_circle (a : ℝ) : Prop :=
  let expr := (x : ℝ) (y : ℝ) => x^2 + y^2 + a * x + 2 * a * y + 2 * a^2 + a - 1
  ∃ (h k r : ℝ), ∀ x y : ℝ, expr x y = (x - h)^2 + (y - k)^2 - r^2

theorem range_of_a :
  ∀ a : ℝ, represents_circle a ↔ a ∈ set.Ioo ℝ (-∞) 2 ∪ set.Ioo 2 ℝ (⊤) :=
by 
    -- Placeholder proof, to be filled in.
    sorry

end range_of_a_l40_40773


namespace percentage_increase_Sakshi_Tanya_l40_40105

def efficiency_Sakshi : ℚ := 1 / 5
def efficiency_Tanya : ℚ := 1 / 4
def percentage_increase_in_efficiency (eff_Sakshi eff_Tanya : ℚ) : ℚ :=
  ((eff_Tanya - eff_Sakshi) / eff_Sakshi) * 100

theorem percentage_increase_Sakshi_Tanya :
  percentage_increase_in_efficiency efficiency_Sakshi efficiency_Tanya = 25 :=
by
  sorry

end percentage_increase_Sakshi_Tanya_l40_40105


namespace remainder_of_8673_div_7_l40_40242

theorem remainder_of_8673_div_7 : 8673 % 7 = 3 :=
by
  -- outline structure, proof to be inserted
  sorry

end remainder_of_8673_div_7_l40_40242


namespace framing_required_l40_40683

/- 
  Problem: A 5-inch by 7-inch picture is enlarged by quadrupling its dimensions.
  A 3-inch-wide border is then placed around each side of the enlarged picture.
  What is the minimum number of linear feet of framing that must be purchased
  to go around the perimeter of the border?
-/
def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

theorem framing_required : (2 * ((original_width * enlargement_factor + 2 * border_width) + (original_height * enlargement_factor + 2 * border_width))) / 12 = 10 :=
by
  sorry

end framing_required_l40_40683


namespace avg_distance_is_600_l40_40436

-- Assuming Mickey runs half as many times as Johnny
def num_laps_johnny := 4
def num_laps_mickey := num_laps_johnny / 2
def lap_distance := 200

-- Calculating distances
def distance_johnny := num_laps_johnny * lap_distance
def distance_mickey := num_laps_mickey * lap_distance

-- Total distance run by both Mickey and Johnny
def total_distance := distance_johnny + distance_mickey

-- Average distance run by Johnny and Mickey
def avg_distance := total_distance / 2

-- Prove that the average distance run by Johnny and Mickey is 600 meters
theorem avg_distance_is_600 : avg_distance = 600 :=
by
  sorry

end avg_distance_is_600_l40_40436


namespace find_missing_number_l40_40277

theorem find_missing_number (n : ℤ) (h : 1234562 - n * 3 * 2 = 1234490) : 
  n = 12 :=
by
  sorry

end find_missing_number_l40_40277


namespace decimal_to_binary_49_l40_40512

theorem decimal_to_binary_49 : ((49:ℕ) = 6 * 2^4 + 3 * 2^3 + 0 * 2^2 + 0 * 2^1 + 0 * 2^0 + 1) ↔ (110001 = 110001) :=
by
  sorry

end decimal_to_binary_49_l40_40512


namespace circle_area_l40_40523

theorem circle_area (θ : ℝ) : 
  let r := 3 * Real.cos θ - 4 * Real.sin θ in
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  let center := (3 / 2 : ℝ, -2 : ℝ) in
  let radius := 5 / 2 in
  (x - center.1)^2 + (y - center.2)^2 = radius^2 →
  ∃ A : ℝ, A = (radius^2) * Real.pi ∧ A = 25 * Real.pi / 4 := 
by sorry

end circle_area_l40_40523


namespace transformed_parabola_eq_l40_40518

-- Definition of the initial quadratic function 
def initial_function (x : ℝ) : ℝ := 3 * x^2

-- Transformation functions
def shift_left (f : ℝ → ℝ) (a : ℝ) := λ x, f (x + a)
def shift_up (f : ℝ → ℝ) (b : ℝ) := λ x, f x + b

-- Statement of the problem
theorem transformed_parabola_eq : 
  let f := initial_function in
  (shift_up (shift_left f 2) 4) = λ x, 3 * (x + 2)^2 + 4 :=
by 
  sorry

end transformed_parabola_eq_l40_40518


namespace count_elements_begin_with_1_l40_40052

noncomputable def T : Set ℕ := {n | ∃ k : ℤ, 0 ≤ k ∧ k ≤ 1500 ∧ n = 3^k}

theorem count_elements_begin_with_1 :
  (∃ count, count = 784 ∧ ∀ n ∈ T, (nat.digits 10 n).head = 1 → count = count + 1) := 
sorry

end count_elements_begin_with_1_l40_40052


namespace six_digit_numbers_with_zero_l40_40961

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40961


namespace product_of_chords_l40_40408

theorem product_of_chords(
  (radius : ℝ) (h_rad : radius = 3)
  (A B : ℂ) 
  (D : ℕ → ℂ)
  (h_AB : |A - B| = 2 * radius)
  (h_dividing_points : (∀ i j : ℕ, i ≠ j → |D i - D j| = |D 0 - D 1|)):
  ∏ i in (finset.range 4), (
    |A - D i| * |B - D i| * |A - D (i + 1)| * |B - D (i + 1)|
  ) = 65610 :=
sorry

end product_of_chords_l40_40408


namespace difference_between_extremes_l40_40614

/-
  Define a set of digits and a property for forming four-digit numbers with a non-zero first digit.
-/
def digits : Set ℕ := {2, 0, 3, 5, 8}

def is_four_digit_number_with_digits (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n ≤ 9999) ∧ (∀ d ∈ digitList n, d ∈ digits) ∧ (digitList n ≠ [0])

def largest_number_with_digits : ℕ :=
  8532

def smallest_number_with_digits : ℕ :=
  2035

theorem difference_between_extremes :
  largest_number_with_digits - smallest_number_with_digits = 6497 :=
  by sorry

end difference_between_extremes_l40_40614


namespace number_of_trumpet_players_l40_40107

def number_of_people_in_orchestra := 21
def number_of_people_known := 1 -- Sebastian
                             + 4 -- Trombone players
                             + 1 -- French horn player
                             + 3 -- Violinists
                             + 1 -- Cellist
                             + 1 -- Contrabassist
                             + 3 -- Clarinet players
                             + 4 -- Flute players
                             + 1 -- Maestro

theorem number_of_trumpet_players : 
  number_of_people_in_orchestra = number_of_people_known + 2 :=
by
  sorry

end number_of_trumpet_players_l40_40107


namespace rationalize_denominator_l40_40100

theorem rationalize_denominator :
  ∃ (A B C D : ℤ), 
    D > 0 ∧
    ¬ (∃ p, prime p ∧ p^2 ∣ B) ∧
    Int.gcd (Int.gcd A C) D = 1 ∧ 
    A * B * (3 + Real.sqrt 8) + C * B = 7 ∧ 
    D = (3 + Real.sqrt 8) * (3 - Real.sqrt 8) ∧ 
    A + B + C + D = 23 :=
sorry

end rationalize_denominator_l40_40100


namespace video_streaming_budget_l40_40743

theorem video_streaming_budget 
  (weekly_food_budget : ℕ) 
  (weeks : ℕ) 
  (total_food_budget : ℕ) 
  (rent : ℕ) 
  (phone : ℕ) 
  (savings_rate : ℝ)
  (total_savings : ℕ) 
  (total_expenses : ℕ) 
  (known_expenses: ℕ) 
  (total_spending : ℕ):
  weekly_food_budget = 100 →
  weeks = 4 →
  total_food_budget = weekly_food_budget * weeks →
  rent = 1500 →
  phone = 50 →
  savings_rate = 0.10 →
  total_savings = 198 →
  total_expenses = total_food_budget + rent + phone →
  total_spending = (total_savings : ℝ) / savings_rate →
  known_expenses = total_expenses →
  total_spending - known_expenses = 30 :=
by sorry

end video_streaming_budget_l40_40743


namespace coffee_break_participants_l40_40581

theorem coffee_break_participants (n : ℕ) (h1 : n = 14) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 6 ∧ n - 2 * k > 0 ∧ n - 2 * k < n)
    (h3 : ∀ m, m < n → (∃ k, m = n - 2 * k → m ∈ {6, 8, 10, 12})): 
    ∃ m, n - 2 * m ∈ {6, 8, 10, 12} := 
by
  use 6
  use 8
  use 10
  use 12
  sorry

end coffee_break_participants_l40_40581


namespace janet_time_and_ratio_l40_40399

variables
  (blocks_north : ℕ := 3)
  (blocks_west : ℕ := 7 * blocks_north)
  (blocks_south : ℕ := 8)
  (walking_speed : ℕ := 2)

def time_to_walk_home : ℕ := blocks_west / walking_speed
def ratio_east_to_south : ℕ × ℕ := (blocks_west, blocks_south)

theorem janet_time_and_ratio :
  time_to_walk_home = 10.5 ∧ ratio_east_to_south = (21, 8) :=
by
  sorry

end janet_time_and_ratio_l40_40399


namespace digits_difference_l40_40491

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end digits_difference_l40_40491


namespace Emily_GRE_Exam_Date_l40_40274

theorem Emily_GRE_Exam_Date : 
  ∃ (exam_date : ℕ) (exam_month : String), 
  exam_date = 5 ∧ exam_month = "September" ∧
  ∀ study_days break_days start_day_cycles start_break_cycles start_month_june total_days S_june_remaining S_remaining_july S_remaining_august September_start_day, 
    study_days = 15 ∧ 
    break_days = 5 ∧ 
    start_day_cycles = 5 ∧ 
    start_break_cycles = 4 ∧ 
    start_month_june = 1 ∧
    total_days = start_day_cycles * study_days + start_break_cycles * break_days ∧ 
    S_june_remaining = 30 - start_month_june ∧ 
    S_remaining = total_days - S_june_remaining ∧ 
    S_remaining_july = S_remaining - 31 ∧ 
    S_remaining_august = S_remaining_july - 31 ∧ 
    September_start_day = S_remaining_august + 1 ∧
    exam_date = September_start_day ∧ 
    exam_month = "September" := by 
  sorry

end Emily_GRE_Exam_Date_l40_40274


namespace range_of_m_l40_40306

def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f (x + 2 * m) + m * f x > 0) ↔ (m > -1/4) :=
sorry

end range_of_m_l40_40306


namespace A_alone_work_in_days_l40_40687

theorem A_alone_work_in_days : 
  ∃ A : ℝ, (1 / A) + (1 / 6) + (1 / 12) = (1 / 2) ∧ A = 4 := 
by 
  exists 4
  split
  { ring_simp,
    norm_num },
  { refl }

end A_alone_work_in_days_l40_40687


namespace domain_width_p_l40_40365

variable (f : ℝ → ℝ)
variable (h_dom_f : ∀ x, -12 ≤ x ∧ x ≤ 12 → f x = f x)

noncomputable def p (x : ℝ) : ℝ := f (x / 3)

theorem domain_width_p : (width : ℝ) = 72 :=
by
  let domain_p : Set ℝ := {x | -36 ≤ x ∧ x ≤ 36}
  have : width = 72 := sorry
  exact this

end domain_width_p_l40_40365


namespace rods_and_connectors_10_row_pyramid_l40_40241

theorem rods_and_connectors_10_row_pyramid : 
  ∑ i in finset.range 10, 4 * (i + 1) + (4 * (i * (i + 1)) / 2) + 4 * 9 = 436 :=
by sorry

end rods_and_connectors_10_row_pyramid_l40_40241


namespace six_digit_numbers_with_zero_l40_40957

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40957


namespace speed_of_first_boy_proof_l40_40149

noncomputable def speed_of_first_boy := 5.9

theorem speed_of_first_boy_proof :
  ∀ (x : ℝ) (t : ℝ) (d : ℝ),
    (d = x * t) → (d = (x - 5.6) * 35) →
    d = 10.5 →
    t = 35 →
    x = 5.9 := 
by
  intros x t d h1 h2 h3 h4
  sorry

end speed_of_first_boy_proof_l40_40149


namespace find_p5_l40_40416

-- Definitions based on conditions from the problem
def p (x : ℝ) : ℝ :=
  x^4 - 10 * x^3 + 35 * x^2 - 50 * x + 18  -- this construction ensures it's a quartic monic polynomial satisfying provided conditions

-- The main theorem we want to prove
theorem find_p5 :
  p 1 = 3 ∧ p 2 = 7 ∧ p 3 = 13 ∧ p 4 = 21 → p 5 = 51 :=
by
  -- The proof will be inserted here later
  sorry

end find_p5_l40_40416


namespace ralph_final_cost_l40_40464

theorem ralph_final_cost :
  ∀ (total items_with_issue: ℝ) (issue_discount overall_discount: ℝ),
    total = 54.00 → items_with_issue = 20.00 → issue_discount = 0.20 → overall_discount = 0.10 →
    (total - items_with_issue * issue_discount) * (1 - overall_discount) = 45.00 :=
by
  intros total items_with_issue issue_discount overall_discount
  intro h_total h_items h_issue_discount h_overall_discount
  rw [h_total, h_items, h_issue_discount, h_overall_discount]
  norm_num
  sorry

end ralph_final_cost_l40_40464


namespace tangent_ball_to_prism_faces_l40_40373

variable (S S1 S2 : ℝ)

theorem tangent_ball_to_prism_faces 
  (hS1 : S1 ≥ 0) 
  (hS2 : S2 ≥ 0) 
  (hS : S ≥ 0) :
  (sqrt S = sqrt S1 + sqrt S2) := 
sorry

end tangent_ball_to_prism_faces_l40_40373


namespace zeros_between_decimal_and_first_nonzero_digit_l40_40650

theorem zeros_between_decimal_and_first_nonzero_digit : 
  let frac : ℚ := 7 / 8000 in
  let exp : ℤ := 6 - nat_digits (875 : ℕ) in
  exp = 3 :=
by sorry

end zeros_between_decimal_and_first_nonzero_digit_l40_40650


namespace rationalize_denominator_l40_40102

theorem rationalize_denominator :
  ∃ (A B C D : ℤ), 
    D > 0 ∧
    ¬ (∃ p, prime p ∧ p^2 ∣ B) ∧
    Int.gcd (Int.gcd A C) D = 1 ∧ 
    A * B * (3 + Real.sqrt 8) + C * B = 7 ∧ 
    D = (3 + Real.sqrt 8) * (3 - Real.sqrt 8) ∧ 
    A + B + C + D = 23 :=
sorry

end rationalize_denominator_l40_40102


namespace probability_max_min_difference_is_five_l40_40144

theorem probability_max_min_difference_is_five : 
  let total_outcomes := 6 ^ 4
  let outcomes_without_1 := 5 ^ 4
  let outcomes_without_6 := 5 ^ 4
  let outcomes_without_1_and_6 := 4 ^ 4
  total_outcomes - 2 * outcomes_without_1 + outcomes_without_1_and_6 = 302 →
  (302 : ℚ) / total_outcomes = 151 / 648 :=
by
  intros
  sorry

end probability_max_min_difference_is_five_l40_40144


namespace consumption_increase_percentage_l40_40143

theorem consumption_increase_percentage : 
  ∀ (T C : ℝ) (x : ℝ),
    (0.75 * T * C * (1 + x / 100) = 0.825 * T * C) →
    x = 10 :=
by
  intros T C x h
  have h1 : 0.75 * (1 + x / 100) = 0.825 :=
    by { rw [mul_assoc, ←mul_assoc T, mul_comm T, ←mul_assoc T C] at h, exact h }
  have h2 : 1 + x / 100 = 0.825 / 0.75 :=
    by { rw [h1, div_eq_mul_inv 0.825 0.75] }
  have h3 : x / 100 = 0.1 :=
    by { rw [←sub_eq_zero, sub_eq_add_neg, h2, sub_self] }
  have h4 : x = 10 :=
    by { rw [mul_div_cancel_left 10 (ne_of_lt (show 100 ≠ 0, by norm_num))] at h3, assumption }
  exact h4

end consumption_increase_percentage_l40_40143


namespace sum_of_real_nonnegative_roots_l40_40762

theorem sum_of_real_nonnegative_roots :
  (∀ x : ℝ, (0 ≤ x) → (x * real.sqrt x - 8 * x + 9 * real.sqrt x - 2 = 0) →
    x = 0 ∨ x = 1 ∨ x = 4 ∨ x = 9 ∨ x = 16 ∨ x = 25 ∨ x = 36 ∨ x = 46) :=
begin
  sorry
end

end sum_of_real_nonnegative_roots_l40_40762


namespace probability_last_red_ball_fourth_draw_l40_40115

-- Define the problem conditions
def initial_balls := (white: ℕ, red: ℕ) := (8, 2) -- 8 white balls and 2 red balls initially

def draw_and_replace (drawn: ℕ) : (ℕ × ℕ) → (ℕ × ℕ)
| (white, red) := if red > 0 then (white + 1, red - 1) else (white + 1, red)

-- Draw four times, replacing the drawn ball with a white one
def draw_four_times (state: (ℕ × ℕ)) := (draw_and_replace 1 (draw_and_replace 1 (draw_and_replace 1 (draw_and_replace 1 state))))

-- Check if the conditions meet the requirement
def meets_condition (state: (ℕ × ℕ)) : Prop :=
state = (11, 0) -- meaning last red ball is drawn on the fourth draw, initially was (8,2)

-- Calculate the probability
noncomputable def calculate_probability (initial_state : (ℕ × ℕ)) : ℚ :=
if meets_condition (draw_four_times initial_state) then (84 / 210 : ℚ) else 0

-- Proof statement
theorem probability_last_red_ball_fourth_draw :
  calculate_probability initial_balls = 2 / 5 :=
by sorry

end probability_last_red_ball_fourth_draw_l40_40115


namespace cost_difference_is_1595_l40_40799

-- Definitions based on the problem conditions
def rental_initial_monthly_cost : Int := 20
def rental_rate_increase : Int := 5
def rental_new_monthly_cost : Int := rental_initial_monthly_cost + rental_rate_increase
def rental_insurance_per_month : Int := 15
def rental_maintenance_first_6_months : Int := 10
def rental_maintenance_last_6_months : Int := 10

def buying_one_time_down_payment : Int := 1500
def buying_monthly_payment : Int := 30
def buying_insurance_per_month : Int := 20
def buying_maintenance_first_6_months : Int := 5
def buying_maintenance_last_6_months : Int := 10

-- Proof problem statement
theorem cost_difference_is_1595 :
  let rental_total_cost_first_month : Int := rental_initial_monthly_cost + rental_insurance_per_month + rental_maintenance_first_6_months
  let rental_total_cost_remaining_months : Int := rental_new_monthly_cost + rental_insurance_per_month + rental_maintenance_last_6_months
  let rental_total_cost : Int := rental_total_cost_first_month + (rental_total_cost_remaining_months * 11)
  let buying_total_cost_first_6_months : Int := buying_monthly_payment + buying_insurance_per_month + buying_maintenance_first_6_months
  let buying_total_cost_last_6_months : Int := buying_monthly_payment + buying_insurance_per_month + buying_maintenance_last_6_months
  let buying_total_cost : Int := buying_one_time_down_payment + (buying_total_cost_first_6_months * 6) + (buying_total_cost_last_6_months * 6)
  rental_total_cost = 45 + 550 → buying_total_cost = 2190 → (buying_total_cost - rental_total_cost) = 1595 :=
by
  intro rental_total_cost_eq buying_total_cost_eq
  rw [rental_total_cost_eq, buying_total_cost_eq]
  exact eq.refl 1595

end cost_difference_is_1595_l40_40799


namespace maria_sixth_test_score_l40_40072

theorem maria_sixth_test_score
  (s1 s2 s3 s4 s5 : ℤ)
  (s1_eq : s1 = 83)
  (s2_eq : s2 = 77)
  (s3_eq : s3 = 92)
  (s4_eq : s4 = 85)
  (s5_eq : s5 = 89)
  (avg : ℤ)
  (avg_eq : avg = 84) :
  let sixth_score := 504 - (s1 + s2 + s3 + s4 + s5)
  in sixth_score = 78 := by 
  sorry

end maria_sixth_test_score_l40_40072


namespace number_of_sequences_to_pick_two_cards_l40_40709

-- Defining the conditions
def total_cards : ℕ := 60

def number_of_suits : ℕ := 5

def cards_per_suit : ℕ := 12

-- Stating the theorem
theorem number_of_sequences_to_pick_two_cards :
  (number_of_suits * cards_per_suit = total_cards) →
  ((∑ i in finset.range total_cards, (total_cards - i - 1)) = 3540) :=
begin
  -- proof omitted
  sorry
end

end number_of_sequences_to_pick_two_cards_l40_40709


namespace six_digit_numbers_with_zero_l40_40979

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40979


namespace six_digit_numbers_with_zero_l40_40986

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40986


namespace lateral_surface_area_of_cone_l40_40808

-- Define the conditions
def cosine_angle_SA_SB : ℝ := 7 / 8
def angle_SA_base : ℝ := π / 4
def area_triangle_SAB : ℝ := 5 * real.sqrt 15

-- Define the statement that needs to be proved
theorem lateral_surface_area_of_cone :
  sorry /* The proof here */ :=
  lateral_surface_area = 40 * real.sqrt 2 * real.pi

end lateral_surface_area_of_cone_l40_40808


namespace correct_statement_about_algorithms_l40_40229

def algorithm_definitions : Prop :=
  ∀ (alg : Type),
  (∀ (P : Type), (P is_solved_by alg → alg.terminates) ∧ (P is_solved_by alg → alg.result_is_deterministic)) →
  ∃ (alg1 alg2 : alg), (alg1 ≠ alg2 ∧ alg1 is_solved_by_problem P ∧ alg2 is_solved_by_problem P)

theorem correct_statement_about_algorithms :
  ∃ (alg1 alg2 : Type),
  (∃ (P : Type), (P is_solved_by alg1 → alg1.terminates ∧ alg1.result_is_deterministic) ∧
                  (P is_solved_by alg2 → alg2.terminates ∧ alg2.result_is_deterministic) ∧
                  (alg1 ≠ alg2 ∧ alg1 is_solved_by_problem P ∧ alg2 is_solved_by_problem P)) :=
by
  apply algorithm_definitions; sorry

end correct_statement_about_algorithms_l40_40229


namespace midline_of_isosceles_trapezoid_l40_40692

/-- 
Given an isosceles trapezoid ABCD with longer leg length 'a' and a circle constructed on the smaller leg AB as its diameter, proving that the midline of the trapezoid is a/2.
--/
theorem midline_of_isosceles_trapezoid (a : ℝ) :
  let AB := (a : ℝ)
  let CD := (a : ℝ)
  -- Assuming other geometric properties as described
  midline CD = a / 2 :=
by
  sorry

end midline_of_isosceles_trapezoid_l40_40692


namespace unique_function_solution_l40_40786

-- Define f to be a function from positive reals to positive reals
def f (x : ℝ) : ℝ := x ^ 2 + (1 / x ^ 2)

-- Function f is increasing
axiom f_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f(x) < f(y)

-- Given condition for the functional equation
axiom functional_equation : ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
  f(a * b) * f(b) * f(c * a) = f(a ^ 2 * b ^ 2 * c ^ 2) + f(a ^ 2) + f(b ^ 2) + f(c ^ 2)

-- The statement to prove
theorem unique_function_solution : ∀ x : ℝ, 0 < x → f(x) = x ^ 2 + (1 / x ^ 2) :=
  sorry

end unique_function_solution_l40_40786


namespace find_vertices_and_circle_equation_l40_40351

open Real EuclideanGeometry

-- Define the vertices of triangle ABC
def vertex_A : Point := (0, 1)
def line_CD : Line := {a := 2, b := -2, c := -1}  -- Equation: 2x - 2y - 1 = 0
def line_BH : Line := {a := 0, b := 1, c := 0}    -- Equation: y = 0

def B : Point := (2, 0)
def C : Point := (0, -1/2)

-- Circle M passes through A, B, and P(m, 0), and a line with slope 1 is tangent at P(m, 0)
def circleM (m : ℝ) (x y : ℝ) := x^2 + y^2 + x + 5*y - 6 = 0

theorem find_vertices_and_circle_equation (m : ℝ) :
  let A := vertex_A
  let P := (m, 0)
  line_contains_point line_CD C ∧ line_contains_point line_BH A ∧
  midpoint (0, 1) B D ∧ line_contains_point line_CD D ∧
  (circleM m (fst A) (snd A) ∧ circleM m (fst B) (snd B) ∧ circleM m m 0) ∧
  line_tangent_to_circle (1, 1 - 1 * 0) (circleM m) P :=
  (B = (2, 0)) ∧ (C = (0, -1/2)) ∧
  (circleM m := λ x y, x^2 + y^2 + x + 5*y - 6 = 0) :=
by sorry

end find_vertices_and_circle_equation_l40_40351


namespace length_chord_eq_l40_40831

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : Prop :=
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity of the hyperbola
def eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = Real.sqrt (1 + b^2 / a^2)

-- Define the circle
def circle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the conditions
variables
  (a b : ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (h_hyperbola : hyperbola a b a_pos b_pos)
  (h_eccentricity : eccentricity a b (Real.sqrt 5))
  (h_circle : circle (2, 3) 1)

-- Prove the length of the chord |AB|
theorem length_chord_eq :
  ∃ A B : ℝ × ℝ,
  (∃ L : ℝ, L = 2 * Real.sqrt ((1:ℝ)^2 - (1 / Real.sqrt 5)^2))
  ∧ |A.1 - B.1| + |A.2 - B.2| = 4 * Real.sqrt 5 / 5 := sorry

end length_chord_eq_l40_40831


namespace expression_equals_36_l40_40057

def k := 13

theorem expression_equals_36 : 13 * (3 - 3 / 13) = 36 := by
  sorry

end expression_equals_36_l40_40057


namespace six_digit_numbers_with_zero_count_l40_40904

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40904


namespace solve_for_x_l40_40304

def δ (x : ℝ) : ℝ := 4 * x + 5
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_for_x (x : ℝ) (h : δ (φ x) = 4) : x = -17 / 20 := by
  sorry

end solve_for_x_l40_40304


namespace platform_length_259_9584_l40_40207

noncomputable def length_of_platform (speed_kmph time_sec train_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600  -- conversion from kmph to m/s
  let distance_covered := speed_mps * time_sec
  distance_covered - train_length_m

theorem platform_length_259_9584 :
  length_of_platform 72 26 260.0416 = 259.9584 :=
by sorry

end platform_length_259_9584_l40_40207


namespace travel_distance_is_96_l40_40083

-- Define the conditions
variable {x : ℝ}
variable (t_monday t_tuesday : ℝ)

-- Assume the time taken on Monday
def time_monday : ℝ := x / 90

-- Assume the time taken on Tuesday
def time_tuesday : ℝ := x / 120

-- Assume the time difference in hours
def time_difference : ℝ := 4 / 15

-- The proof problem statement
theorem travel_distance_is_96 (h1 : t_monday = time_monday)
                              (h2 : t_tuesday = time_tuesday)
                              (h_diff : t_monday - t_tuesday = time_difference) :
  x = 96 :=
by
  -- Proof can be filled in here
  sorry

end travel_distance_is_96_l40_40083


namespace six_digit_numbers_with_zero_count_l40_40905

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40905


namespace six_digit_numbers_with_zero_l40_40959

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40959


namespace total_hair_cut_l40_40780

-- Define the amounts cut on two consecutive days
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- Statement: Prove that the total amount cut off is 0.875 inches
theorem total_hair_cut : first_cut + second_cut = 0.875 :=
by {
  -- The exact proof would go here
  sorry
}

end total_hair_cut_l40_40780


namespace determine_lambda_l40_40327

open_locale big_operators

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

theorem determine_lambda
  (P A B C O : V)
  (h_coplanar : ∃ m n : ℝ, ∀ (A B C P : V), P = m • A + n • B + (1 - m - n) • C)
  (h_eq : P = 2 • A + B + λ • C) :
  λ = 2 :=
sorry

end determine_lambda_l40_40327


namespace westbound_vehicle_count_l40_40688

theorem westbound_vehicle_count 
  (eastbound_speed : ℝ) (westbound_speed : ℝ) (observed_vehicles : ℕ) 
  (interval_minutes : ℝ) (stop_minutes : ℝ) (highway_section : ℝ) :
  eastbound_speed = 70 → 
  westbound_speed = 60 →
  observed_vehicles = 15 →
  interval_minutes = 10 →
  stop_minutes = 2 →
  highway_section = 150 →
  (observed_vehicles : ℝ) / 
    ((eastbound_speed + westbound_speed) * (interval_minutes - stop_minutes) / 60) * 
    highway_section ≈ 130 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end westbound_vehicle_count_l40_40688


namespace circle_area_l40_40542

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l40_40542


namespace coffee_break_participants_l40_40579

theorem coffee_break_participants (n : ℕ) (h1 : n = 14) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 6 ∧ n - 2 * k > 0 ∧ n - 2 * k < n)
    (h3 : ∀ m, m < n → (∃ k, m = n - 2 * k → m ∈ {6, 8, 10, 12})): 
    ∃ m, n - 2 * m ∈ {6, 8, 10, 12} := 
by
  use 6
  use 8
  use 10
  use 12
  sorry

end coffee_break_participants_l40_40579


namespace geometric_sequence_sum_l40_40142

theorem geometric_sequence_sum (S : ℕ → ℚ) (n : ℕ) 
  (hS_n : S n = 54) 
  (hS_2n : S (2 * n) = 60) 
  : S (3 * n) = 60 + 2 / 3 := 
sorry

end geometric_sequence_sum_l40_40142


namespace ratio_of_britney_susie_reds_l40_40495

noncomputable def ratio_of_rhode_island_reds (britney_red_birds susie_red_birds : ℕ) 
  (susie_reds : susie_red_birds = 11)
  (susie_comets : susie_comets = 6)
  (half_britney_comets : britney_comets = susie_comets / 2)
  (more_britney_chickens : britney_red_birds + britney_comets = susie_red_birds + susie_comets + 8) 
  : ℕ := sorry

theorem ratio_of_britney_susie_reds :
  ratio_of_rhode_island_reds britney_red_birds susie_red_birds susie_reds susie_comets half_britney_comets more_britney_chickens = 2 := sorry

end ratio_of_britney_susie_reds_l40_40495


namespace milkshakes_per_hour_l40_40731

variable (L : ℕ) -- number of milkshakes Luna can make per hour

theorem milkshakes_per_hour
  (h1 : ∀ (A : ℕ), A = 3) -- Augustus makes 3 milkshakes per hour
  (h2 : ∀ (H : ℕ), H = 8) -- they have been making milkshakes for 8 hours
  (h3 : ∀ (Total : ℕ), Total = 80) -- together they made 80 milkshakes
  (h4 : ∀ (Augustus_milkshakes : ℕ), Augustus_milkshakes = 3 * 8) -- Augustus made 24 milkshakes in 8 hours
 : L = 7 := sorry

end milkshakes_per_hour_l40_40731


namespace right_triangle_area_l40_40213

theorem right_triangle_area (x : ℝ) 
  (h1 : ∠ABC = 30) 
  (h2 : hypotenuse = 20) 
  (h3 : base = x) :
  ∃ x, area = 50 * sqrt 3 :=
begin
  sorry
end

end right_triangle_area_l40_40213


namespace function_fixed_point_l40_40126

theorem function_fixed_point (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : 
    ∃ x y : ℝ, (y = a^(x - 2016) + 2016) ∧ x = 2016 ∧ y = 2017 :=
by
  use [2016, 2017]
  sorry

end function_fixed_point_l40_40126


namespace smallest_prime_divisor_of_n_is_2_l40_40634

def a := 3^19
def b := 11^13
def n := a + b

theorem smallest_prime_divisor_of_n_is_2 : nat.prime_divisors n = {2} :=
by
  -- The proof is not needed as per instructions.
  -- We'll add a placeholder sorry to complete the statement.
  sorry

end smallest_prime_divisor_of_n_is_2_l40_40634


namespace coffee_participants_l40_40606

noncomputable def participants_went_for_coffee (total participants : ℕ) (stay : ℕ) : Prop :=
  (0 < stay) ∧ (stay < 14) ∧ (stay % 2 = 0) ∧ ∃ k : ℕ, stay = 2 * k

theorem coffee_participants :
  ∀ (total participants : ℕ), total participants = 14 →
  (∀ participant, ((∃ k : ℕ, stay = 2 * k) → (0 < stay) ∧ (stay < 14))) →
  participants_went_for_coffee total_participants (14 - stay) =
  (stay = 12) ∨ (stay = 10) ∨ (stay = 8) ∨ (stay = 6) :=
by
  sorry

end coffee_participants_l40_40606


namespace problem_l40_40485

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end problem_l40_40485


namespace find_b_l40_40390

noncomputable def find_angle (a : ℝ) (b : ℝ) : Prop :=
PBC_is_straight_line ∧ PA_touches_circle_at_A ∧ AB_eq_PB ∧ angle_ACB_eq_a

theorem find_b (a b : ℝ) (h : find_angle a b) : 
  b = 180 - 2 * a :=
sorry

end find_b_l40_40390


namespace ralph_final_cost_l40_40463

theorem ralph_final_cost :
  ∀ (total items_with_issue: ℝ) (issue_discount overall_discount: ℝ),
    total = 54.00 → items_with_issue = 20.00 → issue_discount = 0.20 → overall_discount = 0.10 →
    (total - items_with_issue * issue_discount) * (1 - overall_discount) = 45.00 :=
by
  intros total items_with_issue issue_discount overall_discount
  intro h_total h_items h_issue_discount h_overall_discount
  rw [h_total, h_items, h_issue_discount, h_overall_discount]
  norm_num
  sorry

end ralph_final_cost_l40_40463


namespace find_A_d_minus_B_d_l40_40486

-- Definitions for the proof problem
variables {d A B : ℕ}
variables {ad bd : ℤ} -- Representing A_d and B_d in ℤ for arithmetic operations

-- Conditions
constants (base_check : d > 7)
constants (digit_check : A < d ∧ B < d)
constants (encoded_check : d^1 * A + d^0 * B + d^2 * A + d^1 * A = 1 * d^2 + 7 * d^1 + 2 * d^0)

-- The theorem to prove
theorem find_A_d_minus_B_d : A - B = 5 :=
good sorry

end find_A_d_minus_B_d_l40_40486


namespace tom_seashells_l40_40607

theorem tom_seashells (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) : total_seashells = 35 := 
by 
  sorry

end tom_seashells_l40_40607


namespace asymptotes_of_hyperbola_l40_40125

-- Definition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 9 = 1

-- Definition of the equations of the asymptotes
def asymptote_eq (x y : ℝ) : Prop := y = (3/4)*x ∨ y = -(3/4)*x

-- Theorem statement
theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_eq x y → asymptote_eq x y :=
sorry

end asymptotes_of_hyperbola_l40_40125


namespace diff_of_sequences_l40_40738

-- Define the first sequence as an arithmetic sequence
def seq1 := (list.range 100).map (λ n, 2501 + n)

-- Define the second sequence as an arithmetic sequence
def seq2 := (list.range 100).map (λ n, 201 + n)

-- Define a function to calculate the sum of elements of a list
def sum (l : list ℕ) : ℕ :=
  l.foldr (λ a b, a + b) 0

-- Main statement: Prove the difference between the sums of the two sequences
theorem diff_of_sequences :
  sum seq1 - sum seq2 = 230000 :=
by
  sorry

end diff_of_sequences_l40_40738


namespace price_increase_percentage_l40_40551

theorem price_increase_percentage (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 360) : 
  (new_price - original_price) / original_price * 100 = 20 := 
by
  sorry

end price_increase_percentage_l40_40551


namespace tv_show_length_is_correct_l40_40435

def commercial_lengths := [7 * 2, 13, 5, 9 * 2]
def additional_breaks := [4, 2, 8]

def total_show_time_minutes := (2 * 60) + 30

def total_time_without_commercials_and_breaks (commercials : List ℕ) (breaks : List ℕ) : ℝ :=
  let total_commercials := commercials.sum
  let total_breaks := breaks.sum
  (total_show_time_minutes - (total_commercials + total_breaks)).toReal / 60

def tv_show_length_hours := total_time_without_commercials_and_breaks commercial_lengths additional_breaks

theorem tv_show_length_is_correct :
  tv_show_length_hours = 1.4333 := by
  sorry

end tv_show_length_is_correct_l40_40435


namespace finite_decimal_fractions_count_l40_40765

theorem finite_decimal_fractions_count :
  ∃ count : ℕ, (∀ n : ℕ, (2 ≤ n ∧ n ≤ 2016) →
    (∃ a b : ℕ, n = 2^a * 5^b)) → count = 33 :=
by
  let := sorry

end finite_decimal_fractions_count_l40_40765


namespace locus_of_M_is_Apollonius_l40_40308

-- Define the collinear points A, O, B and their distances on a line
variables (A O B : Point) (a b : ℝ)
variable (x : ℝ)

-- Conditions: Points are collinear, distances, and a > b
axiom collinear_AOB : collinear ([A, O, B])
axiom dist_AO_eq_a : dist A O = a
axiom dist_OB_eq_b : dist O B = b
axiom a_gt_b : a > b

-- Condition: Circle with radius x centered at O
axiom circle_O_x : ∀ x, 0 < x ∧ x < b → Circle O x

-- Condition: Tangents from A and B to the circle intersect at M
axiom tangents_intersect_at_M : ∀ (x : ℝ), 0 < x ∧ x < b → ∃ M, is_tangent A (Circle O x) ∧ is_tangent B (Circle O x) ∧ intersecting_tangents A B M

-- Hypothesis: Locus of point M as x varies from 0 to b
theorem locus_of_M_is_Apollonius : 
  (∀ x, 0 < x ∧ x < b → ∃ M, is_on_locus M (Apollonius_circle A B (a / b))) := 
by {
  sorry
}

end locus_of_M_is_Apollonius_l40_40308


namespace find_counterfeit_coins_l40_40728

structure Coins :=
  (a a₁ b b₁ c c₁ : ℝ)
  (genuine_weight : ℝ)
  (counterfeit_weight : ℝ)
  (a_is_genuine_or_counterfeit : a = genuine_weight ∨ a = counterfeit_weight)
  (a₁_is_genuine_or_counterfeit : a₁ = genuine_weight ∨ a₁ = counterfeit_weight)
  (b_is_genuine_or_counterfeit : b = genuine_weight ∨ b = counterfeit_weight)
  (b₁_is_genuine_or_counterfeit : b₁ = genuine_weight ∨ b₁ = counterfeit_weight)
  (c_is_genuine_or_counterfeit : c = genuine_weight ∨ c = counterfeit_weight)
  (c₁_is_genuine_or_counterfeit : c₁ = genuine_weight ∨ c₁ = counterfeit_weight)
  (counterfeit_pair_ends_unit_segment : (a = counterfeit_weight ∧ a₁ = counterfeit_weight) 
                                        ∨ (b = counterfeit_weight ∧ b₁ = counterfeit_weight)
                                        ∨ (c = counterfeit_weight ∧ c₁ = counterfeit_weight))

theorem find_counterfeit_coins (coins : Coins) : 
  (coins.a = coins.genuine_weight ∧ coins.b = coins.genuine_weight → coins.a₁ = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.a < coins.b → coins.a = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.b < coins.a → coins.b = coins.counterfeit_weight ∧ coins.a₁ = coins.counterfeit_weight) := 
by
  sorry

end find_counterfeit_coins_l40_40728


namespace probability_of_blank_l40_40018

-- Definitions based on conditions
def num_prizes : ℕ := 10
def num_blanks : ℕ := 25
def total_outcomes : ℕ := num_prizes + num_blanks

-- Statement of the proof problem
theorem probability_of_blank : (num_blanks / total_outcomes : ℚ) = 5 / 7 :=
by {
  sorry
}

end probability_of_blank_l40_40018


namespace area_of_circle_l40_40531

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l40_40531


namespace sum_of_digits_707070707_909090909_l40_40135

-- Definitions of the conditions
def number1 := Int.ofNat (String.toNat "707070707070707" -- Repeat this pattern to get the 101-digit number
  -- Note: Construct the 101 digit number properly or use a more abstract representation
def number2 := Int.ofNat (String.toNat "909090909090909" -- Repeat this pattern to get the 101-digit number
  -- Note: Construct the 101 digit number properly or use a more abstract representation

-- Correct answer
theorem sum_of_digits_707070707_909090909 :
  let product := number1 * number2
  let hundreds_digit := (product / 100 % 10)
  let units_digit := (product % 10)
  hundreds_digit + units_digit = 7 :=
by
  sorry

end sum_of_digits_707070707_909090909_l40_40135


namespace six_digit_numbers_with_zero_count_l40_40903

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40903


namespace ratio_area_ARQ_ABC_l40_40180

variable {α : Type*} [LinearOrderedField α]
variables (A B C P Q R : Point α) (k : α) 

-- Definitions of points, lines, and conditions based on the description
variable (AB AC BC : Line α)
variable (RBP QPC : Triangle α)
variable [SimilarTriangles RBP QPC]

-- Given points on lines and parallel lines
variable (hPBC : P ∈ BC)
variable (hQAC : Q ∈ AC)
variable (hRAB : R ∈ AB)
variable (h_parallel_R_AC : Parallel α P AB AC)
variable (h_parallel_P_AB : Parallel α Q P AB)
variable (h_ratio_RBP_QPC : Area(RBP) = k^2 * Area(QPC))

-- Theorem to prove the desired ratio
theorem ratio_area_ARQ_ABC :
  Area(Triangle.insert A R Q) / Area(Triangle.insert A B C) = k / (1 + k)^2 :=
by
  sorry

end ratio_area_ARQ_ABC_l40_40180


namespace cream_ratio_in_coffee_l40_40736

def bob_initial_coffee_volume : ℕ := 20
def bob_drank_coffee_volume : ℕ := 3
def bob_added_cream_volume : ℕ := 4

def bella_initial_coffee_volume : ℕ := 20
def bella_added_cream_volume : ℕ := 4
def bella_drank_volume : ℕ := 3

theorem cream_ratio_in_coffee :
  let bob_final_cream_volume := bob_added_cream_volume in
  let bella_total_volume := bella_initial_coffee_volume + bella_added_cream_volume in
  let bella_cream_ratio := (bella_added_cream_volume : ℚ) / (bella_total_volume : ℚ) in
  let bella_removed_cream := bella_cream_ratio * (bella_drank_volume : ℚ) in
  let bella_final_cream_volume := (bella_added_cream_volume : ℚ) - bella_removed_cream in
  bob_final_cream_volume / bella_final_cream_volume = (8 / 7 : ℚ) :=
by
  sorry

end cream_ratio_in_coffee_l40_40736


namespace problem_solution_l40_40046

noncomputable def exists_A : Prop :=
  ∃ A > 0, ∀ (n : ℕ), n > 1 → ∀ (x : Fin n → ℝ), 
  (∀ i j : Fin n, i ≠ j → x i ≠ x j) → 
  ∃ (seq : List (List ℝ)),
  seq.head = List.ofFn x ∧
  seq.last = seq.head.qsort (· < ·) ∧
  seq.length ≤ A * n * log n

noncomputable def exists_B : Prop :=
  ∃ B > 0, ∀ (n : ℕ), n > 1 → ∃ (x : Fin n → ℝ),
  (∀ i j : Fin n, i ≠ j → x i ≠ x j) ∧ 
  ∀ (seq : List (List ℝ)),
  seq.head = List.ofFn x →
  seq.last = seq.head.qsort (· < ·) →
  seq.length ≥ B * n * log n

-- Example of top level theorem that encompasses both problems
theorem problem_solution : exists_A ∧ exists_B :=
by sorry

end problem_solution_l40_40046


namespace rationalize_fraction_sum_l40_40092

theorem rationalize_fraction_sum :
  ∃ (A B C D : ℤ),
    D > 0 ∧
    ¬ ∃ p : ℕ, p.prime ∧ p^2 ∣ B ∧
    Int.gcd A C D = 1 ∧
    A * 8 + C = 21 - 7 * 3 ∧
    A + B + C + D = 23 :=
sorry

end rationalize_fraction_sum_l40_40092


namespace participants_coffee_l40_40588

theorem participants_coffee (n : ℕ) (h1 : n = 14) (h2 : 0 < (14 - 2 * (n / 2)) < 14) : 
  ∃ (k : ℕ), k ∈ {6, 8, 10, 12} ∧ k = (14 - 2 * (n / 2)) :=
by sorry

end participants_coffee_l40_40588


namespace circle_area_l40_40545

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l40_40545


namespace trigonometric_identity_l40_40760

theorem trigonometric_identity :
  tan (40 * Real.pi / 180) + 3 * sin (40 * Real.pi / 180) + 2 * cos (20 * Real.pi / 180) =
  4 * sin (55 * Real.pi / 180) * cos (15 * Real.pi / 180) :=
sorry

end trigonometric_identity_l40_40760


namespace compute_k_l40_40419

noncomputable def problem_statement (p q : ℤ) (x : ℝ) : Prop :=
  tan x = p / q ∧ 
  tan (3 * x) = q / (p + q) ∧ 
  0 < x

theorem compute_k (p q : ℤ) (x : ℝ) (h : problem_statement p q x) : k = 1 / 2 :=
  sorry

end compute_k_l40_40419


namespace six_digit_numbers_with_at_least_one_zero_l40_40975

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40975


namespace pyramid_height_volume_eq_cube_l40_40199

theorem pyramid_height_volume_eq_cube (h : ℝ) :
  let edge_length : ℝ := 6
      side_length : ℝ := 10
      V_cube : ℝ := edge_length^3
      area_base : ℝ := (sqrt 3 / 4) * side_length^2
      V_pyramid : ℝ := (1 / 3) * area_base * h
  in V_cube = V_pyramid → h = (216 * sqrt 3) / 25 :=
by
  sorry

end pyramid_height_volume_eq_cube_l40_40199


namespace smallest_prime_divisor_of_sum_l40_40627

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end smallest_prime_divisor_of_sum_l40_40627


namespace rectangle_area_l40_40386

noncomputable def area_of_rectangle (PQRS : Type) [rectangle PQRS] (S G H : PQRS)
  (trisect : trisects S G H)
  (on_PQ_H : on_side H PQ)
  (on_PS_G : on_side G PS)
  (QH : distance Q H = 4)
  (PG : distance P G = 3) : ℝ :=
36 * real.sqrt 3 - 36

theorem rectangle_area (PQRS : Type) [rectangle PQRS] (S G H : PQRS)
  (trisect : trisects S G H)
  (on_PQ_H : on_side H PQ)
  (on_PS_G : on_side G PS)
  (QH : distance Q H = 4)
  (PG : distance P G = 3) :
  area_of_rectangle PQRS S G H trisect on_PQ_H on_PS_G QH PG = 36 * real.sqrt 3 - 36 :=
sorry

end rectangle_area_l40_40386


namespace problem1_problem2_l40_40244

theorem problem1 : (sqrt 48 + sqrt 20) - (sqrt 12 - sqrt 5) = 2 * sqrt 3 + 3 * sqrt 5 := 
sorry

theorem problem2 : |2 - sqrt 2| - sqrt (1/12) * sqrt 27 + (sqrt 12 / sqrt 6) = 1 / 2 := 
sorry

end problem1_problem2_l40_40244


namespace net_percentage_error_in_volume_l40_40232

theorem net_percentage_error_in_volume
  (a : ℝ)
  (side_error : ℝ := 0.03)
  (height_error : ℝ := -0.04)
  (depth_error : ℝ := 0.02) :
  ((1 + side_error) * (1 + height_error) * (1 + depth_error) - 1) * 100 = 0.8656 :=
by
  -- Placeholder for the proof
  sorry

end net_percentage_error_in_volume_l40_40232


namespace average_distance_is_600_l40_40442

-- Definitions based on the given conditions
def distance_around_block := 200
def johnny_rounds := 4
def mickey_rounds := johnny_rounds / 2

-- The calculated distances
def johnny_distance := johnny_rounds * distance_around_block
def mickey_distance := mickey_rounds * distance_around_block

-- The average distance computation
def average_distance := (johnny_distance + mickey_distance) / 2

-- The theorem to prove that the average distance is 600 meters
theorem average_distance_is_600 : average_distance = 600 := by sorry

end average_distance_is_600_l40_40442


namespace cannot_serve_as_general_formula_l40_40328

def seq_2_0_2_0 (n : ℕ) : ℕ :=
  if n % 2 = 0 then 0 else 2

def a (n : ℕ) : ℕ := 2 * Real.sin (n * Real.pi)

theorem cannot_serve_as_general_formula :
  ¬ (∀ n,  a n = seq_2_0_2_0 n) :=
begin
  sorry
end

end cannot_serve_as_general_formula_l40_40328


namespace coffee_participants_l40_40605

noncomputable def participants_went_for_coffee (total participants : ℕ) (stay : ℕ) : Prop :=
  (0 < stay) ∧ (stay < 14) ∧ (stay % 2 = 0) ∧ ∃ k : ℕ, stay = 2 * k

theorem coffee_participants :
  ∀ (total participants : ℕ), total participants = 14 →
  (∀ participant, ((∃ k : ℕ, stay = 2 * k) → (0 < stay) ∧ (stay < 14))) →
  participants_went_for_coffee total_participants (14 - stay) =
  (stay = 12) ∨ (stay = 10) ∨ (stay = 8) ∨ (stay = 6) :=
by
  sorry

end coffee_participants_l40_40605


namespace fenced_area_l40_40519

theorem fenced_area (length_large : ℕ) (width_large : ℕ) 
                    (length_cutout : ℕ) (width_cutout : ℕ) 
                    (h_large : length_large = 20 ∧ width_large = 15)
                    (h_cutout : length_cutout = 4 ∧ width_cutout = 2) : 
                    ((length_large * width_large) - (length_cutout * width_cutout) = 292) := 
by
  sorry

end fenced_area_l40_40519


namespace smallest_prime_divisor_of_sum_l40_40626

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end smallest_prime_divisor_of_sum_l40_40626


namespace smallest_prime_divisor_of_sum_l40_40641

theorem smallest_prime_divisor_of_sum (h1 : Nat.odd (Nat.pow 3 19))
                                      (h2 : Nat.odd (Nat.pow 11 13))
                                      (h3 : ∀ a b : Nat, Nat.odd a → Nat.odd b → Nat.even (a + b)) :
  Nat.minFac (Nat.pow 3 19 + Nat.pow 11 13) = 2 :=
by
  -- placeholder proof
  sorry

end smallest_prime_divisor_of_sum_l40_40641


namespace sum_of_smallest_and_largest_prime_l40_40069

theorem sum_of_smallest_and_largest_prime :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in
  (primes.head! + primes.reverse.head! = 49) := 
by
  sorry

end sum_of_smallest_and_largest_prime_l40_40069


namespace coffee_break_l40_40576

variable {n : ℕ}
variables {participants : Finset ℕ} (hparticipants : participants.card = 14)
variables (left : Finset ℕ)
variables (stayed : Finset ℕ) (hstayed : stayed.card = 14 - 2 * left.card)

-- The overall proof problem statement:
theorem coffee_break (h : ∀ x ∈ stayed, ∃! y, (y ∈ left ∧ adjacent x y participants)) :
  left.card = 6 ∨ left.card = 8 ∨ left.card = 10 ∨ left.card = 12 := 
sorry

end coffee_break_l40_40576


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40919

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40919


namespace ratio_population_XZ_l40_40667

variable (Population : Type) [Field Population]
variable (Z : Population) -- Population of City Z
variable (Y : Population) -- Population of City Y
variable (X : Population) -- Population of City X

-- Conditions
def population_Y : Y = 2 * Z := sorry
def population_X : X = 7 * Y := sorry

-- Theorem stating the ratio of populations
theorem ratio_population_XZ : (X / Z) = 14 := by
  -- The proof will use the conditions population_Y and population_X
  sorry

end ratio_population_XZ_l40_40667


namespace find_d_of_ellipse_tangent_l40_40230

noncomputable def ellipse_focus (d : ℝ) : ℝ × ℝ := (d, 9)

theorem find_d_of_ellipse_tangent (d : ℝ) :
  (∃ e : ellipse, e.center = (d + 5) / 2 ∧ e.tangent_x_axis ∧ e.tangent_y_axis ∧ e.focus1 = (5, 9) ∧ e.focus2 = ellipse_focus d) → d = 29.9 :=
sorry

end find_d_of_ellipse_tangent_l40_40230


namespace integral_of_sqrt_and_linear_l40_40783

theorem integral_of_sqrt_and_linear:
  ∫ x in 0..1, (Real.sqrt(1 - x^2) + x) = (Real.pi + 2) / 4 :=
by
  sorry

end integral_of_sqrt_and_linear_l40_40783


namespace remaining_bottles_after_2_days_l40_40204

-- Definitions based on the conditions:
def initial_bottles : ℕ := 24
def fraction_first_day : ℚ := 1 / 3
def fraction_second_day : ℚ := 1 / 2

-- Theorem statement proving the remaining number of bottles after 2 days
theorem remaining_bottles_after_2_days : 
    (initial_bottles - initial_bottles * fraction_first_day) - 
    ((initial_bottles - initial_bottles * fraction_first_day) * fraction_second_day) = 8 := 
by 
    -- Skipping the proof
    sorry

end remaining_bottles_after_2_days_l40_40204


namespace infinitely_many_n_l40_40059

theorem infinitely_many_n :
  ∀ (k : ℕ), k ≥ 2 → ∃∞ n : ℕ, ∃ (u v : ℕ), k * n + 1 = u ^ 2 ∧ (k + 1) * n + 1 = v ^ 2 :=
by
  intros k hk
  sorry

end infinitely_many_n_l40_40059


namespace expected_value_l40_40015

-- Define the scenario
def num_red_balls := 2
def num_yellow_balls := 2
def num_blue_balls := 2
def total_balls := num_red_balls + num_yellow_balls + num_blue_balls

-- Define the expected value function
noncomputable def E_X := (1 * (3 * (2 * 1 / (total_balls * (total_balls - 1))))) +
                         (2 * ((choose 3 1) * (2 / (total_balls * (total_balls - 1)) * 4 / (total_balls * (total_balls - 1) - 1)))) +
                         (3 * ((choose 2 1) * (2 / total_balls * (choose 2 1) * (2 / (total_balls - 1) * (choose 2 1) * (1 / (total_balls - 2))))))

-- Define and prove the expected value
theorem expected_value : E_X = 11 / 5 :=
by
  sorry

end expected_value_l40_40015


namespace hyperbola_eccentricity_asymptotes_l40_40516

theorem hyperbola_eccentricity_asymptotes :
  (∃ e: ℝ, ∃ m: ℝ, 
    (∀ x y, (x^2 / 8 - y^2 / 4 = 1) → e = Real.sqrt 6 / 2 ∧ y = m * x) ∧ 
    (m = Real.sqrt 2 / 2 ∨ m = -Real.sqrt 2 / 2)) :=
sorry

end hyperbola_eccentricity_asymptotes_l40_40516


namespace monica_total_savings_l40_40073

theorem monica_total_savings :
  ∀ (weekly_saving : ℤ) (weeks_per_cycle : ℤ) (cycles : ℤ),
    weekly_saving = 15 →
    weeks_per_cycle = 60 →
    cycles = 5 →
    weekly_saving * weeks_per_cycle * cycles = 4500 :=
by
  intros weekly_saving weeks_per_cycle cycles
  sorry

end monica_total_savings_l40_40073


namespace six_digit_numbers_with_zero_l40_40867

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40867


namespace train_speed_l40_40716

theorem train_speed (length_in_meters : ℝ) (time_in_seconds : ℝ) (speed_in_kmph : ℝ) : 
  length_in_meters = 500 ∧ time_in_seconds = 18 → speed_in_kmph = 100 :=
by
  intros h
  have length_eq : length_in_meters = 500 := by
    exact h.left
  have time_eq : time_in_seconds = 18 := by
    exact h.right
  -- speed = (500 meters / 18 seconds) * (18/5) (conversion factor to km/h)
  calc speed_in_kmph = (500 / 18) * 3.6 : by sorry
    ... = 100 : by sorry

end train_speed_l40_40716


namespace six_digit_numbers_with_at_least_one_zero_l40_40864

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40864


namespace fair_hair_percentage_l40_40680

-- Define the main entities
variables (E F W : ℝ)

-- Define the conditions given in the problem
def women_with_fair_hair : Prop := W = 0.32 * E
def fair_hair_women_ratio : Prop := W = 0.40 * F

-- Define the theorem to prove
theorem fair_hair_percentage
  (hwf: women_with_fair_hair E W)
  (fhr: fair_hair_women_ratio W F) :
  (F / E) * 100 = 80 :=
by
  sorry

end fair_hair_percentage_l40_40680


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40950

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40950


namespace distance_to_pinedale_mall_l40_40497

def bus_speed : ℕ := 60 -- km/h
def time_between_stops : ℕ := 5 -- minutes
def num_stops : ℕ := 6 

theorem distance_to_pinedale_mall (speed : ℕ) (time_per_stop : ℕ) (stops : ℕ) : ℕ :=
  let time := stops * time_per_stop / 60 in
  speed * time

example : distance_to_pinedale_mall bus_speed time_between_stops num_stops = 30 :=
by
  -- Placeholder for the solution proof.
  sorry

end distance_to_pinedale_mall_l40_40497


namespace six_digit_numbers_with_zero_l40_40992

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40992


namespace six_digit_numbers_with_zero_l40_40956

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40956


namespace xiaonan_true_l40_40175

-- Variables representing each person's statements
variables (D X N B : Prop)

-- Conditions based on the problem statement
def xiaodong_statement := ¬X
def xiaoxi_statement := N
def xiaonan_statement := ¬D
def xiaobei_statement := ¬N

-- Only one statement can be true
def exactly_one_true : Prop := (D ↔ xiaodong_statement) ∧ (X ↔ xiaoxi_statement) ∧ (N ↔ xiaonan_statement) ∧ (B ↔ xiaobei_statement) ∧
  (
    (D ∧ ¬X ∧ ¬N ∧ ¬B) ∨ 
    (¬D ∧ X ∧ ¬N ∧ ¬B) ∨ 
    (¬D ∧ ¬X ∧ N ∧ ¬B) ∨ 
    (¬D ∧ ¬X ∧ ¬N ∧ B)
  )

-- The proof statement
theorem xiaonan_true (h : exactly_one_true D X N B) : N := sorry

end xiaonan_true_l40_40175


namespace coffee_break_l40_40590

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end coffee_break_l40_40590


namespace vector_problem_l40_40852

open Real

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  (v.1 ^ 2 + v.2 ^ 2) ^ (1 / 2)

variables (a b : ℝ × ℝ)
variables (h1 : a ≠ (0, 0)) (h2 : b ≠ (0, 0))
variables (h3 : a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0)
variables (h4 : 2 * magnitude a = magnitude b) (h5 : magnitude b =2)

theorem vector_problem : magnitude (2 * a.1 - b.1, 2 * a.2 - b.2) = 2 :=
sorry

end vector_problem_l40_40852


namespace area_of_rhombus_l40_40674

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 22) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 330 :=
by
  rw [h1, h2]
  norm_num

-- Here we state the theorem about the area of the rhombus given its diagonal lengths.

end area_of_rhombus_l40_40674


namespace zeros_in_decimal_representation_l40_40659

theorem zeros_in_decimal_representation :
  let d := (7 : ℚ) / 8000 in
  ∃ n : ℕ, d = 875 / (10 ^ 6) ∧ n = 3 :=
by
  let d := (7 : ℚ) / 8000
  use 3
  constructor
  . norm_num [d]
  . rfl

end zeros_in_decimal_representation_l40_40659


namespace smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40631

/-- 
  Problem statement: Prove that the smallest prime divisor of 3^19 + 11^13 is 2, 
  given the conditions:
   - 3^19 is odd
   - 11^13 is odd
   - The sum of two odd numbers is even
-/

theorem smallest_prime_divisor_of_3_pow_19_plus_11_pow_13 :
  ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ (3^19 + 11^13) := 
by
  sorry

end smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40631


namespace correct_calculation_l40_40662

-- Define the statements for each option
def option_A (a : ℕ) : Prop := (a^2)^3 = a^5
def option_B (a : ℕ) : Prop := a^3 + a^2 = a^6
def option_C (a : ℕ) : Prop := a^6 / a^3 = a^3
def option_D (a : ℕ) : Prop := a^3 * a^2 = a^6

-- Define the theorem stating that option C is the only correct one
theorem correct_calculation (a : ℕ) : ¬option_A a ∧ ¬option_B a ∧ option_C a ∧ ¬option_D a := by
  sorry

end correct_calculation_l40_40662


namespace triangle_perimeter_ge_two_ac_l40_40315

theorem triangle_perimeter_ge_two_ac (a : ℝ) (A B C D F E : ℝ) 
    (h_square: A = 0 ∧ B = a ∧ C = a ∧ D = 0) 
    (F_pos: F ≤ a) (E_pos: E ≤ a):
    let AC := a * Real.sqrt 2
    ∃ DEF : ℝ, DEF ≥ 2 * AC :=
by
  have AC := a * Real.sqrt 2
  use DEF
  sorry

end triangle_perimeter_ge_two_ac_l40_40315


namespace sum_third_fifth_terms_l40_40569

-- Define the sequence and the conditions given in the problem.
def a : ℕ → ℚ 
| 1 := 1
| (n+1) := (n+1)^2 / (n^2)

-- Main statement to prove the sum of the third and fifth terms.
theorem sum_third_fifth_terms : a 3 + a 5 = 61 / 16 := by
  sorry

end sum_third_fifth_terms_l40_40569


namespace students_in_class_l40_40080

theorem students_in_class (S : ℕ) (h1 : S / 3 + 2 * S / 5 + 12 = S) : S = 45 :=
sorry

end students_in_class_l40_40080


namespace tan_sin_equation_solutions_l40_40362

theorem tan_sin_equation_solutions :
  ∀ x ∈ set.Icc 0 real.pi,
  (∃! x ∈ set.Icc 0 real.pi, real.tan (2 * x) = real.sin x) → 
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ set.Icc 0 real.pi ∧ x₂ ∈ set.Icc 0 real.pi ∧ real.tan (2 * x₁) = real.sin x₁ ∧ real.tan (2 * x₂) = real.sin x₂ :=
sorry

end tan_sin_equation_solutions_l40_40362


namespace length_equality_l40_40352

-- Definitions corresponding to the given conditions
variables (V : Type*) [inner_product_space ℝ V]
variables {l1 l2 : line V} {O1 A1 O2 A2 : V}

-- Given conditions
variables (hl1 : is_line l1) (hl2 : is_line l2)
variables (hO1l1 : O1 ∈ l1) (hA1l1 : A1 ∈ l1)
variables (hO2l2 : O2 ∈ l2) (hA2l2 : A2 ∈ l2)
variables (h_perp : O1O2 ⟂ (l1, l2))
variables (h_equal_angles : ∠(A1A2, l1) = ∠(A1A2, l2))

-- Theorem to be proven
theorem length_equality : dist O1 A1 = dist O2 A2 := 
sorry

end length_equality_l40_40352


namespace triangle_CEF_is_isosceles_l40_40084

variables {α : Type*} [linear_ordered_field α] 

-- Definitions of the points and others conditions
variables (A B C D E F : α → α → Prop)
variable (AC : A C)
variable (ABC : isosceles A B C)
variable (ADC : isosceles A D C)
variable (half_plane : ¬same_side B D AC)
variable (angle_ADC_eq_3_angle_ACB : ∀ A B C D, angle A D C = 3 * angle A C B)
variable (AE_bisector : ∀ A B E, is_angle_bisector A B E)
variable (F_intersection : ∀ D E A C, line_intersect D E A C F)

-- Theorem to prove triangle CEF is isosceles
theorem triangle_CEF_is_isosceles : isosceles C E F :=
sorry

end triangle_CEF_is_isosceles_l40_40084


namespace six_digit_numbers_with_zero_l40_40936

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40936


namespace find_k_if_equal_roots_l40_40011

theorem find_k_if_equal_roots (a b k : ℚ) 
  (h1 : 2 * a + b = -4) 
  (h2 : 2 * a * b + a^2 = -60) 
  (h3 : -2 * a^2 * b = k)
  (h4 : a ≠ b)
  (h5 : k > 0) :
  k = 6400 / 27 :=
by {
  sorry
}

end find_k_if_equal_roots_l40_40011


namespace cubics_of_sum_and_product_l40_40369

theorem cubics_of_sum_and_product (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : 
  x^3 + y^3 = 640 :=
by
  sorry

end cubics_of_sum_and_product_l40_40369


namespace length_of_shop_proof_l40_40129

-- Given conditions
def monthly_rent : ℝ := 1440
def width : ℝ := 20
def annual_rent_per_sqft : ℝ := 48

-- Correct answer to be proved
def length_of_shop : ℝ := 18

-- The following statement is the proof problem in Lean 4
theorem length_of_shop_proof (h1 : monthly_rent = 1440) 
                            (h2 : width = 20) 
                            (h3 : annual_rent_per_sqft = 48) : 
  length_of_shop = 18 := 
  sorry

end length_of_shop_proof_l40_40129


namespace largest_integral_value_l40_40287

theorem largest_integral_value (x : ℤ) : (1 / 3 : ℚ) < x / 5 ∧ x / 5 < 5 / 8 → x = 3 :=
by
  sorry

end largest_integral_value_l40_40287


namespace six_digit_numbers_with_zero_l40_40894

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40894


namespace beijing_northeast_of_potala_palace_l40_40498

theorem beijing_northeast_of_potala_palace (B P : Type) [Located P (Direction.southwest B)] :
  Located B (Direction.northeast P) :=
sorry

end beijing_northeast_of_potala_palace_l40_40498


namespace trapezium_area_l40_40668

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 17) : 
  (1 / 2 * (a + b) * h) = 323 :=
by
  have ha' : a = 20 := ha
  have hb' : b = 18 := hb
  have hh' : h = 17 := hh
  rw [ha', hb', hh']
  sorry

end trapezium_area_l40_40668


namespace circle_area_polar_eq_l40_40537

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l40_40537


namespace estimate_3_plus_sqrt_10_l40_40275

theorem estimate_3_plus_sqrt_10 : 6 < 3 + Real.sqrt 10 ∧ 3 + Real.sqrt 10 < 7 :=
by
  have h1 : 3^2 = 9 := rfl
  have h2 : 4^2 = 16 := rfl
  have h3 : 9 < 10 ∧ 10 < 16 := by split; exact Nat.lt_succ_self 9; exact Nat.lt.base 16 10
  have h4 : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := sorry
  have h5 : 6 < 3 + Real.sqrt 10 ∧ 3 + Real.sqrt 10 < 7 := sorry
  exact h5
  sorry

end estimate_3_plus_sqrt_10_l40_40275


namespace largest_possible_real_solutions_l40_40761

theorem largest_possible_real_solutions (a b c : ℝ) :
  let P : Polynomial ℝ := Polynomial.C 1 * Polynomial.X ^ 6 + Polynomial.C a * Polynomial.X ^ 5 + Polynomial.C 60 * Polynomial.X ^ 4 - Polynomial.C 159 * Polynomial.X ^ 3 + Polynomial.C 240 * Polynomial.X ^ 2 + Polynomial.C b * Polynomial.X + Polynomial.C c
  in (∀ (x : ℝ), Polynomial.eval x P = 0 → ∃ (u v w z : ℝ), P.has_root u ∧ P.has_root v ∧ P.has_root w ∧ P.has_root z) → false :=
by
  intros
  sorry -- Proof goes here

end largest_possible_real_solutions_l40_40761


namespace area_transformed_region_l40_40410

-- Define the transformation matrix as a 2x2 matrix
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 1], ![4, 3]]

-- Define the original area of region R
def area_R : ℝ := 9

-- Define the determinant of the transformation matrix
def det_transformation_matrix : ℝ := by
  have m : Matrix (Fin 2) (Fin 2) ℝ := transformation_matrix
  exact Matrix.det m

-- The proof goal: the area of R' is equal to 45
theorem area_transformed_region : det_transformation_matrix * area_R = 45 := by
  have det_m : det_transformation_matrix = 5 := by
    -- Compute the determinant here (9 - 4 = 5)
    sorry -- This is a placeholder to show where the determinant computation would go
  rw det_m
  norm_num
  sorry -- This is a placeholder to indicate where further proof steps would go

end area_transformed_region_l40_40410


namespace participants_coffee_l40_40596

theorem participants_coffee (n : ℕ) (h_n : n = 14) (h_not_all : 0 < n - 2 * k) (h_not_all2 : n - 2 * k < n)
(h_neighbors : ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧ (remaining i → leaving j) ∧ (leaving i → remaining j)) :
  ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 6 ∧ (n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12) :=
by {
  sorry
}

end participants_coffee_l40_40596


namespace circle_area_l40_40544

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l40_40544


namespace six_digit_numbers_with_zero_l40_40892

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40892


namespace x_plus_y_value_l40_40380

def sum_integers_10_to_20 : ℕ := (20 - 10 + 1) * (10 + 20) / 2

def num_even_integers_10_to_20 : ℕ := 6

theorem x_plus_y_value :
  let x := sum_integers_10_to_20 in
  let y := num_even_integers_10_to_20 in
  x + y = 171 :=
by
  sorry

end x_plus_y_value_l40_40380


namespace smallest_prime_divisor_of_n_is_2_l40_40638

def a := 3^19
def b := 11^13
def n := a + b

theorem smallest_prime_divisor_of_n_is_2 : nat.prime_divisors n = {2} :=
by
  -- The proof is not needed as per instructions.
  -- We'll add a placeholder sorry to complete the statement.
  sorry

end smallest_prime_divisor_of_n_is_2_l40_40638


namespace store_sales_correct_l40_40019

def price_eraser_pencil : ℝ := 0.8
def price_regular_pencil : ℝ := 0.5
def price_short_pencil : ℝ := 0.4
def price_mechanical_pencil : ℝ := 1.2
def price_novelty_pencil : ℝ := 1.5

def quantity_eraser_pencil : ℕ := 200
def quantity_regular_pencil : ℕ := 40
def quantity_short_pencil : ℕ := 35
def quantity_mechanical_pencil : ℕ := 25
def quantity_novelty_pencil : ℕ := 15

def total_sales : ℝ :=
  (quantity_eraser_pencil * price_eraser_pencil) +
  (quantity_regular_pencil * price_regular_pencil) +
  (quantity_short_pencil * price_short_pencil) +
  (quantity_mechanical_pencil * price_mechanical_pencil) +
  (quantity_novelty_pencil * price_novelty_pencil)

theorem store_sales_correct : total_sales = 246.5 :=
by sorry

end store_sales_correct_l40_40019


namespace six_digit_numbers_with_zero_l40_40883

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40883


namespace six_digit_numbers_with_zero_l40_40891

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40891


namespace number_of_friends_l40_40238

theorem number_of_friends
    (members : ℕ)
    (needed_tees_per_member : ℕ)
    (packages_of_gen_te_te_amount : ℕ)
    (max_packages_gen_te : ℕ)
    (packages_of_af_te_te_amount : ℕ)
    (packages_of_af_te_needed : ℕ)
    (conditions : members = 4 
                ∧ needed_tees_per_member = 20
                ∧ packages_of_gen_te_te_amount = 12 
                ∧ max_packages_gen_te = 2 
                ∧ packages_of_af_te_te_amount = 2 
                ∧ packages_of_af_te_needed = 28) : (members - 1) = 3 :=
by
  rw [←conditions.1],
  rw [←conditions.2],
  rw [←conditions.3],
  rw [←conditions.4],
  rw [←conditions.5],
  rw [←conditions.6],
  sorry

end number_of_friends_l40_40238


namespace find_two_digit_number_l40_40712

theorem find_two_digit_number (c d : ℕ) (h1 : c < 10) (h2 : d < 10) 
  (h3 : 54 * (0.1 * c + 0.01 * d) = 54 * (((10 * c) + d):ℝ / 99) - 0.36) :
  10 * c + d = 65 :=
sorry

end find_two_digit_number_l40_40712


namespace sequence_sum_200_l40_40848

theorem sequence_sum_200 (a : ℕ → ℝ) (h : ∀ n, n ≥ 2 → a n + a (n - 1) = (-1:ℤ)^n * 3) :
  (∑ n in finset.range 200, a (n + 1)) = 300 :=
by {
  sorry,
}

end sequence_sum_200_l40_40848


namespace average_hit_targets_formula_average_hit_targets_ge_half_l40_40266

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

end average_hit_targets_formula_average_hit_targets_ge_half_l40_40266


namespace exists_y_for_sum_eq_prod_l40_40181

theorem exists_y_for_sum_eq_prod (n : ℝ) (hn : n ≠ 1) : 
  ∃ y : ℝ, n + y = n * y :=
begin
  use 1 + 1 / (n - 1),
  have h1 : n + (1 + 1 / (n - 1)) = n * (1 + 1 / (n - 1)), from by 
  { rw [add_comm n, mul_add],
    set x := 1 / (n - 1),
    field_simp [hn, h1], },
  exact h1,
end

end exists_y_for_sum_eq_prod_l40_40181


namespace coffee_break_l40_40594

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end coffee_break_l40_40594


namespace minimum_slope_tangent_line_l40_40552

theorem minimum_slope_tangent_line (b : ℝ) (hb : 0 < b) : (∃ f : ℝ → ℝ, (∀ x, f x = 2 * real.log x + x^2 - b * x + ((λ (a:ℝ), a) 0)) ∧ (∀ b, 0 < b → (∃ p: ℝ, p = 2/b + b)) ∧ (feq b (2 * real.log b + b^2 - b * b + ((λ (a:ℝ), a) 0)) (2/b + b) ∧ (2/b + b ≥ 2 * real.sqrt(2)))) := sorry


end minimum_slope_tangent_line_l40_40552


namespace volume_of_pyramid_l40_40211

theorem volume_of_pyramid (p α : ℝ) : 
  volume_of_pyramid p α = 
    (9 * p^3 * (Real.tan(α / 2))^3) / 
    (4 * Real.sqrt(3 * (Real.tan(α / 2))^2 - 1)) := sorry

end volume_of_pyramid_l40_40211


namespace ball_height_after_bounces_l40_40684

noncomputable def ball_bounce (n : ℕ) := 200 * (0.8 ^ n)

theorem ball_height_after_bounces :
  ∃ n : ℕ, n = 7 ∧ ball_bounce n < 50 :=
by
  sorry

end ball_height_after_bounces_l40_40684


namespace point_in_first_quadrant_l40_40029

def i : ℂ := complex.I -- Imaginary unit

def z : ℂ := (2 / (1 - i)) - 2 * (i ^ 3)

theorem point_in_first_quadrant : (z.re > 0) ∧ (z.im > 0) := by
  sorry

end point_in_first_quadrant_l40_40029


namespace julia_initial_money_l40_40402

theorem julia_initial_money 
  (M : ℚ) 
  (h1 : M / 2 - M / 8 = 15) : 
  M = 40 := 
sorry

end julia_initial_money_l40_40402


namespace relationship_between_M_and_N_l40_40818

theorem relationship_between_M_and_N
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 1 ≤ n ∧ n ≤ 2002 → 0 < a n)
  (M_def : ∑ i in finset.range 2001, a i * 
            (∑ j in finset.Ico 1 2002, a j) - a 2002 )
  (N_def : (∑ i in finset.range 2001, (a i) - a 2002) *
            ∑ j in finset.range 1 2002, a j) :
  M < N :=
by
  sorry

end relationship_between_M_and_N_l40_40818


namespace participants_coffee_l40_40586

theorem participants_coffee (n : ℕ) (h1 : n = 14) (h2 : 0 < (14 - 2 * (n / 2)) < 14) : 
  ∃ (k : ℕ), k ∈ {6, 8, 10, 12} ∧ k = (14 - 2 * (n / 2)) :=
by sorry

end participants_coffee_l40_40586


namespace range_of_w_l40_40338

noncomputable def f (w x : ℝ) : ℝ := Real.sin (w * x) - Real.sqrt 3 * Real.cos (w * x)

theorem range_of_w (w : ℝ) (h_w : 0 < w) :
  (∀ f_zeros : Finset ℝ, ∀ x ∈ f_zeros, (0 < x ∧ x < Real.pi) → f w x = 0 → f_zeros.card = 2) ↔
  (4 / 3 < w ∧ w ≤ 7 / 3) :=
by sorry

end range_of_w_l40_40338


namespace coffee_break_l40_40593

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end coffee_break_l40_40593


namespace zeros_between_decimal_point_and_first_nonzero_digit_l40_40654

theorem zeros_between_decimal_point_and_first_nonzero_digit :
  let n := (7 : ℚ) / 8000 in
  decimal_zeros_between_decimal_point_and_first_nonzero_digit n = 3 :=
by
  sorry

end zeros_between_decimal_point_and_first_nonzero_digit_l40_40654


namespace six_digit_numbers_with_zero_count_l40_40906

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40906


namespace six_digit_numbers_with_zero_l40_40885

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40885


namespace six_digit_numbers_with_zero_l40_40927

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40927


namespace f_positive_for_specific_a_l40_40055

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x * Real.log x

theorem f_positive_for_specific_a (x : ℝ) (h : x > 0) :
  f x (Real.exp 3 / 4) > 0 := sorry

end f_positive_for_specific_a_l40_40055


namespace original_faculty_is_287_l40_40457

noncomputable def original_faculty (F : ℝ) : Prop :=
  (F * 0.85 * 0.80 = 195)

theorem original_faculty_is_287 : ∃ F : ℝ, original_faculty F ∧ F = 287 := 
by 
  use 287
  sorry

end original_faculty_is_287_l40_40457


namespace rationalized_sum_l40_40098

theorem rationalized_sum : 
  ∃ A B C D : ℤ, 
    (D > 0) ∧ 
    (∀ p : ℕ, prime p → p * p ∣ B → false) ∧ 
    (Int.gcd A (Int.gcd C D) = 1) ∧ 
    (A * (Int.ofNat B).sqrt + C) / D = 7 * (3 - (Int.ofNat B).sqrt) / (3 + (Int.ofNat B).sqrt) ∧ 
    A + B + C + D = 23 := sorry

end rationalized_sum_l40_40098


namespace kanul_initial_amount_l40_40043

theorem kanul_initial_amount (X Y : ℝ) (loan : ℝ) (R : ℝ) 
  (h1 : loan = 2000)
  (h2 : R = 0.20)
  (h3 : Y = 0.15 * X + loan)
  (h4 : loan = R * Y) : 
  X = 53333.33 :=
by 
  -- The proof would come here, but is not necessary for this example
sorry

end kanul_initial_amount_l40_40043


namespace six_digit_numbers_with_zero_l40_40988

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40988


namespace six_digit_numbers_with_zero_l40_40928

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40928


namespace trig_identity_simplify_vector_magnitude_add_l40_40782

-- Problem 1
theorem trig_identity_simplify :
  (√(1 - 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180))) / 
  (Real.cos (15 * Real.pi / 180) - Real.sqrt (1 - Real.cos (165 * Real.pi / 180)^2)) = 1 := 
sorry

-- Problem 2
theorem vector_magnitude_add {a b : ℝ} :
  |a| = 4 → |b| = 2 → ∠(a, b) = 2 * Real.pi / 3 → 
  |a + b| = 2 * √3 := 
sorry

end trig_identity_simplify_vector_magnitude_add_l40_40782


namespace unique_solution_l40_40063

theorem unique_solution (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (f : ℝ → ℝ) (h_f : ∀ x, 0 ≤ x → 0 ≤ f x) (h_eq : ∀ x, 0 ≤ x → f(f(x)) + a * f(x) = b * (a + b) * x) : 
  (∀ x, 0 ≤ x → f(x) = b * x) :=
sorry

end unique_solution_l40_40063


namespace product_of_roots_of_unity_l40_40757

theorem product_of_roots_of_unity : 
  ∏ k in Finset.range 1 16, ∏ j in Finset.range 1 13, (Complex.exp(2 * Real.pi * j * Complex.I / 13) - Complex.exp(2 * Real.pi * k * Complex.I / 17)) = 0 :=
sorry

end product_of_roots_of_unity_l40_40757


namespace real_part_zero_pure_imaginary_condition_l40_40008

theorem real_part_zero (a : ℂ) (ha : (a - 4) = 0) : a = 4 :=
by
  sorry

theorem pure_imaginary_condition (a : ℂ) (h: (a - 2 * complex.I) / (1 + 2 * complex.I)).re = 0 : a = 4 :=
by
  sorry

end real_part_zero_pure_imaginary_condition_l40_40008


namespace coin_flip_probability_l40_40611

noncomputable def num_favorable_outcomes : ℕ := 
  Nat.choose 15 12 + Nat.choose 15 13 + Nat.choose 15 14 + Nat.choose 15 15

noncomputable def total_outcomes : ℕ := 2 ^ 15

noncomputable def at_least_12_heads_probability : ℚ := 
  num_favorable_outcomes / total_outcomes

theorem coin_flip_probability :
  at_least_12_heads_probability = 9 / 512 := 
by
  simp [at_least_12_heads_probability, num_favorable_outcomes, total_outcomes]
  sorry

end coin_flip_probability_l40_40611


namespace no_solution_system_l40_40110

theorem no_solution_system :
  ¬ ∃ (x y z : ℝ), (3 * x - 4 * y + z = 10) ∧ (6 * x - 8 * y + 2 * z = 5) ∧ (2 * x - y - z = 4) :=
by {
  sorry
}

end no_solution_system_l40_40110


namespace avg_distance_is_600_l40_40437

-- Assuming Mickey runs half as many times as Johnny
def num_laps_johnny := 4
def num_laps_mickey := num_laps_johnny / 2
def lap_distance := 200

-- Calculating distances
def distance_johnny := num_laps_johnny * lap_distance
def distance_mickey := num_laps_mickey * lap_distance

-- Total distance run by both Mickey and Johnny
def total_distance := distance_johnny + distance_mickey

-- Average distance run by Johnny and Mickey
def avg_distance := total_distance / 2

-- Prove that the average distance run by Johnny and Mickey is 600 meters
theorem avg_distance_is_600 : avg_distance = 600 :=
by
  sorry

end avg_distance_is_600_l40_40437


namespace smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40630

/-- 
  Problem statement: Prove that the smallest prime divisor of 3^19 + 11^13 is 2, 
  given the conditions:
   - 3^19 is odd
   - 11^13 is odd
   - The sum of two odd numbers is even
-/

theorem smallest_prime_divisor_of_3_pow_19_plus_11_pow_13 :
  ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ (3^19 + 11^13) := 
by
  sorry

end smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40630


namespace find_alpha_l40_40054

-- Given conditions
variables (m n p : ℝ^3) (α : ℝ)
variables (hm : ‖m‖ = 2)
variables (hn : ‖n‖ = 3)
variables (hp : ‖p‖ = 4)
variables (hαmn : angle m n = α)
variables (hαpmxn : angle p (m × n) = α)
variables (h_scalar_triple : n • (p × m) = 3 / 2)

-- Statement to be proved
theorem find_alpha : α = 1 / 2 * arcsin (1 / 16) := sorry

end find_alpha_l40_40054


namespace jordan_time_for_10_miles_l40_40401

-- Definitions of the conditions
def steve_time_for_6_miles : ℕ := 36 -- Steve takes 36 minutes to run 6 miles
def jordan_time_for_4_miles := steve_time_for_6_miles / 3 -- Jordan's time for 4 miles is a third of Steve's time
def jordan_pace_per_mile := jordan_time_for_4_miles / 4 -- Jordan's time per mile

-- The theorem to be proven
theorem jordan_time_for_10_miles : Jordan_time_for_10_miles = 30 :=
by
  -- The necessary definitions
  let jordan_time_for_4_miles : ℕ := steve_time_for_6_miles / 3
  let jordan_pace_per_mile : ℕ := jordan_time_for_4_miles / 4
  let jordan_time_for_10_miles : ℕ := jordan_pace_per_mile * 10
  -- Assert the theorem with the proven result
  exact jordan_time_for_10_miles = 30

end jordan_time_for_10_miles_l40_40401


namespace hundredth_fraction_seq_l40_40449

theorem hundredth_fraction_seq :
  let a_n := λ n : ℕ, 1 + (n - 1) * 2
  let b_n := λ n : ℕ, 2 + (n - 1) * 3
  a_n 100 = 199 ∧ b_n 100 = 299 :=
by
  sorry

end hundredth_fraction_seq_l40_40449


namespace digits_difference_l40_40493

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end digits_difference_l40_40493


namespace knitting_time_is_correct_l40_40431

-- Definitions of the conditions
def time_per_hat : ℕ := 2
def time_per_scarf : ℕ := 3
def time_per_mitten : ℕ := 1
def time_per_sock : ℕ := 3 / 2 -- fractional time in hours
def time_per_sweater : ℕ := 6
def number_of_grandchildren : ℕ := 3

-- Compute total time for one complete outfit
def time_per_outfit : ℕ := time_per_hat + time_per_scarf + (time_per_mitten * 2) + (time_per_sock * 2) + time_per_sweater

-- Compute total time for all outfits
def total_knitting_time : ℕ := number_of_grandchildren * time_per_outfit

-- Prove that total knitting time is 48 hours
theorem knitting_time_is_correct : total_knitting_time = 48 := by
  unfold total_knitting_time time_per_outfit
  norm_num
  sorry

end knitting_time_is_correct_l40_40431


namespace find_other_number_l40_40496

theorem find_other_number (LCM HCF num1 num2 : ℕ) 
  (h1 : LCM = 2310) 
  (h2 : HCF = 30) 
  (h3 : num1 = 330) 
  (h4 : LCM * HCF = num1 * num2) : 
  num2 = 210 := by 
  sorry

end find_other_number_l40_40496


namespace find_f_neg_2_l40_40329

def f (x : ℝ) : ℝ :=
if x > 0 then 2^x else if x < 0 then -2^(-x) else 0

theorem find_f_neg_2 (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_pos : ∀ x > 0, f x = 2^x) : f (-2) = -4 := by
sorry

end find_f_neg_2_l40_40329


namespace parabola_hyperbola_distance_l40_40343

theorem parabola_hyperbola_distance :
  ∀ (a b : ℝ), a > 0 →
  (∃ l, l.slope = 1 ∧ l.contains (a, 0) ∧ segment_length_on_parabola l (parabola (8 * a)) = 16) →
  hyperbola_focus_on_directrix (hyperbola (1 / (a ^ 2)) (1 / (b ^ 2))) (parabola_directrix (8 * a)) →
  distance_from_point_to_asymptote (0, -2) (asymptote_of_hyperbola (1 / (a ^ 2)) (1 / (b ^ 2))) = 1 :=
by
  intros a b ha h_intersect h_focus
  sorry

end parabola_hyperbola_distance_l40_40343


namespace six_digit_numbers_with_zero_l40_40998

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l40_40998


namespace equation_of_line_l40_40326

-- Definition of the function and its derivative
def f (x : ℝ) := x * Real.log x

def f' (x : ℝ) := 1 + Real.log x

-- The point (0, -1) that the line passes through
def point := (0 : ℝ, -1 : ℝ)

-- The tangent point
def tangent_point (x₀ : ℝ) := (x₀, x₀ * Real.log x₀)

-- Proving that the equation of line l is x - y - 1 = 0
theorem equation_of_line (x₀ : ℝ) (hₜ : tangent_point x₀ = point) : x - y - 1 = 0 := sorry

end equation_of_line_l40_40326


namespace log_base_solution_l40_40785

theorem log_base_solution (x : ℝ) (h1: log 2 32 = 5) (h2: log x 125 = 5) : x = 5 := 
sorry

end log_base_solution_l40_40785


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40942

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40942


namespace exists_local_min_exp_greater_than_m_log_x_plus_2_l40_40838

-- Define the function for m > 0
def f (x : ℝ) (m : ℝ) : ℝ := m * exp x - log x - 1

-- Define its derivative
def f_prime (x : ℝ) (m : ℝ) : ℝ := m * exp x - 1 / x

-- Condition I: Existence of local minimum when m = 1
theorem exists_local_min (x : ℝ) : ∃ x₀, f_prime x₀ 1 = 0 ∧ (∀ h, 0 < h → f_prime x₀ h > 0 ∨ f_prime x₀ (-h) > 0) := sorry

-- Define the function for e^x - m * (ln x + 2)
def g (x : ℝ) (m : ℝ) : ℝ := exp x - m * (log x + 2)

-- Condition II: Proving the inequality for 0 < m ≤ 1
theorem exp_greater_than_m_log_x_plus_2 (x : ℝ) (m : ℝ) (hm : 0 < m ∧ m ≤ 1) : exp x > m * (log x + 2) := sorry

end exists_local_min_exp_greater_than_m_log_x_plus_2_l40_40838


namespace six_digit_numbers_with_zero_l40_40987

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40987


namespace prime_sum_1_to_50_l40_40067

theorem prime_sum_1_to_50 :
  let primes := { p : ℕ | p.prime ∧ 1 ≤ p ∧ p ≤ 50 }
  in (2 ∈ primes) ∧ (47 ∈ primes) ∧ (∀ p ∈ primes, p = 2 ∨ p = 47)
   → 2 + 47 = 49 := 
by
  intros _ h_smallest_prime h_largest_prime _
  exact eq.refl 49

end prime_sum_1_to_50_l40_40067


namespace increase_by_150_percent_l40_40644

theorem increase_by_150_percent (y x : ℝ) (h1 : y = 1.5 * 40) (h2 : x = 40 + y) : x = 100 := 
by 
sory

end increase_by_150_percent_l40_40644


namespace find_A_d_minus_B_d_l40_40487

-- Definitions for the proof problem
variables {d A B : ℕ}
variables {ad bd : ℤ} -- Representing A_d and B_d in ℤ for arithmetic operations

-- Conditions
constants (base_check : d > 7)
constants (digit_check : A < d ∧ B < d)
constants (encoded_check : d^1 * A + d^0 * B + d^2 * A + d^1 * A = 1 * d^2 + 7 * d^1 + 2 * d^0)

-- The theorem to prove
theorem find_A_d_minus_B_d : A - B = 5 :=
good sorry

end find_A_d_minus_B_d_l40_40487


namespace maximum_B_at_125_l40_40279

noncomputable def binomial_coeff (n k : ℕ) : ℚ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

def B (k : ℕ) : ℚ := binomial_coeff 500 k * (0.3 ^ k)

theorem maximum_B_at_125 :
  ∃ k, k = 125 ∧ ∀ k', B k' ≤ B k := sorry

end maximum_B_at_125_l40_40279


namespace six_digit_numbers_with_zero_l40_40990

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40990


namespace max_marked_points_l40_40764

theorem max_marked_points (points : Set Point) (A B : Point) :
  let dist := (∃ (A B : Point) (distance_AB = max distance, {distance_AB}))
  ∃ (rays_from_A : Set Ray) (acute_angles : {0 < angle < 90}), {acute angles}
  ∃ (rays_sum <= 178), {maximum sum}  . 
  ∃ points.supportingline (no three points collinear ∧ convex_hull), 
  let N := (number of points : ℕ) 
  N ≤ 180 := by
   sorry

end max_marked_points_l40_40764


namespace zeros_between_decimal_point_and_first_nonzero_digit_l40_40655

theorem zeros_between_decimal_point_and_first_nonzero_digit :
  let n := (7 : ℚ) / 8000 in
  decimal_zeros_between_decimal_point_and_first_nonzero_digit n = 3 :=
by
  sorry

end zeros_between_decimal_point_and_first_nonzero_digit_l40_40655


namespace units_digit_odd_product_l40_40162

theorem units_digit_odd_product (l : List ℕ) (h_odds : ∀ n ∈ l, n % 2 = 1) :
  (∀ x ∈ l, x % 10 = 5) ↔ (5 ∈ l) := by
  sorry

end units_digit_odd_product_l40_40162


namespace chord_length_hyperbola_asymptote_circle_intersection_l40_40830

theorem chord_length_hyperbola_asymptote_circle_intersection {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (a : ℝ) * (b : ℝ) = 2 * (a^2)) (h₄ : dist (2, 3) (line.mk (2, 1) 2) = 1 / sqrt 5) : 
  let A : point ℝ := ...
  let B : point ℝ := ...
  dist A B = 4 * sqrt 5 / 5 :=
sorry

end chord_length_hyperbola_asymptote_circle_intersection_l40_40830


namespace area_of_triangle_XYZ_l40_40195

theorem area_of_triangle_XYZ (r1 r2 : ℝ) (r1_pos : r1 = 3) (r2_pos : r2 = 5) 
  (tangent_condition : tangent YZ r1 r2)
  (congruent_sides : XY = XZ) : 
  area XYZ = 16 * sqrt 55 := 
sorry

end area_of_triangle_XYZ_l40_40195


namespace lisa_total_miles_flown_l40_40071

variable (distance_per_trip : ℝ := 256.0)
variable (number_of_trips : ℝ := 32.0)

theorem lisa_total_miles_flown : distance_per_trip * number_of_trips = 8192.0 := by
  sorry

end lisa_total_miles_flown_l40_40071


namespace six_digit_numbers_with_zero_l40_40873

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40873


namespace bisection_solvable_l40_40168

open Real

theorem bisection_solvable :
  (∃ a b, a < b ∧ (ln a + a < 0) ∧ (ln b + b > 0) ∧ (∀ x, a < x ∧ x < b → continuous_at (λ x, ln x + x) x)) ∧
  (∃ c d, c < d ∧ (exp c - 3 * c > 0) ∧ (exp d - 3 * d < 0) ∧ (∀ x, c < x ∧ x < d → continuous_at (λ x, exp x - 3 * x) x)) ∧
  (∃ e f, e < f ∧ (e^3 - 3 * e + 1 > 0) ∧ (f^3 - 3 * f + 1 < 0) ∧ (∀ x, e < x ∧ x < f → continuous_at (λ x, x^3 - 3 * x + 1) x)) ∧
  ¬(∃ g h, g < h ∧ (∀ x, g ≤ x ∧ x ≤ h → 4 * x^2 - 4 * sqrt 5 * x + 5 ≥ 0) ∧ (∀ x, g < x ∧ x < h → continuous_at (λ x, 4 * x^2 - 4 * sqrt 5 * x + 5) x)) :=
begin
  sorry
end

end bisection_solvable_l40_40168


namespace hyperbola_eccentricity_l40_40427

theorem hyperbola_eccentricity 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (P : ℝ × ℝ) (h_point : P = (2 * Real.sqrt 2, -1))
  (h_hyperbola : (P.1^2) / a^2 - (P.2^2) / b^2 = 1)
  (A1 A2 : ℝ × ℝ) (h_vertices : A1 = (-a, 0) ∧ A2 = (a, 0))
  (h_distance : dist A1 A2 = 4 * b)
  (h_slope_product : (P.2 / (P.1 + a)) * (P.2 / (P.1 - a)) = 2) :
  let e := Real.sqrt (1 + (b^2) / (a^2)) in
  e = Real.sqrt 3 := 
sorry

end hyperbola_eccentricity_l40_40427


namespace simplify_fraction_l40_40470

theorem simplify_fraction (x : ℝ) : (3*x + 2) / 4 + (x - 4) / 3 = (13*x - 10) / 12 := sorry

end simplify_fraction_l40_40470


namespace expected_hit_targets_expected_hit_targets_not_less_than_half_l40_40262

-- Part (a): The expected number of hit targets
theorem expected_hit_targets (n : ℕ) (h : n ≠ 0) :
  E (number_of_hit_targets n) = n * (1 - (1 - (1 / n)) ^ n) :=
sorry

-- Part (b): The expected number of hit targets cannot be less than n / 2
theorem expected_hit_targets_not_less_than_half (n : ℕ) (h : n ≠ 0) :
  n * (1 - (1 - (1 / n)) ^ n) ≥ n / 2 :=
sorry

end expected_hit_targets_expected_hit_targets_not_less_than_half_l40_40262


namespace six_digit_numbers_with_zero_l40_40869

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40869


namespace find_magnitude_l40_40113

open Complex

theorem find_magnitude 
  (w : ℂ)
  (h : w^2 = -75 + 100 * I) :
  abs w = 5 * Real.sqrt 3 :=
sorry

end find_magnitude_l40_40113


namespace tangent_intersection_product_l40_40120

theorem tangent_intersection_product (R r : ℝ) (A B C : ℝ) :
  (AC * CB = R * r) :=
sorry

end tangent_intersection_product_l40_40120


namespace area_five_times_l40_40719

-- Definitions for points and areas in a plane
variables (A B C D A' B' C' D' : Type)

-- Assuming the midpoints conditions for the new quadrilateral
variables (ABCD_convex: ConvexQuadrilateral A B C D)
          (midpoint_A': Midpoint A D A')
          (midpoint_B': Midpoint B A B')
          (midpoint_C': Midpoint C B C')
          (midpoint_D': Midpoint D C D')

-- Lean statement to show the area of A'B'C'D' is five times the area of ABCD
theorem area_five_times (h: ∀ {p1 p2 p3 p4: Type}, Area (Quadrilateral p1 p2 p3 p4)):
  Area (A' B' C' D') = 5 * (Area (A B C D)) := 
sorry

end area_five_times_l40_40719


namespace smallest_integer_with_eight_factors_l40_40160

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, ∃ p q : ℕ, ∃ a b : ℕ,
    (prime p) ∧ (prime q) ∧ (p ≠ q) ∧ (n = p^a * q^b) ∧ ((a + 1) * (b + 1) = 8) ∧ (n = 24) :=
by 
  sorry

end smallest_integer_with_eight_factors_l40_40160


namespace probability_of_two_changing_yao_l40_40024

theorem probability_of_two_changing_yao :
  let p : ℚ := 1/4 in
  let C (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) in
  let probability : ℚ := C 6 2 * (p ^ 2) * ((1 - p) ^ 4) in
  probability = 1215 / 4096 :=
by
  sorry

end probability_of_two_changing_yao_l40_40024


namespace total_cats_l40_40701

def initial_siamese_cats : Float := 13.0
def initial_house_cats : Float := 5.0
def added_cats : Float := 10.0

theorem total_cats : initial_siamese_cats + initial_house_cats + added_cats = 28.0 := by
  sorry

end total_cats_l40_40701


namespace circle_center_radius_l40_40835

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 4 * x + 2 * y - 4 = 0 ↔ (x - 2)^2 + (y + 1)^2 = 3 :=
by
  sorry

end circle_center_radius_l40_40835


namespace find_y_l40_40034

theorem find_y (x y : ℕ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 119) : y = 1 :=
sorry

end find_y_l40_40034


namespace spiral_stripe_length_l40_40193

theorem spiral_stripe_length (circumference height : ℝ) (h₁ : circumference = 12) (h₂ : height = 5) :
  let diagonal := Real.sqrt (circumference^2 + height^2)
  in diagonal = 13 :=
by
  -- The conditions
  rw [h₁, h₂]
  -- Then evaluate the expression
  simp_rhs
  -- The proof follows but is omitted
  sorry

end spiral_stripe_length_l40_40193


namespace area_of_circle_l40_40527

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l40_40527


namespace original_volume_of_ice_l40_40666

variable (V : ℝ) 

theorem original_volume_of_ice (h1 : V * (1 / 4) * (1 / 4) = 0.25) : V = 4 :=
  sorry

end original_volume_of_ice_l40_40666


namespace six_digit_numbers_with_zero_l40_40923

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40923


namespace six_digit_numbers_with_zero_l40_40991

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40991


namespace solve_equation_l40_40254

theorem solve_equation (x : ℝ) : 
  (x = 17 / 9 ∨ x = 44 / 9) ↔ (sqrt (9 * x - 8) + 18 / sqrt (9 * x - 8) = 8) :=
by 
  sorry

end solve_equation_l40_40254


namespace six_digit_numbers_with_zero_l40_40868

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40868


namespace shares_difference_l40_40725

theorem shares_difference (x : ℝ) (hp : ℝ) (hq : ℝ) (hr : ℝ)
  (hx : hp = 3 * x) (hqx : hq = 7 * x) (hrx : hr = 12 * x) 
  (hqr_diff : hr - hq = 3500) : (hq - hp = 2800) :=
by
  -- The proof would be done here, but the problem statement requires only the theorem statement
  sorry

end shares_difference_l40_40725


namespace six_digit_numbers_with_zero_l40_40889

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40889


namespace no_perfect_squares_in_sequence_l40_40299

def tau (a : ℕ) : ℕ := sorry -- Define tau function here

def a_seq (k : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then k else tau (a_seq k (n-1))

theorem no_perfect_squares_in_sequence (k : ℕ) (hk : Prime k) :
  ∀ n : ℕ, ∃ m : ℕ, a_seq k n = m * m → False :=
sorry

end no_perfect_squares_in_sequence_l40_40299


namespace sqrt_expression_eval_l40_40740

theorem sqrt_expression_eval :
  (sqrt 27 - 6 * sqrt (1/3) + sqrt ((-2)^2)) = sqrt 3 + 2 :=
by
  sorry

end sqrt_expression_eval_l40_40740


namespace probability_of_red_or_blue_l40_40675

theorem probability_of_red_or_blue (total_marbles : ℕ)
  (p_white : ℚ) (p_green : ℚ) (p_red_or_blue : ℚ) :
  total_marbles = 84 →
  p_white = 1 / 4 →
  p_green = 1 / 7 →
  p_red_or_blue = 1 - (p_white + p_green) →
  p_red_or_blue = 17 / 28 :=
by
  intros h_total h_pwhite h_pgreen h_pred_or_blue
  rw [h_total, h_pwhite, h_pgreen, h_pred_or_blue]
  norm_num
  sorry

end probability_of_red_or_blue_l40_40675


namespace abs_neg_two_l40_40507

def abs (x : ℤ) : ℤ :=
  if x ≥ 0 then x else -x

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end abs_neg_two_l40_40507


namespace chadSavingsIsCorrect_l40_40744

noncomputable def chadSavingsAfterTaxAndConversion : ℝ :=
  let euroToUsd := 1.20
  let poundToUsd := 1.40
  let euroIncome := 600 * euroToUsd
  let poundIncome := 250 * poundToUsd
  let dollarIncome := 150 + 150
  let totalIncome := euroIncome + poundIncome + dollarIncome
  let taxRate := 0.10
  let taxedIncome := totalIncome * (1 - taxRate)
  let savingsRate := if taxedIncome ≤ 1000 then 0.20
                     else if taxedIncome ≤ 2000 then 0.30
                     else if taxedIncome ≤ 3000 then 0.40
                     else 0.50
  let savings := taxedIncome * savingsRate
  savings

theorem chadSavingsIsCorrect : chadSavingsAfterTaxAndConversion = 369.90 := by
  sorry

end chadSavingsIsCorrect_l40_40744


namespace area_circle_l40_40536

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l40_40536


namespace sum_of_n_such_that_perfect_square_conditions_l40_40291

theorem sum_of_n_such_that_perfect_square_conditions :
  let s := {n : ℤ | ∃ (x : ℤ), n^2 - 17 * n + 72 = x^2 ∧ ∃ (m : ℤ), n = m^2} in
  (sum s) = 17 :=
by
  sorry

end sum_of_n_such_that_perfect_square_conditions_l40_40291


namespace count_factor_pairs_144_l40_40049

def is_valid_factor_pair (n : ℕ) (f1 f2 : ℕ) : Prop :=
  f1 * f2 = n ∧ f1 > 1 ∧ f2 > 1

def count_valid_factor_pairs (n : ℕ) : ℕ :=
  (Finset.univ.filter (λ f1f2 : ℕ × ℕ, f1f2.1 * f1f2.2 = n ∧ f1f2.1 > 1 ∧ f1f2.2 > 1)).card

theorem count_factor_pairs_144 : count_valid_factor_pairs 144 = 6 :=
by
  -- Proof should be filled here.
  sorry

end count_factor_pairs_144_l40_40049


namespace determine_truth_tellers_min_questions_to_determine_truth_tellers_l40_40407

variables (n k : ℕ)
variables (h_n_pos : 0 < n) (h_k_pos : 0 < k) (h_k_le_n : k ≤ n)

theorem determine_truth_tellers (h : k % 2 = 0) : 
  ∃ m : ℕ, m = n :=
  sorry

theorem min_questions_to_determine_truth_tellers :
  ∃ m : ℕ, m = n :=
  sorry

end determine_truth_tellers_min_questions_to_determine_truth_tellers_l40_40407


namespace unacceptable_mass_450_l40_40691

def is_acceptable_mass (mass : ℕ) : Prop :=
  480 ≤ mass ∧ mass ≤ 520

theorem unacceptable_mass_450 : ¬ is_acceptable_mass 450 :=
by
  unfold is_acceptable_mass
  simp
  exact sorry

end unacceptable_mass_450_l40_40691


namespace cubics_of_sum_and_product_l40_40368

theorem cubics_of_sum_and_product (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : 
  x^3 + y^3 = 640 :=
by
  sorry

end cubics_of_sum_and_product_l40_40368


namespace anna_total_value_l40_40234

theorem anna_total_value (total_bills : ℕ) (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ)
  (h1 : total_bills = 12) (h2 : five_dollar_bills = 4) (h3 : ten_dollar_bills = total_bills - five_dollar_bills) :
  5 * five_dollar_bills + 10 * ten_dollar_bills = 100 := by
  sorry

end anna_total_value_l40_40234


namespace six_digit_numbers_with_zero_l40_40993

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l40_40993


namespace AIME_area_l40_40747

-- Declare the parameters for the rhombus and the circle
variables (H M1 M2 T A I M E : Type) [InnerProductSpace ℝ H]
variables (area_rhombus : ℝ) (area_EMT : ℝ) [hrhombus : affineSpace H] [hcirc : metricSpace H]

-- Conditions
axiom rhombus_area : area_rhombus = 1440
axiom EMT_area : area_EMT = 405
axiom inscribed_circle : ∃ (ω : metricSpace H), tangent_to ω (line segment H M1 A) ∧ tangent_to ω (line segment M1 M2 I) ∧ tangent_to ω (line segment M2 T M) ∧ tangent_to ω (line segment T H E)

-- Statement to prove
theorem AIME_area : 
  rhombus_area = 1440 → 
  EMT_area = 405 →
  (∃ (ω : metricSpace H), (tangent_to ω (line segment H M1 A)) ∧ (tangent_to ω (line segment M1 M2 I)) ∧ (tangent_to ω (line segment M2 T M)) ∧ (tangent_to ω (line segment T H E))) →
  area (AIME) = 540 := 
sorry

end AIME_area_l40_40747


namespace six_digit_numbers_with_zero_l40_40924

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40924


namespace coffee_break_l40_40573

variable {n : ℕ}
variables {participants : Finset ℕ} (hparticipants : participants.card = 14)
variables (left : Finset ℕ)
variables (stayed : Finset ℕ) (hstayed : stayed.card = 14 - 2 * left.card)

-- The overall proof problem statement:
theorem coffee_break (h : ∀ x ∈ stayed, ∃! y, (y ∈ left ∧ adjacent x y participants)) :
  left.card = 6 ∨ left.card = 8 ∨ left.card = 10 ∨ left.card = 12 := 
sorry

end coffee_break_l40_40573


namespace find_f_3_l40_40804

def f : ℝ → ℝ := sorry

theorem find_f_3 (hf : ∀ x : ℝ, f (3 * x) = 2^x * Real.log2 x) :
  f 3 = 0 :=
sorry

end find_f_3_l40_40804


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40909

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40909


namespace smallest_integer_with_eight_factors_l40_40158

theorem smallest_integer_with_eight_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (card (factors n) == 8) ∧ (∀ m : ℕ, (0 < m) ∧ (card (factors m) == 8) → n ≤ m) :=
sorry

end smallest_integer_with_eight_factors_l40_40158


namespace tangent_line_at_point_one_unique_zero_of_g_range_of_m_l40_40840

-- Part (I)
theorem tangent_line_at_point_one (a : ℝ) (h_a : a = -1) (f : ℝ → ℝ)
  (hf : ∀ x, f x = (x^2 - 2 * x) * Real.log x + a * x^2 + 2) (g : ℝ → ℝ) 
  (hg : ∀ x, g x = f x - x - 2) :
  ∃ l, ∃ c, (∀ x y, y = f x → y - f 1 = l * (x - 1) → l * x + y + c = 0) :=
sorry

-- Part (II)
theorem unique_zero_of_g (a : ℝ) (f : ℝ → ℝ)
  (hf : ∀ x, f x = (x^2 - 2 * x) * Real.log x + a * x^2 + 2) (g : ℝ → ℝ) 
  (hg : ∀ x, g x = f x - x - 2) (h : ∀ x, g x = 0 → x = 1) :
  a = 1 :=
sorry

-- Part (III)
theorem range_of_m (a : ℝ) (h_a : a = 1) (f : ℝ → ℝ)
  (hf : ∀ x, f x = (x^2 - 2 * x) * Real.log x + a * x^2 + 2) (g : ℝ → ℝ) 
  (hg : ∀ x, g x = f x - x - 2) (m : ℝ) (h : ∀ x, (e^(-2) < x ∧ x < e) → g x ≤ m) :
  m ≥ 2 * e^2 - 3 * e :=
sorry

end tangent_line_at_point_one_unique_zero_of_g_range_of_m_l40_40840


namespace triangle_property_l40_40815

-- Definitions of the triangle with perimeter, circumradius, and inradius
variables {ℝ : Type*} [lineara_Ord.le ℝ] -- We work over the real numbers ℝ

def triangle (A B C : ℝ) := (A + B + C = l) ∧ (is_circumradius R) ∧ (is_inradius r)

-- The statements about the relationships
def statement_A := ∀ (A B C : ℝ), triangle A B C → l > R + r
def statement_B := ∀ (A B C : ℝ), triangle A B C → l ≤ R + r
def statement_C := ∀ (A B C : ℝ), triangle A B C → (1/6) < R + r ∧ R + r < 6*l

-- Final statement to prove
theorem triangle_property {A B C : ℝ} (h : triangle A B C) :
  ¬ statement_A ∧ ¬ statement_B ∧ ¬ statement_C := 
sorry

end triangle_property_l40_40815


namespace smallest_prime_divisor_of_sum_l40_40642

theorem smallest_prime_divisor_of_sum (h1 : Nat.odd (Nat.pow 3 19))
                                      (h2 : Nat.odd (Nat.pow 11 13))
                                      (h3 : ∀ a b : Nat, Nat.odd a → Nat.odd b → Nat.even (a + b)) :
  Nat.minFac (Nat.pow 3 19 + Nat.pow 11 13) = 2 :=
by
  -- placeholder proof
  sorry

end smallest_prime_divisor_of_sum_l40_40642


namespace six_digit_numbers_with_at_least_one_zero_l40_40866

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40866


namespace sampling_probabilities_equal_l40_40295

-- Definitions according to the problem conditions
def population_size := ℕ
def sample_size := ℕ
def simple_random_sampling (N n : ℕ) : Prop := sorry
def systematic_sampling (N n : ℕ) : Prop := sorry
def stratified_sampling (N n : ℕ) : Prop := sorry

-- Probabilities
def P1 : ℝ := sorry -- Probability for simple random sampling
def P2 : ℝ := sorry -- Probability for systematic sampling
def P3 : ℝ := sorry -- Probability for stratified sampling

-- Each definition directly corresponds to a condition in the problem statement.
-- Now, we summarize the equivalent proof problem in Lean.

theorem sampling_probabilities_equal (N n : ℕ) (h1 : simple_random_sampling N n) (h2 : systematic_sampling N n) (h3 : stratified_sampling N n) :
  P1 = P2 ∧ P2 = P3 :=
by sorry

end sampling_probabilities_equal_l40_40295


namespace max_sum_of_grid_is_306_l40_40192

-- Define the 10 x 10 grid
def grid (x : ℕ) (y : ℕ) : ℕ := sorry

noncomputable def sum_of_grid (g : ℕ → ℕ → ℕ) : ℕ :=
List.sum (List.map (λ (x : ℕ), List.sum (List.map (g x) (List.range 10))) (List.range 10))

-- Define the condition that each 1x3 strip sums to 9
def strip_sums_to_9 (g : ℕ → ℕ → ℕ) : Prop :=
(∀i : ℕ, i < 10 → (sum_of_list (g i [0,1,2]) = 9 ∧ 
          sum_of_list (g i [1,2,3]) = 9 ∧ 
          sum_of_list (g i [2,3,4]) = 9 ∧ 
          sum_of_list (g i [3,4,5]) = 9 ∧ 
          sum_of_list (g i [4,5,6]) = 9 ∧ 
          sum_of_list (g i [5,6,7]) = 9 ∧ 
          sum_of_list (g i [6,7,8]) = 9 ∧ 
          sum_of_list (g i [7,8,9]) = 9)) ∧
(∀i : ℕ, i < 10 → (sum_of_list (g [0,1,2] i) = 9 ∧ 
          sum_of_list (g [1,2,3] i) = 9 ∧ 
          sum_of_list (g [2,3,4] i) = 9 ∧ 
          sum_of_list (g [3,4,5] i) = 9 ∧ 
          sum_of_list (g [4,5,6] i) = 9 ∧ 
          sum_of_list (g [5,6,7] i) = 9 ∧ 
          sum_of_list (g [6,7,8] i) = 9 ∧ 
          sum_of_list (g [7,8,9] i) = 9))

-- Prove that maximum value of the sum of the grid is 306
theorem max_sum_of_grid_is_306 : 
  ∀ (g : ℕ → ℕ → ℕ), strip_sums_to_9 g → sum_of_grid g = 306 := 
sorry

end max_sum_of_grid_is_306_l40_40192


namespace minimum_parts_to_cut_rectangular_sheet_l40_40396

noncomputable def minimumRectangularParts (n : ℕ) : ℕ :=
  3 * n + 1

theorem minimum_parts_to_cut_rectangular_sheet (n : ℕ)
  (rectangular_sheet : ℕ)
  (holes : ℕ → Prop)
  (holes_not_overlap : ∀ i j, i ≠ j → holes i → holes j → ¬ (holes i ∧ holes j))
  (holes_not_touch : ∀ i j, holes i → holes j → ¬ (holes i ∧ holes j)) :
  ∃ minimum_parts : ℕ, minimum_parts = minimumRectangularParts n :=
by
  use minimumRectangularParts n
  sorry

end minimum_parts_to_cut_rectangular_sheet_l40_40396


namespace perpendiculars_always_intersect_l40_40809

variables {l l1 l2 l3 : Type}  {A B C : Type} {A1 B1 C1 : Type}

-- Defining distances
variables (a1 a2 b1 b2 c1 c2 x y z : ℝ)

-- Given conditions
axiom lines_perpendicular : (l1 ⊥ l) ∧ (l2 ⊥ l) ∧ (l3 ⊥ l)
axiom A_on_l : A ∈ l
axiom B_on_l : B ∈ l
axiom C_on_l : C ∈ l
axiom A1_on_l1 : A1 ∈ l1
axiom B1_on_l2 : B1 ∈ l2
axiom C1_on_l3 : C1 ∈ l3
axiom perpendiculars_intersect :
  ∃ (P : Type), ¬isEmpty P ∧
    (P ∈ (line_through A (perpendicular_to B1 P))) ∧
    (P ∈ (line_through B (perpendicular_to C1 P))) ∧
    (P ∈ (line_through C (perpendicular_to A1 P)))

-- Resultant proof statement
theorem perpendiculars_always_intersect :
  a1^2 - a2^2 + b1^2 - b2^2 + c1^2 - c2^2 = 0 :=
sorry

end perpendiculars_always_intersect_l40_40809


namespace ellipse_equation_triangle_area_l40_40827

-- Given conditions for ellipse C
variables (C : Ellipse)
variables (center : C.center = (0, 0))
variables (foci_on_x_axis : C.foci_on_x_axis)
variables (major_axis_length : C.major_axis_length = 4)
variables (point_on_ellipse : (1, Real.sqrt 3 / 2) ∈ C)

-- Given conditions for triangle PF₁F₂
variables (P : Point)
variables (second_quadrant : P.x < 0 ∧ P.y > 0)
variables (angle_F2PF1 : ∠ P C.F2 C.F1 = 60)

open RealPoint

-- Proving the equation of the ellipse
theorem ellipse_equation (C : Ellipse) 
    (center : C.center = (0, 0)) 
    (foci_on_x_axis : C.foci_on_x_axis) 
    (major_axis_length : C.major_axis_length = 4) 
    (point_on_ellipse : (1, Real.sqrt 3 / 2) ∈ C) : 
    C.equation = (∀ x y, x^2 / 4 + y^2 = 1) := sorry

-- Proving the area of triangle PF₁F₂
theorem triangle_area (C : Ellipse)
    (center : C.center = (0, 0)) 
    (foci_on_x_axis : C.foci_on_x_axis) 
    (major_axis_length : C.major_axis_length = 4) 
    (point_on_ellipse : (1, Real.sqrt 3 / 2) ∈ C) 
    (P : Point) 
    (second_quadrant : P.x < 0 ∧ P.y > 0) 
    (angle_F2PF1 : ∠ P C.F2 C.F1 = 60) : 
    area (triangle P C.F1 C.F2) = Real.sqrt 3 / 3 := sorry

end ellipse_equation_triangle_area_l40_40827


namespace expected_hit_targets_expected_hit_targets_not_less_than_half_l40_40260

-- Part (a): The expected number of hit targets
theorem expected_hit_targets (n : ℕ) (h : n ≠ 0) :
  E (number_of_hit_targets n) = n * (1 - (1 - (1 / n)) ^ n) :=
sorry

-- Part (b): The expected number of hit targets cannot be less than n / 2
theorem expected_hit_targets_not_less_than_half (n : ℕ) (h : n ≠ 0) :
  n * (1 - (1 - (1 / n)) ^ n) ≥ n / 2 :=
sorry

end expected_hit_targets_expected_hit_targets_not_less_than_half_l40_40260


namespace other_number_l40_40131

theorem other_number (a b : ℝ) (h : a = 0.650) (h2 : a = b + 0.525) : b = 0.125 :=
sorry

end other_number_l40_40131


namespace find_x_l40_40214

theorem find_x (x : ℝ) (h1 : list.set [2, 5, 7, x, 9, 10]) (hx : (medium (h1) = 2 * x)) : x = 3 :=
by
  sorry

end find_x_l40_40214


namespace coordinates_of_A_equidistant_BC_l40_40284

theorem coordinates_of_A_equidistant_BC :
  ∃ z : ℚ, (∀ A B C : ℚ × ℚ × ℚ, A = (0, 0, z) ∧ B = (7, 0, -15) ∧ C = (2, 10, -12) →
  (dist A B = dist A C)) ↔ z = -(13/3) :=
by sorry

end coordinates_of_A_equidistant_BC_l40_40284


namespace range_of_a_l40_40823

noncomputable def condition (x : ℝ) := (1/2 < x) ∧ (∀ a : ℝ, ln (2 * x - 1) ≤ x^2 + a)

theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, (1/2 < x) → ln (2 * x - 1) ≤ x^2 + a) ↔ a ≥ -1 :=
by sorry

end range_of_a_l40_40823


namespace total_savings_l40_40077

def weekly_savings : ℕ := 15
def weeks_per_cycle : ℕ := 60
def number_of_cycles : ℕ := 5

theorem total_savings :
  (weekly_savings * weeks_per_cycle) * number_of_cycles = 4500 := 
sorry

end total_savings_l40_40077


namespace complex_multiplication_eq_two_l40_40820

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := 1 + i

theorem complex_multiplication_eq_two : z * (1 - i) = 2 :=
by
  have h1 : z = 1 + i := rfl
  have h2 : i = complex.I := rfl
  sorry

end complex_multiplication_eq_two_l40_40820


namespace largest_number_in_systematic_sample_l40_40301

theorem largest_number_in_systematic_sample (n_products : ℕ) (start : ℕ) (interval : ℕ) (sample_size : ℕ) (largest_number : ℕ)
  (h1 : n_products = 500)
  (h2 : start = 7)
  (h3 : interval = 25)
  (h4 : sample_size = n_products / interval)
  (h5 : sample_size = 20)
  (h6 : largest_number = start + interval * (sample_size - 1))
  (h7 : largest_number = 482) :
  largest_number = 482 := 
  sorry

end largest_number_in_systematic_sample_l40_40301


namespace problem1_problem2_l40_40243

theorem problem1 : 4 * Real.sqrt 2 + Real.sqrt 8 - Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : Real.sqrt (4 / 3) / Real.sqrt (7 / 3) * Real.sqrt (7 / 5) = 2 * Real.sqrt 5 / 5 := by
  sorry

end problem1_problem2_l40_40243


namespace digits_difference_l40_40490

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end digits_difference_l40_40490


namespace binary_to_base4_conversion_l40_40250

theorem binary_to_base4_conversion : 
  let binary := (1*2^7 + 1*2^6 + 0*2^5 + 1*2^4 + 1*2^3 + 0*2^2 + 0*2^1 + 1*2^0) 
  let base4 := (3*4^3 + 1*4^2 + 2*4^1 + 1*4^0)
  binary = base4 := by
  sorry

end binary_to_base4_conversion_l40_40250


namespace length_chord_eq_l40_40832

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : Prop :=
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity of the hyperbola
def eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = Real.sqrt (1 + b^2 / a^2)

-- Define the circle
def circle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the conditions
variables
  (a b : ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (h_hyperbola : hyperbola a b a_pos b_pos)
  (h_eccentricity : eccentricity a b (Real.sqrt 5))
  (h_circle : circle (2, 3) 1)

-- Prove the length of the chord |AB|
theorem length_chord_eq :
  ∃ A B : ℝ × ℝ,
  (∃ L : ℝ, L = 2 * Real.sqrt ((1:ℝ)^2 - (1 / Real.sqrt 5)^2))
  ∧ |A.1 - B.1| + |A.2 - B.2| = 4 * Real.sqrt 5 / 5 := sorry

end length_chord_eq_l40_40832


namespace initial_honey_amount_l40_40177

-- Defining the conditions of the problem
def jar_full_of_honey (H : ℝ) (draw_percentage : ℝ) (replacements : ℕ) : Prop :=
  (∀ n, n ≤ replacements → n ≥ 0 → 
  let remaining_honey := H * (1 - draw_percentage/100)^n in
  (n = 4 → remaining_honey = 512))

-- Proving the initial amount of honey
theorem initial_honey_amount : ∃ H : ℝ, jar_full_of_honey H 20 4 ∧ H = 1250 :=
begin
  let draw_percentage := 20,
  let replacements := 4,
  let H := 1250,
  let remaining_honey := H * (1 - draw_percentage/100)^replacements,
  use H,
  split,
  { intros n h1 h2,
    cases (eq_or_ne n 4) with h3 h3,
    { rw h3,
      simp,
      have prf := Real.rpow_def_pos,
      sorry -- Proof that remaining_honey = 512
    },
    { sorry -- Proof for n not equal to 4 case
    }
  },
  { reflexivity }
end

end initial_honey_amount_l40_40177


namespace spider_to_fly_routes_l40_40695

-- Given conditions
def start_pos : ℕ × ℕ := (0, 0)
def end_pos : ℕ × ℕ := (5, 3)
def possible_moves : ℕ := end_pos.fst + end_pos.snd  -- Total moves: 5 right and 3 upward

-- Theorem to be proven
theorem spider_to_fly_routes : 
  nat.choose (start_pos.fst + start_pos.snd + possible_moves) end_pos.snd = 56 :=
sorry

end spider_to_fly_routes_l40_40695


namespace find_p_l40_40834

-- Definitions for the Lean 4 Statement
variables (p x y : ℝ)
def parabola (p x : ℝ) : Prop := y^2 = 2 * p * x
def focus_distance (p x : ℝ) : ℝ := real.sqrt ((x - p / 2)^2 + y^2)
def y_axis_distance (x : ℝ) : ℝ := real.abs x

-- Lean 4 Statement
theorem find_p (p_gt_zero : p > 0) (point_on_parabola : parabola p x)
  (distance_condition : focus_distance p x = y_axis_distance x + 2) : p = 4 :=
sorry

end find_p_l40_40834


namespace find_x_of_orthogonal_vectors_l40_40354

variable (x : ℝ)

def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (-4, 2, x)

theorem find_x_of_orthogonal_vectors (h : (2 * -4 + -3 * 2 + 1 * x) = 0) : x = 14 := by
  sorry

end find_x_of_orthogonal_vectors_l40_40354


namespace number_of_ordered_pairs_l40_40412

noncomputable def omega : ℂ :=
  Complex.ofReal (-1 / 2) + Complex.I * (Real.sqrt 3 / 2)

lemma omega_cube_eq_one (h : omega ≠ 1) : omega ^ 3 = 1 :=
by sorry

theorem number_of_ordered_pairs :
  ∃ (n : ℕ), n = 6 ∧
    ∀ (a b : ℤ), |a * omega + b| = 1 → (a, b) ∈ 
      {(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)} :=
by sorry

end number_of_ordered_pairs_l40_40412


namespace six_digit_numbers_with_zero_count_l40_40899

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40899


namespace find_b_l40_40112

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (3568432 * 85^0 - b) % 17 = 0 → b = 3 :=
begin
  sorry
end

end find_b_l40_40112


namespace function_f2_is_quadratic_l40_40170

/-- The function f2(x) = 2x^2 must be a quadratic function. -/
theorem function_f2_is_quadratic :
  ∀ (a b c : ℝ) (x : ℝ), 
  let f1 := (λ x, a * x^2 + b * x + c),
      f2 := (λ x, 2 * x^2),
      f3 := (λ x, x^2 + 1 / x),
      f4 := (λ x, (x + 1)^2 - x^2) in
  (f2 x = 2 * x^2) := sorry

end function_f2_is_quadratic_l40_40170


namespace solution_set_of_inequality_l40_40141

theorem solution_set_of_inequality :
  {x : ℝ | (x - 3) / x ≥ 0} = {x : ℝ | x < 0 ∨ x ≥ 3} :=
sorry

end solution_set_of_inequality_l40_40141


namespace find_A_d_minus_B_d_l40_40478

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end find_A_d_minus_B_d_l40_40478


namespace required_packages_l40_40548

noncomputable def rooms : List ℕ := (List.range' 300 26) ++ (List.range' 400 26)

def digit_count (d : ℕ) : ℕ :=
rooms.foldl (λ acc n => acc + (n.digitCount d)) 0

def most_frequent_digit_count : ℕ :=
List.maximum (List.map digit_count (List.range' 0 10))

theorem required_packages : most_frequent_digit_count = 26 :=
by
  sorry

end required_packages_l40_40548


namespace jimmy_lollipops_count_l40_40042

-- Definitions for the given conditions
def candy_bar_cost : ℝ := 0.75
def lollipop_cost : ℝ := 0.25
def total_driveways : ℝ := 10
def charge_per_driveway : ℝ := 1.5
def fraction_spent_on_candy : ℝ := 1 / 6

-- Statement to prove the number of lollipops Jimmy bought
theorem jimmy_lollipops_count : 
    let total_earned := charge_per_driveway * total_driveways in
    let total_spent := total_earned * fraction_spent_on_candy in
    let spent_on_candy_bars := 2 * candy_bar_cost in
    let spent_on_lollipops := total_spent - spent_on_candy_bars in
    let lollipops_bought := spent_on_lollipops / lollipop_cost in
    lollipops_bought = 4 :=
by 
    sorry

end jimmy_lollipops_count_l40_40042


namespace smallest_prime_divisor_of_sum_l40_40622

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end smallest_prime_divisor_of_sum_l40_40622


namespace maximize_alpha_l40_40050

def func_set (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f(3 * x) ≥ f(f(2 * x)) + x

theorem maximize_alpha :
  ∃ (α : ℝ), α = 1 / 2 ∧ (∀ f : ℝ → ℝ, func_set f → ∀ x : ℝ, x ≥ 0 → f(x) ≥ α * x) :=
sorry

end maximize_alpha_l40_40050


namespace rationalized_sum_l40_40097

theorem rationalized_sum : 
  ∃ A B C D : ℤ, 
    (D > 0) ∧ 
    (∀ p : ℕ, prime p → p * p ∣ B → false) ∧ 
    (Int.gcd A (Int.gcd C D) = 1) ∧ 
    (A * (Int.ofNat B).sqrt + C) / D = 7 * (3 - (Int.ofNat B).sqrt) / (3 + (Int.ofNat B).sqrt) ∧ 
    A + B + C + D = 23 := sorry

end rationalized_sum_l40_40097


namespace coefficient_x6_in_binomial_expansion_l40_40615

open Nat

theorem coefficient_x6_in_binomial_expansion :
  let n := 9
  let a := 3
  let b := 4
  let k := 6
  let binom := binomial n k
  let term_a := a^ (n - k)
  let term_b := b^ k
  binom * term_a * term_b = 9237888 := 
by
  let n := 9
  let a := 3
  let b := 4
  let k := 6
  let binom := binomial n k
  let term_a := a^ (n - k)
  let term_b := b^ k
  sorry

end coefficient_x6_in_binomial_expansion_l40_40615


namespace rationalize_fraction_sum_l40_40091

theorem rationalize_fraction_sum :
  ∃ (A B C D : ℤ),
    D > 0 ∧
    ¬ ∃ p : ℕ, p.prime ∧ p^2 ∣ B ∧
    Int.gcd A C D = 1 ∧
    A * 8 + C = 21 - 7 * 3 ∧
    A + B + C + D = 23 :=
sorry

end rationalize_fraction_sum_l40_40091


namespace angle_A_value_l40_40381

variable (A B C : Type) [Real ℝ]
variable {a b : ℝ}
variable (S : Triangle A B C → ℝ)

theorem angle_A_value (b : ℝ) (c : ℝ) (S : ℝ) :
  b = 8 → c = 8 * Real.sqrt 3 → S = 16 * Real.sqrt 3 → 
  (S = (1 / 2) * b * c * Real.sin A) → (A = Real.pi / 6 ∨ A = 5 * Real.pi / 6) :=
by
  intros hb hc hs hformula
  sorry

end angle_A_value_l40_40381


namespace circle_area_polar_eq_l40_40538

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l40_40538


namespace average_hit_targets_value_average_hit_targets_ge_half_l40_40270

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end average_hit_targets_value_average_hit_targets_ge_half_l40_40270


namespace value_of_m_over_q_l40_40801

-- Definitions for the given conditions
variables (n m p q : ℤ) 

-- Main theorem statement
theorem value_of_m_over_q (h1 : m = 10 * n) (h2 : p = 2 * n) (h3 : p = q / 5) :
  m / q = 1 :=
sorry

end value_of_m_over_q_l40_40801


namespace area_circle_l40_40533

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l40_40533


namespace function_zero_property_l40_40405

noncomputable def A (f : ℝ → ℝ) : Prop := 
  ∀ (f1 f2 : ℝ → ℝ), A f1 → A f2 → ∃ (f3 : ℝ → ℝ), A f3 ∧ 
  ∀ x y : ℝ, f1 (f2 y - x) + 2 * x = f3 (x + y)

theorem function_zero_property (A : (ℝ → ℝ) → Prop)
    (hA: ∀ (f1 f2 : ℝ → ℝ), A f1 → A f2 → ∃ (f3 : ℝ → ℝ), A f3 ∧ ∀ x y : ℝ, f1 (f2 y - x) + 2 * x = f3 (x + y)) :
    ∀ f : ℝ → ℝ, A f → ( ∀ x : ℝ, f (x - f x) = 0 ) :=
sorry

end function_zero_property_l40_40405


namespace count_sets_of_three_consecutive_primes_l40_40699

def ramanujan_number : ℕ := 1729
def reversed_ramanujan_number : ℕ := 9271 -- The reversed number
def primes : list ℕ := [7, 17, 79] -- The prime factors of 9271

def sum_primes_equals_103 (l : list ℕ) : Prop :=
  (l.length = 3) ∧ (∃ p1 p2 p3 : ℕ, p1 + p2 + p3 = 103 ∧ l = [p1, p2, p3]) ∧
  (∀ i ∈ l, i ∈ primes) ∧
  (∀ i ∈ l, is_prime i)

def count_consecutive_prime_sets_sum_103 : ℕ := 1 -- The number of sets

theorem count_sets_of_three_consecutive_primes :
  ∃ (l : list ℕ), (sum_primes_equals_103 l) ∧ (count_consecutive_prime_sets_sum_103 = 1) :=
by
  sorry

end count_sets_of_three_consecutive_primes_l40_40699


namespace probability_of_small_flower_l40_40547

theorem probability_of_small_flower :
  (let p_small_seed := 6 / 10;
       prob_small_flower := 0.9;
       p_large_seed := 4 / 10;
       prob_large_not_flower := 1 - 0.8 in
   p_small_seed * prob_small_flower + p_large_seed * prob_large_not_flower = 0.62) :=
sorry

end probability_of_small_flower_l40_40547


namespace rhombus_side_length_l40_40775

-- Definitions and conditions given in the problem
variables {O M L P R A B C D : Type}
variables (s : ℝ) (circle : Set ℝ) (O L : ℝ)

-- The center of the circle is O, OL = s / 2
def is_center (O : ℝ) (circle : Set ℝ) : Prop := (O ∈ circle)
def tangent_point (M : ℝ) (circle : Set ℝ) : Prop := -- M is a point on the tangent to the circle
def tangent_length (O L : ℝ) (s : ℝ) : Prop := O.2 = (s / 2)

-- Prove that the rhombus constructed around the circle has side length s
theorem rhombus_side_length (O M L P R A B C D : ℝ) (circle : Set ℝ) 
  (s : ℝ) (h1 : is_center O circle) (h2 : tangent_point M circle) (h3 : tangent_length O L s) :
  ∃ (A B C D : ℝ), side_length A B = s := sorry

end rhombus_side_length_l40_40775


namespace sum_of_cubes_l40_40370

theorem sum_of_cubes (x y : ℝ) (hx : x + y = 10) (hxy : x * y = 12) : x^3 + y^3 = 640 := 
by
  sorry

end sum_of_cubes_l40_40370


namespace six_digit_numbers_with_zero_l40_40877

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40877


namespace total_donation_correct_l40_40448

noncomputable def donation_multipliers : List ℝ := [1.03, 1.05, 1.08, 1.02, 1.10, 1.04, 1.06, 1.09, 1.07, 1.03, 1.05]

def calculate_monthly_donations (initial_donation : ℝ) (multipliers : List ℝ) : List ℝ :=
  multipliers.scanl (λ acc multiplier, acc * multiplier) initial_donation

def total_donation (initial_donation : ℝ) (multipliers : List ℝ) : ℝ :=
  (calculate_monthly_donations initial_donation multipliers).sum + initial_donation

theorem total_donation_correct :
  total_donation 1707 donation_multipliers = 29906.10 :=
by
  sorry

end total_donation_correct_l40_40448


namespace winner_vote_count_l40_40382

theorem winner_vote_count (total_votes : ℕ) (x y z w : ℕ) 
  (h1 : total_votes = 5219000) 
  (h2 : y + 22000 = x)
  (h3 : z + 30000 = x)
  (h4 : w + 73000 = x)
  (h5 : x + y + z + w = total_votes) :
  x = 1336000 :=
by
  sorry

end winner_vote_count_l40_40382


namespace total_bill_l40_40224

-- Definitions from conditions
def num_people : ℕ := 3
def amount_per_person : ℕ := 45

-- Mathematical proof problem statement
theorem total_bill : num_people * amount_per_person = 135 := by
  sorry

end total_bill_l40_40224


namespace length_DF_l40_40434

variables {F E P Q G : Type*}
variable [metric_space F]
variables (triangle_DEF : E)
variables (D P E Q : F) 

-- Given conditions
def D_median_P := 18 : ℝ
def E_median_Q := 24 : ℝ
def perpendicular_medians : true := sorry  -- Given that DP and EQ are perpendicular

-- We want to prove that DF = 8√13
theorem length_DF : 
  (distance D E) = 8 * (real.sqrt 13) := sorry

end length_DF_l40_40434


namespace six_digit_numbers_with_zero_l40_40934

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40934


namespace math_problem_l40_40010

-- Defining the operation ø
def op_ø (x w : ℕ) : ℝ := (2 ^ x) / (2 ^ w)

-- Defining the constants
def a : ℕ := 3
def b : ℕ := 2
def c : ℕ := 5
def d : ℕ := 1
def e : ℕ := 4
def f : ℕ := 2

-- Using the conditions to formulate the problem
def prob : Prop :=
  (op_ø (a + b) (c - d)) ø (op_ø e f) = 1 / 4

-- The statement to prove
theorem math_problem : prob := by
  sorry

end math_problem_l40_40010


namespace dress_designs_total_l40_40200

theorem dress_designs_total (colors patterns : ℕ) (h_colors : colors = 4) (h_patterns : patterns = 6) :
  colors * patterns = 24 := 
by
  rw [h_colors, h_patterns]
  sorry

end dress_designs_total_l40_40200


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40941

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40941


namespace six_digit_numbers_with_zero_l40_40872

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40872


namespace monthly_price_reduction_rate_l40_40026

-- Let's define the given conditions
def initial_price_March : ℝ := 23000
def price_in_May : ℝ := 16000

-- Define the monthly average price reduction rate
variable (x : ℝ)

-- Define the statement to be proven
theorem monthly_price_reduction_rate :
  23 * (1 - x) ^ 2 = 16 :=
sorry

end monthly_price_reduction_rate_l40_40026


namespace cake_cut_proof_l40_40686

theorem cake_cut_proof (n : ℕ) (hn : n ≥ 3) :
  ∀ (cuts : Fin n → ℝ → Point → Prop), 
  (∀ (i : Fin n) (l : ℝ), l ≥ 1 → cuts i l) →
  (∃ (i : Fin n), (∃ (l : ℝ), cuts i l)) :=
by
  sorry

end cake_cut_proof_l40_40686


namespace y_intercept_perpendicular_line_tangent_lines_l40_40826

noncomputable def circle (x y R) : Prop := x^2 + y^2 = R^2

/-- Theorem 1: 
Find the y-intercept of a line perpendicular to l_1 that intersects circle C at two distinct points P and Q. -/
theorem y_intercept_perpendicular_line (b : ℝ) : 
  let l_1 := λ x y : ℝ, x - y - 2 * real.sqrt 2 = 0
  let C := circle 0 0 2
  ∃ (P Q: ℝ × ℝ), ∃ (y_int: ℝ), ∃ b : ℝ, l_1 0 b ∧ C P.fst P.snd ∧ C Q.fst Q.snd ∧ P ≠ Q ∧
  ((P.fst - Q.fst) * (P.snd - Q.snd) = -1) ∧ 
  (b = 2 ∨ b = -2) := sorry

/-- Theorem 2: 
Find the equation of the tangent line to circle C passing through point G(1, 3). -/
theorem tangent_lines (k: ℝ) : 
  let G := (1, 3)
  let C := (λ x y : ℝ, x^2 + y^2 = 4)
  ∃ k (k1 k2 : ℝ), 
  (k = (-3 + 2 * real.sqrt 6) / 3 ∨ k = (-3 - 2 * real.sqrt 6) / 3) ∧ 
  (∀ (x y : ℝ), k * x - y + 3 - k = 0) := sorry

end y_intercept_perpendicular_line_tangent_lines_l40_40826


namespace sqrt_sum_ineq_l40_40423

theorem sqrt_sum_ineq (n : ℕ) (a : ℝ) (hn : 2 ≤ n) (ha : 2 ≤ a) : 
  (∑ k in Finset.range n, (λ k, (Nat.sqrt (1 + Nat.sqrt (a + Nat.sqrt (a ^ 2 + ... + Nat.sqrt (a ^ k))))))) < n * a :=
sorry

end sqrt_sum_ineq_l40_40423


namespace arithmetic_sequence_of_Sn_l40_40037

-- Define the condition: sum of the first n terms
def Sn (n : ℕ) : ℤ := 2 * n ^ 2 - 3 * n

-- Define a sequence
def a_n : ℕ → ℤ

-- Assuming the given sum is the sum of the sequence
axiom h1 : ∀ n, (∑ i in finset.range n, a_n i) = Sn n

-- Statement to prove: The sequence a_n is an arithmetic sequence
theorem arithmetic_sequence_of_Sn (a_n : ℕ → ℤ) (Sn : ℕ → ℤ)
  (h1 : ∀ n, (∑ i in finset.range n, a_n i) = Sn n) :
  ∀ n, a_n (n + 1) - a_n n = a_n 1 - a_n 0 :=
by 
  sorry

end arithmetic_sequence_of_Sn_l40_40037


namespace increase_interval_l40_40846

noncomputable def f (x : ℝ) : ℝ := x ^ (-2)

theorem increase_interval (h : f 2 = 1 / 4) :
  (∀ x y : ℝ, x < y ∧ 0 < x → f x < f y) :=
by
  intros x y hx hy
  rw [f, f]
  sorry

end increase_interval_l40_40846


namespace isabel_homework_problems_l40_40398

variable (subtasks_per_problem : ℕ) (total_subtasks : ℕ)

theorem isabel_homework_problems (h1 : subtasks_per_problem = 5)
                                 (h2 : total_subtasks = 200) :
                                 (total_subtasks / subtasks_per_problem) = 40 :=
by
  rw [h1, h2]
  norm_num

end isabel_homework_problems_l40_40398


namespace find_f2_l40_40178

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x, f(x) + 3 * f(8 - x) = x) : f(2) = 2 :=
sorry

end find_f2_l40_40178


namespace equal_lengths_DF_KF_l40_40031

variables {A B C D E G H K F : Type} [Geometry A B C D E G H K F]

-- Conditions
def triangle_isosceles (A B C : Type) (AB AC : Prop) : Prop := AB = AC
def perpendicular (A D B C : Type) : Prop := AD ⟂ BC
def midpoint (G E B : Type) : Prop := midpoint G E B
def perpendicular_at_point (A G H : Type) : Prop := ⟂ A G H

-- Definitions from conditions
noncomputable def is_midpoint_BC (D : Type) [triangle_isosceles A B C AB AC] : Prop :=
  midpoint D B C

noncomputable def is_midpoint_BE (G : Type) [midpoint G E B] : Prop

noncomputable def EH_perpendicular_AG (H : Type) [perpendicular_at_point A G H] : Prop

theorem equal_lengths_DF_KF
  (isosceles : triangle_isosceles A B C AB AC)
  (AD_perp_BC : perpendicular A D B C)
  (DE_perp_AC : perpendicular D E AC)
  (G_mid_BE : is_midpoint_BE G)
  (EH_perp_AG : EH_perpendicular_AG H)
  (BE_inter_AD : intersect BE AD F) :
  DF = KF :=
sorry

end equal_lengths_DF_KF_l40_40031


namespace average_distance_is_600_l40_40440

-- Definitions based on the conditions
def one_lap_distance : ℕ := 200
def johnny_lap_count : ℕ := 4
def mickey_lap_count : ℕ := johnny_lap_count / 2

-- Total distances run by Johnny and Mickey
def johnny_distance : ℕ := johnny_lap_count * one_lap_distance
def mickey_distance : ℕ := mickey_lap_count * one_lap_distance

-- Sum of distances
def total_distance : ℕ := johnny_distance + mickey_distance

-- Number of people
def number_of_people : ℕ := 2

-- Average distance calculation
def average_distance : ℕ := total_distance / number_of_people

-- The theorem to prove: the average distance run by Johnny and Mickey
theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end average_distance_is_600_l40_40440


namespace coffee_break_participants_l40_40580

theorem coffee_break_participants (n : ℕ) (h1 : n = 14) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 6 ∧ n - 2 * k > 0 ∧ n - 2 * k < n)
    (h3 : ∀ m, m < n → (∃ k, m = n - 2 * k → m ∈ {6, 8, 10, 12})): 
    ∃ m, n - 2 * m ∈ {6, 8, 10, 12} := 
by
  use 6
  use 8
  use 10
  use 12
  sorry

end coffee_break_participants_l40_40580


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40945

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40945


namespace proof_of_area_weighted_sum_of_distances_l40_40704

def area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) 
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ) 
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : Prop :=
  t1 * z1 + t2 * z2 + t3 * z3 + t4 * z4 = t * z

theorem proof_of_area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ)
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : area_weighted_sum_of_distances a b a1 a2 a3 a4 b1 b2 b3 b4 t1 t2 t3 t4 t z1 z2 z3 z4 z h1 h2 h3 h4 rect_area :=
  sorry

end proof_of_area_weighted_sum_of_distances_l40_40704


namespace geom_seq_common_ratio_l40_40030

theorem geom_seq_common_ratio (a1 : ℤ) (S3 : ℚ) (q : ℚ) (hq : -2 * (1 + q + q^2) = - (7 / 2)) : 
  q = 1 / 2 ∨ q = -3 / 2 :=
sorry

end geom_seq_common_ratio_l40_40030


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40944

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40944


namespace trapezoid_area_sum_l40_40223

def trapezoid_side_lengths : List ℕ := [4, 6, 8, 10]

theorem trapezoid_area_sum :
  ∃ (r_1 r_2 r_3 n_1 n_2 : ℚ) (S_total : ℚ),
    S_total = r_1 * Real.sqrt n_1 + r_2 * Real.sqrt n_2 + r_3
    ∧ r_1 + r_2 + r_3 + n_1 + n_2 ≤ 274
    ∧ ∀ a₁ a₂ a₃ a₄ ∈ trapezoid_side_lengths,
        a₁ ≠ a₂ → a₁ ≠ a₃ → a₁ ≠ a₄ → a₂ ≠ a₃ → a₂ ≠ a₄ → a₃ ≠ a₄ → 
        let S₁ := Real.sqrt ((a₁ + a₂ + a₃) / 2 * ((a₁ + a₂ + a₃) / 2 - a₁) * ((a₁ + a₂ + a₃) / 2 - a₂) * ((a₁ + a₂ + a₃) / 2 - a₃)),
            S₂ := Real.sqrt ((a₁ + a₂ + a₄) / 2 * ((a₁ + a₂ + a₄) / 2 - a₁) * ((a₁ + a₂ + a₄) / 2 - a₂) * ((a₁ + a₂ + a₄) / 2 - a₄)),
            S₃ := Real.sqrt ((a₁ + a₃ + a₄) / 2 * ((a₁ + a₃ + a₄) / 2 - a₁) * ((a₁ + a₃ + a₄) / 2 - a₃) * ((a₁ + a₃ + a₄) / 2 - a₄)),
            S₄ := Real.sqrt ((a₂ + a₃ + a₄) / 2 * ((a₂ + a₃ + a₄) / 2 - a₂) * ((a₂ + a₃ + a₄) / 2 - a₃) * ((a₂ + a₃ + a₄) / 2 - a₄)),
            A₁ := if n_1 ≠ 0 then S₁ else 0,
            A₂ := if n_2 ≠ 0 then S₂ else 0,
            A_total := A₁ + A₂ in
          S_total = A_total
    ∧ (floor (r_1 + r_2 + r_3 + n_1 + n_2) = 274) := sorry

end trapezoid_area_sum_l40_40223


namespace ratio_sum_odd_even_divisors_l40_40062

theorem ratio_sum_odd_even_divisors (M : ℕ) (h : M = 36 * 36 * 98 * 150) :
  ratio_of_sums_of_odd_and_even_divisors M = 1 / 62 :=
by
  sorry

end ratio_sum_odd_even_divisors_l40_40062


namespace total_packets_needed_l40_40745

theorem total_packets_needed :
  let oak_seedlings := 420
  let oak_per_packet := 7
  let maple_seedlings := 825
  let maple_per_packet := 5
  let pine_seedlings := 2040
  let pine_per_packet := 12
  let oak_packets := oak_seedlings / oak_per_packet
  let maple_packets := maple_seedlings / maple_per_packet
  let pine_packets := pine_seedlings / pine_per_packet
  let total_packets := oak_packets + maple_packets + pine_packets
  total_packets = 395 := 
by {
  sorry
}

end total_packets_needed_l40_40745


namespace six_digit_numbers_with_zero_l40_40999

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l40_40999


namespace monica_total_savings_l40_40075

theorem monica_total_savings :
  ∀ (weekly_saving : ℤ) (weeks_per_cycle : ℤ) (cycles : ℤ),
    weekly_saving = 15 →
    weeks_per_cycle = 60 →
    cycles = 5 →
    weekly_saving * weeks_per_cycle * cycles = 4500 :=
by
  intros weekly_saving weeks_per_cycle cycles
  sorry

end monica_total_savings_l40_40075


namespace partial_fraction_decomposition_l40_40520

theorem partial_fraction_decomposition :
  ∃ A B : ℚ, (A = 43 / 14) ∧ (B = -31 / 14) ∧
  (3 * A + B = 7) ∧ (-2 * A + 4 * B = -15) :=
by
  use 43 / 14, -31 / 14
  split
  { sorry },
  split
  { sorry },
  split
  { sorry },
  { sorry }

end partial_fraction_decomposition_l40_40520


namespace even_digit_right_square_finite_odd_digit_right_square_infinite_l40_40206

-- Definitions for fissile squares
def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def non_zero_starting_square (n : ℕ) : Prop := n > 0 ∧ (n / 10 ^ (Nat.log10 n)) ≠ 0

def is_fissile_square (n : ℕ) : Prop :=
  is_square n ∧ ∃ (left right : ℕ), 
    non_zero_starting_square left ∧
    non_zero_starting_square right ∧
    (Nat.log10 right).odd = false ∧ 
    (left * 10 ^ (Nat.log10 right + 1) + right = n)

-- Part (a) statement: Every square with an even number of digits is the right square of only finitely many fissile squares.
theorem even_digit_right_square_finite (n : ℕ) (h1 : is_square n) (h2 : non_zero_starting_square n) (hn : (Nat.log10 n).odd = false) :
  {m : ℕ | is_fissile_square (m * n)}.finite :=
sorry

-- Part (b) statement: Every square with an odd number of digits is the right square of infinitely many fissile squares.
theorem odd_digit_right_square_infinite (n : ℕ) (h1 : is_square n) (h2 : non_zero_starting_square n) (hn : (Nat.log10 n).odd = true) :
  {m : ℕ | is_fissile_square (m * n)}.infinite :=
sorry

end even_digit_right_square_finite_odd_digit_right_square_infinite_l40_40206


namespace distribution_and_replanting_probability_l40_40148

variable (n : ℕ) (p : ℝ)
variable (X : ℕ → ℕ)

-- Conditions
axiom (h1 : Expectation X = 3)
axiom (h2 : StandardDeviation X = Real.sqrt 1.5)

-- Prove they imply n=6 and p=1/2, and that the distribution probabilities
theorem distribution_and_replanting_probability :
    n = 6 ∧ p = 1/2 ∧ 
    (∀ x, ∃! prob, 
        prob = match x with 
        | 0 => 1/64
        | 1 => 6/64
        | 2 => 15/64
        | 3 => 20/64
        | 4 => 15/64
        | 5 => 6/64
        | 6 => 1/64
        | _ => 0) ∧
    Probability (λ x, x ≤ 3) = 21/32 :=
by
  sorry

end distribution_and_replanting_probability_l40_40148


namespace area_of_circle_l40_40528

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l40_40528


namespace train_length_correct_l40_40221

noncomputable def train_length (speed_kmph: ℝ) (time_sec: ℝ) : ℝ :=
  let speed_mps := speed_kmph * (5 / 18)
  speed_mps * time_sec

theorem train_length_correct : train_length 250 12 = 833.28 := by
  sorry

end train_length_correct_l40_40221


namespace plastering_cost_correct_l40_40714

def length : ℕ := 40
def width : ℕ := 18
def depth : ℕ := 10
def cost_per_sq_meter : ℚ := 1.25

def area_bottom (L W : ℕ) : ℕ := L * W
def perimeter_bottom (L W : ℕ) : ℕ := 2 * (L + W)
def area_walls (P D : ℕ) : ℕ := P * D
def total_area (A_bottom A_walls : ℕ) : ℕ := A_bottom + A_walls
def total_cost (A_total : ℕ) (cost_per_sq_meter : ℚ) : ℚ := A_total * cost_per_sq_meter

theorem plastering_cost_correct :
  total_cost (total_area (area_bottom length width)
                        (area_walls (perimeter_bottom length width) depth))
             cost_per_sq_meter = 2350 :=
by 
  sorry

end plastering_cost_correct_l40_40714


namespace water_to_add_l40_40198

theorem water_to_add (x : ℚ) (alcohol water : ℚ) (ratio : ℚ) :
  alcohol = 4 → water = 4 →
  (3 : ℚ) / (3 + 5) = (3 : ℚ) / 8 →
  (5 : ℚ) / (3 + 5) = (5 : ℚ) / 8 →
  ratio = 5 / 8 →
  (4 + x) / (8 + x) = ratio →
  x = 8 / 3 :=
by
  intros
  sorry

end water_to_add_l40_40198


namespace find_number_l40_40189

theorem find_number (x : ℝ) (h : 50 + 5 * 12 / (x / 3) = 51) : x = 180 := 
by 
  sorry

end find_number_l40_40189


namespace zeros_in_decimal_representation_l40_40660

theorem zeros_in_decimal_representation :
  let d := (7 : ℚ) / 8000 in
  ∃ n : ℕ, d = 875 / (10 ^ 6) ∧ n = 3 :=
by
  let d := (7 : ℚ) / 8000
  use 3
  constructor
  . norm_num [d]
  . rfl

end zeros_in_decimal_representation_l40_40660


namespace constant_sequence_l40_40065

variable (a : ℕ → ℝ)

-- Conditions of the problem
axiom positive_terms : ∀ n, a n > 0
axiom inequality_condition : ∀ n, ( (∑ i in Finset.range n, a i) / n) ≥ sqrt ((∑ i in Finset.range (n+1), (a i) ^ 2) / (n+1))

-- The proof statement
theorem constant_sequence : ∀ n m, a n = a m :=
sorry

end constant_sequence_l40_40065


namespace finitely_generated_ideal_l40_40153

variables {R : Type*} [CommRing R] (α : R) (αs : Fin n → R) (rs : Fin n → R)

def sigma (k : ℕ) : R :=
  k * α + ∑ i, αs i * (rs i) ^ k

theorem finitely_generated_ideal :
  Ideal.fg (Ideal.span (set.range (λ k, sigma α αs rs k))) :=
sorry

end finitely_generated_ideal_l40_40153


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40911

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40911


namespace exists_unique_line_l40_40353

noncomputable def ratio_of_segments (A B C : Point) (m n : ℝ) : Prop :=
  AB_ratio_eq_AC : (dist A B / dist A C) = (m / n)

noncomputable def line_through_point (l1 l2 : Line) (A : Point) (m n : ℝ) : Prop :=
  ∃ B C: Point, (B ∈ l1) ∧ (C ∈ l2) ∧ (line_through A B = line_through A C) ∧ (ratio_of_segments A B C m n)

theorem exists_unique_line (l1 l2 : Line) (A : Point) (m n : ℝ) (h : ¬ parallel l1 l2) :
  ∃! (l : Line), line_through_point l1 l2 A m n :=
sorry

end exists_unique_line_l40_40353


namespace num_zeros_in_decimal_rep_l40_40647

theorem num_zeros_in_decimal_rep (a b : ℕ) (h₁ : a = 7) (h₂ : b = 8000) :
  num_zeros_between_decimal_and_first_nonzero (a / b : ℚ) = 2 :=
sorry

end num_zeros_in_decimal_rep_l40_40647


namespace six_digit_numbers_with_zero_l40_40931

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40931


namespace six_digit_numbers_with_at_least_one_zero_l40_40967

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40967


namespace coffee_break_l40_40574

variable {n : ℕ}
variables {participants : Finset ℕ} (hparticipants : participants.card = 14)
variables (left : Finset ℕ)
variables (stayed : Finset ℕ) (hstayed : stayed.card = 14 - 2 * left.card)

-- The overall proof problem statement:
theorem coffee_break (h : ∀ x ∈ stayed, ∃! y, (y ∈ left ∧ adjacent x y participants)) :
  left.card = 6 ∨ left.card = 8 ∨ left.card = 10 ∨ left.card = 12 := 
sorry

end coffee_break_l40_40574


namespace six_digit_numbers_with_zero_count_l40_40907

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l40_40907


namespace six_digit_numbers_with_zero_l40_40980

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l40_40980


namespace six_digit_numbers_with_at_least_one_zero_l40_40970

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l40_40970


namespace prove_true_propositions_l40_40723

-- Define propositions
def p1 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ (1 / 2) ^ x₀ < (1 / 3) ^ x₀
def p2 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ x₀ < 1 ∧ log (1 / 2) x₀ > log (1 / 3) x₀
def p3 : Prop := ∀ (x : ℝ), 0 < x → (1 / 2) ^ x > log (1 / 2) x
def p4 : Prop := ∀ (x : ℝ), 0 < x ∧ x < 1 / 3 → (1 / 2) ^ x < log (1 / 2) x

-- Theorem statement
theorem prove_true_propositions : (¬ p1) ∧ p2 ∧ (¬ p3) ∧ p4 :=
by
  sorry

end prove_true_propositions_l40_40723


namespace min_value_l40_40807

theorem min_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + y = 1) : 
  ∃ (m : ℝ), m = 16 ∧ (∀ (x y : ℝ), x > 0 → y > 0 → 4 * x + y = 1 → (1 / x + 4 / y) ≥ m) := 
begin
  use 16,
  split,
  { refl, },
  { intros x y hx hy hxy,
    sorry
  }
end

end min_value_l40_40807


namespace arithmetic_sequence_general_formula_l40_40316

theorem arithmetic_sequence_general_formula :
  ∀ (a : ℕ → ℤ) (d : ℤ),
  (d ≠ 0) →
  (a 1 = 3) →
  (a 4 = a 1 + 3 * d) →
  (a 13 = a 1 + 12 * d) →
  (a 4 ^ 2 = a 1 * a 13) →
  ∀ n : ℕ, a n = 2 * n + 1 :=
begin
  sorry
end

end arithmetic_sequence_general_formula_l40_40316


namespace six_digit_numbers_with_at_least_one_zero_l40_40865

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40865


namespace six_digit_numbers_with_at_least_one_zero_correct_l40_40939

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l40_40939


namespace smallest_among_given_numbers_l40_40171

theorem smallest_among_given_numbers : ∀ x ∈ ({-2, -1/2, 1/2, 2} : Set ℝ), -2 ≤ x :=
by {
  intro x,
  intro hx,
  fin_cases hx; sorry
}

end smallest_among_given_numbers_l40_40171


namespace complement_union_correct_l40_40053

noncomputable def U : Set ℕ := {2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {x | x^2 - 6*x + 8 = 0}
noncomputable def B : Set ℕ := {2, 5, 6}

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 5, 6} := 
by
  sorry

end complement_union_correct_l40_40053


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40915

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40915


namespace mark_parking_tickets_l40_40735

theorem mark_parking_tickets (total_tickets : ℕ) (same_speeding_tickets : ℕ) (mark_parking_mult_sarah : ℕ) (sarah_speeding_tickets : ℕ) (mark_speeding_tickets : ℕ) (sarah_parking_tickets : ℕ) :
  total_tickets = 24 →
  mark_parking_mult_sarah = 2 →
  mark_speeding_tickets = same_speeding_tickets →
  sarah_speeding_tickets = same_speeding_tickets →
  same_speeding_tickets = 6 →
  sarah_parking_tickets = (total_tickets - 2 * same_speeding_tickets) / 3 →
  2 * sarah_parking_tickets = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw h1 at h6
  rw h5 at h6
  rw h2 at h6
  sorry

end mark_parking_tickets_l40_40735


namespace scientific_notation_l40_40450

theorem scientific_notation (n : ℝ) (h1 : n = 17600) : ∃ a b, (a = 1.76) ∧ (b = 4) ∧ n = a * 10^b :=
by {
  sorry
}

end scientific_notation_l40_40450


namespace prod_eq_one_l40_40752

noncomputable def P (x : ℂ) : ℂ :=
  ∏ k in finset.range 15, (x - complex.exp (2 * real.pi * complex.I * k / 17))

noncomputable def Q (x : ℂ) : ℂ :=
  ∏ j in finset.range 12, (x - complex.exp (2 * real.pi * complex.I * j / 13))

theorem prod_eq_one :
  (∏ k in finset.range 15, ∏ j in finset.range 12, (complex.exp (2 * real.pi * complex.I * j / 13) - complex.exp (2 * real.pi * complex.I * k / 17))) = 1 := 
by
  sorry

end prod_eq_one_l40_40752


namespace correct_conclusions_l40_40228

theorem correct_conclusions (P1 P2 P3 P4 : Prop) :
  (P1 → ∀ α β : Angle, (α.sides ∥ β.sides) → α = β) →
  (P2 → ∀ l1 l2 m1 m2 : Line, (l1 ∩ l2) ∥ (m1 ∩ m2) → (acuteAngle l1 l2 = acuteAngle m1 m2) ∨ (rightAngle l1 l2 = rightAngle m1 m2)) →
  (P3 → ∀ α β : Angle, (α.sides ⊥ β.sides) → α = β ∨ α + β = 180) →
  (P4 → ∀ l1 l2 l3 : Line, (l1 ∥ l3 ∧ l2 ∥ l3) → l1 ∥ l2) →
  (P2 ∧ P4).
Proof
  intros h1 h2 h3 h4,
  split;
  sorry.

end correct_conclusions_l40_40228


namespace distinct_points_with_integer_distances_l40_40045

theorem distinct_points_with_integer_distances (n : ℕ) (hn : n > 2) : 
  ∃ (points : fin n → ℝ × ℝ), 
    (∀ i j : fin n, i ≠ j → ∃ d : ℕ, 
      real.dist (points i) (points j) = d) 
    ∧ ¬ ∃ l : ℝ, ∃ b : ℝ, ∀ i : fin n, (points i).2 = l * (points i).1 + b :=
sorry

end distinct_points_with_integer_distances_l40_40045


namespace base_of_parallelogram_l40_40283

theorem base_of_parallelogram (area height : ℝ) (h_area : area = 384) (h_height : height = 16) : 
  (area / height) = 24 :=
by
  simp [h_area, h_height]
  norm_num
  sorry

end base_of_parallelogram_l40_40283


namespace arithmetic_sequence_a2_a9_sum_l40_40020

theorem arithmetic_sequence_a2_a9_sum 
  (a : ℕ → ℝ) (d a₁ : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S10 : 10 * a 1 + 45 * d = 120) :
  a 2 + a 9 = 24 :=
sorry

end arithmetic_sequence_a2_a9_sum_l40_40020


namespace max_value_of_f_in_domain_l40_40515

noncomputable def domain : set ℝ := {x | 3 - 4*x + x^2 > 0}
noncomputable def f (x : ℝ) : ℝ := 2^x + 2 - 3 * 4^x

theorem max_value_of_f_in_domain :
  (∀ x ∈ domain, f x ≤ 25 / 12) ∧ (∃ x ∈ domain, f x = 25 / 12) :=
by
  sorry

end max_value_of_f_in_domain_l40_40515


namespace curves_equiv_and_intersection_l40_40468

theorem curves_equiv_and_intersection (C1 C2 : ℝ → ℝ × ℝ) (ρ θ : ℝ) 
  (hC1 : ∀ t, C1 t = (-t, -1 + √3 * t))
  (hC2 : ∀ θ, (ρ, θ) = (2 * sin θ - 2 * √3 * cos θ, θ))
  (A B : ℝ × ℝ) :
  (∀ x y, y = -1 - √3 * x ↔ (x, y) ∈ set.range (C1)) ∧ 
  (∀ x y, x^2 + 2 * √3 * x + y^2 - 2 * y = 0 ↔ (x, y) ∈ set.range (C2)) ∧ 
  dist A B = 2 * sqrt (4 - (1 / 2)^2) := 
sorry

end curves_equiv_and_intersection_l40_40468


namespace six_digit_numbers_with_zero_l40_40882

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40882


namespace students_in_circle_l40_40672

theorem students_in_circle (n : ℕ) (h1 : n > 6) (h2 : n > 16) (h3 : n / 2 = 10) : n + 2 = 22 := by
  sorry

end students_in_circle_l40_40672


namespace solve_problem_l40_40253

open Nat

theorem solve_problem (n : ℕ) 
  (h : ∀ (a : ℕ), Nat.coprime a n → a ≤ 1 + (Nat.sqrt n) → ∃ x : ℤ, a ≡ x^2 [MOD n]) : 
  n = 1 ∨ n = 2 ∨ n = 12 :=
by
  sorry

end solve_problem_l40_40253


namespace average_second_pair_l40_40118

theorem average_second_pair 
  (avg_six : ℝ) (avg_first_pair : ℝ) (avg_third_pair : ℝ) (avg_second_pair : ℝ) 
  (h1 : avg_six = 3.95) 
  (h2 : avg_first_pair = 4.2) 
  (h3 : avg_third_pair = 3.8000000000000007) : 
  avg_second_pair = 3.85 :=
by
  sorry

end average_second_pair_l40_40118


namespace selling_price_range_l40_40690

theorem selling_price_range
  (unit_purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_sales_volume : ℝ)
  (price_increase_effect : ℝ)
  (daily_profit_threshold : ℝ)
  (x : ℝ) :
  unit_purchase_price = 8 →
  initial_selling_price = 10 →
  initial_sales_volume = 100 →
  price_increase_effect = 10 →
  daily_profit_threshold = 320 →
  (initial_selling_price - unit_purchase_price) * initial_sales_volume > daily_profit_threshold →
  12 < x → x < 16 →
  (x - unit_purchase_price) * (initial_sales_volume - price_increase_effect * (x - initial_selling_price)) > daily_profit_threshold :=
sorry

end selling_price_range_l40_40690


namespace dice_composite_probability_l40_40003

theorem dice_composite_probability :
  let total_outcomes := (8:ℕ)^6
  let non_composite_outcomes := 1 + 4 * 6 
  let composite_probability := 1 - (non_composite_outcomes / total_outcomes) 
  composite_probability = 262119 / 262144 := by
  sorry

end dice_composite_probability_l40_40003


namespace abs_neg_two_l40_40499

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end abs_neg_two_l40_40499


namespace passing_percentage_correct_l40_40218

-- The given conditions
def marks_obtained : ℕ := 175
def marks_failed : ℕ := 89
def max_marks : ℕ := 800

-- The theorem to prove
theorem passing_percentage_correct :
  (
    (marks_obtained + marks_failed : ℕ) * 100 / max_marks
  ) = 33 :=
sorry

end passing_percentage_correct_l40_40218


namespace problem_l40_40483

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end problem_l40_40483


namespace find_A_d_minus_B_d_l40_40488

-- Definitions for the proof problem
variables {d A B : ℕ}
variables {ad bd : ℤ} -- Representing A_d and B_d in ℤ for arithmetic operations

-- Conditions
constants (base_check : d > 7)
constants (digit_check : A < d ∧ B < d)
constants (encoded_check : d^1 * A + d^0 * B + d^2 * A + d^1 * A = 1 * d^2 + 7 * d^1 + 2 * d^0)

-- The theorem to prove
theorem find_A_d_minus_B_d : A - B = 5 :=
good sorry

end find_A_d_minus_B_d_l40_40488


namespace volume_of_inscribed_sphere_l40_40710

-- Define the edge length of the cube.
def edge_length : ℝ := 8

-- Calculate the radius of the inscribed sphere.
def radius (cube_edge : ℝ) : ℝ := cube_edge / 2

-- Calculate the volume of the sphere.
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Proof statement
theorem volume_of_inscribed_sphere : 
  volume_of_sphere (radius edge_length) = (256 / 3) * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l40_40710


namespace min_value_f_solve_inequality_f_l40_40414

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- Proof Problem 1
theorem min_value_f : ∃ x : ℝ, f x = 3 :=
by { sorry }

-- Proof Problem 2
theorem solve_inequality_f : {x : ℝ | abs (f x - 6) ≤ 1} = 
    ({x : ℝ | -10/3 ≤ x ∧ x ≤ -8/3} ∪ 
    {x : ℝ | 0 ≤ x ∧ x ≤ 1} ∪ 
    {x : ℝ | 1 < x ∧ x ≤ 4/3}) :=
by { sorry }

end min_value_f_solve_inequality_f_l40_40414


namespace solve_problem_l40_40556

theorem solve_problem (nabla odot : ℕ) 
  (h1 : 0 < nabla) 
  (h2 : nabla < 20) 
  (h3 : 0 < odot) 
  (h4 : odot < 20) 
  (h5 : nabla ≠ odot) 
  (h6 : nabla * nabla * nabla = nabla) : 
  nabla * nabla = 64 :=
by
  sorry

end solve_problem_l40_40556


namespace total_weight_l40_40231

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

-- Define the sum of the arithmetic sequence
def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a₁ + arithmetic_sequence a₁ d n)

-- Define the total weight of the envelopes
def weight_envelopes (weight_each_envelope : ℝ) (number_envelopes : ℕ) : ℝ :=
  (number_envelopes : ℝ) * weight_each_envelope

-- The theorem to prove
theorem total_weight (n : ℕ) : n = 800 →
  let envelope_weight := 8.5,
      mail_initial := 2.0,
      mail_diff := 0.5 in
  (weight_envelopes envelope_weight n + sum_arithmetic_sequence mail_initial mail_diff n = 168200) :=
by
  intros hn h,
  sorry

end total_weight_l40_40231


namespace chocolate_pieces_count_l40_40446

def chocolate_square_area (side: ℕ) : ℕ :=
  side * side

def triangle_area (width height: ℕ) : ℚ :=
  (width * height) / 2

def number_of_triangles (A_square: ℕ) (A_triangle: ℚ) : ℕ :=
  (A_square.to_rat / A_triangle).floor.to_nat

theorem chocolate_pieces_count :
  let side := 10 in
  let width := 1 in
  let height := 3 in
  let A_square := chocolate_square_area side in
  let A_triangle := triangle_area width height in
  number_of_triangles A_square A_triangle = 66 :=
by
  sorry

end chocolate_pieces_count_l40_40446


namespace six_digit_numbers_with_zero_l40_40878

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40878


namespace zeros_in_decimal_representation_l40_40658

theorem zeros_in_decimal_representation :
  let d := (7 : ℚ) / 8000 in
  ∃ n : ℕ, d = 875 / (10 ^ 6) ∧ n = 3 :=
by
  let d := (7 : ℚ) / 8000
  use 3
  constructor
  . norm_num [d]
  . rfl

end zeros_in_decimal_representation_l40_40658


namespace point_on_number_line_left_of_neg2_l40_40702

theorem point_on_number_line_left_of_neg2 :
  ∀ (x : ℤ), (x = -2 - 5) → x = -7 :=
by
  intros x h
  rw [h]
  norm_num
  sorry

end point_on_number_line_left_of_neg2_l40_40702


namespace expected_value_winnings_l40_40216

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def lose_amount_tails : ℚ := -4

theorem expected_value_winnings : 
  probability_heads * win_amount_heads + probability_tails * lose_amount_tails = -2 / 5 := 
by 
  sorry

end expected_value_winnings_l40_40216


namespace largest_sum_36_l40_40616

theorem largest_sum_36 : ∃ n : ℕ, ∃ a : ℕ, (n * a + (n * (n - 1)) / 2 = 36) ∧ ∀ m : ℕ, (m * a + (m * (m - 1)) / 2 = 36) → m ≤ 8 :=
by
  sorry

end largest_sum_36_l40_40616


namespace coprime_and_multiple_exists_l40_40469

theorem coprime_and_multiple_exists (n : ℕ) :
  ∀ (A : Finset ℕ), A.card = n + 1 → A ⊆ Finset.range (2 * n + 1) →
    (∃ x y ∈ A, Nat.gcd x y = 1) ∧ (∃ x y ∈ A, x ≠ y ∧ (x % y = 0 ∨ y % x = 0)) :=
begin
  sorry
end

end coprime_and_multiple_exists_l40_40469


namespace shaded_area_l40_40388

theorem shaded_area (PR PV PQ QR : ℝ) (hPR : PR = 20) (hPV : PV = 12) (hPQ_QR : PQ + QR = PR) :
  PR * PV - 1 / 2 * 12 * PR = 120 :=
by
  -- Definitions used earlier
  have h_area_rectangle : PR * PV = 240 := by
    rw [hPR, hPV]
    norm_num
  have h_half_total_unshaded : (1 / 2) * 12 * PR = 120 := by
    rw [hPR]
    norm_num
  rw [h_area_rectangle, h_half_total_unshaded]
  norm_num

end shaded_area_l40_40388


namespace angle_bisectors_form_triangle_l40_40303

theorem angle_bisectors_form_triangle
  (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : A > 0)
  (h3 : B > 0)
  (h4 : C > 0)
  (h5 : A < π / 2)
  (h6 : B < π / 2)
  (h7 : C < π / 2)
  (h8 : ∀ a₁ b₁ c₁ : ℝ, a₁ + b₁ + c₁ = π)
  (h9 : ∀ a₂ b₂ c₂ : ℝ, a₂ + b₂ + c₂ = π)
  (h10 : ∃ lA lB lC, (lA + lB > lC) ∧ (lA + lC > lB) ∧ (lB + lC > lA))
  : ∃ lA lB lC, (lA + lB > lC) ∧ (lA + lC > lB) ∧ (lB + lC > lA) :=
sorry

end angle_bisectors_form_triangle_l40_40303


namespace find_m_from_root_l40_40322

theorem find_m_from_root (m : ℝ) : (x : ℝ) = 1 → x^2 + m * x + 2 = 0 → m = -3 :=
by
  sorry

end find_m_from_root_l40_40322


namespace coffee_break_l40_40592

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end coffee_break_l40_40592


namespace six_digit_numbers_with_zero_l40_40963

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l40_40963


namespace statistician_made_error_l40_40017

theorem statistician_made_error (n : ℕ) (k : ℕ → ℕ) (h1 : n = 2000) (h2 : ∀ i, i < n → ∃ j, k i = 2^j) 
    (h3 : (∑ i in Finset.range n, 4 ^ (k i)) = 100000) : False := 
by 
  have h_mod3 : (∑ i in Finset.range n, 4 ^ (k i)) % 3 = 2000 % 3 := 
    by 
      have h4k_mod3 : ∀ i, 4 ^ (k i) % 3 = 1 := by intros i; exact Nat.pow_mod 4 (k i) 3
      have : (∑ i in Finset.range n, 4 ^ (k i)) % 3 = ∑ i in Finset.range n, (4 ^ (k i) % 3) := 
        Finset.sum_congr rfl (λ i hm, h4k_mod3 i)
      rw [this, Finset.sum_const, Finset.card_range, mul_one]
  show False
by 
  have : 100000 % 3 = 1 := by norm_num
  rw [h3] at this
  rw [h_mod3] at this
  rw [mod_eq_of_lt, mod_eq_of_lt] at this
  contradiction
  all_goals { try { norm_num } }
  sorry -- This statement assumes the needed properties and omits the actual proof.

end statistician_made_error_l40_40017


namespace product_real_parts_roots_l40_40415

theorem product_real_parts_roots :
  ∀ (i : ℂ) (z : ℂ),
    i^2 = -1 →
    (z^2 - z = 5 - 5 * i) →
    (let z1 := (1 + Complex.sqrt (21 - 20 * i)) / 2,
         z2 := (1 - Complex.sqrt (21 - 20 * i)) / 2 in
     (z1.re * z2.re) = -6) :=
by
intros i z hi hz
let z1 := (1 + Complex.sqrt (21 - 20 * i)) / 2 
let z2 := (1 - Complex.sqrt (21 - 20 * i)) / 2
sorry

end product_real_parts_roots_l40_40415


namespace intersection_M_N_l40_40678

open Set

def M := {x : ℤ | x^2 ≥ 4}
def N := {-3, 0, 1, 3, 4}

theorem intersection_M_N : M ∩ N = {-3, 3, 4} :=
by 
  sorry

end intersection_M_N_l40_40678


namespace composite_function_evaluation_l40_40348

def f (x : ℝ) : ℝ := x^2 + 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x + 4

theorem composite_function_evaluation : f (g (-2)) = 1 := by
  have : f (g (-2)) = f (3 * -2 + 4) := by rfl
  calc
    f (3 * -2 + 4)
      = f (-2) : by rfl
    _ = (-2)^2 + 2 * (-2) + 1 : by rfl
    _ = 4 - 4 + 1 : by rfl
    _ = 1 : by rfl

end composite_function_evaluation_l40_40348


namespace find_p_l40_40332

theorem find_p (p q : ℚ) (h1 : 3 * p + 4 * q = 15) (h2 : 4 * p + 3 * q = 18) : p = 27 / 7 :=
by
  sorry

end find_p_l40_40332


namespace inequality_solution_l40_40554

theorem inequality_solution (x : ℝ) :
  (x * (x + 2) > x * (3 - x) + 1) ↔ (x < -1/2 ∨ x > 1) :=
by sorry

end inequality_solution_l40_40554


namespace expression_is_integer_expression_modulo_3_l40_40458

theorem expression_is_integer (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℤ), (n^3 + (3/2) * n^2 + (1/2) * n - 1) = k := 
sorry

theorem expression_modulo_3 (n : ℕ) (hn : n > 0) : 
  (n^3 + (3/2) * n^2 + (1/2) * n - 1) % 3 = 2 :=
sorry

end expression_is_integer_expression_modulo_3_l40_40458


namespace area_circle_l40_40535

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l40_40535


namespace log_implication_l40_40824

theorem log_implication (m n : ℝ) (h₁ : log m 2 < log n 2) (h₂ : log n 2 < 0) : n < m ∧ m < 1 :=
by
  sorry

end log_implication_l40_40824


namespace circle_area_polar_eq_l40_40539

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l40_40539


namespace problem_statement_l40_40056

def f (x : ℤ) : ℤ := x^2 + 3
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem_statement : f (g 4) - g (f 4) = 129 := by
  sorry

end problem_statement_l40_40056


namespace find_incorrect_statement_l40_40724

variable (q n x y : ℚ)

theorem find_incorrect_statement :
  (∀ q, q < -1 → q < 1/q) ∧
  (∀ n, n ≥ 0 → -n ≥ n) ∧
  (∀ x, x < 0 → x^3 < x) ∧
  (∀ y, y < 0 → y^2 > y) →
  (∃ x, x < 0 ∧ ¬ (x^3 < x)) :=
by
  sorry

end find_incorrect_statement_l40_40724


namespace find_A_d_minus_B_d_l40_40481

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end find_A_d_minus_B_d_l40_40481


namespace a_plus_k_is_7_l40_40726

noncomputable def proof_problem : Prop :=
  let foci1 := (1 : ℝ, 1 : ℝ)
  let foci2 := (1 : ℝ, 3 : ℝ)
  let passing_point := (-4 : ℝ, 2 : ℝ)
  let h := 1
  let k := 2
  let a := 5
  let b := Real.sqrt 26
  in
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧ ((passing_point.1 - foci1.1)^2 + (passing_point.2 - foci1.2)^2).sqrt +
                   ((passing_point.1 - foci2.1)^2 + (passing_point.2 - foci2.2)^2).sqrt = 2 * b ∧
                   h = 1 ∧ k = 2 ∧ a = 5 ∧ b = Real.sqrt 26 ∧ a + k = 7

theorem a_plus_k_is_7 : proof_problem := by
  sorry

end a_plus_k_is_7_l40_40726


namespace incorrect_statement_D_l40_40342

open Real

def inverse_proportional (k : ℝ) (x : ℝ) : ℝ := -k / x

theorem incorrect_statement_D :
  let k := 4 in
  ¬ (∀ x : ℝ, x > 0 → inverse_proportional k x = -k / x → x < y → inverse_proportional k x < inverse_proportional k y) ∧
  ¬ (∀ x : ℝ, x < 0 → inverse_proportional k x = -k / x → x < y → inverse_proportional k x < inverse_proportional k y) :=
by
  sorry

end incorrect_statement_D_l40_40342


namespace delegates_with_at_least_one_female_l40_40302

open Finset

def chooseDelegates (m f : ℕ) : ℕ :=
  (choose f 1 * choose m 2) + (choose f 2 * choose m 1) + (choose f 3)

theorem delegates_with_at_least_one_female :
  ∀ (m f : ℕ), m = 4 → f = 3 → chooseDelegates m f = 31 :=
by
  intros m f hm hf
  rw [hm, hf]
  simp [chooseDelegates, choose]
  sorry

end delegates_with_at_least_one_female_l40_40302


namespace not_symmetric_about_point_l40_40335

def f (x : ℝ) : ℝ :=
  sin (x - (Real.pi / 6)) * cos (x - (Real.pi / 6))

theorem not_symmetric_about_point : ¬(∃ y, f(-Real.pi / 6) = y ∧ f(x) = y - ∃ x, f(x) = 0 ∧ x = -Real.pi / 6) :=
sorry

end not_symmetric_about_point_l40_40335


namespace max_term_in_binomial_expansion_l40_40330

theorem max_term_in_binomial_expansion :
  (∃ n : ℕ, ∑ k in finset.range (n + 1), (nat.choose n k) = 64) →
  ∀ n : ℕ, n = 6 → 
    let term_pos := (n + 1) / 2 + 1 in
    term_pos = 5 :=
by
  intros h n hn
  sorry

end max_term_in_binomial_expansion_l40_40330


namespace original_faculty_is_287_l40_40456

noncomputable def original_faculty (F : ℝ) : Prop :=
  (F * 0.85 * 0.80 = 195)

theorem original_faculty_is_287 : ∃ F : ℝ, original_faculty F ∧ F = 287 := 
by 
  use 287
  sorry

end original_faculty_is_287_l40_40456


namespace mark_parking_tickets_eq_l40_40733

def total_tickets : ℕ := 24
def sarah_speeding_tickets : ℕ := 6
def mark_speeding_tickets : ℕ := 6
def sarah_parking_tickets (S : ℕ) := S
def mark_parking_tickets (S : ℕ) := 2 * S
def total_traffic_tickets (S : ℕ) := S + 2 * S + sarah_speeding_tickets + mark_speeding_tickets

theorem mark_parking_tickets_eq (S : ℕ) (h1 : total_traffic_tickets S = total_tickets)
  (h2 : sarah_speeding_tickets = 6) (h3 : mark_speeding_tickets = 6) :
  mark_parking_tickets S = 8 :=
sorry

end mark_parking_tickets_eq_l40_40733


namespace general_formula_an_sum_formula_sn_l40_40138

namespace Sequences

def arithmeticSequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def geometricSequence (b_n : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, b_n (n + 1) = b_n n * q

variable (a_n b_n : ℕ → ℤ)

-- Given conditions
axiom a2_eq_b1 : a_n 2 = b_n 1
axiom a5_eq_b2 : a_n 5 = b_n 2
axiom arith_seq_an : arithmeticSequence a_n
axiom sum_bn_sn : ∀ n, (1 + ∑ i in Finset.range n, b_n i) = b_n n - 1


theorem general_formula_an :
  ∀ n : ℕ, a_n n = 2 * n - 6 :=
sorry

theorem sum_formula_sn :
  ∀ n : ℕ, (1 + ∑ i in Finset.range n, b_n i) = (-2)^n - 1 :=
sorry

end Sequences

end general_formula_an_sum_formula_sn_l40_40138


namespace quotient_of_division_l40_40514

theorem quotient_of_division (L : ℕ) (S : ℕ) (Q : ℕ) (h1 : L = 1631) (h2 : L - S = 1365) (h3 : L = S * Q + 35) :
  Q = 6 :=
by
  sorry

end quotient_of_division_l40_40514


namespace problem_inequality_l40_40061

theorem problem_inequality 
  (n : ℕ) 
  (h1 : n > 0) 
  (x : fin n → ℝ) 
  (h2 : ∀ i, 0 < x i) : 
  (finset.univ.sum (λ i, x i / (x (⟨(n - 1 - i : ℕ) % n, nat.mod_lt (n - 1 - i) n.pos⟩ : fin n)))) ≥ n := 
begin
  sorry
end

end problem_inequality_l40_40061


namespace smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40632

/-- 
  Problem statement: Prove that the smallest prime divisor of 3^19 + 11^13 is 2, 
  given the conditions:
   - 3^19 is odd
   - 11^13 is odd
   - The sum of two odd numbers is even
-/

theorem smallest_prime_divisor_of_3_pow_19_plus_11_pow_13 :
  ∃ p : ℕ, prime p ∧ p = 2 ∧ p ∣ (3^19 + 11^13) := 
by
  sorry

end smallest_prime_divisor_of_3_pow_19_plus_11_pow_13_l40_40632


namespace proportion_correct_l40_40313

theorem proportion_correct :
  ∃ (a b c d : ℝ), a = 3 ∧ b = a / 0.6 ∧ c * d = 12 ∧ c = d * 0.6 ∧ (a / b = c / d) :=
begin
  use [3, 5, 2.4, 4],
  split,
  { refl },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num },
end

end proportion_correct_l40_40313


namespace movie_ticket_final_price_l40_40294

noncomputable def final_ticket_price (initial_price : ℝ) : ℝ :=
  let price_year_1 := initial_price * 1.12
  let price_year_2 := price_year_1 * 0.95
  let price_year_3 := price_year_2 * 1.08
  let price_year_4 := price_year_3 * 0.96
  let price_year_5 := price_year_4 * 1.06
  let price_after_tax := price_year_5 * 1.07
  let final_price := price_after_tax * 0.90
  final_price

theorem movie_ticket_final_price :
  final_ticket_price 100 = 112.61 := by
  sorry

end movie_ticket_final_price_l40_40294


namespace min_value_of_f_range_of_a_l40_40341

def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

theorem min_value_of_f :
  ∃ x : ℝ, f x = 5 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (-3 : ℝ) (2 : ℝ), f x ≥ |x + a|) ↔ -2 ≤ a ∧ a ≤ 3 :=
sorry

end min_value_of_f_range_of_a_l40_40341


namespace calculate_expression_l40_40739

theorem calculate_expression :
  2⁻¹ + (3 - Real.pi)^0 + abs (2 * Real.sqrt 3 - Real.sqrt 2) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 12 = 3 / 2 :=
sorry

end calculate_expression_l40_40739


namespace six_digit_numbers_with_zero_l40_40887

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40887


namespace problem_solution_l40_40179

-- Definitions of x and y based on the given conditions
def x : ℕ := (Finset.range 21).sum + 210 -- Sum of integers from 10 to 30
def y : ℕ := 11 -- Number of even integers from 10 to 30

theorem problem_solution : x + y = 431 :=
by
  -- The proof is intentionally left as an exercise using sorry
  sorry

end problem_solution_l40_40179


namespace count_integers_satisfying_conditions_l40_40789

/-
  Prove that the number of positive integers x satisfying the following conditions is 2:
  1. x + 7 ≡ 40 (mod 50)
  2. x < 200
  3. x is divisible by 3
-/

theorem count_integers_satisfying_conditions : 
  {x : ℕ | x + 7 ≡ 40 [MOD 50] ∧ x < 200 ∧ x % 3 = 0}.card = 2 :=
by
  sorry

end count_integers_satisfying_conditions_l40_40789


namespace smallest_prime_divisor_of_n_is_2_l40_40636

def a := 3^19
def b := 11^13
def n := a + b

theorem smallest_prime_divisor_of_n_is_2 : nat.prime_divisors n = {2} :=
by
  -- The proof is not needed as per instructions.
  -- We'll add a placeholder sorry to complete the statement.
  sorry

end smallest_prime_divisor_of_n_is_2_l40_40636


namespace angle_B_eq_pi_div_3_sin_A_value_l40_40817

-- Given the conditions
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h_triangle : a^2 + c^2 - b^2 = ac)
variable (h_c_eq_3a : c = 3 * a)

-- Prove the first part: B = π / 3
theorem angle_B_eq_pi_div_3 (a b c : ℝ) (A B C : ℝ) (h_triangle : a^2 + c^2 - b^2 = ac) : 
  B = π / 3 := 
by
  sorry

-- Prove the second part: sin A = sqrt 21 / 14
theorem sin_A_value (a b c : ℝ) (A B C : ℝ) (h_c_eq_3a : c = 3 * a) : 
  b = sqrt 7 * a → a^2 + (3*a)^2 - b^2 = 3*a*a → sin A = sqrt 21 / 14 := 
by
  sorry

end angle_B_eq_pi_div_3_sin_A_value_l40_40817


namespace min_y_value_l40_40048

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16 * x + 50 * y + 64) : y ≥ 0 :=
sorry

end min_y_value_l40_40048


namespace expand_expression_l40_40278

theorem expand_expression : 
  (5 * x^2 + 2 * x - 3) * (3 * x^3 - x^2) = 15 * x^5 + x^4 - 11 * x^3 + 3 * x^2 := 
by
  sorry

end expand_expression_l40_40278


namespace pages_read_by_same_person_l40_40237

theorem pages_read_by_same_person :
  let pages := finset.range 20 in
  let beth_seq := pages.filter (λ n, n % 3 = 0) in
  let carolyn_seq := pages.filter (λ n, n % 3 = 1) in
  let george_seq := pages.filter (λ n, n % 3 = 2) in

  let carolyn_absent_seq := finset.filter (λ n, n % 2 = 0) pages in
  let george_absent_seq := finset.filter (λ n, n % 2 = 1) pages in

  (beth_seq \ carolyn_absent_seq).card + 
  (george_seq \ george_absent_seq).card + 
  (george_seq \ carolyn_absent_seq).card = 6 :=
by sorry

end pages_read_by_same_person_l40_40237


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40922

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40922


namespace cone_volume_l40_40559

-- Define the condition
def cylinder_volume : ℝ := 30

-- Define the statement that needs to be proven
theorem cone_volume (h_cylinder_volume : cylinder_volume = 30) : cylinder_volume / 3 = 10 := 
by 
  -- Proof omitted
  sorry

end cone_volume_l40_40559


namespace six_digit_numbers_with_zero_l40_40995

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l40_40995


namespace abs_fraction_ineq_solution_l40_40473

theorem abs_fraction_ineq_solution (x : ℝ) :
  (| (3 * x + 2) / (x + 1) | > 3) ↔ (x ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo (-5 / 6) (-1)) :=
sorry

end abs_fraction_ineq_solution_l40_40473


namespace mean_and_std_dev_of_normal_pdf_l40_40134

noncomputable def normal_pdf (x : ℝ) : ℝ :=
  (1 / (Real.sqrt (8 * Real.pi))) * Real.exp (-(x^2) / 8)

theorem mean_and_std_dev_of_normal_pdf :
  (∀ x, normal_pdf x = (1 / (Real.sqrt (8 * Real.pi))) * Real.exp (-(x^2) / 8)) →
  ∃ μ θ, μ = 0 ∧ θ = 2 :=
by
  intros h
  use 0, 2
  split
  sorry
  sorry

end mean_and_std_dev_of_normal_pdf_l40_40134


namespace initial_friends_l40_40475

theorem initial_friends (n : ℕ) (h1 : 120 / (n - 4) = 120 / n + 8) : n = 10 := 
by
  sorry

end initial_friends_l40_40475


namespace unit_vector_direction_l40_40816

def pointA : ℝ × ℝ := (1, 1)
def pointB : ℝ × ℝ := (-2, 5)
def vectorAB : ℝ × ℝ := (pointB.1 - pointA.1, pointB.2 - pointA.2)
def magAB : ℝ := real.sqrt ((vectorAB.1)^2 + (vectorAB.2)^2)
def unitVectorAB : ℝ × ℝ := (vectorAB.1 / magAB, vectorAB.2 / magAB)

theorem unit_vector_direction : unitVectorAB = (-3/5, 4/5) := by
  sorry

end unit_vector_direction_l40_40816


namespace I_1_I_2_lim_I_n_l40_40404

noncomputable def I (n : ℕ) : ℝ :=
  ∫ x in 0..Real.sqrt 3, 1 / (1 + x^n)

theorem I_1 : I 1 = Real.log (1 + Real.sqrt 3) :=
  sorry

theorem I_2 : I 2 = Real.pi / 3 :=
  sorry

theorem lim_I_n : filter.tendsto I filter.at_top (nhds 1) :=
  sorry

end I_1_I_2_lim_I_n_l40_40404


namespace g_symmetric_about_y_eq_x_plus_1_l40_40763

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x - 2) / (x - 1)

-- Define the function g(x) as g(x) = f(x + a)
def g (x : ℝ) (a : ℝ) : ℝ := f (x + a)

-- The theorem stating the conditions and requirement
theorem g_symmetric_about_y_eq_x_plus_1 (a : ℝ) : (∀ x : ℝ, g (g (x + 1) a - 1) a = x) ↔ a = 4 :=
sorry

end g_symmetric_about_y_eq_x_plus_1_l40_40763


namespace total_frogs_in_both_ponds_l40_40567

noncomputable def total_frogs_combined : Nat :=
let frogs_in_pond_a : Nat := 32
let frogs_in_pond_b : Nat := frogs_in_pond_a / 2
frogs_in_pond_a + frogs_in_pond_b

theorem total_frogs_in_both_ponds :
  total_frogs_combined = 48 := by
  sorry

end total_frogs_in_both_ponds_l40_40567


namespace container_capacity_in_liters_l40_40040

-- Defining the conditions
def portions : Nat := 10
def portion_size_ml : Nat := 200

-- Statement to prove
theorem container_capacity_in_liters : (portions * portion_size_ml / 1000 = 2) :=
by 
  sorry

end container_capacity_in_liters_l40_40040


namespace select_team_with_smaller_variance_l40_40689

theorem select_team_with_smaller_variance 
    (variance_A variance_B : ℝ)
    (hA : variance_A = 1.5)
    (hB : variance_B = 2.8)
    : variance_A < variance_B → "Team A" = "Team A" :=
by
  intros h
  sorry

end select_team_with_smaller_variance_l40_40689


namespace length_MD_l40_40476

-- Define the conditions
variables (A B C D M E F : ℝ) -- Points in Euclidean space R³
variables (length_AB : ℝ) (length_CD : ℝ) (midpoint_M_arc_AB : Prop)
variables (square_ABCD : Prop)
variables (coplanar_square_semicircle : Prop)

-- Specify the known values
def AB_length : Prop := length_AB = 8
def radius_semicircle : Prop := (∃ E, midpoint E A B ∧ dist A E = 4 ∧ dist B E = 4)
def midpoint_M_properties : Prop := (∃ E, midpoint M A B ∧ dist M E = 4)
def midpoint_E_properties : Prop := ∃ E, midpoint E A B 
def midpoint_F_properties : Prop := (∃ F, midpoint F C D ∧ dist C F = 4 ∧ dist D F = 4)
def MF_equation : Prop := ∃ MF : ℝ, dist M F = dist E M + 8
def right_triangle_MFD : Prop := ∃ (MF FD : ℝ), MF = 12 ∧ FD = 4 
def Pythagorean_theorem : Prop := ∃ (MD MF FD : ℝ), MD^2 = MF^2 + FD^2

-- Main theorem to be proved
theorem length_MD : AB_length → radius_semicircle → midpoint_M_properties → midpoint_E_properties → midpoint_F_properties → MF_equation → right_triangle_MFD → Pythagorean_theorem → ∃ MD : ℝ, MD = 4 * real.sqrt 10 :=
begin
  sorry
end

end length_MD_l40_40476


namespace knitting_time_total_l40_40432

-- Define knitting times for each item
def hat_knitting_time : ℕ := 2
def scarf_knitting_time : ℕ := 3
def mitten_knitting_time : ℕ := 1
def sock_knitting_time : ℕ := 3 / 2
def sweater_knitting_time : ℕ := 6

-- Define the number of grandchildren
def grandchildren_count : ℕ := 3

-- Total knitting time calculation
theorem knitting_time_total : 
  hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time = 16 ∧ 
  (hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time) * grandchildren_count = 48 :=
by 
  sorry

end knitting_time_total_l40_40432


namespace problem_1_interval_max_and_set_of_x_problem_2_range_of_a_l40_40841

noncomputable def f (x : ℝ) : ℝ := 2 * cos x ^ 2 + sin (2 * x - (π / 6))

theorem problem_1_interval_max_and_set_of_x :
  (∀ k : ℤ, ∃ interval : set ℝ,
    interval = { x : ℝ | (k : ℝ) * π - π / 3 ≤ x ∧ x ≤ (k : ℝ) * π + π / 6 }) ∧
  (∀ x : ℝ, f x ≤ 2) ∧
  (∀ x : ℝ, f x = 2 ↔ ∃ k : ℤ, x = (k : ℝ) * π + π / 6) :=
sorry

noncomputable def f_triangle (A : ℝ) : ℝ := sin (2 * A + π / 6) + 1

theorem problem_2_range_of_a (A a b c : ℝ) (hA : 0 < A ∧ A < π)
  (h1 : f_triangle A = 1.5) (h2 : b + c = 2) :
  1 ≤ a ∧ a < 2 :=
sorry

end problem_1_interval_max_and_set_of_x_problem_2_range_of_a_l40_40841


namespace coffee_participants_l40_40603

noncomputable def participants_went_for_coffee (total participants : ℕ) (stay : ℕ) : Prop :=
  (0 < stay) ∧ (stay < 14) ∧ (stay % 2 = 0) ∧ ∃ k : ℕ, stay = 2 * k

theorem coffee_participants :
  ∀ (total participants : ℕ), total participants = 14 →
  (∀ participant, ((∃ k : ℕ, stay = 2 * k) → (0 < stay) ∧ (stay < 14))) →
  participants_went_for_coffee total_participants (14 - stay) =
  (stay = 12) ∨ (stay = 10) ∨ (stay = 8) ∨ (stay = 6) :=
by
  sorry

end coffee_participants_l40_40603


namespace smallest_prime_divisor_of_sum_l40_40643

theorem smallest_prime_divisor_of_sum (h1 : Nat.odd (Nat.pow 3 19))
                                      (h2 : Nat.odd (Nat.pow 11 13))
                                      (h3 : ∀ a b : Nat, Nat.odd a → Nat.odd b → Nat.even (a + b)) :
  Nat.minFac (Nat.pow 3 19 + Nat.pow 11 13) = 2 :=
by
  -- placeholder proof
  sorry

end smallest_prime_divisor_of_sum_l40_40643


namespace six_digit_numbers_with_zero_l40_40994

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l40_40994


namespace simplify_radical_subtraction_l40_40109

theorem simplify_radical_subtraction : 
  (Real.sqrt 18 - Real.sqrt 8) = Real.sqrt 2 := 
by
  sorry

end simplify_radical_subtraction_l40_40109


namespace find_p5_l40_40418

-- Definitions based on conditions from the problem
def p (x : ℝ) : ℝ :=
  x^4 - 10 * x^3 + 35 * x^2 - 50 * x + 18  -- this construction ensures it's a quartic monic polynomial satisfying provided conditions

-- The main theorem we want to prove
theorem find_p5 :
  p 1 = 3 ∧ p 2 = 7 ∧ p 3 = 13 ∧ p 4 = 21 → p 5 = 51 :=
by
  -- The proof will be inserted here later
  sorry

end find_p5_l40_40418


namespace circle_area_l40_40546

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l40_40546


namespace symmetric_point_origin_correct_l40_40187

def symmetric_point_origin (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-P.1, -P.2, -P.3)

theorem symmetric_point_origin_correct : symmetric_point_origin (1, 3, -5) = (-1, -3, 5) := by
  -- proof is omitted.
  sorry

end symmetric_point_origin_correct_l40_40187


namespace error_arrangements_l40_40256

noncomputable def countIncorrectArrangements : ℕ := 
  let totalArrangements := (5.choose 2) * 1 
  totalArrangements - 1

theorem error_arrangements :
  countIncorrectArrangements ("error".toList) = 19 :=
by
  sorry

end error_arrangements_l40_40256


namespace soccer_tournament_probability_l40_40778

theorem soccer_tournament_probability
  (H1 : ∃ (teams : Fin 8 → Prop), ∀ t : Fin 8, teams t)
  (H2 : ∀ t₁ t₂ : Fin 8, t₁ ≠ t₂ → ∃ (game : Prop), game)
  (H3 : ∀ (game : Prop), ¬(game ∧ ¬game))
  (H4 : ∀ t₁ t₂ : Fin 8, t₁ ≠ t₂ → (50% : ℝ))
  (H5 : ∀ (game₁ game₂ : Prop), game₁ ≠ game₂ → Prop)
  (H6 : ∃ (C D : Fin 8), ∀ g : Prop, g)
  : ((4292 : ℚ) / 8192 = 4292 / 8192) := 
begin
  sorry
end

end soccer_tournament_probability_l40_40778


namespace polynomial_roots_eval_l40_40064

theorem polynomial_roots_eval :
  (∀ a b c : ℝ, (Polynomial.X^3 - 15 * Polynomial.X^2 + 25 * Polynomial.X - 10).is_root a ∧
              (Polynomial.X^3 - 15 * Polynomial.X^2 + 25 * Polynomial.X - 10).is_root b ∧
              (Polynomial.X^3 - 15 * Polynomial.X^2 + 25 * Polynomial.X - 10).is_root c →
              (2 + a) * (2 + b) * (2 + c) = 128) :=
by
  sorry

end polynomial_roots_eval_l40_40064


namespace quadratic_root_m_value_l40_40319

theorem quadratic_root_m_value (m : ℝ) (x : ℝ) (h : x = 1) (hx : x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end quadratic_root_m_value_l40_40319


namespace tan_half_angle_l40_40323

-- Definition for the given angle in the third quadrant with a given sine value
def angle_in_third_quadrant_and_sin (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) : Prop :=
  True

-- The main theorem to prove the given condition implies the result
theorem tan_half_angle (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) :
  Real.tan (α / 2) = -4 / 3 :=
by
  sorry

end tan_half_angle_l40_40323


namespace find_A_find_area_case1_find_area_case2_non_existent_case3_l40_40394

-- Definition of the problem using Lean
variables {a b c : ℝ} {A B C : ℝ}

-- Condition: b = a (sin C + cos C)
def condition1 : Prop := b = a * (Real.sin C + Real.cos C)

-- Condition: A = π/4
def angle_A : Prop := A = π / 4

-- Condition: a = 2
def a_eq_2 : Prop := a = 2

-- Condition: B = π/3
def B_eq_pi_div_3 : Prop := B = π / 3

-- Condition: c = sqrt2 * b
def c_eq_sqrt2_b : Prop := c = Real.sqrt 2 * b

-- Find angle A
theorem find_A (h : condition1) : angle_A :=
  sorry

-- Find the area given a=2 and B=π/3
def area_case1 : ℝ := (3 + Real.sqrt 3) / 2

theorem find_area_case1 (h1 : a_eq_2) (h2 : B_eq_pi_div_3) : 
  area_case1 = 1/2 * 2 * (Real.sqrt 3 + 1) * Real.sin (π/3) :=
  sorry

-- Find the area given a=2 and c=sqrt2*b
def area_case2 : ℝ := 2

theorem find_area_case2 (h1 : a_eq_2) (h2 : c_eq_sqrt2_b) :
  area_case2 = 1/2 * 2 * 2 :=
  sorry

-- Prove the triangle does not exist for B=π/3 and c=sqrt2*b
theorem non_existent_case3 (h1 : B_eq_pi_div_3) (h2 : c_eq_sqrt2_b) : 
  ¬ ∃ (a b c A B C : ℝ), triangle_conditions :=
  sorry


end find_A_find_area_case1_find_area_case2_non_existent_case3_l40_40394


namespace probability_relatively_prime_three_elements_l40_40146

theorem probability_relatively_prime_three_elements :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let subsets := s.powerset.filter (λ t, t.card = 3)
  let coprime_subsets := subsets.filter (λ t, ∀ x ∈ t, ∀ y ∈ t, x ≠ y → (x.gcd y) = 1)
  (coprime_subsets.card : ℚ) / subsets.card = 45 / 56 := 
by
  sorry

end probability_relatively_prime_three_elements_l40_40146


namespace coffee_participants_l40_40602

noncomputable def participants_went_for_coffee (total participants : ℕ) (stay : ℕ) : Prop :=
  (0 < stay) ∧ (stay < 14) ∧ (stay % 2 = 0) ∧ ∃ k : ℕ, stay = 2 * k

theorem coffee_participants :
  ∀ (total participants : ℕ), total participants = 14 →
  (∀ participant, ((∃ k : ℕ, stay = 2 * k) → (0 < stay) ∧ (stay < 14))) →
  participants_went_for_coffee total_participants (14 - stay) =
  (stay = 12) ∨ (stay = 10) ∨ (stay = 8) ∨ (stay = 6) :=
by
  sorry

end coffee_participants_l40_40602


namespace sale_price_of_trouser_l40_40400

theorem sale_price_of_trouser : 
  ∀ (P : ℝ) (dp : ℝ), P = 100 ∧ dp = 0.5 → (P - (dp * P) = 50) :=
by
  intro P dp
  intro h
  cases h with h₁ h₂
  sorry

end sale_price_of_trouser_l40_40400


namespace original_faculty_size_l40_40455

theorem original_faculty_size (F : ℝ) (h1 : F * 0.85 * 0.80 = 195) : F = 287 :=
by
  sorry

end original_faculty_size_l40_40455


namespace inverse_f_of_7_l40_40334

def f (x : ℝ) : ℝ := 2 * x^2 + 3

theorem inverse_f_of_7:
  ∀ y : ℝ, f (7) = y ↔ y = 101 :=
by
  sorry

end inverse_f_of_7_l40_40334


namespace sin_sum_to_product_l40_40784

-- Define the problem conditions
variable (x : ℝ)

-- State the problem and answer in Lean 4
theorem sin_sum_to_product :
  sin (5 * x) + sin (7 * x) = 2 * sin (6 * x) * cos (x) :=
sorry

end sin_sum_to_product_l40_40784


namespace volume_range_l40_40811

theorem volume_range (a b c : ℝ) (h1 : a + b + c = 9)
  (h2 : a * b + b * c + a * c = 24) : 16 ≤ a * b * c ∧ a * b * c ≤ 20 :=
by {
  -- Proof would go here
  sorry
}

end volume_range_l40_40811


namespace intercepts_of_line_l40_40698

theorem intercepts_of_line {x y : ℝ} 
  (h : ∀ x y, y - 2 = -3 * (x + 5)) :
  let x_intercept := -13 / 3 in
  let y_intercept := -13 in
  (x_intercept + y_intercept = -52 / 3) ∧ 
  (x_intercept * y_intercept = 169 / 3) :=
by
  sorry

end intercepts_of_line_l40_40698


namespace find_w_l40_40324

theorem find_w (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : (real.sqrt(x) / real.sqrt(y)) - (real.sqrt(y) / real.sqrt(x)) = 7 / 12)
  (h2 : x - y = 7) :
  x + y = 25 := by
  sorry

end find_w_l40_40324


namespace rationalized_sum_l40_40095

theorem rationalized_sum : 
  ∃ A B C D : ℤ, 
    (D > 0) ∧ 
    (∀ p : ℕ, prime p → p * p ∣ B → false) ∧ 
    (Int.gcd A (Int.gcd C D) = 1) ∧ 
    (A * (Int.ofNat B).sqrt + C) / D = 7 * (3 - (Int.ofNat B).sqrt) / (3 + (Int.ofNat B).sqrt) ∧ 
    A + B + C + D = 23 := sorry

end rationalized_sum_l40_40095


namespace total_bill_l40_40225

-- Definitions from conditions
def num_people : ℕ := 3
def amount_per_person : ℕ := 45

-- Mathematical proof problem statement
theorem total_bill : num_people * amount_per_person = 135 := by
  sorry

end total_bill_l40_40225


namespace average_hit_targets_value_average_hit_targets_ge_half_l40_40269

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end average_hit_targets_value_average_hit_targets_ge_half_l40_40269


namespace Daisy_toys_l40_40766

variable (Monday_toys Tuesday_leftover Tuesday_new Wednesday_lost Wednesday_new : ℕ)
variables (Tuesday_percent_loss Wednesday_percent_loss : ℝ)

axiom Monday_condition : Monday_toys = 5
axiom Tuesday_condition : Tuesday_leftover = (1 - Tuesday_percent_loss) * Monday_toys.to_real
axiom Tuesday_loss_condition : Tuesday_percent_loss = 0.40
axiom Tuesday_new_condition : Tuesday_new = 3
axiom Tuesday_total_condition : Tuesday_leftover.to_nat + Tuesday_new = 6
axiom Wednesday_loss_condition : Wednesday_lost = (1 - Wednesday_percent_loss) * (Tuesday_leftover.to_nat + Tuesday_new).to_real
axiom Wednesday_percent_loss : 0.50
axiom Wednesday_new_condition : Wednesday_new = 5
axiom Wednesday_total_condition : (Tuesday_leftover.to_nat + Tuesday_new - Wednesday_lost.to_nat) + Wednesday_new = 8
axiom Total_loss_condition : (Monday_toys - Tuesday_leftover.to_nat) + Wednesday_lost.to_nat = 5

theorem Daisy_toys : (Wednesday_total_condition + Total_loss_condition) = 13 := by
  sorry

end Daisy_toys_l40_40766


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40916

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40916


namespace remainder_when_dividing_by_y_minus_4_l40_40618

def g (y : ℤ) : ℤ := y^5 - 8 * y^4 + 12 * y^3 + 25 * y^2 - 40 * y + 24

theorem remainder_when_dividing_by_y_minus_4 : g 4 = 8 :=
by
  sorry

end remainder_when_dividing_by_y_minus_4_l40_40618


namespace six_digit_numbers_with_at_least_one_zero_l40_40861

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40861


namespace market_value_of_stock_l40_40664

theorem market_value_of_stock (dividend_rate : ℝ) (yield_rate : ℝ) (face_value : ℝ) :
  dividend_rate = 0.12 → yield_rate = 0.08 → face_value = 100 → (dividend_rate * face_value / yield_rate * 100) = 150 :=
by
  intros h1 h2 h3
  sorry

end market_value_of_stock_l40_40664


namespace constant_term_in_expansion_l40_40510

theorem constant_term_in_expansion :
  let expr := (x^2 - (1/x))^6 in
  ∃ (T : ℕ → ℕ), T 4 = 15
by
  sorry

end constant_term_in_expansion_l40_40510


namespace circle_area_polar_eq_l40_40541

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l40_40541


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40914

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40914


namespace calculate_interest_time_l40_40139

def SI : ℝ := 10.92
def P : ℝ := 26
def R : ℝ := 7 / 100  -- Convert rate from paise to rupees
def T : ℝ := 6

theorem calculate_interest_time :
  SI = P * R * T := by
  sorry

end calculate_interest_time_l40_40139


namespace distribution_of_tickets_l40_40774

-- Define the number of total people and the number of tickets
def n : ℕ := 10
def k : ℕ := 3

-- Define the permutation function P(n, k)
def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Main theorem statement
theorem distribution_of_tickets : P n k = 720 := by
  unfold P
  sorry

end distribution_of_tickets_l40_40774


namespace sale_in_first_month_l40_40208

theorem sale_in_first_month (sale1 sale2 sale3 sale4 sale5 : ℕ) 
  (h1 : sale1 = 5660) (h2 : sale2 = 6200) (h3 : sale3 = 6350) (h4 : sale4 = 6500) 
  (h_avg : (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = 6000) : 
  sale5 = 5290 := 
by
  sorry

end sale_in_first_month_l40_40208


namespace cannot_determine_log_21_l40_40350
-- Import the necessary library

-- Define the given logarithmic values
def log_5 : ℝ := 0.6990
def log_7 : ℝ := 0.8451

-- Define the main theorem statement
theorem cannot_determine_log_21 :
  ¬ ∃ (log_3 : ℝ), log 21 = log_3 + log_7 :=
sorry

end cannot_determine_log_21_l40_40350


namespace six_digit_numbers_with_zero_l40_40874

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40874


namespace smallest_prime_divisor_of_sum_l40_40620

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end smallest_prime_divisor_of_sum_l40_40620


namespace lg_vs_sqrt10_l40_40051

theorem lg_vs_sqrt10 (M : ℝ) (hM : 0 < M) : 
  ¬ (∀ M, lg M ≥ sqrt10 M) ∧ ¬ (∀ M, lg M ≤ sqrt10 M) ∧ ¬ (∀ M, lg M ≠ sqrt10 M) := 
sorry

end lg_vs_sqrt10_l40_40051


namespace bisection_solvable_l40_40169

open Real

theorem bisection_solvable :
  (∃ a b, a < b ∧ (ln a + a < 0) ∧ (ln b + b > 0) ∧ (∀ x, a < x ∧ x < b → continuous_at (λ x, ln x + x) x)) ∧
  (∃ c d, c < d ∧ (exp c - 3 * c > 0) ∧ (exp d - 3 * d < 0) ∧ (∀ x, c < x ∧ x < d → continuous_at (λ x, exp x - 3 * x) x)) ∧
  (∃ e f, e < f ∧ (e^3 - 3 * e + 1 > 0) ∧ (f^3 - 3 * f + 1 < 0) ∧ (∀ x, e < x ∧ x < f → continuous_at (λ x, x^3 - 3 * x + 1) x)) ∧
  ¬(∃ g h, g < h ∧ (∀ x, g ≤ x ∧ x ≤ h → 4 * x^2 - 4 * sqrt 5 * x + 5 ≥ 0) ∧ (∀ x, g < x ∧ x < h → continuous_at (λ x, 4 * x^2 - 4 * sqrt 5 * x + 5) x)) :=
begin
  sorry
end

end bisection_solvable_l40_40169


namespace find_dot_product_find_angle_find_magnitude_sum_l40_40307

variables {V : Type*} [inner_product_space ℝ V]

-- Define the vectors a and b
variables (a b : V)

-- Given condition magnitudes and dot product expression
def magnitude_a := ∥a∥ = 4
def magnitude_b := ∥b∥ = 3
def dot_product_expression := (2 • a - 3 • b) ⬝ (2 • a + b) = 61

-- Problems to solve
theorem find_dot_product (h1 : magnitude_a) (h2 : magnitude_b) (h3 : dot_product_expression) :
  a ⬝ b = -6 := sorry

theorem find_angle (h1 : magnitude_a) (h2 : magnitude_b) (h4 : a ⬝ b = -6) :
  real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥)) = 2 * real.pi / 3 := sorry

theorem find_magnitude_sum (h1 : magnitude_a) (h2 : magnitude_b) (h4 : a ⬝ b = -6) :
  ∥a + b∥ = real.sqrt 13 := sorry

end find_dot_product_find_angle_find_magnitude_sum_l40_40307


namespace bottles_remaining_after_2_days_l40_40202

theorem bottles_remaining_after_2_days :
  ∀ (initial_bottles : ℕ), initial_bottles = 24 →
  let bottles_first_day := initial_bottles - initial_bottles / 3 in
  let bottles_after_first_day := initial_bottles - bottles_first_day in
  let bottles_second_day := bottles_after_first_day / 2 in
  let bottles_remaining := bottles_after_first_day - bottles_second_day in
  bottles_remaining = 8 :=
by
  intros initial_bottles h_init
  let bottles_first_day := initial_bottles / 3
  let bottles_after_first_day := initial_bottles - bottles_first_day
  let bottles_second_day := bottles_after_first_day / 2
  let bottles_remaining := bottles_after_first_day - bottles_second_day
  have h_init_val : initial_bottles = 24 := h_init
  rw h_init_val at *
  calc
    bottles_first_day = 8 : by sorry
    bottles_after_first_day = 24 - 8 : by sorry
    _ = 16 : by sorry
    bottles_second_day = 16 / 2 : by sorry
    _ = 8 : by sorry
    bottles_remaining = 16 - 8 : by sorry
    _ = 8 : by sorry

end bottles_remaining_after_2_days_l40_40202


namespace area_of_circle_l40_40529

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l40_40529


namespace participants_coffee_l40_40598

theorem participants_coffee (n : ℕ) (h_n : n = 14) (h_not_all : 0 < n - 2 * k) (h_not_all2 : n - 2 * k < n)
(h_neighbors : ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧ (remaining i → leaving j) ∧ (leaving i → remaining j)) :
  ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 6 ∧ (n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12) :=
by {
  sorry
}

end participants_coffee_l40_40598


namespace square_area_inside_parabola_l40_40217

theorem square_area_inside_parabola : 
  let parabola := λ x : ℝ, x^2 - 6*x + 8
  let s := (1 : ℝ - ( - s - s + 0 ->> bigmulright + %
4* 1) − math.sqrt(x^2unesqrt(1/math.acmpi } :
∃ (s : ℝ), parabola(3 + s) = -s ∧ 0 < s ∧ s^2 = (3 - sqrt(5)) / 2.

sorry

end square_area_inside_parabola_l40_40217


namespace find_x_for_equation_l40_40670

def f (x : ℝ) : ℝ := 2 * x - 3

theorem find_x_for_equation : (2 * f x - 21 = f (x - 4)) ↔ (x = 8) :=
by
  sorry

end find_x_for_equation_l40_40670


namespace six_digit_numbers_with_zero_l40_40879

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l40_40879


namespace canoes_to_kayaks_ratio_l40_40612

theorem canoes_to_kayaks_ratio 
(RevenueCanoe : ℕ := 11) 
(RevenueKayak : ℕ := 16) 
(TotalRevenue : ℕ := 460) 
(ExtraCanoes : ℕ := 5) : 
{c k : ℕ // RevenueCanoe * c + RevenueKayak * k = TotalRevenue ∧ c = k + ExtraCanoes} → 
(c / k = 4 / 3) :=
begin
  sorry
end

end canoes_to_kayaks_ratio_l40_40612


namespace greatest_c_for_expression_domain_all_real_l40_40787

theorem greatest_c_for_expression_domain_all_real :
  ∃ c : ℤ, c ≤ 7 ∧ c ^ 2 < 60 ∧ ∀ d : ℤ, d > 7 → ¬ (d ^ 2 < 60) := sorry

end greatest_c_for_expression_domain_all_real_l40_40787


namespace eval_expression_l40_40781

theorem eval_expression :
  ((5^(1/3:ℝ)) * (3^(1/6:ℝ))) / ((5^(1/2:ℝ)) / (3^(1/3:ℝ))) = (5^(-1/6:ℝ)) * (3^(1/2:ℝ)) :=
by {
  sorry
}

end eval_expression_l40_40781


namespace skew_lines_definition_l40_40511

-- Definition for lines in three-dimensional space
structure Line3D (α : Type) [Add α] [Mul α] [Neg α] :=
  (point : α × α × α)
  (direction : α × α × α)

-- Definition of parallel lines
def are_parallel {α : Type} [Field α] (l₁ l₂ : Line3D α) : Prop :=
  ∃ k : α, l₂.direction = (k * l₁.direction.1, k * l₁.direction.2, k * l₁.direction.3)

-- Definition of intersecting lines
def do_intersect {α : Type} [Field α] (l₁ l₂ : Line3D α) : Prop :=
  ∃ t₁ t₂ : α, l₁.point = (l₂.point.1 + t₁ * l₂.direction.1, l₂.point.2 + t₁ * l₂.direction.2, l₂.point.3 + t₁ * l₂.direction.3)

-- Definition of skew lines
def are_skew {α : Type} [Field α] (l₁ l₂ : Line3D α) : Prop :=
  ¬ are_parallel l₁ l₂ ∧ ¬ do_intersect l₁ l₂

-- Theorem stating two lines that are neither parallel nor intersect are skew lines
theorem skew_lines_definition {α : Type} [Field α] (l₁ l₂ : Line3D α) :
  (¬ are_parallel l₁ l₂ ∧ ¬ do_intersect l₁ l₂) = are_skew l₁ l₂ := 
by
  sorry

end skew_lines_definition_l40_40511


namespace participants_coffee_l40_40595

theorem participants_coffee (n : ℕ) (h_n : n = 14) (h_not_all : 0 < n - 2 * k) (h_not_all2 : n - 2 * k < n)
(h_neighbors : ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧ (remaining i → leaving j) ∧ (leaving i → remaining j)) :
  ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 6 ∧ (n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12) :=
by {
  sorry
}

end participants_coffee_l40_40595


namespace central_ring_50th_element_l40_40006

lemma digits_of_three_times (x : ℕ) : nat.digits 10 (3 * x) > nat.digits 10 x → true := sorry

def is_central_ring_number (x : ℕ) : Prop := 
  nat.digits 10 (3 * x) > nat.digits 10 x

def central_ring_numbers_up_to (n : ℕ) : list ℕ := 
  list.filter is_central_ring_number (list.range (n + 1))

def nth_central_ring_number (k : ℕ) : option ℕ :=
  list.nth (central_ring_numbers_up_to 100) k

theorem central_ring_50th_element : nth_central_ring_number 49 = some 81 :=
by sorry

end central_ring_50th_element_l40_40006


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l40_40920

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l40_40920


namespace six_digit_numbers_with_at_least_one_zero_l40_40858

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40858


namespace factorization_x6_minus_5x4_plus_8x2_minus_4_l40_40280

theorem factorization_x6_minus_5x4_plus_8x2_minus_4 (x : ℝ) :
  x^6 - 5 * x^4 + 8 * x^2 - 4 = (x - 1) * (x + 1) * (x^2 - 2)^2 :=
sorry

end factorization_x6_minus_5x4_plus_8x2_minus_4_l40_40280


namespace distance_point_to_plane_l40_40325

-- Define the normal vector of the plane
def normal_vector : ℝ × ℝ × ℝ := (2, 2, 1)

-- Define the point A which lies on the plane
def point_A : ℝ × ℝ × ℝ := (-1, 3, 0)

-- Define the point P from which we want to find the distance to the plane
def point_P : ℝ × ℝ × ℝ := (2, 1, 3)

-- Prove that the distance from P to the plane defined by normal_vector and point_A is 5/3
theorem distance_point_to_plane : 
  let vector_PA := (point_A.1 - point_P.1, point_A.2 - point_P.2, point_A.3 - point_P.3) in
  let dot_product := vector_PA.1 * normal_vector.1 + vector_PA.2 * normal_vector.2 + vector_PA.3 * normal_vector.3 in
  let normal_magnitude := Math.sqrt (normal_vector.1^2 + normal_vector.2^2 + normal_vector.3^2) in
  abs (dot_product / normal_magnitude) = 5 / 3 :=
sorry

end distance_point_to_plane_l40_40325


namespace smallest_positive_period_max_value_monotonically_decreasing_intervals_l40_40337

noncomputable def f (x : ℝ) : ℝ := 
  cos (π / 3 + x) * cos (π / 3 - x) - sin x * cos x + 1 / 4

theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem max_value : ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = sqrt 2 / 2 :=
by sorry

theorem monotonically_decreasing_intervals : 
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 3 * π / 8 → f x1 ≥ f x2) ∧ 
  (∀ x1 x2 : ℝ, 7 * π / 8 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ π → f x1 ≥ f x2) :=
by sorry

end smallest_positive_period_max_value_monotonically_decreasing_intervals_l40_40337


namespace carnival_tickets_l40_40235

theorem carnival_tickets (total_tickets friends : ℕ) (equal_share : ℕ)
  (h1 : friends = 6)
  (h2 : total_tickets = 234)
  (h3 : total_tickets % friends = 0)
  (h4 : equal_share = total_tickets / friends) : 
  equal_share = 39 := 
by
  sorry

end carnival_tickets_l40_40235


namespace part_one_part_two_l40_40805

noncomputable def f (x a: ℝ) : ℝ := abs (x - 1) + abs (x + a)
noncomputable def g (a : ℝ) : ℝ := a^2 - a - 2

theorem part_one (x : ℝ) : f x 3 > g 3 + 2 ↔ x < -4 ∨ x > 2 := by
  sorry

theorem part_two (a : ℝ) :
  (∀ x : ℝ, -a ≤ x ∧ x ≤ 1 → f x a ≤ g a) ↔ a ≥ 3 := by
  sorry

end part_one_part_two_l40_40805


namespace six_digit_numbers_with_at_least_one_zero_l40_40854

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l40_40854


namespace impossible_8x8_square_l40_40460

theorem impossible_8x8_square :
  ¬ ∃ (grid : ℕ → ℕ → ℕ), 
    (∀ i j : ℕ, 1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8 → 1 ≤ grid i j ∧ grid i j ≤ 64) ∧
    (∀ i j : ℕ, 1 ≤ i ∧ i < 8 ∧ 1 ≤ j ∧ j < 8 →
      |(grid i j) * (grid (i + 1) (j + 1)) - (grid (i + 1) j) * (grid i (j + 1))| = 1) ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 64 → 
      ∃ i j : ℕ, 1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8 ∧ grid i j = k) :=
sorry

end impossible_8x8_square_l40_40460


namespace triangle_product_square_l40_40103

noncomputable def triangle_sides_product_square : ℝ := 
  let x : ℝ := some solution,
  let y : ℝ := 6 / x,
  let w : ℝ := 8 / x in
  (y * w)^2

theorem triangle_product_square (area_T1 area_T2 : ℝ) (x y w : ℝ)
  (h_area_T1 : area_T1 = 3)
  (h_area_T2 : area_T2 = 4)
  (h_T1 : x * y / 2 = area_T1)
  (h_T2 : x * w / 2 = area_T2)
  (h_y : y = 6 / x)
  (h_w : w = 8 / x) :
  triangle_sides_product_square = 64 := 
by
  sorry

end triangle_product_square_l40_40103
