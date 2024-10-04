import Mathlib
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Parity
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Logarithm
import Mathlib.Analysis.Calculus.Polynomial
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Compositions
import Mathlib.Combinatorics.SimpleGraph.Cycle
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.SinCos
import Mathlib.Data.Set.Basic
import Mathlib.Order.Floor
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.IntervalCases

namespace number_of_two_digit_primes_with_units_digit_three_l765_765216

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765216


namespace base_10_to_base_7_l765_765635

theorem base_10_to_base_7 (n : ℤ) (h : n = 947) : 
  (2 * 7^3 + 5 * 7^2 + 2 * 7^1 + 2 * 7^0 = n) := by
  have h_base : 947 = 2 * 7^3 + 5 * 7^2 + 2 * 7^1 + 2 * 7^0 := sorry
  exact h_base

end base_10_to_base_7_l765_765635


namespace time_per_mask_after_first_hour_l765_765019

-- Define the conditions as given in the problem
def rate_in_first_hour := 1 / 4 -- Manolo makes one face-mask every four minutes
def total_face_masks := 45 -- Manolo makes 45 face-masks in four hours
def first_hour_duration := 60 -- The duration of the first hour in minutes
def total_duration := 4 * 60 -- The total duration in minutes (4 hours)

-- Define the number of face-masks made in the first hour
def face_masks_first_hour := first_hour_duration / 4 -- 60 minutes / 4 minutes per face-mask = 15 face-masks

-- Calculate the number of face-masks made in the remaining time
def face_masks_remaining_hours := total_face_masks - face_masks_first_hour -- 45 - 15 = 30 face-masks

-- Define the duration of the remaining hours
def remaining_duration := total_duration - first_hour_duration -- 180 minutes (3 hours)

-- The target is to prove that the rate after the first hour is 6 minutes per face-mask
theorem time_per_mask_after_first_hour : remaining_duration / face_masks_remaining_hours = 6 := by
  sorry

end time_per_mask_after_first_hour_l765_765019


namespace solve_for_three_times_x_plus_ten_l765_765420

theorem solve_for_three_times_x_plus_ten (x : ℝ) (h_eq : 5 * x - 7 = 15 * x + 21) : 3 * (x + 10) = 21.6 := by
  sorry

end solve_for_three_times_x_plus_ten_l765_765420


namespace inequality_system_solution_l765_765576

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l765_765576


namespace farmer_planning_problem_l765_765681

theorem farmer_planning_problem
  (A : ℕ) (D : ℕ)
  (h1 : A = 120 * D)
  (h2 : ∀ t : ℕ, t = 85 * (D + 5) + 40)
  (h3 : 85 * (D + 5) + 40 = 120 * D) : 
  A = 1560 ∧ D = 13 := 
by
  sorry

end farmer_planning_problem_l765_765681


namespace count_two_digit_primes_with_units_digit_3_l765_765321

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765321


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765338

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765338


namespace solve_inequality_system_l765_765564

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765564


namespace nat_pairs_satisfy_conditions_l765_765765

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l765_765765


namespace cos_arithmetic_sequence_l765_765054

theorem cos_arithmetic_sequence (a_n : ℕ → ℝ)
  (h_seq : ∀ n : ℕ, a_n = a_1 + (n - 1) * d)  -- condition that {a_n} is an arithmetic sequence
  (h_sum : a_1 + a_5 + a_9 = 8 * Real.pi)  -- condition a_1 + a_5 + a_9 = 8π
  : Real.cos (a_3 + a_7) = -1 / 2 := by
sorry

end cos_arithmetic_sequence_l765_765054


namespace count_two_digit_primes_with_units_digit_three_l765_765415

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765415


namespace two_digit_primes_with_units_digit_three_l765_765231

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765231


namespace digit_150_of_fraction_5_over_31_is_1_l765_765632

theorem digit_150_of_fraction_5_over_31_is_1 :
  (0.\overline{161290322580645}).digit_after_decimal 150 = 1 :=
sorry

end digit_150_of_fraction_5_over_31_is_1_l765_765632


namespace quadratic_inequality_l765_765612

theorem quadratic_inequality (m : ℝ) : (∃ x : ℝ, x^2 - 3*x - m = 0 ∧ (∃ y : ℝ, y^2 - 3*y - m = 0 ∧ x ≠ y)) ↔ m > - 9 / 4 := 
by
  sorry

end quadratic_inequality_l765_765612


namespace two_digit_primes_with_units_digit_three_l765_765223

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765223


namespace arithmetic_sequence_prop_l765_765824

theorem arithmetic_sequence_prop (a1 d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5)
  (hSn : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) :
  (d < 0) ∧ (S 11 > 0) ∧ (|a1 + 5 * d| > |a1 + 6 * d|) := 
by
  sorry

end arithmetic_sequence_prop_l765_765824


namespace count_two_digit_primes_with_units_digit_three_l765_765407

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765407


namespace cubic_boxes_properties_l765_765647

-- Define the lengths of the edges of the cubic boxes
def edge_length_1 : ℝ := 3
def edge_length_2 : ℝ := 5
def edge_length_3 : ℝ := 6

-- Define the volumes of the respective cubic boxes
def volume (edge_length : ℝ) : ℝ := edge_length ^ 3
def volume_1 := volume edge_length_1
def volume_2 := volume edge_length_2
def volume_3 := volume edge_length_3

-- Define the surface areas of the respective cubic boxes
def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)
def surface_area_1 := surface_area edge_length_1
def surface_area_2 := surface_area edge_length_2
def surface_area_3 := surface_area edge_length_3

-- Total volume and surface area calculations
def total_volume := volume_1 + volume_2 + volume_3
def total_surface_area := surface_area_1 + surface_area_2 + surface_area_3

-- Theorem statement to be proven
theorem cubic_boxes_properties :
  total_volume = 368 ∧ total_surface_area = 420 := by
  sorry

end cubic_boxes_properties_l765_765647


namespace count_two_digit_primes_with_units_digit_3_l765_765165

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765165


namespace two_digit_primes_units_digit_3_count_l765_765348

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765348


namespace bobby_candy_count_l765_765723

theorem bobby_candy_count :
  let initial_candy := 33
  let additional_candy := 4
  let initial_chocolate := 14
  let additional_chocolate := 5
  let licorice := 7
  let total_candy := initial_candy + additional_candy
  let total_chocolate := initial_chocolate + additional_chocolate
  total_candy + total_chocolate + licorice = 63 :=
by
  let initial_candy := 33
  let additional_candy := 4
  let initial_chocolate := 14
  let additional_chocolate := 5
  let licorice := 7
  let total_candy := initial_candy + additional_candy
  let total_chocolate := initial_chocolate + additional_chocolate
  show total_candy + total_chocolate + licorice = 63
  calc 
    total_candy + total_chocolate + licorice
    = (initial_candy + additional_candy) + (initial_chocolate + additional_chocolate) + licorice : by rw [total_candy, total_chocolate]
  ... = (33 + 4) + (14 + 5) + 7 : by rw [initial_candy, additional_candy, initial_chocolate, additional_chocolate, licorice]
  ... = 37 + 19 + 7 : by norm_num
  ... = 63 : by norm_num

end bobby_candy_count_l765_765723


namespace intersection_of_A_and_B_l765_765084

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l765_765084


namespace identical_lines_pairs_count_l765_765010

theorem identical_lines_pairs_count : 
  ∃ P : Finset (ℝ × ℝ), (∀ p ∈ P, 
    (∃ a b, p = (a, b) ∧ 
      (∀ x y, 2 * x + a * y + b = 0 ↔ b * x + 3 * y - 9 = 0))) ∧ P.card = 2 :=
sorry

end identical_lines_pairs_count_l765_765010


namespace count_two_digit_primes_with_units_digit_three_l765_765414

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765414


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765260

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765260


namespace true_statement_given_conditions_l765_765800

theorem true_statement_given_conditions (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a < b) :
  |1| / |a| > |1| / |b| := 
by
  sorry

end true_statement_given_conditions_l765_765800


namespace count_two_digit_prime_numbers_ending_in_3_l765_765239

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765239


namespace cyclic_quadrilateral_angle_condition_l765_765818

theorem cyclic_quadrilateral_angle_condition
  (abc : Triangle)
  (m : Point)
  (hM : m = midpoint abc.A abc.B)
  (p q : Point)
  (hQ : q = reflection p m)
  (d e : Point)
  (hD : d = intersection_line_side (line_through_points abc.A p) abc.BC)
  (hE : e = intersection_line_side (line_through_points abc.B p) abc.AC) :
  (cyclic_quadrilateral abc.A abc.B d e) ↔ (angle abc.AC.P = angle q.C.B) :=
sorry

end cyclic_quadrilateral_angle_condition_l765_765818


namespace num_unique_permutations_l765_765101

-- Given: digits 3 and 8 being repeated as described.
-- Show: number of different permutations of the digits 3, 3, 3, 8, 8 is 10.

theorem num_unique_permutations : 
  let digits := [3, 3, 3, 8, 8] in
  let total_permutations := (5!).nat_abs in             -- 5! permutations
  let repeats_correction := (3!).nat_abs * (2!).nat_abs in -- Adjusting for repeated 3's and 8's
  let unique_permutations := total_permutations / repeats_correction in
  unique_permutations = 10 :=
by
  sorry

end num_unique_permutations_l765_765101


namespace exact_one_true_l765_765853

-- Define the propositions as boolean variables
def Prop1 (L1 L2 L3 : Type) [linear_ordered_ring L1] [linear_ordered_ring L2] [linear_ordered_ring L3] :=
  ∀ (l m n : L1), equal_angles_with_third l m n → parallel l m

def Prop2 (L1 L2 : Type) [linear_ordered_ring L1] [linear_ordered_ring L2] :=
  ∀ (l m n : L1), perpendicular_to_third l m n → parallel l m

def Prop3 (L1 L2 : Type) [linear_ordered_ring L1] [linear_ordered_ring L2] :=
  ∀ (l m n : L1), parallel_to_third l m n → parallel l m

-- State the problem
theorem exact_one_true (L1 L2 L3 : Type) [linear_ordered_ring L1] [linear_ordered_ring L2] [linear_ordered_ring L3] :
  (Prop1 L1 L2 L3) ∧ ¬(Prop2 L1 L2) ∧ ¬(Prop3 L1 L2) ∨
  ¬(Prop1 L1 L2 L3) ∧ (Prop2 L1 L2) ∧ ¬(Prop3 L1 L2) ∨
  ¬(Prop1 L1 L2 L3) ∧ ¬(Prop2 L1 L2) ∧ (Prop3 L1 L2) :=
sorry

end exact_one_true_l765_765853


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765283

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765283


namespace range_of_k_l765_765882

theorem range_of_k (k b : ℝ) 
  (h1 : ∃ b, y = k * x + b)
  (h2 : (2 : ℝ, 2 : ℝ) ∈ graph (λ x, k * x + b))
  (h3 : ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ (k * x + b) = (-x + 3)) :
  k ≤ -2 ∨ (k ≥ - (1 / 2) ∧ k ≠ 0) :=
  sorry

end range_of_k_l765_765882


namespace count_two_digit_primes_with_units_digit_3_l765_765323

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765323


namespace distinctPermutations_test_l765_765099

noncomputable def distinctPermutationsCount (s : Multiset ℕ) : ℕ :=
  (s.card.factorial) / (s.count 3.factorial * s.count 8.factorial)

theorem distinctPermutations_test : distinctPermutationsCount {3, 3, 3, 8, 8} = 10 := by
  sorry

end distinctPermutations_test_l765_765099


namespace num_two_digit_primes_with_units_digit_three_l765_765134

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765134


namespace range_f_l765_765613

-- Define the function and the condition
def f (x : ℝ) : ℝ := log 0.5 (x + 1 / (x - 1) + 1)

-- State the theorem
theorem range_f : ∀ y, (∃ x, x > 1 ∧ f x = y) ↔ y ∈ set.Iic (-2) :=
by sorry

end range_f_l765_765613


namespace two_digit_primes_with_units_digit_three_count_l765_765170

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765170


namespace count_two_digit_primes_with_units_digit_3_l765_765164

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765164


namespace find_k_l765_765909

-- Definitions
variable {a : ℕ → ℝ}
def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) + a (n - 1) = 2 * a n
def condition (a : ℕ → ℝ) : Prop := ∀ n ≥ 2, a (n + 1) - a n ^ 2 + a (n - 1) = 0
def S (k : ℕ) : ℝ := ∑ i in Finset.range (2 * k - 1), a i

-- Theorems and Problem Statement
theorem find_k (h0 : arithmetic_sequence a)
  (h1 : condition a) (h2 : S 12 = 46) : k = 12 := sorry

end find_k_l765_765909


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765385

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765385


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765273

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765273


namespace number_of_factors_of_M_l765_765486

-- a definition used in the condition
def M : ℕ := 47^4 + 4 * 47^3 + 6 * 47^2 + 4 * 47 + 1

-- theorem statement for the proof
theorem number_of_factors_of_M : ∃ n : ℕ, n = 85 ∧ ∀ d : ℕ, d ∣ M → d = 1 ∨ d = M ∨ d ∈ factors(M) := sorry

end number_of_factors_of_M_l765_765486


namespace solve_inequality_system_l765_765585

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l765_765585


namespace fraction_red_knights_magical_l765_765448

theorem fraction_red_knights_magical (total_knights red_knights blue_knights magical_knights : ℕ)
  (fraction_red fraction_magical : ℚ)
  (frac_red_mag : ℚ) :
  (red_knights = total_knights * fraction_red) →
  (fraction_red = 3 / 8) →
  (magical_knights = total_knights * fraction_magical) →
  (fraction_magical = 1 / 4) →
  (frac_red_mag * red_knights + (frac_red_mag / 3) * blue_knights = magical_knights) →
  (frac_red_mag = 3 / 7) :=
by
  -- Skipping proof
  sorry

end fraction_red_knights_magical_l765_765448


namespace children_tv_time_l765_765721

theorem children_tv_time
  (hours_per_2weeks : ℕ)
  (days_per_week : ℕ)
  (weeks : ℕ)
  (minutes_per_hour : ℕ)
  (total_minutes : ℕ)
  (days : ℕ)
  (minutes_per_day : ℕ) :
  hours_per_2weeks = 6 →
  weeks = 2 →
  days_per_week = 4 →
  minutes_per_hour = 60 →
  days = weeks * days_per_week →
  total_minutes = hours_per_2weeks * minutes_per_hour →
  minutes_per_day = total_minutes / days →
  minutes_per_day = 45 :=
begin
  sorry
end

end children_tv_time_l765_765721


namespace proof_problem_l765_765850

-- Define the conditions
def parametric_equation_line_l (m t : ℝ) : ℝ × ℝ :=
  (m + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

def polar_equation_ellipse_C (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ^2 + 3 * ρ^2 * Real.sin θ^2 = 12

def left_focus_F_on_line_l (F : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ t : ℝ, F = parametric_equation_line_l m t

-- Define the proof problem
theorem proof_problem (m : ℝ) (F A B : ℝ × ℝ) : 
  (matching_conditions : left_focus_F_on_line_l F m) → 
  -- First proof: |FA| ⋅ |FB| = 2
  let FA := dist F A,
  let FB := dist F B in 
  (line_intersects_ellipse : ∃ t₁ t₂ : ℝ, A = parametric_equation_line_l m t₁ ∧ B = parametric_equation_line_l m t₂ ∧ A ≠ B) →
  |FA * FB| = 2 ∧ 
  -- Second proof: maximum perimeter of inscribed rectangle is 16
  (∃ M : ℝ × ℝ, polar_equation_ellipse_C M.fst M.snd ∧ 
    let P := 4 * M.fst + 4 * M.snd in P = 16)
:= sorry

end proof_problem_l765_765850


namespace determine_a_l765_765829

noncomputable def imaginary_unit : ℂ := Complex.I

def is_on_y_axis (z : ℂ) : Prop :=
  z.re = 0

theorem determine_a (a : ℝ) : 
  is_on_y_axis (⟨(a - 3 * imaginary_unit.re), -(a - 3 * imaginary_unit.im)⟩ / ⟨(1 - imaginary_unit.re), -(1 - imaginary_unit.im)⟩) → 
  a = -3 :=
sorry

end determine_a_l765_765829


namespace inequality_system_solution_l765_765578

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l765_765578


namespace leap_years_in_200_years_l765_765717

theorem leap_years_in_200_years : 
  let leap_years (n : ℕ) := n % 4 = 0 ∧ ¬(n % 128 = 0) in
  (Finset.card (Finset.filter leap_years (Finset.range 200))) = 49 :=
by
  sorry

end leap_years_in_200_years_l765_765717


namespace circle_ratio_l765_765621

theorem circle_ratio
  (P : Point)
  (Q : Point)
  (O₁ O₂ O₃ : Circle)
  (A₁ A₂ A₃ : Point)
  (h₁ : O₁.contains P)
  (h₂ : O₁.contains Q)
  (h₃ : O₂.contains P)
  (h₄ : O₂.contains Q)
  (h₅ : O₃.contains P)
  (h₆ : O₃.contains Q)
  (L : Line)
  (hl₁ : L.contains P)
  (hl₂ : L.intersects_circle_at_points O₁ (A₁, P))
  (hl₃ : L.intersects_circle_at_points O₂ (A₂, P))
  (hl₄ : L.intersects_circle_at_points O₃ (A₃, P)) :
  ∃ A₁ A₂ A₃ O₁ O₂ O₃, ∃ (A₁ A₂ A₃ O₁ O₂ O₃ : ℝ),
    A₁ / A₂ = O₁ / O₂ :=
sorry

end circle_ratio_l765_765621


namespace count_two_digit_primes_with_units_digit_3_l765_765149

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765149


namespace circle_origin_range_l765_765874

theorem circle_origin_range (m : ℝ) : 
  (0 - m)^2 + (0 + m)^2 < 4 → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
sorry

end circle_origin_range_l765_765874


namespace count_whole_numbers_without_digit_2_1_to_500_l765_765860

def does_not_contain_digit_2 (n : ℕ) : Prop :=
  ¬(∃ k, k ≤ 3 ∧ (n / 10 ^ k) % 10 = 2)

def count_numbers_without_digit_2 (low high : ℕ) : ℕ :=
  (finset.range (high + 1)).filter (λ n, n ≥ low ∧ does_not_contain_digit_2 n).card

theorem count_whole_numbers_without_digit_2_1_to_500 :
  count_numbers_without_digit_2 1 500 = 323 :=
by
  sorry

end count_whole_numbers_without_digit_2_1_to_500_l765_765860


namespace distinctPermutations_test_l765_765097

noncomputable def distinctPermutationsCount (s : Multiset ℕ) : ℕ :=
  (s.card.factorial) / (s.count 3.factorial * s.count 8.factorial)

theorem distinctPermutations_test : distinctPermutationsCount {3, 3, 3, 8, 8} = 10 := by
  sorry

end distinctPermutations_test_l765_765097


namespace two_digit_primes_with_units_digit_three_count_l765_765183

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765183


namespace relationship_among_abc_l765_765490

-- Definitions and conditions based on the problem statement
variables {f : ℝ → ℝ}
def is_even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_increasing_on_pos (f : ℝ → ℝ) := ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- Given constants
def a := f (Real.log 3 / Real.log 2)
def b := f (3 / 2)
def c := f (Real.log 2 / Real.log 3)

-- Theorem: Prove that c < b < a
theorem relationship_among_abc (even_f : is_even_function f) (increasing_f : is_increasing_on_pos f) : c < b ∧ b < a := by
  sorry

end relationship_among_abc_l765_765490


namespace count_prime_units_digit_3_eq_6_l765_765295

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765295


namespace two_digit_primes_with_units_digit_three_l765_765228

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765228


namespace intersection_M_N_l765_765085

open Set

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

theorem intersection_M_N :
  M ∩ N = {-1, 3} := 
sorry

end intersection_M_N_l765_765085


namespace num_valid_subsets_l765_765502

def A : Set ℕ := {1, 2, 3, 4}

def valid_subsets (B : Set ℕ) : Prop :=
  B ⊂ A ∧ 1 ∈ B ∧ 4 ∉ B

theorem num_valid_subsets : finset.card { B : Set ℕ | valid_subsets B } = 4 :=
sorry

end num_valid_subsets_l765_765502


namespace count_two_digit_primes_with_units_digit_3_l765_765192

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765192


namespace angle_A_sides_b_c_l765_765445

noncomputable def triangle_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.sin C - Real.sqrt 3 * c * Real.cos A = 0

theorem angle_A (a b c A B C : ℝ) (h1 : triangle_angles a b c A B C) :
  A = Real.pi / 3 :=
by sorry

noncomputable def triangle_area (a b c S : ℝ) : Prop :=
  S = Real.sqrt 3 ∧ a = 2

theorem sides_b_c (a b c S : ℝ) (h : triangle_area a b c S) :
  b = 2 ∧ c = 2 :=
by sorry

end angle_A_sides_b_c_l765_765445


namespace count_five_digit_numbers_l765_765106

theorem count_five_digit_numbers :
  let digits := [3, 3, 3, 8, 8] in
  fintype.card {l : list ℕ // l.perm digits ∧ (∀ x ∈ l, x ∈ digits)} = 10 :=
by
  sorry

end count_five_digit_numbers_l765_765106


namespace find_a_l765_765046

noncomputable def l1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 1) * y + 1
noncomputable def l2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 2

def perp_lines (a : ℝ) : Prop :=
  let m1 := -a
  let m2 := -1 / a
  m1 * m2 = -1

theorem find_a (a : ℝ) : (perp_lines a) ↔ (a = 0 ∨ a = -2) := 
sorry

end find_a_l765_765046


namespace count_two_digit_primes_with_units_digit_three_l765_765405

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765405


namespace largest_binomial_coefficient_at_x5_l765_765793

theorem largest_binomial_coefficient_at_x5 (n : ℕ) (h : n > 0) :
  (∀ k, binomial n k ≤ binomial n 5) → n = 10 :=
sorry

end largest_binomial_coefficient_at_x5_l765_765793


namespace ln_inequality_l765_765845

/-- 
The given function is \( f(x) = \ln x - k(x - 1) + 1 \).
We need to prove that:
\( \frac{\ln 2}{3} + \frac{\ln 3}{4} + \cdots + \frac{\ln n}{n+1} < \frac{n(n-1)}{4} \) 
for \( n \in \mathbb{N}^{*} \) and \( n > 1 \). 
-/
theorem ln_inequality (n : ℕ) (h1 : 1 < n) : 
  (list.range n).map (λ i, (real.log (i + 2) / (i + 3))).sum < (n * (n - 1) / 4) :=
sorry

end ln_inequality_l765_765845


namespace total_birds_in_pet_store_l765_765666

-- Definitions for the conditions given in the problem
def num_cages : Nat := 6
def parrots_per_cage : Nat := 2
def parakeets_per_cage : Nat := 7

-- Definition to calculate the total number of birds per cage
def birds_per_cage : Nat := parrots_per_cage + parakeets_per_cage

-- Theorem to prove the total number of birds in the pet store
theorem total_birds_in_pet_store : num_cages * birds_per_cage = 54 := 
by 
  -- Calculation based on the given conditions
  have H1 : birds_per_cage = 9 := 
    by 
      simp [birds_per_cage, parrots_per_cage, parakeets_per_cage]
  have H2 : num_cages = 6 := rfl
  calc
    num_cages * birds_per_cage = 6 * 9 : by rw [H1, H2]
                        ...   = 54 : by simp

end total_birds_in_pet_store_l765_765666


namespace count_two_digit_primes_ending_in_3_l765_765366

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765366


namespace button_press_greater_than_1985_l765_765686

theorem button_press_greater_than_1985 (a0 b0 c0 d0 : ℤ) (h_not_all_equal : ¬(a0 = b0 ∧ b0 = c0 ∧ c0 = d0)) :
  ∃ n : ℕ, ∃ a b c d : ℤ, (a, b, c, d) = (a0, b0, c0, d0).iterated ((λ p, (p.1 - p.2, p.2 - p.3, p.3 - p.4, p.4 - p.1))) n ∧ (|a| > 1985 ∨ |b| > 1985 ∨ |c| > 1985 ∨ |d| > 1985) :=
sorry

end button_press_greater_than_1985_l765_765686


namespace coordinate_identification_l765_765840

noncomputable def x1 := (4 * Real.pi) / 5
noncomputable def y1 := -(Real.pi) / 5

noncomputable def x2 := (12 * Real.pi) / 5
noncomputable def y2 := -(3 * Real.pi) / 5

noncomputable def x3 := (4 * Real.pi) / 3
noncomputable def y3 := -(Real.pi) / 3

theorem coordinate_identification :
  (x1, y1) = (4 * Real.pi / 5, -(Real.pi) / 5) ∧
  (x2, y2) = (12 * Real.pi / 5, -(3 * Real.pi) / 5) ∧
  (x3, y3) = (4 * Real.pi / 3, -(Real.pi) / 3) :=
by
  -- proof goes here
  sorry

end coordinate_identification_l765_765840


namespace polygon_perimeter_greater_than_2_l765_765036

-- Definition of the conditions
variable (polygon : Set (ℝ × ℝ))
variable (A B : ℝ × ℝ)
variable (P : ℝ)

axiom point_in_polygon (p : ℝ × ℝ) : p ∈ polygon
axiom A_in_polygon : A ∈ polygon
axiom B_in_polygon : B ∈ polygon
axiom path_length_condition (γ : ℝ → ℝ × ℝ) (γ_in_polygon : ∀ t, γ t ∈ polygon) (hA : γ 0 = A) (hB : γ 1 = B) : ∀ t₁ t₂, 0 ≤ t₁ → t₁ ≤ t₂ → t₂ ≤ 1 → dist (γ t₁) (γ t₂) > 1

-- Statement to prove
theorem polygon_perimeter_greater_than_2 : P > 2 :=
sorry

end polygon_perimeter_greater_than_2_l765_765036


namespace total_questions_in_test_l765_765899

theorem total_questions_in_test :
  ∃ x, (5 * x = total_questions) ∧ 
       (20 : ℚ) / total_questions > (60 / 100 : ℚ) ∧ 
       (20 : ℚ) / total_questions < (70 / 100 : ℚ) ∧ 
       total_questions = 30 :=
by
  sorry

end total_questions_in_test_l765_765899


namespace inequality_solution_l765_765546

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l765_765546


namespace smallest_lcm_l765_765426

theorem smallest_lcm (a b : ℕ) (h₁ : 1000 ≤ a ∧ a < 10000) (h₂ : 1000 ≤ b ∧ b < 10000) (h₃ : Nat.gcd a b = 5) : 
  Nat.lcm a b = 201000 :=
sorry

end smallest_lcm_l765_765426


namespace first_marvelous_monday_after_school_starts_l765_765904

def is_marvelous_monday (year : ℕ) (month : ℕ) (day : ℕ) (start_day : ℕ) : Prop :=
  let days_in_month := if month = 9 then 30 else if month = 10 then 31 else 0
  let fifth_monday := start_day + 28
  let is_monday := (fifth_monday - 1) % 7 = 0
  month = 10 ∧ day = 30 ∧ is_monday

theorem first_marvelous_monday_after_school_starts :
  ∃ (year month day : ℕ),
    year = 2023 ∧ month = 10 ∧ day = 30 ∧ is_marvelous_monday year month day 4 := sorry

end first_marvelous_monday_after_school_starts_l765_765904


namespace count_two_digit_primes_with_units_digit_3_l765_765162

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765162


namespace count_two_digit_prime_numbers_ending_in_3_l765_765244

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765244


namespace two_digit_primes_with_units_digit_three_l765_765226

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765226


namespace flour_to_add_l765_765505

-- Define the conditions
def total_flour_required : ℕ := 9
def flour_already_added : ℕ := 2

-- Define the proof statement
theorem flour_to_add : total_flour_required - flour_already_added = 7 := 
by {
    sorry
}

end flour_to_add_l765_765505


namespace two_digit_primes_with_units_digit_three_count_l765_765167

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765167


namespace derivative_f_at_1_l765_765864

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * Real.sin x

theorem derivative_f_at_1 : (deriv f 1) = 2 + 2 * Real.cos 1 := 
sorry

end derivative_f_at_1_l765_765864


namespace digit_150_of_fraction_5_over_31_is_1_l765_765631

theorem digit_150_of_fraction_5_over_31_is_1 :
  (0.\overline{161290322580645}).digit_after_decimal 150 = 1 :=
sorry

end digit_150_of_fraction_5_over_31_is_1_l765_765631


namespace sum_a_eq_2018_l765_765847

def f (n : ℕ) : ℤ :=
if n % 2 = 1 then n^2 else -(n^2)

def a (n : ℕ) : ℤ :=
f n + f (n + 1)

theorem sum_a_eq_2018 : (finset.range 2018).sum (λ n, a (n + 1)) = 2018 := 
by
  sorry

end sum_a_eq_2018_l765_765847


namespace counting_adjacent_numbers_l765_765518

open Finset

/- The original problem translated into a Lean theorem statement -/
theorem counting_adjacent_numbers (n : ℕ) (k : ℕ) (h1 : n = 49) (h2 : k = 6) :
  (choose n k) - (choose (n - k) k) = (choose 49 6) - (choose 44 6) :=
by
  rw [h1, h2, choose, choose] sorry

end counting_adjacent_numbers_l765_765518


namespace num_two_digit_primes_with_units_digit_three_l765_765131

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765131


namespace find_pairs_l765_765752

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l765_765752


namespace solve_inequality_system_l765_765538

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765538


namespace numberOfCorrectConclusions_l765_765075

theorem numberOfCorrectConclusions :
  ( (∀ (x : ℝ), x > 0 → x > Real.sin x) ∧
    ( (∀ (x : ℝ), x ≠ 0 → x - Real.sin x ≠ 0) ∧
      (¬ (∀ (p q : Prop), (p ∧ q) → (p ∨ q) → True)) ∧
        ¬ (∀ (x : ℝ), 0 < x → x - Real.log x > 0) → 
          (∃ (x : ℝ), 0 < x ∧ x - Real.log x ≤ 0)
    )
  ) →
  3 :=
sorry

end numberOfCorrectConclusions_l765_765075


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765266

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765266


namespace max_slope_of_line_to_origin_l765_765429

theorem max_slope_of_line_to_origin {x y : ℝ} (h : (x - 2)^2 + y^2 = 3) : 
  ∃ k : ℝ, k = (y / x) ∧ k ≤ sqrt 3 :=
sorry

end max_slope_of_line_to_origin_l765_765429


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765272

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765272


namespace two_digit_primes_units_digit_3_count_l765_765360

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765360


namespace above_line_sign_l765_765810

theorem above_line_sign (A B C x y : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
(h_above : ∃ y₁, Ax + By₁ + C = 0 ∧ y > y₁) : 
  (Ax + By + C > 0 ∧ B > 0) ∨ (Ax + By + C < 0 ∧ B < 0) := 
by
  sorry

end above_line_sign_l765_765810


namespace nat_pairs_satisfy_conditions_l765_765764

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l765_765764


namespace remainder_of_sum_of_integers_l765_765975

theorem remainder_of_sum_of_integers (a b c : ℕ)
  (h₁ : a % 30 = 15) (h₂ : b % 30 = 5) (h₃ : c % 30 = 10) :
  (a + b + c) % 30 = 0 := by
  sorry

end remainder_of_sum_of_integers_l765_765975


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765382

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765382


namespace circle_tangent_independence_l765_765889

noncomputable def e1 (r : ℝ) (β : ℝ) := r * Real.tan β
noncomputable def e2 (r : ℝ) (α : ℝ) := r * Real.tan α
noncomputable def e3 (r : ℝ) (β α : ℝ) := r * Real.tan (β - α)

theorem circle_tangent_independence 
  (O : ℝ) (r β α : ℝ) (hβ : β < π / 2) (hα : 0 < α) (hαβ : α < β) :
  (e1 r β) * (e2 r α) * (e3 r β α) / ((e1 r β) - (e2 r α) - (e3 r β α)) = r^2 :=
by
  sorry

end circle_tangent_independence_l765_765889


namespace angle_BXY_is_30_deg_l765_765459

theorem angle_BXY_is_30_deg (AB CD : Line) (parallel_AB_CD : AB || CD) (AXE CYX BXY : Angle)
  (h_AXE_eq_4CYX_sub_90 : AXE = 4 * CYX - 90) (h_AXE_eq_CYX : AXE = CYX) : BXY = 30 := by
  sorry

end angle_BXY_is_30_deg_l765_765459


namespace attendance_difference_l765_765507

theorem attendance_difference :
  let a := 65899
  let b := 66018
  b - a = 119 :=
sorry

end attendance_difference_l765_765507


namespace incorrect_option_d_l765_765828

variable {a b : Set Point}
variable {m n : Set Point}

-- a and b are distinct planes
axiom distinct_planes (h1 : ∀ (x : Point), x ∈ a → x ∈ b → False) : a ≠ b

-- m and n are lines
axiom line_m (h2 : ∀ (x y : Point), x ∈ m → y ∈ m → ∃ ! l, m = l)
axiom line_n (h3 : ∀ (x y : Point), x ∈ n → y ∈ n → ∃ ! l, n = l)

-- perpendicular relationship notation
def perpendicular (x y : Set Point) : Prop := ∀ (p : Point), p ∈ x → p ∈ y → False

-- subset notation
def subset (x y : Set Point) : Prop := ∀ (p : Point), p ∈ x → p ∈ y

theorem incorrect_option_d (h1 : a ≠ b) (h2 : ∀ (x y : Point), x ∈ m → y ∈ m → ∃ ! l, m = l)
  (h3 : ∀ (x y : Point), x ∈ n → y ∈ n → ∃ ! l, n = l) (h4 : perpendicular a b) (h5 : subset m a) :
  ¬ perpendicular m b := by
  sorry

end incorrect_option_d_l765_765828


namespace range_of_values_for_a_l765_765436

theorem range_of_values_for_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ x1^2 + a * x1 + a^2 - 1 = 0 ∧ x2^2 + a * x2 + a^2 - 1 = 0) → (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_values_for_a_l765_765436


namespace AC_passes_through_fixed_point_l765_765469
noncomputable def point_on_AC : Prop :=
  ∀ (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C],
  ∃ p q : ℝ, ∃ α m : ℝ,
    angle B = α ∧ ( 1/ ∥AB∥ + 1/∥BC∥ = m ) → 
    ∃ (x y :ℝ ), (m * y - real.sin α = 0) ∧  (real.sin α * x - (cos α + 1) * y = 0)

-- The following is just an assumption about the coordinates that needs to be derived based on the provided setup and conditions.
def proves_AC_passes_through_fixed_point (A B C : Type) 
  [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
  (p q α m : ℝ) (h1 : angle B = α) (h2 : 1 / (∥AB∥ : ℝ) + 1 / (∥BC∥ : ℝ) = m) : Prop :=
  ∃ (x y : ℝ), (m * y - real.sin α = 0) ∧ 
               (real.sin α * x - (real.cos α + 1) * y = 0) → 
  line_passes_through_point AC (x, y)

theorem AC_passes_through_fixed_point : point_on_AC := sorry

end AC_passes_through_fixed_point_l765_765469


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765259

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765259


namespace count_two_digit_primes_with_units_digit_three_l765_765406

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765406


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765331

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765331


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765389

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765389


namespace solve_inequality_system_l765_765582

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l765_765582


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765263

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765263


namespace problem1_problem2_l765_765078

-- Problem 1
theorem problem1 (x : ℝ) (hx : x ≥ 0) : 
  let f := λ x, Real.exp x - x^2 in
  f x ≥ 1 :=
begin
  let f := λ x, Real.exp x - x^2,
  sorry
end

-- Problem 2
theorem problem2 (a : ℝ) : 
  (∃ x ∈ Set.Ioi 0, f x = 0 ∧ ∀ y ∈ Set.Ioi 0, y ≠ x → f y ≠ 0) ↔ 
  a = Real.exp 2 / (4 : ℝ) :=
begin
  let f := λ x, Real.exp x - a * x^2,
  sorry
end

end problem1_problem2_l765_765078


namespace trajectory_equation_proof_tangent_line_proof_min_value_proof_l765_765808

noncomputable def focus_point : ℝ × ℝ := (0, 1)
noncomputable def directrix : ℝ → ℝ := fun y => -1
noncomputable def parabola_trajectory_equation := ∀ (M : ℝ × ℝ), dist M (0, 1) = dist M (M.fst, -1) → M.snd = (M.fst ^ 2) / 4

theorem trajectory_equation_proof (M : ℝ × ℝ) (h : dist M focus_point = dist M (M.fst, directrix M.snd)) :
  parabola_trajectory_equation M h :=
sorry

noncomputable def line_l : ℝ × ℝ → Prop := λ (P : ℝ × ℝ), P.fst - P.snd - 2 = 0
noncomputable def point_p : ℝ × ℝ := (1 / 2, -3 / 2)
noncomputable def tangent_line_equation := ∀ (x₁ x₂ : ℝ) (y₁ y₂ y₀: ℝ), y₁ = (x₁ ^ 2) / 4 → y₂ = (x₂ ^ 2) / 4 →
  (P : ℝ × ℝ) (hP : line_l P) → (line_AB : ℝ × ℝ → Prop) := λ (P(x₀, y₀) ∈ {(1 / 2, -3 / 2)}) (AB : ℝ × ℝ → Prop), AB.fst - 2 * AB.snd + 6 = 0

theorem tangent_line_proof (P : ℝ × ℝ) (hP: line_l point_p) :
  tangent_line_equation P hP :=
sorry

noncomputable def product_AF_BF := ∀ (P: ℝ × ℝ), line_l P → ∀ (y₀ : ℝ) (h: P.fst = y₀ + 2), min_value |AF * BF| = 9/2

theorem min_value_proof (P: ℝ × ℝ) (hP: line_l P) :
  product_AF_BF P hP :=
sorry

end trajectory_equation_proof_tangent_line_proof_min_value_proof_l765_765808


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765336

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765336


namespace yolandas_husband_will_catch_her_l765_765654

-- Define the given conditions as variables/constants
variable (yolanda_speed1 yolanda_speed2 yolanda_speed3 : ℝ := 20)
variable (speed_increase_distance speed_decrease_distance final_distance : ℝ := 5)
variable (yolanda_stop_time : ℝ := 6) -- in minutes
variable (yolanda_total_distance : ℝ := 20)
variable (husband_speed : ℝ := 40) -- mph
variable (route_difference : ℝ := 10) -- 10 miles shorter

-- Define timings to account the distance traveled by Yolanda
noncomputable def yolanda_time_segment1 : ℝ := speed_increase_distance / yolanda_speed1
noncomputable def yolanda_time_segment2 : ℝ := (speed_increase_distance + 3) / yolanda_speed2
noncomputable def yolanda_time_segment3 : ℝ := final_distance / yolanda_speed3
noncomputable def yolanda_total_time : ℝ := (yolanda_time_segment1 + yolanda_time_segment2 + yolanda_time_segment3) * 60 + 2 * yolanda_stop_time
noncomputable def husband_route_length : ℝ := yolanda_total_distance - route_difference
noncomputable def yolanda_distance_till_husband_start : ℝ := yolanda_speed1 * 15 / 60

-- Define the Lean theorem
theorem yolandas_husband_will_catch_her (yx : yolanda_distance_till_husband_start < yolanda_total_distance) :
  husband_route_length = 10 ∧
  (15 + (husband_route_length / husband_speed * 60)).toNat = 30 := 
by sorry

end yolandas_husband_will_catch_her_l765_765654


namespace inequality_solution_l765_765545

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l765_765545


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765264

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765264


namespace range_of_a_l765_765960

open Real

/-- Proposition p: x^2 + 2*a*x + 4 > 0 for all x in ℝ -/
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: the exponential function (3 - 2*a)^x is increasing -/
def q (a : ℝ) : Prop :=
  3 - 2*a > 1

/-- Given p ∧ q, prove that -2 < a < 1 -/
theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : -2 < a ∧ a < 1 :=
sorry

end range_of_a_l765_765960


namespace find_x_coordinate_l765_765596

open Real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def distance (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x1 - x2)^2 + (y1 - y2)^2)

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

theorem find_x_coordinate (y x : ℝ) (hM_on_parabola : parabola x y)
    (h_distance_to_focus : distance x y 1 0 = 5) : x = 4 := 
begin
  -- sorry as the actual proof steps are not needed
  sorry
end

end find_x_coordinate_l765_765596


namespace curveC_polar_slope_lineL_l765_765458

-- Problem: Define curve C and line l with given conditions, prove the transformations.

variable {θ t α : ℝ}

-- Condition 1: Parametric equations of curve C.
def curveC (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, 3 + sqrt 5 * sin θ)

-- Condition 2: Parametric equations of line l.
def lineL (t α : ℝ) : ℝ × ℝ := (t * cos α, t * sin α)

-- Polar equation of curve C derived from the parametric equations.
theorem curveC_polar (θ : ℝ) : (∀ θ, (curveC θ).1^2 + ((curveC θ).2 - 3)^2 = 5) → (∀ θ, let ρ := sqrt((curveC θ).1^2 + (curveC θ).2^2) in ρ^2 - 6 * ρ * sin θ + 4 = 0) :=
by
  intros h θ
  sorry

-- Slope of line l given intersection conditions.
theorem slope_lineL (t α : ℝ) (dist_AB : ℝ) : dist_AB = 2 * sqrt 3 → ∀ α, let slope := tan α in slope = sqrt 14 / 2 ∨ slope = -sqrt 14 / 2 :=
by
  intros h α
  sorry

end curveC_polar_slope_lineL_l765_765458


namespace minimize_abs_z_l765_765498

noncomputable def smallest_value_of_abs_z (z : ℂ) (condition : |z - 15| + |z + 3 * complex.I| = 20) : ℝ :=
  if |z| = 2.25 then 2.25 else sorry

theorem minimize_abs_z : ∃ z : ℂ, |z - 15| + |z + 3 * complex.I| = 20 → |z| = 2.25 :=
by
  use 2.25
  sorry

end minimize_abs_z_l765_765498


namespace nearest_integer_to_exp_is_752_l765_765643

noncomputable def nearest_integer_to_exp := (3 + Real.sqrt 5) ^ 4

theorem nearest_integer_to_exp_is_752 : Int.round nearest_integer_to_exp = 752 := 
sorry

end nearest_integer_to_exp_is_752_l765_765643


namespace count_two_digit_primes_ending_in_3_l765_765375

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765375


namespace minimum_value_of_f_on_interval_1_4_l765_765611

def f (x : ℝ) : ℝ := -x^2 + 4 * x + 5

theorem minimum_value_of_f_on_interval_1_4 : 
  ∀ x ∈ set.Icc (1:ℝ) 4, f x ≥ 5 :=
sorry

end minimum_value_of_f_on_interval_1_4_l765_765611


namespace count_two_digit_prime_numbers_ending_in_3_l765_765241

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765241


namespace additional_time_due_to_leak_l765_765677

-- Given conditions
def R_f : ℝ := 1 / 5
def R_l : ℝ := 1 / 10

-- Question: Prove that the additional time to fill the cistern with the leak is 5 hours
theorem additional_time_due_to_leak (T_without_leak : ℝ) (T_effective : ℝ) : T_without_leak = 5 → T_effective = 10 → T_effective - T_without_leak = 5 :=
by
  intros h1 h2
  rw [h1, h2]
  linarith
  sorry -- Proof is omitted

end additional_time_due_to_leak_l765_765677


namespace blue_pens_count_l765_765524

variable (x y : ℕ) -- Define x as the number of red pens and y as the number of blue pens.
variable (h1 : 5 * x + 7 * y = 102) -- Condition 1: Total cost equation.
variable (h2 : x + y = 16) -- Condition 2: Total number of pens equation.

theorem blue_pens_count : y = 11 :=
by
  sorry

end blue_pens_count_l765_765524


namespace num_unique_permutations_l765_765103

-- Given: digits 3 and 8 being repeated as described.
-- Show: number of different permutations of the digits 3, 3, 3, 8, 8 is 10.

theorem num_unique_permutations : 
  let digits := [3, 3, 3, 8, 8] in
  let total_permutations := (5!).nat_abs in             -- 5! permutations
  let repeats_correction := (3!).nat_abs * (2!).nat_abs in -- Adjusting for repeated 3's and 8's
  let unique_permutations := total_permutations / repeats_correction in
  unique_permutations = 10 :=
by
  sorry

end num_unique_permutations_l765_765103


namespace roberto_hike_time_l765_765525

def uphill_speed : ℝ := 2 -- speed in miles per hour
def downhill_speed : ℝ := 3 -- speed in miles per hour
def total_trail_length : ℝ := 5 -- length in miles
def uphill_percentage : ℝ := 0.6 -- 60% uphill

-- Define the downhill percentage based on remaining percentage
def downhill_percentage : ℝ := 1 - uphill_percentage

-- Calculate distances
def uphill_distance : ℝ := total_trail_length * uphill_percentage
def downhill_distance : ℝ := total_trail_length * downhill_percentage

-- Calculate time in hours
def uphill_time_hours : ℝ := uphill_distance / uphill_speed
def downhill_time_hours : ℝ := downhill_distance / downhill_speed

-- Convert time to minutes
def uphill_time_minutes : ℝ := uphill_time_hours * 60
def downhill_time_minutes : ℝ := downhill_time_hours * 60

-- Calculate total time
def total_time_minutes : ℝ := uphill_time_minutes + downhill_time_minutes

theorem roberto_hike_time : total_time_minutes = 130 := by
  sorry

end roberto_hike_time_l765_765525


namespace parabola_distance_l765_765434

theorem parabola_distance (m : ℝ) (h : (∀ (p : ℝ), p = 1 / 2 → m = 4 * p)) : m = 2 :=
by
  -- Goal: Prove m = 2 given the conditions.
  sorry

end parabola_distance_l765_765434


namespace total_number_of_items_l765_765024

theorem total_number_of_items (total_items : ℕ) (selected_items : ℕ) (h1 : total_items = 50) (h2 : selected_items = 10) : total_items = 50 :=
by
  exact h1

end total_number_of_items_l765_765024


namespace eulerian_path_exists_l765_765473

-- Define the graph structure

def figure_graph : SimpleGraph ℕ :=
  SimpleGraph.mk' { edges := { ⟨1, 2⟩, ⟨2, 3⟩, ⟨3, 4⟩, ⟨4, 1⟩, -- square 1
                               ⟨3, 5⟩, ⟨5, 6⟩, ⟨6, 1⟩, ⟨1, 2⟩, ⟨2, 3⟩, ⟨3, 6⟩, ⟨6, 5⟩, ⟨5, 4⟩},
                    sym := fun ⟨x, y⟩ h => by cases h; simp [edges] }

-- Define vertices for reference
def vertices : list ℕ := [1, 2, 3, 4, 5, 6]

-- This verifies the conditions stated in the problem; these should be demonstrated in the proof.
-- We must show that this graph satisfies the Eulerian path conditions, which requires proving the existence of such a path.

theorem eulerian_path_exists (G : SimpleGraph ℕ) (v : G.V) : 
  (∃ p : list G.V, G.IsEulerianPath p) ↔ count_odd_degree_neighbors G = 2 := sorry

end eulerian_path_exists_l765_765473


namespace two_digit_primes_with_units_digit_three_l765_765233

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765233


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765269

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765269


namespace overhead_expenses_correct_l765_765700

noncomputable def overhead_expenses (purchase_price selling_price profit_percent : ℚ) : ℚ :=
  let E := (selling_price - (profit_percent / 100) * purchase_price - purchase_price) /
             (1 + (profit_percent / 100))
  in E

theorem overhead_expenses_correct :
  overhead_expenses 225 300 18.577075098814234 ≈ 27.79 :=
by
  sorry

end overhead_expenses_correct_l765_765700


namespace astronomical_magnitudes_and_brightnesses_l765_765905

noncomputable def magnitude_brightness_relation (m₁ m₂ : ℝ) (E₁ E₂ : ℝ) : Prop :=
m₂ - m₁ = (5 / 2) * Real.log10 (E₁ / E₂)

noncomputable def magnitude_approximation (mSirius mPolaris : ℝ) : Prop :=
mSirius / mPolaris ≈ 2

noncomputable def brightness_ratio_sun_sirius (mSun mSirius : ℝ) : Prop :=
let ratio := (5 / 2) * (mSirius - mSun) in
Real.log10 (10^(ratio)) = 10.1

noncomputable def brightness_ratio_sirius_sun (mSun mSirius : ℝ) : Prop :=
let ratio := (5 / 2) * (mSirius - mSun) in
Real.log10 (10^(-ratio)) = -10.1

noncomputable def brightness_ratio_sirius_polaris (mSirius mPolaris : ℝ) : Prop :=
let ratio := (5 / 2) * (mPolaris - mSirius) in
Real.log10 (10^(ratio)) = 0.292

theorem astronomical_magnitudes_and_brightnesses :
  ∀ (mSun mSirius mPolaris ESun ESirius EPolaris : ℝ),
    mSun = -26.7 →
    mSirius = -1.45 →
    mPolaris = -0.72 →
    magnitude_approximation mSirius mPolaris →
    magnitude_brightness_relation mSun mSirius ESun ESirius →
    magnitude_brightness_relation mSirius mPolaris ESirius EPolaris →
    brightness_ratio_sun_sirius mSun mSirius →
    brightness_ratio_sirius_sun mSun mSirius →
    brightness_ratio_sirius_polaris mSirius mPolaris :=
by {
  sorry
}

end astronomical_magnitudes_and_brightnesses_l765_765905


namespace find_m_l765_765432

-- Define the conditions
def parabola_eq (m : ℝ) (x y : ℝ) : Prop := x^2 = m * y
def vertex_to_directrix_dist (d : ℝ) : Prop := d = 1 / 2

-- State the theorem
theorem find_m (m : ℝ) (x y d : ℝ) 
  (h1 : parabola_eq m x y) 
  (h2 : vertex_to_directrix_dist d) :
  m = 2 :=
by
  sorry

end find_m_l765_765432


namespace minimum_a_for_quadratic_roots_l765_765791

theorem minimum_a_for_quadratic_roots :
  ∃ (a b c : ℤ), 0 < a ∧ (ax^2 + bx + c = 0) 
    has roots α and β satisfying 0 < α < β < 1 ∧ a = 5 := 
by 
  sorry

end minimum_a_for_quadratic_roots_l765_765791


namespace parabola_focus_l765_765783

theorem parabola_focus (x y : ℝ) :
  y = 9 * x^2 + 6 * x - 2 → 
  (∃ (h k : ℝ), y = 9 * (x - h)^2 + k ∧ h = -1/3 ∧ k = -3) ∧ 
  (∃ (a : ℝ), a = 9 ∧ (h + k + 1 / (4 * a)) = (-1/3, -107/36)) :=
begin
  sorry
end

end parabola_focus_l765_765783


namespace count_two_digit_primes_ending_in_3_l765_765367

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765367


namespace no_square_ends_with_four_identical_digits_except_0_l765_765659

theorem no_square_ends_with_four_identical_digits_except_0 (n : ℤ) :
  ¬ (∃ k : ℕ, (1 ≤ k ∧ k < 10) ∧ (n^2 % 10000 = k * 1111)) :=
by {
  sorry
}

end no_square_ends_with_four_identical_digits_except_0_l765_765659


namespace function_f_limit_l765_765892

variables {G : Type*} [graph G] (n : ℕ)

def value (e : edge G) : ℝ := sorry

def weight (p : path G) : ℝ := p.edges.map value).sup'

def f (x y : G) : ℝ := (path G x y).inf' (weight)

theorem function_f_limit : ∀ G : Type* [graph G] (n : ℕ) (h : connected_graph G n),
  ∃ s : set ℝ, (∀ x y : G, f x y ∈ s) ∧ s.card ≤ n - 1 :=
by
  sorry

end function_f_limit_l765_765892


namespace masha_combinations_seven_beads_even_beads_combinations_odd_permutations_even_odd_possible_l765_765951

-- Statement 1: Number of combinations Masha got is 127 with 7 beads
theorem masha_combinations_seven_beads : 
  ∀ (n : ℕ), n = 7 → (2^n - 1) = 127 :=
by sorry

-- Statement 2: For an even number of beads, number of combinations excluding the empty set is always odd
theorem even_beads_combinations_odd : 
  ∀ (n : ℕ), n % 2 = 0 → ¬even (2^n - 1) :=
by sorry

-- Statement 3: If the order matters, the total permutations can be even or odd
theorem permutations_even_odd_possible :
  ∀ (n : ℕ), n > 0 → 
  (∑ k in finset.range (n + 1), nat.perm n k) % 2 = 0 ∨
  (∑ k in finset.range (n + 1), nat.perm n k) % 2 = 1 :=
by sorry

end masha_combinations_seven_beads_even_beads_combinations_odd_permutations_even_odd_possible_l765_765951


namespace max_value_at_x_eq_2_l765_765020

noncomputable def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 - 3

theorem max_value_at_x_eq_2 : ∀ x : ℝ, quadratic_function x ≤ quadratic_function 2 := by
  sorry

end max_value_at_x_eq_2_l765_765020


namespace number_of_lines_l765_765089

def Point := ℝ × ℝ

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def distance_to_line (p : Point) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / real.sqrt (a^2 + b^2)

def tangent_count (p1 p2 : Point) (r1 r2 : ℝ) : ℕ :=
  -- function returns the number of tangent lines between two circles
  if distance p1 p2 = r1 + r2 -- externally tangent
  then 3 -- two external and one internal tangent
  else sorry -- other cases not required for this problem

theorem number_of_lines (M N : Point) (r1 r2 : ℝ) :
  M = (1,0) ∧ N = (-3,0) ∧ r1 = 1 ∧ r2 = 3 →
  tangent_count M N r1 r2 = 3 :=
by
  intros h
  -- extract conditions from the hypothesis
  cases h with hM htmp1
  cases htmp1 with hN htmp2
  cases htmp2 with hr1 hr2
  rw [hM, hN, hr1, hr2]
  sorry

end number_of_lines_l765_765089


namespace count_two_digit_primes_with_units_digit_3_l765_765154

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765154


namespace arithmetic_sum_s6_l765_765067

theorem arithmetic_sum_s6 (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ) 
  (h1 : ∀ n, a (n+1) - a n = d)
  (h2 : a 1 = 2)
  (h3 : S 4 = 20)
  (hS : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) :
  S 6 = 42 :=
by sorry

end arithmetic_sum_s6_l765_765067


namespace equivalent_expression_l765_765669

theorem equivalent_expression :
  0.027^(-1/3) + real.root (16^3) 4 - 3^(-1) + (real.sqrt 2 - 1)^0 = 12 := by
sorry

end equivalent_expression_l765_765669


namespace number_of_two_digit_primes_with_units_digit_three_l765_765207

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765207


namespace hyperbola_symmetric_slopes_l765_765838

/-- 
Let \(M(x_0, y_0)\) and \(N(-x_0, -y_0)\) be points symmetric about the origin on the hyperbola 
\(\frac{x^2}{16} - \frac{y^2}{4} = 1\). Let \(P(x, y)\) be any point on the hyperbola. 
When the slopes \(k_{PM}\) and \(k_{PN}\) both exist, then \(k_{PM} \cdot k_{PN} = \frac{1}{4}\),
independent of the position of \(P\).
-/
theorem hyperbola_symmetric_slopes (x x0 y y0: ℝ) 
  (hP: x^2 / 16 - y^2 / 4 = 1)
  (hM: x0^2 / 16 - y0^2 / 4 = 1)
  (h_slop_M : x ≠ x0)
  (h_slop_N : x ≠ x0):
  ((y - y0) / (x - x0)) * ((y + y0) / (x + x0)) = 1 / 4 := 
sorry

end hyperbola_symmetric_slopes_l765_765838


namespace adults_collectively_ate_l765_765895

theorem adults_collectively_ate (A : ℕ) (C : ℕ) (total_cookies : ℕ) (share : ℝ) (each_child_gets : ℕ)
  (hC : C = 4) (hTotal : total_cookies = 120) (hShare : share = 1/3) (hEachChild : each_child_gets = 20)
  (children_gets : ℕ) (hChildrenGets : children_gets = C * each_child_gets) :
  children_gets = (2/3 : ℝ) * total_cookies → (share : ℝ) * total_cookies = 40 :=
by
  -- Placeholder for simplified proof
  sorry

end adults_collectively_ate_l765_765895


namespace rectangular_plot_length_l765_765663

theorem rectangular_plot_length
  (b : ℝ)
  (Hlength : ∀ (b : ℝ), (40 + b))
  (Hcost : 26.50 * (4 * b + 80) = 5300)
  (b_value : b = 30) :
  (length : ℝ) = 70 := by
  sorry

end rectangular_plot_length_l765_765663


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765281

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765281


namespace find_m_l765_765433

-- Define the conditions
def parabola_eq (m : ℝ) (x y : ℝ) : Prop := x^2 = m * y
def vertex_to_directrix_dist (d : ℝ) : Prop := d = 1 / 2

-- State the theorem
theorem find_m (m : ℝ) (x y d : ℝ) 
  (h1 : parabola_eq m x y) 
  (h2 : vertex_to_directrix_dist d) :
  m = 2 :=
by
  sorry

end find_m_l765_765433


namespace prod_divisible_by_factorial_l765_765528

def prod_within_range (n : ℕ) := ∏ k in finset.range n, (2^n - 2^k)

theorem prod_divisible_by_factorial (n : ℕ) (hn : n ≥ 1) :
  prod_within_range n % n.factorial = 0 := 
sorry

end prod_divisible_by_factorial_l765_765528


namespace price_of_cupcake_per_piece_l765_765627

-- Defining the conditions stated in the problem
def price_of_cookies : ℝ := 2
def price_of_biscuits : ℝ := 1
def avg_number_of_cupcakes_per_day : ℕ := 20
def avg_number_of_cookies_per_day : ℕ := 10
def avg_number_of_biscuits_per_day : ℕ := 20
def total_earnings_in_5_days : ℝ := 350

-- Lean proof statement to determine the price of each cupcake
theorem price_of_cupcake_per_piece : 
  let total_cupcakes := avg_number_of_cupcakes_per_day * 5,
      total_cookies := avg_number_of_cookies_per_day * 5,
      total_biscuits := avg_number_of_biscuits_per_day * 5,
      earnings_from_cookies := total_cookies * price_of_cookies,
      earnings_from_biscuits := total_biscuits * price_of_biscuits,
      earnings_from_cupcakes := total_earnings_in_5_days - earnings_from_cookies - earnings_from_biscuits,
      price_of_cupcake := earnings_from_cupcakes / total_cupcakes
  in price_of_cupcake = 1.50 := 
by
  sorry

end price_of_cupcake_per_piece_l765_765627


namespace inequality_system_solution_l765_765572

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l765_765572


namespace inequality_system_solution_l765_765554

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l765_765554


namespace two_digit_primes_with_units_digit_three_count_l765_765182

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765182


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765394

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765394


namespace pages_per_hour_comic_books_l765_765506

noncomputable def total_time_reading : ℝ := 4 -- 24 hours/day * 1/6 day
def time_per_book_type : ℝ := total_time_reading / 3
def pages_per_hour_novels : ℝ := 21
def pages_per_hour_graphic_novels : ℝ := 30
def total_pages : ℝ := 128
def pages_read_novels : ℝ := pages_per_hour_novels * time_per_book_type
def pages_read_graphic_novels : ℝ := pages_per_hour_graphic_novels * time_per_book_type
def pages_read_comic_books : ℝ := total_pages - pages_read_novels - pages_read_graphic_novels

theorem pages_per_hour_comic_books : total_pages - pages_read_novels - pages_read_graphic_novels = 
  pages_per_hour_comic_books * time_per_book_type → (pages_per_hour_comic_books = 45) :=
by
  sorry

end pages_per_hour_comic_books_l765_765506


namespace curve_is_circle_l765_765741

theorem curve_is_circle (θ : ℝ) (r : ℝ) (h : r = 6 * (Real.sin θ) * (Real.csc θ)) :
  ∃ (c : ℝ), (c = 6) ∧ (∀ (θ : ℝ), r = c) :=
by
  sorry

end curve_is_circle_l765_765741


namespace inequality_system_solution_l765_765574

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l765_765574


namespace domain_transformation_l765_765060

variable (f : ℝ → ℝ)
variable [Dom_f : Set.Icc 1 3 → ℝ]

theorem domain_transformation : (Set.Icc 0 1 → ℝ) :=
sorry

end domain_transformation_l765_765060


namespace count_two_digit_primes_ending_in_3_l765_765372

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765372


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765399

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765399


namespace largest_among_all_l765_765492

noncomputable def y := 10^(-1998 : ℝ)

def exprA := 4 + y
def exprB := 4 - y
def exprC := 4 * y
def exprD := 4 / y
def exprE := (4 / y)^2

theorem largest_among_all : 
  exprE >= max (max exprA exprB) (max exprC exprD) := 
by
  sorry

end largest_among_all_l765_765492


namespace max_length_MN_l765_765893

theorem max_length_MN (ABCD : Type) [convex_quadrilateral ABCD] (AD BC : ℝ)
  (M N : ABCD → ℝ) (h_parallel_AD_BC : is_parallel AD BC)
  (h_parallel_L_AD : ∃ L, is_parallel L AD ∧ (∃ M N, intersection_L_AB M ∧ intersection_L_CD N))
  (h_similar : similar_quadrilaterals (AMND L) (MBNC L)) :
  AD + BC ≤ 4 → max_MN MN 2 :=
by sorry

end max_length_MN_l765_765893


namespace one_fourth_of_8_point_8_is_fraction_l765_765000

theorem one_fourth_of_8_point_8_is_fraction:
  (1 / 4) * 8.8 = 11 / 5 :=
by sorry

end one_fourth_of_8_point_8_is_fraction_l765_765000


namespace unique_intersection_point_l765_765786

def line1 (x y : ℝ) : Prop := 3 * x + 2 * y = 9
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 10
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1
def line5 (x y : ℝ) : Prop := x + y = 4

theorem unique_intersection_point :
  ∃! (p : ℝ × ℝ), 
     line1 p.1 p.2 ∧ 
     line2 p.1 p.2 ∧ 
     line3 p.1 ∧ 
     line4 p.2 ∧ 
     line5 p.1 p.2 :=
sorry

end unique_intersection_point_l765_765786


namespace find_pairs_l765_765753

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l765_765753


namespace count_prime_units_digit_3_eq_6_l765_765294

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765294


namespace trapezoid_count_l765_765032

-- Define the necessary entities and conditions
structure DihedralAngle (α β : Type) : Type :=
(PQ : Set α × Set β)
(A B : α)
(C : β)
(not_on_PQ_A : A ∉ PQ.fst)
(not_on_PQ_B : B ∉ PQ.fst)
(not_on_PQ_C : C ∉ PQ.snd)
(dist_A_PQ : Real)
(dist_B_PQ : Real)
(dist_NE : dist_A_PQ ≠ dist_B_PQ)

-- Define a predicate for trapezoid existence under given conditions
def trapezoid_exists (d : DihedralAngle α β) : Prop :=
exists D : α × β, -- There exists a point D
  (D ∈ d.PQ ∧ D ≠ d.A ∧ D ≠ d.B ∧ D ≠ d.C) -- such that D is valid fourth vertex

-- State the main theorem
theorem trapezoid_count (d : DihedralAngle α β) : ∃! D, trapezoid_exists d :=
by
  sorry

end trapezoid_count_l765_765032


namespace rabbit_toy_cost_l765_765920

theorem rabbit_toy_cost 
  (cost_pet_food : ℝ) 
  (cost_cage : ℝ) 
  (found_dollar : ℝ)
  (total_cost : ℝ) 
  (h1 : cost_pet_food = 5.79) 
  (h2 : cost_cage = 12.51)
  (h3 : found_dollar = 1.00)
  (h4 : total_cost = 24.81):
  ∃ (cost_rabbit_toy : ℝ), cost_rabbit_toy = 7.51 := by
  let cost_rabbit_toy := total_cost - (cost_pet_food + cost_cage) + found_dollar
  use cost_rabbit_toy
  sorry

end rabbit_toy_cost_l765_765920


namespace find_m_and_n_l765_765823

def A : set ℝ := { x : ℝ | abs (x + 2) < 3 }
def B (m : ℝ) : set ℝ := { x : ℝ | (x - m) * (x - 2) < 0 }

theorem find_m_and_n (m n : ℝ) (hm : set.inter A (B m) = set.Ioo (-1) n) : 
  m = -1 ∧ n = 1 := 
by 
  sorry

end find_m_and_n_l765_765823


namespace students_play_football_l765_765511

theorem students_play_football (total_students : ℕ) (C : ℕ) (B : ℕ) (neither : ℕ) (F : ℕ)
  (h1 : total_students = 460)
  (h2 : C = 175)
  (h3 : B = 90)
  (h4 : neither = 50)
  (h5 : total_students = neither + F + C - B) : 
  F = 325 :=
by 
  sorry

end students_play_football_l765_765511


namespace factor_polynomial_int_l765_765749

theorem factor_polynomial_int : 
  ∀ x : ℤ, 5 * (x + 3) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 = 
           (5 * x^2 + 81 * x + 315) * (x^2 + 16 * x + 213) := 
by
  intros
  norm_num
  sorry

end factor_polynomial_int_l765_765749


namespace smallest_positive_period_pi_center_of_symmetry_value_of_cos_2x0_l765_765932

def f (x : ℝ) : ℝ := cos (2 * x - π / 6) - sqrt 3 * cos (2 * x) - 1 / 2

-- The smallest positive period of f(x) is π
theorem smallest_positive_period_pi :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

-- The center of symmetry is (kπ/2 + π/6, -1/2) for k ∈ ℤ
theorem center_of_symmetry (k : ℤ) :
  ∃ cₓ cᵧ, ∀ x, f (cₓ + cᵧ - x) = f x ∧ (cₓ, cᵧ) = (k * π / 2 + π / 6, -1 / 2) :=
sorry

-- Given x0 ∈ [5π/12, 2π/3] and f(x0) = sqrt(3)/3 - 1/2, the value of cos(2x0) is -3+sqrt(6)/6
theorem value_of_cos_2x0 (x0 : ℝ) (hx0 : x0 ∈ set.Icc (5 * π / 12) (2 * π / 3)) (hf : f x0 = sqrt 3 / 3 - 1 / 2) :
  cos (2 * x0) = -(3 + sqrt 6) / 6 :=
sorry

end smallest_positive_period_pi_center_of_symmetry_value_of_cos_2x0_l765_765932


namespace solve_inequality_system_l765_765561

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765561


namespace count_two_digit_prime_numbers_ending_in_3_l765_765251

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765251


namespace solve_math_problem_l765_765725

noncomputable def calculate_expr (a n : ℝ) : ℝ :=
  let c1 := a⁻¹ := 1 / (a ^ n)
  let c2 := log 2 2 := 1
  let c3 := log 2 := n * log 2 a
  (8 ^ (-2 / 3)) + 2 * log 2 2 - log 2 (1 / 25) = 9 / 4

theorem solve_math_problem : calculate_expr 8 (-2 / 3) = 9 / 4 :=
sorry

end solve_math_problem_l765_765725


namespace smallest_possible_positive_value_l765_765863

theorem smallest_possible_positive_value (a b : ℤ) (h : a > b) :
  ∃ (x : ℚ), x = (a + b) / (a - b) + (a - b) / (a + b) ∧ x = 2 :=
sorry

end smallest_possible_positive_value_l765_765863


namespace average_speed_l765_765710

theorem average_speed (x : ℝ) (h1 : x > 0) :
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  avg_speed = 200 / 9 :=
by
  -- Definitions
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  -- Proof structure, concluding with the correct answer.
  sorry

end average_speed_l765_765710


namespace michael_combinations_l765_765906

-- Conditions defined as variables
variables (n k : ℕ)

-- The combination formula
def combination (n k : ℕ) : ℕ := n.choose k

-- The specific problem instance
theorem michael_combinations : combination 8 3 = 56 := by
  sorry

end michael_combinations_l765_765906


namespace h_odd_function_f_increasing_solve_inequality_l765_765807

open Real

section
variables {f : ℝ → ℝ}

/- Given Conditions: -/
axiom functional_eq : ∀ x y : ℝ, f(x) + f(y) = f(x + y) + 1
axiom pos_condition : ∀ x : ℝ, x > 0 → f(x) > 1

/- Definitions: -/
def h (x : ℝ) : ℝ := f(x) - 1

/- 1. h(x) is an odd function -/
theorem h_odd_function : ∀ x : ℝ, h(-x) = -h(x) :=
sorry

/- 2. f(x) is increasing on ℝ -/
theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ > x₂ → f(x₁) > f(x₂) :=
sorry

/- 3. Solve the inequality f(x^2) - f(3tx) + f(2t^2 + 2t - x) < 1 for x, where t ∈ ℝ -/
theorem solve_inequality (t x : ℝ) :
  (t > 1 → t + 1 < x ∧ x < 2 * t) ∧ 
  (t < 1 → 2 * t < x ∧ x < t + 1) :=
sorry

end

end h_odd_function_f_increasing_solve_inequality_l765_765807


namespace count_two_digit_primes_with_units_digit_three_l765_765410

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765410


namespace player_B_prevents_player_A_from_winning_l765_765514

theorem player_B_prevents_player_A_from_winning :
  (∃ (board : fin 8 × fin 8 → option color), 
    ∀ turn : ℕ, 
      (turn % 2 = 0 → ∃ i j, board (fin.mk i sorry, fin.mk j sorry) = none) ∧  
      (turn % 2 = 1 → ∃ i j, board (fin.mk i sorry, fin.mk j sorry) = none) ∧ 
      (∀ i j, (∃ i' j', i' ≠ i ∧ j' ≠ j ∧ board (fin.mk i sorry, fin.mk j sorry) = some red ∧
        board (fin.mk (i+1) sorry, fin.mk j sorry) = some red ∧
        board (fin.mk i sorry, fin.mk (j+1) sorry) = some red ∧
        board (fin.mk (i+1) sorry, fin.mk (j+1) sorry) = some red) → false)) :=
sorry

end player_B_prevents_player_A_from_winning_l765_765514


namespace count_two_digit_primes_with_units_digit_three_l765_765413

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765413


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765393

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765393


namespace range_of_a_l765_765090

open Real

-- Definitions of the propositions p and q
def p (a : ℝ) : Prop := (2 - a > 0) ∧ (a + 1 > 0)

def discriminant (a : ℝ) : ℝ := 16 + 4 * a

def q (a : ℝ) : Prop := discriminant a ≥ 0

/--
Given propositions p and q defined above,
prove that the range of real number values for a 
such that ¬p ∧ q is true is
- 4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2
--/
theorem range_of_a (a : ℝ) : (¬ p a ∧ q a) → (-4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l765_765090


namespace arithmetic_sequence_minimum_l765_765043

variables {a_1 d : ℝ} {n : ℕ}
def S_n (a_1 d : ℝ) (n : ℕ) : ℝ := d / 2 * n^2 + (a_1 - d / 2) * n

theorem arithmetic_sequence_minimum (h_a1 : a_1 < 0) (h_d : d > 0) : ∃ m : ℝ, ∀ n : ℕ, S_n a_1 d n ≥ m :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_minimum_l765_765043


namespace line_KL_bisects_segment_AJ_l765_765493

variables {A B C D K L J : Type*}
variables [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq K] [DecidableEq L] [DecidableEq J]
variables [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty K] [Nonempty L] [Nonempty J]

-- Assume a triangle ABC such that \(\angle C < \angle A < 90^\circ\)
variable (triangle_ABC : Triangle A B C)
variable (angle_C_lt_angle_A : ∠C < ∠A)
variable (angle_A_lt_90 : ∠A < 90)

-- Let \( D \) be a point on segment \([AC]\) such that \( BD = BA \)
variable (D_on_segment_AC : is_on_segment D A C)
variable (BD_eq_BA : dist B D = dist B A)

-- Points K and L are the contact points of the incircle of \(\triangle ABC\) with sides \([AB]\) and \([AC]\)
variable (incircle_contact_K : incircle_contact_point (incircle triangle_ABC) K A B)
variable (incircle_contact_L : incircle_contact_point (incircle triangle_ABC) L A C)

-- J is the incenter of \(\triangle BCD\)
variable (J_is_incenter : incenter J B C D)

-- We need to show that the line \( (KL) \) bisects the segment \([AJ]\)
theorem line_KL_bisects_segment_AJ :
  bisects (line_through K L) (segment A J) := 
sorry

end line_KL_bisects_segment_AJ_l765_765493


namespace solve_inequality_system_l765_765533

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765533


namespace find_remainder_l765_765418

def f : ℕ → ℕ
| 2 := 1
| 3 := 6
| n := (Nat.choose n 2) * (2 * f (n - 1) + (n - 1) * f (n - 2))

def N : ℕ := f 8

theorem find_remainder :
  N % 1000 = 530 := 
sorry

end find_remainder_l765_765418


namespace two_digit_primes_with_units_digit_three_l765_765225

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765225


namespace probability_sum_at_least_15_l765_765887

-- Define the total number of balls
def num_balls : ℕ := 8

-- Define the valid outcomes summing to at least 15
def valid_outcomes : List (ℕ × ℕ) := [(7, 8), (8, 7), (8, 8)]

-- Calculate the probability
def probability := (valid_outcomes.length : ℚ) / (num_balls * num_balls)

-- Define the theorem to be proved
theorem probability_sum_at_least_15 : probability = 3 / 64 := by
  sorry

end probability_sum_at_least_15_l765_765887


namespace double_root_quadratic_eq_l765_765819

theorem double_root_quadratic_eq (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : ∃ x, x^2 + 4 * x * real.cos θ + real.cot θ = 0 ∧
  ∀ y, y ≠ x → y^2 + 4 * y * real.cos θ + real.cot θ ≠ 0) :
  θ = π / 12 ∨ θ = 5 * π / 12 :=
sorry

end double_root_quadratic_eq_l765_765819


namespace solve_inequality_system_l765_765558

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765558


namespace count_two_digit_primes_with_units_digit_3_l765_765151

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765151


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765265

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765265


namespace smallest_positive_multiple_l765_765646

theorem smallest_positive_multiple : ∃ (a : ℕ), 32 * a = 2528 ∧ 32 * a % 97 = 6 :=
by
  existsi (2528 / 32)
  split
  case a_left => sorry
  case a_right => sorry

end smallest_positive_multiple_l765_765646


namespace percent_first_question_l765_765674

variable (A B : ℝ) (A_inter_B : ℝ) (A_union_B : ℝ)

-- Given conditions
def condition1 : B = 0.49 := sorry
def condition2 : A_inter_B = 0.32 := sorry
def condition3 : A_union_B = 0.80 := sorry
def union_formula : A_union_B = A + B - A_inter_B := 
by sorry

-- Prove that A = 0.63
theorem percent_first_question (h1 : B = 0.49) 
                               (h2 : A_inter_B = 0.32) 
                               (h3 : A_union_B = 0.80) 
                               (h4 : A_union_B = A + B - A_inter_B) : 
                               A = 0.63 :=
by sorry

end percent_first_question_l765_765674


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765271

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765271


namespace two_digit_primes_with_units_digit_three_count_l765_765173

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765173


namespace count_two_digit_primes_with_units_digit_3_l765_765190

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765190


namespace two_digit_primes_with_units_digit_three_l765_765236

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765236


namespace problem_statement_l765_765061

-- Definitions for even function and conditions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Main theorem statement
theorem problem_statement (a b : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, x^2 + b * x)
  (h2 : ∀ x, x ∈ [-2*a, 3*a-1] → true)  -- Assuming ∀x within the interval is true for the domain
  (h3 : is_even_function f) : a + b = 1 :=
begin
  sorry
end

end problem_statement_l765_765061


namespace probability_Y_greater_X_uniform_l765_765711

theorem probability_Y_greater_X_uniform (x y : ℝ) (hx : x ∈ Set.Icc 0 4000) (hy : y ∈ Set.Icc 0 6000) :
  (∫ x in 0..4000, ∫ y in x..6000, 1 / (4000 * 6000) = (2 / 3)) :=
by
  -- We will defer the proof to another part
  sorry

end probability_Y_greater_X_uniform_l765_765711


namespace num_unique_permutations_l765_765104

-- Given: digits 3 and 8 being repeated as described.
-- Show: number of different permutations of the digits 3, 3, 3, 8, 8 is 10.

theorem num_unique_permutations : 
  let digits := [3, 3, 3, 8, 8] in
  let total_permutations := (5!).nat_abs in             -- 5! permutations
  let repeats_correction := (3!).nat_abs * (2!).nat_abs in -- Adjusting for repeated 3's and 8's
  let unique_permutations := total_permutations / repeats_correction in
  unique_permutations = 10 :=
by
  sorry

end num_unique_permutations_l765_765104


namespace max_n_points_l765_765739

theorem max_n_points (n : ℕ) (A : Fin n → (ℝ × ℝ)) (r : Fin n → ℝ) 
  (h1 : ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → ¬ collinear (A i) (A j) (A k)) 
  (h2 : ∀ (i j k : Fin n), i < j → j < k → area_of_triangle (A i) (A j) (A k) = r i + r j + r k) : 
  n ≤ 4 := 
sorry

end max_n_points_l765_765739


namespace solve_system_of_equations_l765_765588

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + x * y = 15) (h2 : x^2 + x * y = 10) :
  (x = 2 ∧ y = 3) ∨ (x = -2 ∧ y = -3) :=
sorry

end solve_system_of_equations_l765_765588


namespace f_has_one_zero_point_f_has_three_zero_points_l765_765844

theorem f_has_one_zero_point (k : ℝ) 
  (f : ℝ → ℝ := λ x, 2 * x ^ 3 - 6 * x + k)
  (f' : ℝ → ℝ := λ x, 6 * x ^ 2 - 6) :
  ((k < -4) ∨ (k > 4)) ↔ (∃! x : ℝ, f x = 0) :=
by sorry

theorem f_has_three_zero_points (k : ℝ) 
  (f : ℝ → ℝ := λ x, 2 * x ^ 3 - 6 * x + k)
  (f' : ℝ → ℝ := λ x, 6 * x ^ 2 - 6) :
  ((-4 < k) ∧ (k < 4)) ↔ (∃ a b c : ℝ, f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
by sorry

end f_has_one_zero_point_f_has_three_zero_points_l765_765844


namespace cookies_per_day_l765_765017

theorem cookies_per_day (cost_per_cookie : ℕ) (total_spent : ℕ) (days_in_march : ℕ) (h1 : cost_per_cookie = 16) (h2 : total_spent = 992) (h3 : days_in_march = 31) :
  (total_spent / cost_per_cookie) / days_in_march = 2 :=
by sorry

end cookies_per_day_l765_765017


namespace number_of_two_digit_primes_with_units_digit_three_l765_765209

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765209


namespace curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l765_765074

-- Define the equation of the curve C
def curve_C (a x y : ℝ) : Prop := a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

-- Prove that curve C is a circle
theorem curve_C_is_circle (a : ℝ) (h : a ≠ 0) :
  ∃ (h_c : ℝ), ∃ (k : ℝ), ∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), curve_C a x y ↔ (x - h_c)^2 + (y - k)^2 = r^2
:= sorry

-- Prove that the area of triangle AOB is constant
theorem area_AOB_constant (a : ℝ) (h : a ≠ 0) :
  ∃ (A B : ℝ × ℝ), (A = (2 * a, 0) ∧ B = (0, 4 / a)) ∧ 1/2 * (2 * a) * (4 / a) = 4
:= sorry

-- Find valid a and equation of curve C given conditions of line l and points M, N
theorem find_valid_a_and_curve_eq (a : ℝ) (h : a ≠ 0) :
  ∀ (M N : ℝ × ℝ), (|M.1 - 0| = |N.1 - 0| ∧ |M.2 - 0| = |N.2 - 0|) → (M.1 = N.1 ∧ M.2 = N.2) →
  y = -2 * x + 4 →  a = 2 ∧ ∀ (x y : ℝ), curve_C 2 x y ↔ x^2 + y^2 - 4 * x - 2 * y = 0
:= sorry

end curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l765_765074


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765398

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765398


namespace count_two_digit_primes_ending_in_3_l765_765368

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765368


namespace value_of_a_l765_765428

theorem value_of_a (a : ℝ) (k : ℝ) (hA : -5 = k * 3) (hB : a = k * (-6)) : a = 10 :=
by
  sorry

end value_of_a_l765_765428


namespace first_player_wins_if_ne_second_player_wins_if_eq_l765_765888

theorem first_player_wins_if_ne (m n : ℕ) (h : m ≠ n) : 
  ∃ (winning_strategy : strategy) , (first_player_wins winning_strategy m n) :=
sorry

theorem second_player_wins_if_eq (m n : ℕ) (h : m = n) : 
  ∃ (winning_strategy : strategy) , (second_player_wins winning_strategy m n) :=
sorry

end first_player_wins_if_ne_second_player_wins_if_eq_l765_765888


namespace successfully_served_pizzas_l765_765699

theorem successfully_served_pizzas :
  ∀ (total_pizzas returned_pizzas : ℕ) (redistribution_rate : ℝ),
    total_pizzas = 9 →
    returned_pizzas = 5 →
    redistribution_rate = 0.4 →
    let successfully_redistributed := (redistribution_rate * returned_pizzas : ℝ).to_nat in
    let successfully_served_not_returned := total_pizzas - returned_pizzas in
    successfully_served_not_returned + successfully_redistributed = 6 :=
by
  intros total_pizzas returned_pizzas redistribution_rate ht hr hsrr
  let successfully_redistributed := (redistribution_rate * returned_pizzas : ℝ).to_nat
  let successfully_served_not_returned := total_pizzas - returned_pizzas
  have : successfully_served_not_returned + successfully_redistributed = 6 := by sorry
  exact this

end successfully_served_pizzas_l765_765699


namespace pencils_total_l765_765617

/-- The students in class 5A had a total of 2015 pencils. One of them lost a box containing five pencils and replaced it with a box containing 50 pencils. Prove the final number of pencils is 2060. -/
theorem pencils_total {initial_pencils lost_pencils gained_pencils final_pencils : ℕ} 
  (h1 : initial_pencils = 2015) 
  (h2 : lost_pencils = 5) 
  (h3 : gained_pencils = 50) 
  (h4 : final_pencils = (initial_pencils - lost_pencils + gained_pencils)) 
  : final_pencils = 2060 :=
sorry

end pencils_total_l765_765617


namespace divisibility_999x999_tromino_l765_765964

def L_tromino (n : ℕ) : Prop :=
  ∃ m, m = n^2 ∧ n % 3 = 0

theorem divisibility_999x999_tromino :
  L_tromino 999 → (999 * 999) / 3 % 8 = 0 :=
begin
  sorry
end

end divisibility_999x999_tromino_l765_765964


namespace base_10_to_base_7_l765_765634

theorem base_10_to_base_7 (n : ℤ) (h : n = 947) : 
  (2 * 7^3 + 5 * 7^2 + 2 * 7^1 + 2 * 7^0 = n) := by
  have h_base : 947 = 2 * 7^3 + 5 * 7^2 + 2 * 7^1 + 2 * 7^0 := sorry
  exact h_base

end base_10_to_base_7_l765_765634


namespace number_of_two_digit_primes_with_units_digit_three_l765_765206

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765206


namespace complex_number_quadrant_l765_765995

def z : ℂ := (↑complex.I) / (1 + ↑complex.I)

theorem complex_number_quadrant : z.re > 0 ∧ z.im > 0 := sorry

end complex_number_quadrant_l765_765995


namespace count_two_digit_prime_numbers_ending_in_3_l765_765254

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765254


namespace range_of_x_l765_765820

variable (x : ℝ)

def p : Prop := (real.log (x^2 - 2*x - 2) ≥ 0)
def q : Prop := (0 < x) ∧ (x < 4)

theorem range_of_x (hp : p x) (hq : ¬ q x) : x ≥ 4 ∨ x ≤ -1 :=
by
  sorry

end range_of_x_l765_765820


namespace count_prime_units_digit_3_eq_6_l765_765292

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765292


namespace parallel_vectors_l765_765092

theorem parallel_vectors (m : ℝ) :
  let a := (1, 2)
  let b := (-2, m)
  (∃ k : ℝ, a = k • b) → m = -4 := 
begin
  assume h,
  sorry
end

end parallel_vectors_l765_765092


namespace two_digit_primes_with_units_digit_three_count_l765_765180

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765180


namespace base10_to_base7_l765_765636

theorem base10_to_base7 (n : ℕ) (h : n = 947) : ∃ (a b c d : ℕ), 2522 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 :=
by {
  sorry,
}

end base10_to_base7_l765_765636


namespace divisors_sum_eq_self_l765_765483

theorem divisors_sum_eq_self (n a b : ℕ) (h1 : n ≥ 1) (h2 : a > 0) (h3 : b > 0) (h4 : a ∣ n) (h5 : b ∣ n) (h6 : a + b + a * b = n) : a = b := 
by
  have sorry : a = b
  sorry

end divisors_sum_eq_self_l765_765483


namespace two_digit_primes_units_digit_3_count_l765_765354

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765354


namespace problem1_problem2_l765_765832
noncomputable section

-- Problem 1
theorem problem1 (A B : ℝ) (h : A ∈ Ioo 0 (π / 2)) (g : B ∈ Ioo 0 (π / 2)) 
    (m : ℝ×ℝ) (n : ℝ×ℝ) (h1 : m = (Real.cos (A + π/3), Real.sin (A + π/3)))
    (h2 : n = (Real.cos B, Real.sin B))
    (h3 : m.1 * n.1 + m.2 * n.2 = 0) : A - B = π/6 := sorry

-- Problem 2
theorem problem2 (B : ℝ) (AC : ℝ) (h : B ∈ Ioo 0 (π / 2)) 
    (h1 : Real.cos B = 3 / 5) (h2 : AC = 8) : 
    let A := B + π/6 in 
    let sin_A := (4 * Real.sqrt 3 + 3) / 10 in
    let sin_B := (4 / 5) in 
    BC = sin_A / sin_B * AC := sorry

end problem1_problem2_l765_765832


namespace largest_difference_is_56000_l765_765731

-- Definitions for the problem
def C_estimate : ℕ := 100000
def D_estimate : ℕ := 120000
def C_min : ℕ := 85000
def C_max : ℕ := 115000
def D_min : ℕ := 104348
def D_max : ℕ := 141176

-- Define the largest possible difference rounded to the nearest 1000
noncomputable def largest_possible_difference : ℕ :=
  let difference := D_max - C_min in
  ((difference + 500) / 1000) * 1000

-- Statement of the problem
theorem largest_difference_is_56000 :
  largest_possible_difference = 56000 :=
sorry

end largest_difference_is_56000_l765_765731


namespace count_two_digit_primes_with_units_digit_3_l765_765152

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765152


namespace smallest_lcm_example_l765_765424

noncomputable def smallest_lcm (a b : ℕ) : ℕ :=
  if h : a > 999 ∧ a < 10000 ∧ b > 999 ∧ b < 10000 ∧ gcd a b = 5 then
    Nat.lcm a b
  else 0

theorem smallest_lcm_example :
  smallest_lcm 1005 1010 = 203010 :=
by
  unfold smallest_lcm
  split_ifs
  · simp [h]
  · contradiction

end smallest_lcm_example_l765_765424


namespace count_two_digit_primes_with_units_digit_3_l765_765199

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765199


namespace count_prime_units_digit_3_eq_6_l765_765298

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765298


namespace vera_operations_impossible_l765_765035

theorem vera_operations_impossible (N : ℕ) : (N % 3 ≠ 0) → ¬(∃ k : ℕ, ((N + 3 * k) % 5 = 0) → ((N + 3 * k) / 5) = 1) :=
by
  sorry

end vera_operations_impossible_l765_765035


namespace inequality_solution_l765_765541

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l765_765541


namespace convergence_iff_b_eq_2_l765_765016

noncomputable def f (b n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else
    let d := Nat.digits b n |>.length
    in n * f b d

def sum_reciprocal_f (b : ℕ) : ℝ :=
  Real.log (b:ℝ)

theorem convergence_iff_b_eq_2 : ∀ b : ℕ, b ≥ 2 → (summable (λ n : ℕ, 1 / (f b n).toReal) ↔ b = 2) :=
by
  intros
  sorry

end convergence_iff_b_eq_2_l765_765016


namespace one_fourth_of_8_point_8_l765_765003

-- Definition of taking one fourth of a number
def oneFourth (x : ℝ) : ℝ := x / 4

-- Problem statement: One fourth of 8.8 is 11/5 when expressed as a simplified fraction
theorem one_fourth_of_8_point_8 : oneFourth 8.8 = 11 / 5 := by
  sorry

end one_fourth_of_8_point_8_l765_765003


namespace solve_inequality_system_l765_765569

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765569


namespace max_consecutive_odd_sequence_l765_765900

def largest_digit (n : ℕ) : ℕ :=
  n.digits.max'

def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + largest_digit (a n)

def odd_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n % 2 = 1

theorem max_consecutive_odd_sequence :
  ∀ a : ℕ → ℕ,
  sequence a →
  odd_sequence a →
  ∃ N ≤ 5, ∀ m < N, odd_sequence (a ∘ (m + .))
  ∧ ¬odd_sequence (a ∘ ((N + 1) + .)) :=
sorry

end max_consecutive_odd_sequence_l765_765900


namespace moles_of_HCl_l765_765128

theorem moles_of_HCl (moles_C5H12O : ℕ) (moles_C5H11Cl  : ℕ) (moles_H2O : ℕ) 
  (h_eq : ("C5H12O" + "HCl" → "C5H11Cl" + "H2O")) : 
  (moles_C5H12O = 2) → (moles_C5H11Cl = 2) → (moles_H2O = 2) → 
  (∃ moles_HCl : ℕ, moles_HCl = 2) :=
by
  intros h1 h2 h3
  use 2
  sorry

end moles_of_HCl_l765_765128


namespace solve_inequality_system_l765_765567

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765567


namespace diameter_of_inscribed_circle_l765_765529

theorem diameter_of_inscribed_circle (a b c r : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_radius : r = (a + b - c) / 2) : 
  2 * r = a + b - c :=
by
  sorry

end diameter_of_inscribed_circle_l765_765529


namespace base10_to_base7_l765_765637

theorem base10_to_base7 (n : ℕ) (h : n = 947) : ∃ (a b c d : ℕ), 2522 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 :=
by {
  sorry,
}

end base10_to_base7_l765_765637


namespace expected_adjacent_black_pairs_l765_765449

theorem expected_adjacent_black_pairs (total_cards : ℕ) (black_cards : ℕ) (red_cards : ℕ) (is_circle : Prop)
  (h_total : total_cards = 60) (h_black_red : black_cards = 30 ∧ red_cards = 30) :
  (30 * (29 / 59) = 870 / 59) :=
begin
  sorry
end

end expected_adjacent_black_pairs_l765_765449


namespace count_prime_units_digit_3_eq_6_l765_765300

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765300


namespace two_digit_primes_with_units_digit_three_count_l765_765168

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765168


namespace percent_of_y_l765_765661

theorem percent_of_y (y : ℝ) (hy : y > 0) : (8 * y) / 20 + (3 * y) / 10 = 0.7 * y :=
by
  sorry

end percent_of_y_l765_765661


namespace shift_left_4_l765_765606

-- Define the functions in the conditions
def f (x : ℝ) : ℝ := 10^(x - 1)
def g (x : ℝ) : ℝ := 10^(x + 3)

-- State that shifting f 4 units to the left gives g
theorem shift_left_4 (x : ℝ) : g x = f (x + 4) :=
by
  -- Since no proof steps are required, we use sorry.
  sorry

end shift_left_4_l765_765606


namespace num_two_digit_primes_with_units_digit_three_l765_765142

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765142


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765280

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765280


namespace geometric_probability_segment_l765_765034

theorem geometric_probability_segment :
  let length_total := 3
  let length_event := 1
  let probability := length_event / length_total
  probability = (1 / 3) :=
by
  -- calculation of the probability
  let interval := [0, 3]
  let event_interval := [1, 2]
  let length_total := interval.end - interval.start
  let length_event := event_interval.end - event_interval.start
  let probability := length_event / length_total
  show probability = (1 / 3)
  sorry

end geometric_probability_segment_l765_765034


namespace distinctPermutations_test_l765_765100

noncomputable def distinctPermutationsCount (s : Multiset ℕ) : ℕ :=
  (s.card.factorial) / (s.count 3.factorial * s.count 8.factorial)

theorem distinctPermutations_test : distinctPermutationsCount {3, 3, 3, 8, 8} = 10 := by
  sorry

end distinctPermutations_test_l765_765100


namespace problem_inequality_l765_765480

theorem problem_inequality (a : ℤ) (h_a : 0 < a) : 
  ∀ x y : ℤ, (x * (y^2 - 2 * x^2) + x + y + a = 0) → 
  abs x ≤ a + real.sqrt (2 * a^2 + 2) :=
by
  sorry

end problem_inequality_l765_765480


namespace find_value_of_m_l765_765837

-- Definitions based on given conditions
def ellipse_eq (x y : ℝ) (a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (ellipse_eq : ℝ → ℝ → Bool) (p : ℝ × ℝ) : Prop :=
  ellipse_eq p.1 p.2

def eccentricity (a c : ℝ) := c / a

-- The main theorem statement
theorem find_value_of_m :
  ∀ (a b c : ℝ) (m k : ℝ),
    passes_through (ellipse_eq x y a b) (0, -1) →
    eccentricity a c = ( √2 / 2 ) →
    a^2 = b^2 + c^2 →
    b = 1 →
    l_bisects_angle (ellipse_eq x y a b) (P m 0) (1, 0) k →
    k ≠ 0 →
    m = 2 :=
sorry

end find_value_of_m_l765_765837


namespace number_of_two_digit_primes_with_units_digit_three_l765_765205

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765205


namespace find_m_l765_765942

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 3 else x^2 - 2 * x

theorem find_m (m : ℝ) (h : f m = 3) : m = 0 ∨ m = 3 :=
  sorry

end find_m_l765_765942


namespace factorization_l765_765750

noncomputable def poly := 9 * (Polynomial.X + 3) * (Polynomial.X + 7) * (Polynomial.X + 11) * (Polynomial.X + 13) - 4 * Polynomial.X^2
noncomputable def factor1 := 3 * Polynomial.X^2 + 52 * Polynomial.X + 231
noncomputable def factor2 := 3 * Polynomial.X^2 + 56 * Polynomial.X + 231

theorem factorization : poly = factor1 * factor2 :=
by sorry

end factorization_l765_765750


namespace log_div_simplify_l765_765650

theorem log_div_simplify :
  log 2 16 / log 2 (1 / 16) = -1 := by
  sorry

end log_div_simplify_l765_765650


namespace chord_DE_bisects_BC_l765_765456

variable (O : Type) [metric_space O]
variables {A B C D E : O}
variables [circle O]
variables [chord CD AB AE] (r : real)

-- Conditions
axiom CD_perpendicular_to_AB : CD ⊥ AB
axiom AE_bisects_OC : AE bisects OC

-- Conclusion to prove
theorem chord_DE_bisects_BC :
  DE bisects BC :=
sorry

end chord_DE_bisects_BC_l765_765456


namespace hypotenuse_and_exponential_l765_765610

noncomputable def log_9_64 : ℝ := Real.log 64 / Real.log 9
noncomputable def log_3_16 : ℝ := Real.log 16 / Real.log 3 -- using properties of log

theorem hypotenuse_and_exponential (a b : ℝ) (hyp : a = log_9_64) (hypb : b = log_3_16) :
  let k := Real.sqrt (a^2 + b^2),
  9^k = 1024 :=
by
  sorry

end hypotenuse_and_exponential_l765_765610


namespace range_of_k_l765_765881

theorem range_of_k (k b : ℝ) 
  (h1 : ∃ b, y = k * x + b)
  (h2 : (2 : ℝ, 2 : ℝ) ∈ graph (λ x, k * x + b))
  (h3 : ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ (k * x + b) = (-x + 3)) :
  k ≤ -2 ∨ (k ≥ - (1 / 2) ∧ k ≠ 0) :=
  sorry

end range_of_k_l765_765881


namespace Segment_XX_l765_765958

-- Defining the conditions
variable {X X' : Type}
variable {l l' : X → Prop}
variable {t : ℝ}
variables {v v' : ℝ} (hv : v ≠ v')

-- The mathematical equivalent proof problem in Lean 4
theorem Segment_XX'_Locus (hv : v ≠ v') : 
  let X t := (X 0) + t * v
      X' t := (X' 0) + t * v' in
  (fun t : ℝ, (X t, X' t)) = (problem31_067_locus t) :=
sorry

end Segment_XX_l765_765958


namespace value_of_nested_expression_l765_765014

def nested_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2

theorem value_of_nested_expression : nested_expression = 1457 := by
  sorry

end value_of_nested_expression_l765_765014


namespace count_digit_9_is_60_l765_765859

-- Define the function to count occurrences of a specific digit in a given range
def count_digit_occurrences (d : ℕ) (start : ℕ) (end : ℕ) : ℕ :=
  (List.range (end - start + 1)).count (λ n, (n + start).digits 10).count (λ x, x = d)

-- Define the target number and range
def digit_9 := 9
def range_start := 1
def range_end := 300

-- Calculate separate occurrences in unit, tens, and hundreds places
def count_9_in_units_place : ℕ := 30
def count_9_in_tens_place : ℕ := 30
def count_9_in_hundreds_place : ℕ := 0

-- Total occurrences
def total_count_9 := count_9_in_units_place + count_9_in_tens_place + count_9_in_hundreds_place

-- Theorem to be proved
theorem count_digit_9_is_60 : 
  count_digit_occurrences digit_9 range_start range_end = total_count_9 :=
  by
  sorry

end count_digit_9_is_60_l765_765859


namespace natural_number_pairs_int_l765_765774

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l765_765774


namespace cube_volume_l765_765983

theorem cube_volume (d : ℝ) (s : ℝ) (h : d = 3 * Real.sqrt 3) (h_s : s * Real.sqrt 3 = d) : s ^ 3 = 27 := by
  -- Assuming h: the formula for the given space diagonal
  -- Assuming h_s: the formula connecting side length and the space diagonal
  sorry

end cube_volume_l765_765983


namespace num_two_digit_primes_with_units_digit_three_l765_765143

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765143


namespace area_ADC_proof_l765_765461

-- Definitions for the given conditions and question
variables (BD DC : ℝ) (ABD_area ADC_area : ℝ)

-- Conditions
def ratio_condition := BD / DC = 3 / 2
def ABD_area_condition := ABD_area = 30

-- Question rewritten as proof problem
theorem area_ADC_proof (h1 : ratio_condition BD DC) (h2 : ABD_area_condition ABD_area) :
  ADC_area = 20 :=
sorry

end area_ADC_proof_l765_765461


namespace num_integers_digit_sum_18_400_600_l765_765120

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem num_integers_digit_sum_18_400_600 : 
  {n : ℕ // 400 ≤ n ∧ n < 600 ∧ digit_sum n = 18}.card = 21 := sorry

end num_integers_digit_sum_18_400_600_l765_765120


namespace locus_of_X_l765_765045

-- Declaration of fixed points A, B, C, and D on a circle
variables (A B C D P X : Type) [PointsOnCircle A B C D]

-- Declaration of circumcircles of triangles XAB and XCD
def circumcircle_XAB (X : Type) : Type := circumcircle (X, A, B)
def circumcircle_XCD (X : Type) : Type := circumcircle (X, C, D)

-- Declaration for Radical Axis Theorem result
def radical_axis_tangent (X : Type) : Prop :=
  tangent_at (circumcircle_XAB X) X ∧ tangent_at (circumcircle_XCD X) X

-- The point P is the radical center of the circumcircles
variable [RadicalCenter (circumcircle_XAB X) (circumcircle_XCD X) P]

-- The statement that the locus of points X such that the circumcircles are tangent is a circle centered at P with radius sqrt(PA * PB)
theorem locus_of_X (X : Type) [PointsOnCircle A B C D] [RadicalCenter (circumcircle_XAB X) (circumcircle_XCD X) P] :
    ∀ X, radical_axis_tangent X → locus X = circle P (sqrt (dist PA * dist PB)) := sorry

end locus_of_X_l765_765045


namespace sum_of_sequence_is_correct_l765_765076

def f (x : ℝ) (a : ℝ) : ℝ := a * x * x - 1

def tangent_slope_at_A (a : ℝ) : ℝ := 2 * a

theorem sum_of_sequence_is_correct :
  (∀ a, tangent_slope_at_A a = 8 → a = 4) →
  (∀ n, f n 4 ≠ 0) →
  let sequence_term := λ n : ℕ, (1 : ℝ) / f n 4 in
  let S := λ n : ℕ, (0 to n-1).sum sequence_term in
  S 2012 = 2012 / 4025 :=
by
  intros slope_a h_f_n_nonzero sequence_term S
  sorry

end sum_of_sequence_is_correct_l765_765076


namespace chemical_plant_sequences_l765_765917

-- Definitions based on conditions
def available_materials : Finset String := {"A", "B", "C", "D", "E"}

def valid_sequences (mater: Finset String): Finset (List String) := 
  { s | s.length = 2 ∧ (s.nth 0 = some "A" → s.nth 1 ≠ some "B") ∧ s.nth 0 ≠ some "B"}

-- Theorem statement to be proven
theorem chemical_plant_sequences : 
  (valid_sequences available_materials).card = 15 := 
sorry

end chemical_plant_sequences_l765_765917


namespace fixed_monthly_fee_l765_765730

/-
  We want to prove that given two conditions:
  1. x + y = 12.48
  2. x + 2y = 17.54
  The fixed monthly fee (x) is 7.42.
-/

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + y = 12.48) 
  (h2 : x + 2 * y = 17.54) : 
  x = 7.42 := 
sorry

end fixed_monthly_fee_l765_765730


namespace sum_x_coordinates_solutions_l765_765790

theorem sum_x_coordinates_solutions :
  let y := |x^2 - 6x + 5|,
      y_2 := 13 / 2 - x
  in ∀ x (h1 : y = y_2),
     ∑ x ∈ {x | y = y_2}, x = 17 / 2 := sorry

end sum_x_coordinates_solutions_l765_765790


namespace chickens_problem_l765_765594

theorem chickens_problem 
    (john_took_more_mary : ∀ (john mary : ℕ), john = mary + 5)
    (ray_took : ℕ := 10)
    (john_took_more_ray : ∀ (john ray : ℕ), john = ray + 11) :
    ∃ mary : ℕ, ray = mary - 6 :=
by
    sorry

end chickens_problem_l765_765594


namespace cube_tetrahedron_properties_l765_765031

-- Definition of the cube and its properties
def Cube (V : Type) :=
  { A B C D A1 B1 C1 D1 : V  // vertices of the cube
    (a : ℝ)                 // edge length of the cube
    (midpoint_AD : V)       // midpoint of edge AD
    (line_ED1 : set V)      // line passing through points E and D1
    (line_A1R : set V)      // line passing through point A1 intersecting BC at R
    -- Additional properties ensuring E is midpoint, and other geometric properties
    (E_midpoint_AD : midpoint_AD = (A + D) / 2)
  }

-- Definitions of questions as properties to be proved
def part_a (V : Type) [normed_group V] (cube : Cube V) : Prop :=
  let ⟨A, B, C, D, A1, B1, C1, D1, a, midpoint_AD, line_ED1, line_A1R, E_midpoint_AD⟩ := cube in
  ∃ R : V, line_A1R.contains(R) ∧ line BC R ∧ ratio (B - R) (B - C) = 2 / 1

def part_b (V : Type) [normed_group V] (cube : Cube V) : Prop :=
  let ⟨A, B, C, D, A1, B1, C1, D1, a, midpoint_AD, line_ED1, line_A1R, E_midpoint_AD⟩ := cube in
  ∃ (M N P Q : V), line_ED1.contains(M) ∧ line_ED1.contains(N) ∧
  line_A1R.contains(P) ∧ line_A1R.contains(Q) ∧
  let midpoint_MN := (M + N) / 2 in
  let midpoint_PQ := (P + Q) / 2 in
  dist midpoint_MN midpoint_PQ = a * √(2 / 15)

-- Main theorem combining parts (a) and (b)
theorem cube_tetrahedron_properties (V : Type) [normed_group V] (cube : Cube V) :
  part_a V cube ∧ part_b V cube :=
by
  sorry

end cube_tetrahedron_properties_l765_765031


namespace simultaneous_in_Quadrant_I_l765_765929

def in_Quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem simultaneous_in_Quadrant_I (c x y : ℝ) : 
  (2 * x - y = 5) ∧ (c * x + y = 4) ↔ in_Quadrant_I x y ∧ (-2 < c ∧ c < 8 / 5) :=
sorry

end simultaneous_in_Quadrant_I_l765_765929


namespace area_ratio_of_squares_l765_765660

theorem area_ratio_of_squares (x : ℝ) :
  let area_A := x^2,
      length_B := x * Real.sqrt 3,
      area_B := (x * Real.sqrt 3)^2
  in area_B = 3 * area_A :=
by 
  sorry

end area_ratio_of_squares_l765_765660


namespace range_of_a_l765_765049

def proposition_p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def proposition_q (a x : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0
def not_necessary_and_sufficient (a : ℝ) : Prop :=
  ¬((∀ x, proposition_q a x → proposition_p x) ∧ (∀ x, proposition_p x → proposition_q a x)) ∧
  (∀ x, (proposition_p x ∨ proposition_q a x) ∧ ¬(proposition_p x ∧ proposition_q a x)) ∧
  (proposition_p (1 / 2) ∧ proposition_p 1 ∧ proposition_q a (1 / 2) ∧ proposition_q a 1)

theorem range_of_a : ∀ a : ℝ, not_necessary_and_sufficient a → (0 ≤ a ∧ a ≤ 1 / 2) :=
begin
  sorry
end

end range_of_a_l765_765049


namespace ξ_values_count_l765_765446

def ξ_possible_sums : Finset ℕ := 
  Finset.univ.product Finset.univ |>.image (λ (a, b : Fin 5), (a+1) + (b+1))

theorem ξ_values_count : ξ_possible_sums.card = 9 :=
by sorry

end ξ_values_count_l765_765446


namespace custom_op_test_l765_765688

def custom_op (a b : ℝ) : ℝ := real.sqrt b + a

theorem custom_op_test : custom_op 15 196 = 29 :=
by
  sorry

end custom_op_test_l765_765688


namespace count_five_digit_numbers_l765_765109

theorem count_five_digit_numbers :
  let digits := [3, 3, 3, 8, 8] in
  fintype.card {l : list ℕ // l.perm digits ∧ (∀ x ∈ l, x ∈ digits)} = 10 :=
by
  sorry

end count_five_digit_numbers_l765_765109


namespace incorrect_proposition_d_l765_765044

noncomputable theory
open_locale classical

variables {Point : Type} [euclidean_geometry Point]
variables {l m : Line Point} {α β γ : Plane Point}
variables (parallel : Line Point → Plane Point → Prop)
          (perpendicular : Plane Point → Plane Point → Prop)
          (line_parallel : Line Point → Line Point → Prop)
          (plane_parallel : Plane Point → Plane Point → Prop)
          (line_perpendicular : Line Point → Line Point → Prop)
          (plane_intersect : Plane Point → Plane Point → Line Point)

def proposition_d (α β : Plane Point) (l : Line Point) : Prop :=
  plane_parallel α β ∧ parallel l α → parallel l β

theorem incorrect_proposition_d (α β : Plane Point) (l : Line Point) :
  ¬ proposition_d α β l :=
sorry

end incorrect_proposition_d_l765_765044


namespace angle_215_third_quadrant_l765_765068

-- Define the context of the problem
def angle_vertex_origin : Prop := true 

def initial_side_non_negative_x_axis : Prop := true

noncomputable def in_third_quadrant (angle: ℝ) : Prop := 
  180 < angle ∧ angle < 270 

-- The theorem to prove the condition given
theorem angle_215_third_quadrant : 
  angle_vertex_origin → 
  initial_side_non_negative_x_axis → 
  in_third_quadrant 215 :=
by
  intro _ _
  unfold in_third_quadrant
  sorry -- This is where the proof would go

end angle_215_third_quadrant_l765_765068


namespace two_digit_primes_with_units_digit_three_l765_765227

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765227


namespace previous_spider_weight_l765_765682

noncomputable def giant_spider_weight (prev_spider_weight : ℝ) : ℝ :=
  2.5 * prev_spider_weight

noncomputable def leg_cross_sectional_area : ℝ := 0.5
noncomputable def leg_pressure : ℝ := 4
noncomputable def legs : ℕ := 8

noncomputable def force_per_leg : ℝ := leg_pressure * leg_cross_sectional_area
noncomputable def total_weight : ℝ := force_per_leg * (legs : ℝ)

theorem previous_spider_weight (prev_spider_weight : ℝ) (h_giant : giant_spider_weight prev_spider_weight = total_weight) : prev_spider_weight = 6.4 :=
by
  sorry

end previous_spider_weight_l765_765682


namespace count_five_digit_numbers_l765_765110

theorem count_five_digit_numbers :
  let digits := [3, 3, 3, 8, 8] in
  fintype.card {l : list ℕ // l.perm digits ∧ (∀ x ∈ l, x ∈ digits)} = 10 :=
by
  sorry

end count_five_digit_numbers_l765_765110


namespace variance_of_data_set_l765_765873

-- Define the data set and its properties
def data_set := [8, 9, x, 9, 10]

-- Define the condition for the average of the data set
def average_condition (x : ℝ) : Prop :=
  (8 + 9 + x + 9 + 10) / 5 = 9

-- Define the variance formula for the given data set
def variance (x : ℝ) : ℝ :=
  (1 / 5) * ((8 - 9)^2 + (9 - 9)^2 + (9 - 9)^2 + (9 - 9)^2 + (10 - 9)^2)

-- State the theorem
theorem variance_of_data_set (x : ℝ) (h : average_condition x) : 
  variance x = 2 / 5 :=
by sorry

end variance_of_data_set_l765_765873


namespace unique_diff_of_cubes_l765_765963

theorem unique_diff_of_cubes (n k : ℕ) (h : 61 = n^3 - k^3) : n = 5 ∧ k = 4 :=
sorry

end unique_diff_of_cubes_l765_765963


namespace quadratic_no_real_roots_l765_765883

theorem quadratic_no_real_roots (c : ℝ) : (∀ x : ℝ, x^2 + 2 * x + c ≠ 0) → c > 1 :=
by
  sorry

end quadratic_no_real_roots_l765_765883


namespace divides_rectangle_into_four_non_congruent_parts_l765_765737

open Set

def divides_into_four_parts_with_equal_perimeters_and_non_congruent {F : Type u} [MetricSpace F] 
  (F : F) : Prop :=
  ∃ (F1 F2 F3 F4 : Set F), 
    F1 ∪ F2 ∪ F3 ∪ F4 = F ∧ 
    -- Each part is a subset of the original figure 
    F1 ⊆ F ∧ F2 ⊆ F ∧ F3 ⊆ F ∧ F4 ⊆ F ∧ 
    -- Parts do not overlap
    disjoint F1 F2 ∧ disjoint F1 F3 ∧ disjoint F1 F4 ∧ 
    disjoint F2 F3 ∧ disjoint F2 F4 ∧ disjoint F3 F4 ∧ 
    -- Each part has an equal perimeter
    perimeter(F1) = perimeter(F2) ∧ perimeter(F2) = perimeter(F3) ∧ perimeter(F3) = perimeter(F4) ∧ 
    -- None of the parts are congruent to each other
    ¬congruent F1 F2 ∧ ¬congruent F1 F3 ∧ ¬congruent F1 F4 ∧ 
    ¬congruent F2 F3 ∧ ¬congruent F2 F4 ∧ ¬congruent F3 F4

theorem divides_rectangle_into_four_non_congruent_parts (R : Rectangle) :
  divides_into_four_parts_with_equal_perimeters_and_non_congruent R :=
sorry

end divides_rectangle_into_four_non_congruent_parts_l765_765737


namespace two_digit_primes_with_units_digit_three_l765_765235

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765235


namespace complex_quadrant_l765_765998

theorem complex_quadrant (z : ℂ) (h : z = (↑0 + 1*I) / (1 + 1*I)) : z.re > 0 ∧ z.im > 0 := 
by
  sorry

end complex_quadrant_l765_765998


namespace ratio_sum_l765_765025

theorem ratio_sum {x y : ℚ} (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_sum_l765_765025


namespace min_value_a_l765_765056

theorem min_value_a (a b : ℕ) (h1: a = b - 2005) 
  (h2: ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ p + q = a ∧ p * q = b) : a ≥ 95 := sorry

end min_value_a_l765_765056


namespace count_two_digit_primes_with_units_digit_3_l765_765201

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765201


namespace find_sum_l765_765059

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem find_sum (h₁ : a * b = 2 * (a + b))
                (h₂ : b * c = 3 * (b + c))
                (h₃ : c * a = 4 * (a + c))
                (ha : a ≠ 0)
                (hb : b ≠ 0)
                (hc : c ≠ 0) 
                : a + b + c = 1128 / 35 :=
by
  sorry

end find_sum_l765_765059


namespace fraction_of_students_who_actually_like_dancing_l765_765718

theorem fraction_of_students_who_actually_like_dancing :
  let total_students := 100
  let students_who_like := 50
  let students_who_neither := 30
  let students_who_dislike := total_students - students_who_like - students_who_neither

  let like_say_dislike := 0.3 * students_who_like
  let dislike_say_dislike := 0.8 * students_who_dislike
  let neither_say_dislike := 0.6 * students_who_neither

  let total_say_dislike := like_say_dislike + dislike_say_dislike + neither_say_dislike

  like_say_dislike / total_say_dislike = 15 / 49 :=
by
  let total_students := 100
  let students_who_like := 50
  let students_who_neither := 30
  let students_who_dislike := total_students - students_who_like - students_who_neither

  let like_say_dislike := 0.3 * students_who_like
  let dislike_say_dislike := 0.8 * students_who_dislike
  let neither_say_dislike := 0.6 * students_who_neither

  let total_say_dislike := like_say_dislike + dislike_say_dislike + neither_say_dislike

  have : like_say_dislike = 15 := by simp
  have : total_say_dislike = 49 := by simp
  
  sorry

end fraction_of_students_who_actually_like_dancing_l765_765718


namespace inverse_function_valid_l765_765007

def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem inverse_function_valid (x : ℝ) (hx : x > 1) :
  ∃ y, f y = x ∧ y = 1 - sqrt (x - 1) :=
sorry

end inverse_function_valid_l765_765007


namespace K_lies_on_BM_l765_765494

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def midpoint (A C : Point) : Point := sorry
noncomputable def altitude (A1 B1 : Point) : Point := sorry
noncomputable def circumcircle (P Q R : Point) : Circle := sorry

variables (A B C O M Ha Hc K : Point)

-- Conditions
axiom O_is_circumcenter_of_ABC : circumcenter A B C = O
axiom M_is_midpoint_of_AC : midpoint A C = M
axiom Ha_on_altitude_AA1 : altitude A C = Ha
axiom Hc_on_altitude_CC1 : altitude C A = Hc
axiom K_on_circumcircles_intersection : K ∈ (circumcircle B Ha A) ∧ K ∈ (circumcircle B Hc C)

-- Proof Goal
theorem K_lies_on_BM :
  ∃ K : Point, K ∈ line_through B M :=
sorry

end K_lies_on_BM_l765_765494


namespace problem_1_problem_2_problem_3_l765_765813

theorem problem_1 (a b c : ℝ) (h₁ : f(0) = 2) (h₂ : ∀ x : ℝ, f(x + 1) - f(x) = 2x - 1) :
  f(x) = x^2 - 2x + 2 := sorry

theorem problem_2 (t : ℝ) :
  (∀ x ∈ Icc (-1 : ℝ) 2, (x^2 - 2x + 2) - t > 0) ↔ t < 5 := sorry

theorem problem_3 (m : ℝ) :
  (∃ x ∈ Icc (-1 : ℝ) 2, (x^2 - (2 + m)*x + 2) = 0 ∧ ∃ y ∈ Icc 2 4, (y^2 - (2 + m)*y + 2) = 0) ↔ (1 < m ∧ m < 5 / 2) := sorry

end problem_1_problem_2_problem_3_l765_765813


namespace area_proof_l765_765452

-- Define the vertices
structure Point (α : Type) :=
  (x : α)
  (y : α)

def P1 : Point ℤ := ⟨-8, 1⟩
def P2 : Point ℤ := ⟨1, 1⟩
def P3 : Point ℤ := ⟨1, -7⟩
def P4 : Point ℤ := ⟨-8, -7⟩

-- Function to calculate the distance between two points horizontally and vertically
def horizontal_distance (p1 p2 : Point ℤ) : ℤ :=
  abs (p2.x - p1.x)

def vertical_distance (p1 p2 : Point ℤ) : ℤ :=
  abs (p1.y - p2.y)

-- Define the lengths of sides
def length_horizontal_side : ℤ :=
  horizontal_distance P1 P2

def length_vertical_side : ℤ :=
  vertical_distance P3 P4

-- Given conditions formulated as definitions
def length_horizontal_side_def : length_horizontal_side = 9 := sorry
def length_vertical_side_def : length_vertical_side = 8 := sorry

-- The area calculation
def area_of_rectangle : ℤ :=
  length_horizontal_side * length_vertical_side

-- The proof statement
theorem area_proof : area_of_rectangle = 72 := by
  -- Utilize the given conditions
  rw [←length_horizontal_side_def, ←length_vertical_side_def]
  exact rfl

end area_proof_l765_765452


namespace prime_congruences_l765_765517

theorem prime_congruences (p : ℕ) (hp : Nat.Prime p) (hp7 : 7 ≤ p) :
  ∃ (n : ℕ) (x y : Fin n → ℤ),
    (∀ i, x i % p ≠ 0) ∧ -- x_i is not divisible by p
    (∀ i, y i % p ≠ 0) ∧ -- y_i is not divisible by p
    (∀ i, (x i) ^ 2 + (y i) ^ 2 ≡ (x (i + 1)) ^ 2 [MOD p] ∧ i < n - 1) :=
by 
  sorry

end prime_congruences_l765_765517


namespace point_in_first_quadrant_l765_765070

-- Define the conditions given
def complex_number := i * (1 - i)

-- Define the coordinates corresponding to the complex number
def point_Z := (complex_number.re, complex_number.im)

-- Formulate the statement to prove that point_Z is in the first quadrant
theorem point_in_first_quadrant (h : complex_number = i * (1 - i)) : 
  point_Z = (1, 1) ∧ (point_Z.1 > 0 ∧ point_Z.2 > 0) :=
by
  unfold complex_number point_Z
  split
  . norm_num
  . split
    . norm_num
    . norm_num

end point_in_first_quadrant_l765_765070


namespace count_two_digit_primes_with_units_digit_3_l765_765320

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765320


namespace range_of_a_l765_765605

theorem range_of_a (a : ℝ) : 
  let f (x : ℝ) := a * x^3 + a * x^2 - 2 * a * x + 2 * a + 1 in
  (∀ x : ℝ, f(x) < 0 ∨ ∃ y : ℝ, f y > 0) →
  (-∞ < a ∧ a < ∞) :=
sorry

end range_of_a_l765_765605


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765390

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765390


namespace two_digit_primes_units_digit_3_count_l765_765355

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765355


namespace find_fourth_term_l765_765836

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (a_1 a_4 d : ℕ)

-- Conditions
axiom sum_first_5 : S_n 5 = 35
axiom sum_first_9 : S_n 9 = 117
axiom sum_closed_form_first_5 : 5 * a_1 + (5 * (5 - 1)) / 2 * d = 35
axiom sum_closed_form_first_9 : 9 * a_1 + (9 * (9 - 1)) / 2 * d = 117
axiom nth_term_closed_form : ∀ n, a_n n = a_1 + (n-1)*d

-- Target
theorem find_fourth_term : a_4 = 10 := by
  sorry

end find_fourth_term_l765_765836


namespace brad_running_speed_l765_765952

-- Definitions based on the given conditions
def distance_between_homes : ℝ := 24
def maxwell_walking_speed : ℝ := 4
def maxwell_time_to_meet : ℝ := 3

/-- Brad's running speed is 6 km/h given the conditions of the problem. -/
theorem brad_running_speed : (distance_between_homes - (maxwell_walking_speed * maxwell_time_to_meet)) / (maxwell_time_to_meet - 1) = 6 := by
  sorry

end brad_running_speed_l765_765952


namespace fred_gave_sandy_balloons_l765_765023

theorem fred_gave_sandy_balloons :
  ∀ (original_balloons given_balloons final_balloons : ℕ),
    original_balloons = 709 →
    final_balloons = 488 →
    given_balloons = original_balloons - final_balloons →
    given_balloons = 221 := by
  sorry

end fred_gave_sandy_balloons_l765_765023


namespace green_stripe_area_l765_765679

/-- Given:
1. A cylindrical silo with diameter 40 feet.
2. A cylindrical silo with height 120 feet.
3. A green stripe with a horizontal width of 4 feet.
4. The stripe makes three complete revolutions around the silo.

We want to prove that the area of the green stripe is 480 * sqrt(π^2 + 1) square feet.
-/
theorem green_stripe_area (d h w : ℝ) (n : ℕ) (π : ℝ) (pi_ne_zero : π ≠ 0)
  (diameter_eq : d = 40) (height_eq : h = 120) (width_eq : w = 4) (revolutions_eq : n = 3) :
  let C := π * d
  let L := sqrt (C^2 + (h / n)^2)
  let total_length := n * L
  let area := total_length * w
  area = 480 * sqrt (π^2 + 1) := 
by { 
  rw [diameter_eq, height_eq, width_eq, revolutions_eq],
  sorry
}

end green_stripe_area_l765_765679


namespace sum_of_distances_l765_765503

-- Definitions for the problem
variables (r R : ℝ) (A B : Circle) (A0 : Point)
variables (C : ℕ → Circle) (A_n : ℕ → Point)

-- Conditions of the problem
axiom tangent_at_A0 : TangentInternally A B A0
axiom unequal_radii : r ≠ R
axiom tangent_to_both : ∀ n, TangentToBoth C n A B
axiom sequence_touches : ∀ n, TouchesAt (C (n + 1)) (C n) (A_n n)

-- Main statement to be proved
theorem sum_of_distances :
  ∑' n, dist (A_n (n + 1)) (A_n n) < (4 * π * R * r) / (R + r) :=
sorry

end sum_of_distances_l765_765503


namespace units_digit_of_factorial_sum_l765_765648

theorem units_digit_of_factorial_sum :
  (∑ n in Finset.range 800.succ, (nat.factorial n) % 10) % 10 = 3 :=
by
  sorry

end units_digit_of_factorial_sum_l765_765648


namespace handshakes_equality_l765_765970

theorem handshakes_equality (n : ℕ) (h : 1 ≤ n) :
  ∃ (a b : ℕ), a ≠ b ∧ handshakes a = handshakes b :=
sorry

end handshakes_equality_l765_765970


namespace number_of_two_digit_primes_with_units_digit_three_l765_765204

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765204


namespace locus_of_S_l765_765073

theorem locus_of_S (x y x0 y0 : ℝ) 
  (h1 : (x0 ^ 2) / 5 + (y0 ^ 2) / 4 = 1) 
  (h2 : x0 ≠ sqrt 5 ∧ x0 ≠ -sqrt 5) : 
  (x ^ 2) / 5 + (y ^ 2) / 9 = 1 :=
sorry

end locus_of_S_l765_765073


namespace count_prime_units_digit_3_eq_6_l765_765303

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765303


namespace tree_F_height_l765_765619

variable (A B C D E F : ℝ)

def height_conditions : Prop :=
  A = 150 ∧ -- Tree A's height is 150 feet
  B = (2 / 3) * A ∧ -- Tree B's height is 2/3 of Tree A's height
  C = (1 / 2) * B ∧ -- Tree C's height is 1/2 of Tree B's height
  D = C + 25 ∧ -- Tree D's height is 25 feet more than Tree C's height
  E = 0.40 * A ∧ -- Tree E's height is 40% of Tree A's height
  F = (B + D) / 2 -- Tree F's height is the average of Tree B's height and Tree D's height

theorem tree_F_height : height_conditions A B C D E F → F = 87.5 :=
by
  intros
  sorry

end tree_F_height_l765_765619


namespace num_two_digit_primes_with_units_digit_three_l765_765145

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765145


namespace non_empty_combinations_of_7_beads_even_n_combinations_are_odd_permutations_even_and_odd_l765_765948

-- Problem 1: Prove that the number of non-empty combinations of 7 beads is 127
theorem non_empty_combinations_of_7_beads : 2^7 - 1 = 127 :=
by
  calc
    2^7 - 1 = 128 - 1 : rfl
         ... = 127     : rfl

-- Problem 2: Prove that for any even n, 2^n - 1 is odd
theorem even_n_combinations_are_odd (n : Nat) (h_even : n % 2 = 0) : (2^n - 1) % 2 = 1 :=
by
  have h : 2^n % 2 = 0 := by sorry  -- Prove that 2^n is even for any even n
  calc
    (2^n - 1) % 2 = (2^n % 2 - 1 % 2) % 2 : by sorry  -- Apply modulo properties
             ... = (0 - 1) % 2             : by rw [h, Nat.mod_eq_of_lt (by decide: 1 < 2)]
             ... = -1 % 2                  : rfl
             ... = 1                       : by rw [Nat.mod_eq_of_lt (by decide: 1 < 2)]

-- Problem 3: Prove that the number of permutations considering order of beads can be either even or odd
theorem permutations_even_and_odd (n : Nat) : ∃ k : Nat, (∑ i in (range (n + 1)).filter (λ x, x > 0), P n i) % 2 = k % 2 :=
by
  -- Using the formula ∑ P(n, k) and the fact that P(n, k) includes both even and odd numbers
  sorry


end non_empty_combinations_of_7_beads_even_n_combinations_are_odd_permutations_even_and_odd_l765_765948


namespace count_two_digit_primes_with_units_digit_3_l765_765188

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765188


namespace count_integers_with_digit_sum_l765_765126

theorem count_integers_with_digit_sum :
  let count := (finset.range 600).filter (λ n, n >= 400 ∧ (n.digits.sum = 18)).card
  count = 20 :=
by
  sorry

end count_integers_with_digit_sum_l765_765126


namespace count_two_digit_primes_with_units_digit_3_l765_765311

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765311


namespace number_of_good_matrices_l765_765812

noncomputable def good_matrix_count (p : ℕ) [fact (nat.prime p)] : ℕ :=
2 * (p.factorial ^ 2)

theorem number_of_good_matrices (p : ℕ) [fact (nat.prime p)] :
  good_matrix_count p = 2 * (p.factorial ^ 2) :=
sorry

end number_of_good_matrices_l765_765812


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765275

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765275


namespace average_speed_of_car_l765_765665

-- Define the conditions
def distance_first_hour := 20
def distance_second_hour := 60
def total_distance := distance_first_hour + distance_second_hour
def total_time := 2 -- in hours

-- Define the theorem to be proved
theorem average_speed_of_car :
  let avg_speed := total_distance / total_time in
  avg_speed = 40 := 
by
  sorry

end average_speed_of_car_l765_765665


namespace two_digit_primes_units_digit_3_count_l765_765358

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765358


namespace cost_of_5_dozen_l765_765919

noncomputable def price_per_dozen : ℝ :=
  24 / 3

noncomputable def cost_before_tax (num_dozen : ℝ) : ℝ :=
  num_dozen * price_per_dozen

noncomputable def cost_after_tax (num_dozen : ℝ) : ℝ :=
  (1 + 0.10) * cost_before_tax num_dozen

theorem cost_of_5_dozen :
  cost_after_tax 5 = 44 := 
sorry

end cost_of_5_dozen_l765_765919


namespace sara_lunch_total_cost_l765_765969

noncomputable def cost_hotdog : ℝ := 5.36
noncomputable def cost_salad : ℝ := 5.10
noncomputable def cost_soda : ℝ := 2.75
noncomputable def cost_fries : ℝ := 3.20
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08

noncomputable def total_cost_before_discount_tax : ℝ :=
  cost_hotdog + cost_salad + cost_soda + cost_fries

noncomputable def discount : ℝ :=
  discount_rate * total_cost_before_discount_tax

noncomputable def discounted_total : ℝ :=
  total_cost_before_discount_tax - discount

noncomputable def tax : ℝ := 
  tax_rate * discounted_total

noncomputable def final_total : ℝ :=
  discounted_total + tax

theorem sara_lunch_total_cost : final_total = 15.07 :=
by
  sorry

end sara_lunch_total_cost_l765_765969


namespace exists_subset_sum_modulus_at_least_one_sixth_l765_765935

theorem exists_subset_sum_modulus_at_least_one_sixth
  (n : ℕ) (Z : Fin n → ℂ)
  (hZ : ∑ i, Complex.abs (Z i) = 1) :
  ∃ S : Finset (Fin n), Complex.abs (∑ i in S, Z i) ≥ 1 / 6 := 
  sorry

end exists_subset_sum_modulus_at_least_one_sixth_l765_765935


namespace pirate_coins_total_l765_765957

theorem pirate_coins_total (x : ℕ) (hx : x ≠ 0) (h_paul : ∃ k : ℕ, k = x / 2) (h_pete : ∃ m : ℕ, m = 5 * (x / 2)) 
  (h_ratio : (m : ℝ) = (k : ℝ) * 5) : (x = 4) → 
  ∃ total : ℕ, total = k + m ∧ total = 12 :=
by {
  sorry
}

end pirate_coins_total_l765_765957


namespace circle_equation_max_min_xy_l765_765029

noncomputable def rho (theta : ℝ) : ℝ :=
  let c := cos (theta - π / 4)
  let term := 4 * √2 * c
  let discriminant := term ^ 2 - 4 * 6
  -term / 2

noncomputable def x (theta : ℝ) : ℝ :=
  2 + √2 * cos theta

noncomputable def y (theta : ℝ) : ℝ :=
  2 + √2 * sin theta

theorem circle_equation (rho theta : ℝ) :
  (rho ^ 2 - 4 * √2 * rho * cos (theta - π / 4) + 6 = 0) ->
  (x rho = 2 + √2 * cos theta) ->
  (y rho = 2 + √2 * sin theta) ->
  ((x rho) ^ 2 + (y rho) ^ 2 - 4 * (x rho) - 4 * (y rho) + 6 = 0) :=
by
  intros h1 h2 h3
  -- This part will require filling in with the actual transformation steps resulting in the standard form of the equation.
  sorry

theorem max_min_xy (theta : ℝ) :
  (1 ≤ (x theta * y theta)) ∧ ((x theta * y theta) ≤ 9) :=
by
  intros h1
  -- This part will require filling in with the actual proof steps determining the bounds for xy.
  sorry

end circle_equation_max_min_xy_l765_765029


namespace nat_pairs_satisfy_conditions_l765_765762

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l765_765762


namespace solve_inequality_system_l765_765587

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l765_765587


namespace num_unique_permutations_l765_765105

-- Given: digits 3 and 8 being repeated as described.
-- Show: number of different permutations of the digits 3, 3, 3, 8, 8 is 10.

theorem num_unique_permutations : 
  let digits := [3, 3, 3, 8, 8] in
  let total_permutations := (5!).nat_abs in             -- 5! permutations
  let repeats_correction := (3!).nat_abs * (2!).nat_abs in -- Adjusting for repeated 3's and 8's
  let unique_permutations := total_permutations / repeats_correction in
  unique_permutations = 10 :=
by
  sorry

end num_unique_permutations_l765_765105


namespace count_two_digit_prime_numbers_ending_in_3_l765_765240

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765240


namespace count_integers_with_digit_sum_l765_765125

theorem count_integers_with_digit_sum :
  let count := (finset.range 600).filter (λ n, n >= 400 ∧ (n.digits.sum = 18)).card
  count = 20 :=
by
  sorry

end count_integers_with_digit_sum_l765_765125


namespace a4_eq_one_t2_eq_t5_if_a1_then_a2_one_l765_765064

variables {T : ℕ → ℝ} {a : ℕ → ℝ}

-- Condition: The product of the first n terms of a geometric sequence
def geo_seq_prod (n : ℕ) : ℝ := ∏ i in finset.range n, a i

-- Specific condition: third, fourth and fifth terms' product is 1
axiom geo_term_prod : a 3 * a 4 * a 5 = 1

-- Statements to be proven
theorem a4_eq_one (h : a 3 * a 4 * a 5 = 1) : a 4 = 1 :=
sorry

theorem t2_eq_t5 (h : a 3 * a 4 * a 5 = 1) : T 2 = T 5 :=
sorry

theorem if_a1_then_a2_one (h : a 1 = 1) (h : a 3 * a 4 * a 5 = 1) : a 2 = 1 :=
sorry

end a4_eq_one_t2_eq_t5_if_a1_then_a2_one_l765_765064


namespace number_of_two_digit_primes_with_units_digit_three_l765_765214

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765214


namespace count_two_digit_primes_with_units_digit_3_l765_765153

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765153


namespace percent_double_and_undeclared_men_correct_l765_765891

open Nat Real

-- Define the given conditions
def percent_sci_major_men := 24 / 100
def percent_hum_major_men := 13 / 100
def percent_bus_major_men := 18 / 100
def percent_double_or_undeclared_men := 45 / 100
def percent_double_sci_hum_men := 30 / 100
def percent_double_sci_bus_men := 20 / 100
def percent_double_hum_bus_men := 15 / 100

-- Define the expected results based on the solution
def expected_percent_double_sci_hum_men := 13.5 / 100
def expected_percent_double_sci_bus_men := 9 / 100
def expected_percent_double_hum_bus_men := 6.75 / 100
def expected_percent_undeclared_men := 15.75 / 100

theorem percent_double_and_undeclared_men_correct :
  percent_double_sci_hum_men * percent_double_or_undeclared_men = expected_percent_double_sci_hum_men
  ∧ percent_double_sci_bus_men * percent_double_or_undeclared_men = expected_percent_double_sci_bus_men
  ∧ percent_double_hum_bus_men * percent_double_or_undeclared_men = expected_percent_double_hum_bus_men
  ∧ (percent_double_or_undeclared_men - (percent_double_sci_hum_men * percent_double_or_undeclared_men
  + percent_double_sci_bus_men * percent_double_or_undeclared_men
  + percent_double_hum_bus_men * percent_double_or_undeclared_men)) = expected_percent_undeclared_men := 
by
  sorry

end percent_double_and_undeclared_men_correct_l765_765891


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765337

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765337


namespace find_n_from_binomial_variance_l765_765053

variable (ξ : Type)
variable (n : ℕ)
variable (p : ℝ := 0.3)
variable (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p))

-- Given conditions
axiom binomial_distribution : p = 0.3 ∧ Var n p = 2.1

-- Prove n = 10
theorem find_n_from_binomial_variance (ξ : Type) (n : ℕ) (p : ℝ := 0.3) (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p)) :
  p = 0.3 ∧ Var n p = 2.1 → n = 10 :=
by
  sorry

end find_n_from_binomial_variance_l765_765053


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765332

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765332


namespace count_two_digit_primes_with_units_digit_three_l765_765401

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765401


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765257

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765257


namespace count_increasing_4x4_matrices_l765_765733

theorem count_increasing_4x4_matrices :
  let n := 4
  let total_elements := Finset.range (n * n + 1)
  let matrices := { A : matrix (Fin n) (Fin n) ℕ // 
    (∀ i j, i < j → A i 0 < A j 0) ∧ 
    (∀ i j, i < j → A 0 i < A 0 j) ∧ 
    (∀ i j k, j < k → A i j < A i k) ∧ 
    (∀ i j k, j < k → A j i < A k i)  ∧ 
    A.to_finset = total_elements } in
  matrices.card = 175 := sorry

end count_increasing_4x4_matrices_l765_765733


namespace complex_quadrant_l765_765999

theorem complex_quadrant (z : ℂ) (h : z = (↑0 + 1*I) / (1 + 1*I)) : z.re > 0 ∧ z.im > 0 := 
by
  sorry

end complex_quadrant_l765_765999


namespace problem_statement_l765_765867

theorem problem_statement (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 ∧ x^2 + y^2 = 104 :=
by
  sorry

end problem_statement_l765_765867


namespace monotonic_intervals_range_on_0_to_pi_l765_765842

noncomputable def f (x : ℝ) : ℝ :=
  (sin (x / 2) + cos (x / 2))^2 - 2 * sqrt 3 * (cos (x / 2))^2 + sqrt 3

theorem monotonic_intervals :
  ∀ (k : ℤ), 
    (∀ x, 2 * k * π - π / 6 ≤ x ∧ x ≤ 5 * π / 6 + 2 * k * π → f x is_strictly_increasing_on (2 * k * π - π / 6, 5 * π / 6 + 2 * k * π)) ∧ 
    (∀ x, 5 * π / 6 + 2 * k * π ≤ x ∧ x ≤ 11 * π / 6 + 2 * k * π → f x is_strictly_decreasing_on (5 * π / 6 + 2 * k * π, 11 * π / 6 + 2 * k * π)) :=
sorry

theorem range_on_0_to_pi :
  set.range (f ∘ λ x, x ∈ Icc 0 π) = set.Icc (1 - sqrt 3) 3 :=
sorry

end monotonic_intervals_range_on_0_to_pi_l765_765842


namespace num_two_digit_primes_with_units_digit_three_l765_765138

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765138


namespace nat_pairs_satisfy_conditions_l765_765761

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l765_765761


namespace range_of_a_l765_765806

theorem range_of_a (a : ℝ) : (-1/3 ≤ a) ∧ (a ≤ 2/3) ↔ (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → y = a * x + 1/3) :=
by
  sorry

end range_of_a_l765_765806


namespace inequality_system_solution_l765_765552

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l765_765552


namespace count_two_digit_primes_with_units_digit_3_l765_765187

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765187


namespace solve_inequality_system_l765_765563

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765563


namespace complex_number_in_first_quadrant_l765_765992

theorem complex_number_in_first_quadrant :
  let z := (Complex.I / (1 + Complex.I)) in
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l765_765992


namespace num_two_digit_primes_with_units_digit_three_l765_765137

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765137


namespace math_problem_equivalent_l765_765990

noncomputable def quadratic_roots_sum_is_conjugate (a : ℝ) : Prop :=
  ∃ x1 x2 : ℂ, x1 = 1 + complex.I * real.sqrt 3 ∧ x2 = 1 - complex.I * real.sqrt 3 ∧ 
    (x1 + x2 = - (a + 1)) ∧
    (a = -3)

noncomputable def points_distance_is_correct (A B : ℂ) : Prop :=
  A = 1 + complex.I * real.sqrt 3 ∧ B = 1 - complex.I * real.sqrt 3 ∧
    abs (B - A) = 2 * real.sqrt 3

theorem math_problem_equivalent (a : ℝ) (A B : ℂ) : 
  quadratic_roots_sum_is_conjugate a ∧ points_distance_is_correct A B :=
by
  sorry

end math_problem_equivalent_l765_765990


namespace bus_ride_cost_l765_765656

theorem bus_ride_cost (B T : ℝ) (h1 : T = B + 6.85) (h2 : T + B = 9.65) : B = 1.40 :=
sorry

end bus_ride_cost_l765_765656


namespace daniel_siblings_age_sum_l765_765989

theorem daniel_siblings_age_sum {ages : Fin 6 → ℕ}
  (mean_condition : (∑ i, ages i) = 60)
  (median_condition : (ages 2 + ages 3) = 24)
  (sorted : ∀ i j, i ≤ j → ages i ≤ ages j) :
  ages 0 + ages 5 = 12 := 
sorry

end daniel_siblings_age_sum_l765_765989


namespace inequality_solutions_l765_765082

theorem inequality_solutions (a : ℚ) :
  (∀ x : ℕ, 0 < x ∧ x ≤ 3 → 3 * (x - 1) < 2 * (x + a) - 5) →
  (∃ x : ℕ, 0 < x ∧ x = 4 → ¬ (3 * (x - 1) < 2 * (x + a) - 5)) →
  (5 / 2 < a ∧ a ≤ 3) :=
sorry

end inequality_solutions_l765_765082


namespace count_two_digit_primes_ending_in_3_l765_765376

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765376


namespace right_triangle_DB_length_l765_765453

noncomputable def segment_length : ℝ := 2 * Real.sqrt 15

theorem right_triangle_DB_length 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (dist_AB : A → B → ℝ)
  (dist_AD : A → D → ℝ)
  (dist_DB : D → B → ℝ)
  (dist_AC : A → C → ℝ)
  (angle_ABC : Real.Angle)
  (angle_ADB : Real.Angle)
  (right_angle_ABC : angle_ABC = π / 2)
  (right_angle_ADB : angle_ADB = π / 2)
  (AC_len : dist_AC A C = 19)
  (AD_len : dist_AD A D = 4) : 
dist_DB D B = segment_length :=
sorry

end right_triangle_DB_length_l765_765453


namespace not_both_hit_prob_l765_765714

-- Defining the probabilities
def prob_archer_A_hits : ℚ := 1 / 3
def prob_archer_B_hits : ℚ := 1 / 2

-- Defining event B as both hit the bullseye
def prob_both_hit : ℚ := prob_archer_A_hits * prob_archer_B_hits

-- Defining the complementary event of not both hitting the bullseye
def prob_not_both_hit : ℚ := 1 - prob_both_hit

theorem not_both_hit_prob : prob_not_both_hit = 5 / 6 := by
  -- This is the statement we are trying to prove.
  sorry

end not_both_hit_prob_l765_765714


namespace number_of_two_digit_primes_with_units_digit_three_l765_765202

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765202


namespace trajectory_of_P_no_such_line_exists_l765_765854

def circle1 : set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 = 0 }
def circle2 : set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 4 }

def c1_center := (1, 0)
def c2_center := (-1, 0)
def P_condition (P : ℝ × ℝ) := real.sqrt((P.1 - 1)^2 + P.2^2) + real.sqrt((P.1 + 1)^2 + P.2^2) = 2 * real.sqrt 2

theorem trajectory_of_P :
  { P : ℝ × ℝ | P_condition P } = { P : ℝ × ℝ | P.1^2 / 2 + P.2^2 = 1 } :=
sorry

theorem no_such_line_exists (l : ℝ → ℝ) (A := (2, 0)) :
  ¬ ∃ (l : ℝ → ℝ), (∀ (y₁ y₂ : ℝ), ∃ x₁ x₂, (x₁ ≠ x₂) ∧ (y₁ = l x₁) ∧ (y₂ = l x₂) ∧
  (y₁^2 / 2 + y₁^2 = 1) ∧ (y₂^2 / 2 + y₂^2 = 1) ∧
  real.sqrt((x₁ - 1)^2 + y₁^2) = real.sqrt((x₂ - 1)^2 + y₂^2) ∧
  l 2 = A.2) :=
sorry

end trajectory_of_P_no_such_line_exists_l765_765854


namespace points_rectangle_l765_765479

open Set

/--
If S is a set of 2002 points in the real coordinate plane with no two points sharing the same x or y coordinates,
then there exist two points P and Q in S such that the number of points of S lying in the interior of the rectangle formed
with diagonal PQ (and sides parallel to the axes) is at least 400.
-/
theorem points_rectangle (S : Set (ℝ × ℝ)) (h_card : S.card = 2002)
  (h_unique : ∀ (p1 p2 : ℝ × ℝ), p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2) :
  ∃ P Q ∈ S, P ≠ Q ∧ 2002 - 3 ≥ 400 := by
  sorry

end points_rectangle_l765_765479


namespace sally_popped_3_balloons_l765_765477

-- Defining the conditions
def joans_initial_balloons : ℕ := 9
def jessicas_balloons : ℕ := 2
def total_balloons_now : ℕ := 6

-- Definition for the number of balloons Sally popped
def sally_balloons_popped : ℕ := joans_initial_balloons - (total_balloons_now - jessicas_balloons)

-- The theorem statement
theorem sally_popped_3_balloons : sally_balloons_popped = 3 := 
by
  -- Proof omitted; use sorry
  sorry

end sally_popped_3_balloons_l765_765477


namespace solve_inequality_system_l765_765556

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765556


namespace count_two_digit_primes_with_units_digit_3_l765_765184

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765184


namespace count_two_digit_prime_numbers_ending_in_3_l765_765238

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765238


namespace count_two_digit_primes_with_units_digit_3_l765_765322

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765322


namespace num_two_digit_primes_with_units_digit_three_l765_765141

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765141


namespace count_two_digit_primes_with_units_digit_3_l765_765198

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765198


namespace hyperbola_eccentricity_sqrt_6_l765_765849

noncomputable def hyperbola_eccentricity (a : ℝ) (a_pos : 0 < a) : ℝ :=
  let c := sqrt (a^2 + 1)
  c / a

theorem hyperbola_eccentricity_sqrt_6 {a : ℝ} (h_pos: 0 < a) (h_eq : 2 = sqrt (1 - a^2) / a)
: hyperbola_eccentricity a h_pos = sqrt 6 :=
  sorry

end hyperbola_eccentricity_sqrt_6_l765_765849


namespace distance_between_l1_l2_is_5_l765_765595

def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c1 - c2) / (Real.sqrt (a^2 + b^2))

theorem distance_between_l1_l2_is_5 : 
  distance_between_parallel_lines 3 4 2 7 = 5 := by
sorry

end distance_between_l1_l2_is_5_l765_765595


namespace tan_A_in_right_triangle_l765_765468

theorem tan_A_in_right_triangle
  (C : ℝ)
  (A B : ℝ)
  (angle_C_90 : C = π / 2)
  (AB : ℝ)
  (BC : ℝ)
  (AB_given : AB = 13)
  (BC_given : BC = real.sqrt 75) :
  real.tan (B - C) = 5 * real.sqrt 3 / real.sqrt 94 :=
by
  -- Proof goes here
  sorry

end tan_A_in_right_triangle_l765_765468


namespace count_two_digit_primes_ending_in_3_l765_765379

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765379


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765274

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765274


namespace max_safe_caffeine_amount_l765_765988

theorem max_safe_caffeine_amount
  (E : ℕ := 120) 
  (n : ℕ := 4) 
  (C : ℕ := 20) 
  (M : ℕ := 500) :
  M = n * E + C :=
by
  rfl

end max_safe_caffeine_amount_l765_765988


namespace interval_monotonic_decrease_cos_value_range_of_k_l765_765093

-- Define the given vectors m and n as functions of x
def m (x : ℝ) : ℝ × ℝ :=
(√3 * sin (x / 4), 1)

def n (x : ℝ) : ℝ × ℝ :=
(cos (x / 4), cos (x / 4) ^ 2)

-- Define the dot product f(x) for these vectors
def f (x : ℝ) : ℝ :=
let ⟨m1, m2⟩ := m x;
let ⟨n1, n2⟩ := n x;
m1 * n1 + m2 * n2

-- Proof problem 1: Prove the interval of monotonic decrease of f(x)
theorem interval_monotonic_decrease (k : ℤ) :
  ∀ x, 4 * (k : ℝ) * π + (2 * π / 3) ≤ x ∧ x ≤ 4 * (k : ℝ) * π + (4 * π / 3) →
  (∀ x1 x2, (4 * (k : ℝ) * π + (2 * π / 3) ≤ x1 ∧ x1 ≤ 4 * (k : ℝ) * π + (4 * π / 3) ∧
  4 * (k : ℝ) * π + (2 * π / 3) ≤ x2 ∧ x2 ≤ 4 * (k : ℝ) * π + (4 * π / 3) ∧
  x1 < x2) →
  f x1 ≥ f x2) := sorry

-- Proof problem 2: Prove that if f(a) = 3/2, then cos ((2 * π / 3) - a) = 1
theorem cos_value (a : ℝ) (ha : f a = 3 / 2) :
  cos ((2 * π / 3) - a) = 1 := sorry

-- Proof problem 3: Prove the range of k if g(x) = f(x - (2 * π / 3)) and g(x) - k has a zero in [0, 7π/3]
theorem range_of_k (y : ℝ → ℝ) (k : ℝ) (hk : 0 ≤ k ∧ k ≤ 3 / 2) :
  (∀ x, 0 ≤ x ∧ x ≤ 7 * π / 3 → y x = 0) →
  (∀ x, y (x + (2 * π / 3)) = f x) →
  k ∈ Set.Icc 0 (3 / 2) := sorry

end interval_monotonic_decrease_cos_value_range_of_k_l765_765093


namespace regular_octagon_area_l765_765697

theorem regular_octagon_area (r : ℝ) (h : r = 5) : 
  let s := 5 * Real.sqrt(2 - Real.sqrt 2),
  let area := 4 * (5 * s) 
  in area = 100 * Real.sqrt(2 - Real.sqrt 2) :=
by
  sorry

end regular_octagon_area_l765_765697


namespace roots_of_quadratic_eq_l765_765615

theorem roots_of_quadratic_eq {x : ℝ} : 
  (x = 2 ∧ x = 4) ↔ 
  ¬(∃ x, (x - 2 = 0 ∧ x - 4 = 0).x) :=
sorry

end roots_of_quadratic_eq_l765_765615


namespace count_two_digit_prime_numbers_ending_in_3_l765_765252

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765252


namespace nuts_and_bolts_remaining_l765_765894

theorem nuts_and_bolts_remaining : 
  (let initial := 200 in
   let with_worker_A := initial + 50 in
   let with_worker_B := with_worker_A - 50 in
   let after_mary_fine := with_worker_B - 2/5 * with_worker_B in
   let with_mary_add := after_mary_fine + 100 in
   let with_peter_take := with_mary_add / 2 in
   with_mary_add - with_peter_take) = 110 :=
by
  sorry

end nuts_and_bolts_remaining_l765_765894


namespace congruent_triangles_l765_765657

noncomputable def triangle (A B C : Type _) := sorry -- Assume we have some definition

variables {A B C A' B' C' : Type _}

-- Defining the conditions of the problem
variable (d1 : AC = A'C')
variable (d2 : ∠ B = ∠ B')
variable (d3 : angle_bisector_length ABC B = angle_bisector_length A'B'C' B')

-- Goal: Prove the triangles are congruent
theorem congruent_triangles :
  (triangle ABC = triangle A'B'C' ∨ triangle ABC = triangle C'B'A') :=
sorry

end congruent_triangles_l765_765657


namespace abc_not_match_l765_765727

theorem abc_not_match (a b c : ℝ) : 
  ({a, b, c} : set ℝ) ≠ {a + 1, b ^ 2 + 2, c ^ 3 + 3} := 
by 
  sorry

end abc_not_match_l765_765727


namespace chessboard_same_color_l765_765589

/-- For an n x n chessboard painted black and white, where n ≥ 3, and a move is defined 
to choose any 2 x 2 square and flip all of its colors, we can turn all squares to 
the same color if and only if n ≡ 0 [MOD 4] -/
theorem chessboard_same_color (n : ℕ) (h : n ≥ 3) : 
  (∃ moves : list (ℕ × ℕ), ∀ (i j : ℕ), (i < n) → (j < n) → 
   (n % 4 = 0 ∨ (∀ (2x2_move : (ℕ × ℕ)), (flip_color 2x2_move.start_x 2x2_move.start_y))) = 
  (n % 4 = 0) :=
sorry

end chessboard_same_color_l765_765589


namespace proof_problem_l765_765833

section
variables (f : ℝ → ℝ) (a b : ℝ)

-- Given function definition:
def f_def : Prop := ∀ x, f x = (-2^x + b) / (2^(x+1) + a)

-- Condition 1: f is an odd function
def odd_function : Prop := ∀ x, f(-x) = -f(x)

-- Condition 2: f is a decreasing function on ℝ
def decreasing_function : Prop := ∀ x y, x < y → f(x) > f(y)

-- Main proof problem
theorem proof_problem :
  odd_function f ∧ decreasing_function f ∧ f_def f a b →
  (a = 2 ∧ b = 1) ∧ ∀ t, f(t^2 - 2*t) + f(2*t^2 - 1) < 0 → (t > 1 ∨ t < -1/3) :=
by
  intro h
  cases h with odd_f dec_f
  cases odd_f with f_odd f_decreasing
  sorry
end

end proof_problem_l765_765833


namespace count_two_digit_primes_ending_in_3_l765_765364

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765364


namespace two_digit_primes_with_units_digit_three_count_l765_765175

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765175


namespace find_pairs_l765_765768

theorem find_pairs (a b : ℕ) :
  (∃ (a b : ℕ), (b^2 - a ≠ 0) ∧ (a^2 - b ≠ 0) ∧ (a^2 + b) / (b^2 - a) ∈ ℤ ∧ (b^2 + a) / (a^2 - b) ∈ ℤ) → 
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end find_pairs_l765_765768


namespace fixed_point_through_PQ_l765_765041

-- Define the ellipse C equation using the given two points
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Fixed points that the ellipse passes through
def point1 := (-1 : ℝ, 2 * sqrt 2 / 3)
def point2 := (2 : ℝ, sqrt 5 / 3)

-- Define the equation of line PQ passing through the fixed point
def fixed_point := (0 : ℝ, -4/5)

theorem fixed_point_through_PQ : 
  ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a ≠ b) ∧ ellipse point1.1 point1.2 a b ∧ ellipse point2.1 point2.2 a b 
  ∧ (1/a^2 = 1/9) ∧ (1/b^2 = 1) ∧ ∀ k : ℝ, line_through_PQ k (0, 1) = λ x, (k^2 - 1) / (10 * k) * x - 4/5 
  := sorry

noncomputable def line_through_PQ (k : ℝ) (B : ℝ × ℝ) : ℝ → ℝ :=
  λ x, (k^2 - 1) / (10 * k) * x - 4/5

end fixed_point_through_PQ_l765_765041


namespace problem_l765_765065

variable {a : ℕ → ℝ}

-- Conditions
def geometric_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, (∏ i in finset.range n, a (i + 1)) = ℝ.T n

def condition1 : Prop := 
  geometric_seq a

def condition2 : Prop :=
  a 3 * a 4 * a 5 = 1

-- Assertions to Prove
theorem problem (h1 : condition1) (h2 : condition2) :
  (a 4 = 1) ∧ (ℝ.T 2 = ℝ.T 5) ∧ (a 1 = 1 → a 2 = 1) :=
sorry

end problem_l765_765065


namespace angle_sum_property_l765_765460

theorem angle_sum_property (angle_BAC angle_ABC angle_BCA z : ℝ) 
  (h1 : angle_BAC = 50) 
  (h2 : angle_ABC = 70) 
  (h3 : angle_BCA = 180 - angle_BAC - angle_ABC) 
  (h4 : ∠BDC = angle_BCA) 
  (h5 : angle_BDC + z = 90) 
  : z = 30 := 
by 
  have h6 : angle_BCA = 60, from calc
    angle_BCA = 180 - 50 - 70 : by rw [h1, h2, sub_self]
                ... = 60 : rfl,
  have h7 : ∠BDC = 60 := by rw [h4, h6],
  have h8 : 60 + z = 90 := h5,
  have h9 : z = 30 := by linarith,
  exact h9

end angle_sum_property_l765_765460


namespace rectangular_field_area_eq_l765_765910

-- Definitions based on the problem's conditions
def length (x : ℝ) := x
def width (x : ℝ) := 60 - x
def area (x : ℝ) := x * (60 - x)

-- The proof statement
theorem rectangular_field_area_eq (x : ℝ) (h₀ : x + (60 - x) = 60) (h₁ : area x = 864) :
  x * (60 - x) = 864 :=
by
  -- Using the provided conditions and definitions, we aim to prove the equation.
  sorry

end rectangular_field_area_eq_l765_765910


namespace polynomial_division_l765_765788

theorem polynomial_division :
  ∀ (x : ℝ), (8 * x^3 + 4 * x^2 - 6 * x - 9) / (x + 3) = 8 * x^2 - 20 * x + 54 - 171 / (x + 3) :=
by
  -- Define the polynomial p(x) = 8x^3 + 4x^2 - 6x -9
  let p := λ x : ℝ, 8 * x^3 + 4 * x^2 - 6 * x - 9
  -- Define the divisor q(x) = x + 3
  let q := λ x : ℝ, x + 3
  -- Observe (p x) / (q x) = 8 * x^2 - 20 * x + 54 - 171 / (x + 3)
  have h : ∀ (x : ℝ), p x / q x = 8 * x^2 - 20 * x + 54 - 171 / (x + 3) :=
    sorry

  exact h

end polynomial_division_l765_765788


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765276

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765276


namespace count_two_digit_prime_numbers_ending_in_3_l765_765242

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765242


namespace range_of_b_equation_of_circle_circle_passes_through_fixed_point_l765_765908

-- Given the conditions of the quadratic function and its intersections with the coordinate axes.
variables {b : ℝ}

-- Definitions of the corresponding mathematical functions and propositions
def quadratic_function (x : ℝ) := x^2 + 2 * x + b
def circle_eq (x y : ℝ) := x^2 + y^2 + 2 * x - (b + 1) * y + b

-- Conditions on b
axiom b_condition : b < 1 ∧ b ≠ 0

-- Statement 1: Intersection range
theorem range_of_b : b_condition → b < 1 ∧ b ≠ 0 := sorry

-- Statement 2: Equation of the circle
theorem equation_of_circle : 
  (∀ x y, circle_eq x y = 0 ↔ (x = 0 ∨ x = -2) ∧ y = 1) :=
sorry

-- Statement 3: Fixed point independent of b
theorem circle_passes_through_fixed_point :
  ∀ (b : ℝ), 
  b_condition →
  circle_eq 0 1 = 0 ∧ circle_eq (-2) 1 = 0 :=
sorry

end range_of_b_equation_of_circle_circle_passes_through_fixed_point_l765_765908


namespace colored_pencils_false_for_n_ge_2_l765_765516

theorem colored_pencils_false_for_n_ge_2 : ∀ (n : ℕ), n ≥ 2 → ¬ (∀ (box : list ℕ), (box.length = n → ∀ (i j : ℕ), i < n → j < n → i ≠ j → box.nth i = box.nth j)) :=
by
  sorry

end colored_pencils_false_for_n_ge_2_l765_765516


namespace two_digit_primes_with_units_digit_three_count_l765_765176

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765176


namespace circle_tangent_to_hyperbola_asymptotes_l765_765781

noncomputable def center_of_ellipse_right_focus (a b : ℝ) : ℝ :=
  real.sqrt (a^2 - b^2)

noncomputable def equation_of_circle_tangent_to_asymptotes (h_asymptotes : ℝ) (focus : (ℝ × ℝ)) : ℝ :=
  focus.fst^2 + focus.snd^2 - 10 * focus.fst + 9

theorem circle_tangent_to_hyperbola_asymptotes :
  let a_ellipse := 13
  let b_ellipse := 12
  let a_hyperbola := 3
  let b_hyperbola := 4
  let focus := (center_of_ellipse_right_focus a_ellipse b_ellipse, 0)
  equation_of_circle_tangent_to_asymptotes (4) (focus) = 0 :=
by
  let a_ellipse := 13
  let b_ellipse := 12
  let a_hyperbola := 3
  let b_hyperbola := 4
  let focus : ℝ × ℝ := (center_of_ellipse_right_focus a_ellipse b_ellipse, 0)
  let eq_circle := equation_of_circle_tangent_to_asymptotes (4) (focus)
  have : eq_circle = (x^2 + y^2 - 10*x + 9) := sorry
  exact sorry

end circle_tangent_to_hyperbola_asymptotes_l765_765781


namespace two_digit_primes_units_digit_3_count_l765_765359

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765359


namespace area_difference_invariant_l765_765937

theorem area_difference_invariant (b w : ℕ) (hb : 2 ≤ b) (hw : 2 ≤ w) :
  let n := b + w,
      area_B := some_area_calculation(2 * b),
      area_W := some_area_calculation(2 * w) in
  area_B - area_W = some_function_of_b_and_w(b, w) :=
sorry

end area_difference_invariant_l765_765937


namespace inequality_system_solution_l765_765577

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l765_765577


namespace orthocenter_lies_on_g_l765_765925

theorem orthocenter_lies_on_g
  (A B C D M N : Point)
  (g : Line)
  (I1 I2 I3 : Point)
  (h1 : is_circumscribed_quadrilateral A B C D)
  (h2 : lies_on g A)
  (h3 : meets_in_segment g B C M)
  (h4 : meets_line g C D N)
  (h5 : incenter I1 (Triangle A B M))
  (h6 : incenter I2 (Triangle M N C))
  (h7 : incenter I3 (Triangle N D A)) :
  orthocenter (Triangle I1 I2 I3) ∈ g := 
sorry

end orthocenter_lies_on_g_l765_765925


namespace count_two_digit_primes_with_units_digit_3_l765_765191

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765191


namespace find_derivative_at_2_l765_765062

-- Define f(x) as a generic quadratic function symmetric about x = 1
def f (a b : ℝ) (x : ℝ) : ℝ := a * (x - 1) ^ 2 + b

-- The derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 2 * a * (x - 1)

-- The main theorem asserting the result
theorem find_derivative_at_2 (a b : ℝ) (h_slope : f' a b 0 = -2) : f' a b 2 = 2 :=
by
  -- Proof goes here
  sorry

end find_derivative_at_2_l765_765062


namespace probability_same_row_l765_765454

theorem probability_same_row (rows cols students: ℕ) (h_rows: rows = 3) (h_cols: cols = 2) (h_students: students = 6) :
  let total_selections := Nat.choose students 2,
      same_row_selections := Nat.choose rows 1 * Nat.choose cols 2 in
  total_selections > 0 → × ((same_row_selections: ℚ ) / total_selections = 1/5) :=
sorry

end probability_same_row_l765_765454


namespace num_two_digit_primes_with_units_digit_three_l765_765140

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765140


namespace negation_of_p_l765_765083

open set

variable {x : ℝ}

/-- Proposition p: For all x in (1, +∞), x^3 + 1 > 8x -/
def p : Prop := ∀ x ∈ Ioo 1 ⊤, x^3 + 1 > 8 * x

/-- The negation of proposition p -/
def not_p : Prop := ∃ x ∈ Ioo 1 ⊤, x^3 + 1 ≤ 8 * x

theorem negation_of_p : ¬p ↔ not_p :=
sorry

end negation_of_p_l765_765083


namespace parallel_planes_l765_765794

variables {α β : Type*} [Plane α] [Plane β] 
variables (l m : Line) [Skew l m] 

def is_parallel (p q : Type*) [Plane p] [Plane q] : Prop :=
  ∀ (x : Point), ∃ (y : Point), dist x y = 0

def parallel_lines_to_planes (l : Line) (p q : Type*) [Plane p] [Plane q] : Prop :=
  is_parallel l p ∧ is_parallel l q

theorem parallel_planes (α β : Type*) [Plane α] [Plane β] 
    (l m : Line) [Skew l m]
    (h1: parallel_lines_to_planes l α β) 
    (h2: parallel_lines_to_planes m α β) : 
    is_parallel α β :=
sorry

end parallel_planes_l765_765794


namespace count_integers_with_digit_sum_l765_765127

theorem count_integers_with_digit_sum :
  let count := (finset.range 600).filter (λ n, n >= 400 ∧ (n.digits.sum = 18)).card
  count = 20 :=
by
  sorry

end count_integers_with_digit_sum_l765_765127


namespace count_two_digit_prime_numbers_ending_in_3_l765_765246

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765246


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765343

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765343


namespace num_two_digit_primes_with_units_digit_three_l765_765136

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765136


namespace num_integers_digit_sum_18_400_600_l765_765122

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem num_integers_digit_sum_18_400_600 : 
  {n : ℕ // 400 ≤ n ∧ n < 600 ∧ digit_sum n = 18}.card = 21 := sorry

end num_integers_digit_sum_18_400_600_l765_765122


namespace two_digit_primes_with_units_digit_three_l765_765220

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765220


namespace two_digit_primes_units_digit_3_count_l765_765353

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765353


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765340

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765340


namespace total_shaded_area_l765_765626

-- Definitions given as conditions in the problem.
def strip_width : ℝ := 2
def overlap_angle : ℝ := 30
def rectangle_width : ℝ := 1
def rectangle_length : ℝ := 2

-- Translate the problem statement to Lean.
theorem total_shaded_area : 
  let rhombus_area := (strip_width * rectangle_length) / 2,
      rectangle_area := rectangle_width * rectangle_length in
  rhombus_area + rectangle_area = 6 :=
by
  let rhombus_area := (strip_width * rectangle_length) / 2
  let rectangle_area := (rectangle_width * rectangle_length)
  have h_total_area : rhombus_area + rectangle_area = 6, sorry
  exact h_total_area

end total_shaded_area_l765_765626


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765288

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765288


namespace count_prime_units_digit_3_eq_6_l765_765297

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765297


namespace common_roots_l765_765593

theorem common_roots (a b c d : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (k : ℂ) (hP : a * k^3 + b * k^2 + c * k + d = 0) (hQ : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = complex.I ∨ k = -complex.I :=
begin
  sorry
end

end common_roots_l765_765593


namespace find_f1_l765_765027

def f : ℤ → ℤ
| x => if x ≥ 6 then x - 5 else f (x + 2)

theorem find_f1 : f 1 = 2 :=
by
  -- Proof goes here...
  sorry

end find_f1_l765_765027


namespace line_through_intersection_and_origin_l765_765981

-- Definitions of the lines
def l1 (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l2 (x y : ℝ) : Prop := y = 1 - x

-- Prove that the line passing through the intersection of l1 and l2 and the origin has the equation 3x + 2y = 0
theorem line_through_intersection_and_origin (x y : ℝ) 
  (h1 : 2 * x - y + 7 = 0) (h2 : y = 1 - x) : 3 * x + 2 * y = 0 := 
sorry

end line_through_intersection_and_origin_l765_765981


namespace digit_sum_divisible_by_9_l765_765742

theorem digit_sum_divisible_by_9 (n : ℕ) (h : n < 10) : 
  (8 + 6 + 5 + n + 7 + 4 + 3 + 2) % 9 = 0 ↔ n = 1 := 
by sorry 

end digit_sum_divisible_by_9_l765_765742


namespace complex_number_in_first_quadrant_l765_765994

theorem complex_number_in_first_quadrant :
  let z := (Complex.I / (1 + Complex.I)) in
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l765_765994


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765258

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765258


namespace inequality_system_solution_l765_765555

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l765_765555


namespace problem_statement_l765_765026

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := 2⁻¹
noncomputable def c : ℝ := Real.log 6 / Real.log 5

theorem problem_statement : b < a ∧ a < c := by
  sorry

end problem_statement_l765_765026


namespace natural_number_pairs_int_l765_765771

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l765_765771


namespace count_two_digit_primes_with_units_digit_3_l765_765155

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765155


namespace barbara_removed_total_sheets_l765_765720

theorem barbara_removed_total_sheets :
  let bundles_colored := 3
  let bunches_white := 2
  let heaps_scrap := 5
  let sheets_per_bunch := 4
  let sheets_per_bundle := 2
  let sheets_per_heap := 20
  bundles_colored * sheets_per_bundle + bunches_white * sheets_per_bunch + heaps_scrap * sheets_per_heap = 114 :=
by
  sorry

end barbara_removed_total_sheets_l765_765720


namespace number_of_valid_pairs_l765_765851

open BigOperators

-- Definition of the polynomial
def polynomial (coeffs : Fin 2021 → ℤ) (x : ℕ) : ℤ := 
  ∑ i in Finset.range 2021, coeffs ⟨i, by norm_num⟩ * x ^ i

-- Main statement
theorem number_of_valid_pairs (coeffs : Fin 2021 → ℤ) :
  let f := polynomial coeffs
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ (p n : ℕ), (p, n) ∈ pairs → 
        Nat.Prime p ∧ p ^ 2 < n ∧ n < 2020 ∧ ∀ i : ℕ, i ≤ n → 
          (Nat.choose n i + f i) % p = 0) ∧ pairs.card = 8 := 
by
  -- Use appropriate tactics to structure the theorem proof
  sorry

end number_of_valid_pairs_l765_765851


namespace count_two_digit_prime_numbers_ending_in_3_l765_765243

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765243


namespace segments_leq_n_squared_l765_765805

-- Define the problem parameters
variables (n : ℕ) (C : set (ℝ × ℝ)) (points : finset (ℝ × ℝ))

-- Define the unit circle and the points on it
def unit_circle (C : set (ℝ × ℝ)) : Prop := ∀ (p : ℝ × ℝ), p ∈ C ↔ p.1^2 + p.2^2 = 1
def on_unit_circle (C : set (ℝ × ℝ)) (points : finset (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, unit_circle C

-- Define the segment length and the condition where its length is greater than sqrt(2)
def segment_length_gt_sqrt2 (p q : ℝ × ℝ) : Prop := (p.1 - q.1)^2 + (p.2 - q.2)^2 > 2
def num_segments_gt_sqrt2 (points : finset (ℝ × ℝ)) : ℕ :=
  (points.val.bind (λ p, points.val.bind (λ q, if segment_length_gt_sqrt2 p q then [p, q] else []))).length / 2

-- Main theorem statement
theorem segments_leq_n_squared (hC : unit_circle C) (hp : on_unit_circle C points) (hn : points.card = n) :
  3 * num_segments_gt_sqrt2 points ≤ n^2 :=
begin
  sorry
end

end segments_leq_n_squared_l765_765805


namespace count_two_digit_primes_with_units_digit_3_l765_765148

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765148


namespace find_pairs_l765_765769

theorem find_pairs (a b : ℕ) :
  (∃ (a b : ℕ), (b^2 - a ≠ 0) ∧ (a^2 - b ≠ 0) ∧ (a^2 + b) / (b^2 - a) ∈ ℤ ∧ (b^2 + a) / (a^2 - b) ∈ ℤ) → 
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end find_pairs_l765_765769


namespace swimming_speed_in_still_water_l765_765691

theorem swimming_speed_in_still_water (v : ℝ) 
  (h_current_speed : 2 = 2) 
  (h_time_distance : 7 = 7) 
  (h_effective_speed : v - 2 = 14 / 7) : 
  v = 4 :=
sorry

end swimming_speed_in_still_water_l765_765691


namespace total_distance_is_10_83_l765_765885

-- Define the times in hours
def walk_time_hours : ℝ := 30 / 60
def run_time_hours : ℝ := 40 / 60
def cycle_time_hours : ℝ := 20 / 60

-- Define the speeds in mph
def walk_speed_mph : ℝ := 3
def run_speed_mph : ℝ := 8
def cycle_speed_mph : ℝ := 12

-- Calculate the distances in miles
def walk_distance : ℝ := walk_speed_mph * walk_time_hours
def run_distance : ℝ := run_speed_mph * run_time_hours
def cycle_distance : ℝ := cycle_speed_mph * cycle_time_hours

-- Total distance traveled
def total_distance : ℝ := walk_distance + run_distance + cycle_distance

-- Prove the total distance is 10.83 miles
theorem total_distance_is_10_83 : total_distance = 10.83 :=
by sorry

end total_distance_is_10_83_l765_765885


namespace aardvark_total_distance_l765_765625

-- Definitions
def radius_smaller_circle : ℝ := 15
def radius_larger_circle : ℝ := 30

def larger_circle_half_circumference : ℝ := (1 / 2) * 2 * Real.pi * radius_larger_circle
def tangent_distance_between_circles : ℝ := 2 * (radius_larger_circle - radius_smaller_circle)
def smaller_circle_half_circumference : ℝ := (1 / 2) * 2 * Real.pi * radius_smaller_circle

-- Theorem statement
theorem aardvark_total_distance : 
  (larger_circle_half_circumference + tangent_distance_between_circles + smaller_circle_half_circumference) = 45 * Real.pi + 30 :=
by
  -- Skipping the proof
  sorry

end aardvark_total_distance_l765_765625


namespace max_area_convex_quadrilateral_l765_765030

theorem max_area_convex_quadrilateral
  (T1 T2 T3 : ℝ)
  (h1 : T1 = 25 ∨ T1 = 24)
  (h2 : T2 = 25 ∨ T2 = 24)
  (h3 : T3 = 25 ∨ T3 = 24)
  (h_eq : T1 * T3 = T2 * (T1 * T3 / T2)) :
  T1 + T2 + T3 + (T1 * T3 / T2) ≤ 100 + 1 / 24 :=
begin
  sorry,
end

end max_area_convex_quadrilateral_l765_765030


namespace inequality_system_solution_l765_765573

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l765_765573


namespace problem_solution_l765_765799

theorem problem_solution (a : ℝ) (b : ℝ) (h₁ : a < 0) (h₂ : -1 < b) (h₃ : b < 0) : ab > ab^2 > a :=
by
  sorry

end problem_solution_l765_765799


namespace parallelogram_fourth_vertex_distance_l765_765597

theorem parallelogram_fourth_vertex_distance (d1 d2 d3 d4 : ℝ) (h1 : d1 = 1) (h2 : d2 = 3) (h3 : d3 = 5) :
    d4 = 7 :=
sorry

end parallelogram_fourth_vertex_distance_l765_765597


namespace max_distance_ellipse_to_line_l765_765785

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 4) = 1

def line (x y : ℝ) : Prop :=
  x + 2 * y - real.sqrt 2 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_ellipse_to_line : 
  ∃ (x y : ℝ), ellipse x y ∧ (∀ x' y', ellipse x' y' → distance x y x' y' ≤ real.sqrt 11) :=
sorry

end max_distance_ellipse_to_line_l765_765785


namespace number_of_two_digit_primes_with_units_digit_three_l765_765213

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765213


namespace one_fourth_of_8_point_8_l765_765004

-- Definition of taking one fourth of a number
def oneFourth (x : ℝ) : ℝ := x / 4

-- Problem statement: One fourth of 8.8 is 11/5 when expressed as a simplified fraction
theorem one_fourth_of_8_point_8 : oneFourth 8.8 = 11 / 5 := by
  sorry

end one_fourth_of_8_point_8_l765_765004


namespace solve_inequality_system_l765_765557

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765557


namespace bridges_lead_to_shore_l765_765478

theorem bridges_lead_to_shore
  (d1 d2 d3 d4 d5 d6 d7 : ℕ)
  (h1 : d1 ∈ {1, 3, 5})
  (h2 : d2 ∈ {1, 3, 5})
  (h3 : d3 ∈ {1, 3, 5})
  (h4 : d4 ∈ {1, 3, 5})
  (h5 : d5 ∈ {1, 3, 5})
  (h6 : d6 ∈ {1, 3, 5})
  (h7 : d7 ∈ {1, 3, 5}) :
  ∃ dshore, dshore ∈ {0, 2, 4, 6} ∧ (d1 + d2 + d3 + d4 + d5 + d6 + d7 + dshore) % 2 = 0 := sorry

end bridges_lead_to_shore_l765_765478


namespace nat_pairs_satisfy_conditions_l765_765763

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l765_765763


namespace original_visual_range_l765_765675

theorem original_visual_range
  (V : ℝ)
  (h1 : 2.5 * V = 150) :
  V = 60 :=
by
  sorry

end original_visual_range_l765_765675


namespace percentage_of_Hindu_boys_l765_765897

theorem percentage_of_Hindu_boys (total_boys : ℕ) (muslim_percentage : ℕ) (sikh_percentage : ℕ)
  (other_community_boys : ℕ) (H : total_boys = 850) (H1 : muslim_percentage = 44) 
  (H2 : sikh_percentage = 10) (H3 : other_community_boys = 153) :
  let muslim_boys := muslim_percentage * total_boys / 100
  let sikh_boys := sikh_percentage * total_boys / 100
  let non_hindu_boys := muslim_boys + sikh_boys + other_community_boys
  let hindu_boys := total_boys - non_hindu_boys
  (hindu_boys * 100 / total_boys : ℚ) = 28 := 
by
  sorry

end percentage_of_Hindu_boys_l765_765897


namespace partition_positive_integers_no_arithmetic_l765_765519

noncomputable def exists_partition (N : Set ℕ) : Prop :=
  ∃ (A B : Set ℕ), (A ∪ B = N ∧ A ∩ B = ∅) ∧
  (∀ a b c ∈ A, a < b ∧ b < c → b - a ≠ c - b) ∧
  (∀ s : ℕ → ℕ, (∀ n, s n ∈ B ∧ s (n+1) - s n = s 1 - s 0) → ∃ n, s n ∉ B)

theorem partition_positive_integers_no_arithmetic (N := {n : ℕ | 0 < n}) :
  exists_partition N :=
sorry

end partition_positive_integers_no_arithmetic_l765_765519


namespace point_not_in_region_l765_765692

theorem point_not_in_region : ¬ (3 * 2 + 2 * 0 < 6) :=
by simp [lt_irrefl]

end point_not_in_region_l765_765692


namespace nearest_integer_to_exp_is_752_l765_765642

noncomputable def nearest_integer_to_exp := (3 + Real.sqrt 5) ^ 4

theorem nearest_integer_to_exp_is_752 : Int.round nearest_integer_to_exp = 752 := 
sorry

end nearest_integer_to_exp_is_752_l765_765642


namespace count_prime_units_digit_3_eq_6_l765_765293

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765293


namespace count_two_digit_primes_with_units_digit_three_l765_765409

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765409


namespace parabola_axis_l765_765600

theorem parabola_axis (p : ℝ) (h_parabola : ∀ x : ℝ, y = x^2 → x^2 = y) : (y = - p / 2) :=
by
  sorry

end parabola_axis_l765_765600


namespace solve_inequality_system_l765_765559

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765559


namespace garden_roller_length_l765_765978

theorem garden_roller_length
  (diameter : ℝ)
  (total_area : ℝ)
  (revolutions : ℕ)
  (pi : ℝ)
  (circumference : ℝ)
  (area_per_revolution : ℝ)
  (length : ℝ)
  (h1 : diameter = 1.4)
  (h2 : total_area = 44)
  (h3 : revolutions = 5)
  (h4 : pi = (22 / 7))
  (h5 : circumference = pi * diameter)
  (h6 : area_per_revolution = total_area / (revolutions : ℝ))
  (h7 : area_per_revolution = circumference * length) :
  length = 7 := by
  sorry

end garden_roller_length_l765_765978


namespace evaluate_expression_l765_765724

-- Define the condition
variable (x y z : ℝ)

axiom condition : x * y * z = 1

-- State the theorem
theorem evaluate_expression (hx : x * y * z = 1) :
  (1 / (1 + x + x * y) + 1 / (1 + y + y * z) + 1 / (1 + z + z * x) = 1) :=
begin
  sorry
end

end evaluate_expression_l765_765724


namespace maximum_value_of_f_l765_765009

noncomputable def f (x : ℝ) : ℝ := (-x^2 + x - 4) / x

theorem maximum_value_of_f :
  ∃ x : ℝ, x > 0 ∧ f x = -3 ∧ (∀ y > 0, f y ≤ f x) :=
by
  use 2
  have h1 : f 2 = -3 :=
    by
      have : (-2^2 + 2 - 4) = -6 := by norm_num
      show (-6) / 2 = -3 from by norm_num
  split
  exact zero_lt_two
  split
  exact h1
  intro y hy
  have : f y ≤ -3 := sorry
  exact this

end maximum_value_of_f_l765_765009


namespace range_of_PF1_minus_PF2_l765_765086

noncomputable def ellipse_property (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : Prop :=
  ∃ f : ℝ, f = (2 * Real.sqrt 5 / 5) * x0 ∧ f > 0 ∧ f < 2

theorem range_of_PF1_minus_PF2 (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : 
  ellipse_property x0 h1 h2 := by
  sorry

end range_of_PF1_minus_PF2_l765_765086


namespace number_of_two_digit_primes_with_units_digit_three_l765_765219

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765219


namespace solve_inequality_system_l765_765535

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765535


namespace count_two_digit_primes_with_units_digit_three_l765_765412

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765412


namespace count_two_digit_primes_with_units_digit_3_l765_765326

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765326


namespace number_of_valid_functions_l765_765944

def M : Set ℤ := {-2, 0, 1}
def N : Set ℤ := {1, 2, 3, 4, 5}

def valid_function (f : ℤ → ℤ) : Prop :=
  ∀ x ∈ M, (x + f x + x * f x) % 2 = 1

theorem number_of_valid_functions : 
  finset.card (finset.univ.filter valid_function) = 27 :=
sorry

end number_of_valid_functions_l765_765944


namespace count_two_digit_primes_ending_in_3_l765_765381

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765381


namespace initial_mixture_volume_is_165_l765_765687

noncomputable def initial_volume_of_mixture (initial_milk_volume initial_water_volume water_added final_milk_water_ratio : ℕ) : ℕ :=
  if (initial_milk_volume + initial_water_volume) = 5 * (initial_milk_volume / 3) &&
     initial_water_volume = 2 * (initial_milk_volume / 3) &&
     water_added = 66 &&
     final_milk_water_ratio = 3 / 4 then
    5 * (initial_milk_volume / 3)
  else
    0

theorem initial_mixture_volume_is_165 :
  ∀ initial_milk_volume initial_water_volume water_added final_milk_water_ratio,
    initial_volume_of_mixture initial_milk_volume initial_water_volume water_added final_milk_water_ratio = 165 :=
by
  intros
  sorry

end initial_mixture_volume_is_165_l765_765687


namespace appropriate_for_regression_analysis_l765_765651

theorem appropriate_for_regression_analysis : 
  (∀ r a, a = π * r^2) →      -- The area of a circle and its radius have a functional relationship
  (¬ ∃ g c, g = color_blindness c) →  -- There is no relationship between color blindness and gender
  (¬ ∃ h p, h = height p ∧ p = academic_performance) → -- There is no relationship between height and academic performance
  (∃ h w, correlated h w) →   -- There is a correlation between a person's height and weight
  most_appropriate_for_regression B := -- Therefore, the most appropriate for regression analysis is B
sorry

end appropriate_for_regression_analysis_l765_765651


namespace intersection_of_diagonals_on_circle_l765_765789

open EuclideanGeometry Real

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem intersection_of_diagonals_on_circle 
  (M N L K : ℝ × ℝ) 
  (P : ℝ × ℝ := midpoint M N)
  (Q : ℝ × ℝ := midpoint K L) :
  ∃ C : set (ℝ × ℝ), (∀ O : ℝ × ℝ, (O = intersection_of_diagonals (rectangle_with_points M N L K)) → O ∈ C) ∧ is_circle_with_diameter C P Q := 
sorry

end intersection_of_diagonals_on_circle_l765_765789


namespace two_digit_primes_units_digit_3_count_l765_765349

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765349


namespace find_b_l765_765590

def eval_mod (n : ℕ) (base : ℕ) : ℤ :=
  let digits := [
    2, 3, 4, 8, 4, 5, 6, 2, 3
  ]
  digits.foldl (λ acc (d, i), acc + d * base ^ i) 0

theorem find_b (b : ℤ) (h1 : 0 ≤ b ∧ b ≤ 20) (h2 : (eval_mod 73 17) - b % 17 = 0) : b = 1 := by
  sorry

end find_b_l765_765590


namespace f_even_l765_765603

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_const : ¬ (∀ x y : ℝ, f x = f y)
axiom f_equiv1 : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_equiv2 : ∀ x : ℝ, f (1 + x) = -f x

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  sorry

end f_even_l765_765603


namespace two_digit_primes_with_units_digit_three_l765_765221

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765221


namespace solve_inequality_system_l765_765566

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765566


namespace mikaela_tutoring_hours_l765_765953

-- Defining the conditions
def earns_per_hour := 10
def hours_more_second_month := 5
def spent_ratio := 4 / 5
def saved_amount := 150

-- Main statement: Proving that h (hours tutored in the first month) is 35
theorem mikaela_tutoring_hours :
  ∃ (h : ℕ), 
    (let earnings_first_month := earns_per_hour * h in
     let earnings_second_month := earns_per_hour * (h + hours_more_second_month) in
     let total_earnings := earnings_first_month + earnings_second_month in
     let total_saving_ratio := 1 - spent_ratio in
     total_earnings * total_saving_ratio = saved_amount * 5) 
    → h = 35 :=
begin
  sorry
end

end mikaela_tutoring_hours_l765_765953


namespace common_ratio_of_geometric_sequence_l765_765440

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h4 : ∀ n, a n ≤ a (n + 1)) :
  q = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l765_765440


namespace evaluate_expression_l765_765746

theorem evaluate_expression : 2 + (2 / (2 + (2 / (2 + 3)))) = 17 / 6 := 
by
  sorry

end evaluate_expression_l765_765746


namespace collinear_MNO_l765_765811

-- Define the geometrical setup and the problem
variables {A B C D E F G L M N O : Type}
variables [incircle : has_circumcircle A B C D] -- Circumcircles exist by geometry problem setup
variables [is_inside : is_inside D (triangle A B C)]
variables [circ1 : lies_on_circumcircle E (triangle A B D)]
variables [circ2 : lies_on_circumcircle F (triangle B C D)]
variables [circ3 : lies_on_circumcircle G (triangle C A D)]
variables [circ_main : lies_on_circumcircle L (triangle A B C)]
variables [line_intersects1 : line_intersects LE A B M]
variables [line_intersects2 : line_intersects LF B C N]
variables [line_intersects3 : line_intersects LG C A O]

theorem collinear_MNO : is_collinear M N O := sorry

end collinear_MNO_l765_765811


namespace count_two_digit_primes_with_units_digit_three_l765_765404

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765404


namespace geometric_sequence_limit_l765_765069

theorem geometric_sequence_limit (a₁ : ℝ) (x : ℝ) (h_binomial : (x + 1/√x)^6 ≥ 15 * x^3) (h_x_gt : x > 1) :
  (∀ n : ℕ, n ≥ 3 → 
     let S_n := (a₁ * (x^n - 1)) / (x - 1) in
     let S_n_2 := (a₁ * x^2 * (x^(n-2) - 1)) / (x - 1) in
     ∃ l : ℝ, l = 1 ∧
     filter.tendsto (λ n : ℕ, S_n / S_n_2) 
       filter.at_top (nhds l)) :=
sorry

end geometric_sequence_limit_l765_765069


namespace total_books_together_l765_765966

-- Given conditions
def SamBooks : Nat := 110
def JoanBooks : Nat := 102

-- Theorem to prove the total number of books they have together
theorem total_books_together : SamBooks + JoanBooks = 212 := 
by
  sorry

end total_books_together_l765_765966


namespace verify_integer_pairs_l765_765756

open Nat

theorem verify_integer_pairs (a b : ℕ) :
  (∃ k1 : ℤ, ↑(a^2) + ↑b = k1 * (↑(b^2) - ↑a)) ∧
  (∃ k2 : ℤ, ↑(b^2) + ↑a = k2 * (↑(a^2) - ↑b)) →
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ 
  (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end verify_integer_pairs_l765_765756


namespace nearest_integer_power_l765_765641

theorem nearest_integer_power (x : ℝ) (x_val : x = 3 + sqrt 5) : 
  (Real.round (x^4) : ℤ) = 376 :=
by
  sorry

end nearest_integer_power_l765_765641


namespace count_two_digit_primes_with_units_digit_3_l765_765150

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765150


namespace quadrilateral_area_l765_765633

theorem quadrilateral_area 
  (AB BC DC : ℝ)
  (hAB_perp_BC : true)
  (hDC_perp_BC : true)
  (hAB_eq : AB = 8)
  (hDC_eq : DC = 3)
  (hBC_eq : BC = 10) : 
  (1 / 2 * (AB + DC) * BC = 55) :=
by 
  sorry

end quadrilateral_area_l765_765633


namespace count_two_digit_primes_with_units_digit_3_l765_765319

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765319


namespace prime_factors_of_product_l765_765856

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : List ℕ :=
  -- Assume we have a function that returns a list of prime factors of n
  sorry

def num_distinct_primes (n : ℕ) : ℕ :=
  (prime_factors n).toFinset.card

theorem prime_factors_of_product :
  num_distinct_primes (85 * 87 * 91 * 94) = 8 :=
by
  have prod_factorizations : 85 = 5 * 17 ∧ 87 = 3 * 29 ∧ 91 = 7 * 13 ∧ 94 = 2 * 47 := 
    by sorry -- each factorization step
  sorry

end prime_factors_of_product_l765_765856


namespace count_two_digit_primes_ending_in_3_l765_765369

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765369


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765262

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765262


namespace max_value_of_expression_l765_765934

theorem max_value_of_expression (x y z : ℝ) (h₁ : x^2 + y^2 + z^2 = 5) : x + 2 * y + 3 * z ≤ sqrt 70 :=
sorry

end max_value_of_expression_l765_765934


namespace range_of_alpha_l765_765875

noncomputable def is_zero (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

theorem range_of_alpha :
  (∀ f : ℝ → ℝ, (f = λ x, cos (2 * x) + sin (2 * x + π / 6)) →
    (∀ α : ℝ, (∀ x : ℝ, 0 < x ∧ x < α →
      (is_zero f x → (0 < real.roots f (Ioo 0 α)).card = 2))) → 
    (5 * π / 6 < α ∧ α ≤ 4 * π / 3))) :=
begin
  sorry
end

end range_of_alpha_l765_765875


namespace mike_total_spending_is_497_50_l765_765954

def rose_bush_price : ℝ := 75
def rose_bush_count : ℕ := 6
def rose_bush_discount : ℝ := 0.10
def friend_rose_bushes : ℕ := 2
def tax_rose_bushes : ℝ := 0.05

def aloe_price : ℝ := 100
def aloe_count : ℕ := 2
def tax_aloe : ℝ := 0.07

def calculate_total_cost_for_mike : ℝ :=
  let total_rose_bush_cost := rose_bush_price * rose_bush_count
  let discount := total_rose_bush_cost * rose_bush_discount
  let cost_after_discount := total_rose_bush_cost - discount
  let sales_tax_rose_bushes := tax_rose_bushes * cost_after_discount
  let cost_rose_bushes_after_tax := cost_after_discount + sales_tax_rose_bushes

  let total_aloe_cost := aloe_price * aloe_count
  let sales_tax_aloe := tax_aloe * total_aloe_cost

  let total_cost_friend_rose_bushes := friend_rose_bushes * (rose_bush_price - (rose_bush_price * rose_bush_discount))
  let sales_tax_friend_rose_bushes := tax_rose_bushes * total_cost_friend_rose_bushes
  let total_cost_friend := total_cost_friend_rose_bushes + sales_tax_friend_rose_bushes

  let total_mike_rose_bushes := cost_rose_bushes_after_tax - total_cost_friend

  let total_cost_mike_aloe := total_aloe_cost + sales_tax_aloe

  total_mike_rose_bushes + total_cost_mike_aloe

theorem mike_total_spending_is_497_50 : calculate_total_cost_for_mike = 497.50 := by
  sorry

end mike_total_spending_is_497_50_l765_765954


namespace masha_combinations_seven_beads_even_beads_combinations_odd_permutations_even_odd_possible_l765_765950

-- Statement 1: Number of combinations Masha got is 127 with 7 beads
theorem masha_combinations_seven_beads : 
  ∀ (n : ℕ), n = 7 → (2^n - 1) = 127 :=
by sorry

-- Statement 2: For an even number of beads, number of combinations excluding the empty set is always odd
theorem even_beads_combinations_odd : 
  ∀ (n : ℕ), n % 2 = 0 → ¬even (2^n - 1) :=
by sorry

-- Statement 3: If the order matters, the total permutations can be even or odd
theorem permutations_even_odd_possible :
  ∀ (n : ℕ), n > 0 → 
  (∑ k in finset.range (n + 1), nat.perm n k) % 2 = 0 ∨
  (∑ k in finset.range (n + 1), nat.perm n k) % 2 = 1 :=
by sorry

end masha_combinations_seven_beads_even_beads_combinations_odd_permutations_even_odd_possible_l765_765950


namespace inversion_count_seq1_inversion_count_seq2_odd_inversion_count_seq2_even_inversion_count_reversed_seq_l765_765701

-- 1. Inversion count for a_n = -2n + 19
theorem inversion_count_seq1 : 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 100 → (∑ i in Finset.range (n - 1), i) = 4950 := 
by
  sorry

-- 2. Inversion count for piecewise sequence
def piecewise_seq (n : ℕ) : ℚ :=
  if n % 2 = 1 then (1/3)^n else -n/(n+1)

theorem inversion_count_seq2_odd : 
  ∀ (k : ℕ), 1 ≤ k ∧ k % 2 = 1 → (∑ i in Finset.range k, i / 2) + (∑ j in Finset.range ((k-1) / 2), j) = (3 * k^2 - 4 * k + 1) / 8 :=
by
  sorry

theorem inversion_count_seq2_even : 
  ∀ (k : ℕ), 1 ≤ k ∧ k % 2 = 0 → (∑ i in Finset.range k, i / 2) + (∑ j in Finset.range (k / 2), j) = (3 * k^2 - 2 * k) / 8 :=
by
  sorry

-- 3. Inversion count for reversed sequence
theorem inversion_count_reversed_seq : 
  ∀ (n a : ℕ), 0 ≤ a ∧ a ≤ n * (n - 1) / 2 → (∑ i in Finset.range n, (n - 1 - i) - a) = n * (n - 1) / 2 - a :=
by
  sorry

end inversion_count_seq1_inversion_count_seq2_odd_inversion_count_seq2_even_inversion_count_reversed_seq_l765_765701


namespace count_two_digit_primes_with_units_digit_3_l765_765316

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765316


namespace overall_rate_of_profit_l765_765685

noncomputable def costPriceA : ℝ := 50
noncomputable def costPriceB : ℝ := 120
noncomputable def costPriceC : ℝ := 75

noncomputable def sellPriceA : ℝ := 90
noncomputable def sellPriceB : ℝ := 150
noncomputable def sellPriceC : ℝ := 110

noncomputable def totalCP : ℝ := costPriceA + costPriceB + costPriceC
noncomputable def totalSP : ℝ := sellPriceA + sellPriceB + sellPriceC
noncomputable def totalProfit : ℝ := totalSP - totalCP
noncomputable def rateOfProfit : ℝ := (totalProfit / totalCP) * 100

theorem overall_rate_of_profit : 
    rateOfProfit ≈ 42.86 := 
by
  sorry

end overall_rate_of_profit_l765_765685


namespace hardy_inequality_l765_765936

theorem hardy_inequality (n : ℕ) (a : ℕ → ℝ) (p : ℕ) 
  (h0 : ∀ k, k ≤ n → a k ≥ 0)
  (hp : p > 1) :
  (∑ k in Finset.range(n+1).filter (λ k, k > 0), 
    ((∑ i in Finset.range(k+1).filter (λ i, i > 0), a i) / k) ^ p) 
  ≤
  ((p : ℝ) / (p - 1)) ^ p * (∑ k in Finset.range(n+1).filter (λ k, k > 0), (a k) ^ p) := 
sorry

end hardy_inequality_l765_765936


namespace find_pairs_l765_765754

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l765_765754


namespace system_linear_diff_eq_1_part1_solution_system_linear_diff_eq_2_part2_solution_l765_765006

noncomputable def Part1_solution (C1 C2 : ℝ) : ℝ → ℝ × ℝ :=
λ x, (C1 * Real.exp (-x) + C2 * Real.exp (2 * x) - 6 * x ^ 2 + 6 * x - 9,
      (1 / 4) * C1 * Real.exp (-x) + C2 * Real.exp (2 * x) - 3 * x ^ 2 - 3)

theorem system_linear_diff_eq_1_part1_solution
  (C1 C2 : ℝ) :
  ∀ (x : ℝ),
    let (y, z) := Part1_solution C1 C2 x in
    (deriv y + 2 * y - 4 * z = 0) ∧ (deriv z + y - 3 * z = 3 * x ^ 2) := by
  sorry

noncomputable def Part2_solution (C1 C2 C3 : ℝ) : ℝ → ℝ × ℝ × ℝ :=
λ x, (C1 + C2 * Real.cos x + C3 * Real.sin x + Real.exp x,
      2 * C1 + (1/2) * (C3 - C2) * Real.cos x - (1/2) * (C3 + C2) * Real.sin x,
      3 * C1 - (1/2) * (C2 + C3) * Real.cos x + (1/2) * (C2 - C3) * Real.sin x + Real.exp x)

theorem system_linear_diff_eq_2_part2_solution
  (C1 C2 C3 : ℝ) :
  ∀ (x : ℝ),
    let (u, v, w) := Part2_solution C1 C2 C3 x in
    (6 * deriv u - u - 7 * v + 5 * w = 10 * Real.exp x) ∧
    (2 * deriv v + u + v - w = 0) ∧
    (3 * deriv w - u + 2 * v - w = Real.exp x) := by
  sorry

end system_linear_diff_eq_1_part1_solution_system_linear_diff_eq_2_part2_solution_l765_765006


namespace Missy_claims_l765_765729

theorem Missy_claims
  (Jan_capacity : ℕ) 
  (John_capacity : ℕ)
  (Missy_capacity : ℕ)
  (h1 : Jan_capacity = 20)
  (h2 : John_capacity = Jan_capacity + Nat.floor (0.3 * Jan_capacity))
  (h3 : Missy_capacity = John_capacity + 15) :
  Missy_capacity = 41 := by
  -- Normal steps to prove each hypothesis here
  sorry

end Missy_claims_l765_765729


namespace proof_problem_l765_765463

-- Define polar to Cartesian transformation
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

-- Define the curve C in polar coordinates
def curve_C_polar (ρ θ : ℝ) : Prop := ρ = 4 * (cos θ + sin θ)

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

-- Define the parametric line l
def line_l (t : ℝ) : ℝ × ℝ := (1 / 2 * t, 1 + (sqrt 3 / 2) * t)

-- Define the fixed point E
def point_E : ℝ × ℝ := (0, 1)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)^(1/2)

-- Define the main theorem
theorem proof_problem :
  (∀ ρ θ, curve_C_polar ρ θ ↔ curve_C_cartesian (polar_to_cartesian ρ θ).1 (polar_to_cartesian ρ θ).2) ∧
  (∀ t1 t2, line_l t1 ∈ curve_C_cartesian ∧ line_l t2 ∈ curve_C_cartesian →
    distance point_E (line_l t1) * distance point_E (line_l t2) = 3) :=
by
  sorry

end proof_problem_l765_765463


namespace number_of_two_digit_primes_with_units_digit_three_l765_765212

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765212


namespace increasing_impl_a_geq_neg_2_min_value_of_f_in_interval_l765_765841

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * x + 4

theorem increasing_impl_a_geq_neg_2 {a : ℝ} : 
  (∀ x₁ x₂ : ℝ, (1 ≤ x₁ → 1 ≤ x₂ → x₁ ≤ x₂ → f a x₁ ≤ f a x₂)) → a ≥ -2 :=
by
  intro h
  sorry

theorem min_value_of_f_in_interval {a : ℝ} :
  (let fmin :=
    if a > 4 then 8 - 2 * a
    else if -2 ≤ a ∧ a ≤ 4 then 4 - (a^2) / 4
    else 5 + a
  in fmin) = f a (-2)
  ∨ fmin = f a 1
  ∨ fmin = f a (-a / 2) :=
by
  sorry

end increasing_impl_a_geq_neg_2_min_value_of_f_in_interval_l765_765841


namespace area_FND_eq_18_l765_765914

variables (P Q R N D F C : Type) [triangle : Triangle P Q R]
  (angleR : Angle R > 90)
  (PN_eq_NR : Distance P N = Distance N R)
  (ND_perp_QR : Perpendicular N D Q R)
  (FC_perp_QR : Perpendicular F C Q R)
  (area_PQR_eq_36 : Area P Q R = 36)

theorem area_FND_eq_18 : Area F N D = 18 :=
sorry

end area_FND_eq_18_l765_765914


namespace angle_A_is_60_l765_765886

theorem angle_A_is_60 
  (O I : Type) (triangle_ABC : Triangle ABC) 
  (circumcenter : IsCircumcenter O triangle_ABC)
  (incenter : IsIncenter I triangle_ABC) 
  (angle_BOC_eq_angle_BIC : (angle B O C) = (angle B I C)) :
  angle A = 60 :=
sorry

end angle_A_is_60_l765_765886


namespace ellipse_solution_l765_765465

noncomputable def ellipse_equation_proof : Prop :=
  let e := sqrt 5 / 3 in -- Eccentricity
  let b := 2 in         -- Minor axis half-length
  let circle_radius := 2 in -- Circle radius
  let a := sqrt 9 in    -- Major axis half-length based on solving the given conditions
  let h := 3 in -- Half-distance between foci
  let C (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1 in
  let A1 := (-a, 0) in
  let A2 := (a, 0) in
  ∃ (m : ℝ), m = 39 / 5 ∧ ∀ (P : ℝ × ℝ), P ≠ A1 ∧ P ≠ A2 ∧ C P.fst P.snd → 
  ∃ (Q : ℝ × ℝ), Q.fst = m ∧ 
  let k_PA2 := (P.snd - A2.snd) / (P.fst - A2.fst) in
  let k_QA2 := (Q.snd - A2.snd) / (Q.fst - A2.fst) in
  k_PA2 * k_QA2 = -1

theorem ellipse_solution : ellipse_equation_proof := sorry

end ellipse_solution_l765_765465


namespace range_of_a_maximum_of_z_l765_765501

-- Problem 1
theorem range_of_a (a b : ℝ) (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) :
  -2 < a ∧ a < 1 :=
sorry

-- Problem 2
theorem maximum_of_z (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 9) :
  ∃ z, z = a * b^2 ∧ z ≤ 27 :=
sorry


end range_of_a_maximum_of_z_l765_765501


namespace trajectory_not_parabola_l765_765047

noncomputable def circle (O : Point) (r : ℝ) := { P : Point | dist O P = r }
def perpendicular_bisector (A B : Point) : Line := 
  { P | dist A P = dist B P }
def line_through (O B : Point) : Line := 
  -- Assume OB is the line through points O and B
  { P | ∃ k : ℝ, P = O + k * (B - O) }

theorem trajectory_not_parabola 
  (O B : Point) (r : ℝ) (A : Point)
  (hB : B ∈ circle O r) :
  ¬ (∃ P : Point, 
      P ∈ (perpendicular_bisector A B) ∧ 
      P ∈ (line_through O B) ∧ 
      is_parabola { P' | P' = P }) :=
sorry

end trajectory_not_parabola_l765_765047


namespace monotonic_shift_l765_765439

theorem monotonic_shift {α : Type*} [preorder α] {f : α → α} {a b : α} (h : ∀ ⦃x y⦄, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b → (x ≤ y → f x ≤ f y)) :
  ∀ ⦃x y⦄, (a-3 : α) ≤ x ∧ x ≤ (b-3 : α) ∧ (a-3 : α) ≤ y ∧ y ≤ (b-3 : α) → (x ≤ y → f (x+3) ≤ f (y+3)) :=
by {
  sorry,
}

end monotonic_shift_l765_765439


namespace maria_ann_age_problem_l765_765504

theorem maria_ann_age_problem
  (M A : ℕ)
  (h1 : M = 7)
  (h2 : M = A - 3) :
  ∃ Y : ℕ, 7 - Y = 1 / 2 * (10 - Y) := by
  sorry

end maria_ann_age_problem_l765_765504


namespace distinct_zeros_arithmetic_geometric_sequence_l765_765974

theorem distinct_zeros_arithmetic_geometric_sequence 
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : a + b = p)
  (h3 : ab = q)
  (h4 : p > 0)
  (h5 : q > 0)
  (h6 : (a = 4 ∧ b = 1) ∨ (a = 1 ∧ b = 4))
  : p + q = 9 := 
sorry

end distinct_zeros_arithmetic_geometric_sequence_l765_765974


namespace inequality_system_solution_l765_765551

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l765_765551


namespace count_two_digit_prime_numbers_ending_in_3_l765_765248

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765248


namespace count_two_digit_prime_numbers_ending_in_3_l765_765250

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765250


namespace two_digit_primes_units_digit_3_count_l765_765363

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765363


namespace unique_divisors_form_l765_765776

theorem unique_divisors_form (n : ℕ) (h₁ : n > 1)
    (h₂ : ∀ d : ℕ, d ∣ n ∧ d > 1 → ∃ a r : ℕ, a > 1 ∧ r > 1 ∧ d = a^r + 1) :
    n = 10 := by
  sorry

end unique_divisors_form_l765_765776


namespace percentage_sold_last_l765_765703

/-- A shopkeeper has 280 kg of apples.
    He sells 40% of these at 20% profit and the remaining at 20% profit.
    His total profit percentage is 20%.
    What percentage of apples does he sell last? -/
theorem percentage_sold_last :
  ∀ (total_apples : ℕ)
    (percent_sold_first : ℚ)
    (profit_first : ℚ)
    (profit_remaining : ℚ)
    (total_profit_percent : ℚ),
    total_apples = 280 →
    percent_sold_first = 40 →
    profit_first = 20 →
    profit_remaining = 20 →
    total_profit_percent = 20 →
    (100 - percent_sold_first) = 60 :=
by
  intros total_apples percent_sold_first profit_first profit_remaining total_profit_percent
  intros total_apples_eq percent_sold_first_eq profit_first_eq profit_remaining_eq total_profit_percent_eq
  rw [percent_sold_first_eq]
  exact rfl

end percentage_sold_last_l765_765703


namespace part1_answer_part2_answer_l765_765051

def setA : Set ℝ := {x | -2 < x ∧ x < 7}
def setB (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3*a - 2}

-- Part 1
theorem part1_answer : (setA ∪ setB 4 = Ioc (-2 : ℝ) 10) ∧ (compl setA ∩ setB 4 = Icc 7 10) := sorry

-- Part 2
theorem part2_answer (a : ℝ) : (setA ∪ setB a = setA) ↔ (a < 3) := sorry

end part1_answer_part2_answer_l765_765051


namespace probability_exactly_one_correct_l765_765513

def P_A := 0.7
def P_B := 0.8

def P_A_correct_B_incorrect := P_A * (1 - P_B)
def P_A_incorrect_B_correct := (1 - P_A) * P_B

theorem probability_exactly_one_correct :
  P_A_correct_B_incorrect + P_A_incorrect_B_correct = 0.38 :=
by
  sorry

end probability_exactly_one_correct_l765_765513


namespace natural_number_pairs_int_l765_765773

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l765_765773


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765344

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765344


namespace count_two_digit_primes_with_units_digit_3_l765_765325

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765325


namespace remainder_of_polynomial_l765_765938

theorem remainder_of_polynomial (p : ℚ[X]) (h1 : p.eval 2 = 4) (h2 : p.eval 5 = 10) :
  ∃ q : ℚ[X], p = q * (X - 2) * (X - 5) + 2 * X :=
by
  sorry

end remainder_of_polynomial_l765_765938


namespace count_two_digit_prime_numbers_ending_in_3_l765_765247

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765247


namespace number_of_two_digit_primes_with_units_digit_three_l765_765211

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765211


namespace count_two_digit_primes_ending_in_3_l765_765365

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765365


namespace parabola_properties_l765_765005

def parabola_vertex_origin : Prop := 
  ∃ (p : ℝ), p = 3 ∧ (∀ (x y : ℝ), x^2 = -4 * p * y ↔ x^2 = -12 * y) 

def parabola_focus : Prop := 
  ∃ (p : ℝ), p = 3 ∧ y = 3

theorem parabola_properties :
  parabola_vertex_origin ∧ parabola_focus :=
by
  sorry

end parabola_properties_l765_765005


namespace count_two_digit_primes_with_units_digit_3_l765_765157

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765157


namespace necessary_not_sufficient_condition_l765_765499

variable (a : ℕ → ℝ)
variable (a0 : ℝ) (q : ℝ)
variable (h_pos : a0 > 0)
variable (h_geom : ∀ n : ℕ, a(n+1) = a n * q)

noncomputable def geo_sequence : ℕ → ℝ
| 0       := a0
| (n + 1) := geo_sequence n * q

theorem necessary_not_sufficient_condition (q : ℝ) (a0 : ℝ) (h_pos : 0 < a0)
  (h_geom : ∀ n : ℕ, geo_sequence a0 q n = (a0 * q ^ n)) :
  (∀ n : ℕ+, geo_sequence a0 q (2 * n - 1) + geo_sequence a0 q (2 * n) < 0) ↔ q < 0 :=
sorry

end necessary_not_sufficient_condition_l765_765499


namespace min_value_expression_l765_765491

theorem min_value_expression (x y : ℝ) : (∃ z : ℝ, (forall x y : ℝ, z ≤ 5*x^2 + 4*y^2 - 8*x*y + 2*x + 4) ∧ z = 3) :=
sorry

end min_value_expression_l765_765491


namespace find_pairs_l765_765751

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l765_765751


namespace num_integers_digit_sum_18_400_600_l765_765123

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem num_integers_digit_sum_18_400_600 : 
  {n : ℕ // 400 ≤ n ∧ n < 600 ∧ digit_sum n = 18}.card = 21 := sorry

end num_integers_digit_sum_18_400_600_l765_765123


namespace sin_seventeen_pi_over_four_l765_765015

theorem sin_seventeen_pi_over_four : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := sorry

end sin_seventeen_pi_over_four_l765_765015


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765277

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765277


namespace find_vanessa_age_l765_765921

/-- Define the initial conditions and goal -/
theorem find_vanessa_age (V : ℕ) (Kevin_age current_time future_time : ℕ) :
  Kevin_age = 16 ∧ future_time = current_time + 5 ∧
  (Kevin_age + future_time - current_time) = 3 * (V + future_time - current_time) →
  V = 2 := 
by
  sorry

end find_vanessa_age_l765_765921


namespace polynomial_divisible_by_kth_power_l765_765482

variables (n : ℕ) (f : ℕ → ℤ) (m : ℕ)
variables (a : ℕ → ℤ)
variables (k : ℕ)

def polynomial_with_integer_coefficients : Prop :=
  ∀ x, ∃ (a : ℕ → ℤ), f x = a m * x^m + a (m-1) * x^(m-1) + ... + a 1 * x + a 0

def has_valid_coefficients : Prop := 
  (∀ i, 2 ≤ i → i ≤ m → ∃ p : ℕ, Prime p ∧ p ∣ n ∧ p ∣ a i) ∧
  (Nat.Coprime (a 1) n)

theorem polynomial_divisible_by_kth_power (h₀ : 0 < n)
  (h₁ : polynomial_with_integer_coefficients f m a)
  (h₂ : has_valid_coefficients n a m) :
  ∃ c : ℕ, 0 < c ∧ n^k ∣ f c :=
sorry

end polynomial_divisible_by_kth_power_l765_765482


namespace linear_relationship_of_C_l765_765652

-- Definitions based on conditions
def A_related (height : ℝ) (eyesight : ℝ) : Prop := ¬ height = eyesight
def B_related (angle_size : ℝ) (arc_length : ℝ) : Prop := ¬ angle_size = arc_length
def C_related (income : ℝ) (consumption : ℝ) : Prop := income = consumption
def D_related (age : ℝ) (height : ℝ) : Prop := ¬ age = height

-- Theorem to prove
theorem linear_relationship_of_C :
  (∀ (height eyesight : ℝ), A_related height eyesight) →
  (∀ (angle_size arc_length : ℝ), B_related angle_size arc_length) →
  (∀ (income consumption : ℝ), C_related income consumption) →
  (∀ (age height : ℝ), D_related age height) →
  ∃ (income consumption : ℝ), income = consumption :=
by
  intros h1 h2 h3 h4
  use [1, 1]
  exact h3 1 1

end linear_relationship_of_C_l765_765652


namespace two_digit_primes_units_digit_3_count_l765_765357

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765357


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765286

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765286


namespace percentage_increase_from_boys_to_total_l765_765738

def DamesSchoolBoys : ℕ := 2000
def DamesSchoolGirls : ℕ := 5000
def TotalAttendance : ℕ := DamesSchoolBoys + DamesSchoolGirls
def PercentageIncrease (initial final : ℕ) : ℚ := ((final - initial) / initial) * 100

theorem percentage_increase_from_boys_to_total :
  PercentageIncrease DamesSchoolBoys TotalAttendance = 250 :=
by
  sorry

end percentage_increase_from_boys_to_total_l765_765738


namespace binomial_equation_l765_765884

noncomputable def binom : ℕ → ℕ → ℕ := λ n k, if k ≤ n then Nat.choose n k else 0

theorem binomial_equation (n : ℕ) (h : 0 < n) :
  binom n 1 + binom (2 * n) 2 + binom (2 * n) 1 = 40 → n = 4 :=
by
  sorry

end binomial_equation_l765_765884


namespace part_a_part_b_l765_765924

def R : ℕ := ∑ n in (Finset.range 2001), (n + 3) * (n + 9)
def S : ℕ := ∑ n in (Finset.range 2001), (n + 1) * (n + 11)

-- Question (a)
theorem part_a : R > S := sorry

-- Question (b)
theorem part_b : R - S = 32016 := sorry

end part_a_part_b_l765_765924


namespace domain_of_function_l765_765912

def function_undefined_at (x : ℝ) : Prop :=
  ∃ y : ℝ, y = (x - 3) / (x - 2)

theorem domain_of_function (x : ℝ) : ¬(x = 2) ↔ function_undefined_at x :=
sorry

end domain_of_function_l765_765912


namespace two_digit_primes_units_digit_3_count_l765_765346

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765346


namespace count_two_digit_primes_with_units_digit_3_l765_765161

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765161


namespace order_a_b_c_l765_765802

noncomputable def a : ℝ := Real.log 3 / Real.log 4
noncomputable def b : ℝ := Real.log 3
noncomputable def c : ℝ := Real.sqrt 10

theorem order_a_b_c : a < b ∧ b < c :=
by
  have ha : 0 < a ∧ a < 1 := by
    have : (log 4)⁻¹ > 0 := by sorry
    have : log 3 / log 4 = log 3 * (log 4)⁻¹ := by sorry
    sorry
  have hb : 1 < b ∧ b < 2 := by sorry
  have hc : c > 3 := by sorry
  sorry

end order_a_b_c_l765_765802


namespace find_k_when_root_is_zero_l765_765852

-- Define the quadratic equation and what it implies
theorem find_k_when_root_is_zero (k : ℝ) (h : (k-1) * 0^2 + 6 * 0 + k^2 - k = 0) :
  k = 0 :=
by
  -- The proof steps would go here, but we're skipping it as instructed
  sorry

end find_k_when_root_is_zero_l765_765852


namespace complement_union_l765_765088

open Set

-- Define U to be the set of all real numbers
def U := @univ ℝ

-- Define the domain A for the function y = sqrt(x-2) + sqrt(x+1)
def A := {x : ℝ | x ≥ 2}

-- Define the domain B for the function y = sqrt(2x+4) / (x-3)
def B := {x : ℝ | x ≥ -2 ∧ x ≠ 3}

-- Theorem about the union of the complements
theorem complement_union : (U \ A ∪ U \ B) = {x : ℝ | x < 2 ∨ x = 3} := 
by
  sorry

end complement_union_l765_765088


namespace range_of_alpha_l765_765876

noncomputable def is_zero (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

theorem range_of_alpha :
  (∀ f : ℝ → ℝ, (f = λ x, cos (2 * x) + sin (2 * x + π / 6)) →
    (∀ α : ℝ, (∀ x : ℝ, 0 < x ∧ x < α →
      (is_zero f x → (0 < real.roots f (Ioo 0 α)).card = 2))) → 
    (5 * π / 6 < α ∧ α ≤ 4 * π / 3))) :=
begin
  sorry
end

end range_of_alpha_l765_765876


namespace count_two_digit_prime_numbers_ending_in_3_l765_765255

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765255


namespace distinctPermutations_test_l765_765098

noncomputable def distinctPermutationsCount (s : Multiset ℕ) : ℕ :=
  (s.card.factorial) / (s.count 3.factorial * s.count 8.factorial)

theorem distinctPermutations_test : distinctPermutationsCount {3, 3, 3, 8, 8} = 10 := by
  sorry

end distinctPermutations_test_l765_765098


namespace arithmetic_sequence_a2_l765_765827

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a1_a3 : a 1 + a 3 = 2) : a 2 = 1 :=
sorry

end arithmetic_sequence_a2_l765_765827


namespace probability_four_integer_coords_l765_765487

theorem probability_four_integer_coords :
  let R := { d : ℝ // (d = Math.sqrt 2) }
  let valid_diagonal (w : ℝ × ℝ) := 0 ≤ w.1 ∧ w.1 ≤ 100 ∧ 0 ≤ w.2 ∧ w.2 ≤ 100
  let contains_four_integer_coords (w : ℝ × ℝ) :=
    ∃ s : ℝ, s = 1 ∧
      (w.1 - 0.5).ceil + (w.2 - 0.5).ceil + (w.1 + 0.5).floor + (w.2 + 0.5).floor = 4
  let Q_w := { w : ℝ × ℝ // valid_diagonal w ∧ contains_four_integer_coords w }
  Q_w.card / ((101 : ℕ) * (101 : ℕ) : ℝ) = 10000 / 10201 :=
sorry

end probability_four_integer_coords_l765_765487


namespace intersection_length_l765_765916

theorem intersection_length
  (line_eq : ∀ (x y : ℝ), x - (√3) * y - 1 = 0)
  (circle_eq : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 1) :
  let A := (1 + √3, 0)
  let B := (1 - √3, 0)
  dist A B = 2 := by
  sorry

end intersection_length_l765_765916


namespace angle_equality_l765_765620

variables {A B C D K T M : Point}
variables (h1 : IsoscelesTriangle A B C) -- isosceles triangle ABC with base BC
variables (h2 : PointOnLineSegment D A C) -- D is on edge AC
variables (h3 : PointOnMinorArc K C D (CircumcircleTriangle B C D)) -- K is on minor arc CD of circumcircle of BCD
variables (h4 : ParallelThroughPoint A BC TA)
variables (h5 : RayIntersects CK TA T)
variables (h6 : Midpoint M D T) -- M is midpoint of DT

theorem angle_equality (h : ∠ AK T = ∠ CA M) : true :=
sorry

end angle_equality_l765_765620


namespace solve_c_l765_765779

theorem solve_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) :
  a + sqrt (b + c) > b + sqrt (a + c) ↔ c = 1 / 4 :=
by sorry

end solve_c_l765_765779


namespace count_prime_units_digit_3_eq_6_l765_765302

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765302


namespace num_two_digit_primes_with_units_digit_three_l765_765147

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765147


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765396

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765396


namespace cymbal_strike_interval_l765_765653

theorem cymbal_strike_interval (c : ℕ) (h1 : ∃ c : ℕ, lcm c 2 = 14) : c = 14 := 
sorry

end cymbal_strike_interval_l765_765653


namespace two_digit_primes_units_digit_3_count_l765_765362

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765362


namespace max_odd_sums_of_consecutive_triplets_l765_765715

theorem max_odd_sums_of_consecutive_triplets (seq : List ℕ) (h : seq.length = 998) (h_range : ∀ n ∈ seq, 1000 ≤ n ∧ n ≤ 1997) : 
  ∃ arrangement : List ℕ, arrangement.perm seq ∧ (maxOddSumsOfConsecutiveTriplets arrangement = 499) := 
sorry

-- Definitions used in the theorem
def maxOddSumsOfConsecutiveTriplets (seq : List ℕ) : ℕ :=
  -- calculate the maximum number of odd sums of every three consecutive numbers in seq
  sorry

end max_odd_sums_of_consecutive_triplets_l765_765715


namespace cube_path_proof_l765_765941

structure Point (α : Type*) [Add α] [Mul α] := (x y z : α)

def Point.dist {α : Type*} [Mul α] [Add α] [HasSub α] [LinearOrder α]
    (p1 p2 : Point α) : α := 
    (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

axiom exists_path_length_two
  (side : ℝ) (A : Point ℝ) (S : set (Point ℝ)) 
  (A_on_S : A ∈ S) (cube_surface : ∀ P, P ∈ S ↔ ∥P∥ = side) :
  ∃ A : Point ℝ, ∀ P : Point ℝ, P ∈ S → Point.dist A P <= 2

axiom exists_point_exactly_two
  (side : ℝ) (A : Point ℝ) (S : set (Point ℝ)) 
  (A_on_S : A ∈ S) (cube_surface : ∀ P, P ∈ S ↔ ∥P∥ = side) :
  ∃ B : Point ℝ, B ∈ S ∧ Point.dist A B = 2

theorem cube_path_proof :
  ∃ A : Point ℝ, ( ∀ P : Point ℝ, P ∈ S → Point.dist A P <= 2 ) ∧
                  ( ∃ B : Point ℝ, B ∈ S ∧ Point.dist A B = 2 ):=
by 
  apply And.intro sorry sorry

end cube_path_proof_l765_765941


namespace smaller_square_area_percentage_l765_765706

noncomputable def area_percentage_of_smaller_square :=
  let side_length_large_square : ℝ := 4
  let area_large_square := side_length_large_square ^ 2
  let side_length_smaller_square := side_length_large_square / 5
  let area_smaller_square := side_length_smaller_square ^ 2
  (area_smaller_square / area_large_square) * 100
theorem smaller_square_area_percentage :
  area_percentage_of_smaller_square = 4 := 
sorry

end smaller_square_area_percentage_l765_765706


namespace domain_of_y_eq_one_div_log_two_x_plus_one_l765_765598

open Set

noncomputable def domain_of_function := {x : ℝ | x > -1/2 ∧ x ≠ 0}

theorem domain_of_y_eq_one_div_log_two_x_plus_one :
  {x : ℝ | (2 * x + 1 > 0) ∧ (Real.log (2 * x + 1) ≠ 0)} = domain_of_function :=
begin
  sorry
end

end domain_of_y_eq_one_div_log_two_x_plus_one_l765_765598


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765386

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765386


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765287

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765287


namespace number_of_moles_of_KNO3_l765_765787

def chemical_reaction_rule (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl ∧ NaCl = NaNO3 

theorem number_of_moles_of_KNO3 :
  ∀ (NaCl KNO3 NaNO3 KCl : ℕ),
    chemical_reaction_rule NaCl KNO3 NaNO3 KCl →
    NaCl = 3 → NaNO3 = 3 → KNO3 = 3 := by
  intros NaCl KNO3 NaNO3 KCl h_reaction h_NaCl h_NaNO3
  have h1 := h_reaction.2
  rw [h_NaCl, h_NaNO3] at h1
  rw [←h1] at h_reaction
  sorry

end number_of_moles_of_KNO3_l765_765787


namespace four_coloring_exists_l765_765520

def M : set ℕ := {i | 1 ≤ i ∧ i ≤ 1987}

def is_arithmetic_progression (s : list ℕ) : Prop :=
  ∀ i j k, i < j < k → s.nth i = some x → s.nth j = some y → s.nth k = some z → 2 * y = x + z

noncomputable def exists_four_coloring (M : set ℕ) (f : ℕ → ℕ) : Prop :=
  ∀ s, s.to_list.length = 10 ∧ is_arithmetic_progression s.to_list → 
  ∃ i j, i < j ∧ s.nth i ≠ s.nth j

theorem four_coloring_exists :
  ∃ f : ℕ → ℕ, exists_four_coloring M f :=
sorry

end four_coloring_exists_l765_765520


namespace find_pairs_l765_765755

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l765_765755


namespace roberto_hike_time_l765_765526

def uphill_speed : ℝ := 2 -- speed in miles per hour
def downhill_speed : ℝ := 3 -- speed in miles per hour
def total_trail_length : ℝ := 5 -- length in miles
def uphill_percentage : ℝ := 0.6 -- 60% uphill

-- Define the downhill percentage based on remaining percentage
def downhill_percentage : ℝ := 1 - uphill_percentage

-- Calculate distances
def uphill_distance : ℝ := total_trail_length * uphill_percentage
def downhill_distance : ℝ := total_trail_length * downhill_percentage

-- Calculate time in hours
def uphill_time_hours : ℝ := uphill_distance / uphill_speed
def downhill_time_hours : ℝ := downhill_distance / downhill_speed

-- Convert time to minutes
def uphill_time_minutes : ℝ := uphill_time_hours * 60
def downhill_time_minutes : ℝ := downhill_time_hours * 60

-- Calculate total time
def total_time_minutes : ℝ := uphill_time_minutes + downhill_time_minutes

theorem roberto_hike_time : total_time_minutes = 130 := by
  sorry

end roberto_hike_time_l765_765526


namespace xy_half_l765_765419

theorem xy_half (x y : ℝ) (h : (x - 2)^2 + sqrt (y + 1) = 0) : x ^ y = 1 / 2 :=
by
  sorry

end xy_half_l765_765419


namespace total_ice_cream_volume_l765_765609

def cone_height : ℝ := 10
def cone_radius : ℝ := 1.5
def cylinder_height : ℝ := 2
def cylinder_radius : ℝ := 1.5
def hemisphere_radius : ℝ := 1.5

theorem total_ice_cream_volume : 
  (1 / 3 * π * cone_radius ^ 2 * cone_height) +
  (π * cylinder_radius ^ 2 * cylinder_height) +
  (2 / 3 * π * hemisphere_radius ^ 3) = 14.25 * π :=
by sorry

end total_ice_cream_volume_l765_765609


namespace inequality_system_solution_l765_765579

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l765_765579


namespace coefficient_x2_in_2x_minus_1_pow_5_l765_765911

theorem coefficient_x2_in_2x_minus_1_pow_5 :
  ∃ c : ℤ, (∀ x : ℝ, ((2 * x - 1)^5).expand == ∑' (k : ℕ) in finset.range 6, (binom 5 k) * (2 * x)^(5 - k) * (-1)^k * (if k = 2 then c else 1)) → c = -40 :=
sorry

end coefficient_x2_in_2x_minus_1_pow_5_l765_765911


namespace vertical_asymptotes_l765_765021

-- Define the function g(x)
def g (k : ℝ) (x : ℝ) : ℝ := (x^2 - 3 * x + k) / (x^2 - 2 * x - 8)

-- Define the condition for vertical asymptotes
theorem vertical_asymptotes (k : ℝ) : (∃ a b : ℝ, 
  (∀ x, g k x = (a / (x - 4)) + (b / (x + 2)) ) ∧ 
  (k ≠ -4 ∧ k ≠ -10)) :=
sorry

end vertical_asymptotes_l765_765021


namespace inequality_solution_l765_765544

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l765_765544


namespace two_digit_primes_with_units_digit_three_count_l765_765181

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765181


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765329

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765329


namespace circle_lines_total_l765_765967

def nineCircles (a b c d e f g h i: ℕ) : Prop :=
  {1, 2, 3, 4, 5, 6, 7, 8, 9} = {a, b, c, d, e, f, g, h, i} ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

noncomputable def lineTotal (a b c d e f g h i: ℕ) :=
  a + b + c + d = 20 ∧ e + f + g + h = 20 ∧ (a + e + b + i = 20)

theorem circle_lines_total {a b c d e f g h i: ℕ} (h1: nineCircles a b c d e f g h i) :
  lineTotal a b c d e f g h i :=
sorry

end circle_lines_total_l765_765967


namespace one_fourth_of_8_point_8_l765_765001

-- Definition of taking one fourth of a number
def oneFourth (x : ℝ) : ℝ := x / 4

-- Problem statement: One fourth of 8.8 is 11/5 when expressed as a simplified fraction
theorem one_fourth_of_8_point_8 : oneFourth 8.8 = 11 / 5 := by
  sorry

end one_fourth_of_8_point_8_l765_765001


namespace solution_set_of_fg_ineq_l765_765500

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

noncomputable def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g(x)

theorem solution_set_of_fg_ineq
  (f g : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_even : even_function g)
  (h_deriv_pos : ∀ x, x < 0 → f' x * g x + f x * g' x > 0)
  (h_g3_zero : g 3 = 0) :
  { x : ℝ | f x * g x < 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | 0 < x ∧ x < 3 } :=
sorry

end solution_set_of_fg_ineq_l765_765500


namespace digit_sum_400_to_600_equals_18_l765_765116

theorem digit_sum_400_to_600_equals_18 :
  card {n : ℕ | 400 ≤ n ∧ n ≤ 600 ∧ (n.digits 10).sum = 18} = 21 :=
sorry

end digit_sum_400_to_600_equals_18_l765_765116


namespace parabola_min_length_l765_765809

noncomputable def parabola_eq (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c
def point (x y : ℝ) := (x, y)

theorem parabola_min_length (a b c : ℝ)
  (h1: a > 0)
  (h2: parabola_eq a b c 0 = 4)
  (h3: parabola_eq a b c 2 = -2)
  (minimizes_length : a = 9 / 2):
  parabola_eq a b c = parabola_eq (9 / 2) (-12) 4 :=
by
  sorry

end parabola_min_length_l765_765809


namespace find_omega_l765_765438

theorem find_omega
  (f : ℝ → ℝ)
  (ω α β : ℝ)
  (h1 : ∀ x, f x = Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x))
  (h2 : f α = -2)
  (h3 : f β = 0)
  (h4 : ∀ k : ℤ, |α - β| = min (|α - β|) (Real.abs ((α - β) - k * (2 * Real.pi / ω)) = 3 * Real.pi / 4)) :
  ω = 2 / 3 := sorry

end find_omega_l765_765438


namespace find_sale_month_4_l765_765683

-- Define the given sales data
def sale_month_1: ℕ := 5124
def sale_month_2: ℕ := 5366
def sale_month_3: ℕ := 5808
def sale_month_5: ℕ := 6124
def sale_month_6: ℕ := 4579
def average_sale_per_month: ℕ := 5400

-- Define the goal: Sale in the fourth month
def sale_month_4: ℕ := 5399

-- Prove that the total sales conforms to the given average sale
theorem find_sale_month_4 :
  sale_month_1 + sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 + sale_month_6 = 6 * average_sale_per_month :=
by
  sorry

end find_sale_month_4_l765_765683


namespace table_capacity_l765_765678

theorem table_capacity :
  ∀ (n_invited no_show tables : ℕ), n_invited = 47 → no_show = 7 → tables = 8 → 
  (n_invited - no_show) / tables = 5 := by
  intros n_invited no_show tables h_invited h_no_show h_tables
  sorry

end table_capacity_l765_765678


namespace range_of_alpha_l765_765877

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + sin (2 * x + π / 6)

theorem range_of_alpha (α : ℝ) :
  (∃ γ ∈ Ioo (0 : ℝ) α, f γ = 0 ∧ ∃ δ ∈ Ioo (γ : ℝ) α, f δ = 0) →
  α ∈ Ioc (5 * π / 6) (4 * π / 3) :=
sorry

end range_of_alpha_l765_765877


namespace count_two_digit_primes_with_units_digit_3_l765_765197

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765197


namespace Gabriel_must_paint_1552_sq_ft_l765_765798

-- Conditions
def bedrooms := 4
def length := 14 -- in feet
def width := 12 -- in feet
def height := 9 -- in feet
def non_paintable_area_per_bedroom := 80 -- in square feet

-- Question and Answer Tuple (Equivalent Proof Problem)
theorem Gabriel_must_paint_1552_sq_ft :
  let area_of_one_bedroom := 2 * (length * height) + 2 * (width * height)
  let paintable_area_per_bedroom := area_of_one_bedroom - non_paintable_area_per_bedroom
  let total_paintable_area := paintable_area_per_bedroom * bedrooms
  total_paintable_area = 1552 := by
    -- Placeholder to indicate the proof should be here
    sorry

end Gabriel_must_paint_1552_sq_ft_l765_765798


namespace range_of_alpha_l765_765878

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + sin (2 * x + π / 6)

theorem range_of_alpha (α : ℝ) :
  (∃ γ ∈ Ioo (0 : ℝ) α, f γ = 0 ∧ ∃ δ ∈ Ioo (γ : ℝ) α, f δ = 0) →
  α ∈ Ioc (5 * π / 6) (4 * π / 3) :=
sorry

end range_of_alpha_l765_765878


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765268

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765268


namespace num_two_digit_primes_with_units_digit_three_l765_765135

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765135


namespace two_digit_primes_with_units_digit_three_count_l765_765177

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765177


namespace math_problem_l765_765821

variable (x₀ : ℝ) (x : ℝ)

noncomputable def p : Prop := ¬ ∃ x₀ : ℝ, 2^x₀ + 2^(-x₀) = 1

noncomputable def q : Prop := ∀ x : ℝ, log (x^2 + 2*x + 3) > 0

theorem math_problem : (¬ p ∧ q) :=
sorry

end math_problem_l765_765821


namespace set_union_correct_l765_765050

theorem set_union_correct:
  let A := {0, 1, 2, 3}
  let B := {1, 2, 4}
  A ∪ B = {0, 1, 2, 3, 4} :=
by
  sorry

end set_union_correct_l765_765050


namespace perpendicular_tangent_lines_l765_765071

def f (x : ℝ) : ℝ := x^3 + 1

noncomputable def tangent_line_eqs (x₀ : ℝ) (y₀ : ℝ) : Prop :=
  (3 * x₀ - y₀ - 1 = 0) ∨ (3 * x₀ - y₀ + 3 = 0)

theorem perpendicular_tangent_lines (x₀ : ℝ) (hx₀ : x₀ = 1 ∨ x₀ = -1) :
  tangent_line_eqs x₀ (f x₀) := by
  sorry

end perpendicular_tangent_lines_l765_765071


namespace sin_identity_l765_765826

variable (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = Real.sqrt 3 / 2)

theorem sin_identity : Real.sin (3 * Real.pi / 4 - α) = Real.sqrt 3 / 2 := by
  sorry

end sin_identity_l765_765826


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765282

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765282


namespace number_of_two_digit_primes_with_units_digit_three_l765_765208

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765208


namespace inequality_system_solution_l765_765575

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l765_765575


namespace count_two_digit_primes_with_units_digit_three_l765_765403

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765403


namespace part1_sol_set_a_eq_2_part2_integer_values_a_l765_765033

noncomputable def f (a : ℝ) (x : ℝ) := (a^2 - a) / x^2 - 3 / x

theorem part1_sol_set_a_eq_2 : 
  (∀ x : ℝ, f 2 x > -1 ↔ x ∈ (-∞, 0) ∪ (0,1) ∪ (2,∞)) := 
sorry

theorem part2_integer_values_a : 
  (∀ (a : ℤ), (∀ x ∈ (set.Ioc 0 1 : set ℝ), f (a : ℝ) x < 6 / x^3) ↔ a ∈ {-2, -1, 0, 1, 2, 3}) :=
sorry

end part1_sol_set_a_eq_2_part2_integer_values_a_l765_765033


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765330

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765330


namespace num_two_digit_primes_with_units_digit_three_l765_765144

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765144


namespace count_prime_units_digit_3_eq_6_l765_765308

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765308


namespace ratio_of_segments_l765_765926

-- Define all necessary points, lines, and properties of triangle ABC
variable {A B C G H A1 A2 X: Type} 
          [metric_space] 

axiom midpoint_A1 (BC: line_segment B C) : midpoint A1 BC
axiom centroid_G (triangle_ABC: triangle A B C) : centroid G triangle_ABC
axiom square_GBKL (GBKL: square G B K L) : is_on_left_rays GB KL
axiom square_GCMN (GCMN: square G C M N) : is_on_left_rays GC MN
axiom midpoint_A2 (F E: Type) [square_center GBKL F] [square_center GCMN E] : midpoint A2 (line_segment F E)
axiom circumcircle_intersects (circumcircle triangle_A1A2G: circle A1 A2 G) (line_segment: line_segment B C): 
  intersects circumcircle triangle_A1A2G B C
axiom base_altitude_H (altitude_AH: line_segment A H): base altitude_AH H

theorem ratio_of_segments (triangle_ABC: triangle A B C) (circumcircle triangle_A1A2G: circle A1 A2 G) (altitude_AH: line_segment A H) 
  (F E: Type) [square_center GBKL F] [square_center GCMN E] :
  ∃ A1 A2 X H,
    midpoint A1 (line_segment B C) →
    centroid G triangle_ABC →
    is_on_left_rays GBKL →
    is_on_left_rays GCMN →
    midpoint A2 (line_segment F E) →
    intersects circumcircle triangle_A1A2G (line_segment B C) →
    base altitude_AH H →
    ( ∀ F E, square_center GBKL F → square_center GCMN E → 
      \frac{ A1X }{ XH } = \frac{ 1 }{ 2 } )
sorry

end ratio_of_segments_l765_765926


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765397

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765397


namespace min_value_of_a_l765_765055

theorem min_value_of_a (a : ℝ) : 
  (∀ x > 1, x + a / (x - 1) ≥ 5) → a ≥ 4 :=
sorry

end min_value_of_a_l765_765055


namespace repeating_decimal_number_of_numerators_l765_765497

theorem repeating_decimal_number_of_numerators : 
  let N_vals := {N : ℕ | N ≤ 999 ∧ (∀ a b c, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ N = 100 * a + 10 * b + c)} in
  let num_distinct_numerators := {n ∈ N_vals | ¬ (3 ∣ n) ∧ ¬ (37 ∣ n) ∪ {n ∈ N_vals | ∃ k, n = k * 81}} in
  num_distinct_numerators.card = 660 :=
by
  sorry

end repeating_decimal_number_of_numerators_l765_765497


namespace auntie_em_parking_probability_l765_765690

theorem auntie_em_parking_probability :
  let total_spaces := 18
  let cars := 14
  let possible_parks := Nat.choose total_spaces cars
  let ways_to_block = Nat.choose (total_spaces - cars + 4) 4
  let probability_no_park := (ways_to_block : ℚ) / (possible_parks : ℚ)
  let probability_park := 1 - probability_no_park
  in probability_park = 113 / 204 := by
sorry

end auntie_em_parking_probability_l765_765690


namespace find_pairs_l765_765770

theorem find_pairs (a b : ℕ) :
  (∃ (a b : ℕ), (b^2 - a ≠ 0) ∧ (a^2 - b ≠ 0) ∧ (a^2 + b) / (b^2 - a) ∈ ℤ ∧ (b^2 + a) / (a^2 - b) ∈ ℤ) → 
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end find_pairs_l765_765770


namespace police_female_officers_l765_765861

theorem police_female_officers (F M : ℕ) (H1 : 0.27 * F ∈ ℕ) (H2 : is_rat 0.27) (H3 : 275 = M + (0.27 * F)) (H4 : ((4 * 0.27 * F) / 3 : ℝ) = M) : 
F = 437 :=
by
  sorry

end police_female_officers_l765_765861


namespace even_count_in_first_10_rows_l765_765618

def is_even (n : ℕ) : Prop := n % 2 = 0

def pascals_triangle (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1 else pascals_triangle (n - 1) (k - 1) + pascals_triangle (n - 1) k

def count_even_in_pascals_triangle (row : ℕ) : ℕ :=
  (List.range (row + 1)).countp (λ k => is_even (pascals_triangle row k))

def total_even_in_pascals_triangle (rows : ℕ) : ℕ :=
  (List.range (rows + 1)).sum (λ row => count_even_in_pascals_triangle row)

theorem even_count_in_first_10_rows : total_even_in_pascals_triangle 9 = 22 :=
by
  sorry

end even_count_in_first_10_rows_l765_765618


namespace sin_x_tan_double_x_plus_pi_over_4_l765_765825

open Real

theorem sin_x (x : ℝ) (hcos : cos x = - (sqrt 2 / 10)) (hinterval : x ∈ Ioo (π / 2) π) : 
  sin x = 7 * sqrt 2 / 10 := 
sorry

theorem tan_double_x_plus_pi_over_4 (x : ℝ) (hcos : cos x = - (sqrt 2 / 10)) (hinterval : x ∈ Ioo (π / 2) π) : 
  tan (2 * x + π / 4) = 31 / 17 := 
sorry

end sin_x_tan_double_x_plus_pi_over_4_l765_765825


namespace angle_identity_in_acute_triangle_l765_765455

-- Definitions of the geometric entities
structure Triangle :=
  (A B C M D E : Point)
  (acute : is_acute A B C)
  (midpoint_M : midpoint M B C)
  (circle_O : Circle)
  (circle_O_tangent : tangent circle_O C)
  (passes_through_A_C : passes circle_O A C)
  (AM_intersect_D : intersects circle_O A M D ∧ D ≠ A)
  (BD_intersect_E : intersects circle_O B D E ∧ E ≠ D)

-- Proposition to prove that ∠EAC = ∠BAC
theorem angle_identity_in_acute_triangle (T : Triangle) : 
  angle T.E T.A T.C = angle T.B T.A T.C :=
by
  sorry

end angle_identity_in_acute_triangle_l765_765455


namespace inequality_system_solution_l765_765549

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l765_765549


namespace solve_inequality_system_l765_765584

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l765_765584


namespace total_pairs_after_receiving_l765_765628

-- Definitions of the conditions
def pairs_given_away : ℕ := 14
def pairs_left : ℕ := 19

-- Theorem statement to be proved
theorem total_pairs_after_receiving (pairs_given_away = 14) (pairs_left = 19) : 
  pairs_given_away + pairs_left = 33 := 
by 
  sorry

end total_pairs_after_receiving_l765_765628


namespace painter_problem_l765_765902

def color : Type := {x : ℕ // x = 0 ∨ x = 1} -- 0 for white, 1 for green

noncomputable def grid_5x5 : matrix (fin 5) (fin 5) color := 
  ![![color.mk 1, color.mk 1, color.mk 1, color.mk 1, color.mk 0],
    ![color.mk 1, color.mk 0, color.mk 0, color.mk 0, color.mk 1],
    ![color.mk 1, color.mk 0, color.mk 0, color.mk 0, color.mk 1],
    ![color.mk 1, color.mk 0, color.mk 0, color.mk 0, color.mk 1],
    ![color.mk 0, color.mk 1, color.mk 1, color.mk 1, color.mk 1]]

def count_green (m : matrix (fin 3) (fin 3) color) : ℕ :=
  finset.sum finset.univ (λ i, finset.sum finset.univ (λ j, m i j.val))

def count_white (m : matrix (fin 3) (fin 3) color) : ℕ :=
  9 - count_green m

def count_green4x4 (m : matrix (fin 4) (fin 4) color) : ℕ :=
  finset.sum finset.univ (λ i, finset.sum finset.univ (λ j, m i j.val))

def count_white4x4 (m : matrix (fin 4) (fin 4) color) : ℕ :=
  16 - count_green4x4 m

theorem painter_problem :
  (∀ (i j : fin 3), count_white (submatrix grid_5x5 (fin.add i) (fin.add j)) > count_green (submatrix grid_5x5 (fin.add i) (fin.add j))) ∧
  (∀ (i j : fin 2), count_green4x4 (submatrix grid_5x5 (fin.add i) (fin.add j)) > count_white4x4 (submatrix grid_5x5 (fin.add i) (fin.add j)))  :=
sorry

end painter_problem_l765_765902


namespace range_of_a_l765_765931

variables {R : Type*} [AddGroup R] [Module ℝ R] {f : ℝ → ℝ}
variable (a : ℝ)

-- Define f as a periodic function with period 3
def is_periodic (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∀ n : ℤ, f (x + 3 * n) = f x

-- Define f as an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- The given conditions
variables (h_periodic : is_periodic f)
variables (h_odd : is_odd f)
variables (h_greater_than_one : f 1 > 1)
variables (h_f2_eq_a : f 2 = a)

-- The theorem to prove
theorem range_of_a : a < -1 :=
  sorry

end range_of_a_l765_765931


namespace non_empty_combinations_of_7_beads_even_n_combinations_are_odd_permutations_even_and_odd_l765_765949

-- Problem 1: Prove that the number of non-empty combinations of 7 beads is 127
theorem non_empty_combinations_of_7_beads : 2^7 - 1 = 127 :=
by
  calc
    2^7 - 1 = 128 - 1 : rfl
         ... = 127     : rfl

-- Problem 2: Prove that for any even n, 2^n - 1 is odd
theorem even_n_combinations_are_odd (n : Nat) (h_even : n % 2 = 0) : (2^n - 1) % 2 = 1 :=
by
  have h : 2^n % 2 = 0 := by sorry  -- Prove that 2^n is even for any even n
  calc
    (2^n - 1) % 2 = (2^n % 2 - 1 % 2) % 2 : by sorry  -- Apply modulo properties
             ... = (0 - 1) % 2             : by rw [h, Nat.mod_eq_of_lt (by decide: 1 < 2)]
             ... = -1 % 2                  : rfl
             ... = 1                       : by rw [Nat.mod_eq_of_lt (by decide: 1 < 2)]

-- Problem 3: Prove that the number of permutations considering order of beads can be either even or odd
theorem permutations_even_and_odd (n : Nat) : ∃ k : Nat, (∑ i in (range (n + 1)).filter (λ x, x > 0), P n i) % 2 = k % 2 :=
by
  -- Using the formula ∑ P(n, k) and the fact that P(n, k) includes both even and odd numbers
  sorry


end non_empty_combinations_of_7_beads_even_n_combinations_are_odd_permutations_even_and_odd_l765_765949


namespace alley_width_eq_height_l765_765450

theorem alley_width_eq_height (a k h : ℝ) 
  (angle1 : Real.angle) (angle2 : Real.angle)
  (h_angle45 : angle1 = Real.angle.pi / 4)
  (h_angle75 : angle2 = Real.angle.pi * (5 / 12)) :
  let w := h in
  angle1 + angle2 = Real.angle.pi / 3 →
  let Q := h / Real.sin angle1 in
  let R := h / Real.sin angle2 in w = h :=
by
  sorry

end alley_width_eq_height_l765_765450


namespace polynomial_not_9_l765_765694

theorem polynomial_not_9 (P : ℤ[X]) (a b c d e : ℤ)
    (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
    (h_values : P.eval a = 5 ∧ P.eval b = 5 ∧ P.eval c = 5 ∧ P.eval d = 5 ∧ P.eval e = 5) :
    ∀ x : ℤ, P.eval x ≠ 9 := 
begin
  sorry
end

end polynomial_not_9_l765_765694


namespace technician_completed_70_percent_of_round_trip_l765_765709

-- Define the one-way distance to the service center
def one_way_distance (D : ℝ) : ℝ := D

-- Define the round-trip distance
def round_trip_distance (D : ℝ) : ℝ := 2 * one_way_distance D

-- Define the distance traveled to the center and 40 percent back
def distance_traveled (D : ℝ) : ℝ := one_way_distance D + 0.4 * one_way_distance D

-- Define the percentage completed of the round-trip
def percent_completed (D : ℝ) : ℝ := (distance_traveled D / round_trip_distance D) * 100

-- The theorem to be proved
theorem technician_completed_70_percent_of_round_trip (D : ℝ) :
  percent_completed D = 70 := 
by
  sorry

end technician_completed_70_percent_of_round_trip_l765_765709


namespace find_real_values_l765_765777

theorem find_real_values (x : ℝ) : 
  (1 / (x^2 + 2) > 5 / x + 21 / 10) ↔ x ∈ set.Ioo (-2 : ℝ) 0 :=
sorry

end find_real_values_l765_765777


namespace third_factorial_is_7_l765_765607

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Problem conditions
def b : ℕ := 9
def factorial_b_minus_2 : ℕ := factorial (b - 2)
def factorial_b_plus_1 : ℕ := factorial (b + 1)
def GCD_value : ℕ := Nat.gcd (Nat.gcd factorial_b_minus_2 factorial_b_plus_1) (factorial 7)

-- Theorem statement
theorem third_factorial_is_7 :
  Nat.gcd (Nat.gcd (factorial (b - 2)) (factorial (b + 1))) (factorial 7) = 5040 →
  ∃ k : ℕ, factorial k = 5040 ∧ k = 7 :=
by
  sorry

end third_factorial_is_7_l765_765607


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765392

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765392


namespace distance_between_stripes_l765_765708

/-
Problem statement:
Given:
1. The street has parallel curbs 30 feet apart.
2. The length of the curb between the stripes is 10 feet.
3. Each stripe is 60 feet long.

Prove:
The distance between the stripes is 5 feet.
-/

-- Definitions:
def distance_between_curbs : ℝ := 30
def length_between_stripes_on_curb : ℝ := 10
def length_of_each_stripe : ℝ := 60

-- Theorem statement:
theorem distance_between_stripes :
  ∃ d : ℝ, (length_between_stripes_on_curb * distance_between_curbs = length_of_each_stripe * d) ∧ d = 5 :=
by
  sorry

end distance_between_stripes_l765_765708


namespace real_roots_condition_l765_765489

-- Definitions based on conditions
def polynomial (x : ℝ) : ℝ := x^4 - 6 * x - 1
def is_root (a : ℝ) : Prop := polynomial a = 0

-- The statement we want to prove
theorem real_roots_condition (a b : ℝ) (ha: is_root a) (hb: is_root b) : 
  (a * b + 2 * a + 2 * b = 1.5 + Real.sqrt 3) := 
sorry

end real_roots_condition_l765_765489


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765387

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765387


namespace factor_polynomial_equiv_l765_765748

theorem factor_polynomial_equiv :
  (x^2 + 2 * x + 1) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 7 * x + 1) * (x^2 + 3 * x + 7) :=
by sorry

end factor_polynomial_equiv_l765_765748


namespace problem_l765_765066

variable {a : ℕ → ℝ}

-- Conditions
def geometric_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, (∏ i in finset.range n, a (i + 1)) = ℝ.T n

def condition1 : Prop := 
  geometric_seq a

def condition2 : Prop :=
  a 3 * a 4 * a 5 = 1

-- Assertions to Prove
theorem problem (h1 : condition1) (h2 : condition2) :
  (a 4 = 1) ∧ (ℝ.T 2 = ℝ.T 5) ∧ (a 1 = 1 → a 2 = 1) :=
sorry

end problem_l765_765066


namespace count_two_digit_primes_with_units_digit_3_l765_765196

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765196


namespace remaining_dimes_l765_765918

-- Define the initial quantity of dimes Joan had
def initial_dimes : Nat := 5

-- Define the quantity of dimes Joan spent
def dimes_spent : Nat := 2

-- State the theorem we need to prove
theorem remaining_dimes : initial_dimes - dimes_spent = 3 := by
  sorry

end remaining_dimes_l765_765918


namespace volume_of_cylinder_l765_765946

theorem volume_of_cylinder (J : ℝ) (side_length : ℝ) (height : ℝ) (diameter : ℝ)
  (cube_surface_area : side_length^2 * 6 = 54)
  (cylinder_surface_area : π * (diameter^2 / 2 + diameter * height) = 54)
  (cylinder_volume : π * (diameter / 2)^2 * height = J * π / 6) :
  J = 324 * real.sqrt π := 
sorry

end volume_of_cylinder_l765_765946


namespace probability_of_floor_roots_l765_765933

def is_in_range_closed (a b x : ℝ) : Prop := (a ≤ x) ∧ (x < b)

def floor_root_conditions (x : ℝ) : Prop := 
  (is_in_range_closed 484 529 x) ∧ (is_in_range_closed 490 492.204 x)

def probability := 551 / 11250

theorem probability_of_floor_roots (x : ℝ) (hx : 500 ≤ x ∧ x ≤ 550) (h : ⌊real.sqrt x⌋ = 22) : 
  (classical.some (floor_root_conditions x)) = probability :=
sorry

end probability_of_floor_roots_l765_765933


namespace count_two_digit_primes_with_units_digit_3_l765_765200

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765200


namespace two_digit_primes_with_units_digit_three_l765_765234

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765234


namespace tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l765_765094

open Real

axiom sin_add_half_pi_div_4_eq_zero (α : ℝ) : 
  sin (α + π / 4) + 2 * sin (α - π / 4) = 0

axiom tan_sub_half_pi_div_4_eq_inv_3 (β : ℝ) : 
  tan (π / 4 - β) = 1 / 3

theorem tan_alpha_eq_inv_3 (α : ℝ) (h : sin (α + π / 4) + 2 * sin (α - π / 4) = 0) : 
  tan α = 1 / 3 := sorry

theorem tan_alpha_add_beta_eq_1 (α β : ℝ) 
  (h1 : tan α = 1 / 3) (h2 : tan (π / 4 - β) = 1 / 3) : 
  tan (α + β) = 1 := sorry

end tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l765_765094


namespace inequality_system_solution_l765_765553

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l765_765553


namespace two_digit_primes_with_units_digit_three_l765_765224

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765224


namespace count_two_digit_primes_with_units_digit_3_l765_765156

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765156


namespace surface_area_of_sphere_l765_765040

theorem surface_area_of_sphere (S A B C O : Point) (r : ℝ) (π : ℝ) :
  (O = midpoint S C) ∧ (∃ SC, diameter SC O) ∧
  (plane S C A ⊥ plane S C B) ∧ (distance S A = distance A C) ∧
  (distance S B = distance B C) ∧
  (volume_tetrahedron S A B C = 9) →
  4 * π * r^2 = 36 * π :=
by sorry

end surface_area_of_sphere_l765_765040


namespace winner_percentage_l765_765903

variable (votes_winner : ℕ) (win_by : ℕ)
variable (total_votes : ℕ)
variable (percentage_winner : ℕ)

-- Conditions
def conditions : Prop :=
  votes_winner = 930 ∧
  win_by = 360 ∧
  total_votes = votes_winner + (votes_winner - win_by) ∧
  percentage_winner = (votes_winner * 100) / total_votes

-- Theorem to prove
theorem winner_percentage (h : conditions votes_winner win_by total_votes percentage_winner) : percentage_winner = 62 :=
sorry

end winner_percentage_l765_765903


namespace x_B_range_l765_765058

theorem x_B_range 
  (M N : α → α × α)
  (A : α × α)
  (x1 x2 y1 y2 y0 : α)
  (h_parabola_M : (M x1).snd ^ 2 = 4 * (M x1).fst)
  (h_parabola_N : (N x2).snd ^ 2 = 4 * (N x2).fst)
  (h_midpoint_A : A.fst = (M x1).fst + (N x2).fst)
  (h_A_x : A.fst = 3)
  (B : α × α)
  (x_B : α)
  (h_MN_B : B.snd = 0 ∧ B.fst = x_B)
  (h_line_MN_intersects_x_axis : ∃ m t : α, ∀ y, y = 0 ∨ y ^ 2 - 4 * m * y + 4 * t = 0 → B.snd = 0 ∧ B.fst = x_B) :
  -3 < x_B ∧ x_B ≤ 3 := 
sorry

end x_B_range_l765_765058


namespace box_filled_percent_l765_765696

variable (box_length box_width box_height cube_edge : ℝ)
variable (num_cubes : ℕ)

-- Define the interior dimensions of the box and the cube edge length
def conditions : Prop :=
  box_length = 8 ∧
  box_width = 6 ∧
  box_height = 12 ∧
  cube_edge = 4 ∧
  num_cubes = (⌊box_length / cube_edge⌋ * ⌊box_width / cube_edge⌋ * ⌊box_height / cube_edge⌋).to_nat

-- Define the volumes of the cubes and the box
def volume_of_cubes : Real := num_cubes * (cube_edge ^ 3)

def volume_of_box : Real := box_length * box_width * box_height

-- Define the percentage of the box taken up by the cubes
def percentage_filled : Real := (volume_of_cubes / volume_of_box) * 100

-- The theorem to be proven
theorem box_filled_percent (h : conditions) : 
  percentage_filled = 66.6667 := by
  sorry

end box_filled_percent_l765_765696


namespace hyperbola_asymptote_slope_l765_765735

theorem hyperbola_asymptote_slope
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c ≠ -a ∧ c ≠ a)
  (H1 : (c ≠ -a ∧ c ≠ a) ∧ (a ≠ 0) ∧ (b ≠ 0))
  (H_perp : (c + a) * (c - a) * (a * a * a * a) + (b * b * b * b) = 0) :
  abs (b / a) = 1 :=
by
  sorry  -- Proof here is not required as per the given instructions

end hyperbola_asymptote_slope_l765_765735


namespace linear_equation_solution_l765_765441

theorem linear_equation_solution (m n : ℤ) (x y : ℤ)
  (h1 : x + 2 * y = 5)
  (h2 : x + y = 7)
  (h3 : x = -m)
  (h4 : y = -n) :
  (3 * m + 2 * n) / (5 * m - n) = 11 / 14 :=
by
  sorry

end linear_equation_solution_l765_765441


namespace largest_prime_factor_of_two_pow_sixteen_minus_one_l765_765475

theorem largest_prime_factor_of_two_pow_sixteen_minus_one (b : ℕ) (h₁ : b = 2) (h₂ : ∃ p1 p2 p3 p4 : ℕ, [p1, p2, p3, p4].pairwise (≠) ∧ [p1, p2, p3, p4].all (prime) ∧ 2^16 - 1 = p1 * p2 * p3 * p4) : 
  ∃ p1 p2 p3 p4 : ℕ, [p1, p2, p3, p4].pairwise (≠) ∧ [p1, p2, p3, p4].all (prime) ∧ 2^16 - 1 = p1 * p2 * p3 * p4 ∧ 
  max p1 (max p2 (max p3 p4)) = 257 := 
sorry

end largest_prime_factor_of_two_pow_sixteen_minus_one_l765_765475


namespace count_prime_units_digit_3_eq_6_l765_765306

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765306


namespace solve_exponential_equation_l765_765531

theorem solve_exponential_equation (x : ℝ) :
  2 * (5^x + 6^x - 3^x) = 7^x + 9^x ↔ x = 0 ∨ x = 1 := by
  sorry

end solve_exponential_equation_l765_765531


namespace circle_tangent_major_axis_l765_765602

theorem circle_tangent_major_axis
  (F1 F2 P E A B : Point)
  (ellipse : Ellipse)
  (h_foci : ellipse.foci = (F1, F2))
  (h_p_on_ellipse : ellipse.contains P)
  (circle : Circle)
  (h_tangent_PF1 : circle.tangent (LineSeg.mk P F1))
  (h_tangent_ext_F2F1 : circle.tangent (LineSeg.mk A F1))
  (h_tangent_ext_F2P : circle.tangent (LineSeg.mk B F2)) :
  circle.passes_through (ellipse.major_axis_endpoint_closer_to F1) := 
sorry

end circle_tangent_major_axis_l765_765602


namespace max_triangles_formed_l765_765804

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangles (segments : List ℕ) : List (ℕ × ℕ × ℕ) :=
  (List.triangle_combinations segments).filter (λ (t : ℕ × ℕ × ℕ), triangle_inequality t.1 t.2 t.3)

theorem max_triangles_formed 
  (segments : List ℕ)
  (h : segments = [3, 5, 6, 9, 10])
  : (valid_triangles segments).length = 6 := 
  by
    sorry

end max_triangles_formed_l765_765804


namespace count_two_digit_prime_numbers_ending_in_3_l765_765253

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765253


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765284

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765284


namespace maximum_value_problem_l765_765079

def f (x : ℝ) : ℝ := 2*x^3 + 3*x^2 - 12*x + 1

def max_value (f : ℝ → ℝ) (a b : ℝ) : ℝ := 
  real.Sup (set.image f (set.Icc a b))

theorem maximum_value_problem : max_value f (-3) 2 = 21 := by
  sorry

end maximum_value_problem_l765_765079


namespace trapezoid_base_ratio_l765_765510

theorem trapezoid_base_ratio
  (AB CD BC DA : ℝ)
  (angle_BAD angle_ABC : ℝ)
  (trapezoid : AB || CD)
  (isosceles: AB = CD)
  (angle_60 : angle_BAD = 60)
  (circle_inscribed : ∃ O, ∀ P : ℝ, Points_in_circle O P → calc_distance P AB + calc_distance P CD = calc_distance P DA + calc_distance P BC)
  (circle_circumscribed : ∃ O, ∀ P, Points_on_circle O P → calc_angle P AB + calc_angle P CD = 360):
  AD = 3 * BC := 
sorry

end trapezoid_base_ratio_l765_765510


namespace solution_l765_765052

noncomputable def problem_statement (α : ℝ) : Prop :=
  tan (π / 4 + α) = 1 → (2 * sin α + cos α) / (3 * cos α - sin α) = 1 / 3

theorem solution {α : ℝ} : problem_statement α := by
  sorry

end solution_l765_765052


namespace sum_of_four_pairwise_relatively_prime_factors_of_46656_is_70_l765_765797

open Nat

theorem sum_of_four_pairwise_relatively_prime_factors_of_46656_is_70 :
  ∃ (a b c d : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧
    a * b * c * d = 46656 ∧ 
    gcd a b = 1 ∧ gcd a c = 1 ∧ gcd a d = 1 ∧
    gcd b c = 1 ∧ gcd b d = 1 ∧ gcd c d = 1 ∧
    a + b + c + d = 70 :=
by
  sorry

end sum_of_four_pairwise_relatively_prime_factors_of_46656_is_70_l765_765797


namespace complex_number_in_first_quadrant_l765_765993

theorem complex_number_in_first_quadrant :
  let z := (Complex.I / (1 + Complex.I)) in
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l765_765993


namespace count_two_digit_primes_with_units_digit_3_l765_765185

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765185


namespace parabola_distance_l765_765435

theorem parabola_distance (m : ℝ) (h : (∀ (p : ℝ), p = 1 / 2 → m = 4 * p)) : m = 2 :=
by
  -- Goal: Prove m = 2 given the conditions.
  sorry

end parabola_distance_l765_765435


namespace solve_inequality_system_l765_765583

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l765_765583


namespace reflection_about_pole_l765_765915

variable (ρ θ : ℝ)

def reflect_polar_about_pole (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ, θ + Real.pi)

theorem reflection_about_pole (ρ θ : ℝ) :
  reflect_polar_about_pole ρ θ = (ρ, θ + Real.pi) :=
by simp [reflect_polar_about_pole]

end reflection_about_pole_l765_765915


namespace two_digit_primes_with_units_digit_three_count_l765_765179

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765179


namespace count_two_digit_primes_with_units_digit_3_l765_765313

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765313


namespace day50_yearM_minus1_is_Friday_l765_765471

-- Define weekdays
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Weekday

-- Define days of the week for specific days in given years
def day_of (d : Nat) (reference_day : Weekday) (reference_day_mod : Nat) : Weekday :=
  match (reference_day_mod + d - 1) % 7 with
  | 0 => Sunday
  | 1 => Monday
  | 2 => Tuesday
  | 3 => Wednesday
  | 4 => Thursday
  | 5 => Friday
  | 6 => Saturday
  | _ => Thursday -- This case should never occur due to mod 7

def day250_yearM : Weekday := Thursday
def day150_yearM1 : Weekday := Thursday

-- Theorem to prove
theorem day50_yearM_minus1_is_Friday :
    day_of 50 day250_yearM 6 = Friday :=
sorry

end day50_yearM_minus1_is_Friday_l765_765471


namespace circumcircle_radius_ABD_l765_765462

-- Define parallelogram and properties
variables (A B C D : Type*) [AddCommGroup A] [VectorSpace ℝ A] 
          {a b c d : A} (h : affine_independent ℝ ![a, b, c, d]) 
          (parallelogram_ABCD : ((a - b) + (c - d) = (a - d) + (b - c)))

-- Known distances
variable (AC : ℝ) (BD : ℝ) (R : ℝ)
-- circumcircle radius for triangle ADC
variable (radius_ADC : R = 10)
variable (AC_value : AC = 15)
variable (BD_value : BD = 9)

-- The goal is to prove the radius of the circumcircle of triangle ABD is 6
theorem circumcircle_radius_ABD : 
  (∃ R1 : ℝ, R1 = 6) := sorry

end circumcircle_radius_ABD_l765_765462


namespace five_n_minus_twelve_mod_nine_l765_765868

theorem five_n_minus_twelve_mod_nine (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end five_n_minus_twelve_mod_nine_l765_765868


namespace locus_of_A_min_ratio_l765_765817

-- Given conditions
def triangle := (A B C : ℝ × ℝ)
variables (B C : ℝ × ℝ)
def B := (-1, 0)
def C := (1, 0)
def AB_AC := 4

-- To prove the locus of A
theorem locus_of_A (A : ℝ × ℝ) (h : dist A B + dist A C = AB_AC) : 
  (∃ x y : ℝ, (x, y) = A ∧ (x^2 / 4) + (y^2 / 3) = 1 ∧ y ≠ 0) :=
sorry

-- Variables for the second part
variables (P : ℝ × ℝ) (S_1 S_2 : ℝ)
def r_1 := (y_0 : ℝ) := y_0 / 3
def r_2 := (x_0 y_0 : ℝ) := abs(3 / (2 * y_0) + y_0 / 6)

-- To prove the minimum ratio of S_2 to S_1
theorem min_ratio (h : (x_0^2 / 4) + (y_0^2 / 3) = 1) :
  ∃ y_0 : ℝ, y_0^2 = 3 ∧ ∀ y_0, (9 / (2 * y_0^2) + 1 / 2) * (3 / y_0) =
  4 :=
sorry

end locus_of_A_min_ratio_l765_765817


namespace count_five_digit_numbers_l765_765107

theorem count_five_digit_numbers :
  let digits := [3, 3, 3, 8, 8] in
  fintype.card {l : list ℕ // l.perm digits ∧ (∀ x ∈ l, x ∈ digits)} = 10 :=
by
  sorry

end count_five_digit_numbers_l765_765107


namespace two_digit_primes_with_units_digit_three_l765_765230

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765230


namespace count_five_digit_numbers_l765_765108

theorem count_five_digit_numbers :
  let digits := [3, 3, 3, 8, 8] in
  fintype.card {l : list ℕ // l.perm digits ∧ (∀ x ∈ l, x ∈ digits)} = 10 :=
by
  sorry

end count_five_digit_numbers_l765_765108


namespace probability_two_out_of_four_approve_l765_765451

theorem probability_two_out_of_four_approve (p : ℝ) (n : ℕ) (k : ℕ) (probability_approve : p = 0.6) (total_voters : n = 4) (approve_exactly_two : k = 2) :
  let q := 1 - p in
  let binomial_coefficient := Nat.choose n k in
  (binomial_coefficient * (p ^ k) * (q ^ (n - k))) = 0.864 :=
by
  sorry

end probability_two_out_of_four_approve_l765_765451


namespace count_prime_units_digit_3_eq_6_l765_765301

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765301


namespace hyperbola_equation_l765_765872

theorem hyperbola_equation (λ : ℝ) (hyp : λ ≠ 0) 
  (asymptotic_eq : ∀ (x y : ℝ), (y = (1/2) * x ∨ y = -(1/2) * x) -> x^2 - 4 * y^2 = λ)
  (passes_through : (4, Real.sqrt 3) ∈ {p : ℝ × ℝ | p.1^2 - 4 * p.2^2 = λ}) :
  ∀ (x y : ℝ), x^2 / 4 - y^2 = 1 :=
by
  sorry

end hyperbola_equation_l765_765872


namespace triangle_is_isosceles_l765_765870

theorem triangle_is_isosceles (a b c : ℝ) (A B C : ℝ)
  (h1 : ∀ (a b c : ℝ) (A B C : ℝ), 2 * a * cos B = c) : 
  A = B :=
sorry

end triangle_is_isosceles_l765_765870


namespace count_two_digit_primes_with_units_digit_3_l765_765193

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765193


namespace sum_to_infinity_series_l765_765013

theorem sum_to_infinity_series : 
  (∑ (n : ℕ) in (Finset.range (1000)), (-1)^(n+1) / (3 * ↑n - 2)) = 1 / 3 * (Real.log 2 + 2 * Real.pi / Real.sqrt 3) :=
by
  sorry

end sum_to_infinity_series_l765_765013


namespace solve_inequality_system_l765_765568

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765568


namespace linear_function_range_l765_765879

theorem linear_function_range (k : ℝ) (b : ℝ) :
  (∃ b, y = k * x + b) ∧ (f (2, 2) = true) ∧ (∀ x ∈ [0, 3], g(x) = -x + 3)
  → (k ≤ -2 ∨ (k ≥ -1/2 ∧ k ≠ 0)) :=
by sorry

end linear_function_range_l765_765879


namespace two_digit_primes_with_units_digit_three_count_l765_765174

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765174


namespace area_of_ABCD_is_correct_l765_765624

noncomputable def quadrilateral_area (AB BD BC : ℝ) (h1 : AB = 15) (h2 : BD = 20) (h3 : BC = 25)
  (right_triangle_BAD : ∃ AD : ℝ, (AD^2 + AB^2 = BD^2)) 
  (right_triangle_BDC : ∃ DC : ℝ, (DC^2 + BD^2 = BC^2)) : ℝ :=
  (75 * (Real.sqrt 7 + 4)) / 2

theorem area_of_ABCD_is_correct :
  quadrilateral_area 15 20 25 
    (by rfl) 
    (by rfl) 
    (by rfl)
    (by use (5 * Real.sqrt 7); linarith [pow_two_nonneg _, pow_two_nonneg _])
    (by use (15); linarith [pow_two_nonneg _, pow_two_nonneg _])
  = (75 * (Real.sqrt 7 + 4)) / 2 :=
sorry

end area_of_ABCD_is_correct_l765_765624


namespace num_two_digit_primes_with_units_digit_three_l765_765133

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765133


namespace intersection_of_P_and_Q_l765_765421

theorem intersection_of_P_and_Q (P Q : Set ℕ) (h1 : P = {1, 3, 6, 9}) (h2 : Q = {1, 2, 4, 6, 8}) :
  P ∩ Q = {1, 6} :=
by
  sorry

end intersection_of_P_and_Q_l765_765421


namespace tangent_line_sin_at_pi_l765_765782

theorem tangent_line_sin_at_pi :
  ∀ (f : ℝ → ℝ), 
    (∀ x, f x = Real.sin x) → ∀ x y, (x, y) = (Real.pi, 0) → 
    ∃ (m : ℝ) (b : ℝ), (∀ x, y = m * x + b) ∧ (m = -1) ∧ (b = Real.pi) :=
by
  sorry

end tangent_line_sin_at_pi_l765_765782


namespace perpendicular_condition_l765_765801

variables {α : Type*} [plane α]
variables {a l : line α} {ℓ : line} (h₁ : a ⊆ α) (h₂ : ℓ ⊥ a)

theorem perpendicular_condition (h₁ : a ⊆ α) (h₂ : ℓ ⊥ a) :
  (ℓ ⊥ α → ℓ ⊥ a) ∧ ¬(ℓ ⊥ a → ℓ ⊥ α) := 
sorry

end perpendicular_condition_l765_765801


namespace axis_of_symmetry_points_on_curve_y_coordinate_range_l765_765980

noncomputable def curve_eq (x y : ℝ) : Prop := 
  (sqrt(x^2 + (y + 1)^2) * sqrt(x^2 + (y - 1)^2) = 3)

theorem axis_of_symmetry:
  (∀ (x y : ℝ), curve_eq x y → curve_eq (-x) y) := sorry

theorem points_on_curve:
  curve_eq 0 2 ∧ curve_eq 0 (-2) := sorry

theorem y_coordinate_range:
  (∀ (x y : ℝ), curve_eq x y → -2 ≤ y ∧ y ≤ 2) := sorry

end axis_of_symmetry_points_on_curve_y_coordinate_range_l765_765980


namespace count_integers_with_digit_sum_eq_18_l765_765114

theorem count_integers_with_digit_sum_eq_18 : 
  let nums_in_range := {n : ℕ | 400 ≤ n ∧ n ≤ 600}
  let digit_sum_eq (n : ℕ) := (n / 100) + ((n % 100) / 10) + (n % 10) = 18
  ∃ (count : ℕ), count = 21 ∧ count = (nums_in_range.count digit_sum_eq) := 
by 
  sorry

end count_integers_with_digit_sum_eq_18_l765_765114


namespace translated_line_y_intercept_l765_765987

theorem translated_line_y_intercept :
  ∃ b : ℝ, 
  (∀ x : ℝ, (y = λ x, x - 5)) ↔ (x = 0 → y = -5) :=
by sorry

end translated_line_y_intercept_l765_765987


namespace functional_equation_find_expression_value_l765_765984

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (x y : ℝ) : 
  f(2 * x - 3 * y) - f(x + y) = -2 * x + 8 * y := sorry

theorem find_expression_value (t : ℝ) (h : ∀ x y, f(2 * x - 3 * y) - f (x + y) = -2 * x + 8 * y) : 
  (f(5 * t) - f(t)) / (f(4 * t) - f(3 * t)) = 4 :=
sorry

end functional_equation_find_expression_value_l765_765984


namespace find_pairs_l765_765767

theorem find_pairs (a b : ℕ) :
  (∃ (a b : ℕ), (b^2 - a ≠ 0) ∧ (a^2 - b ≠ 0) ∧ (a^2 + b) / (b^2 - a) ∈ ℤ ∧ (b^2 + a) / (a^2 - b) ∈ ℤ) → 
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end find_pairs_l765_765767


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765384

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765384


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765335

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765335


namespace custom_op_test_l765_765689

def custom_op (a b : ℝ) : ℝ := real.sqrt b + a

theorem custom_op_test : custom_op 15 196 = 29 :=
by
  sorry

end custom_op_test_l765_765689


namespace count_prime_units_digit_3_eq_6_l765_765296

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765296


namespace probability_first_three_odd_product_l765_765976

-- Define the problem conditions
variable (selected_integers : Finset ℕ) (h_selected: selected_integers.card = 10) (h_range : ∀ x ∈ selected_integers, 1 ≤ x ∧ x ≤ 3000)

-- Define the probability calculation function (not actual calculations, just variables for the problem statement)
noncomputable def probability_odd_product (selected_integers : Finset ℕ) : ℚ := sorry

-- Define the proof problem statement
theorem probability_first_three_odd_product
  (h_selected: selected_integers.card = 10)
  (h_range : ∀ x ∈ selected_integers, 1 ≤ x ∧ x ≤ 3000) :
  let p := probability_odd_product selected_integers in
  (1/16 : ℚ) < p ∧ p < (1/8 : ℚ) :=
sorry

end probability_first_three_odd_product_l765_765976


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765341

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765341


namespace count_two_digit_primes_with_units_digit_3_l765_765194

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765194


namespace find_b2_l765_765943

noncomputable def sequence_b : ℕ → ℝ
| 1 := 24
| n := 
    if n = 12 then 150 
    else 
        if n ≥ 3 then (sequence_b 1 + sequence_b 2 + ∑ i in finset.range (n-2), sequence_b (i+1)) / (n-1)
        else 0

theorem find_b2 : sequence_b 2 = 276 := by
    sorry

end find_b2_l765_765943


namespace perfect_square_base9_last_digit_l765_765430

-- We define the problem conditions
variable {b d f : ℕ} -- all variables are natural numbers
-- Condition 1: Base 9 representation of a perfect square
variable (n : ℕ) -- n is the perfect square number
variable (sqrt_n : ℕ) -- sqrt_n is the square root of n (so, n = sqrt_n^2)
variable (h1 : n = b * 9^3 + d * 9^2 + 4 * 9 + f)
variable (h2 : b ≠ 0)
-- The question becomes that the possible values of f are 0, 1, or 4
theorem perfect_square_base9_last_digit (h3 : n = sqrt_n^2) (hb : b ≠ 0) : 
  (f = 0) ∨ (f = 1) ∨ (f = 4) :=
by
  sorry

end perfect_square_base9_last_digit_l765_765430


namespace general_term_formula_sum_bounded_l765_765039

open Nat

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}

-- Given condition 1: For any n ∈ ℕ, 2 * a n + S n = 2
def condition1 (n : ℕ) : Prop := 2 * a n + S n = 2

-- Given condition 2: For any n ∈ ℕ, b n = a (n + 1) / ((a (n + 1) + 2) * (a n + 2))
def condition2 (n : ℕ) : Prop := b n = a (n + 1) / ((a (n + 1) + 2) * (a n + 2))

-- Proof problem 1: to find a general formula for a_n, that is, ∃ (c r : ℝ), a n = c * r ^ n
theorem general_term_formula (n : ℕ) (h : ∀ n, condition1 n) : 
    ∃ (c r : ℝ) (c_pos : 0 < c) (r_pos : 0 < r), (c = 2/3) ∧ (r = 2/3) ∧ (a n = c * r ^ n) := 
sorry

-- Proof problem 2: T n < 1/2
theorem sum_bounded (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) : 
    T n < 1 / 2 :=
sorry

end general_term_formula_sum_bounded_l765_765039


namespace series_sum_equality_l765_765732

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, 12^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem series_sum_equality : sum_series = 1 := 
by sorry

end series_sum_equality_l765_765732


namespace demand_relationship_minimum_production_l765_765680

-- Define the total demand function f(x).
def f (x : ℕ) : ℕ := x * (x + 1) * (35 - 2 * x)

-- Define the monthly demand function g(x).
def g (x : ℕ) : ℕ := -6 * x^2 + 72 * x

-- The proof statement 1: relationship between f(x) and g(x).
theorem demand_relationship (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 12) : g(x) = f(x) - f(x - 1) :=
by
  sorry

-- The proof statement 2: minimum production a.
theorem minimum_production (a : ℕ) (hx : ∀ x, 1 ≤ x ∧ x ≤ 12 → a ≥ g(x)) : a ≥ 171 :=
by
  sorry

end demand_relationship_minimum_production_l765_765680


namespace count_valid_M_l765_765858

open Real

def count_valid_integers_less_than (n: ℕ) : ℕ := sorry

theorem count_valid_M :
  count_valid_integers_less_than 2000 = 412 :=
sorry

end count_valid_M_l765_765858


namespace count_prime_units_digit_3_eq_6_l765_765305

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765305


namespace paul_erasers_l765_765956

theorem paul_erasers (E : ℕ) (E_crayons : E + 353 = 391) : E = 38 := 
by
  sorry

end paul_erasers_l765_765956


namespace count_squares_in_G_l765_765616

def G' (x y : ℤ) : Prop := (2 ≤ abs x ∧ abs x ≤ 8) ∧ (2 ≤ abs y ∧ abs y ≤ 8)

theorem count_squares_in_G' : 
  (finset.card (finset.filter 
    (λ ⟨(x1, y1), (x2, y2), (x3, y3), (x4, y4)⟩,
      x1 = x2 ∧ x3 = x4 ∧ y1 = y3 ∧ y2 = y4 ∧
      abs (x1 - x3) = 7 ∧ abs (y1 - y2) = 7 ∧
      G' x1 y1 ∧ G' x2 y2 ∧ G' x3 y3 ∧ G' x4 y4)
    ((finset.product 
        (finset.Icc 2 8) (finset.Icc 2 8)).product
      (finset.product 
        (finset.Icc 2 8) (finset.Icc 2 8)))).filter
    (λ ⟨(x1, y1), (x2, y2), (x3, y3), (x4, y4)⟩,
      x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4)) 
   = 4 :=
begin
  sorry
end

end count_squares_in_G_l765_765616


namespace count_two_digit_prime_numbers_ending_in_3_l765_765249

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765249


namespace digit_sum_400_to_600_equals_18_l765_765118

theorem digit_sum_400_to_600_equals_18 :
  card {n : ℕ | 400 ≤ n ∧ n ≤ 600 ∧ (n.digits 10).sum = 18} = 21 :=
sorry

end digit_sum_400_to_600_equals_18_l765_765118


namespace count_two_digit_primes_with_units_digit_3_l765_765314

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765314


namespace sqrt_E_nature_l765_765927

def E (x : ℤ) : ℤ :=
  let a := x
  let b := x + 1
  let c := a * b
  let d := b * c
  a^2 + b^2 + c^2 + d^2

theorem sqrt_E_nature : ∀ x : ℤ, (∃ n : ℤ, n^2 = E x) ∧ (∃ m : ℤ, m^2 ≠ E x) :=
  by
  sorry

end sqrt_E_nature_l765_765927


namespace variance_binomial_example_l765_765037

variable (X : Type) [Probability (X → Bool)]

def variance_binomial (n : ℕ) (p : ℚ) : ℚ :=
  n * p * (1 - p)

theorem variance_binomial_example : 
  variance_binomial 3 (2/5) = 18 / 25 :=
by
  have h1 : variance_binomial 3 (2/5) = 3 * (2/5) * (1 - (2/5)) := rfl
  have h2 : (1 - (2/5)) = (3/5) := by norm_num
  rw [h2] at h1
  have h3 : 3 * (2/5) * (3/5) = (3 * 2 * 3) / (5 * 5) := by norm_num
  rw [h3] at h1
  exact h1

end variance_binomial_example_l765_765037


namespace digit_sum_400_to_600_equals_18_l765_765117

theorem digit_sum_400_to_600_equals_18 :
  card {n : ℕ | 400 ≤ n ∧ n ≤ 600 ∧ (n.digits 10).sum = 18} = 21 :=
sorry

end digit_sum_400_to_600_equals_18_l765_765117


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765270

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765270


namespace range_of_a_l765_765018

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (x > 0) ∧ (π^x = (a + 1) / (2 - a))) → (1 / 2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l765_765018


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765395

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765395


namespace rectangle_properties_l765_765695

theorem rectangle_properties:
  let l := 10 in
  let w := l - 3 in
  let P := 2 * (l + w) in
  let A := l * w in
  P = 34 ∧ A = 70 :=
by
  sorry

end rectangle_properties_l765_765695


namespace equation_has_three_distinct_solutions_iff_l765_765795

theorem equation_has_three_distinct_solutions_iff (a : ℝ) : 
  (∃ x_1 x_2 x_3 : ℝ, x_1 ≠ x_2 ∧ x_2 ≠ x_3 ∧ x_1 ≠ x_3 ∧ 
    (x_1 * |x_1 - a| = 1) ∧ (x_2 * |x_2 - a| = 1) ∧ (x_3 * |x_3 - a| = 1)) ↔ a > 2 :=
by
  sorry


end equation_has_three_distinct_solutions_iff_l765_765795


namespace problem_solution_l765_765972

noncomputable def solveSystem : Prop :=
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ),
    (x1 + x2 + x3 = 6) ∧
    (x2 + x3 + x4 = 9) ∧
    (x3 + x4 + x5 = 3) ∧
    (x4 + x5 + x6 = -3) ∧
    (x5 + x6 + x7 = -9) ∧
    (x6 + x7 + x8 = -6) ∧
    (x7 + x8 + x1 = -2) ∧
    (x8 + x1 + x2 = 2) ∧
    (x1 = 1) ∧
    (x2 = 2) ∧
    (x3 = 3) ∧
    (x4 = 4) ∧
    (x5 = -4) ∧
    (x6 = -3) ∧
    (x7 = -2) ∧
    (x8 = -1)

theorem problem_solution : solveSystem :=
by
  -- Skip the proof for now
  sorry

end problem_solution_l765_765972


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765339

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765339


namespace sequence_property_l765_765815

noncomputable def sequence (a b : ℕ) : ℕ → ℕ
| 1 => a
| 2 => b
| (n+1) => sequence n - sequence (n-1)

def S (a b : ℕ) (n : ℕ) : ℕ := (List.range n).map (λ i => sequence a b (i + 1)).sum

theorem sequence_property (a b : ℕ) : sequence a b 100 = -a ∧ S a b 100 = 2 * b - a := sorry

end sequence_property_l765_765815


namespace two_digit_primes_units_digit_3_count_l765_765356

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765356


namespace part_a_part_b_part_c_part_d_l765_765521

section
variables {r r_a a p b : ℝ} {α β γ : ℝ}

-- Part (a)
theorem part_a 
  (h1 : a = r * (Real.cot (β / 2) + Real.cot (γ / 2)))
  (h2 : a = r * (Real.cos (α / 2) / (Real.sin (β / 2) * Real.sin (γ / 2)))) :
  a = r * (Real.cot (β / 2) + Real.cot (γ / 2)) ∧ a = r * (Real.cos (α / 2) / (Real.sin (β / 2) * Real.sin (γ / 2))) :=
begin
  exact ⟨h1, h2⟩,
end

-- Part (b)
theorem part_b 
  (h1 : a = r_a * (Real.tan (β / 2) + Real.tan (γ / 2)))
  (h2 : a = r_a * (Real.cos (α / 2) / (Real.cos (β / 2) * Real.cos (γ / 2)))) :
  a = r_a * (Real.tan (β / 2) + Real.tan (γ / 2)) ∧ a = r_a * (Real.cos (α / 2) / (Real.cos (β / 2) * Real.cos (γ / 2))) :=
begin
  exact ⟨h1, h2⟩,
end

-- Part (c)
theorem part_c 
  (h1 : p - b = Real.cot (β / 2))
  (h2 : p - b = r_a * Real.tan (γ / 2)) :
  p - b = Real.cot (β / 2) ∧ p - b = r_a * Real.tan (γ / 2) :=
begin
  exact ⟨h1, h2⟩,
end

-- Part (d)
theorem part_d 
  (h : p = r_a * Real.cot (α / 2)) :
  p = r_a * Real.cot (α / 2) :=
begin
  exact h,
end

end

end part_a_part_b_part_c_part_d_l765_765521


namespace count_two_digit_primes_with_units_digit_3_l765_765186

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765186


namespace find_smallest_angle_l765_765012

noncomputable def smallest_angle (θ : ℝ) : Prop :=
  cos θ = sin (Real.pi / 4) + cos (Real.pi / 3) - sin (Real.pi / 6) - cos (Real.pi / 12)

theorem find_smallest_angle : ∃ θ > 0, θ = 30 ∧ smallest_angle θ :=
by
  sorry

end find_smallest_angle_l765_765012


namespace find_pairs_l765_765766

theorem find_pairs (a b : ℕ) :
  (∃ (a b : ℕ), (b^2 - a ≠ 0) ∧ (a^2 - b ≠ 0) ∧ (a^2 + b) / (b^2 - a) ∈ ℤ ∧ (b^2 + a) / (a^2 - b) ∈ ℤ) → 
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end find_pairs_l765_765766


namespace probability_greater_than_n_l765_765901

theorem probability_greater_than_n (n : ℕ) : 
  (1 ≤ n ∧ n ≤ 5) → (∃ k, k = 6 - n - 1 ∧ k / 6 = 1 / 2) → n = 3 := 
by sorry

end probability_greater_than_n_l765_765901


namespace harrys_father_is_older_l765_765095

def harrys_age := 50
def mothers_age (H : ℕ) := H + 22
def fathers_age (M H : ℕ) := M + (H / 25)

theorem harrys_father_is_older :
  let H := harrys_age in let M := mothers_age H in let F := fathers_age M H in F - H = 24 :=
by
  let H := harrys_age
  let M := mothers_age H
  let F := fathers_age M H
  sorry

end harrys_father_is_older_l765_765095


namespace Problem1_Problem2_l765_765667

theorem Problem1 : 
  let A (n m : ℕ) := Nat.factorial n / Nat.factorial (n - m) in
  A 5 1 + A 5 2 + A 5 3 + A 5 4 + A 5 5 = 325 := 
by {
  -- use the definition of A to get 5 + 20 + 60 + 120 + 120 = 325
  sorry
}

theorem Problem2 (m : ℕ) (h : m > 1) (h_eq : Nat.choose 5 m = Nat.choose 5 (2 * m - 1)) : 
  Nat.choose 6 m + Nat.choose 6 (m + 1) + Nat.choose 7 (m + 2) + Nat.choose 8 (m + 3) = 126 := 
by {
  -- derive m from the conditions to get m=2, then show 
  -- combinatorial identities to sum to 126
  sorry
}

end Problem1_Problem2_l765_765667


namespace exists_distinct_numbers_satisfy_conditions_l765_765474

theorem exists_distinct_numbers_satisfy_conditions :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a + b + c = 6) ∧
  (2 * b = a + c) ∧
  ((b^2 = a * c) ∨ (a^2 = b * c) ∨ (c^2 = a * b)) :=
by
  sorry

end exists_distinct_numbers_satisfy_conditions_l765_765474


namespace number_of_two_digit_primes_with_units_digit_three_l765_765210

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765210


namespace find_range_of_a_l765_765839

theorem find_range_of_a (a : ℝ) :
  (∃ x y: ℝ, x^2 + (a^2 - 1) * x + a - 2 = 0 ∧ y^2 + (a^2 - 1) * y + a - 2 = 0 ∧ x > 1 ∧ y < 1) →
  a ∈ set.Ioo (-2 : ℝ) 1 := 
sorry

end find_range_of_a_l765_765839


namespace inequality_system_solution_l765_765550

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l765_765550


namespace inequality_solution_l765_765543

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l765_765543


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765285

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765285


namespace inequality_solution_l765_765542

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l765_765542


namespace two_digit_primes_units_digit_3_count_l765_765352

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765352


namespace values_of_d_l765_765496

theorem values_of_d (a b c d : ℕ) 
  (h : (ad - 1) / (a + 1) + (bd - 1) / (b + 1) + (cd - 1) / (c + 1) = d) : 
  d = 1 ∨ d = 2 ∨ d = 3 := 
sorry

end values_of_d_l765_765496


namespace complex_number_quadrant_l765_765997

def z : ℂ := (↑complex.I) / (1 + ↑complex.I)

theorem complex_number_quadrant : z.re > 0 ∧ z.im > 0 := sorry

end complex_number_quadrant_l765_765997


namespace count_integers_with_digit_sum_l765_765124

theorem count_integers_with_digit_sum :
  let count := (finset.range 600).filter (λ n, n >= 400 ∧ (n.digits.sum = 18)).card
  count = 20 :=
by
  sorry

end count_integers_with_digit_sum_l765_765124


namespace solve_inequality_system_l765_765580

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l765_765580


namespace count_two_digit_primes_with_units_digit_three_l765_765402

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765402


namespace boy_share_of_payment_l765_765664

noncomputable def man_work_rate := 1 / 10
noncomputable def combined_work_rate := 1 / 6
noncomputable def total_payment := 50

theorem boy_share_of_payment :
  ∃ (boy_work_rate : ℝ) (boy_share : ℝ),
    man_work_rate + boy_work_rate = combined_work_rate ∧ 
    boy_work_rate = 1 / 15 ∧
    boy_share = 20 :=
begin
  existsi (1 / 15 : ℝ),
  existsi (20 : ℝ),
  split; try {sorry},
  split; try {sorry},
  sorry,
end

end boy_share_of_payment_l765_765664


namespace nearest_integer_power_l765_765640

theorem nearest_integer_power (x : ℝ) (x_val : x = 3 + sqrt 5) : 
  (Real.round (x^4) : ℤ) = 376 :=
by
  sorry

end nearest_integer_power_l765_765640


namespace two_digit_primes_units_digit_3_count_l765_765347

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765347


namespace count_prime_units_digit_3_eq_6_l765_765299

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765299


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765391

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765391


namespace number_of_trees_l765_765442

theorem number_of_trees (length_of_road intervals : ℕ) (h_length : length_of_road = 2575) (h_intervals : intervals = 25) :
  (length_of_road / intervals) + 1 = 104 :=
by
  rw [h_length, h_intervals]
  sorry

end number_of_trees_l765_765442


namespace verify_integer_pairs_l765_765760

open Nat

theorem verify_integer_pairs (a b : ℕ) :
  (∃ k1 : ℤ, ↑(a^2) + ↑b = k1 * (↑(b^2) - ↑a)) ∧
  (∃ k2 : ℤ, ↑(b^2) + ↑a = k2 * (↑(a^2) - ↑b)) →
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ 
  (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end verify_integer_pairs_l765_765760


namespace ratio_milk_water_is_29_to_7_l765_765622

noncomputable def ratio_of_milk_to_water 
    (x : ℝ)
    (vessel1_ratio : ℝ × ℝ)
    (vessel2_ratio : ℝ × ℝ)
    (vessel3_ratio : ℝ × ℝ) : ℝ × ℝ :=
let milk1 := vessel1_ratio.1 / (vessel1_ratio.1 + vessel1_ratio.2) * x in
let water1 := vessel1_ratio.2 / (vessel1_ratio.1 + vessel1_ratio.2) * x in
let milk2 := vessel2_ratio.1 / (vessel2_ratio.1 + vessel2_ratio.2) * x in
let water2 := vessel2_ratio.2 / (vessel2_ratio.1 + vessel2_ratio.2) * x in
let milk3 := vessel3_ratio.1 / (vessel3_ratio.1 + vessel3_ratio.2) * x in
let water3 := vessel3_ratio.2 / (vessel3_ratio.1 + vessel3_ratio.2) * x in
let total_milk := milk1 + milk2 + milk3 in
let total_water := water1 + water2 + water3 in
(total_milk, total_water)

theorem ratio_milk_water_is_29_to_7
    (x : ℝ)
    (h1 : vessel1_ratio = (7, 2))
    (h2 : vessel2_ratio = (8, 1))
    (h3 : vessel3_ratio = (9, 3)) : 
    ratio_of_milk_to_water x vessel1_ratio vessel2_ratio vessel3_ratio = (29 / 12 * x, 7 / 12 * x) :=
sorry

end ratio_milk_water_is_29_to_7_l765_765622


namespace main_theorem_l765_765930

-- Given definitions
variable (f : ℝ → ℝ)

-- Imposing the condition of the problem
def functional_equation : Prop :=
  ∀ x y : ℝ, f (x * f y + y) = x * y^2 + f x

-- Main theorem statement
theorem main_theorem (h : functional_equation f) : 
  let n := 1 in
  let s := (0 : ℝ) in
  n * s = 0 :=
by
  sorry

end main_theorem_l765_765930


namespace two_digit_primes_with_units_digit_three_count_l765_765166

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765166


namespace solve_inequality_system_l765_765539

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765539


namespace num_two_digit_primes_with_units_digit_three_l765_765139

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765139


namespace find_a_for_three_distinct_roots_l765_765780

-- Define the condition: the equation |x^3 - a^3| = x - a
def equation_condition (x a : ℝ) : ℝ := |x^3 - a^3| = x - a

-- Define the range of a for which the equation has three distinct roots
theorem find_a_for_three_distinct_roots :
  ∀ a : ℝ, (-sqrt (1 / 3) < a) → (a < -1 / sqrt 3) →
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ 
    equation_condition x1 a ∧ 
    equation_condition x2 a ∧ 
    equation_condition x3 a) := 
sorry

end find_a_for_three_distinct_roots_l765_765780


namespace rational_numbers_c_eq_zero_l765_765523

theorem rational_numbers_c_eq_zero
  (a b c : ℚ)
  (h : (a + b + c) * (a + b - c) = 2 * c^2) :
    c = 0 := 
begin
  sorry
end

end rational_numbers_c_eq_zero_l765_765523


namespace counterfeit_probability_l765_765743

open Finset

theorem counterfeit_probability :
  let A := 5.choose 2 / 20.choose 2
  let B := (5.choose 2 + (5.choose 1 * 15.choose 1)) / 20.choose 2
  P(A | B) = (5.choose 2) / (5.choose 2 + 5.choose 1 * 15.choose 1) := 
by
  sorry

end counterfeit_probability_l765_765743


namespace product_of_five_consecutive_not_square_l765_765530

theorem product_of_five_consecutive_not_square (n : ℤ) :
  ¬ ∃ k : ℤ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = k^2) :=
by
  sorry

end product_of_five_consecutive_not_square_l765_765530


namespace count_two_digit_primes_with_units_digit_3_l765_765159

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765159


namespace inequality_solution_l765_765540

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l765_765540


namespace two_digit_primes_with_units_digit_three_l765_765222

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765222


namespace magnitude_of_z_l765_765592

noncomputable def complex_w : ℂ := sorry -- w is a complex number
noncomputable def complex_z : ℂ := sorry -- z is a complex number

axiom wz_equals : complex_w * complex_z = (15 : ℂ) - (20 : ℂ) * complex.I
axiom abs_w : abs complex_w = real.sqrt 34

theorem magnitude_of_z : abs complex_z = (25 * real.sqrt 34) / 34 :=
by sorry

end magnitude_of_z_l765_765592


namespace count_two_digit_primes_with_units_digit_3_l765_765317

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765317


namespace count_two_digit_primes_with_units_digit_3_l765_765195

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765195


namespace find_initial_population_l765_765512

-- Define the initial population and the given conditions.
def initial_population (population_after_5_years : ℝ) : ℝ :=
  let growth_factors := [1.10, 0.92, 1.15, 0.94, 1.12]
  population_after_5_years / (growth_factors.foldl (· * ·) 1)

-- Problem statement in Lean: prove that the initial population is 13440 given the final population.
theorem find_initial_population : initial_population 16875 = 13440 := by
  sorry

end find_initial_population_l765_765512


namespace count_two_digit_primes_ending_in_3_l765_765374

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765374


namespace value_of_expression_l765_765649

theorem value_of_expression : 
  ∀ (x y z : ℤ), x = -2 → y = 1 → z = 1 → (x^2 * y * z - x * y * z^2) = 6 :=
by
  assume x y z h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end value_of_expression_l765_765649


namespace two_digit_primes_units_digit_3_count_l765_765351

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765351


namespace quadrant_of_angle_l765_765422

-- Define the quadrant property
def is_in_second_quadrant (α : ℝ) : Prop :=
  (Real.sin α > 0) ∧ (Real.cos α < 0)

-- Prove that α is in the second quadrant given the conditions
theorem quadrant_of_angle (α : ℝ) (h_sin : Real.sin α > 0) (h_cos : Real.cos α < 0) : is_in_second_quadrant α :=
by
  exact ⟨h_sin, h_cos⟩

end quadrant_of_angle_l765_765422


namespace sin_value_l765_765831

noncomputable def α : ℝ := sorry

def tan_α : ℝ := Real.tan α

def quadrant_condition : Prop := α ∈ Ioo (π / 2) (3 * π / 2)

def tan_condition : Prop := tan_α = Real.sqrt 2

theorem sin_value :
  quadrant_condition ∧ tan_condition →
  Real.sin α = - Real.sqrt 6 / 3 :=
sorry

end sin_value_l765_765831


namespace solve_inequality_system_l765_765536

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765536


namespace two_digit_primes_with_units_digit_three_count_l765_765171

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765171


namespace monotonicity_f_a_pos_monotonicity_f_a_neg_range_of_a_l765_765803

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x * exp(2 * x)

theorem monotonicity_f_a_pos (a : ℝ) (h : a > 0) : 
  (∀ x, x > -1/2 → deriv (f a) x > 0) ∧ (∀ x, x < -1/2 → deriv (f a) x < 0) :=
sorry

theorem monotonicity_f_a_neg (a : ℝ) (h : a < 0) : 
  (∀ x, x < -1/2 → deriv (f a) x > 0) ∧ (∀ x, x > -1/2 → deriv (f a) x < 0) :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x, x > 0 → f(a, x) - 2 * x - log x ≥ 0) ↔ a ≥ 1/exp(1) :=
sorry

end monotonicity_f_a_pos_monotonicity_f_a_neg_range_of_a_l765_765803


namespace problem_statement_l765_765080

noncomputable def f (x : ℝ) : ℝ := x * real.exp (x + 1)
noncomputable def g (x k : ℝ) : ℝ := k * real.log x + k * (x + 1)
noncomputable def h (x k : ℝ) : ℝ := f x - g x k

theorem problem_statement (k : ℝ) (hk : k > 0) :
  (∀ x, real.has_deriv_at f x ((x + 1) * real.exp (x + 1)) ∧
       (x + 1 > 0 → deriv f x > 0) ∧
       (x + 1 < 0 → deriv f x < 0)) ∧
  (∃ x0, x0 > 0 ∧ f x0 = k ∧ ∀ x, t (x k) = f x - k ∧
       (t (0) < 0) ∧ (t (k) > 0)) ∧
  (∀ x, x > 0 → h x k ≥ 0 → k ≤ real.exp 1) :=
by
  sorry

end problem_statement_l765_765080


namespace solve_inequality_system_l765_765571

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765571


namespace m_range_iff_four_distinct_real_roots_l765_765437

noncomputable def four_distinct_real_roots (m : ℝ) : Prop :=
∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
(x1^2 - 4 * |x1| + 5 = m) ∧
(x2^2 - 4 * |x2| + 5 = m) ∧
(x3^2 - 4 * |x3| + 5 = m) ∧
(x4^2 - 4 * |x4| + 5 = m)

theorem m_range_iff_four_distinct_real_roots (m : ℝ) :
  four_distinct_real_roots m ↔ 1 < m ∧ m < 5 :=
sorry

end m_range_iff_four_distinct_real_roots_l765_765437


namespace linear_function_range_l765_765880

theorem linear_function_range (k : ℝ) (b : ℝ) :
  (∃ b, y = k * x + b) ∧ (f (2, 2) = true) ∧ (∀ x ∈ [0, 3], g(x) = -x + 3)
  → (k ≤ -2 ∨ (k ≥ -1/2 ∧ k ≠ 0)) :=
by sorry

end linear_function_range_l765_765880


namespace count_two_digit_primes_with_units_digit_3_l765_765318

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765318


namespace collinear_vectors_implies_x_value_l765_765091

-- Define the vectors a and b
def vec_a := (1 : ℝ, -2 : ℝ)
def vec_b (x : ℝ) := (-2 : ℝ, x)

-- Define collinearity condition
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- The main theorem to prove
theorem collinear_vectors_implies_x_value (x : ℝ) :
  collinear vec_a (vec_b x) → x = 4 := by
  sorry

end collinear_vectors_implies_x_value_l765_765091


namespace num_two_digit_primes_with_units_digit_three_l765_765130

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765130


namespace count_two_digit_primes_ending_in_3_l765_765373

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765373


namespace MN_perpendicular_DS_l765_765940

noncomputable theory
open_locale classical

variables {A B C D E H S N M : Type}
variables [acute_triangle A B C AB AC] [height_intersection H A B C]
variables [orthocenter D A H BC] [reflection_point E C D] 
variables [intersection S (line_through A E) (line_through B H)]
variables [midpoint N A E] [midpoint M B H]

-- Statement of the problem
theorem MN_perpendicular_DS : ⊥ (line_segment M N) (line_segment D S) :=
sorry

end MN_perpendicular_DS_l765_765940


namespace stan_average_speed_l765_765973

/-- Given two trips with specified distances and times, prove that the overall average speed is 55 mph. -/
theorem stan_average_speed :
  let distance1 := 300
  let hours1 := 5
  let minutes1 := 20
  let distance2 := 360
  let hours2 := 6
  let minutes2 := 40
  let total_distance := distance1 + distance2
  let total_time := (hours1 + minutes1 / 60) + (hours2 + minutes2 / 60)
  total_distance / total_time = 55 := 
sorry

end stan_average_speed_l765_765973


namespace machine_A_time_l765_765947

-- Definition of conditions
def work_rate (hours : ℝ) : ℝ := 1 / hours

-- Definitions for given conditions
def machine_B_time : ℝ := 2
def machine_C_time : ℝ := 6
def combined_time : ℝ := 12 / 11

-- Statement to prove
theorem machine_A_time (t_A : ℝ) :
  work_rate t_A + work_rate machine_B_time + work_rate machine_C_time = work_rate combined_time → t_A = 4 := 
sorry

end machine_A_time_l765_765947


namespace transformed_function_l765_765604

theorem transformed_function (x : ℝ) (h₁ : x ≠ 0) : 
  (let f := (λ x, 1 / x) in
   let g := (λ x, 2 - f x) in
   let h := (λ x, -1 / (x + 2)) in
   g y = h x) :=
by sorry

end transformed_function_l765_765604


namespace solve_system_l765_765945

theorem solve_system (x y : ℝ) (h1 : x + 3 * y = 20) (h2 : x + y = 10) : x = 5 ∧ y = 5 := 
by 
  sorry

end solve_system_l765_765945


namespace planes_parallel_l765_765670

-- Define the cube and necessary conditions
structure Cube :=
  (A B C D A1 B1 C1 D1 : Point)
  (AB_eq : A ≠ B)
  (BC_eq : B ≠ C)
  (CD_eq : C ≠ D)
  (DA_eq : D ≠ A)
  (A1_eq : A1 ≠ B1)
  (B1_eq : B1 ≠ C1)
  (C1_eq : C1 ≠ D1)
  (D1_eq : D1 ≠ A1)
  (AB1_parallel_C1D : ∀ A B1 C1 D, parallel (Line A B1) (Line C1 D))
  (AD1_parallel_BC1 : ∀ A D1 B C1, parallel (Line A D1) (Line B C1))

-- Define lines in the cube
def Line (p1 p2 : Point) : Line := sorry
def parallel (l1 l2 : Line) : Prop := sorry

-- Define planes as collections of three points
def Plane (p1 p2 p3 : Point) : Plane := sorry

-- Define given planes
def Plane_AB1D1 := Plane A B1 D1
def Plane_BC1D := Plane B C1 D

-- The statement of the problem
theorem planes_parallel (c : Cube)
  : parallel (Plane_AB1D1 c.A c.B1 c.D1) (Plane_BC1D c.B c.C1 c.D) := by
  sorry

end planes_parallel_l765_765670


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765342

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765342


namespace hulk_jump_exceeds_20000_l765_765591

theorem hulk_jump_exceeds_20000 :
  ∃ n : ℕ, n = 10 ∧ 3 ^ n > 20000 :=
begin
  use 10,
  split,
  { refl, },
  { norm_num, },
end

end hulk_jump_exceeds_20000_l765_765591


namespace range_of_f_cos_A_minus_B_l765_765843

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (x + π / 3) * Real.cos x

theorem range_of_f :
  set.range f ∩ set.Icc 0 (π / 2) = 
    { y | 0 ≤ y ∧ y ≤ 1 + sqrt 3 / 2 } :=
sorry

theorem cos_A_minus_B (A B : ℝ) 
    (hA : 0 < A ∧ A < π / 2) 
    (hB : 0 < B ∧ B < π / 2) :
    ∃ a b c : ℝ, a = sqrt 7 ∧ b = 2 ∧ c = 3 ∧ 
    ∀ f(A) = sqrt 3 / 2 ∧ 
        0 < a ∧ 0 < b ∧ 0 < c ∧ 
        a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A ∧ 
        ∀ sin(B)=sqrt(21)/7,
    Real.cos(A - B) = 5 * sqrt 7 / 14 :=
sorry

end range_of_f_cos_A_minus_B_l765_765843


namespace meat_needed_for_hamburgers_l765_765965

-- Define initial conditions
def beef_per_10_hamburgers : ℝ := 4
def chicken_per_5_hamburgers : ℝ := 2.5

-- Define the problem scenario
def hamburgers_beef : ℕ := 10
def hamburgers_chicken : ℕ := 5
def target_beef_hamburgers : ℕ := 30
def target_chicken_hamburgers : ℕ := 15

-- Calculate beef and chicken per hamburger
def beef_per_hamburger : ℝ := beef_per_10_hamburgers / hamburgers_beef
def chicken_per_hamburger : ℝ := chicken_per_5_hamburgers / hamburgers_chicken

-- Calculate total meat required
def total_beef_required : ℝ := beef_per_hamburger * target_beef_hamburgers
def total_chicken_required : ℝ := chicken_per_hamburger * target_chicken_hamburgers
def total_meat_required : ℝ := total_beef_required + total_chicken_required

-- Prove the final requirement
theorem meat_needed_for_hamburgers :
  total_meat_required = 19.5 :=
by 
  -- Sorry used to skip the proof
  sorry

end meat_needed_for_hamburgers_l765_765965


namespace count_two_digit_primes_ending_in_3_l765_765380

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765380


namespace triangle_side_relationship_l765_765443

variables {R : Type*} [OrderedSemiring R] (a b c : R)

theorem triangle_side_relationship 
  (A B C : R) 
  (ha : A = 2 * B)
  (A_angle : ∃ (x : ℝ), A = 2 * x ∧ B = x ∧ C = 180 - 3 * x)
  : a^2 = b * (b + c) :=
sorry

end triangle_side_relationship_l765_765443


namespace Emelyanov_inequality_l765_765008

theorem Emelyanov_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  sqrt (a * b / (c + a * b)) + sqrt (b * c / (a + b * c)) + sqrt (c * a / (b + c * a)) ≥ 1 :=
by
  sorry

end Emelyanov_inequality_l765_765008


namespace nearest_integer_to_exp_is_752_l765_765644

noncomputable def nearest_integer_to_exp := (3 + Real.sqrt 5) ^ 4

theorem nearest_integer_to_exp_is_752 : Int.round nearest_integer_to_exp = 752 := 
sorry

end nearest_integer_to_exp_is_752_l765_765644


namespace alcohol_percentage_new_mixture_l765_765672

theorem alcohol_percentage_new_mixture :
  ∀ (v1 v2 v3 : ℝ) (p1 p2 : ℝ),
    v1 = 11 →
    p1 = 42 →
    v2 = 3 →
    v3 = 2 →
    p2 = 60 →
    let total_volume := (v1 + v2 + v3) in
    let total_alcohol := (v1 * (p1 / 100) + v3 * (p2 / 100)) in
    (total_alcohol / total_volume) * 100 = 36.375 := by
  intros v1 v2 v3 p1 p2 h1 h2 h3 h4 h5 total_volume total_alcohol
  rw [h1, h2, h3, h4, h5]
  let total_volume := (11 + 3 + 2)
  let total_alcohol := (11 * 0.42 + 2 * 0.60)
  have h_volume : total_volume = 16 := by norm_num
  have h_alcohol : total_alcohol = 5.82 := by norm_num
  rw [h_volume, h_alcohol]
  norm_num
  sorry

end alcohol_percentage_new_mixture_l765_765672


namespace number_of_multiples_of_6_between_5_and_125_l765_765857

theorem number_of_multiples_of_6_between_5_and_125 : 
  ∃ k : ℕ, (5 < 6 * k ∧ 6 * k < 125) → k = 20 :=
sorry

end number_of_multiples_of_6_between_5_and_125_l765_765857


namespace number_of_two_digit_primes_with_units_digit_three_l765_765218

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765218


namespace count_two_digit_primes_with_units_digit_3_l765_765158

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765158


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765388

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765388


namespace randy_final_amount_l765_765522

-- Conditions as definitions
def initial_dollars : ℝ := 30
def initial_euros : ℝ := 20
def lunch_cost : ℝ := 10
def ice_cream_percentage : ℝ := 0.25
def snack_percentage : ℝ := 0.10
def conversion_rate : ℝ := 0.85

-- Main proof statement without the proof body
theorem randy_final_amount :
  let euros_in_dollars := initial_euros / conversion_rate
  let total_dollars := initial_dollars + euros_in_dollars
  let dollars_after_lunch := total_dollars - lunch_cost
  let ice_cream_cost := dollars_after_lunch * ice_cream_percentage
  let dollars_after_ice_cream := dollars_after_lunch - ice_cream_cost
  let snack_euros := initial_euros * snack_percentage
  let snack_dollars := snack_euros / conversion_rate
  let final_dollars := dollars_after_ice_cream - snack_dollars
  final_dollars = 30.30 :=
by
  sorry

end randy_final_amount_l765_765522


namespace inscribed_square_ab_l765_765707

theorem inscribed_square_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^2 + b^2 = 32) : 2 * a * b = -7 :=
by
  sorry

end inscribed_square_ab_l765_765707


namespace thre_digit_num_condition_l765_765671

theorem thre_digit_num_condition (n : ℕ) (h : n = 735) :
  (n % 35 = 0) ∧ (Nat.digits 10 n).sum = 15 := by
  sorry

end thre_digit_num_condition_l765_765671


namespace Tn_lt_2_l765_765816

-- Definitions based on identified conditions
def a (n : ℕ) : ℕ := 2^(n-1)
def S (n : ℕ) : ℕ := 2^n - 1

def b (n : ℕ) : ℝ := if n = 1 then 1 else (a n) / (a n * a (n+1) - 3 * a n + 1)
def T (n : ℕ) : ℝ := ∑ i in (range 1 (n+1)), b i

theorem Tn_lt_2 (n : ℕ) (hn : n ≥ 2) : T n < 2 := by
  sorry

end Tn_lt_2_l765_765816


namespace one_fourth_of_8_point_8_l765_765002

-- Definition of taking one fourth of a number
def oneFourth (x : ℝ) : ℝ := x / 4

-- Problem statement: One fourth of 8.8 is 11/5 when expressed as a simplified fraction
theorem one_fourth_of_8_point_8 : oneFourth 8.8 = 11 / 5 := by
  sorry

end one_fourth_of_8_point_8_l765_765002


namespace solve_g_inequality_find_a_range_l765_765848

section ProofProblem

variable {a x x1 x2 : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a) + 2 + a
def g (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

theorem solve_g_inequality :
  ∀ x : ℝ, g x < 6 ↔ x ∈ Ioo (-3) 1 :=
by
  sorry

theorem find_a_range :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, g x₁ = f x₂ a) → a ≤ 1 :=
by
  sorry

end ProofProblem

end solve_g_inequality_find_a_range_l765_765848


namespace compute_M_7_5_l765_765488

variable {α : Type*} [field α]

def M (v : vector α 2) : vector α 2 :=
  if v = ![3, 1] then ![2, 4]
  else if v = ![1, 4] then ![1, 2]
  else vector.zero _

theorem compute_M_7_5 :
  M ![7, 5] = ![6, 12] :=
sorry

end compute_M_7_5_l765_765488


namespace even_function_f_neg_inv_5_value_l765_765057

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 32^x + Real.log x / Real.log 5 else 0

theorem even_function_f_neg_inv_5_value :
  (∀ x, f (-x) = f x) →
  (∀ x, x > 0 → f x = 32^x + Real.log x / Real.log 5) →
  f (-1/5) = 1 :=
by
  intro h_even h_def
  have h₁ : f (-1/5) = f (1/5) := by sorry
  have h₂ : f (1/5) = 32^(1/5) + Real.log (1/5) / Real.log 5 := by sorry
  have h₃ : 32^(1/5) = 2 := by sorry
  have h₄ : Real.log (1/5) / Real.log 5 = -1 := by sorry
  rw [h₁, h₂, h₃, h₄]
  linarith

end even_function_f_neg_inv_5_value_l765_765057


namespace count_two_digit_primes_with_units_digit_3_l765_765327

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765327


namespace count_two_digit_primes_with_units_digit_three_l765_765400

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765400


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765278

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765278


namespace faces_painted_morning_l765_765968

def faces_of_cuboid : ℕ := 6
def faces_painted_evening : ℕ := 3

theorem faces_painted_morning : faces_of_cuboid - faces_painted_evening = 3 := 
by 
  sorry

end faces_painted_morning_l765_765968


namespace solve_inequality_system_l765_765537

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765537


namespace tiffany_mother_money_l765_765623

/--
Tiffany attends the carnival and her mother gives her some money to play on a ring toss game.
For every red bucket she tosses a ring into Tiffany gets 2 points.
For every green bucket she gets 3 points.
She gets zero points for a miss.
Every play costs her $1.
She gets 5 rings per play.
She played two games and already gotten 4 red buckets and 5 green buckets.
It is given that Tiffany can get a total of 38 points for all three games.
Prove that her mother gave her 3 dollars.
-/
theorem tiffany_mother_money :
  let points_red : ℕ := 4 * 2,
      points_green : ℕ := 5 * 3,
      total_points_first_two_games : ℕ := points_red + points_green,
      points_needed : ℕ := 38 - total_points_first_two_games,
      rings_needed : ℕ := points_needed / 3,
      games_needed : ℕ := rings_needed / 5,
      cost_per_game : ℕ := 1 in
  (games_needed + 2) * cost_per_game = 3 :=
by
  let points_red : ℕ := 4 * 2
    points_green : ℕ := 5 * 3
    total_points_first_two_games : ℕ := points_red + points_green
    points_needed : ℕ := 38 - total_points_first_two_games
    rings_needed : ℕ := points_needed / 3
    games_needed : ℕ := rings_needed / 5
    cost_per_game : ℕ := 1
  show (games_needed + 2) * cost_per_game = 3
  sorry

end tiffany_mother_money_l765_765623


namespace arith_seq_and_sum_T_sum_range_l765_765835

noncomputable def arithmetic_sequence (n d : ℕ) : ℕ := d * n

noncomputable def sum_arithmetic_sequence (a : ℕ) (n : ℕ) : ℕ := n * (n + 1) / 2

def b_sequence (n : ℕ) : ℚ := 2 * (1 / n - 1 / (n + 1))

def T_sum (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, b_sequence (i + 1)

theorem arith_seq_and_sum (h : sum_arithmetic_sequence 1 10 = 55) :
  (∀ (n : ℕ), arithmetic_sequence n 1 = n) ∧
  (∀ (n : ℕ), sum_arithmetic_sequence 1 n = n * (n + 1) / 2) :=
sorry

theorem T_sum_range (n : ℕ) : 1 ≤ T_sum n ∧ T_sum n < 2 :=
sorry

end arith_seq_and_sum_T_sum_range_l765_765835


namespace two_digit_primes_units_digit_3_count_l765_765350

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765350


namespace right_angled_triangle_max_area_l765_765792

theorem right_angled_triangle_max_area (a b : ℝ) (h : a + b = 4) : (1 / 2) * a * b ≤ 2 :=
by 
  sorry

end right_angled_triangle_max_area_l765_765792


namespace tip_percentage_l765_765959

theorem tip_percentage (s d t : ℕ) (n : ℕ) (h_s : s = 5) (h_d : d = 20) (h_n : n = 18) (h_t : t = 121) : 
  let total_cost_without_tip := s * n + d in
  let tip := t - total_cost_without_tip in
  let tip_percentage := (tip * 100) / total_cost_without_tip in
  tip_percentage = 10 := by
  sorry

end tip_percentage_l765_765959


namespace count_two_digit_primes_with_units_digit_three_l765_765408

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765408


namespace Ratio_BE_BA_l765_765515

theorem Ratio_BE_BA (O A C B E : Point) (h_circle : ∃ r : ℝ, Circle O r)
                    (h_tangent_BA : Tangent (Line B A) (Circle O r))
                    (h_tangent_BC : Tangent (Line B C) (Circle O r))
                    (h_isosceles : ∠ BAC = 80 ∧ ∠ BCA = 80)
                    (h_intersect : Intersects (Circle O r) (Line B A) E) :
  ∃ x : ℝ, x = BE / BA ∧ x = Real.sqrt 3 / 3 := 
sorry

end Ratio_BE_BA_l765_765515


namespace cone_height_to_base_ratio_l765_765038

noncomputable def cone_surface_area (r h V : ℝ) : ℝ :=
  π * r^2 + π * r * (sqrt (r^2 + h^2))

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

variables {V : ℝ} (h r : ℝ)

theorem cone_height_to_base_ratio (V : ℝ) (hb : cone_volume r h = V) : (h / r) = 3 :=
begin
  -- The proof is omitted by 'sorry';
  sorry
end

end cone_height_to_base_ratio_l765_765038


namespace solve_eq_pow_4_l765_765740

/--
Given integers x, y, z, and u, if 4^x + 4^y + 4^z = u^2 and x >= y >= z, then y = (x + z + 1) / 2 and u = 2^x + 2^z.
-/
theorem solve_eq_pow_4 (x y z u : ℤ) (h1 : 4^x + 4^y + 4^z = u^2) (h2 : x ≥ y) (h3 : y ≥ z) :
  y = (x + z + 1) / 2 ∧ u = 2^x + 2^z :=
sorry

end solve_eq_pow_4_l765_765740


namespace angle_equality_2_l765_765464

variable {A B C D M : Point}
variable [geometry : EuclideanGeometry]

open EuclideanGeometry

-- Conditions in the problem
axiom right_angle_A : ∠A = 90
axiom midpoint_M : M = midpoint B C
axiom angle_equality_1 : ∠ADC = ∠BAM

-- Goal to prove
theorem angle_equality_2 : ∠ADB = ∠CAM := sorry

end angle_equality_2_l765_765464


namespace find_marks_in_physics_l765_765655

theorem find_marks_in_physics (P C M : ℕ) (h1 : P + C + M = 225) (h2 : P + M = 180) (h3 : P + C = 140) : 
    P = 95 :=
sorry

end find_marks_in_physics_l765_765655


namespace num_unique_permutations_l765_765102

-- Given: digits 3 and 8 being repeated as described.
-- Show: number of different permutations of the digits 3, 3, 3, 8, 8 is 10.

theorem num_unique_permutations : 
  let digits := [3, 3, 3, 8, 8] in
  let total_permutations := (5!).nat_abs in             -- 5! permutations
  let repeats_correction := (3!).nat_abs * (2!).nat_abs in -- Adjusting for repeated 3's and 8's
  let unique_permutations := total_permutations / repeats_correction in
  unique_permutations = 10 :=
by
  sorry

end num_unique_permutations_l765_765102


namespace area_ratio_BDF_FDCE_l765_765470

-- Define the vertices of the triangle
variables {A B C : Point}
-- Define the points on the sides and midpoints
variables {E D F : Point}
-- Define angles and relevant properties
variables (angle_CBA : Angle B C A = 72)
variables (midpoint_E : Midpoint E A C)
variables (ratio_D : RatioSegment B D D C = 2)
-- Define intersection point F
variables (intersect_F : IntersectLineSegments (LineSegment A D) (LineSegment B E) = F)

theorem area_ratio_BDF_FDCE (h_angle : angle_CBA = 72) 
  (h_midpoint_E : midpoint_E) (h_ratio_D : ratio_D) (h_intersect_F : intersect_F)
  : area_ratio (Triangle.area B D F) (Quadrilateral.area F D C E) = 1 / 5 :=
sorry

end area_ratio_BDF_FDCE_l765_765470


namespace pizza_payment_l765_765955

theorem pizza_payment (n : ℕ) (cost : ℕ) (total : ℕ) 
  (h1 : n = 3) 
  (h2 : cost = 8) 
  (h3 : total = n * cost) : 
  total = 24 :=
by 
  rw [h1, h2] at h3 
  exact h3

end pizza_payment_l765_765955


namespace count_two_digit_primes_with_units_digit_3_l765_765312

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765312


namespace total_questions_in_test_l765_765898

theorem total_questions_in_test :
  ∃ x, (5 * x = total_questions) ∧ 
       (20 : ℚ) / total_questions > (60 / 100 : ℚ) ∧ 
       (20 : ℚ) / total_questions < (70 / 100 : ℚ) ∧ 
       total_questions = 30 :=
by
  sorry

end total_questions_in_test_l765_765898


namespace num_two_digit_prime_numbers_with_units_digit_3_l765_765383

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l765_765383


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765256

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765256


namespace probability_question_l765_765447

-- Definition of balls and drawing conditions
def total_balls : ℕ := 26
def black_balls : ℕ := 10
def red_balls : ℕ := 12
def white_balls : ℕ := 4
def balls_drawn : ℕ := 2

-- Variables for probabilities
def C(n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Definition of probabilities
def P_X_1 : ℚ := (C 22 1 * C 4 1) / C 26 2
def P_X_0 : ℚ := (C 22 2) / C 26 2

-- Statement to be proved
theorem probability_question :
  ∀ (X : ℕ),
  (X = 1 → P_X_1) ∧ (X = 0 → P_X_0) →
  (P_X_1 + P_X_0 = (C 22 1 * C 4 1 + C 22 2) / C 26 2) :=
by
  sorry

end probability_question_l765_765447


namespace john_running_time_l765_765719

theorem john_running_time
  (x : ℚ)
  (h1 : 15 * x + 10 * (9 - x) = 100)
  (h2 : 0 ≤ x)
  (h3 : x ≤ 9) :
  x = 2 := by
  sorry

end john_running_time_l765_765719


namespace symmetric_line_equation_l765_765601

theorem symmetric_line_equation (x y : ℝ) :
  (∀ x y : ℝ, x - 3 * y + 5 = 0 ↔ 3 * x - y - 5 = 0) :=
by 
  sorry

end symmetric_line_equation_l765_765601


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765328

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765328


namespace halfway_fraction_l765_765784

-- Assume a definition for the two fractions
def fracA : ℚ := 1 / 4
def fracB : ℚ := 1 / 7

-- Define the target property we want to prove
theorem halfway_fraction : (fracA + fracB) / 2 = 11 / 56 := 
by 
  -- Proof will happen here, adding sorry to indicate it's skipped for now
  sorry

end halfway_fraction_l765_765784


namespace count_prime_units_digit_3_eq_6_l765_765304

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765304


namespace angle_ratio_half_l765_765444

theorem angle_ratio_half (a b c : ℝ) (A B C : ℝ) (h1 : a^2 = b * (b + c))
  (h2 : A = 2 * B ∨ A + 2 * B = Real.pi) 
  (h3 : A + B + C = Real.pi) : 
  (B / A = 1 / 2) :=
sorry

end angle_ratio_half_l765_765444


namespace erase_500th_digit_gt_3_over_7_l765_765922

theorem erase_500th_digit_gt_3_over_7 :
  let n := 3/7 in 
  let decimal_repr := "428571".cycle -- decimal representation of 3/7 as repeating sequence
  let seq_after_erase := decimal_repr.take 499 ++ decimal_repr.drop 500 in 
  let new_val := (0 " ."" ++ seq_after_erase).to_real in 
  new_val > n :=
  sorry

end erase_500th_digit_gt_3_over_7_l765_765922


namespace find_m_l765_765822

variables (X Y : Type) [ProbabilitySpace X] [ProbabilitySpace Y]
variables (PX : Probability X) (PY : Probability Y)
variables (f : X → ℝ) (g : Y → ℝ)

-- Given conditions
axiom h1 : ∀ x, g (ite (f x = 2) x (0 : Y)) = 12 * f x + 7
axiom h2 : E(g) = 34
axiom h3 : (PX 1) = 1/4
axiom h4 : ∃ m n, (PX 2) = m ∧ (PX 3) = n ∧ (PX 4) = 1/12 ∧ (PX 1) + (PX 2) + (PX 3) + (PX 4) = 1

theorem find_m : ∃ m : ℝ, (∀ (n : ℝ), (12 * (1 * (1/4) + 2 * m + 3 * n + 4 * (1/12)) + 7 = 34 ∧ 1/4 + m + n + 1/12 = 1) → m = 1/3) :=
sorry

end find_m_l765_765822


namespace power_mod_eq_one_remainder_of_3_pow_4050_mod_11_l765_765645

theorem power_mod_eq_one (a b : ℕ) (h : 3^5 ≡ 1 [MOD 11]) : 3^(5 * b) ≡ 1 [MOD 11] :=
by
  sorry

theorem remainder_of_3_pow_4050_mod_11 : 3^4050 ≡ 1 [MOD 11] :=
by
  exact power_mod_eq_one 3 810
    (by norm_num)

end power_mod_eq_one_remainder_of_3_pow_4050_mod_11_l765_765645


namespace locus_of_M_is_perpendicular_line_l765_765028

open Real EuclideanGeometry

noncomputable def locus_points (O P : Point) (R : ℝ) : Set Point :=
  {M : Point | ∃ Q : Point, distance Q O = R ∧ 
                            is_tangent_at (circle O R) Q M ∧
                            M ∈ perpendicular_from O (line_through P Q)}

theorem locus_of_M_is_perpendicular_line (O P : Point) (R : ℝ) :
  ∃ S : Point, (locus_points O P R = {M : Point | ∃ Q : Point, 
                                          M ∈ perpendicular_from S (line_through O P) ∧
                                          distance Q O = R}) :=
sorry

end locus_of_M_is_perpendicular_line_l765_765028


namespace quadratic_root_conditions_l765_765022

theorem quadratic_root_conditions : (∃ p q : ℤ, abs p < 100 ∧ abs q < 100 ∧ (∃ x₁ x₂ : ℝ, x₂ ≠ 0 ∧ x₁ = 4 * x₂ ∧ x₁ * x₂ = q ∧ x₁ + x₂ = -p)) ↔ (p, q) ∈ 
  { (5, 4), (-5, 4), (10, 16), (-10, 16), (15, 36), (-15, 36), (20, 64), (-20, 64) } :=
sorry

end quadratic_root_conditions_l765_765022


namespace count_integers_with_digit_sum_eq_18_l765_765115

theorem count_integers_with_digit_sum_eq_18 : 
  let nums_in_range := {n : ℕ | 400 ≤ n ∧ n ≤ 600}
  let digit_sum_eq (n : ℕ) := (n / 100) + ((n % 100) / 10) + (n % 10) = 18
  ∃ (count : ℕ), count = 21 ∧ count = (nums_in_range.count digit_sum_eq) := 
by 
  sorry

end count_integers_with_digit_sum_eq_18_l765_765115


namespace quadratic_roots_range_l765_765979

theorem quadratic_roots_range (m : ℝ) :
  (∃ p n : ℝ, p > 0 ∧ n < 0 ∧ 2 * p^2 + (m + 1) * p + m = 0 ∧ 2 * n^2 + (m + 1) * n + m = 0) →
  m < 0 :=
by
  sorry

end quadratic_roots_range_l765_765979


namespace two_digit_primes_with_units_digit_three_l765_765232

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765232


namespace angle_at_intersection_l765_765698

theorem angle_at_intersection (n : ℕ) (h₁ : n = 8)
  (h₂ : ∀ i j : ℕ, (i + 1) % n ≠ j ∧ i < j)
  (h₃ : ∀ i : ℕ, i < n)
  (h₄ : ∀ i j : ℕ, (i + 1) % n = j ∨ (i + n - 1) % n = j)
  : (2 * (180 / n - (180 * (n - 2) / n) / 2)) = 90 :=
by
  sorry

end angle_at_intersection_l765_765698


namespace find_f_neg3_l765_765668

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom sum_equation : f 1 + f 2 + f 3 + f 4 + f 5 = 6

theorem find_f_neg3 : f (-3) = 6 := by
  sorry

end find_f_neg3_l765_765668


namespace basketball_first_half_score_l765_765890

/-- 
In a college basketball match between Team Alpha and Team Beta, the game was tied at the end 
of the second quarter. The number of points scored by Team Alpha in each of the four quarters
formed an increasing geometric sequence, and the number of points scored by Team Beta in each
of the four quarters formed an increasing arithmetic sequence. At the end of the fourth quarter, 
Team Alpha had won by two points, with neither team scoring more than 100 points. 
Prove that the total number of points scored by the two teams in the first half is 24.
-/
theorem basketball_first_half_score 
  (a r : ℕ) (b d : ℕ)
  (h1 : a + a * r = b + (b + d))
  (h2 : a + a * r + a * r^2 + a * r^3 = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
  (h3 : a + a * r + a * r^2 + a * r^3 ≤ 100)
  (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 100) : 
  a + a * r + b + (b + d) = 24 :=
  sorry

end basketball_first_half_score_l765_765890


namespace count_two_digit_primes_with_units_digit_three_l765_765417

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765417


namespace solve_inequality_system_l765_765562

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765562


namespace complex_in_first_quadrant_l765_765081

theorem complex_in_first_quadrant (i : ℂ) (h_im : i.im = 1) (h_re : i.re = 0) :
  let z := (2 * i) / (1 + i)
  z.re > 0 ∧ z.im > 0 :=
by
  let z := (2 * ⟨0, 1⟩) / (1 + ⟨0, 1⟩)
  -- sorry to skip the proof.
  sorry

end complex_in_first_quadrant_l765_765081


namespace smallest_lcm_example_l765_765425

noncomputable def smallest_lcm (a b : ℕ) : ℕ :=
  if h : a > 999 ∧ a < 10000 ∧ b > 999 ∧ b < 10000 ∧ gcd a b = 5 then
    Nat.lcm a b
  else 0

theorem smallest_lcm_example :
  smallest_lcm 1005 1010 = 203010 :=
by
  unfold smallest_lcm
  split_ifs
  · simp [h]
  · contradiction

end smallest_lcm_example_l765_765425


namespace two_digit_primes_units_digit_3_count_l765_765361

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l765_765361


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765345

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765345


namespace count_two_digit_primes_with_units_digit_3_l765_765324

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765324


namespace count_two_digit_primes_with_units_digit_3_l765_765315

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765315


namespace number_of_two_digit_primes_with_units_digit_three_l765_765203

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765203


namespace smallest_lcm_l765_765427

theorem smallest_lcm (a b : ℕ) (h₁ : 1000 ≤ a ∧ a < 10000) (h₂ : 1000 ≤ b ∧ b < 10000) (h₃ : Nat.gcd a b = 5) : 
  Nat.lcm a b = 201000 :=
sorry

end smallest_lcm_l765_765427


namespace sequence_general_term_l765_765466

noncomputable def S (n : ℕ) : ℚ := (n + 1) / n

def a : ℕ → ℚ
| 1       := 2
| (n + 1) := -1 / ((n + 1) * n)

theorem sequence_general_term :
  ∀ n : ℕ, a n = if n = 1 then 2 else -1 / (n * (n - 1)) :=
by
  intro n
  cases n
  { simp [a] }
  { sorry }

end sequence_general_term_l765_765466


namespace two_digit_primes_with_units_digit_three_l765_765237

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765237


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765289

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765289


namespace num_integers_digit_sum_18_400_600_l765_765121

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem num_integers_digit_sum_18_400_600 : 
  {n : ℕ // 400 ≤ n ∧ n < 600 ∧ digit_sum n = 18}.card = 21 := sorry

end num_integers_digit_sum_18_400_600_l765_765121


namespace probability_two_mills_rectified_probability_at_least_one_mill_shut_down_average_mills_needing_rectification_l765_765599

section PollutionInspection

-- Conditions
def number_of_paper_mills : ℕ := 5
def prob_pass_initial_inspection : ℝ := 0.5
def prob_pass_after_rectification : ℝ := 0.8

-- Probabilities
def prob_rectification_needed : ℝ := 1 - prob_pass_initial_inspection
def prob_shut_down : ℝ := prob_rectification_needed * (1 - prob_pass_after_rectification)

-- Proving the provided answers
theorem probability_two_mills_rectified :
  (nat.choose number_of_paper_mills 2 : ℝ) *
    (prob_rectification_needed ^ 2) *
    (prob_pass_initial_inspection ^ 3) =
  0.3125 := sorry

theorem probability_at_least_one_mill_shut_down :
  (1 - (1 - prob_shut_down) ^ number_of_paper_mills) =
  0.40951 := sorry

theorem average_mills_needing_rectification :
  (number_of_paper_mills : ℝ) * prob_rectification_needed = 2.5 := sorry

end PollutionInspection

end probability_two_mills_rectified_probability_at_least_one_mill_shut_down_average_mills_needing_rectification_l765_765599


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765267

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765267


namespace parallelogram_angle_ratios_l765_765907

theorem parallelogram_angle_ratios (A B C D : ℝ):
  (∠ A ::: B ::: C ::: D).

end parallelogram_angle_ratios_l765_765907


namespace distance_between_midpoints_l765_765962

-- Define the vertices of the quadrilateral
variables {A B C D : Type}
variables (P : A → B → Type) (Q : A → C → Type) (R : B → D → Type) (S : C → D → Type)

-- Assume convex quadrilateral and midpoint properties
theorem distance_between_midpoints (h : convex_quadrilateral A B C D)
                                  (M N : Type) (hm : midpoint M A C) (hn : midpoint N B D)
                                  (hAD_gt_BC : length A D > length B C) :
  let MN := distance M N in
  MN ≥ |(length A D - length B C) / 2| :=
sorry

end distance_between_midpoints_l765_765962


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765333

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765333


namespace common_divisors_count_l765_765129

-- Given conditions
def num1 : ℕ := 9240
def num2 : ℕ := 8000

-- Prime factorizations from conditions
def fact_num1 : List ℕ := [2^3, 3^1, 5^1, 7^2]
def fact_num2 : List ℕ := [2^6, 5^3]

-- Computing gcd based on factorizations
def gcd : ℕ := 2^3 * 5^1

-- The goal is to prove the number of divisors of gcd is 8
theorem common_divisors_count : 
  ∃ d, d = (3+1)*(1+1) ∧ d = 8 := 
by
  sorry

end common_divisors_count_l765_765129


namespace nearest_integer_power_l765_765639

theorem nearest_integer_power (x : ℝ) (x_val : x = 3 + sqrt 5) : 
  (Real.round (x^4) : ℤ) = 376 :=
by
  sorry

end nearest_integer_power_l765_765639


namespace monotonic_increasing_function_is_f3_l765_765712

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

def f1 : ℝ → ℝ := λ x, -x
def f2 : ℝ → ℝ := λ x, Real.log x / Real.log 3
def f3 : ℝ → ℝ := λ x, x ^ (1 / 3 : ℝ)
def f4 : ℝ → ℝ := λ x, (1 / 2 : ℝ) ^ x

theorem monotonic_increasing_function_is_f3 :
  is_monotonically_increasing f3 ∧
  ¬ is_monotonically_increasing f1 ∧
  ¬ is_monotonically_increasing f2 ∧
  ¬ is_monotonically_increasing f4 :=
by sorry

end monotonic_increasing_function_is_f3_l765_765712


namespace polygon_interior_angle_eq_l765_765713

theorem polygon_interior_angle_eq (n : ℕ) (h : ∀ i, 1 ≤ i → i ≤ n → (interior_angle : ℝ) = 108) : n = 5 := 
sorry

end polygon_interior_angle_eq_l765_765713


namespace calculate_expression_l765_765726

variables (x y : ℝ)

theorem calculate_expression (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x - y) / (Real.sqrt x + Real.sqrt y) - (x - 2 * Real.sqrt (x * y) + y) / (Real.sqrt x - Real.sqrt y) = 0 :=
by
  sorry

end calculate_expression_l765_765726


namespace find_n_in_range_l765_765423

open Int

theorem find_n_in_range (a b : ℤ) (h1 : a ≡ 22 [MOD 36]) (h2 : b ≡ 85 [MOD 36]) :
∃ n ∈ (Set.Icc 120 161), (a - b) ≡ n [MOD 36] :=
by
  have h : (a - b) ≡ 153 [MOD 36] := by sorry
  use 153
  split
  · exact le_refl 153
  · apply le_of_lt
    norm_num
  assumption
  sorry

end find_n_in_range_l765_765423


namespace natural_number_pairs_int_l765_765775

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l765_765775


namespace cos_alpha_through_point_l765_765834

theorem cos_alpha_through_point (α : ℝ) (x y r : ℝ) (h₁ : x = -3) (h₂ : y = 4) (h₃ : r = real.sqrt (x^2 + y^2)) (h₄ : r = 5) :
  real.cos α = -3 / 5 := 
sorry

end cos_alpha_through_point_l765_765834


namespace max_xy_on_line_AB_l765_765048

theorem max_xy_on_line_AB : 
  let A := (3 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 4 : ℝ)
  ∃ P : ℝ × ℝ, P ∈ line_through A B ∧ xy_max P = 3 := 
sorry

end max_xy_on_line_AB_l765_765048


namespace find_functions_l765_765495

-- Define the set S
def S := {n : ℕ // n ≥ 0}

-- Define the functional equation condition
def satisfies_functional_eq (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (m + f n) = f (f m) + f n

-- Lean 4 Statement to prove the problem
theorem find_functions (f : ℕ → ℕ) (hf : ∀ n, f n ∈ S) :
  satisfies_functional_eq f → (∀ n, f n = 0 ∨ f n = n ∨ ∃ (n0 : ℕ) (ar : ℕ), 0 ≤ ar ∧ ar < n0 ∧ ∃ (K : ℕ), f n = ar * n0 + K * n0) :=
by
  sorry

end find_functions_l765_765495


namespace solve_inequality_system_l765_765560

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765560


namespace count_two_digit_primes_with_units_digit_3_l765_765160

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765160


namespace solve_inequality_system_l765_765586

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l765_765586


namespace decreased_expression_value_l765_765865

theorem decreased_expression_value (x y : ℝ) :
  let x' := 0.6 * x in
  let y' := 0.6 * y in
  (x' ^ 2) * (y' ^ 3) = 0.07776 * (x ^ 2 * y ^ 3) :=
by
  sorry

end decreased_expression_value_l765_765865


namespace count_integers_with_digit_sum_eq_18_l765_765112

theorem count_integers_with_digit_sum_eq_18 : 
  let nums_in_range := {n : ℕ | 400 ≤ n ∧ n ≤ 600}
  let digit_sum_eq (n : ℕ) := (n / 100) + ((n % 100) / 10) + (n % 10) = 18
  ∃ (count : ℕ), count = 21 ∧ count = (nums_in_range.count digit_sum_eq) := 
by 
  sorry

end count_integers_with_digit_sum_eq_18_l765_765112


namespace sum_of_intercepts_l765_765684

-- Define the point and slope
def point : (ℝ × ℝ) := (5, 3)
def slope : ℝ := 3

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := y = slope * x - (slope * (point.1) - point.2)

-- Define the intercepts
def x_intercept : ℝ := (point.2 + slope * point.1) / slope
def y_intercept : ℝ := - (slope * (point.1) - point.2)

-- Prove the sum of the intercepts is -8
theorem sum_of_intercepts : x_intercept + y_intercept = -8 := by
  sorry

end sum_of_intercepts_l765_765684


namespace car_n_graph_is_thrice_as_high_and_third_as_long_l765_765728

noncomputable def speed_time_graph (v t : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let car_m := (v, t)
  let car_n := (3 * v, t / 3)
  car_n

-- Theorem: Car N's speed-time graph will be thrice as high and a third as long as Car M's speed-time graph.
theorem car_n_graph_is_thrice_as_high_and_third_as_long (v t : ℝ) (h : 0 < t ∧ 0 < v) :
  let car_m_graph := (v, t)
  let car_n_graph := (3 * v, t / 3)
  car_n_graph = (3 * car_m_graph.1, car_m_graph.2 / 3) :=
by {
  sorry
}

end car_n_graph_is_thrice_as_high_and_third_as_long_l765_765728


namespace shaded_area_correct_l765_765896

noncomputable def grid_width : ℕ := 15
noncomputable def grid_height : ℕ := 5
noncomputable def triangle_base : ℕ := 15
noncomputable def triangle_height : ℕ := 3
noncomputable def total_area : ℝ := (grid_width * grid_height : ℝ)
noncomputable def triangle_area : ℝ := (1 / 2) * triangle_base * triangle_height
noncomputable def shaded_area : ℝ := total_area - triangle_area

theorem shaded_area_correct : shaded_area = 52.5 := 
by sorry

end shaded_area_correct_l765_765896


namespace houses_built_during_boom_l765_765722

theorem houses_built_during_boom :
  ∀ (before after : ℕ), before = 1426 → after = 2000 → after - before = 574 :=
by
  intros before after h_before h_after
  rw [h_before, h_after]
  norm_num
  sorry

end houses_built_during_boom_l765_765722


namespace area_S4_is_3_125_l765_765705

theorem area_S4_is_3_125 (S_1 : Type) (area_S1 : ℝ) 
  (hS1 : area_S1 = 25)
  (bisect_and_construct : ∀ (S : Type) (area : ℝ),
    ∃ S' : Type, ∃ area' : ℝ, area' = area / 2) :
  ∃ S_4 : Type, ∃ area_S4 : ℝ, area_S4 = 3.125 :=
by
  sorry

end area_S4_is_3_125_l765_765705


namespace min_value_of_expression_l765_765939

theorem min_value_of_expression :
  ∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ (x * y * z = 12) → 
  (2 * x + 3 * y + 6 * z ≥ 18 * real.cbrt 2) :=
by
  sorry

end min_value_of_expression_l765_765939


namespace num_two_digit_primes_with_units_digit_three_l765_765132

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765132


namespace find_final_coordinates_l765_765991

-- Define the transformations
def rotate_about_x (p : ℝ × ℝ × ℝ) := (p.1, -p.2, -p.3)
def reflect_xy (p : ℝ × ℝ × ℝ) := (p.1, p.2, -p.3)
def reflect_yz (p : ℝ × ℝ × ℝ) := (-p.1, p.2, p.3)

def final_point (p : ℝ × ℝ × ℝ) :=
  reflect_yz (rotate_about_x (reflect_yz (reflect_xy (rotate_about_x p))))

theorem find_final_coordinates :
  final_point (1, 1, 1) = (1, 1, -1) :=
by
  sorry

end find_final_coordinates_l765_765991


namespace problem_statement_l765_765871

noncomputable def n : ℝ := 2 ^ 0.15
noncomputable def b : ℝ := 39.99999999999998

theorem problem_statement : (n ^ b) = 64 :=
by
  sorry

end problem_statement_l765_765871


namespace cricketer_average_after_19_innings_l765_765855

theorem cricketer_average_after_19_innings (A : ℤ) (h1 : A = 21) : 
  (18 * A + 97) / 19 = A + 4 := 
by {
  have h2 : 18 * A + 97 = 19 * (A + 4),
  { 
    rw h1,
    linarith,
  },
  have h3 : (19 * (A + 4)) / 19 = (A + 4),
  { 
    rw [int.mul_div_cancel, int.coe_nat_succ, int.coe_nat_succ],
    norm_num,
    linarith,
  },
  rw h2,
  rw h3,
  norm_num,
  linarith,
}

end cricketer_average_after_19_innings_l765_765855


namespace find_d_minus_c_l765_765986

theorem find_d_minus_c (c d x : ℝ) (h : c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d) : (d - c = 45) :=
  sorry

end find_d_minus_c_l765_765986


namespace number_of_three_digit_numbers_l765_765527

theorem number_of_three_digit_numbers : 
  let A := {0, 2, 4}
  let B := {1, 3, 5}
  (finset.filter (λ n, n < 1000) ((finset.product A B).val.map (λ(xy) => let (x, y) := xy in x * 100 + (y.1.nat_abs.val * 10 + y.2.nat_abs.val))).val).card = 36 :=
by sorry

end number_of_three_digit_numbers_l765_765527


namespace solve_inequality_system_l765_765581

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l765_765581


namespace maxEdges_no_Kk_l765_765869

def maxEdges (n k : ℕ) : ℕ :=
  let r := n % (k-1)
  (k-2) * (n^2 - r^2) / (2 * (k-1)) + (r.choose 2)

theorem maxEdges_no_Kk (n k : ℕ) (G : SimpleGraph (Fin n)) (hK : ¬(G.isKSubgraph k)) :
  G.numEdges ≤ maxEdges n k := by
  sorry

end maxEdges_no_Kk_l765_765869


namespace solve_inequality_system_l765_765534

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765534


namespace total_weight_of_nuts_l765_765676

theorem total_weight_of_nuts:
  let almonds := 0.14
  let pecans := 0.38
  let walnuts := 0.22
  let cashews := 0.47
  let pistachios := 0.29
  almonds + pecans + walnuts + cashews + pistachios = 1.50 :=
by
  sorry

end total_weight_of_nuts_l765_765676


namespace average_water_per_day_l765_765509

-- Define the given conditions as variables/constants
def day1 := 318
def day2 := 312
def day3_morning := 180
def day3_afternoon := 162

-- Define the total water added on day 3
def day3 := day3_morning + day3_afternoon

-- Define the total water added over three days
def total_water := day1 + day2 + day3

-- Define the number of days
def days := 3

-- The proof statement: the average water added per day is 324 liters
theorem average_water_per_day : total_water / days = 324 :=
by
  -- Placeholder for the proof
  sorry

end average_water_per_day_l765_765509


namespace integer_roots_of_polynomial_l765_765778

theorem integer_roots_of_polynomial :
  {x : ℤ | x^3 - 4*x^2 - 14*x + 24 = 0} = {-4, -3, 3} := by
  sorry

end integer_roots_of_polynomial_l765_765778


namespace depth_of_box_l765_765673

theorem depth_of_box (length width depth : ℕ) (side_length : ℕ)
  (h_length : length = 30)
  (h_width : width = 48)
  (h_side_length : Nat.gcd length width = side_length)
  (h_cubes : side_length ^ 3 = 216)
  (h_volume : 80 * (side_length ^ 3) = length * width * depth) :
  depth = 12 :=
by
  sorry

end depth_of_box_l765_765673


namespace polygon_sides_l765_765693

theorem polygon_sides (n : ℕ) (D : ℕ) (hD : D = 77) (hFormula : D = n * (n - 3) / 2) (hVertex : n = n) : n + 1 = 15 :=
by
  sorry

end polygon_sides_l765_765693


namespace two_digit_primes_with_units_digit_three_count_l765_765172

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765172


namespace ellipse_standard_form_and_max_area_diff_l765_765042

theorem ellipse_standard_form_and_max_area_diff (a b x y k: ℝ) (hx : x = 1) (hy : y = 2) (hK: k ≠ 0) (hEcc : (1/2) = sqrt ((a^2 - b^2)/a^2)) (hEllCond : a > b ∧ b > 0) (hEll : x^2/a^2 + y^2/b^2 = 1) (hPar : y^2 = 9/4 * x):
    let ellipse_eq : Prop := (a = 2 ∧ c = 1 ∧ b^2 = 3 ∧ x^2/4 + y^2/3 = 1)
    let max_area_diff : Prop := (|((2 * |y1| - |y2|) - (2 * |y2| - |y1|))| = sqrt(3))
  in ellipse_eq ∧ max_area_diff :=
begin
  sorry
end

end ellipse_standard_form_and_max_area_diff_l765_765042


namespace verify_integer_pairs_l765_765758

open Nat

theorem verify_integer_pairs (a b : ℕ) :
  (∃ k1 : ℤ, ↑(a^2) + ↑b = k1 * (↑(b^2) - ↑a)) ∧
  (∃ k2 : ℤ, ↑(b^2) + ↑a = k2 * (↑(a^2) - ↑b)) →
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ 
  (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end verify_integer_pairs_l765_765758


namespace two_digit_primes_with_units_digit_three_count_l765_765178

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765178


namespace count_two_digit_primes_ending_in_3_l765_765371

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765371


namespace digit_150_division_l765_765629

theorem digit_150_division : (150thDigit (5 / 31)) = 5 := 
sorry

end digit_150_division_l765_765629


namespace number_of_two_digit_primes_with_units_digit_three_l765_765217

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765217


namespace distance_from_sphere_center_to_triangle_plane_l765_765704

-- Define the center of the sphere and its radius
def sphere_center : ℝ := 0  -- assume the center is at the origin for simplicity
def sphere_radius : ℝ := 8

-- Define the sides of the triangle
def side_a : ℝ := 17
def side_b : ℝ := 17
def side_c : ℝ := 30

-- Condition: the triangle is tangent to the sphere
-- Prove: the distance from the center of the sphere to the plane of the triangle

theorem distance_from_sphere_center_to_triangle_plane 
  (O : ℝ) (R : ℝ) (a b c : ℝ) (tangent : ∀ (P Q : ℝ), {x // x ∈ {a, b, c}}) :
  O = sphere_center → R = sphere_radius → a = side_a → b = side_b → c = side_c →
  ∃ d : ℝ, d = 7.067 := 
by
  intros hO hR ha hb hc
  use 7.067
  sorry

end distance_from_sphere_center_to_triangle_plane_l765_765704


namespace inequality_system_solution_l765_765548

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l765_765548


namespace jane_rejects_percent_l765_765923

theorem jane_rejects_percent :
  -- Declare the conditions as hypotheses
  ∀ (P : ℝ) (J : ℝ) (john_frac_reject : ℝ) (total_reject_percent : ℝ) (jane_inspect_frac : ℝ),
  john_frac_reject = 0.005 →
  total_reject_percent = 0.0075 →
  jane_inspect_frac = 5 / 6 →
  -- Given the rejection equation
  (john_frac_reject * (1 / 6) * P + (J / 100) * jane_inspect_frac * P = total_reject_percent * P) →
  -- Prove that Jane rejected 0.8% of the products she inspected
  J = 0.8 :=
by {
  sorry
}

end jane_rejects_percent_l765_765923


namespace count_two_digit_primes_ending_in_3_l765_765378

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765378


namespace circumsphere_radius_l765_765814

-- Definitions for the lengths of the sides of the right triangular prism.
variable (a b c : ℝ)

-- Condition: side lengths are pairwise perpendicular.
-- Prove that the radius of the circumsphere is given by the provided formula.

theorem circumsphere_radius (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  let diameter := (Real.sqrt (a^2 + b^2 + c^2)) in
  let R := diameter  / 2 in
  R = (Real.sqrt (a^2 + b^2 + c^2)) / 2 :=
sorry

end circumsphere_radius_l765_765814


namespace a4_eq_one_t2_eq_t5_if_a1_then_a2_one_l765_765063

variables {T : ℕ → ℝ} {a : ℕ → ℝ}

-- Condition: The product of the first n terms of a geometric sequence
def geo_seq_prod (n : ℕ) : ℝ := ∏ i in finset.range n, a i

-- Specific condition: third, fourth and fifth terms' product is 1
axiom geo_term_prod : a 3 * a 4 * a 5 = 1

-- Statements to be proven
theorem a4_eq_one (h : a 3 * a 4 * a 5 = 1) : a 4 = 1 :=
sorry

theorem t2_eq_t5 (h : a 3 * a 4 * a 5 = 1) : T 2 = T 5 :=
sorry

theorem if_a1_then_a2_one (h : a 1 = 1) (h : a 3 * a 4 * a 5 = 1) : a 2 = 1 :=
sorry

end a4_eq_one_t2_eq_t5_if_a1_then_a2_one_l765_765063


namespace solve_inequality_system_l765_765570

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765570


namespace distinctPermutations_test_l765_765096

noncomputable def distinctPermutationsCount (s : Multiset ℕ) : ℕ :=
  (s.card.factorial) / (s.count 3.factorial * s.count 8.factorial)

theorem distinctPermutations_test : distinctPermutationsCount {3, 3, 3, 8, 8} = 10 := by
  sorry

end distinctPermutations_test_l765_765096


namespace complex_number_quadrant_l765_765996

def z : ℂ := (↑complex.I) / (1 + ↑complex.I)

theorem complex_number_quadrant : z.re > 0 ∧ z.im > 0 := sorry

end complex_number_quadrant_l765_765996


namespace multiplication_digits_sum_l765_765913

theorem multiplication_digits_sum :
  let product := 19 * 12,
      digits := [2, 2, 8],
      sum_digits := List.sum digits
  in sum_digits = 12 :=
by
  sorry

end multiplication_digits_sum_l765_765913


namespace perpendicular_PA_BC_l765_765716

theorem perpendicular_PA_BC
  (A B C P E F G H O1 O2 : Type*)
  [IncidencePlane A B C E F G H O1 O2]
  (triangle_ABC : Triangle A B C)
  (tangent_E : Tangent O1 (Line A B))
  (tangent_F : Tangent O2 (Line A C))
  (tangent_G : Tangent O1 (Line B C))
  (tangent_H : Tangent O2 (Line B A))
  (intersect_P : ∃ P, Line E G ∩ Line F H = {P}) :
  IsPerpendicular (Line A P) (Line B C) :=
sorry

end perpendicular_PA_BC_l765_765716


namespace count_two_digit_prime_numbers_ending_in_3_l765_765245

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l765_765245


namespace minimal_perimeter_triangle_inscribed_at_point_minimal_perimeter_of_inscribed_triangle_l765_765658

universe u
variables {α : Type u}

-- Definitions and conditions for translating the problem
def is_triangle (A B C : α) : Prop := sorry -- Define conditions for points A, B, C to form a triangle
def is_point_on_segment (P A B : α) : Prop := sorry -- Define conditions for point P to be on segment AB

-- Problem part a)
theorem minimal_perimeter_triangle_inscribed_at_point (A B C P : α) 
  (hABC : is_triangle A B C) (hP_on_AB : is_point_on_segment P A B) : 
  ∃ (X Y : α), is_triangle P X Y ∧ 
  ∀ (P' X' Y' : α), is_triangle P' X' Y' → 
  perimeter (triangle P X Y) ≤ perimeter (triangle P' X' Y') :=
sorry

-- Problem part b)
theorem minimal_perimeter_of_inscribed_triangle (A B C : α) 
  (hABC : is_triangle A B C) : 
  ∃ (P X Y : α), is_point_on_segment P A B ∧ 
  is_triangle P X Y ∧ 
  ∀ (P' X' Y' : α), is_point_on_segment P' A B → is_triangle P' X' Y' → 
  perimeter (triangle P X Y) ≤ perimeter (triangle P' X' Y') :=
sorry

end minimal_perimeter_triangle_inscribed_at_point_minimal_perimeter_of_inscribed_triangle_l765_765658


namespace inequality_add_l765_765862

theorem inequality_add {a b c : ℝ} (h : a > b) : a + c > b + c :=
sorry

end inequality_add_l765_765862


namespace verify_integer_pairs_l765_765759

open Nat

theorem verify_integer_pairs (a b : ℕ) :
  (∃ k1 : ℤ, ↑(a^2) + ↑b = k1 * (↑(b^2) - ↑a)) ∧
  (∃ k2 : ℤ, ↑(b^2) + ↑a = k2 * (↑(a^2) - ↑b)) →
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ 
  (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end verify_integer_pairs_l765_765759


namespace simplify1_proof_simplify2_proof_simplify3_proof_simplify4_proof_l765_765971

noncomputable def simplify1 : α := 
  sqrt 8 + sqrt (1 / 3) - 2 * sqrt 2

noncomputable def simplify2 : α := 
  (π - 2015)^0 + sqrt 12 + abs (sqrt 3 - 2)

noncomputable def simplify3 : α := 
  (2 * sqrt 48 - 3 * sqrt 27) / sqrt 6

noncomputable def simplify4 : α := 
  abs (-sqrt 2) - 1 / sqrt 2 + (1 - sqrt 2)^2

theorem simplify1_proof :
  simplify1 = sqrt 3 / 3 := 
  sorry

theorem simplify2_proof :
  simplify2 = 3 + sqrt 3 := 
  sorry

theorem simplify3_proof :
  simplify3 = -sqrt 2 / 2 := 
  sorry

theorem simplify4_proof :
  simplify4 = 3 - 3 * sqrt 2 / 2 := 
  sorry

end simplify1_proof_simplify2_proof_simplify3_proof_simplify4_proof_l765_765971


namespace one_is_divisible_by_10_l765_765508

theorem one_is_divisible_by_10 (a1 a2 a3 a4 a5 : ℤ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ 
                                                            a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ 
                                                            a3 ≠ a4 ∧ a3 ≠ a5 ∧ a4 ≠ a5)
  (h_divisible : ∀ x y z : {n : ℤ // n = a1 ∨ n = a2 ∨ n = a3 ∨ n = a4 ∨ n = a5}, x ≠ y → y ≠ z → x ≠ z → 
                                          ((x * y * z) % 10) = 0) : 
  ∃ x : {n : ℤ // n = a1 ∨ n = a2 ∨ n = a3 ∨ n = a4 ∨ n = a5}, (x % 10) = 0 := 
sorry

end one_is_divisible_by_10_l765_765508


namespace count_integers_with_digit_sum_eq_18_l765_765113

theorem count_integers_with_digit_sum_eq_18 : 
  let nums_in_range := {n : ℕ | 400 ≤ n ∧ n ≤ 600}
  let digit_sum_eq (n : ℕ) := (n / 100) + ((n % 100) / 10) + (n % 10) = 18
  ∃ (count : ℕ), count = 21 ∧ count = (nums_in_range.count digit_sum_eq) := 
by 
  sorry

end count_integers_with_digit_sum_eq_18_l765_765113


namespace length_squared_n_equals_32_l765_765734

def p (x : ℝ) := -x + 1
def q (x : ℝ) := x + 1
def r : ℝ := 2
def n (x : ℝ) := min (min (p x) (q x)) r

theorem length_squared_n_equals_32 :
  let n_interval1 := (real.sqrt ((-1 - (-3))^2 + (p (-1) - p (-3))^2))
  let n_interval2 := (real.sqrt ((3 - 1)^2 + (q 3 - q 1)^2))
  (n_interval1 + n_interval2)^2 = 32 :=
by
  sorry

end length_squared_n_equals_32_l765_765734


namespace inequality_solution_l765_765547

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l765_765547


namespace find_a_from_binomial_l765_765747

variable (x : ℝ) (a : ℝ)

def binomial_term (r : ℕ) : ℝ :=
  (Nat.choose 5 r) * ((-a)^r) * x^(5 - 2 * r)

theorem find_a_from_binomial :
  (∃ x : ℝ, ∃ a : ℝ, (binomial_term x a 1 = 10)) → a = -2 :=
by 
  sorry

end find_a_from_binomial_l765_765747


namespace equal_focal_lengths_of_ellipses_l765_765072

theorem equal_focal_lengths_of_ellipses (k : ℝ) (hk : k < 16) :
  let a1 := 4
  let b1 := 5
  let c1 := Real.sqrt (b1^2 - a1^2)
  let a2 := Real.sqrt (16 - k)
  let b2 := Real.sqrt (25 - k)
  let c2 := Real.sqrt (b2^2 - a2^2)
  in 2 * c1 = 2 * c2 :=
by
  let a1 := 4
  let b1 := 5
  let c1 := real.sqrt ((b1 ^ 2) - (a1 ^ 2))
  let a2 := real.sqrt (16 - k)
  let b2 := real.sqrt (25 - k)
  let c2 := real.sqrt ((b2 ^ 2) - (a2 ^ 2))
  sorry

end equal_focal_lengths_of_ellipses_l765_765072


namespace beths_average_speed_l765_765662

theorem beths_average_speed (john_speed : ℝ) (john_time : ℝ) (beth_time : ℝ) (distance_diff : ℝ) : 
  john_speed = 40 → 
  john_time = 0.5 → 
  distance_diff = 5 → 
  beth_time = (30 + 20) / 60 → 
  let john_distance := john_speed * john_time in 
  let beth_distance := john_distance + distance_diff in
  beth_distance / beth_time = 30 := 
by
  intros h1 h2 h3 h4,
  let john_distance := john_speed * john_time,
  have hjohn_distance : john_distance = 20, 
  { rw [h1, h2], norm_num },
  let beth_distance := john_distance + distance_diff,
  have hbeth_distance : beth_distance = 25, 
  { rw [hjohn_distance, h3], norm_num },
  have hbeth_speed : beth_distance / beth_time = 30, 
  { rw [hbeth_distance, h4], norm_num },
  exact hbeth_speed

end beths_average_speed_l765_765662


namespace two_digit_primes_with_units_digit_three_l765_765229

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l765_765229


namespace rectangle_area_l765_765985

theorem rectangle_area (W : ℕ) (hW : W = 5) (L : ℕ) (hL : L = 4 * W) : ∃ (A : ℕ), A = L * W ∧ A = 100 := 
by
  use 100
  sorry

end rectangle_area_l765_765985


namespace count_two_digit_primes_with_units_digit_3_l765_765189

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765189


namespace rectangle_length_from_rhombus_l765_765614

theorem rectangle_length_from_rhombus (
  w : ℝ 
  (w = 20) 
  l : ℝ 
  (BF = DE : ℝ) 
  (BF = l / 2) 
  (Perimeter : ℝ)
  (Perimeter = 82)
  (AFCE_side : ℝ)
  (AFCE_side = Perimeter / 4)) :
  l = 35.79 :=
by 
  sorry

end rectangle_length_from_rhombus_l765_765614


namespace count_two_digit_primes_with_units_digit_three_l765_765416

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765416


namespace num_solutions_l765_765011

-- Let x be a real number
variable (x : ℝ)

-- Define the given equation
def equation := (x^2 - 4) * (x^2 - 1) = (x^2 + 3*x + 2) * (x^2 - 8*x + 7)

-- Theorem: The number of values of x that satisfy the equation is 3
theorem num_solutions : ∃ (S : Finset ℝ), (∀ x, x ∈ S ↔ equation x) ∧ S.card = 3 := 
by
  sorry

end num_solutions_l765_765011


namespace balance_test_l765_765796

variable (a b h c : ℕ)

theorem balance_test
  (h1 : 4 * a + 2 * b + h = 21 * c)
  (h2 : 2 * a = b + h + 5 * c) :
  b + 2 * h = 11 * c :=
sorry

end balance_test_l765_765796


namespace num_two_digit_primes_with_units_digit_three_l765_765146

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l765_765146


namespace count_two_digit_prime_numbers_with_units_digit_3_l765_765334

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l765_765334


namespace solve_inequality_system_l765_765532

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765532


namespace number_of_two_digit_primes_with_units_digit_three_l765_765215

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l765_765215


namespace arrangement_count_correct_l765_765457

-- Define a function to count the number of arrangements with given conditions
def numberOfArrangements : ℕ :=
  let mathEnds := 4 * 3
  let historyArr := 4.factorial
  let mathMiddle := 2.factorial
  mathEnds * historyArr * mathMiddle

-- The theorem to be proved
theorem arrangement_count_correct : numberOfArrangements = 576 :=
by
  -- the proof will go here
  sorry

end arrangement_count_correct_l765_765457


namespace smaller_circle_radius_in_trapezoid_l765_765484

theorem smaller_circle_radius_in_trapezoid 
  (A B C D : Point) 
  (AB : Line) 
  (BC : Line) 
  (CD : Line) 
  (DA : Line)
  (r1 r2 : ℝ)
  (radiusA radiusB radiusC radiusD : ℝ)
  (hAB : length AB = 8)
  (hBC : length BC = 7)
  (hDA : length DA = 7)
  (hCD : length CD = 6)
  (hRadiusA : radiusA = 4)
  (hRadiusB : radiusB = 4)
  (hRadiusC : radiusC = 3)
  (hRadiusD : radiusD = 3) :
  let r := (-88 + 56 * Real.sqrt 6) / 26 in
  ∃ O : Point, inside_trapezoid O ABCD ∧ touches_all_circles O radiusA radiusB radiusC radiusD r :=
sorry

end smaller_circle_radius_in_trapezoid_l765_765484


namespace two_digit_prime_numbers_with_units_digit_3_count_l765_765261

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l765_765261


namespace two_digit_primes_with_units_digit_three_count_l765_765169

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l765_765169


namespace double_angle_proof_l765_765736

noncomputable def double_angle_construction (A O B : Point) (hAOB : angle A O B) : Prop :=
∃ B' B'', reflect_ray_O_B_A O B B' B'' ∧
  measure_angle A O B'' = 2 * measure_angle A O B

/-- Given an angle AOB, with O as the apex, and points A and B on the rays of the angle,
  prove that the constructed angle AOB'' formed by reflection is double the original angle. -/
theorem double_angle_proof (A O B : Point) (hAOB : angle A O B) :
  double_angle_construction A O B hAOB :=
sorry

end double_angle_proof_l765_765736


namespace find_a_and_b_max_value_f_in_1_2_l765_765846

noncomputable def f (a b x : ℝ) : ℝ := log (a^x - b^x) / log 2

theorem find_a_and_b :
  (∃ a b : ℝ, f a b 1 = 1 ∧ f a b 2 = log 12 / log 2) ↔ (a = 4 ∧ b = 2) := 
sorry

theorem max_value_f_in_1_2 (a b : ℝ) (ha : a = 4) (hb : b = 2) :
  ∃ max_val : ℝ, max_val = log 12 / log 2 ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a b x ≤ max_val) :=
sorry

end find_a_and_b_max_value_f_in_1_2_l765_765846


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765290

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765290


namespace count_two_digit_primes_with_units_digit_three_l765_765411

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l765_765411


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765291

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765291


namespace digit_sum_400_to_600_equals_18_l765_765119

theorem digit_sum_400_to_600_equals_18 :
  card {n : ℕ | 400 ≤ n ∧ n ≤ 600 ∧ (n.digits 10).sum = 18} = 21 :=
sorry

end digit_sum_400_to_600_equals_18_l765_765119


namespace solve_inequality_system_l765_765565

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l765_765565


namespace proof_complement_U_A_l765_765087

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set A
def A : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := { x ∈ U | x ∉ A }

-- The theorem statement
theorem proof_complement_U_A :
  complement_U_A = {1, 5} :=
by
  -- Proof goes here
  sorry

end proof_complement_U_A_l765_765087


namespace natural_number_pairs_int_l765_765772

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l765_765772


namespace find_n_l765_765866

theorem find_n (n : ℤ) (h : 7^(2*n) = (1/7)^(n-18)) : n = 6 :=
by
  sorry

end find_n_l765_765866


namespace elena_savings_l765_765745

theorem elena_savings :
  let original_cost := 7 * 3
  let discount_rate := 0.25
  let rebate := 5
  let disc_amount := original_cost * discount_rate
  let price_after_discount := original_cost - disc_amount
  let final_price := price_after_discount - rebate
  original_cost - final_price = 10.25 :=
by
  sorry

end elena_savings_l765_765745


namespace construct_circle_feasible_l765_765977

theorem construct_circle_feasible (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : b^2 > (a^2 + c^2) / 2) :
  ∃ x y d : ℝ, 
  d > 0 ∧ 
  (d / 2)^2 = y^2 + (a / 2)^2 ∧ 
  (d / 2)^2 = (y - x)^2 + (b / 2)^2 ∧ 
  (d / 2)^2 = (y - 2 * x)^2 + (c / 2)^2 :=
sorry

end construct_circle_feasible_l765_765977


namespace distinct_four_digit_even_numbers_l765_765111

theorem distinct_four_digit_even_numbers :
  ∃ n : ℤ, n = 10 ∧
  (∀ (d1 d2 d3 d4 : ℕ), {d1, d2, d3, d4} ⊆ {0, 1, 2, 3} ∧ 10 ≤ 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧
   (1000 * d1 + 100 * d2 + 10 * d3 + d4) % 2 = 0 ∧ d1 ≠ 0 ∧ d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ 
   d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 → true) :=
begin
  sorry
end

end distinct_four_digit_even_numbers_l765_765111


namespace cannot_form_complex_pattern_l765_765702

structure GeometricPieces where
  triangles : Nat
  squares : Nat

def possibleToForm (pieces : GeometricPieces) : Bool :=
  sorry -- Since the formation logic is unknown, it is incomplete.

theorem cannot_form_complex_pattern : 
  let pieces := GeometricPieces.mk 8 7
  ¬ possibleToForm pieces = true := 
sorry

end cannot_form_complex_pattern_l765_765702


namespace count_two_digit_primes_with_units_digit_3_l765_765310

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l765_765310


namespace total_water_filled_jars_l765_765476

theorem total_water_filled_jars :
  ∃ x : ℕ, 
    16 * (1/4) + 12 * (1/2) + 8 * 1 + 4 * 2 + x * 3 = 56 ∧
    16 + 12 + 8 + 4 + x = 50 :=
by
  sorry

end total_water_filled_jars_l765_765476


namespace exists_polyhedron_with_given_vertices_and_edges_l765_765744

theorem exists_polyhedron_with_given_vertices_and_edges :
  ∃ (V : Finset (String)) (E : Finset (Finset (String))),
    V = { "A", "B", "C", "D", "E", "F", "G", "H" } ∧
    E = { { "A", "B" }, { "A", "C" }, { "A", "H" }, { "B", "C" },
          { "B", "D" }, { "C", "D" }, { "D", "E" }, { "E", "F" },
          { "E", "G" }, { "F", "G" }, { "F", "H" }, { "G", "H" } } ∧
    (V.card : ℤ) - (E.card : ℤ) + 6 = 2 :=
by
  sorry

end exists_polyhedron_with_given_vertices_and_edges_l765_765744


namespace points_divisibility_l765_765481

theorem points_divisibility {k n : ℕ} (hkn : k ≤ n) (hpositive : 0 < n) 
  (hcondition : ∀ x : Fin n, (∃ m : ℕ, (∀ y : Fin n, x.val ≤ y.val → y.val ≤ x.val + 1 → True) ∧ m % k = 0)) :
  k ∣ n :=
sorry

end points_divisibility_l765_765481


namespace compound_interest_correct_l765_765431

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100) ^ T - P

theorem compound_interest_correct (SI R T : ℝ) (hSI : SI = 2100) (hR : R = 10) (hT : T = 7) :
  let P := SI / (R * T / 100)
  let CI := P * (1 + R / 100) ^ T - P
  CI ≈ 2846.15 :=
by
  have hP : P = 2100 / (10 * 7 / 100) := sorry
  have hCI : CI = 3000 * (1 + 10 / 100) ^ 7 - 3000 := sorry
  have hVal : (1.1)^7 ≈ 1.9487171 := sorry
  show CI ≈ 2846.15 from sorry

end compound_interest_correct_l765_765431


namespace gcd_372_684_is_12_l765_765608

theorem gcd_372_684_is_12 : gcd 372 684 = 12 := by
  sorry

end gcd_372_684_is_12_l765_765608


namespace figure_perimeter_l765_765982

theorem figure_perimeter (a b : ℕ) (h₁ : a = 20) (h₂ : b = 5) : 4 * a + 4 * (3 * b) = 140 :=
by
  rw [h₁, h₂]
  sorry

end figure_perimeter_l765_765982


namespace count_prime_units_digit_3_eq_6_l765_765307

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765307


namespace count_prime_units_digit_3_eq_6_l765_765309

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l765_765309


namespace symmetric_function_a_plus_omega_l765_765077

def f (ω a x : ℝ) : ℝ := sin (ω * x) + a * cos (ω * x)

theorem symmetric_function_a_plus_omega 
  (ω a : ℝ) (ω_gt_zero : ω > 0)
  (h_symm_m : ∀ x, f ω a (π/3 - x) = -f ω a (π/3 + x))
  (h_minimum : ∀ x, f ω a (π/6 - x) ≥ f ω a (π/6)) :
  a + ω = 3 :=
by
  sorry

end symmetric_function_a_plus_omega_l765_765077


namespace min_product_of_positive_numbers_l765_765638

theorem min_product_of_positive_numbers {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : a * b = a + b) : a * b = 4 :=
sorry

end min_product_of_positive_numbers_l765_765638


namespace sum_not_divisible_by_n_plus_2_l765_765961

theorem sum_not_divisible_by_n_plus_2 (n : ℕ) : ¬ ((1 + 2 + ... + n) % (n + 2) = 0) :=
sorry

end sum_not_divisible_by_n_plus_2_l765_765961


namespace count_two_digit_primes_ending_in_3_l765_765370

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765370


namespace henry_collected_points_l765_765467

def points_from_wins (wins : ℕ) : ℕ := wins * 5
def points_from_losses (losses : ℕ) : ℕ := losses * 2
def points_from_draws (draws : ℕ) : ℕ := draws * 3

def total_points (wins losses draws : ℕ) : ℕ := 
  points_from_wins wins + points_from_losses losses + points_from_draws draws

theorem henry_collected_points :
  total_points 2 2 10 = 44 := by
  -- The proof will go here
  sorry

end henry_collected_points_l765_765467


namespace arithmetic_sequence_general_term_l765_765928

noncomputable def a_n (n : ℕ) : ℚ := (2 * n - 1) / 4

theorem arithmetic_sequence_general_term
  (a : ℕ → ℚ)
  (d : ℚ)
  (h_seq_pos: ∀ n, a n > 0)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (S_n : ℕ → ℚ)
  (h_sum : ∀ n, S_n n = n * a 1 + (n * (n - 1) / 2) * d)
  (sqrt_S_n_arith : ∀ n, sqrt (S_n n) = sqrt (S_n 1) + (n - 1) * d)
  : ∀ n, a n = (2 * n - 1) / 4 := 
sorry

end arithmetic_sequence_general_term_l765_765928


namespace main_theorem_l765_765472

-- Definitions based on the given conditions
def Point := ℝ × ℝ
def Angle (A B C : Point) : ℝ := sorry -- placeholder for angle calculation

-- Given conditions
variables (A B C O : Point)
variables (h1 : Angle A B O = Angle C A O)
variables (h2 : Angle B A O = Angle B C O)
variables (h3 : Angle B O C = 90)

-- Goal: To prove the ratio AC : OC is √2
def ratio_AC_OC (A B C O : Point) 
  (h1 : Angle A B O = Angle C A O) 
  (h2 : Angle B A O = Angle B C O) 
  (h3 : Angle B O C = 90) : ℝ :=
  AC / OC

-- Main theorem statement
theorem main_theorem 
  (A B C O : Point) 
  (h1 : Angle A B O = Angle C A O)
  (h2 : Angle B A O = Angle B C O)
  (h3 : Angle B O C = 90) : 
  ratio_AC_OC A B C O h1 h2 h3 = sqrt 2 := 
by
  sorry

end main_theorem_l765_765472


namespace digit_150_division_l765_765630

theorem digit_150_division : (150thDigit (5 / 31)) = 5 := 
sorry

end digit_150_division_l765_765630


namespace count_two_digit_primes_ending_in_3_l765_765377

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l765_765377


namespace ce_eq_df_l765_765485

variables (C₁ C₂ : set point) (A B C D E F : point)
hypothesis h1 : C₁ ∩ C₂ = {A, B}
hypothesis h2 : C ∈ C₁
hypothesis h3 : D ∈ C₂
hypothesis h4 : line_through A B bisects ∠CAD
hypothesis h5 : E ∈ line_segment A C ∧ E ∈ C₂
hypothesis h6 : F ∈ line_segment A D ∧ F ∈ C₁

theorem ce_eq_df : dist C E = dist D F :=
sorry

end ce_eq_df_l765_765485


namespace constant_term_in_binomial_expansion_l765_765830

noncomputable def integral_value : ℝ := ∫ x in (0 : ℝ)..(real.pi / 2), 4 * real.cos x

theorem constant_term_in_binomial_expansion :
  (∃ n : ℕ, n = real.to_nnreal integral_value) →
  ∃ (r : ℕ), r = n / 2 ∧
  (binomial n r = 6) :=
by
  sorry

end constant_term_in_binomial_expansion_l765_765830


namespace count_two_digit_primes_with_units_digit_3_l765_765163

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l765_765163


namespace num_two_digit_prime_with_units_digit_3_eq_6_l765_765279

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l765_765279


namespace verify_integer_pairs_l765_765757

open Nat

theorem verify_integer_pairs (a b : ℕ) :
  (∃ k1 : ℤ, ↑(a^2) + ↑b = k1 * (↑(b^2) - ↑a)) ∧
  (∃ k2 : ℤ, ↑(b^2) + ↑a = k2 * (↑(a^2) - ↑b)) →
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ 
  (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end verify_integer_pairs_l765_765757
