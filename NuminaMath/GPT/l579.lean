import Mathlib
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trigonometry.Basic
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Perm
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Factorization
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic.Basic
import Mathlib.Topology.EuclideanSpace.Basic

namespace partial_fraction_product_zero_l579_579861

theorem partial_fraction_product_zero
  (A B C : ℚ)
  (partial_fraction_eq : ∀ x : ℚ,
    x^2 - 25 = A * (x + 3) * (x - 5) + B * (x - 3) * (x - 5) + C * (x - 3) * (x + 3))
  (fact_3 : C = 0)
  (fact_neg3 : B = 1/3)
  (fact_5 : A = 0) :
  A * B * C = 0 := 
sorry

end partial_fraction_product_zero_l579_579861


namespace projection_of_2m_plus_n_on_m_l579_579873

Variables (λ : ℝ)

def m : ℝ × ℝ := (-1, 2)
def n : ℝ × ℝ := (2, λ)

lemma perp_vectors (h : m.1 * n.1 + m.2 * n.2 = 0) : 
  λ = 1 :=
by linarith

def projection_vector 
  (h : n = (2, 1)) : ℝ × ℝ :=
let two_m_n := (2 * m.1 + n.1, 2 * m.2 + n.2) in
let proj := (two_m_n.1 * m.1 + two_m_n.2 * m.2) / (m.1 * m.1 + m.2 * m.2) in
(proj * m.1, proj * m.2)

theorem projection_of_2m_plus_n_on_m :
  (λ = 1) → projection_vector (by rfl) = (-2, 4) :=
by {
  intro hλ,
  simp [projection_vector, hλ],
  -- Computing intermediary steps
  have two_m_n := (2 * m.1 + n.1, 2 * m.2 + n.2),
  have proj := (two_m_n.1 * m.1 + two_m_n.2 * m.2) / (m.1 * m.1 + m.2 * m.2),
  -- Prove that proj = 2, hence
  have h_proj : proj = 2, sorry,
  simp [two_m_n, proj, h_proj],
}

end projection_of_2m_plus_n_on_m_l579_579873


namespace permutations_banana_l579_579078

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579078


namespace BANANA_arrangements_correct_l579_579453

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579453


namespace BANANA_permutation_l579_579562

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579562


namespace BANANA_arrangements_l579_579029

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579029


namespace white_balls_in_bag_l579_579988

   theorem white_balls_in_bag (m : ℕ) (h : m ≤ 7) :
     (2 * (m * (m - 1) / 2) / (7 * 6 / 2)) + ((m * (7 - m)) / (7 * 6 / 2)) = 6 / 7 → m = 3 :=
   by
     intros h_eq
     sorry
   
end white_balls_in_bag_l579_579988


namespace lunks_needed_for_apples_l579_579881

-- Definitions of conversion rates
def lunks_to_kunks (lunks : ℕ) : ℕ := (lunks * 3) / 5
def kunks_to_apples (kunks : ℕ) : ℕ := (kunks * 4) / 2

-- Definition of the problem
theorem lunks_needed_for_apples (n : ℕ) (lunks_to_kunks : ℕ → ℕ) (kunks_to_apples : ℕ → ℕ) : 
  (lunks_to_kunks (n * 5) = 3 * n) →
  (kunks_to_apples (n * 2) = 4 * n) →
  ∃ lunks, kunks_to_apples (6) = 12 ∧ lunks_to_kunks (10) = 6 → n = 1 :=
by
  sorry

end lunks_needed_for_apples_l579_579881


namespace karl_sticker_count_l579_579937

theorem karl_sticker_count : 
  ∀ (K R B : ℕ), 
    (R = K + 20) → 
    (B = R - 10) → 
    (K + R + B = 105) → 
    K = 25 := 
by
  intros K R B hR hB hSum
  sorry

end karl_sticker_count_l579_579937


namespace BANANA_permutations_l579_579229

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579229


namespace arrange_BANANA_l579_579271

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579271


namespace permutations_of_BANANA_l579_579422

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579422


namespace arrange_BANANA_l579_579712

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579712


namespace num_ways_to_arrange_BANANA_l579_579508

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579508


namespace BANANA_permutations_l579_579203

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579203


namespace banana_permutations_l579_579491

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579491


namespace permutations_of_BANANA_l579_579393

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579393


namespace banana_permutations_l579_579465

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579465


namespace permutations_BANANA_l579_579169

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579169


namespace banana_permutations_l579_579472

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579472


namespace permutations_banana_l579_579093

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579093


namespace distance_between_intersections_is_sqrt_10_l579_579859

noncomputable def l : ℝ → ℝ → Prop := λ x y, 3 * x + y - 6 = 0
noncomputable def C : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2 * y - 4 = 0

theorem distance_between_intersections_is_sqrt_10 :
  ∃ A B : ℝ × ℝ,
    (l A.1 A.2 ∧ C A.1 A.2) ∧
    (l B.1 B.2 ∧ C B.1 B.2) ∧
    (A ≠ B ∧ (real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = real.sqrt 10)) :=
sorry

end distance_between_intersections_is_sqrt_10_l579_579859


namespace BANANA_arrangements_correct_l579_579431

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579431


namespace product_of_divisors_36_l579_579768

-- Problem statement: Prove that the product of the divisors of 36 is 10077696
theorem product_of_divisors_36 : 
  let n := 36 in
  let p := 2 in
  let q := 3 in
  n = p * p * q * q →
  (∏ d in (Multiset.to_finset (Multiset.divisors n)).val, d) = 10077696 :=
by
  sorry

end product_of_divisors_36_l579_579768


namespace banana_permutations_l579_579477

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579477


namespace discount_percentage_is_correct_l579_579979

def price_of_quiche := 15.0
def price_of_croissant := 3.0
def price_of_biscuit := 2.0

def quantity_of_quiche := 2
def quantity_of_croissant := 6
def quantity_of_biscuit := 6

def discounted_price := 54.0

def original_price := (quantity_of_quiche * price_of_quiche) +
                      (quantity_of_croissant * price_of_croissant) +
                      (quantity_of_biscuit * price_of_biscuit)

def discount_amount := original_price - discounted_price

def discount_percentage := (discount_amount / original_price) * 100

theorem discount_percentage_is_correct :
  discount_percentage = 10 := by
  sorry

end discount_percentage_is_correct_l579_579979


namespace banana_permutations_l579_579487

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579487


namespace banana_arrangements_l579_579060

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579060


namespace num_ways_to_arrange_BANANA_l579_579516

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579516


namespace num_ways_to_arrange_BANANA_l579_579539

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579539


namespace permutations_banana_l579_579104

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579104


namespace BANANA_permutation_l579_579564

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579564


namespace arrange_BANANA_l579_579626

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579626


namespace inverse_of_g_l579_579886

theorem inverse_of_g (x : ℝ) (h : g x = 87) : x = 3 :=
by
  -- Define the function g
  let g := fun x => 3 * x^3 + 6
  -- The given condition
  have eq_g : g x = 87 := h
  -- The rest of the proof is trivial and will be skipped
  sorry

end inverse_of_g_l579_579886


namespace remaining_lawn_fraction_l579_579959

theorem remaining_lawn_fraction (mary_time : ℝ) (tom_time : ℝ) (mary_works_hours : ℝ) :
  mary_time = 3 → tom_time = 6 → mary_works_hours = 1 → (1 - mary_works_hours / mary_time) = 2 / 3 :=
by
  intros h_mary_time h_tom_time h_mary_works_hours
  rw [h_mary_time, h_mary_works_hours]
  norm_num
  sorry

end remaining_lawn_fraction_l579_579959


namespace arrange_BANANA_l579_579707

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579707


namespace num_ways_to_arrange_BANANA_l579_579507

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579507


namespace banana_arrangements_l579_579119

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579119


namespace banana_permutations_l579_579353

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579353


namespace problem1_decreasing_increasing_intervals_problem2_minimum_value_of_a_l579_579850

def f (a x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x
def g (x : ℝ) : ℝ := x * Real.exp (1 - x)

theorem problem1_decreasing_increasing_intervals (x : ℝ) : 
  ∀ x ∈ (set.Ioo 0 2), (∀ (a : ℝ), a = 1 → (∃ x, (x ∈ set.Ioo 0 2) → deriv (f a) x < 0)) ∧
  ∀ x ∈ (set.Ioo 2 (⊤ : ℝ)), (∀ (a : ℝ), a = 1 → (∃ x, (x ∈ set.Ioo 2 ⊤) → deriv (f a) x > 0)) := sorry

theorem problem2_minimum_value_of_a : 
  (∀ a ∈ {a : ℝ | a > 2 - 4 * Real.log 2}, ∀ x ∈ (set.Ioo 0 (1 / 2 : ℝ)), f a x > 0) :=
sorry

end problem1_decreasing_increasing_intervals_problem2_minimum_value_of_a_l579_579850


namespace permutations_of_BANANA_l579_579398

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579398


namespace permutations_banana_l579_579110

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579110


namespace compare_polynomials_l579_579783

variable (x : ℝ)
variable (h : x > 1)

theorem compare_polynomials (h : x > 1) : x^3 + 6 * x > x^2 + 6 := 
by
  sorry

end compare_polynomials_l579_579783


namespace BANANA_permutations_l579_579585

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579585


namespace min_value_expression_l579_579787

theorem min_value_expression :
  (∀ y : ℝ, abs y ≤ 1 → ∃ x : ℝ, 2 * x + y = 1 ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1 → 
    (∃ y : ℝ, 2 * x + y = 1 ∧ abs y ≤ 1 ∧ (2 * x ^ 2 + 16 * x + 3 * y ^ 2) = 3))) :=
sorry

end min_value_expression_l579_579787


namespace banana_arrangements_l579_579312

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579312


namespace BANANA_permutations_l579_579224

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579224


namespace find_distance_PO_l579_579802

noncomputable def ellipse_and_foci (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 6 = 1) ∧ 
  (∃ (F1 F2 : ℝ × ℝ), 
     let c := sqrt (3) in
     F1 = (c, 0) ∧ F2 = (-c, 0) ∧
     ∃ (P : ℝ × ℝ), 
       (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
       (cos (angle_at P F1 F2) = 3 / 5) ∧ 
       (∃ (O : ℝ × ℝ), O = (0, 0)))

noncomputable def distance_PO (x y : ℝ) : ℝ :=
  sqrt ((x)^2 + (y)^2)

theorem find_distance_PO : ∃ (x y : ℝ), ellipse_and_foci x y → distance_PO x y = sqrt (30) / 2 :=
sorry

end find_distance_PO_l579_579802


namespace BANANA_permutations_l579_579196

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579196


namespace banana_arrangements_l579_579319

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579319


namespace arrange_BANANA_l579_579624

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579624


namespace number_of_repeating_decimals_l579_579878

open Nat

theorem number_of_repeating_decimals :
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 15) → (¬ ∃ k : ℕ, k * 18 = n) :=
by
  intros n h
  sorry

end number_of_repeating_decimals_l579_579878


namespace BANANA_permutation_l579_579548

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579548


namespace greatest_radius_l579_579895

theorem greatest_radius (r : ℕ) (h : π * (r : ℝ)^2 < 50 * π) : r = 7 :=
sorry

end greatest_radius_l579_579895


namespace no_repetition_of_initial_set_l579_579941

theorem no_repetition_of_initial_set 
  (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_not_all_one : ¬(a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1))
  : ¬∃ n : ℕ, generate_set_sequence n (a, b, c, d) = (a, b, c, d) :=
sorry

-- We also need to define generate_set_sequence which generates the set (ab, bc, cd, da) recursively.
noncomputable def generate_set_sequence : ℕ → (ℝ × ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ × ℝ)
| 0 (a, b, c, d) := (a, b, c, d)
| (n + 1) (a, b, c, d) := generate_set_sequence n (a*b, b*c, c*d, d*a)

end no_repetition_of_initial_set_l579_579941


namespace arrange_BANANA_l579_579306

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579306


namespace weight_difference_proof_l579_579911

theorem weight_difference_proof
  (labrador_start_weight : ℕ) (dachshund_start_weight : ℕ)
  (weight_gain_percentage : ℕ)
  (labrador_start_weight = 40)
  (dachshund_start_weight = 12)
  (weight_gain_percentage = 25) :
  (labrador_start_weight + labrador_start_weight * weight_gain_percentage / 100) -
  (dachshund_start_weight + dachshund_start_weight * weight_gain_percentage / 100) =
  35 := 
  sorry

end weight_difference_proof_l579_579911


namespace banana_arrangements_l579_579140

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579140


namespace banana_arrangements_l579_579056

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579056


namespace permutations_of_BANANA_l579_579416

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579416


namespace workers_in_first_group_l579_579892

theorem workers_in_first_group (W : ℕ) :
  (∃ (w1 w2 : ℕ), w1 = 3 ∧ w2 = 7 ∧ W * w1 = 30 * w2) → W = 70 :=
by
  intro h
  cases h with w1 h1
  cases h1 with w2 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  sorry

end workers_in_first_group_l579_579892


namespace number_of_arrangements_banana_l579_579696

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579696


namespace arrangement_count_BANANA_l579_579266

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579266


namespace banana_permutations_l579_579359

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579359


namespace num_ways_to_arrange_BANANA_l579_579540

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579540


namespace BANANA_arrangements_l579_579020

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579020


namespace arrange_BANANA_l579_579287

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579287


namespace BANANA_arrangements_correct_l579_579434

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579434


namespace number_of_arrangements_banana_l579_579684

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579684


namespace arrangement_count_BANANA_l579_579252

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579252


namespace centers_distance_ABC_l579_579926

-- Define triangle ABC with the given properties
structure RightTriangle (ABC : Type) :=
(angle_A : ℝ)
(angle_C : ℝ)
(shorter_leg : ℝ)

-- Given: angle A is 30 degrees, angle C is 90 degrees, and shorter leg AC is 1
def triangle_ABC : RightTriangle ℝ := {
  angle_A := 30,
  angle_C := 90,
  shorter_leg := 1
}

-- Define the distance between the centers of the inscribed circles of triangles ACD and BCD
noncomputable def distance_between_centers (ABC : RightTriangle ℝ): ℝ :=
  sorry  -- placeholder for the actual proof

-- Example problem statement
theorem centers_distance_ABC (ABC : RightTriangle ℝ) (h_ABC : ABC = triangle_ABC) :
  distance_between_centers ABC = (Real.sqrt 3 - 1) / Real.sqrt 2 :=
sorry

end centers_distance_ABC_l579_579926


namespace banana_arrangements_l579_579318

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579318


namespace BANANA_permutation_l579_579560

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579560


namespace number_of_arrangements_banana_l579_579682

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579682


namespace arrange_BANANA_l579_579649

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579649


namespace min_max_values_of_f_l579_579780

noncomputable def f (x : ℝ) : ℝ := x^2 + 4*x + 3

def g (t : ℝ) : ℝ :=
  if t < -3 then t^2 + 6*t + 8
  else if t > -2 then t^2 + 4*t + 3
  else if -2.5 < t ∧ t < -2 then -1
  else if -3 < t ∧ t <= -2.5 then -1
  else 0  -- default value to handle all cases

def h (t : ℝ) : ℝ :=
  if t < -3 then t^2 + 4*t + 3
  else if t > -2 then t^2 + 6*t + 8
  else if -2.5 < t ∧ t < -2 then t^2 + 6*t + 8
  else if -3 < t ∧ t <= -2.5 then t^2 + 4*t + 3
  else 0  -- default value to handle all cases

-- Theorem statement without proof
theorem min_max_values_of_f (t : ℝ) :
  (g t = if t < -3 then t^2 + 6*t + 8
         else if t > -2 then t^2 + 4*t + 3
         else if -2.5 < t ∧ t < -2 then -1
         else if -3 < t ∧ t <= -2.5 then -1
         else 0)
  ∧
  (h t = if t < -3 then t^2 + 4*t + 3
         else if t > -2 then t^2 + 6*t + 8
         else if -2.5 < t ∧ t < -2 then t^2 + 6*t + 8
         else if -3 < t ∧ t <= -2.5 then t^2 + 4*t + 3
         else 0) := sorry

end min_max_values_of_f_l579_579780


namespace banana_arrangements_l579_579332

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579332


namespace cone_volume_case1_cone_volume_case2_l579_579791

-- Define the lengths of the sides of the right-angled triangle
def side1 : ℝ := 6
def side2 : ℝ := 8
def hypotenuse : ℝ := Real.sqrt (side1 ^ 2 + side2 ^ 2)

-- Define the radius of the cone's base as half the hypotenuse
def radius : ℝ := hypotenuse / 2

-- Define the volume of the cone formula
def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r * r * h

-- Define the volumes resulting from each rotation case
def volume_case1 : ℝ := cone_volume radius side1
def volume_case2 : ℝ := cone_volume radius side2

-- Lean statement to be proved
theorem cone_volume_case1 : volume_case1 = 50 * Real.pi := by sorry
theorem cone_volume_case2 : volume_case2 = (200/3) * Real.pi := by sorry

end cone_volume_case1_cone_volume_case2_l579_579791


namespace distance_from_origin_ellipse_point_l579_579795

theorem distance_from_origin_ellipse_point :
  ∃ P : ℝ × ℝ, 
    (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
    (let c := real.sqrt(3) in 
    let F1 := (-c, 0) in 
    let F2 := (c, 0) in 
    real.cos (
      vector.angle 
       ⟨P.1 - F1.1, P.2 - F1.2⟩ 
       ⟨P.1 - F2.1, P.2 - F2.2⟩
    ) = 3/5) ∧ 
  real.sqrt (P.1^2 + P.2^2) = real.sqrt(30) / 2 := 
sorry

end distance_from_origin_ellipse_point_l579_579795


namespace gasoline_price_change_l579_579913

theorem gasoline_price_change (P0 : ℝ) (P1 P2 P3 P4 : ℝ) (y : ℝ):
  P0 = 100 → 
  P1 = P0 * 1.25 → 
  P2 = P1 * 0.75 → 
  P3 = P2 * 1.30 →
  P4 = P3 * (1 - y / 100) →
  P4 = P0 →
  y = 18 :=
by
  intro hP0 hP1 hP2 hP3 hP4 hPfinal
  rw [hP0, hP4, hP0, hP1, hP2, hP3] at hPfinal
  -- We have:
  -- 100 * 1.25 * 0.75 * 1.30 * (1 - y / 100) = 100
  -- Simplify the LHS:
  -- 121.875 * (1 - y / 100) = 100
  -- Solve for y:
  -- 121.875 * (1 - y / 100) = 100
  -- 1 - y / 100 = 100 / 121.875
  -- y / 100 = 21.875 / 121.875
  -- y = 21.875 * 100 / 121.875 = 17.95 (approximately)
  -- Nearest integer to 17.95 is 18
  sorry

end gasoline_price_change_l579_579913


namespace banana_arrangements_l579_579070

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579070


namespace find_a_plus_b_l579_579887

theorem find_a_plus_b (x a b : ℝ) (hx : x = a + real.sqrt b)
  (h : x ^ 2 + 5 * x + 5 / x + 1 / x ^ 2 = 40)
  (ha : a ∈ ℤ)
  (hb : b ∈ ℤ)
  (ha_pos : a > 0)
  (hb_pos : b > 0) :
  a + b = 11 :=
sorry

end find_a_plus_b_l579_579887


namespace banana_permutations_l579_579500

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579500


namespace arrange_BANANA_l579_579719

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579719


namespace banana_arrangements_l579_579131

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579131


namespace BANANA_permutations_l579_579204

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579204


namespace find_distance_PO_l579_579799

noncomputable def ellipse_and_foci (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 6 = 1) ∧ 
  (∃ (F1 F2 : ℝ × ℝ), 
     let c := sqrt (3) in
     F1 = (c, 0) ∧ F2 = (-c, 0) ∧
     ∃ (P : ℝ × ℝ), 
       (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
       (cos (angle_at P F1 F2) = 3 / 5) ∧ 
       (∃ (O : ℝ × ℝ), O = (0, 0)))

noncomputable def distance_PO (x y : ℝ) : ℝ :=
  sqrt ((x)^2 + (y)^2)

theorem find_distance_PO : ∃ (x y : ℝ), ellipse_and_foci x y → distance_PO x y = sqrt (30) / 2 :=
sorry

end find_distance_PO_l579_579799


namespace find_intersection_l579_579845

variable (A : Set ℝ)
variable (B : Set ℝ := {1, 2})
variable (f : ℝ → ℝ := λ x => x^2)

theorem find_intersection (h : ∀ x, x ∈ A → f x ∈ B) : A ∩ B = ∅ ∨ A ∩ B = {1} :=
by
  sorry

end find_intersection_l579_579845


namespace BANANA_permutation_l579_579550

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579550


namespace banana_arrangements_l579_579043

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579043


namespace arrangement_count_BANANA_l579_579242

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579242


namespace banana_permutations_l579_579498

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579498


namespace parallel_condition_l579_579782

-- Define the conditions in Lean
variables (l m : Line) (α : Plane)

-- Condition: l and m are two different lines
hypothesis h_diff : l ≠ m

-- Condition: l parallel m
hypothesis h_parallel : l ∥ m

-- Question: Is l parallel α a condition for m parallel α?
theorem parallel_condition (h_l_parallel_α : l ∥ α) :
  ¬(l ∥ α ↔ m ∥ α) :=
sorry

end parallel_condition_l579_579782


namespace no_positive_integer_solutions_l579_579757

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), (a > 0) ∧ (b > 0) → 3 * a^2 ≠ b^2 + 1 :=
by
  sorry

end no_positive_integer_solutions_l579_579757


namespace identical_line_pairs_count_l579_579947

theorem identical_line_pairs_count :
  {n : ℕ // ∀ (b c : ℝ), (2 * x + 3 * b * y + c = 0 ∧ c * x + 4 * y + 16 = 0)
  → n = 2} :=
begin
  sorry
end

end identical_line_pairs_count_l579_579947


namespace fraction_to_decimal_terminating_l579_579745

theorem fraction_to_decimal_terminating : 
  (47 / (2^3 * 5^4) : ℝ) = 0.5875 :=
by 
  sorry

end fraction_to_decimal_terminating_l579_579745


namespace BANANA_arrangements_l579_579004

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579004


namespace arrange_BANANA_l579_579705

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579705


namespace BANANA_permutations_l579_579217

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579217


namespace BANANA_arrangements_correct_l579_579425

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579425


namespace BANANA_permutations_l579_579211

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579211


namespace find_quadratic_and_min_value_l579_579790

-- Define the given conditions
variables (b c : ℝ)

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 + b * x + c

-- Define the conditions that the function passes through points (0, 3) and (1, 0)
axiom pass_y_axis : quadratic b c 0 = 3
axiom pass_x_axis : quadratic b c 1 = 0

-- Prove the analytical expression and minimum value
theorem find_quadratic_and_min_value
  (hb : b = -4)
  (hc : c = 3)
  : quadratic b c = λ x, x^2 - 4 * x + 3
  ∧ ∃ x_min, (quadratic b c) (2 : ℝ) = -1 := by
  sorry

end find_quadratic_and_min_value_l579_579790


namespace proof_correct_answer_l579_579785
open Complex

noncomputable def z : ℂ := (1 + I) / (2 - I)

theorem proof_correct_answer :
  (conj z * z = (2 : ℝ) / 5) :=
by
  sorry

end proof_correct_answer_l579_579785


namespace arrange_BANANA_l579_579292

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579292


namespace maximum_lambda_correct_l579_579765

noncomputable def maximum_lambda (a b c : ℝ) : ℝ :=
  let f := λ x : ℝ, x^3 + a*x^2 + b*x + c
  let λ := -1 / 27
  if (∀ x ≥ 0, f x ≥ λ * (x - a) ^ 3) 
     then λ 
     else 0  -- or some other appropriate value to indicate invalidity

theorem maximum_lambda_correct {a b c : ℝ} (h : ∀ x ≥ 0, x^3 + a*x^2 + b*x + c = (x - 1)*(x - 2)*(x - 3)) :
  maximum_lambda a b c = -1 / 27 :=
sorry

end maximum_lambda_correct_l579_579765


namespace banana_permutations_l579_579488

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579488


namespace PO_equals_l579_579807

def ellipse (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 6 = 1

def F1_c (a : ℝ) : ℝ := √(a^2 - 6)
def F2_c (a : ℝ) : ℝ := √(a^2 - 6)

def cos_angle_F1PF2 : ℝ := 3 / 5

noncomputable def PO := (P : ℝ × ℝ) := 
  1 / 2 * ((P.1 - F1_c 3)^2 + (P.2 - F2_c 3)^2 - 2 * (P.1 - F1_c 3) * (P.2 - F2_c 3) * cos_angle_F1PF2)

theorem PO_equals : ∀ (P : ℝ × ℝ), ellipse P.1 P.2 → ∥P∥ = √(30) / 2 :=
by
  intros P P_ellipse_condition
  sorry

end PO_equals_l579_579807


namespace banana_arrangements_l579_579037

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579037


namespace banana_arrangements_l579_579067

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579067


namespace num_ways_to_arrange_BANANA_l579_579518

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579518


namespace banana_arrangements_l579_579048

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579048


namespace BANANA_permutations_l579_579195

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579195


namespace permutations_of_BANANA_l579_579397

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579397


namespace arrange_BANANA_l579_579723

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579723


namespace arrange_BANANA_l579_579636

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579636


namespace BANANA_permutation_l579_579542

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579542


namespace number_of_arrangements_banana_l579_579667

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579667


namespace subset_A_range_m_disjoint_A_B_range_m_l579_579867

noncomputable def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
noncomputable def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem subset_A_range_m (m : ℝ) :
  (∀ x, B m x → A x) ↔ m ∈ set.Iic 3 := sorry

theorem disjoint_A_B_range_m (m : ℝ) :
  (∀ x, ¬ (A x ∧ B m x)) ↔ m < 2 ∨ 4 < m := sorry

end subset_A_range_m_disjoint_A_B_range_m_l579_579867


namespace permutations_of_BANANA_l579_579413

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579413


namespace banana_permutations_l579_579354

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579354


namespace arrange_BANANA_l579_579634

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579634


namespace banana_arrangements_l579_579343

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579343


namespace BANANA_permutations_l579_579214

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579214


namespace number_of_arrangements_banana_l579_579677

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579677


namespace number_of_arrangements_banana_l579_579660

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579660


namespace permutations_of_BANANA_l579_579396

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579396


namespace permutations_of_BANANA_l579_579407

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579407


namespace rectangle_area_l579_579999

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 :=
by
  sorry

end rectangle_area_l579_579999


namespace permutations_banana_l579_579087

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579087


namespace arrange_BANANA_l579_579305

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579305


namespace permutations_of_BANANA_l579_579403

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579403


namespace min_edges_for_mono_triangle_l579_579786

open Finset

-- Define the notion of a monochromatic triangle for a graph coloring.
def monochromatic_triangle {V : Type} (edges : Finset (V × V)) (color : V × V → ℕ) :=
  ∃ v1 v2 v3 : V, (v1, v2) ∈ edges ∧ (v2, v3) ∈ edges ∧ (v1, v3) ∈ edges ∧ 
  color (v1, v2) = color (v2, v3) ∧ color (v1, v3) = color (v2, v3)

-- The main theorem.
theorem min_edges_for_mono_triangle (V : Type) [Fintype V] (hV : Fintype.card V = 9) (hNoCoplanar : ∀ (s : Finset V), s.card = 4 → ¬ ∃ (f : V → ℝ × ℝ × ℝ), 
  ∀ v ∈ s, (f v).2 = (f (s 0)).2) :
  ∃ edges : Finset (V × V), (edges.card = 33) ∧ ∀ color : (V × V) → ℕ, ∃ (mono : monochromatic_triangle edges color) :=
by
  sorry

end min_edges_for_mono_triangle_l579_579786


namespace permutations_banana_l579_579097

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579097


namespace BANANA_arrangements_l579_579017

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579017


namespace number_of_arrangements_banana_l579_579680

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579680


namespace arrange_BANANA_l579_579296

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579296


namespace permutations_BANANA_l579_579188

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579188


namespace smallest_positive_period_l579_579738

theorem smallest_positive_period : (∀ x, f (x + π) = f x)
  (f (x : ℝ) : ℝ := 2 * sin (2 * x - real.pi / 3)) :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) := 
sorry

end smallest_positive_period_l579_579738


namespace largest_divisor_of_n4_minus_n_l579_579773

theorem largest_divisor_of_n4_minus_n (n : ℕ) (h : ¬(Prime n) ∧ n ≠ 1) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_of_n4_minus_n_l579_579773


namespace BANANA_permutation_l579_579570

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579570


namespace permutations_of_BANANA_l579_579401

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579401


namespace banana_arrangements_l579_579126

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579126


namespace arrange_BANANA_l579_579635

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579635


namespace arrange_BANANA_l579_579702

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579702


namespace banana_arrangements_l579_579122

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579122


namespace BANANA_permutation_l579_579566

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579566


namespace BANANA_permutations_l579_579207

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579207


namespace problem_statement_l579_579838

theorem problem_statement :
  (∀ (E : Type) (a b : ℝ) (x y : ℝ), 
     center E = (0, 0) ∧ 
     axes_symmetry E (x, y) ∧ 
     passes_through E (0, -2) (3/2, -1) →
     equation E = (x^2 / 3 + y^2 / 4 = 1) ∧
     (∀ (line : Type) (P M N H : Type)
       (fixed_point : (ℝ × ℝ)), 
       fixed_point = (0, -2) →
       passes_through P (1, -2) ∧
       intersects E line M N ∧
       passes_through_line HN H N →
       passes_through fixed_point HN) :=
sorry

end problem_statement_l579_579838


namespace mapping_is_projective_l579_579788

-- Define lines a and b as sets of points (here we use arbitrary types for simplicity)
noncomputable def line (α : Type) := Set α

-- Define a mapping f from line a to line b
variable {α β : Type}
variable (a : line α) (b : line β)
variable (f : α → β)

-- Define the cross-ratio preservation condition
def preserves_cross_ratio (f : α → β) :=
  ∀ (A B C D : α), cross_ratio (f A) (f B) (f C) (f D) = cross_ratio A B C D

-- Define the cross-ratio function (placeholder for simplicity)
noncomputable def cross_ratio : α → α → α → α → α := sorry

-- Define the projective property (placeholder for simplicity)
def projective (f : α → β) := sorry

-- The statement to be proved
theorem mapping_is_projective (h : preserves_cross_ratio f) : projective f :=
sorry

end mapping_is_projective_l579_579788


namespace complex_number_solution_l579_579897

theorem complex_number_solution (z : ℂ) (h : z = complex.I * (2 + z)) : z = -1 + complex.I :=
by
  sorry

end complex_number_solution_l579_579897


namespace num_ways_to_arrange_BANANA_l579_579522

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579522


namespace BANANA_permutation_l579_579572

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579572


namespace permutations_BANANA_l579_579177

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579177


namespace banana_permutations_l579_579384

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579384


namespace arrange_BANANA_l579_579630

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579630


namespace BANANA_permutations_l579_579616

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579616


namespace banana_permutations_l579_579349

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579349


namespace banana_arrangements_l579_579041

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579041


namespace banana_arrangements_l579_579135

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579135


namespace min_distance_l579_579956

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance :
  ∃ m : ℝ, (∀ x > 0, x ≠ m → (f m - g m) ≤ (f x - g x)) ∧ m = Real.sqrt 2 / 2 :=
by
  sorry

end min_distance_l579_579956


namespace solve_for_ab_l579_579952

variable {a b : ℝ}

noncomputable def condition (a b : ℝ) : Prop :=
  ∀ x ∈ set.Icc 0 1, abs (a * x + b - real.sqrt (1 - x ^ 2)) <= (real.sqrt 2 - 1) / 2

theorem solve_for_ab : condition 0 (3 / 2) := 
by {
  intro x,
  intro h,
  sorry
}

end solve_for_ab_l579_579952


namespace BANANA_arrangements_correct_l579_579456

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579456


namespace arrangement_count_BANANA_l579_579262

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579262


namespace num_ways_to_arrange_BANANA_l579_579509

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579509


namespace banana_arrangements_l579_579329

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579329


namespace rectangle_area_l579_579998

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 :=
by
  sorry

end rectangle_area_l579_579998


namespace BANANA_permutation_l579_579554

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579554


namespace problem_l579_579921

-- Lean definition for given conditions and proof goals
def ellipse (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def passes_through (x0 y0 a b : ℝ) : Prop :=
  (3 : ℝ)^2 / a^2 + (-1 : ℝ)^2 / b^2 = 1

def eccentricity (a b e : ℝ) : Prop :=
  (a^2 - b^2) / a^2 = e^2

def semi_latus_rectum (a b : ℝ) : ℝ :=
  (a^2 - b^2) / a

def fixed_point (a b x0 y0 p : ℝ × ℝ) : Prop :=
  x0 = -2 * real.sqrt 2 → 
  ∀ y0, y0 ∈ set.Ioo (-2 * real.sqrt 3 / 3) (2 * real.sqrt 3 / 3) →
  ∃ a b,
    ∀ y0, line_through_perpendicular (x0, y0) (a, b) = p

theorem problem (a b x0 y0 : ℝ) (e := real.sqrt 6 / 3) :
  a > b → b > 0 → 
  ellipse a b x0 y0 → 
  eccentricity a b e → 
  passes_through x0 y0 a b →
  semi_latus_rectum a b = (2 * real.sqrt 6 / 3) ∧
  fixed_point a b (left := ℝ) (-4 * real.sqrt 2 / 3, 0)
:= sorry

end problem_l579_579921


namespace BANANA_permutations_l579_579604

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579604


namespace BANANA_arrangements_correct_l579_579429

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579429


namespace banana_arrangements_l579_579141

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579141


namespace arrange_BANANA_l579_579622

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579622


namespace arrangement_count_BANANA_l579_579260

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579260


namespace BANANA_permutation_l579_579573

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579573


namespace train_cross_bridge_time_l579_579874

noncomputable def train_length := 180 -- meters
noncomputable def train_speed_kmph := 54 -- kmph
noncomputable def bridge_length := 660 -- meters

def kmph_to_mps (speed_kmph : ℕ) : ℕ := (speed_kmph * 1000) / 3600

theorem train_cross_bridge_time :
  let total_distance := train_length + bridge_length,
      speed_mps := kmph_to_mps train_speed_kmph
  in total_distance / speed_mps = 56 := 
by
  sorry

end train_cross_bridge_time_l579_579874


namespace BANANA_arrangements_l579_579011

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579011


namespace arrange_BANANA_l579_579643

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579643


namespace num_ways_to_arrange_BANANA_l579_579537

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579537


namespace BANANA_permutations_l579_579202

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579202


namespace banana_arrangements_l579_579035

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579035


namespace arrangement_count_BANANA_l579_579235

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579235


namespace number_of_arrangements_banana_l579_579659

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579659


namespace permutations_BANANA_l579_579166

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579166


namespace permutations_banana_l579_579081

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579081


namespace PO_equals_l579_579804

def ellipse (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 6 = 1

def F1_c (a : ℝ) : ℝ := √(a^2 - 6)
def F2_c (a : ℝ) : ℝ := √(a^2 - 6)

def cos_angle_F1PF2 : ℝ := 3 / 5

noncomputable def PO := (P : ℝ × ℝ) := 
  1 / 2 * ((P.1 - F1_c 3)^2 + (P.2 - F2_c 3)^2 - 2 * (P.1 - F1_c 3) * (P.2 - F2_c 3) * cos_angle_F1PF2)

theorem PO_equals : ∀ (P : ℝ × ℝ), ellipse P.1 P.2 → ∥P∥ = √(30) / 2 :=
by
  intros P P_ellipse_condition
  sorry

end PO_equals_l579_579804


namespace banana_arrangements_l579_579130

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579130


namespace BANANA_arrangements_l579_579024

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579024


namespace arrange_BANANA_l579_579299

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579299


namespace permutations_banana_l579_579109

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579109


namespace banana_arrangements_l579_579114

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579114


namespace permutations_banana_l579_579089

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579089


namespace fraction_to_terminating_decimal_l579_579748

theorem fraction_to_terminating_decimal :
  (47 / (2^3 * 5^4) : ℝ) = 0.0094 := by
  sorry

end fraction_to_terminating_decimal_l579_579748


namespace solve_for_a_l579_579994

theorem solve_for_a (a : ℕ) (h : a > 0) (eqn : a / (a + 37) = 925 / 1000) : a = 455 :=
sorry

end solve_for_a_l579_579994


namespace banana_permutations_l579_579351

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579351


namespace sandy_correct_sums_l579_579975

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 45) : c = 21 :=
  sorry

end sandy_correct_sums_l579_579975


namespace num_correct_statements_l579_579828

def arithmetic_sequence := {a : ℕ → ℝ // ∃ d : ℝ, ∃ a₁ : ℝ, ∀ n, a (n + 1) = a n + d}

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := (n * (a 0 + a (n - 1))) / 2

variables (d : ℝ) (a_n : ℕ → ℝ) (S : ℕ → ℝ)
variable (criteria : arithmetic_sequence)
noncomputable def S_series := sum_of_first_n_terms (subtype.val criteria)

axiom condition1 : S 2015 > S 2016
axiom condition2 : S 2016 > S 2014

theorem num_correct_statements (n : ℕ) :
  (d < 0 ∧ S (4029) > 0 ∧ S (4030) > 0 ∧ |a_n 2015| > |a_n 2016| ∧ 1 = n) := 
  sorry

end num_correct_statements_l579_579828


namespace BANANA_arrangements_correct_l579_579447

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579447


namespace arrange_BANANA_l579_579736

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579736


namespace arrange_BANANA_l579_579273

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579273


namespace arrangement_count_BANANA_l579_579249

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579249


namespace num_ways_to_arrange_BANANA_l579_579506

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579506


namespace simplify_fraction_126_11088_l579_579981

theorem simplify_fraction_126_11088 : (126 / 11088 : ℚ) = 1 / 88 := 
by {
  /- 
  Given:
  1. 126 can be factored as 2 * 3^2 * 7.
  2. 11088 can be factored as 2^4 * 3^2 * 7 * 11.
  3. The greatest common divisor (GCD) of 126 and 11088 is 126.
  Simplify the fraction.
  -/
  sorry 
}

end simplify_fraction_126_11088_l579_579981


namespace arrange_BANANA_l579_579275

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579275


namespace number_of_arrangements_banana_l579_579697

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579697


namespace triangle_BC_length_l579_579928

theorem triangle_BC_length
  (A B C : Type)
  (B_right : is_right_triangle B)
  (sin_B : ∀ (BC AB : ℝ), sin B = (AB / BC) → sin B = 4/5)
  (AC : ℝ)
  (AC_length : AC = 3): 
  ∃ (BC : ℝ), BC = 5 :=
by
  sorry

end triangle_BC_length_l579_579928


namespace BANANA_arrangements_correct_l579_579461

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579461


namespace banana_permutations_l579_579376

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579376


namespace drinking_water_and_vegetables_transportation_plans_minimum_cost_l579_579939

theorem drinking_water_and_vegetables (total : ℕ) (excess : ℕ) (water : ℕ) (vegetables : ℕ) 
  (h1 : total = 320) (h2 : excess = 80) (h3 : water + vegetables = total) (h4 : water = vegetables + excess) : 
  water = 200 ∧ vegetables = 120 :=
by
  sorry

theorem transportation_plans (water vegetables : ℕ) (type_A type_B : ℕ)
  (h1 : water = 200) (h2 : vegetables = 120) (h3 : type_A + type_B = 8)
  (h4 : type_A * 40 + type_B * 20 = water) (h5 : type_A * 10 + type_B * 20 = vegetables) :
  (type_A = 2 ∧ type_B = 6) ∨ (type_A = 3 ∧ type_B = 5) ∨ (type_A = 4 ∧ type_B = 4) :=
by
  sorry

theorem minimum_cost (type_A type_B : ℕ)
  (h1 : (type_A = 2 ∧ type_B = 6) ∨ (type_A = 3 ∧ type_B = 5) ∨ (type_A = 4 ∧ type_B = 4))
  (cost_A cost_B : nat)
  (hc_A : cost_A = 400) (hc_B : cost_B = 360) :
  ∃ min_cost,
    min_cost = min ((2 * cost_A) + (6 * cost_B)) (min ((3 * cost_A) + (5 * cost_B)) ((4 * cost_A) + (4 * cost_B)))
    ∧ min_cost = 2960 :=
by
  sorry

end drinking_water_and_vegetables_transportation_plans_minimum_cost_l579_579939


namespace BANANA_permutation_l579_579571

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579571


namespace banana_permutations_l579_579484

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579484


namespace BANANA_permutations_l579_579605

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579605


namespace permutations_of_BANANA_l579_579410

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579410


namespace arrange_BANANA_l579_579274

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579274


namespace permutations_of_BANANA_l579_579418

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579418


namespace number_of_arrangements_banana_l579_579675

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579675


namespace permutations_banana_l579_579098

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579098


namespace BANANA_permutations_l579_579198

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579198


namespace banana_arrangements_l579_579040

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579040


namespace banana_arrangements_l579_579113

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579113


namespace BANANA_arrangements_correct_l579_579437

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579437


namespace extremum_at_neg1_increasing_on_interval_l579_579851

-- Define the function
def f (a x : ℝ) : ℝ := a * x^2 + 2 * log (1 - x)

-- Part (1): Prove the value of a when f(x) has an extremum at x = -1
theorem extremum_at_neg1 (a : ℝ) 
  (h_deriv : ∀ x, f' a x = 2 * a * x - 2 / (1 - x)) 
  (h_extremum : f' a (-1) = 0) : a = -1/2 := by
sorry

-- Part (2): Prove the range of a when f(x) is increasing on [-3, -2]
theorem increasing_on_interval (a : ℝ)
  (h_deriv : ∀ x, f' a x = 2 * a * x - 2 / (1 - x))
  (h_increasing : ∀ x ∈ Icc (-3 : ℝ) (-2), f' a x ≥ 0) : a ≤ -1/6 := by
sorry

end extremum_at_neg1_increasing_on_interval_l579_579851


namespace BANANA_arrangements_l579_579032

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579032


namespace find_theta_l579_579761

theorem find_theta (theta : ℝ) (h : cos (5 * (π / 180)) = sin (25 * (π / 180)) + sin (theta * (π / 180))) : 
  theta = 35 ∨ theta = 360*n + 35 ∨ theta = -360*n + 35 :=
begin
  sorry
end

end find_theta_l579_579761


namespace weight_difference_end_of_year_l579_579903

theorem weight_difference_end_of_year :
  let labrador_starting_weight := 40
  let dachshund_starting_weight := 12
  let weight_gain_percent := 0.25
  let labrador_end_weight := labrador_starting_weight + labrador_starting_weight * weight_gain_percent
  let dachshund_end_weight := dachshund_starting_weight + dachshund_starting_weight * weight_gain_percent
  (labrador_end_weight - dachshund_end_weight) = 35 :=
by
  let labrador_starting_weight := 40
  let dachshund_starting_weight := 12
  let weight_gain_percent := 0.25
  let labrador_end_weight := labrador_starting_weight + labrador_starting_weight * weight_gain_percent
  let dachshund_end_weight := dachshund_starting_weight + dachshund_starting_weight * weight_gain_percent
  -- Statement: difference in weight at end of year is 35 pounds
  have h : (labrador_end_weight - dachshund_end_weight) = 35 := sorry
  exact h

end weight_difference_end_of_year_l579_579903


namespace arrange_BANANA_l579_579629

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579629


namespace BANANA_permutations_l579_579599

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579599


namespace banana_arrangements_l579_579129

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579129


namespace arrangement_count_BANANA_l579_579230

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579230


namespace BANANA_arrangements_correct_l579_579430

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579430


namespace BANANA_arrangements_correct_l579_579433

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579433


namespace BANANA_permutations_l579_579581

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579581


namespace find_distance_PO_l579_579803

noncomputable def ellipse_and_foci (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 6 = 1) ∧ 
  (∃ (F1 F2 : ℝ × ℝ), 
     let c := sqrt (3) in
     F1 = (c, 0) ∧ F2 = (-c, 0) ∧
     ∃ (P : ℝ × ℝ), 
       (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
       (cos (angle_at P F1 F2) = 3 / 5) ∧ 
       (∃ (O : ℝ × ℝ), O = (0, 0)))

noncomputable def distance_PO (x y : ℝ) : ℝ :=
  sqrt ((x)^2 + (y)^2)

theorem find_distance_PO : ∃ (x y : ℝ), ellipse_and_foci x y → distance_PO x y = sqrt (30) / 2 :=
sorry

end find_distance_PO_l579_579803


namespace Sara_pears_left_l579_579977

def Sara_has_left (initial_pears : ℕ) (given_to_Dan : ℕ) (given_to_Monica : ℕ) (given_to_Jenny : ℕ) : ℕ :=
  initial_pears - given_to_Dan - given_to_Monica - given_to_Jenny

theorem Sara_pears_left :
  Sara_has_left 35 28 4 1 = 2 :=
by
  sorry

end Sara_pears_left_l579_579977


namespace number_of_arrangements_banana_l579_579665

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579665


namespace arrange_BANANA_l579_579728

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579728


namespace banana_arrangements_l579_579147

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579147


namespace complex_division_l579_579846

theorem complex_division (z1 z2 : ℂ) (h1 : z1 = 1 + 1 * Complex.I) (h2 : z2 = 0 + 2 * Complex.I) :
  z2 / z1 = 1 + Complex.I :=
by
  sorry

end complex_division_l579_579846


namespace banana_arrangements_l579_579127

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579127


namespace BANANA_permutations_l579_579613

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579613


namespace coefficient_of_x_squared_in_derivative_l579_579849

noncomputable def f (x : ℝ) : ℝ :=
  (1 - 2 * x) ^ 10

theorem coefficient_of_x_squared_in_derivative :
  let f' (x : ℝ) := deriv f x
  (term_coefficient (polynomialExpansion f' 9) 2 = -2880) :=
  sorry

-- term_coefficient and polynomialExpansion are placeholders for 
-- functions handling appropriate operations.

end coefficient_of_x_squared_in_derivative_l579_579849


namespace permutations_banana_l579_579085

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579085


namespace banana_permutations_l579_579355

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579355


namespace banana_permutations_l579_579370

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579370


namespace BANANA_permutation_l579_579547

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579547


namespace permutations_of_BANANA_l579_579399

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579399


namespace banana_permutations_l579_579470

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579470


namespace point_distance_on_ellipse_l579_579812

noncomputable theory
open Real

def ellipse (x y : ℝ) := x^2 / 9 + y^2 / 6 = 1

def foci_distance (a b : ℝ) := 2 * sqrt (a^2 - b^2)

def conditions (P : ℝ × ℝ) (cos_theta : ℝ) : Prop :=
  ellipse P.1 P.2 ∧ cos_theta = 3 / 5

theorem point_distance_on_ellipse (P : ℝ × ℝ) (cos_theta : ℝ)
  (h : conditions P cos_theta) : Real.sqrt (P.1^2 + P.2^2) = sqrt(30) / 2 :=
begin
  sorry,
end

end point_distance_on_ellipse_l579_579812


namespace permutations_BANANA_l579_579158

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579158


namespace arrangement_count_BANANA_l579_579241

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579241


namespace banana_arrangements_l579_579331

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579331


namespace arrange_BANANA_l579_579641

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579641


namespace BANANA_arrangements_correct_l579_579438

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579438


namespace BANANA_arrangements_l579_579034

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579034


namespace permutations_banana_l579_579074

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579074


namespace banana_arrangements_l579_579132

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579132


namespace BANANA_arrangements_correct_l579_579449

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579449


namespace degrees_of_interior_angles_l579_579820

-- Definitions for the problem conditions
variables {a b c h_a h_b S : ℝ} 
variables (ABC : Triangle) 
variables (height_to_bc height_to_ac : ℝ)
variables (le_a_ha : a ≤ height_to_bc)
variables (le_b_hb : b ≤ height_to_ac)
variables (area : S = 1 / 2 * a * height_to_bc)
variables (area_eq : S = 1 / 2 * b * height_to_ac)
variables (ha_eq : height_to_bc = 2 * S / a)
variables (hb_eq : height_to_ac = 2 * S / b)
variables (height_pos : 0 < 2 * S)
variables (length_pos : 0 < a ∧ 0 < b ∧ 0 < c)

-- Conclude the degrees of the interior angles
theorem degrees_of_interior_angles : 
  ∃ A B C : ℝ, A = 45 ∧ B = 45 ∧ C = 90 :=
sorry

end degrees_of_interior_angles_l579_579820


namespace BANANA_permutations_l579_579584

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579584


namespace BANANA_permutations_l579_579593

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579593


namespace arrange_BANANA_l579_579651

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579651


namespace projection_vector_l579_579870

def vect_m : ℝ × ℝ := (-1, 2)
def vect_n (λ : ℝ) : ℝ × ℝ := (2, λ)

theorem projection_vector:
  ∀ (λ : ℝ), (vect_m.1 * vect_n(λ).1 + vect_m.2 * vect_n(λ).2 = 0) → 
  (λ = 1) → 
  ((2 * vect_m.1 + vect_n(λ).1, 2 * vect_m.2 + vect_n(λ).2) : ℝ × ℝ) = (0, 5) → 
  ((2 * vect_m.1 + vect_n(λ).1, 2 * vect_m.2 + vect_n(λ).2) = (0, 5)) →
  let projection := ((0 * vect_m.1 + 5 * vect_m.2) / (vect_m.1^2 + vect_m.2^2)) * vect_m in
  projection = (-2, 4) :=
by
  intros
  sorry

end projection_vector_l579_579870


namespace permutations_BANANA_l579_579168

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579168


namespace BANANA_permutations_l579_579209

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579209


namespace banana_permutations_l579_579497

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579497


namespace banana_arrangements_l579_579064

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579064


namespace BANANA_permutations_l579_579193

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579193


namespace BANANA_permutations_l579_579583

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579583


namespace sin_y_minus_cos2_x_range_l579_579884

-- Define the conditions for the theorem.
theorem sin_y_minus_cos2_x_range (x y : ℝ) 
  (h : real.sin x + real.sin y = 1 / 3) : 
  -11 / 12 ≤ real.sin y - (real.cos x)^2 ∧ real.sin y - (real.cos x)^2 ≤ 4 / 9 :=
sorry

end sin_y_minus_cos2_x_range_l579_579884


namespace permutations_of_BANANA_l579_579400

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579400


namespace permutations_BANANA_l579_579185

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579185


namespace BANANA_arrangements_correct_l579_579439

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579439


namespace BANANA_arrangements_correct_l579_579441

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579441


namespace banana_permutations_l579_579469

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579469


namespace arrange_BANANA_l579_579701

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579701


namespace banana_arrangements_l579_579124

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579124


namespace num_ways_to_arrange_BANANA_l579_579513

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579513


namespace banana_permutations_l579_579501

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579501


namespace arrange_BANANA_l579_579703

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579703


namespace total_peaches_l579_579972

-- Definitions based on the given conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- The proof goal stating the total number of peaches now
theorem total_peaches : initial_peaches + picked_peaches = 68 := by
  sorry

end total_peaches_l579_579972


namespace permutations_of_BANANA_l579_579392

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579392


namespace arrange_BANANA_l579_579304

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579304


namespace BANANA_permutations_l579_579617

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579617


namespace banana_arrangements_l579_579333

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579333


namespace isosceles_triangle_angle_l579_579950

-- Define the basic conditions and properties of the problem
variable (A B C M D : Type) [Isosceles A B C (angle BAC:Real)]
variable (x : Real) (M : Midpoint B C)
variable (D : Reflection M AC)

theorem isosceles_triangle_angle (A B C M D : Type) [Isosceles A B C (angle BAC:Real)]
  (M : Midpoint B C) (D : Reflection M AC) (x : Real) : 
  angle MDC = x / 2 := 
sorry

end isosceles_triangle_angle_l579_579950


namespace correct_propositions_l579_579945

variables {l m : Type*} [line l] [line m]
variables {α : Type*} [plane α]

-- Definitions for the conditions of perpendicularity and parallelism with respect to a plane
def perp_to_plane (l : Type*) (α : Type*) [line l] [plane α] : Prop := sorry
def parallel_to_plane (l : Type*) (α : Type*) [line l] [plane α] : Prop := sorry
def sub_plane (m : Type*) (α : Type*) [line m] [plane α] : Prop := sorry

-- Definitions for parallelism and perpendicularity between lines
def parallel_lines (l m : Type*) [line l] [line m] : Prop := sorry
def perp_lines (l m : Type*) [line l] [line m] : Prop := sorry

theorem correct_propositions :
  (perp_to_plane l α ∧ perp_to_plane m α → parallel_lines l m) ∧
  (perp_to_plane l α ∧ sub_plane m α → perp_lines l m) :=
by
  sorry

end correct_propositions_l579_579945


namespace number_of_integer_pairs_l579_579879

def condition_one (a b : ℤ) : Prop := (a^2 + b^2 < 25)
def condition_two (a b : ℤ) : Prop := (a^2 + b^2 < 9 * a)
def condition_three (a b : ℤ) : Prop := (a^2 + b^2 < 9 * b)
def condition_four (a b : ℤ) : Prop := (a^2 + b^2 < -9 * a + 81)

theorem number_of_integer_pairs : 
  { (a, b) : ℤ × ℤ | condition_one a b ∧ condition_two a b ∧ condition_three a b ∧ condition_four a b }.finite.card = 7 :=
sorry

end number_of_integer_pairs_l579_579879


namespace arrange_BANANA_l579_579291

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579291


namespace sandy_puppies_l579_579974

theorem sandy_puppies :
  ∀ (initial_puppies puppies_given_away remaining_puppies : ℕ),
  initial_puppies = 8 →
  puppies_given_away = 4 →
  remaining_puppies = initial_puppies - puppies_given_away →
  remaining_puppies = 4 :=
by
  intros initial_puppies puppies_given_away remaining_puppies
  intro h_initial
  intro h_given_away
  intro h_remaining
  rw [h_initial, h_given_away] at h_remaining
  exact h_remaining

end sandy_puppies_l579_579974


namespace simplify_expression_l579_579980

theorem simplify_expression : 
  18 * (8 / 15) * (3 / 4) = 12 / 5 := 
by 
  sorry

end simplify_expression_l579_579980


namespace banana_arrangements_l579_579118

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579118


namespace g_monotonicity_intervals_f_decreasing_min_a_range_a_l579_579857

open Set

noncomputable def g (x : ℝ) := x / (Real.log x)

noncomputable def f (x a : ℝ) := g x - a * x

-- Problem (I): Proving intervals of monotonicity for g(x)
theorem g_monotonicity_intervals :
  (∀ x ∈ Ioi (Real.exp 1), 0 < (g x - g (Real.exp 1))) ∧ 
  (∀ x ∈ Iio (Real.exp 1), x ≠ 1 → g (Real.exp 1) > g x ∨ g x > g 1) := sorry

-- Problem (II): Prove minimum value of a for f(x) decreasing on (1, +∞)
theorem f_decreasing_min_a (a : ℝ) :
  (∀ x ∈ Ioi 1, f x a < f 1 a) ↔ a ≥ 1/4 := sorry

-- Problem (III): Range of a given g(x1) ≤ f'(x2) + 2a condition
theorem range_a (a : ℝ) :
  (∀ x1 ∈ Icc (Real.exp 1) (Real.exp 2), ∃ x2 ∈ Icc (Real.exp 1) (Real.exp 2), g x1 ≤ f' x2 + 2 * a) ↔ 
  a ∈ Icc ((Real.exp 2)^2 / 2 - 1/4) (Real.exp 2^2 / 2 - 1/4) := sorry

end g_monotonicity_intervals_f_decreasing_min_a_range_a_l579_579857


namespace banana_permutations_l579_579374

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579374


namespace range_of_a_l579_579995

   noncomputable section

   variable {f : ℝ → ℝ}

   /-- The requried theorem based on the given conditions and the correct answer -/
   theorem range_of_a (even_f : ∀ x, f (-x) = f x)
                      (increasing_f : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y)
                      (h : f a ≤ f 2) : a ≤ -2 ∨ a ≥ 2 :=
   sorry
   
end range_of_a_l579_579995


namespace limit_sin_squared_root_l579_579891

theorem limit_sin_squared_root (n : ℕ) (h : 0 < n) :
  (Real.sqrt (n^2 + n)).tendsto (λ x, ↑n)
  ∧ (λ n, Real.sin ((n : ℝ) + 1 / 2 * Real.pi)).tendsto (λ x, 1):= sorry

end limit_sin_squared_root_l579_579891


namespace arrange_BANANA_l579_579633

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579633


namespace binary_arithmetic_l579_579754

theorem binary_arithmetic :
  (110010:ℕ) * (1100:ℕ) / (100:ℕ) / (10:ℕ) = 100100 :=
by sorry

end binary_arithmetic_l579_579754


namespace point_distance_on_ellipse_l579_579809

noncomputable theory
open Real

def ellipse (x y : ℝ) := x^2 / 9 + y^2 / 6 = 1

def foci_distance (a b : ℝ) := 2 * sqrt (a^2 - b^2)

def conditions (P : ℝ × ℝ) (cos_theta : ℝ) : Prop :=
  ellipse P.1 P.2 ∧ cos_theta = 3 / 5

theorem point_distance_on_ellipse (P : ℝ × ℝ) (cos_theta : ℝ)
  (h : conditions P cos_theta) : Real.sqrt (P.1^2 + P.2^2) = sqrt(30) / 2 :=
begin
  sorry,
end

end point_distance_on_ellipse_l579_579809


namespace sum_of_angles_l579_579992

theorem sum_of_angles (circle_divided_into_18_arcs : ∀ (n : ℕ), n < 18)
  (x_spanning_3_arcs : ∀ (x : ℕ), x = 3)
  (y_spanning_6_arcs : ∀ (y : ℕ), y = 6) :
  let degree_per_arc := 360 / 18,
  let central_angle_x := x_spanning_3_arcs 3 * degree_per_arc,
  let central_angle_y := y_spanning_6_arcs 6 * degree_per_arc,
  let sum_of_angles := central_angle_x + central_angle_y
  in sum_of_angles = 180 :=
by
  sorry

end sum_of_angles_l579_579992


namespace sum_of_zeroes_of_function_l579_579770

noncomputable def sum_zeroes_logarithmic_function (m : ℝ) : ℝ :=
  let x₁ := real.exp m + 2
  let x₂ := -real.exp m + 2
  x₁ + x₂

theorem sum_of_zeroes_of_function (m : ℝ) :
  sum_zeroes_logarithmic_function m = 4 :=
sorry

end sum_of_zeroes_of_function_l579_579770


namespace number_of_arrangements_banana_l579_579668

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579668


namespace arrange_BANANA_l579_579714

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579714


namespace permutations_BANANA_l579_579172

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579172


namespace BANANA_permutations_l579_579218

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579218


namespace banana_permutations_l579_579360

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579360


namespace number_of_arrangements_banana_l579_579672

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579672


namespace permutations_BANANA_l579_579183

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579183


namespace weight_difference_proof_l579_579910

theorem weight_difference_proof
  (labrador_start_weight : ℕ) (dachshund_start_weight : ℕ)
  (weight_gain_percentage : ℕ)
  (labrador_start_weight = 40)
  (dachshund_start_weight = 12)
  (weight_gain_percentage = 25) :
  (labrador_start_weight + labrador_start_weight * weight_gain_percentage / 100) -
  (dachshund_start_weight + dachshund_start_weight * weight_gain_percentage / 100) =
  35 := 
  sorry

end weight_difference_proof_l579_579910


namespace weight_difference_end_of_year_l579_579904

theorem weight_difference_end_of_year :
  let labrador_starting_weight := 40
  let dachshund_starting_weight := 12
  let weight_gain_percent := 0.25
  let labrador_end_weight := labrador_starting_weight + labrador_starting_weight * weight_gain_percent
  let dachshund_end_weight := dachshund_starting_weight + dachshund_starting_weight * weight_gain_percent
  (labrador_end_weight - dachshund_end_weight) = 35 :=
by
  let labrador_starting_weight := 40
  let dachshund_starting_weight := 12
  let weight_gain_percent := 0.25
  let labrador_end_weight := labrador_starting_weight + labrador_starting_weight * weight_gain_percent
  let dachshund_end_weight := dachshund_starting_weight + dachshund_starting_weight * weight_gain_percent
  -- Statement: difference in weight at end of year is 35 pounds
  have h : (labrador_end_weight - dachshund_end_weight) = 35 := sorry
  exact h

end weight_difference_end_of_year_l579_579904


namespace number_of_arrangements_banana_l579_579681

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579681


namespace complement_angle_l579_579779

theorem complement_angle (A : Real) (h : A = 55) : 90 - A = 35 := by
  sorry

end complement_angle_l579_579779


namespace find_n_values_l579_579755

open Nat

noncomputable def exists_distinct_integers (n : ℕ) :=
  ∃ (a : Fin n → ℤ), (∀ i j, i ≠ j → a i ≠ a j) ∧ 
  Real (∑ i, (i+1 : ℝ) / (a i : ℝ)) = Real (∑ i, (a i : ℝ)) / 2

theorem find_n_values :
  {n : ℕ | exists_distinct_integers n} = {3, 5, 6} := 
sorry

end find_n_values_l579_579755


namespace banana_permutations_l579_579483

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579483


namespace BANANA_permutation_l579_579576

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579576


namespace banana_arrangements_l579_579137

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579137


namespace point_distance_on_ellipse_l579_579810

noncomputable theory
open Real

def ellipse (x y : ℝ) := x^2 / 9 + y^2 / 6 = 1

def foci_distance (a b : ℝ) := 2 * sqrt (a^2 - b^2)

def conditions (P : ℝ × ℝ) (cos_theta : ℝ) : Prop :=
  ellipse P.1 P.2 ∧ cos_theta = 3 / 5

theorem point_distance_on_ellipse (P : ℝ × ℝ) (cos_theta : ℝ)
  (h : conditions P cos_theta) : Real.sqrt (P.1^2 + P.2^2) = sqrt(30) / 2 :=
begin
  sorry,
end

end point_distance_on_ellipse_l579_579810


namespace arrangement_count_BANANA_l579_579263

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579263


namespace BANANA_arrangements_l579_579027

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579027


namespace projection_of_2m_plus_n_on_m_l579_579872

Variables (λ : ℝ)

def m : ℝ × ℝ := (-1, 2)
def n : ℝ × ℝ := (2, λ)

lemma perp_vectors (h : m.1 * n.1 + m.2 * n.2 = 0) : 
  λ = 1 :=
by linarith

def projection_vector 
  (h : n = (2, 1)) : ℝ × ℝ :=
let two_m_n := (2 * m.1 + n.1, 2 * m.2 + n.2) in
let proj := (two_m_n.1 * m.1 + two_m_n.2 * m.2) / (m.1 * m.1 + m.2 * m.2) in
(proj * m.1, proj * m.2)

theorem projection_of_2m_plus_n_on_m :
  (λ = 1) → projection_vector (by rfl) = (-2, 4) :=
by {
  intro hλ,
  simp [projection_vector, hλ],
  -- Computing intermediary steps
  have two_m_n := (2 * m.1 + n.1, 2 * m.2 + n.2),
  have proj := (two_m_n.1 * m.1 + two_m_n.2 * m.2) / (m.1 * m.1 + m.2 * m.2),
  -- Prove that proj = 2, hence
  have h_proj : proj = 2, sorry,
  simp [two_m_n, proj, h_proj],
}

end projection_of_2m_plus_n_on_m_l579_579872


namespace permutations_BANANA_l579_579175

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579175


namespace banana_permutations_l579_579499

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579499


namespace permutations_of_BANANA_l579_579405

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579405


namespace symmetric_heights_intersect_at_circumcenter_l579_579741

theorem symmetric_heights_intersect_at_circumcenter 
  (A B C : Point) 
  (H : Triangle A B C) 
  (O : Point)
  (circumcenter_O : IsCircumcenter O A B C)
  (HA : Height A H)
  (HB : Height B H)
  (HC : Height C H)
  (AD : AngleBisector A H)
  (BE : AngleBisector B H)
  (CF : AngleBisector C H)
  (A'A : ReflectAcross AD HA)
  (B'B : ReflectAcross BE HB)
  (C'C : ReflectAcross CF HC):
  intersect A'A B'B C'C O :=
sorry

end symmetric_heights_intersect_at_circumcenter_l579_579741


namespace permutations_BANANA_l579_579181

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579181


namespace arrange_BANANA_l579_579653

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579653


namespace BANANA_permutations_l579_579607

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579607


namespace banana_permutations_l579_579495

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579495


namespace BANANA_arrangements_correct_l579_579445

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579445


namespace arrange_BANANA_l579_579655

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579655


namespace BANANA_permutation_l579_579568

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579568


namespace banana_permutations_l579_579466

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579466


namespace arrange_BANANA_l579_579730

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579730


namespace fixed_point_intersection_l579_579835

-- Given conditions
def center_of_ellipse (E : Ellipse) : Point := Point.O
def axes_of_symmetry (E : Ellipse) : (Line, Line) := (Line.x_axis, Line.y_axis)
def passes_through_points (E : Ellipse) (A B : Point) : Prop :=
  E.contains_point A ∧ E.contains_point B

-- Given specific points
def A := Point.mk 0 (-2)
def B := Point.mk (3 / 2) (-1)
def P := Point.mk 1 (-2)
def K := Point.mk 0 (-2)

-- The ellipse we derived
noncomputable def E : Ellipse := Ellipse.equation3 (1 / 3) (1 / 4)

-- Given geometric constructions
def line_through_points (P : Point) (A : Point) : Line := Line.mk P A
def intersects_ellipse (line : Line) (E : Ellipse) : Point × Point := E.intersection_with_line line
def parallel_to_x_axis (line : Line) : Line := Line.parallel_x line
def intersection (line1 line2 : Line) : Point := line1.intersection line2
def midpoint (M T : Point) : Point := Point.mk ((M.x + T.x) / 2) ((M.y + T.y) / 2)

-- The proof problem
theorem fixed_point_intersection :
  ∀ (M N : Point),
    M ∈ intersects_ellipse (line_through_points P A) E →
    N ∈ intersects_ellipse (line_through_points P B) E →
    let T := intersection (parallel_to_x_axis (Line.mk M (Point.mk 1 0))) (line_through_points A B)
    let H := midpoint M T
    line_through_points H N = line_through_points K N :=
sorry

end fixed_point_intersection_l579_579835


namespace banana_arrangements_l579_579336

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579336


namespace arrange_BANANA_l579_579717

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579717


namespace number_of_arrangements_banana_l579_579664

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579664


namespace BANANA_arrangements_correct_l579_579448

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579448


namespace larger_R2_smaller_residual_sum_squares_l579_579919

theorem larger_R2_smaller_residual_sum_squares
  (R2 : ℝ)
  (S : ℝ) 
  (model_fitting : ℝ → ℝ)
  (h1 : ∀ x, 0 ≤ model_fitting x ∧ model_fitting x ≤ 1) -- R2 is between 0 and 1
  (h2 : ∀ x y, x ≤ y → model_fitting x ≤ model_fitting y) -- The larger the R2, the better the fitting effect
  (h3 : ∀ x, S = ∑ n in range x, (y_n - model_fitting x)^2) -- The sum of squares of residuals
  : (∃ R2 : ℝ, model_fitting R2 > R2) :=
sorry

end larger_R2_smaller_residual_sum_squares_l579_579919


namespace grandma_can_give_cherry_exists_better_grand_strategy_l579_579918

variable (Packet1 : Finset String) (Packet2 : Finset String) (Packet3 : Finset String)
variable (isCabbage : String → Prop) (isCherry : String → Prop)
variable (wholePie : String → Prop)

-- Conditions
axiom Packet1_cond : ∀ p ∈ Packet1, isCabbage p
axiom Packet2_cond : ∀ p ∈ Packet2, isCherry p
axiom Packet3_cond_cabbage : ∃ p ∈ Packet3, isCabbage p
axiom Packet3_cond_cherry : ∃ p ∈ Packet3, isCherry p

-- Question (a)
theorem grandma_can_give_cherry (h1 : ∃ p1 ∈ Packet3, wholePie p1 ∧ isCherry p1 ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy = (2 / 3) * (1 : ℝ) :=
by
  sorry

-- Question (b)
theorem exists_better_grand_strategy (h2 : ∃ p ∈ Packet3, wholePie p ∧ isCherry p ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy > (2 / 3) * (1 : ℝ) :=
by
  sorry

end grandma_can_give_cherry_exists_better_grand_strategy_l579_579918


namespace permutations_banana_l579_579105

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579105


namespace arrangement_count_BANANA_l579_579255

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579255


namespace banana_arrangements_l579_579325

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579325


namespace permutations_BANANA_l579_579152

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579152


namespace permutations_banana_l579_579083

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579083


namespace banana_permutations_l579_579373

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579373


namespace arrangement_count_BANANA_l579_579234

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579234


namespace BANANA_permutations_l579_579606

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579606


namespace BANANA_permutations_l579_579595

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579595


namespace banana_permutations_l579_579471

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579471


namespace permutations_banana_l579_579095

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579095


namespace banana_arrangements_l579_579322

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579322


namespace arrange_BANANA_l579_579307

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579307


namespace number_of_arrangements_banana_l579_579666

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579666


namespace banana_permutations_l579_579380

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579380


namespace banana_arrangements_l579_579334

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579334


namespace permutations_BANANA_l579_579155

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579155


namespace sara_stack_sum_l579_579976

def is_divisible (a b : Nat) : Prop :=
  a % b = 0 ∨ b % a = 0

def stack_alternates_and_sums_to_12 (red_cards blue_cards : List Nat) : Prop :=
  red_cards = [1, 2, 3, 4, 5, 6, 7] ∧ 
  blue_cards = [4, 5, 6, 7, 8, 9] ∧
  (∀ i, i < 6 → is_divisible (blue_cards.nthLe i sorry) (red_cards.nthLe i.succ sorry) ∧
           is_divisible (red_cards.nthLe i.succ sorry) (blue_cards.nthLe i sorry)) ∧
  let middle_three := [red_cards.nthLe 5 sorry, blue_cards.nthLe 2 sorry, red_cards.nthLe 2 sorry] in
  middle_three.sum = 12

theorem sara_stack_sum : 
  ∃ red_cards blue_cards, stack_alternates_and_sums_to_12 red_cards blue_cards :=
sorry

end sara_stack_sum_l579_579976


namespace prove_proposition_l579_579823

def p := ∀ α : ℝ, sin α * sin α ≤ 1
def q := ∃ x₀ : ℝ, x₀ * x₀ + 1 = 0

theorem prove_proposition : p → ¬q → (¬p ∨ ¬q) :=
by
  sorry

end prove_proposition_l579_579823


namespace triangle_shortest_side_l579_579990

theorem triangle_shortest_side (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (base : Real) (base_angle : Real) (sum_other_sides : Real)
    (h1 : base = 80) 
    (h2 : base_angle = 60) 
    (h3 : sum_other_sides = 90) : 
    ∃ shortest_side : Real, shortest_side = 17 :=
by 
    sorry

end triangle_shortest_side_l579_579990


namespace BANANA_arrangements_l579_579030

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579030


namespace po_distance_l579_579818

-- Define the ellispse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 6) = 1

-- Define the condition for cosine of the angle between F1, P, and F2
def cos_angle (cos_val : ℝ) : Prop := cos_val = 3/5

-- Define the proof problem in Lean
theorem po_distance (x y : ℝ) (hx : ellipse x y) (cos_val : ℝ) (h_cos : cos_angle cos_val) : (pow (sqrt ((15:ℝ)/2)) 2) = 30/4 :=
by
  sorry

end po_distance_l579_579818


namespace banana_arrangements_l579_579039

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579039


namespace digit_sum_product_l579_579890

-- Define the sequences with repeated patterns
def a1 : Nat := (151 * 10^3)^1004
def a2 : Nat := (3 * 10^3)^52008

-- Define the function to calculate a sum of digits of a number
def digit_sum (n : Nat) : Nat :=
  n.digits.sum

-- State the theorem to be proven
theorem digit_sum_product :
  digit_sum (a1 * a2) = 18072 := 
sorry

end digit_sum_product_l579_579890


namespace permutations_of_BANANA_l579_579406

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579406


namespace arrange_BANANA_l579_579721

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579721


namespace permutations_of_BANANA_l579_579408

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579408


namespace banana_arrangements_l579_579148

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579148


namespace sin_angle_PGH_l579_579916

-- Define the coordinates of the triangle vertices and the points G, H
def Q := (0 : ℝ, 0 : ℝ)
def R := (8 : ℝ, 0 : ℝ)
def P := (4 : ℝ, 4 * real.sqrt 3)
def G := (2 : ℝ, 0 : ℝ)
def H := (6 : ℝ, 0 : ℝ)

-- Calculation of sin(angle PGH)
theorem sin_angle_PGH : real.sin (angle P G H) = 4 * real.sqrt 3 / 13 := by
  sorry

end sin_angle_PGH_l579_579916


namespace banana_permutations_l579_579468

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579468


namespace num_ways_to_arrange_BANANA_l579_579524

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579524


namespace no_positive_integers_satisfy_l579_579759

theorem no_positive_integers_satisfy (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ¬ (3 * a^2 = b^2 + 1) := 
sorry

end no_positive_integers_satisfy_l579_579759


namespace arrangement_count_BANANA_l579_579265

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579265


namespace BANANA_arrangements_correct_l579_579455

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579455


namespace area_of_rectangle_l579_579996

-- Definitions of the conditions
def length (w : ℝ) : ℝ := 4 * w
def perimeter_eq_200 (w l : ℝ) : Prop := 2 * l + 2 * w = 200

-- Main theorem statement
theorem area_of_rectangle (w l : ℝ) (h1 : length w = l) (h2 : perimeter_eq_200 w l) : l * w = 1600 :=
by
  -- Skip the proof
  sorry

end area_of_rectangle_l579_579996


namespace BANANA_arrangements_l579_579014

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579014


namespace number_of_arrangements_banana_l579_579678

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579678


namespace area_X_l579_579942

/-- Let WXYZ be a convex quadrilateral with specified side lengths and area.
    Determine the area of quadrilateral X'Y'Z'W' when sides are extended as described.
 -/
theorem area_X'Y'Z'W' (W X Y Z X' Y' Z' W' : Type) [ConvexQuadrilateral WXYZ] 
  (WY XZ WZ XY : ℝ) (area_WXYZ : ℝ) (extension_condition : ∀ (WX XY YZ ZW : ℝ), 
  WX = X'X ∧ XY = Y'Y ∧ YZ = Z'Z ∧ ZW = W'W) : 
  area_WXYZ = 36 →
  WY = 12 →
  XZ = 18 →
  WZ = 15 →
  XY = 9 →
  area_X'Y'Z'W' = 108 :=
by sorry

end area_X_l579_579942


namespace mass_percentage_Al_in_AlI3_l579_579764

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_AlI3 : ℝ := molar_mass_Al + 3 * molar_mass_I

theorem mass_percentage_Al_in_AlI3 : 
  (molar_mass_Al / molar_mass_AlI3) * 100 = 6.62 := 
  sorry

end mass_percentage_Al_in_AlI3_l579_579764


namespace BANANA_permutation_l579_579565

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579565


namespace number_of_arrangements_banana_l579_579679

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579679


namespace BANANA_permutation_l579_579578

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579578


namespace arrange_BANANA_l579_579711

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579711


namespace banana_permutations_l579_579502

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579502


namespace arrange_BANANA_l579_579300

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579300


namespace permutations_of_BANANA_l579_579415

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579415


namespace arrange_BANANA_l579_579647

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579647


namespace num_ways_to_arrange_BANANA_l579_579512

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579512


namespace BANANA_permutations_l579_579213

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579213


namespace paco_cookies_l579_579967

theorem paco_cookies (total_cookies : ℕ) (cookies_given : ℕ) (cookies_eaten : ℕ) :
  total_cookies = 17 → 
  cookies_given = 13 → 
  cookies_eaten = cookies_given + 1 → 
  cookies_eaten = 14 :=
by
  intros h1 h2 h3
  rw [h2, h3]
  norm_num
  sorry

end paco_cookies_l579_579967


namespace banana_arrangements_l579_579071

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579071


namespace arrange_BANANA_l579_579295

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579295


namespace permutations_BANANA_l579_579184

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579184


namespace greatest_num_divisors_of_multiple_of_3_in_range_is_24_and_30_l579_579962

def count_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

def multiples_of_3_in_range : list ℕ :=
  list.filter (λ x, x % 3 = 0) (list.range' 1 30)

def max_divisors_multiples_of_3 : list ℕ :=
  let divisors_count := multiples_of_3_in_range.map (λ n, (n, count_divisors n)) in
  let max_div_count := divisors_count.foldl (λ acc p, max acc p.snd) 0 in
  divisors_count.filter (λ p, p.snd = max_div_count).map prod.fst

theorem greatest_num_divisors_of_multiple_of_3_in_range_is_24_and_30 :
  max_divisors_multiples_of_3 = [24, 30] :=
sorry

end greatest_num_divisors_of_multiple_of_3_in_range_is_24_and_30_l579_579962


namespace banana_arrangements_l579_579063

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579063


namespace permutations_BANANA_l579_579165

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579165


namespace banana_arrangements_l579_579323

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579323


namespace remainder_when_divided_l579_579821

theorem remainder_when_divided (P K Q R K' Q' S' T : ℕ)
  (h1 : P = K * Q + R)
  (h2 : Q = K' * Q' + S')
  (h3 : R * Q' = T) :
  P % (K * K') = K * S' + (T / Q') :=
by
  sorry

end remainder_when_divided_l579_579821


namespace num_ways_to_arrange_BANANA_l579_579510

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579510


namespace banana_arrangements_l579_579316

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579316


namespace banana_arrangements_l579_579328

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579328


namespace num_ways_to_arrange_BANANA_l579_579534

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579534


namespace banana_arrangements_l579_579061

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579061


namespace point_distance_on_ellipse_l579_579811

noncomputable theory
open Real

def ellipse (x y : ℝ) := x^2 / 9 + y^2 / 6 = 1

def foci_distance (a b : ℝ) := 2 * sqrt (a^2 - b^2)

def conditions (P : ℝ × ℝ) (cos_theta : ℝ) : Prop :=
  ellipse P.1 P.2 ∧ cos_theta = 3 / 5

theorem point_distance_on_ellipse (P : ℝ × ℝ) (cos_theta : ℝ)
  (h : conditions P cos_theta) : Real.sqrt (P.1^2 + P.2^2) = sqrt(30) / 2 :=
begin
  sorry,
end

end point_distance_on_ellipse_l579_579811


namespace product_min_max_l579_579948

noncomputable def k_min : ℝ := (12 - 3 * Real.sqrt 2) / 9
noncomputable def k_max : ℝ := (12 + 3 * Real.sqrt 2) / 9

theorem product_min_max 
    (x y : ℝ) 
    (h : 3 * x ^ 2 + 9 * x * y + 7 * y ^ 2 = 2) : 
    let k := x^2 + 4 * x * y + 3 * y ^ 2 in
    k_min * k_max = 7 / 9 :=
by sorry

end product_min_max_l579_579948


namespace permutations_banana_l579_579079

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579079


namespace permutations_BANANA_l579_579164

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579164


namespace arrangement_count_BANANA_l579_579245

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579245


namespace BANANA_permutations_l579_579191

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579191


namespace arrange_BANANA_l579_579710

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579710


namespace weight_difference_at_end_of_year_l579_579908

-- Conditions
def labrador_initial_weight : ℝ := 40
def dachshund_initial_weight : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

-- Question: Difference in weight at the end of the year
theorem weight_difference_at_end_of_year : 
  let labrador_final_weight := labrador_initial_weight * (1 + weight_gain_percentage)
  let dachshund_final_weight := dachshund_initial_weight * (1 + weight_gain_percentage)
  labrador_final_weight - dachshund_final_weight = 35 :=
by
  sorry

end weight_difference_at_end_of_year_l579_579908


namespace permutations_banana_l579_579092

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579092


namespace banana_arrangements_l579_579055

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579055


namespace BANANA_permutations_l579_579194

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579194


namespace banana_arrangements_l579_579330

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579330


namespace solve_inequality_l579_579984

theorem solve_inequality : { x : ℝ | (x + 3) * (x - 2) < 0 } = set.Ioo (-3) 2 := 
sorry

end solve_inequality_l579_579984


namespace BANANA_permutations_l579_579197

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579197


namespace banana_arrangements_l579_579136

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579136


namespace arrangement_count_BANANA_l579_579237

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579237


namespace find_k_l579_579894

theorem find_k (x y k : ℝ)
  (h1 : x - 4 * y + 3 ≤ 0)
  (h2 : 3 * x + 5 * y - 25 ≤ 0)
  (h3 : x ≥ 1)
  (h4 : ∃ z, z = k * x + y ∧ z = 12)
  (h5 : ∃ z', z' = k * x + y ∧ z' = 3) :
  k = 2 :=
by sorry

end find_k_l579_579894


namespace BANANA_permutations_l579_579618

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579618


namespace fixed_point_intersection_l579_579834

-- Given conditions
def center_of_ellipse (E : Ellipse) : Point := Point.O
def axes_of_symmetry (E : Ellipse) : (Line, Line) := (Line.x_axis, Line.y_axis)
def passes_through_points (E : Ellipse) (A B : Point) : Prop :=
  E.contains_point A ∧ E.contains_point B

-- Given specific points
def A := Point.mk 0 (-2)
def B := Point.mk (3 / 2) (-1)
def P := Point.mk 1 (-2)
def K := Point.mk 0 (-2)

-- The ellipse we derived
noncomputable def E : Ellipse := Ellipse.equation3 (1 / 3) (1 / 4)

-- Given geometric constructions
def line_through_points (P : Point) (A : Point) : Line := Line.mk P A
def intersects_ellipse (line : Line) (E : Ellipse) : Point × Point := E.intersection_with_line line
def parallel_to_x_axis (line : Line) : Line := Line.parallel_x line
def intersection (line1 line2 : Line) : Point := line1.intersection line2
def midpoint (M T : Point) : Point := Point.mk ((M.x + T.x) / 2) ((M.y + T.y) / 2)

-- The proof problem
theorem fixed_point_intersection :
  ∀ (M N : Point),
    M ∈ intersects_ellipse (line_through_points P A) E →
    N ∈ intersects_ellipse (line_through_points P B) E →
    let T := intersection (parallel_to_x_axis (Line.mk M (Point.mk 1 0))) (line_through_points A B)
    let H := midpoint M T
    line_through_points H N = line_through_points K N :=
sorry

end fixed_point_intersection_l579_579834


namespace banana_arrangements_l579_579073

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579073


namespace polynomial_identity_l579_579744

theorem polynomial_identity (x : ℝ) :
  (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1 = (x - 1)^5 := 
by 
  sorry

end polynomial_identity_l579_579744


namespace BANANA_permutation_l579_579558

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579558


namespace BANANA_permutations_l579_579206

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579206


namespace arrange_BANANA_l579_579278

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579278


namespace arrange_BANANA_l579_579720

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579720


namespace arrangement_count_BANANA_l579_579254

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579254


namespace solve_fraction_eq_l579_579983

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x = -1) ↔ ((x^2 + 2 * x + 3) / (x + 2) = x + 3) := 
by 
  sorry

end solve_fraction_eq_l579_579983


namespace BANANA_permutation_l579_579549

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579549


namespace minimum_value_of_even_function_l579_579898

noncomputable theory
open Real

def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

theorem minimum_value_of_even_function :
  ∀ (a : ℝ), is_even_function (λ x, a * x^2 + (a^2 - 1) * x - 3 * a) ∧ 
             ∀ (x : ℝ), 4 * a + 2 ≤ x ∧ x ≤ a^2 + 1 →
             ∃ min_val, ∀ x, (4 * a + 2 ≤ x) ∧ (x ≤ a^2 + 1) → 
                         (a = -1 → min_val = -1) :=
begin
  sorry
end

end minimum_value_of_even_function_l579_579898


namespace weight_difference_at_end_of_year_l579_579906

-- Conditions
def labrador_initial_weight : ℝ := 40
def dachshund_initial_weight : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

-- Question: Difference in weight at the end of the year
theorem weight_difference_at_end_of_year : 
  let labrador_final_weight := labrador_initial_weight * (1 + weight_gain_percentage)
  let dachshund_final_weight := dachshund_initial_weight * (1 + weight_gain_percentage)
  labrador_final_weight - dachshund_final_weight = 35 :=
by
  sorry

end weight_difference_at_end_of_year_l579_579906


namespace f_x_plus_f_1_minus_x_l579_579852

def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem f_x_plus_f_1_minus_x (x : ℝ) : f x + f (1 - x) = Real.sqrt 3 / 3 := 
by 
  sorry

end f_x_plus_f_1_minus_x_l579_579852


namespace banana_permutations_l579_579377

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579377


namespace arrange_BANANA_l579_579289

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579289


namespace permutations_of_BANANA_l579_579386

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579386


namespace jenny_ate_65_chocolates_l579_579933

noncomputable def chocolates_eaten_by_Jenny : ℕ :=
  let chocolates_mike := 20
  let chocolates_john := chocolates_mike / 2
  let combined_chocolates := chocolates_mike + chocolates_john
  let twice_combined_chocolates := 2 * combined_chocolates
  5 + twice_combined_chocolates

theorem jenny_ate_65_chocolates :
  chocolates_eaten_by_Jenny = 65 :=
by
  -- Skipping the proof details
  sorry

end jenny_ate_65_chocolates_l579_579933


namespace banana_arrangements_l579_579346

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579346


namespace banana_permutations_l579_579478

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579478


namespace banana_permutations_l579_579479

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579479


namespace number_of_arrangements_banana_l579_579687

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579687


namespace num_ways_to_arrange_BANANA_l579_579527

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579527


namespace projection_vector_l579_579871

def vect_m : ℝ × ℝ := (-1, 2)
def vect_n (λ : ℝ) : ℝ × ℝ := (2, λ)

theorem projection_vector:
  ∀ (λ : ℝ), (vect_m.1 * vect_n(λ).1 + vect_m.2 * vect_n(λ).2 = 0) → 
  (λ = 1) → 
  ((2 * vect_m.1 + vect_n(λ).1, 2 * vect_m.2 + vect_n(λ).2) : ℝ × ℝ) = (0, 5) → 
  ((2 * vect_m.1 + vect_n(λ).1, 2 * vect_m.2 + vect_n(λ).2) = (0, 5)) →
  let projection := ((0 * vect_m.1 + 5 * vect_m.2) / (vect_m.1^2 + vect_m.2^2)) * vect_m in
  projection = (-2, 4) :=
by
  intros
  sorry

end projection_vector_l579_579871


namespace permutations_banana_l579_579112

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579112


namespace combined_weighted_average_tax_rate_is_correct_l579_579935

-- Definitions of the given conditions
def john_job_income := 57000
def john_job_tax_rate := 0.30
def john_job_deductions := 5000

def john_rental_income := 11000
def john_rental_tax_rate := 0.25
def john_rental_deductions := 2000

def ingrid_employment_income := 72000
def ingrid_employment_tax_rate := 0.40
def ingrid_employment_exemptions := 8000

def ingrid_investment_income := 4500
def ingrid_investment_tax_rate := 0.15
def ingrid_investment_deductions := 1000

-- Calculation of taxable incomes
def john_taxable_income := (john_job_income - john_job_deductions) + (john_rental_income - john_rental_deductions)
def ingrid_taxable_income := (ingrid_employment_income - ingrid_employment_exemptions) + (ingrid_investment_income - ingrid_investment_deductions)

-- Calculation of total income
def total_taxable_income := john_taxable_income + ingrid_taxable_income

-- Calculation of total taxes paid
def john_total_tax_paid := (john_taxable_income * john_job_tax_rate) + (john_rental_income * john_rental_tax_rate)
def ingrid_total_tax_paid := (ingrid_taxable_income * ingrid_employment_tax_rate) + (ingrid_investment_income * ingrid_investment_tax_rate)

def total_tax_paid := john_total_tax_paid + ingrid_total_tax_paid

-- Calculation of the combined weighted average tax rate
def combined_weighted_average_tax_rate := total_tax_paid / total_taxable_income

-- The target proof problem
theorem combined_weighted_average_tax_rate_is_correct :
  combined_weighted_average_tax_rate = 0.3421 := 
sorry

end combined_weighted_average_tax_rate_is_correct_l579_579935


namespace permutations_BANANA_l579_579189

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579189


namespace arrangement_count_BANANA_l579_579231

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579231


namespace banana_arrangements_l579_579125

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579125


namespace banana_permutations_l579_579367

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579367


namespace number_of_arrangements_banana_l579_579663

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579663


namespace BANANA_permutation_l579_579574

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579574


namespace num_ways_to_arrange_BANANA_l579_579526

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579526


namespace problem_equivalent_l579_579987

theorem problem_equivalent (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) (hz_eq : z = 10 * y) :
  (x + 4 * y + z) / (4 * x - y - z) = 0 :=
by
  sorry

end problem_equivalent_l579_579987


namespace BANANA_arrangements_l579_579007

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579007


namespace BANANA_permutation_l579_579579

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579579


namespace qualified_light_bulb_from_factory_A_l579_579925

variable (P : Set (Set ℝ))
variable (A : Prop)
variable (B : Prop)

-- Given conditions
axiom prob_A : P A = 0.7
axiom prob_B_given_A : P (B ∩ A) / P A = 0.95

-- The theorem to be proven
theorem qualified_light_bulb_from_factory_A :
  P (B ∩ A) = 0.665 := by
  sorry

end qualified_light_bulb_from_factory_A_l579_579925


namespace BANANA_permutations_l579_579216

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579216


namespace permutations_banana_l579_579111

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579111


namespace permutations_banana_l579_579084

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579084


namespace number_of_arrangements_banana_l579_579690

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579690


namespace banana_arrangements_l579_579314

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579314


namespace banana_arrangements_l579_579115

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579115


namespace arrange_BANANA_l579_579732

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579732


namespace band_members_l579_579772

theorem band_members (total_people: ℕ) (portion_band_gets: ℝ) (ticket_price: ℝ) (amount_per_member: ℝ):
  total_people = 500 →
  portion_band_gets = 0.7 →
  ticket_price = 30 →
  amount_per_member = 2625 →
  (total_people * ticket_price * portion_band_gets / amount_per_member) = 4 :=
by
  intros h_total_people h_portion_band_gets h_ticket_price h_amount_per_member
  rw [h_total_people, h_portion_band_gets, h_ticket_price, h_amount_per_member]
  norm_num
  sorry

end band_members_l579_579772


namespace spider_catches_flies_l579_579742

def spider_const_rate (flies : ℕ) (time : ℕ) : Prop :=
  ∃ r : ℕ, flies = r * time

theorem spider_catches_flies : φ : ℕ := 9, t1 : ℕ := 5, t2 : ℕ := 30 :
  spider_const_rate φ t1 →
  spider_const_rate (φ * (t2 / t1)) t2 :=
by
  sorry

end spider_catches_flies_l579_579742


namespace arrangement_count_BANANA_l579_579240

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579240


namespace arrange_BANANA_l579_579625

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579625


namespace BANANA_permutations_l579_579210

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579210


namespace arrange_BANANA_l579_579303

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579303


namespace BANANA_arrangements_l579_579025

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579025


namespace permutations_of_BANANA_l579_579402

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579402


namespace common_difference_max_min_b_l579_579829

open Nat

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}
variable {b : ℕ → ℚ}
variable {d : ℚ}

-- Given conditions
def is_arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n, S n = ∑ i in range n, a i

def sequence_b (a : ℕ → ℚ) (b : ℕ → ℚ) : Prop :=
∀ n, b n = (1 + a n) / a n

-- Proof problem statements
theorem common_difference (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ) (h_arith : is_arithmetic_sequence a d) (h_sum : sum_of_first_n_terms a S) (h_condition : S 4 = 2 * S 2 + 4) :
  d = 1 := 
sorry

theorem max_min_b (a : ℕ → ℚ) (b : ℕ → ℚ) (h_seq_b : sequence_b a b) (h_a1 : a 1 = -5/2) :
  max (b 4) (min (b 1) (b 2)) = 3 ∧ min (b 3) (b 4) = -1 := 
sorry

end common_difference_max_min_b_l579_579829


namespace number_of_proper_subsets_of_M_l579_579954

open Finset

def A : Set ℤ := {1, 0}
def B : Set ℤ := {2, 3}
def M : Set ℤ := {x | ∃ a b, a ∈ A ∧ b ∈ B ∧ x = b * (a + b)}

theorem number_of_proper_subsets_of_M : (M.to_finset.powerset.card - 1) = 15 := by
  sorry

end number_of_proper_subsets_of_M_l579_579954


namespace BANANA_arrangements_correct_l579_579435

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579435


namespace upper_limit_karhtik_father_l579_579938

variables (weight : ℝ)

def karthik_opinion := 55 < weight ∧ weight < 62
def brother_opinion := 50 < weight ∧ weight < 60
def average_weight := ∑ x in { w | karthik_opinion w ∧ brother_opinion w }, x / 2 = 56.5

theorem upper_limit_karhtik_father (h_avg : average_weight weight) : ∃ (upper_limit : ℝ), upper_limit = 58 :=
sorry

end upper_limit_karhtik_father_l579_579938


namespace simplify_expression_l579_579777

theorem simplify_expression :
  ((2 + 3 + 4 + 5) / 2) + ((2 * 5 + 8) / 3) = 13 :=
by
  sorry

end simplify_expression_l579_579777


namespace sequence_sum_equals_299_l579_579927

def seq (a_n : ℕ → ℝ) :=
  ∀ n : ℕ, a_n + a_n.succ + a_n.succ.succ = C

def S_100 (a_n : ℕ → ℝ) : ℝ :=
  (Finset.sum (Finset.range 100) (λ i, a_n i))

theorem sequence_sum_equals_299 (a_n : ℕ → ℝ) (C : ℝ) :
  seq a_n → a_n 7 = 2 → a_n 9 = 3 → a_n 98 = 4 → S_100 a_n = 299 :=
by
  intros h_seq h_a7 h_a9 h_a98
  sorry

end sequence_sum_equals_299_l579_579927


namespace permutations_of_BANANA_l579_579389

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579389


namespace sum_of_p_q_r_s_t_l579_579951

theorem sum_of_p_q_r_s_t (p q r s t : ℤ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = 120) : 
  p + q + r + s + t = 32 := 
sorry

end sum_of_p_q_r_s_t_l579_579951


namespace permutations_banana_l579_579075

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579075


namespace number_of_arrangements_banana_l579_579683

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579683


namespace banana_arrangements_l579_579052

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579052


namespace BANANA_arrangements_l579_579026

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579026


namespace arrange_BANANA_l579_579716

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579716


namespace banana_arrangements_l579_579342

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579342


namespace number_of_arrangements_banana_l579_579689

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579689


namespace banana_permutations_l579_579347

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579347


namespace banana_arrangements_l579_579062

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579062


namespace arrange_BANANA_l579_579281

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579281


namespace number_neither_9_nice_nor_10_nice_500_l579_579774

def is_k_nice (N k : ℕ) : Prop := ∃ a : ℕ, a > 0 ∧ (∃ m : ℕ, N = (k * m) + 1)

def count_k_nice (N k : ℕ) : ℕ :=
  (N - 1) / k + 1

def count_neither_9_nice_nor_10_nice (N : ℕ) : ℕ :=
  let count_9_nice := count_k_nice N 9
  let count_10_nice := count_k_nice N 10
  let lcm_9_10 := 90  -- lcm of 9 and 10
  let count_both := count_k_nice N lcm_9_10
  N - (count_9_nice + count_10_nice - count_both)

theorem number_neither_9_nice_nor_10_nice_500 : count_neither_9_nice_nor_10_nice 500 = 400 :=
  sorry

end number_neither_9_nice_nor_10_nice_500_l579_579774


namespace permutations_BANANA_l579_579173

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579173


namespace half_MN_correct_l579_579868

noncomputable def OM : ℝ × ℝ := (-2, 3)
noncomputable def ON : ℝ × ℝ := (-1, -5)
noncomputable def MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
noncomputable def half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)

theorem half_MN_correct : half_MN = (1 / 2, -4) :=
by
  -- define the values of OM and ON
  let OM : ℝ × ℝ := (-2, 3)
  let ON : ℝ × ℝ := (-1, -5)
  -- calculate MN
  let MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
  -- calculate half of MN
  let half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)
  -- assert the expected value
  exact sorry

end half_MN_correct_l579_579868


namespace BANANA_permutations_l579_579614

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579614


namespace BANANA_permutation_l579_579544

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579544


namespace arrange_BANANA_l579_579734

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579734


namespace permutations_BANANA_l579_579160

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579160


namespace num_ways_to_arrange_BANANA_l579_579531

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579531


namespace permutations_banana_l579_579088

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579088


namespace BANANA_arrangements_l579_579016

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579016


namespace BANANA_arrangements_correct_l579_579427

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579427


namespace range_of_a_l579_579955

theorem range_of_a (a : ℝ) :
  (S := {x : ℝ | (1 / 2) < 2^x ∧ 2^x < 8}) ∧
  (T := {x : ℝ | x < a ∨ x > a + 2}) ∧
  (S ∪ T = set.univ) →
  a ∈ Ioo (-1 : ℝ) 1 :=
by sorry

end range_of_a_l579_579955


namespace BANANA_arrangements_correct_l579_579458

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579458


namespace BANANA_permutations_l579_579592

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579592


namespace BANANA_permutation_l579_579555

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579555


namespace BANANA_arrangements_l579_579028

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579028


namespace distance_from_origin_ellipse_point_l579_579794

theorem distance_from_origin_ellipse_point :
  ∃ P : ℝ × ℝ, 
    (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
    (let c := real.sqrt(3) in 
    let F1 := (-c, 0) in 
    let F2 := (c, 0) in 
    real.cos (
      vector.angle 
       ⟨P.1 - F1.1, P.2 - F1.2⟩ 
       ⟨P.1 - F2.1, P.2 - F2.2⟩
    ) = 3/5) ∧ 
  real.sqrt (P.1^2 + P.2^2) = real.sqrt(30) / 2 := 
sorry

end distance_from_origin_ellipse_point_l579_579794


namespace arrange_BANANA_l579_579642

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579642


namespace BANANA_arrangements_l579_579033

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579033


namespace BANANA_permutation_l579_579551

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579551


namespace banana_arrangements_l579_579341

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579341


namespace permutations_of_BANANA_l579_579411

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579411


namespace determine_range_of_a_l579_579844

-- Define the conditions for the function f
variable (f : ℝ → ℝ)

-- Define that f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

-- Define that f is monotonically increasing on (0, ∞)
def mono_inc_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f(x) < f(y)

-- Define the given inequality condition involving a
def satisfies_inequality (a : ℝ) (f : ℝ → ℝ) : Prop :=
  f (Real.exp (abs (1/2 * a - 1))) + f (-Real.sqrt Real.exp 1) < 0

-- Define the range of a satisfying the conditions
def range_of_a (f : ℝ → ℝ) : Set ℝ :=
  {a : ℝ | satisfies_inequality a f}

-- Theorem statement to prove the range of a under given conditions
theorem determine_range_of_a (f : ℝ → ℝ) (hodd : odd_function f) (hmono : mono_inc_on_positive f) :
  range_of_a f = {a : ℝ | 1 < a ∧ a < 3} :=
by
  sorry

end determine_range_of_a_l579_579844


namespace BANANA_arrangements_correct_l579_579436

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579436


namespace banana_permutations_l579_579474

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579474


namespace distance_from_origin_ellipse_point_l579_579797

theorem distance_from_origin_ellipse_point :
  ∃ P : ℝ × ℝ, 
    (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
    (let c := real.sqrt(3) in 
    let F1 := (-c, 0) in 
    let F2 := (c, 0) in 
    real.cos (
      vector.angle 
       ⟨P.1 - F1.1, P.2 - F1.2⟩ 
       ⟨P.1 - F2.1, P.2 - F2.2⟩
    ) = 3/5) ∧ 
  real.sqrt (P.1^2 + P.2^2) = real.sqrt(30) / 2 := 
sorry

end distance_from_origin_ellipse_point_l579_579797


namespace banana_permutations_l579_579382

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579382


namespace BANANA_permutation_l579_579557

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579557


namespace BANANA_permutations_l579_579219

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579219


namespace num_ways_to_arrange_BANANA_l579_579504

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579504


namespace area_cross_section_l579_579819

variable (a : ℝ)

-- Define parameters and constructs as given in conditions
def equilateral_triangular_prism_base_length (P A B C : ℝ) (a : ℝ) : Prop :=
a > 0 ∧ P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define the conditions: the base edge length and the angle conditions
def conditions (P A B C : ℝ) : Prop :=
equilateral_triangular_prism_base_length P A B C a ∧
-- Additional constraints for the geometric problem setup
(∃ EF : ℝ, parallel EF BC ∧ EF ⊥ PBC ∧ (angle_between_planes AEF ABC = 30) ∧
-- Define geometric relations such as the parallel and perpendicular constraints
through_point A EF ∧
-- Calculate lengths
let AD := (sqrt 3 / 2) * a in 
let AH := AD * (cos (30)) in 
let HD := AD * (sin (30)) in 
let PD := (sqrt 3 / 3) * a in
let PH := PD - HD in
-- Calculate final ratio and area
let EF := (PH / PD) * a in
calc_area A EF = (1 / 2) * AH * EF)

-- Statement to prove
theorem area_cross_section (P A B C : ℝ) (h : conditions P A B C) :
  ∃ area : ℝ, area = (3 / 32) * a ^ 2 :=
sorry

end area_cross_section_l579_579819


namespace permutations_of_BANANA_l579_579419

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579419


namespace PO_equals_l579_579806

def ellipse (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 6 = 1

def F1_c (a : ℝ) : ℝ := √(a^2 - 6)
def F2_c (a : ℝ) : ℝ := √(a^2 - 6)

def cos_angle_F1PF2 : ℝ := 3 / 5

noncomputable def PO := (P : ℝ × ℝ) := 
  1 / 2 * ((P.1 - F1_c 3)^2 + (P.2 - F2_c 3)^2 - 2 * (P.1 - F1_c 3) * (P.2 - F2_c 3) * cos_angle_F1PF2)

theorem PO_equals : ∀ (P : ℝ × ℝ), ellipse P.1 P.2 → ∥P∥ = √(30) / 2 :=
by
  intros P P_ellipse_condition
  sorry

end PO_equals_l579_579806


namespace arrange_BANANA_l579_579735

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579735


namespace banana_permutations_l579_579379

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579379


namespace permutations_banana_l579_579101

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579101


namespace arrange_BANANA_l579_579646

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579646


namespace arrange_BANANA_l579_579725

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579725


namespace banana_permutations_l579_579375

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579375


namespace lcm_12_20_correct_l579_579763

def lcm_12_20_is_60 : Nat := Nat.lcm 12 20

theorem lcm_12_20_correct : Nat.lcm 12 20 = 60 := by
  -- assumed factorization conditions as prerequisites
  have h₁ : Nat.primeFactors 12 = {2, 3} := sorry
  have h₂ : Nat.primeFactors 20 = {2, 5} := sorry
  -- the main proof goal
  exact sorry

end lcm_12_20_correct_l579_579763


namespace line_passes_through_fixed_point_minimal_triangle_area_eq_line_l579_579848

-- Part (1)
theorem line_passes_through_fixed_point (m : ℝ) :
  ∃ M : ℝ × ℝ, M = (-1, -2) ∧
    (∀ m : ℝ, (2 + m) * (-1) + (1 - 2 * m) * (-2) + (4 - 3 * m) = 0) := by
  sorry

-- Part (2)
theorem minimal_triangle_area_eq_line :
  ∃ k : ℝ, k = -2 ∧ 
    (∀ x y : ℝ, y = k * (x + 1) - 2 ↔ y = 2 * x + 4) := by
  sorry

end line_passes_through_fixed_point_minimal_triangle_area_eq_line_l579_579848


namespace banana_arrangements_l579_579315

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579315


namespace BANANA_permutations_l579_579608

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579608


namespace number_of_arrangements_banana_l579_579695

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579695


namespace num_ways_to_arrange_BANANA_l579_579515

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579515


namespace BANANA_permutations_l579_579201

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579201


namespace banana_permutations_l579_579365

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579365


namespace banana_permutations_l579_579383

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579383


namespace permutations_BANANA_l579_579171

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579171


namespace decimal_representation_of_fraction_l579_579753

theorem decimal_representation_of_fraction :
  (47 : ℝ) / (2^3 * 5^4) = 0.0094 :=
by
  sorry

end decimal_representation_of_fraction_l579_579753


namespace BANANA_arrangements_l579_579008

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579008


namespace arrange_BANANA_l579_579277

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579277


namespace students_drawn_from_A_l579_579978

-- Define the conditions as variables (number of students in each school)
def studentsA := 3600
def studentsB := 5400
def studentsC := 1800
def sampleSize := 90

-- Define the total number of students
def totalStudents := studentsA + studentsB + studentsC

-- Define the proportion of students in School A
def proportionA := studentsA / totalStudents

-- Define the number of students to be drawn from School A using stratified sampling
def drawnFromA := sampleSize * proportionA

-- The theorem to prove
theorem students_drawn_from_A : drawnFromA = 30 :=
by
  sorry

end students_drawn_from_A_l579_579978


namespace permutations_BANANA_l579_579174

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579174


namespace BANANA_permutations_l579_579221

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579221


namespace arrange_BANANA_l579_579656

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579656


namespace time_A_is_36_seconds_l579_579912

-- Define the variables and conditions
variables {distance timeB beatsBy timeA : ℕ}

-- Given conditions
def distance := 140
def timeB := 45
def beatsBy := 28

-- The goal is to show the time it takes for A to finish the race.
-- So we define the timeA satisfies the equation
def timeA := (112 * timeB) / distance

theorem time_A_is_36_seconds : timeA = 36 :=
by 
  -- Given that the speed formula and other conditions should be satisfied
  sorry

end time_A_is_36_seconds_l579_579912


namespace BANANA_permutations_l579_579582

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579582


namespace arrange_BANANA_l579_579704

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579704


namespace carrie_first_day_miles_l579_579965

theorem carrie_first_day_miles
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 124) -- Second day
  (h2 : ∀ y : ℕ, y = 159) -- Third day
  (h3 : ∀ y : ℕ, y = 189) -- Fourth day
  (h4 : ∀ z : ℕ, z = 106) -- Phone charge interval
  (h5 : ∀ n : ℕ, n = 7) -- Number of charges
  (h_total : 106 * 7 = x + (x + 124) + 159 + 189)
  : x = 135 :=
by sorry

end carrie_first_day_miles_l579_579965


namespace permutations_of_BANANA_l579_579409

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579409


namespace no_positive_integers_satisfy_l579_579758

theorem no_positive_integers_satisfy (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ¬ (3 * a^2 = b^2 + 1) := 
sorry

end no_positive_integers_satisfy_l579_579758


namespace BANANA_arrangements_l579_579018

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579018


namespace intersect_point_l579_579737

noncomputable def line1 : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), 3 * p.2 = -2 * p.1 + 6
noncomputable def line2 : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), -2 * p.2 = 8 * p.1 - 3

theorem intersect_point :
  ∃ (x y : ℝ), line1 (x, y) ∧ line2 (x, y) ∧ (x = 3 / 20) ∧ (y = 9 / 10) :=
by
  sorry

end intersect_point_l579_579737


namespace banana_arrangements_l579_579123

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579123


namespace arrange_BANANA_l579_579650

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579650


namespace tangent_line_eq_min_value_eq_range_of_a_l579_579855

-- Define the function f
def f (a x : ℝ) : ℝ := Real.exp x + 2 * a * x

-- (I) Tangent line condition
theorem tangent_line_eq (a : ℝ) (h : a = 1) : (3 : ℝ) * (0 : ℝ) - (1 : ℝ) + (1 : ℝ) = 0 := 
by
  sorry -- Proof omitted

-- (II) Minimum value condition
theorem min_value_eq (a : ℝ) (h : ∀ x ≥ (1:ℝ), Real.exp x + 2 * a * x ≥ 0 ∧ ∃ x, Real.exp x + 2 * a * x = 0 ) : a = -Real.exp 1 / 2 :=
by
  sorry -- Proof omitted

-- (III) Function constraints
theorem range_of_a (a : ℝ) (h : ∀ x ≥ (0:ℝ), Real.exp x + 2 * a * x ≥ Real.exp (-x)) : a ∈ Set.Ici (-1) :=
by
  sorry -- Proof omitted

end tangent_line_eq_min_value_eq_range_of_a_l579_579855


namespace permutations_of_BANANA_l579_579390

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579390


namespace vector_magnitude_l579_579831

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

namespace vector_problem

variables {x y : ℝ}
def a := (2 : ℝ, -4 : ℝ)
def b := (x, 1)
def c := (1, y)

def perp (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem vector_magnitude : perp a b ∧ parallel a c → magnitude (b.1 + c.1, b.2 + c.2) = real.sqrt 10 :=
begin
  intros h,
  sorry
end

end vector_problem

end vector_magnitude_l579_579831


namespace BANANA_arrangements_correct_l579_579463

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579463


namespace banana_arrangements_l579_579327

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579327


namespace max_colored_cells_l579_579743

-- Define the problem conditions
def is_colored (cell : ℕ × ℕ) (colored : set (ℕ × ℕ)) : Prop :=
  cell ∈ colored

def shares_side (cell1 cell2 : ℕ × ℕ) : Prop :=
  abs (cell1.1 - cell2.1) + abs (cell1.2 - cell2.2) = 1

def valid_coloring (colored : set (ℕ × ℕ)) : Prop :=
  ∀ cell ∈ colored, ∃ neighbor ∉ colored, shares_side cell neighbor ∧
  ∀ cell ∉ colored, ∃ neighbor ∈ colored, shares_side cell neighbor

-- Statement of the theorem
theorem max_colored_cells : ∀ (colored : set (ℕ × ℕ)),
  valid_coloring colored → |colored| ≤ 12 :=
begin
  sorry,
end


end max_colored_cells_l579_579743


namespace BANANA_permutation_l579_579559

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579559


namespace BANANA_arrangements_l579_579012

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579012


namespace permutations_banana_l579_579106

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579106


namespace banana_arrangements_l579_579317

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579317


namespace banana_arrangements_l579_579324

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579324


namespace arrange_BANANA_l579_579282

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579282


namespace number_of_arrangements_banana_l579_579685

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579685


namespace fixed_point_intersection_l579_579836

-- Given conditions
def center_of_ellipse (E : Ellipse) : Point := Point.O
def axes_of_symmetry (E : Ellipse) : (Line, Line) := (Line.x_axis, Line.y_axis)
def passes_through_points (E : Ellipse) (A B : Point) : Prop :=
  E.contains_point A ∧ E.contains_point B

-- Given specific points
def A := Point.mk 0 (-2)
def B := Point.mk (3 / 2) (-1)
def P := Point.mk 1 (-2)
def K := Point.mk 0 (-2)

-- The ellipse we derived
noncomputable def E : Ellipse := Ellipse.equation3 (1 / 3) (1 / 4)

-- Given geometric constructions
def line_through_points (P : Point) (A : Point) : Line := Line.mk P A
def intersects_ellipse (line : Line) (E : Ellipse) : Point × Point := E.intersection_with_line line
def parallel_to_x_axis (line : Line) : Line := Line.parallel_x line
def intersection (line1 line2 : Line) : Point := line1.intersection line2
def midpoint (M T : Point) : Point := Point.mk ((M.x + T.x) / 2) ((M.y + T.y) / 2)

-- The proof problem
theorem fixed_point_intersection :
  ∀ (M N : Point),
    M ∈ intersects_ellipse (line_through_points P A) E →
    N ∈ intersects_ellipse (line_through_points P B) E →
    let T := intersection (parallel_to_x_axis (Line.mk M (Point.mk 1 0))) (line_through_points A B)
    let H := midpoint M T
    line_through_points H N = line_through_points K N :=
sorry

end fixed_point_intersection_l579_579836


namespace banana_permutations_l579_579482

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579482


namespace permutations_BANANA_l579_579170

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579170


namespace BANANA_arrangements_correct_l579_579444

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579444


namespace number_of_arrangements_banana_l579_579671

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579671


namespace arrange_BANANA_l579_579713

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579713


namespace banana_permutations_l579_579358

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579358


namespace curve_C_cartesian_minimum_distance_PA_PB_l579_579922

def parametric_equation_of_line (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (3 + t * Real.cos α, 1 + t * Real.sin α)

def polar_to_cartesian (θ : ℝ) (ρ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem curve_C_cartesian :
  ∀ x y : ℝ, ρ = 4 * Real.cos θ → (x - 2) ^ 2 + y ^ 2 = 4 :=
sorry

theorem minimum_distance_PA_PB (α : ℝ) :
  let P := (3, 1),
      roots := {t | ∃ x y : ℝ, (x, y) = parametric_equation_of_line α t ∧ (x - 2) ^ 2 + y ^ 2 = 4}
  in ∃ t1 t2 : ℝ, 
        t1 ∈ roots ∧ t2 ∈ roots ∧
        ∀ A B : ℝ × ℝ,
          A = parametric_equation_of_line α t1 ∧ B = parametric_equation_of_line α t2 →
          |P.1 - A.1| + |P.2 - A.2| + |P.1 - B.1| + |P.2 - B.2| = 2 * Real.sqrt 2 :=
sorry

end curve_C_cartesian_minimum_distance_PA_PB_l579_579922


namespace parabola_equation_line_intersection_proof_l579_579860

-- Define the parabola and its properties
def parabola (p x y : ℝ) := y^2 = 2 * p * x

-- Define point A
def A_point (x y₀ : ℝ) := (x, y₀)

-- Define the conditions
axiom p_pos (p : ℝ) : p > 0
axiom passes_A (y₀ : ℝ) (p : ℝ) : parabola p 2 y₀
axiom distance_A_axis (p : ℝ) : 2 + p / 2 = 4

-- Prove the equation of the parabola given the conditions
theorem parabola_equation : ∃ p, parabola p x y ∧ p = 4 := sorry

-- Define line l and its intersection properties
def line_l (m x y : ℝ) := y = x + m
def intersection_PQ (m x₁ x₂ y₁ y₂ : ℝ) := 
  line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧ y₁^2 = 8 * x₁ ∧ y₂^2 = 8 * x₂ ∧ 
  x₁ + x₂ = 8 - 2 * m ∧ x₁ * x₂ = m^2 ∧ y₁ + y₂ = 8 ∧ y₁ * y₂ = 8 * m ∧ 
  x₁ * x₂ + y₁ * y₂ = 0

-- Prove the value of m
theorem line_intersection_proof : ∃ m, ∀ (x₁ x₂ y₁ y₂ : ℝ), 
  intersection_PQ m x₁ x₂ y₁ y₂ -> m = -8 := sorry

end parabola_equation_line_intersection_proof_l579_579860


namespace arrangement_count_BANANA_l579_579264

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579264


namespace permutations_BANANA_l579_579180

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579180


namespace arrange_BANANA_l579_579731

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579731


namespace weight_difference_end_of_year_l579_579905

theorem weight_difference_end_of_year :
  let labrador_starting_weight := 40
  let dachshund_starting_weight := 12
  let weight_gain_percent := 0.25
  let labrador_end_weight := labrador_starting_weight + labrador_starting_weight * weight_gain_percent
  let dachshund_end_weight := dachshund_starting_weight + dachshund_starting_weight * weight_gain_percent
  (labrador_end_weight - dachshund_end_weight) = 35 :=
by
  let labrador_starting_weight := 40
  let dachshund_starting_weight := 12
  let weight_gain_percent := 0.25
  let labrador_end_weight := labrador_starting_weight + labrador_starting_weight * weight_gain_percent
  let dachshund_end_weight := dachshund_starting_weight + dachshund_starting_weight * weight_gain_percent
  -- Statement: difference in weight at end of year is 35 pounds
  have h : (labrador_end_weight - dachshund_end_weight) = 35 := sorry
  exact h

end weight_difference_end_of_year_l579_579905


namespace BANANA_arrangements_correct_l579_579462

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579462


namespace arrange_BANANA_l579_579294

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579294


namespace BANANA_permutations_l579_579226

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579226


namespace BANANA_arrangements_l579_579031

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579031


namespace BANANA_permutation_l579_579546

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579546


namespace probability_diana_gt_apollo_l579_579740

-- Define the type representing all the possible outcomes of a roll for each die
def six_sided_die := {1, 2, 3, 4, 5, 6}
def four_sided_die := {1, 2, 3, 4}

-- Define a computation for successful outcomes where Diana's number is larger than Apollo's
def successful_outcomes : Nat :=
  let outcomes := [(a, b) | a ∈ six_sided_die, b ∈ four_sided_die, a > b]
  List.length outcomes

-- Define the total number of possible outcomes
def total_outcomes : Nat := 6 * 4

-- Define the probability calculation
def probability_diana_wins : ℚ := successful_outcomes / total_outcomes

-- State the theorem
theorem probability_diana_gt_apollo :
  probability_diana_wins = 7 / 12 := sorry

end probability_diana_gt_apollo_l579_579740


namespace banana_arrangements_l579_579036

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579036


namespace number_of_distinct_factors_l579_579877

theorem number_of_distinct_factors 
  (a b c : ℕ) 
  (h₁ : a = 10) 
  (h₂ : b = 5) 
  (h₃ : c = 3) : 
  let n := (2^a) * (5^b) * (7^c) in
  (∃ d e f : ℕ, d ≤ a ∧ e ≤ b ∧ f ≤ c ∧ n = (2^d) * (5^e) * (7^f)) ->
  ∏ i in finset.range (a + 1), ∏ j in finset.range (b + 1), ∏ k in finset.range (c + 1), 1 = 264 :=
by
  sorry

end number_of_distinct_factors_l579_579877


namespace permutations_BANANA_l579_579153

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579153


namespace max_path_number_l579_579989

-- Definitions
def independence_number (G : Type) [graph G] : ℕ := sorry

def path_number (G : Type) [graph G] : ℕ := sorry

variable {G : Type} [graph G] (k : ℕ)

-- Theorem statement
theorem max_path_number (hG : connected G) (hk : independence_number G > 1) :
  path_number G ≤ k - 1 :=
sorry

end max_path_number_l579_579989


namespace permutations_of_BANANA_l579_579420

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579420


namespace find_sin_double_angle_l579_579885

theorem find_sin_double_angle (θ : ℝ) (tanθ : ℝ) (h : tanθ = 1 / 3) : 
  Real.sin (2 * θ) = 3 / 5 :=
sorry

end find_sin_double_angle_l579_579885


namespace BANANA_permutations_l579_579603

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579603


namespace permutations_banana_l579_579099

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579099


namespace arrangement_count_BANANA_l579_579248

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579248


namespace banana_arrangements_l579_579116

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579116


namespace number_of_arrangements_banana_l579_579661

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579661


namespace probability_point_between_A_and_C_l579_579970

theorem probability_point_between_A_and_C {A B C D : ℝ} 
  (h1 : B = 4 * D) (h2 : B = 8 * C) : 
  ∀ (x : ℝ), x ∈ (0, B) → (x < C) ∨ x = C → ∃ p : ℝ, p = 7 / 8 := 
by
    sorry

end probability_point_between_A_and_C_l579_579970


namespace problem_statement_l579_579839

theorem problem_statement :
  (∀ (E : Type) (a b : ℝ) (x y : ℝ), 
     center E = (0, 0) ∧ 
     axes_symmetry E (x, y) ∧ 
     passes_through E (0, -2) (3/2, -1) →
     equation E = (x^2 / 3 + y^2 / 4 = 1) ∧
     (∀ (line : Type) (P M N H : Type)
       (fixed_point : (ℝ × ℝ)), 
       fixed_point = (0, -2) →
       passes_through P (1, -2) ∧
       intersects E line M N ∧
       passes_through_line HN H N →
       passes_through fixed_point HN) :=
sorry

end problem_statement_l579_579839


namespace BANANA_permutation_l579_579552

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579552


namespace banana_arrangements_l579_579337

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579337


namespace number_of_arrangements_banana_l579_579673

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579673


namespace permutations_banana_l579_579103

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579103


namespace intersection_eq_l579_579824

open Set

-- Define the set A based on the condition y = sqrt(1 - x)
def setA : Set ℝ := { x | x ≤ 1 }

-- Explicitly define the set B
def setB : Set ℝ := { -2, -1, 0, 1, 2 }

-- Theorem to prove the intersection of A and B
theorem intersection_eq : setA ∩ setB = { -2, -1, 0, 1 } := by
  sorry

end intersection_eq_l579_579824


namespace BANANA_arrangements_l579_579023

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579023


namespace banana_permutations_l579_579480

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579480


namespace num_ways_to_arrange_BANANA_l579_579521

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579521


namespace arrange_BANANA_l579_579706

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579706


namespace BANANA_permutation_l579_579563

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579563


namespace num_ways_to_arrange_BANANA_l579_579541

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579541


namespace banana_arrangements_l579_579121

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579121


namespace banana_arrangements_l579_579069

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579069


namespace complex_solution_count_l579_579876

-- Definitions to represent the problem conditions
def modulus_less_than_25 (z : ℂ) : Prop := Complex.abs z < 25

def satisfies_equation (z : ℂ) : Prop :=
  Complex.exp z = (z + 1) / (z - 1)

-- The theorem statement that needs proof
theorem complex_solution_count :
  {z : ℂ | modulus_less_than_25 z ∧ satisfies_equation z}.finite.to_finset.card = 8 := sorry

end complex_solution_count_l579_579876


namespace fraction_to_terminating_decimal_l579_579749

theorem fraction_to_terminating_decimal :
  (47 / (2^3 * 5^4) : ℝ) = 0.0094 := by
  sorry

end fraction_to_terminating_decimal_l579_579749


namespace triangle_property_l579_579949

variables (A B C D E F : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]
variables (γ : Angle A C B)
variables (h1 : γ / 2 < Angle B A C)
variables (h2 : γ / 2 < Angle C B A)
variables (h3 : PointOnLine D B C (Angle B A D = γ / 2))
variables (h4 : PointOnLine E C A (Angle E B A = γ / 2))
variables (h5 : IntersectionPoint F (Bisector γ) A B)

theorem triangle_property : (E F + F D = A B) :=
by
  sorry

end triangle_property_l579_579949


namespace banana_permutations_l579_579481

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579481


namespace BANANA_arrangements_correct_l579_579440

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579440


namespace banana_arrangements_l579_579339

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579339


namespace permutations_BANANA_l579_579182

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579182


namespace num_ways_to_arrange_BANANA_l579_579520

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579520


namespace number_of_arrangements_banana_l579_579688

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579688


namespace decimal_representation_of_fraction_l579_579752

theorem decimal_representation_of_fraction :
  (47 : ℝ) / (2^3 * 5^4) = 0.0094 :=
by
  sorry

end decimal_representation_of_fraction_l579_579752


namespace banana_arrangements_l579_579143

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579143


namespace notebooks_distributed_l579_579776

theorem notebooks_distributed  (C : ℕ) (N : ℕ) 
  (h1 : N = C^2 / 8) 
  (h2 : N = 8 * C) : 
  N = 512 :=
by 
  sorry

end notebooks_distributed_l579_579776


namespace BANANA_permutation_l579_579569

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579569


namespace arrange_BANANA_l579_579638

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579638


namespace banana_permutations_l579_579486

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579486


namespace intersection_count_l579_579854

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem intersection_count (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < Real.pi / 2) 
  (h_max : ∀ x, f x ω φ ≤ f (Real.pi / 6) ω φ)
  (h_period : ∀ x, f x ω φ = f (x + 2 * Real.pi / ω) ω φ) :
  ∃! x : ℝ, f x ω φ = -x + 2 * Real.pi / 3 :=
sorry

end intersection_count_l579_579854


namespace banana_permutations_l579_579496

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579496


namespace arrangement_count_BANANA_l579_579258

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579258


namespace arrange_BANANA_l579_579718

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579718


namespace banana_arrangements_l579_579066

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579066


namespace BANANA_permutation_l579_579556

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579556


namespace ellipse_center_origin_axes_symmetry_l579_579840

theorem ellipse_center_origin_axes_symmetry 
  (A B : ℝ × ℝ) (P : ℝ × ℝ)
  (hA : A = (0, -2))
  (hB : B = (3/2, -1))
  (hP : P = (1, -2)) :
  (∃ E : ℝ → ℝ → Prop, 
    (E = λ x y, (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ (x1 y1 x2 y2 : ℝ), 
      E x1 y1 →
      E x2 y2 →
      (x2 - x1) ≠ 0 →
      let M := (x1, y1),
          N := (x2, y2) in
      let T := (2 * x1 + (3/2 - x1) / 2, y1) in
      let H := (3 * x1 / 2 - 3 / 2 + 2, y1) in
      ∃ K : ℝ × ℝ, K = (0, -2) ∧ ((H.1 - N.1) * (K.2 - N.2) = (K.1 - N.1) * (H.2 - N.2)))) 
:= sorry

end ellipse_center_origin_axes_symmetry_l579_579840


namespace BANANA_arrangements_correct_l579_579450

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579450


namespace banana_permutations_l579_579385

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579385


namespace banana_arrangements_l579_579338

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579338


namespace banana_arrangements_l579_579149

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579149


namespace friends_meet_first_time_at_4pm_l579_579775

def lcm_four_times (a b c d : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

def first_meeting_time (start_time_minutes: ℕ) (lap_anna lap_stephanie lap_james lap_carlos: ℕ) : ℕ :=
  start_time_minutes + lcm_four_times lap_anna lap_stephanie lap_james lap_carlos

theorem friends_meet_first_time_at_4pm :
  first_meeting_time 600 5 8 9 12 = 960 :=
by
  -- where 600 represents 10:00 AM in minutes since midnight and 960 represents 4:00 PM
  sorry

end friends_meet_first_time_at_4pm_l579_579775


namespace num_ways_to_arrange_BANANA_l579_579536

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579536


namespace BANANA_permutations_l579_579615

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579615


namespace decimal_representation_of_fraction_l579_579751

theorem decimal_representation_of_fraction :
  (47 : ℝ) / (2^3 * 5^4) = 0.0094 :=
by
  sorry

end decimal_representation_of_fraction_l579_579751


namespace num_ways_to_arrange_BANANA_l579_579528

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579528


namespace banana_arrangements_l579_579133

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579133


namespace arrange_BANANA_l579_579276

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579276


namespace number_of_arrangements_banana_l579_579693

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579693


namespace banana_arrangements_l579_579058

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579058


namespace BANANA_permutations_l579_579223

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579223


namespace books_arrangement_l579_579917

noncomputable def num_ways_arrange_books : Nat :=
  3 * 2 * (Nat.factorial 6)

theorem books_arrangement :
  num_ways_arrange_books = 4320 :=
by
  sorry

end books_arrangement_l579_579917


namespace arrangement_count_BANANA_l579_579259

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579259


namespace BANANA_permutations_l579_579225

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579225


namespace solution_l579_579864

def recurrence (a : ℕ → ℤ) : Prop :=
∀ n ≥ 2, a (n + 1) = a n - a (n - 1)

def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(nat.range n).sum (λ i, a (i + 1))

lemma seq_periodicity (a : ℕ → ℤ) (h : recurrence a) :
  ∀ n ≥ 1, a (n + 6) = a n := 
sorry

lemma initial_conditions (a : ℕ → ℤ) :
  a 1 = 1 ∧ a 2 = 3 :=
sorry

theorem solution : ∃ a : ℕ → ℤ, 
  recurrence a ∧ initial_conditions a ∧ 
  a 100 = -1 ∧ S a 100 = 5 :=
sorry

end solution_l579_579864


namespace sides_of_triangle_l579_579968

-- Define variables and types
variables {α : Type*} [LinearOrderedField α]

-- Definitions of geometric entities
def Point := (α × α)
def Length (p1 p2 : Point) : α := real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Conditions
variables (A B C D M N : Point)
variables (r1 r2 : α)
variables (BM BN : α)

-- Given conditions
def given_conditions : Prop :=
  D ∈ line {A, C} ∧  -- D lies on side AC
  r1 = 2 / real.sqrt 3 ∧  -- Circle inscribed in ABD touches AB at M with radius 2 / sqrt(3)
  r2 = real.sqrt 3 ∧  -- Circle inscribed in BCD touches BC at N with radius sqrt(3)
  Length B M = 6 ∧  -- BM = 6
  Length B N = 5    -- BN = 5

-- Question translated into proof problem
theorem sides_of_triangle (h : given_conditions) : 
  Length A B = 8 ∧ Length B C = 8 ∧ Length A C = 8 :=
sorry

end sides_of_triangle_l579_579968


namespace BANANA_permutations_l579_579199

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579199


namespace arrange_BANANA_l579_579637

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579637


namespace arrangement_count_BANANA_l579_579261

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579261


namespace arrange_BANANA_l579_579272

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579272


namespace BANANA_arrangements_correct_l579_579454

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579454


namespace num_ways_to_arrange_BANANA_l579_579532

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579532


namespace BANANA_permutations_l579_579609

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579609


namespace BANANA_arrangements_correct_l579_579451

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579451


namespace arrangement_count_BANANA_l579_579246

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579246


namespace permutations_BANANA_l579_579163

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579163


namespace perimeter_inequality_l579_579971

-- Define the input conditions
variables {A B C A₁ B₁ C₁ : Point}
variables {P P₁ : ℝ}

-- Define side lengths
variables {AB BC CA : ℝ}
variables {BA₁ CB₁ AC₁ : ℝ}

-- Assume conditions
def conditions : Prop :=
  (BA₁ = 0.75 * BC) ∧
  (CB₁ = 0.75 * CA) ∧
  (AC₁ = 0.75 * AB) ∧
  (P = AB + BC + CA) ∧
  (P₁ = A₁B₁ + B₁C₁ + C₁A₁)

-- Define the theorem
theorem perimeter_inequality (h : conditions) : 0.5 * P < P₁ ∧ P₁ < 0.75 * P :=
sorry

end perimeter_inequality_l579_579971


namespace arrange_BANANA_l579_579648

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579648


namespace sum_first_40_terms_l579_579865

variable (a : ℕ → ℕ)

-- Define the relation given in the conditions
def relation (n : ℕ) : Prop :=
  a n + (-1)^(n+1) * a (n+1) = 2*n - 1

-- Define the sum of the first 40 terms
def S_40 : ℕ := ∑ i in range 40, a i

-- The theorem which states the sum of the first 40 terms is 74
theorem sum_first_40_terms : (∀ n, relation a n) → S_40 a = 74 := 
by
  sorry

end sum_first_40_terms_l579_579865


namespace BANANA_permutations_l579_579200

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579200


namespace arrange_BANANA_l579_579270

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579270


namespace BANANA_permutations_l579_579222

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579222


namespace arrangement_count_BANANA_l579_579232

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579232


namespace part1_part2_l579_579793

-- Defining the sequence {a_n}
def a (n : ℕ) (n_pos : n > 0) := (S n pos - S (n - 1) pos)

-- Given condition about the sum of the first n terms
def Sn (n : ℕ) (n_pos : n > 0) := 4/3 * (a n n_pos - 1)

-- Defining the sequence {b_n} as log_2 {a_n}
def bn (n : ℕ) (n_pos : n > 0) := Real.log 2 (a n n_pos)

-- Defining the sequence {T_n} as the sum of 1 / ((b_n - 1) * (b_n + 1))
def Tn (n : ℕ) := (Finset.range n).sum (λ k, 1 / (bn k + 1) (bn k - 1)))

-- Main problem divided into two parts
-- Part 1: Proving the general formula
theorem part1 (n : ℕ) (n_pos : n > 0):
  a n n_pos = 4 ^ n := sorry

-- Part 2: Proving the given inequality
theorem part2 (n : ℕ) (n_pos : n > 0):
  1/3 ≤ Tn n ∧ Tn n < 1/2 := sorry

end part1_part2_l579_579793


namespace BANANA_arrangements_correct_l579_579452

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579452


namespace BANANA_arrangements_l579_579019

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579019


namespace arrange_BANANA_l579_579640

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579640


namespace po_distance_l579_579815

-- Define the ellispse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 6) = 1

-- Define the condition for cosine of the angle between F1, P, and F2
def cos_angle (cos_val : ℝ) : Prop := cos_val = 3/5

-- Define the proof problem in Lean
theorem po_distance (x y : ℝ) (hx : ellipse x y) (cos_val : ℝ) (h_cos : cos_angle cos_val) : (pow (sqrt ((15:ℝ)/2)) 2) = 30/4 :=
by
  sorry

end po_distance_l579_579815


namespace find_a_in_complex_quadrant_l579_579830

def complex_quad_condition (a : ℝ) : Prop :=
    let z := (2 + a * complex.i) / (2 + complex.i)
    (z.re > 0) ∧ (z.im < 0)

theorem find_a_in_complex_quadrant :
    ∃ a : ℝ, complex_quad_condition a :=
sorry

end find_a_in_complex_quadrant_l579_579830


namespace ratio_area_triangles_l579_579924

variables {A B C D E : Type} [linear_ordered_field β] [field β]

-- Definitions from conditions
def is_diameter (A B : Type) (c : Type → Type) [circle c] (AB : seg A B c) := true
def is_parallel (CD AB : seg Type Type circle) := true
def angle_AED_is_beta (E : Type) (AED : angle Type β) := true

-- The theorem statement
theorem ratio_area_triangles (A B C D E : Type) (β : real) 
  (diam : is_diameter A B circle) 
  (parallel : is_parallel (seg C D) (seg A B))
  (angle_beta : angle_AED_is_beta E (angle A E D) β) :
  area (triangle C D E) / area (triangle A B E) = real.cos β * real.cos β :=
sorry

end ratio_area_triangles_l579_579924


namespace find_angle_C_find_a_and_b_l579_579929

open Real

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 

-- Conditions
def given_1 : Prop := a * b = c^2 - a^2 - b^2

-- Question 1: Prove angle C
theorem find_angle_C (h : given_1) : ∃ (C : ℝ), C = (2 * π) / 3 :=
sorry

-- Question 2: Find values of a and b given area and c
def given_2 : Prop := (a * b * (sqrt 3 / 2)) / 2 = 2 * sqrt 3 ∧ c = 2 * sqrt 7

theorem find_a_and_b (h2 : given_2) : (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) :=
sorry

end find_angle_C_find_a_and_b_l579_579929


namespace banana_permutations_l579_579493

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579493


namespace BANANA_arrangements_l579_579010

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579010


namespace moles_of_CuCN2_formed_l579_579767

-- Define the stoichiometry of the reaction
def balanced_reaction (n_HCN n_CuSO4 n_CuCN2 : ℕ) :=
  2 * n_HCN + n_CuSO4 = n_CuCN2

-- The proof problem statement
theorem moles_of_CuCN2_formed
  (n_HCN : ℕ)
  (n_CuSO4 : ℕ)
  (n_CuCN2 : ℕ)
  (h1 : n_HCN = 2)
  (h2 : n_CuSO4 = 1)
  (h_balance : balanced_reaction n_HCN n_CuSO4 n_CuCN2) :
  n_CuCN2 = 1 :=
by
  rw [h1, h2] at h_balance
  exact h_balance

sorry

end moles_of_CuCN2_formed_l579_579767


namespace BANANA_permutations_l579_579205

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579205


namespace arrange_BANANA_l579_579293

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579293


namespace permutations_of_BANANA_l579_579388

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579388


namespace find_m_value_l579_579863

variable (a b : Vector ℝ 2)
variable (m : ℝ)

def magnitude (v : Vector ℝ 2) : ℝ := Real.sqrt (v.x * v.x + v.y * v.y)

def angle (u v : Vector ℝ 2) : ℝ :=
  let dot_product := u.x * v.x + u.y * v.y
  let magnitudes_product := magnitude u * magnitude v
  Real.acos (dot_product / magnitudes_product)

theorem find_m_value
  (h1 : magnitude a = Real.sqrt 3)
  (h2 : magnitude b = 2)
  (h3 : angle a b = π / 6)
  (h4 : (a.x - m * b.x) * a.x + (a.y - m * b.y) * a.y = 0) :
  m = 1 := by
  sorry

end find_m_value_l579_579863


namespace coefficient_x4_in_expansion_l579_579896

-- Define the problem statement
theorem coefficient_x4_in_expansion :
  (∀ n : ℕ, (Coefficients of the first three terms in the expansion of (x + 1/(2*x))^n form an arithmetic sequence) →
  (coefficient of the x^4 term in the expansion of (x + 1/(2*x))^n) = 7) :=
begin
  -- Provide the conditions
  -- sorry is used here since the proof is not required
  sorry
end

end coefficient_x4_in_expansion_l579_579896


namespace num_ways_to_arrange_BANANA_l579_579525

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579525


namespace problem_statement_l579_579837

theorem problem_statement :
  (∀ (E : Type) (a b : ℝ) (x y : ℝ), 
     center E = (0, 0) ∧ 
     axes_symmetry E (x, y) ∧ 
     passes_through E (0, -2) (3/2, -1) →
     equation E = (x^2 / 3 + y^2 / 4 = 1) ∧
     (∀ (line : Type) (P M N H : Type)
       (fixed_point : (ℝ × ℝ)), 
       fixed_point = (0, -2) →
       passes_through P (1, -2) ∧
       intersects E line M N ∧
       passes_through_line HN H N →
       passes_through fixed_point HN) :=
sorry

end problem_statement_l579_579837


namespace ellipse_center_origin_axes_symmetry_l579_579841

theorem ellipse_center_origin_axes_symmetry 
  (A B : ℝ × ℝ) (P : ℝ × ℝ)
  (hA : A = (0, -2))
  (hB : B = (3/2, -1))
  (hP : P = (1, -2)) :
  (∃ E : ℝ → ℝ → Prop, 
    (E = λ x y, (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ (x1 y1 x2 y2 : ℝ), 
      E x1 y1 →
      E x2 y2 →
      (x2 - x1) ≠ 0 →
      let M := (x1, y1),
          N := (x2, y2) in
      let T := (2 * x1 + (3/2 - x1) / 2, y1) in
      let H := (3 * x1 / 2 - 3 / 2 + 2, y1) in
      ∃ K : ℝ × ℝ, K = (0, -2) ∧ ((H.1 - N.1) * (K.2 - N.2) = (K.1 - N.1) * (H.2 - N.2)))) 
:= sorry

end ellipse_center_origin_axes_symmetry_l579_579841


namespace arrange_BANANA_l579_579657

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579657


namespace find_a_for_opposite_roots_l579_579847

-- Define the equation and condition using the given problem details
theorem find_a_for_opposite_roots (a : ℝ) 
  (h : ∀ (x : ℝ), x^2 - (a^2 - 2 * a - 15) * x + a - 1 = 0 
    → (∃! (x1 x2 : ℝ), x1 + x2 = 0)) :
  a = -3 := 
sorry

end find_a_for_opposite_roots_l579_579847


namespace BANANA_permutations_l579_579587

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579587


namespace permutations_BANANA_l579_579179

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579179


namespace arrange_BANANA_l579_579644

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579644


namespace permutations_BANANA_l579_579162

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579162


namespace banana_arrangements_l579_579072

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579072


namespace BANANA_permutations_l579_579596

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579596


namespace banana_permutations_l579_579494

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579494


namespace arrange_BANANA_l579_579726

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579726


namespace permutations_BANANA_l579_579186

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579186


namespace banana_arrangements_l579_579059

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579059


namespace cos_sin_neg_9pi_over_4_l579_579739

theorem cos_sin_neg_9pi_over_4 :
  cos (-9 * Real.pi / 4) - sin (-9 * Real.pi / 4) = Real.sqrt 2 :=
by
  sorry

end cos_sin_neg_9pi_over_4_l579_579739


namespace banana_arrangements_l579_579054

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579054


namespace BANANA_arrangements_l579_579009

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579009


namespace derivative_1_derivative_2_l579_579762

-- Problem 1
theorem derivative_1 (x : ℝ) (hx : 0 < x) :
  (deriv (λ x, (1 / (sqrt x)) * (cos x))) x = -(cos x + 2 * x * sin x) / (2 * x * sqrt x) := sorry

-- Problem 2
theorem derivative_2 (x : ℝ) (hx : 0 < x) :
  (deriv (λ x, 5 * x^10 * sin x - 2 * sqrt x * cos x - 9)) x = 
    50 * x^9 * sin x + 5 * x^10 * cos x - (sqrt x * cos x) / x + 2 * sqrt x * sin x := sorry

end derivative_1_derivative_2_l579_579762


namespace BANANA_arrangements_l579_579005

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579005


namespace banana_arrangements_l579_579144

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579144


namespace banana_arrangements_l579_579308

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579308


namespace banana_arrangements_l579_579321

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579321


namespace arrange_BANANA_l579_579727

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579727


namespace BANANA_permutations_l579_579597

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579597


namespace number_of_arrangements_banana_l579_579670

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579670


namespace banana_arrangements_l579_579335

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579335


namespace num_ways_to_arrange_BANANA_l579_579505

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579505


namespace BANANA_permutation_l579_579567

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579567


namespace permutations_BANANA_l579_579157

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579157


namespace BANANA_permutations_l579_579228

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579228


namespace banana_arrangements_l579_579139

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579139


namespace permutations_banana_l579_579094

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579094


namespace find_sin_B_find_a_and_c_l579_579902

variables (A B C a b c : ℝ)
variables (hx : a > 0) (hy : b > 0) (hz : c > 0)
variables (h1 : ∀ A B C, b^2 = a^2 + c^2 - 2*a*c*cos B) 

theorem find_sin_B :
  (∀ A B C a b c : ℝ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (h1: a^2 + b^2 = c^2 + 2 * a * c * cos B) ∧
    (h2: sin(B - A) + cos(A + B) = 0) ∧
    (h3: (sin C / cos C) = (sin A + sin B) / (cos A + cos B))) →
  sin B = (sqrt 2 + sqrt 6) / 4 := 
sorry

theorem find_a_and_c
  (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ) (c : ℝ)
  (hAngleSum : A + B + C = π)
  (hArea : (1 / 2) * a * c * sin B = 3 + sqrt 3)
  (hRatio : a / sin A = b / sin B ∧ b / sin B = c / sin C) :
  a = 2 * sqrt 2 ∧ c = 2 * sqrt 3 := 
sorry

end find_sin_B_find_a_and_c_l579_579902


namespace arrange_BANANA_l579_579280

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579280


namespace BANANA_permutation_l579_579577

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579577


namespace arrange_BANANA_l579_579284

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579284


namespace BANANA_permutation_l579_579545

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579545


namespace arrange_BANANA_l579_579729

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579729


namespace permutations_BANANA_l579_579161

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579161


namespace BANANA_arrangements_l579_579021

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579021


namespace permutations_of_BANANA_l579_579412

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579412


namespace BANANA_arrangements_l579_579000

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579000


namespace weight_difference_at_end_of_year_l579_579907

-- Conditions
def labrador_initial_weight : ℝ := 40
def dachshund_initial_weight : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

-- Question: Difference in weight at the end of the year
theorem weight_difference_at_end_of_year : 
  let labrador_final_weight := labrador_initial_weight * (1 + weight_gain_percentage)
  let dachshund_final_weight := dachshund_initial_weight * (1 + weight_gain_percentage)
  labrador_final_weight - dachshund_final_weight = 35 :=
by
  sorry

end weight_difference_at_end_of_year_l579_579907


namespace BANANA_permutation_l579_579543

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579543


namespace banana_permutations_l579_579363

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579363


namespace BANANA_permutations_l579_579602

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579602


namespace time_math_l579_579973

-- Given conditions as definitions
def time_science : ℕ := 60
def time_literature : ℕ := 40
def total_time : ℕ := 3 * 60

-- Proof statement to find time math, M = 80
theorem time_math : ∃ (M : ℕ), time_science + M + time_literature = total_time ∧ M = 80 :=
by
  use 80
  split
  . simp [time_science, time_literature, total_time]
  . refl

end time_math_l579_579973


namespace count_four_digit_numbers_2033_l579_579875

def sum_of_digits (n : ℕ) : ℕ :=
n.digits.sum

def valid_4_digit_numbers := { n // n.digits.length = 4 ∧ (∀ d ∈ n.digits, d ∈ [2, 0, 3, 3]) ∧ sum_of_digits n ≥ 8 }

theorem count_four_digit_numbers_2033 : (set_to_finset valid_4_digit_numbers).card = 15 := 
sorry

end count_four_digit_numbers_2033_l579_579875


namespace arrange_BANANA_l579_579654

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579654


namespace fraction_to_terminating_decimal_l579_579750

theorem fraction_to_terminating_decimal :
  (47 / (2^3 * 5^4) : ℝ) = 0.0094 := by
  sorry

end fraction_to_terminating_decimal_l579_579750


namespace area_of_rectangle_l579_579997

-- Definitions of the conditions
def length (w : ℝ) : ℝ := 4 * w
def perimeter_eq_200 (w l : ℝ) : Prop := 2 * l + 2 * w = 200

-- Main theorem statement
theorem area_of_rectangle (w l : ℝ) (h1 : length w = l) (h2 : perimeter_eq_200 w l) : l * w = 1600 :=
by
  -- Skip the proof
  sorry

end area_of_rectangle_l579_579997


namespace polynomial_divisibility_l579_579932

def Polynomial (n : ℤ) (coeffs : List ℤ) : ℤ :=
  coeffs.foldr (fun (c, i) acc => acc + c * n^i) 0 (List.zip coeffs (List.range coeffs.length).reverse)

theorem polynomial_divisibility 
  (a_2021 a_2020 ... a_1 a_0 : ℤ) 
  (P : ℤ → ℤ) (n : ℤ) :
  (∀ n, P(n) % 2021 = 0) → (∃ coeffs, 
    Polynomial n coeffs = P n ∧ 
    (¬∀ i, coeffs[i] % 2021 = 0)) :=
by
  sorry

end polynomial_divisibility_l579_579932


namespace banana_permutations_l579_579476

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579476


namespace num_ways_to_arrange_BANANA_l579_579517

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579517


namespace cos_alpha_value_l579_579923

theorem cos_alpha_value (θ α : Real) (P : Real × Real)
  (hP : P = (-3/5, 4/5))
  (hθ : θ = Real.arccos (-3/5))
  (hαθ : α = θ - Real.pi / 3) :
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := 
by 
  sorry

end cos_alpha_value_l579_579923


namespace banana_permutations_l579_579366

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579366


namespace arrangement_count_BANANA_l579_579256

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579256


namespace count_words_without_odd_palindromes_l579_579957

-- Definitions for the problem conditions
variables (n k : ℕ)

-- To ensure valid inputs
variable (hn : 2 ≤ n)
variable (hk : 2 ≤ k)

-- The theorem statement
theorem count_words_without_odd_palindromes (hn : 2 ≤ n) (hk : 2 ≤ k) :
  ∑ (count_words : ℕ), count_words = n^2 * (n-1)^(k-2) :=
sorry

end count_words_without_odd_palindromes_l579_579957


namespace BANANA_arrangements_l579_579015

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579015


namespace permutations_BANANA_l579_579159

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579159


namespace banana_arrangements_l579_579145

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579145


namespace point_distance_on_ellipse_l579_579813

noncomputable theory
open Real

def ellipse (x y : ℝ) := x^2 / 9 + y^2 / 6 = 1

def foci_distance (a b : ℝ) := 2 * sqrt (a^2 - b^2)

def conditions (P : ℝ × ℝ) (cos_theta : ℝ) : Prop :=
  ellipse P.1 P.2 ∧ cos_theta = 3 / 5

theorem point_distance_on_ellipse (P : ℝ × ℝ) (cos_theta : ℝ)
  (h : conditions P cos_theta) : Real.sqrt (P.1^2 + P.2^2) = sqrt(30) / 2 :=
begin
  sorry,
end

end point_distance_on_ellipse_l579_579813


namespace banana_permutations_l579_579350

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579350


namespace banana_arrangements_l579_579138

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579138


namespace permutations_BANANA_l579_579187

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579187


namespace fraction_to_decimal_terminating_l579_579746

theorem fraction_to_decimal_terminating : 
  (47 / (2^3 * 5^4) : ℝ) = 0.5875 :=
by 
  sorry

end fraction_to_decimal_terminating_l579_579746


namespace BANANA_arrangements_l579_579013

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579013


namespace permutations_of_BANANA_l579_579404

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579404


namespace permutations_banana_l579_579076

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579076


namespace BANANA_permutations_l579_579611

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579611


namespace banana_permutations_l579_579362

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579362


namespace permutations_BANANA_l579_579154

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579154


namespace find_a_sub_b_l579_579985

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -2 * x + 3
def h (x : ℝ) : ℝ := f (g x)
def h_inv (x : ℝ) : ℝ := x + 9

theorem find_a_sub_b
  (H1 : ∀ x, h (h_inv x) = x)
  (H2 : h x = -2 * a * x + 3 * a + b)
  : a - b = 7 :=
by
  sorry

end find_a_sub_b_l579_579985


namespace arrangement_count_BANANA_l579_579239

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579239


namespace banana_permutations_l579_579467

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579467


namespace find_angle_MPT_l579_579930

-- Defining the necessary objects and conditions
variables {A B C P M T : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space M] [metric_space T]
variables (Triangle : Triangle A B C)
variables (circumcircle : ∀ {A B C}, Metric.Circumcircle A B C)
variables (AB BC : ℝ) (angleB : ℝ) (BP : ℝ)

-- Given conditions
-- In triangle ABC, side AB is longer than side BC
axiom side_condition : AB > BC
-- Angle B is 40 degrees
axiom angle_B_condition : angleB = 40
-- Point P is taken on side AB such that BP = BC
axiom P_on_AB_condition : BP = BC
-- BM is the angle bisector of triangle ABC
axiom BM_angle_bisector : IsAngleBisector B Triangle P M
-- BM intersects the circumcircle of triangle ABC at point T
axiom BM_intersects_circumcircle : ∃ T, circumcircle A B C

-- To prove: find the angle MPT
theorem find_angle_MPT : ∃ M T, Metric.angle M P T = 20 := sorry

end find_angle_MPT_l579_579930


namespace BANANA_permutations_l579_579612

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579612


namespace banana_permutations_l579_579492

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579492


namespace find_distance_PO_l579_579800

noncomputable def ellipse_and_foci (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 6 = 1) ∧ 
  (∃ (F1 F2 : ℝ × ℝ), 
     let c := sqrt (3) in
     F1 = (c, 0) ∧ F2 = (-c, 0) ∧
     ∃ (P : ℝ × ℝ), 
       (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
       (cos (angle_at P F1 F2) = 3 / 5) ∧ 
       (∃ (O : ℝ × ℝ), O = (0, 0)))

noncomputable def distance_PO (x y : ℝ) : ℝ :=
  sqrt ((x)^2 + (y)^2)

theorem find_distance_PO : ∃ (x y : ℝ), ellipse_and_foci x y → distance_PO x y = sqrt (30) / 2 :=
sorry

end find_distance_PO_l579_579800


namespace arrange_BANANA_l579_579627

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579627


namespace banana_arrangements_l579_579309

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579309


namespace quadratic_real_roots_and_triangle_perimeter_l579_579789

-- Conditions
def quadratic_equation (k : ℝ) : (ℝ × ℝ × ℝ) := (1, -(2*k + 1), 4*(k - 1/2))
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c
def roots_of_quadratic (k : ℝ) : (ℝ × ℝ) :=
  let (a, b, c) := quadratic_equation k in
  ((-b + Real.sqrt (discriminant a b c)) / (2*a), (-b - Real.sqrt (discriminant a b c)) / (2*a))

-- Proof Statement
theorem quadratic_real_roots_and_triangle_perimeter {k : ℝ} :
  let delta := (2*k - 3)^2 in
  let (b, c) := roots_of_quadratic k in
  (delta ≥ 0) ∧ b = 2*k - 1 ∧ c = 2 → (4 = 4) ∧ (b = 2*k-1) ∧ (c = 2) → (4 + 4 + 2 = 10) :=
by {
  sorry
}

end quadratic_real_roots_and_triangle_perimeter_l579_579789


namespace arrangement_count_BANANA_l579_579267

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579267


namespace jake_weight_l579_579889

variable (J K : ℕ)

-- Conditions given in the problem
axiom h1 : J - 8 = 2 * K
axiom h2 : J + K = 293

-- Statement to prove
theorem jake_weight : J = 198 :=
by
  sorry

end jake_weight_l579_579889


namespace arrangement_count_BANANA_l579_579251

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579251


namespace solve_for_y_l579_579982

theorem solve_for_y (y : ℝ) (h : (4/7) * (1/5) * y - 2 = 14) : y = 140 := 
sorry

end solve_for_y_l579_579982


namespace weight_difference_proof_l579_579909

theorem weight_difference_proof
  (labrador_start_weight : ℕ) (dachshund_start_weight : ℕ)
  (weight_gain_percentage : ℕ)
  (labrador_start_weight = 40)
  (dachshund_start_weight = 12)
  (weight_gain_percentage = 25) :
  (labrador_start_weight + labrador_start_weight * weight_gain_percentage / 100) -
  (dachshund_start_weight + dachshund_start_weight * weight_gain_percentage / 100) =
  35 := 
  sorry

end weight_difference_proof_l579_579909


namespace probability_x_greater_than_4y_l579_579969

theorem probability_x_greater_than_4y : 
  (∃ (p : ℝ × ℝ), p.1 ∈ set.Icc 0 3000 ∧ p.2 ∈ set.Icc 0 4000 ∧ p.1 > 4 * p.2) 
  → (∃ (A R : ℝ), A = 1125000 ∧ R = 12000000 ∧ A / R = 3 / 32) :=
by
  intro h
  use 1125000, 12000000
  have hA : 1125000 = 1 / 2 * 3000 * 750, by norm_num
  have hR : 12000000 = 3000 * 4000, by norm_num
  have hAR: 1125000 / 12000000 = 3 / 32, by norm_num
  exact ⟨hA, hR, hAR⟩

end probability_x_greater_than_4y_l579_579969


namespace permutations_BANANA_l579_579156

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579156


namespace BANANA_permutations_l579_579619

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579619


namespace permutations_of_BANANA_l579_579423

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579423


namespace arrange_BANANA_l579_579733

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579733


namespace banana_arrangements_l579_579053

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579053


namespace supplements_delivered_l579_579960

-- Define the conditions as given in the problem
def total_medicine_boxes : ℕ := 760
def vitamin_boxes : ℕ := 472

-- Define the number of supplement boxes
def supplement_boxes : ℕ := total_medicine_boxes - vitamin_boxes

-- State the theorem to be proved
theorem supplements_delivered : supplement_boxes = 288 :=
by
  -- The actual proof is not required, so we use "sorry"
  sorry

end supplements_delivered_l579_579960


namespace sequence_propositions_l579_579792

theorem sequence_propositions (a : ℕ → ℝ) (h_seq : a 1 > a 2 ∧ a 2 > a 3 ∧ a 3 > a 4 ∧ a 4 ≥ 0) 
  (h_sub : ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 4 → ∃ k, a i - a j = a k) :
  (∀ k, ∃ d, a k = a 1 - d * (k - 1)) ∧
  (∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ i * a i = j * a j) ∧
  (∃ i, a i = 0) :=
by
  sorry

end sequence_propositions_l579_579792


namespace banana_permutations_l579_579485

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579485


namespace arrangement_count_BANANA_l579_579238

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579238


namespace f_at_neg_one_l579_579899

def f (x : ℝ) : ℝ := sorry

theorem f_at_neg_one (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 :=
by sorry

end f_at_neg_one_l579_579899


namespace BANANA_permutations_l579_579227

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579227


namespace banana_arrangements_l579_579044

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579044


namespace BANANA_permutations_l579_579600

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579600


namespace arrange_BANANA_l579_579620

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579620


namespace intersection_eq_singleton_zero_l579_579825

open Set

namespace IntersectionProblem

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_eq_singleton_zero : M ∩ coe '' N = {0} :=
by
  -- Proof to be constructed
  sorry

end IntersectionProblem

end intersection_eq_singleton_zero_l579_579825


namespace permutations_of_BANANA_l579_579417

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579417


namespace arrange_BANANA_l579_579621

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579621


namespace number_of_arrangements_banana_l579_579692

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579692


namespace permutations_of_BANANA_l579_579414

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579414


namespace banana_arrangements_l579_579150

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579150


namespace banana_permutations_l579_579368

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579368


namespace number_of_ways_to_divide_friends_l579_579880

theorem number_of_ways_to_divide_friends (n : ℕ) (k : ℕ) : n = 8 → k = 3 → (k ^ n) = 6561 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  exact eq.refl 6561

end number_of_ways_to_divide_friends_l579_579880


namespace find_angle_between_vectors_l579_579832

variable {a b : ℝ}
variable {θ : ℝ}

def vector_magnitude (v : ℝ) := Real.sqrt (v * v)

# Definitions of magnitude for vectors |a| and |b|
def a_magnitude : ℝ := 4
def b_magnitude : ℝ := 3

# Definition of combined vector magnitude
def a_b_magnitude : ℝ := Real.sqrt 13

-- Conditions for the problem
axiom cond1 : vector_magnitude a = a_magnitude
axiom cond2 : vector_magnitude b = b_magnitude
axiom cond3 : vector_magnitude (a - b) = a_b_magnitude

-- The problem statement to be proved
theorem find_angle_between_vectors : 
  θ = 60 := by 
sorry

end find_angle_between_vectors_l579_579832


namespace min_distance_point_to_line_l579_579766

theorem min_distance_point_to_line :
    ∀ (x y : ℝ), (x^2 + y^2 - 6 * x - 4 * y + 12 = 0) -> 
    (3 * x + 4 * y - 2 = 0) -> 
    ∃ d: ℝ, d = 2 :=
by sorry

end min_distance_point_to_line_l579_579766


namespace permutations_banana_l579_579100

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579100


namespace BANANA_permutation_l579_579561

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579561


namespace BANANA_permutations_l579_579589

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579589


namespace BANANA_permutations_l579_579220

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579220


namespace banana_permutations_l579_579475

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579475


namespace permutations_of_BANANA_l579_579394

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579394


namespace find_distance_PO_l579_579801

noncomputable def ellipse_and_foci (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 6 = 1) ∧ 
  (∃ (F1 F2 : ℝ × ℝ), 
     let c := sqrt (3) in
     F1 = (c, 0) ∧ F2 = (-c, 0) ∧
     ∃ (P : ℝ × ℝ), 
       (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
       (cos (angle_at P F1 F2) = 3 / 5) ∧ 
       (∃ (O : ℝ × ℝ), O = (0, 0)))

noncomputable def distance_PO (x y : ℝ) : ℝ :=
  sqrt ((x)^2 + (y)^2)

theorem find_distance_PO : ∃ (x y : ℝ), ellipse_and_foci x y → distance_PO x y = sqrt (30) / 2 :=
sorry

end find_distance_PO_l579_579801


namespace arrange_BANANA_l579_579285

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579285


namespace daniel_pages_to_read_l579_579915

-- Definitions from conditions
def total_pages : ℕ := 980
def daniel_read_time_per_page : ℕ := 50
def emma_read_time_per_page : ℕ := 40

-- The theorem that states the solution
theorem daniel_pages_to_read (d : ℕ) :
  d = 436 ↔ daniel_read_time_per_page * d = emma_read_time_per_page * (total_pages - d) :=
by sorry

end daniel_pages_to_read_l579_579915


namespace dot_product_value_l579_579833

variables (e1 e2 : EuclideanSpace ℝ (Fin 2)) (α : ℝ)
noncomputable def e1_unit : Prop := ∥e1∥ = 1
noncomputable def e2_unit : Prop := ∥e2∥ = 1
noncomputable def angle_α : Prop := real.angle.cos α = -1/5

noncomputable def a : EuclideanSpace ℝ (Fin 2) := 2 • e1 - e2
noncomputable def b : EuclideanSpace ℝ (Fin 2) := e1 + 3 • e2

theorem dot_product_value (h1 : e1_unit e1) (h2 : e2_unit e2) (h3 : angle_α α) :
  inner a b = -2 := by sorry

end dot_product_value_l579_579833


namespace num_ways_to_arrange_BANANA_l579_579511

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579511


namespace po_distance_l579_579816

-- Define the ellispse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 6) = 1

-- Define the condition for cosine of the angle between F1, P, and F2
def cos_angle (cos_val : ℝ) : Prop := cos_val = 3/5

-- Define the proof problem in Lean
theorem po_distance (x y : ℝ) (hx : ellipse x y) (cos_val : ℝ) (h_cos : cos_angle cos_val) : (pow (sqrt ((15:ℝ)/2)) 2) = 30/4 :=
by
  sorry

end po_distance_l579_579816


namespace arrange_BANANA_l579_579623

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579623


namespace banana_permutations_l579_579473

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579473


namespace banana_arrangements_l579_579049

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579049


namespace algebra_problem_l579_579883

variable (a : ℝ)

-- Condition: Given (a + 1/a)^3 = 4
def condition : Prop := (a + 1/a)^3 = 4

-- Statement: Prove a^4 + 1/a^4 = -158/81
theorem algebra_problem (h : condition a) : a^4 + 1/a^4 = -158/81 := 
sorry

end algebra_problem_l579_579883


namespace ratio_of_x_to_2y_is_1_l579_579771

variable {x y r : ℝ}

theorem ratio_of_x_to_2y_is_1.5 (h1 : (7 * x + 5 * y) / (x - 2 * y) = 26) (h2 : x / (2 * y) = r) : r = 1.5 := 
sorry

end ratio_of_x_to_2y_is_1_l579_579771


namespace BANANA_arrangements_correct_l579_579459

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579459


namespace number_of_arrangements_banana_l579_579662

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579662


namespace BANANA_arrangements_l579_579002

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579002


namespace BANANA_arrangements_l579_579022

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579022


namespace number_of_arrangements_banana_l579_579676

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579676


namespace num_ways_to_arrange_BANANA_l579_579538

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579538


namespace banana_permutations_l579_579381

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579381


namespace BANANA_permutations_l579_579588

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579588


namespace find_M_plus_N_l579_579882

theorem find_M_plus_N (M N : ℕ) (h1 : 3 / 5 = M / 30) (h2 : 3 / 5 = 90 / N) : M + N = 168 := 
by
  sorry

end find_M_plus_N_l579_579882


namespace arrange_BANANA_l579_579279

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579279


namespace num_ways_to_arrange_BANANA_l579_579519

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579519


namespace po_distance_l579_579817

-- Define the ellispse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 6) = 1

-- Define the condition for cosine of the angle between F1, P, and F2
def cos_angle (cos_val : ℝ) : Prop := cos_val = 3/5

-- Define the proof problem in Lean
theorem po_distance (x y : ℝ) (hx : ellipse x y) (cos_val : ℝ) (h_cos : cos_angle cos_val) : (pow (sqrt ((15:ℝ)/2)) 2) = 30/4 :=
by
  sorry

end po_distance_l579_579817


namespace value_of_x_squared_minus_y_squared_l579_579888

theorem value_of_x_squared_minus_y_squared 
  (x y : ℚ)
  (h1 : x + y = 5 / 8) 
  (h2 : x - y = 3 / 8) :
  x^2 - y^2 = 15 / 64 :=
by 
  sorry

end value_of_x_squared_minus_y_squared_l579_579888


namespace previous_year_height_l579_579966

-- Define the height condition: she was 159 cm on her 15th birthday
def height_on_15th_birthday := 159

-- Define the growth rate: 6%
def growth_rate := 1.06

-- Define the height the previous year
def height_previous_year := height_on_15th_birthday / growth_rate

-- The proof statement: height the previous year equals 150 cm
theorem previous_year_height : height_previous_year = 150 := by
  sorry

end previous_year_height_l579_579966


namespace banana_permutations_l579_579364

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579364


namespace arrange_BANANA_l579_579301

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579301


namespace november_has_five_sundays_if_october_has_five_wednesdays_l579_579986

def has_five_wednesdays (ys: List Nat) : Prop :=
  ys.length = 31 ∧ (List.countp (λ x => x % 7 = 3) ys) = 5

def november_has_five_sundays (ys: List Nat) : Prop :=
  (List.countp (λ x => x % 7 = 0) ys) = 5

theorem november_has_five_sundays_if_october_has_five_wednesdays :
  ∀ ys: List Nat, (has_five_wednesdays ys ∧ ys.length = 31) → ∃ zs, zs.length = 30 ∧ november_has_five_sundays zs :=
begin
  sorry
end

end november_has_five_sundays_if_october_has_five_wednesdays_l579_579986


namespace permutations_banana_l579_579090

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579090


namespace banana_arrangements_l579_579128

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579128


namespace permutations_of_BANANA_l579_579424

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579424


namespace analytical_expression_of_f_inequality_for_f_l579_579781

def f (x : ℝ) (a : ℝ) (b : ℝ) := 2 * Real.log x + a * x + b / x
def f' (x : ℝ) (a : ℝ) (b : ℝ) := ((2 : ℝ) / x) - a - (b / x^2)

theorem analytical_expression_of_f : 
  ∃ a b, a - b = -5 ∧ a + b = -3 ∧ ∀ x, f x a b = 2 * Real.log x - 4 * x + 1 / x :=
by {
  let a := -4,
  let b := 1,
  use a,
  use b,
  simp [f],
  split,
  exact eq.refl (-4 - 1),
  split,
  exact eq.refl (-4 + 1),
  intros,
  refl,
}

theorem inequality_for_f (x : ℝ) (h1 : 1 ≤ x) : 
  let fx := 2 * Real.log x - 4 * x + 1 / x in
  let fx' := ((2 : ℝ) / x) - 4 - ((1 : ℝ) / x^2) in
  fx - fx' ≤ -2 * x + 1 / x + 1 :=
by {
  let fx := 2 * Real.log x - 4 * x + 1 / x,
  let fx' := 2 / x - 4 - 1 / x^2,
  calc
    fx - fx' = 2 * Real.log x - 4 * x + 1 / x - (2 / x - 4 - 1 / x^2) : by simp
          ... = 2 * Real.log x - 4 * x + 1 / x - 2 / x + 4 + 1 / x^2 : by simp
          ... = 2 * Real.log x - 2 * x - 2 / x + 1 / x^2 + 3 : by ring
          ... ≤ -2 * x + 1 / x + 1 : sorry
}


end analytical_expression_of_f_inequality_for_f_l579_579781


namespace banana_permutations_l579_579369

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579369


namespace banana_arrangements_l579_579046

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579046


namespace permutations_banana_l579_579108

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579108


namespace BANANA_arrangements_correct_l579_579432

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579432


namespace tangent_of_angle_between_vectors_l579_579862

variable (a b : EuclideanSpace ℝ (Fin 2))

theorem tangent_of_angle_between_vectors
  (h1 : inner a (a + b) = 5)
  (h2 : ∥a∥ = 2)
  (h3 : ∥b∥ = 1) :
  Real.tan (Real.arccos ((inner a b) / (∥a∥ * ∥b∥))) = Real.sqrt 3 :=
by
  sorry

end tangent_of_angle_between_vectors_l579_579862


namespace banana_permutations_l579_579489

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579489


namespace arrange_BANANA_l579_579724

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579724


namespace arrange_BANANA_l579_579288

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579288


namespace arrange_BANANA_l579_579658

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579658


namespace distance_from_origin_ellipse_point_l579_579796

theorem distance_from_origin_ellipse_point :
  ∃ P : ℝ × ℝ, 
    (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
    (let c := real.sqrt(3) in 
    let F1 := (-c, 0) in 
    let F2 := (c, 0) in 
    real.cos (
      vector.angle 
       ⟨P.1 - F1.1, P.2 - F1.2⟩ 
       ⟨P.1 - F2.1, P.2 - F2.2⟩
    ) = 3/5) ∧ 
  real.sqrt (P.1^2 + P.2^2) = real.sqrt(30) / 2 := 
sorry

end distance_from_origin_ellipse_point_l579_579796


namespace num_ways_to_arrange_BANANA_l579_579503

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579503


namespace num_ways_to_arrange_BANANA_l579_579514

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579514


namespace BANANA_permutations_l579_579212

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579212


namespace tunnel_depth_l579_579993

theorem tunnel_depth (topWidth : ℝ) (bottomWidth : ℝ) (area : ℝ) (h : ℝ)
  (h1 : topWidth = 15)
  (h2 : bottomWidth = 5)
  (h3 : area = 400)
  (h4 : area = (1 / 2) * (topWidth + bottomWidth) * h) :
  h = 40 := 
sorry

end tunnel_depth_l579_579993


namespace banana_arrangements_l579_579047

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579047


namespace BANANA_permutation_l579_579553

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579553


namespace arrange_BANANA_l579_579709

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579709


namespace arrange_BANANA_l579_579715

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579715


namespace permutations_banana_l579_579102

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579102


namespace hyperbola_eccentricity_l579_579858

-- Define the conditions for the hyperbola
variables {a b c : ℝ}
hypothesis h1 : a > 0
hypothesis h2 : b > 0
hypothesis h3 : c = sqrt 2 * a

-- Define the focus of the hyperbola
def F2 := (c, 0)

-- The equation of the asymptote
def asymptote := λ x : ℝ, b / a * x

-- Define point N
def N := (a^2 / c, a * b / c)

-- Define point M
def M := (0, a * c / b)

-- Midpoint condition (simplified and derived from the given problem)
def midpoint_condition := λ x : ℝ, (c + a * c / b) / 2 = a^2 / c

-- Define the eccentricity
def eccentricity := c / a

-- The Lean theorem statement to prove the eccentricity
theorem hyperbola_eccentricity : eccentricity = sqrt 2 :=
by sorry

end hyperbola_eccentricity_l579_579858


namespace arrange_BANANA_l579_579298

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579298


namespace sequence_sum_100_l579_579866

def a (n : ℕ) : ℚ := 1 / (n^2 + n)

theorem sequence_sum_100 :
  (∑ i in Finset.range 100, a (i + 1)) = 100 / 101 :=
by
  sorry

end sequence_sum_100_l579_579866


namespace permutations_BANANA_l579_579167

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579167


namespace banana_arrangements_l579_579045

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579045


namespace permutations_of_BANANA_l579_579421

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579421


namespace BANANA_arrangements_correct_l579_579457

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579457


namespace arrangement_count_BANANA_l579_579250

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579250


namespace total_strength_of_college_l579_579914

-- Declare the variables for number of students playing each sport
variables (C B Both : ℕ)

-- Given conditions in the problem
def cricket_players : ℕ := 500
def basketball_players : ℕ := 600
def both_players : ℕ := 220

-- Theorem stating the total strength of the college
theorem total_strength_of_college (h_C : C = cricket_players) 
                                  (h_B : B = basketball_players) 
                                  (h_Both : Both = both_players) : 
                                  C + B - Both = 880 :=
by
  sorry

end total_strength_of_college_l579_579914


namespace arrange_BANANA_l579_579645

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579645


namespace compare_a_b_c_l579_579944

theorem compare_a_b_c : 
  let a := Real.pow 0.5 3
  let b := Real.pow 3 0.5
  let c := Real.logb 0.5 3
  c < a ∧ a < b := 
by simp [Real.pow, Real.logb];
   sorry

end compare_a_b_c_l579_579944


namespace banana_arrangements_l579_579120

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579120


namespace PO_equals_l579_579805

def ellipse (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 6 = 1

def F1_c (a : ℝ) : ℝ := √(a^2 - 6)
def F2_c (a : ℝ) : ℝ := √(a^2 - 6)

def cos_angle_F1PF2 : ℝ := 3 / 5

noncomputable def PO := (P : ℝ × ℝ) := 
  1 / 2 * ((P.1 - F1_c 3)^2 + (P.2 - F2_c 3)^2 - 2 * (P.1 - F1_c 3) * (P.2 - F2_c 3) * cos_angle_F1PF2)

theorem PO_equals : ∀ (P : ℝ × ℝ), ellipse P.1 P.2 → ∥P∥ = √(30) / 2 :=
by
  intros P P_ellipse_condition
  sorry

end PO_equals_l579_579805


namespace parallelogram_area_l579_579760

theorem parallelogram_area (base height : ℝ) (h_base : base = 36) (h_height : height = 24) : 
    base * height = 864 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l579_579760


namespace BANANA_permutations_l579_579598

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579598


namespace permutations_of_BANANA_l579_579387

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579387


namespace arrange_BANANA_l579_579297

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579297


namespace arrange_BANANA_l579_579302

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579302


namespace BANANA_arrangements_l579_579003

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579003


namespace banana_permutations_l579_579372

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579372


namespace locus_of_points_l579_579822

open EuclideanGeometry

-- Define the points A and B in 2D space.
variables (A B : EuclideanSpace ℝ (fin 2))

-- Define the perpendicular bisector of A and B.
def perp_bisector (A B : EuclideanSpace ℝ (fin 2)) : AffineSubspace ℝ (EuclideanSpace ℝ (fin 2)) :=
{ carrier := { M | dist A M = dist B M },
  direction := sorry }

-- Define the half-plane containing point B excluding the perpendicular bisector.
def half_plane_containing_B_excluding_bisector (A B : EuclideanSpace ℝ (fin 2)) : Set (EuclideanSpace ℝ (fin 2)) :=
{ M | dist A M > dist B M }

-- The main theorem statement.
theorem locus_of_points (A B : EuclideanSpace ℝ (fin 2)) :
  { M | dist A M > dist B M } = half_plane_containing_B_excluding_bisector A B :=
sorry

end locus_of_points_l579_579822


namespace permutations_banana_l579_579077

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579077


namespace find_angle_l579_579901

theorem find_angle (A : ℝ) (h : 0 < A ∧ A < π) 
  (c : 4 * π * Real.sin A - 3 * Real.arccos (-1/2) = 0) :
  A = π / 6 ∨ A = 5 * π / 6 :=
sorry

end find_angle_l579_579901


namespace arrange_BANANA_l579_579699

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579699


namespace arrange_BANANA_l579_579700

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579700


namespace DP_eq_QR_l579_579931

-- Defining points and the function of incircle touch points
variables {A B C D P Q R : Point}
variables [triangle ABC] [on_segment D A B]
variables [on_incircle ABC P (line_segment A B)]
variables [on_incircle ADC Q (line_segment D C)]
variables [on_incircle DBC R (line_segment D C)]

noncomputable def DP : Real :=
  abs ((dist A P) - (dist A D))

noncomputable def DQ : Real :=
  (dist D Q)

noncomputable def DR : Real :=
  (dist D R)

noncomputable def QR : Real :=
  abs (DR - DQ)

theorem DP_eq_QR : DP = QR :=
by 
  sorry -- Proof here

end DP_eq_QR_l579_579931


namespace arrangement_count_BANANA_l579_579243

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579243


namespace banana_arrangements_l579_579068

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579068


namespace negation_of_prop_p_l579_579893

open Classical

variable (p : Prop)

def prop_p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_prop_p : ¬prop_p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end negation_of_prop_p_l579_579893


namespace banana_arrangements_l579_579117

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579117


namespace max_value_of_product_of_distances_to_foci_l579_579826

/--
Let F1 and F2 be the foci of the ellipse x^2 / 25 + y^2 / 9 = 1.
Let P be any point on the ellipse.
Prove that the maximum value of |PF1| * |PF2| is 25.
-/
theorem max_value_of_product_of_distances_to_foci :
  ∀ (P : ℝ × ℝ), (P.1 ^ 2 / 25 + P.2 ^ 2 / 9 = 1) → 
  ∃ max_val, max_val = 25 ∧ (∀ Q, Q = (P.1, P.2) → |PF1 Q| * |PF2 Q| ≤ max_val) :=
sorry

end max_value_of_product_of_distances_to_foci_l579_579826


namespace arrange_BANANA_l579_579631

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579631


namespace part1_solution_set_part2_range_of_m_l579_579946
noncomputable theory

-- Defining the function f with parameter m and variable x
def f (m : ℝ) (x : ℝ) := (m + 1) * x^2 - m * x - 1

-- Part 1: Prove the solution to f(x) ≤ 0 when m = -3 is { x | x ≥ 1 ∨ x ≤ 1/2 }
theorem part1_solution_set (x : ℝ) : f (-3) x ≤ 0 ↔ x ≥ 1 ∨ x ≤ 1 / 2 :=
by
  sorry

-- Part 2: Prove the range of real numbers for m when the inequality f(x) + m > 0 has no solution is (-∞, -2√3/3]
theorem part2_range_of_m (m : ℝ) : (∀ x : ℝ, f m x + m > 0 → false) ↔ m ≤ -2 * (real.sqrt 3) / 3 :=
by
  sorry

end part1_solution_set_part2_range_of_m_l579_579946


namespace arrange_BANANA_l579_579283

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579283


namespace cos_of_Z_l579_579920

open Real

theorem cos_of_Z (X Y Z : ℝ) (h₁ : X = 8) (h₂ : Y = 17) (h₃ : Z = 90)
                (h₄ : ∃ XY YZ XZ : ℝ, XY = X ∧ YZ = Y ∧ ∠X = π / 2) :
  cos (atan (X / Y)) = 15 / 17 :=
by sorry

end cos_of_Z_l579_579920


namespace arrangement_count_BANANA_l579_579244

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579244


namespace banana_arrangements_l579_579146

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579146


namespace BANANA_permutations_l579_579590

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579590


namespace BANANA_arrangements_correct_l579_579460

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579460


namespace banana_arrangements_l579_579142

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579142


namespace dot_triple_product_equals_one_l579_579943

variables {E : Type*} [inner_product_space ℝ E] [finite_dimensional ℝ E]
variables (u v w : E)

-- Define the conditions
def is_unit_vector (v : E) : Prop := ∥v∥ = 1
def orthogonal (u v : E) : Prop := ⟪u, v⟫ = 0
def w_condition (u v : E) : E := (u ×ₗ v) + 2 • u

-- Construct the theorem to prove that the dot product of u and (v cross w) equals 1
theorem dot_triple_product_equals_one
  (hu : is_unit_vector u) 
  (hv : is_unit_vector v) 
  (h_orth : orthogonal u v) 
  (hw : w = w_condition u v) : ⟪u, v ×ₗ w⟫ = 1 :=
sorry

end dot_triple_product_equals_one_l579_579943


namespace compute_floor_T_squared_l579_579953

def T : ℝ := ∑ i in (Finset.range 1007), real.sqrt(1 + 1 / ( (2 * i + 1 : ℝ) ^ 2 ) + 1 / ( (2 * i + 3 : ℝ) ^ 2 ))

theorem compute_floor_T_squared : ⌊T ^ 2⌋ = 1016064 := by
  sorry

end compute_floor_T_squared_l579_579953


namespace banana_arrangements_l579_579065

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579065


namespace permutations_banana_l579_579091

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579091


namespace projection_of_a_in_b_direction_is_neg5_l579_579869

open EuclideanGeometry

-- Definitions based on conditions in the problem statement
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-9 / 2, -6)

-- Statement to prove that the projection of vector a in the direction of vector b equals -5
theorem projection_of_a_in_b_direction_is_neg5 (h_collinear : ∃ k : ℝ, b = (k * 3, k * 4)) :
  (a.1 * b.1 + a.2 * b.2) / real.sqrt (b.1 ^ 2 + b.2 ^ 2) = -5 :=
by
  sorry

end projection_of_a_in_b_direction_is_neg5_l579_579869


namespace BANANA_permutations_l579_579586

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579586


namespace arrangement_count_BANANA_l579_579233

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579233


namespace BANANA_permutations_l579_579601

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579601


namespace banana_arrangements_l579_579038

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579038


namespace arrange_BANANA_l579_579286

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579286


namespace banana_arrangements_l579_579057

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579057


namespace permutations_of_BANANA_l579_579391

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579391


namespace num_ways_to_arrange_BANANA_l579_579529

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579529


namespace number_of_arrangements_banana_l579_579694

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579694


namespace banana_permutations_l579_579490

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579490


namespace arrangement_count_BANANA_l579_579247

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579247


namespace arrangement_count_BANANA_l579_579268

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579268


namespace banana_arrangements_l579_579320

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579320


namespace number_of_arrangements_banana_l579_579691

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579691


namespace BANANA_permutation_l579_579580

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579580


namespace jo_liam_sum_difference_l579_579934

theorem jo_liam_sum_difference : 
  let jo_sum := (50 * 51) / 2 in
  let liam_sum := (0 + 0 + 5 + 5 + 5) + 
                  (10 + 10 + 10 + 10 + 10) + 
                  (15 + 15 + 15 + 15 + 15) + 
                  (20 + 20 + 20 + 20 + 20) + 
                  (25 + 25 + 25 + 25 + 25) + 
                  (30 + 30 + 30 + 30 + 30) + 
                  (35 + 35 + 35 + 35 + 35) + 
                  (40 + 40 + 40 + 40 + 40) + 
                  (45 + 45 + 45 + 45 + 45) + 
                  (50 + 50 + 50) in
  abs (liam_sum - jo_sum) = 90 :=
by
  let jo_sum := (50 * 51) / 2
  let liam_sum := (0 + 0 + 5 + 5 + 5) + 
                  (10 + 10 + 10 + 10 + 10) + 
                  (15 + 15 + 15 + 15 + 15) + 
                  (20 + 20 + 20 + 20 + 20) + 
                  (25 + 25 + 25 + 25 + 25) + 
                  (30 + 30 + 30 + 30 + 30) + 
                  (35 + 35 + 35 + 35 + 35) + 
                  (40 + 40 + 40 + 40 + 40) + 
                  (45 + 45 + 45 + 45 + 45) + 
                  (50 + 50 + 50)
  sorry

end jo_liam_sum_difference_l579_579934


namespace permutations_BANANA_l579_579190

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579190


namespace permutations_BANANA_l579_579178

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579178


namespace banana_arrangements_l579_579311

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579311


namespace oscar_leap_vs_elmer_stride_l579_579964

theorem oscar_leap_vs_elmer_stride:
  let gaps := 50
  let total_distance := 8000
  let elmer_strides_per_gap := 56
  let oscar_leaps_per_gap := 16
  let elmer_total_strides := gaps * elmer_strides_per_gap
  let oscar_total_leaps := gaps * oscar_leaps_per_gap
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  in 
  oscar_leap_length - elmer_stride_length = 7 := 
by
  sorry

end oscar_leap_vs_elmer_stride_l579_579964


namespace comparison_l579_579784

noncomputable def x := sorry
def a (x : ℝ) := Real.log x
def b (x : ℝ) := 2 * Real.log x
def c (x : ℝ) := (Real.log x) ^ 3

theorem comparison (hx : x ∈ Set.Ioo (Real.exp (-1)) 1) : b x < a x ∧ a x < c x :=
by
  sorry

end comparison_l579_579784


namespace BANANA_arrangements_l579_579006

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579006


namespace PO_equals_l579_579808

def ellipse (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 6 = 1

def F1_c (a : ℝ) : ℝ := √(a^2 - 6)
def F2_c (a : ℝ) : ℝ := √(a^2 - 6)

def cos_angle_F1PF2 : ℝ := 3 / 5

noncomputable def PO := (P : ℝ × ℝ) := 
  1 / 2 * ((P.1 - F1_c 3)^2 + (P.2 - F2_c 3)^2 - 2 * (P.1 - F1_c 3) * (P.2 - F2_c 3) * cos_angle_F1PF2)

theorem PO_equals : ∀ (P : ℝ × ℝ), ellipse P.1 P.2 → ∥P∥ = √(30) / 2 :=
by
  intros P P_ellipse_condition
  sorry

end PO_equals_l579_579808


namespace BANANA_permutations_l579_579208

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579208


namespace arrange_BANANA_l579_579290

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579290


namespace tan_alpha_sin_cos_half_alpha_l579_579778

variable (α : ℝ)

-- Conditions given in the problem
def cond1 : Real.sin α = 1 / 3 := sorry
def cond2 : 0 < α ∧ α < Real.pi := sorry

-- Lean proof that given the conditions, the solutions are as follows:
theorem tan_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = Real.sqrt 2 / 4 ∨ Real.tan α = - Real.sqrt 2 / 4 := sorry

theorem sin_cos_half_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.sin (α / 2) + Real.cos (α / 2) = 2 * Real.sqrt 3 / 3 := sorry

end tan_alpha_sin_cos_half_alpha_l579_579778


namespace number_of_arrangements_banana_l579_579669

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579669


namespace BANANA_arrangements_correct_l579_579428

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579428


namespace banana_permutations_l579_579361

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579361


namespace permutations_banana_l579_579080

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579080


namespace banana_arrangements_l579_579345

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579345


namespace arrangement_count_BANANA_l579_579236

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579236


namespace arrangement_count_BANANA_l579_579257

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579257


namespace banana_arrangements_l579_579050

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579050


namespace permutations_banana_l579_579082

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579082


namespace banana_permutations_l579_579378

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579378


namespace banana_arrangements_l579_579051

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579051


namespace arrange_BANANA_l579_579708

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579708


namespace banana_arrangements_l579_579151

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579151


namespace part1_max_min_part2_inequality_l579_579856

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x
noncomputable def g (x : ℝ) : ℝ := (2/3) * x^3 + (1/2) * x^2

theorem part1_max_min : 
  (∀ (x : ℝ), x ∈ set.Icc 1 Real.exp 1 → f x ≤ f (Real.exp 1)) ∧ (f 1 = 1) := sorry

theorem part2_inequality : 
  ∀ (x : ℝ), x > 1 → f x < g x := sorry

end part1_max_min_part2_inequality_l579_579856


namespace lou_runs_3_miles_l579_579958

-- Conditions
variables (distance_per_lap : ℝ) (rosie_laps : ℕ)
variable (speed_ratio : ℝ) -- Ratio of Rosie's speed to Lou's speed

-- Definitions based on the conditions
def lou_laps (rosie_laps : ℕ) (speed_ratio : ℝ) : ℕ := rosie_laps / speed_ratio.to_nat

def total_distance (laps : ℕ) (distance_per_lap : ℝ) : ℝ := laps * distance_per_lap

-- Theorem statement: Lou runs 3 miles
theorem lou_runs_3_miles :
  total_distance (lou_laps rosie_laps speed_ratio) distance_per_lap = 3 :=
by
  -- Typical initial values: distance_per_lap = 0.25, rosie_laps = 24, speed_ratio = 2,
  -- Plugging these values, total_distance = 3 should be proven
  have h_distance_per_lap : distance_per_lap = 0.25 := by sorry,
  have h_rosie_laps : rosie_laps = 24 := by sorry,
  have h_speed_ratio : speed_ratio = 2 := by sorry,
  sorry

end lou_runs_3_miles_l579_579958


namespace banana_arrangements_l579_579042

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l579_579042


namespace angle_equality_l579_579991

noncomputable theory

-- Define the problem by setting up the points and circles
variables {S1 S2 S3 : Type} 
variables (A B C D K : Type)
variables (intersects_at : Circle S1 -> Circle S2 -> Point -> Point -> Prop)
variables (touches_at : Circle S3 -> Circle S1 -> Point -> Prop)
variables (touches_at' : Circle S3 -> Circle S2 -> Point -> Prop)
variables (chord_midpoint : Line -> Point -> Circle S3 -> Point -> Prop)
variables (angle : Point -> Point -> Point -> ℝ)

-- Assume the conditions
axiom h1 : intersects_at S1 S2 A B
axiom h2 : touches_at S3 S1 C
axiom h3 : touches_at' S3 S2 D
axiom h4 : chord_midpoint (line_through A B) (midpoint A B) S3 K

-- The goal is to prove the angle equality
theorem angle_equality : angle C K A = angle D K A :=
sorry

end angle_equality_l579_579991


namespace banana_permutations_l579_579348

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579348


namespace permutations_BANANA_l579_579176

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l579_579176


namespace banana_permutations_l579_579371

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579371


namespace BANANA_permutations_l579_579215

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579215


namespace BANANA_permutations_l579_579192

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l579_579192


namespace BANANA_arrangements_correct_l579_579442

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579442


namespace arrange_BANANA_l579_579722

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579722


namespace banana_arrangements_l579_579134

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l579_579134


namespace average_percent_score_l579_579961

theorem average_percent_score :
    let students := 120
    let score_95 := 95 * 12
    let score_85 := 85 * 24
    let score_75 := 75 * 30
    let score_65 := 65 * 20
    let score_55 := 55 * 18
    let score_45 := 45 * 10
    let score_35 := 35 * 6
    let total_score := score_95 + score_85 + score_75 + score_65 + score_55 + score_45 + score_35
    (total_score.toFloat / students.toFloat) = 69.8333 :=
by
  sorry

end average_percent_score_l579_579961


namespace BANANA_permutations_l579_579591

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579591


namespace permutations_of_BANANA_l579_579395

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l579_579395


namespace permutations_banana_l579_579096

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579096


namespace number_of_arrangements_banana_l579_579674

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579674


namespace triangle_symmedian_l579_579940

theorem triangle_symmedian (A B C D P Q K : Type)
  [Triangle A B C] (h_angle_a : ∠A = 60)
  (h_ad_bisector : is_angle_bisector AD A B C)
  (h_pdq_equilateral : is_equilateral P D Q)
  (h_da_altitude : is_altitude DA P Q)
  (h_pk_intersect_k : intersects PB QC K)
  : is_symmedian AK A B C :=
by sorry

end triangle_symmedian_l579_579940


namespace arrange_BANANA_l579_579698

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l579_579698


namespace BANANA_arrangements_correct_l579_579443

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579443


namespace area_triangle_BMN_l579_579827

-- Define the two circles A and B.
def circle_A (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_B (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define points M and N being the common points of circle A and circle B
def common_points (M N : ℝ × ℝ) : Prop := circle_A M.1 M.2 ∧ circle_B M.1 M.2 ∧ circle_A N.1 N.2 ∧ circle_B N.1 N.2 ∧ M ≠ N

-- Define the center of circle B
def center_B : ℝ × ℝ := (-1, 2)

-- Define the area of the triangle function
def triangle_area (B M N : ℝ × ℝ) : ℝ :=
  (1 / 2) * (abs (B.1 * (M.2 - N.2) + M.1 * (N.2 - B.2) + N.1 * (B.2 - M.2)))

theorem area_triangle_BMN (M N : ℝ × ℝ) (h : common_points M N) :
  triangle_area center_B M N = 3 / 2 :=
sorry

end area_triangle_BMN_l579_579827


namespace permutations_banana_l579_579086

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579086


namespace fraction_to_decimal_terminating_l579_579747

theorem fraction_to_decimal_terminating : 
  (47 / (2^3 * 5^4) : ℝ) = 0.5875 :=
by 
  sorry

end fraction_to_decimal_terminating_l579_579747


namespace banana_arrangements_l579_579313

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579313


namespace distance_from_origin_ellipse_point_l579_579798

theorem distance_from_origin_ellipse_point :
  ∃ P : ℝ × ℝ, 
    (P.1^2 / 9 + P.2^2 / 6 = 1) ∧ 
    (let c := real.sqrt(3) in 
    let F1 := (-c, 0) in 
    let F2 := (c, 0) in 
    real.cos (
      vector.angle 
       ⟨P.1 - F1.1, P.2 - F1.2⟩ 
       ⟨P.1 - F2.1, P.2 - F2.2⟩
    ) = 3/5) ∧ 
  real.sqrt (P.1^2 + P.2^2) = real.sqrt(30) / 2 := 
sorry

end distance_from_origin_ellipse_point_l579_579798


namespace BANANA_permutation_l579_579575

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l579_579575


namespace smallest_n_exists_l579_579769

theorem smallest_n_exists (n : ℕ) (x : Fin n → ℝ) :
  (∀ i, x i ∈ Icc (1 / 2 : ℝ) 2) →
  (finset.univ.sum x ≥ (7 * n : ℝ) / 6) →
  (finset.univ.sum (λ i, 1 / x i) ≥ (4 * n : ℝ) / 3) →
  (n = 9) :=
sorry

end smallest_n_exists_l579_579769


namespace banana_permutations_l579_579352

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579352


namespace banana_arrangements_l579_579340

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579340


namespace banana_permutations_l579_579357

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579357


namespace num_ways_to_arrange_BANANA_l579_579535

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579535


namespace find_a_l579_579853

def f (x : ℝ) : ℝ :=
if x < 2 then - (2:ℝ) ^ x
else log 3 (x^2 - 1)

theorem find_a (a : ℝ) (h : f a = 1) : a = 2 := by
  sorry

end find_a_l579_579853


namespace number_of_arrangements_banana_l579_579686

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l579_579686


namespace arrange_BANANA_l579_579632

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579632


namespace num_ways_to_arrange_BANANA_l579_579523

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579523


namespace arrange_BANANA_l579_579628

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579628


namespace banana_permutations_l579_579464

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l579_579464


namespace permutations_banana_l579_579107

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l579_579107


namespace num_ways_to_arrange_BANANA_l579_579533

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579533


namespace time_juan_ran_l579_579936

variable (Distance Speed : ℝ)
variable (h1 : Distance = 80)
variable (h2 : Speed = 10)

theorem time_juan_ran : (Distance / Speed) = 8 := by
  sorry

end time_juan_ran_l579_579936


namespace BANANA_permutations_l579_579594

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579594


namespace po_distance_l579_579814

-- Define the ellispse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 6) = 1

-- Define the condition for cosine of the angle between F1, P, and F2
def cos_angle (cos_val : ℝ) : Prop := cos_val = 3/5

-- Define the proof problem in Lean
theorem po_distance (x y : ℝ) (hx : ellipse x y) (cos_val : ℝ) (h_cos : cos_angle cos_val) : (pow (sqrt ((15:ℝ)/2)) 2) = 30/4 :=
by
  sorry

end po_distance_l579_579814


namespace no_positive_integer_solutions_l579_579756

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), (a > 0) ∧ (b > 0) → 3 * a^2 ≠ b^2 + 1 :=
by
  sorry

end no_positive_integer_solutions_l579_579756


namespace arrange_BANANA_l579_579639

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579639


namespace arrange_BANANA_l579_579269

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l579_579269


namespace spider_can_reach_fly_within_8_seconds_l579_579963

-- Define the basic setup of the problem using conditions
structure SpiderFlyProblem where
  ceiling_size : ℝ
  time_steps : ℕ
  target_distance : ℝ
  spider_jump : ∀ (x y : ℝ), Prop  -- Allows for movement to midpoints of segments
  fly_position : ℝ × ℝ

-- State the problem in terms of Lean definitions and math proof
theorem spider_can_reach_fly_within_8_seconds :
  ∀ (SFP : SpiderFlyProblem),
    SFP.ceiling_size = 1 ∧
    SFP.time_steps = 8 ∧
    SFP.target_distance = 0.01 →
    ∃ (x y : ℝ), SFP.spider_jump x y ∧ dist(x, y) ≤ SFP.target_distance :=
by
  sorry  -- Proof is omitted

end spider_can_reach_fly_within_8_seconds_l579_579963


namespace BANANA_arrangements_correct_l579_579426

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579426


namespace BANANA_arrangements_correct_l579_579446

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l579_579446


namespace find_k_l579_579843

theorem find_k (x₁ x₂ k : ℝ) (h1 : x₁ * x₁ - 6 * x₁ + k = 0) (h2 : x₂ * x₂ - 6 * x₂ + k = 0) (h3 : (1 / x₁) + (1 / x₂) = 3) :
  k = 2 :=
by
  sorry

end find_k_l579_579843


namespace increasing_function_range_a_l579_579900

noncomputable def is_increasing_function_on (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

noncomputable def function_y (a : ℝ) := λ x : ℝ, cos x + a * x

theorem increasing_function_range_a :
  ∀ a : ℝ, is_increasing_function_on (function_y a) (set.Icc (-π / 2) (π / 2)) → a ≥ 1 :=
by
  sorry

end increasing_function_range_a_l579_579900


namespace BANANA_permutations_l579_579610

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l579_579610


namespace BANANA_arrangements_l579_579001

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l579_579001


namespace banana_arrangements_l579_579344

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579344


namespace arrange_BANANA_l579_579652

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l579_579652


namespace num_ways_to_arrange_BANANA_l579_579530

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l579_579530


namespace banana_permutations_l579_579356

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l579_579356


namespace arrangement_count_BANANA_l579_579253

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l579_579253


namespace banana_arrangements_l579_579326

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579326


namespace ellipse_center_origin_axes_symmetry_l579_579842

theorem ellipse_center_origin_axes_symmetry 
  (A B : ℝ × ℝ) (P : ℝ × ℝ)
  (hA : A = (0, -2))
  (hB : B = (3/2, -1))
  (hP : P = (1, -2)) :
  (∃ E : ℝ → ℝ → Prop, 
    (E = λ x y, (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ (x1 y1 x2 y2 : ℝ), 
      E x1 y1 →
      E x2 y2 →
      (x2 - x1) ≠ 0 →
      let M := (x1, y1),
          N := (x2, y2) in
      let T := (2 * x1 + (3/2 - x1) / 2, y1) in
      let H := (3 * x1 / 2 - 3 / 2 + 2, y1) in
      ∃ K : ℝ × ℝ, K = (0, -2) ∧ ((H.1 - N.1) * (K.2 - N.2) = (K.1 - N.1) * (H.2 - N.2)))) 
:= sorry

end ellipse_center_origin_axes_symmetry_l579_579842


namespace banana_arrangements_l579_579310

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l579_579310
