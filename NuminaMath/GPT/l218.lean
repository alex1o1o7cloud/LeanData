import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Concrete.Nat
import Mathlib.Algebra.Field
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Def
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Order
import Mathlib.Algebra.Order.Sqrt
import Mathlib.Analysis.Calculus.IntermediateValue
import Mathlib.Analysis.Geometry.Euclidean.Various
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Coleman
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perms
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Independent
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Intervals.Basic
import Mathlib.Data.Tree
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GroupTheory.Permutations.Cycle
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMeasure
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Topology.MetricSpace.Basic
import data.complex.basic
import data.real.sqrt
import probability.basic

namespace correct_calculation_l218_218777

theorem correct_calculation (x : ℕ) (h : x + 10 = 21) : x * 10 = 110 :=
by
  sorry

end correct_calculation_l218_218777


namespace triangle_area_l218_218205

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area :
  area_of_triangle (-2) 3 7 (-3) 4 6 = 31.5 := by
  sorry

end triangle_area_l218_218205


namespace milk_mixture_l218_218751

theorem milk_mixture:
  ∀ (x : ℝ), 0.40 * x + 1.6 = 0.20 * (x + 16) → x = 8 := 
by
  intro x
  sorry

end milk_mixture_l218_218751


namespace geometric_sequence_sixth_term_l218_218799

variable (q : ℕ) (a_2 a_6 : ℕ)

-- Given conditions:
axiom h1 : q = 2
axiom h2 : a_2 = 8

-- Prove that a_6 = 128 where a_n = a_2 * q^(n-2)
theorem geometric_sequence_sixth_term : a_6 = a_2 * q^4 → a_6 = 128 :=
by sorry

end geometric_sequence_sixth_term_l218_218799


namespace geometric_sequence_problem_l218_218827

-- Definition of a decreasing geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

-- Given conditions
def a1 : ℝ := 4
def a2 : ℝ := 2
def a3 : ℝ := 1
def q : ℝ := 1 / 2
def a (n : ℕ) : ℝ := a1 * q^n
def S (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)

-- The statement to prove in Lean 4
theorem geometric_sequence_problem :
  a1 = 4 ∧ a2 = 2 ∧ a3 = 1 ∧ q = 1/2 ∧
  (∀ n, is_geometric_sequence a q) →
  (S 8) / (1 - q^4) = 17 / 2 :=
by
  sorry

end geometric_sequence_problem_l218_218827


namespace product_of_primes_l218_218103

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l218_218103


namespace squaring_circle_impossible_l218_218800

def squaring_the_circle := ∃ (circle : ℝ → Prop) (square : ℝ → ℝ → Prop), 
  (∀ r : ℝ, circle (π * r^2)) → (∀ s : ℝ, square s (s^2)) →
  ¬ ∃ r s : ℝ, ∀ (compass_straightedge : ℝ → ℝ), (circle (π * r^2) → square (compass_straightedge r) (compass_straightedge r)^2).

theorem squaring_circle_impossible 
  (π_transcendental : ¬ ∃ (p : polynomial ℚ), p.eval π = 0) 
  (only_algebraic : ∀ x : ℝ, ∃ (p : polynomial ℚ), p.eval x = 0):
  squaring_the_circle :=
sorry

end squaring_circle_impossible_l218_218800


namespace justin_avg_time_to_find_flower_l218_218375

theorem justin_avg_time_to_find_flower :
  let gatheringTime := 120
  let lose := 3
  let additionalTime := 210
  let classmates := 30
  (gatheringTime + additionalTime) / (classmates + lose) = 10 :=
by
  let gatheringTime := 120
  let lose := 3
  let additionalTime := 210
  let classmates := 30
  have h : (gatheringTime + additionalTime) = 330 := by rfl
  have totalFlowers : (classmates + lose) = 33 := by rfl
  show 330 / 33 = 10, from sorry

end justin_avg_time_to_find_flower_l218_218375


namespace sum_of_all_four_digit_numbers_formed_l218_218269

open List

noncomputable def sum_of_four_digit_numbers (digits : List ℕ) : ℕ :=
  let perms := digits.permutations.filter (λ l, l.length = 4)
  let nums := perms.map (λ l, 1000 * l.head + 100 * l.nthLe 1 sorry + 10 * l.nthLe 2 sorry + l.nthLe 3 sorry)
  nums.sum

theorem sum_of_all_four_digit_numbers_formed : sum_of_four_digit_numbers [1, 2, 3, 4, 5] = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_formed_l218_218269


namespace decagon_diagonals_l218_218861

structure Decagon (α : Type) :=
  (vertices : Fin 10 → α) -- 10 vertices
  (circumscribed : α)  -- Circumscribed circle center

def diagonal_length {α : Type} [Field α] [HasSin α] (dec : Decagon α) (i j : Fin 10) : α :=
  2 * sin (π / 10 * (↑j - ↑i))

theorem decagon_diagonals {α : Type} [Field α] [HasCos α] [HasSin α] :
  ∀ (dec : Decagon α) (i j : Fin 10), i < j →
  ∃ k l : Fin 10, k ≠ l ∧
    (diagonal_length dec i k = diagonal_length dec j l ∨ diagonal_length dec i k = π / 2) ∧
    (diagonal_length dec j k = diagonal_length dec i l ∨ diagonal_length dec j k = π / 2) ∧
    (∃ m : α, ((1 - sqrt (5:α)) / 2 * m = 2 * sin (π / 5))) := sorry

end decagon_diagonals_l218_218861


namespace find_real_number_l218_218131

theorem find_real_number :
    (∃ y : ℝ, y = 3 + (5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + sorry)))))))))) ∧ 
    y = (3 + Real.sqrt 29) / 2 :=
by
  sorry

end find_real_number_l218_218131


namespace similar_sizes_combination_possible_l218_218401

theorem similar_sizes_combination_possible 
    (similar : Nat → Nat → Prop := λ x y, x ≤ y ∧ y ≤ 2 * x)
    (combine_piles : List Nat → Nat ∃ combined : Nat, (∀ x y ∈ combined, similar x y) → True
    (piles : List Nat) : True :=
sorry

end similar_sizes_combination_possible_l218_218401


namespace sum_of_integers_ending_in_7_between_100_and_450_l218_218639

theorem sum_of_integers_ending_in_7_between_100_and_450 :
  let a := 107 in
  let d := 10 in
  let n := 35 in
  let a_n := 447 in
  let S_n := (n / 2) * (a + a_n) in
  S_n = 9695 :=
by
  let a := 107
  let d := 10
  let n := 35
  let a_n := 447
  let S_n := (n / 2) * (a + a_n)
  show S_n = 9695
  sorry

end sum_of_integers_ending_in_7_between_100_and_450_l218_218639


namespace find_c_l218_218028

theorem find_c (a b c : ℚ) (h_eqn : ∀ y, a * y^2 + b * y + c = y^2 / 12 + 5 * y / 6 + 145 / 12)
  (h_vertex : ∀ x, x = a * (-5)^2 + b * (-5) + c)
  (h_pass : a * (-1 + 5)^2 + 1 = 4) :
  c = 145 / 12 := by
sorry

end find_c_l218_218028


namespace sum_of_consecutive_integers_l218_218277

theorem sum_of_consecutive_integers (a b : ℕ) (h1 : a < b) (h2 : b = a + 1) (h3 : real.sqrt 17 > a) (h4 : real.sqrt 17 < b) : a + b = 9 := by
  sorry

end sum_of_consecutive_integers_l218_218277


namespace sum_of_all_four_digit_numbers_l218_218235

-- Let us define the set of digits
def digits : set ℕ := {1, 2, 3, 4, 5}

-- We will define a function that generates the four-digit numbers
def four_digit_numbers := {n : ℕ // ∃ a b c d : ℕ, 
                                      a ∈ digits ∧ 
                                      b ∈ digits ∧ 
                                      c ∈ digits ∧ 
                                      d ∈ digits ∧ 
                                      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ 
                                      b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                                      n = 1000 * a + 100 * b + 10 * c + d}

-- Define a function to calculate the sum of all elements in a set of numbers
def sum_set (s : set ℕ) : ℕ := s.fold (λa b, a + b) 0

theorem sum_of_all_four_digit_numbers :
  sum_set {n | ∃ x : four_digit_numbers, x.val = n} = 399960 :=
sorry

end sum_of_all_four_digit_numbers_l218_218235


namespace find_t_value_l218_218846

theorem find_t_value (t : ℝ) (h1 : (t - 6) * (2 * t - 5) = (2 * t - 8) * (t - 5)) : t = 10 :=
sorry

end find_t_value_l218_218846


namespace prob_at_least_one_mul_4_l218_218627

-- Definition of the problem conditions
def is_multiple_of_4 (n : ℕ) : Prop :=
  n % 4 = 0

def probability_at_least_one_multiple_of_4 : ℚ :=
  let total_numbers := 60 in
  let non_multiples_of_4 := total_numbers - 15 in
  let p_none := (non_multiples_of_4 / total_numbers)^3 in
  let p_at_least_one := 1 - p_none in
  p_at_least_one

-- The theorem that needs to be proved
theorem prob_at_least_one_mul_4 :
  probability_at_least_one_multiple_of_4 = 37 / 64 :=
by
  sorry

end prob_at_least_one_mul_4_l218_218627


namespace minimum_positive_period_symmetry_monotonically_decreasing_range_l218_218717

-- Define the function f(x)
def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x + Real.pi / 4)

theorem minimum_positive_period_symmetry (w : ℝ) (k : ℤ) 
  (hw_neg : w = -1/2) :
  (∀ T : ℝ, T = 4 * Real.pi) ∧
  (∀ x : ℝ, ∀ y : ℝ, (x = (Real.pi / 2) - 2 * k * Real.pi) ∧ (y = 0)) ∧
  (∀ x : ℝ, x = - (Real.pi / 2) - 2 * k * Real.pi) :=
  sorry

theorem monotonically_decreasing_range (w : ℝ) 
  (hw_neg : w < 0) (hw_abs : abs w < 1)
  (h_mon_dec : MonotoneDecreasingOn (fun x => f w x) (Set.Ioo (Real.pi / 2) Real.pi)) :
  - (3 / 4) ≤ w ∧ w < 0 :=
  sorry

end minimum_positive_period_symmetry_monotonically_decreasing_range_l218_218717


namespace identify_irrational_number_l218_218184

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem identify_irrational_number :
  ∃ x, x = real.sqrt 3 ∧ is_irrational x :=
by
  use real.sqrt 3
  split
  {
    refl,
  },
  {
    sorry, -- Proof that sqrt 3 is irrational
  }

end identify_irrational_number_l218_218184


namespace fraction_condition_l218_218132

theorem fraction_condition (x : ℚ) :
  (3 + 2 * x) / (4 + 3 * x) = 5 / 9 ↔ x = -7 / 3 :=
by
  sorry

end fraction_condition_l218_218132


namespace solution_set_f_l218_218505

def f (x a b : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_f (a b : ℝ) (h1 : b = 2 * a) (h2 : 0 < a) :
  {x | f (2 - x) a b > 0} = {x | x < 0 ∨ 4 < x} :=
by
  sorry

end solution_set_f_l218_218505


namespace base12_reversed_diff_mod5_l218_218215

variable {A B : ℕ}

/-- Given two distinct digits A and B in base 12, the remainder when the absolute difference of their reversed base-12 numbers is divided by 5, equals the absolute difference of the digits modulo 5. -/
theorem base12_reversed_diff_mod5 (hA : A < 12) (hB : B < 12) (hNeq : A ≠ B) :
  (abs (12 * A + B - (12 * B + A))) % 5 = (abs (A - B)) % 5 :=
by
  sorry

end base12_reversed_diff_mod5_l218_218215


namespace probability_two_volunteers_at_A_distribution_xi_l218_218195
open List

def volunteers := [1, 2, 3, 4, 5]
def positions := ["A", "B", "C"]

theorem probability_two_volunteers_at_A :
  let total_events := 150
  let P_E_A := (5.choose 2 * 3.choose 1 * 2.factorial : ℕ) / (total_events : ℕ)
  P_E_A = 2 / 5
  :=
sorry

theorem distribution_xi :
  let total_events := 150
  let P_xi_1 := (5.choose 1 * 4.choose 1 * 6 : ℕ) / (total_events : ℕ)
  let P_xi_2 := 2 / 5
  let P_xi_3 := 1 - P_xi_1 - P_xi_2
  P_xi_1 = 7 / 15 ∧ P_xi_2 = 2 / 5 ∧ P_xi_3 = 2 / 15
  :=
sorry

end probability_two_volunteers_at_A_distribution_xi_l218_218195


namespace biology_marks_l218_218217

theorem biology_marks 
  (e m p c : ℤ) 
  (avg : ℚ) 
  (marks_biology : ℤ)
  (h1 : e = 70) 
  (h2 : m = 63) 
  (h3 : p = 80)
  (h4 : c = 63)
  (h5 : avg = 68.2) 
  (h6 : avg * 5 = (e + m + p + c + marks_biology)) : 
  marks_biology = 65 :=
sorry

end biology_marks_l218_218217


namespace compound_interest_one_year_l218_218843

-- Define the initial deposit
def P : ℝ := 100

-- Define the annual interest rate
def r : ℝ := 0.20

-- Define the number of compounding periods per year
def n : ℝ := 2

-- Define the time period in years
def t : ℝ := 1

-- Define the compound interest formula
def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- Prove that the amount in the account at the end of one year is 121
theorem compound_interest_one_year : 
  compound_interest P r n t = 121 :=
by sorry

end compound_interest_one_year_l218_218843


namespace points_subtracted_per_wrong_answer_l218_218496

theorem points_subtracted_per_wrong_answer 
  (total_problems : ℕ) 
  (wrong_answers : ℕ) 
  (score : ℕ) 
  (points_per_right_answer : ℕ) 
  (correct_answers : ℕ)
  (subtracted_points : ℕ) 
  (expected_points : ℕ) 
  (points_subtracted : ℕ) :
  total_problems = 25 → 
  wrong_answers = 3 → 
  score = 85 → 
  points_per_right_answer = 4 → 
  correct_answers = total_problems - wrong_answers → 
  expected_points = correct_answers * points_per_right_answer → 
  subtracted_points = expected_points - score → 
  points_subtracted = subtracted_points / wrong_answers → 
  points_subtracted = 1 := 
by
  intros;
  sorry

end points_subtracted_per_wrong_answer_l218_218496


namespace mn_equals_neg16_l218_218302

theorem mn_equals_neg16 (m n : ℤ) (h1 : m = -2) (h2 : |n| = 8) (h3 : m + n > 0) : m * n = -16 := by
  sorry

end mn_equals_neg16_l218_218302


namespace difference_of_students_l218_218044

variable (G1 G2 G5 : ℕ)

theorem difference_of_students (h1 : G1 + G2 > G2 + G5) (h2 : G5 = G1 - 30) : 
  (G1 + G2) - (G2 + G5) = 30 :=
by
  sorry

end difference_of_students_l218_218044


namespace tangent_circles_l218_218549

theorem tangent_circles (a b c : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 = a^2 → (x-b)^2 + (y-c)^2 = a^2) →
    ( (b^2 + c^2) / (a^2) = 4 ) :=
by
  intro h
  have h_dist : (b^2 + c^2) = (2 * a) ^ 2 := sorry
  have h_div : (b^2 + c^2) / (a^2) = 4 := sorry
  exact h_div

end tangent_circles_l218_218549


namespace find_y_l218_218049

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l218_218049


namespace similar_triangles_iff_equilateral_l218_218509

theorem similar_triangles_iff_equilateral (A B C D E F : Type) [triangle ABC] (h : is_incircle_tangent_points ABC D E F) :
  (similar ABC DEF) ↔ (is_equilateral ABC) :=
sorry

end similar_triangles_iff_equilateral_l218_218509


namespace competition_results_l218_218473

-- Participants and positions
inductive Participant : Type
| Oleg
| Olya
| Polya
| Pasha

-- Places in the competition (1st, 2nd, 3rd, 4th)
def Place := Fin 4

-- Statements made by the children
def Olya_statement1 : Prop := ∀ p, p % 2 = 1 -> p = Participant.Oleg ∨ p = Participant.Pasha
def Oleg_statement1 : Prop := ∃ p1 p2: Place, p1 < p2 ∧ (p1 = p2 + 1)
def Pasha_statement1 : Prop := ∀ p, p % 2 = 1 -> (p = Place 1 ∨ p = Place 3)

-- Truthfulness of the statements
def only_one_truthful (Olya_true : Prop) (Oleg_true : Prop) (Pasha_true : Prop) :=
  (Olya_true ∧ ¬ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ ¬ Oleg_true ∧ Pasha_true)

-- The actual positions
def positions : Participant → Place
| Participant.Oleg  := 0
| Participant.Olya  := 1
| Participant.Polya := 2
| Participant.Pasha := 3

-- The Lean statement to prove
theorem competition_results :
  ((Oleg_statement1 ↔ positions Participant.Oleg = 0) ∧ 
  (Olya_statement1 ↔ positions Participant.Olya = 1) ∧ 
  (Pasha_statement1 ↔ positions Participant.Pasha = 3)) ∧ 
  only_one_truthful (positions Participant.Oleg = 0) 
                    (positions Participant.Olya = 0) 
                    (positions Participant.Pasha = 0) ∧
  positions Participant.Oleg = 0 ∧ 
  positions Participant.Olya = 1 ∧
  positions Participant.Polya = 2 ∧
  positions Participant.Pasha = 3 := 
sorry

end competition_results_l218_218473


namespace swimmer_time_against_current_l218_218619

def swimmer_speed_still : ℝ := 4 -- km/h
def current_speed : ℝ := 2 -- km/h
def time_against_current : ℝ := 2.5 -- hours

theorem swimmer_time_against_current :
  let effective_speed_against_current := swimmer_speed_still - current_speed in
  let distance_against_current := effective_speed_against_current * time_against_current in
  time_against_current = distance_against_current / effective_speed_against_current := 
by
  sorry

end swimmer_time_against_current_l218_218619


namespace find_b_value_l218_218062

def perfect_square_trinomial (a b c : ℕ) : Prop :=
  ∃ d, a = d^2 ∧ c = d^2 ∧ b = 2 * d * d

theorem find_b_value (b : ℝ) :
    (∀ x : ℝ, 16 * x^2 - b * x + 9 = (4 * x - 3) * (4 * x - 3) ∨ 16 * x^2 - b * x + 9 = (4 * x + 3) * (4 * x + 3)) -> 
    b = 24 ∨ b = -24 := 
by
  sorry

end find_b_value_l218_218062


namespace circle_and_line_intersection_l218_218741

theorem circle_and_line_intersection :
  (∀ (ρ θ: ℝ), (ρ = 4 * cos θ) ↔ (ρ * ρ = 4 * ρ * cos θ)) ∧
  (∀ (x y: ℝ), (x^2 + y^2 = 4 * x) ↔ (x, y) ∈ set_of (λ p, (p.1 - 2)^2 + p.2^2 = 4)) ∧
  (∀ (t: ℝ), (let x = 1 + (sqrt 3)/2 * t in let y = (1/2) * t in (x, y)) ↔ 
     ((x, y: ℝ) → (x = 1 + (sqrt 3)/2 * t ∧ y = (1/2) * t))) ∧
  (∃ (t₁ t₂: ℝ), t₁ + t₂ = sqrt 3 ∧ t₁ * t₂ = -3 ∧ 
    (abs t₁ + abs t₂ = sqrt (3 + 12))) :=
by
  -- Proof steps needed here
  sorry

end circle_and_line_intersection_l218_218741


namespace intersection_of_A_and_B_l218_218743

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 1 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {2} :=
sorry

end intersection_of_A_and_B_l218_218743


namespace jill_net_monthly_salary_and_annual_interest_l218_218924

variables (S : ℝ) -- Jill's net monthly salary
variables (discretionary_income vacation_fund savings socializing : ℝ)
variables (remaining_amount : ℝ)
variables (interest_rate_min interest_rate_max : ℝ)
variables (average_interest_rate : ℝ)
variables (annual_savings interest_earned : ℝ)

-- Hypotheses according to conditions:
def conditions :=
  discretionary_income = S / 5 ∧
  vacation_fund = 0.30 * discretionary_income ∧
  savings = 0.20 * discretionary_income ∧
  socializing = 0.35 * discretionary_income ∧
  remaining_amount = 108 ∧
  remaining_amount = discretionary_income - (vacation_fund + savings + socializing) ∧
  interest_rate_min = 0.04 ∧
  interest_rate_max = 0.06 ∧
  average_interest_rate = (interest_rate_min + interest_rate_max) / 2
  
-- Proving Jill's net monthly salary and interest earned over the year
theorem jill_net_monthly_salary_and_annual_interest
  (h : conditions) :
  S = 3600 ∧
  interest_earned = annual_savings * average_interest_rate :=
by
  let calculated_savings_per_month := 0.20 * (S / 5)
  let calculated_annual_savings := calculated_savings_per_month * 12
  let calculated_interest_earned := calculated_annual_savings * average_interest_rate
  have h1 : 108 = 0.03 * S, by -- Manipulate given conditions to find S
     sorry,
  have h2 : S = 3600, from -- Solve for S using h1
    sorry,
  have h3 : calculated_annual_savings = 144 * 12, from -- Verify annual savings
    sorry,
  have h4 : calculated_interest_earned = 86.4, from -- Calculate the yearly interest
    sorry,
  exact ⟨h2, calc interest_earned = calculated_interest_earned : by sorry⟩

end jill_net_monthly_salary_and_annual_interest_l218_218924


namespace symmetric_point_l218_218896

theorem symmetric_point (x y : ℝ) (a b : ℝ) :
  (x = 3 ∧ y = 9 ∧ a = -1 ∧ b = -3) ∧ (∀ k: ℝ, k ≠ 0 → (y - 9 = k * (x - 3)) ∧ 
  ((x - 3)^2 + (y - 9)^2 = (a - 3)^2 + (b - 9)^2) ∧ 
  (x >= 0 → (a >= 0 ↔ x = 3) ∧ (b >= 0 ↔ y = 9))) :=
by
  sorry

end symmetric_point_l218_218896


namespace steve_marbles_l218_218488

-- Define the initial condition variables
variables (S Steve_initial Sam_initial Sally_initial Sarah_initial Steve_now : ℕ)

-- Conditions
def cond1 : Sam_initial = 2 * Steve_initial := by sorry
def cond2 : Sally_initial = Sam_initial - 5 := by sorry
def cond3 : Sarah_initial = Steve_initial + 3 := by sorry
def cond4 : Steve_now = Steve_initial + 3 := by sorry
def cond5 : Sam_initial - (3 + 3 + 4) = 6 := by sorry

-- Goal
theorem steve_marbles : Steve_now = 11 := by sorry

end steve_marbles_l218_218488


namespace min_value_of_f_l218_218314

noncomputable def f : ℝ → ℝ :=
λ x, if x > 1 then 2 * Real.sqrt 3 else 4 * Real.sin (Real.pi * x - Real.pi / 3)

theorem min_value_of_f : ∃ x : ℝ, f x = -2 * Real.sqrt 3 :=
by {
  -- Dummy implementation to ensure statement
  use 0,
  sorry
}

end min_value_of_f_l218_218314


namespace position_of_term_in_sequence_l218_218527

theorem position_of_term_in_sequence 
    (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : ∀ n, a (n + 1) - a n = 7 * n) :
    ∃ n, a n = 35351 ∧ n = 101 :=
by
  sorry

end position_of_term_in_sequence_l218_218527


namespace competition_results_correct_l218_218455

theorem competition_results_correct :
  ∃ (first second third fourth : String), 
    (first = "Oleg" ∧ second = "Olya" ∧ third = "Polya" ∧ fourth = "Pasha") ∧
    ∀ (claims : String → String → Prop),
      (claims "Olya" "all_odd_places_boys") ∧ 
      (claims "Oleg" "consecutive_places_with_olya") ∧
      (claims "Pasha" "all_odd_places_names_start_O") ∧
      ∃ (truth_teller : String), 
        truth_teller = "Oleg" ∧ 
        (claims "Oleg" "first_place") ∧ 
        ¬ (claims "Olya" "first_place") ∧ 
        ¬ (claims "Pasha" "first_place") ∧ 
        ¬ (claims "Polya" "first_place") :=
sorry

end competition_results_correct_l218_218455


namespace sqrt_minus_one_condition_odd_prime_divisors_form_l218_218571

theorem sqrt_minus_one_condition (p : ℕ) (hp_prime : p.Prime) (hp_gt2 : p > 2) :
  (∃ (x : ℕ), x * x ≡ -1 [MOD p]) ↔ ∃ k : ℕ, p = 4 * k + 1 := sorry

theorem odd_prime_divisors_form (a p : ℕ) (hp_prime : p.Prime)
  (hp_div : p ∣ (a * a + 1)) : ∃ k : ℕ, p = 4 * k + 1 := sorry

end sqrt_minus_one_condition_odd_prime_divisors_form_l218_218571


namespace intercepts_of_line_l218_218230

theorem intercepts_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : 
  (∃ y, 4 * 0 + 7 * y = 28 ∧ (0, y) = (0, 4)) ∧ 
  (∃ x, 4 * x + 7 * 0 = 28 ∧ (x, 0) = (7, 0)) :=
by
  constructor;
  { use (4 : ℝ),
    split;
    { sorry } };
  { use (7 : ℝ),
    split;
    { sorry } }

end intercepts_of_line_l218_218230


namespace sum_four_digit_numbers_l218_218249

def digits : List ℕ := [1, 2, 3, 4, 5]

/-- 
  Prove that the sum of all four-digit numbers that can be formed 
  using the digits 1, 2, 3, 4, 5 exactly once is 399960.
-/
theorem sum_four_digit_numbers : 
  (Finset.sum 
    (Finset.map 
      (λ l, 
        l.nth_le 0 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1000 + 
        l.nth_le 1 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 100 + 
        l.nth_le 2 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 10 + 
        l.nth_le 3 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1) 
      (digits.permutations.filter (λ l, l.nodup ∧ l.length = 4))) id) 
  = 399960 :=
sorry

end sum_four_digit_numbers_l218_218249


namespace all_three_pass_prob_at_least_one_pass_prob_l218_218543

section
variables {Ω : Type*} {P : ProbabilityMeasure Ω}
variables (A B C : Event Ω)

-- Hypotheses
-- Individual A passes with probability 0.8
-- Individual B passes with probability 0.6
-- Individual C passes with probability 0.5
-- The events are independent
hypothesis p_A : P A = 0.8
hypothesis p_B : P B = 0.6
hypothesis p_C : P C = 0.5
hypothesis ind_AB : Independent P A B
hypothesis ind_AC : Independent P A C
hypothesis ind_BC : Independent P B C

-- Probability that all three individuals pass the test
def prob_all_three_pass : ℝ := P (A ∩ B ∩ C)

-- Probability that at least one of the three individuals passes the test
def prob_at_least_one_pass : ℝ := P (A ∪ B ∪ C)

-- Statements
theorem all_three_pass_prob : prob_all_three_pass = 0.24 :=
begin
  sorry
end

theorem at_least_one_pass_prob : prob_at_least_one_pass = 0.96 :=
begin
  sorry
end

end

end all_three_pass_prob_at_least_one_pass_prob_l218_218543


namespace quadruple_never_return_l218_218581

theorem quadruple_never_return (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
    (∀ n : ℕ, (iterate (λ (t : ℝ × ℝ × ℝ × ℝ), (t.1 * t.2, t.2 * t.3, t.3 * t.4, t.4 * t.1)) n (a, b, c, d) = (a, b, c, d)) → a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
sorry

end quadruple_never_return_l218_218581


namespace pile_division_possible_l218_218421

theorem pile_division_possible (n : ℕ) :
  ∃ (division : list ℕ), (∀ x ∈ division, x = 1) ∧ division.sum = n :=
by
  sorry

end pile_division_possible_l218_218421


namespace median_of_AB_circumcircle_of_ABC_l218_218303

open Real

structure Point :=
(x : ℝ)
(y : ℝ)

def A := Point.mk (-4) 0
def B := Point.mk 0 2
def C := Point.mk 2 (-2)

def midpoint (P Q : Point) : Point :=
  Point.mk ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

def slope (P Q : Point) : ℝ :=
  if Q.x - P.x ≠ 0 then (Q.y - P.y) / (Q.x - P.x) else 0  -- handle vertical line case

def median_equation (A B C : Point) : ℝ × ℝ × ℝ :=
  let M := midpoint A B in
  let m := slope C M in
  (3, 4, -2)  -- Derived from the problem solution

def circumcircle_equation (A B C : Point) : ℝ × ℝ × ℝ × ℝ × ℝ :=
  (1, 1, 2, 2, -8)  -- Derived from the problem solution

theorem median_of_AB : median_equation A B C = (3, 4, -2) :=
sorry

theorem circumcircle_of_ABC : circumcircle_equation A B C = (1, 1, 2, 2, -8) :=
sorry

end median_of_AB_circumcircle_of_ABC_l218_218303


namespace cuboid_skew_lines_count_l218_218309

/-- A cuboid with vertices A, B, C, D, A', B', C', D' has 30 pairs of skew lines among the given 12 lines:
 AB', B A', C D', D C', A D', D A', B C', C B', A C, B D, A' C', B' D' --/
theorem cuboid_skew_lines_count : 
  let lines := ["AB'", "B A'", "C D'", "D C'", "A D'", "D A'", "B C'", "C B'", "A C", "B D", "A' C'", "B' D'"] in
  num_skew_pairs lines = 30 := 
sorry

end cuboid_skew_lines_count_l218_218309


namespace work_days_for_c_l218_218141

theorem work_days_for_c (A B C : ℝ)
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  sorry

end work_days_for_c_l218_218141


namespace product_of_primes_l218_218107

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l218_218107


namespace problem_statement_l218_218990

theorem problem_statement 
  (x1 y1 x2 y2 x3 y3 x4 y4 a b c : ℝ)
  (h1 : x1 > 0) (h2 : y1 > 0)
  (h3 : x2 < 0) (h4 : y2 > 0)
  (h5 : x3 < 0) (h6 : y3 < 0)
  (h7 : x4 > 0) (h8 : y4 < 0)
  (h9 : (x1 - a)^2 + (y1 - b)^2 ≤ c^2)
  (h10 : (x2 - a)^2 + (y2 - b)^2 ≤ c^2)
  (h11 : (x3 - a)^2 + (y3 - b)^2 ≤ c^2)
  (h12 : (x4 - a)^2 + (y4 - b)^2 ≤ c^2) : a^2 + b^2 < c^2 :=
by sorry

end problem_statement_l218_218990


namespace value_of_m_if_A_subset_B_l218_218290

open Set

theorem value_of_m_if_A_subset_B (m : ℝ) : 
  let A := {1}
  let B := {-1, 2 * m - 1}
  A ⊆ B ∧ A ≠ B → m = 1 := 
by
  intro h
  let h₁ := h.1
  let h₂ := h.2
  sorry

end value_of_m_if_A_subset_B_l218_218290


namespace sufficient_but_not_necessary_l218_218579

theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : |a| > 0 → a > 0 ∨ a < 0) : 
  (a > 0 → |a| > 0) ∧ (¬(|a| > 0 → a > 0)) := 
by
  sorry

end sufficient_but_not_necessary_l218_218579


namespace hyperbola_eccentricity_eq_three_l218_218715

theorem hyperbola_eccentricity_eq_three
  (a b : Real) 
  (C : Set (ℝ × ℝ)) 
  (hyperbola_eq : ∀ (x y : ℝ), ((x, y) ∈ C ↔ (x^2 / a^2 - y^2 / b^2 = 1)))
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (vertices_trisect : ∀ (F1 F2 V1 V2 : ℝ × ℝ),
    V1 ≠ V2 ∧ 
    (∥F1 - F2∥ = 3 * ∥V1 - V2∥) ∧ 
    (V1 = (x0, 0) ∨ V2 = (x1, 0)) ∧ 
    ((x0, 0) ∈ C ∧ (x1, 0) ∈ C)) :
  let c := 3 * a in
  let e := c / a in
  e = 3 :=
begin
  sorry
end

end hyperbola_eccentricity_eq_three_l218_218715


namespace avg_vel_eq_inst_vel_l218_218592

noncomputable def motion_eq (g : ℝ) (t : ℝ) : ℝ := (1 / 2) * g * t^2

def avg_velocity (g : ℝ) (t1 t2 : ℝ) : ℝ :=
  (motion_eq g t2 - motion_eq g t1) / (t2 - t1)

def instantaneous_velocity (g : ℝ) (t : ℝ) : ℝ :=
  (motion_eq g t).derivative.eval t -- using the derivative of the motion equation

theorem avg_vel_eq_inst_vel (g : ℝ) :
  avg_velocity g 1 3 = instantaneous_velocity g 2 :=
by
  sorry

end avg_vel_eq_inst_vel_l218_218592


namespace problem1_problem2_l218_218152

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (hf1 : ∀ x, f (f x) = 4 * x + 3) :
  (f = λ x, 2 * x + 1) ∨ (f = λ x, -2 * x + 5) :=
sorry

-- Problem 2
theorem problem2 :
  64^(-1/3) - (-3 * real.sqrt 2 / 2)^0 + (2^(-3))^(4/3) + 16^(-0.75) = -9/16 :=
sorry

end problem1_problem2_l218_218152


namespace cubic_polynomial_values_correct_l218_218906

-- Define the cubic polynomial
def cubic_polynomial (a b c d x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2 + c * x + d

-- Define the sequence of values
def values : List ℝ := [9261, 10648, 12167, 13824, 15625, 17576, 19683, 21952]

-- Define the equally spaced increasing values of x (assuming they are integers starting from 0)
def xs : List ℝ := List.map (fun i => i.toReal) (List.range 8)

-- Prove that the values are generated by a cubic polynomial
theorem cubic_polynomial_values_correct (a b c d : ℝ) :
  ∀ i, 0 ≤ i ∧ i < List.length values → 
    (values.get i = cubic_polynomial a b c d (xs.get i)) :=
by
  sorry

end cubic_polynomial_values_correct_l218_218906


namespace sum_four_digit_numbers_l218_218251

def digits : List ℕ := [1, 2, 3, 4, 5]

/-- 
  Prove that the sum of all four-digit numbers that can be formed 
  using the digits 1, 2, 3, 4, 5 exactly once is 399960.
-/
theorem sum_four_digit_numbers : 
  (Finset.sum 
    (Finset.map 
      (λ l, 
        l.nth_le 0 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1000 + 
        l.nth_le 1 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 100 + 
        l.nth_le 2 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 10 + 
        l.nth_le 3 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1) 
      (digits.permutations.filter (λ l, l.nodup ∧ l.length = 4))) id) 
  = 399960 :=
sorry

end sum_four_digit_numbers_l218_218251


namespace lollipops_left_l218_218495

def problem_conditions : Prop :=
  ∃ (lollipops_bought lollipops_eaten lollipops_left : ℕ),
    lollipops_bought = 12 ∧
    lollipops_eaten = 5 ∧
    lollipops_left = lollipops_bought - lollipops_eaten

theorem lollipops_left (lollipops_bought lollipops_eaten lollipops_left : ℕ) 
  (hb : lollipops_bought = 12) (he : lollipops_eaten = 5) (hl : lollipops_left = lollipops_bought - lollipops_eaten) : 
  lollipops_left = 7 := 
by 
  sorry

end lollipops_left_l218_218495


namespace arithmetic_sequence_and_general_term_l218_218283

noncomputable def sequence (a : ℕ → ℚ) (S : ℕ → ℚ) :=
  (a 1 = 3) ∧ 
  (∀ n ≥ 2, 2 * a n = S n * S (n - 1))

theorem arithmetic_sequence_and_general_term :
  ∀ (a S : ℕ → ℚ), 
  sequence a S →
  (∀ n ≥ 2, (1 / S (n - 1)) - (1 / S n) = 1 / 2) ∧
  (∀ n ≥ 2, a n = 18 / ((5 - 3 * n) * (8 - 3 * n))) :=
by
  sorry

end arithmetic_sequence_and_general_term_l218_218283


namespace product_of_primes_is_66_l218_218125

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l218_218125


namespace largest_possible_perimeter_l218_218623

theorem largest_possible_perimeter (x : ℕ) (h1 : 7 + 9 > x) (h2 : 7 + x > 9) (h3 : 9 + x > 7) : 
  let upper_bound := 16 in
  let range_x := { n : ℕ // n ≥ 3 ∧ n < upper_bound } in
  let largest_side := 15 in
  let perimeter := 7 + 9 + largest_side in
  perimeter = 31 := 
by
  have h : largest_side = 15 := sorry
  exact h

end largest_possible_perimeter_l218_218623


namespace area_of_triangle_AEC_l218_218798

theorem area_of_triangle_AEC (BE EC : ℝ) (h_BE_EC : BE / EC = 3 / 2) 
  (area_abe : ℝ) (h_area_abe : area_abe = 27) :
  let area_aec := (2 / 3) * area_abe in
  area_aec = 18 := 
by {
  -- Definitions for terms involved in the conditions.
  let h_BE_EC' : BE * 2 = EC * 3 := by {
    rw [div_eq_iff, mul_comm],
    assumption_mod_cast,
  },
  -- Calculate the area of triangle AEC.
  exact h_area_abe ▸ rfl,
}

end area_of_triangle_AEC_l218_218798


namespace no_cubic_solution_l218_218514

theorem no_cubic_solution (t : ℤ) : ¬ ∃ k : ℤ, (7 * t + 3 = k ^ 3) := by
  sorry

end no_cubic_solution_l218_218514


namespace units_digit_sum_squares_first_1013_odd_pos_integers_l218_218130

def units_digit (n : ℕ) : ℕ := n % 10

def square_units_digit (n : ℕ) : ℕ :=
  match units_digit n with
  | 1 => 1
  | 3 => 9
  | 5 => 5
  | 7 => 9
  | 9 => 1
  | _ => 0

theorem units_digit_sum_squares_first_1013_odd_pos_integers : 
  units_digit (finset.sum (finset.range 1013) (λ k, (2 * k + 1)^2)) = 5 := 
  sorry

end units_digit_sum_squares_first_1013_odd_pos_integers_l218_218130


namespace Albert_eats_48_slices_l218_218971

theorem Albert_eats_48_slices (large_pizzas : ℕ) (small_pizzas : ℕ) (slices_large : ℕ) (slices_small : ℕ) 
  (h1 : large_pizzas = 2) (h2 : small_pizzas = 2) (h3 : slices_large = 16) (h4 : slices_small = 8) :
  (large_pizzas * slices_large + small_pizzas * slices_small) = 48 := 
by 
  -- sorry is used to skip the proof.
  sorry

end Albert_eats_48_slices_l218_218971


namespace natalie_list_count_l218_218844

theorem natalie_list_count : ∀ n : ℕ, (15 ≤ n ∧ n ≤ 225) → ((225 - 15 + 1) = 211) :=
by
  intros n h
  sorry

end natalie_list_count_l218_218844


namespace no_obtuse_triangle_probability_l218_218544

def is_probability_no_obtuse_triangle (ellipse_centered_at_O : Ellipse) (foci_F : Point) (foci_F_prime : Point) : Prop :=
  -- Probability is defined as per the given problem conditions
  (1 / 8 : ℝ)

theorem no_obtuse_triangle_probability
  (ellipse_centered_at_O : Ellipse)
  (foci_F : Point)
  (foci_F_prime : Point) :
  three_points_chosen_uniformly_at_random ellipse_centered_at_O →
  probability_no_two_points_with_focal_form_obtuse_triangle ellipse_centered_at_O foci_F foci_F_prime =
  is_probability_no_obtuse_triangle ellipse_centered_at_O foci_F foci_F_prime :=
by
  sorry

end no_obtuse_triangle_probability_l218_218544


namespace proposition_false_l218_218150

theorem proposition_false : ¬ ∀ x ∈ ({1, -1, 0} : Set ℤ), 2 * x + 1 > 0 := by
  sorry

end proposition_false_l218_218150


namespace point_O_on_diagonal_l218_218484

-- Define the elements of the problem
variables (A B C D O : Type*) 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]

-- Condition: O is a point inside quadrilateral ABCD
-- Condition: The areas of triangles AOB, BCO, CDO, and DAO are equal
def equal_area_condition (area : Type*) (A B C D O : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] : Prop :=
  (area A O B = area B O C) ∧ (area B O C = area C O D) ∧ (area C O D = area D O A)

-- The goal to prove
theorem point_O_on_diagonal {A B C D O : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] 
  (area : Type*) (h : equal_area_condition area A B C D O) : 
  (O ∈ AC) ∨ (O ∈ BD) := 
by sorry

end point_O_on_diagonal_l218_218484


namespace distinct_constructions_l218_218935

-- Define the number of white and black unit cubes
def num_white_cubes : ℕ := 13
def num_black_cubes : ℕ := 14

-- Define the side length of the cube
def cube_side_length : ℕ := 3

-- Define the condition for the middle layer
def middle_layer_more_white_than_black (cube : array (cube_side_length^3) ℕ) : Prop :=
  let middle_layer := cube[cube_side_length^2 : 2 * cube_side_length^2]
  middle_layer.filter (λ x, x = 0).length > middle_layer.filter (λ x, x = 1).length

-- Define the equivalence condition (rotational symmetry)
def equivalent_constructions (cube1 cube2 : array (cube_side_length^3) ℕ) : Prop :=
  ∃ g : rotation_group, rotate g cube1 = cube2

-- Define the main theorem stating the number of distinct constructions
theorem distinct_constructions :
  ∃! (count : ℕ), 
    (count = 8) ∧
    (∀ (cube : array (cube_side_length^3) ℕ), 
      middle_layer_more_white_than_black cube 
      → (num_white_cubes = 13 ∧ num_black_cubes = 14) 
      → ∀ (cube' : array (cube_side_length^3) ℕ),
      (cube ≠ cube' → ¬ equivalent_constructions cube cube')) :=
sorry

end distinct_constructions_l218_218935


namespace set_D_cannot_form_triangle_l218_218921

-- Definition for triangle inequality theorem
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given lengths
def length_1 := 1
def length_2 := 2
def length_3 := 3

-- The proof problem statement
theorem set_D_cannot_form_triangle : ¬ triangle_inequality length_1 length_2 length_3 :=
  by sorry

end set_D_cannot_form_triangle_l218_218921


namespace angle_between_u_and_v_is_45_degrees_l218_218231

def vector (α : Type) := prod α α

def dot_product (u v : vector ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2
  
def magnitude (v : vector ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)
  
noncomputable def angle_between_vectors (u v : vector ℝ) : ℝ :=
  real.arccos (dot_product u v / (magnitude u * magnitude v))

def u : vector ℝ := (4, -1)
def v : vector ℝ := (5, 3)

theorem angle_between_u_and_v_is_45_degrees :
  angle_between_vectors u v = real.pi / 4 :=
sorry

end angle_between_u_and_v_is_45_degrees_l218_218231


namespace quadratic_function_derivative_l218_218944

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - (1 / 2) * x

theorem quadratic_function_derivative :
  (∀ x : ℝ, deriv f x = 3 * x - 1 / 2) →
  ∃ a b : ℝ, (f = λ x : ℝ, a * x^2 + b * x) ∧ (a = 3 / 2) ∧ (b = -1 / 2) :=
begin
  intros h,
  use [3 / 2, -1 / 2],
  split,
  {
    ext,
    simp [f],
  },
  split,
  { refl },
  { refl }
end

end quadratic_function_derivative_l218_218944


namespace number_of_rigid_motions_l218_218649

theorem number_of_rigid_motions (l : Type*) [line l] (patterns : recurring_pattern l) 
    (side_length : ∀ (squares triangles : l), 1) : 
    (number_of_self_transformations l patterns side_length = 2) :=
sorry

end number_of_rigid_motions_l218_218649


namespace base_7_units_digit_l218_218499

theorem base_7_units_digit : ((156 + 97) % 7) = 1 := 
by
  sorry

end base_7_units_digit_l218_218499


namespace intersection_polar_radius_l218_218740

noncomputable theory

def parametric_line (t : ℝ) : ℝ × ℝ :=
  (t + 1, real.sqrt 3 * t)

def polar_curve (θ : ℝ) : ℝ :=
  2 * real.cos θ

theorem intersection_polar_radius :
  ∃ t θ : ℝ, let (x, y) := parametric_line t,
             x = 2 * real.cos θ ∧ x^2 + y^2 = 2 * x ∧
             ( (ρ = 1 ∨ ρ = real.sqrt 3) ↔ ρ = real.sqrt (x^2 + y^2)) :=
sorry

end intersection_polar_radius_l218_218740


namespace sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218241

theorem sum_of_four_digit_numbers_formed_by_digits_1_to_5 :
  let S := {1, 2, 3, 4, 5}
  let four_digits_sum (n1 n2 n3 n4 : ℕ) :=
    1000 * n1 + 100 * n2 + 10 * n3 + n4
  (∀ a b c d ∈ S, a ≠ b → b ≠ c → c ≠ d → d ≠ a → a ≠ c → b ≠ d 
  → sum (four_digits_sum a b c d) = 399960) := sorry

end sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218241


namespace inequality_solution_l218_218923

theorem inequality_solution (x : ℤ) : (1 + x) / 2 - (2 * x + 1) / 3 ≤ 1 → x ≥ -5 := 
by
  sorry

end inequality_solution_l218_218923


namespace albert_pizza_slices_l218_218974

theorem albert_pizza_slices :
  let large_pizzas := 2
  let slices_per_large_pizza := 16
  let small_pizzas := 2
  let slices_per_small_pizza := 8
  (large_pizzas * slices_per_large_pizza + small_pizzas * slices_per_small_pizza) = 48 :=
by
  have h1 : large_pizzas * slices_per_large_pizza = 32 := by sorry
  have h2 : small_pizzas * slices_per_small_pizza = 16 := by sorry
  have ht : 32 + 16 = 48 := by sorry
  exact ht

end albert_pizza_slices_l218_218974


namespace increased_volume_l218_218173

theorem increased_volume (l w h : ℕ) 
  (volume_eq : l * w * h = 4500) 
  (surface_area_eq : l * w + l * h + w * h = 900) 
  (edges_sum_eq : l + w + h = 54) :
  (l + 1) * (w + 1) * (h + 1) = 5455 := 
by 
  sorry

end increased_volume_l218_218173


namespace base_angles_isosceles_triangle_l218_218187

-- Define the conditions
def isIsoscelesTriangle (A B C : ℝ) : Prop :=
  (A = B ∨ B = C ∨ C = A)

def exteriorAngle (A B C : ℝ) (ext_angle : ℝ) : Prop :=
  ext_angle = (180 - (A + B)) ∨ ext_angle = (180 - (B + C)) ∨ ext_angle = (180 - (C + A))

-- Define the theorem
theorem base_angles_isosceles_triangle (A B C : ℝ) (ext_angle : ℝ) :
  isIsoscelesTriangle A B C ∧ exteriorAngle A B C ext_angle ∧ ext_angle = 110 →
  A = 55 ∨ A = 70 ∨ B = 55 ∨ B = 70 ∨ C = 55 ∨ C = 70 :=
by sorry

end base_angles_isosceles_triangle_l218_218187


namespace problem_l218_218712

open Real

theorem problem (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  let S := x^2 + y^2 in (1 / ((10 : ℝ) / 3) + 1 / ((10 : ℝ) / 13) = 8 / 5) :=
by
  sorry

end problem_l218_218712


namespace seating_arrangement_l218_218540

theorem seating_arrangement (n : ℕ) (h : n ≥ 2) 
  (people : Finset (Fin (2 * n))) 
  (knows : ∀ (p : Fin (2 * n)), (people.filter (λ q, q ≠ p ∧ q ∈ people)).card ≥ n) :
  ∃ a b c d : Fin (2 * n), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (knows a = b ∨ knows a = c ∨ knows a = d) ∧
    (knows b = a ∨ knows b = c ∨ knows b = d) ∧
    (knows c = a ∨ knows c = b ∨ knows c = d) ∧
    (knows d = a ∨ knows d = b ∨ knows d = c) :=
sorry

end seating_arrangement_l218_218540


namespace sum_of_all_four_digit_numbers_formed_l218_218264

open List

noncomputable def sum_of_four_digit_numbers (digits : List ℕ) : ℕ :=
  let perms := digits.permutations.filter (λ l, l.length = 4)
  let nums := perms.map (λ l, 1000 * l.head + 100 * l.nthLe 1 sorry + 10 * l.nthLe 2 sorry + l.nthLe 3 sorry)
  nums.sum

theorem sum_of_all_four_digit_numbers_formed : sum_of_four_digit_numbers [1, 2, 3, 4, 5] = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_formed_l218_218264


namespace garbage_accumulation_correct_l218_218506

-- Given conditions
def garbage_days_per_week : ℕ := 3
def garbage_per_collection : ℕ := 200
def duration_weeks : ℕ := 2

-- Week 1: Full garbage accumulation
def week1_garbage_accumulation : ℕ := garbage_days_per_week * garbage_per_collection

-- Week 2: Half garbage accumulation due to the policy
def week2_garbage_accumulation : ℕ := week1_garbage_accumulation / 2

-- Total garbage accumulation over the 2 weeks
def total_garbage_accumulation (week1 week2 : ℕ) : ℕ := week1 + week2

-- Proof statement
theorem garbage_accumulation_correct :
  total_garbage_accumulation week1_garbage_accumulation week2_garbage_accumulation = 900 := by
  sorry

end garbage_accumulation_correct_l218_218506


namespace solve1_solve2_solve3_solve4_l218_218020

noncomputable section

-- Problem 1
theorem solve1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := sorry

-- Problem 2
theorem solve2 (x : ℝ) : (x + 1)^2 - 144 = 0 ↔ x = 11 ∨ x = -13 := sorry

-- Problem 3
theorem solve3 (x : ℝ) : 3 * (x - 2)^2 = x * (x - 2) ↔ x = 2 ∨ x = 3 := sorry

-- Problem 4
theorem solve4 (x : ℝ) : x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 29) / 2 ∨ x = (-5 - Real.sqrt 29) / 2 := sorry

end solve1_solve2_solve3_solve4_l218_218020


namespace perpendicular_lines_value_of_a_l218_218746

-- Define the lines l1 and l2
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + y - 2 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := 3 * x - (a + 1) * y + 1 = 0

-- Define perpendicularity of two lines based on their slopes
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Derive slope of a line in the form ax + by + c = 0
def slope (a b : ℝ) : ℝ := -a / b

-- Assertion for perpendicular lines and the value of a
theorem perpendicular_lines_value_of_a (a : ℝ) :
  let m1 := slope a 1 in
  let m2 := slope 3 (-(a + 1)) in
  perpendicular m1 m2 → a = 1 / 2 :=
by
  -- Insert the required proof here
  sorry

end perpendicular_lines_value_of_a_l218_218746


namespace subtracted_value_l218_218952

theorem subtracted_value (N V : ℕ) (h1 : N = 800) (h2 : N / 5 - V = 6) : V = 154 :=
by
  sorry

end subtracted_value_l218_218952


namespace sum_four_digit_numbers_l218_218262

theorem sum_four_digit_numbers : 
  let digits := [1, 2, 3, 4, 5]
  let perms := digits.permutations
  ∑ p in perms.filter (λ x, x.length = 4), (1000 * x.head + 100 * x[1] + 10 * x[2] + x[3]) = 399960 := 
by sorry

end sum_four_digit_numbers_l218_218262


namespace molecular_weight_ammonia_l218_218977

def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.008
def count_N : ℕ := 1
def count_H : ℕ := 3

theorem molecular_weight_ammonia :
  (count_N * atomic_weight_N) + (count_H * atomic_weight_H) = 17.034 :=
by
  sorry

end molecular_weight_ammonia_l218_218977


namespace problem_statement_l218_218828

theorem problem_statement :
  let a := (List.range (60 / 12)).card
  let b := (List.range (60 / Nat.lcm (Nat.lcm 2 3) 4)).card
  (a - b) ^ 3 = 0 :=
by
  sorry

end problem_statement_l218_218828


namespace probability_x_add_y_lt_4_in_square_l218_218172

noncomputable def square_area : ℝ := 3 * 3

noncomputable def triangle_area : ℝ := (1 / 2) * 2 * 2

noncomputable def region_area : ℝ := square_area - triangle_area

noncomputable def probability (A B : ℝ) : ℝ := A / B

theorem probability_x_add_y_lt_4_in_square :
  probability region_area square_area = 7 / 9 :=
by 
  sorry

end probability_x_add_y_lt_4_in_square_l218_218172


namespace part_I_part_II_l218_218835

section
variables (m : ℝ)
noncomputable def z : ℂ := (m^2 - 3 * m + 2 : ℝ) + (2 * m^2 - 5 * m + 2 : ℝ) * complex.i

theorem part_I (h : z m.im = 0) : m = 1/2 ∨ m = 2 :=
sorry

theorem part_II (h1 : (m^2 - 3 * m + 2) > 0) (h2 : (2 * m^2 - 5 * m + 2) < 0) : 1/2 < m ∧ m < 1 :=
sorry
end

end part_I_part_II_l218_218835


namespace greatest_divisor_of_arithmetic_sequence_l218_218559

theorem greatest_divisor_of_arithmetic_sequence (x c : ℤ) (h_odd : x % 2 = 1) (h_even : c % 2 = 0) :
  15 ∣ (15 * (x + 7 * c)) :=
sorry

end greatest_divisor_of_arithmetic_sequence_l218_218559


namespace even_k_sequence_property_l218_218393

variable (k : ℕ)

theorem even_k_sequence_property (hk : k % 2 = 0) (hk_pos : k > 0) :
  ∃ (a : Fin k → ℕ), (∀ (i : Fin (k - 1)), a (i + 1) ≠ a i + 1) ∧ 
  (∀ (i : Fin (k - 2)), (a i + a (i + 1)) % k ≠ 0) := 
sorry

end even_k_sequence_property_l218_218393


namespace red_triangle_existence_l218_218477

theorem red_triangle_existence (n : ℕ) (h : 1 < n) (points : Fin 2n → Prop)
  (collinearity : ∀ (i j k : Fin 2n), ¬(points i ∧ points j ∧ points k → collinear ({i, j, k}: set (Fin 2n))))
  (colored_edges : ∀ (edges : finset (Fin 2n × Fin 2n)), edges.card = n^2 + 1 → ∀ e ∈ edges, is_red e) :
  ∃ (triangles : finset (Fin 2n × Fin 2n × Fin 2n)), triangles.card ≥ n ∧ ∀ t ∈ triangles, is_red_triangle t :=
by {
  sorry
}

end red_triangle_existence_l218_218477


namespace not_true_eq_B_l218_218007

noncomputable def vector := ℝ × ℝ

variables (A B C P : vector)

def PA := P - A
def PB := P - B
def PC := P - C

def cond1 := PA + PB + PC = (0:vector)
def cond2 := PA • (PA - PB) = PC • (PA - PB)
def cond3 := PA.norm = PB.norm ∧ PB.norm = PC.norm
def cond4 := PA • PB = PB • PC ∧ PB • PC = PC • PA

theorem not_true_eq_B 
  (h1 : cond1)
  (h2 : cond2)
  (h3 : cond3)
  (h4 : cond4) : false := 
sorry

end not_true_eq_B_l218_218007


namespace instantaneous_velocity_at_t3_l218_218168

def s (t : ℝ) : ℝ := 3 * t^2 + t

theorem instantaneous_velocity_at_t3 : derivative s 3 = 19 :=
sorry

end instantaneous_velocity_at_t3_l218_218168


namespace fraction_to_decimal_l218_218993

theorem fraction_to_decimal (numerator : ℚ) (denominator : ℚ) (h : numerator = 5 ∧ denominator = 40) : 
  (numerator / denominator) = 0.125 :=
sorry

end fraction_to_decimal_l218_218993


namespace range_of_m_l218_218307

theorem range_of_m (m : ℝ) (P : ℝ × ℝ := (m, 2)) (C : ℝ × ℝ → Prop := λ ⟨x, y⟩, x ^ 2 + y ^ 2 = 1) :
  (∃ (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop),
    (l P) ∧
    (l A) ∧
    (l B) ∧ 
    (C A) ∧ 
    (C B) ∧ 
    (∀ (Q : ℝ × ℝ), (Q - P) + (Q - B) = 2 * (Q - A))) ↔ -real.sqrt 5 ≤ m ∧ m ≤ real.sqrt 5 :=
    sorry

end range_of_m_l218_218307


namespace infinite_rational_points_in_region_l218_218036

theorem infinite_rational_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), (∀ p ∈ S, p.1 > 0 ∧ p.2 > 0 ∧ p.1 + 2 * p.2 ≤ 6) ∧ S.Infinite :=
sorry

end infinite_rational_points_in_region_l218_218036


namespace inequality_Ireland_Mathematics_Olympiad_1994_l218_218686

theorem inequality_Ireland_Mathematics_Olympiad_1994 (n : ℕ) (hn : n > 1) :
  n * ((n + 1)^((2 : ℝ) / n.to_real) - 1) < 
  ∑ j in finset.range(n) + 1, (2 * j + 1) / (j : ℝ)^2 ∧
  ∑ j in finset.range(n) + 1, (2 * j + 1) / (j : ℝ)^2 < 
  n * (1 - n.to_real^(-2 / (n - 1).to_real)) + 4 := by
  sorry

end inequality_Ireland_Mathematics_Olympiad_1994_l218_218686


namespace incorrect_conditionB_l218_218005

noncomputable def Point (P : Type) [inner_product_space ℝ P]

variables {A B C P : Point}

-- Conditions
def conditionA : Prop := (P - A) + (P - B) + (P - C) = 0
def conditionB : Prop := inner (P - A) ((P - A) - (P - B)) = inner (P - C) ((P - A) - (P - B))
def conditionC : Prop := dist P A = dist P B ∧ dist P A = dist P C
def conditionD : Prop := inner (P - A) (P - B) = inner (P - B) (P - C) ∧ inner (P - B) (P - C) = inner (P - C) (P - A)

-- Goal
theorem incorrect_conditionB (hA : conditionA) (hC : conditionC) (hD : conditionD) : ¬ conditionB :=
sorry

end incorrect_conditionB_l218_218005


namespace product_of_smallest_primes_l218_218116

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218116


namespace sector_area_l218_218365

theorem sector_area (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 4.5) (h2 : radius = 5) : 
  let θ := arc_length / radius in
  let area_of_circle := Real.pi * radius^2 in
  let area_of_sector := (θ / (2 * Real.pi)) * area_of_circle in
  area_of_sector = 11.25 :=
by
  sorry

end sector_area_l218_218365


namespace frog_year_2033_l218_218000

def frogs (n : ℕ) : ℕ
| 2020 := 2
| 2021 := 9
| (n+2) := 1 + |frogs (n+1) - frogs n|

theorem frog_year_2033 : frogs 2033 = 1 :=
sorry

end frog_year_2033_l218_218000


namespace find_percentage_l218_218157

variable (P : ℝ)

/-- A number P% that satisfies the condition is 65. -/
theorem find_percentage (h : ((P / 100) * 40 = ((5 / 100) * 60) + 23)) : P = 65 :=
sorry

end find_percentage_l218_218157


namespace pile_of_stones_l218_218409

def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

theorem pile_of_stones (n : ℕ) (f : ℕ → ℕ): (∀ i, 1 ≤ f i ∧ f i ≤ n) → 
  (∀ j k, similar_sizes (f j) (f k)) → True :=
by
  simp
  exact true.intro


end pile_of_stones_l218_218409


namespace probability_one_white_one_black_l218_218594

theorem probability_one_white_one_black
  (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ)
  (drawn_balls: ℕ)
  (h_total_balls : total_balls = 15)
  (h_white_balls : white_balls = 7)
  (h_black_balls : black_balls = 8)
  (h_drawn_balls : drawn_balls = 2) :
  ( (7.choose 1) * (8.choose 1) : ℚ ) / (15.choose 2 : ℚ) = (56 / 105) := 
by 
  sorry

end probability_one_white_one_black_l218_218594


namespace necessary_sufficient_conditions_for_altitude_l218_218678

noncomputable theory

-- Define the triangle and its properties.
structure Point (α : Type*) := (x y : α)
structure Triangle (α : Type*) :=
(A B C : Point α)

-- Define the altitude.
def altitude (α : Type*) (T : Triangle α) (h : α) :=
  ∃ (A' : Point α), 
    T.B.y + h = T.C.y ∧
    T.B.x = T.C.x ∧
    T.A.x = A'.x ∧
    T.A.y = 0 ∧
    (A'.x, A'.y) ∈ line T.B T.C

-- Define angle triange conditions.
def right_angle_at_A (α : Type*) (T : Triangle α) :=
  angle T.A T.B T.C = π / 2

def isosceles_AB_AC (α : Type*) (T : Triangle α) :=
  dist T.A T.B = dist T.A T.C

-- Combine all conditions into a theorem.
theorem necessary_sufficient_conditions_for_altitude {
  α : Type*} [linear_ordered_field α] (T : Triangle α) (h : α) :
  altitude α T h →
  (right_angle_at_A α T ∨ isosceles_AB_AC α T) :=
sorry

-- Functionality to compute distance between points.
def dist (p q : Point α) : α :=
  sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Functionality to compute angle between points.
def angle (A B C : Point α) : α :=
  arccos ((dist B C)^2 + (dist B A)^2 - (dist A C)^2) / (2 * (dist B C) * (dist B A))

-- Functionality to determine if a point lies on a line.
def line (B C : Point α) (P : Point α) : Prop := 
  ∃ k : α, ∃ b : α,
    P.y = k * P.x + b ∧ 
    B.y = k * B.x + b ∧
    C.y = k * C.x + b

-- Mark the end of the definition.
end

end necessary_sufficient_conditions_for_altitude_l218_218678


namespace max_area_of_square_with_perimeter_40_l218_218963

theorem max_area_of_square_with_perimeter_40 : 
  ∃ s : ℕ, 4 * s = 40 ∧ s^2 = 100 :=
by {
  let s := 10,
  have h1 : 4 * s = 40 := by norm_num,
  have h2 : s^2 = 100 := by norm_num,
  use s,
  split,
  exact h1,
  exact h2,
  sorry
}

end max_area_of_square_with_perimeter_40_l218_218963


namespace combine_piles_l218_218428

theorem combine_piles (n : ℕ) (piles : list ℕ) (h_piles : list.sum piles = n) (h_similar : ∀ x y ∈ piles, x ≤ y → y ≤ 2 * x) :
  ∃ pile, pile ∈ piles ∧ pile = n := sorry

end combine_piles_l218_218428


namespace F_even_for_all_natural_l218_218485

noncomputable def floorSum (n : ℕ) : ℕ :=
  (Finset.range (n+1)).sum (λ k, Int.floor (n / k))

noncomputable def F (n : ℕ) : ℕ :=
  floorSum n + Int.floor (Real.sqrt n)

theorem F_even_for_all_natural (n : ℕ) : (F n) % 2 = 0 := by
  sorry

end F_even_for_all_natural_l218_218485


namespace not_true_eq_B_l218_218008

noncomputable def vector := ℝ × ℝ

variables (A B C P : vector)

def PA := P - A
def PB := P - B
def PC := P - C

def cond1 := PA + PB + PC = (0:vector)
def cond2 := PA • (PA - PB) = PC • (PA - PB)
def cond3 := PA.norm = PB.norm ∧ PB.norm = PC.norm
def cond4 := PA • PB = PB • PC ∧ PB • PC = PC • PA

theorem not_true_eq_B 
  (h1 : cond1)
  (h2 : cond2)
  (h3 : cond3)
  (h4 : cond4) : false := 
sorry

end not_true_eq_B_l218_218008


namespace club_with_two_thirds_students_l218_218661

theorem club_with_two_thirds_students
    (class : Type)
    (clubs : Type)
    (participates : class → clubs → Prop)
    (students_in_two_clubs : ∀ s : class, ∃ c1 c2 : clubs, c1 ≠ c2 ∧ participates s c1 ∧ participates s c2)
    (students_pair_club : ∀ (s1 s2 : class), ∃ c : clubs, participates s1 c ∧ participates s2 c)
    (total_students : nat) :
  ∃ c : clubs, (card (filter (λ s, participates s c) (univ_class))) ≥ (2 * total_students) / 3 :=
sorry

end club_with_two_thirds_students_l218_218661


namespace cristina_catches_nicky_l218_218449

-- Definitions from the conditions
def cristina_speed : ℝ := 4 -- meters per second
def nicky_speed : ℝ := 3 -- meters per second
def nicky_head_start : ℝ := 36 -- meters

-- The proof to find the time 't'
theorem cristina_catches_nicky (t : ℝ) : cristina_speed * t = nicky_head_start + nicky_speed * t -> t = 36 := by
  intros h
  sorry

end cristina_catches_nicky_l218_218449


namespace order_of_numbers_l218_218518

theorem order_of_numbers :
  let a := 6 ^ 0.5
  let b := 0.5 ^ 6
  let c := Real.log 6 / Real.log 0.5
  c < b ∧ b < a :=
by
  sorry

end order_of_numbers_l218_218518


namespace equilateral_triangle_covering_l218_218849

theorem equilateral_triangle_covering (ABC PQR XYZ : Type) 
[EquilateralTriangle ABC] [EquilateralTriangle PQR] [EquilateralTriangle XYZ] 
(h_PQR_smaller : TriangleSize PQR < TriangleSize ABC) 
(h_XYZ_smaller : TriangleSize XYZ < TriangleSize ABC) :
  ¬(Covers ABC (PQR, XYZ)) := by
  sorry

class EquilateralTriangle (T : Type) :=
(side_length : T)
(area : T)
(equilateral : T)  -- placeholder for property of equilateral triangle

def TriangleSize {T : Type} [HasLT T] [EquilateralTriangle T] : T := EquilateralTriangle.side_length T

class Covers (ABC : Type) (triangles : Type) :=
(cover : ABC -> triangles -> Prop)  -- placeholder for covering relation


end equilateral_triangle_covering_l218_218849


namespace solve_for_x_l218_218667

theorem solve_for_x (x : ℝ) (h : ⌈x⌉ * x = 156) : x = 12 :=
sorry

end solve_for_x_l218_218667


namespace find_y_l218_218050

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l218_218050


namespace additional_dogs_taken_in_l218_218345

theorem additional_dogs_taken_in (C D x : ℕ) (h1 : C = 45) 
    (h2 : 15 * D = 7 * C) 
    (h3 : 15 * (D + x) = 11 * C) : x = 12 := 
by 
  have hC : C = 45 := h1,
  have hD : D = 21,
  {calc
    15 * D = 7 * C : h2
    ... = 7 * 45 : by rw [hC]
    ... = 315,
    exact (nat.mul_left_inj (by norm_num)).mp  h2 },
  have hNewRatio : 15 * (21 + x) = 11 * C := h3,
  have h4x : 15 * (21 + x) = 495 := by
    rw [hC] at h3,
    norm_cast at hD,
    rw [hD, ←hC, hNewRatio],
  calc
    x = 12 : sorry

end additional_dogs_taken_in_l218_218345


namespace relationship_between_c_squared_and_ab_l218_218325

theorem relationship_between_c_squared_and_ab (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_c : c = (a + b) / 2) : 
  c^2 ≥ a * b := 
sorry

end relationship_between_c_squared_and_ab_l218_218325


namespace board_not_necessarily_checkerboard_l218_218501

theorem board_not_necessarily_checkerboard :
  ∃ (board : ℕ → ℕ → bool), (∀ (r : ℕ) (c : ℕ), r < 4028 → c < 4028 → board r c = tt ∨ board r c = ff) ∧
  (∀ (i j : ℕ), i ≤ 2018 ∩ j ≤ 2018 →
    (∀ (r : ℕ) (c : ℕ), r < 2018 → c < 2018 → 
      (board (i + r) (j + c) = tt _.sum + (board (i + r) (j + c) = ff).sum) = 1009 * 1009))
  ↔ ¬∀ (r : ℕ) (c : ℕ), r < 4028 → c < 4028 → board r c ≠ board (r + 1) (c + 1) := sorry

end board_not_necessarily_checkerboard_l218_218501


namespace min_expr_value_l218_218915

theorem min_expr_value (x : ℝ) : (sqrt (x^2 - 6*x + 13) + sqrt (x^2 - 14*x + 58)) >= sqrt 41 :=
by
  sorry

end min_expr_value_l218_218915


namespace find_degree_measure_l218_218779

variables {A B C : ℝ}

noncomputable def sin_sq_diff (B C A : ℝ) := sin B * sin B - sin C * sin C - sin A * sin A

theorem find_degree_measure
  (h : sin_sq_diff B C A = real.sqrt 3 * sin A * sin C) :
  B = 5 * real.pi / 6 := sorry

end find_degree_measure_l218_218779


namespace water_consumption_l218_218968

theorem water_consumption (x y : ℝ)
  (h1 : 120 + 20 * x = 3200000 * y)
  (h2 : 120 + 15 * x = 3000000 * y) :
  x = 200 ∧ y = 50 :=
by
  sorry

end water_consumption_l218_218968


namespace solve_for_r_l218_218865

theorem solve_for_r (r : ℚ) (h : (r + 4) / (r - 3) = (r - 2) / (r + 2)) : r = -2/11 :=
by
  sorry

end solve_for_r_l218_218865


namespace product_of_primes_l218_218098

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l218_218098


namespace piles_to_single_pile_l218_218433

-- Define the condition similar_sizes
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

-- Define the inductive step of combining stones
def combine_stones (piles : List ℕ) : List ℕ :=
  if ∃ x y, x ∈ piles ∧ y ∈ piles ∧ similar_sizes x y then
    let ⟨x, hx, y, hy, hsim⟩ := Classical.some_spec (Classical.some_spec_exists _)
    List.cons (x + y) (List.erase (List.erase piles x) y)
  else
    piles

-- Prove that a collection of piles can be reduced to a single pile of size n
theorem piles_to_single_pile (piles : List ℕ) (h : ∀ x ∈ piles, x = 1) : 
  ∃ p, list.length (Iterator.iterate combine_stones piles.count) 1 = 1 := by
  sorry

end piles_to_single_pile_l218_218433


namespace product_of_primes_l218_218081

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l218_218081


namespace product_of_primes_l218_218109

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l218_218109


namespace combine_piles_l218_218424

theorem combine_piles (n : ℕ) (piles : list ℕ) (h_piles : list.sum piles = n) (h_similar : ∀ x y ∈ piles, x ≤ y → y ≤ 2 * x) :
  ∃ pile, pile ∈ piles ∧ pile = n := sorry

end combine_piles_l218_218424


namespace product_of_smallest_primes_l218_218119

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218119


namespace pile_division_possible_l218_218415

theorem pile_division_possible (n : ℕ) :
  ∃ (division : list ℕ), (∀ x ∈ division, x = 1) ∧ division.sum = n :=
by
  sorry

end pile_division_possible_l218_218415


namespace product_of_smallest_primes_l218_218095

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218095


namespace percentage_shaded_l218_218175

-- Definitions based on conditions
def is_shaded_square (m n : ℕ) : Prop :=
  (m % 6) ∈ {0, 3}.toFinset ∧ n < 6

def total_squares := 6 * 6
def shaded_squares := 6 * 2
def shaded_percentage := (shaded_squares.toRat / total_squares.toRat) * 100

-- The theorem to be proved
theorem percentage_shaded : shaded_percentage = 33.33 :=
by
  sorry

end percentage_shaded_l218_218175


namespace third_number_is_47_l218_218042

def sequence : List ℕ := [11, 23, 47, 83, 131, 191, 263, 347, 443, 551, 671]

theorem third_number_is_47 : sequence.nth 2 = some 47 := 
by
  sorry

end third_number_is_47_l218_218042


namespace total_hats_l218_218911

theorem total_hats : 
  ∃ (B G : ℕ), G = 40 ∧ 6 * B + 7 * G = 550 ∧ B + G = 85 :=
begin
  use 45,
  use 40,
  split,
  { 
    refl,
  },
  split,
  {
    norm_num,
  },
  {
    norm_num,
  },
end

end total_hats_l218_218911


namespace LisaHoldsJellyBeans_l218_218203

/-
  Define the dimensions and volume of Bert's box, and state the relationship between Bert's and Lisa's boxes.
  Define the constants for the dimensions and the proportionality of jellybeans to volume.
  Finally, state and prove the main theorem.
-/

structure Box where
  length : ℕ
  height : ℕ
  width : ℕ

def volume (b : Box) : ℕ := b.length * b.height * b.width

def BertBox : Box := { length := a, height := b, width := c }

def LisaBox : Box := { length := 3 * a, height := 2 * b, width := 4 * c }

-- Given conditions
constant a b c : ℕ
constant JellyBeanCapacityBert : ℕ
axiom BertHoldsJellyBeans : JellyBeanCapacityBert = 150

-- Theorem to prove
theorem LisaHoldsJellyBeans (a b c : ℕ) : 
  volume LisaBox * JellyBeanCapacityBert / volume BertBox = 3600 :=
by
  -- This is where the proof would go, but we're placing sorry to skip it
  sorry

end LisaHoldsJellyBeans_l218_218203


namespace min_radius_at_origin_l218_218679

theorem min_radius_at_origin :
  ∃ (x y : ℝ), (x = 0 ∧ y = 0) ∧ ∀ (x' y' : ℝ), (√(x^2 + y^2) ≤ √(x'^2 + y'^2)) := 
sorry

end min_radius_at_origin_l218_218679


namespace pyramid_volume_is_232_l218_218956

noncomputable def pyramid_volume (length : ℝ) (width : ℝ) (slant_height : ℝ) : ℝ :=
  (1 / 3) * (length * width) * (Real.sqrt ((slant_height)^2 - ((length / 2)^2 + (width / 2)^2)))

theorem pyramid_volume_is_232 :
  pyramid_volume 5 10 15 = 232 := 
by
  sorry

end pyramid_volume_is_232_l218_218956


namespace arithmetic_evaluation_l218_218987

theorem arithmetic_evaluation :
  (3.2 - 2.95) / (0.25 * 2 + 1/4) + (2 * 0.3) / (2.3 - (1 + 2/5)) = 1 := by
  sorry

end arithmetic_evaluation_l218_218987


namespace game_result_2013_game_result_2014_l218_218635

inductive Player
| Barbara
| Jenna

def winning_player (n : ℕ) : Option Player :=
  if n % 5 = 3 then some Player.Jenna
  else if n % 5 = 4 then some Player.Barbara
  else none

theorem game_result_2013 : winning_player 2013 = some Player.Jenna := 
by sorry

theorem game_result_2014 : (winning_player 2014 = some Player.Barbara) ∨ (winning_player 2014 = some Player.Jenna) :=
by sorry

end game_result_2013_game_result_2014_l218_218635


namespace simplify_expression_l218_218863

theorem simplify_expression :
  (∑ n in finset.range 2015, 1 / (real.sqrt n + real.sqrt (n + 1))) = real.sqrt 2016 - 1 :=
sorry

end simplify_expression_l218_218863


namespace sum_of_all_four_digit_numbers_l218_218254

def digits : List ℕ := [1, 2, 3, 4, 5]

noncomputable def four_digit_numbers := 
  (Digits.permutations digits).filter (λ l => l.length = 4)

noncomputable def sum_of_numbers (nums : List (List ℕ)) : ℕ :=
  nums.foldl (λ acc num => acc + (num.foldl (λ acc' digit => acc' * 10 + digit) 0)) 0

theorem sum_of_all_four_digit_numbers :
  sum_of_numbers four_digit_numbers = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_l218_218254


namespace bruno_pens_l218_218636

theorem bruno_pens (units : ℝ) (total_pens : ℝ) (unit_pens : ℝ) 
  (h0 : units = 2.5)
  (h1 : total_pens = 30) :
  unit_pens = total_pens / units :=
by
  -- unit_pens should be 12 since 30 / 2.5 = 12
  have h2 : unit_pens = 30 / 2.5, from sorry,
  sorry

end bruno_pens_l218_218636


namespace calculation_correct_l218_218640

theorem calculation_correct : real.cbrt (-8) + 2 * (real.sqrt 2 + 2) - abs (1 - real.sqrt 2) = 3 + real.sqrt 2 :=
by
  sorry

end calculation_correct_l218_218640


namespace sum_50_eq_l218_218557

noncomputable def sum_50 : ℚ :=
  ∑ k in Finset.range 1 (51), (-1)^k * (k^3 + k^2 + 1) / k.factorial

theorem sum_50_eq : sum_50 = (5101 / 50.factorial) - 1 :=
by {
  sorry
}

end sum_50_eq_l218_218557


namespace prove_S13_eq_0_l218_218528

-- Define the arithmetic sequence and its properties
def is_arithmetic_progression (a : ℕ → ℤ) (d : ℤ) :=
  ∀ n : ℕ, a(n + 1) = a(n) + d

noncomputable def S_n (a : ℕ → ℤ) (n : ℕ) :=
  n * (a(6) + a(8)) / 2

-- The problem conditions
axiom a_seq : ℕ → ℤ
axiom d_nonzero : ℤ
axiom common_diff : d_nonzero ≠ 0
axiom arithmetic_seq : is_arithmetic_progression a_seq d_nonzero
axiom given_eq : a_seq 4 ^ 2 + a_seq 6 ^ 2 = a_seq 8 ^ 2 + a_seq 10 ^ 2

-- Question statement: Prove that S_13 = 0 given the conditions
theorem prove_S13_eq_0 : S_n a_seq 13 = 0 :=
  sorry

end prove_S13_eq_0_l218_218528


namespace collinear_M_D_I_l218_218831

variables {A B C D I J M: Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
          [MetricSpace D] [MetricSpace I] [MetricSpace J] [MetricSpace M]
          [incircle : Incircle A B C] [excircle : Excircle A B C] 
          [homothety: Homothety A B C]

-- Definitions given in the problem
def is_point_of_tangency (X : A → Prop) (P Q : Type) := 
  ∃ (T : Type), incircle.is_tangent P Q T ∧ T = X Q

def is_midpoint (X : A → Prop) (P Q : Type) := 
  ∃ (T : Type), Line.segment.median P Q T ∧ T = X Q

-- Adding assumptions as conditions stated in the problem
axiom Ps : [[is_point_of_tangency D [BC]]]
axiom Ps2 : [[Homothety.exists_midpoint M [altitude from A to BC]]]

-- Statement to prove
theorem collinear_M_D_I : collinear M D I := sorry

end collinear_M_D_I_l218_218831


namespace rational_root_even_coefficient_l218_218553

open Int

theorem rational_root_even_coefficient (a b c : ℤ) (h : a ≠ 0) (r : ℚ) (hr: a * r.num ^ 2 + b * r.num * r.denom + c * r.denom ^ 2 = 0) : Even a ∨ Even b ∨ Even c :=
sorry

end rational_root_even_coefficient_l218_218553


namespace Albert_eats_48_slices_l218_218972

theorem Albert_eats_48_slices (large_pizzas : ℕ) (small_pizzas : ℕ) (slices_large : ℕ) (slices_small : ℕ) 
  (h1 : large_pizzas = 2) (h2 : small_pizzas = 2) (h3 : slices_large = 16) (h4 : slices_small = 8) :
  (large_pizzas * slices_large + small_pizzas * slices_small) = 48 := 
by 
  -- sorry is used to skip the proof.
  sorry

end Albert_eats_48_slices_l218_218972


namespace investment_amount_first_rate_l218_218961

theorem investment_amount_first_rate : ∀ (x y : ℝ) (r : ℝ),
  x + y = 15000 → -- Condition 1 (Total investments)
  8200 * r + 6800 * 0.075 = 1023 → -- Condition 2 (Interest yield)
  x = 8200 → -- Condition 3 (Amount invested at first rate)
  x = 8200 := -- Question (How much was invested)
by
  intros x y r h₁ h₂ h₃
  exact h₃

end investment_amount_first_rate_l218_218961


namespace remaining_nap_time_is_three_hours_l218_218193

-- Define the flight time and the times spent on various activities
def flight_time_minutes := 11 * 60 + 20
def reading_time_minutes := 2 * 60
def movie_time_minutes := 4 * 60
def dinner_time_minutes := 30
def radio_time_minutes := 40
def game_time_minutes := 60 + 10

-- Calculate the total time spent on activities
def total_activity_time_minutes :=
  reading_time_minutes + movie_time_minutes + dinner_time_minutes + radio_time_minutes + game_time_minutes

-- Calculate the remaining time for a nap
def remaining_nap_time_minutes :=
  flight_time_minutes - total_activity_time_minutes

-- Convert the remaining nap time to hours
def remaining_nap_time_hours :=
  remaining_nap_time_minutes / 60

-- The statement to be proved
theorem remaining_nap_time_is_three_hours :
  remaining_nap_time_hours = 3 := by
  sorry

#check remaining_nap_time_is_three_hours -- This will check if the theorem statement is correct

end remaining_nap_time_is_three_hours_l218_218193


namespace piles_to_single_pile_l218_218431

-- Define the condition similar_sizes
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

-- Define the inductive step of combining stones
def combine_stones (piles : List ℕ) : List ℕ :=
  if ∃ x y, x ∈ piles ∧ y ∈ piles ∧ similar_sizes x y then
    let ⟨x, hx, y, hy, hsim⟩ := Classical.some_spec (Classical.some_spec_exists _)
    List.cons (x + y) (List.erase (List.erase piles x) y)
  else
    piles

-- Prove that a collection of piles can be reduced to a single pile of size n
theorem piles_to_single_pile (piles : List ℕ) (h : ∀ x ∈ piles, x = 1) : 
  ∃ p, list.length (Iterator.iterate combine_stones piles.count) 1 = 1 := by
  sorry

end piles_to_single_pile_l218_218431


namespace problem_I_problem_II_l218_218443

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 - m * Real.log(2*x + 1)

theorem problem_I (m : ℝ) (h_m : m > 0) : 
  (∀ x ∈ Ioc (-1/2 : ℝ) (1 : ℝ), Derivative f x m ≤ 0) → m ≥ 3 :=
sorry

theorem problem_II (m : ℝ) (h_m : m > 0) :
  (∃ x : ℝ, x ∈ Ioc (-1/2) 1 ∧ (∀ y ∈ Ioc (-1/2) 1, f x m ≤ f y m)) ∧
  ((0 < m ∧ m < 3) → ∃ x : ℝ, x = (-1 + Real.sqrt (1 + 8*m)) / 4 ∧ ∀ y ∈ Ioc (-1/2) 1, f x m ≤ f y m) ∧
  (m ≥ 3 → ∃ x : ℝ, x = 1 ∧ ∀ y ∈ Ioc (-1/2) 1, f x m ≤ f y m) :=
sorry

end problem_I_problem_II_l218_218443


namespace total_payment_correct_l218_218631

def rate_per_kg_grapes := 68
def quantity_grapes := 7
def rate_per_kg_mangoes := 48
def quantity_mangoes := 9

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes

def total_amount_paid := cost_grapes + cost_mangoes

theorem total_payment_correct :
  total_amount_paid = 908 := by
  sorry

end total_payment_correct_l218_218631


namespace total_amount_spent_l218_218932

theorem total_amount_spent (num_cows num_goats : ℕ) (price_per_goat price_per_cow : ℕ) 
  (h1 : num_cows = 2) (h2 : num_goats = 8) (h3 : price_per_goat = 60) (h4 : price_per_cow = 460) : 
  num_cows * price_per_cow + num_goats * price_per_goat = 1400 :=
by 
  rw [h1, h2, h3, h4]
  norm_num

end total_amount_spent_l218_218932


namespace power_of_point_l218_218384

variables {P : Point} {Γ : Circle}

theorem power_of_point (D1 D2 : Line) (A B C D : Point)
  (hD1 : D1 ∋ P) (hD2 : D2 ∋ P)
  (hA : A ∈ Γ) (hB : B ∈ Γ)
  (hC : C ∈ Γ) (hD : D ∈ Γ)
  (hD1A : A ∈ D1) (hD1B : B ∈ D1)
  (hD2C : C ∈ D2) (hD2D : D ∈ D2) :
  dist P A * dist P B = dist P C * dist P D :=
sorry

end power_of_point_l218_218384


namespace find_original_denominator_l218_218617

theorem find_original_denominator (d : ℕ) 
  (h : (10 : ℚ) / (d + 7) = 1 / 3) : 
  d = 23 :=
by 
  sorry

end find_original_denominator_l218_218617


namespace triangles_congruent_l218_218550

-- Define the necessary points and lines
variables {P Q A A' B B' : Point}
variables {circle1 circle2 : Circle}
variables {line1 line2 : Line}
variables [circle_intersect : Intersect circle1 circle2 P Q]
variables [line1_intersect : Intersect line1 circle1 P A]
variables [line1_intersect' : Intersect line1 circle2 P A']
variables [line2_intersect : Intersect line2 circle1 Q B]
variables [line2_intersect' : Intersect line2 circle2 Q B']
variables [line_parallel : Parallel line1 line2]

-- Define the statement to show that the triangles are congruent
theorem triangles_congruent :
  Congruent (Triangle.mk P B B') (Triangle.mk Q A A') :=
sorry

end triangles_congruent_l218_218550


namespace half_circle_area_outside_triangle_l218_218382

namespace Geometry

-- Definitions and conditions
def triangle_ABC : Type :=
  { A B C : Point // ∠ B A C = 90° ∧ dist A B = 8 ∧ dist A C = 6 }

def circle_tangent (P Q R : Point) (c : Circle) : Prop :=
  c.tangentAt P ∧ c.tangentAt Q ∧ dist P Q = dist P R

-- Given problem transformed into Lean statement
theorem half_circle_area_outside_triangle (A B C X Y X' Y' : Point) 
  (h_triangle : triangle_ABC (A, B, C))
  (h_tangent : circle_tangent X Y (circle (midpoint X' Y') (dist X' Y'/2)))
  (h_diametrically_opposite : diametrically_opposite X X' (circle (midpoint X' Y') (dist X' Y'/2)))
  (h_diametrically_opposite' : diametrically_opposite Y Y' (circle (midpoint X' Y') (dist X' Y'/2)))
  (h_BC : dist B C = sqrt (dist A B ^ 2 + dist A C ^ 2)) :
  (π / 2 - 1 / 2) * (2.5 ^ 2) ≈ 11.78 := sorry

end Geometry

end half_circle_area_outside_triangle_l218_218382


namespace ratio_RN_NS_l218_218366

noncomputable def ratio_RN_NS_is_one_to_one : Prop :=
  let A : (ℝ × ℝ) := (0, 10)
  let B : (ℝ × ℝ) := (10, 10)
  let C : (ℝ × ℝ) := (10, 0)
  let D : (ℝ × ℝ) := (0, 0)
  let F : (ℝ × ℝ) := (3, 0)
  let N : (ℝ × ℝ) := ((0 + 3) / 2, (10 + 0) / 2)
  let slope : ℝ := 3 / 10
  let R : (ℝ × ℝ) := (18, 10)
  let S : (ℝ × ℝ) := (-15, 0)
  let dist := λ P Q : (ℝ × ℝ), real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)
  dist N R = dist N S

theorem ratio_RN_NS : ratio_RN_NS_is_one_to_one := by
  sorry

end ratio_RN_NS_l218_218366


namespace pile_of_stones_l218_218414

def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

theorem pile_of_stones (n : ℕ) (f : ℕ → ℕ): (∀ i, 1 ≤ f i ∧ f i ≤ n) → 
  (∀ j k, similar_sizes (f j) (f k)) → True :=
by
  simp
  exact true.intro


end pile_of_stones_l218_218414


namespace count_bad_arrangements_l218_218038

def numbers : List ℕ := [1, 2, 3, 4, 5, 6]

/-- 
An arrangement is bad if it cannot cover sums 1 to 21 using subsets of consecutive numbers.
Arrangements are considered the same if they differ only by rotation or reflection.
-/
def isBadArrangement (arrangement : List ℕ) : Prop := sorry

def distinctArrangements (arrangements : List (List ℕ)) : List (List ℕ) := sorry

theorem count_bad_arrangements : ∃ (badArrangements : List (List ℕ)),
  allPairs (isBadArrangement badArrangements) ∧ (length (distinctArrangements badArrangements) = 4) :=
sorry

end count_bad_arrangements_l218_218038


namespace angle_between_vectors_is_90_degrees_l218_218294

open Real
open ComplexConjugate

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (k : ℝ)

theorem angle_between_vectors_is_90_degrees (ha : a ≠ 0) (hb : b ≠ 0) (hk : k ≠ 0) 
(h: ∥a + k • b∥ = ∥a - k • b∥) : real.angle (a, b) = π / 2 :=
sorry

end angle_between_vectors_is_90_degrees_l218_218294


namespace product_of_smallest_primes_l218_218092

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218092


namespace third_largest_value_is_13_l218_218607

def list_of_six_integers (l : List ℕ) : Prop :=
  l.length = 6 ∧ 
  (1 ≤ l.length) ∧ 
  (∀ x, x ∈ l → 1 ≤ x)

theorem third_largest_value_is_13 (l : List ℕ) :
  list_of_six_integers l →
  (l.sum = 90) →
  (l.max' (by { cases l, exact (nat.le_refl 0), exact (nat.le_of_lt (nat.lt_add_one_of_le (nat.le_of_lt_succ (nat.zero_lt_succ 0)))))) 
   - l.min' (by { cases l, exact (nat.le_refl 0), exact (nat.le_of_lt (nat.lt_add_one_of_le (nat.le_of_lt_succ (nat.zero_lt_succ 0)))})) = 20 →
  2 ≤ l.count 12 →
  (l.sorted_nth (by sorry) 2 + l.sorted_nth (by sorry) 3 = 26) →
  l.sorted_nth (by sorry) 2 = 13 :=
sorry

end third_largest_value_is_13_l218_218607


namespace smallest_sum_of_sequence_l218_218524

theorem smallest_sum_of_sequence {
  A B C D k : ℕ
} (h1 : 2 * B = A + C)
  (h2 : D - C = (C - B) ^ 2)
  (h3 : 4 * B = 3 * C)
  (h4 : B = 3 * k)
  (h5 : C = 4 * k)
  (h6 : A = 2 * k)
  (h7 : D = 4 * k + k ^ 2) :
  A + B + C + D = 14 :=
by
  sorry

end smallest_sum_of_sequence_l218_218524


namespace collinear_points_d_l218_218657

theorem collinear_points_d (a b c d : ℝ) :
  ∃ d : ℝ, (d = 1/8 ∨ d = -1/32) ↔
  collinear ![
    ![2, 0, a],
    ![b, 2, 0],
    ![0, c, 2],
    ![8d, 8d, -2d]
  ] :=
sorry

end collinear_points_d_l218_218657


namespace sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218240

theorem sum_of_four_digit_numbers_formed_by_digits_1_to_5 :
  let S := {1, 2, 3, 4, 5}
  let four_digits_sum (n1 n2 n3 n4 : ℕ) :=
    1000 * n1 + 100 * n2 + 10 * n3 + n4
  (∀ a b c d ∈ S, a ≠ b → b ≠ c → c ≠ d → d ≠ a → a ≠ c → b ≠ d 
  → sum (four_digits_sum a b c d) = 399960) := sorry

end sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218240


namespace arith_seq_sum_7_8_9_l218_218708

noncomputable def S_n (a : Nat → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n.succ).sum a

def arith_seq (a : Nat → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

theorem arith_seq_sum_7_8_9 (a : Nat → ℝ) (h_arith : arith_seq a)
    (h_S3 : S_n a 3 = 8) (h_S6 : S_n a 6 = 7) : 
  (a 7 + a 8 + a 9) = 1 / 8 := 
  sorry

end arith_seq_sum_7_8_9_l218_218708


namespace circles_intersect_l218_218696

theorem circles_intersect
    (a : ℝ) (h_a_gt_zero : a > 0)
    (h_intersection_length : ∃ (p : ℝ×ℝ) (q : ℝ×ℝ), (p ≠ q) ∧
      (set_of (λ z : ℝ×ℝ, z.1^2 + z.2^2 - 2*a*z.2 = 0 ∧ z.1 + z.2 = 0) = {p, q}) ∧
      dist p q = 2 * sqrt 2)
    : ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 1 →
        ∃ (M N : ℝ × ℝ) (R r : ℝ), M = (0, a) ∧ R = a ∧ N = (1, 1) ∧ r = 1 ∧
          dist M N < R + r ∧ R - r < dist M N := by
  sorry

end circles_intersect_l218_218696


namespace prove_statements_l218_218016

theorem prove_statements (x y z : ℝ) (h : x + y + z = x * y * z) :
  ( (∀ (x y : ℝ), x + y = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → z = 0))
  ∧ (∀ (x y : ℝ), x = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → y = -z))
  ∧ z = (x + y) / (x * y - 1) ) :=
by
  sorry

end prove_statements_l218_218016


namespace solve_real_eq_l218_218672

theorem solve_real_eq (x : ℝ) :
  (16^x + 25^x) / (20^x + 15^x) = 9 / 5 ↔ x = 0 :=
by {
  sorry
}

end solve_real_eq_l218_218672


namespace complex_div_conj_by_i_l218_218716

noncomputable def complex_conjugate (z : ℂ) := conj z

theorem complex_div_conj_by_i (z : ℂ) (i : ℂ) (h_i : i = complex.I) (h_z : z = 3 - 4 * complex.I) : 
  (complex_conjugate z) / i = 4 - 3 * complex.I :=
by
  sorry

end complex_div_conj_by_i_l218_218716


namespace max_elements_A_union_B_l218_218286

noncomputable def sets_with_conditions (A B : Finset ℝ ) (n : ℕ) : Prop :=
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ A → s.sum id ∈ B) ∧
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ B → s.prod id ∈ A)

theorem max_elements_A_union_B {A B : Finset ℝ} (n : ℕ) (hn : 1 < n)
    (hA : A.card ≥ n) (hB : B.card ≥ n)
    (h_condition : sets_with_conditions A B n) :
    A.card + B.card ≤ 2 * n :=
  sorry

end max_elements_A_union_B_l218_218286


namespace cost_of_gallon_l218_218208

variable (pint_cost : ℕ) (pints_per_gallon : ℕ) (doors : ℕ) (savings : ℕ)
variable (cost_separate : ℕ) (cost_aggregate : ℕ)

-- Define the cost of a pint and related conditions
def cost_pint : ℕ := pint_cost

-- Define the total number of doors to paint
def total_doors : ℕ := doors

-- Define how many pints are in a gallon
def pints_in_gallon : ℕ := pints_per_gallon

-- Define the savings Christine makes if she buys a gallon
def savings_buying_gallon : ℕ := savings

-- Define the cost of buying 8 separate pints
def cost_separate_pints : ℕ := total_doors * cost_pint

-- Define the cost of the gallon of paint when factoring in savings
def cost_gallon : ℕ := cost_separate_pints - savings_buying_gallon

-- The main theorem to prove the cost of a gallon of paint
theorem cost_of_gallon : cost_gallon = 55 := by
  sorry

end cost_of_gallon_l218_218208


namespace hitting_next_shot_given_first_l218_218811

variables {A B : Prop}
variable (P : Prop → ℚ)

def student_first_shot_probability := P A = 9 / 10
def consecutive_shots_probability := P (A ∧ B) = 1 / 2

theorem hitting_next_shot_given_first 
    (h1 : student_first_shot_probability P)
    (h2 : consecutive_shots_probability P) :
    (P (A ∧ B) / P A) = 5 / 9 :=
by
  sorry

end hitting_next_shot_given_first_l218_218811


namespace total_points_GoldenState_l218_218355

-- Conditions
def Draymond_points : ℕ := 12
def Curry_points : ℕ := 2 * Draymond_points
def Kelly_points : ℕ := 9
def Durant_points : ℕ := 2 * Kelly_points
def Klay_points : ℕ := Draymond_points / 2
def Jordan_points : ℕ := Nat.ceil (9 * 1.20)
def Migel_points : ℕ := abs (Nat.floor (sqrt 24) - 5)
def Green_points : ℕ := Nat.ceil (Durant_points * 3 / 4)
def Barnes_points : ℕ := Nat.ceil (Float.log2 (Nat.ceil (Durant_points.toNat))).toNat + 8
def Christine_points : ℕ := 3^6

-- The goal
theorem total_points_GoldenState :
  Draymond_points + Curry_points + Kelly_points + Durant_points + Klay_points +
  Jordan_points + Migel_points + Green_points + Barnes_points + Christine_points = 839 := by
  sorry

end total_points_GoldenState_l218_218355


namespace sum_of_all_four_digit_numbers_l218_218252

def digits : List ℕ := [1, 2, 3, 4, 5]

noncomputable def four_digit_numbers := 
  (Digits.permutations digits).filter (λ l => l.length = 4)

noncomputable def sum_of_numbers (nums : List (List ℕ)) : ℕ :=
  nums.foldl (λ acc num => acc + (num.foldl (λ acc' digit => acc' * 10 + digit) 0)) 0

theorem sum_of_all_four_digit_numbers :
  sum_of_numbers four_digit_numbers = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_l218_218252


namespace patio_perimeter_is_100_feet_l218_218664

theorem patio_perimeter_is_100_feet
  (rectangle : Prop)
  (length : ℝ)
  (width : ℝ)
  (length_eq_40 : length = 40)
  (length_eq_4_times_width : length = 4 * width) :
  2 * length + 2 * width = 100 := 
by
  sorry

end patio_perimeter_is_100_feet_l218_218664


namespace elements_belong_to_two_sets_l218_218720

theorem elements_belong_to_two_sets
  (n : ℕ)
  (a : Fin n → Type)
  (A : Fin n → Finset (Subtype a))
  (h : ∀ i j : Fin n, i ≠ j → (A i ∩ A j).Nonempty → (A i = {(a i), (a j)} ∨ A j = {(a i), (a j)}))
  : ∀ j : Fin n, (A.filter (λ S => j ∈ S)).card = 2 :=
by
  sorry

end elements_belong_to_two_sets_l218_218720


namespace distance_between_points_l218_218658

theorem distance_between_points:
  ∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) = (4, -1) → (x₂, y₂) = (12, -9) →
  (dist (x₁, y₁) (x₂, y₂) = 8 * √2) :=
by {
  intro x₁ y₁ x₂ y₂,
  intro h1,
  intro h2,
  have h3 : dist (x₁, y₁) (x₂, y₂) = sqrt ((x₂ - x₁) ^ 2 + (y₂ - y₁) ^ 2), sorry,
  rw [h1, h2] at h3,
  rw [dist] at h3,
  norm_num at h3,
  linarith,
}

end distance_between_points_l218_218658


namespace find_m_n_l218_218886

noncomputable def isRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

noncomputable def floor (x : ℝ) : ℤ := Real.floor x

noncomputable def frac (x : ℝ) : ℝ := x - floor x

theorem find_m_n (m n : ℕ) (b : ℝ)
  (h1 : b = m / n)
  (h2 : isRelativelyPrime m n)
  (h3 : ∀ x : ℝ, floor x * frac x = b * x^2 - 0.5 * x → x ≤ 294) :
  m + n = 202 ∨ m + n = 474 ∨ m + n = 618 ∨ m + n = 847 ∨ m + n = 1025 :=
sorry

end find_m_n_l218_218886


namespace prove_f_range_prove_triangle_properties_l218_218317

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + sqrt 3 * sin x * cos x - 1 / 2

theorem prove_f_range :
  ∀ x : ℝ, 0 ≤ x → x ≤ pi / 2 → -1/2 ≤ f x ∧ f x ≤ 1 :=
by sorry

theorem prove_triangle_properties :
  ∀ (A a c : ℝ), A = pi / 3 → a = 2 * sqrt 3 → c = 4 →
    let b := 2 in
    let S := 2 * sqrt 3 in
    f A = 1 ∧ a^2 = b^2 + c^2 - 2 * b * c * cos A ∧ S = 1 / 2 * b * c * sin A :=
by sorry

end prove_f_range_prove_triangle_properties_l218_218317


namespace pile_of_stones_l218_218410

def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

theorem pile_of_stones (n : ℕ) (f : ℕ → ℕ): (∀ i, 1 ≤ f i ∧ f i ≤ n) → 
  (∀ j k, similar_sizes (f j) (f k)) → True :=
by
  simp
  exact true.intro


end pile_of_stones_l218_218410


namespace symmetry_of_function_l218_218313

noncomputable def f : ℝ → ℝ := λ x => Real.sin (2 * ω * x + π / 3)

theorem symmetry_of_function (ω : ℝ) (hω : ω > 0) :
  SymmetricAboutPoint (f, (-π/12, 0)) := sorry

end symmetry_of_function_l218_218313


namespace find_d_l218_218507

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_d (d : ℝ) :
  ∀ (A B : ℝ × ℝ), 
  A = (2, 5) → B = (8, 11) → 
  let M := midpoint A B in
  (M.1 - M.2 = d) →
  d = -3 :=
by
  intros A B HA HB M hM
  sorry

end find_d_l218_218507


namespace find_range_of_a_l218_218153

theorem find_range_of_a (a : ℝ) :
  (∀ (x : ℝ), x > 0 → log a x > log a 1) ∧ (∃ x : ℝ, x^2 - 2 * a * x + 4 = 0) → a ≥ 2 :=
by
  sorry

end find_range_of_a_l218_218153


namespace volume_theorem_l218_218647

open Real

noncomputable def volume_of_pyramid (A B C D P : ℝ × ℝ × ℝ) (θ : ℝ) (h_base_square : distance A B = 2 ∧ distance B C = 2 ∧ distance C D = 2 ∧ distance D A = 2) 
  (h_vertex_eqdist : distance P A = distance P B ∧ distance P B = distance P C ∧ distance P C = distance P D) 
  (h_angle_APB : ∃ (O : ℝ × ℝ × ℝ), O = midpoint A C ∧ ∠P A B = θ) : ℝ :=
  ((4:ℝ) / 3) * sqrt (tan (θ / 2) ^ 2 + 1)

-- Statement of the theorem to prove
theorem volume_theorem (P A B C D : ℝ × ℝ × ℝ) (θ : ℝ)
  (h_base_square : distance A B = 2 ∧ distance B C = 2 ∧ distance C D = 2 ∧ distance D A = 2)
  (h_vertex_eqdist : distance P A = distance P B ∧ distance P B = distance P C ∧ distance P C = distance P D)
  (h_angle_APB : ∃ (O : ℝ × ℝ × ℝ), O = midpoint A C ∧ ∠P A B = θ) :
  volume_of_pyramid A B C D P θ h_base_square h_vertex_eqdist h_angle_APB = (4 / 3) * sqrt (tan (θ / 2) ^ 2 + 1) := 
  sorry

end volume_theorem_l218_218647


namespace average_water_per_day_l218_218001

variable (day1 : ℕ)
variable (day2 : ℕ)
variable (day3 : ℕ)

def total_water_over_three_days (d1 d2 d3 : ℕ) := d1 + d2 + d3

theorem average_water_per_day :
  day1 = 215 ->
  day2 = 215 + 76 ->
  day3 = 291 - 53 ->
  (total_water_over_three_days day1 day2 day3) / 3 = 248 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_water_per_day_l218_218001


namespace part_I_part_II_l218_218276

variable {a b : ℝ}
variable (x y z : ℝ)

theorem part_I (hab : a > b > 0) : 
  let m := a + 1 / ((a - b) * b)
  in m ≥ 3 :=
  sorry

theorem part_II (h1 : x + y + z = 3) (h2 : x^2 + 4 * y^2 + z^2 = 3) :
  |x + 2 * y + z| ≤ 3 :=
  sorry

end part_I_part_II_l218_218276


namespace exists_root_in_interval_l218_218200

open Real

theorem exists_root_in_interval : ∃ x, 1.1 < x ∧ x < 1.2 ∧ (x^2 + 12*x - 15 = 0) :=
by {
  let f := λ x : ℝ, x^2 + 12*x - 15,
  have h1 : f 1.1 = -0.59 :=  sorry,
  have h2 : f 1.2 = 0.84 := sorry,
  have sign_change : (f 1.1) * (f 1.2) < 0,
  { rw [h1, h2], linarith, },
  exact exists_has_deriv_at_eq_zero (by norm_num1) (by norm_num1) (by linarith)
}

end exists_root_in_interval_l218_218200


namespace collinear_points_b_value_l218_218999

theorem collinear_points_b_value :
  ∀ (b : ℚ), (∃ k : ℚ, k = (-6 - 4) / (4 - (-b + 3)) ∧ k = (-6 - 3) / (4 - (3b + 4))) → b = -3 / 13 :=
begin
  sorry
end

end collinear_points_b_value_l218_218999


namespace complex_root_circle_radius_l218_218628

theorem complex_root_circle_radius:
  ∀ z : ℂ, ((z - 2)^4 = 16 * (z^4)) → 
  ∃ r : ℝ, r = 2 / 3 ∧ (abs (z - 2) = 2 * abs z) :=
begin
  sorry
end

end complex_root_circle_radius_l218_218628


namespace correct_propositions_l218_218920

-- Conditions for the propositions
def circleA := ∀ (x y : ℝ), x^2 + y^2 = 2
def lineA (x y : ℝ) := x - y + 1 = 0
def distanceA_eq := ∃ (px py : ℝ), px^2 + py^2 = 2 ∧ abs((px - py + 1) / sqrt(2)) = sqrt(2) / 2

def curveC1 := ∀ (x y : ℝ), x^2 + y^2 + 2 * x = 0
def curveC2_m (m : ℝ) := ∀ (x y : ℝ), x^2 + y^2 - 4 * x - 8 * y + m = 0
def commonTangentB (m : ℝ) := 4 < m

def circleC := ∀ (x y : ℝ), x^2 + y^2 = 2
def lineC (x y : ℝ) := x + y + 2 * sqrt(3) = 0
def minDistanceC_eq := ∃ (px py : ℝ), abs(sqrt((px^2 + py^2) - 2)) = 2

def circleD := ∀ (x y : ℝ), x^2 + y^2 = 4
def lineD (x y : ℝ) := 2 * x + y - 8 = 0
def intersectionPointD (px py : ℝ) := px = 1 ∧ py = 1 / 2

theorem correct_propositions : 
  (circleA → (∀ x y, lineA x y → distanceA_eq)) ∧
  (curveC1 → (∀ m, curveC2_m m → ¬ commonTangentB m)) ∧
  (circleC → (∀ x y, lineC x y → minDistanceC_eq)) ∧
  (circleD → (∀ x y, lineD x y → intersectionPointD x y)) :=
  sorry

end correct_propositions_l218_218920


namespace find_x_l218_218670

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 156) (h2 : x ≥ 0) : x = 12 :=
sorry

end find_x_l218_218670


namespace piles_can_be_reduced_l218_218395

/-! 
  We define similar sizes as the difference between sizes being at most a factor of two.
  Given any number of piles of stones, we aim to prove that these piles can be combined 
  iteratively into one single pile.
-/

def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

theorem piles_can_be_reduced (n : ℕ) :
  ∃ pile : ℕ, (pile = n) ∧ (∀ piles : list ℕ, list.sum piles = n → 
    (∃ piles' : list ℕ, list.sum piles' = n ∧ list.length piles' = 1)) :=
by
  -- Placeholder for the proof.
  sorry

end piles_can_be_reduced_l218_218395


namespace problem_part1_problem_part2_l218_218703

open Nat

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + n
noncomputable def a (n : ℕ) : ℕ := if n = 0 then 0 else S n - S (n - 1)
noncomputable def b (n : ℕ) : ℕ := 2^(n-1)
noncomputable def T (n : ℕ) : ℕ := (4 * n - 5) * 2^n + 5

theorem problem_part1 (n : ℕ) (hn : n ≥ 1):
  a n = 4 * n - 1 ∧ b n = 2^(n-1) :=
by
  sorry

theorem problem_part2 (n : ℕ) :
  let ab := λ i : ℕ, a i * b i
  (fintype.card (fin n)) = n →
  ∑ i in finset.range n, ab i = (4 * n - 5) * 2^n + 5 :=
by
  sorry

end problem_part1_problem_part2_l218_218703


namespace lily_pads_doubling_l218_218784

theorem lily_pads_doubling (patch_half_day: ℕ) (doubling_rate: ℝ)
  (H1: patch_half_day = 49)
  (H2: doubling_rate = 2): (patch_half_day + 1) = 50 :=
by 
  sorry

end lily_pads_doubling_l218_218784


namespace all_points_below_line_l218_218270

theorem all_points_below_line (a b : ℝ) (n : ℕ) (x y : ℕ → ℝ)
  (h1 : b > a)
  (h2 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k = a + ((k : ℝ) * (b - a) / (n + 1)))
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k = a * (b / a) ^ (k / (n + 1))) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k < x k := 
sorry

end all_points_below_line_l218_218270


namespace ethanol_in_full_tank_l218_218186

theorem ethanol_in_full_tank :
  ∀ (fuelA fuelB : Real) (ethanolA ethanolB capacity : Real), 
    capacity = 200 ∧ 
    fuelA + fuelB = capacity ∧ 
    fuelA = 99.99999999999999 ∧ 
    ethanolA = 0.12 * fuelA ∧ 
    ethanolB = 0.16 * fuelB ∧ 
    fuelB = capacity - fuelA → 
    ethanolA + ethanolB ≈ 28 := by
  sorry

end ethanol_in_full_tank_l218_218186


namespace exists_n_equal_sine_different_sides_l218_218274

noncomputable def smallest_n_convex_polygon (n : ℕ) : Prop :=
  ∃ (polygon : Finₓ n → ℝ × ℝ), 
    (is_convex polygon) ∧
    (∀ i j, i ≠ j → dist (polygon i) (polygon j) ≠ dist (polygon i) (polygon j)) ∧
    (∀ i, sin_angle (polygon i) = constant_sine)

theorem exists_n_equal_sine_different_sides : ∃ (n : ℕ), smallest_n_convex_polygon n ∧
  ∀ m : ℕ, m < n → ¬(smallest_n_convex_polygon m) := by
  sorry

#eval exists_n_equal_sine_different_sides -- The Lean theorem prover can help check for n = 5

end exists_n_equal_sine_different_sides_l218_218274


namespace part1_part2_l218_218489

-- Statements derived from Step c)
theorem part1 {m : ℝ} (h : ∃ x : ℝ, m - |5 - 2 * x| - |2 * x - 1| = 0) : 4 ≤ m := by
  sorry

theorem part2 {x : ℝ} (hx : |x - 3| + |x + 4| ≤ 8) : -9 / 2 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

end part1_part2_l218_218489


namespace game_necessarily_ends_no_winning_strategy_for_starting_player_l218_218646

-- Condition Definitions
constant num_cards : ℕ
constant initial_configuration : vector finite_card num_cards
constant legal_move : ℕ → bool

-- Definitions of the conditions in the problem
def game_conditions : Prop :=
  ∀ (N : ℕ), N > 0 → (vector.repeat finite_card.gold N) == initial_configuration

def game_rule : Prop :=
  ∀ (N : ℕ), N ≥ 50 → vector.repeat finite_card.gold N.nth 0 = finite_card.gold → legal_move (N - 50) = tt

-- Proof statements
theorem game_necessarily_ends : ∀ N, game_conditions N → game_rule N → ∃ m, legal_move m = ff := sorry

theorem no_winning_strategy_for_starting_player : ∀ N, game_conditions N → game_rule N → ¬∃ str, winning_strategy str := sorry

end game_necessarily_ends_no_winning_strategy_for_starting_player_l218_218646


namespace sum_of_squares_of_perpendicular_chords_l218_218545

theorem sum_of_squares_of_perpendicular_chords (R r : ℝ) (h : r^2 ≤ R^2) :
  ∀ (A : EuclideanSpace ℝ [1]) 
    (P Q : EuclideanSpace ℝ [1] → EuclideanSpace ℝ [1])
    (ha : (hamdist O A = r)),
    ∃ (O : EuclideanSpace ℝ [1]), 
    ∀ proj_x proj_y : EuclideanSpace ℝ [1] → ℝ, 
    (proj_x, proj_y) = (P, Q) → 
    ∀ (c1 c2 : ℝ) (m1 m2 : ℝ),
    (c1 ^ 2 + c2 ^ 2 = r^2) →
    (m1 = 2 * sqrt (R^2 - c1 ^ 2)) →
    (m2 = 2 * sqrt (R^2 - c2 ^ 2)) →
    ((m1 ^ 2 + m2 ^ 2) = 8 * R ^ 2 - 4 * r ^ 2) := 
sorry

end sum_of_squares_of_perpendicular_chords_l218_218545


namespace hyperbola_eccentricity_eq_2sqrt3_div_3_l218_218180

variable {a b : ℝ} (ha : a > 0) (hb : b > 0)

def hyperbola : Prop := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

def asymptote_distance (d : ℝ) : Prop := 
  let a_vertex := (a, 0)
  let asymptote := λ x y: ℝ, b * x + a * y = 0
  abs(b * a + a * 0) / sqrt(b^2 + a^2) = d

def eccentricity (e : ℝ) : Prop := e = sqrt(a^2 + b^2) / a

theorem hyperbola_eccentricity_eq_2sqrt3_div_3 :
  ∀ a b : ℝ, ha : a > 0 → hb : b > 0 →
  hyperbola a b → asymptote_distance a b (a / 2) →
  eccentricity a b (2 * sqrt 3 / 3) :=
by
  intros
  sorry

end hyperbola_eccentricity_eq_2sqrt3_div_3_l218_218180


namespace radius_of_first_cylinder_l218_218174

theorem radius_of_first_cylinder :
  ∀ (rounds1 rounds2 : ℕ) (r2 r1 : ℝ), rounds1 = 70 → rounds2 = 49 → r2 = 20 → 
  (2 * Real.pi * r1 * rounds1 = 2 * Real.pi * r2 * rounds2) → r1 = 14 :=
by
  sorry

end radius_of_first_cylinder_l218_218174


namespace product_of_primes_l218_218085

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l218_218085


namespace sum_of_sequence_l218_218705

noncomputable def sequence (n : ℕ) : ℕ :=
if h : n > 0 then
  Nat.recOn n 1 (λ k ak, 2 * ak + k - 1)
else 0

def sequence_sum (n : ℕ) : ℕ :=
(nat.range n).sum (λ i, sequence (i + 1))

theorem sum_of_sequence (n : ℕ) :
  sequence_sum n = 2^(n+1) - 2 - (n * (n + 1)) / 2 := by
  sorry

end sum_of_sequence_l218_218705


namespace polar_equation_of_circle_length_of_segment_PQ_l218_218222

-- Definition of the parametric equations of circle C
def parametric_circle (φ : ℝ) : ℝ × ℝ := (cos φ, 1 + sin φ)

-- Definition of the polar equation of circle C
def polar_circle : ℝ × ℝ → Prop :=
  λ (ρ θ : ℝ), ρ = 2 * sin θ

-- Definition of the polar equation of line l
def polar_line (ρ θ : ℝ) : Prop :=
  2 * ρ * sin (θ + π / 6) = 3 * sqrt 3

-- Definition of the ray OM with θ = π / 6
def ray_OM (θ : ℝ) : Prop :=
  θ = π / 6

-- Prove the polar equation of circle C
theorem polar_equation_of_circle :
  ∀ φ : ℝ,
    let (x, y) := parametric_circle φ in
    ∃ ρ θ,
      (ρ = sqrt (x ^ 2 + y ^ 2) ∧ θ = arctan (y / x)) ∧
      polar_circle ρ θ :=
sorry

-- Prove the length of segment PQ
theorem length_of_segment_PQ :
  ∃ ρ1 ρ2 θ1 θ2,
    (ρ1 = 2 * sin (π / 6) ∧ θ1 = π / 6) ∧
    (polar_line ρ2 θ2 ∧ θ2 = π / 6) ∧
    ρ1 = 1 ∧ ρ2 = 3 ∧
    abs (ρ1 - ρ2) = 2 :=
sorry

end polar_equation_of_circle_length_of_segment_PQ_l218_218222


namespace competition_results_correct_l218_218456

theorem competition_results_correct :
  ∃ (first second third fourth : String), 
    (first = "Oleg" ∧ second = "Olya" ∧ third = "Polya" ∧ fourth = "Pasha") ∧
    ∀ (claims : String → String → Prop),
      (claims "Olya" "all_odd_places_boys") ∧ 
      (claims "Oleg" "consecutive_places_with_olya") ∧
      (claims "Pasha" "all_odd_places_names_start_O") ∧
      ∃ (truth_teller : String), 
        truth_teller = "Oleg" ∧ 
        (claims "Oleg" "first_place") ∧ 
        ¬ (claims "Olya" "first_place") ∧ 
        ¬ (claims "Pasha" "first_place") ∧ 
        ¬ (claims "Polya" "first_place") :=
sorry

end competition_results_correct_l218_218456


namespace connie_num_markers_l218_218211

def num_red_markers (T : ℝ) := 0.41 * T
def num_total_markers (num_blue_markers : ℝ) (T : ℝ) := num_red_markers T + num_blue_markers

theorem connie_num_markers (T : ℝ) (h1 : num_total_markers 23 T = T) : T = 39 :=
by
sorry

end connie_num_markers_l218_218211


namespace distinct_flags_count_l218_218163

def flags := {colors : list (option ℕ) // colors.length = 3 ∧ ∀ i < 2, colors.nth i ≠ colors.nth (i + 1)}

def countFlags : ℕ :=
  (5 : ℕ) * 4 * 3

theorem distinct_flags_count : countFlags = 60 := 
  by sorry

end distinct_flags_count_l218_218163


namespace inequality_condition_l218_218381

theorem inequality_condition
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 2015) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / Real.sqrt 2015 :=
by
  sorry

end inequality_condition_l218_218381


namespace verify_placements_l218_218453

-- Definitions for participants and their possible places
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

-- Each participant should be mapped to a place (1, 2, 3, 4)
def Place : Participant → ℕ := λ p,
  match p with
  | Participant.Olya => 2
  | Participant.Oleg => 1
  | Participant.Polya => 3
  | Participant.Pasha => 4

-- Conditions based on the problem statement
def statement_Olya : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Polya % 2 = 1 ∧ Place Participant.Pasha % 2 = 1)

def statement_Oleg : Prop :=
  (abs (Place Participant.Oleg - Place Participant.Olya) = 1)

def statement_Pasha : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Olya % 2 = 1 ∧ Place Participant.Polya % 2 = 1)

-- Only one child tells the truth and the others lie
def exactly_one_true (a b c : Prop) : Prop := (a ∨ b ∨ c) ∧ (a → ¬b ∧ ¬c) ∧ (b → ¬a ∧ ¬c) ∧ (c → ¬a ∧ ¬b)

-- The main theorem to be proven
theorem verify_placements :
  exactly_one_true (statement_Olya) (statement_Oleg) (statement_Pasha) ∧ 
  Place Participant.Olya = 2 ∧
  Place Participant.Oleg = 1 ∧
  Place Participant.Polya = 3 ∧
  Place Participant.Pasha = 4 :=
by
  sorry

end verify_placements_l218_218453


namespace angle_between_vectors_pi_over_4_l218_218764

variables {V : Type} [inner_product_space ℝ V]

def vectors_angle θ (a b : V) : Prop :=
  (∥a∥ = 1) ∧ (∥b∥ = real.sqrt 2) ∧ (inner a (a - b) = 0) ∧ (real.arccos (inner a b / (∥a∥ * ∥b∥)) = θ)

theorem angle_between_vectors_pi_over_4 (a b : V) :
  vectors_angle (real.pi / 4) a b :=
sorry

end angle_between_vectors_pi_over_4_l218_218764


namespace fuel_proof_problem_l218_218970

-- 1. Define the fuel consumption function
def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

-- 2. Define the travel distance
def distance : ℝ := 100

-- 3. Define the function for total fuel consumption for given speed
def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (distance / x)

-- 4. Given conditions: maximum speed and valid speed range
def max_speed : ℝ := 120
def valid_speed (x : ℝ) : Prop := 0 < x ∧ x ≤ max_speed

-- Lean 4 statement for the proof problem
theorem fuel_proof_problem :
  let fuel_40 := total_fuel 40 in
  valid_speed 40 → fuel_40 = 17.5 ∧ 
  (∃ x, valid_speed x ∧ ∀ y, valid_speed y → total_fuel x ≤ total_fuel y ∧ total_fuel x = 11.25) := 
by 
  intro h_valid_speed_40
  have h_fuel_40 : total_fuel 40 = 17.5 := sorry
  
  let min_fuel_speed : ℝ := 80
  have h_valid_min_speed : valid_speed min_fuel_speed := sorry
  have h_min_fuel : total_fuel min_fuel_speed = 11.25 := sorry
  have min_property : ∀ y, valid_speed y → total_fuel min_fuel_speed ≤ total_fuel y := sorry

  exact ⟨⟨h_fuel_40, ⟨min_fuel_speed, h_valid_min_speed, min_property, h_min_fuel⟩⟩⟩

end fuel_proof_problem_l218_218970


namespace piles_can_be_combined_l218_218440

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l218_218440


namespace arbitrary_large_circle_existence_every_circle_meets_a_line_l218_218698

-- Finite collection of lines
theorem arbitrary_large_circle_existence 
  (P : Type) [plane P] (L : list (line P)) : 
  ∃ (C : circle P), (C.radius = arbitrary_large_radius) ∧ ∀ l ∈ L, ¬(circle_intersects_line C l) :=
begin
  sorry
end

-- Countable infinite sequence of lines
theorem every_circle_meets_a_line 
  (P : Type) [plane P] (L : ℕ → line P) 
  (hL : ∀ n, L n = {p : P | p.y = (rat.cast ∘ nat_to_rat) n}) : 
  ∀ C : circle P, ∃ n : ℕ, circle_intersects_line C (L n) :=
begin
  sorry
end

end arbitrary_large_circle_existence_every_circle_meets_a_line_l218_218698


namespace verify_placements_l218_218450

-- Definitions for participants and their possible places
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

-- Each participant should be mapped to a place (1, 2, 3, 4)
def Place : Participant → ℕ := λ p,
  match p with
  | Participant.Olya => 2
  | Participant.Oleg => 1
  | Participant.Polya => 3
  | Participant.Pasha => 4

-- Conditions based on the problem statement
def statement_Olya : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Polya % 2 = 1 ∧ Place Participant.Pasha % 2 = 1)

def statement_Oleg : Prop :=
  (abs (Place Participant.Oleg - Place Participant.Olya) = 1)

def statement_Pasha : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Olya % 2 = 1 ∧ Place Participant.Polya % 2 = 1)

-- Only one child tells the truth and the others lie
def exactly_one_true (a b c : Prop) : Prop := (a ∨ b ∨ c) ∧ (a → ¬b ∧ ¬c) ∧ (b → ¬a ∧ ¬c) ∧ (c → ¬a ∧ ¬b)

-- The main theorem to be proven
theorem verify_placements :
  exactly_one_true (statement_Olya) (statement_Oleg) (statement_Pasha) ∧ 
  Place Participant.Olya = 2 ∧
  Place Participant.Oleg = 1 ∧
  Place Participant.Polya = 3 ∧
  Place Participant.Pasha = 4 :=
by
  sorry

end verify_placements_l218_218450


namespace TeamC_fee_l218_218653

structure Team :=
(work_rate : ℚ)

def teamA : Team := ⟨1 / 36⟩
def teamB : Team := ⟨1 / 24⟩
def teamC : Team := ⟨1 / 18⟩

def total_fee : ℚ := 36000

def combined_work_rate_first_half (A B C : Team) : ℚ :=
(A.work_rate + B.work_rate + C.work_rate) * 1 / 2

def combined_work_rate_second_half (A C : Team) : ℚ :=
(A.work_rate + C.work_rate) * 1 / 2

def total_work_completed_by_TeamC (A B C : Team) : ℚ :=
C.work_rate * combined_work_rate_first_half A B C + C.work_rate * combined_work_rate_second_half A C

theorem TeamC_fee (A B C : Team) (total_fee : ℚ) :
  total_work_completed_by_TeamC A B C * total_fee = 20000 :=
by
  sorry

end TeamC_fee_l218_218653


namespace refills_count_l218_218765

variable (spent : ℕ) (cost : ℕ)

theorem refills_count (h1 : spent = 40) (h2 : cost = 10) : spent / cost = 4 := 
by
  sorry

end refills_count_l218_218765


namespace correct_choice_l218_218711

variables {m n : Type} {α : Type} {a b c : ℝ}

-- Proposition p
def p := (parallel m n) ∧ (subset (set n) (set α)) → (parallel m α)

-- Proposition q
def q := a > b → a * c > b * c

-- Translate the correct answer to Lean
theorem correct_choice : ¬p ∨ q :=
by
  sorry

end correct_choice_l218_218711


namespace product_of_two_numbers_l218_218039

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x ^ 2 + y ^ 2 = 289)
  (h2 : x + y = 23) : 
  x * y = 120 :=
by
  sorry

end product_of_two_numbers_l218_218039


namespace triangle_area_of_tangent_line_l218_218946

theorem triangle_area_of_tangent_line (l : set (ℝ × ℝ)) (a b : ℝ) :
  (∀ p : ℝ × ℝ, p ∈ l → p.1^2 + p.2^2 = 1) →  -- l is tangent to the circle
  a + b = sqrt 3 →                            -- sum of intercepts is sqrt(3)
  abs a > 1 →                                  -- |a| > 1 due to tangency
  abs b > 1 →                                  -- |b| > 1 due to tangency
  (∃ (x y : ℝ), l = { p | p.1 / a + p.2 / b = 1 }) → -- Line in intercept form
  (1 / 2) * abs (a * b) = 3 / 2 :=             -- Area of the triangle is 3/2
  by
  sorry  -- Proof placeholder

end triangle_area_of_tangent_line_l218_218946


namespace small_cone_altitude_l218_218605

noncomputable def frustum_height : ℝ := 18
noncomputable def lower_base_area : ℝ := 400 * Real.pi
noncomputable def upper_base_area : ℝ := 100 * Real.pi

theorem small_cone_altitude (h_frustum : frustum_height = 18) 
    (A_lower : lower_base_area = 400 * Real.pi) 
    (A_upper : upper_base_area = 100 * Real.pi) : 
    ∃ (h_small_cone : ℝ), h_small_cone = 18 := 
by
  sorry

end small_cone_altitude_l218_218605


namespace piles_to_single_pile_l218_218430

-- Define the condition similar_sizes
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

-- Define the inductive step of combining stones
def combine_stones (piles : List ℕ) : List ℕ :=
  if ∃ x y, x ∈ piles ∧ y ∈ piles ∧ similar_sizes x y then
    let ⟨x, hx, y, hy, hsim⟩ := Classical.some_spec (Classical.some_spec_exists _)
    List.cons (x + y) (List.erase (List.erase piles x) y)
  else
    piles

-- Prove that a collection of piles can be reduced to a single pile of size n
theorem piles_to_single_pile (piles : List ℕ) (h : ∀ x ∈ piles, x = 1) : 
  ∃ p, list.length (Iterator.iterate combine_stones piles.count) 1 = 1 := by
  sorry

end piles_to_single_pile_l218_218430


namespace cube_hamiltonian_cycles_l218_218568

-- Defining the problem

def number_of_ways (d: Type) [fintype d] (edges: set (d × d)) (start: d) : ℕ :=
  -- Some function representing the number of Hamiltonian cycles on a given cube
  sorry

-- Cube faces and edges
inductive face : Type
| one : face
| two : face
| three : face
| four : face
| five : face
| six : face

open face

def cube_edges : set (face × face) :=
  { (one, two), (one, three), (one, four), (one, five),
    (two, one), (two, three), (two, six), (two, four),
    (three, one), (three, two), (three, five), (three, six),
    (four, one), (four, two), (four, six), (four, five),
    (five, one), (five, three), (five, four), (five, six),
    (six, two), (six, three), (six, four), (six, five) }

theorem cube_hamiltonian_cycles : number_of_ways face cube_edges one = 32 :=
sorry

end cube_hamiltonian_cycles_l218_218568


namespace product_of_odd_neg_integers_l218_218912

theorem product_of_odd_neg_integers (n : ℕ) (h1 : n = 1011) :
  let s := finset.range n
  let neg_odds := (s.map (λ x, -((2 * x) + 1))).attach_val
  neg_odds.prod (λ x, x) < 0 ∧ 
  (neg_odds.prod (λ x, x) % 100 = 25) :=
by
  sorry

end product_of_odd_neg_integers_l218_218912


namespace product_calculation_l218_218986

theorem product_calculation :
  5 * 13 * 31 * 73 * 137 = 20152015 :=
begin
  sorry
end

end product_calculation_l218_218986


namespace volume_of_sand_filled_is_1600_l218_218352

-- Definitions for the given conditions
def length := 40
def width := 20
def height := 2

-- The volume definition
def volume := length * width * height

-- The theorem to prove the volume equals the expected answer
theorem volume_of_sand_filled_is_1600 : volume = 1600 := by
  unfold volume
  norm_num
  sorry

end volume_of_sand_filled_is_1600_l218_218352


namespace find_original_speed_l218_218181

noncomputable def wheel_speed (radius_in_feet : ℝ) (initial_speed_mph : ℝ) (increased_speed_mph : ℝ) (time_hours : ℝ) (shortened_time_seconds : ℝ) :=
  let circumference_miles := radius_in_feet / 5280
  let total_time_seconds := 3600
  let initial_time := circumference_miles / initial_speed_mph
  let shortened_time := initial_time - (shortened_time_seconds / total_time_seconds)
  let increased_speed_time_product := (initial_speed_mph + increased_speed_mph) * shortened_time
  let initial_speed_time_product := initial_speed_mph * initial_time
  in increased_speed_time_product = initial_speed_time_product

theorem find_original_speed (radius_in_feet : ℝ) (increased_speed_mph : ℝ) (shortened_time_seconds : ℝ) :
  (∃ (r : ℝ), wheel_speed radius_in_feet r increased_speed_mph (radius_in_feet / (5280 * r)) shortened_time_seconds) → (r = 20) :=
by
  sorry

end find_original_speed_l218_218181


namespace sum_four_digit_numbers_l218_218260

theorem sum_four_digit_numbers : 
  let digits := [1, 2, 3, 4, 5]
  let perms := digits.permutations
  ∑ p in perms.filter (λ x, x.length = 4), (1000 * x.head + 100 * x[1] + 10 * x[2] + x[3]) = 399960 := 
by sorry

end sum_four_digit_numbers_l218_218260


namespace seeds_sum_77_l218_218003

/-- Paige plants different amounts of seeds in 9 flowerbeds. -/
theorem seeds_sum_77 :
  let seed1 := 18 in
  let seed2 := 22 in
  let seed3 := 30 in
  let seed4 := 2 * seed1 in
  let seed5 := seed3 in
  let seed6 := seed2 / 2 in
  let seed7 := seed1 in
  let seed8 := seed4 in
  let seed9 := seed3 - 1 in
  seed1 + seed5 + seed9 = 77 :=
begin
  -- all conditions and calculations have been translated directly from the problem statement
  let seed1 := 18,
  let seed2 := 22,
  let seed3 := 30,
  let seed4 := 2 * seed1,
  let seed5 := seed3,
  let seed6 := seed2 / 2,
  let seed7 := seed1,
  let seed8 := seed4,
  let seed9 := seed3 - 1,
  show seed1 + seed5 + seed9 = 77,
  -- proof is not required
  sorry
end

end seeds_sum_77_l218_218003


namespace pile_division_possible_l218_218419

theorem pile_division_possible (n : ℕ) :
  ∃ (division : list ℕ), (∀ x ∈ division, x = 1) ∧ division.sum = n :=
by
  sorry

end pile_division_possible_l218_218419


namespace product_of_smallest_primes_l218_218117

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218117


namespace cosine_periodicity_l218_218759

theorem cosine_periodicity (α : ℝ) : ∃ k, k = 8 ∧ (cos (α + k * (π / 4)) = cos α) :=
by
  sorry

end cosine_periodicity_l218_218759


namespace sin_add_pi_over_three_l218_218691

theorem sin_add_pi_over_three (α : ℝ) (h : Real.sin (α - 2 * Real.pi / 3) = 1 / 4) : 
  Real.sin (α + Real.pi / 3) = -1 / 4 := by
  sorry

end sin_add_pi_over_three_l218_218691


namespace february_1_day_l218_218574

def january_has_four_mondays_and_fridays (year : ℕ) : Prop :=
∃ (jan_1 : DayOfWeek),
  (january_days := [1, 2, 3, 4, 5, 6, 7].map (λ d, add_days jan_1 (d - 1))) ∧
  (count_days jan_1 monday january_days = 4) ∧
  (count_days jan_1 friday january_days = 4)

theorem february_1_day (year : ℕ) (h : january_has_four_mondays_and_fridays year) :
  (feb_1 := add_days (jan_1 year) 31) ∈ {thursday, friday} :=
sorry

end february_1_day_l218_218574


namespace car_speed_is_100_l218_218533

def avg_speed (d1 d2 t: ℕ) := (d1 + d2) / t = 80

theorem car_speed_is_100 
  (x : ℕ)
  (speed_second_hour : ℕ := 60)
  (total_time : ℕ := 2)
  (h : avg_speed x speed_second_hour total_time):
  x = 100 :=
by
  unfold avg_speed at h
  sorry

end car_speed_is_100_l218_218533


namespace general_equation_of_curve_sum_of_squares_reciprocals_l218_218580

-- Problem 1
theorem general_equation_of_curve (a : ℝ) (φ : ℝ) (h_pos : 0 < a) :
    (∃ (x y : ℝ), x = a * cos φ ∧ y = sqrt 3 * sin φ ∧ ((∃ t, x = 3 + t ∧ y = -1 - t) ∧ y = 0)) → 
    (frac (x^2) (4) + frac (y^2) (3) = 1) :=
sorry

-- Problem 2
theorem sum_of_squares_reciprocals (θ : ℝ) (ρ1 ρ2 ρ3 : ℝ)
    (hA : (ρ1 * cos θ, ρ1 * sin θ) ∈ C)
    (hB : (ρ2 * cos (θ + 2 * π / 3), ρ2 * sin (θ + 2 * π / 3)) ∈ C)
    (hC : (ρ3 * cos (θ + 4 * π / 3), ρ3 * sin (θ + 4 * π / 3)) ∈ C) :
    (frac(1, ρ1^2) + frac(1, ρ2^2) + frac(1, ρ3^2) = frac(7, 8)) :=
sorry

end general_equation_of_curve_sum_of_squares_reciprocals_l218_218580


namespace combine_piles_l218_218427

theorem combine_piles (n : ℕ) (piles : list ℕ) (h_piles : list.sum piles = n) (h_similar : ∀ x y ∈ piles, x ≤ y → y ≤ 2 * x) :
  ∃ pile, pile ∈ piles ∧ pile = n := sorry

end combine_piles_l218_218427


namespace hypotenuse_length_ABC_l218_218805

noncomputable def triangle_ABC (A B C : Point) :=
  ∠ B = 90 ∧ ∠ C = 30 ∧ (dist A C) = 6

theorem hypotenuse_length_ABC (A B C : Point) :
  triangle_ABC A B C → (dist A B) = 12 :=
by
  intro h
  sorry

end hypotenuse_length_ABC_l218_218805


namespace ω_value_monotone_decreasing_interval_l218_218316

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * ω * x) - 2 * Real.sin (ω * x) ^ 2 + 1

-- The given conditions
axiom ω_pos (ω : ℝ) : ω > 0
axiom period_π (ω : ℝ) : ∃ T > 0, (∀ x, f ω (x + T) = f ω x) ∧ T = Real.pi

-- The questions (equivalent proof statements)
theorem ω_value : ∀ ω > 0, (∃ T > 0, (∀ x, f ω (x + T) = f ω x) ∧ T = Real.pi) → ω = 1 :=
by
  intros ω h₁ h₂
  sorry

theorem monotone_decreasing_interval : 
  ∀ x k : ℤ, (∀ x, f 1 (x + Real.pi) = f 1 x) → ∀ x ∈ (Set.Icc (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8)), Real.sin (2*x + Real.pi/4) < Real.sin (2*x + Real.pi/4) :=
by
  intros x k h₁ h₂ h₃ h₄
  sorry

end ω_value_monotone_decreasing_interval_l218_218316


namespace similar_sizes_combination_possible_l218_218403

theorem similar_sizes_combination_possible 
    (similar : Nat → Nat → Prop := λ x y, x ≤ y ∧ y ≤ 2 * x)
    (combine_piles : List Nat → Nat ∃ combined : Nat, (∀ x y ∈ combined, similar x y) → True
    (piles : List Nat) : True :=
sorry

end similar_sizes_combination_possible_l218_218403


namespace points_in_S_three_or_five_l218_218378

-- Definitions as per conditions in a)
def point := ℝ × ℝ

def segment_bisector (P Q : point) : set point := 
  {M : point | M = ((fst P + fst Q) / 2, (snd P + snd Q) / 2)}

def finite_subset_with_properties (S : set point) : Prop :=
  (S.finite ∧ S.nonempty ∧ ∀ P Q : point, P ∈ S → Q ∈ S → P ≠ Q →
    ∃ M ∈ S, M ∈ segment_bisector P Q) ∧
  (∀ P1 Q1 P2 Q2 P3 Q3 : point, 
    P1 ∈ S → Q1 ∈ S → P2 ∈ S → Q2 ∈ S → P3 ∈ S → Q3 ∈ S → 
    (P1 ≠ Q1 ∧ P2 ≠ Q2 ∧ P3 ≠ Q3 ∧ 
     segment_bisector P1 Q1 ≠ segment_bisector P2 Q2 ∧ 
     segment_bisector P2 Q2 ≠ segment_bisector P3 Q3 ∧ 
     segment_bisector P3 Q3 ≠ segment_bisector P1 Q1) → 
    ¬ ∃ M ∈ S, M ∈ segment_bisector P1 Q1 ∧ 
               M ∈ segment_bisector P2 Q2 ∧ 
               M ∈ segment_bisector P3 Q3)

-- Lean 4 statement for the proof problem
theorem points_in_S_three_or_five (S : set point) (h : finite_subset_with_properties S) :
  S.to_finset.card = 3 ∨ S.to_finset.card = 5 := sorry

end points_in_S_three_or_five_l218_218378


namespace PQ_length_l218_218383

noncomputable def P : ℚ × ℚ := (2/3, 4/3)
noncomputable def Q : ℚ × ℚ := (20/3, 16/33)
def R : ℚ × ℚ := (10, 8)

def length (p q : ℚ × ℚ) : ℚ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem PQ_length {P Q R : ℚ × ℚ}
  (h1 : 9 * P.2 = 18 * P.1)
  (h2 : 11 * Q.2 = 4 * Q.1)
  (hR : R = (10, 8))
  (h_mid : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  length P Q = real.sqrt(56512) / 33 :=
by
   sorry

end PQ_length_l218_218383


namespace initial_concentrated_kola_percentage_l218_218588

theorem initial_concentrated_kola_percentage :
  ∃ c : ℝ, 
    let initial_water := 0.80 * 340 in
    let initial_volume := 340 in
    let added_sugar := 3.2 in
    let added_water := 10 in
    let added_conc_kola := 6.8 in
    let final_volume := initial_volume + added_sugar + added_water + added_conc_kola in
    let final_sugar_percentage := 14.111111111111112 / 100 in
    let final_sugar_volume := final_sugar_percentage * final_volume in
    let initial_sugar := initial_volume - initial_water - c * initial_volume in
    initial_sugar + added_sugar = final_sugar_volume → 
    c = 0.06 :=
sorry

end initial_concentrated_kola_percentage_l218_218588


namespace phi_exp_alpha_le_2_characteristic_phi_exp_alpha_gt_2_not_characteristic_phi_rational_alpha_le_2_characteristic_phi_rational_alpha_gt_2_not_characteristic_phi_piecewise_beta_ge_1_characteristic_phi_piecewise_beta_between_0_and_1_not_characteristic_phi_piecewise_linear_half_characteristic_phi_polynomial_3_not_characteristic_l218_218137

section 

noncomputable def is_characteristic_function (ϕ : ℝ → ℝ) : Prop :=
  ∃ (ξ : ℝ), ϕ = λ t, Real.exp (- |t| ^ ξ)

-- Prove that each function is or is not a characteristic function

theorem phi_exp_alpha_le_2_characteristic (α : ℝ) (h : 0 < α ∧ α ≤ 2) :
  is_characteristic_function (λ t, Real.exp (- |t|^ α)) :=
sorry

theorem phi_exp_alpha_gt_2_not_characteristic (α : ℝ) (h : α > 2) :
  ¬ is_characteristic_function (λ t, Real.exp (- |t|^ α)) :=
sorry

theorem phi_rational_alpha_le_2_characteristic (α β : ℝ) (hα : 0 < α ∧ α ≤ 2) (hβ : 0 < β) :
  is_characteristic_function (λ t, (1 + |t|^ α) ^ -β) :=
sorry

theorem phi_rational_alpha_gt_2_not_characteristic (α β : ℝ) (hα : α > 2) (hβ : 0 < β) :
  ¬ is_characteristic_function (λ t, (1 + |t|^ α) ^ -β) :=
sorry

theorem phi_piecewise_beta_ge_1_characteristic (β : ℝ) (hβ : β ≥ 1) :
  is_characteristic_function (λ t, if |t| ≤ 1 then (1 - |t|) ^ β else 0) :=
sorry

theorem phi_piecewise_beta_between_0_and_1_not_characteristic (β : ℝ) (hβ : 0 < β ∧ β < 1) :
  ¬ is_characteristic_function (λ t, if |t| ≤ 1 then (1 - |t|) ^ β else 0) :=
sorry

theorem phi_piecewise_linear_half_characteristic :
  is_characteristic_function (λ t, if |t| ≤ 1 / 2 then 1 - |t| else 1 / (4 * |t|)) :=
sorry

theorem phi_polynomial_3_not_characteristic :
  ¬ is_characteristic_function (λ t, if |t| ≤ 1 then 1 - |t|^3 else 0) :=
sorry

end

end phi_exp_alpha_le_2_characteristic_phi_exp_alpha_gt_2_not_characteristic_phi_rational_alpha_le_2_characteristic_phi_rational_alpha_gt_2_not_characteristic_phi_piecewise_beta_ge_1_characteristic_phi_piecewise_beta_between_0_and_1_not_characteristic_phi_piecewise_linear_half_characteristic_phi_polynomial_3_not_characteristic_l218_218137


namespace alice_coin_problem_l218_218976

theorem alice_coin_problem :
  ∀ (d : ℕ), 1 ≤ d ∧ d ≤ 757 → (3030 = d + (3030 - d)) ∧ (3030 - d) ≥ 3 * d → 
  let total_cents (d : ℕ) := 15150 + 5 * d
  in total_cents 757 - total_cents 1 = 3780 :=
sorry

end alice_coin_problem_l218_218976


namespace smallest_value_of_n_l218_218493

noncomputable def minFactorsOf10 (a b c : ℕ) : ℕ :=
  let n_f (k : ℕ) : ℕ := k / 5 + k / 25 + k / 125 + k / 625 + k / 3125
  n_f(a) + n_f(b) + n_f(c)

theorem smallest_value_of_n (a b c : ℕ) (m n : ℕ) (h1 : a + b + c = 3000)
  (h2 : nat.factorial a * nat.factorial b * nat.factorial c = m * 10 ^ n)
  (h3 : m % 10 ≠ 0) : n = 747 :=
begin
  sorry
end

end smallest_value_of_n_l218_218493


namespace order_of_numbers_l218_218520

theorem order_of_numbers (a b c : ℝ) (h1 : a = 6^0.5) (h2 : b = 0.5^6) (h3 : c = Real.log 6 / Real.log 0.5) : 
  c < b ∧ b < a :=
by {
  have h4 : a > 1, from sorry,
  have h5 : 0 < b ∧ b < 1, from sorry,
  have h6 : c < 0, from sorry,
  exact ⟨h6, h5.2.trans h4⟩,
}

end order_of_numbers_l218_218520


namespace cost_per_day_is_18_l218_218871

def cost_per_day_first_week (x : ℕ) : Prop :=
  let cost_per_day_rest_week := 12
  let total_days := 23
  let total_cost := 318
  let first_week_days := 7
  let remaining_days := total_days - first_week_days
  (first_week_days * x) + (remaining_days * cost_per_day_rest_week) = total_cost

theorem cost_per_day_is_18 : cost_per_day_first_week 18 :=
  sorry

end cost_per_day_is_18_l218_218871


namespace proof_problem_l218_218695

noncomputable def circle_eq (D E F : ℝ) : ℝ → ℝ → ℝ := 
  λ (x y : ℝ), x^2 + y^2 + D*x + E*y + F

def circle_in_second_quadrant (a b : ℝ) : Prop :=
  a < 0 ∧ b > 0

def radius_condition (a b : ℝ) : Prop :=
  (a^2 + b^2 = 2)

def tangent_to_lines (a b : ℝ) : Prop :=
  (λ (a b: ℝ), a = -√2 ∧ (|3*a + 4*b| / 5 = √2))

def correct_D_E_F : Prop := 
  ∀ D E F : ℝ,
  (circle_in_second_quadrant (-√2) (2*√2)) → 
  (radius_condition (-√2) (2*√2)) → 
  (tangent_to_lines (-√2) (2*√2)) →
  (D = 2*√2 ∧ E = -4*√2 ∧ F = 8)

def distance_from_line (a b : ℝ) : ℝ :=
  |a-b+2*√2| / √2

def correct_AB_length : Prop := 
  ∀ a b : ℝ,
  (a = -√2 ∧ b = 2*√2) →
  (distance_from_line a b = 1) →
  (2*√(2-1) = 2)

theorem proof_problem :
  correct_D_E_F ∧ correct_AB_length :=
begin
  sorry,
end

end proof_problem_l218_218695


namespace product_of_smallest_primes_l218_218091

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218091


namespace ryan_flyers_l218_218202

theorem ryan_flyers (total_flyers : ℕ) (alyssa_flyers : ℕ) (scott_flyers : ℕ) (belinda_percentage : ℚ) (belinda_flyers : ℕ) (ryan_flyers : ℕ)
  (htotal : total_flyers = 200)
  (halyssa : alyssa_flyers = 67)
  (hscott : scott_flyers = 51)
  (hbelinda_percentage : belinda_percentage = 0.20)
  (hbelinda : belinda_flyers = belinda_percentage * total_flyers)
  (hryan : ryan_flyers = total_flyers - (alyssa_flyers + scott_flyers + belinda_flyers)) :
  ryan_flyers = 42 := by
    sorry

end ryan_flyers_l218_218202


namespace competition_results_correct_l218_218458

theorem competition_results_correct :
  ∃ (first second third fourth : String), 
    (first = "Oleg" ∧ second = "Olya" ∧ third = "Polya" ∧ fourth = "Pasha") ∧
    ∀ (claims : String → String → Prop),
      (claims "Olya" "all_odd_places_boys") ∧ 
      (claims "Oleg" "consecutive_places_with_olya") ∧
      (claims "Pasha" "all_odd_places_names_start_O") ∧
      ∃ (truth_teller : String), 
        truth_teller = "Oleg" ∧ 
        (claims "Oleg" "first_place") ∧ 
        ¬ (claims "Olya" "first_place") ∧ 
        ¬ (claims "Pasha" "first_place") ∧ 
        ¬ (claims "Polya" "first_place") :=
sorry

end competition_results_correct_l218_218458


namespace product_of_primes_l218_218086

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l218_218086


namespace difference_in_costs_l218_218204

noncomputable def cost_per_capsule (cost : ℝ) (capsules : ℕ) : ℝ :=
  cost / capsules

theorem difference_in_costs :
  let R_cost := 6.25
  let R_capsules := 250
  let T_cost := 3.00
  let T_capsules := 100
  let X_cost := 7.50
  let X_capsules := 300
  let Y_cost := 4.00
  let Y_capsules := 120

  let cp_R := cost_per_capsule R_cost R_capsules
  let cp_T := cost_per_capsule T_cost T_capsules
  let cp_X := cost_per_capsule X_cost X_capsules
  let cp_Y := cost_per_capsule Y_cost Y_capsules

  cp_R = 0.025 ∧
  cp_T = 0.03 ∧
  cp_X = 0.025 ∧
  cp_Y ≈ 0.0333 ∧

  let diff_RT := cp_T - cp_R
  let diff_RX := cp_X - cp_R
  let diff_RY := cp_Y - cp_R
  let diff_TX := cp_X - cp_T
  let diff_TY := cp_Y - cp_T
  let diff_XY := cp_Y - cp_X

  diff_RT = 0.005 ∧
  diff_RX = 0.00 ∧
  diff_RY ≈ 0.0083 ∧
  diff_TX = 0.005 ∧
  diff_TY ≈ 0.0033 ∧
  diff_XY ≈ 0.0083 := by
  sorry

end difference_in_costs_l218_218204


namespace apple_price_33_kgs_l218_218189

theorem apple_price_33_kgs (l q : ℕ) (h1 : 30 * l + 6 * q = 366) (h2 : 15 * l = 150) : 
  30 * l + 3 * q = 333 :=
by
  sorry

end apple_price_33_kgs_l218_218189


namespace orthocenter_on_diagonal_l218_218346

variables (A B C D E : Type)
variables (AB BC : Prop)
variables (angle_ABE angle_DBC angle_EBD angle_AEB angle_BDC : ℝ)
variables (convex_pentagon : Prop)

-- Conditions
axiom H1 : AB = BC
axiom H2 : angle_ABE + angle_DBC = angle_EBD
axiom H3 : angle_AEB + angle_BDC = 180
axiom H4 : convex_pentagon = true

-- Goal: Prove that the orthocenter of triangle BDE lies on diagonal AC
theorem orthocenter_on_diagonal {A B C D E : Type} 
  (AB BC : Prop) (angle_ABE angle_DBC angle_EBD angle_AEB angle_BDC : ℝ)
  (convex_pentagon : Prop) 
  (H1 : AB = BC) 
  (H2 : angle_ABE + angle_DBC = angle_EBD) 
  (H3 : angle_AEB + angle_BDC = 180) 
  (H4 : convex_pentagon = true) :
  ∃ (O : Type), O lies_on line_AC ∧ is_orthocenter O (triangle B D E) :=
sorry

end orthocenter_on_diagonal_l218_218346


namespace rational_sum_zero_l218_218015

theorem rational_sum_zero {a b c : ℚ} (h : (a + b + c) * (a + b - c) = 4 * c^2) : a + b = 0 := 
sorry

end rational_sum_zero_l218_218015


namespace find_antecedent_l218_218786

-- Condition: The ratio is 4:6, simplified to 2:3
def ratio (a b : ℕ) : Prop := (a / gcd a b) = 2 ∧ (b / gcd a b) = 3

-- Condition: The consequent is 30
def consequent (y : ℕ) : Prop := y = 30

-- The problem is to find the antecedent
def antecedent (x : ℕ) (y : ℕ) : Prop := ratio x y

-- The theorem to be proved
theorem find_antecedent:
  ∃ x : ℕ, consequent 30 → antecedent x 30 ∧ x = 20 :=
by
  sorry

end find_antecedent_l218_218786


namespace Jill_total_time_l218_218813

theorem Jill_total_time 
  (t_jack_first_half : ℕ) (t_jack_second_half : ℕ) (t_jack_before_jill : ℕ)
  (jack_first_half : t_jack_first_half = 19)
  (jack_second_half : t_jack_second_half = 6)
  (jack_before_jill : t_jack_before_jill = 7) :
  let t_jack_total := t_jack_first_half + t_jack_second_half in
  let t_jill := t_jack_total + t_jack_before_jill in
  t_jill = 32 := 
by
  let t_jack_total := t_jack_first_half + t_jack_second_half
  let t_jill := t_jack_total + t_jack_before_jill
  show t_jill = 32
  sorry

end Jill_total_time_l218_218813


namespace angle_BED_in_isosceles_triangle_l218_218778

theorem angle_BED_in_isosceles_triangle 
  (A B C D E : Type) [triangle A B C]
  (hA : ∠ A = 60) (hC : ∠ C = 70) (hD_on_AB : D ∈ line_segment A B)
  (hE_on_BC : E ∈ line_segment B C) (hDB_BE : dist D B = dist B E) :
  ∠ B E D = 65 :=
by
  sorry

end angle_BED_in_isosceles_triangle_l218_218778


namespace course_selection_l218_218960

theorem course_selection (courses : Finset ℕ) (A B C : ℕ) (hA : A ∈ courses) (hB : B ∈ courses) (hC : C ∈ courses) :
  (courses.card = 9) →
  (∀ (x y : ℕ), x ∈ (Finset.singleton A ∪ Finset.singleton B ∪ Finset.singleton C) → y ∈ (Finset.singleton A ∪ Finset.singleton B ∪ Finset.singleton C) → x ≠ y) →
  (∃ (subcourses : Finset ℕ), subcourses.card = 4 ∧ ∀ x ∈ subcourses, x ∈ courses) →
  (1 ≤ (Finset.singleton A ∪ Finset.singleton B ∪ Finset.singleton C).card) →
  (C 3 1 * C 6 3 + C 6 4 = 75) :=
begin
  sorry
end

end course_selection_l218_218960


namespace sum_of_powers_and_arithmetic_series_l218_218210

theorem sum_of_powers_and_arithmetic_series :
  let i := complex.I in
  let series_sum := ∑ n in finset.range 104, n in
  let powers_sum := (finset.range 208).sum (λ k, i ^ (k - 103)) in
  powers_sum + series_sum = 5355 :=
begin
  -- We can start by defining i as complex.I
  let i := complex.I,
  
  -- Define the arithmetic series sum using finset sum
  let series_sum := ∑ n in finset.range 104, n,
  
  -- Define the sum of powers of i over the correct range
  let powers_sum := (finset.range 208).sum (λ k, i ^ (k - 103)),
  
  -- Now the main proof statement
  show powers_sum + series_sum = 5355, from sorry
end

end sum_of_powers_and_arithmetic_series_l218_218210


namespace piles_can_be_combined_l218_218438

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l218_218438


namespace jake_weight_loss_l218_218370

-- Define Jake's current weight, the combined weight, and the desired equation
def problem (J S W : ℕ) : Prop :=
  J = 188 ∧ 
  J + S = 278 ∧ 
  J - W = 2 * S

theorem jake_weight_loss : ∃ W : ℕ, problem 188 (278 - 188) W ∧ W = 8 :=
by
  -- Definitions from the problem conditions
  let J := 188
  let S := 278 - J
  
  -- State the core problem conditions and goal
  have h1 : J = 188 := rfl
  have h2 : J + S = 278 := by rw [←h1]; simp
  have h3 : (J: ℕ) - 8 = 2 * S := 
    by
      -- The specific logical exploration and manipulation skipping
      -- with proof that this statement is in the correct structure.
      sorry
  
  -- Main proof expected from
  use 8
  exact And.intro ⟨⟨h1.ham, ⟩ham,⟩⟩ sorry -- Placeholder lean statement illustrating equivalent math proof structures  

end jake_weight_loss_l218_218370


namespace product_of_smallest_primes_l218_218115

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218115


namespace equilateral_pentagon_angles_le_90_l218_218490

theorem equilateral_pentagon_angles_le_90 
  (A B C D E : Point)
  (hAB : A ≠ B) (hBC : B ≠ C) (hCD : C ≠ D) (hDE : D ≠ E) (hEA : E ≠ A)
  (hEquilateral : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E A ∧ dist E A = dist A B)
  (hConvex : convex_hull 𝕜 ({A, B, C, D, E} : set Point) = {A, B, C, D, E}) 
  (hLongestDiagonal : ∀P Q, (P,Q) ∈ [(A,C), (B,D), (C,E), (D,A), (E,B)] → dist P Q ≤ dist A C):
  ∃ P : Point, P ∈ line_segment A C ∧ 
  ∀ Q ∈ {A, B, C, D, E}, ∠ P Q ≤ (90 : ℝ) :=
begin
  sorry
end

end equilateral_pentagon_angles_le_90_l218_218490


namespace cat_starts_moving_away_l218_218937

noncomputable def line1 : ℝ → ℝ := λ x, -4 * x + 34
noncomputable def line2 : ℝ → ℝ := λ x, (1 / 4) * x + 45 / 4

def bird : ℝ × ℝ := (15, 15)

def intersection_point : ℝ × ℝ := (5.35, 13.6)

theorem cat_starts_moving_away:
  ∃ c d, (c, d) = intersection_point ∧ 
               line1 c = d ∧ 
               line2 c = d :=
by
  sorry

end cat_starts_moving_away_l218_218937


namespace smallest_period_max_min_values_l218_218747

def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1/2)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem smallest_period : ∀ x : ℝ, f (x + π) = f x := 
by
  sorry

theorem max_min_values : 
  ∃ c1 c2 ∈ Icc (0:ℝ) (π / 2), f c1 = 1 ∧ f c2 = -1/2 :=
by
  sorry

end smallest_period_max_min_values_l218_218747


namespace AM_divides_BD_in_1_to_2_ratio_l218_218388

structure Parallelogram (A B C D M : Type) :=
  (midpoint_M : mid_point M B C)
  (parallelogram : is_parallelogram A B C D)

theorem AM_divides_BD_in_1_to_2_ratio {A B C D M : Type} 
  [Parallelogram A B C D M] :
  divides_in_ratio A M B D (1 : 2) :=
sorry

end AM_divides_BD_in_1_to_2_ratio_l218_218388


namespace congruency_equivalence_l218_218067

-- Define when two figures are congruent
def congruent (fig1 fig2 : Type) : Prop := ∃ (f : fig1 → fig2) (g : fig2 → fig1), (∀ x, g(f x) = x) ∧ (∀ y, f(g y) = y)

-- Define different properties of figures
def same_area (fig1 fig2 : Type) [HasArea fig1] [HasArea fig2] : Prop := area fig1 = area fig2
def same_shape (fig1 fig2 : Type) [HasShape fig1] [HasShape fig2] : Prop := shape fig1 = shape fig2
def can_completely_overlap (fig1 fig2 : Type) : Prop := ∃ (f : fig1 → fig2), ∀ x, fig2.contains (f x)
def same_perimeter (fig1 fig2 : Type) [HasPerimeter fig1] [HasPerimeter fig2] : Prop := perimeter fig1 = perimeter fig2

-- Define properties for figures that can completely overlap
noncomputable def are_congruent_if_completely_overlap (fig1 fig2 : Type) : Prop :=
  can_completely_overlap fig1 fig2

-- Theorem stating the equivalence of congruency and complete overlap
theorem congruency_equivalence (fig1 fig2 : Type) :
  congruent fig1 fig2 ↔ can_completely_overlap fig1 fig2 := by
  sorry

end congruency_equivalence_l218_218067


namespace ratio_of_x_y_l218_218534

theorem ratio_of_x_y (x y : ℝ) (h : x + y = 3 * (x - y)) : x / y = 2 :=
by
  sorry

end ratio_of_x_y_l218_218534


namespace find_f_f_2_l218_218310

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x + 1 else logBase 2 (x - 1)

theorem find_f_f_2 :
  f (f 2) = 2 :=
by
  sorry

end find_f_f_2_l218_218310


namespace max_pairs_of_acquaintances_l218_218983

theorem max_pairs_of_acquaintances (n : ℕ) (h1 : n = 45)
  (h2 : ∀ (a b : ℕ), (a ≠ b ∧ a + b = n) -> ¬ (a ∧ b)) :
  (∃ pairs : ℕ, pairs = 870 ∧ 
    (∀ (a b : ℕ), a ≠ b ∧ a + b = n → ¬ (a ∧ b))) :=
sorry

end max_pairs_of_acquaintances_l218_218983


namespace remaining_nap_time_is_three_hours_l218_218194

-- Define the flight time and the times spent on various activities
def flight_time_minutes := 11 * 60 + 20
def reading_time_minutes := 2 * 60
def movie_time_minutes := 4 * 60
def dinner_time_minutes := 30
def radio_time_minutes := 40
def game_time_minutes := 60 + 10

-- Calculate the total time spent on activities
def total_activity_time_minutes :=
  reading_time_minutes + movie_time_minutes + dinner_time_minutes + radio_time_minutes + game_time_minutes

-- Calculate the remaining time for a nap
def remaining_nap_time_minutes :=
  flight_time_minutes - total_activity_time_minutes

-- Convert the remaining nap time to hours
def remaining_nap_time_hours :=
  remaining_nap_time_minutes / 60

-- The statement to be proved
theorem remaining_nap_time_is_three_hours :
  remaining_nap_time_hours = 3 := by
  sorry

#check remaining_nap_time_is_three_hours -- This will check if the theorem statement is correct

end remaining_nap_time_is_three_hours_l218_218194


namespace square_roots_of_16_l218_218894

theorem square_roots_of_16 :
  {y : ℤ | y^2 = 16} = {4, -4} :=
by
  sorry

end square_roots_of_16_l218_218894


namespace find_Japanese_students_l218_218197

theorem find_Japanese_students (C K J : ℕ) (hK: K = (6 * C) / 11) (hJ: J = C / 8) (hK_value: K = 48) : J = 11 :=
by
  sorry

end find_Japanese_students_l218_218197


namespace solve_for_x_l218_218018

theorem solve_for_x (x : ℝ) (h : (x - 15) / 3 = (3 * x + 10) / 8) : x = -150 := 
by
  sorry

end solve_for_x_l218_218018


namespace impossible_to_repaint_white_l218_218500

-- Define the board as a 7x7 grid 
def boardSize : ℕ := 7

-- Define the initial coloring function (checkerboard with corners black)
def initialColor (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Define the repainting operation allowed
def repaint (cell1 cell2 : (ℕ × ℕ)) (color1 color2 : Prop) : Prop :=
  ¬color1 = color2 

-- Define the main theorem to prove
theorem impossible_to_repaint_white :
  ¬(∃ f : ℕ × ℕ -> Prop, 
    (∀ i j, (i < boardSize) → (j < boardSize) → (f (i, j) = true)) ∧ 
    (∀ i j, (i < boardSize - 1) → (repaint (i, j) (i, j+1) (f (i, j)) (f (i, j+1))) ∧
             (i < boardSize - 1) → (repaint (i, j) (i+1, j) (f (i, j)) (f (i+1, j)))))
  :=
  sorry

end impossible_to_repaint_white_l218_218500


namespace discount_difference_l218_218951

theorem discount_difference (initial_price : ℝ) (discount_flat : ℝ) (discount_percent : ℝ) : 
  initial_price = 30 ∧ discount_flat = 5 ∧ discount_percent = 0.15 →
  abs ((initial_price - discount_flat) * (1 - discount_percent) - (initial_price * (1 - discount_percent) - discount_flat)) = 75 :=
begin
    sorry
end

end discount_difference_l218_218951


namespace find_angle_B_max_area_l218_218806

noncomputable section

-- Definition of the problem conditions
variables {a b c A B : ℝ}

-- Condition in the problem
axiom condition : b * Real.sin A = a * Real.cos (B - π / 6)

-- Theorem 1: Finding Angle B
theorem find_angle_B (hA : 0 < A ∧ A < π) : B = π / 3 :=
  sorry

-- Theorem 2: Finding Maximum Area
theorem max_area (hB : B = π / 3) (hb : b = 2) : 
  ∃ a c, a * c * Real.sin B / 2 = Real.sqrt 3 :=
  sorry

end find_angle_B_max_area_l218_218806


namespace greeting_cards_distribution_l218_218349

noncomputable def number_of_ways_to_distribute_cards : ℕ :=
  4

theorem greeting_cards_distribution :
  ∃ n : ℕ, n = 9 ∧ ∀ (distrib : Fin 4 → Fin 4), 
  (∀ i, distrib i ≠ i) → n = 9 :=
by
  exact ⟨9, by
    intro distrib h
    have : ∀ i, distrib i ≠ i, from h
    sorry⟩

end greeting_cards_distribution_l218_218349


namespace wire_length_l218_218792

variable (L M l a : ℝ) -- Assume these variables are real numbers.

theorem wire_length (h1 : a ≠ 0) : L = (M / a) * l :=
sorry

end wire_length_l218_218792


namespace locus_of_midpoints_of_chords_through_point_l218_218676

variable {O M : Point} -- Let O be the center of the given circle, and M be a given point.

-- Assume the definition of midpoint
def is_midpoint (A B X : Point) : Prop :=
  dist A X = dist X B

-- Assume the definition of a circle and its chord passing through a point
def chord_passes_through (A B M : Point) : Prop :=
  line_through M (midpoint A B)

-- Given a circle with center O and a point M, the locus theorem
theorem locus_of_midpoints_of_chords_through_point (h_circle : Circle) (h_point : Point)
  (h_center : center_of h_circle = O) (h_given_point : M = h_point) : 
  locus (midpoint_of_chords_through_point h_circle h_point) = Circle (M.dist O / 2) :=
sorry

end locus_of_midpoints_of_chords_through_point_l218_218676


namespace linda_age_difference_l218_218446

/-- 
Linda is some more than 2 times the age of Jane.
In five years, the sum of their ages will be 28.
Linda's age at present is 13.
Prove that Linda's age is 3 years more than 2 times Jane's age.
-/
theorem linda_age_difference {L J : ℕ} (h1 : L = 13)
  (h2 : (L + 5) + (J + 5) = 28) : L - 2 * J = 3 :=
by sorry

end linda_age_difference_l218_218446


namespace paving_stone_length_l218_218596

theorem paving_stone_length (courtyard_length courtyard_width paving_stone_width : ℝ)
  (num_paving_stones : ℕ)
  (courtyard_dims : courtyard_length = 40 ∧ courtyard_width = 20) 
  (paving_stone_dims : paving_stone_width = 2) 
  (num_stones : num_paving_stones = 100) 
  : (courtyard_length * courtyard_width) / (num_paving_stones * paving_stone_width) = 4 :=
by 
  sorry

end paving_stone_length_l218_218596


namespace places_proven_l218_218464

-- Definitions based on the problem conditions
inductive Place
| first
| second
| third
| fourth

def is_boy : String -> Prop
| "Oleg" => True
| "Olya" => False
| "Polya" => False
| "Pasha" => False
| _ => False

def name_starts_with_O : String -> Prop
| n => (n.head! = 'O')

noncomputable def determine_places : Prop :=
  ∃ (olegs_place olyas_place polyas_place pashas_place : Place),
  -- Statements and truth conditions
  ∃ (truthful : String), truthful ∈ ["Oleg", "Olya", "Polya", "Pasha"] ∧ 
  ∀ (person : String), 
    (person ≠ truthful → ∀ (statement : Place -> Prop), ¬ statement (person_to_place person)) ∧
    (person = truthful → person_to_place person = Place.first) ∧
    (person = truthful → 
      match person with
        | "Olya" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → is_boy (place_to_person p)
        | "Oleg" => ∃ (p : Place), (person_to_place "Oleg" = p ∧ person_to_place "Olya" = succ_place p ∨ 
                                    person_to_place "Olya" = p ∧ person_to_place "Oleg" = succ_place p)
        | "Pasha" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → name_starts_with_O (place_to_person p)
        | _ => True
      end)

-- Helper functions to relate places to persons
def person_to_place : String -> Place
| "Oleg" => Place.first
| "Olya" => Place.second
| "Polya" => Place.third
| "Pasha" => Place.fourth
| _ => Place.first -- Default, shouldn't happen

def place_to_person : Place -> String
| Place.first => "Oleg"
| Place.second => "Olya"
| Place.third => "Polya"
| Place.fourth => "Pasha"

def succ_place : Place → Place
| Place.first => Place.second
| Place.second => Place.third
| Place.third => Place.fourth
| Place.fourth => Place.first -- No logical next in this context.

theorem places_proven : determine_places :=
by
  sorry

end places_proven_l218_218464


namespace smallest_of_six_consecutive_even_numbers_l218_218037

theorem smallest_of_six_consecutive_even_numbers (h : ∃ n : ℤ, (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 390) : ∃ m : ℤ, m = 60 :=
by
  have ex : ∃ n : ℤ, 6 * n + 6 = 390 := by sorry
  obtain ⟨n, hn⟩ := ex
  use (n - 4)
  sorry

end smallest_of_six_consecutive_even_numbers_l218_218037


namespace max_value_of_a_l218_218339

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x

noncomputable def f_prime (x a : ℝ) : ℝ := 3 * x^2 - a

def is_monotonically_increasing (f_prime : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ≥ 0, f_prime x a ≥ 0

theorem max_value_of_a {a : ℝ} :
  is_monotonically_increasing (λ x a, f_prime x a) a → a ≤ 0 :=
sorry

end max_value_of_a_l218_218339


namespace triangle_area_l218_218350

-- Definitions based on the conditions in a)
def vertexA : ℝ × ℝ × ℝ := (3, 0, 2)
def vertexB : ℝ × ℝ × ℝ := (6, 3, 1)
def vertexC : ℝ × ℝ × ℝ := (6, -3, 4)

-- Statement of the proof problem
theorem triangle_area :
  let AB := (vertexB.1 - vertexA.1, vertexB.2 - vertexA.2, vertexB.3 - vertexA.3)
  let AC := (vertexC.1 - vertexA.1, vertexC.2 - vertexA.2, vertexC.3 - vertexA.3)
  let crossProduct := (AB.2 * AC.3 - AB.3 * AC.2, AB.3 * AC.1 - AB.1 * AC.3, AB.1 * AC.2 - AB.2 * AC.1)
  let magnitude := real.sqrt(crossProduct.1^2 + crossProduct.2^2 + crossProduct.3^2)
  (1/2 * magnitude = 3/2 * real.sqrt(46)) :=
by
  sorry

end triangle_area_l218_218350


namespace number_of_digits_in_factorial_2015_l218_218644

theorem number_of_digits_in_factorial_2015 :
  let A := 2015 in
  let factorial (n : Nat) := Nat.factorial n in
  let number_of_digits (n : Nat) := Nat.floor (Real.log10 (factorial n)) + 1 in
  number_of_digits A = 5787 := by
  sorry

end number_of_digits_in_factorial_2015_l218_218644


namespace probability_of_three_draws_l218_218155

open_locale big_operators

def sum_chips (chips : list ℕ) : ℕ := chips.sum

def valid_draws (draws : list ℕ) : Prop :=
  (sum_chips (draws.take 2) ≤ 10) ∧ (sum_chips (draws.take 3) > 10) 

def count_valid_draws : ℕ := 13 -- number of valid pairs (by manual count from the pairs listed)

noncomputable def probability_three_draws : ℚ :=
  count_valid_draws * (1/42)

theorem probability_of_three_draws:
  probability_three_draws = 13 / 42 :=
by sorry

end probability_of_three_draws_l218_218155


namespace piles_can_be_combined_l218_218436

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l218_218436


namespace product_of_primes_l218_218079

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l218_218079


namespace common_chords_intersect_at_orthocenter_l218_218776

variable {Point : Type*} [EuclideanGeometry Point]
variable (A B C A₁ B₁ C₁ M : Point)

-- Definitions for the altitudes, orthocenter, and circles with given diameters
def altitude_A_meets_BC_at_A1 : Prop := ∃ (line : Line Point), is_perpendicular line (line_through B C) ∧ passes_through line A₁ ∧ passes_through_altitude line A
def altitude_B_meets_CA_at_B1 : Prop := ∃ (line : Line Point), is_perpendicular line (line_through C A) ∧ passes_through line B₁ ∧ passes_through_altitude line B
def altitude_C_meets_AB_at_C1 : Prop := ∃ (line : Line Point), is_perpendicular line (line_through A B) ∧ passes_through line C₁ ∧ passes_through_altitude line C
def is_orthocenter : Prop := passes_through_altitude (line_through A A₁) A ∧ passes_through_altitude (line_through B B₁) B ∧ passes_through_altitude (line_through C C₁) C

def circle_with_diameter (P Q : Point) : Circle Point := sorry

def common_chord_intersects_orthocenter (circle1 circle2 : Circle Point) : Prop := sorry

theorem common_chords_intersect_at_orthocenter
  (h_orthocenter : is_orthocenter M)
  (h_altitude_A : altitude_A_meets_BC_at_A1 A B C A₁)
  (h_altitude_B : altitude_B_meets_CA_at_B1 B C A B₁)
  (h_altitude_C : altitude_C_meets_AB_at_C1 C A B C₁)
  (circle_A := circle_with_diameter A A₁)
  (circle_B := circle_with_diameter B B₁)
  (circle_C := circle_with_diameter C C₁)
  : common_chord_intersects_orthocenter circle_A circle_B M ∧
    common_chord_intersects_orthocenter circle_B circle_C M ∧
    common_chord_intersects_orthocenter circle_C circle_A M :=
sorry

end common_chords_intersect_at_orthocenter_l218_218776


namespace two_discounts_l218_218522

theorem two_discounts (p : ℝ) : (0.9 * 0.9 * p) = 0.81 * p :=
by
  sorry

end two_discounts_l218_218522


namespace problem1_solution_problem2_solution_l218_218833

-- Definition for Problem 1
noncomputable def problem1_statement (z1 z2 : ℂ) : Prop :=
  (z1.re > 0) ∧ (z2.re > 0) ∧ ((z1^2).re = 2) ∧ ((z2^2).re = 2) →
  (z1 * z2).re = 2

-- Definition for Problem 2
noncomputable def problem2_statement (z1 z2 : ℂ) : Prop :=
  (z1.re > 0) ∧ (z2.re > 0) ∧ ((z1^2).re = 2) ∧ ((z2^2).re = 2) →
  |z1 + 2| + |conj z2 + 2| - |conj z1 - z2| = 4 * real.sqrt 2

-- Theorems
theorem problem1_solution : ∀ (z1 z2 : ℂ), problem1_statement z1 z2 :=
by
  intros,
  sorry

theorem problem2_solution : ∀ (z1 z2 : ℂ), problem2_statement z1 z2 :=
by
  intros,
  sorry

end problem1_solution_problem2_solution_l218_218833


namespace triangle_inequality_range_l218_218341

theorem triangle_inequality_range {x : ℝ} : 
  (3 + (2 * x + 1) > 10) ∧ 
  (3 + 10 > 2 * x + 1) ∧ 
  (10 + (2 * x + 1) > 3) → 
  3 < x ∧ x < 6 := 
begin
  sorry
end

end triangle_inequality_range_l218_218341


namespace find_value_sin_cos_l218_218299

theorem find_value_sin_cos (x y r : ℝ) (hP : r = real.sqrt (x^2 + y^2))
  (hx : x = -4) (hy : y = 3) (hr : r = 5) :
  2 * (y / r) + (x / r) = 2 / 5 :=
by
  rw [hx, hy, hr]
  rw [← real.sqrt_sq_eq_abs]
  rw [real.div_eq_inv_mul]
  sorry

end find_value_sin_cos_l218_218299


namespace sum_of_all_four_digit_numbers_formed_l218_218265

open List

noncomputable def sum_of_four_digit_numbers (digits : List ℕ) : ℕ :=
  let perms := digits.permutations.filter (λ l, l.length = 4)
  let nums := perms.map (λ l, 1000 * l.head + 100 * l.nthLe 1 sorry + 10 * l.nthLe 2 sorry + l.nthLe 3 sorry)
  nums.sum

theorem sum_of_all_four_digit_numbers_formed : sum_of_four_digit_numbers [1, 2, 3, 4, 5] = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_formed_l218_218265


namespace percentage_increase_breadth_is_25_l218_218979

noncomputable theory

def breadth_first_carpet : ℝ := 6
def length_first_carpet : ℝ := 1.44 * breadth_first_carpet
def rate_per_sq_m : ℝ := 45
def cost_second_carpet : ℝ := 4082.4
def length_second_carpet : ℝ := length_first_carpet + 0.4 * length_first_carpet

def area_second_carpet : ℝ := cost_second_carpet / rate_per_sq_m 
def breadth_second_carpet : ℝ := area_second_carpet / length_second_carpet 

def percentage_increase_breadth := ((breadth_second_carpet - breadth_first_carpet) / breadth_first_carpet) * 100

theorem percentage_increase_breadth_is_25 : percentage_increase_breadth = 25 := 
by 
  sorry

end percentage_increase_breadth_is_25_l218_218979


namespace tangent_perpendicular_iff_l218_218014

variable (A B O F : Point)

theorem tangent_perpendicular_iff :
  (is_tangent OA parabola) ∧ (is_tangent OB parabola) ∧ (angle O A B = 90) ↔
  (segment_AB_passes_focus_parabola A B F) ∨ (point_on_directrix_parabola O) :=
sorry

end tangent_perpendicular_iff_l218_218014


namespace correct_calculation_l218_218135

theorem correct_calculation : real.sqrt 8 - real.sqrt 2 = real.sqrt 2 :=
by
  sorry

end correct_calculation_l218_218135


namespace exists_integer_point_in_convex_pentagon_l218_218281

-- Defining what it means to be a convex pentagon with integer coordinates
structure ConvexPentagon : Type :=
  (A B C D E : ℤ × ℤ)  -- The vertices have integer coordinates
  (convex : ∀ p q r : ℤ × ℤ, p ≠ q → q ≠ r → p ≠ r → 
    ∃ s : ℝ × ℝ, s ∈ convex_hull ℚ {A, B, C, D, E} ∧ integer_point_in_pentagon s)

-- Statement of the theorem: For any convex pentagon with integer coordinates, 
-- there exists at least one internal integer coordinate point.
theorem exists_integer_point_in_convex_pentagon (P : ConvexPentagon) : 
  ∃ I : ℤ × ℤ, I ∈ interior (convex_hull ℚ {P.A, P.B, P.C, P.D, P.E}) :=
sorry

end exists_integer_point_in_convex_pentagon_l218_218281


namespace condition_sufficient_but_not_necessary_l218_218151

variable {a b c : ℝ}

theorem condition_sufficient_but_not_necessary (h : c^2 > 0) : 
  (ac^2 > bc^2) ↔ (a > b) :=
begin
  /- Sufficient condition -/
  split,
  {
    intro h1,
    apply (mul_lt_mul_right h).mp,
    exact h1,
  },
  {
    intro h2,
    apply (mul_lt_mul_right h).mpr,
    exact h2,
  }
end

end condition_sufficient_but_not_necessary_l218_218151


namespace maria_rearrangements_time_l218_218448

theorem maria_rearrangements_time :
  let num_letters := 5
  let repeats := 2
  let arrangements := (num_letters.factorial / repeats.factorial : ℝ)
  let write_speed := 8
  let time_in_minutes := arrangements / write_speed
  time_in_minutes / 60 = 0.125 :=
by
  let num_letters := 5
  let repeats := 2
  let arrangements := (num_letters.factorial / repeats.factorial : ℝ)
  let write_speed := 8
  let time_in_minutes := arrangements / write_speed
  show time_in_minutes / 60 = 0.125 from sorry

end maria_rearrangements_time_l218_218448


namespace T_structure_l218_218390

structure point (α : Type) :=
(x : α)
(y : α)

def T (α : Type) [linear_ordered_field α] : set (point α) :=
  {p | (5 = p.x + 3 ∧ p.y ≤ 11) ∨  (5 = p.y - 6 ∧ p.x ≤ 2) ∨  (p.y = p.x + 9 ∧ p.x ≤ 2 ∧ p.y ≤ 11)}

theorem T_structure (α : Type) [linear_ordered_field α] :
  T α = {p | (∃ y, p = ⟨2, y⟩ ∧ y ≤ 11) ∨ (∃ x, p = ⟨x, 11⟩ ∧ x ≤ 2) ∨ (∃ x, p = ⟨x, x + 9⟩ ∧ x ≤ 2 ∧ x + 9 ≤ 11)} :=
sorry

end T_structure_l218_218390


namespace indefinite_integral_example_l218_218147

theorem indefinite_integral_example : 
  ∃ C : ℝ, (∫ x in set.univ, (3 * x^3 + 25) / (x^2 + 3 * x + 2) = (3 / 2) * x^2 - 9 * x + 22 * Real.log (abs (x + 1)) - Real.log (abs (x + 2)) + C) :=
sorry

end indefinite_integral_example_l218_218147


namespace product_of_primes_l218_218099

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l218_218099


namespace stability_of_origin_l218_218324

noncomputable def LyapunovFunction (a b : ℝ) (x y : ℝ) : ℝ :=
  a * x^2 + b * y^2

theorem stability_of_origin (a b : ℝ) (hx : ∀ t: ℝ, (dx dt)(t) = - (x t) + 4 * (y t) - 4 * (x t) * (y t)^3) 
  (hy : ∀ t: ℝ, (dy dt)(t) = - 2 * (y t) - (x t)^2 * (y t)^2) (h_a_pos : a > 0) (h_b : b = 4 * a) :
  ∀ t: ℝ, (x t = 0 ∧ y t = 0) → 
  ∃ v, v = LyapunovFunction a b 0 0 ∧
        v ≥ 0 ∧ 
        (∂ (LyapunovFunction a b (x t) (y t)) / ∂t ≤ 0 → ∂ (LyapunovFunction a b (x t) (y t)) / ∂t < 0 ∧ 
        v ≤ -|⟨0, 0⟩| < 0) :=
sorry

end stability_of_origin_l218_218324


namespace equal_circle_radius_l218_218903

theorem equal_circle_radius (r R : ℝ) (h1: r > 0) (h2: R > 0)
  : ∃ x : ℝ, x = r * R / (R + r) :=
by 
  sorry

end equal_circle_radius_l218_218903


namespace sin_pi_over_2_plus_alpha_l218_218701

noncomputable def r : ℝ := 5
def x : ℝ := -4
def y : ℝ := 3
def α : ℝ := real.arctan2 y x

theorem sin_pi_over_2_plus_alpha (h : x^2 + y^2 = r^2) : 
  real.sin (real.pi / 2 + α) = -4 / 5 :=
by 
  sorry

end sin_pi_over_2_plus_alpha_l218_218701


namespace arithmetic_sequence_properties_l218_218515

theorem arithmetic_sequence_properties
    (n s1 s2 s3 : ℝ)
    (h1 : s1 = 8)
    (h2 : s2 = 50)
    (h3 : s3 = 134)
    (h4 : n = 8) :
    n^2 * s3 - 3 * n * s1 * s2 + 2 * s1^2 = 0 := 
by {
  sorry
}

end arithmetic_sequence_properties_l218_218515


namespace verify_placements_l218_218451

-- Definitions for participants and their possible places
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

-- Each participant should be mapped to a place (1, 2, 3, 4)
def Place : Participant → ℕ := λ p,
  match p with
  | Participant.Olya => 2
  | Participant.Oleg => 1
  | Participant.Polya => 3
  | Participant.Pasha => 4

-- Conditions based on the problem statement
def statement_Olya : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Polya % 2 = 1 ∧ Place Participant.Pasha % 2 = 1)

def statement_Oleg : Prop :=
  (abs (Place Participant.Oleg - Place Participant.Olya) = 1)

def statement_Pasha : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Olya % 2 = 1 ∧ Place Participant.Polya % 2 = 1)

-- Only one child tells the truth and the others lie
def exactly_one_true (a b c : Prop) : Prop := (a ∨ b ∨ c) ∧ (a → ¬b ∧ ¬c) ∧ (b → ¬a ∧ ¬c) ∧ (c → ¬a ∧ ¬b)

-- The main theorem to be proven
theorem verify_placements :
  exactly_one_true (statement_Olya) (statement_Oleg) (statement_Pasha) ∧ 
  Place Participant.Olya = 2 ∧
  Place Participant.Oleg = 1 ∧
  Place Participant.Polya = 3 ∧
  Place Participant.Pasha = 4 :=
by
  sorry

end verify_placements_l218_218451


namespace product_of_primes_l218_218112

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l218_218112


namespace sum_of_all_four_digit_numbers_formed_l218_218266

open List

noncomputable def sum_of_four_digit_numbers (digits : List ℕ) : ℕ :=
  let perms := digits.permutations.filter (λ l, l.length = 4)
  let nums := perms.map (λ l, 1000 * l.head + 100 * l.nthLe 1 sorry + 10 * l.nthLe 2 sorry + l.nthLe 3 sorry)
  nums.sum

theorem sum_of_all_four_digit_numbers_formed : sum_of_four_digit_numbers [1, 2, 3, 4, 5] = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_formed_l218_218266


namespace triangle_bisector_l218_218531

theorem triangle_bisector (PQ PR QR QS RS : ℝ) 
  (h1 : PQ / PR = 3 / 4)
  (h2 : QR = 15)
  (h3 : QR = QS + RS)
  (h4 : QS / RS = PQ / PR) : QS = 45 / 7 :=
by
  have ratio : QS / RS = 3 / 4 := h4.trans h1,
  have sum_length : QS + RS = QR := h3,
  have total_length : QS + RS = 15 := sum_length.trans h2,
  have : (3 / 4) * RS + RS = 15,
  have : (3 / 4 * RS + RS) = 15,
  have : (3 / 4 * RS + 4 / 4 * RS) = 15,
  have : (7 / 4) * RS = 15,
  have : RS = 4 * 15 / 7,
  have : RS = 60 / 7,
  have : QS = 3 * 15 / 7,
  exact sorry

end triangle_bisector_l218_218531


namespace hockey_league_teams_l218_218058
open Real

theorem hockey_league_teams (n : ℕ) 
  (h1 : ∀ m, m ≠ n → m ∈ (Finset.range n) → (n - 1) * 10 games_count : ℕ := 1530) 
  (h2 : (n * 10 * (n - 1)) / 2 = games_count) : 
  n = 18 := sorry

end hockey_league_teams_l218_218058


namespace similar_sizes_combination_possible_l218_218404

theorem similar_sizes_combination_possible 
    (similar : Nat → Nat → Prop := λ x y, x ≤ y ∧ y ≤ 2 * x)
    (combine_piles : List Nat → Nat ∃ combined : Nat, (∀ x y ∈ combined, similar x y) → True
    (piles : List Nat) : True :=
sorry

end similar_sizes_combination_possible_l218_218404


namespace find_b_value_l218_218063

def perfect_square_trinomial (a b c : ℕ) : Prop :=
  ∃ d, a = d^2 ∧ c = d^2 ∧ b = 2 * d * d

theorem find_b_value (b : ℝ) :
    (∀ x : ℝ, 16 * x^2 - b * x + 9 = (4 * x - 3) * (4 * x - 3) ∨ 16 * x^2 - b * x + 9 = (4 * x + 3) * (4 * x + 3)) -> 
    b = 24 ∨ b = -24 := 
by
  sorry

end find_b_value_l218_218063


namespace complex_number_calc_l218_218872

theorem complex_number_calc (z : ℂ) (h : z = (2 - complex.i) / (1 - complex.i)) : z = 3 / 2 + (1 / 2) * complex.i :=
by
  sorry

end complex_number_calc_l218_218872


namespace find_m_if_f_even_l218_218773

theorem find_m_if_f_even (m : ℝ) (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x = x^4 + (m - 1) * x + 1) ∧ (∀ x : ℝ, f x = f (-x)) → m = 1 := 
by 
  sorry

end find_m_if_f_even_l218_218773


namespace line_contains_diameter_of_circle_l218_218504

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 8 = 0

noncomputable def equation_of_line (x y : ℝ) : Prop :=
  2*x - y - 1 = 0

theorem line_contains_diameter_of_circle :
  (∃ x y : ℝ, equation_of_circle x y ∧ equation_of_line x y) :=
sorry

end line_contains_diameter_of_circle_l218_218504


namespace product_of_primes_l218_218083

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l218_218083


namespace pat_more_hours_than_jane_l218_218481

theorem pat_more_hours_than_jane (H P K M J : ℝ) 
  (h_total : H = P + K + M + J)
  (h_pat : P = 2 * K)
  (h_mark : M = (1/3) * P)
  (h_jane : J = (1/2) * M)
  (H290 : H = 290) :
  P - J = 120.83 := 
by
  sorry

end pat_more_hours_than_jane_l218_218481


namespace product_of_primes_l218_218084

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l218_218084


namespace part1_part2_l218_218289

-- Defining the sets A and B
def A : set ℝ := { x | x^2 - 3*x + 2 = 0 }
def B (a : ℝ) : set ℝ := { x | x^2 + 2*(a + 1)*x + (a^2 - 5) = 0 }

-- Condition for part (1)
theorem part1 (a : ℝ) : A ∪ B a = A → a ∈ Iic (-3) := 
by sorry

-- Defining the universal set U
def U : set ℝ := set.univ

-- Complement of B in U
def comp_U_B (a : ℝ) : set ℝ := { x | ¬ (B a x) }

-- Condition for part (2)
theorem part2 (a : ℝ) : A ∩ comp_U_B a = A → 
  a ≠ -3 ∧ a ≠ -1 ∧ a ≠ -1 + real.sqrt 3 ∧ a ≠ -1 - real.sqrt 3 := 
by sorry

end part1_part2_l218_218289


namespace num_ways_to_sum_22_with_primes_l218_218353

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem num_ways_to_sum_22_with_primes :
  { (a, b, c) : ℕ × ℕ × ℕ // 1 < a ∧ a < b ∧ b < c ∧ is_prime a ∧ is_prime b ∧ is_prime c ∧ a + b + c = 22 }
  .card = 2 := sorry

end num_ways_to_sum_22_with_primes_l218_218353


namespace relay_race_time_l218_218585

theorem relay_race_time (M S J T : ℕ) 
(hJ : J = 30)
(hS : S = J + 10)
(hM : M = 2 * S)
(hT : T = M - 7) : 
M + S + J + T = 223 :=
by sorry

end relay_race_time_l218_218585


namespace boys_in_class_is_12_l218_218348

noncomputable def number_of_boys_in_class (n : ℕ) : ℕ :=
  (if n % 2 = 0 then n / 2 else (n - 1) / 2)

theorem boys_in_class_is_12 :
  ∃ (n : ℕ), 
  (52 * n / 100 = 13) ∧
  (boys_and_girls_alternate_in_row n) ∧
  number_of_boys_in_class n = 12 :=
by
  sorry

-- Helper Definitions
def boys_and_girls_alternate_in_row (n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → (i.even → ∃ j : ℕ, j < n ∧ ¬j.even) ∧ (¬i.even → ∃ k : ℕ, k < n ∧ k.even)

end boys_in_class_is_12_l218_218348


namespace find_prices_and_max_pens_l218_218905

-- Define the variables
variable (x y m : ℕ)

-- Given conditions for the price of each pen (x) and each mechanical pencil (y)
def price_conditions := (2 * x + 5 * y = 75) ∧ (3 * x + 2 * y = 85)

-- Given additional conditions for the maximum number of pens (m)
def max_pens_condition := ∀ m : ℕ, (25 * m + 5 * (m + 8) < 670) → m ≤ 20

-- Proof statement
theorem find_prices_and_max_pens (x y m : ℕ) :
  (price_conditions x y) → (x = 25 ∧ y = 5) →
  (max_pens_condition m) :=
by
  intros h1 h2
  sorry

end find_prices_and_max_pens_l218_218905


namespace volume_of_tetrahedron_l218_218958

theorem volume_of_tetrahedron (a : ℝ) (h : a > 0) :
  let A := (0, 0, 0)
  let B := (a, 0, 0)
  let C := (a / 2, (Real.sqrt 3) / 2 * a, 0)
  let D := (a / 2, (Real.sqrt 3) / 6 * a, (Real.sqrt 6) / 3 * a)
  let A1 := midpoint C D
  let B1 := midpoint D A
  let C1 := midpoint A B
  let D1 := midpoint B C
  ∀ (A1 B1 C1 D1 : ℝ⁵), volume_of_tetrahedron A1 B1 C1 D1 = (Real.sqrt 2 * a^3) / 162
  := by
  sorry

end volume_of_tetrahedron_l218_218958


namespace similar_sizes_combination_possible_l218_218402

theorem similar_sizes_combination_possible 
    (similar : Nat → Nat → Prop := λ x y, x ≤ y ∧ y ≤ 2 * x)
    (combine_piles : List Nat → Nat ∃ combined : Nat, (∀ x y ∈ combined, similar x y) → True
    (piles : List Nat) : True :=
sorry

end similar_sizes_combination_possible_l218_218402


namespace solve_eqn_l218_218491

theorem solve_eqn (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ 6) :
  (x + 11) / (x - 4) = (x - 1) / (x + 6) → x = -31 / 11 :=
by
sorry

end solve_eqn_l218_218491


namespace inverse_of_exponential_is_log_l218_218334

noncomputable def f (x : ℝ) : ℝ := 3 ^ x

theorem inverse_of_exponential_is_log (x : ℝ) (hx : 0 < x) : f^{-1} x = Real.log x / Real.log 3 :=
by
  sorry

end inverse_of_exponential_is_log_l218_218334


namespace lilibeth_and_friends_strawberries_l218_218445

-- Define the conditions
def baskets_filled_by_lilibeth : ℕ := 6
def strawberries_per_basket : ℕ := 50
def friends_count : ℕ := 3

-- Define the total number of strawberries picked by Lilibeth and her friends 
def total_strawberries_picked : ℕ :=
  (baskets_filled_by_lilibeth * strawberries_per_basket) * (1 + friends_count)

-- The theorem to prove
theorem lilibeth_and_friends_strawberries : total_strawberries_picked = 1200 := 
by
  sorry

end lilibeth_and_friends_strawberries_l218_218445


namespace shaded_area_eq_l218_218980
open Real

theorem shaded_area_eq :
  let R := 2 * sqrt 3 + 1,
      small_circle_area := 13 * π,
      large_circle_area := π * (R * R),
      shaded_area := large_circle_area - small_circle_area
  in shaded_area = 4 * sqrt 3 * π :=
by
  let R := 2 * sqrt 3 + 1
  let small_circle_area := 13 * π
  let large_circle_area := π * (R * R)
  let shaded_area := large_circle_area - small_circle_area
  sorry

end shaded_area_eq_l218_218980


namespace length_df_range_l218_218803

open Real

variables {A B C A1 B1 C1 G E D F : Point}
variables (AB AC AA1 : Real) (angleBAC : Angle)
variables (G_is_midpoint : midpoint A1 B1 G) (E_is_midpoint : midpoint C C1 E)
variables (D_on_AC : D ∈ segment A C) (F_on_AB : F ∈ segment A B)
variables (GD_perp_EF : perpendicular (vector G D) (vector E F))

theorem length_df_range :
  ∠BAC = π / 2 ∧ AB = 1 ∧ AC = 1 ∧ AA1 = 1 ∧ midpoint A1 B1 G ∧ midpoint C C1 E ∧
  D ∈ segment A C ∧ F ∈ segment A B ∧ perpendicular (vector G D) (vector E F) →
  set.range (λ t : Real, length (D F)) = Icc (1 / √5) 1 :=
sorry

end length_df_range_l218_218803


namespace area_KLMQ_l218_218796

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def JR := 2
def RQ := 3
def JL := 8

def JLMR : Rectangle := {length := JL, width := JR}
def JKQR : Rectangle := {length := RQ, width := JR}

def RM : ℝ := JL
def QM : ℝ := RM - RQ
def LM : ℝ := JR

def KLMQ : Rectangle := {length := QM, width := LM}

theorem area_KLMQ : KLMQ.length * KLMQ.width = 10 :=
by
  sorry

end area_KLMQ_l218_218796


namespace exists_root_in_interval_l218_218201

open Real

theorem exists_root_in_interval : ∃ x, 1.1 < x ∧ x < 1.2 ∧ (x^2 + 12*x - 15 = 0) :=
by {
  let f := λ x : ℝ, x^2 + 12*x - 15,
  have h1 : f 1.1 = -0.59 :=  sorry,
  have h2 : f 1.2 = 0.84 := sorry,
  have sign_change : (f 1.1) * (f 1.2) < 0,
  { rw [h1, h2], linarith, },
  exact exists_has_deriv_at_eq_zero (by norm_num1) (by norm_num1) (by linarith)
}

end exists_root_in_interval_l218_218201


namespace piles_to_single_pile_l218_218434

-- Define the condition similar_sizes
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

-- Define the inductive step of combining stones
def combine_stones (piles : List ℕ) : List ℕ :=
  if ∃ x y, x ∈ piles ∧ y ∈ piles ∧ similar_sizes x y then
    let ⟨x, hx, y, hy, hsim⟩ := Classical.some_spec (Classical.some_spec_exists _)
    List.cons (x + y) (List.erase (List.erase piles x) y)
  else
    piles

-- Prove that a collection of piles can be reduced to a single pile of size n
theorem piles_to_single_pile (piles : List ℕ) (h : ∀ x ∈ piles, x = 1) : 
  ∃ p, list.length (Iterator.iterate combine_stones piles.count) 1 = 1 := by
  sorry

end piles_to_single_pile_l218_218434


namespace quadratic_discriminant_l218_218687

theorem quadratic_discriminant (k : ℝ) :
  (∃ x : ℝ, k*x^2 + 2*x - 1 = 0) ∧ (∀ a b, (a*x + b) ^ 2 = a^2 * x^2 + 2 * a * b * x + b^2) ∧
  (a = k) ∧ (b = 2) ∧ (c = -1) ∧ ((b^2 - 4 * a * c = 0) → (4 + 4 * k = 0)) → k = -1 :=
sorry

end quadratic_discriminant_l218_218687


namespace cube_pyramid_volume_l218_218161

theorem cube_pyramid_volume (h : ℝ) :
  let V_cube := 4^3,
      V_pyramid := (1 / 3) * 8^2 * h in
  V_cube = V_pyramid → h = 3 :=
by
  sorry

end cube_pyramid_volume_l218_218161


namespace trapezoid_ratio_l218_218179

theorem trapezoid_ratio (A B C D M N : Point) (h1 : Trapezoid A B C D)
                       (h2 : OnLine M C D) (h3 : Ratio CM MD = 4 / 3)
                       (h4 : OnLine N A C) (h5 : Segment BM intersects Diagonal AC at N)
                       (h6 : Ratio CN NA = 4 / 3) :
  Ratio AD BC = 7 / 12 := 
sorry

end trapezoid_ratio_l218_218179


namespace combination_8_3_l218_218275

theorem combination_8_3 : nat.choose 8 3 = 56 :=
by sorry

end combination_8_3_l218_218275


namespace total_amount_pqr_l218_218573

theorem total_amount_pqr (p q r : ℕ) (T : ℕ) 
  (hr : r = 2 / 3 * (T - r))
  (hr_value : r = 1600) : 
  T = 4000 :=
by
  sorry

end total_amount_pqr_l218_218573


namespace product_of_smallest_primes_l218_218093

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218093


namespace employed_females_percentage_l218_218145

theorem employed_females_percentage (total_pop employed employed_males : ℝ)
    (h1 : total_pop > 0)
    (h2 : employed = 0.60 * total_pop)
    (h3 : employed_males = 0.48 * total_pop) :
    employed > 0 → (employed - employed_males) / employed * 100 = 20 :=
by
  intro he
  have employed_females : ℝ := employed - employed_males
  have employed_females_percent : ℝ := (employed_females / employed) * 100
  have calc_employed_females : employed_females = 0.12 * total_pop := by sorry
  have calc_employed_females_percent : employed_females_percent = 20 := by sorry
  exact calc_employed_females_percent

end employed_females_percentage_l218_218145


namespace unique_sequence_and_a_2002_l218_218073

-- Define the sequence (a_n)
noncomputable def a : ℕ → ℕ := -- define the correct sequence based on conditions
  -- we would define a such as in the constructive steps in the solution, but here's a placeholder
  sorry

-- Prove the uniqueness and finding a_2002
theorem unique_sequence_and_a_2002 :
  (∀ n : ℕ, ∃! (i j k : ℕ), n = a i + 2 * a j + 4 * a k) ∧ a 2002 = 1227132168 :=
by
  sorry

end unique_sequence_and_a_2002_l218_218073


namespace store_owner_must_sell_1000_pens_l218_218964

/-- Definition of the conditions and required conclusion --/
def problem_statement : Prop :=
  let total_pens := 2000
  let cost_per_pen := (0.15 : ℝ)
  let selling_price_per_pen := (0.30 : ℝ)
  let desired_profit := 150
  let profit_per_pen := selling_price_per_pen - cost_per_pen in
  ∃ n : ℕ, n * profit_per_pen = desired_profit ∧ n = 1000

theorem store_owner_must_sell_1000_pens : problem_statement :=
  by
    let total_pens := 2000
    let cost_per_pen := (0.15 : ℝ)
    let selling_price_per_pen := (0.30 : ℝ)
    let desired_profit := 150
    let profit_per_pen := selling_price_per_pen - cost_per_pen
    let n := 1000
    have h1 : profit_per_pen = 0.15 := by sorry
    have h2 : n * profit_per_pen = desired_profit := by sorry
    existsi n
    split
    exact h2
    refl

end store_owner_must_sell_1000_pens_l218_218964


namespace line_equation_passing_through_point_and_opposite_intercepts_l218_218877

theorem line_equation_passing_through_point_and_opposite_intercepts 
  : ∃ (a b : ℝ), (y = a * x) ∨ (x - y = b) :=
by
  use (3/2), (-1)
  sorry

end line_equation_passing_through_point_and_opposite_intercepts_l218_218877


namespace product_of_primes_l218_218087

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l218_218087


namespace david_swim_time_l218_218996

theorem david_swim_time :
  let freestyle := 48
  let backstroke := freestyle + 4
  let butterfly := backstroke + 3
  let breaststroke := butterfly + 2
  freestyle + backstroke + butterfly + breaststroke = 212 :=
by
  simp only [freestyle, backstroke, butterfly, breaststroke]
  rfl

end david_swim_time_l218_218996


namespace pile_division_possible_l218_218417

theorem pile_division_possible (n : ℕ) :
  ∃ (division : list ℕ), (∀ x ∈ division, x = 1) ∧ division.sum = n :=
by
  sorry

end pile_division_possible_l218_218417


namespace sum_of_all_four_digit_numbers_l218_218237

-- Let us define the set of digits
def digits : set ℕ := {1, 2, 3, 4, 5}

-- We will define a function that generates the four-digit numbers
def four_digit_numbers := {n : ℕ // ∃ a b c d : ℕ, 
                                      a ∈ digits ∧ 
                                      b ∈ digits ∧ 
                                      c ∈ digits ∧ 
                                      d ∈ digits ∧ 
                                      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ 
                                      b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                                      n = 1000 * a + 100 * b + 10 * c + d}

-- Define a function to calculate the sum of all elements in a set of numbers
def sum_set (s : set ℕ) : ℕ := s.fold (λa b, a + b) 0

theorem sum_of_all_four_digit_numbers :
  sum_set {n | ∃ x : four_digit_numbers, x.val = n} = 399960 :=
sorry

end sum_of_all_four_digit_numbers_l218_218237


namespace division_formula_l218_218564

theorem division_formula (d q r : ℕ) (h_d : d = 179) (h_q : q = 89) (h_r : r = 37) :
  (d * q + r = 15968) :=
by
  rw [h_d, h_q, h_r]
  simp
  sorry

end division_formula_l218_218564


namespace polynomial_proof_l218_218693

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

theorem polynomial_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : f a a b c = 0) (h4 : f b a b c = 0) (h5 : f c a b c = 0) : 
  f 0 a b c * f 1 a b c < 0 ∧ f 0 a b c * f 3 a b c > 0 :=
by 
  sorry

end polynomial_proof_l218_218693


namespace min_perimeter_of_triangle_l218_218066

theorem min_perimeter_of_triangle (A B C: Type) 
  (AB AC BC: ℤ) (I: Type)
  (angle_B angle_C: ℤ)
  (h1 : AB = AC)
  (h2 : AB > 0 ∧ AC > 0 ∧ BC > 0)
  (h3 : |BI| = 10)
  (h4 : ∃ α ∈ ({angle_B, angle_C}), ∃ n: ℤ, α = n) :
  ∃ P, P = 40 :=
sorry

end min_perimeter_of_triangle_l218_218066


namespace mpg_calculation_l218_218633

noncomputable def avg_mpg (initial_miles final_miles: ℕ)(init_gas: ℕ)(gas1 gas2 gas3: ℕ): ℝ :=
  let distance := final_miles - initial_miles in
  let total_gas := init_gas + gas1 + gas2 + gas3 in
  distance / total_gas

theorem mpg_calculation:
  (initial_miles = 58000) →
  (final_miles = 59000) →
  (init_gas = 2) →
  (gas1 = 8) →
  (gas2 = 15) →
  (gas3 = 25) →
  avg_mpg 58000 59000 2 8 15 25 = 20.0 :=
begin
  intros,
  sorry,
end

end mpg_calculation_l218_218633


namespace bags_of_oranges_l218_218899

-- Define the total number of oranges in terms of bags B
def totalOranges (B : ℕ) : ℕ := 30 * B

-- Define the number of usable oranges left after considering rotten oranges
def usableOranges (B : ℕ) : ℕ := totalOranges B - 50

-- Define the oranges to be sold after keeping some for juice
def orangesToBeSold (B : ℕ) : ℕ := usableOranges B - 30

-- The theorem to state that given 220 oranges will be sold,
-- we need to find B, the number of bags of oranges
theorem bags_of_oranges (B : ℕ) : orangesToBeSold B = 220 → B = 10 :=
by
  sorry

end bags_of_oranges_l218_218899


namespace find_angle_and_area_l218_218344

theorem find_angle_and_area (a b c : ℝ) (C : ℝ)
  (h₁: (a^2 + b^2 - c^2) * Real.tan C = Real.sqrt 2 * a * b)
  (h₂: c = 2)
  (h₃: b = 2 * Real.sqrt 2) : 
  C = Real.pi / 4 ∧ a = 2 ∧ (∃ S : ℝ, S = 1 / 2 * a * c ∧ S = 2) :=
by
  -- We assume sorry here since the focus is on setting up the problem statement correctly
  sorry

end find_angle_and_area_l218_218344


namespace inequality_solution_set_l218_218492

theorem inequality_solution_set :
  {x : ℝ | 2 * x^4 + x^2 - 2 * x - 3 * x^2 * abs(x - 1) + 1 ≥ 0} =
  Iic (-((1 + Real.sqrt 5) / 2)) ∪ Icc (-1) (1 / 2) ∪ Ici((Real.sqrt 5 - 1) / 2) :=
by
  sorry

end inequality_solution_set_l218_218492


namespace proj_scaled_vector_final_result_l218_218385

variables (u z : ℝ^2)

def proj (z u : ℝ^2) : ℝ^2 :=
  (u ⬝ z / ∥z∥^2) • z

theorem proj_scaled_vector :
  proj z (3 • u) = 3 • proj z u :=
by sorry

theorem final_result
  (h : proj z u = ![-1, 4]) :
  proj z (3 • u) = ![-3, 12] :=
by
  rw [←proj_scaled_vector, h]
  exact rfl

end proj_scaled_vector_final_result_l218_218385


namespace correct_calculation_is_7_88_l218_218133

theorem correct_calculation_is_7_88 (x : ℝ) (h : x * 8 = 56) : (x / 8) + 7 = 7.88 :=
by
  have hx : x = 7 := by
    linarith [h]
  rw [hx]
  norm_num
  sorry

end correct_calculation_is_7_88_l218_218133


namespace competition_end_time_l218_218159

def time := ℕ × ℕ -- Representing time as a pair of hours and minutes

def start_time : time := (15, 15) -- 3:15 PM is represented as 15:15 in 24-hour format
def duration := 1825 -- Duration in minutes
def end_time : time := (21, 40) -- 9:40 PM is represented as 21:40 in 24-hour format

def add_minutes (t : time) (m : ℕ) : time :=
  let (h, min) := t
  let total_minutes := h * 60 + min + m
  (total_minutes / 60 % 24, total_minutes % 60)

theorem competition_end_time :
  add_minutes start_time duration = end_time :=
by
  -- The proof would go here
  sorry

end competition_end_time_l218_218159


namespace places_proven_l218_218463

-- Definitions based on the problem conditions
inductive Place
| first
| second
| third
| fourth

def is_boy : String -> Prop
| "Oleg" => True
| "Olya" => False
| "Polya" => False
| "Pasha" => False
| _ => False

def name_starts_with_O : String -> Prop
| n => (n.head! = 'O')

noncomputable def determine_places : Prop :=
  ∃ (olegs_place olyas_place polyas_place pashas_place : Place),
  -- Statements and truth conditions
  ∃ (truthful : String), truthful ∈ ["Oleg", "Olya", "Polya", "Pasha"] ∧ 
  ∀ (person : String), 
    (person ≠ truthful → ∀ (statement : Place -> Prop), ¬ statement (person_to_place person)) ∧
    (person = truthful → person_to_place person = Place.first) ∧
    (person = truthful → 
      match person with
        | "Olya" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → is_boy (place_to_person p)
        | "Oleg" => ∃ (p : Place), (person_to_place "Oleg" = p ∧ person_to_place "Olya" = succ_place p ∨ 
                                    person_to_place "Olya" = p ∧ person_to_place "Oleg" = succ_place p)
        | "Pasha" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → name_starts_with_O (place_to_person p)
        | _ => True
      end)

-- Helper functions to relate places to persons
def person_to_place : String -> Place
| "Oleg" => Place.first
| "Olya" => Place.second
| "Polya" => Place.third
| "Pasha" => Place.fourth
| _ => Place.first -- Default, shouldn't happen

def place_to_person : Place -> String
| Place.first => "Oleg"
| Place.second => "Olya"
| Place.third => "Polya"
| Place.fourth => "Pasha"

def succ_place : Place → Place
| Place.first => Place.second
| Place.second => Place.third
| Place.third => Place.fourth
| Place.fourth => Place.first -- No logical next in this context.

theorem places_proven : determine_places :=
by
  sorry

end places_proven_l218_218463


namespace age_of_fourth_child_l218_218869

theorem age_of_fourth_child 
  (avg_age : ℕ) 
  (age1 age2 age3 : ℕ) 
  (age4 : ℕ)
  (h_avg : (age1 + age2 + age3 + age4) / 4 = avg_age) 
  (h1 : age1 = 6) 
  (h2 : age2 = 8) 
  (h3 : age3 = 11) 
  (h_avg_val : avg_age = 9) : 
  age4 = 11 := 
by 
  sorry

end age_of_fourth_child_l218_218869


namespace projection_of_a_on_b_l218_218300

theorem projection_of_a_on_b 
  (a b : ℝ → ℝ)
  (angle : ℝ)
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 4)
  (hangle : angle = 2 * real.pi / 3) :
  (dot_product a b) / ‖b‖ = -1 / 2 := 
sorry

end projection_of_a_on_b_l218_218300


namespace interest_percentage_face_value_l218_218817

def face_value : ℝ := 5000
def selling_price : ℝ := 6153.846153846153
def interest_percentage_selling_price : ℝ := 0.065

def interest_amount : ℝ := interest_percentage_selling_price * selling_price

theorem interest_percentage_face_value :
  (interest_amount / face_value) * 100 = 8 :=
by
  sorry

end interest_percentage_face_value_l218_218817


namespace sin_angle_GAC_is_correct_l218_218347

noncomputable def sin_angle_GAC : ℝ :=
  let A := (0:ℝ, 0:ℝ, 0:ℝ)
  let G := (2:ℝ, 2:ℝ, 2:ℝ)
  let C := (2:ℝ, 2:ℝ, 0:ℝ)
  let GA := (A.1 - G.1, A.2 - G.2, A.3 - G.3)
  let CA := (A.1 - C.1, A.2 - C.2, A.3 - C.3)
  let GA_dot_CA := GA.1 * CA.1 + GA.2 * CA.2 + GA.3 * CA.3
  let GA_mag := Real.sqrt (GA.1^2 + GA.2^2 + GA.3^2)
  let CA_mag := Real.sqrt (CA.1^2 + CA.2^2 + CA.3^2)
  let cosθ := GA_dot_CA / (GA_mag * CA_mag)
  Real.sqrt (1 - cosθ^2)

theorem sin_angle_GAC_is_correct :
  sin_angle_GAC = Real.sqrt(3) / 3 := 
by
  sorry

end sin_angle_GAC_is_correct_l218_218347


namespace calc_OP_MF_l218_218728

noncomputable def point (α : Type*) := α × α

def parabola (x y : ℝ) := y^2 = 8 * x

def focus_F := (2, 0)

def M_point (y₀ : ℝ) : point ℝ :=
  (y₀^2 / 8, y₀)

def N_point (y₀ : ℝ) (M : point ℝ) : point ℝ :=
  (y₀^2 / 16, y₀ / 2)

def slope_OM (y₀ : ℝ) : ℝ :=
  8 / y₀

def line_NP_eq (y₀ : ℝ) (x y : ℝ) := 
  y - y₀ / 2 = -y₀ / 8 * (x - y₀^2 / 16)

def intersect_x_axis (y₀ : ℝ) : point ℝ := 
  (y₀^2 / 16 + 4, 0)

def dist_MF (y₀ : ℝ) : ℝ := 
  y₀^2 / 8 + 2

def dist_OP (y₀ : ℝ) : ℝ :=
  y₀^2 / 16 + 4

theorem calc_OP_MF (y₀ : ℝ) : 2 * (dist_OP y₀) - (dist_MF y₀) = 6 :=
by
  sorry

end calc_OP_MF_l218_218728


namespace sum_of_all_four_digit_numbers_l218_218238

-- Let us define the set of digits
def digits : set ℕ := {1, 2, 3, 4, 5}

-- We will define a function that generates the four-digit numbers
def four_digit_numbers := {n : ℕ // ∃ a b c d : ℕ, 
                                      a ∈ digits ∧ 
                                      b ∈ digits ∧ 
                                      c ∈ digits ∧ 
                                      d ∈ digits ∧ 
                                      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ 
                                      b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                                      n = 1000 * a + 100 * b + 10 * c + d}

-- Define a function to calculate the sum of all elements in a set of numbers
def sum_set (s : set ℕ) : ℕ := s.fold (λa b, a + b) 0

theorem sum_of_all_four_digit_numbers :
  sum_set {n | ∃ x : four_digit_numbers, x.val = n} = 399960 :=
sorry

end sum_of_all_four_digit_numbers_l218_218238


namespace sum_of_all_four_digit_numbers_formed_l218_218268

open List

noncomputable def sum_of_four_digit_numbers (digits : List ℕ) : ℕ :=
  let perms := digits.permutations.filter (λ l, l.length = 4)
  let nums := perms.map (λ l, 1000 * l.head + 100 * l.nthLe 1 sorry + 10 * l.nthLe 2 sorry + l.nthLe 3 sorry)
  nums.sum

theorem sum_of_all_four_digit_numbers_formed : sum_of_four_digit_numbers [1, 2, 3, 4, 5] = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_formed_l218_218268


namespace find_x_l218_218669

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 156) (h2 : x ≥ 0) : x = 12 :=
sorry

end find_x_l218_218669


namespace slant_height_of_cone_l218_218532

theorem slant_height_of_cone (r : ℝ) (CSA : ℝ) (h_radius : r = 7) (h_CSA : CSA = 483.80526865282815) : 
  (l : ℝ) (h_l : l = CSA / (Real.pi * r)) (h_l_approx : Real.abs (l - 22) < 1) :=
by
  sorry

end slant_height_of_cone_l218_218532


namespace cubic_root_form_addition_l218_218874

theorem cubic_root_form_addition (p q r : ℕ) 
(h_root_form : ∃ x : ℝ, 2 * x^3 + 3 * x^2 - 5 * x - 2 = 0 ∧ x = (p^(1/3) + q^(1/3) + 2) / r) : 
  p + q + r = 10 :=
sorry

end cubic_root_form_addition_l218_218874


namespace ratio_of_boat_to_stream_l218_218166

theorem ratio_of_boat_to_stream (B S : ℝ) (h : ∀ D : ℝ, D / (B - S) = 2 * (D / (B + S))) :
  B / S = 3 :=
by 
  sorry

end ratio_of_boat_to_stream_l218_218166


namespace range_of_m_l218_218834

variable (m : ℝ)
def setA : Set ℝ := {x | x + m ≥ 0}
def setB : Set ℝ := {x | -2 < x ∧ x < 4}
def universalSet : Set ℝ := Set.univ

theorem range_of_m (h : (universalSet \ setA) ∩ setB = ∅) : 2 ≤ m := by
  sorry

end range_of_m_l218_218834


namespace milk_production_l218_218839

theorem milk_production (a b c x y z w : ℕ) : 
  ((b:ℝ) / c) * w + ((y:ℝ) / z) * w = (bw / c) + (yw / z) := sorry

end milk_production_l218_218839


namespace shooting_accuracy_l218_218809

theorem shooting_accuracy 
  (P_A : ℚ) 
  (P_AB : ℚ) 
  (h1 : P_A = 9 / 10) 
  (h2 : P_AB = 1 / 2) 
  : P_AB / P_A = 5 / 9 := 
by
  sorry

end shooting_accuracy_l218_218809


namespace distance_between_A_and_B_l218_218709

-- Define the points A and B in 3D space
def A : ℝ × ℝ × ℝ := (1, 3, -2)
def B : ℝ × ℝ × ℝ := (-2, 3, 2)

-- Define the function to calculate the distance between two points in 3D space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

-- The theorem that states the distance between points A and B
theorem distance_between_A_and_B : distance A B = 5 :=
  sorry

end distance_between_A_and_B_l218_218709


namespace sum_of_all_four_digit_numbers_l218_218253

def digits : List ℕ := [1, 2, 3, 4, 5]

noncomputable def four_digit_numbers := 
  (Digits.permutations digits).filter (λ l => l.length = 4)

noncomputable def sum_of_numbers (nums : List (List ℕ)) : ℕ :=
  nums.foldl (λ acc num => acc + (num.foldl (λ acc' digit => acc' * 10 + digit) 0)) 0

theorem sum_of_all_four_digit_numbers :
  sum_of_numbers four_digit_numbers = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_l218_218253


namespace tan_3285_eq_1_l218_218209

-- Define the initial angle and the reduced angle
def angle_3285 := 3285
def reduced_angle := angle_3285 % 360

-- State the theorem: Compute the tangent of 3285 degrees
theorem tan_3285_eq_1 : Real.tan (ofNat angle_3285 * Real.pi / 180) = 1 :=
by
  have h : reduced_angle = 45 := calc
    reduced_angle = 3285 % 360 : rfl
    ... = 45 : by norm_num
  rw [← Real.Mod.angle_of_nat_eq_of_int_reduced h]
  exact Real.tan_45

end tan_3285_eq_1_l218_218209


namespace product_of_primes_l218_218077

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l218_218077


namespace math_problem_l218_218218

noncomputable def f (x : ℚ) : ℚ := -2 * x / (4 * x + 3)
noncomputable def g (x : ℚ) : ℚ := (x + 2) / (2 * x + 1)

noncomputable def h : ℕ → (ℚ → ℚ) := λ n x, if n = 1 then g (f x) else g (f (h (n - 1) x))

noncomputable def sum_alternating (n : ℕ) (f : ℕ → ℚ) : ℚ :=
  ∑ i in Finset.range (n + 1), (-1) ^ (i + 1) * f (i + 1)

theorem math_problem :
  ∃ a b c : ℤ, (∑ k in Finset.range 100, (-1 : ℤ) ^ (k + 1) * (h 100 (k + 1) : ℚ)) = (a : ℚ) * (b : ℚ) ^ c ∧
               b ≠ 1 ∧
               a + b + c = 102 :=
by
  sorry

end math_problem_l218_218218


namespace orchids_initially_three_l218_218900

-- Define initial number of roses and provided number of orchids in the vase
def initial_roses : ℕ := 9
def added_orchids (O : ℕ) : ℕ := 13
def added_roses : ℕ := 3
def difference := 10

-- Define initial number of orchids that we need to prove
def initial_orchids (O : ℕ) : Prop :=
  added_orchids O - added_roses = difference →
  O = 3

theorem orchids_initially_three :
  initial_orchids O :=
sorry

end orchids_initially_three_l218_218900


namespace person_speed_kmh_l218_218953

-- Given conditions
def distance_meters : ℝ := 1000
def time_minutes : ℝ := 10

-- Proving the speed in km/h
theorem person_speed_kmh :
  (distance_meters / 1000) / (time_minutes / 60) = 6 :=
  sorry

end person_speed_kmh_l218_218953


namespace gcf_factorial_7_6_l218_218232

theorem gcf_factorial_7_6 (n m : ℕ) (h1 : n = 7!) (h2 : m = 6!) : Nat.gcd n m = 720 :=
by
  have h7 : 7! = 5040 := rfl
  have h6 : 6! = 720 := rfl
  sorry

end gcf_factorial_7_6_l218_218232


namespace find_number_l218_218134

noncomputable def number_found (x : ℝ) : Prop :=
  4 * x - 3 = 9 * (x - 7)

theorem find_number : ∃ (x : ℝ), number_found x ∧ x = 12 :=
by
  use 12
  simp [number_found]
  norm_num
  sorry

end find_number_l218_218134


namespace train_crosses_second_platform_l218_218177

theorem train_crosses_second_platform (
  length_train length_platform1 length_platform2 : ℝ) 
  (time_platform1 : ℝ) 
  (H1 : length_train = 100)
  (H2 : length_platform1 = 200)
  (H3 : length_platform2 = 300)
  (H4 : time_platform1 = 15) :
  ∃ t : ℝ, t = 20 := by
  sorry

end train_crosses_second_platform_l218_218177


namespace verify_placements_l218_218454

-- Definitions for participants and their possible places
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

-- Each participant should be mapped to a place (1, 2, 3, 4)
def Place : Participant → ℕ := λ p,
  match p with
  | Participant.Olya => 2
  | Participant.Oleg => 1
  | Participant.Polya => 3
  | Participant.Pasha => 4

-- Conditions based on the problem statement
def statement_Olya : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Polya % 2 = 1 ∧ Place Participant.Pasha % 2 = 1)

def statement_Oleg : Prop :=
  (abs (Place Participant.Oleg - Place Participant.Olya) = 1)

def statement_Pasha : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Olya % 2 = 1 ∧ Place Participant.Polya % 2 = 1)

-- Only one child tells the truth and the others lie
def exactly_one_true (a b c : Prop) : Prop := (a ∨ b ∨ c) ∧ (a → ¬b ∧ ¬c) ∧ (b → ¬a ∧ ¬c) ∧ (c → ¬a ∧ ¬b)

-- The main theorem to be proven
theorem verify_placements :
  exactly_one_true (statement_Olya) (statement_Oleg) (statement_Pasha) ∧ 
  Place Participant.Olya = 2 ∧
  Place Participant.Oleg = 1 ∧
  Place Participant.Polya = 3 ∧
  Place Participant.Pasha = 4 :=
by
  sorry

end verify_placements_l218_218454


namespace count_points_with_two_same_l218_218361

def coord_set : set ℝ := {2, 4, 6}

def is_valid_point (x y z : ℝ) : Prop :=
  x ∈ coord_set ∧ y ∈ coord_set ∧ z ∈ coord_set

def exactly_two_same (x y z : ℝ) : Prop :=
  (x = y ∧ x ≠ z) ∨ (y = z ∧ y ≠ x) ∨ (x = z ∧ x ≠ y)

theorem count_points_with_two_same : finset.card {
  (x, y, z) ∈ finset.product (finset.product coord_set.to_finset coord_set.to_finset) coord_set.to_finset |
  is_valid_point x y z ∧ exactly_two_same x y z
} = 18 := 
sorry

end count_points_with_two_same_l218_218361


namespace geometric_sequence_log_sum_l218_218386

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_log_sum
  (a : ℕ → ℝ) (r : ℝ) (hgeo : geometric_sequence a r)
  (hpos : ∀ n, 0 < a n)
  (h₇ : a 3 * a 7 = 64) :
  (∑ i in finset.range 9, real.logb 2 (a (i + 1))) = 27 :=
sorry

end geometric_sequence_log_sum_l218_218386


namespace acute_angle_bisectors_l218_218943

def is_convex_quadrilateral (A B C D : Point) : Prop := sorry
def opposite_angles (A B C D : Point) (α γ : ℝ) : Prop := 
  ∠A = α ∧ ∠C = γ ∧ α + γ = π

theorem acute_angle_bisectors 
  (A B C D : Point) (α γ : ℝ) 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_opposite : opposite_angles A B C D α γ)
  (h_order : α < γ) :
  acute_angle_bisector A B C D = (γ - α) / 2 :=
sorry

end acute_angle_bisectors_l218_218943


namespace product_of_primes_is_66_l218_218123

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l218_218123


namespace range_of_a_l218_218772

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (1 / 4)^x + (1 / 2)^(x - 1) + a = 0) →
  (-3 < a ∧ a < 0) :=
by
  sorry

end range_of_a_l218_218772


namespace polynomial_solution_l218_218680

theorem polynomial_solution {p : ℤ[X]} (h : p.eval₂ ((2 : ℤ) * X) (p.eval p) = (2 * X * p.eval X) + (3 * X^2)) : p = 3 * X := 
sorry

end polynomial_solution_l218_218680


namespace largest_possible_perimeter_l218_218622

noncomputable def max_perimeter_triangle : ℤ :=
  let a : ℤ := 7
  let b : ℤ := 9
  let x : ℤ := 15
  a + b + x

theorem largest_possible_perimeter (x : ℤ) (h1 : 7 + 9 > x) (h2 : 7 + x > 9) (h3 : 9 + x > 7) : max_perimeter_triangle = 31 := by
  sorry

end largest_possible_perimeter_l218_218622


namespace candidateA_prob_second_third_correct_after_first_incorrect_candidateA_more_likely_to_be_hired_based_on_variance_l218_218982

-- Definitions/Conditions from the problem
def q_total := 8
def q_select := 3
def correct_A := 6
def correct_prob_B := 3/4

noncomputable def prob_A_first_incorrect : ℚ := (q_total - correct_A) / q_total
noncomputable def prob_A_second_correct_given_first_incorrect : ℚ := correct_A / (q_total - 1)
noncomputable def prob_A_third_correct_given_second_correct : ℚ := (correct_A - 1) / (q_total - 2)
noncomputable def prob_A_B_given_A : ℚ := prob_A_first_incorrect * prob_A_second_correct_given_first_incorrect * prob_A_third_correct_given_second_correct
noncomputable def prob_B_given_A_conditional : ℚ := prob_A_B_given_A / prob_A_first_incorrect

def expectation_X_CandidateA : ℚ := 27/12  -- E(X) for Candidate A
def variance_X_CandidateA : ℚ := 45/112  -- Var(X) for Candidate A
def expectation_Y_CandidateB : ℚ := 27/12  -- E(Y) for Candidate B
def variance_Y_CandidateB : ℚ := 9/16   -- Var(Y) for Candidate B

-- Lean Statement for the first question
theorem candidateA_prob_second_third_correct_after_first_incorrect :
  prob_B_given_A_conditional = 5 / 7 := sorry

-- Lean Statement for the second question based on expectation and variance
theorem candidateA_more_likely_to_be_hired_based_on_variance :
  variance_X_CandidateA < variance_Y_CandidateB := sorry

end candidateA_prob_second_third_correct_after_first_incorrect_candidateA_more_likely_to_be_hired_based_on_variance_l218_218982


namespace parents_not_more_than_children_l218_218351

-- Defining the families and conditions.
variables (F : Type) (P : F → ℕ) (C : F → ℕ) (B G : ℕ) (Boys Girls : set ℕ)

-- Defining parents and children sets.
variables (parents children : set F)

-- Assumptions based on the problem's conditions.
-- Every couple has at least one child.
axiom couple_has_child (f : F) : C f ≥ 1

-- Every child has exactly two parents.
axiom two_parents_per_child (f : F) : P f = 2 * (C f)

-- Every little boy has a sister.
axiom every_boy_has_sister (b : ℕ) : b ∈ Boys → ∃ g ∈ Girls, g ∈ Boys

-- Among the children there are more boys than girls.
axiom more_boys_than_girls : B > G

-- No grandparents living in the building.
axiom no_grandparents (f : F) : parents = ∪ (P f) ∧ children = ∪ (C f)

-- Proof statement that it is not possible for there to be more parents than children.
theorem parents_not_more_than_children :
  ∀ (parents children : F → ℕ), (∀ f, P f = 2) →
  (∀ b ∈ Boys, ∃ g ∈ Girls, children(b) ∈ children) →
  (B > G) →
  ∑ parents < ∑ children
:=
sorry

end parents_not_more_than_children_l218_218351


namespace hyperbola_ellipse_b_value_l218_218699

theorem hyperbola_ellipse_b_value (a c b : ℝ) (h1 : c = 5 * a / 4) (h2 : c^2 - a^2 = (9 * a^2) / 16) (h3 : 4 * (b^2 - 4) = 16 * b^2 / 25) :
  b = 6 / 5 ∨ b = 10 / 3 :=
by
  sorry

end hyperbola_ellipse_b_value_l218_218699


namespace police_officer_can_catch_gangster_l218_218955

theorem police_officer_can_catch_gangster
  (a : ℝ) -- length of the side of the square
  (v_police : ℝ) -- maximum speed of the police officer
  (v_gangster : ℝ) -- maximum speed of the gangster
  (h_gangster_speed : v_gangster = 2.9 * v_police) :
  ∃ (t : ℝ), t ≥ 0 ∧ (a / (2 * v_police)) = t := sorry

end police_officer_can_catch_gangster_l218_218955


namespace parallel_planes_l218_218694

variables (α β : Type) [plane α] [plane β]
variables (a b : Type) [line a] [line b]
variable (A : point)
variables {a_in_alpha : a ∈ α} {b_in_alpha : b ∈ α}
variables (a_parallel_b_not : not (parallel a β)) (b_parallel_b_not : not (parallel b β))

theorem parallel_planes (h_intersect: ∃ (A : point), a ∩ b = A) : parallel α β :=
by
  sorry

end parallel_planes_l218_218694


namespace part_a_part_b_l218_218685

-- Define the functions K_m and K_4
def K (m : ℕ) (x y z : ℝ) : ℝ :=
  x * (x - y)^m * (x - z)^m + y * (y - x)^m * (y - z)^m + z * (z - x)^m * (z - y)^m

-- Define M
def M (x y z : ℝ) : ℝ :=
  (x - y)^2 * (y - z)^2 * (z - x)^2

-- The proof goals:
-- 1. Prove K_m >= 0 for odd positive integer m
theorem part_a (m : ℕ) (hm : m % 2 = 1) (x y z : ℝ) : 
  0 ≤ K m x y z := 
sorry

-- 2. Prove K_7 + M^2 * K_1 >= M * K_4
theorem part_b (x y z : ℝ) : 
  K 7 x y z + (M x y z)^2 * K 1 x y z ≥ M x y z * K 4 x y z := 
sorry

end part_a_part_b_l218_218685


namespace circle_units_diff_l218_218940

-- Define the context where we verify the claim about the circle

noncomputable def radius : ℝ := 3
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def area (r : ℝ) := Real.pi * r ^ 2

-- Lean Theorem statement that needs to be proved
theorem circle_units_diff (r : ℝ) (h₀ : r = radius) :
  circumference r ≠ area r :=
by sorry

end circle_units_diff_l218_218940


namespace sum_four_digit_numbers_l218_218247

def digits : List ℕ := [1, 2, 3, 4, 5]

/-- 
  Prove that the sum of all four-digit numbers that can be formed 
  using the digits 1, 2, 3, 4, 5 exactly once is 399960.
-/
theorem sum_four_digit_numbers : 
  (Finset.sum 
    (Finset.map 
      (λ l, 
        l.nth_le 0 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1000 + 
        l.nth_le 1 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 100 + 
        l.nth_le 2 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 10 + 
        l.nth_le 3 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1) 
      (digits.permutations.filter (λ l, l.nodup ∧ l.length = 4))) id) 
  = 399960 :=
sorry

end sum_four_digit_numbers_l218_218247


namespace road_paving_100_km_l218_218891

noncomputable def a : ℕ → ℝ
| 0       := 0
| (n + 1) := if n = 0 then 1 else a n + (1 / a n)

theorem road_paving_100_km :
  ∃ n : ℕ, a n ≥ 100 :=
begin
  sorry
end

end road_paving_100_km_l218_218891


namespace probability_sum_16_l218_218663

open ProbabilityTheory 

-- Definitions of conditions
def fair_coin := {5, 15}
def fair_die := {1, 2, 3, 4, 5, 6}

-- Probability calculations
def probability_of_15_on_coin : ℝ := 1 / 2
def probability_of_1_on_die : ℝ := 1 / 6

-- Target statement
theorem probability_sum_16 :
  ∃ p : ℝ, p = probability_of_15_on_coin * probability_of_1_on_die ∧ p = 1 / 12 :=
begin
  use 1 / 12,
  split,
  { 
    exact mul_div_cancel' (by norm_num) (by norm_num),
  },
  { 
    refl,
  }
end

end probability_sum_16_l218_218663


namespace solve_system_l218_218866

def equations (x y : ℝ) : Prop :=
  (x^2 * y - x * y^2 - 3 * x + 3 * y + 1 = 0) ∧ 
  (x^3 * y - x * y^3 - 3 * x^2 + 3 * y^2 + 3 = 0)

theorem solve_system : ∃ x y : ℝ, equations x y ∧ x = 2 ∧ y = 1 := 
by
  use 2
  use 1
  unfold equations
  split
  sorry
  split
  rfl
  rfl

end solve_system_l218_218866


namespace perpendicular_vector_x_value_l218_218329

theorem perpendicular_vector_x_value :
  let a := (-5, 1)
  let b (x : ℝ) := (2, x)
  (a.1 * b(x).1 + a.2 * b(x).2 = 0) → x = 10 :=
by
  intros
  sorry

end perpendicular_vector_x_value_l218_218329


namespace taxi_ride_cost_l218_218967

theorem taxi_ride_cost :
  let base_fare : ℝ := 2.00
  let cost_per_mile_first_3 : ℝ := 0.30
  let cost_per_mile_additional : ℝ := 0.40
  let total_distance : ℕ := 8
  let first_3_miles_cost : ℝ := base_fare + 3 * cost_per_mile_first_3
  let additional_miles_cost : ℝ := (total_distance - 3) * cost_per_mile_additional
  let total_cost : ℝ := first_3_miles_cost + additional_miles_cost
  total_cost = 4.90 :=
by
  sorry

end taxi_ride_cost_l218_218967


namespace incorrect_statement_a_l218_218763

-- Define g(x)
def g (x : ℝ) : ℝ := (x - 3) / (x + 4)

-- Define the statement to be proven: statement (A) is incorrect
theorem incorrect_statement_a (x y : ℝ) (h : y = g x) : x ≠ (y - 3) / (y + 4) :=
by sorry

end incorrect_statement_a_l218_218763


namespace piles_can_be_combined_l218_218442

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l218_218442


namespace cone_angle_l218_218510

theorem cone_angle (r l : ℝ) (α : ℝ)
  (h1 : 2 * Real.pi * r = Real.pi * l) 
  (h2 : Real.cos α = r / l) : α = Real.pi / 3 :=
by
  sorry

end cone_angle_l218_218510


namespace ratio_of_bases_of_isosceles_trapezoid_l218_218497

theorem ratio_of_bases_of_isosceles_trapezoid {R : ℝ} (α : ℝ) 
  (hα : α = 75) (AD BC : ℝ) (h1 : AD = 2 * R * real.sin (120))
  (h2 : BC = 2 * R * real.sin (30)) : AD / BC = real.sqrt 3 :=
by
  sorry

end ratio_of_bases_of_isosceles_trapezoid_l218_218497


namespace exists_root_interval_l218_218198

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_interval :
  (f 1.1 < 0) ∧ (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 := 
by
  intro h
  sorry

end exists_root_interval_l218_218198


namespace probability_drawing_red_gold_six_total_l218_218575

noncomputable def red_stars : ℕ := 3
noncomputable def gold_stars : ℕ := 4
noncomputable def silver_stars : ℕ := 5

noncomputable def total_stars : ℕ := red_stars + gold_stars + silver_stars

def probability_red_star_on_top (total_stars red_stars gold_stars silver_stars : ℕ) : ℚ :=
  let first_red := (red_stars : ℚ) / (total_stars : ℚ)
  let remaining_red := red_stars - 1
  let remaining_stars := total_stars - 1
  let ways_to_choose_3_gold := choose gold_stars 3
  let ways_to_choose_2_silver := choose silver_stars 2
  let ways_to_choose_0_red := choose remaining_red 0
  let favorable_outcomes := ways_to_choose_3_gold * ways_to_choose_2_silver * ways_to_choose_0_red
  let total_ways_to_choose_5 := choose remaining_stars 5
  first_red * (favorable_outcomes / total_ways_to_choose_5)

theorem probability_drawing_red_gold_six_total :
  probability_red_star_on_top total_stars red_stars gold_stars silver_stars = 5 / 231 := sorry

end probability_drawing_red_gold_six_total_l218_218575


namespace A_B_independent_hits_probability_l218_218591

noncomputable def probability_A_hits_twice_and_B_hits_thrice 
  (pA : ℚ) (pB : ℚ) (qA : ℚ) (qB : ℚ) (hits_A : ℕ) (shots : ℕ) (hits_B : ℕ) : ℚ :=
  (nat.choose shots hits_A * pA ^ hits_A * qA ^ (shots - hits_A)) *
  (nat.choose shots hits_B * pB ^ hits_B * qB ^ (shots - hits_B))

theorem A_B_independent_hits_probability :
  let pA := 2 / 3,
      pB := 3 / 4,
      qA := 1 - pA,
      qB := 1 - pB,
      shots := 4,
      hits_A := 2,
      hits_B := 3 in
  probability_A_hits_twice_and_B_hits_thrice pA pB qA qB hits_A shots hits_B = 1 / 8 :=
by
  sorry

end A_B_independent_hits_probability_l218_218591


namespace determine_summer_areas_l218_218884

-- Define the predicate for entering summer
def entered_summer (temperatures : List ℕ) : Prop :=
  temperatures.length = 5 ∧ temperatures.sum / 5 ≥ 22

-- Conditions for Area A
def condition_A (temperatures : List ℕ) : Prop :=
  List.median temperatures = 24 ∧ List.mode temperatures = 22

-- Conditions for Area B
def condition_B (temperatures : List ℕ) : Prop :=
  List.median temperatures = 25 ∧ (temperatures.sum / 5 = 24)

-- Conditions for Area C
def condition_C (temperatures : List ℕ) : Prop :=
  (temperatures.sum / 5 = 22) ∧ (List.mode temperatures = 22)

-- Conditions for Area D
def condition_D (temperatures : List ℕ) : Prop :=
  temperatures.contains 28 ∧ (temperatures.sum / 5 = 24) ∧ 
  (List.variance temperatures = 4.8)

-- The main proof problem statement
theorem determine_summer_areas
  (temperatures_A temperatures_B temperatures_C temperatures_D : List ℕ) :
  (condition_A temperatures_A → entered_summer temperatures_A) ∧
  (condition_D temperatures_D → entered_summer temperatures_D) :=
by 
  sorry

end determine_summer_areas_l218_218884


namespace probability_x_plus_y_lt_4_in_square_l218_218169

theorem probability_x_plus_y_lt_4_in_square :
  let square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let region := {p : ℝ × ℝ | p ∈ square ∧ p.1 + p.2 < 4}
  (measure_of region / measure_of square) = 7 / 9 := sorry

end probability_x_plus_y_lt_4_in_square_l218_218169


namespace sum_of_chosen_numbers_l218_218887

theorem sum_of_chosen_numbers (k : ℕ) (hk : k > 0) : 
  (∑ i in finset.range k, k * i + (i + 1)) = (k * (k^2 + 1)) / 2 :=
by
  sorry

end sum_of_chosen_numbers_l218_218887


namespace sqrt_9_eq_3_and_neg3_l218_218895

theorem sqrt_9_eq_3_and_neg3 : { x : ℝ | x^2 = 9 } = {3, -3} :=
by
  sorry

end sqrt_9_eq_3_and_neg3_l218_218895


namespace original_average_weight_l218_218542

theorem original_average_weight 
  (W : ℝ)
  (h1 : 7 * W + 110 + 60 = 9 * 78) : 
  W = 76 := 
by
  sorry

end original_average_weight_l218_218542


namespace countWays_quarter_no_quarter_l218_218333

-- Defining the context that represents the types of coins available
def coin := ℕ   -- Represent as natural number for cent value
def penny : coin := 1
def nickel : coin := 5
def dime : coin := 10
def quarter : coin := 25

-- Function to calculate the number of ways to make change for a given amount
noncomputable def countWays (amount : coin) (excludeQuarter : bool) : ℕ := sorry

-- The theorem to be proved
theorem countWays_quarter_no_quarter : countWays quarter true = 12 := by
  sorry

end countWays_quarter_no_quarter_l218_218333


namespace inverse_of_composite_bijective_l218_218213

open Function

variable {α β γ δ : Type}

theorem inverse_of_composite_bijective (s : β → γ) (t : α → β) (u : δ → α) 
  (hs : Bijective s) (ht : Bijective t) (hu : Bijective u) :
  (s ∘ t ∘ u)⁻¹ = u⁻¹ ∘ t⁻¹ ∘ s⁻¹ :=
by
  sorry

end inverse_of_composite_bijective_l218_218213


namespace percentage_present_l218_218775

theorem percentage_present (total_workers present_workers : ℕ) 
  (h_total : total_workers = 86) 
  (h_present : present_workers = 72) : 
  (float_of_nat present_workers / float_of_nat total_workers) * 100 ≈ 83.7 :=
by
  intros
  have h_fraction : (float_of_nat present_workers / float_of_nat total_workers) * 100 = 83.72093023255814
    := by
  -- sorry: here you'd put the calculation steps, but they're not required
  norm_num at h_fraction -- assumes normative rounding is applied
  -- sorry: prove rounding consistency here
  sorry

end percentage_present_l218_218775


namespace range_of_x_l218_218328

-- Define the function h(a).
def h (a : ℝ) : ℝ := a^2 + 2 * a + 3

-- Define the main theorem
theorem range_of_x (a : ℝ) (x : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) : 
  x^2 + 4 * x - 2 ≤ h a → -5 ≤ x ∧ x ≤ 1 :=
sorry

end range_of_x_l218_218328


namespace linear_function_expression_quadratic_vertex_on_line_range_of_t_l218_218794

noncomputable def linear_function (x : ℝ) (k b : ℝ) := k * x + b

noncomputable def quadratic_function (x : ℝ) (m n : ℝ) := x^2 + m * x + n

def intersects_x_axis_at (f : ℝ → ℝ) (a : ℝ × ℝ) : Prop := f a.1 = a.2
def intersects_y_axis_at (f : ℝ → ℝ) (b : ℝ × ℝ) : Prop := f b.1 = b.2

-- Part 1: Find the expression of the linear function
theorem linear_function_expression :
  ∀ k b : ℝ, k ≠ 0 →
  intersects_x_axis_at (linear_function x k b) (-3, 0) →
  intersects_y_axis_at (linear_function x k b) (0, -3) →
  ∃ k b : ℝ, linear_function x k b = -x - 3 :=
by sorry

-- Part 2: Determine if vertex of the quadratic function lies on the line
theorem quadratic_vertex_on_line :
  ∀ m : ℝ, 
  (∃ n : ℝ, n = 3) →
  intersects_x_axis_at (quadratic_function x m 3) (-3, 0) →
  intersects_y_axis_at (quadratic_function x m 3) (0, 3) →
  (∃ m n : ℝ, quadratic_function x m n = (x + 2)^2 - 1) →
  (∃ v : ℝ × ℝ, v = (-2, -1)) →
  intersects_y_axis_at (linear_function x -1 -3) (-2, -1) :=
by sorry

-- Part 3: Find the range of values for t
theorem range_of_t :
  ∀ (n m : ℝ) (t : ℝ),
  n > 0 →
  m ≤ 5 →
  t = (4 * n - m^2) / 4 →
  ∃ r : set ℝ, r = Ioo (-9 / 4) (-1 / 4) →
  t ∈ r :=
by sorry

end linear_function_expression_quadratic_vertex_on_line_range_of_t_l218_218794


namespace y_when_x_is_4_l218_218055

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l218_218055


namespace even_multiple_of_5_perfect_squares_lt_2500_l218_218753

theorem even_multiple_of_5_perfect_squares_lt_2500 :
  {n : ℕ // n < 2500 ∧ (∃ k, n = 100 * k^2)}.card = 4 :=
by
  sorry

end even_multiple_of_5_perfect_squares_lt_2500_l218_218753


namespace sum_of_fractional_monotonous_l218_218656

noncomputable def is_fractional_monotonous (seq : List ℝ) : Prop :=
  ∀ i j, (i < j ∧ i < seq.length ∧ j < seq.length) → (seq.nth i).getD 0.0 < (seq.nth j).getD 0.0 ∧ (seq.nth i).getD 0.0 % 1 < (seq.nth j).getD 0.0 % 1

theorem sum_of_fractional_monotonous:
  let sequence := [4.3, 4.4, 4.5, 4.6, 4.7, 4.8]
  is_fractional_monotonous sequence →
  sequence.sum = 26.3 := 
by
  intro h
  have h_eq : sequence = [4.3, 4.4, 4.5, 4.6, 4.7, 4.8] := rfl
  simp [h_eq]
  sorry

end sum_of_fractional_monotonous_l218_218656


namespace solve_for_a_l218_218719

theorem solve_for_a (x a : ℝ) (h : x = 3) (eq : 5 * x - a = 8) : a = 7 :=
by
  -- sorry to skip the proof as instructed
  sorry

end solve_for_a_l218_218719


namespace problem_statement_l218_218824

noncomputable def a_seq : ℕ → ℝ
| n => if n = 0 then a_1 else 3 * a_seq (n - 1) - 2 * Real.sqrt (a_seq (n - 1))

def S (n : ℕ) : ℝ := ∑ k in Finset.range n, a_seq k

theorem problem_statement (a_1 : ℝ) (h1 : ∀ n : ℕ, a_seq n > 0) :
  ∃ (n : ℕ), S n > 4 / 9 * n :=
sorry

end problem_statement_l218_218824


namespace min_value_PA_minus_PF_l218_218730

noncomputable def ellipse_condition : Prop :=
  ∃ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1)

noncomputable def focal_property (x y : ℝ) (P : ℝ × ℝ) : Prop :=
  dist P (2, 4) - dist P (1, 0) = 1

theorem min_value_PA_minus_PF :
  ∀ (P : ℝ × ℝ), 
    (∃ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1) 
    → ∃ (a b : ℝ), a = 2 ∧ b = 4 ∧ focal_property x y P :=
  sorry

end min_value_PA_minus_PF_l218_218730


namespace inscribed_n_gon_parallel_circumscribed_n_gon_vertices_on_lines_l218_218570

-- Define the circle and the given n lines through the center O
def circle (O : EuclideanGeometry.Point) (r : ℝ) : Set EuclideanGeometry.Point := 
  { P | euclidean_dist O P = r }

def given_n_lines (O : EuclideanGeometry.Point) (n : ℕ) : Set (Set EuclideanGeometry.Point) :=
  { l | ∃ i, i < n ∧ is_line_through O l }

-- Theorem for part a)
theorem inscribed_n_gon_parallel (O : EuclideanGeometry.Point) (r : ℝ) (n : ℕ) 
  (lines : Fin n → EuclideanGeometry.Point × EuclideanGeometry.Point) :
  (∃ A₁ A₂ ... Aₙ : EuclideanGeometry.Point,
    (A₁ A₂ ... Aₙ are vertices of an inscribed n-gon in (circle O r)) ∧
    (A₁A₂, A₂A₃, ..., AₙA₁ are parallel to lines)) := sorry

-- Theorem for part b)
theorem circumscribed_n_gon_vertices_on_lines (O : EuclideanGeometry.Point) (r : ℝ) (n : ℕ) 
  (lines : Fin n → EuclideanGeometry.Point × EuclideanGeometry.Point) :
  (∃ B₁ B₂ ... Bₙ : EuclideanGeometry.Point,
    (B₁ B₂ ... Bₙ are vertices of a circumscribed n-gon around (circle O r)) ∧
    (B₁, B₂, ..., Bₙ lie on the given lines through O)) := sorry

end inscribed_n_gon_parallel_circumscribed_n_gon_vertices_on_lines_l218_218570


namespace divisor_fourth_power_mod_13_l218_218013

theorem divisor_fourth_power_mod_13
  (a : ℤ) (d : ℕ) 
  (h1 : d > 0)
  (h2 : d ∣ (a^4 + a^3 + 2 * a^2 - 4 * a + 3)) : 
  ∃ k : ℕ, d ≡ k^4 [MOD 13] :=
sorry

end divisor_fourth_power_mod_13_l218_218013


namespace competition_results_correct_l218_218457

theorem competition_results_correct :
  ∃ (first second third fourth : String), 
    (first = "Oleg" ∧ second = "Olya" ∧ third = "Polya" ∧ fourth = "Pasha") ∧
    ∀ (claims : String → String → Prop),
      (claims "Olya" "all_odd_places_boys") ∧ 
      (claims "Oleg" "consecutive_places_with_olya") ∧
      (claims "Pasha" "all_odd_places_names_start_O") ∧
      ∃ (truth_teller : String), 
        truth_teller = "Oleg" ∧ 
        (claims "Oleg" "first_place") ∧ 
        ¬ (claims "Olya" "first_place") ∧ 
        ¬ (claims "Pasha" "first_place") ∧ 
        ¬ (claims "Polya" "first_place") :=
sorry

end competition_results_correct_l218_218457


namespace max_abs_sum_l218_218767

theorem max_abs_sum {x y : ℝ} (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * real.sqrt 2 :=
by
  sorry

end max_abs_sum_l218_218767


namespace combine_piles_l218_218422

theorem combine_piles (n : ℕ) (piles : list ℕ) (h_piles : list.sum piles = n) (h_similar : ∀ x y ∈ piles, x ≤ y → y ≤ 2 * x) :
  ∃ pile, pile ∈ piles ∧ pile = n := sorry

end combine_piles_l218_218422


namespace exists_right_triangle_with_angle_gt_l218_218850

noncomputable theory

/-- Given any acute angle ε such that 0 < ε < π / 4, 
    proving the existence of a right triangle with natural number sides a, b, c,
    where the smaller acute angle of the triangle is greater than ε. -/
theorem exists_right_triangle_with_angle_gt (ε : ℝ) (hε : 0 < ε ∧ ε < π / 4) :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ arctan (a / b : ℝ) > ε :=
sorry

end exists_right_triangle_with_angle_gt_l218_218850


namespace distinct_values_expression_l218_218797

def is_distinct_values (values : List ℕ) : Prop :=
  values.nodup

theorem distinct_values_expression:
  let digits := [1, 2, 5, 6]
  ∃ values : List ℕ, is_distinct_values values ∧ values.length = 3 ∧
  (∀ a b c d, a ∈ digits → b ∈ digits → c ∈ digits → d ∈ digits →
   a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
   ((a * b) + (c * d) ∈ values)) :=
sorry

end distinct_values_expression_l218_218797


namespace find_a_for_cubic_sum_l218_218682

theorem find_a_for_cubic_sum (a : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - a * x1 + a + 2 = 0 ∧ 
    x2^2 - a * x2 + a + 2 = 0 ∧
    x1 + x2 = a ∧
    x1 * x2 = a + 2 ∧
    x1^3 + x2^3 = -8) ↔ a = -2 := 
by
  sorry

end find_a_for_cubic_sum_l218_218682


namespace product_of_primes_is_66_l218_218127

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l218_218127


namespace construct_triangle_l218_218216

def exists_triangle_with_base_angle_median 
    (A B : ℝ) -- points representing the endpoints of the base on the real line
    (c : ℝ)  -- length of the base
    (α : ℝ)  -- measure of the angle opposite the base
    (m : ℝ)  -- length of the median to the base
    : Prop :=
∃ C : ℝ, -- third vertex of the triangle
    dist A B = c ∧
    ∃ α' : ℝ, α' = α ∧
    let M := (A + B) / 2 in
    dist M C = m

theorem construct_triangle (A B : ℝ) (c α m : ℝ) :
  dist A B = c → ∃ C : ℝ, exists_triangle_with_base_angle_median A B c α m :=
sorry

end construct_triangle_l218_218216


namespace at_most_2012_missed_dots_l218_218068

theorem at_most_2012_missed_dots :
  ∀ (dots : ℕ → Prop) (arrows : ℕ → ℕ),
    (∀ m n, m ≠ n → arrows m ≠ arrows n) ∧
    (∀ n, abs (arrows n - n) ≤ 1006) →
    ∃ (t : ℕ), ∀ n, P (n : ℕ) → n ≤ t → ∃ m, arrows m = n → ∃ d, d ≤ 2012 :=
sorry

end at_most_2012_missed_dots_l218_218068


namespace validate_calculation_validate_square_validate_order_of_ops_validate_sub_neg_check_correct_answer_calculation_l218_218918

theorem validate_calculation (a b : ℤ) (h1 : a = 2) (h2 : b = 3) : a - b = -1 :=
by
  calc
    a - b = 2 - 3 : by rw [h1, h2]
        ... = -1   : by norm_num

theorem validate_square (c : ℤ) (h3 : c = -3) : (c * c ≠ -9) :=
by
  calc
    (c * c) = (-3) * (-3) : by rw [h3]
        ... = 9          : by norm_num

theorem validate_order_of_ops (d : ℤ) (h4 : d = 3) : (-d * d ≠ -6) :=
by
  calc
    (-d * d) = -(3 * 3)   : by rw [h4]
          ... = -9        : by norm_num

theorem validate_sub_neg (e f : ℤ) (h5 : e = -3) (h6 : f = -2) : (e - f ≠ -5) :=
by
  calc
    e - f = -3 - (-2)     : by rw [h5, h6]
        ... = -3 + 2      : by norm_num
        ... = -1          : by norm_num

-- stating the main theorem to make it conclusive
theorem check_correct_answer_calculation : (2 - 3 = -1) :=
  validate_calculation 2 3 rfl rfl

end validate_calculation_validate_square_validate_order_of_ops_validate_sub_neg_check_correct_answer_calculation_l218_218918


namespace monotonic_increasing_interval_of_log_function_l218_218035

open Real

noncomputable def f (x : ℝ) : ℝ := log 0.6 (6 + x - x^2)

theorem monotonic_increasing_interval_of_log_function :
  (∀ x, 6 + x - x^2 > 0) → (∀ x, f' x > 0) [\frac{1}{2}, 3) :=
by 
  sorry

end monotonic_increasing_interval_of_log_function_l218_218035


namespace arcsin_solution_l218_218019

-- Definitions to simplify the expressions.
def A (x : ℝ) := (x * Real.sqrt 35) / (4 * Real.sqrt 13)
def B (x : ℝ) := (x * Real.sqrt 35) / (3 * Real.sqrt 13)
def C (x : ℝ) := (x * Real.sqrt 35) / (2 * Real.sqrt 13)

theorem arcsin_solution (x : ℝ) (h : |x| ≤ (2 * Real.sqrt 13) / Real.sqrt 35) :
  Real.arcsin (A x) + Real.arcsin (B x) = Real.arcsin (C x) ↔ x = 0 ∨ x = 13 / 12 ∨ x = -13 / 12 :=
sorry

end arcsin_solution_l218_218019


namespace problem_l218_218167

noncomputable def x : ℕ := 11 * (2^2 * 3^2) * (2 * 3 * 7)
-- Factorize 36 and 42 to establish x's prime factors
def fact36 : ℕ := 2^2 * 3^2
def fact42 : ℕ := 2 * 3 * 7

theorem problem (y : ℕ) (h : y = 7^2 * 11^2) : 
  ∃ (y : ℕ), y = 5929 ∧ is_perfect_cube (x * y) :=
by
  sorry

constants (x : ℕ := 11 * (2^3 * 3^3 * 7)) 
(def is_perfect_cube (n : ℕ) : Prop := ∃ k, k^3 = n)


end problem_l218_218167


namespace union_M_N_intersection_complement_M_N_l218_218689

def M := {x : ℝ | 4 * x^2 - 4 * x - 15 > 0}
def N := {x : ℝ | (x + 1) / (6 - x) < 0}

theorem union_M_N : M ∪ N = {x : ℝ | x < -1 ∨ x ≥ 5 / 2} :=
by sorry

theorem intersection_complement_M_N : (set.univ \ M) ∩ (set.univ \ N) = {x : ℝ | -1 ≤ x ∧ x < 5 / 2} :=
by sorry

end union_M_N_intersection_complement_M_N_l218_218689


namespace albert_pizza_slices_l218_218973

theorem albert_pizza_slices :
  let large_pizzas := 2
  let slices_per_large_pizza := 16
  let small_pizzas := 2
  let slices_per_small_pizza := 8
  (large_pizzas * slices_per_large_pizza + small_pizzas * slices_per_small_pizza) = 48 :=
by
  have h1 : large_pizzas * slices_per_large_pizza = 32 := by sorry
  have h2 : small_pizzas * slices_per_small_pizza = 16 := by sorry
  have ht : 32 + 16 = 48 := by sorry
  exact ht

end albert_pizza_slices_l218_218973


namespace similar_sizes_combination_possible_l218_218406

theorem similar_sizes_combination_possible 
    (similar : Nat → Nat → Prop := λ x y, x ≤ y ∧ y ≤ 2 * x)
    (combine_piles : List Nat → Nat ∃ combined : Nat, (∀ x y ∈ combined, similar x y) → True
    (piles : List Nat) : True :=
sorry

end similar_sizes_combination_possible_l218_218406


namespace car_speed_in_kmph_l218_218595

theorem car_speed_in_kmph (d : ℝ) (t : ℝ) (conv_factor : ℝ) (h_d : d = 400) (h_t : t = 12) (h_conv : conv_factor = 3.6) : 
  (d / t) * conv_factor ≈ 120 :=
by
  have h1 : d / t = 400 / 12 := by rw [h_d, h_t]
  have h2 : (400 / 12) * 3.6 ≈ 120 := by
    calc (400 / 12) * 3.6 ≈ 33.33 * 3.6 : by norm_num
                     ... ≈ 120 : by norm_num
  rw [h1]
  rw [h_conv]
  exact h2

end car_speed_in_kmph_l218_218595


namespace triangle_probability_l218_218739

open Classical

theorem triangle_probability :
  let a := 5
  let b := 6
  let lengths := [1, 2, 6, 11]
  let valid_third_side x := 1 < x ∧ x < 11
  let valid_lengths := lengths.filter valid_third_side
  let probability := valid_lengths.length / lengths.length
  probability = 1 / 2 :=
by {
  sorry
}

end triangle_probability_l218_218739


namespace product_of_smallest_primes_l218_218094

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218094


namespace oatmeal_cookies_count_l218_218936

theorem oatmeal_cookies_count (total_cookies choc_chip_batches choc_chip_per_batch : ℕ) (h1 : total_cookies = 10)
  (h2 : choc_chip_batches = 2) (h3 : choc_chip_per_batch = 3) :
  ∃ oatmeal_cookies, oatmeal_cookies = total_cookies - choc_chip_batches * choc_chip_per_batch ∧ oatmeal_cookies = 4 :=
by {
  use (total_cookies - choc_chip_batches * choc_chip_per_batch),
  split,
  { rw [total_cookies, choc_chip_batches, choc_chip_per_batch],
    exact h1.symm.sub (h2.symm.mul h3.symm) },
  { exact rfl },
  sorry
}

end oatmeal_cookies_count_l218_218936


namespace percent_correct_l218_218476

theorem percent_correct (x : ℕ) : 
  (5 * 100.0 / 7) = 71.43 :=
by
  sorry

end percent_correct_l218_218476


namespace angle_bisector_l218_218697

theorem angle_bisector (ABMC : convex_quadrilateral) (h1 : ABMC.AB = ABMC.BC)
  (h2 : angle BAM = 30) (h3 : angle ACM = 150) : 
  is_angle_bisector AM BMC :=
sorry

end angle_bisector_l218_218697


namespace equilateral_iff_total_distance_4_l218_218376

variable (P Q R : Type)
variable [EuclideanGeometry P] [EuclideanGeometry Q] [EuclideanGeometry R]

def is_centroid (O : P) (A B C : P) (G : P) : Prop :=
  ∃ (G : P), centroid A B C G

def circumcenter {P : Type} [EuclideanGeometry P] (O : P) (A B C : P) : Prop :=
  ∃ (R : ℝ), R = 1 ∧ circle A B C O R

def is_equilateral (A B C : P) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def total_distance (A B C G1 G2 G3 : P) : ℝ := 
  dist A G1 + dist B G2 + dist C G3

theorem equilateral_iff_total_distance_4
  (A B C O G1 G2 G3 : P)
  (hCircumcenter : circumcenter O A B C)
  (hCentroid1 : is_centroid O B C G1)
  (hCentroid2 : is_centroid O A C G2)
  (hCentroid3 : is_centroid O A B G3) :
  is_equilateral A B C ↔ total_distance A B C G1 G2 G3 = 4 := 
sorry

end equilateral_iff_total_distance_4_l218_218376


namespace curve2_equation_curve3_polar_equation_max_value_PQ_l218_218802

open Real

noncomputable def Curve1 (α : ℝ) : ℝ × ℝ :=
  (cos α, sin α)

noncomputable def Curve2 (α : ℝ) : ℝ × ℝ :=
  (2 * cos α, sin α)

noncomputable def Curve3 (α : ℝ) : ℝ × ℝ :=
  (cos α, 1 + sin α)

theorem curve2_equation :
  ∃ α, ∀ x y : ℝ, (x, y) = Curve2 α → x^2 / 4 + y^2 = 1 :=
sorry

theorem curve3_polar_equation :
  ∃ α θ : ℝ, ∀ ρ : ℝ, (ρ, θ) =  (sqrt ((cos α) ^ 2 + (1 + sin α) ^ 2), atan ((1 + sin α) / cos α)) → ρ = 2 * sin θ :=
sorry

theorem max_value_PQ :
  ∃ P Q : ℝ × ℝ, P ∈ (set.range Curve2) ∧ Q ∈ (set.range Curve3) ∧ |(fst P - fst Q)^2 + (snd P - snd Q)^2| ≤ (4 * sqrt 3 / 3 + 1)^2 :=
sorry

end curve2_equation_curve3_polar_equation_max_value_PQ_l218_218802


namespace product_of_primes_l218_218074

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l218_218074


namespace piles_to_single_pile_l218_218432

-- Define the condition similar_sizes
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

-- Define the inductive step of combining stones
def combine_stones (piles : List ℕ) : List ℕ :=
  if ∃ x y, x ∈ piles ∧ y ∈ piles ∧ similar_sizes x y then
    let ⟨x, hx, y, hy, hsim⟩ := Classical.some_spec (Classical.some_spec_exists _)
    List.cons (x + y) (List.erase (List.erase piles x) y)
  else
    piles

-- Prove that a collection of piles can be reduced to a single pile of size n
theorem piles_to_single_pile (piles : List ℕ) (h : ∀ x ∈ piles, x = 1) : 
  ∃ p, list.length (Iterator.iterate combine_stones piles.count) 1 = 1 := by
  sorry

end piles_to_single_pile_l218_218432


namespace Ramu_selling_price_l218_218486

-- Definitions of the conditions
def purchase_price : ℝ := 42000
def repair_costs : ℝ := 13000
def profit_percent : ℝ := 12.545454545454545 / 100

-- Derived values
def total_cost : ℝ := purchase_price + repair_costs
def profit_amount : ℝ := total_cost * profit_percent
def selling_price : ℝ := total_cost + profit_amount

-- Lean statement to prove
theorem Ramu_selling_price :
  total_cost = 55000 →
  profit_amount = 6899.999999999999 →
  selling_price = 61900 :=
by
  intros
  sorry

end Ramu_selling_price_l218_218486


namespace gcd_polynomial_l218_218024

theorem gcd_polynomial (a : ℕ) (h : 270 ∣ a) : Nat.gcd (5 * a^3 + 3 * a^2 + 5 * a + 45) a = 45 :=
sorry

end gcd_polynomial_l218_218024


namespace sum_series_eq_260_l218_218206

theorem sum_series_eq_260 : (2 + 12 + 22 + 32 + 42) + (10 + 20 + 30 + 40 + 50) = 260 := by
  sorry

end sum_series_eq_260_l218_218206


namespace find_y_when_x_4_l218_218046

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l218_218046


namespace integral_eq_l218_218224

noncomputable def integralValue : ℝ :=
  ∫ (x : ℝ) in -1..1, x^2 + real.sqrt (1 - x^2)

theorem integral_eq : integralValue = (2 / 3) + (real.pi / 2) := sorry

end integral_eq_l218_218224


namespace non_binary_listeners_l218_218535

theorem non_binary_listeners (listen_total males_listen females_dont_listen non_binary_dont_listen dont_listen_total : ℕ) 
  (h_listen_total : listen_total = 250) 
  (h_males_listen : males_listen = 85) 
  (h_females_dont_listen : females_dont_listen = 95) 
  (h_non_binary_dont_listen : non_binary_dont_listen = 45) 
  (h_dont_listen_total : dont_listen_total = 230) : 
  (listen_total - males_listen - (dont_listen_total - females_dont_listen - non_binary_dont_listen)) = 70 :=
by 
  -- Let nbl be the number of non-binary listeners
  let nbl := listen_total - males_listen - (dont_listen_total - females_dont_listen - non_binary_dont_listen)
  -- We need to show nbl = 70
  show nbl = 70
  sorry

end non_binary_listeners_l218_218535


namespace competition_results_l218_218472

-- Participants and positions
inductive Participant : Type
| Oleg
| Olya
| Polya
| Pasha

-- Places in the competition (1st, 2nd, 3rd, 4th)
def Place := Fin 4

-- Statements made by the children
def Olya_statement1 : Prop := ∀ p, p % 2 = 1 -> p = Participant.Oleg ∨ p = Participant.Pasha
def Oleg_statement1 : Prop := ∃ p1 p2: Place, p1 < p2 ∧ (p1 = p2 + 1)
def Pasha_statement1 : Prop := ∀ p, p % 2 = 1 -> (p = Place 1 ∨ p = Place 3)

-- Truthfulness of the statements
def only_one_truthful (Olya_true : Prop) (Oleg_true : Prop) (Pasha_true : Prop) :=
  (Olya_true ∧ ¬ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ ¬ Oleg_true ∧ Pasha_true)

-- The actual positions
def positions : Participant → Place
| Participant.Oleg  := 0
| Participant.Olya  := 1
| Participant.Polya := 2
| Participant.Pasha := 3

-- The Lean statement to prove
theorem competition_results :
  ((Oleg_statement1 ↔ positions Participant.Oleg = 0) ∧ 
  (Olya_statement1 ↔ positions Participant.Olya = 1) ∧ 
  (Pasha_statement1 ↔ positions Participant.Pasha = 3)) ∧ 
  only_one_truthful (positions Participant.Oleg = 0) 
                    (positions Participant.Olya = 0) 
                    (positions Participant.Pasha = 0) ∧
  positions Participant.Oleg = 0 ∧ 
  positions Participant.Olya = 1 ∧
  positions Participant.Polya = 2 ∧
  positions Participant.Pasha = 3 := 
sorry

end competition_results_l218_218472


namespace product_of_primes_l218_218089

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l218_218089


namespace sphere_center_plane_intersection_l218_218822

theorem sphere_center_plane_intersection
  (k a b c p q r : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (point_on_plane : (k * a, k * b, k * c))
  (A : (2 * p, 0, 0))
  (B : (0, 2 * q, 0))
  (C : (0, 0, 2 * r)) :
  (ka / p) + (kb / q) + (kc / r) = 2 :=
sorry

end sphere_center_plane_intersection_l218_218822


namespace emails_in_the_morning_l218_218814

variable (afternoon_emails : ℕ)
variable (morning_emails : ℕ)
variable (evening_emails : ℕ)
variable (additional_morning_emails : ℕ)

def emails_received := 
  afternoon_emails = 3 ∧ 
  evening_emails = 16 ∧ 
  additional_morning_emails = 2 ∧ 
  morning_emails = afternoon_emails + additional_morning_emails

theorem emails_in_the_morning (h : emails_received afternoon_emails morning_emails evening_emails additional_morning_emails) : 
  morning_emails = 5 :=
by
  rw [emails_received] at h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h5] at h6
  exact h6

end emails_in_the_morning_l218_218814


namespace remainder_when_divided_by_9_l218_218917

theorem remainder_when_divided_by_9 (x : ℕ) (h : 4 * x % 9 = 2) : x % 9 = 5 :=
by sorry

end remainder_when_divided_by_9_l218_218917


namespace initial_percentage_of_water_l218_218950

variable (V : ℝ) (W : ℝ) (P : ℝ)

theorem initial_percentage_of_water 
  (h1 : V = 120) 
  (h2 : W = 8)
  (h3 : (V + W) * 0.25 = ((P / 100) * V) + W) : 
  P = 20 :=
by
  sorry

end initial_percentage_of_water_l218_218950


namespace parabola_equation_l218_218305

theorem parabola_equation (h_vertex : true) (h_directrix : ∀ x y : ℝ, y^2 = -8*x ↔ (vertex_at_origin x y ∧ directrix_at_2 x y)) : 
  ∃ c : ℝ, c = 8 := 
begin
  sorry
end

def vertex_at_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0
def directrix_at_2 (x y : ℝ) : Prop := x = 2

end parabola_equation_l218_218305


namespace product_of_primes_l218_218106

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l218_218106


namespace union_M_N_l218_218323

def M (x : ℝ) : Prop := -1 ≤ x ∧ x < 8
def N (x : ℝ) : Prop := x > 4

theorem union_M_N : {x | M x} ∪ {x | N x} = {x | -1 ≤ x} :=
by
  simp [Set.ext_iff, Set.mem_union, M, N]
  split
  · intro h
    cases h
    · linarith
    · linarith
  · intro h
    by_cases h' : x < 8
    · left; exact ⟨h, h'⟩
    · right, linarith

end union_M_N_l218_218323


namespace find_a11_l218_218357

variable (a : ℕ → ℝ)

axiom geometric_seq (a : ℕ → ℝ) (r : ℝ) : ∀ n, a (n + 1) = a n * r

variable (r : ℝ)
variable (h3 : a 3 = 4)
variable (h7 : a 7 = 12)

theorem find_a11 : a 11 = 36 := by
  sorry

end find_a11_l218_218357


namespace find_g7_l218_218032

-- Given the required functional equation and specific value g(6) = 7
theorem find_g7 (g : ℝ → ℝ) (H1 : ∀ x y : ℝ, g (x + y) = g x + g y) (H2 : g 6 = 7) : g 7 = 49 / 6 := by
  sorry

end find_g7_l218_218032


namespace pastor_prayer_ratio_l218_218004

theorem pastor_prayer_ratio 
  (R : ℚ) 
  (paul_prays_per_day : ℚ := 20)
  (paul_sunday_times : ℚ := 2 * paul_prays_per_day)
  (paul_total : ℚ := 6 * paul_prays_per_day + paul_sunday_times)
  (bruce_ratio : ℚ := R)
  (bruce_prays_per_day : ℚ := bruce_ratio * paul_prays_per_day)
  (bruce_sunday_times : ℚ := 2 * paul_sunday_times)
  (bruce_total : ℚ := 6 * bruce_prays_per_day + bruce_sunday_times)
  (condition : paul_total = bruce_total + 20) :
  R = 1/2 :=
sorry

end pastor_prayer_ratio_l218_218004


namespace number_of_valid_subsets_l218_218530

-- Define the initial set S
def S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the predicate for a valid subset T
def valid_subset (T : Finset ℕ) : Prop :=
  T.card = 4 ∧ T ⊆ S ∧ ∀ {a b}, a ∈ T → b ∈ T → abs (a - b) ≠ 1

-- The theorem statement
theorem number_of_valid_subsets : 
  (Finset.univ.filter valid_subset).card = 35 :=
sorry

end number_of_valid_subsets_l218_218530


namespace choose_starters_1980_l218_218002

open Finset

noncomputable def num_ways_to_choose_starters (total_players : ℕ) (quadruplets : Finset ℕ) (starter_count : ℕ) (quadruplet_inclusion : ℕ) : ℕ :=
  if quadruplets.card = 4 ∧ quadruplet_inclusion = 3 ∧ starter_count = 7 ∧ total_players = 16 then
    (quadruplets.card.choose quadruplet_inclusion) * ((total_players - quadruplets.card).choose (starter_count - quadruplet_inclusion))
  else 0

theorem choose_starters_1980 : num_ways_to_choose_starters 16 (Finset.range 4) 7 3 = 1980 := by
  sorry

end choose_starters_1980_l218_218002


namespace ratio_slices_l218_218984

theorem ratio_slices (total_slices : ℕ) (calories_per_slice : ℕ) (total_calories_eaten : ℕ) :
  total_slices = 8 →
  calories_per_slice = 300 →
  total_calories_eaten = 1200 →
  (total_calories_eaten / calories_per_slice) = (1 / 2) * total_slices :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end ratio_slices_l218_218984


namespace largest_possible_perimeter_l218_218624

theorem largest_possible_perimeter (x : ℕ) (h1 : 7 + 9 > x) (h2 : 7 + x > 9) (h3 : 9 + x > 7) : 
  let upper_bound := 16 in
  let range_x := { n : ℕ // n ≥ 3 ∧ n < upper_bound } in
  let largest_side := 15 in
  let perimeter := 7 + 9 + largest_side in
  perimeter = 31 := 
by
  have h : largest_side = 15 := sorry
  exact h

end largest_possible_perimeter_l218_218624


namespace root_polynomial_has_given_roots_l218_218690

noncomputable def omega : ℂ := complex.exp (complex.I * (π / 5))

theorem root_polynomial_has_given_roots :
  polynomial.has_root (x^4 - x^3 + x^2 - x + 1) omega ∧
  polynomial.has_root (x^4 - x^3 + x^2 - x + 1) (omega^3) ∧
  polynomial.has_root (x^4 - x^3 + x^2 - x + 1) (omega^7) ∧
  polynomial.has_root (x^4 - x^3 + x^2 - x + 1) (omega^9) := sorry

end root_polynomial_has_given_roots_l218_218690


namespace total_buttons_correct_l218_218868

def typeA_buttons := 3
def typeB_buttons := 5
def typeC_buttons := 6
def typeD_buttons := 4

def order_typeA := 200
def order_typeB := 300
def order_typeC := 150
def order_typeD := 250

def discount := 0.10

noncomputable def total_buttons : ℕ :=
  let typeA_total := order_typeA * typeA_buttons
  let typeB_total := order_typeB * typeB_buttons
  let typeC_total := order_typeC * typeC_buttons
  let typeD_total := order_typeD * typeD_buttons
  let typeB_discount := if order_typeB > 200 then typeB_total * discount else 0
  let typeD_discount := if order_typeD > 200 then typeD_total * discount else 0
  typeA_total + (typeB_total - typeB_discount) + typeC_total + (typeD_total - typeD_discount)

theorem total_buttons_correct : total_buttons = 3750 :=
by
  sorry

end total_buttons_correct_l218_218868


namespace sandwiches_cost_l218_218561

theorem sandwiches_cost (sandwiches sodas : ℝ) 
  (cost_sandwich : ℝ := 2.44)
  (cost_soda : ℝ := 0.87)
  (num_sodas : ℕ := 4)
  (total_cost : ℝ := 8.36)
  (total_soda_cost : ℝ := cost_soda * num_sodas)
  (total_sandwich_cost : ℝ := total_cost - total_soda_cost):
  sandwiches = (total_sandwich_cost / cost_sandwich) → sandwiches = 2 := by 
  sorry

end sandwiches_cost_l218_218561


namespace increasing_interval_of_f_l218_218034

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_of_f :
  ∀ x, x > 2 → ∀ y, y > x → f x < f y :=
sorry

end increasing_interval_of_f_l218_218034


namespace ratio_of_inquisitive_tourist_l218_218642

theorem ratio_of_inquisitive_tourist (questions_per_tourist : ℕ)
                                     (num_group1 : ℕ) (num_group2 : ℕ) (num_group3 : ℕ) (num_group4 : ℕ)
                                     (total_questions : ℕ) 
                                     (inquisitive_tourist_questions : ℕ) :
  questions_per_tourist = 2 ∧ 
  num_group1 = 6 ∧ 
  num_group2 = 11 ∧ 
  num_group3 = 8 ∧ 
  num_group4 = 7 ∧ 
  total_questions = 68 ∧ 
  inquisitive_tourist_questions = (total_questions - (num_group1 * questions_per_tourist + num_group2 * questions_per_tourist +
                                                        (num_group3 - 1) * questions_per_tourist + num_group4 * questions_per_tourist)) →
  (inquisitive_tourist_questions : ℕ) / questions_per_tourist = 3 :=
by sorry

end ratio_of_inquisitive_tourist_l218_218642


namespace product_of_primes_is_66_l218_218124

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l218_218124


namespace photographers_can_photograph_each_other_l218_218808

structure Photographer :=
  (id : ℕ)
  (visible_to : ℕ → Prop)

def arrangement : List Photographer :=
  [{ id := 1, visible_to := λ x, x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 6 },
   { id := 2, visible_to := λ x, x = 1 ∨ x = 3 ∨ x = 4 ∨ x = 6 },
   { id := 3, visible_to := λ x, x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 5 },
   { id := 4, visible_to := λ x, x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 6 },
   { id := 5, visible_to := λ x, x = 1 ∨ x = 3 ∨ x = 4 ∨ x = 6 },
   { id := 6, visible_to := λ x, x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 5 }]

def no_crosses (p1 p2 : Photographer) : Prop :=
  ∀ (x y : Photographer), x ∈ arrangement → y ∈ arrangement →
  p1.visible_to x.id → p2.visible_to y.id → 
  (x.id = p1.id ∨ x.id = p2.id ∨ y.id = p1.id ∨ y.id = p2.id)

theorem photographers_can_photograph_each_other :
  ∃ photographers : List Photographer,
    ( ∀ (p : Photographer) (hp : p ∈ photographers),
      (∃ n : ℕ, n = list.count (λ q, p.visible_to q.id) photographers) ∧
      n = 4) ∧
    ( ∀ (p q : Photographer) (hp : p ∈ photographers) (hq : q ∈ photographers),
      p.visible_to q.id → no_crosses p q) :=
by
  existsi arrangement
  simp [arrangement, no_crosses]
  sorry

end photographers_can_photograph_each_other_l218_218808


namespace larger_number_is_8_l218_218756

-- Define the conditions
def is_twice (x y : ℕ) : Prop := x = 2 * y
def product_is_40 (x y : ℕ) : Prop := x * y = 40
def sum_is_14 (x y : ℕ) : Prop := x + y = 14

-- The proof statement
theorem larger_number_is_8 (x y : ℕ) (h1 : is_twice x y) (h2 : product_is_40 x y) (h3 : sum_is_14 x y) : x = 8 :=
  sorry

end larger_number_is_8_l218_218756


namespace total_hours_charged_l218_218848

variable (K P M : ℕ)

def hours_charged_by_kate (K : ℕ) : ℕ := K
def hours_charged_by_pat (K P : ℕ) [h₁ : P = 2 * K] : ℕ := P
def hours_charged_by_mark (K M : ℕ) [h₂ : M = K + 100] [h₃ : P = 1 / 3 * M] : ℕ := M

theorem total_hours_charged (K P M : ℕ)
  (h₁ : P = 2 * K)
  (h₂ : M = K + 100)
  (h₃ : P = 1 / 3 * M) :
  K + P + M = 180 :=
by {
  sorry
}

end total_hours_charged_l218_218848


namespace number_of_rectangles_with_one_gray_cell_l218_218750

theorem number_of_rectangles_with_one_gray_cell 
    (num_gray_cells : Nat) 
    (num_blue_cells : Nat) 
    (num_red_cells : Nat) 
    (blue_rectangles_per_cell : Nat) 
    (red_rectangles_per_cell : Nat)
    (total_gray_cells_calc : num_gray_cells = 2 * 20)
    (num_gray_cells_definition : num_gray_cells = num_blue_cells + num_red_cells)
    (blue_rect_cond : blue_rectangles_per_cell = 4)
    (red_rect_cond : red_rectangles_per_cell = 8)
    (num_blue_cells_calc : num_blue_cells = 36)
    (num_red_cells_calc : num_red_cells = 4)
  : num_blue_cells * blue_rectangles_per_cell + num_red_cells * red_rectangles_per_cell = 176 := 
  by
  sorry

end number_of_rectangles_with_one_gray_cell_l218_218750


namespace sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218243

theorem sum_of_four_digit_numbers_formed_by_digits_1_to_5 :
  let S := {1, 2, 3, 4, 5}
  let four_digits_sum (n1 n2 n3 n4 : ℕ) :=
    1000 * n1 + 100 * n2 + 10 * n3 + n4
  (∀ a b c d ∈ S, a ≠ b → b ≠ c → c ≠ d → d ≠ a → a ≠ c → b ≠ d 
  → sum (four_digits_sum a b c d) = 399960) := sorry

end sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218243


namespace ryan_spanish_hours_l218_218665

theorem ryan_spanish_hours (S : ℕ) (h : 7 = S + 3) : S = 4 :=
sorry

end ryan_spanish_hours_l218_218665


namespace distinct_primes_p_q_r_l218_218023

theorem distinct_primes_p_q_r (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) (eqn : r * p^3 + p^2 + p = 2 * r * q^2 + q^2 + q) : p * q * r = 2014 :=
by
  sorry

end distinct_primes_p_q_r_l218_218023


namespace cost_of_milk_is_3_l218_218022

-- Definitions based on conditions
def total_groceries_cost : ℝ := 25
def cereal_cost_each : ℝ := 3.5
def cereal_boxes : ℕ := 2
def banana_cost_each : ℝ := 0.25
def banana_count : ℕ := 4
def apple_cost_each : ℝ := 0.5
def apple_count : ℕ := 4
def cookie_boxes : ℕ := 2

-- The cost of cookies per box is twice the cost of a gallon of milk
def cookies_cost_per_box (milk_cost : ℝ) : ℝ := 2 * milk_cost

-- The total known cost of cereal, bananas, and apples
def total_known_cost : ℝ := (cereal_cost_each * cereal_boxes) + (banana_cost_each * banana_count) + (apple_cost_each * apple_count)

-- The remaining cost for milk and cookies
def remaining_cost : ℝ := total_groceries_cost - total_known_cost

-- The total cost of cookies given the cost of a gallon of milk
def total_cookies_cost (milk_cost : ℝ) : ℝ := cookie_boxes * cookies_cost_per_box(milk_cost)

-- The total equation to be solved
def equation (milk_cost : ℝ) : Prop := milk_cost + total_cookies_cost(milk_cost) = remaining_cost

-- The theorem to be proven
theorem cost_of_milk_is_3 : ∃ M, equation M ∧ M = 3 :=
by
  -- This would be the proof, which we skip by adding sorry.
  sorry

end cost_of_milk_is_3_l218_218022


namespace general_formula_b_smallest_n_S_smallest_n_S_validity_l218_218704

-- Definition of the sequence a_n
def a : ℕ → ℕ
| 0       => 0  -- Not used, just to handle a_0
| 1       => 3
| (n + 2) => if n.succ.even then 2 * a (n + 1) else a (n + 1) - 1

-- Definition of the sequence b_n
def b (n : ℕ) : ℕ :=
a (2 * n) + a (2 * n - 1)

-- Definition of the partial sum S_n
def S (n : ℕ) : ℕ :=
(n + 1).sum a

-- The main statements to prove:
theorem general_formula_b (n : ℕ) : b n = 2^n + 3 :=
sorry

theorem smallest_n_S (n : ℕ) : n ≥ 20 → S n > 2023 :=
sorry

theorem smallest_n_S_validity (n : ℕ) : 20 = n → S n > 2023 :=
sorry

end general_formula_b_smallest_n_S_smallest_n_S_validity_l218_218704


namespace sodas_total_l218_218610

def morning_sodas : ℕ := 77
def afternoon_sodas : ℕ := 19
def total_sodas : ℕ := morning_sodas + afternoon_sodas

theorem sodas_total :
  total_sodas = 96 :=
by
  sorry

end sodas_total_l218_218610


namespace probability_correct_l218_218629

def problem_condition := 
  ∃ total_questions science_questions humanities_questions (first_drawn_science: Prop),
  total_questions = 5 ∧
  science_questions = 3 ∧ 
  humanities_questions = 2 ∧ 
  first_drawn_science =
  (science_questions > 0)

def probability_of_drawing_science_given_first_is_science (total_questions science_questions humanities_questions : ℕ) (first_drawn_science) : ℚ :=
  (if first_drawn_science ∧ science_questions > 0 then 2/4 else 0)

theorem probability_correct:
  ∀ total_questions science_questions humanities_questions (first_drawn_science: Prop),
  total_questions = 5 →
  science_questions = 3 → 
  humanities_questions = 2 → 
  first_drawn_science →
  probability_of_drawing_science_given_first_is_science total_questions science_questions humanities_questions first_drawn_science = 1 / 2 :=
by
  intros
  simp [probability_of_drawing_science_given_first_is_science]
  sorry

end probability_correct_l218_218629


namespace simplify_trig_expression_l218_218864

theorem simplify_trig_expression : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := 
sorry

end simplify_trig_expression_l218_218864


namespace f_value_at_10_l218_218296

variable (f : ℝ → ℝ)

-- Condition 1: y = f(x-1) is an odd function
def odd_1_to_f := ∀ x : ℝ, f(-(x - 1)) = -f(x - 1)

-- Condition 2: y = f(x+1) is an even function
def even_1_to_f := ∀ x : ℝ, f(-(x + 1)) = f(x + 1)

-- Condition 3: f(x) = 2^x for 0 ≤ x < 1
def f_condition := ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f(x) = 2^x

-- Formal statement to be proven: f(10) = 1
theorem f_value_at_10 (H1 : odd_1_to_f f) (H2 : even_1_to_f f) (H3 : f_condition f) : f 10 = 1 :=
sorry

end f_value_at_10_l218_218296


namespace sum_of_coeffs_and_discontinuity_l218_218879

theorem sum_of_coeffs_and_discontinuity :
  ∃ A B C D : ℝ, 
    (∀ x : ℝ, x ≠ 1 → (x^3 - 4*x^2 - x + 6) / (x - 1) = A*x^2 + B*x + C) ∧
    D = 1 ∧
    A + B + C + D = -7 :=
begin
  sorry
end

end sum_of_coeffs_and_discontinuity_l218_218879


namespace sum_of_all_four_digit_numbers_l218_218256

def digits : List ℕ := [1, 2, 3, 4, 5]

noncomputable def four_digit_numbers := 
  (Digits.permutations digits).filter (λ l => l.length = 4)

noncomputable def sum_of_numbers (nums : List (List ℕ)) : ℕ :=
  nums.foldl (λ acc num => acc + (num.foldl (λ acc' digit => acc' * 10 + digit) 0)) 0

theorem sum_of_all_four_digit_numbers :
  sum_of_numbers four_digit_numbers = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_l218_218256


namespace decreasing_interval_g_l218_218547

noncomputable def f (x : ℝ) : ℝ := √3 * sin (x / 2) - cos (x / 2)

noncomputable def g (x : ℝ) : ℝ := -2 * cos (x / 2)

theorem decreasing_interval_g : 
  ∃ (a b : ℝ), (-π / 2 < a ∧ a < b ∧ b < -π / 4 ∧ ∀ x y, a < x ∧ x < y ∧ y < b → g y < g x) :=
sorry

end decreasing_interval_g_l218_218547


namespace line_tangent_to_circumcircle_l218_218683

open EuclideanGeometry

variables (A B C L D : Point ℝ) (Γ : Circle ℝ)

-- Conditions of the problem
variable [NonIsoscelesTriangle A B C] -- A, B, C form a non-isosceles triangle
variable [AngleBisector B L (∠ABC)] -- BL is the angle bisector of ∠ABC
variable [MedianFromVertexIntersectsCircumcircle B Γ D] -- The median from vertex B intersects the circumcircle Γ of triangle ABC at point D
variable [LineThroughCircumcenterParallelTo AC BDL Γ] -- A line through the circumcenter of triangle BDL is drawn parallel to AC

-- Theorem to prove
theorem line_tangent_to_circumcircle : TangentLineToCircumcircle l Γ :=
begin
  sorry, -- No proof required
end

end line_tangent_to_circumcircle_l218_218683


namespace sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218242

theorem sum_of_four_digit_numbers_formed_by_digits_1_to_5 :
  let S := {1, 2, 3, 4, 5}
  let four_digits_sum (n1 n2 n3 n4 : ℕ) :=
    1000 * n1 + 100 * n2 + 10 * n3 + n4
  (∀ a b c d ∈ S, a ≠ b → b ≠ c → c ≠ d → d ≠ a → a ≠ c → b ≠ d 
  → sum (four_digits_sum a b c d) = 399960) := sorry

end sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218242


namespace incorrect_conditionB_l218_218006

noncomputable def Point (P : Type) [inner_product_space ℝ P]

variables {A B C P : Point}

-- Conditions
def conditionA : Prop := (P - A) + (P - B) + (P - C) = 0
def conditionB : Prop := inner (P - A) ((P - A) - (P - B)) = inner (P - C) ((P - A) - (P - B))
def conditionC : Prop := dist P A = dist P B ∧ dist P A = dist P C
def conditionD : Prop := inner (P - A) (P - B) = inner (P - B) (P - C) ∧ inner (P - B) (P - C) = inner (P - C) (P - A)

-- Goal
theorem incorrect_conditionB (hA : conditionA) (hC : conditionC) (hD : conditionD) : ¬ conditionB :=
sorry

end incorrect_conditionB_l218_218006


namespace triathlon_ratio_l218_218655

/--
Dave participated in a triathlon. He spent 65 minutes in total on the event. 
He walked for 15 minutes at a speed of 5 km/h, jogged for 25 minutes at a speed of 8 km/h, 
and cycled the rest of the time at a speed of 16 km/h. 
Prove that the ratio of the total distance he covered while jogging and cycling 
to the distance he walked is 8:1.
-/
theorem triathlon_ratio :
  ∀ (total_time min walk_time min jog_time min : ℕ) (walk_speed jog_speed cycle_speed : ℕ),
    total_time = 65 →
    walk_time = 15 →
    jog_time = 25 →
    walk_speed = 5 →
    jog_speed = 8 →
    cycle_speed = 16 →
    let walk_dist := (walk_speed * walk_time) / 60,
        jog_dist := (jog_speed * jog_time) / 60,
        cycle_time := total_time - (walk_time + jog_time),
        cycle_dist := (cycle_speed * cycle_time) / 60,
        total_jog_cycle_dist := jog_dist + cycle_dist in
    (total_jog_cycle_dist / walk_dist : ℚ) = 8 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end triathlon_ratio_l218_218655


namespace largest_prime_factor_f9_over_f3_is_37_l218_218934

def f (n : ℕ) : ℕ := (3^n + 1) / 2

theorem largest_prime_factor_f9_over_f3_is_37 : 
  let f9 := f 9,
      f3 := f 3,
      ratio := f9 / f3 in
  nat.primeFactors ratio = [19, 37] ∧ 37 = List.maximum (nat.primeFactors ratio) :=
by {
  sorry
}

end largest_prime_factor_f9_over_f3_is_37_l218_218934


namespace piles_to_single_pile_l218_218435

-- Define the condition similar_sizes
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

-- Define the inductive step of combining stones
def combine_stones (piles : List ℕ) : List ℕ :=
  if ∃ x y, x ∈ piles ∧ y ∈ piles ∧ similar_sizes x y then
    let ⟨x, hx, y, hy, hsim⟩ := Classical.some_spec (Classical.some_spec_exists _)
    List.cons (x + y) (List.erase (List.erase piles x) y)
  else
    piles

-- Prove that a collection of piles can be reduced to a single pile of size n
theorem piles_to_single_pile (piles : List ℕ) (h : ∀ x ∈ piles, x = 1) : 
  ∃ p, list.length (Iterator.iterate combine_stones piles.count) 1 = 1 := by
  sorry

end piles_to_single_pile_l218_218435


namespace minimal_sum_iff_center_l218_218379

noncomputable def minimal_sum (P : Fin 9 → PPoint) (Q : PPoint) : Prop :=
  let r := distance (P 0) (circle_center (P ∘ Fin.succ)) in
  (∑ i : Fin 8, (distance (P i) (P (i + 1)))^2) = 8 * (2 - sqrt 2) * r^2

theorem minimal_sum_iff_center (P : Fin 9 → PPoint) (Q : PPoint) :
  (∀ i : Fin 8, angle (P (i - 1)) Q (P i) = 45) →
  (minimal_sum P Q ↔ Q = circle_center (P ∘ Fin.succ)) :=
sorry

end minimal_sum_iff_center_l218_218379


namespace problem_correct_answer_l218_218140

theorem problem_correct_answer (x y : ℕ) (h1 : y > 3) (h2 : x^2 + y^4 = 2 * ((x - 6)^2 + (y + 1)^2)) : x^2 + y^4 = 1994 :=
  sorry

end problem_correct_answer_l218_218140


namespace tangent_line_passing_through_point_l218_218725

open Real

def is_tangent (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ x₁ y₁, l x₁ y₁ ∧ C x₁ y₁ ∧ ∀ x₂ y₂, x₂ ≠ x₁ → y₂ ≠ y₁ → ¬ C x₂ y₂

theorem tangent_line_passing_through_point :
  let circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 8*x + 6*y + 21 = 0
  let line_eq1 : ℝ → ℝ → Prop := λ x y, 3*x + 4*y - 10 = 0
  let line_eq2 : ℝ → ℝ → Prop := λ x y, 4*x + 3*y + 3 = 0
  (is_tangent line_eq1 circle_eq ∨ is_tangent line_eq2 circle_eq) ∧
  (line_eq1 (-6) 7 ∨ line_eq2 (-6) 7) ∧
  (∃ center_x center_y radius, center_x = 4 ∧ center_y = -3 ∧ radius = 2) :=
by
  sorry

end tangent_line_passing_through_point_l218_218725


namespace meet_to_determine_initial_region_l218_218599

-- Definitions of the problem conditions --
def closed_non_self_intersecting_curve (C : Set Point) : Prop := 
  -- Definition expressing that C is a closed curve with no self-intersections
  sorry

def divides_plane (C : Set Point) : Prop :=
  -- Definition expressing that C divides the plane into an interior and an exterior region
  sorry

def arbitrary_paths (P1 P2: Path) : Prop := 
  -- Definition expressing arbitrary paths for two people
  sorry

noncomputable def can_determine_initial_region (C : Set Point) (P1 P2 : Path) (k1 k2 : Int) : Bool :=
  if (k1 + k2) % 2 == 0 then true else false

-- Lean 4 statement of the proof problem
theorem meet_to_determine_initial_region (C : Set Point) (P1 P2 : Path) (k1 k2 : Int) (h1 : closed_non_self_intersecting_curve C) (h2 : divides_plane C) (h3 : arbitrary_paths P1 P2) : 
  can_determine_initial_region C P1 P2 k1 k2 = true :=
begin
  -- Proof goes here
  sorry
end

end meet_to_determine_initial_region_l218_218599


namespace product_of_smallest_primes_l218_218118

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218118


namespace integral_cos_3x_pi_six_to_pi_two_l218_218637

noncomputable def integral_cos_3x (a b : ℝ) : ℝ :=
  ∫ x in a..b, real.cos (3 * x)

theorem integral_cos_3x_pi_six_to_pi_two :
  integral_cos_3x (real.pi / 6) (real.pi / 2) = -2 / 3 :=
by
  sorry

end integral_cos_3x_pi_six_to_pi_two_l218_218637


namespace probability_sum_30_l218_218602

def first_die_faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
def second_die_faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

theorem probability_sum_30 : 
  let valid_pairs := [(11, 19), (12, 18), (13, 17), (14, 16), (15, 15), (16, 14), (17, 13), (18, 12), (19, 11)] in
  let total_outcomes := 20 * 20 in
  let successful_outcomes := valid_pairs.countp (λ (p : ℕ × ℕ), p.1 ∈ first_die_faces ∧ p.2 ∈ second_die_faces) in
  (successful_outcomes : ℚ) / total_outcomes = 9 / 400 :=
by sorry

end probability_sum_30_l218_218602


namespace number_of_workers_l218_218927

-- Definitions for conditions
def initial_contribution (W C : ℕ) : Prop := W * C = 300000
def additional_contribution (W C : ℕ) : Prop := W * (C + 50) = 350000

-- Proof statement
theorem number_of_workers (W C : ℕ) (h1 : initial_contribution W C) (h2 : additional_contribution W C) : W = 1000 :=
by
  sorry

end number_of_workers_l218_218927


namespace hitting_next_shot_given_first_l218_218812

variables {A B : Prop}
variable (P : Prop → ℚ)

def student_first_shot_probability := P A = 9 / 10
def consecutive_shots_probability := P (A ∧ B) = 1 / 2

theorem hitting_next_shot_given_first 
    (h1 : student_first_shot_probability P)
    (h2 : consecutive_shots_probability P) :
    (P (A ∧ B) / P A) = 5 / 9 :=
by
  sorry

end hitting_next_shot_given_first_l218_218812


namespace geometric_mean_of_geo_sequence_l218_218483

variables (b : ℕ → ℝ) (b1 q : ℝ) (k p : ℕ)

-- Condition 1: Geometric progression definition
def is_geometric_progression (b : ℕ → ℝ) (b1 q : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b n = b1 * q^(n - 1)

-- Theorem statement: Any member of a positive geometric progression, starting from the second,
-- is equal to the geometric mean of any two members that are equidistant from it.
theorem geometric_mean_of_geo_sequence 
  (hb : is_geometric_progression b b1 q)
  (hk : k ≥ 1) (hp : p ≤ k) :
  b k = real.sqrt (b (k - p) * b (k + p)) :=
sorry

end geometric_mean_of_geo_sequence_l218_218483


namespace plane_equation_through_point_parallel_l218_218674

-- Definition of a plane equation
def plane (A B C D : ℤ) (x y z : ℝ) : ℝ := A * x + B * y + C * z + D

-- The given point and conditions
def point := (2 : ℝ, -3 : ℝ, 5 : ℝ)
def normal_vector := (3, -2, 4)

-- The statement we need to prove
theorem plane_equation_through_point_parallel
  (A B C D : ℤ)
  (h_normal: A = 3 ∧ B = -2 ∧ C = 4)
  (h_point: plane A B C D 2 (-3) 5 = 0) :
  ∃ D, plane A B C D 2 (-3) 5 = 0 ∧ A > 0 ∧ Int.gcd (Int.natAbs A) (Int.natAbs B) (Int.natAbs C) (Int.natAbs D) = 1 :=
  sorry

end plane_equation_through_point_parallel_l218_218674


namespace largest_possible_perimeter_l218_218621

noncomputable def max_perimeter_triangle : ℤ :=
  let a : ℤ := 7
  let b : ℤ := 9
  let x : ℤ := 15
  a + b + x

theorem largest_possible_perimeter (x : ℤ) (h1 : 7 + 9 > x) (h2 : 7 + x > 9) (h3 : 9 + x > 7) : max_perimeter_triangle = 31 := by
  sorry

end largest_possible_perimeter_l218_218621


namespace projection_matrix_correct_l218_218677

def Q := ![
    ![9/26, 3/26, -12/26],
    ![3/26, 1/26, -4/26],
    ![-12/26, -4/26, 16/26]
    ]

def vec3_proj (w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := (3, 1, -4)
  let scale := (w.1 * a.1 + w.2 * a.2 + w.3 * a.3) / (a.1 * a.1 + a.2 * a.2 + a.3 * a.3)
  (scale * a.1, scale * a.2, scale * a.3)

theorem projection_matrix_correct (w : ℝ × ℝ × ℝ) : 
  Q.mul_vec w = vec3_proj w := 
sorry

end projection_matrix_correct_l218_218677


namespace eq_satisfied_in_entire_space_l218_218919

theorem eq_satisfied_in_entire_space (x y z : ℝ) : 
  (x + y + z)^2 = x^2 + y^2 + z^2 ↔ xy + xz + yz = 0 :=
by
  sorry

end eq_satisfied_in_entire_space_l218_218919


namespace range_of_k_intersection_l218_218342

theorem range_of_k_intersection (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k^2 - 1) * x1^2 + 4 * k * x1 + 10 = 0 ∧ (k^2 - 1) * x2^2 + 4 * k * x2 + 10 = 0) ↔ (-1 < k ∧ k < 1) :=
by
  sorry

end range_of_k_intersection_l218_218342


namespace number_of_primitive_pairs_l218_218272

def splendid (m n : ℕ) (f : ℝ → ℝ) (S : set ℝ) : Prop :=
  ∀ (x y : ℝ) (hx : x ∈ S) (hy : y ∈ S), (iter f m x) + (iter f n y) = x + y

def primitive (m n : ℕ) (f : ℝ → ℝ) (S : set ℝ) : Prop :=
  splendid m n f S ∧ 
  ∀ (a b : ℕ), a ≤ m → b ≤ n → (a ≠ m ∨ b ≠ n) → a ≠ b → ¬ splendid a b f S

noncomputable def count_primitive_pairs : ℕ :=
  ∑ n in finset.range 5000, 2

theorem number_of_primitive_pairs : count_primitive_pairs = 9998 :=
sorry

end number_of_primitive_pairs_l218_218272


namespace sum_fractions_geq_n_div_kn_minus_1_l218_218722

theorem sum_fractions_geq_n_div_kn_minus_1 (a : ℝ) (a_i : ℕ → ℝ) (n k : ℕ) 
  (h1 : ∀ i, i ∈ range n → (a_i i) * (a_i 0) ≥ 0)
  (h2 : ∑ i in range n, a_i i = a)
  (h3 : 1 < n)
  (h4 : 1 < k) :
  (∑ i in range n, a_i i / (k * a - a_i i)) ≥ n / (k * n - 1) := 
sorry

end sum_fractions_geq_n_div_kn_minus_1_l218_218722


namespace trig_identity_eq_zero_l218_218221

theorem trig_identity_eq_zero :
  sin (29 / 6 * Real.pi) + cos (-29 / 3 * Real.pi) + tan (-25 / 4 * Real.pi) = 0 :=
by
  sorry

end trig_identity_eq_zero_l218_218221


namespace tetrahedron_inequality_l218_218360

open Real

-- Define the necessary geometric entities and conditions
variables (A B C D : Point) (AB BC CA AD BD CD : ℝ)
variables (orthocenter_of_ABC : Point)

-- Conditions: 
-- Angle ∠BDC is a right angle
def angle_BDC_is_right (A B C D : Point) : Prop :=
  ∃ (BDC : Angle), is_right_angle BDC

-- The foot of the perpendicular from D to the plane ABC coincides with the orthocenter of triangle ABC
def foot_of_perpendicular_is_orthocenter (A B C D : Point) : Prop :=
  ∃ (foot : Point), foot = orthocenter_of_ABC ∧ is_perpendicular_with_plane D foot (Plane.mk A B C)

-- Main theorem statement
theorem tetrahedron_inequality (A B C D : Point)
  (h1 : angle_BDC_is_right A B C D)
  (h2 : foot_of_perpendicular_is_orthocenter A B C D) :
  (dist A B + dist B C + dist C A)^2 ≤ 6 * (dist A D^2 + dist B D^2 + dist C D^2) :=
begin
  sorry,
end

end tetrahedron_inequality_l218_218360


namespace equation_of_line_through_points_on_circle_l218_218883

open Real

noncomputable def midpoint_of_chord := (x_A x_B y_A y_B : ℝ) : ℝ × ℝ := ((x_A + x_B) / 2, (y_A + y_B) / 2)

theorem equation_of_line_through_points_on_circle {a : ℝ} (h : a < 3) :
  let l : Circle := { center := (-1, 2), radius := sqrt ((1+a) - (-4)) },
      A B : Point := some points_on_circle_intersected_with l, 
      C : Point := (-2, 3),
      AB : Line := line_through A B,
      midpoint_AB : Point := midpoint_of_chord A.1 B.1 A.2 B.2
  in
  midpoint_AB = C →
  equation_line AB = "x - y + 5 = 0" :=
by
  sorry

end equation_of_line_through_points_on_circle_l218_218883


namespace rectangular_to_polar_correct_l218_218654

open Real

theorem rectangular_to_polar_correct :
  let x := 8
  let y := 2 * sqrt 3
  let r := sqrt (x^2 + y^2)
  let theta := arctan (y / x)
  (r > 0) ∧ (0 ≤ theta ∧ theta < 2 * pi) ∧ ((x, y) = (8, 2 * sqrt 3)) →
  (r, theta) = (2 * sqrt 19, arctan(sqrt 3 / 4)) :=
by
  intros x y r theta cond
  sorry

end rectangular_to_polar_correct_l218_218654


namespace probability_top_card_hearts_or_diamonds_l218_218615

theorem probability_top_card_hearts_or_diamonds:
  let ranks := 13
  let suits := 4
  let total_cards := 52
  let red_suit_count := 2
  let total_red_cards := 26
  let top_card_probability := total_red_cards / total_cards
    in top_card_probability = 1 / 2 := by
  sorry

end probability_top_card_hearts_or_diamonds_l218_218615


namespace piles_can_be_reduced_l218_218398

/-! 
  We define similar sizes as the difference between sizes being at most a factor of two.
  Given any number of piles of stones, we aim to prove that these piles can be combined 
  iteratively into one single pile.
-/

def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

theorem piles_can_be_reduced (n : ℕ) :
  ∃ pile : ℕ, (pile = n) ∧ (∀ piles : list ℕ, list.sum piles = n → 
    (∃ piles' : list ℕ, list.sum piles' = n ∧ list.length piles' = 1)) :=
by
  -- Placeholder for the proof.
  sorry

end piles_can_be_reduced_l218_218398


namespace find_ω_and_cos_alpha_plus_beta_l218_218734

-- Define the function f(x) with the given period
def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + Real.pi / 6)

-- Condition for the smallest positive period
def smallestPosPeriod (ω : ℝ) : Prop := ∀ x, f ω x = f ω (x + 10 * Real.pi)

-- Define the values of α and β in the given range
variables (α β : ℝ)

theorem find_ω_and_cos_alpha_plus_beta (hω_pos : ω > 0)
  (hperiod : smallestPosPeriod ω)
  (hα_range : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (hβ_range : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h1 : f ω (5 * α + 5 * Real.pi / 3) = -6 / 5)
  (h2 : f ω (5 * β - 5 * Real.pi / 6) = 16 / 17) :
  (ω = 1 / 5) ∧ (Real.cos (α + β) = -13 / 85) := sorry

end find_ω_and_cos_alpha_plus_beta_l218_218734


namespace probability_of_green_ball_is_157_over_495_l218_218494

-- Definitions of the number of balls in each container
def balls_in_container_I := (10, 2, 3) -- (red, green, blue)
def balls_in_container_II := (5, 4, 2) -- (red, green, blue)
def balls_in_container_III := (3, 5, 3) -- (red, green, blue)

-- Definition of random selection probability
def probability_of_selecting_container := (1 : ℚ) / 3

-- Calculating probability of selecting a green ball from a given container
def probability_of_green_ball_from_I : ℚ := 2 / (10 + 2 + 3)
def probability_of_green_ball_from_II : ℚ := 4 / (5 + 4 + 2)
def probability_of_green_ball_from_III : ℚ := 5 / (3 + 5 + 3)

-- Combined probability for each container
def combined_probability_I : ℚ := probability_of_selecting_container * probability_of_green_ball_from_I
def combined_probability_II : ℚ := probability_of_selecting_container * probability_of_green_ball_from_II
def combined_probability_III : ℚ := probability_of_selecting_container * probability_of_green_ball_from_III

-- Total probability of selecting a green ball
def total_probability_of_green_ball : ℚ := combined_probability_I + combined_probability_II + combined_probability_III

-- The statement to be proved in Lean 4
theorem probability_of_green_ball_is_157_over_495 : total_probability_of_green_ball = 157 / 495 :=
by
  sorry

end probability_of_green_ball_is_157_over_495_l218_218494


namespace trader_overall_loss_l218_218770

theorem trader_overall_loss:
  let SP1 := 325475 in
  let SP2 := 325475 in
  let gain_percentage_on_first_car := 0.10 in
  let loss_percentage_on_second_car := 0.10 in
  let CP1 := SP1 / (1 + gain_percentage_on_first_car) in
  let CP2 := SP2 / (1 - loss_percentage_on_second_car) in
  let TCP := CP1 + CP2 in
  let TSP := SP1 + SP2 in
  let Profit_or_Loss := TSP - TCP in
  let Profit_or_Loss_Percent := (Profit_or_Loss / TCP) * 100 in
  Profit_or_Loss_Percent = -1 :=
begin
  sorry
end

end trader_overall_loss_l218_218770


namespace prism_unique_triple_l218_218611

theorem prism_unique_triple :
  ∃! (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ b = 2000 ∧
                  (∃ b' c', b' = 2000 ∧ c' = 2000 ∧
                  (∃ k : ℚ, k = 1/2 ∧
                  (∃ x y z, x = a / 2 ∧ y = 1000 ∧ z = c / 2 ∧ a = 2000 ∧ c = 2000)))
/- The proof is omitted for this statement. -/
:= sorry

end prism_unique_triple_l218_218611


namespace find_m_eccentricity_l218_218502

theorem find_m_eccentricity :
  (∃ m : ℝ, (m > 0) ∧ (∃ c : ℝ, (c = 4 - m ∧ c = (1 / 2) * 2) ∨ (c = m - 4 ∧ c = (1 / 2) * 2)) ∧
  (m = 3 ∨ m = 16 / 3)) :=
sorry

end find_m_eccentricity_l218_218502


namespace conjugate_of_complex_number_l218_218280

theorem conjugate_of_complex_number :
  let z := (1 + Complex.I * Real.sqrt 3) / (Real.sqrt 3 + Complex.I)
  Complex.conj z = (Real.sqrt 3) / 2 - (1 / 2) * Complex.I :=
by
  sorry

end conjugate_of_complex_number_l218_218280


namespace platform_length_605_l218_218969

noncomputable def length_of_platform (speed_kmh : ℕ) (accel : ℚ) (t_platform : ℚ) (t_man : ℚ) (dist_man_from_platform : ℚ) : ℚ :=
  let speed_ms := (speed_kmh : ℚ) * 1000 / 3600
  let distance_man := speed_ms * t_man + 0.5 * accel * t_man^2
  let train_length := distance_man - dist_man_from_platform
  let distance_platform := speed_ms * t_platform + 0.5 * accel * t_platform^2
  distance_platform - train_length

theorem platform_length_605 :
  length_of_platform 54 0.5 40 20 5 = 605 := by
  sorry

end platform_length_605_l218_218969


namespace product_of_primes_l218_218104

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l218_218104


namespace derivative_of_f_l218_218736

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 * x - 2 :=
by
  intro x
  -- proof skipped
  sorry

end derivative_of_f_l218_218736


namespace probability_x_add_y_lt_4_in_square_l218_218171

noncomputable def square_area : ℝ := 3 * 3

noncomputable def triangle_area : ℝ := (1 / 2) * 2 * 2

noncomputable def region_area : ℝ := square_area - triangle_area

noncomputable def probability (A B : ℝ) : ℝ := A / B

theorem probability_x_add_y_lt_4_in_square :
  probability region_area square_area = 7 / 9 :=
by 
  sorry

end probability_x_add_y_lt_4_in_square_l218_218171


namespace sequence_periodicity_equiv_l218_218017

-- Definitions of the conditions and the question
def is_periodic_with_power_of_2 (a : ℕ → ℤ) : Prop :=
  ∃ k : ℕ, ∀ n : ℕ, a (n + 2^k) = a n

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 1 ∨ a n = -1

def polynomial_valued_sequence (a : ℕ → ℤ) (P : ℚ[X]) : Prop :=
  ∀ n : ℕ, a n = (-1) ^ (P.eval n).to_int

theorem sequence_periodicity_equiv (a : ℕ → ℤ) (P : ℚ[X]) :
  is_valid_sequence a →
  ((is_periodic_with_power_of_2 a) ↔ (polynomial_valued_sequence a P)) := 
sorry

end sequence_periodicity_equiv_l218_218017


namespace friends_playing_video_game_l218_218945

def total_lives : ℕ := 64
def lives_per_player : ℕ := 8

theorem friends_playing_video_game (num_friends : ℕ) :
  num_friends = total_lives / lives_per_player :=
sorry

end friends_playing_video_game_l218_218945


namespace relay_race_total_time_l218_218583

noncomputable def mary_time (susan_time : ℕ) : ℕ := 2 * susan_time
noncomputable def susan_time (jen_time : ℕ) : ℕ := jen_time + 10
def jen_time : ℕ := 30
noncomputable def tiffany_time (mary_time : ℕ) : ℕ := mary_time - 7

theorem relay_race_total_time :
  let mary_time := mary_time (susan_time jen_time)
  let susan_time := susan_time jen_time
  let tiffany_time := tiffany_time mary_time
  mary_time + susan_time + jen_time + tiffany_time = 223 := by
  sorry

end relay_race_total_time_l218_218583


namespace Amy_homework_time_l218_218149

def mathProblems : Nat := 18
def spellingProblems : Nat := 6
def problemsPerHour : Nat := 4
def totalProblems : Nat := mathProblems + spellingProblems
def totalHours : Nat := totalProblems / problemsPerHour

theorem Amy_homework_time :
  totalHours = 6 := by
  sorry

end Amy_homework_time_l218_218149


namespace cells_sequence_exists_l218_218618

theorem cells_sequence_exists :
  ∃ (a : Fin 10 → ℚ), 
    a 0 = 9 ∧
    a 8 = 5 ∧
    (∀ i : Fin 8, a i + a (i + 1) + a (i + 2) = 14) :=
sorry

end cells_sequence_exists_l218_218618


namespace nap_time_l218_218192

-- Definitions of given conditions
def flight_duration : ℕ := 680
def reading_time : ℕ := 120
def movie_time : ℕ := 240
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 70

def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Theorem statement
theorem nap_time : (flight_duration - total_activity_time) / 60 = 3 := by
  -- Here would go the proof steps verifying the equality
  sorry

end nap_time_l218_218192


namespace function_passes_fixed_point_l218_218312

theorem function_passes_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
    ∃ x y : ℝ, x = 1 ∧ y = 5 ∧ (4 + a^(x-1) = y) :=
by
  use 1
  use 5
  split
  exact rfl
  split
  exact rfl
  simp
  sorry

end function_passes_fixed_point_l218_218312


namespace finite_set_exists_l218_218012

theorem finite_set_exists (m : ℕ) (hm : m > 0) : 
  ∃ (S : finset (ℝ × ℝ)), ∀ A ∈ S, 
    (finset.filter (λ B, (B ≠ A) ∧ (dist A B = 1)) S).card = m :=
begin
  sorry
end

end finite_set_exists_l218_218012


namespace largest_log_solution_l218_218651

theorem largest_log_solution (x : ℝ) (h : log (5 * x^2) 5 + log (25 * x^3) 5 = -1) : (1 / x^6) = 125 := 
sorry

end largest_log_solution_l218_218651


namespace swimmer_speed_is_4_4_l218_218966

noncomputable def swimmer_speed_in_still_water (distance : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
(distance / time) + current_speed

theorem swimmer_speed_is_4_4 :
  swimmer_speed_in_still_water 7 2.5 3.684210526315789 = 4.4 :=
by
  -- This part would contain the proof to show that the calculated speed is 4.4
  sorry

end swimmer_speed_is_4_4_l218_218966


namespace like_terms_eq_l218_218757

theorem like_terms_eq : 
  ∀ (x y : ℕ), 
  (x + 2 * y = 3) → 
  (2 * x + y = 9) → 
  (x + y = 4) :=
by
  intros x y h1 h2
  sorry

end like_terms_eq_l218_218757


namespace interest_group_selections_l218_218688

-- Define the number of students and the number of interest groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem statement: The total number of different possible selections of interest groups is 81.
theorem interest_group_selections : num_groups ^ num_students = 81 := by
  sorry

end interest_group_selections_l218_218688


namespace find_y_when_x_4_l218_218045

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l218_218045


namespace tommy_first_house_price_l218_218065

theorem tommy_first_house_price (C : ℝ) (P : ℝ) (loan_rate : ℝ) (interest_rate : ℝ)
  (term : ℝ) (property_tax_rate : ℝ) (insurance_cost : ℝ) 
  (price_ratio : ℝ) (monthly_payment : ℝ) :
  C = 500000 ∧ price_ratio = 1.25 ∧ P * price_ratio = C ∧
  loan_rate = 0.75 ∧ interest_rate = 0.035 ∧ term = 15 ∧
  property_tax_rate = 0.015 ∧ insurance_cost = 7500 → 
  P = 400000 :=
by sorry

end tommy_first_house_price_l218_218065


namespace relay_race_total_time_l218_218582

noncomputable def mary_time (susan_time : ℕ) : ℕ := 2 * susan_time
noncomputable def susan_time (jen_time : ℕ) : ℕ := jen_time + 10
def jen_time : ℕ := 30
noncomputable def tiffany_time (mary_time : ℕ) : ℕ := mary_time - 7

theorem relay_race_total_time :
  let mary_time := mary_time (susan_time jen_time)
  let susan_time := susan_time jen_time
  let tiffany_time := tiffany_time mary_time
  mary_time + susan_time + jen_time + tiffany_time = 223 := by
  sorry

end relay_race_total_time_l218_218582


namespace operation_result_l218_218273

theorem operation_result (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : (-3 : ℚ) * (5 : ℚ) = (13 : ℚ) / (11 : ℚ) :=
by
  let op := λ (a b : ℚ), (a - 2 * b) / (2 * a - b)
  have h := op (-3) 5
  sorry

end operation_result_l218_218273


namespace find_d_for_quadratic_root_l218_218890

theorem find_d_for_quadratic_root :
  ∀ d : ℝ, has_quadratic_root d → d = 2 :=
by
  -- Ignore the proof as per instructions
  sorry

def has_quadratic_root (d : ℝ) : Prop :=
  ∀ x : ℝ,
    (x = (3 + Real.sqrt (d - 1)) / 2 ∨ x = (3 - Real.sqrt (d - 1)) / 2) →
    (2 * x) = 3 + Real.sqrt (9 - 4 * d) ∨ (2 * x) = 3 - Real.sqrt (9 - 4 * d)

end find_d_for_quadratic_root_l218_218890


namespace product_of_smallest_primes_l218_218114

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218114


namespace places_proven_l218_218461

-- Definitions based on the problem conditions
inductive Place
| first
| second
| third
| fourth

def is_boy : String -> Prop
| "Oleg" => True
| "Olya" => False
| "Polya" => False
| "Pasha" => False
| _ => False

def name_starts_with_O : String -> Prop
| n => (n.head! = 'O')

noncomputable def determine_places : Prop :=
  ∃ (olegs_place olyas_place polyas_place pashas_place : Place),
  -- Statements and truth conditions
  ∃ (truthful : String), truthful ∈ ["Oleg", "Olya", "Polya", "Pasha"] ∧ 
  ∀ (person : String), 
    (person ≠ truthful → ∀ (statement : Place -> Prop), ¬ statement (person_to_place person)) ∧
    (person = truthful → person_to_place person = Place.first) ∧
    (person = truthful → 
      match person with
        | "Olya" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → is_boy (place_to_person p)
        | "Oleg" => ∃ (p : Place), (person_to_place "Oleg" = p ∧ person_to_place "Olya" = succ_place p ∨ 
                                    person_to_place "Olya" = p ∧ person_to_place "Oleg" = succ_place p)
        | "Pasha" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → name_starts_with_O (place_to_person p)
        | _ => True
      end)

-- Helper functions to relate places to persons
def person_to_place : String -> Place
| "Oleg" => Place.first
| "Olya" => Place.second
| "Polya" => Place.third
| "Pasha" => Place.fourth
| _ => Place.first -- Default, shouldn't happen

def place_to_person : Place -> String
| Place.first => "Oleg"
| Place.second => "Olya"
| Place.third => "Polya"
| Place.fourth => "Pasha"

def succ_place : Place → Place
| Place.first => Place.second
| Place.second => Place.third
| Place.third => Place.fourth
| Place.fourth => Place.first -- No logical next in this context.

theorem places_proven : determine_places :=
by
  sorry

end places_proven_l218_218461


namespace sum_of_all_four_digit_numbers_l218_218257

def digits : List ℕ := [1, 2, 3, 4, 5]

noncomputable def four_digit_numbers := 
  (Digits.permutations digits).filter (λ l => l.length = 4)

noncomputable def sum_of_numbers (nums : List (List ℕ)) : ℕ :=
  nums.foldl (λ acc num => acc + (num.foldl (λ acc' digit => acc' * 10 + digit) 0)) 0

theorem sum_of_all_four_digit_numbers :
  sum_of_numbers four_digit_numbers = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_l218_218257


namespace find_salary_June_l218_218027

variable (J F M A May_s June_s : ℝ)
variable (h1 : J + F + M + A = 4 * 8000)
variable (h2 : F + M + A + May_s = 4 * 8450)
variable (h3 : May_s = 6500)
variable (h4 : M + A + May_s + June_s = 4 * 9000)
variable (h5 : June_s = 1.2 * May_s)

theorem find_salary_June : June_s = 7800 := by
  sorry

end find_salary_June_l218_218027


namespace erik_money_left_after_purchase_l218_218662

noncomputable def total_cost_before_discount (bread_cost orange_juice_cost eggs_cost chocolate_cost : ℝ) : ℝ := 
  3 * bread_cost + 3 * orange_juice_cost + 2 * eggs_cost + 5 * chocolate_cost

noncomputable def discount_amount (total_cost discount_rate : ℝ) : ℝ := 
  total_cost * discount_rate

noncomputable def total_cost_after_discount (total_cost discount : ℝ) : ℝ := 
  total_cost - discount

noncomputable def sales_tax_amount (discounted_total tax_rate : ℝ) : ℝ := 
  discounted_total * tax_rate

noncomputable def total_cost_after_discount_and_tax (discounted_total tax : ℝ) : ℝ := 
  discounted_total + tax

noncomputable def money_left (initial_amount total_cost : ℝ) : ℝ := 
  initial_amount - total_cost

theorem erik_money_left_after_purchase :
  let initial_amount : ℝ := 86
  let bread_cost : ℝ := 3
  let orange_juice_cost : ℝ := 6
  let eggs_cost : ℝ := 4
  let chocolate_cost : ℝ := 2
  let discount_rate : ℝ := 0.1
  let tax_rate : ℝ := 0.05
  let total_cost := total_cost_before_discount bread_cost orange_juice_cost eggs_cost chocolate_cost
  let discount := discount_amount total_cost discount_rate
  let discounted_total := total_cost_after_discount total_cost discount
  let tax := sales_tax_amount discounted_total tax_rate
  let rounded_tax := Real.round (tax * 100) / 100
  let final_cost := total_cost_after_discount_and_tax discounted_total rounded_tax
  money_left initial_amount final_cost = 43.47 :=
by
  simp only [initial_amount, bread_cost, orange_juice_cost, eggs_cost, chocolate_cost, discount_rate, tax_rate, total_cost_before_discount, discount_amount, total_cost_after_discount, sales_tax_amount, total_cost_after_discount_and_tax, money_left]
  have h1 : total_cost_before_discount 3 6 4 2 = 45 := by norm_num
  have h2 : discount_amount 45 0.1 = 4.5 := by norm_num
  have h3 : total_cost_after_discount 45 4.5 = 40.5 := by norm_num
  have h4 : sales_tax_amount 40.5 0.05 = 2.025 := by norm_num
  have h5 : Real.round (2.025 * 100) / 100 = 2.03 := by norm_num
  have h6 : total_cost_after_discount_and_tax 40.5 2.03 = 42.53 := by norm_num
  have h7 : money_left 86 42.53 = 43.47 := by norm_num
  exact h7

end erik_money_left_after_purchase_l218_218662


namespace solve_for_t_l218_218374

theorem solve_for_t (t : ℝ) (h1 : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220) 
  (h2 : 0 ≤ t) : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220 :=
by
  sorry

end solve_for_t_l218_218374


namespace pile_of_stones_l218_218412

def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

theorem pile_of_stones (n : ℕ) (f : ℕ → ℕ): (∀ i, 1 ≤ f i ∧ f i ≤ n) → 
  (∀ j k, similar_sizes (f j) (f k)) → True :=
by
  simp
  exact true.intro


end pile_of_stones_l218_218412


namespace expected_steps_l218_218841

noncomputable def E (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 5 / 4
  else if n = 3 then 25 / 16
  else 125 / 64

theorem expected_steps :
  let expected := 1 + (E 1 + E 2 + E 3 + E 4) / 4 in
  expected = 625 / 256 :=
by {
  -- Proof goes here
  sorry
}

end expected_steps_l218_218841


namespace savings_equal_in_820_weeks_l218_218858

-- Definitions for the conditions
def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

-- The statement we want to prove
theorem savings_equal_in_820_weeks : 
  ∃ (w : ℕ), (sara_initial_savings + w * sara_weekly_savings) = (w * jim_weekly_savings) ∧ w = 820 :=
by
  sorry

end savings_equal_in_820_weeks_l218_218858


namespace OD_expression_l218_218597

open Real

variables {O A B D : Point} {r θ u v : ℝ} 

-- Given conditions
def circle_centered_at_O : Circle O r := sorry
def radius_of_circle : r = 2 := by rfl
def circle_contains_A : A ∈ circle_centered_at_O := sorry
def segment_AB_tangent_at_A : Tangent A B circle_centered_at_O := sorry
def angle_AOB : Angle A O B = 2 * θ := sorry
def D_on_OA : D ∈ Line O A := sorry
def BD_bisects_∠ABO : Bisects D B (Angle A B O) := sorry

-- Definitions for u and v
def u_def : u = sin (2 * θ) := by rfl
def v_def : v = cos (2 * θ) := by rfl

theorem OD_expression : OD = 2 / (1 + u) :=
sorry

end OD_expression_l218_218597


namespace correct_goal_in_speech_correct_goal_of_political_speech_incorrect_impossible_goal_in_speech_correct_aspects_to_consider_in_goal_setting_l218_218612

-- Define what a goal is in the context of a speech
def goal_in_speech : Prop :=
  "A goal is the purpose of the speech, and what it hopes to accomplish."

-- Define what the goal of a political speech is
def goal_of_political_speech : Prop :=
  "The goal of a speech at a political gathering would be to inform the crowd
   about the political position of a candidate and persuade them to vote and 
   campaign for the candidate in question."

-- Define if a speech can allow the speaker to achieve whatever they want
def impossible_goal_in_speech (s : string) : Prop :=
  "A speech cannot possibly allow the speaker to achieve whatever they want."

-- Define the aspects to consider when setting the goal of a speech
def aspects_to_consider_in_goal_setting : Prop :=
  "Several aspects to consider when setting the goal of a speech"

-- The theorems to prove based on the above definitions
theorem correct_goal_in_speech : goal_in_speech = "B" := sorry

theorem correct_goal_of_political_speech : goal_of_political_speech = "A" := sorry

theorem incorrect_impossible_goal_in_speech (s : string) : ¬(impossible_goal_in_speech s = "D") := sorry

theorem correct_aspects_to_consider_in_goal_setting : aspects_to_consider_in_goal_setting = "C" := sorry

end correct_goal_in_speech_correct_goal_of_political_speech_incorrect_impossible_goal_in_speech_correct_aspects_to_consider_in_goal_setting_l218_218612


namespace problem_statement_l218_218761

def f (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem problem_statement : f (f (-1)) = 10 := by
  sorry

end problem_statement_l218_218761


namespace number_of_attendees_from_company_A_l218_218600

-- Define the number of attendees for companies A, B, C, and D.
variable (A : ℕ)

-- Conditions:
def company_B_attendees := 2 * A
def company_C_attendees := A + 10
def company_D_attendees := (A + 10) - 5

-- Total attendees registered.
def total_attendees_registered := A + company_B_attendees A + company_C_attendees A + company_D_attendees A + 20

-- Number of attendees from Company A calculation
theorem number_of_attendees_from_company_A:
  total_attendees_registered A = 185 → A = 30 :=
by
  intro h,
  sorry

end number_of_attendees_from_company_A_l218_218600


namespace similar_sizes_combination_possible_l218_218407

theorem similar_sizes_combination_possible 
    (similar : Nat → Nat → Prop := λ x y, x ≤ y ∧ y ≤ 2 * x)
    (combine_piles : List Nat → Nat ∃ combined : Nat, (∀ x y ∈ combined, similar x y) → True
    (piles : List Nat) : True :=
sorry

end similar_sizes_combination_possible_l218_218407


namespace water_heater_total_capacity_l218_218555

theorem water_heater_total_capacity (W_capacity : ℕ) (A_leak : ℕ) :
  W_capacity = 40 →
  A_leak = 5 →
  let C_capacity := W_capacity / 2 in
  let A_capacity := W_capacity * 3 / 2 in
  let W_filled := W_capacity * 3 / 4 in
  let C_filled := C_capacity * 3 / 4 in
  let A_filled_before_leak := A_capacity * 2 / 3 in
  let A_filled_after_leak := A_filled_before_leak - A_leak in
  W_filled + C_filled + A_filled_after_leak = 80 :=
by {
  -- Given conditions
  intros,
  -- Define intermediate values
  have C_capacity_def : C_capacity = W_capacity / 2 := rfl,
  have A_capacity_def : A_capacity = W_capacity * 3 / 2 := rfl,
  have W_filled_def : W_filled = W_capacity * 3 / 4 := rfl,
  have C_filled_def : C_filled = C_capacity * 3 / 4 := rfl,
  have A_filled_before_leak_def : A_filled_before_leak = A_capacity * 2 / 3 := rfl,
  have A_filled_after_leak_def : A_filled_after_leak = A_filled_before_leak - A_leak := rfl,
  -- Proof follows here using sorry to skip
  sorry
}

end water_heater_total_capacity_l218_218555


namespace area_of_PQRSTU_l218_218673

-- Definitions and conditions
def PQ := 5
def QR := 8
def PT := QR -- by condition (4)
def QS := 4
-- Recall that the polygon can be divided into triangle PQR and parallelogram PQST by condition (3).

-- Proof statement
theorem area_of_PQRSTU : 
  (0.5 * PQ * QR + PT * QS) = 52 := by
  sorry

end area_of_PQRSTU_l218_218673


namespace sum_possible_values_l218_218525

theorem sum_possible_values (M : ℝ) (h : M * (M - 6) = -5) : ∀ x ∈ {M | M * (M - 6) = -5}, x + (-x) = 6 :=
by sorry

end sum_possible_values_l218_218525


namespace second_person_can_ensure_60_digit_second_person_cannot_ensure_14_digit_l218_218552

theorem second_person_can_ensure_60_digit (digits : ℕ → ℕ) :
  (∀ n, digits n ∈ {1, 2, 3, 4, 5}) →
  (∀ n, (n % 2 = 1 → digits n + digits (n - 1) = 6)) →
  (∃ k, (0 ≤ k ∧ k % 9 = 0) ∧ k = ∑ i in finset.range 60, digits i) :=
begin
  sorry
end

theorem second_person_cannot_ensure_14_digit (digits : ℕ → ℕ) :
  (∀ n, digits n ∈ {1, 2, 3, 4, 5}) →
  ∃ (first_player_strategy : ℕ → ℕ → ℕ),
  ∀ (second_person_strategy : ℕ → ℕ → ℕ),
    ∃ k, (0 ≤ k ∧ k % 9 ≠ 0) ∧ k = ∑ i in finset.range 14, digits i := 
begin
  sorry
end

end second_person_can_ensure_60_digit_second_person_cannot_ensure_14_digit_l218_218552


namespace log_order_sine_cosine_tangent_l218_218517

theorem log_order_sine_cosine_tangent :
  0 < 1 → 1 < real.pi / 4 → cos 1 < sin 1 → sin 1 < 1 / real.sqrt 2 → 1 < tan 1 →
  log (sin 1) (tan 1) < log (cos 1) (tan 1) ∧
  log (cos 1) (tan 1) < log (cos 1) (sin 1) ∧
  log (cos 1) (sin 1) < log (sin 1) (cos 1) :=
by
  intros h0 h1 h2 h3 h4
  sorry

end log_order_sine_cosine_tangent_l218_218517


namespace angle_subtended_by_diameter_is_right_angle_l218_218566

theorem angle_subtended_by_diameter_is_right_angle (O A B : Point) (h : IsDiameter O A B) : 
  ∠(A, O, B) = 90 :=
sorry

end angle_subtended_by_diameter_is_right_angle_l218_218566


namespace depth_of_tank_approx_l218_218771

noncomputable def depth_of_conical_tank (V : ℝ) (d : ℝ) : ℝ :=
  let r := d / 2
  let h := (3 * V) / (real.pi * r^2)
  h

theorem depth_of_tank_approx (h : ℝ) (V : ℝ := 1848) (d : ℝ := 14) :
  abs ((depth_of_conical_tank V d) - 36.01) < 0.01 :=
by
  let r := 7
  let calculated_h := (3 * V) / (real.pi * r^2)
  have : depth_of_conical_tank V d = calculated_h := by rfl
  -- Justification for the depth calculation
  sorry

end depth_of_tank_approx_l218_218771


namespace place_nails_l218_218368

-- Define the 8x8 chessboard and the function to check if no three nails lie on the same straight line
structure Chessboard :=
  (size : Nat)

noncomputable def validNails (n : Nat) := n = 16

/-- Define a function to check if no three points are collinear. -/
def noThreeNailsCollinear (positions : List (Nat × Nat)) : Prop :=
  ∀ (p1 p2 p3 : Nat × Nat), p1 ∈ positions → p2 ∈ positions → p3 ∈ positions →
  p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
  ¬ collinear p1 p2 p3
  
/-- Check if points are collinear -/
def collinear (p1 p2 p3 : Nat × Nat) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

/-- The main theorem: proving it's possible to place 16 nails such that no three lie on the same straight line -/
theorem place_nails : ∃ (positions : List (Nat × Nat)), validNails positions.length ∧ noThreeNailsCollinear positions :=
  sorry

end place_nails_l218_218368


namespace nap_time_l218_218191

-- Definitions of given conditions
def flight_duration : ℕ := 680
def reading_time : ℕ := 120
def movie_time : ℕ := 240
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 70

def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Theorem statement
theorem nap_time : (flight_duration - total_activity_time) / 60 = 3 := by
  -- Here would go the proof steps verifying the equality
  sorry

end nap_time_l218_218191


namespace probability_of_divisible_by_3_divides_of_larger_numbers_l218_218902

def set_of_numbers := {1, 2, 3, 4, 5, 6}

noncomputable def total_combinations := (Finset.univ.choose 3).card

def valid_combinations (s : Finset ℕ) : Prop :=
  let x := s.min' (by simp [set_of_numbers]) in
  x % 3 = 0 ∧ ∃ y ∈ s \ {x}, ∃ z ∈ s \ {x}, x ∣ y ∨ x ∣ z

noncomputable def count_valid_combinations := 
  Finset.card (Finset.filter valid_combinations (Finset.univ.choose 3))

theorem probability_of_divisible_by_3_divides_of_larger_numbers : 
  count_valid_combinations = 4 ∧ total_combinations = 20 → 
  (count_valid_combinations : ℚ) / total_combinations = 1 / 5 :=
sorry

end probability_of_divisible_by_3_divides_of_larger_numbers_l218_218902


namespace ab_bc_value_l218_218377

theorem ab_bc_value (A B C D E : Type*) [OrderedField A] 
  [Triangle ABC] (hABC : ∠B = 60)
  (hCDB_eq : EquilateralTriangle CDB)
  (hAEB_eq : EquilateralTriangle AEB)
  (hCD_EB_eq : ∃ d e, CD = d ∧ EB = e ∧ 3d = 3e + 60)
  (hDE_eq_45 : DE = 45) :
  AB * BC = 675 + 300 * sqrt 3 :=
sorry

end ab_bc_value_l218_218377


namespace combine_piles_l218_218425

theorem combine_piles (n : ℕ) (piles : list ℕ) (h_piles : list.sum piles = n) (h_similar : ∀ x y ∈ piles, x ≤ y → y ≤ 2 * x) :
  ∃ pile, pile ∈ piles ∧ pile = n := sorry

end combine_piles_l218_218425


namespace part1_part2_l218_218713

def A : Set ℝ := {x | x^2 + 2 * x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | -a - 1 < x ∧ x < -a + 1}
def complement_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) (a : ℝ) : Prop := x ∈ B a

theorem part1 (a : ℝ) : (\complement_R A) ∪ B a = {x | x < -2 ∨ x ≥ 1} :=
  by
    sorry

theorem part2 (a : ℝ) : (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) ↔ 0 ≤ a ∧ a ≤ 2 :=
  by
    sorry

end part1_part2_l218_218713


namespace convert_7624_octal_to_decimal_l218_218614

theorem convert_7624_octal_to_decimal : 
  let n := (4 * 8^0 + 2 * 8^1 + 6 * 8^2 + 7 * 8^3) 
  in n = 3988 := 
by 
  let n := (4 * 8^0 + 2 * 8^1 + 6 * 8^2 + 7 * 8^3)
  exact eq.refl 3988

end convert_7624_octal_to_decimal_l218_218614


namespace marble_box_l218_218059

theorem marble_box (T: ℕ) 
  (h_white: (1 / 6) * T = T / 6)
  (h_green: (1 / 5) * T = T / 5)
  (h_red_blue: (19 / 30) * T = 19 * T / 30)
  (h_sum: (T / 6) + (T / 5) + (19 * T / 30) = T): 
  ∃ k : ℕ, T = 30 * k ∧ k ≥ 1 :=
by
  sorry

end marble_box_l218_218059


namespace angle_PGQ_is_90_degrees_l218_218837

-- Define triangle and points
variables (A B C G D E P Q : Point)

-- Given conditions
variables (h1 : MediansMeetAtCentroid A B C G)
variables (h2 : OnLine D E BC)
variables (h3 : DC = CE)
variables (h4 : CE = AB)
variables (h5 : OnSegment P BD)
variables (h6 : OnSegment Q BE)
variables (h7 : 2 * BP = PD)
variables (h8 : 2 * BQ = QE)

-- Define the theorem
theorem angle_PGQ_is_90_degrees :
  ∠ P G Q = 90 :=
by
  sorry

end angle_PGQ_is_90_degrees_l218_218837


namespace pile_of_stones_l218_218413

def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

theorem pile_of_stones (n : ℕ) (f : ℕ → ℕ): (∀ i, 1 ≤ f i ∧ f i ≤ n) → 
  (∀ j k, similar_sizes (f j) (f k)) → True :=
by
  simp
  exact true.intro


end pile_of_stones_l218_218413


namespace labeling_possible_iff_even_l218_218057

theorem labeling_possible_iff_even (n : ℕ) (h_gt_2 : n > 2)
    (h_no_parallel : ∀ i j, i ≠ j → ¬parallel i j)
    (h_no_triplet_intersect : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬intersects_at_the_same_point i j k)
    (h_labels : ∀ p, is_intersection_point p → 1 ≤ label(p) ∧ label(p) ≤ n-1) :
    (∃ labels, (∀ line, all_labels_present line labels) ↔ even n) :=
sorry

end labeling_possible_iff_even_l218_218057


namespace expression_value_l218_218826

noncomputable def compute_expression (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80

theorem expression_value (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1)
    : compute_expression ω h h2 = -ω^2 :=
sorry

end expression_value_l218_218826


namespace verify_placements_l218_218452

-- Definitions for participants and their possible places
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

-- Each participant should be mapped to a place (1, 2, 3, 4)
def Place : Participant → ℕ := λ p,
  match p with
  | Participant.Olya => 2
  | Participant.Oleg => 1
  | Participant.Polya => 3
  | Participant.Pasha => 4

-- Conditions based on the problem statement
def statement_Olya : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Polya % 2 = 1 ∧ Place Participant.Pasha % 2 = 1)

def statement_Oleg : Prop :=
  (abs (Place Participant.Oleg - Place Participant.Olya) = 1)

def statement_Pasha : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Olya % 2 = 1 ∧ Place Participant.Polya % 2 = 1)

-- Only one child tells the truth and the others lie
def exactly_one_true (a b c : Prop) : Prop := (a ∨ b ∨ c) ∧ (a → ¬b ∧ ¬c) ∧ (b → ¬a ∧ ¬c) ∧ (c → ¬a ∧ ¬b)

-- The main theorem to be proven
theorem verify_placements :
  exactly_one_true (statement_Olya) (statement_Oleg) (statement_Pasha) ∧ 
  Place Participant.Olya = 2 ∧
  Place Participant.Oleg = 1 ∧
  Place Participant.Polya = 3 ∧
  Place Participant.Pasha = 4 :=
by
  sorry

end verify_placements_l218_218452


namespace cauchy_schwarz_inequality_l218_218721

variable {n : ℕ}
variables (x y : Fin n → ℝ)

theorem cauchy_schwarz_inequality
  (h : ∀ i, y i > 0) :
  (∑ i, y i) * (∑ i, (x i)^2 / (y i)) ≥ (∑ i, x i)^2 :=
by sorry

end cauchy_schwarz_inequality_l218_218721


namespace volume_of_rotated_solid_l218_218043

theorem volume_of_rotated_solid :
  let rect1 := (height := 4, width := 1)
  let rect2 := (height := 2, width := 3)
  let volume_rect1 := π * (rect1.width)^2 * rect1.height
  let volume_rect2 := π * (rect2.width)^2 * rect2.height
  volume_rect1 + volume_rect2 = 22 * π := by
{
  let rect1 := (height := 4, width := 1)
  let rect2 := (height := 2, width := 3)
  let volume_rect1 := π * (rect1.width)^2 * rect1.height
  let volume_rect2 := π * (rect2.width)^2 * rect2.height
  sorry
}

end volume_of_rotated_solid_l218_218043


namespace y_when_x_is_4_l218_218054

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l218_218054


namespace real_part_of_z_l218_218836

-- Define the complex number z based on the given condition
def z : ℂ := (2 + complex.i) / (1 + complex.i)^2

-- The theorem to prove the real part of z is 1/2
theorem real_part_of_z : z.re = 1 / 2 := by
  sorry

end real_part_of_z_l218_218836


namespace vector_magnitude_l218_218330

-- Given conditions
variables (a b : ℝ^n)
variables (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : ∥a + b∥ = 1)

-- Goal: show that the magnitude of (2a + b) is √3
theorem vector_magnitude (a b : ℝ^n) (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : ∥a + b∥ = 1) :
  ∥2 • a + b∥ = √3 :=
begin
  sorry
end

end vector_magnitude_l218_218330


namespace chicken_rabbit_in_cage_l218_218795

theorem chicken_rabbit_in_cage (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : 
  (x + y = 35) ∧ (2 * x + 4 * y = 94) :=
by
  split
  · exact h1
  · exact h2

end chicken_rabbit_in_cage_l218_218795


namespace inequality_solver_l218_218021

variable {m n x : ℝ}

-- Main theorem statement validating the instances described above.
theorem inequality_solver (h : 2 * m * x + 3 < 3 * x + n) :
  (2 * m - 3 > 0 ∧ x < (n - 3) / (2 * m - 3)) ∨ 
  (2 * m - 3 < 0 ∧ x > (n - 3) / (2 * m - 3)) ∨ 
  (m = 3 / 2 ∧ n > 3 ∧ ∀ x : ℝ, true) ∨ 
  (m = 3 / 2 ∧ n ≤ 3 ∧ ∀ x : ℝ, false) :=
sorry

end inequality_solver_l218_218021


namespace concurrency_proof_l218_218830

/-- Given an acute triangle ABC with altitudes AD, BE, and CF,
and points D, E, and F as the points of tangency of the incircle with the sides BC, CA, and AB respectively,
proves that the lines AD, BE, and CF intersect at a single point. --/

def altitudes_concurrent 
  (A B C D E F : Type*)
  [IsAcuteTriangle A B C]
  [IsAltitude A B C D]
  [IsAltitude A B C E]
  [IsAltitude A B C F]
  [IsTangencyPoint D BC]
  [IsTangencyPoint E CA]
  [IsTangencyPoint F AB] : Prop :=
  areConcurrentLines AD BE CF

theorem concurrency_proof :
  altitudes_concurrent A B C AD BE CF D E F :=
sorry

end concurrency_proof_l218_218830


namespace sequence_product_is_128_l218_218916

-- Define the sequence of fractions
def fractional_sequence (n : ℕ) : Rat :=
  if n % 2 = 0 then 1 / (2 : ℕ) ^ ((n + 2) / 2)
  else (2 : ℕ) ^ ((n + 1) / 2)

-- The target theorem: prove the product of the sequence results in 128
theorem sequence_product_is_128 : 
  (List.prod (List.map fractional_sequence [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])) = 128 := 
by
  sorry

end sequence_product_is_128_l218_218916


namespace B_worked_days_l218_218156

-- Definitions for conditions
def A_work_rate : ℝ := 1 / 15
def B_work_rate : ℝ := 1 / 15
def work_done (days : ℝ) (rate : ℝ) := days * rate

-- Theorem statement
theorem B_worked_days (x : ℝ) (total_work : ℝ) (remaining_work_days : ℝ) :
  B_work_rate = 1 / 15 ∧
  A_work_rate = 1 / 15 ∧
  remaining_work_days = 5 ∧
  total_work = 1 →
  work_done (x + remaining_work_days) (A_work_rate + B_work_rate) = total_work →
  x = 5 :=
by
  sorry

end B_worked_days_l218_218156


namespace harvested_potatoes_l218_218162

-- Define the conditions
def potato_bundles := 10
def price_per_potato_bundle := 1.90
def carrot_bundles := 320 / 20
def price_per_carrot_bundle := 2.0
def total_revenue := 51.0

-- Define the number of potatoes per bundle
def potatoes_per_bundle := 25

-- The theorem to be proven
theorem harvested_potatoes :
  (potato_bundles : ℕ) * potatoes_per_bundle = 250 :=
by
  have h1 : potato_bundles * price_per_potato_bundle + carrot_bundles * price_per_carrot_bundle = total_revenue, by sorry
  have h2 : carrot_bundles = 16, by sorry
  -- Conclude the proof using above proofs and conditions
  sorry

end harvested_potatoes_l218_218162


namespace product_of_primes_l218_218113

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l218_218113


namespace entrance_exit_plans_l218_218526

-- Definitions as per the conditions in the problem
def south_gates : Nat := 4
def north_gates : Nat := 3
def west_gates : Nat := 2

-- Conditions translated into Lean definitions
def ways_to_enter := south_gates + north_gates
def ways_to_exit := west_gates + north_gates

-- The theorem to be proved: the number of entrance and exit plans
theorem entrance_exit_plans : ways_to_enter * ways_to_exit = 35 := by
  sorry

end entrance_exit_plans_l218_218526


namespace y_when_x_is_4_l218_218053

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l218_218053


namespace competition_results_l218_218471

-- Participants and positions
inductive Participant : Type
| Oleg
| Olya
| Polya
| Pasha

-- Places in the competition (1st, 2nd, 3rd, 4th)
def Place := Fin 4

-- Statements made by the children
def Olya_statement1 : Prop := ∀ p, p % 2 = 1 -> p = Participant.Oleg ∨ p = Participant.Pasha
def Oleg_statement1 : Prop := ∃ p1 p2: Place, p1 < p2 ∧ (p1 = p2 + 1)
def Pasha_statement1 : Prop := ∀ p, p % 2 = 1 -> (p = Place 1 ∨ p = Place 3)

-- Truthfulness of the statements
def only_one_truthful (Olya_true : Prop) (Oleg_true : Prop) (Pasha_true : Prop) :=
  (Olya_true ∧ ¬ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ ¬ Oleg_true ∧ Pasha_true)

-- The actual positions
def positions : Participant → Place
| Participant.Oleg  := 0
| Participant.Olya  := 1
| Participant.Polya := 2
| Participant.Pasha := 3

-- The Lean statement to prove
theorem competition_results :
  ((Oleg_statement1 ↔ positions Participant.Oleg = 0) ∧ 
  (Olya_statement1 ↔ positions Participant.Olya = 1) ∧ 
  (Pasha_statement1 ↔ positions Participant.Pasha = 3)) ∧ 
  only_one_truthful (positions Participant.Oleg = 0) 
                    (positions Participant.Olya = 0) 
                    (positions Participant.Pasha = 0) ∧
  positions Participant.Oleg = 0 ∧ 
  positions Participant.Olya = 1 ∧
  positions Participant.Polya = 2 ∧
  positions Participant.Pasha = 3 := 
sorry

end competition_results_l218_218471


namespace integer_pairs_3_pow_div_l218_218671

theorem integer_pairs_3_pow_div (m n : ℤ) :
    (m * n ∣ 3^m + 1) ∧ (m * n ∣ 3^n + 1) ↔ (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) :=
by
  sorry

end integer_pairs_3_pow_div_l218_218671


namespace pile_division_possible_l218_218418

theorem pile_division_possible (n : ℕ) :
  ∃ (division : list ℕ), (∀ x ∈ division, x = 1) ∧ division.sum = n :=
by
  sorry

end pile_division_possible_l218_218418


namespace remaining_stock_is_120_l218_218447

-- Definitions derived from conditions
def green_beans_weight : ℕ := 60
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 10
def rice_lost_weight : ℕ := rice_weight / 3
def sugar_lost_weight : ℕ := sugar_weight / 5
def remaining_rice : ℕ := rice_weight - rice_lost_weight
def remaining_sugar : ℕ := sugar_weight - sugar_lost_weight
def remaining_stock_weight : ℕ := remaining_rice + remaining_sugar + green_beans_weight

-- Theorem
theorem remaining_stock_is_120 : remaining_stock_weight = 120 := by
  sorry

end remaining_stock_is_120_l218_218447


namespace pairs_not_sum_to_7_l218_218660

/-- Pair A: (4, 3) --/
def pairA := (4, 3)

/-- Pair B: (-1, 8) --/
def pairB := (-1, 8)

/-- Pair C: (10, -2) --/
def pairC := (10, -2)

/-- Pair D: (2, 5) --/
def pairD := (2, 5)

/-- Pair E: (3, 5) --/
def pairE := (3, 5)

/-- Calculate the sum of a pair --/
def pairSum (pair : Int × Int) : Int := pair.1 + pair.2

/-- Proof statement: Pairs that do not sum to 7 --/
theorem pairs_not_sum_to_7 : pairSum pairC ≠ 7 ∧ pairSum pairE ≠ 7 := by
  sorry

end pairs_not_sum_to_7_l218_218660


namespace correct_option_is_D_l218_218978

-- Define the mathematical conditions from the problem
def log_cond (a b : ℝ) := (a > b ∧ b > 0) → (Real.log a < Real.log b)
def vec_cond (m : ℝ) := ∀ (a b : ℝ × ℝ), (a = (1, m) ∧ b = (m, 2 * m - 1)) → (a.1 * b.1 + a.2 * b.2 = 0) → (m = 1)
def negation_cond := ∀ n : ℕ, (n > 0) → (3^n > (n+2) * 2^(n-1)) → (∃ n : ℕ, (n > 0) ∧ 3^n ≤ (n+2) * 2^(n-1))
def inv_cond {α : Type*} [TopologicalSpace α] {f : α → ℝ} (a b : α) := 
  (ContinuousOn f (Set.Icc a b)) → (f a * f b < 0 → ∃ c ∈ Set.Icc a b, f c = 0) → false

-- Prove that the correct option is D
theorem correct_option_is_D : ∃ (correct : Prop), 
  (correct = ¬ log_cond ∧ correct = ¬ vec_cond ∧ correct = ¬ negation_cond ∧ correct = inv_cond) :=
sorry

end correct_option_is_D_l218_218978


namespace competition_results_l218_218470

-- Participants and positions
inductive Participant : Type
| Oleg
| Olya
| Polya
| Pasha

-- Places in the competition (1st, 2nd, 3rd, 4th)
def Place := Fin 4

-- Statements made by the children
def Olya_statement1 : Prop := ∀ p, p % 2 = 1 -> p = Participant.Oleg ∨ p = Participant.Pasha
def Oleg_statement1 : Prop := ∃ p1 p2: Place, p1 < p2 ∧ (p1 = p2 + 1)
def Pasha_statement1 : Prop := ∀ p, p % 2 = 1 -> (p = Place 1 ∨ p = Place 3)

-- Truthfulness of the statements
def only_one_truthful (Olya_true : Prop) (Oleg_true : Prop) (Pasha_true : Prop) :=
  (Olya_true ∧ ¬ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ ¬ Oleg_true ∧ Pasha_true)

-- The actual positions
def positions : Participant → Place
| Participant.Oleg  := 0
| Participant.Olya  := 1
| Participant.Polya := 2
| Participant.Pasha := 3

-- The Lean statement to prove
theorem competition_results :
  ((Oleg_statement1 ↔ positions Participant.Oleg = 0) ∧ 
  (Olya_statement1 ↔ positions Participant.Olya = 1) ∧ 
  (Pasha_statement1 ↔ positions Participant.Pasha = 3)) ∧ 
  only_one_truthful (positions Participant.Oleg = 0) 
                    (positions Participant.Olya = 0) 
                    (positions Participant.Pasha = 0) ∧
  positions Participant.Oleg = 0 ∧ 
  positions Participant.Olya = 1 ∧
  positions Participant.Polya = 2 ∧
  positions Participant.Pasha = 3 := 
sorry

end competition_results_l218_218470


namespace product_of_primes_l218_218111

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l218_218111


namespace find_width_l218_218160

variable (L : ℕ) (W : ℕ)
variable (brick_length : ℕ) (brick_width : ℕ)
variable (num_bricks : ℕ)
variable (courtyard_area : ℕ) (bricks_area : ℕ)

-- Given conditions
def courtyard_length := 18
def brick_length := 20
def brick_width := 10
def num_bricks := 14400
def courtyard_area := 1800 * W
def bricks_area := num_bricks * (brick_length * brick_width)

-- Proof problem statement
theorem find_width
  (h_courtyard_area : courtyard_area = bricks_area) : W = 16 :=
by
  -- Proof will go here
  sorry

end find_width_l218_218160


namespace debby_water_bottles_l218_218997

theorem debby_water_bottles (total_bottles : ℕ) (days : ℕ) (bottles_per_day : ℕ) :
  total_bottles = 153 → days = 17 → (total_bottles = days * bottles_per_day) → bottles_per_day = 9 := by
  intros h1 h2 h3
  subst h1
  subst h2
  have : 153 = 17 * bottles_per_day := h3
  have : bottles_per_day = 9 :=
    Nat.eq_of_mul_eq_mul_right (dec_trivial) this
  exact this

end debby_water_bottles_l218_218997


namespace shooting_accuracy_l218_218810

theorem shooting_accuracy 
  (P_A : ℚ) 
  (P_AB : ℚ) 
  (h1 : P_A = 9 / 10) 
  (h2 : P_AB = 1 / 2) 
  : P_AB / P_A = 5 / 9 := 
by
  sorry

end shooting_accuracy_l218_218810


namespace possible_values_for_f_zero_l218_218929

noncomputable def poly_val (A n : ℤ) : ℤ :=
  A * (-1)^n * n.factorial + 1

theorem possible_values_for_f_zero (f : ℤ → ℤ) (n : ℤ) (h_deg : polynomial.degree f = polynomial.degree (∏ i in finset.range (n + 1), polynomial.X - polynomial.C i)) 
  (h : ∀ x : ℤ, 1 ≤ x → x ≤ n → f x = x^2 + 1) : 
  ∃ A : ℤ, f 0 = poly_val A n ∨ (n = 1 ∧ ∃ m : ℤ, f 0 = 2 - m) ∨ (n = 2 ∧ (f 0 = 1 ∨ ∃ A : ℤ, f 0 = 2 * A + 1)) :=
sorry

end possible_values_for_f_zero_l218_218929


namespace gcd_840_1764_gcd_459_357_l218_218930

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := sorry

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := sorry

end gcd_840_1764_gcd_459_357_l218_218930


namespace product_of_primes_l218_218078

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l218_218078


namespace area_EPHQ_l218_218853

theorem area_EPHQ {EFGH : Type} 
  (rectangle_EFGH : EFGH) 
  (length_EF : Real) (width_EG : Real) 
  (P_point : Real) (Q_point : Real) 
  (area_EFGH : Real) 
  (area_EFP : Real) 
  (area_EHQ : Real) : 
  length_EF = 12 → width_EG = 6 → P_point = 4 → Q_point = 3 → 
  area_EFGH = length_EF * width_EG →
  area_EFP = (1 / 2) * width_EG * P_point →
  area_EHQ = (1 / 2) * length_EF * Q_point → 
  (area_EFGH - area_EFP - area_EHQ) = 42 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end area_EPHQ_l218_218853


namespace probability_different_colors_l218_218897

/-- There are 5 blue chips and 3 yellow chips in a bag. One chip is drawn from the bag and placed
back into the bag. A second chip is then drawn. Prove that the probability of the two selected chips
being of different colors is 15/32. -/
theorem probability_different_colors : 
  let total_chips := 8
  let blue_chips := 5
  let yellow_chips := 3
  let prob_blue_then_yellow := (blue_chips/total_chips) * (yellow_chips/total_chips)
  let prob_yellow_then_blue := (yellow_chips/total_chips) * (blue_chips/total_chips)
  prob_blue_then_yellow + prob_yellow_then_blue = 15/32 := by
  sorry

end probability_different_colors_l218_218897


namespace vanessa_missed_days_l218_218071

theorem vanessa_missed_days (V M S : ℕ) 
                           (h1 : V + M + S = 17) 
                           (h2 : V + M = 14) 
                           (h3 : M + S = 12) : 
                           V = 5 :=
sorry

end vanessa_missed_days_l218_218071


namespace accounting_vs_calling_clients_l218_218842

/--
Given:
1. Total time Maryann worked today is 560 minutes.
2. Maryann spent 70 minutes calling clients.

Prove:
Maryann spends 7 times longer doing accounting than calling clients.
-/
theorem accounting_vs_calling_clients 
  (total_time : ℕ) 
  (calling_time : ℕ) 
  (h_total : total_time = 560) 
  (h_calling : calling_time = 70) : 
  (total_time - calling_time) / calling_time = 7 :=
  sorry

end accounting_vs_calling_clients_l218_218842


namespace sum_four_digit_numbers_l218_218261

theorem sum_four_digit_numbers : 
  let digits := [1, 2, 3, 4, 5]
  let perms := digits.permutations
  ∑ p in perms.filter (λ x, x.length = 4), (1000 * x.head + 100 * x[1] + 10 * x[2] + x[3]) = 399960 := 
by sorry

end sum_four_digit_numbers_l218_218261


namespace jonathan_daily_burn_l218_218816

-- Conditions
def daily_calories : ℕ := 2500
def extra_saturday_calories : ℕ := 1000
def weekly_deficit : ℕ := 2500

-- Question and Answer
theorem jonathan_daily_burn :
  let weekly_intake := 6 * daily_calories + (daily_calories + extra_saturday_calories)
  let total_weekly_burn := weekly_intake + weekly_deficit
  total_weekly_burn / 7 = 3000 :=
by
  sorry

end jonathan_daily_burn_l218_218816


namespace cannot_all_plus_from_initial_l218_218991

def initial_table :=
  [[true, true, true, false],
   [false, true, true, false],
   [true, true, true, false],
   [true, false, false, false]]

def invert_sign (sign : bool) : bool :=
  not sign

def perform_operation (table : list (list bool)) (row : option ℕ) (col : option ℕ) : list (list bool) :=
  match row, col with
  | some r, none   => table.map_with_index (λ i row, if i = r then row.map invert_sign else row)
  | none, some c   => table.map (λ row, row.map_with_index (λ j sign, if j = c then invert_sign sign else sign))
  | _, _           => table

def all_plus (table : list (list bool)) : bool :=
  table.all (λ row, row.all id)

theorem cannot_all_plus_from_initial :
  ∀ (ops : list (option ℕ × option ℕ)),
  ¬ all_plus (ops.foldl (λ tbl op, perform_operation tbl op.fst op.snd) initial_table) :=
by sorry

end cannot_all_plus_from_initial_l218_218991


namespace roots_equal_iff_m_eq_3_div_2_l218_218659

theorem roots_equal_iff_m_eq_3_div_2 :
  ∀ (x m : ℝ), (∀ x, (x * (x + 3) - (m - 3)) / ((x + 3) * (m + 1)) = x / m) ∧ (∀ x1 x2, 
       ∃ k, x1 = x2 ∧ k * x1 = k * x2) -> m = 3 / 2 :=
by
  intros x m h
  sorry

end roots_equal_iff_m_eq_3_div_2_l218_218659


namespace range_of_t_l218_218040

noncomputable def geometric_sequence (a : ℝ) (n : ℕ) := (a^n)

def S_n (a : ℝ) (n : ℕ) : ℝ := (a*(1 - a^n))/(1-a)

theorem range_of_t (a : ℝ) (t : ℝ) (n : ℕ) (h₁ : a = 1/5) (h₂ : ∀ m n : ℕ, geometric_sequence a (m + n) = geometric_sequence a m * geometric_sequence a n) (h₃ : ∀ n : ℕ, S_n a n < t) :
  t ≥ 1/4 := by
  sorry

end range_of_t_l218_218040


namespace airplane_altitude_correct_l218_218975

noncomputable def airplane_altitude 
  (d_AB : ℝ) (angle_Alice : ℝ) (angle_Bob : ℝ) 
  (distance_Alice_Bob : ℝ) : ℝ :=
  let d_A := distance_Alice_Bob / (real.sqrt 2) in
  d_A

theorem airplane_altitude_correct :
  airplane_altitude 12 45 30 12 = 8.5 :=
by
  unfold airplane_altitude
  -- calculate d_A using the given formula
  sorry

end airplane_altitude_correct_l218_218975


namespace A_excircle_parallelogram_l218_218818

theorem A_excircle_parallelogram
  (A B C O_a A_a C_b A_b A_c B_c X : Point)
  (h_A_excircle : excircle A B C O_a A_a)
  (h_B_excircle : excircle B A C C_b A_b)
  (h_C_excircle : excircle C B A A_c B_c)
  (h_intersection : ∃ P, line C_b A_b = P ∧ line A_c B_c = P)
  (h_intersect_at_X : intersect_at (line C_b A_b) (line A_c B_c) X) :
  is_parallelogram A O_a A_a X :=
sorry

end A_excircle_parallelogram_l218_218818


namespace speed_of_stream_l218_218593

-- Definitions of conditions based on the given problem
def D := 15 + v  -- The distance downstream
def upstream_speed := 15 - v  -- Speed of boat going upstream
def downstream_time := 1  -- Time to travel downstream is 1 hour
def upstream_time := 11.5  -- Time to travel upstream is 11.5 hours
def boat_speed := 15 -- Speed of the boat in still water is 15 kmph

-- The theorem to prove the speed of the stream
theorem speed_of_stream : 
  ∃ v : ℝ, 
  (15 + v) * 1 = (15 - v) * 11.5 → 
  v = 12.6 :=
by
  sorry

end speed_of_stream_l218_218593


namespace jade_rate_ratio_l218_218793

-- Variables and constants for diameter and "jade rates"
variables (a : ℝ)
def sphere_volume := (4 / 3) * Real.pi * (a / 2)^3
def cylinder_volume := a * Real.pi * (a / 2)^2
def cube_volume := a^3

-- Defining the "jade rates"
def k1 := sphere_volume a / a^3
def k2 := cylinder_volume a / a^3
def k3 := cube_volume a / a^3

-- The theorem statement
theorem jade_rate_ratio : k1 a : k2 a : k3 a = (Real.pi / 6) : (Real.pi / 4) : 1 :=
by
  sorry

end jade_rate_ratio_l218_218793


namespace equation_of_tangent_line_min_slope_l218_218298

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + x^2

def deriv_f (x : ℝ) : ℝ := deriv f x

theorem equation_of_tangent_line_min_slope :
  (∀ l : ℝ → ℝ, ∃ x₀ : ℝ, l = λ x, deriv f x₀ * (x - x₀) + f x₀ ∧ 
  ∀ x : ℝ, deriv f x ≥ deriv f x₀ ) →
  (∃ l : ℝ → ℝ, l = λ x, 4 * x - 3) :=
by
  sorry

end equation_of_tangent_line_min_slope_l218_218298


namespace probability_of_good_at_least_def_expected_value_of_defective_items_l218_218297

def total_items : ℕ := 7
def good_items : ℕ := 4
def defective_items : ℕ := 3

-- Definition of random selection of 3 items, and condition of the number of good items being no less than defective items
def probability_condition (n_selected: ℕ) : Prop :=
  let total_combinations := Nat.choose total_items n_selected in
  let good_def1 := Nat.choose good_items 2 * Nat.choose defective_items 1 in
  let good_only := Nat.choose good_items 3 in
  (good_def1 + good_only) / total_combinations = 22 / 35

theorem probability_of_good_at_least_def : probability_condition 3 :=
  sorry

-- Definition of expected value for the number of defective items when selecting 5 items
def expected_value_condition : Prop :=
  let prob_ξ_1 := Nat.choose good_items 4 * Nat.choose defective_items 1 / Nat.choose total_items 5 in
  let prob_ξ_2 := Nat.choose good_items 3 * Nat.choose defective_items 2 / Nat.choose total_items 5 in
  let prob_ξ_3 := Nat.choose good_items 2 * Nat.choose defective_items 3 / Nat.choose total_items 5 in
  (1 * prob_ξ_1 + 2 * prob_ξ_2 + 3 * prob_ξ_3) = 15 / 7

theorem expected_value_of_defective_items : expected_value_condition :=
  sorry

end probability_of_good_at_least_def_expected_value_of_defective_items_l218_218297


namespace product_of_primes_l218_218110

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l218_218110


namespace tiling_rectangles_recurrence_l218_218271

noncomputable def tiling_sequence : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := tiling_sequence (n + 1) + tiling_sequence n

theorem tiling_rectangles_recurrence (n : ℕ) (h : 2 ≤ n):
  tiling_sequence n = tiling_sequence (n - 1) + tiling_sequence (n - 2) :=
by
  sorry

end tiling_rectangles_recurrence_l218_218271


namespace correct_statements_l218_218567

def statementA : Prop := ∀ crystal : Type, (thermal_conductivity crystal ≠ constant) ∨ (electrical_conductivity crystal ≠ constant) ∨ (mechanical_strength crystal ≠ constant)

def statementB : Prop := ∀ dewdrop : Type, (shape dewdrop = spherical) → (reason_for_shape dewdrop = surface_tension)

def statementC : Prop := ∀ object : Type, (high_temperature object) → ∃ molecule : Type, (speed molecule < average_speed)

def statementD : Prop := ∀ ideal_gas : Type, (work_done ideal_gas) → (internal_energy_change ideal_gas ≠ necessarily_increase)

theorem correct_statements : statementA ∧ ¬statementB ∧ statementC ∧ statementD :=
by sorry

end correct_statements_l218_218567


namespace train_platform_length_equal_l218_218511

theorem train_platform_length_equal 
  (v : ℝ) (t : ℝ) (L_train : ℝ)
  (h1 : v = 144 * (1000 / 3600))
  (h2 : t = 60)
  (h3 : L_train = 1200) :
  L_train = 2400 - L_train := 
sorry

end train_platform_length_equal_l218_218511


namespace sum_four_digit_numbers_l218_218263

theorem sum_four_digit_numbers : 
  let digits := [1, 2, 3, 4, 5]
  let perms := digits.permutations
  ∑ p in perms.filter (λ x, x.length = 4), (1000 * x.head + 100 * x[1] + 10 * x[2] + x[3]) = 399960 := 
by sorry

end sum_four_digit_numbers_l218_218263


namespace video_more_dislikes_l218_218590

-- Definition of the problem conditions
def likes : ℕ := 3000
def initial_dislikes : ℕ := (likes / 2) + 100
def final_dislikes : ℕ := 2600

-- The proposition we need to prove
theorem video_more_dislikes : (final_dislikes - initial_dislikes) = 1000 :=
by
  have h1 : initial_dislikes = (likes / 2) + 100 := rfl
  have h2 : likes = 3000 := rfl
  have h3 : final_dislikes = 2600 := rfl
  calc
    final_dislikes - initial_dislikes
        = 2600 - ((3000 / 2) + 100) : by rw [h1, h2, h3]
    ... = 1000 : by sorry

end video_more_dislikes_l218_218590


namespace traffic_goal_possible_l218_218781

-- Definitions for the problem
def city_count := 32
def can_change_direction (days: ℕ) := days >= 214

-- Hypothesis: initial conditions
axiom roads_connected : ∀ (i: ℕ) (j: ℕ), i ≠ j → (i < city_count ∧ j < city_count) 

-- Goal: Prove traffic organization goal can be reached within the given days
theorem traffic_goal_possible : can_change_direction 214 → ∃ days ≤ 214, 
  (∀ i j, i ≠ j → reach_goal i j days) := sorry

end traffic_goal_possible_l218_218781


namespace piles_can_be_reduced_l218_218399

/-! 
  We define similar sizes as the difference between sizes being at most a factor of two.
  Given any number of piles of stones, we aim to prove that these piles can be combined 
  iteratively into one single pile.
-/

def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

theorem piles_can_be_reduced (n : ℕ) :
  ∃ pile : ℕ, (pile = n) ∧ (∀ piles : list ℕ, list.sum piles = n → 
    (∃ piles' : list ℕ, list.sum piles' = n ∧ list.length piles' = 1)) :=
by
  -- Placeholder for the proof.
  sorry

end piles_can_be_reduced_l218_218399


namespace sum_of_undefined_values_for_g_l218_218212

theorem sum_of_undefined_values_for_g :
  let g (x : ℝ) := 2 / (1 + 1 / (2 + 1 / x)) in
  { x : ℝ | g x = 0 }.sum = -5 / 6 :=
by sorry

end sum_of_undefined_values_for_g_l218_218212


namespace piles_can_be_reduced_l218_218397

/-! 
  We define similar sizes as the difference between sizes being at most a factor of two.
  Given any number of piles of stones, we aim to prove that these piles can be combined 
  iteratively into one single pile.
-/

def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

theorem piles_can_be_reduced (n : ℕ) :
  ∃ pile : ℕ, (pile = n) ∧ (∀ piles : list ℕ, list.sum piles = n → 
    (∃ piles' : list ℕ, list.sum piles' = n ∧ list.length piles' = 1)) :=
by
  -- Placeholder for the proof.
  sorry

end piles_can_be_reduced_l218_218397


namespace teacher_work_months_l218_218620

variable (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) (total_earnings : ℕ)

def monthly_earnings (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) : ℕ :=
  periods_per_day * pay_per_period * days_per_month

def number_of_months_worked (total_earnings : ℕ) (monthly_earnings : ℕ) : ℕ :=
  total_earnings / monthly_earnings

theorem teacher_work_months :
  let periods_per_day := 5
  let pay_per_period := 5
  let days_per_month := 24
  let total_earnings := 3600
  number_of_months_worked total_earnings (monthly_earnings periods_per_day pay_per_period days_per_month) = 6 :=
by
  sorry

end teacher_work_months_l218_218620


namespace average_weight_increase_l218_218498

theorem average_weight_increase (A : ℝ) (hA : 8 * A + 20 = (80 : ℝ) + (8 * (A - (60 - 80) / 8))) :
  ((8 * A + 20) / 8) - A = (2.5 : ℝ) :=
by
  sorry

end average_weight_increase_l218_218498


namespace beads_per_package_eq_40_l218_218331

theorem beads_per_package_eq_40 (b r : ℕ) (x : ℕ) (total_beads : ℕ) 
(h1 : b = 3) (h2 : r = 5) (h3 : total_beads = 320) (h4 : total_beads = (b + r) * x) :
  x = 40 := by
  sorry

end beads_per_package_eq_40_l218_218331


namespace sqrt_inequality_l218_218279

theorem sqrt_inequality (n : ℕ) : 
    (Real.root (n + Real.root n n).toReal (n : ℝ)) + (Real.root (n - Real.root n n).toReal (n : ℝ)) ≤ 2 * Real.root n (n : ℝ) :=
sorry

end sqrt_inequality_l218_218279


namespace P_Q_M_collinear_l218_218367

open EuclideanGeometry

variables {A B C K M P Q : Point} {triangle : Triangle ABC}
variables {ω : Circle} {Ω : Circle}

-- Conditions
def K_on_bisector_of_angle_BAC : Prop :=
  ∃ (bisector : Line), K ∈ bisector ∧ bisector.is_bisector (∠ BAC)

def CK_intersects_circumcircle_at_M_ne_C : Prop :=
  M ≠ C ∧ lies_on (line_through C K) M ∧ lies_on ω M ∧ is_circumcircle ω triangle

def Ω_passes_through_A_touches_CM_at_K : Prop :=
  passes_through Ω A ∧ tangent_to Ω (line_through C M) K

def Ω_intersects_AB_at_P_ne_A_and_ω_at_Q_ne_A : Prop :=
  P ≠ A ∧ Q ≠ A ∧ lies_on (line_through A B) P ∧ lies_on Ω P ∧ lies_on ω Q

-- Question
theorem P_Q_M_collinear
  (H1 : K_on_bisector_of_angle_BAC)
  (H2 : CK_intersects_circumcircle_at_M_ne_C)
  (H3 : Ω_passes_through_A_touches_CM_at_K)
  (H4 : Ω_intersects_AB_at_P_ne_A_and_ω_at_Q_ne_A) :
  collinear {P, Q, M} :=
sorry

end P_Q_M_collinear_l218_218367


namespace avg_first_two_numbers_l218_218870

theorem avg_first_two_numbers (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = 3.95)
  (h2 : (a₃ + a₄) / 2 = 3.85)
  (h3 : (a₅ + a₆) / 2 = 4.600000000000001) :
  (a₁ + a₂) / 2 ≈ 3.4 := 
by
  sorry

end avg_first_two_numbers_l218_218870


namespace tank_fill_time_l218_218933

theorem tank_fill_time (R1 R2 : ℝ) (leak_fraction : ℝ) (T : ℝ) 
  (hR1 : R1 = 1 / 20) 
  (hR2 : R2 = 1 / 30) 
  (hleak : leak_fraction = 1 / 3) 
  (hT : T = 18) : 
  1 / ((2 / 3) * (R1 + R2)) = T :=
by 
  rw [hR1, hR2, hleak, hT],
  norm_num,
  rw [← mul_assoc, mul_comm _ (1/18), mul_one, inv_mul_cancel],
  exact two_ne_zero,
  exact six_ne_zero

end tank_fill_time_l218_218933


namespace cauchy_schwarz_inequality_l218_218288

theorem cauchy_schwarz_inequality {n : ℕ} (a b : Fin n → ℝ) :
  (∑ i, a i * b i) ^ 2 ≤ (∑ i, (a i)^2) * (∑ i, (b i)^2) :=
sorry

end cauchy_schwarz_inequality_l218_218288


namespace cistern_capacity_l218_218569

-- Definitions of the given conditions
def capacity (C : ℝ) := C
def leak_rate (C : ℝ) := C / 20
def tap_inflow_rate := 4
def combined_rate (C : ℝ) := C / 24

-- The proof statement to show that given the conditions, the cistern's capacity is 480 liters
theorem cistern_capacity : ∀ (C : ℝ),
  leak_rate C - tap_inflow_rate = combined_rate C →
  C = 480 :=
by
  intros C h
  rw [leak_rate, combined_rate] at h
  sorry

end cistern_capacity_l218_218569


namespace trapezoid_perimeter_l218_218548

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  ( (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 ).sqrt

noncomputable def perimeter (J K L M : point) : ℝ :=
  distance J K + distance K L + distance L M + distance M J

theorem trapezoid_perimeter :
  let J := (-3, -4)
  let K := (-3, 1)
  let L := (5, 7)
  let M := (5, -4)
  perimeter J K L M = 26 :=
by
  sorry

end trapezoid_perimeter_l218_218548


namespace yellow_candies_bounds_l218_218577

noncomputable def candy_problem := sorry

theorem yellow_candies_bounds 
    (total_candies : ℕ) 
    (four_colors : ℕ) 
    (yellow_most : Prop) 
    (turns : ℕ → ℕ) 
    (equal_split : ℕ → ℕ → Prop) 
    (yellow_bounds : ∀ n, total_candies = 22 ∧ four_colors = 4 ∧ yellow_most ∧ 
                       (∀ i ≥ 0, 0 ≤ turns i ≤ 2 ∧ turns i ≤ total_candies) ∧ 
                       (equal_split 11 11) → 
                       8 ≤ n ∧ n ≤ 16) : 
yellow_bounds 16 := 
sorry

end yellow_candies_bounds_l218_218577


namespace definite_integral_x_from_1_to_2_l218_218225

theorem definite_integral_x_from_1_to_2 :
  (∫ x in 1..2, x) = (3 / 2) :=
begin
  sorry
end

end definite_integral_x_from_1_to_2_l218_218225


namespace max_min_sum_l218_218311

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x) * Real.sin (x - 1) + x + 1

theorem max_min_sum (M m : ℝ) (hM : ∀ x ∈ set.Icc (-1:ℝ) 3, f x ≤ M) 
                    (hm : ∀ x ∈ set.Icc (-1:ℝ) 3, f x ≥ m) :
  M + m = 4 :=
sorry

end max_min_sum_l218_218311


namespace area_of_triangle_ABC_l218_218508

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (2, 2)
def C : Point := (2, 0)

def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_ABC :
  triangle_area A B C = 2 :=
by
  sorry

end area_of_triangle_ABC_l218_218508


namespace sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218244

theorem sum_of_four_digit_numbers_formed_by_digits_1_to_5 :
  let S := {1, 2, 3, 4, 5}
  let four_digits_sum (n1 n2 n3 n4 : ℕ) :=
    1000 * n1 + 100 * n2 + 10 * n3 + n4
  (∀ a b c d ∈ S, a ≠ b → b ≠ c → c ≠ d → d ≠ a → a ≠ c → b ≠ d 
  → sum (four_digits_sum a b c d) = 399960) := sorry

end sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218244


namespace brenda_initial_points_l218_218995

theorem brenda_initial_points
  (b : ℕ)  -- points scored by Brenda in her play
  (initial_advantage :ℕ := 22)  -- Brenda is initially 22 points ahead
  (david_score : ℕ := 32)  -- David scores 32 points
  (final_advantage : ℕ := 5)  -- Brenda is 5 points ahead after both plays
  (h : initial_advantage + b - david_score = final_advantage) :
  b = 15 :=
by
  sorry

end brenda_initial_points_l218_218995


namespace abc_gt_16_abc_geq_3125_div_108_l218_218487

variables {a b c α β : ℝ}

-- Define the conditions
def conditions (a b c α β : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b > 0 ∧
  (a * α^2 + b * α - c = 0) ∧
  (a * β^2 + b * β - c = 0) ∧
  (α ≠ β) ∧
  (α^3 + b * α^2 + a * α - c = 0) ∧
  (β^3 + b * β^2 + a * β - c = 0)

-- State the first proof problem
theorem abc_gt_16 (h : conditions a b c α β) : a * b * c > 16 :=
sorry

-- State the second proof problem
theorem abc_geq_3125_div_108 (h : conditions a b c α β) : a * b * c ≥ 3125 / 108 :=
sorry

end abc_gt_16_abc_geq_3125_div_108_l218_218487


namespace maximum_side_length_and_waste_fraction_of_dodecagon_l218_218556

/-- Given an equilateral triangle with side length 20 cm, the maximum side length of a regular dodecagon inscribed in this triangle is approximately 5.1764 cm, and the fraction of the triangle that becomes waste is approximately 0.381. --/
theorem maximum_side_length_and_waste_fraction_of_dodecagon
  (side_triangle : ℝ) (side_dodecagon : ℝ) (waste_fraction : ℝ) 
  (h_eq_triangle : side_triangle = 20) 
  (h_eq_dodecagon : side_dodecagon ≈ 5.1764)
  (h_eq_waste_fraction : waste_fraction ≈ 0.381) :
  ∃ (side_dodecagon : ℝ) (waste_fraction : ℝ),
    side_dodecagon ≈ 5.1764 ∧ waste_fraction ≈ 0.381 :=
by
  sorry

end maximum_side_length_and_waste_fraction_of_dodecagon_l218_218556


namespace reduced_price_per_dozen_l218_218957

theorem reduced_price_per_dozen (P : ℝ) (H : 0.4 * P * (64 / 12) = 40) : 0.6 * P = 7.5 :=
by
  sorry

end reduced_price_per_dozen_l218_218957


namespace percentageReduction_l218_218939

variable (R P : ℝ)

def originalPrice (R : ℝ) (P : ℝ) : Prop :=
  2400 / R - 2400 / P = 8 ∧ R = 120

theorem percentageReduction : 
  originalPrice 120 P → ((P - 120) / P) * 100 = 40 := 
by
  sorry

end percentageReduction_l218_218939


namespace product_of_primes_l218_218076

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l218_218076


namespace hao_hao_age_in_2016_l218_218780

open Nat

noncomputable def haoHaoAge : ℕ :=
  let currentYear := 2015
  let experiencedLeapYears := 2
  let birthYearMultipleOfNine := True
  currentYear + 1 - (2008 + 1) * experiencedLeapYears

theorem hao_hao_age_in_2016
  (currentYear : ℕ = 2015)
  (experiencedLeapYears : ℕ = 2)
  (birthYearMultipleOfNine : ∃ n, n * 9 = 2007)
  : haoHaoAge = 9 :=  
sorry

end hao_hao_age_in_2016_l218_218780


namespace spheres_homothetic_l218_218011

noncomputable def sphere (center : ℝ^3) (radius : ℝ) : Set ℝ^3 :=
  {x | (x - center).norm = radius}

theorem spheres_homothetic (O1 O2 : ℝ^3) (R1 R2 : ℝ) (hR : R1 > R2) :
  ∃ S : ℝ^3, ∃ k : ℝ, ∀ A1 ∈ sphere O1 R1,
  ∃ A2 ∈ sphere O2 R2, (A2 - S) = k • (A1 - S) ∧ (k = R2 / R1 ∨ k = - (R2 / R1)) :=
begin
  sorry
end

end spheres_homothetic_l218_218011


namespace product_of_smallest_primes_l218_218090

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218090


namespace product_of_primes_l218_218101

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l218_218101


namespace total_sales_commission_l218_218959

theorem total_sales_commission :
  ∃ S : ℝ,
    (S ≤ 5000 → S - 0.10 * S = 15000) ∨
    (S > 5000 → S - (500 + 0.05 * (S - 5000)) = 15000) ∧
    S = 16052.63 :=
by {
  existsi 16052.63,
  split,
  { intro h,
    left,
    intro hs,
    have : S - 0.10 * S = 15000 := hs,
    sorry,
  },
  { intro h,
    right,
    intro hs,
    have : S - (500 + 0.05 * (S - 5000)) = 15000 := hs,
    sorry,
  },
}

end total_sales_commission_l218_218959


namespace repeating_decimal_to_fraction_l218_218913

theorem repeating_decimal_to_fraction : (∃ (x : ℚ), x = 0.4 + 4 / 9) :=
sorry

end repeating_decimal_to_fraction_l218_218913


namespace determine_f_750_l218_218829

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f(x * y) = f(x) / y
axiom f_at_1000 : f(1000) = 4

theorem determine_f_750 : f(750) = 16 / 3 := 
by
  sorry

end determine_f_750_l218_218829


namespace line_equation_l218_218030

theorem line_equation
  (P : ℝ × ℝ) (hP : P = (1, -1))
  (h_perp : ∀ x y : ℝ, 3 * x - 2 * y = 0 → 2 * x + 3 * y = 0):
  ∃ m : ℝ, (2 * P.1 + 3 * P.2 + m = 0) ∧ m = 1 :=
by
  sorry

end line_equation_l218_218030


namespace roots_of_polynomial_in_arithmetic_progression_l218_218892

theorem roots_of_polynomial_in_arithmetic_progression :
  ∃ (a b c : ℚ), (a < b) ∧ (b < c) ∧ (b - a = c - b) ∧
  (Polynomial.eval a (Polynomial.of_nat_list [3, -2, -1, 1]) = 0) ∧
  (Polynomial.eval b (Polynomial.of_nat_list [3, -2, -1, 1]) = 0) ∧
  (Polynomial.eval c (Polynomial.of_nat_list [3, -2, -1, 1]) = 0) ∧
  (Set.ofList [a, b, c] = Set.ofList [-1/2, 1/2, 3/2]) := by
sorry

end roots_of_polynomial_in_arithmetic_progression_l218_218892


namespace tree_label_permutation_cycle_l218_218648

theorem tree_label_permutation_cycle (n : ℕ) (T : Tree (Fin n)) 
  (initial_labels : Fin n → Fin n) 
  (swap_operation : (Fin n × Fin n) → (Fin n → Fin n) → Fin n → Fin n)
  (final_labels : Fin n → Fin n)
  (edges : List (Fin n × Fin n))
  (h1 : edges.length = n - 1)
  (h2 : ∀ (i : ℕ) (h : i < edges.length), final_labels = swap_operation (edges.nth_le i h) final_labels) :
  ∃ (cycle : List (Fin n)), final_labels.permute_cycle = cycle ∧ cycle.length = n :=
begin
  sorry
end

end tree_label_permutation_cycle_l218_218648


namespace abc_inequality_l218_218391

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) :=
sorry

end abc_inequality_l218_218391


namespace log_abs_is_even_l218_218219

-- Define the function g(x)
def g (x : ℝ) : ℝ := Real.log (|x|)

-- State the theorem that proves g(x) is an even function
theorem log_abs_is_even : ∀ x : ℝ, g(x) = g(-x) := by
  intro x
  dsimp [g]
  rw [abs_neg]
  sorry

end log_abs_is_even_l218_218219


namespace inscribed_circle_radius_of_isosceles_triangle_l218_218233

noncomputable def triangleInradius 
  (AB AC BC : ℝ) 
  (hAB : AB = 8)
  (hAC : AC = 8)
  (hBC : BC = 10) : ℝ := 
  let s := (AB + AC + BC) / 2 
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem inscribed_circle_radius_of_isosceles_triangle :
  ∀ (AB AC BC : ℝ), 
  AB = 8 → 
  AC = 8 → 
  BC = 10 → 
  triangleInradius AB AC BC AB AC BC = 5 * Real.sqrt 15 / 13 :=
by intros AB AC BC hAB hAC hBC
  simp only [triangleInradius, hAB, hAC, hBC]
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  change K / s = 5 * Real.sqrt 15 / 13
  sorry

end inscribed_circle_radius_of_isosceles_triangle_l218_218233


namespace smallest_number_of_cubes_to_hide_snaps_l218_218954

def cube :=
{ protruding_snap : bool,
  receptacle_holes : ℕ }

def is_connected (c1 c2 : cube) : Prop :=
(c1.protruding_snap = false) ∧ (c2.protruding_snap = false)

def all_snaps_hidden (configuration : list cube) : Prop :=
∀ c ∈ configuration, c.protruding_snap = false

def only_receptacle_holes_showing (configuration : list cube) : Prop :=
∀ c ∈ configuration, c.receptacle_holes = 5

def valid_configuration (configuration : list cube) : Prop :=
configuration.length = 4 ∧
∀ (c1 c2 : cube), c1 ∈ configuration → c2 ∈ configuration → is_connected c1 c2

theorem smallest_number_of_cubes_to_hide_snaps :
  ∀ (cubes : list cube),
  (all_snaps_hidden cubes ∧ only_receptacle_holes_showing cubes) →
  configuration.length = 4 :=
by
  sorry

end smallest_number_of_cubes_to_hide_snaps_l218_218954


namespace alicia_art_left_l218_218183

-- Definition of the problem conditions.
def initial_pieces : ℕ := 70
def donated_pieces : ℕ := 46

-- The theorem to prove the number of art pieces left is 24.
theorem alicia_art_left : initial_pieces - donated_pieces = 24 := 
by
  sorry

end alicia_art_left_l218_218183


namespace sum_of_remainders_mod_53_l218_218744

theorem sum_of_remainders_mod_53 (d e f : ℕ) (hd : d % 53 = 19) (he : e % 53 = 33) (hf : f % 53 = 14) : 
  (d + e + f) % 53 = 13 :=
by
  sorry

end sum_of_remainders_mod_53_l218_218744


namespace piles_to_single_pile_l218_218429

-- Define the condition similar_sizes
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

-- Define the inductive step of combining stones
def combine_stones (piles : List ℕ) : List ℕ :=
  if ∃ x y, x ∈ piles ∧ y ∈ piles ∧ similar_sizes x y then
    let ⟨x, hx, y, hy, hsim⟩ := Classical.some_spec (Classical.some_spec_exists _)
    List.cons (x + y) (List.erase (List.erase piles x) y)
  else
    piles

-- Prove that a collection of piles can be reduced to a single pile of size n
theorem piles_to_single_pile (piles : List ℕ) (h : ∀ x ∈ piles, x = 1) : 
  ∃ p, list.length (Iterator.iterate combine_stones piles.count) 1 = 1 := by
  sorry

end piles_to_single_pile_l218_218429


namespace little_D_can_accomplish_goal_for_all_n_l218_218650

theorem little_D_can_accomplish_goal_for_all_n :
  ∀ n : ℕ,
  (∃ points : fin n → ℤ × ℤ × ℤ, 
    (∀ i j, i ≠ j → points i ≠ points j) ∧ 
    (∃ axis : fin 3, ∀ i, points i axis = points 0 axis + i) ∧
    (∀ planes : fin n → fin 3 × ℤ, 
      ∃ i, ¬ (points i ∈ planes ∧ ∀ j, planes j ≠ axis))),
sorry

end little_D_can_accomplish_goal_for_all_n_l218_218650


namespace series_sum_eq_1_div_400_l218_218645

theorem series_sum_eq_1_div_400 :
  (∑' n : ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 := 
sorry

end series_sum_eq_1_div_400_l218_218645


namespace product_of_primes_l218_218100

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l218_218100


namespace the_inequality_l218_218482

theorem the_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a / (1 + b)) + (b / (1 + c)) + (c / (1 + a)) ≥ 3 / 2 :=
by sorry

end the_inequality_l218_218482


namespace sufficient_b_not_necessary_l218_218308

theorem sufficient_b_not_necessary (b : ℝ) : (|complex.mk (√3) b| = 2) ↔ (b = 1 ∨ b = -1) :=
by {
  sorry
}

end sufficient_b_not_necessary_l218_218308


namespace proofExpression_l218_218641

noncomputable def calcExpression : ℝ :=
  |(-3)| - ((Real.sqrt 10) - 1) ^ 0 + Real.sqrt 2 * Real.cos (Real.pi / 4) + (1 / 4) ^ (-1)

theorem proofExpression :
  calcExpression = 7 := by
  sorry

end proofExpression_l218_218641


namespace part1_part2_l218_218318

noncomputable def f (x : ℝ) : ℝ :=
  4 * sin (π / 3 - x) * cos (π / 6 - x) + 4 * real.sqrt 3 * cos x * cos (π / 2 + x)

theorem part1 :
  ∀ k : ℤ, ∃ interval : set ℝ, interval = set.Icc (-π / 6 + k * π) (k * π + π / 3) ∧ 
  (∀ x ∈ interval, f x = 4 * cos (2 * x + π / 3) + 1 ∧ (∀ y ∈ interval, x < y → f y < f x)) :=
by sorry

noncomputable def A : ℝ := π / 3

theorem part2 {a b c : ℝ} (h1 : f A = -3) (h2 : b + c = 2 * a) (h3 : a = c):
  ∠B = π / 3 :=
by sorry

end part1_part2_l218_218318


namespace probability_interval_l218_218769

theorem probability_interval (x : ℝ) (h₁ : x ∈ set.Ioc 0 (1 / 2)) : 
  probability (x < (1 / 3) | x ∈ set.Ioc 0 (1 / 2)) = 2 / 3 :=
sorry

end probability_interval_l218_218769


namespace find_b_value_l218_218882

theorem find_b_value (b : ℝ) : (∃ (x y : ℝ), (x, y) = ((2 + 4) / 2, (5 + 9) / 2) ∧ x + y = b) ↔ b = 10 :=
by
  sorry

end find_b_value_l218_218882


namespace product_of_primes_is_66_l218_218128

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l218_218128


namespace proportion_relationship_l218_218876

-- Define the distance on the map as a fixed constant
constant d_map : ℝ

-- Define actual distance and scale as two real numbers
variables (d_actual scale : ℝ)

-- Define the condition that their product is the fixed distance on the map
def product_fixed := d_actual * scale = d_map

-- Theorem: The actual distance and the scale are not in direct proportion
theorem proportion_relationship (h : product_fixed d_actual scale) : ¬ (∃ k : ℝ, d_actual = k * scale) :=
sorry

end proportion_relationship_l218_218876


namespace find_m_l218_218356

theorem find_m (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (4 / x) + (9 / y) = m) (h4 : ∃ x y , x + y = 5/6) : m = 30 :=
sorry

end find_m_l218_218356


namespace first_percentage_increase_l218_218889

variable (P : ℝ)  -- The original price of the TV
variable (x : ℝ)  -- The first percentage increase

-- Formalize the conditions
def first_increase (P : ℝ) (x : ℝ) : ℝ :=
  P * (1 + x / 100)

def second_increase (price : ℝ) : ℝ :=
  price * 1.20

def single_increase (P : ℝ) : ℝ :=
  P * (1 + 56.00000000000001 / 100)

-- Goal: Prove the equivalence between the successive increases and the single increase
theorem first_percentage_increase :
  second_increase (first_increase P x) = single_increase P → x = 30 := 
by
  sorry

end first_percentage_increase_l218_218889


namespace product_of_primes_is_66_l218_218126

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l218_218126


namespace imaginary_part_of_z_is_1_over_5_l218_218295

-- Define the imaginary unit and the complex number z
noncomputable def z (a : ℝ) : ℂ := 1 / (a - complex.I)

-- Declare the main theorem stating the given conditions lead to the imaginary part of z being 1/5
theorem imaginary_part_of_z_is_1_over_5 :
  ∀ a : ℝ, (∀ x y : ℝ, (x = a / (a^2 + 1) ∧ y = 1 / (a^2 + 1) ∧ x - 2 * y = 0)) → (complex.im (z a) = 1 / 5) :=
by
  sorry

end imaginary_part_of_z_is_1_over_5_l218_218295


namespace prove_angle_BFD_l218_218760

def given_conditions (A : ℝ) (AFG AGF : ℝ) : Prop :=
  A = 40 ∧ AFG = AGF

theorem prove_angle_BFD (A AFG AGF BFD : ℝ) (h1 : given_conditions A AFG AGF) : BFD = 110 :=
  by
  -- Utilize the conditions h1 stating that A = 40 and AFG = AGF
  sorry

end prove_angle_BFD_l218_218760


namespace reject_null_hypothesis_proof_l218_218554

noncomputable def sample_size_1 : ℕ := 14
noncomputable def sample_size_2 : ℕ := 10
noncomputable def sample_variance_x : ℝ := 0.84
noncomputable def sample_variance_y : ℝ := 2.52
noncomputable def significance_level : ℝ := 0.1

-- Normal populations X and Y
def NormalPopulation_X : Type := sorry
def NormalPopulation_Y : Type := sorry

-- Hypotheses
def null_hypothesis : Prop := σ NormalPopulation_X ^ 2 = σ NormalPopulation_Y ^ 2
def alternative_hypothesis : Prop := σ NormalPopulation_X ^ 2 ≠ σ NormalPopulation_Y ^ 2

-- F-statistic for comparing variances
noncomputable def F_statistic : ℝ := sample_variance_y / sample_variance_x

-- Degrees of freedom
noncomputable def df_numerator : ℕ := sample_size_2 - 1
noncomputable def df_denominator : ℕ := sample_size_1 - 1

-- Critical value at alpha / 2 = 0.05 for two-tailed test
noncomputable def F_critical_value : ℝ := 2.72 -- approximate value

-- Test criterion
def reject_null_hypothesis : Prop := F_statistic > F_critical_value

-- Goal: Prove or disprove the null hypothesis given the conditions
theorem reject_null_hypothesis_proof : reject_null_hypothesis → alternative_hypothesis :=
by
  -- Proof omitted.
  sorry

end reject_null_hypothesis_proof_l218_218554


namespace circle_with_half_points_inside_half_outside_l218_218700

theorem circle_with_half_points_inside_half_outside 
  (n : ℕ) 
  (h : n ≥ 1) 
  (points : set (euclidean_space 2 ℝ))
  (h_card : points.card = 2 * n + 3) 
  (h_no_three_collinear : ∀ (p1 p2 p3 : euclidean_space 2 ℝ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬ collinear ℝ {p1, p2, p3})
  (h_no_four_concyclic : ∀ (p1 p2 p3 p4 : euclidean_space 2 ℝ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p4 → ¬ concyclic ℝ {p1, p2, p3, p4})
  : ∃ (A B C : euclidean_space 2 ℝ), 
      A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
      ∃ circle : circle ℝ, 
        circle.through A ∧ circle.through B ∧ circle.through C ∧ 
        let remaining_points := points \ {A, B, C} in
        (remaining_points.filter (λ p, inside_circle circle p)).card = n ∧
        (remaining_points.filter (λ p, ¬ inside_circle circle p)).card = n :=
by sorry

end circle_with_half_points_inside_half_outside_l218_218700


namespace piles_can_be_combined_l218_218441

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l218_218441


namespace problem_solution_l218_218652

noncomputable def first_sequence (n : ℕ) : ℕ := 5 * n - 2
noncomputable def second_sequence (m : ℕ) : ℕ := 10 * m + 5

noncomputable def first_2500_terms_seq1 : finset ℕ := finset.image first_sequence (finset.range 2500)
noncomputable def first_2500_terms_seq2 : finset ℕ := finset.image second_sequence (finset.range 2500)

noncomputable def union_set : finset ℕ := first_2500_terms_seq1 ∪ first_2500_terms_seq2

theorem problem_solution : finset.card union_set = 3750 :=
sorry

end problem_solution_l218_218652


namespace drain_lake_in_3_6_hours_l218_218479

theorem drain_lake_in_3_6_hours :
  let R1 := 1 / 9
      R2 := 1 / 6
      R_combined := R1 + R2
      t := 1 / R_combined
  in t = 3.6 :=
by 
  -- Definitions and setup
  let R1 := (1:ℚ) / 9
  let R2 := (1:ℚ) / 6
  let R_combined := R1 + R2
  let t := 1 / R_combined
  -- Proof placeholder
  sorry

end drain_lake_in_3_6_hours_l218_218479


namespace negation_existential_proposition_l218_218885

theorem negation_existential_proposition :
  ¬(∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
by sorry

end negation_existential_proposition_l218_218885


namespace profit_function_and_maximum_profit_l218_218165

noncomputable def G (x : ℝ) : ℝ := 2.8 + x

def R (x : ℝ) : ℝ :=
  if (0 ≤ x ∧ x ≤ 5) then (-0.4 * x^2 + 4.2 * x)
  else 11

noncomputable def f (x : ℝ) : ℝ :=
  if (0 ≤ x ∧ x ≤ 5) then (-0.4 * x^2 + 3.2 * x - 2.8)
  else (8.2 - x)

theorem profit_function_and_maximum_profit :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 5 → f(x) = -0.4 * x^2 + 3.2 * x - 2.8) ∧
    (x > 5 → f(x) = 8.2 - x)) ∧
  (∃ x : ℝ, (x = 4 ∧ f(x) = 3.6)) :=
by
  sorry

end profit_function_and_maximum_profit_l218_218165


namespace problem1_problem2_l218_218207

theorem problem1 : |(-3 : ℝ)| + (\pi + 1) ^ 0 - real.sqrt 9 + real.cbrt 8 = 3 := 
  sorry

theorem problem2 : (-2 * real.sqrt 2) ^ 2 + real.sqrt 24 * real.sqrt (1/2) + |real.sqrt 3 - 2| = 10 + real.sqrt 3 := 
  sorry

end problem1_problem2_l218_218207


namespace rhombus_perimeter_greater_than_circle_circumference_l218_218009

theorem rhombus_perimeter_greater_than_circle_circumference :
  let d1 := 1 in
  let d2 := 3 in
  let r := 1 in
  let a := (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) in
  let P := 4 * a in
  let C := 2 * Real.pi * r in
  P > C :=
by
  let d1 := 1
  let d2 := 3
  let r := 1
  let a := (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))
  let P := 4 * a
  let C := 2 * Real.pi * r
  have h1 : a = (Real.sqrt ((1 / 2 : ℝ) ^ 2 + (3 / 2 : ℝ) ^ 2)), by rfl
  rw h1 at *
  have h2 : (1 / 2 : ℝ) ^ 2 = 1 / 4, by norm_num
  have h3 : (3 / 2 : ℝ) ^ 2 = 9 / 4, by norm_num
  rw [h2, h3] at *
  let b := (Real.sqrt ((1 / 4 : ℝ) + (9 / 4 : ℝ)))
  let a := b
  have h4 : b = Real.sqrt (10 / 4), by norm_num1
  rw h4 at *
  let c := Real.sqrt (10) / 2
  have h5 : a = c, by norm_num1
  rw h5 at *
  let P := 4 * c
  let C := 2 * Real.pi * 1
  have h6 : C = 2 * Real.pi, by norm_num1
  rw h6 at *
  have h7 : 2 * Real.pi < 2 * Real.sqrt 10, -- by computation or known fact
    sorry
  exact h7

end rhombus_perimeter_greater_than_circle_circumference_l218_218009


namespace stability_comparison_l218_218069

-- Definitions of conditions
def variance_A : ℝ := 3
def variance_B : ℝ := 1.2

-- Definition of the stability metric
def more_stable (performance_A performance_B : ℝ) : Prop :=
  performance_B < performance_A

-- Target Proposition
theorem stability_comparison (h_variance_A : variance_A = 3)
                            (h_variance_B : variance_B = 1.2) :
  more_stable variance_A variance_B = true :=
by
  sorry

end stability_comparison_l218_218069


namespace rectangular_to_polar_l218_218994
noncomputable def polar_coord_equiv (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

noncomputable def polar_form (ρ θ : ℝ) : Prop :=
  ρ = 6 * cos θ

theorem rectangular_to_polar (x y ρ θ : ℝ) :
  polar_coord_equiv x y ↔ (x = ρ * cos θ) ∧ (y = ρ * sin θ) ∧ polar_form ρ θ :=
by
  sorry

end rectangular_to_polar_l218_218994


namespace max_sequence_length_l218_218787

theorem max_sequence_length (a : ℕ → ℝ) (n : ℕ)
  (H1 : ∀ k : ℕ, k + 4 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4)) < 0)
  (H2 : ∀ k : ℕ, k + 8 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8)) > 0) : 
  n ≤ 12 :=
sorry

end max_sequence_length_l218_218787


namespace ellipse_and_area_proof_l218_218714

noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) : Prop := 
  (λ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_and_area_proof (P : ℝ × ℝ) (hP : P = (-1, sqrt 2 / 2)) 
  (E : ellipse_equation sqrt 2 1) :
  E (-1) (sqrt 2 / 2) ∧
  ∀ (t : ℝ), t^2 = 1 / 3 →
  let C := (λ (y : ℝ), ((t * y + 1, y))) in 
  let D := (λ (y : ℝ), ((t * y + 1, y))) in 
  let y_3 := (-2 * t / (t^2 + 2)) in 
  let y_4 := (-1 / (t^2 + 2)) in 
  let height := sqrt (8 * (1 + t^2)) / (t^2 + 2) in
  abs ((sqrt 2 + sqrt 2) * height / 2) = 4 * sqrt 6 / 7 := 
sorry

end ellipse_and_area_proof_l218_218714


namespace cistern_emptying_time_l218_218158

theorem cistern_emptying_time :
  ∀ (T : ℝ), (1 / 4) - (1 / T) = (1 / 7.2) → T = 9 :=
begin
  intros T h,
  sorry, -- proof to be filled in later
end

end cistern_emptying_time_l218_218158


namespace value_of_expr_l218_218562

theorem value_of_expr : (365^2 - 349^2) / 16 = 714 := by
  sorry

end value_of_expr_l218_218562


namespace find_probability_l218_218823

noncomputable def Q (x : ℝ) : ℝ := 2 * x^2 - 5 * x - 20

def valid_probability (x : ℝ) := 
  (2 ≤ x ∧ x ≤ 10) ∧ (floor (sqrt (Q x)) = sqrt (Q (floor x)))

def probability_in_interval (a b : ℝ) := 
  ((b - a) / (10 - 2))

theorem find_probability :
  (∑ x in Ico 2 10, if valid_probability x then 1 else 0) / (10 - 2) = (sqrt 385 - 19) / 32 :=
sorry

end find_probability_l218_218823


namespace option_C_incorrect_l218_218136

def p (x : ℝ) : Prop := x^2 + x + 1 > 0
def q (x : ℝ) : Prop := x^2 + x + 1 ≤ 0
def f (x : ℝ) : Prop := x^2 - 3*x + 2 = 0
def s (x : ℝ) : Prop := x = 1
def r (a b c : ℝ) : Prop := ac^2 < bc^2
def t (a b : ℝ) : Prop := a < b

theorem option_C_incorrect (a b c : ℝ) :
  (r a b c → t a b) → ¬ (t a b → r a b c) := sorry

end option_C_incorrect_l218_218136


namespace strictly_increasing_on_unit_interval_l218_218733

theorem strictly_increasing_on_unit_interval (a : ℝ) :
  (∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), deriv (λ x, x^3 - 2 * x^2 + a * x + 3) x > 0) ↔ a ≥ 1 := 
sorry

end strictly_increasing_on_unit_interval_l218_218733


namespace red_candy_ratio_l218_218901

theorem red_candy_ratio :
  (∀ (bagA bagB bagC : ℕ) (rA rB rC : ℚ),
    bagA = 27 ∧ rA = 1/3 ∧
    bagB = 36 ∧ rB = 1/4 ∧
    bagC = 45 ∧ rC = 1/5 →
    let total_red_candies := (rA * bagA) + (rB * bagB) + (rC * bagC) in
    let total_candies := bagA + bagB + bagC in
    total_red_candies / total_candies = 1/4) :=
sorry

end red_candy_ratio_l218_218901


namespace square_area_l218_218185

theorem square_area (x : ℚ) (h : 3 * x - 12 = 15 - 2 * x) : (3 * (27 / 5) - 12)^2 = 441 / 25 :=
by
  sorry

end square_area_l218_218185


namespace sum_four_digit_numbers_l218_218250

def digits : List ℕ := [1, 2, 3, 4, 5]

/-- 
  Prove that the sum of all four-digit numbers that can be formed 
  using the digits 1, 2, 3, 4, 5 exactly once is 399960.
-/
theorem sum_four_digit_numbers : 
  (Finset.sum 
    (Finset.map 
      (λ l, 
        l.nth_le 0 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1000 + 
        l.nth_le 1 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 100 + 
        l.nth_le 2 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 10 + 
        l.nth_le 3 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1) 
      (digits.permutations.filter (λ l, l.nodup ∧ l.length = 4))) id) 
  = 399960 :=
sorry

end sum_four_digit_numbers_l218_218250


namespace retail_price_correct_l218_218616

variable (m : ℝ) (a b : ℝ)
-- a and b are percentages, converting them to their decimal equivalents
def retail_price_after_adjustment (m : ℝ) (a b : ℝ) : ℝ := m * (1 + a / 100) * (b / 100)

theorem retail_price_correct : retail_price_after_adjustment m a b = m * (1 + a / 100) * (b / 100) :=
by
  sorry

end retail_price_correct_l218_218616


namespace harmonics_inequality_l218_218860

theorem harmonics_inequality (n : ℕ) (h : n > 0) : 
  (\sum k in finset.range n, 1 / (n + k : ℝ)) ≥ n * (real.exp ((real.log 2) / n) - 1) :=
sorry

end harmonics_inequality_l218_218860


namespace swimming_pool_width_l218_218538

theorem swimming_pool_width (length width vol depth : ℝ) 
  (H_length : length = 60) 
  (H_depth : depth = 0.5) 
  (H_vol_removal : vol = 2250 / 7.48052) 
  (H_vol_eq : vol = (length * width) * depth) : 
  width = 10.019 :=
by
  -- Assuming the correctness of floating-point arithmetic for the purpose of this example
  sorry

end swimming_pool_width_l218_218538


namespace triangle_MNK_is_obtuse_l218_218878

open EuclideanGeometry

variables (A B C D K M N : Point) (AB CD : Line)
variables (h1 : ConvexQuadrilateral A B C D)
           (h2 : extension_of AB A B = K)
           (h3 : extension_of CD C D = K)
           (h4 : M = midpoint A B)
           (h5 : N = midpoint C D)
           (h6 : dist A D = dist B C)

theorem triangle_MNK_is_obtuse : ∃ θ : Angle, Triangle_is_obtuse (MK θ) ∧ (MN θ) := sorry

end triangle_MNK_is_obtuse_l218_218878


namespace sum_of_all_four_digit_numbers_l218_218234

-- Let us define the set of digits
def digits : set ℕ := {1, 2, 3, 4, 5}

-- We will define a function that generates the four-digit numbers
def four_digit_numbers := {n : ℕ // ∃ a b c d : ℕ, 
                                      a ∈ digits ∧ 
                                      b ∈ digits ∧ 
                                      c ∈ digits ∧ 
                                      d ∈ digits ∧ 
                                      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ 
                                      b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                                      n = 1000 * a + 100 * b + 10 * c + d}

-- Define a function to calculate the sum of all elements in a set of numbers
def sum_set (s : set ℕ) : ℕ := s.fold (λa b, a + b) 0

theorem sum_of_all_four_digit_numbers :
  sum_set {n | ∃ x : four_digit_numbers, x.val = n} = 399960 :=
sorry

end sum_of_all_four_digit_numbers_l218_218234


namespace bus_speed_including_stoppages_l218_218666

noncomputable def speedIncludingStoppages (speedWithoutStoppages : ℕ) (stoppageTimeMinutes : ℕ) : ℕ :=
  let stoppageTimeHours := stoppageTimeMinutes / 60
  let distanceLost := stoppageTimeHours * speedWithoutStoppages
  let distanceTraveled := speedWithoutStoppages - distanceLost
  distanceTraveled

theorem bus_speed_including_stoppages:
  speedIncludingStoppages 50 9.6 = 42 := sorry

end bus_speed_including_stoppages_l218_218666


namespace sum_four_digit_numbers_l218_218259

theorem sum_four_digit_numbers : 
  let digits := [1, 2, 3, 4, 5]
  let perms := digits.permutations
  ∑ p in perms.filter (λ x, x.length = 4), (1000 * x.head + 100 * x[1] + 10 * x[2] + x[3]) = 399960 := 
by sorry

end sum_four_digit_numbers_l218_218259


namespace pile_of_stones_l218_218411

def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

theorem pile_of_stones (n : ℕ) (f : ℕ → ℕ): (∀ i, 1 ≤ f i ∧ f i ≤ n) → 
  (∀ j k, similar_sizes (f j) (f k)) → True :=
by
  simp
  exact true.intro


end pile_of_stones_l218_218411


namespace positive_integer_m_l218_218731

theorem positive_integer_m (m x : ℕ) (h : m > 0 ∧ x > 0 ∧ (m ≠ 1) ∧ (x ≠ 8)) :
  (mx / (x - 8) = (4m + x) / (x - 8)) → (m = 3 ∨ m = 5) :=
by sorry

end positive_integer_m_l218_218731


namespace solution_set_of_inequality_l218_218893

theorem solution_set_of_inequality (x : ℝ) : 
  (1 / x ≤ 1 ↔ (0 < x ∧ x < 1) ∨ (1 ≤ x)) :=
  sorry

end solution_set_of_inequality_l218_218893


namespace problem_statement_l218_218293

open Nat

theorem problem_statement (S : ℕ → ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → 2 * S n = 3 * a n - 2) ∧
  (b n = log 3 (S n + 1)) ∧
  (∀ n : ℕ, S n = 3 ^ n - 1) ∧
  (∀ n : ℕ, a n = 2 * 3 ^ (n - 1)) ∧
  (∀ n : ℕ, b (2 * n) = 2 * n) ∧
  (T n = ∑ i in range (n+1), b (2 * i)) →
  T n = n ^ 2 + n :=
by
  sorry

end problem_statement_l218_218293


namespace similar_triangles_side_length_l218_218875

theorem similar_triangles_side_length (A1 A2 : ℕ) (k : ℕ)
  (h1 : A1 - A2 = 32)
  (h2 : A1 = k^2 * A2)
  (h3 : A2 > 0)
  (side2 : ℕ) (h4 : side2 = 5) :
  ∃ side1 : ℕ, side1 = 3 * side2 ∧ side1 = 15 :=
by
  sorry

end similar_triangles_side_length_l218_218875


namespace words_with_B_at_least_once_is_61_l218_218332

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E'}

-- Define the total number of 3-letter words that can be formed from the letters A, B, C, D, and E
def total_words : ℕ := 5^3

-- Define the number of 3-letter words that can be formed excluding B
def words_excluding_B : ℕ := 4^3

-- Define the number of 3-letter words that include B at least once
def words_including_B_at_least_once : ℕ := total_words - words_excluding_B

-- Statement to be proved
theorem words_with_B_at_least_once_is_61 : words_including_B_at_least_once = 61 :=
by
  sorry

end words_with_B_at_least_once_is_61_l218_218332


namespace number_of_integer_points_satisfying_condition_l218_218888

def parabola_focus : Point := (0, 0)
def parabola_points : list Point := [(6, 2), (-6, -2)]

def satisfies_condition (p : Point) : Prop :=
  let (x, y) := p in
  |4 * x + 3 * y| ≤ 1200

def is_on_parabola (p : Point) : Prop :=
  -- placeholder for the actual parabola condition determined by the focus and points
  sorry

theorem number_of_integer_points_satisfying_condition :
  {p : Point | is_on_parabola p ∧ satisfies_condition p}.finite.card = 758 :=
sorry

end number_of_integer_points_satisfying_condition_l218_218888


namespace places_proven_l218_218460

-- Definitions based on the problem conditions
inductive Place
| first
| second
| third
| fourth

def is_boy : String -> Prop
| "Oleg" => True
| "Olya" => False
| "Polya" => False
| "Pasha" => False
| _ => False

def name_starts_with_O : String -> Prop
| n => (n.head! = 'O')

noncomputable def determine_places : Prop :=
  ∃ (olegs_place olyas_place polyas_place pashas_place : Place),
  -- Statements and truth conditions
  ∃ (truthful : String), truthful ∈ ["Oleg", "Olya", "Polya", "Pasha"] ∧ 
  ∀ (person : String), 
    (person ≠ truthful → ∀ (statement : Place -> Prop), ¬ statement (person_to_place person)) ∧
    (person = truthful → person_to_place person = Place.first) ∧
    (person = truthful → 
      match person with
        | "Olya" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → is_boy (place_to_person p)
        | "Oleg" => ∃ (p : Place), (person_to_place "Oleg" = p ∧ person_to_place "Olya" = succ_place p ∨ 
                                    person_to_place "Olya" = p ∧ person_to_place "Oleg" = succ_place p)
        | "Pasha" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → name_starts_with_O (place_to_person p)
        | _ => True
      end)

-- Helper functions to relate places to persons
def person_to_place : String -> Place
| "Oleg" => Place.first
| "Olya" => Place.second
| "Polya" => Place.third
| "Pasha" => Place.fourth
| _ => Place.first -- Default, shouldn't happen

def place_to_person : Place -> String
| Place.first => "Oleg"
| Place.second => "Olya"
| Place.third => "Polya"
| Place.fourth => "Pasha"

def succ_place : Place → Place
| Place.first => Place.second
| Place.second => Place.third
| Place.third => Place.fourth
| Place.fourth => Place.first -- No logical next in this context.

theorem places_proven : determine_places :=
by
  sorry

end places_proven_l218_218460


namespace arthur_walks_distance_l218_218632

variables (blocks_east blocks_north : ℕ) 
variable (distance_per_block : ℝ)
variable (total_blocks : ℕ)
def total_distance (blocks : ℕ) (distance_per_block : ℝ) : ℝ :=
  blocks * distance_per_block

theorem arthur_walks_distance (h_east : blocks_east = 8) (h_north : blocks_north = 10) 
    (h_total_blocks : total_blocks = blocks_east + blocks_north)
    (h_distance_per_block : distance_per_block = 1 / 4) :
  total_distance total_blocks distance_per_block = 4.5 :=
by {
  -- Here we specify the proof, but as required, we use sorry to skip it.
  sorry
}

end arthur_walks_distance_l218_218632


namespace radius_increase_l218_218815

variable {r_orig r_new : ℝ}
variable (dist_orig dist_new : ℝ)
variable (r_increase : ℝ)

def initial_radius := 16 -- inches
def initial_distance := 1000 -- miles
def new_distance := 980 -- miles
def original_circumference := 2 * Real.pi * initial_radius -- inches
def miles_per_rotation_orig := original_circumference / 62560 -- inches to miles
def rotations_orig := initial_distance / miles_per_rotation_orig

theorem radius_increase :
  rotations_orig = new_distance / (2 * Real.pi * r_new / 62560)
  → dist_orig = initial_distance
  → dist_new = new_distance
  → r_new = (initial_distance / new_distance) * initial_radius
  → r_increase = r_new - initial_radius
  → r_increase = 0.33 :=
by
  sorry

end radius_increase_l218_218815


namespace base4_division_quotient_l218_218226

theorem base4_division_quotient : 
  let a := (2 * 4^3 + 0 * 4^2 + 3 * 4^1 + 3 * 4^0)
      b := (2 * 4^1 + 2 * 4^0)
  in a / b = 1 * 4^1 + 1 * 4^0 :=
by
  sorry

end base4_division_quotient_l218_218226


namespace secret_known_by_people_day_when_secret_is_known_by_people_l218_218475

-- Useful imports for working with exponentiation, geometric series, and natural numbers 

theorem secret_known_by_people (n : ℕ) : 
  (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7) = 3280 :=
by
  sorry
  
theorem day_when_secret_is_known_by_people : 
  ∃ (n : ℕ), 3280 = 1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 ∧ n = 7 :=
by
  exists n
  apply secret_known_by_people
  intro h
  exact h
  sorry
end

end secret_known_by_people_day_when_secret_is_known_by_people_l218_218475


namespace pie_remaining_portion_l218_218988

theorem pie_remaining_portion (carlos_portion maria_portion remaining_portion : ℝ)
  (h1 : carlos_portion = 0.6) 
  (h2 : remaining_portion = 1 - carlos_portion)
  (h3 : maria_portion = 0.5 * remaining_portion) :
  remaining_portion - maria_portion = 0.2 := 
by
  sorry

end pie_remaining_portion_l218_218988


namespace tennis_tournament_l218_218789

theorem tennis_tournament (n : ℕ) (h : ∃ (n : ℕ), let t := 4 * n in (n * (4 * n - 1)) mod 4 = 0) : False := 
by {
  let total_matches := 4 * n * (4 * n - 1) / 2,
  let total_wins := 8 * x,
  have h_mem : total_matches = total_wins,
  exact (mod_eq_zero_of_dvd (show 4 | (total_wins), from 
    begin
      -- integer solutions verification
      sorry
    end
  )),
  sorry
}

end tennis_tournament_l218_218789


namespace rocco_total_money_l218_218855

def piles_of_quarters := 4
def piles_of_dimes := 6
def piles_of_nickels := 9
def piles_of_pennies := 5

def coins_per_pile := 10

def value_of_quarter := 0.25
def value_of_dime := 0.10
def value_of_nickel := 0.05
def value_of_penny := 0.01

def total_amount :=
  (coins_per_pile * piles_of_quarters * value_of_quarter) +
  (coins_per_pile * piles_of_dimes * value_of_dime) +
  (coins_per_pile * piles_of_nickels * value_of_nickel) +
  (coins_per_pile * piles_of_pennies * value_of_penny)

theorem rocco_total_money : total_amount = 21.00 := 
  sorry

end rocco_total_money_l218_218855


namespace range_of_m_l218_218735

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then 2^x + 1 else 1 - Real.logb 2 x

theorem range_of_m (m : ℝ) : 
  (f (1 - m^2) > f (2 * m - 2)) ↔ (-3 < m ∧ m < 1) ∨ (m > 3 / 2) := sorry

end range_of_m_l218_218735


namespace incircle_tangent_proof_l218_218608

theorem incircle_tangent_proof
  (A B C D E F I G H K L : Point) -- Points mentioned in the problem
  (hABCtangents : ∀ (X : Set Point), X = { D, E, F } → is_incident (A, B, C) X) -- Tangency points
  (hIncenter : is_incenter I A B C) -- I is the incenter of triangle ABC
  (hAD_incircle : ∃ G, incident (A, D) G ∧ is_on_incircle G (A, B, C)) -- AD hits the incircle at G
  (hGTangentH : tangent_at G H (A, C) (incircle A B C)) -- Tangent at G hits AC at H
  (hIKHAD : intersection_line (IH) (AD) = K) -- IH ∩ AD = K
  (hLPerpendicular : perpendicular_from I (AD) = L) -- Foot of perpendicular from I to AD is L
  : dist I E * dist I K = dist I C * dist I L := sorry

end incircle_tangent_proof_l218_218608


namespace complex_quadrant_l218_218337

def z : ℂ := (-8 + Complex.i) * Complex.i

theorem complex_quadrant : 
  let point := (z.re, z.im)
  point.1 < 0 ∧ point.2 < 0 :=
by 
  -- The proof is omitted as per the instruction
  sorry

end complex_quadrant_l218_218337


namespace pastries_more_than_cakes_l218_218634

def cakes_made : ℕ := 19
def pastries_made : ℕ := 131

theorem pastries_more_than_cakes : pastries_made - cakes_made = 112 :=
by {
  -- Proof will be inserted here
  sorry
}

end pastries_more_than_cakes_l218_218634


namespace transform_sine_wave_l218_218804

theorem transform_sine_wave (λ μ : ℝ) (hλ : λ = 3) (hμ : μ = 1/2) :
  (∀ (x y x' y' : ℝ), y = 2 * (sin (3*x)) ∧ x' = λ * x ∧ y' = μ * y → y' = sin x') :=
by sorry

end transform_sine_wave_l218_218804


namespace prove_correct_conclusions_l218_218737

def f (x : ℝ) : ℝ :=
if x ∈ (1/2, 1] then x / (x + 2)
else if x ∈ [0, 1/2] then - (1/2) * x + (1 / 4)
else 0 -- This additional case is to satisfy Lean's total function requirement

def g (a : ℝ) (x : ℝ) : ℝ :=
a * sin (π/3 * x + 3 * π / 2) - 2 * a + 2

-- Condition a > 0
axiom a_pos : a > 0

-- The main theorem statement
theorem prove_correct_conclusions (a : ℝ) :
  (∀ x ∈ (1/2, 1), f x ∈ [0, 1/3]) ∧
  (∃ x1 x2 ∈ [0,1], f x1 = g a x2 → (5/9 ≤ a ∧ a ≤ 4/5)) ∧
  (∀ x ∈ [0,1], g a x is_increasing) :=
sorry

end prove_correct_conclusions_l218_218737


namespace arithmetic_sequence_general_formula_and_geometric_condition_l218_218727

theorem arithmetic_sequence_general_formula_and_geometric_condition :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ} {k : ℕ}, 
    (∀ n, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)) →
    a 1 = 9 →
    S 3 = 21 →
    a 5 * S k = a 8 ^ 2 →
    k = 5 :=
by 
  intros a S k hS ha1 hS3 hgeom
  sorry

end arithmetic_sequence_general_formula_and_geometric_condition_l218_218727


namespace cat_weight_l218_218625

theorem cat_weight 
  (weight1 weight2 : ℕ)
  (total_weight : ℕ)
  (h1 : weight1 = 2)
  (h2 : weight2 = 7)
  (h3 : total_weight = 13) : 
  ∃ weight3 : ℕ, weight3 = 4 := 
by
  sorry

end cat_weight_l218_218625


namespace sum_four_digit_numbers_l218_218248

def digits : List ℕ := [1, 2, 3, 4, 5]

/-- 
  Prove that the sum of all four-digit numbers that can be formed 
  using the digits 1, 2, 3, 4, 5 exactly once is 399960.
-/
theorem sum_four_digit_numbers : 
  (Finset.sum 
    (Finset.map 
      (λ l, 
        l.nth_le 0 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1000 + 
        l.nth_le 1 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 100 + 
        l.nth_le 2 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 10 + 
        l.nth_le 3 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1) 
      (digits.permutations.filter (λ l, l.nodup ∧ l.length = 4))) id) 
  = 399960 :=
sorry

end sum_four_digit_numbers_l218_218248


namespace frog_escape_probability_l218_218788

def P : ℕ → ℚ

def P_def (N : ℕ) : ℚ :=
  if N = 0 then 0
  else if N = 12 then 1
  else if 0 < N ∧ N < 12 then (approx (N / 12) * P (N - 1)) + ((1 - (approx (N / 12))) * P (N + 1))
  else 0

theorem frog_escape_probability :
  P_def 2 = (149 / 364) := 
sorry

end frog_escape_probability_l218_218788


namespace product_of_primes_l218_218080

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l218_218080


namespace radical_expressions_equal_iff_l218_218148

theorem radical_expressions_equal_iff (a b c : ℤ) :
  (sqrt (a + (b / c)) = a * sqrt (b / c)) ↔ (c = (b * (a^2 - 1)) / a) := 
sorry

end radical_expressions_equal_iff_l218_218148


namespace actual_gain_percent_l218_218949

-- Define the problem parameters
def initial_price : ℝ := 100
def discount : ℝ := 0.10
def sales_tax_on_discounted_price : ℝ := 0.05
def sold_price_including_tax : ℝ := 130
def sales_tax_on_sold_price : ℝ := 0.15

-- Define the goal as a Lean theorem
theorem actual_gain_percent :
  let discounted_price := initial_price * (1 - discount)
  let total_paid := discounted_price * (1 + sales_tax_on_discounted_price)
  let pre_tax_sold_price := sold_price_including_tax / (1 + sales_tax_on_sold_price)
  let gain := pre_tax_sold_price - total_paid
  let gain_percent := (gain / total_paid) * 100
  gain_percent ≈ 19.62 :=
by
  -- Proof goes here
  sorry

end actual_gain_percent_l218_218949


namespace product_of_smallest_primes_l218_218120

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218120


namespace pentagon_area_ratio_l218_218782

theorem pentagon_area_ratio (s : ℝ) (h : s = 2) (A B C D P Q R S T U V W : ℝ) 
  (midpoint_C : C = (V + W) / 2) (midpoint_D : D = (W + P) / 2) :
  (let area_pentagon := 6 in  -- From the solution steps part
   let area_three_squares := 3 * s^2 in
   area_pentagon / area_three_squares = 1 / 2) :=
by
  exact sorry

end pentagon_area_ratio_l218_218782


namespace probability_magnitude_P_eq_2_l218_218536

noncomputable def V : set ℂ := {1, -1, complex.I, -complex.I, real.sqrt 2, -real.sqrt 2, (real.sqrt 2) * complex.I, -(real.sqrt 2) * complex.I}

noncomputable def z_j (j : fin 10) : ℂ := classical.some (classical.some_spec (set.infinite.exists_elim_finset V j))

theorem probability_magnitude_P_eq_2 : 
  (∃ P : ℂ, P = ∏ j in finset.finrange 10, z_j j ∧ abs P = 2) → 
  ∑' (h : finset.finrange 10 ↝ V) in (λ (f : finset.finrange 10 ↝ V), abs (∏ j in finset.finrange 10, f j) = 2), 
  1 / (finset.card V) ^ 10 = 27 / 134217728 :=
sorry

end probability_magnitude_P_eq_2_l218_218536


namespace competition_places_l218_218468

def participants := ["Olya", "Oleg", "Polya", "Pasha"]
def placements := Array.range 1 5

-- Define statements made by each child
def Olya_claims_odd_boys (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → (name = "Oleg" ∨ name = "Pasha")

def Oleg_claims_consecutive_with_Olya (placement : String → Nat) : Prop :=
  abs (placement "Oleg" - placement "Olya") = 1

def Pasha_claims_odd_O_names (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → name.startsWith "O"

-- Define the main problem statement
theorem competition_places :
  ∃ (placement : String → Nat),
    placement "Oleg" = 1 ∧
    placement "Olya" = 2 ∧
    placement "Polya" = 3 ∧
    placement "Pasha" = 4 ∧
    (∃ name, (name = "Oleg" ∨ name = "Olya" ∨ name = "Polya" ∨ name = "Pasha") ∧
      ((name = "Oleg" → (placement "Oleg" = 1 ∧ Oleg_claims_consecutive_with_Olya placement)) ∧
       (name = "Olya" → (placement "Olya" = 1 ∧ Olya_claims_odd_boys placement)) ∧
       (name = "Pasha" → (placement "Pasha" = 1 ∧ Pasha_claims_odd_O_names placement)))) :=
by
  have placement : String → Nat := λ name => match name with
    | "Olya" => 2
    | "Oleg" => 1
    | "Polya" => 3
    | "Pasha" => 4
    | _      => 0
  use placement
  simp [placement, Oleg_claims_consecutive_with_Olya, Olya_claims_odd_boys, Pasha_claims_odd_O_names]
  sorry

end competition_places_l218_218468


namespace max_value_of_reciprocal_eccentricities_l218_218726

-- Definitions based on given conditions
structure Ellipse (a1 b1 : ℝ) := 
(h1 : a1 > b1)
(h2 : b1 > 0)

structure Hyperbola (a2 b2 : ℝ) := 
(h1 : a2 > 0)
(h2 : b2 > 0)

structure Point2D (x y : ℝ)

def eccentricity (c a : ℝ) : ℝ := c / a

variable (a1 b1 a2 b2 c : ℝ)

axiom intersection_point_P : Point2D

axiom intersection_of_ellipse_and_hyperbola_at_P :
  (intersection_point_P.x ^ 2) / (a1 ^ 2) + (intersection_point_P.y ^ 2) / (b1 ^ 2) = 1 ∧
  (intersection_point_P.x ^ 2) / (a2 ^ 2) - (intersection_point_P.y ^ 2) / (b2 ^ 2) = 1

axiom common_foci_F1_F2 : Point2D × Point2D

axiom angle_F1PF2 : ∠ common_foci_F1_F2.fst intersection_point_P common_foci_F1_F2.snd = 2 * Real.pi / 3

def e1 := eccentricity c a1
def e2 := eccentricity c a2

theorem max_value_of_reciprocal_eccentricities :
  ∃ (e1 e2 : ℝ), (3 / e1 ^ 2 + 1 / e2 ^ 2 = 4) ∧ (1 / e1 + 1 / e2 ≤ 4 * Real.sqrt 3 / 3) ∧ 
  (1 / e1 + 1 / e2 = 4 * Real.sqrt 3 / 3) :=
sorry

end max_value_of_reciprocal_eccentricities_l218_218726


namespace product_of_primes_is_66_l218_218129

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l218_218129


namespace regression_slope_l218_218304

theorem regression_slope (x y : ℕ → ℕ) (h1 : ∑ i in finset.range 10, x i = 17)
  (h2 : ∑ i in finset.range 10, y i = 4) :
  let x_bar := (∑ i in finset.range 10, x i) / 10
      y_bar := (∑ i in finset.range 10, y i) / 10 in
  -3 + 2 * x_bar = y_bar :=
by
  sorry

end regression_slope_l218_218304


namespace InfinitePairsExist_l218_218852

theorem InfinitePairsExist (a b : ℕ) : (∀ n : ℕ, ∃ a b : ℕ, a ∣ b^2 + 1 ∧ b ∣ a^2 + 1) :=
sorry

end InfinitePairsExist_l218_218852


namespace choir_members_l218_218513

theorem choir_members (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 11 = 6) (h3 : 200 ≤ n ∧ n ≤ 300) :
  n = 220 :=
sorry

end choir_members_l218_218513


namespace sum_of_digits_of_x_l218_218609

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem sum_of_digits_of_x
  (x : ℕ)
  (h1 : 100 ≤ x ∧ x < 1000) -- x is a three-digit number
  (h2 : is_palindrome x) -- x is a palindrome
  (h3 : is_palindrome (x + 45)) -- x + 45 is a palindrome
  : x.digits.sum = 17 := sorry

end sum_of_digits_of_x_l218_218609


namespace slope_of_line_through_midpoints_l218_218931

theorem slope_of_line_through_midpoints :
  let P₁ := (1, 2)
  let P₂ := (3, 8)
  let P₃ := (4, 3)
  let P₄ := (7, 9)
  let M₁ := ( (P₁.1 + P₂.1)/2, (P₁.2 + P₂.2)/2 )
  let M₂ := ( (P₃.1 + P₄.1)/2, (P₃.2 + P₄.2)/2 )
  let slope := (M₂.2 - M₁.2) / (M₂.1 - M₁.1)
  slope = 2/7 :=
by
  sorry

end slope_of_line_through_midpoints_l218_218931


namespace max_possible_area_of_T_l218_218785

-- Define the radii of the four circles
def r1 := 2
def r2 := 4
def r3 := 6
def r4 := 10

-- Define the condition that the sum of the radii of the smaller circles is not less than the radius of the largest circle
def sum_radii_condition := r1 + r2 + r3 >= r4

-- Define the formula for the area of a circle
def area (r : ℝ) : ℝ := π * r^2

-- Define region T as the area inside exactly one of these four circles
def region_T_area : ℝ := (area r4) + (area r3) - (area r2)

-- State the theorem to be proven: The maximum possible area of region T
theorem max_possible_area_of_T (h : sum_radii_condition) : region_T_area = 120 * π :=
by
  sorry

end max_possible_area_of_T_l218_218785


namespace product_of_smallest_primes_l218_218121

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218121


namespace initial_roses_in_vase_l218_218060

theorem initial_roses_in_vase (current_roses : ℕ) (added_roses : ℕ) (total_garden_roses : ℕ) (initial_roses : ℕ) :
  current_roses = 20 → added_roses = 13 → total_garden_roses = 59 → initial_roses = current_roses - added_roses → 
  initial_roses = 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  sorry

end initial_roses_in_vase_l218_218060


namespace piles_can_be_combined_l218_218439

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l218_218439


namespace part_1_part_2_part_3_l218_218867

-- Define given conditions

def y (x: ℝ) : ℝ := -10 * x + 900
def cost_price : ℝ := 50
def selling_price_not_exceed : ℝ := 75
def monthly_profit_at_least : ℝ := 3000

-- Define monthly profit as a function of selling price x

def W (x: ℝ) : ℝ := (-10 * x + 900) * (x - cost_price)

-- Prove the expression of monthly profit W in terms of x

theorem part_1 : ∀ x: ℝ, W x = -10 * x^2 + 1400 * x - 45000 :=
by {
  intro x,
  calc
    W x = (-10 * x + 900) * (x - cost_price) : sorry
    ... = -10 * x^2 + 1400 * x - 45000         : sorry
}

-- Prove the selling price that maximizes monthly profit

theorem part_2 : ∃ x: ℝ, W x = 4000 ∧ x = 70 :=
by {
  use 70,
  split,
  calc
    W 70 = -10 * 70^2 + 1400 * 70 - 45000 : sorry
    ... = 4000                            : sorry,
  exact rfl
}

-- Prove the range of selling price x given the constraints

theorem part_3 : ∀ x: ℝ, (60 ≤ x ∧ x ≤ 75) → W x ≥ monthly_profit_at_least :=
by {
  intros x hx,
  have h1 := hx.1,
  have h2 := hx.2,
  have h3 : W x ≥ 3000, from sorry,
  exact h3
}

end part_1_part_2_part_3_l218_218867


namespace fraction_German_speakers_French_speakers_l218_218146

theorem fraction_German_speakers_French_speakers 
    (x : ℕ) -- total number of German speakers
    (hx1 : ∀ x > 0, 1/6 * x > 0) -- assumption: number of German speakers who speak English
    (hx2 : ∀ x > 0, 1/3 * (x/2) = x/6) -- assumption: number of English speakers who speak German
    (hx3 : ∀ x > 0, 1/5 * (x/2) = x/10) -- assumption: number of English speakers who speak French
    (hx4 : ∀ x > 0, 1/8 * (x/2) = x/10) -- assumption: number of French speakers who speak English
    (hx5 : ∀ x > 0, 1/2 * (4*x / 5) = 2*x/5) -- assumption: number of French speakers who speak German
    : (∀ x > 0, 2*x/5 = (2/5) * x) :=
begin
  sorry
end

end fraction_German_speakers_French_speakers_l218_218146


namespace shortest_segment_length_l218_218586

theorem shortest_segment_length :
  let total_length := 1
  let red_dot := 0.618
  let yellow_dot := total_length - red_dot  -- yellow_dot is at the same point after fold
  let first_cut := red_dot  -- Cut the strip at the red dot
  let remaining_strip := red_dot
  let distance_between_red_and_yellow := total_length - 2 * yellow_dot
  let second_cut := distance_between_red_and_yellow
  let shortest_segment := remaining_strip - 2 * distance_between_red_and_yellow
  shortest_segment = 0.146 :=
by
  sorry

end shortest_segment_length_l218_218586


namespace order_of_numbers_l218_218521

theorem order_of_numbers (a b c : ℝ) (h1 : a = 6^0.5) (h2 : b = 0.5^6) (h3 : c = Real.log 6 / Real.log 0.5) : 
  c < b ∧ b < a :=
by {
  have h4 : a > 1, from sorry,
  have h5 : 0 < b ∧ b < 1, from sorry,
  have h6 : c < 0, from sorry,
  exact ⟨h6, h5.2.trans h4⟩,
}

end order_of_numbers_l218_218521


namespace obtuse_and_acute_angles_in_convex_octagon_l218_218783

theorem obtuse_and_acute_angles_in_convex_octagon (m n : ℕ) (h₀ : n + m = 8) : m > n :=
sorry

end obtuse_and_acute_angles_in_convex_octagon_l218_218783


namespace power_calculation_l218_218749

theorem power_calculation (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 5) : a^(3*m + 2*n) = 200 := by
  sorry

end power_calculation_l218_218749


namespace find_y_when_x_4_l218_218048

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l218_218048


namespace cannot_form_quadrilateral_l218_218138

theorem cannot_form_quadrilateral
  (a b c d : ℝ)
  (h1 : a = 16)
  (h2 : b = 13)
  (h3 : c = 10)
  (h4 : d = 6)
  (h5 : a ∥ c) : ¬∃ (quadrilateral : Type) (Q : quadrilateral), 
  (Q has_sides [a, b, c, d] ∧ a ∥ c) :=
sorry

end cannot_form_quadrilateral_l218_218138


namespace mean_and_variance_of_transformed_data_l218_218706

variable {n : ℕ}
variable {x : Fin n → ℝ}

-- Conditions: Mean and variance of the original dataset
def mean (x : Fin n → ℝ) : ℝ := (∑ i, x i) / n
def variance (x : Fin n → ℝ) : ℝ := (∑ i, (x i - mean x) ^ 2) / n

theorem mean_and_variance_of_transformed_data (hmean : mean x = 2) (hvar : variance x = 3) :
  mean (λ i, 2 * x i + 5) = 9 ∧ variance (λ i, 2 * x i + 5) = 12 := 
by
  sorry

end mean_and_variance_of_transformed_data_l218_218706


namespace condition_on_y_existence_of_r_l218_218287

-- Define the necessary conditions
variables {a b p : ℕ} (prime_p : Nat.Prime p) (coprime_abp : Nat.gcd a b = 1 ∧ Nat.gcd a p = 1 ∧ Nat.gcd b p = 1)
variables (n : ℕ)

-- T is defined as { x | x = a + n * b, n ∈ {0, 1, ..., p-1} }
def T := {x | ∃ n : ℕ, n < p ∧ x = a + n * b}

-- Proposition 1
theorem condition_on_y (y : ℕ) (ht : ∀ (i j ∈ T), i ≠ j → y ^ i + i % p ≠ y ^ j + j % p) :
  p ∣ y ∨ y ^ b ≡ 1 [MOD p] :=
sorry

-- Proposition 2
theorem existence_of_r (y t : ℕ) :
  ∃ r ∈ {r | ∃ n : ℕ, r = a + n * b}, (y ^ r + r) % p = t % p :=
sorry

end condition_on_y_existence_of_r_l218_218287


namespace probability_of_red_ball_l218_218791

theorem probability_of_red_ball :
  let total_balls := 9
  let red_balls := 6
  let probability := (red_balls : ℚ) / total_balls
  probability = (2 : ℚ) / 3 :=
by
  sorry

end probability_of_red_ball_l218_218791


namespace impossible_to_use_up_parts_l218_218546

-- Define the requirements for each product
structure ProductRequirements :=
  (a_parts : ℕ)
  (b_parts : ℕ)
  (c_parts : ℕ)

-- Define the initial stock and leftover conditions
structure Inventory :=
  (initial_a : ℕ)
  (initial_b : ℕ)
  (initial_c : ℕ)
  (leftover_a : ℕ)
  (leftover_b : ℕ)

def initial_inventory_conditions (p q r : ℕ) (inv : Inventory) : Prop :=
  inv.leftover_a = inv.initial_a - (2 * p + 2 * r) ∧
  inv.leftover_b = inv.initial_b - (2 * p + q) ∧
  inv.leftover_c = 0 ∧
  inv.initial_c = q + r

-- Define the modification condition
def adjusted_inventory_conditions (p q r x y z : ℕ) (inv : Inventory) : Prop :=
  inv.initial_a = 2 * x + 2 * z ∧
  inv.initial_b = 2 * x + y ∧
  inv.initial_c = y + z

theorem impossible_to_use_up_parts (p q r x y z : ℕ) (inv : Inventory) 
  (h1 : initial_inventory_conditions p q r inv)
  (h2 : adjusted_inventory_conditions p q r x y z inv) : 
  False :=
by sorry

end impossible_to_use_up_parts_l218_218546


namespace minimum_time_to_cook_3_pancakes_l218_218910

theorem minimum_time_to_cook_3_pancakes (can_fry_two_pancakes_at_a_time : Prop) 
   (time_to_fully_cook_one_pancake : ℕ) (time_to_cook_one_side : ℕ) :
  can_fry_two_pancakes_at_a_time →
  time_to_fully_cook_one_pancake = 2 →
  time_to_cook_one_side = 1 →
  3 = 3 := 
by
  intros
  sorry

end minimum_time_to_cook_3_pancakes_l218_218910


namespace volume_of_new_parallelepiped_l218_218537

variables (u v w : ℝ^3)
hypothesis vol_original : abs (u • (v × w)) = 6

theorem volume_of_new_parallelepiped 
  (u v w : ℝ^3)
  (vol_original : abs (u • (v × w)) = 6) : 
  abs ((2 • u + 3 • v) • ((v + 2 • w) × (w + 5 • u))) = 78 :=
sorry

end volume_of_new_parallelepiped_l218_218537


namespace product_of_primes_l218_218108

def smallest_one_digit_prime := 2
def second_smallest_one_digit_prime := 3
def smallest_two_digit_prime := 11

theorem product_of_primes: smallest_one_digit_prime * second_smallest_one_digit_prime * smallest_two_digit_prime = 66 :=
by {
  -- Applying the definition of the primes and carrying out the multiplication
  show 2 * 3 * 11 = 66,
  calc
  2 * 3 * 11 = 6 * 11 : by rw [mul_assoc 2 3 11]
          ... = 66    : by norm_num,
}

end product_of_primes_l218_218108


namespace product_of_primes_l218_218082

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l218_218082


namespace piles_can_be_reduced_l218_218400

/-! 
  We define similar sizes as the difference between sizes being at most a factor of two.
  Given any number of piles of stones, we aim to prove that these piles can be combined 
  iteratively into one single pile.
-/

def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

theorem piles_can_be_reduced (n : ℕ) :
  ∃ pile : ℕ, (pile = n) ∧ (∀ piles : list ℕ, list.sum piles = n → 
    (∃ piles' : list ℕ, list.sum piles' = n ∧ list.length piles' = 1)) :=
by
  -- Placeholder for the proof.
  sorry

end piles_can_be_reduced_l218_218400


namespace fraction_product_108_l218_218914

theorem fraction_product_108 : (1/2 : ℚ) * (1/3) * (1/6) * 108 = 3 := by
  sorry

end fraction_product_108_l218_218914


namespace similar_sizes_combination_possible_l218_218405

theorem similar_sizes_combination_possible 
    (similar : Nat → Nat → Prop := λ x y, x ≤ y ∧ y ≤ 2 * x)
    (combine_piles : List Nat → Nat ∃ combined : Nat, (∀ x y ∈ combined, similar x y) → True
    (piles : List Nat) : True :=
sorry

end similar_sizes_combination_possible_l218_218405


namespace find_breadth_of_rectangular_field_l218_218026

theorem find_breadth_of_rectangular_field
  (l : ℝ) (area_square_plot : ℝ) (d : ℝ) (b : ℝ)
  (length_rect_field : l = 90)
  (diagonal_square_plot : d = 120)
  (area_def : area_square_plot = (90 : ℝ) * b)
  (side_square : sqrt (2 : ℝ) * s = d)
  (squared_area : area_square_plot = s^2) :
  b = 80 :=
by
  sorry

end find_breadth_of_rectangular_field_l218_218026


namespace john_spending_l218_218373

def times_per_month := 3
def boxes_per_time := 3
def cost_per_box := 25
def maintenance_cost_per_month := 40
def travel_cost_per_trip := 10
def discount_rate := 0.15

def total_spending_per_month (t b p m r d : ℕ) : ℕ :=
  let cost_of_paintballs := t * b * p
  let total_travel_cost := t * r
  let total_cost := cost_of_paintballs + m + total_travel_cost
  if b * t ≥ 12 then
    total_cost - (cost_of_paintballs * d).to_nat
  else
    total_cost

theorem john_spending : total_spending_per_month times_per_month boxes_per_time cost_per_box maintenance_cost_per_month travel_cost_per_trip discount_rate = 295 :=
by
  sorry

end john_spending_l218_218373


namespace partners_count_l218_218604

theorem partners_count (P A : ℕ) (h1 : P / A = 2 / 63) (h2 : P / (A + 50) = 1 / 34) : P = 20 :=
sorry

end partners_count_l218_218604


namespace zero_in_interval_3_4_l218_218539

-- Definitions
def f (x : ℝ) : ℝ := log x / log 2 - 7 / x

-- Theorem Statement
theorem zero_in_interval_3_4 : ∃ x : ℝ, 3 < x ∧ x < 4 ∧ f x = 0 :=
by {
  -- Given conditions
  have f_continuous : ∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x₁ x₂, abs (x₁ - x₂) < δ → abs (f x₁ - f x₂) < ε := sorry,
  have f3_less_than_zero : f 3 < 0 := by norm_num,
  have f4_greater_than_zero : f 4 > 0 := by norm_num,
  
  sorry  -- proof not required
}

end zero_in_interval_3_4_l218_218539


namespace order_of_numbers_l218_218519

theorem order_of_numbers :
  let a := 6 ^ 0.5
  let b := 0.5 ^ 6
  let c := Real.log 6 / Real.log 0.5
  c < b ∧ b < a :=
by
  sorry

end order_of_numbers_l218_218519


namespace piles_can_be_reduced_l218_218396

/-! 
  We define similar sizes as the difference between sizes being at most a factor of two.
  Given any number of piles of stones, we aim to prove that these piles can be combined 
  iteratively into one single pile.
-/

def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

theorem piles_can_be_reduced (n : ℕ) :
  ∃ pile : ℕ, (pile = n) ∧ (∀ piles : list ℕ, list.sum piles = n → 
    (∃ piles' : list ℕ, list.sum piles' = n ∧ list.length piles' = 1)) :=
by
  -- Placeholder for the proof.
  sorry

end piles_can_be_reduced_l218_218396


namespace concurrency_of_lines_l218_218707

-- Define the structures: a triangle and the points O1, O2, and O3
variables {A B C D E O1 O2 O3 : EuclideanGeometry.Point}
           [EuclideanGeometry.Triangle A B C]

-- Point O1 is defined as the center of rectangle BCDE with DE containing A
-- Points O2 and O3 are similar centers of rectangles on sides AC and AB
-- respectively.
axiom O1_definition : EuclideanGeometry.IsCenter (EuclideanGeometry.Rectangle B C D E) O1 ∧ EuclideanGeometry.LiesOn A (EuclideanGeometry.Line D E)
axiom O2_definition : EuclideanGeometry.IsCenter (EuclideanGeometry.Rectangle A C D E) O2
axiom O3_definition : EuclideanGeometry.IsCenter (EuclideanGeometry.Rectangle A B D E) O3

-- Prove that lines AO1, BO2, and CO3 intersect at a single point
theorem concurrency_of_lines :
  EuclideanGeometry.Concurrent (EuclideanGeometry.LineThrough A O1) (EuclideanGeometry.LineThrough B O2) (EuclideanGeometry.LineThrough C O3) :=
by sorry

end concurrency_of_lines_l218_218707


namespace cost_of_45_daffodils_equals_75_l218_218196

-- Conditions
def cost_of_15_daffodils : ℝ := 25
def number_of_daffodils_in_bouquet_15 : ℕ := 15
def number_of_daffodils_in_bouquet_45 : ℕ := 45
def directly_proportional (n m : ℕ) (c_n c_m : ℝ) : Prop := c_n / n = c_m / m

-- Statement to prove
theorem cost_of_45_daffodils_equals_75 :
  ∀ (c : ℝ), directly_proportional number_of_daffodils_in_bouquet_45 number_of_daffodils_in_bouquet_15 c cost_of_15_daffodils → c = 75 :=
by
  intro c hypothesis
  -- Proof would go here.
  sorry

end cost_of_45_daffodils_equals_75_l218_218196


namespace find_m_range_l218_218742

variable {x m : ℝ}

def p : Prop := abs(x - 1) < 2
def q : Prop := -1 < x ∧ x < m + 1

theorem find_m_range (hp : p) (hq : q) (hsuff : ∀ x, p x → q x ∧ ¬q x → p x) : m > 2 :=
sorry

end find_m_range_l218_218742


namespace pile_division_possible_l218_218420

theorem pile_division_possible (n : ℕ) :
  ∃ (division : list ℕ), (∀ x ∈ division, x = 1) ∧ division.sum = n :=
by
  sorry

end pile_division_possible_l218_218420


namespace y_when_x_is_4_l218_218056

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l218_218056


namespace num_pairs_satisfying_eq_l218_218752

theorem num_pairs_satisfying_eq : 
    { n : Nat // ∃ (pairs : Finset (ℤ × ℤ)), pairs.card = n ∧ ∀ (x y : ℤ), (x, y) ∈ pairs ↔ x + y = 3 * x * y - 1 } = 2 :=
by
  sorry

end num_pairs_satisfying_eq_l218_218752


namespace number_of_speedster_convertibles_l218_218938

def proof_problem (T : ℕ) :=
  let Speedsters := 2 * T / 3
  let NonSpeedsters := 50
  let TotalInventory := NonSpeedsters * 3
  let SpeedsterConvertibles := 4 * Speedsters / 5
  (Speedsters = 2 * TotalInventory / 3) ∧ (SpeedsterConvertibles = 4 * Speedsters / 5)

theorem number_of_speedster_convertibles : proof_problem 150 → ∃ (x : ℕ), x = 80 :=
by
  -- Provide the definition of Speedsters, NonSpeedsters, TotalInventory, and SpeedsterConvertibles
  sorry

end number_of_speedster_convertibles_l218_218938


namespace piles_can_be_combined_l218_218437

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l218_218437


namespace sum_cosine_inequality_l218_218684

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {R : ℝ}

-- Ensure the triangle is acute.
axiom acute_triangle (ABC : Triangle) : ABC.isAcute

-- Main statement of the problem
theorem sum_cosine_inequality (ABC : Triangle) (h : ABC.isAcute) :
  (1 / (cos A + cos B) + 1 / (cos B + cos C) + 1 / (cos C + cos A)) >= 
  (1 / 2 * (a + b + c) * (1 / a + 1 / b + 1 / c) - 3 / 2) :=
  sorry

end sum_cosine_inequality_l218_218684


namespace sufficient_condition_for_increasing_seq_l218_218881

def is_increasing_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) > a n

def abs_seq (c : ℝ) (n : ℕ) : ℝ :=
  |(n : ℝ) - c|

theorem sufficient_condition_for_increasing_seq (c : ℝ) (c_le_one : c ≤ 1) :
  is_increasing_seq (abs_seq c) := 
begin
  sorry
end

end sufficient_condition_for_increasing_seq_l218_218881


namespace meeting_point_l218_218371

def same_start (x : ℝ) (y : ℝ) : Prop := x = y

def walk_time (x : ℝ) (y : ℝ) (t : ℝ) : Prop := 
  x * t + y * t = 24

def hector_speed (s : ℝ) : ℝ := s

def jane_speed (s : ℝ) : ℝ := 3 * s

theorem meeting_point (s t : ℝ) :
  same_start 0 0 ∧ walk_time (hector_speed s) (jane_speed s) t → t = 6 / s ∧ (6 : ℝ) = 6 :=
by
  intros h
  sorry

end meeting_point_l218_218371


namespace goods_amount_decreased_initial_goods_amount_total_fees_l218_218941

-- Define the conditions as variables
def tonnages : List Int := [31, -31, -16, 34, -38, -20]
def final_goods : Int := 430
def fee_per_ton : Int := 5

-- Prove that the amount of goods in the warehouse has decreased
theorem goods_amount_decreased : (tonnages.sum < 0) := by
  sorry

-- Prove the initial amount of goods in the warehouse
theorem initial_goods_amount : (final_goods + tonnages.sum = 470) := by
  sorry

-- Prove the total loading and unloading fees
theorem total_fees : (tonnages.map Int.natAbs).sum * fee_per_ton = 850 := by
  sorry

end goods_amount_decreased_initial_goods_amount_total_fees_l218_218941


namespace square_of_difference_of_solutions_l218_218387

theorem square_of_difference_of_solutions :
  let d e : ℝ := by
    have h : 5 * polynomial.x^2 + 20 * polynomial.x - 55 = 0 := sorry
    let sol := polynomial.roots h
    exact (sol.head, sol.tail.head)
  (d - e)^2 = 600 := sorry

end square_of_difference_of_solutions_l218_218387


namespace present_value_of_machine_l218_218948

theorem present_value_of_machine (r : ℝ) (t : ℕ) (V : ℝ) (P : ℝ) (h1 : r = 0.10) (h2 : t = 2) (h3 : V = 891) :
  V = P * (1 - r)^t → P = 1100 :=
by
  intro h
  rw [h3, h1, h2] at h
  -- The steps to solve for P are omitted as instructed
  sorry

end present_value_of_machine_l218_218948


namespace smallest_f_l218_218819

-- Define separating and covering conditions
def is_separating (N : Finset ℕ) (F : Finset (Finset ℕ)) : Prop :=
  ∀ {x y : ℕ}, x ∈ N → y ∈ N → x ≠ y → ∃ A ∈ F, (A ∩ {x, y}).card = 1

def is_covering (N : Finset ℕ) (F : Finset (Finset ℕ)) : Prop :=
  ∀ x ∈ N, ∃ A ∈ F, x ∈ A

-- Define the main function f(n)
def f (n : ℕ) : ℕ := if n ≥ 2 then Nat.ceil (Real.log2 (n + 1)) else 0

-- Prove that f(n) is the smallest value satisfying the conditions
theorem smallest_f (n : ℕ) (N : Finset ℕ) (F : Finset (Finset ℕ)) :
  (n ≥ 2) → (∀ (x : ℕ), x ∈ N → x ≤ n) →
  (is_separating N F) → (is_covering N F) →
  ∃ t, f(n) = t ∧ (∃ F : Finset (Finset ℕ), is_separating N F ∧ is_covering N F ∧ F.card = t) :=
by
  sorry

end smallest_f_l218_218819


namespace range_of_a_l218_218301

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if x > 1 then 
    exp x - a * x^2 + x - 1 
  else 
    0 -- This definition will be modified as needed to fit within the proof structure, assuming a definition parameter for non-boundary elements of x

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), f (-x) a = -f x a) ∧ (∀ (m : ℝ), m ≠ 0 → f (1 / m) a * f m a = 1) ∧ (range (λ (x : ℝ), f x a) = univ) ∧ (∀ (x : ℝ), (x-2)*exp x - x + 4 > 0) → (a ∈ set.Icc (exp 1 - 1) ((exp 2 + 1) / 4)) :=
  by sorry

end range_of_a_l218_218301


namespace taquin_impossible_l218_218880

/-- 
Define the structure and movement rules of the Taquin game.
The game consists of 15 squares of area 1 placed on a square of area 16.
A move consists of moving the "hole" to a neighboring horizontal or vertical space.
Every square initially starts in its place, and we want to check the possibility of
swapping squares 14 and 15 while keeping all other squares in their places.

Statement: Prove the impossibility of this specific transformation.
-/
theorem taquin_impossible :
  impossible_transformation 16 [(iota.transposition 14 15)] :=
sorry

end taquin_impossible_l218_218880


namespace ellipse_foci_l218_218873

theorem ellipse_foci (m n : ℝ) (h1 : m < n) (h2 : n < 0) : 
  let f1 := (sqrt (n - m), 0)
  let f2 := (-(sqrt (n - m)), 0)
  (f1, f2) = ((sqrt (n - m), 0), (-(sqrt (n - m)), 0)) :=
  by
    sorry

end ellipse_foci_l218_218873


namespace six_digit_odd_number_count_l218_218070

/-- Determine the number of different 6-digit odd numbers 
    that can be formed without repeating any digit using 
    the digits 0, 1, 2, 3, 4, 5 -/
def count_odd_six_digit_numbers : ℕ := 288

theorem six_digit_odd_number_count :
  ∃ n : ℕ, n = count_odd_six_digit_numbers :=
by
  -- Define the available digits
  let digits := [0, 1, 2, 3, 4, 5]
  -- Calculate number of such 6-digit odd numbers under given conditions
  have count : ℕ := 3 * 4 * 4 * 3 * 2 * 1
  exact ⟨count, count_eq_proven_count⟩

-- Placeholder proof for count_eq_proven_count
-- This is where we'd typically prove that our calculations are correct
lemma count_eq_proven_count : 288 = 3 * 4 * 4 * 3 * 2 * 1 := sorry

end six_digit_odd_number_count_l218_218070


namespace piles_can_be_reduced_l218_218394

/-! 
  We define similar sizes as the difference between sizes being at most a factor of two.
  Given any number of piles of stones, we aim to prove that these piles can be combined 
  iteratively into one single pile.
-/

def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

theorem piles_can_be_reduced (n : ℕ) :
  ∃ pile : ℕ, (pile = n) ∧ (∀ piles : list ℕ, list.sum piles = n → 
    (∃ piles' : list ℕ, list.sum piles' = n ∧ list.length piles' = 1)) :=
by
  -- Placeholder for the proof.
  sorry

end piles_can_be_reduced_l218_218394


namespace relay_race_time_l218_218584

theorem relay_race_time (M S J T : ℕ) 
(hJ : J = 30)
(hS : S = J + 10)
(hM : M = 2 * S)
(hT : T = M - 7) : 
M + S + J + T = 223 :=
by sorry

end relay_race_time_l218_218584


namespace trip_cost_is_correct_l218_218444

-- Given conditions
def bills_cost : ℕ := 3500
def save_per_month : ℕ := 500
def savings_duration_months : ℕ := 2 * 12
def savings : ℕ := save_per_month * savings_duration_months
def remaining_after_bills : ℕ := 8500

-- Prove that the cost of the trip to Paris is 3500 dollars
theorem trip_cost_is_correct : (savings - remaining_after_bills) = bills_cost :=
sorry

end trip_cost_is_correct_l218_218444


namespace product_of_smallest_primes_l218_218096

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218096


namespace garden_stone_calculation_l218_218480

/-- A rectangular garden with dimensions 15m by 2m and patio stones of dimensions 0.5m by 0.5m requires 120 stones to be fully covered -/
theorem garden_stone_calculation :
  let garden_length := 15
  let garden_width := 2
  let stone_length := 0.5
  let stone_width := 0.5
  let area_garden := garden_length * garden_width
  let area_stone := stone_length * stone_width
  let num_stones := area_garden / area_stone
  num_stones = 120 :=
by
  sorry

end garden_stone_calculation_l218_218480


namespace emile_tetromino_impossible_l218_218072

theorem emile_tetromino_impossible (n : ℕ) (h : n = 2023) : 
  ¬(∃ (w h : ℕ), w * h = 4 * 7 * 2023 ∧ 
                   ∀ (t : tetromino) (x y : ℕ), 
                     tetromino_placements t (x, y) →
                     no_overlap (tetromino_placements t (x, y)) (w, h)
                   ) := 
by {
  sorry
}

end emile_tetromino_impossible_l218_218072


namespace rocco_total_money_l218_218854

def piles_of_quarters := 4
def piles_of_dimes := 6
def piles_of_nickels := 9
def piles_of_pennies := 5

def coins_per_pile := 10

def value_of_quarter := 0.25
def value_of_dime := 0.10
def value_of_nickel := 0.05
def value_of_penny := 0.01

def total_amount :=
  (coins_per_pile * piles_of_quarters * value_of_quarter) +
  (coins_per_pile * piles_of_dimes * value_of_dime) +
  (coins_per_pile * piles_of_nickels * value_of_nickel) +
  (coins_per_pile * piles_of_pennies * value_of_penny)

theorem rocco_total_money : total_amount = 21.00 := 
  sorry

end rocco_total_money_l218_218854


namespace find_y_when_x_4_l218_218047

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l218_218047


namespace range_of_m_l218_218320

theorem range_of_m (m : ℝ) (A B : set ℝ)  (hA : A = {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1})
  (hB : B = {x : ℝ | x^2 - 2 * x - 15 ≤ 0}) (h_subset : A ⊆ B) :
  2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l218_218320


namespace size_of_each_bottle_l218_218369

-- Defining given conditions
def petals_per_ounce : ℕ := 320
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes : ℕ := 800
def bottles : ℕ := 20

-- Proving the size of each bottle in ounces
theorem size_of_each_bottle : (petals_per_rose * roses_per_bush * bushes / petals_per_ounce) / bottles = 12 := by
  sorry

end size_of_each_bottle_l218_218369


namespace solve_equation_l218_218928

-- Define the logarithmic equation
def equation (x : ℝ) : ℝ :=
  log 3 (x + 2) * log 3 (2 * x + 1) * (3 - log 3 (2 * x^2 + 5 * x + 2))

-- Conditions ensuring logarithmic terms are defined and domain restrictions
def conditions (x : ℝ) : Prop :=
  x > -2 ∧ x > -1/2

-- Assert that x = 1 is the solution to the logarithmic equation under the above conditions
theorem solve_equation : equation 1 = 1 :=
by
  -- Enforce defined conditions
  have h1 : 1 > -2 := by norm_num
  have h2 : 1 > -1/2 := by norm_num
  -- Simplify and evaluate the equation
  calc
    equation 1 = log 3 (1 + 2) * log 3 (2 * 1 + 1) * (3 - log 3 (2 * 1^2 + 5 * 1 + 2)) : by rfl
    ... = log 3 3 * log 3 3 * (3 - log 3 9) : by norm_num
    ... = 1 * 1 * (3 - 2) : by simp [log_base_change]
    ... = 1 : by norm_num

-- Main proof statement
example : ∀ x : ℝ, conditions x → equation x = 1 ↔ x = 1 :=
by
  intro x
  intro h
  cases h with h1 h2
  split
  . intro hx
    apply_fun (λ e, e * x + e * y + z) 1
    exact solve_equation
    -- More steps, if necessary
  . intro hx
    rw hx
    exact solve_equation

end solve_equation_l218_218928


namespace cars_on_river_road_l218_218926

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 60) (h2 : B * 13 = C) : C = 65 :=
sorry

end cars_on_river_road_l218_218926


namespace cos_alpha_minus_beta_l218_218748

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : sin α - sin β = -1 / 2) 
  (h2 : cos α - cos β = 1 / 2) : 
  cos (α - β) = 3 / 4 := 
by
  sorry

end cos_alpha_minus_beta_l218_218748


namespace circle_area_percentage_increase_l218_218343

theorem circle_area_percentage_increase (r : ℝ) (h : r > 0) :
  let original_area := (Real.pi * r^2)
  let new_radius := (2.5 * r)
  let new_area := (Real.pi * new_radius^2)
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  percentage_increase = 525 := by
  let original_area := Real.pi * r^2
  let new_radius := 2.5 * r
  let new_area := Real.pi * new_radius^2
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  sorry

end circle_area_percentage_increase_l218_218343


namespace all_statements_correct_l218_218922

theorem all_statements_correct :
  (molar_mass H₂SO₄ = 98) ∧
  (number_of_atoms (2, NO) ≠ number_of_atoms (2, NO₂)) ∧
  (number_of_oxygen_atoms_equal O₂ O₃) ∧
  (equal_moles_carbon_atoms CO CO₂) :=
by
suffices H₂SO₄,
from ⟨98,
  suffices 2,
  from ⟨
    suffices 6,
    from ⟨
      suffices mol,
      from ⟨CO₂,
        suffices atoms,
        from sorry⟩⟩⟩⟩

end all_statements_correct_l218_218922


namespace calculate_f_f_neg2_l218_218832

def f (x: ℝ) : ℝ := if x >= 0 then 1 - real.sqrt x else 2^x

theorem calculate_f_f_neg2 : f (f (-2)) = 1 / 2 :=
by
  sorry

end calculate_f_f_neg2_l218_218832


namespace probability_of_no_shaded_rectangle_l218_218154

-- Define the parameters
def rows : ℕ := 3
def columns : ℕ := 2004
def total_rectangles : ℕ := (rows * (columns + 1).choose 2)
def shaded_rectangles : ℕ := rows * (columns // 2 + 1) * (columns // 2)

-- Define the probability that a rectangle does NOT include a shaded square
def probability_no_shaded (total_rectangles shaded_rectangles : ℕ) : ℚ := 
  1 - (shaded_rectangles / total_rectangles)

-- Given statements
theorem probability_of_no_shaded_rectangle :
  probability_no_shaded total_rectangles shaded_rectangles = 1003 / 2004 :=
by sorry

end probability_of_no_shaded_rectangle_l218_218154


namespace sum_of_all_four_digit_numbers_l218_218255

def digits : List ℕ := [1, 2, 3, 4, 5]

noncomputable def four_digit_numbers := 
  (Digits.permutations digits).filter (λ l => l.length = 4)

noncomputable def sum_of_numbers (nums : List (List ℕ)) : ℕ :=
  nums.foldl (λ acc num => acc + (num.foldl (λ acc' digit => acc' * 10 + digit) 0)) 0

theorem sum_of_all_four_digit_numbers :
  sum_of_numbers four_digit_numbers = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_l218_218255


namespace sum_x_satisfying_sum_solutions_eq_neg8_l218_218392

-- Definition of the function g
def g (x : ℝ) : ℝ := 3 * x - 2

-- The statement to prove - the sum of all x that satisfy g⁻¹(x) = g(x⁻¹) is -8
theorem sum_x_satisfying (x : ℝ) (h : g⁻¹(x) = g (1/x)) : x = -9 ∨ x = 1 :=
  (by have h1 : (x + 2) / 3 = (3 - 2 * x) / x := sorry
      have hquad : x^2 + 8 * x - 9 = 0 := sorry
      show x = -9 ∨ x = 1 from sorry)

-- Sum the solutions
theorem sum_solutions_eq_neg8 : (-9) + 1 = -8 := rfl

end sum_x_satisfying_sum_solutions_eq_neg8_l218_218392


namespace no_possible_arrangement_l218_218576

-- Define the problem conditions and declare the impossibility theorem.
theorem no_possible_arrangement :
  ∀ (P : Fin 12 → Set (ℝ × ℝ × ℝ)),
  (∀ i, i ∈ (Finset.range 12) → (P i).Nonempty ∧ 
    (∃ S : Fin 12 → Set (ℝ × ℝ), ∀ i j, i ≠ j → S i ∩ S j ≠ ∅ ↔ P i ∩ P j ≠ ∅) ∧
    (P 2 ∩ P 1 = ∅) ∧ (P 2 ∩ P 3 = ∅) ∧
    (P 3 ∩ P 2 = ∅) ∧ (P 3 ∩ P 4 = ∅) ∧
    (P 12 ∩ P 11 = ∅) ∧ (P 12 ∩ P 1 = ∅) ∧
    (P 1 ∩ P 12 = ∅) ∧ (P 1 ∩ P 2 = ∅))
  → False :=
by
sory

end no_possible_arrangement_l218_218576


namespace reflection_vector_l218_218681

theorem reflection_vector (u v : ℝ × ℝ) (h_u : u = (2, 5)) (h_v : v = (-1, 4)) :
  let proj := ((u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)) • v in
  2 • proj - u = (-70/17, 59/17) :=
by {
  -- Definitions and calculations go here, for now we use sorry to skip the proof.
  sorry
}

end reflection_vector_l218_218681


namespace product_of_primes_l218_218088

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l218_218088


namespace greatest_hours_on_tuesday_l218_218033

theorem greatest_hours_on_tuesday :
  let hours : ℕ → ℕ := λ (d : ℕ), if d = 1 then 3 else if d = 2 then 4 else if d = 3 then 2 else if d = 4 then 3 else if d = 5 then 1 else 0 in
  ∃ (d : ℕ), d = 2 ∧ ∀ (d' : ℕ), (d' ≠ 2 → hours d' ≤ hours 2) :=
by
  sorry

end greatest_hours_on_tuesday_l218_218033


namespace center_circle_is_correct_l218_218598

noncomputable def find_center_of_circle : ℝ × ℝ :=
  let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
  let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
  let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
  let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
  (-18, -11)

theorem center_circle_is_correct (x y : ℝ) :
  (let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
   let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
   let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
   let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
   (x, y) = find_center_of_circle) :=
  sorry

end center_circle_is_correct_l218_218598


namespace joe_paint_usage_l218_218572

theorem joe_paint_usage :
  let initial_paint := 360
  let first_week_usage := (1 / 3: ℝ) * initial_paint
  let remaining_after_first_week := initial_paint - first_week_usage
  let second_week_usage := (1 / 5: ℝ) * remaining_after_first_week
  let total_usage := first_week_usage + second_week_usage
  total_usage = 168 :=
by
  sorry

end joe_paint_usage_l218_218572


namespace sara_jim_savings_eq_l218_218856

theorem sara_jim_savings_eq (w : ℕ) : 
  let sara_init_savings := 4100
  let sara_weekly_savings := 10
  let jim_weekly_savings := 15
  (sara_init_savings + sara_weekly_savings * w = jim_weekly_savings * w) → w = 820 :=
by
  intros
  sorry

end sara_jim_savings_eq_l218_218856


namespace sin_600_plus_tan_240_l218_218220

theorem sin_600_plus_tan_240 :
  (Real.sin (600 * Real.pi / 180) + Real.tan (240 * Real.pi / 180)) = sqrt 3 / 2 :=
by
  sorry

end sin_600_plus_tan_240_l218_218220


namespace sum_of_x_where_gx_2005_l218_218164

noncomputable def g (x : ℝ) : ℝ := (21*x + 12 - 7/x) / 8

theorem sum_of_x_where_gx_2005 :
  let T := ∑ x in {x | g x = 2005}, x in -- collect all x such that g(x) = 2005 and sum them
  abs (T - 763) < 1 :=
by
  sorry

end sum_of_x_where_gx_2005_l218_218164


namespace solve_problem_l218_218801

-- Definition of parameters
def pointA : ℝ × ℝ := (real.sqrt 3, π / 6)
def inclination_angle : ℝ := π / 3
def parametric_curve_C (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Polar coordinate equation of line l
def polar_equation_line_l (ρ θ : ℝ) : Prop :=
  ρ * real.sin (π / 3 - θ) = real.sqrt 3 / 2

theorem solve_problem :
  ∃ ρ θ : ℝ, polar_equation_line_l ρ θ ∧ 
  ∀ t1 t2 : ℝ, let A := parametric_curve_C t1,
                  B := parametric_curve_C t2 in
    (B = (real.sqrt(3) / 2, -1)) →
    (real.dist A B = 16 / 3) := 
by
  sorry

end solve_problem_l218_218801


namespace general_formula_a_n_minimum_value_S_n_l218_218321

-- Define the given sequence and sum formula
def S (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Problem 1: Prove the general formula for a_n
theorem general_formula_a_n (n : ℕ) (hn : n ≥ 1) : 
  (a : ℕ → ℤ) := 
  ∀ n, (n = 1 → a = -28) ∧ (n ≥ 2 → a = 4 * n - 32) := by sorry

-- Problem 2: Find the minimum value of S_n and corresponding value of n
theorem minimum_value_S_n : 
  ∃ n, S n = -112 ∧ (n = 7 ∨ n = 8) := by sorry

end general_formula_a_n_minimum_value_S_n_l218_218321


namespace find_BD_l218_218363

-- Definitions from conditions
variables (A B C D : Type) [euclidean_space A B C D]
variables (a b c d : Point A) 

-- Given conditions
variable (hAB : (dist a b) = 22)
variable (hAC : (dist a c) = 10)
variable (hBC : (dist b c) = 10)
variable (hAD : (dist a d) = 12)
variable (hCD : (dist c d) = 4)

-- Proof statement
theorem find_BD : dist b d  = 7 :=
sorry

end find_BD_l218_218363


namespace max_team_members_l218_218613

theorem max_team_members :
  ∀ (members : Finset ℕ), 
  (∀ x ∈ members, 1 ≤ x ∧ x ≤ 100) ∧
  (∀ x y z ∈ members, x ≠ y → x ≠ z → y ≠ z → x + y ≠ z) ∧
  (∀ x y ∈ members, x ≠ y → 2*x ≠ y) → 
  members.card ≤ 50 :=
by
  intro members
  intro h
  sorry

end max_team_members_l218_218613


namespace sum_of_numbers_on_hats_l218_218904

def is_four_digit_perfect_square (n : ℕ) : Prop := 
  (1000 ≤ n) ∧ (n < 10000) ∧ ∃ m : ℕ, m * m = n

def tens_digit_is_zero (n : ℕ) : Prop :=
  (n / 10) % 10 = 0

def units_digit_is_not_zero (n : ℕ) : Prop :=
  (n % 10) ≠ 0

axiom A_statement (b hat_c : ℕ) : (b % 10) = (hat_c % 10)

axiom B_C_statement (hat_a b : ℕ) : (b % 10 = hat_a % 10) → exists n : ℕ, n * n = hat_a

axiom A_conclusion (hat_a hat_b hat_c : ℕ) : 
  B_C_statement hat_a hat_b (A_statement hat_b hat_c) → 
  (hat_a % 2 = 0) ∧ exists n : ℕ, n * n = hat_a

theorem sum_of_numbers_on_hats : 
  ∃ (a b c : ℕ), 
  is_four_digit_perfect_square a ∧ 
  is_four_digit_perfect_square b ∧ 
  is_four_digit_perfect_square c ∧ 
  tens_digit_is_zero a ∧ 
  tens_digit_is_zero b ∧ 
  tens_digit_is_zero c ∧ 
  units_digit_is_not_zero a ∧ 
  units_digit_is_not_zero b ∧ 
  units_digit_is_not_zero c ∧ 
  (A_statement b c) ∧ 
  B_C_statement a b (A_statement b c) ∧ 
  (A_conclusion a b c) ∧
  (a + b + c = 14612)
:= sorry

end sum_of_numbers_on_hats_l218_218904


namespace dissection_side_lengths_rational_l218_218942

variable {P : Type} [convex_polygon P]
variable (sides_diagonals_rational : ∀ (e : edge P), is_rational e.length)

theorem dissection_side_lengths_rational (polygon_small : convex_polygon)
  (h_dissection : dissects P polygon_small) 
  (h_sides_diagonals_rational : ∀ (e : edge P), is_rational e.length)
  : ∀ (e' : edge polygon_small), is_rational (e'.length) :=
by sorry

end dissection_side_lengths_rational_l218_218942


namespace triangle_inequality_expression_non_negative_l218_218578

theorem triangle_inequality_expression_non_negative
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end triangle_inequality_expression_non_negative_l218_218578


namespace equal_distances_AM_AN_l218_218284

variable (A B C M N : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (triangle_abc : IsAcuteAngledTriangle A B C)
variable (semicircle_ab : Semicircle A B)
variable (semicircle_ac : Semicircle A C)
variable (M_is_intersection : IsIntersection (perpendicular_from B) semicircle_ac M)
variable (N_is_intersection : IsIntersection (perpendicular_from C) semicircle_ab N)

theorem equal_distances_AM_AN : dist A M = dist A N :=
by sorry

end equal_distances_AM_AN_l218_218284


namespace sum_of_distinct_x_l218_218214

def g (x : ℝ) : ℝ := (x^2)/2 - x - 2

theorem sum_of_distinct_x : ∑ x in {x : ℝ | g (g (g x)) = -2}.toFinset = 4 :=
by
  sorry

end sum_of_distinct_x_l218_218214


namespace draw_connected_circles_without_lifting_l218_218358

theorem draw_connected_circles_without_lifting (n : ℕ) (h_connected : connected_figure 100) :
  ∃ p : path (connected_figure 100), ∀ e ∈ edges_of_path p, occurs_once e :=
sorry

end draw_connected_circles_without_lifting_l218_218358


namespace sara_jim_savings_eq_l218_218857

theorem sara_jim_savings_eq (w : ℕ) : 
  let sara_init_savings := 4100
  let sara_weekly_savings := 10
  let jim_weekly_savings := 15
  (sara_init_savings + sara_weekly_savings * w = jim_weekly_savings * w) → w = 820 :=
by
  intros
  sorry

end sara_jim_savings_eq_l218_218857


namespace minimum_red_vertices_l218_218820

theorem minimum_red_vertices (n : ℕ) (h : 0 < n) :
  ∃ R : ℕ, (∀ i j : ℕ, i < n ∧ j < n →
    (i + j) % 2 = 0 → true) ∧
    R = Int.ceil (n^2 / 2 : ℝ) :=
sorry

end minimum_red_vertices_l218_218820


namespace find_increase_in_radius_l218_218551

def volume_cylinder (r h : ℝ) := π * r^2 * h

theorem find_increase_in_radius (x : ℝ) :
  let r := 5
  let h1 := 4
  let h2 := h1 + 2
  volume_cylinder (r + x) h1 = volume_cylinder r h2 →
  x = (5*(Real.sqrt 6 - 2))/2 :=
by
  let r := 5
  let h1 := 4
  let h2 := h1 + 2
  let volume_cylinder := fun (r h : ℝ) => π * r^2 * h
  sorry

end find_increase_in_radius_l218_218551


namespace bad_arrangement_count_l218_218516

open List

-- Define the concept of an arrangement being "bad"
def bad_arrangement (l : List ℕ) : Prop :=
  (∀ n, n ∈ range 1 22 → 
    ¬∃ (k : ℕ) (s : List ℕ), s.sum = n ∧ s.length = k ∧ l.rotate k.is_cycle) 

-- Define the count of distinct bad arrangements
def num_bad_arrangements := 
  {l : List ℕ // l.perm [1, 2, 3, 4, 5, 6] 
    ∧ bad_arrangement l}

theorem bad_arrangement_count : 
  Fintype.card num_bad_arrangements = 3 :=
sorry

end bad_arrangement_count_l218_218516


namespace thirtieth_term_in_sequence_l218_218041

def contains_digit_5 (n : ℕ) : Prop :=
  n.digits 10 ∈ (list.filter (eq 5) (n.digits 10)).singleton

def is_sequence_number (n : ℕ) (k : ℕ) : Prop :=
  n % 3 = 0 ∧ contains_digit_5 n ∧ ((list.filter (λ x, contains_digit_5 x ∧ x % 3 = 0) (list.range (n+1))).nth (k - 1) = some n)

theorem thirtieth_term_in_sequence : ∃ n : ℕ, is_sequence_number n 30 ∧ n = 495 :=
by
  existsi 495
  split
  sorry
  refl

end thirtieth_term_in_sequence_l218_218041


namespace radius_fourth_circle_eq_twenty_l218_218908

noncomputable def radius_of_fourth_circle
  (r1 r2 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25) : ℝ :=
  let larger_area := π * r2^2
  let smaller_area := π * r1^2
  let shaded_area := larger_area - smaller_area
  let radius := sqrt (shaded_area / π)
  radius

theorem radius_fourth_circle_eq_twenty :
  radius_of_fourth_circle 15 25 (by rfl) (by rfl) = 20 :=
sorry

end radius_fourth_circle_eq_twenty_l218_218908


namespace intersection_complement_eq_singleton_l218_218838

-- Definitions from the problem conditions
def U := Set.univ -- Universal set ℤ in Lean
def A := {-1, 1, 2}
def B := {-1, 1}
def compl_U_B := {x : ℤ | ¬ (x = -1 ∨ x = 1)}

-- The theorem statement which includes the proof problem
theorem intersection_complement_eq_singleton :
  A ∩ compl_U_B = {2} :=
by
  sorry

end intersection_complement_eq_singleton_l218_218838


namespace savings_equal_in_820_weeks_l218_218859

-- Definitions for the conditions
def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

-- The statement we want to prove
theorem savings_equal_in_820_weeks : 
  ∃ (w : ℕ), (sara_initial_savings + w * sara_weekly_savings) = (w * jim_weekly_savings) ∧ w = 820 :=
by
  sorry

end savings_equal_in_820_weeks_l218_218859


namespace product_of_primes_is_66_l218_218122

theorem product_of_primes_is_66 :
  let p1 : ℕ := 2
      p2 : ℕ := 3
      p3 : ℕ := 11
  in p1 * p2 * p3 = 66 := by
  sorry

end product_of_primes_is_66_l218_218122


namespace mooncake_sales_increased_by_20_daily_sales_profit_function_maximize_daily_sales_profit_l218_218478

-- Define basic parameters
def cost_price := 60
def min_selling_price := 70
def initial_boxes_sold := 500
def decrease_in_boxes_per_yuan := 20

-- Define the problem: (1)
theorem mooncake_sales_increased_by_20 :
  let new_selling_price := min_selling_price + 20
  let boxes_sold := initial_boxes_sold - decrease_in_boxes_per_yuan * 20
  let profit_per_box := new_selling_price - cost_price
  let total_profit := boxes_sold * profit_per_box
  boxes_sold = 100 ∧ total_profit = 3000 :=
by
  sorry

-- Define the problem: (2)
theorem daily_sales_profit_function :
  ∀ x : ℕ,
  let y := (min_selling_price + x - cost_price) * (initial_boxes_sold - decrease_in_boxes_per_yuan * x)
  y = -20 * x ^ 2 + 300 * x + 5000 :=
by 
  sorry

-- Define the problem: (3)
theorem maximize_daily_sales_profit :
  let max_profit := 6120
  let optimal_price1 := 77
  let optimal_price2 := 78
  ( ∃ x : ℕ, x = 7 ∨ x = 8 → (min_selling_price + x = optimal_price1 ∨ min_selling_price + x = optimal_price2) ∧ 
    ( -20 * x ^ 2 + 300 * x + 5000 = max_profit )
  ) :=
by 
  sorry

end mooncake_sales_increased_by_20_daily_sales_profit_function_maximize_daily_sales_profit_l218_218478


namespace pair_count_equals_1999_l218_218529

noncomputable theory

def sequence_a (α β : ℝ) : ℕ → ℝ
| 0       := α
| (n + 1) := α * sequence_a n - β * sequence_b n

def sequence_b (α β : ℝ) : ℕ → ℝ
| 0       := β
| (n + 1) := β * sequence_a n + α * sequence_b n

theorem pair_count_equals_1999 (α β : ℝ) (h1 : sequence_a α β 1996 = β) (h2 : sequence_b α β 1996 = α) : 
  ∃ N, N = 1999 :=
by
  sorry

end pair_count_equals_1999_l218_218529


namespace max_inclination_lineA_l218_218630

-- Definitions for the equations of lines
def lineA : ℝ → ℝ := λ x, -x + 1
def lineB : ℝ → ℝ := λ x, x + 1
def lineC : ℝ → ℝ := λ x, 2 * x + 1
def lineD : ℝ := 1 -- x = 1 (vertical line)

-- Define homogeneous angles of inclination (not real-world-trigonometrically accurate)
def inclination (m : ℝ) : ℝ := 
  if m = -1 then 135
  else if m = 1 then 45
  else if m > 0 then 
    if m = 2 then 60 -- approximation for example
    else 70 -- another approximation
  else 90 -- for the vertical line, line D

theorem max_inclination_lineA :
  inclination (-1) > inclination (1) ∧ inclination (-1) > inclination (2) ∧ inclination (-1) > inclination 0 :=
by
  sorry

end max_inclination_lineA_l218_218630


namespace youngest_child_age_l218_218603

/-- A father and his three children go to a restaurant. The father is charged $4.95, and the 
    rate for each child's age is $0.45 per year. The total bill is $9.45. Prove that the 
    age of the youngest child is 2 years old. -/
theorem youngest_child_age 
  (father_charge : ℝ := 4.95)
  (child_rate_per_year : ℝ := 0.45)
  (total_bill : ℝ := 9.45)
  (twins : ℕ) -- age of each twin
  (youngest : ℕ) -- age of the youngest child
  (total_child_age := (total_bill - father_charge) / child_rate_per_year)
  (total_years := 10 := int.of_nat total_child_age)
  (total_bill_correct : total_bill = 9.45)
  (father_charge_correct : father_charge = 4.95)
  (child_rate_correct : child_rate_per_year = 0.45)
  (total_years_correct : total_years = 10)
  (total_child_age_correct : (total_bill - father_charge) / child_rate_per_year = total_years)
  (twins_age_correct : 2 * twins + youngest = total_years)
  : youngest = 2 := 
sorry

end youngest_child_age_l218_218603


namespace equation_of_line_through_points_l218_218031

-- Definitions for the problem conditions
def point1 : ℝ × ℝ := (-1, 2)
def point2 : ℝ × ℝ := (-3, -2)

-- The theorem stating the equation of the line passing through the given points
theorem equation_of_line_through_points :
  ∃ a b c : ℝ, (a * point1.1 + b * point1.2 + c = 0) ∧ (a * point2.1 + b * point2.2 + c = 0) ∧ 
             (a = 2) ∧ (b = -1) ∧ (c = 4) :=
by
  sorry

end equation_of_line_through_points_l218_218031


namespace train_length_approx_l218_218178

variable (s : ℝ := 300) -- speed in km/hr
variable (t : ℝ := 15) -- time in seconds
-- Conversion factor from km/hr to m/s
def kmph_to_mps (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Calculating the length of the train
def length_of_train (speed_kmph time_sec : ℝ) : ℝ :=
  kmph_to_mps speed_kmph * time_sec

theorem train_length_approx :
  |length_of_train s t - 1249.95| < 0.01 :=
  sorry

end train_length_approx_l218_218178


namespace hugo_climbed_3_times_l218_218755

variable {HugoMountainElevation : ℕ}
variable {BorisMountainDifference : ℕ}
variable {BorisClimbs : ℕ}

-- Assigning the values based on the problem conditions
def HugoMountainElevation : ℕ := 10000
def BorisMountainDifference : ℕ := 2500
def BorisClimbs : ℕ := 4

-- Calculating Boris' mountain elevation
def BorisMountainElevation : ℕ := HugoMountainElevation - BorisMountainDifference

-- Define the number of times Hugo climbed his mountain
def HugoClimbs : ℕ := 3

theorem hugo_climbed_3_times (x : ℕ) (h : HugoMountainElevation * x = BorisMountainElevation * BorisClimbs) : x = 3 :=
by {
  -- Skipping the proof as instructed
  sorry
}

end hugo_climbed_3_times_l218_218755


namespace eccentricity_of_hyperbola_l218_218319

noncomputable def hyperbola (a b : ℝ) (hx1 : a > 0) (hx2 : b > 0) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

noncomputable def asymptote (a b : ℝ) : Prop :=
  ∀ x : ℝ, y = ±(√3 / 3) * x

theorem eccentricity_of_hyperbola (a b c : ℝ) (hx1 : a > 0) (hx2 : b > 0)
  (h_asym : asymptote a b) (h_asym_value : b = (√3 / 3) * a)
  (h_c : c = √(a^2 + b^2)) :
  ∃ e : ℝ, e = (2 * √3) / 3 :=
sorry

end eccentricity_of_hyperbola_l218_218319


namespace tom_total_trip_cost_is_correct_l218_218064

noncomputable def Tom_total_cost : ℝ :=
  let cost_vaccines := 10 * 45
  let cost_doctor := 250
  let total_medical := cost_vaccines + cost_doctor
  
  let insurance_coverage := 0.8 * total_medical
  let out_of_pocket_medical := total_medical - insurance_coverage
  
  let cost_flight := 1200

  let cost_lodging := 7 * 150
  let cost_transportation := 200
  let cost_food := 7 * 60
  let total_local_usd := cost_lodging + cost_transportation + cost_food
  let total_local_bbd := total_local_usd * 2

  let conversion_fee_bbd := 0.03 * total_local_bbd
  let conversion_fee_usd := conversion_fee_bbd / 2

  out_of_pocket_medical + cost_flight + total_local_usd + conversion_fee_usd

theorem tom_total_trip_cost_is_correct : Tom_total_cost = 3060.10 :=
  by
    -- Proof skipped
    sorry

end tom_total_trip_cost_is_correct_l218_218064


namespace adam_red_circle_radius_correct_eva_green_circle_radius_correct_l218_218182

noncomputable def adam_red_circle_radius (r : ℝ) : ℝ :=
  (Real.sqrt 2 - 1) * r

noncomputable def eva_green_circle_radius (r : ℝ) : ℝ :=
  ((2 * Real.sqrt 3) / 3 - 1) * r

theorem adam_red_circle_radius_correct (r : ℝ) : adam_red_circle_radius(r) = (Real.sqrt 2 - 1) * r := by
  rfl

theorem eva_green_circle_radius_correct (r : ℝ) : eva_green_circle_radius(r) = ((2 * Real.sqrt 3) / 3 - 1) * r := by
  rfl

end adam_red_circle_radius_correct_eva_green_circle_radius_correct_l218_218182


namespace only_option_d_determines_location_l218_218565

-- Define the problem conditions in Lean
inductive LocationOption where
  | OptionA : LocationOption
  | OptionB : LocationOption
  | OptionC : LocationOption
  | OptionD : LocationOption

-- Define a function that takes a LocationOption and returns whether it can determine a specific location
def determine_location (option : LocationOption) : Prop :=
  match option with
  | LocationOption.OptionD => True
  | LocationOption.OptionA => False
  | LocationOption.OptionB => False
  | LocationOption.OptionC => False

-- Prove that only option D can determine a specific location
theorem only_option_d_determines_location : ∀ (opt : LocationOption), determine_location opt ↔ opt = LocationOption.OptionD := by
  intro opt
  cases opt
  · simp [determine_location, LocationOption.OptionA]
  · simp [determine_location, LocationOption.OptionB]
  · simp [determine_location, LocationOption.OptionC]
  · simp [determine_location, LocationOption.OptionD]

end only_option_d_determines_location_l218_218565


namespace trigonometric_ratios_of_triangle_not_obtuse_but_right_triangle_radius_of_incircle_radius_of_circumcircle_l218_218364

theorem trigonometric_ratios_of_triangle
    (a b c : ℝ)
    (ha : a / b = 5 / 12)
    (hb : b / c = 12 / 13)
    (hc : a / c = 5 / 13) :
    (sin (angle A / a) = sin (angle B / b) = sin (angle C / c)) :=
by sorry

theorem not_obtuse_but_right_triangle
    (a b c : ℝ)
    (k : ℝ)
    (ha : a = 5 * k)
    (hb : b = 12 * k)
    (hc : c = 13 * k) :
    triangle_right (c =
    let angle_c := acos ((a^2 + b^2 - c^2) / (2 * a * b)) in
    angle_c = π / 2 :=
by sorry

theorem radius_of_incircle
    (a b c : ℝ)
    (c_val : c = 26)
    (S : ℝ)
    (incircle_radius : ℝ)
    (area : S = 1/2 * a * b)
    (r_eq : S = 1/2 * (a + b + c) * incircle_radius) :
    incircle_radius = 4 :=
by sorry

theorem radius_of_circumcircle
    (a b c : ℝ)
    (c_val : c = 26)
    (circum_radius : ℝ)
    (sinC : sin (π / 2) = 1)
    (R_eq : 2 * circum_radius = c / sin (angle C)) :
    circum_radius = 13 :=
by sorry

end trigonometric_ratios_of_triangle_not_obtuse_but_right_triangle_radius_of_incircle_radius_of_circumcircle_l218_218364


namespace sum_squares_nonpositive_l218_218692

theorem sum_squares_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ac ≤ 0 :=
by {
  sorry
}

end sum_squares_nonpositive_l218_218692


namespace original_quantity_of_ghee_l218_218144

-- Define the quantities and conditions
variable (x : ℝ) -- original quantity of the ghee mixture
variable (pure_ghee_original : ℝ) -- original amount of pure ghee
variable (vanaspati_original : ℝ) -- original amount of vanaspati
variable (pure_ghee_added : ℝ := 10) -- amount of pure ghee added
variable (total_new : ℝ) -- new total quantity after adding pure ghee
variable (vanaspati_new : ℝ) -- new amount of vanaspati 
variable (vanaspati_percentage_new : ℝ := 0.2) -- new percentage of vanaspati

-- Setting up conditions
def condition_1 := pure_ghee_original = 0.6 * x
def condition_2 := vanaspati_original = 0.4 * x
def condition_3 := total_new = x + pure_ghee_added
def condition_4 := vanaspati_new = vanaspati_percentage_new * total_new
def condition_5 := vanaspati_original = vanaspati_new

-- The theorem we want to prove
theorem original_quantity_of_ghee : 
    condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5 → 
    x = 10 := by
  sorry

end original_quantity_of_ghee_l218_218144


namespace places_proven_l218_218462

-- Definitions based on the problem conditions
inductive Place
| first
| second
| third
| fourth

def is_boy : String -> Prop
| "Oleg" => True
| "Olya" => False
| "Polya" => False
| "Pasha" => False
| _ => False

def name_starts_with_O : String -> Prop
| n => (n.head! = 'O')

noncomputable def determine_places : Prop :=
  ∃ (olegs_place olyas_place polyas_place pashas_place : Place),
  -- Statements and truth conditions
  ∃ (truthful : String), truthful ∈ ["Oleg", "Olya", "Polya", "Pasha"] ∧ 
  ∀ (person : String), 
    (person ≠ truthful → ∀ (statement : Place -> Prop), ¬ statement (person_to_place person)) ∧
    (person = truthful → person_to_place person = Place.first) ∧
    (person = truthful → 
      match person with
        | "Olya" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → is_boy (place_to_person p)
        | "Oleg" => ∃ (p : Place), (person_to_place "Oleg" = p ∧ person_to_place "Olya" = succ_place p ∨ 
                                    person_to_place "Olya" = p ∧ person_to_place "Oleg" = succ_place p)
        | "Pasha" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → name_starts_with_O (place_to_person p)
        | _ => True
      end)

-- Helper functions to relate places to persons
def person_to_place : String -> Place
| "Oleg" => Place.first
| "Olya" => Place.second
| "Polya" => Place.third
| "Pasha" => Place.fourth
| _ => Place.first -- Default, shouldn't happen

def place_to_person : Place -> String
| Place.first => "Oleg"
| Place.second => "Olya"
| Place.third => "Polya"
| Place.fourth => "Pasha"

def succ_place : Place → Place
| Place.first => Place.second
| Place.second => Place.third
| Place.third => Place.fourth
| Place.fourth => Place.first -- No logical next in this context.

theorem places_proven : determine_places :=
by
  sorry

end places_proven_l218_218462


namespace range_of_b_length_of_AB_l218_218992

noncomputable def discriminant : ℝ → ℝ :=
  λ b, 24 - 8 * b^2

theorem range_of_b (b : ℝ) :
  (3*x^2 + 4*b*x + 2*b^2 - 2 = 0) ∧ (discriminant b > 0) ↔ -Real.sqrt 3 < b ∧ b < Real.sqrt 3 := 
  sorry

theorem length_of_AB (b : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  b = 1 ∧ 
  (x₁ = 0 ∧ x₂ = -4 / 3) ∧ 
  (y₁ = 1 ∧ y₂ = -1 / 3) →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 2 / 3 := 
  sorry

end range_of_b_length_of_AB_l218_218992


namespace arithmetic_sequence_sum_difference_l218_218851

theorem arithmetic_sequence_sum_difference (n : ℕ) (h : Even (2 * n)) (a₁ : ℤ) (d : ℤ) :
    let a := λ k: ℕ, a₁ + d * (k - 1)
    let S₁ := (n * (2 * a (1) + (n - 1) * d)) / 2
    let S₂ := (n * (2 * a (n + 1) + (n - 1) * d)) / 2
    (S₂ - S₁) = d * n^2 :=
by 
  sorry

end arithmetic_sequence_sum_difference_l218_218851


namespace determine_m_l218_218338

-- Define the fractional equation condition
def fractional_eq (m x : ℝ) : Prop := (m/(x - 2) + 2*x/(x - 2) = 1)

-- Define the main theorem statement
theorem determine_m (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ x ≠ 2 ∧ fractional_eq m x) : m = -4 :=
sorry

end determine_m_l218_218338


namespace min_stamps_value_l218_218985

noncomputable def min_stamps : ℕ :=
  Inf { n : ℕ | ∃ (c f : ℕ), 3 * c + 5 * f = 39 ∧ c + f = n }

theorem min_stamps_value : min_stamps = 9 :=
sorry

end min_stamps_value_l218_218985


namespace competition_results_correct_l218_218459

theorem competition_results_correct :
  ∃ (first second third fourth : String), 
    (first = "Oleg" ∧ second = "Olya" ∧ third = "Polya" ∧ fourth = "Pasha") ∧
    ∀ (claims : String → String → Prop),
      (claims "Olya" "all_odd_places_boys") ∧ 
      (claims "Oleg" "consecutive_places_with_olya") ∧
      (claims "Pasha" "all_odd_places_names_start_O") ∧
      ∃ (truth_teller : String), 
        truth_teller = "Oleg" ∧ 
        (claims "Oleg" "first_place") ∧ 
        ¬ (claims "Olya" "first_place") ∧ 
        ¬ (claims "Pasha" "first_place") ∧ 
        ¬ (claims "Polya" "first_place") :=
sorry

end competition_results_correct_l218_218459


namespace length_increase_percentage_l218_218503

theorem length_increase_percentage (L B : ℝ) (x : ℝ) (h1 : (L + (x / 100) * L) * (B - (5 / 100) * B) = 1.14 * L * B) : x = 20 := by 
  sorry

end length_increase_percentage_l218_218503


namespace pile_of_stones_l218_218408

def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

theorem pile_of_stones (n : ℕ) (f : ℕ → ℕ): (∀ i, 1 ≤ f i ∧ f i ≤ n) → 
  (∀ j k, similar_sizes (f j) (f k)) → True :=
by
  simp
  exact true.intro


end pile_of_stones_l218_218408


namespace range_of_a_not_monotonic_l218_218340

noncomputable def is_not_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ (f x < f y ∧ f' f x > f' f y ∨ f x > f y ∧ f' f x < f' f y)

def f (a : ℝ) : ℝ → ℝ :=
  λ x, x + a * Real.log x

def f' (a : ℝ) : ℝ → ℝ :=
  λ x, 1 + a / x

theorem range_of_a_not_monotonic :
  ∀ (a : ℝ), is_not_monotonic (f a) ↔ a < 0 :=
by
  sorry

end range_of_a_not_monotonic_l218_218340


namespace paper_saving_of_assembled_cube_l218_218558

theorem paper_saving_of_assembled_cube :
  let small_cube_edge := 2 in
  let large_cube_edge := 4 * small_cube_edge in
  let small_cube_surface_area := 6 * small_cube_edge^2 in
  let total_small_cubes := 64 in
  let total_surface_area_of_small_cubes := total_small_cubes * small_cube_surface_area in
  let large_cube_surface_area := 6 * large_cube_edge^2 in
  let saved_paper_area := total_surface_area_of_small_cubes - large_cube_surface_area in
  saved_paper_area = 1152 :=
by
  sorry

end paper_saving_of_assembled_cube_l218_218558


namespace sequence_inequality_l218_218702

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

axiom a_pos : ∀ n : ℕ, 0 < a n
axiom Sn_def : ∀ n : ℕ, S n = ((1 + a n) / 2) ^ 2
axiom bn_def : ∀ n : ℕ, b n = (a n) ^ 2

theorem sequence_inequality (n : ℕ) (hn : 0 < n) : 
  (∑ k in Finset.range n, 1 / (b (k + 1))) < 5 / 4 :=
by
  sorry

end sequence_inequality_l218_218702


namespace product_of_smallest_primes_l218_218097

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l218_218097


namespace probability_of_A_losing_l218_218909

variable (p_win p_draw p_lose : ℝ)

def probability_of_A_winning := p_win = (1/3)
def probability_of_draw := p_draw = (1/2)
def sum_of_probabilities := p_win + p_draw + p_lose = 1

theorem probability_of_A_losing
  (h1: probability_of_A_winning p_win)
  (h2: probability_of_draw p_draw)
  (h3: sum_of_probabilities p_win p_draw p_lose) :
  p_lose = (1/6) :=
sorry

end probability_of_A_losing_l218_218909


namespace inv_fun_pass_through_point_l218_218774

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x

theorem inv_fun_pass_through_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
    (h3 : ∃ g, (∀ y, f a (g y) = y) ∧ (g 3 = -1)) : a = 1 / 3 :=
by
  sorry

end inv_fun_pass_through_point_l218_218774


namespace ae_perpendicular_cd_l218_218847

variables (A B C D E : Type) [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space D] [euclidean_space E]

-- Given definitions
def is_isosceles_trapezoid (ABCD : Type) (A B C D : euclidean_space) : Prop :=
  (A ≠ B) ∧ (C ≠ D) ∧ ∃ (M N : Type) [euclidean_space M] [euclidean_space N],
  (M ≠ N) ∧ segment M A = segment M B ∧ segment N C = segment N D

def is_right_isosceles_triangle (B C E : Type) (B C E : euclidean_space) : Prop :=
  (angle B C E = 45) ∧ (angle E C B = 45) ∧ (angle B E C = 90)

-- The statement to prove
theorem ae_perpendicular_cd
  (A B C D E : euclidean_space)
  (h1 : is_isosceles_trapezoid A B C D)
  (h2 : E ∈ segment B D)
  (h3 : is_right_isosceles_triangle B C E) :
  ∠ (A, E) (C, D) = 90 :=
begin
  sorry -- Proof goes here
end

end ae_perpendicular_cd_l218_218847


namespace competition_places_l218_218467

def participants := ["Olya", "Oleg", "Polya", "Pasha"]
def placements := Array.range 1 5

-- Define statements made by each child
def Olya_claims_odd_boys (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → (name = "Oleg" ∨ name = "Pasha")

def Oleg_claims_consecutive_with_Olya (placement : String → Nat) : Prop :=
  abs (placement "Oleg" - placement "Olya") = 1

def Pasha_claims_odd_O_names (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → name.startsWith "O"

-- Define the main problem statement
theorem competition_places :
  ∃ (placement : String → Nat),
    placement "Oleg" = 1 ∧
    placement "Olya" = 2 ∧
    placement "Polya" = 3 ∧
    placement "Pasha" = 4 ∧
    (∃ name, (name = "Oleg" ∨ name = "Olya" ∨ name = "Polya" ∨ name = "Pasha") ∧
      ((name = "Oleg" → (placement "Oleg" = 1 ∧ Oleg_claims_consecutive_with_Olya placement)) ∧
       (name = "Olya" → (placement "Olya" = 1 ∧ Olya_claims_odd_boys placement)) ∧
       (name = "Pasha" → (placement "Pasha" = 1 ∧ Pasha_claims_odd_O_names placement)))) :=
by
  have placement : String → Nat := λ name => match name with
    | "Olya" => 2
    | "Oleg" => 1
    | "Polya" => 3
    | "Pasha" => 4
    | _      => 0
  use placement
  simp [placement, Oleg_claims_consecutive_with_Olya, Olya_claims_odd_boys, Pasha_claims_odd_O_names]
  sorry

end competition_places_l218_218467


namespace hazel_additional_days_l218_218541

theorem hazel_additional_days (school_year_days : ℕ) (miss_percent : ℝ) (already_missed : ℕ)
  (h1 : school_year_days = 180)
  (h2 : miss_percent = 0.05)
  (h3 : already_missed = 6) :
  (⌊miss_percent * school_year_days⌋ - already_missed) = 3 :=
by
  sorry

end hazel_additional_days_l218_218541


namespace range_of_b_l218_218732

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := x^2 + b * x + c

def A (b c : ℝ) := {x : ℝ | f x b c = 0}
def B (b c : ℝ) := {x : ℝ | f (f x b c) b c = 0}

theorem range_of_b (b c : ℝ) (h : ∃ x₀ : ℝ, x₀ ∈ B b c ∧ x₀ ∉ A b c) :
  b < 0 ∨ b ≥ 4 := 
sorry

end range_of_b_l218_218732


namespace problem1_problem2_l218_218176

variable (x y : ℝ)

-- Problem 1: Prove A + B = 12*x^2*y + 3

def B : ℝ := 3 * x^2 * y - 2 * x * y + x + 2
def A_minus_B : ℝ := 6 * x^2 * y + 4 * x * y - 2 * x - 1

theorem problem1 (A B : ℝ) (h₁ : B = 3 * x^2 * y - 2 * x * y + x + 2) (h₂ : A - B = A_minus_B) : 
  A + B = 12 * x^2 * y + 3 :=
sorry

-- Problem 2: Prove y = 1/2 if A - 3B is a constant

theorem problem2 (A : ℝ) (h₁ : B = 3 * x^2 * y - 2 * x * y + x + 2) (h₂ : A + B = 12 * x^2 * y + 3) 
  (h₃ : ∀ x : ℝ, A - 3 * B = (8 * y - 4) * x - 5) : y = 1 / 2 :=
sorry

end problem1_problem2_l218_218176


namespace constant_sequence_if_and_only_if_arith_geo_progression_l218_218229

/-- A sequence a_n is both an arithmetic and geometric progression if and only if it is constant --/
theorem constant_sequence_if_and_only_if_arith_geo_progression (a : ℕ → ℝ) :
  (∃ q d : ℝ, (∀ n : ℕ, a (n+1) - a n = d) ∧ (∀ n : ℕ, a n = a 0 * q ^ n)) ↔ (∃ c : ℝ, ∀ n : ℕ, a n = c) := 
sorry

end constant_sequence_if_and_only_if_arith_geo_progression_l218_218229


namespace sequence_term_l218_218359

theorem sequence_term :
  let S : ℕ → ℤ := λ n, 4 * n ^ 2 - n - 8 in
  a_4 = S 4 - S 3 :=
by
  sorry

end sequence_term_l218_218359


namespace students_selecting_water_l218_218981

-- Definitions of percentages and given values.
def p : ℝ := 0.7
def q : ℝ := 0.1
def n : ℕ := 140

-- The Lean statement to prove the number of students who selected water.
theorem students_selecting_water (p_eq : p = 0.7) (q_eq : q = 0.1) (n_eq : n = 140) :
  ∃ w : ℕ, w = (q / p) * n ∧ w = 20 :=
by sorry

end students_selecting_water_l218_218981


namespace greatest_divisor_remainders_l218_218925

theorem greatest_divisor_remainders (d : ℤ) :
  d > 0 → (1657 % d = 10) → (2037 % d = 7) → d = 1 :=
by
  intros hdg h1657 h2037
  sorry

end greatest_divisor_remainders_l218_218925


namespace fraction_of_married_women_l218_218142

theorem fraction_of_married_women (total_employees : ℕ) 
  (women_fraction : ℝ) (married_fraction : ℝ) (single_men_fraction : ℝ)
  (hwf : women_fraction = 0.64) (hmf : married_fraction = 0.60) 
  (hsf : single_men_fraction = 2/3) : 
  ∃ (married_women_fraction : ℝ), married_women_fraction = 3/4 := 
by
  sorry

end fraction_of_married_women_l218_218142


namespace area_sub_triangles_l218_218807

variables {A B C M K L : Type}

/--
Given a triangle ABC and points M, K, and L on sides AB, BC, and CA respectively
such that none of them coincide with vertices A, B, or C, then at least one of 
the triangles MAL, KBM, or LCK has an area that is not greater than one quarter 
of the area of triangle ABC.
-/
theorem area_sub_triangles {T : Type} [normed_field T] 
  [normed_space T (euclidean_space T (fin 2))] {A B C M K L : T} 
  (hM : M ≠ A ∧ M ≠ B)
  (hK : K ≠ B ∧ K ≠ C)
  (hL : L ≠ C ∧ L ≠ A)
  (hM_on_AB : ∃ t ∈ Ioo (0 : T) 1, M = t • A + (1 - t) • B)
  (hK_on_BC : ∃ t ∈ Ioo (0 : T) 1, K = t • B + (1 - t) • C)
  (hL_on_CA : ∃ t ∈ Ioo (0 : T) 1, L = t • C + (1 - t) • A) :
  ∃ (i : ℕ), i ∈ {0, 1, 2} ∧ 
  area (if i = 0 then triangle (M, A, L)
        else if i = 1 then triangle (K, B, M)
        else triangle (L, C, K)) ≤ (1 / 4) * area (triangle (A, B, C)) := 
sorry

end area_sub_triangles_l218_218807


namespace balance_difference_is_137_l218_218643

noncomputable def cedric_balance (P : ℕ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t

noncomputable def daniel_balance (P : ℕ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

def positive_difference (A B : ℝ) : ℝ :=
  abs (A - B)

theorem balance_difference_is_137 :
  let P := 15000
      r_cedric := 0.06
      r_daniel := 0.08
      t := 10
      cedric_final := cedric_balance P r_cedric t
      daniel_final := daniel_balance P r_daniel t in
  (positive_difference daniel_final cedric_final).round = 137 :=
by
  let P := 15000
  let r_cedric := 0.06
  let r_daniel := 0.08
  let t := 10
  let cedric_final := cedric_balance P r_cedric t
  let daniel_final := daniel_balance P r_daniel t
  show (positive_difference daniel_final cedric_final).round = 137
  -- skipped proof with sorry
  sorry

end balance_difference_is_137_l218_218643


namespace circle_k_range_l218_218762

theorem circle_k_range {k : ℝ}
  (h : ∀ x y : ℝ, x^2 + y^2 - 2*x + y + k = 0) :
  k < 5 / 4 :=
sorry

end circle_k_range_l218_218762


namespace competition_places_l218_218465

def participants := ["Olya", "Oleg", "Polya", "Pasha"]
def placements := Array.range 1 5

-- Define statements made by each child
def Olya_claims_odd_boys (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → (name = "Oleg" ∨ name = "Pasha")

def Oleg_claims_consecutive_with_Olya (placement : String → Nat) : Prop :=
  abs (placement "Oleg" - placement "Olya") = 1

def Pasha_claims_odd_O_names (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → name.startsWith "O"

-- Define the main problem statement
theorem competition_places :
  ∃ (placement : String → Nat),
    placement "Oleg" = 1 ∧
    placement "Olya" = 2 ∧
    placement "Polya" = 3 ∧
    placement "Pasha" = 4 ∧
    (∃ name, (name = "Oleg" ∨ name = "Olya" ∨ name = "Polya" ∨ name = "Pasha") ∧
      ((name = "Oleg" → (placement "Oleg" = 1 ∧ Oleg_claims_consecutive_with_Olya placement)) ∧
       (name = "Olya" → (placement "Olya" = 1 ∧ Olya_claims_odd_boys placement)) ∧
       (name = "Pasha" → (placement "Pasha" = 1 ∧ Pasha_claims_odd_O_names placement)))) :=
by
  have placement : String → Nat := λ name => match name with
    | "Olya" => 2
    | "Oleg" => 1
    | "Polya" => 3
    | "Pasha" => 4
    | _      => 0
  use placement
  simp [placement, Oleg_claims_consecutive_with_Olya, Olya_claims_odd_boys, Pasha_claims_odd_O_names]
  sorry

end competition_places_l218_218465


namespace length_of_AC_l218_218362

variables {A B C D : Type}
variables [trapezoid ABCD]
variables (AB CD BC : ℝ) (angleBCD : ℝ)
variables (cos_angle_BCD : cos angleBCD = -2 / 7)

noncomputable def AC_length (AB CD BC : ℝ) (cos_angle_BCD : cos angleBCD = -2 / 7) : ℝ :=
if BC = 5 then
  if CD = 28 then
    if AB = 27 then 
      let sin_angle_ADC := sqrt (1 - (2 / 7)^2) in
      let DH := CD * cos (angleBCD) in
      let CH := CD * sin_angle_ADC in
      let PH := sqrt (AB^2 - CH^2) in
      let AC1 := sqrt (8^2 + (12 * sqrt 5)^2) in
      let AC2 := sqrt (2^2 + (12 * sqrt 5)^2) in
      if AC1 = 28 ∨ AC2 = 2 * sqrt 181then AC1 else AC2
    else
      0
  else
    0
else 
  0

-- The statement of the theorem.
theorem length_of_AC (h_AB : AB = 27) (h_CD : CD = 28) (h_BC : BC = 5) (h_cos : cos_angle_BCD = - 2 / 7) :
  AC_length AB CD BC cos_angle_BCD = 28 ∨ AC_length AB CD BC cos_angle_BCD = 2 * sqrt 181 :=
sorry

end length_of_AC_l218_218362


namespace no_positive_integers_solution_l218_218862

theorem no_positive_integers_solution (m n : ℕ) (hm : m > 0) (hn : n > 0) : 4 * m * (m + 1) ≠ n * (n + 1) := 
by
  sorry

end no_positive_integers_solution_l218_218862


namespace product_of_primes_l218_218102

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l218_218102


namespace martha_bought_9_tshirts_l218_218840

-- Definitions based on the problem conditions
def jackets_bought : ℕ := 4
def total_clothes : ℕ := 18

def total_jackets_taken_home (jackets_bought : ℕ) : ℕ := jackets_bought + (jackets_bought / 2)
def total_tshirts_taken_home (jackets_bought total_clothes : ℕ) : ℕ :=
  total_clothes - total_jackets_taken_home jackets_bought

-- Proving the number of t-shirts Martha bought
theorem martha_bought_9_tshirts
  (jackets_bought : ℕ) (total_clothes : ℕ)
  (H1 : jackets_bought = 4)
  (H2 : total_clothes = 18)
  (T : ℕ) : (T + (T / 3) = 12) → T = 9 :=
by
  intros h1 h2 T hT
  sorry

end martha_bought_9_tshirts_l218_218840


namespace center_incircle_on_line_MN_l218_218821

-- Definitions and assumptions from problem conditions
variables {A B C D M N : Point}
variables {AB CD : Line}
variables Incircle_ABC : Circle

-- Cyclic trapezoid ABCD, AB parallel CD, and AB > CD
axioms
  (cyclic_ABCD : CyclicQuadrilateral A B C D)
  (parallel_AB_CD : Parallel AB CD)
  (AB_greater_CD : AB > CD)

-- The incircle of triangle ABC is tangent to AB and AC at points M and N respectively
axioms
  (tangent_Incircle_AB_M : TangentPoint Incircle_ABC AB M)
  (tangent_Incircle_AC_N : TangentPoint Incircle_ABC AC N)

-- Center of the incircle of ABCD lies on line MN
theorem center_incircle_on_line_MN :
  Center (Incircle ABCD) ∈ Line M N :=
sorry

end center_incircle_on_line_MN_l218_218821


namespace tangent_line_equation_inequality_holds_l218_218315

-- Define the function
def f (x : ℝ) (a : ℝ) : ℝ := ℓ x - ((x - 1) * (a * x - a + 1)) / x

-- 1. The equation of the tangent line when a = 1
theorem tangent_line_equation (x : ℝ) (y : ℝ) :
  let a := 1
  let f := f(x, a)
  (e - 1) * x + e * y - e = 0 :=
sorry

-- 2. The inequality holding condition and the value of a
theorem inequality_holds (a : ℝ) :
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → (ℓ x / (x - 1) < (1 + a * (x - 1)) / x)) ↔ (a = 1 / 2) :=
sorry

end tangent_line_equation_inequality_holds_l218_218315


namespace proof_M_inter_N_eq_01_l218_218322
open Set

theorem proof_M_inter_N_eq_01 :
  let M := {x : ℤ | x^2 = x}
  let N := {-1, 0, 1}
  M ∩ N = {0, 1} := by
  sorry

end proof_M_inter_N_eq_01_l218_218322


namespace product_of_primes_l218_218075

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l218_218075


namespace sin_B_value_min_value_b2_over_a2_plus_c2_l218_218724

theorem sin_B_value (a b c : ℝ) (h1 : a * c * sin B = b^2 - (a - c)^2) : sin B = 4 / 5 := 
sorry

theorem min_value_b2_over_a2_plus_c2 (a b c : ℝ) (h2 : a * c * sin B = b^2 - (a - c)^2) : 
  (∃ eq_cond, (eq_cond = (a = c)) → (b^2) / (a^2 + c^2) = 2 / 5) :=
sorry

end sin_B_value_min_value_b2_over_a2_plus_c2_l218_218724


namespace solve_for_x_l218_218668

theorem solve_for_x (x : ℝ) (h : ⌈x⌉ * x = 156) : x = 12 :=
sorry

end solve_for_x_l218_218668


namespace exists_consecutive_with_square_factors_l218_218285

theorem exists_consecutive_with_square_factors (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ m : ℕ, m^2 ∣ (k + i) ∧ m > 1 :=
by {
  sorry
}

end exists_consecutive_with_square_factors_l218_218285


namespace number_of_valid_integers_l218_218998

def is_valid_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 9

def valid_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → is_valid_digit d

theorem number_of_valid_integers (count : ℕ) :
  count = 104 ↔
  count = (finset.Ico 1 1000000).card (λ n => n % 7 = 0 ∧ valid_digits n) := by
  sorry

end number_of_valid_integers_l218_218998


namespace sum_of_all_four_digit_numbers_l218_218239

-- Let us define the set of digits
def digits : set ℕ := {1, 2, 3, 4, 5}

-- We will define a function that generates the four-digit numbers
def four_digit_numbers := {n : ℕ // ∃ a b c d : ℕ, 
                                      a ∈ digits ∧ 
                                      b ∈ digits ∧ 
                                      c ∈ digits ∧ 
                                      d ∈ digits ∧ 
                                      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ 
                                      b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                                      n = 1000 * a + 100 * b + 10 * c + d}

-- Define a function to calculate the sum of all elements in a set of numbers
def sum_set (s : set ℕ) : ℕ := s.fold (λa b, a + b) 0

theorem sum_of_all_four_digit_numbers :
  sum_set {n | ∃ x : four_digit_numbers, x.val = n} = 399960 :=
sorry

end sum_of_all_four_digit_numbers_l218_218239


namespace average_percentage_decrease_l218_218601

theorem average_percentage_decrease (x : ℝ) : 60 * (1 - x) * (1 - x) = 48.6 → x = 0.1 :=
by sorry

end average_percentage_decrease_l218_218601


namespace graph_shift_l218_218907

theorem graph_shift (x : ℝ) :
  cos (2 * x + (π / 3)) = sin (2 * (x + (5 * π / 12))) :=
by
  sorry

end graph_shift_l218_218907


namespace find_n_l218_218227

/-- Given a positive integer n, the equation (1/x) + (1/y) = (1/n) has exactly 2011 positive
    integer solutions (x, y) with x ≤ y if and only if n = p^2010 for some prime number p. -/
theorem find_n (n : ℕ) (h : ∀ x y : ℕ, x ≤ y → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / n → (∃ p : ℕ, Prime p ∧ n = p ^ 2010)) :
  n = ∃ p : ℕ, Prime p ∧ n = p^2010 :=
by
  sorry -- proof goes here

end find_n_l218_218227


namespace probability_x_plus_y_lt_4_in_square_l218_218170

theorem probability_x_plus_y_lt_4_in_square :
  let square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let region := {p : ℝ × ℝ | p ∈ square ∧ p.1 + p.2 < 4}
  (measure_of region / measure_of square) = 7 / 9 := sorry

end probability_x_plus_y_lt_4_in_square_l218_218170


namespace find_y_l218_218052

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l218_218052


namespace evaluate_expression_l218_218223

theorem evaluate_expression (A B : ℝ) (hA : A = 2^7) (hB : B = 3^6) : (A ^ (1 / 3)) * (B ^ (1 / 2)) = 108 * 2 ^ (1 / 3) :=
by
  sorry

end evaluate_expression_l218_218223


namespace gym_membership_ratio_l218_218372

-- Define the costs and conditions
def cheap_gym_monthly_cost : ℕ := 10
def cheap_gym_signup_fee : ℕ := 50
def total_first_year_cost : ℕ := 650

-- Let x be the monthly membership cost of the second gym
variable (x : ℕ)

-- Condition that the second gym's sign-up fee is 4 times its monthly membership cost
def second_gym_signup_fee := 4 * x

-- Define the total first year cost for the cheap gym
def cheap_gym_first_year_cost := cheap_gym_signup_fee + 12 * cheap_gym_monthly_cost

-- Define the total first year cost for the second gym
def second_gym_first_year_cost := 16 * x

-- Define the total cost equation
def total_cost_equation := cheap_gym_first_year_cost + second_gym_first_year_cost = total_first_year_cost

-- The theorem to prove the ratio
theorem gym_membership_ratio (h : total_cost_equation x) : 3 * cheap_gym_monthly_cost = x :=
by {
  -- Assume the necessary conditions and simplify the equation
  let cheap_gym_first_year_cost := 170,
  let total_paid := 650,
  let cost_for_second_gym := total_paid - cheap_gym_first_year_cost,
  let x := cost_for_second_gym / 16,
  trivial
}

sorry

end gym_membership_ratio_l218_218372


namespace sale_in_third_month_l218_218606

theorem sale_in_third_month 
  (s1 s2 s4 s5 s6 : ℕ) 
  (avg_sale : ℕ) 
  (h1 : s1 = 5435) 
  (h2 : s2 = 5927) 
  (h4 : s4 = 6230) 
  (h5 : s5 = 5562) 
  (h6 : s6 = 3991) 
  (avg : avg_sale = 5500) : 
  ∃ s3 : ℕ, s3 = 5855 :=
by
  let total_sales := avg_sale * 6
  let known_sales := s1 + s2 + s4 + s5 + s6
  have h_known_sales : known_sales = 27145 := by sorry
  have h_total_sales : total_sales = 33000 := by sorry
  let s3 := total_sales - known_sales
  have h_s3 : s3 = 5855 := by sorry
  use s3
  exact h_s3

end sale_in_third_month_l218_218606


namespace shopkeeper_sold_articles_l218_218962

variable (C N : ℝ)
variable (h_cost_price : N * C = 12 * C + 0.20 * (N * C))

theorem shopkeeper_sold_articles : N = 15 := by
  have h1 : 12 * C + 0.20 * (N * C) = N * C := h_cost_price
  calc
    12 * C + 0.20 * (N * C) = N * C        : by rw [h1]
                            ... = 0.80 * N * C : by sorry
                            ...             ... : by sorry

end shopkeeper_sold_articles_l218_218962


namespace combine_piles_l218_218426

theorem combine_piles (n : ℕ) (piles : list ℕ) (h_piles : list.sum piles = n) (h_similar : ∀ x y ∈ piles, x ≤ y → y ≤ 2 * x) :
  ∃ pile, pile ∈ piles ∧ pile = n := sorry

end combine_piles_l218_218426


namespace competition_results_l218_218474

-- Participants and positions
inductive Participant : Type
| Oleg
| Olya
| Polya
| Pasha

-- Places in the competition (1st, 2nd, 3rd, 4th)
def Place := Fin 4

-- Statements made by the children
def Olya_statement1 : Prop := ∀ p, p % 2 = 1 -> p = Participant.Oleg ∨ p = Participant.Pasha
def Oleg_statement1 : Prop := ∃ p1 p2: Place, p1 < p2 ∧ (p1 = p2 + 1)
def Pasha_statement1 : Prop := ∀ p, p % 2 = 1 -> (p = Place 1 ∨ p = Place 3)

-- Truthfulness of the statements
def only_one_truthful (Olya_true : Prop) (Oleg_true : Prop) (Pasha_true : Prop) :=
  (Olya_true ∧ ¬ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ ¬ Oleg_true ∧ Pasha_true)

-- The actual positions
def positions : Participant → Place
| Participant.Oleg  := 0
| Participant.Olya  := 1
| Participant.Polya := 2
| Participant.Pasha := 3

-- The Lean statement to prove
theorem competition_results :
  ((Oleg_statement1 ↔ positions Participant.Oleg = 0) ∧ 
  (Olya_statement1 ↔ positions Participant.Olya = 1) ∧ 
  (Pasha_statement1 ↔ positions Participant.Pasha = 3)) ∧ 
  only_one_truthful (positions Participant.Oleg = 0) 
                    (positions Participant.Olya = 0) 
                    (positions Participant.Pasha = 0) ∧
  positions Participant.Oleg = 0 ∧ 
  positions Participant.Olya = 1 ∧
  positions Participant.Polya = 2 ∧
  positions Participant.Pasha = 3 := 
sorry

end competition_results_l218_218474


namespace train_stoppage_time_l218_218143

theorem train_stoppage_time (s1 s2 : ℝ) (h1 : s1 = 50) (h2 : s2 = 30) : 
  let time_stopped := (1 - s2 / s1) * 60 in
  time_stopped = 24 :=
by 
  sorry

end train_stoppage_time_l218_218143


namespace bacon_sold_l218_218029

variable (B : ℕ) -- Declare the variable for the number of slices of bacon sold

-- Define the given conditions as Lean definitions
def pancake_price := 4
def bacon_price := 2
def stacks_sold := 60
def total_raised := 420

-- The revenue from pancake sales alone
def pancake_revenue := stacks_sold * pancake_price
-- The revenue from bacon sales
def bacon_revenue := total_raised - pancake_revenue

-- Statement of the theorem
theorem bacon_sold :
  B = bacon_revenue / bacon_price :=
sorry

end bacon_sold_l218_218029


namespace sum_of_all_four_digit_numbers_l218_218236

-- Let us define the set of digits
def digits : set ℕ := {1, 2, 3, 4, 5}

-- We will define a function that generates the four-digit numbers
def four_digit_numbers := {n : ℕ // ∃ a b c d : ℕ, 
                                      a ∈ digits ∧ 
                                      b ∈ digits ∧ 
                                      c ∈ digits ∧ 
                                      d ∈ digits ∧ 
                                      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ 
                                      b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                                      n = 1000 * a + 100 * b + 10 * c + d}

-- Define a function to calculate the sum of all elements in a set of numbers
def sum_set (s : set ℕ) : ℕ := s.fold (λa b, a + b) 0

theorem sum_of_all_four_digit_numbers :
  sum_set {n | ∃ x : four_digit_numbers, x.val = n} = 399960 :=
sorry

end sum_of_all_four_digit_numbers_l218_218236


namespace remainder_of_x_squared_mod_25_l218_218766

theorem remainder_of_x_squared_mod_25 :
  (5 * x ≡ 10 [MOD 25]) → (4 * x ≡ 20 [MOD 25]) → ((x ^ 2) % 25 = 4) := by
  intro h1 h2
  sorry

end remainder_of_x_squared_mod_25_l218_218766


namespace possible_values_of_x_l218_218718

theorem possible_values_of_x (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) 
    (h1 : x + 1 / z = 15) (h2 : z + 1 / x = 9 / 20) :
    x = (15 + 5 * Real.sqrt 11) / 2 ∨ x = (15 - 5 * Real.sqrt 11) / 2 :=
by
  sorry

end possible_values_of_x_l218_218718


namespace count_positive_integers_l218_218754

theorem count_positive_integers (n : ℤ) : 
  (130 * n) ^ 50 > (n : ℤ) ^ 100 ∧ (n : ℤ) ^ 100 > 2 ^ 200 → 
  ∃ k : ℕ, k = 125 := sorry

end count_positive_integers_l218_218754


namespace hyperbola_eccentricity_l218_218738

theorem hyperbola_eccentricity :
  ∀ (a b : ℝ) (x₁ y₁ : ℝ),
    0 < a →
    0 < b →
    (x₁ / a) ^ 2 - (y₁ / b) ^ 2 = 1 →
    (y₁ = (3 * real.sqrt 7 / 7) * x₁) →
    let (c := 2) in  -- c = sqrt(c²) where c² = 4 by given focus (2, 0)
    (b ^ 2 = c ^ 2 - a ^ 2) →
    (real.sqrt (c ^ 2) / a = 2) := sorry

end hyperbola_eccentricity_l218_218738


namespace robot_movement_l218_218710

theorem robot_movement (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
    ∃ n, (∀ k : ℤ, k = n - a + b * m → k ≡ 0 [MOD a] ∨ k ≡ 0 [MOD b]) → 
        n = a + b - Int.gcd a b := 
sorry

end robot_movement_l218_218710


namespace sum_four_digit_numbers_l218_218258

theorem sum_four_digit_numbers : 
  let digits := [1, 2, 3, 4, 5]
  let perms := digits.permutations
  ∑ p in perms.filter (λ x, x.length = 4), (1000 * x.head + 100 * x[1] + 10 * x[2] + x[3]) = 399960 := 
by sorry

end sum_four_digit_numbers_l218_218258


namespace eccentricity_range_l218_218723

open Real

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 - (b^2 / a^2))

theorem eccentricity_range (a b : ℝ) (hab : 0 < b) (ha : 0 < a) (h : b < a)
  (hP : ∃ (m n : ℝ), m^2 + n^2 = 2 * b^2 ∧ m^2 / a^2 + n^2 / b^2 = 1) :
  ∃ (e : ℝ), eccentricity a b = e ∧ ∀ e, e ∈ Icc (sqrt 2 / 2) 1 :=
sorry

end eccentricity_range_l218_218723


namespace product_of_primes_l218_218105

theorem product_of_primes : 2 * 3 * 11 = 66 :=
by 
  -- Start with the multiplication of the first two primes
  have h1 : 2 * 3 = 6 := by norm_num
  -- Then multiply the result with the smallest two-digit prime
  have h2 : 6 * 11 = 66 := by norm_num
  -- Combine the steps to get the final result
  exact eq.trans (congr_arg (λ x, x * 11) h1) h2

end product_of_primes_l218_218105


namespace charlie_first_week_usage_l218_218845

noncomputable def data_used_week1 : ℕ :=
  let data_plan := 8
  let week2_usage := 3
  let week3_usage := 5
  let week4_usage := 10
  let total_extra_cost := 120
  let cost_per_gb_extra := 10
  let total_data_used := data_plan + (total_extra_cost / cost_per_gb_extra)
  let total_data_week_2_3_4 := week2_usage + week3_usage + week4_usage
  total_data_used - total_data_week_2_3_4

theorem charlie_first_week_usage : data_used_week1 = 2 :=
by
  sorry

end charlie_first_week_usage_l218_218845


namespace predicted_sales_volume_l218_218139

-- Define the linear regression equation
def regression_equation (x : ℝ) : ℝ := 2 * x + 60

-- Use the given condition x = 34
def temperature_value : ℝ := 34

-- State the theorem that the predicted sales volume is 128
theorem predicted_sales_volume : regression_equation temperature_value = 128 :=
by
  sorry

end predicted_sales_volume_l218_218139


namespace combine_piles_l218_218423

theorem combine_piles (n : ℕ) (piles : list ℕ) (h_piles : list.sum piles = n) (h_similar : ∀ x y ∈ piles, x ≤ y → y ≤ 2 * x) :
  ∃ pile, pile ∈ piles ∧ pile = n := sorry

end combine_piles_l218_218423


namespace find_m_of_inverse_proportion_l218_218336

theorem find_m_of_inverse_proportion (k : ℝ) (m : ℝ) 
(A_cond : (-1) * 3 = k) 
(B_cond : 2 * m = k) : 
m = -3 / 2 := 
by 
  sorry

end find_m_of_inverse_proportion_l218_218336


namespace sqrt_subtraction_l218_218563

-- Define the initial expressions under the square roots
def expr1 : ℝ := 49 + 121
def expr2 : ℝ := 64 + 16

-- Define the simplified forms
def sqrt_expr1 : ℝ := Real.sqrt expr1
def sqrt_expr2 : ℝ := Real.sqrt expr2

-- Define the term we expect after simplification
def simplified_result : ℝ := Real.sqrt 170 - 4 * Real.sqrt 5

-- The theorem statement
theorem sqrt_subtraction :
  sqrt_expr1 - sqrt_expr2 = simplified_result := by
  sorry

end sqrt_subtraction_l218_218563


namespace proof_equation_ellipse_proof_equation_min_line_l218_218291

noncomputable def equation_of_ellipse : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ (x y : ℝ), (x, y) = (-1, 0) ∨ (x, y) = (1, 0) → x^2/a^2 + y^2/b^2 = 1) ∧
  (1, 3/2) ∈ { (x, y) | x^2/4 + y^2/3 = 1 }

noncomputable def equation_of_min_line : Prop :=
  ∃ (k m : ℝ), m > 0 ∧
  y = kx + m ∧
  (k ≠ 0 ∧ y = kx + m ∧
  ∃ (x y : ℝ), (x, y) ∈ { (x, y) | x^2/4 + y^2/3 = 1 } ∧
  |m|^2 = 4k^2 + 3 ∧
  (1/2) * |m| * |m/k| = 2 * sqrt 3 ∧
  (y = sqrt(3)/2 * x + sqrt(6) ∨ y = -sqrt(3)/2 * x + sqrt(6)))

theorem proof_equation_ellipse : equation_of_ellipse := sorry

theorem proof_equation_min_line : equation_of_min_line := sorry

end proof_equation_ellipse_proof_equation_min_line_l218_218291


namespace non_sum_set_A_non_sum_set_B_not_both_non_sum_sets_max_elements_in_non_sum_set_l218_218768

-- Problem 1 Statement
theorem non_sum_set_A {A : Set ℕ} (hA : A = {1, 3, 5, 7, 9}) : 
    ∀ x y ∈ A, x + y ∉ A := sorry

theorem non_sum_set_B {B : Set ℕ} (hB : B = {5, 6, 7, 8, 9, 10}) : 
    ∃ x y ∈ B, x + y ∈ B := sorry

-- Problem 2 Statement
theorem not_both_non_sum_sets (n : ℕ) (h : n ≥ 5) 
    (A B : Set ℕ) (h_union : A ∪ B = {1, 2, ..., n}) (h_inter : A ∩ B = ∅) : 
    ¬(∀ x y ∈ A, x + y ∉ A) ∨ ¬(∀ x y ∈ B, x + y ∉ B) := sorry

-- Problem 3 Statement
theorem max_elements_in_non_sum_set (n : ℕ) (h : n ≥ 3) 
    (A : Set ℕ) (h_subset : A ⊆ {1, 2, ..., n}) (h_non_sum : ∀ x y ∈ A, x + y ∉ A) : 
    (n % 2 = 0 → ∃ s, s.card = n / 2 ∧ ∀ a ∈ s, a ∈ A) ∧ 
    (n % 2 = 1 → ∃ s, s.card = (n + 1) / 2 ∧ ∀ a ∈ s, a ∈ A) := sorry

end non_sum_set_A_non_sum_set_B_not_both_non_sum_sets_max_elements_in_non_sum_set_l218_218768


namespace problem_l218_218025

noncomputable def estimate_p (p : ℝ) (P : ℕ) :=
  ∀ (A B : ℕ) (score_general : ℕ → ℕ) (score_theme : ℕ → ℕ),
    (∀ x y, x ≠ y → (score_general x, score_theme y) ≠ (score_general y, score_theme x)) → 
    (p = (1 - ((score_general A / score_theme B) * (score_theme A / score_general B)))) →
    P = ⌊10000 * p⌋

theorem problem (p : ℝ) (P : ℕ) (A B : ℕ) (score_general score_theme : ℕ → ℕ)
  (h_distinct : ∀ x y, x ≠ y → (score_general x, score_theme y) ≠ (score_general y, score_theme x))
  (h_p : p = (1 - ((score_general A / score_theme B) * (score_theme A / score_general B)))) :
  P = 2443 :=
begin
  sorry,
end

end problem_l218_218025


namespace m_values_subset_l218_218380

noncomputable def possible_m_values (S : Finset ℕ) (H : S ⊆ Finset.range 99) (m : ℕ) (h : S.card = m) 
  (hm : 3 ≤ m) (hxhy : ∀ x y ∈ S, ∃ z ∈ S, (x + y) % 99 = (2 * z) % 99) : Finset ℕ :=
{3, 9, 11, 33, 99}

theorem m_values_subset (S : Finset ℕ) (H : S ⊆ Finset.range 99) (m : ℕ) (h : S.card = m) 
  (hm : 3 ≤ m) (hxhy : ∀ x y ∈ S, ∃ z ∈ S, (x + y) % 99 = (2 * z) % 99) : 
  m ∈ possible_m_values S H m h hm hxhy :=
sorry

end m_values_subset_l218_218380


namespace limit_of_function_l218_218638

theorem limit_of_function :
  (∃ ε > 0, ∀ x : ℝ, |x| < ε → |(2 + real.log (real.exp 1 + x * real.sin (1 / x)) - 3) / (real.cos x + real.sin x) - 3| < 1) ∧
  (∀ x : ℝ, |real.sin (1 / x)| ≤ 1) ∧
  (∀ x : ℝ, real.exp 1 + x * real.sin (1 / x) > 0) :=
begin
  sorry
end

end limit_of_function_l218_218638


namespace inequality_sqrt_sin_cos_tg_ctg_l218_218010

theorem inequality_sqrt_sin_cos_tg_ctg (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : 
    (Real.sqrt (Real.sin x) * Real.sqrt (Real.sqrt (Real.sqrt (Real.tan x))) +
    Real.sqrt (Real.cos x) * Real.sqrt (Real.sqrt (Real.sqrt (Real.cot x)))) 
    ≥ Real.sqrt (Real.sqrt 2) :=
sorry

end inequality_sqrt_sin_cos_tg_ctg_l218_218010


namespace sphere_radius_eq_three_of_volume_eq_surface_area_l218_218306

theorem sphere_radius_eq_three_of_volume_eq_surface_area
  (r : ℝ) 
  (h1 : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : 
  r = 3 :=
sorry

end sphere_radius_eq_three_of_volume_eq_surface_area_l218_218306


namespace general_term_formula_sum_first_n_terms_range_of_a_l218_218729

section Problem1
variables {a_n : ℕ → ℝ} {S_n: ℕ → ℝ}

-- Given conditions
axiom seq_increasing (n : ℕ) : a_n (n + 1) ≥ a_n n
axiom sum_relation (n : ℕ) : 2 * S_n n = (a_n n)^2 + n

-- Prove the general term formula
theorem general_term_formula (n : ℕ) : a_n n = n := sorry
end Problem1

section Problem2
variables {T_n : ℕ → ℝ} {b_n : ℕ → ℝ}

-- Given conditions
axiom b_n_definition (n : ℕ): b_n n = 4 * (a_n n)^2 / (4 * (a_n n)^2 - 1)
axiom a_n_formula (n : ℕ): a_n n = n

-- Prove the sum of the first n terms
theorem sum_first_n_terms (n : ℕ) : T_n n = n + 1/2 - 1/(4*n+2) := sorry
end Problem2

section Problem3
variables {a : ℝ} {n : ℕ}

-- Given conditions
axiom T_n_definition (n : ℕ) : T_n n = n + 1/2 - 1/(4*n+2)

-- Prove the range of values for a
theorem range_of_a (a : ℝ) (n : ℕ) : 
  (∀ n : ℕ, T_n n + (-1)^(n+1) * a - n > 0) ↔ (-1/3 < a ∧ a < 2/5) := sorry
end Problem3

end general_term_formula_sum_first_n_terms_range_of_a_l218_218729


namespace total_completion_time_l218_218335

variables {n k t : ℕ}

-- Condition 1: n women can complete 3/4 of a job in t days.
def work_rate (n : ℕ) (t : ℕ) : ℚ := 3 / (4 * t)

-- Condition 2: n+k women complete the remaining 1/4 of the job.
def remaining_work_time (n k : ℕ) : ℚ := 4 * (n + k)

-- Conclusion: The total time taken to complete the entire job.
theorem total_completion_time (n k t : ℕ) :
  ∀ (work_rate n t = 3 / (4 * t)), ∀ (remaining_work_time n k = 4 * (n + k)),
  total_time = t + 4 * (n + k) := 
sorry

end total_completion_time_l218_218335


namespace binomial_coefficient_sum_l218_218989

theorem binomial_coefficient_sum :
  Nat.choose 10 3 + Nat.choose 10 2 = 165 := by
  sorry

end binomial_coefficient_sum_l218_218989


namespace sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218245

theorem sum_of_four_digit_numbers_formed_by_digits_1_to_5 :
  let S := {1, 2, 3, 4, 5}
  let four_digits_sum (n1 n2 n3 n4 : ℕ) :=
    1000 * n1 + 100 * n2 + 10 * n3 + n4
  (∀ a b c d ∈ S, a ≠ b → b ≠ c → c ≠ d → d ≠ a → a ≠ c → b ≠ d 
  → sum (four_digits_sum a b c d) = 399960) := sorry

end sum_of_four_digit_numbers_formed_by_digits_1_to_5_l218_218245


namespace exists_root_interval_l218_218199

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_interval :
  (f 1.1 < 0) ∧ (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 := 
by
  intro h
  sorry

end exists_root_interval_l218_218199


namespace total_cost_for_tickets_l218_218589

-- Define the known quantities
def students : Nat := 20
def teachers : Nat := 3
def ticket_cost : Nat := 5

-- Define the total number of people
def total_people : Nat := students + teachers

-- Define the total cost
def total_cost : Nat := total_people * ticket_cost

-- Prove that the total cost is $115
theorem total_cost_for_tickets : total_cost = 115 := by
  -- Sorry is used here to skip the proof
  sorry

end total_cost_for_tickets_l218_218589


namespace find_polynomial_l218_218228

-- Definitions of the conditions
variables {R : Type*} [CharZero R] (n : ℕ) (a_0 : R) (f : R[X])
variable (C_2n_n : ℕ)

-- Conditions
def condition1 (a_0 : R) (f : R[X]) : Prop :=
  ∃ (a : Fin (n+1) → R), f = a_0 * ∑ i in Finset.range (n+1), a i * X^(2 * (n - i))

def condition2 (a_0 : R) (f : R[X]) : Prop :=
  (∃ a : Fin (n + 1) → R, f = a_0 * ∑ i in Finset.range (n + 1), a i * X^(2 * (n - i))) ∧
  (∑ j in Finset.range (n + 1), a j * a (n - j) ≤ C_2n_n * a_0 * a n)

def condition3 (f : R[X]) : Prop :=
  ∀ (z : R), root f z → purely_imaginary z

-- The main proof statement
theorem find_polynomial (n : ℕ) (a_0 : R) (f : R[X]) (C_2n_n : ℕ)
  (h1 : condition1 n a_0 f)
  (h2 : condition2 n a_0 f C_2n_n)
  (h3 : condition3 f) :
  ∃ β : R, β > 0 ∧ f = a_0 * (X^2 + β^2)^n := 
sorry

end find_polynomial_l218_218228


namespace sequence_property_l218_218282

theorem sequence_property (a : ℕ → ℝ) (h : ∀ n : ℕ, ∑ i in range (n + 1), a i = 2 * a 1) : a 1 = 0 :=
by 
  have h1 : a 0 = 2 * a 1 := h 0
  have h2 : a 0 + a 1 = 2 * a 1 := h 1
  rw [h1] at h2
  rw [add_comm] at h2
  linarith

end sequence_property_l218_218282


namespace freshman_class_total_students_l218_218626

theorem freshman_class_total_students (N : ℕ) 
    (h1 : 90 ≤ N) 
    (h2 : 100 ≤ N)
    (h3 : 20 ≤ N) 
    (h4: (90 : ℝ) / N * (20 : ℝ) / 100 = (20 : ℝ) / N):
    N = 450 :=
  sorry

end freshman_class_total_students_l218_218626


namespace surface_area_of_cube_l218_218745

-- Define the points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def P : Point3D := ⟨10, 15, 14⟩
noncomputable def Q : Point3D := ⟨11, 11, 5⟩
noncomputable def R : Point3D := ⟨14, 6, 13⟩

-- Function to compute the distance between two points in 3D space
noncomputable def distance (A B : Point3D) : ℝ :=
  Real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2 + (B.z - A.z) ^ 2)

-- The Lean theorem statement
theorem surface_area_of_cube : 
  distance P Q = distance P R ∧ distance P Q = distance Q R →
  ∃ a : ℝ, surface_area = 6 * a ^ 2 ∧ a = 7 := 
begin
  sorry
end

end surface_area_of_cube_l218_218745


namespace largest_prime_factor_1581_l218_218560

theorem largest_prime_factor_1581 : ∃ p : ℕ, prime p ∧ p ∣ 1581 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1581 → q ≤ p :=
by
  sorry

end largest_prime_factor_1581_l218_218560


namespace probability_sum_18_l218_218061

open BigOperators

-- Definitions of the dice ranges
def Dice1 := {x : ℕ | 1 ≤ x ∧ x ≤ 4}
def Dice2 := {x : ℕ | 5 ≤ x ∧ x ≤ 8}
def Dice3 := {x : ℕ | 9 ≤ x ∧ x ≤ 12}

-- Noncomputable because we are dealing with probabilities
noncomputable def count {α : Type*} (s : Set α) : ℕ := 
  if s.finite then s.to_finset.card else 0

noncomputable def total_outcomes : ℕ :=
  count Dice1 * count Dice2 * count Dice3

-- Defining the predicate for favorable outcomes
def favorable_outcomes (x y z : ℕ) : Prop := 
  x + y + z = 18 ∧ x ∈ Dice1 ∧ y ∈ Dice2 ∧ z ∈ Dice3

-- Statement to prove
theorem probability_sum_18 : 
  (count {p : ℕ × ℕ × ℕ | favorable_outcomes p.1 p.2 p.3}).toReal / total_outcomes.toReal = 3 / 32 := 
by
  sorry

end probability_sum_18_l218_218061


namespace cos_phi_value_l218_218947

-- Define the two direction vectors as vectors in R²
def line1_vec : ℝ × ℝ := (2, 5)
def line2_vec : ℝ × ℝ := (4, 1)

-- Compute the cosine of the angle between the two vectors
def cos_phi : ℝ :=
  let dot_prod := line1_vec.1 * line2_vec.1 + line1_vec.2 * line2_vec.2
  let norm1 := Real.sqrt (line1_vec.1 ^ 2 + line1_vec.2 ^ 2)
  let norm2 := Real.sqrt (line2_vec.1 ^ 2 + line2_vec.2 ^ 2)
  dot_prod / (norm1 * norm2)

-- The theorem asserting the computed cosine value is equal to the target fraction
theorem cos_phi_value : cos_phi = 13 / Real.sqrt 493 :=
by
  sorry

end cos_phi_value_l218_218947


namespace apollo_gold_apples_l218_218188

theorem apollo_gold_apples :
  let first_six_months = 6
  let rate_first_six_months = 3
  let rate_second_six_months = rate_first_six_months * 2
  let total_first_six_months = first_six_months * rate_first_six_months
  let total_second_six_months = first_six_months * rate_second_six_months
  total_first_six_months + total_second_six_months = 54 := 
by 
  sorry

end apollo_gold_apples_l218_218188


namespace train_speed_correct_l218_218587

noncomputable def train_length : ℝ := 250
noncomputable def platform_length : ℝ := 300
noncomputable def crossing_time : ℝ := 35.99712023038157

noncomputable def train_speed_m_per_s : ℝ :=
  (train_length + platform_length) / crossing_time

noncomputable def train_speed_km_per_hr : ℝ :=
  train_speed_m_per_s * 3.6

theorem train_speed_correct : 
  train_speed_km_per_hr ≈ 54.9996 := by
    sorry

end train_speed_correct_l218_218587


namespace two_inverse_exponent_l218_218758

theorem two_inverse_exponent (y : ℚ) (h : 128^7 = 16^y) : 2^(-y) = 1 / 2^(49 / 4) :=
by sorry

end two_inverse_exponent_l218_218758


namespace plane_can_be_colored_l218_218523

-- Define a structure for a triangle and the plane divided into triangles
structure Triangle :=
(vertices : Fin 3 → ℕ) -- vertices labeled with ℕ, interpreted as 0, 1, 2

structure Plane :=
(triangles : Set Triangle)
(adjacent : Triangle → Triangle → Prop)
(labels_correct : ∀ {t1 t2 : Triangle}, adjacent t1 t2 → 
  ∀ i j: Fin 3, t1.vertices i ≠ t1.vertices j)
(adjacent_conditions: ∀ t1 t2: Triangle, adjacent t1 t2 → 
  ∃ v, (∃ i: Fin 3, t1.vertices i = v) ∧ (∃ j: Fin 3, t2.vertices j = v))

theorem plane_can_be_colored (p : Plane) : 
  ∃ (c : Triangle → ℕ), (∀ t1 t2, p.adjacent t1 t2 → c t1 ≠ c t2) :=
sorry

end plane_can_be_colored_l218_218523


namespace complex_transformation_result_l218_218190

/--
  Apply a 60 degree rotation around the origin in the counter-clockwise direction and 
  a dilation centered at the origin with scale factor 2 to the complex number -4 + 3i.
  Prove that the resulting complex number is 5 - sqrt 3 * i.
-/
theorem complex_transformation_result :
  let z : ℂ := -4 + 3 * complex.I
  let rotation : ℂ := 1/2 + (real.sqrt 3) / 2 * complex.I
  let dilation : ℂ := 2
  (z * (rotation * dilation) = 5 - (real.sqrt 3) * complex.I) :=
by {
  sorry
}

end complex_transformation_result_l218_218190


namespace average_annual_growth_rate_eq_l218_218354

-- Definition of variables based on given conditions
def sales_2021 := 298 -- in 10,000 units
def sales_2023 := 850 -- in 10,000 units
def years := 2

-- Problem statement in Lean 4
theorem average_annual_growth_rate_eq :
  sales_2021 * (1 + x) ^ years = sales_2023 :=
sorry

end average_annual_growth_rate_eq_l218_218354


namespace find_y_l218_218051

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l218_218051


namespace competition_places_l218_218469

def participants := ["Olya", "Oleg", "Polya", "Pasha"]
def placements := Array.range 1 5

-- Define statements made by each child
def Olya_claims_odd_boys (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → (name = "Oleg" ∨ name = "Pasha")

def Oleg_claims_consecutive_with_Olya (placement : String → Nat) : Prop :=
  abs (placement "Oleg" - placement "Olya") = 1

def Pasha_claims_odd_O_names (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → name.startsWith "O"

-- Define the main problem statement
theorem competition_places :
  ∃ (placement : String → Nat),
    placement "Oleg" = 1 ∧
    placement "Olya" = 2 ∧
    placement "Polya" = 3 ∧
    placement "Pasha" = 4 ∧
    (∃ name, (name = "Oleg" ∨ name = "Olya" ∨ name = "Polya" ∨ name = "Pasha") ∧
      ((name = "Oleg" → (placement "Oleg" = 1 ∧ Oleg_claims_consecutive_with_Olya placement)) ∧
       (name = "Olya" → (placement "Olya" = 1 ∧ Olya_claims_odd_boys placement)) ∧
       (name = "Pasha" → (placement "Pasha" = 1 ∧ Pasha_claims_odd_O_names placement)))) :=
by
  have placement : String → Nat := λ name => match name with
    | "Olya" => 2
    | "Oleg" => 1
    | "Polya" => 3
    | "Pasha" => 4
    | _      => 0
  use placement
  simp [placement, Oleg_claims_consecutive_with_Olya, Olya_claims_odd_boys, Pasha_claims_odd_O_names]
  sorry

end competition_places_l218_218469


namespace sum_four_digit_numbers_l218_218246

def digits : List ℕ := [1, 2, 3, 4, 5]

/-- 
  Prove that the sum of all four-digit numbers that can be formed 
  using the digits 1, 2, 3, 4, 5 exactly once is 399960.
-/
theorem sum_four_digit_numbers : 
  (Finset.sum 
    (Finset.map 
      (λ l, 
        l.nth_le 0 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1000 + 
        l.nth_le 1 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 100 + 
        l.nth_le 2 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 10 + 
        l.nth_le 3 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1) 
      (digits.permutations.filter (λ l, l.nodup ∧ l.length = 4))) id) 
  = 399960 :=
sorry

end sum_four_digit_numbers_l218_218246


namespace competition_places_l218_218466

def participants := ["Olya", "Oleg", "Polya", "Pasha"]
def placements := Array.range 1 5

-- Define statements made by each child
def Olya_claims_odd_boys (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → (name = "Oleg" ∨ name = "Pasha")

def Oleg_claims_consecutive_with_Olya (placement : String → Nat) : Prop :=
  abs (placement "Oleg" - placement "Olya") = 1

def Pasha_claims_odd_O_names (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → name.startsWith "O"

-- Define the main problem statement
theorem competition_places :
  ∃ (placement : String → Nat),
    placement "Oleg" = 1 ∧
    placement "Olya" = 2 ∧
    placement "Polya" = 3 ∧
    placement "Pasha" = 4 ∧
    (∃ name, (name = "Oleg" ∨ name = "Olya" ∨ name = "Polya" ∨ name = "Pasha") ∧
      ((name = "Oleg" → (placement "Oleg" = 1 ∧ Oleg_claims_consecutive_with_Olya placement)) ∧
       (name = "Olya" → (placement "Olya" = 1 ∧ Olya_claims_odd_boys placement)) ∧
       (name = "Pasha" → (placement "Pasha" = 1 ∧ Pasha_claims_odd_O_names placement)))) :=
by
  have placement : String → Nat := λ name => match name with
    | "Olya" => 2
    | "Oleg" => 1
    | "Polya" => 3
    | "Pasha" => 4
    | _      => 0
  use placement
  simp [placement, Oleg_claims_consecutive_with_Olya, Olya_claims_odd_boys, Pasha_claims_odd_O_names]
  sorry

end competition_places_l218_218466


namespace pile_division_possible_l218_218416

theorem pile_division_possible (n : ℕ) :
  ∃ (division : list ℕ), (∀ x ∈ division, x = 1) ∧ division.sum = n :=
by
  sorry

end pile_division_possible_l218_218416


namespace find_principal_amount_l218_218965

theorem find_principal_amount
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (h1 : SI = 4034.25)
  (h2 : R = 9)
  (h3 : T = 5) :
  P = 8965 :=
by
  sorry

end find_principal_amount_l218_218965


namespace find_f2010_l218_218278

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       := λ x, cos x
| (n + 1) := λ x, (f n x)'

def periodicity := ∀ n: ℕ, f (n + 4) = f n

theorem find_f2010 (x : ℝ) (h : periodicity) : f 2010 x = -sin x :=
  sorry

end find_f2010_l218_218278


namespace jude_tickets_sold_l218_218898

/- 
  There are 100 tickets in total, Andrea sold twice as many tickets as Jude,
  Sandra sold 4 more than half the number of tickets Jude sold,
  and there are 40 tickets left to be sold.
  Prove that Jude sold 16 tickets.
-/

theorem jude_tickets_sold (J : ℕ)
  (total_tickets : ℕ := 100)
  (remaining_tickets : ℕ := 40)
  (andrea_tickets : ℕ := 2 * J)
  (sandra_tickets : ℕ := (1/2 : ℚ) * J + 4)
  (tickets_sold : ℚ := J + andrea_tickets + sandra_tickets) :
  tickets_sold = 60 → J = 16 :=
sorry

end jude_tickets_sold_l218_218898


namespace sum_of_all_four_digit_numbers_formed_l218_218267

open List

noncomputable def sum_of_four_digit_numbers (digits : List ℕ) : ℕ :=
  let perms := digits.permutations.filter (λ l, l.length = 4)
  let nums := perms.map (λ l, 1000 * l.head + 100 * l.nthLe 1 sorry + 10 * l.nthLe 2 sorry + l.nthLe 3 sorry)
  nums.sum

theorem sum_of_all_four_digit_numbers_formed : sum_of_four_digit_numbers [1, 2, 3, 4, 5] = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_formed_l218_218267


namespace greatest_x_for_A_is_perfect_square_l218_218675

theorem greatest_x_for_A_is_perfect_square :
  ∃ x : ℕ, x = 2008 ∧ ∀ y : ℕ, (∃ k : ℕ, 2^182 + 4^y + 8^700 = k^2) → y ≤ 2008 :=
by 
  sorry

end greatest_x_for_A_is_perfect_square_l218_218675


namespace part_a_part_b_l218_218389

def S (n k : ℕ) : ℕ := 
  let digits := (nat.get_digits k n).map (λ d, d + 1)
  digits.foldl (*) 1

theorem part_a : S 2012 3 = 324 :=
by sorry

theorem part_b : S (2012 ^ 2011) 2011 % 2012 = 0 :=
by sorry

end part_a_part_b_l218_218389


namespace max_value_f_l218_218512

noncomputable def max_val (f : ℝ → ℝ) : ℝ :=
  Sup (set.range f)

def f (x : ℝ) : ℝ := x - x^2

theorem max_value_f : max_val f = 1 / 4 :=
by
  sorry

end max_value_f_l218_218512


namespace angle_between_vectors_l218_218327

open Real

variables {a b : ℝ^3} -- Let a and b be vectors in 3-dimensional space ℝ^3

-- Given conditions
def condition_1 : Prop := (∥a + b∥ = 2 * ∥a∥)
def condition_2 : Prop := (∥a - b∥ = 2 * ∥a∥)

-- The proof problem
theorem angle_between_vectors (h1 : condition_1) (h2 : condition_2) :
  let θ := angle (a + b) (b - a) in
  θ = π / 3 := 
by 
  sorry

end angle_between_vectors_l218_218327


namespace bombardment_percentage_l218_218790

variable (P : ℝ := 4599)
variable (P_final : ℝ := 3312)

def percentage_died_from_bombardment (x : ℝ) : Prop :=
  0.80 * P * (1 - x / 100) = P_final

theorem bombardment_percentage : 
  ∃ (x : ℝ), percentage_died_from_bombardment x ∧ abs (x - 9.98) < 0.01 :=
by
  sorry

end bombardment_percentage_l218_218790


namespace max_value_Sn_for_seq_a_max_value_Sn_for_seq_b_l218_218292

-- Define the sequence a_n recursively with the given initial value
def seq_a (n : ℕ) : ℤ 
| 0     := 8
| (n+1) := seq_a n - 2

-- Define the sequence b_n recursively with the given initial value
def seq_b (n : ℕ) : ℤ 
| 0     := 8
| (n+1) := seq_b n - n

-- Define S_n as the sum of the first n terms of a given sequence
def sum_seq (f : ℕ → ℤ) (n : ℕ) : ℤ :=
(nat.sum (finset.range n) f)

-- Proof outline: We need to show that under these definitions, S_n achieves a maximum value for the given sequences.
theorem max_value_Sn_for_seq_a : ∃ n, is_max (sum_seq seq_a n) :=
sorry

theorem max_value_Sn_for_seq_b : ∃ n, is_max (sum_seq seq_b n) :=
sorry

end max_value_Sn_for_seq_a_max_value_Sn_for_seq_b_l218_218292


namespace line_through_points_l218_218326

variable (A1 B1 A2 B2 : ℝ)

def line1 : Prop := -7 * A1 + 9 * B1 = 1
def line2 : Prop := -7 * A2 + 9 * B2 = 1

theorem line_through_points (h1 : line1 A1 B1) (h2 : line1 A2 B2) :
  ∃ (k : ℝ), (∀ (x y : ℝ), y - B1 = k * (x - A1)) ∧ (-7 * (x : ℝ) + 9 * y = 1) := 
by sorry

end line_through_points_l218_218326


namespace num_distinct_products_of_divisors_l218_218825

def divisor_set (n : ℕ) : Set ℕ := {d | d ∣ n ∧ 0 < d}

def generate_elements (a b c x y z : ℕ) : ℕ := 
  if (a, b, c) ≠ (x, y, z)
  then (2^(a + x)) * (3^(b + y)) * (5^(c + z))
  else 0

theorem num_distinct_products_of_divisors :
  let U := divisor_set 36000 in
  let num_elems := 311 in
  (∃ s : Set ℕ, s = {u | ∃ (a b c x y z : ℕ), u = generate_elements a b c x y z}) → 
  s.card = num_elems :=
sorry

end num_distinct_products_of_divisors_l218_218825
