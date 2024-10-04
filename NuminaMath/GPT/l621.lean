import Mathlib

namespace average_rate_of_decline_predicted_price_2015_l621_621399

variables (p2012 p2014 p2015 : ℝ) (x : ℝ)

-- Introduce the conditions as definitions
def price_2012 : p2012 = 5000 := rfl
def price_2014 : p2014 = 4050 := rfl
def decline_condition : p2012 * (1 - x) ^ 2 = p2014 := sorry  -- This needs to be proven or assumed

-- Statement for proving the rate of decline
theorem average_rate_of_decline : x = 0.1 :=
by sorry

-- Using the rate of decline, prove the predicted price in 2015
theorem predicted_price_2015 (h : x = 0.1) : p2015 = 3645 :=
by sorry

end average_rate_of_decline_predicted_price_2015_l621_621399


namespace ceil_neg_sqrt_eq_neg_two_l621_621284

noncomputable def x : ℝ := -Real.sqrt (64 / 9)

theorem ceil_neg_sqrt_eq_neg_two : Real.ceil x = -2 := by
  exact sorry

end ceil_neg_sqrt_eq_neg_two_l621_621284


namespace find_number_l621_621759

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l621_621759


namespace true_propositions_l621_621306

variable (a b c : ℝ)

-- Proposition 2: "a + 5 is irrational" is a necessary and sufficient condition for "a is irrational".
def prop2 : Prop := 
  (irrational (a + 5) ↔ irrational a)

-- Proposition 4: "a < 5" is a necessary condition for "a < 3".
def prop4 : Prop := 
  (a < 3 → a < 5)

theorem true_propositions (a b : ℝ) : prop2 a ∧ prop4 a := by
  sorry

end true_propositions_l621_621306


namespace convert_binary_to_base4_l621_621733

theorem convert_binary_to_base4 : 
  (convert_to_base4 10110010₂) = 2302₄ :=
by
  sorry

end convert_binary_to_base4_l621_621733


namespace min_number_of_participants_l621_621504

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l621_621504


namespace equal_cost_sharing_l621_621686

variable (X Y Z : ℝ)
variable (h : X < Y ∧ Y < Z)

theorem equal_cost_sharing :
  ∃ (amount : ℝ), amount = (Y + Z - 2 * X) / 3 := 
sorry

end equal_cost_sharing_l621_621686


namespace sequence_bn_convergence_l621_621544

noncomputable theory

open_locale big_operators

open Filter Finset

variables {α : Type*} [LinearOrderedField α] {a : ℕ → α}

theorem sequence_bn_convergence
  (h : Summable (λ n, a n / n)) :
  Tendsto (λ n : ℕ, (1 / n) * ∑ i in range n, a i) at_top (nhds 0) :=
begin
  sorry
end

end sequence_bn_convergence_l621_621544


namespace four_digit_prime_product_l621_621781

open Nat

theorem four_digit_prime_product :
  ∃ (p q r s n : ℕ), Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p < q ∧ q < r ∧
  p + q = r - q ∧ p + q + r = s^2 ∧ 1000 ≤ p * q * r ∧ p * q * r ≤ 9999 ∧ n = p * q * r ∧ n = 2015 :=
by
  sorry

end four_digit_prime_product_l621_621781


namespace sum_of_s_r_values_l621_621929

def r_range := {-2, 0, 2, 4, 6}
def s_domain := {0, 1, 2, 3, 4}

noncomputable def r (x : ℤ) : ℤ :=
if x ∈ {-2, -1, 0, 1, 2} then (if x = -2 then -2 else if x = -1 then 0 else if x = 0 then 2 else if x = 1 then 4 else 6) else 0

def s (x : ℤ) : ℤ :=
  x^2 + 1

theorem sum_of_s_r_values : ((s (0 ^ 2 + 1)) + (s (2 ^ 2 + 1)) + (s (4 ^ 2 + 1))) = 23 := by sorry

end sum_of_s_r_values_l621_621929


namespace boundary_length_of_square_l621_621667

theorem boundary_length_of_square (side_length : ℝ) (total_area : ℝ) : 
  side_length = 12 → 
  total_area = 144 →
  (let segment_length := side_length / 4 in 
   let straight_length := 4 * (side_length / 2) in
   let arc_length := 6 * Real.pi in
   let boundary_length := straight_length + arc_length in
   boundary_length ≈ 42.8) :=
begin
  sorry
end

end boundary_length_of_square_l621_621667


namespace digit_sum_26_l621_621408

theorem digit_sum_26 
  (A B C D E : ℕ)
  (h1 : 1 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 0 ≤ E ∧ E ≤ 9)
  (h6 : 100000 + 10000 * A + 1000 * B + 100 * C + 10 * D + E * 3 = 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1):
  A + B + C + D + E = 26 
  := 
  by
    sorry

end digit_sum_26_l621_621408


namespace solve_inequality_l621_621784

theorem solve_inequality (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
  (x^2 / (x + 1) ≥ 2 / (x - 1) + 5 / 4) ↔ (x ∈ set.Ioo (-1) 1 ∪ set.Ici 3) :=
sorry

end solve_inequality_l621_621784


namespace problem_statement_l621_621558

theorem problem_statement (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ) 
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) : 
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := 
sorry

end problem_statement_l621_621558


namespace find_number_l621_621773

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l621_621773


namespace exponentiation_addition_l621_621752

theorem exponentiation_addition : (3^3)^2 + 1 = 730 := by
  sorry

end exponentiation_addition_l621_621752


namespace height_of_taller_tree_l621_621987

-- Define the conditions as hypotheses:
variables (h₁ h₂ : ℝ)
-- The top of one tree is 24 feet higher than the top of another tree
variables (h_difference : h₁ = h₂ + 24)
-- The heights of the two trees are in the ratio 2:3
variables (h_ratio : h₂ / h₁ = 2 / 3)

theorem height_of_taller_tree : h₁ = 72 :=
by
  -- This is the place where the solution steps would be applied
  sorry

end height_of_taller_tree_l621_621987


namespace james_pistachio_price_l621_621031

def ounces_per_can : ℝ := 5
def daily_consumption_ounces : ℝ := 30 / 5
def weekly_expenditure_dollars : ℝ := 84

theorem james_pistachio_price :
  weekly_expenditure_dollars / ((7 / 5) * daily_consumption_ounces / ounces_per_can) = 10 := 
by
  sorry

end james_pistachio_price_l621_621031


namespace each_student_contribution_l621_621648

-- Definitions for conditions in the problem
def numberOfStudents : ℕ := 30
def totalAmount : ℕ := 480
def numberOfFridaysInTwoMonths : ℕ := 8

-- Statement to prove
theorem each_student_contribution (numberOfStudents : ℕ) (totalAmount : ℕ) (numberOfFridaysInTwoMonths : ℕ) : 
  totalAmount / (numberOfFridaysInTwoMonths * numberOfStudents) = 2 := 
by
  sorry

end each_student_contribution_l621_621648


namespace solve_problem_l621_621779

def question : ℝ := -7.8
def answer : ℕ := 22

theorem solve_problem : 2 * (⌊|question|⌋) + (|⌊question⌋|) = answer := by
  sorry

end solve_problem_l621_621779


namespace ranch_cows_variance_l621_621659

variable (n : ℕ)
variable (p : ℝ)

-- Definition of the variance of a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem ranch_cows_variance : 
  binomial_variance 10 0.02 = 0.196 :=
by
  sorry

end ranch_cows_variance_l621_621659


namespace find_unknown_number_l621_621778

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l621_621778


namespace find_f_l621_621914

-- Define the set of positive integers as a subset of natural numbers
def is_pos_int (n : ℕ) : Prop := n > 0

-- Define the function f with the property we need to prove
def satisfies_property (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, is_pos_int n → (n-1)^2 < f(n) * f(f(n)) ∧ f(n) * f(f(n)) < n ^ 2 + n

-- Final theorem statement
theorem find_f : ∀ f : ℕ → ℕ, satisfies_property f → (∀ n : ℕ, is_pos_int n → f(n) = n) :=
by
  intros f h n hn
  sorry

end find_f_l621_621914


namespace nylon_per_cat_collar_l621_621032

def nylon_per_dog_collar : ℕ := 18
def total_nylon : ℕ := 192
def num_dog_collars : ℕ := 9
def num_cat_collars : ℕ := 3

-- Theorem statement
theorem nylon_per_cat_collar :
  let nylon_for_dogs := num_dog_collars * nylon_per_dog_collar in
  let nylon_for_cats := total_nylon - nylon_for_dogs in
  nylon_for_cats / num_cat_collars = 10 :=
by
  sorry

end nylon_per_cat_collar_l621_621032


namespace problem_statement_l621_621426

theorem problem_statement (a b c d : ℝ) (m n : ℕ) (rel_prime : Nat.coprime m n)
    (h1 : a + b = -3)
    (h2 : ab + bc + ca = -4)
    (h3 : abc + bcd + cda + dab = 14)
    (h4 : abcd = 30)
    (h5 : a^2 + b^2 + c^2 + d^2 = (m / n : ℝ)) :
  m = 141 ∧ n = 4 ∧ m + n = 145 :=
by
  sorry

end problem_statement_l621_621426


namespace correct_statements_l621_621223

theorem correct_statements :
  (∀ {A B C D : Type} [linear_order A] [linear_order B] [linear_order C] [linear_order D],
     collinear {A, B, C, D} →
     num_segments {A, B, C, D} = 6) →
  (∀ {P Q : Type} [linear_order P] [linear_order Q],
     P ≠ Q →
     ∃! l : Line, P ∈ l ∧ Q ∈ l) →
  (∀ θ : ℝ, acute θ → 
     supplement θ > complement θ) →
  True := 
begin
  sorry
end

end correct_statements_l621_621223


namespace minimum_participants_l621_621491

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l621_621491


namespace reconstruct_tree_edges_l621_621320

theorem reconstruct_tree_edges (n : ℕ) (b : ℕ → ℕ) (T : List (ℕ × ℕ)) :
  n = 6 →
  (∀ i : ℕ, i ∈ [1, 2, 3, 4] → b i = 1) →
  T ~ [(2, 1), (3, 1), (4, 1), (5, 1), (1, 6)] :=
by
  intros h₁ h₂
  sorry

end reconstruct_tree_edges_l621_621320


namespace knights_on_island_l621_621460

-- Definitions based on conditions
inductive Inhabitant : Type
| knight : Inhabitant
| knave : Inhabitant

open Inhabitant

def statement_1 (inhabitant : Inhabitant) : Prop :=
inhabitant = knight

def statement_2 (inhabitant1 inhabitant2 : Inhabitant) : Prop :=
inhabitant1 = knight ∧ inhabitant2 = knight

def statement_3 (inhabitant1 inhabitant2 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave)) / 2 ≥ 0.5

def statement_4 (inhabitant1 inhabitant2 inhabitant3 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave) + ↑(inhabitant3 = knave)) / 3 ≥ 0.65

def statement_5 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 : Inhabitant) : Prop :=
(↑(inhabitant1 = knight) + ↑(inhabitant2 = knight) + ↑(inhabitant3 = knight) + ↑(inhabitant4 = knight)) / 4 ≥ 0.5

def statement_6 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 inhabitant5 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave) + ↑(inhabitant3 = knave) + ↑(inhabitant4 = knave) + ↑(inhabitant5 = knave)) / 5 ≥ 0.4

def statement_7 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 inhabitant5 inhabitant6 : Inhabitant) : Prop :=
(↑(inhabitant1 = knight) + ↑(inhabitant2 = knight) + ↑(inhabitant3 = knight) + ↑(inhabitant4 = knight) + ↑(inhabitant5 = knight) + ↑(inhabitant6 = knight)) / 6 ≥ 0.65

-- Lean Statement
theorem knights_on_island (inhabitants : Fin 7 → Inhabitant) :
  (∀ i, (inhabitants i = knight ↔ (i = 0) ∨ (i = 1) ∨ (i = 4) ∨ (i = 5) ∨ (i = 6))) → 5 :=
by
  sorry

end knights_on_island_l621_621460


namespace inequality_proof_l621_621480

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x / (y + z)) + (y / (z + x)) + (z / (x + y)) ≥ (3 / 2) :=
sorry

end inequality_proof_l621_621480


namespace P_n_real_roots_P_2018_real_roots_l621_621622

noncomputable def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 1) := λ x, x * P n x - P (n - 1) x

theorem P_n_real_roots (n: ℕ) : ∃ r: ℕ, r ≡ n := sorry

theorem P_2018_real_roots : ∃ r: ℕ, r = 2018 := P_n_real_roots 2018

end P_n_real_roots_P_2018_real_roots_l621_621622


namespace contrapositive_correct_l621_621551

-- Conditions and the proposition
def prop1 (a : ℝ) : Prop := a = -1 → a^2 = 1

-- The contrapositive of the proposition
def contrapositive (a : ℝ) : Prop := a^2 ≠ 1 → a ≠ -1

-- The proof problem statement
theorem contrapositive_correct (a : ℝ) : prop1 a ↔ contrapositive a :=
by sorry

end contrapositive_correct_l621_621551


namespace actual_avg_height_proof_l621_621126

noncomputable def actual_average_height : ℚ :=
  let avg_height := 185
  let n_boys := 35
  let incorrect_height := 166
  let actual_height := 106
  let incorrect_total_height := avg_height * n_boys
  let difference := incorrect_height - actual_height
  let correct_total_height := incorrect_total_height - difference
  let actual_avg_height := correct_total_height / n_boys
  (actual_avg_height).round * 100 / 100

theorem actual_avg_height_proof : actual_average_height = 183.29 :=
by {
  sorry
}

end actual_avg_height_proof_l621_621126


namespace P_2018_roots_l621_621636

def P : ℕ → (ℝ → ℝ)
| 0     := λ x, 1
| 1     := λ x, x
| (n+2) := λ x, x * (P (n+1) x) - (P n x)

theorem P_2018_roots : 
  ∃ S : Finset ℝ, S.card = 2018 ∧ ∀ x ∈ S, P 2018 x = 0 ∧ 
  ∀ x₁ x₂ ∈ S, x₁ ≠ x₂ → x₁ ≠ x₂ :=
begin
  sorry
end

end P_2018_roots_l621_621636


namespace octagon_area_l621_621338

noncomputable section

variables {BDEF : Type} [has_measure BDEF] (AB BC : ℝ) (area_octagon : ℝ)

-- Condition: BDEF is a square
axiom square_BDEF : is_square BDEF

-- Conditions: AB = 2, BC = 2
axiom AB_eq_2 : AB = 2
axiom BC_eq_2 : BC = 2

-- Theorem: The area of the regular octagon is 16 + 16 * sqrt 2 square units
theorem octagon_area 
    (h_square_BDEF : square_BDEF)
    (h_AB_eq_2 : AB_eq_2)
    (h_BC_eq_2 : BC_eq_2) :
  area_octagon = 16 + 16 * Real.sqrt 2 :=
sorry

end octagon_area_l621_621338


namespace range_of_f_l621_621358

noncomputable def f (x : ℝ) := Real.log (2 - x^2) / Real.log (1 / 2)

theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 0 := by
  sorry

end range_of_f_l621_621358


namespace stamps_on_last_page_l621_621413

theorem stamps_on_last_page (total_books : ℕ) (pages_per_book : ℕ) (stamps_per_page_initial : ℕ) (stamps_per_page_new : ℕ)
    (full_books_new : ℕ) (pages_filled_seventh_book : ℕ) (total_stamps : ℕ) (stamps_in_seventh_book : ℕ) 
    (remaining_stamps : ℕ) :
    total_books = 10 →
    pages_per_book = 50 →
    stamps_per_page_initial = 8 →
    stamps_per_page_new = 12 →
    full_books_new = 6 →
    pages_filled_seventh_book = 37 →
    total_stamps = total_books * pages_per_book * stamps_per_page_initial →
    stamps_in_seventh_book = 4000 - (600 * full_books_new) →
    remaining_stamps = stamps_in_seventh_book - (pages_filled_seventh_book * stamps_per_page_new) →
    remaining_stamps = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end stamps_on_last_page_l621_621413


namespace angle_inclination_AB_l621_621346

noncomputable def slope (A B : (ℝ × ℝ)) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

noncomputable def angle_of_inclination (A B : (ℝ × ℝ)) : ℝ :=
  real.arctan (slope A B) * 180 / real.pi

theorem angle_inclination_AB :
  ∀ (A B : ℝ × ℝ), A = (-2, 0) → B = (-5, 3) → angle_of_inclination A B = 135 := 
by
  intros A B hA hB
  rw [hA, hB]
  sorry

end angle_inclination_AB_l621_621346


namespace min_abs_phase_shift_l621_621877

theorem min_abs_phase_shift (ϕ : ℝ) (h : ∀ x, 2 * sin (2 * x + ϕ) = 2 * sin (2 * (2 * π / 3 - x) + ϕ)) :
  |ϕ| = π / 3 :=
sorry

end min_abs_phase_shift_l621_621877


namespace product_of_two_large_integers_l621_621537

theorem product_of_two_large_integers :
  ∃ a b : ℕ, a > 2009^182 ∧ b > 2009^182 ∧ 3^2008 + 4^2009 = a * b :=
by { sorry }

end product_of_two_large_integers_l621_621537


namespace find_number_l621_621761

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l621_621761


namespace constant_term_in_binomial_expansion_l621_621965

theorem constant_term_in_binomial_expansion : 
  ∃ (r : ℕ), (\sum k in range (11), (binom 8 k) * ((sqrt x) ^ (8 - k)) * ((1 / (2 * sqrt x)) ^ k) = ∑ k in range (11), if k = 4 then 35/8 else 0) :=
sorry

end constant_term_in_binomial_expansion_l621_621965


namespace arithmetic_sequence_common_difference_l621_621832

theorem arithmetic_sequence_common_difference 
  (a1 a2 a3 a4 d : ℕ)
  (S : ℕ → ℕ)
  (h1 : S 2 = a1 + a2)
  (h2 : S 4 = a1 + a2 + a3 + a4)
  (h3 : S 2 = 4)
  (h4 : S 4 = 20)
  (h5 : a2 = a1 + d)
  (h6 : a3 = a2 + d)
  (h7 : a4 = a3 + d) :
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l621_621832


namespace average_waiting_time_l621_621697

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l621_621697


namespace songs_today_is_14_l621_621907

-- Define the number of songs Jeremy listened to yesterday
def songs_yesterday (x : ℕ) : ℕ := x

-- Define the number of songs Jeremy listened to today
def songs_today (x : ℕ) : ℕ := x + 5

-- Given conditions
def total_songs (x : ℕ) : Prop := songs_yesterday x + songs_today x = 23

-- Prove the number of songs Jeremy listened to today
theorem songs_today_is_14 : ∃ x: ℕ, total_songs x ∧ songs_today x = 14 :=
by {
  sorry
}

end songs_today_is_14_l621_621907


namespace solve_abs_eqn_l621_621540

theorem solve_abs_eqn (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) :=
by
  sorry

end solve_abs_eqn_l621_621540


namespace old_clock_slower_l621_621690

-- Given conditions
def old_clock_coincidence_minutes : ℕ := 66

-- Standard clock coincidences in 24 hours
def standard_clock_coincidences_in_24_hours : ℕ := 22

-- Standard 24 hours in minutes
def standard_24_hours_in_minutes : ℕ := 24 * 60

-- Total time for old clock in minutes over what should be 24 hours
def total_time_for_old_clock : ℕ := standard_clock_coincidences_in_24_hours * old_clock_coincidence_minutes

-- Problem statement: prove that the old clock's 24 hours is 12 minutes slower 
theorem old_clock_slower : total_time_for_old_clock = standard_24_hours_in_minutes + 12 := by
  sorry

end old_clock_slower_l621_621690


namespace sarah_desserts_l621_621086

/-- Michael saved 5 of his cookies to give Sarah,
    and Sarah saved a third of her 9 cupcakes to give to Michael.
    Prove that Sarah ends up with 11 desserts. -/
theorem sarah_desserts : 
  let initial_cupcakes := 9 in
  let cupcakes_given := initial_cupcakes / 3 in
  let remaining_cupcakes := initial_cupcakes - cupcakes_given in
  let cookies_received := 5 in
  let total_desserts := remaining_cupcakes + cookies_received in
  total_desserts = 11 :=
by 
  let initial_cupcakes := 9 in
  let cupcakes_given := initial_cupcakes / 3 in
  let remaining_cupcakes := initial_cupcakes - cupcakes_given in
  let cookies_received := 5 in
  let total_desserts := remaining_cupcakes + cookies_received in
  calc
    total_desserts = remaining_cupcakes + cookies_received : rfl
    ... = 6 + 5 : by
      { calc remaining_cupcakes = 9 - 3 : rfl
        ... = 6 : by norm_num,
        calc cookies_received = 5 : rfl,
      calc total_desserts = 6 + 5 : rfl,
      }
    ... = 11 : by norm_num

end sarah_desserts_l621_621086


namespace students_transferred_l621_621387

theorem students_transferred (students_before : ℕ) (total_students : ℕ) (students_equal : ℕ) 
  (h1 : students_before = 23) (h2 : total_students = 50) (h3 : students_equal = total_students / 2) : 
  (∃ x : ℕ, students_equal = students_before + x) → (∃ x : ℕ, x = 2) :=
by
  -- h1: students_before = 23
  -- h2: total_students = 50
  -- h3: students_equal = total_students / 2
  -- to prove: ∃ x : ℕ, students_equal = students_before + x → ∃ x : ℕ, x = 2
  sorry

end students_transferred_l621_621387


namespace train_speed_is_correct_l621_621670

def train_length : ℝ := 300
def bridge_length : ℝ := 115
def time_to_pass_bridge : ℝ := 42.68571428571429
def total_distance : ℝ := train_length + bridge_length
def speed_m_per_s : ℝ := total_distance / time_to_pass_bridge
def km_per_hour_conv_factor : ℝ := 3.6
def speed_km_per_h : ℝ := speed_m_per_s * km_per_hour_conv_factor

theorem train_speed_is_correct :
  speed_km_per_h = 35.01 := by
  sorry

end train_speed_is_correct_l621_621670


namespace AD_passes_through_incenter_l621_621023

variables {A B C D : Type} [inner_product_space ℝ D]
variables {triangle_ABC : triangle A B C}

variables {AB AC AD : D}
variables (h1 : ∥AB∥ = 3)
variables (h2 : ∥AC∥ = 2)
variables (h3 : AD = (1/2 : ℝ) • AB + (3/4 : ℝ) • AC)

theorem AD_passes_through_incenter
  (h_triangle : is_triangle triangle_ABC)
  (h_AB : (A - B) = AB)
  (h_AC : (A - C) = AC)
  (h_AD : (A - D) = AD) :
  passes_through_incenter AD h_triangle := 
sorry

end AD_passes_through_incenter_l621_621023


namespace lucy_popsicles_l621_621444

def lucy_budget : ℤ := 1500
def popsicle_cost : ℤ := 240

theorem lucy_popsicles : ∃ (n : ℤ), n * popsicle_cost ≤ lucy_budget ∧ (n + 1) * popsicle_cost > lucy_budget := by
  have H : lucy_budget / popsicle_cost = 6.25 := sorry
  have H_int : (lucy_budget / popsicle_cost).to_int = 6 := sorry
  use H_int
  split
  · rw H_int
    exact sorry
  · exact sorry
  sorry

end lucy_popsicles_l621_621444


namespace average_waiting_time_l621_621694

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l621_621694


namespace measure_of_segment_PB_l621_621390

theorem measure_of_segment_PB 
  (M_is_midpoint_AC : ∀ (A C M : Point), midpoint M A C)
  (MP_perp_AB_at_P : ∀ (A B M P : Point), perpendicular M P A B)
  (measure_AC_x : ∀ (A C : Point), measure (segment A C) = x)
  (measure_AP_x_plus_3 : ∀ (A P : Point), measure (segment A P) = x + 3)
  : ∀ (A B P : Point), measure (segment P B) = x - 3 :=
by
  sorry

end measure_of_segment_PB_l621_621390


namespace number_of_perfect_cubes_l621_621860

theorem number_of_perfect_cubes (n : ℤ) : 
  (∃ (count : ℤ), (∀ (x : ℤ), (100 < x^3 ∧ x^3 < 400) ↔ x = 5 ∨ x = 6 ∨ x = 7) ∧ (count = 3)) := 
sorry

end number_of_perfect_cubes_l621_621860


namespace no_distinct_natural_numbers_exist_l621_621743

theorem no_distinct_natural_numbers_exist 
  (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬ (a + 1 / a = (1 / 2) * (b + 1 / b + c + 1 / c)) :=
sorry

end no_distinct_natural_numbers_exist_l621_621743


namespace minimum_participants_l621_621534

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l621_621534


namespace binomial_variance_is_one_l621_621833

noncomputable def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem binomial_variance_is_one :
  binomial_variance 4 (1 / 2) = 1 := by
  sorry

end binomial_variance_is_one_l621_621833


namespace smallest_solution_x_abs_x_eq_3x_plus_2_l621_621790

theorem smallest_solution_x_abs_x_eq_3x_plus_2 : ∃ x : ℝ, (x * abs x = 3 * x + 2) ∧ (∀ y : ℝ, (y * abs y = 3 * y + 2) → x ≤ y) ∧ x = -2 :=
by
  sorry

end smallest_solution_x_abs_x_eq_3x_plus_2_l621_621790


namespace constant_sum_of_dist_squares_l621_621437

-- Definitions for points, lines, and circle
structure Point :=
  (x y : ℝ)
 
structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Line :=
  (p1 p2 : Point)

variables (O : Circle) (P : Point) (n : ℕ) (lines : fin n → Line)
  (M M' : fin n → Point)
  (R : ℝ) (hR : O.radius = R) (hn : n ≥ 2)
  (h_inside: ∀ p ∈ lines, p ∈ O)
  (h_angle : ∀ i (i < n), (∠ (lines ⟨i, _⟩).(p1) (lines ⟨i, _⟩).(p2) = π / n)

noncomputable def sum_of_squares_P (P : Point) : ℝ :=
  ∑ i : fin n, (dist P (M i))^2 + (dist P (M' i))^2

theorem constant_sum_of_dist_squares (O : Circle) (P : Point) (n : ℕ) (lines : fin n → Line)
  (M M' : fin n → Point) (R : ℝ) (hR : O.radius = R) (hn : n ≥ 2)
  (h_inside : ∀ p ∈ lines, p ∈ O) 
  (h_angle : ∀ i (i < n), (∠ (lines ⟨i, _⟩).p1 (lines ⟨i, _⟩).p2 = π / n)) :
  sum_of_squares_P O P lines M M' = 2 * n * R^2 :=
sorry

end constant_sum_of_dist_squares_l621_621437


namespace track_and_field_and_ball_games_l621_621750

-- Define the variables and conditions
variables (A B C : ℕ)
variables (A_and_B A_and_C B_and_C : ℕ)
variable (total : ℕ)

-- Hypothesize the given conditions
hypothesis h_total : total = 28
hypothesis h_A : A = 15
hypothesis h_B : B = 8
hypothesis h_C : C = 14
hypothesis h_A_and_B : A_and_B = 3
hypothesis h_A_and_C : A_and_C = 3
hypothesis h_everyone_not_in_all_three : B_and_C + 3 + 3 + 3 = 8

-- Define the inclusion-exclusion principle for the given sets
def inclusion_exclusion (A B C A_and_B A_and_C B_and_C : ℕ) : ℕ :=
  A + B + C - A_and_B - A_and_C - B_and_C

theorem track_and_field_and_ball_games :
  B_and_C = 6 := by 
  calc
    total = A + B + C - A_and_B - A_and_C - B_and_C : by sorry
    28 = 15 + 8 + 14 - 3 - 3 - B_and_C : by rw [h_total, h_A, h_B, h_C, h_A_and_B, h_A_and_C]
    28 = 34 - B_and_C : by sorry
    B_and_C = 34 - 28 : by sorry
    B_and_C = 6 : by sorry

end track_and_field_and_ball_games_l621_621750


namespace phoenix_flight_l621_621128

theorem phoenix_flight : ∃ n : ℕ, 3 ^ n > 6560 ∧ ∀ m < n, 3 ^ m ≤ 6560 :=
by sorry

end phoenix_flight_l621_621128


namespace uralan_matches_total_matches_l621_621403

theorem uralan_matches (teams : ℕ) (matches_per_pair : ℕ) (other_teams : ℕ) :
  teams = 16 → matches_per_pair = 2 → other_teams = 15 → "Uralan".matches = matches_per_pair * other_teams :=
by
  intros teams_eq mpp_eq ot_eq
  rw [teams_eq, mpp_eq, ot_eq]
  exact eq.refl (2 * 15)

theorem total_matches (teams : ℕ) (matches_per_pair : ℕ) (matches : ℕ) :
  teams = 16 → matches_per_pair = 2 → matches = (teams * (teams - 1) * matches_per_pair) / 2 :=
by
  intros teams_eq mpp_eq
  rw [teams_eq, mpp_eq]
  exact eq.refl ((16 * 15 * 2) / 2)

end uralan_matches_total_matches_l621_621403


namespace P_2018_has_2018_distinct_real_roots_l621_621626
noncomputable theory

def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 2) := λ x, x * P (n + 1) x - P n x

theorem P_2018_has_2018_distinct_real_roots :
  (∃ xs : fin 2018 → ℝ, ∀ i j : fin 2018, i ≠ j → xs i ≠ xs j ∧ ∀ x : ℝ, P 2018 x = 0 ↔ ∃ k : fin 2018, x = xs k) :=
sorry

end P_2018_has_2018_distinct_real_roots_l621_621626


namespace xiaoqiang_xiaolin_stamps_l621_621605

-- Definitions for initial conditions and constraints
noncomputable def x : ℤ := 227
noncomputable def y : ℤ := 221
noncomputable def k : ℤ := sorry

-- Proof problem as a theorem
theorem xiaoqiang_xiaolin_stamps:
  x + y > 400 ∧
  x - k = (13 / 19) * (y + k) ∧
  y - k = (11 / 17) * (x + k) ∧
  x = 227 ∧ 
  y = 221 :=
by
  sorry

end xiaoqiang_xiaolin_stamps_l621_621605


namespace race_participants_minimum_l621_621529

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l621_621529


namespace equidistant_divisors_multiple_of_6_l621_621739

open Nat

theorem equidistant_divisors_multiple_of_6 (n : ℕ) :
  (∃ a b : ℕ, a ≠ b ∧ a ∣ n ∧ b ∣ n ∧ 
    (a + b = 2 * (n / 3))) → 
  (∃ k : ℕ, n = 6 * k) := 
by
  sorry

end equidistant_divisors_multiple_of_6_l621_621739


namespace probability_f_ge_zero_l621_621325

noncomputable def f (x : ℝ) : ℝ :=
  3 * Real.sin (2 * x - (Real.pi / 6))

def intervalStart : ℝ := -Real.pi / 4
def intervalEnd : ℝ := 2 * Real.pi / 3
def probabilityIntervalStart : ℝ := Real.pi / 12
def probabilityIntervalEnd : ℝ := 7 * Real.pi / 12
def probabilitySolution : ℝ := 6 / 11

theorem probability_f_ge_zero :
  let x : ℝ := if x >= intervalStart ∧ x <= intervalEnd then x else 0
  in ∀ x : ℝ, f x >= 0 → 
  (probabilityIntervalEnd - probabilityIntervalStart) / (intervalEnd - intervalStart) = probabilitySolution :=
by
  sorry

end probability_f_ge_zero_l621_621325


namespace divisors_of_64n4_l621_621310

theorem divisors_of_64n4 (n : ℕ) 
  (h_pos : 0 < n)
  (h_divisors_90n3 : (90 * n^3).numDivisors = 90) :
  (64 * n^4).numDivisors = 275 := 
sorry

end divisors_of_64n4_l621_621310


namespace P_2018_has_2018_distinct_real_roots_l621_621627
noncomputable theory

def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 2) := λ x, x * P (n + 1) x - P n x

theorem P_2018_has_2018_distinct_real_roots :
  (∃ xs : fin 2018 → ℝ, ∀ i j : fin 2018, i ≠ j → xs i ≠ xs j ∧ ∀ x : ℝ, P 2018 x = 0 ↔ ∃ k : fin 2018, x = xs k) :=
sorry

end P_2018_has_2018_distinct_real_roots_l621_621627


namespace eval_expression_l621_621294

theorem eval_expression :
  64^(-1/6 : ℝ) + 81^(-1/4 : ℝ) = 5/6 := 
by
  sorry

end eval_expression_l621_621294


namespace ab_parallel_cd_l621_621398

variables {Point : Type} [affine_space Point]
variables (A B C D R T P S : Point)

-- Define the quadrilateral and conditions
variables (hR : R ∈ line[BC]) (hT : T ∈ line[AD])
variables (hP : ∃ B T AR P, (B ∉ line[AR]) ∧ (T ∉ line[AR]) ∧ P = (line[BT] ∩ line[AR]))
variables (hS : ∃ S CT DR, P ∈ line[CT] ∧ S = (line[CT] ∩ line[DR]) ∧ S = (line[CT] ∩ line[DR]))

-- Define the parallelogram PRST
variables (h_parallelogram : PRST)

-- Prove that AB is parallel to CD
theorem ab_parallel_cd : AB ∥ CD :=
by
  sorry

end ab_parallel_cd_l621_621398


namespace product_of_possible_sums_l621_621019

/-- 
Each vertex of a cube is labeled with either \( +1 \) or \( -1 \).
Each face of the cube has a value equal to the product of the values at its vertices.
Sum the values of the 8 vertices and the 6 faces.
Determine all possible values for this sum.
Find the product of these possible sums.
-/
theorem product_of_possible_sums : 
  let possible_sums := [14, 10, 6, 2, -2, -6, -10]
  in possible_sums.prod = -20160 := 
by sorry

end product_of_possible_sums_l621_621019


namespace race_minimum_participants_l621_621518

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l621_621518


namespace array_sum_remainder_l621_621643

def entry_value (r c : ℕ) : ℚ :=
  (1 / (2 * 1013) ^ r) * (1 / 1013 ^ c)

def array_sum : ℚ :=
  (1 / (2 * 1013 - 1)) * (1 / (1013 - 1))

def m : ℤ := 1
def n : ℤ := 2046300
def mn_sum : ℤ := m + n

theorem array_sum_remainder :
  (mn_sum % 1013) = 442 :=
by
  sorry

end array_sum_remainder_l621_621643


namespace solve_system_of_equations_l621_621117

theorem solve_system_of_equations :
  ∃ (x y z : ℤ), 
    x = -2 ∧ 
    y = 3 ∧ 
    z = 1 ∧
    (x^2 - 3*y + z = -4) ∧
    (x - 3*y + z^2 = -10) ∧
    (3*x + y^2 - 3*z = 0) :=
by {
  use [-2, 3, 1],
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  norm_num,
  split,
  norm_num,
  norm_num,
}

end solve_system_of_equations_l621_621117


namespace coprime_b_and_σb_l621_621915

def σ(n : ℕ) : ℕ :=
  ∑ d in (nat.divisors n), d

theorem coprime_b_and_σb
  (r b : ℕ)
  (h_pos_r : r > 0)
  (h_pos_b : b > 0)
  (h_odd_b : b % 2 = 1)
  (h_σN : σ (2 ^ r * b) = 2 * 2 ^ r * b - 1) :
  Nat.coprime b (σ b) :=
by
  sorry

end coprime_b_and_σb_l621_621915


namespace chromatic_number_T_is_3_l621_621066

-- Define the graph T with the specified vertices and edges
structure Graph (V : Type) :=
  (edges : set (V × V))
  (symm : ∀ {x y : V}, (x, y) ∈ edges → (y, x) ∈ edges)

noncomputable def T : Graph ℤ :=
{ edges := { (a, b) | ∃ (k : ℕ), |a - b| = 2^k },
  symm := by sorry }

-- Statement: chromatic number of T is 3
theorem chromatic_number_T_is_3 : chromatic_number T = 3 :=
by
  sorry

end chromatic_number_T_is_3_l621_621066


namespace benny_days_worked_l621_621716

/-- Benny works 3 hours a day and in total he worked for 18 hours. 
We need to prove that he worked for 6 days. -/
theorem benny_days_worked (hours_per_day : ℕ) (total_hours : ℕ)
  (h1 : hours_per_day = 3)
  (h2 : total_hours = 18) :
  total_hours / hours_per_day = 6 := 
by sorry

end benny_days_worked_l621_621716


namespace KnightsCount_l621_621456

def isKnight (n : ℕ) : Prop := sorry -- Define isKnight
def tellsTruth (n : ℕ) : Prop := sorry -- Define tellsTruth

-- Statements by the inhabitants
axiom H1 : isKnight 0 ↔ tellsTruth 0
axiom H2 : tellsTruth 1 ↔ isKnight 0
axiom H3 : tellsTruth 2 ↔ (¬(isKnight 0) ∨ ¬(isKnight 1))
axiom H4 : tellsTruth 3 ↔ (¬(isKnight 0) ∨ ¬(isKnight 1) ∨ ¬(isKnight 2))
axiom H5 : tellsTruth 4 ↔ (isKnight 0 ∧ isKnight 1 ∧ (isKnight 2 ∨ isKnight 3))
axiom H6 : tellsTruth 5 ↔ ((¬isKnight 0 ∨ ¬isKnight 1 ∨ isKnight 2) ∧ (¬isKnight 3 ∨ isKnight 4))
axiom H7 : tellsTruth 6 ↔ (isKnight 0 ∧ isKnight 1 ∧ isKnight 2 ∧ isKnight 3 ∧ ¬(¬isKnight 4 ∧ ¬isKnight 5))

theorem KnightsCount : ∃ k1 k2 k3 k4 k5 k6 k7 : Prop,
  (isKnight 0 = k1 ∧ isKnight 1 = k2 ∧ isKnight 2 = k3 ∧ isKnight 3 = k4 ∧ isKnight 4 = k5 ∧ isKnight 5 = k6 ∧ isKnight 6 = k7) ∧ 
  tellsTruth 0 ∧ tellsTruth 1 ∧ tellsTruth 2 ∧ tellsTruth 3 ∧ tellsTruth 4 ∧ tellsTruth 5 ∧ tellsTruth 6 ∧
  (5 = 1 + if k1 then 1 else 0 + if k2 then 1 else 0 + if k3 then 1 else 0 + if k4 then 1 else 0 + if k5 then 1 else 0 + if k6 then 1 else 0 + if k7 then 1 else 0)
:=
by
  sorry

end KnightsCount_l621_621456


namespace constant_sum_of_arith_seq_l621_621921

def arith_seq_sum_const (a : ℕ → ℤ) (S : ℕ → ℤ) (k : ℕ) (n : ℕ) : Prop :=
  (∀ m, S m = ∑ i in range m, a i) ∧
  (a k + a (k + 1) = 0) →

  S (2 * k) = S n  

theorem constant_sum_of_arith_seq
  (a : ℕ → ℤ) (S : ℕ → ℤ) (k n : ℕ) :
  (∀ m, S m = ∑ i in range m, a i) ∧
  (a k + a (k + 1) = 0) →
  n % (2 * k) = 0 →
  S (2 * k) = S n :=
sorry

end constant_sum_of_arith_seq_l621_621921


namespace alpha_in_third_quadrant_l621_621866

-- Define a function to determine the quadrant given the conditions
def quadrant (α : ℝ) : ℕ :=
  if (Real.sin α < 0) ∧ (Real.tan α > 0) then 3 else 0 -- only for the specific conditions provided

theorem alpha_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : quadrant α = 3 :=
  by sorry

end alpha_in_third_quadrant_l621_621866


namespace rectangle_area_90_l621_621125

theorem rectangle_area_90 {x y : ℝ} (h1 : (x + 3) * (y - 1) = x * y) (h2 : (x - 3) * (y + 1.5) = x * y) : x * y = 90 := 
  sorry

end rectangle_area_90_l621_621125


namespace p_sufficient_not_necessary_for_q_l621_621319

variable (x : ℝ)

def p : Prop := 1 < x ∧ x < 2
def q : Prop := log x < 1

theorem p_sufficient_not_necessary_for_q : (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l621_621319


namespace min_value_nS_n_l621_621331

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
variable {n m : ℕ}
variable (m >= 2) (S m-1 = -2) (S m = 0) (S m+1 = 3)

theorem min_value_nS_n : ∃ n : ℕ, (nS n) = -9 := by
    sorry

end min_value_nS_n_l621_621331


namespace general_formula_a_n_sum_first_n_terms_T_n_l621_621819

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Condition: S_n = 2a_n - 3
axiom condition_S (n : ℕ) : S_n n = 2 * (a_n n) - 3

-- (I) General formula for a_n
theorem general_formula_a_n (n : ℕ) : a_n n = 3 * 2^(n - 1) := 
sorry

-- (II) General formula for T_n
theorem sum_first_n_terms_T_n (n : ℕ) : T_n n = 3 * (n - 1) * 2^n + 3 := 
sorry

end general_formula_a_n_sum_first_n_terms_T_n_l621_621819


namespace ratio_of_adjacent_sides_of_parallelogram_l621_621547

theorem ratio_of_adjacent_sides_of_parallelogram
  (A B C D E F G H : Type)
  (α β : ℝ)
  (a b : ℝ) (a_gt_b : a > b)
  (area_parallelogram quadrilateral_area: ℝ)
  (area_relation: quadrilateral_area = (1 / 3) * area_parallelogram)
  (parallelogram_condition: ∀ x, x = (b * a * (sin (2 * α))))
  (quadrilateral_condition: ∀ y, y = ((a - b) * (a - b) * (sin (2 * α)))):
  a / b = (4 + real.sqrt 13) / 3 :=
sorry

end ratio_of_adjacent_sides_of_parallelogram_l621_621547


namespace john_feeds_horses_l621_621034

noncomputable def number_of_feedings_per_day (H : ℕ) (F : ℕ) (B : ℕ) (D : ℕ) : ℕ :=
  (B * 1000) / D / F

theorem john_feeds_horses (H : ℕ) (F : ℕ) (B : ℕ) (D : ℕ) (feedings_per_day : ℕ) :
  H = 25 → F = 20 → B = 60 → D = 60 → feedings_per_day = 2 :=
by
  intros h H_eq h_F_eq h_B_eq h_D_eq
  have h: feedings_per_day = number_of_feedings_per_day H F B D := by
    sorry -- The actual arithmetic computation to validate with the provided data.
  exact h

end john_feeds_horses_l621_621034


namespace determine_m_l621_621427

def f (x : ℝ) : ℝ := (x^3)/3 - (3*x^2)/2 + 4*x
def g (x : ℝ) (m : ℝ) : ℝ := 2*x + m
def h (x : ℝ) (m : ℝ) : ℝ := (x^2) - 5*x + 4 - m

theorem determine_m :
  (∀ h : ℝ → ℝ → ℝ, (h 0 m ≥ 0) ∧ (h 3 m ≥ 0) ∧ (h (5/2) m < 0)) →
  (h = λ x m, (x^2) - 5*x + 4 - m) →
  (m > - 9/2) ∧ (m ≤ -2) :=
by
  intros h_def h_eq
  sorry

end determine_m_l621_621427


namespace length_of_PR_circle_l621_621101

noncomputable def length_of_PR (r : ℝ) (PQ : ℝ) (PR : ℝ) : Prop :=
  (PQ = 8) ∧ (r = 7) ∧ (PR = √(98 - 14 * √33))

theorem length_of_PR_circle :
  ∃ PR : ℝ, length_of_PR 7 8 PR :=
begin
  refine ⟨√(98 - 14 * √33), _⟩,
  unfold length_of_PR,
  simp,
  sorry,  -- No proof required
end

end length_of_PR_circle_l621_621101


namespace minimum_participants_l621_621535

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l621_621535


namespace simplify_expression_l621_621950

theorem simplify_expression (a b : ℝ) (h : b ≠ 0 ∧ a^2 ≠ b^2) :
  (let x := (a^2 - b^2) / (2 * b) in
  (a / x + real.sqrt (a^2 / x^2 + 1) = (a + b) / (a - b) ∨
   a / x + real.sqrt (a^2 / x^2 + 1) = (b - a) / (a + b))) :=
begin
  sorry
end

end simplify_expression_l621_621950


namespace probability_odd_product_lt_one_eighth_l621_621586

theorem probability_odd_product_lt_one_eighth :
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  p < 1 / 8 :=
by
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  sorry

end probability_odd_product_lt_one_eighth_l621_621586


namespace homework_problems_left_l621_621100

def math_problems : ℕ := 43
def science_problems : ℕ := 12
def finished_problems : ℕ := 44

theorem homework_problems_left :
  (math_problems + science_problems - finished_problems) = 11 :=
by
  sorry

end homework_problems_left_l621_621100


namespace mental_math_quiz_l621_621576

theorem mental_math_quiz : ∃ (q_i q_c : ℕ), q_c + q_i = 100 ∧ 10 * q_c - 5 * q_i = 850 ∧ q_i = 10 :=
by
  sorry

end mental_math_quiz_l621_621576


namespace max_dinners_with_new_neighbors_l621_621112

theorem max_dinners_with_new_neighbors
  (n : ℕ)
  (h1 : n = 17 ∨ n = 18)
  (h2 : ∀ k, k ≤ n - 1) :
  ∃ k : ℕ, (∀ i : ℕ, i < k → ∀ j : ℕ, j < n → 
  neighbor_condition_met i j (n))
  ∧ k = 8 :=
sorry

end max_dinners_with_new_neighbors_l621_621112


namespace cubic_meter_to_cubic_centimeters_and_total_volume_l621_621368

theorem cubic_meter_to_cubic_centimeters_and_total_volume :
  (1 : ℝ) ^ 3 * (100 : ℝ) ^ 3 + 500 = 1_000_500 := by
sorry

end cubic_meter_to_cubic_centimeters_and_total_volume_l621_621368


namespace trigonometric_expression_l621_621341

variable {α : ℝ} {n : ℤ}

theorem trigonometric_expression (h1 : cos (π + α) = -1/2) : 
  (sin (α + (2 * n + 1) * π) + sin (π + α)) / (sin (π - α) * cos (α + 2 * n * π)) = -4 := 
  sorry

end trigonometric_expression_l621_621341


namespace min_number_of_participants_l621_621502

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l621_621502


namespace math_problem_l621_621857

noncomputable def a : ℝ := 0.137
noncomputable def b : ℝ := 0.098
noncomputable def c : ℝ := 0.123
noncomputable def d : ℝ := 0.086

theorem math_problem : 
  ( ((a + b)^2 - (a - b)^2) / (c * d) + (d^3 - c^3) / (a * b * (a + b)) ) = 4.6886 := 
  sorry

end math_problem_l621_621857


namespace Stuart_initial_marbles_l621_621214

variable (Betty_marbles Stuart_final increased_by: ℤ) 

-- Conditions as definitions
def Betty_has : Betty_marbles = 60 := sorry 
def Stuart_collect_increase : Stuart_final = 80 := sorry 
def percentage_given : ∃ x, x = (40 * Betty_marbles) / 100 := sorry 

-- Theorem to prove Stuart had 56 marbles initially
theorem Stuart_initial_marbles 
  (h1 : Betty_has)
  (h2 : Stuart_collect_increase)
  (h3 : percentage_given) :
  ∃ y, y = Stuart_final - 24 := 
sorry

end Stuart_initial_marbles_l621_621214


namespace count_valid_license_plates_l621_621728

theorem count_valid_license_plates :
  let alphabet := ['A', 'E', 'G', 'I', 'K', 'M', 'O', 'P', 'R', 'U', 'V'] in
  let valid_letters := alphabet.filter (λ x, x ≠ 'S' ∧ x ≠ 'T' ∧ x ≠ 'Z' ∧ x ≠ 'G' ∧ x ≠ 'M') in
  valid_letters.length = 9 →
  ∃ (plates : List String), plates.length = 3024 ∧ ∀ p ∈ plates,
    p.length = 6 ∧
    p.head = 'G' ∧
    p.reverse.head = 'M' ∧
    ('S' ∉ p.toList ∧ 'T' ∉ p.toList ∧ 'Z' ∉ p.toList) ∧
    p.toList.nodup :=
sorry

end count_valid_license_plates_l621_621728


namespace children_in_initial_positions_after_lcm_l621_621122

theorem children_in_initial_positions_after_lcm 
  (k : ℕ) 
  (k₁ k₂ k₃ k₄ k₅ k₆ k₇ k₈ k₉ k₁₀ : ℕ) 
  (perimeter : ℕ) 
  (H_perimeter : perimeter = 200) 
  (velocities : ∀ i, i ∈ [k₁, k₂, k₃, k₄, k₅, k₆, k₇, k₈, k₉, k₁₀].erase_dup → ∃ (k : ℕ), i = 200 / k) :
  ∃ (M : ℕ), M = Nat.lcm k₁ (Nat.lcm k₂ (Nat.lcm k₃ (Nat.lcm k₄ (Nat.lcm k₅ (Nat.lcm k₆ (Nat.lcm k₇ (Nat.lcm k₈ (Nat.lcm k₉ k₁₀)))))))) 
  → ∀ t, t = M → ∀ i, i ∈ [k₁, k₂, k₃, k₄, k₅, k₆, k₇, k₈, k₉, k₁₀].erase_dup → t % i = 0 := 
sorry

end children_in_initial_positions_after_lcm_l621_621122


namespace a_x1_x2_x13_eq_zero_l621_621561

theorem a_x1_x2_x13_eq_zero {a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ}
  (h1: a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) *
             (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2: a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) *
             (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := by
  sorry

end a_x1_x2_x13_eq_zero_l621_621561


namespace travel_archipelago_l621_621226

theorem travel_archipelago (n : ℕ) 
  (cost : Fin n → Fin n → ℕ) 
  (unique_cost : ∀ i j k l, i ≠ j → k ≠ l → cost i j ≠ cost k l) :
  ∃ s : List (Fin n × Fin n), s.length = n - 1 ∧ Increasing (List.map (λ p, cost p.1 p.2) s) :=
by
  sorry

end travel_archipelago_l621_621226


namespace inverse_function_correct_l621_621567

noncomputable def original_function (x : ℝ) : ℝ :=
  2^(x + 1)

noncomputable def inverse_function (y : ℝ) : ℝ :=
  log y / log 2 - 1

theorem inverse_function_correct :
  ∀ x : ℝ, x > 0 → y = original_function (inverse_function y) :=
by
  intro x hx
  have h1 : original_function (inverse_function y) = y
  sorry   -- Skipping the detailed proof steps.

end inverse_function_correct_l621_621567


namespace complex_number_solution_l621_621322

theorem complex_number_solution (z : ℂ) (h : (z + 2 * complex.I) * (z - 2 * complex.I) = 2) : z = complex.sqrt 2 * complex.I ∨ z = -complex.sqrt 2 * complex.I :=
sorry

end complex_number_solution_l621_621322


namespace P_2018_roots_l621_621634

def P : ℕ → (ℝ → ℝ)
| 0     := λ x, 1
| 1     := λ x, x
| (n+2) := λ x, x * (P (n+1) x) - (P n x)

theorem P_2018_roots : 
  ∃ S : Finset ℝ, S.card = 2018 ∧ ∀ x ∈ S, P 2018 x = 0 ∧ 
  ∀ x₁ x₂ ∈ S, x₁ ≠ x₂ → x₁ ≠ x₂ :=
begin
  sorry
end

end P_2018_roots_l621_621634


namespace arithmetic_progression_nth_term_l621_621304

theorem arithmetic_progression_nth_term (a d T_n : ℤ) (n : ℤ) :
  a = 2 → d = 8 → T_n = 90 → T_n = a + (n - 1) * d → n = 12 :=
by
  intro ha hd hTn ht
  have : 90 = 2 + (n - 1) * 8, from ht
  -- Proof steps here
  sorry

end arithmetic_progression_nth_term_l621_621304


namespace product_count_l621_621055

-- Define the conditions
def T := {d : ℕ | d > 0 ∧ d ∣ 72000}
def isProductOfDistinctElementAndSquare (n : ℕ) : Prop :=
  ∃ d ∈ T, n = d * d^2

-- Statement of the proof problem
theorem product_count :
  (∃ T : set ℕ, (∀ d : ℕ, d ∈ T ↔ d > 0 ∧ d ∣ 72000) ∧ (set.card {n | isProductOfDistinctElementAndSquare n}) = 24) :=
by
  use T
  split
  -- Definition of T as the set of all positive divisors of 72000
  { intro d,
    simp [T],
    intro H,
    exact H },
  -- Placeholder for the proof of the cardinality
  { sorry }

end product_count_l621_621055


namespace equalAreasOfQuadrilaterals_l621_621409

open Real EuclideanGeometry

-- Define the condition that the points E and F are the midpoints of AB and CD in trapezoid ABCD
def isMidPoint (A B E : Point) : Prop :=
  dist A E = dist E B

-- Define the setup: 
-- A trapezoid ABCD with midpoints E and F of bases AB and CD, with intersect O of diagonals,
-- and points M, N, P on line parallel to bases at segments OA, OE, OB respectively.
def trapezoidEquality : Prop :=
  ∃ (A B C D E F O M N P : Point),
    isTrapezoid ⟨A, B, C, D⟩ ∧
    isMidPoint A B E ∧
    isMidPoint C D F ∧
    DiagonalsIntersectAt A C B D O ∧
    LineParallelIntersectsAt A O M ∧
    LineParallelIntersectsAt B O P ∧
    LineParallelIntersectsAt E O N ∧
    AreaOfQuadrilateral A P C N = AreaOfQuadrilateral B N D M

-- The theorem to state the equality of areas of quadrilaterals APCN and BNDM
theorem equalAreasOfQuadrilaterals : trapezoidEquality :=
sorry

end equalAreasOfQuadrilaterals_l621_621409


namespace sales_on_same_days_l621_621645

-- Definitions representing the conditions
def bookstore_sales_days : List ℕ := [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
def toy_store_sales_days : List ℕ := [2, 9, 16, 23, 30]

-- Lean statement to prove the number of common sale days
theorem sales_on_same_days : (bookstore_sales_days ∩ toy_store_sales_days).length = 2 :=
by sorry

end sales_on_same_days_l621_621645


namespace triangle_area_l621_621386

noncomputable def area_of_triangle (a b c α β γ : ℝ) :=
  (1 / 2) * a * b * Real.sin γ

theorem triangle_area 
  (a b c A B C : ℝ)
  (h1 : b * Real.cos C = 3 * a * Real.cos B - c * Real.cos B)
  (h2 : (a * b * Real.cos C) / (a * b) = 2) :
  area_of_triangle a b c A B C = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_area_l621_621386


namespace trajectory_of_P_is_parabola_range_of_MN_l621_621324

-- Definitions based on conditions and required proof goals

-- Circle P passing through point F(1,0)
def Circle_P_through_F (x y : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P = (x, y) ∧ P ∈ Sphere (1, 0) 1

-- Circle P tangent to the line l: x = -1
def Circle_P_tangent_l (x y : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P = (x, y) ∧ dist P (-1, y) = 0

-- Circle F defined by the equation (x-1)^2 + y^2 = 1
def circle_F (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Prove the trajectory C is y^2 = 4x
theorem trajectory_of_P_is_parabola {x y : ℝ} 
  (h1 : Circle_P_through_F x y) 
  (h2 : Circle_P_tangent_l x y) :
  y^2 = 4 * x := 
begin
  sorry
end

-- Prove the range of |MN| given Circle P intersects with Circle F
theorem range_of_MN {x y : ℝ}
  (h1 : Circle_P_through_F x y)
  (h2 : Circle_P_tangent_l x y)
  (h3 : circle_F x y) :
  ∃ d : ℝ, sqrt 3 ≤ d ∧ d < 2 :=
begin
  sorry
end

end trajectory_of_P_is_parabola_range_of_MN_l621_621324


namespace second_player_wins_with_optimal_play_l621_621579

-- Definitions based on conditions
def cell : Type := ℕ
def digit_set : set ℕ := {1, 2}

-- 5 cells represented as a list
def cells : Type := list cell

-- Check if a list of digits is divisible by 3
def is_divisible_by_3 (digits : list ℕ) : Prop :=
  (list.sum digits) % 3 = 0

-- Theorem: second player will always win with optimal play
theorem second_player_wins_with_optimal_play :
  ∀ (a1 a2 a3 a4 a5 : ℕ),
  a1 ∈ digit_set →
  a2 ∈ digit_set →
  a3 ∈ digit_set →
  a4 ∈ digit_set →
  a5 ∈ digit_set →
  (is_divisible_by_3 [a1, a2, a3, a4, a5] → false) :=
begin
  intros a1 a2 a3 a4 a5 h1 h2 h3 h4 h5,
  sorry
end

end second_player_wins_with_optimal_play_l621_621579


namespace part_I_part_II_l621_621805

variables {a : ℝ} (f : ℝ → ℝ) (xₙ : ℕ → ℝ)

-- Conditions
def positive_a : Prop := a > 0
def function_f : Prop := ∀ x, f x = exp (a * x) * sin x
def nth_extremum_point (n : ℕ) : Prop := xₙ n = n * real.pi - atan (1 / a) ∧ ∀ m < n, f (xₙ m) ≠ 0

-- Questions and proofs
theorem part_I (h1 : positive_a a) (h2 : function_f f a) 
               (h3 : ∀ n, nth_extremum_point f xₙ n) :
∀ n, f (xₙ n) = (-1)^(n + 1) * exp (a * (n * real.pi - atan (1 / a))) * sin (atan (1 / a)) ∧
∃ r, ∀ n, f (xₙ (n + 1)) = r * f (xₙ n) :=
sorry

theorem part_II (h1 : positive_a a) (h2 : function_f f a)
                (h3 : ∀ n, nth_extremum_point f xₙ n)
                (h4 : a ≥ 1 / real.sqrt (real.exp 2 - 1)) :
∀ n, xₙ n < abs (f (xₙ n)) :=
sorry

end part_I_part_II_l621_621805


namespace overlapping_length_l621_621996

-- Definitions based on the conditions
def single_tape_length : ℕ := 217
def total_tapes : ℕ := 3
def attached_length : ℕ := 627

-- The theorem statement
theorem overlapping_length :
  (∑ _ in finset.range total_tapes, single_tape_length) - attached_length = 2 * 12 :=
by
  sorry

end overlapping_length_l621_621996


namespace one_real_solution_exists_l621_621741

theorem one_real_solution_exists : 
  ∀ x : ℝ, (2 ^ (6 * x + 3)) * (4 ^ (3 * x + 6)) = 8 ^ (-4 * x + 5) ↔ x = 0 := 
by
  sorry

end one_real_solution_exists_l621_621741


namespace problem_Correct_l621_621637

open Classical

variables {Point Line Plane : Type}
variables {α β : Plane} {l m : Line} {P : Point}

-- Definitions for perpendicular and parallel relationships
def perp (x : Line) (π : Plane) : Prop := sorry
def contained_in (x : Line) (π : Plane) : Prop := sorry
def parallel (π₁ π₂ : Plane) : Prop := sorry
def line_parallel (x y : Line) : Prop := sorry

-- Given conditions
axiom l_perp_alpha : perp l α
axiom m_in_beta : contained_in m β

-- What we need to prove
theorem problem_Correct (h₁ : line_parallel l m → perp α β) (h₂ : parallel α β → perp l m) : 
  (line_parallel l m → perp α β) ∧ (parallel α β → perp l m) := 
by
  split
  · exact h₁
  · exact h₂

end problem_Correct_l621_621637


namespace weight_difference_l621_621227

variable (W_A W_D : Nat)

theorem weight_difference : W_A - W_D = 15 :=
by
  -- Given conditions
  have h1 : W_A = 67 := sorry
  have h2 : W_D = 52 := sorry
  -- Proof
  sorry

end weight_difference_l621_621227


namespace no_regular_polygon_with_half_parallel_diagonals_l621_621745

-- Define the concept of a regular polygon with n sides
def is_regular_polygon (n : ℕ) : Prop :=
  n ≥ 3  -- A polygon has at least 3 sides

-- Define the concept of diagonals in the polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Define the concept of diagonals being parallel to the sides
def parallell_diagonals (n : ℕ) : ℕ :=
  -- This needs more formalization if specified, here's a placeholder
  sorry

-- The main theorem to prove
theorem no_regular_polygon_with_half_parallel_diagonals (n : ℕ) (h : is_regular_polygon n) :
  ¬ (∃ k : ℕ, k = num_diagonals n / 2 ∧ k = parallell_diagonals n) :=
begin
  sorry
end

end no_regular_polygon_with_half_parallel_diagonals_l621_621745


namespace symmetric_axis_stretched_shifted_sine_l621_621357

theorem symmetric_axis_stretched_shifted_sine :
  (∃ (x : ℝ), y = sin (6 * x + π / 4) → 
   y = sin (2 * (x - π / 8) + π / 4) ∧ 
   (∃ (k : ℤ), x = k * π / 2 + π / 4)) :=
sorry

end symmetric_axis_stretched_shifted_sine_l621_621357


namespace gcd_multiples_l621_621868

theorem gcd_multiples (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : Nat.gcd p q = 15) : Nat.gcd (8 * p) (18 * q) = 30 :=
by sorry

end gcd_multiples_l621_621868


namespace distance_between_B_and_C_l621_621157

theorem distance_between_B_and_C
  (A B C : Type)
  (AB : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (h_AB : AB = 10)
  (h_angle_A : angle_A = 60)
  (h_angle_B : angle_B = 75) :
  ∃ BC : ℝ, BC = 5 * Real.sqrt 6 :=
by
  sorry

end distance_between_B_and_C_l621_621157


namespace race_participants_minimum_l621_621528

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l621_621528


namespace sarah_desserts_l621_621087

/-- Michael saved 5 of his cookies to give Sarah,
    and Sarah saved a third of her 9 cupcakes to give to Michael.
    Prove that Sarah ends up with 11 desserts. -/
theorem sarah_desserts : 
  let initial_cupcakes := 9 in
  let cupcakes_given := initial_cupcakes / 3 in
  let remaining_cupcakes := initial_cupcakes - cupcakes_given in
  let cookies_received := 5 in
  let total_desserts := remaining_cupcakes + cookies_received in
  total_desserts = 11 :=
by 
  let initial_cupcakes := 9 in
  let cupcakes_given := initial_cupcakes / 3 in
  let remaining_cupcakes := initial_cupcakes - cupcakes_given in
  let cookies_received := 5 in
  let total_desserts := remaining_cupcakes + cookies_received in
  calc
    total_desserts = remaining_cupcakes + cookies_received : rfl
    ... = 6 + 5 : by
      { calc remaining_cupcakes = 9 - 3 : rfl
        ... = 6 : by norm_num,
        calc cookies_received = 5 : rfl,
      calc total_desserts = 6 + 5 : rfl,
      }
    ... = 11 : by norm_num

end sarah_desserts_l621_621087


namespace union_A_B_is_R_l621_621852

def set_A (x : ℝ) : Prop := x^2 - 2 * x > 0
def set_B (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem union_A_B_is_R : (λ x, set_A x ∨ set_B x) = (λ x, True) :=
by
  ext x
  simp only [iff_true, true_and, or_true, pi.zero_apply, pi.one_apply]
  sorry

end union_A_B_is_R_l621_621852


namespace f_l621_621381

def f (x : ℝ) : ℝ := Real.exp x + Real.sin x - Real.cos x

theorem f''_at_0 : (deriv^2 f) 0 = 2 := by
  sorry

end f_l621_621381


namespace three_digit_number_count_l621_621311

theorem three_digit_number_count :
  ∃ n : ℕ, n = 15 ∧
  (∀ a b c : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) →
    (100 * a + 10 * b + c = 37 * (a + b + c) → ∃ k : ℕ, k = n)) :=
sorry

end three_digit_number_count_l621_621311


namespace runners_meetings_on_track_l621_621590

def number_of_meetings (speed1 speed2 laps : ℕ) : ℕ := ((speed1 + speed2) * laps) / (2 * (speed2 - speed1))

theorem runners_meetings_on_track 
  (speed1 speed2 : ℕ) 
  (start_laps : ℕ)
  (speed1_spec : speed1 = 4) 
  (speed2_spec : speed2 = 10) 
  (laps_spec : start_laps = 28) : 
  number_of_meetings speed1 speed2 start_laps = 77 := 
by
  rw [speed1_spec, speed2_spec, laps_spec]
  -- Add further necessary steps or lemmas if required to reach the final proving statement
  sorry

end runners_meetings_on_track_l621_621590


namespace men_in_first_group_l621_621955

theorem men_in_first_group (M : ℕ) (H1 : M * 28 = 20 * 22.4) : M = 16 :=
sorry

end men_in_first_group_l621_621955


namespace general_term_formula_l621_621898

noncomputable def a : ℕ → ℝ
| 0       := 0 -- This condition is needed to define a(0) but won't affect our proof
| 1       := 2
| (n + 2) := a (n + 1) + Real.log (1 + 1 / (n + 1))

theorem general_term_formula (n : ℕ) (h : n ≥ 1) : a (n + 1) = Real.log (n + 1) + 2 := by
  sorry

end general_term_formula_l621_621898


namespace amount_paid_l621_621090

def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def change_received : ℕ := 11

theorem amount_paid (h_cost : ℕ := hamburger_cost) (o_cost : ℕ := onion_rings_cost) (s_cost : ℕ := smoothie_cost) (change : ℕ := change_received) :
  h_cost + o_cost + s_cost + change = 20 := by
  sorry

end amount_paid_l621_621090


namespace f_expr_for_nonneg_l621_621343

-- Define the function f piecewise as per the given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then
    Real.exp (-x) + 2 * x - 1
  else
    -Real.exp x + 2 * x + 1

-- Prove that for x > 0, f(x) = -e^x + 2x + 1 given the conditions
theorem f_expr_for_nonneg (x : ℝ) (h : x ≥ 0) : f x = -Real.exp x + 2 * x + 1 := by
  sorry

end f_expr_for_nonneg_l621_621343


namespace sue_necklace_total_beads_l621_621680

theorem sue_necklace_total_beads :
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  (purple + blue + green = 46) :=
by
  intros purple blue green h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sue_necklace_total_beads_l621_621680


namespace probability_of_log_ge_than_1_l621_621661

noncomputable def probability_log_greater_than_one : ℝ := sorry

theorem probability_of_log_ge_than_1 :
  probability_log_greater_than_one = 1 / 2 :=
sorry

end probability_of_log_ge_than_1_l621_621661


namespace solve_for_x_l621_621953

theorem solve_for_x (x : ℤ) (h : 5 * 2^x = 320) : x = 6 :=
by
  sorry

end solve_for_x_l621_621953


namespace symmetric_points_addition_l621_621007

theorem symmetric_points_addition 
  (m n : ℝ)
  (A : (ℝ × ℝ)) (B : (ℝ × ℝ))
  (hA : A = (2, m)) 
  (hB : B = (n, -1))
  (symmetry : A.1 = B.1 ∧ A.2 = -B.2) : 
  m + n = 3 :=
by
  sorry

end symmetric_points_addition_l621_621007


namespace pyramid_height_correct_l621_621210

variable (L B : ℝ)
variable (P : ℝ)
variable (height : ℝ)

axiom length_breadth_relation : L = 2 * B
axiom perimeter : 2 * (L + B) = 40
axiom distance_apex_vertex : P = 15

noncomputable def height_pyramid
  (L B : ℝ) (P : ℝ) : ℝ :=
  let FB := 0.5 * Real.sqrt (L ^ 2 + B ^ 2)
  in Real.sqrt (P ^ 2 - FB ^ 2)

theorem pyramid_height_correct 
  (L B : ℝ) (P : ℝ)
  (hLb : L = 2 * B)
  (hPer : 2 * (L + B) = 40)
  (hDist : P = 15) :
  height_pyramid L B P = 10 * Real.sqrt 19 / 3 := by
  sorry

end pyramid_height_correct_l621_621210


namespace minimal_sum_cos_product_is_14_l621_621248

theorem minimal_sum_cos_product_is_14 
  (h : ∀ x : ℝ, sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (7 * x) ^ 2 = 2) :
  ∃ a b c : ℕ, (a = 2 ∧ b = 4 ∧ c = 8 ∧ a + b + c = 14 ∧ ∀ x : ℝ, cos (a * x) * cos (b * x) * cos (c * x) = 0) := 
by 
  use [2, 4, 8]
  split; simp
  split; norm_num
  split; norm_num
  intros x
  sorry

end minimal_sum_cos_product_is_14_l621_621248


namespace coefficient_x2_term_l621_621129

open Polynomial

noncomputable def poly1 : Polynomial ℝ := (X - 1)^3
noncomputable def poly2 : Polynomial ℝ := (X - 1)^4

theorem coefficient_x2_term :
  coeff (poly1 + poly2) 2 = 3 :=
sorry

end coefficient_x2_term_l621_621129


namespace GCF_LCM_18_30_10_45_eq_90_l621_621432

-- Define LCM and GCF functions
def LCM (a b : ℕ) := a / Nat.gcd a b * b
def GCF (a b : ℕ) := Nat.gcd a b

-- Define the problem
theorem GCF_LCM_18_30_10_45_eq_90 : 
  GCF (LCM 18 30) (LCM 10 45) = 90 := by
sorry

end GCF_LCM_18_30_10_45_eq_90_l621_621432


namespace lowest_positive_integer_divisible_by_1_through_2_l621_621187

theorem lowest_positive_integer_divisible_by_1_through_2 : ∃ x : ℕ, (∀ n ∈ {1, 2}, n ∣ x) ∧ ∀ y : ℕ, (∀ n ∈ {1, 2}, n ∣ y) → x ≤ y := sorry

end lowest_positive_integer_divisible_by_1_through_2_l621_621187


namespace value_of_a_l621_621380

theorem value_of_a (a : ℝ) : 
  let coeff := (binom 6 3) * (-a)^3 in
  coeff = 20 ↔ a = -1 := 
by 
  sorry

end value_of_a_l621_621380


namespace pool_filling_l621_621658

theorem pool_filling :
  (∃ (r1 r2 r3 rO : ℝ),
  (r1 + r2 - rO = 1 / 6) ∧
  (r1 + r3 - rO = 1 / 5) ∧
  (r2 + r3 - rO = 1 / 4) ∧
  (r1 + r2 + r3 - rO = 1 / 3) ∧ 
  ((r1 + r2 + r3) ≠ 0) ∧
  (1 / (r1 + r2 + r3) = 60 / 23)) :=
begin
  sorry
end

end pool_filling_l621_621658


namespace cube_vertex_face_sum_product_l621_621017

theorem cube_vertex_face_sum_product :
  let possible_sums := {14, 10, 6, 2, -2, -6, -10}
  ∃ (vertices : Fin 8 → ℤ) (faces : Fin 6 → ℤ), 
  (∀ i, vertices i = 1 ∨ vertices i = -1) ∧
  (∀ j, faces j = vertices (face_vertex_index j 0) * vertices (face_vertex_index j 1) * 
                   vertices (face_vertex_index j 2) * vertices (face_vertex_index j 3)) ∧
  ∃ (sums : List ℤ), (∀ sum ∈ sums, sum ∈ possible_sums) ∧
  (List.prod sums) = -20160 := by
  sorry

end cube_vertex_face_sum_product_l621_621017


namespace triangle_perimeter_l621_621982

-- Given conditions
def inradius : ℝ := 2.5
def area : ℝ := 40

-- The formula relating inradius, area, and perimeter
def perimeter_formula (r a p : ℝ) : Prop := a = r * p / 2

-- Prove the perimeter p of the triangle
theorem triangle_perimeter : ∃ (p : ℝ), perimeter_formula inradius area p ∧ p = 32 := by
  sorry

end triangle_perimeter_l621_621982


namespace sum_and_reverse_check_l621_621641

-- Definition of sum operation and digit reversal
def sum (a b : ℕ) := a + b
def reverse_digits (n : ℕ) : ℕ :=
  n.to_string.reverse.to_nat

-- Conditions from the problem
def original_sum : ℕ := sum 137 276
def expected_reversed_sum : ℕ := 534

-- Theorem statement: prove the sum is 413 and its reverse is not 534
theorem sum_and_reverse_check : 
  original_sum = 413 ∧ reverse_digits original_sum ≠ expected_reversed_sum := by
  sorry

end sum_and_reverse_check_l621_621641


namespace sum_of_n_values_with_57_trailing_zeros_l621_621345

def trailing_zeros (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 + n / 15625 + n / 78125 + 
  n / 390625 + n / 1953125 + n / 9765625 + n / 48828125 + n / 244140625 + 
  n / 1220703125 + n / 6103515625 + n / 30517578125 + n / 152587890625 + 
  n / 762939453125 + n / 3814697265625 + n / 19073486328125 + n / 95367431640625

theorem sum_of_n_values_with_57_trailing_zeros : 
  (∑ n in { n : ℕ | trailing_zeros n = 57 }, n) = 1185 :=
by
  sorry

end sum_of_n_values_with_57_trailing_zeros_l621_621345


namespace race_participants_least_number_l621_621497

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l621_621497


namespace quadratic_min_value_l621_621382

theorem quadratic_min_value (p r : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x^2 + 2 * p * x + r) (h₁ : ∃ x₀, f x₀ = 1 ∧ ∀ x, f x₀ ≤ f x) : r = p^2 + 1 :=
by
  sorry

end quadratic_min_value_l621_621382


namespace domain_of_sqrt_fun_l621_621967

theorem domain_of_sqrt_fun : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 7 → 7 + 6 * x - x^2 ≥ 0) :=
sorry

end domain_of_sqrt_fun_l621_621967


namespace range_of_abs_product_l621_621812

noncomputable def find_range_abs_product (z1 z2 : ℂ) 
  (h1 : |z1 + z2| = 4)
  (h2 : |z1 - z2| = 3) : Prop :=
  |z1 * z2| ∈ Icc (7 / 4) (25 / 4)

theorem range_of_abs_product (z1 z2 : ℂ) 
  (h1 : |z1 + z2| = 4)
  (h2 : |z1 - z2| = 3) : find_range_abs_product z1 z2 h1 h2 :=
begin
  sorry,
end

end range_of_abs_product_l621_621812


namespace no_such_function_exists_l621_621078

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, (f(x) + f(y)) / 2 ≥ f((x + y) / 2) + |x - y| := 
by
  sorry

end no_such_function_exists_l621_621078


namespace find_angle_C_find_max_sin_A_plus_sin_B_l621_621881

variable {A B C a b c S : ℝ}
variable (triangle_ABC : ∀ {A B C : ℝ}, Triangle ABC)

-- Conditions
axiom sides_opposite (A B C: ℝ) : S = ℝ
axiom area_S : S = (sqrt 3)/4 * (a^2 + b^2 - c^2)

-- Questions to prove
theorem find_angle_C (A B C: ℝ) (a b c: ℝ) [Triangle ABD]: 
  (S = (sqrt 3) / 4 * (a^2 + b^2 - c^2)) ∧ (triangle_ABC) -> (C = π/3) :=
begin
  sorry
end

theorem find_max_sin_A_plus_sin_B (A B C: ℝ) (a b c: ℝ) [Triangle ABD]: 
  (S = (sqrt 3) / 4 * (a^2 + b^2 - c^2)) ∧ (C = π/3) -> (sin A + sin B ≤ sqrt 3) :=
begin
  sorry
end

end find_angle_C_find_max_sin_A_plus_sin_B_l621_621881


namespace troll_count_by_forest_l621_621255

def bridge_count (T : ℕ) := 4 * T - 6
def plains_count (T : ℕ) := (1 / 2 : ℝ) * (4 * T - 6)

theorem troll_count_by_forest (T : ℕ) (h : T + bridge_count T + (plains_count T).toNat = 33) : T = 6 :=
by
  sorry

end troll_count_by_forest_l621_621255


namespace partI_tangent_line_partII_monotonicity_l621_621842

section ProblemI

def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x - Real.log x

def tangent_line_equation : Prop := ∀ x y : ℝ, x - y - 1 = 0

theorem partI_tangent_line : tangent_line_equation (x := 1) (y := f 1) := 
by 
  sorry

end ProblemI

section ProblemII

def f (a x : ℝ) : ℝ := a * x^2 - a * x - Real.log x
def f' (a x : ℝ) : ℝ := 2 * a * x - a - 1 / x
def g (a x : ℝ) : ℝ := 2 * a * x^2 - a * x - 1

theorem partII_monotonicity (a : ℝ) :
  (a = 0 → ∀ x > 0, Rational.neighbor (f a x) (f a)) ∧
  (-8 ≤ a ∧ a < 0 → ∀ x > 0, Rational.decreasing (f a x) (f a)) ∧
  (a > 0 → 
    ∀ x > 0, 
      (x < (a + Real.sqrt (a^2 + 8 * a)) / (4 * a) → Rational.decreasing (f a x) (f a)) ∧
      (x > (a + Real.sqrt (a^2 + 8 * a)) / (4 * a) → Rational.increasing (f a x) (f a))) ∧
  (a < -8 → 
    ∀ x > 0, 
      if x < (a + Real.sqrt (a^2 + 8 * a)) / (4 * a) then Rational.decreasing (f a x) (f a)
      else if x > (a + Real.sqrt (a^2 + 8 * a)) / (4 * a) ∧ x < (a - Real.sqrt (a^2 + 8 * a)) / (4 * a) then Rational.increasing (f a x) (f a)
      else Rational.decreasing (f a x) (f a)) := by
  sorry

end ProblemII

end partI_tangent_line_partII_monotonicity_l621_621842


namespace average_waiting_time_for_first_bite_l621_621704

theorem average_waiting_time_for_first_bite
  (bites_first_rod : ℝ)
  (bites_second_rod: ℝ)
  (total_bites: ℝ)
  (time_interval: ℝ)
  (H1 : bites_first_rod = 3)
  (H2 : bites_second_rod = 2)
  (H3 : total_bites = 5)
  (H4 : time_interval = 6) :
  1 / (total_bites / time_interval) = 1.2 :=
by
  rw [H3, H4]
  simp
  norm_num
  rw [div_eq_mul_inv, inv_div, inv_inv]
  norm_num
  sorry

end average_waiting_time_for_first_bite_l621_621704


namespace count_bad_arrangements_l621_621573

def is_bad_arrangement (arr : List ℕ) : Prop :=
  let n := arr.length
  ∀ m in (1 : ℕ) :: [2..n], ¬(List.sum (arr.take m) = 10 ∨ List.sum (arr.take m) = 12)

theorem count_bad_arrangements : 
  (Finset.univ.filter is_bad_arrangement).card = 2 := 
sorry

end count_bad_arrangements_l621_621573


namespace P_2018_roots_l621_621633

def P : ℕ → (ℝ → ℝ)
| 0     := λ x, 1
| 1     := λ x, x
| (n+2) := λ x, x * (P (n+1) x) - (P n x)

theorem P_2018_roots : 
  ∃ S : Finset ℝ, S.card = 2018 ∧ ∀ x ∈ S, P 2018 x = 0 ∧ 
  ∀ x₁ x₂ ∈ S, x₁ ≠ x₂ → x₁ ≠ x₂ :=
begin
  sorry
end

end P_2018_roots_l621_621633


namespace area_quadrilateral_regular_polgon_l621_621816

variable (n : ℕ) (h : n > 1) (S : ℝ)

def area_of_quadrilateral (A1 An An1 An2 : ℂ) (poly : list ℂ) (hpoly : n > 1) (hreg : is_regular_polygon poly 4*n) : ℝ :=
  S / 2

theorem area_quadrilateral_regular_polgon (n : ℕ) (h : n > 1) (S : ℝ) (poly : list ℂ) (hreg : is_regular_polygon poly (4*n))
    (A1 An An1 An2 : ℂ) (h1 : poly.nth 0 = some A1) (h2 : poly.nth (n) = some An)
    (h3 : poly.nth (n+1) = some An1) (h4 : poly.nth (n+2) = some An2):
  area_of_quadrilateral n h S A1 An An1 An2 poly h hreg = S / 2 := 
sorry

end area_quadrilateral_regular_polgon_l621_621816


namespace positive_number_is_25_l621_621546

theorem positive_number_is_25 {a x : ℝ}
(h1 : x = (3 * a + 1)^2)
(h2 : x = (-a - 3)^2)
(h_sum : 3 * a + 1 + (-a - 3) = 0) :
x = 25 :=
sorry

end positive_number_is_25_l621_621546


namespace fraction_equivalence_l621_621753

noncomputable def x := 0.812812812 -- Equivalent to 0.\overline{812}
noncomputable def y := 2.406406406 -- Equivalent to 2.\overline{406}

theorem fraction_equivalence : (x / y) = (203 / 601) :=
begin
  sorry
end

end fraction_equivalence_l621_621753


namespace quadratic_root_intervals_l621_621968

theorem quadratic_root_intervals (m : ℝ) :
  (∀ f : ℝ → ℝ, f = (λ x, x^2 - 2*m*x + m^2 - 1) → f 0 > 0 → f 1 < 0 → f 2 < 0 → f 3 > 0) → (1 < m ∧ m < 2) :=
  by {
    intros f hf h0 h1 h2 h3,
    sorry
  }

end quadratic_root_intervals_l621_621968


namespace find_m_l621_621136

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m - 1) > 0) ∧ m = 3 :=
sorry

end find_m_l621_621136


namespace greatest_possible_length_l621_621168

theorem greatest_possible_length :
  ∃ (g : ℕ), g = Nat.gcd 700 (Nat.gcd 385 1295) ∧ g = 35 :=
by
  sorry

end greatest_possible_length_l621_621168


namespace I_nm_expr_I_odd_expr_factorial_binom_expr_l621_621793

noncomputable def I (n m : ℕ) : ℝ := ∫ θ in 0..(Real.pi / 2), (Real.cos θ) ^ n * (Real.sin θ) ^ m

theorem I_nm_expr (n m : ℕ) (h : 2 ≤ n) :
  I n m = (n - 1) / (m + 1) * I (n - 2) (m + 2) :=
by sorry

theorem I_odd_expr (n m : ℕ) :
  I (2 * n + 1) (2 * m + 1) = (1 / 2) * ∫ x in 0..1, x ^ n * (1 - x) ^ m :=
by sorry

theorem factorial_binom_expr (n m : ℕ) :
  (n.factorial * m.factorial : ℝ) / ((n + m + 1).factorial) = 
  ∑ k in Finset.range (m + 1), (-1 : ℝ) ^ k * Nat.choose m k / (n + k + 1) :=
by sorry

end I_nm_expr_I_odd_expr_factorial_binom_expr_l621_621793


namespace a_6_value_l621_621638

-- Definition of the sequence and sum conditions
def a : ℕ → ℕ
def S : ℕ → ℕ

-- Conditions
axiom h1 : a 1 = 1
axiom h2 : ∀ n, a (n + 1) = 3 * S n
axiom h3 : ∀ n, S n = ∑ i in Finset.range n + 1, a i

-- The proof problem statement
theorem a_6_value : a 6 = 3 * 4^4 := by
  sorry

end a_6_value_l621_621638


namespace perpendicular_diagonals_necessity_sufficiency_l621_621228

-- Definitions of quadrilaterals and perpendicular diagonals
structure Quadrilateral :=
(a b c d : Point)
(h_ab : Segment a b)
(h_bc : Segment b c)
(h_cd : Segment c d)
(h_da : Segment d a)

def diagonals_perpendicular (q : Quadrilateral) : Prop :=
  let diag1 := Segment q.a q.c
  let diag2 := Segment q.b q.d
  ∃ m : Point, diag1.contains m ∧ diag2.contains m ∧ IsPerpendicular diag1 diag2

-- Problem statement: Perpendicular diagonals are necessary and sufficient for squares and rhombi, but not for general quadrilaterals or other specific types.
theorem perpendicular_diagonals_necessity_sufficiency :
  ∀ (q : Quadrilateral),
    (is_square q → diagonals_perpendicular q) ∧ 
    (diagonals_perpendicular q → is_square q) ∧
    (is_rhombus q → diagonals_perpendicular q) ∧
    (diagonals_perpendicular q → is_rhombus q) :=
by
  intro q
  apply and.intro
  -- proof here (omitted)
  sorry

end perpendicular_diagonals_necessity_sufficiency_l621_621228


namespace part_1_part_2_l621_621844

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - abs (x - 4)

theorem part_1 :
  ∀ x : ℝ, f x 2 ∈ set.Icc (-2 : ℝ) 2 :=
by
  sorry

theorem part_2 {m : ℝ} :
  (∀ x₀ : ℝ, f x₀ 2 ≤ m - m^2) → m ∈ set.Icc (-1 : ℝ) 2 :=
by
  sorry

end part_1_part_2_l621_621844


namespace visual_acuity_conversion_l621_621476

theorem visual_acuity_conversion (V : ℝ) (L : ℝ) (hV : V = 0.8) (hL : L = 5 + Real.log10 V) (h_log2 : Real.log10 2 = 0.30) : L = 4.9 :=
by
  -- actual proof steps are omitted and denoted by 'sorry'
  sorry

end visual_acuity_conversion_l621_621476


namespace race_minimum_participants_l621_621517

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l621_621517


namespace inequality_x_y_l621_621333

theorem inequality_x_y (x y : ℝ) (h : x^4 + y^4 ≤ 1) : 
  x^6 - y^6 + 2 * y^3 < real.pi / 2 := 
by
  sorry

end inequality_x_y_l621_621333


namespace minimum_participants_l621_621488

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l621_621488


namespace total_profit_at_end_of_year_l621_621959

theorem total_profit_at_end_of_year (investment_suresh investment_rohan investment_sudhir : ℕ)
(join_month_rohan join_month_sudhir : ℕ)
(profit_difference_rohan_sudhir : ℕ)
(total_months : ℕ)
(h_suresh : investment_suresh = 18000)
(h_rohan : investment_rohan = 12000)
(h_sudhir : investment_sudhir = 9000)
(h_join_rohan : join_month_rohan = 3)
(h_join_sudhir : join_month_sudhir = 4)
(h_profit_diff : profit_difference_rohan_sudhir = 352)
(h_total_months : total_months = 12) : 
  (let ratio_suresh := investment_suresh * total_months,
       ratio_rohan := investment_rohan * (total_months - join_month_rohan),
       ratio_sudhir := investment_sudhir * (total_months - join_month_sudhir),
       total_ratio := ratio_suresh + ratio_rohan + ratio_sudhir,
       simplified_ratio_suresh := 6,
       simplified_ratio_rohan := 3,
       simplified_ratio_sudhir := 2,
       total_parts := simplified_ratio_suresh + simplified_ratio_rohan + simplified_ratio_sudhir in
    total_parts * profit_difference_rohan_sudhir = 3872) :=
by
  sorry

end total_profit_at_end_of_year_l621_621959


namespace evaluate_ceil_of_neg_sqrt_l621_621259

-- Define the given expression and its value computation
def given_expression : ℚ := -real.sqrt (64 / 9)

-- Define the expected answer
def expected_answer : ℤ := -2

-- State the theorem to be proven
theorem evaluate_ceil_of_neg_sqrt : (Int.ceil given_expression) = expected_answer := sorry

end evaluate_ceil_of_neg_sqrt_l621_621259


namespace sean_needs_six_packs_l621_621108

/-- 
 Sean needs to replace 2 light bulbs in the bedroom, 
 1 in the bathroom, 1 in the kitchen, and 4 in the basement. 
 He also needs to replace 1/2 of that amount in the garage. 
 The bulbs come 2 per pack. 
 -/
def bedroom_bulbs: ℕ := 2
def bathroom_bulbs: ℕ := 1
def kitchen_bulbs: ℕ := 1
def basement_bulbs: ℕ := 4
def bulbs_per_pack: ℕ := 2

noncomputable def total_bulbs_needed_including_garage: ℕ := 
  let total_rooms_bulbs := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs
  let garage_bulbs := total_rooms_bulbs / 2
  total_rooms_bulbs + garage_bulbs

noncomputable def total_packs_needed: ℕ := total_bulbs_needed_including_garage / bulbs_per_pack

theorem sean_needs_six_packs : total_packs_needed = 6 :=
by
  sorry

end sean_needs_six_packs_l621_621108


namespace relationship_between_a_b_c_l621_621318

def pow (a b : ℝ) : ℝ := a^b

def lb (a : ℝ) : ℝ := log 10 a

def log_base (b x : ℝ) : ℝ := log b x

noncomputable def a : ℝ := pow 2 0.2
noncomputable def b : ℝ := 1 - 2 * lb 2
noncomputable def c : ℝ := 2 - log_base 3 10

theorem relationship_between_a_b_c : a > b ∧ b > c :=
by
  -- skipping proof steps
  sorry

end relationship_between_a_b_c_l621_621318


namespace plane_parallel_and_perpendicular_l621_621856

-- Definitions and theorem statement
theorem plane_parallel_and_perpendicular (α β : Plane) (m : Line) :
  (α ∥ β) → (m ⊥ α) → (m ⊥ β) :=
sorry

end plane_parallel_and_perpendicular_l621_621856


namespace sun_set_delay_minutes_l621_621094

-- Problem: Prove that the number of minutes until the sun sets is 38 minutes
-- given the conditions.

def minutes_until_sunset (initial_sunset_minutes : ℕ) (delay_per_day : ℕ) 
  (days_since_march_first : ℕ) (current_time_minutes : ℕ) : ℕ :=
  let total_delay := delay_per_day * days_since_march_first in
  let new_sunset_time := initial_sunset_minutes + total_delay in
  new_sunset_time - current_time_minutes

theorem sun_set_delay_minutes :
  minutes_until_sunset 1080 1.2.nat 40 1110 = 38 :=
  by sorry

end sun_set_delay_minutes_l621_621094


namespace total_ticket_count_is_59_l621_621997

-- Define the constants and variables
def price_adult : ℝ := 4
def price_student : ℝ := 2.5
def total_revenue : ℝ := 222.5
def student_tickets_sold : ℕ := 9

-- Define the equation representing the total revenue and solve for the number of adult tickets
noncomputable def total_tickets_sold (adult_tickets : ℕ) :=
  adult_tickets + student_tickets_sold

theorem total_ticket_count_is_59 (A : ℕ) 
  (h : price_adult * A + price_student * (student_tickets_sold : ℝ) = total_revenue) :
  total_tickets_sold A = 59 :=
by
  sorry

end total_ticket_count_is_59_l621_621997


namespace num_bad_arrangements_l621_621980
open List

def is_bad_arrangement (l : List ℕ) : Prop :=
  let sums := foldl (λ A i, A ∪ (List.powerset l).filter (λ s, s.length > 0 ).map sum ) ∅ (List.range l.length)
  ∀ n, n ∈ (Finset.range 16) → n ∉ sums

theorem num_bad_arrangements : 
  (Finset.filter is_bad_arrangement (Equiv.List.ListEquivFinset (List![$1, $2, $3, $4, $5]))) := 
  2 := 
sorry

end num_bad_arrangements_l621_621980


namespace sue_necklace_total_beads_l621_621682

theorem sue_necklace_total_beads :
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  (purple + blue + green = 46) :=
by
  intros purple blue green h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sue_necklace_total_beads_l621_621682


namespace coffee_shop_ratio_l621_621749

theorem coffee_shop_ratio (morning_usage afternoon_multiplier weekly_usage days_per_week : ℕ) (r : ℕ) 
  (h_morning : morning_usage = 3)
  (h_afternoon : afternoon_multiplier = 3)
  (h_weekly : weekly_usage = 126)
  (h_days : days_per_week = 7):
  weekly_usage = days_per_week * (morning_usage + afternoon_multiplier * morning_usage + r * morning_usage) →
  r = 2 :=
by
  intros h_eq
  sorry

end coffee_shop_ratio_l621_621749


namespace subset_divisible_by_n_l621_621105

theorem subset_divisible_by_n (n : ℕ) (a : Fin n → ℕ) (hpos : 0 < n) : 
  ∃ (s : Finset (Fin n)), s.nonempty ∧ (∑ i in s, a i) % n = 0 := 
sorry

end subset_divisible_by_n_l621_621105


namespace price_of_movie_ticket_l621_621378

theorem price_of_movie_ticket
  (M F : ℝ)
  (h1 : 8 * M = 2 * F)
  (h2 : 8 * M + 5 * F = 840) :
  M = 30 :=
by
  sorry

end price_of_movie_ticket_l621_621378


namespace semicircle_chords_product_l621_621045

theorem semicircle_chords_product : 
  let A := 3
  let B := -3
  let radius := 3
  let ω := Complex.exp (2 * Real.pi * Complex.I / 18)
  let C := fun k => 3 * ω^k
  (∏ k in Finset.range 1 9, Complex.abs (A - C k)) * 
  (∏ k in Finset.range 1 9, Complex.abs (B - C k)) = 9437184 := 
by
  sorry

end semicircle_chords_product_l621_621045


namespace sum_abs_diff_le_n_squared_sum_abs_diff_eq_n_squared_iff_l621_621103

theorem sum_abs_diff_le_n_squared (n : ℕ) (a : Fin n → ℝ) (h1 : 2 ≤ n)
  (h2 : ∀ i, 0 ≤ a i ∧ a i ≤ 2) :
  (∑ i j, |a i - a j|) ≤ n^2 :=
sorry

theorem sum_abs_diff_eq_n_squared_iff (n : ℕ) (a : Fin n → ℝ) (h1 : 0 < n)
  (h2 : ∀ i, 0 ≤ a i ∧ a i ≤ 2) :
  (∑ i j, |a i - a j| = n^2) ↔ (∃ k, n = 2 * k ∧ (∀ i, (a i = 0) ∨ (a i = 2)) ∧ (∃ S, S.card = k ∧ ∀ i, i ∈ S ↔ a i = 2)) :=
sorry

end sum_abs_diff_le_n_squared_sum_abs_diff_eq_n_squared_iff_l621_621103


namespace ceil_neg_sqrt_64_over_9_l621_621270

theorem ceil_neg_sqrt_64_over_9 : Real.ceil (-Real.sqrt (64 / 9)) = -2 := 
by
  sorry

end ceil_neg_sqrt_64_over_9_l621_621270


namespace range_of_k_l621_621797

noncomputable def f (x : ℝ) : ℝ := Real.log x + x

def is_ktimes_value_function (f : ℝ → ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  0 < k ∧ a < b ∧ f a = k * a ∧ f b = k * b

theorem range_of_k (k : ℝ) : (∃ a b : ℝ, is_ktimes_value_function f k a b) ↔ 1 < k ∧ k < 1 + 1 / Real.exp 1 := by
  sorry

end range_of_k_l621_621797


namespace probability_correct_l621_621313

open Finset

variables (s : Finset ℕ) (n m : ℕ)

-- Define the set of numbers
def set_of_numbers : Finset ℕ := {1, 2, 3, 4}

-- Condition: Two numbers are drawn without replacement
def possible_outcomes (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s \ s.diag

-- Calculate the number of total outcomes
def total_outcomes : ℕ := (possible_outcomes (set_of_numbers)).card

-- Condition: The two numbers should both be even
def even_numbers : Finset ℕ := {2, 4}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (possible_outcomes (even_numbers)).filter (λ (p : ℕ × ℕ), p.1 ∈ even_numbers ∧ p.2 ∈ even_numbers)

def number_of_favorable_outcomes : ℕ := favorable_outcomes.card

-- Final probability calculation
def probability_of_drawing_two_even_numbers := (number_of_favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_correct : probability_of_drawing_two_even_numbers == (1 : ℚ) / (6 : ℚ) :=
  by sorry

end probability_correct_l621_621313


namespace sum_of_ages_is_26_l621_621607

def Yoongi_aunt_age := 38
def Yoongi_age := Yoongi_aunt_age - 23
def Hoseok_age := Yoongi_age - 4
def sum_of_ages := Yoongi_age + Hoseok_age

theorem sum_of_ages_is_26 : sum_of_ages = 26 :=
by
  sorry

end sum_of_ages_is_26_l621_621607


namespace B_share_in_profit_l621_621674

variable 
  (x : ℝ) -- Investment of B
  (total_profit : ℝ := 22000) -- Total profit at the end of the year

-- Conditions
variable 
  (A_investment : ℝ := 3 * x)
  (C_investment : ℝ := 3 / 2 * x)
  (D_investment : ℝ := 1 / 2 * (3 * x + x + 3 / 2 * x))
  (A_time : ℝ := 6)
  (B_time : ℝ := 9)
  (C_time : ℝ := 12)
  (D_time : ℝ := 4)

-- Calculations
def A_share : ℝ := A_investment * A_time
def B_share : ℝ := x * B_time
def C_share : ℝ := C_investment * C_time
def D_share : ℝ := D_investment * D_time

def total_ratio : ℝ := A_share + B_share + C_share + D_share

-- Proof problem
def B_profit_share : ℝ := (B_share / total_ratio) * total_profit

theorem B_share_in_profit : B_profit_share = 3666.67 :=
by 
  sorry

end B_share_in_profit_l621_621674


namespace Sues_necklace_total_beads_l621_621683

theorem Sues_necklace_total_beads 
  (purple_beads : ℕ)
  (blue_beads : ℕ)
  (green_beads : ℕ)
  (h1 : purple_beads = 7)
  (h2 : blue_beads = 2 * purple_beads)
  (h3 : green_beads = blue_beads + 11) :
  purple_beads + blue_beads + green_beads = 46 :=
by
  sorry

end Sues_necklace_total_beads_l621_621683


namespace sequence_general_term_l621_621851

noncomputable def a : ℕ → ℕ
| 0       := 0  -- Normally a_0 would be outside the scope, as n is in ℕ^*
| 1       := 1
| (n + 2) := 3 * a (n + 1) + 4

theorem sequence_general_term (n : ℕ) (h : n ≥ 1) : a n = 3^n - 2 :=
by sorry

end sequence_general_term_l621_621851


namespace find_x_if_opposites_l621_621315

theorem find_x_if_opposites (x : ℝ) (h : 2 * (x - 3) = - 4 * (1 - x)) : x = -1 := 
by
  sorry

end find_x_if_opposites_l621_621315


namespace knights_on_island_l621_621459

-- Definitions based on conditions
inductive Inhabitant : Type
| knight : Inhabitant
| knave : Inhabitant

open Inhabitant

def statement_1 (inhabitant : Inhabitant) : Prop :=
inhabitant = knight

def statement_2 (inhabitant1 inhabitant2 : Inhabitant) : Prop :=
inhabitant1 = knight ∧ inhabitant2 = knight

def statement_3 (inhabitant1 inhabitant2 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave)) / 2 ≥ 0.5

def statement_4 (inhabitant1 inhabitant2 inhabitant3 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave) + ↑(inhabitant3 = knave)) / 3 ≥ 0.65

def statement_5 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 : Inhabitant) : Prop :=
(↑(inhabitant1 = knight) + ↑(inhabitant2 = knight) + ↑(inhabitant3 = knight) + ↑(inhabitant4 = knight)) / 4 ≥ 0.5

def statement_6 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 inhabitant5 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave) + ↑(inhabitant3 = knave) + ↑(inhabitant4 = knave) + ↑(inhabitant5 = knave)) / 5 ≥ 0.4

def statement_7 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 inhabitant5 inhabitant6 : Inhabitant) : Prop :=
(↑(inhabitant1 = knight) + ↑(inhabitant2 = knight) + ↑(inhabitant3 = knight) + ↑(inhabitant4 = knight) + ↑(inhabitant5 = knight) + ↑(inhabitant6 = knight)) / 6 ≥ 0.65

-- Lean Statement
theorem knights_on_island (inhabitants : Fin 7 → Inhabitant) :
  (∀ i, (inhabitants i = knight ↔ (i = 0) ∨ (i = 1) ∨ (i = 4) ∨ (i = 5) ∨ (i = 6))) → 5 :=
by
  sorry

end knights_on_island_l621_621459


namespace one_less_than_neg_one_is_neg_two_l621_621653

theorem one_less_than_neg_one_is_neg_two : (-1 - 1 = -2) :=
by
  sorry

end one_less_than_neg_one_is_neg_two_l621_621653


namespace max_elements_of_T_l621_621052

theorem max_elements_of_T : 
  ∃ (T : Set ℕ), T ⊆ {x | 1 ≤ x ∧ x ≤ 2023} ∧ 
    (∀ a ∈ T, ∀ b ∈ T, a ≠ b → (a - b) % 5 ≠ 0 ∧ (a - b) % 8 ≠ 0) ∧ 
    T.finite ∧ T.to_finset.card = 780 :=
sorry

end max_elements_of_T_l621_621052


namespace largest_integer_divisor_l621_621595

theorem largest_integer_divisor (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
sorry

end largest_integer_divisor_l621_621595


namespace P_n_real_roots_P_2018_real_roots_l621_621621

noncomputable def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 1) := λ x, x * P n x - P (n - 1) x

theorem P_n_real_roots (n: ℕ) : ∃ r: ℕ, r ≡ n := sorry

theorem P_2018_real_roots : ∃ r: ℕ, r = 2018 := P_n_real_roots 2018

end P_n_real_roots_P_2018_real_roots_l621_621621


namespace triangle_cannot_be_divided_l621_621604

-- Definitions of shapes
structure Rectangle where
  a b : ℝ
  h1 : a > 0
  h2 : b > 0

structure Square where
  a : ℝ
  h : a > 0

structure RegularHexagon where
  a : ℝ
  h : a > 0

structure Trapezium where
  a b c d : ℝ
  h1 : (a = b) ∨ (c = d)

structure Triangle where
  a b c : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : c > 0

-- Function to check if a shape can be divided into two trapeziums
def canBeDividedIntoTwoTrapeziums (shape : Type) : Prop := sorry

-- Theorem stating that a triangle cannot be divided into two trapeziums
theorem triangle_cannot_be_divided : ¬ canBeDividedIntoTwoTrapeziums Triangle := sorry

end triangle_cannot_be_divided_l621_621604


namespace ceil_neg_sqrt_64_over_9_l621_621269

theorem ceil_neg_sqrt_64_over_9 : Real.ceil (-Real.sqrt (64 / 9)) = -2 := 
by
  sorry

end ceil_neg_sqrt_64_over_9_l621_621269


namespace sum_bound_l621_621044

theorem sum_bound (n : ℕ) (a : ℕ → ℝ)
  (h1 : (∑ i in Finset.range n, a i) = 0)
  (h2 : (∑ i in Finset.range n, |a i|) = 1) :
  |∑ i in Finset.range n, (i + 1) * a i| ≤ (n - 1) / 2 := 
  sorry 

end sum_bound_l621_621044


namespace find_third_vertex_l621_621592

def third_vertex_coordinates (A B : ℝ × ℝ) (ha : A = (8, 6)) (hb : B = (0, 0)) (area : ℝ) (h_area : area = 48) (H : ℝ × ℝ) : Prop :=
  let height := A.2
  let base := (2 * area) / height
  H = (-base, 0)

theorem find_third_vertex : ∃ H, third_vertex_coordinates (8, 6) (0, 0) 48 (8, 6) H :=
by
  -- H = (-16, 0)
  sorry

end find_third_vertex_l621_621592


namespace sum_fractions_eq_l621_621727

theorem sum_fractions_eq :
  (∑ i in finset.range 16, (i + 2) / 3 : ℚ) = 152 / 3 :=
by
  sorry

end sum_fractions_eq_l621_621727


namespace numbers_div_by_53_l621_621479

theorem numbers_div_by_53 (k : ℕ) : 
    ∃ m : ℕ, (\sum i in Finset.range (k + 1), 10 ^ (k - i) + 7) = 53 * m :=
by
    sorry

end numbers_div_by_53_l621_621479


namespace ceil_neg_sqrt_frac_l621_621265

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l621_621265


namespace number_of_cars_l621_621146

theorem number_of_cars (b c : ℕ) (h1 : b = c / 10) (h2 : c - b = 90) : c = 100 :=
by
  sorry

end number_of_cars_l621_621146


namespace auditorium_total_chairs_l621_621011

theorem auditorium_total_chairs 
  (n : ℕ)
  (h1 : 2 + 5 - 1 = n)   -- n is the number of rows which is equal to 6
  (h2 : 3 + 4 - 1 = n)   -- n is the number of chairs per row which is also equal to 6
  : n * n = 36 :=        -- the total number of chairs is 36
by
  sorry

end auditorium_total_chairs_l621_621011


namespace find_number_l621_621770

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l621_621770


namespace trajectory_and_distance_l621_621829

def midpoint (M N : Point) : Point := 
  ⟨(M.x + N.x) / 2, (M.y + N.y) / 2⟩

def on_circle (M : Point) (r : ℝ) : Prop :=
  M.x^2 + M.y^2 = r^2

def distance_to_line (P : Point) (a b c : ℝ) : ℝ :=
  abs (a * P.x + b * P.y + c) / sqrt (a^2 + b^2)

theorem trajectory_and_distance (M N P : Point) 
  (r : ℝ) (a b c : ℝ) (h1 : on_circle M r) (h2 : N = ⟨4, 0⟩) (h3 : P = midpoint M N) :
  (P.x - 2)^2 + P.y^2 = 1 ∧ 
  (distance_to_line ⟨2, 0⟩ a b c = 4 → 
   ∃ max_dist min_dist : ℝ, max_dist = 5 ∧ min_dist = 3) := 
begin
  sorry
end

end trajectory_and_distance_l621_621829


namespace problem_statement_l621_621349

/- Defining the functions and given conditions -/
def f (x : ℝ) (a : ℝ) : ℝ := log (2, (sqrt (x^2 + a) - x))
def g (x : ℝ) (t : ℝ) (a : ℝ) : ℝ := t - abs (2 * x - a)

/- Stating the theorem -/
theorem problem_statement (a t : ℝ)
  (h_odd_f : ∀ x : ℝ, f (-x) a = - f x a)
  (h_zeroes : ∀ x : ℝ, f x a = 0 ↔ g x t a = 0) : 
  (a = 1 → t = 1) ∧ 
  (∀ x1 x2 : ℝ, (x1 ∈ set.Icc (-3/4 : ℝ) 2) → (x2 ∈ set.Icc (-3/4 : ℝ) 2) → f x1 a ≤ g x2 t a → t ≥ 4) :=
by
  sorry 

end problem_statement_l621_621349


namespace non_square_solution_equiv_l621_621828

theorem non_square_solution_equiv 
  (a b : ℤ) (h1 : ¬∃ k : ℤ, a = k^2) (h2 : ¬∃ k : ℤ, b = k^2) :
  (∃ x y z w : ℤ, x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) ↔
  (∃ x y z : ℤ, x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0)) :=
by sorry

end non_square_solution_equiv_l621_621828


namespace Elizabeth_More_Revenue_Than_Banks_l621_621449

theorem Elizabeth_More_Revenue_Than_Banks : 
  let banks_investments := 8
  let banks_revenue_per_investment := 500
  let elizabeth_investments := 5
  let elizabeth_revenue_per_investment := 900
  let banks_total_revenue := banks_investments * banks_revenue_per_investment
  let elizabeth_total_revenue := elizabeth_investments * elizabeth_revenue_per_investment
  elizabeth_total_revenue - banks_total_revenue = 500 :=
by
  sorry

end Elizabeth_More_Revenue_Than_Banks_l621_621449


namespace pear_sales_l621_621663

theorem pear_sales (sale_afternoon : ℕ) (h1 : sale_afternoon = 260)
  (h2 : ∃ sale_morning : ℕ, sale_afternoon = 2 * sale_morning) :
  sale_afternoon / 2 + sale_afternoon = 390 :=
by
  sorry

end pear_sales_l621_621663


namespace volume_of_pyramid_l621_621729

theorem volume_of_pyramid (A B C : ℝ × ℝ)
  (hA : A = (0, 0)) (hB : B = (28, 0)) (hC : C = (12, 20))
  (D : ℝ × ℝ) (hD : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (E : ℝ × ℝ) (hE : E = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))
  (F : ℝ × ℝ) (hF : F = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (∃ h : ℝ, h = 10 ∧ ∃ V : ℝ, V = (1 / 3) * 70 * h ∧ V = 700 / 3) :=
by sorry

end volume_of_pyramid_l621_621729


namespace solve_diophantine_equation_l621_621740

def is_solution (m n : ℕ) : Prop := 2^m - 3^n = 1

theorem solve_diophantine_equation : 
  { (m, n) : ℕ × ℕ | is_solution m n } = { (1, 0), (2, 1) } :=
by
  sorry

end solve_diophantine_equation_l621_621740


namespace range_of_m_l621_621336

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * x > m
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * m * x + 2 - m ≤ 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ m ∈ Set.Ioo (-2:ℝ) (-1) ∪ Set.Ici 1 :=
sorry

end range_of_m_l621_621336


namespace pos_int_n_real_nums_diff_l621_621298

-- Define the problem in Lean
theorem pos_int_n_real_nums_diff (n : ℕ) (hn : n > 0) :
  (∃ (a : ℕ → ℝ), {z : ℝ | ∃ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n ∧ z = a j - a i} = (finset.range ((n * (n - 1)) / 2)).image (coe : ℕ → ℝ)) ↔ n ∈ ({2, 3, 4} : set ℕ) :=
by
  sorry

end pos_int_n_real_nums_diff_l621_621298


namespace parametric_curve_trace_l621_621073

noncomputable def parametric_curve 
  (a b c d : ℝ) 
  (ϕ : ℝ) : ℝ × ℝ := 
  (a * Real.cos ϕ + b * Real.sin ϕ, c * Real.cos ϕ + d * Real.sin ϕ)

theorem parametric_curve_trace
  (a b c d : ℝ) :
  ∀ ϕ : ℝ, ∃ (x y : ℝ), 
  parametric_curve a b c d ϕ = (x, y) ∧ 
  ((a * d ≠ b * c ∧ (∃ (e f : ℝ), e * x^2 + f * y^2 + ... = 1)) ∨ 
   (a * d = b * c ∧ ∃ (E F G : ℝ), E * x + F * y + G = 0)) :=
 by
  sorry

end parametric_curve_trace_l621_621073


namespace fractional_inequality_solution_set_l621_621149

theorem fractional_inequality_solution_set :
  {x : ℝ | (x > 3 ∨ (1 < x ∧ x ≤ 2) ∨ x ≤ 1/2) } =
  {x : ℝ | ∀ r : ℝ, (x^2 - x - 1) / (x^2 - 4x + 3) ≥ -1 } :=
by
  sorry

end fractional_inequality_solution_set_l621_621149


namespace a_x1_x2_x13_eq_zero_l621_621563

theorem a_x1_x2_x13_eq_zero {a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ}
  (h1: a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) *
             (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2: a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) *
             (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := by
  sorry

end a_x1_x2_x13_eq_zero_l621_621563


namespace average_waiting_time_l621_621695

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l621_621695


namespace problem_statement_a_problem_statement_b_problem_statement_c_l621_621405

-- Define θ2n as described in the problem statement
def θ2n (S : ℕ → ℤ) (n : ℕ) : ℕ :=
if h : ∃ k > 1, (∀ i < k, S i < S k) ∧ (∀ i > k, i ≤ 2 * n → S i ≤ S k) then
  Classical.choose h
else
  0

-- Define the probability measure P
noncomputable def P {Ω : Type*} [ProbabilitySpace Ω] (event : set Ω) : ℝ :=
Probability.measure_space.unnormalized_measure event

-- Define the sequences of random variables S_0, S_1, ..., S_{2n}
variable {Ω : Type*} [ProbabilitySpace Ω] (S : ℕ → Ω → ℤ)

-- Define the indicators u2n, u2k, u2n_2k
variable {u2n u2k u2n_2k : ℕ → ℝ}

-- Prove the three conditions as Lean statements
theorem problem_statement_a (S : ℕ → Ω → ℤ) (u2n : ℕ → ℝ) (n : ℕ) :
  P {ω | θ2n (λ i, S i ω) n = 0} = u2n (2 * n) :=
sorry

theorem problem_statement_b (S : ℕ → Ω → ℤ) (u2n : ℕ → ℝ) (n : ℕ) :
  P {ω | θ2n (λ i, S i ω) n = 2 * n} = (1 / 2) * u2n (2 * n) :=
sorry

theorem problem_statement_c (S : ℕ → Ω → ℤ) (u2k u2n_2k : ℕ → ℝ) (n k : ℕ) (hkn : 0 < k ∧ k < n) :
  P {ω | θ2n (λ i, S i ω) n = 2 * k ∨ θ2n (λ i, S i ω) n = 2 * k + 1} = (1 / 2) * u2k (2 * k) * u2n_2k (2 * (n - k)) :=
sorry

end problem_statement_a_problem_statement_b_problem_statement_c_l621_621405


namespace find_unknown_number_l621_621985

theorem find_unknown_number :
  (0.86 ^ 3 - 0.1 ^ 3) / (0.86 ^ 2) + x + 0.1 ^ 2 = 0.76 → 
  x = 0.115296 :=
sorry

end find_unknown_number_l621_621985


namespace difference_largest_smallest_two_digit_l621_621785

theorem difference_largest_smallest_two_digit (a b c d : ℕ) 
    (h1 : {a, b, c, d} = {8, 3, 4, 6})
    (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
    let largest := max (10 * a + b) (max (10 * a + c) (max (10 * a + d) (max (10 * b + c) (max (10 * b + d) (max (10 * c + d) (max (10 * c + a) (max (10 * d + a) (10 * d + b))))))))
    let smallest := min (10 * a + b) (min (10 * a + c) (min (10 * a + d) (min (10 * b + c) (min (10 * b + d) (min (10 * c + d) (min (10 * c + a) (min (10 * d + a) (10 * d + b))))))))
    in largest - smallest = 52 := 
by
    sorry

end difference_largest_smallest_two_digit_l621_621785


namespace find_k_l621_621875

theorem find_k (k : ℝ) :
  (∀ x, x ≠ 1 → (1 / (x^2 - x) + (k - 5) / (x^2 + x) = (k - 1) / (x^2 - 1))) →
  (1 / (1^2 - 1) + (k - 5) / (1^2 + 1) ≠ (k - 1) / (1^2 - 1)) →
  k = 3 :=
by
  sorry

end find_k_l621_621875


namespace race_participants_least_number_l621_621495

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l621_621495


namespace intersection_area_l621_621587

noncomputable def area_of_intersection (r : ℝ) : ℝ := (π * r^2) * (1/4)

theorem intersection_area :
  let rectangle_vertices := [(1, 7), (14, 7), (14, -4), (1, -4)] in 
  let circle_center := (1, -4) in 
  let circle_radius := 4 in 
  area_of_intersection circle_radius = 4 * π :=
by 
  sorry

end intersection_area_l621_621587


namespace sqrt_of_square_neg_five_eq_five_l621_621721

theorem sqrt_of_square_neg_five_eq_five :
  Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end sqrt_of_square_neg_five_eq_five_l621_621721


namespace monotone_increasing_interval_l621_621808

noncomputable def f (x : ℝ) := sqrt 2 * sin (x + π/4)

theorem monotone_increasing_interval :
  ∃ I : set ℝ, I = set.Icc 0 (π/4) ∧ monotone_on f I :=
sorry

end monotone_increasing_interval_l621_621808


namespace num_of_divisibles_l621_621859

theorem num_of_divisibles : 
  let is_divisible_by := λ (n d : ℕ), ∃ k, n = d * k in
  let count_multiples_in_range := λ (d low high : ℕ), 
    let start := Nat.ceil_div low d in
    let end_ := high / d in
    end_ - start + 1 in
  count_multiples_in_range 137 1000 9999 = 65 :=
by
  sorry

end num_of_divisibles_l621_621859


namespace increasing_if_derivative_positive_not_sufficient_cond_for_increasing_l621_621958

theorem increasing_if_derivative_positive {f : ℝ → ℝ} {a b : ℝ} (h : ∀ x ∈ set.Ioo a b, differentiable_at ℝ f x)
  (h' : ∀ x ∈ set.Ioo a b, 0 < deriv f x) : ∀ x y ∈ set.Ioo a b, x < y → f x < f y :=
sorry

theorem not_sufficient_cond_for_increasing {f : ℝ → ℝ} : 
  ∃ x : ℝ, (∀ y : ℝ, x < y → f x < f y) ∧ ¬ ∀ x, 0 < deriv f x :=
sorry

end increasing_if_derivative_positive_not_sufficient_cond_for_increasing_l621_621958


namespace max_extra_credit_students_l621_621450

theorem max_extra_credit_students (total_students : ℕ) (scores : list ℝ) (h_total : total_students = 150) 
(h_len : scores.length = total_students) :
  (∃ n, n = total_students - 1 ∧ ∀ x ∈ scores, x > (scores.sum / total_students)) :=
sorry

end max_extra_credit_students_l621_621450


namespace total_beads_correct_l621_621679

def purple_beads : ℕ := 7
def blue_beads : ℕ := 2 * purple_beads
def green_beads : ℕ := blue_beads + 11
def total_beads : ℕ := purple_beads + blue_beads + green_beads

theorem total_beads_correct : total_beads = 46 := 
by
  have h1 : purple_beads = 7 := rfl
  have h2 : blue_beads = 2 * 7 := rfl
  have h3 : green_beads = 14 + 11 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end total_beads_correct_l621_621679


namespace solve_for_x_l621_621114

theorem solve_for_x (x : Real) (h : (x - 5)^3 = (1 / 16)^(-1/2)) : x = 5 + Real.cbrt 4 := 
by 
  sorry

end solve_for_x_l621_621114


namespace find_value_of_k_l621_621928

theorem find_value_of_k (p : ℕ) (h_prime : nat.prime p) (h_odd : p % 2 = 1) (k : ℕ) 
  (h_sqrt : ∃ n : ℕ, n * n = k^2 - p * k) : 
  k = (1/4) * (p + 1)^2 := 
begin
  sorry
end

end find_value_of_k_l621_621928


namespace sqrt_16_eq_4_l621_621720

theorem sqrt_16_eq_4 : real.sqrt 16 = 4 :=
sorry

end sqrt_16_eq_4_l621_621720


namespace value_a_2016_l621_621359

def sequence_a : ℕ → ℚ
| 1     := 1
| 2     := 2
| (n+3) := sequence_a (n+2) / sequence_a (n+1)

theorem value_a_2016 : sequence_a 2016 = 1 / 2 :=
by sorry

end value_a_2016_l621_621359


namespace find_number_l621_621771

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l621_621771


namespace f_g_2_eq_36_l621_621871

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 4 * x - 2

theorem f_g_2_eq_36 : f (g 2) = 36 :=
by
  sorry

end f_g_2_eq_36_l621_621871


namespace correlation_coefficient_R2_l621_621179

theorem correlation_coefficient_R2:
  ∀(R2 : ℝ) (residuals : ℝ → ℝ),
    (0 ≤ R2 ∧ R2 ≤ 1) →
    (∀ε, 0 ≤ ε → ε < R2 → ∀ x, residuals x ≤ residuals (x * (1 - ε))) →
    (∀ x1 x2, x1 ≤ x2 → residuals x2 ≤ residuals x1) →
    (∀ x, residuals x = sum (λ y, (y - R2 * x) ^ 2)) →
  ( ∀ x2 x1, x1 < x2 → sum (λ y, (y - R2 * x2) ^ 2) < sum (λ y, (y - R2 * x1) ^ 2) ) :=
by
  intros R2 residuals hR2 hr1 hr2 hr3
  sorry

end correlation_coefficient_R2_l621_621179


namespace num_distinct_total_points_l621_621644

def total_points (x : ℤ) : ℤ := 3 * x + 2 * (7 - x)

theorem num_distinct_total_points : 
  {p : ℤ | ∃ x : ℤ, 0 ≤ x ∧ x ≤ 7 ∧ total_points x = p}.to_finset.card = 8 := 
by
  sorry

end num_distinct_total_points_l621_621644


namespace Emily_sixth_score_l621_621251

theorem Emily_sixth_score :
  let scores := [91, 94, 88, 90, 101]
  let current_sum := scores.sum
  let desired_average := 95
  let num_quizzes := 6
  let total_score_needed := num_quizzes * desired_average
  let sixth_score := total_score_needed - current_sum
  sixth_score = 106 :=
by
  sorry

end Emily_sixth_score_l621_621251


namespace avg_waiting_time_waiting_time_equivalence_l621_621709

-- The first rod receives an average of 3 bites in 6 minutes
def firstRodBites : ℝ := 3 / 6
-- The second rod receives an average of 2 bites in 6 minutes
def secondRodBites : ℝ := 2 / 6
-- Together, they receive an average of 5 bites in 6 minutes
def combinedBites : ℝ := firstRodBites + secondRodBites

-- We need to prove the average waiting time for the first bite
theorem avg_waiting_time : combinedBites = 5 / 6 → (1 / combinedBites) = 6 / 5 :=
by
  intro h
  rw h
  sorry

-- Convert 1.2 minutes into minutes and seconds
def minutes := 1
def seconds := 12

-- Prove the equivalence of waiting time in minutes and seconds
theorem waiting_time_equivalence : (6 / 5 = minutes + seconds / 60) :=
by
  simp [minutes, seconds]
  sorry

end avg_waiting_time_waiting_time_equivalence_l621_621709


namespace quadrilateral_angle_inequality_l621_621890

variables (A B C D A' B' C' D' : Type) [convex_quadrilateral A B C D] [convex_quadrilateral A' B' C' D']
variables (AB A'B' BC B'C' CD C'D' DA D'A' : ℝ)
variables (angleA angleA' angleB angleB' angleC angleC' angleD angleD' : ℝ)
variables (h_sides : AB = A'B' ∧ BC = B'C' ∧ CD = C'D' ∧ DA = D'A')
variables (h_angles : angleA > angleA')

theorem quadrilateral_angle_inequality :
  angleA > angleA' →
  (angleB < angleB') ∧ (angleC > angleC') ∧ (angleD < angleD') :=
sorry

end quadrilateral_angle_inequality_l621_621890


namespace equivalent_expression_l621_621351

theorem equivalent_expression (a : ℝ) (h1 : a ≠ -2) (h2 : a ≠ -1) :
  ( (a^2 + a - 2) / (a^2 + 3*a + 2) * 5 * (a + 1)^2 = 5*a^2 - 5 ) :=
by {
  sorry
}

end equivalent_expression_l621_621351


namespace linear_function_properties_l621_621400

theorem linear_function_properties :
  (∀ (k b : ℝ), 
    (∀ (x1 : ℝ), 
      (∀ (y1 : ℝ), 
        (y1 = k * x1 + b → 
          (x1 = 1 → y1 = 0) ∧ 
          (x1 = 0 → y1 = 2))) ∧
        k ≠ 0)) →
  let y := λ x : ℝ, -2 * x + 2 in
  ((∀ x : ℝ, -2 < x ∧ x ≤ 3 → -4 ≤ y x ∧ y x < 6) ∧ 
   ∀ m n : ℝ, (n = -2 * m + 2) ∧ (m - n = 4) → (m = 2 ∧ n = -2)) :=
by
  intros k b h_kb y
  sorry

end linear_function_properties_l621_621400


namespace polynomial_divisibility_l621_621783

-- Define the statement of the theorem
theorem polynomial_divisibility (P : ℤ[X])
    (h : ∀ a b c : ℤ, a^2 + b^2 - c^2 ∣ (P.eval a + P.eval b - P.eval c)) :
  ∃ (d : ℤ), ∀ x : ℤ, P.eval x = d * x^2 :=
begin
  sorry
end

end polynomial_divisibility_l621_621783


namespace octagon_area_l621_621340

theorem octagon_area (BDEF_square : square) (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 2) :
  octagon_area BDEF_square AB BC = 16 + 16 * Real.sqrt 2 := by
  sorry

end octagon_area_l621_621340


namespace sum_divisible_by_pq_l621_621917

variables (p n k q : ℕ) (M : Finset ℕ)
-- p is a prime number
axiom is_prime_p : Nat.Prime p
-- M is the set of all integers from 0 to n for which m - k is divisible by p
axiom M_def : M = {m : ℕ | m ≤ n ∧ (m - k) % p = 0}.to_finset
-- q ≤ (n - 1) / (p - 1)
axiom q_le_condition : q ≤ (n - 1) / (p - 1)

theorem sum_divisible_by_pq : p^q ∣ ∑ m in M, (-1)^m * (n.choose m) := 
sorry

end sum_divisible_by_pq_l621_621917


namespace fifteen_percent_of_x_is_ninety_l621_621768

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l621_621768


namespace length_of_hypotenuse_is_three_over_two_l621_621889

noncomputable def isosceles_right_triangle_hypotenuse_length : Prop :=
  ∃ (ABC : Type) (BC : ℝ) (D E : Point) (x : ℝ),
  is_isosceles_right_triangle ABC ∧
  divides_in_three_equal_parts BC D E ∧
  AD = tan x ∧
  AE = cot x ∧
  0 < x ∧ x < π / 2 ∧
  BC = 3 / 2

theorem length_of_hypotenuse_is_three_over_two :
  isosceles_right_triangle_hypotenuse_length :=
sorry

end length_of_hypotenuse_is_three_over_two_l621_621889


namespace coloring_5x5_no_two_corners_l621_621156

theorem coloring_5x5_no_two_corners :
  ∀ (color_without_corner : ℕ) (total_ways : ℕ),
    total_ways = 120 →
    color_without_corner = 96 →
    ∃ (color_without_two_corners : ℕ), color_without_two_corners = 78 := 
by
  intros color_without_corner total_ways h_total h_without_corner
  use 78
  sorry

end coloring_5x5_no_two_corners_l621_621156


namespace find_x_if_opposites_l621_621316

theorem find_x_if_opposites (x : ℝ) (h : 2 * (x - 3) = - 4 * (1 - x)) : x = -1 := 
by
  sorry

end find_x_if_opposites_l621_621316


namespace smallest_angle_correct_l621_621930

noncomputable def smallest_angle (R r : ℝ) (h : 0 < r ∧ r < R ∧ 1 < 2 * R) : ℝ :=
if h1 : r < R - 1 then
  0
else if h2 : r >= R - 1 ∧ r < R - 1 / R then
  real.arccos ((R^2 + r^2 - 1) / (2 * R * r))
else
  2 * real.arcsin (1 / (2 * R))
  
theorem smallest_angle_correct (R r : ℝ) (h : 0 < r ∧ r < R ∧ 1 < 2 * R) :
  (∃ A B : ℝ × ℝ, dist A B = 1 ∧ 
    ((circle_center A = (0, 0) ∧ (euclidean_distance (0, 0) A ≤ R) ∧ (euclidean_distance (0, 0) A ≥ r)) ∧ 
    (circle_center B = (0, 0) ∧ (euclidean_distance (0, 0) B ≤ R) ∧ (euclidean_distance (0, 0) B ≥ r)))) →
    let theta := smallest_angle R r h in 
    θ = if h1 : r < R - 1 then 0 
        else if h2 : r >= R - 1 ∧ r < R - 1 / R then real.arccos ((R^2 + r^2 - 1) / (2 * R * r)) 
        else 2 * real.arcsin (1 / (2 * R)) :=
sorry

end smallest_angle_correct_l621_621930


namespace paper_fold_height_l621_621657

theorem paper_fold_height :
  let initial_thickness := 0.1
      folds := 20
      height_per_floor := 3
      thickness := initial_thickness * 2 ^ folds
      height_in_meters := thickness / 1000
      num_floors := height_in_meters / height_per_floor
  in int.ofReal (num_floors) = 35 :=
by
  let initial_thickness := 0.1
  let folds := 20
  let height_per_floor := 3
  let thickness := initial_thickness * 2 ^ folds
  let height_in_meters := thickness / 1000
  let num_floors := height_in_meters / height_per_floor
  exact_mod_cast sorry

end paper_fold_height_l621_621657


namespace max_marks_equals_l621_621475

/-
  Pradeep has to obtain 45% of the total marks to pass.
  He got 250 marks and failed by 50 marks.
  Prove that the maximum marks is 667.
-/

-- Define the passing percentage
def passing_percentage : ℝ := 0.45

-- Define Pradeep's marks and the marks he failed by
def pradeep_marks : ℝ := 250
def failed_by : ℝ := 50

-- Passing marks is the sum of Pradeep's marks and the marks he failed by
def passing_marks : ℝ := pradeep_marks + failed_by

-- Prove that the maximum marks M is 667
theorem max_marks_equals : ∃ M : ℝ, passing_percentage * M = passing_marks ∧ M = 667 :=
sorry

end max_marks_equals_l621_621475


namespace b_range_l621_621843

def f (x b : ℝ) : ℝ := Real.exp x * (x - b)

theorem b_range (b : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (1/2 : ℝ) 2 ∧ f x b + x * (fun x => Real.exp x * (x - b) + Real.exp x) x > 0) →
  b < 8 / 3 :=
by
  sorry

end b_range_l621_621843


namespace intersection_A_B_l621_621443

def A : Set ℝ := {x | 2^(x-2) < 1}
def B : Set ℝ := {x | 1 - x ≥ 0}

theorem intersection_A_B : A ∩ B = {x | x ≤ 1} := by
  sorry

end intersection_A_B_l621_621443


namespace smallest_n_ge_2015_l621_621068

noncomputable theory

open Nat

def f (x : ℕ) : ℤ := sorry

-- Define the conditions
axiom f_condition_1 : f 1 = 0
axiom f_condition_2 : ∀ p : ℕ, Prime p → f p = 1
axiom f_condition_3 : ∀ x y : ℕ, 0 < x → 0 < y → f (x * y) = y * f x + x * f y

-- Main theorem statement
theorem smallest_n_ge_2015 : ∃ n : ℕ, n ≥ 2015 ∧ f n = n ∧ n = 3125 :=
by
  -- Proof goes here, but we're not required to provide it.
  sorry

end smallest_n_ge_2015_l621_621068


namespace triangle_inequality_l621_621806

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) :=
sorry

end triangle_inequality_l621_621806


namespace num_two_digit_numbers_with_swapped_digits_diff_nine_l621_621307

theorem num_two_digit_numbers_with_swapped_digits_diff_nine :
  ∃! n : ℕ, n = 8 ∧ (∀ x, (10 ≤ x ∧ x < 100) → (let (a, b) := (x / 10, x % 10) in (a < b) ∧ (10 * b + a = x + 9)) → ∃! k, x = 10 * (k / 10) + (k % 10) ∧ (swap k = 10 * (k % 10) + (k / 10) + 9))
:= sorry

end num_two_digit_numbers_with_swapped_digits_diff_nine_l621_621307


namespace range_of_f_l621_621070

open Real

def f (x y : ℝ) : ℝ := sqrt((1 + x * y) / (1 + x^2)) + sqrt((1 - x * y) / (1 + y^2))

theorem range_of_f (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) : 
  1 ≤ f x y ∧ f x y ≤ 2 := 
sorry

end range_of_f_l621_621070


namespace wheel_revolutions_l621_621984

noncomputable def radius := 22.4
noncomputable def distance_covered := 351.99999999999994

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

def number_of_revolutions (D C : ℝ) : ℝ := D / C

theorem wheel_revolutions : number_of_revolutions distance_covered (circumference radius) = 2.5 := by
  sorry

end wheel_revolutions_l621_621984


namespace impossible_clear_initial_points_l621_621732

theorem impossible_clear_initial_points :
  let initial_points := {(0,0), (1,0), (0,1)} in
  let operation (occupied : set (ℕ × ℕ)) (p : ℕ × ℕ) : Prop :=
        ∃ (x y : ℕ), p = (x,y) ∧ (x,y) ∈ occupied ∧ (x+1,y) ∉ occupied ∧ (x,y+1) ∉ occupied in
  let weight (p : ℕ × ℕ) : ℝ := 2^(-((p.1:ℕ) + p.2)) in
  ∀ operations : (finset (ℕ × ℕ)) → (ℕ × ℕ) → finset (ℕ × ℕ),
  ∀ occupied : finset (ℕ × ℕ),
  ((0,0) ∈ occupied ∧ (1,0) ∈ occupied ∧ (0,1) ∈ occupied) →
  (∀ p ∈ occupied, 
    (operation occupied p → (occupied → finset (ℕ × ℕ)))) →
  (finset.sum (occupied \ {(0,0), (1,0), (0,1)}) weight) = finset.sum initial_points weight →
  False := sorry

end impossible_clear_initial_points_l621_621732


namespace equilateral_triangle_incenter_half_inradius_l621_621040

theorem equilateral_triangle_incenter_half_inradius
  (A B C A' B' C' : Type)
  [Triangle ABC] [Triangle A'B'C']
  (I_ABC : Incenter ABC) (I_A'B'C' : Incenter A'B'C')
  (r_ABC r_A'B'C' : ℝ)
  (h1 : A' ∈ segment B C)
  (h2 : B' ∈ segment C A)
  (h3 : C' ∈ segment A B)
  (h4 : I_ABC = I_A'B'C')
  (h5 : r_A'B'C' = r_ABC / 2) :
  IsEquilateral ABC := 
sorry

end equilateral_triangle_incenter_half_inradius_l621_621040


namespace sarah_total_desserts_l621_621088

def michael_saves_cookies : ℕ := 5
def sarah_initial_cupcakes : ℕ := 9
def sarah_gives_fraction : ℝ := 1 / 3
def sarah_receives_cookies_from_michael : ℕ := michael_saves_cookies
def sarah_keeps_cupcakes : ℕ := (sarah_initial_cupcakes : ℝ * (1 - sarah_gives_fraction)).toNat

theorem sarah_total_desserts :
  sarah_receives_cookies_from_michael + sarah_keeps_cupcakes = 11 :=
by
  -- Proof goes here
  sorry

end sarah_total_desserts_l621_621088


namespace time_required_painting_rooms_l621_621676

-- Definitions based on the conditions
def alice_rate := 1 / 4
def bob_rate := 1 / 6
def charlie_rate := 1 / 8
def combined_rate := 13 / 24
def required_time : ℚ := 74 / 13

-- Proof problem statement
theorem time_required_painting_rooms (t : ℚ) :
  (combined_rate) * (t - 2) = 2 ↔ t = required_time :=
by
  sorry

end time_required_painting_rooms_l621_621676


namespace domain_of_v_l621_621594

noncomputable def v (x : ℝ) : ℝ := 1 / (x^2 + real.sqrt x)

theorem domain_of_v : ∀ x : ℝ, (v x).domain = (set.Ioi 0) := by
  sorry

end domain_of_v_l621_621594


namespace number_of_cats_l621_621606

theorem number_of_cats (c d : ℕ) (h1 : c = 20 + d) (h2 : c + d = 60) : c = 40 :=
sorry

end number_of_cats_l621_621606


namespace simplify_fraction_l621_621538

theorem simplify_fraction (m : ℤ) : (∃ c d : ℤ, c = 1 ∧ d = 2 ∧ (\frac{6 * m + 12}{6} = c * m + d)) ∧ (∃ c d : ℤ, c = 1 ∧ d = 2 ∧ (c / d = 1 / 2)) :=
by
  sorry

end simplify_fraction_l621_621538


namespace ceil_neg_sqrt_eq_neg_two_l621_621286

noncomputable def x : ℝ := -Real.sqrt (64 / 9)

theorem ceil_neg_sqrt_eq_neg_two : Real.ceil x = -2 := by
  exact sorry

end ceil_neg_sqrt_eq_neg_two_l621_621286


namespace average_bounds_l621_621570

noncomputable def average (a b c : ℝ) : ℝ :=
  (a + b + c) / 3

theorem average_bounds {N : ℝ}
  (hN1 : 12 < N)
  (hN2 : N < 25) :
  (average 9 15 N = 15) ∨ (average 9 15 N = 17) :=
by {
  sorry, -- Proof to be filled in
}

end average_bounds_l621_621570


namespace select_five_books_l621_621003

theorem select_five_books (total_books : ℕ) (required_book_included : ℕ) (ways_to_select : ℕ) :
  total_books = 8 →
  required_book_included = 1 →
  ways_to_select = 35 →
  (∃ ways : ℕ, ways = nat.choose (total_books - 1) 4 ∧ ways = ways_to_select) :=
by
  intros h_total h_required h_ways
  use nat.choose (total_books - 1) 4
  split
  {
    rw h_total
    norm_num
  }
  {
    rw h_ways
  }
  sorry

end select_five_books_l621_621003


namespace ceil_neg_sqrt_eq_neg_two_l621_621281

noncomputable def x : ℝ := -Real.sqrt (64 / 9)

theorem ceil_neg_sqrt_eq_neg_two : Real.ceil x = -2 := by
  exact sorry

end ceil_neg_sqrt_eq_neg_two_l621_621281


namespace problem_XA_XB_XC_sum_l621_621162

noncomputable def XA_XB_XC_sum (XA XB XC : ℝ) : ℝ := XA + XB + XC

theorem problem_XA_XB_XC_sum:
  (AB BC AC : ℝ) (D E F : Point) (X : Point)
  (hAB : AB = 15) (hBC : BC = 16) (hAC : AC = 17)
  (hD : is_midpoint D AB) (hE : is_midpoint E BC) (hF : is_midpoint F AC)
  (hX_circ : X ≠ E ∧ (is_circumcenter_of BE DE) ∧ (is_circumcenter_of CE FE)) :
  XA_XB_XC_sum (XA X) (XB X) (XC X) = (960 * (real.sqrt 39) / 14) :=
begin
  sorry
end

end problem_XA_XB_XC_sum_l621_621162


namespace Elizabeth_More_Revenue_Than_Banks_l621_621448

theorem Elizabeth_More_Revenue_Than_Banks : 
  let banks_investments := 8
  let banks_revenue_per_investment := 500
  let elizabeth_investments := 5
  let elizabeth_revenue_per_investment := 900
  let banks_total_revenue := banks_investments * banks_revenue_per_investment
  let elizabeth_total_revenue := elizabeth_investments * elizabeth_revenue_per_investment
  elizabeth_total_revenue - banks_total_revenue = 500 :=
by
  sorry

end Elizabeth_More_Revenue_Than_Banks_l621_621448


namespace polynomial_collinearity_l621_621297

theorem polynomial_collinearity
  (P : ℝ → ℝ)
  (h : ∀ x y z : ℝ, x + y + z = 0 → collinear ℝ (λ t, (t, P t)) {x, y, z}) :
  ∃ a b c : ℝ, ∀ x : ℝ, P x = a * x^3 + b * x + c :=
begin
  sorry
end

end polynomial_collinearity_l621_621297


namespace sum_final_two_numbers_l621_621151

theorem sum_final_two_numbers (S : ℤ) : 
  let a := (S - 1) / 2 in
  3 * (a + 5) + 3 * (a + 6) = 3 * S + 30 :=
by
  -- start of proof here
  sorry

end sum_final_two_numbers_l621_621151


namespace ham_block_cut_mass_distribution_l621_621193

theorem ham_block_cut_mass_distribution
  (length width height : ℝ) (mass : ℝ)
  (parallelogram_side1 parallelogram_side2 : ℝ)
  (condition1 : length = 12) 
  (condition2 : width = 12) 
  (condition3 : height = 35)
  (condition4 : mass = 5)
  (condition5 : parallelogram_side1 = 15) 
  (condition6 : parallelogram_side2 = 20) :
  ∃ (mass_piece1 mass_piece2 : ℝ),
    mass_piece1 = 1.7857 ∧ mass_piece2 = 3.2143 :=
by
  sorry

end ham_block_cut_mass_distribution_l621_621193


namespace stuart_initial_marbles_is_56_l621_621220

-- Define the initial conditions
def betty_initial_marbles : ℕ := 60
def percentage_given_to_stuart : ℚ := 40 / 100
def stuart_marbles_after_receiving : ℕ := 80

-- Define the calculation of how many marbles Betty gave to Stuart
def marbles_given_to_stuart := (percentage_given_to_stuart * betty_initial_marbles)

-- Define the target: Stuart's initial number of marbles
def stuart_initial_marbles := stuart_marbles_after_receiving - marbles_given_to_stuart

-- Main theorem stating the problem
theorem stuart_initial_marbles_is_56 : stuart_initial_marbles = 56 :=
by 
  sorry

end stuart_initial_marbles_is_56_l621_621220


namespace ellipse_eccentricity_l621_621836

theorem ellipse_eccentricity 
  (a b c: ℝ) 
  (h : a > b ∧ b > 0) 
  (h_eq : c^2 = a^2 - b^2) 
  (h_intersect: (2 * c) ^ 2 / b ^ 2 + c ^ 2 / a ^ 2 = 1) : 
  (c / a) = sqrt 2 - 1 :=
sorry

end ellipse_eccentricity_l621_621836


namespace zeckendorf_theorem_l621_621611

def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fib (n + 1) + fib n

theorem zeckendorf_theorem (N : ℕ) : ∃! (k : ℕ) (a : ℕ → ℕ), 
  (N = (∑ i in finset.range k.succ, a i * fib i)) ∧ 
  (∀ i < k, a i = 0 ∨ a i = 1) ∧ 
  (∀ i < k - 1, a i = 1 → a (i + 1) = 0) :=
sorry

end zeckendorf_theorem_l621_621611


namespace coeff_monomial_l621_621964

theorem coeff_monomial (x y : ℝ) : ∃ (c : ℝ), -2 * x^3 * y = c * x^3 * y ∧ c = -2 :=
by {
  use -2,
  split,
  { ring, },
  { refl, }
}

end coeff_monomial_l621_621964


namespace initial_men_in_fort_l621_621201

theorem initial_men_in_fort (M : ℕ) 
  (h1 : ∀ N : ℕ, M * 35 = (N - 25) * 42) 
  (h2 : 10 + 42 = 52) : M = 150 :=
sorry

end initial_men_in_fort_l621_621201


namespace cyclic_quadrilateral_area_l621_621379
noncomputable def sqrt {α : Type*} [linear_ordered_field α] (x : α) : α := sorry

theorem cyclic_quadrilateral_area (a b c d : ℝ) (A B C D : ℝ)
  (hA : A = real.pi - C)
  (hB : B = real.pi - D)
  (hSides : a + c = b + d) :
  let T := sqrt (a * b * c * d)
  in T = sqrt (a * b * c * d) :=
by
  sorry

end cyclic_quadrilateral_area_l621_621379


namespace summer_camp_days_l621_621668

theorem summer_camp_days (P : Finset ℕ) (D : ℕ) 
  (hP : P.card = 15)
  (hD : ∀ p1 p2 : ℕ, p1 ≠ p2 → 
    ∃! d : ℕ, d ∈ Finset.range D ∧ ∃ p3 p4 p5 : ℕ, {p3, p4, p5} ⊆ P ∧ {p1, p2} ⊆ {p3, p4, p5}
  ) :
  D = 35 :=
by
  sorry

end summer_camp_days_l621_621668


namespace product_of_roots_l621_621429

theorem product_of_roots :
  (p q r s : ℂ) (h : polynomial.eval₂ (ring_hom.id ℂ) p 3 * p ^ 4 
                                        - 8 * p ^ 3 
                                        - 15 * p ^ 2 
                                        + 10 * p 
                                        - 2 = 0 ∧
       polynomial.eval₂ (ring_hom.id ℂ) q 3 * q ^ 4 
                                        - 8 * q ^ 3 
                                        - 15 * q ^ 2 
                                        + 10 * q 
                                        - 2 = 0 ∧
       polynomial.eval₂ (ring_hom.id ℂ) r 3 * r ^ 4 
                                        - 8 * r ^ 3 
                                        - 15 * r ^ 2 
                                        + 10 * r 
                                        - 2 = 0 ∧
       polynomial.eval₂ (ring_hom.id ℂ) s 3 * s ^ 4 
                                        - 8 * s ^ 3 
                                        - 15 * s ^ 2 
                                        + 10 * s 
                                        - 2 = 0) :
  p * q * r * s = 2 / 3 := sorry

end product_of_roots_l621_621429


namespace equalize_costs_l621_621417

variables {G H M : ℝ}

def LeRoy_payment (G H M : ℝ) : ℝ :=
  (2 / 3) * G + (1 / 4) * H + (1 / 3) * M

def Bernardo_payment (G H M : ℝ) : ℝ :=
  (1 / 3) * G + (3 / 4) * H + (2 / 3) * M

def total_expenses (G H M : ℝ) : ℝ :=
  G + H + M

def each_person_pay (G H M : ℝ) : ℝ :=
  total_expenses G H M / 2

def payment_difference (G H M : ℝ) :=
  each_person_pay G H M - LeRoy_payment G H M

theorem equalize_costs
  : payment_difference G H M = (- (1 / 3) * G + (1 / 2) * H - (1 / 3) * M) / 2 :=
sorry

end equalize_costs_l621_621417


namespace rectangle_is_square_l621_621913

-- Definitions and conditions
def Rectangle (A1 A2 A3 A4 : Type) : Type := sorry
def Circle (S : Type) : Type := sorry
def Tangent (S1 S2 : Type) (A : Type) : Type := sorry

-- Rectangle and circles properties
variables (A1 A2 A3 A4 : Type) (S1 S2 S3 S4 : Type)
variable [Rectangle A1 A2 A3 A4]
variable [Circle S1] [Circle S2] [Circle S3] [Circle S4]

-- Tangency conditions
variables [Tangent S1 S2 A1] [Tangent S2 S3 A2] [Tangent S3 S4 A3] [Tangent S4 S1 A4]
variables [Tangent S1 S2 S3] [Tangent S2 S3 S4] [Tangent S3 S4 S1]

theorem rectangle_is_square : is_square A1 A2 A3 A4 := sorry

end rectangle_is_square_l621_621913


namespace molly_age_l621_621152

theorem molly_age
  (avg_age : ℕ)
  (hakimi_age : ℕ)
  (jared_age : ℕ)
  (molly_age : ℕ)
  (h1 : avg_age = 40)
  (h2 : hakimi_age = 40)
  (h3 : jared_age = hakimi_age + 10)
  (h4 : 3 * avg_age = hakimi_age + jared_age + molly_age) :
  molly_age = 30 :=
by
  sorry

end molly_age_l621_621152


namespace avg_waiting_time_waiting_time_equivalence_l621_621712

-- The first rod receives an average of 3 bites in 6 minutes
def firstRodBites : ℝ := 3 / 6
-- The second rod receives an average of 2 bites in 6 minutes
def secondRodBites : ℝ := 2 / 6
-- Together, they receive an average of 5 bites in 6 minutes
def combinedBites : ℝ := firstRodBites + secondRodBites

-- We need to prove the average waiting time for the first bite
theorem avg_waiting_time : combinedBites = 5 / 6 → (1 / combinedBites) = 6 / 5 :=
by
  intro h
  rw h
  sorry

-- Convert 1.2 minutes into minutes and seconds
def minutes := 1
def seconds := 12

-- Prove the equivalence of waiting time in minutes and seconds
theorem waiting_time_equivalence : (6 / 5 = minutes + seconds / 60) :=
by
  simp [minutes, seconds]
  sorry

end avg_waiting_time_waiting_time_equivalence_l621_621712


namespace rearrange_three_people_out_of_eight_l621_621580

theorem rearrange_three_people_out_of_eight :
  (nat.choose 8 3) * 2 = (nat.choose 8 3) * 2 :=
by
  sorry

end rearrange_three_people_out_of_eight_l621_621580


namespace intersection_range_k_second_quadrant_l621_621362

theorem intersection_range_k_second_quadrant (k : ℝ) :
  (∃ (x y : ℝ), k * x - y = k - 1 ∧ k * y - x = 2k ∧ x < 0 ∧ y > 0) ↔ (0 < k ∧ k < 1/2) :=
by
  sorry

end intersection_range_k_second_quadrant_l621_621362


namespace fraction_given_to_cousin_l621_621722

theorem fraction_given_to_cousin
  (initial_candies : ℕ)
  (brother_share sister_share : ℕ)
  (eaten_candies left_candies : ℕ)
  (remaining_candies : ℕ)
  (given_to_cousin : ℕ)
  (fraction : ℚ)
  (h1 : initial_candies = 50)
  (h2 : brother_share = 5)
  (h3 : sister_share = 5)
  (h4 : eaten_candies = 12)
  (h5 : left_candies = 18)
  (h6 : initial_candies - brother_share - sister_share = remaining_candies)
  (h7 : remaining_candies - given_to_cousin - eaten_candies = left_candies)
  (h8 : fraction = (given_to_cousin : ℚ) / (remaining_candies : ℚ))
  : fraction = 1 / 4 := 
sorry

end fraction_given_to_cousin_l621_621722


namespace k_greater_than_half_l621_621104

-- Definition of the problem conditions
variables {a b c k : ℝ}

-- Assume a, b, c are the sides of a triangle
axiom triangle_inequality : a + b > c

-- Given condition
axiom sides_condition : a^2 + b^2 = k * c^2

-- The theorem to prove k > 0.5
theorem k_greater_than_half (h1 : a + b > c) (h2 : a^2 + b^2 = k * c^2) : k > 0.5 :=
by
  sorry

end k_greater_than_half_l621_621104


namespace largest_subset_l621_621051

def is_valid_subset (T : set ℕ) : Prop :=
  ∀ x y ∈ T, x ≠ y → |x - y| ≠ 5 ∧ |x - y| ≠ 8

theorem largest_subset (T : set ℕ) (hT : T ⊆ (set.Icc 1 2023))
  (h_valid: is_valid_subset T) : T.card = 935 :=
sorry

end largest_subset_l621_621051


namespace pencils_multiple_of_10_l621_621569

theorem pencils_multiple_of_10 (pens : ℕ) (students : ℕ) (pencils : ℕ) 
  (h_pens : pens = 1230) 
  (h_students : students = 10) 
  (h_max_distribute : ∀ s, s ≤ students → (∃ pens_per_student, pens = pens_per_student * s ∧ ∃ pencils_per_student, pencils = pencils_per_student * s)) :
  ∃ n, pencils = 10 * n :=
by
  sorry

end pencils_multiple_of_10_l621_621569


namespace tournament_player_count_l621_621888

theorem tournament_player_count (n : ℕ) :
  (∃ points_per_game : ℕ, points_per_game = (n * (n - 1)) / 2) →
  (∃ T : ℕ, T = 90) →
  (n * (n - 1)) / 4 = 90 →
  n = 19 :=
by
  intros h1 h2 h3
  sorry

end tournament_player_count_l621_621888


namespace race_participants_minimum_l621_621527

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l621_621527


namespace find_angle_XZY_l621_621688

open Set

namespace Geometry

-- Definitions for angles in the equilateral triangle and regular pentagon and the measure of angle at vertex XZY.
def interior_angle_pentagon := (108 : ℝ)
def interior_angle_equilateral_triangle := (60 : ℝ)
def shared_vertex := (X : Point)
def vertex_Y := (Y : Point)
def vertex_Z := (Z : Point)

def circle (P : Type*) := {c : set P // ∀ x ∈ c, ∃ r : ℝ, dist x c.center = r} -- Circle definition

def isosceles_triangle (A B C : Point) : Prop :=
  dist A B = dist A C

def measure_angle (A B C : Point) : ℝ := sorry -- Function to measure angles in terms of points (requires geometric definitions)

theorem find_angle_XZY :
  ∀ (X Y Z : Point), 
  (interior_angle_pentagon = 108) →
  (interior_angle_equilateral_triangle = 60) →
  (isosceles_triangle X Y Z) →
  measure_angle X Z Y = 12 :=
begin
  sorry -- proof is deferred
end

end Geometry

end find_angle_XZY_l621_621688


namespace smallest_number_of_students_l621_621093

/-- 
On June 1, a group of students is standing in rows, with every row containing 8 students. 
The same group rearranges itself over the successive days, changing the number of students 
per row each day. This continues until June 15, when they find that no new arranging option 
is available that hadn't been used in the preceding days. The smallest possible number of students 
in the group is a multiple of 8 and has exactly 15 divisors.
-/
theorem smallest_number_of_students : ∃ n : ℕ, n % 8 = 0 ∧ (Nat.divisors n).length = 15 ∧ n = 720 :=
by
  sorry

end smallest_number_of_students_l621_621093


namespace multiple_of_6_cases_l621_621148

theorem multiple_of_6_cases :
  let A_values := {0, 2, 4, 6, 8}
  let B_values := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  A_values × B_values → 
  (A ∈ A_values) →
  (∀ A B, 
    7 + 3 * A + 2 * B ≡ 0 [MOD 3]) →
   ∃! n, n = 15 := sorry

end multiple_of_6_cases_l621_621148


namespace tetrahedron_faces_equal_l621_621618

theorem tetrahedron_faces_equal {a b c a' b' c' : ℝ} (h₁ : a + b + c = a + b' + c') (h₂ : a + b + c = a' + b + b') (h₃ : a + b + c = c' + c + a') :
  (a = a') ∧ (b = b') ∧ (c = c') :=
by
  sorry

end tetrahedron_faces_equal_l621_621618


namespace irrational_sqrt_5_l621_621178

theorem irrational_sqrt_5 : 
  ∃ x ∈ ({real.sqrt 5, real.sqrt 4, 22 / 7, 1.414} : set ℝ), irrational x ∧ x = real.sqrt 5 :=
by
  sorry

end irrational_sqrt_5_l621_621178


namespace complementary_events_B_l621_621884

-- Definitions of events based on the problem's conditions
def white_balls : Finset ℕ := {1, 2, 3}
def black_balls : Finset ℕ := {4, 5, 6, 7}
def bag : Finset ℕ := white_balls ∪ black_balls
def draws : Finset (Finset ℕ) := bag.powerset.filter (λ s, s.card = 3)

def at_least_one_white_ball (s : Finset ℕ) : Prop := ∃ b ∈ s, b ∈ white_balls
def all_black_balls (s : Finset ℕ) : Prop := ∀ b ∈ s, b ∈ black_balls

-- Lean 4 statement to prove the pair are complementary events
theorem complementary_events_B : 
  (∀ s ∈ draws, at_least_one_white_ball s → ¬all_black_balls s) ∧
  (∀ s ∈ draws, (at_least_one_white_ball s ∨ all_black_balls s)) :=
by
  sorry

end complementary_events_B_l621_621884


namespace driver_net_rate_of_pay_l621_621199

def net_rate_of_pay 
  (travel_time : ℕ)        -- 3 hours
  (speed : ℕ)              -- 50 miles per hour
  (fuel_efficiency : ℕ)    -- 25 miles per gallon
  (pay_rate : ℝ)           -- $0.60 per mile
  (gasoline_cost : ℝ) : ℝ  -- $2.50 per gallon
  :=
  (pay_rate * (speed * travel_time)
   - gasoline_cost * ((speed * travel_time) / fuel_efficiency))
  / travel_time

theorem driver_net_rate_of_pay 
  (travel_time : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_rate : ℝ)
  (gasoline_cost : ℝ) :
  travel_time = 3 →
  speed = 50 →
  fuel_efficiency = 25 →
  pay_rate = 0.60 →
  gasoline_cost = 2.50 →
  net_rate_of_pay travel_time speed fuel_efficiency pay_rate gasoline_cost = 25 := 
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end driver_net_rate_of_pay_l621_621199


namespace julie_aaron_age_l621_621415

variables {J A m : ℕ}

theorem julie_aaron_age : (J = 4 * A) → (J + 10 = m * (A + 10)) → (m = 4) :=
by
  intros h1 h2
  sorry

end julie_aaron_age_l621_621415


namespace time_to_cover_escalator_l621_621689

def escalator_speed := 11 -- ft/sec
def escalator_length := 126 -- feet
def person_speed := 3 -- ft/sec

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 9 := by
  sorry

end time_to_cover_escalator_l621_621689


namespace find_y_coordinate_l621_621652

theorem find_y_coordinate (y : ℝ) (h : y > 0) (dist_eq : (10 - 2)^2 + (y - 5)^2 = 13^2) : y = 16 :=
by
  sorry

end find_y_coordinate_l621_621652


namespace linear_function_form_quadratic_function_form_l621_621190

-- Problem (1)
theorem linear_function_form (f : ℝ → ℝ) (h₁ : ∀ x, f[f(x)] = 4 * x + 3) (h₂ : ∃ k b, ∀ x, f(x) = k * x + b ∧ k ≠ 0) :
    (f = λ x, 2 * x + 1) ∨ (f = λ x, -2 * x - 3) :=
sorry

-- Problem (2)
theorem quadratic_function_form (f : ℝ → ℝ) (h₁ : f 0 = 2) (h₂ : ∀ x, f(x + 1) - f(x) = 2*x - 1) (h₃ : ∃ a b c, ∀ x, f(x) = a * x^2 + b * x + c ∧ a ≠ 0) :
    f = λ x, x^2 - 2 * x + 2 :=
sorry

end linear_function_form_quadratic_function_form_l621_621190


namespace arithmetic_sequence_20th_term_l621_621730

theorem arithmetic_sequence_20th_term (a d n : ℕ) (h1 : a = 2) (h2 : d = 5) (h3 : n = 20) :
  a + (n - 1) * d = 97 :=
by
  rw [h1, h2, h3]
  exact (by norm_num : 2 + (20 - 1) * 5 = 97)

end arithmetic_sequence_20th_term_l621_621730


namespace product_of_possible_sums_l621_621020

/-- 
Each vertex of a cube is labeled with either \( +1 \) or \( -1 \).
Each face of the cube has a value equal to the product of the values at its vertices.
Sum the values of the 8 vertices and the 6 faces.
Determine all possible values for this sum.
Find the product of these possible sums.
-/
theorem product_of_possible_sums : 
  let possible_sums := [14, 10, 6, 2, -2, -6, -10]
  in possible_sums.prod = -20160 := 
by sorry

end product_of_possible_sums_l621_621020


namespace find_slope_of_l_l621_621617

-- Definitions
def region_D (x y : ℝ) : Prop :=
  x + y > 0 ∧ x - y < 0

def locus_C (x y : ℝ) : Prop :=
  y^2 - x^2 = 4

def point_F := (2 * Real.sqrt 2, 0 : ℝ × ℝ)

-- Problem Statement
theorem find_slope_of_l (x y : ℝ) (k : ℝ) :
  (region_D x y) →
  (locus_C x y) →
  (∀ (x1 x2 y1 y2 : ℝ), 
    let A : ℝ × ℝ := (x1, y1);
    let B : ℝ × ℝ := (x2, y2);
    A ≠ B →
    (A.1 + B.1) / 2 ≠ 0 →
    x1 / x2 = y1 / y2 →
    (locus_C x1 y1) →
    (locus_C x2 y2) →
    x1 ≠ x2 →
    y1 ≠ y2 →
    ((x1 + x2) / 2, (y1 + y2) / 2) = point_F →
    (x1 + x2) = (4 * Real.sqrt 2 * k^2) / (k^2 - 1)) →
  k = -Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end find_slope_of_l_l621_621617


namespace ceil_of_neg_sqrt_frac_64_over_9_l621_621291

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l621_621291


namespace find_number_l621_621762

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l621_621762


namespace neg_P_l621_621850

/-
Proposition: There exists a natural number n such that 2^n > 1000.
-/
def P : Prop := ∃ n : ℕ, 2^n > 1000

/-
Theorem: The negation of the above proposition P is:
For all natural numbers n, 2^n ≤ 1000.
-/
theorem neg_P : ¬ P ↔ ∀ n : ℕ, 2^n ≤ 1000 :=
by
  sorry

end neg_P_l621_621850


namespace ellipse_equation_l621_621820

-- Defining the conditions
variable {C : Type} (hC1 : ∀ x y, x^2 / 4 + y^2 / 3 = 1)
variable (center_origin : C → Prop) (right_focus : C → Prop) (eccentricity : C → Prop)

def center_at_origin (E : C) : Prop := center_origin E

def focus_at_F (E : C) : Prop := right_focus E

def eccentricity_of_E (E : C) : Prop := eccentricity E = 1 / 2

-- The statement we want to prove
theorem ellipse_equation :
  ∃ C, center_at_origin C ∧ focus_at_F C ∧ eccentricity_of_E C →
  (∀ x y, x^2 / 4 + y^2 / 3 = 1) :=
by
  sorry

end ellipse_equation_l621_621820


namespace remainder_of_91_pow_92_mod_100_l621_621147

theorem remainder_of_91_pow_92_mod_100 : (91 ^ 92) % 100 = 81 :=
by
  sorry

end remainder_of_91_pow_92_mod_100_l621_621147


namespace solve_omega_l621_621839

noncomputable def ω (f : ℝ → ℝ) (m n : ℝ) : ℝ :=
  if ∃ ω: ℝ, ∃ m n: ℝ, f(x) = (cos (ω * x + (π / 3))), f(m) = 1/2 ∧ f(n) = 1/2 ∧ m ≠ n ∧ abs (m - n) = π / 6
  then ω else 0

theorem solve_omega : ∃ ω: ℝ, ω = 4 ∨ ω = -4 :=
by
  sorry

end solve_omega_l621_621839


namespace find_y_value_l621_621614

theorem find_y_value : (12^3 * 6^3 / 432) = 864 := by
  sorry

end find_y_value_l621_621614


namespace tan_quadruple_angle_l621_621867

theorem tan_quadruple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (4 * θ) = -24 / 7 :=
sorry

end tan_quadruple_angle_l621_621867


namespace committee_count_l621_621801

theorem committee_count (students teachers : ℕ) (committee_size : ℕ) 
  (h_students : students = 11) (h_teachers : teachers = 3) 
  (h_committee_size : committee_size = 8) : 
  ∑ (k : ℕ) in finset.range committee_size.succ, (nat.choose (students + teachers) committee_size) - (nat.choose students committee_size) = 2838 := 
by 
  sorry

end committee_count_l621_621801


namespace domain_of_g_max_value_of_g_l621_621809

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

def g (x : ℝ) : ℝ := (f x)^2 + f (x^2)

theorem domain_of_g :
  let Df := Icc 1 9 -- Domain of f(x)
  let Dg := {x | x ∈ Df ∧ x^2 ∈ Df} -- Domain of g(x)
  Dg = Icc 1 3 := sorry

theorem max_value_of_g :
  ∃ x ∈ Icc 1 3, g x = 13 ∧ ∀ y ∈ Icc 1 3, g y ≤ 13 := sorry

end domain_of_g_max_value_of_g_l621_621809


namespace ceil_neg_sqrt_fraction_l621_621277

theorem ceil_neg_sqrt_fraction :
  (⌈-real.sqrt (64 / 9)⌉ = -2) :=
by
  -- Define the necessary conditions
  have h1 : real.sqrt (64 / 9) = 8 / 3 := by sorry,
  have h2 : -real.sqrt (64 / 9) = -8 / 3 := by sorry,
  -- Apply the ceiling function and prove the result
  exact sorry

end ceil_neg_sqrt_fraction_l621_621277


namespace jane_daffodil_bulbs_l621_621906

theorem jane_daffodil_bulbs :
  ∃ (D : ℕ), (0.5 * (20 + (20 / 2) + D + 3 * D) = 75) ∧ D = 30 :=
by
  use 30
  sorry

end jane_daffodil_bulbs_l621_621906


namespace ceil_neg_sqrt_fraction_l621_621278

theorem ceil_neg_sqrt_fraction :
  (⌈-real.sqrt (64 / 9)⌉ = -2) :=
by
  -- Define the necessary conditions
  have h1 : real.sqrt (64 / 9) = 8 / 3 := by sorry,
  have h2 : -real.sqrt (64 / 9) = -8 / 3 := by sorry,
  -- Apply the ceiling function and prove the result
  exact sorry

end ceil_neg_sqrt_fraction_l621_621278


namespace race_minimum_participants_l621_621519

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l621_621519


namespace number_of_knights_is_five_l621_621465

section KnightsAndKnaves

inductive Inhabitant
| knight : Inhabitant
| knave : Inhabitant

open Inhabitant

variables (a1 a2 a3 a4 a5 a6 a7 : Inhabitant)

def tells_truth : Inhabitant → Prop
| knight := True
| knave := False

def statements : Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → ℕ → Prop
| a1 a2 _ _ _ _ _ 1 := (a1 = knight)
| a1 a2 _ _ _ _ _ 2 := (a1 = knight ∧ a2 = knight)
| a1 a2 a3 _ _ _ _ 3 := (a1 = knave ∨ a2 = knave)
| a1 a2 a3 a4 _ _ _ 4 := (a1 = knave ∨ a2 = knave ∨ a3 = knave)
| a1 a2 a3 a4 a5 _ _ 5 := (a1 = knight ∧ a2 = knight ∨ a1 = knave ∧ a2 = knave)
| a1 a2 a3 a4 a5 a6 _ 6 := (a1 = knave ∨ a2 = knave ∨ a3 = knave ∨ a4 = knave ∨ a5 = knave)
| a1 a2 a3 a4 a5 a6 a7 7 := (a1 = knight ∨ a2 = knight ∨ a3 = knight ∨ a4 = knight ∨ a5 = knight ∨ a6 = knight)

theorem number_of_knights_is_five (h1 : tells_truth a1 ↔ a1 = knight)
                                   (h2 : tells_truth a2 ↔ a2 = knight)
                                   (h3 : tells_truth a3 ↔ a3 = knight)
                                   (h4 : tells_truth a4 ↔ a4 = knight)
                                   (h5 : tells_truth a5 ↔ a5 = knight)
                                   (h6 : tells_truth a6 ↔ a6 = knight)
                                   (h7 : tells_truth a7 ↔ a7 = knight)
                                   (s1 : tells_truth a1 → statements a1 a2 a3 a4 a5 a6 a7 1)
                                   (s2 : tells_truth a2 → statements a1 a2 a3 a4 a5 a6 a7 2)
                                   (s3 : tells_truth a3 → statements a1 a2 a3 a4 a5 a6 a7 3)
                                   (s4 : tells_truth a4 → statements a1 a2 a3 a4 a5 a6 a7 4)
                                   (s5 : tells_truth a5 → statements a1 a2 a3 a4 a5 a6 a7 5)
                                   (s6 : tells_truth a6 → statements a1 a2 a3 a4 a5 a6 a7 6)
                                   (s7 : tells_truth a7 → statements a1 a2 a3 a4 a5 a6 a7 7)
                                   : (∀ i, (i = knight → tells_truth i ¬ ↔ i = knight)) → ∃ (n : ℕ), n = 5 := 
sorry

end KnightsAndKnaves

end number_of_knights_is_five_l621_621465


namespace girls_sums_equal_iff_n_odd_l621_621155

theorem girls_sums_equal_iff_n_odd (n : ℕ) (h : n ≥ 3) :
  (∀ i j : ℕ, i ≠ j → i < n → j < n → let gi := n + 1 + i in let gj := n + 1 + j in gi + (2*i + 1) + (2*i + 2) = gj + (2*j - 1) + (2*j) ) ↔ (n % 2 = 1) :=
sorry

end girls_sums_equal_iff_n_odd_l621_621155


namespace valid_5_digit_numbers_divisible_by_7_l621_621428

theorem valid_5_digit_numbers_divisible_by_7 :
  let count_valid_n := finset.filter (λ n, 
    let q := n / 50 
    let r := n % 50 
    (q + r) % 7 = 0) 
      (finset.Icc 10000 99999) in
  count_valid_n.card = 14400 :=
begin
  sorry
end

end valid_5_digit_numbers_divisible_by_7_l621_621428


namespace avg_waiting_time_is_1_point_2_minutes_l621_621702

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l621_621702


namespace cindy_correct_answer_l621_621723

theorem cindy_correct_answer (x : ℕ) (h : (x - 12) / 4 = 28) : (x - 5) / 8 = 14.875 :=
by sorry

end cindy_correct_answer_l621_621723


namespace find_number_l621_621763

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l621_621763


namespace avg_waiting_time_is_1_point_2_minutes_l621_621701

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l621_621701


namespace arithmetic_sequence_problem_l621_621395

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (d : ℚ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + 1 / 2 * a 7 + a 10 = 10) : a 3 + a 11 = 8 :=
sorry

end arithmetic_sequence_problem_l621_621395


namespace peach_count_l621_621995

theorem peach_count (n : ℕ) : n % 4 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ 120 ≤ n ∧ n ≤ 150 → n = 142 :=
sorry

end peach_count_l621_621995


namespace option_c_incorrect_l621_621177

theorem option_c_incorrect (a : ℝ) : a + a^2 ≠ a^3 :=
sorry

end option_c_incorrect_l621_621177


namespace find_m_of_quadratic_fn_l621_621137

theorem find_m_of_quadratic_fn (m : ℚ) (h : 2 * m - 1 = 2) : m = 3 / 2 :=
by
  sorry

end find_m_of_quadratic_fn_l621_621137


namespace permutations_with_P_more_l621_621206

def has_property_P (n : ℕ) (l : List ℕ) : Prop :=
  ∃ i, i < l.length - 1 ∧ (l.nthLe i sorry - l.nthLe (i + 1) sorry).natAbs = n

theorem permutations_with_P_more {n : ℕ} (l : List ℕ) (h_length : l.length = 2 * n) :
  ∃ (A B : Finset (List ℕ)),
  (∀ x ∈ A, has_property_P n x) ∧ (∀ x ∈ B, ¬has_property_P n x) ∧
  A.card > B.card :=
sorry

end permutations_with_P_more_l621_621206


namespace inequality_solution_l621_621986

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (x + 2) ≤ 3 ↔ x ∈ Set.Iic (-6) ∪ Set.Ioi (-2) :=
by
  sorry

end inequality_solution_l621_621986


namespace minimum_value_of_MA_plus_MF_l621_621027

open EuclideanGeometry

noncomputable theory

def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

def circle (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 1)^2 = 1

def focus_of_parabola : ℝ × ℝ := (1, 0)

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem minimum_value_of_MA_plus_MF :
  ∃ (M A : ℝ × ℝ), parabola M.1 M.2 ∧ circle A.1 A.2 ∧ 
  ∀ (M' A' : ℝ × ℝ), 
    parabola M'.1 M'.2 → circle A'.1 A'.2 → 
    dist M' A' + dist M' focus_of_parabola ≥ 4 :=
sorry

end minimum_value_of_MA_plus_MF_l621_621027


namespace P_2018_has_2018_distinct_real_roots_l621_621632

-- Define the sequence of polynomials P_n(x) with given initial conditions.
noncomputable def P : ℕ → Polynomial ℝ
| 0       := 1
| 1       := Polynomial.X
| (n + 1) := Polynomial.X * P n - P (n - 1)

-- The statement to prove that P_2018(x) has exactly 2018 distinct real roots.
theorem P_2018_has_2018_distinct_real_roots :
  ∃ (roots : Fin 2018 → ℝ), ∀ i j : Fin 2018, i ≠ j → roots i ≠ roots j ∧ ∀ x : ℝ, (Polynomial.aeval x (P 2018)) = 0 ↔ x ∈ Set.range roots := sorry

end P_2018_has_2018_distinct_real_roots_l621_621632


namespace minimum_participants_l621_621490

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l621_621490


namespace molecular_weight_CaBr2_l621_621170

theorem molecular_weight_CaBr2 : 
  let atomic_weight_Ca := 40.08
  let atomic_weight_Br := 79.904
  let molecular_weight := atomic_weight_Ca + 2 * atomic_weight_Br
  in molecular_weight = 199.888 := 
by 
  let atomic_weight_Ca := 40.08
  let atomic_weight_Br := 79.904
  let molecular_weight := atomic_weight_Ca + 2 * atomic_weight_Br
  show molecular_weight = 199.888 by sorry

end molecular_weight_CaBr2_l621_621170


namespace find_BX_squared_l621_621420

-- Define the geometric setup
noncomputable def right_triangle (A B C : Point) : Prop :=
  ∠ACB = 90

noncomputable def midpoint (M : Point) (A C : Point) : Prop :=
  M = ((A.x + C.x) / 2, (A.y + C.y) / 2)

noncomputable def angle_bisector (CN : Line) (A B C : Point) : Prop :=
  CN ∈ bisectors ∠ACB

noncomputable def intersection (X : Point) (BM CN : Line) : Prop :=
  X ∈ (BM ∩ CN)

noncomputable def equilateral_triangle (BXN : Triangle) : Prop :=
  BXN.isEquilateral ∧ BXN.side_length = 1

noncomputable def segment_length (A B : Point) : ℝ :=
  sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2)

-- Define the problem to prove
theorem find_BX_squared (A B C M N X : Point) (CN BM : Line) :
  right_triangle A B C → 
  midpoint M A C → 
  angle_bisector CN A B C → 
  intersection X BM CN → 
  equilateral_triangle (triangle B X N) → 
  segment_length A C = 4 → 
  (segment_length B X) ^ 2 = 6 - 4 * sqrt 2 := 
sorry

end find_BX_squared_l621_621420


namespace arithmetic_sequence_a17_l621_621330

theorem arithmetic_sequence_a17 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : S 13 = 78)
  (h2 : a 7 + a 12 = 10)
  (h_sum : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 1 + (a 2 - a 1) / (2 - 1)))
  (h_term : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1) / (2 - 1)) :
  a 17 = 2 :=
by
  sorry

end arithmetic_sequence_a17_l621_621330


namespace min_number_of_participants_l621_621508

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l621_621508


namespace calculate_expression_l621_621235

theorem calculate_expression (y : ℝ) : (20 * y^3) * (7 * y^2) * (1 / (2 * y)^3) = 17.5 * y^2 :=
by
  sorry

end calculate_expression_l621_621235


namespace collinear_B_H_Q_l621_621042

open EuclideanGeometry

variables (A B C H I P K Q : Point)
variables [orthocenter A B C H]
variables [incenter A B C I]
variables (circumcircle_BCI : Circle) (H_projected_on_AI : Line) (reflected_point : Point)
variables [circumcircle_of B C I circumcircle_BCI]
variables [segment_intersection A B circumcircle_BCI P]
variables [different P B]
variables [projection_of H on_line (line_through A I) K]
variables [reflection P in K Q]

theorem collinear_B_H_Q 
  (h1 : is_orthocenter A B C H)
  (h2 : is_incenter A B C I)
  (h3 : circle_containing_points circumcircle_BCI B C I)
  (h4 : segment_intersects_circle [A, B] circumcircle_BCI P)
  (h5 : P ≠ B)
  (h6 : projection_of H (line_through A I) K)
  (h7 : Q = reflection_of P in_point K) :
  collinear [B, H, Q] :=
sorry

end collinear_B_H_Q_l621_621042


namespace rectangular_prism_diagonal_inequality_l621_621077

theorem rectangular_prism_diagonal_inequality 
  (a b c l : ℝ) 
  (h : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
by sorry

end rectangular_prism_diagonal_inequality_l621_621077


namespace distance_AG_l621_621012

-- Definitions and conditions
def quadrilateral (E F G H : ℝ × ℝ) : Prop :=
  (E.1 = 0 ∧ E.2 = 0) ∧ (F.1 = a ∧ F.2 = 0) ∧
  (G.1 = a ∧ G.2 = b) ∧ (H.1 = 0 ∧ H.2 = b)

def right_prism_height (A E : ℝ × ℝ) : ℝ := 
  32 

-- The main theorem
theorem distance_AG (E F G H A : ℝ × ℝ) (a b : ℝ)  (h1 : quadrilateral E F G H) (h2 : right_prism_height A E = 32) : 
  dist A G = 40 :=
sorry

end distance_AG_l621_621012


namespace required_sixth_quiz_score_l621_621253

theorem required_sixth_quiz_score :
  let scores := [91, 94, 88, 90, 101] in
  let sum_scores := scores.sum in
  let num_quizzes := 6 in
  let desired_mean := 95 in
  let required_total := desired_mean * num_quizzes in
  let sixth_score := required_total - sum_scores in
  sixth_score = 106 :=
by
  sorry

end required_sixth_quiz_score_l621_621253


namespace find_angle_A_find_side_a_l621_621024

-- Definitions based on the given conditions
variables {A B C : ℝ}  -- angles in the triangle
variables {a b c : ℝ}  -- lengths of the sides opposite to angles A, B, C

-- Conditions as given in the original problem
def condition1 : Prop := (a + b + c) * (b + c - a) = 3 * b * c
def condition2 : Prop := (b - c = 1)
def condition3 : Prop := ∀ (u v : ℝ), u * v = -1  -- This needs specific vectors; simplified for translation
def angle_A := A

-- The first problem
theorem find_angle_A (h1 : condition1) : angle_A = π / 3 :=
sorry

-- The second problem
theorem find_side_a (h3 : condition3) (h2 : condition2) : a = sqrt 3 :=
sorry

end find_angle_A_find_side_a_l621_621024


namespace ceil_neg_sqrt_eq_neg_two_l621_621282

noncomputable def x : ℝ := -Real.sqrt (64 / 9)

theorem ceil_neg_sqrt_eq_neg_two : Real.ceil x = -2 := by
  exact sorry

end ceil_neg_sqrt_eq_neg_two_l621_621282


namespace room_volume_correct_l621_621001

variable (Length Width Height : ℕ) (Volume : ℕ)

-- Define the dimensions of the room
def roomLength := 100
def roomWidth := 10
def roomHeight := 10

-- Define the volume function
def roomVolume (l w h : ℕ) : ℕ := l * w * h

-- Theorem to prove the volume of the room
theorem room_volume_correct : roomVolume roomLength roomWidth roomHeight = 10000 := 
by
  -- roomVolume 100 10 10 = 10000
  sorry

end room_volume_correct_l621_621001


namespace doubling_time_population_l621_621124

theorem doubling_time_population (b d e i : ℝ) (h_b : b = 39.4) (h_d : d = 19.4) (h_e : e = 3.2) (h_i : i = 5.6) :
  let net_birth_rate := b - d,
      net_migration_rate := i - e,
      total_growth_rate := net_birth_rate + net_migration_rate,
      annual_growth_percentage := total_growth_rate / 10 in
  x = 70 / annual_growth_percentage → x ≈ 31.25 :=
by
  sorry

end doubling_time_population_l621_621124


namespace tickets_sold_total_l621_621999

-- Define the conditions
variables (A : ℕ) (S : ℕ) (total_amount : ℝ := 222.50) (adult_ticket_price : ℝ := 4) (student_ticket_price : ℝ := 2.50)
variables (student_tickets_sold : ℕ := 9)

-- Define the total money equation and the question
theorem tickets_sold_total :
  4 * (A : ℝ) + 2.5 * (9 : ℝ) = 222.50 → A + 9 = 59 :=
by sorry

end tickets_sold_total_l621_621999


namespace fifteen_percent_of_x_is_ninety_l621_621755

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l621_621755


namespace sum_of_ages_is_37_l621_621935

def maries_age : ℕ := 12
def marcos_age (M : ℕ) : ℕ := 2 * M + 1

theorem sum_of_ages_is_37 : maries_age + marcos_age maries_age = 37 := 
by
  -- Inserting the proof details
  sorry

end sum_of_ages_is_37_l621_621935


namespace determine_sets_l621_621188

open Set

noncomputable def sets_satisfy_conditions (A B C : Set ℕ) : Prop :=
  A ⊆ { n : ℕ | ∃k, n = 3 * k - 2 }
  ∧ B ⊆ { n : ℕ | ∃k, n = 3 * k - 1 }
  ∧ C ⊆ { n : ℕ | ∃k, n = 3 * k }
  ∧ (∀ a ∈ A, a + { n : ℕ | ∃k, n = 3 * k } ⊆ A)
  ∧ (∀ b ∈ B, b + { n : ℕ | ∃k, n = 3 * k } ⊆ B)
  ∧ (∀ (a ∈ A) (b ∈ B), a + b ∈ C)

theorem determine_sets (A B C : Set ℤ) :
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧
  A ∪ B ∪ C = univ ∧
  sets_satisfy_conditions A B C :=
sorry

end determine_sets_l621_621188


namespace area_triangle_AEC_l621_621074

-- Define the points and properties of the rhombus
variables (A B C D E F : Type) [Point : AffineSpace ℝ _]
variables [is_rhombus : Rhombus Point A B C D] 
variables [midpoint_E : Midpoint Point E A B]
variables [midpoint_F : Midpoint Point F C D]
variables [diagonal_AC : Diagonal Point A C]

-- Given the area of the rhombus
def area_rhombus : ℝ := 50

-- The problem statement to prove the area of triangle AEC
theorem area_triangle_AEC :
  area (Triangle A E C) = 12.5 :=
sorry

end area_triangle_AEC_l621_621074


namespace ceil_neg_sqrt_64_over_9_l621_621273

theorem ceil_neg_sqrt_64_over_9 : Real.ceil (-Real.sqrt (64 / 9)) = -2 := 
by
  sorry

end ceil_neg_sqrt_64_over_9_l621_621273


namespace find_m_l621_621000

open Real

noncomputable def triangle_angles (ABC : Triangle) (circ : Circle) (B C D A : Point) : Prop :=
  inscribed_in_circle ABC circ ∧
  is_acute ABC ∧
  AB = AC ∧
  tangent_at B circ ∧
  tangent_at C circ ∧
  meet_at D tangent_b tangent_c ∧
  ∠ABC = ∠ACB ∧
  ∠ABC = 3 * ∠D ∧
  ∠BAC = m * π

theorem find_m (ABC : Triangle) (circ : Circle) (B C D A : Point) :
  triangle_angles ABC circ B C D A →
  m = 5 / 11 :=
by
  sorry

end find_m_l621_621000


namespace correct_statements_l621_621406

variables {A B C D A1 B1 C1 D1 M N M1 N1 : Point}

-- Define the key relationships and objects in the problem:
axiom points_on_lines : M ∈ line A B1 ∧ N ∈ line B C1
axiom points_not_coincide : M ≠ B1 ∧ N ≠ C1
axiom distance_equal : dist A M = dist B N

-- The statements we want to prove:
theorem correct_statements :
  (perpendicular (line A A1) (line M N)) ∧
  (parallel (line M N) (plane A1 B1 C1 D1)) :=
sorry

end correct_statements_l621_621406


namespace find_number_l621_621769

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l621_621769


namespace oleksiy_can_guess_more_than_half_winners_l621_621004

theorem oleksiy_can_guess_more_than_half_winners (n : ℕ) (h : n ≥ 2) :
  ∃ k, k > n * (n - 1) / 4 ∧ ∀ tournament_points schedule,
  (∀ i j, i ≠ j → (tournament_points i = tournament_points j + 1 ∨ tournament_points i + 1 = tournament_points j)) →
  (∀ i j, i ≠ j → (schedule i j = 0 ∨ schedule i j = 1)) →
  (∀ i j, i ≠ j → ∃ winner, schedule i j = winner) →
  ∀ guesses,
  (∀ i j, guesses i j = schedule i j ∨ guesses i j ≠ schedule i j) →
  let correct_guesses := ∑ i j, if i ≠ j then if guesses i j = schedule i j then 1 else 0 else 0 in
  correct_guesses > n * (n - 1) / 4 :=
begin
  sorry
end

end oleksiy_can_guess_more_than_half_winners_l621_621004


namespace triangle_area_l621_621974

theorem triangle_area : 
  ∀ (a b c : ℝ),
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) →
  b = 2 * a →
  2 * b = a + c →
  let S := (1 / 2) * | - (c / (a + 2 * b)) | * | - (c / b) | in
  S = (9 / 20) :=
by
  intros a b c ha hb hc h1 h2
  let b := 2 * a
  let c := 3 * a
  sorry

end triangle_area_l621_621974


namespace prob_xi_eq_2_l621_621329

noncomputable def ξ : ℕ → ℝ
| 1 := 1/6
| 2 := 1/3
| 3 := 1/2
| _ := 0

theorem prob_xi_eq_2 :
  (ξ 2) = 1 / 3 := by
  sorry

end prob_xi_eq_2_l621_621329


namespace find_angle_B_l621_621902

theorem find_angle_B (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h1 : c * cos B + b * cos C = a * sin A)
  (h2 : S = (sqrt 3 / 4) * (b^2 + a^2 - c^2)) : 
  B = 30 :=
sorry

end find_angle_B_l621_621902


namespace knights_on_island_l621_621461

-- Definitions based on conditions
inductive Inhabitant : Type
| knight : Inhabitant
| knave : Inhabitant

open Inhabitant

def statement_1 (inhabitant : Inhabitant) : Prop :=
inhabitant = knight

def statement_2 (inhabitant1 inhabitant2 : Inhabitant) : Prop :=
inhabitant1 = knight ∧ inhabitant2 = knight

def statement_3 (inhabitant1 inhabitant2 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave)) / 2 ≥ 0.5

def statement_4 (inhabitant1 inhabitant2 inhabitant3 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave) + ↑(inhabitant3 = knave)) / 3 ≥ 0.65

def statement_5 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 : Inhabitant) : Prop :=
(↑(inhabitant1 = knight) + ↑(inhabitant2 = knight) + ↑(inhabitant3 = knight) + ↑(inhabitant4 = knight)) / 4 ≥ 0.5

def statement_6 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 inhabitant5 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave) + ↑(inhabitant3 = knave) + ↑(inhabitant4 = knave) + ↑(inhabitant5 = knave)) / 5 ≥ 0.4

def statement_7 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 inhabitant5 inhabitant6 : Inhabitant) : Prop :=
(↑(inhabitant1 = knight) + ↑(inhabitant2 = knight) + ↑(inhabitant3 = knight) + ↑(inhabitant4 = knight) + ↑(inhabitant5 = knight) + ↑(inhabitant6 = knight)) / 6 ≥ 0.65

-- Lean Statement
theorem knights_on_island (inhabitants : Fin 7 → Inhabitant) :
  (∀ i, (inhabitants i = knight ↔ (i = 0) ∨ (i = 1) ∨ (i = 4) ∨ (i = 5) ∨ (i = 6))) → 5 :=
by
  sorry

end knights_on_island_l621_621461


namespace problem_statement_l621_621560

theorem problem_statement (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ) 
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) : 
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := 
sorry

end problem_statement_l621_621560


namespace avg_waiting_time_waiting_time_equivalence_l621_621713

-- The first rod receives an average of 3 bites in 6 minutes
def firstRodBites : ℝ := 3 / 6
-- The second rod receives an average of 2 bites in 6 minutes
def secondRodBites : ℝ := 2 / 6
-- Together, they receive an average of 5 bites in 6 minutes
def combinedBites : ℝ := firstRodBites + secondRodBites

-- We need to prove the average waiting time for the first bite
theorem avg_waiting_time : combinedBites = 5 / 6 → (1 / combinedBites) = 6 / 5 :=
by
  intro h
  rw h
  sorry

-- Convert 1.2 minutes into minutes and seconds
def minutes := 1
def seconds := 12

-- Prove the equivalence of waiting time in minutes and seconds
theorem waiting_time_equivalence : (6 / 5 = minutes + seconds / 60) :=
by
  simp [minutes, seconds]
  sorry

end avg_waiting_time_waiting_time_equivalence_l621_621713


namespace number_of_terms_expansion_l621_621849

theorem number_of_terms_expansion :
  let A := {a, b, c}
  let B := {d, e, f, h}
  let C := {i, j, k, l, m}
  |A| * |B| * |C| = 60 :=
by 
  let A := {a, b, c}
  let B := {d, e, f, h}
  let C := {i, j, k, l, m}
  have H1 : |A| = 3 := by sorry
  have H2 : |B| = 4 := by sorry
  have H3 : |C| = 5 := by sorry
  calc
    |A| * |B| * |C| = 3 * 4 * 5 : by rw [H1, H2, H3]
    ... = 60 : by norm_num

end number_of_terms_expansion_l621_621849


namespace distance_point_to_line_l621_621553

theorem distance_point_to_line :
  let P := (2 : ℝ, 5 : ℝ)
  let l := λ x : ℝ, - (Real.sqrt 3) * x
  let d := Real.dist P (λ x : ℝ, - (Real.sqrt 3) * x)
  d = (2 * Real.sqrt 3 + 5) / 2 :=
begin
  sorry
end

end distance_point_to_line_l621_621553


namespace integer_product_zero_l621_621564

theorem integer_product_zero (a : ℤ) (x : Fin 13 → ℤ)
  (h : a = ∏ i, (1 + x i) ∧ a = ∏ i, (1 - x i)) :
  a * ∏ i, x i = 0 :=
sorry

end integer_product_zero_l621_621564


namespace race_participants_minimum_l621_621512

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l621_621512


namespace race_participants_least_number_l621_621500

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l621_621500


namespace average_age_of_dance_club_l621_621963

theorem average_age_of_dance_club 
  (avg_age_females : ℕ) (num_females : ℕ) 
  (avg_age_males : ℕ) (num_males : ℕ)
  (oldest_male_age_diff : ℕ) :
  avg_age_females = 25 →
  num_females = 12 →
  avg_age_males = 40 →
  num_males = 18 →
  oldest_male_age_diff = 10 →
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 34 :=
begin
  intros h1 h2 h3 h4 h5,
  suffices h6: ((12 * 25 + 18 * 40) / (12 + 18) = 34), from h6,
  sorry,
end

end average_age_of_dance_club_l621_621963


namespace domain_of_log_function_l621_621786

noncomputable def domain_f (k : ℤ) : Set ℝ :=
  {x : ℝ | (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
           (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3)}

theorem domain_of_log_function :
  ∀ x : ℝ, (∃ k : ℤ, (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
                      (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3))
  ↔ (3 - 4 * Real.sin x ^ 2 > 0) :=
by {
  sorry
}

end domain_of_log_function_l621_621786


namespace octagon_area_l621_621339

theorem octagon_area (BDEF_square : square) (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 2) :
  octagon_area BDEF_square AB BC = 16 + 16 * Real.sqrt 2 := by
  sorry

end octagon_area_l621_621339


namespace bodyguard_hourly_rate_l621_621160

noncomputable def weekly_payment (hourly_rate : ℝ) : ℝ :=
  7 * (2 * hourly_rate * 8)

theorem bodyguard_hourly_rate 
  (total_payment : ℝ)
  (h1 : total_payment = 2240) :
  ∃ (hourly_rate : ℝ), weekly_payment hourly_rate = total_payment ∧ hourly_rate = 20 := 
by 
  use 20
  split
  · calc
      weekly_payment 20 = 7 * (2 * 20 * 8) : rfl
                      ... = 2240 : by norm_num
  · rfl

end bodyguard_hourly_rate_l621_621160


namespace find_A_l621_621376

theorem find_A (A : ℕ) (hA : A > 0) 
  (h_sum : (∑ n in finset.range A, 1 / ((n + 1) * (n + 3))) = 12/25) : 
  A = 22 :=
sorry

end find_A_l621_621376


namespace segment_CM_divides_area_in_ratio_l621_621941

def triangle_area_ratio (A B C M : Point) (k : ℝ) (h1 : distance A M = 2 * k)
                        (h2 : distance M B = 5 * k) (h3 : distance A B = 7 * k) : Prop :=
  let area_ACM := 1 / 2 * distance A M * height_from C A B in
  let area_BCM := 1 / 2 * distance M B * height_from C A B in
  area_ACM / area_BCM = 2 / 5

axiom height_from (C A B : Point) : ℝ

theorem segment_CM_divides_area_in_ratio (A B C M : Point) (k : ℝ)
    (h1 : distance A M = 2 * k)
    (h2 : distance M B = 5 * k)
    (h3 : distance A B = 7 * k) :
  triangle_area_ratio A B C M k h1 h2 h3 := 
sorry

end segment_CM_divides_area_in_ratio_l621_621941


namespace determine_a_and_theta_l621_621846

noncomputable def f (a θ : ℝ) (x : ℝ) : ℝ := 2 * a * Real.sin (2 * x + θ)

theorem determine_a_and_theta :
  (∃ a θ : ℝ, 0 < θ ∧ θ < π ∧ a ≠ 0 ∧ (∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f a θ x ∈ Set.Icc (-2 : ℝ) 2) ∧ 
  (∀ (x1 x2 : ℝ), x1 ∈ Set.Icc (-5 * π / 12) (π / 12) → x2 ∈ Set.Icc (-5 * π / 12) (π / 12) → x1 < x2 → f a θ x1 > f a θ x2)) →
  (a = -1) ∧ (θ = π / 3) :=
sorry

end determine_a_and_theta_l621_621846


namespace find_number_l621_621760

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l621_621760


namespace probability_sum_six_l621_621175

theorem probability_sum_six (A : Type) [fintype A] [decidable_eq A]
  (die_faces : fin 6) :
  let outcomes := { (a, b) : fin 6 × fin 6 | true }
  let favorable_outcomes := { (a, b) : fin 6 × fin 6 | a.val + b.val + 2 = 6 }
  fintype.card favorable_outcomes.to_finset = 5 →
  fintype.card outcomes.to_finset = 36 →
  (fintype.card favorable_outcomes.to_finset / fintype.card outcomes.to_finset : ℚ) = 5 / 36 :=
by
  intros
  sorry

end probability_sum_six_l621_621175


namespace triangle_angles_from_cube_diagonals_are_60_l621_621099

noncomputable def diagonal_angle_in_cube_triangle : ℝ :=
by
  -- Establish that the shape is a cube
  let cube := unit_cube
  -- Diagonals are drawn on three faces of the cube
  let face_diagonals := draw_diagonals_on_three_faces cube
  -- The triangle formed by the intersecting diagonals
  let triangle := form_triangle_from_face_diagonals face_diagonals
  -- Verify the angles of the equilateral triangle
  have triangle_is_equilateral : is_equilateral_triangle triangle
  exact triangle_is_equilateral
  -- The angles of an equilateral triangle
  let angle := 60
  -- Therefore, the angles of the triangle are each 60 degrees
  exact angle

theorem triangle_angles_from_cube_diagonals_are_60 (cube : unit_cube) :
  let face_diagonals := draw_diagonals_on_three_faces cube
  let triangle := form_triangle_from_face_diagonals face_diagonals
  let angle := 60
  is_equilateral_triangle triangle → triangle.angle = angle :=
by
  intros cube face_diagonals triangle angle triangle_is_equilateral
  exact triangle_is_equilateral

end triangle_angles_from_cube_diagonals_are_60_l621_621099


namespace problem_f_f_neg1_l621_621383

def f (x : ℝ) : ℝ :=
if x ≤ 0 then (x - 1) ^ 2 else x + 1 / x

theorem problem_f_f_neg1 :
  f (f (-1)) = 17 / 4 :=
sorry

end problem_f_f_neg1_l621_621383


namespace spherical_to_rectangular_correct_l621_621735

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_correct : spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) :=
by
  sorry

end spherical_to_rectangular_correct_l621_621735


namespace square_geometry_BF_sum_l621_621541

open Real

theorem square_geometry_BF_sum :
  ∀ (A B C D O G E F : Point) (AB_len : ℝ) (m ∠ EOF : ℝ),
    side_length A B = 1000 ∧
    center_of_square A B C D = O ∧
    midpoint A B = G ∧
    E ∈ segment A B ∧
    F ∈ segment A B ∧
    AE < BF ∧
    E ≠ F ∧
    m ∠ EOF = 30 ∧
    distance E F = 500 →
    ∃ p q r : ℕ, 
      (BF = p + q * sqrt r) ∧
      p > 0 ∧ q > 0 ∧ r > 0 ∧
      (∃ k : ℤ, r = (k:ℤ)^(1/2) ∧ gcf p q r ∣ 1) ∧ -- r must not be divisible by square of any prime.
      p + q + r = 504 := 
begin
  sorry
end

end square_geometry_BF_sum_l621_621541


namespace no_minimum_and_inf_S_eq_zero_l621_621731

open Function Real Interval

def A : Set (ℝ → ℝ) :=
  {f | ContDiffOn ℝ 1 f (Icc (-1 : ℝ) 1) ∧ f (-1) = -1 ∧ f 1 = 1}

def S (f : ℝ → ℝ) : ℝ :=
  ∫ x in -1..1, x^2 * (deriv f x) ^ 2

theorem no_minimum_and_inf_S_eq_zero :
  (¬ ∃ f ∈ A, (∀ g ∈ A, S f ≤ S g)) ∧ (inf {S f | f ∈ A} = 0) :=
by
  sorry

end no_minimum_and_inf_S_eq_zero_l621_621731


namespace minimum_participants_l621_621533

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l621_621533


namespace solution_set_of_inequality_l621_621075

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 else 2^x

theorem solution_set_of_inequality : 
  {x : ℝ | f x ≤ 2} = {x : ℝ | x ≤ -2 ∨ (0 < x ∧ x ≤ 1)} :=
by
  sorry

end solution_set_of_inequality_l621_621075


namespace max_elements_l621_621047

def T : Set ℕ := { x | 1 ≤ x ∧ x ≤ 2023 }

theorem max_elements (T : Set ℕ) (h₁ : ∀ (a b : ℕ), a ∈ T → b ∈ T → (a ≠ b → a ≠ b + 5 ∧ a ≠ b + 8)) :
  ∃ (n : ℕ), n = 780 ∧ ∀ (S : Set ℕ), (S ⊆ T) → (∀ (a b : ℕ), a ∈ S → b ∈ S → (a ≠ b → a ≠ b + 5 ∧ a ≠ b + 8)) → S.card ≤ 780 :=
sorry

end max_elements_l621_621047


namespace two_digit_count_l621_621609

theorem two_digit_count : 
  let digits := {0, 2, 4, 6, 8} 
  ∧ first_digit_choices := {2, 4, 6, 8} 
  ∧ (∀ a ∈ first_digit_choices, ∃ b ∈ digits, a ≠ b) 
  ∧ (∀ a b, a ∈ first_digit_choices → b ∈ digits → a ≠ b) 
  in card (first_digit_choices × (digits \ {0})) = 16 :=
by
  let digits := {0, 2, 4, 6, 8}
  let first_digit_choices := {2, 4, 6, 8}
  have : ∀ a ∈ first_digit_choices, ∃ b ∈ digits, a ≠ b,
    by sorry
  have : ∀ a b, a ∈ first_digit_choices → b ∈ digits → a ≠ b,
    by sorry
  let count := card (first_digit_choices × (digits \ {0}))
  have h : count = 16, by sorry
  exact h

end two_digit_count_l621_621609


namespace num_100_digit_integers_divisible_by_2_pow_100_l621_621642

open Nat

def F (n : ℕ) : Finset ℕ := 
  {x | x.digits 10 |>.length = n ∧ 
       ∀ d ∈ x.digits 10, d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 6 ∨ d = 7 ∧ 
       2^n ∣ x}

def f (n : ℕ) : ℕ := (F n).card

theorem num_100_digit_integers_divisible_by_2_pow_100 : f 100 = 3^100 :=
sorry

end num_100_digit_integers_divisible_by_2_pow_100_l621_621642


namespace minimum_value_of_function_y_l621_621977

def function_y (x : ℝ) : ℝ := real.exp x + 4 * real.exp (-x)

theorem minimum_value_of_function_y :
  ∃ x : ℝ, function_y x = 4 :=
sorry

end minimum_value_of_function_y_l621_621977


namespace domain_of_f_2x_minus_1_l621_621830

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = x) →
  ∀ x, (1 / 2) ≤ x ∧ x ≤ (3 / 2) → ∃ y, f y = (2 * x - 1) :=
by
  intros h x hx
  sorry

end domain_of_f_2x_minus_1_l621_621830


namespace avg_waiting_time_waiting_time_equivalence_l621_621710

-- The first rod receives an average of 3 bites in 6 minutes
def firstRodBites : ℝ := 3 / 6
-- The second rod receives an average of 2 bites in 6 minutes
def secondRodBites : ℝ := 2 / 6
-- Together, they receive an average of 5 bites in 6 minutes
def combinedBites : ℝ := firstRodBites + secondRodBites

-- We need to prove the average waiting time for the first bite
theorem avg_waiting_time : combinedBites = 5 / 6 → (1 / combinedBites) = 6 / 5 :=
by
  intro h
  rw h
  sorry

-- Convert 1.2 minutes into minutes and seconds
def minutes := 1
def seconds := 12

-- Prove the equivalence of waiting time in minutes and seconds
theorem waiting_time_equivalence : (6 / 5 = minutes + seconds / 60) :=
by
  simp [minutes, seconds]
  sorry

end avg_waiting_time_waiting_time_equivalence_l621_621710


namespace pascal_triangle_row_10_sum_l621_621883

theorem pascal_triangle_row_10_sum : ∑ i in Finset.range (10 + 1), Nat.choose 10 i = 1024 := by
  sorry

end pascal_triangle_row_10_sum_l621_621883


namespace minimum_participants_l621_621489

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l621_621489


namespace sqrt_product_simplification_l621_621949

theorem sqrt_product_simplification : sqrt (5 * 3) * sqrt (3^3 * 5^4) = 225 * sqrt 5 := by
  sorry

end sqrt_product_simplification_l621_621949


namespace summation_cos_squared_l621_621724

noncomputable def S : ℝ :=
  ∑ k in finset.range 91, (real.cos (k * 2) * real.cos (k * 2))

theorem summation_cos_squared :
  S = 91 / 2 := by
  sorry

end summation_cos_squared_l621_621724


namespace domain_f_monotonic_f_l621_621353

noncomputable def f (m x : ℝ) : ℝ := log 2 (m + (m - 1) / (x - 1))

theorem domain_f (m : ℝ) (hm : m > 0) :
  (0 < m ∧ m < 1 → ∀ x, x ∈ (-∞, 1) ∪ (1/m, ∞) → 0 < m + (m - 1) / (x - 1)) ∧ 
  (m = 1 → ∀ x, x ∈ (-∞, 1) ∪ (1, ∞) → 0 < m + (m - 1) / (x - 1)) ∧ 
  (m > 1 → ∀ x, x ∈ (-∞, 1/m) ∪ (1, ∞) → 0 < m + (m - 1) / (x - 1)) := sorry

theorem monotonic_f (m : ℝ) (hm : m > 0) (hinc : ∀ x, x ∈ (4, ∞) → (m + (m - 1) / (x - 1)) > 0) :
  1/4 ≤ m ∧ m < 1 := sorry

end domain_f_monotonic_f_l621_621353


namespace differentiate_log_evaluate_integral_l621_621041

variable (a b x : ℝ) (e : ℝ)
variable (h_e_pos : 0 < e) (h_e_ne_neg_inv : e ≠ e⁻¹)

noncomputable def log_derivative (x a : ℝ) : ℝ :=
  1 / real.sqrt (x^2 + a)

theorem differentiate_log (a : ℝ) (x : ℝ) (h_x_pos : 0 < x) :
  deriv (λ x, real.log (x + real.sqrt (x^2 + a))) x = log_derivative x a :=
by
  sorry

noncomputable def integral_result (a b : ℝ) : ℝ :=
  real.log ( (e - e⁻¹) * (real.sqrt (4 / (e - e⁻¹)^2 + 1) + 1) / 2 )

theorem evaluate_integral (b : ℝ) :
  a = 4 * b^2 / (e - e⁻¹)^2 → 
  ∫ x in 0..b, 1 / real.sqrt (x^2 + a) = integral_result a b :=
by
  intro h
  sorry

end differentiate_log_evaluate_integral_l621_621041


namespace perpendicular_lines_k_value_l621_621308

theorem perpendicular_lines_k_value (k : ℝ) : 
  k * (k - 1) + (1 - k) * (2 * k + 3) = 0 ↔ k = -3 ∨ k = 1 :=
by
  sorry

end perpendicular_lines_k_value_l621_621308


namespace unique_solution_c_exceeds_s_l621_621660

-- Problem Conditions
def steers_cost : ℕ := 35
def cows_cost : ℕ := 40
def total_budget : ℕ := 1200

-- Definition of the solution conditions
def valid_purchase (s c : ℕ) : Prop := 
  steers_cost * s + cows_cost * c = total_budget ∧ s > 0 ∧ c > 0

-- Statement to prove
theorem unique_solution_c_exceeds_s :
  ∃ s c : ℕ, valid_purchase s c ∧ c > s ∧ ∀ (s' c' : ℕ), valid_purchase s' c' → s' = 8 ∧ c' = 17 :=
sorry

end unique_solution_c_exceeds_s_l621_621660


namespace standard_equation_of_curve_C_non_exist_line_l_l621_621823

variable {x0 y0 x y t : ℝ}

/-- Condition 1: Definition of points A and B. -/
def point_A (x0 : ℝ) := (x0, 0)
def point_B (y0 : ℝ) := (0, y0)

/-- Condition 2: Distance between points A and B is 1. -/
def distance_AB (x0 y0 : ℝ) := (x0^2 + y0^2 = 1)

/-- Condition 3: Point P satisfies given vector equation. -/
def point_P (x y x0 y0 : ℝ) := (x = 2 * x0) ∧ (y = sqrt 3 * y0)

/-- Part 1: Prove that the standard equation of curve C is given by this ellipse equation. -/
theorem standard_equation_of_curve_C (h₁ : distance_AB x0 y0) (h₂ : point_P x y x0 y0) : 
  (x^2 / 4 + y^2 / 3 = 1) :=
sorry

/-- Part 2: Prove that there does not exist a line l such that the area of triangle ABE is 2√3. -/
theorem non_exist_line_l (h₁ : distance_AB x0 y0) : 
  ¬ ∃ t : ℝ, (12 * sqrt (t^2 + 1) / (3 * t^2 + 4) = 2 * sqrt 3) :=
sorry

end standard_equation_of_curve_C_non_exist_line_l_l621_621823


namespace lucys_doll_collection_l621_621256

theorem lucys_doll_collection (D : ℕ) (h1 : 2 = 0.25 * D) : D + 2 = 10 :=
by
  sorry

end lucys_doll_collection_l621_621256


namespace Sues_necklace_total_beads_l621_621685

theorem Sues_necklace_total_beads 
  (purple_beads : ℕ)
  (blue_beads : ℕ)
  (green_beads : ℕ)
  (h1 : purple_beads = 7)
  (h2 : blue_beads = 2 * purple_beads)
  (h3 : green_beads = blue_beads + 11) :
  purple_beads + blue_beads + green_beads = 46 :=
by
  sorry

end Sues_necklace_total_beads_l621_621685


namespace b_sequence_periodic_l621_621926

theorem b_sequence_periodic (b : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h_b1 : b 1 = 2 + Real.sqrt 3)
  (h_b2021 : b 2021 = 11 + Real.sqrt 3) :
  b 2048 = b 2 :=
sorry

end b_sequence_periodic_l621_621926


namespace find_radius_of_circle_l621_621209

noncomputable def radius_of_circle (P Q R : Point) (d : ℝ) (PQ : ℝ) (QR : ℝ) : ℝ :=
  let PR := PQ + QR in
  let power_eq := PQ * PR = (d - r) * (d + r) in
  if power_eq = 360 then sqrt 316 else 0

theorem find_radius_of_circle (P Q R S : Point) (d PQ QR : ℝ) 
  (h1 : distance P center = 26) 
  (h2 : PQ = 15) 
  (h3 : QR = 9) 
  (h4 : tangent_from P S)
  : radius_of_circle P Q R 26 15 9 = sqrt 316 :=
by 
  sorry

end find_radius_of_circle_l621_621209


namespace count_valid_sequences_l621_621247

-- Variables representing the sets
def A := Fin 8 → Set (Fin 2)

-- Condition: A[m] is a subset of A[n] if m divides n
def is_valid_sequence (A : Fin 8 → Set (Fin 2)) : Prop :=
  ∀ m n : Fin 8, (m + 1) ∣ (n + 1) → A m ⊆ A n

-- The number of valid sequences
def num_valid_sequences : ℕ :=
  2025

-- The theorem we need to prove
theorem count_valid_sequences : (∃ A : Fin 8 → Set (Fin 2), is_valid_sequence A) = num_valid_sequences :=
  sorry

end count_valid_sequences_l621_621247


namespace cole_drive_to_work_time_l621_621613

variables (D : ℝ) (T_work T_home : ℝ)

def speed_work : ℝ := 75
def speed_home : ℝ := 105
def total_time : ℝ := 2

theorem cole_drive_to_work_time :
  (T_work = D / speed_work) ∧
  (T_home = D / speed_home) ∧
  (T_work + T_home = total_time) →
  T_work * 60 = 70 :=
by
  sorry

end cole_drive_to_work_time_l621_621613


namespace same_row_product_l621_621067

noncomputable def distinct (a : List ℝ) : Prop :=
  ∀ i j, i ≠ j → a[i] ≠ a[j]

theorem same_row_product
  (n : ℕ)
  (a b : Fin n → ℝ)
  (ha : distinct (List.ofFn a))
  (hb : distinct (List.ofFn b))
  (H : ∀ j : Fin n, (∏ i, (a i + b j)) = c) :
  ∀ i1 i2 : Fin n, (∏ j, (a i1 + b j)) = (∏ j, (a i2 + b j)) := 
by
  sorry

end same_row_product_l621_621067


namespace total_beads_correct_l621_621677

def purple_beads : ℕ := 7
def blue_beads : ℕ := 2 * purple_beads
def green_beads : ℕ := blue_beads + 11
def total_beads : ℕ := purple_beads + blue_beads + green_beads

theorem total_beads_correct : total_beads = 46 := 
by
  have h1 : purple_beads = 7 := rfl
  have h2 : blue_beads = 2 * 7 := rfl
  have h3 : green_beads = 14 + 11 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end total_beads_correct_l621_621677


namespace fred_paid_amount_l621_621452

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def borrowed_movie_price : ℝ := 6.79
def change_received : ℝ := 1.37

def total_cost : ℝ := (number_of_tickets : ℝ) * ticket_price + borrowed_movie_price
def amount_paid : ℝ := total_cost + change_received

theorem fred_paid_amount : amount_paid = 20.00 := sorry

end fred_paid_amount_l621_621452


namespace nails_on_each_side_of_square_l621_621194

theorem nails_on_each_side_of_square (total_nails : ℕ) 
  (square_sides : ℕ) 
  (total_nails = 96) 
  (square_sides = 4) : 
  (total_nails / square_sides = 24) :=
by sorry

end nails_on_each_side_of_square_l621_621194


namespace fifteen_percent_of_x_is_ninety_l621_621757

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l621_621757


namespace x_le_zero_iff_l621_621189

theorem x_le_zero_iff :
  ∀ (x : ℝ), x ≤ 0 ↔ (x < 0 ∨ x = 0) :=
by
  intro x
  exact (le_iff_lt_or_eq x 0)

end x_le_zero_iff_l621_621189


namespace average_waiting_time_for_first_bite_l621_621705

theorem average_waiting_time_for_first_bite
  (bites_first_rod : ℝ)
  (bites_second_rod: ℝ)
  (total_bites: ℝ)
  (time_interval: ℝ)
  (H1 : bites_first_rod = 3)
  (H2 : bites_second_rod = 2)
  (H3 : total_bites = 5)
  (H4 : time_interval = 6) :
  1 / (total_bites / time_interval) = 1.2 :=
by
  rw [H3, H4]
  simp
  norm_num
  rw [div_eq_mul_inv, inv_div, inv_inv]
  norm_num
  sorry

end average_waiting_time_for_first_bite_l621_621705


namespace ceil_neg_sqrt_frac_l621_621264

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l621_621264


namespace exists_isodynamic_points_l621_621912

theorem exists_isodynamic_points (ABC : Triangle)
  (h_non_equilateral : ¬ (ABC.isEquilateral)) :
  ∃ (P Q : IsodynamicPoint ABC), 
    P.insideCircumcircle ∧ ¬ Q.insideCircumcircle ∧
    (equilateral (ABC.projections P)) ∧ (equilateral (ABC.projections Q)) :=
by
  sorry

end exists_isodynamic_points_l621_621912


namespace lucky_number_52000_l621_621879

noncomputable def lucky_number_distribution (k : ℕ) : ℕ :=
  Nat.choose (k + 5) 6

theorem lucky_number_52000 (a : ℕ) (n : ℕ) (h₁: n = 65) (h₂: a = 2005)
  (h₃: ∀ x, x ∈ List.range 52001 → Nat.digits 10 x |>.sum = 7) :
  ∃ b, b = 52000 ∧ List.nthLe (List.sort (· ≤ ·) (List.filter (λ x, Nat.digits 10 x |>.sum = 7) (List.range 100000))) (5 * n - 1)  (by norm_num) = b := by
  sorry

end lucky_number_52000_l621_621879


namespace problem_solution_l621_621388

noncomputable def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π ∧
  a = 2 ∧
  Sqrt (sin B * sin C * sin (C + A) * sin A * (2b - c) * (sin A * cos C + sin C * cos A)) = 0 ∧
  a^2 = b^2 + c^2 - 2 * b * c * cos A

theorem problem_solution (A B C a b c : ℝ)
  (h₁ : triangle A B C a b c)
  (h₂ : a = 2)
  (h₃ : Sqrt (sin A * b * c) = sqrt 3) :
  A = π / 3 ∧ b = 2 ∧ c = 2 := by
  sorry

end problem_solution_l621_621388


namespace obtain_half_not_obtain_one_l621_621620

theorem obtain_half (x : ℕ) : (10 + x) / (97 + x) = 1 / 2 ↔ x = 77 := 
by
  sorry

theorem not_obtain_one (x k : ℕ) : ¬ ((10 + x) / (97 + x) = 1 ∨ (10 * k) / (97 * k) = 1) := 
by
  sorry

end obtain_half_not_obtain_one_l621_621620


namespace students_not_reading_l621_621647

theorem students_not_reading (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_reading : ℚ) (frac_boys_reading : ℚ)
  (h1 : total_girls = 12) (h2 : total_boys = 10)
  (h3 : frac_girls_reading = 5 / 6) (h4 : frac_boys_reading = 4 / 5) :
  let girls_not_reading := total_girls - total_girls * frac_girls_reading
  let boys_not_reading := total_boys - total_boys * frac_boys_reading
  let total_not_reading := girls_not_reading + boys_not_reading
  total_not_reading = 4 := sorry

end students_not_reading_l621_621647


namespace no_nonneg_rational_sol_for_equation_l621_621692

theorem no_nonneg_rational_sol_for_equation :
  ¬ ∃ (x y z : ℚ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x^5 + 2 * y^5 + 5 * z^5 = 11 :=
by
  sorry

end no_nonneg_rational_sol_for_equation_l621_621692


namespace modulus_of_complex_z_l621_621810

open Complex

theorem modulus_of_complex_z :
  let z := (1 - Complex.i) / (1 + Complex.i) + 2 * Complex.i in
  Complex.abs z = 1 :=
by
  let z := (1 - Complex.i) / (1 + Complex.i) + 2 * Complex.i
  sorry

end modulus_of_complex_z_l621_621810


namespace find_QT_l621_621242

-- Define the basic problem conditions
variables (P Q R S T : Type) -- Variables representing points
variables (PQ RS PR : ℝ) -- Lengths of segments as real numbers
variables (QT : ℝ) -- Length of segment QT
variables (equal_areas : Prop) -- Proposition stating the areas are equal

-- Specify the given lengths
axiom PQ_length : PQ = 15
axiom RS_length : RS = 20
axiom PR_length : PR = 22

-- State the equal areas condition
axiom areas_equal : equal_areas = (abs (P.1 - Q.1) * abs (P.2 - Q.2)) = (abs (Q.1 - T.1) * abs (Q.2 - T.2))

-- State the problem to be proved
theorem find_QT : QT = 66 / 7 := by
  sorry

end find_QT_l621_621242


namespace triangle_inequality_min_sec_squared_l621_621071

noncomputable def sec_sq_div4 (x : ℝ) : ℝ :=
  1 / (Real.cos (x / 4))^2

theorem triangle_inequality_min_sec_squared
  {A B C M : ℝ}
  (h1 : M = M)
  (h2 : ∀ (A B C : ℝ), 0 ≤ A ∧ A + B + C = π):
  let P := (A + B + C) / 2 in
  |M - A| + |M - B| + |M - C| ≥ P * min (sec_sq_div4 A) (min (sec_sq_div4 B) (sec_sq_div4 C)) :=
by
  sorry

end triangle_inequality_min_sec_squared_l621_621071


namespace fifteen_percent_of_x_is_ninety_l621_621754

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l621_621754


namespace least_whole_number_clock_equivalent_l621_621939

theorem least_whole_number_clock_equivalent :
  ∃ h : ℕ, h > 6 ∧ h ^ 2 % 24 = h % 24 ∧ ∀ k : ℕ, k > 6 ∧ k ^ 2 % 24 = k % 24 → h ≤ k := sorry

end least_whole_number_clock_equivalent_l621_621939


namespace prob_slope_gte_one_unit_square_l621_621920
open Real

noncomputable def probability_greater_equal_slope_one (P : (ℝ × ℝ)) : ℝ :=
  (let (x, y) := P in
    if x ∈ (Icc 0 1) ∧ y ∈ (Icc 0 1) then
    if y ≥ x - 1/2 then 1 else 0 else 0)

theorem prob_slope_gte_one_unit_square : 
  let prob := ∫ x in 0..1, ∫ y in 0..1, probability_greater_equal_slope_one (x, y) in
  let actual_prob := prob / ((1 - 0) * (1 - 0)) in
  let m := 1 in let n := 4 in
  m + n = 5 :=
sorry

end prob_slope_gte_one_unit_square_l621_621920


namespace avg_waiting_time_is_1_point_2_minutes_l621_621703

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l621_621703


namespace no_regular_polygon_with_half_parallel_diagonals_l621_621746

-- Define the concept of a regular polygon with n sides
def is_regular_polygon (n : ℕ) : Prop :=
  n ≥ 3  -- A polygon has at least 3 sides

-- Define the concept of diagonals in the polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Define the concept of diagonals being parallel to the sides
def parallell_diagonals (n : ℕ) : ℕ :=
  -- This needs more formalization if specified, here's a placeholder
  sorry

-- The main theorem to prove
theorem no_regular_polygon_with_half_parallel_diagonals (n : ℕ) (h : is_regular_polygon n) :
  ¬ (∃ k : ℕ, k = num_diagonals n / 2 ∧ k = parallell_diagonals n) :=
begin
  sorry
end

end no_regular_polygon_with_half_parallel_diagonals_l621_621746


namespace trig_expr_eq_one_l621_621725

theorem trig_expr_eq_one : 
  (∀ (θ : ℝ), θ = 45 → (tan θ = 1) ∧ (sin θ = sqrt 2 / 2)) →
  (1 - (sqrt 2 / 2)^2) / (1 * (sqrt 2 / 2)^2) = 1 := 
by 
  sorry

end trig_expr_eq_one_l621_621725


namespace fraction_division_l621_621593

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := 
by
  -- We need to convert this division into multiplication by the reciprocal
  -- (3 / 4) / (2 / 5) = (3 / 4) * (5 / 2)
  -- Now perform the multiplication of the numerators and denominators
  -- (3 * 5) / (4 * 2) = 15 / 8
  sorry

end fraction_division_l621_621593


namespace interest_rate_per_annum_l621_621132
noncomputable def interest_rate_is_10 : ℝ := 10
theorem interest_rate_per_annum (P R : ℝ) : 
  (1200 * ((1 + R / 100)^2 - 1) - 1200 * R * 2 / 100 = 12) → P = 1200 → R = 10 := 
by sorry

end interest_rate_per_annum_l621_621132


namespace solve_trig_equation_l621_621115

theorem solve_trig_equation (x : ℝ) (n : ℤ) :
  (sin x ≠ 0 ∧ cos x ≠ 0 ∧ cos x ≠ 1 / sqrt 2 ∧ cos x ≠ -1 / sqrt 2) →
  (\sin 2 * x = 2 * sin x * cos x) →
  (\sin 4 * x = 4 * sin x * cos x * (2 * cos x^2 - 1)) →
  (1 / sin x - 1 / sin (2 * x) = 2 / sin (4 * x)) →
  x = 2 * n * π + 2 * π / 3 ∨ x = 2 * n * π - 2 * π / 3 :=
by 
  sorry

end solve_trig_equation_l621_621115


namespace Cn_is_real_l621_621916

-- Given conditions
variables (a b c : ℂ)
variable (n : ℕ)

-- Assume the conditions
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variables (h_abs_a : complex.abs a = complex.abs b)
variables (h_abs_b : complex.abs b = complex.abs c)
variables (A : ℝ) (B : ℝ)
variable  (hA : a + b + c = A)
variable  (hB : a * b * c = B)

-- Prove the statement
theorem Cn_is_real : ∀ n : ℕ, (a^n + b^n + c^n) ∈ ℝ := 
  by sorry

end Cn_is_real_l621_621916


namespace sum_largest_odd_factors_correct_l621_621794

def largest_odd_factor (k : ℕ) : ℕ :=
if k % 2 = 1 then k else largest_odd_factor (k / 2)

def sum_largest_odd_factors (n : ℕ) : ℕ :=
∑ k in (Finset.range (2^n + 1)).filter (λ x, x > 0), largest_odd_factor k

theorem sum_largest_odd_factors_correct (n : ℕ) : 
  sum_largest_odd_factors n = (4^n + 2) / 3 :=
sorry

end sum_largest_odd_factors_correct_l621_621794


namespace percent_decrease_l621_621549

theorem percent_decrease (orig_cost new_cost : ℝ) (h1 : orig_cost = 41) (h2 : new_cost = 7) :
  ((orig_cost - new_cost) / orig_cost) * 100 ≈ 80 :=
by 
  rw [h1, h2]
  -- Proof steps would be here
  sorry

end percent_decrease_l621_621549


namespace maximal_area_isosceles_l621_621321

noncomputable def circle := {center : Point, radius : ℝ}

def point_not_on_circle (O : Point) (R : ℝ) (C : Point) [C_distinct : C ≠ O] : Prop :=
  (dist O C) ≠ R

theorem maximal_area_isosceles (O : Point) (R : ℝ) (C : Point) [C_not_on_circle : point_not_on_circle O R C] :
  ∃ (A B : Point), (dist O A = R) ∧ (dist O B = R) ∧ isoscelesTriangle A B C :=
sorry

end maximal_area_isosceles_l621_621321


namespace condition_eq_l621_621022

-- We are given a triangle ABC with sides opposite angles A, B, and C being a, b, and c respectively.
variable (A B C a b c : ℝ)

-- Conditions for the problem
def sin_eq (A B : ℝ) := Real.sin A = Real.sin B
def cos_eq (A B : ℝ) := Real.cos A = Real.cos B
def sin2_eq (A B : ℝ) := Real.sin (2 * A) = Real.sin (2 * B)
def cos2_eq (A B : ℝ) := Real.cos (2 * A) = Real.cos (2 * B)

-- The main statement we need to prove
theorem condition_eq (h1 : sin_eq A B) (h2 : cos_eq A B) (h4 : cos2_eq A B) : a = b :=
sorry

end condition_eq_l621_621022


namespace ceil_of_neg_sqrt_frac_64_over_9_l621_621288

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l621_621288


namespace tangency_theorem_l621_621894

variable {A B C D P Q : Point}

-- Definitions based on the conditions
def is_parallelogram (A B C D : Point) : Prop :=
  same_relation (<<AB>> <<CD>>) ∧ same_relation (<<AD>> <<BC>>)

def is_on_arc (P : Point) (A B C : Triangle) : Prop :=
  ∃ O : Point, is_circumcenter O A B C ∧ (¬ circle_contains O A P)

def on_segment (Q : Point) (A C : Point) := lies_on_line_segment Q A C

def has_equal_angles (P B C Q D : Point) : Prop :=
  ∠PBC = ∠CDQ

-- Statement of the theorem
theorem tangency_theorem (h_parallelogram : is_parallelogram A B C D)
    (h_P_arc : is_on_arc P A B C)
    (h_Q_segment: on_segment Q A C) 
    (h_angles : has_equal_angles P B C Q D) :
    tangent_to (circumcircle A P Q) A B :=
sorry

end tangency_theorem_l621_621894


namespace problem_solution_l621_621807

def f (x : ℝ) : ℝ := 2 * (Real.log x / Real.log 3) * (Real.log 3 / Real.log 2) + 11

theorem problem_solution : 
    f 2 + f 4 + f 8 + f 16 + f 32 + f 64 = 108 := by
  sorry

end problem_solution_l621_621807


namespace problem_part1_problem_part2_l621_621317

theorem problem_part1 (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_ineq : λ = 6) :
  ∃ (a b : ℝ), (λ / (a + b)) > (1 / a + 2 / b) := 
sorry

noncomputable def max_lambda (a b : ℝ) : ℝ :=
3 + 2 * real.sqrt 2

theorem problem_part2 (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  λ ≤ max_lambda a b :=
sorry

end problem_part1_problem_part2_l621_621317


namespace spherical_to_rectangular_correct_l621_621736

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) := by
  sorry

end spherical_to_rectangular_correct_l621_621736


namespace sum_of_coordinates_eq_nine_halves_l621_621121

theorem sum_of_coordinates_eq_nine_halves {f : ℝ → ℝ} 
  (h₁ : 2 = (f 1) / 2) :
  (4 + (1 / 2) = 9 / 2) :=
by 
  sorry

end sum_of_coordinates_eq_nine_halves_l621_621121


namespace solution_set_l621_621350

variables {f : ℝ → ℝ}

-- Conditions
def f_even (x : ℝ) : Prop := f(x - 1) = f(-(x - 1))
def f_monotonic_increasing (x1 x2 : ℝ) : Prop := (-1 ≤ x1 ∧ x1 < x2) → f(x1) < f(x2)
def f_monotonic_decreasing (x1 x2 : ℝ) : Prop := (x1 < x2 ∧ x2 ≤ -1) → f(x1) > f(x2)

-- Proof problem statement
theorem solution_set (h1 : ∀ x, f_even x) 
                     (h2 : ∀ x1 x2, f_monotonic_increasing x1 x2)
                     (h3 : ∀ x1 x2, f_monotonic_decreasing x1 x2) :
  {x : ℝ | f (1 - 2 ^ x) < f (-7)} = set.Iio 3 :=
by {
  sorry
}

end solution_set_l621_621350


namespace P_2018_has_2018_distinct_real_roots_l621_621631

-- Define the sequence of polynomials P_n(x) with given initial conditions.
noncomputable def P : ℕ → Polynomial ℝ
| 0       := 1
| 1       := Polynomial.X
| (n + 1) := Polynomial.X * P n - P (n - 1)

-- The statement to prove that P_2018(x) has exactly 2018 distinct real roots.
theorem P_2018_has_2018_distinct_real_roots :
  ∃ (roots : Fin 2018 → ℝ), ∀ i j : Fin 2018, i ≠ j → roots i ≠ roots j ∧ ∀ x : ℝ, (Polynomial.aeval x (P 2018)) = 0 ↔ x ∈ Set.range roots := sorry

end P_2018_has_2018_distinct_real_roots_l621_621631


namespace fish_initial_numbers_l621_621025

theorem fish_initial_numbers (x y : ℕ) (h1 : x + y = 100) (h2 : x - 30 = y - 40) : x = 45 ∧ y = 55 :=
by
  sorry

end fish_initial_numbers_l621_621025


namespace find_number_of_right_triangles_l621_621305

open Classical

-- Defining the right triangle properties and incenter coordinates
def is_right_triangle {A B : ℤ × ℤ} (O : ℤ × ℤ) :=
  A.1 ≠ O.1 ∧ A.2 ≠ O.2 ∧ B.1 ≠ O.1 ∧ B.2 ≠ O.2 ∧
  (O.1 - A.1) * (O.1 - B.1) + (O.2 - A.2) * (O.2 - B.2) = 0

-- Defining the incenter coordinates conditions
def has_incenter_coords (p : ℤ) (A B : ℤ × ℤ) :=
  p > 1 ∧ (A.1 * 96 + B.1 * 96 = p * 192) ∧
  (A.2 * 96 + B.2 * 96 = p * 672)

-- Prime number condition
def is_prime (n : ℤ) := n > 1 ∧ ∀ m : ℤ, m > 1 → m < n → n % m ≠ 0

noncomputable def number_of_right_triangles : ℤ :=
  if hp : ∀ p, is_prime p ∧ has_incenter_coords p ∧ is_right_triangle then
    if p = 2 then
      42
    else if p = 3 then
      60
    else
      108
  else
    0

-- Statement of the problem to prove: The number of such right triangles is one of [{108, 42, 60}]
theorem find_number_of_right_triangles : number_of_right_triangles = 42 ∨ number_of_right_triangles = 60 ∨ number_of_right_triangles = 108 :=
by
  sorry

end find_number_of_right_triangles_l621_621305


namespace probability_odd_product_lt_one_eighth_l621_621585

theorem probability_odd_product_lt_one_eighth :
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  p < 1 / 8 :=
by
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  sorry

end probability_odd_product_lt_one_eighth_l621_621585


namespace sum_diff_squares_l621_621106

theorem sum_diff_squares (a b c : ℤ) : ∃ a b c : ℤ, 2019 = a^2 + b^2 - c^2 :=
by {
  use [506, 504, 1],
  sorry
}

end sum_diff_squares_l621_621106


namespace _l621_621356

noncomputable def center_of_symmetry_closest_to_y_axis (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < π) : ℝ × ℝ :=
  (-π/6, 0)

lemma main_theorem (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < π)
  (ht : ∀ x, sin (ω * (x + π/3) + φ) = sin x)
  (hs: ∀ x, sin (α * x + β) = sin x ↔ (α = 1 ∧ β = 0) ∨ (α = -1 ∧ β = π)): 
  center_of_symmetry_closest_to_y_axis ω φ hω hφ = (-π/6, 0) :=
begin
  sorry, -- Translation of the entire proof problem and solution
end

end _l621_621356


namespace length_of_rectangular_garden_l621_621615

-- Define the perimeter and breadth conditions
def perimeter : ℕ := 950
def breadth : ℕ := 100

-- The formula for the perimeter of a rectangle
def formula (L B : ℕ) : ℕ := 2 * (L + B)

-- State the theorem
theorem length_of_rectangular_garden (L : ℕ) 
  (h1 : perimeter = 2 * (L + breadth)) : 
  L = 375 := 
by
  sorry

end length_of_rectangular_garden_l621_621615


namespace committee_count_l621_621800

theorem committee_count (students teachers : ℕ) (committee_size : ℕ) 
  (h_students : students = 11) (h_teachers : teachers = 3) 
  (h_committee_size : committee_size = 8) : 
  ∑ (k : ℕ) in finset.range committee_size.succ, (nat.choose (students + teachers) committee_size) - (nat.choose students committee_size) = 2838 := 
by 
  sorry

end committee_count_l621_621800


namespace solve_absolute_value_quadratic_l621_621539

theorem solve_absolute_value_quadratic (x : ℝ) (h : x < 4) : 
  |x - 4| = x^2 - 5x + 6 ↔ x = 2 - Real.sqrt 2 := by
sorry

end solve_absolute_value_quadratic_l621_621539


namespace bad_arrangements_count_l621_621979

inductive arrangement : Type
| mk (a b c d e : ℕ)

def bad_arrangement (arr : arrangement) : Prop :=
  ∃ n, (1 ≤ n ∧ n ≤ 15) ∧ ¬ (∃ subset : list ℕ, subset.nodup ∧ subset ≠ [] ∧ subset.sum = n ∧ 
    -- Note that we need consecutive numbers, so we would need a separate condition to check
    (arr.contains_consecutive subset))

-- Since we are counting distinct arrangements considering rotations and reflections, and defining contains_consecutive
def correct_arrangement_count : ℕ := 
  -- the specific calculation for the bad arrangements considering rotations and reflections 
  2

theorem bad_arrangements_count : 
  ∀ arrangements : list arrangement, (list bad_arrangement arrangements).length = 2 :=
sorry

end bad_arrangements_count_l621_621979


namespace octagon_area_l621_621337

noncomputable section

variables {BDEF : Type} [has_measure BDEF] (AB BC : ℝ) (area_octagon : ℝ)

-- Condition: BDEF is a square
axiom square_BDEF : is_square BDEF

-- Conditions: AB = 2, BC = 2
axiom AB_eq_2 : AB = 2
axiom BC_eq_2 : BC = 2

-- Theorem: The area of the regular octagon is 16 + 16 * sqrt 2 square units
theorem octagon_area 
    (h_square_BDEF : square_BDEF)
    (h_AB_eq_2 : AB_eq_2)
    (h_BC_eq_2 : BC_eq_2) :
  area_octagon = 16 + 16 * Real.sqrt 2 :=
sorry

end octagon_area_l621_621337


namespace perpendicular_line_slope_l621_621825

theorem perpendicular_line_slope (a : ℝ) :
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  k_MN * (-a / 2) = -1 → a = 1 :=
by
  intros M N k_MN H
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  sorry

end perpendicular_line_slope_l621_621825


namespace sin_sum_inequality_l621_621780

theorem sin_sum_inequality (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (hy : 0 ≤ y ∧ y ≤ π / 2) :
  sin (x + y) < sin x + sin y :=
by
  sorry

end sin_sum_inequality_l621_621780


namespace min_segments_polyline_l621_621211

theorem min_segments_polyline (n : ℕ) (h : n > 0) :
  ∃ p : list (ℝ × ℝ), 
  (∀ i ∈ p, ∃ k l : ℕ, k < n ∧ l < n ∧ i = (k / n, l / n)) ∧ 
  (∀ (a : ℝ × ℝ) (b : ℝ × ℝ), a ∈ p ⊃ b ∈ p ⊃ a ≠ b → (a.1 = b.1 ∨ a.2 = b.2)) ∧ 
  (p.length = 2 * n - 2) := 
begin
  sorry
end

end min_segments_polyline_l621_621211


namespace residents_different_club_presidents_l621_621393

noncomputable def problem_statement : Prop :=
  ∃ (num_residents : ℕ) (num_clubs : ℕ) (num_members_per_club : ℕ),
  num_residents = 1001 ∧
  num_members_per_club = 13 ∧
  num_clubs = (Finset.powerset_len num_members_per_club (Finset.range num_residents + 1)).card ∧
  ∃ (residents : Finset ℕ),
  residents.card = num_residents ∧ 
  (∃ (club_presidency_counts : residents → ℕ),
  ∀ r1 r2 : residents, r1 ≠ r2 → club_presidency_counts r1 ≠ club_presidency_counts r2)

theorem residents_different_club_presidents : problem_statement := 
by sorry

end residents_different_club_presidents_l621_621393


namespace number_line_distance_l621_621098

theorem number_line_distance (x : ℝ) : |x + 1| = 6 ↔ (x = 5 ∨ x = -7) :=
by
  sorry

end number_line_distance_l621_621098


namespace find_length_of_AB_l621_621411

namespace Proofs

-- Define the setup of the triangle and ratios
def right_triangle (ABC : Type) (AB AC BC : ℝ) : Prop :=
  AB^2 + AC^2 = BC^2

def segment_ratio (A B P : Type) (AP PB : ℝ) : Prop :=
  AP / PB = 2 / 3

def point_on_leg (A B P : Type) (AP PB : ℝ) : Prop :=
  segment_ratio A B P AP PB

-- Define the points and length relationships
variables (AB AC BQ CP : ℝ)
variables {P Q : Type}

-- Given conditions
axiom triangle_right : right_triangle ABC AB AC
axiom ratio_AP_PB : segment_ratio A B P (2 * AB / 5) (3 * AB / 5)
axiom ratio_AQ_QC : segment_ratio A C Q (2 * AC / 5) (3 * AC / 5)
axiom given_BQ : BQ = 18
axiom given_CP : CP = 24

-- Required proof statement
theorem find_length_of_AB : AB = 2 * real.sqrt 58.8 :=
sorry

end Proofs

end find_length_of_AB_l621_621411


namespace area_square_l621_621230

-- Given conditions
variable (ABCD : Type) [MetricSpace ABCD] [HasDist ABCD] [HasInner ABCD Real] [HasZero ABCD]
variable (A B C D P Q R : ABCD)

-- Define the conditions
def radius_one (x : ABCD) := dist x A = 1 -- Radius condition
def tangent (x y : ABCD) := dist x y = 2 -- Tangent condition, because 1 + 1 = 2 for radii

-- Define properties of the problem to match the given conditions
axiom hA : radius_one P
axiom hB : radius_one Q
axiom hC : radius_one R
axiom h1 : tangent P Q
axiom h2 : tangent Q R
axiom h3 : tangent R P
axiom h4 : dist A B = sqrt (1 + sqrt (2) + sqrt (3)) * sqrt 2

-- The theorem to prove
theorem area_square : (dist A C) * (dist A C) = 3 + sqrt 2 + sqrt 6 :=
by
  sorry

end area_square_l621_621230


namespace max_points_of_intersection_l621_621934

-- Define the conditions
variable {α : Type*} [DecidableEq α]
variable (L : Fin 100 → α → α → Prop) -- Representation of the lines

-- Define property of being parallel
variable (are_parallel : ∀ {n : ℕ}, L (5 * n) = L (5 * n + 5))

-- Define property of passing through point B
variable (passes_through_B : ∀ {n : ℕ}, ∃ P B, L (5 * n - 4) P B)

-- Prove the stated result
theorem max_points_of_intersection : 
  ∃ max_intersections, max_intersections = 4571 :=
by {
  sorry
}

end max_points_of_intersection_l621_621934


namespace work_efficiency_l621_621477

theorem work_efficiency (orig_time : ℝ) (new_time : ℝ) (work : ℝ) 
  (h1 : orig_time = 1)
  (h2 : new_time = orig_time * (1 - 0.20))
  (h3 : work = 1) :
  (orig_time / new_time) * 100 = 125 :=
by
  sorry

end work_efficiency_l621_621477


namespace find_tangent_line_l621_621787

noncomputable def tangent_line_eq (f : ℝ → ℝ) (x₀ y₀ : ℝ) : ℝ → ℝ :=
  let slope := deriv f x₀
  in λ x => slope * (x - x₀) + y₀

theorem find_tangent_line : 
  tangent_line_eq (λ x : ℝ => x^3 - 2 * x + 1) 1 0 = λ x => x - 1 :=
by 
  sorry

end find_tangent_line_l621_621787


namespace number_of_workers_in_second_group_l621_621880

theorem number_of_workers_in_second_group (w₁ w₂ d₁ d₂ : ℕ) (total_wages₁ total_wages₂ : ℝ) (daily_wage : ℝ) :
  w₁ = 15 ∧ d₁ = 6 ∧ total_wages₁ = 9450 ∧ 
  w₂ * d₂ * daily_wage = total_wages₂ ∧ d₂ = 5 ∧ total_wages₂ = 9975 ∧ 
  daily_wage = 105 
  → w₂ = 19 :=
by
  sorry

end number_of_workers_in_second_group_l621_621880


namespace abc_mod_n_l621_621064

theorem abc_mod_n (n : ℕ) (a b c : ℤ) (hn : 0 < n)
  (h1 : a * b ≡ 1 [ZMOD n])
  (h2 : c ≡ b [ZMOD n]) : (a * b * c) ≡ 1 [ZMOD n] := sorry

end abc_mod_n_l621_621064


namespace number_of_knights_is_five_l621_621464

section KnightsAndKnaves

inductive Inhabitant
| knight : Inhabitant
| knave : Inhabitant

open Inhabitant

variables (a1 a2 a3 a4 a5 a6 a7 : Inhabitant)

def tells_truth : Inhabitant → Prop
| knight := True
| knave := False

def statements : Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → ℕ → Prop
| a1 a2 _ _ _ _ _ 1 := (a1 = knight)
| a1 a2 _ _ _ _ _ 2 := (a1 = knight ∧ a2 = knight)
| a1 a2 a3 _ _ _ _ 3 := (a1 = knave ∨ a2 = knave)
| a1 a2 a3 a4 _ _ _ 4 := (a1 = knave ∨ a2 = knave ∨ a3 = knave)
| a1 a2 a3 a4 a5 _ _ 5 := (a1 = knight ∧ a2 = knight ∨ a1 = knave ∧ a2 = knave)
| a1 a2 a3 a4 a5 a6 _ 6 := (a1 = knave ∨ a2 = knave ∨ a3 = knave ∨ a4 = knave ∨ a5 = knave)
| a1 a2 a3 a4 a5 a6 a7 7 := (a1 = knight ∨ a2 = knight ∨ a3 = knight ∨ a4 = knight ∨ a5 = knight ∨ a6 = knight)

theorem number_of_knights_is_five (h1 : tells_truth a1 ↔ a1 = knight)
                                   (h2 : tells_truth a2 ↔ a2 = knight)
                                   (h3 : tells_truth a3 ↔ a3 = knight)
                                   (h4 : tells_truth a4 ↔ a4 = knight)
                                   (h5 : tells_truth a5 ↔ a5 = knight)
                                   (h6 : tells_truth a6 ↔ a6 = knight)
                                   (h7 : tells_truth a7 ↔ a7 = knight)
                                   (s1 : tells_truth a1 → statements a1 a2 a3 a4 a5 a6 a7 1)
                                   (s2 : tells_truth a2 → statements a1 a2 a3 a4 a5 a6 a7 2)
                                   (s3 : tells_truth a3 → statements a1 a2 a3 a4 a5 a6 a7 3)
                                   (s4 : tells_truth a4 → statements a1 a2 a3 a4 a5 a6 a7 4)
                                   (s5 : tells_truth a5 → statements a1 a2 a3 a4 a5 a6 a7 5)
                                   (s6 : tells_truth a6 → statements a1 a2 a3 a4 a5 a6 a7 6)
                                   (s7 : tells_truth a7 → statements a1 a2 a3 a4 a5 a6 a7 7)
                                   : (∀ i, (i = knight → tells_truth i ¬ ↔ i = knight)) → ∃ (n : ℕ), n = 5 := 
sorry

end KnightsAndKnaves

end number_of_knights_is_five_l621_621464


namespace probability_comparison_l621_621691

variable (Ω : Type) {P : Ω → Prop}

-- Define events as sets of outcomes where Anya waits for at least certain minutes
def EventA (ω : Ω) : Prop := P ω
def EventB (ω : Ω) : Prop := P ω ∧ (2 ≤ 5)
def EventC (ω : Ω) : Prop := P ω ∧ (5 ≤ 5)

-- Prove the relationship between the probabilities of these events
theorem probability_comparison (h : ∀ {E1 E2 : Ω → Prop}, (∀ ω, E2 ω → E1 ω) → ∀ ω, measure_theory.measure_space.measure (E2) ≤ measure_theory.measure_space.measure (E1)) :
  measure_theory.measure_space.measure (EventC) ≤ measure_theory.measure_space.measure (EventB) ∧
  measure_theory.measure_space.measure (EventB) ≤ measure_theory.measure_space.measure (EventA) := by
  sorry

end probability_comparison_l621_621691


namespace number_less_than_neg_one_is_neg_two_l621_621656

theorem number_less_than_neg_one_is_neg_two : ∃ x : ℤ, x = -1 - 1 ∧ x = -2 := by
  sorry

end number_less_than_neg_one_is_neg_two_l621_621656


namespace total_people_on_bus_l621_621231

def initial_people := 4
def added_people := 13

theorem total_people_on_bus : initial_people + added_people = 17 := by
  sorry

end total_people_on_bus_l621_621231


namespace socks_choice_count_l621_621371

variable (white_socks : ℕ) (brown_socks : ℕ) (blue_socks : ℕ) (black_socks : ℕ)

theorem socks_choice_count :
  white_socks = 5 →
  brown_socks = 4 →
  blue_socks = 2 →
  black_socks = 2 →
  (white_socks.choose 2) + (brown_socks.choose 2) + (blue_socks.choose 2) + (black_socks.choose 2) = 18 :=
by
  -- Here the proof would be elaborated
  sorry

end socks_choice_count_l621_621371


namespace distance_traveled_l621_621909

-- Define the conditions
def rate : Real := 60  -- rate of 60 miles per hour
def total_break_time : Real := 1  -- total break time of 1 hour
def total_trip_time : Real := 9  -- total trip time of 9 hours

-- The theorem to prove the distance traveled
theorem distance_traveled : rate * (total_trip_time - total_break_time) = 480 := 
by
  sorry

end distance_traveled_l621_621909


namespace P_n_real_roots_P_2018_real_roots_l621_621623

noncomputable def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 1) := λ x, x * P n x - P (n - 1) x

theorem P_n_real_roots (n: ℕ) : ∃ r: ℕ, r ≡ n := sorry

theorem P_2018_real_roots : ∃ r: ℕ, r = 2018 := P_n_real_roots 2018

end P_n_real_roots_P_2018_real_roots_l621_621623


namespace fifteen_percent_of_x_is_ninety_l621_621758

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l621_621758


namespace vector_coordinates_sine_value_l621_621364

-- Part 1: Prove the coordinates of the vector
theorem vector_coordinates 
    (a : ℝ × ℝ := (1, Real.sin (Real.pi / 6))) 
    (b : ℝ × ℝ := (3, 1)) 
    (h1 : Real.sin (Real.pi / 6) = 1 / 2) : 
    2 • a + b = (5, 2) := 
    by sorry

-- Part 2: Prove the value of the sine function given the conditions
theorem sine_value
    (θ : ℝ) 
    (a : ℝ × ℝ := (1, sin θ)) 
    (b : ℝ × ℝ := (3, 1)) 
    (h1 : θ ∈ (0, Real.pi / 2)) 
    (h2 : a.2 / a.1 = b.2 / b.1) 
    (h3 : Real.sin θ = 1 / 3) 
    (h4 : Real.cos θ = (2 * Real.sqrt 2) / 3): 
    Real.sin (2 * θ + Real.pi / 4) = (5 * Real.sqrt 2) / 6 := 
    by sorry

end vector_coordinates_sine_value_l621_621364


namespace min_tiles_needed_l621_621662

theorem min_tiles_needed : 
  ∀ (tile_width tile_length region_width region_length : ℕ), 
    tile_width = 5 → tile_length = 6 → 
    region_width = 36 → region_length = 48 → 
      (region_width * region_length : ℚ / (tile_width * tile_length : ℚ)).ceil = 58 := 
by
  intros tile_width tile_length region_width region_length h_tile_width h_tile_length h_region_width h_region_length
  sorry

end min_tiles_needed_l621_621662


namespace derivative_at_point_of_tangency_l621_621326

theorem derivative_at_point_of_tangency 
  (f : ℝ → ℝ) 
  (A : ℝ × ℝ) 
  (hA : A = (1, 0)) 
  (h_slope : ∀ x, x = 1 → ∃ θ, θ = 45 ∧ θ = (real.arctan (deriv f x))) : 
  deriv f 1 = 1 := 
by 
  sorry

end derivative_at_point_of_tangency_l621_621326


namespace required_sixth_quiz_score_l621_621254

theorem required_sixth_quiz_score :
  let scores := [91, 94, 88, 90, 101] in
  let sum_scores := scores.sum in
  let num_quizzes := 6 in
  let desired_mean := 95 in
  let required_total := desired_mean * num_quizzes in
  let sixth_score := required_total - sum_scores in
  sixth_score = 106 :=
by
  sorry

end required_sixth_quiz_score_l621_621254


namespace repeating_decimal_to_fraction_l621_621296

noncomputable def decimal_to_fraction : ℚ := 
  let part1 : ℚ := .6
  let part2 : ℚ := 7.2323
  part1 + part2

theorem repeating_decimal_to_fraction :
  (6 * 10^(5:ℚ) + 23)/(10^2 - 1)
        + (6)/(10)
        == 412 / 495 :=
		    sorry

end repeating_decimal_to_fraction_l621_621296


namespace total_earnings_60k_l621_621612

-- Definitions and conditions
variables {A B C x y : ℝ}
def investment_A := 3 * x
def investment_B := 4 * x
def investment_C := 5 * x

def return_A := 6 * y
def return_B := 5 * y
def return_C := 4 * y

def earnings_A := investment_A * (return_A / 100)
def earnings_B := investment_B * (return_B / 100)
def earnings_C := investment_C * (return_C / 100)

-- Given condition
def B_earns_more := earnings_B - earnings_A = 100

-- Proof statement
theorem total_earnings_60k (h : B_earns_more) : earnings_A + earnings_B + earnings_C = 60000 :=
by sorry

end total_earnings_60k_l621_621612


namespace simplify_fraction_l621_621952

theorem simplify_fraction (a b : ℝ) :
  ( (3 * b) / (2 * a^2) )^3 = 27 * b^3 / (8 * a^6) :=
by
  sorry

end simplify_fraction_l621_621952


namespace two_digit_bombastic_prime_numbers_karel_even_digit_l621_621166

-- Definition: A natural number N is bombastic if it does not contain any zeros 
-- and if no smaller natural number has the same product of digits as N.
def is_digit (d : Nat) : Prop := d < 10

def has_no_zeros (n : Nat) : Prop :=
  n.toString.toList.all (λ c => c ≠ '0')

def digit_product (n : Nat) : Nat :=
  (n.toString.toList.map (λ c => c.toNat - '0'.toNat)).prod

def is_bombastic (n : Nat) : Prop :=
  has_no_zeros n ∧ ∀ m, m < n → digit_product m ≠ digit_product n

-- Statement: Identify all two-digit bombastic prime numbers.
def two_digit_primes : List Nat :=
  [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

noncomputable def bombastic_primes : List Nat :=
  two_digit_primes.filter is_bombastic

-- Statement: The set of two-digit bombastic prime numbers.
theorem two_digit_bombastic_prime_numbers :
  bombastic_primes = [29, 37, 47, 59, 67, 79, 89] := sorry

-- Statement: Karel's number contains digit 3 and exactly one even digit.
def contains_digit (n digit : Nat) : Prop :=
  digit ∈ n.toString.toList.map (λ c => c.toNat - '0'.toNat)

def exactly_one_even_digit (n : Nat) : Prop :=
  (n.toString.toList.map (λ c => (c.toNat - '0'.toNat)).filter (λ d => d % 2 = 0)).length = 1

-- Statement: The even digit in Karel's number is 8.
theorem karel_even_digit (N : Nat) (h1 : is_bombastic N) (h2 : contains_digit N 3) (h3 : exactly_one_even_digit N) :
  ∃ d, d % 2 = 0 ∧ contains_digit N d ∧ d = 8 := sorry

end two_digit_bombastic_prime_numbers_karel_even_digit_l621_621166


namespace rounding_up_more_four_digit_numbers_l621_621165

theorem rounding_up_more_four_digit_numbers :
  let count_round_down (k : ℕ) := if (n : ℕ) (h : n^2 + 1 ≤ k ∧ k ≤ n^2 + n) then 1 else 0 in
  let count_round_up (k : ℕ) := if (n : ℕ) (h : n^2 + n + 1 ≤ k ∧ k ≤ n^2 + 2n) then 1 else 0 in
  let total_round_down := ∑ k in (1000 : ℕ) .. 9999, count_round_down k in
  let total_round_up := ∑ k in (1000 : ℕ) .. 9999, count_round_up k in
  total_round_up = total_round_down + 24 :=
sorry

end rounding_up_more_four_digit_numbers_l621_621165


namespace ellipse_properties_l621_621332

theorem ellipse_properties :
  ∀ (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h_ab : a > b)
    (h_ecc : √2 / 2 = √(a^2 - b^2) / a),
  (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) →
  (∃ x y : ℝ, (x = 2 ∧ y = √2)), -- This would show a = 2, c = √2 and b^2 = 2 follows.
  ellipse_eq :
    (∀ x y : ℝ, (x^2 / 4 + y^2 / 2 = 1)) →
  max_area_triangle :
    ∀ (N : ℝ × ℝ) (hN : N = (1, 0)) (M : ℝ × ℝ) (hM : M = (-2, 0))
    (AB : set (ℝ × ℝ))
    (hAB : AB = {p : ℝ × ℝ | ∃ t : ℝ, p.1 = t * p.2 + 1 ∧ (p.1, p.2) ∈ {q : ℝ × ℝ | q.1^2 / 4 + q.2^2 / 2 = 1}}),
  ∃ S : ℝ, S = (1 / 2) * 3 * (√6)
    sorry

end ellipse_properties_l621_621332


namespace average_megabytes_per_hour_l621_621649

theorem average_megabytes_per_hour
  (days_of_music : ℕ)
  (total_megabytes : ℕ)
  (total_hours : ℕ)
  (days_to_hours : days_of_music * 24 = total_hours)
  (library_occupies : total_megabytes = 20000)
  (music_days : days_of_music = 15) :
  (total_megabytes / total_hours).round = 56 := by
  -- sorry is allowing to skip the proof and ensuring the Lean statement builds successfully
  sorry

end average_megabytes_per_hour_l621_621649


namespace find_equation_of_line_L_find_length_segment_AB_l621_621327

-- Definition of the problem conditions
variable {M : Point}
variable {x1 y1 x2 y2 : Float}
variable {x y : Float}

-- Given data
def point_M := M = (2, 2)

def hyperbola (x y : Float) : Prop := x^2 - y^2 / 4 = 1

def midpoint_M (x1 y1 x2 y2 : Float) : Prop :=
  M = ((x1 + x2) / 2, (y1 + y2) / 2)

def equation_of_line_L (x y : Float) : Prop := 
  y = 4 * x - 6

def distance_AB (x1 y1 x2 y2 : Float) : Float :=
  sqrt ((2 * (x2 - x1))^2 + (2 * (y2 - y1))^2)

-- Theorem statements without proofs
theorem find_equation_of_line_L : 
  point_M → ∃ L, (L = equation_of_line_L) :=
  sorry

theorem find_length_segment_AB :
  point_M → ∃ d, (d = distance_AB x1 y1 x2 y2) ∧ 
  (d = 2 * sqrt 102 / 3) :=
  sorry

end find_equation_of_line_L_find_length_segment_AB_l621_621327


namespace number_of_ways_to_arrange_plants_under_lamps_l621_621487

def Plant := {basil1, basil2, aloe, cactus}
def Lamp := {white1, white2, red1, red2, blue}

noncomputable def number_of_arrangements : ℕ :=
  67

theorem number_of_ways_to_arrange_plants_under_lamps :
  ∃ f : Plant → Lamp,
    function.injective f ∨ (function.injective f ∧ number_of_arrangements = 67) :=
by
  have plant_set : set Plant := ({basil1, basil2, aloe, cactus} : set Plant)
  have lamp_set : set Lamp := ({white1, white2, red1, red2, blue} : set Lamp)
  
  -- plant->lamp mapping f must be a valid injection or surjection as per problem constraints.
  have h_injective_or_injective_and_arrangements : function.injective f ∨ (function.injective f ∧ number_of_arrangements = 67) := sorry

  exact ⟨f, h_injective_or_injective_and_arrangements⟩

end number_of_ways_to_arrange_plants_under_lamps_l621_621487


namespace solution_of_inequality_l621_621150

open Set

theorem solution_of_inequality (x : ℝ) :
  x^2 - 2 * x - 3 > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end solution_of_inequality_l621_621150


namespace camel_cost_l621_621191

theorem camel_cost :
  ∀ (C H O E : ℝ),
    (10 * C = 24 * H) →
    (16 * H = 4 * O) →
    (6 * O = 4 * E) →
    (10 * E = 120000) →
    C = 4800 :=
by
  intros C H O E h1 h2 h3 h4
  -- Allow the proof process to be skipped for now
  sorry

end camel_cost_l621_621191


namespace find_number_l621_621772

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l621_621772


namespace octagon_ratio_l621_621131

theorem octagon_ratio (total_area : ℝ) (area_below_PQ : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) (XQ QY : ℝ) :
  total_area = 10 ∧
  area_below_PQ = 5 ∧
  triangle_base = 5 ∧
  triangle_height = 8 / 5 ∧
  area_below_PQ = 1 + (1 / 2) * triangle_base * triangle_height ∧
  XQ + QY = triangle_base ∧
  (1 / 2) * (XQ + QY) * triangle_height = 5
  → (XQ / QY) = 2 / 3 := 
sorry

end octagon_ratio_l621_621131


namespace disjoint_subsets_mod_1000_l621_621058

open Nat

theorem disjoint_subsets_mod_1000 :
  let T := Finset.range 13
  let m := (3^12 - 2 * 2^12 + 1) / 2
  m % 1000 = 625 := 
by
  let T := Finset.range 13
  let m := (3^12 - 2 * 2^12 + 1) / 2
  have : m % 1000 = 625 := sorry
  exact this

end disjoint_subsets_mod_1000_l621_621058


namespace part_I_part_II_l621_621838

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

theorem part_I (x : ℝ) : f x > 4 ↔ x < -1.5 ∨ x > 2.5 := 
sorry

theorem part_II (a : ℝ) : (∀ x, f x ≥ a) ↔ a ≤ 3 := 
sorry

end part_I_part_II_l621_621838


namespace picked_clovers_when_one_four_found_l621_621911

-- Definition of conditions
def total_leaves : ℕ := 100
def leaves_three_leaved_clover : ℕ := 3
def leaves_four_leaved_clover : ℕ := 4
def one_four_leaved_clover : ℕ := 1

-- Proof Statement
theorem picked_clovers_when_one_four_found (three_leaved_count : ℕ) :
  (total_leaves - leaves_four_leaved_clover) / leaves_three_leaved_clover = three_leaved_count → 
  three_leaved_count = 32 :=
by
  sorry

end picked_clovers_when_one_four_found_l621_621911


namespace max_distance_ellipse_point_to_line_l621_621814
noncomputable def max_distance_point_to_line (θ : ℝ) : ℝ :=
  let P := (4 * Real.cos θ, 3 * Real.sin θ) in
  let distance := (|12 * Real.cos θ - 12 * Real.sin θ - 24| / Real.sqrt (3^2 + (-4)^2)) in
  distance

theorem max_distance_ellipse_point_to_line :
  ∃ θ : ℝ, max_distance_point_to_line θ = (12 / 5) * (2 + Real.sqrt 2) :=
sorry

end max_distance_ellipse_point_to_line_l621_621814


namespace math_problem_replace_digits_l621_621946

theorem math_problem_replace_digits :
  (1 = A) ∧ (6 = B) ∧ (5 = C) ∧ (3 = D) ∧ (2 = E) ∧ (4 = F) →
  (A * 10 + B) ^ C = (D * 10 + E) ^ F :=
by {
  intros,
  simp [*],
  sorry
}

end math_problem_replace_digits_l621_621946


namespace odd_function_value_at_neg2_l621_621374

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_ge_one : ∀ x, 1 ≤ x → f x = 3 * x - 7)

theorem odd_function_value_at_neg2 : f (-2) = 1 :=
by
  -- Proof goes here
  sorry

end odd_function_value_at_neg2_l621_621374


namespace ceil_neg_sqrt_fraction_l621_621276

theorem ceil_neg_sqrt_fraction :
  (⌈-real.sqrt (64 / 9)⌉ = -2) :=
by
  -- Define the necessary conditions
  have h1 : real.sqrt (64 / 9) = 8 / 3 := by sorry,
  have h2 : -real.sqrt (64 / 9) = -8 / 3 := by sorry,
  -- Apply the ceiling function and prove the result
  exact sorry

end ceil_neg_sqrt_fraction_l621_621276


namespace notebook_cost_l621_621391

theorem notebook_cost
  (students : ℕ)
  (majority_students : ℕ)
  (cost : ℕ)
  (notebooks : ℕ)
  (h1 : students = 36)
  (h2 : majority_students > 18)
  (h3 : notebooks > 1)
  (h4 : cost > notebooks)
  (h5 : majority_students * cost * notebooks = 2079) :
  cost = 11 :=
by
  sorry

end notebook_cost_l621_621391


namespace midpoint_distance_l621_621919

theorem midpoint_distance :
  let A := (0 : ℝ, 0 : ℝ, 0 : ℝ)
  let B := (4 : ℝ, 0 : ℝ, 0 : ℝ)
  let C := (4 : ℝ, 3 : ℝ, 0 : ℝ)
  let D := (0 : ℝ, 3 : ℝ, 0 : ℝ)
  let A' := (0 : ℝ, 0 : ℝ, 15 : ℝ)
  let B' := (4 : ℝ, 0 : ℝ, 12 : ℝ)
  let C' := (4 : ℝ, 3 : ℝ, 27 : ℝ)
  let D' := (0 : ℝ, 3 : ℝ, 33 : ℝ)
  let M := ((A'.1 + C'.1) / 2, (A'.2 + C'.2) / 2, (A'.3 + C'.3) / 2)
  let N := ((B'.1 + D'.1) / 2, (B'.2 + D'.2) / 2, (B'.3 + D'.3) / 2)
  dist M N = 1.5 :=
by
  sorry

end midpoint_distance_l621_621919


namespace password_correct_l621_621891

-- conditions
def poly1 (x y : ℤ) : ℤ := x ^ 4 - y ^ 4
def factor1 (x y : ℤ) : ℤ := (x - y) * (x + y) * (x ^ 2 + y ^ 2)

def poly2 (x y : ℤ) : ℤ := x ^ 3 - x * y ^ 2
def factor2 (x y : ℤ) : ℤ := x * (x - y) * (x + y)

-- given values
def x := 18
def y := 5

-- goal
theorem password_correct : factor2 x y = 18 * 13 * 23 :=
by
  -- We setup the goal with the equivalent sequence of the password generation
  sorry

end password_correct_l621_621891


namespace race_participants_minimum_l621_621526

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l621_621526


namespace largest_subset_l621_621050

def is_valid_subset (T : set ℕ) : Prop :=
  ∀ x y ∈ T, x ≠ y → |x - y| ≠ 5 ∧ |x - y| ≠ 8

theorem largest_subset (T : set ℕ) (hT : T ⊆ (set.Icc 1 2023))
  (h_valid: is_valid_subset T) : T.card = 935 :=
sorry

end largest_subset_l621_621050


namespace max_elements_of_T_l621_621053

theorem max_elements_of_T : 
  ∃ (T : Set ℕ), T ⊆ {x | 1 ≤ x ∧ x ≤ 2023} ∧ 
    (∀ a ∈ T, ∀ b ∈ T, a ≠ b → (a - b) % 5 ≠ 0 ∧ (a - b) % 8 ≠ 0) ∧ 
    T.finite ∧ T.to_finset.card = 780 :=
sorry

end max_elements_of_T_l621_621053


namespace monotonicity_f_a_le_0_monotonicity_f_a_gt_0_g_has_no_zeros_values_of_a_l621_621063

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 - a - Real.log x

noncomputable def g (x : ℝ) := 1 / x - Real.exp (1 - x)

theorem monotonicity_f_a_le_0 (a : ℝ) (h : a ≤ 0) :
  ∀ x > 0, (f a)' x < 0 := sorry

theorem monotonicity_f_a_gt_0 (a : ℝ) (h : a > 0) :
  ∀ x, (0 < x ∧ x < (Real.sqrt (2 * a)) / (2 * a)) ∨ (x > (Real.sqrt (2 * a)) / (2 * a)) → 
  (f a)' x < 0 ∨ (f a)' x > 0 := sorry

theorem g_has_no_zeros (x : ℝ) (h : x > 1) :
  g x > 0 := sorry

theorem values_of_a (a : ℝ) (h : a ≥ 1 / 2) :
  ∀ x > 1, f a x > g x := sorry

end monotonicity_f_a_le_0_monotonicity_f_a_gt_0_g_has_no_zeros_values_of_a_l621_621063


namespace probability_hugo_rolls_4_given_he_wins_l621_621886

noncomputable def probability_hugo_wins_game (H1 : ℕ) (W : Prop) : ℚ :=
  if H1 = 4 ∧ W 
  then 1 / 6 * 256 / 1296 / (1 / 5)
  else 0

theorem probability_hugo_rolls_4_given_he_wins 
  (players : Fin 5 → ℕ) 
  (Hugo : Fin 5) 
  (hugo_roll : ℕ)
  (win_event : Prop) 
  (hugo_won : win_event ↔ players Hugo = hugo_roll ∧ hugo_roll = 4) :
  probability_hugo_wins_game hugo_roll win_event = 40 / 243 :=
begin
  sorry
end

end probability_hugo_rolls_4_given_he_wins_l621_621886


namespace find_XY_l621_621424

noncomputable def midpoint (a b : ℝ) : ℝ := (a + b) / 2

theorem find_XY 
  {OX OY: ℝ} 
  (M := midpoint 0 OX)
  (N := midpoint 0 OY)
  (XN = 23 : ℝ)
  (YM = 25 : ℝ)
  (YZ = 8 : ℝ)
  (OY = 8) -- inferred from YZ = OY
  (a := OX/2)
  (b := OY/2) :
  let XY := 2 * real.sqrt (a ^ 2 + b ^ 2)
  in XY = 27 :=
by
  have h1 : XN ^ 2 = (2*a) ^ 2 + b ^ 2 := by sorry
  have h2 : YM ^ 2 = a ^ 2 + (2*b) ^ 2 := by sorry
  have h3 : 529 = 4*a ^ 2 + b ^ 2 := by sorry
  have h4 : 625 = a ^ 2 + 4*b ^ 2 := by sorry
  have h5 : 1154 = 5*a ^ 2 + 5*b ^ 2 := by sorry
  have hab2 := 230.8
  have XY := 2 * real.sqrt (a ^ 2 + b ^ 2)
  exact XY = 27

end find_XY_l621_621424


namespace area_of_quadrilateral_l621_621483

-- Let A, B, C, D be points in a 2-dimensional Euclidean space
variables {A B C D : Point}

-- Given conditions
-- 1. Quadrilateral ABCD has right angles at B and D
-- 2. AC = 5
-- 3. The sides of ABCD are integers with at least two having distinct lengths

-- Definitions of points and sides
def AB := dist A B
def BC := dist B C
def AD := dist A D
def DC := dist D C
def AC := dist A C

-- Define the areas of right-angled triangles
def area_ABC := (1 / 2) * AB * BC
def area_ADC := (1 / 2) * AD * DC

-- The statement to prove
theorem area_of_quadrilateral (h_right_angles: ∠ B = 90 ∧ ∠ D = 90)
  (h_hypotenuse: AC = 5)
  (h_distinct_lengths: AB ≠ AD ∨ AB ≠ DC ∨ AD ≠ DC):
  area_ABC + area_ADC = 12 := 
by
  sorry

end area_of_quadrilateral_l621_621483


namespace fifteen_percent_of_x_is_ninety_l621_621765

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l621_621765


namespace train_speed_after_speedup_l621_621591

theorem train_speed_after_speedup
  (distance : ℤ)
  (speed_factor : ℚ)
  (time_reduction : ℤ)
  (original_speed : ℤ)
  (sped_up_speed : ℤ) :
  distance = 1280 →
  speed_factor = 3.2 →
  time_reduction = 11 →
  original_speed = 80 →
  sped_up_speed = 3.2 * 80 →
  sped_up_speed = 256 :=
by
  intros h1 h2 h3 h4 h5
  rw [h4, h5]
  norm_num
  trivial

end train_speed_after_speedup_l621_621591


namespace solution_interval_l621_621299

theorem solution_interval (x : ℝ) : (2 ≤ x / (3 * x - 7)) ∧ (x / (3 * x - 7) < 9) ↔ x ∈ Ioc (63 / 26) (14 / 5) :=
sorry

end solution_interval_l621_621299


namespace distance_between_A_and_B_l621_621940

-- Define variables representing the speeds of person A and person B
variables (a b : ℕ)

-- Define the condition that they meet after 2 hours
def meet_after_two_hours (a b : ℕ) : Prop :=
  2 * a + 2 * b

-- The theorem statement that we need to prove
theorem distance_between_A_and_B (a b : ℕ) : meet_after_two_hours a b = 2 * a + 2 * b :=
sorry

end distance_between_A_and_B_l621_621940


namespace handshakes_in_tournament_l621_621394

-- Define the entities based on conditions.
def num_teams : Nat := 4
def team_size : Nat := 2
def total_women : Nat := num_teams * team_size
def handshakes_per_woman : Nat := total_women - 1

-- Define the theorem to state the problem and the expected solution.
theorem handshakes_in_tournament : 
  ∀ (num_teams : Nat) (team_size : Nat) (total_women : Nat)
  (handshakes_per_woman : Nat), 
  num_teams = 4 → 
  team_size = 2 → 
  total_women = num_teams * team_size →
  handshakes_per_woman = total_women - 2 →
  total_women * (handshakes_per_woman // 2) = 24 := by
  sorry

end handshakes_in_tournament_l621_621394


namespace inversion_image_of_A_l621_621159

noncomputable def symmetric_point (O A P : Point) : Point := sorry

-- Assuming Point, Circle, Line, and inversion function definitions exist in Mathlib or defining stubs
axiom inversion (S : Circle) (A : Point) : Point

axiom line_intersection (l1 l2 : Line) : Point

structure Point :=
(x : ℝ)
(y : ℝ)

structure Circle :=
(center : Point)
(radius : ℝ)

structure Line :=
(p1 p2 : Point)

def lies_on_circle (P : Point) (S : Circle) : Prop := sorry
def lies_on_line (P : Point) (l : Line) : Prop := sorry

axiom symmetry_wrt_line (M : Point) (l : Line) : Point

theorem inversion_image_of_A (S : Circle) (O A M N : Point) (l : Line) 
  (h1 : lies_on_circle M S) (h2 : lies_on_circle N S) 
  (h3 : lies_on_line M l) (h4 : lies_on_line N l) 
  (h5 : ¬ lies_on_line O l)
  : 
  let M' := symmetric_point O A M,
      N' := symmetric_point O A N,
      A' := line_intersection (line M N') (line M' N) in
  A' = inversion S A :=
sorry

end inversion_image_of_A_l621_621159


namespace integer_product_zero_l621_621566

theorem integer_product_zero (a : ℤ) (x : Fin 13 → ℤ)
  (h : a = ∏ i, (1 + x i) ∧ a = ∏ i, (1 - x i)) :
  a * ∏ i, x i = 0 :=
sorry

end integer_product_zero_l621_621566


namespace video_games_expenditure_l621_621862

theorem video_games_expenditure (allowance : ℝ) (books_expense : ℝ) (snacks_expense : ℝ) (clothes_expense : ℝ) 
    (initial_allowance : allowance = 50)
    (books_fraction : books_expense = 1 / 7 * allowance)
    (snacks_fraction : snacks_expense = 1 / 2 * allowance)
    (clothes_fraction : clothes_expense = 3 / 14 * allowance) :
    50 - (books_expense + snacks_expense + clothes_expense) = 7.15 :=
by
  sorry

end video_games_expenditure_l621_621862


namespace eccentricity_line_equation_l621_621239

-- Defining the ellipse and its constraints
def ellipse (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 = 1 ∧ a > 1

-- Proving the eccentricity of the ellipse
theorem eccentricity {a : ℝ} (h : a > 1) : 
  (dist (0 : ℝ) (line (point.mk 0 (-1)) (point.mk a 0)) = sqrt 3 / 2) →
  ellipse a x y →
  exists (e : ℝ), e = sqrt 6 / 3 :=
sorry

-- Equation of the line l involving coordinates of A, B, and C
theorem line_equation (a : ℝ) (x1 y1 x2 y2 : ℝ) :
  (a > 1) →
  ellipse a x1 y1 →
  ellipse a x2 y2 →
  (∃ (l : ℝ), line_through_foci l x1 y1 x2 y2 x y (sqrt 2) = true) :=
sorry

end eccentricity_line_equation_l621_621239


namespace ratio_AH_HD_correct_l621_621021

noncomputable def ratio_AH_HD (A B C H D : Type) [HasAngle A C]
  (BC AC : ℝ) (H orthocenter : Triangle A B C) (AD : Altitude A to BC) (HD : Altitude H to D)
  (h1 : BC = 6) 
  (h2 : AC = 4 * Real.sqrt 2)
  (h3 : Angle A C B = Real.pi / 3) :
  Real := (2 * Real.sqrt 3 * (Real.sqrt 6 - Real.sqrt 3)) / (6 - 2 * Real.sqrt 2)
  
theorem ratio_AH_HD_correct {A B C H D : Type} [HasAngle A C]
  (BC AC : ℝ) (orthocenter : Triangle A B C) (AD : Altitude A to BC) (HD : Altitude H to D)
  (h1 : BC = 6) 
  (h2 : AC = 4 * Real.sqrt 2)
  (h3 : Angle A C B = Real.pi / 3) :
  ratio_AH_HD A B C H D BC AC orthocenter AD HD h1 h2 h3 = (2 * Real.sqrt 3 * (Real.sqrt 6 - Real.sqrt 3)) / (6 - 2 * Real.sqrt 2) := 
sorry

end ratio_AH_HD_correct_l621_621021


namespace image_of_square_is_vertically_symmetric_l621_621065

noncomputable def transform (x y : ℝ) : ℝ × ℝ :=
  (x^2 + y^2, x^2 * y^2)

def P := (0, 0) : ℝ × ℝ
def Q := (1, 0) : ℝ × ℝ
def R := (1, 1) : ℝ × ℝ
def S := (0, 1) : ℝ × ℝ

def transformed_P := transform 0 0
def transformed_Q := transform 1 0
def transformed_R := transform 1 1
def transformed_S := transform 0 1

theorem image_of_square_is_vertically_symmetric :
  let P' := transformed_P,
      Q' := transformed_Q,
      R' := transformed_R,
      S' := transformed_S in
  true :=
by
  sorry

end image_of_square_is_vertically_symmetric_l621_621065


namespace min_number_of_participants_l621_621503

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l621_621503


namespace fraction_covered_by_triangle_is_13_over_84_l621_621672

/-- Type for representing points in 2D space --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Define the vertices of the triangle --/
def A : Point := ⟨2, 2⟩
def B : Point := ⟨6, 3⟩
def C : Point := ⟨5, 6⟩

/-- Define the dimensions of the grid --/
def grid_width : ℝ := 7
def grid_height : ℝ := 6

/-- Calculate the area of the grid --/
def area_grid : ℝ := grid_width * grid_height

/-- Calculate the area of the triangle using the Shoelace theorem --/
def area_triangle : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

/-- Define the covered fraction --/
def covered_fraction : ℝ := area_triangle / area_grid

/-- Theorem: The fraction of the grid covered by the triangle --/
theorem fraction_covered_by_triangle_is_13_over_84 :
  covered_fraction = (13 / 84) :=
by
  -- Proof would go here.
  sorry

end fraction_covered_by_triangle_is_13_over_84_l621_621672


namespace find_length_AE_l621_621821

-- Definition of an isosceles triangle
structure IsoscelesTriangle (α : Type) [OrderedCommRing α] :=
  (A B C E D : α)
  (AB_eq_BC: A = B ∧ B = C)
  (point_E_beyond_A : E > A)
  (point_D_on_BC : D > B ∧ D < C)
  (angle_ADC_eq_60 : ∠ADC = 60)
  (angle_AEC_eq_60 : ∠AEC = 60)
  (AD_eq_13 : AD = 13)
  (CE_eq_13 : CE = 13)
  (DC_eq_9 : DC = 9)

theorem find_length_AE (α : Type) [OrderedCommRing α] (T : IsoscelesTriangle α) : T.AE = 4 := by
  sorry

end find_length_AE_l621_621821


namespace cost_of_additional_weight_l621_621582

def initial_weight := 60
def percent_increase := 0.60
def ingot_weight := 2
def ingot_cost := 5
def discount_threshold := 10
def discount_rate := 0.20

theorem cost_of_additional_weight :
  let additional_weight := initial_weight * percent_increase in
  let ingots_needed := additional_weight / ingot_weight in
  let total_cost := ingots_needed * ingot_cost in
  let discount := if ingots_needed > discount_threshold then discount_rate * total_cost else 0 in
  total_cost - discount = 72 :=
by
  sorry

end cost_of_additional_weight_l621_621582


namespace find_triangle_lengths_l621_621589

-- Conditions:
-- 1. Two right-angled triangles are similar.
-- 2. Bigger triangle sides: x + 1 and y + 5, Area larger by 8 cm^2

def triangle_lengths (x y : ℝ) : Prop := 
  (y = 5 * x ∧ 
  (5 / 2) * (x + 1) ^ 2 - (5 / 2) * x ^ 2 = 8)

theorem find_triangle_lengths (x y : ℝ) : triangle_lengths x y ↔ (x = 1.1 ∧ y = 5.5) :=
sorry

end find_triangle_lengths_l621_621589


namespace seating_arrangements_count_l621_621960

-- Define the main entities: the three teams and the conditions
inductive Person
| Jupitarian
| Saturnian
| Neptunian

open Person

-- Define the seating problem constraints
def valid_arrangement (seating : Fin 12 → Person) : Prop :=
  seating 0 = Jupitarian ∧ seating 11 = Neptunian ∧
  (∀ i, seating (i % 12) = Jupitarian → seating ((i + 11) % 12) ≠ Neptunian) ∧
  (∀ i, seating (i % 12) = Neptunian → seating ((i + 11) % 12) ≠ Saturnian) ∧
  (∀ i, seating (i % 12) = Saturnian → seating ((i + 11) % 12) ≠ Jupitarian)

-- Main theorem: The number of valid arrangements is 225 * (4!)^3
theorem seating_arrangements_count :
  ∃ M : ℕ, (M = 225) ∧ ∃ arrangements : Fin 12 → Person, valid_arrangement arrangements :=
sorry

end seating_arrangements_count_l621_621960


namespace angle_MCN_is_45_degrees_l621_621097

-- Define the right triangle ABC with ∠C = 90°
variables {A B C M N : Type*} [Nonempty A] [Nonempty B] [Nonempty C]
variables {d : ℝ} (h1 : BC d = BM d) (h2 : AC d = AN d)

-- Main statement rewriting the math problem into Lean statement
theorem angle_MCN_is_45_degrees (h : BC d = BM d) (h' : AC d = AN d) : 
  ∠MCN = 45 :=
begin
  sorry
end

end angle_MCN_is_45_degrees_l621_621097


namespace ceil_neg_sqrt_eq_neg_two_l621_621283

noncomputable def x : ℝ := -Real.sqrt (64 / 9)

theorem ceil_neg_sqrt_eq_neg_two : Real.ceil x = -2 := by
  exact sorry

end ceil_neg_sqrt_eq_neg_two_l621_621283


namespace f_increasing_on_interval_l621_621789

-- Define the function f(x)
def f (x : ℝ) : ℝ := log (1/2) (x^2 - 4)

-- Define the intervals 
def domain := {x : ℝ | x < -2 ∨ x > 2}

-- Define the interval where f(x) is increasing
def increasingInterval := {x : ℝ | x < -2}

theorem f_increasing_on_interval : ∀ x ∈ increasingInterval, ∃ (ε > 0), ∀ y, x < y ∧ y < x + ε → f y > f x :=
by
  sorry

end f_increasing_on_interval_l621_621789


namespace toothpicks_in_25th_stage_l621_621555

theorem toothpicks_in_25th_stage :
  ∃ f : ℕ → ℕ, (∀ n, f n = 3 * n) ∧ f 25 = 75 :=
begin
  use (λ n, 3 * n),
  split,
  { intros n,
    refl },
  { refl }
end

end toothpicks_in_25th_stage_l621_621555


namespace circle_O1_radius_5_l621_621229

theorem circle_O1_radius_5 (radius_O1 : ℝ) (AC AD DE : ℝ) (area_CDE : ℝ) :
  radius_O1 = 5 :=
by
  -- Given conditions
  assume (radius_O1 = 5) -- Condition 4 
    (AC = 8)             -- Condition 5 
    (AD = 12)            -- Condition 6
    (DE = 14)            -- Condition 7
    (area_CDE = 112)     -- Condition 8
  sorry

end circle_O1_radius_5_l621_621229


namespace area_inequality_convex_quadrilateral_l621_621043

theorem area_inequality_convex_quadrilateral 
  (a b c d : ℝ) (S : ℝ) (x1 x2 x3 x4 : ℝ) :
  (∀ (a b c d : ℝ), {x1, x2, x3, x4} = {a, b, c, d}) → 
  (∀ (a b c d : ℝ), S = area_of_quadrilateral a b c d) → 
  S ≤ 0.5 * (x1 * x2 + x3 * x4) :=
by
  sorry

end area_inequality_convex_quadrilateral_l621_621043


namespace beautiful_cell_impossible_l621_621095

def is_beautiful (board : ℕ × ℕ → bool) (x y : ℕ) : Prop :=
  (board (x+1, y) + board (x-1, y) + board (x, y+1) + board (x, y-1)) % 2 = 0

def board_size : Prop := 100 * 100

theorem beautiful_cell_impossible :
  ¬ ∃ (board : ℕ × ℕ → bool) (x y : ℕ), is_beautiful board x y ∧ 
  (∀ x' y', (x', y') ≠ (x, y) → ¬ is_beautiful board x' y') :=
sorry

end beautiful_cell_impossible_l621_621095


namespace MsSatosClassRatioProof_l621_621092

variable (g b : ℕ) -- g is the number of girls, b is the number of boys

def MsSatosClassRatioProblem : Prop :=
  (g = b + 6) ∧ (g + b = 32) → g / b = 19 / 13

theorem MsSatosClassRatioProof : MsSatosClassRatioProblem g b := by
  sorry

end MsSatosClassRatioProof_l621_621092


namespace find_unknown_number_l621_621777

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l621_621777


namespace Stuart_initial_marbles_l621_621215

variable (Betty_marbles Stuart_final increased_by: ℤ) 

-- Conditions as definitions
def Betty_has : Betty_marbles = 60 := sorry 
def Stuart_collect_increase : Stuart_final = 80 := sorry 
def percentage_given : ∃ x, x = (40 * Betty_marbles) / 100 := sorry 

-- Theorem to prove Stuart had 56 marbles initially
theorem Stuart_initial_marbles 
  (h1 : Betty_has)
  (h2 : Stuart_collect_increase)
  (h3 : percentage_given) :
  ∃ y, y = Stuart_final - 24 := 
sorry

end Stuart_initial_marbles_l621_621215


namespace race_minimum_participants_l621_621522

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l621_621522


namespace ceil_neg_sqrt_64_over_9_l621_621272

theorem ceil_neg_sqrt_64_over_9 : Real.ceil (-Real.sqrt (64 / 9)) = -2 := 
by
  sorry

end ceil_neg_sqrt_64_over_9_l621_621272


namespace problem_statement_l621_621855

variables {L : Type} {P : Type} [AffineSpace L P]

def line_perp (a b : L) := sorry     -- line a is perpendicular to line b
def line_subset_plane (a : L) (α : P) := sorry -- line a is in plane α
def plane_parallel (α β : P) := sorry -- plane α is parallel to plane β
def line_perp_plane (b : L) (β : P) := sorry -- line b is perpendicular to plane β

axiom exists_distinct_lines {a b : L} : a ≠ b
axiom exists_distinct_planes {α β : P} : α ≠ β

theorem problem_statement {a b : L} {α β : P}
  (H₁ : line_subset_plane a α)
  (H₂ : line_perp b β)
  (H₃ : plane_parallel α β) :
  line_perp a b :=
by sorry

end problem_statement_l621_621855


namespace Rose_has_20_crystal_beads_l621_621938

noncomputable def num_crystal_beads (metal_beads_Nancy : ℕ) (pearl_beads_more_than_metal : ℕ) (beads_per_bracelet : ℕ)
    (total_bracelets : ℕ) (stone_to_crystal_ratio : ℕ) : ℕ :=
  let pearl_beads_Nancy := metal_beads_Nancy + pearl_beads_more_than_metal
  let total_beads_Nancy := metal_beads_Nancy + pearl_beads_Nancy
  let beads_needed := beads_per_bracelet * total_bracelets
  let beads_Rose := beads_needed - total_beads_Nancy
  beads_Rose / stone_to_crystal_ratio.succ

theorem Rose_has_20_crystal_beads :
  num_crystal_beads 40 20 8 20 2 = 20 :=
by
  sorry

end Rose_has_20_crystal_beads_l621_621938


namespace find_second_number_l621_621158

theorem find_second_number (x : ℕ) (h1 : ∀ d : ℕ, d ∣ 60 → d ∣ x → d ∣ 18) 
                           (h2 : 60 % 18 = 6) (h3 : x % 18 = 10) 
                           (h4 : x > 60) : 
  x = 64 := 
by
  sorry

end find_second_number_l621_621158


namespace angle_R_measure_l621_621900

theorem angle_R_measure (M N P Q R O : Type)
  (intersection : ∃ O : Type, segments_intersect_anticlockwise_at M N P Q R O)
  (isosceles_eq : MO = OP ∧ OP = OR ∧ OR = OQ ∧ OQ = MO)
  (angle_relation : ∠M = 5/2 * ∠N)
  : ∠R = 40 :=
by
  sorry

end angle_R_measure_l621_621900


namespace magnitude_square_l621_621293

def z1 : ℂ := 3 * real.sqrt 2 - complex.I * 5
def z2 : ℂ := 2 * real.sqrt 5 + complex.I * 4

theorem magnitude_square : complex.abs (z1 * z2) ^ 2 = 1548 :=
by {
  sorry
}

end magnitude_square_l621_621293


namespace committee_formations_l621_621802

theorem committee_formations :
  let students := 11
  let teachers := 3
  let total_people := students + teachers
  let committee_size := 8
  (nat.choose total_people committee_size) - (nat.choose students committee_size) = 2838 :=
by
  sorry

end committee_formations_l621_621802


namespace outliers_in_data_set_l621_621130

-- Define the data set
def dataSet : List ℕ := [6, 19, 33, 33, 39, 41, 41, 43, 51, 57]

-- Define the given quartiles
def Q1 : ℕ := 33
def Q3 : ℕ := 43

-- Define the interquartile range
def IQR : ℕ := Q3 - Q1

-- Define the outlier thresholds
def lowerOutlierThreshold : ℕ := Q1 - 3 / 2 * IQR
def upperOutlierThreshold : ℕ := Q3 + 3 / 2 * IQR

-- Define what it means to be an outlier
def isOutlier (x : ℕ) : Bool :=
  x < lowerOutlierThreshold ∨ x > upperOutlierThreshold

-- Count the number of outliers in the data set
def countOutliers (data : List ℕ) : ℕ :=
  (data.filter isOutlier).length

theorem outliers_in_data_set :
  countOutliers dataSet = 1 :=
by
  sorry

end outliers_in_data_set_l621_621130


namespace flower_total_expense_l621_621028

def tulips := 250
def carnations := 375
def roses := 320
def daffodils := 200
def lilies := 100

def price_tulips := 2.0
def price_carnations := 1.5
def price_roses := 3.0
def price_daffodils := 1.0
def price_lilies := 4.0

def total_expenses := tulips * price_tulips + carnations * price_carnations + roses * price_roses + daffodils * price_daffodils + lilies * price_lilies

theorem flower_total_expense : total_expenses = 2622.5 := by
  sorry

end flower_total_expense_l621_621028


namespace minimum_participants_l621_621532

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l621_621532


namespace ceil_neg_sqrt_eq_neg_two_l621_621285

noncomputable def x : ℝ := -Real.sqrt (64 / 9)

theorem ceil_neg_sqrt_eq_neg_two : Real.ceil x = -2 := by
  exact sorry

end ceil_neg_sqrt_eq_neg_two_l621_621285


namespace change_in_expression_l621_621407

-- Let's define the function first
def f (x : ℝ) : ℝ := x^3 - 2 * x + 1

-- Now we construct the statement to prove the desired changes in the expression
theorem change_in_expression (x b : ℝ) (hb : 0 < b) :
  let Δ₊ := f (x + b) - f x,
      Δ₋ := f (x - b) - f x in
  (Δ₊ = 3 * b * x^2 + 3 * b^2 * x + b^3 - 2 * b) ∧
  (Δ₋ = -3 * b * x^2 + 3 * b^2 * x - b^3 + 2 * b) := by
  sorry

end change_in_expression_l621_621407


namespace exists_convex_polyhedron_with_equal_edges_and_diagonals_l621_621026

/-
  Prove that there exists a convex polyhedron with the number of edges equal to the number of diagonals.

  Conditions:
  - A diagonal of a polyhedron is a line segment connecting two vertices that do not lie on the same face.
-/
theorem exists_convex_polyhedron_with_equal_edges_and_diagonals :
  ∃ (P : Polyhedron), P.is_convex ∧ P.edges = P.diagonals :=
sorry

end exists_convex_polyhedron_with_equal_edges_and_diagonals_l621_621026


namespace race_participants_least_number_l621_621499

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l621_621499


namespace probability_midpoint_in_U_l621_621422

-- Defining the set U of points with certain coordinate constraints
def U : set (ℤ × ℤ × ℤ) := {p | (0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 ∧ 0 ≤ p.3 ∧ p.3 ≤ 3)}

-- Define the probability calculation of the midpoint condition
theorem probability_midpoint_in_U :
  let count_valid_pairs : ℕ := 272
  let total_pairs : ℕ := 1128
  probability := (count_valid_pairs : ℚ) / total_pairs 
  in probability = (34 : ℚ) / 141 := by
  sorry

end probability_midpoint_in_U_l621_621422


namespace average_height_is_correct_l621_621029

def height_zara : ℝ := 64
def height_brixton : ℝ := height_zara
def height_zora : ℝ := height_brixton - 8
def height_itzayana : ℝ := height_zora + 4
def height_jaxon : ℝ := 170 / 2.54
def height_leo : ℝ := 1.5 * height_itzayana
def height_dora : ℝ := height_leo - 3.75

def total_height : ℝ := height_zara + height_brixton + height_zora + height_itzayana + height_jaxon + height_leo + height_dora
def average_height : ℝ := total_height / 7

theorem average_height_is_correct : average_height = 69.45 := 
by
  unfold average_height
  unfold total_height
  unfold height_zara
  unfold height_brixton
  unfold height_zora
  unfold height_itzayana
  unfold height_jaxon
  unfold height_leo
  unfold height_dora
  sorry

end average_height_is_correct_l621_621029


namespace no_real_solution_exists_l621_621481

theorem no_real_solution_exists:
  ¬ ∃ (x y z : ℝ), (x ^ 2 + 4 * y * z + 2 * z = 0) ∧
                   (x + 2 * x * y + 2 * z ^ 2 = 0) ∧
                   (2 * x * z + y ^ 2 + y + 1 = 0) :=
by
  sorry

end no_real_solution_exists_l621_621481


namespace quadratic_root_property_l621_621796

theorem quadratic_root_property (a x1 x2 : ℝ) 
  (h_eq : ∀ x, a * x^2 - (3 * a + 1) * x + 2 * (a + 1) = 0)
  (h_distinct : x1 ≠ x2)
  (h_relation : x1 - x1 * x2 + x2 = 1 - a) : a = -1 :=
sorry

end quadratic_root_property_l621_621796


namespace num_bad_arrangements_l621_621981
open List

def is_bad_arrangement (l : List ℕ) : Prop :=
  let sums := foldl (λ A i, A ∪ (List.powerset l).filter (λ s, s.length > 0 ).map sum ) ∅ (List.range l.length)
  ∀ n, n ∈ (Finset.range 16) → n ∉ sums

theorem num_bad_arrangements : 
  (Finset.filter is_bad_arrangement (Equiv.List.ListEquivFinset (List![$1, $2, $3, $4, $5]))) := 
  2 := 
sorry

end num_bad_arrangements_l621_621981


namespace inequality_triangle_areas_l621_621472

theorem inequality_triangle_areas (a b c α β γ : ℝ) (hα : α = 2 * Real.sqrt (b * c)) (hβ : β = 2 * Real.sqrt (a * c)) (hγ : γ = 2 * Real.sqrt (a * b)) : 
  a / α + b / β + c / γ ≥ 3 / 2 := 
by
  sorry

end inequality_triangle_areas_l621_621472


namespace sum_of_digits_b_n_l621_621435

def a (n : ℕ) : ℕ := (10^ (2 * n) - 1) / 9 -- formula for a_n given 2n nines

def b (n : ℕ) : ℕ := (List.range (n + 1)).sum (λ i => a i) -- sum of all a_i from 0 to n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum -- computes the sum of digits of n

theorem sum_of_digits_b_n (n : ℕ) : sum_of_digits (b n) = 9 * 2^n := by
  sorry

end sum_of_digits_b_n_l621_621435


namespace polynomial_value_l621_621848

theorem polynomial_value
  (x : ℝ)
  (h : x^2 + 2 * x - 2 = 0) :
  4 - 2 * x - x^2 = 2 :=
by
  sorry

end polynomial_value_l621_621848


namespace prob_floor_log_eq_l621_621430

noncomputable def probability_eq : Prop :=
  let probability : ℝ := (set.Ioo (0 : ℝ) 1).filter 
                            (λ x, Int.floor (Real.log (5 * x) / Real.log 10) = Int.floor (Real.log x / Real.log 10)).measure 
                            (set.Ioo (0 : ℝ) 1).measure in
  probability = 1 / 9

theorem prob_floor_log_eq : probability_eq :=
by sorry

end prob_floor_log_eq_l621_621430


namespace arith_seq_ratios_l621_621361

theorem arith_seq_ratios (a b : ℕ → ℚ) (A B : ℕ → ℚ)
  (h1 : ∀ n, A n = ∑ i in range n, a i)
  (h2 : ∀ n, B n = ∑ i in range n, b i)
  (h3 : ∀ n, A n / B n = (4 * n + 1) / (8 * n + 3)) :
  (a 10 / b 10 = 1 / 2) ∧
  (a 10 / a 7 = 77 / 53) ∧
  (b 10 / b 7 = 155 / 107) := by
  sorry

end arith_seq_ratios_l621_621361


namespace mother_twice_age_2040_l621_621451

-- Definitions
def alex_age_2010 : ℕ := 10
def mother_age_2010 : ℕ := 5 * alex_age_2010

def year_when_mother_twice_age (y : ℕ) : Prop :=
  mother_age_2010 + y = 2 * (alex_age_2010 + y)

-- Statement of the problem
theorem mother_twice_age_2040 : ∃ y, year_when_mother_twice_age y ∧ (2010 + y = 2040) :=
by {
  use 30,
  split,
  {
    calc mother_age_2010 + 30
        = 50 + 30 : rfl
    ... = 80 : rfl
    ... = 2 * (alex_age_2010 + 30) : by simp [alex_age_2010],
  },
  {
    simp,
  },
  sorry
}

end mother_twice_age_2040_l621_621451


namespace proof_problem_l621_621328

def pos_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
∀ n, 4 * S n = (a n + 1) ^ 2

def sequence_condition (a : ℕ → ℝ) : Prop :=
a 0 = 1 ∧ ∀ n, a (n + 1) - a n = 2

def sum_sequence_T (a : ℕ → ℝ) (T : ℕ → ℝ) :=
∀ n, T n = (1 - 1 / (2 * n + 1))

def range_k (T : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n, T n ≥ k → k ≤ 2 / 3

theorem proof_problem (a : ℕ → ℝ) (S T : ℕ → ℝ) (k : ℝ) :
  pos_sequence a S → sequence_condition a → sum_sequence_T a T → range_k T k :=
by sorry

end proof_problem_l621_621328


namespace no_positive_abc_exists_l621_621744

theorem no_positive_abc_exists 
  (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h1 : b^2 ≥ 4 * a * c)
  (h2 : c^2 ≥ 4 * b * a)
  (h3 : a^2 ≥ 4 * b * c)
  : false :=
sorry

end no_positive_abc_exists_l621_621744


namespace num_players_in_chess_tournament_l621_621389

theorem num_players_in_chess_tournament (p1 points : Nat) (h1 : points = 1979 ∨ points = 1980 ∨ points = 1984 ∨ points = 1985)
    (h2 : ∃ (n : ℕ), 2*n*(n-1)=1984 ∨ 2*n*(n-1)=1980): (p1 = 45) :=
begin
  sorry
end

end num_players_in_chess_tournament_l621_621389


namespace race_minimum_participants_l621_621516

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l621_621516


namespace markup_rate_l621_621036

theorem markup_rate (S : ℝ) (hS : S = 8.00) (profit_rate : ℝ) (expense_rate : ℝ)
  (h_profit_rate : profit_rate = 0.20) (h_expense_rate : expense_rate = 0.10) :
  let C := 0.70 * S in
  ((S - C) / C) * 100 = 42.857 :=
by
  sorry

end markup_rate_l621_621036


namespace parking_problem_l621_621578

noncomputable def count_valid_sequences (n : ℕ) : ℕ :=
  (n + 1) ^ (n - 1)

theorem parking_problem (n : ℕ) (a : fin n → ℕ) :
  (∃ s : fin n → fin n, (∀ i, s i ∈ (fin n).elems) ∧
    (∀ i, ∃! j, s j = some (a i)) ∧
    (∀ i j, i ≠ j → s i ≠ s j)) → 
  count_valid_sequences n = (n + 1) ^ (n - 1) :=
sorry

end parking_problem_l621_621578


namespace max_elements_l621_621048

def T : Set ℕ := { x | 1 ≤ x ∧ x ≤ 2023 }

theorem max_elements (T : Set ℕ) (h₁ : ∀ (a b : ℕ), a ∈ T → b ∈ T → (a ≠ b → a ≠ b + 5 ∧ a ≠ b + 8)) :
  ∃ (n : ℕ), n = 780 ∧ ∀ (S : Set ℕ), (S ⊆ T) → (∀ (a b : ℕ), a ∈ S → b ∈ S → (a ≠ b → a ≠ b + 5 ∧ a ≠ b + 8)) → S.card ≤ 780 :=
sorry

end max_elements_l621_621048


namespace cos_YXW_l621_621901

-- Basic definitions for the problem.
variables {XYZ: Triangle} {X Y Z W: Point}
hypothesis hXY : distance XY = 4
hypothesis hXZ : distance XZ = 5
hypothesis hYZ : distance YZ = 7
hypothesis hW_on_YZ : W ∈ segment YZ
hypothesis hXW_bisects_YXZ : angle_bisector X W Y Z

-- Statement of the problem (proof goal).
theorem cos_YXW : cos (angle Y X W) = (sqrt 10) / 5 :=
sorry

end cos_YXW_l621_621901


namespace average_waiting_time_l621_621698

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l621_621698


namespace find_counterfeit_coin_in_four_weighings_l621_621471

/-- 
There exist four coins {A, B, C, D}.
Among them, three are genuine and weigh the same, while one is counterfeit and its weight is different.
A scale is available that can determine the exact total weight of two or more coins but not a single coin.
We can perform at most 4 weighings to find the counterfeit coin and determine whether it is lighter or heavier.
-/
theorem find_counterfeit_coin_in_four_weighings
  (coins : Fin 4 → ℝ)
  (h_genuine : ∃ w : ℝ, ∀ i j, i ≠ j ∧ i < 3 ∧ j < 3 → coins i = w)
  (h_counterfeit : ∃ i, i < 4 ∧ ∀ w, coins i ≠ w)
  (h_scale : ∀ A B, (0 < Finset.card (A ∪ B)) → Finset.sum A coins = Finset.sum B coins ↔ coins A = coins B) :
  ∃ i, i < 4 ∧ ∀ j, j ≠ i → coins i ≠ coins j ∧ (coins i > coins j ∨ coins i < coins j) ∧ j < 4 ∧ i < 4 :=
sorry

end find_counterfeit_coin_in_four_weighings_l621_621471


namespace minimum_participants_l621_621492

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l621_621492


namespace road_trip_total_miles_l621_621588

theorem road_trip_total_miles (tracy_miles michelle_miles katie_miles : ℕ) (h_michelle : michelle_miles = 294)
    (h_tracy : tracy_miles = 2 * michelle_miles + 20) (h_katie : michelle_miles = 3 * katie_miles):
  tracy_miles + michelle_miles + katie_miles = 1000 :=
by
  sorry

end road_trip_total_miles_l621_621588


namespace knights_on_island_l621_621458

-- Definitions based on conditions
inductive Inhabitant : Type
| knight : Inhabitant
| knave : Inhabitant

open Inhabitant

def statement_1 (inhabitant : Inhabitant) : Prop :=
inhabitant = knight

def statement_2 (inhabitant1 inhabitant2 : Inhabitant) : Prop :=
inhabitant1 = knight ∧ inhabitant2 = knight

def statement_3 (inhabitant1 inhabitant2 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave)) / 2 ≥ 0.5

def statement_4 (inhabitant1 inhabitant2 inhabitant3 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave) + ↑(inhabitant3 = knave)) / 3 ≥ 0.65

def statement_5 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 : Inhabitant) : Prop :=
(↑(inhabitant1 = knight) + ↑(inhabitant2 = knight) + ↑(inhabitant3 = knight) + ↑(inhabitant4 = knight)) / 4 ≥ 0.5

def statement_6 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 inhabitant5 : Inhabitant) : Prop :=
(↑(inhabitant1 = knave) + ↑(inhabitant2 = knave) + ↑(inhabitant3 = knave) + ↑(inhabitant4 = knave) + ↑(inhabitant5 = knave)) / 5 ≥ 0.4

def statement_7 (inhabitant1 inhabitant2 inhabitant3 inhabitant4 inhabitant5 inhabitant6 : Inhabitant) : Prop :=
(↑(inhabitant1 = knight) + ↑(inhabitant2 = knight) + ↑(inhabitant3 = knight) + ↑(inhabitant4 = knight) + ↑(inhabitant5 = knight) + ↑(inhabitant6 = knight)) / 6 ≥ 0.65

-- Lean Statement
theorem knights_on_island (inhabitants : Fin 7 → Inhabitant) :
  (∀ i, (inhabitants i = knight ↔ (i = 0) ∨ (i = 1) ∨ (i = 4) ∨ (i = 5) ∨ (i = 6))) → 5 :=
by
  sorry

end knights_on_island_l621_621458


namespace f_g_2_eq_36_l621_621870

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 4 * x - 2

theorem f_g_2_eq_36 : f (g 2) = 36 :=
by
  sorry

end f_g_2_eq_36_l621_621870


namespace total_digits_of_first_2500_odd_integers_l621_621599

theorem total_digits_of_first_2500_odd_integers : 
  ∑ k in (Finset.filter odd (Finset.range 5000)), Nat.digits 10 k = 9445 :=
by
  -- We need to relate the counts of one-digit, two-digit, three-digit, and four-digit odd integers
  -- to the final sum of the digits across these categories.

  -- Define ranges and their counts.
  let one_digit_odds := 5 -- odd numbers from 1 to 9
  let two_digit_odds := 45 -- odd numbers from 11 to 99
  let three_digit_odds := 450 -- odd numbers from 101 to 999
  let four_digit_odds := 2000 -- odd numbers from 1001 to 4999

  -- Define their contributions to the digit counts
  let digits_one_digit := one_digit_odds * 1
  let digits_two_digits := two_digit_odds * 2
  let digits_three_digits := three_digit_odds * 3
  let digits_four_digits := four_digit_odds * 4

  -- Calculate the total number of digits
  let total_digits := digits_one_digit + digits_two_digits + digits_three_digits + digits_four_digits

  -- The summary of all digits must equal 9445
  have : total_digits = 9445,
    from sorry -- The proof would involve sum calculations as in the solution

  exact this

end total_digits_of_first_2500_odd_integers_l621_621599


namespace cartesian_equation_of_curve_general_equation_of_line_distance_equal_product_of_segments_main_theorem_l621_621009

theorem cartesian_equation_of_curve (rho theta : ℝ) (cond : rho * sin(theta)^2 = 2 * cos(theta)) :
    ∃ x y : ℝ, y^2 = 2x ∧ x = rho * cos(theta) ∧ y = rho * sin(theta) := 
by { 
    use [rho * cos theta, rho * sin theta],
    split,
    { sorry },  -- Proof steps omitted
    split,
    { refl },
    { refl }
}

theorem general_equation_of_line (t : ℝ) :
    let x := -2 - (real.sqrt 2 / 2) * t,
        y := -4 - (real.sqrt 2 / 2) * t
    in x - y - 2 = 0 :=
by {
    intros,
    unfold x y,
    linarith
}

theorem distance_equal_product_of_segments (PA PB AB : ℝ) (cond : PA * PB = AB^2) :
    PA * PB = AB^2 :=
by {
    exact cond
}

theorem main_theorem (rho theta t : ℝ) (x y : ℝ) (cond1 : rho * sin(theta)^2 = 2 * cos(theta))
    (cond2 : x = -2 - (real.sqrt 2 / 2) * t) (cond3 : y = -4 - (real.sqrt 2 / 2) * t)(
    PA PB AB : ℝ) (cond4 : PA * PB = AB^2) :
    (∃ x y : ℝ, y^2 = 2x ∧ x = rho * cos(theta) ∧ y = rho * sin(theta)) ∧
    (x - y - 2 = 0) ∧ (PA * PB = AB^2) := 
by {
    split,
    { exact cartesian_equation_of_curve rho theta cond1 },
    split,
    { exact general_equation_of_line t },
    { exact distance_equal_product_of_segments PA PB AB cond4 }
}

end cartesian_equation_of_curve_general_equation_of_line_distance_equal_product_of_segments_main_theorem_l621_621009


namespace valid_sequences_count_l621_621370

def no_two_consecutive_zeros (l : List ℕ) : Prop :=
  ∀ i, i < l.length - 1 → l.nth_le i sorry = 0 → l.nth_le (i + 1) sorry ≠ 0

def no_four_consecutive_ones (l : List ℕ) : Prop :=
  ∀ i, i < l.length - 3 → l.nth_le i sorry = 1 ∧ l.nth_le (i + 1) sorry = 1 ∧ l.nth_le (i + 2) sorry = 1 → l.nth_le (i + 3) sorry ≠ 1

def valid_sequence (l : List ℕ) : Prop :=
  l.length = 15 ∧ l.head = 0 ∧ l.last = 0 ∧ no_two_consecutive_zeros l ∧ no_four_consecutive_ones l

noncomputable def g (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 1
  else if n = 6 then 2
  else if n = 7 then 2
  else g (n - 2) + 2 * g (n - 3) + g (n - 4)

theorem valid_sequences_count : g 15 = 127 := 
  sorry

end valid_sequences_count_l621_621370


namespace correct_options_l621_621442

-- Define n as a natural number.
variable (n : ℕ)

-- a_n and b_n as given in the problem.
def a_n : ℝ := (ℚ.sqrt 5 + 2)^(2 * n + 1) - (ℚ.sqrt 5 - 2)^(2 * n + 1)
def b_n : ℝ := (ℚ.sqrt 5 - 2)^(2 * n + 1)

-- Statement for the Lean 4 proof problem.
theorem correct_options :
  (∀ n : ℕ, a_n n + b_n n = (ℚ.sqrt 5 + 2)^(2 * n + 1)) →
  (∀ n : ℕ, (∀ m : ℕ, m < n → a_n m < a_n n)) →
  (∀ n : ℕ, b_n n * (a_n n + b_n n) = 1) →
  (∀ n : ℕ, (1 - b_n n) * (a_n n + b_n n) ≠ 1) :=
by
  intros h1 h2 h3 h4
  -- Proof steps go here.
  sorry

end correct_options_l621_621442


namespace min_number_of_participants_l621_621507

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l621_621507


namespace count_odd_five_digit_numbers_l621_621312

theorem count_odd_five_digit_numbers : 
  let digits := {0, 1, 2, 3, 4}
  let valid_units := {1, 3}
  let valid_tens_of_thousands x := x ≠ 0 ∧ x ∈ digits
  ∀ n : ℕ,
  ∃ lst : List ℕ,
    lst.perm_of_digits ∧
    lst.no_duplicate_digits ∧
    lst.length = 5 ∧
    lst.units ∈ valid_units ∧
    lst.tens_of_thousands ∈ valid_tens_of_thousands ∧
    lst.count_odd_five_digit_numbers = 192 :=
by
  sorry

end count_odd_five_digit_numbers_l621_621312


namespace percentage_cleared_land_l621_621651

theorem percentage_cleared_land (T C : ℝ) (hT : T = 5999.999999999999)
                               (hC : C = 5400)
                               (h_conditions : 0.3 * C + 0.6 * C + 540 = C) :
  (C / T * 100) ≈ 90 :=
by sorry

end percentage_cleared_land_l621_621651


namespace number_of_type_A_storefronts_maximize_monthly_income_l621_621975

-- Define the conditions
def totalGreenhouseArea : ℝ := 2400
def numberOfStorefronts : ℕ := 80
def avgAreaA : ℝ := 28
def avgAreaB : ℝ := 20
def rentA : ℝ := 400
def rentB : ℝ := 360
def minAreaPercentage : ℝ := 0.8
def maxAreaPercentage : ℝ := 0.85
def leaseRateA : ℝ := 0.75
def leaseRateB : ℝ := 0.9

-- Definitions of inequalities based on conditions:
def totalAreaMin : ℝ := totalGreenhouseArea * minAreaPercentage
def totalAreaMax : ℝ := totalGreenhouseArea * maxAreaPercentage

-- Proving the number of type A storefronts is between 40 and 55
theorem number_of_type_A_storefronts (x : ℕ) (h : 0 ≤ x ∧ x ≤ numberOfStorefronts) 
  (h1 : avgAreaA * x + avgAreaB * (numberOfStorefronts - x) ≥ totalAreaMin)
  (h2 : avgAreaA * x + avgAreaB * (numberOfStorefronts - x) ≤ totalAreaMax) :
  40 ≤ x ∧ x ≤ 55 :=
sorry

-- Proving that to maximize monthly rental income, 40 type A storefronts should be built
theorem maximize_monthly_income (z : ℕ) (hz : 40 ≤ z ∧ z ≤ 55) :
  let W : ℝ := 400 * leaseRateA * z + 360 * leaseRateB * (numberOfStorefronts - z)
  in W = 25920 - 24 * 40 :=
sorry

end number_of_type_A_storefronts_maximize_monthly_income_l621_621975


namespace driver_total_distance_is_148_l621_621197

-- Definitions of the distances traveled according to the given conditions
def distance_MWF : ℕ := 12 * 3
def total_distance_MWF : ℕ := distance_MWF * 3
def distance_T : ℕ := 9 * 5 / 2  -- using ℕ for 2.5 hours as 5/2
def distance_Th : ℕ := 7 * 5 / 2

-- Statement of the total distance calculation
def total_distance_week : ℕ :=
  total_distance_MWF + distance_T + distance_Th

-- Theorem stating the total distance traveled during the week
theorem driver_total_distance_is_148 : total_distance_week = 148 := by
  sorry

end driver_total_distance_is_148_l621_621197


namespace ellipse_equation_point_p_on_fixed_line_l621_621439

open Real

-- Definitions for the ellipse and its properties
def ellipse (a : ℝ) : Prop := (∃ x y : ℝ, (x^2 / a^2) + (y^2 / (1 - a^2)) = 1)

def foci_on_x_axis (E : ℝ → Prop) : Prop := 
  ∀ a x y, E a → (2a^2 - 1) ≥ 0

def focal_distance_one (E : ℝ → Prop) : Prop := 
  ∀ a, E a → (2a^2 - 1)^0.5 = 1 / 2

-- Proof statements
theorem ellipse_equation (a : ℝ) (E : ℝ → Prop)
  (h1 : ellipse a) (h2 : foci_on_x_axis E) (h3 : focal_distance_one E) :
  E a := 
sorry

theorem point_p_on_fixed_line (a : ℝ) (E : ℝ → Prop) (x0 y0 : ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : ellipse a) (h2 : foci_on_x_axis E) (h3 : focal_distance_one E)
  (h4 : (0 < x0 ∧ 0 < y0) ∧ (∃ c, F1 = (-c, 0) ∧ F2 = (c, 0) ∧ (c = (2 * a^2 - 1)^0.5))
  (h5 : ∀ P : ℝ × ℝ, P = (x0, y0) → (∃ Q : ℝ × ℝ, Q.1 = 0 ∧ ∃ k1 k2, k1 * k2 = -1)) :
  x0 + y0 = 1 := 
sorry

end ellipse_equation_point_p_on_fixed_line_l621_621439


namespace prime_719_exists_l621_621791

theorem prime_719_exists (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) :
  (a^4 + b^4 + c^4 - 3 = 719) → Nat.Prime (a^4 + b^4 + c^4 - 3) := sorry

end prime_719_exists_l621_621791


namespace length_of_AB_in_30_60_90_triangle_l621_621899

theorem length_of_AB_in_30_60_90_triangle
  (A B C : Type) [is_triangle ABC]
  (angle_A : ∠BAC = π/6) -- π/6 radians is 30 degrees
  (BC : ℝ)
  (h : BC = 12) : 
  ∃ AB, AB = 6 * real.sqrt 3 := 
begin
  sorry -- fill in proof here
end

end length_of_AB_in_30_60_90_triangle_l621_621899


namespace exponent_multiplication_l621_621969

variable (a : ℝ) (m : ℤ)

theorem exponent_multiplication (a : ℝ) (m : ℤ) : a^(2 * m + 2) = a^(2 * m) * a^2 := 
sorry

end exponent_multiplication_l621_621969


namespace base_of_number_eq_two_l621_621874

theorem base_of_number_eq_two (k : ℕ) (hk : k ≤ 4) (hdiv : ∃ m, 435961 = 21^k * m) : 
  (k = 1 → ∃ x : ℕ, x^k - k^7 = 1 ∧ x = 2) :=
by {
  intro hk1,
  use 2,
  simp [hk1],
  norm_num,
  exact rfl,
}

end base_of_number_eq_two_l621_621874


namespace positive_difference_median_mode_l621_621837

def data : List ℕ := [21, 23, 23, 25, 25, 32, 32, 32, 40, 40, 47, 48, 51, 52, 53, 54, 63, 67, 68]

def mode (l : List ℕ) : ℕ :=
  l.foldr (λ x m, if l.count x > l.count m then x else m) 0

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  sorted.get! (sorted.length / 2)

theorem positive_difference_median_mode : 
  (data.median - data.mode).abs = 8 := by
  sorry

end positive_difference_median_mode_l621_621837


namespace friend_spent_l621_621183

theorem friend_spent (x you friend total: ℝ) (h1 : total = you + friend) (h2 : friend = you + 3) (h3 : total = 11) : friend = 7 := by
  sorry

end friend_spent_l621_621183


namespace number_of_different_outcomes_l621_621207

-- Definitions of conditions in the problem
def shots : ℕ := 8
def hits : ℕ := 4
def consecutive_hits : ℕ := 3

-- Main theorem statement based on the problem and solution
theorem number_of_different_outcomes : 
  (shots = 8) →
  (hits = 4) →
  (consecutive_hits = 3) →
  (number_of_ways : ℕ) = 20 :=
begin
  intros,
  sorry,
end

end number_of_different_outcomes_l621_621207


namespace tangent_line_at_origin_l621_621876

def f (x : ℝ) : ℝ := Real.exp x + x^2 - x + Real.sin x

theorem tangent_line_at_origin : 
  let f_deriv := λ x, Real.exp x + 2 * x - 1 + Real.cos x in
  let k := f_deriv 0 in
  let y₀ := f 0 in
  y₀ = 1 →
  k = 1 →
  ∀ x : ℝ, (y = k * x + y₀) → (y = x + 1) :=
sorry

end tangent_line_at_origin_l621_621876


namespace find_b_for_perpendicular_bisector_l621_621139

theorem find_b_for_perpendicular_bisector :
  (∃ b : ℝ, ∀ P : ℝ × ℝ, (P = (3, 6)) → (P.1 + P.2 = b)) →
  b = 9 :=
by
  intro h
  cases h with b hb
  specialize hb (3, 6) rfl
  rw [prod.mk.eta (3, 6), add_comm] at hb
  exact hb

end find_b_for_perpendicular_bisector_l621_621139


namespace stuart_initial_marbles_is_56_l621_621222

-- Define the initial conditions
def betty_initial_marbles : ℕ := 60
def percentage_given_to_stuart : ℚ := 40 / 100
def stuart_marbles_after_receiving : ℕ := 80

-- Define the calculation of how many marbles Betty gave to Stuart
def marbles_given_to_stuart := (percentage_given_to_stuart * betty_initial_marbles)

-- Define the target: Stuart's initial number of marbles
def stuart_initial_marbles := stuart_marbles_after_receiving - marbles_given_to_stuart

-- Main theorem stating the problem
theorem stuart_initial_marbles_is_56 : stuart_initial_marbles = 56 :=
by 
  sorry

end stuart_initial_marbles_is_56_l621_621222


namespace odd_n_one_dry_even_n_all_hit_l621_621096

theorem odd_n_one_dry (n : ℕ) (h_odd : n % 2 = 1) (h_distinct : ∀ i j : ℕ, i ≠ j → some_distance_measure i j ≠ 0) : 
  ∃ i : ℕ, i < n ∧ ∀ j : ℕ, j < n → some_distance_measure i j = 0 → j = i :=
sorry

theorem even_n_all_hit (n : ℕ) (h_even : n % 2 = 0) (h_distinct : ∀ i j : ℕ, i ≠ j → some_distance_measure i j ≠ 0) : 
  ∃ (f : ℕ → ℕ), (∀ i : ℕ, i < n → f i < n) ∧ (∀ i : ℕ, i < n → f i ≠ i ) :=
sorry

end odd_n_one_dry_even_n_all_hit_l621_621096


namespace average_of_last_four_numbers_l621_621550

/-- 
  Given:
  - avg10: The average of 10 numbers is 210.
  - avg5: The average of the first 5 numbers is 40.
  - middle: The middle number is 1100.
  Prove that the average of the last 4 numbers is 200.
-/
theorem average_of_last_four_numbers 
  (avg10 : Real)
  (avg5 : Real)
  (middle : Real)
  (sum10 : real → real) 
  (sum5 : real → real) 
  : (4 * ((sum10 (10 * avg10) - sum5 (5 * avg5) - middle) / 4)) = 200 := 
by
  sorry

end average_of_last_four_numbers_l621_621550


namespace leaf_raking_earnings_l621_621717

variable {S M L P : ℕ}

theorem leaf_raking_earnings (h1 : 5 * 4 + 7 * 2 + 10 * 1 + 3 * 1 = 47)
                             (h2 : 5 * 2 + 3 * 1 + 7 * 1 + 10 * 2 = 40)
                             (h3 : 163 - 87 = 76) :
  5 * S + 7 * M + 10 * L + 3 * P = 76 :=
by
  sorry

end leaf_raking_earnings_l621_621717


namespace sarah_total_desserts_l621_621089

def michael_saves_cookies : ℕ := 5
def sarah_initial_cupcakes : ℕ := 9
def sarah_gives_fraction : ℝ := 1 / 3
def sarah_receives_cookies_from_michael : ℕ := michael_saves_cookies
def sarah_keeps_cupcakes : ℕ := (sarah_initial_cupcakes : ℝ * (1 - sarah_gives_fraction)).toNat

theorem sarah_total_desserts :
  sarah_receives_cookies_from_michael + sarah_keeps_cupcakes = 11 :=
by
  -- Proof goes here
  sorry

end sarah_total_desserts_l621_621089


namespace race_participants_minimum_l621_621510

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l621_621510


namespace side_length_of_square_on_hypotenuse_l621_621947

theorem side_length_of_square_on_hypotenuse 
  (PQ PR : ℝ) (hPQ : PQ = 9) (hPR : PR = 12) :
  let s := sqrt (38) / 19 in 
  ∃ s, 
    (QR : ℝ) (hQR : QR = sqrt (PQ^2 + PR^2))
    (area_triangle_PQR : ℝ) (hPQR_area : area_triangle_PQR = 0.5 * PQ * PR) 
    (area_square : ℝ) (h_square : area_square = s^2) 
    (area_ratio : ℝ) (h_area_ratio : area_ratio = area_square / area_triangle_PQR) 
    (side_ratio : ℝ) (h_side_ratio : side_ratio = s / QR),
    side_ratio = s / QR ∧ s = sqrt (38) / 19 :=
begin
  sorry
end

end side_length_of_square_on_hypotenuse_l621_621947


namespace token_game_sum_l621_621107

theorem token_game_sum
  (num_participants : ℕ)
  (t1_initial t2_initial t3_initial : ℤ)
  (press_square : ℤ → ℤ := λ x, x * x)
  (press_cube : ℤ → ℤ := λ x, x * x * x)
  (t1_final : ℤ := press_square (press_square t1_initial))
  (t2_final : ℤ := press_cube (press_cube t2_initial))
  (t3_final : ℤ := press_square t3_initial) :
  num_participants = 50 →
  t1_initial = 2 →
  t2_initial = -2 →
  t3_initial = 0 →
  t1_final + t2_final + t3_final = -496 := 
by
  intros h1 h2 h3 h4
  dsimp [t1_final, t2_final, t3_final, press_square, press_cube]
  rw [h2, h3, h4]
  norm_num
  sorry

end token_game_sum_l621_621107


namespace bread_cost_l621_621238

open_locale real

/-- Assume Clare's mother gave her $47. Clare bought 4 loaves of bread and 2 cartons of milk,
each item cost the same amount, and she has $35 left. Prove that each loaf of bread cost $2. -/
theorem bread_cost (initial_money : ℝ) (loaves : ℕ) (cartons : ℕ) (remaining_money : ℝ)
  (same_cost : ℝ)
  (h1 : initial_money = 47)
  (h2 : loaves = 4)
  (h3 : cartons = 2)
  (h4 : remaining_money = 35)
  (hc : (loaves + cartons) * same_cost = initial_money - remaining_money) :
  same_cost = 2 := 
  sorry

end bread_cost_l621_621238


namespace hyperbola_has_eccentricity_sqrt3_plus_1_l621_621827

noncomputable def hyperbola_eccentricity (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
  let P ∈ { (x, y) | (x^2)/(a^2) - (y^2)/(b^2) = 1 ∧ x > 0 }, 
  let F1 : ℝ × ℝ := (-sqrt(a^2 + b^2), 0),
  let F2 : ℝ × ℝ := (sqrt(a^2 + b^2), 0),
  -- point P condition (OP + OF2) ⋅ F2P = 0
  let cond1 : Prop := (P.1, P.2) • (F2.1 - P.1, F2.2 - P.2) + (F2.1, F2.2) • (F2.1 - P.1, F2.2 - P.2) = 0,
  -- distance condition |PF1| = sqrt(3)|PF2|
  let cond2 : Prop := sqrt((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = sqrt(3) * sqrt((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
  in ∀ P, cond1 ∧ cond2 → (1 + b a = sqrt(3)+1)

theorem hyperbola_has_eccentricity_sqrt3_plus_1 (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  hyperbola_eccentricity a b a_pos b_pos :=
by
  sorry

end hyperbola_has_eccentricity_sqrt3_plus_1_l621_621827


namespace intersection_A_B_l621_621804

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end intersection_A_B_l621_621804


namespace stadium_length_l621_621302

theorem stadium_length
  (W : ℝ) (H : ℝ) (P : ℝ) (L : ℝ)
  (h1 : W = 18)
  (h2 : H = 16)
  (h3 : P = 34)
  (h4 : P^2 = L^2 + W^2 + H^2) :
  L = 24 :=
by
  sorry

end stadium_length_l621_621302


namespace ceil_neg_sqrt_frac_l621_621268

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l621_621268


namespace quadrilateral_area_l621_621485

theorem quadrilateral_area
  (A B C D : Type*)
  (dist : A → A → ℝ)
  (dist_axiom1 : dist B D = 5)
  (dist_axiom2 : dist A C = 5)
  (right_angle_B : ∀ x, dist B x = sqrt (dist B A ^ 2 + dist B C ^ 2))
  (right_angle_D : ∀ x, dist D x = sqrt (dist D A ^ 2 + dist D C ^ 2))
  (distinct_int_lengths : (dist A B = 3 ∨ dist A B = 4) ∧ (dist B C ≠ dist A B ∧ (dist B C = 3 ∨ dist B C = 4))) :
  (dist A C * dist B D) / 2 = 12 :=
by
  sorry

end quadrilateral_area_l621_621485


namespace batsman_average_runs_l621_621185

theorem batsman_average_runs 
    (avg30: ℕ) (avg30_eq: avg30 = 50)
    (avg15: ℕ) (avg15_eq: avg15 = 26) 
    (matches30: ℕ) (matches30_eq: matches30 = 30)
    (matches15: ℕ) (matches15_eq: matches15 = 15)
    : (avg30 * matches30 + avg15 * matches15) / (matches30 + matches15) = 42 :=
by
  rw [avg30_eq, avg15_eq, matches30_eq, matches15_eq]
  -- The step where the actual proof will take place but we are skipping it
  sorry

end batsman_average_runs_l621_621185


namespace range_of_a_l621_621970

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → (2 * x + 2 * (a - 1)) ≤ 0) → a ≤ -3 :=
by
  sorry

end range_of_a_l621_621970


namespace groupC_is_basis_l621_621224

-- Define the four sets of vectors
def vectorA1 : ℝ × ℝ := (0, 0)
def vectorA2 : ℝ × ℝ := (-2, 1)
def vectorB1 : ℝ × ℝ := (4, 6)
def vectorB2 : ℝ × ℝ := (6, 9)
def vectorC1 : ℝ × ℝ := (2, -5)
def vectorC2 : ℝ × ℝ := (-6, 4)
def vectorD1 : ℝ × ℝ := (2, -3)
def vectorD2 : ℝ × ℝ := (1/2, -3/4)

-- Define a predicate for collinearity of two vectors
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- The theorem to prove
theorem groupC_is_basis : ¬ collinear vectorC1 vectorC2 :=
by sorry

end groupC_is_basis_l621_621224


namespace find_difference_l621_621811

noncomputable def S : ℝ := ∑ n in range 50, (2^n) / ((2*n - 1) * (2*n + 1))
noncomputable def T : ℝ := ∑ n in range 49, (2^n) / (2*n + 1)

theorem find_difference : S - T = 1 - 2^49 / 99 := 
by 
  sorry

end find_difference_l621_621811


namespace P_2018_roots_l621_621635

def P : ℕ → (ℝ → ℝ)
| 0     := λ x, 1
| 1     := λ x, x
| (n+2) := λ x, x * (P (n+1) x) - (P n x)

theorem P_2018_roots : 
  ∃ S : Finset ℝ, S.card = 2018 ∧ ∀ x ∈ S, P 2018 x = 0 ∧ 
  ∀ x₁ x₂ ∈ S, x₁ ≠ x₂ → x₁ ≠ x₂ :=
begin
  sorry
end

end P_2018_roots_l621_621635


namespace P_2018_has_2018_distinct_real_roots_l621_621628
noncomputable theory

def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 2) := λ x, x * P (n + 1) x - P n x

theorem P_2018_has_2018_distinct_real_roots :
  (∃ xs : fin 2018 → ℝ, ∀ i j : fin 2018, i ≠ j → xs i ≠ xs j ∧ ∀ x : ℝ, P 2018 x = 0 ↔ ∃ k : fin 2018, x = xs k) :=
sorry

end P_2018_has_2018_distinct_real_roots_l621_621628


namespace difference_of_quarters_l621_621474

variables (n d q : ℕ)

theorem difference_of_quarters :
  (n + d + q = 150) ∧ (5 * n + 10 * d + 25 * q = 1425) →
  (∃ qmin qmax : ℕ, q = qmax - qmin ∧ qmax - qmin = 30) :=
by
  sorry

end difference_of_quarters_l621_621474


namespace penguins_more_than_sea_horses_l621_621715

-- Defining the conditions
def ratio_sea_horses : ℕ := 5
def ratio_penguins : ℕ := 11
def number_sea_horses : ℕ := 70

-- We need to prove that the difference in the number of penguins and sea horses is 84
theorem penguins_more_than_sea_horses :
  let number_penguins := (number_sea_horses / ratio_sea_horses) * ratio_penguins in
  number_penguins - number_sea_horses = 84 :=
by
  sorry

end penguins_more_than_sea_horses_l621_621715


namespace number_of_knights_is_five_l621_621463

section KnightsAndKnaves

inductive Inhabitant
| knight : Inhabitant
| knave : Inhabitant

open Inhabitant

variables (a1 a2 a3 a4 a5 a6 a7 : Inhabitant)

def tells_truth : Inhabitant → Prop
| knight := True
| knave := False

def statements : Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → ℕ → Prop
| a1 a2 _ _ _ _ _ 1 := (a1 = knight)
| a1 a2 _ _ _ _ _ 2 := (a1 = knight ∧ a2 = knight)
| a1 a2 a3 _ _ _ _ 3 := (a1 = knave ∨ a2 = knave)
| a1 a2 a3 a4 _ _ _ 4 := (a1 = knave ∨ a2 = knave ∨ a3 = knave)
| a1 a2 a3 a4 a5 _ _ 5 := (a1 = knight ∧ a2 = knight ∨ a1 = knave ∧ a2 = knave)
| a1 a2 a3 a4 a5 a6 _ 6 := (a1 = knave ∨ a2 = knave ∨ a3 = knave ∨ a4 = knave ∨ a5 = knave)
| a1 a2 a3 a4 a5 a6 a7 7 := (a1 = knight ∨ a2 = knight ∨ a3 = knight ∨ a4 = knight ∨ a5 = knight ∨ a6 = knight)

theorem number_of_knights_is_five (h1 : tells_truth a1 ↔ a1 = knight)
                                   (h2 : tells_truth a2 ↔ a2 = knight)
                                   (h3 : tells_truth a3 ↔ a3 = knight)
                                   (h4 : tells_truth a4 ↔ a4 = knight)
                                   (h5 : tells_truth a5 ↔ a5 = knight)
                                   (h6 : tells_truth a6 ↔ a6 = knight)
                                   (h7 : tells_truth a7 ↔ a7 = knight)
                                   (s1 : tells_truth a1 → statements a1 a2 a3 a4 a5 a6 a7 1)
                                   (s2 : tells_truth a2 → statements a1 a2 a3 a4 a5 a6 a7 2)
                                   (s3 : tells_truth a3 → statements a1 a2 a3 a4 a5 a6 a7 3)
                                   (s4 : tells_truth a4 → statements a1 a2 a3 a4 a5 a6 a7 4)
                                   (s5 : tells_truth a5 → statements a1 a2 a3 a4 a5 a6 a7 5)
                                   (s6 : tells_truth a6 → statements a1 a2 a3 a4 a5 a6 a7 6)
                                   (s7 : tells_truth a7 → statements a1 a2 a3 a4 a5 a6 a7 7)
                                   : (∀ i, (i = knight → tells_truth i ¬ ↔ i = knight)) → ∃ (n : ℕ), n = 5 := 
sorry

end KnightsAndKnaves

end number_of_knights_is_five_l621_621463


namespace select_six_friends_l621_621397

def is_connected {V : Type*} [Fintype V] (G : SimpleGraph V) :=
  ∀ u v, G.reachable u v

def friendship_property {V : Type*} [Fintype V] (G : SimpleGraph V) (selected : Finset V) :=
  ∀ v ∈ G.verts \ selected, ∃ u ∈ selected, G.adj u v ∨ ∃ w, G.adj u w ∧ G.adj w v

theorem select_six_friends (V : Type*) [Fintype V] (G : SimpleGraph V) (hV : Fintype.card V = 20)
    (conn : is_connected G) :
  ∃ (selected : Finset V), selected.card = 6 ∧ friendship_property G selected := 
sorry

end select_six_friends_l621_621397


namespace minimum_value_expression_l621_621925

theorem minimum_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 3) : 
  a^2 + 8 * a * b + 24 * b^2 + 16 * b * c + 6 * c^2 ≥ 54 :=
begin
  sorry
end

end minimum_value_expression_l621_621925


namespace incorrect_statement_l621_621936

/-- Conditions for meiosis and fertilization for determining the incorrect statement. -/
constants (A B D : Prop)
  
/-- After proving the conditions of meiosis and fertilization correctly, we need to prove that C is incorrect. -/
theorem incorrect_statement (A_correct : A) (B_correct : B) (D_correct : D) : ¬C :=
sorry

end incorrect_statement_l621_621936


namespace P_n_real_roots_P_2018_real_roots_l621_621624

noncomputable def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 1) := λ x, x * P n x - P (n - 1) x

theorem P_n_real_roots (n: ℕ) : ∃ r: ℕ, r ≡ n := sorry

theorem P_2018_real_roots : ∃ r: ℕ, r = 2018 := P_n_real_roots 2018

end P_n_real_roots_P_2018_real_roots_l621_621624


namespace adam_final_amount_l621_621675

def initial_savings : ℝ := 1579.37
def money_received_monday : ℝ := 21.85
def money_received_tuesday : ℝ := 33.28
def money_spent_wednesday : ℝ := 87.41

def total_money_received : ℝ := money_received_monday + money_received_tuesday
def new_total_after_receiving : ℝ := initial_savings + total_money_received
def final_amount : ℝ := new_total_after_receiving - money_spent_wednesday

theorem adam_final_amount : final_amount = 1547.09 := by
  -- proof omitted
  sorry

end adam_final_amount_l621_621675


namespace nominations_distribution_l621_621153

theorem nominations_distribution :
  let nominations := 10
  let schools := 7
  (∀ s, 1 ≤ s) → (finset.calculate_combinations (nominations - schools + schools - 1) (schools - 1)) = 84 :=
begin
  -- conditions
  have h1 : nominations = 10 := rfl,
  have h2 : schools = 7 := rfl,
  have h3 : ∀ s, 1 ≤ s := by sorry, -- Since the condition each school receives at least one nomination must hold

  -- calculations
  have h4 : finset.calculate_combinations (nominations - schools + schools - 1) (schools - 1) = 84 := by sorry, -- Using the provided steps

  show (∀ s, 1 ≤ s) → (finset.calculate_combinations (nominations - schools + schools - 1) (schools - 1)) = 84,
end

end nominations_distribution_l621_621153


namespace day_of_250th_in_N_minus_1_is_saturday_l621_621904

-- Noncomputable theory since we are dealing with dates and days of the week.
noncomputable theory

-- Definitions for conditions in the problem
def is_friday (n : ℕ) : Prop := n % 7 = 5

-- Given conditions
axiom cond1 (N : ℕ) : is_friday 250
axiom cond2 (N : ℕ) : is_friday 150

-- Theorem to be proved
theorem day_of_250th_in_N_minus_1_is_saturday (N : ℕ) (h1 : is_friday 250) (h2 : is_friday 150) : 
    ((250 - 365) % 7 + 7) % 7 = 6 := by
  sorry

end day_of_250th_in_N_minus_1_is_saturday_l621_621904


namespace avg_waiting_time_waiting_time_equivalence_l621_621711

-- The first rod receives an average of 3 bites in 6 minutes
def firstRodBites : ℝ := 3 / 6
-- The second rod receives an average of 2 bites in 6 minutes
def secondRodBites : ℝ := 2 / 6
-- Together, they receive an average of 5 bites in 6 minutes
def combinedBites : ℝ := firstRodBites + secondRodBites

-- We need to prove the average waiting time for the first bite
theorem avg_waiting_time : combinedBites = 5 / 6 → (1 / combinedBites) = 6 / 5 :=
by
  intro h
  rw h
  sorry

-- Convert 1.2 minutes into minutes and seconds
def minutes := 1
def seconds := 12

-- Prove the equivalence of waiting time in minutes and seconds
theorem waiting_time_equivalence : (6 / 5 = minutes + seconds / 60) :=
by
  simp [minutes, seconds]
  sorry

end avg_waiting_time_waiting_time_equivalence_l621_621711


namespace interval_of_monotonic_decrease_range_of_k_l621_621355

-- Part (I) Definitions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := - (1 / 2) * a * x^2 + (1 + a) * x - Real.log x

-- Theorem for part (I)
theorem interval_of_monotonic_decrease (a : ℝ) (h : a > 0) : 
  (a ∈ Ioo 0 1 → (∀ x ∈ Ioo 0 1, f a' x ∈ Ioo 0 1) ∧ (∀ x ∈ Ioi (1 / a), f a' x ∈ Ioi 1))
  ∧ (a = 1 → ∀ x ∈ Ioi 0, f a' x ∈ Ioi 0)
  ∧ (a ∈ Ioi 1 → (∀ x ∈ Ioo 0 (1 / a), f a' x ∈ Ioo 0 (1 / a)) ∧ (∀ x ∈ Ioi 1, f a' x ∈ Ioi 1)) := by sorry

-- Part (II) Definitions
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := (x * f 0 x) - k * (x + 2) + 2

-- Theorem for part (II)
theorem range_of_k (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ ∈ Icc (1/2) +∞ ) ∧ (x₂ ∈ Icc (1/2) +∞ ) ∧ g x₁ k = 0 ∧ g x₂ k = 0) ↔ 
  (1 < k ∧ k ≤ (9/10 + Real.log 2 / 5)) := by sorry

end interval_of_monotonic_decrease_range_of_k_l621_621355


namespace tanAPlusPiOver4_eq_neg2plusSqrt3_l621_621342

noncomputable def findTanAPlusPiOver4 (α : ℝ) (h : α ∈ (Ioo (π / 2) π)) (h1 : sin (2 * α) + sin α = 0) : ℝ :=
  tan (α + π / 4)

theorem tanAPlusPiOver4_eq_neg2plusSqrt3 (α : ℝ) (h : α ∈ (Ioo (π / 2) π)) (h1 : sin (2 * α) + sin α = 0) :
  findTanAPlusPiOver4 α h h1 = -2 + Real.sqrt 3 :=
sorry

end tanAPlusPiOver4_eq_neg2plusSqrt3_l621_621342


namespace count_correct_statements_l621_621687

-- Definitions of conditions from the problem
def statement1 :=
  ¬ (∀ x : ℝ, x^2 - 3 * x - 2 ≥ 0) = (∃ x₀ : ℝ, x₀^2 - 3 * x₀ - 2 < 0)

def statement2 (a b β : Type) :=
  ¬ (a ∥ b) ∧ ¬ (b ∥ β) → (a ∥ β ∨ b ⊆ β)

def statement3 :=
  ∃ m : ℝ, ∀ x : ℝ, (f : ℝ → ℝ) (f x = m * x^(m^2 + 2*m)) → monotone_on (set.Ioi 0) f

def statement4 :=
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
  ∀ x y : ℝ,
  (x₂ - x₁) * (y - y₁) - (y₂ - y₁) * (x - x₁) = 0

-- The overall problem statement
theorem count_correct_statements : 
  (¬ statement1 ∧ ¬ statement2 ℝ ℝ ℝ ∧ statement3 ∧ statement4) ↔ 2 = 2 := 
by sorry

end count_correct_statements_l621_621687


namespace find_x_plus_y_l621_621438

variable {x y : ℝ}
def a := (x, 1 : ℝ)
def b := (2, y : ℝ)
def a_plus_2b := (a.1 + 2 * b.1, a.2 + 2 * b.2)

theorem find_x_plus_y (h : a_plus_2b = (5, -3)) : x + y = -1 :=
sorry

end find_x_plus_y_l621_621438


namespace evaluate_ceil_of_neg_sqrt_l621_621260

-- Define the given expression and its value computation
def given_expression : ℚ := -real.sqrt (64 / 9)

-- Define the expected answer
def expected_answer : ℤ := -2

-- State the theorem to be proven
theorem evaluate_ceil_of_neg_sqrt : (Int.ceil given_expression) = expected_answer := sorry

end evaluate_ceil_of_neg_sqrt_l621_621260


namespace d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l621_621120

variable (c d : ℕ)

-- Conditions: c is a multiple of 4 and d is a multiple of 8
def is_multiple_of_4 (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k
def is_multiple_of_8 (n : ℕ) : Prop := ∃ k : ℕ, n = 8 * k

-- Statements to prove:

-- A. d is a multiple of 4
theorem d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 d :=
sorry

-- B. c - d is a multiple of 4
theorem c_minus_d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 (c - d) :=
sorry

-- D. c - d is a multiple of 2
theorem c_minus_d_is_multiple_of_2 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : ∃ k : ℕ, c - d = 2 * k :=
sorry

end d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l621_621120


namespace evaporation_period_l621_621646

theorem evaporation_period
  (total_water : ℕ)
  (daily_evaporation_rate : ℝ)
  (percentage_evaporated : ℝ)
  (evaporation_period_days : ℕ)
  (h_total_water : total_water = 10)
  (h_daily_evaporation_rate : daily_evaporation_rate = 0.006)
  (h_percentage_evaporated : percentage_evaporated = 0.03)
  (h_evaporation_period_days : evaporation_period_days = 50):
  (percentage_evaporated * total_water) / daily_evaporation_rate = evaporation_period_days := by
  sorry

end evaporation_period_l621_621646


namespace simplify_expression_l621_621951

theorem simplify_expression (x y : ℝ) : 7 * x + 8 * y - 3 * x + 4 * y + 10 = 4 * x + 12 * y + 10 :=
by
  sorry

end simplify_expression_l621_621951


namespace min_dist_on_circle_l621_621983

theorem min_dist_on_circle :
  let P (θ : ℝ) := (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)
  let M := (0, 2)
  ∃ θ_min : ℝ, 
    (∀ θ : ℝ, 
      let dist (P : ℝ × ℝ) (M : ℝ × ℝ) := Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)
      dist (P θ) M ≥ dist (P θ_min) M) ∧ 
    dist (P θ_min) M = Real.sqrt 5 - 1 := sorry

end min_dist_on_circle_l621_621983


namespace solve_domino_problem_l621_621192

-- Definition of a domino as a pair of digits (non-negative integers from 0 to 6)
structure Domino where
  left : Nat
  right : Nat
  h_left : left < 7
  h_right : right < 7

-- Definition of a multiplication column
structure MultiplicationColumn where
  multiplicand : Nat
  multiplier : Nat
  product : Nat
  valid_digits : multiplicand < 100 ∧ multiplier < 10 ∧ product < 100 ∧
                 (∀ d, d ∈ [multiplicand, multiplier, product] → (d / 10) < 7 ∧ (d % 10) < 7)

-- Assuming we have a list of 28 domino pieces
constant dominos : List Domino
constant h_length : dominos.length = 28

-- Assuming we have a function that checks if a given set of dominos can form a multiplication column
constant can_form_column : List Domino → MultiplicationColumn → Prop

-- Prove that 28 domino pieces can form 7 valid multiplication columns
theorem solve_domino_problem :
  ∃ columns : List MultiplicationColumn,
    columns.length = 7 ∧
    List.all columns (λ col, can_form_column dominos col) :=
sorry

end solve_domino_problem_l621_621192


namespace spherical_to_rectangular_correct_l621_621734

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_correct : spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) :=
by
  sorry

end spherical_to_rectangular_correct_l621_621734


namespace interval_of_monotonic_decrease_l621_621972

def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

noncomputable def f' (x : ℝ) : ℝ := x - 1 / x

theorem interval_of_monotonic_decrease :
  setOf (λ x : ℝ, 0 < x ∧ x < 1) = { x : ℝ | f' x < 0 } :=
sorry

end interval_of_monotonic_decrease_l621_621972


namespace area_of_region_R_l621_621118

theorem area_of_region_R (A B C D : Point) (ABCD_square : isSquare A B C D)
  (side_length : dist A B = 10) (angle_B : angle A B C = 90)
  : area(regionR A B C D) = 25 := 
sorry

end area_of_region_R_l621_621118


namespace packs_needed_is_six_l621_621110

variable (l_bedroom l_bathroom l_kitchen l_basement : ℕ)

def total_bulbs_needed := l_bedroom + l_bathroom + l_kitchen + l_basement
def garage_bulbs_needed := total_bulbs_needed / 2
def total_bulbs_with_garage := total_bulbs_needed + garage_bulbs_needed
def packs_needed := total_bulbs_with_garage / 2

theorem packs_needed_is_six
    (h1 : l_bedroom = 2)
    (h2 : l_bathroom = 1)
    (h3 : l_kitchen = 1)
    (h4 : l_basement = 4) :
    packs_needed l_bedroom l_bathroom l_kitchen l_basement = 6 := by
  sorry

end packs_needed_is_six_l621_621110


namespace number_less_than_neg_one_is_neg_two_l621_621655

theorem number_less_than_neg_one_is_neg_two : ∃ x : ℤ, x = -1 - 1 ∧ x = -2 := by
  sorry

end number_less_than_neg_one_is_neg_two_l621_621655


namespace area_of_pentagon_FGHIJ_l621_621241

-- Define the convex pentagon with given side lengths and inscribed circle
structure Pentagon :=
  (FG GH HI IJ JF : ℝ)
  (has_inscribed_circle : Prop)

-- Define the specific pentagon FGHIJ
def pentagon_FGHIJ : Pentagon :=
{ FG := 7,
  GH := 8,
  HI := 8,
  IJ := 8,
  JF := 9,
  has_inscribed_circle := true }

-- Define the theorem to prove the area of pentagon FGHIJ is 48
theorem area_of_pentagon_FGHIJ (P : Pentagon) (hP : P = pentagon_FGHIJ) : area P = 48 :=
begin
  sorry
end

end area_of_pentagon_FGHIJ_l621_621241


namespace average_speed_on_way_out_l621_621039

theorem average_speed_on_way_out (v : ℝ) : 
  (∀ (dist : ℝ) (timeTotal : ℝ) (backSpeed : ℝ), 
    dist = 72 ∧ timeTotal = 7 ∧ backSpeed = 18 → 
    let timeOut := dist / 2 / v in
    let timeBack := dist / 2 / backSpeed in
    timeOut + timeBack = timeTotal) → 
  v = 7.2 := 
by
  intros h
  specialize h 72 7 18
  simp at h
  sorry

end average_speed_on_way_out_l621_621039


namespace eiffel_tower_scale_model_height_l621_621225

theorem eiffel_tower_scale_model_height :
  let ratio := 1 / 25
  let actual_height := 1063
  round (actual_height * ratio) = 43 :=
by
  let ratio := 1 / 25
  let actual_height := 1063
  have model_height := actual_height * ratio
  have rounded_height := round model_height
  show rounded_height = 43
  sorry

end eiffel_tower_scale_model_height_l621_621225


namespace symmetry_with_y_eq_x_l621_621557

def f (x : ℝ) : ℝ := 3 ^ x

theorem symmetry_with_y_eq_x (x : ℝ) (h : x > 0) : 
  (∀ y, y = f x ↔ x = logBase 3 y) → f x = 3 ^ x :=
by
  intro h_sym
  sorry

end symmetry_with_y_eq_x_l621_621557


namespace remainder_29_169_1990_mod_11_l621_621596

theorem remainder_29_169_1990_mod_11 :
  (29 * 169 ^ 1990) % 11 = 7 :=
by
  sorry

end remainder_29_169_1990_mod_11_l621_621596


namespace sum_of_possible_n_l621_621905

-- Define the polynomial f in terms of x and n.
def f (x : ℤ) (n : ℤ) := 3 * x^3 - n * x - n - 2

-- Define the condition that f can be factored into two non-constant polynomials
def canBeFactored (n : ℤ) : Prop :=
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (3 * x^3 - n * x - n - 2 = (3 * x^2 + a * x + b) * (x - c))

-- Prove that the sum of all possible values of n is 192
theorem sum_of_possible_n : ∑ n in { n : ℤ | canBeFactored n }, n = 192 := by
  sorry

end sum_of_possible_n_l621_621905


namespace saturday_price_is_correct_l621_621453

-- Define Thursday's price
def thursday_price : ℝ := 50

-- Define the price increase rate on Friday
def friday_increase_rate : ℝ := 0.2

-- Define the discount rate on Saturday
def saturday_discount_rate : ℝ := 0.15

-- Calculate the price on Friday
def friday_price : ℝ := thursday_price * (1 + friday_increase_rate)

-- Calculate the discount amount on Saturday
def saturday_discount : ℝ := friday_price * saturday_discount_rate

-- Calculate the price on Saturday
def saturday_price : ℝ := friday_price - saturday_discount

-- Theorem stating the price on Saturday
theorem saturday_price_is_correct : saturday_price = 51 := by
  -- Definitions are already embedded into the conditions
  -- so here we only state the property to be proved.
  sorry

end saturday_price_is_correct_l621_621453


namespace no_real_solution_sqrt_eq_l621_621954

theorem no_real_solution_sqrt_eq (x : ℝ) : ¬ (∃ x : ℝ, sqrt (x + 4) - sqrt (x - 3) + 1 = 0) :=
by
  sorry

end no_real_solution_sqrt_eq_l621_621954


namespace sodium_hydride_reaction_l621_621369

theorem sodium_hydride_reaction (H2O NaH NaOH H2 : ℕ) 
  (balanced_eq : NaH + H2O = NaOH + H2) 
  (stoichiometry : NaH = H2O → NaOH = H2 → NaH = H2) 
  (h : H2O = 2) : NaH = 2 :=
sorry

end sodium_hydride_reaction_l621_621369


namespace point_on_line_l621_621575

theorem point_on_line (s : ℝ) : 
  (∃ b : ℝ, ∀ x y : ℝ, (y = 3 * x + b) → 
    ((2 = x ∧ y = 8) ∨ (4 = x ∧ y = 14) ∨ (6 = x ∧ y = 20) ∨ (35 = x ∧ y = s))) → s = 107 :=
by
  sorry

end point_on_line_l621_621575


namespace tan_alpha_sub_pi_over_8_l621_621864

theorem tan_alpha_sub_pi_over_8 (α : ℝ) (h : 2 * Real.tan α = 3 * Real.tan (Real.pi / 8)) :
  Real.tan (α - Real.pi / 8) = (5 * Real.sqrt 2 + 1) / 49 :=
by sorry

end tan_alpha_sub_pi_over_8_l621_621864


namespace jimin_rank_l621_621013

theorem jimin_rank (seokjin_rank : ℕ) (h1 : seokjin_rank = 4) (h2 : ∃ jimin_rank, jimin_rank = seokjin_rank + 1) : 
  ∃ jimin_rank, jimin_rank = 5 := 
by
  sorry

end jimin_rank_l621_621013


namespace horses_meet_days_l621_621895

theorem horses_meet_days (
  distance : ℕ := 1125,
  good_first_day : ℕ := 103,
  good_increase : ℕ := 13,
  mediocre_first_day : ℕ := 97,
  mediocre_decrease : ℝ := 0.5) :
  ∃ m : ℕ, 103 * m + (m * (m - 1) * 13) / 2 + 97 * m + (m * (m - 1) * (-0.5)) / 2 = 2 * distance ∧ m = 9 :=
by
  sorry

end horses_meet_days_l621_621895


namespace quadrilateral_area_l621_621486

theorem quadrilateral_area
  (A B C D : Type*)
  (dist : A → A → ℝ)
  (dist_axiom1 : dist B D = 5)
  (dist_axiom2 : dist A C = 5)
  (right_angle_B : ∀ x, dist B x = sqrt (dist B A ^ 2 + dist B C ^ 2))
  (right_angle_D : ∀ x, dist D x = sqrt (dist D A ^ 2 + dist D C ^ 2))
  (distinct_int_lengths : (dist A B = 3 ∨ dist A B = 4) ∧ (dist B C ≠ dist A B ∧ (dist B C = 3 ∨ dist B C = 4))) :
  (dist A C * dist B D) / 2 = 12 :=
by
  sorry

end quadrilateral_area_l621_621486


namespace one_less_than_neg_one_is_neg_two_l621_621654

theorem one_less_than_neg_one_is_neg_two : (-1 - 1 = -2) :=
by
  sorry

end one_less_than_neg_one_is_neg_two_l621_621654


namespace calculate_fg1_l621_621931

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem calculate_fg1 : f (g 1) = -1 :=
by {
  sorry
}

end calculate_fg1_l621_621931


namespace virginia_more_years_l621_621164

variable {V A D x : ℕ}

theorem virginia_more_years (h1 : V + A + D = 75) (h2 : D = 34) (h3 : V = A + x) (h4 : V = D - x) : x = 9 :=
by
  sorry

end virginia_more_years_l621_621164


namespace intersection_point_exists_l621_621988

variable {α : Type*} [EuclideanGeometry α]
variables (A B C D P Q X Y : α)
variable (ω : Circle α)

-- Conditions
variable h_inscribed : Inscribed ω A B C D
variable h_parallel : Parallel A D B C
variable h_incircle_ABC : Incircle A B C
variable h_incircle_ABD : Incircle A B D
variable h_touch_ABC_P : TouchesAt (Incircle A B C) B C P
variable h_touch_ABD_Q : TouchesAt (Incircle A B D) A D Q
variable h_midpoints_X : MidpointArc X ω B C
variable h_midpoints_Y : MidpointArc Y ω A D
variable h_not_contains : ¬(ArcContains ω A B X) ∧ ¬(ArcContains ω A B Y)

-- Question translated to proof problem
theorem intersection_point_exists :
  IntersectOn ω (Line X P) (Line Y Q) :=
sorry

end intersection_point_exists_l621_621988


namespace sum_a_b_is_95_l621_621878

-- Define the conditions
def product_condition (a b : ℕ) : Prop :=
  (a : ℤ) / 3 = 16 ∧ b = a - 1

-- Define the theorem to be proven
theorem sum_a_b_is_95 (a b : ℕ) (h : product_condition a b) : a + b = 95 :=
by
  sorry

end sum_a_b_is_95_l621_621878


namespace trapezoid_inequality_l621_621470

variables {A B C D E X : Point}
variables {AB AX : Real} -- represents lengths of segments AB and AX
variables {DE BX : Line} -- represents parallel lines DE and BX

-- Given conditions
axiom AB_eq_AX : AB = AX
axiom DE_parallel_BX : parallel DE BX

-- To prove
theorem trapezoid_inequality (AD CE BE ED : Real) :
  AD + CE ≥ BE + ED :=
by
  -- sorry is put here to indicate that the proof is not provided
  sorry

end trapezoid_inequality_l621_621470


namespace distinct_solutions_of_transformed_eq_l621_621134

open Function

variable {R : Type} [Field R]

def cubic_func (a b c d : R) (x : R) : R := a*x^3 + b*x^2 + c*x + d

noncomputable def three_distinct_roots {a b c d : R} (f : R → R)
  (h : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) : Prop :=
∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ f α = 0 ∧ f β = 0 ∧ f γ = 0

theorem distinct_solutions_of_transformed_eq
  {a b c d : R} (h : ∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ (cubic_func a b c d α) = 0 ∧ (cubic_func a b c d β) = 0 ∧ (cubic_func a b c d γ) = 0) :
  ∃ p q, p ≠ q ∧ (4 * (cubic_func a b c d p) * (3 * a * p + b) = (3 * a * p^2 + 2 * b * p + c)^2) ∧ 
              (4 * (cubic_func a b c d q) * (3 * a * q + b) = (3 * a * q^2 + 2 * b * q + c)^2) := sorry

end distinct_solutions_of_transformed_eq_l621_621134


namespace simple_interest_years_l621_621212

theorem simple_interest_years (P R : ℝ) (T : ℝ) :
  P = 2500 → (2500 * (R + 2) / 100 * T = 2500 * R / 100 * T + 250) → T = 5 :=
by
  intro hP h
  -- Note: Actual proof details would go here
  sorry

end simple_interest_years_l621_621212


namespace length_MN_is_correct_l621_621334

noncomputable def length_of_MN : ℝ := sorry

theorem length_MN_is_correct :
  ∀ (A : ℝ × ℝ) (slope_angle : ℝ) (parabola : ℝ × ℝ → Prop),
  A = (1, 0) →
  slope_angle = real.pi / 4 →
  parabola = (λ (p : ℝ × ℝ), p.2^2 = 2 * p.1) →
  ∃ M N : ℝ × ℝ,
    (∃ l : ℝ × ℝ → Prop, 
      l A ∧
      (∀ (P : ℝ × ℝ), l P → parabola P) ∧
      l M ∧ l N) ∧
    (∃ (x1 x2 : ℝ), 
      (M = (x1, x1 - 1)) ∧ 
      (N = (x2, x2 - 1)) ∧ 
      x1 + x2 = 4 ∧ 
      x1 * x2 = 1 ∧ 
      length_of_MN = 2 * real.sqrt 6) :=
sorry

end length_MN_is_correct_l621_621334


namespace gum_needed_l621_621038

-- Definitions based on problem conditions
def num_cousins : ℕ := 4
def gum_per_cousin : ℕ := 5

-- Proposition that we need to prove
theorem gum_needed : num_cousins * gum_per_cousin = 20 := by
  sorry

end gum_needed_l621_621038


namespace average_waiting_time_for_first_bite_l621_621708

theorem average_waiting_time_for_first_bite
  (bites_first_rod : ℝ)
  (bites_second_rod: ℝ)
  (total_bites: ℝ)
  (time_interval: ℝ)
  (H1 : bites_first_rod = 3)
  (H2 : bites_second_rod = 2)
  (H3 : total_bites = 5)
  (H4 : time_interval = 6) :
  1 / (total_bites / time_interval) = 1.2 :=
by
  rw [H3, H4]
  simp
  norm_num
  rw [div_eq_mul_inv, inv_div, inv_inv]
  norm_num
  sorry

end average_waiting_time_for_first_bite_l621_621708


namespace evaluate_ceil_of_neg_sqrt_l621_621257

-- Define the given expression and its value computation
def given_expression : ℚ := -real.sqrt (64 / 9)

-- Define the expected answer
def expected_answer : ℤ := -2

-- State the theorem to be proven
theorem evaluate_ceil_of_neg_sqrt : (Int.ceil given_expression) = expected_answer := sorry

end evaluate_ceil_of_neg_sqrt_l621_621257


namespace five_diff_numbers_difference_l621_621792

theorem five_diff_numbers_difference (S : Finset ℕ) (hS_size : S.card = 5) 
    (hS_range : ∀ x ∈ S, x ≤ 10) : 
    ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ c ≠ d ∧ a - b = c - d ∧ a - b ≠ 0 :=
by
  sorry

end five_diff_numbers_difference_l621_621792


namespace sister_functions_count_l621_621873

def is_sister_function (f1 f2 : ℝ → ℝ) (domain1 domain2 : set ℝ) : Prop :=
  (∀ x ∈ domain1, f1 x = |x| ∧ |x| ∈ {0, 1, 2}) ∧
  (∀ x ∈ domain2, f2 x = |x| ∧ |x| ∈ {0, 1, 2}) ∧
  (∀ x ∉ domain1, ∀ y ∉ domain2, x = y → f1 x = f2 y) 

theorem sister_functions_count : 
  ∃ (domain_list : list (set ℝ)), 
  list.length domain_list = 9 ∧ 
  (∀ domain ∈ domain_list, ∀ x ∈ domain, |x| ∈ {0, 1, 2}) ∧ 
  (∀ (i j : ℕ), i ≠ j → ∀ (x : ℝ), x ∈ domain_list.nth_le i sorry → x ∉ domain_list.nth_le j sorry) :=
sorry

end sister_functions_count_l621_621873


namespace infinitely_many_primes_of_form_6n_plus_5_l621_621948

theorem infinitely_many_primes_of_form_6n_plus_5 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ p % 6 = 5 :=
sorry

end infinitely_many_primes_of_form_6n_plus_5_l621_621948


namespace sum_of_three_numbers_l621_621577

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : ab + bc + ca = 131) : 
  a + b + c = 20 := 
by sorry

end sum_of_three_numbers_l621_621577


namespace max_elements_of_T_l621_621054

theorem max_elements_of_T : 
  ∃ (T : Set ℕ), T ⊆ {x | 1 ≤ x ∧ x ≤ 2023} ∧ 
    (∀ a ∈ T, ∀ b ∈ T, a ≠ b → (a - b) % 5 ≠ 0 ∧ (a - b) % 8 ≠ 0) ∧ 
    T.finite ∧ T.to_finset.card = 780 :=
sorry

end max_elements_of_T_l621_621054


namespace angle_A_magnitude_perimeter_range_l621_621903

-- Definitions based on the conditions
variables {a b c : ℝ}
def vector_m := ((b + c) ^ 2, -1)
def vector_n := (1, a ^ 2 + b * c)

-- Proving the magnitude of angle A given the dot product condition
theorem angle_A_magnitude
  (dot_product_zero : vector_m.1 * vector_n.1 + vector_m.2 * vector_n.2 = 0)
: ∠A = 2 * π / 3 :=
by {
  -- definitions and relations show that ∠A = 2π/3 radians or 120 degrees
  sorry
}

-- Proving the range of possible values for the perimeter of triangle ABC
theorem perimeter_range (a_eq_3 : a = 3)
: 6 < a + b + c ∧ a + b + c ≤ 3 + 2 * sqrt 3 :=
by {
  -- derivations and inequalities lead to the range of perimeter
  sorry
}

end angle_A_magnitude_perimeter_range_l621_621903


namespace sum_primes_no_integer_solution_l621_621742

noncomputable def primes_summing_to_ten : ℕ :=
  let primes := {p ∈ {2, 3, 5} | ¬ ∃ x : ℤ, 5 * (12 * x + 2) % p = 3 % p} in
  primes.to_finset.sum

theorem sum_primes_no_integer_solution : primes_summing_to_ten = 10 := by
  sorry

end sum_primes_no_integer_solution_l621_621742


namespace no_distributive_laws_hold_l621_621918

def tripledAfterAdding (a b : ℝ) : ℝ := 3 * (a + b)

theorem no_distributive_laws_hold (x y z : ℝ) :
  ¬ (tripledAfterAdding x (y + z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) ∧
  ¬ (x + (tripledAfterAdding y z) = tripledAfterAdding (x + y) (x + z)) ∧
  ¬ (tripledAfterAdding x (tripledAfterAdding y z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) :=
by sorry

end no_distributive_laws_hold_l621_621918


namespace ceil_neg_sqrt_frac_l621_621266

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l621_621266


namespace factorial_fraction_integer_l621_621113

open Nat

theorem factorial_fraction_integer (m n : ℕ) : 
  ∃ k : ℕ, k = (2 * m).factorial * (2 * n).factorial / (m.factorial * n.factorial * (m + n).factorial) := 
sorry

end factorial_fraction_integer_l621_621113


namespace find_y_l621_621616

theorem find_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x % y = 5.76)
  (h2 : x / y = 96.12) : y = 48 := 
by
  -- We will provide a proof here
  sorry

end find_y_l621_621616


namespace orchestra_ticket_cost_l621_621669

noncomputable def cost_balcony : ℝ := 8  -- cost of balcony tickets
noncomputable def total_sold : ℝ := 340  -- total tickets sold
noncomputable def total_revenue : ℝ := 3320  -- total revenue
noncomputable def extra_balcony : ℝ := 40  -- extra tickets sold for balcony than orchestra

theorem orchestra_ticket_cost (x y : ℝ) (h1 : x + extra_balcony = total_sold)
    (h2 : y = x + extra_balcony) (h3 : x + y = total_sold)
    (h4 : x + cost_balcony * y = total_revenue) : 
    cost_balcony = 8 → x = 12 :=
by
  sorry

end orchestra_ticket_cost_l621_621669


namespace length_of_BE_l621_621897

/--
In the figure, \(ABCD\) is a square piece of paper 8 cm on each side.
Corner \(B\) is folded over so that it coincides with \(F\), the midpoint of \(\overline{AD}\).
If \(\overline{GE}\) represents the crease created by the fold such that \(E\) is on \(\overline{AB}\),
then the length of \(\overline{BE}\) is 5 cm.
-/
theorem length_of_BE :
  let A B C D F E G : Type -- Points on the paper
  in ∀ (s : Real) (length_AD length_AD_half BF : Real),
  (length_AD = 8) → -- Square side length
  (length_AD_half = length_AD / 2) → -- Midpoint calculation
  (BF = 8 - s) → -- Right triangle property: hypotenuse calculation from fold
  s = (80 / 16) → -- Solution from Pythagorean theorem simplification
  s = 5 -- We should prove BE (or s) is 5 cm
:= 
by
intro A B C D F E G s length_AD length_AD_half BF
intro h1 h2 h3 h4
-- Here, insert logic based on the steps to establish the given problem's proof
have h5 : BF = 8 - s := h3
have h6 : s = 5 := h4
have h_length_BE : s = 5 := h6
exact h_length_BE

end length_of_BE_l621_621897


namespace stuart_initial_marbles_l621_621219

theorem stuart_initial_marbles (B S : ℝ) (h1 : B = 60) (h2 : 0.40 * B = 24) (h3 : S + 24 = 80) : S = 56 :=
by
  sorry

end stuart_initial_marbles_l621_621219


namespace find_k_values_l621_621014

noncomputable def long_distance (x y : ℝ) : ℝ := max (abs x) (abs y)

def P : ℝ × ℝ := (-1, 4)

def Q (k : ℝ) : ℝ × ℝ := (k + 3, 4*k - 3)

def are_equidistant_points (P Q : ℝ × ℝ) : Prop :=
  long_distance P.1 P.2 = long_distance Q.1 Q.2

theorem find_k_values :
  {k : ℝ // k = 1 ∨ k = -1/4} :=
begin
  sorry
end

end find_k_values_l621_621014


namespace ratio_of_segments_intersecting_chords_l621_621163

open Real

variables (EQ FQ HQ GQ : ℝ)

theorem ratio_of_segments_intersecting_chords 
  (h1 : EQ = 5) 
  (h2 : GQ = 7) 
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 7 / 5 :=
by
  sorry

end ratio_of_segments_intersecting_chords_l621_621163


namespace triangle_is_right_angled_l621_621384

-- Definitions of the sides and angles of the triangle
variables {a b c : ℝ} {A B C : ℝ}

-- Condition on the sides and angles
def triangle_condition (a b c A B C : ℝ) :=
  2 * c * (sin (A / 2))^2 = c - b

-- Goal: Prove that if the condition holds, then C = π / 2 (or 90 degrees)
theorem triangle_is_right_angled (h : triangle_condition a b c A B C) : C = π / 2 :=
sorry

end triangle_is_right_angled_l621_621384


namespace largest_subset_l621_621049

def is_valid_subset (T : set ℕ) : Prop :=
  ∀ x y ∈ T, x ≠ y → |x - y| ≠ 5 ∧ |x - y| ≠ 8

theorem largest_subset (T : set ℕ) (hT : T ⊆ (set.Icc 1 2023))
  (h_valid: is_valid_subset T) : T.card = 935 :=
sorry

end largest_subset_l621_621049


namespace sue_necklace_total_beads_l621_621681

theorem sue_necklace_total_beads :
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  (purple + blue + green = 46) :=
by
  intros purple blue green h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sue_necklace_total_beads_l621_621681


namespace max_value_of_f_l621_621143

noncomputable def f (theta x : ℝ) : ℝ :=
  (Real.cos theta)^2 - 2 * x * Real.cos theta - 1

noncomputable def M (x : ℝ) : ℝ :=
  if 0 <= x then 
    2 * x
  else 
    -2 * x

theorem max_value_of_f {x : ℝ} : 
  ∃ theta : ℝ, Real.cos theta ∈ [-1, 1] ∧ f theta x = M x :=
by
  sorry

end max_value_of_f_l621_621143


namespace dustin_vs_others_l621_621751

def pages_read_per_hour (pages_per_hour: ℕ) (minutes: ℕ) : ℝ := (pages_per_hour : ℝ) * (minutes : ℝ) / 60

def dustin_pages : ℝ := pages_read_per_hour 75 60
def sam_pages : ℝ := pages_read_per_hour 24 55
def nicole_pages : ℝ := pages_read_per_hour 35 35
def alex_pages : ℝ := pages_read_per_hour 50 50

def total_other_pages : ℝ := sam_pages + nicole_pages + alex_pages

theorem dustin_vs_others : dustin_pages - total_other_pages = -9.09 := by
  sorry

end dustin_vs_others_l621_621751


namespace j_eq_h_shift_reflect_l621_621574

noncomputable def h : ℝ → ℝ := sorry
def j (x : ℝ) : ℝ := h (6 - x)

theorem j_eq_h_shift_reflect (x : ℝ) : j(x) = h(6 - x) :=
by
  unfold j
  simp

end j_eq_h_shift_reflect_l621_621574


namespace log_increasing_incorrect_l621_621244

theorem log_increasing_incorrect {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : log a 3 < log a 9) :
  ¬ (∀ x y, x < y → log (1/3) x < log (1/3) y) :=
by
  sorry

end log_increasing_incorrect_l621_621244


namespace discount_percentage_l621_621366

theorem discount_percentage (wm_cost dryer_cost after_discount before_discount discount_amount : ℝ)
    (h0 : wm_cost = 100) 
    (h1 : dryer_cost = wm_cost - 30) 
    (h2 : after_discount = 153) 
    (h3 : before_discount = wm_cost + dryer_cost) 
    (h4 : discount_amount = before_discount - after_discount) 
    (h5 : (discount_amount / before_discount) * 100 = 10) : 
    True := sorry

end discount_percentage_l621_621366


namespace ce_length_l621_621412

open set

theorem ce_length
  (A B C D E : Type)
  [metric_space B] [metric_space C]
  (AC_len : ℝ) (BC_len : ℝ) (BE_len : ℝ)
  (h_obtuse : ∃ (x : B), ∠A B x = 90 ∨ ∠A B x > 90)
  (h_right : ∠A B D = 90)
  (h_bisect : ∠E B C / 2 = ∠D B C / 2) 
  (len_AC : ∥A - C∥ = 35)
  (len_BC : ∥B - C∥ = 7)
  (len_BE : ∥B - E∥ = 5)
  : ∥C - E∥ = 10 :=
sorry

end ce_length_l621_621412


namespace foci_distance_l621_621548

noncomputable def distance_between_foci
  (asymptote1 asymptote2 : ℝ → ℝ)
  (point : ℝ × ℝ) : ℝ :=
  let b_squared := 16 / 3
  let c := Real.sqrt (2 * b_squared)
  2 * c

theorem foci_distance :
  ∀ (asymptote1 asymptote2 : ℝ → ℝ)
  (point : ℝ × ℝ),
  (asymptote1 = (λ x, 2 * x + 3)) →
  (asymptote2 = (λ x, -2 * x + 3)) →
  (point = (4, 5)) →
  distance_between_foci asymptote1 asymptote2 point = (8 * Real.sqrt 3) / 3 := by
  intros
  -- Conditions based on the problem
  have h_asymptote1 : ∀ (x : ℝ), asymptote1 x = 2 * x + 3 := by assumption
  have h_asymptote2 : ∀ (x : ℝ), asymptote2 x = -2 * x + 3 := by assumption
  have h_point : point = (4, 5) := by assumption
  -- Define constants based on the problem solution steps
  let b_squared := 16 / 3
  have c : ℝ := Real.sqrt (2 * b_squared)
  -- Prove the final distance between the foci
  show distance_between_foci asymptote1 asymptote2 point = (8 * Real.sqrt 3) / 3
  sorry

end foci_distance_l621_621548


namespace trigonometric_relationship_l621_621314

-- Given conditions
variables (x : ℝ) (a b c : ℝ)

-- Required conditions
variables (h1 : π / 4 < x) (h2 : x < π / 2)
variables (ha : a = Real.sin x)
variables (hb : b = Real.cos x)
variables (hc : c = Real.tan x)

-- Proof goal
theorem trigonometric_relationship : b < a ∧ a < c :=
by
  -- Proof will go here
  sorry

end trigonometric_relationship_l621_621314


namespace smallest_w_factor_l621_621184

theorem smallest_w_factor:
  ∃ w : ℕ, (∃ n : ℕ, n = 936 * w ∧ 
              2 ^ 5 ∣ n ∧ 
              3 ^ 3 ∣ n ∧ 
              14 ^ 2 ∣ n) ∧ 
              w = 1764 :=
sorry

end smallest_w_factor_l621_621184


namespace difference_between_extreme_values_of_x_l621_621992

-- Define a problem setup with given conditions
def six_number_set (s : Set ℕ) : Prop :=
  ∃ x : ℕ, s = {4, 314, 710, x} ∧ s.card = 6

-- Define the range condition
def range_w (s : Set ℕ) (w : ℕ) : Prop :=
  w = s.max' sorry - s.min' sorry

-- Problem statement to prove
theorem difference_between_extreme_values_of_x :
  ∀ s : Set ℕ, ∀ x w : ℕ,
  six_number_set (s ∪ {x}) →
  range_w (s ∪ {x}) w →
  w = 12 →
  abs ((s ∪ {x}).max' sorry - (s ∪ {x}).min' sorry) = 682 :=
by
sor

end difference_between_extreme_values_of_x_l621_621992


namespace minimum_participants_l621_621530

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l621_621530


namespace max_sqrt_sum_l621_621418

theorem max_sqrt_sum (n : ℕ) (hn : n ≥ 3 ∧ odd n) (x : fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∑ i in finset.range n, real.sqrt (abs (x i - x ((i + 1) % n))) ≤ n - 2 + real.sqrt 2 :=
sorry

end max_sqrt_sum_l621_621418


namespace ratio_of_areas_of_squares_l621_621956

theorem ratio_of_areas_of_squares (sideC sideD : ℕ) (hC : sideC = 45) (hD : sideD = 60) : 
  (sideC ^ 2) / (sideD ^ 2) = 9 / 16 := 
by
  sorry

end ratio_of_areas_of_squares_l621_621956


namespace find_unknown_number_l621_621774

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l621_621774


namespace sin_theta_for_arithm_prog_l621_621423

theorem sin_theta_for_arithm_prog :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ 
  (cos θ, cos (2 * θ), cos (4 * θ) are_arithm_prog_in_some_order) ∧ 
  sin θ = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_theta_for_arithm_prog_l621_621423


namespace average_waiting_time_for_first_bite_l621_621707

theorem average_waiting_time_for_first_bite
  (bites_first_rod : ℝ)
  (bites_second_rod: ℝ)
  (total_bites: ℝ)
  (time_interval: ℝ)
  (H1 : bites_first_rod = 3)
  (H2 : bites_second_rod = 2)
  (H3 : total_bites = 5)
  (H4 : time_interval = 6) :
  1 / (total_bites / time_interval) = 1.2 :=
by
  rw [H3, H4]
  simp
  norm_num
  rw [div_eq_mul_inv, inv_div, inv_inv]
  norm_num
  sorry

end average_waiting_time_for_first_bite_l621_621707


namespace matthews_contribution_l621_621084

theorem matthews_contribution 
  (total_cost : ℝ) (yen_amount : ℝ) (conversion_rate : ℝ)
  (h1 : total_cost = 18)
  (h2 : yen_amount = 2500)
  (h3 : conversion_rate = 140) :
  (total_cost - (yen_amount / conversion_rate)) = 0.143 :=
by sorry

end matthews_contribution_l621_621084


namespace circle_diameter_l621_621966

theorem circle_diameter
  (O : Type) [metric_space O] 
  (circle : set O) (A B C D E : O) 
  (DC : ℝ) (angle_DAC : ℝ) (angle_DOC : ℝ)
  (h1 : DC = 3)
  (h2 : angle_DAC = 30)
  (h3 : angle_DOC = 7)
  (is_diameter : ∀ (P : O), P ∈ circle → dist A P = dist B P) :
  (2 * dist O D = 2 * sqrt(3) * sin(real.to_radians 7)) →
  ∀ (diameter : ℝ), diameter = 4 * sqrt(3) * sin(real.to_radians 7) :=
by
  sorry

end circle_diameter_l621_621966


namespace race_participants_least_number_l621_621501

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l621_621501


namespace profit_inequality_solution_l621_621200

theorem profit_inequality_solution (x : ℝ) (h₁ : 1 ≤ x) (h₂ : x ≤ 10) :
  100 * 2 * (5 * x + 1 - 3 / x) ≥ 3000 ↔ 3 ≤ x ∧ x ≤ 10 :=
by
  sorry

end profit_inequality_solution_l621_621200


namespace solve_for_t_l621_621440

open Set

variable {t : ℝ}

def A : Set ℝ := {-4, t^2}
def B : Set ℝ := {t - 5, 9, 1 - t}

theorem solve_for_t (h : 9 ∈ A ∩ B) : t = -3 :=
by
  have hA : 9 ∈ A := by sorry
  have hB : 9 ∈ B := by sorry
  have ht : t^2 = 9 := by sorry
  have h1 : t = 3 ∨ t = -3 := by sorry
  have ht_neq_3 : t ≠ 3 := by sorry
  show t = -3

end solve_for_t_l621_621440


namespace Lulu_blueberry_pies_baked_l621_621081

-- Definitions of conditions
def Lola_mini_cupcakes := 13
def Lola_pop_tarts := 10
def Lola_blueberry_pies := 8
def Lola_total_pastries := Lola_mini_cupcakes + Lola_pop_tarts + Lola_blueberry_pies
def Lulu_mini_cupcakes := 16
def Lulu_pop_tarts := 12
def total_pastries := 73

-- Prove that Lulu baked 14 blueberry pies
theorem Lulu_blueberry_pies_baked : 
  ∃ (Lulu_blueberry_pies : Nat), 
    Lola_total_pastries + Lulu_mini_cupcakes + Lulu_pop_tarts + Lulu_blueberry_pies = total_pastries ∧ 
    Lulu_blueberry_pies = 14 := by
  sorry

end Lulu_blueberry_pies_baked_l621_621081


namespace num_triangles_l621_621080

theorem num_triangles (a b : List Point) (h₁ : a.length = 5) (h₂ : b.length = 6) (h₃ : parallel a b) : 
  (num_diff_triangles a b) = 135 :=
sorry

end num_triangles_l621_621080


namespace stuart_initial_marbles_is_56_l621_621221

-- Define the initial conditions
def betty_initial_marbles : ℕ := 60
def percentage_given_to_stuart : ℚ := 40 / 100
def stuart_marbles_after_receiving : ℕ := 80

-- Define the calculation of how many marbles Betty gave to Stuart
def marbles_given_to_stuart := (percentage_given_to_stuart * betty_initial_marbles)

-- Define the target: Stuart's initial number of marbles
def stuart_initial_marbles := stuart_marbles_after_receiving - marbles_given_to_stuart

-- Main theorem stating the problem
theorem stuart_initial_marbles_is_56 : stuart_initial_marbles = 56 :=
by 
  sorry

end stuart_initial_marbles_is_56_l621_621221


namespace magnitude_of_sixth_power_l621_621303

noncomputable def complex_magnitude := (5 : ℂ) + (2 * real.sqrt 3) * complex.I

theorem magnitude_of_sixth_power :
  complex.abs (complex_magnitude ^ 6) = 50653 :=
by
  sorry

end magnitude_of_sixth_power_l621_621303


namespace problem1_problem2_problem3_l621_621824

-- Define the function f as described in the problem
def f (x : ℚ) := if x ≠ 1/2 then (2 * x) / (1 - 2 * x) else -1

-- 1. Prove x₁ + x₂ = 1 and y₁ + y₂ = -2
theorem problem1 (x₁ x₂ y₁ y₂ : ℚ) (hA : y₁ = f x₁) (hB : y₂ = f x₂) (h : x₁ + x₂ = 1) :
  y₁ + y₂ = -2 :=
begin
  -- Proof goes here
  sorry
end

-- 2. Prove Sₙ = 1 - n for n ≥ 2 given S₁ = 0 and the form of Sₙ
def S (n : ℕ) : ℚ := if n = 1 then 0 else ∑ i in finset.range (n-1), f (i.succ / n)

theorem problem2 (n : ℕ) (h : n ≥ 2) :
  S n = 1 - n :=
begin
  -- Proof goes here
  sorry
end

-- 3. Prove c = 1 and m = 1 using the sequence aₙ = 2^Sₙ and sum Tₙ.
def a (n : ℕ) : ℚ := 2^(S n)

def T (n : ℕ) : ℚ := ∑ i in finset.range n, a i.succ

theorem problem3 (c m : ℕ) (hc : c ≠ 0) (hm : m ≠ 0) (hineq : (T m - c) / (T (m + 1) - c) < 1/2) :
  c = 1 ∧ m = 1 :=
begin
  -- Proof goes here
  sorry
end

end problem1_problem2_problem3_l621_621824


namespace collinear_A_H_T_l621_621119

variables {A B C D E F G T H : Type}

-- Definitions of rectangles ABCD and AEFG
variables (isRectangle : ∀ {R : Type} (A B C D : R), Prop)

-- Conditions about the collinear points B, E, D, G
variables (collinear : ∀ {O : Type}, List O → Prop)
variables (B E D G : Type)
variables (collinear_BEDG : collinear [B, E, D, G])

-- Intersection points T and H
variables (intersect : ∀ {P Q R S : Type}, Prop)
variables (T : Type) (H : Type)
variables (intersect_T : intersect (BC A B C D) (GF A E F G))
variables (intersect_H : intersect (DC A B C D) (EF A E F G))

-- Theorem statement
theorem collinear_A_H_T :
  isRectangle A B C D →
  isRectangle A E F G →
  collinear [B, E, D, G] →
  intersect (BC A B C D) (GF A E F G) →
  intersect (DC A B C D) (EF A E F G) →
  collinear [A, H, T] :=
by
  intros h1 h2 h3 h4 h5
  sorry

end collinear_A_H_T_l621_621119


namespace smallest_number_of_marbles_l621_621602

theorem smallest_number_of_marbles :
  ∃ N : ℕ, N > 1 ∧ (N % 9 = 1) ∧ (N % 10 = 1) ∧ (N % 11 = 1) ∧ (∀ m : ℕ, m > 1 ∧ (m % 9 = 1) ∧ (m % 10 = 1) ∧ (m % 11 = 1) → N ≤ m) :=
sorry

end smallest_number_of_marbles_l621_621602


namespace perfect_square_trinomial_l621_621373

theorem perfect_square_trinomial (k x y : ℝ) :
  (∃ a b : ℝ, 9 * x^2 - k * x * y + 4 * y^2 = (a * x + b * y)^2) ↔ (k = 12 ∨ k = -12) :=
by
  sorry

end perfect_square_trinomial_l621_621373


namespace total_beads_correct_l621_621678

def purple_beads : ℕ := 7
def blue_beads : ℕ := 2 * purple_beads
def green_beads : ℕ := blue_beads + 11
def total_beads : ℕ := purple_beads + blue_beads + green_beads

theorem total_beads_correct : total_beads = 46 := 
by
  have h1 : purple_beads = 7 := rfl
  have h2 : blue_beads = 2 * 7 := rfl
  have h3 : green_beads = 14 + 11 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end total_beads_correct_l621_621678


namespace intersection_proof_l621_621853

def M : set ℤ := {-1, 0, 1}
def N : set ℝ := {x | x^2 ≤ x}

theorem intersection_proof : (M : set ℝ) ∩ N = {0, 1} := by
  sorry

end intersection_proof_l621_621853


namespace pills_per_week_l621_621037

theorem pills_per_week (hours_per_pill : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) 
(h1: hours_per_pill = 6) (h2: hours_per_day = 24) (h3: days_per_week = 7) :
(hours_per_day / hours_per_pill) * days_per_week = 28 :=
by
  sorry

end pills_per_week_l621_621037


namespace y_sequence_not_periodic_l621_621817

noncomputable def x_sequence : ℕ → ℕ
| 0       := 2
| (n + 1) := int.floor ((3 : ℚ) / 2 * x_sequence n)

def y_sequence (n : ℕ) : ℤ :=
  (-1) ^ x_sequence n

theorem y_sequence_not_periodic : ¬ ∃ T : ℕ, T > 0 ∧ ∀ n : ℕ, y_sequence (n + T) = y_sequence n :=
begin
  -- Proof goes here
  sorry
end

end y_sequence_not_periodic_l621_621817


namespace ceil_neg_sqrt_64_over_9_l621_621271

theorem ceil_neg_sqrt_64_over_9 : Real.ceil (-Real.sqrt (64 / 9)) = -2 := 
by
  sorry

end ceil_neg_sqrt_64_over_9_l621_621271


namespace ned_initial_lives_l621_621181

variable (lost_lives : ℕ) (current_lives : ℕ) 
variable (initial_lives : ℕ)

theorem ned_initial_lives (h_lost: lost_lives = 13) (h_current: current_lives = 70) :
  initial_lives = current_lives + lost_lives := by
  sorry

end ned_initial_lives_l621_621181


namespace parabola_properties_l621_621552

theorem parabola_properties :
  ∃ (a b : ℝ) (h : parabola_vertex a b), 
    (h.direction_of_opening = "upwards" ∧ h.axis_of_symmetry = 2 ∧ h.vertex = (2, -8)) := 
sorry

end parabola_properties_l621_621552


namespace ceil_neg_sqrt_fraction_l621_621275

theorem ceil_neg_sqrt_fraction :
  (⌈-real.sqrt (64 / 9)⌉ = -2) :=
by
  -- Define the necessary conditions
  have h1 : real.sqrt (64 / 9) = 8 / 3 := by sorry,
  have h2 : -real.sqrt (64 / 9) = -8 / 3 := by sorry,
  -- Apply the ceiling function and prove the result
  exact sorry

end ceil_neg_sqrt_fraction_l621_621275


namespace obtuse_triangle_120_gon_l621_621360

theorem obtuse_triangle_120_gon :
  let vertices := Finset.range 120
  in (∑ k in vertices, (finset.filter (λ l, l ≠ k) vertices).card.choose 2) / 2 = 205320 :=
by
  let vertices := Finset.range 120
  let valid_pairs := ∑ k in vertices, (finset.filter (λ l, l ≠ k) vertices).card.choose 2
  have : valid_pairs = 205320, sorry
  exact this

end obtuse_triangle_120_gon_l621_621360


namespace knights_and_knaves_l621_621469

/-- Proof that there are exactly 5 knights among 7 inhabitants given specific conditions -/
theorem knights_and_knaves : 
  ∀ (inhabitants : Fin 7 → Prop)
    (knight truthful liar : Prop)
    (H1 : knight = truthful)
    (H2 : liar = ¬truthful)
    (H3 : (inhabitants 0 = knight))
    (H4 : (inhabitants 1 = knight))
    (H5 : (inhabitants 2 = liar ∧ (inhabitants 0 = knight ∨ inhabitants 1 = knight)))
    (H6 : (inhabitants 3 = liar ∧ (inhabitants 0 = truth ∧ inhabitants 1 = truth ∧ inhabitants 2 = liar → 2/3 ≥ 0.65)))
    (H7 : (inhabitants 4 = knight))
    (H8 : (inhabitants 5 = knight ∧ (∃ (half_knights : Fin 6 → Prop), (inhabitants 0 = knight) ∧ (inhabitants 1 = knight) ∧ (inhabitants 4 = knight) ∧ (inhabitants 5 = knight) ∧ counting_knaves half_knights < 3))
    (H9 : (inhabitants 6 = knight ∧ (counting_knights inhabitants 6 ≥ 0.65))),
  counting_knights inhabitants = 5 := 
by sorry

def counting_knights (inhabitants : Fin 7 → Prop) : ℝ := sorry
def counting_knaves (inhabitants : Fin 7 → Prop) : ℝ := sorry

end knights_and_knaves_l621_621469


namespace general_term_form_sum_of_first_n_terms_l621_621016

noncomputable theory
open Real Nat

-- Condition definitions
def a (n : ℕ) : ℝ := 
  if n = 0 then 0 else if n = 1 then 1 / 3 else a (n - 1)

def S : ℕ → ℝ
| 0     := 0
| (n+1) := S n + 2^n / 3

def b (n : ℕ) : ℝ := log (S n + 1 / 3)

-- Questions
theorem general_term_form (n : ℕ) (hn : n > 0) : a n = 2^(n-1) / 3 :=
sorry

theorem sum_of_first_n_terms (n : ℕ) : 
  let T : ℕ → ℝ := λ n, (list.range n).sum b in
  T n = n * (n + 1) / 2 * log 2 - n * log 3 := 
sorry

end general_term_form_sum_of_first_n_terms_l621_621016


namespace even_functions_identification_l621_621076

variable (f : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_functions_identification :
  is_even_function (λ x, |x| * f(x^2)) ∧ is_even_function (λ x, f x + f (-x)) :=
by
  sorry

end even_functions_identification_l621_621076


namespace fraction_neg_exponent_l621_621719

theorem fraction_neg_exponent (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (2 * a / (3 * b)) ^ (-2) = (9 * b ^ 2) / (4 * a ^ 2) :=
by sorry

end fraction_neg_exponent_l621_621719


namespace race_participants_least_number_l621_621498

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l621_621498


namespace smallest_k_no_real_roots_l621_621597

theorem smallest_k_no_real_roots : ∃ k : ℤ, 
  (∀ x : ℝ, (2 * k - 1) * x^2 - 10 * x + 12 = 0 → x ∉ ℝ) ∧ k = 3 :=
sorry

end smallest_k_no_real_roots_l621_621597


namespace average_waiting_time_for_first_bite_l621_621706

theorem average_waiting_time_for_first_bite
  (bites_first_rod : ℝ)
  (bites_second_rod: ℝ)
  (total_bites: ℝ)
  (time_interval: ℝ)
  (H1 : bites_first_rod = 3)
  (H2 : bites_second_rod = 2)
  (H3 : total_bites = 5)
  (H4 : time_interval = 6) :
  1 / (total_bites / time_interval) = 1.2 :=
by
  rw [H3, H4]
  simp
  norm_num
  rw [div_eq_mul_inv, inv_div, inv_inv]
  norm_num
  sorry

end average_waiting_time_for_first_bite_l621_621706


namespace round_trip_by_car_time_l621_621673

variable (time_walk time_car : ℕ)
variable (h1 : time_walk + time_car = 20)
variable (h2 : 2 * time_walk = 32)

theorem round_trip_by_car_time : 2 * time_car = 8 :=
by
  sorry

end round_trip_by_car_time_l621_621673


namespace jack_total_payment_l621_621030

variable (squat_rack barbell weights total_cost_before_tax sales_tax total_cost_with_tax : ℝ)
variable (cost_squat : squat_rack = 2500)
variable (cost_barbell : barbell = (1/10) * squat_rack)
variable (cost_weights : weights = 750)
variable (tax_rate : ℝ := 0.06)

def total_cost_before_tax_def := total_cost_before_tax = squat_rack + barbell + weights
def sales_tax_def := sales_tax = tax_rate * total_cost_before_tax
def total_cost_with_tax_def := total_cost_with_tax = total_cost_before_tax + sales_tax

theorem jack_total_payment : 
  total_cost_with_tax = 3710 := by
  rw [total_cost_with_tax_def, sales_tax_def, total_cost_before_tax_def, cost_squat, cost_barbell, cost_weights]
  sorry

end jack_total_payment_l621_621030


namespace P_2018_has_2018_distinct_real_roots_l621_621629

-- Define the sequence of polynomials P_n(x) with given initial conditions.
noncomputable def P : ℕ → Polynomial ℝ
| 0       := 1
| 1       := Polynomial.X
| (n + 1) := Polynomial.X * P n - P (n - 1)

-- The statement to prove that P_2018(x) has exactly 2018 distinct real roots.
theorem P_2018_has_2018_distinct_real_roots :
  ∃ (roots : Fin 2018 → ℝ), ∀ i j : Fin 2018, i ≠ j → roots i ≠ roots j ∧ ∀ x : ℝ, (Polynomial.aeval x (P 2018)) = 0 ↔ x ∈ Set.range roots := sorry

end P_2018_has_2018_distinct_real_roots_l621_621629


namespace G_at_six_l621_621421

noncomputable def G : ℕ → ℤ
| 0     := 1
| (n+1) := 3 * (G n) - 2

theorem G_at_six : G 5 = 1 :=  -- G(6) in the problem corresponds to G(5) in zero-based indexing
by
  sorry

end G_at_six_l621_621421


namespace num_methods_to_select_translators_l621_621671

-- Definitions based on the conditions
def total_translators : ℕ := 8
def english_translators : ℕ := 3
def japanese_translators : ℕ := 3
def both_translators : ℕ := 2
def translators_needed : ℕ := 5
def english_needed : ℕ := 3
def japanese_needed : ℕ := 2
def xiao_zhang_included : Prop := true -- Placeholder, define condition differently in real use
def xiao_li_included : Prop := true -- Placeholder, define condition differently in real use

-- The final theorem to prove the count of methods
theorem num_methods_to_select_translators 
  (total_translators = 8) 
  (english_translators = 3) 
  (japanese_translators = 3) 
  (both_translators = 2) 
  (translators_needed = 5) 
  (english_needed = 3) 
  (japanese_needed = 2) 
  (xiao_zhang_included ∨ xiao_li_included) 
  (¬ (xiao_zhang_included ∧ xiao_li_included)) : 
  ∃ n : ℕ, n = 29 :=
by
  sorry

end num_methods_to_select_translators_l621_621671


namespace circumference_difference_l621_621196

theorem circumference_difference (r : ℝ) (width : ℝ) (hp : width = 10.504226244065093) : 
  2 * Real.pi * (r + width) - 2 * Real.pi * r = 66.00691339889247 := by
  sorry

end circumference_difference_l621_621196


namespace plane_divided_into_segments_l621_621478

-- Define the problem statement
theorem plane_divided_into_segments : 
  ∀ (P : ℝ × ℝ), ∃ S : set (set (ℝ × ℝ)), 
  (∀ s ∈ S, ∃ a b : ℝ × ℝ, a ≠ b ∧ s = {t | t = a ∨ t = b}) ∧
  (∀ s₁ ∈ S, ∀ s₂ ∈ S, s₁ ≠ s₂ → s₁ ∩ s₂ ⊆ {u | ∃ v ∈ s₁, ∃ w ∈ s₂, u = v ∧ u = w}) :=
sorry

end plane_divided_into_segments_l621_621478


namespace total_salmon_count_l621_621402

def chinook_males := 451228
def chinook_females := 164225
def sockeye_males := 212001
def sockeye_females := 76914
def coho_males := 301008
def coho_females := 111873
def pink_males := 518001
def pink_females := 182945
def chum_males := 230023
def chum_females := 81321

theorem total_salmon_count : 
  chinook_males + chinook_females + 
  sockeye_males + sockeye_females + 
  coho_males + coho_females + 
  pink_males + pink_females + 
  chum_males + chum_females = 2329539 := 
by
  sorry

end total_salmon_count_l621_621402


namespace Emily_sixth_score_l621_621252

theorem Emily_sixth_score :
  let scores := [91, 94, 88, 90, 101]
  let current_sum := scores.sum
  let desired_average := 95
  let num_quizzes := 6
  let total_score_needed := num_quizzes * desired_average
  let sixth_score := total_score_needed - current_sum
  sixth_score = 106 :=
by
  sorry

end Emily_sixth_score_l621_621252


namespace sum_first_20_integers_l621_621171

def sum_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem sum_first_20_integers : sum_first_n_integers 20 = 210 :=
by
  -- Provided proof omitted
  sorry

end sum_first_20_integers_l621_621171


namespace min_distance_rational_irrational_l621_621945

-- Definition of paths for Rational Woman and Irrational Woman
def rational_woman_path (t : ℝ) : ℝ × ℝ := (Real.sin t, Real.cos t)
def irrational_woman_path (t : ℝ) : ℝ × ℝ := (2 + 3 * Real.cos (t / Real.sqrt 3), 3 * Real.sin (t / Real.sqrt 3))

-- Problem statement: Prove the smallest distance between a point on Rational Woman's path and a point on
-- Irrational Woman's path is 2.
theorem min_distance_rational_irrational : 
∀ (C : ℝ × ℝ) (D : ℝ × ℝ), 
  (∃ t_C : ℝ, C = rational_woman_path t_C) → 
  (∃ t_D : ℝ, D = irrational_woman_path t_D) →
  (∀ (d : ℝ), d = Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) → d ≥ 2) :=
by
  sorry

end min_distance_rational_irrational_l621_621945


namespace solve_for_b_l621_621410

noncomputable def a : ℝ := real.sqrt 5
noncomputable def c : ℝ := 2
noncomputable def cos_A : ℝ := 2 / 3
noncomputable def b (a c cos_A : ℝ) : ℝ :=
  let b_squared := (a ^ 2) - (c ^ 2) + (2 * c * cos_A)
  real.sqrt b_squared

theorem solve_for_b : b a c cos_A = 3 :=
by
  sorry

end solve_for_b_l621_621410


namespace trajectory_eqn_min_value_l621_621882

-- definitions of point and vector operations
structure Point where
  x : ℝ
  y : ℝ

def vector (A B : Point) : Point :=
  ⟨B.x - A.x, B.y - A.y⟩

-- Given conditions
def A := Point.mk 3 4
def C := Point.mk 0 0 -- Center of the circle at the origin
def radius := 1

-- Given vector relationships
variable (x y : ℝ)
def AM := vector A {x := 0, y := (A.y - radius)}
def AN := vector A {x := (A.x - radius), y := 0}

-- Prove the trajectory equation and minimum value
theorem trajectory_eqn_min_value : (∀ (p : Point),
  let P := p in
  let M := Point.mk 0 ((4 * (x + y - 1)) / x) in
  let N := Point.mk ((3 * (x + y - 1)) / y) 0 in
  let AC := vector A C in
  let AM := vector A M in
  let AN := vector A N in
  let trajectory_eqn := (P.x^2 / 16) + (P.y^2 / 9) = (P.x + P.y - 1)^2 in
  let min_value := 9 * P.x^2 + 16 * P.y^2 in
  (trajectory_eqn ∧ min_value = 4)) :=
sorry

end trajectory_eqn_min_value_l621_621882


namespace boxes_with_neither_l621_621367

-- Definitions based on the conditions given
def total_boxes : Nat := 12
def boxes_with_markers : Nat := 8
def boxes_with_erasers : Nat := 5
def boxes_with_both : Nat := 4

-- The statement we want to prove
theorem boxes_with_neither :
  total_boxes - (boxes_with_markers + boxes_with_erasers - boxes_with_both) = 3 :=
by
  sorry

end boxes_with_neither_l621_621367


namespace fifteen_percent_of_x_is_ninety_l621_621756

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l621_621756


namespace find_r_l621_621693

-- Define our hexagon with relevant points and ratios
structure Hexagon :=
  (A B C D E F : Point)
  (regular : regular_hexagon A B C D E F)

-- Define the collinearity of points B, M, and N
def point_on_diagonal (A C : Point) (r : ℝ) : Point := sorry

def collinear (B M N : Point) : Prop := sorry

-- The statement to prove
theorem find_r (A B C D E F : Point) (r : ℝ)
  (hex : Hexagon A B C D E F)
  (M := point_on_diagonal A C r)
  (N := point_on_diagonal C E r)
  (h_collinear : collinear B M N) :
  r = (Real.sqrt 3) / 3 :=
sorry

end find_r_l621_621693


namespace B_subset_A_implies_range_m_l621_621826

variable {x m : ℝ}

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | -m < x ∧ x < m}

theorem B_subset_A_implies_range_m (m : ℝ) (h : B m ⊆ A) : m ≤ 1 := by
  sorry

end B_subset_A_implies_range_m_l621_621826


namespace minimum_participants_l621_621494

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l621_621494


namespace ceil_of_neg_sqrt_frac_64_over_9_l621_621289

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l621_621289


namespace parallel_EF_HR_l621_621933

theorem parallel_EF_HR
  {A B C H A' E F P Q R : Type}
  [InTriangle A B C H]  -- H is the orthocenter of triangle ABC
  [ReflectOver A BC A'] -- A' is the reflection of A over BC
  [Intersects (Circumcircle A A' E) (Circumcircle A B C)] (P ≠ A)
  [Intersects (Circumcircle A A' F) (Circumcircle A B C)] (Q ≠ A)
  [Intersects BC PQ]  -- Lines BC and PQ intersect at R
  :
  Parallel (Line E F) (Line H R) :=
sorry

end parallel_EF_HR_l621_621933


namespace avg_waiting_time_is_1_point_2_minutes_l621_621699

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l621_621699


namespace evaluate_ceil_of_neg_sqrt_l621_621261

-- Define the given expression and its value computation
def given_expression : ℚ := -real.sqrt (64 / 9)

-- Define the expected answer
def expected_answer : ℤ := -2

-- State the theorem to be proven
theorem evaluate_ceil_of_neg_sqrt : (Int.ceil given_expression) = expected_answer := sorry

end evaluate_ceil_of_neg_sqrt_l621_621261


namespace largest_number_given_HCF_and_LCM_factors_l621_621961

theorem largest_number_given_HCF_and_LCM_factors
  (HCF : ℕ) (a b : ℕ) (H : nat.gcd a b = HCF) 
  (factors : list ℕ) (h1 : HCF = 143) 
  (h2 : factors = [17, 23, 31]) 
  (LCM : ℕ) (h3 : LCM = HCF * factors.prod) :
  max a b = 1621447 := 
by {
   have h4 : factors.prod = 17 * 23 * 31 := by simp [factors],
   rw [h1, h4] at h3,
   have h5 : LCM = 143 * 11339 := by simp [h3],
   have h6 : 143 * 11339 = 1621447 := by norm_num,
   rw h6 at h5,
   exact sorry,
}

end largest_number_given_HCF_and_LCM_factors_l621_621961


namespace greg_less_rain_than_friend_l621_621180

theorem greg_less_rain_than_friend :
  let greg_rain := [3, 6, 5, 7, 4, 8, 9] in
  let total_greg_rain := greg_rain.sum in
  let friend_rain := 65 in
  (friend_rain - total_greg_rain = 23) :=
by
  let greg_rain := [3, 6, 5, 7, 4, 8, 9]
  let total_greg_rain := List.sum greg_rain
  let friend_rain := 65
  have rain_difference : friend_rain - total_greg_rain = 23 := sorry
  exact rain_difference

end greg_less_rain_than_friend_l621_621180


namespace mr_rodgers_chapters_read_l621_621091

theorem mr_rodgers_chapters_read
  (num_books : ℕ) (chapters_per_book : ℕ)
  (h1 : num_books = 10) (h2 : chapters_per_book = 24) :
  num_books * chapters_per_book = 240 :=
by
  rw [h1, h2]
  exact Nat.mul_comm _ _  -- Uses commutativity and basic multiplication
-- Here we use sorry to skip the elaborate proof steps if required, although in this case basic arithmetic should suffice.

end mr_rodgers_chapters_read_l621_621091


namespace number_of_cows_l621_621392

-- Definitions
variables (c h : ℕ)

-- Conditions
def condition1 : Prop := 4 * c + 2 * h = 20 + 2 * (c + h)
def condition2 : Prop := c + h = 12

-- Theorem
theorem number_of_cows : condition1 c h → condition2 c h → c = 10 :=
  by 
  intros h1 h2
  sorry

end number_of_cows_l621_621392


namespace bob_catches_up_in_63_minutes_l621_621910

-- Definitions of the conditions
def john_speed : ℝ := 4  -- John's speed in miles per hour
def bob_speed : ℝ := 7   -- Bob's speed in miles per hour
def bob_wait_time : ℝ := 10 / 60  -- Bob's waiting time in hours
def initial_distance : ℝ := 2  -- Initial distance between Bob and John in miles

-- The statement we want to prove
theorem bob_catches_up_in_63_minutes :
  let relative_speed := bob_speed - john_speed in
  let additional_distance := john_speed * bob_wait_time in
  let total_distance := initial_distance + additional_distance in
  let catch_up_time := total_distance / relative_speed in
  let time_in_minutes := catch_up_time * 60 + 10 in
  time_in_minutes = 63 := 
by
  sorry

end bob_catches_up_in_63_minutes_l621_621910


namespace sin_cos_product_l621_621865

theorem sin_cos_product (x : ℝ) (h : sin x = 2 * cos x) : sin x * cos x = 2 / 5 := by
  sorry

end sin_cos_product_l621_621865


namespace exists_function_satisfying_property_l621_621872

noncomputable def p : ℤ → ℤ :=
λ t, ∑ j in finset.range (2*(nat_abs t) + 1), (t + j) * (t + j - 1) * (t - j)

theorem exists_function_satisfying_property :
  ∃ (p : ℤ → ℤ), 
  (∀ m n : ℤ, n ∣ (p (m + n) - p m)) 
  ∧ (¬ ∃ q : ℤ → ℤ, polynomial.is_polynomial_function q ∧ (∀ t : ℤ, q t = p t)) :=
by
  sorry

end exists_function_satisfying_property_l621_621872


namespace evaluate_ceil_of_neg_sqrt_l621_621258

-- Define the given expression and its value computation
def given_expression : ℚ := -real.sqrt (64 / 9)

-- Define the expected answer
def expected_answer : ℤ := -2

-- State the theorem to be proven
theorem evaluate_ceil_of_neg_sqrt : (Int.ceil given_expression) = expected_answer := sorry

end evaluate_ceil_of_neg_sqrt_l621_621258


namespace rectangle_has_equal_diagonals_not_rhombus_l621_621145

-- Definitions based on conditions
structure Rectangle (R : Type) [MetricSpace R] extends Parallelogram R :=
  (diagonalsEqual : ∀ (d1 d2 : ℝ), d1 = d2)

structure Rhombus (R : Type) [MetricSpace R] extends Parallelogram R := 
  (diagonalsBisect : ∀ (d1 d2 : ℝ), (d1 / 2) = (d2 / 2))

-- Theorem to prove
theorem rectangle_has_equal_diagonals_not_rhombus (R : Type) [MetricSpace R] :
  ∃ (rect : Rectangle R) (rhomb : Rhombus R), rect.diagonalsEqual ∧ ¬ rhomb.diagonalsEqual := by
  sorry

end rectangle_has_equal_diagonals_not_rhombus_l621_621145


namespace juice_oranges_l621_621858

theorem juice_oranges (oranges_per_glass : ℕ) (glasses : ℕ) (total_oranges : ℕ)
  (h1 : oranges_per_glass = 3)
  (h2 : glasses = 10)
  (h3 : total_oranges = oranges_per_glass * glasses) :
  total_oranges = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end juice_oranges_l621_621858


namespace greatest_missed_problems_l621_621714

def ScholarsInstitute := { p : ℕ // p = 50 }
def passing_score := 85 / 100
def score_to_pass (total_problems : ℕ) : ℕ := Nat.ceil (passing_score * total_problems)

theorem greatest_missed_problems (total_problems : ℕ) : total_problems = 50 → total_problems - score_to_pass total_problems = 7 :=
by 
  intro h
  have h1 : score_to_pass 50 = 43 := rfl
  rw h1
  rw [h]
  sorry

end greatest_missed_problems_l621_621714


namespace race_minimum_participants_l621_621521

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l621_621521


namespace circle_tangency_l621_621015

-- Definitions for polar and Cartesian transformations
def circle_c1_polar (θ : ℝ) : ℝ := -2 * real.sqrt 2 * real.cos (θ - real.pi / 4)

structure circle_parametric (m : ℝ) (θ : ℝ) :=
  (x : ℝ := 2 + m * real.cos θ)
  (y : ℝ := 2 + m * real.sin θ)

-- Distance between two points in a plane
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Prove that the value of m satisfies the tangency condition
theorem circle_tangency {m : ℝ} :
  let c1_center := (-1 : ℝ, -1 : ℝ), c1_radius := real.sqrt 2,
      c2_center := (2 : ℝ, 2 : ℝ), c2_radius := abs m in
  distance c1_center c2_center = c1_radius + c2_radius →
  m = 2 * real.sqrt 2 ∨ m = -2 * real.sqrt 2 :=
by
  sorry

end circle_tangency_l621_621015


namespace lucy_money_ratio_l621_621082

theorem lucy_money_ratio
  (initial_amount : ℕ)
  (lost_fraction : ℚ)
  (remaining_after_spent : ℕ)
  (initial_condition : initial_amount = 30)
  (lost_condition : lost_fraction = 1 / 3)
  (remaining_condition : remaining_after_spent = 15) :
  let spent_amount := (initial_amount - (initial_amount * lost_fraction).natAbs) - remaining_after_spent,
      remaining_amount := initial_amount - (initial_amount * lost_fraction).natAbs
  in spent_amount * 4 = remaining_amount :=
by
  -- Proof will go here
  sorry

end lucy_money_ratio_l621_621082


namespace KnightsCount_l621_621455

def isKnight (n : ℕ) : Prop := sorry -- Define isKnight
def tellsTruth (n : ℕ) : Prop := sorry -- Define tellsTruth

-- Statements by the inhabitants
axiom H1 : isKnight 0 ↔ tellsTruth 0
axiom H2 : tellsTruth 1 ↔ isKnight 0
axiom H3 : tellsTruth 2 ↔ (¬(isKnight 0) ∨ ¬(isKnight 1))
axiom H4 : tellsTruth 3 ↔ (¬(isKnight 0) ∨ ¬(isKnight 1) ∨ ¬(isKnight 2))
axiom H5 : tellsTruth 4 ↔ (isKnight 0 ∧ isKnight 1 ∧ (isKnight 2 ∨ isKnight 3))
axiom H6 : tellsTruth 5 ↔ ((¬isKnight 0 ∨ ¬isKnight 1 ∨ isKnight 2) ∧ (¬isKnight 3 ∨ isKnight 4))
axiom H7 : tellsTruth 6 ↔ (isKnight 0 ∧ isKnight 1 ∧ isKnight 2 ∧ isKnight 3 ∧ ¬(¬isKnight 4 ∧ ¬isKnight 5))

theorem KnightsCount : ∃ k1 k2 k3 k4 k5 k6 k7 : Prop,
  (isKnight 0 = k1 ∧ isKnight 1 = k2 ∧ isKnight 2 = k3 ∧ isKnight 3 = k4 ∧ isKnight 4 = k5 ∧ isKnight 5 = k6 ∧ isKnight 6 = k7) ∧ 
  tellsTruth 0 ∧ tellsTruth 1 ∧ tellsTruth 2 ∧ tellsTruth 3 ∧ tellsTruth 4 ∧ tellsTruth 5 ∧ tellsTruth 6 ∧
  (5 = 1 + if k1 then 1 else 0 + if k2 then 1 else 0 + if k3 then 1 else 0 + if k4 then 1 else 0 + if k5 then 1 else 0 + if k6 then 1 else 0 + if k7 then 1 else 0)
:=
by
  sorry

end KnightsCount_l621_621455


namespace square_25_on_top_after_folds_l621_621123

theorem square_25_on_top_after_folds :
  let grid := (List.range 25).map (λ n => n + 1) in
  let step1_grid := [ [16, 17, 18, 19, 20],
                      [21, 22, 23, 24, 25],
                      [11, 12, 13, 14, 15],
                      [ 6,  7,  8,  9, 10],
                      [ 1,  2,  3,  4,  5] ] in
  let step2_grid := [ [19, 20, 18, 17, 16],
                      [24, 25, 23, 22, 21],
                      [14, 15, 13, 12, 11],
                      [ 9, 10,  8,  7,  6],
                      [ 4,  5,  3,  2,  1] ] in
  let step3_grid := [ [ 1,  6, 11, 16, 21],
                      [ 2,  7, 12, 17, 22],
                      [ 3,  8, 13, 18, 23],
                      [ 4,  9, 14, 19, 24],
                      [ 5, 10, 15, 20, 25] ] in
  let final_grid := [ [ 5, 10, 15, 20, 25],
                      [ 4,  9, 14, 19, 24],
                      [ 3,  8, 13, 18, 23],
                      [ 2,  7, 12, 17, 22],
                      [ 1,  6, 11, 16, 21] ] in
  final_grid.head.head = 25 := by
  sorry

end square_25_on_top_after_folds_l621_621123


namespace no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l621_621942

-- Conditions: Expressing the sum of three reciprocals
def sum_of_reciprocals (a b c : ℕ) : ℚ := (1 / a) + (1 / b) + (1 / c)

-- Proof Problem 1: Prove that the sum of the reciprocals of any three positive integers cannot equal 9/11
theorem no_three_reciprocals_sum_to_nine_eleven :
  ∀ (a b c : ℕ), sum_of_reciprocals a b c ≠ 9 / 11 := sorry

-- Proof Problem 2: Prove that there exists no rational number between 41/42 and 1 that can be expressed as the sum of the reciprocals of three positive integers other than 41/42
theorem no_rational_between_fortyone_fortytwo_and_one :
  ∀ (K : ℚ), 41 / 42 < K ∧ K < 1 → ¬ (∃ (a b c : ℕ), sum_of_reciprocals a b c = K) := sorry

end no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l621_621942


namespace smallest_value_a_plus_b_l621_621060

noncomputable def smallest_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) : ℝ :=
a + b

theorem smallest_value_a_plus_b :
  ∃ a b : ℝ, 
    (0 < a) ∧ (0 < b) ∧ (a^2 ≥ 12 * b) ∧ (9 * b^2 ≥ 4 * a) ∧ (smallest_a_plus_b a b _ _ _ _ = (32 * Real.sqrt 3) / 9) :=
sorry

end smallest_value_a_plus_b_l621_621060


namespace find_abc_l621_621173

theorem find_abc (a b c : ℕ) (h1 : (a + b + c) < 30) (h2 : (72 * (1.abc) = 72 * (1 + (abc / 999)))) : (a * 100 + b * 10 + c = 833) :=
sorry

end find_abc_l621_621173


namespace gabriel_pages_correct_l621_621243

-- Given conditions
def beatrix_pages : ℕ := 704

def cristobal_pages (b : ℕ) : ℕ := 3 * b + 15

def gabriel_pages (c b : ℕ) : ℕ := 3 * (c + b)

-- Problem statement
theorem gabriel_pages_correct : gabriel_pages (cristobal_pages beatrix_pages) beatrix_pages = 8493 :=
by 
  sorry

end gabriel_pages_correct_l621_621243


namespace square_length_XY_l621_621542

-- Define the entities and their properties
def Square (P Q R S : Type) := 
  (∃ l : ℝ, l = 15)

def Points (X Y : Type) := 
  (QX : ℝ, PY : ℝ, PX : ℝ, RY : ℝ)

-- Main theorem statement
theorem square_length_XY {P Q R S X Y : Type} 
  (sq : Square P Q R S) 
  (pts : Points X Y) 
  (h1 : pts.QX = 7) 
  (h2 : pts.PY = 7) 
  (h3 : pts.PX = 10) 
  (h4 : pts.RY = 10)
  : XY^2 = 323 :=
sorry

end square_length_XY_l621_621542


namespace knights_and_knaves_l621_621466

/-- Proof that there are exactly 5 knights among 7 inhabitants given specific conditions -/
theorem knights_and_knaves : 
  ∀ (inhabitants : Fin 7 → Prop)
    (knight truthful liar : Prop)
    (H1 : knight = truthful)
    (H2 : liar = ¬truthful)
    (H3 : (inhabitants 0 = knight))
    (H4 : (inhabitants 1 = knight))
    (H5 : (inhabitants 2 = liar ∧ (inhabitants 0 = knight ∨ inhabitants 1 = knight)))
    (H6 : (inhabitants 3 = liar ∧ (inhabitants 0 = truth ∧ inhabitants 1 = truth ∧ inhabitants 2 = liar → 2/3 ≥ 0.65)))
    (H7 : (inhabitants 4 = knight))
    (H8 : (inhabitants 5 = knight ∧ (∃ (half_knights : Fin 6 → Prop), (inhabitants 0 = knight) ∧ (inhabitants 1 = knight) ∧ (inhabitants 4 = knight) ∧ (inhabitants 5 = knight) ∧ counting_knaves half_knights < 3))
    (H9 : (inhabitants 6 = knight ∧ (counting_knights inhabitants 6 ≥ 0.65))),
  counting_knights inhabitants = 5 := 
by sorry

def counting_knights (inhabitants : Fin 7 → Prop) : ℝ := sorry
def counting_knaves (inhabitants : Fin 7 → Prop) : ℝ := sorry

end knights_and_knaves_l621_621466


namespace chelsea_batches_l621_621237

/-- Define the baking time per batch, icing time per batch,
    total time spent, and compute the number of batches
    proving it equals to 4. 
-/
def baking_time_per_batch : ℕ := 20
def icing_time_per_batch : ℕ := 30
def total_time_spent : ℕ := 200

-- Define the total time per batch
def total_time_per_batch : ℕ := baking_time_per_batch + icing_time_per_batch

-- Compute the number of batches
def number_of_batches : ℕ := total_time_spent / total_time_per_batch

-- Proof that Chelsea made 4 batches of cupcakes
theorem chelsea_batches : number_of_batches = 4 := by
  unfold number_of_batches total_time_per_batch
  simp [baking_time_per_batch, icing_time_per_batch, total_time_spent]
  sorry

end chelsea_batches_l621_621237


namespace ceil_of_neg_sqrt_frac_64_over_9_l621_621290

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l621_621290


namespace minimum_participants_l621_621536

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l621_621536


namespace distance_from_focus_to_asymptote_of_hyperbola_l621_621347

theorem distance_from_focus_to_asymptote_of_hyperbola (a : ℝ) (h : 0 < a) :
  let F := (real.sqrt (3 * a + 3), 0)
  let A := (x : ℝ) := (x, 1/real.sqrt(a) * x)
  let B := (x : ℝ) := (x, -1/real.sqrt(a) * x)
  let distance_to_line (x1 y1 a b c : ℝ) := abs (a * x1 + b * y1 + c) / real.sqrt (a^2 + b^2)
  (distance_to_line F.1 F.2 1 (-real.sqrt(a)) 0 = real.sqrt 3) ∨
  (distance_to_line F.1 F.2 1 (real.sqrt(a)) 0 = real.sqrt 3) :=
begin
  -- Proof goes here
  sorry
end

end distance_from_focus_to_asymptote_of_hyperbola_l621_621347


namespace cauliflower_difference_is_401_l621_621204

-- Definitions using conditions from part a)
def garden_area_this_year : ℕ := 40401
def side_length_this_year : ℕ := Nat.sqrt garden_area_this_year
def side_length_last_year : ℕ := side_length_this_year - 1
def garden_area_last_year : ℕ := side_length_last_year ^ 2
def cauliflowers_difference : ℕ := garden_area_this_year - garden_area_last_year

-- Problem statement claiming that the difference in cauliflowers produced is 401
theorem cauliflower_difference_is_401 :
  garden_area_this_year = 40401 →
  side_length_this_year = 201 →
  side_length_last_year = 200 →
  garden_area_last_year = 40000 →
  cauliflowers_difference = 401 :=
by
  intros
  sorry

end cauliflower_difference_is_401_l621_621204


namespace range_of_a_l621_621323

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≥ 2 then (a - 2) * x else (1 / 2) ^ x - 1

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) →
  a ∈ Iic (13 / 8) :=
begin
  sorry
end

end range_of_a_l621_621323


namespace det_matrix_A_l621_621923

def matrix_S : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 0], ![0, 5]]
def matrix_R : Matrix (Fin 2) (Fin 2) ℝ := ![![0, -1], ![1, 0]]
def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := matrix_S ⬝ matrix_R

theorem det_matrix_A : det matrix_A = 25 := by
  sorry

end det_matrix_A_l621_621923


namespace _l621_621441

noncomputable def concave_fn (f : ℝ → ℝ) : Prop :=
∀ x y z : ℝ, 0 < x ∧ x < y ∧ y < z →
  (f (y ⁻¹) - f (x ⁻¹)) / (y - x) ≤ (f (z ⁻¹) - f (y ⁻¹)) / (z - y)

noncomputable def convex_fn (f : ℝ → ℝ) : Prop :=
∀ x y z : ℝ, 0 < x ∧ x < y ∧ y < z →
  (f (z ⁻¹) - f (y ⁻¹)) * (y - x) - (f (y ⁻¹) - f (x ⁻¹)) * (z - y) ≥ 0

noncomputable theorem convex_x_f
  (f : ℝ → ℝ)
  (h1 : concave_fn f)
  (h2 : convex_fn (λ x : ℝ, f (x ⁻¹))) :
  (∀ x : ℝ, 0 < x → is_convex (λ x, x * f x)) ∧
  (∀ ξ : ℝ, 0 < ξ → 
    (1 / ENNReal.toReal (∫ ξ⁻¹)) ≤ f⁻¹ (ENNReal.toReal (∫ f(ξ))) ∧
    f⁻¹ (ENNReal.toReal (∫ f(ξ))) ≤ ENNReal.toReal (∫ ξ) ∧
    ENNReal.toReal (∫ ξ) ≤ f⁻¹ (ENNReal.toReal ((∫ (ξ * f(ξ))) / (∫ ξ))) ∧
    f⁻¹ (ENNReal.toReal ((∫ (ξ * f(ξ))) / (∫ ξ))) ≤ ENNReal.toReal ((∫ ξ^2) / (∫ ξ))) ∧
  (strict_increasing f ∨ constant f) :=
by sorry

end _l621_621441


namespace part1_part2_l621_621841

noncomputable def f (x : Real) (n : Real) : Real :=
  if 0 ≤ x ∧ x < n then Real.sqrt x else (x - 1) ^ 2

noncomputable def g (x : Real) (a : Real) (n : Real) : Real :=
  x^2 + a * (Real.abs (x - n))

theorem part1 (n : Real) (h1 : 1 < n) (h2 : ∃ b : Real, (∃ x1 x2 : Real, f x1 n = b ∧ f x2 n = b ∧ x1 ≠ x2)) :
  n = 2 := sorry

theorem part2 (a : Real) (h : ∀ x y : Real, 0 ≤ x ∧ x < y → g x a 2 ≤ g y a 2) :
  -4 ≤ a ∧ a ≤ 0 := sorry

end part1_part2_l621_621841


namespace isosceles_triangle_l621_621140

variable (A B C K N M : Point)
variable (ABC : Triangle A B C)
variable (incircle : Circle)

-- Condition 1: The inscribed circle of triangle ABC touches its sides at points K, N, M.
axiom incircle_touches_sides (hc : inscribed incircle ABC) 
  (hK : incircle.touches_side incircle ABC.side_AB K)
  (hN : incircle.touches_side incircle ABC.side_BC N)
  (hM : incircle.touches_side incircle ABC.side_CA M)

-- Condition 2: Given angle condition
axiom angle_condition : ∠ A N M = ∠ C K M

-- Question: Prove that the triangle ABC is isosceles.
theorem isosceles_triangle :
  is_isosceles ABC :=
sorry

end isosceles_triangle_l621_621140


namespace bad_arrangements_count_l621_621978

inductive arrangement : Type
| mk (a b c d e : ℕ)

def bad_arrangement (arr : arrangement) : Prop :=
  ∃ n, (1 ≤ n ∧ n ≤ 15) ∧ ¬ (∃ subset : list ℕ, subset.nodup ∧ subset ≠ [] ∧ subset.sum = n ∧ 
    -- Note that we need consecutive numbers, so we would need a separate condition to check
    (arr.contains_consecutive subset))

-- Since we are counting distinct arrangements considering rotations and reflections, and defining contains_consecutive
def correct_arrangement_count : ℕ := 
  -- the specific calculation for the bad arrangements considering rotations and reflections 
  2

theorem bad_arrangements_count : 
  ∀ arrangements : list arrangement, (list bad_arrangement arrangements).length = 2 :=
sorry

end bad_arrangements_count_l621_621978


namespace jimmy_bought_3_pens_l621_621033

def cost_of_notebooks (num_notebooks : ℕ) (price_per_notebook : ℕ) : ℕ := num_notebooks * price_per_notebook
def cost_of_folders (num_folders : ℕ) (price_per_folder : ℕ) : ℕ := num_folders * price_per_folder
def total_cost (cost_notebooks cost_folders : ℕ) : ℕ := cost_notebooks + cost_folders
def total_spent (initial_money change : ℕ) : ℕ := initial_money - change
def cost_of_pens (total_spent amount_for_items : ℕ) : ℕ := total_spent - amount_for_items
def num_pens (cost_pens price_per_pen : ℕ) : ℕ := cost_pens / price_per_pen

theorem jimmy_bought_3_pens :
  let pen_price := 1
  let notebook_price := 3
  let num_notebooks := 4
  let folder_price := 5
  let num_folders := 2
  let initial_money := 50
  let change := 25
  let cost_notebooks := cost_of_notebooks num_notebooks notebook_price
  let cost_folders := cost_of_folders num_folders folder_price
  let total_items_cost := total_cost cost_notebooks cost_folders
  let amount_spent := total_spent initial_money change
  let pen_cost := cost_of_pens amount_spent total_items_cost
  num_pens pen_cost pen_price = 3 :=
by
  sorry

end jimmy_bought_3_pens_l621_621033


namespace incorrect_statement_of_genetic_material_l621_621603

-- Definition for Meiosis only occurring in sexually reproducing organisms
def meiosis_sexually_reproducing (organism : Type) : Prop :=
  organism.seuallyReproducing → meiosis

-- Definition for Recombination of non-allelic genes during meiosis
def recombination_non_allelic_genes : Prop := 
  ∃ (gene : Type), recombination gene.non_allelic

-- Definition for genetic material division in fertilized egg
def genetic_material_division (egg sperm : Type) : Prop :=
  (half_genetic_material egg sperm) ∧ cytoplasmic_material egg

-- Definition for maintaining constant chromosome number
def constant_chromosome_number : Prop :=
  meiosis ∧ fertilization → constant_chromosome

-- Main theorem to prove C is incorrect
theorem incorrect_statement_of_genetic_material : 
  ¬ (∀ (egg sperm : Type), genetic_material_division egg sperm) := 
sorry

end incorrect_statement_of_genetic_material_l621_621603


namespace race_minimum_participants_l621_621520

theorem race_minimum_participants :
  ∃ n : ℕ, ∀ m : ℕ, (m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0 ↔ m = n :=
begin
  let m := 61,
  use m,
  intro k,
  split,
  { intro h,
    cases h with h3 h45,
    cases h45 with h4 h5,
    have h3' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 3)).mp h3,
    have h4' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 4)).mp h4,
    have h5' := Nat.ModEq.symm (Nat.dvd_add_iff_right (Nat.dvd_one_add_self 5)).mp h5,
    have lcm_3_4_5 := Nat.lcm_eq (And.intro h3' (And.intro h4' h5')),
    exact Nat.eq_of_lcm_dvd 1 lcm_3_4_5 },
  { intro hk,
    rw hk,
    split,
    { exact Nat.eq_of_mod_eq (by {norm_num}) },
    { split; exact Nat.eq_of_mod_eq (by {norm_num}) }
  }
end

end race_minimum_participants_l621_621520


namespace eqn_has_three_real_solutions_l621_621922

noncomputable def real_solution_count : ℝ → ℝ := λ x, ⌊2 * x⌋ + ⌊3 * x⌋ - (8 * x - 6)

theorem eqn_has_three_real_solutions :
  ∃ x1 x2 x3 : ℝ, real_solution_count x1 = 0 ∧ real_solution_count x2 = 0 ∧ real_solution_count x3 = 0 ∧
  (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
sorry

end eqn_has_three_real_solutions_l621_621922


namespace problem_statement_l621_621372

theorem problem_statement (x : ℕ) (h : 16^6 = 8^x) : 2^(-x : ℤ) = 1 / 256 := by
  have h1 : 16 = 2^4 := by norm_num
  have h2 : 8  = 2^3 := by norm_num
  rw [h1, h2] at h
  have h3 : (2^4)^6 = (2^3)^x := by assumption
  norm_num at h3
  sorry

end problem_statement_l621_621372


namespace intersection_sum_l621_621831

theorem intersection_sum (h j : ℝ → ℝ)
  (H1 : h 3 = 3 ∧ j 3 = 3)
  (H2 : h 6 = 9 ∧ j 6 = 9)
  (H3 : h 9 = 18 ∧ j 9 = 18)
  (H4 : h 12 = 18 ∧ j 12 = 18) :
  ∃ a b : ℕ, h (3 * a) = b ∧ 3 * j a = b ∧ (a + b = 33) :=
by {
  sorry
}

end intersection_sum_l621_621831


namespace possible_values_of_phi_even_function_l621_621840

def f (x : ℝ) : ℝ := (1 / 2) * Real.sin (2 * x + (Real.pi / 6))

theorem possible_values_of_phi_even_function (φ : ℝ) :
  (∀ x : ℝ, f (x - φ) = f (-(x - φ))) ↔ φ = -Real.pi / 6 :=
sorry

end possible_values_of_phi_even_function_l621_621840


namespace cost_of_balloons_l621_621445

variable holdMoney cake bouquet extraMoneyNeeded : ℕ

theorem cost_of_balloons (holdMoney : ℕ) (cake : ℕ) (bouquet : ℕ) (extraMoneyNeeded : ℕ) 
  (h1 : holdMoney = 50)
  (h2 : cake = 20)
  (h3 : bouquet = 36)
  (h4 : extraMoneyNeeded = 11) : 
  let totalCost := holdMoney + extraMoneyNeeded
  let combinedCost := cake + bouquet
  totalCost - combinedCost = 5 := by
    sorry

end cost_of_balloons_l621_621445


namespace total_students_surveyed_l621_621203

variable (F E S FE FS ES FES N T : ℕ)

def only_one_language := 230
def exactly_two_languages := 190
def all_three_languages := 40
def no_language := 60

-- Summing up all categories
def total_students := only_one_language + exactly_two_languages + all_three_languages + no_language

theorem total_students_surveyed (h1 : F + E + S = only_one_language) 
    (h2 : FE + FS + ES = exactly_two_languages) 
    (h3 : FES = all_three_languages) 
    (h4 : N = no_language) 
    (h5 : T = F + E + S + FE + FS + ES + FES + N) : 
    T = total_students :=
by
  rw [total_students, only_one_language, exactly_two_languages, all_three_languages, no_language]
  sorry

end total_students_surveyed_l621_621203


namespace cube_vertex_face_sum_product_l621_621018

theorem cube_vertex_face_sum_product :
  let possible_sums := {14, 10, 6, 2, -2, -6, -10}
  ∃ (vertices : Fin 8 → ℤ) (faces : Fin 6 → ℤ), 
  (∀ i, vertices i = 1 ∨ vertices i = -1) ∧
  (∀ j, faces j = vertices (face_vertex_index j 0) * vertices (face_vertex_index j 1) * 
                   vertices (face_vertex_index j 2) * vertices (face_vertex_index j 3)) ∧
  ∃ (sums : List ℤ), (∀ sum ∈ sums, sum ∈ possible_sums) ∧
  (List.prod sums) = -20160 := by
  sorry

end cube_vertex_face_sum_product_l621_621018


namespace cube_side_length_equals_six_l621_621571

theorem cube_side_length_equals_six {s : ℝ} (h : 6 * s ^ 2 = s ^ 3) : s = 6 :=
by
  sorry

end cube_side_length_equals_six_l621_621571


namespace david_more_pushups_than_zachary_l621_621610

def zacharyPushUps : ℕ := 59
def davidPushUps : ℕ := 78

theorem david_more_pushups_than_zachary :
  davidPushUps - zacharyPushUps = 19 :=
by
  sorry

end david_more_pushups_than_zachary_l621_621610


namespace length_of_sheetrock_l621_621208

variables (width area length : ℕ)

-- Defining the given conditions
def width_condition := width = 5
def area_condition := area = 30
def length_condition := length = 6

-- The main goal is to prove that the length is equal to 6 feet given the conditions
theorem length_of_sheetrock (h1 : width_condition) (h2 : area_condition) : length_condition :=
by sorry

end length_of_sheetrock_l621_621208


namespace eggs_in_each_basket_l621_621414

theorem eggs_in_each_basket :
  ∃ x : ℕ, x ∣ 30 ∧ x ∣ 42 ∧ x ≥ 5 ∧ x = 6 :=
by
  sorry

end eggs_in_each_basket_l621_621414


namespace quadratic_m_condition_l621_621863

theorem quadratic_m_condition (m : ℝ) (h_eq : (m - 2) * x ^ (m ^ 2 - 2) - m * x + 1 = 0) (h_pow : m ^ 2 - 2 = 2) :
  m = -2 :=
by sorry

end quadratic_m_condition_l621_621863


namespace a_10_eq_18_l621_621010

variable {a : ℕ → ℕ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

axiom a2 : a 2 = 2
axiom a3 : a 3 = 4
axiom arithmetic_seq : is_arithmetic_sequence a

-- problem: prove a_{10} = 18
theorem a_10_eq_18 : a 10 = 18 :=
sorry

end a_10_eq_18_l621_621010


namespace fifteen_percent_of_x_is_ninety_l621_621767

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l621_621767


namespace find_x_when_y4_l621_621957

theorem find_x_when_y4 
  (k : ℝ) 
  (h_var : ∀ y : ℝ, ∃ x : ℝ, x = k * y^2)
  (h_initial : ∃ x : ℝ, x = 6 ∧ 1 = k) :
  ∃ x : ℝ, x = 96 :=
by 
  sorry

end find_x_when_y4_l621_621957


namespace partI_solution_partII_solution_l621_621354

-- Part (I)
theorem partI_solution (x : ℝ) (a : ℝ) (h : a = 5) : (|x + a| + |x - 2| > 9) ↔ (x < -6 ∨ x > 3) :=
by
  sorry

-- Part (II)
theorem partII_solution (a : ℝ) :
  (∀ x : ℝ, (|2*x - 1| ≤ 3) → (|x + a| + |x - 2| ≤ |x - 4|)) → (-1 ≤ a ∧ a ≤ 0) :=
by
  sorry

end partI_solution_partII_solution_l621_621354


namespace committee_formations_l621_621803

theorem committee_formations :
  let students := 11
  let teachers := 3
  let total_people := students + teachers
  let committee_size := 8
  (nat.choose total_people committee_size) - (nat.choose students committee_size) = 2838 :=
by
  sorry

end committee_formations_l621_621803


namespace ceil_of_neg_sqrt_frac_64_over_9_l621_621287

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l621_621287


namespace race_participants_minimum_l621_621509

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l621_621509


namespace minimum_participants_l621_621493

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l621_621493


namespace number_of_liars_is_two_l621_621887

def Piglet := Type
variables (Nif_Nif Naf_Naf Nuf_Nuf : Piglet)

-- Each piglet either always lies or always tells the truth
def always_lies (p : Piglet) : Prop := sorry
def always_tells_truth (p : Piglet) : Prop := sorry

-- Each piglet knows whether the others are liars
def knows_truthfulness (p1 p2 : Piglet) : Prop := sorry

-- Statements made by two of the piglets
def statement1 := always_lies Nif_Nif ∧ always_lies Naf_Naf 
def statement2 := always_lies Nif_Nif ∧ always_lies Nuf_Nuf

theorem number_of_liars_is_two 
  (h1 : always_lies Nif_Nif ∨ always_tells_truth Nif_Nif)
  (h2 : always_lies Naf_Naf ∨ always_tells_truth Naf_Naf)
  (h3 : always_lies Nuf_Nuf ∨ always_tells_truth Nuf_Nuf)
  (h_statement1 : statement1 = true ∨ statement1 = false)
  (h_statement2 : statement2 = true ∨ statement2 = false) :
  (∑ i in [Nif_Nif, Naf_Naf, Nuf_Nuf], if always_lies i then 1 else 0) = 2 :=
sorry

end number_of_liars_is_two_l621_621887


namespace staff_discount_l621_621198

open Real

theorem staff_discount (d : ℝ) (h : d > 0) (final_price_eq : 0.14 * d = 0.35 * d * (1 - 0.6)) : 0.6 * 100 = 60 :=
by
  sorry

end staff_discount_l621_621198


namespace line_passes_through_fixed_point_l621_621141

theorem line_passes_through_fixed_point (k : ℝ) (x y : ℝ) :
  k * x - y + 1 - 3 * k = 0 → (3, 1) = (3 : ℝ, 1 : ℝ) :=
sorry

end line_passes_through_fixed_point_l621_621141


namespace factor_expression_l621_621240

theorem factor_expression (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = (a - b) * (b - c) * (c - a) * (a * b^2 + a * c^2) :=
by 
  sorry

end factor_expression_l621_621240


namespace central_ring_count_l621_621572

-- Given definitions for the problem
def is_central_ring (arrangement : List ℕ) : Prop :=
  let sums := (List.sublists arrangement).filter (λ l, l.sum ∈ [1, 2, 3, 4, 5, 15, 14, 13, 12, 11, 10])
  sums.length = 0

-- Main statement to prove
theorem central_ring_count : 
  ∃ (configs : List (List ℕ)), (configs.length = 2) ∧
  (∀ config ∈ configs, is_central_ring config) ∧
  (∀ c1 c2 ∈ configs, (c1 ≠ c2 → ¬(c1.rotate_eq c2 ∨ c1.reflect_eq c2))) :=
sorry

end central_ring_count_l621_621572


namespace total_cupcakes_needed_l621_621154

-- Definitions based on conditions
def cupcakes_per_event : ℝ := 96.0
def number_of_events : ℝ := 8.0

-- Theorem based on the question and the correct answer
theorem total_cupcakes_needed : (cupcakes_per_event * number_of_events) = 768.0 :=
by 
  sorry

end total_cupcakes_needed_l621_621154


namespace race_participants_minimum_l621_621525

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l621_621525


namespace no_polygon_with_half_parallel_diagonals_l621_621748

open Set

noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

def is_parallel_diagonal (n i j : ℕ) : Bool := 
  -- Here, you should define the mathematical condition of a diagonal being parallel to a side
  ((j - i) % n = 0) -- This is a placeholder; the actual condition would depend on the precise geometric definition.

theorem no_polygon_with_half_parallel_diagonals (n : ℕ) (h1 : n ≥ 3) :
  ¬(∃ (k : ℕ), k = num_diagonals n ∧ (∀ (i j : ℕ), i < j ∧ is_parallel_diagonal n i j = true → k = num_diagonals n / 2)) :=
by
  sorry

end no_polygon_with_half_parallel_diagonals_l621_621748


namespace probability_odd_die_l621_621650

theorem probability_odd_die :
  let faces := [1, 1, 1, 2, 3, 3] in
  let total_faces := faces.length in
  let odd_faces := (faces.countp (λ x => x % 2 = 1)) in
  total_faces = 6 →
  (odd_faces : ℚ) / total_faces = 5 / 6 :=
by
  sorry

end probability_odd_die_l621_621650


namespace total_ticket_count_is_59_l621_621998

-- Define the constants and variables
def price_adult : ℝ := 4
def price_student : ℝ := 2.5
def total_revenue : ℝ := 222.5
def student_tickets_sold : ℕ := 9

-- Define the equation representing the total revenue and solve for the number of adult tickets
noncomputable def total_tickets_sold (adult_tickets : ℕ) :=
  adult_tickets + student_tickets_sold

theorem total_ticket_count_is_59 (A : ℕ) 
  (h : price_adult * A + price_student * (student_tickets_sold : ℝ) = total_revenue) :
  total_tickets_sold A = 59 :=
by
  sorry

end total_ticket_count_is_59_l621_621998


namespace graph_squares_above_line_l621_621556

theorem graph_squares_above_line : 
  let line_eq := λ (x y : ℝ), 5 * x + 255 * y = 2550
  ∃ (n : ℕ), n = 2295 ∧ (∀ (x y : ℕ), 1 ≤ x ∧ x ≤ 510 ∧ 1 ≤ y ∧ y ≤ 10 ∧ (¬ line_eq (x - 1) y ∧ ¬ line_eq x y) → y > (-5/255) * x + 10) ∧
  (∀ (x y : ℕ), 1 ≤ x ∧ x ≤ 510 ∧ 1 ≤ y ∧ y ≤ 10 → n = 2295) :=
sorry

end graph_squares_above_line_l621_621556


namespace numbers_in_100th_bracket_l621_621665

def bracket_number_sequence (n : ℕ) : list ℕ :=
if n % 3 = 1 then [2*n - 1]
else if n % 3 = 2 then [2*n, 2*n + 2]
else [2*n + 3, 2*n + 5, 2*n + 7]

theorem numbers_in_100th_bracket :
  bracket_number_sequence 100 = [65, 67] :=
begin
  sorry
end

end numbers_in_100th_bracket_l621_621665


namespace range_of_dot_product_l621_621835

theorem range_of_dot_product
  (x y : ℝ)
  (on_ellipse : x^2 / 2 + y^2 = 1) :
  ∃ m n : ℝ, (m = 0) ∧ (n = 1) ∧ m ≤ x^2 / 2 ∧ x^2 / 2 ≤ n :=
sorry

end range_of_dot_product_l621_621835


namespace meaningful_sqrt_neg_x_squared_l621_621799

theorem meaningful_sqrt_neg_x_squared (x : ℝ) : (x = 0) ↔ (-(x^2) ≥ 0) :=
by
  sorry

end meaningful_sqrt_neg_x_squared_l621_621799


namespace gift_sequences_count_l621_621937

def num_students : ℕ := 11
def num_meetings : ℕ := 4
def sequences : ℕ := num_students ^ num_meetings

theorem gift_sequences_count : sequences = 14641 := by
  sorry

end gift_sequences_count_l621_621937


namespace stuart_initial_marbles_l621_621217

theorem stuart_initial_marbles (B S : ℝ) (h1 : B = 60) (h2 : 0.40 * B = 24) (h3 : S + 24 = 80) : S = 56 :=
by
  sorry

end stuart_initial_marbles_l621_621217


namespace number_of_knights_is_five_l621_621462

section KnightsAndKnaves

inductive Inhabitant
| knight : Inhabitant
| knave : Inhabitant

open Inhabitant

variables (a1 a2 a3 a4 a5 a6 a7 : Inhabitant)

def tells_truth : Inhabitant → Prop
| knight := True
| knave := False

def statements : Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → Inhabitant → ℕ → Prop
| a1 a2 _ _ _ _ _ 1 := (a1 = knight)
| a1 a2 _ _ _ _ _ 2 := (a1 = knight ∧ a2 = knight)
| a1 a2 a3 _ _ _ _ 3 := (a1 = knave ∨ a2 = knave)
| a1 a2 a3 a4 _ _ _ 4 := (a1 = knave ∨ a2 = knave ∨ a3 = knave)
| a1 a2 a3 a4 a5 _ _ 5 := (a1 = knight ∧ a2 = knight ∨ a1 = knave ∧ a2 = knave)
| a1 a2 a3 a4 a5 a6 _ 6 := (a1 = knave ∨ a2 = knave ∨ a3 = knave ∨ a4 = knave ∨ a5 = knave)
| a1 a2 a3 a4 a5 a6 a7 7 := (a1 = knight ∨ a2 = knight ∨ a3 = knight ∨ a4 = knight ∨ a5 = knight ∨ a6 = knight)

theorem number_of_knights_is_five (h1 : tells_truth a1 ↔ a1 = knight)
                                   (h2 : tells_truth a2 ↔ a2 = knight)
                                   (h3 : tells_truth a3 ↔ a3 = knight)
                                   (h4 : tells_truth a4 ↔ a4 = knight)
                                   (h5 : tells_truth a5 ↔ a5 = knight)
                                   (h6 : tells_truth a6 ↔ a6 = knight)
                                   (h7 : tells_truth a7 ↔ a7 = knight)
                                   (s1 : tells_truth a1 → statements a1 a2 a3 a4 a5 a6 a7 1)
                                   (s2 : tells_truth a2 → statements a1 a2 a3 a4 a5 a6 a7 2)
                                   (s3 : tells_truth a3 → statements a1 a2 a3 a4 a5 a6 a7 3)
                                   (s4 : tells_truth a4 → statements a1 a2 a3 a4 a5 a6 a7 4)
                                   (s5 : tells_truth a5 → statements a1 a2 a3 a4 a5 a6 a7 5)
                                   (s6 : tells_truth a6 → statements a1 a2 a3 a4 a5 a6 a7 6)
                                   (s7 : tells_truth a7 → statements a1 a2 a3 a4 a5 a6 a7 7)
                                   : (∀ i, (i = knight → tells_truth i ¬ ↔ i = knight)) → ∃ (n : ℕ), n = 5 := 
sorry

end KnightsAndKnaves

end number_of_knights_is_five_l621_621462


namespace find_f3_l621_621845

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_f3
  (a b : ℝ)
  (h1 : f a b 3 1 = 7)
  (h2 : f a b 3 2 = 12) :
  f a b 3 3 = 18 :=
sorry

end find_f3_l621_621845


namespace squares_in_H_with_side_at_least_8_l621_621738

-- Define the set H and the relevant constraints
def H : set (ℤ × ℤ) := 
  { p : (ℤ × ℤ) | -8 ≤ p.1 ∧ p.1 ≤ 8 ∧ -8 ≤ p.2 ∧ p.2 ≤ 8 }

-- State the theorem
theorem squares_in_H_with_side_at_least_8 : 
  (finset.filter (λ s : finset (ℤ × ℤ), 
    ∃ (x1 y1 x2 y2 : ℤ), 
      (x1, y1) ∈ H ∧ 
      (x2, y2) ∈ H ∧ 
      s = {(x1, y1), (x1, y2), (x2, y1), (x2, y2)} ∧ 
      x2 - x1 = 8 ∧ y2 - y1 = 8) 
  {s : finset (ℤ × ℤ) | s.card = 4}).card = 4 := 
sorry

end squares_in_H_with_side_at_least_8_l621_621738


namespace songs_in_each_album_l621_621182

variable (X : ℕ)

theorem songs_in_each_album (h : 6 * X + 2 * X = 72) : X = 9 :=
by sorry

end songs_in_each_album_l621_621182


namespace length_PR_l621_621059

-- Given conditions
variables {P Q R S T : Type*}
variables [is_right_triangle : right_triangle P Q R] -- ∠PQR = 90°
variables [is_midpoint_S : midpoint S P Q] -- S is the midpoint of PQ
variables [is_midpoint_T : midpoint T P R] -- T is the midpoint of PR
variables (QT_length : length Q T = 25) -- QT = 25
variables (SP_length : length S P = 27) -- SP = 27

-- Goal: Prove PR = 2 * sqrt(2291 / 15)
theorem length_PR (x y : ℝ) (hx : x = sqrt(1771 / 15)) (hy : y = sqrt(2291 / 15)) : 
  PR = 2 * y :=
by sorry

end length_PR_l621_621059


namespace factorial_divisibility_6k_l621_621375

theorem factorial_divisibility_6k (k : ℕ) : 
  (∃ k, factorial 14 % (6 ^ k) = 0) → k ≤ 5 ∧ (factorial 14 % (6 ^ (k + 1)) ≠ 0) := 
by 
  sorry

end factorial_divisibility_6k_l621_621375


namespace find_unknown_number_l621_621776

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l621_621776


namespace point_P_is_in_third_quadrant_l621_621401

-- Define a type to represent quadrants
inductive Quadrant
| first  : Quadrant
| second : Quadrant
| third  : Quadrant
| fourth : Quadrant

-- Define a function that determines the quadrant of a point in the Cartesian coordinate system
def point_quadrant (x y : ℤ) : Quadrant :=
if x > 0 ∧ y > 0 then Quadrant.first
else if x < 0 ∧ y > 0 then Quadrant.second
else if x < 0 ∧ y < 0 then Quadrant.third
else Quadrant.fourth

-- The point P is given as (-1, -2)
def P := (-1, -2) : ℤ × ℤ

-- The theorem we want to prove
theorem point_P_is_in_third_quadrant : point_quadrant P.1 P.2 = Quadrant.third :=
by
  sorry

end point_P_is_in_third_quadrant_l621_621401


namespace wood_blocks_after_days_l621_621944

-- Defining the known conditions
def blocks_per_tree : Nat := 3
def trees_per_day : Nat := 2
def days : Nat := 5

-- Stating the theorem to prove the total number of blocks of wood after 5 days
theorem wood_blocks_after_days : blocks_per_tree * trees_per_day * days = 30 :=
by
  sorry

end wood_blocks_after_days_l621_621944


namespace solve_rational_inequality_l621_621116

theorem solve_rational_inequality :
  {x : ℝ | (9*x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 4} =
  {x : ℝ | (-10 < x ∧ x < -5) ∨ (2/3 < x ∧ x < 4/3) ∨ (4/3 < x)} :=
by
  sorry

end solve_rational_inequality_l621_621116


namespace equivalent_expression_for_g_l621_621927

theorem equivalent_expression_for_g (x : ℝ) :
  (sqrt (3 * (sin x)^4 + 5 * (cos x)^2)) - (sqrt (3 * (cos x)^4 + 5 * (sin x)^2)) = 3 :=
by
  sorry

end equivalent_expression_for_g_l621_621927


namespace problem_statement_l621_621559

theorem problem_statement (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ) 
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) : 
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := 
sorry

end problem_statement_l621_621559


namespace find_a_l621_621335

variable (a : ℝ)

def p (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def q : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}
def q_negation : Set ℝ := {x | 1 < x ∧ x < 3}

theorem find_a :
  (∀ x, q_negation x → p a x) → a = 2 := by
  sorry

end find_a_l621_621335


namespace ellipse_and_quadrilateral_l621_621008

noncomputable def ellipse_equation (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) = (1, 1.5))

noncomputable def max_area_quadrilateral (S : ℝ) : Prop :=
  ∀ P Q : ℝ × ℝ, P = (-2, 0) ∧ Q = (2, 0) → 
  ∀ m : ℝ, ( ∃ A B : ℝ × ℝ, 
    A ≠ B ∧ (A = (x, y) ∧ B = (x, y) ∧ 
    x = my + 1 ∧
    y^2 (3m^2 + 4) + 6my - 9 = 0) ∧
    (S ≤ 6))

theorem ellipse_and_quadrilateral (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (∃ c : ℝ,
    a = 2 * c ∧ 
    ellipse_equation 2c b h₀ h₁ 
  ∧ (max_area_quadrilateral 6)) :=
begin
  sorry
end

end ellipse_and_quadrilateral_l621_621008


namespace eccentricity_of_ellipse_l621_621834

theorem eccentricity_of_ellipse
  (a b : ℝ) (a_gt_b : a > b) (b_pos : b > 0)
  (hyp1 : ∀ y x : ℝ, (y = 0 ∧ x = √5) → (y^2 / a^2 + x^2 / b^2 = 1))
  (hyp2 : ∀ y x : ℝ, (y = 3 ∧ x = 0) → (y^2 / a^2 + x^2 / b^2 = 1)) :
  let c := Real.sqrt (a^2 - b^2) in
  let e := c / a in
  e = 2 / 3 :=
by
  sorry

end eccentricity_of_ellipse_l621_621834


namespace olympic_iberic_sets_containing_33_l621_621666

/-- A set of positive integers is iberic if it is a subset of {2, 3, ..., 2018},
    and whenever m, n are both in the set, gcd(m, n) is also in the set. -/
def is_iberic_set (X : Set ℕ) : Prop :=
  X ⊆ {n | 2 ≤ n ∧ n ≤ 2018} ∧ ∀ m n, m ∈ X → n ∈ X → Nat.gcd m n ∈ X

/-- An iberic set is olympic if it is not properly contained in any other iberic set. -/
def is_olympic_set (X : Set ℕ) : Prop :=
  is_iberic_set X ∧ ∀ Y, is_iberic_set Y → X ⊂ Y → False

/-- The olympic iberic sets containing 33 are exactly {3, 6, 9, ..., 2016} and {11, 22, 33, ..., 2013}. -/
theorem olympic_iberic_sets_containing_33 :
  ∀ X, is_iberic_set X ∧ 33 ∈ X → X = {n | 3 ∣ n ∧ 2 ≤ n ∧ n ≤ 2016} ∨ X = {n | 11 ∣ n ∧ 11 ≤ n ∧ n ≤ 2013} :=
by
  sorry

end olympic_iberic_sets_containing_33_l621_621666


namespace perpendicular_tangent_inequality_l621_621989

variable {A B C : Type} 

-- Definitions according to conditions in part a)
def isAcuteAngledTriangle (a b c : Type) : Prop :=
  -- A triangle being acute-angled in Euclidean geometry
  sorry

def triangleArea (a b c : Type) : ℝ :=
  -- Definition of the area of a triangle
  sorry

def perpendicularLengthToLine (point line : Type) : ℝ :=
  -- Length of the perpendicular from a point to a line
  sorry

def tangentOfAngleA (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle A in the triangle
  sorry

def tangentOfAngleB (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle B in the triangle
  sorry

def tangentOfAngleC (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle C in the triangle
  sorry

theorem perpendicular_tangent_inequality (a b c line : Type) 
  (ht : isAcuteAngledTriangle a b c)
  (u := perpendicularLengthToLine a line)
  (v := perpendicularLengthToLine b line)
  (w := perpendicularLengthToLine c line):
  u^2 * tangentOfAngleA a b c + v^2 * tangentOfAngleB a b c + w^2 * tangentOfAngleC a b c ≥ 
  2 * triangleArea a b c :=
sorry

end perpendicular_tangent_inequality_l621_621989


namespace a₁_b₁_sum_l621_621664

-- Definitions from conditions
def sequence (n : ℕ) : ℂ × ℂ :=
  if n = 0 then
    (a₁ + b₁ * I, 0)
  else
    λ ⟨a, b⟩, (2 * a - complex.I * (√3 * b), 2 * b + complex.I * (√3 * a))

-- Assumptions from conditions
def a₅₀ : ℂ := 3 * √3 - 3 * I

-- Theorem to simplify and verify the final result.
theorem a₁_b₁_sum (a₁ b₁ : ℝ) (h₅₀ : sequence 50 = a₅₀) : 
  a₁ + b₁ = (3 * (√3 - 1)) / 2^49 :=
sorry

end a₁_b₁_sum_l621_621664


namespace main_theorem_l621_621069

-- Definitions
variables (n : ℕ) (q : ℕ)

-- Conditions
def n_positive : Prop := n > 0
def q_odd : Prop := q >= 3 ∧ q % 2 = 1
def every_prime_factor_of_q_larger_than_n : Prop := ∀ p : ℕ, p.prime → p ∣ q → p > n

-- Goal to prove
def expression_is_integer : Prop :=
  let X := (1 / (Nat.factorial n * (q - 1)^n)) * ∏ i in (Finset.range n).map (λ i, i + 1), (q^i - 1)
  in X.is_integer

def expression_has_no_common_prime_factors : Prop :=
  let gcd_val := Int.gcd (q - 1) 2
  in ∏ i in (Finset.range n).map (λ i, i + 1), (q^i - 1) ∣ (q - 1) ^ n

theorem main_theorem :
  n_positive n → q_odd q → every_prime_factor_of_q_larger_than_n n q →
  expression_is_integer n q ∧ (¬(∃ p : ℕ, Int.prime p ∧ p ∣ (q - 1) / 2 ∧ p ∣ (1 / (Nat.factorial n * (q - 1)^n) * ∏ i in (Finset.range n).map (λ i, i + 1), (q^i - 1)))) :=
by
  intro n_pos q_odd cond_prime_factors
  split
  -- Here should be the proof of expression_is_integer:
  { sorry },
  -- Here should be the proof of expression_has_no_common_prime_factors:
  { sorry }

end main_theorem_l621_621069


namespace P_2018_has_2018_distinct_real_roots_l621_621630

-- Define the sequence of polynomials P_n(x) with given initial conditions.
noncomputable def P : ℕ → Polynomial ℝ
| 0       := 1
| 1       := Polynomial.X
| (n + 1) := Polynomial.X * P n - P (n - 1)

-- The statement to prove that P_2018(x) has exactly 2018 distinct real roots.
theorem P_2018_has_2018_distinct_real_roots :
  ∃ (roots : Fin 2018 → ℝ), ∀ i j : Fin 2018, i ≠ j → roots i ≠ roots j ∧ ∀ x : ℝ, (Polynomial.aeval x (P 2018)) = 0 ↔ x ∈ Set.range roots := sorry

end P_2018_has_2018_distinct_real_roots_l621_621630


namespace find_constants_l621_621301

theorem find_constants (P Q R : ℚ) 
  (h : ∀ x : ℚ, x ≠ 4 → x ≠ 2 → (5 * x + 1) / ((x - 4) * (x - 2) ^ 2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) :
  P = 21 / 4 ∧ Q = 15 ∧ R = -11 / 2 :=
by
  sorry

end find_constants_l621_621301


namespace no_polygon_with_half_parallel_diagonals_l621_621747

open Set

noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

def is_parallel_diagonal (n i j : ℕ) : Bool := 
  -- Here, you should define the mathematical condition of a diagonal being parallel to a side
  ((j - i) % n = 0) -- This is a placeholder; the actual condition would depend on the precise geometric definition.

theorem no_polygon_with_half_parallel_diagonals (n : ℕ) (h1 : n ≥ 3) :
  ¬(∃ (k : ℕ), k = num_diagonals n ∧ (∀ (i j : ℕ), i < j ∧ is_parallel_diagonal n i j = true → k = num_diagonals n / 2)) :=
by
  sorry

end no_polygon_with_half_parallel_diagonals_l621_621747


namespace solve_complex_equation_l621_621782

noncomputable def solveComplexEquation : Set ℂ :=
  { z | z^2 = -91 - 54 * Complex.i }

noncomputable def solutionSet : Set ℂ :=
  { Complex.sqrt ((-91 + Complex.sqrt 11197) / 2) - 27 / Complex.sqrt ((-91 + Complex.sqrt 11197) / 2) * Complex.i,
    -Complex.sqrt ((-91 + Complex.sqrt 11197) / 2) + 27 / Complex.sqrt ((-91 + Complex.sqrt 11197) / 2) * Complex.i }

theorem solve_complex_equation : solveComplexEquation = solutionSet := by
  sorry

end solve_complex_equation_l621_621782


namespace probability_of_TTH_sequence_l621_621446

-- Define the probability of each event in a fair coin flip
noncomputable def fair_coin_flip : ProbabilityMassFunction (Bool) :=
  { support := { tt, ff },
    mass := λ b, if b then 1 / 2 else 1 / 2,
    mass_nonneg := by intros; split_ifs; norm_num,
    mass_sum := by norm_num }

-- Define the sequence we are interested in
def sequence_TTH (flips : list Bool) : Prop :=
  flips = [ff, ff, tt]

-- Define the probability of observing a specific sequence in three independent flips of a fair coin
def probability_TTH : ℚ :=
  (fair_coin_flip.mass false) * (fair_coin_flip.mass false) * (fair_coin_flip.mass true)

-- State the theorem to be proved
theorem probability_of_TTH_sequence : probability_TTH = 1 / 8 :=
by
  unfold probability_TTH fair_coin_flip.mass
  simp
  norm_num

end probability_of_TTH_sequence_l621_621446


namespace average_waiting_time_l621_621696

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l621_621696


namespace trapezoid_diagonals_intersection_dist_l621_621144

theorem trapezoid_diagonals_intersection_dist
  {O P A B C D : Type*}
  [IsoscelesTrapezoid A B C D]
  (AO BO CO DO : ℝ) (angle_AOB : ℝ)
  (h1 : AO = 7) (h2 : CO = 3) (h3 : BO = 7) (h4 : DO = 3) (h5 : angle_AOB = 60) :
  distance O P = 21 / 4 :=
sorry

end trapezoid_diagonals_intersection_dist_l621_621144


namespace jellybeans_in_jar_now_l621_621581

def initial_jellybeans : ℕ := 90
def samantha_takes : ℕ := 24
def shelby_takes : ℕ := 12
def scarlett_takes : ℕ := 2 * shelby_takes
def scarlett_returns : ℕ := scarlett_takes / 2
def shannon_refills : ℕ := (samantha_takes + shelby_takes) / 2

theorem jellybeans_in_jar_now : 
  initial_jellybeans 
  - samantha_takes 
  - shelby_takes 
  + scarlett_returns
  + shannon_refills 
  = 84 := by
  sorry

end jellybeans_in_jar_now_l621_621581


namespace calculate_expression_l621_621425

theorem calculate_expression (a b c d e : ℕ) 
  (h : 2^a * 3^b * 5^c * 7^d * 11^e = 27720) : 
  2 * a + 3 * b + 5 * c + 7 * d + 11 * e = 35 :=
begin
  sorry
end

end calculate_expression_l621_621425


namespace last_triangle_perimeter_is_1506_div_128_l621_621056

def T1 : Triangle := ⟨2002, 2004, 2006⟩

def lastTrianglePerimeter : ℕ → Option ℚ
| 1 => some (2002 + 2004 + 2006)
| n => match lastTrianglePerimeter (n - 1) with
  | none => none
  | some perimeterPrev =>
    let x := 502 * (1 / 2 ^ (n - 4))
    let y := 501 * (1 / 2 ^ (n - 4))
    let z := 500 * (1 / 2 ^ (n - 4))
    if x + y > z ∧ y + z > x ∧ z + x > y then
      some (x + y + z)
    else
      none

theorem last_triangle_perimeter_is_1506_div_128 :
  lastTrianglePerimeter 10 = some (1506 / 128) :=
sorry

end last_triangle_perimeter_is_1506_div_128_l621_621056


namespace paul_salary_loss_l621_621473

variable (S : ℝ)

def decreased_salary (S : ℝ) : ℝ := S - (0.5 * S)

def increased_salary (S : ℝ) : ℝ := decreased_salary S + (0.5 * decreased_salary S)

theorem paul_salary_loss : 
  let final_salary := increased_salary S in
  final_salary = (3 / 4) * S →
  100 * ((S - final_salary) / S) = 25 :=
by
  intros
  sorry

end paul_salary_loss_l621_621473


namespace area_quadrilateral_WXYZ_l621_621892

noncomputable def QuadrilateralWXYZ := {W X Y Z : Type} 
-- Define the radii of the circles
def radiusA : ℝ := 1
def radiusB : ℝ := 2
def radiusC : ℝ := 3
def radiusD : ℝ := 4

-- Define the tangency points which are implicitly given as part of the quadrilateral sides
def tangencyPoints : ℝ := {
  -- Circle A touches WX at Point W
  -- Circle B touches XZ at Point X
  -- Circle C touches ZY at Point Z
  -- Circle D touches YW at Point Y
  tangentAW := radiusA,
  tangentBX := radiusB,
  tangentCY := radiusC,
  tangentDZ := radiusD
}

-- Define the lengths of the sides of quadrilateral WXYZ based on the tangency condition
def sideWX : ℝ := radiusA + radiusB
def sideXZ : ℝ := radiusB + radiusC
def sideZY : ℝ := radiusC + radiusD
def sideYW : ℝ := radiusA + radiusD

-- Assume quadrilateral is an approximate rectangle for simplicity
def approximateRectangleArea : ℝ := sideYW * sideXZ  -- In the problem solution length is taken as 5.

theorem area_quadrilateral_WXYZ : approximateRectangleArea = 25 := 
by
  sorry

end area_quadrilateral_WXYZ_l621_621892


namespace max_possible_value_of_C_l621_621962

theorem max_possible_value_of_C (A B C D : ℕ) (h₁ : A + B + C + D = 200) (h₂ : A + B = 70) (h₃ : 0 < A) (h₄ : 0 < B) (h₅ : 0 < C) (h₆ : 0 < D) :
  C ≤ 129 :=
by
  sorry

end max_possible_value_of_C_l621_621962


namespace prism_volume_l621_621172

theorem prism_volume 
    (x y z : ℝ) 
    (h_xy : x * y = 18) 
    (h_yz : y * z = 12) 
    (h_xz : x * z = 8) 
    (h_longest_shortest : max x (max y z) = 2 * min x (min y z)) : 
    x * y * z = 16 := 
  sorry

end prism_volume_l621_621172


namespace hamburger_cost_l621_621447

variable (H : ℝ)

theorem hamburger_cost :
  (H + 2 + 3 = 20 - 11) → (H = 4) :=
by
  sorry

end hamburger_cost_l621_621447


namespace product_geometric_progression_l621_621943

theorem product_geometric_progression (a₁ q : ℝ) (n : ℕ) :
  (∏ i in Finset.range n, a₁ * q^i) = a₁^n * q^((n * (n - 1)) / 2) := 
by
  sorry

end product_geometric_progression_l621_621943


namespace lawrence_walked_total_distance_l621_621869

noncomputable def distance_per_day : ℝ := 4.0
noncomputable def number_of_days : ℝ := 3.0
noncomputable def total_distance_walked (distance_per_day : ℝ) (number_of_days : ℝ) : ℝ :=
  distance_per_day * number_of_days

theorem lawrence_walked_total_distance :
  total_distance_walked distance_per_day number_of_days = 12.0 :=
by
  -- The detailed proof is omitted as per the instructions.
  sorry

end lawrence_walked_total_distance_l621_621869


namespace race_participants_minimum_l621_621513

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l621_621513


namespace largest_multiple_of_8_negation_l621_621169

-- Definition of the problem conditions
def is_largest_multiple_of_8 (n : ℕ) : Prop :=
  (∃ m : ℤ, n = -m ∧ (m % 8 = 0) ∧ (-m > -200)) ∧
  ∀ k : ℤ, (k % 8 = 0) → (k > -200) → (k ≥ m)

-- Statement to prove
theorem largest_multiple_of_8_negation : ∃ m : ℕ, is_largest_multiple_of_8 m :=
  ∃ m, m = 192 ∧ is_largest_multiple_of_8 m

end largest_multiple_of_8_negation_l621_621169


namespace origami_papers_per_cousin_l621_621174

theorem origami_papers_per_cousin :
  let total_papers := 128.5
  let number_of_cousins := 8.3
  let papers_per_cousin := total_papers / number_of_cousins
  (Int.round papers_per_cousin) = 15 :=
by
  sorry

end origami_papers_per_cousin_l621_621174


namespace best_method_for_vasiliy_l621_621234

def grades : List ℕ := [4, 1, 2, 5, 2]

def methodA (g : List ℕ) : ℕ :=
  (g.sum / g.length.toReal).round.toNat

def methodB (g : List ℕ) : ℕ :=
  let sorted_g := g.qsort (· ≤ ·)
  sorted_g.nth (sorted_g.length / 2) |>.getD 0

theorem best_method_for_vasiliy : methodA grades > methodB grades :=
by
  sorry

end best_method_for_vasiliy_l621_621234


namespace minimum_participants_l621_621531

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l621_621531


namespace coin_toss_odd_heads_prob_l621_621608

theorem coin_toss_odd_heads_prob (n : ℕ) :
  let p_m (m : ℕ) := 1 / (2 * m + 1 : ℝ) in
  let prob_odd_heads := (n : ℝ) / (2 * n + 1 : ℝ) in
  (∑ m in finset.range n, prob_odd_heads * (p_m m) * (1 - p_m m)) = prob_odd_heads := sorry

end coin_toss_odd_heads_prob_l621_621608


namespace min_number_of_participants_l621_621505

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l621_621505


namespace sean_needs_six_packs_l621_621109

/-- 
 Sean needs to replace 2 light bulbs in the bedroom, 
 1 in the bathroom, 1 in the kitchen, and 4 in the basement. 
 He also needs to replace 1/2 of that amount in the garage. 
 The bulbs come 2 per pack. 
 -/
def bedroom_bulbs: ℕ := 2
def bathroom_bulbs: ℕ := 1
def kitchen_bulbs: ℕ := 1
def basement_bulbs: ℕ := 4
def bulbs_per_pack: ℕ := 2

noncomputable def total_bulbs_needed_including_garage: ℕ := 
  let total_rooms_bulbs := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs
  let garage_bulbs := total_rooms_bulbs / 2
  total_rooms_bulbs + garage_bulbs

noncomputable def total_packs_needed: ℕ := total_bulbs_needed_including_garage / bulbs_per_pack

theorem sean_needs_six_packs : total_packs_needed = 6 :=
by
  sorry

end sean_needs_six_packs_l621_621109


namespace expected_value_of_event_A_occurrences_l621_621002

theorem expected_value_of_event_A_occurrences 
  (n : ℕ) (p : ℝ) (h_n_eq_4 : n = 4)
  (h_prob_at_least_once : 1 - (1 - p)^n = 65 / 81) : E (ξ : binomial n p) = 4 / 3 := 
by
  sorry

end expected_value_of_event_A_occurrences_l621_621002


namespace find_ratio_WY_WP_l621_621005

-- Define the problem conditions and the point of intersection
variables (W X Y Z M N P : Type) [AddGroup M] [AddGroup N]
variables (hx : Parallelogram W X Y Z)
variables (hM : OnSegment W Z M (3 / 100))
variables (hN : OnSegment W Y N (3 / 251))
variables (hP : Intersect W Y M N P)

-- Define the theorem to prove the question with conditions
theorem find_ratio_WY_WP : 
  ∀ (W X Y Z M N P : Type) [AddGroup M] [AddGroup N],
    Parallelogram W X Y Z →
    OnSegment W Z M (3 / 100) →
    OnSegment W Y N (3 / 251) →
    Intersect W Y M N P →
  (ratio_WY_WP W Y P = 2) :=
by sorry

end find_ratio_WY_WP_l621_621005


namespace michael_sold_large_paintings_l621_621085

theorem michael_sold_large_paintings 
  (charge_large : ℕ)
  (charge_small : ℕ)
  (num_small_paintings : ℕ)
  (total_earned : ℕ)
  (H1 : charge_large = 100)
  (H2 : charge_small = 80)
  (H3 : num_small_paintings = 8)
  (H4 : total_earned = 1140) : 
  ∃ L : ℕ, charge_large * L + charge_small * num_small_paintings = total_earned ∧ L = 5 :=
by
  use 5
  rw [H1, H2, H3, H4]
  norm_num
  sorry

end michael_sold_large_paintings_l621_621085


namespace greatest_integer_100y_l621_621431

def y : ℝ := (∑ n in Finset.range 45, real.csc (n+1:ℝ) * real.sec (n+1:ℝ)) / 
              (∑ n in Finset.range 45, real.cot (n+1:ℝ))

theorem greatest_integer_100y :
  floor (100 * y) = 222 :=
sorry

end greatest_integer_100y_l621_621431


namespace d_lt_zero_iff_b_decreasing_l621_621924

variable {a : ℕ → ℝ} 
variable {d : ℝ}

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop := 
∀ n : ℕ, a (n + 1) = a n + d

def b (a : ℕ → ℝ) : ℕ → ℝ :=
λ n, 2^(a n)

theorem d_lt_zero_iff_b_decreasing 
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_seq a d) :
  d < 0 ↔ ∀ n : ℕ, b a (n + 1) < b a n :=
begin
  sorry
end

end d_lt_zero_iff_b_decreasing_l621_621924


namespace exists_infinite_sequence_perfect_square_l621_621482

theorem exists_infinite_sequence_perfect_square :
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ n, a n < a (n+1)) ∧ (∀ n, ∃ k, (∑ i in Finset.range (n + 1), (a i) ^ 2) = k^2) :=
begin
  sorry
end

end exists_infinite_sequence_perfect_square_l621_621482


namespace cost_price_equation_l621_621568

-- Define the parameters: marked price, discount rate, profit, and cost price
def marked_price : ℝ := 1375
def discount_rate : ℝ := 0.20
def profit : ℝ := 100
def cost_price : ℝ

-- The proof problem
theorem cost_price_equation (x : ℝ) (h : x = marked_price * (1 - discount_rate) - profit) : 
  marked_price * (1 - discount_rate) = x + profit := 
by 
  rw h
  sorry

end cost_price_equation_l621_621568


namespace ratio_of_years_taught_l621_621249

-- Definitions based on given conditions
def C : ℕ := 4
def A : ℕ := 2 * C
def total_years (S : ℕ) : Prop := C + A + S = 52

-- Proof statement
theorem ratio_of_years_taught (S : ℕ) (h : total_years S) : 
  S / A = 5 / 1 :=
by
  sorry

end ratio_of_years_taught_l621_621249


namespace minimum_binomial_sum_l621_621971

theorem minimum_binomial_sum :
  ∀ (a : Fin 10 → ℤ), (∀ i, a i > 1) → (∑ i in Finset.univ, a i) = 2006 →
  (∑ i in Finset.univ, (a i * (a i - 1) / 2)) = 200200 :=
by
  intros a ha hsum
  sorry

end minimum_binomial_sum_l621_621971


namespace difference_between_max_and_min_is_76_l621_621991

def nums : List ℕ := [27, 103, 78, 35, 97]

theorem difference_between_max_and_min_is_76 :
  (List.maximum nums).getOrElse 0 - (List.minimum nums).getOrElse 0 = 76 :=
by {
  have h_max : List.maximum nums = some 103 := by sorry,
  have h_min : List.minimum nums = some 27 := by sorry,
  rw [h_max, h_min],
  norm_num,
}

end difference_between_max_and_min_is_76_l621_621991


namespace area_of_quadrilateral_l621_621484

-- Let A, B, C, D be points in a 2-dimensional Euclidean space
variables {A B C D : Point}

-- Given conditions
-- 1. Quadrilateral ABCD has right angles at B and D
-- 2. AC = 5
-- 3. The sides of ABCD are integers with at least two having distinct lengths

-- Definitions of points and sides
def AB := dist A B
def BC := dist B C
def AD := dist A D
def DC := dist D C
def AC := dist A C

-- Define the areas of right-angled triangles
def area_ABC := (1 / 2) * AB * BC
def area_ADC := (1 / 2) * AD * DC

-- The statement to prove
theorem area_of_quadrilateral (h_right_angles: ∠ B = 90 ∧ ∠ D = 90)
  (h_hypotenuse: AC = 5)
  (h_distinct_lengths: AB ≠ AD ∨ AB ≠ DC ∨ AD ≠ DC):
  area_ABC + area_ADC = 12 := 
by
  sorry

end area_of_quadrilateral_l621_621484


namespace avg_waiting_time_is_1_point_2_minutes_l621_621700

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l621_621700


namespace quaternary_201_is_33_in_decimal_l621_621167

theorem quaternary_201_is_33_in_decimal:
  (2 * 4^2 + 0 * 4^1 + 1 * 4^0 = 33) :=
by simp [pow]; sorry

end quaternary_201_is_33_in_decimal_l621_621167


namespace parallelogram_AM_BM_eq_AC_BD_sq_l621_621813

noncomputable def parallelogram_ratio (A B C D M : Point)
  (h_parallel1 : Line.parallel (Line.mk A B) (Line.mk C D))
  (h_parallel2 : Line.parallel (Line.mk A D) (Line.mk B C))
  (h_angle1 : ∠ DAC = ∠ MAC )
  (h_angle2 : ∠ CAB = ∠ MAB) : ℝ :=
  ∃ k : ℝ, k = ( Segment.length (Segment.mk A M) / Segment.length (Segment.mk B M)) = 
           (Segment.length (Segment.mk A C) / Segment.length (Segment.mk B D)) ^ 2

theorem parallelogram_AM_BM_eq_AC_BD_sq (A B C D M : Point)
  (h_parallel1 : Line.parallel (Line.mk A B) (Line.mk C D))
  (h_parallel2 : Line.parallel (Line.mk A D) (Line.mk B C))
  (h_angle1 : ∠ DAC = ∠ MAC )
  (h_angle2 : ∠ CAB = ∠ MAB) :
  (Segment.length (Segment.mk A M) / Segment.length (Segment.mk B M)) = 
    (Segment.length (Segment.mk A C) / Segment.length (Segment.mk B D)) ^ 2 :=
by 
  sorry

end parallelogram_AM_BM_eq_AC_BD_sq_l621_621813


namespace determine_extremum_l621_621352

noncomputable def f (x : ℝ) : ℝ := -x^2 + x + 1

theorem determine_extremum : 
  ∃ x_max x_min ∈ set.Icc (0:ℝ) (3/2 : ℝ), (x_max = 1/2 ∧ f x_max = 5/4) ∧ (x_min = 3/2 ∧ f x_min = 1/4) :=
by
  use 1/2, 3/2
  split
  · -- Proving x_max = 1/2 within the interval and its value
    split
    · norm_num
      split
      · norm_num
      · simp [f]
        norm_num
  · -- Proving x_min = 3/2 within the interval and its value
    split
    · norm_num
      linarith
    · simp [f]
      norm_num
sorry

end determine_extremum_l621_621352


namespace num_ordered_4_tuples_l621_621434

noncomputable def N : ℕ := 30 ^ 2015

theorem num_ordered_4_tuples :
  { t : ℕ × ℕ × ℕ × ℕ | (∀ n : ℤ, (t.fst.1 * n^3 + t.fst.2 * n^2 + 2 * t.snd.1 * n + t.snd.2) % N = 0) ∧ 
    (1 ≤ t.fst.1 ∧ t.fst.1 ≤ N) ∧ 
    (1 ≤ t.fst.2 ∧ t.fst.2 ≤ N) ∧ 
    (1 ≤ t.snd.1 ∧ t.snd.1 ≤ N) ∧ 
    (1 ≤ t.snd.2 ∧ t.snd.2 ≤ N) }.card = 24 := 
  sorry

end num_ordered_4_tuples_l621_621434


namespace black_area_sum_infinite_sequence_l621_621233

theorem black_area_sum_infinite_sequence :
  ∃ a b c : ℤ, (a + b + c = 0) ∧ (∀ n : ℕ, n ≥ 1 → let white_square_area := ∑ i in finset.range n, 1 / (4^i)
                                                       let black_circle_area := ∑ i in finset.range n, π / (4^(i+1))
                                                       in black_circle_area - white_square_area = (π - 4) / 3) :=
sorry

end black_area_sum_infinite_sequence_l621_621233


namespace mice_meet_after_three_days_l621_621404

theorem mice_meet_after_three_days 
  (thickness : ℕ) 
  (first_day_distance : ℕ) 
  (big_mouse_double_progress : ℕ → ℕ) 
  (small_mouse_half_remain_distance : ℕ → ℕ) 
  (days : ℕ) 
  (big_mouse_distance : ℚ) : 
  thickness = 5 ∧ 
  first_day_distance = 1 ∧ 
  (∀ n, big_mouse_double_progress n = 2 ^ (n - 1)) ∧ 
  (∀ n, small_mouse_half_remain_distance n = 5 - (5 / 2 ^ (n - 1))) ∧ 
  days = 3 → 
  big_mouse_distance = 3 + 8 / 17 := 
by
  sorry

end mice_meet_after_three_days_l621_621404


namespace jill_has_6_more_dolls_than_jane_l621_621161

theorem jill_has_6_more_dolls_than_jane
  (total_dolls : ℕ) 
  (jane_dolls : ℕ) 
  (more_dolls_than : ℕ → ℕ → Prop)
  (h1 : total_dolls = 32) 
  (h2 : jane_dolls = 13) 
  (jill_dolls : ℕ)
  (h3 : more_dolls_than jill_dolls jane_dolls) :
  (jill_dolls - jane_dolls) = 6 :=
by
  -- the proof goes here
  sorry

end jill_has_6_more_dolls_than_jane_l621_621161


namespace probability_range_l621_621815

noncomputable def ξ : Type -- Define ξ as a type to represent our random variable
noncomputable instance : ProbabilityMassFunction ξ := sorry -- ξ has a probability distribution

-- ξ follows a normal distribution with mean 1 and variance σ²
axiom norm_dist : ∃ σ² : ℝ, ∀ x : ℝ, distribution ξ = normal 1 σ²

-- Given condition P(ξ ≤ 2) = 0.8
axiom given_condition : P(ξ ≤ 2) = 0.8

-- We need to prove P(0 ≤ ξ ≤ 2) = 0.6
theorem probability_range : P(0 ≤ ξ ≤ 2) = 0.6 :=
by sorry

end probability_range_l621_621815


namespace find_missing_number_l621_621295

theorem find_missing_number (x : ℝ) : 11 + Real.sqrt(x + 6 * 4 / 3) = 13 → x = -4 :=
by
  intro h
  sorry

end find_missing_number_l621_621295


namespace collinear_points_of_tangents_l621_621854

/--
Given three pairwise non-intersecting circles, denote by \(A_1, A_2, A_3\) the three points of intersection of the common internal tangents to any two of them, and by \(B_1, B_2, B_3\) the corresponding points of intersection of the external tangents.
Prove that these points are arranged on four lines with three points on each line: \( A_1, A_2, B_3; A_1, B_2, A_3; B_1, A_2, A_3; B_1, B_2, B_3 \).
-/
theorem collinear_points_of_tangents (O1 O2 O3 : Circle) 
  (A1 A2 A3 : Point) (B1 B2 B3 : Point) :
  -- Conditions
  (∀ A_i B_i : Point, is_intersecting_tangent A_i B_i O1 O2 ∧ not_intersects O1 O2 ∧ is_intersecting_tangent A_i B_i O2 O3 ∧ not_intersects O2 O3 ∧ is_intersecting_tangent A_i B_i O3 O1 ∧ not_intersects O3 O1) →
  -- Question
  are_collinear A1 A2 B3 ∧ are_collinear A1 B2 A3 ∧ are_collinear B1 A2 A3 ∧ are_collinear B1 B2 B3 :=
sorry

end collinear_points_of_tangents_l621_621854


namespace general_term_l621_621798

open Classical

-- Define the sequence using the given conditions and prove the general term.
noncomputable def a : ℕ → ℝ
| 0     := 0
| (n+1) := if n = 0 then 1 else 1 / (n+1)^2

def satisfies_condition (a : ℕ → ℝ) : ℕ → Prop
| 0     := True
| (n+1) := n > 0 → 1 / Real.sqrt (a (n+1)) + 1 / Real.sqrt (a n) = 2*n + 1

lemma seq_satisfies_cond : satisfies_condition a :=
begin
  intro n,
  cases n,
  { triv },
  { intro hn,
    rw [a, a, if_neg (nat.succ_ne_zero _), if_neg (nat.succ_pos _).ne'] at *,
    have h1 : a (n+1) = Real.sqrt (a (n+1))^2 := Real.sqrt_sq (a (n+1)) (real.exp_pos (real.log_sq_nonneg _)),
    have h2 : a n = Real.sqrt (a n)^2 := Real.sqrt_sq (a n) (real.exp_pos (real.log_sq_nonneg _)),
    rw [h1, h2, Real.exp_nat_mul_eq_pow] }
end

theorem general_term : ∀ n ∈ ℕ, n ≠ 0 → a n = 1 / n^2 :=
begin
  intros n hn,
  induction n with k hk,
  { contradiction },
  { cases k,
    { simp [a] },
    { rw [a, if_neg],
      exacts [hk, nat.succ_ne_zero _] } }
end

end general_term_l621_621798


namespace min_a1_l621_621062

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 13 * a n - 2 * (n + 1)

theorem min_a1
  (a : ℕ → ℝ)
  (h₀ : ∀ n, a n > 0)
  (h₁ : sequence a) :
  a 1 = 25 / 72 :=
by
  sorry

end min_a1_l621_621062


namespace cot_alpha_third_quadrant_l621_621344

theorem cot_alpha_third_quadrant {α : ℝ} (hα : α ∈ Set.Ioo π (3 * π / 2)) (h_sin : Real.sin α = -1/3) :
  Real.cot α = 2 * Real.sqrt 2 :=
by
  sorry

end cot_alpha_third_quadrant_l621_621344


namespace land_tax_calculation_l621_621236

theorem land_tax_calculation
  (area : ℝ)
  (value_per_acre : ℝ)
  (tax_rate : ℝ)
  (total_cadastral_value : ℝ := area * value_per_acre)
  (land_tax : ℝ := total_cadastral_value * tax_rate) :
  area = 15 → value_per_acre = 100000 → tax_rate = 0.003 → land_tax = 4500 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end land_tax_calculation_l621_621236


namespace monotonic_decreasing_interval_l621_621246

-- Define the given function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the question as a theorem to be proven
theorem monotonic_decreasing_interval : { x : ℝ | x <= 1 } = set_of (λ x, x <= 1) :=
by
  sorry

end monotonic_decreasing_interval_l621_621246


namespace correct_division_result_l621_621176

theorem correct_division_result (x : ℝ) (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by
  sorry

end correct_division_result_l621_621176


namespace domain_of_f_l621_621133

def domain_of_log_func := Set ℝ

def is_valid (x : ℝ) : Prop := x - 1 > 0

def func_domain (f : ℝ → ℝ) : domain_of_log_func := {x : ℝ | is_valid x}

theorem domain_of_f :
  func_domain (λ x => Real.log (x - 1)) = {x : ℝ | 1 < x} := by
  sorry

end domain_of_f_l621_621133


namespace number_of_odd_functions_is_two_l621_621822

-- Define the functions
def f1 (x : ℝ) := 3
def f2 (x : ℝ) := 3 * x
def f3 (x : ℝ) := 3 * x^2
def f4 (x : ℝ) := 3 * x^3

-- Define what it means for a function to be odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Main theorem to prove
theorem number_of_odd_functions_is_two :
  (is_odd_function f1 = false) ∧
  (is_odd_function f2 = true) ∧
  (is_odd_function f3 = false) ∧
  (is_odd_function f4 = true) → (2 = 2) :=
by
  intro h
  sorry

end number_of_odd_functions_is_two_l621_621822


namespace sale_price_per_bearing_before_bulk_discount_l621_621035

-- Define the given conditions
def machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := machines * ball_bearings_per_machine

def normal_cost_per_bearing : ℝ := 1
def total_normal_cost : ℝ := total_ball_bearings * normal_cost_per_bearing

def bulk_discount : ℝ := 0.20
def sale_savings : ℝ := 120

-- The theorem we need to prove
theorem sale_price_per_bearing_before_bulk_discount (P : ℝ) :
  total_normal_cost - (total_ball_bearings * P * (1 - bulk_discount)) = sale_savings → 
  P = 0.75 :=
by sorry

end sale_price_per_bearing_before_bulk_discount_l621_621035


namespace isosceles_triangle_base_length_l621_621186

noncomputable def equilateral_side_length (p_eq : ℕ) : ℕ := p_eq / 3

theorem isosceles_triangle_base_length (p_eq p_iso s b : ℕ) 
  (h1 : p_eq = 45)
  (h2 : p_iso = 40)
  (h3 : s = equilateral_side_length p_eq)
  (h4 : p_iso = s + s + b)
  : b = 10 :=
by
  simp [h1, h2, h3] at h4
  -- steps to solve for b would be written here
  sorry

end isosceles_triangle_base_length_l621_621186


namespace find_fibonacci_x_l621_621600

def is_fibonacci (a b c : ℕ) : Prop :=
  c = a + b

theorem find_fibonacci_x (a b x : ℕ)
  (h₁ : a = 8)
  (h₂ : b = 13)
  (h₃ : is_fibonacci a b x) :
  x = 21 :=
by
  sorry

end find_fibonacci_x_l621_621600


namespace evaluate_ceil_of_neg_sqrt_l621_621262

-- Define the given expression and its value computation
def given_expression : ℚ := -real.sqrt (64 / 9)

-- Define the expected answer
def expected_answer : ℤ := -2

-- State the theorem to be proven
theorem evaluate_ceil_of_neg_sqrt : (Int.ceil given_expression) = expected_answer := sorry

end evaluate_ceil_of_neg_sqrt_l621_621262


namespace cans_needed_for_rooms_l621_621083

theorem cans_needed_for_rooms 
  (initial_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) 
  (initial_rooms = 45) (lost_cans = 5) (remaining_rooms = 35) : 
  ℕ :=
have cans_per_room : ℕ :=
  have rooms_loss_rate : ℕ := initial_rooms - remaining_rooms,
  rooms_loss_rate / lost_cans,
  have _ : cans_per_room = 2, by sorry,
remaining_rooms / cans_per_room = 18 :=
by sorry

end cans_needed_for_rooms_l621_621083


namespace lex_book_pages_l621_621079

theorem lex_book_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 20) (h2 : days = 12) : total_pages = 240 :=
by
  rw [h1, h2]
  exact (nat.mul_eq_mul_left pages_per_day days 20 12).mpr rfl

end lex_book_pages_l621_621079


namespace P_2018_has_2018_distinct_real_roots_l621_621625
noncomputable theory

def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 2) := λ x, x * P (n + 1) x - P n x

theorem P_2018_has_2018_distinct_real_roots :
  (∃ xs : fin 2018 → ℝ, ∀ i j : fin 2018, i ≠ j → xs i ≠ xs j ∧ ∀ x : ℝ, P 2018 x = 0 ↔ ∃ k : fin 2018, x = xs k) :=
sorry

end P_2018_has_2018_distinct_real_roots_l621_621625


namespace ceil_neg_sqrt_fraction_l621_621280

theorem ceil_neg_sqrt_fraction :
  (⌈-real.sqrt (64 / 9)⌉ = -2) :=
by
  -- Define the necessary conditions
  have h1 : real.sqrt (64 / 9) = 8 / 3 := by sorry,
  have h2 : -real.sqrt (64 / 9) = -8 / 3 := by sorry,
  -- Apply the ceiling function and prove the result
  exact sorry

end ceil_neg_sqrt_fraction_l621_621280


namespace maria_average_speed_l621_621250

noncomputable def total_distance := 56 -- miles
noncomputable def total_time := 8      -- hours
noncomputable def rest_time := 2 * 0.5 -- hours

noncomputable def time_cycling : ℝ := total_time - rest_time

theorem maria_average_speed : (total_distance / time_cycling) = 8 :=
  sorry

end maria_average_speed_l621_621250


namespace solve_system_l621_621300

noncomputable def solutions : set (ℝ × ℝ) := 
    {(-1, 2), (2, -1), (1 + Real.sqrt 2, 1 - Real.sqrt 2), 
    (1 - Real.sqrt 2, 1 + Real.sqrt 2), 
    ((-9 + Real.sqrt 57) / 6, (-9 - Real.sqrt 57) / 6), 
    ((-9 - Real.sqrt 57) / 6, (-9 + Real.sqrt 57) / 6)}

theorem solve_system (x y : ℝ) : 
    x^2 - x * y + y^2 = 7 ∧ 
    x^2 * y + x * y^2 = -2 ↔ 
    (x, y) ∈ solutions :=
by 
  sorry

end solve_system_l621_621300


namespace spherical_to_rectangular_correct_l621_621737

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) := by
  sorry

end spherical_to_rectangular_correct_l621_621737


namespace F_at_neg1_eq_zero_l621_621232

noncomputable def F (x : ℝ) : ℝ :=
  real.sqrt (abs (x + 1)) + (9 / real.pi) * real.atan (real.sqrt (abs (x + 1)))

theorem F_at_neg1_eq_zero : F (-1) = 0 :=
by
  sorry

end F_at_neg1_eq_zero_l621_621232


namespace race_participants_minimum_l621_621511

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l621_621511


namespace race_participants_minimum_l621_621514

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l621_621514


namespace range_of_f_l621_621436

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4

theorem range_of_f : Set.Icc 0 (9 / 8) = Set.range f := 
by
  sorry

end range_of_f_l621_621436


namespace KnightsCount_l621_621454

def isKnight (n : ℕ) : Prop := sorry -- Define isKnight
def tellsTruth (n : ℕ) : Prop := sorry -- Define tellsTruth

-- Statements by the inhabitants
axiom H1 : isKnight 0 ↔ tellsTruth 0
axiom H2 : tellsTruth 1 ↔ isKnight 0
axiom H3 : tellsTruth 2 ↔ (¬(isKnight 0) ∨ ¬(isKnight 1))
axiom H4 : tellsTruth 3 ↔ (¬(isKnight 0) ∨ ¬(isKnight 1) ∨ ¬(isKnight 2))
axiom H5 : tellsTruth 4 ↔ (isKnight 0 ∧ isKnight 1 ∧ (isKnight 2 ∨ isKnight 3))
axiom H6 : tellsTruth 5 ↔ ((¬isKnight 0 ∨ ¬isKnight 1 ∨ isKnight 2) ∧ (¬isKnight 3 ∨ isKnight 4))
axiom H7 : tellsTruth 6 ↔ (isKnight 0 ∧ isKnight 1 ∧ isKnight 2 ∧ isKnight 3 ∧ ¬(¬isKnight 4 ∧ ¬isKnight 5))

theorem KnightsCount : ∃ k1 k2 k3 k4 k5 k6 k7 : Prop,
  (isKnight 0 = k1 ∧ isKnight 1 = k2 ∧ isKnight 2 = k3 ∧ isKnight 3 = k4 ∧ isKnight 4 = k5 ∧ isKnight 5 = k6 ∧ isKnight 6 = k7) ∧ 
  tellsTruth 0 ∧ tellsTruth 1 ∧ tellsTruth 2 ∧ tellsTruth 3 ∧ tellsTruth 4 ∧ tellsTruth 5 ∧ tellsTruth 6 ∧
  (5 = 1 + if k1 then 1 else 0 + if k2 then 1 else 0 + if k3 then 1 else 0 + if k4 then 1 else 0 + if k5 then 1 else 0 + if k6 then 1 else 0 + if k7 then 1 else 0)
:=
by
  sorry

end KnightsCount_l621_621454


namespace general_term_sum_formula_l621_621818

-- Define the sequence \{a_n\} with the given conditions
def sequence (n : ℕ) : ℕ
  | 0       := 0
  | (n + 1) := if n = 0 then 1 else 2 * sequence n + 1

-- Define the general term formula
theorem general_term (n : ℕ) (h : n > 0) : sequence n = 2^n - 1 := sorry

-- Define the sum of the first n terms of the sequence
def sum_sequence (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, sequence (k + 1))

-- Prove the sum formula
theorem sum_formula (n : ℕ) : sum_sequence n = 2^(n + 1) - 2 - n := sorry

end general_term_sum_formula_l621_621818


namespace min_number_of_participants_l621_621506

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l621_621506


namespace triangle_ACH_area_and_cube_edge_length_l621_621990

theorem triangle_ACH_area_and_cube_edge_length :
  ∃ (area : ℝ) (edge_length : ℝ), 
    let height : ℝ := 12
    let area := (48 * Real.sqrt 3 : ℝ)
    let edge_length := (4 * Real.sqrt 6 : ℝ)
    area ≈ 83.1 ∧ edge_length ≈ 9.8 := 
by
  let height := 12
  let area := (48 * Real.sqrt 3 : ℝ)
  let edge_length := (4 * Real.sqrt 6 : ℝ)
  use [(48 * Real.sqrt 3 : ℝ), (4 * Real.sqrt 6 : ℝ)]
  split
  · apply Real.approximately_eq.trans _ _
    exact Real.approximately_eq.symm (Real.approx_sqrt3)
    norm_num
  · apply Real.approximately_eq.trans _ _
    exact Real.approximately_eq.symm (Real.approx_sqrt6)
    norm_num
  sorry

end triangle_ACH_area_and_cube_edge_length_l621_621990


namespace race_participants_minimum_l621_621515

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l621_621515


namespace race_participants_minimum_l621_621523

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l621_621523


namespace remainder_of_4521_l621_621788

theorem remainder_of_4521 (h1 : ∃ d : ℕ, d = 88)
  (h2 : 3815 % 88 = 31) : 4521 % 88 = 33 :=
sorry

end remainder_of_4521_l621_621788


namespace sum_of_digits_B_is_7_l621_621433

-- Definitions of A and B based on the conditions given
def sum_of_digits (n : ℕ) : ℕ :=
  (n.toDigits 10).sum

def A := sum_of_digits (4444 ^ 4444)
def B := sum_of_digits A

-- Statement to be proven
theorem sum_of_digits_B_is_7 : sum_of_digits B = 7 :=
  sorry

end sum_of_digits_B_is_7_l621_621433


namespace max_elements_l621_621046

def T : Set ℕ := { x | 1 ≤ x ∧ x ≤ 2023 }

theorem max_elements (T : Set ℕ) (h₁ : ∀ (a b : ℕ), a ∈ T → b ∈ T → (a ≠ b → a ≠ b + 5 ∧ a ≠ b + 8)) :
  ∃ (n : ℕ), n = 780 ∧ ∀ (S : Set ℕ), (S ⊆ T) → (∀ (a b : ℕ), a ∈ S → b ∈ S → (a ≠ b → a ≠ b + 5 ∧ a ≠ b + 8)) → S.card ≤ 780 :=
sorry

end max_elements_l621_621046


namespace stuart_initial_marbles_l621_621218

theorem stuart_initial_marbles (B S : ℝ) (h1 : B = 60) (h2 : 0.40 * B = 24) (h3 : S + 24 = 80) : S = 56 :=
by
  sorry

end stuart_initial_marbles_l621_621218


namespace tan_sum_identity_l621_621363

variable (α : ℝ)

def vec_a : ℝ × ℝ := (-2, Real.cos α)
def vec_b : ℝ × ℝ := (-1, Real.sin α)

-- Condition stating that the vectors are parallel
def parallel_vectors : Prop :=
  -2 * Real.sin α + Real.cos α = 0

-- Assertion to prove
theorem tan_sum_identity (h : parallel_vectors α) : Real.tan (α + Real.pi / 4) = 3 :=
by
  sorry

end tan_sum_identity_l621_621363


namespace find_a_17_l621_621135

variable (a : ℕ → ℝ)

-- Conditions
def a_5 : Prop := a 5 = 5
def a_11 : Prop := a 11 = 40

-- Define that 'a' is a geometric sequence, so there exists a common ratio 'r'.
variable (r : ℝ)
def geometric_sequence : Prop := ∀ n m, a (n + m) = a n * r^m

-- The question (which will be the goal to prove)
theorem find_a_17 (h1 : a_5) (h2 : a_11) (h_geo : geometric_sequence) : a 17 = 320 := by
  sorry

end find_a_17_l621_621135


namespace intersection_complement_l621_621932

open Set

theorem intersection_complement {U A B : Set ℤ} :
  U = {-1, -2, -3, -4, 0} → A = {-1, -2, 0} → B = {-3, -4, 0} →
  ((U \ A) ∩ B) = {-3, -4} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  simp only [Set.mem_set_of_eq, Set.subset_def]
  sorry

end intersection_complement_l621_621932


namespace tetrahedron_faces_equal_l621_621619

theorem tetrahedron_faces_equal {a b c a' b' c' : ℝ} (h₁ : a + b + c = a + b' + c') (h₂ : a + b + c = a' + b + b') (h₃ : a + b + c = c' + c + a') :
  (a = a') ∧ (b = b') ∧ (c = c') :=
by
  sorry

end tetrahedron_faces_equal_l621_621619


namespace Stuart_initial_marbles_l621_621216

variable (Betty_marbles Stuart_final increased_by: ℤ) 

-- Conditions as definitions
def Betty_has : Betty_marbles = 60 := sorry 
def Stuart_collect_increase : Stuart_final = 80 := sorry 
def percentage_given : ∃ x, x = (40 * Betty_marbles) / 100 := sorry 

-- Theorem to prove Stuart had 56 marbles initially
theorem Stuart_initial_marbles 
  (h1 : Betty_has)
  (h2 : Stuart_collect_increase)
  (h3 : percentage_given) :
  ∃ y, y = Stuart_final - 24 := 
sorry

end Stuart_initial_marbles_l621_621216


namespace knights_and_knaves_l621_621468

/-- Proof that there are exactly 5 knights among 7 inhabitants given specific conditions -/
theorem knights_and_knaves : 
  ∀ (inhabitants : Fin 7 → Prop)
    (knight truthful liar : Prop)
    (H1 : knight = truthful)
    (H2 : liar = ¬truthful)
    (H3 : (inhabitants 0 = knight))
    (H4 : (inhabitants 1 = knight))
    (H5 : (inhabitants 2 = liar ∧ (inhabitants 0 = knight ∨ inhabitants 1 = knight)))
    (H6 : (inhabitants 3 = liar ∧ (inhabitants 0 = truth ∧ inhabitants 1 = truth ∧ inhabitants 2 = liar → 2/3 ≥ 0.65)))
    (H7 : (inhabitants 4 = knight))
    (H8 : (inhabitants 5 = knight ∧ (∃ (half_knights : Fin 6 → Prop), (inhabitants 0 = knight) ∧ (inhabitants 1 = knight) ∧ (inhabitants 4 = knight) ∧ (inhabitants 5 = knight) ∧ counting_knaves half_knights < 3))
    (H9 : (inhabitants 6 = knight ∧ (counting_knights inhabitants 6 ≥ 0.65))),
  counting_knights inhabitants = 5 := 
by sorry

def counting_knights (inhabitants : Fin 7 → Prop) : ℝ := sorry
def counting_knaves (inhabitants : Fin 7 → Prop) : ℝ := sorry

end knights_and_knaves_l621_621468


namespace bus_seat_capacity_l621_621885

theorem bus_seat_capacity (x : ℕ) : 15 * x + (15 - 3) * x + 11 = 92 → x = 3 :=
by
  sorry

end bus_seat_capacity_l621_621885


namespace number_of_dogs_l621_621718

-- Define the conditions
variables (D C : ℝ)
variables (N : ℕ)
hypothesis h1 : D ≥ 0  -- ensuring non-negativity as implicit
hypothesis h2 : C = 0.03125
hypothesis h3 : 4 * C = D
hypothesis h4 : D = N * C / (1 - 4 * C)

-- Define the statement
theorem number_of_dogs (D C : ℝ) (N : ℕ) (h1 : D ≥ 0) (h2 : C = 0.03125) (h3 : 4 * C = D) (h4 : D = N * C / (1 - 4 * C)) : N = 7 :=
sorry

end number_of_dogs_l621_621718


namespace ceil_neg_sqrt_frac_l621_621263

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l621_621263


namespace fifteen_percent_of_x_is_ninety_l621_621764

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l621_621764


namespace domain_of_f_l621_621554

-- Define the function f(x)
def f (x : ℝ) : ℝ := sqrt (2 - 2^x) + log x

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 - 2^x ≥ 0
def condition2 (x : ℝ) : Prop := x > 0

-- Translate into a Lean statement
theorem domain_of_f : ∀ x, condition1 x ∧ condition2 x ↔ (0 < x ∧ x ≤ 1) := 
by 
  intro x,
  sorry

end domain_of_f_l621_621554


namespace intersect_y_axis_at_l621_621205

namespace intersect_y_axis

def line_slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.2 - p2.2) / (p1.1 - p2.1)

def line_equation (p : ℝ × ℝ) (m : ℝ) (x : ℝ) : ℝ :=
  m * (x - p.1) + p.2

theorem intersect_y_axis_at {p1 p2 : ℝ × ℝ} (h1 : p1 = (3, 18)) (h2 : p2 = (-9, -6)) :
  ∃ y : ℝ, line_equation p1 (line_slope p1 p2) 0 = y ∧ y = 12 :=
by
  sorry

end intersect_y_axis

end intersect_y_axis_at_l621_621205


namespace measure_of_angle_ABC_l621_621396

noncomputable def angle_measure_pentagon (ABCDE : Type) [Pentagon ABCDE]
    (angle_ABC : ℝ) (angle_DBE : ℝ) 
    (h1 : equilateral_pentagon ABCDE) 
    (h2 : angle_ABC = 2 * angle_DBE) : Prop :=
  angle_ABC = 60

def equilateral_pentagon {ABCDE : Type}
    (ABCDE : Pentagon ABCDE) : Prop :=
  ∀ (A B C D E : Point ABCDE), equilateral ABCDE

theorem measure_of_angle_ABC {ABCDE : Type} [Pentagon ABCDE]
    (h1 : equilateral_pentagon ABCDE)
    (h2 : ∃ angle_ABC angle_DBE : ℝ, angle_ABC = 2 * angle_DBE) :
    ∃ angle_ABC : ℝ, angle_ABC = 60 := by
  sorry

end measure_of_angle_ABC_l621_621396


namespace Sues_necklace_total_beads_l621_621684

theorem Sues_necklace_total_beads 
  (purple_beads : ℕ)
  (blue_beads : ℕ)
  (green_beads : ℕ)
  (h1 : purple_beads = 7)
  (h2 : blue_beads = 2 * purple_beads)
  (h3 : green_beads = blue_beads + 11) :
  purple_beads + blue_beads + green_beads = 46 :=
by
  sorry

end Sues_necklace_total_beads_l621_621684


namespace a_x1_x2_x13_eq_zero_l621_621562

theorem a_x1_x2_x13_eq_zero {a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ}
  (h1: a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) *
             (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2: a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) *
             (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := by
  sorry

end a_x1_x2_x13_eq_zero_l621_621562


namespace sum_of_midpoint_coordinates_l621_621598

theorem sum_of_midpoint_coordinates :
  let A := (10 : ℝ, 24 : ℝ)
      B := (-4 : ℝ, -12 : ℝ)
      M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  in M.1 + M.2 = 9 := 
by
  let A := (10 : ℝ, 24 : ℝ)
  let B := (-4 : ℝ, -12 : ℝ)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  have h1 : M.1 = 3 := by sorry
  have h2 : M.2 = 6 := by sorry
  calc
    M.1 + M.2 = 3 + 6 := by rw [h1, h2]
               ... = 9 := by norm_num

end sum_of_midpoint_coordinates_l621_621598


namespace correct_propositions_l621_621993

-- Define the propositions as conditions.
def prop1 := ∀ (a b : ℝ^2), a = -b → ‖a‖ = ‖b‖
def prop2 := ∀ (A B C D : ℝ^2), collinear A B C D → lie_on_same_line A B C D
def prop3 := ∀ (a b : ℝ^2), ‖a‖ = ‖b‖ → (a = b ∨ a = -b)
def prop4 := ∀ (a b : ℝ^2), a ⬝ b = 0 → (a = 0 ∨ b = 0)

-- Define the number of correct propositions.
def correct_count := (if prop1 then 1 else 0) + (if prop2 then 1 else 0) + (if prop3 then 1 else 0) + (if prop4 then 1 else 0)

-- Prove that the correct count of propositions is 1.
theorem correct_propositions : correct_count = 1 := by
  sorry

end correct_propositions_l621_621993


namespace probability_product_odd_prob_lt_eighth_l621_621583

theorem probability_product_odd_prob_lt_eighth:
  let total_numbers := 2020
  let odd_numbers := 1010
  let first_odd_prob := (odd_numbers : ℚ) / total_numbers
  let second_odd_prob := (odd_numbers - 1 : ℚ) / (total_numbers - 1)
  let third_odd_prob := (odd_numbers - 2 : ℚ) / (total_numbers - 2)
  let p := first_odd_prob * second_odd_prob * third_odd_prob
  p < 1 / 8 :=
by
  sorry

end probability_product_odd_prob_lt_eighth_l621_621583


namespace line_tangent_constant_sum_l621_621847

noncomputable def parabolaEquation (x y : ℝ) : Prop :=
  y ^ 2 = 4 * x

noncomputable def circleEquation (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

noncomputable def isTangent (l : ℝ → ℝ) (x y : ℝ) : Prop :=
  l x = y ∧ ((x - 2) ^ 2 + y ^ 2 = 4)

theorem line_tangent_constant_sum (l : ℝ → ℝ) (A B P : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabolaEquation x₁ y₁ →
  parabolaEquation x₂ y₂ →
  isTangent l (4 / 5) (8 / 5) →
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  let F := (1, 0)
  let distance (p1 p2 : ℝ × ℝ) : ℝ := (Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2))
  (distance F A) + (distance F B) - (distance A B) = 2 :=
sorry

end line_tangent_constant_sum_l621_621847


namespace units_digit_base_8_l621_621545

theorem units_digit_base_8 (a b : ℕ) (ha : a = 348) (hb : b = 76) :
  ((a * b) % 8) = 0 :=
by {
  rw [ha, hb],
  sorry
}

end units_digit_base_8_l621_621545


namespace permutation_10_3_l621_621893

theorem permutation_10_3 : ∃ (P : ℕ → ℕ → ℕ), P 10 3 = 720 :=
  by 
  let P : ℕ → ℕ → ℕ := λ n m, n.factorial / (n - m).factorial
  use P
  sorry

end permutation_10_3_l621_621893


namespace part_I_part_II_l621_621348

variables {a b : ℝ^3}

-- Define the conditions
def angle_between (a b : ℝ^3) := real.angle a b = 2 * real.pi / 3
def norm_a := |a| = 2
def norm_b := |b| = 3
def m := 3 • a - 2 • b
def n (k : ℝ) := 2 • a + k • b

-- Part (I): Prove that if m and n are perpendicular, then k = 4/3
theorem part_I (h1 : m ⬝ n k = 0) : k = 4 / 3 :=
sorry

-- Part (II): Prove that when k = -4/3, the angle between m and n is 0
theorem part_II : angle_between m (n (-4 / 3)) = 0 :=
sorry

end part_I_part_II_l621_621348


namespace race_participants_minimum_l621_621524

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l621_621524


namespace subtracted_number_divisible_by_fifteen_l621_621601

theorem subtracted_number_divisible_by_fifteen :
  ∃ d : ℕ, (d = 15) ∧ (∀ n m : ℕ, n = 427398 → m = n - 3 → m % d = 0) :=
begin
  sorry
end

end subtracted_number_divisible_by_fifteen_l621_621601


namespace find_f_2007_l621_621245

def f : ℝ → ℝ
| x => if x ∈ Icc (-1 : ℝ) 1 then x^3 else sorry  -- we will define f only on [-1, 1] and delay full definition to conditions

lemma f_neg (x : ℝ) : f (-x) = -f x :=
sorry  -- Given as condition

lemma f_sym (x : ℝ) : f (1 + x) = f (1 - x) :=
sorry  -- Given as condition

lemma f_periodic4 (x : ℝ) : f (x + 4) = f x :=
begin
  -- Derive periodicity from the given conditions
  sorry
end

theorem find_f_2007 : f 2007 = -1 :=
begin
  -- Use periodicity and given values to find the required result
  sorry
end

end find_f_2007_l621_621245


namespace converges_iff_subseq_converges_l621_621072

noncomputable def b (n : ℕ) (a : ℕ → ℝ) : ℂ :=
  (1 / n : ℂ) * ∑ r in finset.range (n + 1), complex.exp (complex.I * a r)

theorem converges_iff_subseq_converges
  (a : ℕ → ℝ) (k : ℂ) :
  (tendsto (λ n, b n a) at_top (nhds k)) ↔ (tendsto (λ n, b (n^2) a) at_top (nhds k)) :=
begin
  sorry
end

end converges_iff_subseq_converges_l621_621072


namespace find_y_l621_621543

theorem find_y (y : ℝ) (h : (3 + (3 * y - 4).sqrt).sqrt = 10.sqrt) : y = 53 / 3 := 
by 
  sorry

end find_y_l621_621543


namespace integer_product_zero_l621_621565

theorem integer_product_zero (a : ℤ) (x : Fin 13 → ℤ)
  (h : a = ∏ i, (1 + x i) ∧ a = ∏ i, (1 - x i)) :
  a * ∏ i, x i = 0 :=
sorry

end integer_product_zero_l621_621565


namespace count_integers_in_square_range_l621_621795

theorem count_integers_in_square_range :
  {x : ℕ // 400 ≤ x^2 ∧ x^2 ≤ 600}.to_finset.card = 5 :=
by
  sorry

end count_integers_in_square_range_l621_621795


namespace knights_and_knaves_l621_621467

/-- Proof that there are exactly 5 knights among 7 inhabitants given specific conditions -/
theorem knights_and_knaves : 
  ∀ (inhabitants : Fin 7 → Prop)
    (knight truthful liar : Prop)
    (H1 : knight = truthful)
    (H2 : liar = ¬truthful)
    (H3 : (inhabitants 0 = knight))
    (H4 : (inhabitants 1 = knight))
    (H5 : (inhabitants 2 = liar ∧ (inhabitants 0 = knight ∨ inhabitants 1 = knight)))
    (H6 : (inhabitants 3 = liar ∧ (inhabitants 0 = truth ∧ inhabitants 1 = truth ∧ inhabitants 2 = liar → 2/3 ≥ 0.65)))
    (H7 : (inhabitants 4 = knight))
    (H8 : (inhabitants 5 = knight ∧ (∃ (half_knights : Fin 6 → Prop), (inhabitants 0 = knight) ∧ (inhabitants 1 = knight) ∧ (inhabitants 4 = knight) ∧ (inhabitants 5 = knight) ∧ counting_knaves half_knights < 3))
    (H9 : (inhabitants 6 = knight ∧ (counting_knights inhabitants 6 ≥ 0.65))),
  counting_knights inhabitants = 5 := 
by sorry

def counting_knights (inhabitants : Fin 7 → Prop) : ℝ := sorry
def counting_knaves (inhabitants : Fin 7 → Prop) : ℝ := sorry

end knights_and_knaves_l621_621467


namespace triangle_side_ratio_l621_621385

variables (A B C a b c : ℝ)

theorem triangle_side_ratio (h1 : 2 * cos (A / 2) ^ 2 = (real.sqrt 3 / 3) * sin A)
                           (h2 : sin (B - C) = 4 * cos B * sin C)
                           (hABC : A + B + C = real.pi) :
  b / c = 1 + real.sqrt 6 :=
sorry

end triangle_side_ratio_l621_621385


namespace hinted_prime_factor_of_84_l621_621908

-- Declare the given conditions as hypotheses
def jills_favorite_number_is_even (n : ℕ) : Prop := even n
def jills_favorite_number_has_repeating_prime_factors (n : ℕ) : Prop := ∃ p : ℕ, prime p ∧ p * p ∣ n
def johns_best_guess_is_84 (n : ℕ) : Prop := n = 84

-- Define the main theorem to prove
theorem hinted_prime_factor_of_84 : ∀ (n : ℕ), 
  johns_best_guess_is_84 n ∧ jills_favorite_number_is_even n ∧ jills_favorite_number_has_repeating_prime_factors n → 
  ∃ p : ℕ, prime p ∧ p = 2 :=
begin
  sorry
end

end hinted_prime_factor_of_84_l621_621908


namespace chapsalttocks_boxes_needed_l621_621640

theorem chapsalttocks_boxes_needed :
  (∀ (flour_per_chaps : ℝ) (total_chapsalttocks : ℝ) (flour_per_box : ℝ),
    flour_per_chaps = 1.2 ∧ total_chapsalttocks = 53.2 ∧ flour_per_box = 3 →
    let total_flour_needed := (flour_per_chaps * total_chapsalttocks) / 3.8 in
    ∃ (boxes_needed : ℕ), boxes_needed = ⌈total_flour_needed / flour_per_box⌉) :=
by
  intros flour_per_chaps total_chapsalttocks flour_per_box h
  rcases h with ⟨hflour_per_chaps, htotal_chapsalttocks, hflour_per_box⟩
  have total_flour_needed : ℝ := (flour_per_chaps * total_chapsalttocks) / 3.8
  use ⌈total_flour_needed / flour_per_box⌉
  sorry

end chapsalttocks_boxes_needed_l621_621640


namespace distinct_elements_count_l621_621057

noncomputable def numDistinctElements : ℕ := 1504

theorem distinct_elements_count :
  let S := { ⌊(n^2 : ℚ) / 2005⌋ | n ∈ finset.range 2005 }
  finset.card S = numDistinctElements :=
sorry

end distinct_elements_count_l621_621057


namespace problem1_problem2_l621_621639

-- First Problem
theorem problem1 (x : ℝ) (h₁ : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 3)) :
  1 ≤ sin (2 * x - Real.pi / 6) + 2 ∧ sin (2 * x - Real.pi / 6) + 2 ≤ 3 := 
by 
  sorry

-- Second Problem
theorem problem2 (x : ℝ) (a : ℝ) (h₂ : x ∈ Set.Icc 0 Real.pi) :
  (a ≥ 2 → (sin x ^ 2 - a * cos x + 3 ≤ 3 - a ∧ sin x ^ 2 - a * cos x + 3 ≥ 3 + a)) ∧
  (0 ≤ a ∧ a < 2 → (sin x ^ 2 - a * cos x + 3 ≤ 3 - a ∧ sin x ^ 2 - a * cos x + 3 ≥ 4 + a ^ 2 / 4)) ∧
  (-2 < a ∧ a < 0 → (sin x ^ 2 - a * cos x + 3 ≤ 3 + a ∧ sin x ^ 2 - a * cos x + 3 ≥ 4 + a ^ 2 / 4)) ∧
  (a ≤ -2 → (sin x ^ 2 - a * cos x + 3 ≤ 3 + a ∧ sin x ^ 2 - a * cos x + 3 ≥ 3 - a)) := 
by 
  sorry

end problem1_problem2_l621_621639


namespace total_weight_of_new_individuals_l621_621127

-- Given conditions:
def avg_weight_increase := 6.5
def weights_of_left := [75, 80, 90]
def total_people := 10

-- Define total weight function based on given conditions
def total_weight_increase := total_people * avg_weight_increase
def weight_difference := (weights_of_left.sum : ℝ)

-- Definition of weight change:
def total_weight_change := total_weight_increase + weight_difference

-- Prove the target theorem
theorem total_weight_of_new_individuals : total_weight_change = 65 := by
  sorry

end total_weight_of_new_individuals_l621_621127


namespace scalar_property_l621_621994

variables {α β γ : ℝ} {v : ℝ} 
variables (i j k : ℝ) [∀ (v : ℝ), v * i = 0] [∀ (v : ℝ), v * j = 0] [∀ (v : ℝ), v * k = 0]

theorem scalar_property (α β γ : ℝ) (i j k : ℝ) (v : ℝ) :
  (α * (v * i) + β * (v * j) + γ * (v * k) = v * (α + β + γ - 1)) :=
by sorry

end scalar_property_l621_621994


namespace packs_needed_is_six_l621_621111

variable (l_bedroom l_bathroom l_kitchen l_basement : ℕ)

def total_bulbs_needed := l_bedroom + l_bathroom + l_kitchen + l_basement
def garage_bulbs_needed := total_bulbs_needed / 2
def total_bulbs_with_garage := total_bulbs_needed + garage_bulbs_needed
def packs_needed := total_bulbs_with_garage / 2

theorem packs_needed_is_six
    (h1 : l_bedroom = 2)
    (h2 : l_bathroom = 1)
    (h3 : l_kitchen = 1)
    (h4 : l_basement = 4) :
    packs_needed l_bedroom l_bathroom l_kitchen l_basement = 6 := by
  sorry

end packs_needed_is_six_l621_621111


namespace ceil_neg_sqrt_fraction_l621_621279

theorem ceil_neg_sqrt_fraction :
  (⌈-real.sqrt (64 / 9)⌉ = -2) :=
by
  -- Define the necessary conditions
  have h1 : real.sqrt (64 / 9) = 8 / 3 := by sorry,
  have h2 : -real.sqrt (64 / 9) = -8 / 3 := by sorry,
  -- Apply the ceiling function and prove the result
  exact sorry

end ceil_neg_sqrt_fraction_l621_621279


namespace ceil_neg_sqrt_64_over_9_l621_621274

theorem ceil_neg_sqrt_64_over_9 : Real.ceil (-Real.sqrt (64 / 9)) = -2 := 
by
  sorry

end ceil_neg_sqrt_64_over_9_l621_621274


namespace ceil_of_neg_sqrt_frac_64_over_9_l621_621292

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l621_621292


namespace kain_forces_win_l621_621896

theorem kain_forces_win (a : Fin 100 → ℤ)
  (move : (Fin 100 → ℤ) → Fin 100 → (Fin 100 → ℤ))
  (abel_decision : ∀ (a : Fin 100 → ℤ), bool) :
  ∃ N : ℕ, ∀ n ≥ N, (countp (λ i, a i % 4 = 0) ≥ 98) (move^[n] a) :=
begin
  sorry
end

end kain_forces_win_l621_621896


namespace distance_relation_l621_621195

-- Defining initial conditions and variables
variable (t : ℝ)  -- Time in hours
constant travel_initial : ℝ := 10  -- Initial distance in kilometers
constant speed : ℝ := 60  -- Constant speed in km/h

-- Definition of the total distance travelled
def distance_travelled (t : ℝ) : ℝ := travel_initial + speed * t

-- The theorem that needs to be proven
theorem distance_relation (t : ℝ) : distance_travelled t = 10 + 60 * t := by
  sorry

end distance_relation_l621_621195


namespace train_speed_l621_621213

theorem train_speed
(length_train : ℕ)
(length_bridge : ℕ)
(time_seconds : ℕ)
(h_train : length_train = 120)
(h_bridge : length_bridge = 255)
(h_time : time_seconds = 30) :
  (length_train + length_bridge) / time_seconds * 3.6 = 45 :=
by sorry

end train_speed_l621_621213


namespace KnightsCount_l621_621457

def isKnight (n : ℕ) : Prop := sorry -- Define isKnight
def tellsTruth (n : ℕ) : Prop := sorry -- Define tellsTruth

-- Statements by the inhabitants
axiom H1 : isKnight 0 ↔ tellsTruth 0
axiom H2 : tellsTruth 1 ↔ isKnight 0
axiom H3 : tellsTruth 2 ↔ (¬(isKnight 0) ∨ ¬(isKnight 1))
axiom H4 : tellsTruth 3 ↔ (¬(isKnight 0) ∨ ¬(isKnight 1) ∨ ¬(isKnight 2))
axiom H5 : tellsTruth 4 ↔ (isKnight 0 ∧ isKnight 1 ∧ (isKnight 2 ∨ isKnight 3))
axiom H6 : tellsTruth 5 ↔ ((¬isKnight 0 ∨ ¬isKnight 1 ∨ isKnight 2) ∧ (¬isKnight 3 ∨ isKnight 4))
axiom H7 : tellsTruth 6 ↔ (isKnight 0 ∧ isKnight 1 ∧ isKnight 2 ∧ isKnight 3 ∧ ¬(¬isKnight 4 ∧ ¬isKnight 5))

theorem KnightsCount : ∃ k1 k2 k3 k4 k5 k6 k7 : Prop,
  (isKnight 0 = k1 ∧ isKnight 1 = k2 ∧ isKnight 2 = k3 ∧ isKnight 3 = k4 ∧ isKnight 4 = k5 ∧ isKnight 5 = k6 ∧ isKnight 6 = k7) ∧ 
  tellsTruth 0 ∧ tellsTruth 1 ∧ tellsTruth 2 ∧ tellsTruth 3 ∧ tellsTruth 4 ∧ tellsTruth 5 ∧ tellsTruth 6 ∧
  (5 = 1 + if k1 then 1 else 0 + if k2 then 1 else 0 + if k3 then 1 else 0 + if k4 then 1 else 0 + if k5 then 1 else 0 + if k6 then 1 else 0 + if k7 then 1 else 0)
:=
by
  sorry

end KnightsCount_l621_621457


namespace fifteen_percent_of_x_is_ninety_l621_621766

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l621_621766


namespace hal_paul_difference_l621_621365

def halAnswer : Int := 12 - (3 * 2) + 4
def paulAnswer : Int := (12 - 3) * 2 + 4

theorem hal_paul_difference :
  halAnswer - paulAnswer = -12 := by
  sorry

end hal_paul_difference_l621_621365


namespace ceil_neg_sqrt_frac_l621_621267

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l621_621267


namespace frogs_climbed_onto_logs_l621_621861

-- Definitions of the conditions
def f_lily : ℕ := 5
def f_rock : ℕ := 24
def f_total : ℕ := 32

-- The final statement we want to prove
theorem frogs_climbed_onto_logs : f_total - (f_lily + f_rock) = 3 :=
by
  sorry

end frogs_climbed_onto_logs_l621_621861


namespace find_value_of_sum_of_squares_l621_621061

theorem find_value_of_sum_of_squares
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = 0)
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 6 / 5 := by
  sorry

end find_value_of_sum_of_squares_l621_621061


namespace general_term_of_sequence_l621_621138

theorem general_term_of_sequence (n : ℕ) :
  ∃ (a : ℕ → ℚ),
    a 1 = 1 / 2 ∧ 
    a 2 = -2 ∧ 
    a 3 = 9 / 2 ∧ 
    a 4 = -8 ∧ 
    a 5 = 25 / 2 ∧ 
    ∀ n, a n = (-1) ^ (n + 1) * (n ^ 2 / 2) := 
by
  sorry

end general_term_of_sequence_l621_621138


namespace area_of_triangle_ABC_l621_621142

noncomputable def pointA : ℝ × ℝ :=
  ⟨1, 2⟩

noncomputable def pointB : ℝ × ℝ :=
  ⟨-1, 0⟩

noncomputable def pointC : ℝ × ℝ :=
  ⟨3, 0⟩

theorem area_of_triangle_ABC :
  let A := pointA in
  let B := pointB in
  let C := pointC in
  let bx := B.1 in
  let by := B.2 in
  let cx := C.1 in
  let cy := C.2 in
  let ax := A.1 in
  let ay := A.2 in
  (1/2) * Real.abs ((bx * cy + cx * ay + ax * by) - (by * cx + cy * ax + ay * bx)) = 4 :=
by
  sorry

end area_of_triangle_ABC_l621_621142


namespace find_unknown_number_l621_621775

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l621_621775


namespace race_participants_least_number_l621_621496

noncomputable def minimum_race_participants 
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : ℕ := 61

theorem race_participants_least_number
  (hAndrei : ∀ (x : ℕ), 3*x + 1)
  (hDima : ∀ (y : ℕ), 4*y + 1)
  (hLenya : ∀ (z : ℕ), 5*z + 1) : minimum_race_participants hAndrei hDima hLenya = 61 := 
sorry

end race_participants_least_number_l621_621496


namespace fencing_rate_per_meter_l621_621973

constant len_w_relation : ℕ → ℕ -- The relation between length and width
constant plot_perimeter : ℕ -- The perimeter of the plot
constant total_cost : ℕ -- The total cost for fencing

axiom len_w_relation_ax : ∀ w : ℕ, len_w_relation w = w + 10
axiom plot_perimeter_ax : plot_perimeter = 140
axiom total_cost_ax : total_cost = 910

theorem fencing_rate_per_meter : 
  ∃ r : ℕ, r = total_cost / plot_perimeter ∧ r = 65 / 10 :=
sorry

end fencing_rate_per_meter_l621_621973


namespace find_n_for_K_2016_find_K_2016_value_find_S_2016_value_l621_621377

def V (a b c : ℕ) : ℕ := a^3 + b^3 + c^3 - 3 * a * b * c

def pseudoCubicSeq : ℕ → ℕ
| 0       := 0
| (n + 1) := sorry  -- (this will contain the definition to generate the next pseudo-cubic number)

noncomputable def S (n : ℕ) : ℕ := (∑ i in Finset.range (n + 1), pseudoCubicSeq i)

theorem find_n_for_K_2016 : ∃ n, pseudoCubicSeq n = 2016 :=
  sorry  -- Proof that there exists an n such that K_n = 2016

theorem find_K_2016_value : pseudoCubicSeq 2016 = 2592 :=
  sorry  -- Proof that K_2016 = 2592

theorem find_S_2016_value : S 2016 = 2614032 :=
  sorry  -- Proof that S_2016 = 2614032

end find_n_for_K_2016_find_K_2016_value_find_S_2016_value_l621_621377


namespace volume_of_tetrahedron_is_correct_l621_621006

noncomputable def volume_tetrahedron (PQ PQR PQS: ℝ) (areaPQR areaPQS: ℝ) (angle: ℝ) : ℝ :=
  let height := (areaPQS * sin angle) / (0.5 * PQ)
  (1 / 3) * areaPQR * height

theorem volume_of_tetrahedron_is_correct :
  volume_tetrahedron 5 18 24 18 (real.pi / 4) = 40.3931 :=
by
  sorry

end volume_of_tetrahedron_is_correct_l621_621006


namespace kermit_sleep_positions_l621_621416

theorem kermit_sleep_positions :
  let joules : Int := 100 in
  let energy_consumed (x y : Int) : Bool := (Int.abs x + Int.abs y) = joules in
  let possible_positions (x y : Int) : Prop := energy_consumed x y in
  (∑ y in Finset.range (2 * joules + 1), 2 * (joules - Int.abs (y - joules)) + 1) = 10201 :=
begin
  sorry -- Proof is omitted
end

end kermit_sleep_positions_l621_621416


namespace corrected_mean_is_32_5_l621_621976

-- Define the conditions
def initial_mean : ℝ := 32
def num_observations : ℕ := 50
def incorrect_observation : ℝ := 23
def correct_observation : ℝ := 48

-- Theorem stating the corrected mean
theorem corrected_mean_is_32_5 :
  let incorrect_total_sum := initial_mean * num_observations
      corrected_total_sum := incorrect_total_sum - incorrect_observation + correct_observation
      corrected_mean := corrected_total_sum / num_observations
  in corrected_mean = 32.5 :=
by
  sorry

end corrected_mean_is_32_5_l621_621976


namespace tan_pi_over_9_times_tan_2pi_over_9_times_tan_4pi_over_9_eq_3_l621_621726

noncomputable def tan_poly_identity : Prop :=
  let θ1 := Real.pi / 9 in
  let θ2 := 2 * Real.pi / 9 in
  let θ3 := 4 * Real.pi / 9 in
  tan θ1 * tan θ2 * tan θ3 = 3

theorem tan_pi_over_9_times_tan_2pi_over_9_times_tan_4pi_over_9_eq_3 :
  tan_poly_identity :=
  sorry

end tan_pi_over_9_times_tan_2pi_over_9_times_tan_4pi_over_9_eq_3_l621_621726


namespace slope_of_intersection_points_l621_621309

theorem slope_of_intersection_points :
  ∀ s : ℝ, ∃ k b : ℝ, (∀ (x y : ℝ), (2 * x - 3 * y = 4 * s + 6) ∧ (2 * x + y = 3 * s + 1) → y = k * x + b) ∧ k = -2/13 := 
by
  intros s
  -- Proof to be provided here
  sorry

end slope_of_intersection_points_l621_621309


namespace area_ratio_2_l621_621102
-- Import necessary libraries

-- Define the structures of a point and square
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Square :=
  (A B C D : Point)
  (AB : A.x = B.x ∧ B.y = B.y)
  (BC : B.x = C.x ∧ B.y = C.y)
  (CD : C.x = D.x ∧ C.y = D.y)
  (DA : D.x = A.x ∧ D.y = A.y)

-- Define the given conditions
variables (A B C D E F P Q : Point)
variables (s : Square)

-- Two points E and F on sides BC and CD
def E_on_BC (s : Square) (E : Point) := E.x = s.B.x ∧ E.y = s.C.y
def F_on_CD (s : Square) (F : Point) := F.x = s.C.x ∧ F.y = s.D.y

-- Angle between lines EAF is 45 degrees
def angle_EAF_45 (A E F : Point) : Prop :=
  ∃ θ, θ = 45 ∧ arctan((F.y - A.y) / (F.x - A.x)) - arctan((E.y - A.y) / (E.x - A.x)) = θ

-- Definition of line intersection points P and Q
def intersection_AE_BD (A E B D P : Point) : Prop :=
  P.x = ((B.x * D.x * (E.y - A.y) + A.x * D.x * (B.y - E.y) + A.x * E.x * (D.y - B.y)) /
         (B.x * (E.y - A.y) + A.x * (B.y - E.y) + A.x * (D.y - B.y)))

def intersection_AF_BD (A F B D Q : Point) : Prop :=
  Q.x = ((B.x * D.x * (F.y - A.y) + A.x * D.x * (B.y - F.y) + A.x * F.x * (D.y - B.y)) /
         (B.x * (F.y - A.y) + A.x * (B.y - F.y) + A.x * (D.y - B.y)))

-- Prove that the area ratio is 2
theorem area_ratio_2 (s : Square) (A E F P Q : Point)
  (h1 : E_on_BC s E) (h2 : F_on_CD s F)
  (h3 : angle_EAF_45 A E F)
  (h4 : intersection_AE_BD A E s.B s.D P)
  (h5 : intersection_AF_BD A F s.B s.D Q) :
  (area (triangle A E F)) / (area (triangle A P Q)) = 2 :=
by sorry

end area_ratio_2_l621_621102


namespace smallest_cross_subtracting_number_maximum_cross_subtracting_number_divisible_by_9_l621_621202

def is_cross_subtracting (m : ℕ) : Prop :=
  let a := m / 1000 in
  let b := (m % 1000) / 100 in
  let c := (m % 100) / 10 in
  let d := m % 10 in
  |a - c| = 2 ∧ |b - d| = 1

theorem smallest_cross_subtracting_number : ∃ m : ℕ, is_cross_subtracting m ∧ m = 1031 :=
by {
  use 1031,
  -- proof of conditions |a - c| = 2 and |b - d| = 1
  sorry
}

theorem maximum_cross_subtracting_number_divisible_by_9 (s t : ℕ) : 
  ∃ m : ℕ, is_cross_subtracting m ∧ m % 9 = 0 ∧ (let a := m / 1000 let d := m % 10 in s = a + d) ∧ (let b := (m % 1000) / 100 let c := (m % 100) / 10 in t = b + c) ∧ s / t = 1 ∧ m = 9675 :=
by {
  use 9675,
  -- proof of conditions is_cross_subtracting, divisible by 9, and s / t = integer
  sorry
}

end smallest_cross_subtracting_number_maximum_cross_subtracting_number_divisible_by_9_l621_621202


namespace collinear_d_e_t_l621_621419

open EuclideanGeometry

-- Definitions of geometric entities and properties
variables {A B C D E T : Point}

-- Given conditions
axiom hypotenuse_bc : ∃ (ABC : Triangle), right_triangle ABC ∧ hypot AB BC C
axiom tangent_at_A : tangent (circumcircle ABC) A T
axiom ad_eq_bd : dist A D = dist B D
axiom ae_eq_ce : dist A E = dist C E
axiom angle_cbd_eq_angle_bce : ∠CBD < 90 ∧ ∠CBD = ∠BCE

-- The theorem to prove D, E, T are collinear
theorem collinear_d_e_t (ABC : Triangle) (right_triangle ABC ∧ hypot AB BC C) (tangent (circumcircle ABC) A T) (dist A D = dist B D) (dist A E = dist C E) (∠CBD < 90 ∧ ∠CBD = ∠BCE) : collinear D E T :=
by sorry

end collinear_d_e_t_l621_621419


namespace probability_product_odd_prob_lt_eighth_l621_621584

theorem probability_product_odd_prob_lt_eighth:
  let total_numbers := 2020
  let odd_numbers := 1010
  let first_odd_prob := (odd_numbers : ℚ) / total_numbers
  let second_odd_prob := (odd_numbers - 1 : ℚ) / (total_numbers - 1)
  let third_odd_prob := (odd_numbers - 2 : ℚ) / (total_numbers - 2)
  let p := first_odd_prob * second_odd_prob * third_odd_prob
  p < 1 / 8 :=
by
  sorry

end probability_product_odd_prob_lt_eighth_l621_621584
