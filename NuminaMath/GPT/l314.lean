import Mathlib

namespace max_unique_bills_l314_314756

theorem max_unique_bills (num_bills : ℕ) (num_slots : ℕ) (digits : set ℕ) :
  num_bills = 20 →
  num_slots = 7 →
  digits = {1, 2} →
  ∃ (max_unique : ℕ), max_unique = 2 :=
by
  intros h_bills h_slots h_digits
  use 2
  sorry

end max_unique_bills_l314_314756


namespace books_given_away_l314_314114

theorem books_given_away (original_books : ℝ) (books_left : ℝ) (books_given : ℝ) 
    (h1 : original_books = 54.0) 
    (h2 : books_left = 31) : 
    books_given = original_books - books_left → books_given = 23 :=
by
  sorry

end books_given_away_l314_314114


namespace grandfather_age_5_years_back_l314_314391

variable (F S G : ℕ)

-- Conditions
def father_age : Prop := F = 58
def son_current_age : Prop := S = 58 - S
def son_grandfather_age_relation : Prop := S - 5 = 1 / 2 * (G - 5)

-- Theorem: Prove the grandfather's age 5 years back given the conditions.
theorem grandfather_age_5_years_back (h1 : father_age F) (h2 : son_current_age S) (h3 : son_grandfather_age_relation S G) : G = 2 * S - 5 :=
sorry

end grandfather_age_5_years_back_l314_314391


namespace polynomial_functions_count_equals_16_l314_314471

open Classical

noncomputable def polynomial_functions_number : ℕ :=
  let count := #((λ (f : ℝ → ℝ), ∃ a b c d : ℝ, f = (λ x, a * x^3 + b * x^2 + c * x + d) ∧ 
                      (f(x) * f(-x) = f(x^3))) 
                  | a b c d, 
                    (a = 0 ∨ a = -1) ∧ 
                    (b = 0 ∨ b = 1) ∧ 
                    (c = 0 ∨ c = -1) ∧ 
                    (d = 0 ∨ d = 1)) in 
   count.toNat

theorem polynomial_functions_count_equals_16 : polynomial_functions_number = 16 :=
  sorry

end polynomial_functions_count_equals_16_l314_314471


namespace part_a_part_b_part_c_l314_314363

-- Define initial setup and conditions
def average (scores: List ℚ) : ℚ :=
  scores.sum / scores.length

-- Part (a)
theorem part_a (A B : List ℚ) (a b : ℚ) (A' : List ℚ) (B' : List ℚ) :
  average A = a ∧ average B = b ∧ average A' = a ∧ average B' = b ∧
  average A' > a ∧ average B' > b :=
sorry

-- Part (b)
theorem part_b (A B : List ℚ) : 
  ∀ a b : ℚ, (average A = a ∧ average B = b ∧ ∀ A' : List ℚ, average A' > a ∧ ∀ B' : List ℚ, average B' > b) :=
sorry

-- Part (c)
theorem part_c (A B C : List ℚ) (a b c : ℚ) (A' B' C' A'' B'' C'' : List ℚ) :
  average A = a ∧ average B = b ∧ average C = c ∧
  average A' = a ∧ average B' = b ∧ average C' = c ∧
  average A'' = a ∧ average B'' = b ∧ average C'' = c ∧
  average A' > a ∧ average B' > b ∧ average C' > c ∧
  average A'' > average A' ∧ average B'' > average B' ∧ average C'' > average C' :=
sorry

end part_a_part_b_part_c_l314_314363


namespace units_digit_of_3_pow_y_l314_314620

theorem units_digit_of_3_pow_y
    (x : ℕ)
    (h1 : (2^3)^x = 4096)
    (y : ℕ)
    (h2 : y = x^3) :
    (3^y) % 10 = 1 :=
by
  sorry

end units_digit_of_3_pow_y_l314_314620


namespace f_f_1_equals_4_l314_314006

noncomputable def f (x : ℝ) : ℝ :=
  if x > 2 then x + 1 / (x - 2) else x^2 + 2

theorem f_f_1_equals_4 : f (f 1) = 4 := by sorry

end f_f_1_equals_4_l314_314006


namespace women_more_than_men_l314_314718

def men (W : ℕ) : ℕ := (5 * W) / 11

theorem women_more_than_men (M W : ℕ) (h1 : M + W = 16) (h2 : M = (5 * W) / 11) : W - M = 6 :=
by
  sorry

end women_more_than_men_l314_314718


namespace triple_complement_angle_l314_314285

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314285


namespace parallel_lines_sufficient_not_necessary_l314_314578

-- Definitions and Given Conditions
def l1 (a : ℝ) : ℝ × ℝ → Prop := λ p, a * p.1 + 2 * p.2 = 0 
def l2 (a : ℝ) : ℝ × ℝ → Prop := λ p, p.1 + (a + 1) * p.2 + 4 = 0

-- Theorem Statement: Sufficient but not necessary condition for lines to be parallel
theorem parallel_lines_sufficient_not_necessary (a : ℝ) : (∀p, l1 a p → l2 1 p) ∧ ¬ (∀p, l1 1 p → l2 a p) := 
by
  sorry

end parallel_lines_sufficient_not_necessary_l314_314578


namespace angle_measure_triple_complement_l314_314219

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314219


namespace intersection_points_count_l314_314108

theorem intersection_points_count (A : ℝ) (hA : A > 0) :
  ((A > 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y) ∧
                              (x ≠ 0 ∨ y ≠ 0)) ∧
  ((A ≤ 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y)) :=
by
  sorry

end intersection_points_count_l314_314108


namespace angle_measure_triple_complement_l314_314237

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314237


namespace part1_solution_set_part2_range_a_l314_314943

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314943


namespace marble_probability_l314_314342

theorem marble_probability :
  let red := 4
      blue := 3
      yellow := 6
      green := 2
      total_marbles := red + blue + yellow + green
      favorable_marbles := red + blue + green
  in 
  (favorable_marbles / total_marbles : ℚ) = 3 / 5 :=
by
  sorry

end marble_probability_l314_314342


namespace construct_center_of_circle_l314_314827

open EuclideanGeometry

/--
Given a circle and two parallel chords AB and CD, prove that the line MN, where M and N are intersections of lines AD with BC and AC with BD respectively, passes through the center of the circle, forming a diameter.
--/
theorem construct_center_of_circle (circle : EuclideanGeometry.Circle) (A B C D : EuclideanGeometry.Point)
  (hAB : EuclideanGeometry.Chord circle A B) (hCD : EuclideanGeometry.Chord circle C D) 
  (h_parallel : EuclideanGeometry.Parallel_AB_CD) -- Line AB is parallel to line CD
  (rule_width : EuclideanGeometry.Ruler.width < EuclideanGeometry.Circle.diameter circle) : 
  ∃ M N : EuclideanGeometry.Point, 
  (EuclideanGeometry.Intersection (EuclideanGeometry.Line through A D) (EuclideanGeometry.Line through B C) = M ∧
   EuclideanGeometry.Intersection (EuclideanGeometry.Line through A C) (EuclideanGeometry.Line through B D) = N ∧
   EuclideanGeometry.Line through M N is_diameter_of circle) :=
sorry

end construct_center_of_circle_l314_314827


namespace sum_of_a_and_b_l314_314876

theorem sum_of_a_and_b (f : ℝ → ℝ) (a b : ℝ) 
  (h₁ : f = λ x, x^3 + 3 * x^2 + 6 * x) 
  (h₂ : f a = 1) 
  (h₃ : f b = -9) : 
  a + b = -2 :=
by 
  sorry

end sum_of_a_and_b_l314_314876


namespace min_ω_is_3_over_4_l314_314515

noncomputable def shifted_sine_minimum_omega (ω : ℝ) : Prop :=
  ω > 0 →
  ∃ g : ℝ → ℝ, 
    g = λ x, 2 * Real.sin (ω * (x + π / 6) + π / 3) ∧
    g (π / 2) = 1 →
    ω = 3 / 4

theorem min_ω_is_3_over_4 : shifted_sine_minimum_omega (3 / 4) :=
begin
  sorry
end

end min_ω_is_3_over_4_l314_314515


namespace find_original_price_l314_314413

theorem find_original_price (a b x : ℝ) (h : x * (1 - 0.1) - a = b) : 
  x = (a + b) / (1 - 0.1) :=
sorry

end find_original_price_l314_314413


namespace largest_sum_at_vertex_l314_314825

/--
Consider a newly designed cubic die where the sum of numbers on opposite faces is always 9.
The cube faces are numbered from 1 to 6, and the fold lines of the cube's net form a cross.
Prove that the largest sum of three numbers whose faces come together at one vertex is 12.
-/
theorem largest_sum_at_vertex : 
  let faces := [1, 2, 3, 4, 5, 6],
      opp_pairs := [(1, 8), (2, 7), (3, 6), (4, 5)],
      valid_pairs := opp_pairs.filter (λ p, p.1 ∈ faces ∧ p.2 ∈ faces) in
  ∃ (a b c : ℕ), 
  a ∈ faces ∧ b ∈ faces ∧ c ∈ faces ∧ 
  (a, b) ∉ valid_pairs ∧ (a, c) ∉ valid_pairs ∧ (b, c) ∉ valid_pairs ∧ 
  a + b + c = 12 :=
begin
  sorry
end

end largest_sum_at_vertex_l314_314825


namespace polygon_side_sum_l314_314628

theorem polygon_side_sum 
  (area_polygon : ℝ)
  (AB BC HA : ℝ)
  (DE EF FG GH : ℝ)
  (h1 : area_polygon = 85)
  (h2 : AB = 7)
  (h3 : BC = 10)
  (h4 : HA = 6)
  (h5 : DE = EF)
  (h6 : EF = FG)
  (h7 : FG = GH)
  : DE + EF + FG + GH = 2.5 := 
begin
  sorry
end

end polygon_side_sum_l314_314628


namespace count_incorrect_expressions_l314_314417

open Set

theorem count_incorrect_expressions :
  (¬ {0} ∈ ({1, 2} : Set (Set ℕ))) ∧
  (∅ ⊆ ({0} : Set ℕ)) ∧
  ({0, 1, 2} ⊆ ({1, 2, 0} : Set ℕ)) ∧
  (¬ 0 ∈ (∅ : Set ℕ)) ∧
  (¬ (0 ∩ (∅ : set ℕ) = ∅)) →
  3 = 3 :=
by
  sorry

end count_incorrect_expressions_l314_314417


namespace find_fx_at_pi3_l314_314005

theorem find_fx_at_pi3 (φ : ℝ) (h0 : 0 < φ) (h1 : φ < π / 2)
  (h2 : 3 * sin (4 * (π / 12) + φ) = 3) :
  3 * sin (4 * (π / 3) + φ) = -3 := by
  sorry

end find_fx_at_pi3_l314_314005


namespace part1_solution_set_part2_range_a_l314_314945

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314945


namespace sector_arc_length_l314_314503

theorem sector_arc_length (A θ r l : ℝ) (hA : A = (1 / 2) * θ * r^2) (hA_value : A = 9) (θ_value : θ = 2) : l = r * θ := 
by
  -- Given the area is 9 and using the formula for the area of a sector, solve for r.
  have h1 : (1 / 2) * 2 * r^2 = 9 := by 
    rw [θ_value, hA_value]
    exact hA

  -- From the equation, we get r^2 = 9, hence r = 3 (only positive radius is considered)
  have h2 : r = 3 := by
    sorry

  -- Now use the formula for arc length
  have h3 : l = r * θ := by
    rw [θ_value, h2]
    have r := 3
    exact 3 * 2

  exact h3
sorry
-- Note: This file assumes the proof assistant will be able to close the trivial goals indicated as sorry. 

end sector_arc_length_l314_314503


namespace ratio_out_of_school_friends_to_classmates_l314_314079

variable (F : ℕ) (classmates : ℕ := 20) (parents : ℕ := 2) (sister : ℕ := 1) (total : ℕ := 33)

theorem ratio_out_of_school_friends_to_classmates (h : classmates + F + parents + sister = total) :
  (F : ℚ) / classmates = 1 / 2 := by
    -- sorry allows this to build even if proof is not provided
    sorry

end ratio_out_of_school_friends_to_classmates_l314_314079


namespace intersection_of_sets_l314_314593

variable (A B : Set ℝ) (x : ℝ)

def setA : Set ℝ := { x | x > 0 }
def setB : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_of_sets_l314_314593


namespace part1_solution_set_part2_range_a_l314_314942

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314942


namespace find_n_eq_130_l314_314130

theorem find_n_eq_130 
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : 0 < n)
  (h2 : d1 < d2)
  (h3 : d2 < d3)
  (h4 : d3 < d4)
  (h5 : ∀ d, d ∣ n → d = d1 ∨ d = d2 ∨ d = d3 ∨ d = d4 ∨ d ∣ n → ¬(1 < d ∧ d < d1))
  (h6 : n = d1^2 + d2^2 + d3^2 + d4^2) : n = 130 := 
  sorry

end find_n_eq_130_l314_314130


namespace dice_product_probability_l314_314746

def is_valid_die_value (n : ℕ) : Prop := n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

theorem dice_product_probability :
  ∃ (a b c : ℕ), is_valid_die_value a ∧ is_valid_die_value b ∧ is_valid_die_value c ∧ 
  a * b * c = 8 ∧ 
  (1 / 6 : ℝ) * (1 / 6) * (1 / 6) * (6 + 1) = (7 / 216 : ℝ) :=
sorry

end dice_product_probability_l314_314746


namespace courtiers_selection_l314_314669

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l314_314669


namespace part1_solution_set_part2_range_a_l314_314944

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314944


namespace ratio_areas_l314_314066

theorem ratio_areas (AB AC AD CF : ℝ) (H1 : AB = 115) (H2 : AC = 115) (H3 : AD = 45) (H4 : CF = 92) :
  ∃ [CEF DBE : ℝ], (CEF / DBE = 207 / 45) := 
sorry

end ratio_areas_l314_314066


namespace angle_triple_complement_l314_314289

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314289


namespace contrapositive_equiv_l314_314351

variable {α : Type}  -- Type of elements
variable (P : Set α) (a b : α)

theorem contrapositive_equiv (h : a ∈ P → b ∉ P) : b ∈ P → a ∉ P :=
by
  sorry

end contrapositive_equiv_l314_314351


namespace coplanar_vectors_l314_314416

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

def coplanar (A B C D : V) : Prop :=
∃ (α β γ : ℝ), (α + β + γ = 1) ∧ (α • A + β • B + γ • C = D)

theorem coplanar_vectors
  {O M A B C : V}
  (h : (A - M) + (B - M) + (C - M) = 0) :
  coplanar M A B C :=
begin
  -- proof omitted
  sorry
end

end coplanar_vectors_l314_314416


namespace find_p_q_l314_314037

-- Define the polynomial and its conditions
def polynomial (x p q : ℝ) : ℝ :=
  x^5 - x^4 + 2*x^3 - p*x^2 + q*x - 5

-- Main theorem statement
theorem find_p_q : 
  (∃ p q : ℝ, (polynomial (-1) p q = 0) ∧ (polynomial 2 p q = 0) ∧ set.prod_eq (p, q) (1.5, -10.5)) :=
sorry

end find_p_q_l314_314037


namespace find_other_number_l314_314626

theorem find_other_number (HCF LCM a b : ℕ) (h1 : HCF = 108) (h2 : LCM = 27720) (h3 : a = 216) (h4 : HCF * LCM = a * b) : b = 64 :=
  sorry

end find_other_number_l314_314626


namespace arithmetic_sequence_fifth_term_l314_314636

noncomputable def fifth_term (x y : ℝ) : ℝ :=
  let a1 := x^2 + y^2
  let a2 := x^2 - y^2
  let a3 := x^2 * y^2
  let a4 := x^2 / y^2
  let d := -2 * y^2
  a4 + d

theorem arithmetic_sequence_fifth_term (x y : ℝ) (hy : y ≠ 0) (hx2 : x ^ 2 = 3 * y ^ 2 / (y ^ 2 - 1)) :
  fifth_term x y = 3 / (y ^ 2 - 1) - 2 * y ^ 2 :=
by
  sorry

end arithmetic_sequence_fifth_term_l314_314636


namespace waiting_room_larger_than_interview_room_l314_314176

def number_of_people_in_waiting_room (people_in_waiting_room people_arrive : Nat) : Nat :=
  people_in_waiting_room + people_arrive

def feasible (I n : Nat) : Prop :=
  if I > 1 then (25 : Nat) = n * I else False

def times_larger (people_in_waiting_room' I : Nat) : Nat :=
  people_in_waiting_room' / I

theorem waiting_room_larger_than_interview_room (I n : Nat) :
  let people_in_waiting_room_initial := 22
  let people_arrive := 3
  let people_in_waiting_room_new := number_of_people_in_waiting_room people_in_waiting_room_initial people_arrive in
  feasible I n →
  people_in_waiting_room_new = n * I →
  n = 5 :=
by
  intros
  sorry

end waiting_room_larger_than_interview_room_l314_314176


namespace remainder_of_n_mod_1000_l314_314091

-- Definition of the set S
def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 15 }

-- Define the number of sets of three non-empty disjoint subsets of S
def num_sets_of_three_non_empty_disjoint_subsets (S : Set ℕ) : ℕ :=
  let total_partitions := 4^15
  let single_empty_partition := 3 * 3^15
  let double_empty_partition := 3 * 2^15
  let all_empty_partition := 1
  total_partitions - single_empty_partition + double_empty_partition - all_empty_partition

-- Compute the result of the number modulo 1000
def result := (num_sets_of_three_non_empty_disjoint_subsets S) % 1000

-- Theorem that states the remainder when n is divided by 1000
theorem remainder_of_n_mod_1000 : result = 406 := by
  sorry

end remainder_of_n_mod_1000_l314_314091


namespace euler_totient_17_euler_totient_prime_euler_totient_prime_sq_euler_totient_prime_power_l314_314141

def euler_totient (n : ℕ) : ℕ := 
  Nat.card {m : ℕ // m ≤ n ∧ Nat.coprime n m}
  
theorem euler_totient_17 : euler_totient 17 = 16 := 
  sorry

theorem euler_totient_prime (p : ℕ) (hp : Nat.Prime p) : euler_totient p = p - 1 := 
  sorry

theorem euler_totient_prime_sq (p : ℕ) (hp : Nat.Prime p) : euler_totient (p^2) = p * (p - 1) := 
  sorry

theorem euler_totient_prime_power (p α : ℕ) (hp : Nat.Prime p) (hα : α ≥ 1) : euler_totient (p^α) = p^(α-1) * (p - 1) := 
  sorry

end euler_totient_17_euler_totient_prime_euler_totient_prime_sq_euler_totient_prime_power_l314_314141


namespace find_alpha_l314_314496

theorem find_alpha (α : ℝ) (a : ℕ → ℝ)
    (geom_seq : ∀ n, a (n + 1) = a n * a 1)
    (roots_quadratic : (a 1)^2 - 2 * (a 1) * sin α - sqrt 3 * sin α = 0 ∧ 
                       (a 8)^2 - 2 * (a 8) * sin α - sqrt 3 * sin α = 0 )
    (eq_condition : (a 1 + a 8)^2 = 2 * a 3 * a 6 + 6) :
    α = π / 3 :=
begin
    sorry
end

end find_alpha_l314_314496


namespace parabola_equation_range_b_l314_314904

-- Define the conditions of the problem
def PointA : ℝ × ℝ := (-4, 0)

def SlopeL : ℚ := 1/2

def ParabolaG (p : ℝ) : Prop := ∀ (x y : ℝ), x^2 = 2 * p * y

def VectorRelation (A B C : ℝ × ℝ) : Prop :=
  let ACx := (C.1 - A.1)
  let ACy := (C.2 - A.2)
  let ABx := (B.1 - A.1)
  let ABy := (B.2 - A.2)
  ACx = (1 / 4) * ABx ∧ ACy = (1 / 4) * ABy

-- Define the proof goals
theorem parabola_equation (p : ℝ) (h_pos : p > 0)
  (h_line : ∃ (line : ℝ × ℝ → ℝ), line (PointA) = SlopeL)
  (h_intersect : ∃ (B C : ℝ × ℝ), ∀ (x y : ℝ), ParabolaG p x y → VectorRelation PointA B C) :
  (∃ (x y : ℝ), x^2 = 4 * y) :=
by
  sorry

theorem range_b (k : ℝ) (b : ℝ)
  (h_slope : k ≠ 0)
  (h_quad : ∃ (x1 x2 : ℝ), x1^2 - 4 * k * x1 - 16 * k = 0 ∧ x2^2 - 4 * k * x2 - 16 * k = 0)
  (h_intercept : b = 2 * (k + 1)^2) :
  (b ∈ set.Ioi (2 : ℝ)) :=
by
  sorry

end parabola_equation_range_b_l314_314904


namespace find_b_l314_314902

noncomputable def h (x : ℝ) : ℝ := x^2 + 9
noncomputable def j (x : ℝ) : ℝ := x^2 + 1

theorem find_b (b : ℝ) (hjb : h (j b) = 15) (b_pos : b > 0) : b = Real.sqrt (Real.sqrt 6 - 1) := by
  sorry

end find_b_l314_314902


namespace sin_cos_product_l314_314029

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l314_314029


namespace angle_triple_complement_l314_314201

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314201


namespace quoted_price_of_shares_l314_314782

theorem quoted_price_of_shares :
  ∀ (investment nominal_value dividend_rate annual_income quoted_price : ℝ),
  investment = 4940 →
  nominal_value = 10 →
  dividend_rate = 14 →
  annual_income = 728 →
  quoted_price = 9.5 :=
by
  intros investment nominal_value dividend_rate annual_income quoted_price
  intros h_investment h_nominal_value h_dividend_rate h_annual_income
  sorry

end quoted_price_of_shares_l314_314782


namespace pasture_feeding_l314_314703

-- The definitions corresponding to the given conditions
def portion_per_cow_per_day := 1

def food_needed (cows : ℕ) (days : ℕ) : ℕ := cows * days

def growth_rate (food10for20 : ℕ) (food15for10 : ℕ) (days10_20 : ℕ) : ℕ :=
  (food10for20 - food15for10) / days10_20

def food_growth_rate := growth_rate (food_needed 10 20) (food_needed 15 10) 10

def new_grass_feed_cows_per_day := food_growth_rate / portion_per_cow_per_day

def original_grass := (food_needed 10 20) - (food_growth_rate * 20)

def days_to_feed_30_cows := original_grass / (30 - new_grass_feed_cows_per_day)

-- The statement we want to prove
theorem pasture_feeding :
  new_grass_feed_cows_per_day = 5 ∧ days_to_feed_30_cows = 4 := by
  sorry

end pasture_feeding_l314_314703


namespace equal_angles_ANB_and_BNC_l314_314567

open EuclideanGeometry

noncomputable def cyclic_quadrilateral (A B C D : Point) : Prop :=
is_cyclic (A, B, C, D)

variables {A B C D M N : Point}
variables (hABC : cyclic_quadrilateral A B C D)
variables (hMidM : midpoint M A C)
variables (hMidN : midpoint N B D)
variables (hAngleEq : ∠ A M B = ∠ A M D)

theorem equal_angles_ANB_and_BNC :
  ∠ A N B = ∠ B N C :=
by
  sorry

end equal_angles_ANB_and_BNC_l314_314567


namespace same_selection_exists_l314_314646

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l314_314646


namespace proof_of_greatest_sum_quotient_remainder_l314_314157

def greatest_sum_quotient_remainder : Prop :=
  ∃ q r : ℕ, 1051 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q + r = 61

theorem proof_of_greatest_sum_quotient_remainder : greatest_sum_quotient_remainder := 
sorry

end proof_of_greatest_sum_quotient_remainder_l314_314157


namespace min_value_of_f_l314_314577

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log 2 (x^2 + 2*x + a)

theorem min_value_of_f (a : ℝ) (h1 : 1 < a) (h2 : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x a ≤ 5) : 
  ∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ f x a = 4 :=
by
    sorry

end min_value_of_f_l314_314577


namespace termites_ate_12_black_squares_l314_314140

def is_black_square (r c : ℕ) : Prop :=
  (r + c) % 2 = 0

def eaten_positions : List (ℕ × ℕ) :=
  [(3, 1), (4, 6), (3, 7), (4, 1), (2, 3), (2, 4), (4, 3), (3, 5), (3, 2), (4, 7), (3, 6), (2, 6)]

def count_black_squares (positions : List (ℕ × ℕ)) : ℕ :=
  positions.countp (λ ⟨r, c⟩ => is_black_square r c)

theorem termites_ate_12_black_squares : count_black_squares eaten_positions = 12 := by
  sorry

end termites_ate_12_black_squares_l314_314140


namespace triangle_incenter_length_l314_314068

open EuclideanGeometry

noncomputable def incenter (P Q R : Point) : Point := sorry
noncomputable def inradius (P Q R : Point) : ℝ := sorry

theorem triangle_incenter_length (P Q R J G H I : Point)
  (hPQ : dist P Q = 12) (hPR : dist P R = 13) (hQR : dist Q R = 15)
  (hI : incenter P Q R = J) 
  (hG : dist J G = inradius P Q R) (hH : dist J H = inradius P Q R) (hI' : dist J I = inradius P Q R) :
  dist P J = 3 * sqrt 7 := 
sorry

end triangle_incenter_length_l314_314068


namespace outfits_count_l314_314750

theorem outfits_count (shirts : ℕ) (pants : ℕ) (hats : ℕ) (h_shirts : shirts = 4) (h_pants : pants = 5) (h_hats : hats = 3) :
  shirts * pants * hats = 60 :=
by
  rw [h_shirts, h_pants, h_hats]
  norm_num
  sorry

end outfits_count_l314_314750


namespace exist_two_courtiers_with_same_selection_l314_314691

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l314_314691


namespace find_S2017_l314_314898

-- Setting up the given conditions and sequences
def a1 : ℤ := -2014
def S (n : ℕ) : ℚ := n * a1 + (n * (n - 1) / 2) * 2 -- Using the provided sum formula

theorem find_S2017
  (h1 : a1 = -2014)
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) :
  S 2017 = 4034 := 
sorry

end find_S2017_l314_314898


namespace find_ED_l314_314179

def Parallelogram (A B C D : Type) : Prop := sorry

variables (A B C D E F G : Type) [Parallelogram A B C D]

variables (FG FE BE ED : ℝ)

-- Given conditions
axiom FG_FE_ratio : FG / FE = 7
axiom BE_value : BE = 8

-- Proposition to prove
theorem find_ED (h : E ∈ Segment BD) (h1 : F ∈ Segment CD) (h2 : G ∈ (Line A ∩ Line B ∩ Line C))
: ED = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_ED_l314_314179


namespace lcm_24_36_45_l314_314338

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l314_314338


namespace find_b_minus_a_l314_314640

theorem find_b_minus_a : 
  let circle1 := (x - 6)^2 + (y - 3)^2 = 49,
      circle2 := (x - 2)^2 + (y - 6)^2 = 40 + k,
      d := real.sqrt ((6 - 2)^2 + (3 - 6)^2),
      R1 := 7,
      R2 := real.sqrt (40 + k),
      k_bounds := -36 ≤ k ∧ k ≤ 104 in
  (max k_bounds - min k_bounds) = 140 :=
by
  sorry

end find_b_minus_a_l314_314640


namespace angle_is_67_l314_314211

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314211


namespace pyramid_correct_fills_l314_314559

-- Define a function that represents the pyramid conditions
def pyramid_top_is_plus (a b c d : ℤ) : Prop :=
  (a + b + c + d) % 2 = 1

-- Define a function that counts the number of ways to fill the bottom row to achieve pyramid_top_is_plus
noncomputable def count_valid_pyramid_fills : ℕ :=
  (list.filter (λ (abcd : list ℤ), 
    let [a, b, c, d] := abcd in pyramid_top_is_plus a b c d) 
  (list.foldr (λ x r, list.append (list.map (cons x) r) r) [[]] [0, 1, 0, 1, 0, 1, 0, 1])).length

-- The proof problem statement
theorem pyramid_correct_fills : count_valid_pyramid_fills = 8 :=
  sorry

end pyramid_correct_fills_l314_314559


namespace part1_part2_l314_314930

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314930


namespace angle_triple_complement_l314_314261

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314261


namespace part1_solution_part2_solution_l314_314966

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314966


namespace margo_total_distance_l314_314596

-- Definitions for conditions
def rate_to_friend : ℝ := 4
def time_to_friend : ℝ := 30 / 60
def distance_to_friend : ℝ := rate_to_friend * time_to_friend

def rate_back_home : ℝ := 3
def time_back_home : ℝ := 40 / 60
def distance_back_home : ℝ := rate_back_home * time_back_home

-- The theorem stating the total distance covered
theorem margo_total_distance : distance_to_friend + distance_back_home = 4 := by
  sorry

end margo_total_distance_l314_314596


namespace courtiers_dog_selection_l314_314652

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l314_314652


namespace find_x_for_perpendicular_l314_314521

theorem find_x_for_perpendicular
  (a b c : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (1, 0))
  (h3 : c = (3, 4))
  (h4 : ∀ x, let v := (b.1 + x * a.1, b.2 + x * a.2) in v.1 * c.1 + v.2 * c.2 = 0) :
  x = -3 / 11 :=
by
  sorry

end find_x_for_perpendicular_l314_314521


namespace angle_triple_complement_l314_314316

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314316


namespace angle_triple_complement_l314_314322

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314322


namespace number_of_insects_l314_314815

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h : total_legs = 54) (k : legs_per_insect = 6) :
  total_legs / legs_per_insect = 9 := by
  sorry

end number_of_insects_l314_314815


namespace perpendicular_vectors_x_value_l314_314520

def vector := (ℝ × ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def vector_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : vector := (-1, -2, 1)
  let b : vector := (2, x, 3)
  in dot_product a (vector_add a b) = 0 -> x = 7 / 2 :=
by
  sorry

end perpendicular_vectors_x_value_l314_314520


namespace exist_two_courtiers_with_same_selection_l314_314697

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l314_314697


namespace coin_ordering_expected_weighings_lt_4_8_l314_314715

theorem coin_ordering_expected_weighings_lt_4_8
  (A B C D : ℕ)
  (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧
            B ≠ C ∧ B ≠ D ∧
            C ≠ D)
  (weigh : ℕ → ℕ → bool) :
  ∃ ordering : list ℕ, expected_weighings ordering weigh < 4.8 :=
by
  sorry

end coin_ordering_expected_weighings_lt_4_8_l314_314715


namespace largest_number_with_unique_digits_summing_to_18_is_843210_l314_314328

-- Define the problem with conditions and answer
theorem largest_number_with_unique_digits_summing_to_18_is_843210 :
  ∃ n : ℕ, (∀ d ∈ (int_to_digits n), unique d) ∧ digit_sum n = 18 ∧ is_largest n :=
sorry

-- Definitions used in the theorem statement

-- Helper to convert an integer to a list of its digits
def int_to_digits (n : ℕ) : list ℕ := -- convert integer to list of digits
sorry

-- Helper to check that all elements in the list are unique
def unique (l : list ℕ) : Prop := -- check that all digits are unique
sorry

-- Helper to sum digits of a number
def digit_sum (n : ℕ) : ℕ := (int_to_digits n).sum
  
-- Helper to assert that the number is the largest possible
def is_largest (n : ℕ) : Prop := -- check if the number is the largest possible under given conditions
sorry

end largest_number_with_unique_digits_summing_to_18_is_843210_l314_314328


namespace sum_of_gcd_and_lcm_of_180_and_4620_l314_314343

def gcd_180_4620 : ℕ := Nat.gcd 180 4620
def lcm_180_4620 : ℕ := Nat.lcm 180 4620
def sum_gcd_lcm_180_4620 : ℕ := gcd_180_4620 + lcm_180_4620

theorem sum_of_gcd_and_lcm_of_180_and_4620 :
  sum_gcd_lcm_180_4620 = 13920 :=
by
  sorry

end sum_of_gcd_and_lcm_of_180_and_4620_l314_314343


namespace batsman_average_after_12_l314_314355

variable (runs12 avg11 avg12 : ℕ)
variable (increased_by : ℕ)

-- Conditions defined
def runs12_value := 60
def increased_by_value := 4
def avg12_value := avg11 + increased_by

-- Theorem statement
theorem batsman_average_after_12 (
  h1 : runs12 = runs12_value
  h2 : increased_by = increased_by_value
  h3 : avg12 = avg12_value
) : avg12 = 16 := by
  sorry

end batsman_average_after_12_l314_314355


namespace significant_figures_88_times_10_pow_3_l314_314143

theorem significant_figures_88_times_10_pow_3 :
  (∃ (x : ℕ), x = 8.8 * 10^3) → 
  ("Accurate to the hundreds place, there are 2 significant figures." ≠ "" ∧ 
   "Accurate to the tenths place, there are 2 significant figures." = "" ∧
   "Accurate to the ones place, there are 2 significant figures." = "" ∧
   "Accurate to the thousands place, there are 4 significant figures." = "") :=
sorry

end significant_figures_88_times_10_pow_3_l314_314143


namespace ratio_invariant_l314_314493

variables {P A B O : Type*} [EuclideanGeometry P]
variables 
  (a b l : Line P)
  (O : P) 
  (P : P)
  (A : P) 
  (B : P)

-- Assumptions
variable (ha : a.contains O)
variable (hb : b.contains O)
variable (hp : P ≠ O)
variable (hA : a.contains A)
variable (hB : b.contains B)
variable (hl : l.contains P)
variable (hPA : l.contains A)
variable (hPB : l.contains B)

theorem ratio_invariant 
  (hAO : distance O A ≠ 0)
  (hOB : distance O B ≠ 0)
  (hPA : distance P A ≠ 0)
  (hPB : distance P B ≠ 0)
  : 
  (distance O A / distance O B) / (distance P A / distance P B) = 
  (distance O A / distance O B) / (distance P A / distance P B) := 
sorry

end ratio_invariant_l314_314493


namespace a_eq_one_sufficient_not_necessary_P_subset_M_iff_l314_314591

open Set

-- Define sets P and M based on conditions
def P : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem a_eq_one_sufficient_not_necessary (a : ℝ) : (a = 1) → (P ⊆ M a) := 
by
  sorry

theorem P_subset_M_iff (a : ℝ) : (P ⊆ M a) ↔ (a < 2) :=
by
  sorry

end a_eq_one_sufficient_not_necessary_P_subset_M_iff_l314_314591


namespace angle_triple_complement_l314_314243

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314243


namespace courtiers_dog_selection_l314_314655

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l314_314655


namespace angle_triple_complement_l314_314247

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314247


namespace derivative_at_pi_over_six_l314_314002

theorem derivative_at_pi_over_six : 
  (derivative (λ x : ℝ, Real.sin (2 * x)) (π / 6)) = 1 :=
by
  sorry

end derivative_at_pi_over_six_l314_314002


namespace number_of_valid_functions_l314_314469

noncomputable def count_functions : ℕ := by
  let a_vals := [-1, 0]
  let b_vals := [-1, 0]
  let c_vals := [0, 1]
  let d_vals := [0, 1]
  let valid_functions := 
    { f : ℝ → ℝ // ∃ (a ∈ a_vals) (b ∈ b_vals) (c ∈ c_vals) (d ∈ d_vals), 
      f = λ x, a * x^3 + b * x^2 + c * x + d ∧ 
      ∀ x, (f x) * (f (-x)) = a * x^9 + b * x^6 + c * x^3 + d }
  exact Set.card valid_functions

theorem number_of_valid_functions : count_functions = 16 := by
  sorry

end number_of_valid_functions_l314_314469


namespace toot_has_vertical_symmetry_l314_314353

def has_vertical_symmetry (letter : Char) : Prop :=
  letter = 'T' ∨ letter = 'O'

def word_has_vertical_symmetry (word : List Char) : Prop :=
  ∀ letter ∈ word, has_vertical_symmetry letter

theorem toot_has_vertical_symmetry : word_has_vertical_symmetry ['T', 'O', 'O', 'T'] :=
  by
    sorry

end toot_has_vertical_symmetry_l314_314353


namespace unused_types_l314_314808

theorem unused_types (total_resources : ℕ) (used_types : ℕ) (valid_types : ℕ) :
  total_resources = 6 → used_types = 23 → valid_types = 2^total_resources - 1 - used_types → valid_types = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end unused_types_l314_314808


namespace part1_part2_l314_314931

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314931


namespace expression_evaluation_l314_314433

theorem expression_evaluation : 
  (sqrt 5 * (-sqrt 10) - (1 / 7)⁻¹ + | -2^3 |) = -5 * sqrt 2 + 1 :=
by sorry

end expression_evaluation_l314_314433


namespace min_total_fund_Required_l314_314722

noncomputable def sell_price_A (x : ℕ) : ℕ := x + 10
noncomputable def cost_A (x : ℕ) : ℕ := 600
noncomputable def cost_B (x : ℕ) : ℕ := 400

def num_barrels_A_B_purchased (x : ℕ) := cost_A x / (sell_price_A x) = cost_B x / x

noncomputable def total_cost (m : ℕ) : ℕ := 10 * m + 10000

theorem min_total_fund_Required (price_A price_B m total : ℕ) :
  price_B = 20 →
  price_A = 30 →
  price_A = price_B + 10 →
  (num_barrels_A_B_purchased price_B) →
  total = total_cost m →
  m = 250 →
  total = 12500 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end min_total_fund_Required_l314_314722


namespace max_quartets_in_5x5_max_quartets_in_mxn_l314_314400

def quartet (c : Nat) : Bool := 
  c > 0

theorem max_quartets_in_5x5 : ∃ q, q = 5 ∧ 
  quartet q := by
  sorry

theorem max_quartets_in_mxn 
  (m n : Nat) (Hmn : m > 0 ∧ n > 0) :
  (∃ q, q = (m * (n - 1)) / 4 ∧ quartet q) ∨ 
  (∃ q, q = (m * (n - 1) - 2) / 4 ∧ quartet q) := by
  sorry

end max_quartets_in_5x5_max_quartets_in_mxn_l314_314400


namespace constant_area_value_l314_314837

theorem constant_area_value (a b c d : ℝ) (h_perimeter : a + b = c + d) :
  ∃ n : ℝ, ∀ A₀ A₁, 
  let area_a := a * b + 2 * (a^2 + b^2) / n,
      area_c := c * d + 2 * (c^2 + d^2) / n
  in area_a = A₀ ∧ area_c = A₁ → A₀ = A₁ → n = 4 := sorry

end constant_area_value_l314_314837


namespace proposition_equivalence_l314_314349

open Classical

variable {α : Type*}
variables (P : Set α)

theorem proposition_equivalence:
  (∀ a b, a ∈ P → b ∉ P) ↔ (∀ a b, b ∈ P → a ∉ P) :=
by intros a b; sorry

end proposition_equivalence_l314_314349


namespace cos_555_value_l314_314447

noncomputable def cos_555_equals_neg_sqrt6_add_sqrt2_div4 : Prop :=
  (Real.cos 555 = -((Real.sqrt 6 + Real.sqrt 2) / 4))

theorem cos_555_value : cos_555_equals_neg_sqrt6_add_sqrt2_div4 :=
  by sorry

end cos_555_value_l314_314447


namespace expected_value_two_flips_l314_314794

def coin_flip_expected_value : ℝ :=
  (1 / 4) * 4 + (3 / 4) * -3

theorem expected_value_two_flips : coin_flip_expected_value * 2 = -2.5 := by
  sorry

end expected_value_two_flips_l314_314794


namespace friends_count_l314_314407

noncomputable def university_students := 1995

theorem friends_count (students : ℕ)
  (knows_each_other : (ℕ → ℕ → Prop))
  (acquaintances : ℕ → ℕ)
  (h_university_students : students = university_students)
  (h_knows_iff_same_acq : ∀ a b, knows_each_other a b ↔ acquaintances a = acquaintances b)
  (h_not_knows_iff_diff_acq : ∀ a b, ¬ knows_each_other a b ↔ acquaintances a ≠ acquaintances b) :
  ∃ a, acquaintances a ≥ 62 ∧ ¬ ∃ a, acquaintances a ≥ 63 :=
sorry

end friends_count_l314_314407


namespace same_selection_exists_l314_314643

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l314_314643


namespace sequence_formula_sum_formula_l314_314886

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 0 else n + 1

theorem sequence_formula (n : ℕ) (hn : n ≠ 0) : 
  (a n) = n + 1 :=
by
  sorry

noncomputable def b (n : ℕ) : ℕ :=
  2 * (a n + 1 / 2 ^ (a n))

noncomputable def S (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, b i

theorem sum_formula (n : ℕ) :
  S n = n^2 + 3 * n + 1 - 1 / 2^n :=
by
  sorry

end sequence_formula_sum_formula_l314_314886


namespace magnitude_of_sum_l314_314502

variables (a b : ℝ^2)
variables (θ : ℝ)
variables (cosθ : ℝ)
variables (magnitude_a magnitude_b : ℝ)

-- Given conditions
axiom angle_ab : θ = real.pi / 3
axiom mag_a : ‖a‖ = 2
axiom mag_b : ‖b‖ = 1
axiom cos_ab : cosθ = real.cos (real.pi / 3)
axiom cos_val : cosθ = 1 / 2

theorem magnitude_of_sum : ‖a + 2 • b‖ = 2 * real.sqrt 3 :=
sorry

end magnitude_of_sum_l314_314502


namespace shipping_cost_correct_l314_314764

-- Define the condition of the shipping fee calculation
def shipping_fee (G : ℕ) : ℕ :=
  8 * Int.ceil (G.toRat / 100)

-- Theorem stating the cost of shipping is as described
theorem shipping_cost_correct (G : ℕ) : 
  ∃ cost : ℕ, cost = shipping_fee G :=
by
  exists 8 * Int.ceil (G.toRat / 100)
  refl

end shipping_cost_correct_l314_314764


namespace angle_measure_triple_complement_l314_314232

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314232


namespace locus_equilateral_triangle_vertex_l314_314396

theorem locus_equilateral_triangle_vertex 
  (a : ℝ) :
  (∃ x y : ℝ, (√3) * x - y + a = 0) ∧ (∃ x y : ℝ, (√3) * x + y - a = 0) := 
by {
 sorry
}

end locus_equilateral_triangle_vertex_l314_314396


namespace sin_1200_eq_sin_60_l314_314835

-- Defining the value of sin at 60 degrees
def sin_60_eq_sqrt3_over_2 : Real :=
  Real.sqrt 3 / 2

theorem sin_1200_eq_sin_60 :
  sin 1200 = sin_60_eq_sqrt3_over_2 :=
by
  sorry

end sin_1200_eq_sin_60_l314_314835


namespace K_is_regular_l314_314571

noncomputable def is_regular_polygon (P : Set Point) : Prop := sorry
noncomputable def K (M N : Set Point) : Set Point := 
  {p | ∃ A B : Point, A ∈ M ∧ B ∈ N ∧ p = midpoint A B}
noncomputable def are_homothetic (M N : Set Point) : Prop := sorry
noncomputable def are_rotated_translated (M N : Set Point) (m : ℕ) : Prop := sorry

theorem K_is_regular (M N : Set Point) (m : ℕ) (hM : is_regular_polygon M) (hN : is_regular_polygon N) :
  is_regular_polygon (K M N) ↔ are_homothetic M N ∨ are_rotated_translated M N m := sorry

end K_is_regular_l314_314571


namespace compute_expression_equals_375_l314_314824

theorem compute_expression_equals_375 : 15 * (30 / 6) ^ 2 = 375 := 
by 
  have frac_simplified : 30 / 6 = 5 := by sorry
  have power_calculated : 5 ^ 2 = 25 := by sorry
  have final_result : 15 * 25 = 375 := by sorry
  sorry

end compute_expression_equals_375_l314_314824


namespace clock_angle_at_3_40_l314_314427

theorem clock_angle_at_3_40 : 
    let h := 3
    let m := 40
    let angle := |(60 * h - 11 * m) / 2|
    angle = 130 := 
by
  let h := 3
  let m := 40
  have angle_def : angle = |(60 * h - 11 * m) / 2| := rfl
  calc
    angle = |(60 * h - 11 * m) / 2| : rfl
    ... = |(60 * 3 - 11 * 40) / 2| : rfl
    ... = |180 - 440| / 2 : by norm_num
    ... = |-260| / 2 : rfl
    ... = 260 / 2 : by norm_num
    ... = 130 : by norm_num

end clock_angle_at_3_40_l314_314427


namespace concyclic_points_l314_314188

-- Definitions
variables {A B C D O1 O2 O3 O4 : Type}
variables [MetricSpace O1] [MetricSpace O2] [MetricSpace O3] [MetricSpace O4]
variables {R1 R2 R3 R4 : ℝ}
variables (a b c d : ℝ)
variables (collinear : ∀ {A B C : point_space}, collinear A B C)

-- Conditions
def tangent_perpendicular (A : point_space) (B : point_space) (C O1 O2 O3 O4 : point_space) := 
  collinear O1 A O2 ∧ collinear O2 B O3 ∧ collinear O3 C O4 ∧ collinear O4 D O1

-- Proof problem statement
theorem concyclic_points (A B C D : point_space) 
  (O1 O2 O3 O4 : point_space)
  (R1 R2 R3 R4 : ℝ)
  (h1 : tangent_perpendicular A B C O1 O2 O3 O4) :
  concyclic A B C D :=
sorry -- Proof not required

end concyclic_points_l314_314188


namespace part1_part2_l314_314968

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314968


namespace work_ratio_l314_314762

theorem work_ratio (r : ℕ) (w : ℕ) (m₁ m₂ d₁ d₂ : ℕ)
  (h₁ : m₁ = 5) 
  (h₂ : d₁ = 15) 
  (h₃ : m₂ = 3) 
  (h₄ : d₂ = 25)
  (h₅ : w = (m₁ * r * d₁) + (m₂ * r * d₂)) :
  ((m₁ * r * d₁):ℚ) / (w:ℚ) = 1 / 2 := by
  sorry

end work_ratio_l314_314762


namespace part1_part2_l314_314934

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314934


namespace lcm_24_36_45_l314_314335

open Nat

theorem lcm_24_36_45 : lcm (lcm 24 36) 45 = 360 := by sorry

end lcm_24_36_45_l314_314335


namespace length_of_tangent_segment_to_circle_at_origin_l314_314369

theorem length_of_tangent_segment_to_circle_at_origin :
  ∀ P : Point, 
  (x - 2)^2 + (y - 1)^2 = 1 → 
  distance origin P = 2 → 
  tangent origin P :=
sorry

end length_of_tangent_segment_to_circle_at_origin_l314_314369


namespace train_speed_l314_314798

/-- 
A train travels with a certain speed for 8 hours and covers a distance of 1200 km.
The speed of the train is 150 km/h.
-/
theorem train_speed (distance time : ℕ) (h_distance : distance = 1200) (h_time : time = 8) : distance / time = 150 :=
by
  rw [h_distance, h_time]
  norm_num
  sorry

end train_speed_l314_314798


namespace isosceles_triangle_area_l314_314060

theorem isosceles_triangle_area (A B C D : Point) (hAB : dist A B = 25) (hAC : dist A C = 25)
  (hBC : dist B C = 14) (hAD_alt : is_altitude A D B C) (hBD_equals_DC : midpoint B C D) :
  area_triangle A B C = 168 :=
sorry

end isosceles_triangle_area_l314_314060


namespace part1_solution_set_part2_range_of_a_l314_314923

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314923


namespace largest_unique_digit_sum_eighteen_l314_314326

theorem largest_unique_digit_sum_eighteen :
  ∃ n : ℕ, (∃ ds : List ℕ, ds.Nodup ∧ ds.Sum = 18 ∧ ds.sorted (≥) ∧ ds.join_digits = n) ∧ n = 852410 := sorry

end largest_unique_digit_sum_eighteen_l314_314326


namespace number_of_camels_is_10_l314_314126

variables (cost_10_elephants cost_camel cost_1_elephant cost_6_oxen 
          cost_1_ox cost_16_horses cost_1_horse cost_24_horses : ℕ)
  
-- problem conditions
def conditions := 
  (cost_10_elephants = 120000) ∧
  (cost_camel = 4800) ∧
  (24 * cost_1_horse = cost_24_horses) ∧
  (16 * cost_1_horse = 4 * cost_1_ox) ∧
  (6 * cost_1_ox = 4 * cost_1_elephant) ∧
  (10 * cost_1_elephant = cost_10_elephants)

-- problem statement
theorem number_of_camels_is_10 (h : conditions) : 
  (cost_24_horses / cost_camel = 10) :=
  by sorry

end number_of_camels_is_10_l314_314126


namespace hunting_dogs_theorem_l314_314665

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l314_314665


namespace ferry_time_increases_l314_314392

noncomputable def ferryRoundTrip (S V x : ℝ) : ℝ :=
  (S / (V + x)) + (S / (V - x))

theorem ferry_time_increases (S V x : ℝ) (h_V_pos : 0 < V) (h_x_lt_V : x < V) :
  ferryRoundTrip S V (x + 1) > ferryRoundTrip S V x :=
by
  sorry

end ferry_time_increases_l314_314392


namespace angle_triple_complement_l314_314317

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314317


namespace rowing_distance_l314_314781

theorem rowing_distance
  (rowing_speed : ℝ)
  (current_speed : ℝ)
  (total_time : ℝ)
  (downstream_speed := rowing_speed + current_speed)
  (upstream_speed := rowing_speed - current_speed)
  (distance : ℝ) :
  rowing_speed = 8 ∧ current_speed = 2 ∧ total_time = 2 → 
  (distance / downstream_speed + distance / upstream_speed = total_time) →
  distance = 7.5 :=
by {
  intros h1 h2,
  cases h1 with hr hc,
  cases hr with hs ht,
  sorry
}

end rowing_distance_l314_314781


namespace price_decrease_to_original_l314_314046

theorem price_decrease_to_original (original_price : ℝ) (increase1 decrease increase2 : ℝ) :
  increase1 = 0.25 → decrease = 0.15 → increase2 = 0.10 →
  let new_price1 := original_price * (1 + increase1) in
  let new_price2 := new_price1 * (1 - decrease) in
  let final_price := new_price2 * (1 + increase2) in
  let decrease_percentage := ((final_price - original_price) / final_price) * 100 in
  decrease_percentage ≈ 14.43 :=
by
  intros
  sorry

end price_decrease_to_original_l314_314046


namespace negation_of_universal_statement_l314_314702

open Real

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^3 > x^2) ↔ ∃ x : ℝ, x^3 ≤ x^2 :=
by
  sorry

end negation_of_universal_statement_l314_314702


namespace distance_range_l314_314554

-- Definitions of points P and Q and the line equation l
def point_P : ℝ × ℝ := (-2, 2)
def point_Q : ℝ × ℝ := (1, -2)

def line_l (a b : ℝ) (x y : ℝ) : Prop :=
  a * (x - 1) + b * (y + 2) = 0

-- Distance from a point to a line
def distance_from_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  let (x, y) := P in
  abs (a * x + b * y + c) / sqrt (a * a + b * b)
  
-- Problem statement
theorem distance_range (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (d : ℝ), d ∈ set.Icc 0 5 :=
sorry

end distance_range_l314_314554


namespace two_courtiers_have_same_selection_l314_314689

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l314_314689


namespace folded_grid_middle_position_is_five_l314_314377

theorem folded_grid_middle_position_is_five :
  ∀ (grid : ℕ → ℕ) (folds : list (ℕ → ℕ)),
    (grid 1 = 1 ∧ grid 2 = 2 ∧ grid 3 = 3 ∧
     grid 4 = 4 ∧ grid 5 = 5 ∧ grid 6 = 6 ∧
     grid 7 = 7 ∧ grid 8 = 8 ∧ grid 9 = 9) →
    folds = [λ x, if x = 3 then 2 else if x = 6 then 5 else if x = 9 then 8 else x,
             λ x, if x = 1 then 2 else if x = 4 then 5 else if x = 7 then 8 else x,
             λ x, if x = 7 then 6 else x] →
    (∃ middle_position, grid middle_position = 5).
Proof
  intros grid folds h_grid h_folds,
  sorry

end folded_grid_middle_position_is_five_l314_314377


namespace train_cross_time_l314_314406

-- Definitions of the given conditions
def speed_train_kmh : ℝ := 72
def time_to_pass_pole : ℝ := 10
def length_stationary_train : ℝ := 500

-- Conversion of the speed from km/h to m/s
def speed_train_ms : ℝ := (speed_train_kmh * 1000) / 3600

-- Calculation of the length of the moving train
def length_moving_train : ℝ := speed_train_ms * time_to_pass_pole

-- Total length to be covered to cross the stationary train
def total_length_covered : ℝ := length_moving_train + length_stationary_train

-- Calculation of the time to cross the stationary train
def time_to_cross_stationary_train : ℝ := total_length_covered / speed_train_ms

-- Statement to prove
theorem train_cross_time :
  time_to_cross_stationary_train = 35 :=
by
  sorry

end train_cross_time_l314_314406


namespace angle_measure_triple_complement_l314_314300

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314300


namespace triple_complement_angle_l314_314284

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314284


namespace center_of_circle_in_second_quadrant_l314_314633

theorem center_of_circle_in_second_quadrant (a : ℝ) (h : a > 12) :
  ∃ x y : ℝ, x^2 + y^2 + a * x - 2 * a * y + a^2 + 3 * a = 0 ∧ (-a / 2, a).2 > 0 ∧ (-a / 2, a).1 < 0 :=
by
  sorry

end center_of_circle_in_second_quadrant_l314_314633


namespace part_I_part_II_l314_314884

-- Definitions based on problem conditions
def P := (-4, 3)
def sin_alpha := 3 / 5
def cos_alpha := -4 / 5
def beta := (3 * Real.pi) / 4 -- Beta in the third quadrant with tan(beta) = 1.
def sin_beta := -Real.sqrt 2 / 2
def cos_beta := -Real.sqrt 2 / 2

-- Theorem to prove part (I)
theorem part_I :
  (cos (alpha - π / 2) * sin (2 * π - alpha) * cos (π - alpha)) /
  sin (π / 2 + alpha) = 9 / 25 := by
  sorry

-- Theorem to prove part (II)
theorem part_II :
  cos (2 * alpha - beta) = 17 * Real.sqrt 2 / 50 := by
  sorry

end part_I_part_II_l314_314884


namespace angle_triple_complement_l314_314254

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314254


namespace tangent_line_at_P_l314_314150

noncomputable theory

open Real

-- Define the function f(x) = 2 * ln x
def f (x : ℝ) : ℝ := 2 * log x

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Define the tangent line equation at point P 
def tangent_line (x : ℝ) : ℝ := 2 * x - 2

-- The main proof goal
theorem tangent_line_at_P : ∀ x : ℝ, (tangent_line x) = 2 * x - 2 :=
by
  sorry

end tangent_line_at_P_l314_314150


namespace find_n_l314_314154

theorem find_n (e n : ℕ) (h_lcm : Nat.lcm e n = 690) (h_n_not_div_3 : ¬ (3 ∣ n)) (h_e_not_div_2 : ¬ (2 ∣ e)) : n = 230 :=
by
  sorry

end find_n_l314_314154


namespace lcm_24_36_45_l314_314333

noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem lcm_24_36_45 : lcm_three 24 36 45 = 360 := by
  -- Conditions involves proving the prime factorizations
  have h1 : Nat.factors 24 = [2, 2, 2, 3] := by sorry -- Prime factorization of 24
  have h2 : Nat.factors 36 = [2, 2, 3, 3] := by sorry -- Prime factorization of 36
  have h3 : Nat.factors 45 = [3, 3, 5] := by sorry -- Prime factorization of 45

  -- Least common multiple calculation based on the greatest powers of prime factors
  sorry -- This is where the proof would go

end lcm_24_36_45_l314_314333


namespace initial_number_of_students_l314_314172

theorem initial_number_of_students
  (T : ℕ) (n : ℕ) (avg_initial : n > 0 → T / n = 8)
  (avg_final : n + 1 > 0 → (T + 28) / (n + 1) = 10)
  (students_in_class : n + 1 ≤ 10) :
  n = 9 :=
begin
  sorry
end

end initial_number_of_students_l314_314172


namespace anthony_pets_left_is_8_l314_314423

def number_of_pets_left (initial_pets : ℕ) (lost_pets : ℕ) (fraction_died : ℚ) : ℕ :=
  initial_pets - lost_pets - (fraction_died * (initial_pets - lost_pets)).toInt

theorem anthony_pets_left_is_8 : number_of_pets_left 16 6 (1/5) = 8 :=
by
  sorry

end anthony_pets_left_is_8_l314_314423


namespace part1_part2_l314_314932

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314932


namespace set_equality_l314_314538

theorem set_equality (A : Set ℕ) (h : {1} ∪ A = {1, 3, 5}) : 
  A = {1, 3, 5} ∨ A = {3, 5} :=
  sorry

end set_equality_l314_314538


namespace point_C_coordinates_l314_314116

-- Define the coordinates of points A and B
def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (4, 9)

-- Define the coordinates of point C to be proven
def C : ℝ × ℝ := (22 / 7, 55 / 7)

-- Define the distances between points
def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Prove that C divides the segment AB in the desired ratio
theorem point_C_coordinates (A B C : ℝ × ℝ) 
  (hA : A = (-2, 1)) (hB : B = (4, 9))
  (hAC_4CB : dist A C = 4 * dist C B) : 
  C = (22 / 7, 55 / 7) :=
by
  sorry

end point_C_coordinates_l314_314116


namespace find_f3_l314_314874

theorem find_f3 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f3_l314_314874


namespace red_marbles_count_l314_314868

theorem red_marbles_count :
  ∀ (total marbles white yellow green red : ℕ),
    total = 50 →
    white = total / 2 →
    yellow = 12 →
    green = yellow / 2 →
    red = total - (white + yellow + green) →
    red = 7 :=
by
  intros total marbles white yellow green red Htotal Hwhite Hyellow Hgreen Hred
  sorry

end red_marbles_count_l314_314868


namespace sin_cos_product_l314_314032

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l314_314032


namespace range_of_a_l314_314905

theorem range_of_a (m a : ℝ) (h : ∀ x, (m * (x^2 - 1) + x - a) = 0 → x = 0):
  (m = 0 → a ∈ set.univ) ∧ (m ≠ 0 → a ∈ set.Icc (-1:ℝ) 1) :=
begin
  sorry
end

end range_of_a_l314_314905


namespace probability_at_least_two_worth_visiting_l314_314111

theorem probability_at_least_two_worth_visiting :
  let total_caves := 8
  let worth_visiting := 3
  let select_caves := 4
  let worth_select_2 := Nat.choose worth_visiting 2 * Nat.choose (total_caves - worth_visiting) 2
  let worth_select_3 := Nat.choose worth_visiting 3 * Nat.choose (total_caves - worth_visiting) 1
  let total_select := Nat.choose total_caves select_caves
  let probability := (worth_select_2 + worth_select_3) / total_select
  probability = 1 / 2 := sorry

end probability_at_least_two_worth_visiting_l314_314111


namespace zero_in_interval_l314_314011

theorem zero_in_interval 
  (a b c : ℝ)
  (h₁ : 2 * a + c / 2 > b) 
  (h₂ : c < 0) :
  ∃ x : ℝ, x ∈ Ioo (-2 : ℝ) (0 : ℝ) ∧ (a * x^2 + b * x + c = 0) :=
begin
  sorry -- The proof is omitted as instructed.
end

end zero_in_interval_l314_314011


namespace angle_measure_triple_complement_l314_314231

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314231


namespace hunting_dogs_theorem_l314_314659

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l314_314659


namespace determinant_zero_solve_x_l314_314483

theorem determinant_zero_solve_x (a : ℝ) (h : a ≠ 0) (x : ℝ) :
  (by { let b : ℝ := 0, exact matrix.det
    ![[x + a, b, x],
      [b, x + a, b],
      [x, x, x + a]] = 0 }) :=
  x = -a / 4 := 
sorry

end determinant_zero_solve_x_l314_314483


namespace triangle_angle_A_l314_314490

theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) (hC : C = Real.pi / 6) (hCos : c = 2 * a * Real.cos B) : A = (5 * Real.pi) / 12 :=
  sorry

end triangle_angle_A_l314_314490


namespace part1_solution_set_part2_range_l314_314992

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314992


namespace angle_triple_complement_l314_314251

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314251


namespace angle_triple_complement_l314_314291

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314291


namespace T_le_1_mul_mp_prod_a_j_congr_neg_one_m_p_minus_one_mod_p_l314_314104

noncomputable def k (n : ℕ) : ℕ := 1                               -- Define k as an integer ≥ 1
def a : ℕ → ℕ
| 0     := 0                     -- a₀ = 0
| 1     := 1                     -- a₁ = 1
| (n+2) := k (n+1) * a (n+1) + a n   -- Recurrence: aₙ₊₂ = k * aₙ₊₁ + aₙ

noncomputable def m (p : ℕ) [hp : Fact (Nat.Prime p)] : ℕ :=
  Nat.find (λ m, p ∣ a m) -- The smallest positive integer m such that p | aₘ

noncomputable def T (p : ℕ) [hp : Fact (Nat.Prime p)] : ℕ :=
  Nat.find (λ T, ∀ j : ℕ, p ∣ (a (T + j) - a j)) -- The smallest positive integer T such that p | (aₜ₊ₖ - aₖ)

theorem T_le_1_mul_mp {p : ℕ} [hp : Fact (Nat.Prime p)] : T p ≤ (p - 1) * m p :=
sorry

theorem prod_a_j_congr_neg_one_m_p_minus_one_mod_p
  {p : ℕ} [hp : Fact (Nat.Prime p)]
  (hT : T p = (p - 1) * m p) :
  (∏ i in Finset.range (T p - 1).filter (λ j, j % m p ≠ 0), a i) ≡ (-1) ^ (m p - 1) [MOD p] :=
sorry

end T_le_1_mul_mp_prod_a_j_congr_neg_one_m_p_minus_one_mod_p_l314_314104


namespace complement_union_l314_314575

open Set

variable (U A B : Set ℕ)
variable (C_UA C_UB : Set ℕ)

def U := {0, 1, 2, 3, 4}
def A := {0, 1, 2, 3}
def B := {2, 3, 4}
def C_UA := U \ A
def C_UB := U \ B

theorem complement_union (U A B C_UA C_UB : Set ℕ) (h1 : U = {0, 1, 2, 3, 4}) (h2 : A = {0, 1, 2, 3}) (h3 : B = {2, 3, 4}) (h4 : C_UA = U \ A) (h5 : C_UB = U \ B) :
  C_UA ∪ C_UB = {0, 1, 4} := by
  sorry

end complement_union_l314_314575


namespace angle_measure_triple_complement_l314_314224

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314224


namespace part1_solution_set_part2_range_a_l314_314950

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314950


namespace angle_is_67_l314_314208

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314208


namespace find_sin_A_l314_314552

theorem find_sin_A (A B C : Type) [right_triangle A B C (∠ B = 90)] (h1: 3 * sin A = 2 * cos A) :
  sin A = (2 * real.sqrt 13) / 13 :=
by
  sorry

end find_sin_A_l314_314552


namespace planes_determined_by_three_parallel_lines_l314_314158

-- Define the conditions
def three_lines_mutually_parallel (l1 l2 l3 : ℝ → ℝ → Prop) : Prop :=
  ∀ p q : ℝ → ℝ → Prop, (l1 p q → l2 p q) ∧ (l2 p q → l3 p q)

def three_lines_in_same_plane (l1 l2 l3 : ℝ → ℝ → Prop) (P : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ p : ℝ, P (l1 p) (l2 p) (l3 p)

def three_lines_in_different_planes (l1 l2 l3 : ℝ → ℝ → Prop) (P1 P2 P3 : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ p : ℝ, (P1 (l1 p)) (P2 (l2 p)) (P3 (l3 p))

-- State the theorem
theorem planes_determined_by_three_parallel_lines (l1 l2 l3 : ℝ → ℝ → Prop)
  (same_plane : ∃ P : ℝ → ℝ → ℝ → Prop, three_lines_in_same_plane l1 l2 l3 P)
  (different_planes : ∃ P1 P2 P3 : ℝ → ℝ → ℝ → Prop, three_lines_in_different_planes l1 l2 l3 P1 P2 P3) :
  ∃ n : ℕ, (n = 1 ∨ n = 3) :=
by
  sorry

end planes_determined_by_three_parallel_lines_l314_314158


namespace courtiers_selection_l314_314668

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l314_314668


namespace johns_original_two_digit_number_l314_314080

theorem johns_original_two_digit_number (x : ℕ) (h : 10 ≤ x ∧ x < 100)
  (h2 : let y := 4 * x + 17 in 96 ≤ ((y % 10) * 10 + (y / 10)) ∧ ((y % 10) * 10 + (y / 10)) ≤ 98) : 
  x = 13 ∨ x = 18 :=
begin
  sorry
end

end johns_original_two_digit_number_l314_314080


namespace angle_triple_complement_l314_314256

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314256


namespace population_std_dev_in_confidence_interval_l314_314885

noncomputable def confidence_interval_population_std_dev 
  (n : ℕ) (s : ℝ) (confidence_level : ℝ) : set ℝ :=
{σ | 0.56 < σ ∧ σ < 1.44}

theorem population_std_dev_in_confidence_interval 
  (n : ℕ) (s : ℝ) (confidence_level : ℝ) (h_n : n = 16) (h_s : s = 1) (h_cl : confidence_level = 0.95) :
  ∀ σ : ℝ, σ ∈ confidence_interval_population_std_dev n s confidence_level :=
sorry

end population_std_dev_in_confidence_interval_l314_314885


namespace sum_coefficients_binomial_expansion_l314_314166

theorem sum_coefficients_binomial_expansion :
  (Finset.univ.sum (λ k, Nat.choose 6 k)) = 64 := 
sorry

end sum_coefficients_binomial_expansion_l314_314166


namespace part1_solution_set_part2_range_of_a_l314_314922

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314922


namespace isosceles_trapezoid_shorter_base_length_l314_314551

/--
In an isosceles trapezoid, the line joining the midpoints of the diagonals has length 4 and the longer base is 100.
Prove that the length of the shorter base is 92.
-/
theorem isosceles_trapezoid_shorter_base_length
  (mid_segment_length : ℝ)
  (longer_base : ℝ)
  (shorter_base : ℝ) :
  mid_segment_length = 4 →
  longer_base = 100 →
  shorter_base = 100 - 2 * mid_segment_length →
  shorter_base = 92 :=
by
  intros h_mid_segment h_longer_base h_shorter_base
  unfold shorter_base
  rw [h_mid_segment, h_longer_base] at h_shorter_base
  exact h_shorter_base


end isosceles_trapezoid_shorter_base_length_l314_314551


namespace angle_triple_complement_l314_314246

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314246


namespace distance_between_midpoints_l314_314085

variables {R a : ℝ} (O A B C M N L K P Q : Point)
variables (circle : Circle) (hO : center O circle) (hAB : diameter A B circle) (hAB_len : dist A B = 2 * R)
variables (hC : on_circle C circle) (hM : on_segment M A C) (hMN : ∃ N, perp_from_point N MN A B) (hL : ∃ L, perp_to_line L AC)
variables (hL_on_circle : on_circle L circle) (hL_intersects : on_segment L C A B) (hAN : dist A N = a)

theorem distance_between_midpoints:
  dist (midpoint A O) (midpoint C L) = √(R^2 / 4 + R * a) := 
sorry

end distance_between_midpoints_l314_314085


namespace red_marbles_count_l314_314870

theorem red_marbles_count (W Y G R : ℕ) (total_marbles : ℕ) 
(h1 : total_marbles = 50)
(h2 : W = 50 / 2)
(h3 : Y = 12)
(h4 : G = 12 - (12 * 0.5))
(h5 : W + Y + G + R = total_marbles)
: R = 7 :=
sorry

end red_marbles_count_l314_314870


namespace purchase_gifts_and_have_money_left_l314_314730

/-
  We start with 5000 forints in our pocket to buy gifts, visiting three stores.
  In each store, we find a gift that we like and purchase it if we have enough money. 
  The prices in each store are independently 1000, 1500, or 2000 forints, each with a probability of 1/3. 
  What is the probability that we can purchase gifts from all three stores 
  and still have money left (i.e., the total expenditure is at most 4500 forints)?
-/

def giftProbability (totalForints : ℕ) (prices : List ℕ) : ℚ :=
  let outcomes := prices |>.product prices |>.product prices
  let favorable := outcomes.filter (λ ((p1, p2), p3) => p1 + p2 + p3 <= totalForints)
  favorable.length / outcomes.length

theorem purchase_gifts_and_have_money_left :
  giftProbability 4500 [1000, 1500, 2000] = 17 / 27 :=
sorry

end purchase_gifts_and_have_money_left_l314_314730


namespace same_selection_exists_l314_314642

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l314_314642


namespace largest_divisor_of_expression_l314_314463

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 - n) := 
sorry

end largest_divisor_of_expression_l314_314463


namespace part1_solution_set_part2_range_l314_314990

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314990


namespace f_2009_l314_314497

noncomputable def f : ℝ → ℝ := sorry -- This will be defined by the conditions.

axiom even_f (x : ℝ) : f x = f (-x)
axiom periodic_f (x : ℝ) : f (x + 6) = f x + f 3
axiom f_one : f 1 = 2

theorem f_2009 : f 2009 = 2 :=
by {
  -- The proof would go here, summarizing the logical steps derived in the previous sections.
  sorry
}

end f_2009_l314_314497


namespace area_increase_l314_314743

variable (l w : ℝ)

theorem area_increase (hl : l > 0) (hw : w > 0) :
  let A := l * w
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  (A_new - A) / A * 100 = 56 := by
  let A := l * w
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  have : A_new = 1.56 * A := by
    rw [←mul_assoc, ←mul_assoc, mul_comm 1.3 l, mul_assoc, mul_comm 1.2 w, mul_assoc, mul_comm w l, ←mul_assoc]
    ring
  calc
    (A_new - A) / A * 100 = ((1.56 * A) - A) / A * 100 := by rw this
    ... = (1.56 - 1) * 100 := by field_simp [A, lt_mul_iff_one_lt_left (lt_of_lt_of_le (zero_lt_one) hl), ne_of_gt hl]
    ... = 0.56 * 100 := by ring
    ... = 56 := by norm_num

end area_increase_l314_314743


namespace tangent_lines_from_origin_to_circle_l314_314878

-- Defining the given conditions
variables (m : ℝ) (circle_eq : ∀ x y : ℝ, x^2 + (m+2)*y^2 - 4*x - 8*y - 16*m = 0)

-- Statement of the theorem
theorem tangent_lines_from_origin_to_circle (m : ℝ) :
  (m + 2 = 1) → 
  let C := (2 : ℝ, 4 : ℝ) in
  let R := (2 : ℝ) in
  ∃ k : ℝ, (k = 3/4 ∨ k = 0) ∧ (∀ x y : ℝ, y = k*x) :=
by
  sorry

end tangent_lines_from_origin_to_circle_l314_314878


namespace angle_triple_complement_l314_314250

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314250


namespace abs_sum_eq_3_given_condition_l314_314499

theorem abs_sum_eq_3_given_condition (m n p : ℤ)
  (h : |m - n|^3 + |p - m|^5 = 1) :
  |p - m| + |m - n| + 2 * |n - p| = 3 :=
sorry

end abs_sum_eq_3_given_condition_l314_314499


namespace integral_curve_has_inflection_points_l314_314190

theorem integral_curve_has_inflection_points (x y : ℝ) (f : ℝ → ℝ → ℝ) :
  f x y = y - x^2 + 2*x - 2 →
  (∃ y' y'' : ℝ, y' = f x y ∧ y'' = y - x^2 ∧ y'' = 0) ↔ y = x^2 :=
by
  sorry

end integral_curve_has_inflection_points_l314_314190


namespace reduced_price_per_kg_l314_314792

-- Definitions
variables {P R Q : ℝ}

-- Conditions
axiom reduction_price : R = P * 0.82
axiom original_quantity : Q * P = 1080
axiom reduced_quantity : (Q + 8) * R = 1080

-- Proof statement
theorem reduced_price_per_kg : R = 24.30 :=
by {
  sorry
}

end reduced_price_per_kg_l314_314792


namespace number_of_correct_propositions_l314_314039

theorem number_of_correct_propositions (m n : Line) (h : angle_between m n = 60) : 
  number_of_correct_propositions m n = 3 :=
sorry

end number_of_correct_propositions_l314_314039


namespace exists_non_dueling_trio_l314_314716

-- Define the problem setup
def num_musketeers : ℕ := 50
def days : ℕ := 24
def musketeers := Fin num_musketeers
def duels_per_day : list (musketeers × musketeers) := sorry -- list of daily duels, unspecified here

-- Define the theorem statement
theorem exists_non_dueling_trio :
  ∃ (trio : Finset musketeers), trio.card = 3 ∧ ∀ (x y : musketeers), x ∈ trio → y ∈ trio → (x, y) ∉ (duels_per_day) :=
by
  -- Proof omitted
  sorry

end exists_non_dueling_trio_l314_314716


namespace unique_10_tuple_solution_l314_314467

noncomputable def condition (x : Fin 10 → ℝ) : Prop :=
  (1 - x 0)^2 +
  (x 0 - x 1)^2 + 
  (x 1 - x 2)^2 + 
  (x 2 - x 3)^2 + 
  (x 3 - x 4)^2 + 
  (x 4 - x 5)^2 + 
  (x 5 - x 6)^2 + 
  (x 6 - x 7)^2 + 
  (x 7 - x 8)^2 + 
  (x 8 - x 9)^2 + 
  x 9^2 + 
  (1/2) * (x 9 - x 0)^2 = 1/10

theorem unique_10_tuple_solution : 
  ∃! (x : Fin 10 → ℝ), condition x := 
sorry

end unique_10_tuple_solution_l314_314467


namespace measure_angle_ECD_l314_314049

theorem measure_angle_ECD
  (A B C D E : Type)
  [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty E] 
  (h_isosceles: AC = BC)
  (h_eq_angle: mangle ACB = 60)
  (h_parallel_CD_AB: CD ∥ AB) 
  (h_parallel_DE_BC: ∀ D E, D = E ∧ D ∥ BC) :
  mangle ECD = 60 := 
begin
  sorry
end

end measure_angle_ECD_l314_314049


namespace courtiers_selection_l314_314673

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l314_314673


namespace hunting_dogs_theorem_l314_314664

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l314_314664


namespace find_angle_A_l314_314544

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) 
  (h3 : B = Real.pi / 4) : A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l314_314544


namespace problem_statement_l314_314535

-- Define complex number i
noncomputable def i : ℂ := Complex.I

-- Define x as per the problem statement
noncomputable def x : ℂ := (1 + i * Real.sqrt 3) / 2

-- The main proposition to prove
theorem problem_statement : (1 / (x^2 - x)) = -1 :=
  sorry

end problem_statement_l314_314535


namespace A_is_algebra_A_is_not_sigma_algebra_l314_314102

variable {α : Type*}
variable (𝒜 : ℕ → Set (Set α))

-- Define a sequence of algebras such that 𝒜ₙ ⊆ 𝒜ₙ₊₁ for n ≥ 1
axiom increasing_algebras (n : ℕ) : 𝒜 n ⊆ 𝒜 (n + 1)

-- Define 𝒜 = ⋃ₙ 𝒜ₙ
def union_algebras := ⋃ n, 𝒜 n

-- Theorems to prove
theorem A_is_algebra (α : Type*) (𝒜 : ℕ → Set (Set α)) (h : ∀ n, 𝒜 n ⊆ 𝒜 (n + 1)) : 
  is_algebra (union_algebras 𝒜) :=
sorry

theorem A_is_not_sigma_algebra (α : Type*) (𝒜 : ℕ → Set (Set α)) (h : ∀ n, 𝒜 n ⊆ 𝒜 (n + 1)) : 
  ¬ is_sigma_algebra (union_algebras 𝒜) :=
sorry

end A_is_algebra_A_is_not_sigma_algebra_l314_314102


namespace distance_between_consecutive_trees_approx_l314_314415

-- Definitions and conditions from the problem
def yard_length : ℝ := 1527
def number_of_trees : ℝ := 37
def number_of_gaps : ℝ := number_of_trees - 1
def distance_between_trees := yard_length / number_of_gaps

-- Theorem stating the desired property
theorem distance_between_consecutive_trees_approx :
  abs (distance_between_trees - 42.42) < 0.01 :=
by
  -- The proof goes here.
  sorry

end distance_between_consecutive_trees_approx_l314_314415


namespace bridget_more_than_sarah_l314_314816

theorem bridget_more_than_sarah (total_cents : ℕ) (sarah_cents : ℕ) (bridget_cents : ℕ) 
  (h1 : total_cents = 300) (h2 : sarah_cents = 125) (h3 : bridget_cents = total_cents - sarah_cents) :
  bridget_cents - sarah_cents = 50 := by
  -- problem condition definitions
  have bridget_def : bridget_cents = 300 - 125 := by
    rw [h1, h2]
    exact h3
  rw bridget_def
  norm_num
  sorry

end bridget_more_than_sarah_l314_314816


namespace abs_ineq_solution_set_l314_314708

theorem abs_ineq_solution_set {x : ℝ} : (|x - 1| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end abs_ineq_solution_set_l314_314708


namespace row_sum_lt_518_l314_314059

-- Define the problem conditions and statement
noncomputable def grid : Type := (Fin 8) → (Fin 8) → ℕ

def valid_grid (g : grid) : Prop :=
  ∑ i j, g i j = 1956 ∧
  ∑ k, g k k + ∑ k, g k (Fin.mk (7 - k.val) sorry) = 112 ∧
  (∀ i j : Fin 8, g i j = g (Fin.mk (7 - i.val) sorry) (Fin.mk (7 - j.val) sorry))

theorem row_sum_lt_518 (g : grid) (h : valid_grid g) : ∀ i : Fin 8, ∑ j, g i j < 518 :=
by
  intros i
  sorry

end row_sum_lt_518_l314_314059


namespace impossible_to_cover_cube_faces_l314_314485

def side_length : ℕ := 4
def strip_length : ℕ := 3
def strip_width : ℕ := 1
def total_strips : ℕ := 16

-- Definitions and assumptions
def face_area (side_length: ℕ) := side_length * side_length
def covered_area (total_strips: ℕ) (strip_length: ℕ) (strip_width: ℕ) := total_strips * strip_length * strip_width

-- Cube properties
def cube_faces_with_vertex_covered := 3
def total_face_area := cube_faces_with_vertex_covered * face_area side_length

-- Proof statement: It's impossible to cover three faces entirely with given strips
theorem impossible_to_cover_cube_faces :
  ∀ (side_length strip_length strip_width total_strips : ℕ),
    side_length = 4 →
    strip_length = 3 →
    strip_width = 1 →
    total_strips = 16 →
    total_face_area = covered_area total_strips strip_length strip_width →
    ¬ (exists (P : set (ℕ × ℕ × ℕ)), -- Assume an existing solution
       ∀ (x y z : ℕ), (x, y, z) ∈ P → 
       (x < side_length ∧ y < side_length ∧ z < side_length) ∧
       ∀ (x y z : ℕ), (x, y, z) ∈ P →
       by sorry) -- to be completed
  :=
by sorry

end impossible_to_cover_cube_faces_l314_314485


namespace length_of_AD_l314_314543

noncomputable def vector_space := ℝ -- Defining vector space over ℝ

def triangle (A B C : vector_space) := true -- placeholder definition of a triangle
def angle (A B C : vector_space) := true -- placeholder for the angle notation
def angle_bisector (A B : vector_space) (alpha : ℝ) := true -- placeholder for angle bisector

def length (v : vector_space) : ℝ := sorry -- placeholder for length function

theorem length_of_AD
  (A B C D : vector_space)
  (triangle_ABC : triangle A B C)
  (angle_BAC : angle A B C)
  (angle_cond : angle A B C = 60)
  (angle_bisector_AD : angle_bisector A D 30)
  (AB_eq : length (B - A) = 8)
  (AD_eq : D = (1/4) • ((C - A) : vector_space) + (t • (B - A)))
  : length (D - A) = 6 * real.sqrt 3 :=
sorry

end length_of_AD_l314_314543


namespace solution_set_of_inequality_l314_314165

theorem solution_set_of_inequality (x : ℝ) : 
  (x ≠ 0 ∧ (x * (x - 1)) ≤ 0) ↔ 0 < x ∧ x ≤ 1 :=
sorry

end solution_set_of_inequality_l314_314165


namespace part1_solution_set_part2_range_l314_314988

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314988


namespace range_of_x_l314_314505

-- Given definitions and properties
def g (x : ℝ) : ℝ :=
  if x < 0 then -real.log (1 - x)
  else real.log (1 + x)

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3
  else g x

-- The theorem we need to prove
theorem range_of_x (x : ℝ) (h : f (2 - x^2) > f x) : x ∈ set.Ioo (-2 : ℝ) 1 :=
sorry

end range_of_x_l314_314505


namespace tom_final_amount_l314_314181

-- Conditions and definitions from the problem
def initial_amount : ℝ := 74
def spent_percentage : ℝ := 0.15
def earnings : ℝ := 86
def share_percentage : ℝ := 0.60

-- Lean proof statement
theorem tom_final_amount :
  (initial_amount - (spent_percentage * initial_amount)) + (share_percentage * earnings) = 114.5 :=
by
  sorry

end tom_final_amount_l314_314181


namespace gcd_of_polynomials_l314_314903

theorem gcd_of_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5959 * k) :
  Int.gcd (4 * b^2 + 73 * b + 156) (4 * b + 15) = 1 :=
by
  sorry

end gcd_of_polynomials_l314_314903


namespace cevian_inequality_l314_314550

theorem cevian_inequality
    (A B C A' B' C' O : Point)
    (triangle : Triangle A B C)
    (cep_A : Cevian A' A B C O)
    (cep_B : Cevian B' B A C O)
    (cep_C : Cevian C' C A B O)
    (ineq : dist A A' ≥ dist B B' ∧ dist B B' ≥ dist C C') :
    dist A A' ≥ dist O A' + dist O B' + dist O C' := 
sorry

end cevian_inequality_l314_314550


namespace tan_alpha_equals_one_l314_314901

theorem tan_alpha_equals_one (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos (α + β) = Real.sin (α - β))
  : Real.tan α = 1 := 
by
  sorry

end tan_alpha_equals_one_l314_314901


namespace angle_measure_l314_314273

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314273


namespace circles_intersect_l314_314894

noncomputable def circle (h k r : ℝ) := {p : ℝ × ℝ // (p.1 - h) ^ 2 + (p.2 - k) ^ 2 = r ^ 2}

def center (c : ℝ × ℝ) : ℝ × ℝ := c

def radius (r : ℝ) : ℝ := r

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def intersection_relationship (d r1 r2 : ℝ) : Prop :=
  d > Real.abs (r1 - r2) ∧ d < r1 + r2

theorem circles_intersect :
  let M_center := (0, 2)
  let M_radius := 2
  let N_center := (1, 1)
  let N_radius := 1
  let d := distance M_center N_center
  (circle 0 2 2 == circle 1 1 1) → intersection_relationship d 2 1 :=
by {
  intro h, 
  rw [distance, intersection_relationship], 
  simp, 
  sorry
}

end circles_intersect_l314_314894


namespace g_bound_l314_314098

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f(x + y) + f(x - y) = 2 * f(x) * g(y)
axiom f_nonzero (x : ℝ) : f(x) ≠ 0
axiom f_bound (x : ℝ) : abs (f(x)) ≤ 1

theorem g_bound (y : ℝ) : abs (g(y)) ≤ 1 := by
  sorry

end g_bound_l314_314098


namespace angle_measure_triple_complement_l314_314301

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314301


namespace angle_triple_complement_l314_314203

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314203


namespace hunting_dogs_theorem_l314_314663

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l314_314663


namespace graph_passes_fixed_point_l314_314639

-- Mathematical conditions
variables (a : ℝ)

-- Real numbers and conditions
def is_fixed_point (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧ ∃ x y, (x, y) = (2, 2) ∧ y = a^(x-2) + 1

-- Lean statement for the problem
theorem graph_passes_fixed_point : is_fixed_point a :=
  sorry

end graph_passes_fixed_point_l314_314639


namespace sin_cos_product_l314_314028

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l314_314028


namespace problem_ab_gt_ac_l314_314524

variable {a b c : ℝ}
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : a + b + c = 0)

theorem problem_ab_gt_ac (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : ab > ac :=
sorry

end problem_ab_gt_ac_l314_314524


namespace find_prime_triplets_l314_314848

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_triplet (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ (p * (r + 1) = q * (r + 5))

theorem find_prime_triplets :
  { (p, q, r) | valid_triplet p q r } = {(3, 2, 7), (5, 3, 5), (7, 3, 2)} :=
by {
  sorry -- Proof is to be completed
}

end find_prime_triplets_l314_314848


namespace find_first_term_of_infinite_geometric_series_l314_314167

variable {a r : ℝ}

def geometric_series_first_term (sum sum_of_squares : ℝ) (h_sum : a / (1 - r) = sum) (h_sum_of_squares : a^2 / (1 - r^2) = sum_of_squares) : Prop :=
  a = 5

theorem find_first_term_of_infinite_geometric_series
  (h_sum : a / (1 - r) = 15)
  (h_sum_of_squares : a^2 / (1 - r^2) = 45) :
  geometric_series_first_term 15 45 h_sum h_sum_of_squares :=
begin
  sorry
end

end find_first_term_of_infinite_geometric_series_l314_314167


namespace part1_solution_set_part2_range_a_l314_314948

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314948


namespace central_angle_of_region_l314_314384

theorem central_angle_of_region (A : ℝ) (θ : ℝ) (h : (1:ℝ) / 8 = (θ / 360) * A / A) : θ = 45 :=
by
  sorry

end central_angle_of_region_l314_314384


namespace tangent_at_point_monotonic_intervals_zero_count_l314_314512

def f (a x : ℝ) : ℝ := (exp x - a) / x - a * log x

noncomputable def tangent_line (a : ℝ) : (ℝ × ℝ) → ℝ := 
  λ p, p.snd

theorem tangent_at_point (a : ℝ) : 
  tangent_line a (1, f a 1) = e - a :=
by
  sorry

theorem monotonic_intervals (a : ℝ) : 
  (a ≤ 1) ∨ (a > 1 ∧ a < exp 1) → 
  (∀ x, x ∈ (0, 1) → deriv (f a) x < 0) ∧ 
  (∀ x, x ∈ (1, ∞) → deriv (f a) x > 0) :=
by
  sorry

theorem zero_count (a : ℝ) (h : a ≥ exp 1) : 
  ∃! x, x ∈ (0, ∞) ∧ f a x = 0 :=
by
  sorry

end tangent_at_point_monotonic_intervals_zero_count_l314_314512


namespace true_propositions_count_l314_314161

theorem true_propositions_count (a : ℝ) :
  ((a > -3 → a > -6) ∧ (a > -6 → ¬(a ≤ -3)) ∧ (a ≤ -3 → ¬(a > -6)) ∧ (a ≤ -6 → a ≤ -3)) → 
  2 = 2 := 
by
  sorry

end true_propositions_count_l314_314161


namespace most_people_can_attend_on_Tuesday_l314_314840

-- Define the days
inductive Day : Type
| Mon | Tues | Wed | Thur | Fri

open Day

-- Define the attendance constraints
def Anna (d : Day) : Prop := d = Mon ∨ d = Wed
def Bill (d : Day) : Prop := d = Tues ∨ d = Thur ∨ d = Fri
def Carl (d : Day) : Prop := d = Mon ∨ d = Tues ∨ d = Thur ∨ d = Fri
def Dave (d : Day) : Prop := d = Wed ∨ d = Fri
def Eve (d : Day) : Prop := d = Mon ∨ d = Thur

-- Define the set of participants
def participants (d : Day) : ℕ :=
  if Anna d then 0 else 1 +
  if Bill d then 0 else 1 +
  if Carl d then 0 else 1 +
  if Dave d then 0 else 1 +
  if Eve d then 0 else 1

-- The theorem to prove
theorem most_people_can_attend_on_Tuesday : 
  ∀ d : Day, d ≠ Tues → participants Tues > participants d :=
begin
  sorry
end

end most_people_can_attend_on_Tuesday_l314_314840


namespace part1_solution_part2_solution_l314_314961

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314961


namespace possible_remainder_degrees_l314_314347

theorem possible_remainder_degrees (f : Polynomial ℝ) :
  let g := Polynomial.C (2 : ℝ) * (Polynomial.X ^ 4) - 
           Polynomial.C (5 : ℝ) * (Polynomial.X ^ 2) + 
           Polynomial.C (3 : ℝ)
  in g.degree = 4 → ∃ n : ℕ, n < 4 ∧ (f % g).degree = n :=
by
  intro h
  use [0, 1, 2, 3]
  sorry

end possible_remainder_degrees_l314_314347


namespace original_cost_l314_314360

theorem original_cost (C : ℝ) (h : 670 = C + 0.35 * C) : C = 496.30 :=
by
  -- The proof is omitted
  sorry

end original_cost_l314_314360


namespace angle_measure_triple_complement_l314_314308

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314308


namespace symmetry_y_axis_l314_314344

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 5 * Real.pi / 2)

theorem symmetry_y_axis : ∀ x, f(-x) = f(x) := by
  sorry

end symmetry_y_axis_l314_314344


namespace f1_g1_eq_one_l314_314875

-- Definitions of even and odd functions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given statement to be proved
theorem f1_g1_eq_one (f g : ℝ → ℝ) (h_even : even_function f) (h_odd : odd_function g)
    (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 1 :=
  sorry

end f1_g1_eq_one_l314_314875


namespace chord_length_constant_circle_equation_l314_314879

-- Question 1
theorem chord_length_constant (t : ℝ) (a : ℝ) (h_t : t ≠ 0) (h_a : a > 0) :
  let C := (t, t^2 / (2 * a))
  let A := (0, a)
  let B := (t + a, 0)
  let D := (t - a, 0)
  |(B.1 - D.1)| = 2 * a :=
by
sorry

-- Question 2
theorem circle_equation (t : ℤ) (h_a : a = 1 / 2) (h_t : t ∈ {n : ℤ | ∃ k : ℤ, n = -4 + k ∨ n = 2 + k}) :
  let C := (t, t^2)
  let line_distance := |2 * t + t^2 - 6| / real.sqrt 5
  (line_distance = 2 * real.sqrt 5 / 5 → 
    ∃ k : ℤ, k = 0 → 
      (circle_eq := (x + 4)^2 + (y - 16)^2 = 1025 / 4 ∨
        circle_eq := (x - 2)^2 + (y - 4)^2 = 65 / 4)) :=
by
sorry

end chord_length_constant_circle_equation_l314_314879


namespace minimum_value_of_expression_l314_314016

noncomputable def minimum_value_expression : ℝ :=
  let line1 := (0, 0)
  let line2 := (1, 3)
  let M := (1 / 2, 3 / 2)
  let line_c1c2 := λ x y, 2 * x + 6 * y - 10
  let P := (3, 2)
  let distance := (λ x y, abs (2 * x + 6 * y - 10) / (real.sqrt (4 + 36)))
  let squared_distance := (distance P.1 P.2) ^ 2
  in squared_distance

theorem minimum_value_of_expression : minimum_value_expression = (8 / 5) :=
  sorry

end minimum_value_of_expression_l314_314016


namespace three_digit_numbers_excluding_adjacent_identical_digits_l314_314022

theorem three_digit_numbers_excluding_adjacent_identical_digits : 
    let total_nums := 999 - 100 + 1 in
    let invalid_nums := (9 * 9) + (9 * 9) - 9 in
    let valid_nums := total_nums - invalid_nums in
    valid_nums = 747 :=
by 
  let total_nums := 999 - 100 + 1
  let invalid_nums := (9 * 9) + (9 * 9) - 9
  let valid_nums := total_nums - invalid_nums
  sorry

end three_digit_numbers_excluding_adjacent_identical_digits_l314_314022


namespace angle_triple_complement_l314_314242

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314242


namespace donation_fifth_sixth_l314_314623

-- Conditions definitions
def total_donation := 10000
def first_home := 2750
def second_home := 1945
def third_home := 1275
def fourth_home := 1890

-- Proof statement
theorem donation_fifth_sixth : 
  (total_donation - (first_home + second_home + third_home + fourth_home)) = 2140 := by
  sorry

end donation_fifth_sixth_l314_314623


namespace termites_ate_12_black_squares_l314_314138

def is_black_square (r c : ℕ) : Prop :=
  (r + c) % 2 = 0

def eaten_positions : List (ℕ × ℕ) :=
  [(3, 1), (4, 6), (3, 7), (4, 1), (2, 3), (2, 4), (4, 3), (3, 5), (3, 2), (4, 7), (3, 6), (2, 6)]

def count_black_squares (positions : List (ℕ × ℕ)) : ℕ :=
  positions.countp (λ ⟨r, c⟩ => is_black_square r c)

theorem termites_ate_12_black_squares : count_black_squares eaten_positions = 12 := by
  sorry

end termites_ate_12_black_squares_l314_314138


namespace length_of_XY_l314_314184

noncomputable theory

variables {P Q R X Y Z G H I : Type}
variables [MetricSpace P] [MetricSpace X] [MetricSpace G]

-- Defining the similarity relation between triangles
def similar (a b c d e f : Type) [MetricSpace a] [MetricSpace b] [MetricSpace c] : Prop :=
  ∃ (k : ℝ), k > 0 ∧
  ∀ u v w : ℝ, dist u v = k * dist d e ∧ dist v w = k * dist e f ∧ dist w u = k * dist f d

-- Given conditions
axiom h1 : similar P Q R X Y Z
axiom h2 : similar X Y Z G H I
axiom h3 : dist P Q = 5
axiom h4 : dist Q R = 15
axiom h5 : dist H I = 30

-- Proof statement
theorem length_of_XY : dist X Y = 2.5 :=
sorry

end length_of_XY_l314_314184


namespace quadratic_expression_positive_intervals_l314_314445

noncomputable def quadratic_expression (x : ℝ) : ℝ := (x + 3) * (x - 1)
def interval_1 (x : ℝ) : Prop := x < (1 - Real.sqrt 13) / 2
def interval_2 (x : ℝ) : Prop := x > (1 + Real.sqrt 13) / 2

theorem quadratic_expression_positive_intervals (x : ℝ) :
  quadratic_expression x > 0 ↔ interval_1 x ∨ interval_2 x :=
by {
  sorry
}

end quadratic_expression_positive_intervals_l314_314445


namespace R_depends_on_a_d_n_l314_314584

theorem R_depends_on_a_d_n
  (a d n: ℤ)
  (s1 : ℤ := n * (2 * a + (n - 1) * d) / 2)
  (s2 : ℤ := 3 * n * (2 * a + (3 * n - 1) * d) / 2)
  (s3 : ℤ := 5 * n * (2 * a + (5 * n - 1) * d) / 2)
  (R : ℤ := s3 - s2 - s1) :
  ∃ a d n : ℤ, R = n * a * (2 - d) + 15 * d * n ^ 2 :=
begin
  sorry
end

end R_depends_on_a_d_n_l314_314584


namespace miles_to_add_per_week_l314_314076

theorem miles_to_add_per_week :
  ∀ (initial_miles_per_week : ℝ) 
    (percentage_increase : ℝ) 
    (total_days : ℕ) 
    (days_in_week : ℕ),
    initial_miles_per_week = 100 →
    percentage_increase = 0.2 →
    total_days = 280 →
    days_in_week = 7 →
    ((initial_miles_per_week * (1 + percentage_increase)) - initial_miles_per_week) / (total_days / days_in_week) = 3 :=
by
  intros initial_miles_per_week percentage_increase total_days days_in_week
  intros h1 h2 h3 h4
  sorry

end miles_to_add_per_week_l314_314076


namespace similarity_of_triangles_l314_314569

theorem similarity_of_triangles
  (C1 C2 : Type)
  (O1 O2 A B P Q : Type)
  (l1 l2 : Type)
  (tangent_l1_C1_at_A : tangent_to_circle_at C1 O1 A l1)
  (tangent_l2_C2_at_B : tangent_to_circle_at C2 O2 B l2)
  (intersection_A_B : intersect_at C1 C2 A B)
  (l1_intersects_l2_at_P : intersects_at l1 l2 P)
  (l1_intersects_C2_at_Q : intersects_again_at l1 C2 P Q) :
  similar_triangles (triangle P O1 B) (triangle P O2 Q) :=
sorry

end similarity_of_triangles_l314_314569


namespace f_odd_solve_inequality_l314_314637

variable {f : ℝ → ℝ}

-- The function f is defined on (-1,1) and satisfies the additive condition
axiom f_domain : ∀ x, x ∈ (-1 : ℝ, 1) → f x = f x
axiom f_additive : ∀ x y, x ∈ (-1 : ℝ, 1) → y ∈ (-1 : ℝ, 1) → f(x + y) = f(x) + f(y)

-- Prove that f is an odd function, i.e., f(-x) = -f(x)
theorem f_odd : ∀ x, x ∈ (-1 : ℝ, 1) → f(-x) = -f(x) :=
sorry

-- Solve the inequality f(log2 x - 1) + f(log2 x) < 0 with respect to x
theorem solve_inequality : ∀ x, (0 < log 2 x ∧ log 2 x < (1 / 2 : ℝ)) ↔ (1 < x ∧ x < real.sqrt 2) :=
sorry

end f_odd_solve_inequality_l314_314637


namespace part1_solution_set_part2_range_of_a_l314_314917

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314917


namespace part1_part2_l314_314970

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314970


namespace angle_measure_l314_314272

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314272


namespace main_theorem_l314_314553

/-- Parametric equation of the line l -/
def line_l (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (2 + t * Real.cos α, t * Real.sin α)

/-- Polar equation of curve C -/
def polar_eq_C (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ ^ 2 + 2 * ρ^2 * Real.sin θ ^ 2 = 12

/-- Cartesian equation of curve C -/
def cartesian_eq_C (x y : ℝ) : Prop :=
  x^2 + 2 * y^2 = 12

/-- The point A through which line l always passes -/
def point_A : ℝ × ℝ :=
  (2, 0)

/-- Cartesian equation of line l -/
def cartesian_eq_l (x y α : ℝ) : Prop :=
  y = (Real.sqrt 2 / 2) * (x - 2) ∨ y = -(Real.sqrt 2 / 2) * (x - 2)

/-- The main theorem to be proved -/
theorem main_theorem (α : ℝ) (x y : ℝ) :
  (∃ (t : ℝ), line_l t α = (x, y)) ∧ polar_eq_C (Real.sqrt (x^2 + y^2)) (Real.atan2 y x) →
  cartesian_eq_C x y →
  |(Real.sqrt (2*12))| * |(Real.sqrt (2*12))| = 6 →
  cartesian_eq_l x y α :=
by
  sorry

end main_theorem_l314_314553


namespace probability_two_from_same_province_l314_314393

theorem probability_two_from_same_province :
  let total_singers := 12
  let selected_singers := 4
  let num_provinces := 6
  let singers_per_province := 2
  let total_ways := Nat.choose total_singers selected_singers
  let favorable_ways := num_provinces * Nat.choose singers_per_province singers_per_province *
                        Nat.choose (total_singers - singers_per_province) (selected_singers - singers_per_province) *
                        (total_singers - singers_per_province - (selected_singers - singers_per_province + 1))
  ∃ (p : ℚ), p = (favorable_ways : ℚ) / (total_ways : ℚ) ∧ p = 16 / 33 := 
by by sorry

end probability_two_from_same_province_l314_314393


namespace polygon_sides_l314_314047

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_l314_314047


namespace courtiers_dog_selection_l314_314654

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l314_314654


namespace find_valid_numbers_l314_314556

-- Prime digit definition
def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Prime two-digit pairs
def is_prime_pair (n : ℕ) : Prop :=
  n = 23 ∨ n = 37 ∨ n = 53 ∨ n = 73

-- Four digit number where digits are (A, B, C, D)
def is_valid_number (ABCD : ℕ) : Prop :=
  let A := ABCD / 1000,
      B := (ABCD / 100) % 10,
      C := (ABCD / 10) % 10,
      D := ABCD % 10 in
  is_prime_digit A ∧ is_prime_digit B ∧ is_prime_digit C ∧ is_prime_digit D ∧
  is_prime_pair (A * 10 + B) ∧ is_prime_pair (B * 10 + C) ∧ is_prime_pair (C * 10 + D)

-- The valid four-digit numbers
def valid_numbers : list ℕ := [2373, 3737, 5373, 7373]

theorem find_valid_numbers : 
  (∀ n, is_valid_number n → n ∈ valid_numbers) ∧ 
  (∀ n, n ∈ valid_numbers → is_valid_number n) :=
by
  sorry

end find_valid_numbers_l314_314556


namespace harmonic_mean_sequence_l314_314438

noncomputable def harmonic_mean (P : List ℝ) : ℝ := 
  (P.length : ℝ) / P.sum

theorem harmonic_mean_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) :
  (∀ n, harmonic_mean (List.ofFn (λ i : Fin n, (a (i + 1)))) = 1 / (2 * n + 1)) →
  (∀ n, b n = (a n + 1) / 4) →
  (∑ i in List.range 10, 1 / (b i * b (i + 1))) = 10 / 11 :=
by
  intro hmean hb
  sorry

end harmonic_mean_sequence_l314_314438


namespace angle_measure_l314_314268

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314268


namespace add_base10_to_base3_eq_2100_in_base3_l314_314734

def add_base10_to_base3 : ℕ := 36 + 25 + 2

theorem add_base10_to_base3_eq_2100_in_base3 : add_base10_to_base3 = 2100₃ := by
  sorry

end add_base10_to_base3_eq_2100_in_base3_l314_314734


namespace courtiers_dog_selection_l314_314653

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l314_314653


namespace angle_measure_l314_314270

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314270


namespace equation_of_line_t_intersection_point_l314_314149

def line_s (x : ℝ) : ℝ := (4 / 3) * x - 100
def line_t (x : ℝ) : ℝ := (-3 / 4) * x

theorem equation_of_line_t :
  ∀ x, line_t x = (-3 / 4) * x :=
by sorry

theorem intersection_point :
  ∃ x y, line_s x = y ∧ line_t x = y ∧ x = 48 ∧ y = -36 :=
by sorry

end equation_of_line_t_intersection_point_l314_314149


namespace quadratic_polynomial_l314_314473

theorem quadratic_polynomial (q : ℚ → ℚ) :
  (q(-3) = 0) →
  (q(6) = 0) →
  (q(-1) = -40) →
  q = (λ x, (20 / 7) * x^2 - (60 / 7) * x - (360 / 7)) :=
by
  intros h1 h2 h3
  sorry

end quadratic_polynomial_l314_314473


namespace sharon_total_distance_l314_314613

-- Definitions based on conditions
def miles_per_gallon := 40
def max_gas_tank := 16
def initial_full_tank := 16
def first_leg_distance := 480
def gas_bought := 6
def quarter_tank := max_gas_tank / 4

-- The property to prove
theorem sharon_total_distance : 
  let consumption_first_leg := first_leg_distance / miles_per_gallon in
  let remaining_gas_after_first_leg := initial_full_tank - consumption_first_leg in
  let gas_after_refill := remaining_gas_after_first_leg + gas_bought in
  let gas_used_second_leg := gas_after_refill - quarter_tank in
  let second_leg_distance := gas_used_second_leg * miles_per_gallon in
  let total_distance := first_leg_distance + second_leg_distance in
  total_distance = 720 := 
by 
  sorry

end sharon_total_distance_l314_314613


namespace polynomial_evaluation_ge_three_pow_n_l314_314147

theorem polynomial_evaluation_ge_three_pow_n
  (n : ℕ)
  (a : Fin n → ℝ)
  (h_nonneg : ∀ i, 0 ≤ a i)
  (h_roots : ∀ (x : ℝ), (Polynomial.coeff (Polynomial.C (x^n + (Fin.sum_univ n (λ i, a i * x ^ (n - 1 - i)))) 0 = 0) 
  (roots : Fin n → ℝ)) :
  (∃ roots, ∀ i, Polynomial.eval x (Polynomial.of_sum roots) = 0) → Polynomial.eval 2 (Polynomial.sum roots) >= 3 ^ n := 
begin
  sorry
end

end polynomial_evaluation_ge_three_pow_n_l314_314147


namespace courtiers_selection_l314_314670

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l314_314670


namespace angle_triple_complement_l314_314262

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314262


namespace projection_onto_plane_Q_l314_314089

-- Definitions related to the conditions
def origin : ℝ × ℝ × ℝ := (0, 0, 0)

def v1 : ℝ × ℝ × ℝ := (7, 1, 8)
def proj_v1 : ℝ × ℝ × ℝ := (6, 3, 2)

def v2 : ℝ × ℝ × ℝ := (6, 2, 9)
def result : ℝ × ℝ × ℝ := (9/2, 5, 9/2)

-- Condition: vector from v1 to proj_v1 determines the normal vector to plane Q
def normal_vector (v1 proj_v1 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((v1.1 - proj_v1.1), (v1.2 - proj_v1.2), (v1.3 - proj_v1.3))

def n : ℝ × ℝ × ℝ := normal_vector v1 proj_v1

-- Verify the projection condition
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def scalar_proj (v n : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v n / dot_product n n

def vector_scalar_mul (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

def vector_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

def projected_vector (v n : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  vector_sub v (vector_scalar_mul (scalar_proj v n) n)

-- Lean 4 statement for proof
theorem projection_onto_plane_Q : projected_vector v2 n = result := by
  sorry

end projection_onto_plane_Q_l314_314089


namespace angle_triple_complement_l314_314197

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314197


namespace range_of_b_l314_314479

theorem range_of_b:
  (∀ m : ℝ, ∃ x y : ℝ, x^2 + 2 * y^2 = 3 ∧ y = m * x + b) ↔ (b ∈ set.Icc (- Real.sqrt (3 / 2)) (Real.sqrt (3 / 2))) :=
by
  sorry

end range_of_b_l314_314479


namespace part1_part2_l314_314973

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314973


namespace angle_measure_triple_complement_l314_314311

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314311


namespace part1_solution_set_part2_range_l314_314989

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314989


namespace tangent_line_perpendicular_slope_l314_314541

theorem tangent_line_perpendicular_slope (a : ℝ) :
  (∃ k : ℝ, ∀ (x : ℝ), x > 0 → k = ln x + 1 ∧ (k = 2) ∧ (-1 / a) * k = -1) → a = 2 := 
sorry

end tangent_line_perpendicular_slope_l314_314541


namespace shed_width_l314_314811

theorem shed_width (backyard_length backyard_width shed_length area_needed : ℝ)
  (backyard_area : backyard_length * backyard_width = 260)
  (sod_area : area_needed = 245)
  (shed_dim : shed_length = 3) :
  (backyard_length * backyard_width - area_needed) / shed_length = 5 :=
by
  -- We need to prove the width of the shed given the conditions
  sorry

end shed_width_l314_314811


namespace none_of_these_l314_314448

def y_values_match (f : ℕ → ℕ) : Prop :=
  f 0 = 200 ∧ f 1 = 140 ∧ f 2 = 80 ∧ f 3 = 20 ∧ f 4 = 0

theorem none_of_these :
  ¬ (∃ f : ℕ → ℕ, 
    (∀ x, f x = 200 - 15 * x ∨ 
    f x = 200 - 20 * x + 5 * x^2 ∨ 
    f x = 200 - 30 * x + 10 * x^2 ∨ 
    f x = 150 - 50 * x) ∧ 
    y_values_match f) :=
by sorry

end none_of_these_l314_314448


namespace math_problem_l314_314618

theorem math_problem (a b c : ℝ) (h1 : a = 567.89) (h2 : b = 123.45) (h3 : c = 3) :
  (Float.round ((a - b) * c * 100) / 100) = 1333.32 :=
by
  -- The proof would go here
  sorry

end math_problem_l314_314618


namespace minimum_group_members_round_table_l314_314777

theorem minimum_group_members_round_table (n : ℕ) (h1 : ∀ (a : ℕ),  a < n) : 5 ≤ n :=
by
  sorry

end minimum_group_members_round_table_l314_314777


namespace num_correct_propositions_zero_l314_314418

theorem num_correct_propositions_zero :
  (∀ {A B : ℝ}, ∃ (R : ℝ), 2 * R * sin A > 2 * R * sin B ↔ sin A > sin B → false) ∧
  (∀ {f : ℝ → ℝ}, (∃ (x : ℝ), (1 < x) ∧ (x < 2) ∧ (f x = 0)) ↔ f 1 * f 2 < 0 → false) ∧
  (∀ (a₁ a₅ : ℝ), a₁ = 1 ∧ a₅ = 16 → a₃ = 4 ↔ a₃ = ±4 → false) ∧
  (∀ {x : ℝ}, (y = sin (2 - 2 * x)) → (y = sin (2 - 2 * (x - 2))) = sin (6 - 2 * x) ↔ (y = sin (4 - 2 * x)) → false) :=
sorry

end num_correct_propositions_zero_l314_314418


namespace part1_solution_set_part2_range_l314_314984

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314984


namespace sallys_change_l314_314121

-- Given conditions
def frames_bought : ℕ := 3
def cost_per_frame : ℕ := 3
def payment : ℕ := 20

-- The statement to prove
theorem sallys_change : payment - (frames_bought * cost_per_frame) = 11 := by
  sorry

end sallys_change_l314_314121


namespace cost_of_clarinet_is_125_l314_314621

noncomputable def cost_of_clarinet : ℕ :=
  let initial_savings := 10
  let price_per_book := 5
  let total_books_sold := 25
  let halfway_savings (C : ℕ) := C / 2
  let total_money_from_books := total_books_sold * price_per_book
  total_money_from_books

theorem cost_of_clarinet_is_125 (C : ℕ) :
  let halfway_savings := C / 2 in
  let total_money_from_books := 25 * 5 in
  halfway_savings + halfway_savings = C →
  total_money_from_books = C →
  C = 125 :=
by
  intros
  sorry

end cost_of_clarinet_is_125_l314_314621


namespace fraction_addition_simplest_form_l314_314817

theorem fraction_addition_simplest_form :
  (7 / 8) + (3 / 5) = 59 / 40 :=
by sorry

end fraction_addition_simplest_form_l314_314817


namespace unit_vectors_collinear_with_a_l314_314015

open Real

def vector_a : ℝ × ℝ × ℝ := (1, 2, -2)

def unit_vector (v : ℝ × ℝ × ℝ) : Prop := 
  let mag := sqrt (v.1^2 + v.2^2 + v.3^2) in 
  mag = 1

theorem unit_vectors_collinear_with_a :
  ∃ u : ℝ × ℝ × ℝ, unit_vector u ∧ (u = (1/3, 2/3, -2/3) ∨ u = (-1/3, -2/3, 2/3)) ∧
  ∃ k : ℝ, k ≠ 0 ∧ ∀ i, u.1 = k * vector_a.1 ∧ u.2 = k * vector_a.2 ∧ u.3 = k * vector_a.3 :=
sorry

end unit_vectors_collinear_with_a_l314_314015


namespace circumcenter_is_orthocenter_of_MNL_l314_314568

-- Definitions of the problem elements
variables (ABC : Type*)
variables [triangle ABC]
variables (R r_a : ℝ)
variables [circumcircle_radius ABC R]
variables [A_excircle_radius ABC r_a]
variables (M N L : ABC)
variables [A_excircle_touches_BC_at_M ABC M]
variables [A_excircle_touches_AC_at_N ABC N]
variables [A_excircle_touches_AB_at_L ABC L]
variables (O : ABC)
variables [is_circumcenter ABC O]

-- Definition and theorem of interest
theorem circumcenter_is_orthocenter_of_MNL
  (h : R = r_a) :
  is_orthocenter M N L O := sorry

end circumcenter_is_orthocenter_of_MNL_l314_314568


namespace ladder_slip_l314_314379

theorem ladder_slip 
  (ladder_length : ℝ) 
  (initial_base : ℝ) 
  (slip_height : ℝ) 
  (h_length : ladder_length = 30) 
  (h_base : initial_base = 11) 
  (h_slip : slip_height = 6) 
  : ∃ (slide_distance : ℝ), abs (slide_distance - 9.49) < 0.01 :=
by
  let initial_height := Real.sqrt (ladder_length^2 - initial_base^2)
  let new_height := initial_height - slip_height
  let new_base := Real.sqrt (ladder_length^2 - new_height^2)
  let slide_distance := new_base - initial_base
  use slide_distance
  have h_approx : abs (slide_distance - 9.49) < 0.01 := sorry
  exact h_approx

end ladder_slip_l314_314379


namespace angle_is_67_l314_314215

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314215


namespace area_increase_l314_314745

variable (l w : ℝ)

theorem area_increase (hl : l > 0) (hw : w > 0) :
  let A := l * w
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  (A_new - A) / A * 100 = 56 := by
  let A := l * w
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  have : A_new = 1.56 * A := by
    rw [←mul_assoc, ←mul_assoc, mul_comm 1.3 l, mul_assoc, mul_comm 1.2 w, mul_assoc, mul_comm w l, ←mul_assoc]
    ring
  calc
    (A_new - A) / A * 100 = ((1.56 * A) - A) / A * 100 := by rw this
    ... = (1.56 - 1) * 100 := by field_simp [A, lt_mul_iff_one_lt_left (lt_of_lt_of_le (zero_lt_one) hl), ne_of_gt hl]
    ... = 0.56 * 100 := by ring
    ... = 56 := by norm_num

end area_increase_l314_314745


namespace part1_part2_l314_314940

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314940


namespace farm_area_l314_314720

theorem farm_area (length width area : ℝ) 
  (h1 : length = 0.6) 
  (h2 : width = 3 * length) 
  (h3 : area = length * width) : 
  area = 1.08 := 
by 
  sorry

end farm_area_l314_314720


namespace divisibility_of_n_squared_plus_n_plus_two_l314_314877

-- Definition: n is a natural number.
def n (n : ℕ) : Prop := True

-- Theorem: For any natural number n, n^2 + n + 2 is always divisible by 2, but not necessarily divisible by 5.
theorem divisibility_of_n_squared_plus_n_plus_two (n : ℕ) : 
  (∃ k : ℕ, n^2 + n + 2 = 2 * k) ∧ (¬ ∃ m : ℕ, n^2 + n + 2 = 5 * m) :=
by
  sorry

end divisibility_of_n_squared_plus_n_plus_two_l314_314877


namespace angle_measure_triple_complement_l314_314217

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314217


namespace area_percent_of_circle_l314_314357

theorem area_percent_of_circle (D_s : ℝ) (h: D_s > 0) : 
  let D_r := 0.8 * D_s in
  let R_s := D_s / 2 in
  let R_r := D_r / 2 in
  let A_s := Real.pi * R_s^2 in
  let A_r := Real.pi * R_r^2 in
  (A_r / A_s) * 100 = 64 :=
by
  let D_r := 0.8 * D_s
  let R_s := D_s / 2
  let R_r := D_r / 2
  let A_s := Real.pi * R_s^2
  let A_r := Real.pi * R_r^2
  have h1 : R_r = 0.8 * R_s := by
    rw [R_r, D_r, R_s]
    exact (mul_div_assoc' 0.8 D_s 2).symm
  have h2 : A_r = 0.64 * A_s := by
    rw [A_r, A_s, h1]
    exact (mul_pow 0.8 R_s 2).trans (by
      simp [sq])
  calc
  (A_r / A_s) * 100 = (0.64 * A_s / A_s) * 100 : by rw [h2]
                ... = 0.64 * 100            : by
                  rw [mul_div_cancel_left _ (by linarith : A_s ≠ 0)]
                ... = 64                    : by ring

end area_percent_of_circle_l314_314357


namespace complete_the_square_sum_l314_314595

theorem complete_the_square_sum :
  ∃ p q : ℝ, (∀ x : ℝ, 15 * x^2 - 30 * x - 60 = 0 → (x + p)^2 = q) ∧ p + q = 1 :=
by 
  sorry

end complete_the_square_sum_l314_314595


namespace sailors_release_strategy_exists_l314_314624

/-- A strategy can guarantee the release of 11 sailors if each sailor has a unique integer from 0 to 10 modulo 11
    and the total sum of numbers on their foreheads within modulo 11 ensures at least one correct guess. -/
theorem sailors_release_strategy_exists :
  ∃ (k : Fin 11 → ℕ), (∀ i : Fin 11, k i < 11) ∧
  (∀ (n : Fin 11 → ℕ), (∀ i, 1 ≤ n i ∧ n i ≤ 11) →
    ∃ i, (n i + (∑ j in Finset.univ.filter (λ j, j ≠ i), n j)) % 11 = (∑ j, n j) % 11) :=
by
  sorry

end sailors_release_strategy_exists_l314_314624


namespace part1_part2_l314_314998

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l314_314998


namespace two_courtiers_have_same_selection_l314_314683

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l314_314683


namespace winner_is_C_l314_314804

theorem winner_is_C 
  (won : α → Prop)
  (A B C D : α)
  (H1 : ∃! x, won x)
  (H2 : (won B ∨ won C) ↔ true)  -- A's statement
  (H3 : ¬won A ∧ ¬won C ↔ true)  -- B's statement
  (H4 : won C ↔ true)            -- C's statement
  (H5 : won B ↔ true)            -- D's statement
  (H6 : (H2 ∨ H3 ∧ H4) ∧ (H4 ∨ H5 ∧ H2) ↔ two_true) : won C := 
sorry

end winner_is_C_l314_314804


namespace max_value_u_l314_314478

theorem max_value_u (x y z : ℝ) (h : 2 * x + 3 * y + 5 * z = 29) : 
  (∃ u, u = sqrt (2 * x + 1) + sqrt (3 * y + 4) + sqrt (5 * z + 6) ∧ u = 2 * sqrt 30) :=
by
  sorry

end max_value_u_l314_314478


namespace certain_number_exists_l314_314020

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

theorem certain_number_exists :
  ∃ n, n = 20 ∧ (let start := factorial 10 in 
                 ∃ l, l = [start, start + 5, start + 10, start + 15, start + 20] ∧
                     l.length = 5 ∧
                     ∀ x ∈ l, x % 5 = 0) :=
by
  sorry

end certain_number_exists_l314_314020


namespace arrange_numbers_l314_314425

theorem arrange_numbers :
  (2 : ℝ) ^ 1000 < (5 : ℝ) ^ 500 ∧ (5 : ℝ) ^ 500 < (3 : ℝ) ^ 750 :=
by
  sorry

end arrange_numbers_l314_314425


namespace problem_statement_l314_314581

theorem problem_statement (n : ℕ) (h1 : 0 < n) (h2 : ∃ k : ℤ, (1/2 + 1/3 + 1/11 + 1/n : ℚ) = k) : ¬ (n > 66) := 
sorry

end problem_statement_l314_314581


namespace termites_ate_12_black_squares_l314_314139

def is_black_square (r c : ℕ) : Prop :=
  (r + c) % 2 = 0

def eaten_positions : List (ℕ × ℕ) :=
  [(3, 1), (4, 6), (3, 7), (4, 1), (2, 3), (2, 4), (4, 3), (3, 5), (3, 2), (4, 7), (3, 6), (2, 6)]

def count_black_squares (positions : List (ℕ × ℕ)) : ℕ :=
  positions.countp (λ ⟨r, c⟩ => is_black_square r c)

theorem termites_ate_12_black_squares : count_black_squares eaten_positions = 12 := by
  sorry

end termites_ate_12_black_squares_l314_314139


namespace find_ab_exponent_l314_314482

theorem find_ab_exponent (a b : ℝ) 
  (h : |a - 2| + (b + 1 / 2)^2 = 0) : 
  a^2022 * b^2023 = -1 / 2 := 
sorry

end find_ab_exponent_l314_314482


namespace sallys_change_l314_314119

-- Define the total cost calculation:
def totalCost (numFrames : Nat) (costPerFrame : Nat) : Nat :=
  numFrames * costPerFrame

-- Define the change calculation:
def change (totalAmount : Nat) (amountPaid : Nat) : Nat :=
  amountPaid - totalAmount

-- Define the specific conditions in the problem:
def numFrames := 3
def costPerFrame := 3
def amountPaid := 20

-- Prove that the change Sally gets is $11:
theorem sallys_change : change (totalCost numFrames costPerFrame) amountPaid = 11 := by
  sorry

end sallys_change_l314_314119


namespace part1_solution_part2_solution_l314_314959

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314959


namespace angle_triple_complement_l314_314318

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314318


namespace courtier_selection_l314_314677

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l314_314677


namespace Petya_can_determine_swap_l314_314115

open Finset

-- Define the cards and their positions
def initial_card_positions : Fin 9 → ℕ
| 0 := 8 | 1 := 1 | 2 := 2
| 3 := 3 | 4 := 5 | 5 := 9
| 6 := 6 | 7 := 7 | 8 := 4

-- Define the gray card positions
def gray_positions : Finset (Fin 9) := {1, 3, 7, 5}.to_finset

-- Define the initial sum of the gray positions
def initial_sum : ℕ := gray_positions.sum initial_card_positions

-- Define a function for sum after swap
def current_sum (f : Fin 9 → ℕ) : ℕ :=
  gray_positions.sum f

-- Problem statement: Petya can determine the exact swap
theorem Petya_can_determine_swap (swap : (Fin 9 → ℕ) → (Fin 9 → ℕ))
    (h_swap : ∃ (i j : Fin 9), i ≠ j ∧ ((λ f, swap f) = (λ f, f ∘ equiv.swap i j))) :
    initial_sum = current_sum initial_card_positions → ∃ f', current_sum (swap initial_card_positions) = current_sum f' → (∃ i j, i ≠ j ∧ (initial_card_positions i = f' j ∧ initial_card_positions j = f' i)) :=
by
  sorry

end Petya_can_determine_swap_l314_314115


namespace minimum_of_f_in_domain_l314_314856

-- The function f(x, y)
def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

-- The domain bounds for x and y
def x_bounds (x : ℝ) : Prop := (1/4 : ℝ) ≤ x ∧ x ≤ (3/5 : ℝ)
def y_bounds (y : ℝ) : Prop := (1/5 : ℝ) ≤ y ∧ y ≤ (2/3 : ℝ)

-- The goal is to prove that the minimum value of the function in the domain is 24/73
theorem minimum_of_f_in_domain :
  ∃ (x y : ℝ), x_bounds x ∧ y_bounds y ∧ (∀ (x' y' : ℝ), x_bounds x' ∧ y_bounds y' → f x y ≤ f x' y') ∧ f x y = 24 / 73 := 
sorry

end minimum_of_f_in_domain_l314_314856


namespace part1_part3_l314_314488

noncomputable theory

-- Define the sequence {a_n} with given conditions
def a : ℕ → ℝ
| 0     := 0  -- Base case for a_0 (ignored / not used)
| 1     := 2
| (n+1) := 2^n * (n + 1)

-- Define the sum of the first n terms {S_n}
def S : ℕ → ℝ
| 0     := 0
| (n+1) := (finset.range (n+1)).sum a

-- condition: (n+2)S_{n}=na_{n+1}
axiom S_n_condition (n : ℕ) : (n + 2) * S n = n * a (n + 1)

-- Define the sequence {b_n}
def b (n : ℕ) : ℝ := S n / n

-- Define the sequence {c_n}
def c (n : ℕ) : ℝ := 4^n / (a n)^2

-- Define the sum of the first n terms of {c_n} denoted as T_n
def T (n : ℕ) : ℝ := (finset.range n).sum c

-- Part (1): b_n is a geometric sequence
theorem part1 (n : ℕ) : b (n + 1) = 2 * b n := sorry

-- Part (2): General formula for a_n
lemma a_formula {n : ℕ} : a (n + 1) = (n + 2) * 2^n := sorry

-- Part (3): T_n < 4
theorem part3 (n : ℕ) : T n < 4 := sorry

end part1_part3_l314_314488


namespace magnitude_of_c_l314_314523

variables (a b c : ℝ × ℝ)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem magnitude_of_c
  (ha : a = (-1, 2))
  (hb : b = (3, 4))
  (hc_parallel : ∃ λ : ℝ, c = (-λ, 2λ))
  (ha_ortho : ∀ (b_plus_c : ℝ × ℝ), b_plus_c = (b.1 + c.1, b.2 + c.2) -> a.1 * b_plus_c.1 + a.2 * b_plus_c.2 = 0)
  : magnitude c = real.sqrt 5 :=
sorry

end magnitude_of_c_l314_314523


namespace general_formula_a_sum_first_n_terms_b_l314_314555

-- Define the arithmetic sequence {a_n} and its properties
def a (n : ℕ) : ℕ := 2 * n - 1

-- Define the sequence {b_n} based on the given conditions
def b (n : ℕ) : ℕ :=
if n % 2 = 1 then a ((n + 1) / 2) else 2 ^ (n / 2 - 1)

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) : ℕ :=
if n % 2 = 0 then (n ^ 2) / 4 + 2 ^ (n / 2) - 1
else (n ^ 2 + 2 * n - 3) / 4 + 2 ^ ((n - 1) / 2)

-- The theorem statements
theorem general_formula_a (n : ℕ) : a n = 2 * n - 1 := sorry

theorem sum_first_n_terms_b (n : ℕ) : 
  ∑ i in Finset.range n, b (i + 1) = if n % 2 = 0 then (n ^ 2) / 4 + 2 ^ (n / 2) - 1
                                      else (n ^ 2 + 2 * n - 3) / 4 + 2 ^ ((n - 1) / 2) := sorry

end general_formula_a_sum_first_n_terms_b_l314_314555


namespace drilling_probability_l314_314549

theorem drilling_probability (A_sea A_oil : ℕ) (h_A_sea : A_sea = 10000) (h_A_oil : A_oil = 40) : 
  (A_oil.to_rat / A_sea.to_rat) = (0.004 : ℚ) :=
by
  sorry

end drilling_probability_l314_314549


namespace arc_length_of_sector_l314_314040

theorem arc_length_of_sector (θ r : ℝ) (h1 : θ = 120) (h2 : r = 2) : 
  (θ / 360) * (2 * Real.pi * r) = (4 * Real.pi) / 3 :=
by
  sorry

end arc_length_of_sector_l314_314040


namespace total_pieces_equiv_231_l314_314436

-- Define the arithmetic progression for rods.
def rods_arithmetic_sequence : ℕ → ℕ
| 0 => 0
| n + 1 => 3 * (n + 1)

-- Define the sum of the first 10 terms of the sequence.
def rods_total (n : ℕ) : ℕ :=
  let a := 3
  let d := 3
  n / 2 * (2 * a + (n - 1) * d)

def rods_count : ℕ :=
  rods_total 10

-- Define the 11th triangular number for connectors.
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def connectors_count : ℕ :=
  triangular_number 11

-- Define the total number of pieces.
def total_pieces : ℕ :=
  rods_count + connectors_count

-- The theorem we aim to prove.
theorem total_pieces_equiv_231 : total_pieces = 231 := by
  sorry

end total_pieces_equiv_231_l314_314436


namespace find_n_l314_314462

theorem find_n (n : ℤ) : -180 ≤ n ∧ n ≤ 180 ∧ (Real.sin (n * Real.pi / 180) = Real.cos (690 * Real.pi / 180)) → n = 60 :=
by
  intro h
  sorry

end find_n_l314_314462


namespace range_of_a_l314_314495

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x > 3 → x > a)) ↔ (a ≤ 3) :=
sorry

end range_of_a_l314_314495


namespace sin_cos_product_l314_314030

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l314_314030


namespace part1_solution_set_part2_range_a_l314_314949

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314949


namespace anthony_pets_left_l314_314422

theorem anthony_pets_left : 
  let original_pets := 16 in
  let lost_pets := 6 in
  let died_pets_fraction := 1/5 in
  let remaining_pets_after_loss := original_pets - lost_pets in
  let died_pets := (remaining_pets_after_loss * died_pets_fraction).toNat in
  let remaining_pets := remaining_pets_after_loss - died_pets in
  remaining_pets = 8 := 
by sorry

end anthony_pets_left_l314_314422


namespace g_neg_one_l314_314900

variables {F : Type*} [Field F]

def odd_function (f : F → F) := ∀ x, f (-x) = -f x

variables (f : F → F) (g : F → F)

-- Given conditions
lemma given_conditions :
  (∀ x, f (-x) + (-x)^2 = -(f x + x^2)) ∧
  f 1 = 1 ∧
  (∀ x, g x = f x + 2) :=
sorry

-- Prove that g(-1) = -1
theorem g_neg_one :
  g (-1) = -1 :=
sorry

end g_neg_one_l314_314900


namespace angle_measure_triple_complement_l314_314223

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314223


namespace proper_subset_count_l314_314013

open Set

def M : Set ℤ := {-1, 0}

def N : Set ℝ := {y | ∃ x ∈ M, y = 1 - Real.cos (Real.pi / 2 * x)}

theorem proper_subset_count : ∃ (n : ℕ), n = card {S | S ⊂ (M ∩ N) } ∧ n = 3 := by
  sorry

end proper_subset_count_l314_314013


namespace part1_part2_l314_314935

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314935


namespace find_distinct_natural_numbers_l314_314191

theorem find_distinct_natural_numbers :
  ∃ (x y : ℕ), x ≥ 10 ∧ y ≠ 1 ∧
  (x * y + x) + (x * y - x) + (x * y * x) + (x * y / x) = 576 :=
by
  sorry

end find_distinct_natural_numbers_l314_314191


namespace half_way_fraction_l314_314731

def half_way_between (a b : ℚ) : ℚ := (a + b) / 2

theorem half_way_fraction : 
  half_way_between (1/3) (3/4) = 13/24 :=
by 
  -- Proof follows from the calculation steps, but we leave it unproved.
  sorry

end half_way_fraction_l314_314731


namespace area_of_disks_union_lt_035_l314_314449

theorem area_of_disks_union_lt_035 
  (D : Set (Set Point)) -- D represents the set of discs
  (d : Point → ℝ)       -- d represents the diameter function for discs
  (A : ℝ)               -- A represents the area of the union
  (lt_d : ∀ (p : Point), d p < 0.02)   -- condition for diameters less than 0.02
  (sq : Set Point)      -- sq represents the square
  (in_sq : ∀ (p : Point), p ∈ D → p ∈ sq) -- condition representing placement in the square
  (dist_cond : ∀ (p q : Point), (p ∈ D → q ∈ D → dist(p, q) ≠ 0.02)) -- condition for distance
  : A < 0.35 := sorry -- proving that the union area is less than 0.35

end area_of_disks_union_lt_035_l314_314449


namespace find_a_given_coefficient_l314_314042

theorem find_a_given_coefficient (a : ℝ) (h : (a^3 * 10 = 80)) : a = 2 :=
by
  sorry

end find_a_given_coefficient_l314_314042


namespace angle_measure_triple_complement_l314_314226

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314226


namespace diagonal_AC_circumscribed_quadrilateral_l314_314630

theorem diagonal_AC_circumscribed_quadrilateral (A B C D : Type*) [MetricSpace A]
  (BD_circumscribed : ∀ (x : A), ∥x - B∥ = ∥x - D∥)
  (H_BD : ∥B - D∥ = 2)
  (H_AB : ∥A - B∥ = 1)
  (angle_ratio : ∠ ABD = 4 / 7 * ∠ DBC) :
  ∥A - C∥ = (√2 + √6) / 2 :=
sorry

end diagonal_AC_circumscribed_quadrilateral_l314_314630


namespace rhombus_parallel_lines_l314_314055

variable (A B C D M N P Q E F G H O : Point)
variable (AB BC CD DA MN PQ : Line)
variable [Rhombus ABCD]
variable [IncircleO ABCD]

-- Given that ABCD is a rhombus and O is the incircle touching the sides at E, F, G, H respectively
-- Tangents MN and PQ are drawn as specified, intersecting sides at M, N, P, Q
theorem rhombus_parallel_lines 
  (h_tangent_M : Tangent O E F MN)
  (h_tangent_P : Tangent O G H PQ)
  (h_M_AB : OnLine M AB)
  (h_N_BC : OnLine N BC)
  (h_P_CD : OnLine P CD)
  (h_Q_DA : OnLine Q DA) 
  : Parallel M Q N P := sorry

end rhombus_parallel_lines_l314_314055


namespace angle_measure_triple_complement_l314_314309

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314309


namespace find_constant_d_l314_314036

theorem find_constant_d (d : ℚ) :
  (∃ f : ℚ → ℚ, f = λ x, d * x^4 - 4 * x^3 + 17 * x^2 - 5 * d * x + 60 ∧ (x - 5) ∣ f x) →
  d = 173 / 130 :=
by
  intros h
  simp at h
  sorry

end find_constant_d_l314_314036


namespace angle_measure_triple_complement_l314_314225

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314225


namespace domain_of_f_comp_l314_314045

theorem domain_of_f_comp (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ x^2 - 2 ∧ x^2 - 2 ≤ -1) →
  (∀ x, - (4 : ℝ) / 3 ≤ x ∧ x ≤ -1 → -2 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ -1) :=
by
  sorry

end domain_of_f_comp_l314_314045


namespace volume_tetrahedron_PDEF_l314_314558

noncomputable theory
open_locale classical

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Tetrahedron :=
(vertices : list Point3D) 
(regular : vertices.length = 4) 

def distance (p1 p2 : Point3D) : ℝ :=
real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

theorem volume_tetrahedron_PDEF (P A B C D E F : Point3D)
  (T : Tetrahedron) (is_regular : T = Tetrahedron.mk [P, A, B, C] T.regular)
  (D_on_PA : ∃ t ∈ Icc (0 : ℝ) 1, D = ⟨P.x + t * (A.x - P.x), P.y + t * (A.y - P.y), P.z + t * (A.z - P.z)⟩)
  (E_on_PB : ∃ t ∈ Icc (0 : ℝ) 1, E = ⟨P.x + t * (B.x - P.x), P.y + t * (B.y - P.y), P.z + t * (B.z - P.z)⟩)
  (F_on_PC : ∃ t ∈ Icc (0 : ℝ) 1, F = ⟨P.x + t * (C.x - P.x), P.y + t * (C.y - P.y), P.z + t * (C.z - P.z)⟩)
  (PE_ne_PF : distance P E ≠ distance P F)
  (DE_eq_sqrt7 : distance D E = real.sqrt 7)
  (DF_eq_sqrt7 : distance D F = real.sqrt 7)
  (EF_eq_2 : distance E F = 2) :
  volume_tetrahedron P D E F = real.sqrt 17 / 8 :=
sorry

end volume_tetrahedron_PDEF_l314_314558


namespace k_value_if_divisible_l314_314358

theorem k_value_if_divisible :
  ∀ k : ℤ, (x^2 + k * x - 3) % (x - 1) = 0 → k = 2 :=
by
  intro k
  sorry

end k_value_if_divisible_l314_314358


namespace angle_triple_complement_l314_314241

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314241


namespace pencils_distribution_l314_314453

/--
Eric initially sorted 150 colored pencils into 5 containers. He received 30 more pencils and then 
received 47 additional pencils after finding a sixth container. Prove that he can evenly distribute 
37 pencils per container between the six containers.
-/
theorem pencils_distribution :
  let initial_pencils := 150 in
  let additional_pencils_before_class := 30 in
  let additional_pencils_after_finding_sixth_container := 47 in
  let total_containers := 6 in
  let total_pencils := initial_pencils + additional_pencils_before_class + additional_pencils_after_finding_sixth_container in
  (total_pencils / total_containers) = 37 :=
by
  sorry

end pencils_distribution_l314_314453


namespace lottery_prizes_and_probabilities_l314_314546

theorem lottery_prizes_and_probabilities :
  let balls := (2 : Fin 8) ++ (5 : Fin 2) in
  let draws := { p : List Nat // p.length = 3 ∧ (∀ x ∈ p, x ∈ balls.to_list)} in
  let sum_draws := draws.map (λ p => p.val.sum) in
  let prize_counts := sum_draws.foldl (λ acc x => acc.insert x (acc.find x + 1)) (RBMap.empty _ _) in
  let total_draws := C(10, 3) in
  prize_counts.map (λ (prize, count) => (prize, count / total_draws)) = [(6, 7/15), (9, 7/15), (12, 1/15)] :=
sorry

end lottery_prizes_and_probabilities_l314_314546


namespace triple_complement_angle_l314_314282

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314282


namespace range_of_a_l314_314542

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l314_314542


namespace find_side_AB_l314_314704

-- Define a triangle ABC with a given perimeter
structure Triangle :=
  (A B C : Point)
  (perimeter : ℝ)

-- Define conditions for the problem
def triangle_conditions (T : Triangle) : Prop :=
  T.perimeter = 8 ∧
  ∃ incircle : Incircle T, ∃ t : Tangent, t.parallel_to T.AB ∧ t.segment_length = 1

-- Define the theorem to find the length of side AB
theorem find_side_AB (T : Triangle) (h : triangle_conditions T) : length (T.AB) = 2 :=
  sorry

end find_side_AB_l314_314704


namespace tangent_line_slope_min_area_parabola_line_segment_min_distance_parabola_line_l314_314572

open Real

-- Problem (1)
theorem tangent_line_slope (a : ℝ) (ha : 0 < a) : 
  ∃ k_l, k_l = 2 * (a + sqrt (a^2 - a + 1)) := sorry

-- Problem (2)
theorem min_area_parabola_line_segment (a : ℝ) (ha : 0 < a) :
  (∃ (S_min: ℝ), S_min = (2*(a + sqrt(a^2 - a + 1))*sqrt(a^2 - a + 1)^(1/2)/3) ) := sorry

-- Problem (3)
theorem min_distance_parabola_line : 
  ∃ (d_min : ℝ), d_min = (3 * sqrt 2) / 8 := sorry

end tangent_line_slope_min_area_parabola_line_segment_min_distance_parabola_line_l314_314572


namespace proof_problem_l314_314892

/-- Given an ellipse equation Γ: x^2 / a^2 + y^2 / b^2 = 1 (a > b > 0), and a line l: x + y - 4 * sqrt 2 = 0,
    and the lower endpoint of Γ is A, M is on l, and the left and right foci are F1(-sqrt 2, 0) and F2(sqrt 2, 0).
    Prove:
    1. When a = 2 and the midpoint of AM is on the x-axis, find the coordinates of point M.
    2. If line l intersects the y-axis at point B, line AM passes through the right focus F2, and in triangle ABM
       one of the interior angles has a cosine value of 3/5, find b.
    3. There exists a point P on the ellipse Γ such that the distance from P to l is d and there exists a point on 
       Γ such that |PF1| + |PF2| + d = 6. As a varies, find the minimum value of d.
-/
theorem proof_problem :
  let a := 2
  let Γ : Set (ℝ × ℝ) := {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  let l : Set (ℝ × ℝ) := {p | p.1 + p.2 - 4 * Real.sqrt 2 = 0}
  let A : (ℝ × ℝ) := (0, -Real.sqrt 2)
  let F1 : (ℝ × ℝ) := (-Real.sqrt 2, 0)
  let F2 : (ℝ × ℝ) := (Real.sqrt 2, 0)
  
  ∃ M : (ℝ × ℝ), M = (3 * Real.sqrt 2, Real.sqrt 2) ∧ 
  (Γ = {p | p.1^2 / 4 + p.2^2 / 2 = 1}) ∧ 
  (l = {p | p.1 + p.2 - 4 * Real.sqrt 2 = 0}) ∧
  let B : (ℝ × ℝ) := (0, 4 * Real.sqrt 2),
  let cos_angle : ℝ := 3 / 5,
  (∃ b : ℝ, b = (3 / 4) * Real.sqrt 2 ∨ b = Real.sqrt 2 / 7) ∧
  let P : (ℝ × ℝ) := (a * Real.cos θ, b * Real.sin θ), 
  let dist : ℝ := |P.1 + P.2 - 4 * Real.sqrt 2| / Real.sqrt 2,
  let d_min := 8 / 3, 
  (∃ P, |P.1 - F1.1| + |P.2 - F2.2| + dist = 6 ∧ dist = d_min ) sorry

end proof_problem_l314_314892


namespace max_f_value_range_of_a_l314_314009

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - 4| - a

theorem max_f_value (a : ℝ) : ∃ x, f x a = 5 - a :=
sorry

theorem range_of_a (a : ℝ) : (∃ x, f x a ≥ (4 / a) + 1) ↔ (a = 2 ∨ a < 0) :=
sorry

end max_f_value_range_of_a_l314_314009


namespace sequence_x_value_l314_314516

noncomputable def x_value : ℕ :=
  let seq := [2, 5, 11, 2, x, 47]
  x

theorem sequence_x_value :
  ∃ (x : ℕ), x = 32 ∧ seq = [2, 5, 11, 2, x, 47] :=
begin
  sorry
end

end sequence_x_value_l314_314516


namespace parabola_point_distance_to_x_axis_l314_314786

theorem parabola_point_distance_to_x_axis :
  ∀ (x0 y0 : ℝ), (x0^2 = (1/4) * y0) → ((x0, y0).dist (0, 1 / 16) = 1) → |y0| = 15 / 16 :=
by
  sorry

end parabola_point_distance_to_x_axis_l314_314786


namespace floor_expression_equals_8_l314_314434

theorem floor_expression_equals_8 : 
  let n := 2020 in \[ \left\lfloor \dfrac {2021^3}{2019 \cdot 2020} - \dfrac {2019^3}{2020 \cdot 2021} \right\rfloor = 8.
\] :=
by
  sorry

end floor_expression_equals_8_l314_314434


namespace hunter_cannot_see_rabbit_l314_314428

theorem hunter_cannot_see_rabbit (r : ℝ) (h : ℝ) 
  (h_r : r < 1) 
  (h_h : h > 1/r) :
  ∃ (grid_point : ℤ × ℤ), (grid_point ≠ (0, 0)) ∧ 
   (sqrt ((grid_point.1:ℝ)^2 + (grid_point.2:ℝ)^2) ≤ h + r) := 
sorry

end hunter_cannot_see_rabbit_l314_314428


namespace tenth_day_is_monday_l314_314131

theorem tenth_day_is_monday (runs_20_mins : ∀ d ∈ [1, 7], d = 1 ∨ d = 6 ∨ d = 7 → True)
                            (total_minutes : 5 * 60 = 300)
                            (first_day_is_saturday : 1 = 6) :
   (10 % 7 = 3) :=
by
  sorry

end tenth_day_is_monday_l314_314131


namespace part1_solution_part2_solution_l314_314958

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314958


namespace maximize_revenue_l314_314765

theorem maximize_revenue (p : ℝ) (h1 : p ≤ 25)
  (h2 : ∀ p, revenue p = p * (200 - 8 * p)) :
  maximize revenue p :=
sorry

end maximize_revenue_l314_314765


namespace simplify_fraction_l314_314615

theorem simplify_fraction :
  (5 + 7 * Complex.i) / (3 - 4 * Complex.i) = (43 / 25) + (41 / 25) * Complex.i := 
by 
  sorry

end simplify_fraction_l314_314615


namespace determine_uv_l314_314831

theorem determine_uv :
  ∃ u v : ℝ, (u = 5 / 17) ∧ (v = -31 / 17) ∧
    ((⟨3, -2⟩ : ℝ × ℝ) + u • ⟨5, 8⟩ = (⟨-1, 4⟩ : ℝ × ℝ) + v • ⟨-3, 2⟩) :=
by
  sorry

end determine_uv_l314_314831


namespace calculation1_calculation2_calculation3_calculation4_l314_314430

-- Proving the first calculation: 3 * 232 + 456 = 1152
theorem calculation1 : 3 * 232 + 456 = 1152 := 
by 
  sorry

-- Proving the second calculation: 760 * 5 - 2880 = 920
theorem calculation2 : 760 * 5 - 2880 = 920 :=
by 
  sorry

-- Proving the third calculation: 805 / 7 = 115 (integer division)
theorem calculation3 : 805 / 7 = 115 :=
by 
  sorry

-- Proving the fourth calculation: 45 + 255 / 5 = 96
theorem calculation4 : 45 + 255 / 5 = 96 :=
by 
  sorry

end calculation1_calculation2_calculation3_calculation4_l314_314430


namespace reimbursement_diff_l314_314182

/-- Let Tom, Emma, and Harry share equally the costs for a group activity.
- Tom paid $95
- Emma paid $140
- Harry paid $165
If Tom and Emma are to reimburse Harry to ensure all expenses are shared equally,
prove that e - t = -45 where e is the amount Emma gives Harry and t is the amount Tom gives Harry.
-/
theorem reimbursement_diff :
  let tom_paid := 95
  let emma_paid := 140
  let harry_paid := 165
  let total_cost := tom_paid + emma_paid + harry_paid
  let equal_share := total_cost / 3
  let t := equal_share - tom_paid
  let e := equal_share - emma_paid
  e - t = -45 :=
by {
  sorry
}

end reimbursement_diff_l314_314182


namespace series_convergence_l314_314072

section SeriesConvergence

noncomputable def series (x : ℝ) : ℕ → ℝ :=
  λ n, (n + 1) * (x ^ (n + 1)) / (2 ^ (n + 1))

-- Define the convergence test function
def converges_at (f : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, tendsto (λ n, ∑ i in finset.range (n + 1), f i) at_top (nhds l)

-- State the problems:
theorem series_convergence (x : ℝ) : 
  (x = 1 → converges_at (series 1)) ∧ 
  (x = 3 → ¬ converges_at (series 3)) ∧ 
  (x = -2 → ¬ converges_at (series (-2))) := 
by
  sorry

end SeriesConvergence

end series_convergence_l314_314072


namespace angle_triple_complement_l314_314315

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314315


namespace part1_solution_set_part2_range_a_l314_314951

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314951


namespace proposition_3_proposition_4_l314_314500

variables {m n : Line} {α β : Plane}

-- Proposition ③: If m is perpendicular to α and β, then α is parallel to β
theorem proposition_3 (h₁: m ⟂ α) (h₂: m ⟂ β) : α ∥ β :=
sorry 

-- Proposition ④: If m and n are both perpendicular to β, then m is parallel to n
theorem proposition_4 (h₁: m ⟂ β) (h₂: n ⟂ β) : m ∥ n :=
sorry 

end proposition_3_proposition_4_l314_314500


namespace photo_arrangements_l314_314394

-- Definitions for the problem conditions
def students : List String := ["A", "B", "C", "D", "E"]
def AB_must_be_adjacent (arrangement : List String) : Bool := 
  let idxA := arrangement.indexOf "A"
  let idxB := arrangement.indexOf "B"
  (idxA ≠ -1) && (idxB ≠ -1) && (idxA + 1 == idxB || idxA - 1 == idxB)

-- The proof statement, demonstrating that the number of arrangements where A and B are adjacent is 48.
theorem photo_arrangements : 
  ∃ (arrangements : List (List String)), 
    (∀ arrangement ∈ arrangements, AB_must_be_adjacent arrangement) ∧
    arrangements.length = 48 := 
  sorry

end photo_arrangements_l314_314394


namespace men_wages_l314_314761

def men := 5
def women := 5
def boys := 7
def total_wages := 90
def wage_man := 7.5

theorem men_wages (men women boys : ℕ) (total_wages wage_man : ℝ)
  (h1 : 5 = women) (h2 : women = boys) (h3 : 5 * wage_man + 1 * wage_man + 7 * wage_man = total_wages) :
  5 * wage_man = 37.5 :=
  sorry

end men_wages_l314_314761


namespace part1_solution_set_part2_range_of_a_l314_314925

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314925


namespace angle_is_67_l314_314209

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314209


namespace angle_measure_l314_314269

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314269


namespace cot_45_eq_1_l314_314822

theorem cot_45_eq_1 :
  Real.cot (Real.pi / 4) = 1 := 
  sorry

end cot_45_eq_1_l314_314822


namespace rectangle_area_increase_l314_314740

theorem rectangle_area_increase {l w : ℝ} : 
  let original_area := l * w in
  let new_area := (1.3 * l) * (1.2 * w) in
  ((new_area - original_area) / original_area) * 100 = 56 :=
by
  sorry

end rectangle_area_increase_l314_314740


namespace angle_triple_complement_l314_314257

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314257


namespace angle_triple_complement_l314_314248

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314248


namespace right_angle_A_l314_314069

variable {α : Type*}

-- Triangle sides and angles variables
variables {A B C : α}
variables {a b c : ℝ}

-- Conditions
def triangle_sides_opposite (a b c : ℝ) : Prop := True  -- Given by the problem statement, so it's assumed true

-- Given equation in the conditions
def given_equation (a b c : ℝ) : Prop := (a + b) * (a - b) = c^2

-- To be proved: angle A is a right angle (pythagorean theorem form: a^2 = b^2 + c^2 indicating right triangle)
theorem right_angle_A :
  triangle_sides_opposite a b c → given_equation a b c → (a^2 = b^2 + c^2) :=
by
  intros _ h
  rw given_equation at h
  sorry

end right_angle_A_l314_314069


namespace part1_part2_l314_314972

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314972


namespace highest_power_of_3_M_l314_314625

noncomputable def M : ℕ :=
  let lst := (list.range' 31 45).map (λ x, x.toString).foldl (++) "" in
  lst.toNat

theorem highest_power_of_3_M : ∃ m : ℕ, 3^m ∣ M ∧ ∀ n : ℕ, 3^n ∣ M → n ≤ 0 :=
begin
  use 0,
  split,
  { -- Proof that 3^0 divides M
    sorry },
  { -- Proof that for any other n, 3^n divides M implies n <= 0
    intros n h,
    by_contradiction hn,
    sorry }
end

end highest_power_of_3_M_l314_314625


namespace part1_part2_l314_314933

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314933


namespace a_worked_days_l314_314356

theorem a_worked_days {d_a w_a w_b w_c k : ℝ} (h_daily_wage_c : w_c = 110)
    (h_ratio : w_a / 3 = w_b / 4 ∧ w_b / 4 = w_c / 5)
    (h_b_days : 9)
    (h_c_days : 4)
    (h_total_earning : 1628 = w_a * d_a + w_b * 9 + w_c * 4)
    : d_a = 6 :=
by
  let k := 22
  have h_w_a : w_a = 3 * k, from sorry
  have h_w_b : w_b = 4 * k, from sorry
  have h_unchanged : 1628 = (3 * 22) * d_a + (4 * 22) * 9 + 110 * 4, from sorry
  exact sorry

end a_worked_days_l314_314356


namespace angle_measure_triple_complement_l314_314306

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314306


namespace factorization_of_z6_minus_64_l314_314457

theorem factorization_of_z6_minus_64 :
  ∀ (z : ℝ), (z^6 - 64) = (z - 2) * (z^2 + 2*z + 4) * (z + 2) * (z^2 - 2*z + 4) := 
by
  intros z
  sorry

end factorization_of_z6_minus_64_l314_314457


namespace shift_sin_graph_l314_314180

def f (x : ℝ) := Real.sin (2 * x - Real.pi / 3)
def g (x : ℝ) := Real.sin (2 * x)

theorem shift_sin_graph : ∃ d : ℝ, ∀ x : ℝ, f (x + d) = g x :=
by
  use -Real.pi / 6
  intro x
  sorry

end shift_sin_graph_l314_314180


namespace candy_initial_count_l314_314564

theorem candy_initial_count (candy_given_first candy_given_second candy_given_third candy_bought candy_eaten candy_left initial_candy : ℕ) 
    (h1 : candy_given_first = 18) 
    (h2 : candy_given_second = 12)
    (h3 : candy_given_third = 25)
    (h4 : candy_bought = 10)
    (h5 : candy_eaten = 7)
    (h6 : candy_left = 16)
    (h_initial : candy_left + candy_eaten = initial_candy - candy_bought - candy_given_first - candy_given_second - candy_given_third):
    initial_candy = 68 := 
by 
  sorry

end candy_initial_count_l314_314564


namespace max_area_of_quadrilateral_l314_314087

theorem max_area_of_quadrilateral (ABCD : Type) [ConvexQuadrilateral ABCD]
(AB AD : ℝ) (hAB : AB = 3) (hAD : AD = 4)
(hEquilateralCentroids : ∃ (A B C D : Point ABCD),
  let g1 := centroid (A,B,C),
  let g2 := centroid (A,B,D),
  let g3 := centroid (B,C,D) in
  equilateral_triangle g1 g2 g3) :
  ∃ (area : ℝ), area ≤ 6 :=
begin
  sorry
end

end max_area_of_quadrilateral_l314_314087


namespace smallest_possible_b_l314_314588

theorem smallest_possible_b (a b c : ℚ) (h1 : a < b) (h2 : b < c)
    (arithmetic_seq : 2 * b = a + c) (geometric_seq : c^2 = a * b) :
    b = 1 / 2 :=
by
  let a := 4 * b
  let c := 2 * b - a
  -- rewrite and derived equations will be done in the proof
  sorry

end smallest_possible_b_l314_314588


namespace pyramid_cut_plane_ratio_l314_314799

theorem pyramid_cut_plane_ratio (S A B C P Q R : Point) (hSABC : IsPyramid S A B C)
  (hP_in_SA : P ∈ Segment S A) (hQ_in_AB : Q ∈ Segment A B) (hR_in_AC : R ∈ Segment A C)
  (hAQ_QB_ratio : ratio (length AQ) (length QB) = 1 / 2)
  (hAR_RC_ratio : ratio (length AR) (length RC) = 1 / 2)
  (hAP_PS_ratio : ratio (length AP) (length PS) = 2 / 1)
  (cutting_plane : IsPlane P Q R) :
  volume_ratio (pyramid_volume S A B C) (pyramid_volume P A Q R) = 25 / 2 :=
begin
  sorry -- Proof placeholder
end

end pyramid_cut_plane_ratio_l314_314799


namespace two_courtiers_have_same_selection_l314_314684

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l314_314684


namespace find_theta_and_max_g_l314_314913

noncomputable def f (x θ : ℝ) : ℝ :=
  cos x * cos (x - θ) - (1 / 2) * cos θ

noncomputable def g (x θ : ℝ) : ℝ :=
  2 * f (3 / 2 * x) θ

theorem find_theta_and_max_g :
  (∀ θ ∈ (0 : ℝ, π), ∀ x, f x θ ≤ f (π / 3) θ) → 
  (∃ θ, θ = (2 * π / 3) ∧ ∀ x ∈ set.Icc 0 (π / 3), g x (2 * π / 3) ≤ 1) :=
by
  intro h₁
  use (2 * π / 3)
  split
  · rfl
  · intro x hx
    sorry

end find_theta_and_max_g_l314_314913


namespace largest_divisor_of_dice_product_l314_314795

theorem largest_divisor_of_dice_product :
  ∀ (hidden_num : ℕ), hidden_num ∈ finset.range 1 11 →
  ∃ k : ℕ, k = 2^7 * 3^3 * 5^1 ∧ (10! / hidden_num) % k = 0 :=
begin
  sorry
end

end largest_divisor_of_dice_product_l314_314795


namespace part1_solution_part2_solution_l314_314955

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314955


namespace part1_solution_part2_solution_l314_314954

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314954


namespace max_absolute_difference_l314_314855

theorem max_absolute_difference :
  ∀ a : ℝ, ∃ max_value : ℝ, max_value = |sin (a + π/6) - 2 * cos a| 
          → max_value = sqrt (3 / 2) :=
by sorry

end max_absolute_difference_l314_314855


namespace no_solutions_l314_314364

theorem no_solutions (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : ¬ (x^5 = y^2 + 4) :=
by sorry

end no_solutions_l314_314364


namespace triple_complement_angle_l314_314283

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314283


namespace value_of_f1991_2_pow_1990_l314_314474

def sum_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.foldl (· + ·) 0

def f1 (k : ℕ) : ℕ := (sum_digits k)^2

def f : ℕ → ℕ → ℕ
| 0, k     := k
| (n+1), k := f1 (f n k)

theorem value_of_f1991_2_pow_1990 :
  f 1991 (2^1990) = 169 :=
sorry

end value_of_f1991_2_pow_1990_l314_314474


namespace slope_angle_perpendicular_l314_314506

theorem slope_angle_perpendicular (l : ℝ → ℝ) 
  (h : ∀ x, l x = -x + b) :
  ∃ α : ℝ, α = 135 :=
by
  use 135
  sorry

end slope_angle_perpendicular_l314_314506


namespace frustum_volume_fraction_l314_314402

def volume_pyramid (base_edge : ℝ) (height : ℝ) : ℝ :=
  (1/3) * (base_edge ^ 2) * height

def volume_frustum_as_fraction_of_original 
  (base_edge : ℝ) (height : ℝ) (smaller_height_fraction : ℝ) : ℝ :=
  let original_volume := volume_pyramid base_edge height
  let smaller_volume := volume_pyramid (base_edge * smaller_height_fraction) (height * smaller_height_fraction)
  let frustum_volume := original_volume - smaller_volume
  frustum_volume / original_volume

theorem frustum_volume_fraction :
  volume_frustum_as_fraction_of_original 24 18 (1/3) = 32 / 33 :=
by
  sorry

end frustum_volume_fraction_l314_314402


namespace min_expression_value_l314_314907

variable (α : ℝ)
noncomputable def z1 := complex.mk (sin α) 2
noncomputable def z2 := complex.mk 1 (cos α)
noncomputable def expr := (13 - complex.normSq (z1 + (complex.I * z2))) / complex.norm (z1 - (complex.I * z2))

theorem min_expression_value : ∃ (α : ℝ), expr α = 2 := sorry

end min_expression_value_l314_314907


namespace total_seniors_is_161_l314_314603

def total_students : ℕ := 240

def percentage_statistics : ℚ := 0.45
def percentage_geometry : ℚ := 0.35
def percentage_calculus : ℚ := 0.20

def percentage_stats_and_calc : ℚ := 0.10
def percentage_geom_and_calc : ℚ := 0.05

def percentage_seniors_statistics : ℚ := 0.90
def percentage_seniors_geometry : ℚ := 0.60
def percentage_seniors_calculus : ℚ := 0.80

def students_in_statistics : ℚ := percentage_statistics * total_students
def students_in_geometry : ℚ := percentage_geometry * total_students
def students_in_calculus : ℚ := percentage_calculus * total_students

def students_in_stats_and_calc : ℚ := percentage_stats_and_calc * students_in_statistics
def students_in_geom_and_calc : ℚ := percentage_geom_and_calc * students_in_geometry

def unique_students_in_statistics : ℚ := students_in_statistics - students_in_stats_and_calc
def unique_students_in_geometry : ℚ := students_in_geometry - students_in_geom_and_calc
def unique_students_in_calculus : ℚ := students_in_calculus - students_in_stats_and_calc - students_in_geom_and_calc

def seniors_in_statistics : ℚ := percentage_seniors_statistics * unique_students_in_statistics
def seniors_in_geometry : ℚ := percentage_seniors_geometry * unique_students_in_geometry
def seniors_in_calculus : ℚ := percentage_seniors_calculus * unique_students_in_calculus

def total_seniors : ℚ := seniors_in_statistics + seniors_in_geometry + seniors_in_calculus

theorem total_seniors_is_161 : total_seniors = 161 :=
by
  sorry

end total_seniors_is_161_l314_314603


namespace angle_between_NE_and_SW_l314_314385

theorem angle_between_NE_and_SW
  (n : ℕ) (hn : n = 12)
  (total_degrees : ℚ) (htotal : total_degrees = 360)
  (spaced_rays : ℚ) (hspaced : spaced_rays = total_degrees / n)
  (angles_between_NE_SW : ℕ) (hangles : angles_between_NE_SW = 4) :
  (angles_between_NE_SW * spaced_rays = 120) :=
by
  rw [htotal, hn] at hspaced
  rw [hangles]
  rw [hspaced]
  sorry

end angle_between_NE_and_SW_l314_314385


namespace trapezoid_division_segment_length_l314_314780

variable (a b : ℝ)

theorem trapezoid_division_segment_length 
  (h : ∃ t1 t2 : Trapezoid, similar t1 t2 ∧ t1.base1 = a ∧ t1.base2 = b ∧ t2.base1 = MN ∧ t2.base2 = b)
  : MN = Real.sqrt (a * b) :=
  sorry

end trapezoid_division_segment_length_l314_314780


namespace quartic_root_l314_314790

theorem quartic_root (x a b c : ℝ) (ha : a = 4096) (hb : b = 256) (hc : c = 16) 
(hpoly : 16 * x ^ 4 - 4 * x ^ 3 - 6 * x ^ 2 - 4 * x - 1 = 0) 
(hx : x = (Real.root 4 a + Real.root 4 b + 2) / c) : 
a + b + c = 4368 := by
  sorry

end quartic_root_l314_314790


namespace only_zero_function_satisfies_identity_l314_314847

noncomputable def satisfies_identity (f : ℝ → ℝ) : Prop :=
∀ (x y : ℝ), f(x + 2 * y) * f(x - 2 * y) = (f(x) + f(y))^2 - 16 * y^2 * f(x)

theorem only_zero_function_satisfies_identity : ∀ f : ℝ → ℝ, satisfies_identity f → ∀ x : ℝ, f x = 0 :=
by
  intro f hf x
  sorry

end only_zero_function_satisfies_identity_l314_314847


namespace triangle_area_l314_314627

noncomputable def area_of_triangle (p : ℝ) (hp : p > 0) : ℝ :=
  let A : ℝ × ℝ := (3 * p / 2, (real.sqrt 3) * p) in
  let F : ℝ × ℝ := (p / 2, 0) in
  (1 / 2) * (p / 2) * (real.sqrt 3 * p)

theorem triangle_area (p : ℝ) (hp : p > 0) :
  area_of_triangle p hp = (real.sqrt 3 / 4) * p^2 :=
sorry

end triangle_area_l314_314627


namespace number_of_black_squares_in_eaten_area_l314_314135

def is_black_square (row col : ℕ) : Bool :=
  (row + col) % 2 = 1

theorem number_of_black_squares_in_eaten_area (eaten_area : list (ℕ × ℕ)) :
  eaten_area = list.range (8*8) / 2 := sorry

end number_of_black_squares_in_eaten_area_l314_314135


namespace convex_hulls_common_point_l314_314610

-- Definition of convex hull (informal, actual proof tasks may need more detailed setup)
noncomputable def convex_hull (S : set (ℝ × ℝ)) : set (ℝ × ℝ) := sorry

-- Helper predicates to formalize convex hull intersection
def has_common_point (sets : list (set (ℝ × ℝ))) : Prop := 
∃ p : (ℝ × ℝ), ∀ s ∈ sets, p ∈ convex_hull s

-- Main theorem statement
theorem convex_hulls_common_point (m : ℕ) : ∃ n : ℕ, ∀ (points : set (ℝ × ℝ)), 
  cardinal.mk points = n → 
  ∃ (partition : list (set (ℝ × ℝ))), 
    partition.length = m ∧
    (∀ s ∈ partition, s.nonempty) ∧
    set.univ ⊆ ⋃₀ set.image convex_hull partition ∧
    has_common_point partition :=
sorry

end convex_hulls_common_point_l314_314610


namespace remaining_paint_fraction_l314_314766

theorem remaining_paint_fraction (x : ℝ) (h : 1.2 * x = 1 / 2) : (1 / 2) - x = 1 / 12 :=
by 
  sorry

end remaining_paint_fraction_l314_314766


namespace part1_solution_set_part2_range_a_l314_314952

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314952


namespace at_least_half_of_team_B_can_serve_l314_314723

noncomputable theory

def height_limit : ℕ := 168

-- Definitions for average, median, tallest, and mode
def team_A_average_height : ℕ := 166
def team_B_median_height : ℕ := 167
def team_C_tallest_height : ℕ := 169
def team_D_mode_height : ℕ := 167

-- Definition for "at least half of the sailors in a team can serve on the submarine"
def at_least_half_can_serve (n : ℕ) (team_heights : list ℕ) : Prop :=
  (team_heights.filter (λ h, h ≤ height_limit)).length ≥ team_heights.length / 2

theorem at_least_half_of_team_B_can_serve :
  ∀ (team_heights : list ℕ), team_B_median_height = 167 →
  (at_least_half_can_serve height_limit team_heights ↔ ∃ k, k = team_heights.length / 2 ∧
  (team_heights.filter (λ h, h ≤ 167)).length ≥ k) :=
by
  intros team_heights h
  split
  all_goals { sorry }

end at_least_half_of_team_B_can_serve_l314_314723


namespace triangle_ratio_l314_314142

theorem triangle_ratio (A B C L1 L2 I : Point)
  (h1 : AngleBisector A L1)
  (h2 : AngleBisector B L2)
  (h3 : IntersectsAt A L1 B L2 I)
  (h4 : Ratio AI IL1 = 3)
  (h5 : Ratio BI IL2 = 2) :
  Ratio (Side A B) (Side B C) (Side C A) = (3, 4, 5) :=
sorry

end triangle_ratio_l314_314142


namespace geometry_proof_l314_314409

open EuclideanGeometry

noncomputable def rectangle_proof (A B C D K L M N O : Point) : Prop :=
  is_rectangle A B C D ∧
  on_line K A B ∧
  on_line L B C ∧
  on_line M C D ∧
  on_line N D A ∧
  parallel (line K L) (line M N) ∧
  perpendicular (line K M) (line L N) ∧
  intersection (line K M) (line L N) = some O →
  collinear [O, B, D]

theorem geometry_proof (A B C D K L M N O : Point) :
  rectangle_proof A B C D K L M N O := 
begin 
  sorry,
end

end geometry_proof_l314_314409


namespace courtier_selection_l314_314675

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l314_314675


namespace daily_evaporation_l314_314776

theorem daily_evaporation (initial_water : ℝ) (days : ℕ) (percent_evaporated : ℝ) (total_evaporated : ℝ) : 
  initial_water = 12 ∧ days = 22 ∧ percent_evaporated = 5.5 ∧ total_evaporated = (percent_evaporated / 100) * initial_water → 
  (total_evaporated / days = 0.03) :=
begin
  sorry
end

end daily_evaporation_l314_314776


namespace sum_first_8_terms_l314_314888

noncomputable def a : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := a (n + 1) + a n + list.sum (list.map (λ i, a i) (list.range (n + 1)))

theorem sum_first_8_terms : (list.sum (list.map a (list.range 8))) = 127 := 
by sorry

end sum_first_8_terms_l314_314888


namespace triple_complement_angle_l314_314278

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314278


namespace part1_part2_l314_314978

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314978


namespace minimize_cost_l314_314771

def total_cost (x : ℝ) : ℝ := 40_000 * (400 / x) + 40_000 * x

theorem minimize_cost : ∃ x, x = 20 ∧ ∀ y ≠ 20, total_cost 20 ≤ total_cost y :=
sorry

end minimize_cost_l314_314771


namespace angle_triple_complement_l314_314298

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314298


namespace contrapositive_inequality_l314_314896

theorem contrapositive_inequality (x : ℝ) :
  ((x + 2) * (x - 3) > 0) → (x < -2 ∨ x > 0) :=
by
  sorry

end contrapositive_inequality_l314_314896


namespace angle_measure_l314_314265

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314265


namespace termites_ate_black_squares_l314_314132

-- Define a Lean function to check if a given cell is black on a standard chessboard.
def is_black (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Define the theorem that asserts the number of black squares in a 3x8 block is 12.
theorem termites_ate_black_squares : 
  (finset.univ.filter (λ k : ℕ × ℕ, k.1 < 3 ∧ k.2 < 8 ∧ is_black k.1 k.2)).card = 12 :=
by
  sorry -- proof to be provided

end termites_ate_black_squares_l314_314132


namespace max_length_proof_l314_314774

def side_length : ℝ := 2
def edge_length : ℝ := 2
def face_diagonal_length : ℝ := 2 * Real.sqrt 2
def space_diagonal_length : ℝ := 2 * Real.sqrt 3

def max_possible_length : ℝ :=
  2 * (4 * space_diagonal_length) + 4 * face_diagonal_length

theorem max_length_proof : max_possible_length = 16 * Real.sqrt 3 + 8 * Real.sqrt 2 :=
by 
  have h_edge_length : edge_length = 2 := rfl
  have h_face_diagonal_length : face_diagonal_length = 2 * Real.sqrt 2 := rfl
  have h_space_diagonal_length : space_diagonal_length = 2 * Real.sqrt 3 := rfl
  simp [max_possible_length, h_edge_length, h_face_diagonal_length, h_space_diagonal_length]
  sorry

end max_length_proof_l314_314774


namespace part1_part2_l314_314928

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314928


namespace angle_triple_complement_l314_314299

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314299


namespace gcd_of_items_l314_314437

theorem gcd_of_items :
  ∀ (plates spoons glasses bowls : ℕ),
  plates = 3219 →
  spoons = 5641 →
  glasses = 1509 →
  bowls = 2387 →
  Nat.gcd (Nat.gcd (Nat.gcd plates spoons) glasses) bowls = 1 :=
by
  intros plates spoons glasses bowls
  intros Hplates Hspoons Hglasses Hbowls
  rw [Hplates, Hspoons, Hglasses, Hbowls]
  sorry

end gcd_of_items_l314_314437


namespace angle_triple_complement_l314_314323

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314323


namespace solution_interval_l314_314439

noncomputable def monotonic_function_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → f (f x - log x) = 1

theorem solution_interval (f : ℝ → ℝ) (hf : monotonic_function_condition f)
  (hmono : ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ ≤ x₂ → f x₁ ≤ f x₂) :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x - (1 / x) = 1 :=
sorry

end solution_interval_l314_314439


namespace two_courtiers_have_same_selection_l314_314687

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l314_314687


namespace courtiers_dog_selection_l314_314651

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l314_314651


namespace matches_in_each_box_l314_314563

noncomputable def matches_per_box (dozens_boxes : ℕ) (total_matches : ℕ) : ℕ :=
  total_matches / (dozens_boxes * 12)

theorem matches_in_each_box :
  matches_per_box 5 1200 = 20 :=
by
  sorry

end matches_in_each_box_l314_314563


namespace no_solution_implies_b_positive_l314_314562

theorem no_solution_implies_b_positive (a b : ℝ) :
  (¬ ∃ x y : ℝ, y = x^2 + a * x + b ∧ x = y^2 + a * y + b) → b > 0 :=
by
  sorry

end no_solution_implies_b_positive_l314_314562


namespace points_in_triangle_l314_314491

theorem points_in_triangle (A B C : Point) (hC : angle C = 90) (P : ℕ → Point) (n : ℕ) (P_in_triangle : ∀ i, 1 ≤ i ∧ i ≤ n → Proof (in_triangle P[i] A B C)) :
  ∑ i in range (n-1), (dist^2 P[i]² + dist^2 P[i+1]²) ≤ dist^2 A B := sorry

end points_in_triangle_l314_314491


namespace decagon_area_bisection_ratio_l314_314148

theorem decagon_area_bisection_ratio
  (decagon_area : ℝ := 12)
  (below_PQ_area : ℝ := 6)
  (trapezoid_area : ℝ := 4)
  (b1 : ℝ := 3)
  (b2 : ℝ := 6)
  (h : ℝ := 8/9)
  (XQ : ℝ := 4)
  (QY : ℝ := 2) :
  (XQ / QY = 2) :=
by
  sorry

end decagon_area_bisection_ratio_l314_314148


namespace angle_measure_triple_complement_l314_314303

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314303


namespace angle_measure_l314_314267

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314267


namespace part1_solution_part2_solution_l314_314963

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314963


namespace angle_triple_complement_l314_314288

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314288


namespace courtier_selection_l314_314674

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l314_314674


namespace min_cuts_needed_to_hit_coin_l314_314070

-- Define the problem conditions
def pancake_radius : ℝ := 10
def coin_radius : ℝ := 1

-- Define the mathematically equivalent proof problem
theorem min_cuts_needed_to_hit_coin : ∀ (pancake_radius coin_radius : ℝ), (pancake_radius = 10) -> (coin_radius = 1) -> 
    ∃ (n : ℕ), n = 10 ∧ ∀ c : ℕ, c < n -> (∃ uncovered_segment : ℝ, uncovered_segment > 0 ∧ uncovered_segment < 2) →
    n ≥ 10 :=
by
  intros pancake_radius coin_radius hpancake hcoin
  existsi 10
  split
  . exact rfl
  sorry

end min_cuts_needed_to_hit_coin_l314_314070


namespace area_B_l314_314518

def A : Set (ℝ × ℝ) := { p | |p.1| ≤ 1 ∧ |p.2| ≤ 1 }

def B : Set (ℝ × ℝ) := { p | ∃ a b : ℝ, (a, b) ∈ A ∧ (p.1 - a)^2 + (p.2 - b)^2 ≤ 1 }

theorem area_B : measure_theoretic.measure.measure_univ B = 12 + Real.pi := by
  sorry

end area_B_l314_314518


namespace rectangle_area_increase_l314_314741

theorem rectangle_area_increase {l w : ℝ} : 
  let original_area := l * w in
  let new_area := (1.3 * l) * (1.2 * w) in
  ((new_area - original_area) / original_area) * 100 = 56 :=
by
  sorry

end rectangle_area_increase_l314_314741


namespace angle_measure_l314_314275

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314275


namespace binary101_to_decimal_l314_314828

theorem binary101_to_decimal :
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  binary_101 = 5 := 
by
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  show binary_101 = 5
  sorry

end binary101_to_decimal_l314_314828


namespace find_a_given_coefficient_l314_314041

theorem find_a_given_coefficient (a : ℝ) (h : (a^3 * 10 = 80)) : a = 2 :=
by
  sorry

end find_a_given_coefficient_l314_314041


namespace log_inequality_l314_314346

-- Definitions for the problem
noncomputable def c : ℝ := 2
def a : ℝ := 2
def b : ℝ := 1/2

-- Theorem statement
theorem log_inequality (hc : c > 1) : log a c > log b c := by
  sorry

end log_inequality_l314_314346


namespace cistern_width_l314_314770

variable (L B A w : ℝ)
variable (h : ∀ (L B A w : ℝ), L = 10 ∧ B = 1.35 ∧ A = 103.2 → A = 10 * w + 2 * (10 * B) + 2 * (w * B))

theorem cistern_width (L B A : ℝ) (h_length : L = 10) (h_breadth : B = 1.35) (h_area : A = 103.2) : 
  w = 6 := 
by
  -- use the given variables and the equation from the conditions
  have eq1 : A = 10 * w + 2 * (10 * B) + 2 * (w * B) := by sorry
  -- now plug in the known values to simplify the equality
  have eq2 : 103.2 = 10 * w + 27 + 2.7 * w :=
    by
      rw [h_length, h_breadth] at eq1
      exact eq1
  -- isolate and solve for w
  have eq3 : 76.2 = 12.7 * w := by linarith
  have eq4 : w = 6 := by linarith
  exact eq4

end cistern_width_l314_314770


namespace angle_triple_complement_l314_314294

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314294


namespace number_of_subsets_with_one_isolated_element_is_13_l314_314084

def is_isolated_element (A : Finset ℤ) (k : ℤ) : Prop :=
  k ∈ A ∧ k - 1 ∉ A ∧ k + 1 ∉ A

def has_exactly_one_isolated_element (A : Finset ℤ) (S : Finset ℤ) : Prop :=
  ∃ (k : ℤ), is_isolated_element A k ∧ S = {k}

noncomputable def number_of_subsets_with_one_isolated_element : ℕ :=
  let A : Finset ℤ := {1, 2, 3, 4, 5}
  let subsets_with_one_isolated_element :=
    (A.powerset.filter (λ S, ∃ k ∈ A, has_exactly_one_isolated_element S k))
  subsets_with_one_isolated_element.card 

-- The theorem statement
theorem number_of_subsets_with_one_isolated_element_is_13 :
  number_of_subsets_with_one_isolated_element = 13 := by
  sorry

end number_of_subsets_with_one_isolated_element_is_13_l314_314084


namespace angle_triple_complement_l314_314293

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314293


namespace angle_triple_complement_l314_314296

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314296


namespace range_of_a_l314_314010

variable {x : ℝ}
variable {a : ℝ}

def p := ∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), x ^ 2 - a ≥ 0
def q := ∃ x : ℝ, x ^ 2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a : ¬ (p ∨ q) → 1 < a ∧ a < 2 := by
  intros h
  sorry

end range_of_a_l314_314010


namespace angle_triple_complement_l314_314245

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314245


namespace angle_triple_complement_l314_314292

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314292


namespace problem_l314_314533

open Real

noncomputable def abs (x : ℝ) : ℝ := if x ≥ 0 then x else -x

def parabola (x y : ℝ) : Prop := y^2 = 8 * x

def on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

def focus_distance (x : ℝ) : ℝ := abs (x + 2)

theorem problem 
  (P1 P2 P3 : ℝ × ℝ)
  (hp1 : on_parabola P1)
  (hp2 : on_parabola P2)
  (hp3 : on_parabola P3)
  (x1 x2 x3 : ℝ)
  (hx1 : P1.1 = x1)
  (hx2 : P2.1 = x2)
  (hx3 : P3.1 = x3)
  (h_sum : x1 + x2 + x3 = 10) :
  focus_distance x1 + focus_distance x2 + focus_distance x3 = 16 := 
sorry

end problem_l314_314533


namespace first_vessel_milk_water_l314_314728

variable (V : ℝ)

def vessel_ratio (v1 v2 : ℝ) : Prop := 
  v1 / v2 = 3 / 5

def vessel1_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 1 / 2

def vessel2_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 3 / 2

def mix_ratio (milk1 water1 milk2 water2 : ℝ) : Prop :=
  (milk1 + milk2) / (water1 + water2) = 1

theorem first_vessel_milk_water (V : ℝ) (v1 v2 : ℝ) (m1 w1 m2 w2 : ℝ)
  (hv : vessel_ratio v1 v2)
  (hv1 : vessel1_milk_water_ratio m1 w1)
  (hv2 : vessel2_milk_water_ratio m2 w2)
  (hmix : mix_ratio m1 w1 m2 w2) :
  vessel1_milk_water_ratio m1 w1 :=
  sorry

end first_vessel_milk_water_l314_314728


namespace part1_solution_set_part2_range_l314_314982

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314982


namespace ladder_angle_l314_314779

theorem ladder_angle (θ : ℝ) :
  let length_ladder := 19
  let distance_foot := 9.493063650744542
  θ = Real.arccos (distance_foot / length_ladder)
  → θ ≈ 60 :=
sorry

end ladder_angle_l314_314779


namespace vec_parallel_implies_x_eq_4_l314_314522

-- Defining the vectors and the condition for parallelism
def vec_a (x : ℝ) : ℝ × ℝ × ℝ := (8, 0.5 * x, x)
def vec_b (x : ℝ) : ℝ × ℝ × ℝ := (x, 1, 2)

-- Statement that if vectors are parallel and x > 0, then x = 4
theorem vec_parallel_implies_x_eq_4 {x : ℝ} (h_par : ∃ λ > 0, vec_a x = λ • vec_b x) (h_x_pos : x > 0) : x = 4 := 
sorry

end vec_parallel_implies_x_eq_4_l314_314522


namespace weighted_average_inequality_l314_314403

variable (x y z : ℝ)
variable (h1 : x < y) (h2 : y < z)

theorem weighted_average_inequality :
  (4 * z + x + y) / 6 > (x + y + 2 * z) / 4 :=
by
  sorry

end weighted_average_inequality_l314_314403


namespace instantaneous_velocity_at_1_l314_314785

noncomputable def particle_displacement (t : ℝ) : ℝ := t + Real.log t

theorem instantaneous_velocity_at_1 : 
  let v := fun t => deriv (particle_displacement) t
  v 1 = 2 :=
by
  sorry

end instantaneous_velocity_at_1_l314_314785


namespace percentage_male_voters_for_sobel_l314_314359

theorem percentage_male_voters_for_sobel 
    (total_voters : ℕ)
    (sobel_voters_percentage : ℕ)
    (male_voters_percentage : ℕ)
    (female_voters_for_lange_percentage : ℕ) :
    let sobel_voters := (total_voters * sobel_voters_percentage) / 100
    let lange_voters := total_voters - sobel_voters
    let male_voters := (total_voters * male_voters_percentage) / 100
    let female_voters := total_voters - male_voters
    let female_voters_for_lange := (female_voters * female_voters_for_lange_percentage) / 100
    let male_voters_for_lange := lange_voters - female_voters_for_lange
    let male_voters_for_sobel := male_voters - male_voters_for_lange
    (male_voters_for_sobel * 100) / male_voters = 7333 / 100 :=
begin
  sorry
end

end percentage_male_voters_for_sobel_l314_314359


namespace angle_measure_triple_complement_l314_314222

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314222


namespace part1_solution_set_part2_range_a_l314_314941

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314941


namespace angle_AST_eq_90_l314_314095

open Real Angle

theorem angle_AST_eq_90 
  (ABC : Triangle)
  (h₁ : ABC.isAcute)
  (h₂ : ABC.circumradius = R)
  (D : Point)
  (h₃ : isFootOfAltitude D ABC.A ABC.B ABC.C)
  (T : Point)
  (h₄ : T ∈ Line AD)
  (h₅ : distance A T = 2 * R)
  (h₆ : isBetween D A T)
  (S : Point)
  (h₇ : S = midpointArcBCnotA ABC)
  : ∠ A S T = 90° :=
by
  sorry

end angle_AST_eq_90_l314_314095


namespace angle_is_67_l314_314213

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314213


namespace no_real_solution_for_x_l314_314899

theorem no_real_solution_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 8) (h2 : y + 1 / x = 7 / 20) : false :=
by sorry

end no_real_solution_for_x_l314_314899


namespace find_solution_l314_314460

theorem find_solution :
  (∃ x : ℚ, (∃ y : ℚ, y = 5 - x ∧ y^(1/4 : ℚ) = -2/3)) → x = 389/81 :=
sorry

end find_solution_l314_314460


namespace escalator_time_eq_1667_l314_314420

def escalator_speed : ℝ := 9
def person_speed : ℝ := 3
def escalator_length : ℝ := 200

theorem escalator_time_eq_1667 :
  let combined_speed := escalator_speed + person_speed in
  let time := escalator_length / combined_speed in
  time = 16.67 :=
by
  sorry

end escalator_time_eq_1667_l314_314420


namespace determine_a_l314_314043

theorem determine_a (a : ℝ) (h : (binom 5 3) * a^3 = 80) : a = 2 := by
  sorry

end determine_a_l314_314043


namespace three_digit_numbers_sum_9_div_by_3_l314_314530

theorem three_digit_numbers_sum_9_div_by_3 :
  let num_valid_numbers := (finset.univ.filter (λ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 9)).card in
  num_valid_numbers = 45 :=
begin
  sorry
end

end three_digit_numbers_sum_9_div_by_3_l314_314530


namespace same_foci_of_ellipses_l314_314498

theorem same_foci_of_ellipses (k : ℝ) (hk : k < 4) : 
  let a₁ := 3,
      b₁ := 2,
      a₂ := Real.sqrt (9 - k),
      b₂ := Real.sqrt (4 - k),
      c₁ := Real.sqrt (a₁^2 - b₁^2),
      c₂ := Real.sqrt (a₂^2 - b₂^2)
  in c₁ = c₂ :=
by
  let a₁ := 3
  let b₁ := 2
  let a₂ := Real.sqrt (9 - k)
  let b₂ := Real.sqrt (4 - k)
  let c₁ := Real.sqrt (a₁^2 - b₁^2)
  let c₂ := Real.sqrt ((9 - k) - (4 - k))
  show c₁ = c₂
  sorry

end same_foci_of_ellipses_l314_314498


namespace red_marbles_count_l314_314867

theorem red_marbles_count :
  ∀ (total marbles white yellow green red : ℕ),
    total = 50 →
    white = total / 2 →
    yellow = 12 →
    green = yellow / 2 →
    red = total - (white + yellow + green) →
    red = 7 :=
by
  intros total marbles white yellow green red Htotal Hwhite Hyellow Hgreen Hred
  sorry

end red_marbles_count_l314_314867


namespace max_possible_sum_l314_314757

noncomputable theory

open_locale classical

-- Define the problem conditions with assumptions.
variables {a b : ℝ} (n : ℕ) (points : Type) [fintype points] [decidable_eq points]

-- Define the predicate for non-negative real numbers on segments between points.
variables (f : points → points → ℝ) -- f is the function mapping two points to a real number

-- Define the maximum possible sum
axiom sum_le_one : ∀ (v : finset points), v.card ≥ 3 → finset.sum (v.product v) (λ x, f x.fst x.snd) ≤ 1

-- Define the theorem to state the maximum sum
theorem max_possible_sum {n : ℕ} (hn : even n) (hpoints : fintype.card points = n) :
  ∃ max_sum : ℝ, max_sum = n^2 / 8 :=
sorry

end max_possible_sum_l314_314757


namespace andy_candies_l314_314820

theorem andy_candies :
  ∃ A : ℕ, 
    let B := 6 in
    let C := 11 in
    let F := 36 in
    let given_to_B := 8 in
    let given_to_C := 11 in
    let candy_left_for_A := F - given_to_B - given_to_C in
    let total_C_candies := C + given_to_C in
    let total_A_candies := A + candy_left_for_A in
	(total_A_candies = total_C_candies + 4) → A = 9 :=
by {
    sorry
}

end andy_candies_l314_314820


namespace lcm_24_36_45_l314_314334

open Nat

theorem lcm_24_36_45 : lcm (lcm 24 36) 45 = 360 := by sorry

end lcm_24_36_45_l314_314334


namespace part1_part2_l314_314977

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314977


namespace lcm_24_36_45_l314_314337

open Nat

theorem lcm_24_36_45 : lcm (lcm 24 36) 45 = 360 := by sorry

end lcm_24_36_45_l314_314337


namespace angle_triple_complement_l314_314319

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314319


namespace cosA_sinB_value_l314_314129

theorem cosA_sinB_value (A B : ℝ) (hA1 : 0 < A ∧ A < π / 2) (hB1 : 0 < B ∧ B < π / 2)
  (h_tan_eq : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = Real.sqrt 6 / 6 := sorry

end cosA_sinB_value_l314_314129


namespace correct_operation_l314_314747

theorem correct_operation (a : ℝ) : a^8 / a^2 = a^6 :=
by
  -- proof will go here, let's use sorry to indicate it's unfinished
  sorry

end correct_operation_l314_314747


namespace minimum_unit_cubes_correct_l314_314388

noncomputable def minimum_unit_cubes (n : ℕ) : ℕ :=
if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2

theorem minimum_unit_cubes_correct (n : ℕ) : 
  (minimum_unit_cubes 2 = 2) ∧
  (minimum_unit_cubes 3 = 5) ∧
  (minimum_unit_cubes 4 = 8) ∧
  (minimum_unit_cubes 10 = 50) :=
begin
  -- proof skipped
  sorry
end

end minimum_unit_cubes_correct_l314_314388


namespace no_point_P_exists_l314_314071

theorem no_point_P_exists (heptagon : Type) [heptagon_regular : regular_heptagon heptagon] :
  ¬ ∃ (P : heptagon), (exactly_four_lines_intersect_opposite_sides P) :=
by
  sorry

end no_point_P_exists_l314_314071


namespace max_volume_l314_314477

noncomputable def volume_cone (x : ℝ) : ℝ :=
  let r := (2 * real.pi - x) / (2 * real.pi) in
  let h := real.sqrt (1 - r^2) in
  (1 / 3) * real.pi * r^2 * h

theorem max_volume (x : ℝ) :
  let r := (2 * real.pi - x) / (2 * real.pi) in 
  let h := real.sqrt (1 - r^2) in
  (∀ x' : ℝ, volume_cone x' ≤ volume_cone x) → 
  x = (6 - 2 * real.sqrt 6) / 3 * real.pi :=
begin
  sorry
end

end max_volume_l314_314477


namespace part1_part2_l314_314971

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314971


namespace triple_complement_angle_l314_314279

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314279


namespace garden_dimensions_l314_314395
noncomputable theory

theorem garden_dimensions (b l : ℝ) 
  (h1 : l = (3/5) * b)
  (h2 : l * b = 600)
  (h3 : 2 * l + 2 * b ≤ 120) : 
  b = 10 * real.sqrt 10 ∧ l = 6 * real.sqrt 10 :=
sorry

end garden_dimensions_l314_314395


namespace calc_mean_excluding_highest_lowest_l314_314110

theorem calc_mean_excluding_highest_lowest {α : Type*} {l : list α} [decidable_eq α] [linear_order α] (scores : list ℕ) (h : scores = [85, 90, 70, 95, 80]) : 
  let remaining_scores := scores.erase (scores.maximum) |>.erase (scores.minimum) in
  let mean_score := (remaining_scores.foldr (+) 0) / remaining_scores.length in
  mean_score = 85 := 
by {
  sorry
}

end calc_mean_excluding_highest_lowest_l314_314110


namespace domain_of_function_l314_314632

theorem domain_of_function :
  {x : ℝ | 2 - x ≥ 0} = {x : ℝ | x ≤ 2} :=
by
  sorry

end domain_of_function_l314_314632


namespace tangent_distance_l314_314367

noncomputable def origin : (ℝ × ℝ) := (0, 0)

def circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 2)^2 + (p.2 - 1)^2 = 1

def is_tangent (o p c : ℝ × ℝ) : Prop :=
  let |OP| := real.sqrt ((p.1 - o.1)^2 + (p.2 - o.2)^2)
  let |OC| := real.sqrt ((c.1 - o.1)^2 + (c.2 - o.2)^2)
  let |PC| := 1 -- Radius of the circle
  (OP^2 = OC^2 - PC^2)

theorem tangent_distance
  (o : (ℝ × ℝ))
  (h1 : o = (0, 0)) -- O is the origin
  (p : (ℝ × ℝ))
  (h2 : circle p) -- P is on the circle
  : is_tangent o p (2, 1) → real.sqrt ((p.1 - o.1)^2 + (p.2 - o.2)^2) = 2 := 
sorry

end tangent_distance_l314_314367


namespace limit_calculation_l314_314431

noncomputable def limit_expression (x : ℝ) : ℝ :=
  ((ln x - 1) / (x - real.exp 1)) ^ (real.sin ((real.pi / (2 * real.exp 1)) * x))

theorem limit_calculation : 
  filter.tendsto limit_expression (nhds (real.exp 1)) (nhds (1 / real.exp 1)) :=
sorry

end limit_calculation_l314_314431


namespace safe_dishes_count_l314_314531

theorem safe_dishes_count (total_dishes vegan_dishes vegan_with_nuts : ℕ) 
  (h1 : vegan_dishes = total_dishes / 3) 
  (h2 : vegan_with_nuts = 4) 
  (h3 : vegan_dishes = 6) : vegan_dishes - vegan_with_nuts = 2 :=
by
  sorry

end safe_dishes_count_l314_314531


namespace right_triangle_sine_cosine_l314_314056

theorem right_triangle_sine_cosine {X Y Z : ℝ} (h : real.is_right_triangle X Y Z)
  (XZ : ℝ) (XY : ℝ) (XZ_len : XZ = 15) (XY_len : XY = 9) : 
  sin(X) = 4 / 5 ∧ cos(X) = 3 / 5 := 
by
  sorry

end right_triangle_sine_cosine_l314_314056


namespace part1_part2_l314_314976

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314976


namespace num_ways_to_factor_2210_l314_314023

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_ways_to_factor_2210 : ∃! (a b : ℕ), a * b = 2210 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end num_ways_to_factor_2210_l314_314023


namespace range_of_f_on_interval_l314_314705

noncomputable def f : ℝ → ℝ := λ x, x^2 - 2 * x + 3

theorem range_of_f_on_interval :
  set.image f (set.Icc 0 3) = set.Icc 2 6 :=
begin
  sorry
end

end range_of_f_on_interval_l314_314705


namespace polynomial_no_complex_zero_l314_314787

theorem polynomial_no_complex_zero
  (p q : ℤ) (α β : ℤ)
  (P : ℝ → ℝ)
  (hP_form : ∀ x, P x = (x - p) * (x - q) * (x^2 + α * x + β))
  (h_integer_coeffs : ∀ (x : ℝ), P x ∈ ℤ) :
  ¬(∃ x : ℝ, x ∈ { (2 + complex.i * sqrt 15) / 2, (2 + complex.i) / 2, 1 + complex.i, 2 + complex.i / 2, (2 + complex.i * sqrt 17) / 2 } ∧ P x = 0) :=
sorry

end polynomial_no_complex_zero_l314_314787


namespace diameter_increase_l314_314146

theorem diameter_increase (h : 0.628 = π * d) : d = 0.2 := 
sorry

end diameter_increase_l314_314146


namespace count_FourDigitNumsWithThousandsDigitFive_is_1000_l314_314527

def count_FourDigitNumsWithThousandsDigitFive : Nat :=
  let minNum := 5000
  let maxNum := 5999
  maxNum - minNum + 1

theorem count_FourDigitNumsWithThousandsDigitFive_is_1000 :
  count_FourDigitNumsWithThousandsDigitFive = 1000 :=
by
  sorry

end count_FourDigitNumsWithThousandsDigitFive_is_1000_l314_314527


namespace courtier_selection_l314_314680

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l314_314680


namespace exist_two_courtiers_with_same_selection_l314_314692

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l314_314692


namespace count_correct_propositions_l314_314419

-- Define the propositions as conditions (1) to (4).
def proposition1 : Prop := ¬(isCentrallySymmetric (equilateralTriangle))
def proposition2 : Prop := ¬(∃ (q : Quadrilateral), hasOnePairParallelAndOnePairEqualSides q ∧ isParallelogram q)
def proposition3 : Prop := ∀ (r : Rectangle), (diagonalsPerpendicular r → isSquare r)
def proposition4 : Prop := ¬(∀ (q : Quadrilateral), (diagonalsPerpendicular q ∧ ¬diagonalsBisectEachOther q → isRhombus q))

-- Define the main theorem to prove the count of correct propositions.
theorem count_correct_propositions :
  (proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4) → ∃ n, n = 1 := 
by
  sorry

end count_correct_propositions_l314_314419


namespace arrangement_count_l314_314717

theorem arrangement_count :
  let volunteers := 5
  let elderly := 2
  (count_arrangements (volunteers, elderly) = 960) :=
by
  -- Conditions
  let volunteers := 5
  let elderly := 2
  sorry

end arrangement_count_l314_314717


namespace ratio_elyse_to_rick_l314_314452

-- Define the conditions
def Elyse_initial_gum : ℕ := 100
def Shane_leftover_gum : ℕ := 14
def Shane_chewed_gum : ℕ := 11

-- Theorem stating the ratio of pieces Elyse gave to Rick to the total number of pieces Elyse had
theorem ratio_elyse_to_rick :
  let total_gum := Elyse_initial_gum
  let Shane_initial_gum := Shane_leftover_gum + Shane_chewed_gum
  let Rick_initial_gum := 2 * Shane_initial_gum
  let Elyse_given_to_Rick := Rick_initial_gum
  (Elyse_given_to_Rick : ℚ) / total_gum = 1 / 2 :=
by
  sorry

end ratio_elyse_to_rick_l314_314452


namespace red_marbles_count_l314_314869

theorem red_marbles_count (W Y G R : ℕ) (total_marbles : ℕ) 
(h1 : total_marbles = 50)
(h2 : W = 50 / 2)
(h3 : Y = 12)
(h4 : G = 12 - (12 * 0.5))
(h5 : W + Y + G + R = total_marbles)
: R = 7 :=
sorry

end red_marbles_count_l314_314869


namespace angle_is_67_l314_314204

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314204


namespace function_inverse_overlap_form_l314_314514

theorem function_inverse_overlap_form (a b c d : ℝ) (h : ¬(a = 0 ∧ c = 0)) : 
  (∀ x, (c * x + d) * (dx - b) = (a * x + b) * (-c * x + a)) → 
  (∃ f : ℝ → ℝ, (∀ x, f x = x ∨ f x = (a * x + b) / (c * x - a))) :=
by 
  sorry

end function_inverse_overlap_form_l314_314514


namespace acute_triangle_iff_sum_of_squares_l314_314609

theorem acute_triangle_iff_sum_of_squares (a b c R : ℝ) 
  (hRpos : R > 0) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  (∀ α β γ, (a = 2 * R * Real.sin α) ∧ (b = 2 * R * Real.sin β) ∧ (c = 2 * R * Real.sin γ) → 
   (α < Real.pi / 2 ∧ β < Real.pi / 2 ∧ γ < Real.pi / 2)) ↔ 
  (a^2 + b^2 + c^2 > 8 * R^2) :=
sorry

end acute_triangle_iff_sum_of_squares_l314_314609


namespace volume_P_ABC_l314_314712

noncomputable def volume_of_tetrahedron (a b c pa pb pc : ℝ) : ℝ :=
  let base_area := (Real.sqrt 3) / 4 * a^2 in
  let h := Real.sqrt ((a^2 + b^2 + c^2 - a^2 * b^2 - b^2 * c^2 - c^2 * a^2) / (4 * a * b * c)) in
  (1 / 3) * base_area * h

theorem volume_P_ABC (PA PB PC abc_base : ℝ) (hp : PA = 3) (hq : PB = 4) (hr : PC = 5) :
  volume_of_tetrahedron 3 3 3 PA PB PC = ℝ.sqrt(11) := sorry

end volume_P_ABC_l314_314712


namespace part_one_part_two_l314_314513

noncomputable def f (a x : ℝ) := a * Real.log x - x + 1

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≤ 0) : a = 1 := 
sorry

theorem part_two (h₁ : ∀ x > 0, f 1 x ≤ 0) (x : ℝ) (h₂ : 0 < x) (h₃ : x < Real.pi / 2) :
  Real.exp x * Real.sin x - x > f 1 x :=
sorry

end part_one_part_two_l314_314513


namespace cake_cut_proof_l314_314152

theorem cake_cut_proof (s : ℝ) (A : ℝ) (n : ℕ) (hA : A = s^2 / (n+1)) :
  ¬ ∃ l w : ℝ, (∀ k, 1 ≤ k ∧ k ≤ n → (l(k) * w(k) = A ∧ l(k) > w(k))) ∧ l(n) = w(n) :=
by
  sorry

end cake_cut_proof_l314_314152


namespace angle_measure_triple_complement_l314_314218

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314218


namespace part1_solution_set_part2_range_l314_314987

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314987


namespace area_of_quadrilateral_PRAQ_l314_314185

noncomputable def side_QR : ℝ := real.sqrt (20^2 - 13^2)
noncomputable def side_AQ : ℝ := real.sqrt (21^2 - side_QR^2)

noncomputable def area_PAQ : ℝ := 0.5 * 13 * side_AQ
noncomputable def area_PRQ : ℝ := 0.5 * 21 * side_QR

noncomputable def area_PRAQ : ℝ := area_PAQ + area_PRQ

theorem area_of_quadrilateral_PRAQ :
  triangles_right PAQ PRQ →
  PA = 13 →
  PQ = 20 →
  PR = 21 →
  area_PRAQ = 6.5 * real.sqrt 210 + 10.5 * real.sqrt 231 :=
by
  intros h1 h2 h3 h4
  rw [area_PRAQ, area_PAQ, area_PRQ]
  exact sorry

end area_of_quadrilateral_PRAQ_l314_314185


namespace price_of_toy_organizers_is_78_l314_314566

variable (P : ℝ) -- Price per set of toy organizers

-- Conditions
def total_cost_of_toy_organizers (P : ℝ) : ℝ := 3 * P
def total_cost_of_gaming_chairs : ℝ := 2 * 83
def total_sales (P : ℝ) : ℝ := total_cost_of_toy_organizers P + total_cost_of_gaming_chairs
def delivery_fee (P : ℝ) : ℝ := 0.05 * total_sales P
def total_amount_paid (P : ℝ) : ℝ := total_sales P + delivery_fee P

-- Proof statement
theorem price_of_toy_organizers_is_78 (h : total_amount_paid P = 420) : P = 78 :=
by
  sorry

end price_of_toy_organizers_is_78_l314_314566


namespace part1_solution_part2_solution_l314_314957

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314957


namespace right_triangle_hypotenuse_l314_314699

noncomputable def hypotenuse_log3 (hypotenuse : ℝ) : Prop :=
  let leg1 := real.log 128 / real.log 3 in
  let leg2 := real.log 32 / real.log 9 in
  let actual_hypotenuse := real.sqrt (leg1^2 + leg2^2) in
  leg1^2 + leg2^2 = hypotenuse^2

theorem right_triangle_hypotenuse : 
  ∀ (h : ℝ), hypotenuse_log3 h → 3^h = 8192 :=
by
  intro h
  assume hyp_cond
  sorry

end right_triangle_hypotenuse_l314_314699


namespace part_one_probability_part_two_probability_l314_314173

-- Define the basic setup for the cards
variable (cards1 : List (String × ℕ)) := [("Red", 1), ("Red", 2), ("Red", 3), ("Blue", 1), ("Blue", 2)]
variable (cards2 : List (String × ℕ)) := [("Red", 1), ("Red", 2), ("Red", 3), ("Blue", 1), ("Blue", 2), ("Green", 0)]

-- Define a function to compute the probability for given cards, colors, and condition
def probability_diff_colors_sum_lt_4 (cards : List (String × ℕ)) : ℚ :=
  let pairs := cards.combination 2
  let favorable := pairs.count (λ pair, pair[0].1 ≠ pair[1].1 ∧ pair[0].2 + pair[1].2 < 4)
  favorable / pairs.length

-- Proof statement for the first part
theorem part_one_probability : probability_diff_colors_sum_lt_4 cards1 = 3 / 10 := 
sorry

-- Proof statement for the second part
theorem part_two_probability : probability_diff_colors_sum_lt_4 cards2 = 8 / 15 := 
sorry

end part_one_probability_part_two_probability_l314_314173


namespace triple_complement_angle_l314_314280

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314280


namespace max_min_f_a_minus_2_range_of_a_for_monotonicity_l314_314910

-- Define the function f and set the domain [x]
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x + 3

-- Setting the given conditions
def domain : Set ℝ := {x | -4 ≤ x ∧ x ≤ 6}

-- Problem 1: Proving maximum and minimum values when a = -2
theorem max_min_f_a_minus_2 : 
  (∀ x ∈ domain, f x (-2) ≤ 35) ∧ 
  (∃ x ∈ domain, f x (-2) = 35) ∧ 
  (∀ x ∈ domain, f x (-2) ≥ -1) ∧ 
  (∃ x ∈ domain, f x (-2) = -1) := 
sorry

-- Problem 2: Proving the range of a for monotonicity on the given interval
theorem range_of_a_for_monotonicity: 
  (∀ a, (∀ x1 x2 ∈ domain, x1 ≤ x2 → f x1 a ≤ f x2 a) ↔ a ∈ (-∞ : ℝ, -6] ∪ [4, ∞)) :=
sorry

end max_min_f_a_minus_2_range_of_a_for_monotonicity_l314_314910


namespace part1_part2_l314_314967

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314967


namespace max_seven_acute_triangles_l314_314893

theorem max_seven_acute_triangles (P : Fin 5 → ℝ × ℝ) (h : ∀ i j k : Fin 5, i ≠ j ∧ j ≠ k ∧ i ≠ k → collinear ({P i, P j, P k}) = false) :
  ∃ T : Finset (Finset (Fin 5)), T.card ≤ 7 ∧ (∀ t ∈ T, acute (triangle (P (t.enum 0)) (P (t.enum 1)) (P (t.enum 2)))) :=
sorry

end max_seven_acute_triangles_l314_314893


namespace smallest_n_for_64_pow_k_gt_4_pow_n_l314_314733

theorem smallest_n_for_64_pow_k_gt_4_pow_n (k n : ℕ) (h64_eq : 64 = 2 ^ 6) (h4_eq : 4 = 2 ^ 2) (h_k_eq : k = 8) : 
  64 ^ k > 4 ^ n ↔ n < 24 :=
by
  have h64_pow : 64 ^ k = (2 ^ 6) ^ k, from congr_arg (λ x, x ^ k) h64_eq
  have h4_pow : 4 ^ n = (2 ^ 2) ^ n, from congr_arg (λ x, x ^ n) h4_eq
  rw [h64_pow, h4_pow] at *
  rw [h_k_eq, pow_mul] at *
  sorry

end smallest_n_for_64_pow_k_gt_4_pow_n_l314_314733


namespace log_10_23_between_integers_l314_314169

theorem log_10_23_between_integers :
  (∃ c d : ℤ, 1 < real.log 23 / real.log 10 ∧ real.log 23 / real.log 10 < 2 ∧ c = 1 ∧ d = 2 ∧ c + d = 3) :=
by
  sorry

end log_10_23_between_integers_l314_314169


namespace tuna_per_customer_l314_314604

noncomputable def total_customers := 100
noncomputable def total_tuna := 10
noncomputable def weight_per_tuna := 200
noncomputable def customers_without_fish := 20

theorem tuna_per_customer : (total_tuna * weight_per_tuna) / (total_customers - customers_without_fish) = 25 := by
  sorry

end tuna_per_customer_l314_314604


namespace expand_polynomial_l314_314845

variable (x : ℝ)

theorem expand_polynomial :
  (5 * x ^ 2 + 3 * x - 8) * (3 * x ^ 3) = 15 * x ^ 5 + 9 * x ^ 4 - 24 * x ^ 3 :=
begin
  sorry
end

end expand_polynomial_l314_314845


namespace simplify_sqrt_expression_l314_314736

theorem simplify_sqrt_expression :
  (sqrt 27 + sqrt 243) / sqrt 75 = 12 / 5 := 
by sorry

end simplify_sqrt_expression_l314_314736


namespace part1_solution_set_part2_range_l314_314986

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314986


namespace determine_a_l314_314044

theorem determine_a (a : ℝ) (h : (binom 5 3) * a^3 = 80) : a = 2 := by
  sorry

end determine_a_l314_314044


namespace math_problem_l314_314432

theorem math_problem : (7 ^ (-3)) ^ 0 + (7 ^ 0) ^ 4 = 2 := by
  sorry

end math_problem_l314_314432


namespace trapezoid_area_division_l314_314067

theorem trapezoid_area_division
  (ABCD : Trapezoid)
  (AB_parallel_DC : ABCD.AB ∥ ABCD.DC)
  (CE_angle_bisector_BCD : is_angle_bisector ABCD.CE (angle ABCD.BCD))
  (CE_perpendicular_AD : is_perpendicular ABCD.CE ABCD.AD)
  (DE_eq_2AE : 2 * segment_length ABCD.AE = segment_length ABCD.DE)
  (S1_eq_1 : area (region CE ABCD) = 1) :
  area (region subtract (area ABCD) (region CE ABCD)) = 7 / 8 :=
sorry

end trapezoid_area_division_l314_314067


namespace anthony_pets_left_is_8_l314_314424

def number_of_pets_left (initial_pets : ℕ) (lost_pets : ℕ) (fraction_died : ℚ) : ℕ :=
  initial_pets - lost_pets - (fraction_died * (initial_pets - lost_pets)).toInt

theorem anthony_pets_left_is_8 : number_of_pets_left 16 6 (1/5) = 8 :=
by
  sorry

end anthony_pets_left_is_8_l314_314424


namespace planes_perpendicular_to_same_line_parallel_l314_314803

theorem planes_perpendicular_to_same_line_parallel
  (P Q : Plane) (l : Line)
  (hP : Perpendicular P l) (hQ : Perpendicular Q l) : 
  Parallel P Q :=
sorry

end planes_perpendicular_to_same_line_parallel_l314_314803


namespace mike_initial_marbles_l314_314601

-- Defining the conditions
def gave_marble (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles
def marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles

-- Using the given conditions
def initial_mike_marbles : ℕ := 8
def given_marbles : ℕ := 4
def remaining_marbles : ℕ := 4

-- Proving the statement
theorem mike_initial_marbles :
  initial_mike_marbles - given_marbles = remaining_marbles :=
by
  -- The proof
  sorry

end mike_initial_marbles_l314_314601


namespace probability_one_absent_one_present_l314_314053

-- Define the conditions as per given problem
def prob_absent: ℝ := 1 / 20
def prob_present: ℝ := 1 - prob_absent

-- Lean theorem to state the expected probability condition
theorem probability_one_absent_one_present:
  ((prob_present * prob_absent) + (prob_absent * prob_present)) * 100 = 9.5 :=
by
  -- Proof goes here
  sorry

end probability_one_absent_one_present_l314_314053


namespace angle_is_67_l314_314205

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314205


namespace find_a_if_perpendicular_l314_314539

theorem find_a_if_perpendicular (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 2 = 0 → 2 * x + 3 * y + 1 = 0 → False) →
  a = -2 / 3 :=
by
  sorry

end find_a_if_perpendicular_l314_314539


namespace integer_value_of_expression_l314_314532

theorem integer_value_of_expression (m n p : ℕ) (h1 : 2 ≤ m) (h2 : m ≤ 9)
  (h3 : 2 ≤ n) (h4 : n ≤ 9) (h5 : 2 ≤ p) (h6 : p ≤ 9)
  (h7 : m ≠ n ∧ n ≠ p ∧ m ≠ p) :
  (m + n + p) / (m + n) = 1 :=
sorry

end integer_value_of_expression_l314_314532


namespace part1_solution_part2_solution_l314_314960

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314960


namespace toy_car_cost_l314_314607

theorem toy_car_cost
  (total_cars : ℕ)
  (sasha_work_fraction : ℝ)
  (proceeds_condition : ∀ w, ∃ p s, p + s = proceeds ∧ p / (p + s) = w)
  (money_transfer : ℝ)
  (equal_money_condition : ∀ pasha_extra sasha_extra, pasha_extra = money_transfer ∧ sasha_extra = 1 ∧ pasha_equal sasha_equal := pasha_extra - money_transfer = sasha_extra) :
  total_cars = 3 →
  sasha_work_fraction = 1 / 5 →
  money_transfer = 400 →
  proceeds * sasha_work_fraction = sasha_proceeds →
  proceeds * (1 - sasha_work_fraction) = pasha_proceeds →
  sasha_proceeds + (pasha_proceeds - money_transfer) * 1 = proceeds / 2 →
  proceeds / total_cars = 1000
:=
sorry

end toy_car_cost_l314_314607


namespace sum_of_b_when_quadratic_has_one_solution_l314_314834

theorem sum_of_b_when_quadratic_has_one_solution :
  (∀ b : ℝ, (let discriminant := ((b + 6)^2 - 4 * 3 * 10) in discriminant = 0 ↔ 
  (b = (Real.sqrt 120 - 6) ∨ b = -(Real.sqrt 120 + 6))) →
  (Real.sqrt 120 - 6) + (-(Real.sqrt 120 + 6)) = -12) :=
by
  sorry

end sum_of_b_when_quadratic_has_one_solution_l314_314834


namespace part1_solution_part2_solution_l314_314965

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314965


namespace matrix_power_four_l314_314821

theorem matrix_power_four :
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![3 * Real.sqrt 2, -3],
    ![3, 3 * Real.sqrt 2]
  ]
  (A ^ 4 = ![
    ![ -81, 0],
    ![0, -81]
  ]) :=
by
  sorry

end matrix_power_four_l314_314821


namespace work_hours_required_l314_314024

def initial_hours_per_week : ℝ := 15
def initial_weeks : ℕ := 10
def missed_weeks : ℕ := 3
def total_earnings : ℝ := 2250

-- Total weeks to work
def remaining_weeks : ℕ := initial_weeks - missed_weeks

-- Needed weekly work hours to earn the same amount in the remaining weeks
def new_hours_per_week (initial_hours_per_week : ℝ) (initial_weeks : ℕ) (remaining_weeks : ℕ) : ℝ :=
  (initial_weeks / remaining_weeks) * initial_hours_per_week

-- Using Numerics division
def target_hours_per_week : ℝ := new_hours_per_week initial_hours_per_week initial_weeks remaining_weeks

theorem work_hours_required :
  target_hours_per_week = 21.43 :=
begin
  -- Theorem to show work hours required equals 21.43
  sorry
end

end work_hours_required_l314_314024


namespace concurrency_equiv_l314_314014

variables (A B C A1 A2 B1 B2 C1 C2 : Type*) [Nonempty A] [Nonempty B] [Nonempty C]

-- Assume the conditions
variable (triangle_ABC : triangle A B C)
variable (points_on_sides : 
  (A1 ∈ side BC ∧ A2 ∈ side CA ∧ B1 ∈ side CA ∧ B2 ∈ side AB ∧ C1 ∈ side AB ∧ C2 ∈ side BC))
variable (dist_conditions :
  (dist A B A1 = dist A2 C) ∧ (dist C B B1 = dist B2 A) ∧ (dist A C C1 = dist C2 B))

-- Define lines
variable (lines :
  parallel line A A1 line l_a A2 ∧ 
  parallel line B B1 line l_b B2 ∧ 
  parallel line C C1 line l_c C2)

-- Define concurrency
def are_concurrent (line1 line2 line3 : Type*) := 
  ∃ point, line1 = line_through point ∧ line2 = line_through point ∧ line3 = line_through point

-- The goal statement
theorem concurrency_equiv :
  are_concurrent line A A1 line B B1 line C C1 ↔ 
  are_concurrent line l_a line l_b line l_c :=
sorry

end concurrency_equiv_l314_314014


namespace angle_measure_l314_314274

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314274


namespace area_of_union_of_triangles_l314_314860

theorem area_of_union_of_triangles :
  let s := 4
  let single_triangle_area := (sqrt 3 / 4) * s^2
  let total_area_without_overlap := 5 * single_triangle_area
  let overlapping_triangle_area := (sqrt 3 / 4) * (2)^2
  let total_overlap_area := 4 * overlapping_triangle_area
  let net_area := total_area_without_overlap - total_overlap_area
  net_area = 16 * sqrt 3 :=
by
  sorry

end area_of_union_of_triangles_l314_314860


namespace part1_part2_l314_314937

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314937


namespace same_selection_exists_l314_314648

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l314_314648


namespace part1_solution_set_part2_range_l314_314980

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314980


namespace part1_part2_l314_314993

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l314_314993


namespace vendor_sells_50_percent_second_day_l314_314408

theorem vendor_sells_50_percent_second_day (n : ℕ) (h1 : n > 0):
  let initial_remaining := n * 40 / 100 in
  let remaining_after_first_discard := initial_remaining * 85 / 100 in
  let total_discarded := n * 23 / 100 in
  let second_day_discard := total_discarded - (n * 15 / 100) in
  let second_day_sales := remaining_after_first_discard - second_day_discard in
  second_day_sales * 100 / remaining_after_first_discard = 50 :=
by
  sorry

end vendor_sells_50_percent_second_day_l314_314408


namespace common_divisors_120_n_l314_314725

theorem common_divisors_120_n (n : ℕ) (h1 : 120 % n = 0) (h2 : ∃ q : ℕ, q ∣ 120 ∧ (∀ d : ℕ, d ∣ 120 → d ∣ n → d ∈ {1, q, q^2})) : 
  let common_divisors := {1, 2, 4} in greatest_common_divisor 120 n = 4 :=
by
  sorry

end common_divisors_120_n_l314_314725


namespace no_valid_five_digit_number_l314_314127

def divisible_by (n d : ℕ) : Prop := n % d = 0

theorem no_valid_five_digit_number :
  ¬ ∃ (a b c d e : ℕ),
    {a, b, c, d, e} = {1, 2, 3, 4, 5} ∧
    divisible_by (10 * a + b) 2 ∧
    divisible_by (100 * a + 10 * b + c) 3 ∧
    divisible_by (1000 * a + 100 * b + 10 * c + d) 4 ∧
    divisible_by (10000 * a + 1000 * b + 100 * c + 10 * d + e) 5 := 
by 
  sorry

end no_valid_five_digit_number_l314_314127


namespace initial_legos_l314_314841

-- Definitions and conditions
def legos_won : ℝ := 17.0
def legos_now : ℝ := 2097.0

-- The statement to prove
theorem initial_legos : (legos_now - legos_won) = 2080 :=
by sorry

end initial_legos_l314_314841


namespace part1_solution_set_part2_range_of_a_l314_314919

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314919


namespace square_minor_premise_l314_314065

-- Definitions based on the problem's conditions
def major_premise : Prop := ∀ (P : Type) (p : P → Prop), p P → ∃ Q : P, Q = P
def minor_premise (Q : Type) (R : Type) : Prop := Q = R
def conclusion (Q : Type) (R : Type) : Prop := Q → R

-- The major premise in the syllogism
def diagonal_bisect_parallelogram : Prop := ∀ (A B C D : Type), A = B ∧ C = D → A = C

-- The minor premise
def square_is_parallelogram : Prop := minor_premise (Type 1) (Type 2)

-- The conclusion in the syllogism
def diagonals_bisect_square : Prop:= ∃ (X Y Z : Type), diagonal_bisect_parallelogram ∧ square_is_parallelogram ∧ (X = Y ∧ Y = Z)

-- The theorem stating that "A square is a parallelogram" is the minor premise
theorem square_minor_premise : square_is_parallelogram := 
by {
  exact sorry
}

end square_minor_premise_l314_314065


namespace find_m_sum_T_l314_314889

-- Define the arithmetic sequence properties
def S (n : ℕ) : ℝ := sorry -- Sum of the first n terms of sequence a
def a (n : ℕ) : ℝ := sorry -- nth term of sequence a

-- Define the sum and the properties given in the problem
axiom S_m1_eq_neg4 {m : ℕ} (hm : m ≥ 2) : S (m - 1) = -4
axiom S_m_eq_0 {m : ℕ} (hm : m ≥ 2) : S m = 0
axiom S_m2_eq_14 {m : ℕ} (hm : m ≥ 2) : S (m + 2) = 14

-- Define the sequence b as per the given condition
def b (n : ℕ) : ℝ := a^(n - 3)

-- Define the summation formula T
def T (n : ℕ) : ℝ := ∑ i in Finset.range n, (a i + 6) * b i -- Sum of the first n terms of sequence (a_n + 6) * b_n

-- Prove that m = 5
theorem find_m (m : ℕ) (hm : m ≥ 2) : m = 5 := sorry

-- Prove the sum of the first n terms formula for T
theorem sum_T (a : ℝ) (n : ℕ) : 
  T n = (2 * a^(-2) / (1 - a)^2) - ((2 * (n - n * a + 1) * a^(n - 2)) / (1 - a)^2) := sorry

end find_m_sum_T_l314_314889


namespace tan_alpha_eq_one_l314_314537

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.cos (α + β) = Real.sin (α - β)) : Real.tan α = 1 :=
sorry

end tan_alpha_eq_one_l314_314537


namespace V_not_measurable_l314_314441

open Set
open Classical

noncomputable theory

def equivalence_on_unit_interval (x y : ℝ) : Prop :=
  ∃ q : ℚ, x - y = (q : ℝ)

axiom axiom_of_choice (S : Set (Set α)) : ∃ (f : (Set α → α)), ∀ (B : Set α), B ∈ S → f B ∈ B

noncomputable def select_one_per_class :
  {V : Set ℝ // V ⊆ Icc 0 1 ∧ ∀ (x ∈ V) (y ∈ V), equivalence_on_unit_interval x y → x = y} :=
by
  choose v hv using axiom_of_choice (equivalence_on_unit_interval '' (Icc 0 1))
  use {x | ∃ y ∈ Icc 0 1, v (equivalence_on_unit_interval y) = x}
  split
  · intros x hx
    rcases hx with ⟨y, hy1, hy2⟩
    exact hy1
  · intros x hx y hy hxy
    rcases hx with ⟨x', hx', rfl⟩
    rcases hy with ⟨y', hy', rfl⟩
    by_contra h
    exact h (hv (equivalence_on_unit_interval y x') hxy)

theorem V_not_measurable : ¬ (measurable_set select_one_per_class.val) :=
sorry

end V_not_measurable_l314_314441


namespace part1_part2_l314_314969

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314969


namespace not_all_inequalities_hold_l314_314083

theorem not_all_inequalities_hold (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
(hlt_a : a < 1) (hlt_b : b < 1) (hlt_c : c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
by
  sorry

end not_all_inequalities_hold_l314_314083


namespace find_savings_l314_314153

-- Define the conditions and question as hypotheses
variables (income expenditure : ℕ)
variable (ratio : ℕ × ℕ)
variable (common_factor : ℕ)
variable (income_given : ℕ)

-- Translate the conditions to hypotheses
axiom ratio_condition : ratio = (8, 7)
axiom income_condition : income = 8 * common_factor
axiom income_given_condition : income_given = 40000

-- The Lean 4 statement of the proof problem
theorem find_savings 
  (h1 : ratio_condition)
  (h2 : income_condition)
  (h3 : income_given_condition)
  (h4 : income = income_given)
  : (income - expenditure) = 5000 :=
sorry

end find_savings_l314_314153


namespace angle_is_67_l314_314206

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314206


namespace range_of_a_l314_314001

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * a * x^2 - Real.log x

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 ∈ Set.Icc 1 Real.exp 1 → x2 ∈ Set.Icc 1 Real.exp 1 → |f a x1 - f a x2| ≤ 3) ↔ 
  a ∈ Set.Icc (4 / (1 - Real.exp 2)) (8 / (Real.exp 2 - 1)) := 
sorry

end range_of_a_l314_314001


namespace angle_measure_triple_complement_l314_314228

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314228


namespace geometric_series_sum_eq_l314_314858

theorem geometric_series_sum_eq :
  let a := (1/3 : ℚ)
  let r := (1/3 : ℚ)
  let n := 8
  let S := a * (1 - r^n) / (1 - r)
  S = 3280 / 6561 :=
by
  sorry

end geometric_series_sum_eq_l314_314858


namespace power_mod_equality_l314_314823

theorem power_mod_equality (n : ℕ) : 
  (47 % 8 = 7) → (23 % 8 = 7) → (47 ^ 2500 - 23 ^ 2500) % 8 = 0 := 
by
  intro h1 h2
  sorry

end power_mod_equality_l314_314823


namespace courtier_selection_l314_314681

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l314_314681


namespace divisor_and_remainder_correct_l314_314450

theorem divisor_and_remainder_correct:
  ∃ d r : ℕ, d ≠ 0 ∧ 1270 = 74 * d + r ∧ r = 12 ∧ d = 17 :=
by
  sorry

end divisor_and_remainder_correct_l314_314450


namespace number_of_valid_paintings_l314_314378

-- Define the grid type as a 3 by 3 matrix of colors (green or red)
inductive Color
| red
| green

def Grid := Matrix (Fin 3) (Fin 3) Color

-- A predicate that returns a boolean if a grid configuration is valid
def validGrid (g : Grid) : Prop :=
  ∀ i j, 
    (i < 2 → g i j ≠ g (i + 1) j) ∧ 
    (j < 2 → g i j ≠ g i (j + 1))

-- The theorem to prove:
theorem number_of_valid_paintings : 
  {g : Grid // validGrid g}.finite ∧ 
  {g : Grid // validGrid g}.card = 4 :=
sorry

end number_of_valid_paintings_l314_314378


namespace parabola_expression_shifted_parabola_expression_l314_314882

open Classical

section parabola

variable (b c : ℝ)

def parabola (x : ℝ) : ℝ := -1/2 * x^2 + b * x + c

-- Given conditions
axiom passes_through_1_0 : parabola b c 1 = 0
axiom passes_through_0_3_2 : parabola b c 0 = 3/2

-- First part: Prove the analytical expression of the parabola
theorem parabola_expression :
  parabola b c x = -1/2 * x^2 - x + 3/2 :=
sorry

-- Second part: Prove the vertex coincides with the origin after shift
theorem shifted_parabola_expression :
  parabola (shift_x b) (shift_c c) x = -1/2 * x^2 :=
sorry

end parabola

end parabola_expression_shifted_parabola_expression_l314_314882


namespace sum_all_positive_condition1_sum_all_not_necessarily_positive_condition2_l314_314793

-- Define the sequence as a function from Fin 100 to ℝ
variable (a : Fin 100 → ℝ)

-- Condition 1: sum of any 7 numbers is positive
def condition1 := ∀ (s : Finset (Fin 100)), s.card = 7 → 0 < (s.sum a)

-- Condition 2: sum of any 7 consecutive numbers is positive
def condition2 := ∀ (i : Fin 100), 0 < (Finset.range 7).sum (λ j, a ⟨(i + j) % 100, sorry⟩)

-- Prove the sum of all 100 numbers is positive under condition 1
theorem sum_all_positive_condition1 (h : condition1 a) : 0 < (Finset.range 100).sum a :=
sorry

-- Disprove that the sum of all 100 numbers must be positive under condition 2
theorem sum_all_not_necessarily_positive_condition2 (h : condition2 a) : ∃ (b : Fin 100 → ℝ),
  condition2 b ∧ (Finset.range 100).sum b ≤ 0 :=
sorry

end sum_all_positive_condition1_sum_all_not_necessarily_positive_condition2_l314_314793


namespace angle_measure_triple_complement_l314_314302

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314302


namespace mean_height_basketball_team_l314_314709

def heights : List ℕ :=
  [58, 59, 60, 62, 63, 65, 65, 68, 70, 71, 71, 72, 76, 76, 78, 79, 79]

def mean_height (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem mean_height_basketball_team :
  mean_height heights = 70 := by
  sorry

end mean_height_basketball_team_l314_314709


namespace angle_is_67_l314_314212

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314212


namespace angle_triple_complement_l314_314263

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314263


namespace lcm_24_36_45_l314_314341

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l314_314341


namespace angle_triple_complement_l314_314312

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314312


namespace BD_not_integer_l314_314082

theorem BD_not_integer
  (a b c : ℕ) (α β γ : ℝ) (R : ℝ) 
  (hc1 : a > 0) (hc2 : b > 0) (hc3 : c > 0)
  (hc4 : Nat.gcd a b = 1) (hc5 : Nat.gcd b c = 1) (hc6 : Nat.gcd a c = 1)
  (hc7 : α + β + γ = π) -- Sum of angles in triangle
  (hc8 : sin α = a / (2 * R)) (hc9 : sin β = b / (2 * R)) (hc10 : sin γ = c / (2 * R))
  (hc11 : is_tangent A X D A) -- Tangency at A to circumcircle
  : ¬ ∃ (BD : ℕ), BD > 0 :=
by
  sorry

end BD_not_integer_l314_314082


namespace mike_third_job_hourly_rate_l314_314600

-- Conditions
def total_earnings : ℝ := 430
def first_job_hours : ℝ := 15
def first_job_rate : ℝ := 8
def merchandise_sold : ℝ := 1000
def commission_rate : ℝ := 0.1
def taxes : ℝ := 50
def third_job_hours : ℝ := 12

-- Earnings calculations
def first_job_earnings : ℝ := first_job_hours * first_job_rate
def second_job_commission : ℝ := merchandise_sold * commission_rate
def combined_wages_before_taxes : ℝ := first_job_earnings + second_job_commission
def combined_wages_after_taxes : ℝ := combined_wages_before_taxes - taxes
def third_job_earnings : ℝ := total_earnings - combined_wages_after_taxes
def third_job_hourly_rate : ℝ := third_job_earnings / third_job_hours

-- Proof problem
theorem mike_third_job_hourly_rate : third_job_hourly_rate = 21.67 := by
  sorry

end mike_third_job_hourly_rate_l314_314600


namespace math_problem_168_l314_314574

theorem math_problem_168 :
  let T := ∑ n in {n : ℕ | ∃ m : ℕ, n^2 + 12 * n - 2023 = m^2}, n
  in T % 1000 = 168 :=
by
  sorry

end math_problem_168_l314_314574


namespace midpoint_AB_l314_314585

open_locale real_inner_product_space

variables {A B C M N O : Type*} [real_inner_product_space A] [affine_subspace B C A]
  (hM : is_midpoint A C M)
  (hN : N ∈ line_segment A B)
  (hO : lines_intersect (line_through B M) (line_through C N) O)
  (hAreaEq : area_triangle B O N = area_triangle C O M)

theorem midpoint_AB (hM : is_midpoint A C M)
  (hN : N ∈ line_segment A B)
  (hO : lines_intersect (line_through B M) (line_through C N) O)
  (hAreaEq : area_triangle B O N = area_triangle C O M) :
  is_midpoint A B N :=
sorry

end midpoint_AB_l314_314585


namespace all_values_of_K_real_roots_l314_314830

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem all_values_of_K_real_roots (K : ℝ) : 
  let a := K^3
      b := -(5 * K^3 + 1)
      c := 6 * K^3
  in discriminant a b c ≥ 0 := 
by
  let a := K^3
  let b := -(5 * K^3 + 1)
  let c := 6 * K^3
  have d : ℝ := discriminant a b c
  show d ≥ 0
  have eq_d : d = K^6 + 10 * K^3 + 1 := by 
    simp [discriminant, a, b, c]
  rw eq_d
  sorry

end all_values_of_K_real_roots_l314_314830


namespace two_courtiers_have_same_selection_l314_314688

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l314_314688


namespace angle_triple_complement_l314_314255

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314255


namespace part1_solution_part2_solution_l314_314956

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314956


namespace projectile_initial_distance_l314_314187

-- Define the conditions
def speed1_kmph : ℝ := 432
def speed2_kmph : ℝ := 576
def meeting_time_min : ℝ := 150

-- Convert speeds from km/h to km/min
def speed1_kmpmin : ℝ := speed1_kmph / 60
def speed2_kmpmin : ℝ := speed2_kmph / 60

-- Calculate distances traveled by each projectile
def distance1_km : ℝ := speed1_kmpmin * meeting_time_min
def distance2_km : ℝ := speed2_kmpmin * meeting_time_min

-- Total initial distance
def total_initial_distance_km : ℝ := distance1_km + distance2_km

-- The theorem to be proven
theorem projectile_initial_distance :
  total_initial_distance_km = 2520 := by
  sorry

end projectile_initial_distance_l314_314187


namespace choose_athlete_l314_314769

noncomputable def select_athlete : String :=
  let avg_甲 : ℕ := 91
  let var_甲 : ℕ := 32
  let avg_乙 : ℕ := 93
  let var_乙 : ℕ := 32
  let avg_丙 : ℕ := 93
  let var_丙 : ℕ := 21
  let avg_丁 : ℕ := 91
  let var_丁 : ℕ := 21

  if avg_甲 ≥ avg_乙 ∧ avg_甲 ≥ avg_丙 ∧ avg_甲 ≥ avg_丁 then
    if var_甲 < var_乙 ∧ var_甲 < var_丙 ∧ var_甲 < var_丁 then "甲" else ""
  else if avg_乙 ≥ avg_丙 ∧ avg_乙 ≥ avg_丁 then
    if var_乙 < var_丙 ∧ var_乙 < var_丁 then "乙" else "丙"
  else if avg_丙 ≥ avg_丁 then
    if var_丙 < var_丁 then "丙" else "丁"
  else "丁"

theorem choose_athlete :
  select_athlete = "丙" := by
  -- Proof would go here
  sorry

end choose_athlete_l314_314769


namespace sin_cos_product_l314_314027

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l314_314027


namespace courtier_selection_l314_314676

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l314_314676


namespace sin_cos_product_l314_314031

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l314_314031


namespace part1_part2_l314_314996

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l314_314996


namespace sum_of_a_and_b_eq_three_l314_314509

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 5

theorem sum_of_a_and_b_eq_three
  (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : b = a + 1)
  (h_root : ∃ x₀, x₀ ∈ set.Icc (a : ℝ) (b : ℝ) ∧ f x₀ = 0) : 
  a + b = 3 := 
sorry

end sum_of_a_and_b_eq_three_l314_314509


namespace lcm_24_36_45_l314_314336

open Nat

theorem lcm_24_36_45 : lcm (lcm 24 36) 45 = 360 := by sorry

end lcm_24_36_45_l314_314336


namespace distance_between_B_and_C_l314_314865

/-- Four gas stations located on a circular road: A, B, C, and D with given distances. -/
theorem distance_between_B_and_C 
  (A B C D : Type) 
  (dist : A → A → ℝ) 
  (d_AB : dist A B = 50) 
  (d_AC : dist A C = 40) 
  (d_CD : dist C D = 25) 
  (d_DA : dist D A = 35) 
  (circumference : ∀ x y z : A, dist x y + dist y z + dist z x = 100) :
  dist B C = 10 ∨ dist B C = 90 :=
sorry

end distance_between_B_and_C_l314_314865


namespace calculate_r_over_s_at_1_l314_314638

noncomputable def r (x : ℝ) : ℝ := 3 * (x - 4)
noncomputable def s (x : ℝ) : ℝ := (x - 4) * (x - 3)

theorem calculate_r_over_s_at_1 :
  let y := r 1 / s 1
  y = -15 / 2 :=
begin
  sorry
end

end calculate_r_over_s_at_1_l314_314638


namespace same_selection_exists_l314_314647

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l314_314647


namespace area_of_lattice_triangle_le_half_l314_314508

theorem area_of_lattice_triangle_le_half (A B C : ℤ × ℤ) (O : ℤ × ℤ)
  (h_O_interior : O ∈ interior (triangle A B C))
  (h_only_one_interior : ∀ P, P ∈ interior (triangle A B C) → P = O)
  (h_lattice_points_on_edges : ∀ P, P ∈ boundary (triangle A B C) → P ∈ lattice_points_on_edges (triangle A B C)) :
  area (triangle A B C) ≤ 9 / 2 :=
sorry

end area_of_lattice_triangle_le_half_l314_314508


namespace direction_vector_of_arithmetic_sequence_l314_314710

theorem direction_vector_of_arithmetic_sequence :
  ∀ (n : ℕ), n > 0 →
  let a_n := λ n : ℕ, 4 * n - 1 in
  let P := (n, a_n n) in
  let Q := (n + 2, a_n (n + 2)) in
  let direction_vector := (Q.1 - P.1, Q.2 - P.2) in
  a_n 1 + a_n 2 = 10 ∧ 4 * (a_n 1 + a_n 2 + a_n 3 + a_n 4) / 4 = 36 →
  direction_vector = (-1 / 2, -2) :=
by
  intros n hn a_n P Q direction_vector h
  sorry

end direction_vector_of_arithmetic_sequence_l314_314710


namespace Piravena_trip_distance_is_12000_l314_314608

-- Define the conditions for the problem
variable (CA AB : ℝ)
variable (right_angle_at_C : Prop)

-- Assume city distances and right-angled triangle condition
axiom h1 : CA = 4000
axiom h2 : AB = 5000
axiom h3 : right_angle_at_C = (AB^2 = (BC)^2 + (CA)^2)

-- Compute BC using Pythagorean theorem
noncomputable def BC : ℝ := Real.sqrt (AB^2 - CA^2)

-- Define the total distance function
def total_distance (AB BC CA : ℝ) : ℝ := AB + BC + CA

-- State the proof problem
theorem Piravena_trip_distance_is_12000 :
  total_distance AB BC CA = 12000 := 
by
  -- skip the proof
  sorry

end Piravena_trip_distance_is_12000_l314_314608


namespace part1_solution_set_part2_range_of_a_l314_314918

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314918


namespace proposition_1_proposition_2_proposition_3_proposition_4_correct_propositions_l314_314106

def f (x : ℝ) (b c : ℝ) : ℝ := x * abs x + b * x + c

theorem proposition_1 (b : ℝ) :
  (∀ x : ℝ, f (-x) b 0 = -f x b 0) := 
    sorry

theorem proposition_2 (c : ℝ) (h : c > 0) :
  (∃! x : ℝ, f x 0 c = 0) := 
    sorry

theorem proposition_3 (b c : ℝ) :
  (∀ x : ℝ, f (-x) b c + f x b c = 2 * c) := 
    sorry

theorem proposition_4 (b c : ℝ) :
  (∀ x : ℝ, f x b c = 0 → x^2 ≤ 4 * c) := 
    sorry

theorem correct_propositions (b c : ℝ) (h1 : c = 0) (h2 : h : c > 0) :
  (proposition_1 b) ∧ (proposition_2 c h) ∧ (proposition_3 b c) ∧ ¬(proposition_4 b c) := 
    sorry

end proposition_1_proposition_2_proposition_3_proposition_4_correct_propositions_l314_314106


namespace g_is_even_function_l314_314560

def g (x : ℝ) : ℝ := log (x^2)

theorem g_is_even_function : ∀ x : ℝ, x ≠ 0 → g x = g (-x) :=
by
  intro x hx
  sorry

end g_is_even_function_l314_314560


namespace part1_solution_set_part2_range_l314_314981

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314981


namespace race_outcomes_l314_314414

theorem race_outcomes :
  let participants := 6 in
  (participants * (participants - 1) * (participants - 2) * (participants - 3)) = 360 :=
by
  sorry

end race_outcomes_l314_314414


namespace focus_of_ellipse_isosceles_triangle_range_l314_314908

-- Definition of the ellipse E
def ellipse (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + y^2 = 1

-- Problem (I)
theorem focus_of_ellipse (m : ℝ) (x y : ℝ) (hx : x = Real.sqrt 3) (hy : y = 0)
  (he : ellipse m x y) : m = 1 / 4 := 
sorry

-- Problem (II)
theorem isosceles_triangle_range (m : ℝ)
  (h1 : ∀ x1 y1 x2 y2 : ℝ, (0 < x1 ∧ 0 < x2) → 
                            ellipse m x1 y1 →
                            ellipse m x2 y2 →
                            ((x1, y1) ≠ (0, 1) ∧ (x2, y2) ≠ (0, 1)) →
                            (m * x1^2 + y1^2 = 1 ∧ m * x2^2 + y2^2 = 1) →
                            (let k1 := -(y1 - 1) / x1 in
                             let k2 := -(y2 - 1) / x2 in
                             k1 * k2 = -1) →
                            ((k1 ≠ k2) ∧ (k1 ≠ -k2))) :
  0 < m ∧ m < 1 / 3 :=
sorry

end focus_of_ellipse_isosceles_triangle_range_l314_314908


namespace courtiers_dog_selection_l314_314650

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l314_314650


namespace shooting_sequences_count_l314_314548

theorem shooting_sequences_count :
  let targets_A := 4
  let targets_B := 3
  let targets_C := 2
  let targets_D := 1
  let total_targets := targets_A + targets_B + targets_C + targets_D
  let sequence := ["A", "A", "A", "A", "B", "B", "B", "C", "C", "D"]
  ∀ targets_sequences : Finset (List String),
    targets_sequences = (sequence.permutations : Finset (List String)) →
    targets_sequences.card = 12600 :=
by
  simp only
  sorry -- Proof can be filled in later

end shooting_sequences_count_l314_314548


namespace part1_solution_set_part2_value_of_m_l314_314107

def f(x : ℝ) : ℝ := |2 * x - 3|
def g(x m : ℝ) : ℝ := f(x + m) + f(x - m)

theorem part1_solution_set (x : ℝ) :
  |2 * x - 3| > 5 - |x + 2| ↔ x ∈ set.Ioo (-(⨯∞) : ℝ) 0 ∪ set.Ioo 2 (∞) := sorry

theorem part2_value_of_m (m : ℝ) :
  (∀ x : ℝ, g(x, m) ≥ 4) ∧ (∃ x : ℝ, g(x, m) = 4) ↔ m = 1 ∨ m = -1 := sorry

end part1_solution_set_part2_value_of_m_l314_314107


namespace number_of_ways_to_balance_weights_l314_314175

def weights : List ℕ := [1, 2, 3, 4, 5, 6, 7]
def target_sum : ℕ := 14

theorem number_of_ways_to_balance_weights: 
  (List.partition (λ s, s.sum = target_sum) (List.powerset weights)).length = 4 := 
sorry

end number_of_ways_to_balance_weights_l314_314175


namespace quadratic_residue_mod_power_l314_314583

theorem quadratic_residue_mod_power
  (p : ℕ) (hp : p.prime) (odd_p : p % 2 = 1)
  (a : ℕ) (h : ∃ y : ℕ, y^2 ≡ a [ZMOD p]) :
  ∀ k : ℕ, k ≥ 0 → ∃ z : ℕ, z^2 ≡ a [ZMOD (p^k)] := 
by
  intro k hk
  -- Assuming proof here but omitted due to sorry
  sorry

end quadratic_residue_mod_power_l314_314583


namespace two_courtiers_have_same_selection_l314_314682

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l314_314682


namespace same_selection_exists_l314_314649

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l314_314649


namespace unused_combinations_eq_40_l314_314810

-- Defining the basic parameters
def num_resources : ℕ := 6
def total_combinations : ℕ := 2 ^ num_resources
def used_combinations : ℕ := 23

-- Calculating the number of unused combinations
theorem unused_combinations_eq_40 : total_combinations - 1 - used_combinations = 40 := by
  sorry

end unused_combinations_eq_40_l314_314810


namespace proof_problem_l314_314492

variable (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (T_n : ℕ → ℕ)
variable (m : ℝ)

-- Define the properties of the arithmetic sequence
def arithmetic_sequence (a_n : ℕ → ℕ) (a_5 a_4_plus_a_8 : ℕ) : Prop :=
  a_5 = 6 ∧ a_4_plus_a_8 = 14

-- Define the sum of the first n terms
def sum_of_terms (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) : Prop :=
  S_n = λ n, n * (n + 3) / 2

-- Define the sequence {b_n} and its properties
def sequence_b (b_n : ℕ → ℕ) (S_n : ℕ → ℕ) : Prop :=
  ∀ n, b_n n = 2 / (n * (n + 1))

-- Define the sum of the first n terms of {b_n}
def sum_b_terms (T_n : ℕ → ℕ) (b_n : ℕ → ℕ) : Prop :=
  T_n = λ n, 2 - 2 / (n + 1)

-- Main proof problem statement
theorem proof_problem :
  (∃ a_n S_n, arithmetic_sequence a_n 6 14 ∧ sum_of_terms S_n a_n ∧ 
    (∀ n, S_n(n) - n = 1 / b_n(n)) ∧ sequence_b b_n S_n ∧ sum_b_terms T_n b_n) →
  (∀ n, T_n n < m) → (2 ≤ m) :=
begin
  sorry
end

end proof_problem_l314_314492


namespace triple_complement_angle_l314_314277

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314277


namespace area_of_circle_l314_314507

-- Define the conditions 
def line (m : ℝ) (x : ℝ) : ℝ := m * x
def circle (m : ℝ) (x y : ℝ) : Prop := (x - m)^2 + (y - 1)^2 = m^2 - 1

-- Prove the area of the circle
theorem area_of_circle (m : ℝ) 
  (h_intersection : ∃ A B : ℝ × ℝ, 
                      circle m A.1 A.2 ∧ 
                      circle m B.1 B.2 ∧ 
                      line m A.1 = A.2 ∧ 
                      line m B.1 = B.2 ∧ 
                      ∃ C : ℝ × ℝ, 
                        C = (m, 1) ∧ 
                        ∠ ACB = 60) :
  ∃ r : ℝ, r = sqrt (m^2 - 1) ∧ π * r^2 = 6 * π :=
by 
  sorry

end area_of_circle_l314_314507


namespace ratio_of_averages_l314_314796

variable {x : Fin 50 → ℝ}

def true_average : ℝ :=
  (∑ i, x i) / 50

def erroneous_average : ℝ :=
  (∑ i, x i + true_average * 2) / 52

theorem ratio_of_averages :
  erroneous_average / true_average = 1 :=
by
  sorry

end ratio_of_averages_l314_314796


namespace quadrilateral_area_relation_l314_314573

-- Assuming the definitions related to geometrical constructs
variables {A B C D M : Type} 
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] 
variables [AffineSpace ℝ D] [AffineSpace ℝ M] 
variables {ABCD : Affineℝ} {H_A H_B H_C H_D : Affineℝ} 

-- Conditions: supplements
def is_convex_quadrilateral (Q : Type): ...

def centroid_ABC (A B C : Type) : Type := ...

def centroid_triangle (X Y Z : Type) : Type := ...

def centroid_of_ABC [is_convex_quadrilateral ABCD] (A B C : Type) (M : Type) := 
  centroid_ABC A B C = M

def centroids_of_triangles [is_convex_quadrilateral ABCD] (M X Y Zi : Type) :=
  centroid_triangle M B C = H_A ∧ 
  centroid_triangle M C A = H_B ∧ 
  centroid_triangle M A B = H_C ∧ 
  centroid_triangle M C D = H_D

-- Problem statement
theorem quadrilateral_area_relation 
  (H_A H_B H_C H_D : Type) [centroids_of_triangles ]:
  [H_A H_B H_C H_D] / [ABCD] = 1/9 :=
sorry

end quadrilateral_area_relation_l314_314573


namespace probability_red_envelopes_l314_314838

noncomputable def P_of_red_envelopes : ℚ :=
(4.choose 2) * (0.4^2) * (0.6^2) + (4.choose 3) * (0.4^3) * (0.6)

theorem probability_red_envelopes :
  P_of_red_envelopes = 312 / 625 :=
sorry

end probability_red_envelopes_l314_314838


namespace simplify_fraction_l314_314729

theorem simplify_fraction : 1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 :=
by
sorry

end simplify_fraction_l314_314729


namespace polynomial_functions_count_equals_16_l314_314470

open Classical

noncomputable def polynomial_functions_number : ℕ :=
  let count := #((λ (f : ℝ → ℝ), ∃ a b c d : ℝ, f = (λ x, a * x^3 + b * x^2 + c * x + d) ∧ 
                      (f(x) * f(-x) = f(x^3))) 
                  | a b c d, 
                    (a = 0 ∨ a = -1) ∧ 
                    (b = 0 ∨ b = 1) ∧ 
                    (c = 0 ∨ c = -1) ∧ 
                    (d = 0 ∨ d = 1)) in 
   count.toNat

theorem polynomial_functions_count_equals_16 : polynomial_functions_number = 16 :=
  sorry

end polynomial_functions_count_equals_16_l314_314470


namespace number_of_valid_functions_l314_314468

noncomputable def count_functions : ℕ := by
  let a_vals := [-1, 0]
  let b_vals := [-1, 0]
  let c_vals := [0, 1]
  let d_vals := [0, 1]
  let valid_functions := 
    { f : ℝ → ℝ // ∃ (a ∈ a_vals) (b ∈ b_vals) (c ∈ c_vals) (d ∈ d_vals), 
      f = λ x, a * x^3 + b * x^2 + c * x + d ∧ 
      ∀ x, (f x) * (f (-x)) = a * x^9 + b * x^6 + c * x^3 + d }
  exact Set.card valid_functions

theorem number_of_valid_functions : count_functions = 16 := by
  sorry

end number_of_valid_functions_l314_314468


namespace miles_to_add_per_week_l314_314077

theorem miles_to_add_per_week :
  ∀ (initial_miles_per_week : ℝ) 
    (percentage_increase : ℝ) 
    (total_days : ℕ) 
    (days_in_week : ℕ),
    initial_miles_per_week = 100 →
    percentage_increase = 0.2 →
    total_days = 280 →
    days_in_week = 7 →
    ((initial_miles_per_week * (1 + percentage_increase)) - initial_miles_per_week) / (total_days / days_in_week) = 3 :=
by
  intros initial_miles_per_week percentage_increase total_days days_in_week
  intros h1 h2 h3 h4
  sorry

end miles_to_add_per_week_l314_314077


namespace find_m_l314_314775

def g (n : ℤ) : ℤ :=
if n % 2 = 1 then n + 5 else n / 2

theorem find_m (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 35) : m = 135 :=
sorry

end find_m_l314_314775


namespace sum_of_segments_eq_l314_314088

variable {A B C O P Q : Point}

-- Assume the necessary Point and Line definitions along with distances
axiom acute_triangle (A B C : Point) : Prop
axiom circumcenter (O : Point) (A B C : Point) : Prop
axiom parallel (l1 l2 : Line) : Prop
axiom distance (p1 p2 : Point) : ℝ
axiom line_through (p1 p2 : Point) : Line
axiom intersects (l : Line) (p : Point) (s : Segment) : Prop

-- Given conditions
axiom h1 : acute_triangle A B C
axiom h2 : circumcenter O A B C
axiom h3 : ∃ g, parallel g (line_through B C) ∧ g = line_through O Q ∧ intersects g (line_through A B) P ∧ intersects g (line_through A C) Q
axiom h4 : distance O (line_through A B) + distance O (line_through A C) = distance O A

-- Required to prove
theorem sum_of_segments_eq (h1 : acute_triangle A B C)
                           (h2 : circumcenter O A B C)
                           (h3 : ∃ g, parallel g (line_through B C) ∧ g = line_through O Q ∧ intersects g (line_through A B) P ∧ intersects g (line_through A C) Q)
                           (h4 : distance O (line_through A B) + distance O (line_through A C) = distance O A)
                           : distance P B + distance Q C = distance P Q :=
by
  sorry

end sum_of_segments_eq_l314_314088


namespace five_digit_palindromes_count_l314_314397

def num_five_digit_palindromes : ℕ :=
  let choices_for_A := 9
  let choices_for_B := 10
  let choices_for_C := 10
  choices_for_A * choices_for_B * choices_for_C

theorem five_digit_palindromes_count : num_five_digit_palindromes = 900 :=
by
  unfold num_five_digit_palindromes
  sorry

end five_digit_palindromes_count_l314_314397


namespace probability_arithmetic_progression_l314_314390

theorem probability_arithmetic_progression (num_outcomes : ℕ) (num_favorable : ℕ) :
  num_outcomes = 6^3 ∧ num_favorable = 18 →
  num_favorable / num_outcomes = 1 / 12 :=
begin
  intros h,
  have h_outcomes : num_outcomes = 6^3 := h.1,
  have h_favorable : num_favorable = 18 := h.2,
  rw [h_outcomes, h_favorable],
  norm_num,
  exact one_div 12,
end

end probability_arithmetic_progression_l314_314390


namespace remainder_a4_mod_n_eq_one_l314_314582

variable (n : ℕ) (a : ℤ)
variable [fact (odd n)] [fact (1 < n)]

theorem remainder_a4_mod_n_eq_one
  (h : (a^2) % n = (a^(-2)) % n) : (a^4) % n = 1 % n :=
sorry

end remainder_a4_mod_n_eq_one_l314_314582


namespace james_running_increase_l314_314074

theorem james_running_increase (initial_miles_per_week : ℕ) (percent_increase : ℝ) (total_days : ℕ) (days_in_week : ℕ) :
  initial_miles_per_week = 100 →
  percent_increase = 0.2 →
  total_days = 280 →
  days_in_week = 7 →
  ∃ miles_per_week_to_add : ℝ, miles_per_week_to_add = 3 :=
by
  intros
  sorry

end james_running_increase_l314_314074


namespace largest_prime_divisor_of_expression_l314_314852

theorem largest_prime_divisor_of_expression : 
  ∀ (a b : ℕ), a = 39 → b = 52 → (∀ p : ℕ, prime p → p ∣ (a * a + b * b) → p ≤ 13) ∧ (13 ∣ (a * a + b * b)) := 
by
  intros a b ha hb hprime hpdiv
  sorry

end largest_prime_divisor_of_expression_l314_314852


namespace sallys_change_l314_314118

-- Define the total cost calculation:
def totalCost (numFrames : Nat) (costPerFrame : Nat) : Nat :=
  numFrames * costPerFrame

-- Define the change calculation:
def change (totalAmount : Nat) (amountPaid : Nat) : Nat :=
  amountPaid - totalAmount

-- Define the specific conditions in the problem:
def numFrames := 3
def costPerFrame := 3
def amountPaid := 20

-- Prove that the change Sally gets is $11:
theorem sallys_change : change (totalCost numFrames costPerFrame) amountPaid = 11 := by
  sorry

end sallys_change_l314_314118


namespace unused_combinations_eq_40_l314_314809

-- Defining the basic parameters
def num_resources : ℕ := 6
def total_combinations : ℕ := 2 ^ num_resources
def used_combinations : ℕ := 23

-- Calculating the number of unused combinations
theorem unused_combinations_eq_40 : total_combinations - 1 - used_combinations = 40 := by
  sorry

end unused_combinations_eq_40_l314_314809


namespace exponent_multiplication_l314_314818

theorem exponent_multiplication :
  let a := (1 : ℚ) / 3
  let b := (2 : ℚ) / 5
  a^9 * b^(-4) = 625 / 314928 :=
by
  sorry

end exponent_multiplication_l314_314818


namespace part1_part2_l314_314939

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314939


namespace angle_triple_complement_l314_314199

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314199


namespace part1_solution_set_part2_range_of_a_l314_314916

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314916


namespace tan_squared_sum_l314_314883

theorem tan_squared_sum (P : ℝ × ℝ) (α β : ℝ) 
  (hP : ∃ θ : ℝ, P = (1 + Real.cos θ, 1 + Real.sin θ))
  (hα : ∀ A B C D : ℝ × ℝ, P ∈ Set.incircle (Square A B C D) → ∠APC = α)
  (hβ : ∀ A B C D : ℝ × ℝ, P ∈ Set.incircle (Square A B C D) → ∠BPD = β) :
  Real.tan α ^ 2 + Real.tan β ^ 2 = 8 := 
sorry

end tan_squared_sum_l314_314883


namespace points_of_third_l314_314051

noncomputable def points_of_first : ℕ := 11
noncomputable def points_of_second : ℕ := 7
noncomputable def points_of_fourth : ℕ := 2
noncomputable def johns_total_points : ℕ := 38500

theorem points_of_third :
  ∃ x : ℕ, (points_of_first * points_of_second * x * points_of_fourth ∣ johns_total_points) ∧
    (johns_total_points / (points_of_first * points_of_second * points_of_fourth)) = x := 
sorry

end points_of_third_l314_314051


namespace profit_calculation_l314_314754

-- Define conditions based on investments
def JohnInvestment := 700
def MikeInvestment := 300

-- Define the equality condition where John received $800 more than Mike
theorem profit_calculation (P : ℝ) 
  (h1 : (P / 6 + (7 / 10) * (2 * P / 3)) - (P / 6 + (3 / 10) * (2 * P / 3)) = 800) : 
  P = 3000 := 
sorry

end profit_calculation_l314_314754


namespace count_four_digit_numbers_with_thousands_digit_five_l314_314526

theorem count_four_digit_numbers_with_thousands_digit_five :
  ∃ n : ℕ, n = 1000 ∧ ∀ x : ℕ, 5000 ≤ x ∧ x ≤ 5999 ↔ (x - 4999) ∈ (finset.range 1000 + 1) :=
by
  sorry

end count_four_digit_numbers_with_thousands_digit_five_l314_314526


namespace angle_measure_triple_complement_l314_314239

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314239


namespace students_tour_participation_l314_314866

theorem students_tour_participation :
  let students := {A, B, C, D}
  let days := {Saturday, Sunday}
  ∃ (saturday sunday : Set α),
    saturday ∪ sunday = students ∧
    saturday ∩ sunday = ∅ ∧
    saturday ≠ ∅ ∧
    sunday ≠ ∅ ∧
    (∀ a ∈ students, a ∈ saturday ∨ a ∈ sunday) →
    saturday.card + sunday.card = 4 →
    saturday.card ∈ {1, 2, 3} →
    (students_choose_days : (students → days) → ℕ)
    ∃ n : ℕ, n = 14 := sorry

end students_tour_participation_l314_314866


namespace cannot_be_divided_into_parallelograms_l314_314561

noncomputable def isosceles_triangle (a : ℝ) (θ : ℝ) (hx : 0 < θ ∧ θ < π / 2) : Triangle :=
{ base := a,
  height := a * (cos θ) }

noncomputable def resulting_figure (a : ℝ) (θ : ℝ) 
  (hx : 0 < θ ∧ θ < π / 2) : Polygon :=
{ square := Square (side_length := a),
  triangles := λ s ∈ (Square.edges) => isosceles_triangle a θ hx }

noncomputable def can_be_split_into_parallelograms (fig : Polygon) : Prop :=
fig.decomposition.all (λ piece, is_parallelogram piece)

theorem cannot_be_divided_into_parallelograms 
  (a : ℝ) (θ : ℝ) (hx : 0 < θ ∧ θ < π / 2) : 
  ¬ can_be_split_into_parallelograms (resulting_figure a θ hx) :=
sorry

end cannot_be_divided_into_parallelograms_l314_314561


namespace quartic_polynomial_roots_l314_314459

noncomputable def Q (x : ℝ) : ℝ := x^4 - 5*x^3 + 9*x^2 - 13*x + 14

theorem quartic_polynomial_roots : Q (real.cbrt 7 + 1) = 0 ∧ Q 2 = 0 :=
by
  have h1 : Q (real.cbrt 7 + 1) = (real.cbrt 7 + 1)^4 - 5*(real.cbrt 7 + 1)^3 + 9*(real.cbrt 7 + 1)^2 - 13*(real.cbrt 7 + 1) + 14 
    := by unfold Q
  -- Proving h1 is Q (\sqrt[3]{7} + 1) = 0
  sorry,

  have h2 : Q 2 = 2^4 - 5*2^3 + 9*2^2 - 13*2 + 14 
    := by unfold Q
  -- Proving h2 is Q (2) = 0
  sorry

  exact ⟨h1, h2⟩

end quartic_polynomial_roots_l314_314459


namespace main_theorem_l314_314570

noncomputable def twice_continuously_differentiable (f : ℝ → ℝ → ℝ) : Prop :=
  differentiable ℝ f ∧ differentiable ℝ (λ x, deriv (λ y, f x y)) ∧ differentiable ℝ (λ x, deriv (λ y, f y x))

variable (F : ℝ × ℝ → ℝ)
variable (g : ℝ → ℝ)

-- Given conditions as definitions
def cond1 : Prop := twice_continuously_differentiable F
def cond2 : Prop := ∀ u : ℝ, F (u, u) = 0
def cond3 : Prop := ∀ x : ℝ, g x > 0 ∧ x^2 * g x ≤ 1
def cond4 : Prop := ∀ (u v : ℝ), ∃ k : ℝ, ∇ F (u,v) = k •  ⟨g u, -g v⟩ ∨ ∇ F (u,v) = 0

-- The main theorem to be proven
theorem main_theorem (C : ℝ) : (cond1 F) → (cond2 F) → (cond3 g) → (cond4 F g) →
  ∀ (n : ℕ), n ≥ 2 → ∀ (x : vector ℝ (n + 1)), 
  ∃ i j : fin (n + 1), i ≠ j ∧ |F (x i, x j)| ≤ C / n := 
by
  intros
  -- proof omitted
  sorry

end main_theorem_l314_314570


namespace taco_castle_num_dodge_trucks_l314_314557

theorem taco_castle_num_dodge_trucks
  (D F T V H C : ℕ)
  (hV : V = 5)
  (h1 : F = D / 3)
  (h2 : F = 2 * T)
  (h3 : V = T / 2)
  (h4 : H = 3 * F / 4)
  (h5 : C = 2 * H / 3) :
  D = 60 :=
by
  sorry

end taco_castle_num_dodge_trucks_l314_314557


namespace same_selection_exists_l314_314645

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l314_314645


namespace tan_alpha_tan_beta_is_2_l314_314026

theorem tan_alpha_tan_beta_is_2
  (α β : ℝ)
  (h1 : Real.sin α = 2 * Real.sin β)
  (h2 : Real.sin (α + β) * Real.tan (α - β) = 1) :
  Real.tan α * Real.tan β = 2 :=
sorry

end tan_alpha_tan_beta_is_2_l314_314026


namespace average_speed_l314_314767

-- Define the speeds and times
def speed1 : ℝ := 120 -- km/h
def time1 : ℝ := 1 -- hour

def speed2 : ℝ := 150 -- km/h
def time2 : ℝ := 2 -- hours

def speed3 : ℝ := 80 -- km/h
def time3 : ℝ := 0.5 -- hour

-- Define the conversion factor
def km_to_miles : ℝ := 0.62

-- Calculate total distance (in kilometers)
def distance1 : ℝ := speed1 * time1
def distance2 : ℝ := speed2 * time2
def distance3 : ℝ := speed3 * time3

def total_distance_km : ℝ := distance1 + distance2 + distance3

-- Convert total distance to miles
def total_distance_miles : ℝ := total_distance_km * km_to_miles

-- Calculate total time (in hours)
def total_time : ℝ := time1 + time2 + time3

-- Final proof statement for average speed
theorem average_speed : total_distance_miles / total_time = 81.49 := by {
  sorry
}

end average_speed_l314_314767


namespace quadratic_discriminant_l314_314324

theorem quadratic_discriminant : 
  let a := 4
  let b := -6
  let c := 9
  (b^2 - 4 * a * c = -108) := 
by
  sorry

end quadratic_discriminant_l314_314324


namespace problem1_problem2_l314_314373

-- Problem (Ⅰ)
theorem problem1 : 
  (tan 150 * cos 210 * sin (-60)) / (sin (-30) * cos 120) = -2 := 
sorry

-- Problem (Ⅱ)
theorem problem2 (α : ℝ) : 
  (sin (-α) * cos (π + α) * tan (2 * π + α)) / (cos (2 * π + α) * sin (π - α) * tan (-α)) = -1 := 
sorry

end problem1_problem2_l314_314373


namespace courtiers_selection_l314_314667

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l314_314667


namespace angle_is_67_l314_314210

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314210


namespace number_of_white_pieces_l314_314052

theorem number_of_white_pieces (W B : ℕ) (hB : B = 3) (h1 : (fact W) * (fact B) = 144) : W = 4 :=
by
  sorry

end number_of_white_pieces_l314_314052


namespace coin_probability_l314_314805

theorem coin_probability (p : ℚ) 
  (P_X_3 : ℚ := 10 * p^3 * (1 - p)^2)
  (P_X_4 : ℚ := 5 * p^4 * (1 - p))
  (P_X_5 : ℚ := p^5)
  (w : ℚ := P_X_3 + P_X_4 + P_X_5) :
  w = 5 / 16 → p = 1 / 4 :=
by
  sorry

end coin_probability_l314_314805


namespace circle_through_intersections_distance_AB_l314_314484

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the circle passing through the intersection points of the parabola with the axes
def circle_eqn (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line_eqn (x y : ℝ) : Prop := 2*x - y + 2 = 0

-- Prove the points of intersections and the circle equation
theorem circle_through_intersections : 
  (parabola (1 : ℝ) = 0) ∧ 
  (parabola (3 : ℝ) = 0) ∧ 
  (parabola (0 : ℝ) = 3) ∧ 
  (circle_eqn (1 : ℝ) 0) ∧ 
  (circle_eqn (3 : ℝ) 0) ∧ 
  (circle_eqn 0 (3 : ℝ)) :=
by
  sorry

-- Prove the distance |AB|
theorem distance_AB : 
  ∃ (A B : ℝ × ℝ), 
    line_eqn A.1 A.2 ∧ 
    line_eqn B.1 B.2 ∧ 
    circle_eqn A.1 A.2 ∧ 
    circle_eqn B.1 B.2 ∧ 
    |dist A B| = (6 * sqrt 5) / 5 :=
by
  sorry

end circle_through_intersections_distance_AB_l314_314484


namespace angle_measure_triple_complement_l314_314234

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314234


namespace circumscribed_sphere_volume_l314_314711

-- Given conditions and required definitions
def cube_surface_area : ℝ := 24
def edge_length (a : ℝ) := 6 * a^2 = cube_surface_area
def space_diagonal (a : ℝ) := a * Real.sqrt 3
def radius (a : ℝ) := (space_diagonal a) / 2
def sphere_volume (r : ℝ) := (4 / 3) * Real.pi * r^3

theorem circumscribed_sphere_volume
  (a : ℝ)
  (h_edge_length : edge_length a)
  (h_space_diagonal : space_diagonal a = a * Real.sqrt 3)
  : sphere_volume (radius a) = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end circumscribed_sphere_volume_l314_314711


namespace part1_solution_set_part2_range_l314_314983

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314983


namespace rowing_students_l314_314778

theorem rowing_students (X Y : ℕ) (N : ℕ) :
  (17 * X + 6 = N) →
  (10 * Y + 2 = N) →
  100 < N →
  N < 200 →
  5 ≤ X ∧ X ≤ 11 →
  10 ≤ Y ∧ Y ≤ 19 →
  N = 142 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end rowing_students_l314_314778


namespace ordering_of_variables_l314_314481

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem ordering_of_variables 
  (a b c : ℝ)
  (ha : a - 2 = Real.log (a / 2))
  (hb : b - 3 = Real.log (b / 3))
  (hc : c - 3 = Real.log (c / 2))
  (ha_pos : 0 < a) (ha_lt_one : a < 1)
  (hb_pos : 0 < b) (hb_lt_one : b < 1)
  (hc_pos : 0 < c) (hc_lt_one : c < 1) :
  c < b ∧ b < a :=
sorry

end ordering_of_variables_l314_314481


namespace find_k_l314_314475

noncomputable def g (x : ℝ) : ℝ := cot (x / 3) - cot x

theorem find_k (x : ℝ) (hx : sin (x/3) ≠ 0 ∧ sin x ≠ 0) :
  g(x) = (sin ((2/3)*x) / (sin (x/3) * sin x)) :=
by
  sorry

end find_k_l314_314475


namespace lcm_24_36_45_l314_314339

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l314_314339


namespace number_of_true_statements_is_2_l314_314619

def reciprocal (n : ℕ) := (1 : ℚ) / n

def statement_i := (reciprocal 4 + reciprocal 8 = reciprocal 12)
def statement_ii := (reciprocal 8 - reciprocal 3 = reciprocal 5)
def statement_iii := (reciprocal 3 * reciprocal 9 = reciprocal 27)
def statement_iv := (reciprocal 15 / reciprocal 3 = reciprocal 5)

def true_statements_count :=
  [statement_i, statement_ii, statement_iii, statement_iv].count (λ s => s = true) = 2

theorem number_of_true_statements_is_2 : true_statements_count :=
by
  sorry

end number_of_true_statements_is_2_l314_314619


namespace find_ED_l314_314178

def Parallelogram (A B C D : Type) : Prop := sorry

variables (A B C D E F G : Type) [Parallelogram A B C D]

variables (FG FE BE ED : ℝ)

-- Given conditions
axiom FG_FE_ratio : FG / FE = 7
axiom BE_value : BE = 8

-- Proposition to prove
theorem find_ED (h : E ∈ Segment BD) (h1 : F ∈ Segment CD) (h2 : G ∈ (Line A ∩ Line B ∩ Line C))
: ED = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_ED_l314_314178


namespace angle_measure_l314_314264

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314264


namespace exists_diametrically_opposite_non_square_l314_314159

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem exists_diametrically_opposite_non_square :
  ∀ (vertices : Fin 2000 → ℕ),
  (∀ (A B C D : ℕ), 1 ≤ A ∧ A < B ∧ B < C ∧ C < D ∧ D ≤ 2000 → ¬intersect(vertices A, vertices B, vertices C, vertices D)) →
  ∃ i : Fin 2000, is_perfect_square (vertices i) ∧ ¬is_perfect_square (vertices ((i : ℕ + 1000) % 2000)) :=
by
  sorry

end exists_diametrically_opposite_non_square_l314_314159


namespace figures_obtained_by_intersecting_cube_l314_314737

-- Definitions for the problem
def is_plane_figure_obtained_by_intersecting_cube (figure : Type) : Prop :=
  figure = "EquilateralTriangle" ∨ figure = "Trapezoid" ∨ figure = "Rectangle"

-- The statement we need to prove
theorem figures_obtained_by_intersecting_cube :
  ∀ (figure : String),
    figure = "EquilateralTriangle" ∨ figure = "Trapezoid" ∨ figure = "Rectangle" ↔
    is_plane_figure_obtained_by_intersecting_cube figure :=
by
  sorry

end figures_obtained_by_intersecting_cube_l314_314737


namespace number_of_terms_in_simplified_expression_l314_314617

theorem number_of_terms_in_simplified_expression (x y z w : ℕ) :
  let count_even (n : ℕ) := 1 + n / 2 in
  let count_combinations (a : ℕ) := (2024 - a).choose 2 in
  ∑ a in finset.range 1013, count_combinations (2 * a) = "Number of terms in the simplified expression".
sorry

end number_of_terms_in_simplified_expression_l314_314617


namespace men_per_table_l314_314800

theorem men_per_table (total_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) (total_women : ℕ)
    (h1 : total_tables = 9)
    (h2 : women_per_table = 7)
    (h3 : total_customers = 90)
    (h4 : total_women = women_per_table * total_tables)
    (h5 : total_women + total_men = total_customers) :
  total_men / total_tables = 3 :=
by
  have total_women := 7 * 9
  have total_men := 90 - total_women
  exact sorry

end men_per_table_l314_314800


namespace machine_working_days_l314_314177

variable {V a b c x y z : ℝ} 

noncomputable def machine_individual_times_condition (a b c : ℝ) : Prop :=
  ∀ (x y z : ℝ), (x = a + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (y = b + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (z = (-(c * (a + b)) + c * Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (c > 1)

theorem machine_working_days (h1 : x = (z / c) + a) (h2 : y = (z / c) + b) (h3 : z = c * (z / c)) :
  machine_individual_times_condition a b c :=
by
  sorry

end machine_working_days_l314_314177


namespace angle_measure_l314_314266

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314266


namespace number_of_black_squares_in_eaten_area_l314_314137

def is_black_square (row col : ℕ) : Bool :=
  (row + col) % 2 = 1

theorem number_of_black_squares_in_eaten_area (eaten_area : list (ℕ × ℕ)) :
  eaten_area = list.range (8*8) / 2 := sorry

end number_of_black_squares_in_eaten_area_l314_314137


namespace set_of_all_points_form_ellipse_l314_314105

variables {A B P : Point}
variable {d : ℝ}

-- Definition of distance between points
def distance (X Y : Point) : ℝ := ...

-- Condition: The sum of distances
def sum_distances_to_foci (P A B : Point) (d : ℝ) : Prop :=
  distance P A + distance P B = 2 * distance A B

-- Question: Prove the set of all points meeting the condition form an ellipse
theorem set_of_all_points_form_ellipse :
  ∀ (A B : Point) (d : ℝ), (∃ P, sum_distances_to_foci P A B d ↔ 
    ∃ e, is_ellipse_with_major_minor e A B (2 * d) (sqrt 3 * d / 2)) :=
by
  sorry

end set_of_all_points_form_ellipse_l314_314105


namespace baseball_glove_price_l314_314018

noncomputable def original_price_glove : ℝ := 42.50

theorem baseball_glove_price (cards bat glove_discounted cleats total : ℝ) 
  (h1 : cards = 25) 
  (h2 : bat = 10) 
  (h3 : cleats = 2 * 10)
  (h4 : total = 79) 
  (h5 : glove_discounted = total - (cards + bat + cleats)) 
  (h6 : glove_discounted = 0.80 * original_price_glove) : 
  original_price_glove = 42.50 := by 
  sorry

end baseball_glove_price_l314_314018


namespace angle_is_67_l314_314214

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314214


namespace distance_IP_l314_314383

-- Definitions of given conditions
def vertex_I := sorry -- Point I in 3D space.
def cone_base_radius : ℝ := 1
def cone_slant_height : ℝ := 4

-- Points on the cone
def point_A := sorry -- Point A on the circumference of the base.
def IR_length : ℝ := 3
def point_R := sorry -- Point R on the line segment IA.

-- Definition of shortest path and point P
def shortest_path_Watcostarting_at_R_ending_at_A := sorry -- Define the shortest path.
def point_P := sorry -- Point P on this path closest to I.

-- The goal is to prove the distance IP equals 12/5
theorem distance_IP : dist_I_P = 12 / 5 :=
by sorry

end distance_IP_l314_314383


namespace triangles_congruent_l314_314842

variables {Point : Type} [EuclideanGeometry Point]

-- Define the points A, B, C, D, and O
variables (A B C D O : Point)

-- Define the segments AB and CD and their lengths
variables (AB CD : LineSegment Point)
variables (h_AB : AB = ⟨A, B⟩)
variables (h_CD : CD = ⟨C, D⟩)

-- Given conditions
variables (h_eq_segs : length AB = length CD)
variables (h_inter : (∃ P : Point, P ∈ AB ∧ P ∈ CD))
variables (h_AO_eq_OD : distance A O = distance D O)

-- Theorem statement: Prove that Δ ABC ≅ Δ DCB
theorem triangles_congruent :
  (tri_eq (triangle (A : Point) (B : Point) (C : Point)) 
          (triangle (D : Point) (C : Point) (B : Point))) :=
sorry

end triangles_congruent_l314_314842


namespace angle_measure_triple_complement_l314_314304

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314304


namespace equal_circles_common_point_l314_314375

theorem equal_circles_common_point (n : ℕ) (r : ℝ) 
  (centers : Fin n → ℝ × ℝ)
  (h : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k →
    ∃ (p : ℝ × ℝ),
      dist p (centers i) = r ∧
      dist p (centers j) = r ∧
      dist p (centers k) = r) :
  ∃ O : ℝ × ℝ, ∀ i : Fin n, dist O (centers i) = r := sorry

end equal_circles_common_point_l314_314375


namespace f_neg1_f_2_l314_314103

def f (x : ℝ) : ℝ :=
if x < 0 then 3 * x + 7 else 6 - 3 * x

theorem f_neg1 : f (-1) = 4 :=
by
  -- The required proof will go here.
  sorry

theorem f_2 : f 2 = 0 :=
by
  -- The required proof will go here.
  sorry

end f_neg1_f_2_l314_314103


namespace part1_solution_set_part2_range_a_l314_314947

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314947


namespace angle_measure_triple_complement_l314_314220

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314220


namespace net_percent_change_l314_314399

noncomputable def scale_factor (percent: ℝ) : ℝ := 1 + percent / 100

theorem net_percent_change (i j k : ℝ) :
  let sf_0_1 := scale_factor i  -- Scale factor from t=0 to t=1
  let sf_1_2 := scale_factor j  -- Scale factor from t=1 to t=2
  let sf_2_3 := scale_factor (-k)  -- Scale factor from t=2 to t=3
  let overall_scale_factor := sf_0_1 * sf_1_2 * sf_2_3 in
  overall_scale_factor - 1 = (i + j - k + (i * j - i * k - j * k - (i * j * k / 100)) / 100) / 100 :=
sorry

end net_percent_change_l314_314399


namespace crocus_bulb_cost_l314_314354

theorem crocus_bulb_cost 
  (space_bulbs : ℕ)
  (crocus_bulbs : ℕ)
  (cost_daffodil_bulb : ℝ)
  (budget : ℝ)
  (purchased_crocus_bulbs : ℕ)
  (total_cost : ℝ)
  (c : ℝ)
  (h_space : space_bulbs = 55)
  (h_cost_daffodil : cost_daffodil_bulb = 0.65)
  (h_budget : budget = 29.15)
  (h_purchased_crocus : purchased_crocus_bulbs = 22)
  (h_total_cost_eq : total_cost = (33:ℕ) * cost_daffodil_bulb)
  (h_eqn : (purchased_crocus_bulbs : ℝ) * c + total_cost = budget) :
  c = 0.35 :=
by 
  sorry

end crocus_bulb_cost_l314_314354


namespace problem1_problem2_l314_314380

def sample_space : set (ℕ × ℕ × ℕ) :=
  {(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3),
   (1, 3, 1), (1, 3, 2), (1, 3, 3), (2, 1, 1), (2, 1, 2), (2, 1, 3),
   (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 3, 1), (2, 3, 2), (2, 3, 3),
   (3, 1, 1), (3, 1, 2), (3, 1, 3), (3, 2, 1), (3, 2, 2), (3, 2, 3),
   (3, 3, 1), (3, 3, 2), (3, 3, 3)}

def event_A : set (ℕ × ℕ × ℕ) :=
  {(1, 1, 2), (1, 2, 3), (2, 1, 3)}

def event_B_complement : set (ℕ × ℕ × ℕ) :=
  {(1, 1, 1), (2, 2, 2), (3, 3, 3)}

theorem problem1 :
  (finset.card event_A.to_finset : ℝ) / (finset.card sample_space.to_finset : ℝ) = 1 / 9 :=
by sorry

theorem problem2 :
  (1 - (finset.card event_B_complement.to_finset : ℝ) / (finset.card sample_space.to_finset : ℝ)) = 8 / 9 :=
by sorry

end problem1_problem2_l314_314380


namespace part1_part2_l314_314936

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314936


namespace angle_measure_triple_complement_l314_314236

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314236


namespace find_f3_l314_314873

theorem find_f3 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f3_l314_314873


namespace ap_geq_ai_l314_314096

theorem ap_geq_ai (I P : Point) (A B C : Point) (α β γ : Angle) [Incenter I A B C] :
  (∠ P B A + ∠ P C A = ∠ P B C + ∠ P C B) → (AP ≥ AI) :=
by
  sorry

end ap_geq_ai_l314_314096


namespace angle_triple_complement_l314_314196

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314196


namespace inscribed_sphere_volume_ratio_eq_l314_314706

noncomputable def inscribed_sphere_volume_ratio : ℝ :=
  let a := 1 -- arbitrary positive edge length
  let r_ins := (1 / 2) * a
  let r_circ := (sqrt 3 / 2) * a
  (4/3 * π * r_ins^3) / (4/3 * π * r_circ^3)

theorem inscribed_sphere_volume_ratio_eq :
  inscribed_sphere_volume_ratio = 1 / (3 * sqrt 3) := sorry

end inscribed_sphere_volume_ratio_eq_l314_314706


namespace exist_two_courtiers_with_same_selection_l314_314690

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l314_314690


namespace cube_surface_area_increase_l314_314738

theorem cube_surface_area_increase (s : ℝ) :
  let original_surface_area := 6 * s^2 in
  let new_edge_length := 1.4 * s in
  let new_surface_area := 6 * (new_edge_length)^2 in
  (new_surface_area - original_surface_area) / original_surface_area * 100 = 96 :=
by
  sorry

end cube_surface_area_increase_l314_314738


namespace sq_neg_sqrt_three_eq_three_l314_314616

theorem sq_neg_sqrt_three_eq_three : (-sqrt 3)^2 = 3 := 
by
  -- Placeholder for the actual proof
  sorry

end sq_neg_sqrt_three_eq_three_l314_314616


namespace angle_triple_complement_l314_314314

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314314


namespace intersection_points_with_major_axis_l314_314412

noncomputable def ellipse_equation : ℝ → ℝ → Prop :=
  λ x y, 25 * x ^ 2 + 36 * y ^ 2 = 900

noncomputable def point_M : ℝ × ℝ := (4.8, 3)

noncomputable def point_N : ℝ × ℝ := (0, 25 / 3)

noncomputable def circle : ℝ → ℝ → ℝ → ℝ → Prop :=
  λ x y q r, x ^ 2 + (y - q) ^ 2 = r ^ 2

theorem intersection_points_with_major_axis :
  ∃ (x : ℝ), (circle x 0 ((263 : ℝ) / 75) ((362 : ℝ) / 75))
    ↔ (x = 3.31 ∨ x = -3.31) :=
sorry

end intersection_points_with_major_axis_l314_314412


namespace max_phi_symmetry_l314_314511

theorem max_phi_symmetry (φ : ℝ) (k : ℤ) (h : φ < 0) :
  (∀ x : ℝ, 2 * sin (4 * x + φ) = 2 * sin (4 * (π / 24 - x) + φ)) → 
  φ ≤ -2 * π / 3 :=
by
  sorry

end max_phi_symmetry_l314_314511


namespace four_correct_prob_l314_314174

open Finset

-- Define the set of people and letters
def people : Finset ℕ := Finset.range 7 -- Set of 7 people
def letters : Finset ℕ := Finset.range 7 -- Set of 7 letters

-- Define the derangement function
noncomputable def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

-- Define the combination function
noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Example of the binomial coefficient function (choose)
#eval combination 7 4  -- Should be 35

-- Example of derangement
#eval derangement 3  -- Should be 2

-- Calculate probability
noncomputable def probability_four_right : ℚ :=
  let favorable_outcomes := combination 7 4 * derangement 3 in
  favorable_outcomes / Nat.factorial 7

-- Expected result
def expected_probability : ℚ := 1 / 72

-- Lean statement: 
theorem four_correct_prob :
  probability_four_right = expected_probability :=
by
  sorry

end four_correct_prob_l314_314174


namespace rectangle_breadth_is_11_l314_314700

noncomputable def breadth_of_rectangle (length : ℝ) (area : ℝ) : ℝ :=
  area / length

theorem rectangle_breadth_is_11 (radius : ℝ) (square_area : ℝ) (rectangle_area : ℝ) (h1 : square_area = 16) (h2 : radius = real.sqrt square_area) (h3 : length = 5 * radius) (h4 : rectangle_area = 220) :
  breadth_of_rectangle length rectangle_area = 11 :=
begin
  sorry
end

end rectangle_breadth_is_11_l314_314700


namespace problem1_problem2_l314_314592

noncomputable def interval1 (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}
noncomputable def interval2 : Set ℝ := {x | x < -1 ∨ x > 3}

theorem problem1 (a : ℝ) : (interval1 a ∩ interval2 = interval1 a) ↔ a ∈ {x | x ≤ -2} ∪ {x | 1 ≤ x} := by sorry

theorem problem2 (a : ℝ) : (interval1 a ∩ interval2 ≠ ∅) ↔ a < -1 / 2 := by sorry

end problem1_problem2_l314_314592


namespace matches_played_by_W_l314_314057

theorem matches_played_by_W (P Q R S T W : Type)
  (P_has_played_1 : ∃ p1_opponent, p1_opponent ≠ P ∧ p1_opponent ∈ {Q, R, S, T, W}) 
  (Q_has_played_2 : ∃ q1_opponent q2_opponent, q1_opponent ≠ Q ∧ q2_opponent ≠ Q ∧ q1_opponent ≠ q2_opponent ∧ q1_opponent ∈ {P, R, S, T, W} ∧ q2_opponent ∈ {P, R, S, T, W})
  (R_has_played_3 : ∃ r1_opponent r2_opponent r3_opponent, r1_opponent ≠ R ∧ r2_opponent ≠ R ∧ r3_opponent ≠ R ∧ r1_opponent ≠ r2_opponent ∧ r1_opponent ≠ r3_opponent ∧ r2_opponent ≠ r3_opponent ∧ r1_opponent ∈ {P, Q, S, T, W} ∧ r2_opponent ∈ {P, Q, S, T, W} ∧ r3_opponent ∈ {P, Q, S, T, W})
  (S_has_played_4 : ∃ s1_opponent s2_opponent s3_opponent s4_opponent, s1_opponent ≠ S ∧ s2_opponent ≠ S ∧ s3_opponent ≠ S ∧ s4_opponent ≠ S ∧ s1_opponent ≠ s2_opponent ∧ s1_opponent ≠ s3_opponent ∧ s1_opponent ≠ s4_opponent ∧ s2_opponent ≠ s3_opponent ∧ s2_opponent ≠ s4_opponent ∧ s3_opponent ≠ s4_opponent ∧ s1_opponent ∈ {P, Q, R, T, W} ∧ s2_opponent ∈ {P, Q, R, T, W} ∧ s3_opponent ∈ {P, Q, R, T, W} ∧ s4_opponent ∈ {P, Q, R, T, W})
  (T_has_played_5 : ∃ t1_opponent t2_opponent t3_opponent t4_opponent t5_opponent, t1_opponent ≠ T ∧ t2_opponent ≠ T ∧ t3_opponent ≠ T ∧ t4_opponent ≠ T ∧ t5_opponent ≠ T ∧ t1_opponent ≠ t2_opponent ∧ t1_opponent ≠ t3_opponent ∧ t1_opponent ≠ t4_opponent ∧ t1_opponent ≠ t5_opponent ∧ t2_opponent ≠ t3_opponent ∧ t2_opponent ≠ t4_opponent ∧ t2_opponent ≠ t5_opponent ∧ t3_opponent ≠ t4_opponent ∧ t3_opponent ≠ t5_opponent ∧ t4_opponent ≠ t5_opponent ∧ t1_opponent ∈ {P, Q, R, S, W} ∧ t2_opponent ∈ {P, Q, R, S, W} ∧ t3_opponent ∈ {P, Q, R, S, W} ∧ t4_opponent ∈ {P, Q, R, S, W} ∧ t5_opponent ∈ {P, Q, R, S, W}) : 
  ∃ w : W, w.played_matches = 3 := sorry

end matches_played_by_W_l314_314057


namespace max_t_invariant_interval_l314_314486

def f (x : ℝ) (t : ℝ) : ℝ := |2^x - t|
def F (x : ℝ) (t : ℝ) : ℝ := |2^(-x) - t|

theorem max_t_invariant_interval :
  ∀ t, (∀ x ∈ Icc (1:ℝ) 2, (2^x - t) * (2^(-x) - t) ≤ 0) → t ≤ 2 :=
by
  sorry

end max_t_invariant_interval_l314_314486


namespace urn_probability_l314_314806

theorem urn_probability :
  let urn := (2 : ℕ, 1 : ℕ) in
  let box := (∞ : ℕ, ∞ : ℕ) in
  let operation (urn : ℕ × ℕ) : ℕ × ℕ :=
    let draws := [urn.1, urn.2];
    let i := draws.random_element;
    if i == urn.1 then (urn.1 + 1, urn.2)
    else (urn.1, urn.2 + 1) in
  let final_urn := iterate operation 6 urn in
  final_urn = (4, 4) → (20 * (2/3 * 3/4 * 4/5 * 1/6 * 2/7 * 3/8)) = 1/7 :=
by sorry

end urn_probability_l314_314806


namespace tangent_segment_length_l314_314145

-- Defining the conditions as per the problem statement
structure TriangularPyramid where
  A B C D : Point3D
  ∠A_eq_pi_div_2 : angle A B C = π / 2
  ∠C_eq_pi_div_6 : angle B C A = π / 6
  BC_eq_2_sqrt_2 : dist B C = 2 * sqrt (2)
  AD_eq_BD_CD : dist A D = dist B D ∧ dist A D = dist C D
  sphere_radius_1 : ∃ SphereRadius: ℝ, SphereRadius = 1

-- Proving the length of the tangent segment from A to the sphere
theorem tangent_segment_length (pyr : TriangularPyramid) : 
  ∃ L : ℝ, L = sqrt (3) - 1 := 
sorry

end tangent_segment_length_l314_314145


namespace median_of_data_set_is_5_l314_314489

variable (x : ℝ)
variable (data : List ℝ := [-3, 5, 7, x, 11])

def isMode (n : ℝ) (data : List ℝ) : Prop :=
  n ∈ data ∧ (∀ m ∈ data, (data.count n ≥ data.count m))

def median (l : List ℝ) : ℝ := 
  let sorted := l.qsort (≤)
  sorted.get (sorted.length / 2)

theorem median_of_data_set_is_5
  (hm : isMode 5 data) :
  median data = 5 :=
sorry

end median_of_data_set_is_5_l314_314489


namespace range_of_m_l314_314494

noncomputable def p (m : ℝ) : Prop := ∀ x : ℝ, -m * x ^ 2 + 2 * x - m > 0
noncomputable def q (m : ℝ) : Prop := ∀ x > 0, (4 / x + x - m + 1) > 2

theorem range_of_m : 
  (∃ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m)) → (∃ (m : ℝ), -1 ≤ m ∧ m < 3) :=
by
  intros h
  sorry

end range_of_m_l314_314494


namespace range_of_a_l314_314038

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → |x| < a) → 1 ≤ a :=
by 
  sorry

end range_of_a_l314_314038


namespace find_m_l314_314519

-- Definition of vector
def vector (α : Type*) := α × α

-- Two vectors are collinear and have the same direction
def collinear_and_same_direction (a b : vector ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (k * b.1, k * b.2)

-- The vectors a and b
def a (m : ℝ) : vector ℝ := (m, 1)
def b (m : ℝ) : vector ℝ := (4, m)

-- The theorem we want to prove
theorem find_m (m : ℝ) (h1 : collinear_and_same_direction (a m) (b m)) : m = 2 :=
  sorry

end find_m_l314_314519


namespace triple_complement_angle_l314_314276

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314276


namespace zero_point_six_one_eight_method_l314_314372

theorem zero_point_six_one_eight_method (a b : ℝ) (h : a = 2 ∧ b = 4) : 
  ∃ x₁ x₂, x₁ = a + 0.618 * (b - a) ∧ x₂ = a + b - x₁ ∧ (x₁ = 3.236 ∨ x₂ = 2.764) := by
  sorry

end zero_point_six_one_eight_method_l314_314372


namespace largest_unique_digit_sum_eighteen_l314_314327

theorem largest_unique_digit_sum_eighteen :
  ∃ n : ℕ, (∃ ds : List ℕ, ds.Nodup ∧ ds.Sum = 18 ∧ ds.sorted (≥) ∧ ds.join_digits = n) ∧ n = 852410 := sorry

end largest_unique_digit_sum_eighteen_l314_314327


namespace number_of_two_order_partitions_l314_314862

def S (n : ℕ) := Finset (Fin n)

def two_order_partitions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | _ => 3^(n - 1) + 1 / 2
  end

theorem number_of_two_order_partitions (n : ℕ) :
  ∃ (F : ℕ → ℕ), (F 0 = 1) ∧ (F 1 = 2) ∧ (∀ n, F (n+1) = 2* (3^n + 1)) :=
  by 
    exists two_order_partitions
    split; sorry

end number_of_two_order_partitions_l314_314862


namespace angle_triple_complement_l314_314260

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314260


namespace angle_measure_triple_complement_l314_314216

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314216


namespace num_real_solutions_eq_one_l314_314833

theorem num_real_solutions_eq_one (x : ℝ) :
  ∃ (unique_x : ℝ), (2 ^ (7 * unique_x + 2)) * (4 ^ (2 * unique_x + 5)) = 8 ^ (5 * unique_x + 3) ∧ 
  (∀ z : ℝ, (2 ^ (7 * z + 2)) * (4 ^ (2 * z + 5)) = 8 ^ (5 * z + 3) → z = unique_x) :=
sorry

end num_real_solutions_eq_one_l314_314833


namespace courtiers_selection_l314_314666

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l314_314666


namespace judy_total_spending_l314_314844

-- Definition of the items and their original prices
def price_of_carrot : ℝ := 1.50
def price_of_milk : ℝ := 3.50
def price_of_pineapple : ℝ := 5.00
def price_of_flour : ℝ := 6.00
def price_of_ice_cream : ℝ := 8.00

-- Quantities
def num_carrots : ℕ := 6
def num_milk : ℕ := 4
def num_pineapples : ℕ := 3
def num_flour : ℕ := 3
def num_ice_cream : ℕ := 1

-- Discounts
def discount_pineapple : ℝ := 0.25
def discount_flour : ℝ := 0.10

-- Coupon condition
def coupon_threshold : ℝ := 50
def coupon_amount : ℝ := 10

-- Discounted prices
def discounted_price_of_pineapple : ℝ := price_of_pineapple * (1 - discount_pineapple)
def discounted_price_of_flour : ℝ := price_of_flour * (1 - discount_flour)

-- Total cost calculation before coupon
def total_cost_before_coupon : ℝ :=
  num_carrots * price_of_carrot +
  num_milk * price_of_milk +
  num_pineapples * discounted_price_of_pineapple +
  num_flour * discounted_price_of_flour +
  num_ice_cream * price_of_ice_cream

-- Total cost after applying coupon if the condition is met
def total_cost_after_coupon (total_cost : ℝ) : ℝ :=
  if total_cost >= coupon_threshold then total_cost - coupon_amount else total_cost

-- Statement of the proof problem
theorem judy_total_spending :
  total_cost_after_coupon total_cost_before_coupon = 48.45 := by sorry

end judy_total_spending_l314_314844


namespace sampling_methods_match_l314_314122

inductive SamplingMethod
| simple_random
| stratified
| systematic

open SamplingMethod

def commonly_used_sampling_methods : List SamplingMethod := 
  [simple_random, stratified, systematic]

def option_C : List SamplingMethod := 
  [simple_random, stratified, systematic]

theorem sampling_methods_match : commonly_used_sampling_methods = option_C := by
  sorry

end sampling_methods_match_l314_314122


namespace combined_leftover_value_l314_314073

def quarters_james : Nat := 50
def dimes_james : Nat := 80
def quarters_rebecca : Nat := 170
def dimes_rebecca : Nat := 340
def roll_quarters : Nat := 40
def roll_dimes : Nat := 50

theorem combined_leftover_value :
  let total_quarters := quarters_james + quarters_rebecca
  let total_dimes := dimes_james + dimes_rebecca
  let leftover_quarters := total_quarters % roll_quarters
  let leftover_dimes := total_dimes % roll_dimes
  let value_quarters := leftover_quarters * 0.25
  let value_dimes := leftover_dimes * 0.10
  value_quarters + value_dimes = 7 :=
by
  sorry

end combined_leftover_value_l314_314073


namespace count_four_digit_numbers_with_thousands_digit_five_l314_314525

theorem count_four_digit_numbers_with_thousands_digit_five :
  ∃ n : ℕ, n = 1000 ∧ ∀ x : ℕ, 5000 ≤ x ∧ x ≤ 5999 ↔ (x - 4999) ∈ (finset.range 1000 + 1) :=
by
  sorry

end count_four_digit_numbers_with_thousands_digit_five_l314_314525


namespace min_value_of_g_least_value_of_g_l314_314464

variable {a b c d : ℝ}
variable (ha : a > 0) (hd : d > 0)

def g (x : ℝ) := a * x ^ 2 + b * x + c + d * Real.sin x

theorem min_value_of_g : ∃ x, ∀ y, g y ≥ g x :=
begin
  use -b / (2 * a),
  sorry
end

theorem least_value_of_g : ∀ x, g x ≥ (-(b ^ 2) / (4 * a) + c - d) :=
by
  have h_min : ∃ x, ∀ y, g y ≥ g x, from min_value_of_g ha hd,
  cases h_min with x hx,
  specialize hx x,
  sorry

end min_value_of_g_least_value_of_g_l314_314464


namespace simplify_trig_identity_l314_314124

theorem simplify_trig_identity (α : ℝ) : (sin α + cos α) ^ 2 = 1 + sin (2 * α) :=
by
  sorry

end simplify_trig_identity_l314_314124


namespace angle_triple_complement_l314_314198

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314198


namespace percentage_of_Indian_women_l314_314054

-- Definitions of conditions
def total_people := 700 + 500 + 800
def indian_men := (20 / 100) * 700
def indian_children := (10 / 100) * 800
def total_indian_people := (21 / 100) * total_people
def indian_women := total_indian_people - indian_men - indian_children

-- Statement of the theorem
theorem percentage_of_Indian_women : 
  (indian_women / 500) * 100 = 40 :=
by
  sorry

end percentage_of_Indian_women_l314_314054


namespace congruent_triangles_C_D_l314_314594

theorem congruent_triangles_C_D
  (A B C D : Point)
  (h_A h_B h_C h_D : ℝ)
  (l_A l_B l_C l_D : Line)
  (P Q R S : Point)
  (triangle_A triangle_B triangle_C triangle_D : Triangle)
  (h_A_def : Height triangle_A = h_A)
  (h_B_def : Height triangle_B = h_B)
  (h_C_def : Height triangle_C = h_C)
  (h_D_def : Height triangle_D = h_D)
  (similar_triangles : Similar triangle_A triangle_B ∧ Similar triangle_B triangle_C ∧ Similar triangle_C triangle_D ∧ Similar triangle_D triangle_A)
  (congruent_triangles_A_B : Congruent triangle_A triangle_B)
  (l_A_perp : Perpendicular l_A (HeightLine triangle_A))
  (l_B_perp : Perpendicular l_B (HeightLine triangle_B))
  (l_C_perp : Perpendicular l_C (HeightLine triangle_C))
  (l_D_perp : Perpendicular l_D (HeightLine triangle_D))
  (P_def : Intersection l_A l_B = P)
  (Q_def : Intersection l_B l_C = Q)
  (R_def : Intersection l_C l_D = R)
  (S_def : Intersection l_D l_A = S)
  (PQRS_square : Square P Q R S) :
  Congruent triangle_C triangle_D :=
sorry

end congruent_triangles_C_D_l314_314594


namespace andrew_cookies_per_day_l314_314861

/-- Number of days in May --/
def days_in_may : ℤ := 31

/-- Cost per cookie in dollars --/
def cost_per_cookie : ℤ := 15

/-- Total amount spent by Andrew on cookies in dollars --/
def total_amount_spent : ℤ := 1395

/-- Total number of cookies purchased by Andrew --/
def total_cookies : ℤ := total_amount_spent / cost_per_cookie

/-- Number of cookies purchased per day --/
def cookies_per_day : ℤ := total_cookies / days_in_may

theorem andrew_cookies_per_day : cookies_per_day = 3 := by
  sorry

end andrew_cookies_per_day_l314_314861


namespace triple_complement_angle_l314_314287

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314287


namespace limit_of_sum_of_squares_l314_314759

noncomputable def sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range (n+1), a k

theorem limit_of_sum_of_squares (a : ℕ → ℝ) (C : ℝ) (s : ℝ)
  (hC_pos : 0 < C)
  (h_sum_bound : ∀ n, ∑ k in finset.range (n+1), k * |a k| < n * C)
  (h_lim : tendsto (λ n, (∑ k in finset.range (n+1), sn a k) / (n + 1)) at_top (𝓝 s)) :
  tendsto (λ n, (∑ k in finset.range (n+1), (sn a k) ^ 2) / (n + 1)) at_top (𝓝 (s ^ 2)) :=
sorry

end limit_of_sum_of_squares_l314_314759


namespace angle_triple_complement_l314_314193

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314193


namespace exists_list_containing_all_l314_314058

universe u

structure Tournament (Player : Type u) where
  meet_once : ∀ (x y : Player), x ≠ y → ∃ (result : bool), result = true ∨ result = false
  has_no_draws : ∀ (x y : Player), x ≠ y → ∃ (result : bool), (result = true ∧ ¬ result = false) ∨ (¬ result = true ∧ result = false)

variables (Player : Type u) [inhabited Player] [fintype Player] (T : Tournament Player)

noncomputable def won_against (x : Player) : set Player := {y : Player | ∃ (result : bool), result = true}

noncomputable def defeated_by_winners (x : Player) : set Player :=
  {z : Player | ∃ y : Player, y ∈ won_against T x ∧ z ∈ won_against T y}

noncomputable def player_list (x : Player) : set Player :=
  won_against T x ∪ defeated_by_winners T x

theorem exists_list_containing_all : ∃ x : Player, player_list T x = (@set.univ Player) \ {x} :=
sorry

end exists_list_containing_all_l314_314058


namespace inequality_S_l314_314093

noncomputable def f (x : ℝ) : ℝ :=
  (2 * x^2 + 1) / x

def a_seq : ℕ → ℝ 
| 1 => 2
| (n+1) => (f (a_seq n) - a_seq n) / 2

def b_seq (n : ℕ) : ℝ :=
  (1 / 3)^(2^n)

def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a_seq (i+1)

theorem inequality_S (n : ℕ) (h : 0 < n) : S n < n + 1.5 :=
begin
  sorry
end

end inequality_S_l314_314093


namespace minimum_value_y_l314_314155

noncomputable def y (x : ℝ) : ℝ := 2 * x + 1 / x

theorem minimum_value_y : (∃ x : ℝ, x > 0 ∧ y x = 2 * Real.sqrt 2) ∧ 
                         (∀ x : ℝ, x > 0 → y x ≥ 2 * Real.sqrt 2) := 
begin
  sorry,
end

end minimum_value_y_l314_314155


namespace function_decreasing_range_l314_314007

theorem function_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1) ≤ (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1)) ↔ (0 ≤ a ∧ a ≤ 1 / 3) :=
sorry

end function_decreasing_range_l314_314007


namespace roots_polynomial_sum_l314_314586

theorem roots_polynomial_sum (p q r s : ℂ)
  (h_roots : (p, q, r, s) ∈ { (p, q, r, s) | (Polynomial.eval p (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval q (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval r (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval s (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) })
  (h_sum_two_at_a_time : p*q + p*r + p*s + q*r + q*s + r*s = 20)
  (h_product : p*q*r*s = 6) :
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = 10 / 3 := by
  sorry

end roots_polynomial_sum_l314_314586


namespace angle_triple_complement_l314_314297

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314297


namespace seashells_collected_l314_314606

theorem seashells_collected (x y z : ℕ) (hyp : x + y / 2 + z + 5 = 76) : x + y + z = 71 := 
by {
  sorry
}

end seashells_collected_l314_314606


namespace math_city_police_officers_needed_l314_314598

def number_of_streets : Nat := 10
def initial_intersections : Nat := Nat.choose number_of_streets 2
def non_intersections : Nat := 2
def effective_intersections : Nat := initial_intersections - non_intersections

theorem math_city_police_officers_needed :
  effective_intersections = 43 := by
  sorry

end math_city_police_officers_needed_l314_314598


namespace exists_intersecting_rectangle_l314_314365

theorem exists_intersecting_rectangle 
  (rects : Fin 100 → (ℝ × ℝ) × (ℝ × ℝ))
  (H_intersects : ∀ i, (Fin 100).card - 1 ≤ (Fin 100).count (λ j, intersects (rects i) (rects j))) :
  ∃ R : (ℝ × ℝ) × (ℝ × ℝ), ∀ i, intersects R (rects i) := sorry

/-- Helper function definition -/
def intersects ((x1, y1) : ℝ × ℝ) ((x2, y2) : ℝ × ℝ) : Prop :=
  (x1.1 ≤ x2.2) ∧ (x2.1 ≤ x1.2) ∧ (y1.1 ≤ y2.2) ∧ (y2.1 ≤ y1.2)


end exists_intersecting_rectangle_l314_314365


namespace enclosed_area_four_circles_l314_314476

/-- 
Given four circles each of radius 7 cm such that each circle touches two other circles, 
prove that the area of the space enclosed by the four pieces is 196 cm² - 49π cm².
-/
theorem enclosed_area_four_circles :
  let r := 7 in
  let square_side := 2 * r in
  let square_area := square_side * square_side in
  let circle_area := π * r * r in
  square_area - circle_area = 196 - 49 * π :=
by
  let r := 7
  let square_side := 2 * r
  let square_area := square_side * square_side
  let circle_area := π * r * r
  show square_area - circle_area = 196 - 49 * π
  sorry

end enclosed_area_four_circles_l314_314476


namespace system_solutions_three_system_solutions_two_l314_314849

theorem system_solutions_three (a : ℝ) :
  (∃ x y : ℝ, 3 * |y| - 4 * |x| = 6 ∧ x^2 + y^2 - 14 * y + 49 - a^2 = 0) →
  |a| ∈ {5, 9} :=
sorry

theorem system_solutions_two (a : ℝ) :
  (∃ x y : ℝ, 3 * |y| - 4 * |x| = 6 ∧ x^2 + y^2 - 14 * y + 49 - a^2 = 0) →
  |a| ∈ {3} ∪ {b | 5 < b ∧ b < 9} :=
sorry

end system_solutions_three_system_solutions_two_l314_314849


namespace algebraic_expression_from_k_to_k_plus_one_l314_314189

/-- Theorem to prove the induction step for the given problem. -/
theorem algebraic_expression_from_k_to_k_plus_one (k : ℕ) :
  ((k+2)*(k+3)*...*(2*k)*(2*k+1)*(2*k+2)) / ((k+1)*(k+2)*...*(2*k)) = 2*(2*k+1) :=
sorry

end algebraic_expression_from_k_to_k_plus_one_l314_314189


namespace courtiers_dog_selection_l314_314656

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l314_314656


namespace consecutive_integers_product_divisible_l314_314758

theorem consecutive_integers_product_divisible (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  ∀ n : ℕ, ∃ (x y : ℕ), (n ≤ x) ∧ (x < n + b) ∧ (n ≤ y) ∧ (y < n + b) ∧ (x ≠ y) ∧ (a * b ∣ x * y) :=
by
  sorry

end consecutive_integers_product_divisible_l314_314758


namespace part1_part2_l314_314999

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l314_314999


namespace sum_of_fractions_l314_314826

def S_1 : List ℚ := List.range' 1 10 |>.map (λ n => n / 10)
def S_2 : List ℚ := List.replicate 4 (20 / 10)

def total_sum : ℚ := S_1.sum + S_2.sum

theorem sum_of_fractions : total_sum = 12.5 := by
  sorry

end sum_of_fractions_l314_314826


namespace angle_triple_complement_l314_314240

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314240


namespace salary_increase_l314_314361

theorem salary_increase (S : ℝ) (h : S > 0) : 
    let reduced_salary := S * 0.72 in
    let percentage_increased_salary := reduced_salary * (1 + 38.89 / 100) in
    percentage_increased_salary = S :=
by
  sorry

end salary_increase_l314_314361


namespace problem_statement_l314_314529

-- Definitions corresponding to given conditions
def statement_1 := ¬(0 ∈ (∅ : Set Nat))
def statement_2 := ¬({1} ⊆ {1, 2, 3})
def statement_3 := ¬({1} ∈ ({1, 2, 3}))
def statement_4 := (∅ : Set Nat) ⊆ {0}

-- Proposition that exactly one of the statements is true
theorem problem_statement : ({statement_1, statement_2, statement_3, statement_4} = {false, false, false, true}) := 
by 
  sorry

end problem_statement_l314_314529


namespace min_groups_with_conditions_l314_314714

theorem min_groups_with_conditions (n a b m : ℕ) (h_n : n = 8) (h_a : a = 4) (h_b : b = 1) :
  m ≥ 2 :=
sorry

end min_groups_with_conditions_l314_314714


namespace subset_infinite_multiples_l314_314164

theorem subset_infinite_multiples (k : ℕ) (partition : fin k → set ℕ) (h : ∀ n ∈ set.univ, ∃ i, n ∈ partition i) :
  ∃ S (i : fin k), (∀ n : ℕ, S = partition i → ∃ m : ℕ, m > 0 ∧ set_of (λ x : ℕ, x = m * n) ⊆ S) :=
sorry

end subset_infinite_multiples_l314_314164


namespace function_increasing_interval_l314_314008

def has_extreme_value_at_2 (a : ℝ) : Prop :=
  let y := λ x, 2 * x^3 + a * x^2 + 36 * x - 24
  let y' := λ x, 6 * x^2 + 2 * a * x + 36
  y' 2 = 0

def is_increasing_interval {a : ℝ} (f : ℝ → ℝ) (interval : set ℝ) : Prop :=
  ∀ x y, x ∈ interval → y ∈ interval → x < y → f x < f y

theorem function_increasing_interval :
  ∀ (a : ℝ),
  has_extreme_value_at_2 a →
  a = -15 ∧ 
  is_increasing_interval (λ x, 2 * x^3 + a * x^2 + 36 * x - 24) (set.Ioi 3) :=
by
  intro a
  intro h_extreme
  sorry

end function_increasing_interval_l314_314008


namespace part1_solution_set_part2_range_of_a_l314_314927

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314927


namespace angle_triple_complement_l314_314313

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314313


namespace factorial_div_squared_l314_314435

variable (M : ℕ)

theorem factorial_div_squared :
  (M+1)! / ((M+2)!)^2 = 1 / (M+2)^2 := 
by
  sorry

end factorial_div_squared_l314_314435


namespace mary_initial_stickers_l314_314597

theorem mary_initial_stickers (stickers_remaining : ℕ) 
  (front_page_stickers : ℕ) (other_page_stickers : ℕ) 
  (num_other_pages : ℕ) 
  (h1 : front_page_stickers = 3)
  (h2 : other_page_stickers = 7 * num_other_pages)
  (h3 : num_other_pages = 6)
  (h4 : stickers_remaining = 44) :
  ∃ initial_stickers : ℕ, initial_stickers = front_page_stickers + other_page_stickers + stickers_remaining ∧ initial_stickers = 89 :=
by
  sorry

end mary_initial_stickers_l314_314597


namespace sallys_change_l314_314120

-- Given conditions
def frames_bought : ℕ := 3
def cost_per_frame : ℕ := 3
def payment : ℕ := 20

-- The statement to prove
theorem sallys_change : payment - (frames_bought * cost_per_frame) = 11 := by
  sorry

end sallys_change_l314_314120


namespace max_sum_xy_l314_314461

theorem max_sum_xy (x y : ℤ) (h1 : x^2 + y^2 = 64) (h2 : x ≥ 0) (h3 : y ≥ 0) : x + y ≤ 8 :=
by sorry

end max_sum_xy_l314_314461


namespace angle_triple_complement_l314_314202

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314202


namespace number_of_slices_left_l314_314605

-- Conditions
def total_slices : ℕ := 8
def slices_given_to_joe_and_darcy : ℕ := total_slices / 2
def slices_given_to_carl : ℕ := total_slices / 4

-- Question: How many slices were left?
def slices_left : ℕ := total_slices - (slices_given_to_joe_and_darcy + slices_given_to_carl)

-- Proof statement to demonstrate that slices_left == 2
theorem number_of_slices_left : slices_left = 2 := by
  sorry

end number_of_slices_left_l314_314605


namespace handshakes_correct_l314_314547

-- Define the conditions
def num_of_teams : Nat := 4
def team_sizes : List Nat := [2, 2, 2, 3]

-- Calculate the number of players
def num_of_players : Nat := team_sizes.sum

-- Calculate the total possible handshakes without considering teams
def total_possible_handshakes : Nat := num_of_players * (num_of_players - 1) / 2

-- Calculate internal handshakes for each team
def internal_handshakes (sizes : List Nat) : Nat :=
  sizes.foldl (λ acc size, acc + size * (size - 1) / 2) 0

def total_internal_handshakes : Nat := internal_handshakes team_sizes

-- The total number of inter-team handshakes
noncomputable def total_inter_team_handshakes := total_possible_handshakes - total_internal_handshakes

-- The theorem to prove
theorem handshakes_correct : total_inter_team_handshakes = 30 := by
  sorry

end handshakes_correct_l314_314547


namespace angle_bisector_inequality_l314_314760

variables {A B C D E : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]

variable (triangle_ABC : triangle A B C)

-- Assuming AD and BE are angle bisectors
variable (angle_bisector_AD : is_angle_bisector triangle_ABC A D)
variable (angle_bisector_BE : is_angle_bisector triangle_ABC B E)

-- Given condition AC > BC
variable (AC_gt_BC : ∀ (A C B: Type) [metric_space A] [metric_space B] [metric_space C], dist A C > dist B C)

theorem angle_bisector_inequality 
  (h1 : is_angle_bisector triangle_ABC A D) 
  (h2 : is_angle_bisector triangle_ABC B E) 
  (h3 : ∀ (A C B: Type) [metric_space A] [metric_space B] [metric_space C], dist A C > dist B C) : 
  dist A E > dist D E ∧ dist D E > dist B D :=  
sorry

end angle_bisector_inequality_l314_314760


namespace minimum_real_roots_l314_314755

variable {R : Type} [CommRing R] [IsDomain R] [CharZero R]

theorem minimum_real_roots (P : R[X]) (hodd : degree P % 2 = 1) :
  let n := (roots P).nodup_real in
  (roots (P.comp P)).nodup_real ≥ n :=
sorry

end minimum_real_roots_l314_314755


namespace problem_part_1_problem_part_2_l314_314517

noncomputable def interval_length (a b : ℝ) : ℝ := b - a

def problem_condition_1 (t : ℝ) : Prop :=
  interval_length 2 (Real.log t / Real.log 2) = 3

def problem_condition_2 (t : ℝ) : Prop :=
  (2 : ℝ) < (Real.log t / Real.log 2) ∧ (Real.log t / Real.log 2) ≤ (5 : ℝ)

theorem problem_part_1 (t : ℝ) (h : problem_condition_1 t) : t = 32 := sorry

theorem problem_part_2 (t : ℝ) (h : problem_condition_2 t) : t ∈ Ioc 0 32 := sorry

end problem_part_1_problem_part_2_l314_314517


namespace imaginary_part_of_ai_mul_l314_314025

def pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem imaginary_part_of_ai_mul {a : ℝ} (h : pure_imaginary ((1 - complex.I * a) * complex.I)) : 
  a = 0 :=
sorry

end imaginary_part_of_ai_mul_l314_314025


namespace james_running_increase_l314_314075

theorem james_running_increase (initial_miles_per_week : ℕ) (percent_increase : ℝ) (total_days : ℕ) (days_in_week : ℕ) :
  initial_miles_per_week = 100 →
  percent_increase = 0.2 →
  total_days = 280 →
  days_in_week = 7 →
  ∃ miles_per_week_to_add : ℝ, miles_per_week_to_add = 3 :=
by
  intros
  sorry

end james_running_increase_l314_314075


namespace part1_part2_l314_314929

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314929


namespace minimum_value_expression_l314_314579

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  4 * a ^ 3 + 8 * b ^ 3 + 27 * c ^ 3 + 64 * d ^ 3 + 2 / (a * b * c * d) ≥ 16 * Real.sqrt 3 :=
by
  sorry

end minimum_value_expression_l314_314579


namespace fourth_equation_l314_314112

theorem fourth_equation :
  (5 * 6 * 7 * 8) = (2^4) * 1 * 3 * 5 * 7 :=
by
  sorry

end fourth_equation_l314_314112


namespace dice_probability_l314_314429

theorem dice_probability {p : ℝ} (h : p = 1 / 2) : 
  (5.choose 3 * p^3 * p^2 = 5 / 16) :=
by
  sorry

end dice_probability_l314_314429


namespace three_of_clubs_last_card_l314_314701

theorem three_of_clubs_last_card (pos : ℕ) (h : pos = 26 ∨ pos = 27) 
(deck : list ℕ) (h1 : deck.length = 52) 
(h2 : deck.nth (pos - 1) = some 3) :
  ∃ (f : ℕ → bool), ∀ (choices : list ℕ), f choices.length = tt → deck.nth (pos - choices.length) = some 3 := 
by 
  sorry

end three_of_clubs_last_card_l314_314701


namespace original_flow_rate_l314_314802

theorem original_flow_rate (x : ℝ) (h : 2 = 0.6 * x - 1) : x = 5 :=
by
  sorry

end original_flow_rate_l314_314802


namespace num_pairs_eq_12_l314_314444

theorem num_pairs_eq_12 :
  ∃ (n : ℕ), (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧
    (a + 1/b : ℚ) / (1/a + b : ℚ) = 7 ↔ (7 * b = a)) ∧ n = 12 :=
sorry

end num_pairs_eq_12_l314_314444


namespace angle_triple_complement_l314_314252

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314252


namespace jane_oldest_child_age_l314_314753

noncomputable def currentAgeOldestChildJaneBabysat (current_age_jane : ℕ) (years_since_stopped : ℕ) : ℕ :=
  let age_when_stopped := current_age_jane - years_since_stopped
  let max_age_child_when_stopped := age_when_stopped / 2
  max_age_child_when_stopped + years_since_stopped

theorem jane_oldest_child_age (current_age_jane : ℕ) (years_since_stopped : ℕ) 
  (h_current_age_jane : current_age_jane = 34) (h_years_since_stopped : years_since_stopped = 12) :
  currentAgeOldestChildJaneBabysat current_age_jane years_since_stopped = 23 := by
  unfold currentAgeOldestChildJaneBabysat
  rw [h_current_age_jane, h_years_since_stopped]
  simp
  sorry

end jane_oldest_child_age_l314_314753


namespace unused_types_l314_314807

theorem unused_types (total_resources : ℕ) (used_types : ℕ) (valid_types : ℕ) :
  total_resources = 6 → used_types = 23 → valid_types = 2^total_resources - 1 - used_types → valid_types = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end unused_types_l314_314807


namespace part1_part2_l314_314979

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314979


namespace max_sum_of_digits_in_24_hour_display_l314_314772

theorem max_sum_of_digits_in_24_hour_display : 
  ∃ h m : ℕ, (0 ≤ h ∧ h < 24) ∧ (0 ≤ m ∧ m < 60) ∧ 
  (let digit_sum (n : ℕ) := n / 10 + n % 10 in
  digit_sum h + digit_sum m = 23) := sorry

end max_sum_of_digits_in_24_hour_display_l314_314772


namespace divisible_by_4_digit_count_l314_314843

theorem divisible_by_4_digit_count : 
  (Finset.card (Finset.filter (λ n : ℕ, n%4 = 0) (Finset.range 100))) = 25 :=
by
  sorry

end divisible_by_4_digit_count_l314_314843


namespace equilateral_triangle_l314_314611

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : α + β + γ = π)
  (h8 : a = 2 * Real.sin α)
  (h9 : b = 2 * Real.sin β)
  (h10 : c = 2 * Real.sin γ)
  (h11 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  a = b ∧ b = c :=
sorry

end equilateral_triangle_l314_314611


namespace angle_measure_triple_complement_l314_314221

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314221


namespace simplify_polynomial_l314_314614

theorem simplify_polynomial (p : ℕ) :
  (5 * p^4 + 4 * p^3 - 7 * p^2 + 9 * p - 3) + (-8 * p^4 + 2 * p^3 - p^2 - 3 * p + 4) 
    = -3 * p^4 + 6 * p^3 - 8 * p^2 + 6 * p + 1 :=
by
  sorry

end simplify_polynomial_l314_314614


namespace count_FourDigitNumsWithThousandsDigitFive_is_1000_l314_314528

def count_FourDigitNumsWithThousandsDigitFive : Nat :=
  let minNum := 5000
  let maxNum := 5999
  maxNum - minNum + 1

theorem count_FourDigitNumsWithThousandsDigitFive_is_1000 :
  count_FourDigitNumsWithThousandsDigitFive = 1000 :=
by
  sorry

end count_FourDigitNumsWithThousandsDigitFive_is_1000_l314_314528


namespace courtiers_selection_l314_314671

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l314_314671


namespace termites_ate_black_squares_l314_314134

-- Define a Lean function to check if a given cell is black on a standard chessboard.
def is_black (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Define the theorem that asserts the number of black squares in a 3x8 block is 12.
theorem termites_ate_black_squares : 
  (finset.univ.filter (λ k : ℕ × ℕ, k.1 < 3 ∧ k.2 < 8 ∧ is_black k.1 k.2)).card = 12 :=
by
  sorry -- proof to be provided

end termites_ate_black_squares_l314_314134


namespace focal_length_of_ellipse_l314_314851

def equation_of_ellipse : Prop :=
  ∀ x y : ℝ, y^2 / 2 + x^2 = 1

def semi_major_axis (a : ℝ) : Prop :=
  a = sqrt 2

def semi_minor_axis (b : ℝ) : Prop :=
  b = 1

def semi_focal_distance (c a b : ℝ) : Prop :=
  c = sqrt (a^2 - b^2)

theorem focal_length_of_ellipse : ∃ c : ℝ, semi_focal_distance c (sqrt 2) 1 ∧ 2 * c = 2 :=
by
  use 1
  simp [semi_focal_distance, sqrt, pow_two]
  ring
  sorry


end focal_length_of_ellipse_l314_314851


namespace problem_statement_l314_314004

noncomputable def f (x : ℝ) : ℝ := (2 * x^2) / Real.exp(1) + (Real.exp(1)^2) / x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.exp(1) * Real.log x

theorem problem_statement :
  (∀ x: ℝ, x < 0 → deriv f x < 0) ∧
  (∀ x: ℝ, 0 < x ∧ x < Real.exp(1) / (4 ^ (1 / 3)) → deriv f x < 0) ∧
  (∀ x: ℝ, x > Real.exp(1) / (4 ^ (1 / 3)) → deriv f x > 0) ∧
  (f (Real.exp(1)) = 3 * Real.exp(1)) ∧
  (g (Real.exp(1)) = 3 * Real.exp(1)) ∧
  fun (x0 : ℝ) (H1 : f x0 = g x0) (H2 : deriv f x0 = deriv g x0) => 
    x0 = Real.exp(1) ∧ 
    deriv f (Real.exp(1)) = 3 ∧ 
    deriv g (Real.exp(1)) = 3 ∧ 
    ∀ (x : ℝ), f (Real.exp(1)) + der f (Real.exp(1)) * (x - Real.exp(1)) = 3*x := 
sorry

end problem_statement_l314_314004


namespace number_of_black_squares_in_eaten_area_l314_314136

def is_black_square (row col : ℕ) : Bool :=
  (row + col) % 2 = 1

theorem number_of_black_squares_in_eaten_area (eaten_area : list (ℕ × ℕ)) :
  eaten_area = list.range (8*8) / 2 := sorry

end number_of_black_squares_in_eaten_area_l314_314136


namespace hunting_dogs_theorem_l314_314661

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l314_314661


namespace two_phones_defective_probability_l314_314752

noncomputable def probability_both_defective (total_phones defective_phones : ℕ) : ℝ :=
  (defective_phones / total_phones.to_real) * ((defective_phones - 1) / (total_phones - 1).to_real)

theorem two_phones_defective_probability :
  probability_both_defective 250 67 ≈ 0.071052 :=
by
  sorry

end two_phones_defective_probability_l314_314752


namespace x_varies_inversely_l314_314362

theorem x_varies_inversely (y: ℝ) (x: ℝ): (∃ k: ℝ, (∀ y: ℝ, x = k / y ^ 2) ∧ (1 = k / 3 ^ 2)) → x = 0.5625 :=
by
  sorry

end x_varies_inversely_l314_314362


namespace part1_solution_part2_solution_l314_314962

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314962


namespace calculate_star_l314_314033

def star (A B : ℝ) : ℝ := (A + B) / 2

theorem calculate_star : star (star 5 7) 9 = 7.5 :=
by
  -- Using the definition of star, we calculate (5 * 7) * 9.
  -- 5 * 7 = (5 + 7) / 2 = 6
  -- (5 * 7) * 9 = 6 * 9 = (6 + 9) / 2 = 7.5
  sorry

end calculate_star_l314_314033


namespace divisors_problem_l314_314099

/-
Let \( m = 2^{40}5^{24} \). 
We need to prove that the number of positive integer divisors of \( m^2 \) 
that are less than \( m \), but do not divide \( m \), is \( 959 \).
-/
theorem divisors_problem (h : ∀ m : ℕ, m = 2^40 * 5^24) : 
  let m := 2^40 * 5^24,
      m_sq := m * m,
      num_divisors_m_sq := ∏ x in [(80 + 1), (48 + 1)], x,
      num_divisors_m := ∏ x in [(40 + 1), (24 + 1)], x in
  (num_divisors_m_sq - 1) / 2 - num_divisors_m = 959 :=
by
  sorry

end divisors_problem_l314_314099


namespace locus_of_midpoints_is_annulus_l314_314854

-- Define centers of the circles
variables {O1 O2 : ℝ × ℝ}

-- Define radii of the circles
variables {R r : ℝ}

-- Define circles S_1 and S_2
def circle_S₁ (P : ℝ × ℝ) : Prop := dist P O1 = R
def circle_S₂ (P : ℝ × ℝ) : Prop := dist P O2 = r

-- Non-intersecting condition
variable (h_non_intersecting : ∀ (P1 : ℝ × ℝ), ∀ (P2 : ℝ × ℝ),
  (circle_S₁ P1) → (circle_S₂ P2) → P1 ≠ P2 ∧ dist O1 O2 > R + r)

-- Midpoint definition
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

-- Proof stating the locus of midpoints forms an annulus
theorem locus_of_midpoints_is_annulus :
  ∀ (M : ℝ × ℝ), (∃ (P1 P2 : ℝ × ℝ), 
    circle_S₁ P1 ∧ circle_S₂ P2 ∧ M = midpoint P1 P2) ↔
  dist M O1 ∈ set.Icc ((R - r) / 2) ((R + r) / 2) :=
by sorry

end locus_of_midpoints_is_annulus_l314_314854


namespace volume_pyramid_formula_l314_314859

noncomputable def volume_of_regular_hexagonal_pyramid (b R : ℝ) : ℝ :=
  (b^4 * real.sqrt 3 * (4 * R^2 - b^2)) / (16 * R^3)

theorem volume_pyramid_formula (b R : ℝ) (hb : 0 < b) (hR : 0 < R) :
  volume_of_regular_hexagonal_pyramid b R = 
  (b^4 * real.sqrt 3 * (4 * R^2 - b^2)) / (16 * R^3) := by
  -- proof omitted
  sorry

end volume_pyramid_formula_l314_314859


namespace proposition_equivalence_l314_314348

open Classical

variable {α : Type*}
variables (P : Set α)

theorem proposition_equivalence:
  (∀ a b, a ∈ P → b ∉ P) ↔ (∀ a b, b ∈ P → a ∉ P) :=
by intros a b; sorry

end proposition_equivalence_l314_314348


namespace anthony_pets_left_l314_314421

theorem anthony_pets_left : 
  let original_pets := 16 in
  let lost_pets := 6 in
  let died_pets_fraction := 1/5 in
  let remaining_pets_after_loss := original_pets - lost_pets in
  let died_pets := (remaining_pets_after_loss * died_pets_fraction).toNat in
  let remaining_pets := remaining_pets_after_loss - died_pets in
  remaining_pets = 8 := 
by sorry

end anthony_pets_left_l314_314421


namespace triangle_medial_8_l314_314580

variables (a b c : ℝ)
variables (P Q R : ℝ × ℝ × ℝ)

def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

def is_midpoint (M A B : ℝ × ℝ × ℝ) : Prop :=
  M = midpoint A B

theorem triangle_medial_8
  (h1 : is_midpoint (a, 0, 0) Q R)
  (h2 : is_midpoint (0, b, 0) P R)
  (h3 : is_midpoint (0, 0, c) P Q) :
  (dist Q P)^2 + (dist P R)^2 + (dist R Q)^2 = 8 * (a^2 + b^2 + c^2) :=
sorry

end triangle_medial_8_l314_314580


namespace Betty_will_pay_zero_l314_314186

-- Definitions of the conditions
def Doug_age : ℕ := 40
def Alice_age (D : ℕ) : ℕ := D / 2
def Betty_age (B D A : ℕ) : Prop := B + D + A = 130
def Cost_of_pack_of_nuts (C B : ℕ) : Prop := C = 2 * B
def Decrease_rate : ℕ := 5
def New_cost (C B A : ℕ) : ℕ := max 0 (C - (B - A) * Decrease_rate)
def Total_cost (packs cost_per_pack: ℕ) : ℕ := packs * cost_per_pack

-- The main proposition
theorem Betty_will_pay_zero :
  ∃ B A C, 
    (C = 2 * B) ∧
    (A = Doug_age / 2) ∧
    (B + Doug_age + A = 130) ∧
    (Total_cost 20 (max 0 (C - (B - A) * Decrease_rate)) = 0) :=
by sorry

end Betty_will_pay_zero_l314_314186


namespace distance_to_focus_l314_314536

-- Define the variables and conditions
variables {x y : ℝ}
variable (P : ℝ × ℝ)
variable (O : ℝ × ℝ)
def parabola : Prop := x^2 = 12 * y
def distance_to_origin : Prop := (P.1)^2 + (P.2)^2 = 28

-- Theorem statement
theorem distance_to_focus (h1 : parabola P.1 P.2) (h2 : distance_to_origin P) : 
  ∀ focus : ℝ × ℝ, focus = (0, 3) → dist P focus = 5 := 
begin
  sorry
end

end distance_to_focus_l314_314536


namespace log2_bounds_158489_l314_314170

theorem log2_bounds_158489 :
  (2^16 = 65536) ∧ (2^17 = 131072) ∧ (65536 < 158489 ∧ 158489 < 131072) →
  (16 < Real.log 158489 / Real.log 2 ∧ Real.log 158489 / Real.log 2 < 17) ∧ 16 + 17 = 33 :=
by
  intro h
  have h1 : 2^16 = 65536 := h.1
  have h2 : 2^17 = 131072 := h.2.1
  have h3 : 65536 < 158489 := h.2.2.1
  have h4 : 158489 < 131072 := h.2.2.2
  sorry

end log2_bounds_158489_l314_314170


namespace coal_burn_proof_incorrect_equation_A_l314_314773

def factory_coal_burned (coal_burned_5_days : ℝ) (additional_days : ℕ) : ℝ :=
  let daily_consumption := coal_burned_5_days / 5
  let total_days := 5 + additional_days
  daily_consumption * total_days 

theorem coal_burn_proof (coal_burned_5_days : ℝ) (additional_days : ℕ) :
  coal_burned_5_days = 37.5 → additional_days = 8 →
  factory_coal_burned coal_burned_5_days additional_days = 97.5 :=
by
  intros h1 h2
  rw [h1, h2]
  unfold factory_coal_burned
  norm_num

theorem incorrect_equation_A : ¬ (37.5 * (8 / 5) = 97.5) :=
by
  intro h
  norm_num at h
  exact false_of_ne (by norm_num) h

#check coal_burn_proof
#check incorrect_equation_A

end coal_burn_proof_incorrect_equation_A_l314_314773


namespace part1_solution_set_part2_range_l314_314991

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314991


namespace sum_of_cubes_l314_314846

theorem sum_of_cubes (n : ℕ) : ∑ k in Finset.range (n + 1), k^3 = (n * (n + 1) / 2)^2 :=
by
  sorry

end sum_of_cubes_l314_314846


namespace lcm_24_36_45_l314_314332

noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem lcm_24_36_45 : lcm_three 24 36 45 = 360 := by
  -- Conditions involves proving the prime factorizations
  have h1 : Nat.factors 24 = [2, 2, 2, 3] := by sorry -- Prime factorization of 24
  have h2 : Nat.factors 36 = [2, 2, 3, 3] := by sorry -- Prime factorization of 36
  have h3 : Nat.factors 45 = [3, 3, 5] := by sorry -- Prime factorization of 45

  -- Least common multiple calculation based on the greatest powers of prime factors
  sorry -- This is where the proof would go

end lcm_24_36_45_l314_314332


namespace minimum_value_ineq_l314_314576

def minimum_expression_value (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2

theorem minimum_value_ineq (α β : ℝ) :
  ∃ α β : ℝ, Real.cos α = (10 / 13) ∧ Real.sin α = (12 / 13) ∧ β = (Real.pi / 2) - α ∧
  minimum_expression_value α β = (Real.sqrt 244 - 7)^2 := 
sorry

end minimum_value_ineq_l314_314576


namespace hunting_dogs_theorem_l314_314660

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l314_314660


namespace combined_savings_l314_314612

theorem combined_savings (salary_per_hour : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ)
  (weeks : ℕ) (robby_saving_fraction jaylen_saving_fraction miranda_saving_fraction : ℚ)
  (salary_eq : salary_per_hour = 10)
  (hours_per_day_eq : hours_per_day = 10)
  (days_per_week_eq : days_per_week = 5)
  (weeks_eq : weeks = 4)
  (robby_saving_eq : robby_saving_fraction = 2/5)
  (jaylen_saving_eq : jaylen_saving_fraction = 3/5)
  (miranda_saving_eq : miranda_saving_fraction = 1/2) :
  4 * (50 * $10 * ((2/5) + (3/5) + (1/2))) = 3000 :=
by sorry

end combined_savings_l314_314612


namespace eqn_distinct_real_roots_l314_314000

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then x^2 + 2 else 4 * x * Real.cos x + 1

theorem eqn_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, f x = m * x + 1) → 
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-2 * Real.pi) Real.pi ∧ x₂ ∈ Set.Icc (-2 * Real.pi) Real.pi :=
  sorry

end eqn_distinct_real_roots_l314_314000


namespace angle_measure_triple_complement_l314_314233

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314233


namespace value_of_f_at_2_l314_314735

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem value_of_f_at_2 : f 2 = 3 := by
  -- Definition of the function f.
  -- The goal is to prove that f(2) = 3.
  sorry

end value_of_f_at_2_l314_314735


namespace part1_solution_set_part2_range_of_a_l314_314920

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314920


namespace equal_sum_of_children_numbers_l314_314366

theorem equal_sum_of_children_numbers
  (k : Fin 10 → ℕ) (n : Fin 10 → ℕ)
  (H_k : ∀ i : Fin 10, 1 ≤ k i ∧ k i ≤ 20)
  (H_n : ∀ j : Fin 10, 1 ≤ n j ∧ n j ≤ 20)
  (all_distinct : ∀ i j : Fin 10, i ≠ j → k i ≠ k j ∧ n i ≠ n j)
  (H_total_positions : ∑ i : Fin 10, k i + ∑ j : Fin 10, n j = 210) :
  ∑ i : Fin 10, (20 - k i) = ∑ j : Fin 10, (n j - 1) :=
by
  sorry

end equal_sum_of_children_numbers_l314_314366


namespace find_number_1920_find_number_60_l314_314160

theorem find_number_1920 : 320 * 6 = 1920 :=
by sorry

theorem find_number_60 : (1920 / 7 = 60) :=
by sorry

end find_number_1920_find_number_60_l314_314160


namespace lazy_kingdom_day_l314_314698

theorem lazy_kingdom_day : 
  let decree_day := 1  -- Sunday is denoted as day 1 
  let starting_date := (2007, 4, 1)  -- (Year, Month, Day)
  let days_in_leap_year := 366
  let days_in_lazy_week := 6
  let days_until_april_9_2008 := days_in_leap_year + 8
  let total_weeks := days_until_april_9_2008 / days_in_lazy_week
  let remaining_days := days_until_april_9_2008 % days_in_lazy_week
  remaining_days = 2 -> "Tuesday" :=
begin
  -- Definitions
  let decree_day := 1  -- Sunday is denoted as day 1
  let (year, month, day) := (2007, 4, 1)
  let days_in_leap_year := 366
  let adjust_friday := 1  -- No Friday
  
  -- Calculate total days and remaining days after complete weeks
  let days_until_april_9_2008 := days_in_leap_year + 8
  let days_in_lazy_week := 6
  let (weeks, remainder) := days_until_april_9_2008 /% days_in_lazy_week
  
  -- Prove remainder = 2 implies Tuesday
  have h1 : remainder = 2 := sorry,
  rw h1,
  exact "Tuesday"
end

end lazy_kingdom_day_l314_314698


namespace angle_measure_triple_complement_l314_314229

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314229


namespace minimum_at_neg3_l314_314345

def f (x : ℝ) : ℝ := x^2 + 6x + 1

theorem minimum_at_neg3 : ∀ x : ℝ, f x ≥ f (-3) := by
  sorry

end minimum_at_neg3_l314_314345


namespace number_of_elements_in_M_l314_314590

-- Definitions of sets A, B, and M
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {4, 5}
def M : Set ℕ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a + b}

-- Theorem stating the number of elements in the set M is 4
theorem number_of_elements_in_M : M.card = 4 := by
  sorry

end number_of_elements_in_M_l314_314590


namespace angle_is_67_l314_314207

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l314_314207


namespace angle_triple_complement_l314_314295

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314295


namespace nancy_ate_3_apples_l314_314602

theorem nancy_ate_3_apples
  (mike_apples : ℝ)
  (keith_apples : ℝ)
  (apples_left : ℝ)
  (mike_apples_eq : mike_apples = 7.0)
  (keith_apples_eq : keith_apples = 6.0)
  (apples_left_eq : apples_left = 10.0) :
  mike_apples + keith_apples - apples_left = 3.0 := 
by
  rw [mike_apples_eq, keith_apples_eq, apples_left_eq]
  norm_num

end nancy_ate_3_apples_l314_314602


namespace tank_capacity_l314_314389

theorem tank_capacity :
  (∃ c : ℝ, c > 0 ∧ ∀ w : ℝ, w = c / 5 ∧ (w + 6) / c = 1 / 3) → c = 45 :=
by
  assume h,
  rcases h with ⟨c, c_pos, hc⟩,
  let w := c / 5,
  have hw : w = c / 5 := hc.1,
  have h2 : (w + 6) / c = 1 / 3 := hc.2,
  have h_eq : (c / 5 + 6) / c = 1 / 3,
  { rw hw at h2,
    exact h2 },
  sorry

end tank_capacity_l314_314389


namespace lcm_24_36_45_l314_314330

noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem lcm_24_36_45 : lcm_three 24 36 45 = 360 := by
  -- Conditions involves proving the prime factorizations
  have h1 : Nat.factors 24 = [2, 2, 2, 3] := by sorry -- Prime factorization of 24
  have h2 : Nat.factors 36 = [2, 2, 3, 3] := by sorry -- Prime factorization of 36
  have h3 : Nat.factors 45 = [3, 3, 5] := by sorry -- Prime factorization of 45

  -- Least common multiple calculation based on the greatest powers of prime factors
  sorry -- This is where the proof would go

end lcm_24_36_45_l314_314330


namespace chocolate_chips_l314_314382

theorem chocolate_chips (batches : ℝ) (cups_per_batch : ℝ) (total_chips : ℝ) :
    batches = 11.5 →
    cups_per_batch = 2.0 →
    total_chips = batches * cups_per_batch →
    total_chips = 23 :=
by
  intros hb hc ht
  rw [hb, hc] at ht
  exact ht
  sorry 

end chocolate_chips_l314_314382


namespace angle_measure_triple_complement_l314_314305

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314305


namespace value_of_z_plus_12_over_z_l314_314094

noncomputable def z : ℂ := sorry
axiom h1 : 18 * (complex.abs z)^2 = 2 * (complex.abs (z + 3))^2 + (complex.abs (z^2 + 2))^2 + 48

theorem value_of_z_plus_12_over_z :
  z + (12 / z) = -3 :=
sorry

end value_of_z_plus_12_over_z_l314_314094


namespace solve_factors_l314_314739

theorem solve_factors : ∃ x y : ℕ, (y = x + 10) ∧ 
  (x * y - 40 = 39 * x + 22) ∧ 
  (x = 31) ∧ 
  (y = 41) :=
by
  let x := 31
  let y := 41
  exists x, y
  simp
  split
  {
    use 31, 41
    simp
    sorry
  }
  sorry

end solve_factors_l314_314739


namespace minimum_distance_l314_314426

-- Definition of the curve and the line
def curve (x : ℝ) := x - 2 * Real.log x
def line (x y : ℝ) := x + y + 2

-- Point P on the curve
def P (x: ℝ) := (x, curve x)

-- Distance between a point and a line
def distance_to_line (x y : ℝ) (A B C: ℝ) := abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

-- Specific distance from point (1, curve(1)) to the line x + y + 2 = 0
def distance_from_P_to_line := distance_to_line 1 (curve 1) 1 1 2

-- Theorem claiming the minimum distance
theorem minimum_distance : distance_from_P_to_line = 2 * Real.sqrt 2 := by sorry

end minimum_distance_l314_314426


namespace minimize_integral_l314_314472

open Real
open interval_integral

theorem minimize_integral (a : ℝ) (k : ℝ) (h1 : 0 < a) (h2 : a < (π / 2)) (h3 : cos a = k * a) :
  k = (2 * sqrt 2 / π) * cos (π / (2 * sqrt 2)) :=
sorry

end minimize_integral_l314_314472


namespace angle_measure_triple_complement_l314_314227

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l314_314227


namespace courtiers_selection_l314_314672

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l314_314672


namespace find_f_of_3_l314_314871

theorem find_f_of_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f_of_3_l314_314871


namespace min_value_quadratic_function_l314_314156

open Real

theorem min_value_quadratic_function (m n : ℝ) (h_m : 0 ≤ m) (h_n : 0 ≤ n) :
  (m - sqrt(m) ≥ -1/4) ∧ (1/2 * (m + n)^2 + 1/4 * (m + n) ≥ m * sqrt(n) + n * sqrt(m)) :=
by
  sorry

end min_value_quadratic_function_l314_314156


namespace find_triangle_angles_l314_314724

-- Definitions of the points and triangles
variables {A D B M : Type*}

-- Setup the conditions: M inside the square, and the triangles \triangle MAD and \triangle MAB are congruent.
structure Triangle := (vertex1 vertex2 vertex3 : Type*)
def square (MA MD MB : Type*) (A B D : Type*) := 
  is_square MD MB A B D 

def equal_triangles (MAD MAB : Triangle) :=
  congruent_by_sss MAD MAB

-- The main theorem to be proved
theorem find_triangle_angles 
  (A D B M : Type*)
  (triangle1 : Triangle A D M)
  (triangle2 : Triangle A B M)
  (h_square : square MA MD MB A B D)
  (h_equal : equal_triangles triangle1 triangle2) :
  ∃ α β γ : ℝ, α = 120 ∧ β = 45 ∧ γ = 15 :=
sorry

end find_triangle_angles_l314_314724


namespace find_ratio_l314_314480

theorem find_ratio (a b : ℝ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 3 :=
sorry

end find_ratio_l314_314480


namespace part1_solution_set_part2_range_of_a_l314_314915

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314915


namespace length_of_tangent_segment_to_circle_at_origin_l314_314370

theorem length_of_tangent_segment_to_circle_at_origin :
  ∀ P : Point, 
  (x - 2)^2 + (y - 1)^2 = 1 → 
  distance origin P = 2 → 
  tangent origin P :=
sorry

end length_of_tangent_segment_to_circle_at_origin_l314_314370


namespace proof_problem_l314_314064

def polar_curve_C (ρ : ℝ) : Prop := ρ = 5

def point_P (x y : ℝ) : Prop := x = -3 ∧ y = -3 / 2

def line_l_through_P (x y : ℝ) (k : ℝ) : Prop := y + 3 / 2 = k * (x + 3)

def distance_AB (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64

theorem proof_problem
  (ρ : ℝ) (x y : ℝ) (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : polar_curve_C ρ)
  (h2 : point_P (-3) (-3 / 2))
  (h3 : ∃ k, line_l_through_P x y k)
  (h4 : distance_AB A B) :
  ∃ (x y : ℝ), (x^2 + y^2 = 25) ∧ ((x = -3) ∨ (3 * x + 4 * y + 15 = 0)) := 
sorry

end proof_problem_l314_314064


namespace part1_part2_l314_314995

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l314_314995


namespace Mahesh_completes_in_60_days_l314_314109

noncomputable def MaheshWork (W : ℝ) : ℝ :=
    W / 60

variables (W : ℝ)
variables (M R : ℝ)
variables (daysMahesh daysRajesh daysFullRajesh : ℝ)

theorem Mahesh_completes_in_60_days
  (h1 : daysMahesh = 20)
  (h2 : daysRajesh = 30)
  (h3 : daysFullRajesh = 45)
  (hR : R = W / daysFullRajesh)
  (hM : M = (W - R * daysRajesh) / daysMahesh) :
  W / M = 60 :=
by
  sorry

end Mahesh_completes_in_60_days_l314_314109


namespace time_to_pass_platform_l314_314797

noncomputable def trainLength : ℝ := 720  -- in meters
noncomputable def trainSpeed : ℝ := 90 * 1000 / 3600  -- converting to meters per second
noncomputable def platformLength : ℝ := 580  -- in meters

theorem time_to_pass_platform :
  let totalDistance := trainLength + platformLength,
      speedInMetersPerSecond := trainSpeed in
  let time := totalDistance / speedInMetersPerSecond in
  time = 52 :=
by
  sorry

end time_to_pass_platform_l314_314797


namespace problem_statement_l314_314035

noncomputable section

def complex_series_sum : ℂ :=
  ∑ n in Finset.range 41, (complex.I ^ n * real.cos ((45 + 90 * n) * (real.pi / 180)))

theorem problem_statement :
  complex_series_sum = (real.sqrt 2 / 2 * (21 - 20 * complex.I)) :=
sorry

end problem_statement_l314_314035


namespace ab_relationship_l314_314863

theorem ab_relationship (a b : ℝ) (n : ℕ) (h1 : a^n = a + 1) (h2 : b^(2*n) = b + 3*a) (h3 : n ≥ 2) (h4 : 0 < a) (h5 : 0 < b) :
  a > b ∧ a > 1 ∧ b > 1 :=
sorry

end ab_relationship_l314_314863


namespace exist_two_courtiers_with_same_selection_l314_314695

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l314_314695


namespace top_leftmost_rectangle_is_B_l314_314839

structure Rectangle :=
  (w x y z : ℕ)

def RectangleA := Rectangle.mk 5 1 9 2
def RectangleB := Rectangle.mk 2 0 6 3
def RectangleC := Rectangle.mk 6 7 4 1
def RectangleD := Rectangle.mk 8 4 3 5
def RectangleE := Rectangle.mk 7 3 8 0

-- Problem Statement: Given these rectangles, prove that the top leftmost rectangle is B.
theorem top_leftmost_rectangle_is_B 
  (A : Rectangle := RectangleA)
  (B : Rectangle := RectangleB)
  (C : Rectangle := RectangleC)
  (D : Rectangle := RectangleD)
  (E : Rectangle := RectangleE) : 
  B = Rectangle.mk 2 0 6 3 := 
sorry

end top_leftmost_rectangle_is_B_l314_314839


namespace sum_p_n_eq_factorial_l314_314587

noncomputable def p_n (n k : ℕ) : ℕ := sorry

theorem sum_p_n_eq_factorial (n : ℕ) : 
  (∑ k in finset.range (n + 1), p_n n k) = Nat.factorial n := 
sorry

end sum_p_n_eq_factorial_l314_314587


namespace books_remaining_correct_l314_314183

-- Define the initial number of book donations
def initial_books : ℕ := 300

-- Define the number of people donating and the number of books each donates
def num_people : ℕ := 10
def books_per_person : ℕ := 5

-- Calculate total books donated by all people
def total_donation : ℕ := num_people * books_per_person

-- Define the number of books borrowed by other people
def borrowed_books : ℕ := 140

-- Calculate the total number of books after donations and then subtract the borrowed books
def total_books_remaining : ℕ := initial_books + total_donation - borrowed_books

-- Prove the total number of books remaining is 210
theorem books_remaining_correct : total_books_remaining = 210 := by
  sorry

end books_remaining_correct_l314_314183


namespace find_a_minus_b_l314_314707

-- Define the problem conditions 
def quadratic_inequality_solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, ax² + bx + 2 > 0 ↔ x ∈ Ioo (-1/2) (1/3)

-- Prove the desired result
theorem find_a_minus_b (a b : ℝ) (h : quadratic_inequality_solution_set a b) : a - b = -10 := 
sorry

end find_a_minus_b_l314_314707


namespace angle_measure_l314_314271

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l314_314271


namespace angle_measure_triple_complement_l314_314235

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314235


namespace growing_path_5x5_l314_314631

def point := (ℕ × ℕ)

def distance (p1 p2 : point) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def unique_distances (path : list point) : Prop := 
  ∀ i j, i < j → distance (path.nth i).get_or_else (0, 0) (path.nth (i + 1)).get_or_else (0, 0) <
                 distance (path.nth j).get_or_else (0, 0) (path.nth (j + 1)).get_or_else (0, 0)

def grid (n : ℕ) : list point := 
  (list.range n).bind (λ x, (list.range n).map (λ y, (x, y)))

theorem growing_path_5x5 :
  ∃ path : list point, path.length = 12 ∧ unique_distances path ∧ 
  (list.countp (λ p, unique_distances p ∧ p.length = 12) (grid 5)) = 24 ∧ 12 * 24 = 288 :=
by
  sorry

end growing_path_5x5_l314_314631


namespace lcm_24_36_45_l314_314331

noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem lcm_24_36_45 : lcm_three 24 36 45 = 360 := by
  -- Conditions involves proving the prime factorizations
  have h1 : Nat.factors 24 = [2, 2, 2, 3] := by sorry -- Prime factorization of 24
  have h2 : Nat.factors 36 = [2, 2, 3, 3] := by sorry -- Prime factorization of 36
  have h3 : Nat.factors 45 = [3, 3, 5] := by sorry -- Prime factorization of 45

  -- Least common multiple calculation based on the greatest powers of prime factors
  sorry -- This is where the proof would go

end lcm_24_36_45_l314_314331


namespace max_k_value_l314_314832

theorem max_k_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = a * b + b * c + c * a →
  (a + b + c) * (1 / (a + b) + 1 / (b + c) + 1 / (c + a) - 1) ≥ 1 :=
by
  intros a b c ha hb hc habc_eq
  sorry

end max_k_value_l314_314832


namespace find_sales_discount_l314_314387

noncomputable def salesDiscountPercentage (P N : ℝ) (D : ℝ): Prop :=
  let originalGrossIncome := P * N
  let newPrice := P * (1 - D / 100)
  let newNumberOfItems := N * 1.20
  let newGrossIncome := newPrice * newNumberOfItems
  newGrossIncome = originalGrossIncome * 1.08

theorem find_sales_discount (P N : ℝ) (hP : P > 0) (hN : N > 0) (h: ∃ D, salesDiscountPercentage P N D) :
  ∃ D, D = 10 :=
sorry

end find_sales_discount_l314_314387


namespace symmetry_proof_l314_314063

-- Define the initial point P and its reflection P' about the x-axis
def P : ℝ × ℝ := (-1, 2)
def P' : ℝ × ℝ := (-1, -2)

-- Define the property of symmetry about the x-axis
def symmetric_about_x_axis (P P' : ℝ × ℝ) : Prop :=
  P'.fst = P.fst ∧ P'.snd = -P.snd

-- The theorem to prove that point P' is symmetric to point P about the x-axis
theorem symmetry_proof : symmetric_about_x_axis P P' :=
  sorry

end symmetry_proof_l314_314063


namespace log_inequality_solution_l314_314857

theorem log_inequality_solution (x : ℝ) :
  log (1/3) (2*x - 1) > 1 → (1/2 : ℝ) < x ∧ x < (2/3 : ℝ) :=
sorry

end log_inequality_solution_l314_314857


namespace acute_triangle_eq_AP_AQ_l314_314086

variable {ABC : Type*} 
variables [acute_triangle ABC] (D E F : Point) (alt_BC : Line) (alt_CA : Line) (alt_AB : Line)
          (P Q : Point) (circumcircle_ABC : Circle) 
          (line_EF : Line) (line_BP : Line) (line_DF : Line)

-- Define D, E, F as the feet of the altitudes from the vertices to the opposite sides
variables [feet_alt_D : altitude_from_vertex ABC D alt_BC] 
          [feet_alt_E : altitude_from_vertex ABC E alt_CA] 
          [feet_alt_F : altitude_from_vertex ABC F alt_AB]

-- Define P as the intersection of line (EF) with the circumcircle of triangle ABC
variable [intersection_EF_P : intersection_when_line_circle P line_EF circumcircle_ABC]

-- Define Q as the intersection of lines (BP) and (DF)
variable [intersection_BP_DF_Q : intersection_when_lines Q line_BP line_DF]

-- The goal is to prove: AP = AQ
theorem acute_triangle_eq_AP_AQ : distance_between_points ABC A P = distance_between_points ABC A Q :=
by
  sorry

end acute_triangle_eq_AP_AQ_l314_314086


namespace sum_of_permutation_not_all_ones_l314_314048

-- Definitions based on conditions
def no_zero_digits (n : ℕ) : Prop :=
∀ d ∈ (n.digits 10), d ≠ 0

def is_permutation (a b : ℕ) : Prop :=
a.digits 10 ~ b.digits 10

-- The theorem statement
theorem sum_of_permutation_not_all_ones (a b : ℕ) 
  (h1 : no_zero_digits a)
  (h2 : is_permutation a b) :
  (a + b).digits 10 ≠ list.replicate (a.digits 10).length 1 := 
sorry

end sum_of_permutation_not_all_ones_l314_314048


namespace car_return_speed_l314_314381

theorem car_return_speed (d : ℕ) (speed_CD : ℕ) (avg_speed_round_trip : ℕ) 
  (round_trip_distance : ℕ) (time_CD : ℕ) (time_round_trip : ℕ) (r: ℕ) 
  (h1 : d = 150) (h2 : speed_CD = 75) (h3 : avg_speed_round_trip = 60)
  (h4 : d * 2 = round_trip_distance) 
  (h5 : time_CD = d / speed_CD) 
  (h6 : time_round_trip = time_CD + d / r) 
  (h7 : avg_speed_round_trip = round_trip_distance / time_round_trip) :
  r = 50 :=
by {
  -- proof steps will go here
  sorry
}

end car_return_speed_l314_314381


namespace space_shuttle_speed_l314_314401

-- Define the conditions in Lean
def speed_kmph : ℕ := 43200 -- Speed in kilometers per hour
def seconds_per_hour : ℕ := 60 * 60 -- Number of seconds in an hour

-- Define the proof problem
theorem space_shuttle_speed :
  speed_kmph / seconds_per_hour = 12 := by
  sorry

end space_shuttle_speed_l314_314401


namespace part1_part2_l314_314974

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314974


namespace angle_triple_complement_l314_314200

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314200


namespace turnips_total_l314_314599

def melanie_turnips := 139
def benny_turnips := 113

def total_turnips (melanie_turnips benny_turnips : Nat) : Nat :=
  melanie_turnips + benny_turnips

theorem turnips_total :
  total_turnips melanie_turnips benny_turnips = 252 :=
by
  sorry

end turnips_total_l314_314599


namespace smallest_odd_digit_n_l314_314100

theorem smallest_odd_digit_n {n : ℕ} (h : n > 1) : 
  (∀ d ∈ (Nat.digits 10 (9997 * n)), d % 2 = 1) → n = 3335 :=
sorry

end smallest_odd_digit_n_l314_314100


namespace angle_triple_complement_l314_314258

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314258


namespace part_a_part_b_l314_314440

noncomputable def seq (a : ℕ → ℕ) := 
  a 0 = 0 ∧ a 1 = 1 ∧ ∀ n ≥ 2, a n = 2 * a (n - 1) + a (n - 2)

theorem part_a (a : ℕ → ℕ) (h_seq : seq a) (m : ℕ) (hm : m > 0) (j : ℕ) (hj : j ≤ m) : 
  2 * a m ∣ a (m + j) + (-1) ^ j * a (m - j) := 
sorry

theorem part_b (a : ℕ → ℕ) (h_seq : seq a) (n k : ℕ) (hk : 2 ^ k ∣ n) : 
  2 ^ k ∣ a n := 
sorry

end part_a_part_b_l314_314440


namespace unique_real_value_for_equal_roots_l314_314446

-- Definitions of conditions
def quadratic_eq (p : ℝ) : Prop := 
  ∀ x : ℝ, x^2 - (p + 1) * x + p = 0

-- Statement of the problem
theorem unique_real_value_for_equal_roots :
  ∃! p : ℝ, ∀ x y : ℝ, (x^2 - (p+1)*x + p = 0) ∧ (y^2 - (p+1)*y + p = 0) → x = y := 
sorry

end unique_real_value_for_equal_roots_l314_314446


namespace nearest_value_to_sum_of_fractions_is_5_l314_314749

theorem nearest_value_to_sum_of_fractions_is_5 :
  let sum := (2007 / 2999) + (8001 / 5998) + (2001 / 3999) + (4013 / 7997) + (10007 / 15999) + (2803 / 11998)
  in abs ((5:ℝ) - sum) < abs ((1:ℝ) - sum) ∧ 
     abs ((5:ℝ) - sum) < abs ((2:ℝ) - sum) ∧ 
     abs ((5:ℝ) - sum) < abs ((3:ℝ) - sum) ∧ 
     abs ((5:ℝ) - sum) < abs ((4:ℝ) - sum) := by
  sorry

end nearest_value_to_sum_of_fractions_is_5_l314_314749


namespace permutation_count_l314_314101

theorem permutation_count :
  (∑ k in finset.range 30, abs ((p : ℕ ∘ finset.emb_domain (equiv.perm.of_fin (fin.perm 30 k))) - k) = 450) ->
  (finset.univ.filter (λ p, ∑ k in finset.range 30, abs (p k - k) = 450)).card = (15.factorial)^2 :=
begin
  sorry
end

end permutation_count_l314_314101


namespace seashells_remainder_l314_314443

theorem seashells_remainder :
  let derek := 58
  let emily := 73
  let fiona := 31 
  let total_seashells := derek + emily + fiona
  total_seashells % 10 = 2 :=
by
  sorry

end seashells_remainder_l314_314443


namespace part1_part2_l314_314975

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l314_314975


namespace problem_1_l314_314374

def f (x : ℝ) := 2 * x + 2 * Real.sin x + Real.cos x

theorem problem_1 (α : ℝ) (h : deriv f α = 2) :
  (Real.sin (π - α) + Real.cos (-α)) / (2 * Real.cos (π / 2 - α) + Real.cos (2 * π - α)) = 3 / 5 :=
by sorry

end problem_1_l314_314374


namespace total_pencils_l314_314458

-- Define the number of rows and pencils per row as axioms
axiom num_rows : ℕ
axiom pencils_per_row : ℕ

-- Set the problem-specific values
def rows : ℕ := 30
def pencils_each_row : ℕ := 24

-- Prove that the total number of pencils is equal to 720
theorem total_pencils : num_rows = rows → pencils_per_row = pencils_each_row → num_rows * pencils_per_row = 720 := by
  intros h_rows h_pencils_each_row
  rw [h_rows, h_pencils_each_row]
  norm_num
  exact 720_builtin

end total_pencils_l314_314458


namespace termites_ate_black_squares_l314_314133

-- Define a Lean function to check if a given cell is black on a standard chessboard.
def is_black (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Define the theorem that asserts the number of black squares in a 3x8 block is 12.
theorem termites_ate_black_squares : 
  (finset.univ.filter (λ k : ℕ × ℕ, k.1 < 3 ∧ k.2 < 8 ∧ is_black k.1 k.2)).card = 12 :=
by
  sorry -- proof to be provided

end termites_ate_black_squares_l314_314133


namespace sum_f_1_to_2010_eq_zero_l314_314510

theorem sum_f_1_to_2010_eq_zero :
  let f (x : ℕ) := Real.sin (x * Real.pi / 3)
  in (Finset.range 2010).sum (λ x, f (x + 1)) = 0 :=
by
  let f : ℕ → ℝ := λ x, Real.sin (x * Real.pi / 3)
  have periodicity : ∀ n, f (n + 6) = f n := by
    intro n
    simp [f, Real.sin_add, Real.sin_two_pi, Real.cos_two_pi]
  sorry

end sum_f_1_to_2010_eq_zero_l314_314510


namespace valid_assignment_statement_l314_314352

theorem valid_assignment_statement (S a : ℕ) : (S = a + 1) ∧ ¬(a + 1 = S) ∧ ¬(S - 1 = a) ∧ ¬(S - a = 1) := by
  sorry

end valid_assignment_statement_l314_314352


namespace ellipse_equation_and_fixed_point_l314_314890

theorem ellipse_equation_and_fixed_point :
  ∃ E : ellipse,
    (center E = (0, 0)) ∧
    (foci_distance E = 2) ∧
    (eccentricity E = sqrt 2 / 2) ∧
    (equation E = (λ x y, x^2 / 2 + y^2 = 1)) ∧
    ∃ M : ℝ × ℝ,
      (M = (5/4, 0)) ∧
      ∀ l : line,
        passes_through l (1, 0) →
        ∀ P Q : ℝ × ℝ,
          line_intersects_with_ellipse l E P Q →
          (dot_prod (vector M P) (vector M Q)) = -7/16 :=
sorry

end ellipse_equation_and_fixed_point_l314_314890


namespace coeff_of_x4_in_expansion_l314_314850

noncomputable def find_coef_x4 : ℕ :=
  let coeff_x4 := 15 in
  coeff_x4

theorem coeff_of_x4_in_expansion : 
  ∃ k : ℕ, k = 15 ∧ (∀ f : ℕ → ℝ, f 4 = k) := 
by 
  sorry

end coeff_of_x4_in_expansion_l314_314850


namespace angle_triple_complement_l314_314192

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314192


namespace angle_triple_complement_l314_314194

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314194


namespace exist_two_courtiers_with_same_selection_l314_314696

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l314_314696


namespace greatest_divisible_by_13_l314_314789

theorem greatest_divisible_by_13 (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) : (10000 * A + 1000 * B + 100 * C + 10 * B + A = 96769) 
  ↔ (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 :=
sorry

end greatest_divisible_by_13_l314_314789


namespace part1_solution_set_part2_range_a_l314_314953

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314953


namespace complex_subtraction_multiplication_l314_314454

theorem complex_subtraction_multiplication :
  let i : ℂ := complex.I in
  (6 - 3 * i) - (2 + 5 * i) * (2 * i) = 16 + 8 * i :=
by
  sorry

end complex_subtraction_multiplication_l314_314454


namespace infinite_squares_in_seq_S_l314_314895

def seq_a (u v : ℕ) : ℕ → ℕ
| 1       := u + v
| (2*m)   := seq_a (u := u) (v := v) m + u
| (2*m+1) := seq_a (u := u) (v := v) m + v

def sum_S (u v : ℕ) (m : ℕ) : ℕ :=
(list.range m).sum (λ i, seq_a u v (i + 1))

theorem infinite_squares_in_seq_S (u v : ℕ) :
  ∃ (infinitely_many_n : ℕ → Prop), 
  (∀ n : ℕ, infinitely_many_n n → ∃ m : ℕ, sum_S u v m = n * n) :=
sorry

end infinite_squares_in_seq_S_l314_314895


namespace no_closed_broken_line_with_odd_vertices_l314_314117

theorem no_closed_broken_line_with_odd_vertices :
  (∀ (vertices : List (ℚ × ℚ)), 
    let len_eq_one := ∀ i, dist(vertices[i], vertices[(i+1) % vertices.length]) = 1
    len_eq_one ∧ vertices.length % 2 = 1 → false) :=
begin
  sorry
end

end no_closed_broken_line_with_odd_vertices_l314_314117


namespace total_time_to_make_cookies_l314_314864

def time_to_make_batter := 10
def baking_time := 15
def cooling_time := 15
def white_icing_time := 30
def chocolate_icing_time := 30

theorem total_time_to_make_cookies : 
  time_to_make_batter + baking_time + cooling_time + white_icing_time + chocolate_icing_time = 100 := 
by
  sorry

end total_time_to_make_cookies_l314_314864


namespace find_polynomial_l314_314376

def is_sympa (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

theorem find_polynomial (P : ℝ[X]) :
  (∀ n : ℕ, is_sympa n → is_sympa (P.eval n)) ↔
  (∃ d ≥ 1, ∃ a : ℤ, P = λ x, (10^a / 9) * (9 * x + 1)^d - 1/9) :=
by
  sorry

end find_polynomial_l314_314376


namespace smallest_absolute_value_is_zero_l314_314748

-- Definitions focusing on the natural number, rational number, and absolute value
def is_natural_number (n : ℕ) : Prop := true

def is_rational_number (q : ℚ) : Prop := true

def has_reciprocal (q : ℚ) : Prop := q ≠ 0

def abs_is_smallest (x : ℝ) : Prop := abs x ≥ 0

theorem smallest_absolute_value_is_zero : ∃ x : ℝ, abs x = 0 :=
by {
  use 0,
  simp,
  sorry
}

end smallest_absolute_value_is_zero_l314_314748


namespace max_connected_groups_l314_314171

namespace SpaceStations

-- Defining the conditions as Lean hypotheses
def total_stations : ℕ := 99
def two_way_main_tunnels : ℕ := 99
def one_way_tunnels : ℕ := total_stations * (total_stations - 1) / 2 - two_way_main_tunnels

-- Connected group definition
def is_connected_group (stations : Finset ℕ) (tunnels : Finset (ℕ × ℕ)) : Prop :=
  stations.card = 4 ∧
  (∀ (a b : ℕ), a ∈ stations → b ∈ stations → (a = b) ∨ ((a, b) ∈ tunnels ∧ (b, a) ∈ tunnels))

-- Theorem: Determine the maximum number of connected groups of 4 space stations.
theorem max_connected_groups :
  ∃ n : ℕ, n = (Nat.choose total_stations 4) - total_stations * (Nat.choose 48 3) :=
begin
  sorry,
end

end SpaceStations

end max_connected_groups_l314_314171


namespace part1_part2_l314_314938

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l314_314938


namespace number_of_real_values_of_a_for_equal_roots_l314_314092

theorem number_of_real_values_of_a_for_equal_roots :
  let a : ℝ := sorry
  let quadratic_eq : ℝ → ℝ × ℝ := fun x => (x^2 - (a+1)*x + a, 0)
  (quadratic_eq a).equality_criterion :=
  ∃! a, (quadratic_eq a).equality_criterion := sorry

end number_of_real_values_of_a_for_equal_roots_l314_314092


namespace b_five_b_geq_five_l314_314887

def a : ℕ → ℕ
| 0     := 0  -- since we start sequences at 1
| 1     := 1 
| 2     := 2 
| 3     := 3 
| 4     := 4 
| 5     := 5 
| (n+1) := if n ≥ 5 then a 1 * a 2 * a 3 * a 4 * a 5 - 1 else 0 

def b (n : ℕ) : ℕ := 
match n with
| 0     := 0  -- No definition needed for n=0, if sequences start at 1
| nat.succ n' := a 1 * a 2 * a 3 * a 4 * a 5 - (a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2)

theorem b_five : b 5 = 65 :=
sorry

theorem b_geq_five (n : ℕ) (h : n ≥ 5) : b n = 70 - n :=
sorry

end b_five_b_geq_five_l314_314887


namespace part1_part2_l314_314997

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l314_314997


namespace hunting_dogs_theorem_l314_314658

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l314_314658


namespace part1_part2_l314_314914

def f (a x : ℝ) : ℝ := x - a / x - 2 * Real.log x

-- Part 1: If f has two extreme points, then a ∈ (1, +∞)
theorem part1 (a x1 x2 : ℝ) (h1 : f' a x1 = 0) (h2 : f' a x2 = 0) (h3 : x1 ≠ x2) :
  1 < a :=
sorry

-- Part 2: If f(x1) + f(x2) > -2e, then a ∈ (1, e)
theorem part2 (a x1 x2 : ℝ) (h4 : f a x1 + f a x2 > -2 * Real.exp 1) :
  1 < a ∧ a < Real.exp 1 :=
sorry

end part1_part2_l314_314914


namespace unique_solution_c_eq_one_l314_314466

theorem unique_solution_c_eq_one (b c : ℝ) (hb : b > 0) 
  (h_unique_solution : ∃ x : ℝ, x^2 + (b + 1/b) * x + c = 0 ∧ 
  ∀ y : ℝ, y^2 + (b + 1/b) * y + c = 0 → y = x) : c = 1 :=
by
  sorry

end unique_solution_c_eq_one_l314_314466


namespace slow_train_length_correct_l314_314727

noncomputable def length_of_slower_train (speed_fast speed_slow : ℝ) (length_fast : ℝ) (crossing_time : ℝ) : ℝ :=
  let length_slow := 4900
  in length_slow

theorem slow_train_length_correct :
  ∀ (speed_fast speed_slow : ℝ) (length_fast crossing_time : ℝ),
    speed_fast = 150 → speed_slow = 90 → length_fast = 1.1 → crossing_time = 30 →
    length_of_slower_train speed_fast speed_slow length_fast crossing_time = 4900 :=
by
  intros speed_fast speed_slow length_fast crossing_time
  rintros rfl rfl rfl rfl
  simp [length_of_slower_train]
  sorry

end slow_train_length_correct_l314_314727


namespace angle_measure_triple_complement_l314_314230

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314230


namespace area_relation_l314_314081

variables {A B C K D E F : Type}
variable [realization]

def acute_triangle (ABC : Triangle) : Prop := 
  ∀ (a b c : Point), triangle ABC a b c → 
  ang A B C < π ∧ ang B C A < π ∧ ang C A B < π

def circle_with_diameter (d : Segment) : Circle := sorry

noncomputable def radical_center (k1 k2 k3 : Circle) : Point := sorry

variable (u x y z : ℝ)

def area (𝑇 : Triangle) : ℝ := sorry

theorem area_relation 
  (ABC : Triangle) 
  (k1 k2 k3 : Circle)
  (D E F : Point)
  (BC CA AB : Segment)
  (hacute : acute_triangle ABC)
  (hcircle1 : k1 = circle_with_diameter BC)
  (hcircle2 : k2 = circle_with_diameter CA)
  (hcircle3 : k3 = circle_with_diameter AB)
  (hrad : K = radical_center k1 k2 k3)
  (hint1 : AK ∩ k1 = {D})
  (hint2 : BK ∩ k2 = {E})
  (hint3 : CK ∩ k3 = {F})
  (hu : u = area ABC)
  (hx : x = area (triangle.mk D B C))
  (hy : y = area (triangle.mk E C A))
  (hz : z = area (triangle.mk F A B)) :
  u^2 = x^2 + y^2 + z^2 :=
sorry

end area_relation_l314_314081


namespace min_value_of_sum_cubed_l314_314881

open scoped Classical

noncomputable theory

variables {n : ℕ} (x : ℕ → ℝ)

def conditions (n : ℕ) (x : ℕ → ℝ) :=
  n ≥ 3 ∧ 
  (∑ i in Finset.range n, x i) = n ∧ 
  (∑ i in Finset.range n, (x i)^2) = n^2

theorem min_value_of_sum_cubed (n : ℕ) (x : ℕ → ℝ)
  (h : conditions n x) : ∑ i in Finset.range n, (x i)^3 = -(n^3) + 6*(n^2) - 4*n :=
sorry

end min_value_of_sum_cubed_l314_314881


namespace distinct_four_digit_prime_numbers_l314_314019

theorem distinct_four_digit_prime_numbers : 
  (number_of_distinct_four_digit_numbers {2, 3, 5, 7} = 256) :=
by
  -- Definitions and conditions
  let prime_digits := {2, 3, 5, 7}
  let number_of_positions := 4

  -- Calculation
  let total_number := prime_digits.card ^ number_of_positions

  -- Proof statement
  have : total_number = 256 := sorry
  exact this

end distinct_four_digit_prime_numbers_l314_314019


namespace hunting_dogs_theorem_l314_314662

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l314_314662


namespace intersection_is_correct_l314_314012

noncomputable def A : Set ℝ := { x : ℝ | x^2 - x - 2 ≤ 0 }
noncomputable def B : Set ℝ := { x : ℝ | x < 1 }

theorem intersection_is_correct : (A ∩ B) = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_is_correct_l314_314012


namespace contrapositive_equiv_l314_314350

variable {α : Type}  -- Type of elements
variable (P : Set α) (a b : α)

theorem contrapositive_equiv (h : a ∈ P → b ∉ P) : b ∈ P → a ∉ P :=
by
  sorry

end contrapositive_equiv_l314_314350


namespace percentage_tax_proof_l314_314565

theorem percentage_tax_proof (total_worth tax_free cost taxable tax_rate tax_value percentage_sales_tax : ℝ)
  (h1 : total_worth = 40)
  (h2 : tax_free = 34.7)
  (h3 : tax_rate = 0.06)
  (h4 : total_worth = taxable + tax_rate * taxable + tax_free)
  (h5 : tax_value = tax_rate * taxable)
  (h6 : percentage_sales_tax = (tax_value / total_worth) * 100) :
  percentage_sales_tax = 0.75 :=
by
  sorry

end percentage_tax_proof_l314_314565


namespace two_courtiers_have_same_selection_l314_314685

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l314_314685


namespace rectangle_area_increase_l314_314742

theorem rectangle_area_increase {l w : ℝ} : 
  let original_area := l * w in
  let new_area := (1.3 * l) * (1.2 * w) in
  ((new_area - original_area) / original_area) * 100 = 56 :=
by
  sorry

end rectangle_area_increase_l314_314742


namespace max_min_x1_x2_squared_l314_314501

theorem max_min_x1_x2_squared (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 - (k-2)*x1 + (k^2 + 3*k + 5) = 0)
  (h2 : x2^2 - (k-2)*x2 + (k^2 + 3*k + 5) = 0)
  (h3 : -4 ≤ k ∧ k ≤ -4/3) : 
  (∃ (k_max k_min : ℝ), 
    k = -4 → x1^2 + x2^2 = 18 ∧ k = -4/3 → x1^2 + x2^2 = 50/9) :=
sorry

end max_min_x1_x2_squared_l314_314501


namespace find_f_15_l314_314151

theorem find_f_15
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f x + f (2 * x + y) + 7 * x * y = f (3 * x - 2 * y) + 3 * x ^ 2 + 2) :
  f 15 = 1202 := 
sorry

end find_f_15_l314_314151


namespace angle_triple_complement_l314_314244

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314244


namespace part_a_part_b_part_c_l314_314386

-- Predicate for the transformations
inductive Transform (A B C D : Type) : Prop
| mk : (A = B ∨ C = D) → Transform

-- Definition for the polynomial P(n) in Lean
def P (n : ℕ) : ℕ := sorry

-- Part (a)
theorem part_a (n : ℕ) (h : n ≥ 3) : P(n) ≥ n - 3 :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n ≥ 3) : P(n) ≤ 2 * n - 7 :=
by sorry

-- Part (c)
theorem part_c (n : ℕ) (h : n ≥ 13) : P(n) ≤ 2 * n - 10 :=
by sorry

end part_a_part_b_part_c_l314_314386


namespace part1_solution_set_part2_range_a_l314_314946

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l314_314946


namespace geometric_sequence_sum_five_l314_314090

theorem geometric_sequence_sum_five (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 - a 0 = 3) 
  (h2 : a 3 - a 1 = 6) 
  (h3 : ∀ n, a (n+1) = a n * q) : 
  ∑ i in finset.range 5, a i = 31 := 
sorry

end geometric_sequence_sum_five_l314_314090


namespace involution_count_l314_314534

noncomputable def countInvolutions (n : ℕ) : ℕ :=
  1 + ∑ j in Finset.range (n/2 + 1), Nat.choose n (2 * j) * Nat.double_factorial (2 * j - 1)

theorem involution_count (X : Type) [Finite X] (f : X → X) (n : ℕ) (h : Fintype.card X = n) 
  (involution : ∀ x, f (f x) = x) :
  ∃ k, countInvolutions n = k :=
sorry

end involution_count_l314_314534


namespace problem_statement_l314_314455

def Q : ℚ :=
  (finset.sum (finset.Ico 1 2023) (λ k, (2023 - k) / k)) / 
  (finset.sum (finset.Ico 2 2024) (λ k, 1 / k))

theorem problem_statement : Q = 2023 := by sorry

end problem_statement_l314_314455


namespace triple_complement_angle_l314_314281

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314281


namespace player_jump_height_to_dunk_l314_314163

/-- Definitions given in the conditions -/
def rim_height : ℕ := 120
def player_height : ℕ := 72
def player_reach_above_head : ℕ := 22

/-- The statement to be proven -/
theorem player_jump_height_to_dunk :
  rim_height - (player_height + player_reach_above_head) = 26 :=
by
  sorry

end player_jump_height_to_dunk_l314_314163


namespace together_complete_work_in_3_days_l314_314783

-- Define the conditions
def work_rate (days : ℕ) : ℝ := 1 / days

def woman_rate : ℝ := work_rate 6
def boy_rate : ℝ := work_rate 18
def man_rate : ℝ := work_rate 9

-- Calculate the combined work rate
def combined_work_rate : ℝ := woman_rate + boy_rate + man_rate

-- State the theorem to be proven
theorem together_complete_work_in_3_days : combined_work_rate * 3 = 1 := by
  sorry

end together_complete_work_in_3_days_l314_314783


namespace max_geometric_mean_is_4_l314_314504

noncomputable def max_geometric_mean (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ (a + b) / 2 = 4 then
    max (real.sqrt (a * b)) sorry
  else
    sorry

theorem max_geometric_mean_is_4 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + b) / 2 = 4) : max_geometric_mean a b = 4 := 
sorry

end max_geometric_mean_is_4_l314_314504


namespace algebraic_expression_evaluation_l314_314123

noncomputable def algebraic_expression (x : ℝ) : ℝ :=
  (x / (x^2 + 2*x + 1) - 1 / (2*x + 2)) / ((x - 1) / (4*x + 4))

noncomputable def substitution_value : ℝ :=
  2 * Real.cos (Real.pi / 4) - 1

theorem algebraic_expression_evaluation :
  algebraic_expression substitution_value = Real.sqrt 2 := by
  sorry

end algebraic_expression_evaluation_l314_314123


namespace angle_triple_complement_l314_314320

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314320


namespace tangent_distance_l314_314368

noncomputable def origin : (ℝ × ℝ) := (0, 0)

def circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 2)^2 + (p.2 - 1)^2 = 1

def is_tangent (o p c : ℝ × ℝ) : Prop :=
  let |OP| := real.sqrt ((p.1 - o.1)^2 + (p.2 - o.2)^2)
  let |OC| := real.sqrt ((c.1 - o.1)^2 + (c.2 - o.2)^2)
  let |PC| := 1 -- Radius of the circle
  (OP^2 = OC^2 - PC^2)

theorem tangent_distance
  (o : (ℝ × ℝ))
  (h1 : o = (0, 0)) -- O is the origin
  (p : (ℝ × ℝ))
  (h2 : circle p) -- P is on the circle
  : is_tangent o p (2, 1) → real.sqrt ((p.1 - o.1)^2 + (p.2 - o.2)^2) = 2 := 
sorry

end tangent_distance_l314_314368


namespace isosceles_triangle_perimeter_l314_314726

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : a = c ∨ b = c) :
  a + b + c = 22 :=
by
  -- This part of the proof is simplified using the conditions
  sorry

end isosceles_triangle_perimeter_l314_314726


namespace angle_BEC_80_degrees_l314_314168

/-- The triangle ABC is equilateral. The ray BE intersects the segment AC at D, such that 
    ∠CBE = 20° and |DE| = |AB|. This statement proves that the measure of the angle ∠BEC is 80°. -/
theorem angle_BEC_80_degrees (A B C D E : Type)
  [triangle : Equilateral_ABC A B C]
  (intersects : Intersects_BE_AC B E A C D)
  (angle_CBE_20 : Angle_CBE_eq_20 B C E)
  (DE_eq_AB : length_DE_eq_length_AB D E A B) :
  Angle_BEC_eq_80 B E C := sorry

end angle_BEC_80_degrees_l314_314168


namespace apples_given_by_Anita_l314_314622

variable (original_apples : ℕ) (new_apples : ℕ) (given_apples : ℕ)
def original_apples := 4
def new_apples := 9

theorem apples_given_by_Anita : given_apples = new_apples - original_apples := by
  sorry

end apples_given_by_Anita_l314_314622


namespace part1_solution_set_part2_range_l314_314985

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l314_314985


namespace angle_triple_complement_l314_314249

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l314_314249


namespace vertex_on_x_axis_l314_314836

theorem vertex_on_x_axis (c : ℝ) : 
    let a := 3
    let b := 6
    let y := λ x, a * x^2 + b * x + c
    let h := -b / (2 * a)
    let k := y h
in k = 0 → c = 3 := 
by
  let a := 3
  let b := 6
  let y := λ x, a * x^2 + b * x + c
  let h := -b / (2 * a)
  let k := y h
  sorry

end vertex_on_x_axis_l314_314836


namespace polynomials_solution_l314_314829

theorem polynomials_solution :
  ∀ (f g : ℂ[x]),
  (∀ x : ℂ, f.eval (f.eval x) - g.eval (g.eval x) = 1 + complex.i) ∧
  (∀ x : ℂ, f.eval (g.eval x) - g.eval (f.eval x) = 1 - complex.i) →
  (f = λ x => complex.i * x + 1) ∧ (g = λ x => complex.i * x) :=
by
  sorry

end polynomials_solution_l314_314829


namespace min_value_expression_min_value_reachable_l314_314465

theorem min_value_expression (x y z : ℝ) : 
  (x^2 + x*y + y^2 + z^2) ≥ 0 :=
by
  calc
    x^2 + x*y + y^2 + z^2
        = (x + y/2)^2 + 3/4*y^2 + z^2 : by ring
    ... ≥ 0 : by apply add_nonneg (add_nonneg (sq_nonneg _) (mul_nonneg (by norm_num) (sq_nonneg _))) (sq_nonneg _)

theorem min_value_reachable :
  ∃ x y z : ℝ, x^2 + x*y + y^2 + z^2 = 0 :=
by
  use 0, 0, 0
  show 0^2 + 0*0 + 0^2 + 0^2 = 0
  norm_num

end min_value_expression_min_value_reachable_l314_314465


namespace missing_number_eq_24_l314_314371

theorem missing_number_eq_24 : ∃ x : ℤ, (720 - x) / 120 = 5.8 ∧ x = 24 :=
by 
  use 24
  split
  sorry
  rfl

end missing_number_eq_24_l314_314371


namespace at_least_one_not_less_than_two_l314_314034

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x, (x = a + 1/b ∨ x = b + 1/c ∨ x = c + 1/a) ∧ 2 ≤ x :=
by
  sorry

end at_least_one_not_less_than_two_l314_314034


namespace circles_internally_tangent_l314_314017

def distance (p q : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2))

def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d = r₁ - r₂

theorem circles_internally_tangent :
  let O₁ := (0, 8)
  let O₂ := (-6, 0)
  let r₁ := 12
  let r₂ := 2
  let d := distance O₁ O₂
  internally_tangent r₁ r₂ d :=
begin
  sorry
end

end circles_internally_tangent_l314_314017


namespace find_factor_l314_314784

theorem find_factor (x f : ℕ) (h1 : x = 15) (h2 : (2 * x + 5) * f = 105) : f = 3 :=
sorry

end find_factor_l314_314784


namespace area_increase_l314_314744

variable (l w : ℝ)

theorem area_increase (hl : l > 0) (hw : w > 0) :
  let A := l * w
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  (A_new - A) / A * 100 = 56 := by
  let A := l * w
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  have : A_new = 1.56 * A := by
    rw [←mul_assoc, ←mul_assoc, mul_comm 1.3 l, mul_assoc, mul_comm 1.2 w, mul_assoc, mul_comm w l, ←mul_assoc]
    ring
  calc
    (A_new - A) / A * 100 = ((1.56 * A) - A) / A * 100 := by rw this
    ... = (1.56 - 1) * 100 := by field_simp [A, lt_mul_iff_one_lt_left (lt_of_lt_of_le (zero_lt_one) hl), ne_of_gt hl]
    ... = 0.56 * 100 := by ring
    ... = 56 := by norm_num

end area_increase_l314_314744


namespace largest_number_with_unique_digits_summing_to_18_is_843210_l314_314329

-- Define the problem with conditions and answer
theorem largest_number_with_unique_digits_summing_to_18_is_843210 :
  ∃ n : ℕ, (∀ d ∈ (int_to_digits n), unique d) ∧ digit_sum n = 18 ∧ is_largest n :=
sorry

-- Definitions used in the theorem statement

-- Helper to convert an integer to a list of its digits
def int_to_digits (n : ℕ) : list ℕ := -- convert integer to list of digits
sorry

-- Helper to check that all elements in the list are unique
def unique (l : list ℕ) : Prop := -- check that all digits are unique
sorry

-- Helper to sum digits of a number
def digit_sum (n : ℕ) : ℕ := (int_to_digits n).sum
  
-- Helper to assert that the number is the largest possible
def is_largest (n : ℕ) : Prop := -- check if the number is the largest possible under given conditions
sorry

end largest_number_with_unique_digits_summing_to_18_is_843210_l314_314329


namespace courtier_selection_l314_314678

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l314_314678


namespace exist_two_courtiers_with_same_selection_l314_314694

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l314_314694


namespace angle_triple_complement_l314_314259

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314259


namespace initial_rate_of_commission_is_4_l314_314641

noncomputable def initial_commission_rate (B : ℝ) (x : ℝ) : Prop :=
  B * (x / 100) = 0.8 * B * (5 / 100)

theorem initial_rate_of_commission_is_4 (B : ℝ) (hB : B > 0) :
  initial_commission_rate B 4 :=
by
  unfold initial_commission_rate
  sorry

end initial_rate_of_commission_is_4_l314_314641


namespace range_of_a_l314_314540

theorem range_of_a :
  (∀ x : ℝ, abs (x - a) < 1 ↔ (1 / 2 < x ∧ x < 3 / 2)) → (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by sorry

end range_of_a_l314_314540


namespace assignment_plan_count_l314_314062

noncomputable def number_of_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let tasks := ["translation", "tour guide", "etiquette", "driver"]
  let v1 := ["Xiao Zhang", "Xiao Zhao"]
  let v2 := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Condition: Xiao Zhang and Xiao Zhao can only take positions for translation and tour guide
  -- Calculate the number of ways to assign based on the given conditions
  -- 36 is the total number of assignment plans
  36

theorem assignment_plan_count :
  number_of_assignment_plans = 36 :=
  sorry

end assignment_plan_count_l314_314062


namespace find_E_l314_314629

-- Define the value of a.
def a : ℕ := 30

-- Define the given expression involving a.
def expr := 3 * a - 8

-- Define the average condition.
def average (E : ℕ) : Prop := (E + expr) / 2 = 79

theorem find_E : ∃ E : ℕ, average E ∧ E = 76 := 
by
  existsi 76
  unfold average
  rw [expr, a, Nat.mul_comm 30 3, Nat.mul_comm 3 30, Nat.mul_assoc]
  norm_num
  rw [add_comm]
  norm_num
  exact rfl

end find_E_l314_314629


namespace courtiers_dog_selection_l314_314657

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l314_314657


namespace value_of_collection_l314_314410

theorem value_of_collection (n : ℕ) (v : ℕ → ℕ) (h1 : n = 20) 
    (h2 : v 5 = 20) (h3 : ∀ k1 k2, v k1 = v k2) : v n = 80 :=
by
  sorry

end value_of_collection_l314_314410


namespace polynomial_degree_rational_roots_l314_314788

theorem polynomial_degree_rational_roots :
  ∃ (p : Polynomial ℚ), 
    (∀ n : ℕ, 2 ≤ n ∧ n ≤ 1001 → (p.eval (n + Real.sqrt (n + 1)) = 0) ∧
    (p.eval (n - Real.sqrt (n + 1)) = 0)) ∧
    p.degree = 1969 :=
by
  sorry

end polynomial_degree_rational_roots_l314_314788


namespace parabola_directrix_l314_314635

noncomputable def directrix_value (a : ℝ) : ℝ := -1 / (4 * a)

theorem parabola_directrix (a : ℝ) (h : directrix_value a = 2) : a = -1 / 8 :=
by
  sorry

end parabola_directrix_l314_314635


namespace sum_of_fractions_eq_one_l314_314814

variable {a b c d : ℝ} (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)
          (h_equiv : (a * d + b * c) / (b * d) = (a * c) / (b * d))

theorem sum_of_fractions_eq_one : b / a + d / c = 1 :=
by sorry

end sum_of_fractions_eq_one_l314_314814


namespace product_repeating_decimal_l314_314819

noncomputable def t : ℚ := 456 / 999

theorem product_repeating_decimal (t: ℚ) (h: t = 456 / 999) : (8 * t) = 1216 / 333 := by
  have h1 : 8 * (456 / 999) = 3648 / 999 := by simp [mul_div]
  have h2 : 3648 / 999 = 1216 / 333 := by norm_num
  rw [h, h1, h2]
  sorry

end product_repeating_decimal_l314_314819


namespace find_f_of_3_l314_314872

theorem find_f_of_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f_of_3_l314_314872


namespace jamie_coin_count_l314_314078

-- Definitions of the problem conditions
def num_coins_each_type (x : ℕ) : Prop :=
  let penny_val := 0.01 * x
  let nickel_val := 0.05 * x
  let dime_val := 0.10 * x
  let quarter_val := 0.25 * x
  let total_val := penny_val + nickel_val + dime_val + quarter_val
  total_val = 20

-- The mathematical statement to be proved
theorem jamie_coin_count : ∃ x : ℕ, num_coins_each_type x ∧ x = 50 :=
by
  sorry

end jamie_coin_count_l314_314078


namespace cos_C_of_right_triangle_l314_314050

theorem cos_C_of_right_triangle (ABC : Triangle) (hA : ABC.∠A = 90) (h_tanC : Real.tan (ABC.∠C) = 4) :
  Real.cos (ABC.∠C) = Real.sqrt 17 / 17 := by
  sorry

end cos_C_of_right_triangle_l314_314050


namespace ratio_volumes_l314_314713

/-- The weights per liter of vegetable ghee of brands 'a' and 'b'. -/
def weight_per_liter_a := 900
def weight_per_liter_b := 850

/-- The total volume and total weight of the mixture. -/
def total_volume := 4
def total_weight := 3520

/-- Represents the volumes of brand 'a' and brand 'b'. -/
variables (V_a V_b : ℝ)

/-- The equations representing the problem's conditions. -/
def eq1 := V_a + V_b = total_volume
def eq2 := weight_per_liter_a * V_a + weight_per_liter_b * V_b = total_weight

theorem ratio_volumes (h1 : eq1) (h2 : eq2) : V_a / V_b = 3 / 2 :=
sorry

end ratio_volumes_l314_314713


namespace lcm_24_36_45_l314_314340

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l314_314340


namespace part1_solution_set_part2_range_of_a_l314_314926

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314926


namespace movie_attendance_l314_314768

theorem movie_attendance (total_seats : ℕ) (empty_seats : ℕ) (h1 : total_seats = 750) (h2 : empty_seats = 218) :
  total_seats - empty_seats = 532 := by
  sorry

end movie_attendance_l314_314768


namespace derivative_at_zero_l314_314909

-- Define the function f
def f (x : ℝ) : ℝ := Real.exp (2 * x + 1) - 3 * x

-- State that f'(0) = 2 * e - 3
theorem derivative_at_zero : Deriv f 0 = 2 * Real.exp 1 - 3 := by
  sorry

end derivative_at_zero_l314_314909


namespace total_grading_time_l314_314404

theorem total_grading_time:
  let math_worksheets := 45 in
  let science_worksheets := 37 in
  let history_worksheets := 32 in
  let time_per_math := 15 in
  let time_per_science := 20 in
  let time_per_history := 25 in
  let total_time_math := math_worksheets * time_per_math in
  let total_time_science := science_worksheets * time_per_science in
  let total_time_history := history_worksheets * time_per_history in
  total_time_math + total_time_science + total_time_history = 2215 :=
by 
  sorry

end total_grading_time_l314_314404


namespace median_of_data_is_4_l314_314545

def data : List Int := [5, 4, 3, 5, 5, 2, 5, 3, 4, 1]

theorem median_of_data_is_4 :
  median data = 4 := sorry

end median_of_data_is_4_l314_314545


namespace circle_radius_3_l314_314162

theorem circle_radius_3 :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 2 * y - 7 = 0) → (∃ r : ℝ, r = 3) :=
by
  sorry

end circle_radius_3_l314_314162


namespace solve_congruence_l314_314125

theorem solve_congruence :
  ∃ n : ℤ, 0 ≤ n ∧ n < 47 ∧ 13 * n ≡ 8 [MOD 47] :=
by
  use 21
  split
  . exact dec_trivial
  split
  . exact dec_trivial
  . norm_num
  sorry

end solve_congruence_l314_314125


namespace angle_measure_triple_complement_l314_314238

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l314_314238


namespace white_tulips_multiple_of_seven_l314_314719

/-- Let R be the number of red tulips, which is given as 91. 
    We also know that the greatest number of identical bouquets that can be made without 
    leaving any flowers out is 7.
    Prove that the number of white tulips W is a multiple of 7. -/
theorem white_tulips_multiple_of_seven (R : ℕ) (g : ℕ) (W : ℕ) (hR : R = 91) (hg : g = 7) :
  ∃ w : ℕ, W = 7 * w :=
by
  sorry

end white_tulips_multiple_of_seven_l314_314719


namespace two_courtiers_have_same_selection_l314_314686

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l314_314686


namespace monotonic_interval_when_a_eq_1_range_of_a_for_f_above_g_l314_314911

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x - a) / x
noncomputable def g (x a : ℝ) : ℝ := a * Real.log x + a

theorem monotonic_interval_when_a_eq_1 :
  (∀ x : ℝ, 0 < x → F(x, 1) = (f x 1) - (g x 1)) →
  (∀ x : ℝ, 1 ≤ x → 0 ≤ derivative (fun x => (f x 1) - (g x 1)) x) ∧
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → derivative (fun x => (f x 1) - (g x 1)) x ≤ 0) :=
by sorry

theorem range_of_a_for_f_above_g :
  (∀ x : ℝ, 1 < x → f x a > g x a) →
  a ≤ (1 / 2) * Real.exp 1 :=
by sorry

end monotonic_interval_when_a_eq_1_range_of_a_for_f_above_g_l314_314911


namespace polynomial_real_roots_probability_l314_314791

theorem polynomial_real_roots_probability : 
  ∃ (m n : ℕ), Nat.coprime m n ∧ 
    (∀ a ∈ Set.Icc (-20 : ℝ) (18 : ℝ), 
       (∀ (p : ℝ[X]), 
          p = X^4 + 2 * (Polynomial.C a) * X^3 + (2 * (Polynomial.C a) - Polynomial.C 2) * X^2 +
              ((-4) * (Polynomial.C a) + Polynomial.C 3) * X - Polynomial.C 2 →
            (∀ (x : ℝ), IsRoot p x → IsReal x))) → 
    ((m : ℚ) / (n : ℚ) = 18 / 19) ∧ m + n = 37 := 
by 
  sorry

end polynomial_real_roots_probability_l314_314791


namespace same_selection_exists_l314_314644

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l314_314644


namespace proof_f_2009_l314_314442

-- Define f as a function from ℝ to ℝ
def f : ℝ → ℝ

-- Conditions
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f(x) = f(-x)
def odd_function_shifted (f : ℝ → ℝ) := ∀ x : ℝ, f(-x-1) = -f(x-1)

-- The actual Lean 4 statement
theorem proof_f_2009 (h1 : even_function f) (h2 : odd_function_shifted f) : f 2009 = 0 := 
by 
  sorry

end proof_f_2009_l314_314442


namespace average_age_is_27_l314_314144

variables (a b c : ℕ)

def average_age_of_a_and_c (a c : ℕ) := (a + c) / 2

def age_of_b := 23

def average_age_of_a_b_and_c (a b c : ℕ) := (a + b + c) / 3

theorem average_age_is_27 (h1 : average_age_of_a_and_c a c = 29) (h2 : b = age_of_b) :
  average_age_of_a_b_and_c a b c = 27 := by
  sorry

end average_age_is_27_l314_314144


namespace expand_expression_l314_314456

variable (x y z : ℕ)

theorem expand_expression (x y z: ℕ) : 
  (x + 10) * (3 * y + 5 * z + 15) = 3 * x * y + 5 * x * z + 15 * x + 30 * y + 50 * z + 150 :=
by
  sorry

end expand_expression_l314_314456


namespace gcd_lcm_problem_part1_gcd_lcm_problem_part2_l314_314853

open Int

noncomputable def a1 := 5^2 * 7^4
noncomputable def a2 := 490 * 175

noncomputable def b1 := 2^5 * 3 * 7
noncomputable def b2 := 3^4 * 5^4 * 7^2
noncomputable def b3 := 10000

theorem gcd_lcm_problem_part1 : 
  gcd a1 a2 = 8575 ∧ Nat.lcm a1 a2 = 600250 := 
by
  sorry

theorem gcd_lcm_problem_part2 : 
  gcd (gcd b1 b2) b3 = 1 ∧ Nat.lcm b1 (Nat.lcm b2 b3) = 793881600 := 
by
  sorry

end gcd_lcm_problem_part1_gcd_lcm_problem_part2_l314_314853


namespace train_pass_time_l314_314405

/--
Given:
- The train is 850 meters long.
- The train's speed is 90 kmph.
- The man's speed is 10 kmph.
- They are moving in the same direction.

Prove:
- The time it takes for the train to pass the man is 38.25 seconds.
-/
theorem train_pass_time 
  (L : ℝ) (S_train S_man : ℝ)
  (L_eq : L = 850)
  (S_train_eq : S_train = 90)
  (S_man_eq : S_man = 10)
  (same_direction : True) : (850 / ((90 - 10) * 1000 / 3600)) = 38.25 :=
by
  rw [L_eq, S_train_eq, S_man_eq]
  sorry

end train_pass_time_l314_314405


namespace geom_seq_min_value_l314_314487

theorem geom_seq_min_value (a : ℕ → ℝ) (m n : ℕ) (a_pos : ∀ n, 0 < a n)
  (h1 : a 7 = a 6 + 2 * a 5)
  (h2 : sqrt (a m * a n) = 4 * a 1) :
  m + n = 6 → (1 / m + 4 / n) = 3 / 2 := 
sorry

end geom_seq_min_value_l314_314487


namespace angle_measure_triple_complement_l314_314307

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314307


namespace find_ellipse_equation_no_real_k_for_bisector_l314_314891

noncomputable def equation_of_ellipse := "x^2 / 8 + y^2 / 4 = 1"

def point_F : ℝ × ℝ := (2, 0)
def center_of_symmetry : ℝ × ℝ := (0, 0)
def major_axis_length : ℝ := 4 * Real.sqrt 2

def line_l (k m : ℝ) (x : ℝ) : ℝ := k * x + m

theorem find_ellipse_equation:
  ellipse_eq (x y : ℝ) : ((x^2 / 8) + (y^2 / 4) = 1) := sorry

theorem no_real_k_for_bisector :
  ∀ (k m : ℝ), k ≠ 0 → ¬∃ k : ℝ, ∃ m : ℝ,
  let x1 := sorry in let x2 := sorry in let y1 := line_l k m x1 in let y2 := line_l k m x2 in
  let N := ((x1 + x2) / 2, (y1 + y2) / 2) in
  let perp_bisector_slope := -1 / k in
  let Q := (0, 3) in
  perp_bisector_slope = (N.2 - Q.2) / (N.1 - Q.1) := sorry

end find_ellipse_equation_no_real_k_for_bisector_l314_314891


namespace number_of_geometric_sequences_in_set_l314_314097

theorem number_of_geometric_sequences_in_set :
  let S := {1, 2, 3, ..., 500}
  in ∃ n : ℕ, n = 94 ∧
    ∀ (a q : ℕ), a > 0 ∧ q > 1 ∧ a + q + q^2 + q^3 ≤ 500 → 
    (∃ seq : list ℕ, list.nodup seq ∧ seq = [a, a*q, a*q^2, a*q^3]) :=
by sorry

end number_of_geometric_sequences_in_set_l314_314097


namespace pyramid_volume_l314_314398

noncomputable def volume_of_pyramid (AB BC PA PB: ℝ) (h_AB: AB = 10) 
(h_BC: BC = 5) (h_PA_perp_AD: PA ⊥ AD) (h_PA_perp_AB: PA ⊥ AB) 
(h_PB: PB = 20) : ℝ :=
  let base_area := AB * BC
  let height := PA
  (1 / 3) * base_area * height

theorem pyramid_volume (AB BC PB: ℝ) 
(h_AB: AB = 10) (h_BC: BC = 5) (h_PA_perp_AD: ∀ x, PA x → AD x → false) 
(h_PA_perp_AB: ∀ x, PA x → AB x → false) (h_PB: PB = 20) 
: volume_of_pyramid AB BC (√(PB^2 - AB^2)) PB = (500 * √3) / 3 :=
by
  sorry

end pyramid_volume_l314_314398


namespace part1_solution_set_part2_range_of_a_l314_314921

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314921


namespace find_a_l314_314897

noncomputable def isPointOnEllipse (P : Point) (a b : ℝ) : Prop :=
  let ⟨x, y⟩ := P
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def areaTriangle (P1 P2 P3 : Point) : ℝ :=
  let ⟨x1, y1⟩ := P1
  let ⟨x2, y2⟩ := P2
  let ⟨x3, y3⟩ := P3
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

noncomputable def tanAngle (P1 P2 P3 : Point) : ℝ := -- Note: This is a simplified abstract definition for tan of the angle 
  let ⟨x1, y1⟩ := P1
  let ⟨x2, y2⟩ := P2
  let ⟨x3, y3⟩ := P3
  (y3 - y1) / (x3 - x1)

theorem find_a 
  (a b : ℝ) (P F1 F2 : Point) 
  (hp : isPointOnEllipse P a b) 
  (a_pos : a > 0)
  (b_pos : b > 0)
  (condition1 : areaTriangle P F1 F2 = 1) 
  (condition2 : tanAngle P F1 F2 = 1/2) 
  (condition3 : tanAngle P F2 F1 = -2) 
  : a = (Real.sqrt 15) / 2 :=
sorry

end find_a_l314_314897


namespace angle_triple_complement_l314_314195

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l314_314195


namespace original_flow_rate_l314_314801

theorem original_flow_rate (x : ℝ) (h : 2 = 0.6 * x - 1) : x = 5 :=
by
  sorry

end original_flow_rate_l314_314801


namespace jacket_final_price_l314_314813

theorem jacket_final_price 
  (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) (final_discount : ℝ)
  (price_after_first : ℝ := original_price * (1 - first_discount))
  (price_after_second : ℝ := price_after_first * (1 - second_discount))
  (final_price : ℝ := price_after_second * (1 - final_discount)) :
  original_price = 250 ∧ first_discount = 0.4 ∧ second_discount = 0.3 ∧ final_discount = 0.1 →
  final_price = 94.5 := 
by 
  sorry

end jacket_final_price_l314_314813


namespace circle_equation_l314_314634

theorem circle_equation (a : ℝ) (h1 : ∀ t : ℝ, center = (0, t)) (h2 : radius = 1) (h3 : passes_through (1, 3)) :
  x^2 + (y - 3)^2 = 1 :=
by
  sorry

end circle_equation_l314_314634


namespace greatest_prime_saturated_two_digit_integer_l314_314751

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ n : ℕ, n > 1 ∧ n < p → p % n ≠ 0

def prime_factors (r : ℕ) : List ℕ :=
  List.filter is_prime (List.range (r + 1)).filter (λ n, n > 0 ∧ r % n == 0)

def product (lst : List ℕ) : ℕ :=
  lst.foldr (λ x acc, x * acc) 1

def prime_saturated (r : ℕ) : Prop :=
  product (prime_factors r) < Nat.sqrt r

theorem greatest_prime_saturated_two_digit_integer : ∃ r, r < 100 ∧ 10 ≤ r ∧ prime_saturated r ∧ ∀ r', r' < 100 ∧ 10 ≤ r' ∧ prime_saturated r' → r' ≤ 98 :=
begin
  use 98,
  split, { exact dec_trivial },
  split, { exact dec_trivial },
  split,
  { sorry },
  { intros r' hr_condition,
    sorry }
end

end greatest_prime_saturated_two_digit_integer_l314_314751


namespace max_value_7a_9b_l314_314589

theorem max_value_7a_9b 
    (r_1 r_2 r_3 a b : ℝ) 
    (h_eq : ∀ x, x^3 - x^2 + a * x - b = 0 → (x = r_1 ∨ x = r_2 ∨ x = r_3))
    (h_root_sum : r_1 + r_2 + r_3 = 1)
    (h_root_prod : r_1 * r_2 * r_3 = b)
    (h_root_sumprod : r_1 * r_2 + r_2 * r_3 + r_3 * r_1 = a)
    (h_bounds : ∀ i, i = r_1 ∨ i = r_2 ∨ i = r_3 → 0 < i ∧ i < 1) :
        7 * a - 9 * b ≤ 2 := 
sorry

end max_value_7a_9b_l314_314589


namespace equal_sums_in_5x5_grid_l314_314763

theorem equal_sums_in_5x5_grid :
  ∀ (grid : Fin 5 × Fin 5 → {n : ℕ // n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7}),
  ∃ (i j : Fin 5) (d1 d2 : Fin 10), i ≠ j ∧
  (let row := ∑ k, grid ((i : Fin 5), k),
       col := ∑ k, grid (k, (i : Fin 5)),
       diag1 := ∑ x⋈(x: Fin 5), grid (x, x),  -- main diagonal
       diag2 := ∑ x⋈(x: Fin 5), grid (x, 4 - x)  -- anti diagonal
  in
    row + col + diag1 + diag2) - 
  col - row - diag1 - diag2 = 0 :=
by
  sorry

end equal_sums_in_5x5_grid_l314_314763


namespace part1_solution_part2_solution_l314_314964

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l314_314964


namespace paving_stones_required_l314_314021

-- Define the dimensions
def courtyard_length : ℝ := 75
def courtyard_width : ℝ := 20.75
def stone_length : ℝ := 3.25
def stone_width : ℝ := 2.5

-- Calculate areas
def area_courtyard : ℝ := courtyard_length * courtyard_width
def area_stone : ℝ := stone_length * stone_width

-- Calculate number of paving stones, rounding up
def number_of_stones := ⌈area_courtyard / area_stone⌉.toNat

theorem paving_stones_required : number_of_stones = 192 := by
  unfold number_of_stones
  unfold area_courtyard
  unfold area_stone
  sorry

end paving_stones_required_l314_314021


namespace f_periodic_odd_l314_314880

theorem f_periodic_odd {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (2 + x) = -f (2 - x)) : 
  f 2012 = 0 :=
begin
  sorry
end

end f_periodic_odd_l314_314880


namespace exist_two_courtiers_with_same_selection_l314_314693

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l314_314693


namespace combined_tennis_percentage_is_31_l314_314812

-- Define the initial conditions
def centralHighStudents : ℕ := 1800
def centralHighTennisPercentage : ℝ := 0.25
def northAcademyStudents : ℕ := 3000
def northAcademyTennisPercentage : ℝ := 0.35

-- Define the number of students who prefer tennis at each school
def centralHighTennisStudents : ℕ := (centralHighStudents * (centralHighTennisPercentage)).toNat
def northAcademyTennisStudents : ℕ := (northAcademyStudents * (northAcademyTennisPercentage)).toNat

-- Define the total number of students and tennis students across both schools
def totalStudents : ℕ := centralHighStudents + northAcademyStudents
def totalTennisStudents : ℕ := centralHighTennisStudents + northAcademyTennisStudents

-- Define the combined tennis percentage
def combinedTennisPercentage : ℝ := (totalTennisStudents.toReal / totalStudents.toReal) * 100

theorem combined_tennis_percentage_is_31 :
  combinedTennisPercentage ≈ 31 :=
by
  sorry

end combined_tennis_percentage_is_31_l314_314812


namespace max_omega_value_l314_314912

-- Definitions based on conditions
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- Conditions
variable (ω : ℝ) (φ : ℝ)
variable h_omega_pos : ω > 0
variable h_phi_pos : 0 < φ
variable h_phi_lt_pi_div_2 : φ < Real.pi / 2
variable h_f_neg_pi_div_4_eq_0 : f (-Real.pi / 4) ω φ = 0
variable h_f_symmetry : ∀ x, f (Real.pi / 4 - x) ω φ = f (Real.pi / 4 + x) ω φ
variable h_f_monotonic : MonotoneOn (f · ω φ) (Set.Icc (Real.pi / 18) (2 * Real.pi / 9))

-- Theorem statement
theorem max_omega_value : ω ≤ 5 := sorry

end max_omega_value_l314_314912


namespace angle_triple_complement_l314_314321

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l314_314321


namespace angle_triple_complement_l314_314290

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l314_314290


namespace angle_measure_triple_complement_l314_314310

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l314_314310


namespace purely_imaginary_iff_m_eq_1_l314_314906

theorem purely_imaginary_iff_m_eq_1 (m : ℝ) :
  (m^2 - 1 = 0 ∧ m + 1 ≠ 0) → m = 1 :=
by
  sorry

end purely_imaginary_iff_m_eq_1_l314_314906


namespace domain_of_g_l314_314732

def g (x : ℝ) : ℝ := 1 / ((x - 2)^2 + (x + 3)^2 + 1)

theorem domain_of_g : ∀ x : ℝ, ((x - 2)^2 + (x + 3)^2 + 1) ≠ 0 :=
by {
  sorry,
}

end domain_of_g_l314_314732


namespace courtier_selection_l314_314679

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l314_314679


namespace math_problem_l314_314003

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^3 + x^2 + b * x
noncomputable def f_prime (a b : ℝ) (x : ℝ) := 3 * a * x^2 + 2 * x + b
noncomputable def g (a b : ℝ) (x : ℝ) := f a b x + f_prime a b x

theorem math_problem 
  (a b : ℝ)
  (h : g a b = λ x => - g a b (-x)) :
  (f a b = λ x => -1/3 * x^3 + x^2) ∧ 
  (∃ (xmax xmin : ℝ) (h₁: 1 ≤ xmax ∧ xmax ≤ 2) (h₂: 1 ≤ xmin ∧ xmin ≤ 2), 
    g (-1/3) 0 xmax = 5/3 ∧ g (-1/3) 0 xmin = 4/3) :=
begin
  sorry
end

end math_problem_l314_314003


namespace group_partition_count_l314_314061

theorem group_partition_count :
  (∃ (men women : ℕ), men = 4 ∧ women = 3 ∧ 
  (number_of_ways men women = 54)) :=
sorry

def number_of_ways (men women : ℕ) : ℕ :=
  if men < 2 ∨ women < 2 then 0
  else 
    let choose2men := 4.choose 2 in
    let choose2women := 3.choose 2 in
    let remaining := (4 - 2) + (3 - 2) in
    let choose1remaining := remaining.choose 1 in
    choose2men * choose2women * choose1remaining 

end group_partition_count_l314_314061


namespace round_sum_to_nearest_hundredth_l314_314411

theorem round_sum_to_nearest_hundredth:
  Float.round (74.6893 + 23.152) 0.01 = 97.84 :=
by
  sorry

end round_sum_to_nearest_hundredth_l314_314411


namespace angle_triple_complement_l314_314253

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l314_314253


namespace part1_solution_set_part2_range_of_a_l314_314924

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l314_314924


namespace gcd_8164_2937_l314_314325

/-- Define the two integers a and b -/
def a : ℕ := 8164
def b : ℕ := 2937

/-- Prove that the greatest common divisor of a and b is 1 -/
theorem gcd_8164_2937 : Nat.gcd a b = 1 :=
  by
  sorry

end gcd_8164_2937_l314_314325


namespace root_neg_conjugate_l314_314128

noncomputable theory
open Complex

variables (c₀ c₁ c₂ c₃ c₄ a b : ℝ) (z : ℂ)

def polynomial (z : ℂ) : ℂ := c₄ * z^4 + I * c₃ * z^3 + c₂ * z^2 + I * c₁ * z + c₀

theorem root_neg_conjugate:
  (polynomial c₀ c₁ c₂ c₃ c₄ (a + b * I) = 0) →
  (polynomial c₀ c₁ c₂ c₃ c₄ (-a + b * I) = 0) := 
sorry

end root_neg_conjugate_l314_314128


namespace part1_part2_l314_314994

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l314_314994


namespace barrel_capacities_l314_314113

-- Definitions of the barrels' capacities
variable (x : ℕ) -- capacity of the first barrel
variable (b2 : ℕ) -- capacity of the second barrel
variable (b3 : ℕ) -- capacity of the third barrel

-- Conditions based on the problem
#1 condition as water transfer from the first to second barrel
def second_barrel_capacity := b2 = (3/4 : ℝ) * x

#2 condition of water transfer from the second to the third barrel
def third_barrel_capacity := b3 = (7/12 : ℝ) * x

-- Main condition of final re-transfer to the first barrel
def final_transfer_equation := (7/12 : ℝ) * x + 50 = x

-- Theorems based on the problem to be proved in Lean
theorem barrel_capacities : 
  final_transfer_equation x ∧ 
  second_barrel_capacity x b2 ∧
  third_barrel_capacity x b3 → 
  x = 120 ∧ b2 = 90 ∧ b3 = 70 := by
  sorry

end barrel_capacities_l314_314113


namespace triple_complement_angle_l314_314286

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l314_314286


namespace no_such_positive_sequence_exists_l314_314451

theorem no_such_positive_sequence_exists :
  ¬ ∃ (a : Fin 2002 → ℝ), 
    (∀ i, 0 < a i) ∧
    (∀ k, 1 ≤ k ∧ k ≤ 2002 → 
      (∀ z : ℂ, is_root (∑ i in Finset.range 2002, a ⟨(k + 2001 - i : ℕ) % 2002, sorry⟩ * z ^ (2001 - i)) z → 
      |z.im| ≤ |z.re|)) :=
begin
  sorry, 
end

end no_such_positive_sequence_exists_l314_314451


namespace overlapping_squares_area_l314_314721

theorem overlapping_squares_area :
  let s : ℝ := 5
  let total_area := 3 * s^2
  let redundant_area := s^2 / 8 * 4
  total_area - redundant_area = 62.5 := by
  sorry

end overlapping_squares_area_l314_314721
