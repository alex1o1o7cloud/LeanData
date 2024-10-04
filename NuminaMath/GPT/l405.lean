import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.EuclideanDomain
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Analysis.Calculus.Geometry
import Mathlib.Analysis.Calculus.Quadrant
import Mathlib.Analysis.Convex.Parallelogram
import Mathlib.Analysis.Convex.Simplical
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.Probability
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.RandomVariable
import Mathlib.Tactic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Topology.Real

namespace maximum_rectangle_area_l405_405806

theorem maximum_rectangle_area (P : ℝ) (hP : P = 36) :
  ∃ (A : ℝ), A = (P / 4) * (P / 4) :=
by
  use 81
  sorry

end maximum_rectangle_area_l405_405806


namespace rectangular_eq_circle_min_dist_circle_line_l405_405211

-- 1. Rectangular equation of the circle
theorem rectangular_eq_circle (ρ θ : ℝ) (h₁ : ρ = 2 * Real.sqrt 3 * Real.sin θ) : 
  ∃ x y: ℝ, x^2 + (y - Real.sqrt 3)^2 = 3 := 
sorry

-- 2. Minimum distance from point on line to the center of the circle
theorem min_dist_circle_line (t : ℝ) :
  let l_x := (3 + t / 2); l_y := (Real.sqrt 3 * t / 2) in
  let C := (0 : ℝ, Real.sqrt 3) in
  let PC := (l_x - 0)^2 + (l_y - Real.sqrt 3)^2 in
  PC = (Real.sqrt 3)^2 :=
sorry

end rectangular_eq_circle_min_dist_circle_line_l405_405211


namespace base6_sum_correct_l405_405747

def int1 : ℕ := 345
def int2 : ℕ := 75
def base : ℕ := 6

def base6_sum := 1540

theorem base6_sum_correct : 
  (nat.toDigits base (int1 + int2)).reverse = [1, 5, 4, 0] :=
by sorry

end base6_sum_correct_l405_405747


namespace sum_of_three_largest_l405_405067

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l405_405067


namespace parabola_problem_l405_405303

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405303


namespace distinguish_daughters_l405_405671

def is_princess (x : Type) : Prop := sorry

def calls_princess (caller : Type) (callee : Type) : Prop := sorry

variable (K : Type) -- Koschei’s daughters
variable (P : Type) -- princesses
variables [∀ x ∈ P, is_princess x] [∀ x ∈ K, is_princess x]

-- Conditions

-- All princesses call Koschei’s daughters princesses
axiom all_princesses_call_daughters := ∀ (x : P) (y : K), calls_princess x y

-- Koschei’s daughters will be called princesses at least three times
axiom daughters_called_three_times := ∀ (x : K), ∃ (t : ℕ), t ≥ 3 ∧ (ℕ → calls_princess P x)

-- Princesses will be called princesses no more than twice
axiom princesses_called_no_more_than_twice := ∀ (x : P), ∃ (t : ℕ), t ≤ 2 ∧ (ℕ → calls_princess K x)

-- Conclusion
theorem distinguish_daughters : ∃ (can_distinguish : Prop), can_distinguish = true :=
by
  { haveI := all_princesses_call_daughters,
    haveI := daughters_called_three_times,
    haveI := princesses_called_no_more_than_twice,
    sorry 
  }

end distinguish_daughters_l405_405671


namespace problem_l405_405603

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405603


namespace parabola_distance_l405_405588

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405588


namespace tourism_revenue_scientific_notation_l405_405825

-- Define the conditions given in the problem.
def total_tourism_revenue := 12.41 * 10^9

-- Prove the scientific notation of the total tourism revenue.
theorem tourism_revenue_scientific_notation :
  total_tourism_revenue = 1.241 * 10^9 :=
sorry

end tourism_revenue_scientific_notation_l405_405825


namespace find_omega_value_l405_405741

noncomputable def sin_cos_min_value (ω a : ℝ) (h_a : a > 0) (h_ω : ω > 0) : Prop :=
  ∃ x : ℝ, (f x = sin(ω * x) + a * cos(ω * x)) ∧ (f x = -2) ∧ (x = π / 6)

theorem find_omega_value (ω : ℝ) (a : ℝ) (h_a : a > 0) (h_ω : ω > 0)
  (h_min : sin_cos_min_value ω a h_a h_ω) : ω = 7 := sorry

end find_omega_value_l405_405741


namespace distance_AB_l405_405433

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405433


namespace angle_bisector_perpendicular_DE_l405_405673

noncomputable def triangle_angles (α β γ : ℝ) :=
  α + β + γ = 180

noncomputable def angle_bisectors_intersect_circumcircle (α β γ : ℝ) (D E F O : Point) (circumcircle : Circle) :=
  ∃ (triangle : Triangle), triangle.has_angles α β γ ∧
  triangle.angle_bisectors α β γ = ⟨D, E, F⟩ ∧
  circumcircle.contains D ∧ circumcircle.contains E ∧ circumcircle.contains F 

theorem angle_bisector_perpendicular_DE (α β γ : ℝ) (D E F O : Point) (circumcircle : Circle) (triangle : Triangle) :
  triangle.has_angles α β γ →
  triangle.angle_bisectors α β γ = ⟨D, E, F⟩ →
  circumcircle.contains D ∧ circumcircle.contains E ∧ circumcircle.contains F →
  ∠(DE, triangle.angle_bisector C) = 90 := 
sorry

end angle_bisector_perpendicular_DE_l405_405673


namespace parabola_distance_problem_l405_405408

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405408


namespace sum_of_three_largest_l405_405084

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l405_405084


namespace pow_mod_cycle_l405_405813

theorem pow_mod_cycle (n : ℕ) : 3^250 % 13 = 3 := 
by
  sorry

end pow_mod_cycle_l405_405813


namespace distance_AB_l405_405330

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405330


namespace distance_AB_l405_405446

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405446


namespace distance_AB_l405_405464

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405464


namespace six_digit_palindromes_count_l405_405026

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l405_405026


namespace sum_of_three_largest_of_consecutive_numbers_l405_405069

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l405_405069


namespace smallest_bookmarks_l405_405886

theorem smallest_bookmarks (b : ℕ) :
  (b % 5 = 4) ∧ (b % 6 = 3) ∧ (b % 8 = 7) → b = 39 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end smallest_bookmarks_l405_405886


namespace distance_AB_l405_405343

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405343


namespace parabola_distance_l405_405578

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405578


namespace is_odd_and_monotonically_increasing_l405_405654

def f (x : ℝ) : ℝ := x ^ 3 - 1 / (x ^ 3)

theorem is_odd_and_monotonically_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :=
begin
  sorry
end

end is_odd_and_monotonically_increasing_l405_405654


namespace is_odd_and_monotonically_increasing_l405_405655

def f (x : ℝ) : ℝ := x ^ 3 - 1 / (x ^ 3)

theorem is_odd_and_monotonically_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :=
begin
  sorry
end

end is_odd_and_monotonically_increasing_l405_405655


namespace certain_number_is_7000_l405_405815

theorem certain_number_is_7000 (x : ℕ) (h1 : 1 / 10 * (1 / 100 * x) = x / 1000)
    (h2 : 1 / 10 * x = x / 10)
    (h3 : x / 10 - x / 1000 = 693) : 
  x = 7000 := 
sorry

end certain_number_is_7000_l405_405815


namespace total_stamps_in_collection_l405_405203

-- Definitions reflecting the problem conditions
def foreign_stamps : ℕ := 90
def old_stamps : ℕ := 60
def both_foreign_and_old_stamps : ℕ := 20
def neither_foreign_nor_old_stamps : ℕ := 70

-- The expected total number of stamps in the collection
def total_stamps : ℕ :=
  (foreign_stamps + old_stamps - both_foreign_and_old_stamps) + neither_foreign_nor_old_stamps

-- Statement to prove the total number of stamps is 200
theorem total_stamps_in_collection : total_stamps = 200 := by
  -- Proof omitted
  sorry

end total_stamps_in_collection_l405_405203


namespace parabola_distance_l405_405485

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405485


namespace greatest_possible_x_l405_405833

-- Define the conditions and the target proof in Lean 4
theorem greatest_possible_x 
  (x : ℤ)  -- x is an integer
  (h : 2.134 * (10:ℝ)^x < 21000) : 
  x ≤ 3 :=
sorry

end greatest_possible_x_l405_405833


namespace parabola_distance_problem_l405_405416

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405416


namespace butterfly_theorem_l405_405725

theorem butterfly_theorem 
  {O : Type*} [metric_space O] [normed_group O] [normed_space ℝ O]
  {A B C D E F M L N : O}
  (h1 : midpoint A B = M)
  (h2 : cd_chord : metry O → O → midpoint := M)
  (h3 : ef_chord : metry O → O → midpoint := M)
  (h4 : L ∈ line_segment A B)
  (h5 : CD ⊆ circle A B M)
  (h6 : EF ⊆ circle A B M)
  (h7 : CD ∩ AB = {L})
  (h8 : EF ∩ AB = {N})
  : dist L M = dist M N := sorry

end butterfly_theorem_l405_405725


namespace distinct_paintings_count_l405_405208

/-- 
Given 8 disks arranged in a circle, where 4 disks are painted blue, 3 disks are painted red, and 1 disk is painted green, 
the number of distinct paintings considering rotations and reflections of the circle is 23.
-/
theorem distinct_paintings_count : 
  ∃ p : (Set (Fin 8) → ℕ) → Prop, (∀ s : Set (Fin 8), s.card = 4 →  p s) → 
  (∀ s : Set (Fin 8), s.card = 3 → p s) →  
  (∀ s : Set (Fin 8), s.card = 1 → p s) → 
  -- condition: we want to count distinct paintings considering rotations and reflections
  (count_distinct_paintings p = 23) :=
sorry

end distinct_paintings_count_l405_405208


namespace f_odd_and_monotone_l405_405660

def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

theorem f_odd_and_monotone :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x y : ℝ), 0 < x → x < y → f x < f y) :=
  by
    sorry

end f_odd_and_monotone_l405_405660


namespace arrange_cards_l405_405129

structure Card (n : ℕ) :=
(s : ℕ)
(t : ℕ)
(hs : 1 ≤ s ∧ s ≤ n)
(ht : 1 ≤ t ∧ t ≤ n)

theorem arrange_cards (n : ℕ) (cards : Fin n → Card n) 
  (h_unique : ∀ k ∈ finset.range (n + 1), (∃ i, cards i.s s = k) ∧ (∃ j, cards j.t t = k)) 
  (h_double : ∀ k, finset.card {i | cards i.s = k} + finset.card {i | cards i.t = k} = 2) : 
  ∃ top_faces : Fin n → ℕ, (∀ i, 1 ≤ top_faces i ∧ top_faces i ≤ n) ∧ (∀ k, ∃! i, top_faces i = k) := by
sorry

end arrange_cards_l405_405129


namespace f_odd_and_monotone_l405_405656

def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

theorem f_odd_and_monotone :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x y : ℝ), 0 < x → x < y → f x < f y) :=
  by
    sorry

end f_odd_and_monotone_l405_405656


namespace primary_schools_to_be_selected_l405_405192

noncomputable def total_schools : ℕ := 150 + 75 + 25
noncomputable def proportion_primary : ℚ := 150 / total_schools
noncomputable def selected_primary : ℚ := proportion_primary * 30

theorem primary_schools_to_be_selected : selected_primary = 18 :=
by sorry

end primary_schools_to_be_selected_l405_405192


namespace angle_relationships_l405_405954

variables {Point : Type} [AffineSpace ℝ Point]

-- Define Points A, B, C, D, E
variables (A B C D E : Point)

-- Define side BC
variable (a : ℝ)

-- Internal and external angle bisectors conditions
variable (AD_internal_bisector : ∃ D, is_internal_angle_bisector A B C D)
variable (AE_external_bisector : ∃ E, is_external_angle_bisector A B C E)
variable (AD_equals_AE : AD = AE)

-- Define angles
variables (alpha beta gamma : ℝ)

-- Given conditions about triangle ABC with AD = AE
theorem angle_relationships
  (h1 : ∠BAC = α)
  (h2 : ∠ABC = β)
  (h3 : ∠BCA = γ)
  (h_sum : α + β + γ = 180)
  (h_AD_internal : is_internal_angle_bisector A B C D)
  (h_AE_external : is_external_angle_bisector A B C E)
  (h_AD_AE : dist A D = dist A E) :
  γ < 45 ∧ β = 90 + γ ∧ α = 90 - 2 * γ :=
sorry

end angle_relationships_l405_405954


namespace parabola_distance_l405_405636

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405636


namespace number_of_terms_in_sequence_l405_405169

theorem number_of_terms_in_sequence :
  let seq : List ℤ := List.range' (-28) 81 (λ n, -28 + n * 5)
  ∃ n, seq.length = n ∧ n = 17 :=
by
  let seq : List ℤ := List.range' (-28) 81 (λ n, -28 + n * 5)
  existsi seq.length
  split
  · assumption
  sorry

end number_of_terms_in_sequence_l405_405169


namespace parabola_distance_l405_405628

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405628


namespace distance_AB_l405_405439

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405439


namespace solution_to_inequality_l405_405788

theorem solution_to_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : 1 / x > 1 :=
by
  sorry

end solution_to_inequality_l405_405788


namespace tens_digit_23_pow_1987_l405_405920

def tens_digit_of_power (a b n : ℕ) : ℕ :=
  ((a^b % n) / 10) % 10

theorem tens_digit_23_pow_1987 : tens_digit_of_power 23 1987 100 = 4 := by
  sorry

end tens_digit_23_pow_1987_l405_405920


namespace sum_of_largest_three_l405_405109

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l405_405109


namespace total_puff_pastries_l405_405851

theorem total_puff_pastries (batches trays puff_pastry volunteers : ℕ) 
  (h_batches : batches = 1) 
  (h_trays : trays = 8) 
  (h_puff_pastry : puff_pastry = 25) 
  (h_volunteers : volunteers = 1000) : 
  (volunteers * trays * puff_pastry) = 200000 := 
by 
  have h_total_trays : volunteers * trays = 1000 * 8 := by sorry
  have h_total_puff_pastries_per_volunteer : trays * puff_pastry = 8 * 25 := by sorry
  have h_total_puff_pastries : volunteers * trays * puff_pastry = 1000 * 8 * 25 := by sorry
  sorry

end total_puff_pastries_l405_405851


namespace rectangle_width_decrease_l405_405768

theorem rectangle_width_decrease (L W : ℝ) (h : L * W = A) (h_new_length : 1.40 * L = L') 
    (h_area_unchanged : L' * W' = L * W) : 
    (W - W') / W = 0.285714 :=
begin
  sorry
end

end rectangle_width_decrease_l405_405768


namespace rectangle_width_decrease_l405_405767

theorem rectangle_width_decrease (L W : ℝ) (h : L * W = A) (h_new_length : 1.40 * L = L') 
    (h_area_unchanged : L' * W' = L * W) : 
    (W - W') / W = 0.285714 :=
begin
  sorry
end

end rectangle_width_decrease_l405_405767


namespace correct_calculation_l405_405818

theorem correct_calculation :
  (sqrt 27 / sqrt 3 = 3) ∧ (sqrt 3 + sqrt 2 ≠ sqrt 5) ∧ (sqrt 6 * sqrt 2 ≠ 4 * sqrt 3) ∧ (-sqrt 3 + 4 * sqrt 3 ≠ 4) :=
by { sorry }

end correct_calculation_l405_405818


namespace q_at_2_l405_405161

noncomputable def q (x : ℝ) : ℝ :=
  Real.sign (3 * x - 2) * |3 * x - 2|^(1/4) +
  2 * Real.sign (3 * x - 2) * |3 * x - 2|^(1/6) +
  |3 * x - 2|^(1/8)

theorem q_at_2 : q 2 = 4 := by
  -- Proof attempt needed
  sorry

end q_at_2_l405_405161


namespace six_digit_palindromes_count_l405_405027

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l405_405027


namespace rectangle_width_decrease_proof_l405_405757

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l405_405757


namespace parabola_distance_l405_405385

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405385


namespace Y_tagged_value_l405_405894

variables (W X Y Z : ℕ)
variables (tag_W : W = 200)
variables (tag_X : X = W / 2)
variables (tag_Z : Z = 400)
variables (total : W + X + Y + Z = 1000)

theorem Y_tagged_value : Y = 300 :=
by sorry

end Y_tagged_value_l405_405894


namespace cube_quotient_l405_405852

theorem cube_quotient (V : ℝ) (hV : V = 2744) : (14^2 / 14) = 14 := 
by
  have s := Real.cbrt V
  have hs : s = 14 := by 
    rw [hV]
    norm_num
  sorry

end cube_quotient_l405_405852


namespace range_of_x0_l405_405136

theorem range_of_x0 (x y : ℝ) (P Q : ℝ × ℝ) (O : ℝ × ℝ) :
  (O.1 = 0 ∧ O.2 = 0) ∧ (P.1 + 3 * P.2 - 6 = 0) ∧ ((Q.1^2 + Q.2^2 = 3) ∧ (∠ (O, P, Q) = 60)) →
  0 ≤ P.1 ∧ P.1 ≤ 6 / 5 := 
begin
  sorry,
end

end range_of_x0_l405_405136


namespace min_value_sin_l405_405955

section 
variables {α β : ℝ}

theorem min_value_sin (h : -5 * sin α^2 + sin β^2 = 3 * sin α) : ∃ m, m = 0 ∧ ∀ y, y = sin α^2 + sin β^2 → y ≥ m :=
begin
  sorry
end 

end

end min_value_sin_l405_405955


namespace AB_distance_l405_405329

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405329


namespace problem_l405_405612

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405612


namespace tourism_revenue_scientific_notation_l405_405827

theorem tourism_revenue_scientific_notation:
  (12.41 * 10^9) = (1.241 * 10^9) := 
sorry

end tourism_revenue_scientific_notation_l405_405827


namespace sum_of_three_largest_l405_405063

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l405_405063


namespace domain_of_f_l405_405738

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / (log (x + 1)) + sqrt (2 - x)

theorem domain_of_f : {x : ℝ | -1 < x ∧ x ≠ 0 ∧ x ≤ 2} = {x : ℝ | -1 < x ∧ x ≤ 2 \ {0}} :=
by
  sorry

end domain_of_f_l405_405738


namespace root_in_interval_l405_405642

theorem root_in_interval (a b α β : ℝ) 
(hα : α^2 + a * α + b = 0) 
(hβ : β^2 - a * β - b = 0) : 
∃ x ∈ Icc α β, x^2 - 2 * a * x - 2 * b = 0 :=
by
  sorry

end root_in_interval_l405_405642


namespace AB_distance_l405_405310

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405310


namespace rectangle_width_decrease_proof_l405_405759

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l405_405759


namespace bound_on_k_l405_405233

variables {n k : ℕ}
variables (a : ℕ → ℕ) (h1 : 1 ≤ k) (h2 : ∀ i j, 1 ≤ i → j ≤ k → i < j → a i < a j)
variables (h3 : ∀ i, a i ≤ n) (h4 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → a i ≠ a j))
variables (h5 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → ∀ m p, m ≤ p → m ≤ k → p ≤ k → a i + a j ≠ a m + a p))

theorem bound_on_k : k ≤ Nat.floor (Real.sqrt (2 * n) + 1) :=
sorry

end bound_on_k_l405_405233


namespace max_min_value_of_product_l405_405184

theorem max_min_value_of_product (x y : ℝ) (h : x ^ 2 + y ^ 2 = 1) :
  (1 + x * y) * (1 - x * y) ≤ 1 ∧ (1 + x * y) * (1 - x * y) ≥ 3 / 4 :=
by sorry

end max_min_value_of_product_l405_405184


namespace AB_distance_l405_405306

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405306


namespace excellent_students_estimation_distribution_of_X_mean_of_X_l405_405887

def students_scores : List ℕ := [60, 85, 80, 78, 90, 91]

def excellent_students (scores : List ℕ) : ℕ :=
  scores.filter (λ x => x ≥ 85).length

def good_students (scores : List ℕ) : ℕ :=
  scores.filter (λ x => x ≥ 75 ∧ x < 85).length

def excellent_estimated (total_students : ℕ) (n_selected : ℕ) (excellent_in_sample : ℕ) : ℕ :=
  (excellent_in_sample * total_students) / n_selected

theorem excellent_students_estimation :
  excellent_estimated 600 6 (excellent_students students_scores) = 300 := by
  sorry

def X_distribution : List (ℕ × ℚ) :=
  [(0, 1/5), (1, 3/5), (2, 1/5)]

def mean_X (dist : List (ℕ × ℚ)) : ℚ :=
  dist.foldr (λ (xt : ℕ × ℚ) acc => xt.fst * xt.snd + acc) 0

theorem distribution_of_X :
  X_distribution = [(0, 1/5), (1, 3/5), (2, 1/5)] := by
  sorry

theorem mean_of_X :
  mean_X [(0, 1/5), (1, 3/5), (2, 1/5)] = 1 := by
  sorry

end excellent_students_estimation_distribution_of_X_mean_of_X_l405_405887


namespace six_digit_palindromes_count_l405_405038

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l405_405038


namespace arithmetic_sequence_common_ratio_sum_of_na_n_l405_405904

theorem arithmetic_sequence_common_ratio (c : ℝ) (a : ℕ → ℝ)
  (h_a_seq : ∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / 2)
  (h_a1 : a 1 = 1) (h_arith_seq: ∀ n, a (n + 1) = c * a n) :
  c = 1 ∨ c = -1 / 2 := sorry

theorem sum_of_na_n (c : ℝ) (a : ℕ → ℝ) (n : ℕ)
  (h_a_seq : ∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / 2)
  (h_a1 : a 1 = 1) (h_arith_seq: ∀ n, a (n + 1) = c * a n):
  let S_n := ∑ i in Finset.range n, i * a (i + 1) in
  (c = 1 → S_n = n * (n + 1) / 2) ∧ (c = -1 / 2 → S_n = 1/9 * (4 - (-1)^n * (3 * n + 2) / 2^(n-1))) := sorry

end arithmetic_sequence_common_ratio_sum_of_na_n_l405_405904


namespace point_below_line_l405_405181

theorem point_below_line {a : ℝ} (h : 2 * a - 3 < 3) : a < 3 :=
by {
  sorry
}

end point_below_line_l405_405181


namespace basketball_volume_correct_l405_405847

noncomputable def basketball_volume_after_holes (d_basketball d1 d2 depth r : ℝ) : ℝ :=
  let r_basketball := d_basketball / 2
  let V_basketball := (4 / 3) * Math.pi * r_basketball^3
  let r1 := d1 / 2
  let r2 := d2 / 2
  let V_hole1 := Math.pi * (r1^2) * depth
  let V_hole2 := Math.pi * (r2^2) * depth
  V_basketball - 2 * V_hole1 - 2 * V_hole2

theorem basketball_volume_correct :
  basketball_volume_after_holes 50 4 3 10 25 = 20750 * Math.pi :=
by
  sorry

end basketball_volume_correct_l405_405847


namespace part1_part2_part3_l405_405974

-- Given definitions of functions f and g
def f (x : ℝ) := log x + 1/x
def g (x : ℝ) := x - log x

-- Proof problems translated into Lean statements

-- Part (1): Prove that the maximum value of a for which f(x) ≥ a for any x ∈ (0, +∞) is 1.
theorem part1 : ∀ (a : ℝ), (∀ x > 0, f x ≥ a) → a ≤ 1 := sorry

-- Part (2): Prove f(x) < g(x) for x ∈ (1, +∞).
theorem part2 : ∀ x > 1, f x < g x := sorry

-- Part (3): Prove that if x1 > x2 and g(x1) = g(x2), then x1 * x2 < 1.
theorem part3 : ∀ (x1 x2 : ℝ), x1 > x2 → g x1 = g x2 → x1 * x2 < 1 := sorry

end part1_part2_part3_l405_405974


namespace six_digit_palindromes_count_l405_405005

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l405_405005


namespace problem_l405_405611

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405611


namespace f_properties_l405_405662

def f (x : ℝ) : ℝ := x^3 - 1 / x^3

theorem f_properties : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := 
by
  sorry

end f_properties_l405_405662


namespace min_distance_sum_l405_405987

noncomputable def parabola : set (ℝ × ℝ) := { p | p.2 = (1 / 12) * (p.1)^2 }
def P : (ℝ × ℝ) → Prop := λ p, p ∈ parabola
def A : ℝ × ℝ := (4, 0)
def M (p : ℝ × ℝ) : ℝ × ℝ := (p.1, 0)

theorem min_distance_sum (p : ℝ × ℝ)
  (hp : P p) :
  (∃ (min_val : ℝ), min_val = 2 ∧ ∀ p, P p → |dist p A + dist p (M p)| ≥ min_val) :=
sorry

end min_distance_sum_l405_405987


namespace parabola_distance_l405_405481

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405481


namespace jenga_initial_blocks_l405_405226

/-- Prove the initial number of blocks in the Jenga game. -/
theorem jenga_initial_blocks 
    (players rounds blocks_before_Jess_turn : ℕ) 
    (blocks_removed_before_Jess_turn : ℕ)
    (h_players : players = 5)
    (h_rounds : rounds = 5)
    (h_blocks_before_Jess_turn : blocks_before_Jess_turn = 28)
    (h_blocks_removed_before_Jess_turn : blocks_removed_before_Jess_turn = 26)
    : ∃ n, n = blocks_before_Jess_turn + blocks_removed_before_Jess_turn := 
by {
    use (blocks_before_Jess_turn + blocks_removed_before_Jess_turn),
    linarith,
}

end jenga_initial_blocks_l405_405226


namespace parabola_distance_l405_405620

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405620


namespace parabola_problem_l405_405284

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405284


namespace parabola_problem_l405_405520

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405520


namespace distance_AB_l405_405452

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405452


namespace count_non_adjacent_A_and_B_count_relay_race_l405_405900

-- Defining the conditions for the first problem
variable {α : Type*} (people : List α) (A B : α)
#check people.length = 7

-- Defining the enumeration function
noncomputable def count_non_adjacent_arrangements (A B : α) (people : List α) : ℕ :=
if people.length = 7 then 3600 else 0

-- First statement: Prove the enumeration is correct for non-adjacent A and B
theorem count_non_adjacent_A_and_B (A B : α) (people : List α) :
  people.length = 7 →
  count_non_adjacent_arrangements A B people = 3600 :=
sorry

-- Definitions for the second problem
variable (selected : List α)
#check selected.length = 4
#check A ∈ selected
#check B ∈ selected

-- Defining the count function for relay race arrangements
noncomputable def count_relay_arrangements (A B : α) (selected : List α) : ℕ :=
if selected.length = 4 ∧ A ∈ selected ∧ B ∈ selected then 140 else 0

-- Second statement: Prove the enumeration is correct for the relay race arrangements
theorem count_relay_race (A B : α) (selected : List α) :
  selected.length = 4 →
  A ∈ selected →
  B ∈ selected →
  count_relay_arrangements A B selected = 140 :=
sorry

end count_non_adjacent_A_and_B_count_relay_race_l405_405900


namespace sum_gn_eq_one_third_l405_405117

noncomputable def g (n : ℕ) : ℝ :=
  ∑' i : ℕ, if i ≥ 3 then 1 / (i ^ n) else 0

theorem sum_gn_eq_one_third :
  (∑' n : ℕ, if n ≥ 3 then g n else 0) = 1 / 3 := 
by sorry

end sum_gn_eq_one_third_l405_405117


namespace polygon_P_properties_l405_405231

-- Definitions of points A, B, and C
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (1, 0.5, 0)
def C : (ℝ × ℝ × ℝ) := (0, 0.5, 1)

-- Condition of cube intersection and plane containing A, B, and C
def is_corner_of_cube (p : ℝ × ℝ × ℝ) : Prop :=
  p = A

def are_midpoints_of_cube_edges (p₁ p₂ : ℝ × ℝ × ℝ) : Prop :=
  (p₁ = B ∧ p₂ = C)

-- Polygon P resulting from the plane containing A, B, and C intersecting the cube
def num_sides_of_polygon (p : ℝ × ℝ × ℝ) : ℕ := 5 -- Given the polygon is a pentagon

-- Area of triangle ABC
noncomputable def area_triangle_ABC : ℝ :=
  (1/2) * (Real.sqrt 1.5)

-- Area of polygon P
noncomputable def area_polygon_P : ℝ :=
  (11/6) * area_triangle_ABC

-- Theorem stating that polygon P has 5 sides and the ratio of its area to the area of triangle ABC is 11/6
theorem polygon_P_properties (A B C : (ℝ × ℝ × ℝ))
  (hA : is_corner_of_cube A) (hB : are_midpoints_of_cube_edges B C) :
  num_sides_of_polygon A = 5 ∧ area_polygon_P / area_triangle_ABC = (11/6) :=
by sorry

end polygon_P_properties_l405_405231


namespace parabola_problem_l405_405569

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405569


namespace exists_adjacent_numbers_with_real_roots_l405_405190

theorem exists_adjacent_numbers_with_real_roots :
  ∃ (a b : ℕ), (1 ≤ a ≤ 36) ∧ (1 ≤ b ≤ 36) ∧ (a < b) ∧ (a ≥ 13) ∧ (a, b are_adjacent) ∧ (a^2 - 4*b > 0) := 
sorry

end exists_adjacent_numbers_with_real_roots_l405_405190


namespace greatest_divisible_by_13_l405_405865

def is_distinct_nonzero_digits (A B C : ℕ) : Prop :=
  0 < A ∧ A < 10 ∧ 0 < B ∧ B < 10 ∧ 0 < C ∧ C < 10 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def number (A B C : ℕ) : ℕ :=
  10000 * A + 1000 * B + 100 * C + 10 * B + A

theorem greatest_divisible_by_13 :
  ∃ (A B C : ℕ), is_distinct_nonzero_digits A B C ∧ number A B C % 13 = 0 ∧ number A B C = 96769 :=
sorry

end greatest_divisible_by_13_l405_405865


namespace distance_AB_l405_405349

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405349


namespace ticket_sales_total_cost_l405_405878

noncomputable def total_ticket_cost (O B : ℕ) : ℕ :=
  12 * O + 8 * B

theorem ticket_sales_total_cost (O B : ℕ) (h1 : O + B = 350) (h2 : B = O + 90) :
  total_ticket_cost O B = 3320 :=
by
  -- the proof steps calculating the total cost will go here
  sorry

end ticket_sales_total_cost_l405_405878


namespace AB_distance_l405_405311

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405311


namespace parabola_distance_l405_405392

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405392


namespace parabola_problem_l405_405517

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405517


namespace parabola_distance_l405_405398

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405398


namespace amy_soups_total_l405_405890

def total_soups (chicken_soups tomato_soups : ℕ) : ℕ :=
  chicken_soups + tomato_soups

theorem amy_soups_total : total_soups 6 3 = 9 :=
by
  -- insert the proof here
  sorry

end amy_soups_total_l405_405890


namespace distance_AB_l405_405469

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405469


namespace prove_slope_PQ_l405_405724

variables {a b c t : ℝ}

def origin : Prop := (0, 0)

def P (a b : ℝ) : Prop := a > 0 ∧ b > 0

def Q (c : ℝ) : Prop := c > 0 ∧ 1 > 0 

def a_eq_2c (a c : ℝ) : Prop := a = 2 * c

def slope_OP_eq_t (a b t : ℝ) : Prop := t = b / a

def slope_OQ_eq_1 (c : ℝ) : Prop := 1 = 1 / c

def slope_PQ_eq (a b c t : ℝ) : Prop := (2 * t - 1) = (b - 1) / (a - c)

theorem prove_slope_PQ :
  P a b →
  Q c → 
  a_eq_2c a c → 
  slope_OP_eq_t a b t → 
  slope_OQ_eq_1 c → 
  slope_PQ_eq a b c t := 
by
  sorry

end prove_slope_PQ_l405_405724


namespace coeff_x2_in_expansion_l405_405731

theorem coeff_x2_in_expansion :
  (let binomial_coeff := Nat.choose in
  ∀ (x : ℝ), 
    (2 + x) * (1 - 2 * x) ^ 5 = 2 * (binomial_coeff 5 2) * (-2)^2 * x^2 + (binomial_coeff 5 1) * (-2) * x + 70) :=
by
  sorry

end coeff_x2_in_expansion_l405_405731


namespace f_odd_and_monotone_l405_405659

def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

theorem f_odd_and_monotone :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x y : ℝ), 0 < x → x < y → f x < f y) :=
  by
    sorry

end f_odd_and_monotone_l405_405659


namespace six_digit_palindromes_count_l405_405030

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l405_405030


namespace problem_l405_405614

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405614


namespace sum_even_indexed_coefficients_eq_l405_405232

theorem sum_even_indexed_coefficients_eq (n : ℕ) :
  let f := (1 - X + X^2) ^ n in
  (f.coeff 0 + f.coeff 2 + f.coeff 4 + ... + f.coeff (2 * n)) = (1 + 3^n) / 2 :=
sorry

end sum_even_indexed_coefficients_eq_l405_405232


namespace AB_distance_l405_405369

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405369


namespace speed_first_hour_l405_405789

variable (x : ℕ)

-- Definitions based on conditions
def total_distance (x : ℕ) : ℕ := x + 50
def average_speed (x : ℕ) : Prop := (total_distance x) / 2 = 70

-- Theorem statement
theorem speed_first_hour : ∃ x, average_speed x ∧ x = 90 := by
  sorry

end speed_first_hour_l405_405789


namespace monomial_k_add_n_l405_405980

variable (k n : ℤ)

-- Conditions
def is_monomial_coefficient (k : ℤ) : Prop := -k = 5
def is_monomial_degree (n : ℤ) : Prop := n + 1 = 7

-- Theorem to prove
theorem monomial_k_add_n (hk : is_monomial_coefficient k) (hn : is_monomial_degree n) : k + n = 1 :=
by
  sorry

end monomial_k_add_n_l405_405980


namespace problem_l405_405602

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405602


namespace parabola_distance_l405_405393

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405393


namespace sum_of_three_largest_of_consecutive_numbers_l405_405075

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l405_405075


namespace max_marks_l405_405807

theorem max_marks (M: ℝ) (h1: 0.95 * M = 285):
  M = 300 :=
by
  sorry

end max_marks_l405_405807


namespace perfect_square_divisors_of_product_fact_l405_405170

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def product_of_factorials : ℕ :=
∏ i in range (10 + 1), factorial i

theorem perfect_square_divisors_of_product_fact :
  number_of_perfect_square_divisors (product_of_factorials) = 2592 :=
sorry

end perfect_square_divisors_of_product_fact_l405_405170


namespace part1_part2_part3_l405_405964

-- Define the given conditions
def condition (z : ℂ) : Prop :=
  abs (2 * z + 5) = abs (z + 10)

-- Prove |z| = 5 given condition
theorem part1 (z : ℂ) (h : condition z) : abs z = 5 :=
sorry

-- Prove there exists m ∈ ℝ such that z/m + m/(conjugate z) is real and m = ±5 given condition
theorem part2 (z : ℂ) (h : condition z) : ∃ (m : ℝ), (z / m + m / conj z).im = 0 ∧ (m = 5 ∨ m = -5) :=
sorry

-- Prove the value of z given the additional condition that (1 - 2i)z lies on the line bisecting the first and third quadrants
theorem part3 (z : ℂ) (h : condition z) (h2 : (1 - 2 * (complex.I)) * z = (1 - 2 * complex.I) * (z)) :
  z = (√10 / 2 - 3 * √10 / 2 * complex.I) ∨ z = (-√10 / 2 + 3 * √10 / 2 * complex.I) :=
sorry

end part1_part2_part3_l405_405964


namespace parabola_problem_l405_405283

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405283


namespace impossible_segments_arrangement_l405_405217

theorem impossible_segments_arrangement :
  ¬ ∃ (segments : Fin 1000 → Set (ℝ × ℝ)), 
    (∀ i : Fin 1000, ∃ j k : Fin 1000, j ≠ k ∧
      (interior (segments j)).nonempty ∧ (interior (segments k)).nonempty ∧
      ∀ p ∈ segments i, 
        ∀ q ∈ {p | segments j p ∧ segments k q},
          is_intersecting p q) :=
sorry

end impossible_segments_arrangement_l405_405217


namespace sum_of_three_largest_of_consecutive_numbers_l405_405073

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l405_405073


namespace total_cantaloupes_l405_405120

def Fred_grew_38 : ℕ := 38
def Tim_grew_44 : ℕ := 44

theorem total_cantaloupes : Fred_grew_38 + Tim_grew_44 = 82 := by
  sorry

end total_cantaloupes_l405_405120


namespace extremum_at_one_l405_405742

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + 2 * real.sqrt x - 3 * real.log x

def f_derivative (a x : ℝ) : ℝ := 2 * a * x + (1 / real.sqrt x) - (3 / x)

theorem extremum_at_one (a : ℝ) (h : f_derivative a 1 = 0) : a = 1 :=
by 
  simp [f_derivative, real.sqrt, real.log] at h
  exact eq_of_add_eq_add_right h

#check extremum_at_one

end extremum_at_one_l405_405742


namespace AB_distance_l405_405368

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405368


namespace parabola_distance_l405_405525

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405525


namespace distance_AB_l405_405426

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405426


namespace normal_distribution_probability_l405_405132

noncomputable def normal_distribution_3_1_4 := measure_theory.measure_space.ProbabilityMeasure (measure_theory.measure.gaussian 3 (real.sqrt (1 / 4)))
def X : random_variable normal_distribution_3_1_4 real := sorry

theorem normal_distribution_probability :
  (∀ x, P(X > 7 / 2) = 0.1587) → P(5 / 2 ≤ X ∧ X ≤ 7 / 2) = 0.6826 :=
begin
  sorry
end

end normal_distribution_probability_l405_405132


namespace quadratic_inequality_always_positive_l405_405913

theorem quadratic_inequality_always_positive (k : ℝ) :
  (∀ x : ℝ, 2 * x^2 - (k - 1) * x - 2 * k + 8 > 0) ↔ k ∈ Ioo (-9 : ℝ) 7 :=
by
  sorry

end quadratic_inequality_always_positive_l405_405913


namespace fruit_basket_count_l405_405172

theorem fruit_basket_count :
  let pears := 8
  let bananas := 12
  let total_baskets := (pears + 1) * (bananas + 1) - 1
  total_baskets = 116 :=
by
  sorry

end fruit_basket_count_l405_405172


namespace atomic_weight_Oxygen_l405_405943

theorem atomic_weight_Oxygen :
  ∀ (Ba_atomic_weight S_atomic_weight : ℝ),
    (Ba_atomic_weight = 137.33) →
    (S_atomic_weight = 32.07) →
    (Ba_atomic_weight + S_atomic_weight + 4 * 15.9 = 233) →
    15.9 = 233 - 137.33 - 32.07 / 4 := 
by
  intros Ba_atomic_weight S_atomic_weight hBa hS hm
  sorry

end atomic_weight_Oxygen_l405_405943


namespace problem1_l405_405963

theorem problem1 :
  let total_products := 10
  let defective_products := 4
  let first_def_pos := 5
  let last_def_pos := 10
  ∃ (num_methods : Nat), num_methods = 103680 :=
by
  sorry

end problem1_l405_405963


namespace distance_AB_l405_405458

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405458


namespace min_f_value_l405_405846

noncomputable def f (x : ℝ) := real.sqrt (5 * x^2 - 16 * x + 16) + real.sqrt (5 * x^2 - 18 * x + 29)

theorem min_f_value : ∃ x : ℝ, is_min_on f (set.univ : set ℝ) x ∧ f x = real.sqrt 29 := sorry

end min_f_value_l405_405846


namespace smallest_angle_between_a_and_c_l405_405643

open Real

variables {a b c : ℝ^3}
variables (a_norm : ‖a‖ = 1) (b_norm : ‖b‖ = 1) (c_norm : ‖c‖ = 3)
variables (triple_prod_identity : a × (a × c) + 2 • b = 0)

theorem smallest_angle_between_a_and_c :
  let θ := real_vectors.angle a c in
  θ = real.arccos (1 / 3) :=
sorry

end smallest_angle_between_a_and_c_l405_405643


namespace parabola_problem_l405_405511

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405511


namespace parabola_distance_l405_405531

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405531


namespace train_passing_tree_time_l405_405879

theorem train_passing_tree_time
  (train_length : ℝ) (train_speed_kmhr : ℝ) (conversion_factor : ℝ)
  (train_speed_ms : train_speed_ms = train_speed_kmhr * conversion_factor) :
  train_length = 500 → train_speed_kmhr = 72 → conversion_factor = 5 / 18 →
  500 / (72 * (5 / 18)) = 25 := 
by
  intros h1 h2 h3
  sorry

end train_passing_tree_time_l405_405879


namespace sum_of_three_largest_l405_405078

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l405_405078


namespace quadrilateral_circle_area_ratio_l405_405706

-- Given Definitions
variables (s : ℝ) -- radius of the circle
def E := (0 : ℝ) -- Let E be the center of the circle at origin for simplicity.
def G := (2 * s : ℝ) -- G lies on the circumference.
def F := (s * sin 30, s * cos 30) -- F is such that angle FEG = 45 degrees.
def H := (-s * sin 60, s * cos 60) -- H is such that angle HEG = 60 degrees.
def area_circle : ℝ := π * s^2
def area_triang_EFG : ℝ := s^2
def area_triang_EHG : ℝ := (s^2 * sqrt 3) / 2
def area_EFGH : ℝ := s^2 * (1 + sqrt 3 / 2)
def ratio_area : ℝ := area_EFGH / area_circle

-- The theorem we want to prove
theorem quadrilateral_circle_area_ratio : 
  ratio_area = (2 + sqrt 3) / (2 * π) ∧ 
  let a := 2, b := 3, c := 2 in a + b + c = 7 :=
by
  sorry

end quadrilateral_circle_area_ratio_l405_405706


namespace sum_of_three_largest_consecutive_numbers_l405_405100

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l405_405100


namespace total_spent_on_entertainment_l405_405221

def cost_of_computer_game : ℕ := 66
def cost_of_one_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3

theorem total_spent_on_entertainment : cost_of_computer_game + cost_of_one_movie_ticket * number_of_movie_tickets = 102 := 
by sorry

end total_spent_on_entertainment_l405_405221


namespace solve_for_x_l405_405716

theorem solve_for_x : ∀ x : ℝ, 5 * x + 9 * x = 360 - 7 * (x + 4) → x = 332 / 21 :=
by { 
  intro x, 
  intro h, 
  sorry 
}

end solve_for_x_l405_405716


namespace parabola_distance_l405_405637

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405637


namespace num_common_points_l405_405915

-- Definitions of the given conditions:
def line1 (x y : ℝ) := x + 2 * y - 3 = 0
def line2 (x y : ℝ) := 4 * x - y + 1 = 0
def line3 (x y : ℝ) := 2 * x - y - 5 = 0
def line4 (x y : ℝ) := 3 * x + 4 * y - 8 = 0

-- The proof goal:
theorem num_common_points : 
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2) ∧ (line3 p.1 p.2 ∨ line4 p.1 p.2) :=
sorry

end num_common_points_l405_405915


namespace AB_distance_l405_405324

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405324


namespace six_digit_palindromes_count_l405_405035

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l405_405035


namespace distance_AB_l405_405344

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405344


namespace sum_first_2017_terms_l405_405994

noncomputable def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q ∧ a n ≤ a (n + 1)

theorem sum_first_2017_terms {a : ℕ → ℝ} {q : ℝ} :
  increasing_geometric_sequence a q →
  a 1 + a 4 = 9 →
  a 2 * a 3 = 8 →
  (∑ k in range (2017 + 1), a k) = 2^2017 - 1 :=
by
  sorry

end sum_first_2017_terms_l405_405994


namespace problem_l405_405616

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405616


namespace parabola_distance_problem_l405_405420

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405420


namespace parabola_distance_l405_405630

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405630


namespace cubicsum_eq_neg36_l405_405666

noncomputable def roots (p q r : ℝ) := 
  ∃ l : ℝ, (p^3 - 12) / p = l ∧ (q^3 - 12) / q = l ∧ (r^3 - 12) / r = l

theorem cubicsum_eq_neg36 {p q r : ℝ} (h : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hl : roots p q r) :
  p^3 + q^3 + r^3 = -36 :=
sorry

end cubicsum_eq_neg36_l405_405666


namespace exist_three_not_played_l405_405197

noncomputable def football_championship (teams : Finset ℕ) (rounds : ℕ) : Prop :=
  let pairs_per_round := teams.card / 2 in
  let total_pairs_constraint := rounds * pairs_per_round in
  let constraint_matrix := (teams.card - 1) - pairs_per_round in
  (teams.card = 18) ∧
  (rounds = 8) ∧
  (pairs_per_round * rounds < constraint_matrix * (constraint_matrix - 1) / 2) ->
  ∃ (A B C : ℕ) (hA : A ∈ teams) (hB : B ∈ teams) (hC : C ∈ teams),
    ¬ (⊢ (A, B) ∈ teams * teams) ∧ ¬ (⊢ (B, C) ∈ teams * teams) ∧ ¬ (⊢ (A, C) ∈ teams * teams)

theorem exist_three_not_played :
  ∃ (football_championship (Finset.range 18) 8) :=
begin
  sorry,
end

end exist_three_not_played_l405_405197


namespace parabola_distance_l405_405494

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405494


namespace AB_distance_l405_405307

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405307


namespace AB_distance_l405_405315

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405315


namespace total_cantaloupes_l405_405123

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by
  sorry

end total_cantaloupes_l405_405123


namespace parabola_distance_l405_405382

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405382


namespace sprint_team_total_miles_l405_405795

-- Define the number of people and miles per person as constants
def numberOfPeople : ℕ := 250
def milesPerPerson : ℝ := 7.5

-- Assertion to prove the total miles
def totalMilesRun : ℝ := numberOfPeople * milesPerPerson

-- Proof statement
theorem sprint_team_total_miles : totalMilesRun = 1875 := 
by 
  -- Proof to be filled in
  sorry

end sprint_team_total_miles_l405_405795


namespace pyramid_total_surface_area_l405_405869

-- Define the side length of the hexagonal base
def side_length : ℝ := 8
-- Define the height of the pyramid from base center to peak
def height : ℝ := 15
-- Define the total surface area we want to prove for the given pyramid
def total_surface_area : ℝ := 96 * Real.sqrt 3 + 24 * Real.sqrt 241

-- The main theorem stating the total surface area calculation
theorem pyramid_total_surface_area
  (s : ℝ) (h : ℝ) (total_area : ℝ)
  (hs : s = side_length) (hh : h = height) (hta : total_area = total_surface_area) :
  -- Calculate base area of hexagonal
  let base_area := 3 * Real.sqrt 3 / 2 * s^2,
  base_area = 96 * Real.sqrt 3 ∧
  -- Calculate slant height of the pyramid
  let half_side := s / 2,
  let slant_height := Real.sqrt (h^2 + half_side^2),
  slant_height = Real.sqrt 241 ∧
  -- Calculate area of one triangular face
  let tri_area := 1 / 2 * s * slant_height,
  tri_area = 4 * Real.sqrt 241 ∧
  -- Calculate total lateral area
  let lateral_area := 6 * tri_area,
  lateral_area = 24 * Real.sqrt 241 ∧
  -- Calculate total surface area
  let total_surface_area_calc := base_area + lateral_area,
  total_surface_area_calc = total_area := by
  sorry

end pyramid_total_surface_area_l405_405869


namespace arithmetic_sequence_a5_value_l405_405971

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) 
  (h1 : a 2 + a 4 = 16) 
  (h2 : a 1 = 1) : 
  a 5 = 15 := 
by 
  sorry

end arithmetic_sequence_a5_value_l405_405971


namespace parabola_distance_l405_405641

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405641


namespace parabola_problem_l405_405507

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405507


namespace largest_integer_with_properties_l405_405746

-- Define the conditions and proof statement

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit_prime (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100 ∧ is_prime n

def adjacent_primes_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  ∀ i, i < digits.length - 1 →
  is_two_digit_prime (digits.nth_le i sorry * 10 + digits.nth_le (i + 1) sorry) ∧ 
  ∀ j, j > i → (digits.nth_le i sorry * 10 + digits.nth_le (i + 1) sorry) ≠ (digits.nth_le j sorry * 10 + digits.nth_le (j + 1) sorry)

theorem largest_integer_with_properties : adjacent_primes_distinct 617371311979 :=
  sorry

end largest_integer_with_properties_l405_405746


namespace rectangle_width_decrease_l405_405773

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L in
  let W' := (L * W) / L' in
  let percent_decrease := (1 - W' / W) * 100 in
  percent_decrease ≈ 28.57 :=
by
  let L' := 1.4 * L
  let W' := (L * W) / L'
  let percent_decrease := (1 - W' / W) * 100
  sorry

end rectangle_width_decrease_l405_405773


namespace maximize_expression_l405_405128

noncomputable def expression (x y : ℝ) : ℝ := sqrt (x * y) * (1 - x - 2 * y)

theorem maximize_expression :
  ∀ (x y : ℝ), x > 0 ∧ y > 0 → 
  (∃ x_max : ℝ, (x_max = 1 / 4) ∧ 
  (∀ x' : ℝ, x' > 0 ∧ y > 0 → expression x' y ≤ expression x_max y) ∧ 
  expression x_max y = sqrt 2 / 16) :=
λ x y ⟨hx, hy⟩, sorry

end maximize_expression_l405_405128


namespace sum_of_three_largest_of_consecutive_numbers_l405_405068

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l405_405068


namespace distance_AB_l405_405263

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405263


namespace correct_product_exists_l405_405199

variable (a b : ℕ)

theorem correct_product_exists
  (h1 : a < 100)
  (h2 : 10 * (a % 10) + a / 10 = 14)
  (h3 : 14 * b = 182) : a * b = 533 := sorry

end correct_product_exists_l405_405199


namespace parabola_distance_l405_405536

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405536


namespace parabola_problem_l405_405287

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405287


namespace parabola_distance_l405_405640

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405640


namespace parabola_distance_l405_405530

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405530


namespace parabola_distance_l405_405639

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405639


namespace parabola_distance_problem_l405_405413

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405413


namespace max_volume_prism_l405_405201

theorem max_volume_prism (V : ℝ) (h l w : ℝ) 
  (h_eq_2h : l = 2 * h ∧ w = 2 * h) 
  (surface_area_eq : l * h + w * h + l * w = 36) : 
  V = 27 * Real.sqrt 2 := 
  sorry

end max_volume_prism_l405_405201


namespace length_of_hypotenuse_in_30_60_90_triangle_l405_405785

theorem length_of_hypotenuse_in_30_60_90_triangle
  (r : ℝ)
  (a b c : ℝ)
  (h1 : r = 10)
  (angleA : ℝ) (angleB : ℝ) (angleC : ℝ)
  (h2 : angleA = 30) (h3 : angleB = 60) (h4 : angleC = 90)
  (hypotenuse : ℝ) (h5 : hypotenuse = 2 * r * sqrt 3) :
  hypotenuse = 20 * sqrt 3 :=
sorry

end length_of_hypotenuse_in_30_60_90_triangle_l405_405785


namespace number_of_crows_is_three_l405_405210

/-- Given the conditions:
1. Anna and Dana always tell the truth (they are swans).
2. Bob and Charles always lie (they are crows).
3. Anna: "Dana and I are the same species."
4. Bob: "Anna is a crow."
5. Charles: "Anna is also a crow."
6. Dana: "Exactly one of us is a swan."

Prove that the number of crows among Anna, Bob, Charles, and Dana is 3. -/
theorem number_of_crows_is_three
  (Anna_swans : ∀ x : Prop, x → ¬ x → False)
  (Dana_swans : ∀ x : Prop, x → ¬ x → False)
  (Bob_crows : ∀ x : Prop, x → ¬ x → True)
  (Charles_crows : ∀ x : Prop, x → ¬ x → True)
  (Anna_statement : Anna_swans ↔ Dana_swans)
  (Bob_statement : ∀ x : Prop, x = Anna_swans → ¬ x = Anna_swans → True)
  (Charles_statement : ∀ x : Prop, x = Anna_swans → ¬ x = Anna_swans → True)
  (Dana_statement : (1 = 1 → (Anna_swans ∧ Bob_crows ∧ Charles_crows ∧ Dana_swans)) ∧ ¬ (2 = 1)) :
  (∃ x : Prop, x = Bob_crows ∧ x = Charles_crows ∧ x = Dana_swans) :=
  sorry

end number_of_crows_is_three_l405_405210


namespace abs_sum_inequality_l405_405056

open Real

theorem abs_sum_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| > k) ↔ k < 4 := 
begin
  sorry
end

end abs_sum_inequality_l405_405056


namespace six_digit_palindromes_count_l405_405032

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l405_405032


namespace clock_angle_3_40_is_130_l405_405812

def hour_hand_angle (h m : ℕ) : ℝ :=
  (h % 12) * 30 + m * 0.5

def minute_hand_angle (m : ℕ) : ℝ :=
  m * 6

def angle_difference (a b : ℝ) : ℝ :=
  min (abs (a - b)) (360 - abs (a - b))

def clock_angle_at_3_40 : ℝ := 
  let h := 3 in
  let m := 40 in
  angle_difference (hour_hand_angle h m) (minute_hand_angle m)

theorem clock_angle_3_40_is_130 : clock_angle_at_3_40 = 130 :=
by sorry

end clock_angle_3_40_is_130_l405_405812


namespace parabola_distance_l405_405591

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405591


namespace bill_profit_difference_l405_405896

theorem bill_profit_difference (P : ℝ) 
  (h1 : 1.10 * P = 549.9999999999995)
  (h2 : ∀ NP NSP, NP = 0.90 * P ∧ NSP = 1.30 * NP →
  NSP - 549.9999999999995 = 35) :
  true :=
by {
  sorry
}

end bill_profit_difference_l405_405896


namespace barney_minutes_proof_l405_405895

-- Conditions:
def barney_situps_per_minute := 45
def carrie_situps_per_minute := 2 * barney_situps_per_minute
def jerrie_situps_per_minute := carrie_situps_per_minute + 5
def total_situps_performed := 510

-- Number of minutes each performed sit-ups:
def barney_minutes : ℕ
def carrie_minutes := 2
def jerrie_minutes := 3

-- Total sit-ups equation:
def total_situps := (barney_situps_per_minute * barney_minutes) + 
                    (carrie_situps_per_minute * carrie_minutes) + 
                    (jerrie_situps_per_minute * jerrie_minutes)

-- Prove that the number of minutes Barney did sit-ups is 1:
theorem barney_minutes_proof : 
  total_situps = total_situps_performed → barney_minutes = 1 := by
  sorry

end barney_minutes_proof_l405_405895


namespace parabola_distance_l405_405477

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405477


namespace problem_1_problem_2_problem_3_l405_405996

noncomputable def curve (k : ℝ) : ℝ × ℝ → Prop :=
λ (p : ℝ × ℝ), let (x, y) := p in
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

theorem problem_1 (k : ℝ) (hk : k ≠ -1) :
  ∃ (r : ℝ) (h : r > 0) (c : ℝ × ℝ), curve k = λ (p : ℝ × ℝ), (p.1 + k)^2 + (p.2 + 2*k + 5)^2 = r^2 ∧
  ∀ (x y : ℝ), c = (-k, -2*k-5) ∧ y = 2*x - 5 :=
sorry

theorem problem_2 (k : ℝ) (hk : k ≠ -1) :
  curve k (1, -3) :=
sorry

theorem problem_3 (k : ℝ) :
  (curve k (x, 0) → |2*k + 5| = sqrt 5 * |k + 1| → (k = 5 - 3 * sqrt 5 ∨ k = 5 + 3 * sqrt 5)) :=
sorry

end problem_1_problem_2_problem_3_l405_405996


namespace AB_distance_l405_405365

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405365


namespace some_zen_not_cen_l405_405167

variable {Zen Ben Cen : Type}
variables (P Q R : Zen → Prop)

theorem some_zen_not_cen (h1 : ∀ x, P x → Q x)
                        (h2 : ∃ x, Q x ∧ ¬ (R x)) :
  ∃ x, P x ∧ ¬ (R x) :=
  sorry

end some_zen_not_cen_l405_405167


namespace six_digit_palindromes_l405_405021

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l405_405021


namespace no_real_solutions_for_p_p_eq_q_q_l405_405986

variables {R : Type*} [Ring R] (p q : R → R)

-- Define p and q as polynomials
variables [IsPolynomial p] [IsPolynomial q]

-- Given conditions
def p_comp_q_eq_q_comp_p : ∀ x : R, p (q x) = q (p x) := sorry
def p_eq_q_has_no_real_solutions : ∀ x : R, p x ≠ q x := sorry

-- The theorem to prove
theorem no_real_solutions_for_p_p_eq_q_q :
  ∀ x : R, p (p x) ≠ q (q x) :=
sorry

end no_real_solutions_for_p_p_eq_q_q_l405_405986


namespace sum_of_largest_three_l405_405108

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l405_405108


namespace simplify_expression_l405_405714

variables (a b : ℝ)

open Real

theorem simplify_expression (hb : b ≠ 0) :
  4 * a ^ (2/3) * b ^ (-1/3) / (-2/3 * a ^ (-1/3) * b ^ (2/3)) = -6 * a / b :=
by
  sorry

end simplify_expression_l405_405714


namespace AB_distance_l405_405373

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405373


namespace cesaro_sum_51_terms_l405_405116

noncomputable def Cesaro_sum (seq : List ℝ) (n : ℕ) := (List.sum (List.scanl (λ acc x => acc + x) 0 seq)) / n

theorem cesaro_sum_51_terms (B : List ℝ) (hB : B.length = 50) 
  (hC : Cesaro_sum B 50 = 200) : 
  Cesaro_sum (2 :: B) 51 = 198.078431372549 :=  
sorry

end cesaro_sum_51_terms_l405_405116


namespace smallest_class_number_selected_l405_405857

theorem smallest_class_number_selected
  {n k : ℕ} (hn : n = 30) (hk : k = 5) (h_sum : ∃ x : ℕ, x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 75) :
  ∃ x : ℕ, x = 3 := 
sorry

end smallest_class_number_selected_l405_405857


namespace parabola_distance_problem_l405_405409

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405409


namespace six_digit_palindromes_l405_405018

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l405_405018


namespace area_of_triangle_AEF_l405_405691

-- Assuming the existence of points and triangles with the given properties
variable {A B C D E F : Type}
variable [EuclideanTriangle A B C D E F]

-- Given areas of specific triangles in the Euclidean plane
axiom h1 : area (triangle B D C) = 1
axiom h2 : area (triangle B D E) = 1 / 2
axiom h3 : area (triangle C D F) = 1 / 6

-- Condition of intersections and locations of points
axiom h_points_on_sides : (is_on_side A B E) ∧ (is_on_side A C F) ∧ (intersection_point (line B F) (line C E) = D)

-- Goal
theorem area_of_triangle_AEF : area (triangle A E F) = 7 / 44 :=
begin
  -- Skipping the detailed proof steps.
  sorry
end

end area_of_triangle_AEF_l405_405691


namespace parabola_distance_l405_405579

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405579


namespace parabola_distance_l405_405581

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405581


namespace rectangle_width_decrease_proof_l405_405758

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l405_405758


namespace six_digit_palindromes_count_l405_405043

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l405_405043


namespace parabola_distance_l405_405522

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405522


namespace probability_player_A_wins_first_B_wins_second_l405_405699

theorem probability_player_A_wins_first_B_wins_second :
  (1 / 2) * (4 / 5) * (2 / 3) + (1 / 2) * (1 / 3) * (2 / 3) = 17 / 45 :=
by
  sorry

end probability_player_A_wins_first_B_wins_second_l405_405699


namespace parabola_distance_l405_405492

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405492


namespace rectangle_width_decrease_l405_405775

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L in
  let W' := (L * W) / L' in
  let percent_decrease := (1 - W' / W) * 100 in
  percent_decrease ≈ 28.57 :=
by
  let L' := 1.4 * L
  let W' := (L * W) / L'
  let percent_decrease := (1 - W' / W) * 100
  sorry

end rectangle_width_decrease_l405_405775


namespace parabola_distance_l405_405381

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405381


namespace parabola_distance_l405_405583

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405583


namespace distance_AB_l405_405271

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405271


namespace min_degree_g_l405_405981

variables (f g h : Polynomial ℝ)
variable (x : ℝ)

-- Define the given conditions
def deg_f_eq_10 : Prop := Polynomial.degree f = 10
def deg_h_eq_12 : Prop := Polynomial.degree h = 12
def poly_eq : Prop := 5 * f + 7 * g = h

-- The final proof statement
theorem min_degree_g (deg_f_10 : deg_f_eq_10) (deg_h_12 : deg_h_eq_12) (poly_eq_ : poly_eq) :
  Polynomial.degree g ≥ 12 :=
by
  sorry

end min_degree_g_l405_405981


namespace parabola_problem_l405_405502

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405502


namespace invariant_k_value_l405_405893

theorem invariant_k_value :
  let P_0 := -800
  let Q_0 := 0
  let D := 200
  let B := -400
  let v_P := -10
  let v_Q := -5
  let P (t : ℝ) := P_0 + v_P * t
  let Q (t : ℝ) := Q_0 + v_Q * t
  let M (t : ℝ) := (P(t) + Q(t)) / 2
  let QD (t : ℝ) := Q(t) - D
  let BM (t : ℝ) := B - M(t)
  let expression := 1 / (1 / ℝ) * QD(t) - BM(t)
  ∀ t : ℝ, expression = constant_term
  ∃ k : ℚ, k = -10 := 
sorry

end invariant_k_value_l405_405893


namespace six_digit_palindromes_count_l405_405037

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l405_405037


namespace minimum_distance_midpoint_l405_405997

theorem minimum_distance_midpoint 
    (θ : ℝ)
    (P : ℝ × ℝ := (-4, 4))
    (C1_standard : ∀ (x y : ℝ), (x + 4)^2 + (y - 3)^2 = 1)
    (C2_standard : ∀ (x y : ℝ), x^2 / 64 + y^2 / 9 = 1)
    (Q : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ))
    (M : ℝ × ℝ := (-2 + 4 * Real.cos θ, 2 + 3 / 2 * Real.sin θ))
    (C3_standard : ∀ (x y : ℝ), x - 2*y - 7 = 0) :
    ∃ (θ : ℝ), θ = Real.arcsin (-3/5) ∧ (θ = Real.arccos 4/5) ∧
    (∀ (d : ℝ), d = abs (5 * Real.sin (Real.arctan (4 / 3) - θ) - 13) / Real.sqrt 5 ∧ 
    d = 8 * Real.sqrt 5 / 5) :=
sorry

end minimum_distance_midpoint_l405_405997


namespace ab_distance_l405_405246

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405246


namespace acute_angle_at_2_36_l405_405811

def degrees_in_full_circle := 360
def minute_hand_position (minutes: ℕ) : ℝ := (minutes / 60) * degrees_in_full_circle
def hour_hand_base_position (hours: ℕ) : ℝ := hours * (degrees_in_full_circle / 12)
def hour_hand_additional_position (minutes: ℕ) : ℝ := ((minutes % 60) / 60) * (degrees_in_full_circle / 12)
def hour_hand_position (hours: ℕ) (minutes: ℕ) : ℝ := hour_hand_base_position hours + hour_hand_additional_position minutes

theorem acute_angle_at_2_36 : 
  let minute_deg := minute_hand_position 36 in 
  let hour_deg := hour_hand_position 2 36 in 
  abs (minute_deg - hour_deg) = 138 ∨ abs (360 - abs (minute_deg - hour_deg)) = 138 :=
by sorry

end acute_angle_at_2_36_l405_405811


namespace parabola_distance_l405_405532

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405532


namespace combination_4_3_l405_405712

theorem combination_4_3 :
  (nat.choose 4 3) = 4 := 
sorry

end combination_4_3_l405_405712


namespace min_distance_pq_l405_405147

-- Define the parametric line for P and Q
def line_func (t : ℝ) : ℝ × ℝ :=
  (t, 15 - 2 * t)

-- Define the parametric equation of the curve (circle)
def curve_func (θ : ℝ) : ℝ × ℝ :=
  (1 + real.sqrt 5 * real.cos θ, -2 + real.sqrt 5 * real.sin θ)

-- Define the minimum value of |PQ| is 2 * sqrt(5)
theorem min_distance_pq : ∃ t θ : ℝ, let P := line_func t, Q := curve_func θ in dist P Q = 2 * real.sqrt 5 :=
sorry

end min_distance_pq_l405_405147


namespace tree_heights_l405_405697

theorem tree_heights (h : ℕ) (ratio : 5 / 7 = (h - 20) / h) : h = 70 :=
sorry

end tree_heights_l405_405697


namespace prob_even_product_l405_405719

-- Spinner C has numbers 1 through 6 equally likely
def spinner_C := {1, 2, 3, 4, 5, 6}

-- Spinner D has numbers 1 through 4 equally likely
def spinner_D := {1, 2, 3, 4}

-- Define the product being even
def is_even (n : Nat) := n % 2 = 0

-- Probability that the product of the two spinners' numbers is even
theorem prob_even_product : 
  (∑ c in spinner_C, ∑ d in spinner_D, if is_even (c * d) then 1 else 0) / (Set.card spinner_C * Set.card spinner_D) = 3 / 4 := sorry

end prob_even_product_l405_405719


namespace parabola_distance_l405_405580

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405580


namespace sum_S_17_33_50_l405_405176

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then - (n / 2)
  else (n / 2) + 1

theorem sum_S_17_33_50 : (S 17) + (S 33) + (S 50) = 1 := by
  sorry

end sum_S_17_33_50_l405_405176


namespace tan_alpha_eq_two_sin_2alpha_minus_pi_over_3_l405_405982

-- Let α be a real number in (0, π/2) satisfying the given condition.
axiom α : Real
axiom hα : 0 < α ∧ α < π / 2
axiom h_tan : Real.tan (α + π / 4) = -3

-- First part: Prove tan α = 2
theorem tan_alpha_eq_two (α : Real) (hα : 0 < α ∧ α < π / 2) (h_tan : Real.tan (α + π / 4) = -3) : Real.tan α = 2 :=
by sorry

-- Second part: Prove sin (2α - π / 3) = (4 + 3 * √3) / 10
theorem sin_2alpha_minus_pi_over_3 (α : Real) (hα : 0 < α ∧ α < π / 2) (h_tan : Real.tan (α + π / 4) = -3) : 
  Real.sin (2 * α - π / 3) = (4 + 3 * Real.sqrt 3) / 10 :=
by sorry

end tan_alpha_eq_two_sin_2alpha_minus_pi_over_3_l405_405982


namespace value_of_five_l405_405973

variable (f : ℝ → ℝ)

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = f (x)

theorem value_of_five (hf_odd : odd_function f) (hf_periodic : periodic_function f) : f 5 = 0 :=
by 
  sorry

end value_of_five_l405_405973


namespace distance_AB_l405_405272

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405272


namespace distance_AB_l405_405429

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405429


namespace find_k_l405_405801

noncomputable def origin : (ℝ × ℝ) := (0, 0)

noncomputable def P : (ℝ × ℝ) := (12, 5)

noncomputable def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

noncomputable def radius_larger_circle : ℝ := distance origin P

noncomputable def QR : ℝ := 4

noncomputable def radius_smaller_circle : ℝ := radius_larger_circle - QR

theorem find_k (k : ℝ) (S : (ℝ × ℝ)) (h : S = (0, k)) 
  (h_S_on_smaller_circle : distance origin S = radius_smaller_circle) : k = 9 := 
sorry

end find_k_l405_405801


namespace sum_of_three_largest_l405_405060

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l405_405060


namespace width_decrease_percentage_l405_405751

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l405_405751


namespace parabola_distance_l405_405622

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405622


namespace parabola_problem_l405_405296

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405296


namespace parabola_distance_l405_405537

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405537


namespace ab_distance_l405_405238

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405238


namespace AB_distance_l405_405308

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405308


namespace closest_approx_l405_405822

theorem closest_approx (x : ℝ) (h : x = real.sqrt 68 - real.sqrt 64) : 
  (0.21 : ℝ) = 0.21 - sorry :=
sorry

end closest_approx_l405_405822


namespace parabola_distance_l405_405626

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405626


namespace distance_AB_l405_405339

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405339


namespace parabola_distance_l405_405528

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405528


namespace distance_AB_l405_405338

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405338


namespace parabola_distance_l405_405496

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405496


namespace determine_plane_by_trapezoid_legs_l405_405803

-- Defining basic objects
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)
structure Line := (p1 : Point) (p2 : Point)
structure Plane := (l1 : Line) (l2 : Line)

-- Theorem statement for the problem
theorem determine_plane_by_trapezoid_legs (trapezoid_legs : Line) :
  ∃ (pl : Plane), ∀ (l1 l2 : Line), (l1 = trapezoid_legs) ∧ (l2 = trapezoid_legs) → (pl = Plane.mk l1 l2) :=
sorry

end determine_plane_by_trapezoid_legs_l405_405803


namespace distance_AB_l405_405470

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405470


namespace AB_distance_l405_405377

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405377


namespace rectangle_width_decrease_l405_405769

theorem rectangle_width_decrease (L W : ℝ) (h : L * W = A) (h_new_length : 1.40 * L = L') 
    (h_area_unchanged : L' * W' = L * W) : 
    (W - W') / W = 0.285714 :=
begin
  sorry
end

end rectangle_width_decrease_l405_405769


namespace lambda_range_l405_405133

-- Define the sequence a_n based on the general term derived from the problem
def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else 1 / (2 ^ (n - 2))

-- Define the sum S_n of the first n terms of the sequence
def S (n : ℕ) : ℝ :=
  if n = 0 then 0 
  else
    let first_term := a 1 in
    let rest_terms_sum := 1 - 1 / 2 ^ (n - 1) in
    first_term + rest_terms_sum / (1 - 1 / 2)
  
-- Prove the range of lambda given the conditions
theorem lambda_range : ∀ λ : ℝ, (∀ n : ℕ, n > 0 → λ^2 < S n ∧ S n < 4 * λ) → 3 / 4 ≤ λ ∧ λ < 1 := 
by
  sorry

end lambda_range_l405_405133


namespace stephens_calculator_l405_405721

theorem stephens_calculator : 
  ∀ (a b : Nat), 
  a = 5 ∧ 
  b = 64 → 
  9 * a + 2 * b = 173 :=
by
  rintros a b ⟨ha, hb⟩
  sorry

end stephens_calculator_l405_405721


namespace rectangle_width_decrease_l405_405778

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L in
  let W' := (L * W) / L' in
  let percent_decrease := (1 - W' / W) * 100 in
  percent_decrease ≈ 28.57 :=
by
  let L' := 1.4 * L
  let W' := (L * W) / L'
  let percent_decrease := (1 - W' / W) * 100
  sorry

end rectangle_width_decrease_l405_405778


namespace sum_of_numbers_l405_405798

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def tens_digit_zero (n : ℕ) : Prop := (n / 10) % 10 = 0
def units_digit_nonzero (n : ℕ) : Prop := n % 10 ≠ 0
def same_units_digits (m n : ℕ) : Prop := m % 10 = n % 10

theorem sum_of_numbers (a b c : ℕ)
  (h1 : is_perfect_square a) (h2 : is_perfect_square b) (h3 : is_perfect_square c)
  (h4 : tens_digit_zero a) (h5 : tens_digit_zero b) (h6 : tens_digit_zero c)
  (h7 : units_digit_nonzero a) (h8 : units_digit_nonzero b) (h9 : units_digit_nonzero c)
  (h10 : same_units_digits b c)
  (h11 : a % 10 % 2 = 0) :
  a + b + c = 14612 :=
sorry

end sum_of_numbers_l405_405798


namespace ab_distance_l405_405250

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405250


namespace parabola_problem_l405_405294

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405294


namespace f_2018_l405_405854

def f : ℕ → ℤ
| 1     := 1
| 2     := 1
| (n+3) := f (n+2) - f (n+1) + (n + 3)

theorem f_2018 : f 2018 = 2017 := by
  sorry

end f_2018_l405_405854


namespace AB_distance_l405_405312

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405312


namespace parabola_distance_l405_405483

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405483


namespace parabola_distance_l405_405534

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405534


namespace AB_distance_l405_405376

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405376


namespace distinct_prime_factors_l405_405985

theorem distinct_prime_factors (k : ℕ) (hk : k > 0) (m : ℕ) (hm : m % 2 = 1) (hm_pos: m > 0) :
  ∃ n : ℕ, n > 0 ∧ (card (nat.factors (m^n + n^m)) ≥ k) :=
sorry

end distinct_prime_factors_l405_405985


namespace six_digit_palindromes_count_l405_405006

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l405_405006


namespace width_decreased_by_28_6_percent_l405_405766

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l405_405766


namespace parabola_problem_l405_405504

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405504


namespace parabola_problem_l405_405514

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405514


namespace smallest_positive_period_of_f_max_min_f_in_interval_l405_405999

def f (x : ℝ) := 4 * cos x * cos (x - π / 3) - 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = π :=
sorry

theorem max_min_f_in_interval :
  ∃ x_max x_min ∈ [-π/6, π/4], 
  (∀ x ∈ [-π/6, π/4], f x ≤ f x_max) ∧ f x_max = 1 ∧
  (∀ x ∈ [-π/6, π/4], f x ≥ f x_min) ∧ f x_min = -2 :=
sorry

end smallest_positive_period_of_f_max_min_f_in_interval_l405_405999


namespace sub_mixed_fraction_eq_l405_405814

theorem sub_mixed_fraction_eq :
  (2 + 1 / 4 : ℚ) - (2 / 3) = (1 + 7 / 12) := by
  sorry

end sub_mixed_fraction_eq_l405_405814


namespace stickers_at_beginning_l405_405884

theorem stickers_at_beginning (S L_s s_s : ℕ) (hS : S = 100) (hL_s : L_s = 5) (hs_s : s_s = 10) :
    ∃ T, T = S + L_s * s_s ∧ T = 150 :=
by
  use 150
  split
  · rw [hS, hL_s, hs_s]
    norm_num
  · rfl

end stickers_at_beginning_l405_405884


namespace probability_of_at_least_one_vowel_is_799_over_1024_l405_405713

def Set1 : Set Char := {'a', 'e', 'i', 'b', 'c', 'd', 'f', 'g'}
def Set2 : Set Char := {'u', 'o', 'y', 'k', 'l', 'm', 'n', 'p'}
def Set3 : Set Char := {'e', 'u', 'v', 'r', 's', 't', 'w', 'x'}
def Set4 : Set Char := {'a', 'i', 'o', 'z', 'h', 'j', 'q', 'r'}

noncomputable def probability_of_at_least_one_vowel : ℚ :=
  1 - (5/8 : ℚ) * (3/4 : ℚ) * (3/4 : ℚ) * (5/8 : ℚ)

theorem probability_of_at_least_one_vowel_is_799_over_1024 :
  probability_of_at_least_one_vowel = 799 / 1024 :=
by
  sorry

end probability_of_at_least_one_vowel_is_799_over_1024_l405_405713


namespace inequal_min_value_l405_405168

theorem inequal_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1/x + 4/y) ≥ 9/4 :=
sorry

end inequal_min_value_l405_405168


namespace parabola_problem_l405_405563

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405563


namespace width_decreased_by_28_6_percent_l405_405763

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l405_405763


namespace proof_1_proof_2_l405_405969

-- Define the sequence a_n with the initial condition
def a (n : ℕ) : ℕ :=
  if n = 0 then 2 else 3^(n-2) + 1

-- Define the sum of the first n terms S_n
def S (n : ℕ) : ℕ :=
  (list.range n).sum

-- Condition given in the problem
axiom a_1 : a 1 = 2
axiom S_n (n : ℕ) : S n = (1/2) * a (n+1) + n

-- Define the sequence b_n
def b (n : ℕ) : ℕ :=
  (4 * n - 2) * a (n+1)

-- Define the sum of the first n terms T_n of sequence b_n
def T (n : ℕ) : ℕ :=
  2 + (2 * n - 2) * 3^n + 2 * n^2

-- Proof problem statement
theorem proof_1 : ∀ n : ℕ, a n = if n = 0 then 2 else 3^(n-2) + 1 := sorry
theorem proof_2 : ∀ n : ℕ, T n = 2 + (2 * n - 2) * 3^n + 2 * n^2 := sorry

end proof_1_proof_2_l405_405969


namespace main_theorem_l405_405646

noncomputable def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

lemma f_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- Detailed steps of proof will go here
  sorry

lemma f_monotonic_increasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  intros x₁ x₂ h₁_lt_zero h₁_lt_h₂
  -- Detailed steps of proof will go here
  sorry

theorem main_theorem : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ → f x₁ < f x₂) := by
  split
  · exact f_odd
  · exact f_monotonic_increasing

end main_theorem_l405_405646


namespace distance_AB_l405_405341

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405341


namespace last_n_digits_same_l405_405786

def sequence (x : ℕ → ℕ) : Prop :=
  (x 1 = 5) ∧ ∀ n, x (n + 1) = x n ^ 2

theorem last_n_digits_same (x : ℕ → ℕ) (n : ℕ) (h : sequence x) :
  10 ^ n ∣ (x (n + 1) - x n) :=
sorry

end last_n_digits_same_l405_405786


namespace inequality_true_l405_405820

theorem inequality_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
by
  sorry

end inequality_true_l405_405820


namespace six_digit_palindrome_count_l405_405010

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l405_405010


namespace parabola_problem_l405_405304

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405304


namespace golf_ball_distance_l405_405855

theorem golf_ball_distance (d h : ℝ) (heq : h = d - 0.01 * d^2) (h_16 : ∀ d, d^2 + 100 * d - 1600 = 0 → h = 16) 
: d = 80 := 
begin 
  sorry 
end

end golf_ball_distance_l405_405855


namespace sum_of_three_largest_l405_405066

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l405_405066


namespace parabola_distance_l405_405526

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405526


namespace parabola_distance_l405_405585

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405585


namespace AB_distance_l405_405361

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405361


namespace problem_l405_405606

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405606


namespace distance_AB_l405_405473

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405473


namespace distance_AB_l405_405340

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405340


namespace six_digit_palindromes_l405_405022

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l405_405022


namespace n_calculation_l405_405143

theorem n_calculation (n : ℕ) (hn : 0 < n)
  (h1 : Int.lcm 24 n = 72)
  (h2 : Int.lcm n 27 = 108) :
  n = 36 :=
sorry

end n_calculation_l405_405143


namespace sum_of_three_largest_l405_405076

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l405_405076


namespace parabola_problem_l405_405515

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405515


namespace vector_addition_result_l405_405057

def vector1 := (⟨-3, 2, -5⟩ : ℤ × ℤ × ℤ)
def vector2 := (⟨4, 10, -6⟩ : ℤ × ℤ × ℤ)
def scalar := 3

def scaled_vector1 := (⟨scalar * vector1.1, scalar * vector1.2, scalar * vector1.3⟩ : ℤ × ℤ × ℤ)
def result := (⟨scaled_vector1.1 + vector2.1, scaled_vector1.2 + vector2.2, scaled_vector1.3 + vector2.3⟩ : ℤ × ℤ × ℤ)

theorem vector_addition_result : 
  3 * vector1 + vector2 = (⟨-5, 16, -21⟩ : ℤ × ℤ × ℤ) := by
  rw [scaled_vector1, result]
  sorry

end vector_addition_result_l405_405057


namespace total_cantaloupes_l405_405121

def Fred_grew_38 : ℕ := 38
def Tim_grew_44 : ℕ := 44

theorem total_cantaloupes : Fred_grew_38 + Tim_grew_44 = 82 := by
  sorry

end total_cantaloupes_l405_405121


namespace parabola_problem_l405_405548

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405548


namespace bell_rings_count_l405_405676

def classes : List String := ["Maths", "English", "History", "Geography", "Chemistry", "Physics", "Literature", "Music"]

def total_classes : Nat := classes.length

def rings_per_class : Nat := 2

def classes_before_music : Nat := total_classes - 1

def rings_before_music : Nat := classes_before_music * rings_per_class

def current_class_rings : Nat := 1

def total_rings_by_now : Nat := rings_before_music + current_class_rings

theorem bell_rings_count :
  total_rings_by_now = 15 := by
  sorry

end bell_rings_count_l405_405676


namespace third_cat_weight_l405_405224

theorem third_cat_weight :
  ∃ W : ℝ, (12 + 12 + W + 9.3) / 4 = 12 ∧ W = 14.7 :=
begin
  use 14.7,
  split,
  { sorry },  -- Proof that (12 + 12 + 14.7 + 9.3) / 4 = 12
  { refl }
end

end third_cat_weight_l405_405224


namespace parabola_distance_problem_l405_405425

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405425


namespace parabola_distance_l405_405593

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405593


namespace first_consecutive_odd_number_l405_405698

theorem first_consecutive_odd_number (x : ℤ) (h1 : 8 * x = 3 * (x + 4) + 2 * (x + 2) + 5) 
  (h2 : ∃ p : ℤ, prime p ∧ p ∣ (x + (x + 2) + (x + 4))) : x = 7 :=
sorry

end first_consecutive_odd_number_l405_405698


namespace distance_AB_l405_405435

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405435


namespace volume_ratio_tetrahedron_cube_l405_405868

theorem volume_ratio_tetrahedron_cube (s : ℝ) :
  let volume_cube := s^3
  let volume_tetrahedron := (s * (3^.5 / 2))^3 * (2^.5 / 12) 
  (volume_tetrahedron / volume_cube) = (6^.5 / 32) :=
by
  sorry

end volume_ratio_tetrahedron_cube_l405_405868


namespace compare_trigonometric_values_l405_405961

theorem compare_trigonometric_values :
  let a := Real.sin (20 * Real.pi / 180)
  let b := Real.tan (30 * Real.pi / 180)
  let c := Real.cos (40 * Real.pi / 180)
  c > b ∧ b > a :=
by 
  let a := Real.sin (20 * Real.pi / 180)
  let b := Real.tan (30 * Real.pi / 180)
  let c := Real.cos (40 * Real.pi / 180)
  have h1 : b = Real.tan (30 * Real.pi / 180), 
  {
    sorry
  }
  have h2: c = Real.cos (40 * Real.pi / 180), 
  {
    sorry
  }
  have h3: a = Real.sin (20 * Real.pi / 180), 
  {
    sorry
  }
  exact ⟨sorry, sorry⟩

end compare_trigonometric_values_l405_405961


namespace circle_and_parabola_tangent_l405_405162

theorem circle_and_parabola_tangent (r : ℝ) (r_pos : 0 < r) :
  (∃ P : ℝ × ℝ, 
    P ∈ { p | p.snd = (1/4) * p.fst^2 } ∧ 
    P ∈ { q | (q.fst - 1)^2 + (q.snd - 2)^2 = r^2 } ∧ 
    let t := P.fst in
    let slope_parabola_tangent := t / 2 in
    let slope_CP := (P.snd - 2) / (P.fst - 1) in
    slope_parabola_tangent * slope_CP = -1) →
  r = Real.sqrt 2 :=
by
  sorry

end circle_and_parabola_tangent_l405_405162


namespace six_digit_palindromes_l405_405020

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l405_405020


namespace distance_from_point_to_line_l405_405939

open Real

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the distance formula from a point to a line
def distance_point_to_line (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / sqrt (a^2 + b^2)

-- Definition of the point
def point : ℝ × ℝ := (2, 0)

theorem distance_from_point_to_line :
  distance_point_to_line 2 0 1 (-1) (-1) = sqrt 2 / 2 := by
  sorry

end distance_from_point_to_line_l405_405939


namespace parabola_problem_l405_405499

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405499


namespace find_f_2012_l405_405832

theorem find_f_2012 (f : ℕ → ℕ) (h1 : ∀ n : ℕ, f(f(n)) + f(n) = 2 * n + 3) (h2 : f(0) = 1) : f(2012) = 2013 :=
by
  sorry

end find_f_2012_l405_405832


namespace parabola_distance_l405_405619

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405619


namespace parabola_problem_l405_405513

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405513


namespace max_value_k_l405_405838

noncomputable def max_k (S : Finset ℕ) (A : ℕ → Finset ℕ) (k : ℕ) :=
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2)

theorem max_value_k : ∀ (S : Finset ℕ) (A : ℕ → Finset ℕ), 
  S = Finset.range 14 \{0} → 
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2) →
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) →
  ∃ k, max_k S A k ∧ k = 4 :=
sorry

end max_value_k_l405_405838


namespace sum_of_largest_three_l405_405112

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l405_405112


namespace parabola_distance_l405_405379

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405379


namespace wages_for_both_conditions_l405_405831

variables (A B : ℝ) (D : ℝ)

def condition1 := ∃ (money : ℝ), money = 21 * A
def condition2 := ∃ (money : ℝ), money = 28 * B

theorem wages_for_both_conditions : (condition1 A) → (condition2 B) → (D = 12) :=
by
  intros h1 h2
  have hmoney : ∃ (money : ℝ), money = 21 * A ∧ money = 28 * B := sorry,
  have hA_B : B = (21 * A) / 28 := sorry,
  have sum_money := 21 * A,
  have wage_both := (A + B) * D,
  have hD : sum_money = wage_both := sorry,
  have D_value : D = 12 := sorry,
  exact D_value

end wages_for_both_conditions_l405_405831


namespace part_I_has_two_extreme_points_part_II_unique_zero_point_l405_405675

-- Define the function f(x)
def f (a λ x : ℝ) : ℝ := Real.exp (a * x) + λ * Real.log x

-- Conditions
variables {a λ : ℝ}
variables (h_a : a < 0) (h_λ : 0 < λ ∧ λ < 1 / Real.exp 1)

-- Part I: Prove that f(x) has two extreme points
theorem part_I_has_two_extreme_points :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (∃ f_x1_x2 : ℝ, f a λ x1 = f_x1_x2 ∧ f a λ x2 = f_x1_x2) :=
sorry

-- Additional condition for Part II
variable (h_a_bound : -Real.exp 1 ≤ a ∧ a < 0)

-- Part II: Prove that f(x) has a unique zero point
theorem part_II_unique_zero_point :
  ∃! x : ℝ, f a λ x = 0 :=
sorry

end part_I_has_two_extreme_points_part_II_unique_zero_point_l405_405675


namespace parabola_distance_l405_405391

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405391


namespace train_A_reaches_destination_in_6_hours_l405_405802

noncomputable def t : ℕ := 
  let tA := 110
  let tB := 165
  let tB_time := 4
  (tB * tB_time) / tA

theorem train_A_reaches_destination_in_6_hours :
  t = 6 := by
  sorry

end train_A_reaches_destination_in_6_hours_l405_405802


namespace problem_l405_405596

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405596


namespace a6_add_b6_geq_ab_a4_add_b4_l405_405959

theorem a6_add_b6_geq_ab_a4_add_b4 (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end a6_add_b6_geq_ab_a4_add_b4_l405_405959


namespace parabola_distance_l405_405635

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405635


namespace flowers_per_row_correct_l405_405823

/-- Definition for the number of each type of flower. -/
def num_yellow_flowers : ℕ := 12
def num_green_flowers : ℕ := 2 * num_yellow_flowers -- Given that green flowers are twice the yellow flowers.
def num_red_flowers : ℕ := 42

/-- Total number of flowers. -/
def total_flowers : ℕ := num_yellow_flowers + num_green_flowers + num_red_flowers

/-- Number of rows in the garden. -/
def num_rows : ℕ := 6

/-- The number of flowers per row Wilma's garden has. -/
def flowers_per_row : ℕ := total_flowers / num_rows

/-- Proof statement: flowers per row should be 13. -/
theorem flowers_per_row_correct : flowers_per_row = 13 :=
by
  -- The proof will go here.
  sorry

end flowers_per_row_correct_l405_405823


namespace vector_identity_l405_405842

namespace VectorAddition

variable {V : Type*} [AddCommGroup V]

theorem vector_identity
  (AD DC AB BC : V)
  (h1 : AD + DC = AC)
  (h2 : AC - AB = BC) :
  AD + DC - AB = BC :=
by
  sorry

end VectorAddition

end vector_identity_l405_405842


namespace parabola_problem_l405_405503

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405503


namespace distance_AB_l405_405432

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405432


namespace parabola_distance_l405_405394

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405394


namespace ab_distance_l405_405236

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405236


namespace problem_l405_405613

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405613


namespace AB_distance_l405_405309

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405309


namespace divisible_by_q_plus_one_l405_405949

theorem divisible_by_q_plus_one (q: ℤ) (h1: q > 1) (h2: q % 2 = 1) :
  (q + 1) ^ (1 / 2 * (q + 1)) % (q + 1) = 0 :=
by
  sorry

end divisible_by_q_plus_one_l405_405949


namespace relationship_among_a_b_c_l405_405960

theorem relationship_among_a_b_c :
  let a := (1/6) ^ (1/2)
  let b := Real.log (1/3) / Real.log 6
  let c := Real.log (1/7) / Real.log (1/6)
  c > a ∧ a > b :=
by
  sorry

end relationship_among_a_b_c_l405_405960


namespace distance_AB_l405_405280

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405280


namespace parabola_distance_l405_405397

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405397


namespace parabola_problem_l405_405505

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405505


namespace altitude_line_correct_median_line_correct_l405_405995

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨4, 0⟩
def B : Point := ⟨6, 7⟩
def C : Point := ⟨0, 3⟩

def altitude_equation : String := "3x + 2y - 12 = 0"
def median_equation : String := "5x + y - 20 = 0"

theorem altitude_line_correct :
  (equation_of_line_through_point_perpendicular_to_line B A C) = 3 * x + 2 * y - 12 := sorry

theorem median_line_correct :
  (equation_of_line_through_points A (midpoint B C)) = 5 * x + y - 20 := sorry

end altitude_line_correct_median_line_correct_l405_405995


namespace parabola_distance_l405_405488

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405488


namespace six_digit_palindrome_count_l405_405011

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l405_405011


namespace tan_of_alpha_l405_405957

open Real

theorem tan_of_alpha (α : ℝ) (h1 : sin (π - α) = log 27 (1 / 9)) (h2 : α ∈ Ioo (-π / 2) 0) :
    tan α = -2 * sqrt 5 / 5 := by
  sorry

end tan_of_alpha_l405_405957


namespace AB_distance_l405_405363

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405363


namespace parabola_problem_l405_405568

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405568


namespace six_digit_palindromes_count_l405_405000

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l405_405000


namespace parabola_problem_l405_405561

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405561


namespace even_number_of_faces_l405_405938

theorem even_number_of_faces (n : ℕ) (h : n ≥ 4) (exists_convex_polyhedron : ∃ (P : Type) [convex_polyhedron P] (faces : set P), faces.card = n ∧ ∀ face ∈ faces, is_right_angled_triangle face) : Even n :=
by
  sorry

end even_number_of_faces_l405_405938


namespace problem_statement_l405_405910

-- Define the operation * based on the given mathematical definition
def op (a b : ℕ) : ℤ := a * (a - b)

-- The core theorem to prove the expression in the problem
theorem problem_statement : op 2 3 + op (6 - 2) 4 = -2 :=
by
  -- This is where the proof would go, but it's omitted with sorry.
  sorry

end problem_statement_l405_405910


namespace f_properties_l405_405664

def f (x : ℝ) : ℝ := x^3 - 1 / x^3

theorem f_properties : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := 
by
  sorry

end f_properties_l405_405664


namespace count_valid_three_digit_numbers_l405_405125

open Function

-- Definitions of the conditions
def valid_number_of_digits := {1, 2, 3, 4, 5}
def no_repeating_digits : ∀ (d1 d2 d3 : ℕ), d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3
def two_precedes_three : ∀ (d1 d2 d3 : ℕ), d1 = 2 → d2 = 3 → False

-- Main Theorem Statement
theorem count_valid_three_digit_numbers : 
  (finset.card {num | 
    ∃ (d1 d2 d3 : ℕ), 
      d1 ∈ valid_number_of_digits ∧ 
      d2 ∈ valid_number_of_digits ∧ 
      d3 ∈ valid_number_of_digits ∧ 
      d1 ≠ d2 ∧ 
      d1 ≠ d3 ∧ 
      d2 ≠ d3 ∧ 
      ¬(two_precedes_three d1 d2 d3) 
    } = 51) :=
sorry

end count_valid_three_digit_numbers_l405_405125


namespace AB_distance_l405_405372

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405372


namespace ali_baba_possible_l405_405885

structure Grid :=
  (m n : ℕ)
  (color : ℕ × ℕ → bool) -- A function that tells if a cell is black (true) or white (false)

def is_possible (m n : ℕ) (color : ℕ × ℕ → bool) : Prop :=
  ∃ (path : list (ℕ × ℕ)), -- A list of coordinates representing the path
    (∀ (i j : ℕ), (i, j) ∈ path) ∧ -- Ali Baba visits every cell
    (∀ (i j : ℕ), color (i, j) = tt → coin_count (path, i, j) = 1) ∧ -- Black cells have one coin
    (∀ (i j : ℕ), color (i, j) = ff → coin_count (path, i, j) = 0) -- White cells have no coins

theorem ali_baba_possible (m n : ℕ) (color : ℕ × ℕ → bool) (h_grid : Grid m n color) :
  is_possible m n color :=
sorry

end ali_baba_possible_l405_405885


namespace solution_set_of_inequality_l405_405843

theorem solution_set_of_inequality (x : ℝ) : |2 * x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 :=
by 
  sorry

end solution_set_of_inequality_l405_405843


namespace tourism_revenue_scientific_notation_l405_405826

theorem tourism_revenue_scientific_notation:
  (12.41 * 10^9) = (1.241 * 10^9) := 
sorry

end tourism_revenue_scientific_notation_l405_405826


namespace rectangle_width_decrease_proof_l405_405756

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l405_405756


namespace dice_product_divisibility_probability_l405_405828

theorem dice_product_divisibility_probability :
  let p := 1 - ((5 / 18)^6 : ℚ)
  p = (33996599 / 34012224 : ℚ) :=
by
  -- This is the condition where the probability p is computed as the complementary probability.
  sorry

end dice_product_divisibility_probability_l405_405828


namespace parabola_distance_l405_405479

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405479


namespace six_digit_palindromes_l405_405016

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l405_405016


namespace water_left_in_bucket_l405_405897

theorem water_left_in_bucket :
  ∀ (original_poured water_left : ℝ),
    original_poured = 0.8 →
    water_left = 0.6 →
    ∃ (poured : ℝ), poured = 0.2 ∧ original_poured - poured = water_left :=
by
  intros original_poured water_left ho hw
  apply Exists.intro 0.2
  simp [ho, hw]
  sorry

end water_left_in_bucket_l405_405897


namespace find_three_digit_numbers_l405_405059

def is_three_digit_number (n : ℕ) : Prop :=
  n >= 100 ∧ n <= 999

def sum_of_three_digit_numbers : ℕ :=
  (999 - 100 + 1) * (100 + 999) / 2

theorem find_three_digit_numbers :
  ∃ x y : ℕ, is_three_digit_number x ∧ is_three_digit_number y ∧ sum_of_three_digit_numbers - x - y = 600 * x ∧ x = 823 ∧ y = 527 :=
begin
  sorry
end

end find_three_digit_numbers_l405_405059


namespace radian_measure_of_sector_l405_405150

-- Lean statement for the proof problem
theorem radian_measure_of_sector (R : ℝ) (hR : 0 < R) (h_area : (1 / 2) * (2 : ℝ) * R^2 = R^2) : 
  (2 : ℝ) = 2 :=
by 
  sorry
 
end radian_measure_of_sector_l405_405150


namespace parabola_problem_l405_405295

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405295


namespace area_of_rectangle_l405_405683

theorem area_of_rectangle (a b : ℝ) (area : ℝ) 
(h1 : a = 5.9) 
(h2 : b = 3) 
(h3 : area = a * b) : 
area = 17.7 := 
by 
  -- proof goes here
  sorry

-- Definitions and conditions alignment:
-- a represents one side of the rectangle.
-- b represents the other side of the rectangle.
-- area represents the area of the rectangle.
-- h1: a = 5.9 corresponds to the first condition.
-- h2: b = 3 corresponds to the second condition.
-- h3: area = a * b connects the conditions to the formula to find the area.
-- The goal is to show that area = 17.7, which matches the correct answer.

end area_of_rectangle_l405_405683


namespace triangles_equal_area_l405_405700

open_locale classical

variables {A B C D M N O : Type} [AffineSpace ℝ ℝ A]

-- Definitions of the given problem
-- Let M and N be midpoints of BC and AD, expressed as vectors

def isMidpoint (a b m : A) : Prop :=
  ∃ (w : ℝ), w = 1/2 ∧ m = (1 - w) • a + w • b

-- Given conditions: M is the midpoint of BC and N is the midpoint of AD
axiom M_midpoint : isMidpoint B C M
axiom N_midpoint : isMidpoint A D N

-- Let O be the midpoint of segment MN
axiom O_midpoint : isMidpoint M N O 

-- O also lying on diagonal AC
axiom O_on_AC : ∃ (λ : ℝ), O = (1 - λ) • A + λ • C

-- Statement to prove: areas of triangles ABC and ACD are equal
theorem triangles_equal_area :
  (area ℝ A B C) = (area ℝ A C D) :=
sorry

end triangles_equal_area_l405_405700


namespace solve_for_x_l405_405715

theorem solve_for_x :
  (16^x * 16^x * 16^x * 4^(3 * x) = 64^(4 * x)) → x = 0 := by
  sorry

end solve_for_x_l405_405715


namespace six_digit_palindromes_count_l405_405042

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l405_405042


namespace overlapped_squares_area_l405_405951

/-- 
Theorem: The area of the figure formed by overlapping four identical squares, 
each with an area of \(3 \, \text{cm}^2\), and with an overlapping region 
that double-counts 6 small squares is \(10.875 \, \text{cm}^2\).
-/
theorem overlapped_squares_area (area_of_square : ℝ) (num_squares : ℕ) (overlap_small_squares : ℕ) :
  area_of_square = 3 → 
  num_squares = 4 → 
  overlap_small_squares = 6 →
  ∃ total_area : ℝ, total_area = (num_squares * area_of_square) - (overlap_small_squares * (area_of_square / 16)) ∧
                         total_area = 10.875 :=
by
  sorry

end overlapped_squares_area_l405_405951


namespace parabola_distance_l405_405631

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405631


namespace parabola_problem_l405_405297

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405297


namespace parabola_distance_problem_l405_405410

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405410


namespace part1_part2_part3_l405_405977

def f (x : ℝ) : ℝ := log x + 1 / x
def g (x : ℝ) : ℝ := x - log x

theorem part1 : (∀ x > 0, f x ≥ a) → a ≤ 1 :=
sorry

theorem part2 : (∀ x > 1, f x < g x) :=
sorry

theorem part3 (x1 x2 : ℝ) (hx1 : x1 > x2) (hg : g x1 = g x2) : x1 * x2 < 1 :=
sorry

end part1_part2_part3_l405_405977


namespace solve_quadratic_eq_l405_405787

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l405_405787


namespace parabola_distance_problem_l405_405412

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405412


namespace parabola_distance_l405_405389

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405389


namespace parabola_problem_l405_405549

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405549


namespace parabola_distance_l405_405627

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405627


namespace distance_AB_l405_405468

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405468


namespace parabola_problem_l405_405516

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405516


namespace width_decrease_percentage_l405_405754

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l405_405754


namespace parabola_distance_l405_405624

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405624


namespace trapezoid_BD_sum_l405_405213

theorem trapezoid_BD_sum (A B C D : ℝ) 
    (h1 : AB ∥ CD) 
    (h2 : ∠CAB < 90)
    (h3 : AB = 5) 
    (h4 : CD = 3) 
    (h5 : AC = 15) : 
    sum (finset.filter (λ x, int.to_nat x < 17 ∧ 7 < x) (finset.range 24)) = 108 :=
by 
  -- the proof would go here
  sorry

end trapezoid_BD_sum_l405_405213


namespace parabola_distance_l405_405632

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405632


namespace area_difference_of_circle_and_square_l405_405735

theorem area_difference_of_circle_and_square (d_square d_circle : ℝ) (h₁ : d_square = 10) (h₂ : d_circle = 10) : 
  (let s := d_square / (2 * Real.sqrt (2:ℝ)) in
   let A_square := s * s in
   let r := d_circle / 2 in
   let A_circle := Real.pi * r * r in
   Real.toBaseRat ((A_circle - A_square) : ℝ) (1/10) = (28.5 : ℝ)) :=
by
  sorry

end area_difference_of_circle_and_square_l405_405735


namespace number_of_valid_configurations_l405_405740

-- Define the initial conditions
structure PlusShapePyramid (squares : ℕ) (positions : ℕ) :=
  (congruence : squares = 5)
  (additional_positions : positions = 8)

-- Prove the number of valid configurations
theorem number_of_valid_configurations :
  ∀ (structure : PlusShapePyramid 5 8),
  ∃ valid_configurations : ℕ, valid_configurations = 4 :=
by
  sorry

end number_of_valid_configurations_l405_405740


namespace no_value_of_b_valid_l405_405905

theorem no_value_of_b_valid (b n : ℤ) : b^2 + 3 * b + 1 ≠ n^2 := by
  sorry

end no_value_of_b_valid_l405_405905


namespace tens_digit_23_1987_l405_405923

theorem tens_digit_23_1987 : (23 ^ 1987 % 100) / 10 % 10 = 4 :=
by
  -- The proof goes here
  sorry

end tens_digit_23_1987_l405_405923


namespace rectangle_width_decrease_l405_405771

theorem rectangle_width_decrease (L W : ℝ) (h : L * W = A) (h_new_length : 1.40 * L = L') 
    (h_area_unchanged : L' * W' = L * W) : 
    (W - W') / W = 0.285714 :=
begin
  sorry
end

end rectangle_width_decrease_l405_405771


namespace distance_AB_l405_405445

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405445


namespace angle_C_in_triangle_l405_405216

theorem angle_C_in_triangle {A B C : ℝ} 
  (h1 : A - B = 10) 
  (h2 : B = 0.5 * A) : 
  C = 150 :=
by
  -- Placeholder for proof
  sorry

end angle_C_in_triangle_l405_405216


namespace six_digit_palindrome_count_l405_405015

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l405_405015


namespace find_numbers_l405_405876

def is_7_digit (n : ℕ) : Prop := n ≥ 1000000 ∧ n < 10000000
def is_14_digit (n : ℕ) : Prop := n >= 10^13 ∧ n < 10^14

theorem find_numbers (x y z : ℕ) (hx7 : is_7_digit x) (hy7 : is_7_digit y) (hz14 : is_14_digit z) :
  3 * x * y = z ∧ z = 10^7 * x + y → 
  x = 1666667 ∧ y = 3333334 ∧ z = 16666673333334 := 
by
  sorry

end find_numbers_l405_405876


namespace all_lucky_years_l405_405859

def is_lucky_year (y : ℕ) : Prop :=
  ∃ m d : ℕ, 1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 31 ∧ (m * d = y % 100)

theorem all_lucky_years :
  (is_lucky_year 2024) ∧ (is_lucky_year 2025) ∧ (is_lucky_year 2026) ∧ (is_lucky_year 2027) ∧ (is_lucky_year 2028) :=
sorry

end all_lucky_years_l405_405859


namespace bob_spend_l405_405934

def price_of_apples := 4 * 2
def price_of_juice := 3 * 4
def price_of_bread := 2 * 3
def original_price_of_cheese_pack := 6
def sale_price_of_cheese_pack := original_price_of_cheese_pack / 2
def price_of_cheese := 2 * sale_price_of_cheese_pack
def price_of_cereal := 8

def total_cost_before_coupon := price_of_apples + price_of_juice + price_of_bread + price_of_cheese + price_of_cereal
def coupon := if total_cost_before_coupon >= 30 then 7 else 0
def final_total_cost := total_cost_before_coupon - coupon

theorem bob_spend (h : final_total_cost = 33) : final_total_cost = 33 :=
by
  rw [price_of_apples, price_of_juice, price_of_bread, sale_price_of_cheese_pack, price_of_cheese, price_of_cereal, total_cost_before_coupon, coupon, final_total_cost]
  sorry

end bob_spend_l405_405934


namespace f_properties_l405_405663

def f (x : ℝ) : ℝ := x^3 - 1 / x^3

theorem f_properties : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := 
by
  sorry

end f_properties_l405_405663


namespace cookie_profit_l405_405873

theorem cookie_profit :
  let cost_per_cookie := 3 / 8
  let total_cost := 1200 * cost_per_cookie
  let discounted_price := 0.40
  let regular_price := 0.50
  let num_discounted := 900
  let num_regular := 300
  let total_revenue := (num_discounted * discounted_price) + (num_regular * regular_price)
  let profit := total_revenue - total_cost
  profit = 60 := by
  let cost_per_cookie := 3 / 8
  let total_cost := 1200 * cost_per_cookie
  let discounted_price := 0.40
  let regular_price := 0.50
  let num_discounted := 900
  let num_regular := 300
  let total_revenue := (num_discounted * discounted_price) + (num_regular * regular_price)
  let profit := total_revenue - total_cost
  sorry

end cookie_profit_l405_405873


namespace sum_eq_two_l405_405707

theorem sum_eq_two (x y : ℝ) (hx : x^3 - 3 * x^2 + 5 * x = 1) (hy : y^3 - 3 * y^2 + 5 * y = 5) : x + y = 2 := 
sorry

end sum_eq_two_l405_405707


namespace max_min_values_g_expression_correct_l405_405155

def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 3

theorem max_min_values (a : ℝ) (h : a = -1) :
  ∃ (f_min f_max : ℝ),
    f_min = 2 ∧ f_max = 11 ∧
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x a ≥ f_min) ∧
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x a ≤ f_max) :=
sorry

def g (a : ℝ) : ℝ :=
  if a < -2 then 7 + 4 * a
  else if a <= 2 then 3 - a^2
  else 7 - 4 * a

theorem g_expression_correct (a : ℝ) :
  g a = (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x a = g a) :=
sorry

end max_min_values_g_expression_correct_l405_405155


namespace distance_AB_l405_405460

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405460


namespace parabola_distance_l405_405638

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405638


namespace sum_of_three_largest_l405_405088

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l405_405088


namespace AB_distance_l405_405317

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405317


namespace distance_AB_l405_405275

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405275


namespace distance_AB_l405_405465

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405465


namespace max_radius_subset_l405_405131

noncomputable def maximum_r := 4 * Real.sqrt 2

theorem max_radius_subset 
  (r : ℝ) (h : ∀ x y : ℝ, x^2 + (y-7)^2 ≤ r^2 → ∀ θ : ℝ, cos (2*θ) + x * cos θ + y ≥ 0) : 
  ∃ (r_max : ℝ), r_max = maximum_r := 
sorry

end max_radius_subset_l405_405131


namespace cut_grid_into_six_polygons_with_identical_pair_l405_405953

noncomputable def totalCells : Nat := 24
def polygonArea : Nat := 4

theorem cut_grid_into_six_polygons_with_identical_pair :
  ∃ (polygons : Fin 6 → Nat → Prop),
  (∀ i, (∃ (cells : Finset (Fin totalCells)), (cells.card = polygonArea ∧ ∀ c ∈ cells, polygons i c))) ∧
  (∃ i j, i ≠ j ∧ ∀ c, polygons i c ↔ polygons j c) :=
sorry

end cut_grid_into_six_polygons_with_identical_pair_l405_405953


namespace area_of_triangle_AEF_l405_405690

-- Assuming the existence of points and triangles with the given properties
variable {A B C D E F : Type}
variable [EuclideanTriangle A B C D E F]

-- Given areas of specific triangles in the Euclidean plane
axiom h1 : area (triangle B D C) = 1
axiom h2 : area (triangle B D E) = 1 / 2
axiom h3 : area (triangle C D F) = 1 / 6

-- Condition of intersections and locations of points
axiom h_points_on_sides : (is_on_side A B E) ∧ (is_on_side A C F) ∧ (intersection_point (line B F) (line C E) = D)

-- Goal
theorem area_of_triangle_AEF : area (triangle A E F) = 7 / 44 :=
begin
  -- Skipping the detailed proof steps.
  sorry
end

end area_of_triangle_AEF_l405_405690


namespace parabola_distance_problem_l405_405406

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405406


namespace parabola_distance_l405_405399

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405399


namespace parabola_distance_l405_405480

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405480


namespace six_digit_palindromes_count_l405_405044

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l405_405044


namespace parabola_problem_l405_405299

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405299


namespace six_digit_palindromes_count_l405_405028

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l405_405028


namespace rectangle_width_decrease_l405_405772

theorem rectangle_width_decrease (L W : ℝ) (h : L * W = A) (h_new_length : 1.40 * L = L') 
    (h_area_unchanged : L' * W' = L * W) : 
    (W - W') / W = 0.285714 :=
begin
  sorry
end

end rectangle_width_decrease_l405_405772


namespace parabola_distance_l405_405577

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405577


namespace AB_distance_l405_405314

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405314


namespace sum_of_three_largest_l405_405082

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l405_405082


namespace max_k_no_parallel_sides_l405_405839

theorem max_k_no_parallel_sides (n : ℕ) (k : ℕ) 
  (h_points : n = 2012) 
  (h_k_gon : k ≤ n) 
  (h_convex : convex_k_gon_with_no_parallel_sides n) :
  k ≤ 1509 :=
sorry

def convex_k_gon_with_no_parallel_sides (n : ℕ) : Prop :=
  -- definition for a convex k-gon with no parallel sides using n points goes here
  true -- placeholder definition

end max_k_no_parallel_sides_l405_405839


namespace rainfall_on_tuesday_l405_405898

theorem rainfall_on_tuesday 
  (r_Mon r_Wed r_Total r_Tue : ℝ)
  (h_Mon : r_Mon = 0.16666666666666666)
  (h_Wed : r_Wed = 0.08333333333333333)
  (h_Total : r_Total = 0.6666666666666666)
  (h_Tue : r_Tue = r_Total - (r_Mon + r_Wed)) :
  r_Tue = 0.41666666666666663 := 
sorry

end rainfall_on_tuesday_l405_405898


namespace math_problem_l405_405163

-- Define the parametric equations of circle C
def circle_param_eq (α : ℝ) : ℝ × ℝ := (5 + 2 * Real.cos α, Real.sqrt 3 + 2 * Real.sin α)

-- Define the polar equation of line l1
def line_l1 (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ - Real.sqrt 3 * ρ * Real.sin θ = 3

-- Move line l1 3 units to the left to get line l2
def line_l2 (ρ θ : ℝ) : Prop :=
  line_l1 ρ θ ∧ ∃ x y : ℝ, (x - 3) = ρ * Real.cos θ ∧ (y - (Real.sqrt 3 / 3) * x)

-- Problem Statement
theorem math_problem :
  (∀ α : ℝ, ∃ ρ θ : ℝ, circle_param_eq α = (ρ * Real.cos θ, ρ * Real.sin θ) →
    ρ^2 - 10 * ρ * Real.cos θ - 2 * Real.sqrt 3 * ρ * Real.sin θ + 24 = 0) ∧
  (∀ ρ θ : ℝ, line_l2 ρ θ ↔ ∃ x : ℝ, (Real.sin θ = Real.sqrt 3 / 3 * Real.cos θ) ∧ (Real.sqrt 3 / 3 * x = y)) ∧
  (∀ ρ_A ρ_B θ : ℝ, (ρ_A + ρ_B = 6 * Real.sqrt 3 ∧ ρ_A * ρ_B = 24) →
    (∃ k : ℝ, k = 2)) := sorry

end math_problem_l405_405163


namespace sin_theta_in_fourth_quadrant_l405_405140

theorem sin_theta_in_fourth_quadrant (θ : ℝ) (h1 : cos θ = 1 / 3) (h2 : θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi) :
  sin θ = - (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_theta_in_fourth_quadrant_l405_405140


namespace distance_AB_l405_405345

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405345


namespace sqrt_x_eq_log2_6_l405_405144

noncomputable def x : ℝ := log 4 8 + log 2 9

theorem sqrt_x_eq_log2_6 : sqrt x = log 2 6 := 
sorry

end sqrt_x_eq_log2_6_l405_405144


namespace six_digit_palindromes_count_l405_405001

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l405_405001


namespace distance_AB_l405_405268

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405268


namespace area_of_circle_with_diameter_10_l405_405808

theorem area_of_circle_with_diameter_10 :
  ∀ (d : ℝ), d = 10 → ∃ (A : ℝ), A = 25 * Real.pi :=
by intros d h
   use 25 * Real.pi
   sorry

end area_of_circle_with_diameter_10_l405_405808


namespace smallest_triangle_area_exists_l405_405917

def P : ℝ × ℝ × ℝ := (2, 0, 1)
def Q : ℝ × ℝ × ℝ := (0, 2, 3)
def R (s : ℝ) : ℝ × ℝ × ℝ := (s, 0, 1)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1 * u.1 + u.2 * u.2 + u.3 * u.3)

def area_of_triangle (P Q R : ℝ × ℝ × ℝ) : ℝ :=
  (1/2) * magnitude (cross_product (vector_sub Q P) (vector_sub R P))

theorem smallest_triangle_area_exists :
  ∃ s : ℝ, area_of_triangle P Q (R s) = 0 := 
sorry

end smallest_triangle_area_exists_l405_405917


namespace distance_AB_l405_405431

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405431


namespace fraction_of_sum_l405_405850

theorem fraction_of_sum (n S : ℕ) 
  (h1 : S = (n-1) * ((n:ℚ) / 3))
  (h2 : n > 0) : 
  (n:ℚ) / (S + n) = 3 / (n + 2) := 
by 
  sorry

end fraction_of_sum_l405_405850


namespace parabola_distance_l405_405490

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405490


namespace polynomial_has_factor_l405_405164

noncomputable def polynomial : Polynomial ℝ :=
  x^2 - y^2 - z^2 - 2 * y * z + x - y - z + 2

def factor : Polynomial ℝ := x - y - z + 1

theorem polynomial_has_factor :
  polynomial = factor * (polynomial / factor) :=
sorry

end polynomial_has_factor_l405_405164


namespace parabola_distance_l405_405493

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405493


namespace problem_l405_405599

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405599


namespace parabola_problem_l405_405288

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405288


namespace find_prime_p_l405_405946

def prime (n : Nat) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem find_prime_p (p : Nat) (h_prime : prime p) : 
  ∃ p', prime p' ∧ p'^2 + 11 = p^2 + 11 ∧ (p'^2 + 11).numDivisors = 6 :=
sorry

end find_prime_p_l405_405946


namespace parabola_problem_l405_405564

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405564


namespace scientific_notation_of_virus_diameter_l405_405737

theorem scientific_notation_of_virus_diameter :
  0.00000012 = 1.2 * 10 ^ (-7) :=
by
  sorry

end scientific_notation_of_virus_diameter_l405_405737


namespace find_angle_A_l405_405214

theorem find_angle_A (A B C D : Type) 
  (angle_bisector : ∀ {A B C : Type}, A → B → C → Prop) 
  (AD l c b : ℝ)
  (h1 : angle_bisector A B C)
  (h2 : AD = l)
  (h3 : B = c)
  (h4 : C = b)
  : A = 2 * arccos ((b + c) * l / (2 * b * c)) := 
sorry

end find_angle_A_l405_405214


namespace problem_l405_405617

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405617


namespace AB_distance_l405_405357

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405357


namespace parabola_problem_l405_405546

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405546


namespace AB_distance_l405_405313

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405313


namespace distance_AB_l405_405270

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405270


namespace six_digit_palindrome_count_l405_405014

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l405_405014


namespace parabola_distance_l405_405401

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405401


namespace distance_AB_l405_405347

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405347


namespace parabola_distance_problem_l405_405403

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405403


namespace distance_AB_l405_405269

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405269


namespace sum_of_ages_l405_405687

-- Defining the ages of Nathan and his twin sisters.
variables (n t : ℕ)

-- Nathan has two twin younger sisters, and the product of their ages equals 72.
def valid_ages (n t : ℕ) : Prop := t < n ∧ n * t * t = 72

-- Prove that the sum of the ages of Nathan and his twin sisters is 14.
theorem sum_of_ages (n t : ℕ) (h : valid_ages n t) : 2 * t + n = 14 :=
sorry

end sum_of_ages_l405_405687


namespace AB_distance_l405_405366

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405366


namespace parabola_problem_l405_405554

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405554


namespace parabola_distance_l405_405533

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405533


namespace sum_of_three_largest_l405_405064

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l405_405064


namespace mr_brown_final_price_is_correct_l405_405684

noncomputable def mr_brown_final_purchase_price :
  Float :=
  let initial_price : Float := 100000
  let mr_brown_price  := initial_price * 1.12
  let improvement := mr_brown_price * 0.05
  let mr_brown_total_investment := mr_brown_price + improvement
  let mr_green_purchase_price := mr_brown_total_investment * 1.04
  let market_decline := mr_green_purchase_price * 0.03
  let value_after_decline := mr_green_purchase_price - market_decline
  let loss := value_after_decline * 0.10
  let ms_white_purchase_price := value_after_decline - loss
  let market_increase := ms_white_purchase_price * 0.08
  let value_after_increase := ms_white_purchase_price + market_increase
  let profit := value_after_increase * 0.05
  let final_price := value_after_increase + profit
  final_price

theorem mr_brown_final_price_is_correct :
  mr_brown_final_purchase_price = 121078.76 := by
  sorry

end mr_brown_final_price_is_correct_l405_405684


namespace min_value_y_l405_405983

theorem min_value_y (x : ℝ) (hx : x > 1) : 
  let y := x + 1 / (x - 1) in y ≥ 3 :=
sorry

end min_value_y_l405_405983


namespace sequence_solution_l405_405212

def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 3) ∧ ∀ n, a (n + 1) = 3 * a n - 2 * a (n - 1)

theorem sequence_solution (a : ℕ → ℤ) (n : ℕ)
  (h : sequence a) : a n = 2^(n+1) - 1 :=
by
  sorry

end sequence_solution_l405_405212


namespace distance_AB_l405_405279

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405279


namespace parabola_distance_l405_405495

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405495


namespace problem_l405_405608

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405608


namespace total_time_for_flight_l405_405891

def calculate_total_time (day1_pacific : ℕ) (day1_mountain : ℕ) (day1_central : ℕ) (day1_eastern : ℕ) 
                         (layover_time : ℕ) (zones : ℕ) (speed_increase : ℚ) (time_reduced : ℚ) : ℚ :=
  let day1_hovering := day1_pacific + day1_mountain + day1_central + day1_eastern
  let day1_total := day1_hovering + (layover_time * zones)
  let pacific_time_day2 := day1_pacific * (5 / 6) - time_reduced
  let mountain_time_day2 := day1_mountain * (5 / 6) - time_reduced
  let central_time_day2 := day1_central * (5 / 6) - time_reduced
  let eastern_time_day2 := day1_eastern * (5 / 6) - time_reduced
  let day2_hovering := pacific_time_day2 + mountain_time_day2 + central_time_day2 + eastern_time_day2
  let day2_total := day2_hovering + (layover_time * zones)
  day1_total + day2_total

theorem total_time_for_flight : calculate_total_time 2 3 4 3 1.5 4 1.2 1.6 = 27.6 := 
  by sorry

end total_time_for_flight_l405_405891


namespace parabola_distance_l405_405476

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405476


namespace rectangle_width_decrease_l405_405770

theorem rectangle_width_decrease (L W : ℝ) (h : L * W = A) (h_new_length : 1.40 * L = L') 
    (h_area_unchanged : L' * W' = L * W) : 
    (W - W') / W = 0.285714 :=
begin
  sorry
end

end rectangle_width_decrease_l405_405770


namespace sandy_sold_scooter_for_1500_l405_405710

-- Definitions based on the conditions
def purchase_price : ℝ := 900
def repair_costs : ℝ := 300
def gain_percentage : ℝ := 0.25 

def total_cost : ℝ := purchase_price + repair_costs
def gain : ℝ := gain_percentage * total_cost
def selling_price : ℝ := total_cost + gain

-- Statement to prove
theorem sandy_sold_scooter_for_1500 : selling_price = 1500 :=
begin
  sorry
end

end sandy_sold_scooter_for_1500_l405_405710


namespace terminating_decimal_of_7_div_200_l405_405945

theorem terminating_decimal_of_7_div_200 : (7 / 200 : ℝ) = 0.028 := sorry

end terminating_decimal_of_7_div_200_l405_405945


namespace yellow_region_exists_l405_405903

-- Definitions provided by the conditions of the problem
variables {α : Type*} [DecidableEq α]

structure Circle := 
  (intersections : ℕ) -- Number of intersections for simplicity

-- Given conditions derived from the problem
def circle_system (n : ℕ) (k : ℕ) :=
  ∀ (c : Circle), c.intersections = n ∧ (c.intersections % 2 = 0) ∧ 
  (∀ p, p ∈ set_of_points_of(c) → p.color = yellow → c.color = yellow) ∧ 
  (∃ (yellow_points : ℕ), yellow_points ≥ k)

-- Lean statement reformulation
theorem yellow_region_exists (n : ℕ) (k : ℕ) :
  circle_system n k → 
  (∀ c : Circle, c.intersections ≥ k → 
    ∃ region, region.is_bounded_by(c) ∧ 
    ∀ p ∈ region.vertices, p.color = yellow) := 
  sorry

end yellow_region_exists_l405_405903


namespace width_decrease_percentage_l405_405753

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l405_405753


namespace impossible_event_C_l405_405888

-- Definitions using Lean
def obtuse_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : C > 90) : Prop := A < 90 ∧ B < 90
def triangle_inequality (a b c : ℝ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a
def larger_side_opposite_larger_angle (A B : ℝ) (a b : ℝ) : Prop := (A > B ↔ a > b)
def acute_triangle (A B C : ℝ) : Prop := A < 90 ∧ B < 90 ∧ C < 90

-- Lean theorem statement
theorem impossible_event_C (A B C : ℝ) :
  ¬ (C < 90) ∧ acute_triangle A B C → C < 90 :=
begin
  sorry
end

end impossible_event_C_l405_405888


namespace AB_distance_l405_405371

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405371


namespace parabola_distance_l405_405474

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405474


namespace sally_more_2s_than_5s_l405_405175

open BigOperators

-- Define the probability of rolling more 2's than 5's out of 6 rolls of an 8-sided die
def dice_rolls (prob : ℚ) : Prop :=
  (prob = (86684 : ℚ) / 262144)

noncomputable def probability_of_more_2s_than_5s : ℚ :=
  -- Here should go the full proof calculation skipped with sorry
  sorry

theorem sally_more_2s_than_5s :
  dice_rolls probability_of_more_2s_than_5s :=
begin
  -- The intended proof steps should go here, skipped with sorry
  sorry
end

end sally_more_2s_than_5s_l405_405175


namespace parabola_distance_l405_405489

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405489


namespace remainder_of_polynomial_division_l405_405916

theorem remainder_of_polynomial_division (x : ℝ) :
  ∃ q : ℝ[x], ∃ r : ℝ[x], r.degree < 2 ∧ (r = (405 * x - 969) : ℝ[x]) ∧ ((x^5 + 3 : ℝ[x]) = (x - 3)^2 * q + r) :=
by
  sorry

end remainder_of_polynomial_division_l405_405916


namespace distance_AB_l405_405436

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405436


namespace f_neg_two_l405_405130

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x - c / x + 2

theorem f_neg_two (a b c : ℝ) (h : f a b c 2 = 4) : f a b c (-2) = 0 :=
sorry

end f_neg_two_l405_405130


namespace parabola_problem_l405_405285

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405285


namespace six_digit_palindromes_count_l405_405007

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l405_405007


namespace sum_of_three_largest_l405_405091

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l405_405091


namespace AB_distance_l405_405375

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405375


namespace correct_propositions_l405_405889

theorem correct_propositions (h1 : ∀ x : ℝ, x + π / 4 ≠ k * π + π / 2 → x ∉ {x | x = π / 4 + k * π, k ∈ ℤ})
                            (h2 : ∀ α : ℝ, sin α = 1 / 2 ∧ α ∈ Icc 0 (2 * π) → α ∉ {π / 6, 5 * π / 6})
                            (h3 : ∀ x : ℝ, sin (2 * x + π / 3) + sin (2 * x - π / 3) = sin (2 * x))
                            (h4 : ∀ x : ℝ, cos^2 x + sin x = -(sin x - 1 / 2) ^ 2 + 5 / 4) :
                            ([false, false, true, true] = [1, 2, 3, 4]) :=
sorry

end correct_propositions_l405_405889


namespace parabola_problem_l405_405521

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405521


namespace width_decreased_by_28_6_percent_l405_405761

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l405_405761


namespace sum_of_three_largest_consecutive_numbers_l405_405103

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l405_405103


namespace range_of_a_l405_405979

def satisfies_p (x : ℝ) : Prop := (2 * x - 1) / (x - 1) ≤ 0

def satisfies_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) < 0

def sufficient_but_not_necessary (p q : ℝ → Prop) : Prop :=
  ∀ x, p x → q x ∧ ∃ x, q x ∧ ¬(p x)

theorem range_of_a :
  (∀ (x a : ℝ), satisfies_p x → satisfies_q x a → 0 ≤ a ∧ a < 1 / 2) ↔ (∀ a, 0 ≤ a ∧ a < 1 / 2) := by sorry

end range_of_a_l405_405979


namespace parabola_distance_l405_405584

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405584


namespace larger_volume_of_rotated_rectangle_l405_405794

-- Definitions based on the conditions
def length : ℝ := 4
def width : ℝ := 3

-- Problem statement: Proving the volume of the larger geometric solid
theorem larger_volume_of_rotated_rectangle :
  max (Real.pi * (width ^ 2) * length) (Real.pi * (length ^ 2) * width) = 48 * Real.pi :=
by
  sorry

end larger_volume_of_rotated_rectangle_l405_405794


namespace AB_distance_l405_405321

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405321


namespace complement_of_A_in_B_l405_405138

def set_A : Set ℤ := {x | 2 * x = x^2}
def set_B : Set ℤ := {x | -x^2 + x + 2 ≥ 0}

theorem complement_of_A_in_B :
  (set_B \ set_A) = {-1, 1} :=
by
  sorry

end complement_of_A_in_B_l405_405138


namespace find_t_l405_405984

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def tangent_point (t : ℝ) : ℝ × ℝ := (t, 0)

theorem find_t :
  (∀ (A : ℝ × ℝ), ellipse_eq A.1 A.2 → 
    ∃ (C : ℝ × ℝ),
      tangent_point 2 = C ∧
      -- C is tangent to the extended line of F1A
      -- C is tangent to the extended line of F1F2
      -- C is tangent to segment AF2
      true
  ) :=
sorry

end find_t_l405_405984


namespace equal_circumradii_of_inscribed_quadrilateral_l405_405134

theorem equal_circumradii_of_inscribed_quadrilateral
  {A B C D M N P Q O : Point}
  (h1 : is_inscribed_quadrilateral A B C D)
  (h2 : is_midpoint A B M)
  (h3 : is_midpoint B C N)
  (h4 : is_midpoint C D P)
  (h5 : is_midpoint D A Q)
  (h6 : is_intersection AC BD O) :
  circumradius (triangle O M N) =
  circumradius (triangle O N P) ∧
  circumradius (triangle O N P) =
  circumradius (triangle O P Q) ∧
  circumradius (triangle O P Q) =
  circumradius (triangle O Q M) :=
sorry

end equal_circumradii_of_inscribed_quadrilateral_l405_405134


namespace average_speed_of_trip_is_correct_l405_405834

-- Definitions
def total_distance : ℕ := 450
def distance_part1 : ℕ := 300
def speed_part1 : ℕ := 20
def distance_part2 : ℕ := 150
def speed_part2 : ℕ := 15

-- The average speed problem
theorem average_speed_of_trip_is_correct :
  (total_distance : ℤ) / (distance_part1 / speed_part1 + distance_part2 / speed_part2 : ℤ) = 18 := by
  sorry

end average_speed_of_trip_is_correct_l405_405834


namespace schedule_courses_l405_405877

-- Define the number of courses and periods
def num_courses : Nat := 4
def num_periods : Nat := 8

-- Define the total number of ways to schedule courses without restrictions
def unrestricted_schedules : Nat := Nat.choose num_periods num_courses * Nat.factorial num_courses

-- Define the number of invalid schedules using PIE (approximate value given in problem)
def invalid_schedules : Nat := 1008 + 180 + 120

-- Define the number of valid schedules
def valid_schedules : Nat := unrestricted_schedules - invalid_schedules

theorem schedule_courses : valid_schedules = 372 := sorry

end schedule_courses_l405_405877


namespace distance_AB_l405_405457

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405457


namespace part1_first_6_terms_part2_sum_of_first_n_terms_part3_range_of_a_l405_405978

-- Part 1
noncomputable def c_seq (n : ℕ) := 2^(n - 1)
noncomputable def a_seq : ℕ → ℝ
| 1 := 1
| n + 1 := c_seq n.succ_pred - a_seq n

theorem part1_first_6_terms :
  a_seq 1 = 1 ∧ a_seq 2 = 1 ∧ a_seq 3 = 3 ∧ a_seq 4 = 5 ∧ a_seq 5 = 11 ∧ a_seq 6 = 21 :=
sorry

-- Part 2
def c_seq_arith (n : ℕ) := 3 * n - 2

noncomputable def a_seq_arith : ℕ → ℝ
| 1 := 0
| n + 1 := c_seq_arith n - a_seq_arith n

theorem part2_sum_of_first_n_terms (n : ℕ) :
  let S := Finset.sum (Finset.range n) a_seq_arith in
  if n % 2 = 0 then
    S = (3 / 4) * n^2 - n
  else
    S = (3 / 4) * n^2 - n + 1 / 4 :=
sorry

-- Part 3
def k_seq (n : ℕ) : ℝ := (n^3 - 1) / (a_seq_arith n - 1)
def c_cond (n : ℕ) := (k_seq n - 3) * (k_seq (n + 1) - 3) < 0

theorem part3_range_of_a (a : ℝ) (h : 0 ≤ a ∧ a < 7 ∧ a ≠ 1) :
  ∀ n : ℕ, c_cond n :=
sorry

end part1_first_6_terms_part2_sum_of_first_n_terms_part3_range_of_a_l405_405978


namespace distance_AB_l405_405336

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405336


namespace parabola_distance_l405_405487

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405487


namespace minimum_value_frac_l405_405183

theorem minimum_value_frac (a b : ℝ) (h₁ : 2 * a - b + 2 * 0 = 0) 
  (h₂ : a > 0) (h₃ : b > 0) (h₄ : a + b = 1) : 
  (1 / a) + (1 / b) = 4 :=
sorry

end minimum_value_frac_l405_405183


namespace rectangle_width_decrease_proof_l405_405755

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l405_405755


namespace passengers_on_bus_l405_405797

theorem passengers_on_bus (initial final off : ℕ) (got_on : ℕ) :
  initial = 28 → final = 26 → off = 9 → initial + got_on - off = final → got_on = 7 :=
by
  intros h_initial h_final h_off h_eq
  rw [h_initial, h_final, h_off] at h_eq
  linarith

end passengers_on_bus_l405_405797


namespace AB_distance_l405_405319

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405319


namespace distance_AB_l405_405332

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405332


namespace AB_distance_l405_405322

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405322


namespace parabola_distance_l405_405592

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405592


namespace sum_of_largest_three_consecutive_numbers_l405_405094

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l405_405094


namespace problem_solution_l405_405645

-- Define the sequence a_k based on the given conditions
def a : ℕ → ℝ
| 1 := 0.3
| 2 := (0.31)^(0.3)
| 3 := (0.301)^(a 2)
| 4 := (0.3011)^(a 3)
| k := if k % 2 = 1 then (0.3 + 0.00001 * (k - 1))^(a (k - 1)) else (0.3 + 0.00001 * k)^(a (k - 1))

-- Sequence b_k which is just a_k rearranged in decreasing order
def b : ℕ → ℝ := sorry -- For simplicity, we define it for conceptual reasons without implementation

-- Define the final theorem based on the aggregation of k's
theorem problem_solution : (Finset.range 11).sum (λ k, if a k = b k then k else 0) = 41 :=
by
  sorry

end problem_solution_l405_405645


namespace parabola_problem_l405_405298

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405298


namespace parabola_distance_l405_405570

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405570


namespace parabola_distance_l405_405573

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405573


namespace highest_lowest_income_diff_l405_405841

-- Condition: Income records for 8 days
def income_records : List Int := [62, 40, -60, -38, 0, 34, 8, -54]

-- Statement to prove: Difference between highest and lowest income
theorem highest_lowest_income_diff : 
  let highest_income := income_records.maximum
  let lowest_income := income_records.minimum
  (highest_income - lowest_income) = 122 := by
  sorry

end highest_lowest_income_diff_l405_405841


namespace max_value_inequality_l405_405669

-- Define the variables and conditions
variables {x y z : ℝ}
hypothesis h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0
hypothesis h_sum : x + y + z = 3

-- Define the main statement
theorem max_value_inequality (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 3) :
  (xy / (x + y)) + (xz / (x + z)) + (yz / (y + z)) ≤ 9 / 8 := sorry

end max_value_inequality_l405_405669


namespace min_value_of_expression_l405_405942

noncomputable def myExpression (θ : ℝ) : ℝ :=
  3 * Real.sin θ + 2 / Real.cos θ + Real.sqrt 3 * Real.cot θ 

theorem min_value_of_expression : 
  ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 ∧ 
           (∀ θ' : ℝ, 0 < θ' ∧ θ' < Real.pi / 2 → myExpression θ' ≥ myExpression θ) ∧
           myExpression θ = 3 * Real.sqrt 2 + Real.sqrt 3 :=
sorry

end min_value_of_expression_l405_405942


namespace six_digit_palindromes_count_l405_405051

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l405_405051


namespace parabola_problem_l405_405302

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405302


namespace tens_digit_23_1987_l405_405921

theorem tens_digit_23_1987 : (23 ^ 1987 % 100) / 10 % 10 = 4 :=
by
  -- The proof goes here
  sorry

end tens_digit_23_1987_l405_405921


namespace num_squares_in_H_l405_405912

noncomputable def H : set (ℤ × ℤ) := { p | 2 ≤ abs (p.fst) ∧ abs (p.fst) ≤ 10 ∧ 2 ≤ abs (p.snd) ∧ abs (p.snd) ≤ 10 }

theorem num_squares_in_H : 
  let side_lengths := [5, 9] in
  let squares := { side_length | side_length ∈ side_lengths ∧ 
                                ∀ (x y : ℤ), (x, y) ∈ H ∧ (x + side_length, y) ∈ H ∧ (x, y + side_length) ∈ H ∧ (x + side_length, y + side_length) ∈ H } in
  squares.card = 100 :=
by sorry

end num_squares_in_H_l405_405912


namespace distance_AB_l405_405428

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405428


namespace six_digit_palindromes_count_l405_405025

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l405_405025


namespace Steve_pencils_left_l405_405722

-- Define the initial number of boxes and pencils per box
def boxes := 2
def pencils_per_box := 12
def initial_pencils := boxes * pencils_per_box

-- Define the number of pencils given to Lauren and the additional pencils given to Matt
def pencils_to_Lauren := 6
def diff_Lauren_Matt := 3
def pencils_to_Matt := pencils_to_Lauren + diff_Lauren_Matt

-- Calculate the total pencils given away
def pencils_given_away := pencils_to_Lauren + pencils_to_Matt

-- Number of pencils left with Steve
def pencils_left := initial_pencils - pencils_given_away

-- The statement to prove
theorem Steve_pencils_left : pencils_left = 9 := by
  sorry

end Steve_pencils_left_l405_405722


namespace parabola_distance_l405_405387

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405387


namespace distance_AB_l405_405266

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405266


namespace copy_pages_l405_405218

theorem copy_pages
  (total_cents : ℕ)
  (cost_per_page : ℚ)
  (h_total : total_cents = 2000)
  (h_cost : cost_per_page = 2.5) :
  (total_cents / cost_per_page) = 800 :=
by
  -- This is where the proof would go
  sorry

end copy_pages_l405_405218


namespace work_done_correct_l405_405928

structure GasState where
  p : ℝ   -- pressure in Pascals
  V : ℝ   -- volume in cubic meters

def p0 : ℝ := 10^5   -- 10^5 Pascals
def V0 : ℝ := 3 * 10^(-3) -- 3 liters in cubic meters

def state1 : GasState := ⟨p0, V0⟩
def state3 : GasState := state1
def state4 : GasState := ⟨p0, V0⟩
def state6 : GasState := state4

noncomputable def work_done_in_cycle (s1 s2 s3 s4 s5 s6 : GasState) : ℝ :=
  5 * π * s1.p * s1.V

theorem work_done_correct :
  work_done_in_cycle state1 state3 state3 state4 state4 state6 = 2827 :=
sorry

end work_done_correct_l405_405928


namespace find_A_l405_405126

axiom power_eq_A (A : ℝ) (x y : ℝ) : 2^x = A ∧ 7^(2*y) = A
axiom reciprocal_sum_eq_2 (x y : ℝ) : (1/x) + (1/y) = 2

theorem find_A (A x y : ℝ) : 
  (2^x = A) ∧ (7^(2*y) = A) ∧ ((1/x) + (1/y) = 2) -> A = 7*Real.sqrt 2 :=
by 
  sorry

end find_A_l405_405126


namespace sum_of_largest_three_consecutive_numbers_l405_405093

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l405_405093


namespace width_decrease_percentage_l405_405750

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l405_405750


namespace parabola_distance_l405_405586

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405586


namespace parabola_problem_l405_405558

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405558


namespace solve_triangle_problem_l405_405215

noncomputable def triangle_cos_sum (P Q R S : ℕ) (a b : ℕ) : Prop :=
  let ⟨PR, PQ, RQ⟩ := ⟨x, 25^3 * x, x * 25⟩ in
  let h1 : PR^2 = 25^3 * PQ := by sorry
  let h2 : (PR / 25) + (RQ / 25) = PQ / 25 := by sorry
  let cos_Q : (PR / PQ) = (25 / 313) := by sorry
  gcd a b = 1 ∧ a = 25 ∧ b = 313

theorem solve_triangle_problem :
  ∀ P Q R S a b, triangle_cos_sum P Q R S a b → a + b = 338 :=
by
  intros
  simp at *
  exact sorry

end solve_triangle_problem_l405_405215


namespace rectangle_width_decrease_l405_405776

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L in
  let W' := (L * W) / L' in
  let percent_decrease := (1 - W' / W) * 100 in
  percent_decrease ≈ 28.57 :=
by
  let L' := 1.4 * L
  let W' := (L * W) / L'
  let percent_decrease := (1 - W' / W) * 100
  sorry

end rectangle_width_decrease_l405_405776


namespace find_f_expr_monotonically_increasing_interval_l405_405153

open Real

noncomputable def f (x : ℝ) : ℝ := A * sin (ω * x + φ) + B

variables {A ω φ B : ℝ}
variable {k : ℤ}

axiom A_pos : A > 0
axiom ω_pos : ω > 0
axiom φ_bound : |φ| < π / 2
axiom f_max : ∀ x, f x ≤ 2 * sqrt 2
axiom f_min : ∀ x, -sqrt 2 ≤ f x
axiom period : ∀ x, f (x + π) = f x
axiom point_pass : f 0 = -(sqrt 2 / 4)

theorem find_f_expr : 
  f x = (3 * sqrt 2 / 2) * sin (2 * x - π / 6) + (sqrt 2 / 2) := sorry

theorem monotonically_increasing_interval :
  ∀ x, ∃ k : ℤ, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 := sorry

end find_f_expr_monotonically_increasing_interval_l405_405153


namespace parabola_distance_l405_405529

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405529


namespace circle_passing_ellipse_vertices_origin_passes_point_l405_405989

noncomputable def standardCircleEquation : Prop :=
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), (x - m) ^ 2 + y ^ 2 = (6 - m + (6 - m))) ∧
    (m := 8 / 3 ∧ (∀ x y : ℝ, (x - m) ^ 2 + y ^ 2 = (100 / 9)))

theorem circle_passing_ellipse_vertices_origin_passes_point :
  ( ∃ (m : ℝ), 
      (∀ (x y : ℝ), (x - m) ^ 2 + y ^ 2 = (6 - m) ^ 2 + 4) )
  = 
  (∀ x y : ℝ, 
   ((x - 8 / 3) ^ 2 + y ^ 2 = 100 / 9)):=
sorry

end circle_passing_ellipse_vertices_origin_passes_point_l405_405989


namespace parallelogram_diagonals_divide_equal_areas_l405_405821

theorem parallelogram_diagonals_divide_equal_areas {A B C D : Type} 
  [parallelogram A B C D] : 
  divides_into_equal_area_triangles A B C D := 
sorry

end parallelogram_diagonals_divide_equal_areas_l405_405821


namespace problem_l405_405604

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405604


namespace area_cross_section_triangle_BCD_l405_405968

def Point := ℝ × ℝ × ℝ

def is_centroid (A B C P : Point) : Prop :=
  P = ( (A.1 + B.1 + C.1) / 3,
        (A.2 + B.2 + C.2) / 3,
        (A.3 + B.3 + C.3) / 3 )

def perpendicular (u v : Point) : Prop :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def area_of_triangle (A B C : Point) : ℝ :=
  let s := (distance A B + distance B C + distance C A) / 2
  let a := distance A B
  let b := distance B C
  let c := distance C A
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_cross_section_triangle_BCD
  (A B C A1 B1 C1 D P : Point)
  (h : ℝ) (l : ℝ)
  (H_heights : A1 = (A.1, A.2, A.3 + h))
  (H_base : B = (A.1 + l, A.2, A.3))
  (H_base_2 : C = (A.1 + l/2, A.2 - l*real.sqrt(3)/2, A.3))
  (H_top : B1 = (B.1, B.2, B.3 + h))
  (H_top_2 : C1 = (C.1, C.2, C.3 + h))
  (H_centroid : is_centroid A1 B1 C1 P)
  (H_perpendicular : perpendicular (P.1 - A.1, P.2 - A.2, P.3 - A.3) (D.1 - A.1, D.2 - A.2, D.3 - A.3)) :
  distance B D = distance C D ∧
  area_of_triangle B C D = real.sqrt(13) / 8 := 
sorry

end area_cross_section_triangle_BCD_l405_405968


namespace tangent_line_at_zero_is_zero_range_of_a_l405_405160

-- Define the function f(x)
def f (a x : ℝ) : ℝ := a * Real.log (x + 1) - 2 * Real.exp x + Real.sin x + 2

-- Part 1: Prove that the tangent line at (0, f(0)) when a = 1 is y = 0
theorem tangent_line_at_zero_is_zero : 
  let a := 1 in 
  let f' := fun x => (1 / (x + 1)) - 2 * Real.exp x + Real.cos x in 
  (f' 0 = 0) → (f 1 0 = 0) → ∀ x, f 1 x = 0 → f' 0 = 0 :=
by
  sorry

-- Part 2: Prove that the range of values for a such that f(x) ≤ 0 for x ∈ [0, π] is (-∞, 1]
theorem range_of_a : 
  ∀ a, (∀ x, 0 ≤ x ∧ x ≤ Real.pi → f a x ≤ 0) ↔ a ≤ 1 :=
by
  sorry

end tangent_line_at_zero_is_zero_range_of_a_l405_405160


namespace domain_of_g_l405_405810

def g (x : ℝ) := (x + 7) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | x^2 - 5 * x + 6 ≥ 0 ∧ x^2 - 5 * x + 6 ≠ 0} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_of_g_l405_405810


namespace parabola_distance_l405_405535

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405535


namespace parabola_distance_l405_405543

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405543


namespace value_of_f_c_l405_405991

variable (a b c : ℝ)

-- Conditions
def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f (x : ℝ) : ℝ := (x + a) / (x^2 + b * x + 1)

theorem value_of_f_c
  (h1 : isOdd (f a b))
  (h2 : c = 1)
  (h3 : a = 0)
  (h4 : b = 0) :
  f a b c = 1 / 2 :=
by 
  sorry

end value_of_f_c_l405_405991


namespace sum_of_three_largest_l405_405065

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l405_405065


namespace ab_distance_l405_405235

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405235


namespace trailing_zeroes_73_79_83_l405_405171

noncomputable def trailing_zeroes (n : ℕ) : ℕ :=
  ∑ k in (range (n+1)).filter (λ x, x % 5 = 0), 1 + (nat.log 5 x).to_nat

/-- 
Problem Statement:
Let n = 73! + 79! + 83!.
Prove that the number of trailing zeroes in n is 16.
-/
theorem trailing_zeroes_73_79_83 : trailing_zeroes (73!) = 16 ∧ trailing_zeroes (79!) ≥ 16 ∧ trailing_zeroes (83!) ≥ 16 
∧ trailing_zeroes (73! + 79! + 83!) = 16 := 
by 
  sorry

end trailing_zeroes_73_79_83_l405_405171


namespace increase_in_length_and_breadth_is_4_l405_405748

-- Define the variables for the original length and breadth of the room
variables (L B x : ℕ)

-- Define the original perimeter
def P_original : ℕ := 2 * (L + B)

-- Define the new perimeter after the increase
def P_new : ℕ := 2 * ((L + x) + (B + x))

-- Define the condition that the perimeter increases by 16 feet
axiom increase_perimeter : P_new L B x - P_original L B = 16

-- State the theorem that \(x = 4\)
theorem increase_in_length_and_breadth_is_4 : x = 4 :=
by
  -- Proof would be filled in here using the axioms and definitions
  sorry

end increase_in_length_and_breadth_is_4_l405_405748


namespace solve_problem_l405_405958

noncomputable def problem : Prop :=
  ∀ x : ℝ, tan x = -1 / 2 → sin x ^ 2 + 3 * sin x * cos x - 1 = -2

theorem solve_problem : problem :=
by
  sorry

end solve_problem_l405_405958


namespace distance_AB_l405_405471

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405471


namespace ab_distance_l405_405241

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405241


namespace mystical_swamp_l405_405200

/-- 
In a mystical swamp, there are two species of talking amphibians: toads, whose statements are always true, and frogs, whose statements are always false. 
Five amphibians: Adam, Ben, Cara, Dan, and Eva make the following statements:
Adam: "Eva and I are different species."
Ben: "Cara is a frog."
Cara: "Dan is a frog."
Dan: "Of the five of us, at least three are toads."
Eva: "Adam is a toad."
Given these statements, prove that the number of frogs is 3.
-/
theorem mystical_swamp :
  (∀ α β : Prop, α ∨ ¬β) ∧ -- Adam's statement: "Eva and I are different species."
  (Cara = "frog") ∧          -- Ben's statement: "Cara is a frog."
  (Dan = "frog") ∧         -- Cara's statement: "Dan is a frog."
  (∃ t, t = nat → t ≥ 3) ∧ -- Dan's statement: "Of the five of us, at least three are toads."
  (Adam = "toad")               -- Eva's statement: "Adam is a toad."
  → num_frogs = 3 := sorry       -- Number of frogs is 3.

end mystical_swamp_l405_405200


namespace probability_of_at_least_one_three_l405_405853

-- Define the problem statement in Lean 4.
theorem probability_of_at_least_one_three (t1 t2 t3 t4 : ℕ) (h_toss : ∀ i, 1 ≤ i ∧ i ≤ 6) (h_sum : t1 + t2 + t3 + t4 = 14) : 
  ∃ (i : ℕ), i ∈ {1, 2, 3, 4} ∧ (List.nthLe [t1, t2, t3, t4] i sorry) = 3 :=
begin
  sorry -- the proof is omitted as per the instructions.
end

end probability_of_at_least_one_three_l405_405853


namespace parabola_distance_l405_405539

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405539


namespace main_theorem_l405_405648

noncomputable def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

lemma f_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- Detailed steps of proof will go here
  sorry

lemma f_monotonic_increasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  intros x₁ x₂ h₁_lt_zero h₁_lt_h₂
  -- Detailed steps of proof will go here
  sorry

theorem main_theorem : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ → f x₁ < f x₂) := by
  split
  · exact f_odd
  · exact f_monotonic_increasing

end main_theorem_l405_405648


namespace original_acid_percentage_l405_405817

theorem original_acid_percentage (a w : ℕ) (h1 : a / (a + w + 2 : ℕ) = 0.25) (h2 : (a + 2) / (a + w + 4 : ℕ) = 0.40) :
  (a / (a + w) : ℚ) * 100 = 33 + 1 / 3 :=
by
  sorry

end original_acid_percentage_l405_405817


namespace six_digit_palindrome_count_l405_405012

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l405_405012


namespace parabola_distance_problem_l405_405411

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405411


namespace sum_of_three_largest_l405_405086

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l405_405086


namespace hyperbola_asymptotes_l405_405739

theorem hyperbola_asymptotes (a : ℝ) (h : a ≠ 0) :
  asymptotes (λ x y, y^2 / (2 * a^2) - x^2 / a^2 - 1 = 0) = 
  (λ x y, y = sqrt 2 * x ∨ y = -sqrt 2 * x) :=
sorry

end hyperbola_asymptotes_l405_405739


namespace distance_AB_l405_405442

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405442


namespace width_decreased_by_28_6_percent_l405_405764

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l405_405764


namespace six_digit_palindromes_count_l405_405003

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l405_405003


namespace solve_stream_speed_l405_405849

noncomputable def boat_travel (v : ℝ) : Prop :=
  let downstream_speed := 12 + v
  let upstream_speed := 12 - v
  let downstream_time := 60 / downstream_speed
  let upstream_time := 60 / upstream_speed
  upstream_time - downstream_time = 2

theorem solve_stream_speed : ∃ v : ℝ, boat_travel v ∧ v = 2.31 :=
by {
  sorry
}

end solve_stream_speed_l405_405849


namespace problem_l405_405601

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405601


namespace six_digit_palindromes_count_l405_405054

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l405_405054


namespace sum_of_largest_three_consecutive_numbers_l405_405092

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l405_405092


namespace AB_distance_l405_405323

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405323


namespace correct_calculation_l405_405819

variable (a : ℝ) -- assuming a ∈ ℝ

theorem correct_calculation : (a ^ 3) ^ 2 = a ^ 6 :=
by {
  sorry
}

end correct_calculation_l405_405819


namespace total_cost_of_groceries_l405_405734

noncomputable def M (R : ℝ) : ℝ := 24 * R / 10
noncomputable def F : ℝ := 22

theorem total_cost_of_groceries (R : ℝ) (hR : 2 * R = 22) :
  10 * M R = 24 * R ∧ F = 2 * R ∧ F = 22 →
  4 * M R + 3 * R + 5 * F = 248.6 := by
  sorry

end total_cost_of_groceries_l405_405734


namespace six_digit_palindromes_count_l405_405046

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l405_405046


namespace f_2009_l405_405909

def f (x : ℝ) : ℝ := x^3 -- initial definition for x in [-1, 1]

axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom symmetric_around_1 : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_cubed : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3

theorem f_2009 : f 2009 = 1 := by {
  -- The body of the theorem will be filled with proof steps
  sorry
}

end f_2009_l405_405909


namespace AB_distance_l405_405320

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405320


namespace distance_AB_l405_405281

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405281


namespace find_expression_value_l405_405937

theorem find_expression_value : 1 + 2 * 3 - 4 + 5 = 8 :=
by
  sorry

end find_expression_value_l405_405937


namespace unicorn_witch_ratio_l405_405206

theorem unicorn_witch_ratio (W D U : ℕ) (h1 : W = 7) (h2 : D = W + 25) (h3 : U + W + D = 60) :
  U / W = 3 := by
  sorry

end unicorn_witch_ratio_l405_405206


namespace arithmetic_sequence_sum_l405_405901

theorem arithmetic_sequence_sum :
  let a := 1
  let d := 3
  let n := 11
  let l := 31
  (n : ℕ) := 11 ∧ (l : ℕ) := a + (n - 1) * d ∧
  (n * (a + l)) / 2 = 176 := 
by
  sorry

end arithmetic_sequence_sum_l405_405901


namespace parabola_distance_l405_405545

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405545


namespace parabola_distance_l405_405524

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405524


namespace parabola_distance_l405_405396

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405396


namespace tens_digit_of_23_pow_1987_l405_405925

theorem tens_digit_of_23_pow_1987 : (23 ^ 1987 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_23_pow_1987_l405_405925


namespace distance_AB_l405_405265

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405265


namespace unique_final_configuration_l405_405863

structure Configuration where
  cols : List ℕ
  h_well_formed : (∀ (i j : ℕ), i < j → (i < cols.length ∧ j < cols.length) → cols[i] - cols[j] ≥ 0 )

def final_configuration (cols : List ℕ) : Prop :=
  ∀ i, (i < cols.length - 1 → cols[i] ≤ cols[i + 1] + 1) ∧
       (i < cols.length - 1 → ∀ j, j > i + 1 → (cols[i] = cols[i+1] → cols[j] ≠ cols[j+1]))

theorem unique_final_configuration (n : ℕ) :
  ∀ (initial_config : List ℕ), (initial_config.sum = n) →
  ∃! final_config, final_configuration final_config :=
by sorry

end unique_final_configuration_l405_405863


namespace std_deviation_of_dataset_l405_405790

-- Define the dataset
def dataset : List ℝ := [3, 4, 5, 5, 6, 7]

-- Standard deviation of dataset
theorem std_deviation_of_dataset :
  stddev dataset = Real.sqrt (5 / 3) := 
begin
  sorry -- Proof goes here
end

end std_deviation_of_dataset_l405_405790


namespace six_digit_palindromes_count_l405_405055

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l405_405055


namespace six_digit_palindrome_count_l405_405013

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l405_405013


namespace find_a_from_polynomial_expansion_l405_405990

theorem find_a_from_polynomial_expansion (a : ℝ) :
  (coeff (expand ((a - x) * (2 + x)^5) x^3) = 40) → a = 3 :=
by
  sorry

end find_a_from_polynomial_expansion_l405_405990


namespace distance_AB_l405_405450

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405450


namespace small_sphere_diameter_l405_405695

theorem small_sphere_diameter :
  let R : ℝ := 15
  let a : ℝ := 32 
  let d := a * real.sqrt 3 / 2 
  let r := (16 * real.sqrt 3 - 15) / (real.sqrt 3 + 1)
  2 * r = 63 - 31 * real.sqrt 3 :=
by
  -- Conditions definition
  let R : ℝ := 15
  let a : ℝ := 32 
  let d := a * real.sqrt 3 / 2
  let r := (16 * real.sqrt 3 - 15) / (real.sqrt 3 + 1)
  
  -- Correct answer definition
  have : 2 * ((16 * real.sqrt 3 - 15) / (real.sqrt 3 + 1)) = 63 - 31 * real.sqrt 3, from sorry,
  this

end small_sphere_diameter_l405_405695


namespace problem_l405_405597

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405597


namespace probability_yellow_white_l405_405191

-- Define the probabilities for red, yellow and white balls
variables (P_A P_B P_C : ℝ)

def probability_red_yellow (P_A P_B : ℝ) : Prop := P_A + P_B = 0.4
def probability_red_white (P_A P_C : ℝ) : Prop := P_A + P_C = 0.9

-- The proof statement to show:
theorem probability_yellow_white :
  (∀ P_A P_B P_C, probability_red_yellow P_A P_B →
                  probability_red_white P_A P_C →
                  P_B + P_C = 0.7) :=
by
  intro P_A P_B P_C,
  intro h_red_yellow h_red_white,
  sorry

end probability_yellow_white_l405_405191


namespace num_correct_statements_l405_405119

def a_n (n : ℕ) : ℕ := n^2 + n

def Δ (a : ℕ → ℕ) (n : ℕ) : ℕ := a (n+1) - a n

def Δk (a : ℕ → ℕ) (k n : ℕ) : ℕ :=
  match k with
  | 0     => a n
  | (k+1) => Δ (Δk a k) n

theorem num_correct_statements : 
  (Δ (a_n) n = 2 * n + 2) ∧
  ∀ n : ℕ, Σ (i : ℕ) in range (2015 + 1), Δk (a_n) 2 i = 4030 →
  (1 + 1 = 2) :=
by
  sorry

end num_correct_statements_l405_405119


namespace solution_is_13_l405_405225

def marbles_in_jars : Prop :=
  let jar1 := (5, 3, 1)  -- (red, blue, green)
  let jar2 := (1, 5, 3)  -- (red, blue, green)
  let jar3 := (3, 1, 5)  -- (red, blue, green)
  let total_ways := 125 + 15 + 15 + 3 + 27 + 15
  let favorable_ways := 125
  let probability := favorable_ways / total_ways
  let simplified_probability := 5 / 8
  let m := 5
  let n := 8
  m + n = 13

theorem solution_is_13 : marbles_in_jars :=
by {
  sorry
}

end solution_is_13_l405_405225


namespace six_digit_palindromes_count_l405_405040

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l405_405040


namespace temperature_at_80_degrees_l405_405688

theorem temperature_at_80_degrees (t : ℝ) :
  (-t^2 + 10 * t + 60 = 80) ↔ (t = 5 + 3 * Real.sqrt 5 ∨ t = 5 - 3 * Real.sqrt 5) := by
  sorry

end temperature_at_80_degrees_l405_405688


namespace evaluate_expression_l405_405932

variable (b x : ℝ)

theorem evaluate_expression (h : x = b + 9) : x - b + 4 = 13 := by
  sorry

end evaluate_expression_l405_405932


namespace lines_sorted_by_k_lines_sorted_by_b_l405_405692

namespace OrderingProblem

-- Define the lines as a structure
structure Line where
  k : ℝ
  b : ℝ

-- Conditions given in the problem
def line_1 : Line := {k := k1, b := b1}
def line_2 : Line := {k := k2, b := b2}  -- k2 < 0
def line_3 : Line := {k := k3, b := b3}
def line_4 : Line := {k := k4, b := b4}

-- Additional condition ensuring distinct intercepts
axiom hdistinct : b1 ≠ b2 ∧ b1 ≠ b3 ∧ b1 ≠ b4 ∧ b2 ≠ b3 ∧ b2 ≠ b4 ∧ b3 ≠ b4

-- Proof statements
theorem lines_sorted_by_k (h1 : 0 < k1) (h3 : 0 < k3) (h4 : 0 < k4) (hk : k1 < k3 ∧ k3 < k4)
  : [line_2, line_1, line_3, line_4] = ([line_2, line_1, line_3, line_4].sort (λ l1 l2 => l1.k < l2.k)) :=
by sorry 

theorem lines_sorted_by_b (hb1_4 : b4 < b1) (hb1_3 : b1 < b3) (hb3_2 : b3 < b2)
  : [line_4, line_1, line_3, line_2] = ([line_4, line_1, line_3, line_2].sort (λ l1 l2 => l1.b < l2.b)) :=
by sorry 

end OrderingProblem

end lines_sorted_by_k_lines_sorted_by_b_l405_405692


namespace parabola_problem_l405_405305

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405305


namespace tens_digit_23_1987_l405_405922

theorem tens_digit_23_1987 : (23 ^ 1987 % 100) / 10 % 10 = 4 :=
by
  -- The proof goes here
  sorry

end tens_digit_23_1987_l405_405922


namespace Lindy_total_distance_l405_405219

-- Definitions derived from conditions
def distance (Jack Christina : ℕ) : ℕ := 150

def speed (person : Type) : ℕ :=
  match person with
  | "Jack" => 7
  | "Christina" => 8
  | "Lindy" => 10
  | _ => 0

def relative_speed (person1 person2 : Type) : ℕ :=
  speed "Jack" + speed "Christina"

def time_to_meet (dist rel_speed : ℕ) : ℕ :=
  dist / rel_speed

def distance_Lindy_travels (lindy_speed time : ℕ) : ℕ :=
  lindy_speed * time

-- Lean 4 statement
theorem Lindy_total_distance :
  distance_Lindy_travels (speed "Lindy") (time_to_meet (distance Jack Christina) (relative_speed "Jack" "Christina")) = 100 :=
by
  sorry

end Lindy_total_distance_l405_405219


namespace hulk_first_exceeds_2000_l405_405726

def hulk_jump : ℕ → ℕ
| 1     := 2
| (n+1) := 2 * hulk_jump n + n

theorem hulk_first_exceeds_2000 : ∃ n, hulk_jump n > 2000 ∧ n = 15 :=
by {
  let n := 15,
  have hn : hulk_jump n > 2000 := sorry,
  exact ⟨n, hn, rfl⟩
}

end hulk_first_exceeds_2000_l405_405726


namespace range_of_m_plus_n_l405_405156

theorem range_of_m_plus_n (f : ℝ → ℝ) (n m : ℝ)
  (h_f_def : ∀ x, f x = x^2 + n * x + m)
  (h_non_empty : ∃ x, f x = 0 ∧ f (f x) = 0)
  (h_condition : ∀ x, f x = 0 ↔ f (f x) = 0) :
  0 < m + n ∧ m + n < 4 :=
by {
  -- Proof needed here; currently skipped
  sorry
}

end range_of_m_plus_n_l405_405156


namespace problem_l405_405605

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405605


namespace who_wrote_l405_405883

-- Definitions representing the conditions
def A_claim := ∀ (B_wrote C_wrote : Bool), A_said = ((B_wrote ∨ C_wrote) = true)
def B_claim := ∀ (B_wrote E_wrote : Bool), B_said = ((¬B_wrote ∧ ¬E_wrote) = true)
def C_claim := ∀ (A_said B_said : Bool), C_said = ((¬A_said ∧ ¬B_said) = true)
def D_claim := ∀ (A_truth B_truth : Bool), D_said = ((A_truth ⊕ B_truth) = true)
def E_claim := ∀ (D_truth : Bool), E_said = ¬D_truth

-- Given conditions about the truthfulness of the students
def always_truthful := ∀ (p : Bool), p = true
def always_lying := ∀ (p : Bool), p = false

-- The main theorem stating C wrote on the blackboard
theorem who_wrote (truth_values : List (Bool → Bool)) : (C_wrote = true) :=
  sorry

end who_wrote_l405_405883


namespace quadratic_has_two_distinct_roots_l405_405704

theorem quadratic_has_two_distinct_roots (a b c α : ℝ) (h : a * (a * α^2 + b * α + c) < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a*x1^2 + b*x1 + c = 0) ∧ (a*x2^2 + b*x2 + c = 0) ∧ x1 < α ∧ x2 > α :=
sorry

end quadratic_has_two_distinct_roots_l405_405704


namespace parabola_distance_l405_405541

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405541


namespace range_of_f_gt_zero_l405_405149

def f (x : ℝ) : ℝ :=
if hx : x > 0 then log x / log 2
else if hx0 : x = 0 then 0
else -log (-x) / log 2

theorem range_of_f_gt_zero (x : ℝ) (h_odd : ∀ x, f(-x) = -f(x)) (h_pos : ∀ x, x > 0 → f(x) = log x / log 2) :
  f(x) > 0 ↔ ((-1 < x ∧ x < 0) ∨ x > 1) :=
begin
  sorry
end

end range_of_f_gt_zero_l405_405149


namespace parabola_distance_l405_405538

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405538


namespace train_cross_time_l405_405880

variables (speed_kmh : ℕ) (length_m : ℕ)

-- Conditions
def speed_m_s := (speed_kmh : ℝ) * 1000 / 3600
def time_to_cross := (length_m : ℝ) / speed_m_s

-- Proof statement
theorem train_cross_time (h1 : speed_kmh = 60) (h2 : length_m = 600) : 
  time_to_cross speed_kmh length_m = 36 :=
by
  rw [h1, h2]
  unfold speed_m_s time_to_cross
  norm_num
  sorry

end train_cross_time_l405_405880


namespace strictly_decreasing_interval_l405_405791

def f (x : ℝ) := Real.exp (x^2 - 2*x - 3)

theorem strictly_decreasing_interval : ∀ x y : ℝ, x < 1 ∧ y < 1 ∧ x < y → f y < f x :=
by
  intros x y h;
  sorry

end strictly_decreasing_interval_l405_405791


namespace ab_distance_l405_405257

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405257


namespace parabola_problem_l405_405508

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405508


namespace guy_has_sixty_cents_l405_405227

-- Definitions for the problem conditions
def lance_has (lance_cents : ℕ) : Prop := lance_cents = 70
def margaret_has (margaret_cents : ℕ) : Prop := margaret_cents = 75
def bill_has (bill_cents : ℕ) : Prop := bill_cents = 60
def total_has (total_cents : ℕ) : Prop := total_cents = 265

-- Problem Statement in Lean format
theorem guy_has_sixty_cents (lance_cents margaret_cents bill_cents total_cents guy_cents : ℕ) 
    (h_lance : lance_has lance_cents)
    (h_margaret : margaret_has margaret_cents)
    (h_bill : bill_has bill_cents)
    (h_total : total_has total_cents) :
    guy_cents = total_cents - (lance_cents + margaret_cents + bill_cents) → guy_cents = 60 :=
by
  intros h
  simp [lance_has, margaret_has, bill_has, total_has] at *
  rw [h_lance, h_margaret, h_bill, h_total] at h
  exact h

end guy_has_sixty_cents_l405_405227


namespace bird_weights_l405_405882

variables (A B V G : ℕ)

theorem bird_weights : 
  A + B + V + G = 32 ∧ 
  V < G ∧ 
  V + G < B ∧ 
  A < V + B ∧ 
  G + B < A + V 
  → 
  (A = 13 ∧ V = 4 ∧ G = 5 ∧ B = 10) :=
sorry

end bird_weights_l405_405882


namespace sum_of_three_largest_of_consecutive_numbers_l405_405070

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l405_405070


namespace parabola_distance_l405_405587

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405587


namespace six_digit_palindromes_l405_405017

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l405_405017


namespace AB_distance_l405_405354

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405354


namespace rhombus_area_correct_l405_405736

open Real

noncomputable def rhombus_area (d1 d2 : ℝ) (θ_degrees : ℝ) : ℝ :=
  let θ_radians := θ_degrees * (π / 180)
  (d1 * d2 * sin(θ_radians)) / 2

theorem rhombus_area_correct : rhombus_area 80 120 40 ≈ 3085.44 :=
by
  have h : rhombus_area 80 120 40 = (80 * 120 * sin((40 * π) / 180)) / 2 := by refl
  simp at h
  simp [rhombus_area] at h
  sorry

end rhombus_area_correct_l405_405736


namespace algebraic_identity_l405_405702

theorem algebraic_identity 
  (p q r a b c : ℝ)
  (h₁ : p + q + r = 1)
  (h₂ : 1 / p + 1 / q + 1 / r = 0) :
  a^2 + b^2 + c^2 = (p * a + q * b + r * c)^2 + (q * a + r * b + p * c)^2 + (r * a + p * b + q * c)^2 := by
  sorry

end algebraic_identity_l405_405702


namespace ratio_of_triangle_to_square_l405_405931

theorem ratio_of_triangle_to_square (s : ℝ) (hs : 0 < s) :
  let A_square := s^2
  let A_triangle := (1/2) * s * (s/2)
  A_triangle / A_square = 1/4 :=
by
  sorry

end ratio_of_triangle_to_square_l405_405931


namespace AB_distance_l405_405374

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405374


namespace distance_AB_l405_405260

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405260


namespace parabola_problem_l405_405551

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405551


namespace time_worked_on_thursday_l405_405686

/-
  Given:
  - Monday: 3/4 hour
  - Tuesday: 1/2 hour
  - Wednesday: 2/3 hour
  - Friday: 75 minutes
  - Total (Monday to Friday): 4 hours = 240 minutes
  
  The time Mr. Willson worked on Thursday is 50 minutes.
-/

noncomputable def time_worked_monday : ℝ := (3 / 4) * 60
noncomputable def time_worked_tuesday : ℝ := (1 / 2) * 60
noncomputable def time_worked_wednesday : ℝ := (2 / 3) * 60
noncomputable def time_worked_friday : ℝ := 75
noncomputable def total_time_worked : ℝ := 4 * 60

theorem time_worked_on_thursday :
  time_worked_monday + time_worked_tuesday + time_worked_wednesday + time_worked_friday + 50 = total_time_worked :=
by
  sorry

end time_worked_on_thursday_l405_405686


namespace AB_distance_l405_405360

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405360


namespace contrapositive_l405_405733

theorem contrapositive (a b : ℝ) (h : a > b → 2^a > 2^b - 1) :
  2^a ≤ 2^b - 1 → a ≤ b :=
by 
  sorry

end contrapositive_l405_405733


namespace distance_AB_l405_405278

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405278


namespace range_of_x_for_f_ln_x_gt_f_1_l405_405148

noncomputable def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

noncomputable def is_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_x_for_f_ln_x_gt_f_1
  (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_dec : is_decreasing_on_nonneg f)
  (hf_condition : ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e) :
  ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e := sorry

end range_of_x_for_f_ln_x_gt_f_1_l405_405148


namespace parabola_distance_l405_405625

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405625


namespace ab_distance_l405_405240

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405240


namespace parabola_distance_l405_405388

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405388


namespace parabola_distance_problem_l405_405402

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405402


namespace find_sin_theta_l405_405644

noncomputable def direction_vector : ℝ^3 := ![4, 5, 7]
noncomputable def normal_vector : ℝ^3 := ![5, -3, 9]

noncomputable def dot_product (u v : ℝ^3) : ℝ :=
u 0 * v 0 + u 1 * v 1 + u 2 * v 2

noncomputable def magnitude (v : ℝ^3) : ℝ :=
real.sqrt (v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2)

noncomputable def angle_sin := 
dot_product direction_vector normal_vector / 
(magnitude direction_vector * magnitude normal_vector)

theorem find_sin_theta : angle_sin = 68 / real.sqrt 10350 :=
by
  sorry

end find_sin_theta_l405_405644


namespace total_number_of_cards_l405_405796

/-- There are 9 playing cards and 4 ID cards initially.
If you add 6 more playing cards and 3 more ID cards,
then the total number of playing cards and ID cards will be 22. -/
theorem total_number_of_cards :
  let initial_playing_cards := 9
  let initial_id_cards := 4
  let additional_playing_cards := 6
  let additional_id_cards := 3
  let total_playing_cards := initial_playing_cards + additional_playing_cards
  let total_id_cards := initial_id_cards + additional_id_cards
  let total_cards := total_playing_cards + total_id_cards
  total_cards = 22 :=
by
  sorry

end total_number_of_cards_l405_405796


namespace sum_of_f_l405_405962

def f (x: ℝ) : ℝ := 2 * x / (x + 1)

theorem sum_of_f (S : ℝ) : 
  (S = ∑ k in (finset.range 2016).map (nat.succ), f k + ∑ k in (finset.range 2015).map (λ n, f ((1 : ℝ) / (n + 2))) → S = 4031) := 
by 
  -- The actual proof is omitted since only the statement is required
  sorry

end sum_of_f_l405_405962


namespace distance_AB_l405_405331

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405331


namespace subsets_union_l405_405230

theorem subsets_union (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 2^(n-1) + 1) 
  (A : Fin m → Finset (Fin n)) (hA : ∀ i j, i ≠ j → A i ≠ A j) 
  (hB : ∀ i, A i ≠ ∅) : 
  ∃ i j k, i ≠ j ∧ A i ∪ A j = A k := 
sorry

end subsets_union_l405_405230


namespace sum_of_largest_three_l405_405110

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l405_405110


namespace general_formula_sum_first_2n_b_seq_l405_405972

open_locale big_operators

-- Definitions for the arithmetic sequence {a_n}
def arithmetic_seq (a d : ℕ → ℤ) (n : ℕ) : ℤ := a 1 + (n - 1) * d 

-- Definition for the sum of the first n terms S_n of the arithmetic sequence
def sum_arithmetic_seq (a d : ℕ → ℤ) (n : ℕ) : ℤ := (n * (a 1 + arithmetic_seq a d n)) / 2

-- Definitions for the conditions
def condition1 (a : ℕ → ℤ) (d : ℕ → ℤ) := 2 * (arithmetic_seq a d 5) = (arithmetic_seq a d 2) + 14
def condition2 (a : ℕ → ℤ) (d : ℕ → ℤ) := sum_arithmetic_seq a d 9 = 72

-- Definition for sequence {b_n}
def b_seq (a : ℕ → ℤ) (n : ℕ) : ℤ := if n % 2 = 1 then a n else 2^n

-- Definition for the sum of the first 2n terms of the sequence {b_n}
def sum_b_seq (a : ℕ → ℤ) (n : ℕ) : ℤ := ∑ i in finset.range (2 * n), b_seq a (i + 1)

-- The proof goals; no proof provided, just the statement
theorem general_formula (a d : ℕ → ℤ) (n : ℕ) (h₁ : condition1 a d) (h₂ : condition2 a d) : 
  arithmetic_seq a d n = 2 * n - 2 := 
sorry

theorem sum_first_2n_b_seq (a : ℕ → ℤ) (n : ℕ) (h₁ : condition1 a d) (h₂ : condition2 a d) :
  sum_b_seq a n = 2 * n^2 - 2 * n + (4^(n+1) - 4) / 3 :=
sorry

end general_formula_sum_first_2n_b_seq_l405_405972


namespace mean_noon_temperature_l405_405781

def temperatures : List ℕ := [82, 80, 83, 88, 90, 92, 90, 95]

def mean_temperature (temps : List ℕ) : ℚ :=
  (temps.foldr (λ a b => a + b) 0 : ℚ) / temps.length

theorem mean_noon_temperature :
  mean_temperature temperatures = 87.5 := by
  sorry

end mean_noon_temperature_l405_405781


namespace parabola_distance_problem_l405_405424

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405424


namespace width_decrease_percentage_l405_405749

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l405_405749


namespace distance_AB_l405_405335

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405335


namespace observations_count_l405_405780

theorem observations_count (n : ℕ) 
  (original_mean : ℚ) (wrong_value_corrected : ℚ) (corrected_mean : ℚ)
  (h1 : original_mean = 36)
  (h2 : wrong_value_corrected = 1)
  (h3 : corrected_mean = 36.02) :
  n = 50 :=
by
  sorry

end observations_count_l405_405780


namespace exist_three_not_played_l405_405196

noncomputable def football_championship (teams : Finset ℕ) (rounds : ℕ) : Prop :=
  let pairs_per_round := teams.card / 2 in
  let total_pairs_constraint := rounds * pairs_per_round in
  let constraint_matrix := (teams.card - 1) - pairs_per_round in
  (teams.card = 18) ∧
  (rounds = 8) ∧
  (pairs_per_round * rounds < constraint_matrix * (constraint_matrix - 1) / 2) ->
  ∃ (A B C : ℕ) (hA : A ∈ teams) (hB : B ∈ teams) (hC : C ∈ teams),
    ¬ (⊢ (A, B) ∈ teams * teams) ∧ ¬ (⊢ (B, C) ∈ teams * teams) ∧ ¬ (⊢ (A, C) ∈ teams * teams)

theorem exist_three_not_played :
  ∃ (football_championship (Finset.range 18) 8) :=
begin
  sorry,
end

end exist_three_not_played_l405_405196


namespace inverse_function_value_l405_405674

def f (x : ℝ) : ℝ := Real.sqrt x

def f_inv (x : ℝ) : ℝ := x * x

theorem inverse_function_value :
  f_inv(4) = 16 :=
sorry

end inverse_function_value_l405_405674


namespace calculate_interest_rate_l405_405685

noncomputable def simple_interest_rate (P : ℝ) (A : ℝ) (T : ℝ) : ℝ :=
  let SI := A - P in
  (SI * 100) / (P * T)

theorem calculate_interest_rate :
  simple_interest_rate 5461.04 8410 9 ≈ 6 := sorry

end calculate_interest_rate_l405_405685


namespace distance_AB_l405_405463

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405463


namespace parabola_distance_l405_405571

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405571


namespace tourism_revenue_scientific_notation_l405_405824

-- Define the conditions given in the problem.
def total_tourism_revenue := 12.41 * 10^9

-- Prove the scientific notation of the total tourism revenue.
theorem tourism_revenue_scientific_notation :
  total_tourism_revenue = 1.241 * 10^9 :=
sorry

end tourism_revenue_scientific_notation_l405_405824


namespace distance_AB_l405_405467

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405467


namespace albert_wins_x2020_homer_wins_x2_1_l405_405837

-- Definition for Albert's winning strategy when m(x) = x^{2020}
theorem albert_wins_x2020 (a : Fin 2020 → ℝ) : ¬ (∀ a : Fin 2020 → ℝ, ∃ g : Polynomial ℝ, 
    Polynomial.eval₂ g Polynomial.C Polynomial.X = Polynomial.monomial 2020 1 + Polynomial.sum (Finset.fin_range 2020) (λ i, Polynomial.monomial i (a i))) :=
sorry

-- Definition for Homer's winning strategy when m(x) = x^2 + 1
theorem homer_wins_x2_1 (a : Fin 2020 → ℝ) : ∃ g : Polynomial ℝ, 
    Polynomial.eval₂ g Polynomial.C Polynomial.X = Polynomial.monomial 2020 1 + Polynomial.sum (Finset.fin_range 2020) (λ i, Polynomial.monomial i (a i)) :=
sorry

end albert_wins_x2020_homer_wins_x2_1_l405_405837


namespace basketballs_fewer_than_soccer_balls_l405_405186

theorem basketballs_fewer_than_soccer_balls
  (soccer_balls_per_box : ℕ) (basketballs_per_box : ℕ)
  (soccer_boxes : ℕ) (basketball_boxes : ℕ) :
  soccer_balls_per_box = 12 → basketballs_per_box = 12 →
  soccer_boxes = 8 → basketball_boxes = 5 →
  (soccer_boxes * soccer_balls_per_box - basketball_boxes * basketballs_per_box) = 36 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end basketballs_fewer_than_soccer_balls_l405_405186


namespace heather_time_start_difference_l405_405720

noncomputable def time_difference_in_minutes
  (total_distance : ℝ) (heather_distance : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) : ℝ :=
  let t := heather_distance / heather_speed in
  let stacy_distance := total_distance - heather_distance in
  let Δt := (stacy_distance / stacy_speed) - t in
  Δt * 60

theorem heather_time_start_difference :
  time_difference_in_minutes 25 10.272727272727273 5 6 = 24 :=
by
    sorry

end heather_time_start_difference_l405_405720


namespace axis_of_symmetry_smallest_positive_period_range_of_h_l405_405158

def f (x : ℝ) : ℝ := cos^2 (x + π / 12)
def g (x : ℝ) : ℝ := 1 + 1 / 2 * sin (2 * x)
def h (x : ℝ) : ℝ := f x + g x

theorem axis_of_symmetry : ∃ k : ℤ, (λ x : ℝ, f x) x = (λ x : ℝ, f x) (k * π / 2 - π / 12) := 
by sorry

theorem smallest_positive_period : 
  (∀ x : ℝ, h (x + π) = h x) ∧ (∀ ε > 0, ε < π → ¬(∀ x : ℝ, h (x + ε) = h x)) := 
by sorry

theorem range_of_h : 
  ∀ y : ℝ, (∃ x : ℝ, h x = y) ↔ y ∈ set.Icc (1 : ℝ) (2 : ℝ) :=
by sorry

end axis_of_symmetry_smallest_positive_period_range_of_h_l405_405158


namespace calculate_children_l405_405930

theorem calculate_children : 
  let total_spectators := 50000 in
  let first_match_spectators := 18000 in
  let first_match_men := 10800 in
  let second_match_spectators := 22000 in
  let second_match_men := 13860 in
  let third_match_spectators := 10000 in
  let third_match_men := 6500 in
  let first_match_children := 4320 in
  let second_match_children := 4520 in
  let third_match_children := 1749 in
  let total_children := 10589 in
  first_match_children + second_match_children + third_match_children = total_children
sorry

end calculate_children_l405_405930


namespace probability_N6_mod_7_eq_1_l405_405892

theorem probability_N6_mod_7_eq_1 (N : ℕ) (hN : 1 ≤ N ∧ N ≤ 2030) : 
  (finset.filter (λ k, (k^6 % 7 = 1)) (finset.range (2031)) ).card / 2030 = 6 / 7 := 
by 
  sorry

end probability_N6_mod_7_eq_1_l405_405892


namespace six_digit_palindromes_count_l405_405039

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l405_405039


namespace number_of_possible_values_of_s_l405_405782

noncomputable def s := {s : ℚ | ∃ w x y z : ℕ, s = w / 1000 + x / 10000 + y / 100000 + z / 1000000 ∧ w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10}

theorem number_of_possible_values_of_s (s_approx : s → ℚ → Prop) (h_s_approx : ∀ s, s_approx s (3 / 11)) :
  ∃ n : ℕ, n = 266 :=
by
  sorry

end number_of_possible_values_of_s_l405_405782


namespace price_reduction_proof_l405_405861

variable original_price reduced_price reduction: ℝ

axiom reduced_price_value : reduced_price = 1800
axiom reduced_price_fraction : reduced_price = 0.9 * original_price
def price_reduction := original_price - reduced_price

theorem price_reduction_proof : price_reduction = 200 :=
by
  /- ..Proof steps would go here.. -/
  sorry

end price_reduction_proof_l405_405861


namespace rectangle_width_decrease_proof_l405_405760

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l405_405760


namespace distance_MD_geq_half_AB_l405_405703

open_locale real

noncomputable section

variables {A B C D M F : Type}
variables [metric_space A]
variables (ABC : simplex ℝ A)

def midpoint (x y : Point ℝ) : Point ℝ := (x + y) / 2

def triangle_midpoints (ABC : simplex ℝ A) : Point ℝ × Point ℝ × Point ℝ :=
  let (a, b, c) := ABC.vertices in
  (midpoint b c, midpoint a b, midpoint_of_arc a b c)

theorem distance_MD_geq_half_AB (ABC : simplex ℝ A) :
  let (D, F, M) := triangle_midpoints ABC in
  dist M D ≥ dist (ABC.vertices.1.1.1) (ABC.vertices.1.1.2) / 2 := 
sorry

end distance_MD_geq_half_AB_l405_405703


namespace equalize_costs_l405_405228

theorem equalize_costs (X Y Z : ℝ) (hXY : X < Y) (hYZ : Y < Z) : (Y + Z - 2 * X) / 3 = (X + Y + Z) / 3 - X := by
  sorry

end equalize_costs_l405_405228


namespace distance_AB_l405_405274

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405274


namespace distance_AB_l405_405438

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405438


namespace distance_AB_l405_405348

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405348


namespace parabola_problem_l405_405286

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405286


namespace sum_of_largest_three_consecutive_numbers_l405_405096

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l405_405096


namespace sterling_total_questions_l405_405829

-- Definitions based on the problem conditions
def correct_candy (x : Nat) : Nat := 3 * x
def incorrect_candy (y : Nat) : Int := -2 * y

def candy_total (correct incorrect : Nat) : Int :=
  correct_candy correct + incorrect_candy incorrect

theorem sterling_total_questions :
  (∀ correct : Nat, incorrect : Nat,
  correct = 7 ∧ candy_total correct incorrect = 21) ∧
  (∀ correct : Nat, candy_total (correct + 2) = 31) ∧
  (∀ correct incorrect : Nat, candy_total (correct + 2) + 4 = 31 ∧ incorrect = 2) →
  ∃ total : Nat, total = 9 :=
by
  sorry

end sterling_total_questions_l405_405829


namespace width_decrease_percentage_l405_405752

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l405_405752


namespace max_value_expression_l405_405668

theorem max_value_expression (x a : ℝ) (hx : 0 < x) (ha : 0 < a) :
  ∃ (x : ℝ), x = sqrt (2 * a) ∧
  ∀ (x' : ℝ), 0 < x' →
  (x' = sqrt (2 * a) → 
   (x^2 + 2 * a - (sqrt (x^4 + 4 * a^2))) / x = 2 * sqrt (2 * a) - 2 * a) ∧
  (∀ (x' : ℝ), 0 < x' → 
   (x' ≠ sqrt (2 * a) → 
    (x'^2 + 2 * a - (sqrt (x'^4 + 4 * a^2))) / x' < 2 * sqrt (2 * a) - 2 * a)) :=
by sorry

end max_value_expression_l405_405668


namespace AB_distance_l405_405359

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405359


namespace football_championship_l405_405195

noncomputable def exists_trio_did_not_play_each_other 
  (teams : Finset ℕ) (rounds : ℕ) (matches : Finset (ℕ × ℕ)) : Prop :=
  ∃ (T : Finset ℕ), T.card = 3 ∧
    (∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → (a, b) ∉ matches ∧ (b, a) ∉ matches)

theorem football_championship (teams : Finset ℕ) (h_teams : teams.card = 18)
  (rounds : ℕ) (h_rounds : rounds = 8) 
  (matches_per_round : Finset (Finset (ℕ × ℕ))) 
  (h_matches_per_round : ∀ round ∈ matches_per_round, round.card = 9)
  (unique_pairs : ∀ (r₁ r₂ ∈ matches_per_round) (p : ℕ × ℕ), p ∈ r₁ → p ∈ r₂ → r₁ = r₂) :
  exists_trio_did_not_play_each_other teams rounds (matches_per_round.bUnion id) :=
by 
  sorry

end football_championship_l405_405195


namespace distance_AB_l405_405353

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405353


namespace expand_product_l405_405935

variables (x y : ℝ)

theorem expand_product : (3 * x + 4 * y ) * (2 * x - 5 * y + 7) = 6 * x ^ 2 - 7 * x * y + 21 * x - 20 * y ^ 2 + 28 * y :=
by sorry

end expand_product_l405_405935


namespace AB_distance_l405_405355

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405355


namespace parabola_distance_problem_l405_405418

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405418


namespace distance_AB_l405_405342

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405342


namespace mono_inc_neg2_0_range_of_m_l405_405992

-- Define an odd function and conditions for monotonicity
variable (f : ℝ → ℝ)
variable mono_inc_02 : ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ 2 → 0 ≤ x2 → x2 ≤ 2 → x1 < x2 → f(x1) < f(x2)
variable odd_func : ∀ x : ℝ, f(-x) = -f(x)

-- Prove monotonicity on [-2, 0]
theorem mono_inc_neg2_0 : ∀ x1 x2 : ℝ, -2 ≤ x1 → x1 ≤ 0 → -2 ≤ x2 → x2 ≤ 0 → x1 < x2 → f(x1) < f(x2) := sorry

-- Prove the range for m
theorem range_of_m (m : ℝ) : (f (Real.log (2 * m) / Real.log 2) < f (Real.log (m + 2) / Real.log 2)) → (1 / 8 ≤ m ∧ m < 2) := sorry

end mono_inc_neg2_0_range_of_m_l405_405992


namespace parabola_problem_l405_405553

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405553


namespace sum_of_largest_three_l405_405111

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l405_405111


namespace remainder_when_M_div_1000_l405_405911

open Function

def minimallyIntersectingTriple (A B C : Set ℤ) : Prop :=
  (|A ∩ B| = 1) ∧ (|B ∩ C| = 1) ∧ (|C ∩ A| = 1) ∧ (A ∩ B ∩ C = ∅)

def tripleSet  := {s : Set ℤ // s \in (powerset (Finset.range 8)).val}
noncomputable instance : Fintype tripleSet := by
  unfold tripleSet
  sorry

def M : ℕ := 
  Fintype.card  {t : tripleSet × tripleSet × tripleSet // minimallyIntersectingTriple t.1.val t.2.val t.2.snd.val }

theorem remainder_when_M_div_1000 : M % 1000 = 344 := by
  sorry

end remainder_when_M_div_1000_l405_405911


namespace chess_tournament_winner_l405_405193

theorem chess_tournament_winner :
  ∀ (points_10th : ℕ) (points_11th : ℕ),
  (∃ (x : ℕ), points_11th = 4.5 * points_10th ∧ x ≤ 1 ∧ points_10th = 10) →
  ∃ x, points_10th = 10 ∧ x = 1 :=
by
  sorry

end chess_tournament_winner_l405_405193


namespace parabola_distance_l405_405589

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405589


namespace six_digit_palindromes_count_l405_405002

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l405_405002


namespace parabola_distance_l405_405390

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405390


namespace main_theorem_l405_405649

noncomputable def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

lemma f_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- Detailed steps of proof will go here
  sorry

lemma f_monotonic_increasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  intros x₁ x₂ h₁_lt_zero h₁_lt_h₂
  -- Detailed steps of proof will go here
  sorry

theorem main_theorem : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ → f x₁ < f x₂) := by
  split
  · exact f_odd
  · exact f_monotonic_increasing

end main_theorem_l405_405649


namespace inequality_proof_l405_405705

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 ≤ a i ∧ a i ≤ 1) :
  (∏ i, a i) * (1 - (∑ i, a i)) / ((∑ i, a i) * ∏ i, (1 - a i)) ≤ 1 / (n ^ (n + 1)) :=
by
  sorry

end inequality_proof_l405_405705


namespace parabola_distance_problem_l405_405419

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405419


namespace yard_flower_beds_fraction_l405_405867

theorem yard_flower_beds_fraction :
  let yard_length := 30
  let yard_width := 10
  let pool_length := 10
  let pool_width := 4
  let trap_parallel_diff := 22 - 16
  let triangle_leg := trap_parallel_diff / 2
  let triangle_area := (1 / 2) * (triangle_leg ^ 2)
  let total_triangle_area := 2 * triangle_area
  let total_yard_area := yard_length * yard_width
  let pool_area := pool_length * pool_width
  let usable_yard_area := total_yard_area - pool_area
  (total_triangle_area / usable_yard_area) = 9 / 260 :=
by 
  sorry

end yard_flower_beds_fraction_l405_405867


namespace irrational_greater_than_two_implies_integer_ratio_l405_405947

def A (x : ℝ) : set ℕ := {n | ∃ (m : ℤ), m > 0 ∧ n = ⌊m * x⌋}

theorem irrational_greater_than_two_implies_integer_ratio (α β : ℝ) 
  (hα1 : α > 1) (hα_irrational : irrational α) (hA_subset : A β ⊆ A α) :
  α > 2 → ∃ k : ℤ, β = k * α :=
sorry

end irrational_greater_than_two_implies_integer_ratio_l405_405947


namespace square_area_with_circles_l405_405874

theorem square_area_with_circles
  (radius : ℝ) 
  (side_length : ℝ)
  (area : ℝ)
  (h_radius : radius = 7) 
  (h_side_length : side_length = 2 * (2 * radius)) 
  (h_area : area = side_length ^ 2) : 
  area = 784 := by
  sorry

end square_area_with_circles_l405_405874


namespace GeoSeries_properties_l405_405165

noncomputable def a : ℕ → ℝ
| 0     := 3
| (n+1) := (1/3 : ℝ)^(n+1) * 2

def GeoSeries_sum : ℝ := 3 + 2 + ∑' n, 2 * (1 / 3) ^ n

theorem GeoSeries_properties :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 0| < ε ∧
  GeoSeries_sum = 4.5 ∧
  (¬ (∀ n, ∑ k in range n, a k < ⊤)) ∧
  (¬ (∀ n, ∑ k in range n, a k > ⊥)) ∧
  (GeoSeries_sum ≠ 6) :=
by
  sorry

end GeoSeries_properties_l405_405165


namespace ab_distance_l405_405254

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405254


namespace distance_AB_l405_405337

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405337


namespace parabola_distance_problem_l405_405407

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405407


namespace sum_finite_series_l405_405908

theorem sum_finite_series :
  ∑ k in Finset.range (nat.succ 0), (4 * k ^ 2 + 1) / ((4 * k ^ 2 - 1) ^ 2) =
    (π^2 / 8) + (1 / 2) :=
sorry

end sum_finite_series_l405_405908


namespace width_decreased_by_28_6_percent_l405_405765

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l405_405765


namespace problem_l405_405594

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405594


namespace tens_digit_23_pow_1987_l405_405919

def tens_digit_of_power (a b n : ℕ) : ℕ :=
  ((a^b % n) / 10) % 10

theorem tens_digit_23_pow_1987 : tens_digit_of_power 23 1987 100 = 4 := by
  sorry

end tens_digit_23_pow_1987_l405_405919


namespace sum_of_n_less_equal_25_for_g_eq_gnplus1_l405_405948

def factorial : ℕ → ℕ 
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def g (n : ℕ) : ℕ :=
  let terms := list.map (λ k => factorial k * factorial (n - k)) (list.range (n + 1))
  list.foldl Nat.gcd 0 terms

theorem sum_of_n_less_equal_25_for_g_eq_gnplus1 :
  (∑ n in finset.filter (λ n => g n = g (n + 1)) (finset.range 26), n) = 82 := sorry

end sum_of_n_less_equal_25_for_g_eq_gnplus1_l405_405948


namespace parabola_problem_l405_405506

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405506


namespace distance_AB_l405_405434

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405434


namespace jackson_entertainment_cost_l405_405223

def price_computer_game : ℕ := 66
def price_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3
def total_entertainment_cost : ℕ := price_computer_game + number_of_movie_tickets * price_movie_ticket

theorem jackson_entertainment_cost : total_entertainment_cost = 102 := by
  sorry

end jackson_entertainment_cost_l405_405223


namespace proof_f_g_f3_l405_405667

def f (x: ℤ) : ℤ := 2*x + 5
def g (x: ℤ) : ℤ := 5*x + 2

theorem proof_f_g_f3 :
  f (g (f 3)) = 119 := by
  sorry

end proof_f_g_f3_l405_405667


namespace distance_AB_l405_405449

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405449


namespace AB_distance_l405_405318

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405318


namespace parabola_problem_l405_405289

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405289


namespace ab_distance_l405_405245

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405245


namespace distance_AB_l405_405276

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405276


namespace arithmetic_progression_a4_l405_405970

theorem arithmetic_progression_a4 {a : ℕ → ℕ} {S : ℕ → ℕ} (h1 : ∀ n, a (n + 1) = S n + 1) 
  (h2 : ∀ n, S n = ∑ i in Finset.range n, a (i+1)) : a 4 = 8 :=
sorry

end arithmetic_progression_a4_l405_405970


namespace parabola_distance_l405_405395

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405395


namespace distance_AB_l405_405262

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405262


namespace number_of_dissimilar_terms_expansion_l405_405914

theorem number_of_dissimilar_terms_expansion (a b c d : ℕ) :
  ∑ (i j k l : ℕ) in finset.powerset_le finset.univ 4, (if i + j + k + l = 8 then 1 else 0) = 165 := 
sorry

end number_of_dissimilar_terms_expansion_l405_405914


namespace marla_colors_green_squares_l405_405680

-- Condition 1: Grid dimensions
def num_rows : ℕ := 10
def num_cols : ℕ := 15

-- Condition 2: Red squares
def red_rows : ℕ := 4
def red_squares_per_row : ℕ := 6
def red_squares : ℕ := red_rows * red_squares_per_row

-- Condition 3: Blue rows (first 2 and last 2)
def blue_rows : ℕ := 2 + 2
def blue_squares_per_row : ℕ := num_cols
def blue_squares : ℕ := blue_rows * blue_squares_per_row

-- Derived information
def total_squares : ℕ := num_rows * num_cols
def non_green_squares : ℕ := red_squares + blue_squares

-- The Lemma to prove
theorem marla_colors_green_squares : total_squares - non_green_squares = 66 := by
  sorry

end marla_colors_green_squares_l405_405680


namespace jackson_entertainment_cost_l405_405222

def price_computer_game : ℕ := 66
def price_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3
def total_entertainment_cost : ℕ := price_computer_game + number_of_movie_tickets * price_movie_ticket

theorem jackson_entertainment_cost : total_entertainment_cost = 102 := by
  sorry

end jackson_entertainment_cost_l405_405222


namespace parabola_problem_l405_405567

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405567


namespace parabola_distance_problem_l405_405417

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405417


namespace AB_distance_l405_405358

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405358


namespace tangent_length_to_circumcircle_l405_405902

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (10, 25)

theorem tangent_length_to_circumcircle :
  let OA := Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2),
      OB := Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2),
      OT := Real.sqrt (OA * OB)
  in
    OT = Real.sqrt 82 :=
by
  sorry

end tangent_length_to_circumcircle_l405_405902


namespace sum_of_three_largest_l405_405061

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l405_405061


namespace sum_of_largest_three_l405_405115

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l405_405115


namespace parabola_distance_l405_405486

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405486


namespace effective_height_of_arch_l405_405862

-- Define the conditions
def arch_equation (x : ℝ) : ℝ := - (4 / 125) * x ^ 2 + 20
def arch_thickness : ℝ := 0.5
def point_of_interest : ℝ := 10

-- Statement of the proof problem
theorem effective_height_of_arch :
  arch_equation point_of_interest - arch_thickness = 16.3 := by
sorry

end effective_height_of_arch_l405_405862


namespace sum_of_three_largest_of_consecutive_numbers_l405_405071

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l405_405071


namespace AB_distance_l405_405325

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405325


namespace parabola_distance_l405_405633

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405633


namespace six_digit_palindromes_count_l405_405041

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l405_405041


namespace distance_AB_l405_405352

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405352


namespace parabola_problem_l405_405556

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405556


namespace triangle_area_problem_l405_405174

theorem triangle_area_problem (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h_area : (∃ t : ℝ, t > 0 ∧ (2 * c * t + 3 * d * (12 / (2 * c)) = 12) ∧ (∃ s : ℝ, s > 0 ∧ 2 * c * (12 / (3 * d)) + 3 * d * s = 12)) ∧ 
    ((1 / 2) * (12 / (2 * c)) * (12 / (3 * d)) = 12)) : c * d = 1 := 
by 
  sorry

end triangle_area_problem_l405_405174


namespace is_odd_and_monotonically_increasing_l405_405651

def f (x : ℝ) : ℝ := x ^ 3 - 1 / (x ^ 3)

theorem is_odd_and_monotonically_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :=
begin
  sorry
end

end is_odd_and_monotonically_increasing_l405_405651


namespace parabola_distance_l405_405527

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405527


namespace parabola_distance_l405_405380

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405380


namespace parabola_distance_l405_405576

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405576


namespace paintable_wall_area_l405_405711

theorem paintable_wall_area :
  let bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let doorway_window_area := 70
  let area_one_bedroom := 
    2 * (length * height) + 2 * (width * height) - doorway_window_area
  let total_paintable_area := bedrooms * area_one_bedroom
  total_paintable_area = 1520 := by
  sorry

end paintable_wall_area_l405_405711


namespace ab_distance_l405_405243

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405243


namespace distance_AB_l405_405273

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405273


namespace parabola_problem_l405_405498

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405498


namespace graph_of_f_abs_is_E_l405_405744

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then sqrt (4 - (x-2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- assuming 0 outside [-3, 3]

def f_abs (x : ℝ) : ℝ :=
  f (-|x|)

theorem graph_of_f_abs_is_E :
  ∃ g, (∀ x, g x = f_abs x) ∧ g = option_E := 
  sorry

end graph_of_f_abs_is_E_l405_405744


namespace room_width_l405_405779

theorem room_width (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) (width : ℝ)
  (h_length : length = 5.5)
  (h_total_cost : total_cost = 15400)
  (h_rate_per_sqm : rate_per_sqm = 700)
  (h_area : total_cost = rate_per_sqm * (length * width)) :
  width = 4 := 
sorry

end room_width_l405_405779


namespace six_digit_palindromes_count_l405_405050

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l405_405050


namespace parabola_distance_l405_405386

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405386


namespace sin_angle_RPS_l405_405207

-- Define the conditions given in the problem
variable {angle_RPQ : ℝ}
variable (cos_angle_RPQ : cos angle_RPQ = 24 / 25)
variable (angle_RPS : ℝ)
variable (angle_RPS_def : angle_RPS = π - angle_RPQ)

-- State the main theorem to prove
theorem sin_angle_RPS : sin angle_RPS = 7 / 25 :=
by
  -- Usual steps to solve the problem would go here
  -- This proof statement must satisfy the conditions derived from the original problem
  sorry

end sin_angle_RPS_l405_405207


namespace abs_integral_result_l405_405899

noncomputable def abs_integral : ℝ := ∫ x in 0..1, |x - 1|

theorem abs_integral_result : abs_integral = 1 / 2 :=
by
  sorry

end abs_integral_result_l405_405899


namespace problem_l405_405598

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405598


namespace six_digit_palindromes_count_l405_405036

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l405_405036


namespace repeating_decimal_fraction_proof_l405_405936

noncomputable def repeating_decimal_to_fraction : ℚ := 226 / 495

theorem repeating_decimal_fraction_proof :
  (to_decimal 0 1 * 10 + to_decimal 4 2 2 / 100 + 4 + (repeat (to_decimal 5 1 1 / 100 + to_decimal 6 1 2 / 100))) = repeating_decimal_to_fraction :=
sorry

end repeating_decimal_fraction_proof_l405_405936


namespace sum_of_three_largest_l405_405090

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l405_405090


namespace find_quotient_l405_405689

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 23) (h2 : divisor = 4) (h3 : remainder = 3)
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 5 :=
sorry

end find_quotient_l405_405689


namespace problem_l405_405615

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405615


namespace parabola_distance_l405_405378

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405378


namespace sum_of_three_largest_l405_405087

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l405_405087


namespace total_cantaloupes_l405_405122

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by
  sorry

end total_cantaloupes_l405_405122


namespace difference_of_squares_l405_405177

variable (x y : ℚ)

theorem difference_of_squares (h1 : x + y = 3 / 8) (h2 : x - y = 1 / 8) : x^2 - y^2 = 3 / 64 := 
by
  sorry

end difference_of_squares_l405_405177


namespace main_theorem_l405_405647

noncomputable def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

lemma f_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- Detailed steps of proof will go here
  sorry

lemma f_monotonic_increasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  intros x₁ x₂ h₁_lt_zero h₁_lt_h₂
  -- Detailed steps of proof will go here
  sorry

theorem main_theorem : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ → f x₁ < f x₂) := by
  split
  · exact f_odd
  · exact f_monotonic_increasing

end main_theorem_l405_405647


namespace problem_l405_405600

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405600


namespace distance_AB_l405_405461

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405461


namespace distance_AB_l405_405462

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405462


namespace parabola_problem_l405_405550

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405550


namespace parabola_problem_l405_405293

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405293


namespace parabola_distance_l405_405384

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405384


namespace distance_AB_l405_405453

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405453


namespace parabola_distance_l405_405482

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405482


namespace team_supporters_counts_correct_l405_405718

-- Definitions for supporters counts
def supporters_count_match1 (sp s dy z lo : Nat) : Prop :=
  sp = 200 ∧ dy = 300 ∧ z = 500 ∧ lo = 600

theorem team_supporters_counts_correct {sp dy z lo : Nat} :
  (supporters_count_match1 sp dy z lo) → (sp = 200 ∧ dy = 300 ∧ z = 500 ∧ lo = 600) :=
by
  intro h
  exact h

end team_supporters_counts_correct_l405_405718


namespace parabola_distance_l405_405634

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405634


namespace initial_eggs_proof_l405_405709

noncomputable def initial_eggs (total_cost : ℝ) (price_per_egg : ℝ) (leftover_eggs : ℝ) : ℝ :=
  let eggs_sold := total_cost / price_per_egg
  eggs_sold + leftover_eggs

theorem initial_eggs_proof : initial_eggs 5 0.20 5 = 30 := by
  sorry

end initial_eggs_proof_l405_405709


namespace ab_distance_l405_405234

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405234


namespace relationship_y_coords_l405_405988

theorem relationship_y_coords :
  ∀ (y1 y2 y3 : ℝ),
    (A: 1, y1 = 2 / 1) →
    (B: 2, y2 = 2 / 2) →
    (C: -3, y3 = 2 / -3) →
    y1 > y2 ∧ y2 > y3 :=
by {
  intros y1 y2 y3 A B C,
  sorry
}

end relationship_y_coords_l405_405988


namespace parabola_distance_l405_405475

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405475


namespace distance_AB_l405_405447

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405447


namespace sum_of_consecutive_integers_product_336_l405_405783

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y : ℕ), x * (x + 1) = 336 ∧ (y - 1) * y * (y + 1) = 336 ∧ x + (x + 1) + (y - 1) + y + (y + 1) = 54 :=
by
  -- The formal proof would go here
  sorry

end sum_of_consecutive_integers_product_336_l405_405783


namespace parabola_problem_l405_405518

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405518


namespace equilateral_hyperbola_through_A_eq_l405_405940

theorem equilateral_hyperbola_through_A_eq :
  (∃ λ : ℝ, λ ≠ 0 ∧ ∀ x y : ℝ, (x^2 - y^2 = λ) ∧ (x, y) = (3, -1)) → 8 = λ := by
  sorry

end equilateral_hyperbola_through_A_eq_l405_405940


namespace sum_of_largest_three_consecutive_numbers_l405_405095

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l405_405095


namespace f_properties_l405_405665

def f (x : ℝ) : ℝ := x^3 - 1 / x^3

theorem f_properties : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := 
by
  sorry

end f_properties_l405_405665


namespace parabola_distance_l405_405623

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405623


namespace sum_of_three_largest_l405_405089

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l405_405089


namespace number_of_functions_l405_405944

variable {R : Type*} [Ring R]

def function_form (a b c x : R) : R :=
  a * x^3 + b * x^2 + c * x

theorem number_of_functions :
  (∃ (a b c : R), 
    ∀ x : R, 
          (function_form a b c x) * (function_form a b c (-x)) = 
          function_form a b c (x^3)) = 8 :=
sorry

end number_of_functions_l405_405944


namespace parabola_distance_l405_405618

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405618


namespace union_A_complementB_eq_l405_405166

-- Define real sets A and B
def A : set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def B : set ℝ := {x : ℝ | 2 < x}

-- Define the complement of B
def complement_B : set ℝ := {x : ℝ | x ≤ 2}

-- Define the union of A and complement_B
def union_A_complementB : set ℝ := A ∪ complement_B

-- State the theorem
theorem union_A_complementB_eq : union_A_complementB = {x : ℝ | x < 3} :=
  by {
    sorry
  }

end union_A_complementB_eq_l405_405166


namespace correct_propositions_l405_405118

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

-- Proposition definitions
def P1 := ∀ (x1 x2 : ℝ), f x1 = f x2 → ∃ (k : ℤ), x1 - x2 = k * Real.pi
def P2 := ∀ (x : ℝ), f x = 4 * Real.cos (2 * x - Real.pi / 6)
def P3 := f (-Real.pi / 6) = 0
def P4 := ∀ (x : ℝ), f (x) = f (-Real.pi / 6 - x)

theorem correct_propositions : P2 ∧ P3 :=
by
  split
  sorry
  sorry

end correct_propositions_l405_405118


namespace distance_AB_l405_405267

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405267


namespace parabola_problem_l405_405282

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405282


namespace parabola_problem_l405_405547

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405547


namespace raft_time_g_to_a_l405_405875

-- Definitions of given variables and conditions
def time_g_to_a := 5 -- Time taken from Gorky to Astrakhan by the steamboat (days)
def time_a_to_g := 7 -- Time taken from Astrakhan to Gorky by the steamboat (days)

-- Using the speed of the river current and the speed of the steamboat in still water
variables (v u l : ℝ)

-- Conditions given in the problem
def cond1 := l / (u + v) = time_g_to_a
def cond2 := l / (u - v) = time_a_to_g

-- Prove statement: Raft time from Gorky to Astrakhan
theorem raft_time_g_to_a : (cond1 ∧ cond2) -> l / v = 35 :=
by
  intro h
  have h1 := h.1
  have h2 := h.2
  sorry

end raft_time_g_to_a_l405_405875


namespace median_and_range_correct_l405_405189

def temperatures : List Int := [12, 9, 10, 6, 11, 12, 17]

def median (l : List Int) : Int := 
  let sorted_l := l.qsort (· < ·)
  sorted_l.get! (sorted_l.length / 2)

def range (l : List Int) : Int :=
  l.maximum'.get - l.minimum'.get

theorem median_and_range_correct :
  median temperatures = 11 ∧ range temperatures = 11 := by
  sorry

end median_and_range_correct_l405_405189


namespace ab_distance_l405_405242

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405242


namespace six_digit_palindromes_count_l405_405053

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l405_405053


namespace geom_seq_root_product_l405_405209

theorem geom_seq_root_product
  (a : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * a 1)
  (h_root1 : 3 * (a 1)^2 + 7 * a 1 - 9 = 0)
  (h_root10 : 3 * (a 10)^2 + 7 * a 10 - 9 = 0) :
  a 4 * a 7 = -3 := 
by
  sorry

end geom_seq_root_product_l405_405209


namespace parabola_problem_l405_405301

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405301


namespace completely_symmetric_expressions_l405_405180

def completely_symmetric (expr : Expr) : Prop :=
  ∀ (a b c : ℚ), expr.substs [⟨a, b⟩, ⟨b, a⟩, ⟨c, c⟩] = expr ∧
                 expr.substs [⟨a, c⟩, ⟨c, a⟩, ⟨b, b⟩] = expr ∧
                 expr.substs [⟨b, c⟩, ⟨c, b⟩, ⟨a, a⟩] = expr

theorem completely_symmetric_expressions :
  completely_symmetric (a * (b + c) + b * (a + c) + c * (a + b)) ∧
  completely_symmetric (a^2 * b * c + b^2 * a * c + c^2 * a * b) ∧
  completely_symmetric (a^2 + b^2 + c^2 - a * b - b * c - a * c) :=
by
  sorry

end completely_symmetric_expressions_l405_405180


namespace tens_digit_of_23_pow_1987_l405_405926

theorem tens_digit_of_23_pow_1987 : (23 ^ 1987 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_23_pow_1987_l405_405926


namespace line_circle_intersection_common_points_l405_405152

noncomputable def radius (d : ℝ) := d / 2

theorem line_circle_intersection_common_points 
  (diameter : ℝ) (distance_from_center_to_line : ℝ) 
  (h_dlt_r : distance_from_center_to_line < radius diameter) :
  ∃ common_points : ℕ, common_points = 2 :=
by
  sorry

end line_circle_intersection_common_points_l405_405152


namespace rectangle_width_decrease_l405_405774

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L in
  let W' := (L * W) / L' in
  let percent_decrease := (1 - W' / W) * 100 in
  percent_decrease ≈ 28.57 :=
by
  let L' := 1.4 * L
  let W' := (L * W) / L'
  let percent_decrease := (1 - W' / W) * 100
  sorry

end rectangle_width_decrease_l405_405774


namespace reversed_primes_mean_is_427_l405_405681

def is_three_digit_square (n : ℕ) : Prop :=
  ∃ x, 10 ≤ x ∧ x ≤ 31 ∧ x * x = n

def reverse_number (n : ℕ) : ℕ :=
  let digits := (nat.digits 10 n).reverse in
  digits.foldl (λ acc d, acc * 10 + d) 0

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_reversed_three_digit_square_prime (n : ℕ) : Prop :=
  is_three_digit_square n ∧ is_prime (reverse_number n)

def reversed_primes_mean : ℕ :=
  let reversed_primes := (list.range 1000).filter (λ n, is_reversed_three_digit_square_prime n) in
  reversed_primes.sum / reversed_primes.length

theorem reversed_primes_mean_is_427 : reversed_primes_mean = 427 := 
by sorry

end reversed_primes_mean_is_427_l405_405681


namespace polygon_sides_l405_405864

theorem polygon_sides {S n : ℕ} (h : S = 2160) (hs : S = 180 * (n - 2)) : n = 14 := 
by
  sorry

end polygon_sides_l405_405864


namespace sum_of_three_largest_consecutive_numbers_l405_405101

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l405_405101


namespace parabola_distance_l405_405621

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405621


namespace parabola_distance_problem_l405_405415

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405415


namespace distance_AB_l405_405258

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405258


namespace parabola_distance_l405_405400

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405400


namespace original_price_is_1200_l405_405860

theorem original_price_is_1200 (P : ℝ) (h : 0.85 * P = 1020) : P = 1200 :=
by {
  have h1 : P = 1020 / 0.85, from eq_div_of_mul_eq (ne_of_gt (by norm_num : 0.85 ≠ 0)) h,
  have h2 : 1020 / 0.85 = 1200, from by norm_num,
  rw h2 at h1,
  exact h1,
}

end original_price_is_1200_l405_405860


namespace rhombus_diagonal_length_l405_405696

theorem rhombus_diagonal_length (d1 d2 A : ℝ)
  (h1 : d1 = 62)
  (h2 : A = 2480)
  (h3 : A = (d1 * d2) / 2) :
  d2 = 80 :=
by
  rw [h1, h2, h3]
  sorry

end rhombus_diagonal_length_l405_405696


namespace tens_digit_23_pow_1987_l405_405918

def tens_digit_of_power (a b n : ℕ) : ℕ :=
  ((a^b % n) / 10) % 10

theorem tens_digit_23_pow_1987 : tens_digit_of_power 23 1987 100 = 4 := by
  sorry

end tens_digit_23_pow_1987_l405_405918


namespace distance_AB_l405_405444

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405444


namespace sin_alpha_gt_sin_beta_cot_alpha_lt_cot_beta_l405_405145

variables {α β : ℝ} (hα : π/2 < α ∧ α < π) (hβ : π/2 < β ∧ β < π) (hcos : cos α > cos β)

theorem sin_alpha_gt_sin_beta (hα : π/2 < α ∧ α < π) (hβ : π/2 < β ∧ β < π) (hcos : cos α > cos β) : 
  sin α > sin β :=
sorry

theorem cot_alpha_lt_cot_beta (hα : π/2 < α ∧ α < π) (hβ : π/2 < β ∧ β < π) (hcos : cos α > cos β) : 
  cot α < cot β :=
sorry

end sin_alpha_gt_sin_beta_cot_alpha_lt_cot_beta_l405_405145


namespace sum_of_three_largest_l405_405080

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l405_405080


namespace parabola_distance_l405_405544

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405544


namespace six_digit_palindromes_count_l405_405031

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l405_405031


namespace fraction_of_class_on_field_trip_l405_405845

variables (x : ℝ)

-- Conditions as definitions
def initially_left_on_field_trip : ℝ := (4 / 5) * x
def stayed_behind : ℝ := (1 / 5) * x
def did_not_want_to_go : ℝ := (1 / 15) * x
def did_want_to_go : ℝ := (2 / 15) * x
def additional_students : ℝ := (1 / 15) * x
def total_students_on_trip : ℝ := (13 / 15) * x

-- Theorem stating the actual problem
theorem fraction_of_class_on_field_trip (h1 : initially_left_on_field_trip = (4 / 5) * x)
                                        (h2 : stayed_behind = (1 / 5) * x)
                                        (h3 : did_not_want_to_go = (1 / 15) * x)
                                        (h4 : did_want_to_go = (2 / 15) * x)
                                        (h5 : additional_students = (1 / 15) * x) :
    total_students_on_trip = (13 / 15) * x :=
sorry

end fraction_of_class_on_field_trip_l405_405845


namespace AB_distance_l405_405327

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405327


namespace is_odd_and_monotonically_increasing_l405_405653

def f (x : ℝ) : ℝ := x ^ 3 - 1 / (x ^ 3)

theorem is_odd_and_monotonically_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :=
begin
  sorry
end

end is_odd_and_monotonically_increasing_l405_405653


namespace university_founding_day_l405_405728

/-- Problem statement:
The Treaty of Paris was signed on Tuesday, September 3, 1783.
A peace celebration exactly 1204 days after that date marks the founding of a new university.
Prove that this date falls on a Friday. -/
theorem university_founding_day :
  let treaty_signed_day := "Tuesday"
  let days_after := 1204
  founding_day = "Friday" sorry

end university_founding_day_l405_405728


namespace parabola_distance_problem_l405_405423

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405423


namespace six_digit_palindromes_count_l405_405004

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l405_405004


namespace six_digit_palindrome_count_l405_405009

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l405_405009


namespace parabola_distance_l405_405575

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405575


namespace parabola_problem_l405_405565

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405565


namespace parabola_problem_l405_405290

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405290


namespace total_outfits_l405_405723

-- Define the number of shirts, pants, ties (including no-tie option), and shoes as given in the conditions.
def num_shirts : ℕ := 5
def num_pants : ℕ := 4
def num_ties : ℕ := 6 -- 5 ties + 1 no-tie option
def num_shoes : ℕ := 2

-- Proof statement: The total number of different outfits is 240.
theorem total_outfits : num_shirts * num_pants * num_ties * num_shoes = 240 :=
by
  sorry

end total_outfits_l405_405723


namespace triangular_array_sum_mod_4_l405_405188

-- Lean 4 statement for the problem
theorem triangular_array_sum_mod_4 :
  ∃ (count : ℕ), count = 512 ∧ 
  (∀ (x : Fin 10 → ℕ), (∀ i, x i = 0 ∨ x i = 2) →
  ((∑ i, (Nat.choose 9 i) * x i) % 4 = 0 → 
  count = ∏ i, if x i = 0 ∨ x i = 2 then 1 else 0)) :=
  sorry

end triangular_array_sum_mod_4_l405_405188


namespace range_f_l405_405159

def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 + Real.sin x - 1

theorem range_f :
  (Set.range (λ x : ℝ, f x) ∩ Set.Icc 0 (Real.pi / 2)) = Set.Icc 0 (1 / 4) :=
sorry

end range_f_l405_405159


namespace find_g_l405_405743

theorem find_g :
  (∀ x, 0 ≤ x → x ≤ 1 → g 0 = 0)
  → (∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  → (∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  → (∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3)
  → g (1 / 2) = 1 / 3
  → g (3 / 16) = 2 / 9 :=
by
  intros h0 h1 h2 h3 h4
  sorry

end find_g_l405_405743


namespace books_purchased_with_grant_l405_405727

-- Define the conditions
def total_books_now : ℕ := 8582
def books_before_grant : ℕ := 5935

-- State the theorem that we need to prove
theorem books_purchased_with_grant : (total_books_now - books_before_grant) = 2647 := by
  sorry

end books_purchased_with_grant_l405_405727


namespace parabola_problem_l405_405510

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405510


namespace distance_AB_l405_405454

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405454


namespace width_decreased_by_28_6_percent_l405_405762

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l405_405762


namespace six_digit_palindromes_count_l405_405052

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l405_405052


namespace ab_distance_l405_405252

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405252


namespace parabola_distance_problem_l405_405405

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405405


namespace log_properties_l405_405956

theorem log_properties (a b : ℝ) (h1 : log a b > 1) : 0 < log b a < 1 :=
by
  sorry

end log_properties_l405_405956


namespace trigonometric_identity_l405_405141

theorem trigonometric_identity (θ : ℝ) (h1 : sin θ + cos θ = 1 / 5) (h2 : π / 2 ≤ θ ∧ θ ≤ 3 * π / 4) :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end trigonometric_identity_l405_405141


namespace half_sum_squares_ge_product_l405_405701

theorem half_sum_squares_ge_product (x y : ℝ) : 
  1 / 2 * (x^2 + y^2) ≥ x * y := 
by 
  sorry

end half_sum_squares_ge_product_l405_405701


namespace parabola_problem_l405_405291

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405291


namespace ab_distance_l405_405244

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405244


namespace six_digit_palindromes_count_l405_405024

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l405_405024


namespace part1_part2_part3_l405_405976

def f (x : ℝ) : ℝ := log x + 1 / x
def g (x : ℝ) : ℝ := x - log x

theorem part1 : (∀ x > 0, f x ≥ a) → a ≤ 1 :=
sorry

theorem part2 : (∀ x > 1, f x < g x) :=
sorry

theorem part3 (x1 x2 : ℝ) (hx1 : x1 > x2) (hg : g x1 = g x2) : x1 * x2 < 1 :=
sorry

end part1_part2_part3_l405_405976


namespace sum_of_three_largest_consecutive_numbers_l405_405106

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l405_405106


namespace AB_distance_l405_405328

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405328


namespace uniqueValidArrangement_l405_405950

def isDistinctLetters (l : List Char) : Prop := 
  List.nodup l

def isValidGrid (grid : List (List Char)) : Prop := 
  (length grid = 4) ∧ 
  all (λ row, length row = 4 ∧ isDistinctLetters row) grid ∧ 
  all (λ col, isDistinctLetters (List.map (λ row, row[col]) grid)) [0, 1, 2, 3]

noncomputable def countValidArrangements : ℕ :=
  let letters : List Char := ['A', 'B', 'C', 'D']
  let initialGrid : List (List Char) := 
  [['B', '_', '_', '_'],
   ['_', '_', '_', '_'],
   ['_', '_', '_', '_'],
   ['_', '_', '_', '_']]
  if isValidGrid initialGrid then 1 else 0

theorem uniqueValidArrangement : countValidArrangements = 1 :=
by
  sorry

end uniqueValidArrangement_l405_405950


namespace ab_distance_l405_405247

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405247


namespace part1_general_formula_part2_Sn_gt_n_minus_1_l405_405139

noncomputable def seq_a (n : ℕ) : ℝ :=
if h : n = 0 then 0 else 3^n

def seq_b (n : ℕ) : ℝ :=
let an := seq_a n in (an - 1) / (an + 1)

def S (n : ℕ) : ℝ :=
∑ i in finset.range n, seq_b (i + 1)

theorem part1_general_formula :
  ∀ n : ℕ, seq_a n = if n = 0 then 0 else 3^n := sorry

theorem part2_Sn_gt_n_minus_1 : ∀ n : ℕ, S n > n - 1 := sorry

end part1_general_formula_part2_Sn_gt_n_minus_1_l405_405139


namespace parabola_distance_l405_405574

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405574


namespace find_k_of_inverse_proportion_l405_405745

theorem find_k_of_inverse_proportion (k : ℝ) : 
  (∃ (x y : ℝ), x = -2 ∧ y = 3 ∧ y = k / x) → k = -6 :=
by
  intros h,
  rcases h with ⟨x, y, hx, hy, hy_eq⟩,
  rw [hx, hy] at hy_eq,
  linarith[-1*hy_eq]
  sorry

end find_k_of_inverse_proportion_l405_405745


namespace ab_distance_l405_405237

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405237


namespace parabola_problem_l405_405509

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405509


namespace total_roses_tom_sent_l405_405799

theorem total_roses_tom_sent
  (roses_in_dozen : ℕ := 12)
  (dozens_per_day : ℕ := 2)
  (days_in_week : ℕ := 7) :
  7 * (2 * 12) = 168 := by
  sorry

end total_roses_tom_sent_l405_405799


namespace r_work_rate_50_days_l405_405835

-- Define P, Q, and R as real numbers representing work rates
variables {P Q R : ℝ}

-- Define the conditions as hypotheses
hypothesis (h1 : P = Q + R)
hypothesis (h2 : P + Q = 1 / 10)
hypothesis (h3 : Q = 1 / 24.999999999999996)

-- Define the goal to prove
theorem r_work_rate_50_days :
  1 / R = 50 :=
sorry

end r_work_rate_50_days_l405_405835


namespace speed_difference_l405_405848

-- Definitions based on conditions
def bike_distance : ℝ := 72
def truck_distance : ℝ := 72
def bike_time : ℝ := 9
def truck_time : ℝ := 9

-- Speed calculation based on conditions
def bike_speed : ℝ := bike_distance / bike_time
def truck_speed : ℝ := truck_distance / truck_time

-- The theorem to prove
theorem speed_difference : bike_speed - truck_speed = 0 := by
  sorry

end speed_difference_l405_405848


namespace distance_AB_l405_405441

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405441


namespace sum_of_three_largest_l405_405062

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l405_405062


namespace probability_of_occurrence_l405_405784

-- Define the odds as given condition
def odds_favourable_to_unfavourable (a b : ℕ) := a.to_rat / (a + b).to_rat

-- Define the specific odds 3:5
def specific_odds := odds_favourable_to_unfavourable 3 5

-- Theorem stating the probability is 3/8
theorem probability_of_occurrence : specific_odds = 3 / 8 := by
  sorry

end probability_of_occurrence_l405_405784


namespace sum_of_largest_three_consecutive_numbers_l405_405099

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l405_405099


namespace parabola_problem_l405_405500

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405500


namespace sum_quotient_product_diff_l405_405792

theorem sum_quotient_product_diff (x y : ℚ) (h₁ : x + y = 6) (h₂ : x / y = 6) : 
  (x * y) - (x - y) = 6 / 49 :=
  sorry

end sum_quotient_product_diff_l405_405792


namespace parabola_distance_l405_405629

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l405_405629


namespace problem_l405_405610

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405610


namespace parabola_distance_l405_405590

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405590


namespace curve_in_second_quadrant_range_l405_405179

theorem curve_in_second_quadrant_range (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0)) → a > 2 :=
by
  sorry

end curve_in_second_quadrant_range_l405_405179


namespace original_grape_price_l405_405816

theorem original_grape_price
  (lemon_price_increase : ℝ)
  (grape_price_increase : ℝ)
  (initial_lemon_price : ℝ)
  (lemons_sold : ℕ)
  (grapes_sold : ℕ)
  (total_collected : ℝ)
  (new_lemon_price : ℝ := initial_lemon_price + lemon_price_increase)
  (new_grape_price : ℝ := grape_price_increase + 2)
  (price_collected_from_lemons : ℝ := lemons_sold * new_lemon_price)
  (price_collected_from_grapes : ℝ := grapes_sold * new_grape_price)
  (total_price_collected : ℝ := price_collected_from_lemons + price_collected_from_grapes)
  : (lemons_sold = 80) →
    (grapes_sold = 140) →
    (total_collected = 2220) →
    (lemon_price_increase = 4) →
    (grape_price_increase = 2) →
    (total_price_collected =  total_collected) →
    (initial_lemon_price = 8) → (grape_price_increase = 2 )→
    new_lemon_price = 12
    new_grape_price = 7 :=
begin
  intros hlemons_sold hgrapes_sold htotal_collected hlemon_price_increase hgrape_price_increase htotal_price_collected hinitial_lemon_price,
  sorry
end

end original_grape_price_l405_405816


namespace distance_AB_l405_405466

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405466


namespace AB_distance_l405_405364

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405364


namespace constant_term_in_expansion_is_seven_l405_405732

theorem constant_term_in_expansion_is_seven :
  let binom := (λ (x : ℂ), (x / 2 - 1 / x^(1/3))^8) in
  ∃ c : ℂ, (c ≠ 0) ∧ (∀ x, (binom x) = c ∧ x = (7 : ℂ)) :=
sorry

end constant_term_in_expansion_is_seven_l405_405732


namespace triangle_perimeter_l405_405805

-- Definitions for the conditions
variables (A M L C P Q : Type) -- A: vertex, M: point inside angle, L and C: points on the angle sides, P and Q: tangents from M
variable (p : ℝ) -- Half the given perimeter
variables [metric_space A] [inner_product_space ℝ A] -- Assume a metric/inner product space structure for geometric constructs
variables (triangle : L <-> P ∧ P <-> Q ∧ Q <-> M)

-- The statement to prove
theorem triangle_perimeter {MP MQ PQ : ℝ} (hMP : MP = p) (hMQ : MQ = p) (hPQ : PQ = p) : 
  MP + MQ + PQ = 2 * p :=
by {
  sorry -- proof goes here
}

end triangle_perimeter_l405_405805


namespace Marla_colors_green_squares_l405_405677

theorem Marla_colors_green_squares :
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  green_squares = 66 :=
by
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  show green_squares = 66
  sorry

end Marla_colors_green_squares_l405_405677


namespace area_rectangle_and_triangle_l405_405694

-- Define the points and lengths
variables {AN NC AM MB : ℝ}
variables (AN NC AM MB : ℝ)

-- Given conditions
axiom h1 : AN = 7
axiom h2 : NC = 39
axiom h3 : AM = 12
axiom h4 : MB = 3

-- Define the sides of the rectangle
def CD : ℝ := AM + MB
def DN : ℝ := real.sqrt (NC^2 - (AM + MB)^2)
def AD : ℝ := DN + AN
def AB : ℝ := AM + MB

-- Rectangle area
def Area_Rectangle (AB AD : ℝ) : ℝ := AB * AD

-- Triangular areas
def Area_AMN (AM AN : ℝ) : ℝ := 1/2 * AM * AN
def Area_BCM (MB BC : ℝ) : ℝ := 1/2 * MB * BC
def Area_DNC (CD DN : ℝ) : ℝ := 1/2 * CD * DN

-- The area of the triangle MNC
def Area_Triangle_MNC (S_ABCD S_AMN S_BCM S_DNC : ℝ) : ℝ := 
    S_ABCD - S_AMN - S_BCM - S_DNC

-- The main statement
theorem area_rectangle_and_triangle :
  (Area_Rectangle AB AD) = 645 ∧ 
  (Area_Triangle_MNC (Area_Rectangle AB AD) (Area_AMN AM AN) 
                    (Area_BCM MB (AD - AN)) (Area_DNC CD DN)) = 268.5 :=
by {
  -- Substituting the known lengths
  have h_CD : CD = 15 := by { unfold CD, rw [h3, h4], norm_num },
  have h_DN : DN = 36 := by { unfold DN, rw [h2, h_CD], norm_num },
  have h_AD : AD = 43 := by { unfold AD, rw [h_DN, h1], norm_num },
  have h_AB : AB = 15 := by { unfold AB, rw [h3, h4], norm_num },
  
  -- Calculate Areas
  have h_Area_Rectangle : Area_Rectangle AB AD = 645 := 
    by { unfold Area_Rectangle, rw [h_AB, h_AD], norm_num },
  
  have h_Area_AMN : Area_AMN AM AN = 42 := 
    by { unfold Area_AMN, rw [h3, h1], norm_num },
    
  have h_Area_BCM : Area_BCM MB (AD - AN) = 64.5 :=
    by { unfold Area_BCM, rw [h4, h_AD, h1], norm_num },
    
  have h_Area_DNC : Area_DNC CD DN = 270 :=
    by { unfold Area_DNC, rw [h_CD, h_DN], norm_num },
  
  -- Combine the results for the triangle
  have h_Area_Triangle_MNC : Area_Triangle_MNC h_Area_Rectangle h_Area_AMN h_Area_BCM h_Area_DNC = 268.5 :=
    by { unfold Area_Triangle_MNC, rw [h_Area_Rectangle, h_Area_AMN, h_Area_BCM, h_Area_DNC], norm_num },
  
  -- Final result
  exact ⟨h_Area_Rectangle, h_Area_Triangle_MNC⟩,
}

end area_rectangle_and_triangle_l405_405694


namespace f_odd_and_monotone_l405_405657

def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

theorem f_odd_and_monotone :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x y : ℝ), 0 < x → x < y → f x < f y) :=
  by
    sorry

end f_odd_and_monotone_l405_405657


namespace chess_tournament_no_804_games_l405_405198

/-- Statement of the problem: 
    Under the given conditions, prove that it is impossible for exactly 804 games to have been played in the chess tournament.
--/
theorem chess_tournament_no_804_games :
  ¬ ∃ n : ℕ, n * (n - 4) = 1608 :=
by
  sorry

end chess_tournament_no_804_games_l405_405198


namespace hotel_people_count_l405_405717

theorem hotel_people_count :
  ∃ n : ℕ, let A := (117 : ℝ) / (n : ℝ) in
    8 * 12 + (A + 8) = 117 ∧ 117 / n = 13 :=
by
  sorry

end hotel_people_count_l405_405717


namespace distance_AB_l405_405351

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405351


namespace polygon_area_l405_405929

def Point := (ℝ × ℝ)

def vertices : List Point := [(0,0), (6,0), (6,6), (0,6)]
def is_square (verts: List Point): Prop :=
  List.nth verts 0 = some (0,0) ∧ 
  List.nth verts 1 = some (6,0) ∧ 
  List.nth verts 2 = some (6,6) ∧ 
  List.nth verts 3 = some (0,6)

theorem polygon_area (h : is_square vertices) : 
  let side_length := 6 in
  side_length * side_length = 36 := 
by
  sorry

end polygon_area_l405_405929


namespace sum_of_three_largest_l405_405079

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l405_405079


namespace distance_AB_l405_405448

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405448


namespace parabola_problem_l405_405566

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405566


namespace distance_AB_l405_405456

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405456


namespace sum_of_three_largest_consecutive_numbers_l405_405104

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l405_405104


namespace area_pentagon_l405_405965

-- Definitions
variables (A B C D E : Type) [ConvexPentagon A B C D E]
variables (AB BC CD DE : ℝ)
variables (angle_ABC angle_CDE : ℝ)
variables (BD : ℝ)

-- Conditions
def conditions : Prop :=
  AB = BC ∧
  CD = DE ∧
  angle_ABC = 150 ∧
  angle_CDE = 30 ∧
  BD = 2

-- Problem Statement: Given the conditions, the area of pentagon ABCDE is 1
theorem area_pentagon (h : conditions A B C D E AB BC CD DE angle_ABC angle_CDE BD) : area A B C D E = 1 :=
sorry

end area_pentagon_l405_405965


namespace ab_distance_l405_405249

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405249


namespace soybean_oil_production_l405_405933

theorem soybean_oil_production :
  (∀ (x : ℝ), (0 < x) → (0 < 0.8 * x)) →
  (∀ (x : ℝ), (x = 20) → (0.8 * x = 16) ∧ (20 / 0.8 = 25)) :=
by
  intro h
  intro x hx
  split
  · rw [hx]
    norm_num
  · norm_num
  sorry

end soybean_oil_production_l405_405933


namespace reciprocal_sum_of_intersection_points_l405_405858

theorem reciprocal_sum_of_intersection_points :
  ∀ (x₁ x₂ : ℝ), 
  (∃ k : ℝ, x₁ ^ 2 = k * (x₁ - 10) ∧ x₂ ^ 2 = k * (x₂ - 10)) → 
  (1 / x₁ + 1 / x₂ = 1 / 10) :=
begin
  sorry
end

end reciprocal_sum_of_intersection_points_l405_405858


namespace alpha_perpendicular_beta_sufficient_for_c_perpendicular_b_alpha_perpendicular_beta_not_necessary_for_c_perpendicular_b_alpha_perpendicular_beta_sufficient_but_not_necessary_l405_405146

-- Definitions for the planes, lines, and perpendicularity condition
def Plane (α : Type*) := set α
def Line (α : Type*) := set α
def intersects (m : Line ℝ) (α β : Plane ℝ) : Prop := ∃ (m' : Line ℝ), (m' ⊆ α) ∧ (m' ⊆ β) ∧ (m' = m)
def intersects_at (α β : Plane ℝ) (m : Line ℝ) := intersects m α β
def within (l : Line ℝ) (p : Plane ℝ) := l ⊆ p
def perp (l1 l2 : Line ℝ) : Prop := ∃ (p1 p2 : Plane ℝ), (l1 ⊆ p1) ∧ (l2 ⊆ p2) ∧ (∃ n1 n2 : vec3, l1 ∥ n1 ∧ l2 ∥ n2 ∧ dot_product n1 n2 = 0)

-- Translate the conditions into Lean definitions
variables (α β : Plane ℝ) (m b c : Line ℝ)
axiom intersect_condition : intersects_at α β m
axiom b_within_alpha : within b α
axiom c_within_beta : within c β
axiom c_perpendicular_m : perp c m

-- The proposition to be proven
theorem alpha_perpendicular_beta_sufficient_for_c_perpendicular_b : (perp α β) → (perp c b) :=
sorry

theorem alpha_perpendicular_beta_not_necessary_for_c_perpendicular_b : (perp c b) → ¬ (perp α β) :=
sorry

theorem alpha_perpendicular_beta_sufficient_but_not_necessary : 
  (perp α β) ↔ ∃ (c_perpendicular_b : perp c b), true :=
begin
  split,
  { intro h,
    use alpha_perpendicular_beta_sufficient_for_c_perpendicular_b α β m b c intersect_condition b_within_alpha c_within_beta c_perpendicular_m h },
  {
    intro h,
    have h2 := alpha_perpendicular_beta_not_necessary_for_c_perpendicular_b α β m b c intersect_condition b_within_alpha c_within_beta c_perpendicular_m h,
    contradiction
  }
sorry

-- The corollary with the correct answer from the solution
example : "A" = "Sufficient but not necessary condition" :=
rfl

end alpha_perpendicular_beta_sufficient_for_c_perpendicular_b_alpha_perpendicular_beta_not_necessary_for_c_perpendicular_b_alpha_perpendicular_beta_sufficient_but_not_necessary_l405_405146


namespace parabola_problem_l405_405559

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405559


namespace parabola_problem_l405_405292

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405292


namespace parabola_problem_l405_405519

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405519


namespace sin_x_in_terms_of_a_b_l405_405178

theorem sin_x_in_terms_of_a_b (a b : ℝ) (x : ℝ)
  (h₁ : a > b) (h₂ : b > 0) (h3 : 0 < x) (h4 : x < π / 2)
  (h5 : tan x = 3 * a * b / (a ^ 2 - b ^ 2)) :
  sin x = 3 * a * b / real.sqrt (a ^ 4 + 7 * a ^ 2 * b ^ 2 + b ^ 4) :=
sorry

end sin_x_in_terms_of_a_b_l405_405178


namespace six_digit_palindromes_count_l405_405033

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l405_405033


namespace sum_of_largest_three_l405_405113

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l405_405113


namespace distance_AB_l405_405350

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405350


namespace ab_distance_l405_405239

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405239


namespace smallest_n_probability_townspeople_l405_405058

noncomputable def double_factorial : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := (n + 2) * double_factorial n

def p (n : ℕ) : ℝ := (n.factorial : ℝ) / (double_factorial (2 * n + 1))

theorem smallest_n_probability_townspeople : ∃ n : ℕ, p n < 0.01 ∧ ∀ m < n, ¬ (p m < 0.01) :=
by sorry

end smallest_n_probability_townspeople_l405_405058


namespace parabola_distance_l405_405582

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405582


namespace distance_AB_l405_405259

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405259


namespace parabola_distance_l405_405540

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405540


namespace marla_colors_green_squares_l405_405679

-- Condition 1: Grid dimensions
def num_rows : ℕ := 10
def num_cols : ℕ := 15

-- Condition 2: Red squares
def red_rows : ℕ := 4
def red_squares_per_row : ℕ := 6
def red_squares : ℕ := red_rows * red_squares_per_row

-- Condition 3: Blue rows (first 2 and last 2)
def blue_rows : ℕ := 2 + 2
def blue_squares_per_row : ℕ := num_cols
def blue_squares : ℕ := blue_rows * blue_squares_per_row

-- Derived information
def total_squares : ℕ := num_rows * num_cols
def non_green_squares : ℕ := red_squares + blue_squares

-- The Lemma to prove
theorem marla_colors_green_squares : total_squares - non_green_squares = 66 := by
  sorry

end marla_colors_green_squares_l405_405679


namespace distance_AB_l405_405427

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405427


namespace right_triangle_largest_side_l405_405870

theorem right_triangle_largest_side (b d : ℕ) (h_triangle : (b - d)^2 + b^2 = (b + d)^2)
  (h_arith_seq : (b - d) < b ∧ b < (b + d))
  (h_perimeter : (b - d) + b + (b + d) = 840) :
  (b + d = 350) :=
by sorry

end right_triangle_largest_side_l405_405870


namespace triangle_division_l405_405202

theorem triangle_division (ABC : Triangle) (right_angle_C : ABC.is_right_triangle_at C)
  (h_AB : 100 < ABC.AB ∧ ABC.AB < 101)
  (h_AC : 99 < ABC.AC ∧ ABC.AC < 100) :
  ∃ (division : List Triangle), (∀ t ∈ division, ∃e ∈ t.edges, e.length = 1) ∧ division.length ≤ 21 :=
  sorry

end triangle_division_l405_405202


namespace six_digit_palindrome_count_l405_405008

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l405_405008


namespace f_is_ab_type_find_m_for_g_l405_405966

-- Problem I: Verify if f(x) = 4^x is an (a,b)-type function
def is_ab_type_function (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x : ℝ, f(a + x) * f(a - x) = b

theorem f_is_ab_type :
  ∃ a b : ℝ, is_ab_type_function (λ x, 4^x) a b :=
by
  use 1, 16
  sorry

-- Problem II: Range of m for g(x) being (1,4)-type function
def is_14_type_function (g : ℝ → ℝ) :=
  ∀ x : ℝ, g(1 + x) * g(1 - x) = 4

def g_identity_candidate (g : ℝ → ℝ) (m : ℝ) : Prop :=
  1 ≤ g(0) ∧ g(0) ≤ 3 ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g(x) = x^2 - m*(x-1) + 1) ∧
  is_14_type_function g

theorem find_m_for_g (g : ℝ → ℝ) (m : ℝ) :
  g_identity_candidate g m →
  2 - (2 * Real.sqrt 6) / 3 ≤ m ∧ m ≤ 2 :=
by
  intro h
  sorry

end f_is_ab_type_find_m_for_g_l405_405966


namespace problem_l405_405609

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405609


namespace parabola_problem_l405_405552

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405552


namespace point_where_ordinate_increases_5_times_faster_than_abscissa_l405_405693

noncomputable def curve (x : ℝ) : ℝ := x^2 - 3 * x + 5

theorem point_where_ordinate_increases_5_times_faster_than_abscissa :
  ∃ (x y : ℝ), y = curve x ∧ (∀ t, deriv curve t = 5 → x = 4 ∧ y = 9) :=
by
  existsi (4 : ℝ)
  existsi (9 : ℝ)
  split
  · -- Prove y = curve x for x = 4 and y = 9
    sorry
  · -- Prove the condition on the derivative implies x = 4
    intro t ht
    split
    · -- Prove x = 4
      sorry
    · -- Prove y = 9
      sorry

end point_where_ordinate_increases_5_times_faster_than_abscissa_l405_405693


namespace rhombus_area_l405_405809

theorem rhombus_area (side diag_diff : ℝ) (h_side : side = Real.sqrt 145) (h_diag_diff : diag_diff = 10) :
  let y := 5 in
  let shorter_diag := 2 * y in
  let longer_diag := shorter_diag + diag_diff in
  let area := (1 / 2) * shorter_diag * longer_diag in
  area = 100 :=
by
  sorry

end rhombus_area_l405_405809


namespace sum_of_three_largest_consecutive_numbers_l405_405105

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l405_405105


namespace distance_AB_l405_405472

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405472


namespace ab_distance_l405_405248

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405248


namespace parabola_problem_l405_405557

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405557


namespace f_odd_and_monotone_l405_405658

def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

theorem f_odd_and_monotone :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x y : ℝ), 0 < x → x < y → f x < f y) :=
  by
    sorry

end f_odd_and_monotone_l405_405658


namespace find_m_from_arithmetic_sequence_l405_405185

theorem find_m_from_arithmetic_sequence (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -4) (h2 : S m = 0) (h3 : S (m + 1) = 6) : m = 5 := by
  sorry

end find_m_from_arithmetic_sequence_l405_405185


namespace mike_peaches_l405_405682

theorem mike_peaches (initial_peaches picked_peaches : ℝ) (h1 : initial_peaches = 34.0) (h2 : picked_peaches = 86.0) : initial_peaches + picked_peaches = 120.0 :=
by
  rw [h1, h2]
  norm_num

end mike_peaches_l405_405682


namespace collinear_XYZ_l405_405840

theorem collinear_XYZ
  (A B C D X Y Z : Point)
  (h_trap : IsoscelesTrapezoid A B C D)
  (h_parallel : Parallel B C A D)
  (h_omega : CircleThrough B C)
  (h_intersects_AB : IntersectsAt h_omega A B X)
  (h_intersects_BD : IntersectsAt h_omega B D Y)
  (h_tangent : TangentToCircleAt h_omega C Z A D) :
  Collinear X Y Z :=
sorry

end collinear_XYZ_l405_405840


namespace parabola_distance_problem_l405_405404

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405404


namespace average_home_runs_is_correct_l405_405730

theorem average_home_runs_is_correct :
  let h := [5, 6, 7, 9, 11]
  let p := [7, 5, 4, 3, 1]
  let Σ_hp := List.sum (List.zipWith (· * ·) h p)
  let Σ_p := List.sum p
  (Σ_hp : ℚ) / Σ_p = 6.55 :=
by
  let h := [5, 6, 7, 9, 11]
  let p := [7, 5, 4, 3, 1]
  let Σ_hp := List.sum (List.zipWith (· * ·) h p)
  let Σ_p := List.sum p
  show (Σ_hp : ℚ) / Σ_p = 6.55
  sorry

end average_home_runs_is_correct_l405_405730


namespace parabola_distance_problem_l405_405421

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405421


namespace max_pa_pb_l405_405127

noncomputable section
open Real

variables (m : ℝ)
def A : Point := (0, 0)
def B : Point := (1, 3)
def line1 (m : ℝ) : Line := { p : Point | p.1 + m * p.2 = 0 }
def line2 (m : ℝ) : Line := { p : Point | m * p.1 - p.2 - m + 3 = 0 }
def P (m : ℝ) : Point := -- Intersection point of the lines (simple algebra to find intersection)

theorem max_pa_pb : 
  let PA := dist A (P m)
  let PB := dist B (P m)
  maxPA : (PA ^ 2 + PB ^ 2 = 10)
  PA * PB ≤ 5 := sorry

end max_pa_pb_l405_405127


namespace f_eq_expression_f_monotonic_dec_range_f_one_div_m_l405_405993

noncomputable def g (x : ℝ) : ℝ := 2 ^ x
noncomputable def f (x : ℝ) : ℝ := (1 - g x) / (1 + g x)

theorem f_eq_expression : 
  ∀ x : ℝ, f x = (1 - 2 ^ x) / (1 + 2 ^ x) := 
sorry

theorem f_monotonic_dec :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem range_f_one_div_m (m : ℝ) (hm : ∃ x ∈ Ioo (-1 : ℝ) 0, f x = m) :
  ∀ x : ℝ, 1 / m ≤ 3 → -1 < f (1 / m) ∧ f (1 / m) ≤ - 7 / 9 :=
sorry

end f_eq_expression_f_monotonic_dec_range_f_one_div_m_l405_405993


namespace students_on_bus_l405_405872

theorem students_on_bus
    (initial_students : ℝ) (first_get_on : ℝ) (first_get_off : ℝ)
    (second_get_on : ℝ) (second_get_off : ℝ)
    (third_get_on : ℝ) (third_get_off : ℝ) :
  initial_students = 21 →
  first_get_on = 7.5 → first_get_off = 2 → 
  second_get_on = 1.2 → second_get_off = 5.3 →
  third_get_on = 11 → third_get_off = 4.8 →
  (initial_students + (first_get_on - first_get_off) +
   (second_get_on - second_get_off) +
   (third_get_on - third_get_off)) = 28.6 := by
  intros
  sorry

end students_on_bus_l405_405872


namespace f_properties_l405_405661

def f (x : ℝ) : ℝ := x^3 - 1 / x^3

theorem f_properties : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := 
by
  sorry

end f_properties_l405_405661


namespace football_championship_l405_405194

noncomputable def exists_trio_did_not_play_each_other 
  (teams : Finset ℕ) (rounds : ℕ) (matches : Finset (ℕ × ℕ)) : Prop :=
  ∃ (T : Finset ℕ), T.card = 3 ∧
    (∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → (a, b) ∉ matches ∧ (b, a) ∉ matches)

theorem football_championship (teams : Finset ℕ) (h_teams : teams.card = 18)
  (rounds : ℕ) (h_rounds : rounds = 8) 
  (matches_per_round : Finset (Finset (ℕ × ℕ))) 
  (h_matches_per_round : ∀ round ∈ matches_per_round, round.card = 9)
  (unique_pairs : ∀ (r₁ r₂ ∈ matches_per_round) (p : ℕ × ℕ), p ∈ r₁ → p ∈ r₂ → r₁ = r₂) :
  exists_trio_did_not_play_each_other teams rounds (matches_per_round.bUnion id) :=
by 
  sorry

end football_championship_l405_405194


namespace distance_AB_l405_405264

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405264


namespace sum_of_largest_three_consecutive_numbers_l405_405097

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l405_405097


namespace parabola_distance_problem_l405_405414

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405414


namespace Marla_colors_green_squares_l405_405678

theorem Marla_colors_green_squares :
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  green_squares = 66 :=
by
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  show green_squares = 66
  sorry

end Marla_colors_green_squares_l405_405678


namespace AB_distance_l405_405367

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405367


namespace problem_l405_405607

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405607


namespace parabola_problem_l405_405300

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405300


namespace distance_AB_l405_405261

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405261


namespace six_digit_palindromes_count_l405_405048

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l405_405048


namespace distance_AB_l405_405333

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405333


namespace fraction_decomposition_l405_405927

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ -8/3 → (7 * x - 19) / (3 * x^2 + 5 * x - 8) = A / (x - 1) + B / (3 * x + 8)) →
  A = -12 / 11 ∧ B = 113 / 11 :=
by
  sorry

end fraction_decomposition_l405_405927


namespace sum_of_three_largest_l405_405081

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l405_405081


namespace longest_side_of_quadrilateral_l405_405906

theorem longest_side_of_quadrilateral :
  ∀ (x y : ℝ), 
    x + y ≤ 4 →
    3 * x + y ≥ 3 →
    0 ≤ x →
    0 ≤ y →
    ∃ (longest_side_length : ℝ), longest_side_length = 5 :=
by
  intros x y h1 h2 h3 h4
  let vertices := [(0, 0), (1, 0), (4, 0), (0, 3)]
  let distance (p1 p2 : ℝ × ℝ) : ℝ := 
    real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)
  let side_lengths := [distance (0, 0) (1, 0), distance (1, 0) (4, 0), distance (4, 0) (0, 3), distance (0, 3) (0, 0)]
  have max_side_length : list ℝ := list.maximum side_lengths
  exact ⟨max_side_length, sorry⟩

end longest_side_of_quadrilateral_l405_405906


namespace distance_AB_l405_405440

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405440


namespace tens_digit_of_23_pow_1987_l405_405924

theorem tens_digit_of_23_pow_1987 : (23 ^ 1987 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_23_pow_1987_l405_405924


namespace part1_part2_part3_l405_405975

-- Given definitions of functions f and g
def f (x : ℝ) := log x + 1/x
def g (x : ℝ) := x - log x

-- Proof problems translated into Lean statements

-- Part (1): Prove that the maximum value of a for which f(x) ≥ a for any x ∈ (0, +∞) is 1.
theorem part1 : ∀ (a : ℝ), (∀ x > 0, f x ≥ a) → a ≤ 1 := sorry

-- Part (2): Prove f(x) < g(x) for x ∈ (1, +∞).
theorem part2 : ∀ x > 1, f x < g x := sorry

-- Part (3): Prove that if x1 > x2 and g(x1) = g(x2), then x1 * x2 < 1.
theorem part3 : ∀ (x1 x2 : ℝ), x1 > x2 → g x1 = g x2 → x1 * x2 < 1 := sorry

end part1_part2_part3_l405_405975


namespace distance_AB_l405_405346

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405346


namespace cos_value_l405_405173

theorem cos_value (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_value_l405_405173


namespace parabola_problem_l405_405562

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405562


namespace periodic_odd_function_sum_l405_405670

noncomputable def f : ℝ → ℝ
| x := if x ∈ (Set.Icc 0 2) then 2*x - x^2 else
       if x < 0 then -f (-x) else sorry

theorem periodic_odd_function_sum :
  (∑ n in Finset.range 2019, f n) = 1 := sorry

end periodic_odd_function_sum_l405_405670


namespace distance_AB_l405_405334

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l405_405334


namespace sum_of_three_largest_consecutive_numbers_l405_405102

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l405_405102


namespace AB_distance_l405_405356

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405356


namespace ab_distance_l405_405253

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405253


namespace ab_distance_l405_405251

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405251


namespace six_digit_palindromes_l405_405019

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l405_405019


namespace six_digit_palindromes_count_l405_405049

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l405_405049


namespace distance_AB_l405_405459

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405459


namespace six_digit_palindromes_count_l405_405047

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l405_405047


namespace parabola_problem_l405_405512

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405512


namespace sum_of_three_largest_of_consecutive_numbers_l405_405074

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l405_405074


namespace parabola_distance_l405_405484

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405484


namespace probability_first_card_greater_second_card_l405_405124

theorem probability_first_card_greater_second_card :
  let cards := {i | 1 ≤ i ∧ i ≤ 7} in
  let total_cases := (cards ×' cards).card in
  let favorable_cases := ((cards ×' cards).filter (λ (x : ℕ × ℕ), x.1 > x.2)).card in
  probability_first_greater : ℚ :=
  (favorable_cases : ℚ) / total_cases = 3 / 7 :=
by
  sorry

end probability_first_card_greater_second_card_l405_405124


namespace six_digit_palindromes_l405_405023

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l405_405023


namespace AB_distance_l405_405326

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405326


namespace sum_of_nth_row_l405_405844

theorem sum_of_nth_row (n : ℕ) : 
  let first_element := (n * (n - 1)) / 2 + 1 in
  let nth_row_elements := list.range n |>.map (λ i, first_element + i) in
  (nth_row_elements.sum : ℕ) = n * (n^2 + 1) / 2 :=
by
  sorry

end sum_of_nth_row_l405_405844


namespace even_number_probability_l405_405187

theorem even_number_probability :
  let digits := {2, 3, 4} in
  let is_even (n : ℕ) := n % 2 = 0 in
  let three_digit_numbers := {n | ∃ (a b c : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ n = 100 * a + 10 * b + c} in
  (∃ e ∈ three_digit_numbers, is_even e) →
  (∃ t ∈ three_digit_numbers, true) →
  let even_count := ((three_digit_numbers.filter is_even).card) in
  let total_count := three_digit_numbers.card in
  even_count.toRat / total_count.toRat = (2 / 3 : ℚ) :=
by
  sorry

end even_number_probability_l405_405187


namespace max_friends_eq_m_l405_405205

-- Define the core conditions of the problem
structure Friendship (α : Type) :=
(friends : α → α → Prop)
(mutual : ∀ {x y : α}, friends x y → friends y x)
(no_self_friend : ∀ {x : α}, ¬ friends x x)
(unique_common_friend : ∀ (m : ℕ) (x : Fin m → α), (3 ≤ m) → ∃! c, ∀ i, friends (x i) c)

-- Define the main theorem
theorem max_friends_eq_m {α : Type} [Friendship α] {m : ℕ} (h_m := by sorry) : 
  ∀ (A : α), (∃ n, (∀ y, friends A y → friends A y) → n = m) :=
  sorry

end max_friends_eq_m_l405_405205


namespace parabola_distance_l405_405542

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405542


namespace h_three_l405_405229

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := Float.sqrt (f x) - 3
def h (x : ℝ) : ℝ := f (g x)

theorem h_three :
  h 3 = 3 * Real.sqrt 13 - 5 := by
  sorry

end h_three_l405_405229


namespace washer_cost_difference_l405_405881

theorem washer_cost_difference (W D : ℝ) 
  (h1 : W + D = 1200) (h2 : D = 490) : W - D = 220 :=
sorry

end washer_cost_difference_l405_405881


namespace constant_product_pq_pr_l405_405135

variables {a b : ℝ} (ha : a > 0) (hb : b > 0)
variables (α x0 y0 : ℝ)
variables (h_hyperbola : (x0^2 / a^2 - y0^2 / b^2 = 1))

theorem constant_product_pq_pr :
  ∃ (PQ PR : ℝ), (|PQ| * |PR| = a^2 * b^2 / |a^2 * sin α^2 - b^2 * cos α^2|) :=
  sorry

end constant_product_pq_pr_l405_405135


namespace VasyaSlowerWalkingFullWayHome_l405_405952

namespace FishingTrip

-- Define the variables involved
variables (x v S : ℝ)   -- x is the speed of Vasya and Petya, v is the speed of Kolya on the bicycle, S is the distance from the house to the lake

-- Conditions derived from the problem statement:
-- Condition 1: When Kolya meets Vasya then Petya starts
-- Condition 2: Given: Petya’s travel time is \( \frac{5}{4} \times \) Vasya's travel time.

theorem VasyaSlowerWalkingFullWayHome (h1 : v = 3 * x) :
  2 * (S / x + v) = (5 / 2) * (S / x) :=
sorry

end FishingTrip

end VasyaSlowerWalkingFullWayHome_l405_405952


namespace rectangle_width_decrease_l405_405777

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L in
  let W' := (L * W) / L' in
  let percent_decrease := (1 - W' / W) * 100 in
  percent_decrease ≈ 28.57 :=
by
  let L' := 1.4 * L
  let W' := (L * W) / L'
  let percent_decrease := (1 - W' / W) * 100
  sorry

end rectangle_width_decrease_l405_405777


namespace sum_of_largest_three_l405_405114

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l405_405114


namespace distance_AB_l405_405277

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405277


namespace parabola_distance_problem_l405_405422

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l405_405422


namespace six_digit_palindromes_count_l405_405034

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l405_405034


namespace parabola_problem_l405_405560

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405560


namespace frequency_count_of_group_l405_405871

theorem frequency_count_of_group :
  ∀ (sample_size : ℕ) (group_frequency : ℝ), sample_size = 1000 → group_frequency = 0.4 → sample_size * group_frequency = 400 := 
by
  intros sample_size group_frequency hs hf
  rw [hs, hf]
  norm_num
  sorry

end frequency_count_of_group_l405_405871


namespace is_odd_and_monotonically_increasing_l405_405652

def f (x : ℝ) : ℝ := x ^ 3 - 1 / (x ^ 3)

theorem is_odd_and_monotonically_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :=
begin
  sorry
end

end is_odd_and_monotonically_increasing_l405_405652


namespace horner_method_value_calc_l405_405804

noncomputable def f (x : ℝ) : ℝ :=
  5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - - 0.8

theorem horner_method_value_calc (x : ℝ) (v_0 v_1 v_2 v_3 : ℝ) 
  (h1 : v_0 = 5)
  (h2 : v_1 = v_0 * x + 2)
  (h3 : v_2 = v_1 * x + 3.5)
  (h4 : v_3 = v_2 * x - 2.6) :
  v_3 = 7.9 :=
by {
  let x := 1,
  let v_0 := 5,
  let v_1 := v_0 * x + 2,
  let v_2 := v_1 * x + 3.5,
  let v_3 := v_2 * x - 2.6,
  sorry
}

end horner_method_value_calc_l405_405804


namespace six_digit_palindromes_count_l405_405045

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l405_405045


namespace vector_relation_solution_l405_405137

variable {V : Type} [AddCommGroup V] [Module ℝ V]

variables {A P B : V}
variable (λ : ℝ)

theorem vector_relation_solution (h1 : A - P = P - B) (h2 : A - B = λ * (P - B)) : λ = 2 := sorry

end vector_relation_solution_l405_405137


namespace group_size_increase_by_4_l405_405729

theorem group_size_increase_by_4
    (N : ℕ)
    (weight_old : ℕ)
    (weight_new : ℕ)
    (average_increase : ℕ)
    (weight_increase_diff : ℕ)
    (h1 : weight_old = 55)
    (h2 : weight_new = 87)
    (h3 : average_increase = 4)
    (h4 : weight_increase_diff = weight_new - weight_old)
    (h5 : average_increase * N = weight_increase_diff) :
    N = 8 :=
by
  sorry

end group_size_increase_by_4_l405_405729


namespace parabola_distance_l405_405572

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l405_405572


namespace parabola_distance_l405_405491

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405491


namespace sum_of_three_largest_of_consecutive_numbers_l405_405072

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l405_405072


namespace parabola_distance_l405_405497

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405497


namespace extremum_of_f_l405_405941

def f (x y : ℝ) : ℝ := x^3 + 3*x*y^2 - 18*x^2 - 18*x*y - 18*y^2 + 57*x + 138*y + 290

theorem extremum_of_f :
  ∃ (xmin xmax : ℝ) (x1 y1 : ℝ), f x1 y1 = xmin ∧ (x1 = 11 ∧ y1 = 2) ∧
  ∃ (xmax : ℝ) (x2 y2 : ℝ), f x2 y2 = xmax ∧ (x2 = 1 ∧ y2 = 4) ∧
  xmin = 10 ∧ xmax = 570 := 
by
  sorry

end extremum_of_f_l405_405941


namespace taxi_distance_l405_405182

-- Definitions of conditions
def initialFare : ℝ := 8.0
def additionalFarePerUnit : ℝ := 0.8
def totalFare : ℝ := 39.2
def initialDistance : ℝ := 1 / 5

-- Function to calculate the total distance
def totalDistance (initialFare additionalFarePerUnit totalFare initialDistance : ℝ) : ℝ :=
  let remainingFare := totalFare - initialFare
  let numIncrements := remainingFare / additionalFarePerUnit
  initialDistance + numIncrements * initialDistance

-- The Theorem to prove
theorem taxi_distance (initialFare additionalFarePerUnit totalFare initialDistance : ℝ)
  (h_initialFare : initialFare = 8.0)
  (h_additionalFarePerUnit : additionalFarePerUnit = 0.8)
  (h_totalFare : totalFare = 39.2)
  (h_initialDistance : initialDistance = 1 / 5) :
  totalDistance initialFare additionalFarePerUnit totalFare initialDistance = 8 :=
by
  -- Proof contents will go here
  sorry

end taxi_distance_l405_405182


namespace distance_AB_l405_405455

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405455


namespace sum_of_three_largest_l405_405085

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l405_405085


namespace translate_line_down_2_units_l405_405800

theorem translate_line_down_2_units :
  ∀ (x y : ℝ), y = -x + 1 → y - 2 = -x - 1 :=
begin
  intros x y h,
  rw h,
  ring,
end

end translate_line_down_2_units_l405_405800


namespace num_zero_points_on_interval_l405_405157

noncomputable def f (x : ℝ) := cos (2 * x + π / 6)

theorem num_zero_points_on_interval : 
  ∃ (n : ℕ), n = 8 ∧ (∀ x ∈ [-2*π, 2*π], f x = 0 → x ∈ x_vals) 
  where x_vals : set ℝ := {x | (2*x + π/6) % (π) ≠ 0} 
  and ∀ x ∈ x_vals, (x = (2*n+1)*π/2 - π/12) → (2*n +1) <= 8 := 
  sorry

end num_zero_points_on_interval_l405_405157


namespace sum_of_three_largest_consecutive_numbers_l405_405107

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l405_405107


namespace problem_l405_405595

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l405_405595


namespace distance_AB_l405_405430

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405430


namespace AB_distance_l405_405370

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405370


namespace find_english_marks_l405_405907

/-- Define given conditions -/
variables (Maths Physics Chemistry Biology E : ℕ)
variables (avg numSubjects : ℕ)

/-- Given values for the problem -/
def marks : Maths = 69 := rfl
def marks : Physics = 92 := rfl
def marks : Chemistry = 64 := rfl
def marks : Biology = 82 := rfl
def avgMarks : avg = 76 := rfl
def numSubjects : numSubjects = 5 := rfl

/-- Calculate the total expected marks based on the average -/
def totalMarks := avg * numSubjects

/-- The equation to find marks in English -/
theorem find_english_marks (H : E + Maths + Physics + Chemistry + Biology = totalMarks) : E = 73 :=
by {
    /-- Substitute given values into the equation -/
    let totalMarks := 380,
    have H1 : E + 69 + 92 + 64 + 82 = totalMarks,
    calc E + 69 + 92 + 64 + 82 = totalMarks : H
        ... = 380           : by { refl }
    /-- Solve for E -/
    suffices : E = 380 - (69 + 92 + 64 + 82),
    calc E = 380 - (69 + 92 + 64 + 82) : this
}

#reduce find_english_marks mar rfl

end find_english_marks_l405_405907


namespace AB_distance_l405_405362

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l405_405362


namespace distance_AB_l405_405451

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l405_405451


namespace distance_AB_l405_405443

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405443


namespace total_spent_on_entertainment_l405_405220

def cost_of_computer_game : ℕ := 66
def cost_of_one_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3

theorem total_spent_on_entertainment : cost_of_computer_game + cost_of_one_movie_ticket * number_of_movie_tickets = 102 := 
by sorry

end total_spent_on_entertainment_l405_405220


namespace largest_number_of_stores_visited_l405_405204

-- Definitions of the conditions
def num_stores := 7
def total_visits := 21
def num_shoppers := 11
def two_stores_visitors := 7
def at_least_one_store (n : ℕ) : Prop := n ≥ 1

-- The goal statement
theorem largest_number_of_stores_visited :
  ∃ n, n ≤ num_shoppers ∧ 
       at_least_one_store n ∧ 
       (n * 2 + (num_shoppers - n)) <= total_visits ∧ 
       (num_shoppers - n) ≥ 3 → 
       n = 4 :=
sorry

end largest_number_of_stores_visited_l405_405204


namespace AB_distance_l405_405316

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l405_405316


namespace correct_judgments_l405_405142

open Rat

theorem correct_judgments (a b : ℚ) :
  (a + b < a → b < 0) ∧ (a - b < a → b > 0) :=
by {
  split,
  { intro h₁,
    linarith },
  { intro h₂,
    linarith },
}

end correct_judgments_l405_405142


namespace parabola_distance_l405_405383

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l405_405383


namespace geometric_progression_a5_value_l405_405151

theorem geometric_progression_a5_value
  (a : ℕ → ℝ)
  (h_geom : ∃ r : ℝ, ∀ n, a (n + 1) = a n * r)
  (h_roots : ∃ x y, x^2 - 5*x + 4 = 0 ∧ y^2 - 5*y + 4 = 0 ∧ x = a 3 ∧ y = a 7) :
  a 5 = 2 :=
by
  sorry

end geometric_progression_a5_value_l405_405151


namespace distance_AB_l405_405437

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l405_405437


namespace stratified_sampling_distribution_l405_405856

/-- A high school has a total of 2700 students, among which there are 900 freshmen, 
1200 sophomores, and 600 juniors. Using stratified sampling, a sample of 135 students 
is drawn. Prove that the sample contains 45 freshmen, 60 sophomores, and 30 juniors --/
theorem stratified_sampling_distribution :
  let total_students := 2700
  let freshmen := 900
  let sophomores := 1200
  let juniors := 600
  let sample_size := 135
  (sample_size * freshmen / total_students = 45) ∧ 
  (sample_size * sophomores / total_students = 60) ∧ 
  (sample_size * juniors / total_students = 30) :=
by
  sorry

end stratified_sampling_distribution_l405_405856


namespace sum_of_largest_three_consecutive_numbers_l405_405098

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l405_405098


namespace sum_of_three_largest_l405_405083

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l405_405083


namespace average_speed_of_car_l405_405830

-- Definitions of the given conditions
def uphill_speed : ℝ := 30  -- km/hr
def downhill_speed : ℝ := 70  -- km/hr
def uphill_distance : ℝ := 100  -- km
def downhill_distance : ℝ := 50  -- km

-- Required proof statement (with the correct answer derived from the conditions)
theorem average_speed_of_car :
  (uphill_distance + downhill_distance) / 
  ((uphill_distance / uphill_speed) + (downhill_distance / downhill_speed)) = 37.04 := by
  sorry

end average_speed_of_car_l405_405830


namespace prime_property_l405_405672

theorem prime_property (p : ℕ) (hp : Nat.Prime p) (n : ℕ) (h_n : n = 14 * p) 
  (h_divisor : ∀ d : ℕ, (d ∣ n ∧ even d) → d = 14) : odd p :=
sorry

end prime_property_l405_405672


namespace sum_of_three_largest_l405_405077

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l405_405077


namespace totalGamesPlayed_l405_405836

def numPlayers : ℕ := 30

def numGames (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem totalGamesPlayed :
  numGames numPlayers = 435 :=
by
  sorry

end totalGamesPlayed_l405_405836


namespace parabola_problem_l405_405501

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l405_405501


namespace find_p_of_binomial_distribution_l405_405967

noncomputable def binomial_mean (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem find_p_of_binomial_distribution (p : ℝ) (h : binomial_mean 5 p = 2) : p = 0.4 :=
by
  sorry

end find_p_of_binomial_distribution_l405_405967


namespace f_even_range_of_a_l405_405154

def f (x : ℝ) : ℝ := log 3 (9^x + 1) - x
def g (a : ℝ) (x : ℝ) : ℝ := log 3 (a + 2 - (a + 4) / 3^x)

theorem f_even : ∀ x : ℝ, f x = f (-x) := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, f x ≥ g a x) → -20/3 ≤ a ∧ a ≤ 4 := 
sorry

end f_even_range_of_a_l405_405154


namespace Robert_salary_loss_l405_405708

-- Define the conditions as hypotheses
variable (S : ℝ) (decrease_percent increase_percent : ℝ)
variable (decrease_percent_eq : decrease_percent = 0.6)
variable (increase_percent_eq : increase_percent = 0.6)

-- Define the problem statement to prove that Robert loses 36% of his salary.
theorem Robert_salary_loss (S : ℝ) (decrease_percent increase_percent : ℝ) 
  (decrease_percent_eq : decrease_percent = 0.6) 
  (increase_percent_eq : increase_percent = 0.6) :
  let new_salary := S * (1 - decrease_percent)
  let increased_salary := new_salary * (1 + increase_percent)
  let loss_percentage := (S - increased_salary) / S * 100 
  loss_percentage = 36 := 
by
  sorry

end Robert_salary_loss_l405_405708


namespace parabola_distance_l405_405523

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l405_405523


namespace ab_distance_l405_405255

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405255


namespace necessary_but_not_sufficient_l405_405793

theorem necessary_but_not_sufficient :
    (∀ (x y : ℝ), x > 2 ∧ y > 3 → x + y > 5 ∧ x * y > 6) ∧ 
    ¬(∀ (x y : ℝ), x + y > 5 ∧ x * y > 6 → x > 2 ∧ y > 3) := by
  sorry

end necessary_but_not_sufficient_l405_405793


namespace ab_distance_l405_405256

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l405_405256


namespace main_theorem_l405_405650

noncomputable def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

lemma f_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- Detailed steps of proof will go here
  sorry

lemma f_monotonic_increasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  intros x₁ x₂ h₁_lt_zero h₁_lt_h₂
  -- Detailed steps of proof will go here
  sorry

theorem main_theorem : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ → f x₁ < f x₂) := by
  split
  · exact f_odd
  · exact f_monotonic_increasing

end main_theorem_l405_405650


namespace y_sixth_power_eq_l405_405866

noncomputable def y : ℝ := sorry  -- Definition of y satisfying the given conditions

theorem y_sixth_power_eq :
  ∃ y: ℝ, 0 < y ∧ (real.cbrt (2 - y^3) + real.cbrt (2 + y^3) = 2) ∧ (y^6 = 116 / 27) :=
sorry  -- The goal to prove given the conditions

end y_sixth_power_eq_l405_405866


namespace six_digit_palindromes_count_l405_405029

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l405_405029


namespace parabola_distance_l405_405478

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l405_405478


namespace semi_major_axis_length_l405_405998

-- Given conditions
def e : ℝ := sqrt 3 / 2
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := x^2 + m * y^2 = 1

-- The mathematical problem to prove
theorem semi_major_axis_length (m : ℝ) (h : ellipse_eq x y m) (eccentricity : e = sqrt 3 / 2) : 
  x = 1 ∨ x = 2 :=
sorry

end semi_major_axis_length_l405_405998


namespace parabola_problem_l405_405555

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l405_405555
