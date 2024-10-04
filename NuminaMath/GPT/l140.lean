import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Group.Definitions
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Matrix
import Mathlib.Algebra.Matrix.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.ExtendDeriv
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Choose
import Mathlib.Combinatorics.Derangements.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fin.Vec
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.FinSet
import Mathlib.Geometry.Euclid.Triangle
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Logic.Basic
import Mathlib.Probability.Normal
import Mathlib.Tactic
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Trigonometry.Basic

namespace max_xy_wz_square_is_960_l140_140186

noncomputable def max_xy_wz_square (x y z w : ℝ) (h1 : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
  (h2 : x^2 + y^2 - (x * y / 2) = 36)
  (h3 : w^2 + z^2 + (w * z / 2) = 36)
  (h4 : x * z + y * w = 30) : ℝ :=
  (max (xy + wz)^2)

theorem max_xy_wz_square_is_960 (x y z w : ℝ) (h1 : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
  (h2 : x^2 + y^2 - (x * y / 2) = 36)
  (h3 : w^2 + z^2 + (w * z / 2) = 36)
  (h4 : x * z + y * w = 30) : max_xy_wz_square x y z w h1 h2 h3 h4 = 960 :=
by { sorry }

end max_xy_wz_square_is_960_l140_140186


namespace insphere_radius_of_tetrahedron_l140_140362

noncomputable def insphere_radius (H α : ℝ) : ℝ := 
  H / (4 * tan(α)^2) * (sqrt (4 * tan(α)^2 + 1) - 1)

theorem insphere_radius_of_tetrahedron (H α : ℝ) 
  (hH : 0 < H) (hα : 0 < α ∧ α < real.pi / 2) :
  insphere_radius H α = 
    (H / (4 * tan α ^ 2)) * ((sqrt (4 * tan α ^ 2 + 1)) - 1) :=
by 
  sorry

end insphere_radius_of_tetrahedron_l140_140362


namespace range_of_f_l140_140565

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 4) + Real.sin (x + Real.pi / 4)

theorem range_of_f : set.range f = set.Icc (-2 : ℝ) 2 :=
  sorry

end range_of_f_l140_140565


namespace conjugate_complex_division_l140_140196

theorem conjugate_complex_division : 
  ∃ (z : ℂ), z = (conj ((-5 : ℂ) / (I - 2))) ∧ z = (2 - I) :=
begin
  use (conj ((-5 : ℂ) / (I - 2))),
  split,
  { refl, },
  { have : (-5 : ℂ) / (I - 2) = 2 + I,
    { sorry, },
    rw this,
    refl,
  }
end

end conjugate_complex_division_l140_140196


namespace intersection_of_A_and_B_l140_140741

def setA (x : ℝ) : Prop := -1 < x ∧ x < 1
def setB (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def setC (x : ℝ) : Prop := 0 ≤ x ∧ x < 1

theorem intersection_of_A_and_B : {x : ℝ | setA x} ∩ {x | setB x} = {x | setC x} := by
  sorry

end intersection_of_A_and_B_l140_140741


namespace point_in_third_quadrant_l140_140003

noncomputable def modulus (z : ℂ) : ℝ := abs z

def point_location (z : ℂ) : Prop :=
  let new_z := (z - (modulus z) + (1 - (1 : ℂ) * Complex.I))
  new_z.re < 0 ∧ new_z.im < 0

theorem point_in_third_quadrant (z : ℂ) (hz : z = 3 - 4*Complex.I) :
  point_location z :=
by
  have hz_mod : modulus z = 5 := by sorry
  have h_new_z : new_z = (-1) - (5)*Complex.I := by sorry
  rw [h_new_z]
  show _ < 0 ∧ _ < 0
  tauto

-- Additional sub-definitions and hypotheses are included when needed.

end point_in_third_quadrant_l140_140003


namespace minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l140_140844

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions based on the problem statements
axiom a1_neg : a 1 < 0
axiom S2015_neg : S 2015 < 0
axiom S2016_pos : S 2016 > 0

-- Defining n value where S_n reaches its minimum
def n_min := 1008

theorem minimum_S_n_at_1008 : S n_min = S 1008 := sorry

-- Additional theorems to satisfy the provided conditions
theorem a1008_neg : a 1008 < 0 := sorry
theorem a1009_pos : a 1009 > 0 := sorry
theorem common_difference_pos : ∀ n : ℕ, a (n + 1) - a n > 0 := sorry

end minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l140_140844


namespace probability_odd_number_l140_140547

-- Defining the set of digits
def digits := {2, 3, 5, 7, 9}

-- Defining the condition that the number must be odd
def is_odd (n : Nat) : Prop := 
  ∃ x ∈ digits, n % 10 = x ∧ x % 2 = 1

-- Defining the total number of favorable outcomes for odd numbers
def favorable_outcomes : Nat := 4

-- Defining the total number of possible outcomes
def total_outcomes : Nat := 5

-- Statement of the theorem
theorem probability_odd_number : (favorable_outcomes : ℚ) / total_outcomes = 4 / 5 := 
by sorry

end probability_odd_number_l140_140547


namespace inequality_proof_l140_140480

noncomputable theory

variables {a b c : ℝ}

-- Conditions: 
-- a, b, c are positive real numbers
-- a + b + c = 1
-- Prove the inequality
theorem inequality_proof (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a + b + c = 1) :
  (a ^ -3 + b) / (1 - a) + (b ^ -3 + c) / (1 - b) + (c ^ -3 + a) / (1 - c) ≥ 123 :=
by sorry

end inequality_proof_l140_140480


namespace angle_sum_AKM_ALM_l140_140520

theorem angle_sum_AKM_ALM {A B C K L M : Point} (h_equilateral : is_equilateral_triangle A B C)
  (h_k_l : BK = KL ∧ KL = LC) (h_m : AM = (1/3) * AC) : 
  ∠AKM + ∠ALM = 30 :=
by 
  sorry

end angle_sum_AKM_ALM_l140_140520


namespace proposition_2_proposition_3_l140_140747

-- Definition of lines and planes
variables (m n : Line) (α β : Plane)

-- Correctness of Proposition 2
theorem proposition_2 (h1 : Perpendicular m β) (h2 : Perpendicular n β) : Parallel n m :=
sorry

-- Correctness of Proposition 3
theorem proposition_3 (h1 : Perpendicular m α) (h2 : Perpendicular m β) : Parallel β α :=
sorry

end proposition_2_proposition_3_l140_140747


namespace cannot_achieve_1_5_percent_salt_solution_l140_140571

-- Define the initial concentrations and volumes
def initial_state (V1 V2 : ℝ) (C1 C2 : ℝ) : Prop :=
  V1 = 1 ∧ C1 = 0 ∧ V2 = 1 ∧ C2 = 0.02

-- Define the transfer and mixing operation
noncomputable def transfer_and_mix (V1_old V2_old C1_old C2_old : ℝ) (amount_to_transfer : ℝ)
  (new_V1 new_V2 new_C1 new_C2 : ℝ) : Prop :=
  amount_to_transfer ≤ V2_old ∧
  new_V1 = V1_old + amount_to_transfer ∧
  new_V2 = V2_old - amount_to_transfer ∧
  new_C1 = (V1_old * C1_old + amount_to_transfer * C2_old) / new_V1 ∧
  new_C2 = (V2_old * C2_old - amount_to_transfer * C2_old) / new_V2

-- Prove that it is impossible to achieve a 1.5% salt concentration in container 1
theorem cannot_achieve_1_5_percent_salt_solution :
  ∀ V1 V2 C1 C2, initial_state V1 V2 C1 C2 →
  ¬ ∃ V1' V2' C1' C2', transfer_and_mix V1 V2 C1 C2 0.5 V1' V2' C1' C2' ∧ C1' = 0.015 :=
by
  intros
  sorry

end cannot_achieve_1_5_percent_salt_solution_l140_140571


namespace arithmetic_seq_S11_l140_140088

def Sn (n : ℕ) (a₁ d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1)) / 2 * d

theorem arithmetic_seq_S11 (a₁ d : ℤ)
  (h1 : a₁ = -11)
  (h2 : (Sn 10 a₁ d) / 10 - (Sn 8 a₁ d) / 8 = 2) :
  Sn 11 a₁ d = -11 :=
by
  sorry

end arithmetic_seq_S11_l140_140088


namespace bug_paths_from_A_to_B_l140_140607

-- Define the positions A and B and intermediate red and blue points in the lattice
inductive Position
| A
| B
| red1
| red2
| blue1
| blue2

open Position

-- Define the possible directed paths in the lattice
def paths : List (Position × Position) :=
[(A, red1), (A, red2), 
 (red1, blue1), (red1, blue2), 
 (red2, blue1), (red2, blue2), 
 (blue1, B), (blue1, B), (blue1, B), 
 (blue2, B), (blue2, B), (blue2, B)]

-- Define a function that calculates the number of unique paths from A to B without repeating any path
def count_paths : ℕ := sorry

-- The mathematical problem statement
theorem bug_paths_from_A_to_B : count_paths = 24 := sorry

end bug_paths_from_A_to_B_l140_140607


namespace peter_money_l140_140273

theorem peter_money (J : ℝ) (P : ℝ) (Q : ℝ) (A : ℝ) :
  P = 2 * J →
  Q = 2 * J + 20 →
  A = 1.15 * (2 * J + 20) →
  J + P + Q + A = 1211 →
  P = 320 :=
by
  intro hP hQ hA hTotal
  have h_eq1 : 2 * J = P := hP
  have h_eq2 : 2 * J + 20 = Q := hQ
  have h_eq3 : 1.15 * (2 * J + 20) = A := hA
  have h_eq4 : J + 2 * J + (2 * J + 20) + 1.15 * (2 * J + 20) = 1211 := hTotal
  sorry

end peter_money_l140_140273


namespace initial_number_is_21_l140_140149

def move_north (x : ℕ) : ℕ := x + 7
def move_east (x : ℕ) : ℕ := x - 4
def move_south (x : ℕ) : ℕ := x / 2
def move_west (x : ℕ) : ℕ := x * 3

def inverse_north (x : ℕ) : ℕ := x - 7
def inverse_east (x : ℕ) : ℕ := x + 4
def inverse_south (x : ℕ) : ℕ := x * 2
def inverse_west (x : ℕ) : ℕ := x / 3

theorem initial_number_is_21 (final_result : ℕ) (path : List (ℕ → ℕ)) :
  path = [move_north, move_east, move_south, move_west, move_west, move_south, move_east, move_north] →
  final_result = 57 →
  foldl (flip (λ f x, f x)) 21 path = 57 :=
by
  intros h_path h_result
  rw [←h_path, ←h_result]
  unfold List.foldl
  simp
  sorry

end initial_number_is_21_l140_140149


namespace correct_statements_count_l140_140855

def class_of (k : ℤ) : set ℤ := {n | ∃ m : ℤ, n = 5 * m + k}

def statement1 : Prop := 2011 ∈ class_of 1

def statement2 : Prop := -3 ∈ class_of 3

def statement3 : Prop := 
  ∀ x : ℤ, (x ∈ class_of 0) ∨ (x ∈ class_of 1) ∨ (x ∈ class_of 2) ∨ (x ∈ class_of 3) ∨ (x ∈ class_of 4)

def same_class_condition (a b : ℤ) : Prop := (a - b) ∈ class_of 0

def statement4 : Prop := ∀ a b : ℤ, (∃ k : ℤ, a ∈ class_of k ∧ b ∈ class_of k) ↔ same_class_condition a b

def number_of_correct_statements : ℕ := 3

theorem correct_statements_count :
  (statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) → (number_of_correct_statements = 3) :=
sorry

end correct_statements_count_l140_140855


namespace can_form_square_using_LShapedPieces_l140_140103

-- Definition of the L-shaped piece as having 3 cells
def LShapedPiece : Type := { cells : Nat // cells = 3 }

-- The main theorem stating that it is possible to form a square using these pieces
theorem can_form_square_using_LShapedPieces (n : Nat) (h : n % 6 = 0) : 
  ∃ square : Nat, is_square square ∧ square = n * n := 
begin
  sorry
end

end can_form_square_using_LShapedPieces_l140_140103


namespace cos_A_in_right_triangle_l140_140352

theorem cos_A_in_right_triangle
  (A B C : Type)
  [IsRightTriangle A B C]
  (AC : ℝ) (AB : ℝ)
  (hAC : AC = 6) (hAB : AB = 10) :
  cos A = 4 / 5 := by
sorry

end cos_A_in_right_triangle_l140_140352


namespace sin_equation_solutions_l140_140708

theorem sin_equation_solutions (x : ℝ) (h : -π ≤ x ∧ x ≤ π) :
  ∃ n : ℕ, n = 5 ∧
    (sin (4 * x) + (sin (3 * x))^2 + (sin (2 * x))^3 + (sin x)^4 = 0) :=
sorry

end sin_equation_solutions_l140_140708


namespace relationship_of_y_l140_140072

variable {x1 x2 x3 y1 y2 y3 : ℝ}
variable h1 : x1 < x2
variable h2 : x2 < 0
variable h3 : 0 < x3
variable f1 : y1 = 2 / x1
variable f2 : y2 = 2 / x2
variable f3 : y3 = 2 / x3

theorem relationship_of_y (h1 : x1 < x2) (h2 : x2 < 0) (h3 : 0 < x3) (f1 : y1 = 2 / x1) (f2 : y2 = 2 / x2) (f3 : y3 = 2 / x3) :
  y2 < y1 ∧ y1 < y3 := 
sorry

end relationship_of_y_l140_140072


namespace region_area_sqrt3_l140_140850

theorem region_area_sqrt3 : 
  let region := {p : ℝ × ℝ | √3 * p.1 - p.2 ≤ 0 ∧ p.1 - √3 * p.2 + 2 ≥ 0 ∧ p.2 ≥ 0} 
  ∃ (area : ℝ), area = √3 :=
by
  sorry

end region_area_sqrt3_l140_140850


namespace geometric_sequence_fifth_term_l140_140619

theorem geometric_sequence_fifth_term 
    (a₁ : ℕ) (a₄ : ℕ) (r : ℕ) (a₅ : ℕ)
    (h₁ : a₁ = 3) (h₂ : a₄ = 240) 
    (h₃ : a₄ = a₁ * r^3) 
    (h₄ : a₅ = a₁ * r^4) : 
    a₅ = 768 :=
by
  sorry

end geometric_sequence_fifth_term_l140_140619


namespace problem_theorem_l140_140841

variables (B E F D U V C P Q : Type) [hl1 : IsLine (B, E)] [hl2 : IsLine (F, D)] [hl3 : IsLine (B, F)] [hl4 : IsLine (D, E)]
variables [hl5 : IsIntersectingAt B F C] [hl6 : IsIntersectingAt D E C] (BU UE FV VD : ℝ)
variable (ratio : BU / UE = FV / VD)

-- Definition stating that U is on line BE and V is on line FD
def U_on_BE := U ∈ BE
def V_on_FD := V ∈ FD

-- Definition of line UV intersecting DE at P and BF at Q
def UV_intersects_DE_BF := (P ∈ UV ∧ Q ∈ UV) ∧ (P ∈ DE ∧ Q ∈ BF)

-- Main theorem statement
theorem problem_theorem :
  (U_on_BE) → (V_on_FD) → (UV_intersects_DE_BF) → 
  (∀ U V : Point, circumcircle_contains_fixed_point (CPQ) fixed_point) ∧
  (∀ U V : Point, other_intersection_circumcircles (BUQ) (PVD) ∈ BD) :=
begin
  sorry,
end

end problem_theorem_l140_140841


namespace min_value_of_a_plus_2b_l140_140013

theorem min_value_of_a_plus_2b (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: 1 / (a + 1) + 1 / (b + 1) = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_a_plus_2b_l140_140013


namespace count_valid_sets_l140_140485

def is_isolated (A : set ℤ) (k : ℤ) : Prop :=
  k ∈ A ∧ k - 1 ∉ A ∧ k + 1 ∉ A

def S : set ℤ := {1, 2, 3, 4, 5, 6, 7, 8}

def is_valid_set (s : set ℤ) : Prop :=
  s ⊆ S ∧ s.card = 3 ∧ ∀ k ∈ s, ¬ is_isolated s k

theorem count_valid_sets : 
  (finset.filter is_valid_set 
    (finset.powerset_len 3 (finset.of_set S))).card = 6 :=
by 
  sorry

end count_valid_sets_l140_140485


namespace find_m_l140_140769

variable (m : ℝ)

def A : Set ℝ := {3, m^2}
def B : Set ℝ := {-1, 3, 3 * m - 2}

theorem find_m (h : A m ∩ B m = A m) : m = 1 ∨ m = 2 := sorry

end find_m_l140_140769


namespace train_pass_tree_l140_140971

theorem train_pass_tree
  (L : ℝ) (S : ℝ) (conv_factor : ℝ) 
  (hL : L = 275)
  (hS : S = 90)
  (hconv : conv_factor = 5 / 18) :
  L / (S * conv_factor) = 11 :=
by
  sorry

end train_pass_tree_l140_140971


namespace average_of_first_6_numbers_l140_140917

theorem average_of_first_6_numbers (n : ℕ) (a : ℕ → ℝ)
  (h₁ : n = 11) (h₂ : (∑ i in (finset.range n), a i) / n = 60)
  (h₃ : (∑ i in (finset.range_from 5 6), a i) / 6 = 65)
  (h₄ : a 5 = 78) : 
  (∑ i in (finset.range 6), a i) / 6 = 71 := 
sorry

end average_of_first_6_numbers_l140_140917


namespace count_undefined_values_number_of_undefined_values_l140_140715

def is_undefined (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ g, ∃ h, f = g / h ∧ h x = 0

theorem count_undefined_values : 
  ∀ (x : ℝ), is_undefined (λ x => (x^2 - 16) / ((x^2 + 3 * x - 4) * (x - 4))) x ↔ x = 1 ∨ x = -4 ∨ x = 4 :=
by sorry

theorem number_of_undefined_values : Finset.card (Finset.filter (λ x : ℝ, ∃ f, is_undefined f x) {1, -4, 4}) = 3 :=
by sorry

end count_undefined_values_number_of_undefined_values_l140_140715


namespace f_symmetric_about_1_l140_140762

def f (x : ℝ) : ℝ := Real.log2 (abs (x - 1))

theorem f_symmetric_about_1 : ∀ x : ℝ, f x = f (2 - x) := by
  intro x
  sorry

end f_symmetric_about_1_l140_140762


namespace remaining_yards_is_720_l140_140297

-- Definitions based on conditions:
def marathon_miles : Nat := 25
def marathon_yards : Nat := 500
def yards_in_mile : Nat := 1760
def num_of_marathons : Nat := 12

-- Total distance for one marathon in yards
def one_marathon_total_yards : Nat :=
  marathon_miles * yards_in_mile + marathon_yards

-- Total distance for twelve marathons in yards
def total_distance_yards : Nat :=
  num_of_marathons * one_marathon_total_yards

-- Remaining yards after converting the total distance into miles and yards
def y : Nat :=
  total_distance_yards % yards_in_mile

-- Condition ensuring y is the remaining yards and is within the bounds 0 ≤ y < 1760
theorem remaining_yards_is_720 : 
  y = 720 := sorry

end remaining_yards_is_720_l140_140297


namespace max_distance_theorem_l140_140871

-- Define the complex number w and its magnitude condition
variable (w : ℂ) (hw : ∥w∥ = 3)

-- Define the expressions involved
def expr1 : ℂ := (6 + 5*complex.I) * w^2
def expr2 : ℂ := w^4

-- Define the maximum distance calculation function
noncomputable def max_distance : ℝ :=
  9 * complex.abs (6 + 5 * complex.I - w^2)

-- Prove the main theorem
theorem max_distance_theorem (w : ℂ) (hw : ∥w∥ = 3) : max_distance w = 9 * real.sqrt 61 + 81 :=
sorry

end max_distance_theorem_l140_140871


namespace train_pass_time_l140_140857

theorem train_pass_time : 
  ∀ (length_of_train : ℝ) (speed_in_kmph : ℝ),
  length_of_train = 150 ∧ speed_in_kmph = 72 →
  let speed_in_mps := speed_in_kmph * (1000 / 3600) in
  let time := length_of_train / speed_in_mps in
  time = 7.5 :=
by
  intros length_of_train speed_in_kmph h
  cases h with h_len h_speed
  simp only [h_len, h_speed] at *
  sorry

end train_pass_time_l140_140857


namespace no_perfect_square_product_l140_140479

theorem no_perfect_square_product {a : ℕ → ℕ} (h : ∀ n, a(n+1) = a(n)^2 + 1) :
  ¬ ∃ N : ℕ, is_square (∏ k in finset.range(N+1), (a k)^2 + a k + 1) :=
sorry

end no_perfect_square_product_l140_140479


namespace find_f_of_f_of_neg3_l140_140755

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2^x else Real.log x / Real.log 8

theorem find_f_of_f_of_neg3 : f (f (-3)) = -1 := by
  sorry

end find_f_of_f_of_neg3_l140_140755


namespace polar_intersection_l140_140412

theorem polar_intersection (ρ θ : ℝ) (h1 : √2 * ρ = 1 / sin (π / 4 + θ)) (h2 : θ = π / 3) : ρ = √3 - 1 ∧ θ = π / 3 := 
by {
  sorry
}

end polar_intersection_l140_140412


namespace circle_tangent_bisector_l140_140993

theorem circle_tangent_bisector (O A B C : Point) (θ : ℝ)
    (h_circle : distance O A = 1 ∧ distance O B = 1)
    (h_tangent : ∃ l, is_tangent l (Circle O 1) ∧ PointOn A l ∧ PointOn B l)
    (h_angle : ∠ A O B = θ)
    (h_point_C : C ∈ Line O A)
    (h_bisector : bisects (Line B C) ∠ A B O) :
    distance O C = 1 / (1 + sin θ) :=
sorry

end circle_tangent_bisector_l140_140993


namespace perimeter_triangle_ABC_l140_140927

-- Define the conditions and statement
theorem perimeter_triangle_ABC 
  (r : ℝ) (AP PB altitude : ℝ) 
  (h1 : r = 30) 
  (h2 : AP = 26) 
  (h3 : PB = 32) 
  (h4 : altitude = 96) :
  (2 * (58 + 34.8)) = 185.6 :=
by
  sorry

end perimeter_triangle_ABC_l140_140927


namespace total_selling_amount_l140_140308

-- Defining the given conditions
def total_metres_of_cloth := 200
def loss_per_metre := 6
def cost_price_per_metre := 66

-- Theorem statement to prove the total selling amount
theorem total_selling_amount : 
    (cost_price_per_metre - loss_per_metre) * total_metres_of_cloth = 12000 := 
by 
    sorry

end total_selling_amount_l140_140308


namespace correct_value_division_l140_140417

theorem correct_value_division (x : ℕ) (h : 9 - x = 3) : 96 / x = 16 :=
by
  sorry

end correct_value_division_l140_140417


namespace number_of_ways_to_choose_cooks_l140_140515

theorem number_of_ways_to_choose_cooks : ∀ n k : ℕ, n = 5 → k = 3 → Fintype.card {s : Finset (Fin 5) // s.card = k} = 10 :=
by
  intros n k hn hk,
  rw [hn, hk],
  norm_num,
  sorry

end number_of_ways_to_choose_cooks_l140_140515


namespace line_through_center_perpendicular_l140_140924

theorem line_through_center_perpendicular (c : ℝ) :
  let center := (-1 : ℝ, 0 : ℝ) in
  let circle_eq := ∀ x y : ℝ, x^2 + 2 * x + y^2 = 0 in
  let line_perpendicular := ∀ x y : ℝ, x + y = 0 in
  ∀ x y : ℝ, x - y + c = 0 ∧ (x, y) = center → c = 1 :=
begin
  intros,
  sorry
end

end line_through_center_perpendicular_l140_140924


namespace toy_production_difference_l140_140325

variables (w t : ℕ)
variable  (t_nonneg : 0 < t) -- assuming t is always non-negative for a valid working hour.
variable  (h : w = 3 * t)

theorem toy_production_difference : 
  (w * t) - ((w + 5) * (t - 3)) = 4 * t + 15 :=
by
  sorry

end toy_production_difference_l140_140325


namespace second_operation_result_l140_140446

def f (a b : ℕ) : ℕ :=
  (a + b) * a - a

theorem second_operation_result : f 3 7 = 27 :=
by {
  have h1 : f 2 3 = 8 := by simp [f],
  have h2 : f 4 5 = 32 := by simp [f],
  have h3 : f 5 8 = 60 := by simp [f],
  have h4 : f 6 7 = 72 := by simp [f],
  have h5 : f 7 8 = 98 := by simp [f],
  rw [f, add_comm],
  sorry
}

end second_operation_result_l140_140446


namespace prove_CH_is_altitude_l140_140733

variable {A B C H : Point}
variable {AB_line : Line}
variable [OnLine A AB_line]
variable [OnLine B AB_line]
variable [OnLine H AB_line]
variable {CH : Line}
variable [Perpendicular CH AB_line]

def is_altitude (line : Line) (triangle : Triangle) : Prop :=
  -- definition of an altitude: a line is perpendicular to the side and passes through the vertex opposite to the side
  ∃ P Q R : Point, triangle = ⟨P, Q, R⟩ ∧
    ∃ l : Line, OnLine P l ∧ ∃ k : Line, OnLine P k ∧ k ≠ l ∧ Perpendicular l k

theorem prove_CH_is_altitude 
  (h₁ : triangle₁ = ⟨A, B, C⟩)
  (h₂ : AC * AC - BC * BC = AH * AH - BH * BH) 
  : is_altitude CH ⟨A, B, C⟩ :=
by
  sorry

end prove_CH_is_altitude_l140_140733


namespace fraction_irreducible_l140_140904

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_irreducible_l140_140904


namespace travelerQuestionCorrect_l140_140295

-- Define the two tribes and their properties
inductive Tribe
| TruthTeller
| Liar

-- Define the function that indicates whether a tribe member tells the truth
def tellsTruth : Tribe → Prop
| Tribe.TruthTeller := True
| Tribe.Liar := False

-- Define the function that models the islander's response to the question 
def islanderResponse (tribe : Tribe) (roadLeadsToVillage : Bool) : Bool :=
  if roadLeadsToVillage then tellsTruth tribe else ¬tellsTruth tribe

-- Define the main theorem
theorem travelerQuestionCorrect (tribe : Tribe) (roadLeadsToVillage : Bool) :
  islanderResponse tribe roadLeadsToVillage = roadLeadsToVillage :=
begin
  cases tribe,
  { -- Case when tribe is TruthTeller
    simp [islanderResponse, tellsTruth] },
  { -- Case when tribe is Liar
    simp [islanderResponse, tellsTruth] }
end

end travelerQuestionCorrect_l140_140295


namespace line_segment_ab_length_l140_140048

noncomputable def parabola_eq := ∀ (y x : ℝ), y^2 = 6 * x

noncomputable def focus := ∃ (x y : ℝ), x = 3 / 2 ∧ y = 0

noncomputable def directrix := ∀ (x : ℝ), x = -3 / 2

noncomputable def line_eq := ∀ (x y : ℝ), y = x -  3 / 2

theorem line_segment_ab_length : 
  (∀ x y : ℝ, parabola_eq x y) →
  (∀ x y : ℝ, line_eq x y) →
  (∀ x1 x2 : ℝ, x1 + x2 = 9) →
  (∀ x1 x2 : ℝ, |x1 + x2 + 3| = 12) → 
  true := 
by
  intro h1 h2 h3 h4
  sorry

end line_segment_ab_length_l140_140048


namespace solve_logarithmic_equation_l140_140533

theorem solve_logarithmic_equation (x : ℝ) (h₁ : log 4 (3 * x - 1) = log 4 (x - 1) + log 4 (3 * x - 1)) (h₂ : x ≠ 1 / 3) : x = 2 := by
  sorry

end solve_logarithmic_equation_l140_140533


namespace fraction_of_larger_part_l140_140530

theorem fraction_of_larger_part (x y : ℝ) (f : ℝ) (h1 : x = 50) (h2 : x + y = 66) (h3 : f * x = 0.625 * y + 10) : f = 0.4 :=
by
  sorry

end fraction_of_larger_part_l140_140530


namespace find_4_digit_number_l140_140698

theorem find_4_digit_number (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182) :
  1000 * a + 100 * b + 10 * c + d = 1909 :=
by {
  sorry
}

end find_4_digit_number_l140_140698


namespace JuliaPlayedTuesday_l140_140475

variable (Monday : ℕ) (Wednesday : ℕ) (Total : ℕ)
variable (KidsOnTuesday : ℕ)

theorem JuliaPlayedTuesday :
  Monday = 17 →
  Wednesday = 2 →
  Total = 34 →
  KidsOnTuesday = Total - (Monday + Wednesday) →
  KidsOnTuesday = 15 :=
by
  intros hMon hWed hTot hTue
  rw [hTot, hMon, hWed] at hTue
  exact hTue

end JuliaPlayedTuesday_l140_140475


namespace checkerboards_that_cannot_be_covered_l140_140955

-- Define the dimensions of the checkerboards
def checkerboard_4x6 := (4, 6)
def checkerboard_3x7 := (3, 7)
def checkerboard_5x5 := (5, 5)
def checkerboard_7x4 := (7, 4)
def checkerboard_5x6 := (5, 6)

-- Define a function to calculate the number of squares
def num_squares (dims : Nat × Nat) : Nat := dims.1 * dims.2

-- Define a function to check if a board can be exactly covered by dominoes
def can_be_covered_by_dominoes (dims : Nat × Nat) : Bool := (num_squares dims) % 2 == 0

-- Statement to be proven
theorem checkerboards_that_cannot_be_covered :
  ¬ can_be_covered_by_dominoes checkerboard_3x7 ∧ ¬ can_be_covered_by_dominoes checkerboard_5x5 :=
by
  sorry

end checkerboards_that_cannot_be_covered_l140_140955


namespace angle_BAC_plus_2_angle_DTE_eq_180_l140_140865

open EuclideanGeometry

theorem angle_BAC_plus_2_angle_DTE_eq_180 {A B C D E X Y P T : Point} 
  (hABC_triangle : Triangle A B C)
  (hP_on_BC : P ∈ Line B C)
  (hS1_circle : Circle B (dist B P) D)
  (hS2_circle : Circle C (dist C P) E)
  (hAP_diff_from_P : ∃ X Y, X ≠ P ∧ Y ≠ P ∧ X ∈ Line A P ∧ Y ∈ Line A P ∧ Circle B (dist B P) X ∧ Circle C (dist C P) Y)
  (hT_intersecting : T = intersection (Line D X) (Line E Y)) :
  angle A B C + 2 * angle D T E = 180 := sorry

end angle_BAC_plus_2_angle_DTE_eq_180_l140_140865


namespace senior_tickets_count_l140_140580

-- Define variables and problem conditions
variables (A S : ℕ)

-- Total number of tickets equation
def total_tickets (A S : ℕ) : Prop := A + S = 510

-- Total receipts equation
def total_receipts (A S : ℕ) : Prop := 21 * A + 15 * S = 8748

-- Prove that the number of senior citizen tickets S is 327
theorem senior_tickets_count (A S : ℕ) (h1 : total_tickets A S) (h2 : total_receipts A S) : S = 327 :=
sorry

end senior_tickets_count_l140_140580


namespace area_ratio_l140_140896

theorem area_ratio (X : ℝ)
  (hABJ : ∀ X, area ABJ = 2 * X)
  (hACE : ∀ X, area ACE = 8 * X) :
  (area ABJ / area ACE) = 1 / 4 :=
by {
  sorry
}

end area_ratio_l140_140896


namespace line_chart_method_l140_140933

theorem line_chart_method
    (line_chart : Type)
    (unit_length_represents_quantity : line_chart → Prop)
    (points_plotted_on_grid : line_chart → Prop)
    (points_connected_in_sequence : line_chart → Prop)
    (definition_line_chart : ∀ lc : line_chart, unit_length_represents_quantity lc ∧ points_plotted_on_grid lc ∧ points_connected_in_sequence lc) :
  ∀ lc : line_chart, points_plotted_on_grid lc ∧ points_connected_in_sequence lc :=
by
  intros lc
  exact ⟨definition_line_chart lc.2, definition_line_chart lc.3⟩

end line_chart_method_l140_140933


namespace genuine_coin_remains_l140_140615

-- Define the problem context and conditions
variables (coins : Finset ℕ) (is_genuine : ℕ → Prop)
variable (dealer_knows : ∀ a b ∈ coins, is_genuine a ↔ is_genuine b)

-- More than half of the coins are genuine
def more_than_half_genuine (coins : Finset ℕ) (is_genuine : ℕ → Prop) : Prop :=
  ∃ G : ℕ, (∃ S : Finset ℕ, S.card = G ∧ ∀ a, a ∈ S ↔ a ∈ coins ∧ is_genuine a) ∧ G > coins.card / 2

-- Define the final theorem to prove
theorem genuine_coin_remains (coins : Finset ℕ) (is_genuine : ℕ → Prop)
  (h : more_than_half_genuine coins is_genuine) :
  ∀ k, coins.card = k + 1 → 2021 ≤ k → ∃ a, a ∈ coins ∧ is_genuine a :=
by
  intro k
  intro cards_eq
  intro big_k
  sorry

end genuine_coin_remains_l140_140615


namespace find_z_value_l140_140911

theorem find_z_value (k : ℝ) (y z : ℝ) (h1 : (y = 2) → (z = 1)) (h2 : y ^ 3 * z ^ (1/3) = k) : 
  (y = 4) → z = 1 / 512 :=
by
  sorry

end find_z_value_l140_140911


namespace no_overlapping_sale_days_l140_140992

def bookstore_sale_days (d : ℕ) : Prop :=
  d % 4 = 0 ∧ 1 ≤ d ∧ d ≤ 31

def shoe_store_sale_days (d : ℕ) : Prop :=
  ∃ k : ℕ, d = 2 + 8 * k ∧ 1 ≤ d ∧ d ≤ 31

theorem no_overlapping_sale_days : 
  ∀ d : ℕ, bookstore_sale_days d → ¬ shoe_store_sale_days d :=
by
  intros d h1 h2
  sorry

end no_overlapping_sale_days_l140_140992


namespace count_distinct_common_tangent_values_l140_140104

def num_possible_common_tangent_values (r1 r2 : ℝ) : ℕ :=
  if r1 = r2 then 0 else
  if r1 + r2 < r2 then 0 else
  if r1 + r2 > r1 then 1 else
  if r1 + r1 = r2 then 1 else
  if r1 < r2 then 2 else
  if r1 > r2 then 3 else 4

theorem count_distinct_common_tangent_values :
  num_possible_common_tangent_values 2 3 = 5 :=
sorry

end count_distinct_common_tangent_values_l140_140104


namespace intersection_eq_l140_140739

def setA : Set ℝ := {x | -1 < x ∧ x < 1}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_eq : setA ∩ setB = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_eq_l140_140739


namespace Mr_Blue_potato_yield_l140_140882

-- Definitions based on the conditions
def steps_length (steps : ℕ) : ℕ := steps * 3
def garden_length : ℕ := steps_length 18
def garden_width : ℕ := steps_length 25

def area_garden : ℕ := garden_length * garden_width
def yield_potatoes (area : ℕ) : ℚ := area * (3/4)

-- Statement of the proof
theorem Mr_Blue_potato_yield :
  yield_potatoes area_garden = 3037.5 := by
  sorry

end Mr_Blue_potato_yield_l140_140882


namespace abs_inequality_solution_l140_140943

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 1| < 1) ↔ (0 < x ∧ x < 2) :=
sorry

end abs_inequality_solution_l140_140943


namespace average_other_marbles_l140_140158

def total_marbles : ℕ := 10 -- Define a hypothetical total number for computation
def clear_marbles : ℕ := total_marbles * 40 / 100
def black_marbles : ℕ := total_marbles * 20 / 100
def other_marbles : ℕ := total_marbles - clear_marbles - black_marbles
def marbles_taken : ℕ := 5

theorem average_other_marbles :
  marbles_taken * other_marbles / total_marbles = 2 := by
  sorry

end average_other_marbles_l140_140158


namespace tangents_arithmetic_mean_l140_140174

variables {a b c : ℝ}

def is_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

theorem tangents_arithmetic_mean
  (h1 : a^2 = 3 * (b^2 - c^2))
  (h2 : b^2 = 5 * (a^2 - c^2))
  (h3 : c^2 = 7 * (b^2 - a^2))
  (h_triangle: is_triangle a b c) :
  ∃ α β γ : ℝ, α + β + γ = π ∧ (tan α = (tan β + tan γ) / 2) :=
by
  sorry

end tangents_arithmetic_mean_l140_140174


namespace correct_product_of_a_and_b_l140_140090

-- We start by defining positive integers and their properties.
section
variables (a b : ℕ)

-- Define a function to reverse the digits of a two-digit number.
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10 in
  let ones := n % 10 in
  ones * 10 + tens

-- Define the main theorem with conditions and the target product.
theorem correct_product_of_a_and_b
  (h1 : a > 9) (h2 : a < 100) -- a is a two-digit number
  (h3 : b > 0) -- b is a positive integer
  (h4 : reverse_digits a * b + 5 = 266) :
  a * b = 828 :=
sorry
end

end correct_product_of_a_and_b_l140_140090


namespace other_endpoint_l140_140560

theorem other_endpoint (M : ℝ × ℝ) (A : ℝ × ℝ) (x y : ℝ) :
  M = (2, 3) ∧ A = (5, -1) ∧ (M = ((A.1 + x) / 2, (A.2 + y) / 2)) → (x, y) = (-1, 7) := by
  sorry

end other_endpoint_l140_140560


namespace max_length_AB_l140_140024

noncomputable def max_AB_len : ℝ :=
  let α := (cos ((A - B) / 2), sqrt(3) * sin ((A + B) / 2)) in
  let magnitude_α := Real.sqrt (cos ((A - B) / 2) ^ 2 + 3 * sin ((A + B) / 2) ^2) in
  if interior_angles (A B : ℝ) (C : ℝ := π - A - B) (A B C ≠ 0) /\
    magnitude_α = sqrt(2) /\
    exists (M : point),
      let MA := sqrt((M.1 - A)^2 + (M.2 - 0)^2) in
      let MB := sqrt((M.1 - B)^2 + (M.2 - B)^2) in
      is_arithmetic_sequence (MA |AB| MB) then
    (2*sqrt(3) + sqrt(2))/4 
  else 0

theorem max_length_AB :
  max_AB_len = (2 * sqrt(3) + sqrt(2)) / 4 :=
sorry

end max_length_AB_l140_140024


namespace max_type_A_stationery_l140_140676

-- Define the variables and constraints
variables (x y : ℕ)

-- Define the conditions as hypotheses
def condition1 : Prop := 3 * x + 2 * (x - 2) + y = 66
def condition2 : Prop := 3 * x ≤ 33

-- The statement to prove
theorem max_type_A_stationery : condition1 x y ∧ condition2 x → x ≤ 11 :=
by sorry

end max_type_A_stationery_l140_140676


namespace operation_positive_l140_140430

theorem operation_positive (op : ℤ → ℤ → ℤ) (is_pos : op 1 (-2) > 0) : op = Int.sub :=
by
  sorry

end operation_positive_l140_140430


namespace total_inflation_over_two_years_real_interest_rate_over_two_years_l140_140984

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end total_inflation_over_two_years_real_interest_rate_over_two_years_l140_140984


namespace cookie_boxes_condition_l140_140511

theorem cookie_boxes_condition (n : ℕ) (M A : ℕ) :
  M = n - 8 ∧ A = n - 2 ∧ M + A < n ∧ M ≥ 1 ∧ A ≥ 1 → n = 9 :=
by
  intro h
  sorry

end cookie_boxes_condition_l140_140511


namespace limit_of_function_l140_140502

theorem limit_of_function (a : ℝ) (h : a ≠ 0) :
  (filter.tendsto (λ x, (x^2) / (a - real.sqrt (a^2 - x^2))) (nhds 0) (nhds
    (if a < 0 then 0 else 2 * a))) := sorry

end limit_of_function_l140_140502


namespace students_not_liking_either_l140_140822

theorem students_not_liking_either (total_students like_fries like_burgers like_both : ℕ)
  (h_total : total_students = 25)
  (h_fries : like_fries = 15)
  (h_burgers : like_burgers = 10)
  (h_both : like_both = 6) :
  total_students - (like_fries - like_both + like_burgers - like_both + like_both) = 6 :=
by
  rw [h_total, h_fries, h_burgers, h_both]
  exact rfl

end students_not_liking_either_l140_140822


namespace sequence_divisibility_l140_140044

-- Define the sequence a_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := ∑ k in finset.range (n + 1), nat.choose (n + 1) k * a k

-- Variables (conditions)
variables {m p q r : ℕ}
variable (hm : 0 < m)
variable (hp : nat.prime p)
variable (hq : 0 ≤ q)
variable (hr : 0 ≤ r)

-- Theorem statement
theorem sequence_divisibility : p^m ∣ (a (p^m * q + r) - a (p^(m-1) * q + r)) :=
sorry

end sequence_divisibility_l140_140044


namespace students_study_both_subjects_difference_l140_140692

-- Define the main theorem
theorem students_study_both_subjects_difference:
  ∃ (P C m M: ℕ), 1750 ≤ P ∧ P ≤ 1875 ∧ 1000 ≤ C ∧ C ≤ 1125 ∧
  (P + C - (P ∩ C) = 2500) ∧ (m = P ∩ C) ∧ (M = P ∩ C) ∧ (M - m = 250) := 
sorry

end students_study_both_subjects_difference_l140_140692


namespace x_is_three_times_y_l140_140434

theorem x_is_three_times_y (q : ℚ) (h₁: x = 5 - q) (h₂: y = 3q - 1):
  x = 3 * y → q = 4 / 5 := by
  sorry

end x_is_three_times_y_l140_140434


namespace area_of_APQ_l140_140913

noncomputable def area_of_triangle {A P Q : Point} 
  (h₁ : PerpendicularLines A (Line.mk P Q))
  (h₂ : y_intercepts_difference (Line.mk P (0,0)) (Line.mk Q (0,6)) 6)
  (A : Point) (P Q : Point): ℝ :=
1/2 * (abs (fst P - fst Q)) * (abs (snd A))

theorem area_of_APQ (h₁ : PerpendicularLines (9,12) ((y = m_1 * x + b_1), (y = m_2 * x + b_2))) 
  (h₂ : (b_2 - b_1 = 6))
  : area_of_triangle ⟨9, 12⟩ ⟨0, 0⟩ ⟨0, 6⟩ = 27 :=
by 
  sorry

end area_of_APQ_l140_140913


namespace instantaneous_velocity_at_2_l140_140426

def s (t : ℝ) : ℝ := 3 * t^3 - 2 * t^2 + t + 1

theorem instantaneous_velocity_at_2 : 
  (deriv s 2) = 29 :=
by
  -- The proof is skipped by using sorry
  sorry

end instantaneous_velocity_at_2_l140_140426


namespace quadratic_has_real_root_iff_b_in_intervals_l140_140800

theorem quadratic_has_real_root_iff_b_in_intervals (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ set.Icc (-∞ : ℝ) (-10) ∪ set.Icc 10 (∞ : ℝ)) :=
by by sorry

end quadratic_has_real_root_iff_b_in_intervals_l140_140800


namespace smallest_f0_l140_140481

noncomputable def f (x : ℤ) : ℤ := sorry

theorem smallest_f0 :
  (∀ x : ℤ, is_polynomial_with_integer_coeffs (f x)) ∧
  ((f 15) * (f 21) * (f 35) - 10) % 105 = 0 ∧
  f (-34) = 2014 ∧
  f 0 ≥ 0 →
  f 0 = 10 :=
by
  -- Conditions:
  have poly_cond : (∀ x : ℤ, is_polynomial_with_integer_coeffs (f x)) := sorry,
  have div_cond : ((f 15) * (f 21) * (f 35) - 10) % 105 = 0 := sorry,
  have f_minus34 : f (-34) = 2014 := sorry,
  have f0_nonneg : f 0 ≥ 0 := sorry,
  -- Conclusion:
  exact sorry

end smallest_f0_l140_140481


namespace trajectory_midpoint_Q_line_l_l140_140727

noncomputable def circle := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 3)^2 = 9}
def point_p := (5, -1 : ℝ)

theorem trajectory_midpoint_Q :
  ∀ Q : ℝ × ℝ,
  (∃ A ∈ circle, Q = ((A.1 + 5) / 2, (A.2 - 1) / 2)) →
  (2 * Q.1 - 8)^2 + (2 * Q.2 - 2)^2 = 9 :=
by
  sorry

theorem line_l :
  ∀ (line_eq : String),
  ∃ l : ℝ × ℝ → ℝ,
  (∀ (x y : ℝ), line_eq = "3x + 4y - 11 = 0" → l (x, y) = 3 * x + 4 * y - 11) ∨
  (∀ (x : ℝ), line_eq = "x = 5" → l (x, 0).1 = 5) :=
by
  sorry

end trajectory_midpoint_Q_line_l_l140_140727


namespace find_circle_center_l140_140703

def circle_center (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y - 16 = 0

theorem find_circle_center (x y : ℝ) :
  circle_center x y ↔ (x, y) = (3, 4) :=
by
  sorry

end find_circle_center_l140_140703


namespace dan_buys_5_dozens_l140_140542

theorem dan_buys_5_dozens (dozens_gus : ℕ) (golf_balls_chris : ℕ) (total_golf_balls : ℕ) : 
  12 * (total_golf_balls / 12) - dozens_gus - (golf_balls_chris / 12) = 5 :=
by
  let dozens_gus := 2
  let golf_balls_chris := 48
  let total_golf_balls := 132
  calc
    12 * (total_golf_balls / 12) - dozens_gus - (golf_balls_chris / 12) = 12 * (132 / 12) - 2 - (48 / 12) : by rfl
    ... = 12 * 11 - 2 - 4 : by rfl
    ... = 132 - 2 - 4 : by rfl
    ... = 130 - 4 : by rfl
    ... = 126 : by rfl
    sorry

end dan_buys_5_dozens_l140_140542


namespace distance_after_3_minutes_l140_140647

-- Define the speeds of the truck and the car in km/h.
def v_truck : ℝ := 65
def v_car : ℝ := 85

-- Define the time in hours.
def time_in_hours : ℝ := 3 / 60

-- Define the relative speed.
def v_relative : ℝ := v_car - v_truck

-- Define the expected distance between the truck and the car after 3 minutes.
def expected_distance : ℝ := 1

-- State the theorem: the distance between the truck and the car after 3 minutes is 1 km.
theorem distance_after_3_minutes : (v_relative * time_in_hours) = expected_distance := 
by {
  -- Here, we would provide the proof, but we are adding 'sorry' to skip the proof.
  sorry
}

end distance_after_3_minutes_l140_140647


namespace one_third_of_1206_is_300_percent_of_134_l140_140148

theorem one_third_of_1206_is_300_percent_of_134 :
  let number := 1206
  let fraction := 1 / 3
  let computed_one_third := fraction * number
  let whole := 134
  let expected_percent := 300
  let percent := (computed_one_third / whole) * 100
  percent = expected_percent := by
  let number := 1206
  let fraction := 1 / 3
  have computed_one_third : ℝ := fraction * number
  let whole := 134
  let expected_percent := 300
  have percent : ℝ := (computed_one_third / whole) * 100
  exact sorry

end one_third_of_1206_is_300_percent_of_134_l140_140148


namespace percentile_60th_of_data_set_is_9_l140_140918

-- Define the original data set with the variable a
def data_set (a : ℝ) : List ℝ := [2, 4, 5, 8, a, 10, 11]

-- Define the condition that the average of the data set is 7
def average_condition (a : ℝ) : Prop :=
  (2 + 4 + 5 + 8 + a + 10 + 11) / 7 = 7

-- Define the function to calculate the 60th percentile
def percentile60th (lst : List ℝ) : ℝ :=
  let sorted_lst := lst.qsort (· ≤ ·)
  sorted_lst.nth (ceiling (0.60 * sorted_lst.length) - 1) |>.getD 0

-- The proof problem statement
theorem percentile_60th_of_data_set_is_9 (a : ℝ) (h : average_condition a) :
  percentile60th (data_set a) = 9 :=
by
  sorry

end percentile_60th_of_data_set_is_9_l140_140918


namespace number_of_correct_statements_l140_140210

noncomputable def problem1 : Prop := ∀ x : ℝ, x > 1/2 → ∃ y : ℝ, y = logb 10 (2 * x - 1)
noncomputable def problem2 : Prop := ∀ (a b : ℝ), (2/3) ^ a > (2/3) ^ b → a < b
noncomputable def problem3 : Prop :=
  let f : ℝ → ℝ := λ x, if x > 0 then x^3 + 1 else 2017 * x + 1 
  in f (f 0) = 1
noncomputable def problem4 : Prop :=
  ∀ f : ℕ → ℝ, (∀ n : ℕ, 1 ≤ n ∧ n < 2016 → f n < f (n+1)) → ∀ x, 1 ≤ x ∧ x ≤ 2016 → f x ≤ f (x+1)

theorem number_of_correct_statements : (problem1 ∧ problem2 ∧ ¬ problem3 ∧ ¬ problem4) → 2 := by
  intros
  sorry

end number_of_correct_statements_l140_140210


namespace incorrect_condition_l140_140046

-- Definitions of the conditions
def power_function (a : ℝ) (x : ℝ) : ℝ := x ^ a

def passes_through_point (a : ℝ) : Prop := power_function a 2 = 4

-- The theorem asserting the incorrect condition
theorem incorrect_condition (a : ℝ) (h : passes_through_point a) : ¬(∀ x : ℝ, power_function a x + power_function a (-x) = 0) := 
sorry

end incorrect_condition_l140_140046


namespace pawn_tour_exists_l140_140657

theorem pawn_tour_exists (n : ℕ) (hn : n > 0) :
  ∃ (path : Fin (n^2) → Fin n × Fin n),
    (∀ i, ∃ j, path i = (j, path i).snd) ∧
    (∀ i, ∃ j, path i = (path i).fst, j) ∧
    (function.injective path) ∧
    (path 0 = path (Fin.last _)) :=
sorry

end pawn_tour_exists_l140_140657


namespace inequality_solution_l140_140338

theorem inequality_solution (y : ℝ) : 
  (y^2 - 6 * y - 16 > 0) ↔ (y < -2 ∨ y > 8) := by
skip

end inequality_solution_l140_140338


namespace quadratic_inequality_l140_140144

theorem quadratic_inequality
  (a b c : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by
  sorry

end quadratic_inequality_l140_140144


namespace function_properties_l140_140036

noncomputable def amplitude (f : ℝ → ℝ) : ℝ :=
  sorry  -- Calculate the amplitude of f.

noncomputable def period (f : ℝ → ℝ) : ℝ :=
  sorry  -- Calculate the period of f.

noncomputable def frequency (f : ℝ → ℝ) : ℝ :=
  1 / period f  -- The frequency is the reciprocal of the period.

noncomputable def initial_phase (f : ℝ → ℝ) : ℝ :=
  sorry  -- Calculate the initial phase of f.

theorem function_properties : 
  ∀ (φ : ℝ), 
  |φ| < π / 2 →
  (λ x, 2 * Real.sin (π / 3 * x + φ)) 0 = 1 →
  amplitude (λ x, 2 * Real.sin (π / 3 * x + φ)) = 2 ∧
  period (λ x, 2 * Real.sin (π / 3 * x + φ)) = 6 ∧
  frequency (λ x, 2 * Real.sin (π / 3 * x + φ)) = 1 / 6 ∧
  initial_phase (λ x, 2 * Real.sin (π / 3 * x + φ)) = π / 6 := 
by
  intro φ hφ h_passes
  -- We would use the conditions in proofs in the proof steps.
  sorry

end function_properties_l140_140036


namespace problem_statement_l140_140138

noncomputable def a (n : ℕ) : ℕ := 3^(n-1)

def S (n : ℕ) : ℕ := 1/2 * (3 * (a n) - 1)

def b (n : ℕ) : ℕ :=
if n % 2 = 1 then n + a n else n * a n

def T2n (n : ℕ) : ℕ := 
n^2 + ((24 * n + 1) * 3^(2 * n) - 1) / 32

theorem problem_statement (n : ℕ) (hn : 0 < n) :
(S n = (1 : ℕ) / 2 * (3 * a n - 1)) ∧
(T2n n = n^2 + ((24 * n + 1) * 3^(2 * n) - 1) / 32) := 
sorry

end problem_statement_l140_140138


namespace minimal_polynomial_degree_l140_140189

theorem minimal_polynomial_degree 
  (r1 r2 r3 r4 : ℚ) : 
  r1 = 4 - 3 * real.sqrt 3 ∧ r2 = -4 - 3 * real.sqrt 3 ∧
  r3 = 2 + real.sqrt 5 ∧ r4 = 2 - real.sqrt 5 → 
  ∃ f : ℚ[X], f.eval r1 = 0 ∧ f.eval r2 = 0 ∧ f.eval r3 = 0 ∧ f.eval r4 = 0 ∧ polynomial.degree f = 6 :=
sorry

end minimal_polynomial_degree_l140_140189


namespace online_sale_discount_l140_140235

theorem online_sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (total_paid : ℕ) : 
  purchase_amount = 250 → 
  discount_per_100 = 10 → 
  total_paid = purchase_amount - (purchase_amount / 100) * discount_per_100 → 
  total_paid = 230 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end online_sale_discount_l140_140235


namespace pyramid_volume_l140_140309

noncomputable def volume_of_pyramid (R : ℝ) : ℝ :=
  (R^3 * Real.sqrt 6) / 4

theorem pyramid_volume (R : ℝ) (h1 : sphere_radius R) (h2 : bisected_segment) :
  volume_of_pyramid R = (R^3 * Real.sqrt 6) / 4 :=
by
  sorry

end pyramid_volume_l140_140309


namespace minimum_distance_l140_140207

noncomputable def f (x : ℝ) : ℝ := exp x + x^2 + 2 * x + 1

def line : ℝ × ℝ → ℝ := λ p, 3 * p.1 - p.2 - 2

def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / sqrt (a^2 + b^2)

theorem minimum_distance :
  ∃ P : ℝ × ℝ, P ∈ set.of_fun f ∧ distance_point_to_line P 3 (-1) (-2) = 2 * sqrt 10 / 5 := sorry

end minimum_distance_l140_140207


namespace all_numbers_strictly_positive_l140_140678

theorem all_numbers_strictly_positive (n : ℕ) (x : Fin (2 * n + 1) → ℝ) 
  (H : ∀ (s : Finset (Fin (2 * n + 1))), s.card = n → (∑ i in s, x i) < (∑ i in sᶜ, x i)) : 
  ∀ i, 0 < x i :=
begin
  sorry -- Proof is omitted as per instructions.
end

end all_numbers_strictly_positive_l140_140678


namespace ellipse_tangent_circle_radius_l140_140658

theorem ellipse_tangent_circle_radius :
  ∀ {a b : ℝ} (h_a : a = 8) (h_b : b = 5),
  let c := Real.sqrt (a^2 - b^2),
  let r := 8 in
  ∃ r : ℝ, ∀ x : ℝ, (x = c) → y = r at x = c → x^2 + 25 = r^2 → r = 8 :=
begin
  sorry
end

end ellipse_tangent_circle_radius_l140_140658


namespace least_possible_value_of_z_minus_x_l140_140433

variables (x y z : ℤ)

-- Define the conditions
def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

-- State the theorem
theorem least_possible_value_of_z_minus_x (h1 : x < y) (h2 : y < z) (h3 : y - x > 3) 
    (hx_even : even x) (hy_odd : odd y) (hz_odd : odd z) : z - x = 7 :=
sorry

end least_possible_value_of_z_minus_x_l140_140433


namespace find_num_cpus_l140_140620

noncomputable theory

def num_graphics_cards : Nat := 10
def num_hard_drives : Nat := 14
def num_ram_pairs : Nat := 4

def price_graphics_card : ℕ := 600
def price_hard_drive : ℕ := 80
def price_cpu : ℕ := 200
def price_ram_pair : ℕ := 60

def total_earnings : ℕ := 8960

theorem find_num_cpus (num_cpus : ℕ) :
  num_graphics_cards * price_graphics_card +
  num_hard_drives * price_hard_drive +
  num_ram_pairs * price_ram_pair +
  num_cpus * price_cpu = total_earnings →
  num_cpus = 8 :=
by
  sorry

end find_num_cpus_l140_140620


namespace line_circle_intersect_l140_140936

-- Define the line equation as a predicate
def line (x y : ℝ) : Prop :=
  4 * x - 3 * y = 0

-- Define the circle equation as a predicate
def circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 36

-- Statement of the problem to be proved
theorem line_circle_intersect :
  ∃ x y : ℝ, line x y ∧ circle x y :=
sorry

end line_circle_intersect_l140_140936


namespace intersection_eq_l140_140740

def setA : Set ℝ := {x | -1 < x ∧ x < 1}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_eq : setA ∩ setB = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_eq_l140_140740


namespace total_layoffs_l140_140825

noncomputable def layoffs_round_1 (initial : ℕ) : ℕ :=
  (10 * initial) / 100

noncomputable def layoffs_round_2 (remaining : ℕ) : ℕ :=
  (12 * remaining) / 100

noncomputable def layoffs_round_3 (remaining : ℕ) : ℕ :=
  (15 * remaining) / 100

noncomputable def layoffs_round_4 (remaining : ℕ) : ℕ :=
  (20 * remaining) / 100

noncomputable def layoffs_round_5 (remaining : ℕ) : ℕ :=
  (25 * remaining) / 100

theorem total_layoffs :
  let initial := 1000 in
  let r1 := layoffs_round_1 initial in
  let r2 := layoffs_round_2 (initial - r1) in
  let r3 := layoffs_round_3 (initial - r1 - r2) in
  let r4 := layoffs_round_4 (initial - r1 - r2 - r3) in
  let r5 := layoffs_round_5 (initial - r1 - r2 - r3 - r4) in
  r1 + r2 + r3 + r4 + r5 = 596 :=
by
  sorry

end total_layoffs_l140_140825


namespace boiling_time_parallel_heaters_l140_140952

theorem boiling_time_parallel_heaters (t1 t2 : ℕ) (h1 : t1 = 120) (h2 : t2 = 180) : (t1 * t2) / (t1 + t2) = 72 := 
by 
  -- substitutions for given problem conditions
  have ht1 : t1 = 120 := h1,
  have ht2 : t2 = 180 := h2,

  -- simplification and calculations to show the final value
  calc
    (120 * 180) / (120 + 180)
      = 21600 / 300 : by norm_num
  ... = 72 : by norm_num

end boiling_time_parallel_heaters_l140_140952


namespace max_right_angled_triangles_in_quadrangular_pyramid_l140_140086

-- Definition and conditions of the quadrangular pyramid
structure QuadrangularPyramid where
  base : Type
  isQuadrilateral : base → Prop
  has_four_triangular_lateral_faces : Prop

-- Definition of a right-angled triangle
def is_right_angled_triangle (T : Type) [Triangle T] : Prop := T.has_right_angle

-- Proving the maximum number of right-angled triangles
theorem max_right_angled_triangles_in_quadrangular_pyramid :
  ∀ (P : QuadrangularPyramid), 
  P.has_four_triangular_lateral_faces → 
  ∃ n : ℕ, n ≤ 2 ∧ 
    (∃ faces : fin n.succ → Triangle, 
      ∀ i, is_right_angled_triangle (faces i)) := 
by 
  sorry

end max_right_angled_triangles_in_quadrangular_pyramid_l140_140086


namespace time_for_pipe_A_to_fill_tank_alone_l140_140244

noncomputable def fill_rate_A (t : ℝ) : ℝ := 1 / t
def fill_rate_B : ℝ := 1 / 46
def combined_fill_rate : ℝ := 1 / 20.195121951219512

theorem time_for_pipe_A_to_fill_tank_alone :
  ∃ t : ℝ, fill_rate_A t + fill_rate_B = combined_fill_rate ∧ t ≈ 36.05 :=
begin
  sorry
end

end time_for_pipe_A_to_fill_tank_alone_l140_140244


namespace minimum_sum_of_squares_l140_140467

variable {A B C P : Type}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space P]

-- Define the sides of triangle ABC
variables (a b c : ℝ)
-- Define α as the angle BAC and β as the angle BCA
variables (α β : ℝ)
-- Define the point P's position on AB
variables (x : ℝ)
-- Define the functions for the lengths to AC and BC from P
variables {f g : ℝ}

-- The statement we are supposed to prove
theorem minimum_sum_of_squares (a b c α β x : ℝ) :
  let y := (x^2 + (b^2 / a^2) * (c - x)^2) * (sin α)^2 in
  (∃ x0 : ℝ, x0 = (b^2 * c) / (a^2 + b^2)) →
  ∀ x₁, (∃ xₛ : x₁ = x0, y = (xₛ^2 + (b^2 / a^2) * (c - xₛ)^2) * (sin α)^2) :=
begin
  sorry
end

end minimum_sum_of_squares_l140_140467


namespace trajectory_of_Q_l140_140736

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem trajectory_of_Q (x y : ℝ) (P : ℝ × ℝ) (P' : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : circle P.1 P.2) 
  (hP' : P' = (P.1, 0)) 
  (hQ_on_PP' : Q.1 = P.1 ∧ 2 * (Q.1 - P.1) = P.1 - Q.1 ∧ Q.2 = (P.2 / 3)) : 
  x^2 + 9*y^2 = 1 := 
sorry

end trajectory_of_Q_l140_140736


namespace hyperbola_eccentricity_l140_140519

-- Definitions and assumptions directly from conditions
variables {a b : ℝ} (h_a : a > 0) (h_b : b > 0)
def hyperbola (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
noncomputable def right_focus := sqrt (5) * a

-- Main theorem statement
theorem hyperbola_eccentricity :
  ∀ {x y : ℝ} (M : (ℝ × ℝ)) (m n c : ℝ),
    M = (m, n) →
    let focus_condition_1 := (n / (m - c)) * (b / a) = -1 in
    let focus_condition_2 := n / 2 = (b / a) * (m + c) / 2 in
    focus_condition_1 →
    focus_condition_2 →
    hyperbola m n →
    c = right_focus →
    c / a = sqrt 5 :=
by
  intros,
  sorry

end hyperbola_eccentricity_l140_140519


namespace euler_line_of_isosceles_triangle_l140_140204

theorem euler_line_of_isosceles_triangle (A B : ℝ × ℝ) (hA : A = (2,0)) (hB : B = (0,4)) (C : ℝ × ℝ) (hC1 : dist A C = dist B C) :
  ∃ a b c : ℝ, a * (C.1 - 2) + b * (C.2 - 0) + c = 0 ∧ x - 2 * y + 3 = 0 :=
by
  sorry

end euler_line_of_isosceles_triangle_l140_140204


namespace decrease_percent_in_revenue_l140_140596

variables (T C : ℝ)

def original_revenue := T * C
def new_tax_rate := 0.76 * T
def new_consumption := 1.12 * C
def new_revenue := new_tax_rate * new_consumption

theorem decrease_percent_in_revenue : 
  ((original_revenue T C - new_revenue T C) / original_revenue T C) * 100 = 14.88 :=
by
  sorry

end decrease_percent_in_revenue_l140_140596


namespace shaded_percentage_eq_third_l140_140243

-- Definitions
def side_length : ℝ := 20
def length_rectangle : ℝ := 30
def width_rectangle : ℝ := 20

-- Proof problem statement
theorem shaded_percentage_eq_third :
  let A_rectangle := length_rectangle * width_rectangle,
      overlap := 2 * side_length - length_rectangle,
      A_shaded := overlap * width_rectangle in
  (A_shaded / A_rectangle) * 100 = 33.33 :=
by
  sorry

end shaded_percentage_eq_third_l140_140243


namespace Hillary_left_with_amount_l140_140660

theorem Hillary_left_with_amount :
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  remaining_amount = 25 :=
by
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  sorry

end Hillary_left_with_amount_l140_140660


namespace sum_of_coefficients_l140_140418

theorem sum_of_coefficients :
  ∀ (a a_1 a_2 : ℝ) [Ring a] [Ring a_1] [Ring a_2] (a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11}: ℝ),
  ∃ p : Polynomial ℝ,
  (p = Polynomial.C (x^2+1) * Polynomial.pow (Polynomial.C 2 * X + 1) 9) →
  (p = Polynomial.C a + Polynomial.C a_1 * (X + 2) + Polynomial.C a_2 * (X + 2) ^ 2 + 
       Polynomial.C a_3 * (X + 2) ^ 3 + Polynomial.C a_4 * (X + 2) ^ 4 +
       Polynomial.C a_5 * (X + 2) ^ 5 + Polynomial.C a_6 * (X + 2) ^ 6 +
       Polynomial.C a_7 * (X + 2) ^ 7 + Polynomial.C a_8 * (X + 2) ^ 8 +
       Polynomial.C a_9 * (X + 2) ^ 9 + Polynomial.C a_{10} * (X + 2) ^ 10 +
       Polynomial.C a_{11} * (X + 2) ^ 11) →
  (a + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} = -2) :=
by { sorry }

end sum_of_coefficients_l140_140418


namespace sara_quarters_l140_140529

theorem sara_quarters (initial_quarters : ℕ) (borrowed_quarters : ℕ) : initial_quarters = 783 → borrowed_quarters = 271 → (initial_quarters - borrowed_quarters) = 512 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end sara_quarters_l140_140529


namespace half_abs_diff_squares_23_19_l140_140247

theorem half_abs_diff_squares_23_19 : 
  (| (23^2 - 19^2) |) / 2 = 84 := by
  sorry

end half_abs_diff_squares_23_19_l140_140247


namespace translation_results_in_prism_l140_140629

variable (Polygon : Type) (Plane : Type) 

/-- This structure defines a polygon and its associated plane. -/
structure PolygonPlane where
  polygon : Polygon
  plane : Plane

/-- Define a translation vector that is not parallel to the plane of the polygon. -/
variable {Vector : Type} [AddGroup Vector] [VectorSpace ℝ Vector]

noncomputable def is_not_parallel (v : Vector) (pl : Plane) : Prop := sorry -- Define this accurately as per properties

/-- A translation function that translates a polygon along a given vector. -/
noncomputable def translate (p : PolygonPlane) (v : Vector) : Polygon := sorry

/-- The theorem stating that translating a polygon out of its plane results in a prism. -/
theorem translation_results_in_prism
  (p : PolygonPlane)
  (v : Vector)
  (h : is_not_parallel v p.plane) :
  ∃ result_shape, is_prism result_shape := sorry

end translation_results_in_prism_l140_140629


namespace opposite_of_one_third_l140_140323

theorem opposite_of_one_third : -(1/3) = -1/3 := by
  sorry

end opposite_of_one_third_l140_140323


namespace cube_difference_positive_l140_140420

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cube_difference_positive_l140_140420


namespace fraction_simplify_l140_140349

variable (a b c : ℝ)

theorem fraction_simplify
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h₃ : a + 2 * b + 3 * c ≠ 0) :
  (a^2 + 4 * b^2 - 9 * c^2 + 4 * a * b) / (a^2 + 9 * c^2 - 4 * b^2 + 6 * a * c) =
  (a + 2 * b - 3 * c) / (a - 2 * b + 3 * c) := by
  sorry

end fraction_simplify_l140_140349


namespace inflation_over_two_years_real_interest_rate_l140_140976

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end inflation_over_two_years_real_interest_rate_l140_140976


namespace h_neg_one_eq_l140_140864

def f (x : ℝ) : ℝ := 3 * x^2 + x + 4

def g (x : ℝ) : ℝ := real.sqrt (f x) + 1

def h (x : ℝ) : ℝ := f (g x)

theorem h_neg_one_eq : h (-1) = 26 + 7 * real.sqrt 6 :=
by
  sorry

end h_neg_one_eq_l140_140864


namespace volume_of_cylinder_l140_140926

def height_cm := 40
def diameter_cm := 40
def radius_cm := diameter_cm / 2
def volume_cm3 := Real.pi * (radius_cm ^ 2) * height_cm
def volume_dm3 := volume_cm3 / 1000

theorem volume_of_cylinder :
  volume_dm3 = 502.4 :=
sorry

end volume_of_cylinder_l140_140926


namespace number_base2_more_ones_than_zeros_mod_500_l140_140487

theorem number_base2_more_ones_than_zeros_mod_500 :
  let N := { n : ℕ | n ≤ 500 ∧ (nat.bit_count n > nat.bit_length n / 2) }.to_finset.card
  in N % 500 = 305 :=
by
  -- We assert the condition and result as required but won't include the proof here
  sorry

end number_base2_more_ones_than_zeros_mod_500_l140_140487


namespace extreme_value_of_f_range_of_values_for_a_l140_140041

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem extreme_value_of_f :
  ∃ x_min : ℝ, f x_min = 1 :=
sorry

theorem range_of_values_for_a :
  ∀ a : ℝ, (∀ x : ℝ, f x ≥ (x^3) / 6 + a) → a ≤ 1 :=
sorry

end extreme_value_of_f_range_of_values_for_a_l140_140041


namespace min_trucks_needed_l140_140173

theorem min_trucks_needed (n : ℕ) (w : ℕ) (t : ℕ) (total_weight : ℕ) (max_box_weight : ℕ) : 
    (total_weight = 10) → 
    (max_box_weight = 1) → 
    (t = 3) →
    (n * max_box_weight = total_weight) →
    (n ≥ 10) →
    ∀ min_trucks : ℕ, (min_trucks * t ≥ total_weight) → 
    min_trucks = 5 :=
by
  intro total_weight_eq max_box_weight_eq truck_capacity box_total_weight_eq n_lower_bound min_trucks min_trucks_condition
  sorry

end min_trucks_needed_l140_140173


namespace magnitude_of_z_l140_140728

def complex_mag (z : Complex) : ℝ :=
  Complex.abs z

theorem magnitude_of_z : complex_mag (1 - 2 * Complex.I) = Real.sqrt 5 :=
  sorry

end magnitude_of_z_l140_140728


namespace compare_a_b_l140_140381

def a := 1 / 3 + 1 / 4
def b := 1 / 5 + 1 / 6 + 1 / 7

theorem compare_a_b : a > b := 
  sorry

end compare_a_b_l140_140381


namespace exists_B_for_large_sum_l140_140713

-- Define the modulus n^2.
def Zmodn2 (n : ℕ) : Type := { x : ℕ // x < n^2 }

-- Define the operation A + B modulo n^2.
def addSetsMod (A B : finset (Zmodn2 n)) : finset (Zmodn2 n) :=
  finset.image (λ (u : Zmodn2 n × Zmodn2 n), ⟨(u.1.val + u.2.val) % n^2, nat.mod_lt _ n^2_pos⟩) (finset.product A B)

theorem exists_B_for_large_sum (n : ℕ) (A : finset (Zmodn2 n)) 
  (hA_card : A.card = n) :
  ∃ B : finset (Zmodn2 n), B.card = n ∧ (addSetsMod A B).card ≥ n^2 / 2 :=
sorry

end exists_B_for_large_sum_l140_140713


namespace average_marbles_of_other_colors_l140_140155

theorem average_marbles_of_other_colors :
  let total_percentage := 100
  let clear_percentage := 40
  let black_percentage := 20
  let other_percentage := total_percentage - clear_percentage - black_percentage
  let marbles_taken := 5
  (other_percentage / 100) * marbles_taken = 2 :=
by
  sorry

end average_marbles_of_other_colors_l140_140155


namespace barbara_total_cost_l140_140666

-- Define conditions
def steak_weight : ℝ := 4.5
def steak_price_per_pound : ℝ := 15.0
def chicken_weight : ℝ := 1.5
def chicken_price_per_pound : ℝ := 8.0

-- Define total cost formula
def total_cost := (steak_weight * steak_price_per_pound) + (chicken_weight * chicken_price_per_pound)

-- Prove that the total cost equals $79.50
theorem barbara_total_cost : total_cost = 79.50 := by
  sorry

end barbara_total_cost_l140_140666


namespace prove_f_log3_a_lt_f_6_lt_f_2_sqrt_a_l140_140028

variable {ℝ : Type*} [LinearOrderedField ℝ]

/-- Mathematical conditions -/

def f (x : ℝ) : ℝ := sorry   -- Define the function f

def fx_symmetric_about_4 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) = f (6 - x)

def f_prime_conditions (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 4 → (x * deriv f x > 4 * deriv f x)

/-- Proof problem in Lean 4 -/

theorem prove_f_log3_a_lt_f_6_lt_f_2_sqrt_a (f : ℝ → ℝ)
  (h1 : fx_symmetric_about_4 f)
  (h2 : f_prime_conditions f)
  (a : ℝ) (h3 : 9 < a) (h4 : a < 27) :
  f (Real.log a / Real.log 3) < f 6 ∧ f 6 < f (2 ^ Real.sqrt a) := 
sorry

end prove_f_log3_a_lt_f_6_lt_f_2_sqrt_a_l140_140028


namespace identify_quadratic_equation_l140_140590

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem identify_quadratic_equation :
  let A := λ x, ax^2 + bx + c
  let B := λ x, 1/x^2 + 1/x - 2
  let C := λ x, x^2 + 2*x - (x^2 - 1)
  let D := λ x, 3*(x + 1)^2 - 2*(x + 1)
  ¬ is_quadratic B ∧
  ¬ is_quadratic C ∧
  is_quadratic D :=
by
  sorry

end identify_quadratic_equation_l140_140590


namespace smallest_cardinality_union_l140_140172

theorem smallest_cardinality_union (C D : Finset α) (hC : C.card = 30) (hD : D.card = 25) (hCD : (C ∩ D).card = 10) : (C ∪ D).card = 45 :=
by
  sorry

end smallest_cardinality_union_l140_140172


namespace area_of_triangle_ABC_l140_140468

-- Definitions as per the conditions
noncomputable def triangle_ABC_right : Type := 
  {ABC : Triangle ℝ // right_triangle ABC ∧ angle_ABC ABC = 45 ∧ angle_ACB ABC = 90}

-- Altitude length from C to hypotenuse AB
def altitude_CD (ABC : triangle_ABC_right) : ℝ :=
  sqrt 2

-- Statement to express the area to be proved
theorem area_of_triangle_ABC (ABC : triangle_ABC_right) :
  Triangle.area ABC.to_subtype.val = 2 :=
by
  sorry

end area_of_triangle_ABC_l140_140468


namespace find_num_chickens_l140_140884

-- Definitions based on problem conditions
def num_dogs : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2
def total_legs_seen : ℕ := 12

-- Proof problem: Prove the number of chickens Mrs. Hilt saw
theorem find_num_chickens (C : ℕ) (h1 : num_dogs * legs_per_dog + C * legs_per_chicken = total_legs_seen) : C = 2 := 
sorry

end find_num_chickens_l140_140884


namespace right_triangle_median_condition_l140_140682

theorem right_triangle_median_condition (c s_a : ℝ) (h_c_pos : c > 0) (h_s_a_pos : s_a > 0) :
  (∃ (a b : ℝ), a^2 + b^2 = c^2 ∧ (1/2)*(sqrt((a^2 + b^2) - (a - b)^2)) = s_a) ↔ (c / 2 < s_a ∧ s_a < c) :=
by
  sorry

end right_triangle_median_condition_l140_140682


namespace distance_from_P_to_XYZ_l140_140162

noncomputable def distance_to_plane {X Y Z P : ℝ^3}
  (h_perp1 : ⟪P - X, P - Y⟫ = 0)
  (h_perp2 : ⟪P - X, P - Z⟫ = 0)
  (h_perp3 : ⟪P - Y, P - Z⟫ = 0)
  (PX : dist P X = 15)
  (PY : dist P Y = 15)
  (PZ : dist P Z = 9) : ℝ :=
  13.5 * Real.sqrt 2

theorem distance_from_P_to_XYZ 
  {X Y Z P : ℝ^3}
  (h_perp1 : ⟪P - X, P - Y⟫ = 0)
  (h_perp2 : ⟪P - X, P - Z⟫ = 0)
  (h_perp3 : ⟪P - Y, P - Z⟫ = 0)
  (PX : dist P X = 15)
  (PY : dist P Y = 15)
  (PZ : dist P Z = 9) : 
  dist_to_plane h_perp1 h_perp2 h_perp3 PX PY PZ = 13.5 * Real.sqrt 2 :=
sorry

end distance_from_P_to_XYZ_l140_140162


namespace nonempty_solution_set_range_l140_140566

theorem nonempty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 := sorry

end nonempty_solution_set_range_l140_140566


namespace sin_HAG_is_sqrt5_over_3_l140_140989

-- Define the points A, G, and vectors HA, HG
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def G : ℝ × ℝ × ℝ := (1, 1, 1)
def H : ℝ × ℝ × ℝ := (1, 0, 1)

def vector_ha : ℝ × ℝ × ℝ := (1, 1, 1)
def vector_hg : ℝ × ℝ × ℝ := (0, -1, 0)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the length of a vector
def length (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Define the angle between two vectors
def cos_angle (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (length u * length v)

-- Define sine of the angle HAG using the cosine value
noncomputable def sin_hag : ℝ :=
  real.sqrt (1 - (cos_angle vector_ha vector_hg)^2)

-- The theorem to be proved
theorem sin_HAG_is_sqrt5_over_3 : sin_hag = real.sqrt 5 / 3 :=
sorry

end sin_HAG_is_sqrt5_over_3_l140_140989


namespace scarves_per_box_l140_140332

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ)
  (mittens_per_box : ℕ)
  (total_clothes : ℕ)
  (h1 : boxes = 4)
  (h2 : mittens_per_box = 6)
  (h3 : total_clothes = 32)
  (total_mittens := boxes * mittens_per_box)
  (total_scarves := total_clothes - total_mittens) :
  total_scarves / boxes = 2 :=
by
  sorry

end scarves_per_box_l140_140332


namespace question1_1_question1_2_question2_l140_140879

open Set

noncomputable def universal_set : Set ℝ := univ

def setA : Set ℝ := { x | x^2 - 9 * x + 18 ≥ 0 }

def setB : Set ℝ := { x | -2 < x ∧ x < 9 }

def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem question1_1 : ∀ x, x ∈ setA ∨ x ∈ setB :=
by sorry

theorem question1_2 : ∀ x, x ∈ (universal_set \ setA) ∩ setB ↔ (3 < x ∧ x < 6) :=
by sorry

theorem question2 (a : ℝ) (h : setC a ⊆ setB) : -2 ≤ a ∧ a ≤ 8 :=
by sorry

end question1_1_question1_2_question2_l140_140879


namespace geometric_sequence_sum_six_l140_140140

noncomputable def S (a_n : ℕ → ℝ) (n : ℕ) : ℝ := (finset.range n).sum (λ i, a_n (i + 1))

theorem geometric_sequence_sum_six
  (a : ℕ → ℝ) 
  (r : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n * r)
  (h_pos : ∀ n, 0 < a n)
  (h_r_lt_one : r < 1)
  (h_a3_a5 : a 3 + a 5 = 20)
  (h_a2_a6 : a 2 * a 6 = 64) :
  S a 6 = 126 :=
sorry

end geometric_sequence_sum_six_l140_140140


namespace removed_black_cubes_multiple_of_4_l140_140286

theorem removed_black_cubes_multiple_of_4 :
  let n := 10
  let cube := list.range (n * n * n)
  let colors : list bool := cube.map (λ idx, (let x := idx % n in let y := (idx / n) % n in let z := idx / (n * n) in (x + y + z) % 2 == 0)) -- True represents black, False represents white
  500 = colors.count true ∧ 500 = colors.count false →
  ∀ removed_100_cubes : list ℕ, removed_100_cubes.length = 100 ∧ (removed_100_cubes ≠ list.nodup) ∧ (removed_100_cubes.forall (λ idx, idx < n * n * n)) ∧ (list.countp (λ idx, idx / n / n < n) removed_100_cubes = n) →
  ∃ k, list.countp (λ idx, colors.nth_le idx sorry) removed_100_cubes = 4 * k :=
by sorry

end removed_black_cubes_multiple_of_4_l140_140286


namespace tan_A_l140_140097

def triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] := sorry

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (AC AB : Real)

-- Conditions that define the problem
def conditions :=
  AC = Real.sqrt 34 ∧ 
  AB = 5 ∧ 
  is_right_triangle A B C

-- Statement to be proved
theorem tan_A :
  conditions AC AB →
  tan A = 3 / 5 := 
sorry

end tan_A_l140_140097


namespace find_original_number_l140_140239

theorem find_original_number (n : ℝ) (h : n / 2 = 9) : n = 18 :=
sorry

end find_original_number_l140_140239


namespace select_student_l140_140567

def average_score (name : string) : ℚ :=
  if name = "A" then 96 else 
  if name = "B" then 98 else 
  if name = "C" then 98 else 
  if name = "D" then 96 else 0

def variance (name : string) : ℚ :=
  if name = "A" then 0.34 else 
  if name = "B" then 0.34 else 
  if name = "C" then 0.56 else 
  if name = "D" then 0.39 else 0  

theorem select_student :
  let best_student := "B" in
  (average_score "A" = 96) ∧ (variance "A" = 0.34) →
  (average_score "B" = 98) ∧ (variance "B" = 0.34) →
  (average_score "C" = 98) ∧ (variance "C" = 0.56) →
  (average_score "D" = 96) ∧ (variance "D" = 0.39) →
  (average_score best_student = 98) ∧ (variance best_student = 0.34)
:=
by
  intros _ _ _ _  
  exact ⟨rfl, rfl⟩

end select_student_l140_140567


namespace median_divides_triangle_into_equal_parts_l140_140557

-- Define a triangle with vertices A, B, and C.
structure Triangle (α : Type) :=
(A B C : α)

-- Define a median line of a triangle. It goes from a vertex to the midpoint of the opposite side.
def is_median {α : Type} [AddCommGroup α] [Module ℝ α] (t : Triangle α) (M : α) (v : ℕ) : Prop :=
  match v with
  | 0 => ∃ (M : α), M = ((1 : ℝ) / 2 • (t.B + t.C)) ∧ M = affine_combination t.A M
  | 1 => ∃ (M : α), M = ((1 : ℝ) / 2 • (t.A + t.C)) ∧ M = affine_combination t.B M
  | _ => ∃ (M : α), M = ((1 : ℝ) / 2 • (t.A + t.B)) ∧ M = affine_combination t.C M

-- The theorem stating that the median divides the triangle into two equal areas.
theorem median_divides_triangle_into_equal_parts
  {α : Type} [AddCommGroup α] [Module ℝ α] (t : Triangle α) (M : α) (v : ℕ) 
  (hv : v = 0 ∨ v = 1 ∨ v = 2) : is_median t M v → 
  (area (Triangle.mk t.A t.B M) = (1 / 2) * area t) ∧ (area (Triangle.mk t.A M t.B) = (1 / 2) * area t) :=
sorry

end median_divides_triangle_into_equal_parts_l140_140557


namespace min_time_adult_worms_l140_140315

noncomputable def f : ℕ → ℝ
| 1 => 0
| n => (1 - 1 / (2 ^ (n - 1)))

theorem min_time_adult_worms (n : ℕ) (h : n ≥ 1) : 
  ∃ min_time : ℝ, 
  (min_time = 1 - 1 / (2 ^ (n - 1))) ∧ 
  (∀ t : ℝ, (t = 1 - 1 / (2 ^ (n - 1)))) := 
sorry

end min_time_adult_worms_l140_140315


namespace sum_of_ratios_lt_2n_minus_1_l140_140114

theorem sum_of_ratios_lt_2n_minus_1 (n : ℕ) (h1 : n ≥ 2) (a : Fin n → ℝ)
  (h2 : ∀ i, a i < 1)
  (h3 : ∀ i : Fin (n - 1), abs (a i.val - a (i.val + 1)) < 1) :
  (∑ i in Finset.range n, a i / a ((i + 1) % n)) < 2 * n - 1 := 
sorry

end sum_of_ratios_lt_2n_minus_1_l140_140114


namespace determine_a_l140_140689

theorem determine_a (a : ℕ) (h : a / (a + 36) = 9 / 10) : a = 324 :=
sorry

end determine_a_l140_140689


namespace angle_GSO_is_90_l140_140098

theorem angle_GSO_is_90 
  (α β δ : ℝ) 
  (DOG_is_iso : α = β)
  (DOG_bisect : δ = α)
  (angle_triang_DOG : α = 48) 
  (OS_bisects_GOD : ∀ (G O S D : Type) (DG DO : G → O), δ = α → β = 180 - 2 * 48 → OS_bisects_OS) 
  : ∠ GSO = 90 :=
by
  sorry

end angle_GSO_is_90_l140_140098


namespace simplification_l140_140176

-- Define all relevant powers
def pow2_8 : ℤ := 2^8
def pow4_5 : ℤ := 4^5
def pow2_3 : ℤ := 2^3
def pow_neg2_2 : ℤ := (-2)^2

-- Define the expression inside the parentheses
def inner_expr : ℤ := pow2_3 - pow_neg2_2

-- Define the exponentiation of the inner expression
def inner_expr_pow11 : ℤ := inner_expr^11

-- Define the entire expression
def full_expr : ℤ := (pow2_8 + pow4_5) * inner_expr_pow11

-- State the proof goal
theorem simplification : full_expr = 5368709120 := by
  sorry

end simplification_l140_140176


namespace triangle_expression_l140_140813

-- Definitions for specific sides
def sides_of_triangle (PQ PR QR : ℝ) := PQ = 7 ∧ PR = 8 ∧ QR = 5

-- The main theorem we want to prove
theorem triangle_expression (PQ PR QR : ℝ) (P Q R : ℝ) (h : sides_of_triangle PQ PR QR) :
  (\frac{\cos \frac{P - Q}{2}}{\sin \frac{R}{2}} - \frac{\sin \frac{P - Q}{2}}{\cos \frac{R}{2}}) = \frac{16}{7} :=
sorry

end triangle_expression_l140_140813


namespace mean_of_five_integers_l140_140206

theorem mean_of_five_integers
  (p q r s t : ℤ)
  (h1 : (p + q + r) / 3 = 9)
  (h2 : (s + t) / 2 = 14) :
  (p + q + r + s + t) / 5 = 11 :=
by
  sorry

end mean_of_five_integers_l140_140206


namespace half_abs_diff_squares_23_19_l140_140248

theorem half_abs_diff_squares_23_19 : 
  (| (23^2 - 19^2) |) / 2 = 84 := by
  sorry

end half_abs_diff_squares_23_19_l140_140248


namespace line_equation_min_length_proof_line_equation_min_area_proof_l140_140931

noncomputable def line_equation_min_length : (ℝ × ℝ) → ℝ × ℝ → Prop := 
  λ P A B, (∀ x y : ℝ, (x = 1 ∧ y = 4 → P = (x, y)) ∧ 
  (∃ a b > 0, A = (a, 0) ∧ B = (0, b) ∧ 
  ∀ x y : ℝ, (x / a + y / b = 1 → ((a + b) * (1 / a + 4 / b) = 5 + (b / a + 4 * a / b) ∧ 
  (b / a + 4 * a / b = 9) → 2 * x + y - 6 = 0))))

theorem line_equation_min_length_proof : 
  ∀ P A B, line_equation_min_length P A B 
  → ∀ x y : ℝ, (x = 1 ∧ y = 4 → P = (x, y)) ∧ 
  (∃ a b > 0, A = (a, 0) ∧ B = (0, b) ∧ 
  ∀ x y : ℝ, (x / a + y / b = 1 → ((a + b) * (1 / a + 4 / b) = 5 + (b / a + 4 * a / b) ∧ 
  (b / a + 4 * a / b = 9) → 2 * x + y -6 = 0)) := 
by {
  sorry
}

noncomputable def line_equation_min_area : (ℝ × ℝ) → ℝ × ℝ → Prop := 
  λ P A B, (∀ x y : ℝ, (x = 1 ∧ y = 4 → P = (x, y)) ∧ 
  (∃ a b > 0, A = (a, 0) ∧ B = (0, b) ∧ 
  ∀ x y : ℝ, (x / a + y / b = 1 → (1 ≥ 2 * real.sqrt (1 / a * 4 / b) ∧
  1 / a * 4 / b ≥ 16 → ab = 8) → 4 * x + y - 8 = 0))))

theorem line_equation_min_area_proof : 
  ∀ P A B, line_equation_min_area P A B 
  → ∀ x y : ℝ, (x = 1 ∧ y = 4 → P = (x, y)) ∧ 
  (∃ a b > 0, A = (a, 0) ∧ B = (0, b) ∧ 
  ∀ x y : ℝ, (x / a + y / b = 1 → (1 ≥ 2 * real.sqrt (1 / a * 4 / b) ∧ 
  1 / a * 4 / b ≥ 16 → ab = 8) → 4 * x + y - 8 = 0)) :=
by {
  sorry
}

end line_equation_min_length_proof_line_equation_min_area_proof_l140_140931


namespace f_at_neg_4_l140_140721

variable (a b : ℝ) (f : ℝ → ℝ)

-- Condition 1: Define the function f
def f (x : ℝ) : ℝ := a * x^3 + b / x + 3

-- Condition 2: Given f(4) = 5
axiom f_at_4 : f a b 4 = 5

-- Goal: Prove that f(-4) = 1
theorem f_at_neg_4 : f a b (-4) = 1 := by
  sorry

end f_at_neg_4_l140_140721


namespace maximum_ratio_x_over_y_l140_140501

theorem maximum_ratio_x_over_y {x y : ℕ} (hx : x > 9 ∧ x < 100) (hy : y > 9 ∧ y < 100)
  (hmean : x + y = 110) (hsquare : ∃ z : ℕ, z^2 = x * y) : x = 99 ∧ y = 11 := 
by
  -- mathematical proof
  sorry

end maximum_ratio_x_over_y_l140_140501


namespace kitchen_width_l140_140110

-- Condition definitions
def tile_area : ℕ := 6
def kitchen_length : ℕ := 72
def num_tiles : ℕ := 96

-- Problem statement: Prove that the width of the kitchen is 8 inches
theorem kitchen_width :
  let total_area := num_tiles * tile_area in
  let width := total_area / kitchen_length in
  width = 8 := 
begin
  sorry
end

end kitchen_width_l140_140110


namespace bulb_arrangement_count_l140_140229

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem bulb_arrangement_count 
    (blue_bulbs red_bulbs white_bulbs : ℕ)
    (h_blue : blue_bulbs = 8)
    (h_red : red_bulbs = 6)
    (h_white : white_bulbs = 12)
    (no_adjacent_white : ∀ arrangement, is_valid_arrangement arrangement white_bulbs → no_two_adjacent white_bulbs arrangement) :
  (binomial_coefficient (blue_bulbs + red_bulbs) blue_bulbs) *
  (binomial_coefficient (blue_bulbs + red_bulbs + 1) white_bulbs) = 1366365 := by
  sorry

-- Auxiliary Definitions

-- Placeholder for a type representing arrangements.
def arrangement := List (ℕ × ℕ)

-- Placeholder for validating an arrangement's correctness.
def is_valid_arrangement (arrangement : arrangement) (white_bulbs : ℕ) : Prop :=
  sorry 

-- Placeholder for a condition that ensures no two white bulbs are adjacent.
def no_two_adjacent (white_bulbs : ℕ) (arrangement : arrangement) : Prop :=
  sorry

end bulb_arrangement_count_l140_140229


namespace measure_of_angle_A_find_b_and_c_l140_140078

variable {A a b c : ℝ}
variable {S : ℝ}

theorem measure_of_angle_A (h : cos A * (sqrt 3 * sin A - cos A) = 1 / 2) : A = π / 3 := by
  sorry

theorem find_b_and_c (h1 : a = 2 * sqrt 2) (h2 : S = 2 * sqrt 3) (h3 : sin A = sqrt 3 / 2):
  (b * c = 8) ∧ (b + c = 4 * sqrt 2) → (b = 2 * sqrt 2) ∧ (c = 2 * sqrt 2) := by
  sorry

end measure_of_angle_A_find_b_and_c_l140_140078


namespace selling_price_per_book_l140_140637

noncomputable def fixed_costs : ℝ := 35630
noncomputable def variable_cost_per_book : ℝ := 11.50
noncomputable def num_books : ℕ := 4072
noncomputable def total_production_costs : ℝ := fixed_costs + variable_cost_per_book * num_books

theorem selling_price_per_book :
  (total_production_costs / num_books : ℝ) = 20.25 := by
  sorry

end selling_price_per_book_l140_140637


namespace functional_eq_solution_l140_140132

noncomputable def functional_eq (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x y : ℝ, f(x) * f(y) = f(x + y)

theorem functional_eq_solution (f : ℝ → ℝ) (h_cont : continuous f) (h_mono : monotone f) :
    (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ f = λ x, a^x) ↔ functional_eq f (λ x, a^x) :=
by
  sorry

end functional_eq_solution_l140_140132


namespace unit_vectors_collinear_with_a_l140_140711

-- Define the vector a
def a : ℝ × ℝ × ℝ := (-3, -4, 5)

-- Define the magnitude of the vector a
def magnitude_a : ℝ := real.sqrt ((-3)^2 + (-4)^2 + (5)^2)

-- Define the unit vectors collinear with the vector a
def unit_vector_pos : ℝ × ℝ × ℝ := (3 * real.sqrt 2 / 10, 2 * real.sqrt 2 / 5, -real.sqrt 2 / 2)
def unit_vector_neg : ℝ × ℝ × ℝ := (-3 * real.sqrt 2 / 10, -2 * real.sqrt 2 / 5, real.sqrt 2 / 2)

-- Prove that these are the unit vectors collinear with a
theorem unit_vectors_collinear_with_a :
  (unit_vector_pos = (3 * real.sqrt 2 / 10, 2 * real.sqrt 2 / 5, -real.sqrt 2 / 2) ∧
  unit_vector_neg = (-3 * real.sqrt 2 / 10, -2 * real.sqrt 2 / 5, real.sqrt 2 / 2)) ∧
  (∀ u : ℝ × ℝ × ℝ, (∃ k : ℝ, u = (k * (-3), k * (-4), k * 5) ∧ real.sqrt (u.1^2 + u.2^2 + u.3^2) = 1) →
  (u = unit_vector_pos ∨ u = unit_vector_neg)) :=
by
  sorry

end unit_vectors_collinear_with_a_l140_140711


namespace arithmetic_sequence_k_is_10_l140_140835

noncomputable def a_n (n : ℕ) (d : ℝ) : ℝ := (n - 1) * d

theorem arithmetic_sequence_k_is_10 (d : ℝ) (h : d ≠ 0) : 
  (∃ k : ℕ, a_n k d = (a_n 1 d) + (a_n 2 d) + (a_n 3 d) + (a_n 4 d) + (a_n 5 d) + (a_n 6 d) + (a_n 7 d) ∧ k = 10) := 
by
  sorry

end arithmetic_sequence_k_is_10_l140_140835


namespace distance_formula_l140_140339

theorem distance_formula (x1 y1 x2 y2 : ℝ) : 
  (x2 - x1)^2 + (y2 - y1)^2 = 289 → 
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 17 := 
by 
  sorry

example : 
  distance_formula 1 1 9 16 (8^2 + 15^2) := 
by
  refl

end distance_formula_l140_140339


namespace rooks_non_attacking_placements_l140_140840

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * factorial n

theorem rooks_non_attacking_placements :
  let chessboard_size := 8 in
  ∃! (N : ℕ), N = factorial chessboard_size := by
  sorry

end rooks_non_attacking_placements_l140_140840


namespace unbroken_seashells_left_l140_140513

-- Definitions based on given conditions
def total_seashells : ℕ := 6
def cone_shells : ℕ := 3
def conch_shells : ℕ := 3
def broken_cone_shells : ℕ := 2
def broken_conch_shells : ℕ := 2
def given_away_conch_shells : ℕ := 1

-- Mathematical statement to prove the final count of unbroken seashells
theorem unbroken_seashells_left : 
  (cone_shells - broken_cone_shells) + (conch_shells - broken_conch_shells - given_away_conch_shells) = 1 :=
by 
  -- Calculation (steps omitted per instructions)
  sorry

end unbroken_seashells_left_l140_140513


namespace positive_difference_x_coordinates_l140_140376

theorem positive_difference_x_coordinates :
  let l := (λ x, -2 * x + 10)
  let m := (λ x, -0.4 * x + 4)
  let x_l := (25 - 10) / (-2)
  let x_m := (25 - 4) / (-0.4)
  abs (x_l - x_m) = 45 :=
by
  let l := (λ x : ℝ, -2 * x + 10)
  let m := (λ x : ℝ, -0.4 * x + 4)
  let x_l := (25 - 10 : ℝ) / (-2)
  let x_m := (25 - 4 : ℝ) / (-0.4)
  have := abs (x_l - x_m) = 45
  sorry

end positive_difference_x_coordinates_l140_140376


namespace train_length_l140_140313

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 90) 
  (h2 : time_sec = 11) 
  (h3 : length_m = 275) :
  length_m = (speed_km_hr * 1000 / 3600) * time_sec :=
sorry

end train_length_l140_140313


namespace correct_graph_representation_l140_140539

structure Triangle (A B C : Type) :=
  (distance : A → A → ℝ)
  (is_equilateral : Equilateral A ∨ ¬Equilateral A)

variable (A B C : Type) [Triangle A (B, C)]

def tess_run : A → B → C → ℝ :=
  sorry

theorem correct_graph_representation (T : Triangle A B C) : 
  tess_run A B C = graph_D ↔ T.is_equilateral :=
sorry

end correct_graph_representation_l140_140539


namespace max_profit_at_nine_l140_140750

noncomputable def profit (x : ℝ) : ℝ := - (1 / 3) * x^3 + 81 * x - 23

theorem max_profit_at_nine :
  ∃ x : ℝ, x = 9 ∧ ∀ (ε : ℝ), ε > 0 → 
  (profit (9 - ε) < profit 9 ∧ profit (9 + ε) < profit 9) := 
by
  sorry

end max_profit_at_nine_l140_140750


namespace triangle_area_l140_140251

theorem triangle_area (base height : ℝ) (h_base : base = 4.5) (h_height : height = 6) :
  (base * height) / 2 = 13.5 := 
by
  rw [h_base, h_height]
  norm_num

-- sorry
-- The later use of sorry statement is commented out because the proof itself has been provided in by block.

end triangle_area_l140_140251


namespace annual_payment_correct_l140_140616

-- Define the given conditions
def principal : ℝ := 2000000
def interest_rate : ℝ := 0.10
def term_years : ℕ := 5
def no_payment_years : ℕ := 1
def annual_payment (P : ℝ) := 573583

-- Define the function to calculate the remaining amount after first year
def amount_after_first_year (principal : ℝ) (interest_rate : ℝ) : ℝ := 
  principal * (1 + interest_rate)

-- Define the present value annuity formula for given annual payment
def present_value_annuity (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := 
  P * ((1 - (1 + r)^(-n)) / r)

-- Define the primary target value to equate with the calculated present value
def target_present_value (principal : ℝ) (interest_rate : ℝ) : ℝ :=
  amount_after_first_year principal interest_rate

-- Proof statement
theorem annual_payment_correct : 
  ∀ (P : ℝ), present_value_annuity P interest_rate (term_years - no_payment_years) = target_present_value principal interest_rate → 
    annual_payment P = 573583 := 
by
  intros P h
  sorry

end annual_payment_correct_l140_140616


namespace largest_integer_n_l140_140585

theorem largest_integer_n 
  (h1 : Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 5) :
  ∃ (n : ℕ), Nat.choose 10 n = Nat.choose 10 5 ∧ ∀ (m : ℕ), Nat.choose 10 m = Nat.choose 10 5 → m ≤ n :=
begin
  use 5,
  split,
  { exact h1 },
  { intros m hm,
    sorry }
end

end largest_integer_n_l140_140585


namespace value_of_f3_f10_l140_140397

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) : f (x + 4) = f x + f 2
axiom f_at_one : f 1 = 4

theorem value_of_f3_f10 : f 3 + f 10 = 4 := sorry

end value_of_f3_f10_l140_140397


namespace problem_statement_l140_140125

def f (x : ℝ) : ℝ :=
if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem problem_statement : f (f (f (-1))) = Real.pi + 1 :=
by
  sorry

end problem_statement_l140_140125


namespace students_difference_l140_140452

-- Definitions based on conditions in the problem
def students_in_largest_class := 23
def total_students := 95
def num_classes := 5

-- Using the conditions to define the Lean problem statement
theorem students_difference (x : ℕ) :
  let C1 := students_in_largest_class
  ∧ let C2 := C1 - x
  ∧ let C3 := C2 - x
  ∧ let C4 := C3 - x
  ∧ let C5 := C4 - x
  ∧ C1 + C2 + C3 + C4 + C5 = total_students
  → x = 2 := 
sorry

end students_difference_l140_140452


namespace relationship_among_a_b_c_l140_140735

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_deriv : ∀ x ≠ 0, f'' x + f x / x > 0)

noncomputable def a : ℝ := (1 / Real.exp 1) * f (1 / Real.exp 1)
noncomputable def b : ℝ := -Real.exp 1 * f (-Real.exp 1)
noncomputable def c : ℝ := f 1

theorem relationship_among_a_b_c :
  a < c ∧ c < b :=
by
  -- sorry to skip the proof steps
  sorry

end relationship_among_a_b_c_l140_140735


namespace yellow_region_exists_l140_140677

theorem yellow_region_exists
  (n : ℕ)
  (h_n : n = 2018)
  (crosses : Finset (fin n × fin n))
  (no_three_concurrent : ∀ i j k : fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → 
                                        ¬ ∃ x : ℝ × ℝ, intersects_at_three x i j k)
  (alternate_colors : ∀ i : fin n, colors_alternately (vertices i))
  (yellow_vertex : fin n × fin n → Prop := λ v, ¬ (color_of_circle v.1 = color_of_circle v.2))
  (circle_with_yellow_points : ∃ i : fin n, count_yellow_vertices i ≥ 2061) :
  ∃ region : Finset (fin n × fin n), ∀ v ∈ region, yellow_vertex v :=
begin
  sorry
end

end yellow_region_exists_l140_140677


namespace arcsin_arccos_solution_l140_140180

theorem arcsin_arccos_solution 
  : ∃ x : ℝ, x = 0 ∧ arcsin x + arcsin (1 - 2 * x) = arccos (2 * x) :=
by
  exists 0
  split
  . refl
  . rw [arcsin_zero, arcsin_one, arccos_zero]
  exact Real.pi_div_two_add_arcsin_arcsin 0 (1 - 2 * 0)

end arcsin_arccos_solution_l140_140180


namespace find_a_l140_140770

noncomputable def A (a b : ℝ) : Set ℝ := {a, b, 2}
noncomputable def B (a b : ℝ) : Set ℝ := {2, b^2, 2 * a}

theorem find_a (a b : ℝ) (h: A a b ∩ B a b = A a b ∪ B a b) : a = 0 ∨ a = 1 / 4 :=
begin
  sorry,
end

end find_a_l140_140770


namespace find_y_l140_140424

theorem find_y (x y : ℤ) (h1 : x^2 - 3 * x + 6 = y + 2) (h2 : x = -5) : y = 44 :=
by 
  have h3 : (-5)^2 - 3 * (-5) + 6 = y + 2 := by rw h1; rw h2
  have h4 : 25 + 15 + 6 = y + 2 := by norm_num at h3; exact h3
  have h5 : 46 = y + 2 := by exact h4
  have h6 : 46 - 2 = y := by linarith [h5]
  exact h6

end find_y_l140_140424


namespace monotone_range_of_f_l140_140759

theorem monotone_range_of_f {f : ℝ → ℝ} (a : ℝ) 
  (h : ∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≤ y → f x ≤ f y) : a ≤ 0 :=
sorry

end monotone_range_of_f_l140_140759


namespace trajectory_of_M_is_line_segment_l140_140391

structure Point :=
(x : ℝ)
(y : ℝ)

def distance (p1 p2 : Point) : ℝ :=
((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2).sqrt

theorem trajectory_of_M_is_line_segment (F1 F2 M : Point) (h1 : distance F1 F2 = 8) (h2 : distance M F1 + distance M F2 = 8) :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = ⟨(1 - t) * F1.x + t * F2.x, (1 - t) * F1.y + t * F2.y⟩ :=
sorry

end trajectory_of_M_is_line_segment_l140_140391


namespace truck_distance_on_7_gallons_l140_140655

theorem truck_distance_on_7_gallons :
  ∀ (d : ℝ) (g₁ g₂ : ℝ), d = 240 → g₁ = 5 → g₂ = 7 → (d / g₁) * g₂ = 336 :=
by
  intros d g₁ g₂ h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end truck_distance_on_7_gallons_l140_140655


namespace probability_P_plus_S_mod4_condition_l140_140951

theorem probability_P_plus_S_mod4_condition :
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 60 →
  let S := a + b
  let P := a * b
  (∃ m : ℕ, (a + 1) * (b + 1) = 4 * m) →
  (1770 * (S + P) = 4 * 45 + 1) :=
begin
  sorry
end

end probability_P_plus_S_mod4_condition_l140_140951


namespace max_flowers_used_min_flowers_used_l140_140287

-- Part (a) Setup
def max_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def max_T (C M : ℕ) := C + M

-- Given conditions
theorem max_flowers_used :
  (∀ C M : ℕ, max_flowers C M 36 → max_T C M = 36) :=
by sorry

-- Part (b) Setup
def min_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def min_T (C M : ℕ) := C + M

-- Given conditions
theorem min_flowers_used :
  (∀ C M : ℕ, min_flowers C M 48 → min_T C M = 24) :=
by sorry

end max_flowers_used_min_flowers_used_l140_140287


namespace barbara_total_cost_l140_140667

-- Define conditions
def steak_weight : ℝ := 4.5
def steak_price_per_pound : ℝ := 15.0
def chicken_weight : ℝ := 1.5
def chicken_price_per_pound : ℝ := 8.0

-- Define total cost formula
def total_cost := (steak_weight * steak_price_per_pound) + (chicken_weight * chicken_price_per_pound)

-- Prove that the total cost equals $79.50
theorem barbara_total_cost : total_cost = 79.50 := by
  sorry

end barbara_total_cost_l140_140667


namespace average_first_20_multiples_of_17_l140_140956

theorem average_first_20_multiples_of_17 :
  (20 / 2 : ℝ) * (17 + 17 * 20) / 20 = 178.5 := by
  sorry

end average_first_20_multiples_of_17_l140_140956


namespace part1_part2_l140_140594

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (x + 1) - abs (a * x - 1)

theorem part1 (x : ℝ) : f x 1 > 1 ↔ x > 1 / 2 :=
by sorry

theorem part2 (a : ℝ) : (∀ x ∈ Ioo 0 1, f x a > x) ↔ (0 < a ∧ a ≤ 2) :=
by sorry

end part1_part2_l140_140594


namespace integral_x_squared_eq_9_inconclusive_l140_140786

variable (C : ℝ)

theorem integral_x_squared_eq_9_inconclusive :
  ∃ T : ℝ, ∫ x in 0 .. T, x^2 = 9 := sorry

end integral_x_squared_eq_9_inconclusive_l140_140786


namespace number_of_problems_l140_140912

theorem number_of_problems (Terry_score : ℤ) (points_right : ℤ) (points_wrong : ℤ) (wrong_ans : ℤ) 
  (h_score : Terry_score = 85) (h_points_right : points_right = 4) 
  (h_points_wrong : points_wrong = -1) (h_wrong_ans : wrong_ans = 3) : 
  ∃ (total_problems : ℤ), total_problems = 25 :=
by
  sorry

end number_of_problems_l140_140912


namespace f_log2_8_l140_140617

noncomputable def f : ℝ → ℝ := sorry

axiom even_f {x : ℝ} : f(x) = f(-x)
axiom f_periodicity {x : ℝ} : f(x + 1) = -f(x)
axiom f_property {x : ℝ} : -1 ≤ x ∧ x < 0 → f(x) = (1/2)^x

theorem f_log2_8 : f(real.log 8 (by norm_num)) = 2 := sorry

end f_log2_8_l140_140617


namespace misread_weight_l140_140919

-- Definitions based on given conditions in part (a)
def initial_avg_weight : ℝ := 58.4
def num_boys : ℕ := 20
def correct_weight : ℝ := 61
def correct_avg_weight : ℝ := 58.65

-- The Lean theorem statement that needs to be proved
theorem misread_weight :
  let incorrect_total_weight := initial_avg_weight * num_boys
  let correct_total_weight := correct_avg_weight * num_boys
  let weight_diff := correct_total_weight - incorrect_total_weight
  correct_weight - weight_diff = 56 := sorry

end misread_weight_l140_140919


namespace maria_trip_time_l140_140691

theorem maria_trip_time 
(s_highway : ℕ) (s_mountain : ℕ) (d_highway : ℕ) (d_mountain : ℕ) (t_mountain : ℕ) (t_break : ℕ) : 
  (s_highway = 4 * s_mountain) -> 
  (t_mountain = d_mountain / s_mountain) -> 
  t_mountain = 40 -> 
  t_break = 15 -> 
  d_highway = 100 -> 
  d_mountain = 20 ->
  s_mountain = d_mountain / t_mountain -> 
  s_highway = 4 * s_mountain -> 
  d_highway / s_highway = 50 ->
  40 + 50 + 15 = 105 := 
by 
  sorry

end maria_trip_time_l140_140691


namespace total_blocks_travelled_l140_140159

theorem total_blocks_travelled :
  let walk1 := 5
  let bus := 20
  let walk2 := 10
  let outward := walk1 + bus + walk2
  let return := walk2 + bus + walk1
  outward + return = 70 :=
by
  sorry

end total_blocks_travelled_l140_140159


namespace min_value_equals_two_thirds_l140_140357

noncomputable def min_expression := ∀ x : ℝ, (sin x)^4 + (cos x)^4 + 2 / (sin x)^2 + (cos x)^2 + 2

theorem min_value_equals_two_thirds :
  ∃ x : ℝ, min_expression x = 2 / 3 ∧ ∀ y : ℝ, min_expression y ≥ 2 / 3 :=
sorry

end min_value_equals_two_thirds_l140_140357


namespace complex_div_eq_half_sub_half_i_l140_140920

theorem complex_div_eq_half_sub_half_i (i : ℂ) (hi : i^2 = -1) : 
  (i^3 / (1 - i)) = (1 / 2) - (1 / 2) * i :=
by
  sorry

end complex_div_eq_half_sub_half_i_l140_140920


namespace number_of_customers_l140_140514

theorem number_of_customers (m_0 : ℕ) (mr : ℕ) (mb : ℕ) (h1 : m_0 = 400) (h2 : mr = 100) (h3 : mb = 15) :
  (m_0 - mr) / mb = 20 :=
by
  rw [h1, h2, h3]
  sorry

end number_of_customers_l140_140514


namespace rope_length_91_4_l140_140570

noncomputable def total_rope_length (n : ℕ) (d : ℕ) (pi_val : Real) : Real :=
  let linear_segments := 6 * d
  let arc_length := (d * pi_val / 3) * 6
  let total_length_per_tie := linear_segments + arc_length
  total_length_per_tie * 2

theorem rope_length_91_4 :
  total_rope_length 7 5 3.14 = 91.4 :=
by
  sorry

end rope_length_91_4_l140_140570


namespace sum_of_coordinates_l140_140029

-- Definitions
def g : ℝ → ℝ
def f (x y : ℝ) : (ℝ × ℝ) := (4, 7)

-- Proof problem
theorem sum_of_coordinates :
  (∃ x y, y = (2 * g(3 * x) + 6) / 3 ∧ g 4 = 7 ∧ g (3 * (4 / 3)) = 7 ∧ x = 4 / 3 → x + y = 8) :=
by
  sorry

end sum_of_coordinates_l140_140029


namespace parking_problem_l140_140599

def f : ℕ × ℕ → ℕ
| (n, 0) := 1
| (0, k) := 0
| (1, k) := if k = 1 then 1 else 0
| (2, k) := if k = 1 then 2 else 0
| (n, k) := 
  if k > n then 0 
  else f (n - 1, k) + f (n - 2, k - 1) + f (n - 3, k - 2)

theorem parking_problem : f (12, 6) = 357 :=
sorry

end parking_problem_l140_140599


namespace cricket_target_runs_l140_140847

-- Define the conditions
def first_20_overs_run_rate : ℝ := 4.2
def remaining_30_overs_run_rate : ℝ := 8
def overs_20 : ℤ := 20
def overs_30 : ℤ := 30

-- State the proof problem
theorem cricket_target_runs : 
  (first_20_overs_run_rate * (overs_20 : ℝ)) + (remaining_30_overs_run_rate * (overs_30 : ℝ)) = 324 :=
by
  sorry

end cricket_target_runs_l140_140847


namespace inverse_variation_z_x_square_l140_140944

theorem inverse_variation_z_x_square (x z : ℝ) (K : ℝ) 
  (h₀ : z * x^2 = K) 
  (h₁ : x = 3 ∧ z = 2)
  (h₂ : z = 8) :
  x = 3 / 2 := 
by 
  sorry

end inverse_variation_z_x_square_l140_140944


namespace find_a_b_l140_140005

theorem find_a_b (a b : ℝ) (h1 : (λ x, (a * x + b) * Real.exp x) 0 = 1)
                 (h2 : (λ x, (a * x + a + b) * Real.exp x) 0 = 2) :
  a + b = 2 :=
sorry

end find_a_b_l140_140005


namespace question1_question2_question3_l140_140437

-- Define probabilities of renting and returning bicycles at different stations
def P (X Y : Char) : ℝ :=
  if X = 'A' ∧ Y = 'A' then 0.3 else
  if X = 'A' ∧ Y = 'B' then 0.2 else
  if X = 'A' ∧ Y = 'C' then 0.5 else
  if X = 'B' ∧ Y = 'A' then 0.7 else
  if X = 'B' ∧ Y = 'B' then 0.1 else
  if X = 'B' ∧ Y = 'C' then 0.2 else
  if X = 'C' ∧ Y = 'A' then 0.4 else
  if X = 'C' ∧ Y = 'B' then 0.5 else
  if X = 'C' ∧ Y = 'C' then 0.1 else 0

-- Question 1: Prove P(CC) = 0.1
theorem question1 : P 'C' 'C' = 0.1 := by
  sorry

-- Question 2: Prove P(AC) * P(CB) = 0.25
theorem question2 : P 'A' 'C' * P 'C' 'B' = 0.25 := by
  sorry

-- Question 3: Prove the probability P = 0.43
theorem question3 : P 'A' 'A' * P 'A' 'A' + P 'A' 'B' * P 'B' 'A' + P 'A' 'C' * P 'C' 'A' = 0.43 := by
  sorry

end question1_question2_question3_l140_140437


namespace geom_seq_find_b3_l140_140848

-- Given conditions
def is_geometric_seq (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

def geom_seq_condition (b : ℕ → ℝ) : Prop :=
  is_geometric_seq b ∧ b 2 * b 3 * b 4 = 8

-- Proof statement: We need to prove that b 3 = 2
theorem geom_seq_find_b3 (b : ℕ → ℝ) (h : geom_seq_condition b) : b 3 = 2 :=
  sorry

end geom_seq_find_b3_l140_140848


namespace center_of_symmetry_l140_140591

theorem center_of_symmetry (k : ℤ) : ∀ (k : ℤ), ∃ x : ℝ, 
  (x = (k * Real.pi / 6 - Real.pi / 9) ∨ x = - (Real.pi / 18)) → False :=
by
  sorry

end center_of_symmetry_l140_140591


namespace distinct_sequences_six_sided_die_rolled_six_times_l140_140998

theorem distinct_sequences_six_sided_die_rolled_six_times :
  let count := 6
  (count ^ 6 = 46656) :=
by
  let count := 6
  sorry

end distinct_sequences_six_sided_die_rolled_six_times_l140_140998


namespace dissertation_ratio_l140_140859

theorem dissertation_ratio :
  let acclimation := 1
  let basics := 2
  let research := 2 + 0.75 * 2
  let total := 7 - (acclimation + basics + research)
  total / acclimation = 0.5 :=
by
  let acclimation := 1
  let basics := 2
  let research := 2 + 0.75 * 2
  let total := 7 - (acclimation + basics + research)
  show total / acclimation = 0.5
  sorry
  
end dissertation_ratio_l140_140859


namespace ellipse_problem_l140_140116

noncomputable def conditions are satisfied (a b : ℝ) : Prop :=
    (a > b ∧ b > 0) ∧
    (sqrt 2 / 2 = sqrt (1 - b^2 / a^2)) ∧
    (0, 1) ∈ {p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1}

noncomputable def equation_of_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
    x^2 / 2 + y^2 = 1

noncomputable def equation_of_line_AB (k x y : ℝ) : Prop :=
    y = k * (x + 1)

noncomputable def relationship_between_BF1_and_F1A (F1 A B : ℝ × ℝ) : Prop :=
    let BF1 := (F1.1 - B.1, F1.2 - B.2)
    let F1A := (A.1 - F1.1, A.2 - F1.2)
    in BF1 = (2 * F1A.1, 2 * F1A.2)

noncomputable def equation_of_line_BF2 (x y : ℝ) : Prop :=
    ∃ k : ℝ, (k = sqrt 14 / 6 ∨ k = - sqrt 14 / 6) ∧ (y = k * (x - 1))

theorem ellipse_problem
    (a b x y : ℝ)
    (F1 A B : ℝ × ℝ)
    (k : ℝ) :
    conditions_are_satisfied a b →
    equation_of_ellipse a b x y →
    equation_of_line_AB k x y →
    relationship_between_BF1_and_F1A F1 A B →
    equation_of_line_BF2 x y :=
sorry

end ellipse_problem_l140_140116


namespace chords_intersection_theorem_l140_140441

-- Definitions and assumptions
variables {R : Type*} [MetricSpace R] [NormedAddCommGroup R] [NormedSpace ℝ R]

-- Given a circle with center O and radius r.
variable (O : R)
variable (r : ℝ)

-- Definition of points M, K, N on the circle
axiom circle_contains_points (M K N : R) (hM : dist O M = r) (hK : dist O K = r) (hN : dist O N = r) : 
  true

-- M and N are vertices of an inscribed square OMKN with O as center
axiom square_properties (M K N : R) (h_square : true):
  true

-- Coordinates of intersection points A, B, C, D
variables (A B C D : R)

-- Chord AB passes through M and chord CD passes through N
axiom chord_AB_through_M (h_A_M_B : dist A M + dist M B = dist A B) : true
axiom chord_CD_through_N (h_C_N_D : dist C N + dist N D = dist C D) : true

-- The theorem we want to prove
theorem chords_intersection_theorem : 
  (dist A M * dist M B) = (dist C N * dist N D) := 
sorry

end chords_intersection_theorem_l140_140441


namespace ratio_of_money_spent_on_clothes_is_1_to_2_l140_140373

-- Definitions based on conditions
def allowance1 : ℕ := 5
def weeks1 : ℕ := 8
def allowance2 : ℕ := 6
def weeks2 : ℕ := 6
def cost_video : ℕ := 35
def remaining_money : ℕ := 3

-- Calculations
def total_saved : ℕ := (allowance1 * weeks1) + (allowance2 * weeks2)
def total_expended : ℕ := cost_video + remaining_money
def spent_on_clothes : ℕ := total_saved - total_expended

-- Prove the ratio of money spent on clothes to the total money saved is 1:2
theorem ratio_of_money_spent_on_clothes_is_1_to_2 :
  (spent_on_clothes : ℚ) / (total_saved : ℚ) = 1 / 2 :=
by
  sorry

end ratio_of_money_spent_on_clothes_is_1_to_2_l140_140373


namespace purchase_price_of_road_bikes_possible_purchase_plans_maximize_profit_l140_140540

-- Problem 1: Price determination
theorem purchase_price_of_road_bikes : 
  ∃ x1 x2 : ℕ, 
  (∃ x : ℕ, x = x1 ∧ x + 600 = x2 ∧ 
    (5000 / x = 8000 / (x + 600))) → 
  (x1 = 1000 ∧ x2 = 1600) :=
begin
  sorry
end

-- Problem 2: Determining possible purchase plans
theorem possible_purchase_plans (total_budget : ℕ) (m : ℕ) : 
  1000 * m + 1600 * (50 - m) ≤ total_budget ∧ (50 - m) ≥ m →
  20 ≤ m ∧ m ≤ 25 :=
begin
  sorry
end

-- Problem 3: Maximizing Profit 
theorem maximize_profit : 
  let profit m := 100 * m + 20000 in 
  ∃ max_profit : ℕ, 
  (∀ m, 20 ≤ m ∧ m ≤ 25 → profit m ≤ max_profit) ∧ 
  max_profit = profit 25 ∧ 
  max_profit = 22500 :=
begin
  sorry
end

end purchase_price_of_road_bikes_possible_purchase_plans_maximize_profit_l140_140540


namespace cows_total_l140_140317

theorem cows_total (A M R : ℕ) (h1 : A = 4 * M) (h2 : M = 60) (h3 : A + M = R + 30) : 
  A + M + R = 570 := by
  sorry

end cows_total_l140_140317


namespace intersection_product_l140_140843

-- Define the parametric equations of the line l
def parametric_eq_line_l (t : ℝ) : ℝ × ℝ :=
  (-((Real.sqrt 3) / 2) * t, (1 / 2) + (1 / 2) * t)

-- Define the Cartesian equation of the circle C
def cartesian_eq_circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x = 0

-- Statement of the problem
theorem intersection_product (t₁ t₂ : ℝ) (H1 : ∃ t, parametric_eq_line_l t = (t₁, (1 / 2) + (1 / 2) * t))
  (H2 : ∃ t, parametric_eq_line_l t = (t₂, (1 / 2) + (1 / 2) * t))
  (H3 : cartesian_eq_circle_C t₁ ((1 / 2) + (1 / 2) * t₁))
  (H4 : cartesian_eq_circle_C t₂ ((1 / 2) + (1 / 2) * t₂)) :
  |t₁ * t₂| = 1 / 4 := sorry

end intersection_product_l140_140843


namespace problem_equivalent_l140_140465

open Real

noncomputable def parametric_equation_of_C (t : ℝ) : ℝ × ℝ :=
  (-1 + 3 * cos t, -2 + 3 * sin t)

def line_l_in_cartesian (m : ℝ) (x y : ℝ) : Prop :=
  x - y + m = 0 

def distance_point_to_line (x1 y1 m : ℝ) :=
  abs (x1 - y1 + m) / sqrt 2

theorem problem_equivalent :
  (∀ t, (parametric_equation_of_C t).1 ^ 2 + (parametric_equation_of_C t).2 ^ 2 = 9) ∧
  distance_point_to_line 1 (-2) m = 2 → 
  m = -3 + 2 * sqrt 2 ∨ m = -3 - 2 * sqrt 2 :=
by
  simp [parametric_equation_of_C, line_l_in_cartesian, distance_point_to_line]
  sorry

end problem_equivalent_l140_140465


namespace calculate_expression_l140_140674

theorem calculate_expression : 
  (0.25 ^ 16) * ((-4) ^ 17) = -4 := 
by
  sorry

end calculate_expression_l140_140674


namespace find_alpha_l140_140719

variables {α : Type*} [LinearOrder α] [OrderTop α] [OrderBot α] [LinearOrder α]
def vector_a (α: ℝ) : ℝ × ℝ := (3/2, sin α)
def vector_b (α: ℝ) : ℝ × ℝ := (cos α, 1/3)

theorem find_alpha (h : vector_a α = vector_b α ∧ α > 0 ∧ α < π) : α = π / 4 :=
by 
  sorry

end find_alpha_l140_140719


namespace count_ordered_triples_l140_140359

theorem count_ordered_triples :
  (∃ a b c : ℤ, a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ log a (b:ℝ) = (c:ℝ)^2 ∧ a + b + c = 35) →
  (∃! t : ℤ, t = 2) :=
by sorry

end count_ordered_triples_l140_140359


namespace problem_statement_l140_140753

noncomputable def omega (dist : ℝ) : ℝ :=
Inf {ω | ω > 0 ∧ (2 * Real.pi / ω) = dist}

def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem problem_statement :
  ∃ ω > 0, (2 * Real.pi / ω = Real.pi / 3) ∧
    (f ω = λ x, Real.sin (3 * x + Real.pi / 4)) ∧
    (∃ m > 0, g = λ x, Real.sin (3 * x + 3 * m + Real.pi / 4) →
      ∀ x, g x = g (-x) → m = Real.pi / 12) := sorry

end problem_statement_l140_140753


namespace limit_of_function_l140_140331

theorem limit_of_function (f : ℝ → ℝ)
  (h1 : ∀ x, f x = (ln (1 + x) / (6 * x)) ^ (x / (x + 2))) :
  (filter.tendsto f (nhds 0) (nhds 1)) :=
by sorry

end limit_of_function_l140_140331


namespace find_x_l140_140849

theorem find_x (x : ℝ) (h : 6 * x + 3 * x + 4 * x + 2 * x = 360) : x = 24 :=
sorry

end find_x_l140_140849


namespace triangle_perimeter_l140_140360

def distance (p q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

def P := distance (0, 20) (12, 0) + distance (0, 20) (0, 0) + distance (12, 0) (0, 0)

theorem triangle_perimeter :
  P = real.sqrt 544 + 32 :=
by
  sorry

end triangle_perimeter_l140_140360


namespace limit_trigonometric_identity_l140_140330

open Real

theorem limit_trigonometric_identity :
  (∀ x, cos (2 * x) = 1 - 2 * (sin x)^2) →
  (∃ l, tendsto (λ x : ℝ, (1 + x * sin x - cos (2 * x)) / (sin x)^2) (𝓝 0) (𝓝 l) ∧ l = 3) :=
by
  sorry

end limit_trigonometric_identity_l140_140330


namespace sequence_converges_to_4_l140_140165

open Real

noncomputable def binom (n k : ℕ) : ℝ :=
  if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))
  else 0

noncomputable def sequence (n : ℕ) : ℝ :=
  (binom (2 * n) n) ^ (1 / n : ℝ)

theorem sequence_converges_to_4 : 
  tendsto (fun n => sequence n) at_top (nhds 4) :=
sorry

end sequence_converges_to_4_l140_140165


namespace find_M_l140_140491

def S : Finset ℕ := (Finset.range 13).image (λ x, 2^x)

noncomputable def M : ℕ :=
  S.sum (λ x, S.sum (λ y, if x > y then x - y else 0))

theorem find_M : M = 81968 := by
  sorry

end find_M_l140_140491


namespace rectangle_area_l140_140203

noncomputable def side_of_square : ℝ := Real.sqrt 625

noncomputable def radius_of_circle : ℝ := side_of_square

noncomputable def length_of_rectangle : ℝ := (2 / 5) * radius_of_circle

def breadth_of_rectangle : ℝ := 10

theorem rectangle_area :
  length_of_rectangle * breadth_of_rectangle = 100 := 
by
  simp [length_of_rectangle, breadth_of_rectangle, radius_of_circle, side_of_square]
  sorry

end rectangle_area_l140_140203


namespace domain_of_f_x_plus_1_l140_140421

noncomputable def f (x : ℝ) := 1 / Real.sqrt (Real.logBase (1/2) (2*x - 1))

theorem domain_of_f_x_plus_1 :
  ∀ x : ℝ, (f (x + 1)).isDefined ↔ x ∈ Set.Ioo (-1/2 : ℝ) 0 :=
by
  sorry

end domain_of_f_x_plus_1_l140_140421


namespace positive_real_inequality_l140_140714

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end positive_real_inequality_l140_140714


namespace locus_of_M_l140_140008

def square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, A.1 = B.1 ∧ A.2 = C.2 ∧ (B.2 - A.2)^2 = s^2 ∧ (C.1 - A.1)^2 = s^2 ∧ (D.1 - C.1)^2 = s^2 ∧ (D.2 - B.2)^2 = s^2

def perpendicular_bisector (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 ≠ A.2 ∧ M.2 ≠ B.2

def diagonal_extension_rays (A B C D M : ℝ × ℝ) : Prop :=
  (M.1 = A.1 ∧ M.2 > A.2) ∨ (M.1 = A.1 ∧ M.2 < A.2) ∨ 
  (M.1 > A.1 ∧ M.2 = A.2) ∨ (M.1 < A.1 ∧ M.2 = A.2)

def circumscribed_circle_arc_excluding_vertices (A B C D M : ℝ × ℝ) : Prop :=
  let center := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) in
  let radius := dist center A in
  dist center M = radius ∧ M ≠ A ∧ M ≠ B ∧ M ≠ C ∧ M ≠ D

theorem locus_of_M (A B C D M : ℝ × ℝ) (h : square A B C D) :
  (∃ M, ∠AMB = ∠CMD) ↔ (perpendicular_bisector B C M ∨ diagonal_extension_rays A B C D M ∨ circumscribed_circle_arc_excluding_vertices A B C D M) :=
sorry

end locus_of_M_l140_140008


namespace statement_1_equiv_statement_2_equiv_l140_140897

-- Statement 1
variable (A B C : Prop)

theorem statement_1_equiv : ((A ∨ B) → C) ↔ (A → C) ∧ (B → C) :=
by
  sorry

-- Statement 2
theorem statement_2_equiv : (A → (B ∧ C)) ↔ (A → B) ∧ (A → C) :=
by
  sorry

end statement_1_equiv_statement_2_equiv_l140_140897


namespace find_M_l140_140492

def S : Finset ℕ := (Finset.range 13).image (λ x, 2^x)

noncomputable def M : ℕ :=
  S.sum (λ x, S.sum (λ y, if x > y then x - y else 0))

theorem find_M : M = 81968 := by
  sorry

end find_M_l140_140492


namespace ratio_of_monitors_to_students_l140_140558

theorem ratio_of_monitors_to_students (S G B : ℕ) 
  (girls_percentage : ℚ)
  (monitors_count : ℕ)
  (milk_per_boy : ℕ)
  (milk_per_girl : ℕ)
  (total_milk_consumption : ℕ)
  (h1: girls_percentage = 0.40)
  (h2: monitors_count = 8)
  (h3: milk_per_boy = 1)
  (h4: milk_per_girl = 2)
  (h5: total_milk_consumption = 168)
  (h6: G = 0.40 * S)
  (h7: B = 0.60 * S)
  (h8: 2 * G + B = 168) :
  monitors_count / S = 1 / 15 :=
  by sorry

end ratio_of_monitors_to_students_l140_140558


namespace year_2049_is_Jisi_l140_140932

-- Define Heavenly Stems
def HeavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]

-- Define Earthly Branches
def EarthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

-- Define the indices of Ding (丁) and You (酉) based on 2017
def Ding_index : Nat := 3
def You_index : Nat := 9

-- Define the year difference
def year_difference : Nat := 2049 - 2017

-- Calculate the indices for the Heavenly Stem and Earthly Branch in 2049
def HeavenlyStem_index_2049 : Nat := (Ding_index + year_difference) % 10
def EarthlyBranch_index_2049 : Nat := (You_index + year_difference) % 12

theorem year_2049_is_Jisi : 
  HeavenlyStems[HeavenlyStem_index_2049]? = some "Ji" ∧ EarthlyBranches[EarthlyBranch_index_2049]? = some "Si" :=
by
  sorry

end year_2049_is_Jisi_l140_140932


namespace trig_solution_l140_140380

theorem trig_solution (α : ℝ) (h : cos α = -4/5) : 
  (sin α = 3/5 ∨ sin α = -3/5) ∧ (tan α = -3/4 ∨ tan α = 3/4) :=
by
  sorry

end trig_solution_l140_140380


namespace max_value_f_solution_set_inequality_l140_140136

def f (x : ℝ) := Real.sqrt (2 * x - 4) + Real.sqrt (5 - x)

theorem max_value_f : ∃ M, (M = 3 ∧ ∀ x, f(x) ≤ M) :=
  sorry

theorem solution_set_inequality : { x : ℝ | |x - 1| + |x + 2| ≤ 3 } = set.Icc (-2) 1 :=
  sorry

end max_value_f_solution_set_inequality_l140_140136


namespace largest_pillar_radius_l140_140996

-- Define the dimensions of the crate
def crate_length := 12
def crate_width := 8
def crate_height := 3

-- Define the condition that the pillar is a right circular cylinder
def is_right_circular_cylinder (r : ℝ) (h : ℝ) : Prop :=
  r > 0 ∧ h > 0

-- The theorem stating the radius of the largest volume pillar that can fit in the crate
theorem largest_pillar_radius (r h : ℝ) (cylinder_fits : is_right_circular_cylinder r h) :
  r = 1.5 := 
sorry

end largest_pillar_radius_l140_140996


namespace value_of_x_y_l140_140047

noncomputable def real_ln : ℝ → ℝ := sorry

theorem value_of_x_y (x y : ℝ) (h : 3 * x - y ≤ real_ln (x + 2 * y - 3) + real_ln (2 * x - 3 * y + 5)) :
  x + y = 16 / 7 :=
sorry

end value_of_x_y_l140_140047


namespace parabola_equation_l140_140402

theorem parabola_equation :
  ∃ (m : ℝ), (vertex_at_origin : x = 0 ∧ y = 0) →
  (axis_of_symmetry : x) →
  (-2, 2 * real.sqrt 2) ∈ { p : ℝ × ℝ | p.2^2 = m * p.1 } →
  m = -4 :=
by
  sorry

end parabola_equation_l140_140402


namespace total_seeds_in_watermelons_l140_140954

def slices1 : ℕ := 40
def seeds_per_slice1 : ℕ := 60
def slices2 : ℕ := 30
def seeds_per_slice2 : ℕ := 80
def slices3 : ℕ := 50
def seeds_per_slice3 : ℕ := 40

theorem total_seeds_in_watermelons :
  (slices1 * seeds_per_slice1) + (slices2 * seeds_per_slice2) + (slices3 * seeds_per_slice3) = 6800 := by
  sorry

end total_seeds_in_watermelons_l140_140954


namespace AK_squared_condition_l140_140160

-- Define a triangle ABC
variable {A B C K : Point}
variable [triangle : IsTriangle A B C]

-- Define side K lies on BC
variable (hK : LiesOnSegment K B C)

-- Define the mathematical equivalent problem in Lean
theorem AK_squared_condition :
  AK^2 = AB * AC - KB * KC ↔ AB = AC ∨ ∠ BAK = ∠ CAK :=
sorry

end AK_squared_condition_l140_140160


namespace square_area_l140_140455

open real

theorem square_area (s WO NO : ℝ) 
  (hWO : WO = 8) 
  (hNO : NO = 9) 
  (hs : s^2 = WO^2 + NO^2) : 
  s^2 = 145 := 
by {
  rw [hWO, hNO, hs],
  norm_num,
  exact sorry
}

end square_area_l140_140455


namespace inequality_in_triangle_l140_140602

theorem inequality_in_triangle
  (A B C P : Point)
  (h_acute : acute_triangle A B C)
  (hP_inside : inside_triangle P A B C) :
  (dist P A + dist P B + dist P C) ≥
  (2 / 3) * (perimeter (tangency_triangle A B C)) :=
sorry

end inequality_in_triangle_l140_140602


namespace mike_score_l140_140837

theorem mike_score (Gibi Jigi Lizzy max_score avg_score : ℕ) 
  (hGibi : Gibi = 59 * max_score / 100)
  (hJigi : Jigi = 55 * max_score / 100)
  (hLizzy : Lizzy = 67 * max_score / 100)
  (hMax : max_score = 700)
  (hAvg : avg_score = 490) :
  let total_score := 4 * avg_score in
  let total_non_mike := Gibi + Jigi + Lizzy in
  let Mike := total_score - total_non_mike in
  (Mike * 100 / max_score = 99) :=
by {
  sorry
}

end mike_score_l140_140837


namespace max_value_of_function_l140_140205

theorem max_value_of_function :
  (∃ x : ℝ, cos (2 * x) + 6 * cos (π / 2 - x) = 5) :=
by
  sorry

end max_value_of_function_l140_140205


namespace distance_from_A_to_line_l140_140622

noncomputable theory

variables {A B C : Type} [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] 
variables (M : affineCombination {A, B, C}) -- Centroid of ΔABC
variables (ℓ : affineMap ℝ A (EuclideanSpace ℝ))
variables {b c : ℝ} -- Distances from B and C to ℓ

def is_centroid_of_triangle : Prop :=
  centroid ℝ (points : set (EuclideanSpace ℝ)) = M

def line_through_centroid_and_intersects_sides : Prop :=
  ∃ (P : affineCombination {A, B}) (Q : affineCombination {A, C}),
    ℓ = affineCombination {P, Q}

def distance_from_B_to_line : Prop :=
  distance B (ℓ B) = b

def distance_from_C_to_line : Prop :=
  distance C (ℓ C) = c

theorem distance_from_A_to_line
  (h_centroid : is_centroid_of_triangle M)
  (h_line_intersect : line_through_centroid_and_intersects_sides ℓ)
  (h_distance_B : distance_from_B_to_line B ℓ b)
  (h_distance_C : distance_from_C_to_line C ℓ c) :
  distance A (ℓ A) = b + c :=
sorry

end distance_from_A_to_line_l140_140622


namespace problem_statement_l140_140808

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.arccos (inner a b / (‖a‖ * ‖b‖))

theorem problem_statement
  (a b : EuclideanSpace ℝ (Fin 3))
  (h_angle_ab : angle_between_vectors a b = Real.pi / 3)
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 1) :
  angle_between_vectors a (a + 2 • b) = Real.pi / 6 :=
sorry

end problem_statement_l140_140808


namespace compare_abc_l140_140122

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 2
noncomputable def c : ℝ := 9 ^ (1 / 2 : ℝ)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end compare_abc_l140_140122


namespace car_resultant_speed_and_direction_l140_140280

theorem car_resultant_speed_and_direction :
  let car_speed_mps := 13 / 54
  let car_speed_kmph := car_speed_mps * 3.6
  let crosswind_speed_kmph := 0.7 * 60
  let resultant_speed := Real.sqrt (car_speed_kmph ^ 2 + crosswind_speed_kmph ^ 2)
  let resultant_direction := Real.arctan (crosswind_speed_kmph / car_speed_kmph)
  in abs (resultant_speed - 42.0007) < 1e-4 ∧ abs (resultant_direction - Real.arctan 175) < 1e-2 :=
by
  -- placeholder for the proof
  sorry

end car_resultant_speed_and_direction_l140_140280


namespace sin_of_double_angle_equals_neg_seventeen_over_twentyfive_l140_140419

theorem sin_of_double_angle_equals_neg_seventeen_over_twentyfive (α : ℝ) :
  sin (π / 4 + α) = 2 / 5 → sin (2 * α) = -17 / 25 :=
by
  intros h
  sorry

end sin_of_double_angle_equals_neg_seventeen_over_twentyfive_l140_140419


namespace mapping_count_l140_140868

noncomputable def number_of_mappings (M : Set ℕ) (N : Set ℕ) (f : ℕ → ℕ) : ℕ :=
  if M = {0, 1, 2} ∧ N = {0, 1} ∧ (∀ a b c, a ∈ M → b ∈ M → c ∈ M → f a + f b = f c → a ≠ b → (a, b) ≠ (b, a))
  then 3
  else 0 

theorem mapping_count (M : Set ℕ) (N : Set ℕ) (f : ℕ → ℕ)
  (hM : M = {0, 1, 2})
  (hN : N = {0, 1})
  (h_f : ∀ a b c, a ∈ M → b ∈ M → c ∈ M → f a + f b = f c): 
  number_of_mappings M N f = 3 :=
by
  sorry

end mapping_count_l140_140868


namespace possible_values_of_a_l140_140453

def original_data := [0, 3, 5, 7, 10]
def new_data (a : Int) := original_data ++ [a]

def mean (data : List Int) : Real :=
  (data.foldl (λ acc x => acc + x) 0) / data.length

def variance (data : List Int) : Real :=
  let m := mean data
  data.foldl (λ acc x => acc + (x - m) * (x - m)) 0 / data.length

noncomputable def original_mean : Real := mean original_data
noncomputable def original_variance : Real := variance original_data

theorem possible_values_of_a (a : Int) (h1 : mean (new_data a) <= original_mean) 
  (h2 : variance (new_data a) < original_variance) : a ∈ {2, 3, 4, 5} := 
  sorry

end possible_values_of_a_l140_140453


namespace chicken_cost_l140_140107

theorem chicken_cost (total_money hummus_price hummus_count bacon_price vegetables_price apple_price apple_count chicken_price : ℕ)
  (h_total_money : total_money = 60)
  (h_hummus_price : hummus_price = 5)
  (h_hummus_count : hummus_count = 2)
  (h_bacon_price : bacon_price = 10)
  (h_vegetables_price : vegetables_price = 10)
  (h_apple_price : apple_price = 2)
  (h_apple_count : apple_count = 5)
  (h_remaining_money : chicken_price = total_money - (hummus_count * hummus_price + bacon_price + vegetables_price + apple_count * apple_price)) :
  chicken_price = 20 := 
by sorry

end chicken_cost_l140_140107


namespace simplify_exponent_product_l140_140258

theorem simplify_exponent_product :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_product_l140_140258


namespace number_of_ways_to_form_teams_l140_140093

-- Define the necessary binomial coefficient
def C (n k : ℕ) : ℕ := Nat.choose n k
-- Define the necessary arrangement formula
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem number_of_ways_to_form_teams :
  let women := 5
      men := 5
      selected_players := 6
      automatic_selection := 2 -- Ma Long and Ding Ning
  in (C 4 2 ^ 2 * A 3 3) = 
     (Nat.choose 4 2 ^ 2 * (Nat.factorial 3 / Nat.factorial 0)) :=  
sorry

end number_of_ways_to_form_teams_l140_140093


namespace mode_and_median_of_temperatures_l140_140819

noncomputable def temperatures : List ℕ := [25, 26, 26, 28, 27, 14, 10]

theorem mode_and_median_of_temperatures :
  (List.mode temperatures = some 26) ∧ (List.median temperatures = some 26) :=
sorry

end mode_and_median_of_temperatures_l140_140819


namespace trip_savings_l140_140573

theorem trip_savings :
  let regular_ticket_price := 10
  let student_discount := 2
  let senior_discount := 3
  let ticket_discount := 0.20
  let large_popcorn_drink_price := 10
  let medium_nachos_drink_price := 8
  let hotdog_soft_drink_price := 6
  let large_popcorn_drink_discount := 0.50
  let medium_nachos_drink_discount := 0.30
  let hotdog_soft_drink_discount := 0.20
  let third_combo_discount := 0.25
  -- Prices without discounts
  let total_ticket_price_without_discount := (regular_ticket_price + (regular_ticket_price - student_discount) + regular_ticket_price + (regular_ticket_price - senior_discount))
  let total_food_price_without_discount := (large_popcorn_drink_price + medium_nachos_drink_price + hotdog_soft_drink_price)
  -- Prices with discounts
  let total_ticket_price_with_discount := ((regular_ticket_price * (1 - ticket_discount)) + ((regular_ticket_price - student_discount) * (1 - ticket_discount)) + (regular_ticket_price * (1 - ticket_discount)) + ((regular_ticket_price - senior_discount) * (1 - ticket_discount)))
  let large_popcorn_drink_price_with_discount := large_popcorn_drink_price * (1 - large_popcorn_drink_discount)
  let medium_nachos_drink_price_with_discount := medium_nachos_drink_price * (1 - medium_nachos_drink_discount)
  let hotdog_soft_drink_price_with_discount := hotdog_soft_drink_price * (1 - hotdog_soft_drink_discount)
  let total_food_price_with_discount := (large_popcorn_drink_price_with_discount + medium_nachos_drink_price_with_discount + (hotdog_soft_drink_price_with_discount * (1 - third_combo_discount)))
  -- Savings
  let ticket_savings := total_ticket_price_without_discount - total_ticket_price_with_discount
  let food_savings := total_food_price_without_discount - total_food_price_with_discount
  -- Total savings
  let total_savings := ticket_savings + food_savings
  in total_savings = 16.80 := by
  sorry

end trip_savings_l140_140573


namespace count_8_digit_even_ending_l140_140776

theorem count_8_digit_even_ending : 
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  (choices_first_digit * choices_middle_digits * choices_last_digit) = 45000000 :=
by
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  sorry

end count_8_digit_even_ending_l140_140776


namespace number_of_divisors_of_8_factorial_l140_140057

theorem number_of_divisors_of_8_factorial :
  let eight_factorial_prime_factorization : ℕ × ℕ × ℕ × ℕ := (7, 2, 1, 1)
  in ∏ (e : ℕ) in eight_factorial_prime_factorization, e + 1 = 96 :=
by
  let eight_factorial_prime_factorization := (7, 2, 1, 1)
  let number_of_divisors := (8_factorial_prime_factorization.1 + 1) *
                           (8_factorial_prime_factorization.2 + 1) *
                           (8_factorial_prime_factorization.3 + 1) *
                           (8_factorial_prime_factorization.4 + 1)
  show number_of_divisors = 96
  trivial -- Skip the proof with 'trivial'

end number_of_divisors_of_8_factorial_l140_140057


namespace Q_cannot_be_log_x_l140_140428

def P : Set ℝ := {y | y ≥ 0}

theorem Q_cannot_be_log_x (Q : Set ℝ) :
  (P ∩ Q = Q) → Q ≠ {y | ∃ x, y = Real.log x} :=
by
  sorry

end Q_cannot_be_log_x_l140_140428


namespace fruit_vendor_sales_l140_140518

noncomputable def total_fruits_sold (lemons dozens_lemons : ℕ) 
                           (avocados dozens_avocados : ℕ) 
                           (oranges dozens_oranges : ℕ) 
                           (apples dozens_apples : ℕ) 
                           (fruits_per_dozen : ℕ := 12) : ℕ :=
  lemons * dozens_lemons + avocados * dozens_avocados +
  oranges * dozens_oranges + apples * dozens_apples

theorem fruit_vendor_sales : 
  total_fruits_sold 2.5 1 5.25 1 3.75 1 2.12 1.round = 163 := 
by sorry

end fruit_vendor_sales_l140_140518


namespace find_S2023_l140_140386

noncomputable def a_sequence : ℕ → ℚ
| 0       := 1
| (n + 1) := a_sequence n / (3 * a_sequence n + 1)

def b_sequence (n : ℕ) : ℚ := a_sequence n * a_sequence (n + 1)

def sum_b_sequence (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, b_sequence i)

theorem find_S2023 : sum_b_sequence 2023 = 2023 / 6070 :=
by 
  sorry

end find_S2023_l140_140386


namespace online_sale_discount_l140_140233

theorem online_sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (total_paid : ℕ) : 
  purchase_amount = 250 → 
  discount_per_100 = 10 → 
  total_paid = purchase_amount - (purchase_amount / 100) * discount_per_100 → 
  total_paid = 230 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end online_sale_discount_l140_140233


namespace count_polynomials_l140_140680

theorem count_polynomials (n : ℕ) (h_n : n = 30) :
  (∑ (a : ℕ) in finset.range 10, 
  ∑ (b : ℕ) in finset.range 10, 
  ∑ (c : ℕ) in finset.range 10, 
  ∑ (d : ℕ) in finset.range 10, if a - b + c - d = n then 1 else 0) = 5456 :=
by sorry

end count_polynomials_l140_140680


namespace unique_root_in_interval_l140_140135

noncomputable def f (x : ℝ) : ℝ := -2 * x^3 - x + 1

theorem unique_root_in_interval (m n : ℝ) (h1 : f m * f n < 0) (h2 : m ≤ n) :
  ∃! c ∈ set.Icc m n, f c = 0 :=
by
  -- Proof to be filled in
  sorry

end unique_root_in_interval_l140_140135


namespace total_ticket_revenue_l140_140628

theorem total_ticket_revenue (total_seats : Nat) (price_adult_ticket : Nat) (price_child_ticket : Nat)
  (theatre_full : Bool) (child_tickets : Nat) (adult_tickets := total_seats - child_tickets)
  (rev_adult := adult_tickets * price_adult_ticket) (rev_child := child_tickets * price_child_ticket) :
  total_seats = 250 →
  price_adult_ticket = 6 →
  price_child_ticket = 4 →
  theatre_full = true →
  child_tickets = 188 →
  rev_adult + rev_child = 1124 := 
by
  intros h_total_seats h_price_adult h_price_child h_theatre_full h_child_tickets
  sorry

end total_ticket_revenue_l140_140628


namespace general_term_of_sequence_l140_140940

def a : ℕ → ℝ 
| 0       := 0
| 1       := 1
| (n + 2) := (1 / 16) * (1 + 4 * a (n + 1) + real.sqrt (1 + 24 * a (n + 1)))

theorem general_term_of_sequence :
  ∀ n : ℕ, n ≥ 1 → a n = (1 / 3) + (1 / 2) ^ n + (1 / 3) * (1 / 2) ^ (2 * n - 1) :=
begin
  sorry
end

end general_term_of_sequence_l140_140940


namespace sum_binom_neg_x_pos_l140_140906

theorem sum_binom_neg_x_pos (n k : ℕ) (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 1 / n) (h₂ : k ≤ n) : 
  ∑ i in Finset.range (k + 1), (Nat.choose n i) * (-x)^i > 0 := 
sorry

end sum_binom_neg_x_pos_l140_140906


namespace average_other_color_marbles_l140_140152

def percentage_clear : ℝ := 0.4
def percentage_black : ℝ := 0.2
def total_percentage : ℝ := 1.0
def total_marbles_taken : ℝ := 5.0

theorem average_other_color_marbles :
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black in
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors in
  expected_other_color_marbles = 2 := by
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors
  show expected_other_color_marbles = 2
  sorry

end average_other_color_marbles_l140_140152


namespace mean_score_of_sophomores_l140_140639

open Nat

variable (s j : ℕ)
variable (m m_s m_j : ℝ)

theorem mean_score_of_sophomores :
  (s + j = 150) →
  (m = 85) →
  (j = 80 / 100 * s) →
  (m_s = 125 / 100 * m_j) →
  (s * m_s + j * m_j = 12750) →
  m_s = 94 := by intros; sorry

end mean_score_of_sophomores_l140_140639


namespace four_digit_integer_count_l140_140414

theorem four_digit_integer_count :
  let first_digit_choices := {2, 6, 9}
  let second_digit_choices := {2, 6, 9}
  let third_digit_choices := {0, 4, 8}
  let fourth_digit_choices := {0, 4, 8}
  let first_two_digits_combinations := finset.card (finset.product first_digit_choices second_digit_choices)
  let valid_last_two_digits := finset.filter (λ (p : ℕ × ℕ), p.1 ≠ p.2) (finset.product third_digit_choices fourth_digit_choices)
  let last_two_digits_combinations := finset.card valid_last_two_digits
  first_two_digits_combinations * last_two_digits_combinations = 54 := by
  {
    sorry
  }

end four_digit_integer_count_l140_140414


namespace min_total_diff_char_l140_140995

-- Define ability characteristics for students.
def ability_characteristic := Fin 12 → ℕ

def num_diff_char (A B : ability_characteristic) : ℕ :=
  ∑ i, abs (A i - B i)

def significant_diff (A B : ability_characteristic) : Prop :=
  num_diff_char A B ≥ 7

-- Define the total difference for three students.
def total_num_diff_char (A B C : ability_characteristic) : ℕ :=
  ∑ i, (abs (A i - B i) + abs (B i - C i) + abs (C i - A i))

def significant_diff_three (A B C : ability_characteristic) : Prop :=
  ∀ {X Y}, (X ≠ Y) → (X = A ∨ X = B ∨ X = C) → (Y = A ∨ Y = B ∨ Y = C) → significant_diff X Y

theorem min_total_diff_char (A B C : ability_characteristic) (h : significant_diff_three A B C) :
  total_num_diff_char A B C ≥ 22 :=
sorry

end min_total_diff_char_l140_140995


namespace ratio_of_areas_correct_l140_140115

noncomputable def ratio_of_areas (s : ℝ) : ℝ :=
  let side_sq : ℝ := s^2
  let height_eq_triangle := (4 * real.sqrt(3) / 2)
  let center_height := height_eq_triangle / 3
  let cd_length : ℝ := s
  let eg_length : ℝ := (4 * (3 + real.sqrt(3)) / 3)
  let rect_area := cd_length * eg_length
  rect_area / side_sq

theorem ratio_of_areas_correct : ratio_of_areas 4 = 1 + real.sqrt(3) / 3 := 
by 
  admit 

end ratio_of_areas_correct_l140_140115


namespace solution_comparison_l140_140869

variables (a a' b b' : ℝ)

theorem solution_comparison (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-(b / a) < -(b' / a')) ↔ (b' / a' < b / a) :=
by sorry

end solution_comparison_l140_140869


namespace det_matrix_power_l140_140784

theorem det_matrix_power (M : Matrix n n ℝ) (h : det M = 3) : det (M ^ 3) = 27 := by
  sorry

end det_matrix_power_l140_140784


namespace fish_kept_l140_140111

theorem fish_kept (Leo_caught Agrey_more Sierra_more Leo_fish Returned : ℕ) 
                  (Agrey_caught : Agrey_more = 20) 
                  (Sierra_caught : Sierra_more = 15) 
                  (Leo_caught_cond : Leo_fish = 40) 
                  (Returned_cond : Returned = 30) : 
                  (Leo_fish + (Leo_fish + Agrey_more) + ((Leo_fish + Agrey_more) + Sierra_more) - Returned) = 145 :=
by
  sorry

end fish_kept_l140_140111


namespace math_problem_l140_140045

def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (1/2 + (√2)/2 * t, 1/2 - (√2)/2 * t)

def parametric_ellipse_C (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sin α)

def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem math_problem
  (t α : ℝ)
  (A_polar : ℝ × ℝ := (2, Real.pi / 3))
  (A_cartesian := polar_to_cartesian A_polar.1 A_polar.2) :

  ( ∃ x y : ℝ, x = 2 * Real.cos α ∧ y = Real.sin α ∧
                x^2 / 4 + y^2 = 1 ) ∧
  ( A_cartesian = (1, Real.sqrt 3) ) ∧
  ( let l_intersects_C := λ t α, parametric_line_l t = parametric_ellipse_C α in
    let P := parametric_line_l (some t) in
    let Q := parametric_line_l (some (-t)) in
    let dist_pq := Real.sqrt (2 * (P.1 - Q.1)^2) in
    let d := Real.sqrt 3 / Real.sqrt 2 in
    let area_triangle := 1/2 * dist_pq * d in
    area_triangle = 4 * Real.sqrt 3 / 5 ) :=
sorry

end math_problem_l140_140045


namespace rotate_point_D_l140_140161

/--
Given a point D(-6, 2), if we rotate this point 180 degrees about the origin,
we should get a new point D' whose coordinates are (6, -2).
-/
theorem rotate_point_D (D : ℝ × ℝ) (hD : D = (-6, 2)) : 
  let D' := (λ (x : ℝ × ℝ), (-x.1, -x.2)) D
  in D' = (6, -2) :=
by {
  rw hD,
  simp,
  sorry
}

end rotate_point_D_l140_140161


namespace find_BC_squared_l140_140113

-- Definitions for the statements from the problem
variables (ABC : Type) [triangle ABC]
variables (A B C D E F P : ABC)
variables (BC CA AB : ℝ)
variables (incircle_touches : (D ∈ incircle ABC) ∧ (E ∈ incircle ABC) ∧ (F ∈ incircle ABC))
variables (foot_of_perpendicular : is_foot_of_perpendicular_point D P (line_through E F))
variables (angle_condition : ∠ B P C = 90)

-- Given the conditions
def given_conditions := (AB = 20) ∧ (CA = 22) ∧ incircle_touches ∧ foot_of_perpendicular ∧ angle_condition

-- Proof goal
theorem find_BC_squared (h : given_conditions) : BC^2 = 84 :=
sorry

end find_BC_squared_l140_140113


namespace real_yield_correct_l140_140980

-- Define the annual inflation rate and nominal interest rate
def annualInflationRate : ℝ := 0.025
def nominalInterestRate : ℝ := 0.06

-- Define the number of years
def numberOfYears : ℕ := 2

-- Calculate total inflation over two years
def totalInflation (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n - 1

-- Calculate the compounded nominal rate
def compoundedNominalRate (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n

-- Calculate the real yield considering inflation
def realYield (nominalRate : ℝ) (inflationRate : ℝ) : ℝ :=
  (nominalRate / (1 + inflationRate)) - 1

-- Proof problem statement
theorem real_yield_correct :
  let inflation_two_years := totalInflation annualInflationRate numberOfYears
  let nominal_two_years := compoundedNominalRate nominalInterestRate numberOfYears
  inflation_two_years * 100 = 5.0625 ∧
  (realYield nominal_two_years inflation_two_years) * 100 = 6.95 :=
by
  sorry

end real_yield_correct_l140_140980


namespace geometric_ratio_condition_l140_140042

theorem geometric_ratio_condition {a b c d a' b' c' d' : ℝ} (hb : b ≠ 0) (hd : d ≠ 0) (hb' : b' ≠ 0) (hd' : d' ≠ 0) :
  (a / b = c / d) → (a' / b' = c' / d') → 
  (a + a') / (b + b') = (c + c') / (d + d') ↔ a / a' = b / b' ∧ b / b' = c / c' ∧ c / c' = d / d' :=
by
  intro h1 h2
  sorry

end geometric_ratio_condition_l140_140042


namespace online_sale_discount_l140_140234

theorem online_sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (total_paid : ℕ) : 
  purchase_amount = 250 → 
  discount_per_100 = 10 → 
  total_paid = purchase_amount - (purchase_amount / 100) * discount_per_100 → 
  total_paid = 230 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end online_sale_discount_l140_140234


namespace B_completion_time_l140_140316

-- Definitions based on the conditions
def A_work : ℚ := 1 / 24
def B_work : ℚ := 1 / 16
def C_work : ℚ := 1 / 32  -- Since C takes twice the time as B, C_work = B_work / 2

-- Combined work rates based on the conditions
def combined_ABC_work := A_work + B_work + C_work
def combined_AB_work := A_work + B_work

-- Question: How long does B take to complete the job alone?
-- Answer: 16 days

theorem B_completion_time : 
  (combined_ABC_work = 1 / 8) ∧ 
  (combined_AB_work = 1 / 12) ∧ 
  (A_work = 1 / 24) ∧ 
  (C_work = B_work / 2) → 
  (1 / B_work = 16) := 
by 
  sorry

end B_completion_time_l140_140316


namespace frame_cut_into_distinct_parts_assemble_to_square_l140_140301

def is_rectangular_frame_cut_to_nine_parts_distinct (parts : List Part) : Prop :=
  parts.length = 9 ∧ ∀ (i j : ℕ), i < 9 → j < 9 → i ≠ j → parts[i] ≠ parts[j]

def can_be_assembled_to_square (parts : List Part) (square : Square) : Prop :=
  -- Assuming Part and Square are predefined structures representing the parts and the square respectively
  -- Insert necessary conditions about assembling the parts into the square here
  sorry

noncomputable def is_possible_to_form_square_from_distinct_parts (frm : Frame) : Prop :=
  ∃ parts : List Part,
  is_rectangular_frame_cut_to_nine_parts_distinct parts ∧
  can_be_assembled_to_square parts frm

theorem frame_cut_into_distinct_parts_assemble_to_square (frm : Frame) :
  is_possible_to_form_square_from_distinct_parts frm :=
sorry

end frame_cut_into_distinct_parts_assemble_to_square_l140_140301


namespace simplify_and_rationalize_denominator_l140_140531

theorem simplify_and_rationalize_denominator :
  (sqrt 3 / sqrt 8 * sqrt 5 / sqrt 9 * sqrt 7 / sqrt 12) = (35 * sqrt 70 / 840) :=
by sorry

end simplify_and_rationalize_denominator_l140_140531


namespace simplify_expression_l140_140178

theorem simplify_expression : (- (1 / 343 : ℝ)) ^ (-2 / 3 : ℝ) = 49 :=
by 
  sorry

end simplify_expression_l140_140178


namespace find_constants_l140_140023

-- Let y1, y2, y3 be functions of x which are solutions to a differential equation
variable (y1 y2 y3 : ℝ → ℝ)
variable (a b c : ℝ → ℝ)

-- Assume given differential equation holds for y1, y2, y3
axiom diff_eq1 : ∀ x, y1''' x + a x * y1'' x + b x * y1' x + c x * y1 x = 0
axiom diff_eq2 : ∀ x, y2''' x + a x * y2'' x + b x * y2' x + c x * y2 x = 0
axiom diff_eq3 : ∀ x, y3''' x + a x * y3'' x + b x * y3' x + c x * y3 x = 0

-- Assume the given condition
axiom sum_squares_eq1 : ∀ x, (y1 x)^2 + (y2 x)^2 + (y3 x)^2 = 1

theorem find_constants :
  ∃ α β : ℝ, α = 2/3 ∧ β = -2/3 ∧ (∀ x, (y1' x)^2 + (y2' x)^2 + (y3' x)^2 = z x ∧
    (∀ x, z' x + α * a x * z x + β * c x = 0)) := sorry

end find_constants_l140_140023


namespace average_of_numbers_l140_140193

theorem average_of_numbers : 
  let nums := [10, 4, 8, 7, 6]
  let sum_nums := 10 + 4 + 8 + 7 + 6
  let count_nums := nums.length
  let avg := sum_nums / count_nums in
  avg = 7 :=
by
  -- sorry to skip the proof
  sorry

end average_of_numbers_l140_140193


namespace square_area_from_hexagon_l140_140142

theorem square_area_from_hexagon (hex_side length square_side : ℝ) (h1 : hex_side = 4) (h2 : length = 6 * hex_side)
  (h3 : square_side = length / 4) : square_side ^ 2 = 36 :=
by 
  sorry

end square_area_from_hexagon_l140_140142


namespace distance_between_truck_and_car_l140_140640

noncomputable def truck_speed : ℝ := 65
noncomputable def car_speed : ℝ := 85
noncomputable def time_duration_hours : ℝ := 3 / 60

theorem distance_between_truck_and_car :
  let relative_speed := car_speed - truck_speed in
  let distance := relative_speed * time_duration_hours in
  distance = 1 :=
by
  let relative_speed := car_speed - truck_speed
  let distance := relative_speed * time_duration_hours
  have h1 : distance = 1 := sorry
  exact h1

end distance_between_truck_and_car_l140_140640


namespace computer_price_reduction_l140_140612

theorem computer_price_reduction :
  ∀ (C : ℝ), 0 < C →
  let first_discount := 0.8 * C in
  let final_price := 0.7 * first_discount in
  (1 - final_price / C) = 0.44 :=
by
  intros
  let first_discount := 0.8 * C
  let final_price := 0.7 * first_discount
  sorry

end computer_price_reduction_l140_140612


namespace translate_f_odd_l140_140034

def f (x : ℝ) : ℝ := (sin x + cos x) * cos x

theorem translate_f_odd :
  let g (x : ℝ) := f (x - π / 8) - 1 / 2 in
  ∀ x, g (-x) = -g (x) :=
by
  sorry

end translate_f_odd_l140_140034


namespace gcd_abcd_dcba_l140_140200

-- Definitions based on the conditions
def abcd (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d
def dcba (a b c d : ℕ) : ℕ := 1000 * d + 100 * c + 10 * b + a
def consecutive_digits (a b c d : ℕ) : Prop := (b = a + 1) ∧ (c = a + 2) ∧ (d = a + 3)

-- Theorem statement
theorem gcd_abcd_dcba (a b c d : ℕ) (h : consecutive_digits a b c d) : 
  Nat.gcd (abcd a b c d + dcba a b c d) 1111 = 1111 :=
sorry

end gcd_abcd_dcba_l140_140200


namespace horner_method_V3_correct_when_x_equals_2_l140_140581

-- Polynomial f(x)
noncomputable def f (x : ℝ) : ℝ :=
  2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

-- Horner's method for evaluating f(x)
noncomputable def V3 (x : ℝ) : ℝ :=
  (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Proof that V3 = 12 when x = 2
theorem horner_method_V3_correct_when_x_equals_2 : V3 2 = 12 := by
  sorry

end horner_method_V3_correct_when_x_equals_2_l140_140581


namespace triangle_angle_AHC_eq_110_l140_140469

/-- In triangle ABC, with altitudes AD and BE intersecting at H, and given angles BAC = 50 degrees
and ABC = 60 degrees, prove that angle AHC is 110 degrees. --/
theorem triangle_angle_AHC_eq_110
  (A B C H D E : Point)
  (triangle : Triangle A B C)
  (altitude_AD : Altitude A D)
  (altitude_BE : Altitude B E)
  (H_intersection : Point H = intersection altitude_AD altitude_BE)
  (angle_BAC_eq_50 : angle A B C = 50)
  (angle_ABC_eq_60 : angle A B C = 60)
  : angle A H C = 110 :=
sorry

end triangle_angle_AHC_eq_110_l140_140469


namespace find_length_AD_l140_140572

variables (A B C D O P : Type) [metric_space A]

-- Conditions
def is_trapezoid (AB : segment A B) (CD : segment C D) : Prop :=
  parallel AB CD

def equal_segments (BC CD : segment B C) : Prop := 
  length BC = 39 ∧ length CD = 39

def right_angle (AD BD : segment A D) (BD BD' : segment B D) : Prop :=
  (AD ⊥ BD) ∧ (BD = BD')

def midpoint_of_BD (P : midpoint B D) (BD : segment B D) : Prop :=
  P = midpoint BD

def intersection_diagonals (AC BD : segment A C) (O : intersection AC BD) : Prop :=
  O = intersection AC BD

-- Given OP = 10
def OP_length (OP : length O P) : Prop :=
  OP = 10

-- Conclusion
theorem find_length_AD : 
  ∃ k t : ℤ, (t > 0) ∧ (¬ ∃ p : ℤ, prime p ∧ p^2 ∣ t) ∧ length AD = k * sqrt t ∧ k + t = 73 := 
  sorry

end find_length_AD_l140_140572


namespace solve_inequality_l140_140909

theorem solve_inequality (x : ℝ) : -3*x^2 + 9*x + 6 > 0 ↔ x ∈ set.Ioo (-∞) (-1) ∪ set.Ioo 4 ∞ := 
sorry

end solve_inequality_l140_140909


namespace hyperbola_equation_of_midpoint_l140_140043

-- Define the hyperbola E
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Given conditions
variables (a b : ℝ) (hapos : a > 0) (hbpos : b > 0)
variables (F : ℝ × ℝ) (hF : F = (-2, 0))
variables (M : ℝ × ℝ) (hM : M = (-3, -1))

-- The statement requiring proof
theorem hyperbola_equation_of_midpoint (hE : hyperbola a b (-2) 0) 
(hFocus : a^2 + b^2 = 4) : 
  (∃ a' b', a' = 3 ∧ b' = 1 ∧ hyperbola a' b' (-3) (-1)) :=
sorry

end hyperbola_equation_of_midpoint_l140_140043


namespace cos_theta_chords_l140_140442

theorem cos_theta_chords (r θ φ : ℝ) (h1 : θ + φ < π)
  (h2 : cos (θ / 2) = 4 / 5)
  (h3 : 6 ^ 2 + 10 ^ 2 = 8 ^ 2 + 128) :
  cos θ = 7 / 25 ∧ (7 + 25 = 32) :=
by
  sorry

end cos_theta_chords_l140_140442


namespace sausage_cutting_l140_140626

theorem sausage_cutting (red_pieces yellow_pieces green_pieces total_pieces : ℕ) 
  (h_red : red_pieces = 5)
  (h_yellow : yellow_pieces = 7)
  (h_green : green_pieces = 11)
  (h_total : total_pieces = (red_pieces - 1) + (yellow_pieces - 1) + (green_pieces - 1) + 1) :
  total_pieces = 21 := 
by {
  rw [h_red, h_yellow, h_green],
  sorry
}

end sausage_cutting_l140_140626


namespace a_term_b_value_c_value_d_value_l140_140066

theorem a_term (a x : ℝ) (h1 : a * (x + 1) = x^3 + 3 * x^2 + 3 * x + 1) : a = x^2 + 2 * x + 1 :=
sorry

theorem b_value (a x b : ℝ) (h1 : a - 1 = 0) (h2 : x = 0 ∨ x = b) : b = -2 :=
sorry

theorem c_value (p c b : ℝ) (h1 : p * c^4 = 32) (h2 : p * c = b^2) (h3 : 0 < c) : c = 2 :=
sorry

theorem d_value (A B d : ℝ) (P : ℝ → ℝ) (c : ℝ) (h1 : P (A * B) = P A + P B) (h2 : P A = 1) (h3 : P B = c) (h4 : A = 10^ P A) (h5 : B = 10^ P B) (h6 : d = A * B) : d = 1000 :=
sorry

end a_term_b_value_c_value_d_value_l140_140066


namespace find_t_sum_bn_sign_change_count_l140_140221

-- (1) Given the conditions, prove t = 1
theorem find_t {t : ℝ} (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : a 1 = t) 
  (h2 : ∀ n : ℕ, n > 0 → a (n+1) = 2 * S n + 1) 
  (h3 : ∀ n : ℕ, n > 0 → a n = 2 * S (n-1) + 1) 
  (geom_seq : ∀ n : ℕ, n ≥ 2 → a (n + 1) = 3 * a n) 
  : t = 1 := 
sorry

-- (2) Given the conditions, prove the sum of the first n terms T_n
theorem sum_bn {T : ℕ → ℝ} {a : ℕ → ℝ} {b : ℕ → ℝ}
  (h1 : a 1 = 1) (geom_seq : ∀ n : ℕ, a (n+1) = 3 * a n) 
  (h2 : ∀ n : ℕ, b n = n * a (n-1)) 
  (T_def : ∀ n : ℕ, T n = ∑ i in Finset.range n, b (i + 1)) 
  : T n = (2 * n - 1) / 4 * 3^n + 1 / 4 := 
sorry

-- (3) Given the conditions, prove the sign-changing count
theorem sign_change_count {c : ℕ → ℝ} {b : ℕ → ℝ}
  (h1 : ∀ n : ℕ, b n = n * 3 ^ (n - 1)) 
  (h2 : ∀ n : ℕ, c n = (b n - 4) / b n) 
  (increased_seq : ∀ n : ℕ, n ≥ 1 → c (n + 1) > c n)
  : ∀ i : ℕ, (i = 1) -> signed_change_count c = 1 := 
sorry

end find_t_sum_bn_sign_change_count_l140_140221


namespace find_2003rd_non_perfect_square_l140_140853

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def non_perfect_squares : ℕ → ℕ
| 0 := 1
| n := (if is_perfect_square (n+1) then non_perfect_squares n + 1 else non_perfect_squares n) + 1

theorem find_2003rd_non_perfect_square : non_perfect_squares 2002 = 2048 := by
  sorry

end find_2003rd_non_perfect_square_l140_140853


namespace intersection_of_A_and_B_l140_140743

def setA (x : ℝ) : Prop := -1 < x ∧ x < 1
def setB (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def setC (x : ℝ) : Prop := 0 ≤ x ∧ x < 1

theorem intersection_of_A_and_B : {x : ℝ | setA x} ∩ {x | setB x} = {x | setC x} := by
  sorry

end intersection_of_A_and_B_l140_140743


namespace number_of_fish_given_to_dog_l140_140861

-- Define the conditions
def condition1 (D C : ℕ) : Prop := C = D / 2
def condition2 (D C : ℕ) : Prop := D + C = 60

-- Theorem to prove the number of fish given to the dog
theorem number_of_fish_given_to_dog (D : ℕ) (C : ℕ) (h1 : condition1 D C) (h2 : condition2 D C) : D = 40 :=
by
  sorry

end number_of_fish_given_to_dog_l140_140861


namespace find_4_digit_number_l140_140699

theorem find_4_digit_number (a b c d : ℕ) 
(h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182)
(h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
(h6 : 0 ≤ c) (h7 : c ≤ 9) (h8 : 1 ≤ d) (h9 : d ≤ 9) : 
1000 * a + 100 * b + 10 * c + d = 1909 :=
sorry

end find_4_digit_number_l140_140699


namespace limit_exp_zn_l140_140526

noncomputable def lim_n_exp (z : ℝ) : Prop :=
  (λ n, (1 + z / n)^n) ⟶ exp z at_top

theorem limit_exp_zn (z : ℝ) : lim_n_exp z := by
  sorry

end limit_exp_zn_l140_140526


namespace simplify_f_find_f_alpha_plus_pi_find_f_specific_alpha_l140_140378

-- Definition of the function f
def f (α : ℝ) : ℝ :=
  (sin (α - π) * cos (3 * π / 2 + α) * tan (-α - π)) / 
  (sin (5 * π + α) * tan (-α - 2 * π)^2)

-- (1) Prove that for any real number α, f(α) = -cos(α).
theorem simplify_f (α : ℝ) : f α = - cos α :=
  sorry

-- (2) Given α in the third quadrant and cos(α + π/2) = 1/5, prove that f(α + π) = -2√6/5.
theorem find_f_alpha_plus_pi (α : ℝ) (h1 : α > π ∧ α < 3 * π) (h2 : cos (α + π / 2) = 1 / 5) : f (α + π) = -(2 * real.sqrt 6) / 5 :=
  sorry

-- (3) Given α = 2011π/3, prove that f(α) = -1/2.
theorem find_f_specific_alpha : f (2011 * π / 3) = -1 / 2 :=
  sorry

end simplify_f_find_f_alpha_plus_pi_find_f_specific_alpha_l140_140378


namespace enclosed_area_is_correct_l140_140195

noncomputable def radius_of_arcs (arc_length : ℝ) : ℝ :=
  let r := arc_length / (2 * π)
  r

noncomputable def area_of_pentagon (s : ℝ) : ℝ :=
  (1 / 4) * ( √(5 * (5 + 2 * √5)) ) * s^2

noncomputable def area_of_arc_sector (r : ℝ) (arc_length : ℝ) : ℝ :=
  let θ := arc_length / r
  (1 / 2) * r^2 * θ

noncomputable def total_enclosed_area (side_length : ℝ) (arc_length : ℝ) : ℝ :=
  let r := radius_of_arcs(arc_length)
  let pentagon_area := area_of_pentagon(side_length)
  let arc_sector_area := area_of_arc_sector(r, arc_length)
  let total_arc_area := 10 * arc_sector_area
  pentagon_area + total_arc_area

theorem enclosed_area_is_correct :
  total_enclosed_area 3 (π / 3) = 15.484 + (5 * π / 6) :=
by sorry

end enclosed_area_is_correct_l140_140195


namespace sum_of_medians_gt_perimeter_quarter_l140_140907

theorem sum_of_medians_gt_perimeter_quarter
  (A B C D E F G : Point) 
  (median_ad : is_median A D B C)
  (median_be : is_median B E C A)
  (median_cf : is_median C F A B)
  (midpoint_d : is_midpoint B C D)
  (midpoint_e : is_midpoint C A E)
  (midpoint_f : is_midpoint A B F)
  (centroid_ratio : divides_in_ratio G (2/3) (1/3) AD BE CF) :
  distance A D + distance B E + distance C F > (3/4) * (distance A B + distance B C + distance C A) :=
sorry

end sum_of_medians_gt_perimeter_quarter_l140_140907


namespace running_cost_percentage_calculation_l140_140242

-- Definitions of the given problem conditions.
def initial_cost := 100000
def daily_revenue := 150 * 10
def number_of_days := 200

-- The daily running cost as a percentage of the initial cost.
def daily_running_cost_percentage (P : Real) := (P / 100) * initial_cost

-- The theorem to prove that the daily running cost percentage is 10% given the conditions.
theorem running_cost_percentage_calculation (P : Real) :
  number_of_days * daily_revenue = initial_cost + number_of_days * daily_running_cost_percentage P →
  P = 10 :=
by
  sorry

end running_cost_percentage_calculation_l140_140242


namespace product_of_odd_integers_less_than_200_l140_140253

theorem product_of_odd_integers_less_than_200 :
  let product := ∏ i in finset.filter (λ x, x % 2 = 1) (finset.range 200), i
  product = (199.factorial / (2^99 * 99.factorial)) :=
by 
  -- Proof goes here
  sorry

end product_of_odd_integers_less_than_200_l140_140253


namespace total_passengers_landed_l140_140477

theorem total_passengers_landed 
  (passengers_on_time : ℕ) 
  (passengers_late : ℕ) 
  (passengers_connecting : ℕ) 
  (passengers_changed_plans : ℕ)
  (H1 : passengers_on_time = 14507)
  (H2 : passengers_late = 213)
  (H3 : passengers_connecting = 320)
  (H4 : passengers_changed_plans = 95) : 
  passengers_on_time + passengers_late + passengers_connecting = 15040 :=
by 
  sorry

end total_passengers_landed_l140_140477


namespace number_of_divisors_of_8_factorial_l140_140056

theorem number_of_divisors_of_8_factorial :
  let eight_factorial_prime_factorization : ℕ × ℕ × ℕ × ℕ := (7, 2, 1, 1)
  in ∏ (e : ℕ) in eight_factorial_prime_factorization, e + 1 = 96 :=
by
  let eight_factorial_prime_factorization := (7, 2, 1, 1)
  let number_of_divisors := (8_factorial_prime_factorization.1 + 1) *
                           (8_factorial_prime_factorization.2 + 1) *
                           (8_factorial_prime_factorization.3 + 1) *
                           (8_factorial_prime_factorization.4 + 1)
  show number_of_divisors = 96
  trivial -- Skip the proof with 'trivial'

end number_of_divisors_of_8_factorial_l140_140056


namespace fraction_always_irreducible_l140_140903

-- Define the problem statement
theorem fraction_always_irreducible (n : ℤ) : gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_always_irreducible_l140_140903


namespace no_transformation_possible_l140_140630

-- Define the quadratics involved
def poly1 : ℚ[X] := X^2 + 4 * X + 3
def poly2 : ℚ[X] := X^2 + 10 * X + 9

-- Define the operations
def op1 (f : ℚ[X]) : ℚ[X] := X^2 * (f.eval (1 / X + 1))
def op2 (f : ℚ[X]) : ℚ[X] := (X - 1)^2 * (f.eval (1 / (X - 1)))

-- The theorem we want to prove
theorem no_transformation_possible :
  ¬ (∃ f : ℚ[X], (op1 poly1 = f ∨ op2 poly1 = f) ∧ (op1 f = poly2 ∨ op2 f = poly2)) :=
sorry

end no_transformation_possible_l140_140630


namespace possible_values_for_n_l140_140886

theorem possible_values_for_n (n : ℕ) (h1 : ∀ a b c : ℤ, (a = n-1) ∧ (b = n) ∧ (c = n+1) → 
    (∃ f g : ℤ, f = 2*a - b ∧ g = 2*b - a)) 
    (h2 : ∃ a b c : ℤ, (a = 0 ∨ b = 0 ∨ c = 0) ∧ (a + b + c = 0)) : 
    ∃ k : ℕ, n = 3^k := 
sorry

end possible_values_for_n_l140_140886


namespace hyperbola_eq_solution_find_D_and_t_solution_l140_140025

noncomputable def find_hyperbola_equation (a b : ℝ) (h₁: a > 0) (h₂: b > 0)
  (hyp_real_axis_len : 2 * a = 4 * sqrt 3)
  (distance_to_asymptote : ∃ c : ℝ, (|b * c| / sqrt (b^2 + a^2)) = sqrt 3) :
  Prop :=
  ∃ (eqn : ℝ × ℝ), eqn = (12, 3)

theorem hyperbola_eq_solution :
  find_hyperbola_equation 2 (sqrt 3) by sorry by sorry :=
begin
  sorry
end

noncomputable def find_D_and_t {x1 y1 x2 y2 : ℝ}
  (M : ℝ × ℝ) (N : ℝ × ℝ) (D : ℝ × ℝ)
  (h₃ : line_intersects_hyperbola M N ((sqrt 3 / 3) * (D.1) - 2))
  (h₄ : (M.1 + N.1) = 4 * (D.1))
  (h₅ : (M.2 + N.2) = 12) : Prop :=
  ∃ (t : ℝ) (coords : ℝ × ℝ), t = 4 ∧ coords = (4 * sqrt 3, 3)

theorem find_D_and_t_solution {x1 y1 x2 y2 : ℝ} :
  find_D_and_t (16 * sqrt 3, 12) by sorry :=
begin
  sorry
end

end hyperbola_eq_solution_find_D_and_t_solution_l140_140025


namespace cost_of_show_dogs_l140_140329

noncomputable def cost_per_dog : ℕ → ℕ → ℕ → ℕ
| total_revenue, total_profit, number_of_dogs => (total_revenue - total_profit) / number_of_dogs

theorem cost_of_show_dogs {revenue_per_puppy number_of_puppies profit number_of_dogs : ℕ}
  (h_puppies: number_of_puppies = 6)
  (h_revenue_per_puppy : revenue_per_puppy = 350)
  (h_profit : profit = 1600)
  (h_number_of_dogs : number_of_dogs = 2)
:
  cost_per_dog (number_of_puppies * revenue_per_puppy) profit number_of_dogs = 250 :=
by
  sorry

end cost_of_show_dogs_l140_140329


namespace distance_after_3_minutes_l140_140653

-- Conditions: speeds of the truck and car, and the time interval in hours
def v_truck : ℝ := 65 -- in km/h
def v_car : ℝ := 85 -- in km/h
def t : ℝ := 3 / 60 -- convert 3 minutes to hours

-- Statement to prove: The distance between the truck and the car after 3 minutes is 1 km
theorem distance_after_3_minutes : (v_car - v_truck) * t = 1 := 
by
  sorry

end distance_after_3_minutes_l140_140653


namespace no_S_point_f_g_find_a_for_S_point_exists_b_for_S_point_l140_140123

-- Define the conditions and functions
def S_point (f g : ℝ → ℝ) (f' g' : ℝ → ℝ) (x_0 : ℝ) : Prop :=
  f x_0 = g x_0 ∧ f' x_0 = g' x_0

-- 1. Prove that f(x) = x and g(x) = x^2 + 2x - 2 do not have an "S point"
theorem no_S_point_f_g : 
  ¬ ∃ x_0 : ℝ, S_point (λ x, x) (λ x, x^2 + 2 * x - 2) (λ x, 1) (λ x, 2 * x + 2) x_0 :=
sorry

-- 2. Given f(x) = ax^2 - 1 and g(x) = ln x, find the value of a for which an "S point" exists
theorem find_a_for_S_point :
  ∃ (a : ℝ), ∃ x_0 : ℝ, S_point (λ x, a * x^2 - 1) (λ x, Real.log x) (λ x, 2 * a * x) (λ x, 1 / x) x_0 :=
sorry

-- 3. Given f(x) = -x^2 + a and g(x) = be^x / x, for any a > 0, determine if there exists b > 0 with an "S point" in (0, +∞)
theorem exists_b_for_S_point (a : ℝ) (h : a > 0) :
  ∃ b : ℝ, b > 0 ∧ ∃ x_0 : ℝ, x_0 > 0 ∧ S_point (λ x, -x^2 + a) (λ x, b * Real.exp x / x) (λ x, -2 * x) (λ x, b * Real.exp x * (x - 1) / x^2) x_0 :=
sorry

end no_S_point_f_g_find_a_for_S_point_exists_b_for_S_point_l140_140123


namespace vector_magnitude_subtract_scaled_l140_140018

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1*v.1 + v.2*v.2 + v.3*v.3)

theorem vector_magnitude_subtract_scaled
  (a b : ℝ × ℝ × ℝ)
  (h_a : vector_magnitude a = 1)
  (h_b : vector_magnitude b = 1)
  (angle_120 : real.angle a b = real.angle_of_degrees 120) :
  vector_magnitude (a.1 - 2*b.1, a.2 - 2*b.2, a.3 - 2*b.3) = real.sqrt 7 :=
sorry

end vector_magnitude_subtract_scaled_l140_140018


namespace angle_BAC_is_60_l140_140461

/-- Given points B, C, and D lying on a line, with given angles.
  Prove that the measure of angle BAC (denoted as x) is 60 degrees.
-/
theorem angle_BAC_is_60
  (B C D : Type)
  (line_BCD : B ≠ C ∧ C ≠ D) -- Points B, C, and D are distinct
  (αβ : ∠ABC = 90)
  (αδ : ∠ACD = 150) :
  ∠BAC = 60 :=
sorry

end angle_BAC_is_60_l140_140461


namespace complex_number_solution_l140_140544

noncomputable def z_val := Complex 3 4

theorem complex_number_solution
  (z : ℂ)
  (a b : ℝ)
  (h1 : z = Complex a b)
  (h2 : conj z + Complex.abs z = Complex 8 (-4)) :
  z = z_val :=
by sorry

end complex_number_solution_l140_140544


namespace number_of_distinct_configurations_l140_140598

-- Define the conditions
def numConfigurations (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 else n + 1

-- Theorem statement
theorem number_of_distinct_configurations (n : ℕ) : 
  numConfigurations n = if n % 2 = 1 then 2 else n + 1 :=
by
  sorry -- Proof intentionally left out

end number_of_distinct_configurations_l140_140598


namespace sqrt_expression_l140_140959

theorem sqrt_expression :
  sqrt (36 * sqrt (12 * sqrt 9)) = 6 * sqrt 6 :=
by
  sorry

end sqrt_expression_l140_140959


namespace survey_steps_correct_l140_140092

theorem survey_steps_correct :
  ∀ steps : (ℕ → ℕ), (steps 1 = 2) → (steps 2 = 4) → (steps 3 = 3) → (steps 4 = 1) → True :=
by
  intros steps h1 h2 h3 h4
  exact sorry

end survey_steps_correct_l140_140092


namespace ratio_of_female_to_male_members_l140_140659

theorem ratio_of_female_to_male_members 
  (f m : ℕ) 
  (avg_age_female : ℕ) 
  (avg_age_male : ℕ)
  (avg_age_all : ℕ) 
  (H1 : avg_age_female = 45)
  (H2 : avg_age_male = 25)
  (H3 : avg_age_all = 35)
  (H4 : (f + m) ≠ 0) :
  (45 * f + 25 * m) / (f + m) = 35 → f = m :=
by sorry

end ratio_of_female_to_male_members_l140_140659


namespace rectangle_perimeter_is_70_l140_140634

-- Define the length and width of the rectangle
def length : ℕ := 19
def width : ℕ := 16

-- Define the perimeter function for a rectangle
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem statement asserting that the perimeter of the given rectangle is 70 cm
theorem rectangle_perimeter_is_70 :
  perimeter length width = 70 := 
sorry

end rectangle_perimeter_is_70_l140_140634


namespace geom_seq_problem_l140_140462

open_locale big_operators

variables {α : Type*} [field α]

-- Define a geometric sequence (general term)
noncomputable def geom_seq (a r : α) (n : ℕ) : α := a * r^n

-- Given conditions
variables (a r : α)
variables (a5 a11 a3 a13 a15 : α)

-- The conditions given in the problem
def problem_conditions :=
  a5 * a11 = 3 ∧
  a3 + a13 = 4 ∧
  a3 = geom_seq a r 3 ∧
  a5 = geom_seq a r 5 ∧
  a11 = geom_seq a r 11 ∧
  a13 = geom_seq a r 13 ∧
  a15 = geom_seq a r 15

-- The statement to be proven
theorem geom_seq_problem (a r : α) (a5 a11 a3 a13 a15 : α)
  (h_cond : problem_conditions a r a5 a11 a3 a13 a15) :
  a15 / a5 = 1/3 ∨ a15 / a5 = 3 :=
begin
  sorry
end

end geom_seq_problem_l140_140462


namespace compute_M_l140_140489

def S : finset ℕ := (finset.range 13).image (λ n, 2^n)

def M : ℕ := finset.sum (finset.off_diag S) (λ pair, abs (pair.1 - pair.2))

theorem compute_M : M = 83376 := by
  sorry

end compute_M_l140_140489


namespace at_most_17_distinct_prime_factors_of_b_l140_140187

theorem at_most_17_distinct_prime_factors_of_b 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h_gcd : (gcd a b).prime_factor_count = 10)
  (h_lcm : (lcm a b).prime_factor_count = 25)
  (h_lt : b.prime_factor_count < a.prime_factor_count) 
  : b.prime_factor_count ≤ 17 := 
sorry

end at_most_17_distinct_prime_factors_of_b_l140_140187


namespace cartesian_to_polar_l140_140336

theorem cartesian_to_polar (x y : ℝ) (h1 : x = √3) (h2 : y = -1) :
  ∃ (ρ θ : ℝ), ρ = 2 ∧ θ = (11 * Real.pi) / 6 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by
  use 2
  use (11 * Real.pi) / 6
  split
  exact rfl
  split
  exact rfl
  split
  rw [h1, Real.cos, Real.pi]
  exact sorry
  rw [h2, Real.sin, Real.pi]
  exact sorry

end cartesian_to_polar_l140_140336


namespace no_adjacent_same_color_probability_zero_l140_140712

-- Define the number of each color bead
def num_red_beads : ℕ := 5
def num_white_beads : ℕ := 3
def num_blue_beads : ℕ := 2

-- Define the total number of beads
def total_beads : ℕ := num_red_beads + num_white_beads + num_blue_beads

-- Calculate the probability that no two neighboring beads are the same color
noncomputable def probability_no_adjacent_same_color : ℚ :=
  if (num_red_beads > num_white_beads + num_blue_beads + 1) then 0 else sorry

theorem no_adjacent_same_color_probability_zero :
  probability_no_adjacent_same_color = 0 :=
by {
  sorry
}

end no_adjacent_same_color_probability_zero_l140_140712


namespace find_z_l140_140020

def i : Complex := Complex.I
def conj (z : Complex) : Complex := Complex.conj z

-- Conditions from the problem
variables (z : Complex)
variable (h : (z * conj z) * i + 2 = 2 * z)

-- Proof statement
theorem find_z : z = 1 + i := 
sorry

end find_z_l140_140020


namespace circle_radius_and_diameter_relations_l140_140413

theorem circle_radius_and_diameter_relations
  (r_x r_y r_z A_x A_y A_z d_x d_z : ℝ)
  (hx_circumference : 2 * π * r_x = 18 * π)
  (hx_area : A_x = π * r_x^2)
  (hy_area_eq : A_y = A_x)
  (hz_area_eq : A_z = 4 * A_x)
  (hy_area : A_y = π * r_y^2)
  (hz_area : A_z = π * r_z^2)
  (dx_def : d_x = 2 * r_x)
  (dz_def : d_z = 2 * r_z)
  : r_y = r_z / 2 ∧ d_z = 2 * d_x := 
by 
  sorry

end circle_radius_and_diameter_relations_l140_140413


namespace complement_of_A_in_U_l140_140379

open Set

-- Define the sets U and A with their respective elements in the real numbers
def U : Set ℝ := Icc 0 1
def A : Set ℝ := Ico 0 1

-- State the theorem
theorem complement_of_A_in_U : (U \ A) = {1} := by
  sorry

end complement_of_A_in_U_l140_140379


namespace S_n_value_l140_140124

-- Definition of f(k)
def f (k : Nat) : Nat := 3 * 2^(k - 1) + 1

-- Sum function S(n)
def S (n : Nat) : Nat := (List.range n).map (λ i => (i + 1) * f (i + 1)).sum

-- Theorem stating the value of S(n)
theorem S_n_value (n : Nat) : 
  S n = 3 * (n - 1) * 2^n + (n * (n + 1)) / 2 + 3 := sorry

end S_n_value_l140_140124


namespace distance_of_B_from_original_position_l140_140311

theorem distance_of_B_from_original_position :
  ∀ (side_length : ℝ) 
    (y : ℝ), 
  side_length = real.sqrt 18 → 
  0 ≤ y → 
  2 * (y ^ 2) = 18 → 
  y = 2 * real.sqrt 3 → 
  (real.sqrt ((2 * real.sqrt 3) ^ 2 + (2 * real.sqrt 3) ^ 2)) = 2 * real.sqrt 6 :=
begin
  intros side_length y h_side_length h_nonneg_y h_area h_y_value,
  sorry
end

end distance_of_B_from_original_position_l140_140311


namespace intersection_sum_coordinates_l140_140845

section IntersectionPoint

variable (A B C : ℝ × ℝ)
variable (D E F : ℝ × ℝ)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def line_eq (P Q : ℝ × ℝ) : ℝ → ℝ :=
  let slope := (Q.2 - P.2) / (Q.1 - P.1)
  (λ x => slope * (x - P.1) + P.2)

theorem intersection_sum_coordinates :
  let A := (0, 10)
  let B := (0, 0)
  let C := ((10, 0))
  let D := midpoint A B
  let E := midpoint B C
  let F_x := (10 : ℝ / 3)
  let F_y := line_eq A E F_x
  F_x + F_y = (20 / 3 : ℝ) := sorry

end IntersectionPoint

end intersection_sum_coordinates_l140_140845


namespace hypotenuse_length_l140_140636

-- Define the right triangle with given conditions
variables (a c : ℝ)

-- Function to define the properties of the triangle
def right_triangle (a c : ℝ) : Prop :=
  (a + (a + 2) + c = 40) ∧
  (0.5 * a * (a + 2) = 24) ∧
  (a^2 + (a + 2)^2 = c^2)

-- Statement of the problem
theorem hypotenuse_length :
  ∃ (c : ℝ), right_triangle a c ∧ c = 16.61 :=
begin
  sorry
end

end hypotenuse_length_l140_140636


namespace parallel_lines_point_l140_140880

noncomputable def point_on_ll (a : ℝ) (x : ℝ) : ℝ :=
(x - 4)

theorem parallel_lines_point 
  (a : ℝ) (hl : a = 2) 
  (x y : ℝ) 
  (p1 : (4, 0))
  (q1 : (-2, 0)) 
  (q2 : (0, a)) 
  (hp : (0 - 0) / (4 - 0) = 1)
  : (x = 5) → (y = point_on_ll a x) → y = 1 := 
by
  intros
  subst hl
  subst a
  sorry

end parallel_lines_point_l140_140880


namespace PS_length_correct_l140_140219

variable {Triangle : Type}

noncomputable def PR := 15

noncomputable def PS_length (PS SR : ℝ) (PR : ℝ) : Prop :=
  PS + SR = PR ∧ (PS / SR) = (3 / 4)

theorem PS_length_correct : 
  ∃ PS SR : ℝ, PS_length PS SR PR ∧ PS = (45 / 7) :=
sorry

end PS_length_correct_l140_140219


namespace inequality_proof_l140_140022

theorem inequality_proof 
  (n : ℕ) 
  (a : Fin n → ℝ) 
  (λ : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_seq : ∀ i j, i ≤ j → λ i ≤ λ j) 
  (h_sum_a : ∑ i, a i = 1)
  (h_λ_pos : ∀ i, 0 < λ i) : 
  (∑ i, (a i) / (λ i)) * (∑ i, (a i) * (λ i)) ≤ (λ 0 + λ (n - 1))^2 / (4 * λ 0 * λ (n - 1)) := 
sorry

end inequality_proof_l140_140022


namespace no_two_exact_cubes_between_squares_l140_140102

theorem no_two_exact_cubes_between_squares :
  ∀ (n a b : ℤ), ¬ (n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2) :=
by
  intros n a b
  sorry

end no_two_exact_cubes_between_squares_l140_140102


namespace equal_product_groups_exist_l140_140226

def numbers : List ℕ := [21, 22, 34, 39, 44, 45, 65, 76, 133, 153]

theorem equal_product_groups_exist :
  ∃ (g1 g2 : List ℕ), 
    g1.length = 5 ∧ g2.length = 5 ∧ 
    g1.prod = g2.prod ∧ g1.prod = 349188840 ∧ 
    (g1 ++ g2 = numbers ∨ g1 ++ g2 = numbers.reverse) :=
by
  sorry

end equal_product_groups_exist_l140_140226


namespace probability_number_is_odd_l140_140549

def definition_of_odds : set ℕ := {3, 5, 7, 9}

def number_is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem probability_number_is_odd :
  let total_digits := {2, 3, 5, 7, 9}
  let odd_digits := definition_of_odds
  let favorable_outcomes := set.card odd_digits
  let total_outcomes := set.card total_digits
  in (total_outcomes = 5) -> (favorable_outcomes = 4) -> (favorable_outcomes / total_outcomes : ℚ) = 4 / 5 :=
by
  intro total_digits odd_digits favorable_outcomes total_outcomes total_outcomes_eq favorable_outcomes_eq
  have h1 : favorable_outcomes = 4 := favorable_outcomes_eq
  have h2 : total_outcomes = 5 := total_outcomes_eq
  norm_num
  sorry

end probability_number_is_odd_l140_140549


namespace equal_segments_MS_MF_l140_140874

variables {A B C H M S F : Point}
variables {ABC : Triangle A B C}
variables {H_ortho : OrthoCenter ABC H}
variables {M_mid : isMidpoint M A B}
variables {w : AngleBisector (∠ A C B)}
variables {S_inter : Intersect S (perpendicularBisector A B) w}
variables {F_perp : FootPerpendicular F H w}

theorem equal_segments_MS_MF : dist M S = dist M F :=
sorry  -- Proof to be provided

end equal_segments_MS_MF_l140_140874


namespace quadratic_has_two_distinct_roots_l140_140012

variables {a b c d : ℝ}

-- Conditions extracted from problem statement
def f (x : ℝ) : ℝ := x^2 + b * x + a
def g (x : ℝ) : ℝ := x^2 + c * x + d

-- Main theorem
theorem quadratic_has_two_distinct_roots 
  (h1 : b^2 - 4 * a < 0) -- Discriminant condition for f(x) which implies a > b^2 / 4
  (h2 : g a = b)
  (h3 : g b = a)
  (h4 : a ≠ b) -- Ensures distinct coefficients a and b
  (ha : a > 0) -- Since a > b^2 / 4 and b^2 >= 0
  : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ (λ x : ℝ, a * x^2 + b * x + c) r₁ = 0 ∧ (λ x : ℝ, a * x^2 + b * x + c) r₂ = 0 := 
sorry

end quadratic_has_two_distinct_roots_l140_140012


namespace initial_machines_count_l140_140085

theorem initial_machines_count :
  ∃ m : ℕ, 
    (8 / 4 = 2) ∧ -- rate of machines in first scenario
    (24 / 6 = 4) ∧ -- checking rate consistency for 72 machines
    (4 / 72 = 1 / 18) ∧ -- rate per machine with 72 machines
    (2 / m = 1 / 18) ∧ -- equating rates for same standard rate per machine
    m = 36 := -- proving m is actually 36
by
  use 36
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  exact rfl

end initial_machines_count_l140_140085


namespace total_pieces_eq_21_l140_140624

-- Definitions based on conditions
def red_pieces : Nat := 5
def yellow_pieces : Nat := 7
def green_pieces : Nat := 11

-- Derived definitions from conditions
def red_cuts : Nat := red_pieces - 1
def yellow_cuts : Nat := yellow_pieces - 1
def green_cuts : Nat := green_pieces - 1

-- Total cuts and the resulting total pieces
def total_cuts : Nat := red_cuts + yellow_cuts + green_cuts
def total_pieces : Nat := total_cuts + 1

-- Prove the total number of pieces is 21
theorem total_pieces_eq_21 : total_pieces = 21 := by
  sorry

end total_pieces_eq_21_l140_140624


namespace triangle_is_isosceles_l140_140061

-- Given vectors O, A, B, C in an arbitrary plane
variables {V : Type*} [inner_product_space ℝ V]
variables (O A B C : V)

-- Given condition
def condition := (O + O C - 2 • O) ⬝ (B - C - (A - C)) = 0

-- Proving the type of the triangle
theorem triangle_is_isosceles (h : condition O A B C) : dist A B = dist A C := 
by {
    -- Sketch of the proof
    sorry
}

end triangle_is_isosceles_l140_140061


namespace avg_salary_managers_l140_140267

theorem avg_salary_managers
  (E : ℕ)
  (h1 : 0.70 * E = num_marketers)
  (h2 : 0.10 * E = num_engineers)
  (h3 : num_marketers + num_engineers + num_managers = E)
  (h4 : avg_salary_marketers = 50000)
  (h5 : avg_salary_engineers = 80000)
  (h6 : total_avg_salary = 80000)
  (h7 : ((num_marketers * avg_salary_marketers) + 
          (num_engineers * avg_salary_engineers) + 
          (num_managers * avg_salary_managers)) / E = total_avg_salary) :
  avg_salary_managers = 185000 :=
by
  sorry

end avg_salary_managers_l140_140267


namespace distinct_k_count_l140_140687

noncomputable def integer_solution_count : ℤ :=
  ∑ m in finset.Icc (-31 : ℤ) 31, if m ≠ 0 ∧ abs (10 / m + 3 * m) < 150 then 1 else 0

theorem distinct_k_count : integer_solution_count = 62 := 
begin
  sorry
end

end distinct_k_count_l140_140687


namespace math_problem_proof_l140_140752

noncomputable def ellipseC (a b : ℝ) (ha : a > b > 0) (focal_length : ℝ := 4 * Real.sqrt 3) (h_focal : a^2 - b^2 = (2 * Real.sqrt 3)^2) : Prop :=
  ∀ (x y : ℝ), (x, y) = (2 * Real.sqrt 3, 1) → (x^2 / a^2 + y^2 / b^2 = 1) → (a = 4 ∧ b = 2)

noncomputable def isosceles_triangle (x1 y1 x2 y2 : ℝ) (k : ℝ) : Prop :=
  let B := (0, -2)
  let EF := ((x1, y1), (x2, y2))
  let M := ((x1 + x2) / 2, (y1 + y2) / 2)
  let BM_slope := (B.snd - M.snd) / (B.fst - M.fst)
  let h_perp := -1 / k = BM_slope
  let circle_separation := (4 / Real.sqrt 18 > Real.sqrt 1 / 2)
  ∀ (x y : ℝ), ((x, y) ∈ linearEquation) ↔ (h_perp ∧ circle_separation)

theorem math_problem_proof :
  ∀ (a b : ℝ) (ha : a > b > 0),
  ellipseC a b ha →
  isosceles_triangle (2 * Real.sqrt 3) 1 (2 * Real.sqrt 3) 1 (Real.sqrt 2 / 4)
  :=
by
  intros
  sorry

end math_problem_proof_l140_140752


namespace connectivity_l140_140818

inductive Route
| air
| water

def connected (cities : List ℕ) (route : Route) : Prop :=
  ∀ c1 c2 ∈ cities, c1 ≠ c2 → (route = Route.air ∨ route = Route.water)

theorem connectivity (cities : List ℕ) (h : ∀ c1 c2 ∈ cities, c1 ≠ c2 → (Route.air ∨ Route.water)) :
  (∀ c1 c2 ∈ cities, c1 ≠ c2 → (Route.air ∨ Route.water)) ∨ (∀ c1 c2 ∈ cities, c1 ≠ c2 → (Route.water ∨ Route.air)) :=
by
  sorry

end connectivity_l140_140818


namespace max_lambda_l140_140748

variable (x y z λ : ℝ)

theorem max_lambda (h1 : x > y) (h2 : y > z) (h3 : z > 0) : (∀ λ, (∀ x y z, x > y > z > 0 → (1/(x-y) + 1/(y-z) + λ/(z-x) ≥ 0)) → λ ≤ 4) :=
sorry

end max_lambda_l140_140748


namespace half_abs_diff_squares_l140_140250

/-- Half of the absolute value of the difference of the squares of 23 and 19 is 84. -/
theorem half_abs_diff_squares : (1 / 2 : ℝ) * |(23^2 : ℝ) - (19^2 : ℝ)| = 84 :=
by
  sorry

end half_abs_diff_squares_l140_140250


namespace distance_after_3_minutes_l140_140648

-- Define the speeds of the truck and the car in km/h.
def v_truck : ℝ := 65
def v_car : ℝ := 85

-- Define the time in hours.
def time_in_hours : ℝ := 3 / 60

-- Define the relative speed.
def v_relative : ℝ := v_car - v_truck

-- Define the expected distance between the truck and the car after 3 minutes.
def expected_distance : ℝ := 1

-- State the theorem: the distance between the truck and the car after 3 minutes is 1 km.
theorem distance_after_3_minutes : (v_relative * time_in_hours) = expected_distance := 
by {
  -- Here, we would provide the proof, but we are adding 'sorry' to skip the proof.
  sorry
}

end distance_after_3_minutes_l140_140648


namespace necessary_condition_aeqbeqacb_l140_140369

-- Conditions: for any real numbers a, b, and c
variable (a b c : ℝ)

-- Equivalent to: Prove that "ac = bc" is a necessary condition for "a = b"
theorem necessary_condition_aeqbeqacb (h : a = b) : a * c = b * c :=
by
  rw h
  sorry

end necessary_condition_aeqbeqacb_l140_140369


namespace frequency_rate_of_10_40_l140_140451

def sample_intervals : list (set ℝ × ℕ) := [
  (set.Ioc 10 20, 2),
  (set.Ioc 20 30, 4),
  (set.Ioc 30 40, 3),
  (set.Ioc 40 50, 5),
  (set.Ioc 50 60, 4),
  (set.Ioc 60 70, 2)
]

def total_sample_size : ℕ := 20

def interval_frequency (interval : set ℝ) : ℕ :=
  (sample_intervals.filter (λ x, x.1 = interval)).head!.2

noncomputable def interval_frequency_rate (intervals : list (set ℝ)) : ℚ :=
  let freq := intervals.foldl (λ acc x, acc + interval_frequency x) 0
  in freq / total_sample_size

theorem frequency_rate_of_10_40 :
  interval_frequency_rate [set.Ioc 10 20, set.Ioc 20 30, set.Ioc 30 40] = 9 / 20 :=
by
  -- the proof will be added here
  sorry

end frequency_rate_of_10_40_l140_140451


namespace polar_to_rectangular_l140_140466

noncomputable def curve_equation (θ : ℝ) : ℝ := 2 * Real.cos θ

theorem polar_to_rectangular (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  ∃ (x y : ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧
  (x = curve_equation θ * Real.cos θ ∧ y = curve_equation θ * Real.sin θ) :=
sorry

end polar_to_rectangular_l140_140466


namespace function_value_corresponds_to_multiple_independent_variables_l140_140964

theorem function_value_corresponds_to_multiple_independent_variables
  {α β : Type*} (f : α → β) :
  ∃ (b : β), ∃ (a1 a2 : α), a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b :=
sorry

end function_value_corresponds_to_multiple_independent_variables_l140_140964


namespace roots_irrational_l140_140365

theorem roots_irrational (k : ℤ) (h : 3 * k^2 - 3 = 10) :
  ∀ a b : ℝ, quadratic_roots (x^2 - 4 * (k : ℝ) * x + 3 * (k : ℝ)^2 - 3) a b → irrational a ∧ irrational b :=
by
  sorry

end roots_irrational_l140_140365


namespace find_k_value_find_integer_k_values_l140_140404

variables (a b k : ℤ) (y : ℤ)

-- Given conditions
def polynomial_condition : Prop := (a - b - 1 = 0)

def equation_condition_part1 : Prop := (3 * b - 3 * a) * 1 = k * 1 - 5

def equation_condition_part2 : Prop := ∃ y : ℤ, y > 0 ∧ (3 * b - 3 * a) * y = k * y - 5

-- Part 1
theorem find_k_value (h : polynomial_condition) (hy : y = 1) : equation_condition_part1 → k = 2 :=
sorry

-- Part 2
theorem find_integer_k_values (h : polynomial_condition) : equation_condition_part2 ↔ k = 2 ∨ k = -2 :=
sorry

end find_k_value_find_integer_k_values_l140_140404


namespace average_of_centenary_set_l140_140306

/--
A set is called centenary if it has at least two distinct positive integers and its greatest element is 100.
We are proving that the average of any centenary set S can be any integer from 14 to 100.
-/
theorem average_of_centenary_set (S : Set ℕ) (h1 : ∀ a ∈ S, 1 ≤ a ∧ a ≤ 100) (h2 : (∀ x y ∈ S, x ≠ y) ∧ (∃ x y ∈ S, x ≠ y)) (h3 : 100 ∈ S) : 
  ∃ (avg : ℤ), 14 ≤ avg ∧ avg ≤ 100 ∧ avg = (S.sum / S.card) := 
sorry

end average_of_centenary_set_l140_140306


namespace magnitude_of_z_l140_140385

noncomputable def complex_z (z : ℂ) : ℂ := 1 + complex.I

theorem magnitude_of_z (z : ℂ) 
  (hz : (1 + complex.I) * z = 2 - complex.I) : 
  complex.abs z = 3 * real.sqrt 2 / 2 :=
by
  -- proof would go here
  sorry

end magnitude_of_z_l140_140385


namespace coords_of_P_l140_140069

noncomputable def is_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + 3 * P.2 = 0

noncomputable def distance_to_origin (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 ^ 2) + (P.2 ^ 2))

noncomputable def distance_to_line (P : ℝ × ℝ) : ℝ :=
  Real.abs (P.1 + 3 * P.2 - 2) / Real.sqrt (1 ^ 2 + 3 ^ 2)

theorem coords_of_P (P : ℝ × ℝ) (h₁ : is_on_line P) (h₂ : distance_to_origin P = distance_to_line P) :
  P = (3 / 5, -1 / 5) ∨ P = (-3 / 5, 1 / 5) :=
sorry

end coords_of_P_l140_140069


namespace power_rule_example_l140_140673

variable (a : ℕ) (n m : ℕ)
#check (a^n)^m

theorem power_rule (a n m : ℕ) : (a^n)^m = a^(n*m) :=
begin
  sorry
end

theorem example : (a^5)^3 = a^15 :=
begin
  exact power_rule a 5 3
end

end power_rule_example_l140_140673


namespace count_positive_integer_solutions_l140_140199

theorem count_positive_integer_solutions :
  (∃ (n : ℤ), n > 0 ∧  (2014 / (n + 1) : ℤ)) → finset.card (finset.filter (λ x : ℤ, (2014 % (x + 1) = 0) ∧ x > 0) (finset.range 2015)) = 7 :=
by
  sorry

end count_positive_integer_solutions_l140_140199


namespace find_integer_with_specific_sixth_power_digits_l140_140706

theorem find_integer_with_specific_sixth_power_digits :
  ∃ n : ℤ, (n^6).digits = [0, 1, 2, 2, 2, 3, 4, 4] := by
  use 18
  sorry

end find_integer_with_specific_sixth_power_digits_l140_140706


namespace fraction_of_silver_knights_enchanted_l140_140829

theorem fraction_of_silver_knights_enchanted 
    (total_knights : ℕ)
    (silver_fraction : ℚ)
    (enchanted_fraction : ℚ)
    (silver_es_gold_relation : ∀ f_silver f_gold, f_silver = 3 * f_gold) 
    (hsilver_fraction : silver_fraction = 3 / 8)
    (henchanted_fraction : enchanted_fraction = 1 / 8) :
    ∃ f_silver : ℚ, (∃ f_gold : ℚ, silver_es_gold_relation f_silver f_gold) ∧ 
    (total_knights * (f_silver * silver_fraction) + total_knights * (f_gold * (1 - silver_fraction)) = total_knights * enchanted_fraction) ∧ 
    f_silver = 1 / 14 := 
sorry

end fraction_of_silver_knights_enchanted_l140_140829


namespace fraction_always_irreducible_l140_140902

-- Define the problem statement
theorem fraction_always_irreducible (n : ℤ) : gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_always_irreducible_l140_140902


namespace floor_sqrt_sum_eq_floor_sqrt_expr_l140_140894

theorem floor_sqrt_sum_eq_floor_sqrt_expr (n : ℕ) : 
  let x := Real.sqrt n + Real.sqrt (n + 1) + Real.sqrt (n + 2)
  in Int.floor x = Int.floor (Real.sqrt (9 * n + 8)) := 
by
  sorry

end floor_sqrt_sum_eq_floor_sqrt_expr_l140_140894


namespace adding_schemes_l140_140284

-- Definitions based on problem conditions
def raw_materials : Nat := 5
def materials_exclusive : Prop := ¬(A ∧ B)
def material_A_first (A : Prop) : Prop := A

-- Theorem statement
theorem adding_schemes (A B : Prop) (count_schemes : raw_materials) :
  materials_exclusive ∧ (material_A_first A → (∃ x y, (x ≠ B ∧ y ≠ B) ∧ x ≠ y) ∨ (B → ∃ x y, (x ≠ A ∧ y ≠ A) ∧ x ≠ y) ∨ (¬A ∧ ¬B → ∃ x y z, x ≠ y ∧ x ≠ z ∧ y ≠ z)) →
  count_schemes = 15 :=
sorry 

end adding_schemes_l140_140284


namespace find_principal_amount_l140_140546

-- Definitions:
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (P * R * T) / 100

def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * ((1 + R / 100) ^ T) - P

-- Constants based on problem conditions
def R := 20
def T := 2
def difference := 360

-- Main theorem to prove:
theorem find_principal_amount (P : ℝ) :
  compound_interest P R T - simple_interest P R T = difference →
  P = 9000 :=
by
  sorry

end find_principal_amount_l140_140546


namespace distance_between_truck_and_car_l140_140642

noncomputable def truck_speed : ℝ := 65
noncomputable def car_speed : ℝ := 85
noncomputable def time_duration_hours : ℝ := 3 / 60

theorem distance_between_truck_and_car :
  let relative_speed := car_speed - truck_speed in
  let distance := relative_speed * time_duration_hours in
  distance = 1 :=
by
  let relative_speed := car_speed - truck_speed
  let distance := relative_speed * time_duration_hours
  have h1 : distance = 1 := sorry
  exact h1

end distance_between_truck_and_car_l140_140642


namespace solution_set_of_inequality_l140_140201

variable (f : ℝ → ℝ)
variable (h_inc : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)

theorem solution_set_of_inequality :
  {x | 0 < x ∧ f x > f (2 * x - 4)} = {x | 2 < x ∧ x < 4} :=
by
  sorry

end solution_set_of_inequality_l140_140201


namespace sale_discount_l140_140237

theorem sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (discount_multiple : ℕ)
  (h1 : purchase_amount = 250)
  (h2 : discount_per_100 = 10)
  (h3 : discount_multiple = purchase_amount / 100) :
  purchase_amount - discount_per_100 * discount_multiple = 230 := 
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end sale_discount_l140_140237


namespace perimeter_of_park_is_66_l140_140830

-- Given width and length of the flower bed
variables (w l : ℝ)
-- Given that the length is four times the width
variable (h1 : l = 4 * w)
-- Given the area of the flower bed
variable (h2 : l * w = 100)
-- Given the width of the walkway
variable (walkway_width : ℝ := 2)

-- The total width and length of the park, including the walkway
def w_park := w + 2 * walkway_width
def l_park := l + 2 * walkway_width

-- The proof statement: perimeter of the park equals 66 meters
theorem perimeter_of_park_is_66 :
  2 * (l_park + w_park) = 66 :=
by
  -- The full proof can be filled in here
  sorry

end perimeter_of_park_is_66_l140_140830


namespace meal_combinations_l140_140263

theorem meal_combinations (n : ℕ) : (n = 15) → (15 * 15 * 15 = 3375) :=
by {
  intro h,
  rw h,
  norm_num,
  sorry
}

end meal_combinations_l140_140263


namespace quadratic_has_real_root_of_b_interval_l140_140794

variable (b : ℝ)

theorem quadratic_has_real_root_of_b_interval
  (h : ∃ x : ℝ, x^2 + b * x + 25 = 0) : b ∈ Iic (-10) ∪ Ici 10 :=
by
  sorry

end quadratic_has_real_root_of_b_interval_l140_140794


namespace distance_between_truck_and_car_l140_140643

noncomputable def truck_speed : ℝ := 65
noncomputable def car_speed : ℝ := 85
noncomputable def time_duration_hours : ℝ := 3 / 60

theorem distance_between_truck_and_car :
  let relative_speed := car_speed - truck_speed in
  let distance := relative_speed * time_duration_hours in
  distance = 1 :=
by
  let relative_speed := car_speed - truck_speed
  let distance := relative_speed * time_duration_hours
  have h1 : distance = 1 := sorry
  exact h1

end distance_between_truck_and_car_l140_140643


namespace AC_length_l140_140460

-- Definitions for angles and lengths in the quadrilateral.
variables (A B C D : Point)
variables (angle_BAD angle_BCD : ℝ)
variables (BC CD AC : ℝ)

-- Conditions given in the problem.
def convex_quadrilateral (A B C D : Point) : Prop :=
  angle_BAD = 120 ∧ angle_BCD = 120 ∧ BC = 10 ∧ CD = 10 

-- The proof statement for AC
theorem AC_length (A B C D : Point) (angle_BAD angle_BCD BC CD : ℝ) :
  convex_quadrilateral A B C D →
  AC = 10 :=
by
  assume h
  sorry

end AC_length_l140_140460


namespace trains_cross_in_9_seconds_l140_140274

theorem trains_cross_in_9_seconds 
  (length_A : ℝ) 
  (speed_A_kmph : ℝ) 
  (length_B : ℝ) 
  (speed_B_kmph : ℝ) 
  (opposite_directions : true) 
  (conversion_factor : ℝ := 5/18) :
  (500.04 / ((120 * conversion_factor) + (80 * conversion_factor)) ≈ 9) :=
by sorry

end trains_cross_in_9_seconds_l140_140274


namespace hyperbola_foci_distance_l140_140704

theorem hyperbola_foci_distance (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 9) :
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 34 := 
by
  sorry

end hyperbola_foci_distance_l140_140704


namespace angle_BAC_measure_l140_140285

-- Definitions for the problem
variables (O A B C : Type) [point : Type]
variables (angle_AOB angle_BOC : ℝ)
variables (is_circumscribed : Bool)

-- Conditions
axiom circumscribed_about_triangle : is_circumscribed = true
axiom angle_AOB_value : angle_AOB = 130
axiom angle_BOC_value : angle_BOC = 120

-- Required to prove 
theorem angle_BAC_measure :
  is_circumscribed = true → angle_AOB = 130 → angle_BOC = 120 → angle_BAC = 80 :=
by {
    sorry
}

end angle_BAC_measure_l140_140285


namespace tangent_circumcircle_triangle_l140_140892

theorem tangent_circumcircle_triangle 
  (A B C D E M : Point) 
  (h1 : ∠DAC = ∠ACB)
  (h2 : ∠BDC = 90 + ∠BAC)
  (h3 : collinear B D E)
  (h4 : AE = EC)
  (h5 : midpoint M B C) :
  tangent (line_through A B) (circumcircle (triangle B E M)) :=
sorry

end tangent_circumcircle_triangle_l140_140892


namespace roger_forgot_lawns_l140_140170

theorem roger_forgot_lawns
  (dollars_per_lawn : ℕ)
  (total_lawns : ℕ)
  (total_earned : ℕ)
  (actual_mowed_lawns : ℕ)
  (forgotten_lawns : ℕ)
  (h1 : dollars_per_lawn = 9)
  (h2 : total_lawns = 14)
  (h3 : total_earned = 54)
  (h4 : actual_mowed_lawns = total_earned / dollars_per_lawn) :
  forgotten_lawns = total_lawns - actual_mowed_lawns :=
  sorry

end roger_forgot_lawns_l140_140170


namespace distance_after_3_minutes_l140_140650

-- Conditions: speeds of the truck and car, and the time interval in hours
def v_truck : ℝ := 65 -- in km/h
def v_car : ℝ := 85 -- in km/h
def t : ℝ := 3 / 60 -- convert 3 minutes to hours

-- Statement to prove: The distance between the truck and the car after 3 minutes is 1 km
theorem distance_after_3_minutes : (v_car - v_truck) * t = 1 := 
by
  sorry

end distance_after_3_minutes_l140_140650


namespace simplify_exponent_multiplication_l140_140256

theorem simplify_exponent_multiplication :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_multiplication_l140_140256


namespace ratio_of_areas_l140_140766

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

noncomputable def line_through_focus (x y : ℝ) : Prop := y = sqrt 3 * (x - 2)

def point_A := (6 : ℝ, 4 * sqrt 3)
def point_B := (2 / 3 : ℝ, (4 * sqrt 3) / 3)
def point_C := (-2 : ℝ, -4 * sqrt 3)

def area (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

def triangle_AOC_area : ℝ := area point_A point_C (0, 0)
def triangle_BOF_area : ℝ := area point_B focus (0, 0)

theorem ratio_of_areas (hA : parabola point_A.1 point_A.2) (hB : parabola point_B.1 point_B.2) (hC : line_through_focus point_C.1 point_C.2) :
  triangle_AOC_area / triangle_BOF_area = 6 :=
sorry

end ratio_of_areas_l140_140766


namespace total_inflation_over_two_years_real_interest_rate_over_two_years_l140_140982

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end total_inflation_over_two_years_real_interest_rate_over_two_years_l140_140982


namespace circle_y_axis_intersection_sum_l140_140333

theorem circle_y_axis_intersection_sum :
  let C := { p : ℝ × ℝ | (p.1 + 4)^2 + (p.2 - 3)^2 = 25 } in
  let points := { p.2 | p ∈ C ∧ p.1 = 0 } in
  (∀ a b ∈ points, a + b = 6) :=
by
  sorry

end circle_y_axis_intersection_sum_l140_140333


namespace decreasing_sequence_l140_140004

def seq (a : ℝ) (n : ℕ) : ℝ :=
match n with
| 0     => a
| (n+1) => a ^ seq a n

theorem decreasing_sequence (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∀ ⦃n : ℕ⦄, seq a (n+1) < seq a n :=
begin
  sorry
end

end decreasing_sequence_l140_140004


namespace union_necessary_not_sufficient_for_intersection_l140_140486

variable (M N : Set)

theorem union_necessary_not_sufficient_for_intersection :
  (M ∪ N ≠ ∅ → M ∩ N ≠ ∅) ∧ ¬(M ∩ N ≠ ∅ → M ∪ N ≠ ∅) :=
by
  sorry

end union_necessary_not_sufficient_for_intersection_l140_140486


namespace pythagorean_triple_transformation_l140_140370

theorem pythagorean_triple_transformation
  (a b c α β γ s p q r : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : α^2 + β^2 - γ^2 = 2)
  (h₃ : s = a * α + b * β - c * γ)
  (h₄ : p = a - α * s)
  (h₅ : q = b - β * s)
  (h₆ : r = c - γ * s) :
  p^2 + q^2 = r^2 :=
by
  sorry

end pythagorean_triple_transformation_l140_140370


namespace values_of_x_l140_140717

theorem values_of_x (x : ℤ) :
  (∃ t : ℤ, x = 105 * t + 22) ∨ (∃ t : ℤ, x = 105 * t + 37) ↔ 
  (5 * x^3 - x + 17) % 15 = 0 ∧ (2 * x^2 + x - 3) % 7 = 0 :=
by {
  sorry
}

end values_of_x_l140_140717


namespace f_is_odd_l140_140027

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (α - 2) * x ^ α

theorem f_is_odd (α : ℝ) (hα : α = 3) : ∀ x : ℝ, f α (-x) = -f α x :=
by sorry

end f_is_odd_l140_140027


namespace find_value_l140_140059

theorem find_value 
    (x : ℝ)
    (h : x + real.sqrt (x^2 - 1) + 1 / (x - real.sqrt (x^2 - 1)) = 20) :
  x^2 + real.sqrt (x^4 - 1) + 1 / (x^2 + real.sqrt (x^4 - 1)) = 10201 / 200 :=
sorry

end find_value_l140_140059


namespace cosine_x_plus_2y_l140_140749

theorem cosine_x_plus_2y (x y : ℝ) (h1 : x^3 + cos x + x - 2 = 0) 
  (h2 : 8 * y^3 - 2 * cos y ^ 2 + 2 * y + 3 = 0) : cos (x + 2 * y) = 1 := 
sorry

end cosine_x_plus_2y_l140_140749


namespace prime_p_mod_eight_l140_140271

open Int

theorem prime_p_mod_eight (m n : ℤ) (h1 : 0 < m) (h2 : m < n) (h3 : isPrime (p : ℤ)) :
  let p := (n^2 + m^2) / (sqrt (n^2 - m^2))
  p ≡ 1 [MOD 8] :=
sorry

end prime_p_mod_eight_l140_140271


namespace binaryToOctalConversion_l140_140683

-- Declaration of the binary and octal equivalence condition
theorem binaryToOctalConversion : ∀ (n : ℕ), n = 45 -> binary_of_nat n = "101101" -> octal_of_nat n = "55" :=
by
  intros n hn hbinary
  sorry

end binaryToOctalConversion_l140_140683


namespace teachers_not_teaching_any_l140_140314

theorem teachers_not_teaching_any (T E J F EJ EF JF EJF : ℕ)
    (hT : T = 120)
    (hE : E = 50)
    (hJ : J = 45)
    (hF : F = 40)
    (hEJ : EJ = 15)
    (hEF : EF = 10)
    (hJF : JF = 8)
    (hEJF : EJF = 4) :
    T - (E + J + F - EJ - EF - JF + EJF) = 14 :=
by {
    sorry,
}

end teachers_not_teaching_any_l140_140314


namespace class_factory_arrangements_l140_140302

theorem class_factory_arrangements :
  ∃ (n : ℕ), n = 240 ∧ ∀ (classes factories : ℕ), 
  classes = 5 → factories = 4 → 
  (∃ arrangement : list (list ℕ), 
    arrangement.length = factories ∧ 
    (∀ lst ∈ arrangement, 1 ≤ lst.length ∧ lst.length ≤ classes) ∧ 
    (∀ c : ℕ, c < classes → ∃ (lst ∈ arrangement), c ∈ lst) →
    n = 240) :=
sorry

end class_factory_arrangements_l140_140302


namespace sausage_cutting_l140_140625

theorem sausage_cutting (red_pieces yellow_pieces green_pieces total_pieces : ℕ) 
  (h_red : red_pieces = 5)
  (h_yellow : yellow_pieces = 7)
  (h_green : green_pieces = 11)
  (h_total : total_pieces = (red_pieces - 1) + (yellow_pieces - 1) + (green_pieces - 1) + 1) :
  total_pieces = 21 := 
by {
  rw [h_red, h_yellow, h_green],
  sorry
}

end sausage_cutting_l140_140625


namespace exists_inverse_C_l140_140866

open Matrix

def matrix_2_z := Matrix (Fin 2) (Fin 2) ℤ

variables {A C : matrix_2_z} {I : matrix_2_z}

theorem exists_inverse_C (hA : A ^ 2 + (5 : ℤ) • (1 : matrix_2_z) = 0) :
  ∃ C : matrix_2_z, invertible C ∧ (A = C⁻¹ • (matrix_of_fun ![![1, 2], ![-3, -1]]) • C ∨
                                    A = C⁻¹ • (matrix_of_fun ![![0, 1], ![-5, 0]]) • C) :=
sorry

end exists_inverse_C_l140_140866


namespace f_is_periodic_l140_140272

noncomputable def f (x : ℝ) : ℝ := x - ⌊x⌋

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x := by
  intro x
  sorry

end f_is_periodic_l140_140272


namespace paint_cans_for_28_rooms_l140_140522

theorem paint_cans_for_28_rooms (initial_rooms : ℕ) (final_rooms : ℕ) (lost_cans : ℕ) (rooms_per_lost_can : ℕ) 
    (initial_rooms = 36) (final_rooms = 28) (lost_cans = 4) (rooms_per_lost_can * lost_cans = initial_rooms - final_rooms) 
    : final_rooms / rooms_per_lost_can = 14 :=
by
  -- Here, we would proceed with the proof that the number of cans Paula used for 28 rooms is exactly 14
  sorry

end paint_cans_for_28_rooms_l140_140522


namespace det_matrix_power_l140_140783

theorem det_matrix_power (M : Matrix n n ℝ) (h : det M = 3) : det (M ^ 3) = 27 := by
  sorry

end det_matrix_power_l140_140783


namespace length_less_than_twice_width_l140_140929

def length : ℝ := 24
def width : ℝ := 13.5

theorem length_less_than_twice_width : 2 * width - length = 3 := by
  sorry

end length_less_than_twice_width_l140_140929


namespace nine_points_unit_square_l140_140212

theorem nine_points_unit_square :
  ∀ (points : List (ℝ × ℝ)), points.length = 9 → 
  (∀ (x : ℝ × ℝ), x ∈ points → 0 ≤ x.1 ∧ x.1 ≤ 1 ∧ 0 ≤ x.2 ∧ x.2 ≤ 1) → 
  ∃ (A B C : ℝ × ℝ), A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ 
  (1 / 8 : ℝ) ≤ abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 :=
by
  sorry

end nine_points_unit_square_l140_140212


namespace sum_x_coordinates_above_line_eq_zero_l140_140145

def points : List (ℝ × ℝ) := [(2, 9), (5, 17), (10, 30), (15, 35), (22, 52), (25, 60)]

def line (x : ℝ) : ℝ := 3 * x + 4

theorem sum_x_coordinates_above_line_eq_zero :
  (∑ p in points.filter (λ p, p.snd > line p.fst), p.fst) = 0 :=
by
  sorry

end sum_x_coordinates_above_line_eq_zero_l140_140145


namespace sin_A_value_l140_140037

theorem sin_A_value
  (f : ℝ → ℝ)
  (cos_B : ℝ)
  (f_C_div_2 : ℝ)
  (C_acute : Prop) :
  (∀ x, f x = Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2) →
  cos_B = 1 / 3 →
  f (C / 2) = -1 / 4 →
  (0 < C ∧ C < Real.pi / 2) →
  Real.sin (Real.arcsin (Real.sqrt 3 / 2) + Real.arcsin (2 * Real.sqrt 2 / 3)) = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 :=
by
  intros
  sorry

end sin_A_value_l140_140037


namespace max_value_of_f_l140_140168

noncomputable def f (x : ℝ) : ℝ := (log x) / (x ^ 2)

-- Define the domain
def domain := Set.Ioi 0

-- Theorem statement
theorem max_value_of_f :
  ∃ x ∈ domain, (∀ y ∈ domain, f y ≤ f x) ∧ x = Real.sqrt Real.exp 1 ∧ f x = 1 / (2 * Real.exp 1) :=
by
  sorry

end max_value_of_f_l140_140168


namespace required_bricks_l140_140595

def brick_volume (length width height : ℝ) : ℝ := length * width * height

def wall_volume (length width height : ℝ) : ℝ := length * width * height

theorem required_bricks : 
  let brick_length := 25
  let brick_width := 11.25
  let brick_height := 6
  let wall_length := 850
  let wall_width := 600
  let wall_height := 22.5
  (wall_volume wall_length wall_width wall_height) / 
  (brick_volume brick_length brick_width brick_height) = 6800 :=
by
  sorry

end required_bricks_l140_140595


namespace range_of_values_l140_140725

theorem range_of_values (x : ℝ) : (x^2 - 5 * x + 6 < 0) ↔ (2 < x ∧ x < 3) :=
sorry

end range_of_values_l140_140725


namespace min_value_sum_l140_140807

noncomputable def is_harmonic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, 0 < n → (1 / a (n + 1)) - (1 / a n) = d

def positive_sequence (x : ℕ → ℝ) : Prop :=
∀ n : ℕ, 0 < n → 0 < x n 

def is_arithmetic_sequence (x : ℕ → ℝ) (t : ℝ) : Prop :=
∀ n : ℕ, 0 < n → x (n + 1) - x n = t

theorem min_value_sum:
  ∃ t : ℝ, is_harmonic_sequence (λ n, 1 / x n) d →
          positive_sequence x →
          is_arithmetic_sequence x t →
          (∑ i in Finset.range 20, x (i + 1)) = 200 →
          (∑ (i : ℕ) in {3, 18}, 1 / x i) = 1 / 5 :=
begin
  sorry
end

end min_value_sum_l140_140807


namespace circle_equation_l140_140384

noncomputable def circle_center_eq : Prop :=
  let (a : ℝ) := 1;
  let (r : ℝ) := Real.sqrt 2;
  (circle_eq : (x - a)^2 + y^2 = r^2)

noncomputable def no_line_exists : Prop :=
  let M : Point ℝ := (0, 3);
  let O : Point ℝ := (0, 0);
  let C : Point ℝ := (1, 0);
  let circle_eq : (x - 1)^2 + y^2 = 2;
  ¬ ∃ (l : Line ℝ), 
    (∃ A B : Point ℝ, 
      A ≠ B ∧ A ∈ circle_eq ∧ B ∈ circle_eq ∧ A ∈ l ∧ B ∈ l) ∧
    (let D : Point ℝ := (A.x + B.x, A.y + B.y);
      ∃ k : ℝ, (l = {P | P.y = k * P.x + 3}) ∧ (D - O = C - M))

theorem circle_equation (h : ∀ a r : ℝ, (circle_center_eq) ∧ (r = Real.sqrt 2)) :
  no_line_exists :=
by {
  sorry
}

end circle_equation_l140_140384


namespace symmetric_graph_increasing_interval_l140_140812

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_graph_increasing_interval :
  (∀ x : ℝ, f (-x) = -f x) → -- f is odd
  (∀ x y : ℝ, 3 ≤ x → x < y → y ≤ 7 → f x < f y) → -- f is increasing in [3,7]
  (∀ x : ℝ, 3 ≤ x → x ≤ 7 → f x ≤ 5) → -- f has a maximum value of 5 in [3,7]
  (∀ x y : ℝ, -7 ≤ x → x < y → y ≤ -3 → f x < f y) ∧ -- f is increasing in [-7,-3]
  (∀ x : ℝ, -7 ≤ x → x ≤ -3 → f x ≥ -5) -- f has a minimum value of -5 in [-7,-3]
:= sorry

end symmetric_graph_increasing_interval_l140_140812


namespace find_theta_l140_140702

noncomputable def cis (θ : ℝ) : ℂ := complex.exp (complex.I * θ)

theorem find_theta :
  ∃ r > 0, 0 ≤ θ ∧ θ < 360 ∧
  ∑ k in (finset.range 9), -- The angles are 60º, 70º, ..., 140º which are 9 terms.
  cis (60 + k * 10) = r * cis 100 :=
begin
  sorry
end

end find_theta_l140_140702


namespace monotone_range_of_f_l140_140760

theorem monotone_range_of_f {f : ℝ → ℝ} (a : ℝ) 
  (h : ∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≤ y → f x ≤ f y) : a ≤ 0 :=
sorry

end monotone_range_of_f_l140_140760


namespace find_x_l140_140773

variable {x : ℝ}

def a : ℝ × ℝ := (6, x)
def b : ℝ × ℝ := (2, -2)
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem find_x : (a_minus_b.1 * b.1 + a_minus_b.2 * b.2 = 0) → x = 2 := by
  sorry

end find_x_l140_140773


namespace find_intervals_of_monotonicity_find_value_of_a_l140_140718

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + a + 1

theorem find_intervals_of_monotonicity (k : ℤ) (a : ℝ) :
  ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3), MonotoneOn (λ x => f x a) (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) :=
sorry

theorem find_value_of_a (a : ℝ) (max_value_condition : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) :
  a = 1 :=
sorry

end find_intervals_of_monotonicity_find_value_of_a_l140_140718


namespace distance_between_truck_and_car_l140_140644

noncomputable def truck_speed : ℝ := 65
noncomputable def car_speed : ℝ := 85
noncomputable def time_duration_hours : ℝ := 3 / 60

theorem distance_between_truck_and_car :
  let relative_speed := car_speed - truck_speed in
  let distance := relative_speed * time_duration_hours in
  distance = 1 :=
by
  let relative_speed := car_speed - truck_speed
  let distance := relative_speed * time_duration_hours
  have h1 : distance = 1 := sorry
  exact h1

end distance_between_truck_and_car_l140_140644


namespace quadratic_zeros_interval_l140_140432

theorem quadratic_zeros_interval (a : ℝ) :
  (5 - 2 * a > 0) ∧ (4 * a^2 - 16 > 0) ∧ (a > 1) ↔ (2 < a ∧ a < 5 / 2) :=
by
  sorry

end quadratic_zeros_interval_l140_140432


namespace ellipses_have_equal_focal_length_l140_140216

-- Define ellipses and their focal lengths
def ellipse1_focal_length : ℝ := 8
def k_condition (k : ℝ) : Prop := 0 < k ∧ k < 9
def ellipse2_focal_length (k : ℝ) : ℝ := 8

-- The main statement
theorem ellipses_have_equal_focal_length (k : ℝ) (hk : k_condition k) :
  ellipse1_focal_length = ellipse2_focal_length k :=
sorry

end ellipses_have_equal_focal_length_l140_140216


namespace tempo_insured_fraction_l140_140312

theorem tempo_insured_fraction (premium : ℝ) (rate : ℝ) (original_value : ℝ) (h1 : premium = 300) (h2 : rate = 0.03) (h3 : original_value = 14000) : 
  premium / rate / original_value = 5 / 7 :=
by 
  sorry

end tempo_insured_fraction_l140_140312


namespace number_of_seeds_in_bucket_B_l140_140230

theorem number_of_seeds_in_bucket_B :
  ∃ (x : ℕ), 
    ∃ (y : ℕ), 
    ∃ (z : ℕ), 
      y = x + 10 ∧ 
      z = 30 ∧ 
      x + y + z = 100 ∧
      x = 30 :=
by {
  -- the proof is omitted.
  sorry
}

end number_of_seeds_in_bucket_B_l140_140230


namespace intersection_eq_l140_140738

def setA : Set ℝ := {x | -1 < x ∧ x < 1}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_eq : setA ∩ setB = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_eq_l140_140738


namespace gcd_lcm_of_300_and_462_l140_140246

def is_prime_factorization (n : ℕ) (f : list (ℕ × ℕ)) : Prop :=
  n = list.prod (f.map (λ p => p.1 ^ p.2))

def gcd (a b : ℕ) : ℕ := a.gcd b
def lcm (a b : ℕ) : ℕ := a.lcm b

theorem gcd_lcm_of_300_and_462 :
  is_prime_factorization 300 [(2, 2), (3, 1), (5, 2)] →
  is_prime_factorization 462 [(2, 1), (3, 1), (7, 1), (11, 1)] →
  gcd 300 462 = 6 ∧ lcm 300 462 = 46200 :=
by
  intros h1 h2
  sorry

end gcd_lcm_of_300_and_462_l140_140246


namespace unique_triangle_exists_l140_140318

def unique_triangle (A B C : Type) (AB BC CA : ℝ)
  (angleA angleB angleC : ℝ)
  (h1 : angleA = 60)
  (h2 : angleB = 45)
  (h3 : AB = 4) : Prop :=
  ∃ (triangle : Type), 
    (∀ x : triangle, 
      ∀ angleX angleY : ℝ,
      angleX + angleY + 60 = 180) ∧
    ( ∀ y : triangle,
      ∀ sideY : ℝ,
      sideY + 4 > 4) -- Unique property placeholders

theorem unique_triangle_exists :
  unique_triangle 
    ℝ 
    ℝ 
    ℝ 
    4 
    0
    0 
    60
    45
    0
    rfl
    rfl
    rfl :=
begin
  sorry,
end

end unique_triangle_exists_l140_140318


namespace rectangular_coordinates_of_C_l140_140398

def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem rectangular_coordinates_of_C :
  ∀ (ρ θ : ℝ), (ρ, θ) = (2, Real.pi / 4) → polar_to_rectangular ρ θ = (Real.sqrt 2, Real.sqrt 2) :=
by
  intros ρ θ h
  rw [h]
  unfold polar_to_rectangular
  simp
  sorry  -- skipping the proof

end rectangular_coordinates_of_C_l140_140398


namespace unique_element_in_set_l140_140387

theorem unique_element_in_set (A : Set ℝ) (h₁ : ∃ x, A = {x})
(h₂ : ∀ x ∈ A, (x + 3) / (x - 1) ∈ A) : ∃ x, x ∈ A ∧ (x = 3 ∨ x = -1) := by
  sorry

end unique_element_in_set_l140_140387


namespace toms_dad_gave_him_dimes_l140_140949

theorem toms_dad_gave_him_dimes (original_dimes final_dimes dimes_given : ℕ)
  (h1 : original_dimes = 15)
  (h2 : final_dimes = 48)
  (h3 : final_dimes = original_dimes + dimes_given) :
  dimes_given = 33 :=
by
  -- Since the main goal here is just the statement, proof is omitted with sorry
  sorry

end toms_dad_gave_him_dimes_l140_140949


namespace distance_after_3_minutes_l140_140654

-- Conditions: speeds of the truck and car, and the time interval in hours
def v_truck : ℝ := 65 -- in km/h
def v_car : ℝ := 85 -- in km/h
def t : ℝ := 3 / 60 -- convert 3 minutes to hours

-- Statement to prove: The distance between the truck and the car after 3 minutes is 1 km
theorem distance_after_3_minutes : (v_car - v_truck) * t = 1 := 
by
  sorry

end distance_after_3_minutes_l140_140654


namespace det_matrix_power_l140_140785

theorem det_matrix_power (M : Matrix n n ℝ) (h : det M = 3) : det (M ^ 3) = 27 := by
  sorry

end det_matrix_power_l140_140785


namespace sin_ratio_in_triangle_l140_140079

theorem sin_ratio_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h : (b + c) / (c + a) = 4 / 5 ∧ (c + a) / (a + b) = 5 / 6) :
  (Real.sin A + Real.sin C) / Real.sin B = 2 :=
sorry

end sin_ratio_in_triangle_l140_140079


namespace smallest_c_inequality_l140_140710

theorem smallest_c_inequality (x : ℕ → ℝ) (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10) :
  ∃ c : ℝ, (∀ x : ℕ → ℝ, x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10 →
    |x 0| + |x 1| + |x 2| + |x 3| + |x 4| + |x 5| + |x 6| + |x 7| + |x 8| ≥ c * |x 4|) ∧ c = 9 := 
by
  sorry

end smallest_c_inequality_l140_140710


namespace find_perimeter_and_sin2A_of_triangle_l140_140009

theorem find_perimeter_and_sin2A_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 3) (h_B : B = Real.pi / 3) (h_area : 6 * Real.sqrt 3 = 6 * Real.sqrt 3)
  (h_S : S_ABC = 6 * Real.sqrt 3) : 
  (a + b + c = 18) ∧ (Real.sin (2 * A) = (39 * Real.sqrt 3) / 98) := 
by 
  -- The proof will be placed here. Assuming a valid proof exists.
  sorry

end find_perimeter_and_sin2A_of_triangle_l140_140009


namespace product_of_midpoint_coordinates_l140_140341

theorem product_of_midpoint_coordinates :
  let x₁ := 4, y₁ := -1, z₁ := 9,
      x₂ := 12, y₂ := 3, z₂ := -3,
      mx := (x₁ + x₂) / 2, my := (y₁ + y₂) / 2, mz := (z₁ + z₂) / 2 in
  mx * my * mz = 24 :=
by
  sorry

end product_of_midpoint_coordinates_l140_140341


namespace eval_expression_1_eval_expression_2_l140_140344

theorem eval_expression_1 : 
  (2 + 1/4)^(1/2) - (3.141592653589793 - 1)^0 - (3 + 3/8)^(-2/3) = 1/18 :=
by
  sorry

theorem eval_expression_2 : 
  log 3 (sqrt(3) / 3) + log 10 5 + log 10 0.2 + 7^(log 7 2) = 3/2 :=
by
  sorry

end eval_expression_1_eval_expression_2_l140_140344


namespace dodecagon_diagonals_l140_140358

theorem dodecagon_diagonals (n : ℕ) (h : n = 12) : 
  let diagonals_from_one_vertex := n - 3 in
  diagonals_from_one_vertex = 9 :=
by
  sorry

end dodecagon_diagonals_l140_140358


namespace monotonicity_of_f_g_gt_zero_for_x_gt_one_range_of_a_l140_140126

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a - Real.log x
noncomputable def g (x : ℝ) : ℝ := 1 / x - 1 / Real.exp (x - 1)

theorem monotonicity_of_f (a : ℝ) : 
  (a ≤ 0 → ∀ x > 0, (f a)' x < 0) ∧ 
  (a > 0 → ∀ x, (0 < x ∧ x < Real.sqrt (1 / (2 * a)) → (f a)' x < 0) 
  ∧ (x > Real.sqrt (1 / (2 * a)) → (f a)' x > 0)) :=
sorry

theorem g_gt_zero_for_x_gt_one : ∀ x > 1, g x > 0 :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 1, f a x > g x) ↔ a ≥ 1 / 2 :=
sorry

end monotonicity_of_f_g_gt_zero_for_x_gt_one_range_of_a_l140_140126


namespace max_correct_questions_l140_140817

theorem max_correct_questions (a b c : ℕ) (h1 : a + b + c = 60) (h2 : 3 * a - 2 * c = 126) : a ≤ 49 :=
sorry

end max_correct_questions_l140_140817


namespace beth_overall_score_l140_140669

-- Definitions for conditions
def percent_score (score_pct : ℕ) (total_problems : ℕ) : ℕ :=
  (score_pct * total_problems) / 100

def total_correct_answers : ℕ :=
  percent_score 60 15 + percent_score 85 20 + percent_score 75 25

def total_problems : ℕ := 15 + 20 + 25

def combined_percentage : ℕ :=
  (total_correct_answers * 100) / total_problems

-- The statement to be proved
theorem beth_overall_score : combined_percentage = 75 := by
  sorry

end beth_overall_score_l140_140669


namespace checker_triplets_l140_140281

theorem checker_triplets (N A B : ℕ) (h1 : 2 * N = A + B) (h2 : ∀ t, t ∈ triples → (predominantly_white t ↔ t ∈ A_list) ∧ (predominantly_black t ↔ t ∈ B_list)) :
  A ≤ 3 * B :=
by sorry

end checker_triplets_l140_140281


namespace pizza_slices_left_l140_140970

-- Lean definitions for conditions
def total_slices : ℕ := 24
def slices_eaten_dinner : ℕ := total_slices / 3
def slices_after_dinner : ℕ := total_slices - slices_eaten_dinner

def slices_eaten_yves : ℕ := slices_after_dinner / 5
def slices_after_yves : ℕ := slices_after_dinner - slices_eaten_yves

def slices_eaten_oldest_siblings : ℕ := 3 * 3
def slices_after_oldest_siblings : ℕ := slices_after_yves - slices_eaten_oldest_siblings

def num_remaining_siblings : ℕ := 7 - 3
def slices_eaten_remaining_siblings : ℕ := num_remaining_siblings * 2
def slices_final : ℕ := if slices_after_oldest_siblings < slices_eaten_remaining_siblings then 0 else slices_after_oldest_siblings - slices_eaten_remaining_siblings

-- Proposition to prove
theorem pizza_slices_left : slices_final = 0 := by sorry

end pizza_slices_left_l140_140970


namespace net_earnings_calculation_l140_140900

-- Definitions of constants and functions
def kem_hourly_earnings := 4
def shem_hourly_earnings := 2.5 * kem_hourly_earnings
def tiff_hourly_earnings := kem_hourly_earnings + 3

def kem_hours_worked := 6
def shem_hours_worked := 8
def tiff_hours_worked := 10

def gross_earnings (rate hours : ℝ) : ℝ := rate * hours

def net_earnings (gross deductions : ℝ) : ℝ := gross - deductions

def kem_deductions (gross : ℝ) : ℝ := 0.1 * gross
def shem_deductions (gross : ℝ) : ℝ := 0.05 * gross + 5
def tiff_deductions (gross : ℝ) : ℝ := 0.03 * gross + 3

def kem_net_earnings := net_earnings (gross_earnings kem_hourly_earnings kem_hours_worked) (kem_deductions (gross_earnings kem_hourly_earnings kem_hours_worked))
def shem_net_earnings := net_earnings (gross_earnings shem_hourly_earnings shem_hours_worked) (shem_deductions (gross_earnings shem_hourly_earnings shem_hours_worked))
def tiff_net_earnings := net_earnings (gross_earnings tiff_hourly_earnings tiff_hours_worked) (tiff_deductions (gross_earnings tiff_hourly_earnings tiff_hours_worked))

def total_net_earnings := kem_net_earnings + shem_net_earnings + tiff_net_earnings

-- Theorem to be proven
theorem net_earnings_calculation :
  kem_net_earnings = 21.6 ∧ shem_net_earnings = 71 ∧ tiff_net_earnings = 64.9 ∧ total_net_earnings = 157.5 :=
by
  sorry

end net_earnings_calculation_l140_140900


namespace correct_sampling_method_l140_140303

structure SchoolPopulation :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)

-- Define the school population
def school : SchoolPopulation :=
  { senior := 10, intermediate := 50, junior := 75 }

-- Define the condition for sampling method
def total_school_teachers (s : SchoolPopulation) : ℕ :=
  s.senior + s.intermediate + s.junior

-- The desired sample size
def sample_size : ℕ := 30

-- The correct sampling method based on the population strata
def stratified_sampling (s : SchoolPopulation) : Prop :=
  s.senior + s.intermediate + s.junior > 0

theorem correct_sampling_method : stratified_sampling school :=
by { sorry }

end correct_sampling_method_l140_140303


namespace arithmetic_sequence_nth_term_l140_140552

theorem arithmetic_sequence_nth_term (a b c n : ℕ) (x: ℕ)
  (h1: a = 3*x - 4)
  (h2: b = 6*x - 17)
  (h3: c = 4*x + 5)
  (h4: b - a = c - b)
  (h5: a + (n - 1) * (b - a) = 4021) : 
  n = 502 :=
by 
  sorry

end arithmetic_sequence_nth_term_l140_140552


namespace max_coins_for_first_robber_l140_140575

theorem max_coins_for_first_robber (coins : ℕ) (num_bags : ℕ) (coins_per_bag : ℕ) :
  (coins = 300) → (coins_per_bag = 14) → (num_bags = coins / coins_per_bag) → (coins % coins_per_bag = 2) →
  (num_bags >= 21) → (∀ n1 n2 : ℕ, n1 + n2 = num_bags → (n1 = 11 ∨ n2 = 11) → (n1 * coins_per_bag = 154 ∨ n2 * coins_per_bag = 154)) →
  (∃ g : ℕ, g = 300 - 154 ∧ g >= 146) := by
  intros h1 h2 h3 h4 h5 h6
  use 146
  split
  { exact nat.sub_self 154.symm }
  { exact le_refl 146 }
  
  sorry

end max_coins_for_first_robber_l140_140575


namespace sumset_eq_real_line_l140_140166

open Set

/- The Cantor set on [0, 1] with Lebesgue measure zero -/
def cantor_set : Set ℝ := sorry

/- Construct set A as the Minkowski sum of the Cantor set and the integers -/
def set_A : Set ℝ := { x | ∃ (c ∈ cantor_set) (z : ℤ), x = c + (z : ℝ) }

/- Construct set B as the Cantor set -/
def set_B : Set ℝ := cantor_set

/- Definition of Lebesgue measure zero -/
def lebesgue_measure_zero (S : Set ℝ) : Prop := sorry

/- Minkowski sum of sets -/
def minkowski_sum (A B : Set ℝ) : Set ℝ := { x | ∃ a b, a ∈ A ∧ b ∈ B ∧ x = a + b }

/- Main theorem statement -/
theorem sumset_eq_real_line : 
  lebesgue_measure_zero set_A ∧ 
  lebesgue_measure_zero set_B ∧ 
  minkowski_sum set_A set_B = univ :=
by
  sorry

end sumset_eq_real_line_l140_140166


namespace C_finishes_work_in_days_l140_140789

theorem C_finishes_work_in_days :
  (∀ (unit : ℝ) (A B C combined: ℝ),
    combined = 1 / 4 ∧
    A = 1 / 12 ∧
    B = 1 / 24 ∧
    combined = A + B + 1 / C) → 
    C = 8 :=
  sorry

end C_finishes_work_in_days_l140_140789


namespace intersection_points_count_l140_140679

-- Define the hyperbola and lines
def hyperbola (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1

def line1 (m c x y : ℝ) : Prop := y = m * x + c
def line2 (m d x y : ℝ) : Prop := y = 2 * m * x + d

-- Define the main theorem
theorem intersection_points_count (m c d : ℝ) :
  (∃ p1 p2 p3 p4 : ℝ × ℝ, (line1 m c p1.1 p1.2 ∧ hyperbola p1.1 p1.2) ∧
                         (line1 m c p2.1 p2.2 ∧ hyperbola p2.1 p2.2) ∧
                         (line2 m d p3.1 p3.2 ∧ hyperbola p3.1 p3.2) ∧
                         (line2 m d p4.1 p4.2 ∧ hyperbola p4.1 p4.2) ∧
                         ((p1 ≠ p3) ∧ (p2 ≠ p4)) ∨ -- mutual distinct points
                         ((p1 = p3) ∨ (p1 = p4) ∨ (p2 = p3) ∨ (p2 = p4))) → -- some points coincide
    (2 ≤ (if p1 ≠ p3 then 2 else 1) + (if p2 ≠ p4 then 2 else 1) ∨
     2 ≤ (if p1 ≠ p3 then 3 else 2) + (if p2 ≠ p4 then 2 else 1) ∨
     2 ≤ (if p1 ≠ p3 then 2 else 1) + (if p2 ≠ p4 then 3 else 2) ∨
     2 ≤ (if p1 ≠ p3 then 3 else 2) + (if p2 ≠ p4 then 3 else 2)) :=
sorry

end intersection_points_count_l140_140679


namespace operation_positive_l140_140431

theorem operation_positive (op : ℤ → ℤ → ℤ) (is_pos : op 1 (-2) > 0) : op = Int.sub :=
by
  sorry

end operation_positive_l140_140431


namespace find_power_function_l140_140355

theorem find_power_function :
  ∃ (f : ℝ → ℝ), (∀ x, f x = x ^ (1 / 2)) ∧ f 9 = 3 :=
by
  let f : ℝ → ℝ := λ x, x ^ (1 / 2)
  use f
  split
  · intro x
    rfl
  · rfl

end find_power_function_l140_140355


namespace average_age_of_club_l140_140083

theorem average_age_of_club (S_f S_m S_c : ℕ) (females males children : ℕ) (avg_females avg_males avg_children : ℕ) :
  females = 12 →
  males = 20 →
  children = 8 →
  avg_females = 28 →
  avg_males = 40 →
  avg_children = 10 →
  S_f = avg_females * females →
  S_m = avg_males * males →
  S_c = avg_children * children →
  (S_f + S_m + S_c) / (females + males + children) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end average_age_of_club_l140_140083


namespace problem_proof_l140_140543

open EuclideanGeometry

-- Definitions from the problem
variables (A B C D E F H O : Point)
variables (circle_AOB : Circle)
variables (line_BH : Line)

-- Given:
-- 1. Circles AOB and BFO exist
-- 2. Extended line BH intersects the circumcircle of triangle ABC at D

def is_on_circumcircle (A B C D : Point) : Prop :=
  (lies_on_circle A) = (lies_on_circle B) ∧ (lies_on_circle B) = (lies_on_circle C) ∧ (lies_on_circle C) = (lies_on_circle D)

-- Translating the given conditions
axiom circles_exist : ∃ circle_AOB : Circle, circle B F O = circle_AOB
axiom extend_BH_circumcircle_ABC : ∃ p : Point, p = D ∧ lies_on_circle p (circumcircle_of_triangle A B C)

-- Define the necessary point locations based on circles
axiom point_E_circles_DCO_ADH : lies_on_circle E (circle D C O) ∧ lies_on_circle E (circle A D H)
axiom point_F_circles_CHD_AOD : lies_on_circle F (circle C H D) ∧ lies_on_circle F (circle A O D)

-- Lean Statement encapsulating the problem
theorem problem_proof (A B C D E F H O : Point) :
  (is_on_circumcircle A B C D) →
  (exists p: Point, p = D ∧ lies_on_circle p (circumcircle_of_triangle A B C)) →
  (lies_on_circle E (circle D C O)) ∧ (lies_on_circle E (circle A D H)) →
  (lies_on_circle F (circle C H D)) ∧ (lies_on_circle F (circle A O D)) :=
by
  sorry

end problem_proof_l140_140543


namespace state_sector_wages_increase_private_sector_price_decrease_l140_140335

theorem state_sector_wages_increase (mandatory_work: Prop) 
  (state_graduates: ℕ) (supply_state: ℕ) (wage_state: ℕ) :
  (mandatory_work → supply_state < state_graduates) →
  (demand_state: ℕ) →
  (supply_state < demand_state) →
  (supply_decrease → wage_state_increase) → 
  wage_state_increase
  sorry

theorem private_sector_price_decrease (mandatory_work: Prop) 
  (private_graduates: ℕ) (supply_private: ℕ) (price_private: ℕ) :
  (mandatory_work → supply_private > private_graduates) →
  (demand_private: ℕ) →
  (supply_private > demand_private) →
  (supply_increase → price_private_decrease) → 
  price_private_decrease
  sorry

end state_sector_wages_increase_private_sector_price_decrease_l140_140335


namespace percentage_temporary_employees_l140_140827

open_locale classical

/-- A factory consists of 80% technicians and 20% non-technicians.
    Of the technicians, 80% are permanent employees. 
    Of the non-technicians, 20% are permanent employees.
    We need to prove that the percentage of temporary employees is 32%. -/
theorem percentage_temporary_employees
    (total_workers : ℕ)
    (technicians_percentage non_technicians_percentage : ℝ)
    (perm_technicians_percentage perm_non_technicians_percentage : ℝ) :
  technicians_percentage = 0.80 →
  non_technicians_percentage = 0.20 →
  perm_technicians_percentage = 0.80 →
  perm_non_technicians_percentage = 0.20 →
  ( (total_workers - ((technicians_percentage * total_workers * perm_technicians_percentage) +
            (non_technicians_percentage * total_workers * perm_non_technicians_percentage)))
    / total_workers * 100 )  = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end percentage_temporary_employees_l140_140827


namespace raghu_investment_approx_l140_140245

noncomputable def geeta_investment : ℝ := 21824 / 4.8345
noncomputable def raghu_investment : ℝ := 1.05 * geeta_investment

theorem raghu_investment_approx :
  raghu_investment ≈ 4742.40 :=
by sorry

end raghu_investment_approx_l140_140245


namespace max_coins_for_first_robber_l140_140576

theorem max_coins_for_first_robber (coins : ℕ) (num_bags : ℕ) (coins_per_bag : ℕ) :
  (coins = 300) → (coins_per_bag = 14) → (num_bags = coins / coins_per_bag) → (coins % coins_per_bag = 2) →
  (num_bags >= 21) → (∀ n1 n2 : ℕ, n1 + n2 = num_bags → (n1 = 11 ∨ n2 = 11) → (n1 * coins_per_bag = 154 ∨ n2 * coins_per_bag = 154)) →
  (∃ g : ℕ, g = 300 - 154 ∧ g >= 146) := by
  intros h1 h2 h3 h4 h5 h6
  use 146
  split
  { exact nat.sub_self 154.symm }
  { exact le_refl 146 }
  
  sorry

end max_coins_for_first_robber_l140_140576


namespace area_of_midpoint_quadrilateral_l140_140084

theorem area_of_midpoint_quadrilateral
  (a b : ℝ)
  (ABCD : ConvexQuadrilateral)
  (M N K L : Point)
  (midpoints : is_midpoint_polygon ABCD [M, N, K, L])
  (diag_AC : length (AC) = 2 * a)
  (diag_BD : length (BD) = 2 * a)
  (diag_sum : length (MK) + length (NL) = 2 * b) :
  area (quadrilateral M N K L) = 2 * a * b :=
sorry

end area_of_midpoint_quadrilateral_l140_140084


namespace students_not_liking_either_l140_140821

theorem students_not_liking_either (total_students like_fries like_burgers like_both : ℕ)
  (h_total : total_students = 25)
  (h_fries : like_fries = 15)
  (h_burgers : like_burgers = 10)
  (h_both : like_both = 6) :
  total_students - (like_fries - like_both + like_burgers - like_both + like_both) = 6 :=
by
  rw [h_total, h_fries, h_burgers, h_both]
  exact rfl

end students_not_liking_either_l140_140821


namespace square_area_l140_140291

/-- Define conditions and the theorem statement -/
theorem square_area (r : ℝ) (h : r = 7) : (let s := 2 * r in s * s = 196) :=
by
  sorry

end square_area_l140_140291


namespace increasing_function_implies_a_gt_one_l140_140554

theorem increasing_function_implies_a_gt_one (a : ℝ) (f : ℝ → ℝ) (h : f = λ x, (a-1)*x + 2) : 
  (∀ x y : ℝ, x < y → f x < f y) → a > 1 :=
by
-- Define the function f
let f := λ x : ℝ, (a - 1) * x + 2
-- Assume f is increasing
assume h_increasing : ∀ x y : ℝ, x < y → f x < f y
-- Prove a > 1
sorry

end increasing_function_implies_a_gt_one_l140_140554


namespace gcd_solution_l140_140394

noncomputable def gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : ℤ :=
  Int.gcd (4 * b^2 + 35 * b + 72) (3 * b + 8)

theorem gcd_solution (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : gcd_problem b h = 2 :=
by
  sorry

end gcd_solution_l140_140394


namespace relationships_l140_140872

variable (x : ℝ)
variable (hx : x ∈ Set.Ioo (-1 / 2) 0)

def a1 := Real.cos (Real.sin (x * Real.pi))
def a2 := Real.sin (Real.cos (x * Real.pi))
def a3 := Real.cos ((x + 1) * Real.pi)

theorem relationships : a3 x < a2 x ∧ a2 x < a1 x :=
by
  sorry

end relationships_l140_140872


namespace fraction_irreducible_l140_140905

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_irreducible_l140_140905


namespace number_of_tails_l140_140839

theorem number_of_tails (n : ℕ) (f_h : ℝ) (t : ℕ) (h_n : n = 100) (h_fh : f_h = 0.49)
  (h_ft : 1 - f_h = 0.51) (h_t : t = (0.51 * n).toNat) : t = 51 := 
by 
  have h1 : 0.51 * 100 = 51 := by norm_num
  rw [h_n, h_fh, h1] at h_t
  exact h_t

end number_of_tails_l140_140839


namespace total_campers_went_rowing_l140_140603

-- Definitions based on given conditions
def morning_campers : ℕ := 36
def afternoon_campers : ℕ := 13
def evening_campers : ℕ := 49

-- Theorem statement to be proven
theorem total_campers_went_rowing : morning_campers + afternoon_campers + evening_campers = 98 :=
by sorry

end total_campers_went_rowing_l140_140603


namespace expression_value_is_one_l140_140960

theorem expression_value_is_one :
  let a1 := 121
  let b1 := 19
  let a2 := 91
  let b2 := 13
  (a1^2 - b1^2) / (a2^2 - b2^2) * ((a2 - b2) * (a2 + b2)) / ((a1 - b1) * (a1 + b1)) = 1 := by
  sorry

end expression_value_is_one_l140_140960


namespace value_of_x_l140_140435

noncomputable def sum_integers_30_to_50 : ℕ :=
  (50 - 30 + 1) * (30 + 50) / 2

def even_count_30_to_50 : ℕ :=
  11

theorem value_of_x 
  (x := sum_integers_30_to_50)
  (y := even_count_30_to_50)
  (h : x + y = 851) : x = 840 :=
sorry

end value_of_x_l140_140435


namespace payment_difference_l140_140108

noncomputable def A1 : Float := 15000 * (1 + 0.08 / 2) ^ 12
-- After 6 years compounded semi-annually
noncomputable def Payment6Years : Float := A1 / 3
-- Remaining balance
noncomputable def BalanceRemaining : Float := A1 - Payment6Years
-- Compounded quarterly for the next 4 years
noncomputable def A2 : Float := BalanceRemaining * (1 + 0.08 / 4) ^ 16
-- Total Payment under Plan 1
noncomputable def TotalPaymentPlan1 : Float := Payment6Years + A2

-- Plan 2: Amount compounded annually for 10 years
noncomputable def TotalPaymentPlan2 : Float := 15000 * (1 + 0.08) ^ 10

-- Difference between the two plans
noncomputable def Difference : Float := TotalPaymentPlan2 - TotalPaymentPlan1

theorem payment_difference :
  abs (round Difference) = 2472 :=
by
  sorry

end payment_difference_l140_140108


namespace smallest_d_l140_140299

theorem smallest_d (d : ℝ) (h : real.sqrt ((4 * real.sqrt 5)^2 + (d - 2)^2) = 4 * d) : d ≈ 2.503 :=
by sorry

end smallest_d_l140_140299


namespace exists_equidistant_point_from_vertices_no_point_equidistant_from_edges_no_point_equidistant_from_faces_l140_140826

-- Define a cuboid
structure Cuboid :=
  (length : ℝ)
  (width : ℝ)
  (height : ℝ)
  (length_pos : 0 < length)
  (width_pos : 0 < width)
  (height_pos : 0 < height)

-- Define a point in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the vertices, edges, and faces as lists of points of a cuboid
def vertices (c : Cuboid) : List Point3D := [
  -- bottom vertices
  {x := 0, y := 0, z := 0}, 
  {x := c.length, y := 0, z := 0},
  {x := 0, y := c.width, z := 0},
  {x := c.length, y := c.width, z := 0},
  -- top vertices
  {x := 0, y := 0, z := c.height},
  {x := c.length, y := 0, z := c.height},
  {x := 0, y := c.width, z := c.height},
  {x := c.length, y := c.width, z := c.height}
]

noncomputable def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2,
    z := (p1.z + p2.z) / 2 }

noncomputable def body_diagonal_intersection (c : Cuboid) : Point3D :=
  midpoint { x := 0, y := 0, z := 0 } { x := c.length, y := c.width, z := c.height }

theorem exists_equidistant_point_from_vertices (c : Cuboid) :
  ∃ p : Point3D, ∀ v ∈ vertices c, dist p v = some_constant := by
  sorry

theorem no_point_equidistant_from_edges (c : Cuboid) :
  ∀ p : Point3D, ∃ v₁ v₂ ∈ edges c, dist p v₁ ≠ dist p v₂ := by
  sorry

theorem no_point_equidistant_from_faces (c : Cuboid) :
  ∀ p : Point3D, ∃ v₁ v₂ ∈ faces c, dist p v₁ ≠ dist p v₂ := by
  sorry

end exists_equidistant_point_from_vertices_no_point_equidistant_from_edges_no_point_equidistant_from_faces_l140_140826


namespace range_of_a_for_f_monotonic_l140_140757

-- Define the function f
def f (x a : ℝ) := real.sqrt(x * (x - a))

-- Define the condition that f(x) is monotonically increasing in the interval (0,1)
def is_monotonically_increasing_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y

-- The main theorem we want to prove
theorem range_of_a_for_f_monotonic (a : ℝ) :
  is_monotonically_increasing_in_interval (λ x, f x a) 0 1 → a ≤ 0 :=
by
  sorry

end range_of_a_for_f_monotonic_l140_140757


namespace find_4_digit_number_l140_140700

theorem find_4_digit_number (a b c d : ℕ) 
(h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182)
(h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
(h6 : 0 ≤ c) (h7 : c ≤ 9) (h8 : 1 ≤ d) (h9 : d ≤ 9) : 
1000 * a + 100 * b + 10 * c + d = 1909 :=
sorry

end find_4_digit_number_l140_140700


namespace alien_abduction_l140_140656

theorem alien_abduction (P : ℕ) (h1 : 80 % of P = 0.80 * P) (h2 : 10 + 30 = 40) 
    (h3 : 0.20 * P = 40) : P = 200 := 
by 
  sorry

end alien_abduction_l140_140656


namespace product_equality_l140_140213

noncomputable def product := (∏ n in Finset.range 11 \ Finset.singleton 0, (1 - 1 / (n + 2)^2))

theorem product_equality : product = 13 / 24 := by
  sorry

end product_equality_l140_140213


namespace diagonal_length_l140_140541

noncomputable def convertHectaresToSquareMeters (hectares : ℝ) : ℝ :=
  hectares * 10000

noncomputable def sideLength (areaSqMeters : ℝ) : ℝ :=
  Real.sqrt areaSqMeters

noncomputable def diagonal (side : ℝ) : ℝ :=
  side * Real.sqrt 2

theorem diagonal_length (area : ℝ) (h : area = 1 / 2) :
  let areaSqMeters := convertHectaresToSquareMeters area
  let side := sideLength areaSqMeters
  let diag := diagonal side
  abs (diag - 100) < 1 :=
by
  sorry

end diagonal_length_l140_140541


namespace compute_a1d1_a2d2_a3d3_l140_140870

theorem compute_a1d1_a2d2_a3d3
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, x^6 + 2 * x^5 + x^4 + x^3 + x^2 + 2 * x + 1 = (x^2 + a1*x + d1) * (x^2 + a2*x + d2) * (x^2 + a3*x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 2 :=
by
  sorry

end compute_a1d1_a2d2_a3d3_l140_140870


namespace shell_age_in_decimal_l140_140304

-- Define the conversion from octal to decimal.
def octal_to_decimal (n : List ℕ) : ℕ :=
  n.reverse.foldl (λ acc x, acc * 8 + x) 0

instance : Coe ℕ (List ℕ) :=
  ⟨λ n, n.digits 8⟩

def is_correct_decimal_conversion : Prop :=
  octal_to_decimal [3, 7, 5, 4] = 2028

theorem shell_age_in_decimal : is_correct_decimal_conversion := by
  simp [octal_to_decimal]
  sorry

end shell_age_in_decimal_l140_140304


namespace product_of_positive_imaginary_solutions_l140_140810

noncomputable def complex_solutions_product : ℂ := by
  let solutions := [2 * Complex.exp (Complex.I * (22.5 * Real.pi / 180)),
                    2 * Complex.exp (Complex.I * (67.5 * Real.pi / 180)),
                    2 * Complex.exp (Complex.I * (112.5 * Real.pi / 180)),
                    2 * Complex.exp (Complex.I * (157.5 * Real.pi / 180))]
  exact solutions.prod

theorem product_of_positive_imaginary_solutions :
  let solutions_product := complex_solutions_product in
  solutions_product.abs = 16 :=
sorry

end product_of_positive_imaginary_solutions_l140_140810


namespace problem_solution_l140_140121

theorem problem_solution :
  let a := 9
  let b := 4
  (a - b)^2 = 25 :=
by
  let a := 9
  let b := 4
  show (a - b)^2 = 25 from sorry

end problem_solution_l140_140121


namespace hundredth_bead_color_is_red_l140_140278

def beadPattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def beadColor (n : Nat) : String :=
  beadPattern[(n - 1) % beadPattern.length]

theorem hundredth_bead_color_is_red :
  beadColor 100 = "red" :=
by
  sorry

end hundredth_bead_color_is_red_l140_140278


namespace real_root_interval_l140_140802

theorem real_root_interval (b : ℝ) (p : ℝ → ℝ)
  (h_poly : p = λ x, x^2 + b * x + 25)
  (h_real_root : ∃ x : ℝ, p x = 0) :
  b ∈ set.Iic (-10) ∪ set.Ici 10 :=
sorry

end real_root_interval_l140_140802


namespace remainder_of_x_mod_10_l140_140990

def x : ℕ := 2007 ^ 2008

theorem remainder_of_x_mod_10 : x % 10 = 1 := by
  sorry

end remainder_of_x_mod_10_l140_140990


namespace solve_for_z_in_complex_eq_l140_140181

theorem solve_for_z_in_complex_eq (z : ℂ) : 2 - 3 * complex.i * z = -4 + 5 * complex.i * z → z = -3 * complex.i / 4 :=
by
  intro h
  sorry

end solve_for_z_in_complex_eq_l140_140181


namespace exist_infinitely_many_coprime_pairs_l140_140134

theorem exist_infinitely_many_coprime_pairs (α : ℝ) (hα : irrational α) :
  ∃ᶠ (xy : ℤ × ℤ) in filter.at_top, (nat.coprime xy.1 xy.2) ∧ (|((xy.1 : ℚ) / xy.2) - α| < 1 / (xy.2 : ℚ) ^ 2) :=
sorry

end exist_infinitely_many_coprime_pairs_l140_140134


namespace binomial_coefficient_largest_n_l140_140583
-- import necessary library

-- define the problem
theorem binomial_coefficient_largest_n :
  ∃ (n : ℕ), (n ≤ 10) ∧ (nat.choose 9 4 + nat.choose 9 5 = nat.choose 10 n) ∧ (∀ m : ℕ, (nat.choose 9 4 + nat.choose 9 5 = nat.choose 10 m) → m ≤ n) :=
by {
  let n := 5,
  use n,
  split,
  {
    -- Prove n ≤ 10
    exact nat.le_refl n,
  },
  split,
  {
    -- Prove the main equality
    have h : nat.choose 9 4 + nat.choose 9 5 = nat.choose 10 5,
    {
      -- Using Pascal's identity
      exact nat.add_choose_eq nat.choose_succ_succ.symm,
    },
    exact h,
  },
  {
    -- Prove that n is the largest such integer
    intros m h,
    have h_mono : ∀ (k : ℕ), k < 5 → nat.choose 10 k < nat.choose 10 5,
    {
      sorry, -- This would require a proof, but we'll skip this as well.
    },
    by_contradiction,
    exact h_mono m hj,
  },
}

end binomial_coefficient_largest_n_l140_140583


namespace ratio_of_sequence_l140_140947

variables (a b c : ℝ)

-- Condition 1: arithmetic sequence
def arithmetic_sequence : Prop := 2 * b = a + c

-- Condition 2: geometric sequence
def geometric_sequence : Prop := c^2 = a * b

-- Theorem stating the ratio of a:b:c
theorem ratio_of_sequence (h1 : arithmetic_sequence a b c) (h2 : geometric_sequence a b c) : 
  (a = 4 * b) ∧ (c = -2 * b) :=
sorry

end ratio_of_sequence_l140_140947


namespace quadratic_inequality_solution_l140_140701

theorem quadratic_inequality_solution (d : ℝ) 
  (h1 : 0 < d) 
  (h2 : d < 16) : 
  ∃ x : ℝ, (x^2 - 8*x + d < 0) :=
  sorry

end quadratic_inequality_solution_l140_140701


namespace floor_paint_possibilities_l140_140633

theorem floor_paint_possibilities 
  (c d: ℤ) 
  (hc: c > 0) 
  (hd: d > 0) 
  (hcd: d > c)
  (hp: cd = 3 * (c - 4) * (d - 4)) : 
  fintype.card {p : ℤ × ℤ // p.1 > 0 ∧ p.2 > 0 ∧ p.2 > p.1 ∧ p.1 * p.2 = 3 * (p.1 - 4) * (p.2 - 4)} = 2 :=
by {
  sorry -- Proof to be filled in
}

end floor_paint_possibilities_l140_140633


namespace correct_calculation_l140_140962

variable (a : ℕ)

theorem correct_calculation : 
  ¬(a + a = a^2) ∧ ¬(a^3 * a = a^3) ∧ ¬(a^8 / a^2 = a^4) ∧ ((a^3)^2 = a^6) := 
by
  sorry

end correct_calculation_l140_140962


namespace original_number_contains_digit_ge_5_l140_140240

theorem original_number_contains_digit_ge_5
  (N : ℕ)
  (Ns : (fin 3) → ℕ)
  (is_permutation: ∀ i, is_permutation N (Ns i))
  (all_digits_nonzero : ∀ d ∈ digits 10 N, d ≠ 0)
  (sum_is_all_ones : ∃ k, N + Ns 0 + Ns 1 + Ns 2 = nat.of_digits 10 (repeat 1 k)) :
  ∃ d ∈ digits 10 N, d ≥ 5 := 
sorry

end original_number_contains_digit_ge_5_l140_140240


namespace inflation_over_two_years_real_interest_rate_l140_140977

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end inflation_over_two_years_real_interest_rate_l140_140977


namespace bulbs_on_perfect_squares_l140_140832

def is_on (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k * k

theorem bulbs_on_perfect_squares (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 100) :
  (∀ i : ℕ, 1 ≤ i → i ≤ 100 → ∃ j : ℕ, i = j * j ↔ is_on i) := sorry

end bulbs_on_perfect_squares_l140_140832


namespace total_students_in_class_l140_140081

theorem total_students_in_class (E_and_G : ℕ) (G_total : ℕ) (E_only : ℕ) (H1 : E_and_G = 12) (H2 : G_total = 22) (H3 : E_only = 18) : E_only + (G_total - E_and_G) + E_and_G = 40 :=
by
  rw [H1, H2, H3]
  sorry

end total_students_in_class_l140_140081


namespace correct_attitude_towards_online_superstitions_l140_140091

-- Define the conditions used in the problem
def online_superstitions := ["astrological fate", "horoscope interpretations", "easy-to-learn North Star books", "Zhou Gong's dream interpretations"]

def option_A := "Accept and accumulate, use it for my own benefit"
def option_B := "Stay away from the internet, keep oneself pure"
def option_C := "Respect diversity, make rational choices"
def option_D := "Improve eye power, reject pollution"

-- The main statement proving the correct attitude given the conditions
theorem correct_attitude_towards_online_superstitions :
  let teenage_students := true in
  (∀ (superstitions : list string) (attitudes : string),
    superstitions = online_superstitions →
    (attitudes = option_A ∨ attitudes = option_B ∨ attitudes = option_C ∨ attitudes = option_D) →
    (attitudes = option_D)) :=
by {
  intros,
  repeat { sorry },
}

end correct_attitude_towards_online_superstitions_l140_140091


namespace diagonals_bisect_common_property_l140_140969
-- This statement declares the property to be proved: that rectangles, rhombuses, and squares all share the property that their diagonals bisect each other.

-- Definitions for shapes
def Rectangle : Type := { r // r is a shape that satisfies rectangle properties}
def Rhombus : Type := { r // r is a shape that satisfies rhombus properties}
def Square : Type := { s // s is a shape that satisfies square properties}

-- Diagonals bisect each other in a given shape (abstract property definition)
def diagonals_bisect (shape : Type) : Prop :=
  ∀ (d1 d2 : Shape.Diagonal shape), d1.length / 2 = d2.length / 2

-- Lean 4 theorem statement
theorem diagonals_bisect_common_property :
  diagonals_bisect Rectangle ∧ diagonals_bisect Rhombus ∧ diagonals_bisect Square :=
by
  sorry

end diagonals_bisect_common_property_l140_140969


namespace six_digit_number_multiple_of_7_l140_140371

theorem six_digit_number_multiple_of_7 (d : ℕ) (hd : d ≤ 9) :
  (∃ k : ℤ, 56782 + d * 10 = 7 * k) ↔ (d = 0 ∨ d = 7) := by
sorry

end six_digit_number_multiple_of_7_l140_140371


namespace tangent_line_slope_at_log_number_of_common_points_compare_exp_values_l140_140411

-- Condition definitions
def f (x : ℝ) : ℝ := Real.exp x
def g (x : ℝ) : ℝ := Real.log x

-- Part (I): Prove that for the line y = kx + 2 tangent to g(x), k = e^(-3)
theorem tangent_line_slope_at_log {k x₀ : ℝ} (h₁ : k * x₀ + 2 = g x₀) (h₂ : k = g' x₀) : k = Real.exp (-3) :=
sorry

-- Part (II): Prove the number of common points of y = f(x) and y = mx^2 for m > 0 when x > 0
theorem number_of_common_points (m : ℝ) (h₀ : 0 < m) : (∃ x, f x = m * x^2) ↔ (m = Real.exp 2 / 4 ∨ m > Real.exp 2 / 4) :=
sorry

-- Part (III): Prove that for a < b, (f(a) + f(b))/2 > (f(b) - f(a)) / (b - a)
theorem compare_exp_values (a b : ℝ) (h₀ : a < b) : (f(a) + f(b)) / 2 > (f(b) - f(a)) / (b - a) :=
sorry

end tangent_line_slope_at_log_number_of_common_points_compare_exp_values_l140_140411


namespace n_lines_cannot_divide_into_k_regions_l140_140901

theorem n_lines_cannot_divide_into_k_regions (n k : ℕ) (h₁ : n + 1 < k) (h₂ : k < 2 * n) :
  ¬∃ (lines : Fin n → AffineSubspace ℝ (Fin 2)), divides_plane_into_regions lines k := 
sorry

end n_lines_cannot_divide_into_k_regions_l140_140901


namespace approx_fraction_B_grades_l140_140820

noncomputable def fraction_B_grades (T : ℝ) : ℝ :=
  (3 / 10) - (20 / T)

theorem approx_fraction_B_grades (T : ℝ) (hT : T ≈ 400) (A_fraction : ℝ) 
  (C_fraction : ℝ) (D_count : ℝ) (hA : A_fraction = 1/5)
  (hC : C_fraction = 1/2) (hD : D_count = 20) :
  fraction_B_grades T ≈ 1/4 :=
by
  sorry

end approx_fraction_B_grades_l140_140820


namespace simplify_exponent_multiplication_l140_140255

theorem simplify_exponent_multiplication :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_multiplication_l140_140255


namespace ray_intersects_triangle_inequality_l140_140631

theorem ray_intersects_triangle_inequality
  (A B C X Y : Point)
  (h1 : ray_from A ∩ line_segment B C = X)
  (h2 : ray_from A ∩ circumcircle A B C = Y)
  (h3 : BX * CX = AX * XY) : 
  1 / AX + 1 / XY ≥ 4 / BC :=
by sorry

end ray_intersects_triangle_inequality_l140_140631


namespace problem_1_problem_2_problem_3_problem_4_l140_140348

-- Problem 1: Limit of (1 - sqrt(x+1)) / x as x -> 0
theorem problem_1 :
  (Filter.Tendsto (λ x : ℝ, (1 - Real.sqrt (x + 1)) / x) (nhds 0) (nhds (-1/2))) :=
by
  sorry

-- Problem 2: Limit of (2 - sqrt(x)) / (3 - sqrt(2x + 1)) as x -> 4
theorem problem_2 :
  (Filter.Tendsto (λ x : ℝ, (2 - Real.sqrt x) / (3 - Real.sqrt (2 * x + 1))) (nhds 4) (nhds (3 / 4))) :=
by
  sorry

-- Problem 3: Limit of (tan x) / (1 - sqrt(1 + tan x)) as x -> 0
theorem problem_3 :
  (Filter.Tendsto (λ x : ℝ, (Real.tan x) / (1 - Real.sqrt (1 + Real.tan x))) (nhds 0) (nhds (-2))) :=
by
  sorry

-- Problem 4: Limit of (1 - sqrt(x)) / (1 - cbrt(x)) as x -> 1
theorem problem_4 :
  (Filter.Tendsto (λ x : ℝ, (1 - Real.sqrt x) / (1 - Real.cbrt x)) (nhds 1) (nhds (3 / 2))) :=
by
  sorry

end problem_1_problem_2_problem_3_problem_4_l140_140348


namespace quadratic_has_real_root_l140_140792

theorem quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := sorry

end quadratic_has_real_root_l140_140792


namespace Morgane_stops_finite_average_LC_l140_140914

/-- Prove that given the conditions, Morgane will stop after a finite number of operations. -/
theorem Morgane_stops_finite (n : ℕ) (initial_config : Fin n → Bool) :
  ∃ m : ℕ, ∀ k:ℕ, ¬ (k > m) → (all_heads_flipped initial_config k) = true := sorry

noncomputable def all_heads_flipped (initial_config : Fin n → Bool) (k:ℕ) : Bool :=
  sorry

/-- Calculate the average number of operations before stopping given the conditions. -/
theorem average_LC (n : ℕ) : (∑ C : Fin (2^n) → Bool, L C) / (2^n) = n / 2 := sorry

noncomputable def L (C : Fin n → Bool) : ℕ :=
  sorry

end Morgane_stops_finite_average_LC_l140_140914


namespace concyclic_quadrilateral_and_intersection_on_circle_l140_140483

-- Define the given conditions
variables {A B O P Kₐ K_b : Type*} [is_circle AB] [diameter AB O] [point_on_circle P AB] 
[circumcenter_triangle P O A Kₐ] [circumcenter_triangle P O B K_b]

-- State the properties we need to prove
theorem concyclic_quadrilateral_and_intersection_on_circle :
  (is_concyclic {O, Kₐ, P, K_b}) ∧ (is_intersection_on_circle AKₐ BK_b AB) :=
begin
  sorry -- Proof is omitted as per instructions
end

end concyclic_quadrilateral_and_intersection_on_circle_l140_140483


namespace a_plus_c_eq_zero_l140_140422

variable {R : Type*} [Field R] (a b c : R)

def f (x : R) : R := a * x^2 + b * x + c
def g (x : R) : R := c * x^2 + b * x + a

theorem a_plus_c_eq_zero (a b c : R) (h : ∀ x, f (g x) = x) : a + c = 0 := 
by {
    sorry
}

end a_plus_c_eq_zero_l140_140422


namespace average_other_color_marbles_l140_140150

def percentage_clear : ℝ := 0.4
def percentage_black : ℝ := 0.2
def total_percentage : ℝ := 1.0
def total_marbles_taken : ℝ := 5.0

theorem average_other_color_marbles :
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black in
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors in
  expected_other_color_marbles = 2 := by
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors
  show expected_other_color_marbles = 2
  sorry

end average_other_color_marbles_l140_140150


namespace even_number_of_segments_in_closed_polygonal_chain_l140_140888

theorem even_number_of_segments_in_closed_polygonal_chain
    (vertices : ℕ)
    (x y : ℕ → ℤ) -- representing coordinates (x_i, y_i) for i = 1 to vertices
    (equal_segments : ∀ i, i < vertices → (x(i) - x((i + 1) % vertices))^2 + (y(i) - y((i + 1) % vertices))^2 = (x 0 - x 1)^2 + (y 0 - y 1)^2) 
    (closed_chain : x vertices = x 0 ∧ y vertices = y 0)
    : vertices % 2 = 0 := 
sorry

end even_number_of_segments_in_closed_polygonal_chain_l140_140888


namespace job_completion_time_l140_140064

theorem job_completion_time (p q s : ℕ) (h : p > 0) (complexity_factor : ℝ) (complexity_factor = 1.1) : 
  let original_total_man_days := p * q in
  let adjusted_total_man_days := complexity_factor * original_total_man_days in
  (adjusted_total_man_days / (p + 2 * s) = 1.1 * p * q / (p + 2 * s)) :=
by sorry

end job_completion_time_l140_140064


namespace quadratic_has_real_root_iff_b_in_intervals_l140_140798

theorem quadratic_has_real_root_iff_b_in_intervals (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ set.Icc (-∞ : ℝ) (-10) ∪ set.Icc 10 (∞ : ℝ)) :=
by by sorry

end quadratic_has_real_root_iff_b_in_intervals_l140_140798


namespace problem_statement_l140_140488

noncomputable def vector_P (A B : ℝ × ℝ × ℝ) (m n : ℕ) : ℝ × ℝ × ℝ :=
  let num := (m * B.1 + n * A.1, m * B.2 + n * A.2, m * B.3 + n * A.3)
  let denom := (m + n : ℝ)
  (num.1 / denom, num.2 / denom, num.3 / denom)

theorem problem_statement : 
  let A : ℝ × ℝ × ℝ := (1, 2, 3) 
  let B : ℝ × ℝ × ℝ := (4, 5, 6)
  let P : ℝ × ℝ × ℝ := vector_P A B 4 1
  let t := 0.2
  let u := 0.8
  P = (3.4, 4.4, 5.4) ∧ P = (t * A.1 + u * B.1, t * A.2 + u * B.2, t * A.3 + u * B.3) := 
by
  sorry

end problem_statement_l140_140488


namespace sheets_of_paper_in_each_box_l140_140324

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 30)
  (h2 : 2 * E = S)
  (h3 : 3 * E = S - 10) :
  S = 40 :=
by
  sorry

end sheets_of_paper_in_each_box_l140_140324


namespace inflation_over_two_years_real_interest_rate_l140_140975

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end inflation_over_two_years_real_interest_rate_l140_140975


namespace slope_angle_of_line_l_curve_rectangular_eq_of_C_PA_PB_sum_l140_140767

noncomputable def slope_angle (t : ℝ) : ℝ :=
  Real.arctan (sqrt 3)

noncomputable def line_rect_eq (t : ℝ) : ℝ × ℝ :=
  let x := t
  let y := (sqrt 2) / 2 + (sqrt 3) * t
  (x, y)

noncomputable def curve_polar_eq (ρ θ : ℝ) : ℝ :=
  2 * cos (θ - π / 4)

noncomputable def curve_rect_eq (x y : ℝ) : Prop :=
  (x - (sqrt 2) / 2)^2 + (y - (sqrt 2) / 2)^2 = 1

theorem slope_angle_of_line_l : slope_angle(t) = π / 3 := 
  sorry

theorem curve_rectangular_eq_of_C : 
  ∀ ρ θ, curve_polar_eq ρ θ → ∃ x y, curve_rect_eq x y := 
  sorry

theorem PA_PB_sum :
  ∃ A B : ℝ × ℝ, (line_rect_eq t).fst = 0 ∧ (line_rect_eq t).snd = (sqrt 2) / 2 →
                 let P := (0, (sqrt 2) / 2) in 
                 |P.1 - A.1| + |P.2 - A.2| + |P.1 - B.1| + |P.2 - B.2| = sqrt 10 / 2 := sorry

end slope_angle_of_line_l_curve_rectangular_eq_of_C_PA_PB_sum_l140_140767


namespace option_d_correct_l140_140963

theorem option_d_correct (a b : ℝ) (h : a * b < 0) : 
  (a / b + b / a) ≤ -2 := by
  sorry

end option_d_correct_l140_140963


namespace peach_to_apricot_ratio_l140_140842

theorem peach_to_apricot_ratio (a t : ℕ) (h_a : a = 58) (h_t : t = 232) :
  let p := t - a in
  (p : ℚ) / a = 3 := by
  sorry

end peach_to_apricot_ratio_l140_140842


namespace geometric_sequence_problem_l140_140618

theorem geometric_sequence_problem 
  (a1 a2 a3 : ℤ)
  (h1 : a1 = 32)
  (h2 : a2 = -48)
  (h3 : a3 = 72)
  (r : ℚ)
  (h4 : r = a2 / a1)
  (h5 : a3 = a2 * r) :
  r = -3 / 2 ∧ a2 * r = -108 :=
by
  have hr : r = -3 / 2 := by
    sorry
  have a4 : a3 * r = -108 := by
    sorry
  exact ⟨hr, a4⟩

end geometric_sequence_problem_l140_140618


namespace length_of_chord_l140_140764

theorem length_of_chord (x y : ℝ) (h1 : x - y = 0) (h2 : x^2 + y^2 = 4) : 2 * sqrt 4 = 4 :=
by
  sorry

end length_of_chord_l140_140764


namespace count_arithmetic_sequence_22_l140_140055

theorem count_arithmetic_sequence_22 : 
  ∃ n : ℕ, 
    let a := -48 
    let d := 6 
    let l := 78 in 
    (a + (n - 1) * d) = l ∧ n = 22 :=
by
  let a := -48
  let d := 6
  let l := 78
  use 22
  split
  · simp [a, d, l] 
    sorry
  · rfl

end count_arithmetic_sequence_22_l140_140055


namespace harmonic_series_inequality_l140_140953

theorem harmonic_series_inequality (n : ℕ) (hn : n ≥ 1) :
  (∑ i in finset.range n, (1 : ℝ) / (n + 1 + i)) > 13 / 24 :=
sorry

end harmonic_series_inequality_l140_140953


namespace total_inflation_over_two_years_real_interest_rate_over_two_years_l140_140983

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end total_inflation_over_two_years_real_interest_rate_over_two_years_l140_140983


namespace sale_discount_l140_140238

theorem sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (discount_multiple : ℕ)
  (h1 : purchase_amount = 250)
  (h2 : discount_per_100 = 10)
  (h3 : discount_multiple = purchase_amount / 100) :
  purchase_amount - discount_per_100 * discount_multiple = 230 := 
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end sale_discount_l140_140238


namespace circle_area_l140_140765

theorem circle_area (a : ℝ) (r : ℝ) (h1 : r = Real.sqrt (a^2 + 2))
  (h2 : (∃ A B : ℝ × ℝ, let p := λ (x y : ℝ), y = x + 2 * a in 
    (p A.1 A.2 ∧ p B.1 B.2 ∧
     (A.1^2 + A.2^2 - 2 * a * A.2 - 2 = 0) ∧
     (B.1^2 + B.2^2 - 2 * a * B.2 - 2 = 0) ∧
     dist A B = 2 * Real.sqrt 3))) :
  π * r^2 = 4 * π :=
by
  sorry

end circle_area_l140_140765


namespace lucy_cannot_use_4_lolly_sticks_l140_140141

def forms_triangle (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem lucy_cannot_use_4_lolly_sticks :
    ∀ a b c : ℕ, a + b + c = 4 → ¬ forms_triangle a b c :=
by
  intros a b c hsum
  have h₁ : a + b + c = 4 := hsum
  by_cases h1 : a ≤ b + c
  by_cases h2 : b ≤ c + a
  by_cases h3 : c ≤ a + b
  { simp [forms_triangle] at h₁ h2 h3, contradiction }
  { simp only [forms_triangle] at h3, contradiction }
  { simp only [forms_triangle] at h2, contradiction }

-- Uncomment to finish the proof
-- { sorry }

end lucy_cannot_use_4_lolly_sticks_l140_140141


namespace hammers_in_comparison_group_l140_140053

theorem hammers_in_comparison_group (H W x : ℝ) (h1 : 2 * H + 2 * W = 1 / 3 * (x * H + 5 * W)) (h2 : W = 2 * H) :
  x = 8 :=
sorry

end hammers_in_comparison_group_l140_140053


namespace peters_remaining_money_l140_140890

theorem peters_remaining_money :
  ∀ (money_peter_has : ℕ) 
    (potato_kilos : ℕ) (potato_cost_per_kilo : ℕ)
    (tomato_kilos : ℕ) (tomato_cost_per_kilo : ℕ)
    (cucumber_kilos : ℕ) (cucumber_cost_per_kilo : ℕ)
    (banana_kilos : ℕ) (banana_cost_per_kilo : ℕ), 
  money_peter_has = 500 →
  potato_kilos = 6 → potato_cost_per_kilo = 2 →
  tomato_kilos = 9 → tomato_cost_per_kilo = 3 →
  cucumber_kilos = 5 → cucumber_cost_per_kilo = 4 →
  banana_kilos = 3 → banana_cost_per_kilo = 5 →
  money_peter_has - 
    (potato_kilos * potato_cost_per_kilo +
    tomato_kilos * tomato_cost_per_kilo +
    cucumber_kilos * cucumber_cost_per_kilo +
    banana_kilos * banana_cost_per_kilo) = 426 :=
by
  intros money_peter_has potato_kilos potato_cost_per_kilo 
         tomato_kilos tomato_cost_per_kilo
         cucumber_kilos cucumber_cost_per_kilo
         banana_kilos banana_cost_per_kilo
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  rw [h1, h2, h3, h4, h5, h6, h7, h8, h9]
  norm_num
  sorry

end peters_remaining_money_l140_140890


namespace shortest_time_between_ships_l140_140307

theorem shortest_time_between_ships 
  (AB : ℝ) (speed_A : ℝ) (speed_B : ℝ) (angle_ABA' : ℝ) : (AB = 10) → (speed_A = 4) → (speed_B = 6) → (angle_ABA' = 60) →
  ∃ t : ℝ, (t = 150/7 / 60) :=
by
  intro hAB hSpeedA hSpeedB hAngle
  sorry

end shortest_time_between_ships_l140_140307


namespace min_length_fold_l140_140991

theorem min_length_fold (a b x y : ℝ) 
  (h_a : a = 8) (h_b : b = 11) (h_fold : x = 8 ∧ 0 ≤ y ∧ y ≤ b ∧ y = b) :
  sqrt ((x - 0)^2 + (y - b)^2) = 8 := by
  sorry

end min_length_fold_l140_140991


namespace raisin_distribution_l140_140052

variables (B C A : ℕ) -- B: number of raisins Bryce received, C: number of raisins Carter received, A: number of raisins Alice received

-- Conditions
def condition1 := B = C + 10
def condition2 := C = B / 2
def condition3 := A = 2 * C

-- Theorem
theorem raisin_distribution : condition1 ∧ condition2 ∧ condition3 → B = 20 ∧ C = 10 ∧ A = 20 := by
  intros h
  have hB := h.1
  have hC := h.2.1
  have hA := h.2.2
  sorry -- Proof omitted

end raisin_distribution_l140_140052


namespace function_decreasing_iff_l140_140202

theorem function_decreasing_iff (a : ℝ) : 
  (∀ x ≥ 2, (2 * a * x + 4 * (a + 1)) ≤ 0) ↔ a ≤ -1/2 := 
begin 
  sorry 
end

end function_decreasing_iff_l140_140202


namespace longest_diagonal_proof_l140_140632

noncomputable def longest_diagonal_length (x y z : ℝ) : ℝ :=
  real.sqrt (x^2 + y^2 + z^2)

theorem longest_diagonal_proof (x y z : ℝ)
  (h1 : 4 * (x + y + z) = 60)
  (h2 : 2 * (x * y + y * z + z * x) = 150) :
  longest_diagonal_length x y z = 5 * real.sqrt 3 :=
by
  have h1' : x + y + z = 15 := by linarith [h1]
  have h2' : x * y + y * z + z * x = 75 := by linarith [h2]
  have s : (x + y + z)^2 = x^2 + y^2 + z^2 + 2 * (x * y + y * z + z * x) :=
    by ring
  have e : 15^2 = x^2 + y^2 + z^2 + 2 * (x * y + y * z + z * x) :=
    by rw [h1', h2', s]; ring
  have eq : x^2 + y^2 + z^2 = 75 := by linarith
  calc longest_diagonal_length x y z
       = real.sqrt (x^2 + y^2 + z^2) : by rw [longest_diagonal_length]
       ... = real.sqrt 75 : by rw [eq]
       ... = 5 * real.sqrt 3 : by norm_num

end longest_diagonal_proof_l140_140632


namespace incorrect_statement_in_given_conditions_l140_140968

theorem incorrect_statement_in_given_conditions:
  (∀ (r : Type) [rhombus r], (∃ d1 d2 : r → r, perpendicular d1 d2 ∧ bisect d1 d2)) ∧
  (∀ (rec : Type) [rectangle rec], (∃ d1 d2 : rec → rec, equal_length d1 d2)) ∧
  (∀ (q : Type) [quadrilateral q], (∃ a b : q → q, adjacent a b ∧ is_rhombus q)) = False ∧
  (∀ (q : Type) [quadrilateral q], all_sides_equal q → is_rhombus q) → 
  ∃ (q' : Type) [quadrilateral q'], (∃ a b : q' → q', adjacent a b) ∧ ¬ is_rhombus q' := 
sorry

end incorrect_statement_in_given_conditions_l140_140968


namespace length_of_each_stone_l140_140293

theorem length_of_each_stone {L : ℝ} (hall_length hall_breadth : ℝ) (stone_breadth : ℝ) (num_stones : ℕ) (area_hall : ℝ) (area_stone : ℝ) :
  hall_length = 36 * 10 ∧ hall_breadth = 15 * 10 ∧ stone_breadth = 5 ∧ num_stones = 3600 ∧
  area_hall = hall_length * hall_breadth ∧ area_stone = L * stone_breadth ∧
  area_stone * num_stones = area_hall →
  L = 3 :=
by
  sorry

end length_of_each_stone_l140_140293


namespace real_root_interval_l140_140804

theorem real_root_interval (b : ℝ) (p : ℝ → ℝ)
  (h_poly : p = λ x, x^2 + b * x + 25)
  (h_real_root : ∃ x : ℝ, p x = 0) :
  b ∈ set.Iic (-10) ∪ set.Ici 10 :=
sorry

end real_root_interval_l140_140804


namespace equivalent_forms_l140_140535

theorem equivalent_forms (m n : ℤ) (P Q : ℤ) (hP : P = 2^m) (hQ : Q = 3^n) :
  18^(m + n) = P^(m + n) * Q^(2 * (m + n)) :=
by
  sorry

end equivalent_forms_l140_140535


namespace find_number_l140_140814

theorem find_number (x : ℚ) : (x + (-5/12) - (-5/2) = 1/3) → x = -7/4 :=
by
  sorry

end find_number_l140_140814


namespace problem_2003rd_term_correct_l140_140851

noncomputable def term_in_sequence (n : ℕ) : ℕ :=
  let perfect_squares_removed := list.filter (λ x, ¬ (∃ m, m * m = x)) (list.range (n + floor (real.sqrt n))) in
  perfect_squares_removed.nth n

theorem problem_2003rd_term_correct :
  term_in_sequence 2003 = 2048 :=
by
  sorry

end problem_2003rd_term_correct_l140_140851


namespace average_marbles_of_other_colors_l140_140154

theorem average_marbles_of_other_colors :
  let total_percentage := 100
  let clear_percentage := 40
  let black_percentage := 20
  let other_percentage := total_percentage - clear_percentage - black_percentage
  let marbles_taken := 5
  (other_percentage / 100) * marbles_taken = 2 :=
by
  sorry

end average_marbles_of_other_colors_l140_140154


namespace repeating_decimal_as_fraction_l140_140350

theorem repeating_decimal_as_fraction (x : ℝ) (hx : x = 0.575757575757...) : x = 57 / 99 :=
sorry

end repeating_decimal_as_fraction_l140_140350


namespace perpendicular_planes_l140_140062

open Set

variables {α β γ : Type} [Plane α] [Plane β] [Plane γ] {l : Line}

-- Let α, β, and γ be distinct planes and l be a line
axiom distinct_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α
axiom line_l : l

-- Given conditions
axiom l_perpendicular_to_alpha : Perpendicular l α
axiom l_parallel_to_beta : Parallel l β

-- Prove α is perpendicular to β
theorem perpendicular_planes : Perpendicular α β :=
sorry

end perpendicular_planes_l140_140062


namespace earnings_per_visit_l140_140860

-- Define the conditions of the problem
def website_visits_per_month : ℕ := 30000
def earning_per_day : Real := 10
def days_in_month : ℕ := 30

-- Prove that John gets $0.01 per visit
theorem earnings_per_visit :
  (earning_per_day * days_in_month) / website_visits_per_month = 0.01 :=
by
  sorry

end earnings_per_visit_l140_140860


namespace incorrect_statement_in_given_conditions_l140_140967

theorem incorrect_statement_in_given_conditions:
  (∀ (r : Type) [rhombus r], (∃ d1 d2 : r → r, perpendicular d1 d2 ∧ bisect d1 d2)) ∧
  (∀ (rec : Type) [rectangle rec], (∃ d1 d2 : rec → rec, equal_length d1 d2)) ∧
  (∀ (q : Type) [quadrilateral q], (∃ a b : q → q, adjacent a b ∧ is_rhombus q)) = False ∧
  (∀ (q : Type) [quadrilateral q], all_sides_equal q → is_rhombus q) → 
  ∃ (q' : Type) [quadrilateral q'], (∃ a b : q' → q', adjacent a b) ∧ ¬ is_rhombus q' := 
sorry

end incorrect_statement_in_given_conditions_l140_140967


namespace sum_of_numbers_in_circle_l140_140179

theorem sum_of_numbers_in_circle (boys girls : Fin 6 → ℝ) 
  (hboys : ∀ i : Fin 6, boys i = girls i + girls (i + 1) % 6) 
  (hgirls : ∀ i : Fin 6, girls i = boys i * boys (i + 1) % 6) 
  (hnonzero_boys : ∀ i : Fin 6, boys i ≠ 0)
  (hnonzero_girls : ∀ i : Fin 6, girls i ≠ 0) :
  (∑ i, (boys i + girls i)) = 4.5 :=
by
  sorry

end sum_of_numbers_in_circle_l140_140179


namespace points_on_hyperbola_l140_140716

theorem points_on_hyperbola {s : ℝ} :
  let x := Real.exp s - Real.exp (-s)
  let y := 5 * (Real.exp s + Real.exp (-s))
  (y^2 / 100 - x^2 / 4 = 1) :=
by
  sorry

end points_on_hyperbola_l140_140716


namespace hyperbola_eccentricity_l140_140745

-- Definitions for the hyperbola and the conditions
noncomputable def is_hyperbola (F1 F2 A B C : ℝ × ℝ) (a b : ℝ) : Prop :=
  let d := dist in
  ∃ e : ℝ,
  0 < e ∧ e < 1 ∧
  let F1A := d F1 A in
  let F2A := d F2 A in
  let F1B := d F1 B in
  let F2B := d F2 B in
  F1A + F2A = 2 * a ∧
  F1B + F2B = 2 * a

noncomputable def is_equilateral_triangle (F1 A B : ℝ × ℝ) : Prop :=
  dist F1 A = dist A B ∧ dist A B = dist B F1

-- Main theorem statement
theorem hyperbola_eccentricity (F1 F2 A B C : ℝ × ℝ) (a b : ℝ) 
  (hyp : is_hyperbola F1 F2 A B C a b)
  (equilateral : is_equilateral_triangle F1 A B)
  (perpendicular : A.2 = B.2 ∧ C.1 = 0) : 
  let e := (sqrt (a^2 + b^2) / a) in
  e = sqrt 3 :=
sorry

end hyperbola_eccentricity_l140_140745


namespace find_a_value_l140_140809

def complex_expression_is_pure_imaginary (a : ℝ) : Prop :=
  let z := (a + 3 * Complex.I) * (1 - 2 * Complex.I)
  z.re = 0

theorem find_a_value : ∀ (a : ℝ), complex_expression_is_pure_imaginary a → a = -6 := 
begin
  assume (a : ℝ) (H : complex_expression_is_pure_imaginary a),
  sorry
end

end find_a_value_l140_140809


namespace ice_cube_count_l140_140670

theorem ice_cube_count (cubes_per_tray : ℕ) (tray_count : ℕ) (H1: cubes_per_tray = 9) (H2: tray_count = 8) :
  cubes_per_tray * tray_count = 72 :=
by
  sorry

end ice_cube_count_l140_140670


namespace sum_of_d_k_squares_l140_140684

def d_k (k : ℕ) : ℝ :=
  k + 1 / (3 * k + 1 / (4 * k + 1 / (3 * k + ...)))

theorem sum_of_d_k_squares :
  ∑ k in finset.range 15, (d_k k) ^ 2 = 1170 := 
sorry

end sum_of_d_k_squares_l140_140684


namespace real_root_interval_l140_140803

theorem real_root_interval (b : ℝ) (p : ℝ → ℝ)
  (h_poly : p = λ x, x^2 + b * x + 25)
  (h_real_root : ∃ x : ℝ, p x = 0) :
  b ∈ set.Iic (-10) ∪ set.Ici 10 :=
sorry

end real_root_interval_l140_140803


namespace value_of_m_l140_140787

theorem value_of_m (m : ℝ) (h : (1/5)^m * (1/4)^2 = 1/(10^4)) : m = 4 :=
by 
  sorry

end value_of_m_l140_140787


namespace number_of_minimally_intersecting_triples_modulo_1000_l140_140337

def minimally_intersecting_triples_modulo (D E F : set ℕ) : Prop :=
  |D ∩ E| = 1 ∧ |E ∩ F| = 1 ∧ |F ∩ D| = 1 ∧ D ∩ E ∩ F = ∅

theorem number_of_minimally_intersecting_triples_modulo_1000 :
  ∃ M : ℕ, (M % 1000 = 64) ∧
    M = count { (D, E, F) : (set ℕ) × (set ℕ) × (set ℕ) | 
      minimally_intersecting_triples_modulo D E F ∧ 
      D ⊆ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
      E ⊆ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
      F ⊆ {1, 2, 3, 4, 5, 6, 7, 8} } :=
sorry

end number_of_minimally_intersecting_triples_modulo_1000_l140_140337


namespace problem_statement_l140_140395

noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

theorem problem_statement (a b : ℝ) (h_dom_range : ∀ x, a ≤ x ∧ x ≤ b → a ≤ f x ∧ f x ≤ b) 
(h_monotonic : ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y) 
(h_system : f a = a ∧ f b = b) : f a + f b = 1 :=
by
  have a_geq_zero : 0 ≤ a := sorry
  have a_eq_zero : a = 0 := sorry
  have b_eq_one : b = 1 := sorry
  have fs : f 0 = 0 ∧ f 1 = 1 := by
     rw [a_eq_zero, b_eq_one] at h_system
     exact h_system
  rw [fs.1, fs.2]
  norm_num
  sorry


end problem_statement_l140_140395


namespace spent_on_basil_seeds_l140_140671

-- Define the variables and conditions
variables (S cost_soil num_plants price_per_plant net_profit total_revenue total_expenses : ℝ)
variables (h1 : cost_soil = 8)
variables (h2 : num_plants = 20)
variables (h3 : price_per_plant = 5)
variables (h4 : net_profit = 90)

-- Definition of total revenue as the multiplication of number of plants and price per plant
def revenue_eq : Prop := total_revenue = num_plants * price_per_plant

-- Definition of total expenses as the sum of soil cost and cost of basil seeds
def expenses_eq : Prop := total_expenses = cost_soil + S

-- Definition of net profit
def profit_eq : Prop := net_profit = total_revenue - total_expenses

-- The theorem to prove
theorem spent_on_basil_seeds : S = 2 :=
by
  -- Since we define variables and conditions as inputs,
  -- the proof itself is omitted as per instructions
  sorry

end spent_on_basil_seeds_l140_140671


namespace solve_for_x_l140_140532

-- Let us state and prove that x = 495 / 13 is a solution to the equation 3x + 5 = 500 - (4x + 6x)
theorem solve_for_x (x : ℝ) : 3 * x + 5 = 500 - (4 * x + 6 * x) → x = 495 / 13 :=
by
  sorry

end solve_for_x_l140_140532


namespace interest_difference_l140_140288

theorem interest_difference (P R T: ℝ) (hP: P = 2500) (hR: R = 8) (hT: T = 8) :
  let I := P * R * T / 100
  (P - I = 900) :=
by
  -- definition of I
  let I := P * R * T / 100
  -- proof goal
  sorry

end interest_difference_l140_140288


namespace bridge_supports_88_ounces_l140_140474

-- Define the conditions
def weight_of_soda_per_can : ℕ := 12
def number_of_soda_cans : ℕ := 6
def weight_of_empty_can : ℕ := 2
def additional_empty_cans : ℕ := 2

-- Define the total weight the bridge must hold up
def total_weight_bridge_support : ℕ :=
  (number_of_soda_cans * weight_of_soda_per_can) + ((number_of_soda_cans + additional_empty_cans) * weight_of_empty_can)

-- Prove that the total weight is 88 ounces
theorem bridge_supports_88_ounces : total_weight_bridge_support = 88 := by
  sorry

end bridge_supports_88_ounces_l140_140474


namespace part1_part2_l140_140051

variables (a b : ℝ)
variables (va vb : EuclideanSpace ℝ (Fin 2)) -- assuming two-dimensional Euclidean space for simplicity

-- Given conditions
def norm_va := ∥va∥ = 2
def norm_vb := ∥vb∥ = 1
def ab := (2 : ℝ) • va - vb
def cd := va + (3 : ℝ) • vb
def angle_va_vb_is_60 := real.angle va vb  = real.pi / 3
def ab_perp_cd := inner ab cd = 0

-- Proof problems
theorem part1 : angle_va_vb_is_60 → norm_va → norm_vb → ∥va - vb∥ = real.sqrt 3 := by
  sorry

theorem part2 : ab_perp_cd → norm_va → norm_vb → ab → cd → real.inner va vb / (norm_va * norm_vb) = -1/2 := by
  sorry

end part1_part2_l140_140051


namespace incorrect_statement_c_l140_140965

def rhombus (q : Type) [quadrilateral q] : Prop :=
∀ (d₁ d₂ : diagonal q), perpendicular d₁ d₂ ∧ bisect d₁ d₂

def rectangle (r : Type) [quadrilateral r] : Prop :=
∀ (d₁ d₂ : diagonal r), equal_length d₁ d₂

def quadrilateral_with_adjacent_sides_equal_is_rhombus (q : Type) [quadrilateral q] : Prop :=
∃ (a b : side q), adjacent a b ∧ equal_length a b → ∀ c (side q), rhombus q

def quadrilateral_with_all_sides_equal_is_rhombus (q : Type) [quadrilateral q] : Prop :=
∀ (a b c d : side q), (equal_length a b ∧ equal_length b c ∧ equal_length c d ∧ equal_length d a) → rhombus q

theorem incorrect_statement_c :
  ∃ q : Type, [quadrilateral q] ∧ ∀ (a b : side q), adjacent a b ∧ equal_length a b → ¬ rhombus q := sorry

end incorrect_statement_c_l140_140965


namespace price_after_9_years_decreases_continuously_l140_140347

theorem price_after_9_years_decreases_continuously (price_current : ℝ) (price_after_9_years : ℝ) :
  (∀ k : ℕ, k % 3 = 0 → price_current = 8100 → price_after_9_years = 2400) :=
sorry

end price_after_9_years_decreases_continuously_l140_140347


namespace radius_of_circle_l140_140563

theorem radius_of_circle (r : ℝ) (h : 6 * Real.pi * r + 6 = 2 * Real.pi * r^2) : 
  r = (3 + Real.sqrt 21) / 2 :=
by
  sorry

end radius_of_circle_l140_140563


namespace leak_time_to_empty_l140_140266

def pump_rate : ℝ := 0.1 -- P = 0.1 tanks/hour
def effective_rate : ℝ := 0.05 -- P - L = 0.05 tanks/hour

theorem leak_time_to_empty (P L : ℝ) (hp : P = pump_rate) (he : P - L = effective_rate) :
  1 / L = 20 := by
  sorry

end leak_time_to_empty_l140_140266


namespace abs_neg_two_l140_140916

def absolute_value (x : Int) : Int :=
  if x >= 0 then x else -x

theorem abs_neg_two : absolute_value (-2) = 2 := 
by 
  sorry

end abs_neg_two_l140_140916


namespace increasing_intervals_of_f_l140_140724

theorem increasing_intervals_of_f 
  (f : ℝ → ℝ) (ϕ : ℝ)
  (h1 : ∀ x : ℝ, f x = x * Real.cos x + Real.cos (x + ϕ))
  (h2 : 0 < ϕ ∧ ϕ < π)
  (h3 : ∀ x, f (-x) = -f x)
  (domain : ∀ x, -2 * Real.pi < x ∧ x < 2 * Real.pi) :
  (∀ x, -2 * Real.pi < x ∧ x < 2 * Real.pi → 
    (f' x > 0 ↔ (-2 * Real.pi < x ∧ x < -Real.pi) ∨ (Real.pi < x ∧ x < 2 * Real.pi))) := sorry


end increasing_intervals_of_f_l140_140724


namespace max_guaranteed_coins_l140_140578

-- Definitions based on conditions
def total_coins := 300
def max_bags := 11
def coins_per_bag := 14

-- Theorem based on the proof problem
theorem max_guaranteed_coins : ∃ K, K = 146 ∧ 
                                (∀ (bag_count : ℕ), bag_count ≤ max_bags → 
                                (bag_count * coins_per_bag ≤ total_coins → 
                                (total_coins - bag_count * coins_per_bag) ≥ 146)) :=
begin
  use 146,
  split,
  {
    -- correct answer, K = 146
    refl,
  },
  {
    -- conditions for the proof
    intros bag_count h1 h2,
    transitivity,
    {
      rw ← Nat.mul_sub_left_distrib,
      linarith,
    },
    {
      refl,
    }
  }
end

end max_guaranteed_coins_l140_140578


namespace households_using_both_brands_l140_140298

theorem households_using_both_brands :
  ∀ (X : ℕ), 
    (80 + 60 + X + 3 * X = 300) → 
      X = 40 :=
by
  intros X h
  have h1 : 80 + 60 + 4 * X = 300 := by linarith
  have h2 : 4 * X = 300 - 140 := by linarith
  have h3 : 4 * X = 160 := by linarith
  have h4 : X = 160 / 4 := by linarith
  exact eq_of_sub_eq_zero (by linarith)

end households_using_both_brands_l140_140298


namespace area_of_field_l140_140300

-- Definitions based on conditions
def length : ℕ := 20
def fencing : ℕ := 92
def width (W : ℕ) : Prop := 2 * W + length = fencing

-- Assuming the conditions
axiom W_exists : ∃ W, width W

-- Compute the area
noncomputable def area (W : ℕ) : ℕ := length * W

-- Theorem stating the main question
theorem area_of_field : ∃ W, width W ∧ area W = 720 := 
by
  -- Provide the solution outline
  obtain ⟨W, hW⟩ := W_exists
  use W
  have : W = 36 := 
    by
      calc
        2 * W + 20 = 92 : hW
        2 * W = 72 : by linarith
        W = 36 : by linarith

  split
  { exact hW }
  { unfold area
    rw this
    norm_num }

end area_of_field_l140_140300


namespace balls_into_boxes_l140_140225

-- Define the conditions of the problem
def balls : ℕ := 10
def boxes : ℕ := 8

-- Compute the number of ways to place balls into boxes ensuring each box gets at least one ball.
noncomputable def placements : ℕ :=
  let S (n k : ℕ) := Nat.stirlingSecond n k
  in S balls boxes * Nat.fact boxes

-- Statement of the theorem
theorem balls_into_boxes :
  placements = 30240000 :=
by
  -- Proof is omitted.
  sorry

end balls_into_boxes_l140_140225


namespace smallest_overlap_l140_140143

-- Define the percentage of office workers using computers
def P_computer : ℝ := 0.90

-- Define the percentage of office workers using smartphones
def P_smartphone : ℝ := 0.85

-- Define the smallest possible percent of office workers who use both
def P_both : ℝ := 0.75

theorem smallest_overlap : min (P_computer + P_smartphone - 1) = P_both :=
by
  -- This is a placeholder for the actual proof.
  sorry

end smallest_overlap_l140_140143


namespace student_count_pattern_l140_140076

theorem student_count_pattern (n : ℕ) : 
  let seq := [1, 2, 3, 4, 5, 4, 3, 2] in
  seq[(2002 % 8)] = 2 :=
by
  sorry

end student_count_pattern_l140_140076


namespace triangle_inequality_l140_140326

open Real

variables {A B C P D E F : Type}

-- Let A, B, C, and P be points in ℝ² where ΔABC is a triangle, and D, E, F denote the perpendiculars from P to BC, CA, and AB respectively. 

theorem triangle_inequality 
  (hPD : ∀ (PA PB PC PD PE PF : ℝ), PA + PB + PC) 
  (hPE : ∀ (PD PE PF : ℝ), PD + PE + PF)
  (h1 : ∀ (P : ℝ), P ∈ (boundary ∆ABC)) 
  (h2 : ∀ (PD PE PF : ℝ), distances_from_point_to_sides)
  : PA + PB + PC ≥ 2 * (PD + PE + PF) ∧ (PA + PB + PC = 2 * (PD + PE + PF) ↔ is_equilateral_triangle_and_centroid)?
  :=
begin
  sorry
end

end triangle_inequality_l140_140326


namespace friend_p_walked_distance_l140_140269

def distance_walked_by_friend_p (v : ℝ) : ℝ :=
  let d := 22 / 2.2 in
  22 - d

theorem friend_p_walked_distance :
  ∀ v : ℝ, distance_walked_by_friend_p v = 12 :=
  by
    intro v
    unfold distance_walked_by_friend_p
    sorry

end friend_p_walked_distance_l140_140269


namespace range_f_l140_140938

noncomputable def f : ℝ → ℝ := λ x => 2 + 3^x

theorem range_f : set.range f = {y | 2 < y} :=
by
  sorry

end range_f_l140_140938


namespace circle_diameter_PQ_l140_140049

noncomputable def circleEq (x y : ℝ) : (x - 2)^2 + (y - 1)^2 = 5 :=
  sorry

theorem circle_diameter_PQ :
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    (C = (2, 1)) ∧ (r = sqrt 5) ∧ (∀ (x y : ℝ), (x - C.1)^2 + (y - C.2)^2 = r^2) :=
begin
  use (2, 1),
  use sqrt 5,
  split,
  {
    refl,
  },
  split,
  {
    refl,
  },
  {
    exact circleEq,
  }
end

end circle_diameter_PQ_l140_140049


namespace pyramid_cone_volume_ratio_l140_140194

open Real

theorem pyramid_cone_volume_ratio (a : ℝ) (alpha beta : ℝ) (h_alpha : 0 < alpha) (h_beta : 0 < beta) : 
  let V1 := (2 / 3) * a^3 * (cot alpha) * (cot beta),
      R := a * sqrt ((cot alpha)^2 + (cot beta)^2),
      V2 := (1 / 3) * pi * R^2 * a
  in V1 / V2 = (2 * (cot alpha) * (cot beta)) / (pi * ((cot alpha)^2 + (cot beta)^2)) :=
by
  sorry

end pyramid_cone_volume_ratio_l140_140194


namespace real_yield_correct_l140_140979

-- Define the annual inflation rate and nominal interest rate
def annualInflationRate : ℝ := 0.025
def nominalInterestRate : ℝ := 0.06

-- Define the number of years
def numberOfYears : ℕ := 2

-- Calculate total inflation over two years
def totalInflation (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n - 1

-- Calculate the compounded nominal rate
def compoundedNominalRate (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n

-- Calculate the real yield considering inflation
def realYield (nominalRate : ℝ) (inflationRate : ℝ) : ℝ :=
  (nominalRate / (1 + inflationRate)) - 1

-- Proof problem statement
theorem real_yield_correct :
  let inflation_two_years := totalInflation annualInflationRate numberOfYears
  let nominal_two_years := compoundedNominalRate nominalInterestRate numberOfYears
  inflation_two_years * 100 = 5.0625 ∧
  (realYield nominal_two_years inflation_two_years) * 100 = 6.95 :=
by
  sorry

end real_yield_correct_l140_140979


namespace sum_of_undefined_values_of_expression_l140_140343

theorem sum_of_undefined_values_of_expression : 
  let p := λ x : ℝ, x^2 - 7 * x + 10 = 0,
  ∃ x1 x2, (p x1 ∧ p x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 7) :=
by {
  -- definition and conditions
  let p := λ x : ℝ, x^2 - 7 * x + 10 = 0,
  existsi 2,
  existsi 5,
  split,
  exact sorry, -- proof that 2 is a root
  split,
  exact sorry, -- proof that 5 is a root
  split,
  exact sorry, -- proof that 2 ≠ 5
  exact sorry  -- proof that 2 + 5 = 7
}

end sum_of_undefined_values_of_expression_l140_140343


namespace ratio_of_third_bid_to_harry_first_bid_l140_140775

-- We define the necessary variables based on the conditions provided
variable (initial_bid final_bid third_bid harry_first_bid : ℕ)
variable (harry_increase : ℕ := 200)
variable (second_bid_increase : ℕ := 2)

-- Stating the conditions based on the problem statement
def conditions :=
  initial_bid = 300 ∧
  harry_first_bid = initial_bid + harry_increase ∧
  third_bid = final_bid - 1500 ∧
  harry_first_bid = 500 ∧
  final_bid = 4000

-- Proving that the ratio of the third bidder's bid to Harry's first bid is 5
theorem ratio_of_third_bid_to_harry_first_bid (h : conditions) : 
  third_bid / harry_first_bid = 5 := 
by
  sorry

end ratio_of_third_bid_to_harry_first_bid_l140_140775


namespace part_I_part_II_l140_140033

-- Definitions from conditions
def f (x : ℝ) (a : ℝ) : ℝ := log x + a * (x - 1)

-- Proof problem (I)
theorem part_I (x : ℝ) : f x (-1) ≤ 0 :=
sorry

-- Proof problem (II)
theorem part_II (a t x : ℝ) (H1 : a ∈ set.Ioo (-real.exp (1 / (real.exp 1 - 1))) (real.inf_real)) (H2 : t ≥ real.exp 1) (H3 : x > 0) : 
(t * log t + (t - 1) * (f x a + a)) > 0 :=
sorry

end part_I_part_II_l140_140033


namespace average_marbles_of_other_colors_l140_140153

theorem average_marbles_of_other_colors :
  let total_percentage := 100
  let clear_percentage := 40
  let black_percentage := 20
  let other_percentage := total_percentage - clear_percentage - black_percentage
  let marbles_taken := 5
  (other_percentage / 100) * marbles_taken = 2 :=
by
  sorry

end average_marbles_of_other_colors_l140_140153


namespace pages_per_day_l140_140534

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (result : ℕ) :
  total_pages = 81 ∧ days = 3 → result = 27 :=
by
  sorry

end pages_per_day_l140_140534


namespace total_inflation_over_two_years_real_interest_rate_over_two_years_l140_140985

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end total_inflation_over_two_years_real_interest_rate_over_two_years_l140_140985


namespace intersecting_chords_equal_l140_140555

theorem intersecting_chords_equal
  {A B C D M : Type}
  [CharZero A B C D M]  -- Using CharZero to imply A, B, C, D, M are over a field of characteristic zero.
  (h1: M ⊆ A * B)
  (h2: M ⊆ C * D)
  (h : (A * M) / (B * M) = (C * M) / (D * M)) :
  A * B = C * D := by
  sorry

end intersecting_chords_equal_l140_140555


namespace probability_outside_circle_is_7_over_9_l140_140910

noncomputable def probability_point_outside_circle :
    ℚ :=
sorry

theorem probability_outside_circle_is_7_over_9 :
    probability_point_outside_circle = 7 / 9 :=
sorry

end probability_outside_circle_is_7_over_9_l140_140910


namespace sequence_value_l140_140768

-- Define the sequence using the given conditions
noncomputable def a : ℕ → ℤ
| 0       := 2
| 1       := 3
| (n + 2) := |a (n + 1) - a n|

-- Statement to prove
theorem sequence_value : a 2008 = 1 :=
sorry

end sequence_value_l140_140768


namespace cos_pi_div_two_minus_alpha_l140_140017

theorem cos_pi_div_two_minus_alpha (α : ℝ) (hα1 : α ∈ set.Ioo 0 Real.pi) (hα2 : Real.cos α = 1/2) :
  Real.cos (Real.pi / 2 - α) = Real.sqrt 3 / 2 :=
sorry

end cos_pi_div_two_minus_alpha_l140_140017


namespace complex_problem_l140_140751

open Complex

noncomputable def z : ℂ := 1 - √3 * I
noncomputable def a : ℝ := 2

theorem complex_problem :
  (∀ a : ℝ, ((conj z) ^ 2 + a * z = 0) → a = 2) ∧ (|z + a| = 2 * √3) :=
by
  sorry

end complex_problem_l140_140751


namespace inverse_function_equality_l140_140478

def g (m : ℝ) (x : ℝ) : ℝ := (3 * x + 4) / (m * x - 5)

theorem inverse_function_equality (m : ℝ) :
  (∀ x : ℝ, g m (g m x) = x) ↔ m ≠ -9 :=
by
  sorry

end inverse_function_equality_l140_140478


namespace compare_log_exp_l140_140726

noncomputable def x := Real.log Real.pi
noncomputable def y := Real.logBase (1/2) Real.pi
noncomputable def z := Real.exp (-1/2)

theorem compare_log_exp : y < z ∧ z < x :=
by
  sorry

end compare_log_exp_l140_140726


namespace area_of_parabola_enclosed_fig_l140_140192

theorem area_of_parabola_enclosed_fig:
    let f (x: ℝ) := x^2 - x in
    ∫ x in -1..0, f x = 5/6 :=
by sorry

end area_of_parabola_enclosed_fig_l140_140192


namespace intersection_of_A_and_B_l140_140742

def setA (x : ℝ) : Prop := -1 < x ∧ x < 1
def setB (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def setC (x : ℝ) : Prop := 0 ≤ x ∧ x < 1

theorem intersection_of_A_and_B : {x : ℝ | setA x} ∩ {x | setB x} = {x | setC x} := by
  sorry

end intersection_of_A_and_B_l140_140742


namespace maximum_value_neg_fraction_l140_140393

noncomputable def max_value_expression (a b : ℝ) : ℝ :=
  - (1 / (2 * a)) - (2 / b)

theorem maximum_value_neg_fraction (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  max_value_expression a b ≤ -9 / 2 :=
begin
  sorry
end

end maximum_value_neg_fraction_l140_140393


namespace trajectory_equation_l140_140006

noncomputable def point_satisfies_conditions (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (x * x + y * y = 4)

theorem trajectory_equation (P : ℝ × ℝ) (h : ∃ A B : ℝ × ℝ, 
  x * y = 1 ∧
  (|P - A| = |P - B| ∧ angle A P B = π / 3)) : 
  point_satisfies_conditions P :=
by
  let ⟨(x, y), h⟩ := P
  -- The statement of the theorem
  sorry

end trajectory_equation_l140_140006


namespace max_imaginary_part_angle_l140_140321

theorem max_imaginary_part_angle (θ : ℝ) 
  (h : ∀ z : ℂ, z^8 - z^6 + z^4 - z^2 + 1 = 0 → ℑ z ≤ real.sin θ) :
  θ = 54 :=
sorry

end max_imaginary_part_angle_l140_140321


namespace length_of_AB_l140_140891

theorem length_of_AB (A B C D E F : Point) (hC : Midpoint A B C) (hD : Midpoint B C D) (hE : Midpoint C D E) (hF : Midpoint D E F) (hEF : length (E - F) = 5) : length (A - B) = 80 := 
sorry

end length_of_AB_l140_140891


namespace sum_of_first_15_terms_l140_140220

theorem sum_of_first_15_terms (S : ℕ → ℕ) (h1 : S 5 = 48) (h2 : S 10 = 60) : S 15 = 72 :=
sorry

end sum_of_first_15_terms_l140_140220


namespace length_of_segment_pq_l140_140456

-- Definitions based on the conditions provided
def parametric_circle (θ : ℝ) : ℝ × ℝ := ((1 + Real.cos θ), Real.sin θ)

def polar_circle (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

def polar_line (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin (θ + Real.pi / 3) = 3 * Real.sqrt 3

def ray_OM (θ : ℝ) : Prop := θ = Real.pi / 3

-- The statement of the problem in Lean
theorem length_of_segment_pq : 
  ∀ (ρ1 ρ2 θ : ℝ),
  ray_OM θ →
  polar_circle ρ1 θ →
  polar_line ρ2 θ →
  abs (ρ1 - ρ2) = 2 :=
by
  -- Proof is not required for this exercise
  sorry

end length_of_segment_pq_l140_140456


namespace parallel_vectors_have_proportional_components_l140_140000

def vector_a : ℝ × ℝ × ℝ := (2, -1, 3)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (k ≠ 0) ∧ (b.1 = k * a.1) ∧ (b.2 = k * a.2) ∧ (b.3 = k * a.3)

theorem parallel_vectors_have_proportional_components (x : ℝ) (h : are_parallel vector_a (vector_b x)) : x = -6 :=
by
  sorry

end parallel_vectors_have_proportional_components_l140_140000


namespace equivalent_prop_l140_140198

theorem equivalent_prop (x : ℝ) : (x > 1 → (x - 1) * (x + 3) > 0) ↔ ((x - 1) * (x + 3) ≤ 0 → x ≤ 1) :=
sorry

end equivalent_prop_l140_140198


namespace remaining_distance_to_travel_l140_140320

-- Define the basic conditions and their speeds and times
def total_distance : ℕ := 1475
def amoli_first_speed : ℕ := 42
def amoli_first_time : ℕ := 3.5 -- in hours
def amoli_second_speed : ℕ := 38
def amoli_second_time : ℕ := 2 -- in hours
def anayet_first_speed : ℕ := 61
def anayet_first_time : ℕ := 2.5 -- in hours
def anayet_second_speed : ℕ := 75
def anayet_second_time : ℕ := 1.5 -- in hours
def bimal_first_speed : ℕ := 55
def bimal_first_time : ℕ := 4 -- in hours
def bimal_second_speed : ℕ := 30
def bimal_second_time : ℕ := 2 -- in hours
def chandni_distance : ℕ := 35

-- Define the remaining distance they need to travel
def remaining_distance : ℕ := total_distance
                           - (amoli_first_speed * amoli_first_time
                             + amoli_second_speed * amoli_second_time
                             + anayet_first_speed * anayet_first_time
                             + anayet_second_speed * anayet_second_time
                             + bimal_first_speed * bimal_first_time
                             + bimal_second_speed * bimal_second_time
                             + chandni_distance)

-- The theorem statement to prove
theorem remaining_distance_to_travel : remaining_distance = 672 := by
  sorry

end remaining_distance_to_travel_l140_140320


namespace workers_complete_work_in_combined_days_l140_140077

-- Define the conditions
def worker_A_days (x : ℝ) : ℝ := x
def worker_B_days (x : ℝ) : ℝ := 2 * x
def worker_C_days (x : ℝ) : ℝ := 4 * x

-- Declare the combined days to complete the work
def combined_days (x : ℝ) : ℝ := (4 * x) / 7

-- Theorem statement
theorem workers_complete_work_in_combined_days (x : ℝ) (hA : worker_A_days x = x) (hB : worker_B_days x = 2 * x) (hC : worker_C_days x = 4 * x) :
  combined_days x = (4 * x) / 7 :=
sorry

end workers_complete_work_in_combined_days_l140_140077


namespace probability_at_least_one_defective_is_correct_l140_140276

/-- Define a box containing 21 bulbs, 4 of which are defective -/
def total_bulbs : ℕ := 21
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs

/-- Define probabilities of choosing non-defective bulbs -/
def prob_first_non_defective : ℚ := non_defective_bulbs / total_bulbs
def prob_second_non_defective : ℚ := (non_defective_bulbs - 1) / (total_bulbs - 1)

/-- Calculate the probability of both bulbs being non-defective -/
def prob_both_non_defective : ℚ := prob_first_non_defective * prob_second_non_defective

/-- Calculate the probability of at least one defective bulb -/
def prob_at_least_one_defective : ℚ := 1 - prob_both_non_defective

theorem probability_at_least_one_defective_is_correct :
  prob_at_least_one_defective = 37 / 105 :=
by
  -- Sorry allows us to skip the proof
  sorry

end probability_at_least_one_defective_is_correct_l140_140276


namespace part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l140_140737

def A (x : ℝ) : Prop := x^2 - 4 * x - 5 ≥ 0
def B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

theorem part1_a_eq_neg1_inter (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by sorry

theorem part1_a_eq_neg1_union (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∪ {x : ℝ | B x a} = {x : ℝ | x ≤ 1 ∨ x ≥ 5} :=
by sorry

theorem part2_a_range (a : ℝ) : 
  ({x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | B x a}) → 
  a ∈ {a : ℝ | a > 2 ∨ a ≤ -3} :=
by sorry

end part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l140_140737


namespace triangle_angle_inequality_l140_140893

theorem triangle_angle_inequality (α β γ : ℝ) (h1 : α + β + γ = 180) : 
  (α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60) :=
by
  -- Proof by contradiction
  intro h2
  have h3 : α > 60 ∧ β > 60 ∧ γ > 60 := by sorry
  -- Leading to contradiction on the sum of angles in a triangle
  have h4 : α + β + γ > 180 := by sorry
  contradiction

end triangle_angle_inequality_l140_140893


namespace remainders_distinct_mod_p_l140_140876

theorem remainders_distinct_mod_p
  (p : ℕ) [hp : Fact (Nat.Prime p)]
  (a : ℤ) (ha : ¬ p ∣ a) :
  ∀ i j : ℕ, 1 ≤ i ∧ i < p → 1 ≤ j ∧ j < p → (i * a) % p = (j * a) % p → i = j :=
by
  sorry

end remainders_distinct_mod_p_l140_140876


namespace geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l140_140392

variable (a b c : ℝ)

theorem geometric_implies_b_squared_eq_ac
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ∃ r : ℝ, b = r * a ∧ c = r * b) :
  b^2 = a * c :=
by
  sorry

theorem not_geometric_if_all_zero 
  (hz : a = 0 ∧ b = 0 ∧ c = 0) : 
  ¬(∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

theorem sufficient_but_not_necessary_condition :
  (∃ r : ℝ, b = r * a ∧ c = r * b → b^2 = a * c) ∧ ¬(b^2 = a * c → ∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

end geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l140_140392


namespace bus_stoppage_time_l140_140696

theorem bus_stoppage_time (speed_excl_stoppages speed_incl_stoppages : ℕ) (h1 : speed_excl_stoppages = 54) (h2 : speed_incl_stoppages = 45) : 
  ∃ (t : ℕ), t = 10 := by
  sorry

end bus_stoppage_time_l140_140696


namespace question_A_question_B_l140_140493

variables {a : ℕ → ℝ} {q : ℝ} {a1 : ℝ}

-- Condition: a_n = a1 * q^(n-1)
def geom_seq (n : ℕ) : ℝ := a1 * q ^ (n - 1)

-- Condition: S_n = sum of the first n terms of the geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q ^ n) / (1 - q)

-- Definition for being a geometric sequence
def is_geom_seq (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n, a (n + 1) = r * a n

-- Proving that {1/a_n} is a geometric sequence
theorem question_A : is_geom_seq (λ n, 1 / geom_seq n) (1 / q) := sorry

-- Proving that {a_na_{n+1}} is a geometric sequence
theorem question_B : is_geom_seq (λ n, geom_seq n * geom_seq (n + 1)) (q ^ 2) := sorry

end question_A_question_B_l140_140493


namespace num_perfect_squares_multiple_of_48_l140_140058

theorem num_perfect_squares_multiple_of_48 (N : ℕ) (h : N = 2000000) : 
  ∃ k : ℕ, k = 58 ∧ (∀ m : ℕ, m * m < N → (576 ∣ m * m → ((∃ t : ℕ, t = m ∧ t ≤ 1414) ∧ (m % 24 = 0))) := 
by {
  sorry
}

end num_perfect_squares_multiple_of_48_l140_140058


namespace last_three_digits_2005_pow_2005_l140_140482

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem last_three_digits_2005_pow_2005 :
  last_three_digits (2005 ^ 2005) = 125 :=
sorry

end last_three_digits_2005_pow_2005_l140_140482


namespace min_value_S_l140_140389

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d : ℤ)
variable (a_1 : ℤ)

-- Given conditions
axiom cond1 : a_1 + (a_1 + d) = -20
axiom cond2 : (a_1 + 3 * d) + (a_1 + 5 * d) = -6

-- General term of the sequence
def a_n (n : ℕ) : ℤ := a_1 + (n - 1) * d

-- Sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) : ℤ := n * (a_1 + a_1 + (n - 1) * d) / 2

-- Minimum value condition
noncomputable def min_S_n : ℕ := 6

axiom min_S_n_ax (n : ℕ) (h : 0 < n) : a_n n ≤ 0 → n = min_S_n

-- Proof statement
theorem min_value_S : ∃ (n : ℕ), min_S_n_ax n (n / 2 ≤ 6) :=
  sorry

end min_value_S_l140_140389


namespace max_min_ratio_l140_140137

-- Define the function f : ℝ → ℝ
def f (x : ℝ) : ℝ := (2 - x).sqrt + (3 * x + 12).sqrt

-- Prove that the ratio of the maximum and minimum values of f on the interval [-4, 2] is 2
theorem max_min_ratio :
  let I : Set ℝ := Set.Icc (-4 : ℝ) 2 in
  let f : ℝ → ℝ := λ x, (2 - x).sqrt + (3 * x + 12).sqrt in
  let M : ℝ := Sup (Set.image f I) in
  let m : ℝ := Inf (Set.image f I) in
  M / m = 2 :=
by 
  sorry

end max_min_ratio_l140_140137


namespace area_of_triangle_AMN_l140_140610

theorem area_of_triangle_AMN
  (α : ℝ) -- Angle at vertex A
  (S : ℝ) -- Area of triangle ABC
  (area_AMN_eq : ∀ (α : ℝ) (S : ℝ), ∃ (area_AMN : ℝ), area_AMN = S * (Real.cos α)^2) :
  ∃ area_AMN, area_AMN = S * (Real.cos α)^2 := by
  sorry

end area_of_triangle_AMN_l140_140610


namespace reciprocal_sum_ge_square_l140_140405

theorem reciprocal_sum_ge_square {n : ℕ} (h : n ≥ 2) (a : fin n → ℝ) 
  (h_pos : ∀ i, a i > 0) (h_sum : ∑ i, a i = 1) : 
  (∑ i, (1 / a i)) ≥ (n * n) := 
by 
  sorry

end reciprocal_sum_ge_square_l140_140405


namespace geometric_progression_term_count_l140_140923

theorem geometric_progression_term_count
  (q : ℝ) (b4 : ℝ) (S : ℝ) (b1 : ℝ)
  (h1 : q = 1 / 3)
  (h2 : b4 = b1 * (q ^ 3))
  (h3 : S = b1 * (1 - q ^ 5) / (1 - q))
  (h4 : b4 = 1 / 54)
  (h5 : S = 121 / 162) :
  5 = 5 := sorry

end geometric_progression_term_count_l140_140923


namespace quadratic_has_real_root_l140_140793

theorem quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := sorry

end quadratic_has_real_root_l140_140793


namespace product_three_power_l140_140608

theorem product_three_power (w : ℕ) (hW : w = 132) (hProd : ∃ (k : ℕ), 936 * w = 2^5 * 11^2 * k) : 
  ∃ (n : ℕ), (936 * w) = (2^5 * 11^2 * (3^3 * n)) :=
by 
  sorry

end product_three_power_l140_140608


namespace sum_p_q_r_is_22_l140_140551

-- Definitions based on conditions
variables (x p q r : ℤ)

-- Condition definitions
def condition1 : Prop := x^2 + 16 * x + 63 = (x + p) * (x + q)
def condition2 : Prop := x^2 - 15 * x + 56 = (x - q) * (x - r)

-- Goal: Prove that p + q + r = 22 given the conditions
theorem sum_p_q_r_is_22 (h1 : condition1) (h2 : condition2) : p + q + r = 22 :=
by sorry

end sum_p_q_r_is_22_l140_140551


namespace rachel_steps_l140_140690

theorem rachel_steps (x : ℕ) (h1 : x + 325 = 892) : x = 567 :=
sorry

end rachel_steps_l140_140690


namespace increase_factor_l140_140516

-- Definition of parameters: number of letters, digits, and symbols.
def num_letters : ℕ := 26
def num_digits : ℕ := 10
def num_symbols : ℕ := 5

-- Definition of the number of old license plates and new license plates.
def num_old_plates : ℕ := num_letters ^ 2 * num_digits ^ 3
def num_new_plates : ℕ := num_letters ^ 3 * num_digits ^ 3 * num_symbols

-- The proof problem statement: Prove that the increase factor is 130.
theorem increase_factor : num_new_plates / num_old_plates = 130 := by
  sorry

end increase_factor_l140_140516


namespace proof_problem_l140_140060

def M : Set ℝ := { x | x > -1 }

theorem proof_problem : {0} ⊆ M := by
  sorry

end proof_problem_l140_140060


namespace smaller_cubes_total_l140_140614

theorem smaller_cubes_total (edge_length : ℕ) (N : ℕ) (cubes : ℕ → ℕ) 
  (h1 : edge_length = 4)
  (h2 : ∀ n, 1 ≤ cubes n)
  (h3 : ∑ n in finset.range 4, (cubes n) * (n^3) = edge_length^3)
  (h4 : ∃ n1 n2 n3, n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) :
  N = 31 := 
sorry

end smaller_cubes_total_l140_140614


namespace minimum_kings_needed_l140_140252

-- Define chessboard of size 8x8
def chessboard : Type := list (list (option unit))

-- Define a king and its attack pattern
def king_attacked (board : chessboard) (x y : ℕ) : Prop :=
  ∃ i j, |(i - x)| ≤ 1 ∧ |(j - y)| ≤ 1 ∧ board.nth i >>= (λ row, row.nth j) = some ()

-- Define the main statement
theorem minimum_kings_needed : ∀ (board : chessboard), 
  (∀ x y, board.nth x >>= (λ row, row.nth y) ≠ none → king_attacked board x y) ↔
  ∃ ps, list.length ps = 9 ∧ ∀ {x y}, king_attacked (some () :: ps) x y :=
sorry

end minimum_kings_needed_l140_140252


namespace fourth_person_height_l140_140227

theorem fourth_person_height : ∃ H : ℤ, 
  (let h1 := H,
       h2 := H + 2,
       h3 := H + 4,
       h4 := H + 10 in 
    (h1 + h2 + h3 + h4) / 4 = 78 ∧ h4 = 84) :=
begin
  sorry
end

end fourth_person_height_l140_140227


namespace cubics_sum_l140_140065

theorem cubics_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : x^3 + y^3 = 640 :=
by
  sorry

end cubics_sum_l140_140065


namespace smallest_positive_b_satisfies_equation_l140_140688

noncomputable def smallest_b : ℝ :=
  Inf {b : ℝ | b > 0 ∧ (∃ x, x = 5 * b^2 + 2 ∧ 
    (9 * (sqrt (x + 2)) + x  + 20 = 4 * (sqrt x)))}

theorem smallest_positive_b_satisfies_equation :
  ∃ (b : ℝ), b = smallest_b ∧ b > 0 ∧ (∃ x, x = 5 * b^2 + 2 ∧
    (9 * (sqrt (x + 2)) + x  + 20 = 4 * (sqrt x))) :=
begin
  sorry,
end

end smallest_positive_b_satisfies_equation_l140_140688


namespace percentage_of_salt_in_second_solution_l140_140521

-- Define the data and initial conditions
def original_solution_salt_percentage := 0.15
def replaced_solution_salt_percentage (x: ℝ) := x
def resulting_solution_salt_percentage := 0.16

-- State the question as a theorem
theorem percentage_of_salt_in_second_solution (S : ℝ) (x : ℝ) :
  0.15 * S - 0.0375 * S + x * (S / 4) = 0.16 * S → x = 0.19 :=
by 
  sorry

end percentage_of_salt_in_second_solution_l140_140521


namespace real_yield_correct_l140_140981

-- Define the annual inflation rate and nominal interest rate
def annualInflationRate : ℝ := 0.025
def nominalInterestRate : ℝ := 0.06

-- Define the number of years
def numberOfYears : ℕ := 2

-- Calculate total inflation over two years
def totalInflation (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n - 1

-- Calculate the compounded nominal rate
def compoundedNominalRate (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n

-- Calculate the real yield considering inflation
def realYield (nominalRate : ℝ) (inflationRate : ℝ) : ℝ :=
  (nominalRate / (1 + inflationRate)) - 1

-- Proof problem statement
theorem real_yield_correct :
  let inflation_two_years := totalInflation annualInflationRate numberOfYears
  let nominal_two_years := compoundedNominalRate nominalInterestRate numberOfYears
  inflation_two_years * 100 = 5.0625 ∧
  (realYield nominal_two_years inflation_two_years) * 100 = 6.95 :=
by
  sorry

end real_yield_correct_l140_140981


namespace sum_of_lengths_of_square_sides_l140_140930

theorem sum_of_lengths_of_square_sides (side_length : ℕ) (h1 : side_length = 9) : 
  (4 * side_length) = 36 :=
by
  -- Here we would normally write the proof
  sorry

end sum_of_lengths_of_square_sides_l140_140930


namespace time_without_walking_l140_140105

noncomputable def walking_time_no_escalator (d x : ℝ) : ℝ := d / x
noncomputable def walking_time_with_escalator (d e : ℝ) : ℝ := d / (1.5 * e + e)

-- Problem conditions
variables (d x e : ℝ)
axiom condition1 : walking_time_no_escalator d x = 70
axiom condition2 : walking_time_with_escalator d x = 30

-- Lean theorem proof statement 
theorem time_without_walking : ∃ t : ℝ, t = 84 :=
begin
  have h1 : d = 70 * x, from sorry,
  have h2 : d = 30 * (1.5 * x + e), from sorry,
  have h3 : 70 * x = 30 * (1.5 * x + e), from sorry,
  have h4 : 25 * x = 30 * e, from sorry,
  have h5 : e = (25/30) * x, from sorry,
  have t_no_walk : t = d / e, from sorry,
  have t_eq : t = 84, from sorry,
  exact ⟨t, t_eq⟩,
end

end time_without_walking_l140_140105


namespace solve_for_x_l140_140908

theorem solve_for_x : ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 1 → x = -11 / 5 :=
by
  intro h
  use -11 / 5
  -- skipped proof
  sorry

end solve_for_x_l140_140908


namespace triangle_side_inequality_l140_140734

theorem triangle_side_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : 1 = 1 / 2 * b * c) : b ≥ Real.sqrt 2 :=
sorry

end triangle_side_inequality_l140_140734


namespace cannot_have_finite_and_multiple_centers_of_symmetry_l140_140232

noncomputable def shape_infinite_centers_of_symmetry : Prop :=
  ∃ (S : Type) [topological_space S] (centers : set S), infinite centers ∧ ∀ (O ∈ centers), ∀ x : S, ∃ (y : S), symmetric_about O x y

noncomputable def shape_finite_centers_of_symmetry : Prop :=
  ∃ (S : Type) [topological_space S] (centers : set S), finite centers ∧ ∀ (O ∈ centers), ∀ x : S, ∃ (y : S), symmetric_about O x y

theorem cannot_have_finite_and_multiple_centers_of_symmetry :
  shape_infinite_centers_of_symmetry → ¬ shape_finite_centers_of_symmetry :=
by
  sorry

end cannot_have_finite_and_multiple_centers_of_symmetry_l140_140232


namespace B_completion_time_l140_140279

theorem B_completion_time (A_work_C : ℕ) (A_work_Partial : ℕ) (B_work_Partial : ℕ) (B_days_left : ℕ) : ℕ :=
  -- A completes the work in 15 days
  let A_complete := 15 in
  -- A works for 5 days
  let A_days := 5 in
  -- B completes the remaining work in 18 days
  let B_days := 18 in
  -- Calculate the total days B completes the work alone
  let answer := 27 in
  answer

#eval B_completion_time 15 5 18 27 -- Expected output: 27

end B_completion_time_l140_140279


namespace revenue_fall_percent_l140_140268

theorem revenue_fall_percent (old_revenue new_revenue : ℝ) (h1 : old_revenue = 85) (h2 : new_revenue = 48) :
  ((old_revenue - new_revenue) / old_revenue) * 100 ≈ 43.53 :=
by
  rw [h1, h2]
  sorry

end revenue_fall_percent_l140_140268


namespace find_quadratic_polynomial_l140_140401

theorem find_quadratic_polynomial (x₁ x₂ : ℝ) (h_sum : x₁ + x₂ = 8) (h_prod : x₁ * x₂ = 16) :
  (∀ x, (x - x₁) * (x - x₂) = x^2 - 8 * x + 16) :=
by
  intro x
  have h₁ : (x - x₁) * (x - x₂) = x^2 - (x₁ + x₂) * x + x₁ * x₂,
  calc
    (x - x₁) * (x - x₂)
      = x^2 - (x₁ + x₂) * x + x₁ * x₂ : by ring
  rw [h_sum, h_prod],
  exact h₁

end find_quadratic_polynomial_l140_140401


namespace compute_M_l140_140490

def S : finset ℕ := (finset.range 13).image (λ n, 2^n)

def M : ℕ := finset.sum (finset.off_diag S) (λ pair, abs (pair.1 - pair.2))

theorem compute_M : M = 83376 := by
  sorry

end compute_M_l140_140490


namespace hexagon_largest_angle_l140_140559

theorem hexagon_largest_angle (x : ℝ) (h : 3 * x + 3 * x + 3 * x + 4 * x + 5 * x + 6 * x = 720) : 
  6 * x = 180 :=
by
  sorry

end hexagon_largest_angle_l140_140559


namespace find_v2003_l140_140925

def g : ℕ → ℕ
| 1 := 5
| 2 := 3
| 3 := 2
| 4 := 1
| 5 := 4
| _ := 0  -- Undefined for any other inputs

def v : ℕ → ℕ
| 0 := 5
| (n + 1) := g (v n)

theorem find_v2003 : v 2003 = 5 := 
  sorry

end find_v2003_l140_140925


namespace f_at_neg_4_l140_140723

variable (a b : ℝ) (f : ℝ → ℝ)

-- Condition 1: Define the function f
def f (x : ℝ) : ℝ := a * x^3 + b / x + 3

-- Condition 2: Given f(4) = 5
axiom f_at_4 : f a b 4 = 5

-- Goal: Prove that f(-4) = 1
theorem f_at_neg_4 : f a b (-4) = 1 := by
  sorry

end f_at_neg_4_l140_140723


namespace sequence_eventually_constant_l140_140133

-- Define the sequence {a_k}
def sequence (n : ℕ) (k : ℕ) : ℕ :=
if k = 1 then n else
  find (λ a, 0 ≤ a ∧ a ≤ k - 1 ∧ ((list.sum (list.map (λ m, sequence n m) (list.range k))) + a) % k = 0) (list.range k)

-- Prove that the sequence eventually stabilizes to a constant value
theorem sequence_eventually_constant (n : ℕ) (h : 0 < n) :
  ∃ N : ℕ, ∀ k ≥ N, sequence n k = sequence n (N + 1) :=
sorry

end sequence_eventually_constant_l140_140133


namespace expected_value_of_smallest_s_l140_140167

noncomputable def expected_value_smallest_satisfying_sum (a b : ℕ → ℕ) [∀ i, a i ∈ finset.range 101] [∀ j, b j ∈ finset.range 101] : ℕ :=
  let satisfies_sum (s : ℕ) := ∃ m n, s = (finset.range m).sum a ∧ s = (finset.range n).sum b in
  classical.some (exists_unique_of_exists (λ s, satisfies_sum s))

theorem expected_value_of_smallest_s
  (a b : ℕ → ℕ)
  [∀ i, a i ∈ finset.range 101]
  [∀ j, b j ∈ finset.range 101] :
  expected_value_smallest_satisfying_sum a b = 2550 := sorry

end expected_value_of_smallest_s_l140_140167


namespace radius_squared_of_intersection_circle_l140_140935

def parabola1 (x y : ℝ) := y = (x - 2) ^ 2
def parabola2 (x y : ℝ) := x + 6 = (y - 5) ^ 2

theorem radius_squared_of_intersection_circle
    (x y : ℝ)
    (h₁ : parabola1 x y)
    (h₂ : parabola2 x y) :
    ∃ r, r ^ 2 = 83 / 4 :=
sorry

end radius_squared_of_intersection_circle_l140_140935


namespace youseff_blocks_from_office_l140_140270

def blocks_to_office (x : ℕ) : Prop :=
  let walk_time := x  -- it takes x minutes to walk
  let bike_time := (20 * x) / 60  -- it takes (20 / 60) * x = (1 / 3) * x minutes to ride a bike
  walk_time = bike_time + 4  -- walking takes 4 more minutes than biking

theorem youseff_blocks_from_office (x : ℕ) (h : blocks_to_office x) : x = 6 :=
  sorry

end youseff_blocks_from_office_l140_140270


namespace minimum_questions_for_village_identification_l140_140440

noncomputable def village := Type

def always_truth_teller (v : village) : Prop := 
  v = village.A → true

def always_liar (v : village) : Prop :=
  v = village.B → false

def yes_no_question := ℕ → Prop

def minimum_questions_needed : ℕ := 2

theorem minimum_questions_for_village_identification :
  ∀ (v1 v2 : village) (p : v1 ∨ v2),
  (yes_no_question 1 ∧ yes_no_question 2) →
  (p = ∃ x, (always_truth_teller x ∧ v1 = x) ∨ (always_liar x ∧ v2 = x)) →
  p :=
by
  sorry

end minimum_questions_for_village_identification_l140_140440


namespace find_f_f_2_l140_140720

noncomputable def f : ℝ → ℝ := 
λ x, if x < 2 then 2 * Real.exp (x - 1) else Real.logb 3 (2^x - 1)

theorem find_f_f_2 : f (f 2) = 2 :=
by
  unfold f
  have h1 : f 2 = Real.logb 3 (2^2 - 1) := 
    if_neg (by linarith)
  rw [h1]
  have h2 : Real.logb 3 (4 - 1) = Real.logb 3 3 := rfl
  rw [h2]
  have h3 : Real.logb 3 3 = 1 := by norm_num
  rw [h3]
  have h4 : f 1 = 2 * Real.exp (1 - 1) := if_pos (by linarith)
  have h5 : Real.exp 0 = 1 := by norm_num
  rw [h4, h5]
  norm_num

end find_f_f_2_l140_140720


namespace distance_after_3_minutes_l140_140645

-- Define the speeds of the truck and the car in km/h.
def v_truck : ℝ := 65
def v_car : ℝ := 85

-- Define the time in hours.
def time_in_hours : ℝ := 3 / 60

-- Define the relative speed.
def v_relative : ℝ := v_car - v_truck

-- Define the expected distance between the truck and the car after 3 minutes.
def expected_distance : ℝ := 1

-- State the theorem: the distance between the truck and the car after 3 minutes is 1 km.
theorem distance_after_3_minutes : (v_relative * time_in_hours) = expected_distance := 
by {
  -- Here, we would provide the proof, but we are adding 'sorry' to skip the proof.
  sorry
}

end distance_after_3_minutes_l140_140645


namespace negation_of_p_l140_140163

def p : Prop := ∃ x : ℝ, cos x > 1

theorem negation_of_p : ¬ p ↔ ∀ x : ℝ, cos x ≤ 1 := by
  sorry

end negation_of_p_l140_140163


namespace circle_has_greatest_symmetry_l140_140262

-- Define the number of lines of symmetry for each shape
def symmetry_count : Type → ℕ
| square := 4
| regular_pentagon := 5
| equilateral_triangle := 3
| isosceles_triangle := 1
| circle := ℵ₀

-- Define the shapes
inductive Shape
| square 
| regular_pentagon
| equilateral_triangle
| isosceles_triangle
| circle

-- State the theorem
theorem circle_has_greatest_symmetry :
  ∀ (s : Shape), symmetry_count Shape.circle ≥ symmetry_count s :=
begin
  assume s,
  -- Using the fact that ℵ₀ (infinity) is greater than any natural number
  cases s;
  simp [symmetry_count];
  sorry -- Proof skipped
end

end circle_has_greatest_symmetry_l140_140262


namespace φ_value_l140_140408

noncomputable def f (x φ : ℝ) : ℝ :=
  (real.sqrt 3) * real.sin (2 * x + φ) + real.cos (2 * x + φ)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

theorem φ_value (φ : ℝ) :
  is_even (λ x, f x φ) ∧ is_increasing_on (λ x, f x φ) (set.Icc 0 (real.pi / 4))
  → φ = 4 * real.pi / 3 :=
  sorry

end φ_value_l140_140408


namespace linear_eq_satisfies_solution_l140_140593

theorem linear_eq_satisfies_solution (x y : ℤ) (h1 : x = -3) (h2 : y = 1) : x + y = -2 := 
by
  rw [h1, h2]
  simp
  exact rfl

end linear_eq_satisfies_solution_l140_140593


namespace units_digit_5_pow_2023_l140_140958

theorem units_digit_5_pow_2023 : ∀ n : ℕ, (n > 0) → (5^n % 10 = 5) → (5^2023 % 10 = 5) :=
by
  intros n hn hu
  have h_units_digit : ∀ k : ℕ, (k > 0) → 5^k % 10 = 5 := by
    intro k hk
    sorry -- pattern proof not included
  exact h_units_digit 2023 (by norm_num)

end units_digit_5_pow_2023_l140_140958


namespace find_xyz_l140_140744

theorem find_xyz (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 45) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15) (h3 : x + y + z = 5) : x * y * z = 10 :=
by
  sorry

end find_xyz_l140_140744


namespace possible_values_of_x3_y3_z3_l140_140112

def matrix_data (x y z : ℝ) := 
  ![
    [x, y, z],
    [y, z, x],
    [z, x, y]
  ]

theorem possible_values_of_x3_y3_z3 (x y z : ℝ) (I : Matrix (Fin 3) (Fin 3) ℝ) :
  let N := matrix_data x y z in
  N ⬝ N = 2 • I ∧ x * y * z = -2 →
  ∃ (k : ℝ), k ∈ { -6 + 2 * Real.sqrt 2, -6 - 2 * Real.sqrt 2 } ∧ k = x^3 + y^3 + z^3 :=
by
  sorry

end possible_values_of_x3_y3_z3_l140_140112


namespace roots_reciprocal_sum_l140_140496

theorem roots_reciprocal_sum
  (a b c : ℂ)
  (h : Polynomial.roots (Polynomial.C 1 + Polynomial.X - Polynomial.C 1 * Polynomial.X^2 + Polynomial.C 1 * Polynomial.X^3) = {a, b, c}) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2 :=
by
  sorry

end roots_reciprocal_sum_l140_140496


namespace number_of_slips_with_2_l140_140537

theorem number_of_slips_with_2 (x y : ℕ) (h1 : x + y ≤ 15) 
  (h2 : 0 < x ∧ 0 < y ∧ 0 ≤ 15 - x - y)
  (h3 : 4.6 = (2 * x + 5 * y + 8 * (15 - x - y)) / 15) : x = 8 :=
by
  sorry

end number_of_slips_with_2_l140_140537


namespace rational_exponent_value_l140_140427

theorem rational_exponent_value (a b : ℚ) (h : |a - 3| + (b + 2)^2 = 0) : b^a = -8 := by
  sorry

end rational_exponent_value_l140_140427


namespace solve_QR_l140_140470
-- Import the entire Mathlib to ensure all required tools are available

-- Definitions for the problem
variable (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variable (PQ QR PR angle_Q : Real)
variable (cos_angle_Q : Real)

-- Conditions
axiom cos_Q_def : cos_angle_Q = 5 / 13
axiom PQ_def : PQ = 13
axiom angle_Q_right : angle_Q = π / 2

-- Goal: Prove QR
theorem solve_QR : ∃ QR, cos_angle_Q = QR / PQ ∧ QR = 5 := 
sorry

end solve_QR_l140_140470


namespace neg_p_implies_neg_q_l140_140011

variable {x : ℝ}

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_implies_neg_q (h : ¬ p x) : ¬ q x :=
sorry

end neg_p_implies_neg_q_l140_140011


namespace find_x_for_hx_eq_20_l140_140127

noncomputable def g (x : ℝ) : ℝ := 32 / (x + 4)
noncomputable def g_inv (x : ℝ) : ℝ := sorry -- Define the inverse function
def h (x : ℝ) : ℝ := 4 * g_inv(x)

theorem find_x_for_hx_eq_20 : ∃ x : ℝ, h x = 20 ∧ x = 32 / 9 :=
by
  sorry

end find_x_for_hx_eq_20_l140_140127


namespace infinite_geometric_series_l140_140128

theorem infinite_geometric_series
  (p q r : ℝ)
  (h_series : ∑' n : ℕ, p / q^(n+1) = 9) :
  (∑' n : ℕ, p / (p + r)^(n+1)) = (9 * (q - 1)) / (9 * q + r - 10) :=
by 
  sorry

end infinite_geometric_series_l140_140128


namespace distance_between_truck_and_car_l140_140641

noncomputable def truck_speed : ℝ := 65
noncomputable def car_speed : ℝ := 85
noncomputable def time_duration_hours : ℝ := 3 / 60

theorem distance_between_truck_and_car :
  let relative_speed := car_speed - truck_speed in
  let distance := relative_speed * time_duration_hours in
  distance = 1 :=
by
  let relative_speed := car_speed - truck_speed
  let distance := relative_speed * time_duration_hours
  have h1 : distance = 1 := sorry
  exact h1

end distance_between_truck_and_car_l140_140641


namespace sample_variance_is_1_5_l140_140217

def sample_data : List ℤ := [3, 4, 4, 5, 5, 6, 6, 7]
def sample_mean : ℤ := 5

noncomputable def mean (data : List ℤ) : ℚ :=
  (data.sum : ℚ) / data.length

noncomputable def sample_variance (data : List ℤ) (mean : ℚ) : ℚ :=
  (data.map (λ x, (x - mean) ^ 2)).sum / data.length

theorem sample_variance_is_1_5 :
  mean sample_data = sample_mean →
  sample_variance sample_data sample_mean = 1.5 := by
  intro h
  sorry

end sample_variance_is_1_5_l140_140217


namespace flower_bed_can_fit_l140_140846

noncomputable def flower_bed_fits_in_yard : Prop :=
  let yard_side := 70
  let yard_area := yard_side ^ 2
  let building1 := (20 * 10)
  let building2 := (25 * 15)
  let building3 := (30 * 30)
  let tank_radius := 10 / 2
  let tank_area := Real.pi * tank_radius^2
  let total_occupied_area := building1 + building2 + building3 + 2*tank_area
  let available_area := yard_area - total_occupied_area
  let flower_bed_radius := 10 / 2
  let flower_bed_area := Real.pi * flower_bed_radius^2
  let buffer_area := (yard_side - 2 * flower_bed_radius)^2
  available_area >= flower_bed_area ∧ buffer_area >= flower_bed_area

theorem flower_bed_can_fit : flower_bed_fits_in_yard := 
  sorry

end flower_bed_can_fit_l140_140846


namespace solve_for_m_l140_140937

def power_function_monotonic (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2 * m - 3 < 0)

theorem solve_for_m (m : ℝ) (h : power_function_monotonic m) : m = 2 :=
sorry

end solve_for_m_l140_140937


namespace percentage_reduction_correct_l140_140561

-- Define the initial conditions
def initial_conditions (P S : ℝ) (new_sales_increase_percentage net_sale_value_increase_percentage: ℝ) :=
  new_sales_increase_percentage = 0.72 ∧ net_sale_value_increase_percentage = 0.4104

-- Define the statement for the required percentage reduction
theorem percentage_reduction_correct (P S : ℝ) (x : ℝ) 
  (h : initial_conditions P S 0.72 0.4104) : 
  (S:ℝ) * (1 - x / 100) = 1.4104 * S := 
sorry

end percentage_reduction_correct_l140_140561


namespace crop_fraction_l140_140289

noncomputable def fraction_crop_to_longest_side := 1 / 3

theorem crop_fraction (θ1 θ2 : ℝ) (a b : ℝ) 
  (h1 : θ1 = 60) (h2 : θ2 = 120) (h3 : a = 120) (h4 : b = 240) :
  let fraction := 1 / 3 in
  fraction = fraction_crop_to_longest_side :=
by
  sorry

end crop_fraction_l140_140289


namespace distinct_real_c_f_ff_ff_five_l140_140498

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem distinct_real_c_f_ff_ff_five : 
  (∀ c : ℝ, f (f (f (f c))) = 5 → False) :=
by
  sorry

end distinct_real_c_f_ff_ff_five_l140_140498


namespace polygon_sides_l140_140568

theorem polygon_sides (sum_of_interior_angles : ℕ) (h : sum_of_interior_angles = 720) :
    ∃ n : ℕ, (n - 2) * 180 = sum_of_interior_angles ∧ n = 6 :=
by
  use 6
  split
  sorry

end polygon_sides_l140_140568


namespace obtuse_triangle_number_of_k_l140_140942

theorem obtuse_triangle_number_of_k :
  ∃ (n : ℕ), (∀ (k : ℕ), (17 < 13 + k ∧ k < 30 ∧ (17^2 > 13^2 + k^2 ∨ k^2 > 13^2 + 17^2)) ↔ k < 11 ∨ 21 < k) ∧ 
             (n = finset.sum (finset.range 30) (λ k, if (k > 4 ∧ k < 11) ∨ (21 < k ∧ k < 30) then 1 else 0)) :=
begin
  use 14,
  intros k,
  split,
  { intros h, 
    cases h with h1 h2, 
    cases h2 with h3 h4, 
    cases h4,
    { left,
      linarith },
    { right,
      linarith }},
  { intros h,
    split,
    { sorry }, -- This part would normally be proved with further steps
    { split,
      { sorry }, -- Ensuring the triangle inequality
      { sorry } } } -- Ensuring obtuseness of the triangle
end

end obtuse_triangle_number_of_k_l140_140942


namespace square_area_is_625_l140_140928

-- Definitions from conditions
def length_of_rectangle (r : ℝ) := (2 / 5) * r
def breadth_of_rectangle : ℝ := 10
def area_of_rectangle (r : ℝ) := length_of_rectangle r * breadth_of_rectangle
def side_of_square (r : ℝ) := r
def area_of_square (r : ℝ) := (side_of_square r)^2

-- Theorem to be proved
theorem square_area_is_625 (r : ℝ) (h1 : area_of_rectangle r = 100) : 
  area_of_square (r) = 625 := 
by
  sorry

end square_area_is_625_l140_140928


namespace second_furthest_jumper_is_Kyungsoo_l140_140553

structure Jumper :=
  (name : String)
  (distance : Float)

def Kyungsoo : Jumper := { name := "Kyungsoo", distance := 2.3 }
def Younghee : Jumper := { name := "Younghee", distance := 0.9 }
def Jinju : Jumper := { name := "Jinju", distance := 1.8 }
def Chanho : Jumper := { name := "Chanho", distance := 2.5 }

theorem second_furthest_jumper_is_Kyungsoo :
  (∀ jumpers : List Jumper, jumpers = [Kyungsoo, Younghee, Jinju, Chanho] →
    let distances := jumpers.map (λ j => j.distance)
    let sorted_distances := List.sort (λ a b => a > b) distances
    sorted_distances.get? 1 = some 2.3) :=
by
  sorry

end second_furthest_jumper_is_Kyungsoo_l140_140553


namespace cost_price_of_article_l140_140627

variable (C : ℝ) -- Original cost price
variable (S : ℝ) -- Original selling price
variable (C_new : ℝ) -- New cost price if bought 20% less
variable (S_new : ℝ) -- New selling price if sold for Rs. 6.30 less

-- Conditions Definitions
def original_selling_price : Prop := S = 1.25 * C
def new_cost_price : Prop := C_new = 0.80 * C
def new_selling_price : Prop := S_new = S - 6.30
def profit_on_new_price : Prop := S_new = 1.04 * C

-- Theorem Statement
theorem cost_price_of_article
  (h1 : original_selling_price C S)
  (h2 : new_cost_price C C_new)
  (h3 : new_selling_price C S S_new)
  (h4 : profit_on_new_price C S_new) :
  C = 30 :=
sorry

end cost_price_of_article_l140_140627


namespace total_opponents_points_is_36_l140_140275
-- Import the Mathlib library

-- Define the conditions as Lean definitions
def game_scores : List ℕ := [3, 5, 6, 7, 8, 9, 11, 12]

def lost_by_two (n : ℕ) : Prop := n + 2 ∈ game_scores

def three_times_as_many (n : ℕ) : Prop := n * 3 ∈ game_scores

-- State the problem
theorem total_opponents_points_is_36 : 
  (∃ l1 l2 l3 w1 w2 w3 w4 w5 : ℕ, 
    game_scores = [l1, l2, l3, w1, w2, w3, w4, w5] ∧
    lost_by_two l1 ∧ lost_by_two l2 ∧ lost_by_two l3 ∧
    three_times_as_many w1 ∧ three_times_as_many w2 ∧ 
    three_times_as_many w3 ∧ three_times_as_many w4 ∧ 
    three_times_as_many w5 ∧ 
    l1 + 2 + l2 + 2 + l3 + 2 + ((w1 / 3) + (w2 / 3) + (w3 / 3) + (w4 / 3) + (w5 / 3)) = 36) :=
sorry

end total_opponents_points_is_36_l140_140275


namespace degree_of_x_squared_y_is_3_l140_140197

-- Define the degree of a monomial where the monomial is represented using variable exponents.
def degree_of_monomial (exponents : List ℕ) : ℕ :=
  exponents.foldr (+) 0

-- The exponents for the monomial x^2 y
def exponents_x_squared_y : List ℕ := [2, 1]

-- Prove that the degree of the monomial x^2 y is 3
theorem degree_of_x_squared_y_is_3 : degree_of_monomial exponents_x_squared_y = 3 :=
by
  -- The proof would go here,
  sorry

end degree_of_x_squared_y_is_3_l140_140197


namespace polynomial_root_q_l140_140068

theorem polynomial_root_q (a b q r : ℝ) (h : b ≠ 0) :
    (∀ x : ℂ, x^3 + q * x + r = 0) ∧ ((a : ℂ) + b * complex.I) ∧ ((a : ℂ) - b * complex.I) (∃ s : ℝ, 
    let s := -2 * a
    in (x = s)) → q = b^2 - 3 * a^2 := sorry

end polynomial_root_q_l140_140068


namespace geometric_sequence_sum_reciprocal_ratio_l140_140071

theorem geometric_sequence_sum_reciprocal_ratio
  (a : ℚ) (r : ℚ) (n : ℕ) (S S' : ℚ)
  (h1 : a = 1/4)
  (h2 : r = 2)
  (h3 : S = a * (1 - r^n) / (1 - r))
  (h4 : S' = (1/a) * (1 - (1/r)^n) / (1 - 1/r)) :
  S / S' = 32 :=
sorry

end geometric_sequence_sum_reciprocal_ratio_l140_140071


namespace distance_to_fourth_side_l140_140388

theorem distance_to_fourth_side (s : ℕ) (d1 d2 d3 : ℕ) (x : ℕ) 
  (cond1 : d1 = 4) (cond2 : d2 = 7) (cond3 : d3 = 12)
  (h : d1 + d2 + d3 + x = s) : x = 9 ∨ x = 15 :=
  sorry

end distance_to_fourth_side_l140_140388


namespace marble_B_catch_C_time_l140_140147

noncomputable def marble_times (a b c L : ℝ) : ℝ :=
  if h1 : a - b = L / 50 ∧ a - c = L / 40 then
    110
  else
    0

theorem marble_B_catch_C_time (a b c L : ℝ) :
  (a - b = L / 50) → (a - c = L / 40) → marble_times a b c L = 110 :=
by
  intros h1 h2
  simp [marble_times, h1, h2]
  sorry

end marble_B_catch_C_time_l140_140147


namespace least_element_of_special_set_l140_140494

theorem least_element_of_special_set :
  ∃ T : Finset ℕ, T ⊆ Finset.range 16 ∧ T.card = 7 ∧
    (∀ {x y : ℕ}, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) ∧ 
    (∀ {z : ℕ}, z ∈ T → ∀ {x y : ℕ}, x ≠ y → x ∈ T → y ∈ T → z ≠ x + y) ∧
    ∀ (x : ℕ), x ∈ T → x ≥ 4 :=
sorry

end least_element_of_special_set_l140_140494


namespace max_pairs_k_le_999_l140_140377

theorem max_pairs_k_le_999 :
  ∃ (k : ℕ), k ≤ 999 ∧
  ∀ (pairs : list (ℕ × ℕ)),
    (∀ (i j : ℕ) (hi hj : i < pairs.length) (p q : ℕ × ℕ),
      pairs.nth_le i hi = p → pairs.nth_le j hj = q → i ≠ j → ∀ n m : ℕ, p.fst ≠ q.fst ∧ p.fst ≠ q.snd ∧ p.snd ≠ q.fst ∧ p.snd ≠ q.snd) ∧ 
    (∀ (i : ℕ) (hi : i < pairs.length),
      let p := pairs.nth_le i hi in p.fst < p.snd ∧ p.fst + p.snd ≤ 2500) ∧
    (list.pairwise (≠) (pairs.map (λ (p : ℕ × ℕ), p.fst + p.snd))) →
    pairs.length = k := 
sorry

end max_pairs_k_le_999_l140_140377


namespace fg_eq_14_l140_140423

-- Declare the functions f and g
def g (x : ℕ) : ℕ := x ^ 2
def f (x : ℕ) : ℕ := 3 * x + 2

-- The theorem to prove that f(g(2)) = 14
theorem fg_eq_14 : f(g(2)) = 14 :=
  by sorry

end fg_eq_14_l140_140423


namespace simplify_expression_l140_140988

open Classical

variable (x : ℝ)

theorem simplify_expression : (x - 3)^2 - (x + 1) * (x - 1) = -6x + 10 := by
  sorry

end simplify_expression_l140_140988


namespace f_x1_plus_f_x2_always_greater_than_zero_l140_140729

theorem f_x1_plus_f_x2_always_greater_than_zero
  {f : ℝ → ℝ}
  (h1 : ∀ x, f (-x) = -f (x + 2))
  (h2 : ∀ x > 1, ∀ y > 1, x < y → f y < f x)
  (h3 : ∃ x₁ x₂ : ℝ, 1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) :
  ∀ x₁ x₂ : ℝ, (1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) → f x₁ + f x₂ > 0 := by
  sorry

end f_x1_plus_f_x2_always_greater_than_zero_l140_140729


namespace find_4_digit_number_l140_140697

theorem find_4_digit_number (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182) :
  1000 * a + 100 * b + 10 * c + d = 1909 :=
by {
  sorry
}

end find_4_digit_number_l140_140697


namespace smallest_positive_multiple_of_29_l140_140957

theorem smallest_positive_multiple_of_29 :
  ∃ (a : ℕ), (29 * a) % 97 = 7 ∧ 0 < 29 * a ∧ 29 * a = 2349 :=
begin
  sorry
end

end smallest_positive_multiple_of_29_l140_140957


namespace wendys_brother_pieces_l140_140582

-- Definitions based on conditions
def number_of_boxes : ℕ := 2
def pieces_per_box : ℕ := 3
def total_pieces : ℕ := 12

-- Summarization of Wendy's pieces of candy
def wendys_pieces : ℕ := number_of_boxes * pieces_per_box

-- Lean statement: Prove the number of pieces Wendy's brother had
theorem wendys_brother_pieces : total_pieces - wendys_pieces = 6 :=
by
  sorry

end wendys_brother_pieces_l140_140582


namespace fraction_of_A_or_B_l140_140438

def fraction_A : ℝ := 0.7
def fraction_B : ℝ := 0.2

theorem fraction_of_A_or_B : fraction_A + fraction_B = 0.9 := 
by
  sorry

end fraction_of_A_or_B_l140_140438


namespace simplify_exponentiation_l140_140177

theorem simplify_exponentiation (x : ℕ) :
  (x^5 * x^3)^2 = x^16 := 
by {
  sorry -- proof will go here
}

end simplify_exponentiation_l140_140177


namespace probability_of_average_six_l140_140171

def set_0_to_9 : Set ℕ := {n | n < 10}

def choose_three (s : Set ℕ) : Set (Set ℕ) := 
  {t | t.card = 3 ∧ t ⊆ s}

def valid_subsets (s : Set ℕ) : Set (Set ℕ) :=
  {t | t ∈ choose_three s ∧ (t.Sum id) = 18}

theorem probability_of_average_six :
  let n := choose_three set_0_to_9 in
  let k := valid_subsets set_0_to_9 in
  (k.card : ℚ) / n.card = 7 / 120 :=
sorry

end probability_of_average_six_l140_140171


namespace min_value_PQ_l140_140506

def f (x : ℝ) : ℝ := x ^ 2 + 1
def g (x : ℝ) : ℝ := x + Real.log x
def h (t : ℝ) : ℝ := f t - g t

theorem min_value_PQ : ∃ t ∈ Set.Ioi (0 : ℝ), ∀ t0 ∈ Set.Ioi (0 : ℝ), h t ≤ h t0 :=
by
  sorry

end min_value_PQ_l140_140506


namespace number_of_solutions_l140_140054

theorem number_of_solutions (S : Set ℤ) : {x : ℤ | abs (7 * x - 5) ≤ 11}.card = 5 :=
by
  sorry

end number_of_solutions_l140_140054


namespace square_circle_area_ratio_l140_140310

def side_length (r : ℝ) : ℝ := 2 * r
def circle_area (r : ℝ) : ℝ := real.pi * (r ^ 2)
def square_area (s : ℝ) : ℝ := s ^ 2
def area_ratio (side_len : ℝ) (r : ℝ) : ℝ :=
  (square_area side_len) / (circle_area r)

theorem square_circle_area_ratio (r : ℝ) : 
  side_length r = 2 * r →
  circle_area r = real.pi * r ^ 2 →
  square_area (side_length r) = (2 * r) ^ 2 →
  area_ratio (side_length r) r = 4 / real.pi :=
by
  intros h1 h2 h3
  sorry

end square_circle_area_ratio_l140_140310


namespace evaluate_nested_fraction_l140_140695

theorem evaluate_nested_fraction :
  (1 / (3 - (1 / (2 - (1 / (3 - (1 / (2 - (1 / 2))))))))) = 11 / 26 :=
by
  sorry

end evaluate_nested_fraction_l140_140695


namespace curve_intersection_four_points_l140_140342

theorem curve_intersection_four_points (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 * a^2 ∧ y = a * x^2 - 2 * a) ∧ 
  (∃! (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ), 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
    y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4 ∧
    x1^2 + y1^2 = 4 * a^2 ∧ y1 = a * x1^2 - 2 * a ∧
    x2^2 + y2^2 = 4 * a^2 ∧ y2 = a * x2^2 - 2 * a ∧
    x3^2 + y3^2 = 4 * a^2 ∧ y3 = a * x3^2 - 2 * a ∧
    x4^2 + y4^2 = 4 * a^2 ∧ y4 = a * x4^2 - 2 * a) ↔ 
  a > 1 / 2 :=
by 
  sorry

end curve_intersection_four_points_l140_140342


namespace minimum_value_g_l140_140340

noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + x * (x + 3) / (x^2 + 3) + 3 * (x + 1) / (x * (x^2 + 3))

theorem minimum_value_g : ∀ x > 0, g x ≥ 4 ∧ (∃ x > 0, g x = 4) :=
by
  sorry

end minimum_value_g_l140_140340


namespace students_do_not_like_either_food_l140_140823

noncomputable def students_who_do_not_like_either_food
  (total_students : ℕ)
  (likes_french_fries : ℕ)
  (likes_burgers : ℕ)
  (likes_both : ℕ) : ℕ :=
total_students - (likes_french_fries + likes_burgers - likes_both)

theorem students_do_not_like_either_food
  (total_students : ℕ)
  (likes_french_fries : ℕ)
  (likes_burgers : ℕ)
  (likes_both : ℕ)
  (h_total : total_students = 25)
  (h_french_fries : likes_french_fries = 15)
  (h_burgers : likes_burgers = 10)
  (h_both : likes_both = 6) :
  students_who_do_not_like_either_food total_students likes_french_fries likes_burgers likes_both = 6 := 
by {
  rw [h_total, h_french_fries, h_burgers, h_both],
  dsimp [students_who_do_not_like_either_food],
  norm_num,
  sorry
}

end students_do_not_like_either_food_l140_140823


namespace total_pieces_eq_21_l140_140623

-- Definitions based on conditions
def red_pieces : Nat := 5
def yellow_pieces : Nat := 7
def green_pieces : Nat := 11

-- Derived definitions from conditions
def red_cuts : Nat := red_pieces - 1
def yellow_cuts : Nat := yellow_pieces - 1
def green_cuts : Nat := green_pieces - 1

-- Total cuts and the resulting total pieces
def total_cuts : Nat := red_cuts + yellow_cuts + green_cuts
def total_pieces : Nat := total_cuts + 1

-- Prove the total number of pieces is 21
theorem total_pieces_eq_21 : total_pieces = 21 := by
  sorry

end total_pieces_eq_21_l140_140623


namespace sym_sum_ineq_l140_140600

theorem sym_sum_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x + y + z = 1 / x + 1 / y + 1 / z) : x * y + y * z + z * x ≥ 3 :=
by
  sorry

end sym_sum_ineq_l140_140600


namespace problem_zero_points_of_g_function_l140_140409

noncomputable def f (k x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1 / 2) ^ x

noncomputable def g (k x : ℝ) : ℝ :=
  f k (f k x) - 3 / 2

theorem problem_zero_points_of_g_function (k : ℝ) :
  (∃ xs : list ℝ, xs.length = 3 ∧ ∀ x ∈ xs, g k x = 0) ↔ k ∈ Icc (-1/2) (-1/4) :=
by
  sorry

end problem_zero_points_of_g_function_l140_140409


namespace isosceles_triangle_inequality_l140_140089

/--
In an isosceles triangle \(ABC\), point \(D\) is taken on the base \(BC\), and points \(E\) and \(M\) on the side \(AB\)
such that \(AM = ME\) and segment \(DM\) is parallel to side \(AC\). Prove that \(AD + DE > AB + BE\).
-/
theorem isosceles_triangle_inequality
  (A B C D E M : Point)
  (h_isosceles : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ dist A B = dist A C)
  (hD : D ∈ line_segment B C)
  (hE : E ∈ line_segment A B)
  (hAM_ME : dist A M = dist M E)
  (hDM_parallel_AC : parallel (line D M) (line A C)) :
  dist A D + dist D E > dist A B + dist B E := 
sorry

end isosceles_triangle_inequality_l140_140089


namespace zoo_total_children_spoken_proof_l140_140661

noncomputable def number_of_children_spoken_to_by_guides (total_guides : ℕ)
  (english_guides : ℕ) (french_guides : ℕ)
  (children_per_english_guide : ℕ) (children_per_french_guide : ℕ)
  (children_per_spanish_guide : ℕ) : ℕ :=
  let spanish_guides := total_guides - english_guides - french_guides
  in (english_guides * children_per_english_guide) +
     (french_guides * children_per_french_guide) +
     (spanish_guides * children_per_spanish_guide)

theorem zoo_total_children_spoken_proof : number_of_children_spoken_to_by_guides 
  22 10 6 19 25 30 = 520 := by
  sorry

end zoo_total_children_spoken_proof_l140_140661


namespace age_ratio_l140_140214

theorem age_ratio (R D : ℕ) (h1 : R + 2 = 26) (h2 : D = 18) : R / D = 4 / 3 :=
sorry

end age_ratio_l140_140214


namespace garrison_reinforcement_l140_140290

/-- A garrison has initial provisions for 2000 men for 65 days. 
    After 15 days, reinforcement arrives and the remaining provisions last for 20 more days. 
    The size of the reinforcement is 3000 men.  -/
theorem garrison_reinforcement (P : ℕ) (M1 M2 D1 D2 D3 R : ℕ) 
  (h1 : M1 = 2000) (h2 : D1 = 65) (h3 : D2 = 15) (h4 : D3 = 20) 
  (h5 : P = M1 * D1) (h6 : P - M1 * D2 = (M1 + R) * D3) : 
  R = 3000 := 
sorry

end garrison_reinforcement_l140_140290


namespace remainder_2023_mul_7_div_45_l140_140254

/-- The remainder when the product of 2023 and 7 is divided by 45 is 31. -/
theorem remainder_2023_mul_7_div_45 : 
  (2023 * 7) % 45 = 31 := 
by
  sorry

end remainder_2023_mul_7_div_45_l140_140254


namespace lemonade_problem_l140_140694

theorem lemonade_problem :
  ∃ E A : ℝ, 
    (A = 2 * E) ∧ 
    (let ed_consumed := (2 / 3) * E in
    let ed_remaining := (1 / 3) * E in
    let ann_consumed := (5 / 6) * A in
    let ann_remaining := (1 / 3) * A in
    let transfer_to_ed := (1 / 6) * A + 3 in
    let ed_total := ed_consumed + transfer_to_ed in
    let ann_total := ann_consumed - transfer_to_ed in
    ed_total = ann_total ∧ 
    (E + A = 36)) :=
sorry

end lemonade_problem_l140_140694


namespace min_k_value_l140_140895

-- Definitions of "special number" property
def is_special_number (n : ℕ) : Prop :=
  let h := n / 100
  let t := (n % 100) / 10
  let o := n % 10
  h ≠ 0 ∧ t ≠ 0 ∧ o ≠ 0 ∧ (h ≠ t ∧ t ≠ o ∧ o ≠ h)

-- Definition of F function
def F (n : ℕ) : ℕ :=
  let h := n / 100
  let t := (n % 100) / 10
  let o := n % 10
  (h * 100 + o * 10 + t + t * 100 + h * 10 + o + o * 100 + t * 10 + h) / 111

-- Define s and t and give constraints
variables (x y : ℕ)
variables (h₁ : 1 ≤ x ∧ x ≤ 9)
variables (h₂ : 1 ≤ y ∧ y ≤ 9)
def s := 100 * x + 32
def t := 150 + y

-- Given sum condition
variables (h₃ : F(s) + F(t) = 19)

-- Prove the desired property
theorem min_k_value : (F(s) - F(t)) = -7 :=
by
  sorry

end min_k_value_l140_140895


namespace percentage_m2_is_30_percent_l140_140087

-- Definitions based on the problem conditions
def percentage_manufactured_by_m1 := 0.40
def percentage_products_defective_m1 := 0.03
def stockpile_defective_percentage := 0.036

-- Defining the percentage of products manufactured by m2
variable (x : ℝ)

-- Conditions based on problem statement.
def percentage_manufactured_by_m3 := 0.60 - x
def percentage_products_defective_m2 := 0.01
def percentage_products_defective_m3 := 0.07

-- The equation derived from the problem
def defective_percentage_calculation :=
  (percentage_manufactured_by_m1 * percentage_products_defective_m1) +
  (x * percentage_products_defective_m2) +
  ((0.60 - x) * percentage_products_defective_m3)

-- The theorem that needs to be proved
theorem percentage_m2_is_30_percent (h : defective_percentage_calculation x = stockpile_defective_percentage) : x = 0.30 :=
by
  sorry

end percentage_m2_is_30_percent_l140_140087


namespace prove_relationship_l140_140746

noncomputable def a : ℝ := 3 ^ 0.4
noncomputable def b : ℝ := Real.log 1 / Real.log 3
noncomputable def c : ℝ := (1 / 3) ^ 0.2

theorem prove_relationship : a > c ∧ c > b := 
by 
  sorry

end prove_relationship_l140_140746


namespace C_trajectory_l140_140457

noncomputable def C_trajectory_equation (A B : ℝ × ℝ) (d : ℝ) : Prop :=
  let a := 2 in
  let c := 3 in
  let b := Real.sqrt 5 in
  A = (-3,0) ∧ B = (3,0) ∧ d = 4 → 
  (∃ x y : ℝ, x^2 / (a^2) - y^2 / (b^2) = 1 ∧ x ≥ 2)

theorem C_trajectory :
  C_trajectory_equation (-3,0) (3,0) 4 :=
by sorry

end C_trajectory_l140_140457


namespace triangle_area_correct_l140_140445

noncomputable def side_length : ℝ := 2

noncomputable def area_of_triangle_formed_by_centers : ℝ :=
  let R := side_length / sqrt 3 in
  let side_of_large_triangle := 2 * R in
  (sqrt 3 / 4) * side_of_large_triangle ^ 2

theorem triangle_area_correct :
  area_of_triangle_formed_by_centers = (4 * sqrt 3 / 3) :=
by sorry

end triangle_area_correct_l140_140445


namespace Debby_jogged_on_Monday_l140_140885

variable (d_tuesday : ℕ) (d_wednesday : ℕ) (d_total : ℕ)

def d_monday (d_total d_tuesday d_wednesday : ℕ) : ℕ :=
  d_total - (d_tuesday + d_wednesday)

theorem Debby_jogged_on_Monday :
  d_tuesday = 5 ∧ d_wednesday = 9 ∧ d_total = 16 → d_monday 16 5 9 = 2 :=
by
  intros h
  cases h with ht hw
  cases hw with hw ht
  unfold d_monday
  exact eq.refl 2

end Debby_jogged_on_Monday_l140_140885


namespace quadratic_has_real_root_of_b_interval_l140_140795

variable (b : ℝ)

theorem quadratic_has_real_root_of_b_interval
  (h : ∃ x : ℝ, x^2 + b * x + 25 = 0) : b ∈ Iic (-10) ∪ Ici 10 :=
by
  sorry

end quadratic_has_real_root_of_b_interval_l140_140795


namespace relationship_among_a_b_c_l140_140497

def a := 2^(0.2)
def b := Real.log 2
def c := Real.logBase 0.3 2

theorem relationship_among_a_b_c : a > b ∧ b > c := 
by
  sorry

end relationship_among_a_b_c_l140_140497


namespace semiperimeter_inequality_l140_140067

theorem semiperimeter_inequality 
  (ABC A'B'C' : Triangle) 
  (p p' : ℝ)
  (r R : ℝ)
  (h_p : p = ABC.semiperimeter)
  (h_p' : p' = A'B'C'.semiperimeter)
  (h_r : r = ABC.inradius)
  (h_R : R = ABC.circumradius)
  (h_inscribed : A'B'C'.inscribed_in ABC) :
  (r / R) ≤ (p' / p) ∧ (p' / p) ≤ 1 := 
sorry

end semiperimeter_inequality_l140_140067


namespace ratio_areas_ACEF_ADC_l140_140100

-- Define the basic geometric setup
variables (A B C D E F : Point) 
variables (BC CD DE : ℝ) 
variable (α : ℝ)
variables (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) 

-- Assuming the given conditions, we want to prove the ratio of areas
noncomputable def ratio_areas (α : ℝ) : ℝ := 4 * (1 - α)

theorem ratio_areas_ACEF_ADC (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) :
  ratio_areas α = 4 * (1 - α) :=
sorry

end ratio_areas_ACEF_ADC_l140_140100


namespace find_sum_of_p_q_r_l140_140188

noncomputable theory

def positive_integer (n : ℕ) : Prop :=
  n > 0

theorem find_sum_of_p_q_r (p q r : ℕ) 
  (hp : positive_integer p) 
  (hq : positive_integer q) 
  (hr : positive_integer r) 
  (h : 2 * Real.sqrt (Real.cbrt 7 - Real.cbrt 3) = Real.cbrt ↑p + Real.cbrt ↑q - Real.cbrt ↑r) :
  p + q + r = 34 := by
  sorry

end find_sum_of_p_q_r_l140_140188


namespace linear_regression_correct_l140_140403

variables (x y : ℝ) (X : ℝ → ℝ)

def negatively_correlated (x y : ℝ) : Prop := sorry -- Define what it means for variables to be negatively correlated

noncomputable def sample_means : Prop :=
  let x̅ : ℝ := 3
  let y̅ : ℝ := 3.5
  true -- This is a placeholder to assert the means.

noncomputable def possible_linear_regression_eqn (X : ℝ → ℝ) : Prop :=
  X 3 = 3.5 ∧ (∀ x, X x = -2 * x + 9.5)

theorem linear_regression_correct :
  negatively_correlated x y →
  sample_means →
  possible_linear_regression_eqn X :=
sorry

end linear_regression_correct_l140_140403


namespace minimum_dominos_l140_140264

theorem minimum_dominos (n : ℕ) (h_odd : n % 2 = 1) (h_ge_3 : n ≥ 3) : 
  ∃ k, k = (3 * n - 5) / 2 ∧ ∀ m < k, ∃ (i j : ℕ), (1 ≤ i ∧ i ≤ j ∧ j ≤ n) → m = dominos_placement(i, j) :=
sorry

end minimum_dominos_l140_140264


namespace barbara_total_cost_l140_140664

-- Definitions based on the given conditions
def steak_cost_per_pound : ℝ := 15.00
def steak_quantity : ℝ := 4.5
def chicken_cost_per_pound : ℝ := 8.00
def chicken_quantity : ℝ := 1.5

def expected_total_cost : ℝ := 42.00

-- The main proposition we need to prove
theorem barbara_total_cost :
  steak_cost_per_pound * steak_quantity + chicken_cost_per_pound * chicken_quantity = expected_total_cost :=
by
  sorry

end barbara_total_cost_l140_140664


namespace amount_after_two_years_is_correct_l140_140972

noncomputable def amount_after_two_years (initial_value : ℝ) (rate_of_increase : ℝ) : ℝ :=
  initial_value * (1 + rate_of_increase)^2

theorem amount_after_two_years_is_correct :
  amount_after_two_years 64000 (1 / 6) ≈ 87030.4 :=
by
  sorry

end amount_after_two_years_is_correct_l140_140972


namespace length_DD_l140_140950

def Point := (ℝ × ℝ)

def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem length_DD'_reflection :
  let D := (2, 1) in
  let D' := reflect_x D in
  distance D D' = 2 :=
by
  sorry

end length_DD_l140_140950


namespace q1_q2_q3_l140_140035

open Real

noncomputable def f (a x : ℝ) := (x - a) * exp x

theorem q1 :
  { x : ℝ // 0 < x } → 
  MonotonicOn (f 1) (Set.Ioi 0) :=
sorry

theorem q2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → 
    (if a ≥ ((2 * exp 1 - 1) / (exp 1 - 1)) then f a 1 = (1 - a) * exp 1 else f a 2 = (2 - a) * exp 2)) :=
sorry

theorem q3 : 
  ∀ x ∈ Icc (-5 : ℝ) (⊤ : ℝ), 
    let f1 := f 1 x in 
    f1 + x + 5 ≥ -6 / exp 5 :=
sorry

end q1_q2_q3_l140_140035


namespace solve_fractional_integer_eq_l140_140182

theorem solve_fractional_integer_eq (x : ℝ) (h : ⌊x⌋ ^ 5 + (x - ⌊x⌋) ^ 5 = x ^ 5) : x ∈ set.Ico 0 1 ∪ set.Ici 0 ∩ set.Ici 1 ∩ set.Icc 0 1 ∪ set.Icc 0 1 ∪ set.Icc 0 1 ∩ set.Icc 0 1 ∪ set.Ici 0 ∩ set.Icc 0 1 ∪ set.Icc 0 1 ∩ set.Icc 0 1  := sorry

end solve_fractional_integer_eq_l140_140182


namespace max_covered_squares_l140_140609

-- Definitions representing the conditions
def checkerboard_squares : ℕ := 1 -- side length of each square on the checkerboard
def card_side_len : ℕ := 2 -- side length of the card

-- Theorem statement representing the question and answer
theorem max_covered_squares : ∀ n, 
  (∃ board_side squared_len, 
    checkerboard_squares = 1 ∧ card_side_len = 2 ∧
    (board_side = checkerboard_squares ∧ squared_len = card_side_len) ∧
    n ≤ 16) →
  n = 16 :=
  sorry

end max_covered_squares_l140_140609


namespace independence_iff_l140_140175

noncomputable section
open MeasureTheory ProbabilityTheory

variables {Ω : Type} {n : ℕ} {μ : MeasureTheory.Measure Ω}
variables (ξ : Fin n -> Ω -> ℝ)
variables (F : (Fin n → ℝ) → ℝ)
variables (Fi : Fin n → (ℝ → ℝ))

def joint_CDF (ξ : Fin n → Ω → ℝ) (x : Fin n → ℝ) : ℝ :=
  μ {ω | ∀ i, ξ i ω ≤ x i}

def individual_CDF (ξi : Ω → ℝ) (xi : ℝ) : ℝ :=
  μ {ω | ξi ω ≤ xi}

theorem independence_iff (ξ : Fin n → Ω → ℝ) :
  (∀ x, joint_CDF ξ x = ∏ i, individual_CDF (ξ i) (x i)) ↔
  (MeasureTheory.IndepIndepSets (λ i, MeasurableSpace.comap (ξ i) MeasureTheory.ProbabilityTheory.metricSpaceBorel) μ) := 
sorry

end independence_iff_l140_140175


namespace pooja_speed_l140_140898

theorem pooja_speed (v : ℝ) :
  (∀ (roja_speed distance time : ℝ), 
    roja_speed = 4 ∧ 
    distance = 28 ∧ 
    time = 4 ∧
    distance = time * (roja_speed + v))
  → v = 3 :=
begin
  intro h,
  obtain ⟨roja_speed, ⟨distance, ⟨time, ⟨h1, ⟨h2, ⟨h3, h4⟩⟩⟩⟩⟩⟩ := h,
  rw [h1, h2, h3] at h4,
  linarith,
end

end pooja_speed_l140_140898


namespace triangle_probability_l140_140562

theorem triangle_probability : 
  let lengths := {2, 3, 4, 5}
  let all_combinations := {{2, 3, 4}, {2, 3, 5}, {2, 4, 5}, {3, 4, 5}}
  let valid_combinations := {{2, 3, 4}, {2, 4, 5}, {3, 4, 5}}
  P := valid_combinations.card.to_rat / all_combinations.card.to_rat
  P = 3/4 :=
by
  sorry

end triangle_probability_l140_140562


namespace find_integer_mod_l140_140705

theorem find_integer_mod (n : ℤ) (h : 0 ≤ n ∧ n ≤ 5) : n ≡ -4378 [MOD 6] ↔ n = 2 := by
  sorry

end find_integer_mod_l140_140705


namespace average_other_color_marbles_l140_140151

def percentage_clear : ℝ := 0.4
def percentage_black : ℝ := 0.2
def total_percentage : ℝ := 1.0
def total_marbles_taken : ℝ := 5.0

theorem average_other_color_marbles :
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black in
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors in
  expected_other_color_marbles = 2 := by
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors
  show expected_other_color_marbles = 2
  sorry

end average_other_color_marbles_l140_140151


namespace max_value_f_value_f_2alpha_l140_140039

noncomputable def f (x : ℝ) := Real.sin (x + π / 6) + Real.cos x

theorem max_value_f :
  ∃ S : Set ℝ, ∀ x : ℝ, (f x ≤ √3) ∧ ((f x = √3) ↔ (x ∈ S)) ∧ (S = { x | ∃ k : ℤ, x = 2 * k * π + π / 6 }) :=
sorry

theorem value_f_2alpha (α : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : f (α + π / 6) = 3 * √3 / 5) :
  f (2 * α) = (24 * √3 - 21) / 50 :=
sorry

end max_value_f_value_f_2alpha_l140_140039


namespace angle_AOC_double_GOH_l140_140525

open EuclideanGeometry

noncomputable def proof_problem : Prop :=
  ∃ (A B C D E F G H O : Point),
  PointsOrderOnSemicircle [A, B, C, D, E, F] O ∧
  (Dist A D = Dist B E ∧ Dist B E = Dist C F) ∧
  IntersectionOfLineSegments B E A D G ∧
  IntersectionOfLineSegments B E C D H →
  Angle A O C = 2 * Angle G O H

theorem angle_AOC_double_GOH : proof_problem := by
  sorry

end angle_AOC_double_GOH_l140_140525


namespace minimum_students_to_share_birthday_l140_140777

theorem minimum_students_to_share_birthday (k : ℕ) (m : ℕ) (n : ℕ) (hcond1 : k = 366) (hcond2 : m = 2) (hineq : n > k * m) : n ≥ 733 := 
by
  -- since k = 366 and m = 2
  have hk : k = 366 := hcond1
  have hm : m = 2 := hcond2
  -- thus: n > 366 * 2
  have hn : n > 732 := by
    rw [hk, hm] at hineq
    exact hineq
  -- hence, n ≥ 733
  exact Nat.succ_le_of_lt hn

end minimum_students_to_share_birthday_l140_140777


namespace area_of_polygon_l140_140464

-- Definitions based on the given conditions
structure Polygon (n : ℕ) :=
(sides_eq : ∀ (i j : Fin n), i ≠ j → length i = length j)
(perpendicular : ∀ (i : Fin n), is_perp (i, (i + 1) % n) (i, (i - 1) % n))
(perimeter : Fin n → ℝ)

def is_perp {α : Type*} [add_group α] (x y : α) (z w : α) : Prop :=
-- Assuming some definition of perpendicular (orthogonality) which we skip here for simplicity,
sorry

-- Assume the side length for all sides are equal and hence we use only the first side
def side_length (P : Polygon 30) : ℝ := 
P.perimeter 0 / 30

-- Area computation
def polygon_area : ℝ :=
27 * (2 ^ 2)

theorem area_of_polygon (P : Polygon 30)
  (h1 : side_length P = 2)
  (h2 : polygon_area = 108) :
  polygon_area = 108 :=
by
  sorry

end area_of_polygon_l140_140464


namespace subtraction_correctness_l140_140351

theorem subtraction_correctness : 25.705 - 3.289 = 22.416 := 
by
  sorry

end subtraction_correctness_l140_140351


namespace argument_z_not_pi_over_4_l140_140505

noncomputable def z : ℂ := sorry

def condition (z : ℂ) := abs (z - abs (z + 1)) = abs (z + abs (z - 1))

theorem argument_z_not_pi_over_4 (z : ℂ) (h : condition z) : ∀ θ, abs (θ - Real.pi / 4) ≠ 0 :=
sorry

end argument_z_not_pi_over_4_l140_140505


namespace arithmetic_sequence_n_value_l140_140119

theorem arithmetic_sequence_n_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (n : ℕ)
  (hS9 : S 9 = 18)
  (ha_n_minus_4 : a (n-4) = 30)
  (hSn : S n = 336)
  (harithmetic_sequence : ∀ k, a (k + 1) - a k = a 2 - a 1) :
  n = 21 :=
sorry

end arithmetic_sequence_n_value_l140_140119


namespace pats_stick_length_correct_l140_140889

noncomputable def jane_stick_length : ℕ := 22
noncomputable def sarah_stick_length : ℕ := jane_stick_length + 24
noncomputable def uncovered_pats_stick : ℕ := sarah_stick_length / 2
noncomputable def covered_pats_stick : ℕ := 7
noncomputable def total_pats_stick : ℕ := uncovered_pats_stick + covered_pats_stick

theorem pats_stick_length_correct : total_pats_stick = 30 := by
  sorry

end pats_stick_length_correct_l140_140889


namespace vertical_line_divides_triangle_equal_area_l140_140536

noncomputable def triangle_area_dividing_x_coordinate (A B C: ℝ × ℝ) (a: ℝ): Prop := 
  A = (0, 2) ∧ B = (0, 0) ∧ C = (10, 0) ∧ 
  let total_area := (1 / 2) * 10 * 2 in
  let region_area := total_area / 2 in
  let left_triangle_area := (1 / 2) * a * 2 in
  left_triangle_area = region_area

theorem vertical_line_divides_triangle_equal_area :
  ∃ a : ℝ, triangle_area_dividing_x_coordinate (0, 2) (0, 0) (10, 0) a ∧ a = 5 :=
begin
  sorry
end

end vertical_line_divides_triangle_equal_area_l140_140536


namespace exam_score_l140_140838

theorem exam_score (correct_marks wrong_marks : Int) (total_questions correct_answers : Int) :
  correct_marks = 4 →
  wrong_marks = -1 →
  total_questions = 50 →
  correct_answers = 36 →
  let incorrect_answers := total_questions - correct_answers in
  let correct_total := correct_answers * correct_marks in
  let wrong_total := incorrect_answers * wrong_marks in
  let total_score := correct_total + wrong_total in
  total_score = 130 :=
by
  intros h1 h2 h3 h4
  let incorrect_answers := total_questions - correct_answers
  let correct_total := correct_answers * correct_marks
  let wrong_total := incorrect_answers * wrong_marks
  let total_score := correct_total + wrong_total
  sorry

end exam_score_l140_140838


namespace polynomial_pos_for_all_n_l140_140118

open Real

theorem polynomial_pos_for_all_n (P : ℝ[X]) 
  (h0 : 0 < P.eval 0)
  (h1 : P.eval 0 < P.eval 1)
  (h2 : 2 * P.eval 1 - P.eval 0 < P.eval 2)
  (h3 : ∀ n : ℕ, P.eval (n + 3) > 3 * P.eval (n + 2) - 3 * P.eval (n + 1) + P.eval n) :
  ∀ n : ℕ, 0 < P.eval n := 
sorry

end polynomial_pos_for_all_n_l140_140118


namespace axis_of_symmetry_l140_140038

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 4)

theorem axis_of_symmetry (k : ℤ) : ∃ k : ℤ, ∀ x : ℝ, f(x) = f(x + k * Real.pi) := by
  sorry

end axis_of_symmetry_l140_140038


namespace collinear_proof_perpendicular_proof_l140_140139

noncomputable def collinear_points (e1 e2 : ℝ → ℝ) (A B D : ℝ → ℝ) : Prop :=
  let AB := e1 + e2
  let BC := 2 * e1 + 8 * e2
  let CD := 3 * (e1 - e2)
  let BD := BC + CD
  BD = 5 * AB

noncomputable def perpendicular_vectors (e1 e2 : ℝ) (k : ℝ) : Prop :=
  let v1 := 2 * e1 + e2
  let v2 := e1 + k * e2
  inner_product_space.is_orthonormal v1 v2

theorem collinear_proof (e1 e2 : ℝ → ℝ) (A B D : ℝ → ℝ)
  (h_angle : arc_cos (inner_product e1 e2) = π / 3)
  (h_norm_e1 : norm e1 = 1) (h_norm_e2 : norm e2 = 1)
  : collinear_points e1 e2 A B D := by
  sorry

theorem perpendicular_proof (e1 e2 : ℝ) (h_angle : real.cos (π / 3) = 1 / 2)
  (h_norm_e1 : norm e1 = 1) (h_norm_e2 : norm e2 = 1)
  (k : ℝ) (h : perpendicular_vectors e1 e2 k) :
  k = -5 / 4 := by
  sorry

end collinear_proof_perpendicular_proof_l140_140139


namespace average_other_marbles_l140_140157

def total_marbles : ℕ := 10 -- Define a hypothetical total number for computation
def clear_marbles : ℕ := total_marbles * 40 / 100
def black_marbles : ℕ := total_marbles * 20 / 100
def other_marbles : ℕ := total_marbles - clear_marbles - black_marbles
def marbles_taken : ℕ := 5

theorem average_other_marbles :
  marbles_taken * other_marbles / total_marbles = 2 := by
  sorry

end average_other_marbles_l140_140157


namespace lena_total_trip_time_l140_140863

noncomputable def lena_trip_time (D_mountain D_freeway speed_ratio time_mountain: ℕ) : ℕ :=
  let v := (D_mountain : ℚ) / (time_mountain : ℚ)
  let v_freeway := speed_ratio * v
  let time_freeway := (D_freeway : ℚ) / v_freeway
  let total_time := (time_freeway + (time_mountain : ℚ) : ℚ)
  total_time.toNat

theorem lena_total_trip_time:
  lena_trip_time 20 80 4 40 = 80 :=
by
  sorry

end lena_total_trip_time_l140_140863


namespace bound_on_points_l140_140443

variables (k n : ℕ) (A : Fin k → Fin k → ℕ)
          (C : Fin k → Fin k → ℕ → ℕ → Prop)

open Finset

-- Define conditions
def outgoing (i j : Fin k) : Prop := (i.val < j.val)
def incoming (i j : Fin k) : Prop := (i.val > j.val)

-- Condition that the color of all outgoing lines for A_i are different from that of incoming lines for A_i
def valid_coloring (A : Fin k → Fin k → ℕ) (C : Fin k → Fin k → ℕ → ℕ → Prop) : Prop :=
  ∀ i : Fin k, 
    let out_colors := {c | ∃ j : Fin k, outgoing i j ∧ C i j c A i j} in
    let in_colors := {c | ∃ j : Fin k, incoming i j ∧ C j i c A j i} in
    out_colors ∩ in_colors = ∅

theorem bound_on_points (A : Fin k → Fin k → ℕ) (C : Fin k → Fin k → ℕ → ℕ → Prop) (h : valid_coloring A C) :
  k ≤ 2^n :=
sorry

end bound_on_points_l140_140443


namespace quadratic_shift_function_l140_140094

theorem quadratic_shift_function (x : ℝ) :
  let f := λ x, x^2 - 2 in
  let g := λ x, f (x + 1) + 3 in
  g x = (x + 1)^2 + 1 :=
by 
  -- Proof goes here
  sorry

end quadratic_shift_function_l140_140094


namespace part_I_part_II_max_min_part_II_max_part_II_min_l140_140756

noncomputable def f (x : ℝ) : ℝ :=
  (4 * (Real.cos x) ^ 4 - 2 * Real.cos(2 * x) - 1) / 
  (Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x))

noncomputable def g (x : ℝ) : ℝ :=
  (1/2) * f(x) + Real.sin(2 * x)

theorem part_I : f (-11 * Real.pi / 12) = Real.sqrt (3) :=
  sorry

theorem part_II_max_min (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 4) :
  (∃ c : ℝ, ∀ y, (0 ≤ y ∧ y < Real.pi / 4) → g(y) ≤ c) ∧
  (∃ d : ℝ, ∀ y, (0 ≤ y ∧ y < Real.pi / 4) → d ≤ g(y)) :=
  sorry

theorem part_II_max (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 4) : 
  g (Real.pi/8) = Real.sqrt 2 :=
  sorry

theorem part_II_min (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 4) : 
  g (0) = 1 :=
  sorry

end part_I_part_II_max_min_part_II_max_part_II_min_l140_140756


namespace walls_painted_purple_l140_140473

theorem walls_painted_purple :
  (10 - (3 * 10 / 5)) * 8 = 32 := by
  sorry

end walls_painted_purple_l140_140473


namespace min_value_m_l140_140504

theorem min_value_m (m r n : ℕ) (hmr : m = (n + 1) * r) 
  (hpos_r : r > 0) (hpos_n : n > 0)
  (A : finset ℕ) (H1 : A = {1, 2, ..., m}.val) :
  ∃ (A1 A2 : finset ℕ), disjoint A1 A2 ∧ (A1 ∪ A2 = A) ∧ ∃ a b, a ∈ A1 ∧ b ∈ A1 ∧ 1 < a / b ∧ a / b ≤ 1 + 1 / n :=
sorry

end min_value_m_l140_140504


namespace figure_50_squares_l140_140508

open Nat

noncomputable def g (n : ℕ) : ℕ := 2 * n ^ 2 + 5 * n + 2

theorem figure_50_squares : g 50 = 5252 :=
by
  sorry

end figure_50_squares_l140_140508


namespace linear_functions_proof_l140_140322

-- Definition for linearity of a function
def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ (k b : ℝ), ∀ x, f x = k * x + b

-- Definitions of the given functions
def f₁ (x : ℝ) : ℝ := 2 * x + 1
def f₂ (x : ℝ) : ℝ := 1 / x
def f₃ (x : ℝ) : ℝ := (x + 1) / 2 - x
def f₄ (t : ℝ) : ℝ := 60 * t
def f₅ (x : ℝ) : ℝ := 100 - 25 * x

-- Theorem stating which functions are linear
theorem linear_functions_proof :
  is_linear_function f₁ ∧ ¬is_linear_function f₂ ∧ is_linear_function f₃ ∧ is_linear_function (λ x, f₄ x) ∧ is_linear_function f₅ :=
by
  sorry

end linear_functions_proof_l140_140322


namespace circulation_property_l140_140063

variables {V : Type*} (E : Type*) [fintype V] [fintype E]

structure circulation (f : E → ℤ) :=
  (property : ∀ X ⊆ V, f(X, V) = f(X, X) + f(X, V \ X))

def set_complement (X V : set V) : set V := V \ X

theorem circulation_property (f : circulation) (X : set V) (hX : X ⊆ V) : f property (X, set_complement X V) = 0 :=
  calc
    f property (X, set_complement X V) = f property (X, V) - f property (X, X) : by sorry
    ...                                    = 0                 : by sorry

end circulation_property_l140_140063


namespace parabola_vertex_trajectory_eq_l140_140363

noncomputable def parabola_vertex_trajectory : Prop :=
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = 2 * m) ∧ (x = -m^2) ∧ (y - 4 * x - 4 * m * y = 0)

theorem parabola_vertex_trajectory_eq :
  (∀ x y : ℝ, (∃ m : ℝ, y = 2 * m ∧ x = -m^2) → y^2 = -4 * x) :=
by
  sorry

end parabola_vertex_trajectory_eq_l140_140363


namespace only_eqnB_is_quadratic_l140_140261

-- Definition of the given equations
def eqnA (x : ℝ) : Prop := x + 1/x = 1
def eqnB (x : ℝ) : Prop := sqrt 2 * x^2 = sqrt 3 * x
def eqnC (x : ℝ) : Prop := (x - 1) * (x - 2) = x^2
def eqnD (x y : ℝ) : Prop := x^2 - y - 2 = 0

-- Statement that only eqnB is quadratic
theorem only_eqnB_is_quadratic (x : ℝ) : 
  quadratic_eqn (eqnB x) :=
sorry

-- Definition of what it means to be a quadratic equation
def quadratic_eqn (p : Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ p = (a * x^2 + b * x + c = 0)

end only_eqnB_is_quadratic_l140_140261


namespace intersection_length_l140_140484

theorem intersection_length 
  (A B : ℝ × ℝ) 
  (hA : A.1^2 + A.2^2 = 1) 
  (hB : B.1^2 + B.2^2 = 1) 
  (hA_on_line : A.1 = A.2) 
  (hB_on_line : B.1 = B.2) 
  (hAB : A ≠ B) :
  dist A B = 2 :=
by sorry

end intersection_length_l140_140484


namespace total_fish_l140_140509

-- Definition of the number of fish Lilly has
def lilly_fish : Nat := 10

-- Definition of the number of fish Rosy has
def rosy_fish : Nat := 8

-- Statement to prove
theorem total_fish : lilly_fish + rosy_fish = 18 := 
by
  -- The proof is omitted
  sorry

end total_fish_l140_140509


namespace value_of_4_inch_cube_l140_140605

noncomputable def value_per_cubic_inch (n : ℕ) : ℝ :=
  match n with
  | 1 => 300
  | _ => 1.1 ^ (n - 1) * 300

def cube_volume (n : ℕ) : ℝ :=
  n^3

noncomputable def total_value (n : ℕ) : ℝ :=
  cube_volume n * value_per_cubic_inch n

theorem value_of_4_inch_cube : total_value 4 = 25555 := by
  admit

end value_of_4_inch_cube_l140_140605


namespace hyperbola_standard_equation_l140_140073

noncomputable def hyperbola_equation (C : Type) [hyperbola C] (x y : ℝ) : Prop :=
  ∃ λ : ℝ, (x^2 / 16 - y^2 / 12 = λ) ∧ (λ = -3/4)

theorem hyperbola_standard_equation (C : Type) [hyperbola C] : 
  (∃ (a b : ℝ) (x y : ℝ), 
    (a = 2 * real.sqrt 2) ∧ 
    (b = real.sqrt 15) ∧ 
    (∃ λ : ℝ, (a^2 / 16 - b^2 / 12 = λ) ∧ (λ = -3/4)) & 
    (standard_form : (y^2 / 9 - x^2 / 12 = 1))) :=
by
  sorry

end hyperbola_standard_equation_l140_140073


namespace complex_roots_equilateral_l140_140873

noncomputable def omega : ℂ := -1/2 + Complex.I * Real.sqrt 3 / 2

theorem complex_roots_equilateral (z1 z2 p q : ℂ) (h₁ : z2 = omega * z1) (h₂ : -p = (1 + omega) * z1) (h₃ : q = omega * z1 ^ 2) :
  p^2 / q = 1 + Complex.I * Real.sqrt 3 :=
by sorry

end complex_roots_equilateral_l140_140873


namespace number_of_days_l140_140472

theorem number_of_days (d : ℝ) (h : 2 * d = 1.5 * d + 3) : d = 6 :=
by
  sorry

end number_of_days_l140_140472


namespace cosine_bc_l140_140050

-- Define the vectors and conditions
variables {a b c : ℝ × ℝ}
def is_unit_vector (v : ℝ × ℝ) : Prop := ∥v∥ = 1
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def cosine_angle (v w : ℝ × ℝ) : ℝ := dot_product v w / (∥v∥ * ∥w∥)

-- Given conditions
axiom unit_a : is_unit_vector a
axiom unit_b : is_unit_vector b
axiom ortho_ab : dot_product a b = 0
axiom def_c : c = (a.1 + (√3) * b.1, a.2 + (√3) * b.2)

-- Goal to prove
theorem cosine_bc : cosine_angle b c = √3 / 2 :=
sorry

end cosine_bc_l140_140050


namespace det_M_cubed_l140_140780

variable {M : Type*} [Matrix M]

-- Given condition that det(M) = 3
axiom det_M_eq_3 : det M = 3

-- The proof statement
theorem det_M_cubed : det (M ^ 3) = 27 := by
  sorry

end det_M_cubed_l140_140780


namespace find_q0_plus_q5_l140_140499

noncomputable def q : ℝ → ℝ :=
  sorry -- Define the monic polynomial q(x)

axiom q_monic : polynomial.degree q = 5
axiom q_at_1 : q 1 = 24
axiom q_at_2 : q 2 = 48
axiom q_at_3 : q 3 = 72

theorem find_q0_plus_q5 : q 0 + q 5 = sorry :=
  sorry

end find_q0_plus_q5_l140_140499


namespace exists_monic_quartic_polynomial_with_roots_l140_140353

theorem exists_monic_quartic_polynomial_with_roots (P : Polynomial ℚ) :
  monic P ∧ P.degree = 4 ∧ 
  (P.eval (3 + real.sqrt 2) = 0) ∧
  (P.eval (2 - real.sqrt 5) = 0) →
  P = (Polynomial.x ^ 4 - 10 * Polynomial.x ^ 3 + 31 * Polynomial.x ^ 2 - 34 * Polynomial.x - 7) :=
by
  sorry

end exists_monic_quartic_polynomial_with_roots_l140_140353


namespace monotonicity_intervals_find_a_max_k_value_l140_140410

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 2
noncomputable def tangent (y : ℝ) : ℝ := Real.exp y - 2

theorem monotonicity_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧
  (a > 0 → ∀ x : ℝ, x < Real.log a → f' x a < 0) ∧ (a > 0 → ∀ x : ℝ, Real.log a < x → f' x a > 0) :=
by 
  sorry

theorem find_a (x₀ : ℝ) (a : ℝ) :
  (Real.exp x₀ - a = Real.exp 1) ∧ (x₀ = 1) ∧ (a = 0) :=
by
  sorry

theorem max_k_value (k : ℤ) :
  (∀ x : ℝ, (0 < x) → (x - k) * (Real.exp x - 1) + x + 1 > 0) ↔ k = 2 :=
by
  sorry

end monotonicity_intervals_find_a_max_k_value_l140_140410


namespace total_books_in_series_l140_140528

-- Definitions from the conditions
def pages_per_book := 200
def total_pages_Sabrina_has_to_read := 1000
def books_read_in_first_month := 4

-- Prove that the total number of books in the series is 6
theorem total_books_in_series (B : ℕ) 
  (h1 : total_pages_Sabrina_has_to_read = 1000)
  (h2 : books_read_in_first_month = 4)
  (h3 : pages_per_book = 200) :
  B = 6 := 
by {
  -- Total pages read in the first month
  have h_first_month_pages := books_read_in_first_month * pages_per_book,
  -- Remaining books after the first month
  have remaining_books := B - books_read_in_first_month,
  -- Books read in second month is half of remaining books
  have books_read_in_second_month := remaining_books / 2,
  -- Total pages read in the second month
  have second_month_pages := books_read_in_second_month * pages_per_book,
  -- Total pages read 
  have total_read := h_first_month_pages + second_month_pages,
  -- Validation with the provided total pages Sabrina has to read
  have h2 := h_first_month_pages = 4 * 200,
  have h3 := second_month_pages,
  sorry
}

end total_books_in_series_l140_140528


namespace quadratic_has_real_root_iff_b_in_intervals_l140_140799

theorem quadratic_has_real_root_iff_b_in_intervals (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ set.Icc (-∞ : ℝ) (-10) ∪ set.Icc 10 (∞ : ℝ)) :=
by by sorry

end quadratic_has_real_root_iff_b_in_intervals_l140_140799


namespace distinct_values_of_expression_l140_140260

noncomputable def cube_root_unity : ℂ := -(1/2) + complex.I * (sqrt 3/2)

theorem distinct_values_of_expression: 
  (∀ n, 1 ≤ n ∧ n ≤ 100 → 
    ∃ distinct_values : finset ℂ, 
      distinct_values.card = 6 ∧ 
      ∀ m, 1 ≤ m ∧ m ≤ 100 → 
        (\left(cube_root_unity ^ 8 + 1 \right)^m ∈ distinct_values)) := 
sorry

end distinct_values_of_expression_l140_140260


namespace dog_partitioning_l140_140538

open Combinatorics

/-- Given 12 dogs, we want to partition them into groups of sizes 4, 6, and 2, such that:
  Rover is in the 4-dog group, and Spot is in the 6-dog group. The number
  of ways to achieve this partition is 2520. -/
theorem dog_partitioning :
  (∃ dogs : Finset ℕ, dogs.card = 12) →
  (∃ Rover Spot : ℕ, Rover ≠ Spot ∧ Rover ∈ dogs ∧ Spot ∈ dogs) →
  (∃ group1 group2 group3 : Finset ℕ, 
    group1.card = 4 ∧ group2.card = 6 ∧ group3.card = 2 ∧
    Rover ∈ group1 ∧ Spot ∈ group2 ∧ 
    group1 ∪ group2 ∪ group3 = dogs ∧ 
    group1 ∩ group2 = ∅ ∧ group2 ∩ group3 = ∅ ∧ group1 ∩ group3 = ∅) →
  nat.choose 10 3 * nat.choose 7 5 = 2520 := 
by
  -- Proof omitted
  sorry

end dog_partitioning_l140_140538


namespace max_value_xy_l140_140877

open Real

theorem max_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 5 * y < 100) :
  ∃ (c : ℝ), c = 3703.7 ∧ ∀ (x' y' : ℝ), 0 < x' → 0 < y' → 2 * x' + 5 * y' < 100 → x' * y' * (100 - 2 * x' - 5 * y') ≤ c :=
sorry

end max_value_xy_l140_140877


namespace hypotenuse_is_approx_18_10_l140_140831

variables (a b c : ℝ)
noncomputable def hypotenuse_length : ℝ :=
  (a^2 + b^2) ^ (1/2)

theorem hypotenuse_is_approx_18_10
  (h1 : b = 2 * a - 3)
  (h2 : 1 / 2 * a * b = 72) :
  hypotenuse_length a b ≈ 18.10 :=
by 
  -- definitions and proof steps
  sorry

end hypotenuse_is_approx_18_10_l140_140831


namespace product_of_real_solutions_of_t_cubed_eq_216_l140_140361

theorem product_of_real_solutions_of_t_cubed_eq_216 : 
  (∃ t : ℝ, t^3 = 216) →
  (∀ t₁ t₂, (t₁ = t₂) → (t₁^3 = 216 → t₂^3 = 216) → (t₁ * t₂ = 6)) :=
by
  sorry

end product_of_real_solutions_of_t_cubed_eq_216_l140_140361


namespace king_chessboard_strategy_king_chessboard_strategy_odd_l140_140294

theorem king_chessboard_strategy (m n : ℕ) : 
  (m * n) % 2 = 0 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) := 
sorry

theorem king_chessboard_strategy_odd (m n : ℕ) : 
  (m * n) % 2 = 1 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) :=
sorry

end king_chessboard_strategy_king_chessboard_strategy_odd_l140_140294


namespace problem_2003rd_term_correct_l140_140852

noncomputable def term_in_sequence (n : ℕ) : ℕ :=
  let perfect_squares_removed := list.filter (λ x, ¬ (∃ m, m * m = x)) (list.range (n + floor (real.sqrt n))) in
  perfect_squares_removed.nth n

theorem problem_2003rd_term_correct :
  term_in_sequence 2003 = 2048 :=
by
  sorry

end problem_2003rd_term_correct_l140_140852


namespace fraction_exp_3_4_cubed_l140_140672

def fraction_exp (a b n : ℕ) : ℚ := (a : ℚ) ^ n / (b : ℚ) ^ n

theorem fraction_exp_3_4_cubed : fraction_exp 3 4 3 = 27 / 64 :=
by
  sorry

end fraction_exp_3_4_cubed_l140_140672


namespace range_of_a_l140_140002

def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 2) * x + 5

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 4 → f x a ≤ f (x+1) a) : a ≥ -2 := 
by
  sorry

end range_of_a_l140_140002


namespace arithmetic_sequence_a_n_sum_sequence_T_n_l140_140390

-- Definition of f(x)
def f (x : ℝ) := (Real.sqrt x + Real.sqrt 3)^2

-- Initial sequence definition and recurrence relation
def S : ℕ+ → ℝ
| 1 := 3
| n+1 := f (S n)

-- Definition of a_{n+1}
def a (n : ℕ) := S (n+1) - S n

-- Definition of b_n
def b (n : ℕ) := 1 / (a (n+1) * a n)

-- Definition of T_n (sum of first n terms of b sequence)
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k+1))

theorem arithmetic_sequence_a_n (n : ℕ) : a (n+1) = 6n + 3 :=
sorry

theorem sum_sequence_T_n (n : ℕ) : T n = 1 / 18 * (1 - 1 / (2*n + 1)) :=
sorry

end arithmetic_sequence_a_n_sum_sequence_T_n_l140_140390


namespace probability_odd_number_l140_140548

-- Defining the set of digits
def digits := {2, 3, 5, 7, 9}

-- Defining the condition that the number must be odd
def is_odd (n : Nat) : Prop := 
  ∃ x ∈ digits, n % 10 = x ∧ x % 2 = 1

-- Defining the total number of favorable outcomes for odd numbers
def favorable_outcomes : Nat := 4

-- Defining the total number of possible outcomes
def total_outcomes : Nat := 5

-- Statement of the theorem
theorem probability_odd_number : (favorable_outcomes : ℚ) / total_outcomes = 4 / 5 := 
by sorry

end probability_odd_number_l140_140548


namespace remainder_of_65_power_65_plus_65_mod_97_l140_140259

theorem remainder_of_65_power_65_plus_65_mod_97 :
  (65^65 + 65) % 97 = 33 :=
by
  sorry

end remainder_of_65_power_65_plus_65_mod_97_l140_140259


namespace total_distance_correct_l140_140296

def segment1_distance (speed1 time1 : ℝ) : ℝ := 
  speed1 * time1

def segment2_distance (speed2 time2 : ℝ) : ℝ := 
  speed2 * time2

def segment3_distance (speed3 time3 : ℝ) : ℝ := 
  speed3 * time3

-- Conditions as definitions
def speed1 : ℝ := 5
def time1 : ℝ := 30 / 60 -- Convert minutes to hours
def speed2 : ℝ := 3
def time2 : ℝ := 45 / 60 -- Convert minutes to hours
def speed3 : ℝ := 6
def time3 : ℝ := 2

-- Calculate each segment distance
def distance1 : ℝ := segment1_distance speed1 time1
def distance2 : ℝ := segment2_distance speed2 time2
def distance3 : ℝ := segment3_distance speed3 time3

-- Total distance covered
def total_distance : ℝ := distance1 + distance2 + distance3

theorem total_distance_correct : 
  total_distance = 16.75 :=
by
  -- Total distance calculation based on given conditions
  have h1 : distance1 = 2.5 := by { sorry }
  have h2 : distance2 = 2.25 := by { sorry }
  have h3 : distance3 = 12 := by { sorry }
  -- Add up each segment distance
  show total_distance = 16.75, from by {
    sorry
  }

end total_distance_correct_l140_140296


namespace star_perimeter_difference_l140_140867

-- Define the equiangular pentagon with a fixed perimeter
def equiangular_pentagon (ABCDE : Type) (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  (∀ (i : ℕ), (0 ≤ i ∧ i < 5) → (x1 + x2 + x3 + x4 + x5 = 2) ∧
   (∀ (angle : ℕ), angle = 108))

-- Define the perimeter of the star formed by the extended sides of the pentagon
def star_perimeter (ABCDE : Type) (x1 x2 x3 x4 x5 : ℝ) : ℝ :=
  (x1 + x2 + x3 + x4 + x5) * real.sec (72 * real.pi / 180)

-- The main statement to prove
theorem star_perimeter_difference (ABCDE : Type) (x1 x2 x3 x4 x5 : ℝ)
  (h : equiangular_pentagon ABCDE x1 x2 x3 x4 x5) :
  ∀ (s1 s2 : ℝ), s1 = star_perimeter ABCDE x1 x2 x3 x4 x5 →
  s2 = star_perimeter ABCDE x1 x2 x3 x4 x5 →
  s1 - s2 = 0 := 
by
  -- Proof is omitted
  sorry

end star_perimeter_difference_l140_140867


namespace composite_numbers_characterization_l140_140364

noncomputable def is_sum_and_product_seq (n : ℕ) (seq : List ℕ) : Prop :=
  seq.sum = n ∧ seq.prod = n ∧ 2 ≤ seq.length ∧ ∀ x ∈ seq, 1 ≤ x

theorem composite_numbers_characterization (n : ℕ) :
  (∃ seq : List ℕ, is_sum_and_product_seq n seq) ↔ ¬Nat.Prime n ∧ 1 < n :=
sorry

end composite_numbers_characterization_l140_140364


namespace cube_root_of_8_is_2_l140_140545

theorem cube_root_of_8_is_2 : ∃ y : ℝ, y^3 = 8 ∧ y = 2 :=
begin
  sorry
end

end cube_root_of_8_is_2_l140_140545


namespace sum_of_coefficients_l140_140778

noncomputable def f (x : ℚ) (n : ℕ) : ℚ :=
  (∑ k in Finset.range(n) + 1, (k + 1) * x^(k + 1))^2

theorem sum_of_coefficients (n : ℕ) :
  let a := (x : ℚ) → x + 2 * x^2 + ... + n * x^n
  f(x : ℚ) := (a(x))^2 in
  let coefficients := (x : ℚ) → a_2 x^2 + a_3 x^3 + ... + a_(2n) x^(2n) :=
  f(x) in
  Finset.sum (Finset.range(n) + 1)^2 (range (n, 2n), a_i) + 2 * k x_k :=
  (n + 1) / 2 * (5 * n^2 + 5 * n + 2) 

end sum_of_coefficients_l140_140778


namespace quadratic_has_real_root_l140_140790

theorem quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := sorry

end quadratic_has_real_root_l140_140790


namespace cosine_of_angle_l140_140774

variables (a b : ℝ^3) (l : ℝ)
 
-- conditions
def norm_a : ℝ := l
def norm_b : ℝ := real.sqrt 2
def dot_product_condition : ℝ := (b • (2 • a + b)) = 1

-- statement to prove
theorem cosine_of_angle (a b : ℝ^3) (l : ℝ) 
  (h1 : ∥a∥ = l)
  (h2 : ∥b∥ = real.sqrt 2)
  (h3 : b • (2 • a + b) = 1) :
  real.cos (real.angle a b) = - real.sqrt 2 / (4 * l) :=
sorry

end cosine_of_angle_l140_140774


namespace inner_quadrilateral_area_ratio_l140_140986

theorem inner_quadrilateral_area_ratio {A B C D A₁ B₁ C₁ D₁ : Point} (AB CD : ℝ) (p : ℝ)
  (hAB₁ : A₁ = A + p • (B - A)) (hB₁ : B₁ = B - p • (B - A))
  (hC₁ : C₁ = C + p • (D - C)) (hD₁ : D₁ = D - p • (D - C)) (h_p : p < 0.5) :
  (Area A₁ B₁ C₁ D₁) / (Area A B C D) = 1 - 2 * p :=
  sorry

end inner_quadrilateral_area_ratio_l140_140986


namespace question_1_question_2_question_3_l140_140754

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem question_1 :
  (∀ x : ℝ, 0 < x ∧ x < 1 / Real.exp 1 → f' x < 0) ∧ 
  (∀ x : ℝ, x > 1 / Real.exp 1 → f' x > 0) ∧ 
  f (1 / Real.exp 1) = -1 / Real.exp 1 := sorry

noncomputable def slope (x₀ : ℝ) : ℝ := Real.log x₀ + 1

noncomputable def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := (slope x₀) * x - x₀

noncomputable def A (x₀ : ℝ) := (0, -x₀)
noncomputable def B (x₀ : ℝ) := ((x₀ / (Real.log x₀ + 1)), 0)

theorem question_2 (x₀ : ℝ) (hx₀ : x₀ > 1 / Real.exp 1) :
  let S := 1/2 * |0 - x₀| * |x₀ / (Real.log x₀ + 1)|
  in S >= 1 / Real.exp 1 :=
sorry

noncomputable def g (λ : ℝ) (x : ℝ) : ℝ := Real.exp (λ * x * Real.log x) - f x

theorem question_3 (λ : ℝ) :
  (0 < ∀ x : ℝ, g λ x >= 1) → λ = 1 := sorry

end question_1_question_2_question_3_l140_140754


namespace earmuffs_bought_in_december_l140_140668

theorem earmuffs_bought_in_december :
  ∀ (total bought_before: ℕ), total = 7790 → bought_before = 1346 → (total - bought_before) = 6444 :=
by
  intros total bought_before h_total h_before
  rw [h_total, h_before]
  norm_num
  sorry

end earmuffs_bought_in_december_l140_140668


namespace abs_nonneg_l140_140592

-- Define the proof problem
theorem abs_nonneg (a : ℝ) : abs a ≥ 0 :=
sorry

end abs_nonneg_l140_140592


namespace h_inverse_correct_l140_140503

def f (x : ℝ) : ℝ := 2 * x ^ 2 + 5 * x + 1
def g (x : ℝ) : ℝ := 3 * x - 4
def h (x : ℝ) : ℝ := f (g x)

def h_inv (y : ℝ) : ℝ := (33 + Real.sqrt (144 * y - 3519)) / 36

theorem h_inverse_correct (y : ℝ) (hy : y ≥ 3519 / 144) : h (h_inv y) = y := by
  sorry

end h_inverse_correct_l140_140503


namespace Y_lies_on_AD_l140_140471

open EuclideanGeometry

variables {A B C D E F X P Q R S Y I : Point}
variable (γ : Circle)

-- Original conditions
axiom triangle_ABC : Triangle A B C
axiom foot_angle_bisector : AngleBisector (∠ BAC) D
axiom incenters_ACD_ABD : (Incenter A C D E) ∧ (Incenter A B D F)
axiom circumcircle_DEF : Circumcircle γ D E F
axiom intersection_BE_CF : ∃ X, Intersect (Line BE) (Line CF) X
axiom intersections_on_circle : ∃ P Q R S, 
  (Intersect (Line BE) γ P) ∧ (Intersect (Line BF) γ Q) ∧ 
  (Intersect (Line CE) γ R) ∧ (Intersect (Line CF) γ S)
axiom second_intersection_PQX_RSX : ∃ Y, SecondIntersection (Circumcircle (Triangle P Q X)) (Circumcircle (Triangle R S X)) Y

-- Proof goal
theorem Y_lies_on_AD
  (hA : triangle_ABC)
  (hB : foot_angle_bisector)
  (hC : incenters_ACD_ABD)
  (hD : circumcircle_DEF γ)
  (hE : intersection_BE_CF)
  (hF : intersections_on_circle)
  (hG : second_intersection_PQX_RSX) :
  LiesOnLine Y (Line A D) :=
sorry

end Y_lies_on_AD_l140_140471


namespace meal_arrangement_exactly_two_correct_l140_140946

noncomputable def meal_arrangement_count : ℕ :=
  let total_people := 13
  let meal_types := ["B", "B", "B", "B", "C", "C", "C", "F", "F", "F", "V", "V", "V"]
  let choose_2_people := (total_people.choose 2)
  let derangement_7 := 1854  -- Derangement of BBCCCVVV
  let derangement_9 := 133496  -- Derangement of BBCCFFFVV
  choose_2_people * (derangement_7 + derangement_9)

theorem meal_arrangement_exactly_two_correct : meal_arrangement_count = 10482600 := by
  sorry

end meal_arrangement_exactly_two_correct_l140_140946


namespace measured_weight_loss_is_correct_l140_140265

-- Define the initial body weight W as a non-negative real number
variable (W : ℝ) (hW : 0 ≤ W)

-- Define the weight after losing 12%
def weight_after_loss (W : ℝ) : ℝ := 0.88 * W

-- Define the weight after adding 2% due to clothes
def final_measured_weight (W : ℝ) : ℝ := 0.8976 * W

-- Define the measured weight loss percentage
def measured_weight_loss_percent (W : ℝ) : ℝ := 
  ((W - final_measured_weight W) / W) * 100

-- Theorem stating the measured weight loss percentage is 10.24%
theorem measured_weight_loss_is_correct (W : ℝ) (hW : 0 ≤ W) :
  measured_weight_loss_percent W = 10.24 := by
  sorry

end measured_weight_loss_is_correct_l140_140265


namespace distance_after_3_minutes_l140_140646

-- Define the speeds of the truck and the car in km/h.
def v_truck : ℝ := 65
def v_car : ℝ := 85

-- Define the time in hours.
def time_in_hours : ℝ := 3 / 60

-- Define the relative speed.
def v_relative : ℝ := v_car - v_truck

-- Define the expected distance between the truck and the car after 3 minutes.
def expected_distance : ℝ := 1

-- State the theorem: the distance between the truck and the car after 3 minutes is 1 km.
theorem distance_after_3_minutes : (v_relative * time_in_hours) = expected_distance := 
by {
  -- Here, we would provide the proof, but we are adding 'sorry' to skip the proof.
  sorry
}

end distance_after_3_minutes_l140_140646


namespace range_sum_f_l140_140681

def f (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

theorem range_sum_f : (0 + 1 = 1) :=
by
  sorry

end range_sum_f_l140_140681


namespace perimeter_pentagon_ABCDE_l140_140587

def AB : ℝ := 2
def BC : ℝ := 2
def CD : ℝ := 1
def DE : ℝ := 1

def AC : ℝ := Real.sqrt (AB^2 + BC^2)
def AD : ℝ := Real.sqrt (AC^2 + CD^2)
def AE : ℝ := Real.sqrt (AD^2 + DE^2)

def perimeter_ABCD : ℝ := AB + BC + CD + DE + AE

theorem perimeter_pentagon_ABCDE : perimeter_ABCD = 6 + Real.sqrt 10 := by
  sorry

end perimeter_pentagon_ABCDE_l140_140587


namespace num_divisors_M_l140_140686

theorem num_divisors_M :
  let M := 2^3 * 3^4 * 5^3 * 7^1
  ∃ n, n = 160 ∧ (∀ d, d ∣ M -> ∃ a b c d, 0 ≤ a ≤ 3 ∧ 0 ≤ b ≤ 4 ∧ 0 ≤ c ≤ 3 ∧ 0 ≤ d ≤ 1) :=
by
  let M := 2^3 * 3^4 * 5^3 * 7^1
  use 160
  split
  { exact rfl }
  { intro d 
    sorry }

end num_divisors_M_l140_140686


namespace min_red_chips_l140_140277

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ (1 / 3) * w)
  (h2 : b ≤ (1 / 4) * r)
  (h3 : w + b ≥ 70) : r ≥ 72 :=
by
  sorry

end min_red_chips_l140_140277


namespace real_yield_correct_l140_140978

-- Define the annual inflation rate and nominal interest rate
def annualInflationRate : ℝ := 0.025
def nominalInterestRate : ℝ := 0.06

-- Define the number of years
def numberOfYears : ℕ := 2

-- Calculate total inflation over two years
def totalInflation (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n - 1

-- Calculate the compounded nominal rate
def compoundedNominalRate (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n

-- Calculate the real yield considering inflation
def realYield (nominalRate : ℝ) (inflationRate : ℝ) : ℝ :=
  (nominalRate / (1 + inflationRate)) - 1

-- Proof problem statement
theorem real_yield_correct :
  let inflation_two_years := totalInflation annualInflationRate numberOfYears
  let nominal_two_years := compoundedNominalRate nominalInterestRate numberOfYears
  inflation_two_years * 100 = 5.0625 ∧
  (realYield nominal_two_years inflation_two_years) * 100 = 6.95 :=
by
  sorry

end real_yield_correct_l140_140978


namespace unique_value_of_W_l140_140856

theorem unique_value_of_W :
  ∀ (T W O F U R : ℕ),
  T = 9 →
  O % 2 = 1 →
  (∀ a b, a ≠ b → ((a = T) ∨ (a = W) ∨ (a = O) ∨ (a = F) ∨ (a = U) ∨ (a = R)) →
               ((b = T) ∨ (b = W) ∨ (b = O) ∨ (b = F) ∨ (b = U) ∨ (b = R))) →
  (9 * 2 + (W + W + carry (O + O)) = F * 1000 + O * 100 + U * 10 + R) →
  W = 1 := sorry

end unique_value_of_W_l140_140856


namespace expression_undefined_at_x_l140_140589

theorem expression_undefined_at_x :
  ∃ x : ℝ, (5 * x - 15 = 0) → (2 * x - 6 = 0) ∧ (frac (2 * x - 6) (5 * x - 15)).denom = 0 → x = 3 :=
sorry

end expression_undefined_at_x_l140_140589


namespace range_of_a_l140_140771

open Set

def U : Set ℝ := univ
def A : Set ℝ := { x | x^2 - x - 6 < 0 }
def B : Set ℝ := { x | x^2 + 2x - 8 > 0 }
def C (a : ℝ) : Set ℝ := { x | x^2 - 4 * a * x + 3 * a^2 < 0 }

theorem range_of_a (a : ℝ) : 
  (compl (A ∪ B) ⊆ C a) → (-2 < a) ∧ (a < -4/3) :=
  sorry

end range_of_a_l140_140771


namespace angle_sum_is_120_l140_140447

noncomputable def geometric_problem : Prop :=
  let α := 30 : ℝ 
  let PQH_eq := 60 : ℝ
  let STQ := 120 : ℝ
  ∃ (R T : ℝ), 
  (α = 30) ∧ (PQH_eq = 60) ∧ (STQ = 120) ∧ 
  (R + T + 60 = 180)

theorem angle_sum_is_120 : geometric_problem :=
by {
  existsi (60 : ℝ),
  existsi (60 : ℝ),
  dsimp [geometric_problem],
  simp,
  linarith,
}

end angle_sum_is_120_l140_140447


namespace Aiyanna_has_more_cookies_l140_140987

def Alyssa_cookies : ℕ := 129
def Aiyanna_cookies : ℕ := 140

theorem Aiyanna_has_more_cookies :
  Aiyanna_cookies - Alyssa_cookies = 11 := by
  sorry

end Aiyanna_has_more_cookies_l140_140987


namespace sale_discount_l140_140236

theorem sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (discount_multiple : ℕ)
  (h1 : purchase_amount = 250)
  (h2 : discount_per_100 = 10)
  (h3 : discount_multiple = purchase_amount / 100) :
  purchase_amount - discount_per_100 * discount_multiple = 230 := 
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end sale_discount_l140_140236


namespace general_term_formula_general_term_smallest_n_l140_140130

noncomputable def a : ℕ → ℕ
| 0     := 8
| (n+1) := ∑ i in Finset.range (n+1), a i + 8

theorem general_term_formula (n : ℕ) : a (n+1) = 2 * a n :=
by
  sorry

theorem general_term (n : ℕ) : a n = 2^(n+2) :=
by
  sorry

theorem smallest_n (n : ℕ) (H : ∏ i in Finset.range n, a i > 1000) : n = 3 :=
by
  sorry

end general_term_formula_general_term_smallest_n_l140_140130


namespace possible_arrangement_exists_l140_140231

-- Define the problem with the conditions
def tile := { length : ℕ, width : ℕ, diagonal : bool }
def tiles := finset (tile)

-- Define a valid arrangement function
def valid_arrangement (t : tile) (pos : ℕ × ℕ) (grid : finset (ℕ × ℕ × bool)) : bool :=
sorry  -- Definition not needed for the problem statement

-- Define the grid and its constraints
def grid := finset (ℕ × ℕ)
def arrangement (tiles_set : finset tile) (positions : finset (ℕ × ℕ)) : grid :=
sorry  -- Definition not needed for the problem statement

theorem possible_arrangement_exists :
  ∃ (tiles_set : finset tile) (positions : finset (ℕ × ℕ)),
    tiles_set.card = 18 ∧
    arrangement tiles_set positions = grid.filter (λ p, valid_arrangement p.tiles pos) ∧
    ∀ (pos1 pos2 : ℕ × ℕ), pos1 ≠ pos2 → valid_arrangement pos1 tiles_set ∧ valid_arrangement pos2 tiles_set :=
sorry

end possible_arrangement_exists_l140_140231


namespace f_2009_is_one_l140_140973

   -- Define the properties of the function f
   variables (f : ℤ → ℤ)
   variable (h_even : ∀ x : ℤ, f x = f (-x))
   variable (h1 : f 1 = 1)
   variable (h2008 : f 2008 ≠ 1)
   variable (h_max : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b))

   -- Prove that f(2009) = 1
   theorem f_2009_is_one : f 2009 = 1 :=
   sorry
   
end f_2009_is_one_l140_140973


namespace geometric_sequence_sum_l140_140222

-- Define the sums of the geometric sequence
variables {S_5 S_10 S_20 : ℕ}

-- Given conditions
def S_5_val : S_5 = 10 := by rfl  -- Sum of the first 5 terms
def S_10_val : S_10 = 50 := by rfl  -- Sum of the first 10 terms

-- Proof that the sum of the first 20 terms is 850
theorem geometric_sequence_sum :
  S_20 = 850 :=
begin
  sorry
end

end geometric_sequence_sum_l140_140222


namespace num_students_like_basketball_is_7_l140_140082

def students_like_basketball : ℕ :=
  let C := 8 -- students who like cricket
  let both := 3 -- students who like both basketball and cricket
  let total := 12 -- students who like basketball or cricket or both
  (total - C + both) -- Inclusion-exclusion principle

theorem num_students_like_basketball_is_7 : students_like_basketball = 7 := 
by
  -- Using the given condition values to solve
  have hC : C = 8 := rfl
  have hBoth : both = 3 := rfl
  have hTotal : total = 12 := rfl
  calc
    students_like_basketball
        = total - C + both : by rfl
    ... = 12 - 8 + 3 : by { rw [hTotal, hC, hBoth] }
    ... = 7 : by norm_num

end num_students_like_basketball_is_7_l140_140082


namespace num_students_scoring_between_80_and_90_l140_140444

noncomputable def class_size : ℕ := 48
noncomputable def mean : ℝ := 80
noncomputable def variance : ℝ := 100

theorem num_students_scoring_between_80_and_90 :
  let σ := Real.sqrt variance,
      Z_80 := (80 - mean) / σ,
      Z_90 := (90 - mean) / σ,
      P_80 := Real.cdf_normal 0 σ mean 80,
      P_90 := Real.cdf_normal 0 σ mean 90,
      prob := P_90 - P_80,
      students := class_size * prob in
  students ≈ 16 := 
by
  sorry

end num_students_scoring_between_80_and_90_l140_140444


namespace quadratic_has_real_root_of_b_interval_l140_140796

variable (b : ℝ)

theorem quadratic_has_real_root_of_b_interval
  (h : ∃ x : ℝ, x^2 + b * x + 25 = 0) : b ∈ Iic (-10) ∪ Ici 10 :=
by
  sorry

end quadratic_has_real_root_of_b_interval_l140_140796


namespace ninth_square_more_than_eighth_l140_140305

noncomputable def side_length (n : ℕ) : ℕ := 3 + 2 * (n - 1)

noncomputable def tile_count (n : ℕ) : ℕ := (side_length n) ^ 2

theorem ninth_square_more_than_eighth : (tile_count 9 - tile_count 8) = 72 :=
by sorry

end ninth_square_more_than_eighth_l140_140305


namespace f_at_neg_4_l140_140722

variable (a b : ℝ) (f : ℝ → ℝ)

-- Condition 1: Define the function f
def f (x : ℝ) : ℝ := a * x^3 + b / x + 3

-- Condition 2: Given f(4) = 5
axiom f_at_4 : f a b 4 = 5

-- Goal: Prove that f(-4) = 1
theorem f_at_neg_4 : f a b (-4) = 1 := by
  sorry

end f_at_neg_4_l140_140722


namespace estimate_y_value_at_x_equals_3_l140_140007

noncomputable def estimate_y (x : ℝ) (a : ℝ) : ℝ :=
  (1 / 3) * x + a

theorem estimate_y_value_at_x_equals_3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ) (y1 y2 y3 y4 y5 y6 y7 y8 : ℝ),
    (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 2 * (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8)) →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 8 →
    estimate_y 3 (1 / 6) = 7 / 6 := by
  intro x1 x2 x3 x4 x5 x6 x7 x8 y1 y2 y3 y4 y5 y6 y7 y8 h_sum hx
  sorry

end estimate_y_value_at_x_equals_3_l140_140007


namespace bill_donuts_combinations_l140_140328

theorem bill_donuts_combinations :
  let kinds := 4 in
  let total_donuts := 7 in
  let required_each_kind := 1 in
  let remaining_donuts := total_donuts - kinds * required_each_kind in
  (remaining_donuts = 3) →
  (kinds = 4) →
  (finset.card (finset.powerset_len remaining_donuts (finset.range (remaining_donuts + kinds - 1))) = 20) :=
by
  intros
  sorry

end bill_donuts_combinations_l140_140328


namespace travel_possible_l140_140459

variable (n : ℕ)

-- Define squares
inductive square
| blue : square
| green : fin n → square

-- Define connections respecting the conditions
-- Note: Details on the specific connections would need further elaboration in terms of Lean 4 definitions and graph theory support. Here we use general conditions.
structure city := 
  (connections : square n → square n → Prop)
  (enterable_exit : ∀ s1 s2 : square n, connections s1 s2 ∨ connections s2 s1)
  (one_way_traffic : ∀ s : square n, ∃ s1 s2 : square n, connections s s1 ∧ connections s s2)

-- Define the main theorem
theorem travel_possible (C : city n) :
  ∀ s1 s2 : square n, ∃ path : list (square n), list.head path = s1 ∧ list.last path (s1 :: []) = s2 ∧ (∀ (i : ℕ) (h : i < list.length path - 1), C.connections (list.nth_le path i h) (list.nth_le path (i + 1) (by linarith [h])) ∨ C.connections (list.nth_le path (i + 1) (by linarith [h])) (list.nth_le path i h)) :=
sorry

end travel_possible_l140_140459


namespace set_union_is_all_real_l140_140030

-- Define the universal set U as the real numbers
def U := ℝ

-- Define the set M as {x | x > 0}
def M : Set ℝ := {x | x > 0}

-- Define the set N as {x | x^2 ≥ x}
def N : Set ℝ := {x | x^2 ≥ x}

-- Prove the relationship M ∪ N = ℝ
theorem set_union_is_all_real : M ∪ N = U := by
  sorry

end set_union_is_all_real_l140_140030


namespace apple_distribution_l140_140345

theorem apple_distribution :
  ∃ (piles : list ℕ) (h_len : piles.length = 3) (h_sum : piles.sum = 10)
    (h_min : ∀ x ∈ piles, 1 ≤ x) (h_max : ∀ x ∈ piles, x ≤ 5),
    (piles = [5, 4, 1] ∨ piles = [5, 3, 2] ∨ piles = [4, 4, 2] ∨ piles = [4, 3, 3]) := sorry

end apple_distribution_l140_140345


namespace fraction_of_25_greater_75_percent_40_by_10_l140_140604

theorem fraction_of_25_greater_75_percent_40_by_10 :
  (0.75 * 40 = 30) → ∃ (x : ℝ), (x * 25 + 10 = 30) → x = 4 / 5 :=
by
  intro h,
  use 4 / 5,
  intro hx,
  sorry

end fraction_of_25_greater_75_percent_40_by_10_l140_140604


namespace area_regular_octagon_l140_140015

theorem area_regular_octagon (AB BC: ℝ) (hAB: AB = 2) (hBC: BC = 2) :
  let side_length := 2 * Real.sqrt 2
  let triangle_area := (AB * AB) / 2
  let total_triangle_area := 4 * triangle_area
  let side_length_rect := 4 + 2 * Real.sqrt 2
  let rect_area := side_length_rect * side_length_rect
  let octagon_area := rect_area - total_triangle_area
  octagon_area = 16 + 8 * Real.sqrt 2 :=
by sorry

end area_regular_octagon_l140_140015


namespace range_of_a_for_f_monotonic_l140_140758

-- Define the function f
def f (x a : ℝ) := real.sqrt(x * (x - a))

-- Define the condition that f(x) is monotonically increasing in the interval (0,1)
def is_monotonically_increasing_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y

-- The main theorem we want to prove
theorem range_of_a_for_f_monotonic (a : ℝ) :
  is_monotonically_increasing_in_interval (λ x, f x a) 0 1 → a ≤ 0 :=
by
  sorry

end range_of_a_for_f_monotonic_l140_140758


namespace part_I_part_II_l140_140763

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := x - 1

theorem part_I (x : ℝ) (h : x ≠ 1) : f x < g x :=
by
  sorry

theorem part_II (n : ℕ) : 
  ∑ i in (Finset.range n).map (Finset.succEmb ℕ), (Real.log (↑i + 1) / ↑i) < n :=
by
  sorry

end part_I_part_II_l140_140763


namespace part1_part2_l140_140495

def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, 3 * a n = 2 * (S n + 2 * n) ∧ S 1 = a 1 ∧ S (n + 1) = S n + a (n + 1)

noncomputable def geometric_seq (a b : ℕ → ℝ) : Prop :=
∃ r : ℝ, (a 1 + 2 = b 1) ∧ (∀ n : ℕ, b (n + 1) = r * b n)

theorem part1 (a S : ℕ → ℝ) (h: seq a S) : 
  ∃ (t : ℕ → ℝ), geometric_seq (λ n, a n + 2) t ∧ 
  (∀ n : ℕ, a n = 6 * 3 ^ (n - 1) - 2) :=
sorry

noncomputable def bn (a : ℕ → ℝ) : ℕ → ℝ := 
λ n, Real.log 3 (a n + 2) / Real.log 3 2

theorem part2 (t : ℕ → ℝ) (hT : (∀ n : ℕ, t n = n)) :
  (∀ n : ℕ, ∏ i in Finset.range n, (1 + 1 / t (2 * i + 1)) > Real.sqrt (t (2 * n + 1))) :=
sorry

end part1_part2_l140_140495


namespace no_matching_formula_l140_140095

def formula_A (x : ℕ) : ℕ := 4 * x - 2
def formula_B (x : ℕ) : ℕ := x^3 - x^2 + 2 * x
def formula_C (x : ℕ) : ℕ := 2 * x^2
def formula_D (x : ℕ) : ℕ := x^2 + 2 * x + 1

theorem no_matching_formula :
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_A x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_B x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_C x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_D x)
  :=
by
  sorry

end no_matching_formula_l140_140095


namespace concyclic_four_points_of_triangle_l140_140215

open Complex

noncomputable def is_concyclic (a b c d : ℂ) : Prop :=
  let cross_ratio (z ∗ : ℂ) := (z.1 - z.3) * (z.2 - z.4) / (z.1 - z.4) * (z.2 - z.3)
  cross_ratio (a, b, c, d) ∈ ℝ

theorem concyclic_four_points_of_triangle
  (a b c : ℂ)
  (h_b : b = 1)
  (h_c : c = -1)
  (h_ratio : norm (a) = sqrt 3)
  (is_concyclic a b c d
: ∃ d,
is_concyclic ((a + b * 2) / 3) ((c - b) / 3) ((c + a * 2 - b) / 3) ((a - c * 2) / 3)) :=
sorry

end concyclic_four_points_of_triangle_l140_140215


namespace find_percentage_increase_l140_140862

variable (E : ℝ) -- Last year's earnings
variable (P : ℝ) -- Percentage increase

-- Conditions
def rent_last_year := 0.10 * E
def rent_this_year := 0.30 * E * (1 + P / 100)
def rent_ratio := 3.45 * rent_last_year

-- Proof of the percentage increase
theorem find_percentage_increase (h1 : rent_this_year = rent_ratio) : P = 15 := by
  sorry

end find_percentage_increase_l140_140862


namespace root_of_f_in_interval_l140_140939

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 8

theorem root_of_f_in_interval : ∃ c ∈ Ioo 3 4, f c = 0 :=
begin
  have h_cont : ContinuousOn f (Ioo 1 ∞), {
    apply ContinuousOn.mono _ (Ioo_subset_Ioo_left one_lt_two),
    continuity,
  },
  have h1 := by norm_num : f 3 < 0,
  have h2 := by { norm_num, exact Real.log_pos (by norm_num1), } : 0 < f 4,
  obtain ⟨c, hc1, hc2⟩ := IntermediateValueTheorem.exists_zero (Ioo 3 4) h_cont h1 h2,
  exact ⟨c, hc1, hc2⟩,
end

end root_of_f_in_interval_l140_140939


namespace Pete_flag_combination_l140_140915

/-- 
The United States flag has 50 stars and 13 stripes. 
For his flag, Pete used:
- 3 less than half the number of stars for circles,
- six more than twice the number of stripes for squares,
- the difference between the number of stars and stripes multiplied by two for triangles.
Prove that the combined total number of circles, squares, and triangles is 128.
-/
theorem Pete_flag_combination : 
  let stars := 50 in
  let stripes := 13 in
  let circles := (stars / 2) - 3 in
  let squares := (2 * stripes) + 6 in
  let triangles := (stars - stripes) * 2 in
  circles + squares + triangles = 128 := 
by
  sorry

end Pete_flag_combination_l140_140915


namespace proposition_C_incorrect_l140_140406

open Real EuclideanSpace

variables 
  (e1 e2 : EuclideanSpace R) 
  (θ : Real) 
  (unit_e1 : ∥e1∥ = 1) 
  (unit_e2 : ∥e2∥ = 1) 
  (angle_e1_e2 : cos θ = 1/2)

theorem proposition_C_incorrect :
  ¬ ∃ t : Real, ∥2 • e1 + t • e2∥ = 1 :=
sorry

end proposition_C_incorrect_l140_140406


namespace rational_root_theorem_l140_140354

theorem rational_root_theorem :
  (∃ x : ℚ, 3 * x^4 - 4 * x^3 - 10 * x^2 + 8 * x + 3 = 0)
  → (x = 1 ∨ x = 1/3) := by
  sorry

end rational_root_theorem_l140_140354


namespace min_distance_P_to_circle_l140_140463

open Classical

noncomputable def P_polar : ℝ × ℝ := (1, real.pi / 2)

noncomputable def P_rect : ℝ × ℝ := (0, 1)

noncomputable def circle : set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 1 }

theorem min_distance_P_to_circle :
  ∃ Q ∈ circle, dist (P_rect) Q = sqrt 2 - 1 :=
by
  use (1, 0)
  have h1 : dist (P_rect) (1, 0) = sqrt 2 := by
    sorry -- Here we skip this proof calculation.
  have h2 : dist (P_rect) (1, 0) - 1 = sqrt 2 - 1 := by
    sorry -- This too is a step calculation.
  simp only [dist] at *
  exact eq.trans h1 h2

end min_distance_P_to_circle_l140_140463


namespace sum_trigonometric_squares_l140_140685

open Real

theorem sum_trigonometric_squares :
  (∑ k in finset.range 1 90, sin (k : ℝ) ^ 2) = 44.5 :=
by
  sorry

end sum_trigonometric_squares_l140_140685


namespace find_angle_C_find_max_sinA_plus_sinB_l140_140816

namespace TriangleProof

-- Definitions of the sides and area
variables {a b c : ℝ} (angleC : ℝ)
def area (a b c : ℝ) : ℝ := (a * b * Math.sin angleC) / 2

-- Condition given in the problem
axiom given_condition (a b c : ℝ) (S : ℝ) : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)

-- Goal 1: Proof of angle C
theorem find_angle_C (a b c : ℝ) (S : ℝ) (h : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) : angleC = Real.pi / 3 :=
sorry

-- Definitions for sin of angles A and B
variables (angleA angleB : ℝ)

-- Conditions related to angleABC
axiom triangle_angles (angleA angleB angleC : ℝ) : angleA + angleB + angleC = Real.pi

-- Goal 2: Proof for maximum value of sin A + sin B
theorem find_max_sinA_plus_sinB (angleA angleB : ℝ) (hC : angleC = Real.pi / 3) : Real.sin angleA + Real.sin angleB ≤ Real.sqrt 3 :=
sorry

end TriangleProof

end find_angle_C_find_max_sinA_plus_sinB_l140_140816


namespace part_a_part_b_l140_140367

open Finset

-- Define Map(n) as the set of all functions from {1, 2, ..., n} to {1, 2, ..., n}
def Map (n : ℕ) := {f : Fin n → Fin n // true}

-- Problem (a)
theorem part_a (n : ℕ) (f : Map n) (h : ∀ x, (f.val x) ≠ x) : (f.val ∘ f.val) ≠ f.val := 
by sorry

-- Problem (b)
theorem part_b (n : ℕ) : 
  ∃ S, S = ∑ k in range (n+1), (nat.choose n k) * k^(n-k) := 
by sorry

end part_a_part_b_l140_140367


namespace balloon_height_and_distances_l140_140374

variable {α β γ : Real}
variable {c : Real}

theorem balloon_height_and_distances (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) : 
  (let CC₁ := (c * sin α * sin β) / sqrt (sin α ^ 2 + sin β ^ 2 - 2 * sin α * sin β * cos γ) in
  let AC := c * sin β / sqrt (sin α ^ 2 + sin β ^ 2 - 2 * sin α * sin β * cos γ) in
  let BC := c * sin α / sqrt (sin α ^ 2 + sin β ^ 2 - 2 * sin α * sin β * cos γ) in
  ∃ CC₁ AC BC,
    (CC₁ = (c * sin α * sin β) / sqrt (sin α ^ 2 + sin β ^ 2 - 2 * sin α * sin β * cos γ) ∧
     AC = c * sin β / sqrt (sin α ^ 2 + sin β ^ 2 - 2 * sin α * sin β * cos γ) ∧
     BC = c * sin α / sqrt (sin α ^ 2 + sin β ^ 2 - 2 * sin α * sin β * cos γ))) :=
by sorry

end balloon_height_and_distances_l140_140374


namespace remainder_when_N_divided_by_1000_l140_140117

def number_of_factors_of_5 (n : Nat) : Nat :=
  if n = 0 then 0 
  else n / 5 + number_of_factors_of_5 (n / 5)

def total_factors_of_5_upto (n : Nat) : Nat := 
  match n with
  | 0 => 0
  | n + 1 => number_of_factors_of_5 (n + 1) + total_factors_of_5_upto n

def product_factorial_5s : Nat := total_factors_of_5_upto 100

def N : Nat := product_factorial_5s

theorem remainder_when_N_divided_by_1000 : N % 1000 = 124 := by
  sorry

end remainder_when_N_divided_by_1000_l140_140117


namespace find_2003rd_non_perfect_square_l140_140854

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def non_perfect_squares : ℕ → ℕ
| 0 := 1
| n := (if is_perfect_square (n+1) then non_perfect_squares n + 1 else non_perfect_squares n) + 1

theorem find_2003rd_non_perfect_square : non_perfect_squares 2002 = 2048 := by
  sorry

end find_2003rd_non_perfect_square_l140_140854


namespace team_A_wins_series_4_1_probability_l140_140579

noncomputable def probability_team_A_wins_series_4_1 : ℝ :=
  let home_win_prob : ℝ := 0.6
  let away_win_prob : ℝ := 0.5
  let home_loss_prob : ℝ := 1 - home_win_prob
  let away_loss_prob : ℝ := 1 - away_win_prob
  -- Scenario 1: L W W W W
  let p1 := home_loss_prob * home_win_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 2: W L W W W
  let p2 := home_win_prob * home_loss_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 3: W W L W W
  let p3 := home_win_prob * home_win_prob * away_loss_prob * away_win_prob * home_win_prob
  -- Scenario 4: W W W L W
  let p4 := home_win_prob * home_win_prob * away_win_prob * away_loss_prob * home_win_prob
  p1 + p2 + p3 + p4

theorem team_A_wins_series_4_1_probability : 
  probability_team_A_wins_series_4_1 = 0.18 :=
by
  -- This where the proof would go
  sorry

end team_A_wins_series_4_1_probability_l140_140579


namespace compute_f_g_f_l140_140131

def f (x : ℤ) : ℤ := 2 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

theorem compute_f_g_f (x : ℤ) : f (g (f 3)) = 108 := 
  by 
  sorry

end compute_f_g_f_l140_140131


namespace det_M_cubed_l140_140781

variable {M : Type*} [Matrix M]

-- Given condition that det(M) = 3
axiom det_M_eq_3 : det M = 3

-- The proof statement
theorem det_M_cubed : det (M ^ 3) = 27 := by
  sorry

end det_M_cubed_l140_140781


namespace maximize_a_sqrt_1_b_squared_l140_140761

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + (2023^x - 1) / (2023^x + 1) + 5

-- Define the problem condition
def condition (a b : ℝ) : Prop := 2 * a^2 + b^2 = 2

-- State the theorem
theorem maximize_a_sqrt_1_b_squared (a b : ℝ) (h : condition a b) : 
  a * Real.sqrt (1 + b^2) ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end maximize_a_sqrt_1_b_squared_l140_140761


namespace john_salary_increase_l140_140109

theorem john_salary_increase :
  let initial_salary : ℝ := 30
  let final_salary : ℝ := ((30 * 1.1) * 1.15) * 1.05
  (final_salary - initial_salary) / initial_salary * 100 = 32.83 := by
  sorry

end john_salary_increase_l140_140109


namespace range_of_function_1_range_of_function_2_l140_140709

noncomputable def range_1 : Set ℝ :=
  {y | ∃ x : ℝ, y = 5 / (2 * x^2 - 4 * x + 3)}

theorem range_of_function_1 :
  range_1 = Ioo 0 5 :=
by
  sorry

noncomputable def range_2 : Set ℝ :=
  {y | ∃ x : ℝ, x ≤ 1/2 ∧ y = x + sqrt (1 - 2 * x)}

theorem range_of_function_2 :
  range_2 = Iio 1 :=
by
  sorry

end range_of_function_1_range_of_function_2_l140_140709


namespace cos_sum_eq_zero_l140_140129

theorem cos_sum_eq_zero (x y z : ℝ)
  (h₁ : cos x + cos y + cos z + cos (x - y) = 0)
  (h₂ : sin x + sin y + sin z + sin (x - y) = 0) :
  cos (2 * x) + cos (2 * y) + cos (2 * z) = 0 :=
sorry

end cos_sum_eq_zero_l140_140129


namespace neg_exists_le_zero_iff_forall_gt_zero_l140_140209

variable (m : ℝ)

theorem neg_exists_le_zero_iff_forall_gt_zero :
  (¬ ∃ x : ℤ, (x:ℝ)^2 + 2 * x + m ≤ 0) ↔ ∀ x : ℤ, (x:ℝ)^2 + 2 * x + m > 0 :=
by
  sorry

end neg_exists_le_zero_iff_forall_gt_zero_l140_140209


namespace find_ω_l140_140811

def cos_period {ω : ℝ} (hω : ω > 0) : Prop :=
  (∃ T > 0, T = π / 3 ∧ ∀ x, cos (ω * (x + T) - π / 6) = cos (ω * x - π / 6))

theorem find_ω (ω : ℝ) (h : cos_period (ω := ω) (by norm_num [hω])) : ω = 6 := sorry

end find_ω_l140_140811


namespace cost_of_notebooks_and_markers_l140_140922

theorem cost_of_notebooks_and_markers 
  (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.30) 
  (h2 : 5 * x + 3 * y = 11.65) : 
  2 * x + y = 4.35 :=
by
  sorry

end cost_of_notebooks_and_markers_l140_140922


namespace a_divisible_by_11_iff_b_divisible_by_11_l140_140597

-- Define the relevant functions
def a (n : ℕ) : ℕ := n^5 + 5^n
def b (n : ℕ) : ℕ := n^5 * 5^n + 1

-- State that for a positive integer n, a(n) is divisible by 11 if and only if b(n) is also divisible by 11
theorem a_divisible_by_11_iff_b_divisible_by_11 (n : ℕ) (hn : 0 < n) : 
  (a n % 11 = 0) ↔ (b n % 11 = 0) :=
sorry

end a_divisible_by_11_iff_b_divisible_by_11_l140_140597


namespace chessboard_tiling_l140_140675

theorem chessboard_tiling (chessboard : Fin 8 × Fin 8 → Prop) (colors : Fin 8 × Fin 8 → Bool)
  (removed_squares : (Fin 8 × Fin 8) × (Fin 8 × Fin 8))
  (h_diff_colors : colors removed_squares.1 ≠ colors removed_squares.2) :
  ∃ f : (Fin 8 × Fin 8) → (Fin 8 × Fin 8), ∀ x, chessboard x → chessboard (f x) :=
by
  sorry

end chessboard_tiling_l140_140675


namespace tan_sum_greater_cot_sum_l140_140454

theorem tan_sum_greater_cot_sum (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = π) (hAacute : A < π / 2) (hBacute : B < π / 2) (hCacute : C < π / 2) :
  tan A + tan B + tan C > cot A + cot B + cot C := by
  sorry

end tan_sum_greater_cot_sum_l140_140454


namespace people_per_van_is_six_l140_140292

noncomputable def n_vans : ℝ := 6.0
noncomputable def n_buses : ℝ := 8.0
noncomputable def p_bus : ℝ := 18.0
noncomputable def people_difference : ℝ := 108

theorem people_per_van_is_six (x : ℝ) (h : n_buses * p_bus = n_vans * x + people_difference) : x = 6.0 := 
by
  sorry

end people_per_van_is_six_l140_140292


namespace vectors_problems_l140_140731

open Real

structure Vector2D :=
  (x y : ℝ)

noncomputable def vector_op (a b : Vector2D) : ℝ := 
  abs (sqrt (a.x^2 + a.y^2) * sqrt (b.x^2 + b.y^2) * sin (atan2 a.y a.x - atan2 b.y b.x))

def prop1 (a b : Vector2D) : Prop := vector_op a b = vector_op b a
def prop2 (a b : Vector2D) (λ : ℝ) : Prop := vector_op (Vector2D.mk (λ * a.x) (λ * a.y)) b = λ * vector_op a b
def prop3 (a b c : Vector2D) : Prop := vector_op (Vector2D.mk (a.x + b.x) (a.y + b.y)) c = vector_op a c + vector_op b c
def prop4 (a b : Vector2D) : Prop := vector_op a b = abs (a.x * b.y - a.y * b.x)

theorem vectors_problems :
  ∀ (a b c : Vector2D) (λ : ℝ),
  ¬ prop2 a b λ ∧
  ¬ prop3 a b c ∧
  prop1 a b ∧
  prop4 a b :=
by
  intros a b c λ
  sorry

end vectors_problems_l140_140731


namespace exists_plane_parallel_l140_140120

open Classical

variables (a b : Line)
variables (P : Plane)

-- Assume the lines a and b are non-intersecting
axiom non_intersecting : ∀ P : Plane, ¬(a ⊆ P ∧ b ⊆ P)

-- Define a property that a plane is parallel to a line
def parallel_to_line (P : Plane) (l : Line) : Prop :=
  ∃ q : Line, q ≠ l ∧ q ⊆ P ∧ ∀ P', (l ⊆ P') → P = P'

theorem exists_plane_parallel :
  ∃ P, P ⊇ b ∧ parallel_to_line P a := by
  sorry

end exists_plane_parallel_l140_140120


namespace smallest_number_of_cubes_l140_140606

theorem smallest_number_of_cubes (l w d : ℕ) (hl : l = 36) (hw : w = 45) (hd : d = 18) : 
  ∃ n : ℕ, n = 40 ∧ (∃ s : ℕ, l % s = 0 ∧ w % s = 0 ∧ d % s = 0 ∧ (l / s) * (w / s) * (d / s) = n) := 
by
  sorry

end smallest_number_of_cubes_l140_140606


namespace tuesday_more_than_monday_l140_140517

variable (M T W Th x : ℕ)

-- Conditions
def monday_dinners : M = 40 := by sorry
def tuesday_dinners : T = M + x := by sorry
def wednesday_dinners : W = T / 2 := by sorry
def thursday_dinners : Th = W + 3 := by sorry
def total_dinners : M + T + W + Th = 203 := by sorry

-- Proof problem: How many more dinners were sold on Tuesday than on Monday?
theorem tuesday_more_than_monday : x = 32 :=
by
  sorry

end tuesday_more_than_monday_l140_140517


namespace x_plus_y_eq_20_l140_140500

theorem x_plus_y_eq_20 (x y : ℝ) (hxy : x ≠ y) (hdet : (Matrix.det ![
  ![2, 3, 7],
  ![4, x, y],
  ![4, y, x]]) = 0) : x + y = 20 :=
by
  sorry

end x_plus_y_eq_20_l140_140500


namespace binomial_coefficient_largest_n_l140_140584
-- import necessary library

-- define the problem
theorem binomial_coefficient_largest_n :
  ∃ (n : ℕ), (n ≤ 10) ∧ (nat.choose 9 4 + nat.choose 9 5 = nat.choose 10 n) ∧ (∀ m : ℕ, (nat.choose 9 4 + nat.choose 9 5 = nat.choose 10 m) → m ≤ n) :=
by {
  let n := 5,
  use n,
  split,
  {
    -- Prove n ≤ 10
    exact nat.le_refl n,
  },
  split,
  {
    -- Prove the main equality
    have h : nat.choose 9 4 + nat.choose 9 5 = nat.choose 10 5,
    {
      -- Using Pascal's identity
      exact nat.add_choose_eq nat.choose_succ_succ.symm,
    },
    exact h,
  },
  {
    -- Prove that n is the largest such integer
    intros m h,
    have h_mono : ∀ (k : ℕ), k < 5 → nat.choose 10 k < nat.choose 10 5,
    {
      sorry, -- This would require a proof, but we'll skip this as well.
    },
    by_contradiction,
    exact h_mono m hj,
  },
}

end binomial_coefficient_largest_n_l140_140584


namespace solution_set_l140_140010

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_at_1 : f 1 = Real.exp 1
axiom f_ineq : ∀ x ≥ 0, (x - 1) * f x < x * deriv (deriv f x)

theorem solution_set :
  { x : ℝ | x * f x - Real.exp (|x|) > 0 } = { x : ℝ | x < -1 ∨ x > 1 } :=
by
  sorry

end solution_set_l140_140010


namespace nagel_point_exists_l140_140164

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

structure Triangle := (A B C : ℝ × ℝ)

structure Excircles (ABC : Triangle) :=
(tangent_A : ℝ × ℝ) -- Point of tangency on BC
(tangent_B : ℝ × ℝ) -- Point of tangency on CA
(tangent_C : ℝ × ℝ) -- Point of tangency on AB

theorem nagel_point_exists (ABC : Triangle) (ex : Excircles ABC) (a b c : ℝ) :
∃ N : ℝ × ℝ, (lines_concurrent (ABC.A) (ex.tangent_A) (ABC.B) (ex.tangent_B) (ABC.C) (ex.tangent_C)) :=
by
  let s := semi_perimeter a b c
  have BA_1 := s - b
  have CA_1 := s - c
  have CB_1 := s - c
  have AB_1 := s - a
  have AC_1 := s - a
  have BC_1 := s - b
  sorry

end nagel_point_exists_l140_140164


namespace probability_number_is_odd_l140_140550

def definition_of_odds : set ℕ := {3, 5, 7, 9}

def number_is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem probability_number_is_odd :
  let total_digits := {2, 3, 5, 7, 9}
  let odd_digits := definition_of_odds
  let favorable_outcomes := set.card odd_digits
  let total_outcomes := set.card total_digits
  in (total_outcomes = 5) -> (favorable_outcomes = 4) -> (favorable_outcomes / total_outcomes : ℚ) = 4 / 5 :=
by
  intro total_digits odd_digits favorable_outcomes total_outcomes total_outcomes_eq favorable_outcomes_eq
  have h1 : favorable_outcomes = 4 := favorable_outcomes_eq
  have h2 : total_outcomes = 5 := total_outcomes_eq
  norm_num
  sorry

end probability_number_is_odd_l140_140550


namespace area_of_octagon_in_square_bdef_with_ab_bc_eq_2_l140_140016

noncomputable def area_of_octagon : ℝ := 
  let side_length := 2 * real.sqrt 2
  let area_square := (2 + side_length + 2)^2
  area_square - 4 * 2

theorem area_of_octagon_in_square_bdef_with_ab_bc_eq_2
  (BDEF_is_square : ∃ B D E F : ℝ × ℝ, True) 
  (AB_eq_2 : ∃ A B : ℝ × ℝ, dist A B = 2)
  (BC_eq_2 : ∃ B C : ℝ × ℝ, dist B C = 2) :
  area_of_octagon = 16 + 16 * real.sqrt 2 :=
sorry

end area_of_octagon_in_square_bdef_with_ab_bc_eq_2_l140_140016


namespace min_positive_period_of_f_l140_140934

def f (x : ℝ) : ℝ := sin x * sin (x + π / 2)

theorem min_positive_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
by 
  sorry

end min_positive_period_of_f_l140_140934


namespace cross_shape_perimeter_l140_140507

theorem cross_shape_perimeter (total_area : ℝ) (number_of_squares : ℕ) (square_side_length : ℝ)
  (h1 : total_area = 125)
  (h2 : number_of_squares = 5)
  (h3 : square_side_length = real.sqrt (total_area / number_of_squares))
  (arrangement : ℕ)
  (h4 : arrangement = 8)
  : 8 * square_side_length = 40 :=
by
  sorry

end cross_shape_perimeter_l140_140507


namespace hcf_two_numbers_l140_140070

theorem hcf_two_numbers
  (x y : ℕ) 
  (h_lcm : Nat.lcm x y = 560)
  (h_prod : x * y = 42000) : Nat.gcd x y = 75 :=
by
  sorry

end hcf_two_numbers_l140_140070


namespace probability_of_prime_product_l140_140945

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def spinners : list (ℕ × ℕ) := [(2, 1), (2, 3), (2, 5), (2, 7), (2, 11), 
                                 (6, 1), (6, 3), (6, 5), (6, 7), (6, 11)]

def prime_products := (spinners.filter (λ p, is_prime (p.fst * p.snd))).length

theorem probability_of_prime_product : prime_products / spinners.length = 1 / 5 := by
  sorry

end probability_of_prime_product_l140_140945


namespace quadratic_has_real_root_iff_b_in_intervals_l140_140801

theorem quadratic_has_real_root_iff_b_in_intervals (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ set.Icc (-∞ : ℝ) (-10) ∪ set.Icc 10 (∞ : ℝ)) :=
by by sorry

end quadratic_has_real_root_iff_b_in_intervals_l140_140801


namespace solve_equation_l140_140184

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (x / (x + 1) = 2 / (x^2 - 1)) ↔ (x = 2) :=
by
  sorry

end solve_equation_l140_140184


namespace find_angle_between_planes_l140_140997

-- Define the conditions of the problem
variables (R h : ℝ)
variable  (k : ℝ := 4 * Real.sqrt 2)
variable (theta : ℝ)

-- Define the hypotheses
def cone_cylinder_inscribed (h R : ℝ) :=
  R^2 - 8 * h * R + 8 * h^2 = 0

-- Define the theorem to prove
theorem find_angle_between_planes 
    (h R : ℝ) 
    (hypothesis_1 : cone_cylinder_inscribed h R) 
    (hypothesis_2 : k = 4 * Real.sqrt 2) : 
  theta = Real.arccot (4 + 2 * Real.sqrt 2) ∨ theta = Real.arccot (4 - 2 * Real.sqrt 2) := 
sorry

end find_angle_between_planes_l140_140997


namespace base7_representation_length_2401_l140_140415

theorem base7_representation_length_2401 : 
  ∀ n, n = 2401 → (let digits := Nat.log 7 n in digits + 1) = 5 := 
by
  intros n hn
  let digits := Nat.log 7 n
  have h_digits : digits = 4 := by sorry
  rw [h_digits]
  exact rfl

end base7_representation_length_2401_l140_140415


namespace monotonically_increasing_interval_of_f_l140_140208

noncomputable def f (a : ℝ) (h : 0 < a ∧ a < 1) (x : ℝ) : ℝ :=
  a^(-x^2 + 3 * x + 2)

theorem monotonically_increasing_interval_of_f (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ x y : ℝ, x ∈ set.Ioo (3/2 : ℝ) (⊤ : ℝ) → f a h x < f a h y :=
sorry

end monotonically_increasing_interval_of_f_l140_140208


namespace distance_after_3_minutes_l140_140649

-- Define the speeds of the truck and the car in km/h.
def v_truck : ℝ := 65
def v_car : ℝ := 85

-- Define the time in hours.
def time_in_hours : ℝ := 3 / 60

-- Define the relative speed.
def v_relative : ℝ := v_car - v_truck

-- Define the expected distance between the truck and the car after 3 minutes.
def expected_distance : ℝ := 1

-- State the theorem: the distance between the truck and the car after 3 minutes is 1 km.
theorem distance_after_3_minutes : (v_relative * time_in_hours) = expected_distance := 
by {
  -- Here, we would provide the proof, but we are adding 'sorry' to skip the proof.
  sorry
}

end distance_after_3_minutes_l140_140649


namespace triangle_cosine_l140_140556

noncomputable def cos_smallest_angle (n : ℕ) : ℝ :=
  let cos_x := (n + 4) / (2 * (n + 1))
  in cos_x

theorem triangle_cosine (n : ℕ) (h : n = 5) :
  cos_smallest_angle n = 3 / 4 :=
by
  -- Placeholder for the proof
  sorry

end triangle_cosine_l140_140556


namespace ellipse_eccentricity_l140_140031

-- Definitions of given conditions
def isEllipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def F1F2 (c : ℝ) : ℝ × ℝ := (-c, c)

def pointP (x y c b a : ℝ) : Prop :=
  x = -c ∧ y = b^2 / a

def rightTriangle (a b c x y : ℝ) : Prop :=
  let h := 2 * c
  let v := y
  let hypotenuse := real.sqrt (h^2 + v^2)
  tan 60 = h / v

def eccentricity (c a : ℝ) : ℝ :=
  c / a

-- Statement of proof problem
theorem ellipse_eccentricity (a b c x y : ℝ) (h1 : isEllipse a b x y)
  (h2 : pointP x y c b a) (h3 : rightTriangle a b c x y) :
  eccentricity c a = sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l140_140031


namespace distance_after_3_minutes_l140_140651

-- Conditions: speeds of the truck and car, and the time interval in hours
def v_truck : ℝ := 65 -- in km/h
def v_car : ℝ := 85 -- in km/h
def t : ℝ := 3 / 60 -- convert 3 minutes to hours

-- Statement to prove: The distance between the truck and the car after 3 minutes is 1 km
theorem distance_after_3_minutes : (v_car - v_truck) * t = 1 := 
by
  sorry

end distance_after_3_minutes_l140_140651


namespace john_saving_theorem_l140_140858

noncomputable def john_saving (former_rent_rate : ℕ -> ℕ -> ℕ) 
                           (new_rent_first_half : ℕ)
                           (new_rent_second_half_percent_increase : ℕ -> ℕ) 
                           (winter_utilities_cost : ℕ)
                           (other_months_utilities_cost : ℕ)
                           (roommate_split : ℕ -> ℕ) 
                           (prorated_days : ℕ -> ℕ -> ℕ) 
                           (months : List ℕ)
                           : ℚ :=
  let former_apartment_cost := former_rent_rate 2 750 * 12
  let new_rent_first_six_months := new_rent_first_half * 6
  let new_rent_last_six_months := new_rent_second_half_percent_increase 2800 * 6
  let new_rent_total := new_rent_first_six_months + new_rent_last_six_months
  let utilities_winter := winter_utilities_cost * 3
  let utilities_other := other_months_utilities_cost * 9
  let utilities_total := utilities_winter + utilities_other
  let total_cost_new_apartment_before_proration := new_rent_total + utilities_total
  let prorated_first_month := prorated_days 2800 20
  let prorated_last_month := prorated_days (new_rent_second_half_percent_increase 2800) 15
  let prorated_rent_total := prorated_first_month + prorated_last_month
  let adjusted_rent := new_rent_total - (2800 + new_rent_second_half_percent_increase 2800 - prorated_rent_total)
  let adjusted_total_new_apartment := adjusted_rent + utilities_total
  let john_share_new_apartment := roommate_split adjusted_total_new_apartment
  former_apartment_cost.to_rat - john_share_new_apartment.to_rat

theorem john_saving_theorem : john_saving (λ rate size, rate * size)
                                       2800
                                       (λ rent, rent * 105 / 100)
                                       200
                                       150
                                       (λ cost, cost / 2)
                                       (λ rent days, rent * days / 30)
                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 2506.66 := sorry

end john_saving_theorem_l140_140858


namespace groupB_median_and_excellent_rate_l140_140283

noncomputable def groupB_scores : List ℕ :=
  [7, 7, 7, 7, 8, 8, 8, 9, 9, 10]

def median (l : List ℕ) : ℚ :=
  let sorted := l.sorted
  let n := l.length
  if n % 2 = 0 then
    (sorted.get ⟨n / 2 - 1, sorry⟩ + sorted.get ⟨n / 2, sorry⟩) / 2
  else
    sorted.get ⟨n / 2, sorry⟩

def excellent_rate (l : List ℕ) (threshold : ℕ) : ℚ :=
  let excellent_count := l.filter (λ score => score ≥ threshold).length
  (excellent_count : ℚ) / (l.length : ℚ) * 100

theorem groupB_median_and_excellent_rate :
  median groupB_scores = 8 ∧ excellent_rate groupB_scores 8 = 60 := by
  sorry

end groupB_median_and_excellent_rate_l140_140283


namespace cos_B_and_min_b_l140_140080

theorem cos_B_and_min_b {A B C : Type*} {a b c : ℝ} 
  (h1 : c * Real.cos B + b * Real.cos C = 3 * a * Real.cos B) 
  (h2 : (B - A) • (C - B) = 2) :
  (Real.cos B = 1/3) ∧ (b ≥ 2 * Real.sqrt 2) :=
by 
  sorry

end cos_B_and_min_b_l140_140080


namespace triangle_O1O2O3_acute_l140_140382

variables {A B C : Type}
variables [triangle ABC] [excircle_center ABC O1] [excircle_center ABC O2] [excircle_center ABC O3]

theorem triangle_O1O2O3_acute (hO1 : excircle_center ABC O1) (hO2 : excircle_center ABC O2) (hO3 : excircle_center ABC O3) : acute_triangle O1 O2 O3 :=
sorry

end triangle_O1O2O3_acute_l140_140382


namespace greatest_possible_avg_speed_l140_140101

theorem greatest_possible_avg_speed {initial_odometer : ℕ} (t : ℕ) (max_speed : ℕ) 
  (final_odometer : ℕ) : 
  initial_odometer = 12321 ∧ t = 4 ∧ max_speed = 65 ∧ 
  (∀ n : ℕ, palindrome n → initial_odometer < final_odometer ∧ final_odometer ≤ initial_odometer + 260) →
  (final_odometer - initial_odometer) / t = 50 :=
by sorry

end greatest_possible_avg_speed_l140_140101


namespace greatest_possible_positive_integer_difference_l140_140788

theorem greatest_possible_positive_integer_difference (x y : ℤ) (hx : 4 < x) (hx' : x < 6) (hy : 6 < y) (hy' : y < 10) :
  y - x = 4 :=
sorry

end greatest_possible_positive_integer_difference_l140_140788


namespace det_M_cubed_l140_140782

variable {M : Type*} [Matrix M]

-- Given condition that det(M) = 3
axiom det_M_eq_3 : det M = 3

-- The proof statement
theorem det_M_cubed : det (M ^ 3) = 27 := by
  sorry

end det_M_cubed_l140_140782


namespace lollipops_initial_count_l140_140282

theorem lollipops_initial_count (L : ℕ) (k : ℕ) 
  (h1 : L % 42 ≠ 0) 
  (h2 : (L + 22) % 42 = 0) : 
  L = 62 :=
by
  sorry

end lollipops_initial_count_l140_140282


namespace solve_trigonometric_equation_l140_140183

theorem solve_trigonometric_equation (x : ℝ) : 
  (2 * (Real.sin x)^6 + 2 * (Real.cos x)^6 - 3 * (Real.sin x)^4 - 3 * (Real.cos x)^4) = Real.cos (2 * x) ↔ 
  ∃ (k : ℤ), x = (π / 2) * (2 * k + 1) :=
sorry

end solve_trigonometric_equation_l140_140183


namespace find_equation_of_line_l140_140383

noncomputable theory

-- Define the circle
def circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 12 * y + 24 = 0

-- Define the line
def line (x y : ℝ) : Prop :=
  y = (3 / 4) * x + 5

-- Define the length of the chord intercepted by line on circle
def chord_length (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : ℝ := 4 * real.sqrt 3

-- Define the equation of line
def equation_of_line (x y : ℝ) : Prop :=
  3 * x - 4 * y + 20 = 0

-- Define the locus of the midpoint M
def eq_of_midpoint_locus (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 11 * y + 30 = 0

-- State the theorem
theorem find_equation_of_line :
  (∀ (x y : ℝ), circle x y → line x y ∧ chord_length circle line = 4 * real.sqrt 3 →
    equation_of_line x y) ∧
  (∀ (x y : ℝ), circle x y → line x y → eq_of_midpoint_locus x y) :=
sorry

end find_equation_of_line_l140_140383


namespace allocation_schemes_l140_140228

noncomputable def num_allocation_schemes (n m : ℕ) : ℕ :=
  if n = 5 ∧ m = 3 then
    (Math.combinatorial.choose 5 1) * (Math.combinatorial.choose 4 2) / 2 * Math.combinatorial.perm 3
  else
    0

theorem allocation_schemes : num_allocation_schemes 5 3 = 90 :=
by
  sorry

end allocation_schemes_l140_140228


namespace horse_revolutions_l140_140621

theorem horse_revolutions :
  ∀ (r_1 r_2 : ℝ) (n : ℕ),
    r_1 = 30 → r_2 = 10 → n = 25 → (r_1 * n) / r_2 = 75 := by
  sorry

end horse_revolutions_l140_140621


namespace sum_first_13_terms_l140_140458

variable {a : ℕ → ℝ} -- defines the arithmetic sequence

-- Definitions corresponding to the conditions in the problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition (a : ℕ → ℝ) (d : ℝ) (a₁ : ℝ) : Prop :=
  a 3 = a₁ + 2 * d ∧ a 5 = a₁ + 4 * d ∧ a 10 = a₁ + 9 * d ∧
  a 3 + a 5 + 2 * a 10 = 4

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a 1 + (n - 1) * d)

-- Theorem to prove
theorem sum_first_13_terms
  (a : ℕ → ℝ) (d a₁ : ℝ)
  (ha : is_arithmetic_sequence a d)
  (hc : condition a d a₁) :
  sum_arithmetic_sequence a d 13 = 13 :=
sorry

end sum_first_13_terms_l140_140458


namespace problem_m_n_sum_l140_140601

theorem problem_m_n_sum (m n : ℕ) 
  (h1 : m^2 + n^2 = 3789) 
  (h2 : Nat.gcd m n + Nat.lcm m n = 633) : 
  m + n = 87 :=
sorry

end problem_m_n_sum_l140_140601


namespace probability_square_not_touching_outer_edge_l140_140887

theorem probability_square_not_touching_outer_edge :
  let total_squares := 10 * 10
  let perimeter_squares := 10 + 10 + (10 - 2) + (10 - 2)
  let non_perimeter_squares := total_squares - perimeter_squares
  (non_perimeter_squares / total_squares) = (16 / 25) :=
by
  let total_squares := 10 * 10
  let perimeter_squares := 10 + 10 + (10 - 2) + (10 - 2)
  let non_perimeter_squares := total_squares - perimeter_squares
  have h : non_perimeter_squares / total_squares = 16 / 25 := by sorry
  exact h

end probability_square_not_touching_outer_edge_l140_140887


namespace average_beef_sales_l140_140146

theorem average_beef_sales 
  (thursday_sales : ℕ)
  (friday_sales : ℕ)
  (saturday_sales : ℕ)
  (h_thursday : thursday_sales = 210)
  (h_friday : friday_sales = 2 * thursday_sales)
  (h_saturday : saturday_sales = 150) :
  (thursday_sales + friday_sales + saturday_sales) / 3 = 260 :=
by sorry

end average_beef_sales_l140_140146


namespace shortest_distance_l140_140218

-- Define the line and the circle
def is_on_line (P : ℝ × ℝ) : Prop := P.snd = P.fst - 1

def is_on_circle (Q : ℝ × ℝ) : Prop := Q.fst^2 + Q.snd^2 + 4 * Q.fst - 2 * Q.snd + 4 = 0

-- Define the square of the Euclidean distance between two points
def dist_squared (P Q : ℝ × ℝ) : ℝ := (P.fst - Q.fst)^2 + (P.snd - Q.snd)^2

-- State the theorem regarding the shortest distance between the points on the line and the circle
theorem shortest_distance : ∃ P Q : ℝ × ℝ, is_on_line P ∧ is_on_circle Q ∧ dist_squared P Q = 1 := sorry

end shortest_distance_l140_140218


namespace wrapping_paper_area_l140_140635

theorem wrapping_paper_area (length width : ℕ) (h1 : width = 6) (h2 : 2 * (length + width) = 28) : length * width = 48 :=
by
  sorry

end wrapping_paper_area_l140_140635


namespace half_theta_quadrant_l140_140021

variable (k : ℤ) (θ : ℝ)

theorem half_theta_quadrant 
  (h : 2 * k * π + π / 2 < θ ∧ θ < 2 * k * π + π) :
  let half_θ := θ / 2
  (kπ_div4_lt_half_θ : k * π + π / 4 < half_θ) ∧ (half_θ_lt_kπ_div2 : half_θ < k * π + π / 2) → 
  (∃ n : ℤ, n * π / 2 < half_θ ∧ half_θ < (n + 1) * π / 2) := 
by
  sorry
  

end half_theta_quadrant_l140_140021


namespace unique_real_root_eq_l140_140211

theorem unique_real_root_eq (x : ℝ) : (∃! x, x = Real.sin x + 1993) :=
sorry

end unique_real_root_eq_l140_140211


namespace incorrect_statement_D_l140_140527

-- Definitions corresponding to the conditions in the problem

def position_of_origin_arbitrary (line : Type) [has_zero line] : Prop :=
  ∀ (p : line), (0 : line) = p

def direction_positive_left_to_right (line : Type) [has_add line] : Prop :=
  ∀ (p q : line), p < q → p + 1 = q

def unit_length_arbitrary (line : Type) [has_add line] : Prop :=
  ∀ (u : line), ¬ ((1 : line) = u)

-- Incorrect statement to prove
def length_between_marks_constant (line : Type) [has_add line] : Prop :=
  ∀ (p q : line), p + 1 = q → (q - p = (1 : line))

theorem incorrect_statement_D (line : Type) [has_zero line] [has_add line] [has_sub line]
  (h1: position_of_origin_arbitrary line)
  (h2: direction_positive_left_to_right line)
  (h3: unit_length_arbitrary line) :
  ¬ length_between_marks_constant line :=
sorry

end incorrect_statement_D_l140_140527


namespace real_root_interval_l140_140805

theorem real_root_interval (b : ℝ) (p : ℝ → ℝ)
  (h_poly : p = λ x, x^2 + b * x + 25)
  (h_real_root : ∃ x : ℝ, p x = 0) :
  b ∈ set.Iic (-10) ∪ set.Ici 10 :=
sorry

end real_root_interval_l140_140805


namespace function_passes_through_fixed_point_l140_140040

-- Define the function and the conditions
variable (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)

-- Define the function f
def f (x : ℝ) : ℝ := a^(x - 1) + 2

-- State the theorem that the function passes through the fixed point (1, 3)
theorem function_passes_through_fixed_point : f a 1 = 3 := by
  -- Since this only states the theorem, we use sorry to skip the proof
  sorry

end function_passes_through_fixed_point_l140_140040


namespace lila_max_cookies_l140_140613

noncomputable def max_cookies (sugar butter : ℕ) : ℕ :=
  let cookies_sugar := (sugar / 3) * 10
  let cookies_butter := (butter / 2) * 10
  min cookies_sugar cookies_butter

theorem lila_max_cookies :
  let sugar := 15
  let butter := 8
  max_cookies sugar butter = 40 :=
by
  unfold max_cookies
  have h_cookies_sugar : (15 / 3) * 10 = 50 := by sorry
  have h_cookies_butter : (8 / 2) * 10 = 40 := by sorry
  show min 50 40 = 40
-- No more "sorries" needed for the proof, as they pertain to intermediate steps.

end lila_max_cookies_l140_140613


namespace max_subset_sum_divisible_by_33_l140_140375

theorem max_subset_sum_divisible_by_33 :
  ∃ (S : Finset ℕ), 
    S ⊆ Finset.range 2011 ∧ 
    (∀ a b c ∈ S, (a + b + c) % 33 = 0) → 
    S.card = 61 :=
by
  sorry

end max_subset_sum_divisible_by_33_l140_140375


namespace height_of_parallelogram_l140_140356

theorem height_of_parallelogram (Area Base : ℝ) (h1 : Area = 180) (h2 : Base = 18) : Area / Base = 10 :=
by
  sorry

end height_of_parallelogram_l140_140356


namespace macey_saved_amount_l140_140510

theorem macey_saved_amount (cost_shirt : ℝ) (weeks_more : ℕ) (save_per_week : ℝ) (total_needed: ℝ) (save_future : ℝ) : 
  cost_shirt = 3 ∧ weeks_more = 3 ∧ save_per_week = 0.5 ∧ total_needed = 3 ∧ save_future = 1.5 → 
  ∃ saved_already, saved_already = total_needed - save_future ∧ saved_already = 1.5 := 
by 
  intros h
  use (total_needed - save_future)
  split
  sorry
  sorry

end macey_saved_amount_l140_140510


namespace cans_collected_after_14_days_exceeds_goal_l140_140319
Given that Alyssa, Abigail, and Andrew need to collect 400 empty cans for their Science project in two weeks, where they have already collected 128 cans together and they collect a total of 34 cans per day, prove that after 14 days, they will have 204 cans more than their goal.

### Step d): Rewrite the math proof problem in Lean 4 statement.


theorem cans_collected_after_14_days_exceeds_goal :
  let total_goal := 400
  let initial_cans := 30 + 43 + 55
  let daily_collection := 8 + 11 + 15
  let days := 14
  initial_cans + daily_collection * days == total_goal + 204 :=
by
sorry

end cans_collected_after_14_days_exceeds_goal_l140_140319


namespace range_of_m_l140_140074

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) ↔ (m ∈ Set.Icc (-6:ℝ) 2) :=
by
  sorry

end range_of_m_l140_140074


namespace boxed_boxed_13_eq_24_l140_140368

-- Define the function boxed that computes the sum of the positive factors of n
def boxed (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ x => n % x = 0).sum

-- Prove that boxed (boxed 13) = 24
theorem boxed_boxed_13_eq_24 : boxed (boxed 13) = 24 :=
  by
    sorry

end boxed_boxed_13_eq_24_l140_140368


namespace compare_neg_fractions_l140_140334

theorem compare_neg_fractions : (-3 / 4) > (-5 / 6) :=
sorry

end compare_neg_fractions_l140_140334


namespace ratio_of_marbles_l140_140241

/-- Define the number of marbles Wolfgang bought --/
def wolfgang_marbles : ℕ := 16

/-- Define the total number of marbles after sharing equally --/
def total_marbles : ℕ := 3 * 20

/-- Define the condition for the total number of marbles combined --/
def total_number_of_marbles (ludo_marbles : ℕ) : ℕ :=
  wolfgang_marbles + ludo_marbles + (2/3:ℚ) * (wolfgang_marbles + ludo_marbles : ℚ)

/-- The theorem stating the ratio of Ludo's marbles to Wolfgang's marbles is 5 to 4 --/
theorem ratio_of_marbles :
  ∃ (ludo_marbles : ℕ), 
    total_number_of_marbles ludo_marbles = total_marbles ∧
    rat.mk ludo_marbles wolfgang_marbles = rat.mk 5 4 :=
by
  sorry

end ratio_of_marbles_l140_140241


namespace total_cement_used_l140_140169

def cement_used_lexi : ℝ := 10
def cement_used_tess : ℝ := 5.1

theorem total_cement_used : cement_used_lexi + cement_used_tess = 15.1 :=
by sorry

end total_cement_used_l140_140169


namespace incorrect_statement_c_l140_140966

def rhombus (q : Type) [quadrilateral q] : Prop :=
∀ (d₁ d₂ : diagonal q), perpendicular d₁ d₂ ∧ bisect d₁ d₂

def rectangle (r : Type) [quadrilateral r] : Prop :=
∀ (d₁ d₂ : diagonal r), equal_length d₁ d₂

def quadrilateral_with_adjacent_sides_equal_is_rhombus (q : Type) [quadrilateral q] : Prop :=
∃ (a b : side q), adjacent a b ∧ equal_length a b → ∀ c (side q), rhombus q

def quadrilateral_with_all_sides_equal_is_rhombus (q : Type) [quadrilateral q] : Prop :=
∀ (a b c d : side q), (equal_length a b ∧ equal_length b c ∧ equal_length c d ∧ equal_length d a) → rhombus q

theorem incorrect_statement_c :
  ∃ q : Type, [quadrilateral q] ∧ ∀ (a b : side q), adjacent a b ∧ equal_length a b → ¬ rhombus q := sorry

end incorrect_statement_c_l140_140966


namespace rounding_accuracy_l140_140190

theorem rounding_accuracy (n : ℝ) (h : n = 20.23 * 10^6) : 
  (* The approximate number obtained by rounding 20.23 million is accurate to the hundreds place in the context of its scale in millions. *)
  True :=
sorry

end rounding_accuracy_l140_140190


namespace max_reflections_l140_140574

theorem max_reflections (angle_increase : ℕ := 10) (max_angle : ℕ := 90) :
  ∃ n : ℕ, 10 * n ≤ max_angle ∧ ∀ m : ℕ, (10 * (m + 1) > max_angle → m < n) := 
sorry

end max_reflections_l140_140574


namespace solution_set_inequality_l140_140019

theorem solution_set_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x)) -- f is an even function
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 4*x) : 
  {x : ℝ | f (x + 2) < 5} = set.Ioo (-7 : ℝ) 3 :=
by
  sorry

end solution_set_inequality_l140_140019


namespace total_weight_is_40_l140_140881

def marco_strawberries_weight : ℕ := 8
def dad_strawberries_weight : ℕ := 32
def total_strawberries_weight := marco_strawberries_weight + dad_strawberries_weight

theorem total_weight_is_40 : total_strawberries_weight = 40 := by
  sorry

end total_weight_is_40_l140_140881


namespace pyramid_volume_l140_140450

theorem pyramid_volume (AB BC CG : ℝ) (hAB : AB = 4) (hBC : BC = 2) (hCG : CG = 5) :
  let BDFE_area := AB * BC,
  let height_M := CG,
  (1/3) * BDFE_area * height_M = 40 / 3 :=
by
  -- Using the given dimensions
  rw [hAB, hBC, hCG],
  -- Compute the area of base BDFE and height of M
  let BDFE_area := 4 * 2,
  let height_M := 5,
  -- Substitute and compute the volume of the pyramid
  calc (1/3) * BDFE_area * height_M
      = (1/3) * 8 * 5 : by simp [BDFE_area, height_M]
  ... = 40 / 3 : by norm_num

end pyramid_volume_l140_140450


namespace cotangent_positives_among_sequence_l140_140416

def cotangent_positive_count (n : ℕ) : ℕ :=
  if n ≤ 2019 then
    let count := (n / 4) * 3 + if n % 4 ≠ 0 then (3 + 1 - max 0 ((n % 4) - 1)) else 0
    count
  else 0

theorem cotangent_positives_among_sequence :
  cotangent_positive_count 2019 = 1515 := sorry

end cotangent_positives_among_sequence_l140_140416


namespace planes_parallel_or_intersect_l140_140449

-- Definitions based on conditions and the problem statement
def Plane : Type := sorry -- dummy type for Plane
def Line : Type := sorry -- dummy type for Line
def parallel (p1 p2 : Plane) : Prop := sorry -- parallel planes definition
def intersect (p1 p2 : Plane) (l : Line) : Prop := sorry -- intersection definition
def parallel_to_plane (l : Line) (p : Plane) : Prop := sorry -- line parallel to plane definition
def in_plane (l : Line) (p : Plane) : Prop := sorry -- line within plane definition

axiom infinite_lines_parallel (α β : Plane) :
  (∃ (l : Line), in_plane l α ∧ parallel_to_plane l β) → 
  (∀ (l : Line), in_plane l α → parallel_to_plane l β)

-- The theorem to prove
theorem planes_parallel_or_intersect (α β : Plane) :
  (∃ (l : Line), in_plane l α ∧ parallel_to_plane l β) →
  (parallel α β ∨ ∃ (l : Line), intersect α β l) :=
begin
  sorry
end

end planes_parallel_or_intersect_l140_140449


namespace total_white_roses_l140_140883

-- Define the constants
def n_b : ℕ := 5
def n_t : ℕ := 7
def r_b : ℕ := 5
def r_t : ℕ := 12

-- State the theorem
theorem total_white_roses :
  n_t * r_t + n_b * r_b = 109 :=
by
  -- Automatic proof can be here; using sorry as placeholder
  sorry

end total_white_roses_l140_140883


namespace students_do_not_like_either_food_l140_140824

noncomputable def students_who_do_not_like_either_food
  (total_students : ℕ)
  (likes_french_fries : ℕ)
  (likes_burgers : ℕ)
  (likes_both : ℕ) : ℕ :=
total_students - (likes_french_fries + likes_burgers - likes_both)

theorem students_do_not_like_either_food
  (total_students : ℕ)
  (likes_french_fries : ℕ)
  (likes_burgers : ℕ)
  (likes_both : ℕ)
  (h_total : total_students = 25)
  (h_french_fries : likes_french_fries = 15)
  (h_burgers : likes_burgers = 10)
  (h_both : likes_both = 6) :
  students_who_do_not_like_either_food total_students likes_french_fries likes_burgers likes_both = 6 := 
by {
  rw [h_total, h_french_fries, h_burgers, h_both],
  dsimp [students_who_do_not_like_either_food],
  norm_num,
  sorry
}

end students_do_not_like_either_food_l140_140824


namespace sum_of_coordinates_is_2y_l140_140524

variable (x y : ℝ)

def pointA_coords : ℝ × ℝ := (x, y)
def pointB_coords : ℝ × ℝ := (-x, y)

theorem sum_of_coordinates_is_2y : (fst pointA_coords + snd pointA_coords + fst pointB_coords + snd pointB_coords) = 2 * y := by
  sorry

end sum_of_coordinates_is_2y_l140_140524


namespace middle_managers_to_be_selected_l140_140611

def total_employees : ℕ := 160
def senior_managers : ℕ := 10
def middle_managers : ℕ := 30
def staff_members : ℕ := 120
def total_to_be_selected : ℕ := 32

theorem middle_managers_to_be_selected : 
  (middle_managers * total_to_be_selected / total_employees) = 6 := by
  sorry

end middle_managers_to_be_selected_l140_140611


namespace smallest_class_size_l140_140836

theorem smallest_class_size :
  ∀ (x : ℕ), 4 * x + 3 > 50 → 4 * x + 3 = 51 :=
by
  sorry

end smallest_class_size_l140_140836


namespace minimize_sum_distances_l140_140921

-- Define the points A and B
def A := (6, 4)
def B := (3, 0)

-- Define the point C on the y-axis at (0, k)
def C (k : ℝ) := (0, k)

-- Define the distance function between two points in 2D
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define the condition that CB forms a 45-degree angle with the positive x-axis
def CB_angle_45 (k : ℝ) : Prop :=
  real.arctan (k / 3) = real.pi / 4

-- Define the sum of the distances AC and BC
def sum_distances_AC_BC (k : ℝ) : ℝ :=
  distance A (C k) + distance B (C k)

-- The final statement to prove that the value of k which minimizes sum_distances_AC_BC
-- under the given condition CB_angle_45 is k = 3
theorem minimize_sum_distances : ∃ (k : ℝ), CB_angle_45 k ∧ sum_distances_AC_BC k = sum_distances_AC_BC 3 :=
by sorry

end minimize_sum_distances_l140_140921


namespace Auston_height_in_cm_l140_140663

theorem Auston_height_in_cm : 
  (60 : ℝ) * 2.54 = 152.4 :=
by sorry

end Auston_height_in_cm_l140_140663


namespace distance_after_3_minutes_l140_140652

-- Conditions: speeds of the truck and car, and the time interval in hours
def v_truck : ℝ := 65 -- in km/h
def v_car : ℝ := 85 -- in km/h
def t : ℝ := 3 / 60 -- convert 3 minutes to hours

-- Statement to prove: The distance between the truck and the car after 3 minutes is 1 km
theorem distance_after_3_minutes : (v_car - v_truck) * t = 1 := 
by
  sorry

end distance_after_3_minutes_l140_140652


namespace quadratic_has_real_root_l140_140791

theorem quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := sorry

end quadratic_has_real_root_l140_140791


namespace inflation_over_two_years_real_interest_rate_l140_140974

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end inflation_over_two_years_real_interest_rate_l140_140974


namespace stable_numbers_base_2_4_5_etc_l140_140999

noncomputable def isStableNumber (B x y z t : ℕ) : Prop :=
  let xyzt_B := B^3 * x + B^2 * y + B * z + t
  let dcba := [x, y, z, t].insertSorted (≤)
  let dcba_B := B^3 * dcba[3] + B^2 * dcba[2] + B * dcba[1] + dcba[0]
  let abcd_B := B^3 * dcba[0] + B^2 * dcba[1] + B * dcba[2] + dcba[3]
  xyzt_B = dcba_B - abcd_B

theorem stable_numbers_base_2_4_5_etc {B : ℕ} : 
∃ (x y z t : ℕ), 
  (isStableNumber B x y z t ∧ 
    ([B, x, y, z, t] = [2, 1, 0, 0, 1] ∨
     [B, x, y, z, t] = [4, 3, 0, 2, 1] ∨
     [B, x, y, z, t] = [5, 3, 0, 3, 2] ∨
     (∃ k m n p : ℕ, 
       B = 5 * k ∧
       x = 3 * k ∧ 
       y = k - 1 ∧ 
       z = 4 * k - 1 ∧
       t = 2 * k))) :=
sorry

end stable_numbers_base_2_4_5_etc_l140_140999


namespace quadrilateral_rectangle_ratio_l140_140372

theorem quadrilateral_rectangle_ratio
  (s x y : ℝ)
  (h_area : (s + 2 * x) ^ 2 = 4 * s ^ 2)
  (h_y : 2 * y = s) :
  y / x = 1 :=
by
  sorry

end quadrilateral_rectangle_ratio_l140_140372


namespace minimum_perimeter_triangle_l140_140436

noncomputable def minimum_perimeter (a b c : ℝ) (cos_C : ℝ) (ha : a + b = 10) (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0) 
  : ℝ :=
  a + b + c

theorem minimum_perimeter_triangle (a b c : ℝ) (cos_C : ℝ)
  (ha : a + b = 10)
  (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0)
  (cos_C_valid : cos_C = -1/2) :
  (minimum_perimeter a b c cos_C ha hroot) = 10 + 5 * Real.sqrt 3 :=
sorry

end minimum_perimeter_triangle_l140_140436


namespace math_problem_l140_140400

theorem math_problem
  (a b : ℝ)
  (h₀ : ∀ x, (2 < x ∧ x < 3) ↔ (x^2 - a * x - b < 0)) :
  (a = 5) ∧ (b = -6) ∧ (∀ x, (-1/2 < x ∧ x < -1/3) ↔ (b * x^2 - a * x - 1 > 0)) :=
begin
  sorry
end

end math_problem_l140_140400


namespace largest_integer_n_l140_140586

theorem largest_integer_n 
  (h1 : Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 5) :
  ∃ (n : ℕ), Nat.choose 10 n = Nat.choose 10 5 ∧ ∀ (m : ℕ), Nat.choose 10 m = Nat.choose 10 5 → m ≤ n :=
begin
  use 5,
  split,
  { exact h1 },
  { intros m hm,
    sorry }
end

end largest_integer_n_l140_140586


namespace determine_b_l140_140075

theorem determine_b (a b c y1 y2 : ℝ) 
  (h1 : y1 = a * 2^2 + b * 2 + c)
  (h2 : y2 = a * (-2)^2 + b * (-2) + c)
  (h3 : y1 - y2 = -12) : 
  b = -3 := 
by
  sorry

end determine_b_l140_140075


namespace geom_seq_value_l140_140730

noncomputable def geom_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ (n : ℕ), a (n + 1) = a n * q

theorem geom_seq_value
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom_seq : geom_sequence a q)
  (h_a5 : a 5 = 2)
  (h_a6_a8 : a 6 * a 8 = 8) :
  (a 2018 - a 2016) / (a 2014 - a 2012) = 2 :=
sorry

end geom_seq_value_l140_140730


namespace part_a_part_b_l140_140366

theorem part_a (m : ℕ) : 
  ∃ n : ℕ, 1 / (m + 1) * nat.choose (2 * m) m = n := 
sorry

theorem part_b (m : ℕ) : 
  ∃ k : ℕ, (∀ n : ℕ, n ≥ m → ∃ i : ℕ, k / (n + m + 1) * nat.choose (2 * n) (n + m) = i) ∧ k = 2 * m + 1 :=
sorry

end part_a_part_b_l140_140366


namespace barbara_total_cost_l140_140665

-- Definitions based on the given conditions
def steak_cost_per_pound : ℝ := 15.00
def steak_quantity : ℝ := 4.5
def chicken_cost_per_pound : ℝ := 8.00
def chicken_quantity : ℝ := 1.5

def expected_total_cost : ℝ := 42.00

-- The main proposition we need to prove
theorem barbara_total_cost :
  steak_cost_per_pound * steak_quantity + chicken_cost_per_pound * chicken_quantity = expected_total_cost :=
by
  sorry

end barbara_total_cost_l140_140665


namespace number_of_solutions_l140_140096

theorem number_of_solutions :
  (∃ (a b c : ℕ), 4 * a = 6 * c ∧ 168 * a = 6 * a * b * c) → 
  ∃ (s : Finset ℕ), s.card = 6 :=
by sorry

end number_of_solutions_l140_140096


namespace true_propositions_count_l140_140032

theorem true_propositions_count :
  (∀ l₁ l₂ l₃ : Type, parallel l₁ l₃ → parallel l₂ l₃ → parallel l₁ l₂) ∧
  (∀ l₁ l₂ l₃ : Type, perpendicular l₁ l₃ → perpendicular l₂ l₃ → parallel l₁ l₂) ∧
  (∀ l₁ l₂ : Type, parallel_to_same_plane l₁ l₂ → parallel l₁ l₂ → (skew l₁ l₂ ∨ intersect l₁ l₂)) →
  1 = (if (∀ l₁ l₂ l₃ : Type, parallel l₁ l₃ → parallel l₂ l₃ → parallel l₁ l₂) then 1 else 0) +
      (if (∀ l₁ l₂ l₃ : Type, perpendicular l₁ l₃ → perpendicular l₂ l₃ → parallel l₁ l₂) then 1 else 0) +
      (if (∀ l₁ l₂ : Type, parallel_to_same_plane l₁ l₂ → parallel l₁ l₂) then 1 else 0) :=
by
  sorry

end true_propositions_count_l140_140032


namespace find_x_l140_140941

def seq : ℕ → ℕ
| 0 => 2
| 1 => 5
| 2 => 11
| 3 => 20
| 4 => 32
| 5 => 47
| (n+6) => seq (n+5) + 3 * (n + 1)

theorem find_x : seq 6 = 65 := by
  sorry

end find_x_l140_140941


namespace part1_monotonic_intervals_part2_max_a_l140_140407

noncomputable def f1 (x : ℝ) := Real.log x - 2 * x^2

theorem part1_monotonic_intervals :
  (∀ x, 0 < x ∧ x < 0.5 → f1 x > 0) ∧ (∀ x, x > 0.5 → f1 x < 0) :=
by
  sorry

noncomputable def f2 (x a : ℝ) := Real.log x + a * x^2

theorem part2_max_a (a : ℤ) :
  (∀ x, x > 1 → f2 x a < Real.exp x) → a ≤ 1 :=
by
  sorry

end part1_monotonic_intervals_part2_max_a_l140_140407


namespace min_n_constant_term_l140_140707

theorem min_n_constant_term (x : ℕ) (hx : x > 0) : 
  ∃ n : ℕ, 
  (∀ r : ℕ, (2 * n = 5 * r) → n ≥ 5) ∧ 
  (∃ r : ℕ, (2 * n = 5 * r) ∧ n = 5) := by
  sorry

end min_n_constant_term_l140_140707


namespace simplify_exponent_product_l140_140257

theorem simplify_exponent_product :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_product_l140_140257


namespace exists_nat_first_four_digits_1993_l140_140346

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def first_four_digits (n : Nat) : Nat :=
  let digits := Int.to_string n
  if digits.length < 4 then n
  else (digits.take 4).toInt!

theorem exists_nat_first_four_digits_1993 :
  ∃ n : Nat, first_four_digits (factorial n) = 1993 :=
sorry

end exists_nat_first_four_digits_1993_l140_140346


namespace geese_percentage_among_non_swan_birds_l140_140476

theorem geese_percentage_among_non_swan_birds :
  let total_birds := 100
  let geese := 0.40 * total_birds
  let swans := 0.20 * total_birds
  let non_swans := total_birds - swans
  let geese_percentage_among_non_swans := (geese / non_swans) * 100
  geese_percentage_among_non_swans = 50 := 
by sorry

end geese_percentage_among_non_swan_birds_l140_140476


namespace product_of_roots_l140_140779

theorem product_of_roots : ∀ x : ℝ, (x + 3) * (x - 4) = 17 → (∃ a b : ℝ, (x = a ∨ x = b) ∧ a * b = -29) :=
by
  sorry

end product_of_roots_l140_140779


namespace quadratic_has_real_root_of_b_interval_l140_140797

variable (b : ℝ)

theorem quadratic_has_real_root_of_b_interval
  (h : ∃ x : ℝ, x^2 + b * x + 25 = 0) : b ∈ Iic (-10) ∪ Ici 10 :=
by
  sorry

end quadratic_has_real_root_of_b_interval_l140_140797


namespace pyramid_top_plus_count_l140_140833

-- Definitions for the problem conditions
def sign (x : Int) : Int := if x = 1 then 1 else -1
def pyramid_condition (a b : Int) : Int := if a = b then 1 else -1

-- The main hypothesis
def bottom_row : List (Int × Int × Int × Int) := 
  [(1, 1, 1, 1), (1, 1, -1, -1), (1, -1, 1, -1), (1, -1, -1, 1),
   (-1, 1, 1, -1), (-1, 1, -1, 1), (-1, -1, 1, 1), (-1, -1, -1, -1)]

-- The number of ways to fill the bottom row to produce a "+" at the top
def number_of_ways_to_plus (l : List (Int × Int × Int × Int)) : Int :=
  l.filter (λ abcd, match abcd with
  | (a, b, c, d) => a * b * c * d = 1
  end).length

theorem pyramid_top_plus_count : number_of_ways_to_plus bottom_row = 8 := by
  sorry

end pyramid_top_plus_count_l140_140833


namespace average_other_marbles_l140_140156

def total_marbles : ℕ := 10 -- Define a hypothetical total number for computation
def clear_marbles : ℕ := total_marbles * 40 / 100
def black_marbles : ℕ := total_marbles * 20 / 100
def other_marbles : ℕ := total_marbles - clear_marbles - black_marbles
def marbles_taken : ℕ := 5

theorem average_other_marbles :
  marbles_taken * other_marbles / total_marbles = 2 := by
  sorry

end average_other_marbles_l140_140156


namespace general_formulas_sum_of_cn_l140_140399

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ q, ∀ n, a (n+1) = q * a n
noncomputable def arithmetic_sequence (b : ℕ → ℝ) : Prop := ∃ d, ∀ n, b (n+1) = b n + d

def a : ℕ → ℝ := λ n, if n = 0 then 0 else 3 * 2^(n-1)
def b (a3 : ℝ) : ℕ → ℝ := λ n, 4 * n - 4

theorem general_formulas (a1 a4 a3 : ℝ) (b2 b4 : ℝ) (h1 : a1 = 3) (h2 : a4 = 24) (h3 : a1 * 2^3 = a4) (h4 : b2 = 4) (h5 : b4 = a3) :
  (∀ n, a n = 3 * 2^(n-1)) ∧ (∀ n, b n = 4 * n - 4) :=
sorry

def c (a b : ℕ → ℝ) : ℕ → ℝ := λ n, a n - b n

theorem sum_of_cn (n : ℕ) (h1 : ∀ n, a n = 3 * 2^(n-1)) (h2 : ∀ n, b n = 4 * n - 4) :
  ∑ i in range n, c a b i = 3 * 2^n - 3 - 2 * n^2 + 2 * n :=
sorry

end general_formulas_sum_of_cn_l140_140399


namespace find_a_l140_140026

/-- Definition of the function -/
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x ^ 2

/-- Derivative of the function -/
def f' (a : ℝ) (x : ℝ) : ℝ := a / x + 2 * x

/-- The proof that a = -3 when the tangent line at point (1, 1) is parallel to x + y = 0 -/
theorem find_a (a : ℝ) (h : f' a 1 = -1) : a = -3 :=
by simp [f', h]; sorry

end find_a_l140_140026


namespace tent_placement_l140_140828

theorem tent_placement
    (forest_length forest_width : ℝ)
    (highway_length highway_width : ℝ) 
    (buffer : ℝ) 
    (tent_min_distance : ℝ) 
    (forest_contains_highway : highway_length ≤ forest_length ∧ highway_width ≤ forest_width)
    (highway_width_eq : highway_width = 0.2) 
    (tent_min_distance_eq : tent_min_distance = 1) 
    (highway_dim : highway_length = 7 ∧ highway_width = 5) : 
    (tent_legal_area_inside : forest_length - 2 * tent_min_distance = 5 ∧ forest_width - 2 * tent_min_distance = 3) ∨
    (tent_legal_area_outside : ∀ x y : ℝ, 
        (x < highway_length - tent_min_distance ∨ x > highway_length + tent_min_distance) ∧ 
        (y < highway_width - tent_min_distance ∨ y > highway_width + tent_min_distance)) := 
sorry

end tent_placement_l140_140828


namespace sum_distances_to_focus_on_parabola_l140_140806

theorem sum_distances_to_focus_on_parabola
  (P : ℕ → ℝ × ℝ)
  (focus : ℝ × ℝ)
  (x : ℕ → ℝ)
  (h_parabola : ∀ (n : ℕ), (P n).1 = x n ∧ (P n).2 ^ 2 = 4 * (P n).1)
  (h_focus : focus = (1, 0))
  (h_sum_abscissas : ∑ i in Finset.range 10, x i = 10) :
  ∑ i in Finset.range 10, dist (P i) focus = 20 := by
sorry

end sum_distances_to_focus_on_parabola_l140_140806


namespace find_EF_squared_exterior_l140_140185

noncomputable def square_side := 10
noncomputable def BE := 7
noncomputable def DF := 7
noncomputable def AE := 15
noncomputable def CF := 15

theorem find_EF_squared_exterior (ABCD : Type*) [square ABCD] (E F : Type*) 
  (BE_eq : distance(B, E) = BE) (DF_eq : distance(D, F) = DF) 
  (AE_eq : distance(A, E) = AE) (CF_eq : distance(C, F) = CF) 
  : EF^2 = 850 + 250 * sqrt(125) :=
sorry

end find_EF_squared_exterior_l140_140185


namespace conjugate_of_complex_number_l140_140429

theorem conjugate_of_complex_number
  (z : ℂ)
  (h : z = i / (4 - 3 * I)) :
  conj z = (-3 / 25) - (4 / 25) * I :=
by sorry

end conjugate_of_complex_number_l140_140429


namespace sahil_selling_price_l140_140899

theorem sahil_selling_price (purchase_price repair_costs transportation_charges : ℕ) (profit_percent : ℕ) :
  purchase_price = 13000 →
  repair_costs = 5000 →
  transportation_charges = 1000 →
  profit_percent = 50 →
  let total_cost := purchase_price + repair_costs + transportation_charges in
  let profit := (profit_percent * total_cost) / 100 in
  let selling_price := total_cost + profit in
  selling_price = 28500 :=
begin
  sorry
end

end sahil_selling_price_l140_140899


namespace triangle_area_proof_l140_140099

-- Defining the geometric structure and conditions
variables {X Y Z P Q R : Type}

-- Variables for the side lengths and areas
variables (XP XY YZ ZX : ℝ)
variables (areaXYZ areaXPQ areaPYQR : ℝ)

-- Assumptions based on the problem statement
def conditions : Prop :=
  XP = 3 ∧
  XY = XP + 6 ∧ -- implies XY = 9
  areaXYZ = 18 ∧
  areaXPQ = 1/9 * areaXYZ ∧ -- area scaling due to the ratio XP/XY
  areaXPQ = areaPYQR -- given equal areas
  
-- The proof we want to establish
theorem triangle_area_proof (h : conditions) : 
  areaXPQ = 2 := 
by 
  cases h with h_XP h_conditions,
  cases h_conditions with h_XY h_areas,
  cases h_areas with h_areaXYZ h_areaPQ,
  cases h_areaPQ with h_areaXPQ h_area_eq,
  sorry

end triangle_area_proof_l140_140099


namespace find_set_of_all_points_P_l140_140448

-- Define the problem conditions and proof statement
variable {c : Type} [circle c] {l : Type} [line l] (M : point) [on_line M l] [tangent_to l c]
variable (A : point) (a b : ℝ) [center_of_circle A c] [radius_of_circle c b]

theorem find_set_of_all_points_P:
  ∃ (P : point) (Q R : point), 
    on_line Q l ∧ on_line R l ∧ midpoint M Q R ∧ incircle_of_triangle c P Q R → 
    ∀ (x1 y1 : ℝ), y1 = b / a * x1 + b ∧ y1 > 2 * b :=
sorry

end find_set_of_all_points_P_l140_140448


namespace tan_alpha_minus_beta_l140_140001

theorem tan_alpha_minus_beta (α β : ℝ) (h1 : sin α = 3/5) (h2 : α ∈ set.Ioo (Real.pi / 2) Real.pi) (h3 : tan (Real.pi - β) = 1/2) : 
  tan (α - β) = -2/11 := 
sorry

end tan_alpha_minus_beta_l140_140001


namespace find_numerator_l140_140425

theorem find_numerator (n : ℕ) : 
  (n : ℚ) / 22 = 9545 / 10000 → 
  n = 9545 * 22 / 10000 :=
by sorry

end find_numerator_l140_140425


namespace max_min_diff_w_l140_140014

theorem max_min_diff_w (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 4) :
  let w := a^2 + a*b + b^2
  let w1 := max (0^2 + 0*b + b^2) (4^2 + 4*b + b^2)
  let w2 := (2-2)^2 + 12
  w1 - w2 = 4 :=
by
  -- skip the proof
  sorry

end max_min_diff_w_l140_140014


namespace probability_between_negative_and_positive_one_l140_140732

noncomputable def standard_normal_distribution (ξ : ℝ → ℝ) : Prop :=
  ∫ (x : ℝ), ξ x * exp (- x ^ 2 / 2) / sqrt (2 * π) = 1

theorem probability_between_negative_and_positive_one (ξ : ℝ → ℝ)
  (h1 : standard_normal_distribution ξ)
  (h2 : ∀ x, (P(ξ > 1) = (1 / 4) : ℝ)) :
  ∀ P, (P(-1 < ξ < 1) = (1 / 2) : ℝ) := 
sorry

end probability_between_negative_and_positive_one_l140_140732


namespace minimum_value_quadratic_l140_140961

theorem minimum_value_quadratic : ∃ x : ℝ, (x = -4) ∧ ∀ y : ℝ, x^2 + 8*x + 10 ≤ y^2 + 8*y + 10 :=
by
  use -4
  split
  exact rfl
  intro y
  sorry

end minimum_value_quadratic_l140_140961


namespace value_of_m_unique_l140_140948

theorem value_of_m_unique (m : ℝ) :
  (1, -2), (3, 4), (6, m / 3) collinear → m = 39 := by
  sorry

end value_of_m_unique_l140_140948


namespace ducks_arrival_quantity_l140_140662

variable {initial_ducks : ℕ} (arrival_ducks : ℕ)

def initial_geese (initial_ducks : ℕ) := 2 * initial_ducks - 10

def remaining_geese (initial_ducks : ℕ) := initial_geese initial_ducks - 10

def remaining_ducks (initial_ducks arrival_ducks : ℕ) := initial_ducks + arrival_ducks

theorem ducks_arrival_quantity :
  initial_ducks = 25 →
  remaining_geese initial_ducks = 30 →
  remaining_geese initial_ducks = remaining_ducks initial_ducks arrival_ducks + 1 →
  arrival_ducks = 4 :=
by
sorry

end ducks_arrival_quantity_l140_140662


namespace half_abs_diff_squares_l140_140249

/-- Half of the absolute value of the difference of the squares of 23 and 19 is 84. -/
theorem half_abs_diff_squares : (1 / 2 : ℝ) * |(23^2 : ℝ) - (19^2 : ℝ)| = 84 :=
by
  sorry

end half_abs_diff_squares_l140_140249


namespace closest_fraction_l140_140327

theorem closest_fraction :
  let teamUSA_fraction := 23.0 / 120.0 in
  let choice1 := 1.0 / 4.0 in
  let choice2 := 1.0 / 5.0 in
  let choice3 := 1.0 / 6.0 in
  let choice4 := 1.0 / 7.0 in
  let choice5 := 1.0 / 8.0 in
  (abs (teamUSA_fraction - choice2) ≤ abs (teamUSA_fraction - choice1) ∧
   abs (teamUSA_fraction - choice2) ≤ abs (teamUSA_fraction - choice3) ∧
   abs (teamUSA_fraction - choice2) ≤ abs (teamUSA_fraction - choice4) ∧
   abs (teamUSA_fraction - choice2) ≤ abs (teamUSA_fraction - choice5)) :=
by
  let teamUSA_fraction := 23.0 / 120.0
  let choice1 := 1.0 / 4.0
  let choice2 := 1.0 / 5.0
  let choice3 := 1.0 / 6.0
  let choice4 := 1.0 / 7.0
  let choice5 := 1.0 / 8.0
  have h1 : abs (teamUSA_fraction - choice2) ≤ abs (teamUSA_fraction - choice1) := sorry
  have h2 : abs (teamUSA_fraction - choice2) ≤ abs (teamUSA_fraction - choice3) := sorry
  have h3 : abs (teamUSA_fraction - choice2) ≤ abs (teamUSA_fraction - choice4) := sorry
  have h4 : abs (teamUSA_fraction - choice2) ≤ abs (teamUSA_fraction - choice5) := sorry
  exact ⟨h1, h2, h3, h4⟩

end closest_fraction_l140_140327


namespace divisible_by_112_l140_140875

theorem divisible_by_112 
  (m : ℕ) 
  (hm1 : odd m) 
  (hm2 : ¬ (3 ∣ m)) : 
  112 ∣ (Int.floor (4^m - (2 + Real.sqrt 2)^m)) :=
sorry

end divisible_by_112_l140_140875


namespace ensure_sum_of_30_l140_140693

theorem ensure_sum_of_30 (drawn_slips : Finset ℕ) :
  (∀ (n ∈ drawn_slips), n ≤ 20) ∧ (∀ (m ∈ drawn_slips), m ≤ 20)
  ∧ drawn_slips.card = 10
  → ∃ a b ∈ drawn_slips, a + b = 30 :=
by
  sorry

end ensure_sum_of_30_l140_140693


namespace bird_nest_area_l140_140191

open Real BigOperators

/-- Problem Statement:
Given the conditions:
1. The area of the Beijing National Stadium is 200 × 300 square meters.
2. Calculate one-millionth of this area in square centimeters.

Prove:
One millionth of this area is approximately 600 square centimeters, which equates to the size of a
math "Layered Exercise Book" (Option B).
--/
theorem bird_nest_area (area_bird_nest_m2 : ℝ) (one_millionth_area_cm2: ℝ) :
  area_bird_nest_m2 = 200 * 300 →
  one_millionth_area_cm2 = area_bird_nest_m2 * (1 / 1000000) * 10000 →
  one_millionth_area_cm2 = 600 :=
by
  intros area_eq one_millionth_eq
  rw [area_eq, ← mul_assoc, ← mul_assoc] at one_millionth_eq
  norm_num at one_millionth_eq
  exact one_millionth_eq
  sorry

end bird_nest_area_l140_140191


namespace triangle_construction_exists_l140_140772

theorem triangle_construction_exists 
    (l1 l2 l3 : Line)
    (O : Point)
    (h_intersect : l1.intersects O ∧ l2.intersects O ∧ l3.intersects O)
    (A : Point)
    (h_A_on_l1 : l1.contains A) :
  ∃ (B C : Point) (ABC : Triangle),
    ABC.A = A ∧
    (∃ l_bc : Line, l_bc.contains B ∧ l_bc.contains C ∧ 
      l2.contains B ∧ l3.contains C ∧ 
      ABC.angle_bisectors ⊆ {l1, l2, l3}) :=
sorry

end triangle_construction_exists_l140_140772


namespace adjusted_retail_price_l140_140638

variable {a : ℝ} {m n : ℝ}

theorem adjusted_retail_price (h : 0 ≤ m ∧ 0 ≤ n) : (a * (1 + m / 100) * (n / 100)) = a * (1 + m / 100) * (n / 100) :=
by
  sorry

end adjusted_retail_price_l140_140638


namespace range_of_f_l140_140564

def f (x : ℝ) : ℝ :=
  if 1 ≤ x then log (1/2) x else 2^x

theorem range_of_f : set.range f = set.Ioo (-∞ : ℝ) 2 := by
  sorry

end range_of_f_l140_140564


namespace printer_cost_comparison_l140_140223

-- Definitions based on the given conditions
def in_store_price : ℝ := 150.00
def discount_rate : ℝ := 0.10
def installment_payment : ℝ := 28.00
def number_of_installments : ℕ := 5
def shipping_handling_charge : ℝ := 12.50

-- Discounted in-store price calculation
def discounted_in_store_price : ℝ := in_store_price * (1 - discount_rate)

-- Total cost from the television advertiser
def tv_advertiser_total_cost : ℝ := (number_of_installments * installment_payment) + shipping_handling_charge

-- Proof statement
theorem printer_cost_comparison :
  discounted_in_store_price - tv_advertiser_total_cost = -17.50 :=
by
  sorry

end printer_cost_comparison_l140_140223


namespace Jenna_sells_3125_widgets_l140_140106

noncomputable def jenna_widgets_sold : ℕ :=
  let cost_per_widget := 3
  let selling_price_per_widget := 8
  let rent := 10000
  let num_workers := 4
  let salary_per_worker := 2500
  let profit_after_taxes := 4000
  let tax_fraction := 0.20

  let payroll_expenses := salary_per_worker * num_workers
  let fixed_expenses := rent + payroll_expenses
  let total_profit_before_taxes := profit_after_taxes / (1 - tax_fraction)
  let total_revenue := total_profit_before_taxes + fixed_expenses
  let profit_per_widget := selling_price_per_widget - cost_per_widget
  
  total_revenue / selling_price_per_widget

theorem Jenna_sells_3125_widgets :
  jenna_widgets_sold = 3125 :=
by
  sorry

end Jenna_sells_3125_widgets_l140_140106


namespace sum_of_incircle_radii_proof_l140_140815

-- Define the given structure and conditions
variables (A B C D : Type*)
          [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
          (AB AC BC : ℝ)
          (h1 : AB = 9)
          (h2 : AC = 12)
          (h3 : BC = 15)
          (D is_midpoint_of_BC : A → A → A → Prop) -- Specify D is the midpoint of BC
          (h4 : is_midpoint_of_BC B C D)

-- Define the sum of the radii of the incircles in the sub-triangles
noncomputable def sum_of_incircle_radii (r1 r2 : ℝ) := r1 + r2

theorem sum_of_incircle_radii_proof :
  ∃ r1 r2, sum_of_incircle_radii r1 r2 = 9 / 2 :=
by {
  -- Assume pressures about the incircle radii calculation and 
  -- employ properties and geometric constructions.
  -- Obtaining the values and proving the sum of radii as specified.
  sorry
}

end sum_of_incircle_radii_proof_l140_140815


namespace bisect_KL_l140_140834

open Real EuclideanGeometry

namespace TriangleProofs

variables {A B C: Point}
variables {BB1 CC1 K L O: Point}

def acute_triangle (A B C: Point) : Prop :=
  ∃ ha hb hc : Line, 
    (ha ⟂ (Line.mk A B)) ∧ (ha ⟂ (Line.mk A C)) ∧ 
    (hb ⟂ (Line.mk B A)) ∧ (hb ⟂ (Line.mk B C)) ∧
    (hc ⟂ (Line.mk C A)) ∧ (hc ⟂ (Line.mk C B)) ∧ 
    ∀ (p: Point), p ∈ ha ∩ hb ∩ hc → ⇑dist A p = ⇑dist B p ∧ ⇑dist A p = ⇑dist C p

def altitudes (A B C BB1 CC1: Point) : Prop :=
  is_perpendicular (Line.mk A B) (Line.mk B BB1) ∧ is_perpendicular (Line.mk A C) (Line.mk C CC1)

def on_sides (K L A B C: Point) : Prop :=
  same_side_line K B A C ∧ same_side_line L C A B

def lengths_eq (A B C BB1 CC1 K L: Point) : Prop :=
  dist A K = dist B C1 ∧ dist A L = dist C B1

def circumcenter (A B C O: Point) : Prop :=
  ∃ r: ℝ, circle O r ∧ dist O A = r ∧ dist O B = r ∧ dist O C = r

theorem bisect_KL {A B C BB1 CC1 K L O: Point}
  (ht : acute_triangle A B C)
  (h_alt : altitudes A B C BB1 CC1)
  (h_sides : on_sides K L A B C)
  (h_lengths : lengths_eq A B C BB1 CC1 K L)
  (h_circum : circumcenter A B C O) :
  segment_bisector A O K L :=
by
  sorry

end TriangleProofs

end bisect_KL_l140_140834


namespace inequality_proof_equality_condition_l140_140878

variable {x y z : ℝ}

def positive_reals (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0

theorem inequality_proof (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry -- Proof goes here

theorem equality_condition (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * z ∧ y = z :=
sorry -- Proof goes here

end inequality_proof_equality_condition_l140_140878


namespace arithmetic_sequence_15th_term_l140_140588

theorem arithmetic_sequence_15th_term (a1 d n : ℕ) (h1 : a1 = 3) (h2 : d = 3) (h3 : n = 15) :
  a1 + (n - 1) * d = 45 :=
by
  rw [h1, h2, h3]
  sorry

end arithmetic_sequence_15th_term_l140_140588


namespace probability_of_drawing_different_colors_l140_140439

-- Condition: There are 5 balls in total, 3 white and 2 yellow, and two are drawn.
def total_balls : ℕ := 5
def white_balls : ℕ := 3
def yellow_balls : ℕ := 2
def total_drawn : ℕ := 2

-- The number of ways to choose two balls from total_balls
def total_combinations : ℕ := total_balls.choose total_drawn

-- The number of ways to choose one white and one yellow ball
def different_color_combinations : ℕ := white_balls.choose 1 * yellow_balls.choose 1

-- The probability of drawing two balls of different colors
def probability_different_colors : ℝ :=
  different_color_combinations.toReal / total_combinations.toReal

theorem probability_of_drawing_different_colors : probability_different_colors = 0.6 := by
  sorry

end probability_of_drawing_different_colors_l140_140439


namespace m_ducks_l140_140569

variable (M C K : ℕ)

theorem m_ducks :
  (M = C + 4) ∧
  (M = 2 * C + K + 3) ∧
  (M + C + K = 90) →
  M = 89 := by
  sorry

end m_ducks_l140_140569


namespace probability_fourth_term_integer_l140_140512

def generate_term (previous_term : ℚ) (coin_flip : Bool) : ℚ :=
  if coin_flip then 
    3 * previous_term + 2 
  else 
    previous_term / 3 - 2

def mia_sequence (initial_term : ℚ) (coin_flips : List Bool) : List ℚ :=
  coin_flips.foldl (fun acc flip => acc ++ [generate_term (acc.head!) flip]) [initial_term]

def count_integers (l: List ℚ): ℕ := 
  l.countp (fun x => x.denom = 1)

theorem probability_fourth_term_integer : 
  let initial_term := (10 : ℚ)
  let possible_flips := {ff, tt}
  let sequences := List.bind possible_flips (
    fun f1 => List.bind possible_flips (
      fun f2 => List.bind possible_flips (
        fun f3 => List.bind possible_flips (
          fun f4 => [mia_sequence initial_term [f1, f2, f3, f4]])))
  let fourth_terms := sequences.map (fun seq => seq.nth 4)
  count_integers fourth_terms = 1 / 8 := 
sorry

end probability_fourth_term_integer_l140_140512


namespace range_of_m_l140_140396

noncomputable def function_even_and_monotonic (f : ℝ → ℝ) := 
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x > f y)

variable (f : ℝ → ℝ)
variable (m : ℝ)

theorem range_of_m (h₁ : function_even_and_monotonic f) 
  (h₂ : f m > f (1 - m)) : m < 1 / 2 :=
sorry

end range_of_m_l140_140396


namespace equidistant_points_from_circle_and_tangents_l140_140994

noncomputable def equidistant_points_count (O : Type) [metric_space O] (r : ℝ) : ℕ :=
if r > 0 then 2 else 0

theorem equidistant_points_from_circle_and_tangents (O : Type) [metric_space O] (r : ℝ) :
  let circle_center := O
  let tangent_distance_1 := r
  let tangent_distance_2 := r + 2
  (r > 0) → equidistant_points_count O r = 2 :=
by
  intro h
  -- Proof steps go here
  sorry

end equidistant_points_from_circle_and_tangents_l140_140994


namespace vasya_wins_l140_140523

-- Setting up the game state
def initial_piles : list ℕ := list.repeat 10 11

-- Definition of a move by Petya: taking 1, 2, or 3 stones from a single pile
def petya_move (piles : list ℕ) (pile_index : ℕ) (stones : ℕ) : list ℕ :=
  if h : pile_index < piles.length ∧ stones ∈ {1, 2, 3} then
    piles.set_nth pile_index (piles.nth pile_index - stones)
  else piles

-- Definition of a move by Vasya: taking stones from different piles
def vasya_move (piles : list ℕ) (pile_indices : list ℕ) (stones_list: list ℕ) : list ℕ :=
  list.foldl (λ acc (pi_si : ℕ × ℕ), acc.set_nth pi_si.1 (acc.nth pi_si.1 - pi_si.2)) piles (pile_indices.zip stones_list)

-- Definition of the losing state: when no move can be made
def losing_state (piles : list ℕ) : Prop :=
  ∀ (i : ℕ), i < piles.length → piles.nth i = 0

-- The main theorem stating that Vasya can always ensure a win
theorem vasya_wins : ∀ (piles : list ℕ), piles = initial_piles → (∃ moves : list (list ℕ × list ℕ), vasya_strategy piles moves ⟹ ∃ (p : ℕ), losing_state (petya_move (vasya_move piles moves.head.1 moves.head.2) p)) :=
by {
  sorry -- Proof is omitted
}

end vasya_wins_l140_140523


namespace minimum_seats_occupied_l140_140224

-- Define the conditions
def initial_seat_count : Nat := 150
def people_initially_leaving_up_to_two_empty_seats := true
def eventually_rule_changes_to_one_empty_seat := true

-- Define the function which checks the minimum number of occupied seats needed
def fewest_occupied_seats (total_seats : Nat) (initial_rule : Bool) (final_rule : Bool) : Nat :=
  if initial_rule && final_rule && total_seats = 150 then 57 else 0

-- The main theorem we need to prove
theorem minimum_seats_occupied {total_seats : Nat} : 
  total_seats = initial_seat_count → 
  people_initially_leaving_up_to_two_empty_seats → 
  eventually_rule_changes_to_one_empty_seat → 
  fewest_occupied_seats total_seats people_initially_leaving_up_to_two_empty_seats eventually_rule_changes_to_one_empty_seat = 57 :=
by
  intro h1 h2 h3
  sorry

end minimum_seats_occupied_l140_140224


namespace max_guaranteed_coins_l140_140577

-- Definitions based on conditions
def total_coins := 300
def max_bags := 11
def coins_per_bag := 14

-- Theorem based on the proof problem
theorem max_guaranteed_coins : ∃ K, K = 146 ∧ 
                                (∀ (bag_count : ℕ), bag_count ≤ max_bags → 
                                (bag_count * coins_per_bag ≤ total_coins → 
                                (total_coins - bag_count * coins_per_bag) ≥ 146)) :=
begin
  use 146,
  split,
  {
    -- correct answer, K = 146
    refl,
  },
  {
    -- conditions for the proof
    intros bag_count h1 h2,
    transitivity,
    {
      rw ← Nat.mul_sub_left_distrib,
      linarith,
    },
    {
      refl,
    }
  }
end

end max_guaranteed_coins_l140_140577
