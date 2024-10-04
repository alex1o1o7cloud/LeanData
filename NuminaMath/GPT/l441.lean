import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Set
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Integral
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace Mack_journal_pages_total_l441_441269

theorem Mack_journal_pages_total (
  (T1 : ℕ) (R1 : ℕ) (T2 : ℕ) (R2 : ℕ) (P3 : ℕ) (T4 : ℕ) (T5 : ℕ) (R3 : ℕ) (R4 : ℕ)
  (h1 : T1 = 60) (h2 : R1 = 30) (h3 : T2 = 45) (h4 : R2 = 15) (h5 : P3 = 5)
  (h6 : T4 = 90) (h7 : T5 = 30) (h8 : R3 = 10) (h9 : R4 = 20)
) : (T1 / R1 + T2 / R2 + P3 + (T5 / R3 + (T4 - T5) / R4) = 16) :=
by {
  rw [h1, h2, h3, h4, h5, h6, h7, h8, h9],
  norm_num,
  sorry
}

end Mack_journal_pages_total_l441_441269


namespace isosceles_triangle_l441_441760

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441760


namespace dalmatians_with_right_ear_spots_l441_441709

def TotalDalmatians := 101
def LeftOnlySpots := 29
def RightOnlySpots := 17
def NoEarSpots := 22

theorem dalmatians_with_right_ear_spots : 
  (TotalDalmatians - LeftOnlySpots - NoEarSpots) = 50 :=
by
  -- Proof goes here, but for now, we use sorry
  sorry

end dalmatians_with_right_ear_spots_l441_441709


namespace term_61_is_201_l441_441225

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (a5 : ℤ)

-- Define the general formula for the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ :=
  a5 + (n - 5) * d

-- Given variables and conditions:
axiom h1 : a5 = 33
axiom h2 : d = 3

theorem term_61_is_201 :
  arithmetic_sequence a5 d 61 = 201 :=
by
  -- proof here
  sorry

end term_61_is_201_l441_441225


namespace square_equal_triangle_apothem_apothem_l441_441345

theorem square_equal_triangle_apothem_apothem :
  ∀ s t : ℝ, 
  s^2 = 4 * s → 
  (s / 2) = 2 → 
  (t * sqrt 3) / 6 = 2 → 
  (s / 2) = (t * sqrt 3) / 6 :=
begin
  intros s t hs hs_apothem ht,
  sorry
end

end square_equal_triangle_apothem_apothem_l441_441345


namespace CP_perpendicular_AB_l441_441641

variable {α : Type*} [EuclideanGeometry α]
variables {A B C L M N P : α}

-- Hypotheses (Conditions)
hypothesis (Hacute : ∀ {X Y Z : α}, Triangle X Y Z → AcuteTriangle X Y Z) :
hypothesis (Hbisector : angleBisector C A B C L) :
hypothesis (Hperpendicular_CM : ⊥ CM AC) :
hypothesis (Hperpendicular_CN : ⊥ CN BC) :
hypothesis (Hintersection_AN_BM : intersection (line A N) (line B M) = P) :

-- Goal (Question)
theorem CP_perpendicular_AB : Perpendicular CP AB :=
sorry

end CP_perpendicular_AB_l441_441641


namespace conic_section_is_ellipse_l441_441520

theorem conic_section_is_ellipse :
  (∃ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0) ∧
  ∀ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0 →
    ((x - 2)^2 / (20 / 3) + (y - 2)^2 / 20 = 1) :=
sorry

end conic_section_is_ellipse_l441_441520


namespace probability_15th_roll_last_is_approximately_l441_441622

noncomputable def probability_15th_roll_last : ℝ :=
  (7 / 8) ^ 13 * (1 / 8)

theorem probability_15th_roll_last_is_approximately :
  abs (probability_15th_roll_last - 0.022) < 0.001 :=
by sorry

end probability_15th_roll_last_is_approximately_l441_441622


namespace last_trip_l441_441316

def initial_order : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

def boatCapacity : Nat := 4  -- Including Snow White

def adjacentPairsQuarrel (adjPairs : List (String × String)) : Prop :=
  ∀ (d1 d2 : String), (d1, d2) ∈ adjPairs → (d2, d1) ∈ adjPairs → False

def canRow (person : String) : Prop := person = "Snow White"

noncomputable def final_trip (remainingDwarfs : List String) (allTrips : List (List String)) : List String := ["Grumpy", "Bashful", "Doc"]

theorem last_trip (adjPairs : List (String × String))
  (h_adj : adjacentPairsQuarrel adjPairs)
  (h_row : canRow "Snow White")
  (dwarfs_order : List String = initial_order)
  (without_quarrels : ∀ trip : List String, trip ∈ allTrips → 
    ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → (d1, d2) ∈ adjPairs → 
    ("Snow White" ∈ trip) → True) :
  final_trip ["Grumpy", "Bashful", "Doc"] allTrips = ["Grumpy", "Bashful", "Doc"] :=
sorry

end last_trip_l441_441316


namespace minimum_value_pm_pn_is_sqrt10_plus_1_l441_441087

open EuclideanGeometry
open Real

noncomputable def point := (ℝ × ℝ × ℝ)

def distance (p1 p2 : point) : ℝ := 
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2 |> sqrt

def point_on_segment (p1 p2: point) (t: ℝ) : point := 
  ((1 - t) * p1.1 + t * p2.1, (1 - t) * p1.2 + t * p2.2, (1 - t) * p1.3 + t * p2.3)

def minimum_value_pm_pn : ℝ :=
  let O := (2, 0, 2)
  let B := (4, 0, 0)
  let B₁ := (4, 0, 4)
  let C := (4, 4, 0)
  let C₁ := (4, 4, 4)
  let E := (4, 2, 4)
  let F := (4, 4, 2)
  let P (t: ℝ) := point_on_segment O B t
  let N (t: ℝ) := point_on_segment E F t
  let M (x y z: ℝ) := (x, y, z)
  let dist_pigen_sum (t₁ t₂ t₃ x y z: ℝ) := 
    distance (P t₁) (M x y z) + distance (P t₁) (N t₂)
  ((range (λ x, dist_pigen_sum x 0.0 0.0 0.0 0.0 0.0)).inf)

theorem minimum_value_pm_pn_is_sqrt10_plus_1 :
  minimum_value_pm_pn = sqrt 10 + 1 :=
sorry

end minimum_value_pm_pn_is_sqrt10_plus_1_l441_441087


namespace weight_of_each_bag_of_flour_l441_441389

theorem weight_of_each_bag_of_flour
  (flour_weight_needed : ℕ)
  (cost_per_bag : ℕ)
  (salt_weight_needed : ℕ)
  (salt_cost_per_pound : ℚ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_made : ℕ)
  (profit : ℕ)
  (total_flour_cost : ℕ)
  (num_bags : ℕ)
  (weight_per_bag : ℕ)
  (calc_salt_cost : ℚ := salt_weight_needed * salt_cost_per_pound)
  (calc_total_earnings : ℕ := tickets_sold * ticket_price)
  (calc_total_cost : ℚ := calc_total_earnings - profit)
  (calc_flour_cost : ℚ := calc_total_cost - calc_salt_cost - promotion_cost)
  (calc_num_bags : ℚ := calc_flour_cost / cost_per_bag)
  (calc_weight_per_bag : ℚ := flour_weight_needed / calc_num_bags) :
  flour_weight_needed = 500 ∧
  cost_per_bag = 20 ∧
  salt_weight_needed = 10 ∧
  salt_cost_per_pound = 0.2 ∧
  promotion_cost = 1000 ∧
  ticket_price = 20 ∧
  tickets_sold = 500 ∧
  total_made = 8798 ∧
  profit = 10000 - total_made ∧
  calc_salt_cost = 2 ∧
  calc_total_earnings = 10000 ∧
  calc_total_cost = 1202 ∧
  calc_flour_cost = 200 ∧
  calc_num_bags = 10 ∧
  calc_weight_per_bag = 50 :=
by {
  sorry
}

end weight_of_each_bag_of_flour_l441_441389


namespace minimum_value_of_x_l441_441619

theorem minimum_value_of_x {x : ℝ} (h1 : 0 < x) (h2 : log x ≥ log 3 + (2 / 3) * log x) : x ≥ 27 := by
  sorry

end minimum_value_of_x_l441_441619


namespace number_of_younger_employees_correct_l441_441468

noncomputable def total_employees : ℕ := 200
noncomputable def younger_employees : ℕ := 120
noncomputable def sample_size : ℕ := 25

def number_of_younger_employees_to_be_drawn (total younger sample : ℕ) : ℕ :=
  sample * younger / total

theorem number_of_younger_employees_correct :
  number_of_younger_employees_to_be_drawn total_employees younger_employees sample_size = 15 := by
  sorry

end number_of_younger_employees_correct_l441_441468


namespace find_x_minus_y_l441_441692

theorem find_x_minus_y {x y z : ℤ} (h1 : x - (y + z) = 5) (h2 : x - y + z = -1) : x - y = 2 :=
by
  sorry

end find_x_minus_y_l441_441692


namespace correct_props_l441_441171

-- Defining the propositions
def prop1 (L1 L2 : Line) (P1 P2 : Plane) : Prop :=
  (L1.parallel_to_Plane P2 ∧ L2.parallel_to_Plane P2) → P1.parallel P2 

def prop2 (P1 P2 : Plane) (L : Line) : Prop :=
  (L.perpendicular_to P2 ∧ L ∈ P1) → P1.perpendicular P2 

def prop3 (L1 L2 L3 : Line) : Prop :=
  (L1.perpendicular_to L3 ∧ L2.perpendicular_to L3) → L1.parallel L2 

def prop4 (P1 P2 : Plane) (L : Line) : Prop :=
  (P1.perpendicular P2 ∧ L ∈ P1 ∧ ¬ L.perpendicular_to (P1 ∩ P2)) → ¬ L.perpendicular_to P2 

-- Proving the correct propositions among the given ones
theorem correct_props : prop2 ∧ prop4 ∧ ¬ prop1 ∧ ¬ prop3 :=
by 
  -- Here, we would provide the detailed proof steps
  sorry

end correct_props_l441_441171


namespace domain_of_log_function_l441_441352

def domain (f : ℝ → ℝ) := {x : ℝ | ∃ y, f x = y}

theorem domain_of_log_function : domain (λ x : ℝ, log 2 ((1 : ℝ) / (3 * x - 1))) = { x : ℝ | x > 1 / 3 } :=
by
  sorry

end domain_of_log_function_l441_441352


namespace AC_diagonal_length_l441_441216

noncomputable def AC_length (AD DC : ℝ) (angle_ADC : ℝ) : ℝ :=
  Real.sqrt (AD^2 + DC^2 - 2 * AD * DC * Real.cos angle_ADC)

theorem AC_diagonal_length :
  let AD := 15
  let DC := 15
  let angle_ADC := 2 * Real.pi / 3 -- 120 degrees in radians
  AC_length AD DC angle_ADC = 15 :=
by
  have h : AC_length 15 15 (2 * Real.pi / 3) = Real.sqrt (15^2 + 15^2 - 2 * 15 * 15 * Real.cos (2 * Real.pi / 3)),
  { unfold AC_length },
  rw h,
  have h_cos : Real.cos (2 * Real.pi / 3) = -1 / 2,
  { sorry }, -- intermediate steps to find cosine of 120 degrees
  rw [h_cos, sq],
  norm_num,
  refl

end AC_diagonal_length_l441_441216


namespace remaining_number_is_zero_l441_441026

theorem remaining_number_is_zero :
  let n := 1987 in
  let initial_numbers := List.range (n + 1).tail! in
  let final_numbers := [987, 0] in
  (∀ s ∈ initial_numbers.sublists, (s.sum % 7 = 0) -> final_numbers = [987, 0]) → 
  true :=
by
  sorry

end remaining_number_is_zero_l441_441026


namespace points_for_tie_l441_441223

-- Defining the conditions
def num_teams : ℕ := 6
def points_win : ℕ := 3
def points_loss : ℕ := 0
def max_points : ℕ := (num_teams * (num_teams - 1) / 2) * points_win
def min_max_diff : ℕ := 15

-- To be proven: points for a tie (T)
theorem points_for_tie : ℕ :=
  ∃ T : ℕ, max_points - (num_teams * (num_teams - 1) / 2) * T = min_max_diff ∧  T = 2 :=
begin
  sorry
end

end points_for_tie_l441_441223


namespace fruit_basket_l441_441380

-- Define the quantities and their relationships
variables (O A B P : ℕ)

-- State the conditions
def condition1 : Prop := A = O - 2
def condition2 : Prop := B = 3 * A
def condition3 : Prop := P = B / 2
def condition4 : Prop := O + A + B + P = 28

-- State the theorem
theorem fruit_basket (h1 : condition1 O A) (h2 : condition2 A B) (h3 : condition3 B P) (h4 : condition4 O A B P) : O = 6 :=
sorry

end fruit_basket_l441_441380


namespace absolute_value_inequality_l441_441544

theorem absolute_value_inequality (x : ℝ) : 
  (|3 * x + 1| > 2) ↔ (x > 1/3 ∨ x < -1) := by
  sorry

end absolute_value_inequality_l441_441544


namespace final_trip_theorem_l441_441293

/-- Define the lineup of dwarfs -/
inductive Dwarf where
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

open Dwarf

/-- Define the conditions -/
-- The dwarfs initially standing next to each other
def adjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = Happy ∧ d2 = Grumpy) ∨
  (d1 = Grumpy ∧ d2 = Dopey) ∨
  (d1 = Dopey ∧ d2 = Bashful) ∨
  (d1 = Bashful ∧ d2 = Sleepy) ∨
  (d1 = Sleepy ∧ d2 = Doc) ∨
  (d1 = Doc ∧ d2 = Sneezy)

-- Snow White is the only one who can row
constant snowWhite_can_row : Prop := true

-- The boat can hold Snow White and up to 3 dwarfs
constant boat_capacity : ℕ := 4

-- Define quarrel if left without Snow White
def quarrel_without_snowwhite (d1 d2 : Dwarf) : Prop := adjacent d1 d2

-- Define the final trip setup
def final_trip (dwarfs : List Dwarf) : Prop :=
  dwarfs = [Grumpy, Bashful, Doc]

-- Theorem to prove the final trip
theorem final_trip_theorem : ∃ dwarfs, final_trip dwarfs :=
  sorry

end final_trip_theorem_l441_441293


namespace tan_3_75_square_root_expression_sum_abcd_is_13_l441_441525

noncomputable def tan_3_75 (cos_7_5 sin_7_5 : ℝ) (a b c d : ℕ) : ℝ :=
  (1 - cos_7_5) / sin_7_5

theorem tan_3_75_square_root_expression
  (hcos : cos_7_5 = (Real.sqrt 6 + Real.sqrt 2) / 4)
  (hsin : sin_7_5 = (Real.sqrt 6 - Real.sqrt 2) / 4)
  (a b c d : ℕ)
  (ha : a = 6) (hb : b = 2) (hc : c = 3) (hd : d = 2)
  (h_pos : 0 < a) (h_ge_1 : a ≥ b) (h_ge_2 : b ≥ c) (h_ge_3 : c ≥ d) :
  tan_3_75 cos_7_5 sin_7_5 a b c d = Real.sqrt a - Real.sqrt b + Real.sqrt c - d :=
by
  sorry

theorem sum_abcd_is_13 : 6 + 2 + 3 + 2 = 13 :=
by
  rw [←add_assoc, ←add_assoc 6, add_comm 3 2, add_assoc _ 3 2]
  rfl

end tan_3_75_square_root_expression_sum_abcd_is_13_l441_441525


namespace problem_statement_l441_441388

variable {x y z : ℝ}

theorem problem_statement (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
    (h₁ : x^2 - y^2 = y * z) (h₂ : y^2 - z^2 = x * z) : 
    x^2 - z^2 = x * y := 
by
  sorry

end problem_statement_l441_441388


namespace triangle_is_isosceles_l441_441889

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441889


namespace triangle_is_isosceles_l441_441847

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441847


namespace provable_propositions_l441_441716

-- Proposition p: \( \forall x \in \mathbb{N}, x^3 \ge x^2 \)
def prop_p : Prop := ∀ (x : ℕ), x^3 ≥ x^2

-- Proposition q: \( \forall a \in (0,1) \cup (1,+\infty), \log_a(1) = 0 \)
def prop_q : Prop := ∀ (a : ℝ), (0 < a ∧ a < 1 ∨ 1 < a) → Real.log a (1) = 0

theorem provable_propositions : ¬ prop_p ∧ prop_q := by
  sorry

end provable_propositions_l441_441716


namespace triangle_is_isosceles_l441_441807

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441807


namespace triangle_ABC_is_isosceles_l441_441915

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441915


namespace not_perfect_square_m_3_not_perfect_square_m_4_not_perfect_square_m_5_not_perfect_square_m_6_l441_441275

def sum_of_consecutive_squares (n : ℕ) (m : ℕ) : ℕ :=
  Σ k in range m, (n+k)^2

theorem not_perfect_square_m_3 (n : ℕ) : ¬ ∃ (k : ℕ), 3 * n^2 + 6 * n + 5 = k * k := by
  sorry

theorem not_perfect_square_m_4 (n : ℕ) : ¬ ∃ (k : ℕ), 4 * n^2 + 12 * n + 14 = k * k := by
  sorry

theorem not_perfect_square_m_5 (n : ℕ) : ¬ ∃ (k : ℕ), 5 * n^2 + 20 * n + 30 = k * k := by
  sorry

theorem not_perfect_square_m_6 (n : ℕ) : ¬ ∃ (k : ℕ), 6 * n^2 + 30 * n + 55 = k * k := by
  sorry

end not_perfect_square_m_3_not_perfect_square_m_4_not_perfect_square_m_5_not_perfect_square_m_6_l441_441275


namespace afternoon_sales_l441_441479

theorem afternoon_sales (x : ℕ) (H1 : 2 * x + x = 390) : 2 * x = 260 :=
by
  sorry

end afternoon_sales_l441_441479


namespace count_valid_smooth_integers_l441_441502

def is_smooth_integer (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = list.sort digits

def is_divisible_by_six (n : ℕ) : Prop :=
  (n % 6 = 0)

def valid_smooth_integer (n : ℕ) : Prop :=
  10 < n ∧ n < 1000 ∧ is_smooth_integer n ∧ is_divisible_by_six n

theorem count_valid_smooth_integers : 
  (finset.filter valid_smooth_integer (finset.range 1000)).card = 57 := 
sorry

end count_valid_smooth_integers_l441_441502


namespace units_digit_17_pow_2024_l441_441980

theorem units_digit_17_pow_2024 : (17 ^ 2024) % 10 = 1 := 
by
  sorry

end units_digit_17_pow_2024_l441_441980


namespace sequence_sum_identity_l441_441182

theorem sequence_sum_identity 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ)
  (h1 : ∀ n, b_n n - a_n n = 2^n + 1)
  (h2 : ∀ n, S_n n + T_n n = 2^(n+1) + n^2 - 2) : 
  ∀ n, 2 * T_n n = n * (n - 1) :=
by sorry

end sequence_sum_identity_l441_441182


namespace cos_A_l441_441584

noncomputable def vector : Type := unit → ℝ

variables (A B C O : vector) (AO AB AC : vector)

axiom circumcenter_condition : AO = (1/3:ℝ) • (AB + AC)

theorem cos_A (h : circumcenter_condition AO AB AC) : real.cos (60 * (real.pi / 180)) = 1 / 2 :=
by sorry

end cos_A_l441_441584


namespace triangle_isosceles_if_parallel_l441_441874

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441874


namespace find_average_time_l441_441465

noncomputable theory

-- Definition of positions
def car_position (t : ℝ) : ℝ × ℝ := (t, 0)
def storm_position (t : ℝ) : ℝ × ℝ := (t / Real.sqrt 2, 130 - t / Real.sqrt 2)

-- Distance formula between car and storm center
def distance_between (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Condition that car is within storm's radius
def within_storm (t : ℝ) : Prop :=
  distance_between (car_position t) (storm_position t) ≤ 75

-- Main theorem statement
theorem find_average_time :
  ∃ t1 t2 : ℝ,
    within_storm t1 ∧ within_storm t2 ∧ (t1 < t2) ∧ 
    (∀ t, within_storm t → (t1 ≤ t) ∧ (t ≤ t2)) ∧
    (1/2:ℝ) * (t1 + t2) = 260 :=
sorry

end find_average_time_l441_441465


namespace exists_same_color_triangle_l441_441123

theorem exists_same_color_triangle (color : ℝ² → Prop)
  (h : ∀ p : ℝ², color p ∨ ¬ color p) :
  ∃ (a b c : ℝ²), (color a = color b ∧ color b = color c) ∧ 
                   (dist a b = 1 ∧ dist b c = 1 ∧ dist c a = 1 ∨
                    dist a b = √3 ∧ dist b c = √3 ∧ dist c a = √3) :=
by
  sorry

end exists_same_color_triangle_l441_441123


namespace perp_lines_l441_441344

open set

variables {A B C H A' B' X Y : Point}

-- Definitions and conditions from the problem
def is_orthocenter (H : Point) (A B C : Point) := ∃ (A' B' C'), 
  altitude A A' B C ∧ altitude B B' A C ∧ altitude C C' A B ∧ 
  intersection (line A A') (line B B') = H

def midpoint (P Q M : Point) := dist P M = dist Q M ∧ collinear P Q M

-- Lean statement to prove the problem
theorem perp_lines (
  h_orthocenter : is_orthocenter H A B C,
  h_midpointX : midpoint A B X,
  h_midpointY : midpoint C H Y
): are_perpendicular (line X Y) (line A' B') :=
sorry

end perp_lines_l441_441344


namespace largest_fraction_l441_441010

theorem largest_fraction:
  let A := 397 / 101
  let B := 487 / 121
  let C := 596 / 153
  let D := 678 / 173
  let E := 796 / 203
  A < 4 →
  B > 4 →
  C < 4 →
  D < 4 →
  E < 4 →
  ∀ x, x ∈ {A, B, C, D, E} → x ≤ B :=
by
  intros A_lt_4 B_gt_4 C_lt_4 D_lt_4 E_lt_4 x x_in_set
  sorry

end largest_fraction_l441_441010


namespace unique_lcm_function_l441_441279

open Nat

def f : ℕ × ℕ → ℕ := sorry

theorem unique_lcm_function :
  (∀ m n : ℕ, f(m, n) = f(n, m)) ∧
  (∀ n : ℕ, f(n, n) = n) ∧
  (∀ m n : ℕ, n > m → (n - m) * f(m, n) = n * f(m, n - m)) →
  (∀ m n : ℕ, f(m, n) = lcm m n) :=
by
  sorry

end unique_lcm_function_l441_441279


namespace largest_non_sum_217_l441_441398

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end largest_non_sum_217_l441_441398


namespace f_decreasing_ln_less_than_ax_power_less_than_e_l441_441513

-- Define the initial function
def f (x : ℝ) : ℝ := (ln (1 + x)) / x

-- Statement Ⅰ: Prove that f(x) is decreasing for x > 0
theorem f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x :=
sorry

-- Statement Ⅱ: Determine if there exists a real number a such that ln(1 + x) < ax holds for all x > 0
theorem ln_less_than_ax (a : ℝ) : (∀ x > 0, ln (1 + x) < a * x) ↔ a ≥ 1 :=
sorry

-- Statement Ⅲ: Prove that (1 + 1/n)^n < e for all n ∈ ℕ⁺
theorem power_less_than_e (n : ℕ) (hn : 0 < n) : (1 + 1 / (↑n : ℝ)) ^ n < Real.exp 1 :=
sorry

end f_decreasing_ln_less_than_ax_power_less_than_e_l441_441513


namespace count_unique_positive_integers_using_six_matchsticks_l441_441698

def matchsticks (d : Nat) : Nat :=
  match d with
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 3 => 5
  | 4 => 4
  | 5 => 5
  | 6 => 6
  | 7 => 3
  | 8 => 7
  | 9 => 6
  | _ => 0

def digits_with_exact_six_matchsticks : List Nat :=
  List.filter (λ n => (n.digits 10).map matchsticks |>.sum = 6) (List.range 1000)

theorem count_unique_positive_integers_using_six_matchsticks :
  digits_with_exact_six_matchsticks.length = 6 := by
  sorry

end count_unique_positive_integers_using_six_matchsticks_l441_441698


namespace median_of_series_l441_441646

def list_of_integer_occurrences : List ℕ :=
  List.join (List.map (fun n => List.repeat n n) (List.range 151).tail)

def median (l : List ℕ) : ℕ :=
  match l.length % 2 with
  | 0 => (l.nth (l.length / 2) + l.nth ((l.length / 2) - 1)) / 2
  | _ => l.nth (l.length / 2)

theorem median_of_series : median list_of_integer_occurrences = 106 := 
by 
  -- Proof will be provided here
  sorry

end median_of_series_l441_441646


namespace find_divisor_l441_441705

theorem find_divisor (q r : ℤ) : ∃ d : ℤ, 151 = d * q + r ∧ q = 11 ∧ r = -4 → d = 14 :=
by
  intros
  sorry

end find_divisor_l441_441705


namespace remaining_payment_l441_441285

theorem remaining_payment
  (part_payment : ℝ := 400)
  (part_payment_pct : ℝ := 0.12)
  (discount_pct : ℝ := 0.09)
  (remaining_amount : ℝ := 2633.33) :
  let total_cost_before_discount := part_payment / part_payment_pct
  let discount_amount := discount_pct * total_cost_before_discount
  let discounted_price := total_cost_before_discount - discount_amount
  in discounted_price - part_payment = remaining_amount :=
by
  let total_cost_before_discount : ℝ := part_payment / part_payment_pct
  let discount_amount : ℝ := discount_pct * total_cost_before_discount
  let discounted_price : ℝ := total_cost_before_discount - discount_amount
  have h : discounted_price - part_payment = remaining_amount := sorry
  exact h

end remaining_payment_l441_441285


namespace find_f_13_l441_441167

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

theorem find_f_13 (f : ℝ → ℝ) 
  (h_period : periodic f 1.5) 
  (h_val : f 1 = 20) 
  : f 13 = 20 :=
by
  sorry

end find_f_13_l441_441167


namespace incorrect_conditions_A_and_B_l441_441436

def ConditionA (P : α → Prop) (L : set α) : Prop :=
∀ x, x ∈ L → P x ∧ ¬ (∀ x, P x → x ∈ L)

def ConditionB (P : α → Prop) (L : set α) : Prop :=
∀ x, x ∉ L → ¬ P x ∧ ¬ (∀ x, P x → x ∈ L)

theorem incorrect_conditions_A_and_B {α : Type*} (P : α → Prop) (L : set α) :
  ConditionA P L ∧ ConditionB P L := 
sorry

end incorrect_conditions_A_and_B_l441_441436


namespace max_height_of_rock_l441_441058

theorem max_height_of_rock : 
    ∃ t_max : ℝ, (∀ t : ℝ, -5 * t^2 + 25 * t + 10 ≤ -5 * t_max^2 + 25 * t_max + 10) ∧ (-5 * t_max^2 + 25 * t_max + 10 = 165 / 4) := 
sorry

end max_height_of_rock_l441_441058


namespace triangle_ABC_is_isosceles_l441_441907

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441907


namespace daily_food_cost_is_20_l441_441663

def purchase_price : ℝ := 600
def vaccine_cost : ℝ := 500
def selling_price : ℝ := 2500
def total_profit : ℝ := 600
def number_of_days : ℝ := 40

noncomputable def daily_food_cost : ℝ :=
  let total_cost := purchase_price + vaccine_cost in
  let actual_profit := selling_price - total_cost in
  let food_cost := actual_profit - total_profit in
  food_cost / number_of_days

theorem daily_food_cost_is_20 : daily_food_cost = 20 := by
  sorry

end daily_food_cost_is_20_l441_441663


namespace final_trip_theorem_l441_441295

/-- Define the lineup of dwarfs -/
inductive Dwarf where
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

open Dwarf

/-- Define the conditions -/
-- The dwarfs initially standing next to each other
def adjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = Happy ∧ d2 = Grumpy) ∨
  (d1 = Grumpy ∧ d2 = Dopey) ∨
  (d1 = Dopey ∧ d2 = Bashful) ∨
  (d1 = Bashful ∧ d2 = Sleepy) ∨
  (d1 = Sleepy ∧ d2 = Doc) ∨
  (d1 = Doc ∧ d2 = Sneezy)

-- Snow White is the only one who can row
constant snowWhite_can_row : Prop := true

-- The boat can hold Snow White and up to 3 dwarfs
constant boat_capacity : ℕ := 4

-- Define quarrel if left without Snow White
def quarrel_without_snowwhite (d1 d2 : Dwarf) : Prop := adjacent d1 d2

-- Define the final trip setup
def final_trip (dwarfs : List Dwarf) : Prop :=
  dwarfs = [Grumpy, Bashful, Doc]

-- Theorem to prove the final trip
theorem final_trip_theorem : ∃ dwarfs, final_trip dwarfs :=
  sorry

end final_trip_theorem_l441_441295


namespace impossible_arrangement_of_numbers_l441_441461

theorem impossible_arrangement_of_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) 
(hpos : ∀ i, 0 < a i)
(hdiff : ∃ i, ∀ j ≠ i, a j = a ((j + 1) % n) - a ((j - 1 + n) % n)):
  false :=
by
  sorry

end impossible_arrangement_of_numbers_l441_441461


namespace percentage_increase_bears_l441_441660

-- Define the initial conditions
variables (B H : ℝ) -- B: bears per week without an assistant, H: hours per week without an assistant

-- Define the rate without assistant
def rate_without_assistant : ℝ := B / H

-- Define the working hours with an assistant
def hours_with_assistant : ℝ := 0.9 * H

-- Define the rate with an assistant (100% increase)
def rate_with_assistant : ℝ := 2 * rate_without_assistant

-- Define the number of bears per week with an assistant
def bears_with_assistant : ℝ := rate_with_assistant * hours_with_assistant

-- Prove the percentage increase in the number of bears made per week
theorem percentage_increase_bears (hB : B > 0) (hH : H > 0) :
  ((bears_with_assistant B H - B) / B) * 100 = 80 :=
by
  unfold bears_with_assistant rate_with_assistant hours_with_assistant rate_without_assistant
  simp
  sorry

end percentage_increase_bears_l441_441660


namespace probability_product_gt_5_l441_441593

theorem probability_product_gt_5 : ∀ (x: ℝ), 0 < x ∧ x < 5 ∧ 
  (∃ m : ℝ, (list.sum [1, 2, 3, 4, x]) / 5 = m ∧ (list.median [1, 2, 3, 4, x] = m)) → 
  x = 2.5 →
  (∃ p : ℚ, 
    (p = ( ∑ (a b: ℝ) in [(1,2), (1,2.5), (1,3), (1,4), (2,2.5), (2,3), (2,4), (2.5,3), (2.5,4), (3,4)], 
    if a * b > 5 then 1/10 else 0/10)) ∧ p = 1/2) :=
begin
  intro x,
  rintros ⟨h1, h2⟩ ⟨m, hm1, hm2⟩ hx,
  sorry
end

end probability_product_gt_5_l441_441593


namespace isosceles_triangle_of_parallel_l441_441779

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441779


namespace snow_white_last_trip_l441_441311

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l441_441311


namespace triangle_is_isosceles_l441_441809

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441809


namespace min_room_size_l441_441490

theorem min_room_size (S : ℕ) (h : S ≥ Real.sqrt (9 * 9 + 12 * 12)) : S = 15 :=
by
  have h_diag : Real.sqrt (9 * 9 + 12 * 12) = 15 := by sorry
  linarith [h, h_diag]

end min_room_size_l441_441490


namespace has_root_in_interval_l441_441360

def f (x : ℝ) : ℝ := x^3 - x^2 - x - 1

theorem has_root_in_interval : ∃ x ∈ Ioo 1 2, f x = 0 :=
by
  sorry

end has_root_in_interval_l441_441360


namespace prove_y_is_odd_l441_441205

noncomputable def is_odd_function (y : ℝ → ℝ) : Prop :=
  ∀ x, y (-x) = -y (x)

def f (m : ℤ) : ℝ → ℝ := λ x, x^m
def g (n : ℤ) : ℝ → ℝ := λ x, x^n

theorem prove_y_is_odd (m n : ℤ) (h_f_mono_dec : ∀ x (hx : 0 < x), f m (x) > f m (x + 1)) 
    (h_g_mono_inc : ∀ x (hx : 0 < x), g n (x) < g n (x + 1)) :
  is_odd_function (λ x, f m x + g n x) ↔ (m % 2 = 1 ∧ m < 0) ∧ (n % 2 = 1 ∧ n > 0) :=
sorry

end prove_y_is_odd_l441_441205


namespace roundness_of_8000000_l441_441524

theorem roundness_of_8000000 : ∀ (n : ℕ), 
  (∃ (a b : ℕ), n = 2^a * 5^b) → 
  (a = 9 ∧ b = 6) → 
  (a + b = 15) := 
by
  intros n h_factorization h_exponents
  cases h_exponents with ha hb
  cases h_factorization with a hb
  sorry

end roundness_of_8000000_l441_441524


namespace perfect_square_trinomial_m_l441_441190

theorem perfect_square_trinomial_m (m : ℝ) (h : ∃ b : ℝ, (x : ℝ) ↦ (x + b)^2 = x^2 + mx + 16) : m = 8 ∨ m = -8 :=
sorry

end perfect_square_trinomial_m_l441_441190


namespace area_of_quadrilateral_is_3_5_l441_441003

def vertex (x y : ℝ) := (x, y)

def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (x4, y4) := D
  (1 / 2) * abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

def A := vertex 0 0
def B := vertex 4 3
def C := vertex 7 0
def D := vertex 4 4

theorem area_of_quadrilateral_is_3_5 : area_quadrilateral A B C D = 3.5 := by
  sorry

end area_of_quadrilateral_is_3_5_l441_441003


namespace largest_non_sum_217_l441_441399

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end largest_non_sum_217_l441_441399


namespace isosceles_triangle_of_parallel_l441_441878

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441878


namespace triangle_ABC_isosceles_l441_441845

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441845


namespace triangle_isosceles_if_parallel_l441_441862

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441862


namespace last_trip_l441_441312

def initial_order : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

def boatCapacity : Nat := 4  -- Including Snow White

def adjacentPairsQuarrel (adjPairs : List (String × String)) : Prop :=
  ∀ (d1 d2 : String), (d1, d2) ∈ adjPairs → (d2, d1) ∈ adjPairs → False

def canRow (person : String) : Prop := person = "Snow White"

noncomputable def final_trip (remainingDwarfs : List String) (allTrips : List (List String)) : List String := ["Grumpy", "Bashful", "Doc"]

theorem last_trip (adjPairs : List (String × String))
  (h_adj : adjacentPairsQuarrel adjPairs)
  (h_row : canRow "Snow White")
  (dwarfs_order : List String = initial_order)
  (without_quarrels : ∀ trip : List String, trip ∈ allTrips → 
    ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → (d1, d2) ∈ adjPairs → 
    ("Snow White" ∈ trip) → True) :
  final_trip ["Grumpy", "Bashful", "Doc"] allTrips = ["Grumpy", "Bashful", "Doc"] :=
sorry

end last_trip_l441_441312


namespace parallelogram_base_l441_441539

theorem parallelogram_base
  (Area Height Base : ℕ)
  (h_area : Area = 120)
  (h_height : Height = 10)
  (h_area_eq : Area = Base * Height) :
  Base = 12 :=
by
  /- 
    We assume the conditions:
    1. Area = 120
    2. Height = 10
    3. Area = Base * Height 
    Then, we need to prove that Base = 12.
  -/
  sorry

end parallelogram_base_l441_441539


namespace least_possible_value_of_m_plus_n_l441_441676

theorem least_possible_value_of_m_plus_n 
(m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) 
(hgcd : Nat.gcd (m + n) 210 = 1) 
(hdiv : ∃ k, m^m = k * n^n)
(hnotdiv : ¬ ∃ k, m = k * n) : 
  m + n = 407 := 
sorry

end least_possible_value_of_m_plus_n_l441_441676


namespace isosceles_triangle_l441_441793

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441793


namespace length_CF_l441_441390

noncomputable def circle_radius : ℝ := 7

-- Points A and B are intersection points of two circles with the same radius.
-- Point C is on the first circle, and point D is on the second circle.
-- Point B lies on segment CD and ∠CAD = 90°.
-- Point F lies on the perpendicular through B to CD such that BF = BD.
-- Prove that CF equals 14.

theorem length_CF :
    let R := circle_radius
    ∃ (B C D F : ℝ -> ℝ -> Prop), 
    (∀ (C : ℝ -> ℝ -> Prop), C (circle_radius, 0)) → -- C is on the first circle
    (∀ (D : ℝ -> ℝ -> Prop), D (-circle_radius, 0)) → -- D is on the second circle
    (∃ (B : ℝ -> ℝ -> Prop), B (0, 0) ∧ -- B lies on segment CD
    ∃ (F : ℝ -> ℝ -> Prop), (F (0, -circle_radius) ∧ -- F is on the perpendicular through B to CD
    (∃ (angle_CAD : ℝ), angle_CAD = 90)) ∧ -- ∠CAD = 90°
    let CF := 2 * R in 
    CF = 14 :=
begin
  sorry
end

end length_CF_l441_441390


namespace trigonometric_correctness_l441_441585

theorem trigonometric_correctness (x y : ℝ) (h_x : x = 3) (h_y : y = -4) :
  let r := Real.sqrt (x^2 + y^2)
  in tan (atan2 y x) = -4 / 3 := 
by
  -- Conditions
  let r := Real.sqrt (x^2 + y^2)
  have h_r : r = 5 := by
    simp [*, Real.sqrt_eq_rpow, Real.sqrt, Real.sqrt, sq]
    sorry

  -- Show that tan (atan2 y x) = -4 / 3
  let α := atan2 y x
  have h_tan_α : tan α = y / x := by
    simp [tan_of_atan2]
    sorry

  -- Substituting y = -4 and x = 3
  rw [h_x, h_y] at h_tan_α
  have : tan α = -4 / 3 := by
    simp [*]
    sorry

  exact this

end trigonometric_correctness_l441_441585


namespace vectors_coplanar_l441_441084

-- Define the vectors a, b, and c as conditions
def a : Vector ℝ 3 := ⟨[3, 4, 2]⟩
def b : Vector ℝ 3 := ⟨[1, 1, 0]⟩
def c : Vector ℝ 3 := ⟨[8, 11, 6]⟩

-- Statement: Prove that the vectors are coplanar, which is equivalent to saying that the determinant is zero
theorem vectors_coplanar : Matrix.det (Matrix.ofVecs [a, b, c]) = 0 := 
sorry

end vectors_coplanar_l441_441084


namespace rectangle_length_twice_breadth_l441_441024

theorem rectangle_length_twice_breadth
  (b : ℝ) 
  (l : ℝ)
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 4) = l * b + 75) :
  l = 190 / 3 :=
sorry

end rectangle_length_twice_breadth_l441_441024


namespace eggs_supplied_l441_441695

-- Define the conditions
def daily_eggs_first_store (D : ℕ) : ℕ := 12 * D
def daily_eggs_second_store : ℕ := 30
def total_weekly_eggs (D : ℕ) : ℕ := 7 * (daily_eggs_first_store D + daily_eggs_second_store)

-- Statement: prove that if the total number of eggs supplied in a week is 630,
-- then Mark supplies 5 dozen eggs to the first store each day.
theorem eggs_supplied (D : ℕ) (h : total_weekly_eggs D = 630) : D = 5 :=
by
  sorry

end eggs_supplied_l441_441695


namespace donation_to_second_orphanage_l441_441287

variable (total_donation : ℝ) (first_donation : ℝ) (third_donation : ℝ)

theorem donation_to_second_orphanage :
  total_donation = 650 ∧ first_donation = 175 ∧ third_donation = 250 →
  (total_donation - first_donation - third_donation = 225) := by
  sorry

end donation_to_second_orphanage_l441_441287


namespace find_f_60_l441_441943

def f : ℝ → ℝ := sorry

axiom h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f(x * y) = f(x) / y^2
axiom h2 : f 40 = 30

theorem find_f_60 : f 60 = 40 / 3 :=
by
  sorry

end find_f_60_l441_441943


namespace diana_total_extra_video_game_time_l441_441526

-- Definitions from the conditions
def minutesPerHourReading := 30
def raisePercent := 20
def choresToMinutes := 10
def maxChoresBonusMinutes := 60
def sportsPracticeHours := 8
def homeworkHours := 4
def totalWeekHours := 24
def readingHours := 8
def choresCompleted := 10

-- Deriving some necessary facts
def baseVideoGameTime := readingHours * minutesPerHourReading
def raiseMinutes := baseVideoGameTime * (raisePercent / 100)
def videoGameTimeWithRaise := baseVideoGameTime + raiseMinutes

def bonusesFromChores := (choresCompleted / 2) * choresToMinutes
def limitedChoresBonus := min bonusesFromChores maxChoresBonusMinutes

-- Total extra video game time
def totalExtraVideoGameTime := videoGameTimeWithRaise + limitedChoresBonus

-- The proof problem
theorem diana_total_extra_video_game_time : totalExtraVideoGameTime = 338 := by
  sorry

end diana_total_extra_video_game_time_l441_441526


namespace triangle_is_isosceles_l441_441806

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441806


namespace remaining_cube_height_l441_441102

noncomputable def cube_edge := 2
noncomputable def remaining_height (e: ℝ) : ℝ := 
  let diagonal := real.sqrt (e^2 + e^2 + e^2) in
  let triangle_side_length := real.sqrt (e^2 + e^2) in
  let tetrahedron_volume := 1/3 * (sqrt 3 / 4 * triangle_side_length^2) * 1 in
  e - 1  -- resulting height after removing 1 unit from edge length
-- Define final theorem as the proof problem statement
theorem remaining_cube_height : remaining_height cube_edge = 1 := sorry


end remaining_cube_height_l441_441102


namespace probability_at_least_one_l441_441122

theorem probability_at_least_one (p1 p2 : ℝ) (hp1 : 0 ≤ p1) (hp2 : 0 ≤ p2) (hp1p2 : p1 ≤ 1) (hp2p2 : p2 ≤ 1)
  (h0 : 0 ≤ 1 - p1) (h1 : 0 ≤ 1 - p2) (h2 : 1 - (1 - p1) ≥ 0) (h3 : 1 - (1 - p2) ≥ 0) :
  1 - (1 - p1) * (1 - p2) = 1 - (1 - p1) * (1 - p2) := by
  sorry

end probability_at_least_one_l441_441122


namespace triangle_is_isosceles_l441_441818

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441818


namespace evaluate_program_output_l441_441715

def final_output_value_of_M : ℕ :=
  let M₀ := 1 in
  let M₁ := M₀ + 1 in
  let M₂ := M₁ + 2 in
  M₂

theorem evaluate_program_output : final_output_value_of_M = 4 :=
  by
    sorry

end evaluate_program_output_l441_441715


namespace double_sum_evaluation_l441_441529

theorem double_sum_evaluation :
  (∑ m in Finset.range 100 \ Finset.range 2, ∑ n in Finset.Ici 3, 1 / (m * n * (m + n + 2))) = 103 / 208 :=
by
  sorry

end double_sum_evaluation_l441_441529


namespace triangle_is_isosceles_l441_441814

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441814


namespace function_increasing_l441_441116

-- Define the function y
def y (x : ℝ) : ℝ := Real.logb (1/2 : ℝ) (x^2 - 3*x + 2)

-- Define the condition: the function is increasing on the interval (2, +∞)
theorem function_increasing : ∀ x1 x2 : ℝ, 
  (2 < x1 ∧ 2 < x2 ∧ x1 < x2) → y x1 < y x2 :=
by
  sorry

end function_increasing_l441_441116


namespace final_trip_theorem_l441_441297

/-- Define the lineup of dwarfs -/
inductive Dwarf where
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

open Dwarf

/-- Define the conditions -/
-- The dwarfs initially standing next to each other
def adjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = Happy ∧ d2 = Grumpy) ∨
  (d1 = Grumpy ∧ d2 = Dopey) ∨
  (d1 = Dopey ∧ d2 = Bashful) ∨
  (d1 = Bashful ∧ d2 = Sleepy) ∨
  (d1 = Sleepy ∧ d2 = Doc) ∨
  (d1 = Doc ∧ d2 = Sneezy)

-- Snow White is the only one who can row
constant snowWhite_can_row : Prop := true

-- The boat can hold Snow White and up to 3 dwarfs
constant boat_capacity : ℕ := 4

-- Define quarrel if left without Snow White
def quarrel_without_snowwhite (d1 d2 : Dwarf) : Prop := adjacent d1 d2

-- Define the final trip setup
def final_trip (dwarfs : List Dwarf) : Prop :=
  dwarfs = [Grumpy, Bashful, Doc]

-- Theorem to prove the final trip
theorem final_trip_theorem : ∃ dwarfs, final_trip dwarfs :=
  sorry

end final_trip_theorem_l441_441297


namespace keychain_arrangements_l441_441602

theorem keychain_arrangements : 
  let house_and_car := 2
  let office_and_mailbox := 2
  let remaining_keys := 2
  let total_units := house_and_car / 2 + office_and_mailbox / 2 + remaining_keys
  let distinct_arrangements_with_rotation := 3 // (4-1)
  let pair_arrangements := factorial 2 * factorial 2
  total_units = 4 ∧ distinct_arrangements_with_rotation = 3 
  ∧ pair_arrangements = 4 
  →  distinct_arrangements_with_rotation * pair_arrangements = 12 :=
by
  sorry

end keychain_arrangements_l441_441602


namespace problem_equivalence_l441_441028

theorem problem_equivalence : 4 * 299 + 3 * 299 + 2 * 299 + 298 = 2989 := by
  sorry

end problem_equivalence_l441_441028


namespace average_debt_payment_l441_441466

noncomputable def total_amount (number_of_installments first_installments remaining_installments first_amount increment_amount : ℕ) : ℕ :=
  (first_installments * first_amount) + (remaining_installments * (first_amount + increment_amount))

noncomputable def average_payment (total_amount number_of_installments : ℕ) : ℕ :=
  total_amount / number_of_installments

theorem average_debt_payment :
  let number_of_installments := 52
  let first_installments := 12
  let remaining_installments := number_of_installments - first_installments
  let first_amount := 410
  let increment_amount := 65
  let total := total_amount number_of_installments first_installments remaining_installments first_amount increment_amount
  let average := average_payment total number_of_installments
  average = 460 :=
by
  rw [
    total_amount,
    average_payment,
    number_of_installments,
    first_installments,
    remaining_installments,
    first_amount,
    increment_amount
  ]
  sorry

end average_debt_payment_l441_441466


namespace max_profit_l441_441044

theorem max_profit{L1 L2 L : ℕ → ℝ} :
  (L1 = λ x, 5.06 * x - 0.15 * x ^ 2) →
  (L2 = λ x, 2 * x) →
  (L = λ x, L1 x + L2 (15 - x)) →
  ∃ (x : ℕ), x ∈ {0, 1, ..., 15} ∧ L x = 45.6 :=
by sorry

end max_profit_l441_441044


namespace speed_of_truck_l441_441052

variables (num_cattle truck_capacity distance_one_way total_time total_distance total_trips : ℕ)
variable (speed : ℕ)
variables (total_cattle_transported_per_trip : ℕ)

-- Conditions
def num_cattle := 400
def truck_capacity := 20
def distance_one_way := 60
def total_time := 40

def total_trips : ℕ := num_cattle / truck_capacity
def total_distance : ℕ := total_trips * (2 * distance_one_way)

-- Proposition to prove
theorem speed_of_truck (h : total_distance = total_trips * 2 * distance_one_way ∧ total_time = 40) :
  speed = total_distance / total_time :=
begin
  have trips := num_cattle / truck_capacity,
  have dist := trips * (2 * distance_one_way),
  have total_time := 40,
  sorry
end

end speed_of_truck_l441_441052


namespace largest_integer_not_sum_of_30_and_composite_l441_441403

theorem largest_integer_not_sum_of_30_and_composite : 
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b < 30 ∧ ¬ prime b ∧ (n = 30 * a + b) → n = 157 :=
by
  sorry

end largest_integer_not_sum_of_30_and_composite_l441_441403


namespace frog_jumping_sequences_l441_441512

theorem frog_jumping_sequences (hexagon : hexagon) (start : vertex)
  (jumps : vertex → vertex → bool) (stops_condition : vertex → ℕ → bool) :
  (∃ (A D : vertex), 
    hexagon.is_regular ∧ 
    start = A ∧ 
    stops_condition D 5 ∧ 
    ∀ v w : vertex, jumps v w → adjacent v w) 
  → (number_of_sequences : ℕ) 
  → number_of_sequences = 26 := 
by {
  sorry
}

end frog_jumping_sequences_l441_441512


namespace bill_length_l441_441382

variable (width area : ℝ)

theorem bill_length (w3 : width = 3) (a177 : area = 1.77) : length = 0.59 :=
by
  -- Define the formula for area
  let length := area / width
  have h : area = width * length, from sorry
  rw [←a177, ←w3] at h
  exact sorry

end bill_length_l441_441382


namespace chameleon_colors_cannot_cycle_l441_441703

def chameleons_problem : Prop :=
  (∀ c : list ℕ, c.length = 35 ∧ (∀ i, c[i] = c[(i + 1) % 35] ∨ c[i] = c[(i - 1) % 35])) → 
  ¬ (∀ n, ∃ i, c[i] = n) → ∃ (t₀ t₁ t₂: ℕ), t₀ ≠ t₁ ∧ t₁ ≠ t₂ ∧ t₂ ≠ t₀

theorem chameleon_colors_cannot_cycle : chameleons_problem :=
by
  sorry

end chameleon_colors_cannot_cycle_l441_441703


namespace normal_intersects_at_l441_441668

def parabola (x : ℝ) : ℝ := x^2

def slope_of_tangent (x : ℝ) : ℝ := 2 * x

-- C = (2, 4) is a point on the parabola
def C : ℝ × ℝ := (2, parabola 2)

-- Normal to the parabola at C intersects again at point D
-- Prove that D = (-9/4, 81/16)
theorem normal_intersects_at (D : ℝ × ℝ) :
  D = (-9/4, 81/16) :=
sorry

end normal_intersects_at_l441_441668


namespace triangle_ABC_isosceles_l441_441841

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441841


namespace excenters_concyclic_l441_441639

theorem excenters_concyclic
  (A B C H J I P : Type*)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited H] [Inhabited J] [Inhabited I] [Inhabited P]
  (triangle_acute : ∀ (A B C : Type*), Prop)
  (foot_perpendicular : ∀ (A B C H : Type*), Prop)
  (is_excenter : ∀ (A B C H J I : Type*), Prop)
  (incircle_touch : ∀ (A B C P : Type*), Prop)
  (H_perpendicular : foot_perpendicular A B C H)
  (J_excenter : is_excenter A B C H J I)
  (I_excenter : is_excenter A B C H J I)
  (P_incircle : incircle_touch A B C P) :
  concyclic I J P H :=
sorry

end excenters_concyclic_l441_441639


namespace intersection_in_second_quadrant_l441_441433

theorem intersection_in_second_quadrant (m : ℝ) : 
  (-4 < m) ∧ (m < 4) ↔ 
  let x := (m - 4) / 4; let y := (m + 4) / 2 in x < 0 ∧ y > 0 :=
begin
  sorry
end

end intersection_in_second_quadrant_l441_441433


namespace barycentric_coordinates_vector_l441_441262

-- Defining vectors and their basic operations
section
variables {V : Type} [add_comm_group V] [vector_space ℝ V]

-- Definitions for points and barycentric coordinates
variables (α β γ : ℝ)
variables (A B C X M : V)
variables (bary_coords : V → ℝ × ℝ × ℝ)

-- Centroid definition
def centroid (A B C : V) : V := (A + B + C) / 3

-- Vector definitions
def vector (P Q : V) : V := Q - P

-- Given conditions
variables (h1 : bary_coords X = (α, β, γ)) (h2 : M = centroid A B C)

-- The statement to be proved
theorem barycentric_coordinates_vector :
  3 • vector X M = (α - β) • vector A B + (β - γ) • vector B C + (γ - α) • vector C A :=
sorry
end

end barycentric_coordinates_vector_l441_441262


namespace problem_statement_l441_441745

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441745


namespace flower_bed_weeds_count_l441_441693

theorem flower_bed_weeds_count (F : ℕ) (earnings_per_weed : ℕ) (vegetable_patch_weeds : ℕ) (half_grass_weeds : ℕ) (total_earnings_before_soda : ℕ) (soda_cost : ℕ) (earnings_left : ℕ) :
  earnings_per_weed = 6 →
  vegetable_patch_weeds = 14 →
  half_grass_weeds = 16 →
  soda_cost = 99 →
  earnings_left = 147 →
  total_earnings_before_soda = 6 * (F + vegetable_patch_weeds + half_grass_weeds) →
  147 + 99 = 246 →
  6 * (F + 14 + 16) = 246 →
  F = 11 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  simp only [Nat.succ_add_eq_succ_add, eq_self_iff_true, add_assoc, add_zero]
  exact by
    rw [add_comm] at h8
    linarith


end flower_bed_weeds_count_l441_441693


namespace isosceles_triangle_l441_441822

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441822


namespace last_trip_l441_441314

def initial_order : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

def boatCapacity : Nat := 4  -- Including Snow White

def adjacentPairsQuarrel (adjPairs : List (String × String)) : Prop :=
  ∀ (d1 d2 : String), (d1, d2) ∈ adjPairs → (d2, d1) ∈ adjPairs → False

def canRow (person : String) : Prop := person = "Snow White"

noncomputable def final_trip (remainingDwarfs : List String) (allTrips : List (List String)) : List String := ["Grumpy", "Bashful", "Doc"]

theorem last_trip (adjPairs : List (String × String))
  (h_adj : adjacentPairsQuarrel adjPairs)
  (h_row : canRow "Snow White")
  (dwarfs_order : List String = initial_order)
  (without_quarrels : ∀ trip : List String, trip ∈ allTrips → 
    ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → (d1, d2) ∈ adjPairs → 
    ("Snow White" ∈ trip) → True) :
  final_trip ["Grumpy", "Bashful", "Doc"] allTrips = ["Grumpy", "Bashful", "Doc"] :=
sorry

end last_trip_l441_441314


namespace isosceles_triangle_of_parallel_l441_441770

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441770


namespace norm_two_v_l441_441148

variable (v : Vector ℝ)

-- Conditions
axiom norm_v : ‖v‖ = 5
axiom norm_scalar : ∀ (k : ℝ) (v : Vector ℝ), ‖k • v‖ = |k| * ‖v‖

-- Statement to prove
theorem norm_two_v : ‖2 • v‖ = 10 :=
by
  sorry

end norm_two_v_l441_441148


namespace largest_product_of_consecutive_integers_l441_441480

-- Define the set of numbers
def number_set : Set ℤ := {-4, -3, -1, 5, 6, 7}

-- Predicate to check if two numbers are consecutive
def consecutive (a b : ℤ) : Prop := abs (a - b) = 1

-- Function to calculate the product of a list of four numbers
def product_of_four (a b c d : ℤ) : ℤ := a * b * c * d

-- Problem statement: Proving that the largest product of any four different numbers with two consecutive integers is -210
theorem largest_product_of_consecutive_integers :
  ∃ a b c d ∈ number_set, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ consecutive a b ∧
  (∀ x y z w ∈ number_set, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧ consecutive x y → product_of_four x y z w ≤ -210) :=
sorry

end largest_product_of_consecutive_integers_l441_441480


namespace polygons_intersection_area_at_least_one_l441_441441

theorem polygons_intersection_area_at_least_one :
  ∀ (p1 p2 p3 : Set ℝ²), (Area p1 = 3) ∧ (Area p2 = 3) ∧ (Area p3 = 3) → (Area (p1 ∩ p2) + Area (p2 ∩ p3) + Area (p1 ∩ p3) + Area (p1 ∩ p2 ∩ p3)) ≤ 6 →
  ∃ (pi pj : Set ℝ²), pi ≠ pj ∧ Area (pi ∩ pj) ≥ 1 := 
by
  intros p1 p2 p3 h_area h_total_area
  sorry

end polygons_intersection_area_at_least_one_l441_441441


namespace fraction_simplified_l441_441720

-- Define the fraction function
def fraction (n : ℕ) := (21 * n + 4, 14 * n + 3)

-- Define the gcd function to check if fractions are simplified.
def is_simplified (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Main theorem
theorem fraction_simplified (n : ℕ) : is_simplified (21 * n + 4) (14 * n + 3) :=
by
  -- Rest of the proof
  sorry

end fraction_simplified_l441_441720


namespace exists_segment_with_points_l441_441667

theorem exists_segment_with_points (S : Finset ℕ) (n : ℕ) (hS : S.card = 6 * n)
  (hB : ∃ B : Finset ℕ, B ⊆ S ∧ B.card = 4 * n) (hG : ∃ G : Finset ℕ, G ⊆ S ∧ G.card = 2 * n) :
  ∃ t : Finset ℕ, t ⊆ S ∧ t.card = 3 * n ∧ (∃ B' : Finset ℕ, B' ⊆ t ∧ B'.card = 2 * n) ∧ (∃ G' : Finset ℕ, G' ⊆ t ∧ G'.card = n) :=
  sorry

end exists_segment_with_points_l441_441667


namespace length_of_diagonal_AC_l441_441218

-- Definitions based on the conditions
variable (AB BC CD DA AC : ℝ)
variable (angle_ADC : ℝ)

-- Conditions
def conditions : Prop :=
  AB = 12 ∧ BC = 12 ∧ CD = 15 ∧ DA = 15 ∧ angle_ADC = 120

theorem length_of_diagonal_AC (h : conditions AB BC CD DA angle_ADC) : AC = 15 :=
sorry

end length_of_diagonal_AC_l441_441218


namespace area_AOC_is_1_l441_441263

noncomputable def point := (ℝ × ℝ) -- Define a point in 2D space

def vector_add (v1 v2 : point) : point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_zero : point := (0, 0)

def scalar_mul (r : ℝ) (v : point) : point :=
  (r * v.1, r * v.2)

def vector_eq (v1 v2 : point) : Prop := 
  v1.1 = v2.1 ∧ v1.2 = v2.2

variables (A B C O : point)
variable (area_ABC : ℝ)

-- Conditions:
-- Point O is a point inside triangle ABC with an area of 4
-- \(\overrightarrow {OA} + \overrightarrow {OB} + 2\overrightarrow {OC} = \overrightarrow {0}\)
axiom condition_area : area_ABC = 4
axiom condition_vector : vector_eq (vector_add (vector_add O A) (vector_add O B)) (scalar_mul (-2) O)

-- Theorem to prove: the area of triangle AOC is 1
theorem area_AOC_is_1 : (area_ABC / 4) = 1 := 
sorry

end area_AOC_is_1_l441_441263


namespace largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441408

theorem largest_integer_not_sum_of_multiple_of_30_and_composite :
  ∃ n : ℕ, ∀ a b : ℕ, 0 ≤ b ∧ b < 30 ∧ b.prime ∧ (∀ k < a, (b + 30 * k).prime)
    → (30 * a + b = n) ∧
      (∀ m : ℕ, ∀ c d : ℕ, 0 ≤ d ∧ d < 30 ∧ d.prime ∧ (∀ k < c, (d + 30 * k).prime) 
        → (30 * c + d ≤ n)) ∧
      n = 93 :=
by
  sorry

end largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441408


namespace isosceles_triangle_of_parallel_l441_441787

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441787


namespace min_value_of_reciprocal_sum_l441_441578

variable (a b : ℝ)
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (condition : 2 * a + b = 1)

theorem min_value_of_reciprocal_sum : (1 / a) + (1 / b) = 3 + 2 * Real.sqrt 2 :=
by
  -- Proof is skipped
  sorry

end min_value_of_reciprocal_sum_l441_441578


namespace cuboid_cut_parallelogram_possible_l441_441107

noncomputable def is_cross_section_parallelogram (C : Type) [cuboid C] (P : Type) [plane P] : Prop :=
  ∃ Q : Type, [quadilateral Q] (cut_cuboid_with_plane C P Q ∧ is_paralellogram Q)

theorem cuboid_cut_parallelogram_possible :
  ∀ (C : Type) [cuboid C] (P : Type) [plane P],
  (¬ plane.is_perpendicular_to_cuboid C P) → 
  (plane.passes_through_four_faces_of_cuboid C P) → 
  is_cross_section_parallelogram C P :=
begin
  sorry
end

end cuboid_cut_parallelogram_possible_l441_441107


namespace complex_in_fourth_quadrant_l441_441008

noncomputable def complex_quadrant (m : ℝ) : Prop :=
  let z : ℂ := (3 - 2 * m) + (1 - m) * complex.I
  z.re > 0 ∧ z.im < 0

theorem complex_in_fourth_quadrant (m : ℝ) (h1 : 1 < m) (h2 : m < 3 / 2) :
  complex_quadrant m :=
sorry

end complex_in_fourth_quadrant_l441_441008


namespace triangle_is_isosceles_if_parallel_l441_441732

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441732


namespace sin_cos_value_l441_441615

theorem sin_cos_value (θ : ℝ) (h : (sin θ + cos θ) / (sin θ - cos θ) = 2) : sin θ * cos θ = 3 / 10 :=
by
  sorry

end sin_cos_value_l441_441615


namespace increasing_range_of_a_l441_441631

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 2 then -x^2 + 2 * a * x else (6 - a) * x + 2

theorem increasing_range_of_a :
  (∀ x, f a x ≤ f a (x + 1)) ↔ (2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end increasing_range_of_a_l441_441631


namespace trip_time_is_8_hours_29_minutes_l441_441553

noncomputable def start_time := 
  let hour_angle := 30 * 7 in
  let start_minute := (90 / 5.5).approx in
  (7, start_minute)

noncomputable def end_time := 
  let end_hour := 15 in
  let end_minute := (270 / 6).approx in
  (15, end_minute)

noncomputable def trip_duration (start_hour start_minute end_hour end_minute : ℝ) : ℝ :=
  (end_hour - start_hour) + ((end_minute - start_minute) / 60)

theorem trip_time_is_8_hours_29_minutes 
    (start_hour start_minute end_hour end_minute : ℝ) 
    (h_start: start_time = (start_hour, start_minute))
    (h_end: end_time = (end_hour, end_minute))
    : trip_duration start_hour start_minute end_hour end_minute = 8 + (29/60) :=
by
  sorry

end trip_time_is_8_hours_29_minutes_l441_441553


namespace acute_angle_perpendicular_vectors_l441_441600

variable (x : ℝ)

def vector_a := (1, 2) : ℝ × ℝ
def vector_b := (x, 1) : ℝ × ℝ

theorem acute_angle : (1 * x + 2 * 1 > 0) → x ∈ set.Ioi (-2) \ {1/2} :=
by sorry

def vector_a2_b := (1 + 2 * x, 4) : ℝ × ℝ
def vector_2a_b := (2 - x, 3) : ℝ × ℝ

theorem perpendicular_vectors : ((1 + 2 * x) * (2 - x) + 4 * 3 = 0) → x = 7 / 2 :=
by sorry

end acute_angle_perpendicular_vectors_l441_441600


namespace eggs_left_l441_441048

def initial_eggs := 20
def mother_used := 5
def father_used := 3
def chicken1_laid := 4
def chicken2_laid := 3
def chicken3_laid := 2
def oldest_took := 2

theorem eggs_left :
  initial_eggs - (mother_used + father_used) + (chicken1_laid + chicken2_laid + chicken3_laid) - oldest_took = 19 := 
by
  sorry

end eggs_left_l441_441048


namespace angle_bisector_passes_through_circumcenter_l441_441083

-- Given an acute-angled triangle ABC with angle A = 60 degrees
variables {A B C X Y H O : Type} [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq X] [decidable_eq Y] [decidable_eq H] [decidable_eq O]
variable (triangle_ABC : Triangle A B C)
variable (angle_A_eq_60 : angle (triangle_ABC.vertex_A) = 60)
variable (BX_altitude : altitude (triangle_ABC.vertex_B) (triangle_ABC.side_AC))
variable (CY_altitude : altitude (triangle_ABC.vertex_C) (triangle_ABC.side_AB))
variable (intersection_H : orthocenter BX_altitude CY_altitude = H)
variable (circumcenter_O : circumcenter triangle_ABC = O)

-- Prove that the bisector of angle XHO passes through the circumcenter of the triangle ABC
theorem angle_bisector_passes_through_circumcenter 
  (hx : is_angle_bisector angle (X H O)) : 
  is_on_line O X H :=
sorry

end angle_bisector_passes_through_circumcenter_l441_441083


namespace ferris_wheel_large_seats_people_l441_441932

theorem ferris_wheel_large_seats_people (large_seats : ℕ) (weight_limit_per_large_seat : ℕ) (average_weight_per_person : ℕ) :
  large_seats = 7 →
  weight_limit_per_large_seat = 1500 →
  average_weight_per_person = 180 →
  7 * (1500 / 180) = 56 := 
by
  intros
  simp
  sorry

end ferris_wheel_large_seats_people_l441_441932


namespace tangentCircleDiameterAB_parabola_circle_tangent_l441_441996

noncomputable def parabolaCEquation : Prop :=
  ∃ (a : ℝ), (a = 1/2) ∧ (∀ (x y: ℝ), ((y^2) = (4 * a * x) ↔ y^2 = 2*x))

theorem tangentCircleDiameterAB : Prop :=
  let A := (2:ℝ, 2:ℝ) in
  let B := (2:ℝ, -2:ℝ) in
  let M := ((2:ℝ, 0:ℝ)) in
  let r : ℝ := (2:ℝ) in
  let d_M : ℝ := abs (M.1 - -1/2) in
  d_M = r

theorem parabola_circle_tangent : parabolaCEquation ∧ tangentCircleDiameterAB :=
by {
  sorry
}

end tangentCircleDiameterAB_parabola_circle_tangent_l441_441996


namespace will_buy_toys_l441_441027

theorem will_buy_toys : 
  ∀ (initialMoney spentMoney toyCost : ℕ), 
  initialMoney = 83 → spentMoney = 47 → toyCost = 4 → 
  (initialMoney - spentMoney) / toyCost = 9 :=
by
  intros initialMoney spentMoney toyCost hInit hSpent hCost
  sorry

end will_buy_toys_l441_441027


namespace angle_ACB_is_75_degrees_l441_441206

theorem angle_ACB_is_75_degrees
  (A B C D : Type)
  [inhabited A] [inhabited B] [inhabited C] [inhabited D]
  (angle_ABC : ℝ)
  (BD CD : ℝ)
  (angle_DAB : ℝ)
  (BC : ℝ)
  (h1 : angle_ABC = 45)
  (h2 : 2 * BD = CD)
  (h3 : angle_DAB = 15) :
  ∃ θ : ℝ, θ = 75 ∧ ∠ ACB = θ := 
begin
  sorry
end

end angle_ACB_is_75_degrees_l441_441206


namespace hyungjun_initial_ribbon_length_l441_441187

noncomputable def initial_ribbon_length (R: ℝ) : Prop :=
  let used_for_first_box := R / 2 + 2000
  let remaining_after_first := R - used_for_first_box
  let used_for_second_box := (remaining_after_first / 2) + 2000
  remaining_after_first - used_for_second_box = 0

theorem hyungjun_initial_ribbon_length : ∃ R: ℝ, initial_ribbon_length R ∧ R = 12000 :=
  by
  exists 12000
  unfold initial_ribbon_length
  simp
  sorry

end hyungjun_initial_ribbon_length_l441_441187


namespace triangle_is_isosceles_l441_441853

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441853


namespace simplify_expression_l441_441925

theorem simplify_expression (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + ((a + 1) / (1 - a)))) = (1 + a) / 2 := 
by
  sorry

end simplify_expression_l441_441925


namespace minimum_area_of_sap_circle_l441_441948

noncomputable def function_y (x : ℝ) : ℝ := 1 / (|x| - 1)

def y_axis_intersection : ℝ := function_y 0

def symmetric_point : ℝ × ℝ := (0, -y_axis_intersection)

def distance (x : ℝ) : ℝ :=
  real.sqrt (x^2 + (function_y x - y_axis_intersection)^2)

def minimum_distance : ℝ :=
  real.sqrt 3

def radius : ℝ :=
  minimum_distance

theorem minimum_area_of_sap_circle : real.pi * radius^2 = 3 * real.pi :=
by
  sorry

end minimum_area_of_sap_circle_l441_441948


namespace infinite_solutions_when_c_eq_2_5_l441_441119

-- Conditions given
theorem infinite_solutions_when_c_eq_2_5 :
  ∀ y : ℝ, 3 * (5 + 2 * 2.5 * y) = 15 * y + 15 :=
by {
  intros y,
  sorry
}

end infinite_solutions_when_c_eq_2_5_l441_441119


namespace questionnaire_B_count_l441_441995

theorem questionnaire_B_count :
  ∃ n : ℕ, n = 10 ∧ ∀ k : ℕ, 16 ≤ k ∧ k ≤ 25 → (30 * k - 21) ∈ set.Icc 451 750 :=
by {
  sorry
}

end questionnaire_B_count_l441_441995


namespace area_of_veranda_l441_441448

-- Definitions based on the conditions
def length_room : ℝ := 21
def width_room : ℝ := 12
def width_veranda : ℝ := 2

-- Theorem statement based on the question and conditions
theorem area_of_veranda:
  let length_total := length_room + 2 * width_veranda,
      width_total := width_room + 2 * width_veranda,
      area_total := length_total * width_total,
      area_room := length_room * width_room,
      area_veranda := area_total - area_room
  in area_veranda = 148 := by
  sorry

end area_of_veranda_l441_441448


namespace solve_expression_l441_441675

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem solve_expression : f (g 3) - g (f 3) = -5 := by
  sorry

end solve_expression_l441_441675


namespace find_m_l441_441586

theorem find_m (m : ℝ) : 
  let a := (1, 2, -2)
  let b := (-2, 3, m)
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0 -> m = 2 :=
begin
  let a := (1 : ℝ, 2 : ℝ, -2 : ℝ),
  let b := (-2 : ℝ, 3 : ℝ, m),
  assume h: a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0,
  sorry
end

end find_m_l441_441586


namespace perfect_squares_good_l441_441484

def good_set (A : set ℕ) : Prop :=
∀ n > 0, ∃! p : ℕ, prime p ∧ (n - p) ∈ A

def perfect_squares : set ℕ := {n | ∃ m : ℕ, n = m * m}

theorem perfect_squares_good : good_set perfect_squares :=
sorry

end perfect_squares_good_l441_441484


namespace isosceles_triangle_l441_441823

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441823


namespace final_trip_l441_441329

-- Definitions for the conditions
def dwarf := String
def snow_white := "Snow White" : dwarf
def Happy : dwarf := "Happy"
def Grumpy : dwarf := "Grumpy"
def Dopey : dwarf := "Dopey"
def Bashful : dwarf := "Bashful"
def Sleepy : dwarf := "Sleepy"
def Doc : dwarf := "Doc"
def Sneezy : dwarf := "Sneezy"

-- The dwarfs lineup from left to right
def lineup : List dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- The boat can hold Snow White and up to 3 dwarfs
def boat_capacity (load : List dwarf) : Prop := snow_white ∈ load ∧ load.length ≤ 4

-- Any two dwarfs standing next to each other in the original lineup will quarrel if left without Snow White
def will_quarrel (d1 d2 : dwarf) : Prop :=
  (d1, d2) ∈ (lineup.zip lineup.tail)

-- The objective: Prove that on the last trip, Snow White will take Grumpy, Bashful, and Doc
theorem final_trip : ∃ load : List dwarf, 
  set.load ⊆ {snow_white, Grumpy, Bashful, Doc} ∧ boat_capacity load :=
  sorry

end final_trip_l441_441329


namespace value_of_x_l441_441376

variable (x y z : ℕ)

theorem value_of_x : x = 10 :=
  assume h1 : x = y / 2,
  assume h2 : y = z / 4,
  assume h3 : z = 80,
  sorry

end value_of_x_l441_441376


namespace nat_pair_solution_l441_441127

theorem nat_pair_solution (x y : ℕ) : 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by
  sorry

end nat_pair_solution_l441_441127


namespace probability_is_one_eighteenth_l441_441145

noncomputable def num_faces := 6
def num_dice := 4

def total_outcomes : ℕ := num_faces ^ num_dice

def favorable_sequences_one : List (List ℕ) := [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
def favorable_sequences_two : List (List ℕ) := []  -- No valid sequences in range

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def arrangements (l : List ℕ) : ℕ := factorial l.length

def total_favorable_outcomes : ℕ :=
  favorable_sequences_one.length * arrangements (favorable_sequences_one.head!) + 
  favorable_sequences_two.length * arrangements (favorable_sequences_two.head!)

def probability_favorable : ℚ := total_favorable_outcomes / total_outcomes

theorem probability_is_one_eighteenth : probability_favorable = 1 / 18 :=
  by
    rw [probability_favorable, total_outcomes, total_favorable_outcomes]
    have : total_outcomes = 6 ^ 4 := rfl
    have : factorial 4 = 24 := rfl
    have : favorable_sequences_one.length = 3 := rfl
    have : arrangements [1, 2, 3, 4] = 24 := rfl
    have : favorable_sequences_two.length = 0 := rfl
    normalize_num
    sorry

end probability_is_one_eighteenth_l441_441145


namespace triangle_ABC_is_isosceles_l441_441909

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441909


namespace bruce_total_amount_paid_l441_441443

-- Definitions for quantities and rates
def quantity_of_grapes : Nat := 8
def rate_per_kg_grapes : Nat := 70
def quantity_of_mangoes : Nat := 11
def rate_per_kg_mangoes : Nat := 55

-- Calculate individual costs
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes

-- Calculate total amount paid
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Statement to prove
theorem bruce_total_amount_paid : total_amount_paid = 1165 := by
  -- Proof is intentionally left as a placeholder
  sorry

end bruce_total_amount_paid_l441_441443


namespace symmetric_y_axis_l441_441609

theorem symmetric_y_axis (θ : ℝ) :
  (cos θ = -cos (θ + π / 6)) ∧ (sin θ = sin (θ + π / 6)) → θ = 5 * π / 12 :=
by
  sorry

end symmetric_y_axis_l441_441609


namespace value_of_x_l441_441428

theorem value_of_x (x : ℝ) : 8^4 + 8^4 + 8^4 = 2^x → x = Real.log 3 / Real.log 2 + 12 :=
by
  sorry

end value_of_x_l441_441428


namespace part1_part2_l441_441031

-- Part 1
theorem part1 :
  let a : ℚ := 2 + 1/4,
      b : ℚ := (-9.6 : ℚ),
      c : ℚ := 3 + 3/8,
      d : ℚ := 1.5
  in (a ^ (3/2) - b ^ 0 - c ^ (2/3) + d ^ (-2)) = (1/2) :=
by
  let a : ℚ := 9 / 4,
      b : ℚ := -9.6,
      c : ℚ := 27 / 8,
      d : ℚ := 1.5
  calc
    a ^ (3/2) - b ^ 0 - c ^ (2/3) + d ^ (-2)
    = (9/4) ^ (3/2) - (-9.6) ^ 0 - (27/8) ^ (2/3) + (1.5) ^ (-2) : by sorry
    = 1/2 : by sorry

-- Part 2
theorem part2 {a b m : ℝ} (h₁ : (2 : ℝ) ^ a = (5 : ℝ) ^ b) (h₂ : (1 / a) + (1 / b) = 2) :
  m = Real.sqrt 10 :=
by
  have ha := Real.log (2 ^ a),
  have hb := Real.log (5 ^ b),
  calc
    m = Real.sqrt 10 : by sorry
  
sorry

end part1_part2_l441_441031


namespace sum_of_vectors_eq_zero_l441_441157

-- Define points and vectors
variables {Point : Type*}
variables (V : Type*) [AddCommGroup V] [Module ℝ V]

-- Define the vector between each pair of points
variables (A B : Point)
variables (vector : Point → Point → V)

-- Define the conditions
variables (points : set Point)
variables (pairs : set (Point × Point))
variables (condition : ∀ p : Point, 
    (point.count {(A, B) ∈ pairs | A = p} = point.count {(A, B) ∈ pairs | B = p}))

-- Define the sum of all chosen vectors
def sum_of_vectors (pairs : set (Point × Point)) : V := 
  ∑ (A, B) in pairs, vector A B

-- The theorem statement
theorem sum_of_vectors_eq_zero 
    (points : set Point) 
    (pairs : set (Point × Point)) 
    (condition : ∀ p : Point, 
        (point.count {(A, B) ∈ pairs | A = p} = point.count {(A, B) ∈ pairs | B = p})) 
    : sum_of_vectors pairs = 0 := 
  sorry

end sum_of_vectors_eq_zero_l441_441157


namespace triangle_isosceles_if_parallel_l441_441868

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441868


namespace geometric_sequence_a5_l441_441227

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 * a 5 = 16) (h2 : a 4 = 8) (h3 : ∀ n, a n > 0) : a 5 = 16 := 
by
  sorry

end geometric_sequence_a5_l441_441227


namespace female_democrats_count_l441_441379

variable (F M : ℕ)
def total_participants : Prop := F + M = 720
def female_democrats (D_F : ℕ) : Prop := D_F = 1 / 2 * F
def male_democrats (D_M : ℕ) : Prop := D_M = 1 / 4 * M
def total_democrats (D_F D_M : ℕ) : Prop := D_F + D_M = 1 / 3 * 720

theorem female_democrats_count
  (F M D_F D_M : ℕ)
  (h1 : total_participants F M)
  (h2 : female_democrats F D_F)
  (h3 : male_democrats M D_M)
  (h4 : total_democrats D_F D_M) :
  D_F = 120 :=
sorry

end female_democrats_count_l441_441379


namespace system_infinite_solutions_a_eq_neg2_l441_441178

theorem system_infinite_solutions_a_eq_neg2 
  (x y a : ℝ)
  (h1 : 2 * x + 2 * y = -1)
  (h2 : 4 * x + a^2 * y = a) 
  (infinitely_many_solutions : ∃ (a : ℝ), ∀ (c : ℝ), 4 * x + a^2 * y = c) :
  a = -2 :=
by
  sorry

end system_infinite_solutions_a_eq_neg2_l441_441178


namespace find_number_l441_441933

theorem find_number :
  ∃ x : ℝ, (10 + x + 60) / 3 = (10 + 40 + 25) / 3 + 5 ∧ x = 20 :=
by
  sorry

end find_number_l441_441933


namespace rectangle_reassembly_l441_441277

theorem rectangle_reassembly (a b : ℝ) (S : ℝ) (h_ab : a * b = S) (c d : ℝ) (h_cd : c * d = S) (h_a_le_b : a ≤ b) (h_c_le_d : c ≤ d) (h_s_le_b : real.sqrt S ≤ b) (h_a_le_d : a ≤ d) :
  ∃ (a' b' : ℝ), a' = 1 ∧ a' * b' = S :=
sorry

end rectangle_reassembly_l441_441277


namespace triangle_isosceles_if_parallel_l441_441865

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441865


namespace fraction_simplified_l441_441719

-- Define the fraction function
def fraction (n : ℕ) := (21 * n + 4, 14 * n + 3)

-- Define the gcd function to check if fractions are simplified.
def is_simplified (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Main theorem
theorem fraction_simplified (n : ℕ) : is_simplified (21 * n + 4) (14 * n + 3) :=
by
  -- Rest of the proof
  sorry

end fraction_simplified_l441_441719


namespace ball_min_bounces_reach_target_height_l441_441999

noncomputable def minimum_bounces (initial_height : ℝ) (ratio : ℝ) (target_height : ℝ) : ℕ :=
  Nat.ceil (Real.log (target_height / initial_height) / Real.log ratio)

theorem ball_min_bounces_reach_target_height :
  minimum_bounces 20 (2 / 3) 2 = 6 :=
by
  -- This is where the proof would go, but we use sorry to skip it
  sorry

end ball_min_bounces_reach_target_height_l441_441999


namespace largest_integer_not_sum_of_30_and_composite_l441_441405

theorem largest_integer_not_sum_of_30_and_composite : 
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b < 30 ∧ ¬ prime b ∧ (n = 30 * a + b) → n = 157 :=
by
  sorry

end largest_integer_not_sum_of_30_and_composite_l441_441405


namespace percent_hispanics_in_west_is_39_l441_441089

-- Define the population data
structure PopulationData where
  NE : ℕ
  MW : ℕ
  South : ℕ
  West : ℕ

-- Define the specific populations for each ethnic group
def White : PopulationData := ⟨50, 60, 70, 40⟩
def Black : PopulationData := ⟨6, 6, 18, 3⟩
def Asian : PopulationData := ⟨2, 2, 2, 5⟩
def Hispanic : PopulationData := ⟨4, 5, 10, 12⟩

-- Total population
def totalHispanicPopulation : ℕ := Hispanic.NE + Hispanic.MW + Hispanic.South + Hispanic.West

-- Population from the West
def westHispanicPopulation : ℕ := Hispanic.West

-- Calculate percentage
def percentageHispanicInWest : ℚ := (westHispanicPopulation : ℚ) / (totalHispanicPopulation : ℚ) * 100

-- Lean theorem statement
theorem percent_hispanics_in_west_is_39 :
  percentageHispanicInWest ≈ 39 :=
by
  sorry

end percent_hispanics_in_west_is_39_l441_441089


namespace seminar_total_cost_l441_441059

theorem seminar_total_cost 
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ) 
  (food_allowance_per_teacher : ℝ)
  (total_cost : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.05)
  (h3 : num_teachers = 10) 
  (h4 : food_allowance_per_teacher = 10)
  (h5 : total_cost = regular_fee * num_teachers * (1 - discount_rate) + food_allowance_per_teacher * num_teachers) :
  total_cost = 1525 := 
sorry

end seminar_total_cost_l441_441059


namespace isosceles_triangle_of_parallel_l441_441771

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441771


namespace smallest_k_for_zero_l441_441140

noncomputable def v (n : ℕ) : ℕ := n^2 + 2 * n

def Δ1 (v : ℕ → ℕ) (n : ℕ) : ℕ := v (n + 1) - v n
def Δk (k : ℕ) (v : ℕ → ℕ) (n : ℕ) : ℕ :=
  if k = 1 then Δ1 v n else Δ1 (Δk (k - 1) v) n

theorem smallest_k_for_zero (n : ℕ) : (k : ℕ) (h1 : ∀ n, Δk k v n = 0) → k = 3 := by
  sorry

end smallest_k_for_zero_l441_441140


namespace no_2003_segments_with_exactly_3_intersections_l441_441503

theorem no_2003_segments_with_exactly_3_intersections :
  ¬ ∃ (S : Finset (Set (ℝ × ℝ))) (h: S.card = 2003), ∀ s ∈ S, (Finset.filter (λ t, t ≠ s ∧ (s ∩ t).Nonempty) S).card = 3 := sorry

end no_2003_segments_with_exactly_3_intersections_l441_441503


namespace max_edges_triangle_free_max_triangles_in_edges_l441_441256

-- Definitions for the first part
def triangle_free (G : SimpleGraph V) : Prop :=
  ∀ (u v w : V), u ≠ v → u ≠ w → v ≠ w → G.Adj u v → G.Adj u w → G.Adj v w → false

theorem max_edges_triangle_free {V : Type*} (G : SimpleGraph V) (n : ℕ) [Fintype V] [DecidableRel G.Adj]
  (h_card : Fintype.card V = n) (h_tri_free : triangle_free G) : G.edgeFinset.card ≤ (n^2) / 4 := sorry

-- Definitions for the bonus part
theorem max_triangles_in_edges (k : ℕ) : ∃ G : SimpleGraph (Fin (binom k 2)), triangle_free G ∧ G.edgeFinset.card = k :=
sorry

end max_edges_triangle_free_max_triangles_in_edges_l441_441256


namespace triangle_ABC_is_isosceles_l441_441912

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441912


namespace similar_figures_rectangles_l441_441011

theorem similar_figures_rectangles :
  ¬(∀ (r1 r2 : Rectangle), similar r1 r2) ∧
  (∀ (s1 s2 : Square), similar s1 s2) ∧
  (∀ (e1 e2 : EquilateralTriangle), similar e1 e2) ∧
  (∀ (i1 i2 : IsoscelesRightTriangle), similar i1 i2) :=
by
  sorry

end similar_figures_rectangles_l441_441011


namespace angle_sum_equilateral_triangle_l441_441571

theorem angle_sum_equilateral_triangle 
  (ABC : Type) [triangle ABC] (A B C K L M : ABC)
  (h_equilateral : is_equilateral_triangle ABC A B C)
  (h_divide_BC : divides_in_three_equal_parts B C K L)
  (h_divide_AC : divides_in_ratio A C M (1 : ℕ) (2 : ℕ)):
  angle_sum (A K M) (A L M) = 30 :=
by
  sorry

end angle_sum_equilateral_triangle_l441_441571


namespace first_tier_price_level_l441_441106

-- Variables for car price and tax values
def car_price : ℝ := 30000
def total_tax : ℝ := 5500

-- Variables for tax rates and the unknown tier level
def tax_rate_tier1 : ℝ := 0.25
def tax_rate_tier2 : ℝ := 0.15

theorem first_tier_price_level :
  ∃ P : ℝ, (tax_rate_tier1 * P + tax_rate_tier2 * (car_price - P) = total_tax) ∧ P = 10000 :=
by
  existsi (10000 : ℝ)
  split
  -- Prove that the tax calculation holds
  · sorry
  -- Show P equals 10000
  · rfl

end first_tier_price_level_l441_441106


namespace min_Sn_is_at_n_10_l441_441569

open Classical

variable {α : Type*} [OrderedRing α]

def is_arithmetic_sequence (a : ℕ → α) (d : α) := ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α := ∑ i in Finset.range n, a i

noncomputable def Sn (a : ℕ → α) (n : ℕ) : α := sum_of_first_n_terms a n

noncomputable def arithmetic_sequence_minimized_n (a : ℕ → α) (d : α) (H : is_arithmetic_sequence a d)
  (H1 : a 0 < 0) (H2 : 0 < d) (H3 : Sn a 20 / a 9 < 0) : ℕ :=
  10

theorem min_Sn_is_at_n_10 (a : ℕ → α) (d : α) (H : is_arithmetic_sequence a d)
  (H1 : a 0 < 0) (H2 : 0 < d) (H3 : Sn a 20 / a 9 < 0) :
  (∃ n, Sn a n < Sn a 10) → n = 10 :=
sorry

end min_Sn_is_at_n_10_l441_441569


namespace range_of_g_l441_441543

noncomputable def g (x : ℝ) : ℝ := 
  (sin x)^3 + 4 * (sin x)^2 + 3 * sin x + 2 * (cos x)^2 - 6 / (sin x - 1)

theorem range_of_g : { y : ℝ | ∃ x : ℝ, g x = y ∧ sin x ≠ 1 } = set.Icc 1 9 :=
by
  sorry

end range_of_g_l441_441543


namespace infinite_composites_in_sequence_l441_441080

-- Defining the sequence of digits
def sequence : ℕ → ℕ := sorry

-- Defining what it means for a sequence to be digits from 0 to 8
def is_digit (d : ℕ) : Prop := d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8}

-- Every element in the sequence is a digit from 0 to 8
axiom sequence_is_digits : ∀ n, is_digit (sequence n)

-- Defining the number formed by concatenation of first n elements
def number_formed (n : ℕ) : ℕ :=
  ∑ i in (finset.range n), (sequence i) * (10 ^ i)

-- Predicate to check if a number is composite
def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, (m > 1 ∧ k > 1 ∧ m * k = n)

-- The main statement to be proved
theorem infinite_composites_in_sequence :
  ∃ᶠ n in at_top, is_composite (number_formed n) :=
sorry

end infinite_composites_in_sequence_l441_441080


namespace isosceles_triangle_l441_441757

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441757


namespace sine_not_increasing_in_first_and_fourth_quadrants_l441_441945

def f (x : ℝ) : ℝ := Real.sin x

theorem sine_not_increasing_in_first_and_fourth_quadrants :
  ∃ (x1 x2 : ℝ), 
  ((0 ≤ x1 ∧ x1 ≤ (π / 2)) ∨ (-π / 2 ≤ x1 ∧ x1 ≤ 0)) ∧
  ((0 ≤ x2 ∧ x2 ≤ (π / 2)) ∨ (-π / 2 ≤ x2 ∧ x2 ≤ 0)) ∧
  x1 < x2 ∧ f x1 ≥ f x2 :=
sorry

end sine_not_increasing_in_first_and_fourth_quadrants_l441_441945


namespace number_of_five_dollar_bills_l441_441266

theorem number_of_five_dollar_bills (total_money denomination expected_bills : ℕ) 
  (h1 : total_money = 45) 
  (h2 : denomination = 5) 
  (h3 : expected_bills = total_money / denomination) : 
  expected_bills = 9 :=
by
  sorry

end number_of_five_dollar_bills_l441_441266


namespace relay_race_arrangements_l441_441286

theorem relay_race_arrangements :
  let boys := 6
      girls := 2
      totalSelections := 4
      selectedBoys := 3
      selectedGirls := 1 in 
  (nat.choose girls selectedGirls) * (nat.choose boys selectedBoys) * selectedBoys *
  (nat.perm 3 3) = 720 := sorry

end relay_race_arrangements_l441_441286


namespace final_trip_theorem_l441_441294

/-- Define the lineup of dwarfs -/
inductive Dwarf where
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

open Dwarf

/-- Define the conditions -/
-- The dwarfs initially standing next to each other
def adjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = Happy ∧ d2 = Grumpy) ∨
  (d1 = Grumpy ∧ d2 = Dopey) ∨
  (d1 = Dopey ∧ d2 = Bashful) ∨
  (d1 = Bashful ∧ d2 = Sleepy) ∨
  (d1 = Sleepy ∧ d2 = Doc) ∨
  (d1 = Doc ∧ d2 = Sneezy)

-- Snow White is the only one who can row
constant snowWhite_can_row : Prop := true

-- The boat can hold Snow White and up to 3 dwarfs
constant boat_capacity : ℕ := 4

-- Define quarrel if left without Snow White
def quarrel_without_snowwhite (d1 d2 : Dwarf) : Prop := adjacent d1 d2

-- Define the final trip setup
def final_trip (dwarfs : List Dwarf) : Prop :=
  dwarfs = [Grumpy, Bashful, Doc]

-- Theorem to prove the final trip
theorem final_trip_theorem : ∃ dwarfs, final_trip dwarfs :=
  sorry

end final_trip_theorem_l441_441294


namespace isosceles_triangle_of_parallel_l441_441783

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441783


namespace triangle_ABC_isosceles_l441_441836

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441836


namespace find_a_prime_l441_441197

open Classical

noncomputable theory

-- Define a predicate to check if a number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

variable (a p q : ℕ)

-- Define the theorem with given conditions and prove the required result.
theorem find_a_prime (h1 : is_prime a) (h2 : is_prime p) (h3 : is_prime q)
  (h4 : a < p) (h5 : a + p = q) : a = 2 :=
by
  sorry

end find_a_prime_l441_441197


namespace min_value_of_quadratic_l441_441522

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 9

theorem min_value_of_quadratic : ∃ (x : ℝ), f x = 6 :=
by sorry

end min_value_of_quadratic_l441_441522


namespace time_B_alone_l441_441038

noncomputable def work_rate_A := 1 / 4
noncomputable def work_rate_D := 1 / 5

axiom work_rate_A_C : work_rate_A + C_work = 1 / 2
axiom work_rate_B_C : B_work + C_work = 1 / 3
axiom work_rate_all : work_rate_A + B_work + C_work + work_rate_D = 1

theorem time_B_alone :
  let B_work := 13 / 60
  let B_time := 60 / 13
  B_time ≈ 4.62 := sorry

end time_B_alone_l441_441038


namespace Marcia_wardrobe_cost_l441_441694

-- Definitions from the problem
def skirt_price : ℝ := 20
def blouse_price : ℝ := 15
def pant_price : ℝ := 30

def num_skirts : ℕ := 3
def num_blouses : ℕ := 5
def num_pants : ℕ := 2

-- The main theorem statement
theorem Marcia_wardrobe_cost :
  (num_skirts * skirt_price) + (num_blouses * blouse_price) + (pant_price + (pant_price / 2)) = 180 :=
by
  sorry

end Marcia_wardrobe_cost_l441_441694


namespace n_minus_m_l441_441677

variable (m n : ℕ)

def is_congruent_to_5_mod_13 (x : ℕ) : Prop := x % 13 = 5
def is_smallest_three_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 100 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 100 → x ≤ y

def is_smallest_four_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 1000 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 1000 → x ≤ y

theorem n_minus_m
  (h₁ : is_smallest_three_digit_integer_congruent_to_5_mod_13 m)
  (h₂ : is_smallest_four_digit_integer_congruent_to_5_mod_13 n) :
  n - m = 897 := sorry

end n_minus_m_l441_441677


namespace isosceles_triangle_of_parallel_l441_441778

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441778


namespace find_coefficients_l441_441134

theorem find_coefficients (c d : ℝ)
  (h : ∃ u v : ℝ, u ≠ v ∧ (u^3 + c * u^2 + 10 * u + 4 = 0) ∧ (v^3 + c * v^2 + 10 * v + 4 = 0)
     ∧ (u^3 + d * u^2 + 13 * u + 5 = 0) ∧ (v^3 + d * v^2 + 13 * v + 5 = 0)) :
  (c, d) = (7, 8) :=
by
  sorry

end find_coefficients_l441_441134


namespace geometric_series_3000_terms_sum_l441_441962

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_series_3000_terms_sum
    (a r : ℝ)
    (h_r : r ≠ 1)
    (sum_1000 : geometric_sum a r 1000 = 500)
    (sum_2000 : geometric_sum a r 2000 = 950) :
  geometric_sum a r 3000 = 1355 :=
by 
  sorry

end geometric_series_3000_terms_sum_l441_441962


namespace prove_k_in_terms_of_x_l441_441342

variables {A B k x : ℝ}

-- given conditions
def positive_numbers (A B : ℝ) := A > 0 ∧ B > 0
def ratio_condition (A B k : ℝ) := A = k * B
def percentage_condition (A B x : ℝ) := A = B + (x / 100) * B

-- proof statement
theorem prove_k_in_terms_of_x (A B k x : ℝ) (h1 : positive_numbers A B) (h2 : ratio_condition A B k) (h3 : percentage_condition A B x) (h4 : k > 1) :
  k = 1 + x / 100 :=
sorry

end prove_k_in_terms_of_x_l441_441342


namespace janet_lost_16_lives_l441_441661

variable (x : ℕ)
variable (initial_lives gained_lives final_lives : ℕ)
variable (lost_lives : ℕ)

axiom initial_lives_val : initial_lives = 38
axiom gained_lives_val : gained_lives = 32
axiom final_lives_val : final_lives = 54
axiom lost_lives_val : lost_lives = initial_lives - x
axiom total_lives_after_gain : lost_lives + gained_lives = final_lives

theorem janet_lost_16_lives : x = 16 :=
by
  have h1 : initial_lives = 38 := initial_lives_val
  have h2 : gained_lives = 32 := gained_lives_val
  have h3 : final_lives = 54 := final_lives_val
  have h4 : lost_lives = initial_lives - x := lost_lives_val
  have h5 : lost_lives + gained_lives = final_lives := total_lives_after_gain
  sorry

end janet_lost_16_lives_l441_441661


namespace geometric_sum_2015_2016_l441_441546

theorem geometric_sum_2015_2016 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 2)
  (h_a2_a5 : a 2 + a 5 = 0)
  (h_Sn : ∀ n, S n = (1 - (-1)^n)) :
  S 2015 + S 2016 = 2 :=
by sorry

end geometric_sum_2015_2016_l441_441546


namespace find_m_l441_441192

noncomputable def isPerfectSquareTrinomial (f : ℤ → ℤ) : Prop :=
  ∃ (a b : ℤ), f = λ x, a^2 * x^2 + 2 * a * b * x + b^2

theorem find_m (m : ℤ) : isPerfectSquareTrinomial (λ x : ℤ, x^2 + m * x + 16) → (m = 8 ∨ m = -8) :=
by
  sorry

end find_m_l441_441192


namespace isosceles_triangle_of_parallel_l441_441888

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441888


namespace m_range_l441_441243

noncomputable def G (x : ℝ) (m : ℝ) : ℝ := (8 * x^2 + 22 * x + 5 * m) / 8

theorem m_range (m : ℝ) : 2.5 ≤ m ∧ m ≤ 3.5 ↔ m = 121 / 40 := by
  sorry

end m_range_l441_441243


namespace triangle_ABC_isosceles_l441_441839

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441839


namespace cost_of_graveling_per_sq_meter_l441_441476

theorem cost_of_graveling_per_sq_meter
    (length_lawn : ℝ) (breadth_lawn : ℝ)
    (width_road : ℝ) (total_cost_gravel : ℝ)
    (length_road_area : ℝ) (breadth_road_area : ℝ) (intersection_area : ℝ)
    (total_graveled_area : ℝ) (cost_per_sq_meter : ℝ) :
    length_lawn = 55 →
    breadth_lawn = 35 →
    width_road = 4 →
    total_cost_gravel = 258 →
    length_road_area = length_lawn * width_road →
    intersection_area = width_road * width_road →
    breadth_road_area = breadth_lawn * width_road - intersection_area →
    total_graveled_area = length_road_area + breadth_road_area →
    cost_per_sq_meter = total_cost_gravel / total_graveled_area →
    cost_per_sq_meter = 0.75 :=
by
  intros
  sorry

end cost_of_graveling_per_sq_meter_l441_441476


namespace isosceles_triangle_of_parallel_l441_441782

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441782


namespace probability_Q_between_lines_is_2_over_3_l441_441271

noncomputable def probability_between_lines (a b : ℕ) (hrelprime : Nat.coprime a b) : ℕ :=
100 * a + b

def evenly_spaced_parallel_lines (l1 l2 l3 l4 : Line) (d : ℝ) : Prop :=
DistanceBetweenLines l1 l2 = d ∧ DistanceBetweenLines l2 l3 = d ∧ DistanceBetweenLines l3 l4 = d

def square_with_points_on_lines (A C : Point) (l1 l4 : Line) : Prop :=
A ∈ l1 ∧ C ∈ l4 ∧ IsSquare (A, B, C, D)

def uniformly_random_point_in_square (P : Point) (A B C D : Point) : Prop :=
IsInInteriorOfSquare P A B C D

def uniformly_random_point_on_perimeter (Q : Point) (A B C D : Point) : Prop :=
IsOnPerimeterOfSquare Q A B C D

theorem probability_Q_between_lines_is_2_over_3
  (l1 l2 l3 l4 : Line) (P Q : Point) (A B C D : Point) (a b : ℕ)
  (hp1 : evenly_spaced_parallel_lines l1 l2 l3 l4 (1/3))
  (hp2 : square_with_points_on_lines A C l1 l4)
  (hp3 : uniformly_random_point_in_square P A B C D)
  (hp4 : uniformly_random_point_on_perimeter Q A B C D)
  (hp5 : probability (IsBetween P l2 l3) = 53 / 100)
  (hrelprime : Nat.coprime a b)
  : probability_between_lines 2 3 hrelprime = 206 := 
sorry

end probability_Q_between_lines_is_2_over_3_l441_441271


namespace AC_diagonal_length_l441_441217

noncomputable def AC_length (AD DC : ℝ) (angle_ADC : ℝ) : ℝ :=
  Real.sqrt (AD^2 + DC^2 - 2 * AD * DC * Real.cos angle_ADC)

theorem AC_diagonal_length :
  let AD := 15
  let DC := 15
  let angle_ADC := 2 * Real.pi / 3 -- 120 degrees in radians
  AC_length AD DC angle_ADC = 15 :=
by
  have h : AC_length 15 15 (2 * Real.pi / 3) = Real.sqrt (15^2 + 15^2 - 2 * 15 * 15 * Real.cos (2 * Real.pi / 3)),
  { unfold AC_length },
  rw h,
  have h_cos : Real.cos (2 * Real.pi / 3) = -1 / 2,
  { sorry }, -- intermediate steps to find cosine of 120 degrees
  rw [h_cos, sq],
  norm_num,
  refl

end AC_diagonal_length_l441_441217


namespace final_grade_calculation_l441_441483

theorem final_grade_calculation
  (exam_score homework_score class_participation_score : ℝ)
  (exam_weight homework_weight participation_weight : ℝ)
  (h_exam_score : exam_score = 90)
  (h_homework_score : homework_score = 85)
  (h_class_participation_score : class_participation_score = 80)
  (h_exam_weight : exam_weight = 3)
  (h_homework_weight : homework_weight = 2)
  (h_participation_weight : participation_weight = 5) :
  (exam_score * exam_weight + homework_score * homework_weight + class_participation_score * participation_weight) /
  (exam_weight + homework_weight + participation_weight) = 84 :=
by
  -- The proof would go here
  sorry

end final_grade_calculation_l441_441483


namespace possible_denominators_count_l441_441343

theorem possible_denominators_count 
  (a b : ℕ) 
  (h1 : 0 ≤ a) (h2 : a ≤ 9) 
  (h3 : 0 ≤ b) (h4 : b ≤ 9) 
  (h5 : ¬ (a = 5 ∧ b = 5))
  (h6 : ¬ (a = 0 ∧ b = 0)) : 
  finset.card (finset.filter (λ d, ∃ k : ℕ, 0.\overline{k} * d = ab) 
  ({1, 2, 5, 10, 11, 22, 55, 110} : finset ℕ)) = 6 := 
by sorry

end possible_denominators_count_l441_441343


namespace ladder_length_l441_441991

theorem ladder_length (θ : ℝ) (d : ℝ) (cos_val : ℝ) (hypotenuse : ℝ) 
  (h1 : θ = 60)
  (h2 : d = 4.6)
  (h3 : cos θ = cos_val)
  (h4 : cos_val = 0.5)
  (h5 : hypotenuse = d / cos_val) :
  hypotenuse = 9.2 :=
by
  -- This is where the proof would go.
  sorry

end ladder_length_l441_441991


namespace sum_of_extrema_of_expression_l441_441151

theorem sum_of_extrema_of_expression (x y : ℝ) (h₁ : 1 ≤ x^2 + y^2) (h₂ : x^2 + y^2 ≤ 4) :
  let f z t := z^2 - z*t + t^2 in
  let min_value := 1/2 in
  let max_value := 6 in
  min_value + max_value = 13/2 :=
sorry

end sum_of_extrema_of_expression_l441_441151


namespace emily_total_points_l441_441528

def score_round_1 : ℤ := 16
def score_round_2 : ℤ := 33
def score_round_3 : ℤ := -25
def score_round_4 : ℤ := 46
def score_round_5 : ℤ := 12
def score_round_6 : ℤ := 30 - (2 * score_round_5 / 3)

def total_score : ℤ :=
  score_round_1 + score_round_2 + score_round_3 + score_round_4 + score_round_5 + score_round_6

theorem emily_total_points : total_score = 104 := by
  sorry

end emily_total_points_l441_441528


namespace smallest_integer_gcd_6_l441_441427

theorem smallest_integer_gcd_6 : ∃ n : ℕ, n > 100 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n :=
by
  let n := 114
  have h1 : n > 100 := sorry
  have h2 : gcd n 18 = 6 := sorry
  have h3 : ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n := sorry
  exact ⟨n, h1, h2, h3⟩

end smallest_integer_gcd_6_l441_441427


namespace angle_AOB_eq_pi_div_2_max_area_EGFH_l441_441561

-- Statement for problem (1)
theorem angle_AOB_eq_pi_div_2 (k : ℝ) : 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ A.1 ^ 2 + A.2 ^ 2 = 2 ∧ B.1 ^ 2 + B.2 ^ 2 = 2 ∧ 
   ((B.2 - A.2) / (B.1 - A.1) = k) ∧ ∠ ((0, 0) : ℝ × ℝ) A B = π / 2) →
  k = sqrt 3 ∨ k = -sqrt 3 :=
  sorry

-- Statement for problem (2)
theorem max_area_EGFH (M : ℝ × ℝ) :
  M = (1, sqrt 2 / 2) →
  (EF GH : ℝ) →
  (∀ d1 d2 : ℝ, d1^2 + d2^2 = 3 / 2 → 
  let area := 2 * sqrt (2 - d1^2) * 2 * sqrt (2 - d2^2) in
  area ≤ 5 / 2 ∧ 
  (∀ d1 d2, d1 = sqrt 3 / 2 ∧ d2 = sqrt 3 / 2 → area = 5 / 2)
  ) :=
  sorry

end angle_AOB_eq_pi_div_2_max_area_EGFH_l441_441561


namespace minimum_value_fraction_l441_441574

theorem minimum_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 3) : 
  (2 / x + 1 / y ≥ 3) :=
begin
  sorry,
end

end minimum_value_fraction_l441_441574


namespace radius_is_correct_l441_441450

noncomputable def radius_of_hemisphere (V : ℝ) : ℝ :=
  (3 * V / (2 * Real.pi))^(1/3)

theorem radius_is_correct (h : radius_of_hemisphere 19404 ≈ 21.02) : true :=
  sorry

end radius_is_correct_l441_441450


namespace snow_white_last_trip_l441_441306

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l441_441306


namespace painting_time_l441_441124

theorem painting_time (n₁ t₁ n₂ t₂ : ℕ) (h1 : n₁ = 8) (h2 : t₁ = 12) (h3 : n₂ = 6) (h4 : n₁ * t₁ = n₂ * t₂) : t₂ = 16 :=
by
  sorry

end painting_time_l441_441124


namespace symmetric_points_y_axis_l441_441606

theorem symmetric_points_y_axis (θ : ℝ) :
  (cos θ = -cos (θ + π / 6)) ∧ (sin θ = sin (θ + π / 6)) → θ = 5 * π / 12 :=
by
  -- Proof omitted
  sorry

end symmetric_points_y_axis_l441_441606


namespace problem_statement_l441_441735

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441735


namespace a_beats_b_by_32_meters_l441_441440

-- Define the known conditions.
def distance_a_in_t : ℕ := 224 -- Distance A runs in 28 seconds
def time_a : ℕ := 28 -- Time A takes to run 224 meters
def distance_b_in_t : ℕ := 224 -- Distance B runs in 32 seconds
def time_b : ℕ := 32 -- Time B takes to run 224 meters

-- Define the speeds.
def speed_a : ℕ := distance_a_in_t / time_a
def speed_b : ℕ := distance_b_in_t / time_b

-- Define the distances each runs in 32 seconds.
def distance_a_in_32_sec : ℕ := speed_a * 32
def distance_b_in_32_sec : ℕ := speed_b * 32

-- The proof statement
theorem a_beats_b_by_32_meters :
  distance_a_in_32_sec - distance_b_in_32_sec = 32 := 
sorry

end a_beats_b_by_32_meters_l441_441440


namespace molecular_weight_of_CaBr2_l441_441422

theorem molecular_weight_of_CaBr2 (mw_4moles : ℕ) (h : mw_4moles = 800) : ∃ mw : ℕ, mw = 200 :=
by
  use mw_4moles / 4
  have h1 : mw_4moles / 4 = 800 / 4, from congr_arg (λ x, x / 4) h
  norm_num at h1
  assumption

end molecular_weight_of_CaBr2_l441_441422


namespace fraction_simplification_l441_441354

theorem fraction_simplification :
  ∃ (p q : ℕ), p = 2021 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (1011 / 1010) - (1010 / 1011) = (p : ℚ) / q := 
sorry

end fraction_simplification_l441_441354


namespace triangle_is_isosceles_l441_441900

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441900


namespace seminar_total_cost_l441_441060

theorem seminar_total_cost 
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ) 
  (food_allowance_per_teacher : ℝ)
  (total_cost : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.05)
  (h3 : num_teachers = 10) 
  (h4 : food_allowance_per_teacher = 10)
  (h5 : total_cost = regular_fee * num_teachers * (1 - discount_rate) + food_allowance_per_teacher * num_teachers) :
  total_cost = 1525 := 
sorry

end seminar_total_cost_l441_441060


namespace trirectangular_tetrahedron_max_volume_l441_441632

noncomputable def max_volume_trirectangular_tetrahedron (S : ℝ) : ℝ :=
  S^3 * (Real.sqrt 2 - 1)^3 / 162

theorem trirectangular_tetrahedron_max_volume
  (a b c : ℝ) (H : a > 0 ∧ b > 0 ∧ c > 0)
  (S : ℝ)
  (edge_sum :
    S = a + b + c + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2))
  : ∃ V, V = max_volume_trirectangular_tetrahedron S :=
by
  sorry

end trirectangular_tetrahedron_max_volume_l441_441632


namespace gerbil_sales_revenue_and_profit_l441_441055

noncomputable def N : ℕ := 450
noncomputable def purchase_price : ℝ := 8
noncomputable def original_price : ℝ := 12
noncomputable def percent_sold_weekend : ℝ := 0.35
noncomputable def discount_rate : ℝ := 0.20
noncomputable def tax_rate : ℝ := 0.05

theorem gerbil_sales_revenue_and_profit:
  let gerbils_sold_weekend := (percent_sold_weekend * N).floor
  let revenue_weekend := gerbils_sold_weekend * original_price
  let cost_weekend := gerbils_sold_weekend * purchase_price
  let profit := revenue_weekend - cost_weekend
  let sales_tax := revenue_weekend * tax_rate
  let total_revenue := revenue_weekend + sales_tax
  total_revenue = 1978.20 ∧ profit = 628 := 
by
  sorry

end gerbil_sales_revenue_and_profit_l441_441055


namespace perpendicular_squares_eq_conditions_l441_441681

variables {A B C D E F M N P Q R S : Type} [metric_space A] [metric_space B] [metric_space C]
[metric_space D] [metric_space E] [metric_space F]
-- Assumption that ground type should be a metric space is made here to define distances

-- Midpoints definition
def midpoint (a b : A) : A := sorry -- we can define midpoint based on the metric space

-- Hexagon vertices as type variables
axiom Hexagon (A B C D E F : Type) [midpoint A B = M] [midpoint B C = N] [midpoint C D = P] 
[midpoint D E = Q] [midpoint E F = R] [midpoint F A = S] 

-- Proof statement
theorem perpendicular_squares_eq_conditions :
  (M Q ∥ P S) ↔ (dist R N)^2 = (dist M Q)^2 + (dist P S)^2 := 
sorry

end perpendicular_squares_eq_conditions_l441_441681


namespace regular_polygon_sides_l441_441636

theorem regular_polygon_sides (interior_angle : ℝ) (h1 : interior_angle = 135) : 
  ∃ n : ℕ, n = 8 :=
by
  have exterior_angle := 180 - interior_angle
  have h2 : exterior_angle = 45 := by { rw [interior_angle, sub_eq_add_neg, sub_self, add_zero], norm_num2 } -- sorry if you don't want the output
  have n := 360 / exterior_angle
  exact ⟨8, by norm_num2⟩ -- sorry if you don't want the output

end regular_polygon_sides_l441_441636


namespace evaluate_expression_at_one_l441_441290

theorem evaluate_expression_at_one : 
  (4 + (4 + x^2) / x) / ((x + 2) / x) = 3 := by
  sorry

end evaluate_expression_at_one_l441_441290


namespace triangle_ABC_isosceles_l441_441837

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441837


namespace value_of_2_pow_b_l441_441624

theorem value_of_2_pow_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h1 : (2 ^ a) ^ b = 2 ^ 2) (h2 : 2 ^ a * 2 ^ b = 8) : 2 ^ b = 4 :=
by
  sorry

end value_of_2_pow_b_l441_441624


namespace isosceles_triangle_l441_441794

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441794


namespace BobsWalkingRate_l441_441708

def YolandaWalkingRate : ℝ := 8
def totalDistance : ℝ := 80
def distanceBobWalked : ℝ := 38.11764705882353
def yolandaExtraTime : ℝ := 1

theorem BobsWalkingRate 
  (YolandaWalkingRate = 8) 
  (totalDistance = 80) 
  (distanceBobWalked = 38.11764705882353) 
  (yolandaExtraTime = 1) :
  BobWalkingRate = 9 := 
by
  sorry

end BobsWalkingRate_l441_441708


namespace count_even_integers_between_400_and_700_with_digit_sum_18_l441_441185

def sum_of_digits (n : ℕ) : ℕ := 
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem count_even_integers_between_400_and_700_with_digit_sum_18 :
  (finset.filter (λ x => (x % 2 = 0) ∧ (sum_of_digits x = 18)) (finset.Ico 400 700)).card = 10 := 
sorry

end count_even_integers_between_400_and_700_with_digit_sum_18_l441_441185


namespace remaining_volume_of_cube_after_cylinder_removal_l441_441511

def cube_side_length := 6
def cylinder_radius := 3
def cylinder_height := cube_side_length

theorem remaining_volume_of_cube_after_cylinder_removal :
  let cube_volume := cube_side_length ^ 3 in
  let cylinder_volume := Real.pi * (cylinder_radius ^ 2) * cylinder_height in
  cube_volume - cylinder_volume = 216 - 54 * Real.pi :=
by
  sorry

end remaining_volume_of_cube_after_cylinder_removal_l441_441511


namespace number_of_valid_n_l441_441133

theorem number_of_valid_n : 
  (set.count 
    (λ n, ∃ a b : ℤ, (n : ℤ) = -(a + b) ∧ (n + 1 : ℤ) = -(a * b) ∧ 1 ≤ n ∧ n ≤ 2000)
    (finset.range 2001).1
  ) = 2000 := 
sorry

end number_of_valid_n_l441_441133


namespace solution_sets_equivalent_l441_441592

-- Definitions

def solution_set_ax2_minus_5x_plus_b_gt_0 (a b : ℝ) : set ℝ :=
  { x : ℝ | -3 < x ∧ x < 2 ∧ a * x^2 - 5 * x + b > 0 }

def solution_set_bx2_minus_5x_plus_a_gt_0 (a b : ℝ) : set ℝ :=
  { x : ℝ | (x < -(1/3) ∨ x > (1/2)) ∧ b * x^2 - 5 * x + a > 0 }

-- Theorem statement
theorem solution_sets_equivalent (a b : ℝ) :
  solution_set_ax2_minus_5x_plus_b_gt_0 a b = { x : ℝ | -3 < x ∧ x < 2 } →
  solution_set_bx2_minus_5x_plus_a_gt_0 a b = { x : ℝ | (x < -(1/3) ∨ x > (1/2)) } :=
sorry

end solution_sets_equivalent_l441_441592


namespace triangle_is_isosceles_l441_441902

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441902


namespace max_intersection_points_l441_441421

noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

theorem max_intersection_points (h1 : degree p = 5) 
                               (h2 : leading_coeff p = 1)
                               (h3 : degree q = 5)
                               (h4 : leading_coeff q = 1)
                               (h5 : p ≠ q)
                               (h6 : ∀ x, p x = q x → False): 
  ∃ n, n ≤ 4 ∧ ∀ x, p x = q x ↔ x = n :=
by {
  sorry
}

end max_intersection_points_l441_441421


namespace area_code_combinations_l441_441921

theorem area_code_combinations : 
  let digits := {9, 8, 7, 6} in
  ∀ (area_code : list ℕ), 
  (area_code.length = 4 ∧ ∀ x ∈ area_code, x ∈ digits ∧ (area_code.nodup)) →
  (∃! (ac : list ℕ), ac = area_code) ∧ 
  list.permutations digits |>.length = 24 := 
sorry

end area_code_combinations_l441_441921


namespace problem_proof_l441_441620

noncomputable def a := 160
noncomputable def b := 240
noncomputable def c := 0.50 * a - 0.10 * b
noncomputable def d := 2 ^ a
noncomputable def e := Real.log10 b
noncomputable def f := Nat.factorial c.toInt

theorem problem_proof :
  0.005 * a = 0.80 ∧
  0.0025 * b = 0.60 ∧
  c = 0.50 * a - 0.10 * b ∧
  d = 2 ^ a ∧
  e = Real.log10 b ∧
  c.toInt = 56 ∧
  f = Nat.factorial 56 := by
  sorry

end problem_proof_l441_441620


namespace triangle_ABC_isosceles_l441_441846

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441846


namespace isosceles_triangle_l441_441792

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441792


namespace binom_18_6_mul_smallest_prime_gt_10_eq_80080_l441_441509

theorem binom_18_6_mul_smallest_prime_gt_10_eq_80080 :
  (Nat.choose 18 6) * 11 = 80080 := sorry

end binom_18_6_mul_smallest_prime_gt_10_eq_80080_l441_441509


namespace cans_dad_brought_home_l441_441100

noncomputable section

def price_per_can : ℝ := 0.25
def cans_from_home : ℕ := 12
def cans_from_grandparents : ℕ := cans_from_home * 3
def cans_from_neighbor : ℕ := 46
def savings : ℝ := 43
def total_money : ℝ := savings * 2

def total_cans_collected_before_office : ℕ := 
  cans_from_home + cans_from_grandparents + cans_from_neighbor

def money_from_collected_cans : ℝ :=
  total_cans_collected_before_office * price_per_can

def money_from_dad : ℝ :=
  total_money - money_from_collected_cans

def cans_from_dad : ℕ :=
  money_from_dad / price_per_can

theorem cans_dad_brought_home : cans_from_dad = 250 := by
  sorry

end cans_dad_brought_home_l441_441100


namespace candidate_lost_by_2400_votes_l441_441039

-- Definitions based on given conditions
def total_votes : ℕ := 8000
def candidate_percentage : ℝ := 0.35
def rival_percentage : ℝ := 0.65  -- Since 100% - 35% = 65%

-- Calculations as definitions
def candidate_votes : ℕ := (candidate_percentage * (total_votes : ℝ)).to_nat
def rival_votes : ℕ := (rival_percentage * (total_votes : ℝ)).to_nat

-- The proof statement
theorem candidate_lost_by_2400_votes : rival_votes - candidate_votes = 2400 := sorry

end candidate_lost_by_2400_votes_l441_441039


namespace triangle_isosceles_if_parallel_l441_441867

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441867


namespace acute_triangle_probability_l441_441551

/-- Define the probability that any three of four randomly chosen points on a circle form acute triangles with the circle's center. -/
noncomputable def probability_acute_triangle := (3 : ℝ) / 64

/-- Given four points chosen uniformly at random on a circle, the probability that any three points combined with the circle's center always form an acute triangle is 3/64. -/
theorem acute_triangle_probability (points : Fin 4 → ℝ) (h : ∀ i j k, i < j → j < k → k < 4 → (points i) ∠ (points j) ∠ (points k)) :
  probability_acute_triangle = 3 / 64 := sorry

end acute_triangle_probability_l441_441551


namespace isosceles_triangle_l441_441799

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441799


namespace trigonometric_solution_l441_441000

theorem trigonometric_solution (a : ℝ) (n : ℤ):
  (sin a = -cos (3 * a)) ∧ (sin (3 * a) = -cos a) →
  ∃ n : ℤ, a = -π / 8 + (π * n) / 2 :=
by
  sorry

end trigonometric_solution_l441_441000


namespace primes_or_prime_squares_l441_441110

theorem primes_or_prime_squares (n : ℕ) (h1 : n > 1)
  (h2 : ∀ d, d ∣ n → d > 1 → (d - 1) ∣ (n - 1)) : 
  (∃ p, Nat.Prime p ∧ (n = p ∨ n = p * p)) :=
by
  sorry

end primes_or_prime_squares_l441_441110


namespace problem_statement_l441_441741

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441741


namespace triangle_is_isosceles_if_parallel_l441_441733

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441733


namespace isosceles_triangle_l441_441830

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441830


namespace sum_first_10_b_n_eq_10_over_11_l441_441591

theorem sum_first_10_b_n_eq_10_over_11
  (a : ℕ → ℕ)
  (h_arith_seq : ∃ d > 0, ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 = 5)
  (h2 : a 2 * a 3 = 6)
  (b : ℕ → ℚ := λ n, 1 / (a n * a (n + 1))) :
  (Finset.range 10).sum b = 10 / 11 := by
sorry

end sum_first_10_b_n_eq_10_over_11_l441_441591


namespace value_of_p_plus_q_l441_441125

noncomputable def radius_of_circle_E (A B C D E : Type)
  (T : A)
  (rA : ℝ) (rB : ℝ) (rC : ℝ) (rD : ℝ)
  [hA : Inscribed T A 10]
  [hB : Tangent B A (vertex T one) 4]
  [hC : Tangent C A (vertex T two) 2]
  [hD : Tangent D A (vertex T three) 2]
  [hE1 : Tangent B E 4]
  [hE2 : Tangent C E 2]
  [hE3 : Tangent D E 2] : ℝ := 
  7/5

theorem value_of_p_plus_q
  (A B C D E : Type)
  (T : A)
  (rA : ℝ := 10) (rB : ℝ := 4) (rC : ℝ := 2) (rD : ℝ := 2)
  [hA : Inscribed T A 10]
  [hB : Tangent B A (vertex T one) 4]
  [hC : Tangent C A (vertex T two) 2]
  [hD : Tangent D A (vertex T three) 2]
  [hE1 : Tangent B E 4]
  [hE2 : Tangent C E 2]
  [hE3 : Tangent D E 2] :
  7 + 5 = 12 :=
by
  sorry

end value_of_p_plus_q_l441_441125


namespace isosceles_triangle_of_parallel_l441_441887

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441887


namespace triangle_isosceles_if_parallel_l441_441873

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441873


namespace no_possible_k_for_prime_roots_l441_441498

theorem no_possible_k_for_prime_roots :
  ∃ (p q : ℕ), nat.prime p ∧ nat.prime q ∧ p + q = 108 → False :=
by {
  sorry
}

end no_possible_k_for_prime_roots_l441_441498


namespace sum_of_first_53_odd_numbers_l441_441446

theorem sum_of_first_53_odd_numbers :
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  let sum := 53 / 2 * (first_term + last_term)
  sum = 2809 :=
by
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  have last_term_val : last_term = 105 := by
    sorry
  let sum := 53 / 2 * (first_term + last_term)
  have sum_val : sum = 2809 := by
    sorry
  exact sum_val

end sum_of_first_53_odd_numbers_l441_441446


namespace triangle_is_isosceles_l441_441854

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441854


namespace existence_of_solution_values_continuous_solution_value_l441_441245

noncomputable def functional_equation_has_solution (a : ℝ) (f : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ f 1 = 1 ∧ ∀ x y, (x ≤ y → f ((x + y) / 2) = (1 - a) * f x + a * f y)

theorem existence_of_solution_values :
  {a : ℝ | ∃ f : ℝ → ℝ, functional_equation_has_solution a f} = {0, 1/2, 1} :=
sorry

theorem continuous_solution_value :
  {a : ℝ | ∃ (f : ℝ → ℝ) (hf : Continuous f), functional_equation_has_solution a f} = {1/2} :=
sorry

end existence_of_solution_values_continuous_solution_value_l441_441245


namespace number_of_packs_l441_441200

-- Given conditions
def cost_per_pack : ℕ := 11
def total_money : ℕ := 110

-- Statement to prove
theorem number_of_packs :
  total_money / cost_per_pack = 10 := by
  sorry

end number_of_packs_l441_441200


namespace probability_A2_selected_l441_441078

theorem probability_A2_selected : 
  let english_students := {A1, A2}
  let japanese_students := {B1, B2, B3}
  let sample_space := { (e, j) | e ∈ english_students, j ∈ japanese_students }
  let outcomes_with_A2 := { (A2, j) | j ∈ japanese_students }
  ∑ _ in outcomes_with_A2, (1 : ℝ) / (sample_space.card : ℝ) = (1 / 2 : ℝ) :=
by
  let english_students := {A1, A2}
  let japanese_students := {B1, B2, B3}
  let sample_space := { (e, j) | e ∈ english_students, j ∈ japanese_students }
  let outcomes_with_A2 := { (A2, j) | j ∈ japanese_students }
  have h1 : sample_space.card = 6, sorry
  have h2 : outcomes_with_A2.card = 3, sorry
  have h3 : ∑ _ in outcomes_with_A2, (1 : ℝ) / (sample_space.card : ℝ) = 3 * (1 / 6 : ℝ), sorry
  have h4 : 3 * (1 / 6 : ℝ) = 1 / 2, sorry
  exact h4

end probability_A2_selected_l441_441078


namespace triangle_is_isosceles_if_parallel_l441_441723

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441723


namespace expand_expression_l441_441126

theorem expand_expression (x : ℝ) : (x + 3) * (6x - 12) = 6x^2 + 6x - 36 := 
by 
  sorry

end expand_expression_l441_441126


namespace snow_white_last_trip_l441_441334

universe u

-- Define the dwarf names as an enumerated type
inductive Dwarf : Type
| Happy | Grumpy | Dopey | Bashful | Sleepy | Doc | Sneezy
deriving DecidableEq, Repr

open Dwarf

-- Define conditions
def initial_lineup : List Dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- Define the condition of adjacency
def adjacent (d1 d2 : Dwarf) : Prop :=
  List.pairwise_adjacent (· = d2) initial_lineup d1

-- Define the boat capacity
def boat_capacity : Fin 4 := by sorry

-- Snow White is the only one who can row
def snow_white_rows : Prop := true

-- No quarrels condition
def no_quarrel_without_snow_white (group : List Dwarf) : Prop :=
  ∀ d1 d2, d1 ∈ group → d2 ∈ group → ¬ adjacent d1 d2

-- Objective: Transfer all dwarfs without quarrels
theorem snow_white_last_trip (trips : List (List Dwarf)) :
  ∃ last_trip : List Dwarf,
    last_trip = [Grumpy, Bashful, Doc] ∧
    no_quarrel_without_snow_white (initial_lineup.diff (trips.join ++ last_trip)) :=
sorry

end snow_white_last_trip_l441_441334


namespace new_op_calculation_l441_441518

def new_op (x y a : ℤ) : ℤ := 18 + x - a * y

theorem new_op_calculation (
  a : ℤ,
  h1 : new_op 2 3 a = 8
) : new_op 3 5 a = 1 ∧ new_op 5 3 a = 11 :=
by
  sorry

end new_op_calculation_l441_441518


namespace probability_not_in_sin_region_l441_441013

theorem probability_not_in_sin_region :
  let S1 := ∫ x in 0..π, sin x
  let S2 := π * 1
  1 - S1 / S2 = 1 - (2 / π) :=
begin
  sorry
end

end probability_not_in_sin_region_l441_441013


namespace simplify_expression_l441_441252

theorem simplify_expression (x y z : ℤ) (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by
  sorry

end simplify_expression_l441_441252


namespace triangle_is_isosceles_l441_441855

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441855


namespace total_amount_spent_l441_441699

-- Definitions for problem conditions
def mall_spent_before_discount : ℝ := 250
def clothes_discount_percent : ℝ := 0.15
def mall_tax_percent : ℝ := 0.08

def movie_ticket_price : ℝ := 24
def num_movies : ℝ := 3
def ticket_discount_percent : ℝ := 0.10
def movie_tax_percent : ℝ := 0.05

def beans_price : ℝ := 1.25
def num_beans : ℝ := 20
def cucumber_price : ℝ := 2.50
def num_cucumbers : ℝ := 5
def tomato_price : ℝ := 5.00
def num_tomatoes : ℝ := 3
def pineapple_price : ℝ := 6.50
def num_pineapples : ℝ := 2
def market_tax_percent : ℝ := 0.07

-- Proof statement
theorem total_amount_spent :
  let mall_spent_after_discount := mall_spent_before_discount * (1 - clothes_discount_percent)
  let mall_tax := mall_spent_after_discount * mall_tax_percent
  let total_mall_spent := mall_spent_after_discount + mall_tax

  let total_ticket_cost_before_discount := num_movies * movie_ticket_price
  let ticket_cost_after_discount := total_ticket_cost_before_discount * (1 - ticket_discount_percent)
  let movie_tax := ticket_cost_after_discount * movie_tax_percent
  let total_movie_spent := ticket_cost_after_discount + movie_tax

  let total_beans_cost := num_beans * beans_price
  let total_cucumbers_cost := num_cucumbers * cucumber_price
  let total_tomatoes_cost := num_tomatoes * tomato_price
  let total_pineapples_cost := num_pineapples * pineapple_price
  let total_market_spent_before_tax := total_beans_cost + total_cucumbers_cost + total_tomatoes_cost + total_pineapples_cost
  let market_tax := total_market_spent_before_tax * market_tax_percent
  let total_market_spent := total_market_spent_before_tax + market_tax
  
  let total_spent := total_mall_spent + total_movie_spent + total_market_spent
  total_spent = 367.63 :=
by
  sorry

end total_amount_spent_l441_441699


namespace selection_count_l441_441488

-- Definitions from conditions
def graduates : Finset ℕ := Finset.range 11
def A : ℕ := 0  -- Assign A as graduate 0
def B : ℕ := 1  -- Assign B as graduate 1
def C : ℕ := 2  -- Assign C as graduate 2
def selected (s : Finset ℕ) : Prop := s.card = 3 ∧ C ∉ s ∧ (A ∈ s ∨ B ∈ s)

-- Main statement to prove
theorem selection_count : (Finset.filter selected (Finset.powersetLen 3 graduates)).card = 64 := sorry

end selection_count_l441_441488


namespace largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441407

theorem largest_integer_not_sum_of_multiple_of_30_and_composite :
  ∃ n : ℕ, ∀ a b : ℕ, 0 ≤ b ∧ b < 30 ∧ b.prime ∧ (∀ k < a, (b + 30 * k).prime)
    → (30 * a + b = n) ∧
      (∀ m : ℕ, ∀ c d : ℕ, 0 ≤ d ∧ d < 30 ∧ d.prime ∧ (∀ k < c, (d + 30 * k).prime) 
        → (30 * c + d ≤ n)) ∧
      n = 93 :=
by
  sorry

end largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441407


namespace snow_white_last_trip_l441_441309

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l441_441309


namespace gadget_marked_price_l441_441478

theorem gadget_marked_price:
  ∀ (op dp1 gp oc dp2 : ℝ),  
    op = 50 → 
    dp1 = 0.15 → 
    gp = 0.4 → 
    oc = 5 → 
    dp2 = 0.25 → 
    let purchase_price := op * (1 - dp1) in
    let gain := purchase_price * gp in
    let adjusted_selling_price := purchase_price + gain + oc in
    let mp := adjusted_selling_price / (1 - dp2) in
    mp = 86 :=
by
  intros _ _ _ _ _ h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  let purchase_price := 50 * (1 - 0.15)
  let gain := purchase_price * 0.4
  let adjusted_selling_price := purchase_price + gain + 5
  let mp := adjusted_selling_price / (1 - 0.25)
  have : purchase_price = 42.5 := by norm_num
  have : gain = 17 := by norm_num
  have : adjusted_selling_price = 64.5 := by norm_num
  have : mp = 86 := by norm_num
  exact this

end gadget_marked_price_l441_441478


namespace snow_white_last_trip_l441_441310

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l441_441310


namespace last_trip_l441_441315

def initial_order : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

def boatCapacity : Nat := 4  -- Including Snow White

def adjacentPairsQuarrel (adjPairs : List (String × String)) : Prop :=
  ∀ (d1 d2 : String), (d1, d2) ∈ adjPairs → (d2, d1) ∈ adjPairs → False

def canRow (person : String) : Prop := person = "Snow White"

noncomputable def final_trip (remainingDwarfs : List String) (allTrips : List (List String)) : List String := ["Grumpy", "Bashful", "Doc"]

theorem last_trip (adjPairs : List (String × String))
  (h_adj : adjacentPairsQuarrel adjPairs)
  (h_row : canRow "Snow White")
  (dwarfs_order : List String = initial_order)
  (without_quarrels : ∀ trip : List String, trip ∈ allTrips → 
    ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → (d1, d2) ∈ adjPairs → 
    ("Snow White" ∈ trip) → True) :
  final_trip ["Grumpy", "Bashful", "Doc"] allTrips = ["Grumpy", "Bashful", "Doc"] :=
sorry

end last_trip_l441_441315


namespace no_empty_boxes_prob_l441_441274

def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem no_empty_boxes_prob :
  let num_balls := 3
  let num_boxes := 3
  let total_outcomes := num_boxes ^ num_balls
  let favorable_outcomes := P num_balls num_boxes
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end no_empty_boxes_prob_l441_441274


namespace problem_statement_l441_441739

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441739


namespace action_figures_ratio_l441_441082

noncomputable theory

theorem action_figures_ratio (S : ℕ) (h1 : 24 - S - (24 - S) / 3 = 12) : S / 24 = 1 / 4 :=
by
  -- Using hypothesis h1 to find S
  sorry

end action_figures_ratio_l441_441082


namespace Rachel_age_when_father_is_60_l441_441919

-- Given conditions
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Proof problem statement
theorem Rachel_age_when_father_is_60 : Rachel_age + (60 - Father_age) = 25 :=
by sorry

end Rachel_age_when_father_is_60_l441_441919


namespace similar_triangles_with_same_color_l441_441101

theorem similar_triangles_with_same_color (color : Point → Prop) 
    (exists_color_triangle : ∀ a : ℝ, ∃ (P1 P2 : Point), color P1 ∧ color P2 ∧ dist P1 P2 = a) :
  ∃ (Δ1 Δ2 : Triangle), is_similar Δ1 Δ2 ∧ similarity_ratio Δ1 Δ2 = 1995 ∧ (∀ v ∈ Δ1, color v) ∧ (∀ v ∈ Δ2, color v) :=
sorry

end similar_triangles_with_same_color_l441_441101


namespace shaded_region_area_is_correct_l441_441477

noncomputable def area_of_shaded_region : ℝ :=
  let side_length := 9
  let radius := 4
  let angle := 120
  let hex_area := 6 * (sqrt 3 / 4) * (side_length ^ 2)
  let sector_area := 6 * (angle / 360) * π * (radius ^ 2)
  hex_area - sector_area

theorem shaded_region_area_is_correct :
  area_of_shaded_region = 121.5 * sqrt 3 - 32 * π :=
by
  sorry

end shaded_region_area_is_correct_l441_441477


namespace count_multiples_ending_with_5_l441_441603

theorem count_multiples_ending_with_5 :
  (finset.filter (λ n : ℕ, n < 500 ∧ n % 10 = 5) (finset.range 500)).card = 50 :=
begin
  sorry
end

end count_multiples_ending_with_5_l441_441603


namespace triangle_ABC_is_isosceles_l441_441906

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441906


namespace min_a_plus_b_l441_441351

open Real

theorem min_a_plus_b (a b : ℕ) (h_a_pos : a > 1) (h_ab : ∃ a b, (a^2 * b - 1) / (a * b^2) = 1 / 2024) :
  a + b = 228 :=
sorry

end min_a_plus_b_l441_441351


namespace quadratic_roots_ratio_l441_441365

theorem quadratic_roots_ratio {m n p : ℤ} (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : p ≠ 0)
  (h₃ : ∃ r1 r2 : ℤ, r1 * r2 = m ∧ n = 9 * r1 * r2 ∧ p = -(r1 + r2) ∧ m = -3 * (r1 + r2)) :
  n / p = -27 := by
  sorry

end quadratic_roots_ratio_l441_441365


namespace integral_eq_result_l441_441500

open Real

theorem integral_eq_result:
  ∫ (x: ℝ) in (π/4)..3, (3 * x - x^2) * sin(2 * x) =
    (π - 6 + 2 * cos(6) - 6 * sin(6)) / 8 := 
by
  sorry

end integral_eq_result_l441_441500


namespace interval_tolling_l441_441463

-- Defining the conditions
variables (n : ℕ) (bells : ℕ)
def initial_toll := 6 -- Number of bells tolling together initially
def total_interval := 30 -- Total interval in minutes
def times_toll := 16 -- Number of times the bells toll together within total_interval

-- Main theorem statement for the problem
theorem interval_tolling (h1 : bells = initial_toll) (h2 : times_toll = 16) (h3 : total_interval = 30) :
  (total_interval / (times_toll - 1) = 2) :=
by
  rw [← h3, ← h2]
  exact (div_eq_of_eq_mul' (60 * 2) rfl).symm

end interval_tolling_l441_441463


namespace product_distances_l441_441939

-- Define the function f(x)
noncomputable def f : ℝ → ℝ := λ x, 2 * x + 5 / x

-- Define the distance from the point to the line y = 2x
noncomputable def distance_to_line (x : ℝ) : ℝ := 
  abs (5 / x) / sqrt 5

-- Define the distance from the point to the y-axis
def distance_to_y_axis (x : ℝ) : ℝ := 
  abs x

theorem product_distances (x : ℝ) (hx : x ≠ 0) : 
  distance_to_line x * distance_to_y_axis x = sqrt 5 := by
  sorry

end product_distances_l441_441939


namespace find_t_l441_441363

theorem find_t (t : ℝ) : 
  (∃ (m b : ℝ), (∀ x y, (y = m * x + b) → ((x = 1 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 5 ∧ y = 19))) ∧ (28 = 28 * m + b) ∧ (t = 28 * m + b)) → 
  t = 88 :=
by
  sorry

end find_t_l441_441363


namespace largest_integer_not_sum_of_30_and_composite_l441_441419

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_not_sum_of_30_and_composite (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ b < 30 ∧ is_composite(b) ∧ n = 30 * a + b

theorem largest_integer_not_sum_of_30_and_composite :
  ∃ n : ℕ, is_not_sum_of_30_and_composite(n) ∧ ∀ m : ℕ, is_not_sum_of_30_and_composite(m) → m ≤ n :=
  ⟨93, sorry⟩

end largest_integer_not_sum_of_30_and_composite_l441_441419


namespace permutations_five_three_eq_sixty_l441_441040

theorem permutations_five_three_eq_sixty : (Nat.factorial 5) / (Nat.factorial (5 - 3)) = 60 := 
by
  sorry

end permutations_five_three_eq_sixty_l441_441040


namespace slope_range_l441_441573

open Real

theorem slope_range {k : ℝ} 
  (A B P : ℝ × ℝ) 
  (hA : A = (-2, 3)) 
  (hB : B = (3, 2)) 
  (hP : P = (0, -2)) 
  (h_intersects : ∃ k, (P.1 + k ≠ A.1) ∧ (P.1 + k ≠ B.1)) :
  k ∈ set.Icc (-5/2 : ℝ) (4/3 : ℝ) := by
  sorry

end slope_range_l441_441573


namespace monotonic_iff_midpoint_condition_l441_441686

variables {I : set ℝ} {f : ℝ → ℝ}

def is_interval (I : set ℝ) : Prop := 
  ∀ (x y z : ℝ), x ∈ I → z ∈ I → x ≤ y → y ≤ z → y ∈ I

def condition (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, |f x - f y| ≤ |x - y|

def monotonic (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

def midpoint_condition (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, (f x ≤ f ((x + y) / 2) ∧ f ((x + y) / 2) ≤ f y) ∨ 
             (f y ≤ f ((x + y) / 2) ∧ f ((x + y) / 2) ≤ f x)

theorem monotonic_iff_midpoint_condition
  (hI : is_interval I) (hf : condition f I) :
  monotonic f I ↔ midpoint_condition f I :=
sorry

end monotonic_iff_midpoint_condition_l441_441686


namespace share_per_person_l441_441021

-- Defining the total cost and number of people
def total_cost : ℝ := 12100
def num_people : ℝ := 11

-- The theorem stating that each person's share is $1,100.00
theorem share_per_person : total_cost / num_people = 1100 := by
  sorry

end share_per_person_l441_441021


namespace platform_length_l441_441989

open Real

noncomputable def trainLength : ℝ := 250
noncomputable def trainSpeedKmph : ℝ := 72
noncomputable def crossingTime : ℝ := 25

noncomputable def convertSpeedToMps (speedKmph : ℝ) : ℝ :=
  speedKmph * (1000 / 3600)

noncomputable def totalDistanceCovered (speedMps timeSec : ℝ) : ℝ :=
  speedMps * timeSec

theorem platform_length :
  let speedMps := convertSpeedToMps trainSpeedKmph in
  let distanceCovered := totalDistanceCovered speedMps crossingTime in
  distanceCovered = trainLength + 250 :=
by
  sorry

end platform_length_l441_441989


namespace isosceles_triangle_of_parallel_l441_441790

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441790


namespace combination_equality_l441_441029

theorem combination_equality : 
  Nat.choose 5 2 + Nat.choose 5 3 = 20 := 
by 
  sorry

end combination_equality_l441_441029


namespace isosceles_triangle_of_parallel_l441_441773

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441773


namespace snow_white_last_trip_l441_441325

-- Definitions based on the problem's conditions
inductive Dwarf
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

def is_adjacent (d1 d2 : Dwarf) : Prop :=
  (d1, d2) ∈ [
    (Dwarf.Happy, Dwarf.Grumpy),
    (Dwarf.Grumpy, Dwarf.Happy),
    (Dwarf.Grumpy, Dwarf.Dopey),
    (Dwarf.Dopey, Dwarf.Grumpy),
    (Dwarf.Dopey, Dwarf.Bashful),
    (Dwarf.Bashful, Dwarf.Dopey),
    (Dwarf.Bashful, Dwarf.Sleepy),
    (Dwarf.Sleepy, Dwarf.Bashful),
    (Dwarf.Sleepy, Dwarf.Doc),
    (Dwarf.Doc, Dwarf.Sleepy),
    (Dwarf.Doc, Dwarf.Sneezy),
    (Dwarf.Sneezy, Dwarf.Doc)
  ]

def boat_capacity : ℕ := 3

variable (snowWhite : Prop)

-- The theorem to prove that the dwarfs Snow White will take in the last trip are Grumpy, Bashful and Doc
theorem snow_white_last_trip 
  (h1 : snowWhite)
  (h2 : boat_capacity = 3)
  (h3 : ∀ d1 d2, is_adjacent d1 d2 → snowWhite)
  : (snowWhite ∧ (Dwarf.Grumpy ∧ Dwarf.Bashful ∧ Dwarf.Doc)) :=
sorry

end snow_white_last_trip_l441_441325


namespace part1_part2_part3_l441_441689

noncomputable def a (n : ℕ) : ℝ := 
if n = 1 then 1 else 
if n = 2 then 3/2 else 
if n = 3 then 5/4 else 
sorry

noncomputable def S (n : ℕ) : ℝ := sorry

axiom recurrence {n : ℕ} (h : n ≥ 2) : 4 * S (n + 2) + 5 * S n = 8 * S (n + 1) + S (n - 1)

-- Part 1
theorem part1 : a 4 = 7 / 8 :=
sorry

-- Part 2
theorem part2 : ∃ (r : ℝ) (b : ℕ → ℝ), (r = 1/2) ∧ (∀ n ≥ 1, a (n + 1) - r * a n = b n) :=
sorry

-- Part 3
theorem part3 : ∀ n, a n = (2 * n - 1) / 2^(n - 1) :=
sorry

end part1_part2_part3_l441_441689


namespace rectangle_height_proof_l441_441976

def side_square : ℕ := 20
def width_rectangle : ℕ := 14
def string_length := 4 * side_square

def height_rectangle : ℕ :=
  (string_length - 2 * width_rectangle) / 2

theorem rectangle_height_proof : height_rectangle = 26 := by
  unfold height_rectangle
  unfold string_length
  simp
  sorry

end rectangle_height_proof_l441_441976


namespace speeds_of_A_and_B_l441_441997

noncomputable theory
open Classical

-- Definitions using conditions from the problem
def speed_of_B : ℝ := 23 / 6
def speed_of_A : ℝ := 9 / 2

-- Conditions as definitions
def track_length (x_A x_B : ℝ) : ℝ :=
  let y := (x_A + x_B) * 48 in y

def same_direction_meeting_time (x_A x_B : ℝ) : Bool :=
  let y := track_length x_A x_B in
  (x_A - x_B) * 600 = y

-- Lean statement to verify the proof problem
theorem speeds_of_A_and_B :
  ∃ (x_A x_B : ℝ), x_A = speed_of_A ∧ x_B = speed_of_B ∧
  (track_length x_A x_B = (x_A + x_B) * 48) ∧
  same_direction_meeting_time x_A x_B :=
by
  -- Introducing the speed constants
  let x_B := speed_of_B
  let x_A := speed_of_A

  -- Prove that these speeds satisfy the track length and meeting time conditions
  use [x_A, x_B]
  split; simp [x_A, x_B] {contextual := tt}
  split; simp [x_A, x_B] {contextual := tt}

  -- Showing the first condition (track length when running in opposite directions)
  calc 
    (x_A + x_B) * 48 = 400 : by norm_num

  -- Showing the second condition (meeting time when running in the same direction)
  calc 
    (x_A - x_B) * 600 = 400 : by norm_num

  -- Final conclusion
  exact ⟨400, 400, rfl, rfl⟩

-- Mark as incomplete proof to skip the actual proof steps
sorry

end speeds_of_A_and_B_l441_441997


namespace abc_value_l441_441581

-- Define constants for the problem
variable (a b c k : ℕ)

-- Assumptions based on the given conditions
axiom h1 : a - b = 3
axiom h2 : a^2 + b^2 = 29
axiom h3 : a^2 + b^2 + c^2 = k
axiom pos_k : k > 0
axiom pos_a : a > 0

-- The goal is to prove that abc = 10
theorem abc_value : a * b * c = 10 :=
by
  sorry

end abc_value_l441_441581


namespace existence_of_unusual_100_digit_numbers_l441_441464

theorem existence_of_unusual_100_digit_numbers :
  ∃ (n₁ n₂ : ℕ), 
  (n₁ = 10^100 - 1) ∧ (n₂ = 5 * 10^99 - 1) ∧ 
  (∀ x : ℕ, x = n₁ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) ∧
  (∀ x : ℕ, x = n₂ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) := 
sorry

end existence_of_unusual_100_digit_numbers_l441_441464


namespace find_y_l441_441007

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 4) : y = 3 := 
by sorry

end find_y_l441_441007


namespace minimal_abs_diff_l441_441616

theorem minimal_abs_diff (a b : ℕ) (h : a * b - 3 * a + 4 * b = 149) : |a - b| = 33 :=
sorry

end minimal_abs_diff_l441_441616


namespace find_values_l441_441545

def problem_statement (x y : ℝ) (a b c : ℝ) : Prop :=
  (a + 1) * x ^ 2 - (2 + b) * x * y - y ^ 2 = 5 * x ^ 2 - 9 * x * y + c * y ^ 2

theorem find_values (x y : ℝ) :
  ∃ a b c : ℝ, (a + 1 = 5) ∧ (2 + b = 9) ∧ (c = -1) ∧ problem_statement x y a b c :=
by
  -- Declare the values of a, b, and c
  let a := 4
  let b := 7
  let c := -1
  -- Introduce hypothesis conditions
  have h1 : a + 1 = 5 := by sorry
  have h2 : 2 + b = 9 := by sorry
  have h3 : c = -1 := by sorry
  -- Combine hypothesis conditions to prove the theorem
  use [a, b, c]
  exact ⟨h1, h2, h3, by sorry⟩

end find_values_l441_441545


namespace max_x_plus_y_l441_441241

-- Define the conditions as hypotheses in a Lean statement
theorem max_x_plus_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^4 = (x - 1) * (y^3 - 23) - 1) :
  x + y ≤ 7 ∧ (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^4 = (x - 1) * (y^3 - 23) - 1 ∧ x + y = 7) :=
by
  sorry

end max_x_plus_y_l441_441241


namespace velocity_vector_turns_90_deg_in_2_root_2_seconds_l441_441036

theorem velocity_vector_turns_90_deg_in_2_root_2_seconds :
  let g : ℝ := 10  -- acceleration due to gravity in m/s^2
  let vo : ℝ := 20  -- initial speed in m/s
  let theta : ℝ := (Math.pi / 4)  -- projection angle in radians (45 degrees)
  let ty : ℝ := (vo * Real.sin theta)  -- vertical component of initial velocity
  let t : ℝ := (2 * Real.sqrt 2)
  (2 * ty) / g = t := by
  -- Using the kinematic equation for vertical motion across time with the quadratic roots.
  sorry

end velocity_vector_turns_90_deg_in_2_root_2_seconds_l441_441036


namespace fraction_less_than_thirty_percent_l441_441041

theorem fraction_less_than_thirty_percent (x : ℚ) (hx : x * 180 = 36) (hx_lt : x < 0.3) : x = 1 / 5 := 
by
  sorry

end fraction_less_than_thirty_percent_l441_441041


namespace Phil_quarters_l441_441711

theorem Phil_quarters (initial_amount : ℝ)
  (pizza : ℝ) (soda : ℝ) (jeans : ℝ) (book : ℝ) (gum : ℝ) (ticket : ℝ)
  (quarter_value : ℝ) (spent := pizza + soda + jeans + book + gum + ticket)
  (remaining := initial_amount - spent)
  (quarters := remaining / quarter_value) :
  initial_amount = 40 ∧ pizza = 2.75 ∧ soda = 1.50 ∧ jeans = 11.50 ∧
  book = 6.25 ∧ gum = 1.75 ∧ ticket = 8.50 ∧ quarter_value = 0.25 →
  quarters = 31 :=
by
  intros
  sorry

end Phil_quarters_l441_441711


namespace find_number_l441_441199

theorem find_number (p q N : ℝ) (h1 : N / p = 8) (h2 : N / q = 18) (h3 : p - q = 0.20833333333333334) : N = 3 :=
sorry

end find_number_l441_441199


namespace recurrence_relation_l441_441043

def u (n : ℕ) : ℕ := sorry

theorem recurrence_relation (n : ℕ) : 
  u (n + 1) = (n + 1) * u n - (n * (n - 1)) / 2 * u (n - 2) :=
sorry

end recurrence_relation_l441_441043


namespace triangle_is_isosceles_l441_441812

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441812


namespace symmetric_points_y_axis_l441_441608

theorem symmetric_points_y_axis (θ : ℝ) :
  (cos θ = -cos (θ + π / 6)) ∧ (sin θ = sin (θ + π / 6)) → θ = 5 * π / 12 :=
by
  -- Proof omitted
  sorry

end symmetric_points_y_axis_l441_441608


namespace evaluate_g_at_3_l441_441189

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

theorem evaluate_g_at_3 : g 3 = 79 := by
  sorry

end evaluate_g_at_3_l441_441189


namespace triangle_is_isosceles_if_parallel_l441_441724

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441724


namespace monthly_income_of_P_l441_441346

theorem monthly_income_of_P (P Q R : ℝ) 
    (h1 : (P + Q) / 2 = 2050) 
    (h2 : (Q + R) / 2 = 5250) 
    (h3 : (P + R) / 2 = 6200) : 
    P = 3000 :=
by
  sorry

end monthly_income_of_P_l441_441346


namespace even_function_increasing_l441_441941

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * b * x + 1

theorem even_function_increasing (h_even : ∀ x : ℝ, f a b x = f a b (-x))
  (h_increasing : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → f a b x < f a b y) :
  f a b (a-2) < f a b (b+1) :=
sorry

end even_function_increasing_l441_441941


namespace isosceles_triangle_l441_441762

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441762


namespace grid_marking_perimeter_l441_441704

theorem grid_marking_perimeter :
  ∀ (grid : Matrix (Fin 10) (Fin 10) Bool) (h : marked_cells_count grid = 10),
  ∃ (r₁ r₂ c₁ c₂ : Fin 10), r₁ ≤ r₂ ∧ c₁ ≤ c₂ ∧  
  2 * (int.of_nat (r₂ - r₁ + 1) + int.of_nat (c₂ - c₁ + 1)) ≥ 20 :=
by
  -- Introduce the necessary assumptions and goal.
  intro grid h,
  -- Prove that the given conditions necessarily lead to the desired perimeter being found.
  sorry

end grid_marking_perimeter_l441_441704


namespace sufficient_not_necessary_not_necessary_condition_sufficient_but_not_necessary_l441_441687

theorem sufficient_not_necessary (a : ℝ) (ha : a > 1) : (1 / a < 1) :=
begin
  have ha_pos : a ≠ 0 := ne_of_gt ha,
  field_simp [ha_pos],
  linarith,
end

theorem not_necessary_condition (a : ℝ) (h : 1 / a < 1) : a > 1 ∨ a < 0 :=
begin
  by_cases ha : a = 0,
  { exfalso,
    rw [ha, one_div_zero] at h,
    exact lt_irrefl 1 h, },
  { field_simp [ha] at h,
    by_cases ha1 : a > 0,
    { left,
      linarith, },
    { right,
      linarith, }, }
end

theorem sufficient_but_not_necessary (a : ℝ) : 
  (1 / a < 1 → a > 1 ∨ a < 0) ∧ (a > 1 → 1 / a < 1) :=
begin
  split,
  { intro h,
    exact not_necessary_condition a h, },
  { intro ha,
    exact sufficient_not_necessary a ha, }
end

end sufficient_not_necessary_not_necessary_condition_sufficient_but_not_necessary_l441_441687


namespace find_polynomial_l441_441536

theorem find_polynomial (P : ℝ → ℝ) (h_poly : ∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) :
  ∃ r s : ℝ, ∀ x : ℝ, P x = r * x^4 + s * x^2 :=
sorry

end find_polynomial_l441_441536


namespace part_a_l441_441458

theorem part_a (x y : ℝ) : x^2 - 2*y^2 = -((x + 2*y)^2 - 2*(x + y)^2) :=
sorry

end part_a_l441_441458


namespace suitable_survey_is_D_l441_441076

-- Define the surveys corresponding to options A, B, C, D
inductive Survey
| A : Survey
| B : Survey
| C : Survey
| D : Survey

-- Define what it means for a survey to be suitable for a census
def suitable_for_census : Survey → Prop
| Survey.A := False
| Survey.B := False
| Survey.C := False
| Survey.D := True

-- The problem statement: Prove that D is the only suitable survey for a census.
theorem suitable_survey_is_D : 
  (Survey.A ≠ Survey.D) ∧ (Survey.B ≠ Survey.D) ∧ (Survey.C ≠ Survey.D) ∧ suitable_for_census Survey.D :=
by sorry

end suitable_survey_is_D_l441_441076


namespace arithmetic_sequence_condition_l441_441617

theorem arithmetic_sequence_condition (a b c d : ℝ) :
  (∃ k : ℝ, b = a + k ∧ c = a + 2*k ∧ d = a + 3*k) ↔ (a + d = b + c) :=
sorry

end arithmetic_sequence_condition_l441_441617


namespace part1_part2_l441_441173

theorem part1 (a : ℝ) (h : ∃ x ∈ Icc (-1:ℝ) 1, x^2 - 4 * x + a + 3 = 0) : -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem part2 (b : ℝ) (h : ∀ x1 ∈ Icc (1:ℝ) 4, ∃ x2 ∈ Icc (1:ℝ) 4, b * x1 + 5 - 2 * b = x2^2 - 4 * x2 + 6) :
  -1 ≤ b ∧ b ≤ 1 / 2 :=
sorry

end part1_part2_l441_441173


namespace max_imaginary_part_of_roots_l441_441070

noncomputable def find_phi : Prop :=
  ∃ z : ℂ, z^6 - z^4 + z^2 - 1 = 0 ∧ (∀ w : ℂ, w^6 - w^4 + w^2 - 1 = 0 → z.im ≤ w.im) ∧ z.im = Real.sin (Real.pi / 4)

theorem max_imaginary_part_of_roots : find_phi :=
sorry

end max_imaginary_part_of_roots_l441_441070


namespace next_number_in_sequence_is_131_l441_441992

/-- Define the sequence increments between subsequent numbers -/
def sequencePattern : List ℕ := [1, 2, 2, 4, 2, 4, 2, 4, 6, 2]

-- Function to apply a sequence of increments starting from an initial value
def computeNext (initial : ℕ) (increments : List ℕ) : ℕ :=
  increments.foldl (λ acc inc => acc + inc) initial

-- Function to get the sequence's nth element 
def sequenceNthElement (n : ℕ) : ℕ :=
  (computeNext 12 (sequencePattern.take n))

-- Proof that the next number in the sequence is 131 
theorem next_number_in_sequence_is_131 :
  sequenceNthElement 10 = 131 :=
  by
  -- Proof omitted
  sorry

end next_number_in_sequence_is_131_l441_441992


namespace isosceles_triangle_l441_441803

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441803


namespace problem_solution_l441_441577

-- Definitions for the problem
variable {A B : ℝ}

-- Given Conditions
variable (h1 : 0 < A ∧ A < π / 2)
variable (h2 : 0 < B ∧ B < π / 2)
variable (h3 : (cos (A + π / 3), sin (A + π / 3)) ⊥ (cos B, sin B))
variable (h4 : cos B = 3 / 5)
variable (h5 : AC = 8)

-- Theorem statement to prove
theorem problem_solution :
  (A - B = π / 6) ∧ (BC = 4 * sqrt 3 + 3) :=
sorry

end problem_solution_l441_441577


namespace bisect_segment_with_parallel_lines_l441_441181

theorem bisect_segment_with_parallel_lines (A B P C D Q : Point) (l1 l2 : Line)
  (h_parallel : parallel l1 l2)
  (h_AB_on_l1 : OnLine l1 A ∧ OnLine l1 B)
  (h_P_not_on_l1_l2 : ¬ OnLine l1 P ∧ ¬ OnLine l2 P)
  (h_PA_intersect_l2_C : LineSegment P A ⊓ l2 = C)
  (h_PB_intersect_l2_D : LineSegment P B ⊓ l2 = D)
  (h_AD_intersect_BC_Q : LineSegment A D ⊓ LineSegment B C = Q)
  (h_PQ_intersects_AB_M : Intersects (LineSegment P Q) (Midpoint A B)) : 
  bisect A B Q :=
sorry

end bisect_segment_with_parallel_lines_l441_441181


namespace increase_then_decrease_l441_441034

theorem increase_then_decrease (x : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) (result : ℝ) :
  x = 950 →
  increase_rate = 0.80 →
  decrease_rate = 0.65 →
  result = x * (1 + increase_rate) * (1 - decrease_rate) →
  result = 598.5 :=
by
  intros h_x h_incr h_decr h_result
  rw [h_x, h_incr, h_decr] at h_result
  linarith

end increase_then_decrease_l441_441034


namespace triangle_is_isosceles_if_parallel_l441_441729

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441729


namespace coefficient_x3_expansion_l441_441349

theorem coefficient_x3_expansion:
  let f := (3 - 2*x - x^4) * (2*x - 1)^6 in
  (coeff f 3) = -600 :=
by
  sorry

end coefficient_x3_expansion_l441_441349


namespace number_of_triangles_l441_441568

theorem number_of_triangles (m : ℕ) (h : m > 0) :
  ∃ n : ℕ, n = (m * (m + 1)) / 2 :=
by sorry

end number_of_triangles_l441_441568


namespace problem_statement_l441_441737

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441737


namespace odd_factor_form_l441_441260

theorem odd_factor_form (n : ℕ) (x y : ℕ) (h_n : n > 0) (h_gcd : Nat.gcd x y = 1) :
  ∀ p, p ∣ (x ^ (2 ^ n) + y ^ (2 ^ n)) ∧ Odd p → ∃ k > 0, p = 2^(n+1) * k + 1 := 
by
  sorry

end odd_factor_form_l441_441260


namespace mutually_exclusive_case_B_l441_441982

open Finset

-- Define the events
def atLeastOneHead (outcome : set (Fin 3 × bool)) : Prop :=
  {h | ∃ i : Fin 3, (i, true) ∈ outcome} ⊆ outcome

def atMostOneHead (outcome : set (Fin 3 × bool)) : Prop :=
  {h | ∃ i : Fin 3, ∀ j : Fin 3, j ≠ i → (j, false) ∈ outcome} ⊆ outcome

def atLeastTwoHeads (outcome : set (Fin 3 × bool)) : Prop :=
  have heads : Nat := (count (λ b : bool, b == true) (outcome.toFinMap.range)),
  (heads ≥ 2)

def exactlyTwoHeads (outcome : set (Fin 3 × bool)) : Prop :=
  have heads : Nat := (count (λ b : bool, b == true) (outcome.toFinMap.range)),
  (heads == 2)

-- Main theorem statement
theorem mutually_exclusive_case_B (outcome: set (Fin 3 × bool)): 
  atMostOneHead outcome → ¬ atLeastTwoHeads outcome := by
  sorry

end mutually_exclusive_case_B_l441_441982


namespace snow_white_last_trip_l441_441320

-- Definitions based on the problem's conditions
inductive Dwarf
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

def is_adjacent (d1 d2 : Dwarf) : Prop :=
  (d1, d2) ∈ [
    (Dwarf.Happy, Dwarf.Grumpy),
    (Dwarf.Grumpy, Dwarf.Happy),
    (Dwarf.Grumpy, Dwarf.Dopey),
    (Dwarf.Dopey, Dwarf.Grumpy),
    (Dwarf.Dopey, Dwarf.Bashful),
    (Dwarf.Bashful, Dwarf.Dopey),
    (Dwarf.Bashful, Dwarf.Sleepy),
    (Dwarf.Sleepy, Dwarf.Bashful),
    (Dwarf.Sleepy, Dwarf.Doc),
    (Dwarf.Doc, Dwarf.Sleepy),
    (Dwarf.Doc, Dwarf.Sneezy),
    (Dwarf.Sneezy, Dwarf.Doc)
  ]

def boat_capacity : ℕ := 3

variable (snowWhite : Prop)

-- The theorem to prove that the dwarfs Snow White will take in the last trip are Grumpy, Bashful and Doc
theorem snow_white_last_trip 
  (h1 : snowWhite)
  (h2 : boat_capacity = 3)
  (h3 : ∀ d1 d2, is_adjacent d1 d2 → snowWhite)
  : (snowWhite ∧ (Dwarf.Grumpy ∧ Dwarf.Bashful ∧ Dwarf.Doc)) :=
sorry

end snow_white_last_trip_l441_441320


namespace roberts_total_sales_l441_441284

theorem roberts_total_sales 
  (basic_salary : ℝ := 1250) 
  (commission_rate : ℝ := 0.10) 
  (savings_rate : ℝ := 0.20) 
  (monthly_expenses : ℝ := 2888) 
  (S : ℝ) : S = 23600 :=
by
  have total_earnings := basic_salary + commission_rate * S
  have used_for_expenses := (1 - savings_rate) * total_earnings
  have expenses_eq : used_for_expenses = monthly_expenses := sorry
  have expense_calc : (1 - savings_rate) * (basic_salary + commission_rate * S) = monthly_expenses := sorry
  have simplify_eq : 0.80 * (1250 + 0.10 * S) = 2888 := sorry
  have open_eq : 1000 + 0.08 * S = 2888 := sorry
  have isolate_S : 0.08 * S = 1888 := sorry
  have solve_S : S = 1888 / 0.08 := sorry
  have final_S : S = 23600 := sorry
  exact final_S

end roberts_total_sales_l441_441284


namespace disjoint_subsets_same_sum_l441_441392

theorem disjoint_subsets_same_sum (s : Finset ℕ) (h₁ : s.card = 10) (h₂ : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 100) :
  ∃ A B : Finset ℕ, A ⊆ s ∧ B ⊆ s ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end disjoint_subsets_same_sum_l441_441392


namespace determine_8_genuine_coins_l441_441014

-- Assume there are 11 coins and one may be counterfeit.
variable (coins : Fin 11 → ℝ)
variable (is_counterfeit : Fin 11 → Prop)
variable (genuine_weight : ℝ)
variable (balance : (Fin 11 → ℝ) → (Fin 11 → ℝ) → Prop)

-- The weight of genuine coins.
axiom genuine_coins_weight : ∀ i, ¬ is_counterfeit i → coins i = genuine_weight

-- The statement of the mathematical problem in Lean 4.
theorem determine_8_genuine_coins :
  ∃ (genuine_set : Finset (Fin 11)), genuine_set.card ≥ 8 ∧ ∀ i ∈ genuine_set, ¬ is_counterfeit i :=
sorry

end determine_8_genuine_coins_l441_441014


namespace expected_sixes_correct_l441_441973

-- Define probabilities for rolling individual numbers on a die
def P (n : ℕ) (k : ℕ) : ℚ := if k = n then 1 / 6 else 0

-- Expected value calculation for two dice
noncomputable def expected_sixes_two_dice_with_resets : ℚ :=
(0 * (13/18)) + (1 * (2/9)) + (2 * (1/36))

-- Main theorem to prove
theorem expected_sixes_correct :
  expected_sixes_two_dice_with_resets = 5 / 18 :=
by
  -- The actual proof steps go here; added sorry to skip the proof.
  sorry

end expected_sixes_correct_l441_441973


namespace max_diff_x_y_l441_441253

theorem max_diff_x_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : 
  x - y ≤ Real.sqrt (4 / 3) := 
by
  sorry

end max_diff_x_y_l441_441253


namespace symmetric_points_y_axis_l441_441607

theorem symmetric_points_y_axis (θ : ℝ) :
  (cos θ = -cos (θ + π / 6)) ∧ (sin θ = sin (θ + π / 6)) → θ = 5 * π / 12 :=
by
  -- Proof omitted
  sorry

end symmetric_points_y_axis_l441_441607


namespace greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l441_441395

def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry -- Implementation of finding greatest prime factor goes here

theorem greatest_prime_factor_of_5_pow_7_plus_6_pow_6 : 
  greatest_prime_factor (5^7 + 6^6) = 211 := 
by 
  sorry -- Proof of the theorem goes here

end greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l441_441395


namespace trajectory_is_ellipse_min_value_of_MN_l441_441564

variable (P : ℝ × ℝ)
variable (F : ℝ × ℝ := (Real.sqrt 2, 0))
variable (l : ℝ := 2 * Real.sqrt 2)
variable (y1 y2 : ℝ)

def trajectory_equation (P : ℝ × ℝ) :=
  let x := P.1
  let y := P.2 in
  (√((x - √2)^2 + y^2) / |x - 2 * √2| = √2 / 2)

theorem trajectory_is_ellipse (P : ℝ × ℝ) (h : trajectory_equation P) :
  ∃ x y : ℝ, P = (x, y) ∧ (x^2) / 4 + (y^2) / 2 = 1 :=
sorry

variable (E : ℝ × ℝ := (-Real.sqrt 2, 0))
variable (M : ℝ × ℝ := (2 * Real.sqrt 2, y1))
variable (N : ℝ × ℝ := (2 * Real.sqrt 2, y2))

def EM_FN_dot_product_zero :=
  (3 * Real.sqrt 2, y1) • (Real.sqrt 2, y2) = 0

theorem min_value_of_MN (h : EM_FN_dot_product_zero) :
  y1 > y2 → minM(value : ℝ) = 2 * Real.sqrt 6 :=
sorry

end trajectory_is_ellipse_min_value_of_MN_l441_441564


namespace trig_identity_l441_441576

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) : 
  ∃ (res : ℝ), res = 10 / 7 ∧ res = Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3) := by
  sorry

end trig_identity_l441_441576


namespace smallest_set_handshakes_l441_441462

-- Define the number of people
def num_people : Nat := 36

-- Define a type for people
inductive Person : Type
| a : Fin num_people → Person

-- Define the handshake relationship
def handshake (p1 p2 : Person) : Prop :=
  match p1, p2 with
  | Person.a i, Person.a j => i.val = (j.val + 1) % num_people ∨ j.val = (i.val + 1) % num_people

-- Define the problem statement
theorem smallest_set_handshakes :
  ∃ s : Finset Person, (∀ p : Person, p ∈ s ∨ ∃ q ∈ s, handshake p q) ∧ s.card = 18 :=
sorry

end smallest_set_handshakes_l441_441462


namespace isosceles_triangle_of_parallel_l441_441885

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441885


namespace problem_statement_l441_441746

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441746


namespace min_value_of_m_l441_441944

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

def g (x m : ℝ) : ℝ := f (x - m)

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem min_value_of_m {m : ℝ} (h1 : 0 < m) (h2 : is_even (g m)) : m = Real.pi / 3 :=
sorry

end min_value_of_m_l441_441944


namespace calculate_non_defective_m3_percentage_l441_441212

def percentage_non_defective_m3 : ℝ := 93

theorem calculate_non_defective_m3_percentage 
  (P : ℝ) -- Total number of products
  (P_pos : 0 < P) -- Total number of products is positive
  (percentage_m1 : ℝ := 0.40)
  (percentage_m2 : ℝ := 0.30)
  (percentage_m3 : ℝ := 0.30)
  (defective_m1 : ℝ := 0.03)
  (defective_m2 : ℝ := 0.01)
  (total_defective : ℝ := 0.036) :
  percentage_non_defective_m3 = 93 :=
by sorry -- The actual proof is omitted

end calculate_non_defective_m3_percentage_l441_441212


namespace isosceles_triangle_l441_441804

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441804


namespace value_added_to_each_number_is_11_l441_441665

-- Given definitions and conditions
def initial_average : ℝ := 40
def number_count : ℕ := 15
def new_average : ℝ := 51

-- Mathematically equivalent proof statement
theorem value_added_to_each_number_is_11 (x : ℝ) 
  (h1 : number_count * initial_average = 600)
  (h2 : (600 + number_count * x) / number_count = new_average) : 
  x = 11 := 
by 
  sorry

end value_added_to_each_number_is_11_l441_441665


namespace expand_expression_l441_441521

theorem expand_expression (a b : ℤ) : (-1 + a * b^2)^2 = 1 - 2 * a * b^2 + a^2 * b^4 :=
by sorry

end expand_expression_l441_441521


namespace prism_surface_area_is_14_l441_441386

-- Definition of the rectangular prism dimensions
def prism_length : ℕ := 3
def prism_width : ℕ := 1
def prism_height : ℕ := 1

-- Definition of the surface area of the rectangular prism
def surface_area (l w h : ℕ) : ℕ :=
  2 * (l * w + w * h + h * l)

-- Theorem statement: The surface area of the resulting prism is 14
theorem prism_surface_area_is_14 : surface_area prism_length prism_width prism_height = 14 :=
  sorry

end prism_surface_area_is_14_l441_441386


namespace isosceles_triangle_l441_441802

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441802


namespace percentage_increase_correct_l441_441657

-- Defining initial conditions
variables (B H : ℝ) -- bears per week without assistant and hours per week without assistant

-- Defining the rate of making bears per hour without assistant
def rate_without_assistant := B / H

-- Defining the rate with an assistant (100% increase in output per hour)
def rate_with_assistant := 2 * rate_without_assistant

-- Defining the number of hours worked per week with an assistant (10% fewer hours)
def hours_with_assistant := 0.9 * H

-- Calculating the number of bears made per week with an assistant
def bears_with_assistant := rate_with_assistant * hours_with_assistant

-- Calculating the percentage increase in the number of bears made per week when Jane works with an assistant
def percentage_increase : ℝ := ((bears_with_assistant / B) - 1) * 100

-- The theorem to prove
theorem percentage_increase_correct : percentage_increase B H = 80 :=
  by sorry

end percentage_increase_correct_l441_441657


namespace isosceles_triangle_of_parallel_l441_441879

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441879


namespace prob_all_three_defective_approx_l441_441481

noncomputable def prob_defective (total: ℕ) (defective: ℕ) := (defective : ℚ) / (total : ℚ)

theorem prob_all_three_defective_approx :
  let p_total := 120
  let s_total := 160
  let b_total := 60
  let p_def := 26
  let s_def := 68
  let b_def := 30
  let p_prob := prob_defective p_total p_def
  let s_prob := prob_defective s_total s_def
  let b_prob := prob_defective b_total b_def
  let combined_prob := p_prob * s_prob * b_prob
  abs (combined_prob - (221 / 4800 : ℚ)) < 0.001 :=
by
  sorry

end prob_all_three_defective_approx_l441_441481


namespace a_n_expression_l441_441153

noncomputable def sum_of_first_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then 1 else (1 / 2) * (3 / 2) ^ (n - 1)

theorem a_n_expression (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : a 1 = 1) 
(h2 : ∀ n, S n = sum_of_first_n a n) 
(h3 : ∀ n, S n = 2 * a (n + 1)) :
  ∀ n, a n = if n = 1 then 1 else (1 / 2) * (3 / 2) ^ (n - 2) :=
by
  sorry

end a_n_expression_l441_441153


namespace largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441415

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441415


namespace find_a_plus_b_l441_441259

noncomputable def imaginary_unit : ℂ := complex.I

theorem find_a_plus_b (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b)
  (h : complex.abs ((a:ℂ) + imaginary_unit * 1) * complex.abs (2 + imaginary_unit * 1)
       = complex.abs ((b:ℂ) - imaginary_unit * 1) / complex.abs (2 - imaginary_unit)) :
  a + b = 8 :=
by
  sorry

end find_a_plus_b_l441_441259


namespace digit_B_divisibility_l441_441385

theorem digit_B_divisibility :
  ∃ B : ℕ, B < 10 ∧ (2 * 100 + B * 10 + 9) % 13 = 0 ↔ B = 0 :=
by
  sorry

end digit_B_divisibility_l441_441385


namespace coeff_x_expression_l441_441113

theorem coeff_x_expression : 
  (∀ x : ℝ, (5 * (x - 6) - 6 * (3 - x^2 + 3 * x) + 7 * (4 * x - 5)) = 
            (6 * x^2 + 15 * x - 83)) →
  (15 : ℝ) :=
by
  intro h
  sorry

end coeff_x_expression_l441_441113


namespace cost_per_pack_l441_441273

def chores_per_week := 4
def weeks := 10
def siblings := 2
def cookies_per_chore := 3
def total_money := 15
def cookies_per_pack := 24

theorem cost_per_pack :
  let total_chores_per_sibling := chores_per_week * weeks in
  let total_chores := total_chores_per_sibling * siblings in
  let total_cookies_needed := cookies_per_chore * total_chores in
  let total_packs_needed := total_cookies_needed / cookies_per_pack in
  (total_money / total_packs_needed : ℝ) = 1.50 :=
by
  sorry

end cost_per_pack_l441_441273


namespace triangle_XDE_area_l441_441231

theorem triangle_XDE_area 
  (XY YZ XZ : ℝ) (hXY : XY = 8) (hYZ : YZ = 12) (hXZ : XZ = 14)
  (D E : ℝ → ℝ) (XD XE : ℝ) (hXD : XD = 3) (hXE : XE = 9) :
  ∃ (A : ℝ), A = 1/2 * XD * XE * (15 * Real.sqrt 17 / 56) ∧ A = 405 * Real.sqrt 17 / 112 :=
  sorry

end triangle_XDE_area_l441_441231


namespace largest_non_sum_217_l441_441396

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end largest_non_sum_217_l441_441396


namespace circle_x_intercept_l441_441348

def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def radius (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) / 2

theorem circle_x_intercept (p1 p2 : (ℝ × ℝ)) : 
  p1 = (2, 2) → 
  p2 = (10, 8) → 
  let c := midpoint p1 p2 in 
  let r := radius p1 p2 in 
  let x0 := c.1 in 
  let y0 := c.2 in 
  let x := Real.sqrt ((x0^2 - 2*x0*y0 + y0^2)/(y0^2)) in
  x = 6 :=
by
  sorry

end circle_x_intercept_l441_441348


namespace smallest_height_proof_l441_441637

noncomputable def smallest_height_of_scalene_triangle (a b c : ℤ) (h_triangle : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_equation : (a^2) / c - (a - c)^2 = (b^2) / c - (b - c)^2) : ℝ :=
  let height := (2 * (Real.sqrt((a:ℝ)^2 + (b:ℝ)^2 - (c:ℝ)^2).abs)) / c
  in if h_triangle ∧ h_equation then height else 0

theorem smallest_height_proof :
  ∀ a b c : ℤ, (a ≠ b ∧ b ≠ c ∧ a ≠ c) → ((a^2) / c - (a - c)^2 = (b^2) / c - (b - c)^2) → smallest_height_of_scalene_triangle a b c (a ≠ b ∧ b ≠ c ∧ a ≠ c) ((a^2) / c - (a - c)^2 = (b^2) / c - (b - c)^2) = 2.4 :=
by
  intros a b c h_triangle h_equation
  let height := (2 * (Real.sqrt((a:ℝ)^2 + (b:ℝ)^2 - (c:ℝ)^2).abs)) / c
  have h_smallest : height = 2.4 := sorry
  exact h_smallest

end smallest_height_proof_l441_441637


namespace value_of_f_x_plus_5_l441_441248

open Function

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem value_of_f_x_plus_5 (x : ℝ) : f (x + 5) = 3 * x + 16 :=
by
  sorry

end value_of_f_x_plus_5_l441_441248


namespace range_of_k_l441_441595

theorem range_of_k (k : ℝ) :
  ∀ x : ℝ, ∃ a b c : ℝ, (a = k-1) → (b = -2) → (c = 1) → (a ≠ 0) → ((b^2 - 4 * a * c) ≥ 0) → k ≤ 2 ∧ k ≠ 1 :=
by
  sorry

end range_of_k_l441_441595


namespace market_value_after_two_years_l441_441951

noncomputable def depreciation_market_value (initial_value : ℝ) (depreciation_rate : ℝ) (inflation_rate : ℝ) : ℝ :=
  let value_after_depreciation := initial_value * (1 - depreciation_rate)
  let value_after_inflation := value_after_depreciation * (1 + inflation_rate)
  value_after_inflation

theorem market_value_after_two_years
  (initial_value : ℝ) (depreciation_rate1 : ℝ) (depreciation_rate2 : ℝ) (inflation_rate : ℝ) :
  depreciation_market_value (depreciation_market_value initial_value depreciation_rate1 inflation_rate)
    depreciation_rate2 inflation_rate = 4939.20 :=
by
  let year1_value := depreciation_market_value initial_value depreciation_rate1 inflation_rate
  let year2_value := depreciation_market_value year1_value depreciation_rate2 inflation_rate
  have year1_calc : year1_value = 5880 :=
    by
      simp [depreciation_market_value]
      norm_num
  have year2_calc : year2_value = 4939.20 :=
    by
      simp [year1_calc, depreciation_market_value]
      norm_num
  exact year2_calc


end market_value_after_two_years_l441_441951


namespace augmented_matrix_solution_l441_441166

theorem augmented_matrix_solution (a b : ℝ) 
    (h1 : (∀ (x y : ℝ), (a * x = 2 ∧ y = b ↔ x = 2 ∧ y = 1))) : 
    a + b = 2 :=
by
  sorry

end augmented_matrix_solution_l441_441166


namespace journey_speed_first_half_l441_441470

theorem journey_speed_first_half (total_distance : ℕ) (total_time : ℕ) (second_half_distance : ℕ) (second_half_speed : ℕ)
  (distance_first_half_eq_half_total : second_half_distance = total_distance / 2)
  (time_for_journey_eq : total_time = 20)
  (journey_distance_eq : total_distance = 240)
  (second_half_speed_eq : second_half_speed = 15) :
  let v := second_half_distance / (total_time - (second_half_distance / second_half_speed))
  v = 10 := 
by
  sorry

end journey_speed_first_half_l441_441470


namespace BC_perp_AF_l441_441690

theorem BC_perp_AF
  (A B C D E F : Type)
  [Inhabited A] [Inhabited B] [Inhabited C]
  [Inhabited D] [Inhabited E] [Inhabited F]
  (angle_ABC angle_BAC angle_DCB angle_EBC : ℝ)
  (h1 : angle_ABC = 60)
  (h2 : angle_BAC = 40)
  (h3 : angle_DCB = 70)
  (h4 : angle_EBC = 40)
  (h5 : intersecting BE CD = F)
: perpendicular BC AF := sorry

end BC_perp_AF_l441_441690


namespace triangle_is_isosceles_l441_441810

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441810


namespace gas_pipe_probability_l441_441068

theorem gas_pipe_probability :
  (let L := 200 in
  let condition1 (x y : ℝ) := 50 ≤ x ∧ 50 ≤ y ∧ 50 ≤ L - x - y in
  let area_original := (1/2 : ℝ) * L * L in
  let area_feasible := (1/2 : ℝ) * (L - 50) * (L - 50) in
  let probability := area_feasible / area_original in
  probability = 9 / 16) := sorry

end gas_pipe_probability_l441_441068


namespace cistern_fill_time_l441_441467

def fill_rate : ℝ := 1 / 4
def empty_rate : ℝ := 1 / 6

theorem cistern_fill_time : (1 / (fill_rate - empty_rate)) = 12 := by
  have h_fill : fill_rate = 1 / 4 := rfl
  have h_empty : empty_rate = 1 / 6 := rfl
  have h_net : fill_rate - empty_rate = 1 / 12 :=
    calc fill_rate - empty_rate
      = 1 / 4 - 1 / 6 : by rw [h_fill, h_empty]
  ... = 1 / 12 : by norm_num
  calc 1 / (fill_rate - empty_rate)
    = 1 / (1 / 12) : by rw [h_net]
  ... = 12 : by norm_num

end cistern_fill_time_l441_441467


namespace problem_statement_l441_441740

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441740


namespace power_of_p_in_product_l441_441628

theorem power_of_p_in_product (p q : ℕ) (x : ℕ) (hp : Prime p) (hq : Prime q) 
  (h : (x + 1) * 6 = 30) : x = 4 := 
by sorry

end power_of_p_in_product_l441_441628


namespace pencils_left_l441_441922

-- Define the initial quantities
def MondayPencils := 35
def TuesdayPencils := 42
def WednesdayPencils := 3 * TuesdayPencils
def WednesdayLoss := 20
def ThursdayPencils := WednesdayPencils / 2
def FridayPencils := 2 * MondayPencils
def WeekendLoss := 50

-- Define the total number of pencils Sarah has at the end of each day
def TotalMonday := MondayPencils
def TotalTuesday := TotalMonday + TuesdayPencils
def TotalWednesday := TotalTuesday + WednesdayPencils - WednesdayLoss
def TotalThursday := TotalWednesday + ThursdayPencils
def TotalFriday := TotalThursday + FridayPencils
def TotalWeekend := TotalFriday - WeekendLoss

-- The proof statement
theorem pencils_left : TotalWeekend = 266 :=
by
  sorry

end pencils_left_l441_441922


namespace increasing_condition_neither_necessary_nor_sufficient_l441_441160

-- Define the conditions
variables (f : ℝ → ℝ) (g : ℝ → ℝ) (h0 : ∀ x : ℝ, 0 < x → f x = x * f(x)) -- Condition 1 and 2

-- Statement of the proof problem
theorem increasing_condition_neither_necessary_nor_sufficient :
  ¬((∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) ↔ (∀ x y : ℝ, 0 < x → 0 < y → x < y → g x < g y)) :=
sorry -- proof is not required

end increasing_condition_neither_necessary_nor_sufficient_l441_441160


namespace correct_average_l441_441934

theorem correct_average (incorrect_avg : ℝ) (num_values : ℕ) (misread_value actual_value : ℝ) 
  (h1 : incorrect_avg = 16) 
  (h2 : num_values = 10)
  (h3 : misread_value = 26)
  (h4 : actual_value = 46) : 
  (incorrect_avg * num_values + (actual_value - misread_value)) / num_values = 18 := 
by
  sorry

end correct_average_l441_441934


namespace algebra_inequality_l441_441203

theorem algebra_inequality (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end algebra_inequality_l441_441203


namespace no_n_such_that_n_times_s_is_20222022_l441_441251

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem
theorem no_n_such_that_n_times_s_is_20222022 :
  ∀ n : ℕ, n * sum_of_digits n ≠ 20222022 :=
by
  sorry

end no_n_such_that_n_times_s_is_20222022_l441_441251


namespace intersection_is_correct_l441_441115

-- Define the line y = -3x - 2
def line1 (x : ℚ) : ℚ := -3 * x - 2

-- Define the line perpendicular to line1 passing through (3, -3)
def line2 (x : ℚ) : ℚ := (1 / 3) * x - 4

-- Define the intersection point
def intersection_point : ℚ × ℚ := (3 / 5, -19 / 5)

-- The theorem stating that the intersection of the lines is the given point
theorem intersection_is_correct : ∃ x y : ℚ, 
  line1 x = y ∧ line2 x = y ∧ 
  (x, y) = intersection_point :=
by 
  use (3 / 5)
  use (-19 / 5)
  split
  · show line1 (3 / 5) = -19 / 5
    sorry
  split
  · show line2 (3 / 5) = -19 / 5
    sorry
  · refl

end intersection_is_correct_l441_441115


namespace snow_white_last_trip_dwarfs_l441_441300

-- Definitions for the conditions
def original_lineup := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]
def only_snow_white_can_row := True
def boat_capacity_snow_white_and_dwarfs := 3
def dwarfs_quarrel_if_adjacent (d1 d2 : String) : Prop :=
  let index_d1 := List.indexOf original_lineup d1
  let index_d2 := List.indexOf original_lineup d2
  abs (index_d1 - index_d2) = 1

-- Theorem to prove the correct answer
theorem snow_white_last_trip_dwarfs :
  let last_trip_dwarfs := ["Grumpy", "Bashful", "Sneezy"]
  ∃ (trip : List String), trip = last_trip_dwarfs ∧ 
  ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → d1 ≠ d2 → ¬dwarfs_quarrel_if_adjacent d1 d2 :=
by
  sorry

end snow_white_last_trip_dwarfs_l441_441300


namespace sum_first_39_natural_numbers_l441_441990

theorem sum_first_39_natural_numbers :
  (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end sum_first_39_natural_numbers_l441_441990


namespace solve_inequality_problem_l441_441165

noncomputable def inequality_problem (a b c : ℝ) (h0 : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : a + b + c = 1) : Prop :=
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2

theorem solve_inequality_problem (a b c : ℝ) (h0 : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : a + b + c = 1) : inequality_problem a b c h0 h1 h2 h3 :=
begin
  sorry
end

end solve_inequality_problem_l441_441165


namespace complex_modulus_calculation_l441_441562

variable (z : ℂ)

theorem complex_modulus_calculation 
  (h : (z + complex.i) / (-2 * complex.i ^ 3 - z) = complex.i) :
  complex.abs (complex.conj z + 1) = real.sqrt 2 / 2 := sorry

end complex_modulus_calculation_l441_441562


namespace triangle_isosceles_if_parallel_l441_441872

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441872


namespace train_speed_conversion_l441_441067

/-- Define a function to convert kmph to m/s --/
def kmph_to_ms (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

/-- Theorem stating that 72 kmph is equivalent to 20 m/s --/
theorem train_speed_conversion : kmph_to_ms 72 = 20 :=
by
  sorry

end train_speed_conversion_l441_441067


namespace manager_is_lying_l441_441452

def residents (A B C : Set ℕ) : Prop :=
  A.card = 25 ∧
  B.card = 30 ∧
  C.card = 28 ∧
  (A ∩ C).card = 18 ∧
  (B ∩ C).card = 17 ∧
  (A ∩ B).card = 16 ∧
  (A ∩ B ∩ C).card = 15

theorem manager_is_lying (A B C : Set ℕ) (claimed_residents : ℕ) (h : residents A B C) :
  claimed_residents = 45 → (A ∪ B ∪ C).card ≠ claimed_residents :=
by
  unfold residents at h
  rw Set.card_eq at h
  sorry

end manager_is_lying_l441_441452


namespace last_trip_l441_441318

def initial_order : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

def boatCapacity : Nat := 4  -- Including Snow White

def adjacentPairsQuarrel (adjPairs : List (String × String)) : Prop :=
  ∀ (d1 d2 : String), (d1, d2) ∈ adjPairs → (d2, d1) ∈ adjPairs → False

def canRow (person : String) : Prop := person = "Snow White"

noncomputable def final_trip (remainingDwarfs : List String) (allTrips : List (List String)) : List String := ["Grumpy", "Bashful", "Doc"]

theorem last_trip (adjPairs : List (String × String))
  (h_adj : adjacentPairsQuarrel adjPairs)
  (h_row : canRow "Snow White")
  (dwarfs_order : List String = initial_order)
  (without_quarrels : ∀ trip : List String, trip ∈ allTrips → 
    ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → (d1, d2) ∈ adjPairs → 
    ("Snow White" ∈ trip) → True) :
  final_trip ["Grumpy", "Bashful", "Doc"] allTrips = ["Grumpy", "Bashful", "Doc"] :=
sorry

end last_trip_l441_441318


namespace triangle_is_isosceles_l441_441859

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441859


namespace triangle_sides_area_triangle_area_given_sine_l441_441207

-- Conditions definition
variables {a b c : ℝ} {A B C : ℝ}

-- Proof problem for Question 1
theorem triangle_sides_area (h1 : c = 4) (h2 : C = π / 3) (area : 1 / 2 * a * b * sin C = 4 * sqrt 3) : a = 4 ∧ b = 4 :=
sorry

-- Proof problem for Question 2
theorem triangle_area_given_sine (h1 : c = 4) (h2 : C = π / 3) (h3 : sin B = 2 * sin A) : 1 / 2 * a * (2 * a) * sin C = 8 * sqrt 3 / 3 :=
sorry

end triangle_sides_area_triangle_area_given_sine_l441_441207


namespace triangle_is_isosceles_l441_441808

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441808


namespace tile_grid_with_l_pieces_l441_441549

theorem tile_grid_with_l_pieces (n : ℕ) (hn : n ≥ 1) :
  (∃ (f : ℕ → ℕ → bool), (∀ i j, i < n → j < n → (f i j = tt ∨ f i j = ff)) ∧ 
    (∀ i j, i < n → j < n → (L_shaped_piece f i j)) ∧ 
    (∀ i j, i < n → j < n → ¬overlap_or_overhang f i j)) ↔ 4 ∣ n :=
sorry

end tile_grid_with_l_pieces_l441_441549


namespace max_friends_in_compartment_l441_441635

/-- Definition of compartment and properties -/
def Compartment (P : Type) [Fintype P] (m : ℕ) (h_m : 3 ≤ m) :=
  ∀ (A : P) (S : Finset P), 
    A ∉ S → S.card = m - 1 → 
    ∃! C : P, C ∈ S ∧ ∀ (B : P), B ∈ S → (isFriend B C ∧ isFriend C B)
    where
      isFriend : P → P → Prop
      isFriend_symm : symmetric isFriend
      isFriend_irrefl : irreflexive isFriend

/-- Proving the maximum number of friends a person can have in the compartment(P), given the conditions -/
theorem max_friends_in_compartment (P : Type) [Fintype P] (m : ℕ) (h_m : 3 ≤ m) (cpt : Compartment P m h_m):
  ∀ (A : P), (∃ (k : ℕ), ∀ B : P, B ≠ A → isFriend A B → k <= m) := 
sorry

end max_friends_in_compartment_l441_441635


namespace non_adjacent_green_plates_arrangement_l441_441049

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem non_adjacent_green_plates_arrangement :
  let total_plates := 14
  let blue_plates := 6
  let red_plates := 3
  let green_plates := 3
  let orange_plates := 2
  let total_arrangements := factorial (total_plates - 1) / (factorial blue_plates * factorial red_plates * factorial green_plates * factorial orange_plates)
  let adjacent_green_arrangements := factorial (total_plates - green_plates - 1) / (factorial blue_plates * factorial red_plates * factorial orange_plates * factorial 1)
  in total_arrangements - adjacent_green_arrangements = 1349070 :=
by
  let total_plates := 14
  let blue_plates := 6
  let red_plates := 3
  let green_plates := 3
  let orange_plates := 2
  
  let total_arrangements := factorial (total_plates - 1) / (factorial blue_plates * factorial red_plates * factorial green_plates * factorial orange_plates)
  let adjacent_green_arrangements := factorial (total_plates - green_plates - 1) / (factorial blue_plates * factorial red_plates * factorial orange_plates * factorial 1)
  
  have h : total_arrangements - adjacent_green_arrangements = 1349070 := sorry

  exact h

end non_adjacent_green_plates_arrangement_l441_441049


namespace smallest_integer_proof_l441_441424

def smallest_integer_with_gcd_18_6 : Nat :=
  let n := 114
  if n > 100 ∧  (Nat.gcd n 18) = 6 then n else 0

theorem smallest_integer_proof : smallest_integer_with_gcd_18_6 = 114 := 
  by
    unfold smallest_integer_with_gcd_18_6
    have h₁ : 114 > 100 := by decide
    have h₂ : Nat.gcd 114 18 = 6 := by decide
    simp [h₁, h₂]
    sorry

end smallest_integer_proof_l441_441424


namespace value_of_x_l441_441431

theorem value_of_x (x : ℕ) : (8^4 + 8^4 + 8^4 = 2^x) → x = 13 :=
by
  sorry

end value_of_x_l441_441431


namespace problem_one_problem_two_l441_441063

variable (A B : Type)
variable (red white : A → B → Prop)
variable [decidable_pred red]
variable [decidable_pred white]
variable (boxA boxB : list B)
variable (prob : B → ℚ)

-- Conditions
def boxA_red := (boxA.count red) = 4
def boxA_white := (boxA.count white) = 6
def boxB_red := (boxB.count red) = 5
def boxB_white := (boxB.count white) = 5
def independent_draws := ∀ (a1 a2 : B), red a1 → red a2 → (prob a1 * prob a2 = prob a1 + prob a2)

-- Theorem statements
theorem problem_one : boxA_red → boxA_white → boxB_red → boxB_white → independent_draws →
  (prob (red _ _)) = 7/10 := 
sorry

theorem problem_two : boxA_red → boxA_white → boxB_red → boxB_white → independent_draws →
  (prob (red _ _)) = 1/5 → (3 times independent_draws) → 
  (prob_wins_at_most_twice (red _ _)) = 124/125 := 
sorry


end problem_one_problem_two_l441_441063


namespace tangent_circle_given_r_l441_441618

theorem tangent_circle_given_r (r : ℝ) (h_pos : 0 < r)
    (h_tangent : ∀ x y : ℝ, (2 * x + y = r) → (x^2 + y^2 = 2 * r))
  : r = 10 :=
sorry

end tangent_circle_given_r_l441_441618


namespace problem_part1_problem_part2_l441_441176

noncomputable def f (m x : ℝ) := Real.log (m * x) - x + 1
noncomputable def g (m x : ℝ) := (x - 1) * Real.exp x - m * x

theorem problem_part1 (m : ℝ) (h : m > 0) (hf : ∀ x, f m x ≤ 0) : m = 1 :=
sorry

theorem problem_part2 (m : ℝ) (h : m > 0) :
  ∃ x₀, (∀ x, g m x ≤ g m x₀) ∧ (1 / 2 * Real.log (m + 1) < x₀ ∧ x₀ < m) :=
sorry

end problem_part1_problem_part2_l441_441176


namespace circle_rolling_start_point_l441_441064

theorem circle_rolling_start_point (x : ℝ) (h1 : ∃ x, (x + 2 * Real.pi = -1) ∨ (x - 2 * Real.pi = -1)) :
  x = -1 - 2 * Real.pi ∨ x = -1 + 2 * Real.pi :=
by
  sorry

end circle_rolling_start_point_l441_441064


namespace find_m_l441_441162

variables (OA OB OC : ℝ ^ 3) (m λ : ℝ)

-- Given conditions
axiom norm_OA : ∥OA∥ = 1
axiom norm_OB : ∥OB∥ = m
axiom angle_AOB : innerProductSpace.angle OA OB = 3 / 4 * real.pi
axiom orthogonality_OA_OC : innerProductSpace.dotProduct OA OC = 0
axiom OC_expression : OC = 2 * λ • OA + λ • OB
axiom lambda_ne_zero : λ ≠ 0

-- Statement to prove
theorem find_m : m = 2 * real.sqrt 2 := sorry

end find_m_l441_441162


namespace sum_3000_l441_441960

-- Definitions based on conditions
def geo_seq_sum (a r : ℝ) (n : ℕ) : ℝ :=
if r = 1 then a * n else a * (1 - r^n) / (1 - r)

variables (a r : ℝ)

-- Given conditions
def sum_1000 : Prop := geo_seq_sum a r 1000 = 500
def sum_2000 : Prop := geo_seq_sum a r 2000 = 950

-- The statement to prove
theorem sum_3000 (h1 : sum_1000 a r) (h2 : sum_2000 a r) :
  geo_seq_sum a r 3000 = 1355 :=
sorry

end sum_3000_l441_441960


namespace sum_3000_l441_441959

-- Definitions based on conditions
def geo_seq_sum (a r : ℝ) (n : ℕ) : ℝ :=
if r = 1 then a * n else a * (1 - r^n) / (1 - r)

variables (a r : ℝ)

-- Given conditions
def sum_1000 : Prop := geo_seq_sum a r 1000 = 500
def sum_2000 : Prop := geo_seq_sum a r 2000 = 950

-- The statement to prove
theorem sum_3000 (h1 : sum_1000 a r) (h2 : sum_2000 a r) :
  geo_seq_sum a r 3000 = 1355 :=
sorry

end sum_3000_l441_441959


namespace hexagon_area_l441_441180

-- Given conditions
variables (d e f R : ℝ)
hypothesis h1 : d + e + f = 24
hypothesis h2 : R = 5

-- Proof statement
theorem hexagon_area : (R * (d + e + f)) / 4 = 30 :=
by
  -- Adding the hypothesis into the context
  have hR : R = 5 := h2
  have hPerimeter : d + e + f = 24 := h1
  -- Calculating the area using given conditions
  sorry

end hexagon_area_l441_441180


namespace problem_statement_l441_441232

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (A B C D : V)
  (BC CD : V)
  (AC AB AD : V)

theorem problem_statement
  (h1 : BC = 2 • CD)
  (h2 : BC = AC - AB) :
  AD = (3 / 2 : ℝ) • AC - (1 / 2 : ℝ) • AB :=
sorry

end problem_statement_l441_441232


namespace no_suitable_start_day_l441_441237

open List

-- Define the weekdays and closed days.
inductive Weekday : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

def is_weekend : Weekday → Prop
| Weekday.Saturday := True
| Weekday.Sunday := True
| _ := False

-- Define the function to calculate the day after n*12 days starting from a given day.
def add_days_to_weekday (start : Weekday) (n : ℕ) : Weekday :=
  match start with
  | Weekday.Monday    => Weekday.of_nat ((5 + 12 * n) % 7)
  | Weekday.Tuesday   => Weekday.of_nat ((6 + 12 * n) % 7)
  | Weekday.Wednesday => Weekday.of_nat ((7 + 12 * n) % 7)
  | Weekday.Thursday  => Weekday.of_nat ((8 + 12 * n) % 7)
  | Weekday.Friday    => Weekday.of_nat ((9 + 12 * n) % 7)
  | Weekday.Saturday  => Weekday.of_nat ((10 + 12 * n) % 7)
  | Weekday.Sunday    => Weekday.of_nat ((11 + 12 * n) % 7)

-- Define a property that checks if all days are not weekend for the next 8 coupons.
def all_days_non_weekend (start : Weekday) : Prop :=
  ∀ n < 8, ¬is_weekend (add_days_to_weekday start n)

-- Prove there is no suitable starting day.
theorem no_suitable_start_day : ¬ ∃ d : Weekday, all_days_non_weekend d :=
begin
  sorry
end

end no_suitable_start_day_l441_441237


namespace area_of_triangle_AEB_l441_441644

-- Defining point coordinates in the rectangle
noncomputable def A : ℝ × ℝ := (0, 5)
noncomputable def B : ℝ × ℝ := (8, 5)
noncomputable def C : ℝ × ℝ := (8, 0)
noncomputable def D : ℝ × ℝ := (0, 0)
noncomputable def F : ℝ × ℝ := (3, 0)
noncomputable def G : ℝ × ℝ := (7, 0)

-- Point E is the intersection of lines AF and BG
noncomputable def E : ℝ × ℝ := sorry -- defining intersect point need more geometry.
-- Declare the required conditions as hypotheses in Lean
theorem area_of_triangle_AEB : 
  let area := (1 / 2) * 8 * 5 in area = 40 :=
by sorry

end area_of_triangle_AEB_l441_441644


namespace determine_time_l441_441472

variable (g a V_0 V S t : ℝ)

def velocity_eq : Prop := V = (g + a) * t + V_0
def displacement_eq : Prop := S = 1 / 2 * (g + a) * t^2 + V_0 * t

theorem determine_time (h1 : velocity_eq g a V_0 V t) (h2 : displacement_eq g a V_0 S t) :
  t = 2 * S / (V + V_0) := 
sorry

end determine_time_l441_441472


namespace arithmetic_sequence_iff_c_zero_l441_441144

theorem arithmetic_sequence_iff_c_zero (a b c : ℝ) :
  (∀ n : ℕ, 0 < n → ∃ S : ℝ, S = a * n^2 + b * n + c) →
  (∀ n m : ℕ, 0 < n ∧ 0 < m → Succeeds ((a * (2 * n - 1) + b) = (a * (2 * (n + 1) - 1) + b)) (2*a)) ↔
  c = 0 :=
sorry

end arithmetic_sequence_iff_c_zero_l441_441144


namespace JukuExit_l441_441940

def JukuEscalator (escalator_steps : ℕ) (escalator_rate : ℝ) (initial_position : ℕ) (time_units : ℕ) : ℝ :=
  let juku_rate := -⅔ -- One step back every second effectively.
  let net_juku_rate := juku_rate - (1 / 2)
  initial_position + (time_units * net_juku_rate)

def willJukuExitAt (exit_position : ℕ) : Prop :=
  JukuEscalator 75 (1 / 2) 38 45 = exit_position

theorem JukuExit : willJukuExitAt 23 :=
  by
    sorry

end JukuExit_l441_441940


namespace find_number_l441_441195

theorem find_number (n : ℕ) (some_number : ℕ) 
  (h : (1/5 : ℝ)^n * (1/4 : ℝ)^(18 : ℕ) = 1 / (2 * (some_number : ℝ)^n))
  (hn : n = 35) : some_number = 10 := 
by 
  sorry

end find_number_l441_441195


namespace isosceles_triangle_of_parallel_l441_441777

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441777


namespace product_of_values_of_c_l441_441967

open_locale classical

theorem product_of_values_of_c :
  (∃ (c : ℕ), ∃ (d : ℕ), 3 * (c : ℤ) * (d : ℤ) = 8 ∧
    (∃ (r : ℚ), 3 * r^2 + 7 * r + c = 0 ∧
    ∃ (s : ℚ), 3 * s^2 + 7 * s + d = 0)) :=
sorry

end product_of_values_of_c_l441_441967


namespace gcd_143_144_l441_441541

def a : ℕ := 143
def b : ℕ := 144

theorem gcd_143_144 : Nat.gcd a b = 1 :=
by
  sorry

end gcd_143_144_l441_441541


namespace contrapositive_example_l441_441936

theorem contrapositive_example (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → (a^2 + b^2 ≠ 0) :=
by
  sorry

end contrapositive_example_l441_441936


namespace triangle_ABC_is_isosceles_l441_441905

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441905


namespace value_of_x_l441_441373

theorem value_of_x (x y z : ℝ) (h1 : x = (1 / 2) * y) (h2 : y = (1 / 4) * z) (h3 : z = 80) : x = 10 := by
  sorry

end value_of_x_l441_441373


namespace length_of_diagonal_AC_l441_441219

-- Definitions based on the conditions
variable (AB BC CD DA AC : ℝ)
variable (angle_ADC : ℝ)

-- Conditions
def conditions : Prop :=
  AB = 12 ∧ BC = 12 ∧ CD = 15 ∧ DA = 15 ∧ angle_ADC = 120

theorem length_of_diagonal_AC (h : conditions AB BC CD DA angle_ADC) : AC = 15 :=
sorry

end length_of_diagonal_AC_l441_441219


namespace quadrilateral_sides_equal_l441_441917

theorem quadrilateral_sides_equal (a b c d : ℕ) (h1 : a ∣ b + c + d) (h2 : b ∣ a + c + d) (h3 : c ∣ a + b + d) (h4 : d ∣ a + b + c) : a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end quadrilateral_sides_equal_l441_441917


namespace simplify_fraction_product_l441_441288

theorem simplify_fraction_product : 
  (21 / 28) * (14 / 33) * (99 / 42) = 1 := 
by 
  sorry

end simplify_fraction_product_l441_441288


namespace correlation_index_implies_better_fitting_l441_441221

-- The correlation index R^2 is a measure used in regression analysis.
def correlation_index (R2 : ℝ) := R2

-- Definition: The closer R^2 is to 1, the better the fitting effect of the model.
def better_fitting_effect (R2 : ℝ) : Prop := (R2 → (R2 = 1))

/--
Theorem: A larger correlation index R^2 implies a better fitting effect of the regression model.
-/
theorem correlation_index_implies_better_fitting (R2 : ℝ) (h : R2 → (R2 = 1)) : better_fitting_effect R2 := 
by
  sorry

end correlation_index_implies_better_fitting_l441_441221


namespace distance_between_vertices_of_hyperbola_l441_441131

-- Define the hyperbola and the distance between its vertices
def hyperbola (x y : ℝ) : Prop := ((x - 1)^2 / 16) - (y^2 / 25) = 1

theorem distance_between_vertices_of_hyperbola : 
  (∀ x y : ℝ, hyperbola x y) → 
  (distance 5 (-3) = 8) := 
by
  sorry

end distance_between_vertices_of_hyperbola_l441_441131


namespace percentage_increase_bears_l441_441659

-- Define the initial conditions
variables (B H : ℝ) -- B: bears per week without an assistant, H: hours per week without an assistant

-- Define the rate without assistant
def rate_without_assistant : ℝ := B / H

-- Define the working hours with an assistant
def hours_with_assistant : ℝ := 0.9 * H

-- Define the rate with an assistant (100% increase)
def rate_with_assistant : ℝ := 2 * rate_without_assistant

-- Define the number of bears per week with an assistant
def bears_with_assistant : ℝ := rate_with_assistant * hours_with_assistant

-- Prove the percentage increase in the number of bears made per week
theorem percentage_increase_bears (hB : B > 0) (hH : H > 0) :
  ((bears_with_assistant B H - B) / B) * 100 = 80 :=
by
  unfold bears_with_assistant rate_with_assistant hours_with_assistant rate_without_assistant
  simp
  sorry

end percentage_increase_bears_l441_441659


namespace limit_proof_l441_441276

open Real

theorem limit_proof :
  ∀ ε > 0, ∃ δ > 0, (∀ x, 0 < |x + 4| ∧ |x + 4| < δ → |(2 * x ^ 2 + 6 * x - 8) / (x + 4) + 10| < ε) := sorry

end limit_proof_l441_441276


namespace problem_statement_l441_441747

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441747


namespace volume_common_part_of_tetrahedra_l441_441567

theorem volume_common_part_of_tetrahedra (a b c a1 b1 c1 : Point) 
  (hp: Prism ABC_A1B1C1) 
  (hedge_lengths: ∀ (x y : Point), (distance x y) = 1) 
  : volume_common_tetrahedra A1ABC B1ABC C1ABC = (Real.sqrt 3) / 36 :=
sorry

end volume_common_part_of_tetrahedra_l441_441567


namespace isosceles_triangle_of_parallel_l441_441785

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441785


namespace final_trip_l441_441327

-- Definitions for the conditions
def dwarf := String
def snow_white := "Snow White" : dwarf
def Happy : dwarf := "Happy"
def Grumpy : dwarf := "Grumpy"
def Dopey : dwarf := "Dopey"
def Bashful : dwarf := "Bashful"
def Sleepy : dwarf := "Sleepy"
def Doc : dwarf := "Doc"
def Sneezy : dwarf := "Sneezy"

-- The dwarfs lineup from left to right
def lineup : List dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- The boat can hold Snow White and up to 3 dwarfs
def boat_capacity (load : List dwarf) : Prop := snow_white ∈ load ∧ load.length ≤ 4

-- Any two dwarfs standing next to each other in the original lineup will quarrel if left without Snow White
def will_quarrel (d1 d2 : dwarf) : Prop :=
  (d1, d2) ∈ (lineup.zip lineup.tail)

-- The objective: Prove that on the last trip, Snow White will take Grumpy, Bashful, and Doc
theorem final_trip : ∃ load : List dwarf, 
  set.load ⊆ {snow_white, Grumpy, Bashful, Doc} ∧ boat_capacity load :=
  sorry

end final_trip_l441_441327


namespace cost_per_box_l441_441434

-- Define the dimensions of the box
def length := 20
def width := 20
def height := 12

-- Define the total volume needed
def total_volume := 2160000

-- Define the minimum total cost
def minimum_total_cost := 225

-- Calculate the volume of one box
def volume_one_box := length * width * height

-- Calculate the number of boxes needed
def number_of_boxes := total_volume / volume_one_box

-- The proof problem: cost per box
theorem cost_per_box :
  (minimum_total_cost / number_of_boxes) = 0.50 :=
by
  sorry

end cost_per_box_l441_441434


namespace interval_solution_l441_441143

-- Let the polynomial be defined
def polynomial (x : ℝ) : ℝ := x^3 - 12 * x^2 + 30 * x

-- Prove the inequality for the specified intervals
theorem interval_solution :
  { x : ℝ | polynomial x > 0 } = { x : ℝ | (0 < x ∧ x < 5) ∨ x > 6 } :=
by
  sorry

end interval_solution_l441_441143


namespace num_correct_props_l441_441074

-- Define the propositions
def prop1 (p q : Prop) : Prop := (p ∧ q) → (p ∨ q) ∧ ¬ ((p ∨ q) → (p ∧ q))
def prop2 : Prop := (∃ x : ℝ, x^2 + 2 * x ≤ 0) ↔ (¬ ∀ x : ℝ, x^2 + 2 * x > 0)
def prop3 : Prop := (∀ x : ℝ, x^2 - 2 * x + 3 > 0) ↔ (¬ ∃ x : ℝ, x^2 - 2 * x + 3 ≤ 0)
def prop4 (p q : Prop) : Prop := (¬ p → q) ↔ (p → ¬ q)

-- Define the theorem that states that only one of the above propositions is correct
theorem num_correct_props (p q : Prop) : (count (λ prop, match prop with
  | prop1 p q => false
  | prop2 => true
  | prop3 => false
  | prop4 p q => false 
  end) [prop1 p q, prop2, prop3, prop4 p q] = 1) :=
  sorry

end num_correct_props_l441_441074


namespace simplify_expression_find_m_n_value_l441_441455

-- Problem 1 statement
theorem simplify_expression
  (x y : ℤ)
  (h₁ : x = 1)
  (h₂ : y = -2) :
  7 * x * y - 2 * (5 * x * y - 2 * x ^ 2 * y) + 3 * x * y = -8 :=
by sorry

-- Problem 2 statement
theorem find_m_n_value
  (m n : ℤ)
  (h₁ : 3 * (1:ℤ) * (1:ℤ) - n * (1:ℤ) ^ (m + 1) * (1:ℤ) - (1:ℤ) = 3)
  (h₂ : m + 2 = 3)
  (h₃ : -n = 2) :
  m^2 + n^3 = -7 :=
by sorry

end simplify_expression_find_m_n_value_l441_441455


namespace largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441406

theorem largest_integer_not_sum_of_multiple_of_30_and_composite :
  ∃ n : ℕ, ∀ a b : ℕ, 0 ≤ b ∧ b < 30 ∧ b.prime ∧ (∀ k < a, (b + 30 * k).prime)
    → (30 * a + b = n) ∧
      (∀ m : ℕ, ∀ c d : ℕ, 0 ≤ d ∧ d < 30 ∧ d.prime ∧ (∀ k < c, (d + 30 * k).prime) 
        → (30 * c + d ≤ n)) ∧
      n = 93 :=
by
  sorry

end largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441406


namespace fraction_never_simplifiable_l441_441718

theorem fraction_never_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_never_simplifiable_l441_441718


namespace triangle_ABC_isosceles_l441_441833

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441833


namespace line_intersect_parabola_l441_441599

-- Define the parabola and points conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the midpoint condition
def midpoint (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2) / 2 = 1 / 2 ∧ (y1 + y2) / 2 = -1

-- Problem statement in Lean
theorem line_intersect_parabola (x1 y1 x2 y2 : ℝ) (hA : parabola x1 y1) (hB : parabola x2 y2) (hM : midpoint x1 y1 x2 y2) :
  ∃ (m b : ℝ), (∀ (x : ℝ), y = m * x + b) ∧ ∀ (x : ℝ), parabola x (m * x + b) → m = -2 ∧ b = 0 :=
begin
  sorry
end

end line_intersect_parabola_l441_441599


namespace concyclic_points_l441_441647

open EuclideanGeometry

variables {A B C A1 C1 P Q R : Point}
variables [HD : DenseSquare]

-- Assume non-isosceles acute triangle ABC
axiom non_isosceles_acute_triangle (hA : AcuteAngle (TriangleAngle A B C))
  (hB : AcuteAngle (TriangleAngle B A C)) : ¬IsoscelesTriangle A B C

-- Assume AA1 and CC1 are the altitudes of triangle ABC
axiom AA1_altitude (H : Orthocenter A B C) : Altitude A A1 B C
axiom CC1_altitude (H : Orthocenter A B C) : Altitude C C1 A B

-- Assume the angle bisectors of acute angles formed by AA1, CC1 intersect sides AB and BC at P and Q
axiom angle_bisectors (H : Orthocenter A B C) :
  (AngleBisector (Angle (Altitude A A1 B C) (Altitude C C1 A B)) P A B C) ∧
  (AngleBisector (Angle (Altitude A A1 B C) (Altitude C C1 A B)) Q B C A)

-- Assume the angle bisector of ∠ B intersects line segment joining orthocenter and midpoint of AC at point R
axiom B_angle_bisector (H : Orthocenter A B C) (M : MidPoint A C) :
  AngleBisector (Angle B A C) R (LineSegment H M)

-- Prove that points P, B, Q, R are concyclic
theorem concyclic_points (H : Orthocenter A B C) (M : MidPoint A C) :
  ∀ (A B C A1 C1 P Q R : Point),
    non_isosceles_acute_triangle (AcuteAngle (TriangleAngle A B C)) (AcuteAngle (TriangleAngle B A C)) →
    AA1_altitude H →
    CC1_altitude H →
    angle_bisectors H →
    B_angle_bisector H M →
    Concyclic P B Q R :=
sorry

end concyclic_points_l441_441647


namespace cubic_sum_identity_l441_441930

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + a * c + b * c = -6) (h3 : a * b * c = -3) :
  a^3 + b^3 + c^3 = 27 :=
by
  sorry

end cubic_sum_identity_l441_441930


namespace symmetric_intersection_range_l441_441204

theorem symmetric_intersection_range (k m p : ℝ)
  (intersection_symmetric : ∀ (x y : ℝ), 
    (x = k*y - 1 ∧ (x^2 + y^2 + k*x + m*y + 2*p = 0)) → 
    (y = x)) 
  : p < -3/2 := 
sorry

end symmetric_intersection_range_l441_441204


namespace isosceles_triangle_of_parallel_l441_441763

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441763


namespace probability_sum_multiple_three_l441_441387

theorem probability_sum_multiple_three :
  let outcomes := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes := { (a, b, c) ∈ outcomes × outcomes × outcomes | (a + b + c) % 3 = 0 }
  (favorable_outcomes.to_finset.card / outcomes.to_finset.card ^ 3 : ℚ) = 1 / 3 :=
by
  sorry

end probability_sum_multiple_three_l441_441387


namespace find_k_l441_441184

variable (a b : ℝ × ℝ)
variable k : ℝ

def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Given vectors a and b, where a = (1, 2) and b = (4, k)
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (k : ℝ) : ℝ × ℝ := (4, k)

-- The condition that (2 * a - b) is perpendicular to (2 * a + b)
def condition (k : ℝ) : Prop :=
  let u := (2 * vector_a.1 - vector_b k.1, 2 * vector_a.2 - vector_b k.2)
  let v := (2 * vector_a.1 + vector_b k.1, 2 * vector_a.2 + vector_b k.2)
  perpendicular u v

-- The theorem stating the value of k
theorem find_k (k : ℝ) : condition k → (k = 2 ∨ k = -2) :=
sorry

end find_k_l441_441184


namespace triangle_ABC_is_isosceles_l441_441908

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441908


namespace system_of_equations_solution_l441_441369

theorem system_of_equations_solution :
  ∃ x y : ℝ, (2 * x + y = 6) ∧ (x - y = 3) ∧ (x = 3) ∧ (y = 0) :=
by
  sorry

end system_of_equations_solution_l441_441369


namespace triangle_ABC_is_isosceles_l441_441911

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441911


namespace negation_of_p_l441_441630

theorem negation_of_p :
  (¬ ∃ x : ℝ, sin x ≥ 1) ↔ (∀ x : ℝ, sin x < 1) :=
by
  sorry

end negation_of_p_l441_441630


namespace largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441411

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441411


namespace period_of_tan_3x_plus_pi_div_4_l441_441423

theorem period_of_tan_3x_plus_pi_div_4 :
  (∃ p : ℝ, ∀ x : ℝ, tan (3 * (x + p) + π/4) = tan (3 * x + π/4)) ∧ ∀ p' : ℝ, (∀ x : ℝ, tan (3 * (x + p') + π/4) = tan (3 * x + π/4) → p' = π/3) :=
by
  sorry

end period_of_tan_3x_plus_pi_div_4_l441_441423


namespace travis_total_cost_l441_441972

namespace TravelCost

def cost_first_leg : ℝ := 1500
def discount_first_leg : ℝ := 0.25
def fees_first_leg : ℝ := 100

def cost_second_leg : ℝ := 800
def discount_second_leg : ℝ := 0.20
def fees_second_leg : ℝ := 75

def cost_third_leg : ℝ := 1200
def discount_third_leg : ℝ := 0.35
def fees_third_leg : ℝ := 120

def discounted_cost (cost : ℝ) (discount : ℝ) : ℝ :=
  cost - (cost * discount)

def total_leg_cost (cost : ℝ) (discount : ℝ) (fees : ℝ) : ℝ :=
  (discounted_cost cost discount) + fees

def total_journey_cost : ℝ :=
  total_leg_cost cost_first_leg discount_first_leg fees_first_leg + 
  total_leg_cost cost_second_leg discount_second_leg fees_second_leg + 
  total_leg_cost cost_third_leg discount_third_leg fees_third_leg

theorem travis_total_cost : total_journey_cost = 2840 := by
  sorry

end TravelCost

end travis_total_cost_l441_441972


namespace curve_C_polar_equivalence_l441_441222

-- Define the curve C using parametric equations
def curve_C_parametric (α : ℝ) : ℝ × ℝ :=
  (Real.cos α + 1, Real.sin α)

-- Define the Cartesian form of the curve
def curve_C_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Define the polar form of the curve
def curve_C_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ

-- The proof statement: Prove that the given parametric curve in Cartesian coordinates transforms to the given polar form
theorem curve_C_polar_equivalence (α : ℝ) (ρ θ : ℝ) :
  curve_C_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ) → curve_C_polar ρ θ :=
by
  sorry

end curve_C_polar_equivalence_l441_441222


namespace solve_poly_eq_l441_441340

-- Define omega as the cube root of unity
def omega := Complex.exp(2 * Real.pi * Complex.I / 3)

-- Define the polynomial equation
def poly_eq (x : ℂ) : ℂ := (x - Complex.sqrt 3) ^ 4 + (x - Complex.sqrt 3)

-- Theorem statement
theorem solve_poly_eq :
  {x : ℂ // poly_eq x = 0} = {Complex.sqrt 3, -1 + Complex.sqrt 3, omega - 1 + Complex.sqrt 3, omega^2 - 1 + Complex.sqrt 3} :=
by
-- Proof goes here
sorry

end solve_poly_eq_l441_441340


namespace baron_not_boasting_l441_441496

-- Define a function to verify if a given list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- Define a list that represents the sequence given in the solution
def sequence_19 : List ℕ :=
  [9, 18, 7, 16, 5, 14, 3, 12, 1, 10, 11, 2, 13, 4, 15, 6, 17, 8, 19]

-- Prove that the sequence forms a palindrome
theorem baron_not_boasting : is_palindrome sequence_19 :=
by {
  -- Insert actual proof steps here
  sorry
}

end baron_not_boasting_l441_441496


namespace isosceles_triangle_l441_441791

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441791


namespace trig_identity_iff_pi_over_2_l441_441558

variable (α β : ℝ)

theorem trig_identity_iff_pi_over_2 (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  (α + β = π / 2) ↔ (sin α ^ 3 / cos β + cos α ^ 3 / sin β = 1) :=
sorry

end trig_identity_iff_pi_over_2_l441_441558


namespace tangent_iff_right_angled_l441_441651

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Definitions based on given conditions
def is_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  ∃ (a b c : A), A = a ∧ B = b ∧ C = c

def internal_angle_bisector (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : A :=
  sorry -- Assume existence of internal angle bisector AL_a

def external_angle_bisector (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : A :=
  sorry -- Assume existence of external angle bisector AM_a

def circumscribed_circle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : A :=
  sorry -- Assume existence of circumscribed circle Ω_a

def symmetric_circle (A : Type) [MetricSpace A] : A :=
  sorry -- Assume existence of symmetric circle ω_a

def apollonian_circle {A : Type} (A B C ω_a : A) [MetricSpace A] : A :=
  sorry -- Define ω_a Apollonian circle as per given property

def right_angled_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  ∠ACB = 90°

-- Proof equivalence
theorem tangent_iff_right_angled (ω_a ω_b : Type) (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] : 
  symmetric_circle A = ω_a ∧ symmetric_circle B = ω_b →
  (∃ (A B C : Type), is_triangle A B C) →
  (right_angled_triangle A B C ↔ (ω_a ∧ ω_b).tangent) :=
sorry

end tangent_iff_right_angled_l441_441651


namespace cost_per_slipper_l441_441095

-- Define the cost of each type of item
def cost_lipstick := 1.25
def cost_hair_color := 3.0

-- Define the quantities ordered
def count_slippers := 6
def count_lipsticks := 4
def count_hair_colors := 8

-- Total amount paid
def total_paid := 44.0

-- Prove the cost per slipper
theorem cost_per_slipper: 
  (total_paid - (count_lipsticks * cost_lipstick + count_hair_colors * cost_hair_color)) / count_slippers = 2.5 :=
by
  sorry

end cost_per_slipper_l441_441095


namespace triangle_is_isosceles_l441_441890

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441890


namespace soap_brand_ratio_l441_441471

theorem soap_brand_ratio :
  ∀ (total households not_using either brand A B households only_A both_A_B only_B : ℕ),
  total = 260 →
  households not_using = 80 →
  households only_A = 60 →
  households both_A_B = 30 →
  households = total - households not_using →
  only_B = households - households only_A - households both_A_B →
  (only_B / gcd only_B households both_A_B = 3) ∧ (households both_A_B / gcd only_B households both_A_B = 1) :=
by
  intros total households not_using either brand A B households only_A both_A_B only_B 
  sorry

end soap_brand_ratio_l441_441471


namespace triangle_is_isosceles_if_parallel_l441_441730

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441730


namespace ellipse_product_l441_441714

theorem ellipse_product:
  ∀ (P G X Y W Z : Type) [EllipticData P G X Y W Z], 
  PG = 9 → 
  inscribed_circle_pwq P W G = 6 →
  ∃ a b: ℕ, 
  a = PX ∧ a = PY ∧ b = PW ∧ b = PZ ∧ a^2 - b^2 = 81 ∧ a - b = 3 
→ 
  (2 * a) * (2 * b) = 720 :=
begin
  sorry,
end

class EllipticData (P G X Y W Z : Type) :=
  (PG: P → G → ℕ)
  (inscribed_circle_pwq : P → W → G → ℕ)
  (PX PY PW PZ: P → X → ℕ)

end ellipse_product_l441_441714


namespace non_zero_number_is_nine_l441_441447

theorem non_zero_number_is_nine (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 :=
by
  sorry

end non_zero_number_is_nine_l441_441447


namespace some_magical_creatures_are_mystical_beings_l441_441625

-- Definitions and conditions based on the problem
def Dragon : Type := sorry
def MagicalCreature : Type := sorry
def MysticalBeing : Type := sorry

-- Main condition: All dragons are magical creatures
axiom (all_dragons_are_magical : ∀ d : Dragon, MagicalCreature)

-- Another condition: Some mystical beings are dragons
axiom (some_mystical_beings_are_dragons : ∃ m : MysticalBeing, ∃ d : Dragon, m = d)

-- Our task is to prove the required statement:
theorem some_magical_creatures_are_mystical_beings : ∃ mc : MagicalCreature, ∃ mb : MysticalBeing, mc = mb :=
begin
  sorry
end

end some_magical_creatures_are_mystical_beings_l441_441625


namespace product_area_ratios_eq_l441_441103

def area_polygon (n : ℕ) (h : n ≥ 2) : ℝ :=
  2^(n-1) * Real.sin (2 * Real.pi / 2^n)

theorem product_area_ratios_eq :
  (∏ i in Finset.Ico 2 ∞, area_polygon i (Finset.le_refl) / area_polygon (i+1) (Nat.succ_le_succ (Finset.le_refl))) = 2 / Real.pi :=
by
  sorry

end product_area_ratios_eq_l441_441103


namespace range_of_a_l441_441168

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x)
  (hf_expr_pos : ∀ x, x > 0 → f x = -x^2 + ax - 1 - a)
  (hf_monotone : ∀ x y, x < y → f y ≤ f x) :
  -1 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l441_441168


namespace last_two_digits_sum_factorials_l441_441132

theorem last_two_digits_sum_factorials : 
  (3! + 7! + 11! + 15! + 19! + 23! + 27! + 31! + 35! + 39! + 43! + 47! + 51! + 55! + 59! + 63! + 67! + 71! + 75! + 79! + 83! + 87! + 91! + 95! + 99!) % 100 = 46 :=
by {
  sorry
}

end last_two_digits_sum_factorials_l441_441132


namespace snow_white_last_trip_l441_441319

-- Definitions based on the problem's conditions
inductive Dwarf
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

def is_adjacent (d1 d2 : Dwarf) : Prop :=
  (d1, d2) ∈ [
    (Dwarf.Happy, Dwarf.Grumpy),
    (Dwarf.Grumpy, Dwarf.Happy),
    (Dwarf.Grumpy, Dwarf.Dopey),
    (Dwarf.Dopey, Dwarf.Grumpy),
    (Dwarf.Dopey, Dwarf.Bashful),
    (Dwarf.Bashful, Dwarf.Dopey),
    (Dwarf.Bashful, Dwarf.Sleepy),
    (Dwarf.Sleepy, Dwarf.Bashful),
    (Dwarf.Sleepy, Dwarf.Doc),
    (Dwarf.Doc, Dwarf.Sleepy),
    (Dwarf.Doc, Dwarf.Sneezy),
    (Dwarf.Sneezy, Dwarf.Doc)
  ]

def boat_capacity : ℕ := 3

variable (snowWhite : Prop)

-- The theorem to prove that the dwarfs Snow White will take in the last trip are Grumpy, Bashful and Doc
theorem snow_white_last_trip 
  (h1 : snowWhite)
  (h2 : boat_capacity = 3)
  (h3 : ∀ d1 d2, is_adjacent d1 d2 → snowWhite)
  : (snowWhite ∧ (Dwarf.Grumpy ∧ Dwarf.Bashful ∧ Dwarf.Doc)) :=
sorry

end snow_white_last_trip_l441_441319


namespace correct_average_of_set_l441_441062

theorem correct_average_of_set 
  (numbers : Fin 20 → ℤ)
  (incorrect_average : ℚ := 32) 
  (incorrect_readings : Vector (ℤ × ℤ) 4 := [(65, 45), (-42, -28), (75, 55), (-35, -25)]) 
  (incorrect_sum : ℤ := incorrect_average * 20) :
  let correction_factors := [20, -14, 20, -10]
  let total_correction := correction_factors.sum
  let correct_sum := incorrect_sum + total_correction
  let correct_average := correct_sum / 20
  correct_average = 32.8 := by 
  sorry

end correct_average_of_set_l441_441062


namespace isosceles_triangle_l441_441821

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441821


namespace triangle_is_isosceles_l441_441850

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441850


namespace solution_set_l441_441597

def f (x : ℝ) := Real.sin x - x

theorem solution_set (x : ℝ) : f (x + 2) + f (1 - 2 * x) < 0 ↔ x < 3 :=
  sorry

end solution_set_l441_441597


namespace triangle_is_isosceles_l441_441852

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441852


namespace range_of_m_l441_441159

theorem range_of_m (x : ℝ) (m : ℝ) (h : sin x = m - 1) : 0 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l441_441159


namespace rohan_monthly_salary_l441_441920

theorem rohan_monthly_salary :
  ∃ S : ℝ, 
    (0.4 * S) + (0.2 * S) + (0.1 * S) + (0.1 * S) + 1000 = S :=
by
  sorry

end rohan_monthly_salary_l441_441920


namespace find_r_of_tangential_cones_l441_441969

theorem find_r_of_tangential_cones (r : ℝ) : 
  (∃ (r1 r2 r3 R : ℝ), r1 = 2 * r ∧ r2 = 3 * r ∧ r3 = 10 * r ∧ R = 15 ∧
  -- Additional conditions to ensure the three cones touch and share a slant height
  -- with the truncated cone of radius R
  true) → r = 29 :=
by
  intro h
  sorry

end find_r_of_tangential_cones_l441_441969


namespace problem_statement_l441_441072

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) := 
  ∀ x1 x2, x1 ∈ s → x2 ∈ s → x1 < x2 → f x1 < f x2

theorem problem_statement :
  ∀ x1 x2 : ℝ, (0 < x1 ∧ 0 < x2) → x1 < x2 → ln (x1 + 1) < ln (x2 + 1) :=
by
  sorry

end problem_statement_l441_441072


namespace ratio_of_first_to_second_ball_l441_441240

theorem ratio_of_first_to_second_ball 
  (x y z : ℕ) 
  (h1 : 3 * x = 27) 
  (h2 : y = 18) 
  (h3 : z = 3 * x) : 
  x / y = 1 / 2 := 
sorry

end ratio_of_first_to_second_ball_l441_441240


namespace BLCK_cyclic_l441_441685

variable (A B C D O K L : Type)
variable [IsCyclicQuadrilateral A B C D]
variable [Intersection O (Diag A C) (Diag B D)]
variable [Intersection K (Circumcircle A B O) (Circumcircle C O D)]
variable [Similarity (Triangle B L C) (Triangle A K D)]
variable [ConvexQuadrilateral B L C K]

theorem BLCK_cyclic : CyclicQuadrilateral B L C K :=
sorry

end BLCK_cyclic_l441_441685


namespace ellipse_k_range_ellipse_k_eccentricity_l441_441596

theorem ellipse_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) ↔ (1 < k ∧ k < 5 ∨ 5 < k ∧ k < 9) := 
sorry

theorem ellipse_k_eccentricity (k : ℝ) (h : ∃ x y : ℝ, x^2/(9 - k) + y^2/(k - 1) = 1) : 
  eccentricity = Real.sqrt (6/7) → (k = 2 ∨ k = 8) := 
sorry

end ellipse_k_range_ellipse_k_eccentricity_l441_441596


namespace triangle_is_isosceles_l441_441891

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441891


namespace triangle_isosceles_if_parallel_l441_441864

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441864


namespace five_digit_numbers_valid_count_l441_441186

def valid_five_digit_numbers_count : ℕ :=
  let valid_first_digit := 7 -- choices from 3 to 9 (greater than 29999)
  let valid_last_digit := 10 -- choices from 0 to 9
  let total_middle_combinations := 9 * 9 * 9 -- all middle digit combinations
  let excluded_combinations := 50 -- estimate of invalid combinations with product ≤ 10
  let valid_middle_combinations := total_middle_combinations - excluded_combinations
  in valid_first_digit * valid_middle_combinations * valid_last_digit

theorem five_digit_numbers_valid_count : valid_five_digit_numbers_count = 47930 :=
by {
  -- Mathematically equivalent proof problem adapted from the original problem statement
  sorry -- proof to be done
}

end five_digit_numbers_valid_count_l441_441186


namespace largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441409

theorem largest_integer_not_sum_of_multiple_of_30_and_composite :
  ∃ n : ℕ, ∀ a b : ℕ, 0 ≤ b ∧ b < 30 ∧ b.prime ∧ (∀ k < a, (b + 30 * k).prime)
    → (30 * a + b = n) ∧
      (∀ m : ℕ, ∀ c d : ℕ, 0 ≤ d ∧ d < 30 ∧ d.prime ∧ (∀ k < c, (d + 30 * k).prime) 
        → (30 * c + d ≤ n)) ∧
      n = 93 :=
by
  sorry

end largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441409


namespace shuttle_speed_in_km_per_sec_l441_441065

variable (speed_mph : ℝ) (miles_to_km : ℝ) (hour_to_sec : ℝ)

theorem shuttle_speed_in_km_per_sec
  (h_speed_mph : speed_mph = 18000)
  (h_miles_to_km : miles_to_km = 1.60934)
  (h_hour_to_sec : hour_to_sec = 3600) :
  (speed_mph * miles_to_km) / hour_to_sec = 8.046 := by
sorry

end shuttle_speed_in_km_per_sec_l441_441065


namespace nearest_integer_to_sum_l441_441451

theorem nearest_integer_to_sum (a b c : ℚ) (h₁ : a = 2007 / 2999) (h₂ : b = 8001 / 5998) (h₃ : c = 2001 / 3999) :
  Int.nearest (a + b + c) = 3 :=
by
  sorry

end nearest_integer_to_sum_l441_441451


namespace problem_statement_l441_441742

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441742


namespace isosceles_triangle_of_parallel_l441_441767

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441767


namespace yellow_yellow_pairs_count_l441_441091

theorem yellow_yellow_pairs_count (N_blue N_yellow : ℕ) (total_pairs : ℕ)
    (blue_pairs : ℕ) : ℕ :=
  let blue_students_in_blue_pairs := blue_pairs * 2
  let blue_students_in_mixed_pairs := N_blue - blue_students_in_blue_pairs
  let yellow_students_in_mixed_pairs := blue_students_in_mixed_pairs
  let yellow_students_in_yellow_pairs := N_yellow - yellow_students_in_mixed_pairs
  let yellow_pairs := yellow_students_in_yellow_pairs / 2
  yellow_pairs

example : yellow_yellow_pairs_count 60 84 72 25 = 37 := by
  unfold yellow_yellow_pairs_count
  norm_num
  sorry

end yellow_yellow_pairs_count_l441_441091


namespace length_diagonal_PC_l441_441482

variable (s x : ℝ) -- Declare the side of the square and the length of AP, PB, DR, and RC.

-- Assume the given conditions.
axiom square_side : s^2 = 2 * x^2
axiom triangle_area : 2 * (1/2 * x^2) = 72

theorem length_diagonal_PC : (PC : ℝ) := 
  let AP := x
  let PB := x
  let DR := x
  let RC := x
  let side_diff := s - 12 * Real.sqrt 2 in
  let diagonal_length := 2 * side_diff ^ 2 in
  let square_diagonal := s in
  square_diagonal - 24 = 12
  sorry

end length_diagonal_PC_l441_441482


namespace final_trip_l441_441331

-- Definitions for the conditions
def dwarf := String
def snow_white := "Snow White" : dwarf
def Happy : dwarf := "Happy"
def Grumpy : dwarf := "Grumpy"
def Dopey : dwarf := "Dopey"
def Bashful : dwarf := "Bashful"
def Sleepy : dwarf := "Sleepy"
def Doc : dwarf := "Doc"
def Sneezy : dwarf := "Sneezy"

-- The dwarfs lineup from left to right
def lineup : List dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- The boat can hold Snow White and up to 3 dwarfs
def boat_capacity (load : List dwarf) : Prop := snow_white ∈ load ∧ load.length ≤ 4

-- Any two dwarfs standing next to each other in the original lineup will quarrel if left without Snow White
def will_quarrel (d1 d2 : dwarf) : Prop :=
  (d1, d2) ∈ (lineup.zip lineup.tail)

-- The objective: Prove that on the last trip, Snow White will take Grumpy, Bashful, and Doc
theorem final_trip : ∃ load : List dwarf, 
  set.load ⊆ {snow_white, Grumpy, Bashful, Doc} ∧ boat_capacity load :=
  sorry

end final_trip_l441_441331


namespace value_of_x_l441_441430

theorem value_of_x (x : ℕ) : (8^4 + 8^4 + 8^4 = 2^x) → x = 13 :=
by
  sorry

end value_of_x_l441_441430


namespace set_difference_correct_l441_441517

variable M : Set ℕ := {1, 3, 5, 7, 9}
variable N : Set ℕ := {2, 3, 5}

noncomputable def set_difference (A B : Set ℕ) : Set ℕ := { x | x ∈ A ∧ x ∉ B }

theorem set_difference_correct : set_difference M N = {1, 7, 9} :=
  by
  sorry

end set_difference_correct_l441_441517


namespace isosceles_triangle_l441_441825

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441825


namespace projection_onto_v_l441_441954

open Matrix

-- Define the vectors and projections
def u : ℝ^3 := ![1, 2, 3]
def v_proj : ℝ^3 := ![2, 1, -1]
def w : ℝ^3 := ![4, -1, 0]
def w_proj : ℝ^3 := ![7/3, 7/6, -7/6]

noncomputable def v : ℝ^3 := (2 : ℝ) • v_proj

-- Projection function for 3D vectors
def proj (a b : ℝ^3) : ℝ^3 :=
  ((inner a b) / (inner b b)) • b

-- The theorem we need to prove
theorem projection_onto_v (huv : proj u v = v_proj) : proj w v = w_proj :=
by
  sorry

end projection_onto_v_l441_441954


namespace find_other_number_l441_441449

def HCF (a b : ℕ) : ℕ := sorry
def LCM (a b : ℕ) : ℕ := sorry

theorem find_other_number (B : ℕ) 
 (h1 : HCF 24 B = 15) 
 (h2 : LCM 24 B = 312) 
 : B = 195 := 
by
  sorry

end find_other_number_l441_441449


namespace triangle_ABC_is_isosceles_l441_441916

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441916


namespace isosceles_triangle_of_parallel_l441_441768

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441768


namespace sin_add_angle_eq_24_over_25_l441_441213

theorem sin_add_angle_eq_24_over_25
  (A B : ℝ)
  (h_acute_triangle : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < A + B ∧ A + B < π)
  (h_B_gt_pi_six : B > π / 6)
  (h_sin_A_plus_pi_six : sin (A + π / 6) = 3 / 5)
  (h_cos_B_minus_pi_six : cos (B - π / 6) = 4 / 5) :
  sin (A + B) = 24 / 25 :=
by sorry

end sin_add_angle_eq_24_over_25_l441_441213


namespace cube_sin_angle_SPC_eq_one_l441_441456

-- Defining the cube and positions
structure Cube :=
(face1 : ℝ × ℝ × ℝ)
(face_center : ℝ × ℝ × ℝ)
(opposite_vertex : ℝ × ℝ × ℝ)

-- Defining a function to calculate the sine of an angle within the cube
noncomputable def sin_angle_SPC (c : Cube) : ℝ :=
  let P := c.face_center in
  let S := c.face1 in
  let C := c.opposite_vertex in
  -- Since \(P\) is the center of a face in a cube and \(C\) is the opposite corner
  -- \(\angle SPC\) forms a right angle with \(\sin\) value being 1.
  1

-- The statement to be proved in Lean
theorem cube_sin_angle_SPC_eq_one (c : Cube) : sin_angle_SPC c = 1 :=
sorry

end cube_sin_angle_SPC_eq_one_l441_441456


namespace symmetric_y_axis_l441_441610

theorem symmetric_y_axis (θ : ℝ) :
  (cos θ = -cos (θ + π / 6)) ∧ (sin θ = sin (θ + π / 6)) → θ = 5 * π / 12 :=
by
  sorry

end symmetric_y_axis_l441_441610


namespace quadratic_eq_solutions_l441_441566

noncomputable def quadratic_solution (a : ℝ) (x : ℝ) : Prop :=
x^2 - (1 - a) * x - 2 = 0

theorem quadratic_eq_solutions (a : ℝ) (x : ℝ) :
  (a ∈ {-1, 1, a^2}) → (quadratic_solution a x ↔ (x = -1 ∨ x = 2)) :=
by
  sorry

end quadratic_eq_solutions_l441_441566


namespace modulus_c_for_distinct_roots_l441_441691

theorem modulus_c_for_distinct_roots (c : ℂ) :
  (∃ (p : polynomial ℂ), p = (X^2 - 2*X + 2) * (X^2 - c*X + 5) * (X^2 - 5*X + 13) ∧ p.roots.nodup ∧ p.root_set ℂ).card = 4 → |c| = real.sqrt 12.5 :=
begin
  sorry,
end

end modulus_c_for_distinct_roots_l441_441691


namespace sum_of_angles_is_290_l441_441210

-- Given conditions
def angle_A : ℝ := 40
def angle_C : ℝ := 70
def angle_D : ℝ := 50
def angle_F : ℝ := 60

-- Calculate angle B (which is same as angle E)
def angle_B : ℝ := 180 - angle_A - angle_C
def angle_E := angle_B  -- by the condition that B and E are identical

-- Total sum of angles
def total_angle_sum : ℝ := angle_A + angle_B + angle_C + angle_D + angle_F

-- Theorem statement
theorem sum_of_angles_is_290 : total_angle_sum = 290 := by
  sorry

end sum_of_angles_is_290_l441_441210


namespace percentage_increase_correct_l441_441656

-- Defining initial conditions
variables (B H : ℝ) -- bears per week without assistant and hours per week without assistant

-- Defining the rate of making bears per hour without assistant
def rate_without_assistant := B / H

-- Defining the rate with an assistant (100% increase in output per hour)
def rate_with_assistant := 2 * rate_without_assistant

-- Defining the number of hours worked per week with an assistant (10% fewer hours)
def hours_with_assistant := 0.9 * H

-- Calculating the number of bears made per week with an assistant
def bears_with_assistant := rate_with_assistant * hours_with_assistant

-- Calculating the percentage increase in the number of bears made per week when Jane works with an assistant
def percentage_increase : ℝ := ((bears_with_assistant / B) - 1) * 100

-- The theorem to prove
theorem percentage_increase_correct : percentage_increase B H = 80 :=
  by sorry

end percentage_increase_correct_l441_441656


namespace fraction_of_marbles_taken_away_l441_441507

theorem fraction_of_marbles_taken_away (Chris_marbles Ryan_marbles remaining_marbles total_marbles taken_away_marbles : ℕ) 
    (hChris : Chris_marbles = 12) 
    (hRyan : Ryan_marbles = 28) 
    (hremaining : remaining_marbles = 20) 
    (htotal : total_marbles = Chris_marbles + Ryan_marbles) 
    (htaken_away : taken_away_marbles = total_marbles - remaining_marbles) : 
    (taken_away_marbles : ℚ) / total_marbles = 1 / 2 := 
by 
  sorry

end fraction_of_marbles_taken_away_l441_441507


namespace arithmetic_sequence_sum_lt_2_l441_441224

noncomputable def a : ℕ → ℕ
| 0 => 0
| n+1 => n+1

noncomputable def b : ℕ → ℝ
| n => (1 / 2) ^ (a n)

theorem arithmetic_sequence_sum_lt_2 :
  ∀ n, (∑ i in finset.range n, (a (i+1)) * (b (i+1))) < 2 :=
by
  sorry

end arithmetic_sequence_sum_lt_2_l441_441224


namespace polynomial_representation_l441_441684

theorem polynomial_representation
  (F G : Polynomial ℤ)
  (hF : ∀ i, F.coeff i = 0 ∨ F.coeff i = 1)
  (hG : ∀ i, G.coeff i = 0 ∨ G.coeff i = 1)
  (hFG : (1 + X + X^2 + ... + X^(n-1) = F * G))
  (hn : n > 1) :
  ∃ k (T : Polynomial ℤ), k > 1 ∧ (∀ i, T.coeff i = 0 ∨ T.coeff i = 1) ∧ 
  ((G = (1 + X + X^2 + ... + X^(k-1)) * T) ∨ (F = (1 + X + X^2 + ... + X^(k-1)) * T)) :=
begin
  sorry
end

end polynomial_representation_l441_441684


namespace isosceles_triangle_of_parallel_l441_441789

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441789


namespace triangle_is_isosceles_l441_441849

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441849


namespace centers_coincide_l441_441965

variables {A B C D A₁ B₁ C₁ D₁ : Type}
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
variables [AddCommGroup A₁] [AddCommGroup B₁] [AddCommGroup C₁] [AddCommGroup D₁]

/-- Given the parallelograms ABCD and A₁B₁C₁D₁ with vertices A₁, B₁, C₁, D₁ on sides AB, BC, CD, DA of ABCD, show that their centers coincide -/
theorem centers_coincide
  (hA₁ : A₁ ∈ segment A B)
  (hB₁ : B₁ ∈ segment B C)
  (hC₁ : C₁ ∈ segment C D)
  (hD₁ : D₁ ∈ segment D A)
  (centroid_ABCD : Type) -- Placeholder for the proof of centroid relation
  (centroid_A₁B₁C₁D₁ : Type) -- Placeholder for the proof of centroid relation
  (hcoincide : centroid_ABCD = centroid_A₁B₁C₁D₁) :
  ∃ O, O = centroid_ABCD ∧ O = centroid_A₁B₁C₁D₁ :=
sorry

end centers_coincide_l441_441965


namespace snow_white_last_trip_dwarfs_l441_441303

-- Definitions for the conditions
def original_lineup := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]
def only_snow_white_can_row := True
def boat_capacity_snow_white_and_dwarfs := 3
def dwarfs_quarrel_if_adjacent (d1 d2 : String) : Prop :=
  let index_d1 := List.indexOf original_lineup d1
  let index_d2 := List.indexOf original_lineup d2
  abs (index_d1 - index_d2) = 1

-- Theorem to prove the correct answer
theorem snow_white_last_trip_dwarfs :
  let last_trip_dwarfs := ["Grumpy", "Bashful", "Sneezy"]
  ∃ (trip : List String), trip = last_trip_dwarfs ∧ 
  ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → d1 ≠ d2 → ¬dwarfs_quarrel_if_adjacent d1 d2 :=
by
  sorry

end snow_white_last_trip_dwarfs_l441_441303


namespace box_dimensions_l441_441626

noncomputable def cube_volume : ℝ := 2

noncomputable def box_volume_40_percent (a1 a2 a3 : ℝ) : Prop :=
  0.4 * (a1 * a2 * a3) = 2 * ⌊a1 / (∛cube_volume)⌋ * ⌊a2 / (∛cube_volume)⌋ * ⌊a3 / (∛cube_volume)⌋

theorem box_dimensions (a1 a2 a3 : ℝ) (h1 : a1 ≤ a2) (h2 : a2 ≤ a3) :
  box_volume_40_percent a1 a2 a3 →
  (a1 = 2 ∧ a2 = 3 ∧ a3 = 5) ∨ (a1 = 2 ∧ a2 = 5 ∧ a3 = 6) :=
by
  sorry

end box_dimensions_l441_441626


namespace find_unknown_number_l441_441347

theorem find_unknown_number
  (S : ℕ)
  (X : ℕ)
  (h1 : S / 50 = 38)
  (h2 : (S - X - 55) / 48 = 37.5) :
  X = 45 :=
sorry

end find_unknown_number_l441_441347


namespace triangle_is_isosceles_if_parallel_l441_441725

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441725


namespace sequence_solution_l441_441648

theorem sequence_solution : ∀ n : ℕ, let a : ℕ → ℝ := λ n,
  if n = 0 then 2 else (a n) / (1 + 3 * (a n)) in
  a n = 2 / (6 * n - 5) := 
  sorry

end sequence_solution_l441_441648


namespace isosceles_triangle_of_parallel_l441_441784

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441784


namespace ramesh_paid_price_l441_441282

variable (P : ℝ) (P_paid : ℝ)

-- conditions
def discount_price (P : ℝ) : ℝ := 0.80 * P
def additional_cost : ℝ := 125 + 250
def total_cost_with_discount (P : ℝ) : ℝ := discount_price P + additional_cost
def selling_price_without_discount (P : ℝ) : ℝ := 1.10 * P
def given_selling_price : ℝ := 18975

-- the theorem to prove
theorem ramesh_paid_price :
  (∃ P : ℝ, selling_price_without_discount P = given_selling_price ∧ total_cost_with_discount P = 14175) :=
by
  sorry

end ramesh_paid_price_l441_441282


namespace g_of_a_l441_441556

theorem g_of_a (a : ℝ) (h : a ≥ 0) (m n : ℝ)
  (h1 : (∀ y, y = 2^|x| ↔ -2 ≤ x ∧ x ≤ a) ↔ [m, n]) :
  g(a) = if 0 ≤ a ∧ a ≤ 2 then 3 else 1 - 2^a :=
by
  sorry

end g_of_a_l441_441556


namespace isosceles_triangle_angles_l441_441642

theorem isosceles_triangle_angles (h : ℝ) (A B C : Type) :
  ∀ (AB AC : ℝ), 
  ∀ (BC : ℝ), 
  ∀ (x : ℝ),
  (BC = 2 * h) ∧ (AB = AC) ∧ (BC = 2 * h) ∧ (∀ (M : ℝ), (BM = MC = h) ∧ (AM = h) ∧ (x = 45)) 
  → ( angle A B C == 45 ∧ angle B A C == 45 ∧ angle C A B == 90) :=
by
  sorry

end isosceles_triangle_angles_l441_441642


namespace problem_statement_l441_441736

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441736


namespace snow_white_last_trip_l441_441335

universe u

-- Define the dwarf names as an enumerated type
inductive Dwarf : Type
| Happy | Grumpy | Dopey | Bashful | Sleepy | Doc | Sneezy
deriving DecidableEq, Repr

open Dwarf

-- Define conditions
def initial_lineup : List Dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- Define the condition of adjacency
def adjacent (d1 d2 : Dwarf) : Prop :=
  List.pairwise_adjacent (· = d2) initial_lineup d1

-- Define the boat capacity
def boat_capacity : Fin 4 := by sorry

-- Snow White is the only one who can row
def snow_white_rows : Prop := true

-- No quarrels condition
def no_quarrel_without_snow_white (group : List Dwarf) : Prop :=
  ∀ d1 d2, d1 ∈ group → d2 ∈ group → ¬ adjacent d1 d2

-- Objective: Transfer all dwarfs without quarrels
theorem snow_white_last_trip (trips : List (List Dwarf)) :
  ∃ last_trip : List Dwarf,
    last_trip = [Grumpy, Bashful, Doc] ∧
    no_quarrel_without_snow_white (initial_lineup.diff (trips.join ++ last_trip)) :=
sorry

end snow_white_last_trip_l441_441335


namespace isosceles_triangle_l441_441801

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441801


namespace final_trip_theorem_l441_441296

/-- Define the lineup of dwarfs -/
inductive Dwarf where
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

open Dwarf

/-- Define the conditions -/
-- The dwarfs initially standing next to each other
def adjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = Happy ∧ d2 = Grumpy) ∨
  (d1 = Grumpy ∧ d2 = Dopey) ∨
  (d1 = Dopey ∧ d2 = Bashful) ∨
  (d1 = Bashful ∧ d2 = Sleepy) ∨
  (d1 = Sleepy ∧ d2 = Doc) ∨
  (d1 = Doc ∧ d2 = Sneezy)

-- Snow White is the only one who can row
constant snowWhite_can_row : Prop := true

-- The boat can hold Snow White and up to 3 dwarfs
constant boat_capacity : ℕ := 4

-- Define quarrel if left without Snow White
def quarrel_without_snowwhite (d1 d2 : Dwarf) : Prop := adjacent d1 d2

-- Define the final trip setup
def final_trip (dwarfs : List Dwarf) : Prop :=
  dwarfs = [Grumpy, Bashful, Doc]

-- Theorem to prove the final trip
theorem final_trip_theorem : ∃ dwarfs, final_trip dwarfs :=
  sorry

end final_trip_theorem_l441_441296


namespace focus_of_parabola_l441_441114

theorem focus_of_parabola (x y : ℝ) (h : y = 4 * x^2) : (0, 1 / 16) ∈ {p : ℝ × ℝ | ∃ x y, y = 4 * x^2 ∧ p = (0, 1 / (4 * (1 / y)))} :=
by
  sorry

end focus_of_parabola_l441_441114


namespace rectangle_area_perimeter_max_l441_441475

-- Define the problem conditions
variables {A P : ℝ}

-- Main statement: prove that the maximum value of A / P^2 for a rectangle results in m+n = 17
theorem rectangle_area_perimeter_max (h1 : A = l * w) (h2 : P = 2 * (l + w)) :
  let m := 1
  let n := 16
  m + n = 17 :=
sorry

end rectangle_area_perimeter_max_l441_441475


namespace translated_graph_min_point_l441_441357

theorem translated_graph_min_point :
  let original_min := (0, 2)
  let new_x := original_min.1 + 4
  let new_y := original_min.2 + 5
  (new_x, new_y) = (4, 7) :=
by
  let original_min := (0, 2)
  let new_x := original_min.1 + 4
  let new_y := original_min.2 + 5
  have h : (new_x, new_y) = (4, 7) := by
    dsimp [original_min, new_x, new_y]
    simp
  exact h

end translated_graph_min_point_l441_441357


namespace f_inequality_l441_441674

-- Defining the preconditions and the function
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (f' : ℝ → ℝ)

-- Defining the preconditions as hypotheses
hypothesis h1 : ∀ x, f (2 + x) = f (2 - x)
hypothesis h2 : ∀ x, (x - 2) * f' x > 0
hypothesis h3 : continuous f
hypothesis ha : 2 < a ∧ a < 4

-- Statement to be proved
theorem f_inequality : f 2 < f (Real.log 2 a) ∧ f (Real.log 2 a) < f (2 ^ a) :=
sorry

end f_inequality_l441_441674


namespace e3_is_quadratic_l441_441984

def is_quadratic (e : Expr) : Prop := 
  ∃ a b c : ℤ, e = a * x^2 + b * x + c = 0

noncomputable def e1 : Expr := x - 1 / (x - 1) = 0
noncomputable def e2 : Expr := 7 * x^2 + 1 / x^2 - 1 = 0
noncomputable def e3 : Expr := x^2 = 0
noncomputable def e4 : Expr := (x + 1) * (x - 2) = x * (x + 1)

theorem e3_is_quadratic : is_quadratic e3 :=
  sorry

end e3_is_quadratic_l441_441984


namespace rhombus_if_equal_perimeters_l441_441120

variable (A B C D O : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]
variables (AB AC BD : metric_space.dist (A , B)) (AV : metric_space.dist (O , A))
variable (OB : metric_space.dist (O , B)) (OC : metric_space.dist (O , C)) (OD : metric_space.dist (O , D))

def is_rhombus (A B C D O : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] 
    (AV : metric_space.dist (O , A)) (OB : metric_space.dist (O , B)) (OC : metric_space.dist (O , C)) (OD : metric_space.dist (O , D)) : Prop :=
  ∀ (a b c d p q r s : ℝ), 
    (AV = a) → (OB = b) → (OC = c) → (OD = d) 
    → a + b + p = b + c + q
    ∧ b + c + q = c + d + r
    ∧ c + d + r = d + a + s 
    ∧ d + a + s = a + b + p 
    → p + r = q + s 
    → a = b 
    ∧ b = c 
    → (metric_space.dist (p, r) = metric_space.dist (q, s)) 
    ∧ p = q 
    → r = s 
    → metric_space.dist (A , B) = metric_space.dist (C , D) 
    → metric_space.dist (A , C) = metric_space.dist (B , D) 
    → metric_space.dist (A , C) = metric_space.dist (B , D)
    → A ≈ B 
    ∧ B ≈ C 
    ∧ C ≈ D

theorem rhombus_if_equal_perimeters 
: ∀ (A B C D O : Type), (metric_space A) → (metric_space B) → (metric_space C) → (metric_space D) → (metric_space O) 
    → (AB:metric_space.dist (A,B)) → (AO:metric_space.dist (O,A))
    → (OB:metric_space.dist (O,B)) → (OC:metric_space.dist (O,C)) → (OD:metric_space.dist (O,D)) 
    → (∀ (a b c d p q r s : ℝ), 
    (metric_space.dist (A , O) = a) 
    → (metric_space.dist (O , B) = b) 
    → (metric_space.dist (O , C) = c) 
    → (metric_space.dist (O , D) = d) 
    → a + b + p = b + c + q
    ∧ b + c + q = c + d + r
    ∧ c + d + r = d + a + s 
    ∧ d + a + s = a + b + p 
    → p + r = q + s 
    → a = b 
    ∧ b = c 
    → metric_space.dist (p, r) = metric_space.dist (q, s) 
    ∧ p = q 
    → r = s 
    → is_rhombus A B C D O (metric_space.dist (A , B)) (metric_space.dist (A , O)))
  sorry

end rhombus_if_equal_perimeters_l441_441120


namespace problem1_problem2_l441_441032

-- Given conditions
def is_non_zero_integer (x : ℤ) : Prop := x ≠ 0
def abs_less_than (x : ℤ) (n : ℤ) : Prop := |x| < n
def conditions (a b c : ℤ) : Prop :=
  is_non_zero_integer a ∧ is_non_zero_integer b ∧ is_non_zero_integer c ∧ abs_less_than a 10^6 ∧ abs_less_than b 10^6 ∧ abs_less_than c 10^6

-- Proof problem 1
theorem problem1 : ∃ (a b c : ℤ), conditions a b c ∧ |a + b * (Real.sqrt 2) + c * (Real.sqrt 3)| < 10^(-11) :=
  sorry

-- Proof problem 2
theorem problem2 (a b c : ℤ) (h : conditions a b c) : |a + b * (Real.sqrt 2) + c * (Real.sqrt 3)| > 10^(-21) :=
  sorry

end problem1_problem2_l441_441032


namespace find_number_l441_441432

variable (x : ℝ)

theorem find_number : ((x * 5) / 2.5 - 8 * 2.25 = 5.5) -> x = 11.75 :=
by
  intro h
  sorry

end find_number_l441_441432


namespace surface_area_after_removing_corners_l441_441527

-- Define the dimensions of the cubes
def original_cube_side : ℝ := 4
def corner_cube_side : ℝ := 2

-- The surface area function for a cube with given side length
def surface_area (side : ℝ) : ℝ := 6 * side * side

theorem surface_area_after_removing_corners :
  surface_area original_cube_side = 96 :=
by
  sorry

end surface_area_after_removing_corners_l441_441527


namespace triangle_isosceles_if_parallel_l441_441871

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441871


namespace ana_bonita_age_difference_l441_441492

theorem ana_bonita_age_difference (A B n : ℕ) 
  (h1 : A = B + n)
  (h2 : A - 1 = 7 * (B - 1))
  (h3 : A = B^3) : 
  n = 6 :=
sorry

end ana_bonita_age_difference_l441_441492


namespace last_trip_l441_441317

def initial_order : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

def boatCapacity : Nat := 4  -- Including Snow White

def adjacentPairsQuarrel (adjPairs : List (String × String)) : Prop :=
  ∀ (d1 d2 : String), (d1, d2) ∈ adjPairs → (d2, d1) ∈ adjPairs → False

def canRow (person : String) : Prop := person = "Snow White"

noncomputable def final_trip (remainingDwarfs : List String) (allTrips : List (List String)) : List String := ["Grumpy", "Bashful", "Doc"]

theorem last_trip (adjPairs : List (String × String))
  (h_adj : adjacentPairsQuarrel adjPairs)
  (h_row : canRow "Snow White")
  (dwarfs_order : List String = initial_order)
  (without_quarrels : ∀ trip : List String, trip ∈ allTrips → 
    ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → (d1, d2) ∈ adjPairs → 
    ("Snow White" ∈ trip) → True) :
  final_trip ["Grumpy", "Bashful", "Doc"] allTrips = ["Grumpy", "Bashful", "Doc"] :=
sorry

end last_trip_l441_441317


namespace fourth_derivative_l441_441129

noncomputable def f (x : ℝ) : ℝ := (5 * x - 8) * 2^(-x)

theorem fourth_derivative (x : ℝ) : 
  deriv (deriv (deriv (deriv f))) x = 2^(-x) * (Real.log 2)^4 * (5 * x - 9) :=
sorry

end fourth_derivative_l441_441129


namespace find_z_l441_441955

theorem find_z (z : ℚ) :
  let u := ⟨4, 1, z⟩,
      v := ⟨2, -4, 3⟩,
      proj_u_v := (11 / 29 : ℚ) • v
  in (u ⬝ v) / (v ⬝ v) • v = proj_u_v →
     z = 7 / 3 :=
by
  let u := ⟨4, 1, z⟩,
      v := ⟨2, -4, 3⟩
  have h1 : u ⬝ v = 8 - 4 + 3 * z,
    sorry
  have h2 : v ⬝ v = 29,
    sorry
  simp [h1, h2, proj_u_v]
  intro h
  linarith

end find_z_l441_441955


namespace range_of_g_l441_441523

def g (x : ℝ) : ℝ := ⌊2 * x⌋ - 2 * x

theorem range_of_g : set.range g = set.Icc (-1 : ℝ) 0 :=
by sorry

end range_of_g_l441_441523


namespace snow_white_last_trip_dwarfs_l441_441301

-- Definitions for the conditions
def original_lineup := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]
def only_snow_white_can_row := True
def boat_capacity_snow_white_and_dwarfs := 3
def dwarfs_quarrel_if_adjacent (d1 d2 : String) : Prop :=
  let index_d1 := List.indexOf original_lineup d1
  let index_d2 := List.indexOf original_lineup d2
  abs (index_d1 - index_d2) = 1

-- Theorem to prove the correct answer
theorem snow_white_last_trip_dwarfs :
  let last_trip_dwarfs := ["Grumpy", "Bashful", "Sneezy"]
  ∃ (trip : List String), trip = last_trip_dwarfs ∧ 
  ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → d1 ≠ d2 → ¬dwarfs_quarrel_if_adjacent d1 d2 :=
by
  sorry

end snow_white_last_trip_dwarfs_l441_441301


namespace triangle_is_isosceles_l441_441811

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441811


namespace geometric_series_3000_terms_sum_l441_441961

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_series_3000_terms_sum
    (a r : ℝ)
    (h_r : r ≠ 1)
    (sum_1000 : geometric_sum a r 1000 = 500)
    (sum_2000 : geometric_sum a r 2000 = 950) :
  geometric_sum a r 3000 = 1355 :=
by 
  sorry

end geometric_series_3000_terms_sum_l441_441961


namespace rectangle_length_l441_441949

theorem rectangle_length
  (w l : ℝ)
  (h1 : l = 4 * w)
  (h2 : l * w = 100) :
  l = 20 :=
sorry

end rectangle_length_l441_441949


namespace problem1_problem2_l441_441150

variables (a b : ℝ)

-- Given conditions for the first problem
axiom mag_a : |a| = 4
axiom mag_b : |b| = 3
axiom angle_ab : angle a b = π / 3

-- First problem: Prove (a + 2 * b) • (a - 3 * b) = -44
theorem problem1 :
  (a + 2 * b) • (a - 3 * b) = -44 := sorry

-- Given conditions for the second problem
axiom dot_product_condition : (2 * a - 3 * b) • (2 * a + b) = 61

-- Second problem: Prove angle a b = 2 * π / 3
theorem problem2 :
  angle a b = 2 * π / 3 := sorry

end problem1_problem2_l441_441150


namespace isosceles_triangle_l441_441797

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441797


namespace triangle_is_isosceles_l441_441899

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441899


namespace intersect_sum_l441_441590

noncomputable def p (x : ℝ) : ℝ := sorry
noncomputable def q (x : ℝ) : ℝ := sorry

theorem intersect_sum :
  p 1 = q 1 = 1 →
  p 3 = q 3 = 3 →
  p 5 = q 5 = 5 →
  p 7 = q 7 = 7 →
  let a := 3.5 in
  let b := 7 in
  a + b = 10.5 :=
by
  intros h1 h3 h5 h7
  exact sorry

end intersect_sum_l441_441590


namespace length_of_EF_Ef_theorem_l441_441214

-- Defining the geometric setup
variable (D E F G : Type)
variable (DE DF EF GF EG : ℝ)
variable (isosceles_triangle : DE = 5 ∧ DF = 5 ∧ EG = 4 * GF)

-- Defining the proof problem statement
theorem length_of_EF (h₁ : DE = 5) (h₂ : DF = 5) (h₃ : EG = 4 * GF) : EF = 5 * GF :=
by sorry

-- Defining the final answer
theorem Ef_theorem (h₁ : DE = 5) (h₂ : DF = 5) (h₃ : EG = 4 * GF) : EF = 5 * (5 / 4 * sqrt (10)) := 
by sorry

end length_of_EF_Ef_theorem_l441_441214


namespace snow_white_last_trip_l441_441339

universe u

-- Define the dwarf names as an enumerated type
inductive Dwarf : Type
| Happy | Grumpy | Dopey | Bashful | Sleepy | Doc | Sneezy
deriving DecidableEq, Repr

open Dwarf

-- Define conditions
def initial_lineup : List Dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- Define the condition of adjacency
def adjacent (d1 d2 : Dwarf) : Prop :=
  List.pairwise_adjacent (· = d2) initial_lineup d1

-- Define the boat capacity
def boat_capacity : Fin 4 := by sorry

-- Snow White is the only one who can row
def snow_white_rows : Prop := true

-- No quarrels condition
def no_quarrel_without_snow_white (group : List Dwarf) : Prop :=
  ∀ d1 d2, d1 ∈ group → d2 ∈ group → ¬ adjacent d1 d2

-- Objective: Transfer all dwarfs without quarrels
theorem snow_white_last_trip (trips : List (List Dwarf)) :
  ∃ last_trip : List Dwarf,
    last_trip = [Grumpy, Bashful, Doc] ∧
    no_quarrel_without_snow_white (initial_lineup.diff (trips.join ++ last_trip)) :=
sorry

end snow_white_last_trip_l441_441339


namespace smallest_c_in_range_l441_441137

theorem smallest_c_in_range : 
  ∃ c : ℝ, -3 ∈ {y : ℝ | ∃ x : ℝ, x^2 + 3 * x + c = y} ∧ ∀ c' : ℝ, (-3 ∈ {y : ℝ | ∃ x : ℝ, x^2 + 3 * x + c' = y} → c' ≥ c) → c = -3/4 :=
begin
  sorry
end

end smallest_c_in_range_l441_441137


namespace percentage_increase_correct_l441_441655

-- Defining initial conditions
variables (B H : ℝ) -- bears per week without assistant and hours per week without assistant

-- Defining the rate of making bears per hour without assistant
def rate_without_assistant := B / H

-- Defining the rate with an assistant (100% increase in output per hour)
def rate_with_assistant := 2 * rate_without_assistant

-- Defining the number of hours worked per week with an assistant (10% fewer hours)
def hours_with_assistant := 0.9 * H

-- Calculating the number of bears made per week with an assistant
def bears_with_assistant := rate_with_assistant * hours_with_assistant

-- Calculating the percentage increase in the number of bears made per week when Jane works with an assistant
def percentage_increase : ℝ := ((bears_with_assistant / B) - 1) * 100

-- The theorem to prove
theorem percentage_increase_correct : percentage_increase B H = 80 :=
  by sorry

end percentage_increase_correct_l441_441655


namespace iterative_smile_l441_441109

def smile (a b : ℝ) : ℝ := (a * b) / (a + b)

theorem iterative_smile {a : ℝ} (h : a = 2010) :
  let X := Nat.iterate (smile 2010) 9 2010 in
  X = 201 := by
  sorry

end iterative_smile_l441_441109


namespace isosceles_triangle_of_parallel_l441_441786

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441786


namespace arithmetic_seq_perfect_sixth_power_l441_441474

theorem arithmetic_seq_perfect_sixth_power 
  (a h : ℤ)
  (seq : ∀ n : ℕ, ℤ)
  (h_seq : ∀ n, seq n = a + n * h)
  (h1 : ∃ s₁ x, seq s₁ = x^2)
  (h2 : ∃ s₂ y, seq s₂ = y^3) :
  ∃ k s, seq s = k^6 := 
sorry

end arithmetic_seq_perfect_sixth_power_l441_441474


namespace percentage_of_female_employees_l441_441643

variable (E F M : ℕ)
variable (totalEmployees : E = 1200)
variable (computerLiteratePercent : 62 / 100 * E = 744)
variable (computerLiterateFemales : 504)
variable (computerLiterateMales : 744 - computerLiterateFemales = 240)
variable (maleEmployeesPercent : M = 2 * 240)
variable (femaleEmployees : F = E - M)
variable (percentageFemales : (F / E : ℚ) * 100 = 60)

theorem percentage_of_female_employees
    (totalEmployees : E = 1200)
    (computerLiteratePercent : 62 / 100 * E = 744)
    (computerLiterateFemales : 504)
    (computerLiterateMales : 744 - computerLiterateFemales = 240)
    (maleEmployeesPercent : M = 2 * 240)
    (femaleEmployees : F = E - M)
  : (F / E : ℚ) * 100 = 60 := 
sorry

end percentage_of_female_employees_l441_441643


namespace train_passing_time_l441_441001

noncomputable def time_to_pass (speed_A_kmh speed_B_kmh : ℝ) (length_B : ℝ) (crossing_time_A : ℝ) : ℝ :=
  let speed_A_ms := speed_A_kmh * (1000 / 3600) in
  let speed_B_ms := speed_B_kmh * (1000 / 3600) in
  let length_A := speed_A_ms * crossing_time_A in
  let relative_speed := speed_A_ms + speed_B_ms in
  let total_length := length_A + length_B in
  total_length / relative_speed

theorem train_passing_time (speed_A_kmh speed_B_kmh length_B crossing_time_A : ℝ) (hA : speed_A_kmh = 60) (hB : speed_B_kmh = 80) (hB_len : length_B = 200) (crossA : crossing_time_A = 9) : 
  time_to_pass speed_A_kmh speed_B_kmh length_B crossing_time_A = 9 :=
by
  rw [hA, hB, hB_len, crossA]
  simp only [time_to_pass]
  norm_num
  sorry

end train_passing_time_l441_441001


namespace problem_statement_l441_441744

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441744


namespace number_of_arrangements_l441_441713

-- Definitions based on given conditions
def cards := {1, 2, 3, 4, 5, 6}
def envelopes := {A, B, C}

def valid_arrangement (arrangement : cards → envelopes) : Prop :=
  arrangement 1 = arrangement 2 ∧ (∀ e ∈ envelopes, ∃ c1 c2 ∈ cards, c1 ≠ c2 ∧ arrangement c1 = e ∧ arrangement c2 = e)

-- Theorem statement
theorem number_of_arrangements : 
  (∃! arrangement : cards → envelopes, valid_arrangement arrangement) →
  (count arrangements = 18) :=
sorry

end number_of_arrangements_l441_441713


namespace mam_mgm_bound_l441_441670

-- Let a and b be distinct positive numbers such that a < b
variables (a b : Real) (h_cond : a < b) (h_a_pos : 0 < a) (h_b_pos : 0 < b)

-- Define the modified arithmetic mean (M.A.M.)
def MAM : Real := (a^(1/3) + b^(1/3)) / 2

-- Define the modified geometric mean (M.G.M.)
def MGM : Real := (a * b)^(1/6)

-- Define the difference between MAM and MGM
def Diff : Real := MAM a b - MGM a b

-- Define the maximum bound we need to prove
theorem mam_mgm_bound : Diff a b h_cond h_a_pos h_b_pos < (b - a) / (2 * b) := sorry

end mam_mgm_bound_l441_441670


namespace simplify_expression_is_one_fourth_l441_441289

noncomputable def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
noncomputable def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
noncomputable def simplified_expression : ℝ := (fourth_root 81 - square_root 12.25) ^ 2

theorem simplify_expression_is_one_fourth : simplified_expression = 1 / 4 := 
by
  sorry

end simplify_expression_is_one_fourth_l441_441289


namespace sum_of_reciprocals_of_divisors_eq_two_28_and_496_l441_441234

def sum_of_reciprocals_of_divisors (n : ℕ) : ℚ :=
  ∑ d in (finset.divisors n), 1 / d

theorem sum_of_reciprocals_of_divisors_eq_two_28_and_496 :
  sum_of_reciprocals_of_divisors 28 = 2 ∧ sum_of_reciprocals_of_divisors 496 = 2 :=
by
  sorry

end sum_of_reciprocals_of_divisors_eq_two_28_and_496_l441_441234


namespace jars_needed_l441_441712

variable (days : ℕ) (servings_per_day : ℕ) (servings_per_jar : ℕ)

theorem jars_needed (h1 : days = 30) (h2 : servings_per_day = 2) (h3 : servings_per_jar = 15) :
  (days * servings_per_day) / servings_per_jar = 4 :=
by
  rw [h1, h2, h3]
  exact Nat.div_eq_of_eq_mul_left (by decide) (by decide)
  sorry

end jars_needed_l441_441712


namespace ant_shortest_paths_l441_441086

-- Define the vertices A and B of the cube
def Vertex : Type := ℕ  -- let's assume vertex identifiers are of type ℕ
def A : Vertex := 0  -- vertex A is represented as 0
def B : Vertex := 1  -- vertex B is represented as 1

-- Define the cube and the movement of the ant from A to B along the cube's edges
def isShortestPath (A B : Vertex) (path : list (Vertex × Vertex)) : Prop :=
  -- Path should be non-empty
  path ≠ [] ∧
  -- Path should start at A and end at B
  path.head?.getD (0, 0).fst = A ∧
  path.reverse.head?.getD (0, 0).snd = B ∧
  -- The length of the path should be minimal (3 edges for a cube)
  list.length path = 3 ∧
  -- Each pair in the path represents an edge of the cube
  ∀ (v1 v2 : Vertex), (v1, v2) ∈ path → abs (v1 - v2) = 1

-- Define a function to count the number of shortest paths from A to B
noncomputable def countShortestPaths (A B : Vertex) : ℕ :=
  sorry -- Implementation of counting paths is skipped here

theorem ant_shortest_paths:
  countShortestPaths A B = 6 :=
sorry

end ant_shortest_paths_l441_441086


namespace triangle_is_isosceles_l441_441815

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441815


namespace triangle_ABC_isosceles_l441_441844

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441844


namespace largest_integer_not_sum_of_30_and_composite_l441_441417

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_not_sum_of_30_and_composite (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ b < 30 ∧ is_composite(b) ∧ n = 30 * a + b

theorem largest_integer_not_sum_of_30_and_composite :
  ∃ n : ℕ, is_not_sum_of_30_and_composite(n) ∧ ∀ m : ℕ, is_not_sum_of_30_and_composite(m) → m ≤ n :=
  ⟨93, sorry⟩

end largest_integer_not_sum_of_30_and_composite_l441_441417


namespace find_f_10_l441_441444

variable (f : ℝ → ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, f x = 2 * x^2 + y
def condition2 : Prop := f 2 = 30

-- Theorem to prove
theorem find_f_10 (h1 : condition1 f y) (h2 : condition2 f) : f 10 = 222 := 
sorry

end find_f_10_l441_441444


namespace mary_peter_lucy_chestnuts_l441_441697

noncomputable def mary_picked : ℕ := 12
noncomputable def peter_picked : ℕ := mary_picked / 2
noncomputable def lucy_picked : ℕ := peter_picked + 2
noncomputable def total_picked : ℕ := mary_picked + peter_picked + lucy_picked

theorem mary_peter_lucy_chestnuts : total_picked = 26 := by
  sorry

end mary_peter_lucy_chestnuts_l441_441697


namespace max_area_rectangle_perimeter_156_l441_441662

theorem max_area_rectangle_perimeter_156 (x y : ℕ) 
  (h : 2 * (x + y) = 156) : ∃x y, x * y = 1521 :=
by
  sorry

end max_area_rectangle_perimeter_156_l441_441662


namespace triangle_is_isosceles_if_parallel_l441_441728

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441728


namespace isosceles_triangle_l441_441759

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441759


namespace divisible_by_five_l441_441002

theorem divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
sorry

end divisible_by_five_l441_441002


namespace carolyn_sum_of_removed_numbers_l441_441505

theorem carolyn_sum_of_removed_numbers (n : ℕ) (h_n: n = 8) (h_first: true):
  (sum_of_values_removed_by_carolyn 8 [4]) = 12 := 
by sorry

end carolyn_sum_of_removed_numbers_l441_441505


namespace problem_solution_l441_441514

-- Define the sequence a_n
noncomputable def a : ℕ → ℕ
| 1     => 3
| (n+1) => a n ^ 2 - 2 * n * a n + 2

-- Define the sum of the first n terms of the sequence
noncomputable def S (n : ℕ) : ℕ :=
  ∑ i in Finset.range 1 n.succ, a i

-- Define the conjectured formula for the sequence
def a_conjecture (n : ℕ) : ℕ := 2 * n + 1

-- State the theorem to prove
theorem problem_solution :
  (a 2 = 5) ∧ (a 3 = 7) ∧ (a 4 = 9) ∧
  (∀ n, a n = a_conjecture n) ∧
  (∀ n ≥ 6, S n < 2^n) :=
sorry

end problem_solution_l441_441514


namespace isosceles_triangle_of_parallel_l441_441780

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441780


namespace tangent_parallel_to_line_l441_441937

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_to_line
  (P0_coord : set (ℝ × ℝ)) :
  (∃ P0 ∈ P0_coord,
    ∃ m : ℝ, (∀ x ∈ P0_coord, (derivative f x) = m) ∧ m = 4) →
  P0_coord = {(1, 0), (-1, -4)} :=
by
  sorry

end tangent_parallel_to_line_l441_441937


namespace cube_surface_area_l441_441383

theorem cube_surface_area (side_perimeter : ℤ) (h : side_perimeter = 52) : 
  let s := side_perimeter / 4 in
  let area_one_face := s * s in
  6 * area_one_face = 1014 :=
by
  -- declaration of the variables based on the conditions
  let s := side_perimeter / 4
  let area_one_face := s * s
  -- assertion of the hypothesis
  have hs : s = 13 := by rw [h]; exact rfl
  have ha : area_one_face = 169 := by rw [hs]; exact rfl
  -- conclusion
  show 6 * area_one_face = 1014 from by rw [ha]; exact rfl

end cube_surface_area_l441_441383


namespace solve_exponential_equation_l441_441368

theorem solve_exponential_equation : ∀ (x : ℝ), 4^x - 2^(x + 1) = 0 ↔ x = 1 :=
by
  intros x
  sorry

end solve_exponential_equation_l441_441368


namespace range_of_a_l441_441170

theorem range_of_a 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : r > 0) 
  (cos_le_zero : (3 * a - 9) / r ≤ 0) 
  (sin_gt_zero : (a + 2) / r > 0) : 
  -2 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l441_441170


namespace snow_white_last_trip_l441_441338

universe u

-- Define the dwarf names as an enumerated type
inductive Dwarf : Type
| Happy | Grumpy | Dopey | Bashful | Sleepy | Doc | Sneezy
deriving DecidableEq, Repr

open Dwarf

-- Define conditions
def initial_lineup : List Dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- Define the condition of adjacency
def adjacent (d1 d2 : Dwarf) : Prop :=
  List.pairwise_adjacent (· = d2) initial_lineup d1

-- Define the boat capacity
def boat_capacity : Fin 4 := by sorry

-- Snow White is the only one who can row
def snow_white_rows : Prop := true

-- No quarrels condition
def no_quarrel_without_snow_white (group : List Dwarf) : Prop :=
  ∀ d1 d2, d1 ∈ group → d2 ∈ group → ¬ adjacent d1 d2

-- Objective: Transfer all dwarfs without quarrels
theorem snow_white_last_trip (trips : List (List Dwarf)) :
  ∃ last_trip : List Dwarf,
    last_trip = [Grumpy, Bashful, Doc] ∧
    no_quarrel_without_snow_white (initial_lineup.diff (trips.join ++ last_trip)) :=
sorry

end snow_white_last_trip_l441_441338


namespace maria_needs_flour_l441_441056

-- Definitions from conditions
def cups_of_flour_per_cookie (c : ℕ) (f : ℚ) : ℚ := f / c

def total_cups_of_flour (cps_per_cookie : ℚ) (num_cookies : ℕ) : ℚ := cps_per_cookie * num_cookies

-- Given values
def cookies_20 := 20
def flour_3 := 3
def cookies_100 := 100

theorem maria_needs_flour :
  total_cups_of_flour (cups_of_flour_per_cookie cookies_20 flour_3) cookies_100 = 15 :=
by
  sorry -- Proof is omitted

end maria_needs_flour_l441_441056


namespace fraction_of_phone_numbers_l441_441495

theorem fraction_of_phone_numbers (a b : ℕ) 
  (valid_phone_numbers_count: b = 7 * 10^6)
  (valid_phone_numbers_with_even_end_count: a = 3.5 * 10^6):
  a / b = 1 / 2 := by
  sorry

end fraction_of_phone_numbers_l441_441495


namespace final_trip_l441_441328

-- Definitions for the conditions
def dwarf := String
def snow_white := "Snow White" : dwarf
def Happy : dwarf := "Happy"
def Grumpy : dwarf := "Grumpy"
def Dopey : dwarf := "Dopey"
def Bashful : dwarf := "Bashful"
def Sleepy : dwarf := "Sleepy"
def Doc : dwarf := "Doc"
def Sneezy : dwarf := "Sneezy"

-- The dwarfs lineup from left to right
def lineup : List dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- The boat can hold Snow White and up to 3 dwarfs
def boat_capacity (load : List dwarf) : Prop := snow_white ∈ load ∧ load.length ≤ 4

-- Any two dwarfs standing next to each other in the original lineup will quarrel if left without Snow White
def will_quarrel (d1 d2 : dwarf) : Prop :=
  (d1, d2) ∈ (lineup.zip lineup.tail)

-- The objective: Prove that on the last trip, Snow White will take Grumpy, Bashful, and Doc
theorem final_trip : ∃ load : List dwarf, 
  set.load ⊆ {snow_white, Grumpy, Bashful, Doc} ∧ boat_capacity load :=
  sorry

end final_trip_l441_441328


namespace two_digit_numbers_count_tens_digit_less_than_units_digit_divisible_by_3_two_digit_numbers_l441_441264

def setA : set ℕ := {2, 4, 6, 8}
def setB : set ℕ := {1, 3, 5, 7, 9}

theorem two_digit_numbers_count : 
  (setA.card * setB.card = 20) := 
by sorry

theorem tens_digit_less_than_units_digit : 
  (setA.product setB).count (λ p, p.1 < p.2) = 10 := 
by sorry

theorem divisible_by_3_two_digit_numbers : 
  (setA.product setB).count (λ p, (p.1 + p.2) % 3 = 0) = 7 := 
by sorry

end two_digit_numbers_count_tens_digit_less_than_units_digit_divisible_by_3_two_digit_numbers_l441_441264


namespace periodic_function_with_period_sqrt2_l441_441682

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Definition of symmetry about x = sqrt(2)/2
def is_symmetric_about_line (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c - x) = f (c + x)

-- Main theorem to prove
theorem periodic_function_with_period_sqrt2 (f : ℝ → ℝ) :
  is_even_function f → is_symmetric_about_line f (Real.sqrt 2 / 2) → ∃ T, T = Real.sqrt 2 ∧ ∀ x, f (x + T) = f x :=
by
  sorry

end periodic_function_with_period_sqrt2_l441_441682


namespace emily_disproved_jacob_by_turnover_5_and_7_l441_441926

def is_vowel (c : Char) : Prop :=
  c = 'A'

def is_consonant (c : Char) : Prop :=
  ¬ is_vowel c

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

def card_A_is_vowel : Prop := is_vowel 'A'
def card_1_is_odd : Prop := ¬ is_even 1 ∧ ¬ is_prime 1
def card_8_is_even : Prop := is_even 8 ∧ ¬ is_prime 8
def card_R_is_consonant : Prop := is_consonant 'R'
def card_S_is_consonant : Prop := is_consonant 'S'
def card_5_conditions : Prop := ¬ is_even 5 ∧ is_prime 5
def card_7_conditions : Prop := ¬ is_even 7 ∧ is_prime 7

theorem emily_disproved_jacob_by_turnover_5_and_7 :
  card_5_conditions ∧ card_7_conditions →
  (∃ (c : Char), (is_prime 5 ∧ is_consonant c)) ∨
  (∃ (c : Char), (is_prime 7 ∧ is_consonant c)) :=
by sorry

end emily_disproved_jacob_by_turnover_5_and_7_l441_441926


namespace remainder_is_zero_l441_441005

def remainder_when_multiplied_then_subtracted (a b : ℕ) : ℕ :=
  (a * b - 8) % 8

theorem remainder_is_zero : remainder_when_multiplied_then_subtracted 104 106 = 0 := by
  sorry

end remainder_is_zero_l441_441005


namespace area_AKM_less_than_area_ABC_l441_441220

-- Define the rectangle ABCD
structure Rectangle :=
(A B C D : ℝ) -- Four vertices of the rectangle
(side_AB : ℝ) (side_BC : ℝ) (side_CD : ℝ) (side_DA : ℝ)

-- Define the arbitrary points K and M on sides BC and CD respectively
variables (B C D K M : ℝ)

-- Define the area of triangle function and area of rectangle function
def area_triangle (A B C : ℝ) : ℝ := sorry -- Assuming a function calculating area of triangle given 3 vertices
def area_rectangle (A B C D : ℝ) : ℝ := sorry -- Assuming a function calculating area of rectangle given 4 vertices

-- Assuming the conditions given in the problem statement
variables (A : ℝ) (rect : Rectangle)

-- Prove that the area of triangle AKM is less than the area of triangle ABC
theorem area_AKM_less_than_area_ABC : 
  ∀ (K M : ℝ), K ∈ [B,C] → M ∈ [C,D] →
    area_triangle A K M < area_triangle A B C := sorry

end area_AKM_less_than_area_ABC_l441_441220


namespace kolya_win_l441_441666

theorem kolya_win : ∀ stones : ℕ, stones = 100 → (∃ strategy : (ℕ → ℕ × ℕ), ∀ opponent_strategy : (ℕ → ℕ × ℕ), true → true) :=
by
  sorry

end kolya_win_l441_441666


namespace total_cost_of_pens_and_notebooks_l441_441054

theorem total_cost_of_pens_and_notebooks (a b : ℝ) : 5 * a + 8 * b = 5 * a + 8 * b := 
by 
  sorry

end total_cost_of_pens_and_notebooks_l441_441054


namespace min_rubles_to_reverse_order_l441_441453

-- We use noncomputable because dealing with combinatorial problems like this 
-- often involves nonconstructive proofs.
noncomputable def reverse_chips_min_cost (n : ℕ) : ℕ :=
  if h : n = 100 then 50 else 0

-- Statement to prove
theorem min_rubles_to_reverse_order (n : ℕ) (h : n = 100) :
  reverse_chips_min_cost n = 50 :=
by
  rw reverse_chips_min_cost
  simp [h]
  sorry

end min_rubles_to_reverse_order_l441_441453


namespace complex_exponent_approx_l441_441554

noncomputable def e_to_pi_div_i (k : ℤ) : ℂ := complex.exp ((real.pi + 2 * k * real.pi) / complex.I)

theorem complex_exponent_approx (k : ℤ) : e_to_pi_div_i k ≈ real.sqrt (23^ (1 / 7)) :=
by
  set LHS := e_to_pi_div_i 0 with hLHS
  set RHS := real.sqrt (23^(1/7)) with hRHS
  -- add relevant assumptions used in the problem
  have h0 : complex.exp (real.pi / complex.I) = LHS,
  {
    apply congr_arg,
    simp only [mul_zero, add_zero],
  },
  sorry

end complex_exponent_approx_l441_441554


namespace suitable_temperature_range_l441_441958

theorem suitable_temperature_range 
  (a_min a_max b_min b_max : ℝ) 
  (hA : a_min = 1 ∧ a_max = 5)
  (hB : b_min = 3 ∧ b_max = 8) : 
  (max a_min b_min) ≤ (min a_max b_max) :=
by
  intro hA hB
  -- The suitable temperature range overlap for A and B
  have hA := hA.1, hA' := hA.2, hB := hB.1, hB' := hB.2
  have h1 : max 1 3 = 3 := rfl
  have h2 : min 5 8 = 5 := rfl
  rw [h1, h2]
  sorry  

end suitable_temperature_range_l441_441958


namespace isosceles_triangle_of_parallel_l441_441876

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441876


namespace klinker_age_problem_l441_441267

theorem klinker_age_problem : ∃ (x : ℕ), (47 + x = 3 * (13 + x)) ∧ x = 4 :=
by
  use 4
  split
  · calc 47 + 4 = 51 : by rfl
          3 * (13 + 4) = 3 * 17 : by rfl
                     = 51 : by rfl
  · rfl
  -- sorry

end klinker_age_problem_l441_441267


namespace methane_balancing_and_requirement_l441_441104

theorem methane_balancing_and_requirement :
  ∃ (a b c d e f g h : ℕ) (total_methane : ℕ),
    (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 2) ∧
    (e = 1 ∧ f = 2 ∧ g = 1 ∧ h = 2) ∧
    (total_methane = 2) :=
begin
  use [1, 2, 1, 2, 1, 2, 1, 2, 2],
  simp,
end

end methane_balancing_and_requirement_l441_441104


namespace triangle_is_isosceles_l441_441856

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441856


namespace isosceles_triangle_l441_441750

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441750


namespace distinct_numbers_in_S_l441_441242

def first_sequence (k : ℕ) : ℕ := 3 * k + 1
def second_sequence (l : ℕ) : ℕ := 7 * l + 9

def set_A : finset ℕ := (finset.range 2004).image first_sequence
def set_B : finset ℕ := (finset.range 2004).image second_sequence

def set_S : finset ℕ := set_A ∪ set_B

theorem distinct_numbers_in_S : set_S.card = 3722 := 
sorry

end distinct_numbers_in_S_l441_441242


namespace f_800_l441_441247

noncomputable def f : ℕ → ℕ := sorry

axiom axiom1 : ∀ x y : ℕ, 0 < x → 0 < y → f (x * y) = f x + f y
axiom axiom2 : f 10 = 10
axiom axiom3 : f 40 = 14

theorem f_800 : f 800 = 26 :=
by
  -- Apply the conditions here
  sorry

end f_800_l441_441247


namespace chess_mixed_games_l441_441019

theorem chess_mixed_games :
  ∃ W M : ℕ, nat.choose W 2 = 45 ∧ nat.choose M 2 = 190 ∧ W * M = 200 :=
by
  sorry

end chess_mixed_games_l441_441019


namespace aram_fraction_of_fine_l441_441494

theorem aram_fraction_of_fine (F : ℝ) (H1 : Joe_paid = (1/4)*F + 3)
  (H2 : Peter_paid = (1/3)*F - 3)
  (H3 : Aram_paid = (1/2)*F - 4)
  (H4 : Joe_paid + Peter_paid + Aram_paid = F) : 
  Aram_paid / F = 5 / 12 := 
sorry

end aram_fraction_of_fine_l441_441494


namespace find_initial_population_l441_441364

-- Definition of the initial population given the final population, rate of increase, and number of years.
def initial_population (P_f : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P_f / (1 + r)^n

-- Main theorem stating the initial population given specific values.
theorem find_initial_population :
  initial_population 262500 0.05 10 ≈ 161182 :=
by
  unfold initial_population
  have h1 : (1 + 0.05) ^ 10 = 1.628894626777442 := by sorry
  have h2 : 262500 / 1.628894626777442 ≈ 161182 := by sorry
  exact h2

end find_initial_population_l441_441364


namespace restaurant_made_correct_amount_l441_441057

noncomputable def restaurant_revenue : ℝ := 
  let price1 := 8
  let qty1 := 10
  let price2 := 10
  let qty2 := 5
  let price3 := 4
  let qty3 := 20
  let total_sales := qty1 * price1 + qty2 * price2 + qty3 * price3
  let discount := 0.10
  let discounted_total := total_sales * (1 - discount)
  let sales_tax := 0.05
  let final_amount := discounted_total * (1 + sales_tax)
  final_amount

theorem restaurant_made_correct_amount : restaurant_revenue = 198.45 := by
  sorry

end restaurant_made_correct_amount_l441_441057


namespace casey_stays_for_n_months_l441_441506

-- Definitions based on conditions.
def weekly_cost : ℕ := 280
def monthly_cost : ℕ := 1000
def weeks_per_month : ℕ := 4
def total_savings : ℕ := 360

-- Calculate monthly cost when paying weekly.
def monthly_cost_weekly := weekly_cost * weeks_per_month

-- Calculate savings per month when paying monthly instead of weekly.
def savings_per_month := monthly_cost_weekly - monthly_cost

-- Define the problem statement.
theorem casey_stays_for_n_months :
  (total_savings / savings_per_month) = 3 := by
  -- Proof is omitted.
  sorry

end casey_stays_for_n_months_l441_441506


namespace correct_functions_are_even_and_decreasing_l441_441009

open Real

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃a b⦄, a ∈ s → b ∈ s → a < b → f b ≤ f a

def f_A (x : ℝ) : ℝ := x^(4/5)
def f_C (x : ℝ) : ℝ := log (x^2 + 1)

theorem correct_functions_are_even_and_decreasing :
  is_even f_A ∧ is_decreasing_on f_A {x | x < 0} ∧
  is_even f_C ∧ is_decreasing_on f_C {x | x < 0} :=
by
  sorry

end correct_functions_are_even_and_decreasing_l441_441009


namespace max_fraction_value_l441_441671

theorem max_fraction_value (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_bc_le_a : b + c ≤ a) :
  ∃ x : ℝ, ∀ b c, x = max (bc / (a^2 + 2ab + b^2)) (1 / 8) := 
sorry

end max_fraction_value_l441_441671


namespace largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441412

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441412


namespace area_of_region_l441_441128

noncomputable def region_area : ℝ :=
  sorry

theorem area_of_region :
  region_area = sorry := 
sorry

end area_of_region_l441_441128


namespace volume_of_rectangular_prism_l441_441372

theorem volume_of_rectangular_prism (l w h : ℕ) (x : ℕ) 
  (h_ratio : l = 3 * x ∧ w = 2 * x ∧ h = x)
  (h_edges : 4 * l + 4 * w + 4 * h = 72) : 
  l * w * h = 162 := 
by
  sorry

end volume_of_rectangular_prism_l441_441372


namespace paving_cost_l441_441023

theorem paving_cost (l w r : ℝ) (h_l : l = 5.5) (h_w : w = 4) (h_r : r = 700) :
  l * w * r = 15400 :=
by sorry

end paving_cost_l441_441023


namespace sequence_difference_l441_441230

theorem sequence_difference
  (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a (n + 1) - a n - n = 0) :
  a 2017 - a 2016 = 2016 :=
by
  sorry

end sequence_difference_l441_441230


namespace isosceles_triangle_l441_441827

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441827


namespace count_x_squared_plus_4x_plus_4_between_30_and_60_l441_441547

theorem count_x_squared_plus_4x_plus_4_between_30_and_60 : 
  ∃ k, k = finset.card ({ x : ℕ | 30 < x^2 + 4 * x + 4 ∧ x^2 + 4 * x + 4 < 60 | }.to_finset) ∧ k = 2 := 
by
  sorry

end count_x_squared_plus_4x_plus_4_between_30_and_60_l441_441547


namespace avg_highway_mpg_l441_441079

noncomputable def highway_mpg (total_distance : ℕ) (fuel : ℕ) : ℝ :=
  total_distance / fuel
  
theorem avg_highway_mpg :
  highway_mpg 305 25 = 12.2 :=
by
  sorry

end avg_highway_mpg_l441_441079


namespace isosceles_triangle_l441_441832

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441832


namespace last_two_digits_of_sum_l441_441542

-- Definitions
def factorial (n : ℕ) : ℕ := (List.range (n + 1)).prod

def last_two_digits (n : ℕ) : ℕ := n % 100

-- Main Statement
theorem last_two_digits_of_sum : 
  last_two_digits (∑ k in Finset.range 16, factorial (6 * (k + 1))) = 20 :=
by
  -- Each factorial term
  have h₁ : factorial 6 = 720 := by decide
  have h₂ : ∀ n, 10 ≤ n → last_two_digits (factorial n) = 0 := 
    by sorry           -- This would be proved by considering the factors of 5

  -- Sum consists of terms 6!, 12!, 18!, ..., 96!
  have h₃ : ∑ k in Finset.range 16, last_two_digits (factorial (6 * (k + 1)))
             = last_two_digits (factorial 6) := 
    by sorry           -- Only 6! has non-zero last two digits, the rest are zero

  -- Applying our known values
  show last_two_digits 720 = 20 from by decide

end last_two_digits_of_sum_l441_441542


namespace trains_pass_bridge_time_l441_441975

theorem trains_pass_bridge_time:
  let length_train_A := 360   -- Length of Train A in meters
  let speed_train_A := 50 / 3.6  -- Speed of Train A in m/s
  let length_train_B := 480   -- Length of Train B in meters
  let speed_train_B := 60 / 3.6  -- Speed of Train B in m/s
  let length_bridge := 140    -- Length of the bridge in meters
  let total_distance := length_train_A + length_train_B + length_bridge
  let relative_speed := speed_train_A + speed_train_B
  let time := total_distance / relative_speed
  time ≈ 32.06 :=
by
  sorry

end trains_pass_bridge_time_l441_441975


namespace area_of_Phi_l441_441152

variable (M : Type) [ConvexPolygon M]
variable (S : ℝ) (P : ℝ) (x : ℝ) (Hx : x > 0)

def area_set_Phi : ℝ := S + P * x + Real.pi * x ^ 2

theorem area_of_Phi (S : ℝ) (P : ℝ) (x : ℝ) (Hx : x > 0) :
  area_set_Phi M S P x = S + P * x + Real.pi * x ^ 2 :=
sorry

end area_of_Phi_l441_441152


namespace sequence_100_eq_1_over_50_l441_441367

/-- Define our sequence a_n with given conditions and recursive relation -/
def sequence (n : ℕ) : ℝ :=
  if n = 1 then 2 else
  if n = 2 then 1 else
  sorry -- For n >= 3, define using the given recursive relation


/-- Prove that the 100th term is 1/50 -/
theorem sequence_100_eq_1_over_50 : sequence 100 = 1 / 50 :=
  sorry

end sequence_100_eq_1_over_50_l441_441367


namespace cost_of_fencing_is_correct_l441_441022

noncomputable def pi : ℝ := 3.1416 -- Define the approximation of pi
noncomputable def area_hectares : ℝ := 17.56 -- Area in hectares
noncomputable def area_m² : ℝ := area_hectares * 10000 -- Convert area to square meters
noncomputable def cost_per_meter : ℝ := 7.0 -- Cost per meter

-- Calculate the radius
noncomputable def radius : ℝ := Real.sqrt (area_m² / pi)

-- Calculate the circumference
noncomputable def circumference : ℝ := 2 * pi * radius

-- Compute the total cost of fencing
noncomputable def total_cost : ℝ := circumference * cost_per_meter

theorem cost_of_fencing_is_correct :
  abs (total_cost - 10393.6) < 1 :=
by
  sorry

end cost_of_fencing_is_correct_l441_441022


namespace snow_white_last_trip_l441_441336

universe u

-- Define the dwarf names as an enumerated type
inductive Dwarf : Type
| Happy | Grumpy | Dopey | Bashful | Sleepy | Doc | Sneezy
deriving DecidableEq, Repr

open Dwarf

-- Define conditions
def initial_lineup : List Dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- Define the condition of adjacency
def adjacent (d1 d2 : Dwarf) : Prop :=
  List.pairwise_adjacent (· = d2) initial_lineup d1

-- Define the boat capacity
def boat_capacity : Fin 4 := by sorry

-- Snow White is the only one who can row
def snow_white_rows : Prop := true

-- No quarrels condition
def no_quarrel_without_snow_white (group : List Dwarf) : Prop :=
  ∀ d1 d2, d1 ∈ group → d2 ∈ group → ¬ adjacent d1 d2

-- Objective: Transfer all dwarfs without quarrels
theorem snow_white_last_trip (trips : List (List Dwarf)) :
  ∃ last_trip : List Dwarf,
    last_trip = [Grumpy, Bashful, Doc] ∧
    no_quarrel_without_snow_white (initial_lineup.diff (trips.join ++ last_trip)) :=
sorry

end snow_white_last_trip_l441_441336


namespace snow_white_last_trip_l441_441308

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l441_441308


namespace correctness_statements_l441_441438

-- Definitions for the mathematical conditions
def statementA (x : ℝ) := x + 4 / (x - 1)
def statementB (x : ℝ) := x * real.sqrt (1 - x^2)
def statementD (x y : ℝ) := (x > 0 ∧ y > 0 ∧ x + 2 * y = 1) →
  2 / x + 1 / y

-- Statements for correctness
theorem correctness_statements :
  ¬(∀ x : ℝ, statementA x ≥ 5) ∧ -- Statement A is false
  (∀ x : ℝ, statementB x ≤ 1 / 2) ∧ (∃ x : ℝ, statementB x = 1 / 2) ∧ -- Statement B is true
  (∀ (x y c : ℝ), (c > 0 ∧ x > y) ↔ (x / c^2 > y / c^2)) ∧ -- Statement C is true
  ((∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2 * y = 1) → ∀ (x y : ℝ), statementD x y ≥ 8 ∧ (statementD (1 / 2) (1 / 4) = 8)) -- Statement D is true
:= by
  sorry

end correctness_statements_l441_441438


namespace largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441414

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441414


namespace triangle_is_isosceles_l441_441805

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441805


namespace find_m_over_n_l441_441183

variable (a b : ℝ × ℝ)
variable (m n : ℝ)
variable (n_nonzero : n ≠ 0)

axiom a_def : a = (1, 2)
axiom b_def : b = (-2, 3)
axiom collinear : ∃ k : ℝ, m • a - n • b = k • (a + 2 • b)

theorem find_m_over_n : m / n = -1 / 2 := by
  sorry

end find_m_over_n_l441_441183


namespace odd_cross_exists_l441_441640

-- Define the problem statement in Lean
theorem odd_cross_exists (m n : ℕ) (hm : m % 2 = 0) (hn : n % 2 = 0) 
  (grid : Fin m × Fin n → Bool) (h_black_cells : ∃ i j, grid (i, j) = true) :
  ∃ i j, (Finset.univ.filter (λ k, grid(i, k) = true)).card % 2 = 1 ∧ 
         (Finset.univ.filter (λ k, grid(k, j) = true)).card % 2 = 1 :=
by
  sorry

end odd_cross_exists_l441_441640


namespace evaluate_nested_function_l441_441688

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1 / 3) ^ x else Real.log x / Real.log 3

theorem evaluate_nested_function :
  f (f (1 / 9)) = 9 :=
by
  sorry

end evaluate_nested_function_l441_441688


namespace remaining_yards_after_marathons_l441_441053

noncomputable def marathon_distance_miles : ℕ := 26
noncomputable def marathon_distance_yards : ℕ := 395
noncomputable def conversion_factor : ℕ := 1760
noncomputable def num_marathons : ℕ := 15

theorem remaining_yards_after_marathons : 
  let total_yards := num_marathons * (marathon_distance_miles * conversion_factor + marathon_distance_yards) in
  let miles := total_yards / conversion_factor in
  let yards := total_yards % conversion_factor in
  yards = 645 := 
by
  sorry

end remaining_yards_after_marathons_l441_441053


namespace triangle_is_isosceles_l441_441892

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441892


namespace joe_lists_count_l441_441459

def num_options (n : ℕ) (k : ℕ) : ℕ := n ^ k

theorem joe_lists_count : num_options 12 3 = 1728 := by
  unfold num_options
  sorry

end joe_lists_count_l441_441459


namespace distance_is_sqrt_41_l441_441130

noncomputable def distance_between_points : ℝ :=
  let (x1, y1) := (2, 5)
  let (x2, y2) := (7, 1)
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_is_sqrt_41 : distance_between_points = real.sqrt 41 := by
  sorry

end distance_is_sqrt_41_l441_441130


namespace asymptote_intersection_l441_441117

theorem asymptote_intersection 
  (H1 : ∃ x, (x = 3 → (x^2 - 6*x + 9) = 0) ∧ ¬(x = 3 → x^2 - 6*x + 9 = 0)) 
  (H2 : ∀ x, x ≠ 3 → y = 1 - (1 / (x^2 - 6*x + 9))) 
  (H3 : ∀ x ε, ε > 0 → (∃ δ, δ > 0 ∧ (|x| > δ → |(1 / (x^2 - 6*x + 9)) - 0| < ε))) :
  ∃ p : ℝ × ℝ, p = (3, 1) := 
sorry

end asymptote_intersection_l441_441117


namespace total_silver_dollars_l441_441700

-- Definitions based on conditions
def chiu_silver_dollars : ℕ := 56
def phung_silver_dollars : ℕ := chiu_silver_dollars + 16
def ha_silver_dollars : ℕ := phung_silver_dollars + 5

-- Theorem statement
theorem total_silver_dollars : chiu_silver_dollars + phung_silver_dollars + ha_silver_dollars = 205 :=
by
  -- We use "sorry" to fill in the proof part as instructed
  sorry

end total_silver_dollars_l441_441700


namespace snow_white_last_trip_l441_441307

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l441_441307


namespace final_trip_l441_441332

-- Definitions for the conditions
def dwarf := String
def snow_white := "Snow White" : dwarf
def Happy : dwarf := "Happy"
def Grumpy : dwarf := "Grumpy"
def Dopey : dwarf := "Dopey"
def Bashful : dwarf := "Bashful"
def Sleepy : dwarf := "Sleepy"
def Doc : dwarf := "Doc"
def Sneezy : dwarf := "Sneezy"

-- The dwarfs lineup from left to right
def lineup : List dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- The boat can hold Snow White and up to 3 dwarfs
def boat_capacity (load : List dwarf) : Prop := snow_white ∈ load ∧ load.length ≤ 4

-- Any two dwarfs standing next to each other in the original lineup will quarrel if left without Snow White
def will_quarrel (d1 d2 : dwarf) : Prop :=
  (d1, d2) ∈ (lineup.zip lineup.tail)

-- The objective: Prove that on the last trip, Snow White will take Grumpy, Bashful, and Doc
theorem final_trip : ∃ load : List dwarf, 
  set.load ⊆ {snow_white, Grumpy, Bashful, Doc} ∧ boat_capacity load :=
  sorry

end final_trip_l441_441332


namespace triangle_ABC_is_isosceles_l441_441913

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441913


namespace snow_white_last_trip_l441_441323

-- Definitions based on the problem's conditions
inductive Dwarf
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

def is_adjacent (d1 d2 : Dwarf) : Prop :=
  (d1, d2) ∈ [
    (Dwarf.Happy, Dwarf.Grumpy),
    (Dwarf.Grumpy, Dwarf.Happy),
    (Dwarf.Grumpy, Dwarf.Dopey),
    (Dwarf.Dopey, Dwarf.Grumpy),
    (Dwarf.Dopey, Dwarf.Bashful),
    (Dwarf.Bashful, Dwarf.Dopey),
    (Dwarf.Bashful, Dwarf.Sleepy),
    (Dwarf.Sleepy, Dwarf.Bashful),
    (Dwarf.Sleepy, Dwarf.Doc),
    (Dwarf.Doc, Dwarf.Sleepy),
    (Dwarf.Doc, Dwarf.Sneezy),
    (Dwarf.Sneezy, Dwarf.Doc)
  ]

def boat_capacity : ℕ := 3

variable (snowWhite : Prop)

-- The theorem to prove that the dwarfs Snow White will take in the last trip are Grumpy, Bashful and Doc
theorem snow_white_last_trip 
  (h1 : snowWhite)
  (h2 : boat_capacity = 3)
  (h3 : ∀ d1 d2, is_adjacent d1 d2 → snowWhite)
  : (snowWhite ∧ (Dwarf.Grumpy ∧ Dwarf.Bashful ∧ Dwarf.Doc)) :=
sorry

end snow_white_last_trip_l441_441323


namespace Olga_paints_no_boards_in_1point5_hours_l441_441706

-- Definitions based on the given conditions
def Valera_time_to_paint_boards := 2
def Valera_boards_painted := 11
def combined_time_painting := 3
def combined_boards_painted := 8
def Olga_return_time := 1.5

-- Calculate painting speeds
def Valera_painting_speed : ℝ := Valera_boards_painted / (Valera_time_to_paint_boards - 1)
def combined_painting_speed : ℝ := combined_boards_painted / (combined_time_painting - 1)
def Olga_painting_speed : ℝ := combined_painting_speed - Valera_painting_speed

-- Proof to show that the number of boards Olga will paint is 0 given her return time constraint
theorem Olga_paints_no_boards_in_1point5_hours :
  Olga_painting_speed * (Olga_return_time - 1) = 0 :=
by
  -- Since Valera_painting_speed = 11 / 1 = 11, combined_painting_speed = 8 / 2 = 4
  -- Olga's painting speed is therefore 4 - 11 = -7 (which is not realistic for the problem)
  -- This invalidity leads us to inherently conclude Olga can't paint any boards if she must return in 1.5 hours
  have h : Olga_painting_speed = 0, proof impossible,
  show 0 * 0.5 = 0, proof unfold multiplication sorry

end Olga_paints_no_boards_in_1point5_hours_l441_441706


namespace initial_amount_l441_441233

theorem initial_amount (P : ℝ) :
  (P * 1.0816 - P * 1.08 = 3.0000000000002274) → P = 1875.0000000001421 :=
by
  sorry

end initial_amount_l441_441233


namespace gcd_372_684_is_12_l441_441359

theorem gcd_372_684_is_12 :
  Nat.gcd 372 684 = 12 :=
sorry

end gcd_372_684_is_12_l441_441359


namespace sector_area_maximization_sector_maximum_area_u8_l441_441069

-- Definition of the sector's area maximization problem with given conditions
theorem sector_area_maximization (u : ℝ) (r : ℝ) (i : ℝ) (t : ℝ) 
(h1 : u = i + 2 * r) 
(h2 : t = (i * r) / 2) : 
r = u / 4 ∧ t = u^2 / 16 := 
sorry

-- Specific case when u = 8
theorem sector_maximum_area_u8 : 
let u := 8 in 
let r := u / 4 in 
let t := u^2 / 16 in 
t = 4 := 
sorry

end sector_area_maximization_sector_maximum_area_u8_l441_441069


namespace isosceles_triangle_l441_441824

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441824


namespace problem_part1_problem_part2_l441_441355

noncomputable def f (x : ℝ) : ℝ := 5 * sin (2 * x + (π / 6))

theorem problem_part1 : 
  ∀ k : ℤ, ∃ a b : ℝ, (a = k * π + π / 6) ∧ (b = k * π + 2 * π / 3) ∧ 
    (∀ x : ℝ, a ≤ x ∧ x ≤ b → deriv f x < 0) :=
sorry

noncomputable def g (x m : ℝ) : ℝ := 5 * sin (2 * x - 2 * m + (π / 6))

theorem problem_part2 : 
  ∃ m : ℝ, m = π / 3 ∧ ∀ x : ℝ, g (x, m) = g (-x, m) :=
sorry

end problem_part1_problem_part2_l441_441355


namespace coeff_x3_in_expansion_l441_441112

-- Define the binomial coefficient and expansion term
noncomputable def binomial_coeff (n k : ℕ) : ℕ := nat.choose n k
noncomputable def binomial_term (r : ℕ) (x : ℝ) : ℝ := (-3)^r * binomial_coeff 7 r * x^((7 - r) / 2)

-- Define the coefficient of x^3
def coefficient_of_x3 (x : ℝ) : ℝ :=
  let r := 1 in
  (-3)^r * binomial_coeff 7 r

-- Assert that the coefficient of x^3 in the expansion is -21
theorem coeff_x3_in_expansion : coefficient_of_x3 x = -21 := by
  sorry

end coeff_x3_in_expansion_l441_441112


namespace star_example_l441_441142

def star (a b : ℝ) : ℝ :=
  (Real.sqrt (a + b)) / (a - b)

theorem star_example : star 4 8 = - (Real.sqrt 3) / 2 :=
by
  sorry

end star_example_l441_441142


namespace solution_set_inequality_five_elements_l441_441149

theorem solution_set_inequality_five_elements (x a : ℕ) (hxpos : 0 < x) (hapos : 0 < a) 
  (hset : {x | |x - 1| < a}.card = 5) : a = 5 :=
sorry

end solution_set_inequality_five_elements_l441_441149


namespace isosceles_triangle_of_parallel_l441_441776

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441776


namespace ProblemConcurrentLines_l441_441085

theorem ProblemConcurrentLines
  (ABC : Triangle)
  (circle_BC : Circle)
  (H : Orthocenter ABC)
  (H' : Orthocenter (triangleADE : Triangle))
  (D E : Point)
  (BD CE HH' are concurrent : Prop)
  (on_circle : is_on (B C D E))
  (H : is_orthocenter ABC)
  (H' : is_orthocenter (triangleADE : Triangle))
  (intersects_AB : intersects_side E (side AB))
  (intersects_AC : intersects_side D (side AC))
  : BD ∧ CE ∧ HH' are_concurrent :=
  sorry

end ProblemConcurrentLines_l441_441085


namespace roots_quadratic_sum_of_squares_l441_441198

theorem roots_quadratic_sum_of_squares (a b m n : ℝ)
  (h1 : a + b = m)
  (h2 : a * b = n)
  (h3 : Polynomial.eval₂ Polynomial.C Polynomial.X (Polynomial.X^2 - m * Polynomial.X + n) a = 0)
  (h4 : Polynomial.eval₂ Polynomial.C Polynomial.X (Polynomial.X^2 - m * Polynomial.X + n) b = 0) :
  a^2 + b^2 = m^2 - 2 * n :=
by {
  -- The proof would go here
  sorry
}

end roots_quadratic_sum_of_squares_l441_441198


namespace probability_of_inequality_l441_441582

def problem : Prop :=
  ∃ (P : ℝ), 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → x ∈ set.Icc (0 : ℝ) (2 * Real.pi)) →
  (∀ (x : ℝ), x ∈ set.Icc (0 : ℝ) (2 * Real.pi) → 2 * Real.sin x > 1) →
  P = 1 / 3

theorem probability_of_inequality : problem := 
sorry

end probability_of_inequality_l441_441582


namespace triangle_is_isosceles_l441_441894

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441894


namespace triangle_ABC_is_isosceles_l441_441904

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441904


namespace prob_solution_l441_441598

def given_cond (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 / 2 < x ∧ x < 2 -> ax^2 + 5x - 2 > 0)

theorem prob_solution (a : ℝ) :
  (given_cond a) →
  a = -2 ∧ ( ∀ x : ℝ, (-3 < x ∧ x < 1 / 2 -> a * x^2 - 5 * x + a^2 - 1 > 0) ) :=
by
  sorry

end prob_solution_l441_441598


namespace solution_set_of_inequality_l441_441588

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_decreasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → (f' x) < 0

theorem solution_set_of_inequality :
  is_even f →
  is_decreasing_on_positive f →
  (f (x^2 - x) - f x > 0) ↔ x ∈ (set.Ioo 0 2) :=
sorry

end solution_set_of_inequality_l441_441588


namespace probability_of_winning_pair_l441_441047

/--
A deck consists of five red cards and five green cards, with each color having cards labeled from A to E. 
Two cards are drawn from this deck.
A winning pair is defined as two cards of the same color or two cards of the same letter. 
Prove that the probability of drawing a winning pair is 5/9.
-/
theorem probability_of_winning_pair :
  let total_cards := 10
  let total_ways := Nat.choose total_cards 2
  let same_letter_ways := 5
  let same_color_red_ways := Nat.choose 5 2
  let same_color_green_ways := Nat.choose 5 2
  let same_color_ways := same_color_red_ways + same_color_green_ways
  let favorable_outcomes := same_letter_ways + same_color_ways
  favorable_outcomes / total_ways = 5 / 9 := by
  sorry

end probability_of_winning_pair_l441_441047


namespace isosceles_triangle_of_parallel_l441_441886

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441886


namespace isosceles_triangle_of_parallel_l441_441772

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441772


namespace Josie_waited_18_minutes_l441_441270

/- Assume the given conditions -/
variables (w_cart w_employee w_stocker t_shopping t_total_shopping_trip t_waiting_in_line : ℕ)

-- Define the conditions given in the problem
def conditions : Prop :=
  w_cart = 3 ∧ 
  w_employee = 13 ∧ 
  w_stocker = 14 ∧ 
  t_shopping = 42 ∧ 
  t_total_shopping_trip = 90

-- Define the required property to be proven
def waiting_in_line_correct : Prop := 
  (t_total_shopping_trip - t_shopping) - (w_cart + w_employee + w_stocker) = t_waiting_in_line

-- Main theorem: Prove that Josie waited 18 minutes in line to check out
theorem Josie_waited_18_minutes :
  conditions →
  t_waiting_in_line = 18 :=
by sorry

end Josie_waited_18_minutes_l441_441270


namespace bridge_length_is_205_l441_441950

-- Definitions for the given conditions
def length_of_train : ℝ := 170
def speed_kmph : ℝ := 45
def crossing_time : ℝ := 30

-- Conversion factor from km/hr to m/s
def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Calculate the speed in m/s
def speed_mps := kmph_to_mps speed_kmph

-- Calculate the total distance travelled in 30 seconds
def total_distance := speed_mps * crossing_time

-- Calculate the length of the bridge
def length_of_bridge := total_distance - length_of_train

-- The theorem to prove
theorem bridge_length_is_205 : length_of_bridge = 205 := by
  -- The focus is on the statement, so proof details are skipped
  sorry

end bridge_length_is_205_l441_441950


namespace problem_statement_l441_441748

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441748


namespace snow_white_last_trip_l441_441333

universe u

-- Define the dwarf names as an enumerated type
inductive Dwarf : Type
| Happy | Grumpy | Dopey | Bashful | Sleepy | Doc | Sneezy
deriving DecidableEq, Repr

open Dwarf

-- Define conditions
def initial_lineup : List Dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- Define the condition of adjacency
def adjacent (d1 d2 : Dwarf) : Prop :=
  List.pairwise_adjacent (· = d2) initial_lineup d1

-- Define the boat capacity
def boat_capacity : Fin 4 := by sorry

-- Snow White is the only one who can row
def snow_white_rows : Prop := true

-- No quarrels condition
def no_quarrel_without_snow_white (group : List Dwarf) : Prop :=
  ∀ d1 d2, d1 ∈ group → d2 ∈ group → ¬ adjacent d1 d2

-- Objective: Transfer all dwarfs without quarrels
theorem snow_white_last_trip (trips : List (List Dwarf)) :
  ∃ last_trip : List Dwarf,
    last_trip = [Grumpy, Bashful, Doc] ∧
    no_quarrel_without_snow_white (initial_lineup.diff (trips.join ++ last_trip)) :=
sorry

end snow_white_last_trip_l441_441333


namespace isosceles_triangle_l441_441758

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441758


namespace find_c_l441_441261

def p (x : ℝ) : ℝ := 3 * x - 8
def q (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

theorem find_c (c : ℝ) (h : p (q 3 c) = 14) : c = 23 / 3 :=
by
  sorry

end find_c_l441_441261


namespace binomial_expansion_l441_441278

theorem binomial_expansion (a b : ℤ) (m : ℕ) (h_pos : 0 < m) :
  (a + b) ^ m = a ^ m + m * a ^ (m - 1) * b + b ^ 2 * (∑ k in Finset.range (m - 1).succ \ {0, 1}, (Nat.choose m k) * a ^ (m - k) * b ^ (k - 2)) :=
by
  sorry

end binomial_expansion_l441_441278


namespace parabola_equation_existence_of_line_l_l441_441177

/-- A proof problem regarding a parabola and intersecting lines -/

-- Condition definitions
def parabola (x y : ℝ) (p : ℝ) := x^2 = 2 * p * y
def point_on_parabola (x y p : ℝ) := parabola x y p

def focus_distance (p y0 : ℝ) := abs (y0 + p / 2)
def intercept_distance (p y0 : ℝ) := abs (y0)

def right_angle_at_A (A M N : ℝ × ℝ) :=
  let (Ax, Ay) := A in
  let (Mx, My) := M in
  let (Nx, Ny) := N in
  (Mx + Ax) * (Nx + Ax) + (My - Ay) * (Ny - Ay) = 0

-- Hypotheses
variable {p : ℝ} (hp : 0 < p)
variable {a : ℝ} (ha : 0 < a)

-- Problem 1: Proving the equation of the parabola
theorem parabola_equation :
  (∀ (x y : ℝ), parabola x y p → x^2 = 4 * y) :=
sorry

-- Problem 2: Proving the existence of the line l
theorem existence_of_line_l :
  ∃ (k : ℝ), k = 1 ∧ (∀ x y, parabola x y 2 → y = x + 4) ∧
  (∃ M N : ℝ × ℝ, 
     let line_l (x y : ℝ) := y = k * x + 4 in
     (line_l M.1 M.2 ∧ point_on_parabola M.1 M.2 2) ∧
     (line_l N.1 N.2 ∧ point_on_parabola N.1 N.2 2) ∧
     right_angle_at_A (-a, a) M N) :=
sorry

end parabola_equation_existence_of_line_l_l441_441177


namespace part1_intervals_of_increase_part2_number_of_zeros_in_interval_l441_441174

noncomputable def f (ω x : ℝ) : ℝ := 2 * sin(ω * x) * cos(ω * x) - sqrt(3) + 2 * sqrt(3) * (sin(ω * x))^2
noncomputable def g (x : ℝ) : ℝ := 2 * sin(2 * x) - 1

theorem part1_intervals_of_increase (ω : ℝ) (hω : ω > 0) (ht : (2 * π / ω) = π) : 
  ∀ k : ℤ, ∀ x : ℝ, k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12 :=
sorry

theorem part2_number_of_zeros_in_interval : 
  ∃ n : ℕ, ∀ x : ℝ, 0 ≤ x ∧ x ≤ 20, g x = 0 ↔ n = 40 :=
sorry

end part1_intervals_of_increase_part2_number_of_zeros_in_interval_l441_441174


namespace final_trip_theorem_l441_441292

/-- Define the lineup of dwarfs -/
inductive Dwarf where
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

open Dwarf

/-- Define the conditions -/
-- The dwarfs initially standing next to each other
def adjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = Happy ∧ d2 = Grumpy) ∨
  (d1 = Grumpy ∧ d2 = Dopey) ∨
  (d1 = Dopey ∧ d2 = Bashful) ∨
  (d1 = Bashful ∧ d2 = Sleepy) ∨
  (d1 = Sleepy ∧ d2 = Doc) ∨
  (d1 = Doc ∧ d2 = Sneezy)

-- Snow White is the only one who can row
constant snowWhite_can_row : Prop := true

-- The boat can hold Snow White and up to 3 dwarfs
constant boat_capacity : ℕ := 4

-- Define quarrel if left without Snow White
def quarrel_without_snowwhite (d1 d2 : Dwarf) : Prop := adjacent d1 d2

-- Define the final trip setup
def final_trip (dwarfs : List Dwarf) : Prop :=
  dwarfs = [Grumpy, Bashful, Doc]

-- Theorem to prove the final trip
theorem final_trip_theorem : ∃ dwarfs, final_trip dwarfs :=
  sorry

end final_trip_theorem_l441_441292


namespace triangle_is_isosceles_l441_441898

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441898


namespace triangle_isosceles_if_parallel_l441_441870

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441870


namespace magic_square_sum_l441_441228

variable {a b c d e : ℕ} {S : ℕ}

-- Conditions from the problem
def cond1 : Prop := 30 + e + 18 = S
def cond2 : Prop := 15 + c + d = S
def cond3 : Prop := a + 27 + b = S
def cond4 : Prop := 30 + 15 + a = S
def cond5 : Prop := e + c + 27 = S
def cond6 : Prop := 18 + d + b = S
def cond7 : Prop := 30 + c + b = S
def cond8 : Prop := 18 + c + a = S

theorem magic_square_sum :
  cond1 → cond2 → cond3 → cond4 → cond5 → cond6 → cond7 → cond8 → d + e = 108 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end magic_square_sum_l441_441228


namespace max_ab_l441_441244

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4*b + a*b = 3) : ab ≤ 1 :=
begin
  sorry
end

end max_ab_l441_441244


namespace lambda_exists_l441_441557

theorem lambda_exists (a1 a2 a3 b1 b2 b3 : ℕ) (a1_pos : a1 > 0) (a2_pos : a2 > 0) (a3_pos : a3 > 0) (b1_pos : b1 > 0) (b2_pos : b2 > 0) (b3_pos : b3 > 0) :
  ∃ (λ1 λ2 λ3 : ℕ), (λ1 ∈ {0, 1, 2}) ∧ (λ2 ∈ {0, 1, 2}) ∧ (λ3 ∈ {0, 1, 2}) ∧ 
  (λ1 + λ2 + λ3 ≠ 0) ∧ 
  ((λ1 * a1 + λ2 * a2 + λ3 * a3) % 3 = 0) ∧ ((λ1 * b1 + λ2 * b2 + λ3 * b3) % 3 = 0) := 
  sorry

end lambda_exists_l441_441557


namespace mary_cleaned_homes_l441_441696

theorem mary_cleaned_homes (earning_per_home : ℝ) (total_earning : ℝ) (h1 : earning_per_home = 46.0) (h2 : total_earning = 12696.0) :
  total_earning / earning_per_home = 276 :=
by
  rw [h1, h2]
  norm_num
  sorry

end mary_cleaned_homes_l441_441696


namespace problem_statement_l441_441738

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441738


namespace triangle_is_isosceles_l441_441817

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441817


namespace percentage_increase_bears_with_assistant_l441_441653

theorem percentage_increase_bears_with_assistant
  (B H : ℝ)
  (h_positive_hours : H > 0)
  (h_positive_bears : B > 0)
  (hours_with_assistant : ℝ := 0.90 * H)
  (rate_increase : ℝ := 2 * B / H) :
  ((rate_increase * hours_with_assistant) - B) / B * 100 = 80 := by
  -- This is the statement for the given problem.
  sorry

end percentage_increase_bears_with_assistant_l441_441653


namespace triangle_is_isosceles_l441_441860

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441860


namespace median_eq_mean_sum_l441_441979

noncomputable def mean (a b c d e : ℝ) : ℝ :=
  (a + b + c + d + e) / 5

noncomputable def is_median (a b c d e m : ℝ) : Prop :=
  (∃ (l1 l2 l3 l4 l5 : ℝ), l1 ≤ l2 ∧ l2 ≤ l3 ∧ l3 ≤ l4 ∧ l4 ≤ l5 ∧
  {l1, l2, l3, l4, l5} = {a, b, c, d, e} ∧ l3 = m)

theorem median_eq_mean_sum (x : ℝ) (h : is_median 3 5 7 x 20 ((mean 3 5 7 x 20))) : 
  x = -10 := 
sorry

end median_eq_mean_sum_l441_441979


namespace snow_white_last_trip_l441_441322

-- Definitions based on the problem's conditions
inductive Dwarf
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

def is_adjacent (d1 d2 : Dwarf) : Prop :=
  (d1, d2) ∈ [
    (Dwarf.Happy, Dwarf.Grumpy),
    (Dwarf.Grumpy, Dwarf.Happy),
    (Dwarf.Grumpy, Dwarf.Dopey),
    (Dwarf.Dopey, Dwarf.Grumpy),
    (Dwarf.Dopey, Dwarf.Bashful),
    (Dwarf.Bashful, Dwarf.Dopey),
    (Dwarf.Bashful, Dwarf.Sleepy),
    (Dwarf.Sleepy, Dwarf.Bashful),
    (Dwarf.Sleepy, Dwarf.Doc),
    (Dwarf.Doc, Dwarf.Sleepy),
    (Dwarf.Doc, Dwarf.Sneezy),
    (Dwarf.Sneezy, Dwarf.Doc)
  ]

def boat_capacity : ℕ := 3

variable (snowWhite : Prop)

-- The theorem to prove that the dwarfs Snow White will take in the last trip are Grumpy, Bashful and Doc
theorem snow_white_last_trip 
  (h1 : snowWhite)
  (h2 : boat_capacity = 3)
  (h3 : ∀ d1 d2, is_adjacent d1 d2 → snowWhite)
  : (snowWhite ∧ (Dwarf.Grumpy ∧ Dwarf.Bashful ∧ Dwarf.Doc)) :=
sorry

end snow_white_last_trip_l441_441322


namespace largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441413

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end largest_positive_integer_not_sum_of_multiple_30_and_composite_l441_441413


namespace total_weight_and_deficiency_l441_441378

theorem total_weight_and_deficiency 
  (weights : List ℤ)
  (standard_weight total_baskets : ℕ)
  (h_weights : weights = [3, -6, -4, 2, -1])
  (h_standard_weight : standard_weight = 50)
  (h_total_baskets : total_baskets = 5) :
  let total_deficiency := weights.sum in
  let total_standard_weight := standard_weight * total_baskets in
  total_deficiency = -6 ∧ 
  total_standard_weight + total_deficiency = 244 := 
by
  sorry

end total_weight_and_deficiency_l441_441378


namespace snow_white_last_trip_l441_441305

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l441_441305


namespace correct_option_is_C_l441_441985

-- Define the options as propositions
def optionA := ∀ x, x - (1 / (x - 1)) = 0
def optionB := ∀ x, 7 * x^2 + (1 / x^2) - 1 = 0
def optionC := ∀ x, x^2 = 0
def optionD := ∀ x, (x + 1) * (x - 2) = x * (x + 1)

-- Define what it means to be a quadratic equation
def is_quadratic_eq (eq : ∀ x, x^2 = 0) := 
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, a * x^2 + b * x + c = 0

-- Proof statement that Option C is the quadratic equation and others are not
theorem correct_option_is_C : 
  (∃ x, ∀ eq, optionA → ¬is_quadratic_eq optionA) ∧
  (∃ x, ∀ eq, optionB → ¬is_quadratic_eq optionB) ∧
  (∃ x, ∀ eq, optionC → is_quadratic_eq optionC) ∧
  (∃ x, ∀ eq, optionD → ¬is_quadratic_eq optionD) :=
sorry

end correct_option_is_C_l441_441985


namespace final_trip_l441_441326

-- Definitions for the conditions
def dwarf := String
def snow_white := "Snow White" : dwarf
def Happy : dwarf := "Happy"
def Grumpy : dwarf := "Grumpy"
def Dopey : dwarf := "Dopey"
def Bashful : dwarf := "Bashful"
def Sleepy : dwarf := "Sleepy"
def Doc : dwarf := "Doc"
def Sneezy : dwarf := "Sneezy"

-- The dwarfs lineup from left to right
def lineup : List dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- The boat can hold Snow White and up to 3 dwarfs
def boat_capacity (load : List dwarf) : Prop := snow_white ∈ load ∧ load.length ≤ 4

-- Any two dwarfs standing next to each other in the original lineup will quarrel if left without Snow White
def will_quarrel (d1 d2 : dwarf) : Prop :=
  (d1, d2) ∈ (lineup.zip lineup.tail)

-- The objective: Prove that on the last trip, Snow White will take Grumpy, Bashful, and Doc
theorem final_trip : ∃ load : List dwarf, 
  set.load ⊆ {snow_white, Grumpy, Bashful, Doc} ∧ boat_capacity load :=
  sorry

end final_trip_l441_441326


namespace denominator_of_fraction_is_correct_l441_441952

theorem denominator_of_fraction_is_correct :
  let numerator := (100)!
  let denominator := (6 ^ 100)
  let num_factors_2 := 97
  let num_factors_3 := 48
  let irreducible_denominator := (2 ^ 3) * (3 ^ 52)
  irreducible_denominator = (2 ^ 3) * (3 ^ 52) :=
by
  sorry

end denominator_of_fraction_is_correct_l441_441952


namespace not_equivalent_condition_l441_441012

-- Define the main scientific notation we are comparing against
def target_notation : ℝ := 4.5 * 10^(-5)

-- Define each of the given options
def option_a : ℝ := 4.5 * 10^(-5)
def option_b : ℝ := 45 * 10^(-6)
def option_c : ℝ := (9 / 2) * 10^(-5)
def option_d : ℝ := (9 / 200) * 10^(-4)
def option_e : ℝ := 45 / 1000000

-- Theorem to prove option_d is not equivalent while others are equivalent
theorem not_equivalent_condition : 
  (option_a = target_notation) ∧ 
  (option_b = target_notation) ∧ 
  (option_c = target_notation) ∧ 
  (option_e = target_notation) ∧ 
  (option_d ≠ target_notation) := 
by 
  sorry

end not_equivalent_condition_l441_441012


namespace find_u_plus_v_l441_441605

theorem find_u_plus_v (u v : ℚ) (h1: 5 * u - 3 * v = 26) (h2: 3 * u + 5 * v = -19) :
  u + v = -101 / 34 :=
sorry

end find_u_plus_v_l441_441605


namespace perimeter_of_tangents_triangle_l441_441391

theorem perimeter_of_tangents_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
    (4 * a * Real.sqrt (a * b)) / (a - b) = 4 * a * (Real.sqrt (a * b) / (a - b)) := 
sorry

end perimeter_of_tangents_triangle_l441_441391


namespace continued_fraction_equality_l441_441280

theorem continued_fraction_equality (n : ℕ) :
  continued_fraction (repeat 2 n) = ( (1 + Real.sqrt 2)^(n + 1) - (1 - Real.sqrt 2)^(n + 1)) / ((1 + Real.sqrt 2)^n - (1 - Real.sqrt 2)^n) :=
sorry

end continued_fraction_equality_l441_441280


namespace value_of_x_l441_441375

variable (x y z : ℕ)

theorem value_of_x : x = 10 :=
  assume h1 : x = y / 2,
  assume h2 : y = z / 4,
  assume h3 : z = 80,
  sorry

end value_of_x_l441_441375


namespace problem_statement_l441_441994

-- Define a set S
variable {S : Type*}

-- Define the binary operation on S
variable (mul : S → S → S)

-- Assume the given condition: (a * b) * a = b for all a, b in S
axiom given_condition : ∀ (a b : S), (mul (mul a b) a) = b

-- Prove that a * (b * a) = b for all a, b in S
theorem problem_statement : ∀ (a b : S), mul a (mul b a) = b :=
by
  sorry

end problem_statement_l441_441994


namespace triangle_ABC_isosceles_l441_441838

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441838


namespace triangle_is_isosceles_l441_441901

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441901


namespace distinct_counts_eq_l441_441993

theorem distinct_counts_eq (boxes : List ℕ) :
  let trays := List.range (boxes.foldr Nat.max 0)
                |> List.foldl (λ acc i => acc ++ List.map (λ b => if b > i then b - i else 0) boxes) []
  boxes.erase_dup.length = trays.erase_dup.length :=
by
  sorry

end distinct_counts_eq_l441_441993


namespace sum_arithmetic_series_l441_441501

def first_term : Int := -45
def common_difference : Int := 2
def last_term : Int := 23

theorem sum_arithmetic_series :
  (∃ n : Nat, first_term + (n - 1) * common_difference = last_term) ∧
  (∑ i in Finset.range 35, first_term + i * common_difference) = -385 :=
by
  sorry

end sum_arithmetic_series_l441_441501


namespace actual_distance_between_towns_l441_441938

noncomputable def map_distance := 18 -- The distance on the map in inches
noncomputable def scale_inches := 0.5 -- Scale inches
noncomputable def scale_miles := 6 -- Scale miles
noncomputable def actual_miles (d : ℕ) (si sm : ℕ) := d * (sm / si)

theorem actual_distance_between_towns:
  actual_miles map_distance scale_inches scale_miles = 216 := 
  sorry

end actual_distance_between_towns_l441_441938


namespace positive_integer_solutions_l441_441537

theorem positive_integer_solutions
  (m n k : ℕ)
  (hm : 0 < m) (hn : 0 < n) (hk : 0 < k) :
  3 * m + 4 * n = 5 * k ↔ (m = 1 ∧ n = 2 ∧ k = 2) := 
by
  sorry

end positive_integer_solutions_l441_441537


namespace find_number_l441_441196

theorem find_number (X : ℝ) (h : 50 = 0.20 * X + 47) : X = 15 :=
sorry

end find_number_l441_441196


namespace isosceles_triangle_of_parallel_l441_441775

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441775


namespace symmetric_y_axis_l441_441611

theorem symmetric_y_axis (θ : ℝ) :
  (cos θ = -cos (θ + π / 6)) ∧ (sin θ = sin (θ + π / 6)) → θ = 5 * π / 12 :=
by
  sorry

end symmetric_y_axis_l441_441611


namespace asia_discount_problem_l441_441088

theorem asia_discount_problem
  (originalPrice : ℝ)
  (storeDiscount : ℝ)
  (memberDiscount : ℝ)
  (finalPriceUSD : ℝ)
  (exchangeRate : ℝ)
  (finalDiscountPercentage : ℝ) :
  originalPrice = 300 →
  storeDiscount = 0.20 →
  memberDiscount = 0.10 →
  finalPriceUSD = 224 →
  exchangeRate = 1.10 →
  finalDiscountPercentage = 28 :=
by
  sorry

end asia_discount_problem_l441_441088


namespace sum_of_four_consecutive_even_numbers_l441_441371

theorem sum_of_four_consecutive_even_numbers (n : ℤ) (h : n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344) :
  n + (n + 2) + (n + 4) + (n + 6) = 36 := sorry

end sum_of_four_consecutive_even_numbers_l441_441371


namespace scientific_notation_l441_441702

theorem scientific_notation (x : ℝ) (h : x = 0.00000008) : 
  x = 8 * 10^(-8) := 
by 
  simp [h]
  sorry

end scientific_notation_l441_441702


namespace problem_statement_l441_441555

-- Define necessary conditions and parameters
variables {n : ℕ} 
variables {r s t u v : Fin n → ℝ} 

-- Define each r_i, s_i, t_i, u_i, v_i > 1
axiom h_r : ∀ i, r i > 1
axiom h_s : ∀ i, s i > 1
axiom h_t : ∀ i, t i > 1
axiom h_u : ∀ i, u i > 1
axiom h_v : ∀ i, v i > 1

-- Define the averages
def R := (∑ i in Finset.range n, r i) / n
def S := (∑ i in Finset.range n, s i) / n
def T := (∑ i in Finset.range n, t i) / n
def U := (∑ i in Finset.range n, u i) / n
def V := (∑ i in Finset.range n, v i) / n

theorem problem_statement : 
  (∑ i in Finset.range n, (r i * s i * t i * u i * v i + 1) / (r i * s i * t i * u i * v i - 1)) 
  ≥ (π.sqrt (R * S * T * U * V) + 1 / (R * S * T * U * V - 1)) ^ n :=
sorry

end problem_statement_l441_441555


namespace triangle_ABC_is_isosceles_l441_441903

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441903


namespace largest_integer_not_sum_of_30_and_composite_l441_441418

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_not_sum_of_30_and_composite (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ b < 30 ∧ is_composite(b) ∧ n = 30 * a + b

theorem largest_integer_not_sum_of_30_and_composite :
  ∃ n : ℕ, is_not_sum_of_30_and_composite(n) ∧ ∀ m : ℕ, is_not_sum_of_30_and_composite(m) → m ≤ n :=
  ⟨93, sorry⟩

end largest_integer_not_sum_of_30_and_composite_l441_441418


namespace sin_theta_of_triangle_area_side_median_l441_441487

-- Defining the problem statement and required conditions
theorem sin_theta_of_triangle_area_side_median (A : ℝ) (a m : ℝ) (θ : ℝ) 
  (hA : A = 30)
  (ha : a = 12)
  (hm : m = 8)
  (hTriangleArea : A = 1/2 * a * m * Real.sin θ) :
  Real.sin θ = 5 / 8 :=
by
  -- Proof omitted
  sorry

end sin_theta_of_triangle_area_side_median_l441_441487


namespace min_magnitude_lemma_l441_441678

noncomputable def min_magnitude (p q r : ℤ) (ξ : ℂ) : ℝ :=
|p + q * ξ + r * ξ^3|

theorem min_magnitude_lemma (p q r : ℤ) (ξ : ℂ) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) (hξ_norm : ξ^4 = 1) (hξ_nontrivial : ξ ≠ 1) : 
  ∃ p q r : ℤ, min_magnitude p q r ξ = real.sqrt 5 := 
sorry

end min_magnitude_lemma_l441_441678


namespace max_min_diff_function_domain_trigonometric_inequality_l441_441158

noncomputable def g (t : ℝ) : ℝ :=
  (8 * Real.sqrt (t ^ 2 + 1) * (2 * t ^ 2 + 5)) / (16 * t ^ 2 + 25)

theorem max_min_diff_function_domain:
  (∀ α β t : ℝ,
    (Polynominal.degree (4 * X^2 - 4 * t * X - 1) = 2) → 
    (∃ α β : ℝ, IsRoot (4 * X^2 - 4 * t * X - 1) α ∧ IsRoot (4 * X^2 - 4 * t * X - 1) β ∧ α ≠ β) → 
    g(t) = {(8 * Real.sqrt (t ^ 2 + 1) * (2 * t ^ 2 + 5)) / (16 * t ^ 2 + 25)}) :=
sorry

theorem trigonometric_inequality:
  (u1 u2 u3 : ℝ) (hu : ∀ i : ℕ, 0 < u i /\
    u i < π / 2) (hsum: Real.sin u1 + Real.sin u2 + Real.sin u3 = 1) →
    ( (1 / g (Real.tan u1)) + (1 / g (Real.tan u2)) + (1 / g (Real.tan u3)) < 3 * Real.sqrt 6 / 4) :=
sorry

end max_min_diff_function_domain_trigonometric_inequality_l441_441158


namespace decreasing_interval_sin_2x_minus_2x_l441_441350

theorem decreasing_interval_sin_2x_minus_2x (k : ℤ) :
  let f : ℝ → ℝ := λ x, sin (2 * x) - 2 * x,
  a : ℝ := k * π + π / 4,
  b : ℝ := k * π + 3 * π / 4 in
  ∀ x : ℝ, a ≤ x ∧ x ≤ b → f x = sin (2 * x) - 2 * x ∧ f x < 0 :=
by
  sorry

end decreasing_interval_sin_2x_minus_2x_l441_441350


namespace solution_set_xf_lt_0_l441_441249

theorem solution_set_xf_lt_0 (f : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_decr : ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ ≥ f x₂)
  (h_f_neg1 : f (-1) = 0) :
  {x : ℝ | x * f x < 0} = set.Ico (-1 : ℝ) 0 ∪ set.Ioi 1 :=
by 
  sorry

end solution_set_xf_lt_0_l441_441249


namespace distance_between_A_and_B_l441_441155

def A : ℝ × ℝ × ℝ := (1, 3, -2)
def B : ℝ × ℝ × ℝ := (-2, 3, 2)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_between_A_and_B : distance A B = 5 := by
  sorry

end distance_between_A_and_B_l441_441155


namespace twice_product_of_numbers_l441_441963

theorem twice_product_of_numbers (x y : ℝ) (h1 : x + y = 80) (h2 : x - y = 10) : 2 * (x * y) = 3150 := by
  sorry

end twice_product_of_numbers_l441_441963


namespace triangle_ABC_isosceles_l441_441834

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441834


namespace pascal_triangle_41_39_l441_441977

-- Define the factorial function
noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

-- Define the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

-- Define the 39th number in the 41st row of Pascal's triangle
def pascal_41_39 := binomial 40 2

-- Prove that the 39th number is 780
theorem pascal_triangle_41_39 : pascal_41_39 = 780 := by
  sorry

end pascal_triangle_41_39_l441_441977


namespace find_r_l441_441538

theorem find_r (r : ℝ) (h : ⌊r⌋ + r = 20.7) : r = 10.7 := 
by 
  sorry 

end find_r_l441_441538


namespace max_faces_seen_l441_441981

theorem max_faces_seen (position : ℝ^3) (object : Set (ℝ^3)) (rectangular_prism : Set (ℝ^3)) :
  (∀ pos ∈ object, pos = position) ∧ (∀ v ∈ rectangular_prism.vertices, faces_seen_from v = 3) → maximal_faces_seen_from_one_position = 3 :=
sorry

end max_faces_seen_l441_441981


namespace possible_values_of_a_l441_441381

theorem possible_values_of_a :
  ∃ a b c : ℤ, 
    (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ↔ 
    (a = 3 ∨ a = 7) :=
by
  sorry

end possible_values_of_a_l441_441381


namespace range_even_function_l441_441579

theorem range_even_function (a b : ℝ) (h1 : ∀ x : ℝ, f x = a * x^2 + b * x + 2)
  (h2 : ∀ x : ℝ, f (-x) = f x)
  (domain : set.Icc (1 + a) 2) :
  set.range (λ x, f x) = set.Icc (-10 : ℝ) 2 :=
by
  unfold f at *
  sorry

end range_even_function_l441_441579


namespace isosceles_triangle_l441_441829

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441829


namespace isosceles_triangle_l441_441761

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441761


namespace flo_initial_seat_l441_441927

variables (initial_seats : Fin 6 → ℕ) -- Represent the initial seats where each friend sits (1-6)
variable (flo_initial_edge : initial_seats ⟨5, by norm_num⟩ = 1 ∨ initial_seats ⟨5, by norm_num⟩ = 6) -- Flo's initial position is on an edge
variable (ada_moves : initial_seats ⟨0, by norm_num⟩ = initial_seats ⟨0, by norm_num⟩ - 1) -- Ada moves one seat to the left
variable (bea_moves : initial_seats ⟨1, by norm_num⟩ = initial_seats ⟨1, by norm_num⟩ + 3) -- Bea moves three seats to the right
variable (ceci_dee_switch : initial_seats ⟨2, by norm_num⟩ = initial_seats ⟨3, by norm_num⟩ ∧ initial_seats ⟨3, by norm_num⟩= initial_seats ⟨2, by norm_num⟩) -- Ceci and Dee switch seats
variable (edie_moves : initial_seats ⟨4, by norm_num⟩= initial_seats ⟨4, by norm_num⟩ + 1) -- Edie moves one seat to the right
variable (flo_ends_on_edge : initial_seats ⟨5, by norm_num⟩ = 1 ∨ initial_seats ⟨5, by norm_num⟩ = 6) -- Flo returns to an end seat

theorem flo_initial_seat : flo_initial_edge → flo_ends_on_edge → initial_seats ⟨5, by norm_num⟩ = 6 := sorry

end flo_initial_seat_l441_441927


namespace max_triangle_area_l441_441594

-- Define the ellipse equation
def is_on_ellipse (x y b : ℝ) : Prop :=
  (x^2) / 4 + (y^2) / (b^2) = 1

-- Define the ordinate range for b
def valid_b (b : ℝ) : Prop := 0 < b ∧ b < 2

-- Define the focus coordinates
def focus_coordinates (b : ℝ) : ℝ × ℝ := (Math.sqrt (4 - b^2), 0)

-- Area of triangle ABF
def triangle_area (b : ℝ) : ℝ :=
  b * (Math.sqrt (4 - b^2))

-- Main theorem statement
theorem max_triangle_area : ∀ b, valid_b b → (triangle_area b) ≤ 2 :=
  by
    sorry

end max_triangle_area_l441_441594


namespace trajectory_of_M_l441_441169

noncomputable def trajectory_eq (M: ℝ × ℝ) (A: ℝ × ℝ := (-3, 0)) (B: ℝ × ℝ := (3, 0)) : Prop :=
  let (x, y) := M in
  ((y / (x - 3)) * (y / (x + 3)) = 4) → (x^2 / 9 - y^2 / 36 = 1)

theorem trajectory_of_M (M: ℝ × ℝ) (h: trajectory_eq M) : 
  let (x, y) := M in (x^2 / 9 - y^2 / 36 = 1) :=
by
  sorry

end trajectory_of_M_l441_441169


namespace general_formula_find_k_l441_441570

-- Defining the arithmetic sequence with its properties
def arith_seq (a : ℕ → ℝ) :=
  ∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

-- Given conditions as Lean definitions
def conditions (a : ℕ → ℝ) (a1 d : ℝ) :=
  a 2 + a 5 = 19 ∧ a 6 - a 3 = 9

-- Statement for part 1
theorem general_formula (a : ℕ → ℝ) (a1 d : ℝ) (h : conditions a a1 d) :
  arith_seq a :=
sorry

-- Sum of first n terms of an arithmetic sequence
def sum_arith_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

-- Statement for part 2
theorem find_k (a : ℕ → ℝ) (a1 d : ℝ) (h : conditions a a1 d) 
  (S11 :  sum_arith_seq a 11 = 187) :
  ∃ k, sum_arith_seq a 11 + sum_arith_seq a k = sum_arith_seq a (k + 2) :=
sorry

end general_formula_find_k_l441_441570


namespace tan_ratio_sum_l441_441679

-- Define x and y as real numbers
variables (x y : ℝ)

-- Define the conditions as hypotheses
axiom h1 : (sin x / cos y) + (sin y / cos x) = 2
axiom h2 : (cos x / sin y) + (cos y / sin x) = 3

-- State the theorem to be proven
theorem tan_ratio_sum : (tan x / tan y) + (tan y / tan x) = 10 / 9 :=
by
  sorry

end tan_ratio_sum_l441_441679


namespace isosceles_triangle_l441_441831

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441831


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l441_441970

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ p2 = p1 + 1 ∧ is_prime p3 ∧ p3 = p2 + 1

def sum_divisible_by_5 (p1 p2 p3 : ℕ) : Prop :=
  (p1 + p2 + p3) % 5 = 0

theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (p1 p2 p3 : ℕ), consecutive_primes p1 p2 p3 ∧ sum_divisible_by_5 p1 p2 p3 ∧ p1 + p2 + p3 = 10 :=
by
  sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l441_441970


namespace shooting_probability_l441_441473

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

theorem shooting_probability (acc : ℝ) (hits total consecutive : ℕ) 
  (h_acc : acc = 0.6) (h_hits : hits = 5) (h_total : total = 8) (h_consecutive : consecutive = 4) : 
  (A 4 2 * acc^5 * (1 - acc)^3) = A 4 2 * 0.6^5 * 0.4^3 :=
by
  rw [h_acc, h_hits, h_total, h_consecutive]
  sorry

end shooting_probability_l441_441473


namespace triangle_is_isosceles_l441_441816

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441816


namespace isosceles_triangle_of_parallel_l441_441884

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441884


namespace isosceles_triangle_l441_441795

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441795


namespace min_B_minus_A_l441_441572

noncomputable def S_n (n : ℕ) : ℚ :=
  let a1 : ℚ := 2
  let r : ℚ := -1 / 3
  a1 * (1 - r ^ n) / (1 - r)

theorem min_B_minus_A :
  ∃ A B : ℚ, 
    (∀ n : ℕ, 1 ≤ n → A ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B) ∧
    ∀ A' B' : ℚ, 
      (∀ n : ℕ, 1 ≤ n → A' ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B') → 
      B' - A' ≥ 9 / 4 ∧ B - A = 9 / 4 :=
sorry

end min_B_minus_A_l441_441572


namespace snow_white_last_trip_dwarfs_l441_441298

-- Definitions for the conditions
def original_lineup := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]
def only_snow_white_can_row := True
def boat_capacity_snow_white_and_dwarfs := 3
def dwarfs_quarrel_if_adjacent (d1 d2 : String) : Prop :=
  let index_d1 := List.indexOf original_lineup d1
  let index_d2 := List.indexOf original_lineup d2
  abs (index_d1 - index_d2) = 1

-- Theorem to prove the correct answer
theorem snow_white_last_trip_dwarfs :
  let last_trip_dwarfs := ["Grumpy", "Bashful", "Sneezy"]
  ∃ (trip : List String), trip = last_trip_dwarfs ∧ 
  ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → d1 ≠ d2 → ¬dwarfs_quarrel_if_adjacent d1 d2 :=
by
  sorry

end snow_white_last_trip_dwarfs_l441_441298


namespace triangle_isosceles_if_parallel_l441_441869

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441869


namespace cyclist_wait_time_l441_441051

theorem cyclist_wait_time 
    (h_r : ℝ) -- hiker's rate in km/h
    (c_r : ℝ) -- cyclist's rate in km/h
    (t_s : ℝ) -- time in hours after which cyclist stops (5 minutes = 5/60 hours)
    : h_r = 4 ∧ c_r = 18 ∧ t_s = 5/60 → 
    let rel_speed := c_r - h_r in
    let rel_speed_min := rel_speed / 60 in -- relative speed in km/min
    let dist := rel_speed_min * t_s in
    let hiker_speed_min := h_r / 60 in -- hiker speed in km/min
    let wait_time := dist / hiker_speed_min in
    wait_time = 17.5 :=
begin
  sorry
end

end cyclist_wait_time_l441_441051


namespace units_digit_sum_l441_441645

theorem units_digit_sum (a b c d e f g : ℕ) 
  (h1 : (a ∈ {0, 1, 2, 3, 4, 5, 6}) ∧ (b ∈ {0, 1, 2, 3, 4, 5, 6}) ∧
        (c ∈ {0, 1, 2, 3, 4, 5, 6}) ∧ (d ∈ {0, 1, 2, 3, 4, 5, 6}) ∧
        (e ∈ {0, 1, 2, 3, 4, 5, 6}) ∧ (f ∈ {0, 1, 2, 3, 4, 5, 6}) ∧
        (g ∈ {0, 1, 2, 3, 4, 5, 6})) 
  (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ 
       b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ 
       c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ 
       d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ 
       e ≠ f ∧ e ≠ g ∧ 
       f ≠ g) 
  (h3 : e = 1) (h4 : (10 * a + b) + (10 * c + d) = 100 * e + 10 * f + g) :
  g = 5 :=
by sorry

end units_digit_sum_l441_441645


namespace largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441410

theorem largest_integer_not_sum_of_multiple_of_30_and_composite :
  ∃ n : ℕ, ∀ a b : ℕ, 0 ≤ b ∧ b < 30 ∧ b.prime ∧ (∀ k < a, (b + 30 * k).prime)
    → (30 * a + b = n) ∧
      (∀ m : ℕ, ∀ c d : ℕ, 0 ≤ d ∧ d < 30 ∧ d.prime ∧ (∀ k < c, (d + 30 * k).prime) 
        → (30 * c + d ≤ n)) ∧
      n = 93 :=
by
  sorry

end largest_integer_not_sum_of_multiple_of_30_and_composite_l441_441410


namespace odd_function_increasing_interval_l441_441953

noncomputable def f (x : ℝ) : ℝ := sorry

theorem odd_function_increasing_interval (h1 : ∀ x, f (-x) = -f x) 
  (h2 : ∀ x ∈ Icc (3 : ℝ) 7, increasing_on f (Icc 3 7))
  (h3 : ∀ x ∈ Icc (3 : ℝ) 6, (f x ≤ 8 ∧ f x ≥ 1)) 
  (h4 : f 3 = 1) 
  (h5 : f 6 = 8) :
  f (-3) + 2 * f 6 = 15 := 
begin
  -- Proof would go here
  sorry
end

end odd_function_increasing_interval_l441_441953


namespace triangle_is_isosceles_l441_441851

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441851


namespace integral_solution_l441_441510
noncomputable theory
open Real

theorem integral_solution :
  ∫ x in 0..(arccos (4 / sqrt 17)), 
  (3 + 2 * tan x) / (2 * sin x ^ 2 + 3 * cos x ^ 2 - 1) = 
  (3 / sqrt 2) * arctan (1 / (4 * sqrt 2)) + log (33 / 32) :=
by
  sorry

end integral_solution_l441_441510


namespace triangle_is_isosceles_if_parallel_l441_441726

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441726


namespace isosceles_triangle_l441_441752

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441752


namespace percentage_increase_bears_with_assistant_l441_441652

theorem percentage_increase_bears_with_assistant
  (B H : ℝ)
  (h_positive_hours : H > 0)
  (h_positive_bears : B > 0)
  (hours_with_assistant : ℝ := 0.90 * H)
  (rate_increase : ℝ := 2 * B / H) :
  ((rate_increase * hours_with_assistant) - B) / B * 100 = 80 := by
  -- This is the statement for the given problem.
  sorry

end percentage_increase_bears_with_assistant_l441_441652


namespace positive_int_solutions_of_inequality_l441_441111

theorem positive_int_solutions_of_inequality : 
  {x : ℕ | 14 < -2 * (x : ℤ) + 17}.to_finset.card = 1 :=
by
  sorry

end positive_int_solutions_of_inequality_l441_441111


namespace how_many_necklaces_given_away_l441_441093

-- Define the initial conditions
def initial_necklaces := 50
def broken_necklaces := 3
def bought_necklaces := 5
def final_necklaces := 37

-- Define the question proof statement
theorem how_many_necklaces_given_away : 
  (initial_necklaces - broken_necklaces + bought_necklaces - final_necklaces) = 15 :=
by sorry

end how_many_necklaces_given_away_l441_441093


namespace magnitude_sum_of_vectors_l441_441601

variables (a b : ℝ → ℝ)
variables (angle_ab : real.angle) (ha : |a| = 1) (hb : |b| = 1) (h_angle : angle_ab = real.angle.of_deg 60)

theorem magnitude_sum_of_vectors : 
  |a + b| = real.sqrt 3 :=
sorry

end magnitude_sum_of_vectors_l441_441601


namespace true_propositions_l441_441942

-- Definitions based on given conditions
def converse_additive_inverses (x y : ℝ) : Prop :=
  x + y = 0 → x = -y

def negation_congruent_triangle_areas (A B : ℝ) : Prop :=
  ¬ (A = B → ¬A = B)

def contrapositive_real_roots (q : ℝ) : Prop :=
  (q ≤ 1) → ∃ (x : ℝ), x^2 + 2*x + q = 0

def converse_equilateral_triangle (angles_equal : Prop) : Prop :=
  angles_equal → ∀ (a b : ℝ), a = b

-- The main proof problem
theorem true_propositions : converse_additive_inverses ∧ contrapositive_real_roots := by
  sorry

end true_propositions_l441_441942


namespace minimum_jumps_l441_441377

theorem minimum_jumps (a b : ℕ) (h : 2 * a + 3 * b = 2016) : a + b = 673 :=
sorry

end minimum_jumps_l441_441377


namespace isosceles_triangle_l441_441800

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441800


namespace order_of_logs_l441_441246

open Real

noncomputable def a := log 10 / log 5
noncomputable def b := log 12 / log 6
noncomputable def c := 1 + log 2 / log 7

theorem order_of_logs : a > b ∧ b > c :=
by
  sorry

end order_of_logs_l441_441246


namespace correct_sum_is_1826_l441_441439

-- Define the four-digit number representation
def four_digit (A B C D : ℕ) := 1000 * A + 100 * B + 10 * C + D

-- Condition: Yoongi confused the units digit (9 as 6)
-- The incorrect number Yoongi used
def incorrect_number (A B C : ℕ) := four_digit A B C 6

-- The correct number
def correct_number (A B C : ℕ) := four_digit A B C 9

-- The sum obtained by Yoongi
def yoongi_sum (A B C : ℕ) := incorrect_number A B C + 57

-- The correct sum 
def correct_sum (A B C : ℕ) := correct_number A B C + 57

-- Condition: Yoongi's sum is 1823
axiom yoongi_sum_is_1823 (A B C: ℕ) : yoongi_sum A B C = 1823

-- Proof Problem: Prove that the correct sum is 1826
theorem correct_sum_is_1826 (A B C : ℕ) : correct_sum A B C = 1826 := by
  -- The proof goes here
  sorry

end correct_sum_is_1826_l441_441439


namespace inequality_proof_l441_441258

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_proof_l441_441258


namespace isosceles_triangle_l441_441798

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441798


namespace expected_number_of_baskets_variance_of_baskets_l441_441491

theorem expected_number_of_baskets (p : ℝ) (n : ℕ) : p = 0.6 → n = 5 → let η := n * p in η = 3 :=
by
  intros h₁ h₂
  simp [h₁, h₂]

theorem variance_of_baskets (p : ℝ) (n : ℕ) : p = 0.6 → n = 5 → let D := n * p * (1 - p) in D = 1.2 :=
by
  intros h₁ h₂
  simp [h₁, h₂]

end expected_number_of_baskets_variance_of_baskets_l441_441491


namespace smallest_integer_gcd_6_l441_441426

theorem smallest_integer_gcd_6 : ∃ n : ℕ, n > 100 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n :=
by
  let n := 114
  have h1 : n > 100 := sorry
  have h2 : gcd n 18 = 6 := sorry
  have h3 : ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n := sorry
  exact ⟨n, h1, h2, h3⟩

end smallest_integer_gcd_6_l441_441426


namespace find_real_root_a_l441_441672

theorem find_real_root_a (a b c : ℂ) (ha : a.im = 0) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 3) : a = 1 :=
sorry

end find_real_root_a_l441_441672


namespace irrational_count_l441_441073

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_count :
  (∃ (x : ℝ), x = sqrt 8 ∧ is_irrational x) ∧
  (∃ (x : ℝ), x = 2 * real.pi ∧ is_irrational x) ∧
  (∃ (x : ℝ), x = 2 - sqrt 2 ∧ is_irrational x) ∧
  (∃ (x : ℝ), x = 1.21221222122221 ∧ is_irrational (x)) →
  (4 = 4) :=
by
    sorry

end irrational_count_l441_441073


namespace find_vector_a_l441_441179
-- Import Mathlib to bring in necessary mathematical definitions and theorems

-- Define the conditions
def magnitude (a : ℝ × ℝ) : ℝ := 
  real.sqrt (a.1 ^ 2 + a.2 ^ 2)

def orthogonal (a b : ℝ × ℝ) : Prop := 
  a.1 * b.1 + a.2 * b.2 = 0

-- Define the vectors a and b
def a := (3 : ℝ) -- Define this scalarly first for simplicity, we will have an actual vector as the solution
def b : ℝ × ℝ := (1, 2)

-- Theorem stating the coordinates of vector a
theorem find_vector_a (x y : ℝ) (h1 : magnitude (x, y) = 3) (h2 : orthogonal (x, y) b) :
  (x, y) = (-6 * real.sqrt 5 / 5, 3 * real.sqrt 5 / 5) 
    ∨ (x, y) = (6 * real.sqrt 5 / 5, -3 * real.sqrt 5 / 5) :=
  sorry

end find_vector_a_l441_441179


namespace find_polynomial_q_l441_441135

theorem find_polynomial_q :
  ∃ q : Polynomial ℝ, q = 9 * X^3 - 3 ∧ 
  ∀ x : ℝ, Polynomial.eval (x^3) q - Polynomial.eval (x^3 - 3) q = (Polynomial.eval x q)^2 - 18 := 
begin
  sorry
end

end find_polynomial_q_l441_441135


namespace isosceles_right_triangle_area_relationship_l441_441254

-- Definitions for triangles and their properties
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def area_of_isosceles_triangle (a b c : ℝ) (h : is_isosceles_triangle a b c) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

def area_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  (1 / 2) * a * b

-- Theorem statement
theorem isosceles_right_triangle_area_relationship :
  let A := area_of_isosceles_triangle 13 13 10 (Or.inl rfl)
  let B := area_of_right_triangle 5 12 13 (by norm_num [sq]; linarith)
  in A = 2 * B := by
  sorry

end isosceles_right_triangle_area_relationship_l441_441254


namespace largest_integer_not_sum_of_30_and_composite_l441_441416

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_not_sum_of_30_and_composite (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ b < 30 ∧ is_composite(b) ∧ n = 30 * a + b

theorem largest_integer_not_sum_of_30_and_composite :
  ∃ n : ℕ, is_not_sum_of_30_and_composite(n) ∧ ∀ m : ℕ, is_not_sum_of_30_and_composite(m) → m ≤ n :=
  ⟨93, sorry⟩

end largest_integer_not_sum_of_30_and_composite_l441_441416


namespace triangle_ABC_is_isosceles_l441_441914

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441914


namespace value_of_D_l441_441515

variables (A B C D : ℕ)

-- Conditions
def number1 : ℕ := 8 * 10^7 + 4 * 10^6 + A * 10^5 + 5 * 10^4 + 3 * 10^3 + B * 10^2 + 1 * 10 + C
def number2 : ℕ := 3 * 10^7 + 2 * 10^6 + 7 * 10^5 + A * 10^4 + B * 10^3 + 5 * 10^2 + C * 10 + D

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

-- Proof statement
theorem value_of_D 
  (h1 : is_multiple_of_4 number1)
  (h2 : is_multiple_of_3 number2)
  (hC : C = 2) :
  D = 2 :=
sorry

end value_of_D_l441_441515


namespace isosceles_triangle_of_parallel_l441_441766

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441766


namespace triangle_ABC_isosceles_l441_441835

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441835


namespace beth_extra_crayons_l441_441094

variable (packs : Nat) (crayons_per_pack : Nat) (total_crayons : Nat)

theorem beth_extra_crayons (packs_eq : packs = 4) (crayons_per_pack_eq : crayons_per_pack = 10)
                           (total_crayons_eq : total_crayons = 40) :
                           (total_crayons - (packs * crayons_per_pack) = 0) :=
by
  rw [packs_eq, crayons_per_pack_eq, total_crayons_eq]
  sorry

end beth_extra_crayons_l441_441094


namespace monomial_coefficient_and_degree_l441_441935

theorem monomial_coefficient_and_degree :
  ∀ (a b : ℝ), monomial_coefficient (-3 * a * b^4) = -3 ∧ monomial_degree (-3 * a * b^4) = 5 :=
by
  sorry

end monomial_coefficient_and_degree_l441_441935


namespace ati_pots_rearrangement_count_l441_441092

/-
Ati has 7 pots of flowers ordered as P_1, P_2, P_3, P_4, P_5, P_6, P_7. 
She wants to rearrange the positions of these pots to B_1, B_2, B_3, B_4, B_5, B_6, B_7 such that 
for every positive integer n < 7, B_1, B_2, ..., B_n is not the permutation of P_1, P_2, ..., P_7. 
-/

def P := list ℕ -- Representing pots P_1, P_2, ..., P_7 as a list of natural numbers

noncomputable def count_indec_permutations (n : ℕ) : ℕ := 
  if n = 0 then 1 else n! - ∑ i in range(n), count_indec_permutations i * (n - i)!

theorem ati_pots_rearrangement_count :
  count_indec_permutations 7 = 3447 :=
sorry

end ati_pots_rearrangement_count_l441_441092


namespace triangle_is_isosceles_l441_441857

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441857


namespace exists_polynomial_pairwise_rel_prime_l441_441236

theorem exists_polynomial_pairwise_rel_prime :
  ∃ f : ℤ[X], degree f = 2003 ∧
  (∀ n : ℤ, ∀ m k : ℕ, m ≠ k → is_coprime (f^[m] n) (f^[k] n)) :=
sorry

end exists_polynomial_pairwise_rel_prime_l441_441236


namespace largest_non_sum_217_l441_441400

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end largest_non_sum_217_l441_441400


namespace triangle_isosceles_if_parallel_l441_441861

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441861


namespace cyclic_quadrilateral_not_necessarily_have_diameter_diagonal_l441_441139

theorem cyclic_quadrilateral_not_necessarily_have_diameter_diagonal
    (A B C D : Point) (R : ℝ) 
    (circumcircle : Circle) (is_cyclic: IsInscribedInCircle (quad A B C D) circumcircle)
    (radius_circumcircle : circumcircle.radius = R)
    (sum_of_squares : (dist A B)^2 + (dist B C)^2 + (dist C D)^2 + (dist D A)^2 = 8 * R^2) :
    ¬ (∃ (E F : Point), IsDiameter (line_segment E F) circumcircle ∧ IsDiagonal (quad A B C D) (line_segment E F)) := 
sorry

end cyclic_quadrilateral_not_necessarily_have_diameter_diagonal_l441_441139


namespace cubic_polynomial_roots_l441_441673

theorem cubic_polynomial_roots:
  ∀ (a b c : ℝ),
  a + b + c = 0 → ab + ac + bc = -2 → abc = 2 → 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -18 :=
by {
  intros a b c h₁ h₂ h₃,
  sorry
} 

end cubic_polynomial_roots_l441_441673


namespace sum_of_squares_l441_441097

theorem sum_of_squares (a b c : ℝ) (h_arith : a + b + c = 30) (h_geom : a * b * c = 216) 
(h_harm : 1/a + 1/b + 1/c = 3/4) : a^2 + b^2 + c^2 = 576 := 
by 
  sorry

end sum_of_squares_l441_441097


namespace danny_initial_caps_l441_441516

-- Define the conditions
variables (lostCaps : ℕ) (currentCaps : ℕ)
-- Assume given conditions
axiom lost_caps_condition : lostCaps = 66
axiom current_caps_condition : currentCaps = 25

-- Define the total number of bottle caps Danny had at first
def originalCaps (lostCaps currentCaps : ℕ) : ℕ := lostCaps + currentCaps

-- State the theorem to prove the number of bottle caps Danny originally had is 91
theorem danny_initial_caps : originalCaps lostCaps currentCaps = 91 :=
by
  -- Insert the proof here when available
  sorry

end danny_initial_caps_l441_441516


namespace correct_calculation_l441_441435

theorem correct_calculation : (sqrt 0.36 = 0.6) :=
by
  sorry

end correct_calculation_l441_441435


namespace triangle_is_isosceles_l441_441897

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441897


namespace ratio_part_to_whole_number_l441_441707

theorem ratio_part_to_whole_number (P N : ℚ) 
  (h1 : (1 / 4) * (1 / 3) * P = 25) 
  (h2 : 0.40 * N = 300) : P / N = 2 / 5 :=
by
  sorry

end ratio_part_to_whole_number_l441_441707


namespace isosceles_triangle_of_parallel_l441_441881

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441881


namespace fill_time_for_taps_l441_441486

-- Definitions and conditions
def time_for_tap_B := 110
def time_for_tap_A := time_for_tap_B + 22
def combined_filling_time := 60

-- Proof statement
theorem fill_time_for_taps (x : ℝ) (hB : time_for_tap_B = x) (hA : time_for_tap_A = x + 22)
  (hCombined : 1 / time_for_tap_B + 1 / (time_for_tap_B + 22) = 1 / combined_filling_time) : 
  (time_for_tap_B = 110 ∧ time_for_tap_A = 132) :=
begin
  sorry
end

end fill_time_for_taps_l441_441486


namespace problem_1_problem_2_problem_3_l441_441175

-- Define the quadratic function
def f (a x : ℝ) := x^2 + 2*(a-1)*x + 2*a + 6

-- Proof 1: The function is monotonically increasing in [4, +∞) implies a ≥ -3
theorem problem_1 (a : ℝ) : (∀ x : ℝ, x ≥ 4 → deriv (f a x) ≥ 0) → a ≥ -3 := by
  sorry

-- Proof 2: The function is non-negative for all x ∈ ℝ implies -1 ≤ a ≤ 5
theorem problem_2 (a : ℝ) : (∀ x : ℝ, f a x ≥ 0) → -1 ≤ a ∧ a ≤ 5 := by
  sorry

-- Proof 3: The function has two distinct real roots greater than 1 implies -5/4 < a < -1
theorem problem_3 (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ > 1 ∧ f a x₁ = 0 ∧ f a x₂ = 0) → -5/4 < a ∧ a < -1 := by
  sorry

end problem_1_problem_2_problem_3_l441_441175


namespace isosceles_triangle_of_parallel_l441_441764

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441764


namespace max_value_inequality_l441_441548

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (abc (a + b + c)) / ((a + b)^2 * (b + c)^2) ≤ 1 / 4 :=
sorry

end max_value_inequality_l441_441548


namespace snow_white_last_trip_dwarfs_l441_441299

-- Definitions for the conditions
def original_lineup := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]
def only_snow_white_can_row := True
def boat_capacity_snow_white_and_dwarfs := 3
def dwarfs_quarrel_if_adjacent (d1 d2 : String) : Prop :=
  let index_d1 := List.indexOf original_lineup d1
  let index_d2 := List.indexOf original_lineup d2
  abs (index_d1 - index_d2) = 1

-- Theorem to prove the correct answer
theorem snow_white_last_trip_dwarfs :
  let last_trip_dwarfs := ["Grumpy", "Bashful", "Sneezy"]
  ∃ (trip : List String), trip = last_trip_dwarfs ∧ 
  ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → d1 ≠ d2 → ¬dwarfs_quarrel_if_adjacent d1 d2 :=
by
  sorry

end snow_white_last_trip_dwarfs_l441_441299


namespace cotangent_sum_inequality_l441_441361

-- Using noncomputable theory for geometric reasoning
noncomputable theory

-- Define the context and conditions
variables {α : Type*} [EuclideanGeometry α]

-- Let's introduce a triangle ABC
variables {A B C : α}

-- Assume the medians AM and BN intersect at the centroid G, and are perpendicular
variables {G M N : α}
variables {AM : Line α} {BN : Line α} 
variable (centroid_condition : Centroid G A B C)
variable (median_condition_A : LineThrough AM A G)
variable (median_condition_B : LineThrough BN B G)
variable (perpendicular_condition : Perpendicular AM BN)

-- We want to prove the inequality about cotangents
theorem cotangent_sum_inequality :
  cot (angle A B C) + cot (angle B A C) ≥ 2 / 3 :=
sorry

end cotangent_sum_inequality_l441_441361


namespace e3_is_quadratic_l441_441983

def is_quadratic (e : Expr) : Prop := 
  ∃ a b c : ℤ, e = a * x^2 + b * x + c = 0

noncomputable def e1 : Expr := x - 1 / (x - 1) = 0
noncomputable def e2 : Expr := 7 * x^2 + 1 / x^2 - 1 = 0
noncomputable def e3 : Expr := x^2 = 0
noncomputable def e4 : Expr := (x + 1) * (x - 2) = x * (x + 1)

theorem e3_is_quadratic : is_quadratic e3 :=
  sorry

end e3_is_quadratic_l441_441983


namespace triangle_isosceles_if_parallel_l441_441866

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441866


namespace largest_integer_not_sum_of_30_and_composite_l441_441420

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_not_sum_of_30_and_composite (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ b < 30 ∧ is_composite(b) ∧ n = 30 * a + b

theorem largest_integer_not_sum_of_30_and_composite :
  ∃ n : ℕ, is_not_sum_of_30_and_composite(n) ∧ ∀ m : ℕ, is_not_sum_of_30_and_composite(m) → m ≤ n :=
  ⟨93, sorry⟩

end largest_integer_not_sum_of_30_and_composite_l441_441420


namespace perfect_square_101_fact_102_squared_l441_441437

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- The theorem to prove
theorem perfect_square_101_fact_102_squared :
  is_perfect_square (factorial 101 * 102 ^ 2) :=
sorry

end perfect_square_101_fact_102_squared_l441_441437


namespace triangle_is_isosceles_if_parallel_l441_441734

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441734


namespace heavy_cream_cost_l441_441265

theorem heavy_cream_cost
  (cost_strawberries : ℕ)
  (cost_raspberries : ℕ)
  (total_cost : ℕ)
  (cost_heavy_cream : ℕ) :
  (cost_strawberries = 3 * 2) →
  (cost_raspberries = 5 * 2) →
  (total_cost = 20) →
  (cost_heavy_cream = total_cost - (cost_strawberries + cost_raspberries)) →
  cost_heavy_cream = 4 :=
by
  sorry

end heavy_cream_cost_l441_441265


namespace length_of_bridge_l441_441004

variable (length_of_train : ℕ)
variable (speed_of_train_kmh : ℕ)
variable (time_to_cross : ℕ)

theorem length_of_bridge (h1 : length_of_train = 156)
                         (h2 : speed_of_train_kmh = 45)
                         (h3 : time_to_cross = 30) :
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  let total_distance := speed_of_train_ms * time_to_cross in
  total_distance - length_of_train = 219 :=
by
  sorry

end length_of_bridge_l441_441004


namespace isosceles_triangle_l441_441755

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441755


namespace min_value_of_expr_min_value_achieved_final_statement_l441_441680

theorem min_value_of_expr (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  1 ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem min_value_achieved (x y z : ℝ) (h1 : x = 1) (h2 : y = 1) (h3 : z = 1) :
  1 = (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem final_statement (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  ∃ (x y z : ℝ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x + y + z = 3) ∧ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) = 1) :=
by
  sorry

end min_value_of_expr_min_value_achieved_final_statement_l441_441680


namespace tangent_parallel_l441_441583

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.cos x
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

theorem tangent_parallel (a : ℝ) (H : ∀ x1 : ℝ, ∃ x2 : ℝ, (a - 2 * Real.sin x1) = (-Real.exp x2 - 1)) :
  a < -3 := by
  sorry

end tangent_parallel_l441_441583


namespace coneCannotBeQuadrilateral_l441_441075

-- Define types for our geometric solids
inductive Solid
| Cylinder
| Cone
| FrustumCone
| Prism

-- Define a predicate for whether the cross-section can be a quadrilateral
def canBeQuadrilateral (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => true
  | Solid.Cone => false
  | Solid.FrustumCone => true
  | Solid.Prism => true

-- The theorem we need to prove
theorem coneCannotBeQuadrilateral : canBeQuadrilateral Solid.Cone = false := by
  sorry

end coneCannotBeQuadrilateral_l441_441075


namespace carl_table_paint_grid_l441_441272

theorem carl_table_paint_grid :
  ∃ (table : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ),
  (table = (2, 2, 1, 1, 2, 1, 1, 2)) ∧
  (table.1 + table.2 + table.3 + table.4 = 6) ∧
  (table.5 + table.6 + table.7 + table.8 = 6) :=
by {
  let table := (2, 2, 1, 1, 2, 1, 1, 2),
  existsi table,
  split,
  {
    refl,
  },
  split;
  {
    rfl,
  },
  sorry -- Proof omitted for brevity
}

end carl_table_paint_grid_l441_441272


namespace balloons_blue_l441_441077

variable (total_balloons : Nat) (red_fraction green_fraction purple_fraction : ℝ)

theorem balloons_blue (h1 : total_balloons = 200)
                      (h2 : red_fraction = 0.35)
                      (h3 : green_fraction = 0.25)
                      (h4 : purple_fraction = 0.15) :
    (total_balloons - ((red_fraction * total_balloons).toNat +
                        (green_fraction * total_balloons).toNat +
                        (purple_fraction * total_balloons).toNat)) = 50 :=
by
    sorry

end balloons_blue_l441_441077


namespace general_term_formula_l441_441356

def seq (n : ℕ) : ℝ := n + 1/(2^n)

theorem general_term_formula :
  ∀ n : ℕ, seq n = n + 1/(2^n) := 
by
  sorry

end general_term_formula_l441_441356


namespace betty_passes_alice_l441_441974

/-- Define constants and initial conditions -/
def circumference : ℝ := 400
def speed_Alice : ℝ := 10 -- in meters per second
def speed_Betty : ℝ := 15 -- in meters per second
def initial_gap : ℝ := 40 -- in meters
def total_time : ℝ := 15 * 60 -- 15 minutes in seconds

/-- Prove that after 15 minutes, Betty will have passed Alice 11 times -/
theorem betty_passes_alice :
  let relative_speed := speed_Betty - speed_Alice in
  let time_to_catch := initial_gap / relative_speed in
  let total_distance_betty := speed_Betty * total_time in
  let total_distance_alice := speed_Alice * total_time in
  let laps_betty := total_distance_betty / circumference in
  let laps_alice := total_distance_alice / circumference in
  ⌊laps_betty - laps_alice⌋ = 11 :=
by
  sorry

end betty_passes_alice_l441_441974


namespace parallelogram_to_rectangle_incircles_parallelogram_to_rectangle_circumcircles_l441_441565

-- Part (a)
theorem parallelogram_to_rectangle_incircles (A B C D : Point)
  (h1 : Parallelogram A B C D)
  (h2 : Inradius (Triangle ABC) = Inradius (Triangle ABD)) : Rectangle A B C D :=
sorry

-- Part (b)
theorem parallelogram_to_rectangle_circumcircles (A B C D : Point)
  (h1 : Parallelogram A B C D)
  (h2 : Circumradius (Triangle ABC) = Circumradius (Triangle ABD)) : Rectangle A B C D :=
sorry

end parallelogram_to_rectangle_incircles_parallelogram_to_rectangle_circumcircles_l441_441565


namespace triangle_psr_area_l441_441650

theorem triangle_psr_area {
  P Q R S : Point ℝ,
  angle_PQR_90 : angle P Q R = 90,
  PS_bisect : angle_bisector P S Q R,
  PQ_80 : distance P Q = 80,
  QR_y : ∃ y : ℝ, distance Q R = y,
  PR_2y_minus_10 : ∃ y : ℝ, distance P R = 2 * y - 10
  } : 
  ∃ A : ℝ, abs (A - 2061) < 1 :=
sorry

end triangle_psr_area_l441_441650


namespace wool_production_equivalence_l441_441929

variable (x y z w v : ℕ)

def wool_per_sheep_of_breed_A_per_day : ℚ :=
  (y:ℚ) / ((x:ℚ) * (z:ℚ))

def wool_per_sheep_of_breed_B_per_day : ℚ :=
  2 * wool_per_sheep_of_breed_A_per_day x y z

def total_wool_produced_by_breed_B (x y z w v: ℕ) : ℚ :=
  (w:ℚ) * wool_per_sheep_of_breed_B_per_day x y z * (v:ℚ)

theorem wool_production_equivalence :
  total_wool_produced_by_breed_B x y z w v = 2 * (y:ℚ) * (w:ℚ) * (v:ℚ) / ((x:ℚ) * (z:ℚ)) := by
  sorry

end wool_production_equivalence_l441_441929


namespace ellipse_equation_find_m_and_min_AB_l441_441154

-- Define the ellipse properties
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

-- Define the problem given the conditions
theorem ellipse_equation :
  (∃ a b, a > b ∧ b > 0 ∧ 
    ellipse a b 1 (sqrt 3 / 2) ∧ 
    -- The minor axis endpoints and the right focus form an equilateral triangle.
    -- This is given as a condition we can encapsulate:
    let c := sqrt (a^2 - b^2) in 
    let endpoints := [(a, 0), (-a, 0)] in
    let right_focus := (c, 0) in
    ∃ m, m > 0 ∧ 
    ∀ A B : (ℝ × ℝ), (∃ θ : ℝ, θ ∈ [0, 2*π] ∧ A = (2*cos θ, sin θ)) ∧ 
    B = (0, m) ∧ 
    (angle (A.1, A.2) (0, 0) (B.1, B.2) = π/2) ∧ 
    distance (0, 0) B / distance (A, B) = constant_distance :=
    (∃ a b, a = 2 ∧ b = 1 ∧ Eq (x y) (0, a))) sorry

-- Define the theorem about the distance being constant and finding m
theorem find_m_and_min_AB :
  (∃ O B l m, m > 0 ∧ l = (λ x y, y = m) ∧ O = (0, 0) ∧ (distance O l = constant)) →
  (m = 2 * sqrt 3 / 3 ∧ min_ab = 2) sorry

end ellipse_equation_find_m_and_min_AB_l441_441154


namespace min_edges_for_k5_subgraph_l441_441081

/-- 
An undirected graph with 10 vertices and m edges. 
Prove that the minimum value of m such that there exists 
a vertex-induced subgraph where all vertices have 
degree at least 5 is 31.
-/
theorem min_edges_for_k5_subgraph (G : SimpleGraph (Fin 10))
    (h : G.edge_count = m) : 
    ∃ (H : SimpleGraph (Fin 10)), H.induced G  ∧ 
    (∀ v : (Fin 10), H.degree v ≥ 5) → m ≥ 31 :=
sorry

end min_edges_for_k5_subgraph_l441_441081


namespace final_trip_theorem_l441_441291

/-- Define the lineup of dwarfs -/
inductive Dwarf where
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

open Dwarf

/-- Define the conditions -/
-- The dwarfs initially standing next to each other
def adjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = Happy ∧ d2 = Grumpy) ∨
  (d1 = Grumpy ∧ d2 = Dopey) ∨
  (d1 = Dopey ∧ d2 = Bashful) ∨
  (d1 = Bashful ∧ d2 = Sleepy) ∨
  (d1 = Sleepy ∧ d2 = Doc) ∨
  (d1 = Doc ∧ d2 = Sneezy)

-- Snow White is the only one who can row
constant snowWhite_can_row : Prop := true

-- The boat can hold Snow White and up to 3 dwarfs
constant boat_capacity : ℕ := 4

-- Define quarrel if left without Snow White
def quarrel_without_snowwhite (d1 d2 : Dwarf) : Prop := adjacent d1 d2

-- Define the final trip setup
def final_trip (dwarfs : List Dwarf) : Prop :=
  dwarfs = [Grumpy, Bashful, Doc]

-- Theorem to prove the final trip
theorem final_trip_theorem : ∃ dwarfs, final_trip dwarfs :=
  sorry

end final_trip_theorem_l441_441291


namespace addition_in_M_l441_441141

def M (α : ℝ) : set (ℝ → ℝ) :=
  {f | ∀ x1 x2 : ℝ, x2 > x1 → -α * (x2 - x1) < f x2 - f x1 ∧ 
                                      f x2 - f x1 < α * (x2 - x1)}

theorem addition_in_M (α1 α2 : ℝ) (f g : ℝ → ℝ) (hf : f ∈ M α1) (hg : g ∈ M α2) :
  (λ x, f x + g x) ∈ M (α1 + α2) :=
sorry

end addition_in_M_l441_441141


namespace bryan_total_earnings_l441_441499

-- Declare the data given in the problem:
def num_emeralds : ℕ := 3
def num_rubies : ℕ := 2
def num_sapphires : ℕ := 3

def price_emerald : ℝ := 1785
def price_ruby : ℝ := 2650
def price_sapphire : ℝ := 2300

-- Calculate the total earnings from each type of stone:
def total_emeralds : ℝ := num_emeralds * price_emerald
def total_rubies : ℝ := num_rubies * price_ruby
def total_sapphires : ℝ := num_sapphires * price_sapphire

-- Calculate the overall total earnings:
def total_earnings : ℝ := total_emeralds + total_rubies + total_sapphires

-- Prove that Bryan got 17555 dollars in total:
theorem bryan_total_earnings : total_earnings = 17555 := by
  simp [total_earnings, total_emeralds, total_rubies, total_sapphires, num_emeralds, num_rubies, num_sapphires, price_emerald, price_ruby, price_sapphire]
  sorry

end bryan_total_earnings_l441_441499


namespace complex_power_twenty_l441_441534

noncomputable def polar_form (z : ℂ) : ℂ :=
  complex.norm z * complex.exp (complex.arg z * complex.I)

def complex_expr : ℂ := (1 - complex.I) / real.sqrt 2

theorem complex_power_twenty :
  complex_expr ^ 20 = 1 :=
by
  sorry

end complex_power_twenty_l441_441534


namespace find_third_side_l441_441638

theorem find_third_side (n : ℤ) 
  (h1 : real.of_int n > 2.47)
  (h2 : real.of_int n < 3.81) : 
  n = 3 := by
  sorry

end find_third_side_l441_441638


namespace jim_miles_driven_l441_441664

theorem jim_miles_driven (total_journey : ℕ) (miles_needed : ℕ) (h : total_journey = 1200 ∧ miles_needed = 985) : total_journey - miles_needed = 215 := 
by sorry

end jim_miles_driven_l441_441664


namespace largest_integer_not_sum_of_30_and_composite_l441_441402

theorem largest_integer_not_sum_of_30_and_composite : 
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b < 30 ∧ ¬ prime b ∧ (n = 30 * a + b) → n = 157 :=
by
  sorry

end largest_integer_not_sum_of_30_and_composite_l441_441402


namespace triangle_is_isosceles_l441_441895

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441895


namespace isosceles_triangle_of_parallel_l441_441880

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441880


namespace find_exponent_l441_441621

theorem find_exponent (n : ℕ) (some_number : ℕ) (h1 : n = 27) 
  (h2 : 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) = 4 ^ some_number) :
  some_number = 28 :=
by 
  sorry

end find_exponent_l441_441621


namespace probability_correct_l441_441037

def ball_numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 13}

def odd_ball_numbers : finset ℕ := {1, 3, 5, 7, 9, 11, 13}
def even_ball_numbers : finset ℕ := {2, 4, 6, 8, 10, 12}

noncomputable def probability_sum_is_odd : ℚ := 
  (finset.choose 7 5 * finset.choose 6 2 + finset.choose 7 3 * finset.choose 6 4 + finset.choose 7 1 * finset.choose 6 6) / finset.choose 13 7

theorem probability_correct :
  probability_sum_is_odd = 847 / 1716 :=
  sorry

end probability_correct_l441_441037


namespace find_ticket_price_l441_441971

theorem find_ticket_price
  (P : ℝ) -- The original price of each ticket
  (h1 : 10 * 0.6 * P + 20 * 0.85 * P + 26 * P = 980) :
  P = 20 :=
sorry

end find_ticket_price_l441_441971


namespace exponential_function_is_one_half_inv_x_l441_441358

-- Define the conditions
def passesThrough (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

def inverseFunctionCondition (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, f (g x) = x) ∧ (∀ y, g (f y) = y) ∧ passesThrough g (2, -1)

-- The proof problem statement
theorem exponential_function_is_one_half_inv_x (f : ℝ → ℝ)
  (exponential : ∃ a : ℝ, ∀ x, f x = a^x)
  (inv_cond : inverseFunctionCondition f) :
  f = λ x, (1/2)^x :=
by
  sorry

end exponential_function_is_one_half_inv_x_l441_441358


namespace largest_integer_not_sum_of_30_and_composite_l441_441404

theorem largest_integer_not_sum_of_30_and_composite : 
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b < 30 ∧ ¬ prime b ∧ (n = 30 * a + b) → n = 157 :=
by
  sorry

end largest_integer_not_sum_of_30_and_composite_l441_441404


namespace find_first_number_in_sequence_l441_441061

theorem find_first_number_in_sequence :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℚ),
    (a3 = a2 * a1) ∧ 
    (a4 = a3 * a2) ∧ 
    (a5 = a4 * a3) ∧ 
    (a6 = a5 * a4) ∧ 
    (a7 = a6 * a5) ∧ 
    (a8 = a7 * a6) ∧ 
    (a9 = a8 * a7) ∧ 
    (a10 = a9 * a8) ∧ 
    (a8 = 36) ∧ 
    (a9 = 324) ∧ 
    (a10 = 11664) ∧ 
    (a1 = 59049 / 65536) := 
sorry

end find_first_number_in_sequence_l441_441061


namespace find_reciprocal_square_sum_of_roots_l441_441353

theorem find_reciprocal_square_sum_of_roots :
  ∃ (a b c : ℝ), 
    (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (a^3 - 6 * a^2 - a + 3 = 0) ∧ 
    (b^3 - 6 * b^2 - b + 3 = 0) ∧ 
    (c^3 - 6 * c^2 - c + 3 = 0) ∧ 
    (a + b + c = 6) ∧
    (a * b + b * c + c * a = -1) ∧
    (a * b * c = -3)) 
    → (1 / a^2 + 1 / b^2 + 1 / c^2 = 37 / 9) :=
sorry

end find_reciprocal_square_sum_of_roots_l441_441353


namespace isosceles_triangle_l441_441749

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441749


namespace triangle_is_isosceles_l441_441858

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441858


namespace a2014_is_zero_l441_441701

def periodic_sequence (a : ℕ → ℤ) (p : ℕ) :=
  ∀ n : ℕ, a (n + p) = a n

def sequence : ℕ → ℤ
| 0       := -4
| 1       := 0
| 2       := 4
| 3       := 1
| (n + 4) := sequence n

lemma sequence_periodic : periodic_sequence sequence 4 :=
by
  intro n
  induction n
  case zero => rfl
  case succ n ih => rfl

theorem a2014_is_zero : sequence 2014 = 0 :=
by
  rw [← nat.mod_add_div 2014 4]
  have h : 2014 % 4 = 2 := by norm_num [nat.mod_eq_of_lt, nat.div_eq_of_lt]
  rw [h]
  rfl

end a2014_is_zero_l441_441701


namespace isosceles_triangle_l441_441756

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441756


namespace symmetric_theta_l441_441614

theorem symmetric_theta:
  (P Q : ℝ × ℝ) (θ : ℝ) 
  (hP : P = (Real.cos θ, Real.sin θ))
  (hQ : Q = (Real.cos (θ + π / 6), Real.sin (θ + π / 6)))
  (h_symmetry: P.1 = -Q.1 ∧ P.2 = Q.2) :
  θ = 5 * π / 12 :=
by
  sorry

end symmetric_theta_l441_441614


namespace auditorium_rows_l441_441968

theorem auditorium_rows (x : ℕ) (hx : (320 / x + 4) * (x + 1) = 420) : x = 20 :=
by
  sorry

end auditorium_rows_l441_441968


namespace original_number_of_workers_l441_441015

theorem original_number_of_workers (W A : ℕ)
  (h1 : W * 75 = A)
  (h2 : (W + 10) * 65 = A) :
  W = 65 :=
by
  sorry

end original_number_of_workers_l441_441015


namespace abel_overtake_kelly_chris_overtake_both_l441_441209

-- Given conditions and variables
variable (d : ℝ)  -- distance at which Abel overtakes Kelly
variable (d_c : ℝ)  -- distance at which Chris overtakes both Kelly and Abel
variable (t_k : ℝ)  -- time taken by Kelly to run d meters
variable (t_a : ℝ)  -- time taken by Abel to run (d + 3) meters
variable (t_c : ℝ)  -- time taken by Chris to run the required distance
variable (k_speed : ℝ := 9)  -- Kelly's speed
variable (a_speed : ℝ := 9.5)  -- Abel's speed
variable (c_speed : ℝ := 10)  -- Chris's speed
variable (head_start_k : ℝ := 3)  -- Kelly's head start over Abel
variable (head_start_c : ℝ := 2)  -- Chris's head start behind Abel
variable (lost_by : ℝ := 0.75)  -- Abel lost by distance

-- Proof problem for Abel overtaking Kelly
theorem abel_overtake_kelly 
  (hk : t_k = d / k_speed) 
  (ha : t_a = (d + head_start_k) / a_speed) 
  (h_lost : lost_by = 0.75):
  d + lost_by = 54.75 := 
sorry

-- Proof problem for Chris overtaking both Kelly and Abel
theorem chris_overtake_both 
  (hc : t_c = (d_c + 5) / c_speed)
  (h_56 : d_c = 56):
  d_c = c_speed * (56 / c_speed) :=
sorry

end abel_overtake_kelly_chris_overtake_both_l441_441209


namespace path_conditions_l441_441121

-- Definitions based on the problem conditions
def diameter (A B : Point) := dist A B = 16
def is_on_diameter (C D A B : Point) := dist A C = 6 ∧ dist B D = 6
def is_on_opposite_semicircle (P : Point) := P ∈ (semicircle (O, A, B) \ (arc A B))

-- Proof goals based on the correct answers
theorem path_conditions
  (A B C D P : Point)
  (O : Point)
  (h1 : diameter A B)
  (h2 : is_on_diameter C D A B)
  (h3 : is_on_opposite_semicircle P) :
  (P = A ∨ P = B → dist C P + dist P D = 16) ∧
  (P = midpoint (semicircle (O, A, B) \ (arc A B)) → dist C P + dist P D = 2 * sqrt 68) ∧ 
  (dist C P + dist P D < 16 → false) ∧
  (dist C P + dist P D > 16) ∧
  (dist C P + dist P D = 16 ↔ P = A ∨ P = B)
:= sorry

end path_conditions_l441_441121


namespace relationship_between_a_b_c_l441_441147

theorem relationship_between_a_b_c (a b c : ℝ) (h1 : 2^a = 5) (h2 : 2^b = 8) (h3 : 2^c = 20) : 
  a + b - c = 1 := 
sorry

end relationship_between_a_b_c_l441_441147


namespace problem_statement_l441_441156

-- Define vectors a and b
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (k : ℝ) : ℝ × ℝ := (2, k)

-- Define proposition p
def prop_p (k : ℝ) : Prop :=
  let acute := (vector_a.1 * vector_b(k).1 + vector_a.2 * vector_b(k).2 > 0)
  in acute ∧ k ≠ 4 ↔ k > -1

-- Define function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then
    Real.sin (x + (Real.pi / 3))
  else
    Real.cos (x + (Real.pi / 6))

-- Define proposition q
def prop_q : Prop :=
  ∀ x : ℝ, f(-x) = f(x)

-- Define the compound proposition
def true_proposition (k : ℝ) : Prop :=
  ¬ prop_p k ∧ prop_q

-- State the theorem
theorem problem_statement (k : ℝ) : true_proposition k :=
sorry

end problem_statement_l441_441156


namespace snow_white_last_trip_l441_441324

-- Definitions based on the problem's conditions
inductive Dwarf
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

def is_adjacent (d1 d2 : Dwarf) : Prop :=
  (d1, d2) ∈ [
    (Dwarf.Happy, Dwarf.Grumpy),
    (Dwarf.Grumpy, Dwarf.Happy),
    (Dwarf.Grumpy, Dwarf.Dopey),
    (Dwarf.Dopey, Dwarf.Grumpy),
    (Dwarf.Dopey, Dwarf.Bashful),
    (Dwarf.Bashful, Dwarf.Dopey),
    (Dwarf.Bashful, Dwarf.Sleepy),
    (Dwarf.Sleepy, Dwarf.Bashful),
    (Dwarf.Sleepy, Dwarf.Doc),
    (Dwarf.Doc, Dwarf.Sleepy),
    (Dwarf.Doc, Dwarf.Sneezy),
    (Dwarf.Sneezy, Dwarf.Doc)
  ]

def boat_capacity : ℕ := 3

variable (snowWhite : Prop)

-- The theorem to prove that the dwarfs Snow White will take in the last trip are Grumpy, Bashful and Doc
theorem snow_white_last_trip 
  (h1 : snowWhite)
  (h2 : boat_capacity = 3)
  (h3 : ∀ d1 d2, is_adjacent d1 d2 → snowWhite)
  : (snowWhite ∧ (Dwarf.Grumpy ∧ Dwarf.Bashful ∧ Dwarf.Doc)) :=
sorry

end snow_white_last_trip_l441_441324


namespace frenchwoman_present_l441_441090

theorem frenchwoman_present
    (M_F M_R W_R : ℝ)
    (condition_1 : M_F > M_R + W_R)
    (condition_2 : W_R > M_F + M_R) 
    : false :=
by
  -- We would assume the opposite of what we know to lead to a contradiction here.
  -- This is a placeholder to indicate the proof should lead to a contradiction.
  sorry

end frenchwoman_present_l441_441090


namespace triangle_is_isosceles_if_parallel_l441_441721

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441721


namespace percentage_increase_bears_with_assistant_l441_441654

theorem percentage_increase_bears_with_assistant
  (B H : ℝ)
  (h_positive_hours : H > 0)
  (h_positive_bears : B > 0)
  (hours_with_assistant : ℝ := 0.90 * H)
  (rate_increase : ℝ := 2 * B / H) :
  ((rate_increase * hours_with_assistant) - B) / B * 100 = 80 := by
  -- This is the statement for the given problem.
  sorry

end percentage_increase_bears_with_assistant_l441_441654


namespace integral_equals_pi_div_two_l441_441531

open Real

noncomputable def integral_value : ℝ := 
  ∫ x in -1..1, (sqrt (1 - x^2) - x)

theorem integral_equals_pi_div_two : integral_value = π / 2 :=
  sorry

end integral_equals_pi_div_two_l441_441531


namespace unique_triple_l441_441257

theorem unique_triple (a b c : ℕ) (ha : a > 0) (hc : c > 0) (hb : b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ((a + b / 9)^2 = c + 7 / 9 ∧ (c + a) / (c - a) ∈ ℤ) → (a = 1 ∧ b = 6 ∧ c = 2) := by
  sorry

end unique_triple_l441_441257


namespace y_coordinate_sum_of_circle_on_y_axis_l441_441099

-- Define the properties of the circle
def center := (-3, 1)
def radius := 8

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y - 1) ^ 2 = 64

-- Define the Lean theorem statement
theorem y_coordinate_sum_of_circle_on_y_axis 
  (h₁ : center = (-3, 1)) 
  (h₂ : radius = 8) 
  (h₃ : ∀ y : ℝ, circle_eq 0 y → (∃ y1 y2 : ℝ, y = y1 ∨ y = y2) ) : 
  ∃ y1 y2 : ℝ, (y1 + y2 = 2) ∧ (circle_eq 0 y1) ∧ (circle_eq 0 y2) := 
by 
  sorry

end y_coordinate_sum_of_circle_on_y_axis_l441_441099


namespace isosceles_triangle_of_parallel_l441_441882

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441882


namespace last_trip_l441_441313

def initial_order : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

def boatCapacity : Nat := 4  -- Including Snow White

def adjacentPairsQuarrel (adjPairs : List (String × String)) : Prop :=
  ∀ (d1 d2 : String), (d1, d2) ∈ adjPairs → (d2, d1) ∈ adjPairs → False

def canRow (person : String) : Prop := person = "Snow White"

noncomputable def final_trip (remainingDwarfs : List String) (allTrips : List (List String)) : List String := ["Grumpy", "Bashful", "Doc"]

theorem last_trip (adjPairs : List (String × String))
  (h_adj : adjacentPairsQuarrel adjPairs)
  (h_row : canRow "Snow White")
  (dwarfs_order : List String = initial_order)
  (without_quarrels : ∀ trip : List String, trip ∈ allTrips → 
    ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → (d1, d2) ∈ adjPairs → 
    ("Snow White" ∈ trip) → True) :
  final_trip ["Grumpy", "Bashful", "Doc"] allTrips = ["Grumpy", "Bashful", "Doc"] :=
sorry

end last_trip_l441_441313


namespace snow_white_last_trip_l441_441321

-- Definitions based on the problem's conditions
inductive Dwarf
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

def is_adjacent (d1 d2 : Dwarf) : Prop :=
  (d1, d2) ∈ [
    (Dwarf.Happy, Dwarf.Grumpy),
    (Dwarf.Grumpy, Dwarf.Happy),
    (Dwarf.Grumpy, Dwarf.Dopey),
    (Dwarf.Dopey, Dwarf.Grumpy),
    (Dwarf.Dopey, Dwarf.Bashful),
    (Dwarf.Bashful, Dwarf.Dopey),
    (Dwarf.Bashful, Dwarf.Sleepy),
    (Dwarf.Sleepy, Dwarf.Bashful),
    (Dwarf.Sleepy, Dwarf.Doc),
    (Dwarf.Doc, Dwarf.Sleepy),
    (Dwarf.Doc, Dwarf.Sneezy),
    (Dwarf.Sneezy, Dwarf.Doc)
  ]

def boat_capacity : ℕ := 3

variable (snowWhite : Prop)

-- The theorem to prove that the dwarfs Snow White will take in the last trip are Grumpy, Bashful and Doc
theorem snow_white_last_trip 
  (h1 : snowWhite)
  (h2 : boat_capacity = 3)
  (h3 : ∀ d1 d2, is_adjacent d1 d2 → snowWhite)
  : (snowWhite ∧ (Dwarf.Grumpy ∧ Dwarf.Bashful ∧ Dwarf.Doc)) :=
sorry

end snow_white_last_trip_l441_441321


namespace isosceles_triangle_l441_441753

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441753


namespace inequality_problem_l441_441683

noncomputable def nonneg_real := {x : ℝ // 0 ≤ x}

theorem inequality_problem (x y z : nonneg_real) (h : x.val * y.val + y.val * z.val + z.val * x.val = 1) :
  1 / (x.val + y.val) + 1 / (y.val + z.val) + 1 / (z.val + x.val) ≥ 5 / 2 :=
sorry

end inequality_problem_l441_441683


namespace paving_stone_width_l441_441042

theorem paving_stone_width
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_length : ℝ)
  (courtyard_area : ℝ) (paving_stone_area : ℝ)
  (width : ℝ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16.5)
  (h3 : num_paving_stones = 99)
  (h4 : paving_stone_length = 2.5)
  (h5 : courtyard_area = courtyard_length * courtyard_width)
  (h6 : courtyard_area = 495)
  (h7 : paving_stone_area = courtyard_area / num_paving_stones)
  (h8 : paving_stone_area = 5)
  (h9 : paving_stone_area = paving_stone_length * width) :
  width = 2 := by
  sorry

end paving_stone_width_l441_441042


namespace CarlsPlaygroundArea_l441_441504

/-- Carl's playground problem as described:
Carl bought 24 fence posts and placed them evenly every 3 yards along the edges of a rectangular playground.
The longer side of the playground has twice the number of posts as the shorter side.
Prove that the area of the playground is 324 square yards. 
-/
theorem CarlsPlaygroundArea : 
  let posts := 24 in
  let distance_between_posts := 3 in
  let longer_side_ratio := 2 in
  ∃ (short_side long_side : ℕ), 
    short_side ≠ 0 ∧ long_side ≠ 0 ∧ 
    short_side + 2 * long_side - 4 = posts ∧ 
    (longer_side_ratio * (short_side - 1) * distance_between_posts) 
    * ((short_side - 1) * distance_between_posts) = 324 :=
by
  sorry

end CarlsPlaygroundArea_l441_441504


namespace initial_students_began_contest_l441_441211

theorem initial_students_began_contest
  (n : ℕ)
  (first_round_fraction : ℚ)
  (second_round_fraction : ℚ)
  (remaining_students : ℕ) :
  first_round_fraction * second_round_fraction * n = remaining_students →
  remaining_students = 18 →
  first_round_fraction = 0.3 →
  second_round_fraction = 0.5 →
  n = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_students_began_contest_l441_441211


namespace calc_xy_square_l441_441096

theorem calc_xy_square
  (x y z : ℝ)
  (h1 : 2 * x * (y + z) = 1 + y * z)
  (h2 : 1 / x - 2 / y = 3 / 2)
  (h3 : x + y + 1 / 2 = 0) :
  (x + y + z) ^ 2 = 1 :=
by
  sorry

end calc_xy_square_l441_441096


namespace triangle_is_isosceles_if_parallel_l441_441727

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441727


namespace tan_sec_expression_l441_441623

variable (θ : ℝ) (b : ℝ)

-- Conditions: θ is an acute angle and sin(2θ) = b
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

axiom sin_double_angle : sin (2 * θ) = b

theorem tan_sec_expression
  (h_acute : is_acute_angle θ)
  (h_sin : sin_double_angle θ b) :
  tan θ - sec θ = (sin θ - 1) / cos θ :=
sorry

end tan_sec_expression_l441_441623


namespace time_to_reach_ship_l441_441016

def rate_of_descent : ℕ := 35
def depth : ℕ := 3500

theorem time_to_reach_ship : depth / rate_of_descent = 100 := by
  exact Nat.div_eq_of_eq_mul (by norm_num)
  sorry

end time_to_reach_ship_l441_441016


namespace perfect_square_trinomial_m_l441_441191

theorem perfect_square_trinomial_m (m : ℝ) (h : ∃ b : ℝ, (x : ℝ) ↦ (x + b)^2 = x^2 + mx + 16) : m = 8 ∨ m = -8 :=
sorry

end perfect_square_trinomial_m_l441_441191


namespace eq_max_distance_line_l441_441033

def point := (ℝ × ℝ)
def equation_of_line (m : ℝ) (p : point) : Prop := ∃ b : ℝ, (λ (x y : ℝ), y = m * x + b) p.1 p.2

def max_distance_line (A : point) (origin : point) (eqn : Prop) : Prop :=
  let m := -1 / (A.2 / A.1) in
  equation_of_line m A ∧ eqn = (λ (x y : ℝ), x + 2 * y - 5 = 0)

theorem eq_max_distance_line :
  ∃ L : Prop, max_distance_line (1, 2) (0, 0) L :=
by {
  sorry
}

end eq_max_distance_line_l441_441033


namespace triangle_ABC_properties_l441_441633

-- Definitions based on conditions
variables (A B C a b c : ℝ)
variable h1 : (b : ℝ)

-- Given in the problem statement
variable h2 : ∀ (x y : ℝ), (x * (sin A - sin B) + y * sin B = c * sin C) :=
  sorry
variable h3 : ∀ (a b : ℝ), (a^2 + b^2 - 6*(a + b) + 18 = 0) := sorry

-- The proof statement
theorem triangle_ABC_properties :
  C = π / 3 ∧ (a = 3 ∧ b = 3 → (1 / 2) * a * b * sin (π / 3) = (9 * sqrt 3) / 4) :=
by
  sorry

end triangle_ABC_properties_l441_441633


namespace isosceles_triangle_l441_441796

theorem isosceles_triangle (A B C P Q : Point) (h_parallel : parallel PQ AC) : is_isosceles (Triangle A B C) :=
sorry

end isosceles_triangle_l441_441796


namespace fraction_arithmetic_l441_441006

theorem fraction_arithmetic :
  (3 / 4) / (5 / 8) + (1 / 8) = 53 / 40 :=
by
  sorry

end fraction_arithmetic_l441_441006


namespace first_sculpture_weight_is_five_l441_441238

variable (w x y z : ℝ)

def hourly_wage_exterminator := 70
def daily_hours := 20
def price_per_pound := 20
def second_sculpture_weight := 7
def total_income := 1640

def income_exterminator := daily_hours * hourly_wage_exterminator
def income_sculptures := total_income - income_exterminator
def income_second_sculpture := second_sculpture_weight * price_per_pound
def income_first_sculpture := income_sculptures - income_second_sculpture

def weight_first_sculpture := income_first_sculpture / price_per_pound

theorem first_sculpture_weight_is_five :
  weight_first_sculpture = 5 := sorry

end first_sculpture_weight_is_five_l441_441238


namespace isosceles_triangle_of_parallel_l441_441769

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441769


namespace prime_implies_power_of_two_l441_441924

-- Conditions:
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Problem:
theorem prime_implies_power_of_two (n : ℕ) (h : is_prime (2^n + 1)) : ∃ k : ℕ, n = 2^k := sorry

end prime_implies_power_of_two_l441_441924


namespace triangle_ABC_is_isosceles_l441_441910

-- Define the geometric setting and properties
variables {A B C P Q : Type}
-- We assume that P, Q, A, B, and C are points.

-- Define the parallel condition PQ parallel to AC
axiom PQ_parallel_AC : ∥ PQ ∥ ∥ AC 

-- The theorem to prove that triangle ABC is isosceles
theorem triangle_ABC_is_isosceles (h : PQ_parallel_AC) : is_isosceles_triangle A B C := 
sorry

end triangle_ABC_is_isosceles_l441_441910


namespace calc1_calc2_calc3_calc4_calc5_calc6_l441_441710

theorem calc1 : 320 + 16 * 27 = 752 :=
by
  -- Proof goes here
  sorry

theorem calc2 : 1500 - 125 * 8 = 500 :=
by
  -- Proof goes here
  sorry

theorem calc3 : 22 * 22 - 84 = 400 :=
by
  -- Proof goes here
  sorry

theorem calc4 : 25 * 8 * 9 = 1800 :=
by
  -- Proof goes here
  sorry

theorem calc5 : (25 + 38) * 15 = 945 :=
by
  -- Proof goes here
  sorry

theorem calc6 : (62 + 12) * 38 = 2812 :=
by
  -- Proof goes here
  sorry

end calc1_calc2_calc3_calc4_calc5_calc6_l441_441710


namespace mutually_exclusive_event_b_l441_441552

-- Define the bag and ball drawing conditions
inductive BallColor
| red
| white

def bag : List BallColor := [BallColor.red, BallColor.red, BallColor.white, BallColor.white]

def draw_two_balls (bag : List BallColor) : List (BallColor × BallColor) :=
  [(x, y) | x <- bag, y <- bag.erase x] -- Draw two distinct balls from the bag

-- Define events
def at_least_one_white (draw : BallColor × BallColor) : Prop :=
  draw.fst = BallColor.white ∨ draw.snd = BallColor.white

def both_red (draw : BallColor × BallColor) : Prop :=
  draw.fst = BallColor.red ∧ draw.snd = BallColor.red

-- Theorem stating that the two events are mutually exclusive
theorem mutually_exclusive_event_b :
  ∀ (draw : BallColor × BallColor), at_least_one_white draw → ¬ both_red draw :=
by
  intros draw h
  cases h
  case or.inl h_left =>
    unfold both_red
    rw [h_left]
    unfold not
    intros h_and
    cases h_and
    contradiction
  case or.inr h_right =>
    unfold both_red
    rw [h_right]
    unfold not
    intros h_and
    cases h_and
    contradiction

end mutually_exclusive_event_b_l441_441552


namespace largest_non_sum_217_l441_441397

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end largest_non_sum_217_l441_441397


namespace isosceles_triangle_l441_441826

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441826


namespace increasing_function_range_l441_441172

def f (a : ℝ) (x : ℝ) : ℝ :=
if x >= 1 then Real.log x / Real.log a else (2 - a) * x - a / 2

theorem increasing_function_range (a : ℝ) :
  (∀ x1 x2, x1 < x2 → f a x1 ≤ f a x2) ↔ (4 / 3 ≤ a ∧ a < 2) := by
  sorry

end increasing_function_range_l441_441172


namespace number_of_zeros_of_g_is_zero_l441_441161

noncomputable def f (x : ℝ) : ℝ := sorry

theorem number_of_zeros_of_g_is_zero (h1 : ∀ x, Continuous (f x))
    (h2 : ∀ x, Differentiable ℝ (f x))
    (h3 : ∀ x : ℝ, 0 < x → x * (deriv (deriv f)) x + f x > 0) :
    ∀ x : ℝ, 0 < x → x * f x + 1 ≠ 0 :=
begin
    intros x hx,
    sorry
end

end number_of_zeros_of_g_is_zero_l441_441161


namespace isosceles_triangle_l441_441819

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441819


namespace inequality_proof_l441_441580

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := by
  sorry

end inequality_proof_l441_441580


namespace X_is_fixed_as_Γ_varies_l441_441255

variables {A B C T S X : Point}
variables {Γ : Circle}

-- Conditions and assumptions
def collinear (A B C : Point) : Prop := ∃ (l : Line), A ∈ l ∧ B ∈ l ∧ C ∈ l

axiom circle_through_points (A C : Point) : ∃ Γ : Circle, A ∈ Γ ∧ C ∈ Γ

axiom tangents_intersect_at_T {Γ : Circle} (A C : Point) (hAC : A ∈ Γ ∧ C ∈ Γ) :
  ∃ T : Point, tangent_to_circle A Γ ∧ tangent_to_circle C Γ ∧ T = intersection_of_tangents A C Γ

axiom TB_intersects_Γ_at_S (T B : Point) (hT : tangent_point T) (B ∉ T) :
  ∃ S : Point, line_through T B ∧ S ∈ Γ

axiom X_is_intersection_of_AC_and_angle_bisector (A C S : Point) :
  ∃ X : Point, X = intersection_of_angle_bisector_of_ASA_angle A C S (line_through A C)

-- Theorem to be proved
theorem X_is_fixed_as_Γ_varies (A B C T S X : Point) (Γ : Circle)
  (h_collinear : collinear A B C) (h_circle : circle_through_points A C)
  (h_tangents_intersect : tangents_intersect_at_T A C h_circle) (h_TB_intersect : TB_intersects_Γ_at_S T B h_tangents_intersect)
  (h_X_intersect : X_is_intersection_of_AC_and_angle_bisector A C S) :
  is_fixed_point X := sorry

end X_is_fixed_as_Γ_varies_l441_441255


namespace angle_MP_AD_is_ninety_degrees_l441_441281

variables {α : Type*} [LieGroup α]

structure Point (α : Type*) : Type* :=
(x : α) -- x-coordinate of a point
(y : α) -- y-coordinate of a point

structure Line (α : Type*) : Type* :=
(start : Point α)
(end : Point α)

variables (A B C D M K P : Point α) (AD MP : Line α)

noncomputable def midpoint (A B : Point α) : Point α :=
sorry -- Midpoint definition

noncomputable def perpendicular_bisector (B C M : Point α) : Line α :=
sorry -- Perpendicular bisector definition

noncomputable def circle_diameter (K C : Point α) : Line α :=
sorry -- Circle with diameter definition

noncomputable def intersects (circle_diam : Line α) (segment : Line α) : Point α :=
sorry -- Intersection of circle and segment definition

noncomputable def angle_between_lines (L1 L2 : Line α) : Real :=
sorry -- Angle between two lines definition

theorem angle_MP_AD_is_ninety_degrees 
    (h1 : inscribed_quadrilateral A B C D) -- ABCD is cyclic
    (h2 : midpoint B C = M) -- M is the midpoint of BC
    (h3 : ∃ K, perpendicular_bisector B C M ∩ AB = K) -- K as described
    (h4 : let ω := circle_diameter K C in intersects ω CD = P ∧ P ≠ C) -- circle with diameter KC intersects CD at P
    (h_angle : angle_between_lines MP AD = 90) 
    : angle_between_lines MP AD = 90 :=
by
  sorry

end angle_MP_AD_is_ninety_degrees_l441_441281


namespace problem_solution_l441_441559

theorem problem_solution :
  let m := 9
  let n := 20
  let lhs := (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8)
  let rhs := 9 / 20
  lhs = rhs → 10 * m + n = 110 :=
by sorry

end problem_solution_l441_441559


namespace correct_option_is_C_l441_441986

-- Define the options as propositions
def optionA := ∀ x, x - (1 / (x - 1)) = 0
def optionB := ∀ x, 7 * x^2 + (1 / x^2) - 1 = 0
def optionC := ∀ x, x^2 = 0
def optionD := ∀ x, (x + 1) * (x - 2) = x * (x + 1)

-- Define what it means to be a quadratic equation
def is_quadratic_eq (eq : ∀ x, x^2 = 0) := 
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, a * x^2 + b * x + c = 0

-- Proof statement that Option C is the quadratic equation and others are not
theorem correct_option_is_C : 
  (∃ x, ∀ eq, optionA → ¬is_quadratic_eq optionA) ∧
  (∃ x, ∀ eq, optionB → ¬is_quadratic_eq optionB) ∧
  (∃ x, ∀ eq, optionC → is_quadratic_eq optionC) ∧
  (∃ x, ∀ eq, optionD → ¬is_quadratic_eq optionD) :=
sorry

end correct_option_is_C_l441_441986


namespace scientist_can_buy_ticket_l441_441025

noncomputable def probability_of_buying_ticket (initial_rubles : ℕ) (ticket_cost : ℕ) (lottery_cost : ℕ) (win_amount : ℕ) (p : ℚ) (q : ℚ) : ℚ :=
  let x2 : ℚ := p^2 * (1 + 2*q) / (1 - 2*p*q^2)
  in x2

theorem scientist_can_buy_ticket : probability_of_buying_ticket 20 45 10 30 0.1 0.9 = 0.033 := 
begin
  -- This is where the proof would be written explicitly; for now, we use sorry to skip the proof.
  sorry
end

end scientist_can_buy_ticket_l441_441025


namespace find_playground_side_length_l441_441384

-- Define the conditions
def playground_side_length (x : ℝ) : Prop :=
  let perimeter_square := 4 * x
  let perimeter_garden := 2 * (12 + 9)
  let total_perimeter := perimeter_square + perimeter_garden
  total_perimeter = 150

-- State the main theorem to prove that the side length of the square fence around the playground is 27 yards
theorem find_playground_side_length : ∃ x : ℝ, playground_side_length x ∧ x = 27 :=
by
  exists 27
  sorry

end find_playground_side_length_l441_441384


namespace isosceles_triangle_l441_441828

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441828


namespace snow_white_last_trip_l441_441337

universe u

-- Define the dwarf names as an enumerated type
inductive Dwarf : Type
| Happy | Grumpy | Dopey | Bashful | Sleepy | Doc | Sneezy
deriving DecidableEq, Repr

open Dwarf

-- Define conditions
def initial_lineup : List Dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- Define the condition of adjacency
def adjacent (d1 d2 : Dwarf) : Prop :=
  List.pairwise_adjacent (· = d2) initial_lineup d1

-- Define the boat capacity
def boat_capacity : Fin 4 := by sorry

-- Snow White is the only one who can row
def snow_white_rows : Prop := true

-- No quarrels condition
def no_quarrel_without_snow_white (group : List Dwarf) : Prop :=
  ∀ d1 d2, d1 ∈ group → d2 ∈ group → ¬ adjacent d1 d2

-- Objective: Transfer all dwarfs without quarrels
theorem snow_white_last_trip (trips : List (List Dwarf)) :
  ∃ last_trip : List Dwarf,
    last_trip = [Grumpy, Bashful, Doc] ∧
    no_quarrel_without_snow_white (initial_lineup.diff (trips.join ++ last_trip)) :=
sorry

end snow_white_last_trip_l441_441337


namespace yaya_bike_walk_l441_441987

theorem yaya_bike_walk (x y : ℝ) : 
  (x + y = 1.5 ∧ 15 * x + 5 * y = 20) ↔ (x + y = 1.5 ∧ 15 * x + 5 * y = 20) :=
by 
  sorry

end yaya_bike_walk_l441_441987


namespace triangle_isosceles_if_parallel_l441_441863

theorem triangle_isosceles_if_parallel
  (A B C P Q : Type*)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (h : parallel PQ AC) : isosceles_triangle A B C := 
sorry

end triangle_isosceles_if_parallel_l441_441863


namespace ratio_B_to_A_l441_441485

theorem ratio_B_to_A (A B C : ℝ) 
  (hA : A = 1 / 21) 
  (hC : C = 2 * B) 
  (h_sum : A + B + C = 1 / 3) : 
  B / A = 2 := 
by 
  /- Proof goes here, but it's omitted as per instructions -/
  sorry

end ratio_B_to_A_l441_441485


namespace inequality_solution_l441_441136

theorem inequality_solution (x : ℝ) :
  (abs ((x^2 - 5 * x + 4) / 3) < 1) ↔ 
  ((5 - Real.sqrt 21) / 2 < x) ∧ (x < (5 + Real.sqrt 21) / 2) := 
sorry

end inequality_solution_l441_441136


namespace technician_drive_from_center_l441_441066

-- Given conditions
variables (D : ℝ) -- The distance from the starting point to the service center (one way)
variables (total_completion_percent : ℝ) -- Percentage of round-trip completion
variables (distance_to_center : ℝ) -- Distance covered when the technician completes the drive to the center
variables (distance_covered : ℝ) -- Total distance covered by the technician

-- Define the completion percentage as given in the problem
def round_trip_distance := 2 * D
def completion_percent := 0.65
def completed_distance := total_completion_percent * round_trip_distance
def drive_percent_from_center := (completed_distance - distance_to_center) / D * 100

-- Given conditions in the form of variable assignments
noncomputable def condition_1 := distance_to_center = D
noncomputable def condition_2 := total_completion_percent = completion_percent
noncomputable def condition_3 := distance_covered = completed_distance

-- Theorem statement to prove
theorem technician_drive_from_center : drive_percent_from_center D total_completion_percent distance_to_center distance_covered = 30 :=
by
  rw [condition_1, condition_2, condition_3]
  sorry

end technician_drive_from_center_l441_441066


namespace fraction_equivalence_l441_441457

theorem fraction_equivalence : (8 : ℝ) / (5 * 48) = 0.8 / (5 * 0.48) :=
  sorry

end fraction_equivalence_l441_441457


namespace translated_midpoint_l441_441923

/-- Given segment s1 with endpoints (2, -4) and (10, 4), and a translation of (-3, -5),
    the midpoint of the translated segment s2 is (3, -5). -/
theorem translated_midpoint (x1 y1 x2 y2 : ℤ) (tx ty : ℤ) :
    (x1, y1) = (2, -4) → (x2, y2) = (10, 4) → (tx, ty) = (-3, -5) →
    let midpoint_s1 := ((x1 + x2) / 2, (y1 + y2) / 2) in
    let midpoint_s2 := (midpoint_s1.1 + tx, midpoint_s1.2 + ty) in
    midpoint_s2 = (3, -5) :=
by
  intros h1 h2 h3
  have h_midpoint_s1 : midpoint_s1 = (6, 0) :=
    by calc
      midpoint_s1 = ((2 + 10) / 2, (-4 + 4) / 2) : by simp [h1, h2]
      ...          = (6, 0)                        : by norm_num
  have h_midpoint_s2 : midpoint_s2 = (6 - 3, 0 - 5) := by simp [h_midpoint_s1, h3]
  show midpoint_s2 = (3, -5) from
    calc
      midpoint_s2 = (6 - 3, 0 - 5) : by simp [h_midpoint_s2]
      ...           = (3, -5)        : by norm_num

-- End of Lean statement.

end translated_midpoint_l441_441923


namespace midpoint_segment_AE_l441_441393

open Classical
open PointGeometry

variables {Point : Type} [Torsor ℝ Point]

-- Conditions as assumptions
variables (A B C D E M : Point)
variables (is_isosceles : ∃ O, O = midpoint ℝ A C ∧ O = midpoint ℝ A B)
variables (line_d : line ℝ C) (h_perp : Perpendicular line_d (line ℝ B C))
variables (D_on_d : OnLine ℝ D line_d)
variables (E_condition : Parallelogram ℝ A E D B)
variables (M_intersection : Intersection ℝ (line ℝ A E) line_d M)

-- Theorem statement
theorem midpoint_segment_AE : Midpoint ℝ A E M :=
begin
  sorry
end

end midpoint_segment_AE_l441_441393


namespace problem_b_problem_c_l441_441575

variable (α β : ℝ)

-- Definition of acute angles
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2

-- Conditions given in the problem
axiom α_acute : is_acute α
axiom β_acute : is_acute β
axiom sin_add_beta : sin (α + β) = 2 * sin (α - β)

-- The main theorems to prove
theorem problem_b : 0 < α - β ∧ α - β ≤ π / 6 :=
by
  sorry

theorem problem_c (h : sin α = 2 * sin β) : cos α = sqrt 6 / 4 :=
by
  sorry

end problem_b_problem_c_l441_441575


namespace area_of_WXYZ_l441_441550

-- Define the conditions
def longer_side : ℝ := 6
def shorter_side : ℝ := longer_side / 3
def side_of_square : ℝ := 2 * longer_side
def area_of_square : ℝ := side_of_square ^ 2

-- Statement claiming the area of the square WXYZ is 144 square feet
theorem area_of_WXYZ :
  area_of_square = 144 := by
  sorry

end area_of_WXYZ_l441_441550


namespace points_four_units_away_l441_441956

theorem points_four_units_away (x : ℚ) (h : |x| = 4) : x = -4 ∨ x = 4 := 
by 
  sorry

end points_four_units_away_l441_441956


namespace final_trip_l441_441330

-- Definitions for the conditions
def dwarf := String
def snow_white := "Snow White" : dwarf
def Happy : dwarf := "Happy"
def Grumpy : dwarf := "Grumpy"
def Dopey : dwarf := "Dopey"
def Bashful : dwarf := "Bashful"
def Sleepy : dwarf := "Sleepy"
def Doc : dwarf := "Doc"
def Sneezy : dwarf := "Sneezy"

-- The dwarfs lineup from left to right
def lineup : List dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- The boat can hold Snow White and up to 3 dwarfs
def boat_capacity (load : List dwarf) : Prop := snow_white ∈ load ∧ load.length ≤ 4

-- Any two dwarfs standing next to each other in the original lineup will quarrel if left without Snow White
def will_quarrel (d1 d2 : dwarf) : Prop :=
  (d1, d2) ∈ (lineup.zip lineup.tail)

-- The objective: Prove that on the last trip, Snow White will take Grumpy, Bashful, and Doc
theorem final_trip : ∃ load : List dwarf, 
  set.load ⊆ {snow_white, Grumpy, Bashful, Doc} ∧ boat_capacity load :=
  sorry

end final_trip_l441_441330


namespace bakery_storage_sugar_amount_l441_441649

variables (S F B : ℝ)

theorem bakery_storage_sugar_amount
  (h1 : S / F = 3 / 8)
  (h2 : F / B = 10 / 1)
  (h3 : F / (B + 60) = 8 / 1) :
  S = 900 :=
begin
  sorry
end

end bakery_storage_sugar_amount_l441_441649


namespace problem_statement_l441_441743

variables {A B C P Q I Z : Point}
variable triangle_ABC_is_isosceles : Prop

-- Definitions based on given conditions
def Parallel (PQ AC : Line) : Prop := -- Definition of parallelism
sorry

def Chord (PQ : Segment) (circle1 circle2 : Circle) : Prop := -- Definition of common chord
sorry

-- Incenter and center properties
def Incenter (I : Point) (ABC : Triangle) : Prop := -- Definition of incenter of a triangle
sorry

def Center (Z : Point) (circle : Circle) : Prop := -- Definition of center of a circle
sorry

-- Problem
theorem problem_statement
  (H_parallel : Parallel PQ AC)
  (H_common_chord : Chord PQ (incircle (△ A B C)) (circle A I C))
  (H_incenter : Incenter I (△ A B C))
  (H_center : Center Z (circle A I C)) :
  (triangle_ABC_is_isosceles = (AB = BC)) :=
sorry

end problem_statement_l441_441743


namespace triangle_is_isosceles_if_parallel_l441_441731

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441731


namespace ellipse_hyperbola_right_triangle_l441_441587

noncomputable def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
let AB := (B.1 - A.1, B.2 - A.2)
let AC := (C.1 - A.1, C.2 - A.2)
in AB.1 * AC.1 + AB.2 * AC.2 = 0

theorem ellipse_hyperbola_right_triangle
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (h_condition : a > b)
  (h_foci_shared : true) -- abstracted out for focusing on the core proof
  (h_product_eccentricities : (sqrt(2) * (sqrt(2) / 2)) = 1)
  (P : ℝ × ℝ)
  (h_int_point : (P.1^2 - P.2^2 = 1) ∧ (P.1^2 / 4 + P.2^2 / 2 = 1)) :
  is_right_triangle (sqrt(2), 0) P (-sqrt(2), 0) :=
sorry

end ellipse_hyperbola_right_triangle_l441_441587


namespace clock_angle_5_30_l441_441978

theorem clock_angle_5_30 (h_degree : ℕ → ℝ) (m_degree : ℕ → ℝ) (hours_pos : ℕ → ℝ) :
  (h_degree 12 = 360) →
  (m_degree 60 = 360) →
  (hours_pos 5 + h_degree 1 - (m_degree 30 / 2) = 165) →
  (m_degree 30 = 180) →
  ∃ θ : ℝ, θ = abs (m_degree 30 - (hours_pos 5 + h_degree 1 - (m_degree 30 / 2))) ∧ θ = 15 :=
by
  sorry

end clock_angle_5_30_l441_441978


namespace evaluate_expression_l441_441530

variable (y : ℕ)

theorem evaluate_expression (h : y = 3) : 
    (y^(1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) / y^(2 + 4 + 6 + 8 + 10 + 12)) = 3^58 :=
by
  -- Proof will be done here
  sorry

end evaluate_expression_l441_441530


namespace sum_of_digits_S_n_l441_441669

def S (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_S_n (n : ℕ) : S n = 1384 → S (n + 1) = 15 := by
  sorry

end sum_of_digits_S_n_l441_441669


namespace percentage_increase_bears_l441_441658

-- Define the initial conditions
variables (B H : ℝ) -- B: bears per week without an assistant, H: hours per week without an assistant

-- Define the rate without assistant
def rate_without_assistant : ℝ := B / H

-- Define the working hours with an assistant
def hours_with_assistant : ℝ := 0.9 * H

-- Define the rate with an assistant (100% increase)
def rate_with_assistant : ℝ := 2 * rate_without_assistant

-- Define the number of bears per week with an assistant
def bears_with_assistant : ℝ := rate_with_assistant * hours_with_assistant

-- Prove the percentage increase in the number of bears made per week
theorem percentage_increase_bears (hB : B > 0) (hH : H > 0) :
  ((bears_with_assistant B H - B) / B) * 100 = 80 :=
by
  unfold bears_with_assistant rate_with_assistant hours_with_assistant rate_without_assistant
  simp
  sorry

end percentage_increase_bears_l441_441658


namespace roots_separation_condition_l441_441164

theorem roots_separation_condition (m n p q : ℝ)
  (h_1 : ∃ (x1 x2 : ℝ), x1 + x2 = -m ∧ x1 * x2 = n ∧ x1 ≠ x2)
  (h_2 : ∃ (x3 x4 : ℝ), x3 + x4 = -p ∧ x3 * x4 = q ∧ x3 ≠ x4)
  (h_3 : (∀ x1 x2 x3 x4 : ℝ, x1 + x2 = -m ∧ x1 * x2 = n ∧ x3 + x4 = -p ∧ x3 * x4 = q → 
         (x3 - x1) * (x3 - x2) * (x4 - x1) * (x4 - x2) < 0)) : 
  (n - q)^2 + (m - p) * (m * q - n * p) < 0 :=
sorry

end roots_separation_condition_l441_441164


namespace find_c_l441_441604

variables {x b c : ℝ}

theorem find_c (H : (x + 3) * (x + b) = x^2 + c * x + 12) (hb : b = 4) : c = 7 :=
by sorry

end find_c_l441_441604


namespace general_equation_of_line_rectangular_coordinate_equation_of_curve_distance_between_intersections_l441_441229

-- Definitions for conditions
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + (√2 / 2) * t, 2 + (√2 / 2) * t)

def polar_curve (θ : ℝ) : ℝ :=
  4 * Real.sin θ

-- Problem statements

-- 1. Proving the general equation of the line
theorem general_equation_of_line :
  ∃ (a b c : ℝ), ∀ (t : ℝ), 
  let (x, y) := parametric_line t in
  a * x + b * y + c = 0 := sorry

-- 2. Proving the rectangular coordinate equation of the curve
theorem rectangular_coordinate_equation_of_curve :
  ∀ (x y : ℝ), 
  (∃ θ : ℝ, x = 4 * Real.sin θ * Real.cos θ 
            ∧ y = 4 * Real.sin θ * (1 + Real.sin θ)) ↔
  x^2 + (y - 2)^2 = 4 := sorry

-- 3. Proving the distance between intersection points
theorem distance_between_intersections :
  ∀ (t_A t_B : ℝ), 
  (t_A^2 + √2 * t_A - 3 = 0) ∧ 
  (t_B^2 + √2 * t_B - 3 = 0) →
  (t_A + t_B = -√2) ∧ 
  (t_A * t_B = -3) →
  Real.sqrt ((t_A + t_B)^2 - 4 * (t_A * t_B)) = √14 := sorry

end general_equation_of_line_rectangular_coordinate_equation_of_curve_distance_between_intersections_l441_441229


namespace trigonometric_solution_l441_441928

theorem trigonometric_solution (x y : ℝ) : 
  (∃ (k l : ℤ), 
      x = (π / 2) * (2 * k + l + 1) ∧ y = (π / 2) * (2 * k - l)) ∨ 
  (∃ (m n : ℤ), 
      x = (π / 2) * (2 * m + n) ∧ y = (π / 2) * (2 * m - n - 1)) ↔ 
  sin (x + y) ^ 2 - cos (x - y) ^ 2 = 1 :=
by 
  sorry

end trigonometric_solution_l441_441928


namespace isosceles_triangle_of_parallel_l441_441788

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441788


namespace initial_alcohol_percentage_l441_441035

theorem initial_alcohol_percentage (P : ℚ) (initial_volume : ℚ) (added_alcohol : ℚ) (added_water : ℚ)
  (final_percentage : ℚ) (final_volume : ℚ) (alcohol_volume_in_initial_solution : ℚ) :
  initial_volume = 40 ∧ 
  added_alcohol = 3.5 ∧ 
  added_water = 6.5 ∧ 
  final_percentage = 0.11 ∧ 
  final_volume = 50 ∧ 
  alcohol_volume_in_initial_solution = (P / 100) * initial_volume ∧ 
  alcohol_volume_in_initial_solution + added_alcohol = final_percentage * final_volume
  → P = 5 :=
by
  sorry

end initial_alcohol_percentage_l441_441035


namespace sin_add_tan_is_odd_and_periodic_l441_441946

-- Definitions and conditions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def period (f : ℝ → ℝ) (T : ℝ) := ∀ x, f (x + T) = f x

def sin_period_2pi := period sin (2 * Real.pi)

def sin_is_odd := is_odd sin

def tan_period_pi := period tan Real.pi

def tan_is_odd := is_odd tan

-- Proof statement
theorem sin_add_tan_is_odd_and_periodic :
  (is_odd (λ x, sin x + tan x)) ∧ (period (λ x, sin x + tan x) (2 * Real.pi)) :=
  by
    sorry

end sin_add_tan_is_odd_and_periodic_l441_441946


namespace cos_pi_minus_alpha_l441_441163

theorem cos_pi_minus_alpha (α : ℝ) (h_interval : α ∈ Ioo (-π) (-π / 2)) (h_sin : Real.sin α = -5 / 13) :
  Real.cos (π - α) = 12 / 13 :=
by
  sorry

end cos_pi_minus_alpha_l441_441163


namespace isosceles_triangle_of_parallel_l441_441875

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441875


namespace replace_minus_with_plus_to_2013_l441_441235

theorem replace_minus_with_plus_to_2013 (E : ℕ) (h : E = 2013^2 - 2012^2 - 2011^2 - ... - 2^2 - 1^2) : 
  ∃ f : ℕ → ℤ, sumRange 1 2013 (λ n, f(n) * n^2) = 2013 :=
sorry

end replace_minus_with_plus_to_2013_l441_441235


namespace isosceles_triangle_l441_441820

theorem isosceles_triangle (A B C P Q : Type) [linear_order P] [linear_order Q] (h : PQ ∥ AC) : is_isosceles A B C :=
sorry

end isosceles_triangle_l441_441820


namespace power_of_point_thm_l441_441918

noncomputable def power_of_point (Ω A B C D T O : Point ℝ) (r : ℝ) (Γ : Circle) : Prop :=
  Ω ∉ Γ ∧
  ΩA * ΩB = ΩC * ΩD ∧
  ΩA * ΩB = ΩT * ΩT ∧
  ΩA * ΩB = ΩO² - r²

-- Define geometric entities and properties
def Point (ℝ : Type) : Type := ℝ × ℝ
structure Circle :=
  (center : Point ℝ)
  (radius : ℝ)

-- Define the theorem and conditions
theorem power_of_point_thm (Ω A B C D T O : Point ℝ) (r : ℝ) (Γ : Circle) :
  power_of_point Ω A B C D T O r Γ :=
begin
  sorry
end

end power_of_point_thm_l441_441918


namespace common_denominator_first_set_common_denominator_second_set_l441_441540

theorem common_denominator_first_set (x y : ℕ) (h₁ : y ≠ 0) : Nat.lcm (3 * y) (2 * y^2) = 6 * y^2 :=
by sorry

theorem common_denominator_second_set (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : Nat.lcm (a^2 * b) (3 * a * b^2) = 3 * a^2 * b^2 :=
by sorry

end common_denominator_first_set_common_denominator_second_set_l441_441540


namespace cars_on_river_road_l441_441366

theorem cars_on_river_road (B C : ℕ) (h_ratio : B / C = 1 / 3) (h_fewer : C = B + 40) : C = 60 :=
sorry

end cars_on_river_road_l441_441366


namespace equation1_solution_equation2_solution_l441_441341

variable (x : ℝ)

theorem equation1_solution :
  ((2 * x - 5) / 6 - (3 * x + 1) / 2 = 1) → (x = -2) :=
by
  sorry

theorem equation2_solution :
  (3 * x - 7 * (x - 1) = 3 - 2 * (x + 3)) → (x = 5) :=
by
  sorry

end equation1_solution_equation2_solution_l441_441341


namespace triangle_is_isosceles_l441_441896

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441896


namespace coach_mike_change_in_usd_l441_441508

theorem coach_mike_change_in_usd
  (cost_per_cup_euros : ℝ)
  (payment_euros : ℝ)
  (conversion_rate : ℝ)
  (cost_per_cup_feq : cost_per_cup_euros = 0.58)
  (payment_feq : payment_euros = 1)
  (conversion_rate_feq : conversion_rate = 1.18) :
  (payment_euros * conversion_rate - cost_per_cup_euros * conversion_rate) = 0.4956 :=
by
  rw [cost_per_cup_feq, payment_feq, conversion_rate_feq]
  norm_num
  sorry

end coach_mike_change_in_usd_l441_441508


namespace _l441_441563

noncomputable def cyclic_quadrilateral (A B C D : Type) :=
  ∃ O : Type, ∀ P : Type, P ∈ {A, B, C, D} → ∃ R, circle O P R

noncomputable def midpoint (M C D : Type) := ∃ O : Type, segment M C ≃ segment M D

noncomputable theorem collinearity_E_F_N 
  (A B C D E F M N : Type)
  (h_cyclic_quad : cyclic_quadrilateral A B C D)
  (h_inter_AD_BC : ∃ E, line_through A D ∩ line_through B C = {E})
  (h_inter_AC_BD : ∃ F, line_through A C ∩ line_through B D = {F})
  (h_midpoint : midpoint M C D)
  (h_on_circumcircle : ∃ O, on_circumcircle A B M N O ∧ N ≠ M)
  (h_ratios_equal : ∀ P Q : Type, ratio (segment A N) (segment B N) = ratio (segment A M) (segment B M)) :
  collinear {E, F, N} :=
sorry

end _l441_441563


namespace find_principal_l441_441018

-- Conditions as definitions
def amount : ℝ := 1120
def rate : ℝ := 0.05
def time : ℝ := 2

-- Required to add noncomputable due to the use of division and real numbers
noncomputable def principal : ℝ := amount / (1 + rate * time)

-- The main theorem statement which needs to be proved
theorem find_principal :
  principal = 1018.18 :=
sorry  -- Proof is not required; it is left as sorry

end find_principal_l441_441018


namespace find_m_l441_441193

noncomputable def isPerfectSquareTrinomial (f : ℤ → ℤ) : Prop :=
  ∃ (a b : ℤ), f = λ x, a^2 * x^2 + 2 * a * b * x + b^2

theorem find_m (m : ℤ) : isPerfectSquareTrinomial (λ x : ℤ, x^2 + m * x + 16) → (m = 8 ∨ m = -8) :=
by
  sorry

end find_m_l441_441193


namespace find_N_l441_441138

variable {ℕ : Type}

theorem find_N (x y N : ℕ) (h1 : x / (2 * y) = N / 2) (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : N = 3 := 
by
  sorry

end find_N_l441_441138


namespace computer_proficiency_test_population_l441_441215

theorem computer_proficiency_test_population (participants : ℕ) (sample_size : ℕ) (total_group : ℕ) : 
  participants = 5000 → 
  sample_size = 200 → 
  total_group = participants →
  is_population total_group :=
by
  intros h₁ h₂ h₃
  sorry

end computer_proficiency_test_population_l441_441215


namespace different_coordinate_l441_441489

def M : ℝ × ℝ := (-5, real.pi / 3)
def A : ℝ × ℝ := (5, -real.pi / 3)
def B : ℝ × ℝ := (5, 4 * real.pi / 3)
def C : ℝ × ℝ := (5, -2 * real.pi / 3)
def D : ℝ × ℝ := (-5, -5 * real.pi / 3)

theorem different_coordinate : M ≠ A := by
  sorry

end different_coordinate_l441_441489


namespace two_digit_numbers_l441_441071

theorem two_digit_numbers (n m : ℕ) (Hn : 1 ≤ n ∧ n ≤ 9) (Hm : n < m ∧ m ≤ 9) :
  ∃ (count : ℕ), count = 36 :=
by
  sorry

end two_digit_numbers_l441_441071


namespace cosine_identity_l441_441030

variable {α β γ : ℝ}
variable {a b c : ℝ}

def law_of_cosines (α β γ : ℝ) (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 - 2*b*c*cos α ∧
  b^2 = a^2 + c^2 - 2*a*c*cos β ∧
  c^2 = a^2 + b^2 - 2*a*b*cos γ

theorem cosine_identity
  (h : law_of_cosines α β γ a b c) :
  a*b*cos γ + b*c*cos α + c*a*cos β = (a^2 + b^2 + c^2) / 2 := by
  sorry

end cosine_identity_l441_441030


namespace find_f_at_6_l441_441589

def f (x : ℝ) : ℝ := a ^ x

theorem find_f_at_6 (a : ℝ) (h : a ^ 3 = 8) : f a 6 = 64 :=
by
  sorry

end find_f_at_6_l441_441589


namespace snow_white_last_trip_dwarfs_l441_441304

-- Definitions for the conditions
def original_lineup := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]
def only_snow_white_can_row := True
def boat_capacity_snow_white_and_dwarfs := 3
def dwarfs_quarrel_if_adjacent (d1 d2 : String) : Prop :=
  let index_d1 := List.indexOf original_lineup d1
  let index_d2 := List.indexOf original_lineup d2
  abs (index_d1 - index_d2) = 1

-- Theorem to prove the correct answer
theorem snow_white_last_trip_dwarfs :
  let last_trip_dwarfs := ["Grumpy", "Bashful", "Sneezy"]
  ∃ (trip : List String), trip = last_trip_dwarfs ∧ 
  ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → d1 ≠ d2 → ¬dwarfs_quarrel_if_adjacent d1 d2 :=
by
  sorry

end snow_white_last_trip_dwarfs_l441_441304


namespace largest_of_30_consecutive_odd_integers_l441_441370

theorem largest_of_30_consecutive_odd_integers (sum : ℕ) (h_sum : sum = 12000) : 
  ∃ y, (∀ k < 30, (y + 2 * k) % 2 = 1) ∧ (∑ k in finset.range 30, (y + 2 * k) = 12000) ∧ (y + 58 = 429) :=
by
  let y := 371
  use y
  have h1 : ∑ k in finset.range 30, (y + 2 * k) = 12000 := sorry
  have h2 : y + 58 = 429 := sorry
  split
  {
    intro k hk
    exact mod_eq_of_lt_dec (2 * k).odd_mod_two
  }
  {
    split
    {
      exact h1
    }
    {
      exact h2
    }
  }

end largest_of_30_consecutive_odd_integers_l441_441370


namespace expand_expression_l441_441532

theorem expand_expression (x : ℝ) : 3 * (x - 6) * (x - 7) = 3 * x^2 - 39 * x + 126 := by
  sorry

end expand_expression_l441_441532


namespace terry_mary_same_color_probability_l441_441469

theorem terry_mary_same_color_probability :
  let initial_red := 12
  let initial_blue := 8
  let total_candies := initial_red + initial_blue
  let terry_picks_two_red := (initial_red.choose 2) / (total_candies.choose 2)
  let terry_picks_two_blue := (initial_blue.choose 2) / (total_candies.choose 2)
  let remaining_for_mary_if_terry_red := initial_red - 2 - 1
  let remaining_for_mary_if_terry_blue := initial_blue - 2
  let mary_picks_two_red_if_terry_red := (remaining_for_mary_if_terry_red.choose 2) / ((total_candies - 3).choose 2)
  let mary_picks_two_blue_if_terry_blue := (remaining_for_mary_if_terry_blue.choose 2) / ((total_candies - 2).choose 2)
  let same_red_prob := terry_picks_two_red * mary_picks_two_red_if_terry_red
  let same_blue_prob := terry_picks_two_blue * mary_picks_two_blue_if_terry_blue
  in (same_red_prob + same_blue_prob) = 1217/10299 :=
sorry

end terry_mary_same_color_probability_l441_441469


namespace total_students_l441_441020

theorem total_students (boys girls : ℕ) (h_ratio : boys / girls = 8 / 5) (h_girls : girls = 120) : boys + girls = 312 :=
by
  sorry

end total_students_l441_441020


namespace triangle_is_isosceles_l441_441848

theorem triangle_is_isosceles (A B C P Q : Type*) (h : P Q ∥ A C) : is_isosceles A B C :=
sorry

end triangle_is_isosceles_l441_441848


namespace solution_set_f_gt_1_l441_441105

def f (x : ℝ) : ℝ :=
  if x < 2 then exp (x - 1) else -log x / log 3

theorem solution_set_f_gt_1 : { x : ℝ | f x > 1 } = set.Ioo 1 2 :=
by
  sorry

end solution_set_f_gt_1_l441_441105


namespace find_angle_A_l441_441208

open Real

-- Define the main theorem
theorem find_angle_A 
  (C : ℝ) (C_eq : C = π / 3)
  (b : ℝ) (b_eq : b = sqrt 2)
  (c : ℝ) (c_eq : c = sqrt 3) : 
  ∃ A : ℝ, A = 5 * π / 12 := 
by
  -- Proof goes here, but is skipped
  sorry

end find_angle_A_l441_441208


namespace isosceles_triangle_of_parallel_l441_441774

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441774


namespace symmetric_theta_l441_441612

theorem symmetric_theta:
  (P Q : ℝ × ℝ) (θ : ℝ) 
  (hP : P = (Real.cos θ, Real.sin θ))
  (hQ : Q = (Real.cos (θ + π / 6), Real.sin (θ + π / 6)))
  (h_symmetry: P.1 = -Q.1 ∧ P.2 = Q.2) :
  θ = 5 * π / 12 :=
by
  sorry

end symmetric_theta_l441_441612


namespace y_value_of_point_P_terminal_side_l441_441201

theorem y_value_of_point_P_terminal_side :
  (∃ y : ℝ, (∃ P : ℝ × ℝ, P = (-1, y)
            ∧ (∃ θ : ℝ, θ = 2 * π / 3 ∧ P = (-cos θ, sin θ)))) →
  y = sqrt 3 :=
by
  sorry

end y_value_of_point_P_terminal_side_l441_441201


namespace find_intersection_probability_l441_441629

-- Define probabilities of events a, b, and c
variable (p : Set α -> ℝ)
variable (a b c : Set α)
hypothesis pa : p a = 1/5
hypothesis pb : p b = 2/5
hypothesis pc : p c = 3/5
hypothesis independent : ∀ (x y : Set α), x ∈ {a, b, c} → y ∈ {a, b, c} → x ≠ y → p (x ∩ y) = p x * p y

-- Problem statement
theorem find_intersection_probability : p (a ∩ b ∩ c) = 6/125 := by
  sorry

end find_intersection_probability_l441_441629


namespace fraction_of_paint_after_second_day_l441_441050

def initial_paint : ℝ := 1
def first_day_usage : ℝ := initial_paint / 3
def remaining_paint_after_first_day : ℝ := initial_paint - first_day_usage
def second_day_usage : ℝ := remaining_paint_after_first_day / 3
def remaining_paint_after_second_day : ℝ := remaining_paint_after_first_day - second_day_usage

theorem fraction_of_paint_after_second_day : 
  remaining_paint_after_second_day = 4 / 9 := 
by
  sorry

end fraction_of_paint_after_second_day_l441_441050


namespace ball_min_bounces_reach_target_height_l441_441998

noncomputable def minimum_bounces (initial_height : ℝ) (ratio : ℝ) (target_height : ℝ) : ℕ :=
  Nat.ceil (Real.log (target_height / initial_height) / Real.log ratio)

theorem ball_min_bounces_reach_target_height :
  minimum_bounces 20 (2 / 3) 2 = 6 :=
by
  -- This is where the proof would go, but we use sorry to skip it
  sorry

end ball_min_bounces_reach_target_height_l441_441998


namespace problem_statement_l441_441202

noncomputable def f : ℝ → ℝ := sorry -- Assume f exists but definition is not provided

axiom h : ∀ x : ℝ, f' x > f x -- Condition: f'(x) > f(x)

theorem problem_statement : f 2012 > real.exp 2012 * f 0 :=
by sorry

end problem_statement_l441_441202


namespace triangle_ABC_isosceles_l441_441840

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441840


namespace loss_percent_l441_441442

theorem loss_percent (CP SP : ℝ) (h_CP : CP = 600) (h_SP : SP = 550) :
  ((CP - SP) / CP) * 100 = 8.33 := by
  sorry

end loss_percent_l441_441442


namespace distance_between_A_and_B_l441_441098

-- Define the initial conditions and the statement to prove
theorem distance_between_A_and_B (meet_time : ℕ) (additional_time : ℕ) (dist_A_to_B : ℕ) (dist_B_to_A : ℕ) :
    meet_time = 5 ∧ additional_time = 3 ∧ dist_A_to_B = 130 ∧ dist_B_to_A = 160 → (dist_A_to_B + dist_B_to_A) / (meet_time - additional_time) * meet_time = 290 :=
by
  intros h
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  rw [h1, h3, h5, h6],
  have temp : (130 + 160 : ℕ) / (5 - 3) = 145 := rfl,
  rw temp,
  simp,
  done

end distance_between_A_and_B_l441_441098


namespace triangle_is_isosceles_if_parallel_l441_441722

theorem triangle_is_isosceles_if_parallel (A B C P Q : Point) : 
  (P Q // AC) → is_isosceles_triangle A B C :=
by
  sorry

end triangle_is_isosceles_if_parallel_l441_441722


namespace museum_total_visitors_l441_441460

theorem museum_total_visitors (yesterday_visitors : ℕ) (more_today : ℕ) :
  yesterday_visitors = 247 →
  more_today = 131 →
  let today_visitors := yesterday_visitors + more_today in
  yesterday_visitors + today_visitors = 625 :=
by
  intros hyesterday hmore
  let today_visitors := yesterday_visitors + more_today
  have hyesterday_visitors : yesterday_visitors = 247 := hyesterday
  have hmore_today : more_today = 131 := hmore
  calc
    yesterday_visitors + today_visitors
        = 247 + (247 + 131) : by rw [hyesterday_visitors, hmore_today]
    ... = 247 + 378 : by rfl
    ... = 625 : by rfl

end museum_total_visitors_l441_441460


namespace isosceles_triangle_of_parallel_l441_441781

theorem isosceles_triangle_of_parallel (A B C P Q : Point) : 
  (PQ ∥ AC) → 
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_parallel_l441_441781


namespace necessarily_positive_b_plus_c_l441_441283

-- Defining the conditions as hypotheses
variables (a b c : ℝ)
hypothesis ha : 0 < a ∧ a < 2
hypothesis hb : -2 < b ∧ b < 0
hypothesis hc : 0 < c ∧ c < 3

-- Statement to prove
theorem necessarily_positive_b_plus_c :
  b + c > 0 :=
sorry

end necessarily_positive_b_plus_c_l441_441283


namespace problem_solution_l441_441560

theorem problem_solution :
  let m := 9
  let n := 20
  let lhs := (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8)
  let rhs := 9 / 20
  lhs = rhs → 10 * m + n = 110 :=
by sorry

end problem_solution_l441_441560


namespace isosceles_triangle_of_parallel_l441_441765

theorem isosceles_triangle_of_parallel (A B C P Q : Point) (h_parallel : PQ ∥ AC) : 
  isosceles_triangle A B C := 
sorry

end isosceles_triangle_of_parallel_l441_441765


namespace value_of_expression_l441_441194

theorem value_of_expression (x : ℝ) (h : x ^ 2 - 3 * x + 1 = 0) : 
  x ≠ 0 → (x ^ 2) / (x ^ 4 + x ^ 2 + 1) = 1 / 8 :=
by 
  intros h1 
  sorry

end value_of_expression_l441_441194


namespace symmetric_theta_l441_441613

theorem symmetric_theta:
  (P Q : ℝ × ℝ) (θ : ℝ) 
  (hP : P = (Real.cos θ, Real.sin θ))
  (hQ : Q = (Real.cos (θ + π / 6), Real.sin (θ + π / 6)))
  (h_symmetry: P.1 = -Q.1 ∧ P.2 = Q.2) :
  θ = 5 * π / 12 :=
by
  sorry

end symmetric_theta_l441_441613


namespace find_2_pow_x_l441_441533

theorem find_2_pow_x (x y : ℝ) 
  (h1 : 2^x + 5^y = 7)
  (h2 : 2^(x+3) + 5^(y+1) = 152): 2^x = 39 :=
by
  sorry

end find_2_pow_x_l441_441533


namespace speed_boat_in_still_water_l441_441957

-- Define the conditions
def speed_of_current := 20
def speed_upstream := 30

-- Define the effective speed given conditions
def effective_speed (speed_in_still_water : ℕ) := speed_in_still_water - speed_of_current

-- Theorem stating the problem
theorem speed_boat_in_still_water : 
  ∃ (speed_in_still_water : ℕ), effective_speed speed_in_still_water = speed_upstream ∧ speed_in_still_water = 50 := 
by 
  -- Proof to be filled in
  sorry

end speed_boat_in_still_water_l441_441957


namespace exists_m_such_that_m_plus_one_pow_zero_eq_one_l441_441535

theorem exists_m_such_that_m_plus_one_pow_zero_eq_one : 
  ∃ m : ℤ, (m + 1)^0 = 1 ∧ m ≠ -1 :=
by
  sorry

end exists_m_such_that_m_plus_one_pow_zero_eq_one_l441_441535


namespace largest_integer_not_sum_of_30_and_composite_l441_441401

theorem largest_integer_not_sum_of_30_and_composite : 
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b < 30 ∧ ¬ prime b ∧ (n = 30 * a + b) → n = 157 :=
by
  sorry

end largest_integer_not_sum_of_30_and_composite_l441_441401


namespace connie_initial_marbles_l441_441454

theorem connie_initial_marbles (x : ℕ) (marbles_given : ℕ) (marbles_left : ℕ) 
  (h1 : marbles_given = 73) 
  (h2 : marbles_left = 70) 
  (initial_marbles : x = marbles_left + marbles_given) : 
  x = 143 :=
by 
  have h3 : marbles_left + marbles_given = 70 + 73, from congr_arg2 (+) h2 h1
  rw [h3] at initial_marbles
  exact initial_marbles

end connie_initial_marbles_l441_441454


namespace wendi_owns_rabbits_l441_441394

/-- Wendi's plot of land is 200 feet by 900 feet. -/
def area_land_in_feet : ℕ := 200 * 900

/-- One rabbit can eat enough grass to clear ten square yards of lawn area per day. -/
def rabbit_clear_per_day : ℕ := 10

/-- It would take 20 days for all of Wendi's rabbits to clear all the grass off of her grassland property. -/
def days_to_clear : ℕ := 20

/-- Convert feet to yards (3 feet in a yard). -/
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

/-- Calculate the total area of the land in square yards. -/
def area_land_in_yards : ℕ := (feet_to_yards 200) * (feet_to_yards 900)

theorem wendi_owns_rabbits (total_area : ℕ := area_land_in_yards)
                            (clear_area_per_rabbit : ℕ := rabbit_clear_per_day * days_to_clear) :
  total_area / clear_area_per_rabbit = 100 := 
sorry

end wendi_owns_rabbits_l441_441394


namespace distance_from_M_to_F_is_10_l441_441362

-- Define the parabola and the conditions
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the point M and focus F
def point_M (x : ℝ) : Prop := x = 6
def focus_F : ℝ × ℝ := (4, 0)  -- Inferred from parabola y^2 = 16x, focus is at (4, 0)

-- Define the function to calculate distance
def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Define the problem statement
theorem distance_from_M_to_F_is_10 :
  ∀ (x y : ℝ), parabola x y → point_M x → distance (x, y) focus_F = 10 :=
by
  intros x y parabola_cond point_M_cond
  sorry

end distance_from_M_to_F_is_10_l441_441362


namespace isosceles_triangle_l441_441751

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441751


namespace smallest_integer_proof_l441_441425

def smallest_integer_with_gcd_18_6 : Nat :=
  let n := 114
  if n > 100 ∧  (Nat.gcd n 18) = 6 then n else 0

theorem smallest_integer_proof : smallest_integer_with_gcd_18_6 = 114 := 
  by
    unfold smallest_integer_with_gcd_18_6
    have h₁ : 114 > 100 := by decide
    have h₂ : Nat.gcd 114 18 = 6 := by decide
    simp [h₁, h₂]
    sorry

end smallest_integer_proof_l441_441425


namespace root_range_m_l441_441627

theorem root_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * m * x + 4 = 0 → (x > 1 ∧ ∃ y : ℝ, y < 1 ∧ y^2 - 2 * m * y + 4 = 0)
  ∨ (x < 1 ∧ ∃ y : ℝ, y > 1 ∧ y^2 - 2 * m * y + 4 = 0))
  → m > 5 / 2 := 
sorry

end root_range_m_l441_441627


namespace minimum_birds_on_circle_l441_441966

theorem minimum_birds_on_circle (S : Finset ℕ) (hS : S.card = 10)
    (h : ∀ T : Finset ℕ, T ⊆ S → T.card = 5 → ∃ C : Finset ℕ, C ⊆ S ∧ C.card ≥ 4 ∧ (T ∩ C).card ≥ 4) :
    ∃ C : Finset ℕ, C ⊆ S ∧ C.card = 9 ∧ (∀ T : Finset ℕ, T ⊆ S → T.card = 5 → (T ∩ C).card ≥ 4) :=
begin
    sorry
end

end minimum_birds_on_circle_l441_441966


namespace ratio_tends_to_zero_as_n_tends_to_infinity_l441_441250

def smallest_prime_not_dividing (n : ℕ) : ℕ :=
  -- Function to find the smallest prime not dividing n
  sorry

theorem ratio_tends_to_zero_as_n_tends_to_infinity :
  ∀ ε > 0, ∃ N, ∀ n > N, (smallest_prime_not_dividing n : ℝ) / (n : ℝ) < ε := by
  sorry

end ratio_tends_to_zero_as_n_tends_to_infinity_l441_441250


namespace snow_white_last_trip_dwarfs_l441_441302

-- Definitions for the conditions
def original_lineup := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]
def only_snow_white_can_row := True
def boat_capacity_snow_white_and_dwarfs := 3
def dwarfs_quarrel_if_adjacent (d1 d2 : String) : Prop :=
  let index_d1 := List.indexOf original_lineup d1
  let index_d2 := List.indexOf original_lineup d2
  abs (index_d1 - index_d2) = 1

-- Theorem to prove the correct answer
theorem snow_white_last_trip_dwarfs :
  let last_trip_dwarfs := ["Grumpy", "Bashful", "Sneezy"]
  ∃ (trip : List String), trip = last_trip_dwarfs ∧ 
  ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → d1 ≠ d2 → ¬dwarfs_quarrel_if_adjacent d1 d2 :=
by
  sorry

end snow_white_last_trip_dwarfs_l441_441302


namespace log_sum_geometric_sequence_l441_441226

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), (∀ (n : ℕ), a (n+1) = r * a n)

theorem log_sum_geometric_sequence {a : ℕ → ℝ} (h : geometric_sequence a) (h4_5 : a 4 * a 5 = 32) :
  (log 2 (a 1) + log 2 (a 2) + log 2 (a 3) + log 2 (a 4) + log 2 (a 5) + log 2 (a 6) + log 2 (a 7) + log 2 (a 8)) = 20 :=
sorry

end log_sum_geometric_sequence_l441_441226


namespace triangle_is_isosceles_l441_441893

variables {A B C P Q I Z : Type}
variables [IsTriangle A B C]

-- Given conditions
variable (PQ_parallel_AC : ∀ {A C P Q : Type}, PQ ∥ AC)

-- Conclusion
theorem triangle_is_isosceles (PQ_parallel_AC : PQ ∥ AC) : IsIsoscelesTriangle A B C :=
sorry

end triangle_is_isosceles_l441_441893


namespace hypercoplanar_values_l441_441519

-- Define the points in 4-dimensional space
def point1 : ℝ × ℝ × ℝ × ℝ := (0, 0, 0, 0)
def point2 (b : ℝ) : ℝ × ℝ × ℝ × ℝ := (1, b, 0, 0)
def point3 (b : ℝ) : ℝ × ℝ × ℝ × ℝ := (0, 1, b, 0)
def point4 (b : ℝ) : ℝ × ℝ × ℝ × ℝ := (b, 0, 1, 0)
def point5 (b : ℝ) : ℝ × ℝ × ℝ × ℝ := (0, b, 0, 1)

-- Define the matrix using these points
def matrix_from_points (b : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![[1, 0, b, 0],
    [b, 1, 0, b],
    [0, b, 1, 0],
    [0, 0, 0, 1]]

-- Define the condition for hypercoplanarity using the determinant
def hypercoplanar (b : ℝ) : Prop := Matrix.det (matrix_from_points b) = 0

-- Define the theorem to be proven
theorem hypercoplanar_values :
  ∀ (b : ℝ), hypercoplanar b ↔ b = (1 / Real.sqrt 2) ∨ b = -(1 / Real.sqrt 2) :=
by
  sorry

end hypercoplanar_values_l441_441519


namespace maxElegantPairs_l441_441988

noncomputable def isElegantPair (a b : ℕ) : Prop :=
  ∃ k : ℕ, a + b = 2^k

theorem maxElegantPairs (n : ℕ) (h : n ≥ 2) (s : Finset ℕ) (hs : s.card = n) (hd : s.Nodup) :
  ∃ pairs : Finset (ℕ × ℕ), 
    (∀ p ∈ pairs, p.1 < p.2 ∧ p.1 ∈ s ∧ p.2 ∈ s ∧ isElegantPair p.1 p.2) ∧
    pairs.card = n - 1 :=
by
  sorry

end maxElegantPairs_l441_441988


namespace value_of_x_l441_441429

theorem value_of_x (x : ℝ) : 8^4 + 8^4 + 8^4 = 2^x → x = Real.log 3 / Real.log 2 + 12 :=
by
  sorry

end value_of_x_l441_441429


namespace tan_alpha_3_l441_441188

variable (α : ℝ)

theorem tan_alpha_3 (h1 : α ∈ Ioo 0 (π / 2))
  (h2 : sin α ^ 2 + cos (π / 2 + 2 * α) = 3 / 10) :
  tan α = 3 := 
sorry

end tan_alpha_3_l441_441188


namespace transformation_matrix_l441_441118

noncomputable def rotation_scaling_matrix (θ : ℝ) (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  s • Matrix.of ![
    ![Real.cos θ, -Real.sin θ],
    ![Real.sin θ, Real.cos θ]
  ]

noncomputable def translation_matrix (dx dy : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.of ![
    ![1, 0, dx],
    ![0, 1, dy],
    ![0, 0, 1]
  ]

noncomputable def combined_transform_matrix (θ : ℝ) (s dx dy : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  let rs := rotation_scaling_matrix θ s
  let tm : Matrix (Fin 3) (Fin 3) ℝ := Matrix.of ![
    ![rs 0 0, rs 0 1, 0],
    ![rs 1 0, rs 1 1, 0],
    ![0, 0, 1]
  ]
  translation_matrix dx dy ⬝ tm

theorem transformation_matrix :
  combined_transform_matrix (π / 3) (Real.sqrt 3) 2 3 =
  Matrix.of ![
    ![Real.sqrt 3 / 2, -Real.sqrt 3 / 2, 2],
    ![Real.sqrt 3 / 2, Real.sqrt 3 / 2, 3],
    ![0, 0, 1]
  ] :=
by
  sorry

end transformation_matrix_l441_441118


namespace fixed_point_exists_l441_441947

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) + 2

theorem fixed_point_exists (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : 
  f a 0 = 2 := by
  sorry

end fixed_point_exists_l441_441947


namespace find_minimum_of_g_l441_441108

def g (x : ℝ) := ⨆ y ∈ set.Icc 0 1, |y^2 - x * y|

theorem find_minimum_of_g : infi (λ x : ℝ, g x) = 3 - 2 * Real.sqrt 2 :=
by
  sorry

end find_minimum_of_g_l441_441108


namespace comic_books_order_count_l441_441268

theorem comic_books_order_count 
  (Batman_books : Finset ℕ)
  (XMen_books : Finset ℕ)
  (Calvin_Hobbes_books : Finset ℕ) 
  (h1 : Batman_books.card = 5)
  (h2 : XMen_books.card = 4)
  (h3 : Calvin_Hobbes_books.card = 3) :
  (5.factorial * 4.factorial * 3.factorial * 3.factorial = 103680) :=
by 
  sorry

end comic_books_order_count_l441_441268


namespace initial_deadline_l441_441045

theorem initial_deadline (D : ℝ) :
  (∀ (n : ℝ), (10 * 20) / 4 = n / 1) → 
  (∀ (m : ℝ), 8 * 75 = m * 3) →
  (∀ (d1 d2 : ℝ), d1 = 20 ∧ d2 = 93.75 → D = d1 + d2) →
  D = 113.75 :=
by {
  sorry
}

end initial_deadline_l441_441045


namespace room_total_space_l441_441239

-- Definitions based on the conditions
def bookshelf_space : ℕ := 80
def reserved_space : ℕ := 160
def number_of_shelves : ℕ := 3

-- The theorem statement
theorem room_total_space : 
  (number_of_shelves * bookshelf_space) + reserved_space = 400 := 
by
  sorry

end room_total_space_l441_441239


namespace arabella_first_step_time_l441_441493

def time_first_step (x : ℝ) : Prop :=
  let time_second_step := x / 2
  let time_third_step := x + x / 2
  (x + time_second_step + time_third_step = 90)

theorem arabella_first_step_time (x : ℝ) (h : time_first_step x) : x = 30 :=
by
  sorry

end arabella_first_step_time_l441_441493


namespace forming_quadrilateral_with_perimeters_l441_441931

noncomputable def right_triangle : Type :=
{ side1 : ℝ // side1 = 3 } × { side2 : ℝ // side2 = 4 } × { hypotenuse : ℝ // hypotenuse = 5 }

noncomputable def form_quadrilateral (triangles: list right_triangle) (perimeter: ℝ) : Prop :=
∃ (qs : list (list right_triangle)), 
  (∀ q ∈ qs, (q.foldr (λ t acc, t.1.1 + t.1.2 + t.2.1 + t.1.2 + t.1.1 + t.2.1 + acc) 0) = perimeter)
  ∧ (∀ q1 q2 ∈ qs, q1 ≠ q2)

theorem forming_quadrilateral_with_perimeters :
  ∃ qs14 qs18 qs22 qs26 : list (list right_triangle),
  (∃ t : list right_triangle, form_quadrilateral t 14) ∧
  (∃ t : list right_triangle, form_quadrilateral t 18) ∧
  (∃ t : list right_triangle, form_quadrilateral t 22) ∧
  (∃ t : list right_triangle, form_quadrilateral t 26) ∧ 
  (∀ q1 q2 ∈ qs14, q1 ≠ q2) ∧ 
  (∀ q1 q2 ∈ qs18, q1 ≠ q2) ∧ 
  (∀ q1 q2 ∈ qs22, q1 ≠ q2) ∧ 
  (∀ q1 q2 ∈ qs26, q1 ≠ q2) :=
by
  sorry

end forming_quadrilateral_with_perimeters_l441_441931


namespace isosceles_triangle_of_parallel_l441_441883

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441883


namespace triangle_is_isosceles_l441_441813

variables {Point : Type*} {Line : Type*} {Triangle : Type*} [Geometry Point Line Triangle]

def is_parallel (l1 l2 : Line) : Prop := parallel l1 l2
def is_isosceles (T : Triangle) : Prop := is_isosceles_triangle T

variables {A B C P Q : Point} {AC PQ : Line} {T : Triangle}

axiom PQ_parallel_AC : is_parallel PQ AC
axiom triangle_ABC : T = ⟨A, B, C⟩ -- Assuming a custom type for Triangle taking vertices A, B, C

theorem triangle_is_isosceles (PQ_parallel_AC : is_parallel PQ AC)
  (triangle_ABC : T = ⟨A, B, C⟩) : is_isosceles T := by
  sorry

end triangle_is_isosceles_l441_441813


namespace number_of_female_workers_l441_441634

theorem number_of_female_workers (M F : ℕ) (M_no F_no : ℝ) 
  (hM : M = 112)
  (h1 : M_no = 0.40 * M)
  (h2 : F_no = 0.25 * F)
  (h3 : M_no / (M_no + F_no) = 0.30)
  (h4 : F_no / (M_no + F_no) = 0.70)
  : F = 420 := 
by 
  sorry

end number_of_female_workers_l441_441634


namespace roja_work_rate_l441_441445

theorem roja_work_rate (W : ℝ) : 
  (1 / 60 + 1 / x = 1 / 35) → x = 84 :=
by
  intros h,
  sorry

end roja_work_rate_l441_441445


namespace isosceles_triangle_l441_441754

theorem isosceles_triangle (A B C P Q : Type) [LinearOrder P] [LinearOrder Q] 
  (h_parallel : PQ ∥ AC) : IsoscelesTriangle ABC :=
sorry

end isosceles_triangle_l441_441754


namespace find_a_and_b_l441_441497

noncomputable def a : ℝ := 3
noncomputable def b : ℝ := 3

theorem find_a_and_b :
  (∀ x : ℝ, (a * cos (b * x)) ≤ 3)
  ∧ (a * (cos 0) = 3)
  ∧ (a * cos (b * (π / 6)) = 0) :=
by 
  split; {
    assumption -- Placeholder to ensure the theorem complies with conditions
  }
  sorry

end find_a_and_b_l441_441497


namespace triangle_ABC_isosceles_l441_441842

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441842


namespace triangle_ABC_isosceles_l441_441843

theorem triangle_ABC_isosceles (A B C P Q : Type)
  [euclidean_space A] [euclidean_space B] [euclidean_space C]
  [parallel P Q] [parallel A C] :
  is_isosceles_triangle A B C :=
sorry

end triangle_ABC_isosceles_l441_441843


namespace isosceles_triangle_of_parallel_l441_441877

theorem isosceles_triangle_of_parallel (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] (h_parallel : PQ.parallel AC) : 
  is_isosceles_triangle ABC :=
sorry

end isosceles_triangle_of_parallel_l441_441877


namespace period_of_P_heartbeats_per_minute_blood_pressure_reading_l441_441964

-- Define the blood pressure function P(t)
def P(t : ℝ) := 115 + 25 * Real.sin (160 * Real.pi * t)

-- Problem (1): Prove the period of the function P(t)
theorem period_of_P : (∀ t : ℝ, P(t + 1/80) = P(t)) := by
  sorry

-- Problem (2): Prove the number of heartbeats per minute
theorem heartbeats_per_minute : (1 / (1 / 80) = 80) := by
  sorry

-- Problem (3): Prove the max and min values of P(t) and the blood pressure reading
theorem blood_pressure_reading :
  ∃ t : ℝ, P(t) = 140 ∧ ∃ t : ℝ, P(t) = 90 ∧ 140 / 90 = 140 / 90 := by
  sorry

end period_of_P_heartbeats_per_minute_blood_pressure_reading_l441_441964


namespace bruce_total_payment_l441_441017

-- Define the conditions
def quantity_grapes : Nat := 7
def rate_grapes : Nat := 70
def quantity_mangoes : Nat := 9
def rate_mangoes : Nat := 55

-- Define the calculation for total amount paid
def total_amount_paid : Nat :=
  (quantity_grapes * rate_grapes) + (quantity_mangoes * rate_mangoes)

-- Proof statement
theorem bruce_total_payment : total_amount_paid = 985 :=
by
  -- Proof steps would go here
  sorry

end bruce_total_payment_l441_441017


namespace cost_of_larger_cylindrical_can_l441_441046

noncomputable def volume (r h : ℝ) : ℝ := π * r^2 * h

noncomputable def price_per_volume (price volume : ℝ) : ℝ := price / volume

noncomputable def cost_of_larger_can 
  (diameter_small height_small diameter_large height_large price_small : ℝ) : ℝ :=
  let r_small := diameter_small / 2
  let r_large := diameter_large / 2
  let volume_small := volume r_small height_small
  let volume_large := volume r_large height_large
  let rate := price_per_volume price_small volume_small
  rate * volume_large

theorem cost_of_larger_cylindrical_can
  (diameter_small height_small price_small diameter_large height_large : ℝ)
  (h1 : diameter_small = 4) (h2 : height_small = 5) (h3 : price_small = 0.80)
  (h4 : diameter_large = 8) (h5 : height_large = 10) :
  cost_of_larger_can diameter_small height_small diameter_large height_large price_small = 6.40 :=
by
  simp [h1, h2, h3, h4, h5, cost_of_larger_can, volume, price_per_volume]
  sorry

end cost_of_larger_cylindrical_can_l441_441046


namespace fraction_never_simplifiable_l441_441717

theorem fraction_never_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_never_simplifiable_l441_441717


namespace mutually_exclusive_not_opposite_l441_441146

namespace PencilCase

def pencil_case := { pens : ℕ, pencils : ℕ }

structure Event :=
(events : List String)

def mutually_exclusive (E1 E2 : Event) : Prop :=
∀ (e1 ∈ E1.events) (e2 ∈ E2.events), e1 ≠ e2

def opposite (E1 E2 : Event) : Prop :=
∀ (e1 ∈ E1.events) (e2 ∈ E2.events), ¬(e1 = e2)

def exactly_one_pen : Event := { events := ["exactly_one_pen"] }
def exactly_two_pencils : Event := { events := ["exactly_two_pencils"] }

theorem mutually_exclusive_not_opposite 
(h_case : pencil_case) (h_condition : h_case = { pens := 2, pencils := 2 }) :
mutually_exclusive exactly_one_pen exactly_two_pencils ∧ ¬ opposite exactly_one_pen exactly_two_pencils :=
by
  sorry

end PencilCase

end mutually_exclusive_not_opposite_l441_441146


namespace value_of_x_l441_441374

theorem value_of_x (x y z : ℝ) (h1 : x = (1 / 2) * y) (h2 : y = (1 / 4) * z) (h3 : z = 80) : x = 10 := by
  sorry

end value_of_x_l441_441374
